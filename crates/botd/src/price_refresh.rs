//! Periodic refresh of the persistent market caches.

use bot_adapters::coinmarketcap::{ReqwestCoinMarketCapTransport, refresh_market_snapshot};
use bot_adapters::dollar::{ReqwestDollarTransport, refresh_dollar_snapshot};
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_json_cache::RedisJsonCache;
use bot_adapters::yahoo_finance::{ReqwestYahooFinanceTransport, YahooQuoteLoad, load_quote};

use crate::background::BackgroundWorker;

const FAILURE_REPORT_THRESHOLD: usize = 3;

trait PriceRefreshJob: Send {
    fn refresh(&mut self, now_epoch_seconds: i64) -> Result<(), String>;
}

struct ClosureJob<F>(F);

impl<F> PriceRefreshJob for ClosureJob<F>
where
    F: FnMut(i64) -> Result<(), String> + Send,
{
    fn refresh(&mut self, now_epoch_seconds: i64) -> Result<(), String> {
        (self.0)(now_epoch_seconds)
    }
}

struct NamedJob {
    name: &'static str,
    job: Box<dyn PriceRefreshJob>,
}

/// Runs every refresh even when an earlier provider or cache fails.
pub struct PriceCacheRefreshWorker {
    jobs: Vec<NamedJob>,
    consecutive_failed_cycles: usize,
}

impl PriceCacheRefreshWorker {
    fn new(jobs: Vec<NamedJob>) -> Self {
        Self {
            jobs,
            consecutive_failed_cycles: 0,
        }
    }
}

impl BackgroundWorker for PriceCacheRefreshWorker {
    fn run_once(&mut self, now_epoch_seconds: i64) -> Result<(), String> {
        let failures = self
            .jobs
            .iter_mut()
            .filter_map(|job| {
                job.job
                    .refresh(now_epoch_seconds)
                    .err()
                    .map(|error| format!("{}: {error}", job.name))
            })
            .collect::<Vec<_>>();
        if failures.is_empty() {
            self.consecutive_failed_cycles = 0;
            Ok(())
        } else {
            self.consecutive_failed_cycles = self.consecutive_failed_cycles.saturating_add(1);
            let message = failures.join("; ");
            if self.consecutive_failed_cycles >= FAILURE_REPORT_THRESHOLD {
                Err(format!(
                    "{} consecutive refresh cycles failed: {message}",
                    self.consecutive_failed_cycles
                ))
            } else {
                eprintln!(
                    "transient price-cache refresh failure ({}/{}): {message}",
                    self.consecutive_failed_cycles, FAILURE_REPORT_THRESHOLD
                );
                Ok(())
            }
        }
    }
}

fn diagnostics(label: &str, diagnostics: Vec<String>) -> Result<(), String> {
    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(format!("{label}: {}", diagnostics.join("; ")))
    }
}

pub fn production_price_refresh_worker(
    redis_endpoint: &RedisEndpoint,
    coinmarketcap_key: Option<&str>,
) -> Result<PriceCacheRefreshWorker, String> {
    let dollar_transport = ReqwestDollarTransport::new()
        .map_err(|error| format!("could not construct dollar transport: {error:?}"))?;
    let mut dollar_cache =
        RedisJsonCache::new(redis_endpoint).map_err(|error| error.to_string())?;
    let mut jobs = vec![NamedJob {
        name: "dollar",
        job: Box::new(ClosureJob(move |now| {
            diagnostics(
                "dollar refresh",
                refresh_dollar_snapshot(&dollar_transport, &mut dollar_cache, now),
            )
        })),
    }];

    if let Some(api_key) = coinmarketcap_key.filter(|value| !value.is_empty()) {
        for currency in ["ARS", "USD"] {
            let transport = coinmarketcap_transport()?;
            let mut cache =
                RedisJsonCache::new(redis_endpoint).map_err(|error| error.to_string())?;
            let api_key = api_key.to_owned();
            jobs.push(NamedJob {
                name: if currency == "ARS" {
                    "crypto-ars"
                } else {
                    "crypto-usd"
                },
                job: Box::new(ClosureJob(move |now| {
                    diagnostics(
                        "CoinMarketCap refresh",
                        refresh_market_snapshot(&transport, &mut cache, &api_key, currency, now),
                    )
                })),
            });
        }
    }

    let oil_transport = ReqwestYahooFinanceTransport::new()
        .map_err(|error| format!("could not construct Yahoo Finance transport: {error:?}"))?;
    let mut oil_cache = RedisJsonCache::new(redis_endpoint).map_err(|error| error.to_string())?;
    jobs.push(NamedJob {
        name: "oil",
        job: Box::new(ClosureJob(move |now| {
            refresh_oil(|symbol| load_quote(&oil_transport, &mut oil_cache, symbol, now))
        })),
    });
    Ok(PriceCacheRefreshWorker::new(jobs))
}

fn coinmarketcap_transport() -> Result<ReqwestCoinMarketCapTransport, String> {
    ReqwestCoinMarketCapTransport::new()
        .map_err(|error| format!("could not construct CoinMarketCap transport: {error:?}"))
}

fn refresh_oil(mut load_quote: impl FnMut(&str) -> YahooQuoteLoad) -> Result<(), String> {
    let mut failures = Vec::new();
    for symbol in ["BZ=F", "CL=F"] {
        let load = load_quote(symbol);
        if load.quote.is_none() {
            failures.extend(load.diagnostics);
        }
    }
    diagnostics("Yahoo oil refresh", failures)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use bot_adapters::yahoo_finance::YahooQuoteLoad;
    use bot_core::stocks::StockQuote;

    use super::{
        ClosureJob, NamedJob, PriceCacheRefreshWorker, production_price_refresh_worker, refresh_oil,
    };
    use crate::background::BackgroundWorker;
    use bot_adapters::redis_connection::RedisEndpoint;

    #[test]
    fn runs_all_jobs_and_reports_each_failure() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let jobs = [
            ("first", false),
            ("second", true),
            ("third", true),
            ("fourth", false),
        ]
        .into_iter()
        .map(|(name, fails)| {
            let calls = calls.clone();
            NamedJob {
                name,
                job: Box::new(ClosureJob(move |now| {
                    calls
                        .lock()
                        .map_err(|_| "call log lock was poisoned".to_owned())?
                        .push((name, now));
                    if fails {
                        Err("synthetic failure".to_owned())
                    } else {
                        Ok(())
                    }
                })),
            }
        })
        .collect();
        let mut worker = PriceCacheRefreshWorker::new(jobs);
        assert!(worker.run_once(121).is_ok());
        assert!(worker.run_once(122).is_ok());
        let result = worker.run_once(123);
        let recorded = calls.lock().map(|calls| calls.clone()).unwrap_or_default();
        assert_eq!(recorded.len(), 12);
        assert_eq!(
            &recorded[8..],
            [
                ("first", 123),
                ("second", 123),
                ("third", 123),
                ("fourth", 123),
            ]
        );
        let error = result.err().unwrap_or_default();
        assert!(error.contains("second: synthetic failure"));
        assert!(error.contains("third: synthetic failure"));
    }

    #[test]
    fn production_worker_composes_optional_market_jobs_without_provider_io() {
        let endpoint = RedisEndpoint {
            host: "synthetic.invalid".to_owned(),
            port: 6379,
            password: Some("synthetic-password".to_owned()),
        };
        let without_crypto =
            production_price_refresh_worker(&endpoint, None).unwrap_or_else(|_| unreachable!());
        assert_eq!(without_crypto.jobs.len(), 2);
        assert_eq!(without_crypto.jobs[0].name, "dollar");
        assert_eq!(without_crypto.jobs[1].name, "oil");

        let with_crypto = production_price_refresh_worker(&endpoint, Some("synthetic-market-key"))
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(with_crypto.jobs.len(), 4);
        assert_eq!(
            with_crypto
                .jobs
                .iter()
                .map(|job| job.name)
                .collect::<Vec<_>>(),
            ["dollar", "crypto-ars", "crypto-usd", "oil"]
        );
    }

    #[test]
    fn a_successful_cycle_resets_the_consecutive_failure_budget() {
        let outcomes = Arc::new(Mutex::new(vec![true, true, false, true, true, true]));
        let script = Arc::clone(&outcomes);
        let mut worker = PriceCacheRefreshWorker::new(vec![NamedJob {
            name: "flaky",
            job: Box::new(ClosureJob(move |_now| {
                let fails = script
                    .lock()
                    .map_err(|_| "script lock was poisoned".to_owned())?
                    .remove(0);
                if fails {
                    Err("synthetic outage".to_owned())
                } else {
                    Ok(())
                }
            })),
        }]);
        assert_eq!(worker.run_once(1), Ok(()));
        assert_eq!(worker.run_once(2), Ok(()));
        assert_eq!(worker.consecutive_failed_cycles, 2);
        assert_eq!(worker.run_once(3), Ok(()));
        assert_eq!(worker.consecutive_failed_cycles, 0);
        assert_eq!(worker.run_once(4), Ok(()));
        assert_eq!(worker.run_once(5), Ok(()));
        assert_eq!(
            worker.run_once(6),
            Err("3 consecutive refresh cycles failed: flaky: synthetic outage".to_owned())
        );
        assert!(
            outcomes
                .lock()
                .map(|script| script.is_empty())
                .unwrap_or(false)
        );
    }

    #[test]
    fn oil_refresh_loads_both_benchmarks_and_reports_only_missing_quotes() {
        let mut requested = Vec::new();
        let result = refresh_oil(|symbol| {
            requested.push(symbol.to_owned());
            YahooQuoteLoad {
                candles: Vec::new(),
                quote: (symbol == "BZ=F").then(|| StockQuote {
                    symbol: symbol.to_owned(),
                    name: "Brent".to_owned(),
                    price: 98.15,
                    currency: "USD".to_owned(),
                    exchange: "NYM".to_owned(),
                    asset_type: String::new(),
                    variation: -8.8,
                }),
                diagnostics: vec![format!("synthetic diagnostic for {symbol}")],
            }
        });
        assert_eq!(requested, ["BZ=F", "CL=F"]);
        assert_eq!(
            result,
            Err("Yahoo oil refresh: synthetic diagnostic for CL=F".to_owned())
        );

        let healthy = refresh_oil(|symbol| YahooQuoteLoad {
            candles: Vec::new(),
            quote: Some(StockQuote {
                symbol: symbol.to_owned(),
                name: String::new(),
                price: 1.0,
                currency: "USD".to_owned(),
                exchange: String::new(),
                asset_type: String::new(),
                variation: 0.0,
            }),
            diagnostics: vec!["ignored when the quote is usable".to_owned()],
        });
        assert_eq!(healthy, Ok(()));
    }
}
