//! Periodic refresh of the persistent market caches.

use std::fmt::Debug;

use bot_adapters::coinmarketcap::{
    CoinMarketCapMarketTransport, ReqwestCoinMarketCapTransport, refresh_market_snapshot,
};
use bot_adapters::dollar::{
    DollarCache, DollarTransport, ReqwestDollarTransport, refresh_dollar_snapshot,
};
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_json_cache::RedisJsonCache;
use bot_adapters::request_cache::RequestCache;
use bot_adapters::yahoo_finance::{
    ReqwestYahooFinanceTransport, YahooFinanceTransport, YahooQuoteLoad, load_quote,
};

use crate::background::BackgroundWorker;

const FAILURE_REPORT_THRESHOLD: usize = 3;

trait PriceRefreshJob: Send {
    fn refresh(&mut self, now_epoch_seconds: i64) -> Result<(), String>;
}

/// Refreshes the persistent CriptoYa dollar snapshot.
struct DollarJob<Transport, Cache> {
    transport: Transport,
    cache: Cache,
}

impl<Transport, Cache> PriceRefreshJob for DollarJob<Transport, Cache>
where
    Transport: DollarTransport + Send,
    Cache: DollarCache + Send,
{
    fn refresh(&mut self, now_epoch_seconds: i64) -> Result<(), String> {
        diagnostics(
            "dollar refresh",
            refresh_dollar_snapshot(&self.transport, &mut self.cache, now_epoch_seconds),
        )
    }
}

/// Refreshes the CoinMarketCap listing snapshot for one quote currency.
struct CryptoJob<Transport, Cache> {
    transport: Transport,
    cache: Cache,
    api_key: String,
    currency: &'static str,
}

impl<Transport, Cache> PriceRefreshJob for CryptoJob<Transport, Cache>
where
    Transport: CoinMarketCapMarketTransport + Send,
    Cache: RequestCache + Send,
{
    fn refresh(&mut self, now_epoch_seconds: i64) -> Result<(), String> {
        diagnostics(
            "CoinMarketCap refresh",
            refresh_market_snapshot(
                &self.transport,
                &mut self.cache,
                &self.api_key,
                self.currency,
                now_epoch_seconds,
            ),
        )
    }
}

/// Refreshes the Brent and WTI benchmark quotes.
struct OilJob<Transport, Cache> {
    transport: Transport,
    cache: Cache,
}

impl<Transport, Cache> PriceRefreshJob for OilJob<Transport, Cache>
where
    Transport: YahooFinanceTransport + Send,
    Cache: RequestCache + Send,
{
    fn refresh(&mut self, now_epoch_seconds: i64) -> Result<(), String> {
        let (transport, cache) = (&self.transport, &mut self.cache);
        refresh_oil(|symbol| load_quote(transport, cache, symbol, now_epoch_seconds))
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
    let cache = || RedisJsonCache::new(redis_endpoint).map_err(crate::error_text);
    let mut jobs = vec![NamedJob {
        name: "dollar",
        job: Box::new(DollarJob {
            transport: ReqwestDollarTransport::new().map_err(construction_error("dollar"))?,
            cache: cache()?,
        }),
    }];
    if let Some(api_key) = coinmarketcap_key.filter(|value| !value.is_empty()) {
        for (name, currency) in [("crypto-ars", "ARS"), ("crypto-usd", "USD")] {
            jobs.push(NamedJob {
                name,
                job: Box::new(CryptoJob {
                    transport: ReqwestCoinMarketCapTransport::new()
                        .map_err(construction_error("CoinMarketCap"))?,
                    cache: cache()?,
                    api_key: api_key.to_owned(),
                    currency,
                }),
            });
        }
    }
    jobs.push(NamedJob {
        name: "oil",
        job: Box::new(OilJob {
            transport: ReqwestYahooFinanceTransport::new()
                .map_err(construction_error("Yahoo Finance"))?,
            cache: cache()?,
        }),
    });
    Ok(PriceCacheRefreshWorker::new(jobs))
}

fn construction_error<E: Debug>(provider: &'static str) -> impl FnOnce(E) -> String {
    move |error| format!("could not construct {provider} transport: {error:?}")
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
        CryptoJob, DollarJob, NamedJob, OilJob, PriceCacheRefreshWorker, PriceRefreshJob,
        construction_error, production_price_refresh_worker, refresh_oil,
    };

    struct ClosureJob<F>(F);

    impl<F> PriceRefreshJob for ClosureJob<F>
    where
        F: FnMut(i64) -> Result<(), String> + Send,
    {
        fn refresh(&mut self, now_epoch_seconds: i64) -> Result<(), String> {
            (self.0)(now_epoch_seconds)
        }
    }
    use crate::background::BackgroundWorker;
    use bot_adapters::redis_connection::RedisEndpoint;
    use bot_adapters::redis_json_cache::RedisJsonCache;

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
                    calls.lock().map_err(crate::error_text)?.push((name, now));
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
    fn production_worker_composes_optional_market_jobs_without_provider_io()
    -> crate::test_env::TestResult {
        let endpoint = RedisEndpoint {
            host: "synthetic.invalid".to_owned(),
            port: 6379,
            password: Some("synthetic-password".to_owned()),
        };
        let without_crypto = production_price_refresh_worker(&endpoint, None)?;
        assert_eq!(without_crypto.jobs.len(), 2);
        assert_eq!(without_crypto.jobs[0].name, "dollar");
        assert_eq!(without_crypto.jobs[1].name, "oil");

        let with_crypto = production_price_refresh_worker(&endpoint, Some("synthetic-market-key"))?;
        assert_eq!(with_crypto.jobs.len(), 4);
        assert_eq!(
            with_crypto
                .jobs
                .iter()
                .map(|job| job.name)
                .collect::<Vec<_>>(),
            ["dollar", "crypto-ars", "crypto-usd", "oil"]
        );
        let empty_key = production_price_refresh_worker(&endpoint, Some(""))?;
        assert_eq!(empty_key.jobs.len(), 2);
        Ok(())
    }

    #[test]
    fn a_successful_cycle_resets_the_consecutive_failure_budget() {
        let outcomes = Arc::new(Mutex::new(vec![true, true, false, true, true, true]));
        let script = Arc::clone(&outcomes);
        let mut worker = PriceCacheRefreshWorker::new(vec![NamedJob {
            name: "flaky",
            job: Box::new(ClosureJob(move |_now| {
                let fails = script.lock().map_err(crate::error_text)?.remove(0);
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

    /// Rejects every provider request, as when the network is down.
    struct OfflineTransport;

    impl bot_adapters::dollar::DollarTransport for OfflineTransport {
        fn get(
            &self,
        ) -> Result<bot_adapters::dollar::HttpResponse, bot_adapters::dollar::TransportFailureKind>
        {
            Err(bot_adapters::dollar::TransportFailureKind::Connection)
        }
    }

    impl bot_adapters::coinmarketcap::CoinMarketCapMarketTransport for OfflineTransport {
        fn get_market(
            &self,
            _request: &bot_adapters::coinmarketcap::MarketRequest,
        ) -> Result<
            bot_adapters::coinmarketcap::HttpResponse,
            bot_adapters::coinmarketcap::TransportFailureKind,
        > {
            Err(bot_adapters::coinmarketcap::TransportFailureKind::Connection)
        }
    }

    impl bot_adapters::yahoo_finance::YahooFinanceTransport for OfflineTransport {
        fn chart(
            &self,
            _request: &bot_adapters::yahoo_finance::YahooChartRequest,
        ) -> Result<
            bot_adapters::yahoo_finance::HttpResponse,
            bot_adapters::yahoo_finance::TransportFailureKind,
        > {
            Err(bot_adapters::yahoo_finance::TransportFailureKind::Connection)
        }

        fn search(
            &self,
            _request: &bot_adapters::yahoo_finance::YahooSearchRequest,
        ) -> Result<
            bot_adapters::yahoo_finance::HttpResponse,
            bot_adapters::yahoo_finance::TransportFailureKind,
        > {
            Err(bot_adapters::yahoo_finance::TransportFailureKind::Connection)
        }
    }

    fn unreachable_cache() -> Result<RedisJsonCache, Box<dyn std::error::Error>> {
        Ok(RedisJsonCache::new(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port: 1,
            password: None,
        })?)
    }

    #[test]
    fn each_refresh_job_labels_provider_and_cache_failures() -> crate::test_env::TestResult {
        let mut dollar = DollarJob {
            transport: OfflineTransport,
            cache: unreachable_cache()?,
        };
        let mut crypto = CryptoJob {
            transport: OfflineTransport,
            cache: unreachable_cache()?,
            api_key: "synthetic-market-key".to_owned(),
            currency: "USD",
        };
        let mut oil = OilJob {
            transport: OfflineTransport,
            cache: unreachable_cache()?,
        };
        let failures = [
            (dollar.refresh(1_700_000_000), "dollar refresh: "),
            (crypto.refresh(1_700_000_000), "CoinMarketCap refresh: "),
            (oil.refresh(1_700_000_000), "Yahoo oil refresh: "),
        ];
        for (result, label) in failures {
            let error = result.err().unwrap_or_default();
            assert!(error.starts_with(label), "{error}");
            assert!(error.contains("Connection"), "{error}");
        }
        let search = bot_adapters::yahoo_finance::YahooFinanceTransport::search(
            &OfflineTransport,
            &bot_adapters::yahoo_finance::YahooSearchRequest {
                query: "BZ=F".to_owned(),
            },
        );
        assert_eq!(
            search,
            Err(bot_adapters::yahoo_finance::TransportFailureKind::Connection)
        );
        Ok(())
    }

    #[test]
    fn construction_failures_name_the_provider() {
        assert_eq!(
            construction_error("dollar")("synthetic TLS failure"),
            "could not construct dollar transport: \"synthetic TLS failure\""
        );
    }
}
