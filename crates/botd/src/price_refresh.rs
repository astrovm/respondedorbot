//! Periodic refresh of the market caches shared with the legacy runtime.

use bot_adapters::coinmarketcap::{ReqwestCoinMarketCapTransport, refresh_market_snapshot};
use bot_adapters::dollar::{ReqwestDollarTransport, refresh_dollar_snapshot};
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_json_cache::RedisJsonCache;
use bot_adapters::yahoo_finance::{ReqwestYahooFinanceTransport, load_quote};

use crate::background::BackgroundWorker;

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
}

impl PriceCacheRefreshWorker {
    fn new(jobs: Vec<NamedJob>) -> Self {
        Self { jobs }
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
            Ok(())
        } else {
            Err(failures.join("; "))
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
            let transport = ReqwestCoinMarketCapTransport::new().map_err(|error| {
                format!("could not construct CoinMarketCap transport: {error:?}")
            })?;
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
            let mut failures = Vec::new();
            for symbol in ["BZ=F", "CL=F"] {
                let load = load_quote(&oil_transport, &mut oil_cache, symbol, now);
                if load.quote.is_none() {
                    failures.extend(load.diagnostics);
                }
            }
            diagnostics("Yahoo oil refresh", failures)
        })),
    });
    Ok(PriceCacheRefreshWorker::new(jobs))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{ClosureJob, NamedJob, PriceCacheRefreshWorker};
    use crate::background::BackgroundWorker;

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
        let result = worker.run_once(123);
        let recorded = calls.lock().map(|calls| calls.clone()).unwrap_or_default();
        assert_eq!(
            recorded,
            vec![
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
}
