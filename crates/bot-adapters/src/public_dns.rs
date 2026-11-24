//! A DNS resolver for reqwest that refuses a host when any of its addresses is
//! not public. The fetch code checks each URL before it asks for it, but
//! reqwest looks the host up again when it connects, and the second answer
//! can be different. With this resolver the connection uses only addresses
//! that passed the check.

use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll, Waker};

use reqwest::dns::{Addrs, Name, Resolve, Resolving};

use crate::web_fetch::{HostResolver, SystemHostResolver, is_public_ip};

type Answer = Result<Vec<IpAddr>, String>;
type Resolved = Result<Addrs, Box<dyn std::error::Error + Send + Sync>>;

pub(crate) struct PublicOnlyDns;

impl Resolve for PublicOnlyDns {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_owned();
        // reqwest puts the URL's port on the addresses.
        Box::pin(Lookup::start(move || {
            SystemHostResolver.addresses(&host, 0)
        }))
    }
}

#[derive(Default)]
struct Slot {
    answer: Option<Answer>,
    waker: Option<Waker>,
}

/// A lookup on its own thread, so a slow DNS server does not stall the other
/// requests on the client's runtime.
struct Lookup(Arc<Mutex<Slot>>);

impl Lookup {
    fn start(lookup: impl FnOnce() -> Answer + Send + 'static) -> Self {
        let slot = Arc::new(Mutex::new(Slot::default()));
        let shared = Arc::clone(&slot);
        std::thread::spawn(move || {
            let answer = lookup();
            let mut slot = shared.lock().unwrap_or_else(PoisonError::into_inner);
            slot.answer = Some(answer);
            slot.waker.take().into_iter().for_each(Waker::wake);
        });
        Self(slot)
    }
}

impl Future for Lookup {
    type Output = Resolved;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Resolved> {
        let mut slot = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        slot.waker = Some(context.waker().clone());
        slot.answer
            .take()
            .map_or(Poll::Pending, |answer| Poll::Ready(public_only(answer)))
    }
}

fn public_only(answer: Answer) -> Resolved {
    let addresses = answer?;
    if addresses.is_empty() || !addresses.iter().all(|address| is_public_ip(*address)) {
        return Err("the host has an address that is not public".into());
    }
    Ok(Box::new(
        addresses
            .into_iter()
            .map(|address| SocketAddr::new(address, 0)),
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::task::Wake;
    use std::thread::Thread;

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    struct Unpark(Thread);

    impl Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }

    /// Polls `future` once, runs `between`, then waits for it on this thread.
    fn wait_for(future: impl Future<Output = Resolved>, between: impl FnOnce()) -> Answer {
        let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
        let mut context = Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        let mut first = future.as_mut().poll(&mut context);
        between();
        loop {
            if let Poll::Ready(resolved) = first {
                return resolved
                    .map(|addresses| addresses.map(|address| address.ip()).collect())
                    .map_err(|error| error.to_string());
            }
            std::thread::park();
            first = future.as_mut().poll(&mut context);
        }
    }

    #[test]
    fn a_lookup_wakes_its_request_when_the_answer_arrives() {
        let (release, released) = mpsc::channel::<()>();
        let lookup = Lookup::start(move || {
            let _ = released.recv();
            Ok(vec![IpAddr::from([93, 184, 215, 14])])
        });
        let answer = wait_for(lookup, || {
            let _ = release.send(());
        });
        assert_eq!(answer, Ok(vec![IpAddr::from([93, 184, 215, 14])]));
    }

    #[test]
    fn only_hosts_with_all_public_addresses_resolve() -> TestResult {
        let resolve = |host: &str| -> Result<Answer, Box<dyn std::error::Error>> {
            Ok(wait_for(PublicOnlyDns.resolve(host.parse()?), || {}))
        };
        assert_eq!(resolve("1.1.1.1")?, Ok(vec![IpAddr::from([1, 1, 1, 1])]));
        assert_eq!(
            resolve("localhost")?,
            Err("the host has an address that is not public".to_owned())
        );
        assert!(
            wait_for(Lookup::start(|| Err("no such host".to_owned())), || {})
                .is_err_and(|error| error == "no such host")
        );
        assert!(wait_for(Lookup::start(|| Ok(Vec::new())), || {}).is_err());
        assert!(
            wait_for(
                Lookup::start(|| Ok(vec![
                    IpAddr::from([1, 1, 1, 1]),
                    IpAddr::from([10, 0, 0, 1])
                ])),
                || {}
            )
            .is_err()
        );
        Ok(())
    }
}
