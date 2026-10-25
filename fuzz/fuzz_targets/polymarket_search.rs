//! Polymarket search responses: whatever parses must stay within the shown
//! market cap, keep finite prices, render, and survive the cache round trip.

#![no_main]

use bot_core::locale::Locale;
use bot_core::polymarket::{
    SEARCH_MARKETS_PER_EVENT, SearchEvent, parse_search_events, render_market_search,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|payload: &str| {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
        return;
    };
    let events = parse_search_events(&value);
    for event in &events {
        assert!(event.markets.len() <= SEARCH_MARKETS_PER_EVENT);
        assert!(event.liquidity.is_finite() && event.volume.is_finite());
        for outcome in event.markets.iter().flat_map(|market| &market.outcomes) {
            assert!((0.0..=1.0).contains(&outcome.probability));
        }
    }
    let _ = render_market_search("query", &events, Locale::Es);
    let encoded = serde_json::to_string(&events).expect("parsed events encode");
    let decoded =
        serde_json::from_str::<Vec<SearchEvent>>(&encoded).expect("encoded events decode");
    assert_eq!(decoded, events);
});
