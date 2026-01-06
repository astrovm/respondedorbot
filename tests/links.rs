use wasm_bindgen_test::wasm_bindgen_test;
use respondedorbot::links::{is_social_frontend, strip_tracking};

#[wasm_bindgen_test]
fn test_is_social_frontend() {
    assert!(is_social_frontend("twitter.com"));
    assert!(is_social_frontend("sub.reddit.com"));
    assert!(!is_social_frontend("example.com"));
}

#[wasm_bindgen_test]
fn test_strip_tracking_social() {
    let url = "https://twitter.com/user/status/1?utm=1#frag";
    let stripped = strip_tracking(url);
    assert_eq!(stripped, "https://twitter.com/user/status/1");
}
