//! Scheduled-task records stored in Redis: anything that decodes must encode
//! again and decode back to the same task. Encoding drops legacy fields the
//! schedule doesn't use, so the encoded form, not the input, is the fixed point.

#![no_main]

use bot_adapters::task_record::{decode_task_record, encode_task_record};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|payload: &str| {
    if let Ok(record) = decode_task_record(payload) {
        let encoded = encode_task_record(&record).expect("a decoded record encodes");
        let decoded = decode_task_record(&encoded).expect("an encoded record decodes");
        assert_eq!(decoded.task, record.task);
        let reencoded = encode_task_record(&decoded).expect("a decoded record encodes");
        assert_eq!(reencoded, encoded);
    }
});
