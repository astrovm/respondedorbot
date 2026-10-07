use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};

use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_poll_store::RedisPollStore;
use bot_core::polls::{MAX_POLLS_PER_CHAT, PollAnswer, PollRecord, PollState, PollVote, poll_key};

fn endpoint() -> Option<RedisEndpoint> {
    let port = std::env::var("TEST_REDIS_PORT").ok()?.parse().ok()?;
    Some(RedisEndpoint {
        host: std::env::var("TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned()),
        port,
        password: std::env::var("TEST_REDIS_PASSWORD")
            .ok()
            .filter(|value| !value.is_empty()),
    })
}

fn record(poll_id: &str, chat_id: i64, created_at: i64) -> PollRecord {
    PollRecord {
        poll_id: poll_id.to_owned(),
        chat_id,
        message_id: 3,
        question: "¿Asado?".to_owned(),
        options: vec!["Sí".to_owned(), "No".to_owned()],
        anonymous: false,
        multiple_answers: false,
        created_at,
        closed: false,
        counts: None,
        total_voters: None,
    }
}

fn answer(poll_id: &str, voter_id: &str, name: &str, option_ids: Vec<usize>) -> PollAnswer {
    PollAnswer {
        poll_id: poll_id.to_owned(),
        voter_id: voter_id.to_owned(),
        vote: PollVote {
            name: name.to_owned(),
            option_ids,
        },
    }
}

#[test]
fn polls_votes_retractions_and_state_round_trip() -> Result<(), Box<dyn Error>> {
    let Some(endpoint) = endpoint() else {
        return Ok(());
    };
    let store = RedisPollStore::new(&endpoint)?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let chat_id = -i64::try_from(nonce % 1_000_000_000_000)? - 1;
    let poll_id = format!("poll-{nonce}");

    assert!(!store.record_answer(&answer(&poll_id, "1", "Early", vec![0]))?);
    assert!(!store.apply_state(PollState {
        poll_id: poll_id.clone(),
        options: Vec::new(),
        counts: Vec::new(),
        total_voters: 0,
        closed: true,
    })?);
    assert!(store.recent_polls(chat_id)?.is_empty());

    store.save_poll(&record(&poll_id, chat_id, 100))?;
    assert!(store.record_answer(&answer(&poll_id, "2", "Beto", vec![1]))?);
    assert!(store.record_answer(&answer(&poll_id, "1", "Ana", vec![0]))?);
    assert!(store.record_answer(&answer(&poll_id, "3", "Caro", vec![0]))?);
    assert!(store.record_answer(&answer(&poll_id, "3", "Caro", Vec::new()))?);
    let polls = store.recent_polls(chat_id)?;
    assert_eq!(polls.len(), 1);
    assert_eq!(
        polls[0].1,
        vec![
            PollVote {
                name: "Ana".to_owned(),
                option_ids: vec![0],
            },
            PollVote {
                name: "Beto".to_owned(),
                option_ids: vec![1],
            },
        ]
    );

    assert!(store.apply_state(PollState {
        poll_id: poll_id.clone(),
        options: vec!["Sí".to_owned(), "No".to_owned(), "Tal vez".to_owned()],
        counts: vec![1, 1, 0],
        total_voters: 2,
        closed: true,
    })?);
    let polls = store.recent_polls(chat_id)?;
    assert!(polls[0].0.closed);
    assert_eq!(polls[0].0.counts, Some(vec![1, 1, 0]));
    assert_eq!(polls[0].0.options.len(), 3);

    for index in 0..=MAX_POLLS_PER_CHAT {
        store.save_poll(&record(
            &format!("{poll_id}-{index}"),
            chat_id,
            200 + i64::try_from(index)?,
        ))?;
    }
    let polls = store.recent_polls(chat_id)?;
    assert_eq!(polls.len(), MAX_POLLS_PER_CHAT);
    assert_eq!(
        polls[0].0.poll_id,
        format!("{poll_id}-{MAX_POLLS_PER_CHAT}")
    );
    assert!(polls.iter().all(|(poll, _)| poll.poll_id != poll_id));
    Ok(())
}

#[test]
fn malformed_stored_polls_and_votes_are_skipped() -> Result<(), Box<dyn Error>> {
    let Some(endpoint) = endpoint() else {
        return Ok(());
    };
    let store = RedisPollStore::new(&endpoint)?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let chat_id = -i64::try_from(nonce % 1_000_000_000_000)? - 2;
    let broken = format!("broken-{nonce}");
    let good = format!("good-{nonce}");
    store.save_poll(&record(&broken, chat_id, 1))?;
    store.save_poll(&record(&good, chat_id, 2))?;
    let client = redis::Client::open(format!("redis://{}:{}/", endpoint.host, endpoint.port))?;
    let mut connection = client.get_connection()?;
    redis::cmd("SET")
        .arg(poll_key(&broken))
        .arg("not json")
        .query::<()>(&mut connection)?;
    redis::cmd("HSET")
        .arg(format!("poll_votes:{good}"))
        .arg("9")
        .arg("not json")
        .query::<()>(&mut connection)?;
    let polls = store.recent_polls(chat_id)?;
    assert_eq!(polls.len(), 1);
    assert_eq!(polls[0].0.poll_id, good);
    assert!(polls[0].1.is_empty());
    assert!(!store.apply_state(PollState {
        poll_id: broken,
        options: Vec::new(),
        counts: Vec::new(),
        total_voters: 0,
        closed: false,
    })?);
    Ok(())
}
