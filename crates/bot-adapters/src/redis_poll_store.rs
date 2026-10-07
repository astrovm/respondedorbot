//! Redis storage for polls the bot sent and the votes Telegram reports.

use std::collections::HashMap;

use thiserror::Error;

use bot_core::polls::{
    MAX_POLLS_PER_CHAT, POLL_TTL_SECONDS, PollAnswer, PollRecord, PollState, PollVote,
    chat_polls_key, poll_key, poll_votes_key,
};

use crate::redis_connection::{RedisEndpoint, RedisPool, pool};

/// Keeps the poll, adds it to the chat index, and trims the index to the
/// newest polls.
const SAVE_POLL_SCRIPT: &str = "redis.call('SETEX', KEYS[1], ARGV[1], ARGV[2]); redis.call('ZADD', KEYS[2], ARGV[3], ARGV[4]); redis.call('ZREMRANGEBYRANK', KEYS[2], 0, -tonumber(ARGV[5]) - 1); redis.call('EXPIRE', KEYS[2], ARGV[1]); return 1";
/// Votes for polls the bot never stored (sent before this feature, or
/// expired) are dropped. An empty vote is a retraction.
const RECORD_VOTE_SCRIPT: &str = "if redis.call('EXISTS', KEYS[1]) == 0 then return 0 end; if ARGV[2] == '' then redis.call('HDEL', KEYS[2], ARGV[1]) else redis.call('HSET', KEYS[2], ARGV[1], ARGV[2]) end; redis.call('EXPIRE', KEYS[2], ARGV[3]); return 1";

#[derive(Debug, Error)]
pub enum RedisPollStoreError {
    #[error("Redis poll-store operation failed: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("poll record could not be encoded: {0}")]
    Encode(#[from] serde_json::Error),
}

#[derive(Clone)]
pub struct RedisPollStore {
    client: RedisPool,
}

impl RedisPollStore {
    pub fn new(endpoint: &RedisEndpoint) -> Result<Self, RedisPollStoreError> {
        Ok(Self {
            client: pool(endpoint)?,
        })
    }

    pub fn save_poll(&self, record: &PollRecord) -> Result<(), RedisPollStoreError> {
        let encoded = serde_json::to_string(record)?;
        let mut connection = self.client.get_connection()?;
        redis::cmd("EVAL")
            .arg(SAVE_POLL_SCRIPT)
            .arg(2)
            .arg(poll_key(&record.poll_id))
            .arg(chat_polls_key(record.chat_id))
            .arg(POLL_TTL_SECONDS)
            .arg(encoded)
            .arg(record.created_at)
            .arg(&record.poll_id)
            .arg(MAX_POLLS_PER_CHAT)
            .query::<i64>(&mut connection)?;
        Ok(())
    }

    /// Returns whether the vote belonged to a stored poll.
    pub fn record_answer(&self, answer: &PollAnswer) -> Result<bool, RedisPollStoreError> {
        let vote = if answer.vote.option_ids.is_empty() {
            String::new()
        } else {
            serde_json::to_string(&answer.vote)?
        };
        let mut connection = self.client.get_connection()?;
        let stored: i64 = redis::cmd("EVAL")
            .arg(RECORD_VOTE_SCRIPT)
            .arg(2)
            .arg(poll_key(&answer.poll_id))
            .arg(poll_votes_key(&answer.poll_id))
            .arg(&answer.voter_id)
            .arg(vote)
            .arg(POLL_TTL_SECONDS)
            .query(&mut connection)?;
        Ok(stored == 1)
    }

    /// Returns whether the update belonged to a stored poll. Each update
    /// carries the full state, so the latest one wins.
    pub fn apply_state(&self, state: PollState) -> Result<bool, RedisPollStoreError> {
        let key = poll_key(&state.poll_id);
        let mut connection = self.client.get_connection()?;
        let stored: Option<String> = redis::cmd("GET").arg(&key).query(&mut connection)?;
        let Some(mut record) =
            stored.and_then(|stored| serde_json::from_str::<PollRecord>(&stored).ok())
        else {
            return Ok(false);
        };
        record.apply_state(state);
        redis::cmd("SET")
            .arg(&key)
            .arg(serde_json::to_string(&record)?)
            .arg("KEEPTTL")
            .query::<()>(&mut connection)?;
        Ok(true)
    }

    /// The chat's newest polls first, each with its stored votes.
    pub fn recent_polls(
        &self,
        chat_id: i64,
    ) -> Result<Vec<(PollRecord, Vec<PollVote>)>, RedisPollStoreError> {
        let mut connection = self.client.get_connection()?;
        let poll_ids: Vec<String> = redis::cmd("ZREVRANGE")
            .arg(chat_polls_key(chat_id))
            .arg(0)
            .arg(MAX_POLLS_PER_CHAT - 1)
            .query(&mut connection)?;
        let mut polls = Vec::with_capacity(poll_ids.len());
        for poll_id in poll_ids {
            let stored: Option<String> = redis::cmd("GET")
                .arg(poll_key(&poll_id))
                .query(&mut connection)?;
            let Some(record) =
                stored.and_then(|stored| serde_json::from_str::<PollRecord>(&stored).ok())
            else {
                continue;
            };
            let votes: HashMap<String, String> = redis::cmd("HGETALL")
                .arg(poll_votes_key(&poll_id))
                .query(&mut connection)?;
            let mut votes = votes
                .into_iter()
                .filter_map(|(voter, vote)| {
                    serde_json::from_str::<PollVote>(&vote)
                        .ok()
                        .map(|vote| (voter, vote))
                })
                .collect::<Vec<_>>();
            votes.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            polls.push((record, votes.into_iter().map(|(_, vote)| vote).collect()));
        }
        Ok(polls)
    }
}
