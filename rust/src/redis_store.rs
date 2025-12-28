use redis::AsyncCommands;

pub fn create_redis_client(
    host: &str,
    port: u16,
    password: Option<&str>,
) -> redis::RedisResult<redis::Client> {
    let url = if let Some(pw) = password {
        format!("redis://:{}@{}:{}/", pw, host, port)
    } else {
        format!("redis://{}:{}/", host, port)
    };
    redis::Client::open(url)
}

pub async fn redis_get_string(
    client: &mut redis::aio::MultiplexedConnection,
    key: &str,
) -> redis::RedisResult<Option<String>> {
    let raw: Option<String> = client.get(key).await?;
    Ok(raw)
}

pub async fn redis_set_string(
    client: &mut redis::aio::MultiplexedConnection,
    key: &str,
    value: &str,
) -> redis::RedisResult<bool> {
    let result: bool = client.set(key, value).await?;
    Ok(result)
}

pub async fn redis_setex_string(
    client: &mut redis::aio::MultiplexedConnection,
    key: &str,
    ttl_seconds: u64,
    value: &str,
) -> redis::RedisResult<bool> {
    let result: bool = client.set_ex(key, value, ttl_seconds).await?;
    Ok(result)
}

pub async fn redis_incr_with_ttl(
    client: &mut redis::aio::MultiplexedConnection,
    key: &str,
    ttl_seconds: u64,
) -> redis::RedisResult<i64> {
    let count: i64 = client.incr(key, 1).await?;
    if count == 1 {
        let _: () = client.expire(key, ttl_seconds as i64).await?;
    }
    Ok(count)
}
