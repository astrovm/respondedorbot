use reqwest::Client;

pub async fn get_with_ssl_fallback(url: &str) -> Result<reqwest::Response, reqwest::Error> {
    let client = Client::builder().build()?;
    match client.get(url).send().await {
        Ok(resp) => Ok(resp),
        Err(err) => {
            let insecure = Client::builder().danger_accept_invalid_certs(true).build()?;
            insecure.get(url).send().await.or(Err(err))
        }
    }
}
