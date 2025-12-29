use crate::http::HttpClient;
use crate::http_cache::cached_get_json;
use crate::storage::Storage;

const TTL_WEATHER: u64 = 1800;

pub async fn get_weather_context(http: &HttpClient, storage: &Storage) -> Option<String> {
    let params = [
        ("latitude", "-34.5429".to_string()),
        ("longitude", "-58.7119".to_string()),
        (
            "hourly",
            "apparent_temperature,precipitation_probability,weather_code,cloud_cover,visibility"
                .to_string(),
        ),
        ("timezone", "auto".to_string()),
        ("forecast_days", "2".to_string()),
    ];

    let data = cached_get_json(
        http,
        storage,
        "https://api.open-meteo.com/v1/forecast",
        Some(&params),
        None,
        TTL_WEATHER,
    )
    .await?;

    let hourly = data.get("hourly")?;
    let times = hourly.get("time")?.as_array()?;
    let codes = hourly.get("weather_code")?.as_array()?;
    let temps = hourly.get("apparent_temperature")?.as_array()?;
    let precip = hourly.get("precipitation_probability")?.as_array()?;

    let now = chrono::Local::now();
    let mut idx = None;
    let now_key = now.format("%Y-%m-%dT%H").to_string();
    for (i, t) in times.iter().enumerate() {
        if let Some(ts) = t.as_str() {
            if ts.starts_with(&now_key) {
                idx = Some(i);
                break;
            }
        }
    }

    let i = idx.unwrap_or(0);
    let code = codes.get(i)?.as_i64().unwrap_or(0) as i32;
    let temp = temps.get(i)?.as_f64().unwrap_or(0.0);
    let precip = precip.get(i)?.as_f64().unwrap_or(0.0);

    let desc = weather_description(code);
    Some(format!(
        "Ahora: {desc}. Sensación térmica {temp:.1}°C, lluvia {precip:.0}%",
    ))
}

fn weather_description(code: i32) -> &'static str {
    match code {
        0 => "despejado",
        1 => "mayormente despejado",
        2 => "parcialmente nublado",
        3 => "nublado",
        45 => "neblina",
        48 => "niebla",
        51 => "llovizna leve",
        53 => "llovizna moderada",
        55 => "llovizna intensa",
        61 => "lluvia leve",
        63 => "lluvia moderada",
        65 => "lluvia intensa",
        80 => "lluvia leve intermitente",
        81 => "lluvia moderada intermitente",
        82 => "lluvia fuerte intermitente",
        95 => "tormenta",
        _ => "clima raro",
    }
}
