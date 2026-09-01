//! Provider retry-window parsing and rate-limit header precedence.

const RETRY_HEADERS: usize = 4;

#[must_use]
pub fn parse_retry_window_seconds(value: Option<&str>, now_unix_seconds: f64) -> Option<i64> {
    let raw = value.unwrap_or_default().trim();
    if raw.is_empty() || !now_unix_seconds.is_finite() {
        return None;
    }
    if let Ok(number) = raw.parse::<f64>()
        && number.is_finite()
        && number >= i64::MIN as f64
        && number <= i64::MAX as f64
    {
        return Some((number.trunc() as i64).max(0));
    }
    if let Some(seconds) = parse_duration(raw) {
        return Some(seconds.max(0));
    }
    let retry_at = parse_http_date(raw)?;
    let remaining = retry_at as f64 - now_unix_seconds;
    if remaining <= 0.0 {
        Some(0)
    } else if remaining <= i64::MAX as f64 {
        Some(remaining.trunc() as i64)
    } else {
        None
    }
}

#[must_use]
pub fn select_rate_limit_backoff_seconds(
    header_values: [Option<&str>; RETRY_HEADERS],
    fallback_seconds: Option<i64>,
    now_unix_seconds: f64,
) -> Option<i64> {
    header_values
        .into_iter()
        .find_map(|value| parse_retry_window_seconds(value, now_unix_seconds))
        .or(fallback_seconds)
}

fn parse_duration(raw: &str) -> Option<i64> {
    let lower = raw.to_ascii_lowercase();
    let (amount, multiplier) = if let Some(value) = lower.strip_suffix("ms") {
        (value, 0.001)
    } else if let Some(value) = lower.strip_suffix('s') {
        (value, 1.0)
    } else if let Some(value) = lower.strip_suffix('m') {
        (value, 60.0)
    } else {
        (lower.strip_suffix('h')?, 3_600.0)
    };
    if !is_unsigned_decimal(amount) {
        return None;
    }
    let value = amount.parse::<f64>().ok()?.checked_mul(multiplier)?;
    if value.is_finite() && value <= i64::MAX as f64 {
        Some(value.trunc() as i64)
    } else {
        None
    }
}

trait CheckedFloatMultiply {
    fn checked_mul(self, other: Self) -> Option<Self>
    where
        Self: Sized;
}

impl CheckedFloatMultiply for f64 {
    fn checked_mul(self, other: Self) -> Option<Self> {
        let result = self * other;
        result.is_finite().then_some(result)
    }
}

fn is_unsigned_decimal(value: &str) -> bool {
    let mut digits = 0_usize;
    let mut decimal_points = 0_usize;
    for character in value.bytes() {
        if character.is_ascii_digit() {
            digits += 1;
        } else if character == b'.' {
            decimal_points += 1;
        } else {
            return false;
        }
    }
    digits > 0 && decimal_points <= 1 && !value.starts_with('.') && !value.ends_with('.')
}

fn parse_http_date(raw: &str) -> Option<i64> {
    let fields: Vec<&str> = raw.split_whitespace().collect();
    if fields.len() == 6 && fields[0].ends_with(',') {
        return date_time_to_unix(
            fields[3].parse().ok()?,
            month(fields[2])?,
            fields[1].parse().ok()?,
            fields[4],
            fields[5],
        );
    }
    if fields.len() == 4 && fields[0].ends_with(',') {
        let date_parts: Vec<&str> = fields[1].split('-').collect();
        if date_parts.len() != 3 {
            return None;
        }
        let short_year: i32 = date_parts[2].parse().ok()?;
        let year = if short_year <= 68 {
            2_000 + short_year
        } else {
            1_900 + short_year
        };
        return date_time_to_unix(
            year,
            month(date_parts[1])?,
            date_parts[0].parse().ok()?,
            fields[2],
            fields[3],
        );
    }
    if fields.len() == 5 && !fields[0].ends_with(',') {
        return date_time_to_unix(
            fields[4].parse().ok()?,
            month(fields[1])?,
            fields[2].parse().ok()?,
            fields[3],
            "GMT",
        );
    }
    None
}

fn month(value: &str) -> Option<u32> {
    match value.to_ascii_lowercase().as_str() {
        "jan" => Some(1),
        "feb" => Some(2),
        "mar" => Some(3),
        "apr" => Some(4),
        "may" => Some(5),
        "jun" => Some(6),
        "jul" => Some(7),
        "aug" => Some(8),
        "sep" => Some(9),
        "oct" => Some(10),
        "nov" => Some(11),
        "dec" => Some(12),
        _ => None,
    }
}

fn date_time_to_unix(year: i32, month: u32, day: u32, time: &str, timezone: &str) -> Option<i64> {
    if !(1..=12).contains(&month) || day == 0 || day > days_in_month(year, month) {
        return None;
    }
    let mut parts = time.split(':');
    let hour: i64 = parts.next()?.parse().ok()?;
    let minute: i64 = parts.next()?.parse().ok()?;
    let second: i64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let offset = timezone_offset_seconds(timezone)?;
    days_from_civil(year, month, day)
        .checked_mul(86_400)?
        .checked_add(hour.checked_mul(3_600)?)?
        .checked_add(minute.checked_mul(60)?)?
        .checked_add(second)?
        .checked_sub(offset)
}

fn timezone_offset_seconds(value: &str) -> Option<i64> {
    if value.eq_ignore_ascii_case("gmt") || value.eq_ignore_ascii_case("utc") {
        return Some(0);
    }
    let (sign, value) = value
        .strip_prefix('+')
        .map(|value| (1_i64, value))
        .or_else(|| value.strip_prefix('-').map(|value| (-1_i64, value)))?;
    if value.len() != 4 || !value.bytes().all(|character| character.is_ascii_digit()) {
        return None;
    }
    let hours: i64 = value[..2].parse().ok()?;
    let minutes: i64 = value[2..].parse().ok()?;
    if hours > 23 || minutes > 59 {
        return None;
    }
    sign.checked_mul(
        hours
            .checked_mul(3_600)?
            .checked_add(minutes.checked_mul(60)?)?,
    )
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 31,
    }
}

fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let adjusted_year = i64::from(year) - i64::from(month <= 2);
    let era = adjusted_year.div_euclid(400);
    let year_of_era = adjusted_year - era * 400;
    let month = i64::from(month);
    let shifted_month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
mod tests {
    use super::{parse_retry_window_seconds, select_rate_limit_backoff_seconds};

    #[test]
    fn parses_numeric_units_and_all_http_date_forms() {
        let now = 784_111_777.75;
        for (raw, expected) in [
            ("", None),
            ("-1", Some(0)),
            ("1.9", Some(1)),
            ("999ms", Some(0)),
            ("1.5s", Some(1)),
            ("2m", Some(120)),
            ("1h", Some(3_600)),
            ("Sun, 06 Nov 1994 08:49:39 GMT", Some(1)),
            ("Sunday, 06-Nov-94 08:49:39 GMT", Some(1)),
            ("Sun Nov  6 08:49:39 1994", Some(1)),
            ("Sun, 06 Nov 1994 09:49:39 +0100", Some(1)),
            ("not a retry window", None),
        ] {
            assert_eq!(
                parse_retry_window_seconds(Some(raw), now),
                expected,
                "{raw}"
            );
        }
    }

    #[test]
    fn selects_the_first_parseable_header_then_fallback() {
        assert_eq!(
            select_rate_limit_backoff_seconds(
                [Some("invalid"), Some("2m"), Some("3m"), None],
                Some(300),
                0.0,
            ),
            Some(120)
        );
        assert_eq!(
            select_rate_limit_backoff_seconds([Some("invalid"), None, None, None], Some(300), 0.0,),
            Some(300)
        );
        assert_eq!(
            select_rate_limit_backoff_seconds([Some("-1"), Some("2m"), None, None], None, 0.0,),
            Some(0)
        );
        assert_eq!(
            select_rate_limit_backoff_seconds([None, None, None, None], None, 0.0),
            None
        );
    }

    #[test]
    fn validates_calendar_timezones_and_malformed_windows() {
        for (raw, now, expected) in [
            ("Thu, 01 Jan 1970 00:00:00 UTC", 0.0, Some(0)),
            ("Thu, 01 Jan 1970 01:00:00 +0100", 0.0, Some(0)),
            ("Thu, 01 Jan 1970 00:00:00 -0100", 0.0, Some(3_600)),
            ("Tue, 29 Feb 2000 00:00:00 GMT", 951_782_400.0, Some(0)),
            ("Thu, 29 Feb 1900 00:00:00 GMT", 0.0, None),
            ("Thu, 01 Foo 1970 00:00:00 GMT", 0.0, None),
            ("Thu, 01 Jan 1970 25:00:00 GMT", 0.0, None),
            ("Thu, 01 Jan 1970 00:00:00 +9999", 0.0, None),
            (".5m", 0.0, None),
            ("1.m", 0.0, None),
            ("1d", 0.0, None),
        ] {
            assert_eq!(
                parse_retry_window_seconds(Some(raw), now),
                expected,
                "{raw}"
            );
        }
        assert_eq!(parse_retry_window_seconds(Some("1m"), f64::NAN), None);
    }
}
