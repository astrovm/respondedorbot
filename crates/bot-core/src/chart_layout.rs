//! Pure layout math for price charts: candle merging, round price ticks and
//! local-time axis labels. Candles are `[time, open, high, low, close, volume?]`.

use chrono::{DateTime, Datelike, FixedOffset, Timelike};

/// Merge consecutive candles so at most `max` remain. Groups are counted from
/// the newest candle, so only the oldest group can be partial.
#[must_use]
pub fn merge_candles(candles: Vec<Vec<f64>>, max: usize) -> Vec<Vec<f64>> {
    let max = max.max(1);
    if candles.len() <= max {
        return candles;
    }
    let size = candles.len().div_ceil(max);
    let mut end = match candles.len() % size {
        0 => size,
        partial => partial,
    };
    let mut start = 0;
    let mut merged = Vec::with_capacity(max);
    while start < candles.len() {
        merged.extend(merge_group(&candles[start..end]));
        start = end;
        end = (end + size).min(candles.len());
    }
    merged
}

fn merge_group(group: &[Vec<f64>]) -> Option<Vec<f64>> {
    let (first, last) = (group.first()?, group.last()?);
    let value = |candle: &Vec<f64>, index: usize| candle.get(index).copied();
    let high = group
        .iter()
        .filter_map(|candle| value(candle, 2))
        .fold(f64::NEG_INFINITY, f64::max);
    let low = group
        .iter()
        .filter_map(|candle| value(candle, 3))
        .fold(f64::INFINITY, f64::min);
    let volume = group
        .iter()
        .map(|candle| value(candle, 5).unwrap_or(0.0))
        .sum::<f64>();
    Some(vec![
        value(first, 0)?,
        value(first, 1)?,
        high,
        low,
        value(last, 4)?,
        volume,
    ])
}

/// Round price levels between `minimum` and `maximum`, about `target` of them.
#[derive(Debug, Clone, PartialEq)]
pub struct PriceTicks {
    pub values: Vec<f64>,
    /// Decimals that show every tick exactly, such as 1 for a 0.5 step.
    pub decimals: usize,
}

#[must_use]
pub fn price_ticks(minimum: f64, maximum: f64, target: usize) -> PriceTicks {
    let span = maximum - minimum;
    if !span.is_finite() || span <= 0.0 || target == 0 {
        return PriceTicks {
            values: Vec::new(),
            decimals: 0,
        };
    }
    let magnitude = 10_f64.powf((span / target as f64).log10().floor());
    // The round step whose tick count lands closest to the target.
    let count = |step: f64| (maximum / step).floor() - (minimum / step).ceil() + 1.0;
    let step = [1.0, 2.0, 2.5, 5.0, 10.0]
        .into_iter()
        .map(|nice| nice * magnitude)
        .min_by(|left, right| {
            (count(*left) - target as f64)
                .abs()
                .total_cmp(&(count(*right) - target as f64).abs())
        })
        .unwrap_or(magnitude);
    let decimals = (0..=16)
        .find(|decimals| {
            let scaled = step * 10_f64.powi(*decimals);
            (scaled - scaled.round()).abs() < 1e-6 * scaled.max(1.0)
        })
        .unwrap_or(16) as usize;
    let first = (minimum / step).ceil() as i64;
    let last = (maximum / step).floor() as i64;
    PriceTicks {
        values: (first..=last).map(|index| index as f64 * step).collect(),
        decimals,
    }
}

/// One mark on the time axis, placed before the candle at `index`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeTick {
    pub index: usize,
    /// Shown text. Unlabeled ticks are day boundaries kept only as lines.
    pub label: Option<String>,
    /// The candle at `index` starts a new local day.
    pub day_start: bool,
}

/// Formats a tick time; the flag says whether it starts a new local day.
type LabelFormat = fn(&DateTime<FixedOffset>, bool) -> String;

const HOUR: i64 = 3_600;
const DAY: i64 = 86_400;

/// Time-axis marks in the chat's UTC offset: hours for charts up to two days,
/// days up to four months, then months. At most `max_labels` carry a label.
#[must_use]
pub fn time_ticks(times: &[f64], utc_offset_hours: i64, max_labels: usize) -> Vec<TimeTick> {
    let Some(offset) = FixedOffset::east_opt((utc_offset_hours * HOUR) as i32) else {
        return Vec::new();
    };
    let local = times
        .iter()
        .map(|time| {
            time.is_finite()
                .then(|| DateTime::from_timestamp(*time as i64, 0))
                .flatten()
                .map(|time| time.with_timezone(&offset))
        })
        .collect::<Option<Vec<_>>>();
    let Some(local) = local.filter(|local| local.len() >= 2) else {
        return Vec::new();
    };
    let span = times[times.len() - 1] - times[0];
    let max_labels = max_labels.max(1);
    let day_key = |time: &DateTime<FixedOffset>| (time.year(), time.ordinal());
    let month_key = |time: &DateTime<FixedOffset>| (time.year(), time.month());
    let boundaries = |key: &dyn Fn(&DateTime<FixedOffset>) -> i64| {
        (1..local.len())
            .filter(|index| key(&local[*index]) != key(&local[*index - 1]))
            .collect::<Vec<_>>()
    };
    let day_starts = (1..local.len())
        .filter(|index| day_key(&local[*index]) != day_key(&local[*index - 1]))
        .collect::<Vec<_>>();
    let (labeled, format): (Vec<usize>, LabelFormat) = if span <= (2 * DAY) as f64 {
        // The smallest hour step that fits, counted in local hours.
        let labeled = [1, 2, 3, 4, 6, 12, 24]
            .into_iter()
            .map(|hours| {
                boundaries(&|time: &DateTime<FixedOffset>| {
                    (time.timestamp() + time.offset().local_minus_utc() as i64)
                        .div_euclid(hours * HOUR)
                })
            })
            .find(|ticks| ticks.len() < max_labels)
            .unwrap_or_else(|| day_starts.clone());
        (labeled, |time, day_start| {
            if day_start {
                format!("{:02}/{:02}", time.day(), time.month())
            } else {
                // Merged candles start a few minutes past the hour.
                format!("{:02}:00", time.hour())
            }
        })
    } else if span <= (120 * DAY) as f64 {
        (thin(&day_starts, max_labels - 1), |time, _| {
            format!("{:02}/{:02}", time.day(), time.month())
        })
    } else {
        let months = (1..local.len())
            .filter(|index| month_key(&local[*index]) != month_key(&local[*index - 1]))
            .collect::<Vec<_>>();
        (thin(&months, max_labels - 1), |time, _| {
            format!("{:02}/{}", time.month(), time.year())
        })
    };
    // The first candle carries its date, so every chart says which day it shows.
    let mut ticks = vec![TimeTick {
        index: 0,
        label: Some(format!("{:02}/{:02}", local[0].day(), local[0].month())),
        day_start: false,
    }];
    let mut indexes = labeled.clone();
    indexes.extend(day_starts.iter().copied());
    indexes.sort_unstable();
    indexes.dedup();
    ticks.extend(indexes.into_iter().map(|index| {
        let day_start = day_starts.binary_search(&index).is_ok();
        TimeTick {
            index,
            label: labeled
                .binary_search(&index)
                .is_ok()
                .then(|| format(&local[index], day_start)),
            day_start,
        }
    }));
    ticks
}

/// Keep every n-th mark so at most `max` remain.
fn thin(indexes: &[usize], max: usize) -> Vec<usize> {
    let every = indexes.len().div_ceil(max.max(1)).max(1);
    indexes.iter().copied().step_by(every).collect()
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{TimeTick, merge_candles, price_ticks, time_ticks};

    fn candle(time: f64, open: f64, high: f64, low: f64, close: f64) -> Vec<f64> {
        vec![time, open, high, low, close, 1.0]
    }

    #[test]
    fn merging_keeps_ohlc_and_ends_on_the_newest_candle() {
        let candles = vec![
            candle(1.0, 10.0, 11.0, 9.0, 10.5),
            candle(2.0, 10.5, 13.0, 10.0, 12.0),
            candle(3.0, 12.0, 12.5, 8.0, 9.0),
            candle(4.0, 9.0, 9.5, 8.5, 9.2),
            candle(5.0, 9.2, 10.0, 9.1, 9.9),
        ];
        assert_eq!(merge_candles(candles.clone(), 5), candles);
        assert_eq!(
            merge_candles(candles, 2),
            vec![
                vec![1.0, 10.0, 13.0, 9.0, 12.0, 2.0],
                vec![3.0, 12.0, 12.5, 8.0, 9.9, 3.0],
            ]
        );
    }

    proptest! {
        #[test]
        fn merging_never_exceeds_the_limit_or_loses_extremes(
            closes in prop::collection::vec(1.0_f64..1_000.0, 1..2_000),
            max in 1_usize..300,
        ) {
            let candles = closes
                .iter()
                .enumerate()
                .map(|(index, close)| candle(index as f64, *close, close * 1.01, close * 0.99, *close))
                .collect::<Vec<_>>();
            let merged = merge_candles(candles.clone(), max);
            prop_assert!(merged.len() <= max);
            prop_assert_eq!(merged.last().map(|c| c[4]), candles.last().map(|c| c[4]));
            prop_assert_eq!(merged.first().map(|c| c[1]), candles.first().map(|c| c[1]));
            let high = |rows: &[Vec<f64>]| rows.iter().map(|c| c[2]).fold(f64::MIN, f64::max);
            prop_assert_eq!(high(&merged), high(&candles));
            prop_assert_eq!(merged.iter().map(|c| c[5]).sum::<f64>(), candles.len() as f64);
        }

        #[test]
        fn price_ticks_are_round_and_inside_the_range(
            minimum in 1e-6_f64..1e6,
            ratio in 1.0001_f64..3.0,
        ) {
            let maximum = minimum * ratio;
            let ticks = price_ticks(minimum, maximum, 5);
            prop_assert!((2..=11).contains(&ticks.values.len()));
            for value in &ticks.values {
                prop_assert!(*value >= minimum && *value <= maximum);
                let shown: f64 = format!("{value:.*}", ticks.decimals).parse().unwrap_or(f64::NAN);
                prop_assert!((shown - value).abs() <= value.abs() * 1e-9);
            }
        }
    }

    #[test]
    fn price_ticks_pick_round_steps() {
        let ticks = price_ticks(177.67, 232.652, 5);
        assert_eq!(ticks.values, vec![180.0, 190.0, 200.0, 210.0, 220.0, 230.0]);
        assert_eq!(ticks.decimals, 0);
        let small = price_ticks(0.000_123, 0.000_131, 5);
        assert_eq!(small.decimals, 6);
        assert_eq!(price_ticks(5.0, 5.0, 5).values, Vec::<f64>::new());
    }

    // 2026-11-06 00:00:00 UTC, a Friday.
    const FRIDAY: f64 = 1_793_923_200.0;

    #[test]
    fn short_charts_label_local_hours_and_mark_the_new_day() {
        // Hourly candles from 20:00 to 08:00 UTC, shown in UTC-3.
        let times = (20..33)
            .map(|hour| FRIDAY + f64::from(hour) * 3_600.0)
            .collect::<Vec<_>>();
        let ticks = time_ticks(&times, -3, 7);
        assert_eq!(
            ticks.first(),
            Some(&TimeTick {
                index: 0,
                label: Some("06/11".into()),
                day_start: false
            })
        );
        // Local midnight is 03:00 UTC, the 8th candle.
        assert!(ticks.iter().any(|tick| tick.index == 7
            && tick.day_start
            && tick.label.as_deref() == Some("07/11")));
        let labels = ticks.iter().filter(|tick| tick.label.is_some()).count();
        assert!((3..=7).contains(&labels));
        assert!(
            ticks
                .iter()
                .any(|tick| tick.label.as_deref() == Some("20:00"))
        );
    }

    #[test]
    fn long_charts_label_days_then_months_and_keep_every_day_line() {
        let days = (0..30)
            .map(|day| FRIDAY + f64::from(day) * 86_400.0 + 50_000.0)
            .collect::<Vec<_>>();
        let ticks = time_ticks(&days, 0, 7);
        assert_eq!(ticks.iter().filter(|tick| tick.day_start).count(), 29);
        assert!(ticks.iter().filter(|tick| tick.label.is_some()).count() <= 7);
        let months = (0..400)
            .map(|day| FRIDAY + f64::from(day) * 86_400.0)
            .collect::<Vec<_>>();
        let ticks = time_ticks(&months, 0, 7);
        assert!(
            ticks
                .iter()
                .any(|tick| tick.label.as_deref() == Some("01/2027")
                    || tick.label.as_deref() == Some("12/2026"))
        );
        assert!(ticks.iter().filter(|tick| tick.label.is_some()).count() <= 7);
    }

    #[test]
    fn time_ticks_need_two_valid_candles_and_a_real_offset() {
        assert!(time_ticks(&[FRIDAY], 0, 7).is_empty());
        assert!(time_ticks(&[FRIDAY, FRIDAY + 60.0], 40, 7).is_empty());
        assert!(time_ticks(&[f64::NAN, FRIDAY], 0, 7).is_empty());
    }
}
