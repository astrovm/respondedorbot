//! Scheduled-task list rendering and deletion authorization.

use chrono::{FixedOffset, TimeZone, Utc, Weekday};

use crate::locale::Locale;
use crate::scheduled_tasks::{ScheduledTask, TaskId, TaskSchedule};
use crate::telegram_actions::{InlineKeyboardButton, InlineKeyboardMarkup};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaskListView {
    pub text: String,
    pub keyboard: Option<InlineKeyboardMarkup>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TaskCallbackParse {
    Delete(TaskId),
    Guard,
}

#[must_use]
pub fn parse_task_callback(data: &str) -> TaskCallbackParse {
    let mut parts = data.splitn(3, ':');
    let (Some(prefix), Some(_action), Some(task_id)) = (parts.next(), parts.next(), parts.next())
    else {
        return TaskCallbackParse::Guard;
    };
    if prefix != "task" {
        return TaskCallbackParse::Guard;
    }
    TaskId::new(task_id).map_or(TaskCallbackParse::Guard, TaskCallbackParse::Delete)
}

#[must_use]
pub fn can_delete_task(
    is_group: bool,
    request_user_id: Option<i64>,
    owner_user_id: Option<i64>,
    is_group_admin: bool,
) -> bool {
    if !is_group {
        return true;
    }
    request_user_id.is_some() && request_user_id == owner_user_id || is_group_admin
}

fn interval_text(seconds: i64, locale: Locale) -> String {
    let (value, singular_es, plural_es, singular_en, plural_en) = if seconds >= 86_400 {
        (seconds / 86_400, "dia", "dias", "day", "days")
    } else if seconds >= 3_600 {
        (seconds / 3_600, "hora", "horas", "hour", "hours")
    } else {
        (seconds / 60, "minuto", "minutos", "minute", "minutes")
    };
    let unit = match (locale, value == 1) {
        (Locale::Es, true) => singular_es,
        (Locale::Es, false) => plural_es,
        (Locale::En, true) => singular_en,
        (Locale::En, false) => plural_en,
    };
    match locale {
        Locale::Es => format!("cada {value} {unit}"),
        Locale::En => format!("every {value} {unit}"),
    }
}

fn weekday_text(value: Weekday, locale: Locale) -> &'static str {
    match (value, locale) {
        (Weekday::Mon, Locale::Es) => "lun",
        (Weekday::Tue, Locale::Es) => "mar",
        (Weekday::Wed, Locale::Es) => "mie",
        (Weekday::Thu, Locale::Es) => "jue",
        (Weekday::Fri, Locale::Es) => "vie",
        (Weekday::Sat, Locale::Es) => "sab",
        (Weekday::Sun, Locale::Es) => "dom",
        (Weekday::Mon, Locale::En) => "mon",
        (Weekday::Tue, Locale::En) => "tue",
        (Weekday::Wed, Locale::En) => "wed",
        (Weekday::Thu, Locale::En) => "thu",
        (Weekday::Fri, Locale::En) => "fri",
        (Weekday::Sat, Locale::En) => "sat",
        (Weekday::Sun, Locale::En) => "sun",
    }
}

fn frequency(schedule: &TaskSchedule, locale: Locale) -> Option<String> {
    match schedule {
        TaskSchedule::Once => None,
        TaskSchedule::IntervalSeconds { seconds } => Some(interval_text(*seconds, locale)),
        TaskSchedule::IntervalDays { days } => Some(match locale {
            Locale::Es => format!("cada {days} dias"),
            Locale::En => format!("every {days} days"),
        }),
        TaskSchedule::Cron {
            hour,
            minute,
            weekdays,
            day,
        } => {
            let time = format!("{hour:02}:{minute:02}");
            if !weekdays.is_empty() {
                let weekdays = weekdays
                    .iter()
                    .map(|value| weekday_text(*value, locale))
                    .collect::<Vec<_>>()
                    .join(", ");
                Some(match locale {
                    Locale::Es => format!("los {weekdays} a las {time}"),
                    Locale::En => format!("on {weekdays} at {time}"),
                })
            } else if let Some(day) = day {
                Some(match locale {
                    Locale::Es => format!("el dia {day} de cada mes a las {time}"),
                    Locale::En => format!("on day {day} of every month at {time}"),
                })
            } else {
                Some(match locale {
                    Locale::Es => format!("todos los dias a las {time}"),
                    Locale::En => format!("every day at {time}"),
                })
            }
        }
    }
}

fn local_time(timestamp: Option<i64>, timezone_offset: i32) -> String {
    let Some(timestamp) = timestamp else {
        return "unknown".to_owned();
    };
    let Some(offset) = timezone_offset
        .checked_mul(3_600)
        .and_then(FixedOffset::east_opt)
    else {
        return "unknown".to_owned();
    };
    Utc.timestamp_opt(timestamp, 0).single().map_or_else(
        || "unknown".to_owned(),
        |value| {
            value
                .with_timezone(&offset)
                .format("%d/%m %H:%M")
                .to_string()
        },
    )
}

fn no_mention(value: &str) -> String {
    value.replace('@', "@\u{200b}")
}

#[must_use]
pub fn format_task_summary(task: &ScheduledTask, locale: Locale) -> String {
    let text = no_mention(&task.text);
    let owner = if task.user_name.is_empty() {
        String::new()
    } else {
        format!(" ({})", no_mention(&task.user_name))
    };
    let next = local_time(task.next_run_at, task.timezone_offset);
    if let Some(frequency) = frequency(&task.schedule, locale) {
        let next = match locale {
            Locale::Es => format!("prox: {next}"),
            Locale::En => format!("next: {next}"),
        };
        format!("[{}] {text}{owner} - {frequency}, {next}", task.id.as_str())
    } else {
        format!("[{}] {text}{owner} - {next}", task.id.as_str())
    }
}

#[must_use]
pub fn render_task_list(tasks: &[ScheduledTask], locale: Locale) -> TaskListView {
    if tasks.is_empty() {
        return TaskListView {
            text: match locale {
                Locale::Es => "no hay tareas",
                Locale::En => "there are no tasks",
            }
            .to_owned(),
            keyboard: None,
        };
    }
    let text = tasks
        .iter()
        .map(|task| format!("• {}", format_task_summary(task, locale)))
        .collect::<Vec<_>>()
        .join("\n");
    let keyboard = InlineKeyboardMarkup {
        inline_keyboard: tasks
            .iter()
            .map(|task| {
                vec![InlineKeyboardButton {
                    text: match locale {
                        Locale::Es => format!("borrar {}", task.id.as_str()),
                        Locale::En => format!("delete {}", task.id.as_str()),
                    },
                    url: None,
                    callback_data: Some(format!("task:del:{}", task.id.as_str())),
                    copy_text: None,
                }]
            })
            .collect(),
    };
    TaskListView {
        text,
        keyboard: Some(keyboard),
    }
}

#[must_use]
pub fn task_not_found(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "esa tarea no existe",
        Locale::En => "that task does not exist",
    }
}

#[must_use]
pub fn task_load_failed(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "no pude leer las tareas, probá de nuevo",
        Locale::En => "I could not load the tasks, try again",
    }
}

#[must_use]
pub fn task_delete_failed(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "no pude borrar la tarea, probá de nuevo",
        Locale::En => "I could not delete the task, try again",
    }
}

#[must_use]
pub fn task_delete_forbidden(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "solo el creador o un admin pueden borrar esta tarea",
        Locale::En => "only the creator or an admin can delete this task",
    }
}

#[must_use]
pub fn task_deleted(task_id: &TaskId, locale: Locale) -> String {
    match locale {
        Locale::Es => format!("tarea {} borrada", task_id.as_str()),
        Locale::En => format!("task {} deleted", task_id.as_str()),
    }
}

#[cfg(test)]
mod tests {
    use chrono::Weekday;

    use super::{
        TaskCallbackParse, can_delete_task, format_task_summary, parse_task_callback,
        render_task_list, task_delete_failed, task_delete_forbidden, task_deleted,
        task_load_failed, task_not_found,
    };
    use crate::locale::Locale;
    use crate::scheduled_tasks::{ScheduledTask, TaskId, TaskSchedule, TaskStateError};

    fn task(
        id: &str,
        schedule: TaskSchedule,
        next_run_at: i64,
    ) -> Result<ScheduledTask, TaskStateError> {
        Ok(ScheduledTask {
            id: TaskId::new(id)?,
            chat_id: "-100123".to_owned(),
            text: "avisar a @user".to_owned(),
            user_name: "@owner".to_owned(),
            user_id: Some(42),
            schedule,
            timezone_offset: -3,
            locale: "es".to_owned(),
            schedule_anchor_at: None,
            next_run_at: Some(next_run_at),
            last_execution_id: None,
        })
    }

    #[test]
    fn renders_one_shot_and_recurring_summaries_without_mentions() -> Result<(), TaskStateError> {
        let one_shot = task("once0001", TaskSchedule::Once, 1_777_523_400)?;
        assert_eq!(
            format_task_summary(&one_shot, Locale::Es),
            "[once0001] avisar a @\u{200b}user (@\u{200b}owner) - 30/04 01:30"
        );
        let interval = task(
            "repeat01",
            TaskSchedule::IntervalSeconds { seconds: 3_600 },
            1_777_523_400,
        )?;
        assert_eq!(
            format_task_summary(&interval, Locale::En),
            "[repeat01] avisar a @\u{200b}user (@\u{200b}owner) - every 1 hour, next: 30/04 01:30"
        );
        Ok(())
    }

    #[test]
    fn renders_daily_weekly_monthly_and_day_intervals() -> Result<(), TaskStateError> {
        let timestamp = 1_777_523_400;
        let weekly = task(
            "weekly01",
            TaskSchedule::Cron {
                hour: 9,
                minute: 5,
                weekdays: vec![Weekday::Mon, Weekday::Wed],
                day: None,
            },
            timestamp,
        )?;
        assert!(format_task_summary(&weekly, Locale::Es).contains("los lun, mie a las 09:05"));
        let monthly = task(
            "monthly1",
            TaskSchedule::Cron {
                hour: 7,
                minute: 0,
                weekdays: Vec::new(),
                day: Some(12),
            },
            timestamp,
        )?;
        assert!(
            format_task_summary(&monthly, Locale::En).contains("on day 12 of every month at 07:00")
        );
        let daily = task(
            "daily001",
            TaskSchedule::Cron {
                hour: 20,
                minute: 30,
                weekdays: Vec::new(),
                day: None,
            },
            timestamp,
        )?;
        assert!(format_task_summary(&daily, Locale::Es).contains("todos los dias a las 20:30"));
        let days = task(
            "days0001",
            TaskSchedule::IntervalDays { days: 2 },
            timestamp,
        )?;
        assert!(format_task_summary(&days, Locale::En).contains("every 2 days"));
        Ok(())
    }

    #[test]
    fn renders_list_keyboard_and_empty_state() -> Result<(), TaskStateError> {
        let item = task("once0001", TaskSchedule::Once, 1_777_523_400)?;
        let list = render_task_list(&[item], Locale::Es);
        assert!(list.text.starts_with("• [once0001]"));
        let keyboard = list
            .keyboard
            .map_or(Vec::new(), |value| value.inline_keyboard);
        assert_eq!(keyboard[0][0].text, "borrar once0001");
        assert_eq!(
            keyboard[0][0].callback_data.as_deref(),
            Some("task:del:once0001")
        );
        assert_eq!(render_task_list(&[], Locale::En).text, "there are no tasks");
        Ok(())
    }

    #[test]
    fn parses_legacy_callback_shape_and_authorizes_owner_or_admin() {
        assert!(matches!(
            parse_task_callback("task:del:once0001"),
            TaskCallbackParse::Delete(_)
        ));
        assert_eq!(parse_task_callback("task:del"), TaskCallbackParse::Guard);
        assert_eq!(
            parse_task_callback("task:del:bad:id"),
            TaskCallbackParse::Guard
        );
        assert!(can_delete_task(true, Some(42), Some(42), false));
        assert!(can_delete_task(true, Some(7), Some(42), true));
        assert!(!can_delete_task(true, Some(7), Some(42), false));
        assert!(can_delete_task(false, None, Some(42), false));
    }

    #[test]
    fn localizes_callback_answers() {
        let id = TaskId::new("once0001");
        assert_eq!(task_not_found(Locale::Es), "esa tarea no existe");
        assert_eq!(
            task_delete_forbidden(Locale::En),
            "only the creator or an admin can delete this task"
        );
        assert_eq!(
            task_load_failed(Locale::Es),
            "no pude leer las tareas, probá de nuevo"
        );
        assert_eq!(
            task_delete_failed(Locale::En),
            "I could not delete the task, try again"
        );
        assert_eq!(
            id.as_ref().map(|id| task_deleted(id, Locale::En)),
            Ok("task once0001 deleted".to_owned())
        );
    }

    #[test]
    fn covers_all_localized_frequency_and_unknown_time_shapes() -> Result<(), TaskStateError> {
        assert_eq!(
            parse_task_callback("other:del:once0001"),
            TaskCallbackParse::Guard
        );

        let minute = task(
            "minute01",
            TaskSchedule::IntervalSeconds { seconds: 300 },
            1_777_523_400,
        )?;
        assert!(format_task_summary(&minute, Locale::Es).contains("cada 5 minutos"));
        let day = task(
            "day00001",
            TaskSchedule::IntervalSeconds { seconds: 86_400 },
            1_777_523_400,
        )?;
        assert!(format_task_summary(&day, Locale::En).contains("every 1 day"));
        let days = task(
            "days0002",
            TaskSchedule::IntervalSeconds { seconds: 172_800 },
            1_777_523_400,
        )?;
        assert!(format_task_summary(&days, Locale::Es).contains("cada 2 dias"));
        let interval_days = task(
            "days0003",
            TaskSchedule::IntervalDays { days: 1 },
            1_777_523_400,
        )?;
        assert!(format_task_summary(&interval_days, Locale::Es).contains("cada 1 dias"));

        let all_weekdays = vec![
            Weekday::Mon,
            Weekday::Tue,
            Weekday::Wed,
            Weekday::Thu,
            Weekday::Fri,
            Weekday::Sat,
            Weekday::Sun,
        ];
        let weekly = task(
            "weekly02",
            TaskSchedule::Cron {
                hour: 8,
                minute: 0,
                weekdays: all_weekdays,
                day: None,
            },
            1_777_523_400,
        )?;
        assert!(
            format_task_summary(&weekly, Locale::Es).contains("lun, mar, mie, jue, vie, sab, dom")
        );
        assert!(
            format_task_summary(&weekly, Locale::En).contains("mon, tue, wed, thu, fri, sat, sun")
        );

        let mut unknown = task("unknown1", TaskSchedule::Once, 1_777_523_400)?;
        unknown.user_name.clear();
        unknown.next_run_at = None;
        assert_eq!(
            format_task_summary(&unknown, Locale::En),
            "[unknown1] avisar a @\u{200b}user - unknown"
        );
        unknown.next_run_at = Some(i64::MAX);
        unknown.timezone_offset = i32::MAX;
        assert!(format_task_summary(&unknown, Locale::Es).ends_with(" - unknown"));

        assert_eq!(render_task_list(&[], Locale::Es).text, "no hay tareas");
        let english = render_task_list(&[weekly], Locale::En);
        assert_eq!(
            english
                .keyboard
                .and_then(|keyboard| keyboard.inline_keyboard.into_iter().next())
                .and_then(|row| row.into_iter().next())
                .map(|button| button.text),
            Some("delete weekly02".to_owned())
        );
        assert_eq!(task_not_found(Locale::En), "that task does not exist");
        assert_eq!(
            task_delete_forbidden(Locale::Es),
            "solo el creador o un admin pueden borrar esta tarea"
        );
        let id = TaskId::new("once0001")?;
        assert_eq!(task_deleted(&id, Locale::Es), "tarea once0001 borrada");
        Ok(())
    }
}
