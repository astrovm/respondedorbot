//! Plan how a completed vision result enriches a chat request.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageContextPlan {
    NoImage,
    DescriptionFailed,
    DescriptionReady {
        updated_last_content: Option<String>,
    },
}

impl ImageContextPlan {
    #[must_use]
    pub const fn action(&self) -> &'static str {
        match self {
            Self::NoImage => "no_image",
            Self::DescriptionFailed => "description_failed",
            Self::DescriptionReady { .. } => "description_ready",
        }
    }

    #[must_use]
    pub fn updated_last_content(self) -> Option<String> {
        match self {
            Self::DescriptionReady {
                updated_last_content,
            } => updated_last_content,
            Self::NoImage | Self::DescriptionFailed => None,
        }
    }
}

#[must_use]
pub fn plan_image_context(
    has_image_data: bool,
    description: Option<&str>,
    last_text_content: Option<&str>,
    localized_context: &str,
) -> ImageContextPlan {
    if !has_image_data {
        return ImageContextPlan::NoImage;
    }
    if description.is_none() {
        return ImageContextPlan::DescriptionFailed;
    }
    ImageContextPlan::DescriptionReady {
        updated_last_content: last_text_content
            .map(|content| format!("{content}\n\n{localized_context}")),
    }
}

#[cfg(test)]
mod tests {
    use super::{ImageContextPlan, plan_image_context};

    #[test]
    fn missing_image_or_description_does_not_change_messages() {
        let no_image = plan_image_context(false, Some("description"), Some("prompt"), "context");
        assert_eq!(no_image.action(), "no_image");
        assert_eq!(no_image.updated_last_content(), None);

        let failed = plan_image_context(true, None, Some("prompt"), "context");
        assert_eq!(failed.action(), "description_failed");
        assert_eq!(failed.updated_last_content(), None);
    }

    #[test]
    fn completed_description_is_appended_only_to_a_text_tail() {
        let appended = plan_image_context(
            true,
            Some("description"),
            Some("user prompt"),
            "Image: description",
        );
        assert_eq!(appended.action(), "description_ready");
        assert_eq!(
            appended.updated_last_content().as_deref(),
            Some("user prompt\n\nImage: description"),
        );

        let without_tail = plan_image_context(true, Some("description"), None, "context");
        assert_eq!(without_tail.action(), "description_ready");
        assert_eq!(without_tail.updated_last_content(), None);
    }

    #[test]
    fn plan_variants_have_stable_equality_for_state_machine_tests() {
        assert_eq!(
            plan_image_context(false, None, None, ""),
            ImageContextPlan::NoImage,
        );
        assert_eq!(
            plan_image_context(true, None, None, ""),
            ImageContextPlan::DescriptionFailed,
        );
    }
}
