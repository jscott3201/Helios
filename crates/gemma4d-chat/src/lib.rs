#![doc = "Placeholder crate for chat-template application and prompt hashing."]

pub const CRATE_NAME: &str = "gemma4d-chat";

pub fn bootstrap_status() -> &'static str {
    "placeholder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_placeholder_status() {
        assert_eq!(CRATE_NAME, "gemma4d-chat");
        assert_eq!(bootstrap_status(), "placeholder");
    }
}
