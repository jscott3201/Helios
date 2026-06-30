#![doc = "Placeholder crate for the future local OpenAI-compatible server."]

pub const CRATE_NAME: &str = "gemma4d-server";

pub fn bootstrap_status() -> &'static str {
    "placeholder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_placeholder_status() {
        assert_eq!(CRATE_NAME, "gemma4d-server");
        assert_eq!(bootstrap_status(), "placeholder");
    }
}
