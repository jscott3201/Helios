#![doc = "Placeholder crate for the Ratatui local operator console."]

pub const CRATE_NAME: &str = "gemma4d-tui";

pub fn bootstrap_status() -> &'static str {
    "placeholder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_placeholder_status() {
        assert_eq!(CRATE_NAME, "gemma4d-tui");
        assert_eq!(bootstrap_status(), "placeholder");
    }
}
