#![doc = "Placeholder crate for tokenizer loading, validation, and hashing."]

pub const CRATE_NAME: &str = "gemma4d-tokenizer";

pub fn bootstrap_status() -> &'static str {
    "placeholder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_placeholder_status() {
        assert_eq!(CRATE_NAME, "gemma4d-tokenizer");
        assert_eq!(bootstrap_status(), "placeholder");
    }
}
