#![doc = "Placeholder crate for RAM and SSD prefix-cache policy."]

pub const CRATE_NAME: &str = "gemma4d-cache";

pub fn bootstrap_status() -> &'static str {
    "placeholder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_placeholder_status() {
        assert_eq!(CRATE_NAME, "gemma4d-cache");
        assert_eq!(bootstrap_status(), "placeholder");
    }
}
