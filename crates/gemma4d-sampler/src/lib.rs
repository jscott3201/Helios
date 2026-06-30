#![doc = "Placeholder crate for sampling policy and deterministic greedy mode."]

pub const CRATE_NAME: &str = "gemma4d-sampler";

pub fn bootstrap_status() -> &'static str {
    "placeholder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_placeholder_status() {
        assert_eq!(CRATE_NAME, "gemma4d-sampler");
        assert_eq!(bootstrap_status(), "placeholder");
    }
}
