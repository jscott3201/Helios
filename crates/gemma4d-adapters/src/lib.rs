#![doc = "Placeholder crate for adapter manifests, trust policy, and routing state."]

pub const CRATE_NAME: &str = "gemma4d-adapters";

pub fn bootstrap_status() -> &'static str {
    "placeholder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_placeholder_status() {
        assert_eq!(CRATE_NAME, "gemma4d-adapters");
        assert_eq!(bootstrap_status(), "placeholder");
    }
}
