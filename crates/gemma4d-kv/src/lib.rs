#![doc = "Placeholder crate for KV cache identities, block metadata, and rollback state."]

pub const CRATE_NAME: &str = "gemma4d-kv";

pub fn bootstrap_status() -> &'static str {
    "placeholder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_placeholder_status() {
        assert_eq!(CRATE_NAME, "gemma4d-kv");
        assert_eq!(bootstrap_status(), "placeholder");
    }
}
