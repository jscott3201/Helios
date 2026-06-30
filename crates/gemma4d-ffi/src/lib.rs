#![doc = "Placeholder crate for the narrow Rust/native FFI boundary."]

pub const CRATE_NAME: &str = "gemma4d-ffi";

pub fn bootstrap_status() -> &'static str {
    "placeholder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_placeholder_status() {
        assert_eq!(CRATE_NAME, "gemma4d-ffi");
        assert_eq!(bootstrap_status(), "placeholder");
    }
}
