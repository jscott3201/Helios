.PHONY: metadata fmt clippy test native-smoke verify

metadata:
	cargo metadata --format-version=1 --no-deps >/dev/null

fmt:
	cargo fmt --all --check

clippy:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
	cargo test --workspace --all-targets --all-features

native-smoke:
	./scripts/native-smoke.sh

verify:
	./scripts/verify.sh
