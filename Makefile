.PHONY: metadata fmt clippy test ffi-smoke native-smoke native-mlx-diagnostics verify

metadata:
	cargo metadata --format-version=1 --no-deps >/dev/null

fmt:
	cargo fmt --all --check

clippy:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
	cargo test --workspace --all-targets --all-features

ffi-smoke:
	cargo test -p gemma4d-ffi

native-smoke:
	./scripts/native-smoke.sh

native-mlx-diagnostics:
	./scripts/mlx-diagnostics.sh

verify:
	./scripts/verify.sh
