# Native Build Layout

M00/M01 should create:

```text
native/gemma4_mlx/CMakeLists.txt
native/gemma4_mlx/include/gemma4_mlx.h
native/gemma4_mlx/src/runtime.cc
crates/gemma4d-ffi/build.rs
```

Build contract:

```bash
cargo test -p gemma4d-ffi
```

should build native smoke code and run lifecycle tests without downloading a model.

Full model tests must require:

```bash
GEMMA4D_FULL_MODEL_TESTS=1 GEMMA4D_MODEL_PATH=/path/to/model cargo test --features full-model
```
