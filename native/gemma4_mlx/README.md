# gemma4_mlx Native Skeleton

M01 provides a narrow C ABI smoke target for Rust FFI lifecycle tests. It does
not load MLX models or allocate MLX arrays.

Default smoke command:

```bash
cmake -S native/gemma4_mlx -B target/native-smoke
cmake --build target/native-smoke
```

The default configuration does not require MLX so repository bootstrap can be
verified on a clean machine without downloading model artifacts. To force the
real dependency check, configure with:

```bash
cmake -S native/gemma4_mlx -B target/native-smoke -DGEMMA4D_REQUIRE_MLX=ON
```

If MLX is unavailable in that mode, CMake fails while resolving `find_package(MLX
CONFIG REQUIRED)`, which is the documented dependency failure for this skeleton.

The Rust FFI crate builds this native library through `crates/gemma4d-ffi/build.rs`
during `cargo test -p gemma4d-ffi`.
