# 03 — Rust / MLX FFI Contract

## Goal

Expose the smallest stable C ABI needed for Rust to drive Gemma 4 12B inference without leaking MLX internals into Rust.

## Principles

- Rust sees opaque handles, not raw MLX arrays.
- Native side owns MLX lifetimes and device synchronization.
- Every FFI call returns an explicit status and error string retrieval path.
- No C++ exceptions cross the ABI.
- Rust wrapper types are `Send`/`Sync` only when proven safe.
- Unsafe is isolated to `gemma4d-ffi`.

## Opaque handles

```c
typedef struct Gemma4Target Gemma4Target;
typedef struct Gemma4Drafter Gemma4Drafter;
typedef struct Gemma4KvCache Gemma4KvCache;
typedef struct Gemma4Adapter Gemma4Adapter;
typedef struct Gemma4StepResult Gemma4StepResult;
```

## Initial C ABI surface

See `references/ffi/gemma4_mlx.h` for the draft header.

Required first functions:

```c
Gemma4Status gemma4_runtime_version(Gemma4VersionInfo* out);
Gemma4Status gemma4_load_target(const Gemma4LoadConfig* cfg, Gemma4Target** out);
Gemma4Status gemma4_free_target(Gemma4Target* target);
Gemma4Status gemma4_kv_create(const Gemma4KvPolicy* policy, Gemma4KvCache** out);
Gemma4Status gemma4_prefill(Gemma4Target* target, Gemma4KvCache* cache, const int32_t* tokens, size_t n, Gemma4StepResult* out);
Gemma4Status gemma4_decode_one(Gemma4Target* target, Gemma4KvCache* cache, int32_t token, Gemma4StepResult* out);
Gemma4Status gemma4_get_last_error(char* buffer, size_t buffer_len);
```

MTP functions are added in M06. Adapter functions are added in M10.

## Build contract

- Use `cmake` to build native MLX shim.
- Rust `build.rs` must locate/build/link the native static or dynamic library reproducibly.
- `cargo test` must run FFI smoke tests without loading the full model by default.
- Full model tests must be gated behind an environment flag such as `GEMMA4D_FULL_MODEL_TESTS=1`.

## Safety checklist

- Null pointer checks on every exported function.
- Input lengths validated before pointer dereference.
- Ownership transfer documented for every handle.
- Explicit free functions for every allocated handle.
- Error paths do not leak native allocations.
- Rust wrappers provide `Drop` for owned handles.
- Panic boundaries: Rust does not unwind into C++; C++ does not throw into C.
