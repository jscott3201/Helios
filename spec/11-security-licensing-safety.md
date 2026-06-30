# 11 — Security, Licensing, and Safety

## Local security posture

The runtime is a local developer tool first. Even so:

- Bind to localhost by default.
- Require explicit config to bind non-localhost.
- Do not enable unauthenticated adapter load endpoints on non-localhost.
- Reject path traversal in adapter/cache/model paths.
- Keep cache files under configured directories.
- Avoid exposing raw prompt/cache data through metrics.

## Adapter safety

Adapters are executable-influence artifacts even when they are not code. The engine must:

- load adapters only from trusted local paths by default,
- validate manifest hashes and shapes,
- reject `adapter_model.bin` unless explicitly enabled because pickle-backed formats can be unsafe,
- prefer safetensors,
- reject tokenizer changes unless a local config explicitly allows them,
- reject unexpected `modules_to_save` in MVP.

## FFI safety

- No C++ exception crosses C ABI.
- No Rust panic crosses C ABI.
- Every unsafe block has a comment explaining invariants.
- All native allocations have matching free functions.
- Native error messages are copied safely.

## Licensing review

Before copying code from any external repo, run a license review. Use external repos primarily as design references unless their licenses are compatible and attribution obligations are understood.

Do not import GPL code into a differently licensed runtime without explicit project decision.
