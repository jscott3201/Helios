#include "gemma4_mlx.h"

#include <cstdio>
#include <cstring>
#include <new>

namespace {

constexpr uint64_t kTargetMagic = 0x47454d3444415447ULL;
constexpr uint64_t kKvCacheMagic = 0x47454d344b564347ULL;
thread_local char g_last_error[512] = "";

struct NativeTarget {
    uint64_t magic;
};

struct NativeKvCache {
    uint64_t magic;
    Gemma4KvPolicy policy;
};

void store_error(const char* message) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", message ? message : "unknown native error");
}

Gemma4Status fail(Gemma4Status status, const char* message) {
    store_error(message);
    return status;
}

Gemma4Status ok() {
    g_last_error[0] = '\0';
    return GEMMA4_OK;
}

bool is_empty(const char* value) {
    return value == nullptr || value[0] == '\0';
}

void clear_step_result(Gemma4StepResult* out) {
    if (out != nullptr) {
        std::memset(out, 0, sizeof(Gemma4StepResult));
    }
}

} // namespace

struct Gemma4Target : NativeTarget {};
struct Gemma4KvCache : NativeKvCache {};
struct Gemma4Drafter {};
struct Gemma4Adapter {};

Gemma4Status gemma4_runtime_version(Gemma4VersionInfo* out) {
    if (out == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_runtime_version requires a non-null out pointer");
    }

    out->abi_version = 1;
    out->backend_name = "gemma4_mlx";
    out->backend_version = "m01-ffi-smoke";
    return ok();
}

Gemma4Status gemma4_get_last_error(char* buffer, size_t buffer_len) {
    if (buffer == nullptr || buffer_len == 0) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_get_last_error requires a writable buffer");
    }

    std::snprintf(buffer, buffer_len, "%s", g_last_error);
    return GEMMA4_OK;
}

Gemma4Status gemma4_load_target(const Gemma4LoadConfig* config, Gemma4Target** out) {
    if (out == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_target requires a non-null out pointer");
    }
    *out = nullptr;

    if (config == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_target requires a non-null config");
    }
    if (is_empty(config->model_path)) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_target requires a non-empty model_path");
    }
    if (config->max_context_tokens == 0) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_target requires max_context_tokens > 0");
    }

    Gemma4Target* target = new (std::nothrow) Gemma4Target{};
    if (target == nullptr) {
        return fail(GEMMA4_ERR_RUNTIME, "gemma4_load_target could not allocate target handle");
    }

    target->magic = kTargetMagic;
    *out = target;
    return ok();
}

Gemma4Status gemma4_free_target(Gemma4Target* target) {
    if (target == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_free_target requires a non-null target");
    }
    if (target->magic != kTargetMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_free_target received an invalid target handle");
    }

    target->magic = 0;
    delete target;
    return ok();
}

Gemma4Status gemma4_kv_create(const Gemma4KvPolicy* policy, Gemma4KvCache** out) {
    if (out == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_create requires a non-null out pointer");
    }
    *out = nullptr;

    if (policy == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_create requires a non-null policy");
    }
    if (policy->block_size_tokens == 0) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_create requires block_size_tokens > 0");
    }

    Gemma4KvCache* cache = new (std::nothrow) Gemma4KvCache{};
    if (cache == nullptr) {
        return fail(GEMMA4_ERR_RUNTIME, "gemma4_kv_create could not allocate KV cache handle");
    }

    cache->magic = kKvCacheMagic;
    cache->policy = *policy;
    *out = cache;
    return ok();
}

Gemma4Status gemma4_kv_free(Gemma4KvCache* cache) {
    if (cache == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_free requires a non-null cache");
    }
    if (cache->magic != kKvCacheMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_free received an invalid cache handle");
    }

    cache->magic = 0;
    delete cache;
    return ok();
}

Gemma4Status gemma4_kv_reset(Gemma4KvCache* cache) {
    if (cache == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_reset requires a non-null cache");
    }
    if (cache->magic != kKvCacheMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_reset received an invalid cache handle");
    }

    return ok();
}

Gemma4Status gemma4_prefill(
    Gemma4Target* target,
    Gemma4KvCache* cache,
    const int32_t* tokens,
    size_t token_count,
    Gemma4StepResult* out) {
    clear_step_result(out);

    if (target == nullptr || target->magic != kTargetMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_prefill requires a valid target handle");
    }
    if (cache == nullptr || cache->magic != kKvCacheMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_prefill requires a valid cache handle");
    }
    if (out == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_prefill requires a non-null step result");
    }
    if (token_count > 0 && tokens == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_prefill requires tokens when token_count > 0");
    }

    return fail(GEMMA4_ERR_UNSUPPORTED_CONFIG, "gemma4_prefill is not implemented in M01 smoke runtime");
}

Gemma4Status gemma4_decode_one(
    Gemma4Target* target,
    Gemma4KvCache* cache,
    int32_t token,
    Gemma4StepResult* out) {
    (void)token;
    clear_step_result(out);

    if (target == nullptr || target->magic != kTargetMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_decode_one requires a valid target handle");
    }
    if (cache == nullptr || cache->magic != kKvCacheMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_decode_one requires a valid cache handle");
    }
    if (out == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_decode_one requires a non-null step result");
    }

    return fail(GEMMA4_ERR_UNSUPPORTED_CONFIG, "gemma4_decode_one is not implemented in M01 smoke runtime");
}
