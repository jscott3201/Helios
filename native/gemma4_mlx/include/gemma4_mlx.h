#ifndef GEMMA4_MLX_H
#define GEMMA4_MLX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum Gemma4Status {
    GEMMA4_OK = 0,
    GEMMA4_ERR_INVALID_ARGUMENT = 1,
    GEMMA4_ERR_UNSUPPORTED_CONFIG = 2,
    GEMMA4_ERR_MODEL_LOAD = 3,
    GEMMA4_ERR_RUNTIME = 4,
    GEMMA4_ERR_MEMORY_GUARD = 5,
    GEMMA4_ERR_CACHE = 6,
    GEMMA4_ERR_ADAPTER = 7
} Gemma4Status;

typedef struct Gemma4Target Gemma4Target;
typedef struct Gemma4Drafter Gemma4Drafter;
typedef struct Gemma4KvCache Gemma4KvCache;
typedef struct Gemma4Adapter Gemma4Adapter;

typedef struct Gemma4VersionInfo {
    uint32_t abi_version;
    const char* backend_name;
    const char* backend_version;
} Gemma4VersionInfo;

typedef struct Gemma4LoadConfig {
    const char* model_path;
    const char* model_id;
    const char* model_revision;
    const char* expected_architecture;
    uint32_t max_context_tokens;
    bool allow_unsupported_config;
} Gemma4LoadConfig;

typedef enum Gemma4KvMode {
    GEMMA4_KV_BF16 = 0,
    GEMMA4_KV_MLX_AFFINE_Q8 = 1,
    GEMMA4_KV_MLX_AFFINE_Q4 = 2,
    GEMMA4_KV_PLANAR4_K_BF16_V = 10,
    GEMMA4_KV_PLANAR3_K_BF16_V = 11,
    GEMMA4_KV_ISO4_KV = 12,
    GEMMA4_KV_EXPERIMENTAL_TURBO = 20
} Gemma4KvMode;

typedef struct Gemma4KvPolicy {
    Gemma4KvMode active_mode;
    Gemma4KvMode ram_prefix_mode;
    Gemma4KvMode ssd_prefix_mode;
    uint32_t block_size_tokens;
    uint32_t quantized_kv_start;
    bool compress_global_layers;
    bool compress_sliding_layers;
    bool keep_mtp_shared_layers_bf16;
    bool allow_active_compressed_decode;
} Gemma4KvPolicy;

typedef struct Gemma4StepResult {
    int32_t greedy_token;
    float greedy_logit;
    float peak_memory_gb;
    float peak_rss_mb;
    uint64_t sequence_len;
    /* Opaque view owned by the KV cache; valid until cache reset/free or next cache-advancing call. */
    void* native_last_hidden;
} Gemma4StepResult;

Gemma4Status gemma4_runtime_version(Gemma4VersionInfo* out);
Gemma4Status gemma4_get_last_error(char* buffer, size_t buffer_len);

Gemma4Status gemma4_load_target(const Gemma4LoadConfig* config, Gemma4Target** out);
Gemma4Status gemma4_free_target(Gemma4Target* target);

Gemma4Status gemma4_kv_create(const Gemma4KvPolicy* policy, Gemma4KvCache** out);
Gemma4Status gemma4_kv_free(Gemma4KvCache* cache);
Gemma4Status gemma4_kv_reset(Gemma4KvCache* cache);

Gemma4Status gemma4_prefill(
    Gemma4Target* target,
    Gemma4KvCache* cache,
    const int32_t* tokens,
    size_t token_count,
    Gemma4StepResult* out);

Gemma4Status gemma4_decode_one(
    Gemma4Target* target,
    Gemma4KvCache* cache,
    int32_t token,
    Gemma4StepResult* out);

Gemma4Status gemma4_load_drafter(
    const Gemma4LoadConfig* config,
    Gemma4Target* target,
    Gemma4Drafter** out);
Gemma4Status gemma4_free_drafter(Gemma4Drafter* drafter);
Gemma4Status gemma4_mtp_draft_block(
    Gemma4Drafter* drafter,
    Gemma4KvCache* cache,
    uint32_t block_size,
    int32_t* out_tokens,
    size_t* inout_count);
Gemma4Status gemma4_verify_tokens(
    Gemma4Target* target,
    Gemma4KvCache* cache,
    const int32_t* draft_tokens,
    size_t draft_count,
    Gemma4StepResult* out);

#ifdef __cplusplus
}
#endif

#endif /* GEMMA4_MLX_H */
