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
typedef struct Gemma4DSparkDrafter Gemma4DSparkDrafter;
typedef struct Gemma4KvCache Gemma4KvCache;
typedef struct Gemma4KvSnapshot Gemma4KvSnapshot;
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

typedef enum Gemma4PrefillChunkMode {
    GEMMA4_PREFILL_CHUNK_DISABLED = 0,
    GEMMA4_PREFILL_CHUNK_FIXED_TOKENS = 1,
    GEMMA4_PREFILL_CHUNK_LONG_CONTEXT_256 = 2
} Gemma4PrefillChunkMode;

typedef struct Gemma4PrefillChunkPolicy {
    Gemma4PrefillChunkMode mode;
    uint32_t fixed_chunk_tokens;
} Gemma4PrefillChunkPolicy;

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

#define GEMMA4_MTP_TRACE_MAX_POSITIONS 16
#define GEMMA4_MTP_MAX_DRAFT_TOKENS (GEMMA4_MTP_TRACE_MAX_POSITIONS - 1)
#define GEMMA4_MTP_MAX_COMMITTED_TOKENS GEMMA4_MTP_TRACE_MAX_POSITIONS
#define GEMMA4_MTP_TRACE_TOP_K 5
#define GEMMA4_MTP_TRACE_MAX_RANK 4

#define GEMMA4_DSPARK_TARGET_TAP_COUNT 5
#define GEMMA4_DSPARK_MAX_DRAFT_TOKENS 7
#define GEMMA4_DSPARK_TAP_MAX_RANK 4

typedef struct Gemma4DSparkTapConfig {
    bool enabled;
    uint32_t layer_count;
    uint32_t layer_ids[GEMMA4_DSPARK_TARGET_TAP_COUNT];
} Gemma4DSparkTapConfig;

typedef struct Gemma4DSparkTapInfo {
    bool has_last_hidden;
    uint32_t layer_count;
    uint32_t layer_ids[GEMMA4_DSPARK_TARGET_TAP_COUNT];
    uint32_t tap_ranks[GEMMA4_DSPARK_TARGET_TAP_COUNT];
    uint64_t tap_shapes[GEMMA4_DSPARK_TARGET_TAP_COUNT * GEMMA4_DSPARK_TAP_MAX_RANK];
    uint64_t tap_bytes;
} Gemma4DSparkTapInfo;

typedef struct Gemma4DSparkDraftResult {
    uint32_t token_count;
    int32_t tokens[GEMMA4_DSPARK_MAX_DRAFT_TOKENS];
    float logits[GEMMA4_DSPARK_MAX_DRAFT_TOKENS];
    float logit_margins[GEMMA4_DSPARK_MAX_DRAFT_TOKENS];
    float confidence[GEMMA4_DSPARK_MAX_DRAFT_TOKENS];
    double draft_ms;
    uint64_t scheduler_us;
} Gemma4DSparkDraftResult;

typedef struct Gemma4MtpTraceInfo {
    uint32_t position_count;
    uint32_t top_k;
    uint64_t context_sequence_len;
    uint64_t first_position;
    uint64_t position_offsets[GEMMA4_MTP_TRACE_MAX_POSITIONS];
    int32_t draft_tokens[GEMMA4_MTP_TRACE_MAX_POSITIONS];
    int32_t target_tokens[GEMMA4_MTP_TRACE_MAX_POSITIONS];
    float target_logits[GEMMA4_MTP_TRACE_MAX_POSITIONS];
    float draft_logits[GEMMA4_MTP_TRACE_MAX_POSITIONS];
    float logit_margins[GEMMA4_MTP_TRACE_MAX_POSITIONS];
    bool draft_in_top_k[GEMMA4_MTP_TRACE_MAX_POSITIONS];
    int32_t top_token_ids[GEMMA4_MTP_TRACE_MAX_POSITIONS * GEMMA4_MTP_TRACE_TOP_K];
    float top_logits[GEMMA4_MTP_TRACE_MAX_POSITIONS * GEMMA4_MTP_TRACE_TOP_K];
    uint32_t hidden_rank;
    uint64_t hidden_shape[GEMMA4_MTP_TRACE_MAX_RANK];
    uint32_t full_attention_layer;
    uint32_t full_attention_key_rank;
    uint64_t full_attention_key_shape[GEMMA4_MTP_TRACE_MAX_RANK];
    uint32_t full_attention_value_rank;
    uint64_t full_attention_value_shape[GEMMA4_MTP_TRACE_MAX_RANK];
    uint32_t sliding_attention_layer;
    uint32_t sliding_attention_key_rank;
    uint64_t sliding_attention_key_shape[GEMMA4_MTP_TRACE_MAX_RANK];
    uint32_t sliding_attention_value_rank;
    uint64_t sliding_attention_value_shape[GEMMA4_MTP_TRACE_MAX_RANK];
} Gemma4MtpTraceInfo;

typedef struct Gemma4StepResult {
    int32_t greedy_token;
    float greedy_logit;
    float peak_memory_gb;
    float peak_rss_mb;
    uint64_t sequence_len;
    uint64_t active_kv_bytes;
    double verify_stage_ms;
    double verify_forward_ms;
    double verify_repair_ms;
    double repair_clone_ms;
    double repair_forward_ms;
    double repair_fallback_ms;
    uint32_t accepted_draft_count;
    uint32_t committed_count;
    int32_t committed_tokens[GEMMA4_MTP_MAX_COMMITTED_TOKENS];
    /* Opaque view owned by the KV cache; valid until cache reset/free or next cache-advancing call. */
    void* native_last_hidden;
    /* Trace diagnostics populated by MTP verification; native prefill/decode may carry one target top-k step. */
    Gemma4MtpTraceInfo mtp_trace;
} Gemma4StepResult;

typedef struct Gemma4KvSnapshotInfo {
    uint64_t sequence_len;
    uint64_t active_kv_bytes;
    uint64_t token_count;
    bool has_last_step;
} Gemma4KvSnapshotInfo;

typedef struct Gemma4AdapterLoadConfig {
    const char* adapter_path;
    const char* adapter_id;
    const char* adapter_weight_hash;
    const char* target_modules_csv;
    uint32_t rank;
    float alpha;
} Gemma4AdapterLoadConfig;

typedef struct Gemma4AdapterInfo {
    uint64_t module_count;
    uint64_t resident_bytes;
    uint64_t load_latency_us;
    bool active;
} Gemma4AdapterInfo;

Gemma4Status gemma4_runtime_version(Gemma4VersionInfo* out);
Gemma4Status gemma4_get_last_error(char* buffer, size_t buffer_len);

Gemma4Status gemma4_load_target(const Gemma4LoadConfig* config, Gemma4Target** out);
Gemma4Status gemma4_free_target(Gemma4Target* target);
Gemma4Status gemma4_target_set_prefill_chunk_policy(
    Gemma4Target* target,
    const Gemma4PrefillChunkPolicy* policy);
Gemma4Status gemma4_target_set_dspark_taps(
    Gemma4Target* target,
    const Gemma4DSparkTapConfig* config);

Gemma4Status gemma4_load_adapter(
    Gemma4Target* target,
    const Gemma4AdapterLoadConfig* config,
    Gemma4Adapter** out,
    Gemma4AdapterInfo* info);
Gemma4Status gemma4_free_adapter(Gemma4Adapter* adapter);
Gemma4Status gemma4_set_adapter(Gemma4Target* target, Gemma4Adapter* adapter, Gemma4AdapterInfo* info);
Gemma4Status gemma4_clear_adapter(Gemma4Target* target, Gemma4AdapterInfo* info);

Gemma4Status gemma4_kv_create(const Gemma4KvPolicy* policy, Gemma4KvCache** out);
Gemma4Status gemma4_kv_free(Gemma4KvCache* cache);
Gemma4Status gemma4_kv_reset(Gemma4KvCache* cache);
Gemma4Status gemma4_kv_last_step(const Gemma4KvCache* cache, Gemma4StepResult* out);
Gemma4Status gemma4_kv_dspark_tap_info(const Gemma4KvCache* cache, Gemma4DSparkTapInfo* out);
Gemma4Status gemma4_kv_snapshot_export(const Gemma4KvCache* cache, Gemma4KvSnapshot** out);
Gemma4Status gemma4_kv_snapshot_import(Gemma4KvCache* cache, const Gemma4KvSnapshot* snapshot);
Gemma4Status gemma4_kv_snapshot_info(const Gemma4KvSnapshot* snapshot, Gemma4KvSnapshotInfo* out);
Gemma4Status gemma4_kv_snapshot_save(const Gemma4KvSnapshot* snapshot, const char* payload_path);
Gemma4Status gemma4_kv_snapshot_save_mtp_parity(
    const Gemma4KvSnapshot* snapshot,
    Gemma4Target* target,
    const int32_t* token_ids,
    size_t token_count,
    const char* payload_path);
Gemma4Status gemma4_kv_snapshot_save_compressed(
    const Gemma4KvSnapshot* snapshot,
    const char* payload_path,
    Gemma4KvMode mode,
    bool compress_global_layers,
    bool compress_sliding_layers);
Gemma4Status gemma4_kv_snapshot_load(const char* payload_path, Gemma4KvSnapshot** out);
Gemma4Status gemma4_kv_snapshot_free(Gemma4KvSnapshot* snapshot);

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
Gemma4Status gemma4_decode_block(
    Gemma4Target* target,
    Gemma4KvCache* cache,
    const int32_t* tokens,
    size_t token_count,
    int32_t* out_greedy_tokens,
    float* out_greedy_logits,
    size_t* inout_count,
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
    float* out_logits,
    float* out_logit_margins,
    size_t* inout_count);
Gemma4Status gemma4_load_dspark_drafter(
    const Gemma4LoadConfig* config,
    Gemma4Target* target,
    Gemma4DSparkDrafter** out);
Gemma4Status gemma4_free_dspark_drafter(Gemma4DSparkDrafter* drafter);
Gemma4Status gemma4_dspark_draft_block(
    Gemma4DSparkDrafter* drafter,
    Gemma4KvCache* cache,
    uint32_t max_block_size,
    Gemma4DSparkDraftResult* out);
Gemma4Status gemma4_verify_tokens(
    Gemma4Target* target,
    Gemma4KvCache* cache,
    const int32_t* draft_tokens,
    size_t draft_count,
    Gemma4StepResult* out);
Gemma4Status gemma4_verify_tokens_terminal_no_lookahead(
    Gemma4Target* target,
    Gemma4KvCache* cache,
    const int32_t* draft_tokens,
    size_t draft_count,
    size_t terminal_commit_count,
    Gemma4StepResult* out);

#ifdef __cplusplus
}
#endif

#endif /* GEMMA4_MLX_H */
