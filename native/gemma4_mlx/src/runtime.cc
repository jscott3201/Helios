#include "gemma4_mlx.h"
#include "model_manifest.h"
#include "native_model.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <filesystem>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#ifdef GEMMA4D_MLX_AVAILABLE
#include <mlx/version.h>
#endif

#define GEMMA4D_STRINGIFY_DETAIL(value) #value
#define GEMMA4D_STRINGIFY(value) GEMMA4D_STRINGIFY_DETAIL(value)

#ifndef GEMMA4D_MLX_LM_HELPER_PATH
#define GEMMA4D_MLX_LM_HELPER_PATH "native/gemma4_mlx/scripts/gemma4d_mlx_lm_helper.py"
#endif

#ifndef GEMMA4D_MLX_LM_PYTHON
#define GEMMA4D_MLX_LM_PYTHON "/opt/homebrew/opt/mlx-lm/libexec/bin/python"
#endif

namespace {

constexpr uint64_t kTargetMagic = 0x47454d3444415447ULL;
constexpr uint64_t kDrafterMagic = 0x47454d3444524146ULL;
constexpr uint64_t kKvCacheMagic = 0x47454d344b564347ULL;
constexpr uint64_t kKvSnapshotMagic = 0x47454d344b565347ULL;
constexpr uint64_t kAdapterMagic = 0x47454d3441445054ULL;
thread_local char g_last_error[512] = "";

double elapsed_ms(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
}

#ifdef GEMMA4D_MLX_AVAILABLE
constexpr const char* kBackendVersion =
    "m03-mlx-build-gated-mlx-" GEMMA4D_STRINGIFY(MLX_VERSION_MAJOR) "." GEMMA4D_STRINGIFY(
        MLX_VERSION_MINOR) "." GEMMA4D_STRINGIFY(MLX_VERSION_PATCH);
#else
constexpr const char* kBackendVersion = "m03-smoke-no-mlx";
#endif

struct NativeTarget {
    uint64_t magic;
    bool model_loaded;
    bool use_native_graph;
    uint64_t sequence_len;
    bool has_prefill_chunk_policy_override;
    Gemma4PrefillChunkPolicy prefill_chunk_policy;
    gemma4d::Gemma4ModelManifest manifest;
    std::unique_ptr<gemma4d::NativeTextModel> native_model;
    pid_t helper_pid;
    FILE* helper_in;
    FILE* helper_out;
};

struct NativeKvCache {
    uint64_t magic;
    Gemma4KvPolicy policy;
    std::vector<int32_t> native_tokens;
    std::unique_ptr<gemma4d::NativeKvState> native_kv_state;
    std::unique_ptr<gemma4d::NativeHiddenState> last_hidden;
    bool has_last_step;
    Gemma4StepResult last_step;
    struct PendingMtpDraftScore {
        int32_t token = 0;
        float logit = 0.0f;
        float margin = 0.0f;
    };
    std::vector<PendingMtpDraftScore> pending_mtp_draft_scores;
};

struct NativeKvSnapshot {
    uint64_t magic;
    Gemma4KvPolicy policy;
    std::vector<int32_t> native_tokens;
    std::unique_ptr<gemma4d::NativeKvState> native_kv_state;
    std::unique_ptr<gemma4d::NativeHiddenState> last_hidden;
    bool has_last_step;
    Gemma4StepResult last_step;
};

struct NativeDrafter {
    uint64_t magic;
    bool model_loaded;
    std::string model_path;
    gemma4d::Gemma4ModelManifest manifest;
    const gemma4d::NativeTextModel* target_native_model;
    std::unique_ptr<gemma4d::NativeMtpAssistantModel> native_model;
};

struct NativeAdapter {
    uint64_t magic;
    std::shared_ptr<const gemma4d::NativeLoraAdapter> native_adapter;
    uint64_t load_latency_us;
};

void store_error(const char* message) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", message ? message : "unknown native error");
}

Gemma4Status fail(Gemma4Status status, const char* message) {
    store_error(message);
    return status;
}

Gemma4Status fail(Gemma4Status status, const std::string& message) {
    store_error(message.c_str());
    return status;
}

Gemma4Status ok() {
    g_last_error[0] = '\0';
    return GEMMA4_OK;
}

bool is_empty(const char* value) {
    return value == nullptr || value[0] == '\0';
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
        std::strcmp(value, "FALSE") != 0 && std::strcmp(value, "off") != 0 &&
        std::strcmp(value, "OFF") != 0;
}

bool mtp_real_margins_enabled() {
    static const bool enabled = env_flag_enabled("GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS");
    return enabled;
}

void clear_step_result(Gemma4StepResult* out) {
    if (out != nullptr) {
        std::memset(out, 0, sizeof(Gemma4StepResult));
    }
}

bool same_kv_policy(const Gemma4KvPolicy& left, const Gemma4KvPolicy& right) {
    return left.active_mode == right.active_mode && left.ram_prefix_mode == right.ram_prefix_mode &&
        left.ssd_prefix_mode == right.ssd_prefix_mode &&
        left.block_size_tokens == right.block_size_tokens &&
        left.quantized_kv_start == right.quantized_kv_start &&
        left.compress_global_layers == right.compress_global_layers &&
        left.compress_sliding_layers == right.compress_sliding_layers &&
        left.keep_mtp_shared_layers_bf16 == right.keep_mtp_shared_layers_bf16 &&
        left.allow_active_compressed_decode == right.allow_active_compressed_decode;
}

void remember_last_step(NativeKvCache* cache, const Gemma4StepResult* step) {
    if (cache == nullptr || step == nullptr) {
        return;
    }
    cache->last_step = *step;
    cache->last_step.native_last_hidden = cache->last_hidden.get();
    cache->last_step.verify_stage_ms = 0.0;
    cache->last_step.verify_forward_ms = 0.0;
    cache->last_step.verify_repair_ms = 0.0;
    cache->last_step.repair_clone_ms = 0.0;
    cache->last_step.repair_forward_ms = 0.0;
    cache->last_step.repair_fallback_ms = 0.0;
    cache->has_last_step = true;
}

void initialize_mtp_trace(Gemma4MtpTraceInfo* trace, uint64_t context_sequence_len) {
    if (trace == nullptr) {
        return;
    }
    std::memset(trace, 0, sizeof(Gemma4MtpTraceInfo));
    trace->context_sequence_len = context_sequence_len;
    trace->first_position = context_sequence_len == 0 ? 0 : context_sequence_len - 1;
    trace->top_k = mtp_real_margins_enabled() ? GEMMA4_MTP_TRACE_TOP_K : 1;
    trace->full_attention_layer = 47;
    trace->sliding_attention_layer = 46;
    for (size_t index = 0; index < GEMMA4_MTP_TRACE_MAX_POSITIONS; ++index) {
        trace->draft_tokens[index] = -1;
        trace->target_tokens[index] = -1;
        for (size_t rank = 0; rank < GEMMA4_MTP_TRACE_TOP_K; ++rank) {
            trace->top_token_ids[index * GEMMA4_MTP_TRACE_TOP_K + rank] = -1;
        }
    }
}

void record_mtp_target_step(
    Gemma4MtpTraceInfo* trace,
    size_t index,
    uint64_t context_sequence_len,
    const Gemma4StepResult& step,
    const gemma4d::NativeTopKEntries* target_top_k = nullptr) {
    if (trace == nullptr) {
        return;
    }
    if (index >= GEMMA4_MTP_TRACE_MAX_POSITIONS) {
        std::ostringstream message;
        message << "native MTP trace target position " << index
                << " exceeds trace capacity " << GEMMA4_MTP_TRACE_MAX_POSITIONS;
        throw std::runtime_error(message.str());
    }
    trace->position_count = std::max<uint32_t>(trace->position_count, static_cast<uint32_t>(index + 1));
    trace->position_offsets[index] = (context_sequence_len == 0 ? 0 : context_sequence_len - 1) + index;
    trace->target_tokens[index] = step.greedy_token;
    trace->target_logits[index] = step.greedy_logit;
    const size_t base = index * GEMMA4_MTP_TRACE_TOP_K;
    if (target_top_k != nullptr) {
        for (size_t rank = 0; rank < GEMMA4_MTP_TRACE_TOP_K; ++rank) {
            trace->top_token_ids[base + rank] = (*target_top_k)[rank].token_id;
            trace->top_logits[base + rank] = (*target_top_k)[rank].logit;
        }
        return;
    }
    if (step.mtp_trace.top_k > 0 && step.mtp_trace.position_count > 0) {
        const size_t source_top_k = std::min<size_t>(step.mtp_trace.top_k, GEMMA4_MTP_TRACE_TOP_K);
        const uint64_t expected_offset = step.sequence_len == 0 ? 0 : step.sequence_len - 1;
        size_t source_index = GEMMA4_MTP_TRACE_MAX_POSITIONS;
        for (size_t position = 0;
             position < std::min<size_t>(step.mtp_trace.position_count, GEMMA4_MTP_TRACE_MAX_POSITIONS);
             ++position) {
            if (step.mtp_trace.position_offsets[position] == expected_offset &&
                step.mtp_trace.target_tokens[position] == step.greedy_token) {
                source_index = position;
                break;
            }
        }
        if (source_index == GEMMA4_MTP_TRACE_MAX_POSITIONS && step.mtp_trace.position_count == 1) {
            source_index = 0;
        }
        if (source_index == GEMMA4_MTP_TRACE_MAX_POSITIONS) {
            trace->top_token_ids[base] = step.greedy_token;
            trace->top_logits[base] = step.greedy_logit;
            return;
        }
        const size_t source_base = source_index * GEMMA4_MTP_TRACE_TOP_K;
        for (size_t rank = 0; rank < source_top_k; ++rank) {
            trace->top_token_ids[base + rank] = step.mtp_trace.top_token_ids[source_base + rank];
            trace->top_logits[base + rank] = step.mtp_trace.top_logits[source_base + rank];
        }
        return;
    }
    trace->top_token_ids[base] = step.greedy_token;
    trace->top_logits[base] = step.greedy_logit;
}

void record_mtp_draft_score(
    Gemma4MtpTraceInfo* trace,
    size_t index,
    int32_t draft_token,
    const NativeKvCache::PendingMtpDraftScore* draft_score) {
    if (trace == nullptr) {
        return;
    }
    if (index >= GEMMA4_MTP_TRACE_MAX_POSITIONS) {
        std::ostringstream message;
        message << "native MTP trace draft position " << index
                << " exceeds trace capacity " << GEMMA4_MTP_TRACE_MAX_POSITIONS;
        throw std::runtime_error(message.str());
    }
    trace->draft_tokens[index] = draft_token;
    if (draft_score != nullptr && draft_score->token == draft_token) {
        trace->draft_logits[index] = draft_score->logit;
        trace->logit_margins[index] = draft_score->margin;
    }
    const size_t base = index * GEMMA4_MTP_TRACE_TOP_K;
    const size_t top_k = std::min<size_t>(trace->top_k, GEMMA4_MTP_TRACE_TOP_K);
    for (size_t rank = 0; rank < top_k; ++rank) {
        if (trace->top_token_ids[base + rank] == draft_token) {
            trace->draft_in_top_k[index] = true;
            break;
        }
    }
}

void record_mtp_hidden_shape(Gemma4MtpTraceInfo* trace, const gemma4d::NativeHiddenState* hidden) {
    if (trace == nullptr || hidden == nullptr || hidden->hidden_size() == 0) {
        return;
    }
    trace->hidden_rank = 3;
    trace->hidden_shape[0] = 1;
    trace->hidden_shape[1] = 1;
    trace->hidden_shape[2] = hidden->hidden_size();
}

std::string join_i32_list(const int32_t* values, size_t count) {
    std::ostringstream out;
    for (size_t index = 0; index < count; ++index) {
        if (index != 0) {
            out << ',';
        }
        out << values[index];
    }
    return out.str();
}

std::string join_vector_i32(const std::vector<int32_t>& values) {
    return values.empty() ? std::string() : join_i32_list(values.data(), values.size());
}

std::vector<int32_t> parse_i32_list(const std::string& value) {
    std::vector<int32_t> values;
    if (value.empty()) {
        return values;
    }
    std::stringstream input(value);
    std::string part;
    while (std::getline(input, part, ',')) {
        if (!part.empty()) {
            values.push_back(static_cast<int32_t>(std::stoi(part)));
        }
    }
    return values;
}

std::vector<std::string> parse_csv_list(const char* value) {
    std::vector<std::string> values;
    if (is_empty(value)) {
        return values;
    }
    std::stringstream input(value);
    std::string part;
    while (std::getline(input, part, ',')) {
        size_t start = 0;
        while (start < part.size() && std::isspace(static_cast<unsigned char>(part[start]))) {
            ++start;
        }
        size_t end = part.size();
        while (end > start && std::isspace(static_cast<unsigned char>(part[end - 1]))) {
            --end;
        }
        if (end > start) {
            values.push_back(part.substr(start, end - start));
        }
    }
    return values;
}

void fill_adapter_info(
    const std::shared_ptr<const gemma4d::NativeLoraAdapter>& adapter,
    uint64_t load_latency_us,
    bool active,
    Gemma4AdapterInfo* info) {
    if (info == nullptr) {
        return;
    }
    info->module_count = adapter ? adapter->module_count() : 0;
    info->resident_bytes = adapter ? adapter->resident_bytes() : 0;
    info->load_latency_us = load_latency_us;
    info->active = active;
}

const std::string& required_metadata(
    const std::unordered_map<std::string, std::string>& metadata,
    const char* key) {
    const auto found = metadata.find(key);
    if (found == metadata.end()) {
        throw std::runtime_error(std::string("snapshot metadata is missing ") + key);
    }
    return found->second;
}

bool metadata_flag(const std::unordered_map<std::string, std::string>& metadata, const char* key) {
    const std::string& value = required_metadata(metadata, key);
    return value == "true" || value == "1";
}

int metadata_i32(const std::unordered_map<std::string, std::string>& metadata, const char* key) {
    return std::stoi(required_metadata(metadata, key));
}

uint32_t metadata_u32(const std::unordered_map<std::string, std::string>& metadata, const char* key) {
    return static_cast<uint32_t>(std::stoul(required_metadata(metadata, key)));
}

uint64_t metadata_u64(const std::unordered_map<std::string, std::string>& metadata, const char* key) {
    return std::stoull(required_metadata(metadata, key));
}

float metadata_float(const std::unordered_map<std::string, std::string>& metadata, const char* key) {
    return std::stof(required_metadata(metadata, key));
}

std::unordered_map<std::string, std::string> snapshot_metadata(const NativeKvSnapshot* snapshot) {
    std::unordered_map<std::string, std::string> metadata;
    metadata["snapshot_format"] = "gemma4d_native_snapshot_v1";
    metadata["policy.active_mode"] = std::to_string(static_cast<int>(snapshot->policy.active_mode));
    metadata["policy.ram_prefix_mode"] = std::to_string(static_cast<int>(snapshot->policy.ram_prefix_mode));
    metadata["policy.ssd_prefix_mode"] = std::to_string(static_cast<int>(snapshot->policy.ssd_prefix_mode));
    metadata["policy.block_size_tokens"] = std::to_string(snapshot->policy.block_size_tokens);
    metadata["policy.quantized_kv_start"] = std::to_string(snapshot->policy.quantized_kv_start);
    metadata["policy.compress_global_layers"] = snapshot->policy.compress_global_layers ? "true" : "false";
    metadata["policy.compress_sliding_layers"] = snapshot->policy.compress_sliding_layers ? "true" : "false";
    metadata["policy.keep_mtp_shared_layers_bf16"] =
        snapshot->policy.keep_mtp_shared_layers_bf16 ? "true" : "false";
    metadata["policy.allow_active_compressed_decode"] =
        snapshot->policy.allow_active_compressed_decode ? "true" : "false";
    metadata["native_tokens.count"] = std::to_string(snapshot->native_tokens.size());
    metadata["native_tokens.csv"] = join_vector_i32(snapshot->native_tokens);
    metadata["has_last_step"] = snapshot->has_last_step ? "true" : "false";
    metadata["last_step.greedy_token"] = std::to_string(snapshot->last_step.greedy_token);
    metadata["last_step.greedy_logit"] = std::to_string(snapshot->last_step.greedy_logit);
    metadata["last_step.peak_memory_gb"] = std::to_string(snapshot->last_step.peak_memory_gb);
    metadata["last_step.peak_rss_mb"] = std::to_string(snapshot->last_step.peak_rss_mb);
    metadata["last_step.sequence_len"] = std::to_string(snapshot->last_step.sequence_len);
    metadata["last_step.active_kv_bytes"] = std::to_string(snapshot->last_step.active_kv_bytes);
    metadata["last_step.accepted_draft_count"] = std::to_string(snapshot->last_step.accepted_draft_count);
    metadata["last_step.committed_count"] = std::to_string(snapshot->last_step.committed_count);
    metadata["last_step.committed_tokens"] = join_i32_list(
        snapshot->last_step.committed_tokens,
        GEMMA4_MTP_MAX_COMMITTED_TOKENS);
    return metadata;
}

void apply_snapshot_metadata(
    const std::unordered_map<std::string, std::string>& metadata,
    NativeKvSnapshot* snapshot) {
    if (required_metadata(metadata, "snapshot_format") != "gemma4d_native_snapshot_v1") {
        throw std::runtime_error("snapshot metadata has an unsupported snapshot format");
    }
    snapshot->policy.active_mode = static_cast<Gemma4KvMode>(metadata_i32(metadata, "policy.active_mode"));
    snapshot->policy.ram_prefix_mode =
        static_cast<Gemma4KvMode>(metadata_i32(metadata, "policy.ram_prefix_mode"));
    snapshot->policy.ssd_prefix_mode =
        static_cast<Gemma4KvMode>(metadata_i32(metadata, "policy.ssd_prefix_mode"));
    snapshot->policy.block_size_tokens = metadata_u32(metadata, "policy.block_size_tokens");
    snapshot->policy.quantized_kv_start = metadata_u32(metadata, "policy.quantized_kv_start");
    snapshot->policy.compress_global_layers = metadata_flag(metadata, "policy.compress_global_layers");
    snapshot->policy.compress_sliding_layers = metadata_flag(metadata, "policy.compress_sliding_layers");
    snapshot->policy.keep_mtp_shared_layers_bf16 = metadata_flag(metadata, "policy.keep_mtp_shared_layers_bf16");
    snapshot->policy.allow_active_compressed_decode =
        metadata_flag(metadata, "policy.allow_active_compressed_decode");
    snapshot->native_tokens = parse_i32_list(required_metadata(metadata, "native_tokens.csv"));
    const uint64_t token_count = metadata_u64(metadata, "native_tokens.count");
    if (snapshot->native_tokens.size() != token_count) {
        throw std::runtime_error("snapshot metadata token count does not match token payload");
    }
    snapshot->has_last_step = metadata_flag(metadata, "has_last_step");
    snapshot->last_step = Gemma4StepResult{};
    snapshot->last_step.greedy_token = metadata_i32(metadata, "last_step.greedy_token");
    snapshot->last_step.greedy_logit = metadata_float(metadata, "last_step.greedy_logit");
    snapshot->last_step.peak_memory_gb = metadata_float(metadata, "last_step.peak_memory_gb");
    snapshot->last_step.peak_rss_mb = metadata_float(metadata, "last_step.peak_rss_mb");
    snapshot->last_step.sequence_len = metadata_u64(metadata, "last_step.sequence_len");
    snapshot->last_step.active_kv_bytes = metadata_u64(metadata, "last_step.active_kv_bytes");
    snapshot->last_step.accepted_draft_count = metadata_u32(metadata, "last_step.accepted_draft_count");
    snapshot->last_step.committed_count = std::min<uint32_t>(
        metadata_u32(metadata, "last_step.committed_count"),
        GEMMA4_MTP_MAX_COMMITTED_TOKENS);
    const std::vector<int32_t> committed = parse_i32_list(required_metadata(metadata, "last_step.committed_tokens"));
    for (size_t index = 0; index < committed.size() && index < GEMMA4_MTP_MAX_COMMITTED_TOKENS; ++index) {
        snapshot->last_step.committed_tokens[index] = committed[index];
    }
}

bool has_safetensors_file(const std::filesystem::path& model_dir) {
    std::error_code error;
    std::filesystem::directory_iterator current(model_dir, error);
    std::filesystem::directory_iterator end;
    while (!error && current != end) {
        const std::filesystem::directory_entry& entry = *current;
        if (entry.is_regular_file(error) && entry.path().extension() == ".safetensors") {
            return true;
        }
        current.increment(error);
    }
    return false;
}

std::string errno_message(const char* action) {
    std::ostringstream message;
    message << action << ": " << std::strerror(errno);
    return message.str();
}

Gemma4Status validate_strict_model_artifacts(const char* model_path) {
    std::error_code error;
    const std::filesystem::path path(model_path);

    if (!std::filesystem::exists(path, error)) {
        return fail(GEMMA4_ERR_MODEL_LOAD, "model_path does not exist: " + path.string());
    }
    if (!std::filesystem::is_directory(path, error)) {
        return fail(GEMMA4_ERR_MODEL_LOAD, "model_path is not a directory: " + path.string());
    }
    if (!std::filesystem::exists(path / "config.json", error)) {
        return fail(GEMMA4_ERR_MODEL_LOAD, "model_path is missing config.json: " + path.string());
    }
    if (!std::filesystem::exists(path / "tokenizer.json", error)) {
        return fail(GEMMA4_ERR_MODEL_LOAD, "model_path is missing tokenizer.json: " + path.string());
    }
    if (!has_safetensors_file(path)) {
        return fail(
            GEMMA4_ERR_MODEL_LOAD,
            "model_path is missing one or more .safetensors weight shards: " + path.string());
    }

    return GEMMA4_OK;
}

bool read_helper_line(NativeTarget* target, std::string* line) {
    line->clear();
    if (target == nullptr || target->helper_out == nullptr) {
        return false;
    }

    char buffer[4096];
    if (std::fgets(buffer, sizeof(buffer), target->helper_out) == nullptr) {
        return false;
    }
    *line = buffer;
    while (!line->empty() && line->back() != '\n') {
        if (std::fgets(buffer, sizeof(buffer), target->helper_out) == nullptr) {
            break;
        }
        *line += buffer;
    }
    return true;
}

std::string json_string_value(const std::string& line, const char* key) {
    const std::string needle = std::string("\"") + key + "\":\"";
    const size_t start = line.find(needle);
    if (start == std::string::npos) {
        return "";
    }
    const size_t value_start = start + needle.size();
    const size_t value_end = line.find('"', value_start);
    if (value_end == std::string::npos) {
        return "";
    }
    return line.substr(value_start, value_end - value_start);
}

bool json_ok(const std::string& line) {
    return line.find("\"ok\":true") != std::string::npos;
}

bool json_number_slice(const std::string& line, const char* key, std::string* out) {
    const std::string needle = std::string("\"") + key + "\":";
    const size_t start = line.find(needle);
    if (start == std::string::npos) {
        return false;
    }
    size_t value_start = start + needle.size();
    size_t value_end = value_start;
    while (value_end < line.size()) {
        const char c = line[value_end];
        if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
            ++value_end;
        } else {
            break;
        }
    }
    if (value_end == value_start) {
        return false;
    }
    *out = line.substr(value_start, value_end - value_start);
    return true;
}

bool parse_step_response(const std::string& line, Gemma4StepResult* out) {
    if (!json_ok(line)) {
        return false;
    }

    std::string value;
    if (!json_number_slice(line, "greedy_token", &value)) {
        return false;
    }
    out->greedy_token = std::stoi(value);

    if (!json_number_slice(line, "greedy_logit", &value)) {
        return false;
    }
    out->greedy_logit = std::stof(value);

    if (json_number_slice(line, "peak_memory_gb", &value)) {
        out->peak_memory_gb = std::stof(value);
    }

    if (json_number_slice(line, "peak_rss_mb", &value)) {
        out->peak_rss_mb = std::stof(value);
    }

    if (!json_number_slice(line, "sequence_len", &value)) {
        return false;
    }
    out->sequence_len = std::stoull(value);
    out->active_kv_bytes = 0;
    out->native_last_hidden = nullptr;
    return true;
}

void stop_helper(NativeTarget* target) {
    if (target == nullptr) {
        return;
    }

    if (target->helper_in != nullptr) {
        std::fputs("{\"cmd\":\"shutdown\"}\n", target->helper_in);
        std::fflush(target->helper_in);
        if (target->helper_out != nullptr) {
            std::string ignored;
            read_helper_line(target, &ignored);
        }
    }
    if (target->helper_in != nullptr) {
        std::fclose(target->helper_in);
        target->helper_in = nullptr;
    }
    if (target->helper_out != nullptr) {
        std::fclose(target->helper_out);
        target->helper_out = nullptr;
    }
    if (target->helper_pid > 0) {
        int status = 0;
        waitpid(target->helper_pid, &status, 0);
        target->helper_pid = -1;
    }
}

Gemma4Status start_helper(NativeTarget* target, const char* model_path) {
    int to_child[2] = {-1, -1};
    int from_child[2] = {-1, -1};
    if (pipe(to_child) != 0) {
        return fail(GEMMA4_ERR_RUNTIME, errno_message("pipe to helper failed"));
    }
    if (pipe(from_child) != 0) {
        close(to_child[0]);
        close(to_child[1]);
        return fail(GEMMA4_ERR_RUNTIME, errno_message("pipe from helper failed"));
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(to_child[0]);
        close(to_child[1]);
        close(from_child[0]);
        close(from_child[1]);
        return fail(GEMMA4_ERR_RUNTIME, errno_message("fork helper failed"));
    }

    if (pid == 0) {
        dup2(to_child[0], STDIN_FILENO);
        dup2(from_child[1], STDOUT_FILENO);
        close(to_child[0]);
        close(to_child[1]);
        close(from_child[0]);
        close(from_child[1]);

        const char* python = std::getenv("GEMMA4D_MLX_LM_PYTHON");
        if (python == nullptr || python[0] == '\0') {
            python = GEMMA4D_MLX_LM_PYTHON;
        }
        const char* helper = std::getenv("GEMMA4D_MLX_LM_HELPER");
        if (helper == nullptr || helper[0] == '\0') {
            helper = GEMMA4D_MLX_LM_HELPER_PATH;
        }
        execl(python, python, helper, model_path, static_cast<char*>(nullptr));
        std::fprintf(stderr, "failed to exec Gemma4D MLX-LM helper: %s\n", std::strerror(errno));
        _exit(127);
    }

    close(to_child[0]);
    close(from_child[1]);

    target->helper_pid = pid;
    target->helper_in = fdopen(to_child[1], "w");
    target->helper_out = fdopen(from_child[0], "r");
    if (target->helper_in == nullptr || target->helper_out == nullptr) {
        stop_helper(target);
        return fail(GEMMA4_ERR_RUNTIME, errno_message("fdopen helper pipe failed"));
    }

    std::string line;
    if (!read_helper_line(target, &line)) {
        stop_helper(target);
        return fail(GEMMA4_ERR_MODEL_LOAD, "MLX-LM helper exited before reporting readiness");
    }
    if (!json_ok(line)) {
        std::string error = json_string_value(line, "error");
        stop_helper(target);
        return fail(
            GEMMA4_ERR_MODEL_LOAD,
            error.empty() ? "MLX-LM helper failed to load model" : error);
    }

    target->model_loaded = true;
    return ok();
}

std::string tokens_json(const int32_t* tokens, size_t token_count) {
    std::ostringstream json;
    json << '[';
    for (size_t i = 0; i < token_count; ++i) {
        if (i != 0) {
            json << ',';
        }
        json << tokens[i];
    }
    json << ']';
    return json.str();
}

Gemma4Status helper_command(NativeTarget* target, const std::string& command, Gemma4StepResult* out) {
    if (target->helper_in == nullptr || target->helper_out == nullptr) {
        return fail(GEMMA4_ERR_RUNTIME, "MLX-LM helper is not running");
    }
    if (std::fputs(command.c_str(), target->helper_in) == EOF || std::fputc('\n', target->helper_in) == EOF) {
        return fail(GEMMA4_ERR_RUNTIME, errno_message("write to MLX-LM helper failed"));
    }
    if (std::fflush(target->helper_in) != 0) {
        return fail(GEMMA4_ERR_RUNTIME, errno_message("flush to MLX-LM helper failed"));
    }

    std::string line;
    if (!read_helper_line(target, &line)) {
        return fail(GEMMA4_ERR_RUNTIME, "MLX-LM helper exited while waiting for a response");
    }
    if (!json_ok(line)) {
        std::string error = json_string_value(line, "error");
        return fail(
            GEMMA4_ERR_RUNTIME,
            error.empty() ? "MLX-LM helper command failed" : error);
    }
    if (!parse_step_response(line, out)) {
        return fail(GEMMA4_ERR_RUNTIME, "MLX-LM helper returned an invalid step response");
    }
    target->sequence_len = out->sequence_len;
    return ok();
}

} // namespace

struct Gemma4Target : NativeTarget {};
struct Gemma4KvCache : NativeKvCache {};
struct Gemma4KvSnapshot : NativeKvSnapshot {};
struct Gemma4Drafter : NativeDrafter {};
struct Gemma4Adapter : NativeAdapter {};

Gemma4Status gemma4_runtime_version(Gemma4VersionInfo* out) {
    if (out == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_runtime_version requires a non-null out pointer");
    }

    out->abi_version = 4;
    out->backend_name = "gemma4_mlx";
    out->backend_version = kBackendVersion;
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
    target->model_loaded = false;
    target->use_native_graph = false;
    target->sequence_len = 0;
    target->has_prefill_chunk_policy_override = false;
    target->prefill_chunk_policy = Gemma4PrefillChunkPolicy{
        GEMMA4_PREFILL_CHUNK_DISABLED,
        0,
    };
    target->manifest = gemma4d::Gemma4ModelManifest{};
    target->native_model.reset();
    target->helper_pid = -1;
    target->helper_in = nullptr;
    target->helper_out = nullptr;

    if (!config->allow_unsupported_config) {
        Gemma4Status status = validate_strict_model_artifacts(config->model_path);
        if (status != GEMMA4_OK) {
            delete target;
            return status;
        }
        std::string manifest_error;
        if (!gemma4d::load_gemma4_model_manifest(config->model_path, &target->manifest, &manifest_error)) {
            delete target;
            return fail(
                GEMMA4_ERR_UNSUPPORTED_CONFIG,
                "unsupported Gemma 4 model manifest: " + manifest_error);
        }
        if (env_flag_enabled("GEMMA4D_USE_NATIVE_GRAPH")) {
            std::string native_error;
            if (!gemma4d::NativeTextModel::load(
                    config->model_path,
                    target->manifest,
                    &target->native_model,
                    &native_error)) {
                delete target;
                return fail(GEMMA4_ERR_MODEL_LOAD, native_error);
            }
            target->use_native_graph = true;
            target->model_loaded = true;
            *out = target;
            return ok();
        }
        status = start_helper(target, config->model_path);
        if (status != GEMMA4_OK) {
            delete target;
            return status;
        }
    }

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
    stop_helper(target);
    delete target;
    return ok();
}

Gemma4Status gemma4_target_set_prefill_chunk_policy(
    Gemma4Target* target,
    const Gemma4PrefillChunkPolicy* policy) {
    if (target == nullptr) {
        return fail(
            GEMMA4_ERR_INVALID_ARGUMENT,
            "gemma4_target_set_prefill_chunk_policy requires a non-null target");
    }
    if (target->magic != kTargetMagic) {
        return fail(
            GEMMA4_ERR_INVALID_ARGUMENT,
            "gemma4_target_set_prefill_chunk_policy received an invalid target handle");
    }
    if (policy == nullptr) {
        return fail(
            GEMMA4_ERR_INVALID_ARGUMENT,
            "gemma4_target_set_prefill_chunk_policy requires a non-null policy");
    }

    switch (policy->mode) {
        case GEMMA4_PREFILL_CHUNK_DISABLED:
        case GEMMA4_PREFILL_CHUNK_LONG_CONTEXT_256:
            break;
        case GEMMA4_PREFILL_CHUNK_FIXED_TOKENS:
            if (policy->fixed_chunk_tokens == 0) {
                return fail(
                    GEMMA4_ERR_INVALID_ARGUMENT,
                    "fixed prefill chunk policy requires fixed_chunk_tokens > 0");
            }
            break;
        default:
            return fail(
                GEMMA4_ERR_INVALID_ARGUMENT,
                "unknown prefill chunk policy mode");
    }

    if (target->model_loaded && !target->use_native_graph) {
        return fail(
            GEMMA4_ERR_UNSUPPORTED_CONFIG,
            "prefill chunk policy can only be applied to native graph targets");
    }

    target->has_prefill_chunk_policy_override = true;
    target->prefill_chunk_policy = *policy;
    if (target->use_native_graph && target->native_model != nullptr) {
        target->native_model->set_prefill_chunk_policy(*policy);
    }
    return ok();
}

Gemma4Status gemma4_load_adapter(
    Gemma4Target* target,
    const Gemma4AdapterLoadConfig* config,
    Gemma4Adapter** out,
    Gemma4AdapterInfo* info) {
    if (out == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_adapter requires a non-null out pointer");
    }
    *out = nullptr;
    fill_adapter_info(nullptr, 0, false, info);

    if (target == nullptr || target->magic != kTargetMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_adapter requires a valid target handle");
    }
    if (config == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_adapter requires a non-null config");
    }
    if (is_empty(config->adapter_path)) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_adapter requires a non-empty adapter_path");
    }
    if (is_empty(config->adapter_id)) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_adapter requires a non-empty adapter_id");
    }
    if (is_empty(config->adapter_weight_hash)) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_adapter requires a non-empty adapter_weight_hash");
    }
    if (config->rank == 0) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_adapter requires rank > 0");
    }
    if (!target->use_native_graph || target->native_model == nullptr) {
        return fail(
            GEMMA4_ERR_UNSUPPORTED_CONFIG,
            "gemma4_load_adapter requires a loaded native target graph");
    }

    const std::vector<std::string> target_modules = parse_csv_list(config->target_modules_csv);
    if (target_modules.empty()) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_adapter requires target_modules_csv");
    }

    std::shared_ptr<const gemma4d::NativeLoraAdapter> native_adapter;
    uint64_t load_latency_us = 0;
    std::string native_error;
    if (!gemma4d::NativeLoraAdapter::load_peft(
            config->adapter_path,
            config->adapter_id,
            config->adapter_weight_hash,
            config->rank,
            config->alpha,
            target_modules,
            *target->native_model,
            &native_adapter,
            &load_latency_us,
            &native_error)) {
        return fail(GEMMA4_ERR_ADAPTER, native_error);
    }

    Gemma4Adapter* adapter = new (std::nothrow) Gemma4Adapter{};
    if (adapter == nullptr) {
        return fail(GEMMA4_ERR_RUNTIME, "gemma4_load_adapter could not allocate adapter handle");
    }
    adapter->magic = kAdapterMagic;
    adapter->native_adapter = std::move(native_adapter);
    adapter->load_latency_us = load_latency_us;
    fill_adapter_info(adapter->native_adapter, adapter->load_latency_us, false, info);
    *out = adapter;
    return ok();
}

Gemma4Status gemma4_free_adapter(Gemma4Adapter* adapter) {
    if (adapter == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_free_adapter requires a non-null adapter");
    }
    if (adapter->magic != kAdapterMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_free_adapter received an invalid adapter handle");
    }

    adapter->magic = 0;
    delete adapter;
    return ok();
}

Gemma4Status gemma4_set_adapter(Gemma4Target* target, Gemma4Adapter* adapter, Gemma4AdapterInfo* info) {
    fill_adapter_info(nullptr, 0, false, info);
    if (target == nullptr || target->magic != kTargetMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_set_adapter requires a valid target handle");
    }
    if (adapter == nullptr || adapter->magic != kAdapterMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_set_adapter requires a valid adapter handle");
    }
    if (!target->use_native_graph || target->native_model == nullptr) {
        return fail(
            GEMMA4_ERR_UNSUPPORTED_CONFIG,
            "gemma4_set_adapter requires a loaded native target graph");
    }

    std::string native_error;
    if (!target->native_model->set_adapter(adapter->native_adapter, &native_error)) {
        return fail(GEMMA4_ERR_ADAPTER, native_error);
    }
    fill_adapter_info(adapter->native_adapter, adapter->load_latency_us, true, info);
    return ok();
}

Gemma4Status gemma4_clear_adapter(Gemma4Target* target, Gemma4AdapterInfo* info) {
    fill_adapter_info(nullptr, 0, false, info);
    if (target == nullptr || target->magic != kTargetMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_clear_adapter requires a valid target handle");
    }
    if (!target->use_native_graph || target->native_model == nullptr) {
        return fail(
            GEMMA4_ERR_UNSUPPORTED_CONFIG,
            "gemma4_clear_adapter requires a loaded native target graph");
    }
    target->native_model->clear_adapter();
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
    cache->last_hidden.reset();
    cache->has_last_step = false;
    clear_step_result(&cache->last_step);
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

    cache->native_tokens.clear();
    cache->native_kv_state.reset();
    cache->last_hidden.reset();
    cache->has_last_step = false;
    clear_step_result(&cache->last_step);
    cache->pending_mtp_draft_scores.clear();
    return ok();
}

Gemma4Status gemma4_kv_last_step(const Gemma4KvCache* cache, Gemma4StepResult* out) {
    clear_step_result(out);

    if (cache == nullptr || cache->magic != kKvCacheMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_last_step requires a valid cache handle");
    }
    if (out == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_last_step requires a non-null step result");
    }
    if (!cache->has_last_step) {
        return fail(GEMMA4_ERR_CACHE, "gemma4_kv_last_step requires a cache with a native prefill/decode result");
    }

    *out = cache->last_step;
    out->native_last_hidden = cache->last_hidden.get();
    return ok();
}

Gemma4Status gemma4_kv_snapshot_export(const Gemma4KvCache* cache, Gemma4KvSnapshot** out) {
    if (out == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_export requires a non-null out pointer");
    }
    *out = nullptr;

    if (cache == nullptr || cache->magic != kKvCacheMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_export requires a valid cache handle");
    }
    if (cache->native_tokens.empty() || cache->native_kv_state == nullptr || !cache->has_last_step) {
        return fail(
            GEMMA4_ERR_CACHE,
            "gemma4_kv_snapshot_export requires a cache populated by the native incremental path");
    }

    Gemma4KvSnapshot* snapshot = new (std::nothrow) Gemma4KvSnapshot{};
    if (snapshot == nullptr) {
        return fail(GEMMA4_ERR_RUNTIME, "gemma4_kv_snapshot_export could not allocate snapshot handle");
    }

    snapshot->magic = kKvSnapshotMagic;
    snapshot->policy = cache->policy;
    snapshot->native_tokens = cache->native_tokens;
    snapshot->native_kv_state = cache->native_kv_state->clone();
    snapshot->last_hidden = cache->last_hidden == nullptr ? nullptr : cache->last_hidden->clone();
    snapshot->has_last_step = cache->has_last_step;
    snapshot->last_step = cache->last_step;
    snapshot->last_step.native_last_hidden = snapshot->last_hidden.get();
    if (snapshot->native_kv_state == nullptr) {
        delete snapshot;
        return fail(GEMMA4_ERR_CACHE, "gemma4_kv_snapshot_export could not clone native KV state");
    }

    *out = snapshot;
    return ok();
}

Gemma4Status gemma4_kv_snapshot_import(Gemma4KvCache* cache, const Gemma4KvSnapshot* snapshot) {
    if (cache == nullptr || cache->magic != kKvCacheMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_import requires a valid cache handle");
    }
    if (snapshot == nullptr || snapshot->magic != kKvSnapshotMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_import requires a valid snapshot handle");
    }
    if (!same_kv_policy(cache->policy, snapshot->policy)) {
        return fail(GEMMA4_ERR_CACHE, "gemma4_kv_snapshot_import rejected incompatible KV policy");
    }
    if (snapshot->native_tokens.empty() || snapshot->native_kv_state == nullptr || !snapshot->has_last_step) {
        return fail(GEMMA4_ERR_CACHE, "gemma4_kv_snapshot_import requires a populated native snapshot");
    }

    std::unique_ptr<gemma4d::NativeKvState> cloned_kv = snapshot->native_kv_state->clone();
    if (cloned_kv == nullptr) {
        return fail(GEMMA4_ERR_CACHE, "gemma4_kv_snapshot_import could not clone native KV state");
    }
    std::unique_ptr<gemma4d::NativeHiddenState> cloned_hidden =
        snapshot->last_hidden == nullptr ? nullptr : snapshot->last_hidden->clone();

    cache->native_tokens = snapshot->native_tokens;
    cache->native_kv_state = std::move(cloned_kv);
    cache->last_hidden = std::move(cloned_hidden);
    cache->has_last_step = snapshot->has_last_step;
    cache->last_step = snapshot->last_step;
    cache->last_step.native_last_hidden = cache->last_hidden.get();
    cache->pending_mtp_draft_scores.clear();
    return ok();
}

Gemma4Status gemma4_kv_snapshot_info(const Gemma4KvSnapshot* snapshot, Gemma4KvSnapshotInfo* out) {
    if (out == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_info requires a non-null out pointer");
    }
    std::memset(out, 0, sizeof(Gemma4KvSnapshotInfo));

    if (snapshot == nullptr || snapshot->magic != kKvSnapshotMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_info requires a valid snapshot handle");
    }

    out->sequence_len = snapshot->native_kv_state == nullptr ? 0 : snapshot->native_kv_state->sequence_len();
    out->active_kv_bytes = snapshot->native_kv_state == nullptr ? 0 : snapshot->native_kv_state->active_bytes();
    out->token_count = snapshot->native_tokens.size();
    out->has_last_step = snapshot->has_last_step;
    return ok();
}

Gemma4Status gemma4_kv_snapshot_save(const Gemma4KvSnapshot* snapshot, const char* payload_path) {
    if (snapshot == nullptr || snapshot->magic != kKvSnapshotMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_save requires a valid snapshot handle");
    }
    if (is_empty(payload_path)) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_save requires a non-empty payload path");
    }
    if (snapshot->native_kv_state == nullptr || snapshot->native_tokens.empty() || !snapshot->has_last_step) {
        return fail(GEMMA4_ERR_CACHE, "gemma4_kv_snapshot_save requires a populated native snapshot");
    }

    std::string native_error;
    if (!snapshot->native_kv_state->save_safetensors(
            payload_path,
            snapshot->last_hidden.get(),
            snapshot_metadata(snapshot),
            &native_error)) {
        return fail(GEMMA4_ERR_RUNTIME, native_error);
    }
    return ok();
}

Gemma4Status gemma4_kv_snapshot_save_mtp_parity(
    const Gemma4KvSnapshot* snapshot,
    Gemma4Target* target,
    const int32_t* token_ids,
    size_t token_count,
    const char* payload_path) {
    if (snapshot == nullptr || snapshot->magic != kKvSnapshotMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_save_mtp_parity requires a valid snapshot handle");
    }
    if (target == nullptr || target->magic != kTargetMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_save_mtp_parity requires a valid target handle");
    }
    if (is_empty(payload_path)) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_save_mtp_parity requires a non-empty payload path");
    }
    if (token_ids == nullptr || token_count == 0) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_save_mtp_parity requires token ids");
    }
    if (snapshot->native_kv_state == nullptr || snapshot->native_tokens.empty() || !snapshot->has_last_step) {
        return fail(GEMMA4_ERR_CACHE, "gemma4_kv_snapshot_save_mtp_parity requires a populated native snapshot");
    }
    if (!target->use_native_graph || target->native_model == nullptr) {
        return fail(
            GEMMA4_ERR_UNSUPPORTED_CONFIG,
            "gemma4_kv_snapshot_save_mtp_parity requires a loaded native target graph");
    }

    auto metadata = snapshot_metadata(snapshot);
    metadata["diagnostic"] = "xr54_mtp_drafter_pytorch_parity";
    metadata["diagnostic.target_token_count"] = std::to_string(token_count);
    std::vector<int32_t> token_id_list(token_ids, token_ids + token_count);

    std::string native_error;
    if (!snapshot->native_kv_state->save_safetensors(
            payload_path,
            snapshot->last_hidden.get(),
            metadata,
            &native_error,
            target->native_model.get(),
            &token_id_list)) {
        return fail(GEMMA4_ERR_RUNTIME, native_error);
    }
    return ok();
}

Gemma4Status gemma4_kv_snapshot_save_compressed(
    const Gemma4KvSnapshot* snapshot,
    const char* payload_path,
    Gemma4KvMode mode,
    bool compress_global_layers,
    bool compress_sliding_layers) {
    if (snapshot == nullptr || snapshot->magic != kKvSnapshotMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_save_compressed requires a valid snapshot handle");
    }
    if (is_empty(payload_path)) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_save_compressed requires a non-empty payload path");
    }
    if (snapshot->native_kv_state == nullptr || snapshot->native_tokens.empty() || !snapshot->has_last_step) {
        return fail(GEMMA4_ERR_CACHE, "gemma4_kv_snapshot_save_compressed requires a populated native snapshot");
    }
    if (mode != GEMMA4_KV_BF16 && mode != GEMMA4_KV_MLX_AFFINE_Q8 && mode != GEMMA4_KV_MLX_AFFINE_Q4) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_save_compressed supports only BF16, MLX affine q8, or MLX affine q4");
    }

    std::unordered_map<std::string, std::string> metadata = snapshot_metadata(snapshot);
    metadata["policy.ssd_prefix_mode"] = std::to_string(static_cast<int>(mode));
    metadata["policy.compress_global_layers"] = compress_global_layers ? "true" : "false";
    metadata["policy.compress_sliding_layers"] = compress_sliding_layers ? "true" : "false";
    metadata["policy.allow_active_compressed_decode"] = "false";

    std::string native_error;
    if (!snapshot->native_kv_state->save_compressed_safetensors(
            payload_path,
            snapshot->last_hidden.get(),
            metadata,
            mode,
            compress_global_layers,
            compress_sliding_layers,
            &native_error)) {
        return fail(GEMMA4_ERR_RUNTIME, native_error);
    }
    return ok();
}

Gemma4Status gemma4_kv_snapshot_load(const char* payload_path, Gemma4KvSnapshot** out) {
    if (out == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_load requires a non-null out pointer");
    }
    *out = nullptr;

    if (is_empty(payload_path)) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_load requires a non-empty payload path");
    }

    Gemma4KvSnapshot* snapshot = new (std::nothrow) Gemma4KvSnapshot{};
    if (snapshot == nullptr) {
        return fail(GEMMA4_ERR_RUNTIME, "gemma4_kv_snapshot_load could not allocate snapshot handle");
    }

    std::unordered_map<std::string, std::string> metadata;
    std::string native_error;
    if (!gemma4d::NativeKvState::load_safetensors(
            payload_path,
            &snapshot->native_kv_state,
            &snapshot->last_hidden,
            &metadata,
            &native_error)) {
        delete snapshot;
        return fail(GEMMA4_ERR_RUNTIME, native_error);
    }
    try {
        snapshot->magic = kKvSnapshotMagic;
        apply_snapshot_metadata(metadata, snapshot);
        if (snapshot->native_kv_state == nullptr || snapshot->native_tokens.empty() || !snapshot->has_last_step) {
            delete snapshot;
            return fail(GEMMA4_ERR_CACHE, "gemma4_kv_snapshot_load read an incomplete native snapshot");
        }
        snapshot->last_step.native_last_hidden = snapshot->last_hidden.get();
    } catch (const std::exception& ex) {
        delete snapshot;
        return fail(GEMMA4_ERR_CACHE, std::string("gemma4_kv_snapshot_load rejected metadata: ") + ex.what());
    }

    *out = snapshot;
    return ok();
}

Gemma4Status gemma4_kv_snapshot_free(Gemma4KvSnapshot* snapshot) {
    if (snapshot == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_free requires a non-null snapshot");
    }
    if (snapshot->magic != kKvSnapshotMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_kv_snapshot_free received an invalid snapshot handle");
    }

    snapshot->magic = 0;
    delete snapshot;
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
    if (token_count == 0) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_prefill requires at least one token");
    }
    if (!target->model_loaded) {
        return fail(
            GEMMA4_ERR_UNSUPPORTED_CONFIG,
            "gemma4_prefill requires a loaded Gemma 4 target model; smoke handles do not execute");
    }
    if (target->use_native_graph) {
        if (target->native_model == nullptr) {
            return fail(GEMMA4_ERR_RUNTIME, "native Gemma 4 model state is missing");
        }
        cache->pending_mtp_draft_scores.clear();
        cache->native_tokens.assign(tokens, tokens + token_count);
        std::string native_error;
        if (!target->native_model->prefill_incremental(
                cache->native_tokens,
                out,
                &native_error,
                &cache->native_kv_state,
                &cache->last_hidden)) {
            return fail(GEMMA4_ERR_RUNTIME, native_error);
        }
        out->native_last_hidden = cache->last_hidden.get();
        remember_last_step(cache, out);
        target->sequence_len = out->sequence_len;
        return ok();
    }

    std::string command = "{\"cmd\":\"prefill\",\"tokens\":" + tokens_json(tokens, token_count) + "}";
    return helper_command(target, command, out);
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
    if (!target->model_loaded) {
        return fail(
            GEMMA4_ERR_UNSUPPORTED_CONFIG,
            "gemma4_decode_one requires a loaded Gemma 4 target model; smoke handles do not execute");
    }
    if (target->use_native_graph) {
        if (target->native_model == nullptr) {
            return fail(GEMMA4_ERR_RUNTIME, "native Gemma 4 model state is missing");
        }
        if (cache->native_kv_state == nullptr) {
            return fail(GEMMA4_ERR_RUNTIME, "native Gemma 4 incremental decode requires a prior prefill");
        }
        cache->pending_mtp_draft_scores.clear();
        cache->native_tokens.push_back(token);
        std::string native_error;
        if (!target->native_model->decode_incremental(
                token,
                cache->native_kv_state.get(),
                out,
                &native_error,
                &cache->last_hidden)) {
            return fail(GEMMA4_ERR_RUNTIME, native_error);
        }
        out->native_last_hidden = cache->last_hidden.get();
        remember_last_step(cache, out);
        target->sequence_len = out->sequence_len;
        return ok();
    }

    std::ostringstream command;
    command << "{\"cmd\":\"decode_one\",\"token\":" << token << "}";
    return helper_command(target, command.str(), out);
}

Gemma4Status gemma4_decode_block(
    Gemma4Target* target,
    Gemma4KvCache* cache,
    const int32_t* tokens,
    size_t token_count,
    int32_t* out_greedy_tokens,
    float* out_greedy_logits,
    size_t* inout_count,
    Gemma4StepResult* out) {
    clear_step_result(out);

    if (target == nullptr || target->magic != kTargetMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_decode_block requires a valid target handle");
    }
    if (cache == nullptr || cache->magic != kKvCacheMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_decode_block requires a valid cache handle");
    }
    if (out == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_decode_block requires a non-null step result");
    }
    if (tokens == nullptr || token_count == 0) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_decode_block requires at least one token");
    }
    if (token_count > GEMMA4_MTP_MAX_DRAFT_TOKENS) {
        std::ostringstream message;
        message << "gemma4_decode_block supports token_count <= " << GEMMA4_MTP_MAX_DRAFT_TOKENS;
        return fail(GEMMA4_ERR_UNSUPPORTED_CONFIG, message.str());
    }
    if (out_greedy_tokens == nullptr || out_greedy_logits == nullptr || inout_count == nullptr) {
        return fail(
            GEMMA4_ERR_INVALID_ARGUMENT,
            "gemma4_decode_block requires greedy token/logit output buffers and count");
    }
    if (*inout_count < token_count) {
        *inout_count = token_count;
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_decode_block output buffers are too small");
    }
    *inout_count = 0;
    if (!target->model_loaded) {
        return fail(
            GEMMA4_ERR_UNSUPPORTED_CONFIG,
            "gemma4_decode_block requires a loaded Gemma 4 target model; smoke handles do not execute");
    }
    if (!target->use_native_graph) {
        return fail(GEMMA4_ERR_UNSUPPORTED_CONFIG, "gemma4_decode_block requires the native target graph");
    }
    if (target->native_model == nullptr) {
        return fail(GEMMA4_ERR_RUNTIME, "native Gemma 4 model state is missing");
    }
    if (cache->native_kv_state == nullptr) {
        return fail(GEMMA4_ERR_RUNTIME, "native Gemma 4 block decode requires a prior prefill");
    }

    cache->pending_mtp_draft_scores.clear();
    std::string native_error;
    std::vector<int32_t> greedy_tokens;
    std::vector<float> greedy_logits;
    if (!target->native_model->decode_incremental_block(
            tokens,
            token_count,
            cache->native_kv_state.get(),
            out,
            &greedy_tokens,
            &greedy_logits,
            &native_error,
            &cache->last_hidden)) {
        return fail(GEMMA4_ERR_RUNTIME, native_error);
    }
    if (greedy_tokens.size() < token_count || greedy_logits.size() < token_count) {
        return fail(GEMMA4_ERR_RUNTIME, "native Gemma 4 block decode returned incomplete logits");
    }

    for (size_t index = 0; index < token_count; ++index) {
        cache->native_tokens.push_back(tokens[index]);
        out_greedy_tokens[index] = greedy_tokens[index];
        out_greedy_logits[index] = greedy_logits[index];
    }
    *inout_count = token_count;
    out->native_last_hidden = cache->last_hidden.get();
    remember_last_step(cache, out);
    target->sequence_len = out->sequence_len;
    return ok();
}

Gemma4Status gemma4_load_drafter(
    const Gemma4LoadConfig* config,
    Gemma4Target* target,
    Gemma4Drafter** out) {
    if (out == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_drafter requires a non-null out pointer");
    }
    *out = nullptr;

    if (config == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_drafter requires a non-null config");
    }
    if (target == nullptr || target->magic != kTargetMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_drafter requires a valid target handle");
    }
    if (is_empty(config->model_path)) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_load_drafter requires a non-empty model_path");
    }
    if (target->use_native_graph && target->native_model != nullptr && target->native_model->has_adapter()) {
        return fail(
            GEMMA4_ERR_ADAPTER,
            "gemma4_load_drafter is disabled while a standard LoRA adapter is active");
    }

    Gemma4Drafter* drafter = new (std::nothrow) Gemma4Drafter{};
    if (drafter == nullptr) {
        return fail(GEMMA4_ERR_RUNTIME, "gemma4_load_drafter could not allocate drafter handle");
    }

    drafter->magic = kDrafterMagic;
    drafter->model_loaded = false;
    drafter->model_path = config->model_path;
    drafter->manifest = gemma4d::Gemma4ModelManifest{};
    drafter->target_native_model = target->use_native_graph ? target->native_model.get() : nullptr;
    drafter->native_model.reset();

    if (!config->allow_unsupported_config) {
        Gemma4Status status = validate_strict_model_artifacts(config->model_path);
        if (status != GEMMA4_OK) {
            delete drafter;
            return status;
        }
        std::string manifest_error;
        if (!gemma4d::load_gemma4_mtp_assistant_manifest(
                config->model_path, &drafter->manifest, &manifest_error)) {
            delete drafter;
            return fail(
                GEMMA4_ERR_UNSUPPORTED_CONFIG,
                "unsupported Gemma 4 drafter manifest: " + manifest_error);
        }
        if (target->use_native_graph) {
            std::string native_error;
            if (!gemma4d::NativeMtpAssistantModel::load(
                    config->model_path,
                    drafter->manifest,
                    &drafter->native_model,
                    &native_error)) {
                delete drafter;
                return fail(GEMMA4_ERR_MODEL_LOAD, native_error);
            }
        }
        drafter->model_loaded = true;
    }

    *out = drafter;
    return ok();
}

Gemma4Status gemma4_free_drafter(Gemma4Drafter* drafter) {
    if (drafter == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_free_drafter requires a non-null drafter");
    }
    if (drafter->magic != kDrafterMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_free_drafter received an invalid drafter handle");
    }

    drafter->magic = 0;
    delete drafter;
    return ok();
}

Gemma4Status gemma4_mtp_draft_block(
    Gemma4Drafter* drafter,
    Gemma4KvCache* cache,
    uint32_t block_size,
    int32_t* out_tokens,
    float* out_logits,
    float* out_logit_margins,
    size_t* inout_count) {
    if (drafter == nullptr || drafter->magic != kDrafterMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_mtp_draft_block requires a valid drafter handle");
    }
    if (cache == nullptr || cache->magic != kKvCacheMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_mtp_draft_block requires a valid cache handle");
    }
    const bool capture_scores = mtp_real_margins_enabled();
    if (out_tokens == nullptr || inout_count == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_mtp_draft_block requires token output buffer and count");
    }
    if (capture_scores && (out_logits == nullptr || out_logit_margins == nullptr)) {
        return fail(
            GEMMA4_ERR_INVALID_ARGUMENT,
            "gemma4_mtp_draft_block requires score output buffers when real margins are enabled");
    }
    if (block_size == 0) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_mtp_draft_block requires block_size > 0");
    }
    if (*inout_count < block_size) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_mtp_draft_block output buffer is smaller than block_size");
    }
    cache->pending_mtp_draft_scores.clear();
    if (!drafter->model_loaded) {
        *inout_count = 0;
        return fail(
            GEMMA4_ERR_UNSUPPORTED_CONFIG,
            "gemma4_mtp_draft_block requires a loaded Gemma 4 MTP assistant; smoke handles do not draft");
    }
    if (cache->last_hidden == nullptr) {
        *inout_count = 0;
        return fail(
            GEMMA4_ERR_UNSUPPORTED_CONFIG,
            "gemma4_mtp_draft_block requires materialized last target hidden/shared views; call gemma4_prefill or gemma4_decode_one first on the native target graph");
    }
    if (!cache->last_hidden->has_shared_kv()) {
        *inout_count = 0;
        return fail(
            GEMMA4_ERR_UNSUPPORTED_CONFIG,
            "gemma4_mtp_draft_block requires both full-attention and sliding-attention shared KV views");
    }
    if (drafter->native_model == nullptr || drafter->target_native_model == nullptr) {
        *inout_count = 0;
        return fail(
            GEMMA4_ERR_UNSUPPORTED_CONFIG,
            "gemma4_mtp_draft_block requires a native target graph and loaded native MTP assistant tensors");
    }
    if (drafter->target_native_model->has_adapter()) {
        *inout_count = 0;
        return fail(
            GEMMA4_ERR_ADAPTER,
            "gemma4_mtp_draft_block is disabled while a standard LoRA adapter is active");
    }

    std::string native_error;
    const bool lazy_second_draft =
        env_flag_enabled("GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT") && cache->has_last_step;
    if (!drafter->native_model->draft_block(
            *drafter->target_native_model,
            *cache->last_hidden,
            cache->native_tokens,
            block_size,
            out_tokens,
            out_logits,
            out_logit_margins,
            inout_count,
            &native_error,
            lazy_second_draft,
            cache->last_step.greedy_token)) {
        *inout_count = 0;
        cache->pending_mtp_draft_scores.clear();
        return fail(GEMMA4_ERR_RUNTIME, native_error);
    }
    cache->pending_mtp_draft_scores.clear();
    if (out_logits != nullptr && out_logit_margins != nullptr) {
        cache->pending_mtp_draft_scores.reserve(*inout_count);
        for (size_t index = 0; index < *inout_count; ++index) {
            cache->pending_mtp_draft_scores.push_back(NativeKvCache::PendingMtpDraftScore{
                out_tokens[index],
                out_logits[index],
                out_logit_margins[index],
            });
        }
    }
    return ok();
}

Gemma4Status verify_tokens_impl(
    Gemma4Target* target,
    Gemma4KvCache* cache,
    const int32_t* draft_tokens,
    size_t draft_count,
    size_t terminal_commit_count,
    Gemma4StepResult* out) {
    clear_step_result(out);

    if (target == nullptr || target->magic != kTargetMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_verify_tokens requires a valid target handle");
    }
    if (cache == nullptr || cache->magic != kKvCacheMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_verify_tokens requires a valid cache handle");
    }
    if (out == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_verify_tokens requires a non-null step result");
    }
    if (draft_count > 0 && draft_tokens == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_verify_tokens requires draft tokens when draft_count > 0");
    }
    if (draft_count == 0) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_verify_tokens requires at least one draft token");
    }
    if (terminal_commit_count > draft_count) {
        return fail(
            GEMMA4_ERR_INVALID_ARGUMENT,
            "gemma4_verify_tokens terminal commit count cannot exceed draft_count");
    }
    if (!target->model_loaded) {
        return fail(
            GEMMA4_ERR_UNSUPPORTED_CONFIG,
            "gemma4_verify_tokens requires a loaded Gemma 4 target model; smoke handles do not execute");
    }
    if (target->use_native_graph) {
        if (target->native_model == nullptr) {
            return fail(GEMMA4_ERR_RUNTIME, "native Gemma 4 model state is missing");
        }
        if (target->native_model->has_adapter()) {
            return fail(
                GEMMA4_ERR_ADAPTER,
                "gemma4_verify_tokens is disabled while a standard LoRA adapter is active");
        }
        if (cache->native_tokens.empty()) {
            return fail(
                GEMMA4_ERR_UNSUPPORTED_CONFIG,
                "gemma4_verify_tokens requires a prefilled native target cache");
        }
        if (cache->native_kv_state == nullptr || !cache->has_last_step) {
            return fail(
                GEMMA4_ERR_CACHE,
                "gemma4_verify_tokens requires a native incremental KV state and last-step prediction");
        }
        if (draft_count > GEMMA4_MTP_MAX_DRAFT_TOKENS ||
            draft_count + 1 > GEMMA4_MTP_TRACE_MAX_POSITIONS) {
            std::ostringstream message;
            message << "native MTP verify supports draft_count <= " << GEMMA4_MTP_MAX_DRAFT_TOKENS;
            return fail(GEMMA4_ERR_UNSUPPORTED_CONFIG, message.str());
        }

        std::vector<NativeKvCache::PendingMtpDraftScore> draft_scores =
            std::move(cache->pending_mtp_draft_scores);
        cache->pending_mtp_draft_scores.clear();
        auto draft_score_for =
            [&](size_t index, int32_t token) -> const NativeKvCache::PendingMtpDraftScore* {
            if (index >= draft_scores.size() || draft_scores[index].token != token) {
                return nullptr;
            }
            return &draft_scores[index];
        };

        std::string native_error;
        double verify_stage_ms = 0.0;
        double verify_forward_ms = 0.0;
        double verify_repair_ms = 0.0;
        double repair_clone_ms = 0.0;
        double repair_forward_ms = 0.0;
        double repair_fallback_ms = 0.0;
        auto attach_verify_timings = [&](Gemma4StepResult* step) {
            if (step == nullptr) {
                return;
            }
            step->verify_stage_ms = verify_stage_ms;
            step->verify_forward_ms = verify_forward_ms;
            step->verify_repair_ms = verify_repair_ms;
            step->repair_clone_ms = repair_clone_ms;
            step->repair_forward_ms = repair_forward_ms;
            step->repair_fallback_ms = repair_fallback_ms;
        };
        // Experimental XR30 prototype: when the first draft token already
        // disagrees with the cached target greedy token, no speculative token
        // can be accepted. Commit the known fallback directly and avoid the
        // staged KV clone used by rollback-capable verifier paths. This is a
        // success-path measurement only; a late decode failure can invalidate
        // the live cache, so keep it default-off.
        if (terminal_commit_count == 0 &&
            env_flag_enabled("GEMMA4D_EXPERIMENTAL_MTP_DIRECT_FIRST_REJECT") &&
            draft_tokens[0] != cache->last_step.greedy_token) {
            const uint64_t context_sequence_len = cache->native_tokens.size();
            Gemma4MtpTraceInfo trace{};
            initialize_mtp_trace(&trace, context_sequence_len);
            record_mtp_target_step(&trace, 0, context_sequence_len, cache->last_step);
            record_mtp_draft_score(&trace, 0, draft_tokens[0], draft_score_for(0, draft_tokens[0]));

            const int32_t fallback_token = cache->last_step.greedy_token;
            Gemma4StepResult fallback_step{};
            std::unique_ptr<gemma4d::NativeHiddenState> fallback_hidden;
            const auto forward_started = std::chrono::steady_clock::now();
            if (!target->native_model->decode_incremental(
                    fallback_token,
                    cache->native_kv_state.get(),
                    &fallback_step,
                    &native_error,
                    &fallback_hidden)) {
                return fail(GEMMA4_ERR_RUNTIME, native_error);
            }
            verify_forward_ms += elapsed_ms(forward_started);

            const size_t lookahead_index = 1;
            fallback_step.peak_memory_gb =
                std::max(fallback_step.peak_memory_gb, cache->last_step.peak_memory_gb);
            fallback_step.active_kv_bytes =
                std::max(fallback_step.active_kv_bytes, cache->native_kv_state->active_bytes());
            record_mtp_target_step(&trace, lookahead_index, context_sequence_len, fallback_step);
            record_mtp_hidden_shape(&trace, fallback_hidden.get());

            fallback_step.accepted_draft_count = 0;
            fallback_step.committed_count = 1;
            fallback_step.committed_tokens[0] = fallback_token;
            fallback_step.mtp_trace = trace;
            attach_verify_timings(&fallback_step);

            cache->native_tokens.push_back(fallback_token);
            cache->last_hidden = std::move(fallback_hidden);
            fallback_step.native_last_hidden = cache->last_hidden.get();
            *out = fallback_step;
            out->native_last_hidden = cache->last_hidden.get();
            remember_last_step(cache, out);
            target->sequence_len = out->sequence_len;
            return ok();
        }

        // Experimental XR22 prototype: use one block target decode when the
        // first draft is already known accepted, and keep an exact prefix KV for
        // fallback if the next draft is rejected. For N>2, later partial
        // accepts use serial repair rather than truncating post-block KV, which
        // is not exact for sliding-window layers near the window boundary.
        if (terminal_commit_count == 0 &&
            env_flag_enabled("GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK") && draft_count >= 2 &&
            draft_tokens[0] == cache->last_step.greedy_token) {
            const bool serial_state_repair =
                env_flag_enabled("GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_SERIAL_STATE_REPAIR");
            const bool partial_only_repair_full_accept =
                env_flag_enabled("GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_ONLY_REPAIR");
            const bool partial_reject_serial_repair =
                env_flag_enabled("GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_REJECT_REPAIR");
            const bool state_only_serial_repair =
                (serial_state_repair || partial_reject_serial_repair) &&
                env_flag_enabled("GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_STATE_ONLY_REPAIR");
            const auto stage_started = std::chrono::steady_clock::now();
            std::unique_ptr<gemma4d::NativeKvState> block_kv = cache->native_kv_state->clone();
            if (block_kv == nullptr) {
                return fail(GEMMA4_ERR_RUNTIME, "native MTP block-prefix verify failed to clone target KV state");
            }
            std::unique_ptr<gemma4d::NativeKvState> prefix_kv(new gemma4d::NativeKvState());
            verify_stage_ms += elapsed_ms(stage_started);

            Gemma4StepResult block_step{};
            std::vector<int32_t> block_greedy_tokens;
            std::vector<float> block_greedy_logits;
            std::vector<gemma4d::NativeTopKEntries> block_target_top_k;
            std::unique_ptr<gemma4d::NativeHiddenState> block_hidden;
            size_t retroactive_prefix_count = 0;
            const auto forward_started = std::chrono::steady_clock::now();
            if (!target->native_model->decode_incremental_block_with_retroactive_prefix(
                    draft_tokens,
                    draft_count,
                    block_kv.get(),
                    prefix_kv.get(),
                    &retroactive_prefix_count,
                    &block_step,
                    &block_greedy_tokens,
                    &block_greedy_logits,
                    &native_error,
                    &block_hidden,
                    &block_target_top_k)) {
                return fail(GEMMA4_ERR_RUNTIME, native_error);
            }
            verify_forward_ms += elapsed_ms(forward_started);
            if (block_greedy_tokens.size() < draft_count || block_greedy_logits.size() < draft_count ||
                block_target_top_k.size() < draft_count) {
                return fail(GEMMA4_ERR_RUNTIME, "native MTP block-prefix verify returned incomplete target logits");
            }

            size_t accepted_prefix_count = 1;
            for (size_t index = 1; index < draft_count; ++index) {
                if (draft_tokens[index] != block_greedy_tokens[index - 1]) {
                    break;
                }
                ++accepted_prefix_count;
            }
            const bool full_block_accepted = accepted_prefix_count == draft_count;
            if (retroactive_prefix_count != accepted_prefix_count) {
                return fail(
                    GEMMA4_ERR_RUNTIME,
                    "native MTP block-prefix retroactive prefix count disagreed with verifier comparison");
            }

            const uint64_t context_sequence_len = cache->native_tokens.size();
            Gemma4MtpTraceInfo trace{};
            initialize_mtp_trace(&trace, context_sequence_len);
            record_mtp_target_step(&trace, 0, context_sequence_len, cache->last_step);
            record_mtp_draft_score(&trace, 0, draft_tokens[0], draft_score_for(0, draft_tokens[0]));

            const size_t trace_draft_count = full_block_accepted
                ? draft_count
                : std::min(draft_count, accepted_prefix_count + 1);
            for (size_t index = 1; index < trace_draft_count; ++index) {
                Gemma4StepResult target_step{};
                target_step.greedy_token = block_greedy_tokens[index - 1];
                target_step.greedy_logit = block_greedy_logits[index - 1];
                target_step.sequence_len = context_sequence_len + index;
                target_step.active_kv_bytes = index == 1 ? prefix_kv->active_bytes() : block_step.active_kv_bytes;
                target_step.peak_memory_gb =
                    std::max(block_step.peak_memory_gb, cache->last_step.peak_memory_gb);
                record_mtp_target_step(&trace, index, context_sequence_len, target_step, &block_target_top_k[index - 1]);
                record_mtp_draft_score(&trace, index, draft_tokens[index], draft_score_for(index, draft_tokens[index]));
            }

            auto commit_serial_repaired_state =
                [&](const std::vector<int32_t>& committed_tokens, uint32_t accepted_count) -> Gemma4Status {
                const auto repair_started = std::chrono::steady_clock::now();
                const auto clone_started = std::chrono::steady_clock::now();
                std::unique_ptr<gemma4d::NativeKvState> serial_kv = cache->native_kv_state->clone();
                if (serial_kv == nullptr) {
                    return fail(
                        GEMMA4_ERR_RUNTIME,
                        "native MTP block-prefix serial-state repair failed to clone target KV state");
                }
                repair_clone_ms += elapsed_ms(clone_started);
                Gemma4StepResult serial_step{};
                std::unique_ptr<gemma4d::NativeHiddenState> serial_hidden;
                float peak_memory_gb = std::max(block_step.peak_memory_gb, cache->last_step.peak_memory_gb);
                const size_t committed_count = committed_tokens.size();
                for (size_t index = 0; index < committed_count; ++index) {
                    if (state_only_serial_repair && index + 1 < committed_count) {
                        Gemma4StepResult state_step{};
                        const auto forward_started = std::chrono::steady_clock::now();
                        if (!target->native_model->decode_incremental_state_only(
                                committed_tokens[index],
                                serial_kv.get(),
                                &state_step,
                                &native_error)) {
                            return fail(GEMMA4_ERR_RUNTIME, native_error);
                        }
                        repair_forward_ms += elapsed_ms(forward_started);
                        peak_memory_gb = std::max(peak_memory_gb, state_step.peak_memory_gb);
                        continue;
                    }
                    const auto forward_started = std::chrono::steady_clock::now();
                    if (!target->native_model->decode_incremental(
                            committed_tokens[index],
                            serial_kv.get(),
                            &serial_step,
                            &native_error,
                            &serial_hidden)) {
                        return fail(GEMMA4_ERR_RUNTIME, native_error);
                    }
                    repair_forward_ms += elapsed_ms(forward_started);
                    peak_memory_gb = std::max(peak_memory_gb, serial_step.peak_memory_gb);
                }
                verify_repair_ms += elapsed_ms(repair_started);
                record_mtp_target_step(&trace, committed_count, context_sequence_len, serial_step);
                record_mtp_hidden_shape(&trace, serial_hidden.get());

                serial_step.peak_memory_gb = peak_memory_gb;
                serial_step.active_kv_bytes = serial_kv->active_bytes();
                serial_step.accepted_draft_count = accepted_count;
                serial_step.committed_count = static_cast<uint32_t>(committed_count);
                for (size_t index = 0;
                     index < committed_count && index < GEMMA4_MTP_MAX_COMMITTED_TOKENS;
                     ++index) {
                    serial_step.committed_tokens[index] = committed_tokens[index];
                }
                serial_step.mtp_trace = trace;
                attach_verify_timings(&serial_step);

                for (size_t index = 0; index < committed_count; ++index) {
                    cache->native_tokens.push_back(committed_tokens[index]);
                }
                cache->native_kv_state = std::move(serial_kv);
                cache->last_hidden = std::move(serial_hidden);
                serial_step.native_last_hidden = cache->last_hidden.get();
                *out = serial_step;
                out->native_last_hidden = cache->last_hidden.get();
                remember_last_step(cache, out);
                target->sequence_len = out->sequence_len;
                return ok();
            };

            auto commit_prefix_repaired_state =
                [&](const std::vector<int32_t>& committed_tokens, uint32_t accepted_count) -> Gemma4Status {
                if (accepted_count == 0 || accepted_count >= committed_tokens.size()) {
                    return fail(
                        GEMMA4_ERR_RUNTIME,
                        "native MTP block-prefix repair requires accepted drafts followed by fallback token");
                }
                const size_t accepted_count_usize = static_cast<size_t>(accepted_count);
                const int32_t fallback_token = committed_tokens.back();
                const auto repair_started = std::chrono::steady_clock::now();
                if (prefix_kv == nullptr || prefix_kv->sequence_len() !=
                        cache->native_tokens.size() + accepted_count_usize) {
                    return fail(
                        GEMMA4_ERR_RUNTIME,
                        "native MTP block-prefix repair missing retroactive accepted-prefix KV");
                }

                Gemma4StepResult fallback_step{};
                std::unique_ptr<gemma4d::NativeHiddenState> fallback_hidden;
                const auto fallback_started = std::chrono::steady_clock::now();
                if (!target->native_model->decode_incremental(
                        fallback_token,
                        prefix_kv.get(),
                        &fallback_step,
                        &native_error,
                        &fallback_hidden)) {
                    return fail(GEMMA4_ERR_RUNTIME, native_error);
                }
                repair_fallback_ms += elapsed_ms(fallback_started);
                verify_repair_ms += elapsed_ms(repair_started);
                fallback_step.peak_memory_gb = std::max(
                    {fallback_step.peak_memory_gb,
                     block_step.peak_memory_gb,
                     cache->last_step.peak_memory_gb});
                fallback_step.active_kv_bytes = prefix_kv->active_bytes();
                record_mtp_target_step(&trace, committed_tokens.size(), context_sequence_len, fallback_step);
                record_mtp_hidden_shape(&trace, fallback_hidden.get());

                fallback_step.accepted_draft_count = accepted_count;
                fallback_step.committed_count = static_cast<uint32_t>(committed_tokens.size());
                for (size_t index = 0;
                     index < committed_tokens.size() && index < GEMMA4_MTP_MAX_COMMITTED_TOKENS;
                     ++index) {
                    fallback_step.committed_tokens[index] = committed_tokens[index];
                }
                fallback_step.mtp_trace = trace;
                attach_verify_timings(&fallback_step);

                for (int32_t token : committed_tokens) {
                    cache->native_tokens.push_back(token);
                }
                cache->native_kv_state = std::move(prefix_kv);
                cache->last_hidden = std::move(fallback_hidden);
                fallback_step.native_last_hidden = cache->last_hidden.get();
                *out = fallback_step;
                out->native_last_hidden = cache->last_hidden.get();
                remember_last_step(cache, out);
                target->sequence_len = out->sequence_len;
                return ok();
            };

            if (full_block_accepted) {
                std::vector<int32_t> committed_tokens(draft_tokens, draft_tokens + draft_count);
                if (serial_state_repair || partial_only_repair_full_accept) {
                    return commit_serial_repaired_state(
                        committed_tokens,
                        static_cast<uint32_t>(draft_count));
                }

                Gemma4StepResult lookahead_step = block_step;
                lookahead_step.greedy_token = block_greedy_tokens[draft_count - 1];
                lookahead_step.greedy_logit = block_greedy_logits[draft_count - 1];
                record_mtp_target_step(
                    &trace,
                    draft_count,
                    context_sequence_len,
                    lookahead_step,
                    &block_target_top_k[draft_count - 1]);
                record_mtp_hidden_shape(&trace, block_hidden.get());

                block_step.greedy_token = block_greedy_tokens[draft_count - 1];
                block_step.greedy_logit = block_greedy_logits[draft_count - 1];
                block_step.peak_memory_gb = std::max(block_step.peak_memory_gb, cache->last_step.peak_memory_gb);
                block_step.accepted_draft_count = static_cast<uint32_t>(draft_count);
                block_step.committed_count = static_cast<uint32_t>(draft_count);
                for (size_t index = 0; index < draft_count && index < GEMMA4_MTP_MAX_COMMITTED_TOKENS; ++index) {
                    block_step.committed_tokens[index] = draft_tokens[index];
                }
                block_step.mtp_trace = trace;
                attach_verify_timings(&block_step);

                for (size_t index = 0; index < draft_count; ++index) {
                    cache->native_tokens.push_back(draft_tokens[index]);
                }
                cache->native_kv_state = std::move(block_kv);
                cache->last_hidden = std::move(block_hidden);
                block_step.native_last_hidden = cache->last_hidden.get();
                *out = block_step;
                out->native_last_hidden = cache->last_hidden.get();
                remember_last_step(cache, out);
                target->sequence_len = out->sequence_len;
                return ok();
            }

            const int32_t fallback_token = block_greedy_tokens[accepted_prefix_count - 1];
            std::vector<int32_t> committed_tokens;
            committed_tokens.reserve(accepted_prefix_count + 1);
            for (size_t index = 0; index < accepted_prefix_count; ++index) {
                committed_tokens.push_back(draft_tokens[index]);
            }
            committed_tokens.push_back(fallback_token);
            if (serial_state_repair || partial_reject_serial_repair) {
                return commit_serial_repaired_state(
                    committed_tokens,
                    static_cast<uint32_t>(accepted_prefix_count));
            }
            if (accepted_prefix_count > 1) {
                return commit_prefix_repaired_state(
                    committed_tokens,
                    static_cast<uint32_t>(accepted_prefix_count));
            }

            Gemma4StepResult fallback_step{};
            std::unique_ptr<gemma4d::NativeHiddenState> fallback_hidden;
            const auto repair_started = std::chrono::steady_clock::now();
            if (prefix_kv == nullptr || prefix_kv->sequence_len() !=
                    cache->native_tokens.size() + accepted_prefix_count) {
                return fail(
                    GEMMA4_ERR_RUNTIME,
                    "native MTP block-prefix repair missing one-token accepted-prefix KV");
            }
            const auto fallback_started = std::chrono::steady_clock::now();
            if (!target->native_model->decode_incremental(
                    fallback_token,
                    prefix_kv.get(),
                    &fallback_step,
                    &native_error,
                    &fallback_hidden)) {
                return fail(GEMMA4_ERR_RUNTIME, native_error);
            }
            repair_fallback_ms += elapsed_ms(fallback_started);
            verify_repair_ms += elapsed_ms(repair_started);
            fallback_step.peak_memory_gb =
                std::max(fallback_step.peak_memory_gb, block_step.peak_memory_gb);
            fallback_step.active_kv_bytes = prefix_kv->active_bytes();
            record_mtp_target_step(&trace, committed_tokens.size(), context_sequence_len, fallback_step);
            record_mtp_hidden_shape(&trace, fallback_hidden.get());

            fallback_step.accepted_draft_count = static_cast<uint32_t>(accepted_prefix_count);
            fallback_step.committed_count = static_cast<uint32_t>(committed_tokens.size());
            for (size_t index = 0;
                 index < committed_tokens.size() && index < GEMMA4_MTP_MAX_COMMITTED_TOKENS;
                 ++index) {
                fallback_step.committed_tokens[index] = committed_tokens[index];
            }
            fallback_step.mtp_trace = trace;
            attach_verify_timings(&fallback_step);

            for (int32_t token : committed_tokens) {
                cache->native_tokens.push_back(token);
            }
            cache->native_kv_state = std::move(prefix_kv);
            cache->last_hidden = std::move(fallback_hidden);
            fallback_step.native_last_hidden = cache->last_hidden.get();
            *out = fallback_step;
            out->native_last_hidden = cache->last_hidden.get();
            remember_last_step(cache, out);
            target->sequence_len = out->sequence_len;
            return ok();
        }

        // Experimental XR16 prototype: commit a batched block only when all
        // draft tokens are accepted; otherwise fall back to exact sequential
        // rollback without changing the default verifier.
        if (terminal_commit_count == 0 &&
            env_flag_enabled("GEMMA4D_EXPERIMENTAL_MTP_BATCH_VERIFY") && draft_count == 2 &&
            draft_tokens[0] == cache->last_step.greedy_token) {
            const auto stage_started = std::chrono::steady_clock::now();
            std::unique_ptr<gemma4d::NativeKvState> block_kv = cache->native_kv_state->clone();
            if (block_kv == nullptr) {
                return fail(GEMMA4_ERR_RUNTIME, "native MTP batch verify failed to clone target KV state");
            }
            verify_stage_ms += elapsed_ms(stage_started);

            Gemma4StepResult block_step{};
            std::vector<int32_t> block_greedy_tokens;
            std::vector<float> block_greedy_logits;
            std::vector<gemma4d::NativeTopKEntries> block_target_top_k;
            std::unique_ptr<gemma4d::NativeHiddenState> block_hidden;
            const auto forward_started = std::chrono::steady_clock::now();
            if (!target->native_model->decode_incremental_block(
                    draft_tokens,
                    draft_count,
                    block_kv.get(),
                    &block_step,
                    &block_greedy_tokens,
                    &block_greedy_logits,
                    &native_error,
                    &block_hidden,
                    &block_target_top_k)) {
                return fail(GEMMA4_ERR_RUNTIME, native_error);
            }
            verify_forward_ms += elapsed_ms(forward_started);
            if (block_greedy_tokens.size() < draft_count || block_greedy_logits.size() < draft_count ||
                block_target_top_k.size() < draft_count) {
                return fail(GEMMA4_ERR_RUNTIME, "native MTP batch verify returned incomplete target logits");
            }

            const bool full_block_accepted = draft_tokens[1] == block_greedy_tokens[0];
            if (full_block_accepted) {
                const uint64_t context_sequence_len = cache->native_tokens.size();
                Gemma4MtpTraceInfo trace{};
                initialize_mtp_trace(&trace, context_sequence_len);

                record_mtp_target_step(&trace, 0, context_sequence_len, cache->last_step);
                record_mtp_draft_score(&trace, 0, draft_tokens[0], draft_score_for(0, draft_tokens[0]));

                Gemma4StepResult second_target_step{};
                second_target_step.greedy_token = block_greedy_tokens[0];
                second_target_step.greedy_logit = block_greedy_logits[0];
                second_target_step.sequence_len = context_sequence_len + 1;
                second_target_step.active_kv_bytes = block_kv->active_bytes();
                record_mtp_target_step(&trace, 1, context_sequence_len, second_target_step, &block_target_top_k[0]);
                record_mtp_draft_score(&trace, 1, draft_tokens[1], draft_score_for(1, draft_tokens[1]));

                Gemma4StepResult lookahead_step = block_step;
                lookahead_step.greedy_token = block_greedy_tokens[1];
                lookahead_step.greedy_logit = block_greedy_logits[1];
                record_mtp_target_step(&trace, 2, context_sequence_len, lookahead_step, &block_target_top_k[1]);
                record_mtp_hidden_shape(&trace, block_hidden.get());

                block_step.greedy_token = block_greedy_tokens[1];
                block_step.greedy_logit = block_greedy_logits[1];
                block_step.peak_memory_gb = std::max(block_step.peak_memory_gb, cache->last_step.peak_memory_gb);
                block_step.accepted_draft_count = 2;
                block_step.committed_count = 2;
                block_step.committed_tokens[0] = draft_tokens[0];
                block_step.committed_tokens[1] = draft_tokens[1];
                block_step.mtp_trace = trace;
                attach_verify_timings(&block_step);

                cache->native_tokens.push_back(draft_tokens[0]);
                cache->native_tokens.push_back(draft_tokens[1]);
                cache->native_kv_state = std::move(block_kv);
                cache->last_hidden = std::move(block_hidden);
                block_step.native_last_hidden = cache->last_hidden.get();
                *out = block_step;
                out->native_last_hidden = cache->last_hidden.get();
                remember_last_step(cache, out);
                target->sequence_len = out->sequence_len;
                return ok();
            }
        }

        const uint64_t context_sequence_len = cache->native_tokens.size();
        Gemma4MtpTraceInfo trace{};
        initialize_mtp_trace(&trace, context_sequence_len);

        // XR18 prototype: measure KV clone/copy overhead while keeping the
        // failure-atomic staged verifier as the default path.
        const bool in_place_verify =
            terminal_commit_count == 0 && env_flag_enabled("GEMMA4D_EXPERIMENTAL_MTP_INPLACE_VERIFY");
        std::unique_ptr<gemma4d::NativeKvState> staged_kv;
        std::vector<int32_t> staged_tokens;
        gemma4d::NativeKvState* verify_kv = cache->native_kv_state.get();
        std::vector<int32_t>* verify_tokens = &cache->native_tokens;
        if (!in_place_verify) {
            const auto stage_started = std::chrono::steady_clock::now();
            staged_kv = cache->native_kv_state->clone();
            if (staged_kv == nullptr) {
                return fail(GEMMA4_ERR_RUNTIME, "native MTP verify failed to clone target KV state");
            }
            staged_tokens = cache->native_tokens;
            verify_kv = staged_kv.get();
            verify_tokens = &staged_tokens;
            verify_stage_ms += elapsed_ms(stage_started);
        }

        std::vector<int32_t> committed_tail;
        committed_tail.reserve(draft_count + 1);
        Gemma4StepResult current_step = cache->last_step;
        float peak_memory_gb = current_step.peak_memory_gb;
        uint64_t active_kv_bytes = verify_kv->active_bytes();
        uint32_t accepted_count = 0;
        std::unique_ptr<gemma4d::NativeHiddenState> staged_hidden;
        bool terminal_lookahead_skipped = false;

        for (size_t index = 0; index < draft_count; ++index) {
            record_mtp_target_step(&trace, index, context_sequence_len, current_step);
            record_mtp_draft_score(&trace, index, draft_tokens[index], draft_score_for(index, draft_tokens[index]));

            const bool accepted = draft_tokens[index] == current_step.greedy_token;
            const int32_t token_to_commit = accepted ? draft_tokens[index] : current_step.greedy_token;
            committed_tail.push_back(token_to_commit);
            verify_tokens->push_back(token_to_commit);
            if (accepted) {
                ++accepted_count;
            }

            if (terminal_commit_count > 0 && committed_tail.size() >= terminal_commit_count) {
                terminal_lookahead_skipped = true;
                break;
            }

            Gemma4StepResult next_step{};
            std::unique_ptr<gemma4d::NativeHiddenState> next_hidden;
            const auto forward_started = std::chrono::steady_clock::now();
            if (!target->native_model->decode_incremental(
                    token_to_commit,
                    verify_kv,
                    &next_step,
                    &native_error,
                    &next_hidden)) {
                return fail(GEMMA4_ERR_RUNTIME, native_error);
            }
            verify_forward_ms += elapsed_ms(forward_started);
            current_step = next_step;
            staged_hidden = std::move(next_hidden);
            peak_memory_gb = std::max(peak_memory_gb, current_step.peak_memory_gb);
            active_kv_bytes = std::max(active_kv_bytes, current_step.active_kv_bytes);

            if (!accepted) {
                break;
            }
        }

        if (!terminal_lookahead_skipped) {
            const size_t lookahead_index = committed_tail.size();
            record_mtp_target_step(&trace, lookahead_index, context_sequence_len, current_step);
            record_mtp_hidden_shape(&trace, staged_hidden.get());
        }

        if (staged_hidden == nullptr && !terminal_lookahead_skipped) {
            return fail(GEMMA4_ERR_RUNTIME, "native MTP verify did not produce a committed hidden state");
        }

        if (terminal_lookahead_skipped) {
            current_step.sequence_len = context_sequence_len + committed_tail.size();
            active_kv_bytes = std::max(active_kv_bytes, current_step.active_kv_bytes);
        }

        *out = current_step;
        out->peak_memory_gb = peak_memory_gb;
        out->active_kv_bytes = active_kv_bytes;
        out->accepted_draft_count = accepted_count;
        out->committed_count = static_cast<uint32_t>(committed_tail.size());
        for (size_t index = 0;
             index < committed_tail.size() && index < GEMMA4_MTP_MAX_COMMITTED_TOKENS;
             ++index) {
            out->committed_tokens[index] = committed_tail[index];
        }
        out->mtp_trace = trace;
        attach_verify_timings(out);

        if (terminal_lookahead_skipped) {
            out->native_last_hidden = nullptr;
            target->sequence_len = out->sequence_len;
            return ok();
        }

        if (!in_place_verify) {
            cache->native_tokens = std::move(staged_tokens);
            cache->native_kv_state = std::move(staged_kv);
        }
        cache->last_hidden = std::move(staged_hidden);
        out->native_last_hidden = cache->last_hidden.get();
        remember_last_step(cache, out);
        target->sequence_len = out->sequence_len;
        return ok();
    }

    return fail(
        GEMMA4_ERR_UNSUPPORTED_CONFIG,
        "gemma4_verify_tokens exact rollback requires the native target graph");
}

Gemma4Status gemma4_verify_tokens(
    Gemma4Target* target,
    Gemma4KvCache* cache,
    const int32_t* draft_tokens,
    size_t draft_count,
    Gemma4StepResult* out) {
    try {
        return verify_tokens_impl(target, cache, draft_tokens, draft_count, 0, out);
    } catch (const std::exception& ex) {
        return fail(GEMMA4_ERR_RUNTIME, ex.what());
    } catch (...) {
        return fail(GEMMA4_ERR_RUNTIME, "gemma4_verify_tokens failed with an unknown exception");
    }
}

Gemma4Status gemma4_verify_tokens_terminal_no_lookahead(
    Gemma4Target* target,
    Gemma4KvCache* cache,
    const int32_t* draft_tokens,
    size_t draft_count,
    size_t terminal_commit_count,
    Gemma4StepResult* out) {
    try {
        return verify_tokens_impl(target, cache, draft_tokens, draft_count, terminal_commit_count, out);
    } catch (const std::exception& ex) {
        return fail(GEMMA4_ERR_RUNTIME, ex.what());
    } catch (...) {
        return fail(
            GEMMA4_ERR_RUNTIME,
            "gemma4_verify_tokens_terminal_no_lookahead failed with an unknown exception");
    }
}
