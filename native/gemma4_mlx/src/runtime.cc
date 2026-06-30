#include "gemma4_mlx.h"
#include "model_manifest.h"
#include "native_model.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <filesystem>
#include <memory>
#include <new>
#include <sstream>
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
thread_local char g_last_error[512] = "";

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
    cache->has_last_step = true;
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
    metadata["last_step.committed_tokens"] =
        join_i32_list(snapshot->last_step.committed_tokens, 4);
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
    snapshot->last_step.committed_count =
        std::min<uint32_t>(metadata_u32(metadata, "last_step.committed_count"), 4);
    const std::vector<int32_t> committed = parse_i32_list(required_metadata(metadata, "last_step.committed_tokens"));
    for (size_t index = 0; index < committed.size() && index < 4; ++index) {
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
struct Gemma4Adapter {};

Gemma4Status gemma4_runtime_version(Gemma4VersionInfo* out) {
    if (out == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_runtime_version requires a non-null out pointer");
    }

    out->abi_version = 1;
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
    size_t* inout_count) {
    if (drafter == nullptr || drafter->magic != kDrafterMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_mtp_draft_block requires a valid drafter handle");
    }
    if (cache == nullptr || cache->magic != kKvCacheMagic) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_mtp_draft_block requires a valid cache handle");
    }
    if (out_tokens == nullptr || inout_count == nullptr) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_mtp_draft_block requires token output buffers");
    }
    if (block_size == 0) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_mtp_draft_block requires block_size > 0");
    }
    if (*inout_count < block_size) {
        return fail(GEMMA4_ERR_INVALID_ARGUMENT, "gemma4_mtp_draft_block output buffer is smaller than block_size");
    }
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

    std::string native_error;
    if (!drafter->native_model->draft_block(
            *drafter->target_native_model,
            *cache->last_hidden,
            cache->native_tokens,
            block_size,
            out_tokens,
            inout_count,
            &native_error)) {
        *inout_count = 0;
        return fail(GEMMA4_ERR_RUNTIME, native_error);
    }
    return ok();
}

Gemma4Status gemma4_verify_tokens(
    Gemma4Target* target,
    Gemma4KvCache* cache,
    const int32_t* draft_tokens,
    size_t draft_count,
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
    if (!target->model_loaded) {
        return fail(
            GEMMA4_ERR_UNSUPPORTED_CONFIG,
            "gemma4_verify_tokens requires a loaded Gemma 4 target model; smoke handles do not execute");
    }
    if (target->use_native_graph) {
        if (target->native_model == nullptr) {
            return fail(GEMMA4_ERR_RUNTIME, "native Gemma 4 model state is missing");
        }
        if (cache->native_tokens.empty()) {
            return fail(
                GEMMA4_ERR_UNSUPPORTED_CONFIG,
                "gemma4_verify_tokens requires a prefilled native target cache");
        }

        std::vector<int32_t> committed_tokens;
        std::string native_error;
        if (!target->native_model->verify_draft_block(
                cache->native_tokens,
                draft_tokens,
                draft_count,
                &committed_tokens,
                out,
                &native_error,
                nullptr)) {
            return fail(GEMMA4_ERR_RUNTIME, native_error);
        }
        cache->native_tokens = std::move(committed_tokens);
        std::unique_ptr<gemma4d::NativeKvState> rebuilt_kv_state;
        std::unique_ptr<gemma4d::NativeHiddenState> rebuilt_hidden;
        Gemma4StepResult rebuilt_step{};
        if (!target->native_model->prefill_incremental(
                cache->native_tokens,
                &rebuilt_step,
                &native_error,
                &rebuilt_kv_state,
                &rebuilt_hidden)) {
            return fail(GEMMA4_ERR_RUNTIME, native_error);
        }
        cache->native_kv_state = std::move(rebuilt_kv_state);
        cache->last_hidden = std::move(rebuilt_hidden);
        out->active_kv_bytes = rebuilt_step.active_kv_bytes;
        if (out->peak_memory_gb < rebuilt_step.peak_memory_gb) {
            out->peak_memory_gb = rebuilt_step.peak_memory_gb;
        }
        out->native_last_hidden = cache->last_hidden.get();
        remember_last_step(cache, out);
        target->sequence_len = out->sequence_len;
        return ok();
    }

    return fail(
        GEMMA4_ERR_UNSUPPORTED_CONFIG,
        "gemma4_verify_tokens exact rollback requires the native target graph");
}
