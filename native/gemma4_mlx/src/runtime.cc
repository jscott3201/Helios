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
};

struct NativeDrafter {
    uint64_t magic;
    bool model_loaded;
    std::string model_path;
    gemma4d::Gemma4ModelManifest manifest;
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

    (void)cache;
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
        if (!target->native_model->forward_greedy(cache->native_tokens, out, &native_error)) {
            return fail(GEMMA4_ERR_RUNTIME, native_error);
        }
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
        cache->native_tokens.push_back(token);
        std::string native_error;
        if (!target->native_model->forward_greedy(cache->native_tokens, out, &native_error)) {
            return fail(GEMMA4_ERR_RUNTIME, native_error);
        }
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

    (void)cache;
    (void)out_tokens;
    *inout_count = 0;
    return fail(
        GEMMA4_ERR_UNSUPPORTED_CONFIG,
        "native MTP drafter execution is not implemented; last target hidden/shared views are not materialized");
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
        cache->native_tokens.insert(cache->native_tokens.end(), draft_tokens, draft_tokens + draft_count);
        std::string native_error;
        if (!target->native_model->forward_greedy(cache->native_tokens, out, &native_error)) {
            return fail(GEMMA4_ERR_RUNTIME, native_error);
        }
        target->sequence_len = out->sequence_len;
        return ok();
    }

    Gemma4Status status = GEMMA4_OK;
    for (size_t index = 0; index < draft_count; ++index) {
        std::ostringstream command;
        command << "{\"cmd\":\"decode_one\",\"token\":" << draft_tokens[index] << "}";
        status = helper_command(target, command.str(), out);
        if (status != GEMMA4_OK) {
            return status;
        }
    }
    return ok();
}
