#include "native_model.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef GEMMA4D_MLX_AVAILABLE
#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/io.h>
#include <mlx/memory.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>
#endif

namespace gemma4d {

struct NativeHiddenState::Impl {
#ifdef GEMMA4D_MLX_AVAILABLE
    struct DSparkTap {
        uint32_t layer_id = 0;
        mlx::core::array hidden;
    };
    mlx::core::array hidden;
    std::optional<mlx::core::array> full_attention_key;
    std::optional<mlx::core::array> full_attention_value;
    std::optional<mlx::core::array> sliding_attention_key;
    std::optional<mlx::core::array> sliding_attention_value;
    std::vector<DSparkTap> dspark_taps;
#endif
    uint64_t sequence_len = 0;
    uint32_t hidden_size = 0;
};

struct NativeKvState::Impl {
#ifdef GEMMA4D_MLX_AVAILABLE
    struct Layer {
        bool full_attention = false;
        std::optional<mlx::core::array> key;
        std::optional<mlx::core::array> value;
    };
    std::vector<Layer> layers;
    std::vector<NativeHiddenState::Impl::DSparkTap> dspark_context_taps;
#endif
    uint64_t sequence_len = 0;
    uint64_t active_bytes = 0;
};

struct NativeLoraAdapter::Impl {
#ifdef GEMMA4D_MLX_AVAILABLE
    struct Module {
        std::string prefix;
        std::string target_module;
        mlx::core::array a_transposed;
        mlx::core::array b_transposed;
        float scale = 1.0f;
        uint64_t resident_bytes = 0;
    };
    std::vector<Module> modules;
#endif
    std::string adapter_id;
    std::string adapter_weight_hash;
    std::vector<std::string> target_modules;
    uint32_t rank = 0;
    float alpha = 0.0f;
    uint64_t resident_bytes = 0;
};

struct NativeTextModel::Impl {
#ifdef GEMMA4D_MLX_AVAILABLE
    std::unordered_map<std::string, mlx::core::array> tensors;
    std::shared_ptr<const NativeLoraAdapter> active_adapter;
#endif
    QuantizationSpec default_quantization;
    std::unordered_map<std::string, QuantizationSpec> quantization_overrides;
    size_t safetensor_file_count = 0;
    size_t language_tensor_count = 0;
    size_t total_tensor_count_seen = 0;
    std::string manifest_summary;
    bool experimental_gather_greedy_logit = false;
    size_t native_prefill_chunk_tokens = 0;
    bool native_prefill_policy_long_context_256 = false;
    bool experimental_skip_decode_peak_reset = false;
    std::vector<uint32_t> dspark_tap_layer_ids;
};

struct NativeMtpAssistantModel::Impl {
#ifdef GEMMA4D_MLX_AVAILABLE
    std::unordered_map<std::string, mlx::core::array> tensors;
#endif
    QuantizationSpec default_quantization;
    std::unordered_map<std::string, QuantizationSpec> quantization_overrides;
    size_t safetensor_file_count = 0;
    size_t assistant_tensor_count = 0;
    size_t total_tensor_count_seen = 0;
    std::string manifest_summary;
};

struct NativeDSparkModel::Impl {
#ifdef GEMMA4D_MLX_AVAILABLE
    std::unordered_map<std::string, mlx::core::array> tensors;
#endif
    size_t safetensor_file_count = 0;
    size_t dspark_tensor_count = 0;
    size_t total_tensor_count_seen = 0;
    uint32_t block_size = 0;
    uint32_t markov_rank = 0;
    uint32_t mask_token_id = 0;
    uint32_t vocab_size = 0;
    std::vector<uint32_t> target_layer_ids;
    std::string manifest_summary;
};

namespace {

std::vector<std::filesystem::path> safetensor_files(const std::filesystem::path& model_path) {
    std::vector<std::filesystem::path> files;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(model_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

bool is_language_tensor(const std::string& key) {
    return key.rfind("language_model.model.", 0) == 0;
}

bool is_assistant_tensor(const std::string& key) {
    return key.rfind("model.", 0) == 0 || key.rfind("pre_projection.", 0) == 0 ||
        key.rfind("post_projection.", 0) == 0;
}

bool is_dspark_tensor(const std::string& key) {
    return key.rfind("layers.", 0) == 0 || key.rfind("embed_tokens.", 0) == 0 ||
        key.rfind("lm_head.", 0) == 0 || key.rfind("fc.", 0) == 0 ||
        key.rfind("hidden_norm.", 0) == 0 || key.rfind("norm.", 0) == 0 ||
        key.rfind("markov_head.", 0) == 0 || key.rfind("confidence_head.", 0) == 0;
}

bool experimental_mtp_real_margins_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
            std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0 &&
            std::strcmp(value, "off") != 0 && std::strcmp(value, "OFF") != 0;
    }();
    return enabled;
}

#ifdef GEMMA4D_MLX_AVAILABLE
bool& xr57_target_logits_anchor_armed() {
    static bool armed = false;
    return armed;
}

bool& xr57_target_logits_anchor_saved() {
    static bool saved = false;
    return saved;
}
#endif

#ifdef GEMMA4D_MLX_AVAILABLE

using mlx::core::array;

array model_dtype(array value);
array model_scalar(float value);

array to_float32(array value) {
    return mlx::core::astype(std::move(value), mlx::core::float32);
}

template <typename Impl>
const array& tensor_or_throw(const Impl& impl, const std::string& key) {
    const auto found = impl.tensors.find(key);
    if (found == impl.tensors.end()) {
        throw std::runtime_error("missing loaded tensor " + key);
    }
    return found->second;
}

template <typename Impl>
QuantizationSpec quantization_for(const Impl& impl, const std::string& prefix) {
    const auto found = impl.quantization_overrides.find(prefix);
    if (found != impl.quantization_overrides.end()) {
        return found->second;
    }
    return impl.default_quantization;
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string trim_ascii(const std::string& value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

std::string canonical_lora_prefix(std::string prefix) {
    if (starts_with(prefix, "base_model.model.model.")) {
        return "language_model.model." + prefix.substr(std::string("base_model.model.model.").size());
    }
    if (starts_with(prefix, "base_model.model.")) {
        return "language_model.model." + prefix.substr(std::string("base_model.model.").size());
    }
    if (starts_with(prefix, "model.")) {
        return "language_model.model." + prefix.substr(std::string("model.").size());
    }
    return prefix;
}

std::optional<std::string> lora_tensor_prefix(const std::string& name, const char* suffix) {
    const std::string suffix_value(suffix);
    if (!ends_with(name, suffix_value)) {
        return std::nullopt;
    }
    return canonical_lora_prefix(name.substr(0, name.size() - suffix_value.size()));
}

std::string target_module_for_prefix(const std::string& prefix) {
    const size_t last_dot = prefix.rfind('.');
    if (last_dot == std::string::npos || last_dot + 1 >= prefix.size()) {
        return prefix;
    }
    return prefix.substr(last_dot + 1);
}

bool target_module_allowed(const std::string& prefix, const std::vector<std::string>& target_modules) {
    const std::string leaf = target_module_for_prefix(prefix);
    for (const std::string& module : target_modules) {
        const std::string trimmed = trim_ascii(module);
        if (trimmed.empty()) {
            continue;
        }
        if (leaf == trimmed || prefix == trimmed || ends_with(prefix, "." + trimmed)) {
            return true;
        }
    }
    return false;
}

uint64_t quantized_linear_input_dim(const QuantizationSpec& spec, const array& weight) {
    const auto& shape = weight.shape();
    if (shape.size() != 2 || spec.bits == 0 || 32 % spec.bits != 0) {
        throw std::runtime_error("unsupported quantized weight shape for LoRA validation");
    }
    return static_cast<uint64_t>(shape[1]) * static_cast<uint64_t>(32 / spec.bits);
}

const NativeLoraAdapter::Impl::Module* active_lora_module(
    const NativeTextModel::Impl& impl,
    const std::string& prefix) {
    if (!impl.active_adapter) {
        return nullptr;
    }
    const NativeLoraAdapter::Impl* adapter = impl.active_adapter->impl();
    if (adapter == nullptr) {
        return nullptr;
    }
    for (const NativeLoraAdapter::Impl::Module& module : adapter->modules) {
        if (module.prefix == prefix) {
            return &module;
        }
    }
    return nullptr;
}

template <typename Impl>
array add_lora_delta_if_active(
    const Impl&,
    const array&,
    const std::string&,
    array base_output) {
    return base_output;
}

array add_lora_delta_if_active(
    const NativeTextModel::Impl& impl,
    const array& x,
    const std::string& prefix,
    array base_output) {
    const NativeLoraAdapter::Impl::Module* module = active_lora_module(impl, prefix);
    if (module == nullptr) {
        return base_output;
    }
    array low_rank = mlx::core::matmul(to_float32(x), module->a_transposed);
    array delta = mlx::core::matmul(low_rank, module->b_transposed) * model_scalar(module->scale);
    return model_dtype(base_output + model_dtype(delta));
}

bool force_float32_enabled() {
    const char* value = std::getenv("GEMMA4D_NATIVE_FORCE_FLOAT32");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

array model_dtype(array value) {
    if (force_float32_enabled()) {
        return value;
    }
    return mlx::core::astype(std::move(value), mlx::core::bfloat16);
}

array model_scalar(float value) {
    if (force_float32_enabled()) {
        return array(value);
    }
    return array(value, mlx::core::bfloat16);
}

template <typename Impl>
array quantized_linear(const Impl& impl, const array& x, const std::string& prefix) {
    const QuantizationSpec spec = quantization_for(impl, prefix);
    array output = model_dtype(mlx::core::quantized_matmul(
        x,
        tensor_or_throw(impl, prefix + ".weight"),
        tensor_or_throw(impl, prefix + ".scales"),
        std::optional<array>(tensor_or_throw(impl, prefix + ".biases")),
        true,
        static_cast<int>(spec.group_size),
        static_cast<int>(spec.bits),
        "affine"));
    return add_lora_delta_if_active(impl, x, prefix, output);
}

array quantized_embedding(const NativeTextModel::Impl& impl, const array& token_ids) {
    const QuantizationSpec spec = quantization_for(impl, "language_model.model.embed_tokens");
    return model_dtype(mlx::core::dequantize(
        mlx::core::take(tensor_or_throw(impl, "language_model.model.embed_tokens.weight"), token_ids, 0),
        mlx::core::take(tensor_or_throw(impl, "language_model.model.embed_tokens.scales"), token_ids, 0),
        std::optional<array>(mlx::core::take(
            tensor_or_throw(impl, "language_model.model.embed_tokens.biases"),
            token_ids,
            0)),
        static_cast<int>(spec.group_size),
        static_cast<int>(spec.bits),
        "affine"));
}

array geglu(const array& gate, const array& x) {
    constexpr float kTanhApprox = 0.7978845608028654f;
    constexpr float kGeluCubic = 0.044715f;
    const array gate_cubed = mlx::core::power(gate, model_scalar(3.0f));
    const array gelu = model_scalar(0.5f) * gate *
        (model_scalar(1.0f) + mlx::core::tanh(
            model_scalar(kTanhApprox) * (gate + model_scalar(kGeluCubic) * gate_cubed)));
    return model_dtype(gelu * x);
}

array proportional_rope_freqs(int head_dim, int rotated_dims, float base) {
    const array exponents =
        mlx::core::arange(0, rotated_dims, 2, mlx::core::float32) / static_cast<double>(head_dim);
    const array rotated = mlx::core::power(array(base), exponents);
    const int unrotated_pairs = (head_dim - rotated_dims) / 2;
    if (unrotated_pairs <= 0) {
        return rotated;
    }
    return mlx::core::concatenate(
        {rotated, mlx::core::full({unrotated_pairs}, std::numeric_limits<float>::infinity(), mlx::core::float32)},
        0);
}

array apply_rope(const array& x, bool full_attention, int head_dim, int offset) {
    if (!full_attention) {
        return mlx::core::fast::rope(
            x,
            head_dim,
            false,
            std::optional<float>(10000.0f),
            1.0f,
            offset);
    }
    const array freqs = proportional_rope_freqs(head_dim, static_cast<int>(head_dim * 0.25f), 1000000.0f);
    return mlx::core::fast::rope(
        x,
        head_dim,
        false,
        std::nullopt,
        1.0f,
        offset,
        std::optional<array>(freqs));
}

std::optional<array> sliding_causal_mask(int sequence_len, int window_size) {
    if (sequence_len <= 1 || sequence_len <= window_size) {
        return std::nullopt;
    }
    const array rinds = mlx::core::expand_dims(mlx::core::arange(sequence_len), 0);
    const array linds = mlx::core::expand_dims(mlx::core::arange(sequence_len), 1);
    return (linds >= rinds) && (linds < (rinds + window_size));
}

struct SharedKvArrays {
    std::optional<array> full_attention_key;
    std::optional<array> full_attention_value;
    std::optional<array> sliding_attention_key;
    std::optional<array> sliding_attention_value;
};

struct BlockPrefixSourceLayer {
    bool full_attention = false;
    int n_kv_heads = 0;
    int head_dim = 0;
    int previous_key_len = 0;
    int attention_key_len = 0;
    std::optional<array> attention_keys;
    std::optional<array> attention_values;
};

struct BlockPrefixSources {
    uint64_t previous_sequence_len = 0;
    size_t token_count = 0;
    std::vector<BlockPrefixSourceLayer> layers;
    std::vector<NativeHiddenState::Impl::DSparkTap> previous_dspark_context_taps;
    std::vector<NativeHiddenState::Impl::DSparkTap> block_dspark_taps;
};

enum class PrefillKvEvalMode {
    PerLayer,
    EndOfPrefill,
    SelectiveFullAttention,
};

enum class DecodeKvEvalMode {
    PerLayer,
    EndOfDecode,
    SelectiveFullAttention,
    DeferToLogits,
};

constexpr int kTargetLayerCount = 48;
constexpr int kHiddenSize = 3840;
constexpr int kSlidingWindowSize = 1024;
constexpr uint64_t kBf16Bytes = 2;
constexpr int kDSparkLayerCount = 5;
constexpr int kDSparkNumHeads = 16;
constexpr int kDSparkKvHeads = 1;
constexpr int kDSparkHeadDim = 512;
constexpr int kDSparkVocabSize = 262144;
constexpr int kDSparkMarkovRank = 256;
constexpr int kDSparkFeatureSize = kHiddenSize * GEMMA4_DSPARK_TARGET_TAP_COUNT;

std::optional<array> decode_block_causal_mask(
    int query_len,
    int key_len,
    int first_query_position,
    int first_key_position,
    bool full_attention) {
    if (query_len <= 1) {
        return std::nullopt;
    }
    const array query_positions = mlx::core::expand_dims(
        mlx::core::arange(first_query_position, first_query_position + query_len),
        1);
    const array key_positions = mlx::core::expand_dims(
        mlx::core::arange(first_key_position, first_key_position + key_len),
        0);
    array mask = key_positions <= query_positions;
    if (!full_attention) {
        mask = mask && (key_positions > (query_positions - kSlidingWindowSize));
    }
    return mask;
}

bool target_layer_full_attention(uint32_t layer_idx) {
    return ((layer_idx + 1) % 6) == 0;
}

PrefillKvEvalMode prefill_kv_eval_mode() {
    const char* value = std::getenv("GEMMA4D_NATIVE_PREFILL_KV_EVAL");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "current") == 0 ||
        std::strcmp(value, "per_layer") == 0) {
        return PrefillKvEvalMode::PerLayer;
    }
    if (std::strcmp(value, "end") == 0 || std::strcmp(value, "end_of_prefill") == 0 ||
        std::strcmp(value, "grouped") == 0) {
        return PrefillKvEvalMode::EndOfPrefill;
    }
    if (std::strcmp(value, "selective") == 0 ||
        std::strcmp(value, "selective_full_attention") == 0) {
        return PrefillKvEvalMode::SelectiveFullAttention;
    }
    return PrefillKvEvalMode::PerLayer;
}

bool eval_prefill_kv_when_stored(PrefillKvEvalMode mode, bool full_attention) {
    return mode == PrefillKvEvalMode::PerLayer ||
        (mode == PrefillKvEvalMode::SelectiveFullAttention && full_attention);
}

bool eval_prefill_kv_at_end(PrefillKvEvalMode mode, bool full_attention) {
    return mode == PrefillKvEvalMode::EndOfPrefill ||
        (mode == PrefillKvEvalMode::SelectiveFullAttention && !full_attention);
}

DecodeKvEvalMode decode_kv_eval_mode() {
    const char* value = std::getenv("GEMMA4D_NATIVE_DECODE_KV_EVAL");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "current") == 0 ||
        std::strcmp(value, "per_layer") == 0) {
        return DecodeKvEvalMode::PerLayer;
    }
    if (std::strcmp(value, "end") == 0 || std::strcmp(value, "end_of_decode") == 0 ||
        std::strcmp(value, "grouped") == 0) {
        return DecodeKvEvalMode::EndOfDecode;
    }
    if (std::strcmp(value, "selective") == 0 ||
        std::strcmp(value, "selective_full_attention") == 0) {
        return DecodeKvEvalMode::SelectiveFullAttention;
    }
    if (std::strcmp(value, "defer") == 0 || std::strcmp(value, "defer_to_logits") == 0 ||
        std::strcmp(value, "logits") == 0) {
        return DecodeKvEvalMode::DeferToLogits;
    }
    return DecodeKvEvalMode::PerLayer;
}

bool eval_decode_kv_when_stored(DecodeKvEvalMode mode, bool full_attention) {
    return mode == DecodeKvEvalMode::PerLayer ||
        (mode == DecodeKvEvalMode::SelectiveFullAttention && full_attention);
}

bool eval_decode_kv_at_end(DecodeKvEvalMode mode, bool full_attention) {
    return mode == DecodeKvEvalMode::EndOfDecode ||
        (mode == DecodeKvEvalMode::SelectiveFullAttention && !full_attention);
}

uint64_t estimate_target_kv_bytes(uint64_t sequence_len) {
    const uint64_t full_layer_count = 8;
    const uint64_t sliding_layer_count = kTargetLayerCount - full_layer_count;
    const uint64_t sliding_len = std::min<uint64_t>(sequence_len, kSlidingWindowSize);
    const uint64_t full_layer_bytes = sequence_len * 1 * 512 * kBf16Bytes * 2;
    const uint64_t sliding_layer_bytes = sliding_len * 8 * 256 * kBf16Bytes * 2;
    return full_layer_count * full_layer_bytes + sliding_layer_count * sliding_layer_bytes;
}

void store_target_layer_kv(
    NativeKvState::Impl::Layer* layer,
    bool full_attention,
    const array& keys,
    const array& values,
    int sequence_len,
    int n_kv_heads,
    int head_dim) {
    if (layer == nullptr) {
        return;
    }
    layer->full_attention = full_attention;
    if (full_attention || sequence_len <= kSlidingWindowSize) {
        layer->key = keys;
        layer->value = values;
    } else {
        const int start = sequence_len - kSlidingWindowSize;
        layer->key = mlx::core::slice(keys, {0, 0, start, 0}, {1, n_kv_heads, sequence_len, head_dim});
        layer->value = mlx::core::slice(values, {0, 0, start, 0}, {1, n_kv_heads, sequence_len, head_dim});
    }
    if (eval_prefill_kv_when_stored(prefill_kv_eval_mode(), full_attention)) {
        mlx::core::eval({*layer->key, *layer->value});
    }
}

void eval_deferred_prefill_kv(NativeKvState::Impl* target_kv, PrefillKvEvalMode mode) {
    if (target_kv == nullptr || mode == PrefillKvEvalMode::PerLayer) {
        return;
    }
    std::vector<array> eval_arrays;
    eval_arrays.reserve(target_kv->layers.size() * 2);
    for (const NativeKvState::Impl::Layer& layer : target_kv->layers) {
        if (!eval_prefill_kv_at_end(mode, layer.full_attention)) {
            continue;
        }
        if (layer.key.has_value()) {
            eval_arrays.push_back(*layer.key);
        }
        if (layer.value.has_value()) {
            eval_arrays.push_back(*layer.value);
        }
    }
    if (!eval_arrays.empty()) {
        mlx::core::eval(eval_arrays);
    }
}

bool should_capture_shared_kv(uint32_t layer_idx, bool full_attention) {
    if (full_attention) {
        return layer_idx == 47;
    }
    return layer_idx == 46;
}

void capture_shared_kv(
    SharedKvArrays* shared_kv,
    uint32_t layer_idx,
    bool full_attention,
    const array& keys,
    const array& values) {
    if (shared_kv == nullptr || !should_capture_shared_kv(layer_idx, full_attention)) {
        return;
    }
    if (full_attention) {
        shared_kv->full_attention_key = keys;
        shared_kv->full_attention_value = values;
    } else {
        shared_kv->sliding_attention_key = keys;
        shared_kv->sliding_attention_value = values;
    }
}

bool trace_layer0_detail_enabled();
bool dump_selected_layer(uint32_t layer_idx);
void trace_feature_stats(const char* label, const array& h, int sequence_len, int feature_dim, bool enabled);
void trace_head_stats(const char* label, const array& h, int sequence_len, int head_dim, bool enabled);
void dump_layer0_tensor(const char* label, const array& h);
void dump_hidden_tensor(const char* label, const array& h);

array attention_forward(
    const NativeTextModel::Impl& impl,
    const array& x,
    uint32_t layer_idx,
    int sequence_len,
    SharedKvArrays* shared_kv,
    NativeKvState::Impl::Layer* target_kv = nullptr) {
    const bool full_attention = target_layer_full_attention(layer_idx);
    const int head_dim = full_attention ? 512 : 256;
    const int n_heads = 16;
    const int n_kv_heads = full_attention ? 1 : 8;
    const std::string base = "language_model.model.layers." + std::to_string(layer_idx);
    const bool trace_layer0 = layer_idx == 0 && trace_layer0_detail_enabled();

    array queries = quantized_linear(impl, x, base + ".self_attn.q_proj");
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("q_proj", queries);
    }
    trace_feature_stats("layer0.q_proj", queries, sequence_len, n_heads * head_dim, trace_layer0);
    queries = mlx::core::reshape(queries, {1, sequence_len, n_heads, head_dim});
    queries = model_dtype(mlx::core::fast::rms_norm(
        queries,
        std::optional<array>(tensor_or_throw(impl, base + ".self_attn.q_norm.weight")),
        1e-6f));

    array keys = quantized_linear(impl, x, base + ".self_attn.k_proj");
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("k_proj", keys);
    }
    trace_feature_stats("layer0.k_proj", keys, sequence_len, n_kv_heads * head_dim, trace_layer0);
    keys = mlx::core::reshape(keys, {1, sequence_len, n_kv_heads, head_dim});
    array values = keys;
    if (!full_attention) {
        values = quantized_linear(impl, x, base + ".self_attn.v_proj");
        if (dump_selected_layer(layer_idx)) {
            dump_layer0_tensor("v_proj", values);
        }
        trace_feature_stats("layer0.v_proj", values, sequence_len, n_kv_heads * head_dim, trace_layer0);
        values = mlx::core::reshape(values, {1, sequence_len, n_kv_heads, head_dim});
    }

    keys = model_dtype(mlx::core::fast::rms_norm(
        keys,
        std::optional<array>(tensor_or_throw(impl, base + ".self_attn.k_norm.weight")),
        1e-6f));
    keys = mlx::core::transpose(keys, {0, 2, 1, 3});
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("k_norm", keys);
    }
    trace_head_stats("layer0.k_norm", keys, sequence_len, head_dim, trace_layer0);
    keys = apply_rope(keys, full_attention, head_dim, 0);
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("k_rope", keys);
    }
    trace_head_stats("layer0.k_rope", keys, sequence_len, head_dim, trace_layer0);

    values = model_dtype(mlx::core::fast::rms_norm(values, std::nullopt, 1e-6f));
    values = mlx::core::transpose(values, {0, 2, 1, 3});
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("v_norm", values);
    }
    trace_head_stats("layer0.v_norm", values, sequence_len, head_dim, trace_layer0);

    queries = mlx::core::transpose(queries, {0, 2, 1, 3});
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("q_norm", queries);
    }
    trace_head_stats("layer0.q_norm", queries, sequence_len, head_dim, trace_layer0);
    queries = apply_rope(queries, full_attention, head_dim, 0);
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("q_rope", queries);
    }
    trace_head_stats("layer0.q_rope", queries, sequence_len, head_dim, trace_layer0);
    capture_shared_kv(shared_kv, layer_idx, full_attention, keys, values);
    store_target_layer_kv(target_kv, full_attention, keys, values, sequence_len, n_kv_heads, head_dim);

    const std::optional<array> mask = full_attention ? std::nullopt : sliding_causal_mask(sequence_len, 1024);
    const std::string mask_mode = sequence_len == 1 || mask.has_value() ? "" : "causal";
    array output = mlx::core::fast::scaled_dot_product_attention(
        queries,
        keys,
        values,
        1.0f,
        mask_mode,
        mask);
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("sdpa", output);
    }
    trace_head_stats("layer0.sdpa", output, sequence_len, head_dim, trace_layer0);
    output = mlx::core::transpose(output, {0, 2, 1, 3});
    output = mlx::core::reshape(output, {1, sequence_len, n_heads * head_dim});
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("attn_merge", output);
    }
    trace_feature_stats("layer0.attn_merge", output, sequence_len, n_heads * head_dim, trace_layer0);
    array attn = quantized_linear(impl, output, base + ".self_attn.o_proj");
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("attn_out", attn);
    }
    return attn;
}

array layer_forward(
    const NativeTextModel::Impl& impl,
    const array& x,
    uint32_t layer_idx,
    int sequence_len,
    SharedKvArrays* shared_kv,
    NativeKvState::Impl::Layer* target_kv = nullptr) {
    const std::string base = "language_model.model.layers." + std::to_string(layer_idx);
    const array residual = x;
    const bool trace_layer0 = layer_idx == 0 && trace_layer0_detail_enabled();

    array h = model_dtype(mlx::core::fast::rms_norm(
        x,
        std::optional<array>(tensor_or_throw(impl, base + ".input_layernorm.weight")),
        1e-6f));
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("input_norm", h);
    }
    trace_feature_stats("layer0.input_norm", h, sequence_len, 3840, trace_layer0);
    h = attention_forward(impl, h, layer_idx, sequence_len, shared_kv, target_kv);
    trace_feature_stats("layer0.attn_out", h, sequence_len, 3840, trace_layer0);
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, base + ".post_attention_layernorm.weight")),
        1e-6f));
    trace_feature_stats("layer0.post_attn_norm", h, sequence_len, 3840, trace_layer0);
    h = model_dtype(residual + h);
    trace_feature_stats("layer0.attn_residual", h, sequence_len, 3840, trace_layer0);

    const array mlp_residual = h;
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, base + ".pre_feedforward_layernorm.weight")),
        1e-6f));
    trace_feature_stats("layer0.pre_ff_norm", h, sequence_len, 3840, trace_layer0);
    array gate = quantized_linear(impl, h, base + ".mlp.gate_proj");
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("gate_proj", gate);
    }
    trace_feature_stats("layer0.gate_proj", gate, sequence_len, 15360, trace_layer0);
    array up = quantized_linear(impl, h, base + ".mlp.up_proj");
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("up_proj", up);
    }
    trace_feature_stats("layer0.up_proj", up, sequence_len, 15360, trace_layer0);
    h = model_dtype(geglu(gate, up));
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("geglu", h);
    }
    trace_feature_stats("layer0.geglu", h, sequence_len, 15360, trace_layer0);
    h = quantized_linear(impl, h, base + ".mlp.down_proj");
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("down_proj", h);
    }
    trace_feature_stats("layer0.down_proj", h, sequence_len, 3840, trace_layer0);
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, base + ".post_feedforward_layernorm.weight")),
        1e-6f));
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("post_ff_norm", h);
    }
    trace_feature_stats("layer0.post_ff_norm", h, sequence_len, 3840, trace_layer0);
    h = model_dtype(mlp_residual + h);
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("ff_residual", h);
    }
    trace_feature_stats("layer0.ff_residual", h, sequence_len, 3840, trace_layer0);

    h = model_dtype(h * tensor_or_throw(impl, base + ".layer_scalar"));
    if (dump_selected_layer(layer_idx)) {
        dump_layer0_tensor("layer_scalar", h);
    }
    trace_feature_stats("layer0.layer_scalar", h, sequence_len, 3840, trace_layer0);
    return h;
}

array target_attention_decode_forward(
    const NativeTextModel::Impl& impl,
    const array& x,
    uint32_t layer_idx,
    uint64_t previous_sequence_len,
    NativeKvState::Impl::Layer* target_kv,
    SharedKvArrays* shared_kv) {
    if (target_kv == nullptr || !target_kv->key.has_value() || !target_kv->value.has_value()) {
        throw std::runtime_error("native incremental decode requires materialized per-layer KV state");
    }
    const bool full_attention = target_layer_full_attention(layer_idx);
    const int head_dim = full_attention ? 512 : 256;
    const int n_heads = 16;
    const int n_kv_heads = full_attention ? 1 : 8;
    const std::string base = "language_model.model.layers." + std::to_string(layer_idx);

    array queries = quantized_linear(impl, x, base + ".self_attn.q_proj");
    queries = mlx::core::reshape(queries, {1, 1, n_heads, head_dim});
    queries = model_dtype(mlx::core::fast::rms_norm(
        queries,
        std::optional<array>(tensor_or_throw(impl, base + ".self_attn.q_norm.weight")),
        1e-6f));
    queries = mlx::core::transpose(queries, {0, 2, 1, 3});
    queries = apply_rope(queries, full_attention, head_dim, static_cast<int>(previous_sequence_len));

    array keys = quantized_linear(impl, x, base + ".self_attn.k_proj");
    keys = mlx::core::reshape(keys, {1, 1, n_kv_heads, head_dim});
    array values = keys;
    if (!full_attention) {
        values = quantized_linear(impl, x, base + ".self_attn.v_proj");
        values = mlx::core::reshape(values, {1, 1, n_kv_heads, head_dim});
    }
    keys = model_dtype(mlx::core::fast::rms_norm(
        keys,
        std::optional<array>(tensor_or_throw(impl, base + ".self_attn.k_norm.weight")),
        1e-6f));
    keys = mlx::core::transpose(keys, {0, 2, 1, 3});
    keys = apply_rope(keys, full_attention, head_dim, static_cast<int>(previous_sequence_len));
    values = model_dtype(mlx::core::fast::rms_norm(values, std::nullopt, 1e-6f));
    values = mlx::core::transpose(values, {0, 2, 1, 3});

    array cached_keys = mlx::core::concatenate({*target_kv->key, keys}, 2);
    array cached_values = mlx::core::concatenate({*target_kv->value, values}, 2);
    if (!full_attention) {
        const uint64_t combined_len = std::min<uint64_t>(previous_sequence_len, kSlidingWindowSize) + 1;
        if (combined_len > kSlidingWindowSize) {
            cached_keys = mlx::core::slice(
                cached_keys,
                {0, 0, static_cast<int>(combined_len - kSlidingWindowSize), 0},
                {1, n_kv_heads, static_cast<int>(combined_len), head_dim});
            cached_values = mlx::core::slice(
                cached_values,
                {0, 0, static_cast<int>(combined_len - kSlidingWindowSize), 0},
                {1, n_kv_heads, static_cast<int>(combined_len), head_dim});
        }
    }
    target_kv->full_attention = full_attention;
    target_kv->key = cached_keys;
    target_kv->value = cached_values;
    if (eval_decode_kv_when_stored(decode_kv_eval_mode(), full_attention)) {
        mlx::core::eval({*target_kv->key, *target_kv->value});
    }
    capture_shared_kv(shared_kv, layer_idx, full_attention, *target_kv->key, *target_kv->value);

    array output = mlx::core::fast::scaled_dot_product_attention(
        queries,
        *target_kv->key,
        *target_kv->value,
        1.0f,
        "",
        std::nullopt);
    output = mlx::core::transpose(output, {0, 2, 1, 3});
    output = mlx::core::reshape(output, {1, 1, n_heads * head_dim});
    return quantized_linear(impl, output, base + ".self_attn.o_proj");
}

array target_attention_decode_block_forward(
    const NativeTextModel::Impl& impl,
    const array& x,
    uint32_t layer_idx,
    uint64_t previous_sequence_len,
    int block_len,
    NativeKvState::Impl::Layer* target_kv,
    SharedKvArrays* shared_kv,
    BlockPrefixSourceLayer* prefix_source = nullptr) {
    if (target_kv == nullptr || !target_kv->key.has_value() || !target_kv->value.has_value()) {
        throw std::runtime_error("native incremental block decode requires materialized per-layer KV state");
    }
    const bool full_attention = target_layer_full_attention(layer_idx);
    const int head_dim = full_attention ? 512 : 256;
    const int n_heads = 16;
    const int n_kv_heads = full_attention ? 1 : 8;
    const std::string base = "language_model.model.layers." + std::to_string(layer_idx);

    const auto& previous_key_shape = target_kv->key->shape();
    if (previous_key_shape.size() < 3) {
        throw std::runtime_error("native incremental block decode received malformed KV key shape");
    }
    const int previous_key_len = previous_key_shape[2];
    const int first_key_position = full_attention
        ? 0
        : static_cast<int>(previous_sequence_len) - previous_key_len;

    array queries = quantized_linear(impl, x, base + ".self_attn.q_proj");
    queries = mlx::core::reshape(queries, {1, block_len, n_heads, head_dim});
    queries = model_dtype(mlx::core::fast::rms_norm(
        queries,
        std::optional<array>(tensor_or_throw(impl, base + ".self_attn.q_norm.weight")),
        1e-6f));
    queries = mlx::core::transpose(queries, {0, 2, 1, 3});
    queries = apply_rope(queries, full_attention, head_dim, static_cast<int>(previous_sequence_len));

    array keys = quantized_linear(impl, x, base + ".self_attn.k_proj");
    keys = mlx::core::reshape(keys, {1, block_len, n_kv_heads, head_dim});
    array values = keys;
    if (!full_attention) {
        values = quantized_linear(impl, x, base + ".self_attn.v_proj");
        values = mlx::core::reshape(values, {1, block_len, n_kv_heads, head_dim});
    }
    keys = model_dtype(mlx::core::fast::rms_norm(
        keys,
        std::optional<array>(tensor_or_throw(impl, base + ".self_attn.k_norm.weight")),
        1e-6f));
    keys = mlx::core::transpose(keys, {0, 2, 1, 3});
    keys = apply_rope(keys, full_attention, head_dim, static_cast<int>(previous_sequence_len));
    values = model_dtype(mlx::core::fast::rms_norm(values, std::nullopt, 1e-6f));
    values = mlx::core::transpose(values, {0, 2, 1, 3});

    // Attend over the unsliced cache plus new block keys; slice only the stored
    // KV below so earlier queries in the block keep their valid sliding window.
    array attention_keys = mlx::core::concatenate({*target_kv->key, keys}, 2);
    array attention_values = mlx::core::concatenate({*target_kv->value, values}, 2);
    array stored_keys = attention_keys;
    array stored_values = attention_values;
    const auto& attention_key_shape = attention_keys.shape();
    if (attention_key_shape.size() < 3) {
        throw std::runtime_error("native incremental block decode produced malformed KV key shape");
    }
    const int attention_key_len = attention_key_shape[2];
    if (prefix_source != nullptr) {
        prefix_source->full_attention = full_attention;
        prefix_source->n_kv_heads = n_kv_heads;
        prefix_source->head_dim = head_dim;
        prefix_source->previous_key_len = previous_key_len;
        prefix_source->attention_key_len = attention_key_len;
        prefix_source->attention_keys = attention_keys;
        prefix_source->attention_values = attention_values;
    }
    if (!full_attention && attention_key_len > kSlidingWindowSize) {
        stored_keys = mlx::core::slice(
            attention_keys,
            {0, 0, attention_key_len - kSlidingWindowSize, 0},
            {1, n_kv_heads, attention_key_len, head_dim});
        stored_values = mlx::core::slice(
            attention_values,
            {0, 0, attention_key_len - kSlidingWindowSize, 0},
            {1, n_kv_heads, attention_key_len, head_dim});
    }
    target_kv->full_attention = full_attention;
    target_kv->key = stored_keys;
    target_kv->value = stored_values;
    if (eval_decode_kv_when_stored(decode_kv_eval_mode(), full_attention)) {
        mlx::core::eval({*target_kv->key, *target_kv->value});
    }
    capture_shared_kv(shared_kv, layer_idx, full_attention, *target_kv->key, *target_kv->value);

    const std::optional<array> mask = decode_block_causal_mask(
        block_len,
        attention_key_len,
        static_cast<int>(previous_sequence_len),
        first_key_position,
        full_attention);
    array output = mlx::core::fast::scaled_dot_product_attention(
        queries,
        attention_keys,
        attention_values,
        1.0f,
        "",
        mask);
    output = mlx::core::transpose(output, {0, 2, 1, 3});
    output = mlx::core::reshape(output, {1, block_len, n_heads * head_dim});
    return quantized_linear(impl, output, base + ".self_attn.o_proj");
}

array target_layer_decode_forward(
    const NativeTextModel::Impl& impl,
    const array& x,
    uint32_t layer_idx,
    uint64_t previous_sequence_len,
    NativeKvState::Impl::Layer* target_kv,
    SharedKvArrays* shared_kv) {
    const std::string base = "language_model.model.layers." + std::to_string(layer_idx);
    const array residual = x;

    array h = model_dtype(mlx::core::fast::rms_norm(
        x,
        std::optional<array>(tensor_or_throw(impl, base + ".input_layernorm.weight")),
        1e-6f));
    h = target_attention_decode_forward(impl, h, layer_idx, previous_sequence_len, target_kv, shared_kv);
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, base + ".post_attention_layernorm.weight")),
        1e-6f));
    h = model_dtype(residual + h);

    const array mlp_residual = h;
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, base + ".pre_feedforward_layernorm.weight")),
        1e-6f));
    array gate = quantized_linear(impl, h, base + ".mlp.gate_proj");
    array up = quantized_linear(impl, h, base + ".mlp.up_proj");
    h = model_dtype(geglu(gate, up));
    h = quantized_linear(impl, h, base + ".mlp.down_proj");
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, base + ".post_feedforward_layernorm.weight")),
        1e-6f));
    h = model_dtype(mlp_residual + h);
    return model_dtype(h * tensor_or_throw(impl, base + ".layer_scalar"));
}

array target_layer_decode_block_forward(
    const NativeTextModel::Impl& impl,
    const array& x,
    uint32_t layer_idx,
    uint64_t previous_sequence_len,
    int block_len,
    NativeKvState::Impl::Layer* target_kv,
    SharedKvArrays* shared_kv,
    BlockPrefixSourceLayer* prefix_source = nullptr) {
    const std::string base = "language_model.model.layers." + std::to_string(layer_idx);
    const array residual = x;

    array h = model_dtype(mlx::core::fast::rms_norm(
        x,
        std::optional<array>(tensor_or_throw(impl, base + ".input_layernorm.weight")),
        1e-6f));
    h = target_attention_decode_block_forward(
        impl,
        h,
        layer_idx,
        previous_sequence_len,
        block_len,
        target_kv,
        shared_kv,
        prefix_source);
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, base + ".post_attention_layernorm.weight")),
        1e-6f));
    h = model_dtype(residual + h);

    const array mlp_residual = h;
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, base + ".pre_feedforward_layernorm.weight")),
        1e-6f));
    array gate = quantized_linear(impl, h, base + ".mlp.gate_proj");
    array up = quantized_linear(impl, h, base + ".mlp.up_proj");
    h = model_dtype(geglu(gate, up));
    h = quantized_linear(impl, h, base + ".mlp.down_proj");
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, base + ".post_feedforward_layernorm.weight")),
        1e-6f));
    h = model_dtype(mlp_residual + h);
    return model_dtype(h * tensor_or_throw(impl, base + ".layer_scalar"));
}

void eval_deferred_decode_kv(NativeKvState::Impl* target_kv, DecodeKvEvalMode mode) {
    if (target_kv == nullptr || mode == DecodeKvEvalMode::PerLayer ||
        mode == DecodeKvEvalMode::DeferToLogits) {
        return;
    }
    std::vector<array> eval_arrays;
    eval_arrays.reserve(target_kv->layers.size() * 2);
    for (const NativeKvState::Impl::Layer& layer : target_kv->layers) {
        if (!eval_decode_kv_at_end(mode, layer.full_attention)) {
            continue;
        }
        if (layer.key.has_value()) {
            eval_arrays.push_back(*layer.key);
        }
        if (layer.value.has_value()) {
            eval_arrays.push_back(*layer.value);
        }
    }
    if (!eval_arrays.empty()) {
        mlx::core::eval(eval_arrays);
    }
}

bool trace_layer_stats_enabled() {
    const char* value = std::getenv("GEMMA4D_NATIVE_TRACE_LAYER_STATS");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

bool trace_layer0_detail_enabled() {
    const char* value = std::getenv("GEMMA4D_NATIVE_TRACE_LAYER0_DETAIL");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

bool dump_selected_layer(uint32_t layer_idx) {
    const char* dump_dir = std::getenv("GEMMA4D_NATIVE_DUMP_LAYER0");
    if (dump_dir == nullptr || dump_dir[0] == '\0') {
        return false;
    }
    const char* selected = std::getenv("GEMMA4D_NATIVE_DUMP_LAYER_INDEX");
    if (selected == nullptr || selected[0] == '\0') {
        return layer_idx == 0;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(selected, &end, 10);
    return end != selected && *end == '\0' && parsed == layer_idx;
}

void dump_layer0_tensor(const char* label, const array& h) {
    const char* dump_dir = std::getenv("GEMMA4D_NATIVE_DUMP_LAYER0");
    if (dump_dir == nullptr || dump_dir[0] == '\0') {
        return;
    }

    const std::filesystem::path dir(dump_dir);
    std::filesystem::create_directories(dir);
    std::unordered_map<std::string, array> tensors;
    tensors.emplace("tensor", h);
    mlx::core::save_safetensors((dir / (std::string(label) + ".safetensors")).string(), std::move(tensors));
}

void dump_hidden_tensor(const char* label, const array& h) {
    const char* dump_dir = std::getenv("GEMMA4D_NATIVE_DUMP_HIDDEN");
    if (dump_dir == nullptr || dump_dir[0] == '\0') {
        return;
    }

    const std::filesystem::path dir(dump_dir);
    std::filesystem::create_directories(dir);
    std::unordered_map<std::string, array> tensors;
    tensors.emplace("tensor", h);
    mlx::core::save_safetensors((dir / (std::string(label) + ".safetensors")).string(), std::move(tensors));
}

void trace_feature_stats(const char* label, const array& h, int sequence_len, int feature_dim, bool enabled) {
    if (!enabled) {
        return;
    }

    array last = mlx::core::reshape(
        mlx::core::slice(h, {0, sequence_len - 1, 0}, {1, sequence_len, feature_dim}),
        {feature_dim});
    array rms = to_float32(mlx::core::sqrt(mlx::core::mean(mlx::core::square(last))));

    const std::vector<int32_t> sample_dims = {0, 1, 2, 3};
    array sample_dim_ids(sample_dims.begin(), {static_cast<int>(sample_dims.size())}, mlx::core::int32);
    array sample = to_float32(mlx::core::take(last, sample_dim_ids, 0));
    mlx::core::eval({rms, sample});

    std::cerr << "gemma4d_native_trace " << label << " last_rms=" << rms.item<float>()
              << " first4=[";
    for (size_t index = 0; index < sample_dims.size(); ++index) {
        if (index != 0) {
            std::cerr << ',';
        }
        array scalar = mlx::core::slice(sample, {static_cast<int>(index)}, {static_cast<int>(index + 1)});
        mlx::core::eval(scalar);
        std::cerr << scalar.item<float>();
    }
    std::cerr << "]\n";
}

void trace_head_stats(const char* label, const array& h, int sequence_len, int head_dim, bool enabled) {
    if (!enabled) {
        return;
    }

    array last = mlx::core::reshape(
        mlx::core::slice(h, {0, 0, sequence_len - 1, 0}, {1, 1, sequence_len, head_dim}),
        {head_dim});
    array rms = to_float32(mlx::core::sqrt(mlx::core::mean(mlx::core::square(last))));

    const std::vector<int32_t> sample_dims = {0, 1, 2, 3};
    array sample_dim_ids(sample_dims.begin(), {static_cast<int>(sample_dims.size())}, mlx::core::int32);
    array sample = to_float32(mlx::core::take(last, sample_dim_ids, 0));
    mlx::core::eval({rms, sample});

    std::cerr << "gemma4d_native_trace " << label << " head0_last_rms=" << rms.item<float>()
              << " head0_first4=[";
    for (size_t index = 0; index < sample_dims.size(); ++index) {
        if (index != 0) {
            std::cerr << ',';
        }
        array scalar = mlx::core::slice(sample, {static_cast<int>(index)}, {static_cast<int>(index + 1)});
        mlx::core::eval(scalar);
        std::cerr << scalar.item<float>();
    }
    std::cerr << "]\n";
}

void trace_hidden_stats(const char* label, const array& h, int sequence_len) {
    trace_feature_stats(label, h, sequence_len, 3840, trace_layer_stats_enabled());
}

struct DSparkHiddenTaps {
    std::vector<NativeHiddenState::Impl::DSparkTap> taps;
};

bool should_capture_dspark_tap(const NativeTextModel::Impl& impl, uint32_t layer_id) {
    return std::find(
               impl.dspark_tap_layer_ids.begin(),
               impl.dspark_tap_layer_ids.end(),
               layer_id) != impl.dspark_tap_layer_ids.end();
}

void maybe_capture_dspark_tap(
    const NativeTextModel::Impl& impl,
    uint32_t layer_id,
    const array& hidden,
    DSparkHiddenTaps* taps) {
    if (taps == nullptr || !should_capture_dspark_tap(impl, layer_id)) {
        return;
    }
    taps->taps.push_back(NativeHiddenState::Impl::DSparkTap{layer_id, hidden});
}

DSparkHiddenTaps last_token_dspark_taps(DSparkHiddenTaps taps, int sequence_len) {
    if (sequence_len <= 0) {
        taps.taps.clear();
        return taps;
    }
    for (NativeHiddenState::Impl::DSparkTap& tap : taps.taps) {
        tap.hidden = mlx::core::slice(
            tap.hidden,
            {0, sequence_len - 1, 0},
            {1, sequence_len, kHiddenSize});
    }
    return taps;
}

void replace_dspark_context_taps(NativeKvState::Impl* target_kv, const DSparkHiddenTaps& taps) {
    if (target_kv == nullptr) {
        return;
    }
    target_kv->dspark_context_taps = taps.taps;
}

void append_dspark_context_taps(NativeKvState::Impl* target_kv, const DSparkHiddenTaps& taps) {
    if (target_kv == nullptr || taps.taps.empty()) {
        return;
    }
    if (target_kv->dspark_context_taps.empty()) {
        target_kv->dspark_context_taps = taps.taps;
        return;
    }
    if (target_kv->dspark_context_taps.size() != taps.taps.size()) {
        throw std::runtime_error("native DSpark context tap append received a different tap count");
    }
    for (size_t index = 0; index < taps.taps.size(); ++index) {
        NativeHiddenState::Impl::DSparkTap& existing = target_kv->dspark_context_taps[index];
        const NativeHiddenState::Impl::DSparkTap& next = taps.taps[index];
        if (existing.layer_id != next.layer_id) {
            throw std::runtime_error("native DSpark context tap append received mismatched layer ids");
        }
        existing.hidden = mlx::core::concatenate({existing.hidden, next.hidden}, 1);
    }
}

std::vector<NativeHiddenState::Impl::DSparkTap> prefix_dspark_context_taps(
    const std::vector<NativeHiddenState::Impl::DSparkTap>& block_taps,
    size_t prefix_token_count) {
    if (prefix_token_count == 0 || prefix_token_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("native DSpark context prefix count is invalid");
    }
    std::vector<NativeHiddenState::Impl::DSparkTap> prefix_taps;
    prefix_taps.reserve(block_taps.size());
    for (const NativeHiddenState::Impl::DSparkTap& tap : block_taps) {
        prefix_taps.push_back(NativeHiddenState::Impl::DSparkTap{
            tap.layer_id,
            mlx::core::slice(
                tap.hidden,
                {0, 0, 0},
                {1, static_cast<int>(prefix_token_count), kHiddenSize})});
    }
    return prefix_taps;
}

std::vector<NativeHiddenState::Impl::DSparkTap> last_token_from_dspark_context_taps(
    const std::vector<NativeHiddenState::Impl::DSparkTap>& context_taps,
    uint64_t sequence_len) {
    if (sequence_len == 0 || sequence_len > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        return {};
    }
    std::vector<NativeHiddenState::Impl::DSparkTap> last_taps;
    last_taps.reserve(context_taps.size());
    const int stop = static_cast<int>(sequence_len);
    for (const NativeHiddenState::Impl::DSparkTap& tap : context_taps) {
        last_taps.push_back(NativeHiddenState::Impl::DSparkTap{
            tap.layer_id,
            mlx::core::slice(tap.hidden, {0, stop - 1, 0}, {1, stop, kHiddenSize})});
    }
    return last_taps;
}

struct NativeHiddenArrays {
    array hidden;
    SharedKvArrays shared_kv;
    DSparkHiddenTaps dspark_taps;
};

struct NativeForwardArrays {
    array logits;
    array last_hidden;
    SharedKvArrays shared_kv;
    DSparkHiddenTaps dspark_taps;
    BlockPrefixSources prefix_sources;
};

NativeHiddenArrays decode_block_hidden(
    const NativeTextModel::Impl& impl,
    const int32_t* tokens,
    size_t token_count,
    NativeKvState::Impl* target_kv) {
    if (target_kv == nullptr || target_kv->sequence_len == 0 || target_kv->layers.size() != kTargetLayerCount) {
        throw std::runtime_error("native incremental block decode requires a populated target KV cache");
    }
    if (tokens == nullptr || token_count == 0) {
        throw std::runtime_error("native incremental block decode requires at least one token");
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("native incremental block decode token count exceeds MLX shape limits");
    }
    if (target_kv->sequence_len + token_count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("native incremental block decode position exceeds MLX shape limits");
    }

    const uint64_t previous_sequence_len = target_kv->sequence_len;
    const int block_len = static_cast<int>(token_count);
    std::vector<int32_t> ids(tokens, tokens + token_count);
    array token_ids(ids.begin(), {1, block_len}, mlx::core::int32);
    array h = model_dtype(quantized_embedding(impl, token_ids) * model_scalar(std::sqrt(3840.0f)));
    SharedKvArrays shared_kv;
    DSparkHiddenTaps dspark_taps;

    for (uint32_t layer = 0; layer < kTargetLayerCount; ++layer) {
        h = target_layer_decode_block_forward(
            impl,
            h,
            layer,
            previous_sequence_len,
            block_len,
            &target_kv->layers[layer],
            &shared_kv);
        maybe_capture_dspark_tap(impl, layer, h, &dspark_taps);
    }
    append_dspark_context_taps(target_kv, dspark_taps);
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, "language_model.model.norm.weight")),
        1e-6f));
    eval_deferred_decode_kv(target_kv, decode_kv_eval_mode());
    target_kv->sequence_len = previous_sequence_len + token_count;
    target_kv->active_bytes = estimate_target_kv_bytes(target_kv->sequence_len);
    return NativeHiddenArrays{std::move(h), std::move(shared_kv), std::move(dspark_taps)};
}

NativeHiddenArrays forward_hidden(
    const NativeTextModel::Impl& impl,
    const std::vector<int32_t>& tokens,
    NativeKvState::Impl* target_kv = nullptr) {
    if (tokens.empty()) {
        throw std::runtime_error("native forward requires at least one token");
    }
    if (tokens.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("native forward token count exceeds MLX shape limits");
    }

    const int sequence_len = static_cast<int>(tokens.size());
    array token_ids(tokens.begin(), {1, sequence_len}, mlx::core::int32);
    array h = model_dtype(quantized_embedding(impl, token_ids) * model_scalar(std::sqrt(3840.0f)));
    SharedKvArrays shared_kv;
    DSparkHiddenTaps dspark_taps;
    if (target_kv != nullptr) {
        target_kv->layers.clear();
        target_kv->layers.reserve(kTargetLayerCount);
        target_kv->sequence_len = 0;
        target_kv->active_bytes = 0;
    }
    dump_hidden_tensor("embed", h);
    trace_hidden_stats("embed", h, sequence_len);

    for (uint32_t layer = 0; layer < kTargetLayerCount; ++layer) {
        NativeKvState::Impl::Layer* layer_kv = nullptr;
        if (target_kv != nullptr) {
            target_kv->layers.emplace_back();
            layer_kv = &target_kv->layers.back();
        }
        h = layer_forward(impl, h, layer, sequence_len, &shared_kv, layer_kv);
        maybe_capture_dspark_tap(impl, layer, h, &dspark_taps);
        const std::string label = "layer" + std::to_string(layer);
        dump_hidden_tensor(label.c_str(), h);
        trace_hidden_stats(label.c_str(), h, sequence_len);
    }
    if (target_kv != nullptr) {
        replace_dspark_context_taps(target_kv, dspark_taps);
    }
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, "language_model.model.norm.weight")),
        1e-6f));
    dump_hidden_tensor("final_norm", h);
    trace_hidden_stats("final_norm", h, sequence_len);
    if (target_kv != nullptr) {
        eval_deferred_prefill_kv(target_kv, prefill_kv_eval_mode());
        target_kv->sequence_len = tokens.size();
        target_kv->active_bytes = estimate_target_kv_bytes(tokens.size());
    }

    return NativeHiddenArrays{std::move(h), std::move(shared_kv), std::move(dspark_taps)};
}

array target_logits_for_hidden(const NativeTextModel::Impl& impl, const array& h) {
    const QuantizationSpec embed_quantization = quantization_for(impl, "language_model.model.embed_tokens");
    array logits = mlx::core::quantized_matmul(
        h,
        tensor_or_throw(impl, "language_model.model.embed_tokens.weight"),
        tensor_or_throw(impl, "language_model.model.embed_tokens.scales"),
        std::optional<array>(tensor_or_throw(impl, "language_model.model.embed_tokens.biases")),
        true,
        static_cast<int>(embed_quantization.group_size),
        static_cast<int>(embed_quantization.bits),
        "affine");
    logits = model_dtype(logits);
    return model_dtype(mlx::core::tanh(logits / model_scalar(30.0f)) * model_scalar(30.0f));
}

NativeForwardArrays last_logits_from_hidden(
    const NativeTextModel::Impl& impl,
    NativeHiddenArrays forward,
    int sequence_len) {
    array last_hidden = mlx::core::slice(forward.hidden, {0, sequence_len - 1, 0}, {1, sequence_len, 3840});
    array logits = target_logits_for_hidden(impl, last_hidden);
    logits = mlx::core::reshape(logits, {262144});
    dump_hidden_tensor("logits", logits);
    return NativeForwardArrays{
        std::move(logits),
        std::move(last_hidden),
        std::move(forward.shared_kv),
        last_token_dspark_taps(std::move(forward.dspark_taps), sequence_len),
        BlockPrefixSources{}};
}

NativeForwardArrays forward_last_logits(const NativeTextModel::Impl& impl, const std::vector<int32_t>& tokens) {
    return last_logits_from_hidden(impl, forward_hidden(impl, tokens), static_cast<int>(tokens.size()));
}

NativeForwardArrays prefill_last_logits(
    const NativeTextModel::Impl& impl,
    const std::vector<int32_t>& tokens,
    NativeKvState::Impl* target_kv) {
    return last_logits_from_hidden(impl, forward_hidden(impl, tokens, target_kv), static_cast<int>(tokens.size()));
}

NativeForwardArrays prefill_chunked_last_logits(
    const NativeTextModel::Impl& impl,
    const std::vector<int32_t>& tokens,
    NativeKvState::Impl* target_kv,
    size_t chunk_tokens) {
    if (chunk_tokens == 0 || chunk_tokens >= tokens.size()) {
        return prefill_last_logits(impl, tokens, target_kv);
    }

    const size_t first_count = std::min(chunk_tokens, tokens.size());
    std::vector<int32_t> first_chunk(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(first_count));
    NativeHiddenArrays forward = forward_hidden(impl, first_chunk, target_kv);
    size_t offset = first_count;
    size_t last_count = first_count;
    while (offset < tokens.size()) {
        const size_t current_count = std::min(chunk_tokens, tokens.size() - offset);
        forward = decode_block_hidden(impl, tokens.data() + offset, current_count, target_kv);
        offset += current_count;
        last_count = current_count;
    }

    return last_logits_from_hidden(impl, std::move(forward), static_cast<int>(last_count));
}

NativeForwardArrays decode_last_logits(
    const NativeTextModel::Impl& impl,
    int32_t token,
    NativeKvState::Impl* target_kv) {
    if (target_kv == nullptr || target_kv->sequence_len == 0 || target_kv->layers.size() != kTargetLayerCount) {
        throw std::runtime_error("native incremental decode requires a populated target KV cache");
    }
    if (target_kv->sequence_len > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("native incremental decode position exceeds MLX shape limits");
    }

    const uint64_t previous_sequence_len = target_kv->sequence_len;
    std::vector<int32_t> ids = {token};
    array token_ids(ids.begin(), {1, 1}, mlx::core::int32);
    array h = model_dtype(quantized_embedding(impl, token_ids) * model_scalar(std::sqrt(3840.0f)));
    SharedKvArrays shared_kv;
    DSparkHiddenTaps dspark_taps;

    for (uint32_t layer = 0; layer < kTargetLayerCount; ++layer) {
        h = target_layer_decode_forward(
            impl,
            h,
            layer,
            previous_sequence_len,
            &target_kv->layers[layer],
            &shared_kv);
        maybe_capture_dspark_tap(impl, layer, h, &dspark_taps);
    }
    append_dspark_context_taps(target_kv, dspark_taps);
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, "language_model.model.norm.weight")),
        1e-6f));
    array logits = target_logits_for_hidden(impl, h);
    logits = mlx::core::reshape(logits, {262144});
    eval_deferred_decode_kv(target_kv, decode_kv_eval_mode());
    target_kv->sequence_len = previous_sequence_len + 1;
    target_kv->active_bytes = estimate_target_kv_bytes(target_kv->sequence_len);
    return NativeForwardArrays{
        std::move(logits),
        std::move(h),
        std::move(shared_kv),
        std::move(dspark_taps),
        BlockPrefixSources{}};
}

NativeHiddenArrays decode_last_hidden(
    const NativeTextModel::Impl& impl,
    int32_t token,
    NativeKvState::Impl* target_kv) {
    if (target_kv == nullptr || target_kv->sequence_len == 0 || target_kv->layers.size() != kTargetLayerCount) {
        throw std::runtime_error("native incremental state advance requires a populated target KV cache");
    }
    if (target_kv->sequence_len > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("native incremental state advance position exceeds MLX shape limits");
    }

    const uint64_t previous_sequence_len = target_kv->sequence_len;
    std::vector<int32_t> ids = {token};
    array token_ids(ids.begin(), {1, 1}, mlx::core::int32);
    array h = model_dtype(quantized_embedding(impl, token_ids) * model_scalar(std::sqrt(3840.0f)));
    SharedKvArrays shared_kv;
    DSparkHiddenTaps dspark_taps;

    for (uint32_t layer = 0; layer < kTargetLayerCount; ++layer) {
        h = target_layer_decode_forward(
            impl,
            h,
            layer,
            previous_sequence_len,
            &target_kv->layers[layer],
            &shared_kv);
        maybe_capture_dspark_tap(impl, layer, h, &dspark_taps);
    }
    append_dspark_context_taps(target_kv, dspark_taps);
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, "language_model.model.norm.weight")),
        1e-6f));
    eval_deferred_decode_kv(target_kv, decode_kv_eval_mode());
    target_kv->sequence_len = previous_sequence_len + 1;
    target_kv->active_bytes = estimate_target_kv_bytes(target_kv->sequence_len);
    return NativeHiddenArrays{std::move(h), std::move(shared_kv), std::move(dspark_taps)};
}

NativeForwardArrays decode_block_logits(
    const NativeTextModel::Impl& impl,
    const int32_t* tokens,
    size_t token_count,
    NativeKvState::Impl* target_kv,
    BlockPrefixSources* prefix_sources = nullptr) {
    if (target_kv == nullptr || target_kv->sequence_len == 0 || target_kv->layers.size() != kTargetLayerCount) {
        throw std::runtime_error("native incremental block decode requires a populated target KV cache");
    }
    if (tokens == nullptr || token_count == 0) {
        throw std::runtime_error("native incremental block decode requires at least one token");
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("native incremental block decode token count exceeds MLX shape limits");
    }
    if (target_kv->sequence_len + token_count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("native incremental block decode position exceeds MLX shape limits");
    }

    const uint64_t previous_sequence_len = target_kv->sequence_len;
    const int block_len = static_cast<int>(token_count);
    std::vector<int32_t> ids(tokens, tokens + token_count);
    array token_ids(ids.begin(), {1, block_len}, mlx::core::int32);
    array h = model_dtype(quantized_embedding(impl, token_ids) * model_scalar(std::sqrt(3840.0f)));
    SharedKvArrays shared_kv;
    DSparkHiddenTaps dspark_taps;
    if (prefix_sources != nullptr) {
        prefix_sources->previous_sequence_len = previous_sequence_len;
        prefix_sources->token_count = token_count;
        prefix_sources->layers.clear();
        prefix_sources->layers.reserve(kTargetLayerCount);
        prefix_sources->previous_dspark_context_taps = target_kv->dspark_context_taps;
        prefix_sources->block_dspark_taps.clear();
    }

    for (uint32_t layer = 0; layer < kTargetLayerCount; ++layer) {
        BlockPrefixSourceLayer* prefix_source_layer = nullptr;
        if (prefix_sources != nullptr) {
            prefix_sources->layers.emplace_back();
            prefix_source_layer = &prefix_sources->layers.back();
        }
        h = target_layer_decode_block_forward(
            impl,
            h,
            layer,
            previous_sequence_len,
            block_len,
            &target_kv->layers[layer],
            &shared_kv,
            prefix_source_layer);
        maybe_capture_dspark_tap(impl, layer, h, &dspark_taps);
    }
    if (prefix_sources != nullptr) {
        prefix_sources->block_dspark_taps = dspark_taps.taps;
    }
    append_dspark_context_taps(target_kv, dspark_taps);
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, "language_model.model.norm.weight")),
        1e-6f));
    array logits = target_logits_for_hidden(impl, h);
    const int stop = static_cast<int>(token_count);
    array last_hidden = mlx::core::slice(h, {0, stop - 1, 0}, {1, stop, 3840});
    eval_deferred_decode_kv(target_kv, decode_kv_eval_mode());
    target_kv->sequence_len = previous_sequence_len + token_count;
    target_kv->active_bytes = estimate_target_kv_bytes(target_kv->sequence_len);
    BlockPrefixSources captured_sources;
    if (prefix_sources != nullptr) {
        captured_sources = std::move(*prefix_sources);
    }
    return NativeForwardArrays{
        std::move(logits),
        std::move(last_hidden),
        std::move(shared_kv),
        last_token_dspark_taps(std::move(dspark_taps), block_len),
        std::move(captured_sources)};
}

void materialize_block_prefix_kv(
    const BlockPrefixSources& sources,
    size_t prefix_token_count,
    NativeKvState::Impl* prefix_kv) {
    if (prefix_kv == nullptr) {
        throw std::runtime_error("native incremental block prefix materialization requires output KV state");
    }
    if (prefix_token_count == 0 || prefix_token_count > sources.token_count) {
        throw std::runtime_error("native incremental block prefix materialization count is invalid");
    }
    if (sources.layers.size() != kTargetLayerCount) {
        throw std::runtime_error("native incremental block prefix materialization has incomplete layer sources");
    }

    prefix_kv->layers.clear();
    prefix_kv->layers.reserve(kTargetLayerCount);
    const DecodeKvEvalMode mode = decode_kv_eval_mode();
    for (const BlockPrefixSourceLayer& source : sources.layers) {
        if (!source.attention_keys.has_value() || !source.attention_values.has_value()) {
            throw std::runtime_error("native incremental block prefix materialization has missing source tensors");
        }
        if (prefix_token_count > static_cast<size_t>(std::numeric_limits<int>::max() - source.previous_key_len)) {
            throw std::runtime_error("native incremental block prefix materialization length exceeds MLX limits");
        }
        const int prefix_key_len = source.previous_key_len + static_cast<int>(prefix_token_count);
        if (prefix_key_len <= 0 || prefix_key_len > source.attention_key_len) {
            throw std::runtime_error("native incremental block prefix materialization KV length is invalid");
        }
        const int prefix_start = (!source.full_attention && prefix_key_len > kSlidingWindowSize)
            ? prefix_key_len - kSlidingWindowSize
            : 0;

        prefix_kv->layers.emplace_back();
        NativeKvState::Impl::Layer& layer = prefix_kv->layers.back();
        layer.full_attention = source.full_attention;
        layer.key = mlx::core::slice(
            *source.attention_keys,
            {0, 0, prefix_start, 0},
            {1, source.n_kv_heads, prefix_key_len, source.head_dim});
        layer.value = mlx::core::slice(
            *source.attention_values,
            {0, 0, prefix_start, 0},
            {1, source.n_kv_heads, prefix_key_len, source.head_dim});
        if (eval_decode_kv_when_stored(mode, source.full_attention)) {
            mlx::core::eval({*layer.key, *layer.value});
        }
    }

    eval_deferred_decode_kv(prefix_kv, mode);
    prefix_kv->sequence_len = sources.previous_sequence_len + prefix_token_count;
    prefix_kv->active_bytes = estimate_target_kv_bytes(prefix_kv->sequence_len);
    prefix_kv->dspark_context_taps = sources.previous_dspark_context_taps;
    if (!sources.block_dspark_taps.empty()) {
        DSparkHiddenTaps prefix_taps;
        prefix_taps.taps = prefix_dspark_context_taps(sources.block_dspark_taps, prefix_token_count);
        append_dspark_context_taps(prefix_kv, prefix_taps);
    }
}

array target_token_embedding(const NativeTextModel::Impl& impl, int32_t token_id) {
    const std::vector<int32_t> ids = {token_id};
    array token_ids(ids.begin(), {1, 1}, mlx::core::int32);
    return model_dtype(quantized_embedding(impl, token_ids) * model_scalar(std::sqrt(3840.0f)));
}

int64_t target_embedding_vocab_size(const NativeTextModel::Impl& impl) {
    const auto& shape = tensor_or_throw(impl, "language_model.model.embed_tokens.weight").shape();
    if (shape.empty()) {
        throw std::runtime_error("target embedding weight has no vocab dimension");
    }
    return static_cast<int64_t>(shape[0]);
}

int64_t assistant_embedding_vocab_size(const NativeMtpAssistantModel::Impl& impl) {
    const auto& shape = tensor_or_throw(impl, "model.embed_tokens.weight").shape();
    if (shape.empty()) {
        throw std::runtime_error("assistant embedding weight has no vocab dimension");
    }
    return static_cast<int64_t>(shape[0]);
}

array assistant_logits(const NativeMtpAssistantModel::Impl& impl, const array& h) {
    const QuantizationSpec embed_quantization = quantization_for(impl, "model.embed_tokens");
    array logits = mlx::core::quantized_matmul(
        h,
        tensor_or_throw(impl, "model.embed_tokens.weight"),
        tensor_or_throw(impl, "model.embed_tokens.scales"),
        std::optional<array>(tensor_or_throw(impl, "model.embed_tokens.biases")),
        true,
        static_cast<int>(embed_quantization.group_size),
        static_cast<int>(embed_quantization.bits),
        "affine");
    logits = model_dtype(logits);
    return mlx::core::reshape(logits, {static_cast<int>(assistant_embedding_vocab_size(impl))});
}

bool assistant_layer_full_attention(uint32_t layer_idx) {
    return layer_idx == 3;
}

array assistant_attention_forward(
    const NativeMtpAssistantModel::Impl& impl,
    const NativeHiddenState::Impl& shared,
    const array& x,
    uint32_t layer_idx,
    int position_offset) {
    const bool full_attention = assistant_layer_full_attention(layer_idx);
    const int head_dim = full_attention ? 512 : 256;
    const int n_heads = 16;
    const std::string base = "model.layers." + std::to_string(layer_idx);

    array queries = quantized_linear(impl, x, base + ".self_attn.q_proj");
    queries = mlx::core::reshape(queries, {1, 1, n_heads, head_dim});
    queries = model_dtype(mlx::core::fast::rms_norm(
        queries,
        std::optional<array>(tensor_or_throw(impl, base + ".self_attn.q_norm.weight")),
        1e-6f));
    queries = mlx::core::transpose(queries, {0, 2, 1, 3});
    queries = apply_rope(queries, full_attention, head_dim, position_offset);

    const array& keys = full_attention ? *shared.full_attention_key : *shared.sliding_attention_key;
    const array& values = full_attention ? *shared.full_attention_value : *shared.sliding_attention_value;
    array output = mlx::core::fast::scaled_dot_product_attention(
        queries,
        keys,
        values,
        1.0f,
        "",
        std::nullopt);
    output = mlx::core::transpose(output, {0, 2, 1, 3});
    output = mlx::core::reshape(output, {1, 1, n_heads * head_dim});
    return quantized_linear(impl, output, base + ".self_attn.o_proj");
}

array assistant_layer_forward(
    const NativeMtpAssistantModel::Impl& impl,
    const NativeHiddenState::Impl& shared,
    const array& x,
    uint32_t layer_idx,
    int position_offset) {
    const std::string base = "model.layers." + std::to_string(layer_idx);
    const array residual = x;

    array h = model_dtype(mlx::core::fast::rms_norm(
        x,
        std::optional<array>(tensor_or_throw(impl, base + ".input_layernorm.weight")),
        1e-6f));
    h = assistant_attention_forward(impl, shared, h, layer_idx, position_offset);
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, base + ".post_attention_layernorm.weight")),
        1e-6f));
    h = model_dtype(residual + h);

    const array mlp_residual = h;
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, base + ".pre_feedforward_layernorm.weight")),
        1e-6f));
    array gate = quantized_linear(impl, h, base + ".mlp.gate_proj");
    array up = quantized_linear(impl, h, base + ".mlp.up_proj");
    h = model_dtype(geglu(gate, up));
    h = quantized_linear(impl, h, base + ".mlp.down_proj");
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, base + ".post_feedforward_layernorm.weight")),
        1e-6f));
    h = model_dtype(mlp_residual + h);
    return model_dtype(h * tensor_or_throw(impl, base + ".layer_scalar"));
}

struct NativeMtpDraftStep {
    int32_t token = 0;
    float logit = 0.0f;
    float margin = 0.0f;
    array projected_hidden;
};

NativeTopKEntries top_k_for_flat_logits(const array& flat_logits, size_t requested_k, bool allow_anchor = false);

bool experimental_mtp_skip_final_projection_enabled() {
    const char* value = std::getenv("GEMMA4D_EXPERIMENTAL_MTP_SKIP_FINAL_PROJECTION");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
        std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0 &&
        std::strcmp(value, "off") != 0 && std::strcmp(value, "OFF") != 0;
}

NativeMtpDraftStep assistant_draft_one(
    const NativeMtpAssistantModel::Impl& assistant,
    const NativeTextModel::Impl& target,
    const NativeHiddenState::Impl& shared,
    const array& current_hidden,
    int32_t token_id,
    int position_offset,
    bool need_projected_hidden) {
    array token_embedding = target_token_embedding(target, token_id);
    array input = mlx::core::concatenate({token_embedding, current_hidden}, 2);
    array h = quantized_linear(assistant, input, "pre_projection");

    for (uint32_t layer = 0; layer < 4; ++layer) {
        h = assistant_layer_forward(assistant, shared, h, layer, position_offset);
    }
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(assistant, "model.norm.weight")),
        1e-6f));

    array logits = assistant_logits(assistant, h);
    array greedy = mlx::core::argmax(logits);
    float top_logit = 0.0f;
    float margin = 0.0f;
    if (experimental_mtp_real_margins_enabled()) {
        NativeTopKEntries draft_top_k = top_k_for_flat_logits(logits, 2);
        top_logit = draft_top_k[0].logit;
        margin = top_logit - draft_top_k[1].logit;
    }
    if (!need_projected_hidden) {
        mlx::core::eval(greedy);
        return NativeMtpDraftStep{greedy.item<int>(), top_logit, margin, std::move(h)};
    }
    array projected = quantized_linear(assistant, h, "post_projection");
    mlx::core::eval({greedy, projected});

    return NativeMtpDraftStep{greedy.item<int>(), top_logit, margin, std::move(projected)};
}

bool trace_parity_logits_enabled() {
    const char* value = std::getenv("GEMMA4D_NATIVE_TRACE_PARITY_LOGITS");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

bool experimental_native_gather_greedy_logit_env_enabled() {
    const char* value = std::getenv("GEMMA4D_EXPERIMENTAL_NATIVE_GATHER_GREEDY_LOGIT");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

size_t native_prefill_chunk_tokens_env() {
    const char* value = std::getenv("GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || (end != nullptr && *end != '\0')) {
        return 0;
    }
    return static_cast<size_t>(parsed);
}

bool native_prefill_policy_long_context_256_env_enabled() {
    const char* value = std::getenv("GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY");
    return value != nullptr && std::string(value) == "long_context_256";
}

size_t selected_native_prefill_chunk_tokens(const NativeTextModel::Impl& impl, size_t token_count) {
    if (impl.native_prefill_chunk_tokens != 0) {
        return impl.native_prefill_chunk_tokens;
    }
    if (impl.native_prefill_policy_long_context_256 && token_count >= 4096) {
        return 256;
    }
    return 0;
}

bool experimental_skip_decode_peak_reset_env_enabled() {
    const char* value = std::getenv("GEMMA4D_EXPERIMENTAL_NATIVE_SKIP_DECODE_PEAK_RESET");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

array greedy_logit_for_vector_logits(const array& logits, const array& greedy, bool use_gather) {
    if (use_gather) {
        return to_float32(mlx::core::take(logits, greedy, 0));
    }
    return to_float32(mlx::core::max(logits));
}

NativeTopKEntries empty_top_k_entries() {
    NativeTopKEntries entries{};
    for (auto& entry : entries) {
        entry.token_id = -1;
        entry.logit = 0.0f;
    }
    return entries;
}

NativeTopKEntries top_k_from_greedy(int32_t token_id, float logit) {
    NativeTopKEntries entries = empty_top_k_entries();
    entries[0] = NativeTopKEntry{token_id, logit};
    return entries;
}

void maybe_save_xr57_target_logits_anchor(
    const array& flat_logits,
    const NativeTopKEntries& trace_top_k,
    size_t requested_k) {
    if (!xr57_target_logits_anchor_armed() || xr57_target_logits_anchor_saved()) {
        return;
    }
    const char* path = std::getenv("GEMMA4D_XR57_TARGET_LOGITS_ANCHOR_PATH");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    const auto& shape = flat_logits.shape();
    if (shape.size() != 1 || shape[0] <= 0) {
        throw std::runtime_error("XR57 target-logits anchor expected flat logits");
    }

    const std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    array logits_f32 = to_float32(flat_logits);
    mlx::core::eval(logits_f32);
    const float* logits = logits_f32.data<float>();
    const int vocab_size = shape[0];

    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("failed to open XR57 target-logits anchor at " + output_path.string());
    }
    out << std::setprecision(9);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"source\": \"native_target_logits\",\n";
    out << "  \"vocab_size\": " << vocab_size << ",\n";
    out << "  \"requested_k\": " << requested_k << ",\n";
    out << "  \"trace_top_token_ids\": [";
    for (size_t rank = 0; rank < GEMMA4_MTP_TRACE_TOP_K; ++rank) {
        if (rank > 0) {
            out << ", ";
        }
        out << trace_top_k[rank].token_id;
    }
    out << "],\n";
    out << "  \"trace_top_logits\": [";
    for (size_t rank = 0; rank < GEMMA4_MTP_TRACE_TOP_K; ++rank) {
        if (rank > 0) {
            out << ", ";
        }
        out << trace_top_k[rank].logit;
    }
    out << "],\n";
    out << "  \"logits\": [";
    for (int index = 0; index < vocab_size; ++index) {
        if (index > 0) {
            out << ", ";
        }
        out << logits[index];
    }
    out << "]\n";
    out << "}\n";
    if (!out) {
        throw std::runtime_error("failed to write XR57 target-logits anchor at " + output_path.string());
    }
    xr57_target_logits_anchor_saved() = true;
    xr57_target_logits_anchor_armed() = false;
}

NativeTopKEntries top_k_for_flat_logits(const array& flat_logits, size_t requested_k, bool allow_anchor) {
    NativeTopKEntries result = empty_top_k_entries();
    const auto& shape = flat_logits.shape();
    if (shape.size() != 1) {
        throw std::runtime_error("native top-k trace expects flat logits");
    }
    const int vocab_size = shape[0];
    if (vocab_size <= 0) {
        throw std::runtime_error("native top-k trace received empty logits");
    }
    const size_t k = std::min<size_t>(requested_k, GEMMA4_MTP_TRACE_TOP_K);
    if (k == 0 || k > static_cast<size_t>(vocab_size)) {
        throw std::runtime_error("native top-k trace requested an invalid k");
    }

    const int partition_start = vocab_size - static_cast<int>(k);
    array partition = mlx::core::argpartition(flat_logits, partition_start);
    array top_ids = mlx::core::astype(
        mlx::core::slice(partition, {partition_start}, {vocab_size}),
        mlx::core::int32);
    array top_logits = to_float32(mlx::core::take(flat_logits, top_ids, 0));
    mlx::core::eval({top_ids, top_logits});

    const int* id_data = top_ids.data<int>();
    const float* logit_data = top_logits.data<float>();
    std::vector<NativeTopKEntry> sorted;
    sorted.reserve(k);
    for (size_t index = 0; index < k; ++index) {
        sorted.push_back(NativeTopKEntry{id_data[index], logit_data[index]});
    }
    std::sort(sorted.begin(), sorted.end(), [](const NativeTopKEntry& left, const NativeTopKEntry& right) {
        if (left.logit == right.logit) {
            return left.token_id < right.token_id;
        }
        return left.logit > right.logit;
    });
    for (size_t index = 0; index < sorted.size(); ++index) {
        result[index] = sorted[index];
    }
    if (allow_anchor) {
        maybe_save_xr57_target_logits_anchor(flat_logits, result, requested_k);
    }
    return result;
}

std::vector<NativeTopKEntries> top_k_for_block_logits(const array& logits, size_t token_count) {
    const auto& shape = logits.shape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] < static_cast<int>(token_count)) {
        throw std::runtime_error("native block top-k trace received unexpected logits shape");
    }
    const int vocab_size = shape[2];
    std::vector<NativeTopKEntries> per_position;
    per_position.reserve(token_count);
    for (size_t index = 0; index < token_count; ++index) {
        array slot_logits = mlx::core::reshape(
            mlx::core::slice(
                logits,
                {0, static_cast<int>(index), 0},
                {1, static_cast<int>(index + 1), vocab_size}),
            {vocab_size});
        per_position.push_back(top_k_for_flat_logits(slot_logits, GEMMA4_MTP_TRACE_TOP_K, true));
    }
    return per_position;
}

array dense_linear(const NativeDSparkModel::Impl& impl, const array& x, const std::string& prefix) {
    return model_dtype(mlx::core::matmul(x, mlx::core::transpose(tensor_or_throw(impl, prefix + ".weight"))));
}

array dense_linear_with_bias(const NativeDSparkModel::Impl& impl, const array& x, const std::string& prefix) {
    return model_dtype(dense_linear(impl, x, prefix) + tensor_or_throw(impl, prefix + ".bias"));
}

array dspark_token_embedding(const NativeDSparkModel::Impl& impl, const std::vector<int32_t>& token_ids) {
    if (token_ids.empty() || token_ids.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("native DSpark embedding received invalid token count");
    }
    array ids(token_ids.begin(), {1, static_cast<int>(token_ids.size())}, mlx::core::int32);
    return model_dtype(
        mlx::core::take(tensor_or_throw(impl, "embed_tokens.weight"), ids, 0) *
        model_scalar(std::sqrt(static_cast<float>(kHiddenSize))));
}

array dspark_project_context(
    const NativeDSparkModel::Impl& impl,
    const std::vector<NativeHiddenState::Impl::DSparkTap>& context_taps,
    int context_len) {
    if (context_taps.size() != GEMMA4_DSPARK_TARGET_TAP_COUNT || context_len <= 0) {
        throw std::runtime_error("native DSpark context projection received invalid tap context");
    }
    std::vector<array> tap_arrays;
    tap_arrays.reserve(context_taps.size());
    for (const NativeHiddenState::Impl::DSparkTap& tap : context_taps) {
        tap_arrays.push_back(tap.hidden);
    }
    array h = mlx::core::concatenate(std::move(tap_arrays), 2);
    const auto& shape = h.shape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != context_len || shape[2] != kDSparkFeatureSize) {
        throw std::runtime_error("native DSpark context projection produced an unexpected feature shape");
    }
    h = dense_linear(impl, h, "fc");
    return model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, "hidden_norm.weight")),
        1e-6f));
}

array dspark_attention_forward(
    const NativeDSparkModel::Impl& impl,
    const array& target_context,
    const array& x,
    uint32_t layer_idx,
    int context_len,
    int block_len,
    int anchor_position) {
    const std::string base = "layers." + std::to_string(layer_idx);

    array queries = dense_linear(impl, x, base + ".self_attn.q_proj");
    queries = mlx::core::reshape(queries, {1, block_len, kDSparkNumHeads, kDSparkHeadDim});
    queries = model_dtype(mlx::core::fast::rms_norm(
        queries,
        std::optional<array>(tensor_or_throw(impl, base + ".self_attn.q_norm.weight")),
        1e-6f));
    queries = mlx::core::transpose(queries, {0, 2, 1, 3});
    queries = apply_rope(queries, true, kDSparkHeadDim, anchor_position);

    array context_keys = dense_linear(impl, target_context, base + ".self_attn.k_proj");
    context_keys = mlx::core::reshape(context_keys, {1, context_len, kDSparkKvHeads, kDSparkHeadDim});
    array context_values = context_keys;
    context_keys = model_dtype(mlx::core::fast::rms_norm(
        context_keys,
        std::optional<array>(tensor_or_throw(impl, base + ".self_attn.k_norm.weight")),
        1e-6f));
    context_keys = mlx::core::transpose(context_keys, {0, 2, 1, 3});
    context_keys = apply_rope(context_keys, true, kDSparkHeadDim, 0);
    context_values = model_dtype(mlx::core::fast::rms_norm(context_values, std::nullopt, 1e-6f));
    context_values = mlx::core::transpose(context_values, {0, 2, 1, 3});

    array noise_keys = dense_linear(impl, x, base + ".self_attn.k_proj");
    noise_keys = mlx::core::reshape(noise_keys, {1, block_len, kDSparkKvHeads, kDSparkHeadDim});
    array noise_values = noise_keys;
    noise_keys = model_dtype(mlx::core::fast::rms_norm(
        noise_keys,
        std::optional<array>(tensor_or_throw(impl, base + ".self_attn.k_norm.weight")),
        1e-6f));
    noise_keys = mlx::core::transpose(noise_keys, {0, 2, 1, 3});
    noise_keys = apply_rope(noise_keys, true, kDSparkHeadDim, anchor_position);
    noise_values = model_dtype(mlx::core::fast::rms_norm(noise_values, std::nullopt, 1e-6f));
    noise_values = mlx::core::transpose(noise_values, {0, 2, 1, 3});

    const int attention_len = context_len + block_len;
    array keys = mlx::core::concatenate({context_keys, noise_keys}, 2);
    array values = mlx::core::concatenate({context_values, noise_values}, 2);
    keys = mlx::core::broadcast_to(keys, {1, kDSparkNumHeads, attention_len, kDSparkHeadDim});
    values = mlx::core::broadcast_to(values, {1, kDSparkNumHeads, attention_len, kDSparkHeadDim});

    array output = mlx::core::fast::scaled_dot_product_attention(
        queries,
        keys,
        values,
        1.0f,
        "",
        std::nullopt);
    output = mlx::core::transpose(output, {0, 2, 1, 3});
    output = mlx::core::reshape(output, {1, block_len, kDSparkNumHeads * kDSparkHeadDim});
    return dense_linear(impl, output, base + ".self_attn.o_proj");
}

array dspark_layer_forward(
    const NativeDSparkModel::Impl& impl,
    const array& target_context,
    const array& x,
    uint32_t layer_idx,
    int context_len,
    int block_len,
    int anchor_position) {
    const std::string base = "layers." + std::to_string(layer_idx);
    const array residual = x;

    array h = model_dtype(mlx::core::fast::rms_norm(
        x,
        std::optional<array>(tensor_or_throw(impl, base + ".input_layernorm.weight")),
        1e-6f));
    h = dspark_attention_forward(impl, target_context, h, layer_idx, context_len, block_len, anchor_position);
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, base + ".post_attention_layernorm.weight")),
        1e-6f));
    h = model_dtype(residual + h);

    const array mlp_residual = h;
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, base + ".pre_feedforward_layernorm.weight")),
        1e-6f));
    array gate = dense_linear(impl, h, base + ".mlp.gate_proj");
    array up = dense_linear(impl, h, base + ".mlp.up_proj");
    h = model_dtype(geglu(gate, up));
    h = dense_linear(impl, h, base + ".mlp.down_proj");
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, base + ".post_feedforward_layernorm.weight")),
        1e-6f));
    h = model_dtype(mlp_residual + h);
    return model_dtype(h * tensor_or_throw(impl, base + ".layer_scalar"));
}

array dspark_forward_hidden(
    const NativeDSparkModel::Impl& impl,
    const std::vector<NativeHiddenState::Impl::DSparkTap>& context_taps,
    int32_t anchor_token_id,
    int context_len,
    int block_len) {
    if (block_len != GEMMA4_DSPARK_MAX_DRAFT_TOKENS) {
        throw std::runtime_error("native DSpark fixed-prefix path expects the released block-7 drafter");
    }
    const int anchor_position = context_len - 1;
    std::vector<int32_t> draft_ids(static_cast<size_t>(block_len), static_cast<int32_t>(impl.mask_token_id));
    draft_ids[0] = anchor_token_id;

    array target_context = dspark_project_context(impl, context_taps, context_len);
    array h = dspark_token_embedding(impl, draft_ids);
    for (uint32_t layer = 0; layer < kDSparkLayerCount; ++layer) {
        h = dspark_layer_forward(impl, target_context, h, layer, context_len, block_len, anchor_position);
    }
    return model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, "norm.weight")),
        1e-6f));
}

array dspark_logits_for_hidden(const NativeDSparkModel::Impl& impl, const array& h) {
    array logits = dense_linear(impl, h, "lm_head");
    return model_dtype(mlx::core::tanh(logits / model_scalar(30.0f)) * model_scalar(30.0f));
}

array dspark_markov_embedding(const NativeDSparkModel::Impl& impl, int32_t token_id) {
    if (token_id < 0 || token_id >= static_cast<int32_t>(impl.vocab_size)) {
        throw std::runtime_error("native DSpark Markov head received an out-of-vocabulary token id");
    }
    const std::vector<int32_t> ids = {token_id};
    array token_ids(ids.begin(), {1}, mlx::core::int32);
    return model_dtype(mlx::core::reshape(
        mlx::core::take(tensor_or_throw(impl, "markov_head.markov_w1.weight"), token_ids, 0),
        {static_cast<int>(impl.markov_rank)}));
}

array dspark_markov_bias(const NativeDSparkModel::Impl& impl, const array& markov_embedding) {
    array bias = mlx::core::matmul(
        mlx::core::reshape(markov_embedding, {1, static_cast<int>(impl.markov_rank)}),
        mlx::core::transpose(tensor_or_throw(impl, "markov_head.markov_w2.weight")));
    return model_dtype(mlx::core::reshape(bias, {static_cast<int>(impl.vocab_size)}));
}

float dspark_confidence(
    const NativeDSparkModel::Impl& impl,
    const array& hidden,
    int step,
    const array& markov_embedding) {
    array slot_hidden = mlx::core::slice(hidden, {0, step, 0}, {1, step + 1, kHiddenSize});
    array markov = mlx::core::reshape(markov_embedding, {1, 1, static_cast<int>(impl.markov_rank)});
    array feature = mlx::core::concatenate({slot_hidden, markov}, 2);
    array confidence = to_float32(mlx::core::sigmoid(dense_linear_with_bias(impl, feature, "confidence_head.proj")));
    mlx::core::eval(confidence);
    return confidence.item<float>();
}

void initialize_single_target_trace(Gemma4MtpTraceInfo* trace, uint64_t sequence_len, uint32_t top_k) {
    std::memset(trace, 0, sizeof(Gemma4MtpTraceInfo));
    trace->context_sequence_len = sequence_len;
    trace->first_position = sequence_len == 0 ? 0 : sequence_len - 1;
    trace->position_count = 1;
    trace->top_k = std::min<uint32_t>(top_k, GEMMA4_MTP_TRACE_TOP_K);
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

void populate_single_target_trace(Gemma4StepResult* out, const NativeTopKEntries& top_k, uint32_t top_k_count) {
    if (out == nullptr) {
        return;
    }
    initialize_single_target_trace(&out->mtp_trace, out->sequence_len, top_k_count);
    out->mtp_trace.position_offsets[0] = out->sequence_len == 0 ? 0 : out->sequence_len - 1;
    out->mtp_trace.target_tokens[0] = out->greedy_token;
    out->mtp_trace.target_logits[0] = out->greedy_logit;
    for (size_t rank = 0; rank < GEMMA4_MTP_TRACE_TOP_K; ++rank) {
        out->mtp_trace.top_token_ids[rank] = top_k[rank].token_id;
        out->mtp_trace.top_logits[rank] = top_k[rank].logit;
    }
}

void trace_parity_logits(const std::vector<int32_t>& tokens, const array& logits) {
    if (!trace_parity_logits_enabled()) {
        return;
    }

    const std::vector<int32_t> candidate_ids = {236761, 236772};
    array candidate_token_ids(candidate_ids.begin(), {static_cast<int>(candidate_ids.size())}, mlx::core::int32);
    array candidate_logits = to_float32(mlx::core::take(logits, candidate_token_ids, 0));
    mlx::core::eval(candidate_logits);

    std::cerr << "gemma4d_native_trace tokens=[";
    for (size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) {
            std::cerr << ',';
        }
        std::cerr << tokens[index];
    }
    std::cerr << "] logits={";
    for (size_t index = 0; index < candidate_ids.size(); ++index) {
        if (index != 0) {
            std::cerr << ',';
        }
        array scalar = mlx::core::slice(
            candidate_logits,
            {static_cast<int>(index)},
            {static_cast<int>(index + 1)});
        mlx::core::eval(scalar);
        std::cerr << candidate_ids[index] << ':' << scalar.item<float>();
    }
    std::cerr << "}\n";
}

std::string bool_metadata(bool value) {
    return value ? "true" : "false";
}

bool metadata_bool(const std::unordered_map<std::string, std::string>& metadata, const std::string& key) {
    const auto found = metadata.find(key);
    if (found == metadata.end()) {
        return false;
    }
    return found->second == "true" || found->second == "1";
}

uint64_t metadata_u64(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::string& key,
    uint64_t fallback = 0) {
    const auto found = metadata.find(key);
    if (found == metadata.end() || found->second.empty()) {
        return fallback;
    }
    return std::stoull(found->second);
}

std::string shape_metadata(const array& value) {
    std::ostringstream out;
    const auto& shape = value.shape();
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            out << 'x';
        }
        out << shape[index];
    }
    return out.str();
}

mlx::core::Shape parse_shape_metadata(const std::string& value) {
    mlx::core::Shape shape;
    std::stringstream input(value);
    std::string part;
    while (std::getline(input, part, 'x')) {
        if (!part.empty()) {
            shape.push_back(static_cast<mlx::core::ShapeElem>(std::stoi(part)));
        }
    }
    if (shape.empty()) {
        throw std::runtime_error("empty tensor shape metadata");
    }
    return shape;
}

std::string compression_mode_label(Gemma4KvMode mode) {
    switch (mode) {
    case GEMMA4_KV_BF16:
        return "bf16";
    case GEMMA4_KV_MLX_AFFINE_Q8:
        return "mlx_affine_q8";
    case GEMMA4_KV_MLX_AFFINE_Q4:
        return "mlx_affine_q4";
    default:
        return "unsupported";
    }
}

Gemma4KvMode compression_mode_from_label(const std::string& value) {
    if (value == "bf16" || value.empty()) {
        return GEMMA4_KV_BF16;
    }
    if (value == "mlx_affine_q8") {
        return GEMMA4_KV_MLX_AFFINE_Q8;
    }
    if (value == "mlx_affine_q4") {
        return GEMMA4_KV_MLX_AFFINE_Q4;
    }
    throw std::runtime_error("unsupported tensor compression mode " + value);
}

bool should_compress_tensor(
    bool full_attention,
    Gemma4KvMode mode,
    bool compress_global_layers,
    bool compress_sliding_layers) {
    if (mode == GEMMA4_KV_BF16) {
        return false;
    }
    return (full_attention && compress_global_layers) ||
        (!full_attention && compress_sliding_layers);
}

array scalar_array(float value) {
    return array(value, mlx::core::float32);
}

array affine_scale(const array& source, float levels) {
    const array minimum = mlx::core::min(source);
    const array maximum = mlx::core::max(source);
    return mlx::core::maximum(
        (maximum - minimum) / scalar_array(levels),
        scalar_array(std::numeric_limits<float>::epsilon()));
}

array affine_quantize(const array& value, Gemma4KvMode mode, array* out_minimum, array* out_scale) {
    const float levels = mode == GEMMA4_KV_MLX_AFFINE_Q8 ? 255.0f : 15.0f;
    const array source = to_float32(value);
    *out_minimum = mlx::core::min(source);
    *out_scale = affine_scale(source, levels);
    const array normalized = mlx::core::round((source - *out_minimum) / *out_scale);
    const array clipped = mlx::core::clip(
        normalized,
        std::optional<array>(scalar_array(0.0f)),
        std::optional<array>(scalar_array(levels)));
    return mlx::core::astype(clipped, mlx::core::uint8);
}

array pack_q4_values(const array& quantized, uint64_t* value_count) {
    array flat = mlx::core::flatten(quantized);
    *value_count = flat.size();
    const size_t padded_count = flat.size() + (flat.size() % 2);
    if (padded_count != flat.size()) {
        flat = mlx::core::concatenate({flat, mlx::core::zeros({1}, mlx::core::uint8)}, 0);
    }

    const array low = mlx::core::slice(
        flat,
        {0},
        {static_cast<int>(padded_count)},
        {2});
    const array high = mlx::core::slice(
        flat,
        {1},
        {static_cast<int>(padded_count)},
        {2});
    const array mask = array(static_cast<uint8_t>(0x0f));
    const array shift = array(static_cast<uint8_t>(4));
    return mlx::core::bitwise_or(
        mlx::core::bitwise_and(low, mask),
        mlx::core::left_shift(mlx::core::bitwise_and(high, mask), shift));
}

array unpack_q4_values(const array& packed, uint64_t value_count, const mlx::core::Shape& shape) {
    const array mask = array(static_cast<uint8_t>(0x0f));
    const array shift = array(static_cast<uint8_t>(4));
    const array low = mlx::core::bitwise_and(packed, mask);
    const array high = mlx::core::bitwise_and(mlx::core::right_shift(packed, shift), mask);
    const array paired = mlx::core::flatten(mlx::core::stack({low, high}, 1));
    const array trimmed = mlx::core::slice(
        paired,
        {0},
        {static_cast<int>(value_count)});
    return mlx::core::reshape(trimmed, shape);
}

void add_encoded_tensor(
    const std::string& name,
    const array& value,
    bool compress,
    Gemma4KvMode mode,
    std::unordered_map<std::string, array>* arrays,
    std::unordered_map<std::string, std::string>* metadata,
    std::vector<array>* eval_arrays,
    uint64_t* compressed_tensor_count) {
    (*metadata)[name + ".shape"] = shape_metadata(value);
    (*metadata)[name + ".bf16_bytes"] = std::to_string(value.size() * 2);
    if (!compress) {
        arrays->insert_or_assign(name, value);
        (*metadata)[name + ".compression_mode"] = "bf16";
        (*metadata)[name + ".encoded_shape"] = shape_metadata(value);
        (*metadata)[name + ".encoded_bytes"] = std::to_string(value.nbytes());
        eval_arrays->push_back(value);
        return;
    }

    array minimum = scalar_array(0.0f);
    array scale = scalar_array(1.0f);
    array quantized = affine_quantize(value, mode, &minimum, &scale);
    (*metadata)[name + ".compression_mode"] = compression_mode_label(mode);
    (*metadata)[name + ".affine_min_name"] = name + ".affine_min";
    (*metadata)[name + ".affine_scale_name"] = name + ".affine_scale";
    arrays->insert_or_assign(name + ".affine_min", minimum);
    arrays->insert_or_assign(name + ".affine_scale", scale);
    eval_arrays->push_back(minimum);
    eval_arrays->push_back(scale);

    if (mode == GEMMA4_KV_MLX_AFFINE_Q4) {
        uint64_t value_count = 0;
        array packed = pack_q4_values(quantized, &value_count);
        (*metadata)[name + ".quantized_value_count"] = std::to_string(value_count);
        (*metadata)[name + ".encoded_shape"] = shape_metadata(packed);
        (*metadata)[name + ".encoded_bytes"] = std::to_string(packed.nbytes());
        arrays->insert_or_assign(name, packed);
        eval_arrays->push_back(packed);
    } else {
        (*metadata)[name + ".quantized_value_count"] = std::to_string(quantized.size());
        (*metadata)[name + ".encoded_shape"] = shape_metadata(quantized);
        (*metadata)[name + ".encoded_bytes"] = std::to_string(quantized.nbytes());
        arrays->insert_or_assign(name, quantized);
        eval_arrays->push_back(quantized);
    }
    *compressed_tensor_count += 1;
}

array decode_encoded_tensor(
    const std::string& name,
    const array& encoded,
    const std::unordered_map<std::string, array>& arrays,
    const std::unordered_map<std::string, std::string>& metadata) {
    const auto found_mode = metadata.find(name + ".compression_mode");
    const Gemma4KvMode mode = found_mode == metadata.end()
        ? GEMMA4_KV_BF16
        : compression_mode_from_label(found_mode->second);
    if (mode == GEMMA4_KV_BF16) {
        return encoded;
    }

    const auto min_name = metadata.find(name + ".affine_min_name");
    const auto scale_name = metadata.find(name + ".affine_scale_name");
    if (min_name == metadata.end() || scale_name == metadata.end()) {
        throw std::runtime_error("compressed tensor " + name + " is missing affine metadata names");
    }
    const auto minimum = arrays.find(min_name->second);
    const auto scale = arrays.find(scale_name->second);
    if (minimum == arrays.end() || scale == arrays.end()) {
        throw std::runtime_error("compressed tensor " + name + " is missing affine min/scale tensors");
    }

    array quantized = encoded;
    if (mode == GEMMA4_KV_MLX_AFFINE_Q4) {
        const uint64_t value_count = metadata_u64(metadata, name + ".quantized_value_count");
        quantized = unpack_q4_values(
            encoded,
            value_count,
            parse_shape_metadata(metadata.at(name + ".shape")));
    }

    const array reconstructed =
        mlx::core::astype(quantized, mlx::core::float32) * scale->second + minimum->second;
    return model_dtype(reconstructed);
}

#endif

} // namespace

void arm_xr57_target_logits_anchor() {
#ifdef GEMMA4D_MLX_AVAILABLE
    xr57_target_logits_anchor_armed() = true;
#endif
}

NativeHiddenState::NativeHiddenState(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

NativeHiddenState::~NativeHiddenState() = default;

NativeHiddenState::NativeHiddenState(NativeHiddenState&&) noexcept = default;

NativeHiddenState& NativeHiddenState::operator=(NativeHiddenState&&) noexcept = default;

uint64_t NativeHiddenState::sequence_len() const {
    return impl_ == nullptr ? 0 : impl_->sequence_len;
}

uint32_t NativeHiddenState::hidden_size() const {
    return impl_ == nullptr ? 0 : impl_->hidden_size;
}

bool NativeHiddenState::has_shared_kv() const {
    if (impl_ == nullptr) {
        return false;
    }
#ifndef GEMMA4D_MLX_AVAILABLE
    return false;
#else
    return impl_->full_attention_key.has_value() && impl_->full_attention_value.has_value() &&
        impl_->sliding_attention_key.has_value() && impl_->sliding_attention_value.has_value();
#endif
}

void NativeHiddenState::fill_dspark_tap_info(Gemma4DSparkTapInfo* out) const {
    if (out == nullptr) {
        return;
    }
    std::memset(out, 0, sizeof(Gemma4DSparkTapInfo));
    if (impl_ == nullptr) {
        return;
    }
    out->has_last_hidden = true;
#ifdef GEMMA4D_MLX_AVAILABLE
    const size_t count = std::min<size_t>(impl_->dspark_taps.size(), GEMMA4_DSPARK_TARGET_TAP_COUNT);
    out->layer_count = static_cast<uint32_t>(count);
    for (size_t index = 0; index < count; ++index) {
        const NativeHiddenState::Impl::DSparkTap& tap = impl_->dspark_taps[index];
        out->layer_ids[index] = tap.layer_id;
        const auto& shape = tap.hidden.shape();
        const size_t rank = std::min<size_t>(shape.size(), GEMMA4_DSPARK_TAP_MAX_RANK);
        out->tap_ranks[index] = static_cast<uint32_t>(rank);
        for (size_t dim = 0; dim < rank; ++dim) {
            out->tap_shapes[(index * GEMMA4_DSPARK_TAP_MAX_RANK) + dim] =
                static_cast<uint64_t>(shape[dim]);
        }
        out->tap_bytes += static_cast<uint64_t>(tap.hidden.nbytes());
    }
#endif
}

std::unique_ptr<NativeHiddenState> NativeHiddenState::clone() const {
    if (impl_ == nullptr) {
        return nullptr;
    }
#ifdef GEMMA4D_MLX_AVAILABLE
    std::unique_ptr<NativeHiddenState::Impl> cloned_impl(new NativeHiddenState::Impl{
        impl_->hidden,
        impl_->full_attention_key,
        impl_->full_attention_value,
        impl_->sliding_attention_key,
        impl_->sliding_attention_value,
        impl_->dspark_taps,
        impl_->sequence_len,
        impl_->hidden_size,
    });
#else
    std::unique_ptr<NativeHiddenState::Impl> cloned_impl(new NativeHiddenState::Impl{
        impl_->sequence_len,
        impl_->hidden_size,
    });
#endif
    return std::unique_ptr<NativeHiddenState>(new NativeHiddenState(std::move(cloned_impl)));
}

NativeKvState::NativeKvState() : impl_(std::make_unique<Impl>()) {}

NativeKvState::~NativeKvState() = default;

NativeKvState::NativeKvState(NativeKvState&&) noexcept = default;

NativeKvState& NativeKvState::operator=(NativeKvState&&) noexcept = default;

void NativeKvState::clear() {
    if (impl_ != nullptr) {
#ifdef GEMMA4D_MLX_AVAILABLE
        impl_->layers.clear();
        impl_->dspark_context_taps.clear();
#endif
        impl_->sequence_len = 0;
        impl_->active_bytes = 0;
    }
}

uint64_t NativeKvState::sequence_len() const {
    return impl_ == nullptr ? 0 : impl_->sequence_len;
}

uint64_t NativeKvState::active_bytes() const {
    return impl_ == nullptr ? 0 : impl_->active_bytes;
}

std::unique_ptr<NativeKvState> NativeKvState::clone() const {
    if (impl_ == nullptr) {
        return nullptr;
    }
    std::unique_ptr<NativeKvState> cloned(new NativeKvState());
#ifdef GEMMA4D_MLX_AVAILABLE
    cloned->impl_->layers = impl_->layers;
    cloned->impl_->dspark_context_taps = impl_->dspark_context_taps;
#endif
    cloned->impl_->sequence_len = impl_->sequence_len;
    cloned->impl_->active_bytes = impl_->active_bytes;
    return cloned;
}

bool NativeKvState::save_safetensors(
    const std::filesystem::path& payload_path,
    const NativeHiddenState* last_hidden,
    const std::unordered_map<std::string, std::string>& metadata,
    std::string* error,
    const NativeTextModel* token_embedding_model,
    const std::vector<int32_t>* token_embedding_token_ids) const {
    if (error == nullptr) {
        return false;
    }
    error->clear();

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)payload_path;
    (void)last_hidden;
    (void)metadata;
    (void)token_embedding_model;
    (void)token_embedding_token_ids;
    *error = "native KV snapshot payload save requires MLX";
    return false;
#else
    try {
        if (impl_ == nullptr || impl_->layers.empty() || impl_->sequence_len == 0) {
            *error = "native KV snapshot payload save requires a populated KV state";
            return false;
        }
        if ((token_embedding_model == nullptr) != (token_embedding_token_ids == nullptr)) {
            *error = "native KV snapshot token embedding export requires both target model and token ids";
            return false;
        }

        std::unordered_map<std::string, array> arrays;
        std::unordered_map<std::string, std::string> payload_metadata = metadata;
        payload_metadata["format"] = "gemma4d_native_kv_snapshot_v1";
        payload_metadata["kv_sequence_len"] = std::to_string(impl_->sequence_len);
        payload_metadata["kv_active_bytes"] = std::to_string(impl_->active_bytes);
        payload_metadata["kv_layer_count"] = std::to_string(impl_->layers.size());

        std::vector<array> eval_arrays;
        eval_arrays.reserve((impl_->layers.size() * 2) + impl_->dspark_context_taps.size() + 5);
        for (size_t index = 0; index < impl_->layers.size(); ++index) {
            const NativeKvState::Impl::Layer& layer = impl_->layers[index];
            const std::string prefix = "kv.layer_" + std::to_string(index);
            payload_metadata[prefix + ".full_attention"] = bool_metadata(layer.full_attention);
            payload_metadata[prefix + ".has_key"] = bool_metadata(layer.key.has_value());
            payload_metadata[prefix + ".has_value"] = bool_metadata(layer.value.has_value());
            if (layer.key.has_value()) {
                const std::string name = prefix + ".key";
                arrays.insert_or_assign(name, *layer.key);
                payload_metadata[name + ".shape"] = shape_metadata(*layer.key);
                eval_arrays.push_back(*layer.key);
            }
            if (layer.value.has_value()) {
                const std::string name = prefix + ".value";
                arrays.insert_or_assign(name, *layer.value);
                payload_metadata[name + ".shape"] = shape_metadata(*layer.value);
                eval_arrays.push_back(*layer.value);
            }
        }
        payload_metadata["dspark_context_tap_count"] = std::to_string(impl_->dspark_context_taps.size());
        for (size_t index = 0; index < impl_->dspark_context_taps.size(); ++index) {
            const NativeHiddenState::Impl::DSparkTap& tap = impl_->dspark_context_taps[index];
            const std::string prefix = "dspark_context.tap_" + std::to_string(index);
            payload_metadata[prefix + ".layer_id"] = std::to_string(tap.layer_id);
            arrays.insert_or_assign(prefix + ".hidden", tap.hidden);
            payload_metadata[prefix + ".hidden.shape"] = shape_metadata(tap.hidden);
            eval_arrays.push_back(tap.hidden);
        }

        if (last_hidden != nullptr && last_hidden->impl_ != nullptr) {
            payload_metadata["hidden_present"] = "true";
            payload_metadata["hidden_sequence_len"] = std::to_string(last_hidden->impl_->sequence_len);
            payload_metadata["hidden_size"] = std::to_string(last_hidden->impl_->hidden_size);
            arrays.insert_or_assign("hidden.last", last_hidden->impl_->hidden);
            payload_metadata["hidden.last.shape"] = shape_metadata(last_hidden->impl_->hidden);
            eval_arrays.push_back(last_hidden->impl_->hidden);

            auto add_hidden = [&](const char* name, const std::optional<array>& value) {
                const std::string key = std::string("hidden.") + name;
                payload_metadata[key + ".present"] = bool_metadata(value.has_value());
                if (value.has_value()) {
                    arrays.insert_or_assign(key, *value);
                    payload_metadata[key + ".shape"] = shape_metadata(*value);
                    eval_arrays.push_back(*value);
                }
            };
            add_hidden("full_attention_key", last_hidden->impl_->full_attention_key);
            add_hidden("full_attention_value", last_hidden->impl_->full_attention_value);
            add_hidden("sliding_attention_key", last_hidden->impl_->sliding_attention_key);
            add_hidden("sliding_attention_value", last_hidden->impl_->sliding_attention_value);
        } else {
            payload_metadata["hidden_present"] = "false";
        }

        if (token_embedding_model != nullptr && token_embedding_token_ids != nullptr) {
            if (token_embedding_model->impl_ == nullptr ||
                token_embedding_model->impl_->language_tensor_count == 0) {
                *error = "native KV snapshot token embedding export requires a loaded native target graph";
                return false;
            }
            if (token_embedding_token_ids->empty()) {
                *error = "native KV snapshot token embedding export requires at least one token id";
                return false;
            }
            std::vector<array> token_embeddings;
            token_embeddings.reserve(token_embedding_token_ids->size());
            std::string token_id_csv;
            int64_t token_embedding_vocab_size = 0;
            try {
                token_embedding_vocab_size =
                    target_embedding_vocab_size(*token_embedding_model->impl_);
            } catch (const std::exception& ex) {
                *error = std::string("native KV snapshot token embedding export failed: ") + ex.what();
                return false;
            }
            for (size_t index = 0; index < token_embedding_token_ids->size(); ++index) {
                if (!token_id_csv.empty()) {
                    token_id_csv.push_back(',');
                }
                const int32_t token_id = (*token_embedding_token_ids)[index];
                if (token_id < 0 ||
                    static_cast<int64_t>(token_id) >= token_embedding_vocab_size) {
                    std::ostringstream message;
                    message
                        << "native KV snapshot token embedding export token id out of bounds: "
                        << token_id << " not in [0," << token_embedding_vocab_size << ")";
                    *error = message.str();
                    return false;
                }
                token_id_csv += std::to_string(token_id);
                token_embeddings.push_back(target_token_embedding(*token_embedding_model->impl_, token_id));
            }
            array token_embedding_batch = mlx::core::concatenate(token_embeddings, 1);
            arrays.insert_or_assign("target.token_embeddings", token_embedding_batch);
            payload_metadata["target.token_embeddings.present"] = "true";
            payload_metadata["target.token_embeddings.token_ids"] = token_id_csv;
            payload_metadata["target.token_embeddings.shape"] = shape_metadata(token_embedding_batch);
            eval_arrays.push_back(token_embedding_batch);
        } else {
            payload_metadata["target.token_embeddings.present"] = "false";
        }

        if (arrays.empty()) {
            *error = "native KV snapshot payload save found no arrays to persist";
            return false;
        }
        if (!eval_arrays.empty()) {
            mlx::core::eval(eval_arrays);
        }
        if (!payload_path.parent_path().empty()) {
            std::filesystem::create_directories(payload_path.parent_path());
        }
        mlx::core::save_safetensors(payload_path.string(), std::move(arrays), std::move(payload_metadata));
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("native KV snapshot payload save failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "native KV snapshot payload save failed with an unknown exception";
        return false;
    }
#endif
}

bool NativeKvState::save_compressed_safetensors(
    const std::filesystem::path& payload_path,
    const NativeHiddenState* last_hidden,
    const std::unordered_map<std::string, std::string>& metadata,
    Gemma4KvMode mode,
    bool compress_global_layers,
    bool compress_sliding_layers,
    std::string* error) const {
    if (error == nullptr) {
        return false;
    }
    error->clear();

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)payload_path;
    (void)last_hidden;
    (void)metadata;
    (void)mode;
    (void)compress_global_layers;
    (void)compress_sliding_layers;
    *error = "native compressed KV snapshot payload save requires MLX";
    return false;
#else
    try {
        if (impl_ == nullptr || impl_->layers.empty() || impl_->sequence_len == 0) {
            *error = "native compressed KV snapshot payload save requires a populated KV state";
            return false;
        }
        if (mode != GEMMA4_KV_BF16 && mode != GEMMA4_KV_MLX_AFFINE_Q8 && mode != GEMMA4_KV_MLX_AFFINE_Q4) {
            *error = "native compressed KV snapshot payload save received an unsupported compression mode";
            return false;
        }

        std::unordered_map<std::string, array> arrays;
        std::unordered_map<std::string, std::string> payload_metadata = metadata;
        payload_metadata["format"] = "gemma4d_native_kv_snapshot_v1";
        payload_metadata["kv_sequence_len"] = std::to_string(impl_->sequence_len);
        payload_metadata["kv_active_bytes"] = std::to_string(impl_->active_bytes);
        payload_metadata["kv_layer_count"] = std::to_string(impl_->layers.size());
        payload_metadata["compression.mode"] = compression_mode_label(mode);
        payload_metadata["compression.algorithm"] =
            mode == GEMMA4_KV_BF16 ? "none" : "mlx_affine_per_tensor_min_scale";
        payload_metadata["compression.compress_global_layers"] = bool_metadata(compress_global_layers);
        payload_metadata["compression.compress_sliding_layers"] = bool_metadata(compress_sliding_layers);
        payload_metadata["compression.active_decode_enabled"] = "false";
        payload_metadata["compression.q4_packing"] =
            mode == GEMMA4_KV_MLX_AFFINE_Q4 ? "packed_two_values_per_u8" : "not_applicable";

        uint64_t compressed_tensor_count = 0;
        uint64_t full_attention_tensor_count = 0;
        uint64_t sliding_attention_tensor_count = 0;
        std::vector<array> eval_arrays;
        eval_arrays.reserve((impl_->layers.size() * 4) + impl_->dspark_context_taps.size() + 5);
        for (size_t index = 0; index < impl_->layers.size(); ++index) {
            const NativeKvState::Impl::Layer& layer = impl_->layers[index];
            const std::string prefix = "kv.layer_" + std::to_string(index);
            payload_metadata[prefix + ".full_attention"] = bool_metadata(layer.full_attention);
            payload_metadata[prefix + ".has_key"] = bool_metadata(layer.key.has_value());
            payload_metadata[prefix + ".has_value"] = bool_metadata(layer.value.has_value());
            if (layer.full_attention) {
                full_attention_tensor_count += layer.key.has_value() ? 1 : 0;
                full_attention_tensor_count += layer.value.has_value() ? 1 : 0;
            } else {
                sliding_attention_tensor_count += layer.key.has_value() ? 1 : 0;
                sliding_attention_tensor_count += layer.value.has_value() ? 1 : 0;
            }
            const bool compress_layer = should_compress_tensor(
                layer.full_attention,
                mode,
                compress_global_layers,
                compress_sliding_layers);
            if (layer.key.has_value()) {
                add_encoded_tensor(
                    prefix + ".key",
                    *layer.key,
                    compress_layer,
                    mode,
                    &arrays,
                    &payload_metadata,
                    &eval_arrays,
                    &compressed_tensor_count);
            }
            if (layer.value.has_value()) {
                add_encoded_tensor(
                    prefix + ".value",
                    *layer.value,
                    compress_layer,
                    mode,
                    &arrays,
                    &payload_metadata,
                    &eval_arrays,
                    &compressed_tensor_count);
            }
        }
        payload_metadata["compression.compressed_tensor_count"] = std::to_string(compressed_tensor_count);
        payload_metadata["compression.full_attention_tensor_count"] = std::to_string(full_attention_tensor_count);
        payload_metadata["compression.sliding_attention_tensor_count"] = std::to_string(sliding_attention_tensor_count);
        payload_metadata["dspark_context_tap_count"] = std::to_string(impl_->dspark_context_taps.size());
        for (size_t index = 0; index < impl_->dspark_context_taps.size(); ++index) {
            const NativeHiddenState::Impl::DSparkTap& tap = impl_->dspark_context_taps[index];
            const std::string prefix = "dspark_context.tap_" + std::to_string(index);
            payload_metadata[prefix + ".layer_id"] = std::to_string(tap.layer_id);
            arrays.insert_or_assign(prefix + ".hidden", tap.hidden);
            payload_metadata[prefix + ".hidden.shape"] = shape_metadata(tap.hidden);
            eval_arrays.push_back(tap.hidden);
        }

        if (last_hidden != nullptr && last_hidden->impl_ != nullptr) {
            payload_metadata["hidden_present"] = "true";
            payload_metadata["hidden_sequence_len"] = std::to_string(last_hidden->impl_->sequence_len);
            payload_metadata["hidden_size"] = std::to_string(last_hidden->impl_->hidden_size);
            arrays.insert_or_assign("hidden.last", last_hidden->impl_->hidden);
            payload_metadata["hidden.last.shape"] = shape_metadata(last_hidden->impl_->hidden);
            eval_arrays.push_back(last_hidden->impl_->hidden);

            auto add_hidden = [&](const char* name, const std::optional<array>& value) {
                const std::string key = std::string("hidden.") + name;
                payload_metadata[key + ".present"] = bool_metadata(value.has_value());
                if (value.has_value()) {
                    arrays.insert_or_assign(key, *value);
                    payload_metadata[key + ".shape"] = shape_metadata(*value);
                    eval_arrays.push_back(*value);
                }
            };
            add_hidden("full_attention_key", last_hidden->impl_->full_attention_key);
            add_hidden("full_attention_value", last_hidden->impl_->full_attention_value);
            add_hidden("sliding_attention_key", last_hidden->impl_->sliding_attention_key);
            add_hidden("sliding_attention_value", last_hidden->impl_->sliding_attention_value);
        } else {
            payload_metadata["hidden_present"] = "false";
        }

        if (arrays.empty()) {
            *error = "native compressed KV snapshot payload save found no arrays to persist";
            return false;
        }
        if (!eval_arrays.empty()) {
            mlx::core::eval(eval_arrays);
        }
        if (!payload_path.parent_path().empty()) {
            std::filesystem::create_directories(payload_path.parent_path());
        }
        mlx::core::save_safetensors(payload_path.string(), std::move(arrays), std::move(payload_metadata));
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("native compressed KV snapshot payload save failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "native compressed KV snapshot payload save failed with an unknown exception";
        return false;
    }
#endif
}

bool NativeKvState::load_safetensors(
    const std::filesystem::path& payload_path,
    std::unique_ptr<NativeKvState>* kv_state,
    std::unique_ptr<NativeHiddenState>* last_hidden,
    std::unordered_map<std::string, std::string>* metadata,
    std::string* error) {
    if (kv_state == nullptr || last_hidden == nullptr || metadata == nullptr || error == nullptr) {
        return false;
    }
    kv_state->reset();
    last_hidden->reset();
    metadata->clear();
    error->clear();

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)payload_path;
    *error = "native KV snapshot payload load requires MLX";
    return false;
#else
    try {
        mlx::core::SafetensorsLoad loaded = mlx::core::load_safetensors(payload_path.string());
        std::unordered_map<std::string, array>& arrays = loaded.first;
        *metadata = std::move(loaded.second);

        const auto format = metadata->find("format");
        if (format == metadata->end() || format->second != "gemma4d_native_kv_snapshot_v1") {
            *error = "native KV snapshot payload has an unsupported format";
            return false;
        }

        const uint64_t layer_count_u64 = metadata_u64(*metadata, "kv_layer_count");
        if (layer_count_u64 == 0 || layer_count_u64 > 4096) {
            *error = "native KV snapshot payload has an invalid layer count";
            return false;
        }
        const size_t layer_count = static_cast<size_t>(layer_count_u64);
        std::unique_ptr<NativeKvState> state(new NativeKvState());
        state->impl_->layers.resize(layer_count);
        state->impl_->sequence_len = metadata_u64(*metadata, "kv_sequence_len");
        state->impl_->active_bytes = metadata_u64(*metadata, "kv_active_bytes");

        for (size_t index = 0; index < layer_count; ++index) {
            const std::string prefix = "kv.layer_" + std::to_string(index);
            NativeKvState::Impl::Layer& layer = state->impl_->layers[index];
            layer.full_attention = metadata_bool(*metadata, prefix + ".full_attention");
            if (metadata_bool(*metadata, prefix + ".has_key")) {
                const auto found = arrays.find(prefix + ".key");
                if (found == arrays.end()) {
                    *error = "native KV snapshot payload is missing " + prefix + ".key";
                    return false;
                }
                layer.key = decode_encoded_tensor(prefix + ".key", found->second, arrays, *metadata);
            }
            if (metadata_bool(*metadata, prefix + ".has_value")) {
                const auto found = arrays.find(prefix + ".value");
                if (found == arrays.end()) {
                    *error = "native KV snapshot payload is missing " + prefix + ".value";
                    return false;
                }
                layer.value = decode_encoded_tensor(prefix + ".value", found->second, arrays, *metadata);
            }
        }
        const uint64_t dspark_tap_count = metadata_u64(*metadata, "dspark_context_tap_count", 0);
        if (dspark_tap_count > GEMMA4_DSPARK_TARGET_TAP_COUNT) {
            *error = "native KV snapshot payload has too many DSpark context taps";
            return false;
        }
        state->impl_->dspark_context_taps.clear();
        state->impl_->dspark_context_taps.reserve(static_cast<size_t>(dspark_tap_count));
        for (size_t index = 0; index < static_cast<size_t>(dspark_tap_count); ++index) {
            const std::string prefix = "dspark_context.tap_" + std::to_string(index);
            const auto found = arrays.find(prefix + ".hidden");
            if (found == arrays.end()) {
                *error = "native KV snapshot payload is missing " + prefix + ".hidden";
                return false;
            }
            state->impl_->dspark_context_taps.push_back(NativeHiddenState::Impl::DSparkTap{
                static_cast<uint32_t>(metadata_u64(*metadata, prefix + ".layer_id")),
                found->second,
            });
        }

        if (metadata_bool(*metadata, "hidden_present")) {
            const auto hidden = arrays.find("hidden.last");
            if (hidden == arrays.end()) {
                *error = "native KV snapshot payload declares hidden state but is missing hidden.last";
                return false;
            }
            auto optional_array = [&](const char* name) -> std::optional<array> {
                const std::string key = std::string("hidden.") + name;
                if (!metadata_bool(*metadata, key + ".present")) {
                    return std::nullopt;
                }
                const auto found = arrays.find(key);
                if (found == arrays.end()) {
                    throw std::runtime_error("native KV snapshot payload is missing " + key);
                }
                return found->second;
            };

            const uint64_t hidden_sequence_len = metadata_u64(*metadata, "hidden_sequence_len");
            std::unique_ptr<NativeHiddenState::Impl> hidden_impl(new NativeHiddenState::Impl{
                hidden->second,
                optional_array("full_attention_key"),
                optional_array("full_attention_value"),
                optional_array("sliding_attention_key"),
                optional_array("sliding_attention_value"),
                last_token_from_dspark_context_taps(state->impl_->dspark_context_taps, hidden_sequence_len),
                hidden_sequence_len,
                static_cast<uint32_t>(metadata_u64(*metadata, "hidden_size")),
            });
            last_hidden->reset(new NativeHiddenState(std::move(hidden_impl)));
        }

        *kv_state = std::move(state);
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("native KV snapshot payload load failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "native KV snapshot payload load failed with an unknown exception";
        return false;
    }
#endif
}

NativeLoraAdapter::NativeLoraAdapter() : impl_(std::make_unique<Impl>()) {}

NativeLoraAdapter::NativeLoraAdapter(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

NativeLoraAdapter::~NativeLoraAdapter() = default;

NativeLoraAdapter::NativeLoraAdapter(NativeLoraAdapter&&) noexcept = default;

NativeLoraAdapter& NativeLoraAdapter::operator=(NativeLoraAdapter&&) noexcept = default;

bool NativeLoraAdapter::load_peft(
    const std::filesystem::path& adapter_path,
    const std::string& adapter_id,
    const std::string& adapter_weight_hash,
    uint32_t rank,
    float alpha,
    const std::vector<std::string>& target_modules,
    const NativeTextModel& target_model,
    std::shared_ptr<const NativeLoraAdapter>* out,
    uint64_t* load_latency_us,
    std::string* error) {
    if (out == nullptr || error == nullptr || load_latency_us == nullptr) {
        return false;
    }
    out->reset();
    *load_latency_us = 0;
    error->clear();
    const auto started = std::chrono::steady_clock::now();

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)adapter_path;
    (void)adapter_id;
    (void)adapter_weight_hash;
    (void)rank;
    (void)alpha;
    (void)target_modules;
    (void)target_model;
    *error = "native LoRA adapter loading requires an MLX build";
    return false;
#else
    try {
        if (adapter_id.empty()) {
            *error = "adapter_id must not be empty";
            return false;
        }
        if (adapter_weight_hash.empty()) {
            *error = "adapter_weight_hash must not be empty";
            return false;
        }
        if (rank == 0) {
            *error = "LoRA rank must be greater than zero";
            return false;
        }
        if (!(alpha > 0.0f) || !std::isfinite(alpha)) {
            *error = "LoRA alpha must be finite and positive";
            return false;
        }
        if (target_modules.empty()) {
            *error = "LoRA target_modules must not be empty";
            return false;
        }
        if (target_model.impl_ == nullptr || target_model.impl_->language_tensor_count == 0) {
            *error = "native target model must be loaded before adapter shape validation";
            return false;
        }

        const std::filesystem::path weights_path = adapter_path / "adapter_model.safetensors";
        if (!std::filesystem::exists(weights_path)) {
            *error = "adapter_model.safetensors not found at " + weights_path.string();
            return false;
        }

        auto loaded = mlx::core::load_safetensors(weights_path.string());
        std::unordered_map<std::string, array> lora_a;
        std::unordered_map<std::string, array> lora_b;
        for (const auto& entry : loaded.first) {
            if (const std::optional<std::string> prefix =
                    lora_tensor_prefix(entry.first, ".lora_A.weight")) {
                if (target_module_allowed(*prefix, target_modules)) {
                    lora_a.emplace(*prefix, entry.second);
                }
            } else if (const std::optional<std::string> prefix =
                           lora_tensor_prefix(entry.first, ".lora_B.weight")) {
                if (target_module_allowed(*prefix, target_modules)) {
                    lora_b.emplace(*prefix, entry.second);
                }
            }
        }

        std::unique_ptr<Impl> impl(new Impl());
        impl->adapter_id = adapter_id;
        impl->adapter_weight_hash = adapter_weight_hash;
        impl->target_modules = target_modules;
        impl->rank = rank;
        impl->alpha = alpha;
        const float scale = alpha / static_cast<float>(rank);

        std::vector<array> eval_arrays;
        for (const auto& entry : lora_a) {
            const std::string& prefix = entry.first;
            const auto b_found = lora_b.find(prefix);
            if (b_found == lora_b.end()) {
                *error = "missing lora_B tensor for " + prefix;
                return false;
            }
            const auto weight_found = target_model.impl_->tensors.find(prefix + ".weight");
            if (weight_found == target_model.impl_->tensors.end()) {
                *error = "adapter target module does not exist in native model: " + prefix;
                return false;
            }

            const array& a = entry.second;
            const array& b = b_found->second;
            const auto& a_shape = a.shape();
            const auto& b_shape = b.shape();
            const auto& weight_shape = weight_found->second.shape();
            if (a_shape.size() != 2 || b_shape.size() != 2 || weight_shape.size() != 2) {
                *error = "LoRA tensors and target weight must be rank-2 for " + prefix;
                return false;
            }
            const QuantizationSpec spec = quantization_for(*target_model.impl_, prefix);
            const uint64_t expected_in = quantized_linear_input_dim(spec, weight_found->second);
            const uint64_t expected_out = static_cast<uint64_t>(weight_shape[0]);
            if (static_cast<uint64_t>(a_shape[0]) != rank ||
                static_cast<uint64_t>(a_shape[1]) != expected_in ||
                static_cast<uint64_t>(b_shape[0]) != expected_out ||
                static_cast<uint64_t>(b_shape[1]) != rank) {
                std::ostringstream message;
                message << "LoRA shape mismatch for " << prefix
                        << ": A=[" << a_shape[0] << ',' << a_shape[1]
                        << "] B=[" << b_shape[0] << ',' << b_shape[1]
                        << "] expected A=[" << rank << ',' << expected_in
                        << "] B=[" << expected_out << ',' << rank << ']';
                *error = message.str();
                return false;
            }

            NativeLoraAdapter::Impl::Module module{
                prefix,
                target_module_for_prefix(prefix),
                mlx::core::transpose(to_float32(a), {1, 0}),
                mlx::core::transpose(to_float32(b), {1, 0}),
                scale,
                static_cast<uint64_t>(a.nbytes() + b.nbytes()),
            };
            impl->resident_bytes += module.resident_bytes;
            eval_arrays.push_back(module.a_transposed);
            eval_arrays.push_back(module.b_transposed);
            impl->modules.push_back(std::move(module));
        }

        if (impl->modules.empty()) {
            *error = "adapter contains no supported LoRA tensor pairs for requested target_modules";
            return false;
        }
        for (const std::string& requested : target_modules) {
            const std::string trimmed = trim_ascii(requested);
            if (trimmed.empty()) {
                continue;
            }
            bool covered = false;
            for (const NativeLoraAdapter::Impl::Module& module : impl->modules) {
                if (target_module_allowed(module.prefix, {trimmed})) {
                    covered = true;
                    break;
                }
            }
            if (!covered) {
                *error = "adapter contains no LoRA tensor pair for target_module " + trimmed;
                return false;
            }
        }

        mlx::core::eval(eval_arrays);
        *load_latency_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count());
        std::unique_ptr<NativeLoraAdapter> adapter(new NativeLoraAdapter(std::move(impl)));
        *out = std::shared_ptr<const NativeLoraAdapter>(std::move(adapter));
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("MLX native LoRA adapter load failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "MLX native LoRA adapter load failed with an unknown exception";
        return false;
    }
#endif
}

const std::string& NativeLoraAdapter::adapter_id() const {
    static const std::string empty;
    return impl_ == nullptr ? empty : impl_->adapter_id;
}

const std::string& NativeLoraAdapter::adapter_weight_hash() const {
    static const std::string empty;
    return impl_ == nullptr ? empty : impl_->adapter_weight_hash;
}

size_t NativeLoraAdapter::module_count() const {
#ifdef GEMMA4D_MLX_AVAILABLE
    return impl_ == nullptr ? 0 : impl_->modules.size();
#else
    return 0;
#endif
}

uint64_t NativeLoraAdapter::resident_bytes() const {
    return impl_ == nullptr ? 0 : impl_->resident_bytes;
}

const NativeLoraAdapter::Impl* NativeLoraAdapter::impl() const {
    return impl_.get();
}

NativeTextModel::NativeTextModel() : impl_(std::make_unique<Impl>()) {}

NativeTextModel::~NativeTextModel() = default;

NativeTextModel::NativeTextModel(NativeTextModel&&) noexcept = default;

NativeTextModel& NativeTextModel::operator=(NativeTextModel&&) noexcept = default;

bool NativeTextModel::load(
    const std::filesystem::path& model_path,
    const Gemma4ModelManifest& manifest,
    std::unique_ptr<NativeTextModel>* out,
    std::string* error) {
    if (out == nullptr || error == nullptr) {
        return false;
    }
    out->reset();
    error->clear();

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)model_path;
    (void)manifest;
    *error = "native Gemma 4 graph was requested, but gemma4_mlx was not built with MLX";
    return false;
#else
    try {
        std::unique_ptr<NativeTextModel> model(new NativeTextModel());
        model->impl_->manifest_summary = manifest.summary();
        model->impl_->default_quantization = manifest.default_quantization();
        model->impl_->quantization_overrides = manifest.quantization_overrides;
        model->impl_->experimental_gather_greedy_logit =
            experimental_native_gather_greedy_logit_env_enabled();
        model->impl_->native_prefill_chunk_tokens = native_prefill_chunk_tokens_env();
        model->impl_->native_prefill_policy_long_context_256 =
            native_prefill_policy_long_context_256_env_enabled();
        model->impl_->experimental_skip_decode_peak_reset =
            experimental_skip_decode_peak_reset_env_enabled();

        const std::vector<std::filesystem::path> files = safetensor_files(model_path);
        if (files.empty()) {
            *error = "no safetensors files found in " + model_path.string();
            return false;
        }

        for (const std::filesystem::path& file : files) {
            auto loaded = mlx::core::load_safetensors(file.string());
            ++model->impl_->safetensor_file_count;
            model->impl_->total_tensor_count_seen += loaded.first.size();
            for (auto& entry : loaded.first) {
                if (!is_language_tensor(entry.first)) {
                    continue;
                }
                auto inserted = model->impl_->tensors.emplace(std::move(entry.first), std::move(entry.second));
                if (!inserted.second) {
                    *error = "duplicate language tensor while loading " + file.string();
                    return false;
                }
            }
        }

        model->impl_->language_tensor_count = model->impl_->tensors.size();
        if (model->impl_->safetensor_file_count != manifest.safetensor_file_count ||
            model->impl_->total_tensor_count_seen != manifest.total_tensor_count ||
            model->impl_->language_tensor_count != manifest.language_tensor_count) {
            std::ostringstream message;
            message << "native loaded tensor inventory does not match manifest: files="
                    << model->impl_->safetensor_file_count << " tensors="
                    << model->impl_->total_tensor_count_seen << " language_tensors="
                    << model->impl_->language_tensor_count;
            *error = message.str();
            return false;
        }

        *out = std::move(model);
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("MLX native model load failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "MLX native model load failed with an unknown exception";
        return false;
    }
#endif
}

size_t NativeTextModel::tensor_count() const {
    return impl_ == nullptr ? 0 : impl_->language_tensor_count;
}

std::string NativeTextModel::summary() const {
    if (impl_ == nullptr) {
        return "native Gemma 4 text model is empty";
    }
    std::ostringstream out;
    out << "native Gemma 4 text model loaded " << impl_->language_tensor_count
        << " language tensors from " << impl_->safetensor_file_count
        << " safetensor files (" << impl_->total_tensor_count_seen << " tensors scanned)";
    if (!impl_->manifest_summary.empty()) {
        out << "; " << impl_->manifest_summary;
    }
    return out.str();
}

void NativeTextModel::set_prefill_chunk_policy(const Gemma4PrefillChunkPolicy& policy) {
#ifdef GEMMA4D_MLX_AVAILABLE
    if (impl_ == nullptr) {
        return;
    }
    switch (policy.mode) {
        case GEMMA4_PREFILL_CHUNK_FIXED_TOKENS:
            impl_->native_prefill_chunk_tokens = policy.fixed_chunk_tokens;
            impl_->native_prefill_policy_long_context_256 = false;
            break;
        case GEMMA4_PREFILL_CHUNK_LONG_CONTEXT_256:
            impl_->native_prefill_chunk_tokens = 0;
            impl_->native_prefill_policy_long_context_256 = true;
            break;
        case GEMMA4_PREFILL_CHUNK_DISABLED:
        default:
            impl_->native_prefill_chunk_tokens = 0;
            impl_->native_prefill_policy_long_context_256 = false;
            break;
    }
#else
    (void)policy;
#endif
}

void NativeTextModel::set_dspark_taps(const uint32_t* layer_ids, size_t layer_count) {
#ifdef GEMMA4D_MLX_AVAILABLE
    if (impl_ == nullptr) {
        return;
    }
    impl_->dspark_tap_layer_ids.clear();
    if (layer_ids == nullptr || layer_count == 0) {
        return;
    }
    impl_->dspark_tap_layer_ids.assign(layer_ids, layer_ids + layer_count);
#else
    (void)layer_ids;
    (void)layer_count;
#endif
}

bool NativeTextModel::set_adapter(std::shared_ptr<const NativeLoraAdapter> adapter, std::string* error) {
    if (error == nullptr) {
        return false;
    }
    error->clear();
#ifndef GEMMA4D_MLX_AVAILABLE
    (void)adapter;
    *error = "native LoRA adapter activation requires an MLX build";
    return false;
#else
    if (impl_ == nullptr || impl_->language_tensor_count == 0) {
        *error = "native Gemma 4 model state is not loaded";
        return false;
    }
    if (!adapter || adapter->module_count() == 0) {
        *error = "cannot activate an empty LoRA adapter";
        return false;
    }
    impl_->active_adapter = std::move(adapter);
    return true;
#endif
}

void NativeTextModel::clear_adapter() {
#ifdef GEMMA4D_MLX_AVAILABLE
    if (impl_ != nullptr) {
        impl_->active_adapter.reset();
    }
#endif
}

bool NativeTextModel::has_adapter() const {
#ifdef GEMMA4D_MLX_AVAILABLE
    return impl_ != nullptr && impl_->active_adapter != nullptr;
#else
    return false;
#endif
}

std::string NativeTextModel::active_adapter_id() const {
#ifdef GEMMA4D_MLX_AVAILABLE
    if (impl_ == nullptr || !impl_->active_adapter) {
        return std::string();
    }
    return impl_->active_adapter->adapter_id();
#else
    return std::string();
#endif
}

size_t NativeTextModel::active_adapter_module_count() const {
#ifdef GEMMA4D_MLX_AVAILABLE
    if (impl_ == nullptr || !impl_->active_adapter) {
        return 0;
    }
    return impl_->active_adapter->module_count();
#else
    return 0;
#endif
}

uint64_t NativeTextModel::active_adapter_resident_bytes() const {
#ifdef GEMMA4D_MLX_AVAILABLE
    if (impl_ == nullptr || !impl_->active_adapter) {
        return 0;
    }
    return impl_->active_adapter->resident_bytes();
#else
    return 0;
#endif
}

bool NativeTextModel::forward_greedy(
    const std::vector<int32_t>& tokens,
    Gemma4StepResult* out,
    std::string* error,
    std::unique_ptr<NativeHiddenState>* last_hidden) const {
    if (out == nullptr || error == nullptr) {
        return false;
    }
    *out = Gemma4StepResult{};
    error->clear();
    if (last_hidden != nullptr) {
        last_hidden->reset();
    }

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)tokens;
    *error = "native Gemma 4 graph was requested, but gemma4_mlx was not built with MLX";
    return false;
#else
    try {
        if (impl_ == nullptr || impl_->language_tensor_count == 0) {
            *error = "native Gemma 4 model state is not loaded";
            return false;
        }
        mlx::core::reset_peak_memory();
        NativeForwardArrays forward = forward_last_logits(*impl_, tokens);
        array logits = std::move(forward.logits);
        array greedy = mlx::core::argmax(logits);
        array max_logit =
            greedy_logit_for_vector_logits(logits, greedy, impl_->experimental_gather_greedy_logit);
        const bool real_margins = experimental_mtp_real_margins_enabled();
        NativeTopKEntries target_top_k = real_margins
            ? top_k_for_flat_logits(logits, GEMMA4_MTP_TRACE_TOP_K, true)
            : empty_top_k_entries();
        mlx::core::eval({greedy, max_logit});
        trace_parity_logits(tokens, logits);

        std::unique_ptr<NativeHiddenState> hidden;
        if (last_hidden != nullptr) {
            std::unique_ptr<NativeHiddenState::Impl> hidden_impl(new NativeHiddenState::Impl{
                std::move(forward.last_hidden),
                std::move(forward.shared_kv.full_attention_key),
                std::move(forward.shared_kv.full_attention_value),
                std::move(forward.shared_kv.sliding_attention_key),
                std::move(forward.shared_kv.sliding_attention_value),
                std::move(forward.dspark_taps.taps),
                static_cast<uint64_t>(tokens.size()),
                3840,
            });
            hidden.reset(new NativeHiddenState(std::move(hidden_impl)));
        }

        out->greedy_token = greedy.item<int>();
        out->greedy_logit = max_logit.item<float>();
        out->sequence_len = tokens.size();
        out->peak_memory_gb = static_cast<float>(mlx::core::get_peak_memory()) / 1'000'000'000.0f;
        out->peak_rss_mb = 0.0f;
        out->native_last_hidden = hidden.get();
        if (!real_margins) {
            target_top_k = top_k_from_greedy(out->greedy_token, out->greedy_logit);
        }
        populate_single_target_trace(out, target_top_k, real_margins ? GEMMA4_MTP_TRACE_TOP_K : 1);
        if (last_hidden != nullptr) {
            *last_hidden = std::move(hidden);
        }
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("native Gemma 4 forward failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "native Gemma 4 forward failed with an unknown exception";
        return false;
    }
#endif
}

bool NativeTextModel::prefill_incremental(
    const std::vector<int32_t>& tokens,
    Gemma4StepResult* out,
    std::string* error,
    std::unique_ptr<NativeKvState>* kv_state,
    std::unique_ptr<NativeHiddenState>* last_hidden) const {
    if (out == nullptr || error == nullptr || kv_state == nullptr) {
        return false;
    }
    *out = Gemma4StepResult{};
    error->clear();
    kv_state->reset();
    if (last_hidden != nullptr) {
        last_hidden->reset();
    }

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)tokens;
    *error = "native Gemma 4 graph was requested, but gemma4_mlx was not built with MLX";
    return false;
#else
    try {
        if (impl_ == nullptr || impl_->language_tensor_count == 0) {
            *error = "native Gemma 4 model state is not loaded";
            return false;
        }
        if (tokens.empty()) {
            *error = "native incremental prefill requires at least one token";
            return false;
        }
        if (tokens.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            *error = "native incremental prefill token count exceeds MLX shape limits";
            return false;
        }

        mlx::core::reset_peak_memory();
        std::unique_ptr<NativeKvState> state(new NativeKvState());
        NativeForwardArrays forward = prefill_chunked_last_logits(
            *impl_,
            tokens,
            state->impl_.get(),
            selected_native_prefill_chunk_tokens(*impl_, tokens.size()));
        array logits = std::move(forward.logits);
        array greedy = mlx::core::argmax(logits);
        array max_logit =
            greedy_logit_for_vector_logits(logits, greedy, impl_->experimental_gather_greedy_logit);
        const bool real_margins = experimental_mtp_real_margins_enabled();
        NativeTopKEntries target_top_k = real_margins
            ? top_k_for_flat_logits(logits, GEMMA4_MTP_TRACE_TOP_K, true)
            : empty_top_k_entries();
        mlx::core::eval({greedy, max_logit});
        trace_parity_logits(tokens, logits);

        std::unique_ptr<NativeHiddenState> hidden;
        if (last_hidden != nullptr) {
            std::unique_ptr<NativeHiddenState::Impl> hidden_impl(new NativeHiddenState::Impl{
                std::move(forward.last_hidden),
                std::move(forward.shared_kv.full_attention_key),
                std::move(forward.shared_kv.full_attention_value),
                std::move(forward.shared_kv.sliding_attention_key),
                std::move(forward.shared_kv.sliding_attention_value),
                std::move(forward.dspark_taps.taps),
                static_cast<uint64_t>(tokens.size()),
                kHiddenSize,
            });
            hidden.reset(new NativeHiddenState(std::move(hidden_impl)));
        }

        out->greedy_token = greedy.item<int>();
        out->greedy_logit = max_logit.item<float>();
        out->sequence_len = tokens.size();
        out->active_kv_bytes = state->active_bytes();
        out->peak_memory_gb = static_cast<float>(mlx::core::get_peak_memory()) / 1'000'000'000.0f;
        out->peak_rss_mb = 0.0f;
        out->native_last_hidden = hidden.get();
        if (!real_margins) {
            target_top_k = top_k_from_greedy(out->greedy_token, out->greedy_logit);
        }
        populate_single_target_trace(out, target_top_k, real_margins ? GEMMA4_MTP_TRACE_TOP_K : 1);
        *kv_state = std::move(state);
        if (last_hidden != nullptr) {
            *last_hidden = std::move(hidden);
        }
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("native Gemma 4 incremental prefill failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "native Gemma 4 incremental prefill failed with an unknown exception";
        return false;
    }
#endif
}

bool NativeTextModel::decode_incremental(
    int32_t token,
    NativeKvState* kv_state,
    Gemma4StepResult* out,
    std::string* error,
    std::unique_ptr<NativeHiddenState>* last_hidden,
    NativeTopKEntries* target_top_k) const {
    if (out == nullptr || error == nullptr || kv_state == nullptr) {
        return false;
    }
    *out = Gemma4StepResult{};
    error->clear();
    if (last_hidden != nullptr) {
        last_hidden->reset();
    }

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)token;
    *error = "native Gemma 4 graph was requested, but gemma4_mlx was not built with MLX";
    return false;
#else
    try {
        if (impl_ == nullptr || impl_->language_tensor_count == 0) {
            *error = "native Gemma 4 model state is not loaded";
            return false;
        }
        if (kv_state->impl_ == nullptr || kv_state->sequence_len() == 0) {
            *error = "native incremental decode requires a prior native prefill";
            return false;
        }

        if (!impl_->experimental_skip_decode_peak_reset) {
            mlx::core::reset_peak_memory();
        }
        NativeForwardArrays forward = decode_last_logits(*impl_, token, kv_state->impl_.get());
        array logits = std::move(forward.logits);
        array greedy = mlx::core::argmax(logits);
        array max_logit =
            greedy_logit_for_vector_logits(logits, greedy, impl_->experimental_gather_greedy_logit);
        const bool real_margins = experimental_mtp_real_margins_enabled();
        NativeTopKEntries computed_top_k = real_margins
            ? top_k_for_flat_logits(logits, GEMMA4_MTP_TRACE_TOP_K, true)
            : empty_top_k_entries();
        mlx::core::eval({greedy, max_logit});

        std::unique_ptr<NativeHiddenState> hidden;
        if (last_hidden != nullptr) {
            std::unique_ptr<NativeHiddenState::Impl> hidden_impl(new NativeHiddenState::Impl{
                std::move(forward.last_hidden),
                std::move(forward.shared_kv.full_attention_key),
                std::move(forward.shared_kv.full_attention_value),
                std::move(forward.shared_kv.sliding_attention_key),
                std::move(forward.shared_kv.sliding_attention_value),
                std::move(forward.dspark_taps.taps),
                kv_state->sequence_len(),
                kHiddenSize,
            });
            hidden.reset(new NativeHiddenState(std::move(hidden_impl)));
        }

        out->greedy_token = greedy.item<int>();
        out->greedy_logit = max_logit.item<float>();
        out->sequence_len = kv_state->sequence_len();
        out->active_kv_bytes = kv_state->active_bytes();
        out->peak_memory_gb = static_cast<float>(mlx::core::get_peak_memory()) / 1'000'000'000.0f;
        out->peak_rss_mb = 0.0f;
        out->native_last_hidden = hidden.get();
        if (!real_margins) {
            computed_top_k = top_k_from_greedy(out->greedy_token, out->greedy_logit);
        }
        populate_single_target_trace(out, computed_top_k, real_margins ? GEMMA4_MTP_TRACE_TOP_K : 1);
        if (target_top_k != nullptr) {
            *target_top_k = computed_top_k;
        }
        if (last_hidden != nullptr) {
            *last_hidden = std::move(hidden);
        }
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("native Gemma 4 incremental decode failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "native Gemma 4 incremental decode failed with an unknown exception";
        return false;
    }
#endif
}

bool NativeTextModel::decode_incremental_state_only(
    int32_t token,
    NativeKvState* kv_state,
    Gemma4StepResult* out,
    std::string* error) const {
    if (out == nullptr || error == nullptr || kv_state == nullptr) {
        return false;
    }
    *out = Gemma4StepResult{};
    error->clear();

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)token;
    *error = "native Gemma 4 graph was requested, but gemma4_mlx was not built with MLX";
    return false;
#else
    try {
        if (impl_ == nullptr || impl_->language_tensor_count == 0) {
            *error = "native Gemma 4 model state is not loaded";
            return false;
        }
        if (kv_state->impl_ == nullptr || kv_state->sequence_len() == 0) {
            *error = "native incremental state advance requires a prior native prefill";
            return false;
        }

        mlx::core::reset_peak_memory();
        NativeHiddenArrays forward = decode_last_hidden(*impl_, token, kv_state->impl_.get());
        mlx::core::eval(forward.hidden);

        out->sequence_len = kv_state->sequence_len();
        out->active_kv_bytes = kv_state->active_bytes();
        out->peak_memory_gb = static_cast<float>(mlx::core::get_peak_memory()) / 1'000'000'000.0f;
        out->peak_rss_mb = 0.0f;
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("native Gemma 4 incremental state advance failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "native Gemma 4 incremental state advance failed with an unknown exception";
        return false;
    }
#endif
}

bool NativeTextModel::decode_incremental_block(
    const int32_t* tokens,
    size_t token_count,
    NativeKvState* kv_state,
    Gemma4StepResult* out,
    std::vector<int32_t>* greedy_tokens,
    std::vector<float>* greedy_logits,
    std::string* error,
    std::unique_ptr<NativeHiddenState>* last_hidden,
    std::vector<NativeTopKEntries>* target_top_k) const {
    if (out == nullptr || greedy_tokens == nullptr || greedy_logits == nullptr || error == nullptr || kv_state == nullptr) {
        return false;
    }
    *out = Gemma4StepResult{};
    greedy_tokens->clear();
    greedy_logits->clear();
    error->clear();
    if (last_hidden != nullptr) {
        last_hidden->reset();
    }

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)tokens;
    (void)token_count;
    *error = "native Gemma 4 graph was requested, but gemma4_mlx was not built with MLX";
    return false;
#else
    try {
        if (impl_ == nullptr || impl_->language_tensor_count == 0) {
            *error = "native Gemma 4 model state is not loaded";
            return false;
        }
        if (tokens == nullptr || token_count == 0) {
            *error = "native incremental block decode requires at least one token";
            return false;
        }
        if (token_count > GEMMA4_MTP_MAX_DRAFT_TOKENS) {
            std::ostringstream message;
            message << "native incremental block decode supports token_count <= "
                    << GEMMA4_MTP_MAX_DRAFT_TOKENS;
            *error = message.str();
            return false;
        }
        if (kv_state->impl_ == nullptr || kv_state->sequence_len() == 0) {
            *error = "native incremental block decode requires a prior native prefill";
            return false;
        }

        mlx::core::reset_peak_memory();
        NativeForwardArrays forward = decode_block_logits(*impl_, tokens, token_count, kv_state->impl_.get());
        array logits = std::move(forward.logits);
        array greedy = mlx::core::argmax(logits, -1);
        array max_logits = to_float32(mlx::core::max(logits, -1));
        const bool real_margins = experimental_mtp_real_margins_enabled();
        std::vector<NativeTopKEntries> computed_top_k;
        if (real_margins) {
            computed_top_k = top_k_for_block_logits(logits, token_count);
        }
        mlx::core::eval({greedy, max_logits, forward.last_hidden});

        const int* token_data = greedy.data<int>();
        const float* logit_data = max_logits.data<float>();
        greedy_tokens->reserve(token_count);
        greedy_logits->reserve(token_count);
        for (size_t index = 0; index < token_count; ++index) {
            greedy_tokens->push_back(token_data[index]);
            greedy_logits->push_back(logit_data[index]);
        }
        if (!real_margins) {
            computed_top_k.reserve(token_count);
            for (size_t index = 0; index < token_count; ++index) {
                computed_top_k.push_back(top_k_from_greedy((*greedy_tokens)[index], (*greedy_logits)[index]));
            }
        }

        std::unique_ptr<NativeHiddenState> hidden;
        if (last_hidden != nullptr) {
            std::unique_ptr<NativeHiddenState::Impl> hidden_impl(new NativeHiddenState::Impl{
                std::move(forward.last_hidden),
                std::move(forward.shared_kv.full_attention_key),
                std::move(forward.shared_kv.full_attention_value),
                std::move(forward.shared_kv.sliding_attention_key),
                std::move(forward.shared_kv.sliding_attention_value),
                std::move(forward.dspark_taps.taps),
                kv_state->sequence_len(),
                kHiddenSize,
            });
            hidden.reset(new NativeHiddenState(std::move(hidden_impl)));
        }

        out->greedy_token = greedy_tokens->empty() ? 0 : greedy_tokens->back();
        out->greedy_logit = greedy_logits->empty() ? 0.0f : greedy_logits->back();
        out->sequence_len = kv_state->sequence_len();
        out->active_kv_bytes = kv_state->active_bytes();
        out->peak_memory_gb = static_cast<float>(mlx::core::get_peak_memory()) / 1'000'000'000.0f;
        out->peak_rss_mb = 0.0f;
        out->native_last_hidden = hidden.get();
        if (target_top_k != nullptr) {
            *target_top_k = std::move(computed_top_k);
        }
        if (last_hidden != nullptr) {
            *last_hidden = std::move(hidden);
        }
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("native Gemma 4 incremental block decode failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "native Gemma 4 incremental block decode failed with an unknown exception";
        return false;
    }
#endif
}

bool NativeTextModel::decode_incremental_block_with_retroactive_prefix(
    const int32_t* tokens,
    size_t token_count,
    NativeKvState* kv_state,
    NativeKvState* prefix_kv_state,
    size_t* out_accepted_prefix_count,
    Gemma4StepResult* out,
    std::vector<int32_t>* greedy_tokens,
    std::vector<float>* greedy_logits,
    std::string* error,
    std::unique_ptr<NativeHiddenState>* last_hidden,
    std::vector<NativeTopKEntries>* target_top_k) const {
    if (out == nullptr || greedy_tokens == nullptr || greedy_logits == nullptr || error == nullptr ||
        kv_state == nullptr || prefix_kv_state == nullptr || out_accepted_prefix_count == nullptr) {
        return false;
    }
    *out = Gemma4StepResult{};
    *out_accepted_prefix_count = 0;
    greedy_tokens->clear();
    greedy_logits->clear();
    error->clear();
    if (last_hidden != nullptr) {
        last_hidden->reset();
    }
    prefix_kv_state->clear();

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)tokens;
    (void)token_count;
    *error = "native Gemma 4 graph was requested, but gemma4_mlx was not built with MLX";
    return false;
#else
    try {
        if (impl_ == nullptr || impl_->language_tensor_count == 0) {
            *error = "native Gemma 4 model state is not loaded";
            return false;
        }
        if (tokens == nullptr || token_count == 0) {
            *error = "native incremental block decode requires at least one token";
            return false;
        }
        if (token_count > GEMMA4_MTP_MAX_DRAFT_TOKENS) {
            std::ostringstream message;
            message << "native incremental block decode supports token_count <= "
                    << GEMMA4_MTP_MAX_DRAFT_TOKENS;
            *error = message.str();
            return false;
        }
        if (kv_state->impl_ == nullptr || kv_state->sequence_len() == 0) {
            *error = "native incremental block decode requires a prior native prefill";
            return false;
        }
        if (prefix_kv_state->impl_ == nullptr) {
            *error = "native incremental block decode prefix KV state is missing";
            return false;
        }

        mlx::core::reset_peak_memory();
        BlockPrefixSources prefix_sources;
        NativeForwardArrays forward = decode_block_logits(
            *impl_,
            tokens,
            token_count,
            kv_state->impl_.get(),
            &prefix_sources);
        array logits = std::move(forward.logits);
        array greedy = mlx::core::argmax(logits, -1);
        array max_logits = to_float32(mlx::core::max(logits, -1));
        const bool real_margins = experimental_mtp_real_margins_enabled();
        std::vector<NativeTopKEntries> computed_top_k;
        if (real_margins) {
            computed_top_k = top_k_for_block_logits(logits, token_count);
        }
        mlx::core::eval({greedy, max_logits, forward.last_hidden});

        const int* token_data = greedy.data<int>();
        const float* logit_data = max_logits.data<float>();
        greedy_tokens->reserve(token_count);
        greedy_logits->reserve(token_count);
        for (size_t index = 0; index < token_count; ++index) {
            greedy_tokens->push_back(token_data[index]);
            greedy_logits->push_back(logit_data[index]);
        }
        if (!real_margins) {
            computed_top_k.reserve(token_count);
            for (size_t index = 0; index < token_count; ++index) {
                computed_top_k.push_back(top_k_from_greedy((*greedy_tokens)[index], (*greedy_logits)[index]));
            }
        }

        size_t accepted_prefix_count = 1;
        for (size_t index = 1; index < token_count; ++index) {
            if (tokens[index] != (*greedy_tokens)[index - 1]) {
                break;
            }
            ++accepted_prefix_count;
        }
        *out_accepted_prefix_count = accepted_prefix_count;
        if (accepted_prefix_count < token_count) {
            materialize_block_prefix_kv(
                forward.prefix_sources,
                accepted_prefix_count,
                prefix_kv_state->impl_.get());
        }

        std::unique_ptr<NativeHiddenState> hidden;
        if (last_hidden != nullptr) {
            std::unique_ptr<NativeHiddenState::Impl> hidden_impl(new NativeHiddenState::Impl{
                std::move(forward.last_hidden),
                std::move(forward.shared_kv.full_attention_key),
                std::move(forward.shared_kv.full_attention_value),
                std::move(forward.shared_kv.sliding_attention_key),
                std::move(forward.shared_kv.sliding_attention_value),
                std::move(forward.dspark_taps.taps),
                kv_state->sequence_len(),
                kHiddenSize,
            });
            hidden.reset(new NativeHiddenState(std::move(hidden_impl)));
        }

        out->greedy_token = greedy_tokens->empty() ? 0 : greedy_tokens->back();
        out->greedy_logit = greedy_logits->empty() ? 0.0f : greedy_logits->back();
        out->sequence_len = kv_state->sequence_len();
        out->active_kv_bytes = kv_state->active_bytes();
        out->peak_memory_gb = static_cast<float>(mlx::core::get_peak_memory()) / 1'000'000'000.0f;
        out->peak_rss_mb = 0.0f;
        out->native_last_hidden = hidden.get();
        if (target_top_k != nullptr) {
            *target_top_k = std::move(computed_top_k);
        }
        if (last_hidden != nullptr) {
            *last_hidden = std::move(hidden);
        }
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("native Gemma 4 incremental block decode failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "native Gemma 4 incremental block decode failed with an unknown exception";
        return false;
    }
#endif
}

NativeMtpAssistantModel::NativeMtpAssistantModel() : impl_(std::make_unique<Impl>()) {}

NativeMtpAssistantModel::~NativeMtpAssistantModel() = default;

NativeMtpAssistantModel::NativeMtpAssistantModel(NativeMtpAssistantModel&&) noexcept = default;

NativeMtpAssistantModel& NativeMtpAssistantModel::operator=(NativeMtpAssistantModel&&) noexcept = default;

bool NativeMtpAssistantModel::load(
    const std::filesystem::path& model_path,
    const Gemma4ModelManifest& manifest,
    std::unique_ptr<NativeMtpAssistantModel>* out,
    std::string* error) {
    if (out == nullptr || error == nullptr) {
        return false;
    }
    out->reset();
    error->clear();

    if (!manifest.is_assistant) {
        *error = "native MTP assistant load requires an assistant manifest";
        return false;
    }

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)model_path;
    (void)manifest;
    *error = "native Gemma 4 MTP assistant was requested, but gemma4_mlx was not built with MLX";
    return false;
#else
    try {
        std::unique_ptr<NativeMtpAssistantModel> model(new NativeMtpAssistantModel());
        model->impl_->manifest_summary = manifest.summary();
        model->impl_->default_quantization = manifest.default_quantization();
        model->impl_->quantization_overrides = manifest.quantization_overrides;

        const std::vector<std::filesystem::path> files = safetensor_files(model_path);
        if (files.empty()) {
            *error = "no safetensors files found in " + model_path.string();
            return false;
        }

        for (const std::filesystem::path& file : files) {
            auto loaded = mlx::core::load_safetensors(file.string());
            ++model->impl_->safetensor_file_count;
            model->impl_->total_tensor_count_seen += loaded.first.size();
            for (auto& entry : loaded.first) {
                if (!is_assistant_tensor(entry.first)) {
                    continue;
                }
                auto inserted = model->impl_->tensors.emplace(std::move(entry.first), std::move(entry.second));
                if (!inserted.second) {
                    *error = "duplicate MTP assistant tensor while loading " + file.string();
                    return false;
                }
            }
        }

        model->impl_->assistant_tensor_count = model->impl_->tensors.size();
        if (model->impl_->safetensor_file_count != manifest.safetensor_file_count ||
            model->impl_->total_tensor_count_seen != manifest.total_tensor_count ||
            model->impl_->assistant_tensor_count != manifest.language_tensor_count) {
            std::ostringstream message;
            message << "native loaded MTP assistant tensor inventory does not match manifest: files="
                    << model->impl_->safetensor_file_count << " tensors="
                    << model->impl_->total_tensor_count_seen << " assistant_tensors="
                    << model->impl_->assistant_tensor_count;
            *error = message.str();
            return false;
        }

        *out = std::move(model);
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("MLX native MTP assistant load failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "MLX native MTP assistant load failed with an unknown exception";
        return false;
    }
#endif
}

size_t NativeMtpAssistantModel::tensor_count() const {
    return impl_ == nullptr ? 0 : impl_->assistant_tensor_count;
}

std::string NativeMtpAssistantModel::summary() const {
    if (impl_ == nullptr) {
        return "native Gemma 4 MTP assistant model is empty";
    }
    std::ostringstream out;
    out << "native Gemma 4 MTP assistant loaded " << impl_->assistant_tensor_count
        << " assistant tensors from " << impl_->safetensor_file_count
        << " safetensor files (" << impl_->total_tensor_count_seen << " tensors scanned)";
    if (!impl_->manifest_summary.empty()) {
        out << "; " << impl_->manifest_summary;
    }
    return out.str();
}

NativeDSparkModel::NativeDSparkModel() : impl_(std::make_unique<Impl>()) {}

NativeDSparkModel::~NativeDSparkModel() = default;

NativeDSparkModel::NativeDSparkModel(NativeDSparkModel&&) noexcept = default;

NativeDSparkModel& NativeDSparkModel::operator=(NativeDSparkModel&&) noexcept = default;

bool NativeDSparkModel::load(
    const std::filesystem::path& model_path,
    const Gemma4ModelManifest& manifest,
    std::unique_ptr<NativeDSparkModel>* out,
    std::string* error) {
    if (out == nullptr || error == nullptr) {
        return false;
    }
    out->reset();
    error->clear();

    if (!manifest.is_dspark) {
        *error = "native DSpark load requires a DSpark manifest";
        return false;
    }

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)model_path;
    (void)manifest;
    *error = "native Gemma 4 DSpark drafter was requested, but gemma4_mlx was not built with MLX";
    return false;
#else
    try {
        std::unique_ptr<NativeDSparkModel> model(new NativeDSparkModel());
        model->impl_->manifest_summary = manifest.summary();
        model->impl_->block_size = manifest.block_size;
        model->impl_->markov_rank = manifest.markov_rank;
        model->impl_->mask_token_id = manifest.mask_token_id;
        model->impl_->vocab_size = manifest.vocab_size;
        model->impl_->target_layer_ids = manifest.target_layer_ids;

        const std::vector<std::filesystem::path> files = safetensor_files(model_path);
        if (files.empty()) {
            *error = "no safetensors files found in " + model_path.string();
            return false;
        }

        for (const std::filesystem::path& file : files) {
            auto loaded = mlx::core::load_safetensors(file.string());
            ++model->impl_->safetensor_file_count;
            model->impl_->total_tensor_count_seen += loaded.first.size();
            for (auto& entry : loaded.first) {
                if (!is_dspark_tensor(entry.first)) {
                    continue;
                }
                auto inserted = model->impl_->tensors.emplace(std::move(entry.first), std::move(entry.second));
                if (!inserted.second) {
                    *error = "duplicate DSpark tensor while loading " + file.string();
                    return false;
                }
            }
        }

        model->impl_->dspark_tensor_count = model->impl_->tensors.size();
        if (model->impl_->safetensor_file_count != manifest.safetensor_file_count ||
            model->impl_->total_tensor_count_seen != manifest.total_tensor_count ||
            model->impl_->dspark_tensor_count != manifest.language_tensor_count) {
            std::ostringstream message;
            message << "native loaded DSpark tensor inventory does not match manifest: files="
                    << model->impl_->safetensor_file_count << " tensors="
                    << model->impl_->total_tensor_count_seen << " dspark_tensors="
                    << model->impl_->dspark_tensor_count;
            *error = message.str();
            return false;
        }

        *out = std::move(model);
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("MLX native DSpark load failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "MLX native DSpark load failed with an unknown exception";
        return false;
    }
#endif
}

size_t NativeDSparkModel::tensor_count() const {
    return impl_ == nullptr ? 0 : impl_->dspark_tensor_count;
}

std::string NativeDSparkModel::summary() const {
    if (impl_ == nullptr) {
        return "native Gemma 4 DSpark model is empty";
    }
    std::ostringstream out;
    out << "native Gemma 4 DSpark loaded " << impl_->dspark_tensor_count
        << " DSpark tensors from " << impl_->safetensor_file_count
        << " safetensor files (" << impl_->total_tensor_count_seen << " tensors scanned)";
    if (!impl_->manifest_summary.empty()) {
        out << "; " << impl_->manifest_summary;
    }
    return out.str();
}

bool NativeDSparkModel::draft_block(
    const NativeKvState& target_kv,
    const NativeHiddenState& last_hidden,
    const std::vector<int32_t>& context_tokens,
    uint32_t block_size,
    int32_t* out_tokens,
    float* out_logits,
    float* out_logit_margins,
    float* out_confidence,
    size_t* inout_count,
    std::string* error) const {
    if (out_tokens == nullptr || out_logits == nullptr || out_logit_margins == nullptr ||
        out_confidence == nullptr || inout_count == nullptr || error == nullptr) {
        return false;
    }
    error->clear();
    const size_t capacity = *inout_count;
    *inout_count = 0;
    if (block_size == 0) {
        *error = "native DSpark draft requires block_size > 0";
        return false;
    }
    if (block_size > GEMMA4_DSPARK_MAX_DRAFT_TOKENS) {
        std::ostringstream message;
        message << "native DSpark draft supports block_size <= " << GEMMA4_DSPARK_MAX_DRAFT_TOKENS;
        *error = message.str();
        return false;
    }
    if (capacity < block_size) {
        *error = "native DSpark draft output buffer is smaller than block_size";
        return false;
    }
    if (context_tokens.empty()) {
        *error = "native DSpark draft requires at least one context token";
        return false;
    }

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)target_kv;
    (void)last_hidden;
    (void)context_tokens;
    (void)out_tokens;
    (void)out_logits;
    (void)out_logit_margins;
    (void)out_confidence;
    *error = "native Gemma 4 DSpark drafter was requested, but gemma4_mlx was not built with MLX";
    return false;
#else
    try {
        if (impl_ == nullptr || impl_->dspark_tensor_count == 0) {
            *error = "native Gemma 4 DSpark model state is not loaded";
            return false;
        }
        if (impl_->block_size != GEMMA4_DSPARK_MAX_DRAFT_TOKENS) {
            *error = "native Gemma 4 DSpark model is not the XR60 block-7 drafter";
            return false;
        }
        if (impl_->markov_rank != kDSparkMarkovRank || impl_->vocab_size != kDSparkVocabSize) {
            *error = "native Gemma 4 DSpark model does not match the XR60 Markov/vocab shape";
            return false;
        }
        if (target_kv.impl_ == nullptr || target_kv.impl_->sequence_len == 0) {
            *error = "native DSpark draft requires a populated target KV state";
            return false;
        }
        if (last_hidden.impl_ == nullptr || last_hidden.impl_->sequence_len == 0) {
            *error = "native DSpark draft requires a populated target hidden state";
            return false;
        }
        if (target_kv.impl_->sequence_len != context_tokens.size() ||
            last_hidden.impl_->sequence_len != context_tokens.size()) {
            *error = "native DSpark draft context tokens do not match the materialized target state";
            return false;
        }
        if (impl_->target_layer_ids.size() != GEMMA4_DSPARK_TARGET_TAP_COUNT ||
            target_kv.impl_->dspark_context_taps.size() != GEMMA4_DSPARK_TARGET_TAP_COUNT ||
            last_hidden.impl_->dspark_taps.size() != GEMMA4_DSPARK_TARGET_TAP_COUNT) {
            *error = "native DSpark draft requires cached target tap context for all XR60 layers";
            return false;
        }
        for (size_t index = 0; index < GEMMA4_DSPARK_TARGET_TAP_COUNT; ++index) {
            const uint32_t expected_layer = impl_->target_layer_ids[index];
            const NativeHiddenState::Impl::DSparkTap& context_tap =
                target_kv.impl_->dspark_context_taps[index];
            const NativeHiddenState::Impl::DSparkTap& last_tap = last_hidden.impl_->dspark_taps[index];
            if (context_tap.layer_id != expected_layer || last_tap.layer_id != expected_layer) {
                *error = "native DSpark draft target tap layer ids do not match the loaded drafter";
                return false;
            }
            const auto& context_shape = context_tap.hidden.shape();
            if (context_shape.size() != 3 || context_shape[0] != 1 ||
                context_shape[1] != static_cast<int>(context_tokens.size()) ||
                context_shape[2] != kHiddenSize) {
                std::ostringstream message;
                message << "native DSpark cached tap context has invalid shape for layer "
                        << expected_layer;
                *error = message.str();
                return false;
            }
            const auto& last_shape = last_tap.hidden.shape();
            if (last_shape.size() != 3 || last_shape[0] != 1 || last_shape[1] != 1 ||
                last_shape[2] != kHiddenSize) {
                std::ostringstream message;
                message << "native DSpark last tap has invalid shape for layer " << expected_layer;
                *error = message.str();
                return false;
            }
        }

        const int context_len = static_cast<int>(context_tokens.size());
        const int full_block_len = static_cast<int>(impl_->block_size);
        array hidden = dspark_forward_hidden(
            *impl_,
            target_kv.impl_->dspark_context_taps,
            context_tokens.back(),
            context_len,
            full_block_len);
        array logits = dspark_logits_for_hidden(*impl_, hidden);

        int32_t previous_token = context_tokens.back();
        size_t produced = 0;
        for (uint32_t step = 0; step < block_size; ++step) {
            array slot_logits = mlx::core::reshape(
                mlx::core::slice(
                    logits,
                    {0, static_cast<int>(step), 0},
                    {1, static_cast<int>(step + 1), static_cast<int>(impl_->vocab_size)}),
                {static_cast<int>(impl_->vocab_size)});
            array markov_embedding = dspark_markov_embedding(*impl_, previous_token);
            array corrected_logits = model_dtype(slot_logits + dspark_markov_bias(*impl_, markov_embedding));
            NativeTopKEntries top_k = top_k_for_flat_logits(corrected_logits, 2);
            const int32_t token = top_k[0].token_id;
            out_tokens[produced] = token;
            out_logits[produced] = top_k[0].logit;
            out_logit_margins[produced] = top_k[0].logit - top_k[1].logit;
            out_confidence[produced] =
                dspark_confidence(*impl_, hidden, static_cast<int>(step), markov_embedding);
            previous_token = token;
            ++produced;
        }

        *inout_count = produced;
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("native Gemma 4 DSpark draft failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "native Gemma 4 DSpark draft failed with an unknown exception";
        return false;
    }
#endif
}

bool NativeMtpAssistantModel::draft_block(
    const NativeTextModel& target_model,
    const NativeHiddenState& last_hidden,
    const std::vector<int32_t>& context_tokens,
    uint32_t block_size,
    int32_t* out_tokens,
    float* out_logits,
    float* out_logit_margins,
    size_t* inout_count,
    std::string* error,
    bool lazy_second_draft,
    int32_t first_accept_token) const {
    if (out_tokens == nullptr || inout_count == nullptr || error == nullptr) {
        return false;
    }
    error->clear();
    const bool capture_scores = experimental_mtp_real_margins_enabled();
    if (capture_scores && (out_logits == nullptr || out_logit_margins == nullptr)) {
        *error = "native MTP draft requires score output buffers when real margins are enabled";
        return false;
    }
    if (!capture_scores && (out_logits != nullptr || out_logit_margins != nullptr)) {
        *error =
            "native MTP draft score output buffers require GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1";
        return false;
    }
    const size_t capacity = *inout_count;
    *inout_count = 0;

    if (block_size == 0) {
        *error = "native MTP draft requires block_size > 0";
        return false;
    }
    if (block_size > GEMMA4_MTP_MAX_DRAFT_TOKENS) {
        std::ostringstream message;
        message << "native MTP draft supports block_size <= " << GEMMA4_MTP_MAX_DRAFT_TOKENS;
        *error = message.str();
        return false;
    }
    if (capacity < block_size) {
        *error = "native MTP draft output buffer is smaller than block_size";
        return false;
    }
    if (context_tokens.empty()) {
        *error = "native MTP draft requires at least one context token";
        return false;
    }
    if (last_hidden.sequence_len() == 0 || context_tokens.size() != last_hidden.sequence_len()) {
        *error = "native MTP draft context tokens do not match the materialized hidden state";
        return false;
    }

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)target_model;
    (void)last_hidden;
    (void)context_tokens;
    (void)block_size;
    (void)out_tokens;
    (void)out_logits;
    (void)out_logit_margins;
    *error = "native Gemma 4 MTP assistant was requested, but gemma4_mlx was not built with MLX";
    return false;
#else
    try {
        if (impl_ == nullptr || impl_->assistant_tensor_count == 0) {
            *error = "native Gemma 4 MTP assistant model state is not loaded";
            return false;
        }
        if (target_model.impl_ == nullptr || target_model.impl_->language_tensor_count == 0) {
            *error = "native Gemma 4 target model state is not loaded for MTP token embeddings";
            return false;
        }
        if (last_hidden.impl_ == nullptr || !last_hidden.has_shared_kv()) {
            *error = "native MTP draft requires materialized target hidden/shared KV state";
            return false;
        }
        const uint64_t first_position = last_hidden.sequence_len() - 1;
        if (first_position + block_size > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            *error = "native MTP draft position offset exceeds MLX shape limits";
            return false;
        }

        array current_hidden = last_hidden.impl_->hidden;
        int32_t token_id = context_tokens.back();
        size_t produced = 0;
        std::array<float, GEMMA4_MTP_MAX_DRAFT_TOKENS> scratch_logits{};
        std::array<float, GEMMA4_MTP_MAX_DRAFT_TOKENS> scratch_margins{};
        float* logits_out = out_logits == nullptr ? scratch_logits.data() : out_logits;
        float* margins_out = out_logit_margins == nullptr ? scratch_margins.data() : out_logit_margins;
        const bool skip_final_projection = experimental_mtp_skip_final_projection_enabled();
        const bool lazy_first_draft = lazy_second_draft;
        for (uint32_t step = 0; step < block_size; ++step) {
            const bool defer_first_projection = lazy_first_draft && step == 0;
            const bool need_projected_hidden =
                (!skip_final_projection || step + 1 < block_size) && !defer_first_projection;
            NativeMtpDraftStep draft = assistant_draft_one(
                *impl_,
                *target_model.impl_,
                *last_hidden.impl_,
                current_hidden,
                token_id,
                // HF SinglePositionMultiTokenCandidateGenerator computes
                // position_ids once before its drafter loop; shared target KV
                // effectively locks Gemma 4 MTP to this constant position.
                static_cast<int>(first_position),
                need_projected_hidden);
            out_tokens[produced++] = draft.token;
            logits_out[produced - 1] = draft.logit;
            margins_out[produced - 1] = draft.margin;
            if (defer_first_projection && draft.token != first_accept_token) {
                break;
            }
            token_id = draft.token;
            if (step + 1 < block_size) {
                if (defer_first_projection) {
                    current_hidden = quantized_linear(*impl_, draft.projected_hidden, "post_projection");
                    mlx::core::eval(current_hidden);
                } else {
                    current_hidden = std::move(draft.projected_hidden);
                }
            }
        }

        *inout_count = produced;
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("native Gemma 4 MTP assistant draft failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "native Gemma 4 MTP assistant draft failed with an unknown exception";
        return false;
    }
#endif
}

} // namespace gemma4d
