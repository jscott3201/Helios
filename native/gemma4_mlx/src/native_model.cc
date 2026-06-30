#include "native_model.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
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
    mlx::core::array hidden;
    std::optional<mlx::core::array> full_attention_key;
    std::optional<mlx::core::array> full_attention_value;
    std::optional<mlx::core::array> sliding_attention_key;
    std::optional<mlx::core::array> sliding_attention_value;
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
#endif
    uint64_t sequence_len = 0;
    uint64_t active_bytes = 0;
};

struct NativeTextModel::Impl {
#ifdef GEMMA4D_MLX_AVAILABLE
    std::unordered_map<std::string, mlx::core::array> tensors;
#endif
    QuantizationSpec default_quantization;
    std::unordered_map<std::string, QuantizationSpec> quantization_overrides;
    size_t safetensor_file_count = 0;
    size_t language_tensor_count = 0;
    size_t total_tensor_count_seen = 0;
    std::string manifest_summary;
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

#ifdef GEMMA4D_MLX_AVAILABLE

using mlx::core::array;

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
    return model_dtype(mlx::core::quantized_matmul(
        x,
        tensor_or_throw(impl, prefix + ".weight"),
        tensor_or_throw(impl, prefix + ".scales"),
        std::optional<array>(tensor_or_throw(impl, prefix + ".biases")),
        true,
        static_cast<int>(spec.group_size),
        static_cast<int>(spec.bits),
        "affine"));
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

constexpr int kTargetLayerCount = 48;
constexpr int kHiddenSize = 3840;
constexpr int kSlidingWindowSize = 1024;
constexpr uint64_t kBf16Bytes = 2;

bool target_layer_full_attention(uint32_t layer_idx) {
    return ((layer_idx + 1) % 6) == 0;
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
    mlx::core::eval({*layer->key, *layer->value});
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
    mlx::core::eval({*target_kv->key, *target_kv->value});
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

struct NativeHiddenArrays {
    array hidden;
    SharedKvArrays shared_kv;
};

struct NativeForwardArrays {
    array logits;
    array last_hidden;
    SharedKvArrays shared_kv;
};

struct NativeVerifyArrays {
    std::vector<int32_t> greedy_tokens;
    std::vector<float> greedy_logits;
    array last_hidden;
    SharedKvArrays shared_kv;
    float peak_memory_gb = 0.0f;
};

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
        const std::string label = "layer" + std::to_string(layer);
        dump_hidden_tensor(label.c_str(), h);
        trace_hidden_stats(label.c_str(), h, sequence_len);
    }
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, "language_model.model.norm.weight")),
        1e-6f));
    dump_hidden_tensor("final_norm", h);
    trace_hidden_stats("final_norm", h, sequence_len);
    if (target_kv != nullptr) {
        target_kv->sequence_len = tokens.size();
        target_kv->active_bytes = estimate_target_kv_bytes(tokens.size());
    }

    return NativeHiddenArrays{std::move(h), std::move(shared_kv)};
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

NativeForwardArrays forward_last_logits(const NativeTextModel::Impl& impl, const std::vector<int32_t>& tokens) {
    NativeHiddenArrays forward = forward_hidden(impl, tokens);
    const int sequence_len = static_cast<int>(tokens.size());
    array last_hidden = mlx::core::slice(forward.hidden, {0, sequence_len - 1, 0}, {1, sequence_len, 3840});
    array logits = target_logits_for_hidden(impl, last_hidden);
    logits = mlx::core::reshape(logits, {262144});
    dump_hidden_tensor("logits", logits);
    return NativeForwardArrays{std::move(logits), std::move(last_hidden), std::move(forward.shared_kv)};
}

NativeForwardArrays prefill_last_logits(
    const NativeTextModel::Impl& impl,
    const std::vector<int32_t>& tokens,
    NativeKvState::Impl* target_kv) {
    NativeHiddenArrays forward = forward_hidden(impl, tokens, target_kv);
    const int sequence_len = static_cast<int>(tokens.size());
    array last_hidden = mlx::core::slice(forward.hidden, {0, sequence_len - 1, 0}, {1, sequence_len, 3840});
    array logits = target_logits_for_hidden(impl, last_hidden);
    logits = mlx::core::reshape(logits, {262144});
    dump_hidden_tensor("logits", logits);
    return NativeForwardArrays{std::move(logits), std::move(last_hidden), std::move(forward.shared_kv)};
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

    for (uint32_t layer = 0; layer < kTargetLayerCount; ++layer) {
        h = target_layer_decode_forward(
            impl,
            h,
            layer,
            previous_sequence_len,
            &target_kv->layers[layer],
            &shared_kv);
    }
    h = model_dtype(mlx::core::fast::rms_norm(
        h,
        std::optional<array>(tensor_or_throw(impl, "language_model.model.norm.weight")),
        1e-6f));
    array logits = target_logits_for_hidden(impl, h);
    logits = mlx::core::reshape(logits, {262144});
    target_kv->sequence_len = previous_sequence_len + 1;
    target_kv->active_bytes = estimate_target_kv_bytes(target_kv->sequence_len);
    return NativeForwardArrays{std::move(logits), std::move(h), std::move(shared_kv)};
}

NativeVerifyArrays forward_verify_logits(
    const NativeTextModel::Impl& impl,
    const std::vector<int32_t>& tokens,
    size_t first_position,
    size_t position_count) {
    if (position_count == 0) {
        throw std::runtime_error("native MTP verify requires at least one logit position");
    }
    if (first_position + position_count > tokens.size()) {
        throw std::runtime_error("native MTP verify logit positions exceed token context");
    }
    if (first_position + position_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("native MTP verify position exceeds MLX shape limits");
    }

    mlx::core::reset_peak_memory();
    NativeHiddenArrays forward = forward_hidden(impl, tokens);
    const int first = static_cast<int>(first_position);
    const int stop = static_cast<int>(first_position + position_count);
    array selected_hidden = mlx::core::slice(forward.hidden, {0, first, 0}, {1, stop, 3840});
    array logits = target_logits_for_hidden(impl, selected_hidden);
    array greedy = mlx::core::argmax(logits, -1);
    array greedy_logits = to_float32(mlx::core::max(logits, -1));
    array last_hidden = mlx::core::slice(
        forward.hidden,
        {0, static_cast<int>(tokens.size() - 1), 0},
        {1, static_cast<int>(tokens.size()), 3840});
    mlx::core::eval({greedy, greedy_logits, last_hidden});

    std::vector<int32_t> greedy_tokens;
    std::vector<float> greedy_logits_out;
    greedy_tokens.reserve(position_count);
    greedy_logits_out.reserve(position_count);
    const int* token_data = greedy.data<int>();
    const float* logit_data = greedy_logits.data<float>();
    for (size_t index = 0; index < position_count; ++index) {
        greedy_tokens.push_back(static_cast<int32_t>(token_data[index]));
        greedy_logits_out.push_back(logit_data[index]);
    }
    const float peak_memory_gb = static_cast<float>(mlx::core::get_peak_memory()) / 1'000'000'000.0f;
    return NativeVerifyArrays{
        std::move(greedy_tokens),
        std::move(greedy_logits_out),
        std::move(last_hidden),
        std::move(forward.shared_kv),
        peak_memory_gb,
    };
}

array target_token_embedding(const NativeTextModel::Impl& impl, int32_t token_id) {
    const std::vector<int32_t> ids = {token_id};
    array token_ids(ids.begin(), {1, 1}, mlx::core::int32);
    return model_dtype(quantized_embedding(impl, token_ids) * model_scalar(std::sqrt(3840.0f)));
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
    return mlx::core::reshape(logits, {262144});
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
    array projected_hidden;
};

NativeMtpDraftStep assistant_draft_one(
    const NativeMtpAssistantModel::Impl& assistant,
    const NativeTextModel::Impl& target,
    const NativeHiddenState::Impl& shared,
    const array& current_hidden,
    int32_t token_id,
    int position_offset) {
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
    array projected = quantized_linear(assistant, h, "post_projection");
    mlx::core::eval({greedy, projected});

    return NativeMtpDraftStep{greedy.item<int>(), std::move(projected)};
}

bool trace_parity_logits_enabled() {
    const char* value = std::getenv("GEMMA4D_NATIVE_TRACE_PARITY_LOGITS");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
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

#endif

} // namespace

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

NativeKvState::NativeKvState() : impl_(std::make_unique<Impl>()) {}

NativeKvState::~NativeKvState() = default;

NativeKvState::NativeKvState(NativeKvState&&) noexcept = default;

NativeKvState& NativeKvState::operator=(NativeKvState&&) noexcept = default;

void NativeKvState::clear() {
    if (impl_ != nullptr) {
#ifdef GEMMA4D_MLX_AVAILABLE
        impl_->layers.clear();
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
        array max_logit = to_float32(mlx::core::max(logits));
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
        NativeForwardArrays forward = prefill_last_logits(*impl_, tokens, state->impl_.get());
        array logits = std::move(forward.logits);
        array greedy = mlx::core::argmax(logits);
        array max_logit = to_float32(mlx::core::max(logits));
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
    std::unique_ptr<NativeHiddenState>* last_hidden) const {
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

        mlx::core::reset_peak_memory();
        NativeForwardArrays forward = decode_last_logits(*impl_, token, kv_state->impl_.get());
        array logits = std::move(forward.logits);
        array greedy = mlx::core::argmax(logits);
        array max_logit = to_float32(mlx::core::max(logits));
        mlx::core::eval({greedy, max_logit});

        std::unique_ptr<NativeHiddenState> hidden;
        if (last_hidden != nullptr) {
            std::unique_ptr<NativeHiddenState::Impl> hidden_impl(new NativeHiddenState::Impl{
                std::move(forward.last_hidden),
                std::move(forward.shared_kv.full_attention_key),
                std::move(forward.shared_kv.full_attention_value),
                std::move(forward.shared_kv.sliding_attention_key),
                std::move(forward.shared_kv.sliding_attention_value),
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

bool NativeTextModel::verify_draft_block(
    const std::vector<int32_t>& context_tokens,
    const int32_t* draft_tokens,
    size_t draft_count,
    std::vector<int32_t>* committed_tokens,
    Gemma4StepResult* out,
    std::string* error,
    std::unique_ptr<NativeHiddenState>* last_hidden) const {
    if (committed_tokens == nullptr || out == nullptr || error == nullptr) {
        return false;
    }
    committed_tokens->clear();
    *out = Gemma4StepResult{};
    error->clear();
    if (last_hidden != nullptr) {
        last_hidden->reset();
    }
    if (context_tokens.empty()) {
        *error = "native MTP verify requires a non-empty accepted context";
        return false;
    }
    if (draft_count == 0 || draft_tokens == nullptr) {
        *error = "native MTP verify requires at least one draft token";
        return false;
    }
    if (draft_count > 2) {
        *error = "native MTP verify currently supports draft_count <= 2 for M06";
        return false;
    }
    if (context_tokens.size() + draft_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        *error = "native MTP verify token count exceeds MLX shape limits";
        return false;
    }

#ifndef GEMMA4D_MLX_AVAILABLE
    (void)draft_tokens;
    (void)last_hidden;
    *error = "native Gemma 4 graph was requested, but gemma4_mlx was not built with MLX";
    return false;
#else
    try {
        if (impl_ == nullptr || impl_->language_tensor_count == 0) {
            *error = "native Gemma 4 model state is not loaded";
            return false;
        }

        std::vector<int32_t> candidate_tokens = context_tokens;
        candidate_tokens.insert(candidate_tokens.end(), draft_tokens, draft_tokens + draft_count);
        NativeVerifyArrays verified = forward_verify_logits(
            *impl_,
            candidate_tokens,
            context_tokens.size() - 1,
            draft_count + 1);

        size_t accepted_count = 0;
        bool rejected = false;
        int32_t fallback_token = 0;
        for (size_t index = 0; index < draft_count; ++index) {
            const int32_t target_token = verified.greedy_tokens[index];
            if (draft_tokens[index] == target_token) {
                ++accepted_count;
                continue;
            }
            rejected = true;
            fallback_token = target_token;
            break;
        }

        if (rejected) {
            std::vector<int32_t> fallback_tokens = context_tokens;
            fallback_tokens.insert(fallback_tokens.end(), draft_tokens, draft_tokens + accepted_count);
            fallback_tokens.push_back(fallback_token);

            Gemma4StepResult fallback_step{};
            std::unique_ptr<NativeHiddenState> fallback_hidden;
            if (!forward_greedy(fallback_tokens, &fallback_step, error, &fallback_hidden)) {
                return false;
            }
            if (fallback_step.peak_memory_gb < verified.peak_memory_gb) {
                fallback_step.peak_memory_gb = verified.peak_memory_gb;
            }
            *committed_tokens = std::move(fallback_tokens);
            *out = fallback_step;
            if (last_hidden != nullptr) {
                *last_hidden = std::move(fallback_hidden);
                out->native_last_hidden = last_hidden->get();
            }
            return true;
        }

        std::unique_ptr<NativeHiddenState> hidden;
        if (last_hidden != nullptr) {
            std::unique_ptr<NativeHiddenState::Impl> hidden_impl(new NativeHiddenState::Impl{
                std::move(verified.last_hidden),
                std::move(verified.shared_kv.full_attention_key),
                std::move(verified.shared_kv.full_attention_value),
                std::move(verified.shared_kv.sliding_attention_key),
                std::move(verified.shared_kv.sliding_attention_value),
                static_cast<uint64_t>(candidate_tokens.size()),
                3840,
            });
            hidden.reset(new NativeHiddenState(std::move(hidden_impl)));
        }

        *committed_tokens = std::move(candidate_tokens);
        out->greedy_token = verified.greedy_tokens[draft_count];
        out->greedy_logit = verified.greedy_logits[draft_count];
        out->sequence_len = committed_tokens->size();
        out->peak_memory_gb = verified.peak_memory_gb;
        out->peak_rss_mb = 0.0f;
        out->native_last_hidden = hidden.get();
        if (last_hidden != nullptr) {
            *last_hidden = std::move(hidden);
        }
        return true;
    } catch (const std::exception& ex) {
        *error = std::string("native Gemma 4 MTP verify failed: ") + ex.what();
        return false;
    } catch (...) {
        *error = "native Gemma 4 MTP verify failed with an unknown exception";
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

bool NativeMtpAssistantModel::draft_block(
    const NativeTextModel& target_model,
    const NativeHiddenState& last_hidden,
    const std::vector<int32_t>& context_tokens,
    uint32_t block_size,
    int32_t* out_tokens,
    size_t* inout_count,
    std::string* error) const {
    if (out_tokens == nullptr || inout_count == nullptr || error == nullptr) {
        return false;
    }
    error->clear();
    const size_t capacity = *inout_count;
    *inout_count = 0;

    if (block_size == 0) {
        *error = "native MTP draft requires block_size > 0";
        return false;
    }
    if (block_size > 2) {
        *error = "native MTP draft currently supports block_size <= 2 for M06";
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
        for (uint32_t step = 0; step < block_size; ++step) {
            NativeMtpDraftStep draft = assistant_draft_one(
                *impl_,
                *target_model.impl_,
                *last_hidden.impl_,
                current_hidden,
                token_id,
                static_cast<int>(first_position + step));
            out_tokens[produced++] = draft.token;
            token_id = draft.token;
            current_hidden = std::move(draft.projected_hidden);
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
