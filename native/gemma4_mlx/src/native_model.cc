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

#ifdef GEMMA4D_MLX_AVAILABLE

using mlx::core::array;

array to_float32(array value) {
    return mlx::core::astype(std::move(value), mlx::core::float32);
}

const array& tensor_or_throw(const NativeTextModel::Impl& impl, const std::string& key) {
    const auto found = impl.tensors.find(key);
    if (found == impl.tensors.end()) {
        throw std::runtime_error("missing loaded tensor " + key);
    }
    return found->second;
}

QuantizationSpec quantization_for(const NativeTextModel::Impl& impl, const std::string& prefix) {
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

array quantized_linear(const NativeTextModel::Impl& impl, const array& x, const std::string& prefix) {
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
    int sequence_len) {
    const bool full_attention = ((layer_idx + 1) % 6) == 0;
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

    const std::optional<array> mask = full_attention ? std::nullopt : sliding_causal_mask(sequence_len, 1024);
    const std::string mask_mode = mask.has_value() || sequence_len == 1 ? "" : "causal";
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

array layer_forward(const NativeTextModel::Impl& impl, const array& x, uint32_t layer_idx, int sequence_len) {
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
    h = attention_forward(impl, h, layer_idx, sequence_len);
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

array forward_last_logits(const NativeTextModel::Impl& impl, const std::vector<int32_t>& tokens) {
    if (tokens.empty()) {
        throw std::runtime_error("native forward requires at least one token");
    }
    if (tokens.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("native forward token count exceeds MLX shape limits");
    }

    const int sequence_len = static_cast<int>(tokens.size());
    array token_ids(tokens.begin(), {1, sequence_len}, mlx::core::int32);
    array h = model_dtype(quantized_embedding(impl, token_ids) * model_scalar(std::sqrt(3840.0f)));
    dump_hidden_tensor("embed", h);
    trace_hidden_stats("embed", h, sequence_len);

    for (uint32_t layer = 0; layer < 48; ++layer) {
        h = layer_forward(impl, h, layer, sequence_len);
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

    h = mlx::core::slice(h, {0, sequence_len - 1, 0}, {1, sequence_len, 3840});
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
    logits = model_dtype(mlx::core::tanh(logits / model_scalar(30.0f)) * model_scalar(30.0f));
    logits = mlx::core::reshape(logits, {262144});
    dump_hidden_tensor("logits", logits);
    return logits;
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
    std::string* error) const {
    if (out == nullptr || error == nullptr) {
        return false;
    }
    *out = Gemma4StepResult{};
    error->clear();

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
        array logits = forward_last_logits(*impl_, tokens);
        array greedy = mlx::core::argmax(logits);
        array max_logit = to_float32(mlx::core::max(logits));
        mlx::core::eval({greedy, max_logit});
        trace_parity_logits(tokens, logits);

        out->greedy_token = greedy.item<int>();
        out->greedy_logit = max_logit.item<float>();
        out->sequence_len = tokens.size();
        out->peak_memory_gb = static_cast<float>(mlx::core::get_peak_memory()) / 1'000'000'000.0f;
        out->peak_rss_mb = 0.0f;
        out->native_last_hidden = nullptr;
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

} // namespace gemma4d
