#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>

namespace gemma4d {

struct QuantizationSpec {
    uint32_t bits = 0;
    uint32_t group_size = 0;
};

struct Gemma4ModelManifest {
    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t num_hidden_layers = 0;
    uint32_t num_attention_heads = 0;
    uint32_t num_key_value_heads = 0;
    uint32_t num_global_key_value_heads = 0;
    uint32_t vocab_size = 0;
    uint32_t sliding_window = 0;
    uint32_t quantization_bits = 0;
    uint32_t quantization_group_size = 0;
    std::unordered_map<std::string, QuantizationSpec> quantization_overrides;
    bool attention_k_eq_v = false;
    bool tie_word_embeddings = false;

    size_t safetensor_file_count = 0;
    size_t total_tensor_count = 0;
    size_t language_tensor_count = 0;
    size_t quantized_linear_count = 0;
    size_t ignored_multimodal_tensor_count = 0;

    std::string summary() const;
    QuantizationSpec default_quantization() const;
    QuantizationSpec quantization_for(const std::string& prefix) const;
};

bool load_gemma4_model_manifest(
    const std::filesystem::path& model_path,
    Gemma4ModelManifest* out,
    std::string* error);

} // namespace gemma4d
