#include "model_manifest.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string_view>
#include <vector>

namespace gemma4d {
namespace {

bool parse_json_string(const std::string& text, size_t* pos, std::string* out);
void skip_ws(const std::string& text, size_t* pos);
bool skip_json_value(const std::string& text, size_t* pos);

std::string read_text_file(const std::filesystem::path& path, std::string* error) {
    std::ifstream input(path);
    if (!input) {
        *error = "could not open " + path.string();
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

bool has_text(const std::string& haystack, std::string_view needle) {
    return haystack.find(needle) != std::string::npos;
}

size_t find_anchor(const std::string& text, std::string_view anchor) {
    const size_t pos = text.find(anchor);
    return pos == std::string::npos ? 0 : pos;
}

bool parse_json_uint_after(
    const std::string& text,
    std::string_view anchor,
    std::string_view key,
    uint32_t* out) {
    const size_t start = find_anchor(text, anchor);
    const std::string quoted_key = "\"" + std::string(key) + "\"";
    const size_t key_pos = text.find(quoted_key, start);
    if (key_pos == std::string::npos) {
        return false;
    }
    size_t pos = text.find(':', key_pos + quoted_key.size());
    if (pos == std::string::npos) {
        return false;
    }
    ++pos;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    if (pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[pos]))) {
        return false;
    }
    uint64_t value = 0;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
        value = (value * 10) + static_cast<uint64_t>(text[pos] - '0');
        if (value > UINT32_MAX) {
            return false;
        }
        ++pos;
    }
    *out = static_cast<uint32_t>(value);
    return true;
}

bool parse_json_bool_after(
    const std::string& text,
    std::string_view anchor,
    std::string_view key,
    bool* out) {
    const size_t start = find_anchor(text, anchor);
    const std::string quoted_key = "\"" + std::string(key) + "\"";
    const size_t key_pos = text.find(quoted_key, start);
    if (key_pos == std::string::npos) {
        return false;
    }
    size_t pos = text.find(':', key_pos + quoted_key.size());
    if (pos == std::string::npos) {
        return false;
    }
    ++pos;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    if (text.compare(pos, 4, "true") == 0) {
        *out = true;
        return true;
    }
    if (text.compare(pos, 5, "false") == 0) {
        *out = false;
        return true;
    }
    return false;
}

bool parse_quantization_overrides(
    const std::string& config,
    Gemma4ModelManifest* manifest,
    std::string* error) {
    size_t pos = config.find("\"quantization_config\"");
    if (pos == std::string::npos) {
        pos = config.find("\"quantization\"");
    }
    if (pos == std::string::npos) {
        return true;
    }
    pos = config.find('{', pos);
    if (pos == std::string::npos) {
        *error = "config.json quantization object is malformed";
        return false;
    }
    ++pos;
    while (pos < config.size()) {
        skip_ws(config, &pos);
        if (pos < config.size() && config[pos] == '}') {
            return true;
        }
        std::string key;
        if (!parse_json_string(config, &pos, &key)) {
            *error = "could not parse config.json quantization key";
            return false;
        }
        skip_ws(config, &pos);
        if (pos >= config.size() || config[pos] != ':') {
            *error = "malformed config.json quantization entry for " + key;
            return false;
        }
        ++pos;
        skip_ws(config, &pos);
        const size_t value_start = pos;
        if (!skip_json_value(config, &pos)) {
            *error = "could not parse config.json quantization entry for " + key;
            return false;
        }
        if (key.rfind("language_model.model.", 0) == 0) {
            const std::string value = config.substr(value_start, pos - value_start);
            QuantizationSpec spec{
                manifest->quantization_bits,
                manifest->quantization_group_size,
            };
            if (parse_json_uint_after(value, "", "bits", &spec.bits)) {
                (void)parse_json_uint_after(value, "", "group_size", &spec.group_size);
                if (spec.group_size != 64 || (spec.bits != 4 && spec.bits != 8)) {
                    *error = "unsupported quantization override for " + key;
                    return false;
                }
                manifest->quantization_overrides[key] = spec;
            }
        }
        skip_ws(config, &pos);
        if (pos < config.size() && config[pos] == ',') {
            ++pos;
        }
    }
    *error = "unterminated config.json quantization object";
    return false;
}

bool parse_text_config(
    const std::string& config,
    Gemma4ModelManifest* manifest,
    std::string* error) {
    constexpr std::string_view text_anchor = "\"text_config\"";
    const struct Field {
        std::string_view key;
        uint32_t* out;
    } fields[] = {
        {"hidden_size", &manifest->hidden_size},
        {"intermediate_size", &manifest->intermediate_size},
        {"num_hidden_layers", &manifest->num_hidden_layers},
        {"num_attention_heads", &manifest->num_attention_heads},
        {"num_key_value_heads", &manifest->num_key_value_heads},
        {"num_global_key_value_heads", &manifest->num_global_key_value_heads},
        {"vocab_size", &manifest->vocab_size},
        {"sliding_window", &manifest->sliding_window},
    };
    for (const Field& field : fields) {
        if (!parse_json_uint_after(config, text_anchor, field.key, field.out)) {
            *error = "config.json is missing text_config." + std::string(field.key);
            return false;
        }
    }
    if (!parse_json_uint_after(config, "\"quantization\"", "bits", &manifest->quantization_bits) ||
        !parse_json_uint_after(config, "\"quantization\"", "group_size", &manifest->quantization_group_size)) {
        *error = "config.json is missing quantization bits/group_size";
        return false;
    }
    if (!parse_quantization_overrides(config, manifest, error)) {
        return false;
    }
    if (!parse_json_bool_after(config, text_anchor, "attention_k_eq_v", &manifest->attention_k_eq_v) ||
        !parse_json_bool_after(config, text_anchor, "tie_word_embeddings", &manifest->tie_word_embeddings)) {
        *error = "config.json is missing text_config attention/tied embedding flags";
        return false;
    }
    (void)parse_json_uint_after(config, text_anchor, "num_kv_shared_layers", &manifest->num_kv_shared_layers);
    return true;
}

bool parse_config(
    const std::filesystem::path& model_path,
    Gemma4ModelManifest* manifest,
    std::string* error) {
    const std::string config = read_text_file(model_path / "config.json", error);
    if (config.empty()) {
        return false;
    }
    if (!has_text(config, "\"Gemma4UnifiedForConditionalGeneration\"")) {
        *error = "config.json does not advertise Gemma4UnifiedForConditionalGeneration";
        return false;
    }
    if (!has_text(config, "\"model_type\": \"gemma4_unified\"")) {
        *error = "config.json model_type is not gemma4_unified";
        return false;
    }

    manifest->is_assistant = false;
    if (!parse_text_config(config, manifest, error)) {
        return false;
    }

    if (manifest->hidden_size != 3840 || manifest->intermediate_size != 15360 ||
        manifest->num_hidden_layers != 48 || manifest->num_attention_heads != 16 ||
        manifest->num_key_value_heads != 8 || manifest->num_global_key_value_heads != 1 ||
        manifest->vocab_size != 262144 || manifest->quantization_bits != 4 ||
        manifest->quantization_group_size != 64 || !manifest->attention_k_eq_v ||
        !manifest->tie_word_embeddings) {
        *error = "config.json does not match the expected Gemma 4 12B text-only shape";
        return false;
    }
    return true;
}

bool parse_assistant_config(
    const std::filesystem::path& model_path,
    Gemma4ModelManifest* manifest,
    std::string* error) {
    const std::string config = read_text_file(model_path / "config.json", error);
    if (config.empty()) {
        return false;
    }
    if (!has_text(config, "\"Gemma4UnifiedAssistantForCausalLM\"")) {
        *error = "config.json does not advertise Gemma4UnifiedAssistantForCausalLM";
        return false;
    }
    if (!has_text(config, "\"model_type\": \"gemma4_unified_assistant\"")) {
        *error = "config.json model_type is not gemma4_unified_assistant";
        return false;
    }

    manifest->is_assistant = true;
    if (!parse_json_uint_after(config, "", "backbone_hidden_size", &manifest->backbone_hidden_size)) {
        *error = "config.json is missing backbone_hidden_size";
        return false;
    }
    if (!parse_text_config(config, manifest, error)) {
        return false;
    }

    if (manifest->backbone_hidden_size != 3840 || manifest->hidden_size != 1024 ||
        manifest->intermediate_size != 8192 || manifest->num_hidden_layers != 4 ||
        manifest->num_attention_heads != 16 || manifest->num_key_value_heads != 8 ||
        manifest->num_global_key_value_heads != 1 || manifest->num_kv_shared_layers != 4 ||
        manifest->vocab_size != 262144 || manifest->quantization_bits != 4 ||
        manifest->quantization_group_size != 64 || !manifest->attention_k_eq_v ||
        !manifest->tie_word_embeddings) {
        *error = "config.json does not match the expected Gemma 4 12B MTP assistant shape";
        return false;
    }
    return true;
}

bool parse_json_string(const std::string& text, size_t* pos, std::string* out) {
    if (*pos >= text.size() || text[*pos] != '"') {
        return false;
    }
    ++(*pos);
    out->clear();
    while (*pos < text.size()) {
        const char c = text[*pos];
        ++(*pos);
        if (c == '"') {
            return true;
        }
        if (c == '\\') {
            if (*pos >= text.size()) {
                return false;
            }
            out->push_back(text[*pos]);
            ++(*pos);
        } else {
            out->push_back(c);
        }
    }
    return false;
}

void skip_ws(const std::string& text, size_t* pos) {
    while (*pos < text.size() && std::isspace(static_cast<unsigned char>(text[*pos]))) {
        ++(*pos);
    }
}

bool skip_json_value(const std::string& text, size_t* pos) {
    skip_ws(text, pos);
    if (*pos >= text.size()) {
        return false;
    }
    if (text[*pos] == '"') {
        std::string ignored;
        return parse_json_string(text, pos, &ignored);
    }
    int object_depth = 0;
    int array_depth = 0;
    bool in_string = false;
    bool escaped = false;
    while (*pos < text.size()) {
        const char c = text[*pos];
        if (in_string) {
            escaped = (!escaped && c == '\\');
            if (!escaped && c == '"') {
                in_string = false;
            } else if (c != '\\') {
                escaped = false;
            }
            ++(*pos);
            continue;
        }
        if (c == '"') {
            in_string = true;
            ++(*pos);
            continue;
        }
        if (c == '{') {
            ++object_depth;
        } else if (c == '}') {
            if (object_depth == 0 && array_depth == 0) {
                return true;
            }
            --object_depth;
        } else if (c == '[') {
            ++array_depth;
        } else if (c == ']') {
            --array_depth;
        } else if (c == ',' && object_depth == 0 && array_depth == 0) {
            return true;
        }
        ++(*pos);
        if (object_depth < 0 || array_depth < 0) {
            return true;
        }
    }
    return true;
}

bool extract_safetensor_keys(const std::filesystem::path& path, std::vector<std::string>* keys, std::string* error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        *error = "could not open " + path.string();
        return false;
    }
    uint64_t header_len = 0;
    input.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!input || header_len == 0 || header_len > (64ULL * 1024ULL * 1024ULL)) {
        *error = "invalid safetensors header length in " + path.string();
        return false;
    }
    std::string header(static_cast<size_t>(header_len), '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!input) {
        *error = "could not read safetensors header in " + path.string();
        return false;
    }

    size_t pos = 0;
    skip_ws(header, &pos);
    if (pos >= header.size() || header[pos] != '{') {
        *error = "safetensors header is not a JSON object in " + path.string();
        return false;
    }
    ++pos;
    while (pos < header.size()) {
        skip_ws(header, &pos);
        if (pos < header.size() && header[pos] == '}') {
            return true;
        }
        std::string key;
        if (!parse_json_string(header, &pos, &key)) {
            *error = "could not parse safetensors tensor key in " + path.string();
            return false;
        }
        skip_ws(header, &pos);
        if (pos >= header.size() || header[pos] != ':') {
            *error = "malformed safetensors header after key " + key;
            return false;
        }
        ++pos;
        if (key != "__metadata__") {
            keys->push_back(key);
        }
        if (!skip_json_value(header, &pos)) {
            *error = "could not skip safetensors value for key " + key;
            return false;
        }
        skip_ws(header, &pos);
        if (pos < header.size() && header[pos] == ',') {
            ++pos;
        }
    }
    *error = "unterminated safetensors header in " + path.string();
    return false;
}

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

void require_key(const std::set<std::string>& keys, const std::string& key, std::vector<std::string>* missing) {
    if (keys.find(key) == keys.end()) {
        missing->push_back(key);
    }
}

bool validate_tensor_inventory(
    const std::set<std::string>& keys,
    Gemma4ModelManifest* manifest,
    std::string* error) {
    std::vector<std::string> missing;
    require_key(keys, "language_model.model.embed_tokens.weight", &missing);
    require_key(keys, "language_model.model.embed_tokens.scales", &missing);
    require_key(keys, "language_model.model.embed_tokens.biases", &missing);
    require_key(keys, "language_model.model.norm.weight", &missing);

    for (uint32_t layer = 0; layer < manifest->num_hidden_layers; ++layer) {
        const std::string base = "language_model.model.layers." + std::to_string(layer);
        require_key(keys, base + ".input_layernorm.weight", &missing);
        require_key(keys, base + ".post_attention_layernorm.weight", &missing);
        require_key(keys, base + ".pre_feedforward_layernorm.weight", &missing);
        require_key(keys, base + ".post_feedforward_layernorm.weight", &missing);
        require_key(keys, base + ".layer_scalar", &missing);
        require_key(keys, base + ".self_attn.q_norm.weight", &missing);
        require_key(keys, base + ".self_attn.k_norm.weight", &missing);

        const bool full_attention = ((layer + 1) % 6) == 0;
        const char* projections[] = {
            ".self_attn.q_proj",
            ".self_attn.k_proj",
            ".self_attn.o_proj",
            ".mlp.gate_proj",
            ".mlp.up_proj",
            ".mlp.down_proj",
        };
        for (const char* projection : projections) {
            const std::string prefix = base + projection;
            require_key(keys, prefix + ".weight", &missing);
            require_key(keys, prefix + ".scales", &missing);
            require_key(keys, prefix + ".biases", &missing);
        }
        if (!full_attention) {
            const std::string prefix = base + ".self_attn.v_proj";
            require_key(keys, prefix + ".weight", &missing);
            require_key(keys, prefix + ".scales", &missing);
            require_key(keys, prefix + ".biases", &missing);
        }
    }

    if (!missing.empty()) {
        *error = "missing required Gemma 4 tensor: " + missing.front();
        return false;
    }

    for (const std::string& key : keys) {
        if (key.rfind("language_model.model.", 0) == 0) {
            ++manifest->language_tensor_count;
        } else {
            ++manifest->ignored_multimodal_tensor_count;
        }
        if (ends_with(key, ".weight")) {
            const std::string base = key.substr(0, key.size() - std::string(".weight").size());
            if (keys.find(base + ".scales") != keys.end() && keys.find(base + ".biases") != keys.end()) {
                ++manifest->quantized_linear_count;
            }
        }
    }

    return true;
}

bool validate_assistant_tensor_inventory(
    const std::set<std::string>& keys,
    Gemma4ModelManifest* manifest,
    std::string* error) {
    std::vector<std::string> missing;
    require_key(keys, "model.embed_tokens.weight", &missing);
    require_key(keys, "model.embed_tokens.scales", &missing);
    require_key(keys, "model.embed_tokens.biases", &missing);
    require_key(keys, "model.norm.weight", &missing);
    require_key(keys, "pre_projection.weight", &missing);
    require_key(keys, "pre_projection.scales", &missing);
    require_key(keys, "pre_projection.biases", &missing);
    require_key(keys, "post_projection.weight", &missing);
    require_key(keys, "post_projection.scales", &missing);
    require_key(keys, "post_projection.biases", &missing);

    for (uint32_t layer = 0; layer < manifest->num_hidden_layers; ++layer) {
        const std::string base = "model.layers." + std::to_string(layer);
        require_key(keys, base + ".input_layernorm.weight", &missing);
        require_key(keys, base + ".post_attention_layernorm.weight", &missing);
        require_key(keys, base + ".pre_feedforward_layernorm.weight", &missing);
        require_key(keys, base + ".post_feedforward_layernorm.weight", &missing);
        require_key(keys, base + ".layer_scalar", &missing);
        require_key(keys, base + ".self_attn.q_norm.weight", &missing);

        const char* projections[] = {
            ".self_attn.q_proj",
            ".self_attn.o_proj",
            ".mlp.gate_proj",
            ".mlp.up_proj",
            ".mlp.down_proj",
        };
        for (const char* projection : projections) {
            const std::string prefix = base + projection;
            require_key(keys, prefix + ".weight", &missing);
            require_key(keys, prefix + ".scales", &missing);
            require_key(keys, prefix + ".biases", &missing);
        }
    }

    if (!missing.empty()) {
        *error = "missing required Gemma 4 MTP assistant tensor: " + missing.front();
        return false;
    }

    for (const std::string& key : keys) {
        if (key.rfind("model.", 0) == 0 || key.rfind("pre_projection.", 0) == 0 ||
            key.rfind("post_projection.", 0) == 0) {
            ++manifest->language_tensor_count;
        } else {
            ++manifest->ignored_multimodal_tensor_count;
        }
        if (ends_with(key, ".weight")) {
            const std::string base = key.substr(0, key.size() - std::string(".weight").size());
            if (keys.find(base + ".scales") != keys.end() && keys.find(base + ".biases") != keys.end()) {
                ++manifest->quantized_linear_count;
            }
        }
    }

    return true;
}

bool collect_safetensor_keys(
    const std::filesystem::path& model_path,
    Gemma4ModelManifest* out,
    std::set<std::string>* keys,
    std::string* error) {
    std::vector<std::filesystem::path> files;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(model_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        *error = "no safetensors files found in " + model_path.string();
        return false;
    }

    std::vector<std::string> key_vector;
    for (const std::filesystem::path& file : files) {
        if (!extract_safetensor_keys(file, &key_vector, error)) {
            return false;
        }
    }
    out->safetensor_file_count = files.size();
    out->total_tensor_count = key_vector.size();

    *keys = std::set<std::string>(key_vector.begin(), key_vector.end());
    if (keys->size() != key_vector.size()) {
        *error = "duplicate tensor names across safetensor shards";
        return false;
    }
    return true;
}

} // namespace

std::string Gemma4ModelManifest::summary() const {
    std::ostringstream out;
    out << (is_assistant ? "Gemma4 MTP assistant manifest: layers=" : "Gemma4 text manifest: layers=")
        << num_hidden_layers
        << " hidden=" << hidden_size
        << " backbone_hidden=" << backbone_hidden_size
        << " kv_shared_layers=" << num_kv_shared_layers
        << " vocab=" << vocab_size
        << " quant=" << quantization_bits << "bit/group" << quantization_group_size
        << " quant_overrides=" << quantization_overrides.size()
        << " safetensors=" << safetensor_file_count
        << " tensors=" << total_tensor_count
        << " language_tensors=" << language_tensor_count
        << " quantized_linears=" << quantized_linear_count
        << " ignored_multimodal_tensors=" << ignored_multimodal_tensor_count;
    return out.str();
}

QuantizationSpec Gemma4ModelManifest::default_quantization() const {
    return QuantizationSpec{quantization_bits, quantization_group_size};
}

QuantizationSpec Gemma4ModelManifest::quantization_for(const std::string& prefix) const {
    const auto found = quantization_overrides.find(prefix);
    if (found != quantization_overrides.end()) {
        return found->second;
    }
    return default_quantization();
}

bool load_gemma4_model_manifest(
    const std::filesystem::path& model_path,
    Gemma4ModelManifest* out,
    std::string* error) {
    if (out == nullptr || error == nullptr) {
        return false;
    }
    *out = Gemma4ModelManifest{};
    error->clear();
    if (!parse_config(model_path, out, error)) {
        return false;
    }

    std::set<std::string> keys;
    if (!collect_safetensor_keys(model_path, out, &keys, error)) {
        return false;
    }
    if (!validate_tensor_inventory(keys, out, error)) {
        return false;
    }

    if (out->language_tensor_count != 1324 || out->quantized_linear_count != 332 ||
        (out->ignored_multimodal_tensor_count != 0 && out->ignored_multimodal_tensor_count != 17) ||
        out->total_tensor_count != out->language_tensor_count + out->ignored_multimodal_tensor_count) {
        *error =
            "Gemma 4 tensor inventory did not match expected 12B 4-bit counts: tensors=" +
            std::to_string(out->total_tensor_count) +
            " language_tensors=" + std::to_string(out->language_tensor_count) +
            " quantized_groups=" + std::to_string(out->quantized_linear_count) +
            " ignored_multimodal_tensors=" + std::to_string(out->ignored_multimodal_tensor_count);
        return false;
    }

    return true;
}

bool load_gemma4_mtp_assistant_manifest(
    const std::filesystem::path& model_path,
    Gemma4ModelManifest* out,
    std::string* error) {
    if (out == nullptr || error == nullptr) {
        return false;
    }
    *out = Gemma4ModelManifest{};
    error->clear();
    if (!parse_assistant_config(model_path, out, error)) {
        return false;
    }

    std::set<std::string> keys;
    if (!collect_safetensor_keys(model_path, out, &keys, error)) {
        return false;
    }
    if (!validate_assistant_tensor_inventory(keys, out, error)) {
        return false;
    }

    if (out->language_tensor_count != 94 || out->quantized_linear_count != 23 ||
        out->ignored_multimodal_tensor_count != 0 ||
        out->total_tensor_count != out->language_tensor_count) {
        *error =
            "Gemma 4 MTP assistant tensor inventory did not match expected 4-bit counts: tensors=" +
            std::to_string(out->total_tensor_count) +
            " assistant_tensors=" + std::to_string(out->language_tensor_count) +
            " quantized_groups=" + std::to_string(out->quantized_linear_count) +
            " ignored_tensors=" + std::to_string(out->ignored_multimodal_tensor_count);
        return false;
    }

    return true;
}

} // namespace gemma4d
