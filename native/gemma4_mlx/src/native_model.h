#pragma once

#include "model_manifest.h"
#include "gemma4_mlx.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gemma4d {

class NativeHiddenState {
public:
    struct Impl;

    ~NativeHiddenState();

    NativeHiddenState(const NativeHiddenState&) = delete;
    NativeHiddenState& operator=(const NativeHiddenState&) = delete;
    NativeHiddenState(NativeHiddenState&&) noexcept;
    NativeHiddenState& operator=(NativeHiddenState&&) noexcept;

    uint64_t sequence_len() const;
    uint32_t hidden_size() const;
    bool has_shared_kv() const;
    std::unique_ptr<NativeHiddenState> clone() const;

private:
    explicit NativeHiddenState(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;

    friend class NativeMtpAssistantModel;
    friend class NativeKvState;
    friend class NativeTextModel;
};

class NativeKvState {
public:
    struct Impl;

    NativeKvState();
    ~NativeKvState();

    NativeKvState(const NativeKvState&) = delete;
    NativeKvState& operator=(const NativeKvState&) = delete;
    NativeKvState(NativeKvState&&) noexcept;
    NativeKvState& operator=(NativeKvState&&) noexcept;

    void clear();
    uint64_t sequence_len() const;
    uint64_t active_bytes() const;
    std::unique_ptr<NativeKvState> clone() const;
    bool save_safetensors(
        const std::filesystem::path& payload_path,
        const NativeHiddenState* last_hidden,
        const std::unordered_map<std::string, std::string>& metadata,
        std::string* error) const;
    bool save_compressed_safetensors(
        const std::filesystem::path& payload_path,
        const NativeHiddenState* last_hidden,
        const std::unordered_map<std::string, std::string>& metadata,
        Gemma4KvMode mode,
        bool compress_global_layers,
        bool compress_sliding_layers,
        std::string* error) const;

    static bool load_safetensors(
        const std::filesystem::path& payload_path,
        std::unique_ptr<NativeKvState>* kv_state,
        std::unique_ptr<NativeHiddenState>* last_hidden,
        std::unordered_map<std::string, std::string>* metadata,
        std::string* error);

private:
    std::unique_ptr<Impl> impl_;

    friend class NativeTextModel;
};

class NativeTextModel {
public:
    struct Impl;

    NativeTextModel();
    ~NativeTextModel();

    NativeTextModel(const NativeTextModel&) = delete;
    NativeTextModel& operator=(const NativeTextModel&) = delete;
    NativeTextModel(NativeTextModel&&) noexcept;
    NativeTextModel& operator=(NativeTextModel&&) noexcept;

    static bool load(
        const std::filesystem::path& model_path,
        const Gemma4ModelManifest& manifest,
        std::unique_ptr<NativeTextModel>* out,
        std::string* error);

    size_t tensor_count() const;
    std::string summary() const;

    bool forward_greedy(
        const std::vector<int32_t>& tokens,
        Gemma4StepResult* out,
        std::string* error,
        std::unique_ptr<NativeHiddenState>* last_hidden = nullptr) const;

    bool prefill_incremental(
        const std::vector<int32_t>& tokens,
        Gemma4StepResult* out,
        std::string* error,
        std::unique_ptr<NativeKvState>* kv_state,
        std::unique_ptr<NativeHiddenState>* last_hidden = nullptr) const;

    bool decode_incremental(
        int32_t token,
        NativeKvState* kv_state,
        Gemma4StepResult* out,
        std::string* error,
        std::unique_ptr<NativeHiddenState>* last_hidden = nullptr) const;

    bool verify_draft_block(
        const std::vector<int32_t>& context_tokens,
        const int32_t* draft_tokens,
        size_t draft_count,
        std::vector<int32_t>* committed_tokens,
        Gemma4StepResult* out,
        std::string* error,
        std::unique_ptr<NativeHiddenState>* last_hidden = nullptr) const;

private:
    std::unique_ptr<Impl> impl_;

    friend class NativeMtpAssistantModel;
};

class NativeMtpAssistantModel {
public:
    struct Impl;

    NativeMtpAssistantModel();
    ~NativeMtpAssistantModel();

    NativeMtpAssistantModel(const NativeMtpAssistantModel&) = delete;
    NativeMtpAssistantModel& operator=(const NativeMtpAssistantModel&) = delete;
    NativeMtpAssistantModel(NativeMtpAssistantModel&&) noexcept;
    NativeMtpAssistantModel& operator=(NativeMtpAssistantModel&&) noexcept;

    static bool load(
        const std::filesystem::path& model_path,
        const Gemma4ModelManifest& manifest,
        std::unique_ptr<NativeMtpAssistantModel>* out,
        std::string* error);

    size_t tensor_count() const;
    std::string summary() const;

    bool draft_block(
        const NativeTextModel& target_model,
        const NativeHiddenState& last_hidden,
        const std::vector<int32_t>& context_tokens,
        uint32_t block_size,
        int32_t* out_tokens,
        size_t* inout_count,
        std::string* error) const;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace gemma4d
