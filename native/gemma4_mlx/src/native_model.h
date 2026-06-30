#pragma once

#include "model_manifest.h"
#include "gemma4_mlx.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace gemma4d {

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
        std::string* error) const;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace gemma4d
