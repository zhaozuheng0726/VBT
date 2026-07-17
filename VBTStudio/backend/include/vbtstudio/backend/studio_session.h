#pragma once

#include "vbtstudio/backend/camera.h"
#include "vbtstudio/backend/material.h"
#include "vbtstudio/backend/timeline.h"
#include "vbtstudio/backend/vbt_asset.h"

#include <filesystem>
#include <optional>
#include <string>

namespace vbtstudio::backend {

class StudioSession {
public:
    bool open_asset(const std::filesystem::path& path);
    bool add_field(const std::filesystem::path& path);
    void update(double elapsed_seconds);

    [[nodiscard]] const std::optional<VbtAssetInfo>& asset() const noexcept;
    [[nodiscard]] const std::string& last_error() const noexcept;
    [[nodiscard]] const std::optional<VbtAssetInfo>& secondary_field() const noexcept;
    [[nodiscard]] const std::optional<VbtAssetInfo>& temperature_field() const noexcept;

    Timeline& timeline() noexcept;
    const Timeline& timeline() const noexcept;
    MaterialState& material() noexcept;
    const MaterialState& material() const noexcept;
    CameraState& camera() noexcept;
    const CameraState& camera() const noexcept;

private:
    std::optional<VbtAssetInfo> asset_;
    std::optional<VbtAssetInfo> secondary_field_;
    std::optional<VbtAssetInfo> temperature_field_;
    std::string last_error_;
    Timeline timeline_;
    MaterialState material_;
    CameraState camera_;
};

} // namespace vbtstudio::backend
