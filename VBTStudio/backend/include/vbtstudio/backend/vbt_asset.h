#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace vbtstudio::backend {

enum class FieldRole {
    Density,
    Flames,
    Temperature,
    Scientific,
    LevelSet,
    Unknown,
};

struct VbtAssetInfo {
    std::filesystem::path path;
    std::filesystem::path metadata_path;
    std::uint64_t file_bytes = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t offset_table_offset = 0;
    std::uint64_t payload_offset = 0;
    std::uint32_t version = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    std::uint32_t frames = 0;
    std::uint32_t leaf_size = 0;
    std::uint32_t coarse_resolution = 0;
    std::uint32_t leaf_count = 0;
    std::uint32_t profile_type = 0;
    float voxel_size = 1.0f;
    float background_value = 0.0f;
    std::string conversion_mode;
    std::array<std::int32_t, 3> bbox_min{};
    std::array<std::int32_t, 3> bbox_max{};
    FieldRole role = FieldRole::Unknown;
};

struct AssetOpenResult {
    std::optional<VbtAssetInfo> asset;
    std::string error;

    explicit operator bool() const noexcept { return asset.has_value(); }
};

AssetOpenResult inspect_vbt_asset(const std::filesystem::path& path);
std::string field_role_name(FieldRole role);

} // namespace vbtstudio::backend
