#include "vbtstudio/backend/vbt_asset.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>

namespace vbtstudio::backend {
namespace {

constexpr std::uint64_t header_bytes = 8u + 11u * sizeof(std::uint32_t);

struct Header {
    std::array<char, 8> magic{};
    std::uint32_t version = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    std::uint32_t frames = 0;
    std::uint32_t leaf_size = 0;
    std::uint32_t coarse_resolution = 0;
    std::uint32_t max_coarse_keep = 0;
    std::uint32_t leaf_count = 0;
    std::uint32_t profile_type = 0;
    std::uint32_t coarse_scale_count = 0;
};

template <typename T>
bool read_value(std::ifstream& input, T& value)
{
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(input);
}

std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::filesystem::path find_metadata_path(const std::filesystem::path& asset_path)
{
    const auto expected = asset_path.parent_path() / (asset_path.stem().string() + ".metadata.json");
    std::error_code error_code;
    if (std::filesystem::is_regular_file(expected, error_code)) return expected;

    std::filesystem::path fallback;
    std::size_t count = 0;
    for (std::filesystem::directory_iterator iterator(asset_path.parent_path(), error_code), end;
         !error_code && iterator != end;
         iterator.increment(error_code)) {
        if (!iterator->is_regular_file(error_code)) continue;
        const std::string name = lowercase(iterator->path().filename().string());
        if (name.ends_with(".metadata.json")) {
            fallback = iterator->path();
            ++count;
        }
    }
    return count == 1 ? fallback : expected;
}

FieldRole infer_role(const std::filesystem::path& path, std::uint32_t profile_type)
{
    const std::string name = lowercase(path.stem().string());
    if (name.find("flame") != std::string::npos) return FieldRole::Flames;
    if (name.find("temperature") != std::string::npos) return FieldRole::Temperature;
    if (name.find("levelset") != std::string::npos || name.find("level_set") != std::string::npos) {
        return FieldRole::LevelSet;
    }
    if (name.find("density") != std::string::npos || profile_type == 1u) return FieldRole::Density;
    if (profile_type == 2u) return FieldRole::LevelSet;
    if (profile_type == 0u) return FieldRole::Scientific;
    return FieldRole::Unknown;
}

bool checked_multiply(std::uint64_t left, std::uint64_t right, std::uint64_t& result)
{
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

} // namespace

AssetOpenResult inspect_vbt_asset(const std::filesystem::path& path)
{
    AssetOpenResult result;
    std::error_code error_code;
    const auto canonical_path = std::filesystem::weakly_canonical(path, error_code);
    const auto resolved_path = error_code ? path : canonical_path;
    const auto file_size = std::filesystem::file_size(resolved_path, error_code);
    if (error_code || file_size < header_bytes) {
        result.error = "Unable to read VBT file size: " + resolved_path.string();
        return result;
    }

    std::ifstream input(resolved_path, std::ios::binary);
    Header header;
    if (!input || !read_value(input, header.magic) || !read_value(input, header.version) ||
        !read_value(input, header.width) || !read_value(input, header.height) ||
        !read_value(input, header.depth) || !read_value(input, header.frames) ||
        !read_value(input, header.leaf_size) || !read_value(input, header.coarse_resolution) ||
        !read_value(input, header.max_coarse_keep) || !read_value(input, header.leaf_count) ||
        !read_value(input, header.profile_type) || !read_value(input, header.coarse_scale_count)) {
        result.error = "Unable to read VBTPACK4 header: " + resolved_path.string();
        return result;
    }

    if (std::memcmp(header.magic.data(), "VBTPACK4", 8) != 0 || header.version != 4u) {
        result.error = "Unsupported file. Expected VBTPACK4 version 4.";
        return result;
    }
    if (header.width == 0 || header.height == 0 || header.depth == 0 || header.frames == 0 ||
        header.leaf_size != 8u || header.coarse_resolution == 0 || header.coarse_resolution > 8u ||
        header.profile_type > 2u) {
        result.error = "Invalid VBTPACK4 dimensions or profile metadata.";
        return result;
    }

    const std::uint64_t leaf_count_x = (header.width + header.leaf_size - 1u) / header.leaf_size;
    const std::uint64_t leaf_count_y = (header.height + header.leaf_size - 1u) / header.leaf_size;
    const std::uint64_t leaf_count_z = (header.depth + header.leaf_size - 1u) / header.leaf_size;
    std::uint64_t leaf_xy = 0;
    std::uint64_t expected_leaf_count = 0;
    if (!checked_multiply(leaf_count_x, leaf_count_y, leaf_xy) ||
        !checked_multiply(leaf_xy, leaf_count_z, expected_leaf_count) ||
        expected_leaf_count != header.leaf_count) {
        result.error = "VBTPACK4 leaf count does not match dimensions.";
        return result;
    }

    const std::uint64_t offset_table_position =
        header_bytes + static_cast<std::uint64_t>(header.coarse_scale_count) * sizeof(float);
    const std::uint64_t offset_table_bytes =
        (static_cast<std::uint64_t>(header.leaf_count) + 1u) * sizeof(std::uint32_t);
    if (offset_table_position + offset_table_bytes > file_size) {
        result.error = "VBTPACK4 offset table is truncated.";
        return result;
    }

    input.seekg(static_cast<std::streamoff>(offset_table_position + offset_table_bytes - sizeof(std::uint32_t)));
    std::uint32_t payload_words = 0;
    if (!read_value(input, payload_words)) {
        result.error = "Unable to read the final VBTPACK4 payload offset.";
        return result;
    }
    const std::uint64_t payload_bytes = static_cast<std::uint64_t>(payload_words) * sizeof(std::uint32_t);
    const std::uint64_t expected_file_size = offset_table_position + offset_table_bytes + payload_bytes;
    if (expected_file_size != file_size) {
        result.error = "VBTPACK4 payload size does not match the file size.";
        return result;
    }

    VbtAssetInfo asset;
    asset.path = resolved_path;
    asset.metadata_path = find_metadata_path(resolved_path);
    asset.file_bytes = file_size;
    asset.payload_bytes = payload_bytes;
    asset.offset_table_offset = offset_table_position;
    asset.payload_offset = offset_table_position + offset_table_bytes;
    asset.version = header.version;
    asset.width = header.width;
    asset.height = header.height;
    asset.depth = header.depth;
    asset.frames = header.frames;
    asset.leaf_size = header.leaf_size;
    asset.coarse_resolution = header.coarse_resolution;
    asset.leaf_count = header.leaf_count;
    asset.profile_type = header.profile_type;
    asset.role = infer_role(resolved_path, header.profile_type);
    asset.bbox_max = {
        static_cast<std::int32_t>(header.width - 1u),
        static_cast<std::int32_t>(header.height - 1u),
        static_cast<std::int32_t>(header.depth - 1u),
    };
    if (std::ifstream metadata(asset.metadata_path); metadata) {
        try {
            const nlohmann::json document = nlohmann::json::parse(metadata);
            const auto minimum = document.at("bbox_min").get<std::array<std::int32_t, 3>>();
            const auto maximum = document.at("bbox_max").get<std::array<std::int32_t, 3>>();
            const std::array<std::uint32_t, 3> dimensions{header.width, header.height, header.depth};
            bool valid = true;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                valid = valid && maximum[axis] >= minimum[axis] &&
                        static_cast<std::uint32_t>(maximum[axis] - minimum[axis] + 1) == dimensions[axis];
            }
            if (valid) {
                asset.bbox_min = minimum;
                asset.bbox_max = maximum;
            }
            asset.voxel_size = document.value("voxel_size", 1.0f);
            asset.background_value = document.value("data_max", 0.0f);
            if (asset.background_value <= 0.0f) {
                asset.background_value = document.value("half_width_voxels", 0.0f) * asset.voxel_size;
            }
            asset.conversion_mode = lowercase(document.value("conversion_mode", std::string{}));
            if (asset.conversion_mode == "levelset") asset.role = FieldRole::LevelSet;
        }
        catch (...) {
        }
    }
    result.asset = std::move(asset);
    return result;
}

std::string field_role_name(FieldRole role)
{
    switch (role) {
    case FieldRole::Density: return "Density";
    case FieldRole::Flames: return "Flames";
    case FieldRole::Temperature: return "Temperature";
    case FieldRole::Scientific: return "Scientific";
    case FieldRole::LevelSet: return "Level Set";
    case FieldRole::Unknown: return "Unknown";
    }
    return "Unknown";
}

} // namespace vbtstudio::backend
