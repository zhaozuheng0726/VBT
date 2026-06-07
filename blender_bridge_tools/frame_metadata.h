#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vdbtools {

struct FrameMetadata {
    std::string sourceDir;
    std::string gridName = "density";
    std::string conversionMode = "fog";
    int width = 0;
    int height = 0;
    int depth = 0;
    int frames = 0;
    std::array<int, 3> bboxMin{0, 0, 0};
    std::array<int, 3> bboxMax{0, 0, 0};
    float dataMin = 0.0f;
    float dataMax = 0.0f;
    float voxelSize = 1.0f;
    float shellWidthVoxels = 1.5f;
    float halfWidth = 3.0f;
};

inline std::string readTextFile(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open metadata file: " + path.string());
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

inline std::string extractStringField(const std::string& text, const std::string& key, const std::string& fallback = "")
{
    const std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch m;
    if (std::regex_search(text, m, re)) return m[1].str();
    return fallback;
}

inline int extractIntField(const std::string& text, const std::string& key, int fallback = 0)
{
    const std::regex re("\"" + key + "\"\\s*:\\s*(-?\\d+)");
    std::smatch m;
    if (std::regex_search(text, m, re)) return std::stoi(m[1].str());
    return fallback;
}

inline float extractFloatField(const std::string& text, const std::string& key, float fallback = 0.0f)
{
    const std::regex re("\"" + key + "\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)");
    std::smatch m;
    if (std::regex_search(text, m, re)) return std::stof(m[1].str());
    return fallback;
}

inline std::array<int, 3> extractIntArray3(const std::string& text, const std::string& key)
{
    const std::regex re("\"" + key + "\"\\s*:\\s*\\[\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*\\]");
    std::smatch m;
    if (!std::regex_search(text, m, re)) {
        throw std::runtime_error("Missing array field in metadata: " + key);
    }
    return {std::stoi(m[1].str()), std::stoi(m[2].str()), std::stoi(m[3].str())};
}

inline std::array<int, 4> extractIntArray4OrDefault(const std::string& text, const std::string& key,
                                                    const std::array<int, 4>& fallback = {0, 0, 0, 0})
{
    const std::regex re("\"" + key + "\"\\s*:\\s*\\[\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*\\]");
    std::smatch m;
    if (!std::regex_search(text, m, re)) {
        return fallback;
    }
    return {std::stoi(m[1].str()), std::stoi(m[2].str()), std::stoi(m[3].str()), std::stoi(m[4].str())};
}

inline FrameMetadata loadFrameMetadata(const std::filesystem::path& path)
{
    const std::string text = readTextFile(path);
    FrameMetadata meta;
    meta.sourceDir = extractStringField(text, "source_dir");
    meta.gridName = extractStringField(text, "grid_name", "density");
    meta.conversionMode = extractStringField(text, "conversion_mode", "fog");
    meta.width = extractIntField(text, "width");
    meta.height = extractIntField(text, "height");
    meta.depth = extractIntField(text, "depth");
    meta.frames = extractIntField(text, "frames");
    if (meta.width <= 0 || meta.height <= 0 || meta.depth <= 0 || meta.frames <= 0) {
        const auto dims = extractIntArray4OrDefault(text, "dimensions");
        if (dims[0] > 0 && dims[1] > 0 && dims[2] > 0 && dims[3] > 0) {
            meta.width = dims[0];
            meta.height = dims[1];
            meta.depth = dims[2];
            meta.frames = dims[3];
        }
    }
    try {
        meta.bboxMin = extractIntArray3(text, "bbox_min");
        meta.bboxMax = extractIntArray3(text, "bbox_max");
    } catch (...) {
        meta.bboxMin = {0, 0, 0};
        meta.bboxMax = {std::max(0, meta.width - 1), std::max(0, meta.height - 1), std::max(0, meta.depth - 1)};
    }
    meta.dataMin = extractFloatField(text, "data_min", extractFloatField(text, "global_min"));
    meta.dataMax = extractFloatField(text, "data_max", extractFloatField(text, "global_max"));
    meta.voxelSize = extractFloatField(text, "voxel_size", 1.0f);
    meta.shellWidthVoxels = extractFloatField(text, "shell_width_voxels", 1.5f);
    meta.halfWidth = extractFloatField(text, "half_width_voxels", 3.0f);
    if (meta.width <= 0 || meta.height <= 0 || meta.depth <= 0 || meta.frames <= 0) {
        throw std::runtime_error("Invalid dimensions in metadata: " + path.string());
    }
    return meta;
}

inline size_t frameVoxelCount(const FrameMetadata& meta)
{
    return static_cast<size_t>(meta.width) * static_cast<size_t>(meta.height) * static_cast<size_t>(meta.depth);
}

} // namespace vdbtools
