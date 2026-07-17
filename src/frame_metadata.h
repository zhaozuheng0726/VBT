#pragma once

#include "json.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace vbt {

enum class RawStorageOrder {
    FrameMajor,
    TimeFastest,
};

struct FrameMetadata {
    std::string sourceDir;
    std::string gridName = "density";
    int width = 0;
    int height = 0;
    int depth = 0;
    int frames = 0;
    std::array<int, 3> bboxMin{0, 0, 0};
    std::array<int, 3> bboxMax{0, 0, 0};
    float dataMin = 0.0f;
    float dataMax = 0.0f;
    std::string conversionMode;
    std::string sourceType;
    float shellWidthVoxels = 0.0f;
    float voxelSize = 1.0f;
    std::string dataType = "float32";
    std::array<std::string, 4> axisOrder{"X", "Y", "Z", "T"};
    bool axisOrderDeclared = false;
    bool timeFastestDeclared = false;
    RawStorageOrder rawStorageOrder = RawStorageOrder::FrameMajor;
};

inline std::string upperAscii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

inline std::string readStringOr(const nlohmann::json& document,
                                const char* key,
                                const std::string& fallback = {})
{
    const auto it = document.find(key);
    if (it == document.end() || it->is_null()) return fallback;
    if (!it->is_string()) {
        throw std::runtime_error(std::string("Metadata field must be a string: ") + key);
    }
    return it->get<std::string>();
}

inline int readIntOr(const nlohmann::json& document, const char* key, int fallback = 0)
{
    const auto it = document.find(key);
    if (it == document.end() || it->is_null()) return fallback;
    if (!it->is_number_integer()) {
        throw std::runtime_error(std::string("Metadata field must be an integer: ") + key);
    }
    return it->get<int>();
}

inline float readFloatOr(const nlohmann::json& document, const char* key, float fallback = 0.0f)
{
    const auto it = document.find(key);
    if (it == document.end() || it->is_null()) return fallback;
    if (!it->is_number()) {
        throw std::runtime_error(std::string("Metadata field must be numeric: ") + key);
    }
    return it->get<float>();
}

template <size_t N>
inline std::array<int, N> readIntArray(const nlohmann::json& document,
                                      const char* key,
                                      const std::array<int, N>& fallback,
                                      bool* found = nullptr)
{
    const auto it = document.find(key);
    if (it == document.end() || it->is_null()) {
        if (found) *found = false;
        return fallback;
    }
    if (!it->is_array() || it->size() != N) {
        throw std::runtime_error(std::string("Metadata field must be an integer array of length ") +
                                 std::to_string(N) + ": " + key);
    }

    std::array<int, N> result{};
    for (size_t i = 0; i < N; ++i) {
        if (!(*it)[i].is_number_integer()) {
            throw std::runtime_error(std::string("Metadata array contains a non-integer value: ") + key);
        }
        result[i] = (*it)[i].get<int>();
    }
    if (found) *found = true;
    return result;
}

inline FrameMetadata loadFrameMetadata(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open metadata file: " + path.string());

    nlohmann::json document;
    try {
        in >> document;
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("Failed to parse metadata JSON '" + path.string() + "': " + e.what());
    }
    if (!document.is_object()) {
        throw std::runtime_error("Metadata root must be a JSON object: " + path.string());
    }

    FrameMetadata meta;
    meta.sourceDir = readStringOr(document, "source_dir");
    meta.gridName = readStringOr(document, "grid_name", "density");
    meta.width = readIntOr(document, "width");
    meta.height = readIntOr(document, "height");
    meta.depth = readIntOr(document, "depth");
    meta.frames = readIntOr(document, "frames");

    bool hasDimensions = false;
    const auto dims = readIntArray<4>(document, "dimensions", {0, 0, 0, 0}, &hasDimensions);
    if (hasDimensions) {
        for (int dim : dims) {
            if (dim <= 0) {
                throw std::runtime_error("Metadata dimensions must be positive: " + path.string());
            }
        }
        const std::array<int, 4> named{meta.width, meta.height, meta.depth, meta.frames};
        for (size_t i = 0; i < dims.size(); ++i) {
            if (named[i] > 0 && named[i] != dims[i]) {
                throw std::runtime_error("Named metadata dimensions disagree with dimensions[]: " + path.string());
            }
        }
        meta.width = dims[0];
        meta.height = dims[1];
        meta.depth = dims[2];
        meta.frames = dims[3];
    }

    const auto axisIt = document.find("axis_order");
    if (axisIt != document.end() && !axisIt->is_null()) {
        if (!axisIt->is_array() || axisIt->size() != 4) {
            throw std::runtime_error("Metadata axis_order must contain X, Y, Z, T: " + path.string());
        }
        const std::array<std::string, 4> expected{"X", "Y", "Z", "T"};
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!(*axisIt)[i].is_string()) {
                throw std::runtime_error("Metadata axis_order entries must be strings: " + path.string());
            }
            meta.axisOrder[i] = upperAscii((*axisIt)[i].get<std::string>());
            if (meta.axisOrder[i] != expected[i]) {
                throw std::runtime_error(
                    "Unsupported metadata axis_order; expected [X,Y,Z,T]: " + path.string());
            }
        }
        meta.axisOrderDeclared = true;
    }

    const auto fastestIt = document.find("time_is_fastest_dimension");
    if (fastestIt != document.end() && !fastestIt->is_null()) {
        if (!fastestIt->is_boolean()) {
            throw std::runtime_error("Metadata time_is_fastest_dimension must be boolean: " + path.string());
        }
        meta.timeFastestDeclared = true;
        meta.rawStorageOrder =
            fastestIt->get<bool>() ? RawStorageOrder::TimeFastest : RawStorageOrder::FrameMajor;
    }

    meta.bboxMin = readIntArray<3>(document, "bbox_min", {0, 0, 0});
    meta.bboxMax = readIntArray<3>(
        document,
        "bbox_max",
        {std::max(0, meta.width - 1), std::max(0, meta.height - 1), std::max(0, meta.depth - 1)});
    meta.dataMin = readFloatOr(document, "data_min", readFloatOr(document, "global_min"));
    meta.dataMax = readFloatOr(document, "data_max", readFloatOr(document, "global_max"));
    meta.conversionMode = readStringOr(document, "conversion_mode");
    meta.sourceType = readStringOr(document, "source_type");
    meta.shellWidthVoxels = readFloatOr(document, "shell_width_voxels", 0.0f);
    meta.voxelSize = readFloatOr(document, "voxel_size", 1.0f);
    meta.dataType = upperAscii(readStringOr(document, "data_type", "float32"));

    if (meta.width <= 0 || meta.height <= 0 || meta.depth <= 0 || meta.frames <= 0) {
        throw std::runtime_error("Invalid metadata dimensions: " + path.string());
    }
    return meta;
}

inline size_t frameVoxelCount(const FrameMetadata& meta)
{
    return static_cast<size_t>(meta.width) * static_cast<size_t>(meta.height) * static_cast<size_t>(meta.depth);
}

} // namespace vbt
