#pragma once

#include "../../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace vdbtools {

enum class RawStorageOrder {
    FrameMajor,
    TimeFastest,
};

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
    std::array<double, 16> indexToWorld{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0};
    bool indexToWorldDeclared = false;
    float shellWidthVoxels = 1.5f;
    float halfWidth = 3.0f;
    std::string dataType = "FLOAT32";
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

inline std::string lowerAscii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
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

inline int readIntOr(const nlohmann::json& document,
                     const char* key,
                     int fallback = 0,
                     bool* found = nullptr)
{
    const auto it = document.find(key);
    if (it == document.end() || it->is_null()) {
        if (found) *found = false;
        return fallback;
    }
    if (!it->is_number_integer()) {
        throw std::runtime_error(std::string("Metadata field must be an integer: ") + key);
    }
    try {
        const int value = it->get<int>();
        if (found) *found = true;
        return value;
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("Metadata integer is out of range: ") + key);
    }
}

inline float readFloatOr(const nlohmann::json& document,
                         const char* key,
                         float fallback = 0.0f,
                         bool* found = nullptr)
{
    const auto it = document.find(key);
    if (it == document.end() || it->is_null()) {
        if (found) *found = false;
        return fallback;
    }
    if (!it->is_number()) {
        throw std::runtime_error(std::string("Metadata field must be numeric: ") + key);
    }

    double value = 0.0;
    try {
        value = it->get<double>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("Metadata number is out of range: ") + key);
    }
    const double floatMax = static_cast<double>(std::numeric_limits<float>::max());
    if (!std::isfinite(value) || value < -floatMax || value > floatMax) {
        throw std::runtime_error(std::string("Metadata number is not a finite float: ") + key);
    }
    if (found) *found = true;
    return static_cast<float>(value);
}

inline float readFloatAlias(const nlohmann::json& document,
                            const char* primary,
                            const char* secondary,
                            float fallback,
                            bool* found = nullptr)
{
    bool hasPrimary = false;
    bool hasSecondary = false;
    const float primaryValue = readFloatOr(document, primary, fallback, &hasPrimary);
    const float secondaryValue = readFloatOr(document, secondary, fallback, &hasSecondary);
    if (hasPrimary && hasSecondary && primaryValue != secondaryValue) {
        throw std::runtime_error(
            std::string("Metadata fields disagree: ") + primary + " and " + secondary);
    }
    if (found) *found = hasPrimary || hasSecondary;
    if (hasPrimary) return primaryValue;
    if (hasSecondary) return secondaryValue;
    return fallback;
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
        try {
            result[i] = (*it)[i].get<int>();
        } catch (const nlohmann::json::exception&) {
            throw std::runtime_error(std::string("Metadata array integer is out of range: ") + key);
        }
    }
    if (found) *found = true;
    return result;
}

template <size_t N>
inline std::array<double, N> readNumberArray(const nlohmann::json& document,
                                             const char* key,
                                             const std::array<double, N>& fallback,
                                             bool* found = nullptr)
{
    const auto it = document.find(key);
    if (it == document.end() || it->is_null()) {
        if (found) *found = false;
        return fallback;
    }
    if (!it->is_array() || it->size() != N) {
        throw std::runtime_error(std::string("Metadata field must be a numeric array of length ") +
                                 std::to_string(N) + ": " + key);
    }

    std::array<double, N> result{};
    for (size_t i = 0; i < N; ++i) {
        if (!(*it)[i].is_number()) {
            throw std::runtime_error(std::string("Metadata array contains a non-numeric value: ") + key);
        }
        result[i] = (*it)[i].get<double>();
        if (!std::isfinite(result[i])) {
            throw std::runtime_error(std::string("Metadata array contains a non-finite value: ") + key);
        }
    }
    if (found) *found = true;
    return result;
}

inline size_t checkedMultiplySize(size_t lhs, size_t rhs, const char* description)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::runtime_error(std::string("Metadata size overflows size_t: ") + description);
    }
    return lhs * rhs;
}

inline size_t frameVoxelCount(const FrameMetadata& meta)
{
    size_t count = checkedMultiplySize(
        static_cast<size_t>(meta.width), static_cast<size_t>(meta.height), "frame voxel count");
    return checkedMultiplySize(count, static_cast<size_t>(meta.depth), "frame voxel count");
}

inline size_t totalVoxelCount(const FrameMetadata& meta)
{
    return checkedMultiplySize(
        frameVoxelCount(meta), static_cast<size_t>(meta.frames), "total voxel count");
}

inline uintmax_t rawByteCount(const FrameMetadata& meta)
{
    const size_t bytes = checkedMultiplySize(totalVoxelCount(meta), sizeof(float), "RAW byte count");
    return static_cast<uintmax_t>(bytes);
}

inline FrameMetadata loadFrameMetadata(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open metadata file: " + path.string());
    }

    nlohmann::json document;
    try {
        in >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Failed to parse metadata JSON '" + path.string() + "': " + error.what());
    }
    if (!document.is_object()) {
        throw std::runtime_error("Metadata root must be a JSON object: " + path.string());
    }

    FrameMetadata meta;
    meta.sourceDir = readStringOr(document, "source_dir");
    meta.gridName = readStringOr(document, "grid_name", "density");
    if (meta.gridName.empty()) {
        throw std::runtime_error("Metadata grid_name must not be empty: " + path.string());
    }

    std::array<bool, 4> namedFound{};
    meta.width = readIntOr(document, "width", 0, &namedFound[0]);
    meta.height = readIntOr(document, "height", 0, &namedFound[1]);
    meta.depth = readIntOr(document, "depth", 0, &namedFound[2]);
    meta.frames = readIntOr(document, "frames", 0, &namedFound[3]);
    const std::array<int, 4> named{meta.width, meta.height, meta.depth, meta.frames};
    for (size_t i = 0; i < named.size(); ++i) {
        if (namedFound[i] && named[i] <= 0) {
            throw std::runtime_error("Named metadata dimensions must be positive: " + path.string());
        }
    }

    bool hasDimensions = false;
    const auto dimensions = readIntArray<4>(
        document, "dimensions", {0, 0, 0, 0}, &hasDimensions);
    if (hasDimensions) {
        for (int dimension : dimensions) {
            if (dimension <= 0) {
                throw std::runtime_error("Metadata dimensions must be positive: " + path.string());
            }
        }
        for (size_t i = 0; i < dimensions.size(); ++i) {
            if (namedFound[i] && named[i] != dimensions[i]) {
                throw std::runtime_error(
                    "Named metadata dimensions disagree with dimensions[]: " + path.string());
            }
        }
        meta.width = dimensions[0];
        meta.height = dimensions[1];
        meta.depth = dimensions[2];
        meta.frames = dimensions[3];
    }
    if (meta.width <= 0 || meta.height <= 0 || meta.depth <= 0 || meta.frames <= 0) {
        throw std::runtime_error("Invalid metadata dimensions: " + path.string());
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
            throw std::runtime_error(
                "Metadata time_is_fastest_dimension must be boolean: " + path.string());
        }
        meta.timeFastestDeclared = true;
        meta.rawStorageOrder =
            fastestIt->get<bool>() ? RawStorageOrder::TimeFastest : RawStorageOrder::FrameMajor;
    }

    meta.indexToWorld = readNumberArray<16>(
        document, "index_to_world", meta.indexToWorld, &meta.indexToWorldDeclared);
    if (meta.indexToWorldDeclared &&
        (std::abs(meta.indexToWorld[3]) > 1.0e-12 ||
         std::abs(meta.indexToWorld[7]) > 1.0e-12 ||
         std::abs(meta.indexToWorld[11]) > 1.0e-12 ||
         std::abs(meta.indexToWorld[15] - 1.0) > 1.0e-12)) {
        throw std::runtime_error("Metadata index_to_world must be affine: " + path.string());
    }

    bool hasBboxMin = false;
    bool hasBboxMax = false;
    const auto bboxMin = readIntArray<3>(document, "bbox_min", {0, 0, 0}, &hasBboxMin);
    const auto bboxMax = readIntArray<3>(document, "bbox_max", {0, 0, 0}, &hasBboxMax);
    if (hasBboxMin != hasBboxMax) {
        throw std::runtime_error(
            "Metadata bbox_min and bbox_max must be provided together: " + path.string());
    }
    if (hasBboxMin) {
        const std::array<int, 3> spatialDimensions{meta.width, meta.height, meta.depth};
        for (size_t axis = 0; axis < 3; ++axis) {
            const int64_t extent =
                static_cast<int64_t>(bboxMax[axis]) - static_cast<int64_t>(bboxMin[axis]) + 1;
            if (extent <= 0) {
                throw std::runtime_error("Metadata bbox is inverted: " + path.string());
            }
            if (extent != spatialDimensions[axis]) {
                throw std::runtime_error(
                    "Metadata bbox extent disagrees with dimensions: " + path.string());
            }
        }
        meta.bboxMin = bboxMin;
        meta.bboxMax = bboxMax;
    } else {
        meta.bboxMin = {0, 0, 0};
        meta.bboxMax = {meta.width - 1, meta.height - 1, meta.depth - 1};
    }

    bool hasDataMin = false;
    bool hasDataMax = false;
    meta.dataMin = readFloatAlias(document, "data_min", "global_min", 0.0f, &hasDataMin);
    meta.dataMax = readFloatAlias(document, "data_max", "global_max", 0.0f, &hasDataMax);
    if (hasDataMin && hasDataMax && meta.dataMin > meta.dataMax) {
        throw std::runtime_error("Metadata data_min must be <= data_max: " + path.string());
    }

    meta.voxelSize = readFloatOr(document, "voxel_size", 1.0f);
    if (!(meta.voxelSize > 0.0f)) {
        throw std::runtime_error("Metadata voxel_size must be positive: " + path.string());
    }
    meta.shellWidthVoxels = readFloatOr(document, "shell_width_voxels", 1.5f);
    if (meta.shellWidthVoxels < 0.0f) {
        throw std::runtime_error(
            "Metadata shell_width_voxels must be non-negative: " + path.string());
    }
    meta.halfWidth = readFloatOr(document, "half_width_voxels", 3.0f);
    if (!(meta.halfWidth > 0.0f)) {
        throw std::runtime_error(
            "Metadata half_width_voxels must be positive: " + path.string());
    }

    meta.conversionMode = lowerAscii(readStringOr(document, "conversion_mode", "fog"));
    if (meta.conversionMode != "fog" &&
        meta.conversionMode != "interior" &&
        meta.conversionMode != "shell" &&
        meta.conversionMode != "levelset") {
        throw std::runtime_error(
            "Unsupported metadata conversion_mode; expected fog|interior|shell|levelset: " +
            path.string());
    }

    meta.dataType = upperAscii(readStringOr(document, "data_type", "float32"));
    if (meta.dataType != "FLOAT32") {
        throw std::runtime_error("Unsupported metadata data_type; expected float32: " + path.string());
    }

    const auto frameFilesIt = document.find("frame_files");
    if (frameFilesIt != document.end() && !frameFilesIt->is_null()) {
        if (!frameFilesIt->is_array() ||
            frameFilesIt->size() != static_cast<size_t>(meta.frames)) {
            throw std::runtime_error(
                "Metadata frame_files must be an array with one entry per frame: " + path.string());
        }
        for (const auto& frameFile : *frameFilesIt) {
            if (!frameFile.is_string()) {
                throw std::runtime_error(
                    "Metadata frame_files entries must be strings: " + path.string());
            }
        }
    }

    (void)rawByteCount(meta);
    return meta;
}

} // namespace vdbtools
