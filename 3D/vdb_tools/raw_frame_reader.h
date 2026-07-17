#pragma once

#include "frame_metadata.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace vdbtools {

inline void readExact(std::ifstream& input, float* destination, size_t valueCount)
{
    const size_t byteCount = checkedMultiplySize(valueCount, sizeof(float), "RAW read size");
    if (byteCount > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("RAW read size exceeds streamsize");
    }
    input.read(
        reinterpret_cast<char*>(destination), static_cast<std::streamsize>(byteCount));
    if (!input) {
        throw std::runtime_error("Failed to read requested data from RAW file");
    }
}

inline void validateRawFileSize(const std::filesystem::path& path, const FrameMetadata& meta)
{
    std::error_code error;
    const uintmax_t actualBytes = std::filesystem::file_size(path, error);
    if (error) {
        throw std::runtime_error(
            "Failed to query RAW file size '" + path.string() + "': " + error.message());
    }
    const uintmax_t expectedBytes = rawByteCount(meta);
    if (actualBytes != expectedBytes) {
        throw std::runtime_error(
            "RAW file size mismatch: expected " + std::to_string(expectedBytes) +
            " bytes, got " + std::to_string(actualBytes));
    }
}

inline std::vector<float> loadRawFrame(const std::filesystem::path& path,
                                       const FrameMetadata& meta,
                                       int frameIndex)
{
    if (frameIndex < 0 || frameIndex >= meta.frames) {
        throw std::runtime_error(
            "Frame index out of range: " + std::to_string(frameIndex) +
            " / " + std::to_string(meta.frames));
    }
    validateRawFileSize(path, meta);

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open RAW file: " + path.string());
    }

    const size_t voxelCount = frameVoxelCount(meta);
    std::vector<float> frame(voxelCount, 0.0f);
    if (meta.rawStorageOrder == RawStorageOrder::FrameMajor) {
        const size_t frameBytes = checkedMultiplySize(voxelCount, sizeof(float), "frame byte count");
        const size_t offsetBytes = checkedMultiplySize(
            static_cast<size_t>(frameIndex), frameBytes, "frame byte offset");
        if (offsetBytes > static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
            throw std::runtime_error("RAW frame offset exceeds streamoff");
        }
        input.seekg(static_cast<std::streamoff>(offsetBytes), std::ios::beg);
        if (!input) {
            throw std::runtime_error("Failed to seek to requested RAW frame");
        }
        readExact(input, frame.data(), voxelCount);
        return frame;
    }

    const size_t rowValueCount = checkedMultiplySize(
        static_cast<size_t>(meta.width), static_cast<size_t>(meta.frames), "T-fastest row size");
    std::vector<float> row(rowValueCount, 0.0f);
    for (int z = 0; z < meta.depth; ++z) {
        for (int y = 0; y < meta.height; ++y) {
            readExact(input, row.data(), row.size());
            const size_t destinationRow =
                (static_cast<size_t>(z) * static_cast<size_t>(meta.height) +
                 static_cast<size_t>(y)) *
                static_cast<size_t>(meta.width);
            for (int x = 0; x < meta.width; ++x) {
                frame[destinationRow + static_cast<size_t>(x)] =
                    row[static_cast<size_t>(x) * static_cast<size_t>(meta.frames) +
                        static_cast<size_t>(frameIndex)];
            }
        }
    }
    return frame;
}

} // namespace vdbtools
