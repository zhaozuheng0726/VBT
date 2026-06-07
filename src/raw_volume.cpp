#include "raw_volume.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace vbt {

size_t RawVolume4D::frameVoxelCount() const
{
    return static_cast<size_t>(meta.width) * static_cast<size_t>(meta.height) * static_cast<size_t>(meta.depth);
}

size_t RawVolume4D::totalVoxelCount() const
{
    return frameVoxelCount() * static_cast<size_t>(meta.frames);
}

float RawVolume4D::at(int x, int y, int z, int t) const
{
    const size_t frameStride = frameVoxelCount();
    const size_t idx =
        static_cast<size_t>(t) * frameStride +
        (static_cast<size_t>(z) * static_cast<size_t>(meta.height) + static_cast<size_t>(y)) * static_cast<size_t>(meta.width) +
        static_cast<size_t>(x);
    return values[idx];
}

std::filesystem::path guessMetadataPathForRaw(const std::filesystem::path& rawPath)
{
    const auto direct = rawPath.parent_path() / (rawPath.stem().string() + ".metadata.json");
    if (std::filesystem::exists(direct)) return direct;
    return {};
}

RawVolume4D loadRawVolume(const std::filesystem::path& rawPath,
                          const std::filesystem::path& metadataPath)
{
    RawVolume4D volume;
    volume.meta = loadFrameMetadata(metadataPath);

    std::ifstream in(rawPath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open raw file: " + rawPath.string());
    }

    volume.values.resize(volume.totalVoxelCount());
    in.read(reinterpret_cast<char*>(volume.values.data()),
            static_cast<std::streamsize>(volume.values.size() * sizeof(float)));
    if (!in) {
        throw std::runtime_error("Failed to read raw data: " + rawPath.string());
    }
    return volume;
}

RawVolume4D cropRawVolume(const RawVolume4D& source,
                          int x0, int y0, int z0,
                          int width, int height, int depth)
{
    if (width <= 0 || height <= 0 || depth <= 0) {
        throw std::runtime_error("Invalid crop size");
    }
    RawVolume4D cropped;
    cropped.meta = source.meta;
    cropped.meta.width = width;
    cropped.meta.height = height;
    cropped.meta.depth = depth;
    cropped.meta.bboxMin = {
        source.meta.bboxMin[0] + x0,
        source.meta.bboxMin[1] + y0,
        source.meta.bboxMin[2] + z0
    };
    cropped.meta.bboxMax = {
        cropped.meta.bboxMin[0] + width - 1,
        cropped.meta.bboxMin[1] + height - 1,
        cropped.meta.bboxMin[2] + depth - 1
    };
    cropped.values.resize(cropped.totalVoxelCount());

    float minValue = std::numeric_limits<float>::infinity();
    float maxValue = -std::numeric_limits<float>::infinity();
    for (int t = 0; t < source.meta.frames; ++t) {
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const float v = source.at(x0 + x, y0 + y, z0 + z, t);
                    const size_t idx =
                        static_cast<size_t>(t) * static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(depth) +
                        (static_cast<size_t>(z) * static_cast<size_t>(height) + static_cast<size_t>(y)) * static_cast<size_t>(width) +
                        static_cast<size_t>(x);
                    cropped.values[idx] = v;
                    minValue = std::min(minValue, v);
                    maxValue = std::max(maxValue, v);
                }
            }
        }
    }
    cropped.meta.dataMin = std::isfinite(minValue) ? minValue : 0.0f;
    cropped.meta.dataMax = std::isfinite(maxValue) ? maxValue : 0.0f;
    return cropped;
}

} // namespace vbt
