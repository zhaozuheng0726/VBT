#pragma once

#include "frame_metadata.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <vector>

namespace vbt {

struct RawVolume4D {
    FrameMetadata meta;
    std::vector<float> values;
    std::shared_ptr<const float> mappedValues;
    size_t mappedValueCount = 0;

    size_t frameVoxelCount() const;
    size_t totalVoxelCount() const;
    float at(int x, int y, int z, int t) const;
};

std::filesystem::path guessMetadataPathForRaw(const std::filesystem::path& rawPath);
RawVolume4D loadRawVolume(const std::filesystem::path& rawPath,
                          const std::filesystem::path& metadataPath);
RawVolume4D loadRawVolumeMapped(const std::filesystem::path& rawPath,
                                const std::filesystem::path& metadataPath);
RawVolume4D cropRawVolume(const RawVolume4D& source,
                          int x0, int y0, int z0,
                          int width, int height, int depth);

} // namespace vbt
