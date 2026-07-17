#include "raw_volume.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace vbt {

namespace {

size_t checkedMultiply(size_t lhs, size_t rhs, const char* label)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::runtime_error(std::string("Raw volume size overflow while computing ") + label);
    }
    return lhs * rhs;
}

void readExact(std::ifstream& in, void* destination, size_t bytes, const std::filesystem::path& path)
{
    if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Raw read is too large for std::streamsize: " + path.string());
    }
    in.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    if (!in || static_cast<size_t>(in.gcount()) != bytes) {
        throw std::runtime_error("Failed to read exact raw data from: " + path.string());
    }
}

} // namespace

size_t RawVolume4D::frameVoxelCount() const
{
    const size_t xy = checkedMultiply(static_cast<size_t>(meta.width),
                                      static_cast<size_t>(meta.height),
                                      "frame width*height");
    return checkedMultiply(xy, static_cast<size_t>(meta.depth), "frame voxel count");
}

size_t RawVolume4D::totalVoxelCount() const
{
    return checkedMultiply(frameVoxelCount(), static_cast<size_t>(meta.frames), "total voxel count");
}

float RawVolume4D::at(int x, int y, int z, int t) const
{
    const size_t frameStride = frameVoxelCount();
    const size_t spatial =
        (static_cast<size_t>(z) * static_cast<size_t>(meta.height) + static_cast<size_t>(y)) *
            static_cast<size_t>(meta.width) +
        static_cast<size_t>(x);
    if (mappedValues) {
        const size_t idx = meta.rawStorageOrder == RawStorageOrder::TimeFastest
            ? spatial * static_cast<size_t>(meta.frames) + static_cast<size_t>(t)
            : static_cast<size_t>(t) * frameStride + spatial;
        if (idx >= mappedValueCount) {
            throw std::runtime_error("Mapped RAW index out of range");
        }
        return mappedValues.get()[idx];
    }
    return values[static_cast<size_t>(t) * frameStride + spatial];
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
    if (volume.meta.dataType != "FLOAT32" && volume.meta.dataType != "FLOAT") {
        throw std::runtime_error("Unsupported raw data_type '" + volume.meta.dataType +
                                 "'; VBT currently requires float32: " + metadataPath.string());
    }

    std::ifstream in(rawPath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open raw file: " + rawPath.string());
    }

    const size_t frameStride = volume.frameVoxelCount();
    const size_t totalVoxels = volume.totalVoxelCount();
    const size_t expectedBytes = checkedMultiply(totalVoxels, sizeof(float), "raw byte count");
    std::error_code fileSizeError;
    const uintmax_t actualBytes = std::filesystem::file_size(rawPath, fileSizeError);
    if (fileSizeError) {
        throw std::runtime_error("Failed to query raw file size: " + rawPath.string());
    }
    if (actualBytes != expectedBytes) {
        throw std::runtime_error("Raw file size mismatch for '" + rawPath.string() +
                                 "': expected " + std::to_string(expectedBytes) +
                                 " bytes, got " + std::to_string(actualBytes));
    }

    volume.values.resize(totalVoxels);
    if (volume.meta.rawStorageOrder == RawStorageOrder::FrameMajor) {
        readExact(in, volume.values.data(), expectedBytes, rawPath);
        return volume;
    }

    // T-fastest source layout stores all time samples for one spatial voxel
    // contiguously. Convert it once into the canonical internal [T][Z][Y][X]
    // layout used by all encoders and render probes.
    constexpr size_t kTransposeBufferBytes = 8u * 1024u * 1024u;
    const size_t frames = static_cast<size_t>(volume.meta.frames);
    const size_t spatialChunk =
        std::max<size_t>(1, kTransposeBufferBytes / checkedMultiply(frames, sizeof(float), "transpose row"));
    std::vector<float> sourceChunk;
    for (size_t spatialBase = 0; spatialBase < frameStride; spatialBase += spatialChunk) {
        const size_t spatialCount = std::min(spatialChunk, frameStride - spatialBase);
        const size_t chunkValues = checkedMultiply(spatialCount, frames, "transpose chunk values");
        sourceChunk.resize(chunkValues);
        readExact(in,
                  sourceChunk.data(),
                  checkedMultiply(chunkValues, sizeof(float), "transpose chunk bytes"),
                  rawPath);

#ifdef VBT_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long t = 0; t < static_cast<long long>(frames); ++t) {
            const size_t outputBase = static_cast<size_t>(t) * frameStride + spatialBase;
            for (size_t localSpatial = 0; localSpatial < spatialCount; ++localSpatial) {
                volume.values[outputBase + localSpatial] =
                    sourceChunk[localSpatial * frames + static_cast<size_t>(t)];
            }
        }
    }
    return volume;
}

RawVolume4D loadRawVolumeMapped(const std::filesystem::path& rawPath,
                                const std::filesystem::path& metadataPath)
{
    RawVolume4D volume;
    volume.meta = loadFrameMetadata(metadataPath);
    if (volume.meta.dataType != "FLOAT32" && volume.meta.dataType != "FLOAT") {
        throw std::runtime_error("Unsupported raw data_type '" + volume.meta.dataType +
                                 "'; VBT currently requires float32: " + metadataPath.string());
    }

    const size_t totalVoxels = volume.totalVoxelCount();
    const size_t expectedBytes = checkedMultiply(totalVoxels, sizeof(float), "mapped raw byte count");
    std::error_code fileSizeError;
    const uintmax_t actualBytes = std::filesystem::file_size(rawPath, fileSizeError);
    if (fileSizeError) {
        throw std::runtime_error("Failed to query raw file size: " + rawPath.string());
    }
    if (actualBytes != expectedBytes) {
        throw std::runtime_error("Raw file size mismatch for '" + rawPath.string() +
                                 "': expected " + std::to_string(expectedBytes) +
                                 " bytes, got " + std::to_string(actualBytes));
    }

#ifdef _WIN32
    HANDLE file = CreateFileW(rawPath.c_str(),
                              GENERIC_READ,
                              FILE_SHARE_READ,
                              nullptr,
                              OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
                              nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open mapped raw file: " + rawPath.string());
    }
    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        CloseHandle(file);
        throw std::runtime_error("Failed to create raw file mapping: " + rawPath.string());
    }
    const auto* data = static_cast<const float*>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
    if (data == nullptr) {
        CloseHandle(mapping);
        CloseHandle(file);
        throw std::runtime_error("Failed to map raw file: " + rawPath.string());
    }
    volume.mappedValues = std::shared_ptr<const float>(data, [mapping, file](const float* pointer) {
        UnmapViewOfFile(pointer);
        CloseHandle(mapping);
        CloseHandle(file);
    });
#else
    const int fd = ::open(rawPath.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Failed to open mapped raw file: " + rawPath.string());
    }
    void* address = mmap(nullptr, expectedBytes, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (address == MAP_FAILED) {
        throw std::runtime_error("Failed to map raw file: " + rawPath.string());
    }
    const auto* data = static_cast<const float*>(address);
    volume.mappedValues = std::shared_ptr<const float>(data, [expectedBytes](const float* pointer) {
        munmap(const_cast<float*>(pointer), expectedBytes);
    });
#endif
    volume.mappedValueCount = totalVoxels;
    return volume;
}

RawVolume4D cropRawVolume(const RawVolume4D& source,
                          int x0, int y0, int z0,
                          int width, int height, int depth)
{
    if (width <= 0 || height <= 0 || depth <= 0) {
        throw std::runtime_error("Invalid crop size");
    }
    if (x0 < 0 || y0 < 0 || z0 < 0 ||
        x0 + width > source.meta.width ||
        y0 + height > source.meta.height ||
        z0 + depth > source.meta.depth) {
        throw std::runtime_error("Crop bounds exceed source volume");
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
