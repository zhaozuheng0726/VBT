#include "vbt_file.h"

#include "../../src/field_profile.h"
#include "../../src/frame_metadata.h"
#include "../../src/render_temporal_decode.h"
#include "../../src/render_temporal_payload.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef VBT_USE_OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;
using namespace vbt;
using namespace vbt::render;

namespace {

struct SmokeRayPacked {
    float ox = 0.0f;
    float oy = 0.0f;
    float oz = 0.0f;
    float tMin = 0.0f;
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 1.0f;
    float tMax = 0.0f;
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct SmokeProbeSummary {
    uint32_t imageWidth = 0;
    uint32_t imageHeight = 0;
    uint32_t rayCount = 0;
    uint32_t stepCount = 0;
    int frame = 0;
};

struct Options {
    fs::path probeSummaryPath;
    fs::path rayBinPath;
    fs::path inputVbt;
    fs::path metadataPath;
    fs::path outputSummaryJson;
    int frameIndex = -1;
    int frameStart = -1;
    uint32_t frameCount = 1;
    float compareTolerance = 1.0e-3f;
};

struct PushConstants {
    uint32_t dimX = 0;
    uint32_t dimY = 0;
    uint32_t dimZ = 0;
    uint32_t frames = 0;
    uint32_t leafSize = 0;
    uint32_t leafCountX = 0;
    uint32_t leafCountY = 0;
    uint32_t imageWidth = 0;
    uint32_t imageHeight = 0;
    uint32_t rayCount = 0;
    uint32_t stepCount = 0;
    int32_t frameIndex = 0;
    float bboxMinX = 0.0f;
    float bboxMinY = 0.0f;
    float bboxMinZ = 0.0f;
};
static_assert(sizeof(PushConstants) <= 128, "Push constants too large");

void printUsage()
{
    std::cout
        << "Usage: vbt_smoke_vbt_bench --probe-summary <summary.json> --ray-bin <rays.bin>\n"
        << "                          --input-vbt <file.vbtp> --metadata <meta.json>\n"
        << "                          [--frame 100] [--frame-start 96] [--frame-count 6]\n"
        << "                          [--compare-tol 1e-3]\n"
        << "                          [--output-summary vbt_smoke_bench.json]\n";
}

bool parseArgs(int argc, char** argv, Options& opt)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--probe-summary" && i + 1 < argc) {
            opt.probeSummaryPath = argv[++i];
        } else if (arg == "--ray-bin" && i + 1 < argc) {
            opt.rayBinPath = argv[++i];
        } else if (arg == "--input-vbt" && i + 1 < argc) {
            opt.inputVbt = argv[++i];
        } else if (arg == "--metadata" && i + 1 < argc) {
            opt.metadataPath = argv[++i];
        } else if (arg == "--frame" && i + 1 < argc) {
            opt.frameIndex = std::stoi(argv[++i]);
        } else if (arg == "--frame-start" && i + 1 < argc) {
            opt.frameStart = std::stoi(argv[++i]);
        } else if (arg == "--frame-count" && i + 1 < argc) {
            opt.frameCount = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--compare-tol" && i + 1 < argc) {
            opt.compareTolerance = std::stof(argv[++i]);
        } else if (arg == "--output-summary" && i + 1 < argc) {
            opt.outputSummaryJson = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            printUsage();
            return false;
        }
    }
    if (opt.probeSummaryPath.empty() || opt.rayBinPath.empty() ||
        opt.inputVbt.empty() || opt.metadataPath.empty()) {
        printUsage();
        return false;
    }
    if (opt.frameCount == 0) {
        std::cerr << "frame-count must be > 0\n";
        return false;
    }
    return true;
}

SmokeProbeSummary loadProbeSummary(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open probe summary: " + path.string());
    }
    nlohmann::json document;
    try {
        in >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Failed to parse probe summary '" + path.string() + "': " + error.what());
    }
    auto readInt = [&](const char* key) {
        const auto it = document.find(key);
        if (it == document.end() || !it->is_number_integer()) {
            throw std::runtime_error(
                "Probe summary field must be an integer: " + std::string(key));
        }
        return it->get<int>();
    };
    SmokeProbeSummary summary;
    summary.imageWidth = static_cast<uint32_t>(readInt("image_width"));
    summary.imageHeight = static_cast<uint32_t>(readInt("image_height"));
    summary.rayCount = static_cast<uint32_t>(readInt("ray_count"));
    summary.stepCount = static_cast<uint32_t>(readInt("step_count"));
    summary.frame = readInt("frame");
    if (summary.imageWidth == 0 || summary.imageHeight == 0 ||
        summary.rayCount == 0 || summary.stepCount == 0) {
        throw std::runtime_error("Invalid probe summary: " + path.string());
    }
    return summary;
}

std::vector<SmokeRayPacked> loadRays(const fs::path& path, uint32_t expectedCount)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open ray bin: " + path.string());
    }
    const std::streamsize size = in.tellg();
    if (size < 0 || (size % static_cast<std::streamsize>(sizeof(SmokeRayPacked))) != 0) {
        throw std::runtime_error("Invalid ray bin size: " + path.string());
    }
    const size_t rayCount = static_cast<size_t>(size) / sizeof(SmokeRayPacked);
    if (expectedCount != 0 && rayCount != expectedCount) {
        throw std::runtime_error("Ray bin count mismatch with probe summary");
    }
    std::vector<SmokeRayPacked> rays(rayCount);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(rays.data()), size);
    return rays;
}

std::vector<float> controlCoords(int resolution)
{
    std::vector<float> coords(static_cast<size_t>(resolution), 0.0f);
    if (resolution <= 1) return coords;
    const float scale = 7.0f / static_cast<float>(resolution - 1);
    for (int i = 0; i < resolution; ++i) {
        coords[static_cast<size_t>(i)] = static_cast<float>(i) * scale;
    }
    return coords;
}

const std::vector<float>& coordsForResolution(int resolution)
{
    static std::unordered_map<int, std::vector<float>> cache;
    auto it = cache.find(resolution);
    if (it == cache.end()) {
        it = cache.emplace(resolution, controlCoords(resolution)).first;
    }
    return it->second;
}

std::array<float, 3> locateCoord(float p, int extent, int resolution)
{
    const auto& coords = coordsForResolution(resolution);
    const float maxP = static_cast<float>(std::max(0, extent - 1));
    p = std::clamp(p, 0.0f, maxP);
    int i1 = 1;
    while (i1 < resolution - 1 && coords[static_cast<size_t>(i1)] < p) ++i1;
    const int i0 = std::max(0, i1 - 1);
    const float c0 = coords[static_cast<size_t>(i0)];
    const float c1 = coords[static_cast<size_t>(i1)];
    const float w = (c1 > c0) ? ((p - c0) / (c1 - c0)) : 0.0f;
    return {static_cast<float>(i0), static_cast<float>(i1), w};
}

float sampleControlGrid(const std::vector<float>& grid,
                        int resolution,
                        int leafWidth,
                        int leafHeight,
                        int leafDepth,
                        float fx,
                        float fy,
                        float fz)
{
    const auto lx = locateCoord(fx, leafWidth, resolution);
    const auto ly = locateCoord(fy, leafHeight, resolution);
    const auto lz = locateCoord(fz, leafDepth, resolution);
    auto at = [&](int ix, int iy, int iz) {
        return grid[static_cast<size_t>((iz * resolution + iy) * resolution + ix)];
    };
    const int x0 = static_cast<int>(lx[0]);
    const int x1 = static_cast<int>(lx[1]);
    const int y0 = static_cast<int>(ly[0]);
    const int y1 = static_cast<int>(ly[1]);
    const int z0 = static_cast<int>(lz[0]);
    const int z1 = static_cast<int>(lz[1]);
    const float tx = lx[2];
    const float ty = ly[2];
    const float tz = lz[2];
    const float c00 = at(x0, y0, z0) * (1.0f - tx) + at(x1, y0, z0) * tx;
    const float c01 = at(x0, y0, z1) * (1.0f - tx) + at(x1, y0, z1) * tx;
    const float c10 = at(x0, y1, z0) * (1.0f - tx) + at(x1, y1, z0) * tx;
    const float c11 = at(x0, y1, z1) * (1.0f - tx) + at(x1, y1, z1) * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

bool shellOccupancyContains(const RenderTemporalShellOccupancySection& occupancy, uint16_t voxelIndex)
{
    if (voxelIndex >= 512u) return false;
    const uint32_t group = voxelIndex >> 6u;
    const uint32_t bit = voxelIndex & 63u;
    return ((occupancy.shellMask[group] >> bit) & 1ull) != 0ull;
}

void decodeLeafFrameValues(const uint8_t* leafBase,
                           size_t leafBytes,
                           const VbtFileHeader& fileHeader,
                           int frameIndex,
                           int leafWidth,
                           int leafHeight,
                           int leafDepth,
                           std::vector<float>& outLeaf)
{
    const auto leafView = parseRenderTemporalPackedLeaf(leafBase, leafBytes, true);
    if (leafView.header.mode == TemporalFirstPackedMode::EMPTY) {
        std::fill(outLeaf.begin(), outLeaf.end(), 0.0f);
        return;
    }

    const int coarseResolution = static_cast<int>(leafView.header.coarseResolution);
    const int fineResolution = static_cast<int>(leafView.header.fineResolution);
    const uint16_t coarseControlCount =
        static_cast<uint16_t>(coarseResolution * coarseResolution * coarseResolution);
    std::vector<float> coarseGrid(static_cast<size_t>(coarseControlCount), 0.0f);
    for (uint16_t i = 0; i < coarseControlCount; ++i) {
        coarseGrid[static_cast<size_t>(i)] =
            decodeRenderTemporalLeafControlValue(leafBase,
                                                 leafView.header,
                                                 leafView.prefix,
                                                 leafView.layout,
                                                 false,
                                                 true,
                                                 static_cast<int>(fileHeader.frames),
                                                 i,
                                                 frameIndex);
    }

    std::vector<float> fineGrid;
    const bool hasFine =
        fineResolution > 0 &&
        leafView.layout.fine.totalBytes > 0 &&
        leafView.header.mode != TemporalFirstPackedMode::TEMPORAL_GRID4;
    if (hasFine) {
        const uint16_t fineControlCount =
            static_cast<uint16_t>(fineResolution * fineResolution * fineResolution);
        fineGrid.resize(static_cast<size_t>(fineControlCount), 0.0f);
        for (uint16_t i = 0; i < fineControlCount; ++i) {
            fineGrid[static_cast<size_t>(i)] =
                decodeRenderTemporalLeafControlValue(leafBase,
                                                     leafView.header,
                                                     leafView.prefix,
                                                     leafView.layout,
                                                     true,
                                                     true,
                                                     static_cast<int>(fileHeader.frames),
                                                     i,
                                                     frameIndex);
        }
    }

    RenderTemporalShellOccupancySection shellOccupancy{};
    const bool hasShellOccupancy = leafView.header.mode == TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT;
    if (hasShellOccupancy) {
        shellOccupancy = unpackRenderTemporalShellOccupancySection(
            leafBase + leafView.layout.fine.descriptorOffset,
            leafView.layout.fine.descriptorBytes);
    }

    for (int lz = 0; lz < leafDepth; ++lz) {
        for (int ly = 0; ly < leafHeight; ++ly) {
            for (int lx = 0; lx < leafWidth; ++lx) {
                const uint16_t voxelIndex = renderTemporalLeafVoxelIndex(
                    static_cast<uint16_t>(lx),
                    static_cast<uint16_t>(ly),
                    static_cast<uint16_t>(lz),
                    static_cast<uint16_t>(fileHeader.leafSize));
                float value = sampleControlGrid(coarseGrid,
                                                coarseResolution,
                                                leafWidth,
                                                leafHeight,
                                                leafDepth,
                                                static_cast<float>(lx),
                                                static_cast<float>(ly),
                                                static_cast<float>(lz));
                if (!fineGrid.empty()) {
                    value += sampleControlGrid(fineGrid,
                                               fineResolution,
                                               leafWidth,
                                               leafHeight,
                                               leafDepth,
                                               static_cast<float>(lx),
                                               static_cast<float>(ly),
                                               static_cast<float>(lz));
                } else if (leafView.header.mode == TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT) {
                    value += decodeRenderTemporalLeafShellVoxelValue(leafBase,
                                                                     leafView,
                                                                     true,
                                                                     static_cast<int>(fileHeader.frames),
                                                                     voxelIndex,
                                                                     frameIndex);
                }
                if (hasShellOccupancy &&
                    !shellOccupancyContains(shellOccupancy, voxelIndex) &&
                    value < 0.0f) {
                    value = 0.0f;
                }
                const size_t localIdx =
                    (static_cast<size_t>(lz) * static_cast<size_t>(leafHeight) + static_cast<size_t>(ly)) *
                        static_cast<size_t>(leafWidth) +
                    static_cast<size_t>(lx);
                outLeaf[localIdx] = value;
            }
        }
    }
}

std::vector<float> decodeFrameToDense(const VbtFile& file, int frameIndex)
{
    const int leafSize = static_cast<int>(file.header.leafSize);
    const int leafCountX = (static_cast<int>(file.header.width) + leafSize - 1) / leafSize;
    const int leafCountY = (static_cast<int>(file.header.height) + leafSize - 1) / leafSize;
    const int leafCountZ = (static_cast<int>(file.header.depth) + leafSize - 1) / leafSize;
    const size_t voxelCount = static_cast<size_t>(file.header.width) *
                              static_cast<size_t>(file.header.height) *
                              static_cast<size_t>(file.header.depth);
    std::vector<float> frameValues(voxelCount, 0.0f);
    const uint8_t* payloadBase = reinterpret_cast<const uint8_t*>(file.payloadWords.data());

    for (int bz = 0; bz < leafCountZ; ++bz) {
        for (int by = 0; by < leafCountY; ++by) {
            for (int bx = 0; bx < leafCountX; ++bx) {
                const uint32_t leafIndex =
                    static_cast<uint32_t>((bz * leafCountY + by) * leafCountX + bx);
                const uint32_t wordBegin = file.offsetsWords[leafIndex];
                const uint32_t wordEnd = file.offsetsWords[leafIndex + 1u];
                const size_t leafBytes = static_cast<size_t>(wordEnd - wordBegin) * sizeof(uint32_t);
                const uint8_t* leafBase =
                    payloadBase + static_cast<size_t>(wordBegin) * sizeof(uint32_t);

                const int baseX = bx * leafSize;
                const int baseY = by * leafSize;
                const int baseZ = bz * leafSize;
                const int leafWidth =
                    std::min(leafSize, static_cast<int>(file.header.width) - baseX);
                const int leafHeight =
                    std::min(leafSize, static_cast<int>(file.header.height) - baseY);
                const int leafDepth =
                    std::min(leafSize, static_cast<int>(file.header.depth) - baseZ);

                std::vector<float> leafDecoded(static_cast<size_t>(leafWidth) *
                                                   static_cast<size_t>(leafHeight) *
                                                   static_cast<size_t>(leafDepth),
                                               0.0f);
                decodeLeafFrameValues(leafBase,
                                      leafBytes,
                                      file.header,
                                      frameIndex,
                                      leafWidth,
                                      leafHeight,
                                      leafDepth,
                                      leafDecoded);

                for (int lz = 0; lz < leafDepth; ++lz) {
                    for (int ly = 0; ly < leafHeight; ++ly) {
                        const size_t localRow =
                            (static_cast<size_t>(lz) * static_cast<size_t>(leafHeight) +
                             static_cast<size_t>(ly)) *
                            static_cast<size_t>(leafWidth);
                        const size_t globalRow =
                            (static_cast<size_t>(baseZ + lz) * static_cast<size_t>(file.header.height) +
                             static_cast<size_t>(baseY + ly)) *
                            static_cast<size_t>(file.header.width);
                        for (int lx = 0; lx < leafWidth; ++lx) {
                            frameValues[globalRow + static_cast<size_t>(baseX + lx)] =
                                leafDecoded[localRow + static_cast<size_t>(lx)];
                        }
                    }
                }
            }
        }
    }
    return frameValues;
}

float samplePackedLeafGridValue(const uint8_t* leafBase,
                                size_t leafBytes,
                                const RenderTemporalPackedLeafView& leafView,
                                bool useFineStream,
                                int resolution,
                                int frames,
                                int leafWidth,
                                int leafHeight,
                                int leafDepth,
                                int frameIndex,
                                float fx,
                                float fy,
                                float fz)
{
    const auto coords = controlCoords(resolution);
    auto locate = [&](float p, int extent) {
        const float maxP = static_cast<float>(std::max(0, extent - 1));
        p = std::clamp(p, 0.0f, maxP);
        int i1 = 1;
        while (i1 < resolution - 1 && coords[static_cast<size_t>(i1)] < p) ++i1;
        const int i0 = std::max(0, i1 - 1);
        const float c0 = coords[static_cast<size_t>(i0)];
        const float c1 = coords[static_cast<size_t>(i1)];
        const float w = (c1 > c0) ? ((p - c0) / (c1 - c0)) : 0.0f;
        return std::array<float, 3>{static_cast<float>(i0), static_cast<float>(i1), w};
    };
    const auto lx = locate(fx, leafWidth);
    const auto ly = locate(fy, leafHeight);
    const auto lz = locate(fz, leafDepth);
    auto at = [&](int ix, int iy, int iz) {
        const int idx = (iz * resolution + iy) * resolution + ix;
        return decodeRenderTemporalLeafControlValue(leafBase,
                                                    leafView.header,
                                                    leafView.prefix,
                                                    leafView.layout,
                                                    useFineStream,
                                                    true,
                                                    frames,
                                                    static_cast<uint16_t>(idx),
                                                    frameIndex);
    };
    const int x0 = static_cast<int>(lx[0]);
    const int x1 = static_cast<int>(lx[1]);
    const int y0 = static_cast<int>(ly[0]);
    const int y1 = static_cast<int>(ly[1]);
    const int z0 = static_cast<int>(lz[0]);
    const int z1 = static_cast<int>(lz[1]);
    const float tx = lx[2];
    const float ty = ly[2];
    const float tz = lz[2];
    const float c00 = at(x0, y0, z0) * (1.0f - tx) + at(x1, y0, z0) * tx;
    const float c01 = at(x0, y0, z1) * (1.0f - tx) + at(x1, y0, z1) * tx;
    const float c10 = at(x0, y1, z0) * (1.0f - tx) + at(x1, y1, z0) * tx;
    const float c11 = at(x0, y1, z1) * (1.0f - tx) + at(x1, y1, z1) * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

float samplePackedValueAtWorld(const VbtFile& file,
                               const FrameMetadata& meta,
                               int frameIndex,
                               float xWorld,
                               float yWorld,
                               float zWorld)
{
    const float x = xWorld - static_cast<float>(meta.bboxMin[0]);
    const float y = yWorld - static_cast<float>(meta.bboxMin[1]);
    const float z = zWorld - static_cast<float>(meta.bboxMin[2]);
    if (x < 0.0f || y < 0.0f || z < 0.0f ||
        x > static_cast<float>(meta.width - 1) ||
        y > static_cast<float>(meta.height - 1) ||
        z > static_cast<float>(meta.depth - 1)) {
        return 0.0f;
    }

    const int leafSize = static_cast<int>(file.header.leafSize);
    const int leafCountX = (static_cast<int>(file.header.width) + leafSize - 1) / leafSize;
    const int leafCountY = (static_cast<int>(file.header.height) + leafSize - 1) / leafSize;
    const int leafCountZ = (static_cast<int>(file.header.depth) + leafSize - 1) / leafSize;
    const int xi = std::clamp(static_cast<int>(std::floor(x)), 0, static_cast<int>(meta.width) - 1);
    const int yi = std::clamp(static_cast<int>(std::floor(y)), 0, static_cast<int>(meta.height) - 1);
    const int zi = std::clamp(static_cast<int>(std::floor(z)), 0, static_cast<int>(meta.depth) - 1);
    const int bx = std::min(leafCountX - 1, xi / leafSize);
    const int by = std::min(leafCountY - 1, yi / leafSize);
    const int bz = std::min(leafCountZ - 1, zi / leafSize);
    const uint32_t leafIndex =
        static_cast<uint32_t>((bz * leafCountY + by) * leafCountX + bx);

    const uint32_t wordBegin = file.offsetsWords[leafIndex];
    const uint32_t wordEnd = file.offsetsWords[leafIndex + 1u];
    if (wordEnd <= wordBegin) {
        return 0.0f;
    }

    const uint8_t* payloadBase = reinterpret_cast<const uint8_t*>(file.payloadWords.data());
    const uint8_t* leafBase = payloadBase + static_cast<size_t>(wordBegin) * sizeof(uint32_t);
    const size_t leafBytes = static_cast<size_t>(wordEnd - wordBegin) * sizeof(uint32_t);
    const auto leafView = parseRenderTemporalPackedLeaf(leafBase, leafBytes, true);
    if (leafView.header.mode == TemporalFirstPackedMode::EMPTY) {
        return 0.0f;
    }

    const int baseX = bx * leafSize;
    const int baseY = by * leafSize;
    const int baseZ = bz * leafSize;
    const int leafWidth = std::min(leafSize, static_cast<int>(file.header.width) - baseX);
    const int leafHeight = std::min(leafSize, static_cast<int>(file.header.height) - baseY);
    const int leafDepth = std::min(leafSize, static_cast<int>(file.header.depth) - baseZ);
    const float fx = x - static_cast<float>(baseX);
    const float fy = y - static_cast<float>(baseY);
    const float fz = z - static_cast<float>(baseZ);

    float value = samplePackedLeafGridValue(leafBase,
                                            leafBytes,
                                            leafView,
                                            false,
                                            static_cast<int>(leafView.header.coarseResolution),
                                            static_cast<int>(file.header.frames),
                                            leafWidth,
                                            leafHeight,
                                            leafDepth,
                                            frameIndex,
                                            fx,
                                            fy,
                                            fz);
    if (leafView.header.mode == TemporalFirstPackedMode::TEMPORAL_FINE6 &&
        leafView.header.fineResolution > 0u) {
        value += samplePackedLeafGridValue(leafBase,
                                           leafBytes,
                                           leafView,
                                           true,
                                           static_cast<int>(leafView.header.fineResolution),
                                           static_cast<int>(file.header.frames),
                                           leafWidth,
                                           leafHeight,
                                           leafDepth,
                                           frameIndex,
                                           fx,
                                           fy,
                                           fz);
    }
    return value;
}

inline size_t denseIndex(const FrameMetadata& meta, int x, int y, int z)
{
    return (static_cast<size_t>(z) * static_cast<size_t>(meta.height) + static_cast<size_t>(y)) *
               static_cast<size_t>(meta.width) +
           static_cast<size_t>(x);
}

float sampleDenseTrilinear(const std::vector<float>& dense,
                           const FrameMetadata& meta,
                           float xWorld,
                           float yWorld,
                           float zWorld)
{
    const float x = xWorld - static_cast<float>(meta.bboxMin[0]);
    const float y = yWorld - static_cast<float>(meta.bboxMin[1]);
    const float z = zWorld - static_cast<float>(meta.bboxMin[2]);
    if (x < 0.0f || y < 0.0f || z < 0.0f ||
        x > static_cast<float>(meta.width - 1) ||
        y > static_cast<float>(meta.height - 1) ||
        z > static_cast<float>(meta.depth - 1)) {
        return 0.0f;
    }

    const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, meta.width - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, meta.height - 1);
    const int z0 = std::clamp(static_cast<int>(std::floor(z)), 0, meta.depth - 1);
    const int x1 = std::min(x0 + 1, meta.width - 1);
    const int y1 = std::min(y0 + 1, meta.height - 1);
    const int z1 = std::min(z0 + 1, meta.depth - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float tz = z - static_cast<float>(z0);

    auto at = [&](int sx, int sy, int sz) -> float {
        return dense[denseIndex(meta, sx, sy, sz)];
    };

    const float c00 = at(x0, y0, z0) * (1.0f - tx) + at(x1, y0, z0) * tx;
    const float c10 = at(x0, y1, z0) * (1.0f - tx) + at(x1, y1, z0) * tx;
    const float c01 = at(x0, y0, z1) * (1.0f - tx) + at(x1, y0, z1) * tx;
    const float c11 = at(x0, y1, z1) * (1.0f - tx) + at(x1, y1, z1) * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

std::vector<float> runCpuReference(const std::vector<SmokeRayPacked>& rays,
                                   const VbtFile& file,
                                   const FrameMetadata& meta,
                                   int frameIndex,
                                   uint32_t stepCount)
{
    std::vector<float> out(rays.size(), 0.0f);
#ifdef VBT_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(rays.size()); ++i) {
        const SmokeRayPacked& ray = rays[static_cast<size_t>(i)];
        const float dt = ray.tMax / std::max(1u, stepCount);
        float t = ray.tMin + 0.5f * dt;
        float accum = 0.0f;
        for (uint32_t s = 0; s < stepCount; ++s) {
            const float x = ray.ox + ray.dx * t;
            const float y = ray.oy + ray.dy * t;
            const float z = ray.oz + ray.dz * t;
            accum += samplePackedValueAtWorld(file, meta, frameIndex, x, y, z);
            t += dt;
        }
        out[static_cast<size_t>(i)] = accum;
    }
    return out;
}

std::vector<char> readFileBytes(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open shader: " + path);
    }
    const size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> bytes(size);
    file.seekg(0);
    file.read(bytes.data(), static_cast<std::streamsize>(size));
    return bytes;
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeBits, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("No suitable Vulkan memory type found");
}

void createBuffer(VkPhysicalDevice physicalDevice,
                  VkDevice device,
                  VkDeviceSize size,
                  VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags properties,
                  Buffer& out)
{
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = std::max<VkDeviceSize>(size, 4);
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &out.buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan buffer");
    }

    VkMemoryRequirements memReq{};
    vkGetBufferMemoryRequirements(device, out.buffer, &memReq);
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memReq.memoryTypeBits, properties);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &out.memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate Vulkan memory");
    }
    if (vkBindBufferMemory(device, out.buffer, out.memory, 0) != VK_SUCCESS) {
        throw std::runtime_error("Failed to bind Vulkan buffer memory");
    }
    out.size = bufferInfo.size;
}

void destroyBuffer(VkDevice device, Buffer& buffer)
{
    if (device == VK_NULL_HANDLE) return;
    if (buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
    }
    if (buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }
    buffer.size = 0;
}

void copyBuffer(VkDevice device,
                VkQueue queue,
                VkCommandPool commandPool,
                VkBuffer src,
                VkBuffer dst,
                VkDeviceSize size)
{
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &allocInfo, &cmd) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate copy command buffer");
    }

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copy{};
    copy.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &copy);

    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = dst;
    barrier.offset = 0;
    barrier.size = size;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0,
                         nullptr,
                         1,
                         &barrier,
                         0,
                         nullptr);

    vkEndCommandBuffer(cmd);
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code)
{
    VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }
    return module;
}

std::string jsonEscape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += c; break;
        }
    }
    return out;
}

double averageMs(const std::vector<double>& xs)
{
    if (xs.empty()) return 0.0;
    return std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
}

double minMs(const std::vector<double>& xs)
{
    return xs.empty() ? 0.0 : *std::min_element(xs.begin(), xs.end());
}

double maxMs(const std::vector<double>& xs)
{
    return xs.empty() ? 0.0 : *std::max_element(xs.begin(), xs.end());
}

void writeSummary(const fs::path& path,
                  const Options& opt,
                  const SmokeProbeSummary& probe,
                  const FrameMetadata& meta,
                  const VbtFile& file,
                  size_t residentBytes,
                  double cpuDecodeMs,
                  double cpuReferenceMs,
                  double residentUploadMs,
                  const std::vector<double>& dispatchMs,
                  double readbackMs,
                  float minValue,
                  float maxValue,
                  float avgValue,
                  uint32_t mismatchCount,
                  float meanAbsDiff,
                  float maxAbsDiff)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open VBT smoke summary output: " + path.string());
    }

    const int frameStart = (opt.frameStart >= 0) ? opt.frameStart :
                           ((opt.frameIndex >= 0) ? opt.frameIndex : probe.frame);
    const double totalSamples = static_cast<double>(probe.rayCount) * static_cast<double>(probe.stepCount);
    const double dispatchAvgMs = averageMs(dispatchMs);
    const double samplesPerSec = (dispatchAvgMs > 0.0) ? (totalSamples / (dispatchAvgMs * 1.0e-3)) : 0.0;
    const double raysPerSec = (dispatchAvgMs > 0.0) ? (static_cast<double>(probe.rayCount) / (dispatchAvgMs * 1.0e-3)) : 0.0;

    out << "{\n";
    out << "  \"probe_summary\": \"" << jsonEscape(opt.probeSummaryPath.string()) << "\",\n";
    out << "  \"ray_bin\": \"" << jsonEscape(opt.rayBinPath.string()) << "\",\n";
    out << "  \"input_vbt\": \"" << jsonEscape(opt.inputVbt.string()) << "\",\n";
    out << "  \"metadata\": \"" << jsonEscape(opt.metadataPath.string()) << "\",\n";
    out << "  \"frame_start\": " << frameStart << ",\n";
    out << "  \"frame_count\": " << opt.frameCount << ",\n";
    out << "  \"image_width\": " << probe.imageWidth << ",\n";
    out << "  \"image_height\": " << probe.imageHeight << ",\n";
    out << "  \"ray_count\": " << probe.rayCount << ",\n";
    out << "  \"step_count\": " << probe.stepCount << ",\n";
    out << "  \"profile_type\": " << file.header.profileType << ",\n";
    out << "  \"leaf_size\": " << file.header.leafSize << ",\n";
    out << "  \"resident_bytes\": " << residentBytes << ",\n";
    out << "  \"resident_mb\": " << std::fixed << std::setprecision(3)
        << (static_cast<double>(residentBytes) / 1.0e6) << ",\n";
    out << "  \"cpu_decode_ms\": " << cpuDecodeMs << ",\n";
    out << "  \"cpu_reference_ms\": " << cpuReferenceMs << ",\n";
    out << "  \"resident_upload_ms\": " << residentUploadMs << ",\n";
    out << "  \"gpu_dispatch_ms\": " << dispatchAvgMs << ",\n";
    out << "  \"gpu_dispatch_avg_ms\": " << dispatchAvgMs << ",\n";
    out << "  \"gpu_dispatch_min_ms\": " << minMs(dispatchMs) << ",\n";
    out << "  \"gpu_dispatch_max_ms\": " << maxMs(dispatchMs) << ",\n";
    out << "  \"frame_advance_upload_ms\": 0.0,\n";
    out << "  \"readback_ms\": " << readbackMs << ",\n";
    out << "  \"samples_per_sec\": " << samplesPerSec << ",\n";
    out << "  \"rays_per_sec\": " << raysPerSec << ",\n";
    out << "  \"result_min\": " << minValue << ",\n";
    out << "  \"result_max\": " << maxValue << ",\n";
    out << "  \"result_avg\": " << avgValue << ",\n";
    out << "  \"compare_tolerance\": " << opt.compareTolerance << ",\n";
    out << "  \"mismatch_count\": " << mismatchCount << ",\n";
    out << "  \"mean_abs_diff\": " << meanAbsDiff << ",\n";
    out << "  \"max_abs_diff\": " << maxAbsDiff << "\n";
    out << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkQueryPool timestampPool = VK_NULL_HANDLE;
    Buffer rayBuffer{};
    Buffer offsetBuffer{};
    Buffer payloadBuffer{};
    Buffer resultBuffer{};
    Buffer stagingOffsetBuffer{};
    Buffer stagingPayloadBuffer{};
    Buffer stagingRayBuffer{};

    try {
        Options opt;
        if (!parseArgs(argc, argv, opt)) return 1;

        const SmokeProbeSummary probe = loadProbeSummary(opt.probeSummaryPath);
        const FrameMetadata meta = loadFrameMetadata(opt.metadataPath);
        const int frameStart = (opt.frameStart >= 0) ? opt.frameStart :
                               ((opt.frameIndex >= 0) ? opt.frameIndex : probe.frame);
        if (frameStart < 0 || frameStart >= meta.frames) {
            throw std::runtime_error("Frame start out of range");
        }
        if (frameStart + static_cast<int>(opt.frameCount) > meta.frames) {
            throw std::runtime_error("Frame sequence exceeds available frames");
        }

        VbtFile file;
        std::string error;
        if (!loadVbtFile(opt.inputVbt, file, error)) {
            throw std::runtime_error(error);
        }
        if (file.header.profileType != static_cast<uint32_t>(FieldType::DENSITY)) {
            throw std::runtime_error("Smoke VBT benchmark only supports render density .vbtp");
        }

        const std::vector<SmokeRayPacked> rays = loadRays(opt.rayBinPath, probe.rayCount);

        const auto cpuRefStart = std::chrono::high_resolution_clock::now();
        const std::vector<float> cpuReference =
            runCpuReference(rays, file, meta, frameStart, probe.stepCount);
        const auto cpuRefEnd = std::chrono::high_resolution_clock::now();
        const double cpuDecodeMs = 0.0;
        const double cpuReferenceMs =
            std::chrono::duration<double, std::milli>(cpuRefEnd - cpuRefStart).count();

        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.pApplicationName = "Smoke VBT Bench";
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.pEngineName = "None";
        appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instanceInfo.pApplicationInfo = &appInfo;
        if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance");
        }

        uint32_t physicalCount = 0;
        vkEnumeratePhysicalDevices(instance, &physicalCount, nullptr);
        if (physicalCount == 0) {
            throw std::runtime_error("No Vulkan physical device available");
        }
        std::vector<VkPhysicalDevice> physicalDevices(physicalCount);
        vkEnumeratePhysicalDevices(instance, &physicalCount, physicalDevices.data());

        uint32_t queueFamilyIndex = std::numeric_limits<uint32_t>::max();
        VkPhysicalDeviceLimits limits{};
        for (VkPhysicalDevice candidate : physicalDevices) {
            uint32_t queueCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueCount, nullptr);
            std::vector<VkQueueFamilyProperties> queueProps(queueCount);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueCount, queueProps.data());
            for (uint32_t i = 0; i < queueCount; ++i) {
                if ((queueProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    physicalDevice = candidate;
                    queueFamilyIndex = i;
                    VkPhysicalDeviceProperties properties{};
                    vkGetPhysicalDeviceProperties(candidate, &properties);
                    limits = properties.limits;
                    break;
                }
            }
            if (physicalDevice != VK_NULL_HANDLE) break;
        }
        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("No Vulkan compute queue family found");
        }

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueInfo.queueFamilyIndex = queueFamilyIndex;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan device");
        }
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool");
        }

        VkCommandBufferAllocateInfo cmdAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cmdAlloc.commandPool = commandPool;
        cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAlloc.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &cmdAlloc, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffer");
        }

        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create fence");
        }

        const auto uploadStart = std::chrono::high_resolution_clock::now();
        const VkDeviceSize rayBytes = static_cast<VkDeviceSize>(rays.size() * sizeof(SmokeRayPacked));
        const VkDeviceSize offsetBytes = static_cast<VkDeviceSize>(file.offsetsWords.size() * sizeof(uint32_t));
        const VkDeviceSize payloadBytes = static_cast<VkDeviceSize>(file.payloadWords.size() * sizeof(uint32_t));
        const VkDeviceSize resultBytes = static_cast<VkDeviceSize>(rays.size() * sizeof(float));

        createBuffer(physicalDevice, device,
                     rayBytes,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     rayBuffer);
        createBuffer(physicalDevice, device,
                     offsetBytes,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     offsetBuffer);
        createBuffer(physicalDevice, device,
                     payloadBytes,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     payloadBuffer);
        createBuffer(physicalDevice, device,
                     resultBytes,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     resultBuffer);

        createBuffer(physicalDevice, device,
                     rayBytes,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingRayBuffer);
        createBuffer(physicalDevice, device,
                     offsetBytes,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingOffsetBuffer);
        createBuffer(physicalDevice, device,
                     payloadBytes,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingPayloadBuffer);

        void* mapped = nullptr;
        vkMapMemory(device, stagingRayBuffer.memory, 0, stagingRayBuffer.size, 0, &mapped);
        std::memcpy(mapped, rays.data(), static_cast<size_t>(rayBytes));
        vkUnmapMemory(device, stagingRayBuffer.memory);

        vkMapMemory(device, stagingOffsetBuffer.memory, 0, stagingOffsetBuffer.size, 0, &mapped);
        std::memcpy(mapped, file.offsetsWords.data(), static_cast<size_t>(offsetBytes));
        vkUnmapMemory(device, stagingOffsetBuffer.memory);

        vkMapMemory(device, stagingPayloadBuffer.memory, 0, stagingPayloadBuffer.size, 0, &mapped);
        std::memcpy(mapped, file.payloadWords.data(), static_cast<size_t>(payloadBytes));
        vkUnmapMemory(device, stagingPayloadBuffer.memory);

        copyBuffer(device, queue, commandPool, stagingRayBuffer.buffer, rayBuffer.buffer, rayBytes);
        copyBuffer(device, queue, commandPool, stagingOffsetBuffer.buffer, offsetBuffer.buffer, offsetBytes);
        copyBuffer(device, queue, commandPool, stagingPayloadBuffer.buffer, payloadBuffer.buffer, payloadBytes);

        VkDescriptorSetLayoutBinding bindings[4]{};
        for (uint32_t i = 0; i < 4; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = 4;
        layoutInfo.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout");
        }

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout");
        }

        const auto shaderCode = readFileBytes(VBT_SMOKE_SHADER_SPV_PATH);
        shaderModule = createShaderModule(device, shaderCode);

        VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shaderModule;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = pipelineLayout;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute pipeline");
        }

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 4;
        VkDescriptorPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolCreateInfo.maxSets = 1;
        poolCreateInfo.poolSizeCount = 1;
        poolCreateInfo.pPoolSizes = &poolSize;
        if (vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool");
        }

        VkDescriptorSetAllocateInfo setAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        setAlloc.descriptorPool = descriptorPool;
        setAlloc.descriptorSetCount = 1;
        setAlloc.pSetLayouts = &descriptorSetLayout;
        if (vkAllocateDescriptorSets(device, &setAlloc, &descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor set");
        }

        VkDescriptorBufferInfo rayInfo{rayBuffer.buffer, 0, rayBuffer.size};
        VkDescriptorBufferInfo offsetInfo{offsetBuffer.buffer, 0, offsetBuffer.size};
        VkDescriptorBufferInfo payloadInfo{payloadBuffer.buffer, 0, payloadBuffer.size};
        VkDescriptorBufferInfo resultInfo{resultBuffer.buffer, 0, resultBuffer.size};
        const VkDescriptorBufferInfo infos[4] = {rayInfo, offsetInfo, payloadInfo, resultInfo};
        std::array<VkWriteDescriptorSet, 4> writes{};
        for (uint32_t i = 0; i < 4; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptorSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkQueryPoolCreateInfo queryPoolInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryPoolInfo.queryCount = 2;
        if (vkCreateQueryPool(device, &queryPoolInfo, nullptr, &timestampPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create timestamp query pool");
        }

        const auto uploadEnd = std::chrono::high_resolution_clock::now();
        const double residentUploadMs =
            std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();

        PushConstants push{};
        push.dimX = file.header.width;
        push.dimY = file.header.height;
        push.dimZ = file.header.depth;
        push.frames = file.header.frames;
        push.leafSize = file.header.leafSize;
        push.leafCountX = (file.header.width + file.header.leafSize - 1u) / file.header.leafSize;
        push.leafCountY = (file.header.height + file.header.leafSize - 1u) / file.header.leafSize;
        push.imageWidth = probe.imageWidth;
        push.imageHeight = probe.imageHeight;
        push.rayCount = probe.rayCount;
        push.stepCount = probe.stepCount;
        push.frameIndex = frameStart;
        push.bboxMinX = static_cast<float>(meta.bboxMin[0]);
        push.bboxMinY = static_cast<float>(meta.bboxMin[1]);
        push.bboxMinZ = static_cast<float>(meta.bboxMin[2]);

        const uint32_t groupX = (probe.imageWidth + 7u) / 8u;
        const uint32_t groupY = (probe.imageHeight + 7u) / 8u;
        std::vector<double> dispatchTimesMs;
        dispatchTimesMs.reserve(opt.frameCount);
        double readbackMs = 0.0;
        std::vector<float> gpuResults(rays.size(), 0.0f);

        for (uint32_t seq = 0; seq < opt.frameCount; ++seq) {
            push.frameIndex = frameStart + static_cast<int>(seq);
            vkResetCommandBuffer(commandBuffer, 0);

            VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(commandBuffer, &beginInfo);
            vkCmdResetQueryPool(commandBuffer, timestampPool, 0, 2);
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampPool, 0);
            vkCmdBindDescriptorSets(commandBuffer,
                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipelineLayout,
                                    0,
                                    1,
                                    &descriptorSet,
                                    0,
                                    nullptr);
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdPushConstants(commandBuffer,
                               pipelineLayout,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0,
                               sizeof(PushConstants),
                               &push);
            vkCmdDispatch(commandBuffer, groupX, groupY, 1);

            VkBufferMemoryBarrier resultBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            resultBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            resultBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            resultBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            resultBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            resultBarrier.buffer = resultBuffer.buffer;
            resultBarrier.offset = 0;
            resultBarrier.size = resultBuffer.size;
            vkCmdPipelineBarrier(commandBuffer,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 1,
                                 &resultBarrier,
                                 0,
                                 nullptr);
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampPool, 1);
            vkEndCommandBuffer(commandBuffer);

            VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;
            vkResetFences(device, 1, &fence);
            vkQueueSubmit(queue, 1, &submitInfo, fence);
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

            uint64_t timestamps[2]{};
            double dispatchMs = 0.0;
            if (vkGetQueryPoolResults(device,
                                      timestampPool,
                                      0,
                                      2,
                                      sizeof(timestamps),
                                      timestamps,
                                      sizeof(uint64_t),
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
                const double periodNs = static_cast<double>(limits.timestampPeriod);
                dispatchMs = (static_cast<double>(timestamps[1] - timestamps[0]) * periodNs) * 1.0e-6;
            }
            dispatchTimesMs.push_back(dispatchMs);

            if (seq == 0u) {
                const auto readbackStart = std::chrono::high_resolution_clock::now();
                void* mappedResults = nullptr;
                vkMapMemory(device, resultBuffer.memory, 0, resultBuffer.size, 0, &mappedResults);
                std::memcpy(gpuResults.data(), mappedResults, static_cast<size_t>(resultBytes));
                vkUnmapMemory(device, resultBuffer.memory);
                const auto readbackEnd = std::chrono::high_resolution_clock::now();
                readbackMs =
                    std::chrono::duration<double, std::milli>(readbackEnd - readbackStart).count();
            }
        }

        uint32_t mismatchCount = 0;
        double diffSum = 0.0;
        float maxAbsDiff = 0.0f;
        float minValue = std::numeric_limits<float>::max();
        float maxValue = std::numeric_limits<float>::lowest();
        double avgAccum = 0.0;
        for (size_t i = 0; i < gpuResults.size(); ++i) {
            const float value = gpuResults[i];
            const float diff = std::abs(value - cpuReference[i]);
            diffSum += static_cast<double>(diff);
            maxAbsDiff = std::max(maxAbsDiff, diff);
            if (diff > opt.compareTolerance) {
                ++mismatchCount;
            }
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
            avgAccum += static_cast<double>(value);
        }
        const float meanAbsDiff = gpuResults.empty()
            ? 0.0f
            : static_cast<float>(diffSum / static_cast<double>(gpuResults.size()));
        const float avgValue = gpuResults.empty()
            ? 0.0f
            : static_cast<float>(avgAccum / static_cast<double>(gpuResults.size()));

        const size_t residentBytes =
            static_cast<size_t>(offsetBytes) + static_cast<size_t>(payloadBytes);

        std::cout << "VBT smoke GPU benchmark\n";
        std::cout << "  frameStart: " << frameStart << "\n";
        std::cout << "  frameCount: " << opt.frameCount << "\n";
        std::cout << "  rays: " << probe.rayCount << "\n";
        std::cout << "  image: " << probe.imageWidth << " x " << probe.imageHeight << "\n";
        std::cout << "  stepCount: " << probe.stepCount << "\n";
        std::cout << "  residentMB: " << std::fixed << std::setprecision(3)
                  << (static_cast<double>(residentBytes) / 1.0e6) << "\n";
        std::cout << "  cpuDecodeMs: " << cpuDecodeMs << "\n";
        std::cout << "  cpuReferenceMs: " << cpuReferenceMs << "\n";
        std::cout << "  residentUploadMs: " << residentUploadMs << "\n";
        std::cout << "  gpuDispatchAvgMs: " << averageMs(dispatchTimesMs) << "\n";
        std::cout << "  readbackMs: " << readbackMs << "\n";
        std::cout << "  meanAbsDiff: " << meanAbsDiff << "\n";
        std::cout << "  maxAbsDiff: " << maxAbsDiff << "\n";
        std::cout << "  mismatchCount(>" << opt.compareTolerance << "): " << mismatchCount << "\n";

        if (!opt.outputSummaryJson.empty()) {
            writeSummary(opt.outputSummaryJson,
                         opt,
                         probe,
                         meta,
                         file,
                         residentBytes,
                         cpuDecodeMs,
                         cpuReferenceMs,
                         residentUploadMs,
                         dispatchTimesMs,
                         readbackMs,
                         minValue,
                         maxValue,
                         avgValue,
                         mismatchCount,
                         meanAbsDiff,
                         maxAbsDiff);
            std::cout << "  wrote summary: " << opt.outputSummaryJson.string() << "\n";
        }

        vkDeviceWaitIdle(device);
        destroyBuffer(device, stagingRayBuffer);
        destroyBuffer(device, stagingOffsetBuffer);
        destroyBuffer(device, stagingPayloadBuffer);
        destroyBuffer(device, rayBuffer);
        destroyBuffer(device, offsetBuffer);
        destroyBuffer(device, payloadBuffer);
        destroyBuffer(device, resultBuffer);
        if (timestampPool != VK_NULL_HANDLE) vkDestroyQueryPool(device, timestampPool, nullptr);
        if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
        if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        if (shaderModule != VK_NULL_HANDLE) vkDestroyShaderModule(device, shaderModule, nullptr);
        if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
        if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, nullptr);
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "VBT smoke benchmark failed: " << ex.what() << "\n";
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
        }
        destroyBuffer(device, stagingRayBuffer);
        destroyBuffer(device, stagingOffsetBuffer);
        destroyBuffer(device, stagingPayloadBuffer);
        destroyBuffer(device, rayBuffer);
        destroyBuffer(device, offsetBuffer);
        destroyBuffer(device, payloadBuffer);
        destroyBuffer(device, resultBuffer);
        if (timestampPool != VK_NULL_HANDLE) vkDestroyQueryPool(device, timestampPool, nullptr);
        if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
        if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        if (shaderModule != VK_NULL_HANDLE) vkDestroyShaderModule(device, shaderModule, nullptr);
        if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
        if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, nullptr);
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
        return 10;
    }
}
