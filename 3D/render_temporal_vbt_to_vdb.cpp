#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "../render/src/vbt_file.h"
#include "../src/field_profile.h"
#include "../src/render_temporal_decode.h"
#include "../src/render_temporal_payload.h"
#include "vdb_tools/frame_metadata.h"

namespace fs = std::filesystem;
using namespace vbt;
using namespace vbt::render;

namespace {

struct Options {
    fs::path inputVbt;
    fs::path metadataPath;
    fs::path outputVdb;
    int frameIndex = 0;
    float background = 0.0f;
    float sparseThreshold = 0.0f;
    std::string gridNameOverride;
    float clampMin = std::numeric_limits<float>::quiet_NaN();
    float clampMax = std::numeric_limits<float>::quiet_NaN();
    float shellEmptyScale = 1.0f;
    float shellEmptyMax = std::numeric_limits<float>::quiet_NaN();
};

void printUsage()
{
    std::cout
        << "Usage:\n"
        << "  render_temporal_vbt_to_vdb.exe --input-vbt <file.vbtp> --metadata <meta.json> --output-vdb <out.vdb> --frame <index>\n"
        << "                                [--background 0] [--sparse-threshold 0] [--grid-name density]\n"
        << "                                [--clamp-min <value>] [--clamp-max <value>]\n"
        << "                                [--shell-empty-scale 0.0] [--shell-empty-max 0.2]\n";
}

bool parseArgs(int argc, char** argv, Options& opt)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--input-vbt" && i + 1 < argc) {
            opt.inputVbt = argv[++i];
        } else if (arg == "--metadata" && i + 1 < argc) {
            opt.metadataPath = argv[++i];
        } else if (arg == "--output-vdb" && i + 1 < argc) {
            opt.outputVdb = argv[++i];
        } else if (arg == "--frame" && i + 1 < argc) {
            opt.frameIndex = std::stoi(argv[++i]);
        } else if (arg == "--background" && i + 1 < argc) {
            opt.background = std::stof(argv[++i]);
        } else if (arg == "--sparse-threshold" && i + 1 < argc) {
            opt.sparseThreshold = std::stof(argv[++i]);
        } else if (arg == "--grid-name" && i + 1 < argc) {
            opt.gridNameOverride = argv[++i];
        } else if (arg == "--clamp-min" && i + 1 < argc) {
            opt.clampMin = std::stof(argv[++i]);
        } else if (arg == "--clamp-max" && i + 1 < argc) {
            opt.clampMax = std::stof(argv[++i]);
        } else if (arg == "--shell-empty-scale" && i + 1 < argc) {
            opt.shellEmptyScale = std::stof(argv[++i]);
        } else if (arg == "--shell-empty-max" && i + 1 < argc) {
            opt.shellEmptyMax = std::stof(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            printUsage();
            return false;
        }
    }
    if (opt.inputVbt.empty() || opt.metadataPath.empty() || opt.outputVdb.empty()) {
        printUsage();
        return false;
    }
    if (!std::isnan(opt.clampMin) && !std::isnan(opt.clampMax) && opt.clampMin > opt.clampMax) {
        std::cerr << "--clamp-min must be <= --clamp-max\n";
        return false;
    }
    if (opt.shellEmptyScale < 0.0f || opt.shellEmptyScale > 1.0f) {
        std::cerr << "--shell-empty-scale must be within [0, 1]\n";
        return false;
    }
    return true;
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
                           const Options& options,
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
    const uint16_t coarseControlCount = static_cast<uint16_t>(coarseResolution * coarseResolution * coarseResolution);
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
        const uint16_t fineControlCount = static_cast<uint16_t>(fineResolution * fineResolution * fineResolution);
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
                    !std::isnan(options.shellEmptyMax) &&
                    options.shellEmptyScale < 1.0f &&
                    !shellOccupancyContains(shellOccupancy, voxelIndex) &&
                    value > 0.0f &&
                    value <= options.shellEmptyMax) {
                    value *= options.shellEmptyScale;
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

} // namespace

int main(int argc, char** argv)
{
    Options opt;
    if (!parseArgs(argc, argv, opt)) return 1;

    const vdbtools::FrameMetadata meta = vdbtools::loadFrameMetadata(opt.metadataPath);
    const bool useClampMin = !std::isnan(opt.clampMin);
    const bool useClampMax = !std::isnan(opt.clampMax);
    if (opt.frameIndex < 0 || opt.frameIndex >= meta.frames) {
        std::cerr << "Frame index out of range: " << opt.frameIndex << " / " << meta.frames << "\n";
        return 2;
    }

    VbtFile file;
    std::string error;
    if (!loadVbtFile(opt.inputVbt, file, error)) {
        std::cerr << "Failed to load VBT file: " << error << "\n";
        return 3;
    }
    if (file.header.profileType != static_cast<uint32_t>(FieldType::DENSITY)) {
        std::cerr << "render_temporal_vbt_to_vdb only supports render/density payloads\n";
        return 4;
    }
    if (file.header.width != static_cast<uint32_t>(meta.width) ||
        file.header.height != static_cast<uint32_t>(meta.height) ||
        file.header.depth != static_cast<uint32_t>(meta.depth) ||
        file.header.frames != static_cast<uint32_t>(meta.frames)) {
        std::cerr << "Dimension mismatch between VBTPACK4 and metadata\n";
        return 5;
    }

    const int leafSize = static_cast<int>(file.header.leafSize);
    const int leafCountX = (static_cast<int>(file.header.width) + leafSize - 1) / leafSize;
    const int leafCountY = (static_cast<int>(file.header.height) + leafSize - 1) / leafSize;
    const int leafCountZ = (static_cast<int>(file.header.depth) + leafSize - 1) / leafSize;
    const size_t voxelCount = static_cast<size_t>(file.header.width) *
                              static_cast<size_t>(file.header.height) *
                              static_cast<size_t>(file.header.depth);
    std::vector<float> frameValues(voxelCount, opt.background);
    const uint8_t* payloadBase = reinterpret_cast<const uint8_t*>(file.payloadWords.data());

    for (int bz = 0; bz < leafCountZ; ++bz) {
        for (int by = 0; by < leafCountY; ++by) {
            for (int bx = 0; bx < leafCountX; ++bx) {
                const uint32_t leafIndex =
                    static_cast<uint32_t>((bz * leafCountY + by) * leafCountX + bx);
                const uint32_t wordBegin = file.offsetsWords[leafIndex];
                const uint32_t wordEnd = file.offsetsWords[leafIndex + 1u];
                const size_t leafBytes = static_cast<size_t>(wordEnd - wordBegin) * sizeof(uint32_t);
                const uint8_t* leafBase = payloadBase + static_cast<size_t>(wordBegin) * sizeof(uint32_t);

                const int baseX = bx * leafSize;
                const int baseY = by * leafSize;
                const int baseZ = bz * leafSize;
                const int leafWidth = std::min(leafSize, static_cast<int>(file.header.width) - baseX);
                const int leafHeight = std::min(leafSize, static_cast<int>(file.header.height) - baseY);
                const int leafDepth = std::min(leafSize, static_cast<int>(file.header.depth) - baseZ);

                std::vector<float> leafDecoded(static_cast<size_t>(leafWidth) *
                                               static_cast<size_t>(leafHeight) *
                                               static_cast<size_t>(leafDepth),
                                               0.0f);
                decodeLeafFrameValues(leafBase,
                                      leafBytes,
                                      file.header,
                                      opt,
                                      opt.frameIndex,
                                      leafWidth,
                                      leafHeight,
                                      leafDepth,
                                      leafDecoded);

                for (int lz = 0; lz < leafDepth; ++lz) {
                    for (int ly = 0; ly < leafHeight; ++ly) {
                        const size_t localRow =
                            (static_cast<size_t>(lz) * static_cast<size_t>(leafHeight) + static_cast<size_t>(ly)) *
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

    openvdb::initialize();
    auto grid = openvdb::FloatGrid::create(opt.background);
    grid->setGridClass(meta.conversionMode == "levelset" ? openvdb::GRID_LEVEL_SET : openvdb::GRID_FOG_VOLUME);
    grid->setName(opt.gridNameOverride.empty() ? meta.gridName : opt.gridNameOverride);
    if (meta.indexToWorldDeclared) {
        openvdb::math::Mat4d matrix;
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                matrix(row, column) = meta.indexToWorld[static_cast<size_t>(row * 4 + column)];
            }
        }
        grid->setTransform(openvdb::math::Transform::createLinearTransform(matrix));
    } else {
        grid->setTransform(openvdb::math::Transform::createLinearTransform(meta.voxelSize));
    }
    auto accessor = grid->getAccessor();
    const int x0 = meta.bboxMin[0];
    const int y0 = meta.bboxMin[1];
    const int z0 = meta.bboxMin[2];
    for (int z = 0; z < file.header.depth; ++z) {
        for (int y = 0; y < file.header.height; ++y) {
            const size_t zy =
                (static_cast<size_t>(z) * static_cast<size_t>(file.header.height) + static_cast<size_t>(y)) *
                static_cast<size_t>(file.header.width);
            for (int x = 0; x < file.header.width; ++x) {
                float v = frameValues[zy + static_cast<size_t>(x)];
                if (useClampMin) v = std::max(v, opt.clampMin);
                if (useClampMax) v = std::min(v, opt.clampMax);
                if (std::abs(v - opt.background) <= opt.sparseThreshold) continue;
                accessor.setValue(openvdb::Coord(x0 + x, y0 + y, z0 + z), v);
            }
        }
    }
    grid->tree().prune();

    fs::create_directories(opt.outputVdb.parent_path());
    openvdb::io::File out(opt.outputVdb.string());
    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    out.write(grids);
    out.close();

    std::cout << "Render-temporal VBT -> VDB done\n"
              << "  vbt:    " << opt.inputVbt << "\n"
              << "  meta:   " << opt.metadataPath << "\n"
              << "  frame:  " << opt.frameIndex << "\n"
              << "  clamp:  [" << (useClampMin ? std::to_string(opt.clampMin) : std::string("-inf"))
              << ", " << (useClampMax ? std::to_string(opt.clampMax) : std::string("+inf")) << "]\n"
              << "  shell-empty: scale=" << opt.shellEmptyScale
              << " max=" << (!std::isnan(opt.shellEmptyMax) ? std::to_string(opt.shellEmptyMax) : std::string("disabled")) << "\n"
              << "  output: " << opt.outputVdb << "\n";
    return 0;
}
