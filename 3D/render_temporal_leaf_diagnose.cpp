#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../render/src/vbt_file.h"
#include "../src/field_profile.h"
#include "../src/render_temporal_decode.h"
#include "../src/render_temporal_payload.h"
#include "vdb_tools/raw_frame_reader.h"

namespace fs = std::filesystem;
using namespace vbt;
using namespace vbt::render;

namespace {

struct Options {
    fs::path inputVbt;
    fs::path inputRaw;
    fs::path rawMetadata;
    int frameIndex = 0;
    float isoValue = 0.12f;
    float bandWidth = 0.03f;
    int topCount = 20;
};

struct LeafStats {
    uint32_t leafIndex = 0;
    int bx = 0;
    int by = 0;
    int bz = 0;
    int baseX = 0;
    int baseY = 0;
    int baseZ = 0;
    int width = 0;
    int height = 0;
    int depth = 0;
    TemporalFirstPackedMode mode = TemporalFirstPackedMode::EMPTY;
    RenderTemporalSequenceCodec fineCodec = RenderTemporalSequenceCodec::DP_KEYFRAME_FP16;
    uint32_t payloadBytes = 0;
    uint32_t coarseKeyframes = 0;
    uint32_t fineKeyframes = 0;
    uint32_t shellActiveVoxelCount = 0;
    uint32_t voxelCount = 0;
    uint32_t activeCount = 0;
    uint32_t bandCount = 0;
    uint32_t signFlipCount = 0;
    uint32_t coarseSignFlipCount = 0;
    uint32_t negativeCount = 0;
    uint32_t overOneCount = 0;
    double sumSq = 0.0;
    double coarseSumSq = 0.0;
    double activeSumSq = 0.0;
    double coarseActiveSumSq = 0.0;
    double bandSumSq = 0.0;
    double coarseBandSumSq = 0.0;
    float predMin = std::numeric_limits<float>::max();
    float predMax = std::numeric_limits<float>::lowest();
    float truthMin = std::numeric_limits<float>::max();
    float truthMax = std::numeric_limits<float>::lowest();
    double maxAbsError = 0.0;
    int worstX = 0;
    int worstY = 0;
    int worstZ = 0;
    float worstTruth = 0.0f;
    float worstCoarse = 0.0f;
    float worstResidual = 0.0f;
    float worstPred = 0.0f;
};

struct ModeSummary {
    uint32_t leafCount = 0;
    uint64_t voxelCount = 0;
    uint64_t activeCount = 0;
    uint64_t bandCount = 0;
    uint64_t signFlipCount = 0;
    uint64_t coarseSignFlipCount = 0;
    uint64_t negativeCount = 0;
    uint64_t overOneCount = 0;
    uint64_t fineKeyframes = 0;
    uint64_t shellActiveVoxelCount = 0;
    double sumSq = 0.0;
    double coarseSumSq = 0.0;
    double activeSumSq = 0.0;
    double coarseActiveSumSq = 0.0;
    double bandSumSq = 0.0;
    double coarseBandSumSq = 0.0;
};

void printUsage()
{
    std::cout
        << "Usage:\n"
        << "  render_temporal_leaf_diagnose.exe --input-vbt <file.vbtp> --input-raw <file.raw> --raw-metadata <meta.json>\n"
        << "                                   [--frame <index>] [--iso-value 0.12] [--band-width 0.03] [--top 20]\n";
}

bool parseArgs(int argc, char** argv, Options& opt)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--input-vbt" && i + 1 < argc) {
            opt.inputVbt = argv[++i];
        } else if (arg == "--input-raw" && i + 1 < argc) {
            opt.inputRaw = argv[++i];
        } else if (arg == "--raw-metadata" && i + 1 < argc) {
            opt.rawMetadata = argv[++i];
        } else if (arg == "--frame" && i + 1 < argc) {
            opt.frameIndex = std::stoi(argv[++i]);
        } else if (arg == "--iso-value" && i + 1 < argc) {
            opt.isoValue = std::stof(argv[++i]);
        } else if (arg == "--band-width" && i + 1 < argc) {
            opt.bandWidth = std::stof(argv[++i]);
        } else if (arg == "--top" && i + 1 < argc) {
            opt.topCount = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            printUsage();
            return false;
        }
    }
    if (opt.inputVbt.empty() || opt.inputRaw.empty() || opt.rawMetadata.empty()) {
        printUsage();
        return false;
    }
    if (opt.topCount <= 0) opt.topCount = 20;
    if (opt.bandWidth <= 0.0f) opt.bandWidth = 0.03f;
    return true;
}

const char* modeName(TemporalFirstPackedMode mode)
{
    switch (mode) {
    case TemporalFirstPackedMode::EMPTY:
        return "EMPTY";
    case TemporalFirstPackedMode::TEMPORAL_GRID4:
        return "TEMPORAL_GRID4";
    case TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT:
        return "TEMPORAL_FINE_COMPACT";
    case TemporalFirstPackedMode::TEMPORAL_FINE6:
        return "TEMPORAL_FINE6";
    default:
        return "UNKNOWN";
    }
}

const char* codecName(RenderTemporalSequenceCodec codec)
{
    switch (codec) {
    case RenderTemporalSequenceCodec::DP_KEYFRAME_FP32:
        return "FP32";
    case RenderTemporalSequenceCodec::DP_KEYFRAME_FP16:
    default:
        return "FP16";
    }
}

double safeRmse(double sumSq, uint64_t count)
{
    if (count == 0) return 0.0;
    return std::sqrt(sumSq / static_cast<double>(count));
}

double safeRate(uint64_t num, uint64_t den)
{
    if (den == 0) return 0.0;
    return static_cast<double>(num) / static_cast<double>(den);
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
    static std::map<int, std::vector<float>> cache;
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

void decodeLeafFrameValues(const uint8_t* leafBase,
                           size_t leafBytes,
                           const VbtFileHeader& fileHeader,
                           int frameIndex,
                           int leafWidth,
                           int leafHeight,
                           int leafDepth,
                           std::vector<float>& outLeaf,
                           std::vector<float>* outCoarseLeaf,
                           std::vector<float>* outResidualLeaf)
{
    const auto leafView = parseRenderTemporalPackedLeaf(leafBase, leafBytes, true);
    if (leafView.header.mode == TemporalFirstPackedMode::EMPTY) {
        std::fill(outLeaf.begin(), outLeaf.end(), 0.0f);
        if (outCoarseLeaf != nullptr) std::fill(outCoarseLeaf->begin(), outCoarseLeaf->end(), 0.0f);
        if (outResidualLeaf != nullptr) std::fill(outResidualLeaf->begin(), outResidualLeaf->end(), 0.0f);
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

    for (int lz = 0; lz < leafDepth; ++lz) {
        for (int ly = 0; ly < leafHeight; ++ly) {
            for (int lx = 0; lx < leafWidth; ++lx) {
                float value = sampleControlGrid(coarseGrid,
                                                coarseResolution,
                                                leafWidth,
                                                leafHeight,
                                                leafDepth,
                                                static_cast<float>(lx),
                                                static_cast<float>(ly),
                                                static_cast<float>(lz));
                const float coarseValue = value;
                float residualValue = 0.0f;
                if (!fineGrid.empty()) {
                    residualValue = sampleControlGrid(fineGrid,
                                                      fineResolution,
                                                      leafWidth,
                                                      leafHeight,
                                                      leafDepth,
                                                      static_cast<float>(lx),
                                                      static_cast<float>(ly),
                                                      static_cast<float>(lz));
                    value += residualValue;
                } else if (leafView.header.mode == TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT) {
                    residualValue = decodeRenderTemporalLeafShellVoxelValue(
                        leafBase,
                        leafView,
                        true,
                        static_cast<int>(fileHeader.frames),
                        renderTemporalLeafVoxelIndex(
                            static_cast<uint16_t>(lx),
                            static_cast<uint16_t>(ly),
                            static_cast<uint16_t>(lz),
                            static_cast<uint16_t>(fileHeader.leafSize)),
                        frameIndex);
                    value += residualValue;
                }
                const size_t localIdx =
                    (static_cast<size_t>(lz) * static_cast<size_t>(leafHeight) + static_cast<size_t>(ly)) *
                        static_cast<size_t>(leafWidth) +
                    static_cast<size_t>(lx);
                if (outCoarseLeaf != nullptr) {
                    (*outCoarseLeaf)[localIdx] = coarseValue;
                }
                if (outResidualLeaf != nullptr) {
                    (*outResidualLeaf)[localIdx] = residualValue;
                }
                outLeaf[localIdx] = value;
            }
        }
    }
}

std::string formatDouble(double value, int precision = 6)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

} // namespace

int main(int argc, char** argv)
{
    Options opt;
    if (!parseArgs(argc, argv, opt)) return 1;

    std::string error;
    VbtFile file;
    if (!loadVbtFile(opt.inputVbt, file, error)) {
        std::cerr << "Failed to load VBT file: " << error << "\n";
        return 2;
    }
    if (file.header.profileType != static_cast<uint32_t>(FieldType::DENSITY)) {
        std::cerr << "Only density/render VBTPACK4 files are supported\n";
        return 3;
    }

    const vdbtools::FrameMetadata rawMetadata =
        vdbtools::loadFrameMetadata(opt.rawMetadata);
    if (opt.frameIndex < 0 || opt.frameIndex >= rawMetadata.frames) {
        std::cerr << "Frame index out of range: " << opt.frameIndex
                  << " / " << rawMetadata.frames << "\n";
        return 4;
    }
    if (rawMetadata.width != static_cast<int>(file.header.width) ||
        rawMetadata.height != static_cast<int>(file.header.height) ||
        rawMetadata.depth != static_cast<int>(file.header.depth) ||
        rawMetadata.frames != static_cast<int>(file.header.frames)) {
        std::cerr << "Raw volume dimensions do not match VBTPACK4 header\n";
        return 5;
    }
    const std::vector<float> truthFrame =
        vdbtools::loadRawFrame(opt.inputRaw, rawMetadata, opt.frameIndex);

    const int leafSize = static_cast<int>(file.header.leafSize);
    const int leafCountX = (static_cast<int>(file.header.width) + leafSize - 1) / leafSize;
    const int leafCountY = (static_cast<int>(file.header.height) + leafSize - 1) / leafSize;
    const int leafCountZ = (static_cast<int>(file.header.depth) + leafSize - 1) / leafSize;
    const uint8_t* payloadBase = reinterpret_cast<const uint8_t*>(file.payloadWords.data());

    std::vector<LeafStats> allLeaves;
    allLeaves.reserve(file.header.leafCount);
    std::map<TemporalFirstPackedMode, ModeSummary> modeSummary;

    for (int bz = 0; bz < leafCountZ; ++bz) {
        for (int by = 0; by < leafCountY; ++by) {
            for (int bx = 0; bx < leafCountX; ++bx) {
                const uint32_t leafIndex =
                    static_cast<uint32_t>((bz * leafCountY + by) * leafCountX + bx);
                const uint32_t wordBegin = file.offsetsWords[leafIndex];
                const uint32_t wordEnd = file.offsetsWords[leafIndex + 1u];
                const size_t leafBytes = static_cast<size_t>(wordEnd - wordBegin) * sizeof(uint32_t);
                const uint8_t* leafBase = payloadBase + static_cast<size_t>(wordBegin) * sizeof(uint32_t);
                const auto leafView = parseRenderTemporalPackedLeaf(leafBase, leafBytes, true);

                const int baseX = bx * leafSize;
                const int baseY = by * leafSize;
                const int baseZ = bz * leafSize;
                const int leafWidth = std::min(leafSize, static_cast<int>(file.header.width) - baseX);
                const int leafHeight = std::min(leafSize, static_cast<int>(file.header.height) - baseY);
                const int leafDepth = std::min(leafSize, static_cast<int>(file.header.depth) - baseZ);

                std::vector<float> decoded(static_cast<size_t>(leafWidth) *
                                           static_cast<size_t>(leafHeight) *
                                           static_cast<size_t>(leafDepth),
                                           0.0f);
                std::vector<float> coarseDecoded(decoded.size(), 0.0f);
                std::vector<float> residualDecoded(decoded.size(), 0.0f);
                decodeLeafFrameValues(leafBase,
                                      leafBytes,
                                      file.header,
                                      opt.frameIndex,
                                      leafWidth,
                                      leafHeight,
                                      leafDepth,
                                      decoded,
                                      &coarseDecoded,
                                      &residualDecoded);

                LeafStats stats;
                stats.leafIndex = leafIndex;
                stats.bx = bx;
                stats.by = by;
                stats.bz = bz;
                stats.baseX = baseX;
                stats.baseY = baseY;
                stats.baseZ = baseZ;
                stats.width = leafWidth;
                stats.height = leafHeight;
                stats.depth = leafDepth;
                stats.mode = leafView.header.mode;
                stats.fineCodec = leafView.header.fineCodec;
                stats.payloadBytes = static_cast<uint32_t>(leafBytes);
                stats.coarseKeyframes = leafView.prefix.coarseKeyframeCount;
                stats.fineKeyframes = leafView.prefix.fineKeyframeCount;
                stats.shellActiveVoxelCount = leafView.shellActiveVoxelCount;
                stats.voxelCount = static_cast<uint32_t>(decoded.size());

                for (int lz = 0; lz < leafDepth; ++lz) {
                    for (int ly = 0; ly < leafHeight; ++ly) {
                        for (int lx = 0; lx < leafWidth; ++lx) {
                            const size_t localIdx =
                                (static_cast<size_t>(lz) * static_cast<size_t>(leafHeight) + static_cast<size_t>(ly)) *
                                    static_cast<size_t>(leafWidth) +
                                static_cast<size_t>(lx);
                            const float pred = decoded[localIdx];
                            const float coarsePred = coarseDecoded[localIdx];
                            const float residualPred = residualDecoded[localIdx];
                            const size_t truthIndex =
                                (static_cast<size_t>(baseZ + lz) *
                                     static_cast<size_t>(rawMetadata.height) +
                                 static_cast<size_t>(baseY + ly)) *
                                    static_cast<size_t>(rawMetadata.width) +
                                static_cast<size_t>(baseX + lx);
                            const float truth = truthFrame[truthIndex];
                            stats.predMin = std::min(stats.predMin, pred);
                            stats.predMax = std::max(stats.predMax, pred);
                            stats.truthMin = std::min(stats.truthMin, truth);
                            stats.truthMax = std::max(stats.truthMax, truth);
                            const double diff = static_cast<double>(pred) - static_cast<double>(truth);
                            const double coarseDiff = static_cast<double>(coarsePred) - static_cast<double>(truth);
                            stats.sumSq += diff * diff;
                            stats.coarseSumSq += coarseDiff * coarseDiff;
                            if (truth > 1.0e-6f) {
                                stats.activeCount += 1u;
                                stats.activeSumSq += diff * diff;
                                stats.coarseActiveSumSq += coarseDiff * coarseDiff;
                            }
                            if (std::abs(truth - opt.isoValue) <= opt.bandWidth) {
                                stats.bandCount += 1u;
                                stats.bandSumSq += diff * diff;
                                stats.coarseBandSumSq += coarseDiff * coarseDiff;
                                if ((truth >= opt.isoValue) != (coarsePred >= opt.isoValue)) {
                                    stats.coarseSignFlipCount += 1u;
                                }
                                if ((truth >= opt.isoValue) != (pred >= opt.isoValue)) {
                                    stats.signFlipCount += 1u;
                                }
                            }
                            if (pred < 0.0f) stats.negativeCount += 1u;
                            if (pred > 1.0f) stats.overOneCount += 1u;
                            const double absErr = std::abs(diff);
                            if (absErr > stats.maxAbsError) {
                                stats.maxAbsError = absErr;
                                stats.worstX = baseX + lx;
                                stats.worstY = baseY + ly;
                                stats.worstZ = baseZ + lz;
                                stats.worstTruth = truth;
                                stats.worstCoarse = coarsePred;
                                stats.worstResidual = residualPred;
                                stats.worstPred = pred;
                            }
                        }
                    }
                }

                auto& summary = modeSummary[stats.mode];
                summary.leafCount += 1u;
                summary.voxelCount += stats.voxelCount;
                summary.activeCount += stats.activeCount;
                summary.bandCount += stats.bandCount;
                summary.signFlipCount += stats.signFlipCount;
                summary.coarseSignFlipCount += stats.coarseSignFlipCount;
                summary.negativeCount += stats.negativeCount;
                summary.overOneCount += stats.overOneCount;
                summary.fineKeyframes += stats.fineKeyframes;
                summary.shellActiveVoxelCount += stats.shellActiveVoxelCount;
                summary.sumSq += stats.sumSq;
                summary.coarseSumSq += stats.coarseSumSq;
                summary.activeSumSq += stats.activeSumSq;
                summary.coarseActiveSumSq += stats.coarseActiveSumSq;
                summary.bandSumSq += stats.bandSumSq;
                summary.coarseBandSumSq += stats.coarseBandSumSq;

                allLeaves.push_back(stats);
            }
        }
    }

    std::sort(allLeaves.begin(), allLeaves.end(), [](const LeafStats& a, const LeafStats& b) {
        const double aFlip = safeRate(a.signFlipCount, a.bandCount);
        const double bFlip = safeRate(b.signFlipCount, b.bandCount);
        if (a.bandCount == 0 && b.bandCount > 0) return false;
        if (a.bandCount > 0 && b.bandCount == 0) return true;
        if (aFlip != bFlip) return aFlip > bFlip;
        const double aBandRmse = safeRmse(a.bandSumSq, a.bandCount);
        const double bBandRmse = safeRmse(b.bandSumSq, b.bandCount);
        if (aBandRmse != bBandRmse) return aBandRmse > bBandRmse;
        return a.leafIndex < b.leafIndex;
    });

    std::cout << "# Render Temporal Leaf Diagnose\n\n";
    std::cout << "- input-vbt: `" << opt.inputVbt.string() << "`\n";
    std::cout << "- input-raw: `" << opt.inputRaw.string() << "`\n";
    std::cout << "- frame: `" << opt.frameIndex << "`\n";
    std::cout << "- iso-value: `" << opt.isoValue << "`\n";
    std::cout << "- band-width: `" << opt.bandWidth << "`\n";
    std::cout << "- leaf-size: `" << file.header.leafSize << "`\n";
    std::cout << "- leaf-count: `" << file.header.leafCount << "`\n\n";

    std::cout << "## Mode Summary\n\n";
    std::cout << "| Mode | Leaves | Voxels | Active Voxels | Shell Active | Fine KFs | KF/Active | Band Voxels | Coarse Flip N | Coarse Flip | Final Flip N | Final Flip | Coarse Band RMSE | Final Band RMSE | Coarse Active RMSE | Final Active RMSE | NegFrac | Over1Frac |\n";
    std::cout << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    ModeSummary totalSummary;
    for (const auto& [mode, summary] : modeSummary) {
        const double kfPerActive = safeRate(summary.fineKeyframes, summary.shellActiveVoxelCount);
        std::cout << "| " << modeName(mode)
                  << " | " << summary.leafCount
                  << " | " << summary.voxelCount
                  << " | " << summary.activeCount
                  << " | " << summary.shellActiveVoxelCount
                  << " | " << summary.fineKeyframes
                  << " | " << formatDouble(kfPerActive, 2)
                  << " | " << summary.bandCount
                  << " | " << summary.coarseSignFlipCount
                  << " | " << formatDouble(safeRate(summary.coarseSignFlipCount, summary.bandCount), 4)
                  << " | " << summary.signFlipCount
                  << " | " << formatDouble(safeRate(summary.signFlipCount, summary.bandCount), 4)
                  << " | " << formatDouble(safeRmse(summary.coarseBandSumSq, summary.bandCount), 5)
                  << " | " << formatDouble(safeRmse(summary.bandSumSq, summary.bandCount), 5)
                  << " | " << formatDouble(safeRmse(summary.coarseActiveSumSq, summary.activeCount), 5)
                  << " | " << formatDouble(safeRmse(summary.activeSumSq, summary.activeCount), 5)
                  << " | " << formatDouble(safeRate(summary.negativeCount, summary.voxelCount), 4)
                  << " | " << formatDouble(safeRate(summary.overOneCount, summary.voxelCount), 4)
                  << " |\n";
        totalSummary.leafCount += summary.leafCount;
        totalSummary.voxelCount += summary.voxelCount;
        totalSummary.activeCount += summary.activeCount;
        totalSummary.bandCount += summary.bandCount;
        totalSummary.signFlipCount += summary.signFlipCount;
        totalSummary.coarseSignFlipCount += summary.coarseSignFlipCount;
        totalSummary.negativeCount += summary.negativeCount;
        totalSummary.overOneCount += summary.overOneCount;
        totalSummary.fineKeyframes += summary.fineKeyframes;
        totalSummary.shellActiveVoxelCount += summary.shellActiveVoxelCount;
        totalSummary.sumSq += summary.sumSq;
        totalSummary.coarseSumSq += summary.coarseSumSq;
        totalSummary.activeSumSq += summary.activeSumSq;
        totalSummary.coarseActiveSumSq += summary.coarseActiveSumSq;
        totalSummary.bandSumSq += summary.bandSumSq;
        totalSummary.coarseBandSumSq += summary.coarseBandSumSq;
    }
    std::cout << "| TOTAL"
              << " | " << totalSummary.leafCount
              << " | " << totalSummary.voxelCount
              << " | " << totalSummary.activeCount
              << " | " << totalSummary.shellActiveVoxelCount
              << " | " << totalSummary.fineKeyframes
              << " | " << formatDouble(safeRate(totalSummary.fineKeyframes, totalSummary.shellActiveVoxelCount), 2)
              << " | " << totalSummary.bandCount
              << " | " << totalSummary.coarseSignFlipCount
              << " | " << formatDouble(safeRate(totalSummary.coarseSignFlipCount, totalSummary.bandCount), 4)
              << " | " << totalSummary.signFlipCount
              << " | " << formatDouble(safeRate(totalSummary.signFlipCount, totalSummary.bandCount), 4)
              << " | " << formatDouble(safeRmse(totalSummary.coarseBandSumSq, totalSummary.bandCount), 5)
              << " | " << formatDouble(safeRmse(totalSummary.bandSumSq, totalSummary.bandCount), 5)
              << " | " << formatDouble(safeRmse(totalSummary.coarseActiveSumSq, totalSummary.activeCount), 5)
              << " | " << formatDouble(safeRmse(totalSummary.activeSumSq, totalSummary.activeCount), 5)
              << " | " << formatDouble(safeRate(totalSummary.negativeCount, totalSummary.voxelCount), 4)
              << " | " << formatDouble(safeRate(totalSummary.overOneCount, totalSummary.voxelCount), 4)
              << " |\n";

    std::cout << "\n## Worst Leaves By Iso-Band Instability\n\n";
    std::cout << "| Rank | Leaf | Mode | FineCodec | Block `(bx,by,bz)` | Base `(x,y,z)` | Payload B | Shell Active | Fine KFs | KF/Active | Band Voxels | Coarse Flip N | Coarse Flip | Final Flip N | Final Flip | Coarse Band RMSE | Final Band RMSE | Coarse Active RMSE | Final Active RMSE | Final Range | MaxAbsErr |\n";
    std::cout << "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|\n";
    const int limit = std::min<int>(opt.topCount, static_cast<int>(allLeaves.size()));
    for (int i = 0; i < limit; ++i) {
        const auto& s = allLeaves[static_cast<size_t>(i)];
        const double kfPerActive = safeRate(s.fineKeyframes, s.shellActiveVoxelCount);
        std::cout << "| " << (i + 1)
                  << " | " << s.leafIndex
                  << " | " << modeName(s.mode)
                  << " | " << codecName(s.fineCodec)
                  << " | (" << s.bx << "," << s.by << "," << s.bz << ")"
                  << " | (" << s.baseX << "," << s.baseY << "," << s.baseZ << ")"
                  << " | " << s.payloadBytes
                  << " | " << s.shellActiveVoxelCount
                  << " | " << s.fineKeyframes
                  << " | " << formatDouble(kfPerActive, 2)
                  << " | " << s.bandCount
                  << " | " << s.coarseSignFlipCount
                  << " | " << formatDouble(safeRate(s.coarseSignFlipCount, s.bandCount), 4)
                  << " | " << s.signFlipCount
                  << " | " << formatDouble(safeRate(s.signFlipCount, s.bandCount), 4)
                  << " | " << formatDouble(safeRmse(s.coarseBandSumSq, s.bandCount), 5)
                  << " | " << formatDouble(safeRmse(s.bandSumSq, s.bandCount), 5)
                  << " | " << formatDouble(safeRmse(s.coarseActiveSumSq, s.activeCount), 5)
                  << " | " << formatDouble(safeRmse(s.activeSumSq, s.activeCount), 5)
                  << " | [" << formatDouble(s.predMin, 3) << ", " << formatDouble(s.predMax, 3) << "]"
                  << " | " << formatDouble(s.maxAbsError, 5)
                  << " |\n";
    }

    std::cout << "\n## Worst-Voxel Details For Top Leaves\n\n";
    std::cout << "| Rank | Leaf | Worst `(x,y,z)` | Truth | Coarse | Residual | Final | AbsErr | CoarseAbsErr | Residual Helps |\n";
    std::cout << "|---|---:|---|---:|---:|---:|---:|---:|---:|---|\n";
    for (int i = 0; i < limit; ++i) {
        const auto& s = allLeaves[static_cast<size_t>(i)];
        const double coarseAbsErr = std::abs(static_cast<double>(s.worstCoarse) - static_cast<double>(s.worstTruth));
        const bool residualHelps = s.maxAbsError < coarseAbsErr;
        std::cout << "| " << (i + 1)
                  << " | " << s.leafIndex
                  << " | (" << s.worstX << "," << s.worstY << "," << s.worstZ << ")"
                  << " | " << formatDouble(s.worstTruth, 5)
                  << " | " << formatDouble(s.worstCoarse, 5)
                  << " | " << formatDouble(s.worstResidual, 5)
                  << " | " << formatDouble(s.worstPred, 5)
                  << " | " << formatDouble(s.maxAbsError, 5)
                  << " | " << formatDouble(coarseAbsErr, 5)
                  << " | " << (residualHelps ? "yes" : "no")
                  << " |\n";
    }

    return 0;
}
