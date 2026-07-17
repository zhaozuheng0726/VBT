#include "raw_volume.h"
#include "render_temporal_decode.h"
#include "render_temporal_formal.h"
#include "render_temporal_mainline.h"
#include "render_temporal_payload.h"
#include "render_temporal_route.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef VBT_USE_OPENMP
#include <omp.h>
#endif

using namespace vbt;

namespace {

struct ProbeCliOptions {
    std::filesystem::path inputRaw;
    std::filesystem::path metadataPath;
    std::filesystem::path reportPath;
    std::filesystem::path saveVbtPath;
    std::filesystem::path exportFrameRawPath;
    std::filesystem::path exportFrameMetadataPath;
    std::string exportVariantName;
    int exportFrameIndex = -1;
    int sampleStep = 4;
    int ompThreads = 0;
    bool finalOnly = false;
    float cutoff = 0.0003f;
    float cutoffBand = 0.0001f;
    float temporalEpsAbs = 1e-5f;
    float temporalEpsRel = 0.02f;
    float temporalGammaDelta = 0.2f;
    float controlEpsScale = 1.0f;
    float bgZeroRatio = 0.30f;
    bool cutoffProtect = true;
    float routeEmptyVisibleThr = 0.001f;
    float routeFineVisibleThr = 0.02f;
    float routeFineBandThr = 0.01f;
    float routeCoarseRmseThr = 0.015f;
    float routeCoarsePeakThr = 0.06f;
    float routeFineGainThr = 0.15f;
    float routeShellFracThr = 1.0f;
    float routeCoarseShellRmseThr = 1.0f;
    float routeCoarseShellPeakThr = 1.0f;
    float routeShellGainThr = 1.0f;
    RenderTemporalSequenceCodec packedCodec = RenderTemporalSequenceCodec::DP_KEYFRAME_FP16;
    uint64_t savedFileBytes = 0;
    bool preferShellVoxelSave = false;
    bool levelSetSurfaceMode = false;
    bool legacyLevelSetRoute = false;
    float levelSetSurfaceBand = 0.0f;
    float levelSetCoarseGuardBand = 0.0f;
    float levelSetTemporalBand = 0.0f;
};

struct DensityTemporalProfile {
    float epsAbs = 1e-5f;
    float epsRel = 0.02f;
    float gammaDelta = 0.2f;
    float renderCutoff = 0.0003f;
    float cutoffBand = 0.0001f;
    bool cutoffTemporalProtect = true;
    bool isoSurfaceProtect = false;
    float isoValue = 0.0f;
    float isoBand = 0.0f;
};

struct GenericTemporalStats {
    float mean = 0.0f;
    float stddev = 0.0f;
    float deltaScale = 1.0f;
};

struct TemporalStats {
    uint64_t voxelCount = 0;
    uint64_t backgroundZeroedVoxels = 0;
    uint64_t totalKeyframes = 0;
    uint64_t temporalProtectedSeries = 0;
    int maxKeyframes = 0;
    double rmse = 0.0;
    double psnr = 0.0;
};

struct VariantResult {
    std::string name;
    int fineResolution = 0;
    uint64_t leafCount = 0;
    uint64_t emptyLeafCount = 0;
    uint64_t fineLeafCount = 0;
    uint64_t coarseKeyframes = 0;
    uint64_t fineKeyframes = 0;
    uint64_t estimatedBytes = 0;
    uint64_t samples = 0;
    double rmse = 0.0;
    double psnr = 0.0;
};

struct BuiltControlStream {
    RenderTemporalControlStreamData data;
    uint8_t requiredMaxBinLocalKeys = 1;
};

struct LeafSamplePoint {
    int lx = 0;
    int ly = 0;
    int lz = 0;
    int t = 0;
    float truth = 0.0f;
    float coarsePred = 0.0f;
    float fine6Pred = 0.0f;
    float fine6BandPred = 0.0f;
    float fine6GainPred = 0.0f;
    float shellVoxelPred = 0.0f;
};

constexpr int kLeafSize = 8;
constexpr int kLeafVoxelCount = kLeafSize * kLeafSize * kLeafSize;
constexpr int kCoarseRes = 4;

void printUsage()
{
    std::cout
        << "Usage:\n"
        << "  vbt_render_temporal_kf_probe --input-raw <raw> [--metadata <meta.json>]\n"
        << "                              [--export-frame-raw <frame.raw> --export-frame <index> --export-variant <name>]\n"
        << "                              [--export-frame-metadata <frame.metadata.json>]\n"
        << "                              [--sample-step 4]\n"
        << "                              [--cutoff 0.0003] [--cutoff-band 0.0001]\n"
        << "                              [--temporal-eps-abs 1e-5] [--temporal-eps-rel 0.02]\n"
        << "                              [--control-eps-scale 1.0]\n"
        << "                              [--temporal-gamma-delta 0.2] [--bg-zero-ratio 0.30]\n"
        << "                              [--packed-keyframe-codec fp16|fp32]\n"
        << "                              [--final-only]\n"
        << "                              [--route-empty-visible-thr 0.001] [--route-fine-visible-thr 0.02]\n"
        << "                              [--route-fine-band-thr 0.01] [--route-coarse-rmse-thr 0.015]\n"
        << "                              [--route-coarse-peak-thr 0.06] [--route-fine-gain-thr 0.15]\n"
        << "                              [--levelset-surface-band <world-distance>]\n"
        << "                              [--levelset-coarse-guard-band <world-distance>]\n"
        << "                              [--levelset-temporal-band <world-distance>] [--legacy-levelset-route]\n"
        << "                              [--cutoff-protect|--no-cutoff-protect]\n"
        << "                              [--omp-threads N] [--save-vbt <file.vbtp>] [--report <path>]\n";
}

double computePsnr(double rmse, double peak)
{
    if (rmse <= 0.0) return 120.0;
    return 20.0 * std::log10(std::max(peak, 1e-12) / rmse);
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

GenericTemporalStats computeStats(const std::vector<float>& values)
{
    GenericTemporalStats stats;
    if (values.empty()) return stats;
    double sum = 0.0;
    double sum2 = 0.0;
    for (float v : values) {
        sum += v;
        sum2 += static_cast<double>(v) * v;
    }
    stats.mean = static_cast<float>(sum / static_cast<double>(values.size()));
    const double var =
        std::max(0.0, sum2 / static_cast<double>(values.size()) - static_cast<double>(stats.mean) * stats.mean);
    stats.stddev = static_cast<float>(std::sqrt(var));
    if (values.size() >= 2) {
        std::vector<float> deltas;
        deltas.reserve(values.size() - 1);
        for (size_t i = 1; i < values.size(); ++i) {
            deltas.push_back(std::abs(values[i] - values[i - 1]));
        }
        std::sort(deltas.begin(), deltas.end());
        const size_t idx = std::min(deltas.size() - 1, (deltas.size() * 9) / 10);
        stats.deltaScale = std::max(1e-6f, deltas[idx]);
    }
    return stats;
}

int cutoffState(float v, float cutoff, float band)
{
    if (cutoff <= 0.0f) return 1;
    if (v < cutoff - band) return 0;
    if (v > cutoff + band) return 2;
    return 1;
}

float epsilonAt(const std::vector<float>& values,
                int t,
                const DensityTemporalProfile& profile,
                const GenericTemporalStats& stats)
{
    float eps = profile.epsAbs + profile.epsRel * std::max(std::abs(values[static_cast<size_t>(t)]), stats.stddev);
    if (profile.gammaDelta > 0.0f && t > 0) {
        const float delta = std::abs(values[static_cast<size_t>(t)] - values[static_cast<size_t>(t - 1)]) /
                            std::max(1e-6f, stats.deltaScale);
        eps /= (1.0f + profile.gammaDelta * delta);
    }
    return std::max(eps, std::max(1e-6f, profile.epsAbs * 0.1f));
}

void dpRecurse(const std::vector<float>& values,
               int a,
               int b,
               const DensityTemporalProfile& profile,
               const GenericTemporalStats& stats,
               std::vector<int>& keys)
{
    if (b - a <= 1) return;
    const double v0 = values[static_cast<size_t>(a)];
    const double v1 = values[static_cast<size_t>(b)];
    double maxErrNorm = 0.0;
    int split = -1;
    for (int t = a + 1; t < b; ++t) {
        const double alpha = static_cast<double>(t - a) / static_cast<double>(b - a);
        const double pred = v0 + alpha * (v1 - v0);
        const double eps = epsilonAt(values, t, profile, stats);
        const double errNorm = std::abs(static_cast<double>(values[static_cast<size_t>(t)]) - pred) / eps;
        if (errNorm > maxErrNorm) {
            maxErrNorm = errNorm;
            split = t;
        }
    }
    if (maxErrNorm > 1.0 && split != -1) {
        keys.push_back(split);
        dpRecurse(values, a, split, profile, stats, keys);
        dpRecurse(values, split, b, profile, stats, keys);
    }
}

std::vector<int> detectDensityKeyFrames(const std::vector<float>& values,
                                        const DensityTemporalProfile& profile,
                                        bool* usedCutoffProtect);
void addIsoSurfaceKeyFrames(const std::vector<float>& values,
                           float isoValue,
                           float isoBand,
                           std::vector<int>& keys);

void reconstructFromKeys(const std::vector<float>& values,
                         const std::vector<int>& keys,
                         std::vector<float>& out);

inline int localVoxelIndex(int lx, int ly, int lz)
{
    return static_cast<int>(renderTemporalLeafVoxelIndex(
        static_cast<uint16_t>(lx),
        static_cast<uint16_t>(ly),
        static_cast<uint16_t>(lz),
        static_cast<uint16_t>(kLeafSize)));
}

inline int localTileVoxelIndex(int lx, int ly, int lz)
{
    return (lz * 4 + ly) * 4 + lx;
}

inline float leafTemporalAt(const std::vector<float>& leafTemporal, int voxelIdx, int frames, int t)
{
    return leafTemporal[static_cast<size_t>(voxelIdx) * static_cast<size_t>(frames) + static_cast<size_t>(t)];
}

float sampleLeafBufferAtFractional(const std::vector<float>& leafTemporal,
                                   int frames,
                                   int t,
                                   int leafWidth,
                                   int leafHeight,
                                   int leafDepth,
                                   float fx,
                                   float fy,
                                   float fz);

float sampleControlGridValue(const std::vector<float>& gridSeries,
                             int resolution,
                             int frames,
                             int t,
                             int leafWidth,
                             int leafHeight,
                             int leafDepth,
                             float fx,
                             float fy,
                             float fz);

float shellGainFromCoarse(float coarseValue, float dataMax, float lowRatio, float highRatio, float maxGain)
{
    if (maxGain <= 1.0f || dataMax <= 0.0f) return 1.0f;
    const float low = dataMax * lowRatio;
    const float high = dataMax * highRatio;
    if (coarseValue <= low) return 1.0f;
    if (coarseValue >= high) return maxGain;
    if (high <= low) return maxGain;
    const float u = (coarseValue - low) / (high - low);
    const float s = u * u * (3.0f - 2.0f * u);
    return 1.0f + (maxGain - 1.0f) * s;
}

void smoothSeriesInPlace(std::vector<float>& series, float alpha, int passes)
{
    if (series.size() < 3 || alpha <= 0.0f || passes <= 0) return;
    alpha = std::clamp(alpha, 0.0f, 0.49f);
    std::vector<float> tmp(series.size(), 0.0f);
    for (int pass = 0; pass < passes; ++pass) {
        tmp.front() = series.front();
        tmp.back() = series.back();
        for (size_t i = 1; i + 1 < series.size(); ++i) {
            tmp[i] = (1.0f - 2.0f * alpha) * series[i] +
                     alpha * series[i - 1] +
                     alpha * series[i + 1];
        }
        series.swap(tmp);
    }
}

float smoothGuidanceValue(float prev, float cur, float next, float alpha)
{
    alpha = std::clamp(alpha, 0.0f, 0.49f);
    return (1.0f - 2.0f * alpha) * cur + alpha * prev + alpha * next;
}

void writeReport(const std::filesystem::path& path,
                 const ProbeCliOptions& cli,
                 const RawVolume4D& volume,
                 const TemporalStats& temporalStats,
                 const std::vector<VariantResult>& variants);

void writeSingleFrameRaw(const std::filesystem::path& path, const std::vector<float>& frameValues);
void writeSingleFrameMetadata(const std::filesystem::path& path,
                              const FrameMetadata& meta,
                              const std::filesystem::path& sourceRaw);
uint64_t writeRenderTemporalProbeFile(const std::filesystem::path& path,
                                      const RawVolume4D& volume,
                                      int leafSize,
                                      int coarseResolution,
                                      const std::vector<std::vector<uint8_t>>& leafPayloads);
std::vector<uint8_t> buildBinIndexFromKeys(const std::vector<int>& keys, int frames, int timeBinCount);
BuiltControlStream buildControlStreamFromGridSeries(const std::vector<float>& gridSeries,
                                                    int controlCount,
                                                    int frames,
                                                    const DensityTemporalProfile& profile,
                                                    int timeBinCount);
void appendControlSequenceToStream(RenderTemporalControlStreamData& stream,
                                   uint8_t& requiredMaxBinLocalKeys,
                                   const std::vector<float>& series,
                                   const std::vector<int>& keys,
                                   int frames,
                                   int timeBinCount);
float samplePackedLeafGridValue(const std::vector<uint8_t>& leafBytes,
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
                                float fz);

} // namespace

static int runRenderTemporalPipeline(ProbeCliOptions cli, bool printConsole)
{
    if (cli.inputRaw.empty()) {
        printUsage();
        return 1;
    }
    if (cli.metadataPath.empty()) {
        cli.metadataPath = guessMetadataPathForRaw(cli.inputRaw);
        if (cli.metadataPath.empty()) {
            throw std::runtime_error("Metadata not found for raw");
        }
    }
    if (cli.reportPath.empty()) {
        cli.reportPath = std::filesystem::path("reports") / (cli.inputRaw.stem().string() + "_temporal_kf_probe.md");
    }
    const bool exportFrameEnabled = !cli.exportFrameRawPath.empty();

#ifdef VBT_USE_OPENMP
    if (cli.ompThreads > 0) omp_set_num_threads(cli.ompThreads);
#endif

    const auto raw = loadRawVolumeMapped(cli.inputRaw, cli.metadataPath);
    cli.levelSetSurfaceMode =
        raw.meta.conversionMode == "levelset" && !cli.legacyLevelSetRoute;
    if (cli.levelSetSurfaceMode) {
        if (cli.levelSetSurfaceBand <= 0.0f) {
            cli.levelSetSurfaceBand =
                (raw.meta.shellWidthVoxels > 0.0f ? raw.meta.shellWidthVoxels : 1.5f) *
                raw.meta.voxelSize;
        }
        if (cli.levelSetCoarseGuardBand <= 0.0f) {
            cli.levelSetCoarseGuardBand =
                std::max(0.5f * raw.meta.voxelSize, cli.levelSetSurfaceBand * 0.5f);
        }
        if (cli.levelSetTemporalBand <= 0.0f) {
            cli.levelSetTemporalBand = 0.1f * raw.meta.voxelSize;
        }
    }
    const int frames = raw.meta.frames;
    const int leafCountX = (raw.meta.width + kLeafSize - 1) / kLeafSize;
    const int leafCountY = (raw.meta.height + kLeafSize - 1) / kLeafSize;
    const int leafCountZ = (raw.meta.depth + kLeafSize - 1) / kLeafSize;
    const uint64_t totalLeafCount =
        static_cast<uint64_t>(leafCountX) * static_cast<uint64_t>(leafCountY) * static_cast<uint64_t>(leafCountZ);
    const double peak = std::max(1e-6f, raw.meta.dataMax - raw.meta.dataMin);

    DensityTemporalProfile temporalProfile;
    temporalProfile.epsAbs = cli.temporalEpsAbs;
    temporalProfile.epsRel = cli.temporalEpsRel;
    temporalProfile.gammaDelta = cli.temporalGammaDelta;
    temporalProfile.renderCutoff = cli.cutoff;
    temporalProfile.cutoffBand = cli.cutoffBand;
    temporalProfile.cutoffTemporalProtect = cli.cutoffProtect;
    temporalProfile.isoSurfaceProtect = cli.levelSetSurfaceMode;
    temporalProfile.isoValue = 0.0f;
    temporalProfile.isoBand = cli.levelSetTemporalBand;
    DensityTemporalProfile controlProfile = temporalProfile;
    controlProfile.epsAbs *= std::max(1.0f, cli.controlEpsScale);
    controlProfile.epsRel *= std::max(1.0f, cli.controlEpsScale);
    controlProfile.isoSurfaceProtect = false;
    const double bgThreshold = cli.cutoff > 0.0f ? static_cast<double>(cli.bgZeroRatio * cli.cutoff) : -1.0;

    std::vector<RenderTemporalVariantSpec> variantSpecs = buildRenderTemporalFormalProbeVariants();
    if (cli.finalOnly) {
        variantSpecs.erase(
            std::remove_if(
                variantSpecs.begin(),
                variantSpecs.end(),
                [](const RenderTemporalVariantSpec& spec) {
                    return spec.name != "coarse_only" && spec.name != "fine6_full" && !spec.routeSelect;
                }),
            variantSpecs.end());
    }
    const size_t baseVariantCount = variantSpecs.size();
    std::vector<VariantResult> variants;
    variants.reserve(baseVariantCount + 3);
    for (const auto& spec : variantSpecs) {
        VariantResult v;
        v.fineResolution = spec.fineResolution;
        v.name = spec.name;
        v.leafCount = totalLeafCount;
        variants.push_back(v);
    }
    variants.push_back({"fine6_full_packed", 6, totalLeafCount});
    variants.push_back({"shellvoxel35_packed", 0, totalLeafCount});
    const std::string routedPackedVariantName =
        cli.levelSetSurfaceMode ? "levelset_surface_packed" : "routed_empty_grid4_fine6_packed";
    variants.push_back({routedPackedVariantName, cli.levelSetSurfaceMode ? 0 : 6, totalLeafCount});
    const size_t packedFine6VariantIndex = baseVariantCount;
    const size_t packedShellVoxelVariantIndex = baseVariantCount + 1;
    const size_t packedRoutedVariantIndex = baseVariantCount + 2;
    int exportVariantIndex = -1;
    if (exportFrameEnabled) {
        if (cli.exportFrameIndex < 0 || cli.exportFrameIndex >= frames) {
            throw std::runtime_error("export-frame must be within [0, frames)");
        }
        if (cli.exportVariantName.empty()) {
            cli.exportVariantName =
                cli.levelSetSurfaceMode ? routedPackedVariantName : "routed_empty_grid4_fine6";
        }
        for (size_t i = 0; i < variants.size(); ++i) {
            if (variants[i].name == cli.exportVariantName) {
                exportVariantIndex = static_cast<int>(i);
                break;
            }
        }
        if (exportVariantIndex < 0) {
            throw std::runtime_error("Unknown export variant: " + cli.exportVariantName);
        }
        if (cli.exportFrameMetadataPath.empty()) {
            cli.exportFrameMetadataPath = cli.exportFrameRawPath;
            cli.exportFrameMetadataPath.replace_extension(".metadata.json");
        }
    }
    std::vector<float> exportedFrame;
    if (exportFrameEnabled) {
        exportedFrame.assign(raw.frameVoxelCount(), 0.0f);
    }
    std::vector<std::vector<uint8_t>> savedLeafPayloads;
    if (!cli.saveVbtPath.empty()) {
        savedLeafPayloads.resize(static_cast<size_t>(totalLeafCount));
    }

    TemporalStats temporalStats{};
    const auto coarseCoords = controlCoords(kCoarseRes);
    const auto fine6Coords = controlCoords(6);
    const auto tile3Coords = controlCoords(3);
    auto routeOptions = makeRenderTemporalRouteDefaults(cli.cutoff, cli.cutoffBand, cli.bgZeroRatio);
    routeOptions.emptyVisibleFracThreshold = cli.routeEmptyVisibleThr;
    routeOptions.fineVisibleFracThreshold = cli.routeFineVisibleThr;
    routeOptions.fineBandFracThreshold = cli.routeFineBandThr;
    routeOptions.coarseRmseThreshold = cli.routeCoarseRmseThr;
    routeOptions.coarsePeakThreshold = cli.routeCoarsePeakThr;
    routeOptions.fineGainThreshold = cli.routeFineGainThr;
    routeOptions.shellFracThreshold = cli.routeShellFracThr;
    routeOptions.coarseShellRmseThreshold = cli.routeCoarseShellRmseThr;
    routeOptions.coarseShellPeakThreshold = cli.routeCoarseShellPeakThr;
    routeOptions.shellGainThreshold = cli.routeShellGainThr;

    std::atomic<bool> workerFailed{false};
    std::string workerError;

#ifdef VBT_USE_OPENMP
#pragma omp parallel
#endif
    {
        std::vector<float> voxelSeries(static_cast<size_t>(frames), 0.0f);
        std::vector<float> voxelRecon;
        std::vector<float> levelSetTruthSeries(static_cast<size_t>(frames), 0.0f);
        std::vector<float> leafTemporal(static_cast<size_t>(kLeafVoxelCount) * static_cast<size_t>(frames), 0.0f);
        std::vector<float> coarseGrid(static_cast<size_t>(kCoarseRes * kCoarseRes * kCoarseRes * frames), 0.0f);
        std::vector<float> fine6Grid(static_cast<size_t>(6 * 6 * 6 * frames), 0.0f);
        std::vector<float> fine6BandGrid(static_cast<size_t>(6 * 6 * 6 * frames), 0.0f);
        std::vector<float> shellVoxelGrid(static_cast<size_t>(kLeafVoxelCount * frames), 0.0f);
        std::vector<float> fine6GainGrid(static_cast<size_t>(6 * 6 * 6 * frames), 0.0f);
        std::vector<float> tile3Residual(static_cast<size_t>(8 * 27 * frames), 0.0f);
        std::vector<LeafSamplePoint> samplePoints;

        double localTemporalErr2 = 0.0;
        uint64_t localBgZero = 0;
        uint64_t localTotalKeys = 0;
        uint64_t localProtected = 0;
        int localMaxKeys = 0;

        std::vector<double> localErr2(variants.size(), 0.0);
        std::vector<uint64_t> localSamples(variants.size(), 0);
        std::vector<uint64_t> localEmptyLeaves(variants.size(), 0);
        std::vector<uint64_t> localFineLeaves(variants.size(), 0);
        std::vector<uint64_t> localCoarseKf(variants.size(), 0);
        std::vector<uint64_t> localFineKf(variants.size(), 0);
        std::vector<uint64_t> localPackedBytes(variants.size(), 0);

        // main streaming loop appended below
#ifdef VBT_USE_OPENMP
#pragma omp for schedule(dynamic, 2)
#endif
        for (int bz = 0; bz < leafCountZ; ++bz) {
            if (workerFailed.load(std::memory_order_relaxed)) continue;
            int currentBx = -1;
            int currentBy = -1;
            try {
            for (int by = 0; by < leafCountY; ++by) {
                currentBy = by;
                for (int bx = 0; bx < leafCountX; ++bx) {
                    currentBx = bx;
                    const uint64_t leafLinearIndex =
                        (static_cast<uint64_t>(bz) * static_cast<uint64_t>(leafCountY) +
                         static_cast<uint64_t>(by)) *
                            static_cast<uint64_t>(leafCountX) +
                        static_cast<uint64_t>(bx);
                    const int baseX = bx * kLeafSize;
                    const int baseY = by * kLeafSize;
                    const int baseZ = bz * kLeafSize;
                    const int leafWidth = std::min(kLeafSize, raw.meta.width - baseX);
                    const int leafHeight = std::min(kLeafSize, raw.meta.height - baseY);
                    const int leafDepth = std::min(kLeafSize, raw.meta.depth - baseZ);

                    std::fill(leafTemporal.begin(), leafTemporal.end(), 0.0f);
                    float leafMax = 0.0f;

                    for (int lz = 0; lz < leafDepth; ++lz) {
                        for (int ly = 0; ly < leafHeight; ++ly) {
                            for (int lx = 0; lx < leafWidth; ++lx) {
                                const int voxelIdx = localVoxelIndex(lx, ly, lz);
                                float voxelMax = 0.0f;
                                for (int t = 0; t < frames; ++t) {
                                    const float v = raw.at(baseX + lx, baseY + ly, baseZ + lz, t);
                                    voxelSeries[static_cast<size_t>(t)] = v;
                                    voxelMax = std::max(voxelMax, v);
                                }

                                if (bgThreshold >= 0.0 && static_cast<double>(voxelMax) < bgThreshold) {
                                    localBgZero += 1;
                                    localTotalKeys += 1;
                                    localMaxKeys = std::max(localMaxKeys, 1);
                                    for (int t = 0; t < frames; ++t) {
                                        const float truth = voxelSeries[static_cast<size_t>(t)];
                                        const double diff = static_cast<double>(truth);
                                        localTemporalErr2 += diff * diff;
                                        leafTemporal[static_cast<size_t>(voxelIdx) * static_cast<size_t>(frames) + static_cast<size_t>(t)] = 0.0f;
                                    }
                                    continue;
                                }

                                bool usedCutoffProtect = false;
                                const auto keys = detectDensityKeyFrames(voxelSeries, temporalProfile, &usedCutoffProtect);
                                reconstructFromKeys(voxelSeries, keys, voxelRecon);
                                localTotalKeys += static_cast<uint64_t>(keys.size());
                                localMaxKeys = std::max(localMaxKeys, static_cast<int>(keys.size()));
                                if (usedCutoffProtect) localProtected += 1;

                                for (int t = 0; t < frames; ++t) {
                                    const float truth = voxelSeries[static_cast<size_t>(t)];
                                    const float pred = voxelRecon[static_cast<size_t>(t)];
                                    leafTemporal[static_cast<size_t>(voxelIdx) * static_cast<size_t>(frames) + static_cast<size_t>(t)] = pred;
                                    leafMax = std::max(leafMax, pred);
                                    const double diff = static_cast<double>(truth) - static_cast<double>(pred);
                                    localTemporalErr2 += diff * diff;
                                }
                            }
                        }
                    }

                    const bool emptyLeaf = (bgThreshold >= 0.0 && static_cast<double>(leafMax) < bgThreshold);
                    if (emptyLeaf) {
                        for (size_t vi = 0; vi < baseVariantCount; ++vi) {
                            if (!variantSpecs[vi].routeSelect) localEmptyLeaves[vi] += 1;
                        }
                    }

                    bool hasFine6 = false;
                    bool hasFine6Band = false;
                    bool hasFine6Gain = false;
                    bool hasShellVoxel = false;
                    // Mode2 (TemporalShellVoxel) layout reconstruction is in progress; the
                    // probe still hits a STATUS_STACK_BUFFER_OVERRUN somewhere in the sample
                    // loop after the shell-residual stream is written. Bypass keeps the rest
                    // of the pipeline (Empty/Grid4/Fine6) functioning while we finish the
                    // reverse-engineering of the saved-leaf byte layout.
                    constexpr bool kBypassShellMode = false;
                    bool hasTile3 = false;
                    uint64_t leafCoarseKf = 0;
                    uint64_t leafFine6Kf = 0;
                    uint64_t leafFine6BandKf = 0;
                    uint64_t leafFine6GainKf = 0;
                    uint64_t leafShellVoxelKf = 0;
                    std::array<uint64_t, 8> leafTile4Kf{};
                    BuiltControlStream coarsePackedStream;
                    coarsePackedStream.data.controlCount = static_cast<uint16_t>(kCoarseRes * kCoarseRes * kCoarseRes);
                    coarsePackedStream.data.timeBinCount = 8;
                    BuiltControlStream fine6PackedStream;
                    fine6PackedStream.data.controlCount = static_cast<uint16_t>(6 * 6 * 6);
                    fine6PackedStream.data.timeBinCount = 8;
                    BuiltControlStream shellVoxelPackedStream;
                    shellVoxelPackedStream.data.timeBinCount = kRenderTemporalShellPackedTimeBinCount;
                    RenderTemporalShellOccupancySection shellVoxelOccupancy{};
                    std::vector<bool> enableFine(variants.size(), false);
                    std::vector<bool> enableFineBand(variants.size(), false);
                    std::vector<bool> enableFineGain(variants.size(), false);
                    std::vector<bool> enableShellVoxel(variants.size(), false);
                    std::vector<std::array<bool, 8>> enableTile(variants.size());
                    const RenderTemporalVariantSpec* gainSpec = nullptr;

                    if (!emptyLeaf) {
                        for (int gz = 0; gz < kCoarseRes; ++gz) {
                            for (int gy = 0; gy < kCoarseRes; ++gy) {
                                for (int gx = 0; gx < kCoarseRes; ++gx) {
                                    const int ctrlIdx = (gz * kCoarseRes + gy) * kCoarseRes + gx;
                                    for (int t = 0; t < frames; ++t) {
                                        voxelSeries[static_cast<size_t>(t)] =
                                            sampleLeafBufferAtFractional(leafTemporal, frames, t, leafWidth, leafHeight, leafDepth,
                                                                         coarseCoords[static_cast<size_t>(gx)],
                                                                         coarseCoords[static_cast<size_t>(gy)],
                                                                         coarseCoords[static_cast<size_t>(gz)]);
                                    }
                                    const auto keys = detectDensityKeyFrames(voxelSeries, controlProfile, nullptr);
                                    leafCoarseKf += static_cast<uint64_t>(keys.size());
                                    appendControlSequenceToStream(coarsePackedStream.data,
                                                                  coarsePackedStream.requiredMaxBinLocalKeys,
                                                                  voxelSeries,
                                                                  keys,
                                                                  frames,
                                                                  8);
                                    reconstructFromKeys(voxelSeries, keys, voxelRecon);
                                    for (int t = 0; t < frames; ++t) {
                                        coarseGrid[static_cast<size_t>(ctrlIdx) * static_cast<size_t>(frames) + static_cast<size_t>(t)] =
                                            voxelRecon[static_cast<size_t>(t)];
                                    }
                                }
                            }
                        }

                        const double cutoffLo = static_cast<double>(cli.cutoff - cli.cutoffBand);
                        std::array<uint64_t, 8> tileVisibleCount{};
                        std::array<uint64_t, 8> tileShellCount{};
                        std::array<float, 8> tileShellPeak{};
                        std::array<uint64_t, 8> tileTotalCount{};
                        const double shellValueCut =
                            static_cast<double>(std::max(0.0f, raw.meta.dataMax * 0.35f));
                        for (int t = 0; t < frames; ++t) {
                            for (int lz = 0; lz < leafDepth; ++lz) {
                                for (int ly = 0; ly < leafHeight; ++ly) {
                                    for (int lx = 0; lx < leafWidth; ++lx) {
                                        const int tileIdx = (lz / 4) * 4 + (ly / 4) * 2 + (lx / 4);
                                        tileTotalCount[static_cast<size_t>(tileIdx)] += 1;
                                        const float truth =
                                            leafTemporalAt(leafTemporal, localVoxelIndex(lx, ly, lz), frames, t);
                                        if (static_cast<double>(truth) >= cutoffLo) {
                                            tileVisibleCount[static_cast<size_t>(tileIdx)] += 1;
                                        }
                                        if (static_cast<double>(truth) >= shellValueCut) {
                                            tileShellCount[static_cast<size_t>(tileIdx)] += 1;
                                        }
                                        tileShellPeak[static_cast<size_t>(tileIdx)] =
                                            std::max(tileShellPeak[static_cast<size_t>(tileIdx)], truth);
                                    }
                                }
                            }
                        }

                        for (size_t vi = 0; vi < baseVariantCount; ++vi) {
                            if (!variantSpecs[vi].routeSelect) {
                                localCoarseKf[vi] += leafCoarseKf;
                            }
                            if (variantSpecs[vi].tileLocal) {
                                bool anyTile = false;
                                for (int tileIdx = 0; tileIdx < 8; ++tileIdx) {
                                    const double frac = tileTotalCount[static_cast<size_t>(tileIdx)] > 0
                                        ? static_cast<double>(tileVisibleCount[static_cast<size_t>(tileIdx)]) / static_cast<double>(tileTotalCount[static_cast<size_t>(tileIdx)])
                                        : 0.0;
                                    const double shellFrac = tileTotalCount[static_cast<size_t>(tileIdx)] > 0
                                        ? static_cast<double>(tileShellCount[static_cast<size_t>(tileIdx)]) / static_cast<double>(tileTotalCount[static_cast<size_t>(tileIdx)])
                                        : 0.0;
                                    bool enable = false;
                                    if (variantSpecs[vi].shellAwareTile) {
                                        const double shellValueThr = static_cast<double>(
                                            variantSpecs[vi].shellValueThreshold > 0.0f
                                                ? variantSpecs[vi].shellValueThreshold
                                                : raw.meta.dataMax * 0.35f);
                                        enable = shellFrac >= static_cast<double>(variantSpecs[vi].shellFracThreshold) ||
                                                 static_cast<double>(tileShellPeak[static_cast<size_t>(tileIdx)]) >=
                                                     std::max(shellValueCut, shellValueThr) ||
                                                 frac >= std::max(0.01, static_cast<double>(variantSpecs[vi].visibleFracThreshold));
                                    } else {
                                        enable = frac >= static_cast<double>(variantSpecs[vi].visibleFracThreshold);
                                    }
                                    if (enable) {
                                        enableTile[vi][static_cast<size_t>(tileIdx)] = true;
                                        anyTile = true;
                                        hasTile3 = true;
                                    }
                                }
                                if (anyTile) localFineLeaves[vi] += 1;
                            } else if (!variantSpecs[vi].routeSelect && variants[vi].fineResolution == 6) {
                                if (variantSpecs[vi].coarseShellWeightedResidual) {
                                    enableFineGain[vi] = true;
                                    hasFine6Gain = true;
                                    if (gainSpec == nullptr) gainSpec = &variantSpecs[vi];
                                } else if (variantSpecs[vi].shellVoxelResidual) {
                                    if (!kBypassShellMode) {
                                        enableShellVoxel[vi] = true;
                                        hasShellVoxel = true;
                                    }
                                } else if (variantSpecs[vi].shellBandResidual) {
                                    enableFineBand[vi] = true;
                                    hasFine6Band = true;
                                } else {
                                    enableFine[vi] = true;
                                    hasFine6 = true;
                                }
                                localFineLeaves[vi] += 1;
                            }
                        }

                        if (hasFine6 || hasFine6Band || hasFine6Gain) {
                            for (int gz = 0; gz < 6; ++gz) {
                                for (int gy = 0; gy < 6; ++gy) {
                                    for (int gx = 0; gx < 6; ++gx) {
                                        const int ctrlIdx = (gz * 6 + gy) * 6 + gx;
                                        const float fx = fine6Coords[static_cast<size_t>(gx)];
                                        const float fy = fine6Coords[static_cast<size_t>(gy)];
                                        const float fz = fine6Coords[static_cast<size_t>(gz)];
                                        for (int t = 0; t < frames; ++t) {
                                            const float rawLike =
                                                sampleLeafBufferAtFractional(leafTemporal, frames, t, leafWidth, leafHeight, leafDepth, fx, fy, fz);
                                            const float coarse =
                                                sampleControlGridValue(coarseGrid, kCoarseRes, frames, t, leafWidth, leafHeight, leafDepth, fx, fy, fz);
                                            voxelSeries[static_cast<size_t>(t)] = rawLike - coarse;
                                        }
                                        if (hasFine6) {
                                            const auto keys = detectDensityKeyFrames(voxelSeries, controlProfile, nullptr);
                                            leafFine6Kf += static_cast<uint64_t>(keys.size());
                                            appendControlSequenceToStream(fine6PackedStream.data,
                                                                          fine6PackedStream.requiredMaxBinLocalKeys,
                                                                          voxelSeries,
                                                                          keys,
                                                                          frames,
                                                                          8);
                                            reconstructFromKeys(voxelSeries, keys, voxelRecon);
                                            for (int t = 0; t < frames; ++t) {
                                                fine6Grid[static_cast<size_t>(ctrlIdx) * static_cast<size_t>(frames) + static_cast<size_t>(t)] =
                                                    voxelRecon[static_cast<size_t>(t)];
                                            }
                                        }
                                        if (hasFine6Band) {
                                            const auto* bandSpec = [&]() -> const RenderTemporalVariantSpec* {
                                                for (size_t svi = 0; svi < baseVariantCount; ++svi) {
                                                    if (enableFineBand[svi]) return &variantSpecs[svi];
                                                }
                                                return nullptr;
                                            }();
                                            const float lowRatio = bandSpec != nullptr ? bandSpec->shellBandThreshold : 0.20f;
                                            const float highRatio = bandSpec != nullptr ? bandSpec->shellValueThreshold : 0.50f;
                                            const bool adaptiveBand = bandSpec != nullptr ? bandSpec->adaptiveShellBand : false;
                                            float shellBandLow = raw.meta.dataMax * lowRatio;
                                            float shellBandHigh = raw.meta.dataMax * highRatio;
                                            if (adaptiveBand) {
                                                shellBandLow = std::max(shellBandLow, leafMax * lowRatio);
                                                shellBandHigh = std::max(shellBandHigh, leafMax * highRatio);
                                            }
                                            std::vector<float> bandSeries(static_cast<size_t>(frames), 0.0f);
                                            for (int t = 0; t < frames; ++t) {
                                                const float rawLike =
                                                    sampleLeafBufferAtFractional(leafTemporal, frames, t, leafWidth, leafHeight, leafDepth, fx, fy, fz);
                                                float weight = 0.0f;
                                                if (rawLike >= shellBandHigh) {
                                                    weight = 1.0f;
                                                } else if (rawLike > shellBandLow && shellBandHigh > shellBandLow) {
                                                    weight = (rawLike - shellBandLow) / (shellBandHigh - shellBandLow);
                                                }
                                                bandSeries[static_cast<size_t>(t)] =
                                                    weight * voxelSeries[static_cast<size_t>(t)];
                                            }
                                            const auto bandKeys = detectDensityKeyFrames(bandSeries, controlProfile, nullptr);
                                            leafFine6BandKf += static_cast<uint64_t>(bandKeys.size());
                                            reconstructFromKeys(bandSeries, bandKeys, voxelRecon);
                                            for (int t = 0; t < frames; ++t) {
                                                fine6BandGrid[static_cast<size_t>(ctrlIdx) * static_cast<size_t>(frames) + static_cast<size_t>(t)] =
                                                    voxelRecon[static_cast<size_t>(t)];
                                            }
                                        }
                                        if (hasFine6Gain) {
                                            std::vector<float> gainSeries(static_cast<size_t>(frames), 0.0f);
                                            std::vector<float> residualSeries(static_cast<size_t>(frames), 0.0f);
                                            std::vector<float> guidanceSeries(static_cast<size_t>(frames), 0.0f);
                                            const float gainLow =
                                                gainSpec != nullptr ? gainSpec->shellWeightLow : 0.20f;
                                            const float gainHigh =
                                                gainSpec != nullptr ? gainSpec->shellWeightHigh : 0.60f;
                                            const float gainMax =
                                                gainSpec != nullptr ? gainSpec->shellMaxGain : 4.0f;
                                            for (int t = 0; t < frames; ++t) {
                                                const float rawLike =
                                                    sampleLeafBufferAtFractional(leafTemporal, frames, t, leafWidth, leafHeight, leafDepth, fx, fy, fz);
                                                const float coarse =
                                                    sampleControlGridValue(coarseGrid, kCoarseRes, frames, t, leafWidth, leafHeight, leafDepth, fx, fy, fz);
                                                residualSeries[static_cast<size_t>(t)] = rawLike - coarse;
                                                guidanceSeries[static_cast<size_t>(t)] = coarse;
                                            }
                                            if (gainSpec != nullptr && gainSpec->temporalSmoothGuidance) {
                                                smoothSeriesInPlace(guidanceSeries,
                                                                   gainSpec->temporalSmoothAlpha,
                                                                   gainSpec->temporalSmoothPasses);
                                            }
                                            for (int t = 0; t < frames; ++t) {
                                                const float gain = shellGainFromCoarse(
                                                    guidanceSeries[static_cast<size_t>(t)], raw.meta.dataMax, gainLow, gainHigh, gainMax);
                                                gainSeries[static_cast<size_t>(t)] =
                                                    residualSeries[static_cast<size_t>(t)] * gain;
                                            }
                                            if (gainSpec != nullptr && gainSpec->temporalSmoothResidual) {
                                                smoothSeriesInPlace(gainSeries,
                                                                   gainSpec->temporalSmoothAlpha,
                                                                   gainSpec->temporalSmoothPasses);
                                            }
                                            const auto gainKeys = detectDensityKeyFrames(gainSeries, controlProfile, nullptr);
                                            leafFine6GainKf += static_cast<uint64_t>(gainKeys.size());
                                            reconstructFromKeys(gainSeries, gainKeys, voxelRecon);
                                            for (int t = 0; t < frames; ++t) {
                                                fine6GainGrid[static_cast<size_t>(ctrlIdx) * static_cast<size_t>(frames) + static_cast<size_t>(t)] =
                                                    voxelRecon[static_cast<size_t>(t)];
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if (hasShellVoxel) {
                            std::fill(shellVoxelGrid.begin(), shellVoxelGrid.end(), 0.0f);
                            shellVoxelOccupancy = {};
                            shellVoxelPackedStream = {};
                            shellVoxelPackedStream.data.timeBinCount = kRenderTemporalShellPackedTimeBinCount;
                            const float shellVoxelCut = raw.meta.dataMax * 0.35f;
                            uint16_t activeVoxelCount = 0;
                            for (int group = 0; group < 8; ++group) {
                                shellVoxelOccupancy.groupPrefix[static_cast<size_t>(group)] = activeVoxelCount;
                                uint64_t word = 0ull;
                                for (int bit = 0; bit < 64; ++bit) {
                                    const int voxelIdx = group * 64 + bit;
                                    const int lx = voxelIdx % 8;
                                    const int ly = (voxelIdx / 8) % 8;
                                    const int lz = voxelIdx / 64;
                                    if (lx >= leafWidth || ly >= leafHeight || lz >= leafDepth) {
                                        continue;
                                    }
                                    bool active = false;
                                    for (int t = 0; t < frames; ++t) {
                                        const float truth = leafTemporalAt(leafTemporal, voxelIdx, frames, t);
                                        if (cli.levelSetSurfaceMode) {
                                            const float coarse = sampleControlGridValue(
                                                coarseGrid,
                                                kCoarseRes,
                                                frames,
                                                t,
                                                leafWidth,
                                                leafHeight,
                                                leafDepth,
                                                static_cast<float>(lx),
                                                static_cast<float>(ly),
                                                static_cast<float>(lz));
                                            const bool signMismatch = (truth >= 0.0f) != (coarse >= 0.0f);
                                            if (std::abs(truth) <= cli.levelSetSurfaceBand ||
                                                std::abs(coarse) <= cli.levelSetCoarseGuardBand ||
                                                signMismatch) {
                                                active = true;
                                                break;
                                            }
                                        } else if (truth >= shellVoxelCut) {
                                            active = true;
                                            break;
                                        }
                                    }
                                    if (!active) continue;
                                    word |= (1ull << bit);
                                    for (int t = 0; t < frames; ++t) {
                                        const float truth = leafTemporalAt(leafTemporal, voxelIdx, frames, t);
                                        const float coarse =
                                            sampleControlGridValue(coarseGrid, kCoarseRes, frames, t, leafWidth, leafHeight, leafDepth,
                                                                   static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                                        voxelSeries[static_cast<size_t>(t)] =
                                            truth - coarse;
                                        if (cli.levelSetSurfaceMode) {
                                            levelSetTruthSeries[static_cast<size_t>(t)] = truth;
                                        }
                                    }
                                    auto keys = detectDensityKeyFrames(voxelSeries, controlProfile, nullptr);
                                    if (cli.levelSetSurfaceMode) {
                                        addIsoSurfaceKeyFrames(
                                            levelSetTruthSeries,
                                            0.0f,
                                            cli.levelSetTemporalBand,
                                            keys);
                                    }
                                    leafShellVoxelKf += static_cast<uint64_t>(keys.size());
                                    reconstructFromKeys(voxelSeries, keys, voxelRecon);
                                    for (int t = 0; t < frames; ++t) {
                                        shellVoxelGrid[static_cast<size_t>(voxelIdx) * static_cast<size_t>(frames) + static_cast<size_t>(t)] =
                                            voxelRecon[static_cast<size_t>(t)];
                                    }
                                    appendControlSequenceToStream(shellVoxelPackedStream.data,
                                                                  shellVoxelPackedStream.requiredMaxBinLocalKeys,
                                                                  voxelRecon,
                                                                  keys,
                                                                  frames,
                                                                  kRenderTemporalShellPackedTimeBinCount);
                                    activeVoxelCount += 1;
                                }
                                shellVoxelOccupancy.shellMask[static_cast<size_t>(group)] = word;
                            }
                            shellVoxelPackedStream.data.controlCount = activeVoxelCount;
                            shellVoxelPackedStream.data.maxBinLocalKeys = shellVoxelPackedStream.requiredMaxBinLocalKeys;
                            if (activeVoxelCount == 0) {
                                hasShellVoxel = false;
                            }
                        }

                        if (hasTile3) {
                            std::fill(tile3Residual.begin(), tile3Residual.end(), 0.0f);
                            for (int tileZ = 0; tileZ < 2; ++tileZ) {
                                for (int tileY = 0; tileY < 2; ++tileY) {
                                    for (int tileX = 0; tileX < 2; ++tileX) {
                                        const int tileIdx = tileZ * 4 + tileY * 2 + tileX;
                                        bool needTile = false;
                                        for (size_t vi = 0; vi < variants.size(); ++vi) {
                                            if (variantSpecs[vi].tileLocal && enableTile[vi][static_cast<size_t>(tileIdx)]) {
                                                needTile = true;
                                                break;
                                            }
                                        }
                                        if (!needTile) continue;

                                        const int localMaxX = std::min(4, leafWidth - tileX * 4);
                                        const int localMaxY = std::min(4, leafHeight - tileY * 4);
                                        const int localMaxZ = std::min(4, leafDepth - tileZ * 4);
                                        for (int gz = 0; gz < 3; ++gz) {
                                            for (int gy = 0; gy < 3; ++gy) {
                                                for (int gx = 0; gx < 3; ++gx) {
                                                    const int localIdx = (gz * 3 + gy) * 3 + gx;
                                                    const float tx = tile3Coords[static_cast<size_t>(gx)];
                                                    const float ty = tile3Coords[static_cast<size_t>(gy)];
                                                    const float tz = tile3Coords[static_cast<size_t>(gz)];
                                                    const int lx = tileX * 4;
                                                    const int ly = tileY * 4;
                                                    const int lz = tileZ * 4;
                                                    for (int t = 0; t < frames; ++t) {
                                                        const float rawLike =
                                                            sampleLeafBufferAtFractional(leafTemporal, frames, t, leafWidth, leafHeight, leafDepth,
                                                                                         static_cast<float>(lx) + tx,
                                                                                         static_cast<float>(ly) + ty,
                                                                                         static_cast<float>(lz) + tz);
                                                        const float coarse =
                                                            sampleControlGridValue(coarseGrid, kCoarseRes, frames, t, leafWidth, leafHeight, leafDepth,
                                                                                   static_cast<float>(lx) + tx,
                                                                                   static_cast<float>(ly) + ty,
                                                                                   static_cast<float>(lz) + tz);
                                                        voxelSeries[static_cast<size_t>(t)] = rawLike - coarse;
                                                    }
                                                    const auto keys = detectDensityKeyFrames(voxelSeries, controlProfile, nullptr);
                                                    leafTile4Kf[static_cast<size_t>(tileIdx)] += static_cast<uint64_t>(keys.size());
                                                    reconstructFromKeys(voxelSeries, keys, voxelRecon);
                                                    for (int t = 0; t < frames; ++t) {
                                                        tile3Residual[(static_cast<size_t>(tileIdx) * 27ull + static_cast<size_t>(localIdx)) * static_cast<size_t>(frames) + static_cast<size_t>(t)] =
                                                            voxelRecon[static_cast<size_t>(t)];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        for (size_t vi = 0; vi < baseVariantCount; ++vi) {
                            if (enableFine[vi]) {
                                localFineKf[vi] += leafFine6Kf;
                            } else if (enableFineBand[vi]) {
                                localFineKf[vi] += leafFine6BandKf;
                            } else if (enableFineGain[vi]) {
                                localFineKf[vi] += leafFine6GainKf;
                            } else if (enableShellVoxel[vi]) {
                                localFineKf[vi] += leafShellVoxelKf;
                            } else if (variantSpecs[vi].tileLocal) {
                                uint64_t tileSum = 0;
                                for (int tileIdx = 0; tileIdx < 8; ++tileIdx) {
                                    if (enableTile[vi][static_cast<size_t>(tileIdx)]) tileSum += leafTile4Kf[static_cast<size_t>(tileIdx)];
                                }
                                localFineKf[vi] += tileSum;
                            }
                        }
                    }

                    samplePoints.clear();
                    for (int t = 0; t < raw.meta.frames; t += cli.sampleStep) {
                        for (int lz = 0; lz < leafDepth; ++lz) {
                            const int z = baseZ + lz;
                            if ((z % cli.sampleStep) != 0) continue;
                            for (int ly = 0; ly < leafHeight; ++ly) {
                                const int y = baseY + ly;
                                if ((y % cli.sampleStep) != 0) continue;
                                for (int lx = 0; lx < leafWidth; ++lx) {
                                    const int x = baseX + lx;
                                    if ((x % cli.sampleStep) != 0) continue;
                                    const float truth = raw.at(x, y, z, t);
                                    float coarsePred = 0.0f;
                                    float fine6Pred = 0.0f;
                                    float fine6BandPred = 0.0f;
                                    float fine6GainPred = 0.0f;
                                    float shellVoxelPred = 0.0f;
                                    const int tileIdx = (lz / 4) * 4 + (ly / 4) * 2 + (lx / 4);
                                    if (!emptyLeaf) {
                                        coarsePred = sampleControlGridValue(coarseGrid, kCoarseRes, frames, t, leafWidth, leafHeight, leafDepth,
                                                                            static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                                        if (hasFine6) {
                                            fine6Pred = sampleControlGridValue(fine6Grid, 6, frames, t, leafWidth, leafHeight, leafDepth,
                                                                               static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                                        }
                                        if (hasFine6Band) {
                                            fine6BandPred = sampleControlGridValue(fine6BandGrid, 6, frames, t, leafWidth, leafHeight, leafDepth,
                                                                                   static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                                        }
                                        if (hasFine6Gain) {
                                            const float gainEncoded =
                                                sampleControlGridValue(fine6GainGrid, 6, frames, t, leafWidth, leafHeight, leafDepth,
                                                                       static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                                            const float gainLow =
                                                gainSpec != nullptr ? gainSpec->shellWeightLow : 0.20f;
                                            const float gainHigh =
                                                gainSpec != nullptr ? gainSpec->shellWeightHigh : 0.60f;
                                            const float gainMax =
                                                gainSpec != nullptr ? gainSpec->shellMaxGain : 4.0f;
                                            float gainGuide = coarsePred;
                                            if (gainSpec != nullptr && gainSpec->temporalSmoothGuidance) {
                                                const int prevT = std::max(0, t - 1);
                                                const int nextT = std::min(frames - 1, t + 1);
                                                const float prevCoarse =
                                                    sampleControlGridValue(coarseGrid, kCoarseRes, frames, prevT, leafWidth, leafHeight, leafDepth,
                                                                           static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                                                const float nextCoarse =
                                                    sampleControlGridValue(coarseGrid, kCoarseRes, frames, nextT, leafWidth, leafHeight, leafDepth,
                                                                           static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                                                gainGuide = smoothGuidanceValue(prevCoarse,
                                                                               coarsePred,
                                                                               nextCoarse,
                                                                               gainSpec->temporalSmoothAlpha);
                                            }
                                            const float gain =
                                                shellGainFromCoarse(gainGuide, raw.meta.dataMax, gainLow, gainHigh, gainMax);
                                            fine6GainPred = gainEncoded / std::max(1e-6f, gain);
                                        }
                                        if (hasShellVoxel) {
                                            const int voxelIdx = localVoxelIndex(lx, ly, lz);
                                            shellVoxelPred =
                                                shellVoxelGrid[static_cast<size_t>(voxelIdx) * static_cast<size_t>(frames) + static_cast<size_t>(t)];
                                        }
                                    }
                                    samplePoints.push_back({lx, ly, lz, t, truth, coarsePred, fine6Pred, fine6BandPred, fine6GainPred, shellVoxelPred});
                                }
                            }
                        }
                    }

                    RenderTemporalRouteMetrics routedMetrics{};
                    if (!samplePoints.empty()) {
                        double coarseErr2 = 0.0;
                        double fineErr2 = 0.0;
                        double coarseShellErr2 = 0.0;
                        double shellVoxelErr2 = 0.0;
                        double coarsePeak = 0.0;
                        double finePeak = 0.0;
                        double coarseShellPeak = 0.0;
                        double shellVoxelPeak = 0.0;
                        uint64_t visibleCount = 0;
                        uint64_t bandCount = 0;
                        uint64_t shellCount = 0;
                        const double cutoffLo = static_cast<double>(cli.cutoff - cli.cutoffBand);
                        const double cutoffHi = static_cast<double>(cli.cutoff + cli.cutoffBand);
                        const double shellCut = static_cast<double>(raw.meta.dataMax) * 0.35;
                        for (const auto& sp : samplePoints) {
                            const double truth = static_cast<double>(sp.truth);
                            if (truth >= cutoffLo) visibleCount += 1;
                            if (truth >= cutoffLo && truth <= cutoffHi) bandCount += 1;

                            const double coarseDiff = truth - static_cast<double>(sp.coarsePred);
                            const double fineDiff = truth - static_cast<double>(sp.coarsePred + sp.fine6Pred);
                            coarseErr2 += coarseDiff * coarseDiff;
                            fineErr2 += fineDiff * fineDiff;
                            coarsePeak = std::max(coarsePeak, std::abs(coarseDiff) / peak);
                            finePeak = std::max(finePeak, std::abs(fineDiff) / peak);
                            if (truth >= shellCut) {
                                shellCount += 1;
                                const double shellDiff = truth - static_cast<double>(sp.coarsePred + sp.shellVoxelPred);
                                coarseShellErr2 += coarseDiff * coarseDiff;
                                shellVoxelErr2 += shellDiff * shellDiff;
                                coarseShellPeak = std::max(coarseShellPeak, std::abs(coarseDiff) / peak);
                                shellVoxelPeak = std::max(shellVoxelPeak, std::abs(shellDiff) / peak);
                            }
                        }
                        const double sampleCount = static_cast<double>(samplePoints.size());
                        routedMetrics.visibleFrac = static_cast<double>(visibleCount) / sampleCount;
                        routedMetrics.bandFrac = static_cast<double>(bandCount) / sampleCount;
                        routedMetrics.shellFrac = static_cast<double>(shellCount) / sampleCount;
                        routedMetrics.coarseRmse = std::sqrt(coarseErr2 / sampleCount) / peak;
                        routedMetrics.fine6Rmse = std::sqrt(fineErr2 / sampleCount) / peak;
                        routedMetrics.coarsePeak = coarsePeak;
                        routedMetrics.fine6Peak = finePeak;
                        if (shellCount > 0) {
                            const double shellSampleCount = static_cast<double>(shellCount);
                            routedMetrics.coarseShellRmse = std::sqrt(coarseShellErr2 / shellSampleCount) / peak;
                            routedMetrics.shellVoxelRmse = std::sqrt(shellVoxelErr2 / shellSampleCount) / peak;
                            routedMetrics.coarseShellPeak = coarseShellPeak;
                            routedMetrics.shellVoxelPeak = shellVoxelPeak;
                        }
                    }
                    const auto routedMode =
                        cli.levelSetSurfaceMode
                            ? (hasShellVoxel
                                   ? RenderTemporalFormalMode::FINE_COMPACT
                                   : RenderTemporalFormalMode::COARSE_ONLY)
                            : routeRenderTemporalLeaf(emptyLeaf, routedMetrics, routeOptions);

                    std::vector<uint8_t> packedFine6LeafBytes;
                    std::vector<uint8_t> packedShellVoxelLeafBytes;
                    std::vector<uint8_t> packedRoutedLeafBytes;
                    RenderTemporalPackedLeafView packedFine6LeafView{};
                    RenderTemporalPackedLeafView packedShellVoxelLeafView{};
                    RenderTemporalPackedLeafView packedRoutedLeafView{};
                    bool hasPackedFine6Leaf = false;
                    bool hasPackedShellVoxelLeaf = false;
                    bool hasPackedRoutedLeaf = false;

                    if (emptyLeaf) {
                        RenderTemporalPackedHeaderFields emptyHeader;
                        emptyHeader.mode = TemporalFirstPackedMode::EMPTY;
                        packedFine6LeafBytes =
                            buildRenderTemporalPackedLeafBytes(emptyHeader, {}, {}, {}, true);
                        packedRoutedLeafBytes = packedFine6LeafBytes;
                        packedFine6LeafView = parseRenderTemporalPackedLeaf(packedFine6LeafBytes.data(),
                                                                            packedFine6LeafBytes.size(),
                                                                            true);
                        packedShellVoxelLeafBytes = packedFine6LeafBytes;
                        packedShellVoxelLeafView = packedFine6LeafView;
                        packedRoutedLeafView = packedFine6LeafView;
                        hasPackedFine6Leaf = true;
                        hasPackedShellVoxelLeaf = true;
                        hasPackedRoutedLeaf = true;
                        localEmptyLeaves[packedFine6VariantIndex] += 1;
                        localEmptyLeaves[packedShellVoxelVariantIndex] += 1;
                        localEmptyLeaves[packedRoutedVariantIndex] += 1;
                        localPackedBytes[packedFine6VariantIndex] += static_cast<uint64_t>(packedFine6LeafBytes.size());
                        localPackedBytes[packedShellVoxelVariantIndex] += static_cast<uint64_t>(packedShellVoxelLeafBytes.size());
                        localPackedBytes[packedRoutedVariantIndex] += static_cast<uint64_t>(packedRoutedLeafBytes.size());
                    } else {
                        auto coarsePackedData = coarsePackedStream.data;
                        coarsePackedData.codec = cli.packedCodec;
                        coarsePackedData.maxBinLocalKeys = coarsePackedStream.requiredMaxBinLocalKeys;
                        const auto coarseBytes = packRenderTemporalControlStream(coarsePackedData, true);

                        std::vector<uint8_t> fineBytes;
                        auto finePackedData = fine6PackedStream.data;
                        if (hasFine6) {
                            finePackedData.codec = cli.packedCodec;
                            finePackedData.maxBinLocalKeys = fine6PackedStream.requiredMaxBinLocalKeys;
                            fineBytes = packRenderTemporalControlStream(finePackedData, true);
                        }
                        std::vector<uint8_t> shellOccupancyBytes;
                        std::vector<uint8_t> shellResidualBytes;
                        auto shellPackedData = shellVoxelPackedStream.data;
                        const bool shellPayloadFits =
                            coarsePackedStream.data.keyframes.size() <= std::numeric_limits<uint16_t>::max() &&
                            shellPackedData.keyframes.size() <= std::numeric_limits<uint16_t>::max();
                        if (hasShellVoxel && shellPackedData.controlCount > 0 && shellPayloadFits) {
                            shellPackedData.codec = cli.packedCodec;
                            shellPackedData.maxBinLocalKeys = shellVoxelPackedStream.requiredMaxBinLocalKeys;
                            shellOccupancyBytes = packRenderTemporalShellOccupancySection(shellVoxelOccupancy);
                            shellResidualBytes = packRenderTemporalControlStream(shellPackedData, true);
                        }

                        {
                            RenderTemporalPackedHeaderFields header;
                            header.mode = hasFine6 ? TemporalFirstPackedMode::TEMPORAL_FINE6
                                                   : TemporalFirstPackedMode::TEMPORAL_GRID4;
                            header.coarseCodec = coarsePackedData.codec;
                            header.fineCodec = hasFine6 ? finePackedData.codec : RenderTemporalSequenceCodec::DP_KEYFRAME_FP16;
                            header.coarseResolution = static_cast<uint8_t>(kCoarseRes);
                            header.fineResolution = hasFine6 ? 6u : 0u;
                            auto prefix = makeRenderTemporalLeafPrefix(header.mode,
                                                                       static_cast<uint32_t>(coarsePackedStream.data.keyframes.size()),
                                                                       static_cast<uint32_t>(hasFine6 ? fine6PackedStream.data.keyframes.size() : 0));
                            packedFine6LeafBytes =
                                buildRenderTemporalPackedLeafBytes(header, prefix, coarseBytes, fineBytes, true);
                            packedFine6LeafView =
                                parseRenderTemporalPackedLeaf(packedFine6LeafBytes.data(), packedFine6LeafBytes.size(), true);
                            hasPackedFine6Leaf = true;
                            localPackedBytes[packedFine6VariantIndex] += static_cast<uint64_t>(packedFine6LeafBytes.size());
                            localCoarseKf[packedFine6VariantIndex] += leafCoarseKf;
                            localFineKf[packedFine6VariantIndex] += leafFine6Kf;
                            localFineLeaves[packedFine6VariantIndex] += 1;
                        }

                        {
                            RenderTemporalPackedHeaderFields header;
                            const bool useShellMode =
                                hasShellVoxel && shellPackedData.controlCount > 0 && shellPayloadFits;
                            header.mode = useShellMode ? TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT
                                                       : TemporalFirstPackedMode::TEMPORAL_GRID4;
                            header.coarseCodec = coarsePackedData.codec;
                            header.fineCodec = useShellMode ? shellPackedData.codec : RenderTemporalSequenceCodec::DP_KEYFRAME_FP16;
                            header.coarseResolution = static_cast<uint8_t>(kCoarseRes);
                            header.fineResolution = 0u;
                            auto prefix = makeRenderTemporalLeafPrefix(header.mode,
                                                                       static_cast<uint32_t>(coarsePackedStream.data.keyframes.size()),
                                                                       static_cast<uint32_t>(useShellMode ? shellPackedData.keyframes.size() : 0));
                            if (useShellMode) {
                                packedShellVoxelLeafBytes =
                                    buildRenderTemporalPackedLeafBytes(header, prefix, coarseBytes, shellOccupancyBytes, shellResidualBytes, true);
                            } else {
                                packedShellVoxelLeafBytes =
                                    buildRenderTemporalPackedLeafBytes(header, prefix, coarseBytes, {}, true);
                            }
                            packedShellVoxelLeafView =
                                parseRenderTemporalPackedLeaf(packedShellVoxelLeafBytes.data(), packedShellVoxelLeafBytes.size(), true);
                            hasPackedShellVoxelLeaf = true;
                            localPackedBytes[packedShellVoxelVariantIndex] += static_cast<uint64_t>(packedShellVoxelLeafBytes.size());
                            localCoarseKf[packedShellVoxelVariantIndex] += leafCoarseKf;
                            if (useShellMode) {
                                localFineKf[packedShellVoxelVariantIndex] += leafShellVoxelKf;
                                localFineLeaves[packedShellVoxelVariantIndex] += 1;
                            }
                        }

                        {
                            RenderTemporalPackedHeaderFields header;
                            header.coarseCodec = coarsePackedData.codec;
                            header.fineCodec = hasFine6 ? finePackedData.codec : RenderTemporalSequenceCodec::DP_KEYFRAME_FP16;
                            header.coarseResolution = static_cast<uint8_t>(kCoarseRes);
                            if (routedMode == RenderTemporalFormalMode::EMPTY) {
                                header.mode = TemporalFirstPackedMode::EMPTY;
                                packedRoutedLeafBytes =
                                    buildRenderTemporalPackedLeafBytes(header, {}, {}, {}, true);
                            } else if (routedMode == RenderTemporalFormalMode::FINE_COMPACT) {
                                if (!shellPayloadFits) {
                                    throw std::runtime_error(
                                        "Mode2 keyframe count exceeds VBTPACK4 uint16 capacity: coarse=" +
                                        std::to_string(coarsePackedStream.data.keyframes.size()) +
                                        ", surface=" + std::to_string(shellPackedData.keyframes.size()) +
                                        ", limit=" +
                                        std::to_string(std::numeric_limits<uint16_t>::max()));
                                }
                                const bool useShellMode = hasShellVoxel && shellPackedData.controlCount > 0;
                                header.mode = useShellMode ? TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT
                                                           : TemporalFirstPackedMode::TEMPORAL_GRID4;
                                header.fineResolution = 0u;
                                auto prefix = makeRenderTemporalLeafPrefix(header.mode,
                                                                           static_cast<uint32_t>(coarsePackedStream.data.keyframes.size()),
                                                                           static_cast<uint32_t>(useShellMode ? shellPackedData.keyframes.size() : 0));
                                if (useShellMode) {
                                    packedRoutedLeafBytes =
                                        buildRenderTemporalPackedLeafBytes(header, prefix, coarseBytes, shellOccupancyBytes, shellResidualBytes, true);
                                } else {
                                    packedRoutedLeafBytes =
                                        buildRenderTemporalPackedLeafBytes(header, prefix, coarseBytes, {}, true);
                                }
                            } else if (routedMode == RenderTemporalFormalMode::FINE6_FULL) {
                                header.mode = TemporalFirstPackedMode::TEMPORAL_FINE6;
                                header.fineResolution = 6u;
                                auto prefix = makeRenderTemporalLeafPrefix(header.mode,
                                                                           static_cast<uint32_t>(coarsePackedStream.data.keyframes.size()),
                                                                           static_cast<uint32_t>(fine6PackedStream.data.keyframes.size()));
                                packedRoutedLeafBytes =
                                    buildRenderTemporalPackedLeafBytes(header, prefix, coarseBytes, fineBytes, true);
                            } else {
                                header.mode = TemporalFirstPackedMode::TEMPORAL_GRID4;
                                header.fineResolution = 0u;
                                auto prefix = makeRenderTemporalLeafPrefix(header.mode,
                                                                           static_cast<uint32_t>(coarsePackedStream.data.keyframes.size()),
                                                                           0);
                                packedRoutedLeafBytes =
                                    buildRenderTemporalPackedLeafBytes(header, prefix, coarseBytes, {}, true);
                            }
                            packedRoutedLeafView =
                                parseRenderTemporalPackedLeaf(packedRoutedLeafBytes.data(), packedRoutedLeafBytes.size(), true);
                            hasPackedRoutedLeaf = true;
                            localPackedBytes[packedRoutedVariantIndex] += static_cast<uint64_t>(packedRoutedLeafBytes.size());
                            if (routedMode == RenderTemporalFormalMode::EMPTY) {
                                localEmptyLeaves[packedRoutedVariantIndex] += 1;
                            } else {
                                localCoarseKf[packedRoutedVariantIndex] += leafCoarseKf;
                                if (routedMode == RenderTemporalFormalMode::FINE6_FULL) {
                                    localFineKf[packedRoutedVariantIndex] += leafFine6Kf;
                                    localFineLeaves[packedRoutedVariantIndex] += 1;
                                } else if (routedMode == RenderTemporalFormalMode::FINE_COMPACT) {
                                    localFineKf[packedRoutedVariantIndex] += leafShellVoxelKf;
                                    localFineLeaves[packedRoutedVariantIndex] += 1;
                                }
                            }
                        }
                    }

                    for (size_t vi = 0; vi < baseVariantCount; ++vi) {
                        if (variantSpecs[vi].routeSelect) {
                            if (routedMode == RenderTemporalFormalMode::EMPTY) {
                                localEmptyLeaves[vi] += 1;
                            } else {
                                localCoarseKf[vi] += leafCoarseKf;
                                if (routedMode == RenderTemporalFormalMode::FINE6_FULL) {
                                    localFineLeaves[vi] += 1;
                                    localFineKf[vi] += leafFine6Kf;
                                } else if (routedMode == RenderTemporalFormalMode::FINE_COMPACT) {
                                    localFineLeaves[vi] += 1;
                                    localFineKf[vi] += leafShellVoxelKf;
                                }
                            }
                            }
                        }

                        if (!cli.saveVbtPath.empty()) {
                            if (cli.preferShellVoxelSave) {
                                savedLeafPayloads[static_cast<size_t>(leafLinearIndex)] = packedShellVoxelLeafBytes;
                            } else {
                                savedLeafPayloads[static_cast<size_t>(leafLinearIndex)] = packedRoutedLeafBytes;
                            }
                        }

                        if (exportFrameEnabled) {
                        const bool exportIsPackedFine6 =
                            static_cast<size_t>(exportVariantIndex) == packedFine6VariantIndex;
                        const bool exportIsPackedShellVoxel =
                            static_cast<size_t>(exportVariantIndex) == packedShellVoxelVariantIndex;
                        const bool exportIsPackedRouted =
                            static_cast<size_t>(exportVariantIndex) == packedRoutedVariantIndex;
                        const auto* exportSpec =
                            (static_cast<size_t>(exportVariantIndex) < baseVariantCount)
                                ? &variantSpecs[static_cast<size_t>(exportVariantIndex)]
                                : nullptr;
                        for (int lz = 0; lz < leafDepth; ++lz) {
                            const int z = baseZ + lz;
                            for (int ly = 0; ly < leafHeight; ++ly) {
                                const int y = baseY + ly;
                                for (int lx = 0; lx < leafWidth; ++lx) {
                                    const int x = baseX + lx;
                                    const int tileIdx = (lz / 4) * 4 + (ly / 4) * 2 + (lx / 4);
                                    float pred = 0.0f;
                                    if (exportIsPackedFine6) {
                                        if (hasPackedFine6Leaf && packedFine6LeafView.header.mode != TemporalFirstPackedMode::EMPTY) {
                                            pred = samplePackedLeafGridValue(packedFine6LeafBytes,
                                                                             packedFine6LeafView,
                                                                             false,
                                                                             kCoarseRes,
                                                                             frames,
                                                                             leafWidth,
                                                                             leafHeight,
                                                                             leafDepth,
                                                                             cli.exportFrameIndex,
                                                                             static_cast<float>(lx),
                                                                             static_cast<float>(ly),
                                                                             static_cast<float>(lz));
                                            if (packedFine6LeafView.header.mode == TemporalFirstPackedMode::TEMPORAL_FINE6) {
                                                pred += samplePackedLeafGridValue(packedFine6LeafBytes,
                                                                                  packedFine6LeafView,
                                                                                  true,
                                                                                  6,
                                                                                  frames,
                                                                                  leafWidth,
                                                                                  leafHeight,
                                                                                  leafDepth,
                                                                                  cli.exportFrameIndex,
                                                                                  static_cast<float>(lx),
                                                                                  static_cast<float>(ly),
                                                                                  static_cast<float>(lz));
                                            }
                                        }
                                    } else if (exportIsPackedShellVoxel) {
                                        if (hasPackedShellVoxelLeaf && packedShellVoxelLeafView.header.mode != TemporalFirstPackedMode::EMPTY) {
                                            pred = samplePackedLeafGridValue(packedShellVoxelLeafBytes,
                                                                             packedShellVoxelLeafView,
                                                                             false,
                                                                             kCoarseRes,
                                                                             frames,
                                                                             leafWidth,
                                                                             leafHeight,
                                                                             leafDepth,
                                                                             cli.exportFrameIndex,
                                                                             static_cast<float>(lx),
                                                                             static_cast<float>(ly),
                                                                             static_cast<float>(lz));
                                            if (packedShellVoxelLeafView.header.mode == TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT) {
                                                pred += decodeRenderTemporalLeafShellVoxelValue(
                                                    packedShellVoxelLeafBytes.data(),
                                                    packedShellVoxelLeafView,
                                                    true,
                                                    frames,
                                                    static_cast<uint16_t>(localVoxelIndex(lx, ly, lz)),
                                                    cli.exportFrameIndex);
                                            }
                                        }
                                    } else if (exportIsPackedRouted) {
                                        if (hasPackedRoutedLeaf && packedRoutedLeafView.header.mode != TemporalFirstPackedMode::EMPTY) {
                                            pred = samplePackedLeafGridValue(packedRoutedLeafBytes,
                                                                             packedRoutedLeafView,
                                                                             false,
                                                                             kCoarseRes,
                                                                             frames,
                                                                             leafWidth,
                                                                             leafHeight,
                                                                             leafDepth,
                                                                             cli.exportFrameIndex,
                                                                             static_cast<float>(lx),
                                                                             static_cast<float>(ly),
                                                                             static_cast<float>(lz));
                                            if (packedRoutedLeafView.header.mode == TemporalFirstPackedMode::TEMPORAL_FINE6) {
                                                pred += samplePackedLeafGridValue(packedRoutedLeafBytes,
                                                                                  packedRoutedLeafView,
                                                                                  true,
                                                                                  6,
                                                                                  frames,
                                                                                  leafWidth,
                                                                                  leafHeight,
                                                                                  leafDepth,
                                                                                  cli.exportFrameIndex,
                                                                                  static_cast<float>(lx),
                                                                                  static_cast<float>(ly),
                                                                                  static_cast<float>(lz));
                                            } else if (packedRoutedLeafView.header.mode == TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT) {
                                                pred += decodeRenderTemporalLeafShellVoxelValue(
                                                    packedRoutedLeafBytes.data(),
                                                    packedRoutedLeafView,
                                                    true,
                                                    frames,
                                                    static_cast<uint16_t>(localVoxelIndex(lx, ly, lz)),
                                                    cli.exportFrameIndex);
                                            }
                                        }
                                    } else if (exportSpec != nullptr && exportSpec->routeSelect) {
                                        if (routedMode != RenderTemporalFormalMode::EMPTY) {
                                            pred = sampleControlGridValue(coarseGrid, kCoarseRes, frames, cli.exportFrameIndex, leafWidth, leafHeight, leafDepth,
                                                                          static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                                            if (routedMode == RenderTemporalFormalMode::FINE6_FULL) {
                                                pred += sampleControlGridValue(fine6Grid, 6, frames, cli.exportFrameIndex, leafWidth, leafHeight, leafDepth,
                                                                               static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                                            } else if (routedMode == RenderTemporalFormalMode::FINE_COMPACT) {
                                                const int voxelIdx = localVoxelIndex(lx, ly, lz);
                                                pred += shellVoxelGrid[static_cast<size_t>(voxelIdx) * static_cast<size_t>(frames) +
                                                                       static_cast<size_t>(cli.exportFrameIndex)];
                                            }
                                        }
                                    } else if (exportSpec != nullptr && !emptyLeaf) {
                                        pred = sampleControlGridValue(coarseGrid, kCoarseRes, frames, cli.exportFrameIndex, leafWidth, leafHeight, leafDepth,
                                                                      static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                                        if (enableFine[static_cast<size_t>(exportVariantIndex)]) {
                                            pred += sampleControlGridValue(fine6Grid, 6, frames, cli.exportFrameIndex, leafWidth, leafHeight, leafDepth,
                                                                           static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                                        } else if (enableFineGain[static_cast<size_t>(exportVariantIndex)]) {
                                            const float gainEncoded =
                                                sampleControlGridValue(fine6GainGrid, 6, frames, cli.exportFrameIndex, leafWidth, leafHeight, leafDepth,
                                                                       static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                                            const float gainLow =
                                                gainSpec != nullptr ? gainSpec->shellWeightLow : 0.20f;
                                            const float gainHigh =
                                                gainSpec != nullptr ? gainSpec->shellWeightHigh : 0.60f;
                                            const float gainMax =
                                                gainSpec != nullptr ? gainSpec->shellMaxGain : 4.0f;
                                            float gainGuide = pred;
                                            if (gainSpec != nullptr && gainSpec->temporalSmoothGuidance) {
                                                const int prevT = std::max(0, cli.exportFrameIndex - 1);
                                                const int nextT = std::min(frames - 1, cli.exportFrameIndex + 1);
                                                const float prevCoarse =
                                                    sampleControlGridValue(coarseGrid, kCoarseRes, frames, prevT, leafWidth, leafHeight, leafDepth,
                                                                           static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                                                const float nextCoarse =
                                                    sampleControlGridValue(coarseGrid, kCoarseRes, frames, nextT, leafWidth, leafHeight, leafDepth,
                                                                           static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                                                gainGuide = smoothGuidanceValue(prevCoarse,
                                                                               pred,
                                                                               nextCoarse,
                                                                               gainSpec->temporalSmoothAlpha);
                                            }
                                            const float gain =
                                                shellGainFromCoarse(gainGuide, raw.meta.dataMax, gainLow, gainHigh, gainMax);
                                            pred += gainEncoded / std::max(1e-6f, gain);
                                        } else if (enableShellVoxel[static_cast<size_t>(exportVariantIndex)]) {
                                            const int voxelIdx = localVoxelIndex(lx, ly, lz);
                                            pred += shellVoxelGrid[static_cast<size_t>(voxelIdx) * static_cast<size_t>(frames) +
                                                                   static_cast<size_t>(cli.exportFrameIndex)];
                                        } else if (enableFineBand[static_cast<size_t>(exportVariantIndex)]) {
                                            pred += sampleControlGridValue(fine6BandGrid, 6, frames, cli.exportFrameIndex, leafWidth, leafHeight, leafDepth,
                                                                           static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                                        } else if (exportSpec->tileLocal && enableTile[static_cast<size_t>(exportVariantIndex)][static_cast<size_t>(tileIdx)]) {
                                            pred += sampleControlGridValue(tile3Residual, 3, frames, cli.exportFrameIndex, 4, 4, 4,
                                                                           static_cast<float>(lx % 4),
                                                                           static_cast<float>(ly % 4),
                                                                           static_cast<float>(lz % 4));
                                        }
                                    }
                                    const size_t idx =
                                        (static_cast<size_t>(z) * static_cast<size_t>(raw.meta.height) + static_cast<size_t>(y)) *
                                            static_cast<size_t>(raw.meta.width) +
                                        static_cast<size_t>(x);
                                    exportedFrame[idx] = pred;
                                }
                            }
                        }
                    }

                        for (const auto& sp : samplePoints) {
                            const int tileIdx = (sp.lz / 4) * 4 + (sp.ly / 4) * 2 + (sp.lx / 4);
                            for (size_t vi = 0; vi < variants.size(); ++vi) {
                                float pred = sp.coarsePred;
                            if (vi == packedFine6VariantIndex) {
                                if (hasPackedFine6Leaf && packedFine6LeafView.header.mode != TemporalFirstPackedMode::EMPTY) {
                                    pred = samplePackedLeafGridValue(packedFine6LeafBytes,
                                                                     packedFine6LeafView,
                                                                     false,
                                                                     kCoarseRes,
                                                                     frames,
                                                                     leafWidth,
                                                                     leafHeight,
                                                                     leafDepth,
                                                                     sp.t,
                                                                     static_cast<float>(sp.lx),
                                                                     static_cast<float>(sp.ly),
                                                                     static_cast<float>(sp.lz));
                                    if (packedFine6LeafView.header.mode == TemporalFirstPackedMode::TEMPORAL_FINE6) {
                                        pred += samplePackedLeafGridValue(packedFine6LeafBytes,
                                                                          packedFine6LeafView,
                                                                          true,
                                                                          6,
                                                                          frames,
                                                                          leafWidth,
                                                                          leafHeight,
                                                                          leafDepth,
                                                                          sp.t,
                                                                          static_cast<float>(sp.lx),
                                                                          static_cast<float>(sp.ly),
                                                                          static_cast<float>(sp.lz));
                                    }
                                } else {
                                    pred = 0.0f;
                                }
                            } else if (vi == packedShellVoxelVariantIndex) {
                                if (hasPackedShellVoxelLeaf && packedShellVoxelLeafView.header.mode != TemporalFirstPackedMode::EMPTY) {
                                    pred = samplePackedLeafGridValue(packedShellVoxelLeafBytes,
                                                                     packedShellVoxelLeafView,
                                                                     false,
                                                                     kCoarseRes,
                                                                     frames,
                                                                     leafWidth,
                                                                     leafHeight,
                                                                     leafDepth,
                                                                     sp.t,
                                                                     static_cast<float>(sp.lx),
                                                                     static_cast<float>(sp.ly),
                                                                     static_cast<float>(sp.lz));
                                    if (packedShellVoxelLeafView.header.mode == TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT) {
                                        pred += decodeRenderTemporalLeafShellVoxelValue(
                                            packedShellVoxelLeafBytes.data(),
                                            packedShellVoxelLeafView,
                                            true,
                                            frames,
                                            static_cast<uint16_t>(localVoxelIndex(sp.lx, sp.ly, sp.lz)),
                                            sp.t);
                                    }
                                } else {
                                    pred = 0.0f;
                                }
                            } else if (vi == packedRoutedVariantIndex) {
                                if (hasPackedRoutedLeaf && packedRoutedLeafView.header.mode != TemporalFirstPackedMode::EMPTY) {
                                    pred = samplePackedLeafGridValue(packedRoutedLeafBytes,
                                                                     packedRoutedLeafView,
                                                                     false,
                                                                     kCoarseRes,
                                                                     frames,
                                                                     leafWidth,
                                                                     leafHeight,
                                                                     leafDepth,
                                                                     sp.t,
                                                                     static_cast<float>(sp.lx),
                                                                     static_cast<float>(sp.ly),
                                                                     static_cast<float>(sp.lz));
                                    if (packedRoutedLeafView.header.mode == TemporalFirstPackedMode::TEMPORAL_FINE6) {
                                        pred += samplePackedLeafGridValue(packedRoutedLeafBytes,
                                                                          packedRoutedLeafView,
                                                                          true,
                                                                          6,
                                                                          frames,
                                                                          leafWidth,
                                                                          leafHeight,
                                                                          leafDepth,
                                                                          sp.t,
                                                                          static_cast<float>(sp.lx),
                                                                          static_cast<float>(sp.ly),
                                                                          static_cast<float>(sp.lz));
                                    } else if (packedRoutedLeafView.header.mode == TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT) {
                                        pred += decodeRenderTemporalLeafShellVoxelValue(
                                            packedRoutedLeafBytes.data(),
                                            packedRoutedLeafView,
                                            true,
                                            frames,
                                            static_cast<uint16_t>(localVoxelIndex(sp.lx, sp.ly, sp.lz)),
                                            sp.t);
                                    }
                                } else {
                                    pred = 0.0f;
                                }
                            } else if (vi < baseVariantCount && variantSpecs[vi].routeSelect) {
                                if (routedMode == RenderTemporalFormalMode::EMPTY) {
                                    pred = 0.0f;
                                } else if (routedMode == RenderTemporalFormalMode::FINE6_FULL) {
                                    pred += sp.fine6Pred;
                                } else if (routedMode == RenderTemporalFormalMode::FINE_COMPACT) {
                                    pred += sp.shellVoxelPred;
                                }
                            } else if (emptyLeaf) {
                                pred = 0.0f;
                            } else if (vi < baseVariantCount && enableFine[vi]) {
                                pred += sp.fine6Pred;
                            } else if (vi < baseVariantCount && enableFineGain[vi]) {
                                pred += sp.fine6GainPred;
                            } else if (vi < baseVariantCount && enableShellVoxel[vi]) {
                                pred += sp.shellVoxelPred;
                            } else if (vi < baseVariantCount && enableFineBand[vi]) {
                                pred += sp.fine6BandPred;
                            } else if (vi < baseVariantCount &&
                                       variantSpecs[vi].tileLocal &&
                                       enableTile[vi][static_cast<size_t>(tileIdx)]) {
                                pred += sampleControlGridValue(tile3Residual, 3, frames, sp.t, 4, 4, 4,
                                                               static_cast<float>(sp.lx % 4),
                                                               static_cast<float>(sp.ly % 4),
                                                               static_cast<float>(sp.lz % 4));
                            }
                            const double diff = static_cast<double>(sp.truth) - static_cast<double>(pred);
                            localErr2[vi] += diff * diff;
                            localSamples[vi] += 1;
                        }
                    }
                }
            }
            } catch (const std::exception& ex) {
#ifdef VBT_USE_OPENMP
#pragma omp critical(render_temporal_worker_error)
#endif
                {
                    if (workerError.empty()) {
                        workerError = "Render-temporal leaf (" + std::to_string(currentBx) + "," +
                                      std::to_string(currentBy) + "," + std::to_string(bz) +
                                      ") failed: " + ex.what();
                    }
                }
                workerFailed.store(true, std::memory_order_relaxed);
            } catch (...) {
#ifdef VBT_USE_OPENMP
#pragma omp critical(render_temporal_worker_error)
#endif
                {
                    if (workerError.empty()) {
                        workerError = "Render-temporal leaf (" + std::to_string(currentBx) + "," +
                                      std::to_string(currentBy) + "," + std::to_string(bz) +
                                      ") failed with an unknown exception";
                    }
                }
                workerFailed.store(true, std::memory_order_relaxed);
            }
        }

#ifdef VBT_USE_OPENMP
#pragma omp critical
#endif
        {
            temporalStats.backgroundZeroedVoxels += localBgZero;
            temporalStats.totalKeyframes += localTotalKeys;
            temporalStats.temporalProtectedSeries += localProtected;
            temporalStats.maxKeyframes = std::max(temporalStats.maxKeyframes, localMaxKeys);
            temporalStats.rmse += localTemporalErr2;
            for (size_t vi = 0; vi < variants.size(); ++vi) {
                variants[vi].emptyLeafCount += localEmptyLeaves[vi];
                variants[vi].fineLeafCount += localFineLeaves[vi];
                variants[vi].coarseKeyframes += localCoarseKf[vi];
                variants[vi].fineKeyframes += localFineKf[vi];
                variants[vi].samples += localSamples[vi];
                variants[vi].rmse += localErr2[vi];
                variants[vi].estimatedBytes += localPackedBytes[vi];
            }
        }
    }

    if (workerFailed.load(std::memory_order_relaxed)) {
        throw std::runtime_error(workerError.empty() ? "Render-temporal worker failed" : workerError);
    }

    temporalStats.voxelCount = static_cast<uint64_t>(raw.frameVoxelCount());
    temporalStats.rmse = std::sqrt(temporalStats.rmse / std::max(1.0, static_cast<double>(raw.totalVoxelCount())));
    temporalStats.psnr = computePsnr(temporalStats.rmse, peak);
    for (size_t vi = 0; vi < variants.size(); ++vi) {
        auto& variant = variants[vi];
        variant.rmse = std::sqrt(variant.rmse / std::max(1.0, static_cast<double>(variant.samples)));
        variant.psnr = computePsnr(variant.rmse, peak);
        if (vi < baseVariantCount) {
            variant.estimatedBytes = variant.leafCount * 4ull + variant.coarseKeyframes * 3ull + variant.fineKeyframes * 3ull;
        }
    }

    if (!cli.saveVbtPath.empty()) {
        cli.savedFileBytes = writeRenderTemporalProbeFile(cli.saveVbtPath,
                                                          raw,
                                                          kLeafSize,
                                                          kCoarseRes,
                                                          savedLeafPayloads);
    }

    std::filesystem::create_directories(cli.reportPath.parent_path());
    writeReport(cli.reportPath, cli, raw, temporalStats, variants);
    if (exportFrameEnabled) {
        std::filesystem::create_directories(cli.exportFrameRawPath.parent_path());
        writeSingleFrameRaw(cli.exportFrameRawPath, exportedFrame);
        writeSingleFrameMetadata(cli.exportFrameMetadataPath, raw.meta, cli.exportFrameRawPath);
    }

    if (printConsole) {
        std::cout << "=== Render Temporal-First KF Spatial Probe (streaming) ===\n";
        std::cout << "Input                 : " << cli.inputRaw.string() << "\n";
        std::cout << "Dimensions            : " << raw.meta.width << "x" << raw.meta.height << "x" << raw.meta.depth << " x " << raw.meta.frames << "\n";
        std::cout << "Temporal-only PSNR    : " << temporalStats.psnr << " dB\n";
        std::cout << "Temporal avg KFs      : "
                  << (temporalStats.voxelCount > 0 ? static_cast<double>(temporalStats.totalKeyframes) / static_cast<double>(temporalStats.voxelCount) : 0.0)
                  << "\n";
        for (const auto& v : variants) {
            std::cout << "Variant               : " << v.name
                      << "  estMB=" << std::fixed << std::setprecision(3) << (static_cast<double>(v.estimatedBytes) / (1024.0 * 1024.0))
                      << "  PSNR=" << std::setprecision(4) << v.psnr
                      << " dB  coarseKF=" << v.coarseKeyframes
                      << "  fineKF=" << v.fineKeyframes
                      << "\n";
        }
        std::cout << "Report                : " << cli.reportPath.string() << "\n";
        if (!cli.saveVbtPath.empty()) {
            std::cout << "Saved probe file      : " << cli.saveVbtPath.string() << "\n";
            std::cout << "Saved probe bytes     : " << cli.savedFileBytes << "\n";
        }
        if (exportFrameEnabled) {
            std::cout << "Export raw            : " << cli.exportFrameRawPath.string() << "\n";
            std::cout << "Export metadata       : " << cli.exportFrameMetadataPath.string() << "\n";
            std::cout << "Export variant/frame  : " << cli.exportVariantName << " / " << cli.exportFrameIndex << "\n";
        }
    }
    return 0;
}

#ifdef VBT_RENDER_TEMPORAL_PROBE_STANDALONE
int main(int argc, char** argv)
{
    ProbeCliOptions cli;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input-raw" && i + 1 < argc) cli.inputRaw = argv[++i];
        else if (arg == "--metadata" && i + 1 < argc) cli.metadataPath = argv[++i];
        else if (arg == "--export-frame-raw" && i + 1 < argc) cli.exportFrameRawPath = argv[++i];
        else if (arg == "--export-frame-metadata" && i + 1 < argc) cli.exportFrameMetadataPath = argv[++i];
        else if (arg == "--export-variant" && i + 1 < argc) cli.exportVariantName = argv[++i];
        else if (arg == "--export-frame" && i + 1 < argc) cli.exportFrameIndex = std::stoi(argv[++i]);
        else if (arg == "--sample-step" && i + 1 < argc) cli.sampleStep = std::stoi(argv[++i]);
        else if (arg == "--cutoff" && i + 1 < argc) cli.cutoff = std::stof(argv[++i]);
        else if (arg == "--cutoff-band" && i + 1 < argc) cli.cutoffBand = std::stof(argv[++i]);
        else if (arg == "--temporal-eps-abs" && i + 1 < argc) cli.temporalEpsAbs = std::stof(argv[++i]);
        else if (arg == "--temporal-eps-rel" && i + 1 < argc) cli.temporalEpsRel = std::stof(argv[++i]);
        else if (arg == "--control-eps-scale" && i + 1 < argc) cli.controlEpsScale = std::stof(argv[++i]);
        else if (arg == "--temporal-gamma-delta" && i + 1 < argc) cli.temporalGammaDelta = std::stof(argv[++i]);
        else if (arg == "--bg-zero-ratio" && i + 1 < argc) cli.bgZeroRatio = std::stof(argv[++i]);
        else if (arg == "--packed-keyframe-codec" && i + 1 < argc) {
            const std::string codec = argv[++i];
            if (codec == "fp16") cli.packedCodec = RenderTemporalSequenceCodec::DP_KEYFRAME_FP16;
            else if (codec == "fp32") cli.packedCodec = RenderTemporalSequenceCodec::DP_KEYFRAME_FP32;
            else throw std::runtime_error("packed-keyframe-codec must be fp16 or fp32");
        }
        else if (arg == "--final-only") cli.finalOnly = true;
        else if (arg == "--route-empty-visible-thr" && i + 1 < argc) cli.routeEmptyVisibleThr = std::stof(argv[++i]);
        else if (arg == "--route-fine-visible-thr" && i + 1 < argc) cli.routeFineVisibleThr = std::stof(argv[++i]);
        else if (arg == "--route-fine-band-thr" && i + 1 < argc) cli.routeFineBandThr = std::stof(argv[++i]);
        else if (arg == "--route-coarse-rmse-thr" && i + 1 < argc) cli.routeCoarseRmseThr = std::stof(argv[++i]);
        else if (arg == "--route-coarse-peak-thr" && i + 1 < argc) cli.routeCoarsePeakThr = std::stof(argv[++i]);
        else if (arg == "--route-fine-gain-thr" && i + 1 < argc) cli.routeFineGainThr = std::stof(argv[++i]);
        else if (arg == "--levelset-surface-band" && i + 1 < argc) cli.levelSetSurfaceBand = std::stof(argv[++i]);
        else if (arg == "--levelset-coarse-guard-band" && i + 1 < argc) cli.levelSetCoarseGuardBand = std::stof(argv[++i]);
        else if (arg == "--levelset-temporal-band" && i + 1 < argc) cli.levelSetTemporalBand = std::stof(argv[++i]);
        else if (arg == "--legacy-levelset-route") cli.legacyLevelSetRoute = true;
        else if (arg == "--cutoff-protect") cli.cutoffProtect = true;
        else if (arg == "--no-cutoff-protect") cli.cutoffProtect = false;
        else if (arg == "--omp-threads" && i + 1 < argc) cli.ompThreads = std::stoi(argv[++i]);
        else if (arg == "--save-vbt" && i + 1 < argc) cli.saveVbtPath = argv[++i];
        else if (arg == "--report" && i + 1 < argc) cli.reportPath = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage();
            return 1;
        }
    }
    try {
        return runRenderTemporalPipeline(cli, true);
    } catch (const std::exception& ex) {
        std::cerr << "[fatal] " << ex.what() << std::endl;
        std::cerr.flush();
        return 99;
    }
}
#endif

namespace vbt {

int runRenderTemporalMainline(const std::filesystem::path& inputRaw,
                              const std::filesystem::path& metadataPath,
                              const std::filesystem::path& reportPath,
                              const SpatialFirstOptions& options)
{
    const auto meta = loadFrameMetadata(metadataPath);
    const bool preferLiquidShellMode =
        meta.conversionMode == "shell" ||
        meta.sourceType.find("alembic_mesh_sequence") != std::string::npos;

    ProbeCliOptions cli;
    cli.inputRaw = inputRaw;
    cli.metadataPath = metadataPath;
    cli.reportPath = reportPath;
    cli.sampleStep = std::max(1, std::min({options.sampleStepX, options.sampleStepY, options.sampleStepZ, options.sampleStepT}));
    cli.cutoff = options.profile.den.renderCutoff > 0.0f ? options.profile.den.renderCutoff : 0.0003f;
    cli.cutoffBand = options.profile.den.cutoffBand > 0.0f ? options.profile.den.cutoffBand : 0.0001f;
    cli.temporalEpsAbs = options.renderTemporalProbeEpsAbs > 0.0f ? options.renderTemporalProbeEpsAbs : 1e-5f;
    cli.temporalEpsRel = options.renderTemporalProbeEpsRel;
    cli.temporalGammaDelta = options.renderTemporalProbeGammaDelta;
    cli.bgZeroRatio = options.renderTemporalProbeBgZeroRatio;
    cli.cutoffProtect = options.renderTemporalProbeCutoffProtect;
    cli.ompThreads = options.ompThreads;
    cli.packedCodec = RenderTemporalSequenceCodec::DP_KEYFRAME_FP16;
    cli.controlEpsScale = makeRenderTemporalFormalDefaults().controlEpsScale;
    if (meta.conversionMode == "levelset") {
        cli.cutoff = 0.0f;
        cli.cutoffBand = 0.0f;
        cli.cutoffProtect = false;
        cli.levelSetSurfaceBand =
            (meta.shellWidthVoxels > 0.0f ? meta.shellWidthVoxels : 1.5f) * meta.voxelSize;
    }

    const auto routeDefaults = makeRenderTemporalRouteDefaults(cli.cutoff, cli.cutoffBand, cli.bgZeroRatio);
    cli.routeEmptyVisibleThr = routeDefaults.emptyVisibleFracThreshold;
    cli.routeFineVisibleThr = routeDefaults.fineVisibleFracThreshold;
    cli.routeFineBandThr = routeDefaults.fineBandFracThreshold;
    cli.routeCoarseRmseThr = routeDefaults.coarseRmseThreshold;
    cli.routeCoarsePeakThr = routeDefaults.coarsePeakThreshold;
    cli.routeFineGainThr = routeDefaults.fineGainThreshold;
    cli.routeShellFracThr = routeDefaults.shellFracThreshold;
    cli.routeCoarseShellRmseThr = routeDefaults.coarseShellRmseThreshold;
    cli.routeCoarseShellPeakThr = routeDefaults.coarseShellPeakThreshold;
    cli.routeShellGainThr = routeDefaults.shellGainThreshold;
    if (preferLiquidShellMode) {
        cli.routeFineVisibleThr = 1.0f;
        cli.routeFineBandThr = 1.0f;
        cli.routeFineGainThr = 1.0f;
        cli.preferShellVoxelSave = true;
    } else {
        cli.routeShellFracThr = 1.0f;
        cli.routeCoarseShellRmseThr = 1.0f;
        cli.routeCoarseShellPeakThr = 1.0f;
        cli.routeShellGainThr = 1.0f;
    }
    cli.saveVbtPath = options.saveVbtPath;

    return runRenderTemporalPipeline(cli, true);
}

} // namespace vbt

namespace {

void addIsoSurfaceKeyFrames(const std::vector<float>& values,
                           float isoValue,
                           float isoBand,
                           std::vector<int>& keys)
{
    if (values.empty()) return;
    const float band = std::max(0.0f, isoBand);
    auto addWithNeighbors = [&](int t) {
        if (t > 0) keys.push_back(t - 1);
        keys.push_back(t);
        if (t + 1 < static_cast<int>(values.size())) keys.push_back(t + 1);
    };
    for (int t = 0; t < static_cast<int>(values.size()); ++t) {
        const float current = values[static_cast<size_t>(t)] - isoValue;
        if (std::abs(current) <= band) {
            addWithNeighbors(t);
        }
        if (t > 0) {
            const float previous = values[static_cast<size_t>(t - 1)] - isoValue;
            if ((previous >= 0.0f) != (current >= 0.0f)) {
                addWithNeighbors(t - 1);
                addWithNeighbors(t);
            }
        }
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
}

std::vector<int> detectDensityKeyFrames(const std::vector<float>& values,
                                        const DensityTemporalProfile& profile,
                                        bool* usedCutoffProtect)
{
    if (usedCutoffProtect) *usedCutoffProtect = false;
    const int n = static_cast<int>(values.size());
    if (n == 0) return {};
    if (n == 1) return {0};

    const auto stats = computeStats(values);
    const float epsFloor = std::max(1e-6f, profile.epsAbs * 0.1f);
    {
        double sum = 0.0;
        double sum2 = 0.0;
        for (float v : values) {
            sum += v;
            sum2 += static_cast<double>(v) * v;
        }
        const double mean = sum / static_cast<double>(n);
        const double var = std::max(0.0, sum2 / static_cast<double>(n) - mean * mean);
        if (var < static_cast<double>(epsFloor) * epsFloor) return {0};
    }

    std::vector<int> keys{0, n - 1};
    dpRecurse(values, 0, n - 1, profile, stats, keys);

    bool protectedKeysAdded = false;
    if (profile.cutoffTemporalProtect && profile.renderCutoff > 0.0f && profile.cutoffBand >= 0.0f) {
        const float cutoff = profile.renderCutoff;
        const float band = profile.cutoffBand;
        for (int t = 0; t < n; ++t) {
            const int s = cutoffState(values[static_cast<size_t>(t)], cutoff, band);
            if (s == 1) {
                keys.push_back(t);
                if (t > 0) keys.push_back(t - 1);
                if (t + 1 < n) keys.push_back(t + 1);
                protectedKeysAdded = true;
            }
            if (t > 0) {
                const int prev = cutoffState(values[static_cast<size_t>(t - 1)], cutoff, band);
                if (prev != s) {
                    keys.push_back(t - 1);
                    keys.push_back(t);
                    if (t + 1 < n) keys.push_back(t + 1);
                    protectedKeysAdded = true;
                }
            }
        }
    }
    if (profile.isoSurfaceProtect) {
        const size_t keyCountBefore = keys.size();
        addIsoSurfaceKeyFrames(values, profile.isoValue, profile.isoBand, keys);
        protectedKeysAdded = protectedKeysAdded || keys.size() > keyCountBefore;
    }

    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    if (usedCutoffProtect) *usedCutoffProtect = protectedKeysAdded;
    return keys;
}

void reconstructFromKeys(const std::vector<float>& values,
                         const std::vector<int>& keys,
                         std::vector<float>& out)
{
    out.resize(values.size(), 0.0f);
    if (values.empty()) return;
    if (keys.empty()) {
        out = values;
        return;
    }
    if (keys.size() == 1) {
        std::fill(out.begin(), out.end(), values[static_cast<size_t>(keys.front())]);
        return;
    }
    size_t seg = 0;
    for (int t = 0; t < static_cast<int>(values.size()); ++t) {
        while (seg + 1 < keys.size() && t > keys[seg + 1]) ++seg;
        if (seg + 1 >= keys.size()) {
            out[static_cast<size_t>(t)] = values[static_cast<size_t>(keys.back())];
            continue;
        }
        const int a = keys[seg];
        const int b = keys[seg + 1];
        if (t <= a) {
            out[static_cast<size_t>(t)] = values[static_cast<size_t>(a)];
            continue;
        }
        const double alpha = static_cast<double>(t - a) / static_cast<double>(b - a);
        out[static_cast<size_t>(t)] =
            static_cast<float>(static_cast<double>(values[static_cast<size_t>(a)]) +
                               alpha * static_cast<double>(values[static_cast<size_t>(b)] - values[static_cast<size_t>(a)]));
    }
}

float sampleLeafBufferAtFractional(const std::vector<float>& leafTemporal,
                                   int frames,
                                   int t,
                                   int leafWidth,
                                   int leafHeight,
                                   int leafDepth,
                                   float fx,
                                   float fy,
                                   float fz)
{
    const float maxX = static_cast<float>(std::max(0, leafWidth - 1));
    const float maxY = static_cast<float>(std::max(0, leafHeight - 1));
    const float maxZ = static_cast<float>(std::max(0, leafDepth - 1));
    const float x = std::clamp(fx, 0.0f, maxX);
    const float y = std::clamp(fy, 0.0f, maxY);
    const float z = std::clamp(fz, 0.0f, maxZ);
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int z0 = static_cast<int>(std::floor(z));
    const int x1 = std::min(leafWidth - 1, x0 + 1);
    const int y1 = std::min(leafHeight - 1, y0 + 1);
    const int z1 = std::min(leafDepth - 1, z0 + 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float tz = z - static_cast<float>(z0);
    auto at = [&](int lx, int ly, int lz) {
        return leafTemporalAt(leafTemporal, localVoxelIndex(lx, ly, lz), frames, t);
    };
    const float c00 = at(x0, y0, z0) * (1.0f - tx) + at(x1, y0, z0) * tx;
    const float c01 = at(x0, y0, z1) * (1.0f - tx) + at(x1, y0, z1) * tx;
    const float c10 = at(x0, y1, z0) * (1.0f - tx) + at(x1, y1, z0) * tx;
    const float c11 = at(x0, y1, z1) * (1.0f - tx) + at(x1, y1, z1) * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

float sampleControlGridValue(const std::vector<float>& gridSeries,
                             int resolution,
                             int frames,
                             int t,
                             int leafWidth,
                             int leafHeight,
                             int leafDepth,
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
        return gridSeries[static_cast<size_t>(idx * frames + t)];
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

std::vector<uint8_t> buildBinIndexFromKeys(const std::vector<int>& keys, int frames, int timeBinCount)
{
    std::vector<uint8_t> out(static_cast<size_t>(timeBinCount), 0u);
    if (keys.empty()) return out;
    for (int bin = 0; bin < timeBinCount; ++bin) {
        const int binStart =
            static_cast<int>((static_cast<int64_t>(bin) * static_cast<int64_t>(frames)) / std::max(1, timeBinCount));
        uint8_t localStart = static_cast<uint8_t>(keys.size() - 1);
        for (size_t i = 0; i < keys.size(); ++i) {
            if (keys[i] >= binStart) {
                localStart = static_cast<uint8_t>(i > 0 ? (i - 1) : 0);
                break;
            }
        }
        out[static_cast<size_t>(bin)] = localStart;
    }
    return out;
}

BuiltControlStream buildControlStreamFromGridSeries(const std::vector<float>& gridSeries,
                                                    int controlCount,
                                                    int frames,
                                                    const DensityTemporalProfile& profile,
                                                    int timeBinCount)
{
    BuiltControlStream out;
    out.data.controlCount = static_cast<uint16_t>(controlCount);
    out.data.timeBinCount = static_cast<uint8_t>(timeBinCount);
    out.data.descriptors.reserve(static_cast<size_t>(controlCount));
    out.data.binIndex.reserve(static_cast<size_t>(controlCount) * static_cast<size_t>(timeBinCount));

    std::vector<float> series(static_cast<size_t>(frames), 0.0f);
    for (int ctrl = 0; ctrl < controlCount; ++ctrl) {
        for (int t = 0; t < frames; ++t) {
            series[static_cast<size_t>(t)] =
                gridSeries[static_cast<size_t>(ctrl) * static_cast<size_t>(frames) + static_cast<size_t>(t)];
        }

        const auto keys = detectDensityKeyFrames(series, profile, nullptr);
        RenderTemporalControlDescriptor desc;
        desc.keyStart = static_cast<uint16_t>(out.data.keyframes.size());
        desc.keyCount = static_cast<uint8_t>(std::min<size_t>(255, keys.size()));
        const auto bins = buildBinIndexFromKeys(keys, frames, timeBinCount);
        int localMaxBinKeys = 1;
        for (int bin = 0; bin < timeBinCount; ++bin) {
            const int curr = static_cast<int>(bins[static_cast<size_t>(bin)]);
            const int next = (bin + 1 < timeBinCount)
                                 ? static_cast<int>(bins[static_cast<size_t>(bin + 1)])
                                 : static_cast<int>(keys.size());
            localMaxBinKeys = std::max(localMaxBinKeys, std::max(1, next - curr));
        }
        desc.reserved = static_cast<uint8_t>(std::min(localMaxBinKeys, 255));
        out.data.descriptors.push_back(desc);
        out.data.binIndex.insert(out.data.binIndex.end(), bins.begin(), bins.end());
        for (int keyFrame : keys) {
            RenderTemporalControlKeyframe key{};
            key.frame = static_cast<uint16_t>(keyFrame);
            key.value = series[static_cast<size_t>(keyFrame)];
            out.data.keyframes.push_back(key);
        }
        for (int bin = 0; bin < timeBinCount; ++bin) {
            const int curr = static_cast<int>(bins[static_cast<size_t>(bin)]);
            const int next = (bin + 1 < timeBinCount)
                                 ? static_cast<int>(bins[static_cast<size_t>(bin + 1)])
                                 : static_cast<int>(keys.size());
            out.requiredMaxBinLocalKeys = static_cast<uint8_t>(
                std::max<int>(out.requiredMaxBinLocalKeys, std::max(1, next - curr)));
        }
    }
    out.data.maxBinLocalKeys = out.requiredMaxBinLocalKeys;
    return out;
}

void appendControlSequenceToStream(RenderTemporalControlStreamData& stream,
                                   uint8_t& requiredMaxBinLocalKeys,
                                   const std::vector<float>& series,
                                   const std::vector<int>& keys,
                                   int frames,
                                   int timeBinCount)
{
    RenderTemporalControlDescriptor desc;
    desc.keyStart = static_cast<uint16_t>(stream.keyframes.size());
    desc.keyCount = static_cast<uint8_t>(std::min<size_t>(255, keys.size()));
    const auto bins = buildBinIndexFromKeys(keys, frames, timeBinCount);
    int localMaxBinKeys = 1;
    for (int bin = 0; bin < timeBinCount; ++bin) {
        const int curr = static_cast<int>(bins[static_cast<size_t>(bin)]);
        const int next = (bin + 1 < timeBinCount)
                             ? static_cast<int>(bins[static_cast<size_t>(bin + 1)])
                             : static_cast<int>(keys.size());
        localMaxBinKeys = std::max(localMaxBinKeys, std::max(1, next - curr));
    }
    desc.reserved = static_cast<uint8_t>(std::min(localMaxBinKeys, 255));
    stream.descriptors.push_back(desc);

    stream.binIndex.insert(stream.binIndex.end(), bins.begin(), bins.end());
    for (int keyFrame : keys) {
        RenderTemporalControlKeyframe key{};
        key.frame = static_cast<uint16_t>(keyFrame);
        key.value = series[static_cast<size_t>(keyFrame)];
        stream.keyframes.push_back(key);
    }
    for (int bin = 0; bin < timeBinCount; ++bin) {
        const int curr = static_cast<int>(bins[static_cast<size_t>(bin)]);
        const int next = (bin + 1 < timeBinCount)
                             ? static_cast<int>(bins[static_cast<size_t>(bin + 1)])
                             : static_cast<int>(keys.size());
        requiredMaxBinLocalKeys = static_cast<uint8_t>(
            std::max<int>(requiredMaxBinLocalKeys, std::max(1, next - curr)));
    }
}

float samplePackedLeafGridValue(const std::vector<uint8_t>& leafBytes,
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
        return decodeRenderTemporalLeafControlValue(leafBytes.data(),
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

void writeReport(const std::filesystem::path& path,
                 const ProbeCliOptions& cli,
                 const RawVolume4D& volume,
                 const TemporalStats& temporalStats,
                 const std::vector<VariantResult>& variants)
{
    std::ofstream out(path);
    out << "# Render Temporal-First Keyframe Spatial Probe\n\n";
    out << "- input: `" << cli.inputRaw.string() << "`\n";
    out << "- dimensions: `" << volume.meta.width << "x" << volume.meta.height << "x" << volume.meta.depth << " x " << volume.meta.frames << "`\n";
    out << "- sample step: `" << cli.sampleStep << "`\n";
    out << "- cutoff: `" << cli.cutoff << "`\n";
    out << "- cutoff band: `" << cli.cutoffBand << "`\n";
    out << "- temporal eps abs/rel: `" << cli.temporalEpsAbs << " / " << cli.temporalEpsRel << "`\n";
    out << "- control eps scale: `" << cli.controlEpsScale << "`\n";
    out << "- final-only variants: `" << (cli.finalOnly ? "true" : "false") << "`\n";
    out << "- temporal gamma delta: `" << cli.temporalGammaDelta << "`\n";
    out << "- bg zero ratio: `" << cli.bgZeroRatio << "`\n";
    out << "- route empty visible threshold: `" << cli.routeEmptyVisibleThr << "`\n";
    out << "- route fine visible threshold: `" << cli.routeFineVisibleThr << "`\n";
    out << "- route fine band threshold: `" << cli.routeFineBandThr << "`\n";
    out << "- route coarse RMSE threshold: `" << cli.routeCoarseRmseThr << "`\n";
    out << "- route coarse peak threshold: `" << cli.routeCoarsePeakThr << "`\n";
    out << "- route fine gain threshold: `" << cli.routeFineGainThr << "`\n";
    out << "- cutoff protect: `" << (cli.cutoffProtect ? "true" : "false") << "`\n";
    out << "- level-set surface mode: `" << (cli.levelSetSurfaceMode ? "true" : "false") << "`\n";
    if (cli.levelSetSurfaceMode) {
        out << "- level-set surface band: `" << cli.levelSetSurfaceBand << "`\n";
        out << "- level-set coarse guard band: `" << cli.levelSetCoarseGuardBand << "`\n";
        out << "- level-set temporal protect band: `" << cli.levelSetTemporalBand << "`\n";
    }
    if (!cli.saveVbtPath.empty()) {
        out << "- saved probe file: `" << cli.saveVbtPath.string() << "`\n";
        out << "- saved probe bytes: `" << cli.savedFileBytes << "`\n";
    }
    out << "\n";
    out << "## Temporal-First Frontend\n\n";
    out << "- exact temporal-only RMSE: `" << temporalStats.rmse << "`\n";
    out << "- exact temporal-only PSNR: `" << temporalStats.psnr << " dB`\n";
    out << "- voxel count: `" << temporalStats.voxelCount << "`\n";
    out << "- average keyframes per voxel: `"
        << (temporalStats.voxelCount > 0 ? static_cast<double>(temporalStats.totalKeyframes) / static_cast<double>(temporalStats.voxelCount) : 0.0)
        << "`\n";
    out << "- max keyframes per voxel: `" << temporalStats.maxKeyframes << "`\n";
    out << "- background-zero voxels: `" << temporalStats.backgroundZeroedVoxels << "`\n";
    out << "- cutoff-protected voxel series: `" << temporalStats.temporalProtectedSeries << "`\n\n";
    out << "## Spatial Backend Variants\n\n";
    out << "| Variant | Est. Bytes | Sampled RMSE | Sampled PSNR | coarse KFs | fine KFs | Empty Leaves | Fine Leaves |\n";
    out << "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
    for (const auto& v : variants) {
        out << "| `" << v.name << "` | `"
            << std::fixed << std::setprecision(3) << (static_cast<double>(v.estimatedBytes) / (1024.0 * 1024.0))
            << " MB` | `" << std::setprecision(9) << v.rmse
            << "` | `" << std::setprecision(4) << v.psnr
            << " dB` | `" << v.coarseKeyframes
            << "` | `" << v.fineKeyframes
            << "` | `" << v.emptyLeafCount
            << "` | `" << v.fineLeafCount
            << "` |\n";
    }
}

void writeSingleFrameRaw(const std::filesystem::path& path, const std::vector<float>& frameValues)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open frame raw for write: " + path.string());
    }
    out.write(reinterpret_cast<const char*>(frameValues.data()),
              static_cast<std::streamsize>(frameValues.size() * sizeof(float)));
    if (!out) {
        throw std::runtime_error("Failed to write frame raw: " + path.string());
    }
}

void writeSingleFrameMetadata(const std::filesystem::path& path,
                              const FrameMetadata& meta,
                              const std::filesystem::path& sourceRaw)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open frame metadata for write: " + path.string());
    }
    out << "{\n";
    out << "  \"source_dir\": \"" << sourceRaw.parent_path().string() << "\",\n";
    out << "  \"grid_name\": \"" << meta.gridName << "\",\n";
    out << "  \"format\": \"float32_raw_dense_index_space\",\n";
    out << "  \"width\": " << meta.width << ",\n";
    out << "  \"height\": " << meta.height << ",\n";
    out << "  \"depth\": " << meta.depth << ",\n";
    out << "  \"frames\": 1,\n";
    out << "  \"bbox_min\": [" << meta.bboxMin[0] << ", " << meta.bboxMin[1] << ", " << meta.bboxMin[2] << "],\n";
    out << "  \"bbox_max\": [" << meta.bboxMax[0] << ", " << meta.bboxMax[1] << ", " << meta.bboxMax[2] << "],\n";
    out << "  \"data_min\": " << meta.dataMin << ",\n";
    out << "  \"data_max\": " << meta.dataMax << "\n";
    out << "}\n";
    if (!out) {
        throw std::runtime_error("Failed to write frame metadata: " + path.string());
    }
}

uint64_t writeRenderTemporalProbeFile(const std::filesystem::path& path,
                                      const RawVolume4D& volume,
                                      int leafSize,
                                      int coarseResolution,
                                      const std::vector<std::vector<uint8_t>>& leafPayloads)
{
    auto alignPayloadBytes = [](size_t bytes) -> size_t {
        return static_cast<size_t>(((static_cast<uint64_t>(bytes) + 3ull) / 4ull) * 4ull);
    };

    std::filesystem::create_directories(path.parent_path());
    std::vector<uint32_t> offsetsWords(leafPayloads.size() + 1u, 0u);
    std::vector<uint8_t> payloadPool;
    payloadPool.reserve(leafPayloads.size() * 128ull);

    for (size_t i = 0; i < leafPayloads.size(); ++i) {
        offsetsWords[i] = static_cast<uint32_t>(payloadPool.size() / 4u);
        std::vector<uint8_t> payload = leafPayloads[i];
        const size_t alignedSize = alignPayloadBytes(payload.size());
        payload.resize(alignedSize, 0u);
        payloadPool.insert(payloadPool.end(), payload.begin(), payload.end());
    }
    offsetsWords.back() = static_cast<uint32_t>(payloadPool.size() / 4u);

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open render temporal output file: " + path.string());
    }

    const char magic[8] = {'V','B','T','P','A','C','K','4'};
    out.write(magic, sizeof(magic));
    const uint32_t version = 4;
    auto writeU32 = [&](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
    writeU32(version);
    writeU32(static_cast<uint32_t>(volume.meta.width));
    writeU32(static_cast<uint32_t>(volume.meta.height));
    writeU32(static_cast<uint32_t>(volume.meta.depth));
    writeU32(static_cast<uint32_t>(volume.meta.frames));
    writeU32(static_cast<uint32_t>(leafSize));
    writeU32(static_cast<uint32_t>(coarseResolution));
    writeU32(0u);
    writeU32(static_cast<uint32_t>(leafPayloads.size()));
    writeU32(static_cast<uint32_t>(FieldType::DENSITY));
    writeU32(0u);
    out.write(reinterpret_cast<const char*>(offsetsWords.data()),
              static_cast<std::streamsize>(sizeof(uint32_t) * offsetsWords.size()));
    if (!payloadPool.empty()) {
        out.write(reinterpret_cast<const char*>(payloadPool.data()),
                  static_cast<std::streamsize>(payloadPool.size()));
    }
    out.flush();
    out.close();
    return std::filesystem::file_size(path);
}

} // namespace
