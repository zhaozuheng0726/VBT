#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "block_tree.h"
#include "field_profile.h"
#include "keyframe_detector.h"
#include "volume_loader.h"
#include "vdb_tools/frame_metadata.h"

using Volume4D = std::vector<std::vector<std::vector<std::vector<float>>>>;
using CompressedVolume4D = std::vector<std::vector<std::vector<std::vector<Point1D>>>>;

static std::filesystem::path guessMetadataPathForRaw(const std::string& inputFile) {
    namespace fs = std::filesystem;
    const fs::path rawPath(inputFile);
    std::vector<fs::path> candidates;
    candidates.push_back(rawPath.parent_path() / (rawPath.stem().string() + ".metadata.json"));
    candidates.push_back(rawPath.parent_path() / (rawPath.filename().string() + ".metadata.json"));
    for (const auto& p : candidates) {
        if (fs::exists(p)) return p;
    }
    return {};
}

struct ProfileOptions {
    std::string name = "SDF";
    float epsAbs = 1.0f;
    float epsRel = 0.05f;
    float gammaDelta = 0.2f;
    float baseEps = 6.0f;
    float renderCutoff = -1.0f;
    float cutoffBand = 0.0f;
    bool cutoffTemporalProtect = false;
    bool cutoffClusterProtect = false;
    float bgZeroRatio = 0.30f;
    float bgConstRatio = 0.60f;
    float iso = 128.0f;
    float wband = 16.0f;
    float nearBand = 4.0f;
    float criticalBand = 2.0f;
    float epsNear = 2.0f;
    float epsFar = 6.0f;
    float epsCritical = 0.5f;
};

struct CalibrationOptions {
    bool enabled = false;
    int subW = 64;
    int subH = 64;
    int subD = 64;
    int subF = 32;
    int numBlocks = 4;
    int startX = -1;
    int startY = -1;
    int startZ = -1;
    int startT = -1;
    double targetP99 = -1.0;
    double targetP999 = -1.0;
    double targetPsnr = -1.0;
    double targetRmse = -1.0;
};

struct LeafAnalysisOptions {
    bool enabled = false;
    int topK = 20;
};

struct SpatialProbeOptions {
    std::string mode;
    int topK = 64;
    std::string leafCsv;
    double threshold = -1.0;
    double adaptiveBaseFactor = -1.0;
    double adaptiveFine8Trigger = -1.0;
    double scoreT1 = -1.0;
    double scoreT2 = -1.0;
    double scoreWRmse = -1.0;
    double scoreWP99 = -1.0;
    double scoreWVis = -1.0;
    double scoreWGrad = -1.0;
    double scoreWVar = -1.0;
    double fine8Improve = -1.0;
    int exportFrame = -1;
    std::string exportDir;
    std::string exportName;
};

struct ResidualOptions {
    int grid4DenseHotspots = 0;
    int grid4ResidualHotspots = 0;
    int grid4ResidualRegionSeeds = 0;
    int grid4ResidualRegionRadius = 1;
    int grid4ResidualLeafSampleStep = 1;
    double grid4ResidualThr = -1.0;
    double grid4ResidualRelThr = -1.0;
    double grid4ResidualLocalFloor = -1.0;
    double grid4ResidualBandFactor = 0.70;
    double grid4ResidualKeepRel = 0.05;
    bool grid4ResidualRankNormalized = false;
    double grid4ResidualDpEps = 2.0;
    int poly11DenseHotspots = 0;
    int poly11ResidualHotspots = 0;
    double poly11ResidualThr = -1.0;
    double poly11ResidualDpEps = 2.0;
};

struct Grid4EncoderOptions {
    bool controlOnlyTemporal = false;
};

struct Grid4MultiscaleOptions {
    bool enabled = false;
    double scoreT1 = 0.35;
    double scoreT2 = 0.75;
    double scoreWRmse = 0.40;
    double scoreWP99 = 0.20;
    double scoreWVis = 0.20;
    double scoreWGrad = 0.10;
    double scoreWVar = 0.10;
    double scoreRmseTargetRel = -1.0;
    double fine8Improve = 0.90;
    double visPromoteFine6 = 0.03;
    double visPromoteFine8 = 0.02;
    double visFine8Improve = 0.80;
    double fineCtrlBandFactor = 0.0;
    double fineCtrlKeepRel = -1.0;
    double fineResidualDpEps = 0.35;
    double fineResidualCutoff = 0.005;
    double fineResidualCutoffBand = 0.002;
    bool fineResidualCutoffProtect = true;
    double genericPromoteRmseNorm = 0.08;
    double genericPromoteP99Norm = 0.18;
    double genericFine6Improve = 0.75;
    double genericFine8Improve = 0.85;
    double genericFineCostKfPerCtrl = 24.0;
    double genericFineCostImprove = 0.55;
    double genericBudgetFraction = 0.15;
    int genericBudgetTopK = 0;
};

struct AutoPolicySearchOptions {
    bool enabled = false;
    int sampleStepX = 4;
    int sampleStepY = 4;
    int sampleStepZ = 4;
    int sampleStepT = 4;
};

enum class AutoSpatialRouteMode {
    CLUSTERED = 0,
    GRID4 = 1,
    GRID4_RESIDUAL = 2,
    GRID4_MULTISCALE = 3
};

struct AutoSpatialRouteDecision {
    AutoSpatialRouteMode mode = AutoSpatialRouteMode::GRID4;
    bool temporalLimited = false;
    double temporalNormRmse = 0.0;
    double avgKfPerVoxel = 0.0;
    double spatialSmoothNorm = 0.0;
    int residualTopK = 0;
    double residualDp = 6.0;
    const char* reason = "default GRID4-family route";
};

struct EvalMetrics {
    size_t vbtBytes = 0;
    int leaves = 0;
    long long totalKF = 0;
    double mean = 0.0;
    double max = 0.0;
    double p99 = 0.0;
    double p999 = 0.0;
    double rmse = 0.0;
    double psnr = 0.0;
    double renderProxyRmse = 0.0;
    double renderProxyPsnr = 0.0;
    bool valid = true;
};

struct SpatialPolicyCandidate {
    std::string name;
    AutoSpatialRouteMode mode = AutoSpatialRouteMode::GRID4;
    Grid4MultiscaleOptions grid4Ms;
    ResidualOptions residual;
};

struct SpatialPolicyEvalResult {
    SpatialPolicyCandidate candidate;
    EvalMetrics metrics;
};

struct LeafErrorStats {
    int leafId = -1;
    int bx = 0;
    int by = 0;
    int bz = 0;
    long long samples = 0;
    double range = 0.0;
    double mean = 0.0;
    double max = 0.0;
    double p99 = 0.0;
    double p999 = 0.0;
    double rmse = 0.0;
    double temporalRmse = 0.0;
    double normRmse = 0.0;
    double temporalNormRmse = 0.0;
    double ampVsTemporal = 1.0;
    double sse = 0.0;
};

struct HotspotLeafRow {
    int leafId = -1;
    int bx = 0;
    int by = 0;
    int bz = 0;
    double range = 0.0;
    double baselineRmse = 0.0;
    double baselineP99 = 0.0;
    double baselineP999 = 0.0;
    double baselineMax = 0.0;
    double temporalRmse = 0.0;
    double ampVsTemporal = 0.0;
};

struct GridProbeLeafResult {
    HotspotLeafRow base;
    double gridMean = 0.0;
    double gridRmse = 0.0;
    double gridP99 = 0.0;
    double gridP999 = 0.0;
    double gridMax = 0.0;
    long long controlKF = 0;
    long long fullLeafKF = 0;
};

struct CalibrationCandidate {
    float epsAbs = 0.0f;
    float epsRel = 0.0f;
    float gammaDelta = 0.0f;
    float clusterThr = 1.0f;
};

struct CalibrationResult {
    CalibrationCandidate candidate;
    EvalMetrics metrics;
};

struct SubvolumeSpec {
    int startX = 0;
    int startY = 0;
    int startZ = 0;
    int startT = 0;
    std::string tag;
};

struct BlockScore {
    SubvolumeSpec spec;
    double range = 0.0;
    double deltaMean = 0.0;
    double stddev = 0.0;
    double absMean = 0.0;
};

struct SampledFieldStats {
    double minValue = 0.0;
    double maxValue = 0.0;
    double q01 = 0.0;
    double q99 = 0.0;
    double q005 = 0.0;
    double q995 = 0.0;
    double robustRange = 0.0;
    double tighterRange = 0.0;
    double deltaMean = 0.0;
    double deltaP90 = 0.0;
    double deltaP99 = 0.0;
    double spatialMean = 0.0;
    double spatialP90 = 0.0;
    double spatialP99 = 0.0;
    double lag1Autocorr = 1.0;
    int stepX = 1;
    int stepY = 1;
    int stepZ = 1;
    int stepT = 1;
    size_t valueSamples = 0;
    size_t deltaSamples = 0;
    size_t spatialSamples = 0;
};

static size_t computeFlatVbtBytes(const BlockTree& tree);

static bool recommendGrid4Spatial(
    const std::string& profileName,
    int width, int height, int depth,
    long long totalKF,
    const SampledFieldStats& sampledStats)
{
    if (profileName == "SDF") return true;
    const double avgKfPerVoxel =
        static_cast<double>(totalKF) / std::max(1LL, static_cast<long long>(width) * height * depth);
    const double spatialSmoothNorm = sampledStats.spatialP90 / std::max(1e-6, sampledStats.robustRange);
    return avgKfPerVoxel <= 6.0 &&
           sampledStats.lag1Autocorr >= 0.82 &&
           spatialSmoothNorm <= 0.12;
}

static AutoSpatialRouteDecision recommendSpatialRoute(
    const std::string& profileName,
    int width, int height, int depth,
    long long totalKF,
    const SampledFieldStats& sampledStats,
    double temporalRmse,
    double dataMin,
    double dataMax)
{
    AutoSpatialRouteDecision d;
    const double range = std::max(1e-6, dataMax - dataMin);
    d.temporalNormRmse = temporalRmse / range;
    d.avgKfPerVoxel =
        static_cast<double>(totalKF) / std::max(1LL, static_cast<long long>(width) * height * depth);
    d.spatialSmoothNorm = sampledStats.spatialP90 / std::max(1e-6, sampledStats.robustRange);

    if (profileName == "SDF") {
        d.mode = AutoSpatialRouteMode::GRID4_MULTISCALE;
        d.reason = "SDF defaults to GRID4 coarse + fine residual";
        return d;
    }

    if (d.temporalNormRmse > 0.08) {
        d.temporalLimited = true;
    }

    const bool legacyClusterLike =
        !d.temporalLimited &&
        d.temporalNormRmse <= 0.02 &&
        d.avgKfPerVoxel >= 6.0 &&
        d.avgKfPerVoxel <= 16.0 &&
        sampledStats.lag1Autocorr >= 0.82 &&
        d.spatialSmoothNorm <= 0.16;
    const bool smoothContinuousSpatialFailure =
        !d.temporalLimited &&
        d.temporalNormRmse <= 0.015 &&
        d.avgKfPerVoxel < 6.0 &&
        sampledStats.lag1Autocorr >= 0.80 &&
        d.spatialSmoothNorm <= 0.10;
    const bool difficultGenericField =
        profileName == "GENERIC" &&
        (d.spatialSmoothNorm > 0.22 ||
         (d.avgKfPerVoxel > 8.0 && d.spatialSmoothNorm > 0.16));

    if (profileName == "DENSITY") {
        d.mode = AutoSpatialRouteMode::GRID4_MULTISCALE;
        d.reason = "rendering-oriented field prefers fine residual grid";
        return d;
    }

    if (!d.temporalLimited &&
        d.temporalNormRmse <= 0.015 &&
        (difficultGenericField ||
         smoothContinuousSpatialFailure ||
         sampledStats.lag1Autocorr <= 0.35)) {
        d.mode = AutoSpatialRouteMode::GRID4_RESIDUAL;
        if (difficultGenericField) {
            d.reason = "difficult numeric field prefers sparse residual";
            d.residualTopK = (d.spatialSmoothNorm > 0.35) ? 8192 : 4096;
        } else if (smoothContinuousSpatialFailure) {
            d.reason = "smooth continuous field but clustered-seq likely over-merges";
            d.residualTopK = 4096;
        } else if (sampledStats.lag1Autocorr <= 0.20) {
            d.reason = "severe spatial failure risk";
            d.residualTopK = 32768;
        } else if (sampledStats.lag1Autocorr <= 0.28) {
            d.reason = "medium spatial failure risk";
            d.residualTopK = 8192;
        } else {
            d.reason = "light spatial failure risk";
            d.residualTopK = 4096;
        }
        d.residualDp = 6.0;
        return d;
    }

    d.mode = AutoSpatialRouteMode::GRID4_MULTISCALE;
    d.reason = legacyClusterLike
        ? "legacy clustered-like case now defaults to GRID4 multiscale"
        : (d.temporalLimited
            ? "temporal-limited scalar but GRID4 multiscale still preferred"
            : "continuous scalar defaults to GRID4 multiscale");
    return d;
}

static double computePsnr(double rmse, double peak) {
    return (rmse > 1e-10) ? 20.0 * std::log10(peak / rmse)
                          : std::numeric_limits<double>::infinity();
}

static double percentileFromSorted(const std::vector<float>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    const double clamped = std::clamp(p, 0.0, 1.0);
    const size_t idx = std::min(sorted.size() - 1,
                                static_cast<size_t>(clamped * static_cast<double>(sorted.size() - 1)));
    return sorted[idx];
}

static SampledFieldStats computeSampledFieldStats(
    const Volume4D& volumeSequence,
    int width, int height, int depth, int frames)
{
    constexpr double kTargetValueSamples = 250000.0;
    SampledFieldStats stats;

    const double totalPoints = static_cast<double>(width) * height * depth * frames;
    const int uniformStep = std::max(1, static_cast<int>(std::ceil(std::pow(std::max(1.0, totalPoints / kTargetValueSamples), 0.25))));
    stats.stepX = std::min(width, uniformStep);
    stats.stepY = std::min(height, uniformStep);
    stats.stepZ = std::min(depth, uniformStep);
    stats.stepT = std::min(frames, uniformStep);

    std::vector<float> values;
    std::vector<float> deltas;
    std::vector<float> spatialDiffs;
    values.reserve(static_cast<size_t>(kTargetValueSamples * 1.1));
    deltas.reserve(static_cast<size_t>(kTargetValueSamples * 0.5));
    spatialDiffs.reserve(static_cast<size_t>(kTargetValueSamples * 0.75));

    double sumX = 0.0;
    double sumY = 0.0;
    double sumX2 = 0.0;
    double sumY2 = 0.0;
    double sumXY = 0.0;
    size_t corrCount = 0;

    stats.minValue = std::numeric_limits<double>::infinity();
    stats.maxValue = -std::numeric_limits<double>::infinity();

    for (int t = 0; t < frames; t += stats.stepT) {
        for (int z = 0; z < depth; z += stats.stepZ) {
            for (int y = 0; y < height; y += stats.stepY) {
                for (int x = 0; x < width; x += stats.stepX) {
                    const float v = volumeSequence[t][z][y][x];
                    values.push_back(v);
                    stats.minValue = std::min(stats.minValue, static_cast<double>(v));
                    stats.maxValue = std::max(stats.maxValue, static_cast<double>(v));

                    if (t + stats.stepT < frames) {
                        const float vn = volumeSequence[t + stats.stepT][z][y][x];
                        deltas.push_back(std::abs(vn - v));
                        sumX += v;
                        sumY += vn;
                        sumX2 += static_cast<double>(v) * v;
                        sumY2 += static_cast<double>(vn) * vn;
                        sumXY += static_cast<double>(v) * vn;
                        ++corrCount;
                    }
                    if (x + stats.stepX < width) {
                        spatialDiffs.push_back(std::abs(volumeSequence[t][z][y][x + stats.stepX] - v));
                    }
                    if (y + stats.stepY < height) {
                        spatialDiffs.push_back(std::abs(volumeSequence[t][z][y + stats.stepY][x] - v));
                    }
                    if (z + stats.stepZ < depth) {
                        spatialDiffs.push_back(std::abs(volumeSequence[t][z + stats.stepZ][y][x] - v));
                    }
                }
            }
        }
    }

    if (values.empty()) {
        stats.minValue = 0.0;
        stats.maxValue = 0.0;
        return stats;
    }

    std::sort(values.begin(), values.end());
    stats.q01 = percentileFromSorted(values, 0.01);
    stats.q99 = percentileFromSorted(values, 0.99);
    stats.q005 = percentileFromSorted(values, 0.005);
    stats.q995 = percentileFromSorted(values, 0.995);
    stats.robustRange = std::max(1e-6, stats.q99 - stats.q01);
    stats.tighterRange = std::max(1e-6, stats.q995 - stats.q005);
    stats.valueSamples = values.size();

    if (!deltas.empty()) {
        double deltaSum = 0.0;
        for (float d : deltas) deltaSum += d;
        std::sort(deltas.begin(), deltas.end());
        stats.deltaMean = deltaSum / deltas.size();
        stats.deltaP90 = percentileFromSorted(deltas, 0.90);
        stats.deltaP99 = percentileFromSorted(deltas, 0.99);
        stats.deltaSamples = deltas.size();
    }

    if (!spatialDiffs.empty()) {
        double spatialSum = 0.0;
        for (float d : spatialDiffs) spatialSum += d;
        std::sort(spatialDiffs.begin(), spatialDiffs.end());
        stats.spatialMean = spatialSum / spatialDiffs.size();
        stats.spatialP90 = percentileFromSorted(spatialDiffs, 0.90);
        stats.spatialP99 = percentileFromSorted(spatialDiffs, 0.99);
        stats.spatialSamples = spatialDiffs.size();
    }

    if (corrCount > 1) {
        const double meanX = sumX / corrCount;
        const double meanY = sumY / corrCount;
        const double cov = sumXY / corrCount - meanX * meanY;
        const double varX = std::max(0.0, sumX2 / corrCount - meanX * meanX);
        const double varY = std::max(0.0, sumY2 / corrCount - meanY * meanY);
        const double denom = std::sqrt(std::max(1e-12, varX * varY));
        stats.lag1Autocorr = (denom > 0.0) ? std::clamp(cov / denom, -1.0, 1.0) : 1.0;
    }

    return stats;
}

static double chooseAdaptiveScalarTemporalRange(const SampledFieldStats& stats, double fullRange) {
    // Robust quantile span is stable against outliers, but using it alone can
    // over-tighten turbulent signed scalar fields. Keep a floor tied to the
    // full dynamic range so weakly correlated data is not forced into KF blow-up.
    return std::max(stats.robustRange, fullRange * 0.80);
}

static float chooseAdaptiveScalarEpsAbs(const SampledFieldStats& stats, double fullRange) {
    const double temporalRange = chooseAdaptiveScalarTemporalRange(stats, fullRange);
    return static_cast<float>(std::max(1e-6, temporalRange * (6.0 / 255.0)));
}

static float chooseAdaptiveScalarClusterThr(const SampledFieldStats& stats) {
    const double legacy = 8.0 / 6.0;
    const double normalizedSpatial = stats.spatialP90 / std::max(1e-6, stats.robustRange);
    const double scale = std::clamp(std::pow(0.10 / std::max(1e-4, normalizedSpatial), 0.25), 0.85, 1.35);
    return static_cast<float>(legacy * scale);
}

static float sampleTemporalKfs(const std::vector<Point1D>& kfs, int t) {
    if (kfs.empty()) return 0.0f;
    float recon = static_cast<float>(kfs[0].value);
    for (int k = 0; k + 1 < static_cast<int>(kfs.size()); ++k) {
        if (kfs[k].index <= t && t <= kfs[k + 1].index) {
            double a = (t - kfs[k].index) / static_cast<double>(kfs[k + 1].index - kfs[k].index);
            recon = static_cast<float>(kfs[k].value + a * (kfs[k + 1].value - kfs[k].value));
            break;
        }
    }
    return recon;
}

static std::vector<Point1D> buildTemporalPointKfs(const std::vector<float>& values,
                                                  const FieldProfile& profile) {
    std::vector<Point1D> out;
    if (values.empty()) return out;
    const std::vector<int> kfIdx = detectKeyFrames(values, 0.0, profile);
    out.reserve(std::max<size_t>(1, kfIdx.size()));
    if (kfIdx.empty()) {
        out.emplace_back(0, static_cast<double>(values.front()));
        if (values.size() > 1) {
            out.emplace_back(static_cast<int>(values.size() - 1), static_cast<double>(values.back()));
        }
        return out;
    }
    for (int idx : kfIdx) {
        out.emplace_back(idx, static_cast<double>(values[static_cast<size_t>(idx)]));
    }
    return out;
}

static std::vector<Point1D> buildTemporalPointKfsWeighted(const std::vector<float>& values,
                                                          const FieldProfile& profile,
                                                          const std::vector<float>& frameWeights) {
    std::vector<Point1D> out;
    if (values.empty()) return out;
    const std::vector<int> kfIdx = detectKeyFrames(values, 0.0, profile, frameWeights);
    out.reserve(std::max<size_t>(1, kfIdx.size()));
    if (kfIdx.empty()) {
        out.emplace_back(0, static_cast<double>(values.front()));
        if (values.size() > 1) {
            out.emplace_back(static_cast<int>(values.size() - 1), static_cast<double>(values.back()));
        }
        return out;
    }
    for (int idx : kfIdx) {
        out.emplace_back(idx, static_cast<double>(values[static_cast<size_t>(idx)]));
    }
    return out;
}

static std::array<int, 4> grid4ControlCoordsMain() {
    return {0, 2, 5, 7};
}

static bool isGrid4ControlVoxel(int x, int y, int z) {
    const auto ctrl = grid4ControlCoordsMain();
    const int lx = x & (BT_LEAF_SIZE - 1);
    const int ly = y & (BT_LEAF_SIZE - 1);
    const int lz = z & (BT_LEAF_SIZE - 1);
    const bool cx = std::find(ctrl.begin(), ctrl.end(), lx) != ctrl.end();
    const bool cy = std::find(ctrl.begin(), ctrl.end(), ly) != ctrl.end();
    const bool cz = std::find(ctrl.begin(), ctrl.end(), lz) != ctrl.end();
    return cx && cy && cz;
}

static CompressedVolume4D temporalCompressGrid4ControlsOnly(
    const Volume4D& volumeSequence,
    int width, int height, int depth, int frames,
    const FieldProfile& fieldProfile,
    long long& totalOrig,
    long long& totalKF,
    bool verbose)
{
    CompressedVolume4D compressedVolume(depth);
    for (int z = 0; z < depth; ++z) {
        compressedVolume[z].resize(height);
        for (int y = 0; y < height; ++y) {
            compressedVolume[z][y].resize(width);
        }
    }

    totalOrig = 0;
    totalKF = 0;
    long long totalControls = 0;
    int actualThreads = 1;

#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp master
        {
            actualThreads = omp_get_num_threads();
        }
        #pragma omp for collapse(3) schedule(dynamic, 8) reduction(+:totalOrig,totalKF,totalControls)
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    if (!isGrid4ControlVoxel(x, y, z)) continue;
                    std::vector<float> seq(frames);
                    for (int t = 0; t < frames; ++t) seq[t] = volumeSequence[t][z][y][x];
                    std::vector<int> kfIdx = detectKeyFrames(seq, 2.0, fieldProfile);
                    std::vector<Point1D> kfs;
                    kfs.reserve(kfIdx.size());
                    for (int idx : kfIdx) kfs.push_back(Point1D(idx, seq[idx]));
                    compressedVolume[z][y][x] = std::move(kfs);
                    totalOrig += frames;
                    totalKF += static_cast<long long>(kfIdx.size());
                    ++totalControls;
                }
            }
        }
    }
#else
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (!isGrid4ControlVoxel(x, y, z)) continue;
                std::vector<float> seq(frames);
                for (int t = 0; t < frames; ++t) seq[t] = volumeSequence[t][z][y][x];
                std::vector<int> kfIdx = detectKeyFrames(seq, 2.0, fieldProfile);
                std::vector<Point1D> kfs;
                kfs.reserve(kfIdx.size());
                for (int idx : kfIdx) kfs.push_back(Point1D(idx, seq[idx]));
                compressedVolume[z][y][x] = std::move(kfs);
                totalOrig += frames;
                totalKF += static_cast<long long>(kfIdx.size());
                ++totalControls;
            }
        }
    }
#endif

    if (verbose) {
#ifdef _OPENMP
        printf("  [OpenMP] temporalCompressGrid4ControlsOnly threads=%d\n", actualThreads);
#endif
        const long long totalVoxels = static_cast<long long>(width) * height * depth;
        printf("  GRID4 control-only temporal: controls=%lld / %lld voxels (%.2f%%)\n",
               totalControls, totalVoxels, 100.0 * totalControls / std::max(1LL, totalVoxels));
    }
    return compressedVolume;
}

static std::array<float, 11> fitPoly11Coeffs(const std::array<float, BT_LEAF_VOXELS>& values);

static float sampleLeafSequenceLocal(const LeafBlock& leaf, int localIdx, float t) {
    auto axisPos = [](int localCoord) {
        const float u = (3.0f * localCoord) / 7.0f;
        int i0 = static_cast<int>(std::floor(u));
        i0 = std::clamp(i0, 0, 2);
        return std::array<float, 3>{static_cast<float>(i0), static_cast<float>(i0 + 1), u - i0};
    };
    switch (leaf.mode) {
    case LeafMode::UNIFORM:
        return kfInterp(leaf.codebook[0], t);
    case LeafMode::CLUSTERED:
        return kfInterp(leaf.codebook[leaf.assign[localIdx]], t);
    case LeafMode::GRID4: {
        const int lx = localIdx % BT_LEAF_SIZE;
        const int ly = (localIdx / BT_LEAF_SIZE) % BT_LEAF_SIZE;
        const int lz = localIdx / (BT_LEAF_SIZE * BT_LEAF_SIZE);
        const auto gx = axisPos(lx);
        const auto gy = axisPos(ly);
        const auto gz = axisPos(lz);
        auto sampleNode = [&](int ix, int iy, int iz) -> float {
            const int ctrlIdx = ix + 4 * (iy + 4 * iz);
            return kfInterp(leaf.codebook[ctrlIdx], t);
        };
        const int x0 = static_cast<int>(gx[0]), x1 = static_cast<int>(gx[1]);
        const int y0 = static_cast<int>(gy[0]), y1 = static_cast<int>(gy[1]);
        const int z0 = static_cast<int>(gz[0]), z1 = static_cast<int>(gz[1]);
        const float ax = gx[2], ay = gy[2], az = gz[2];
        const float c000 = sampleNode(x0, y0, z0);
        const float c100 = sampleNode(x1, y0, z0);
        const float c010 = sampleNode(x0, y1, z0);
        const float c110 = sampleNode(x1, y1, z0);
        const float c001 = sampleNode(x0, y0, z1);
        const float c101 = sampleNode(x1, y0, z1);
        const float c011 = sampleNode(x0, y1, z1);
        const float c111 = sampleNode(x1, y1, z1);
        const float c00 = c000 + ax * (c100 - c000);
        const float c10 = c010 + ax * (c110 - c010);
        const float c01 = c001 + ax * (c101 - c001);
        const float c11 = c011 + ax * (c111 - c011);
        const float c0 = c00 + ay * (c10 - c00);
        const float c1 = c01 + ay * (c11 - c01);
        return c0 + az * (c1 - c0);
    }
    case LeafMode::GRID4_RESIDUAL: {
        const int lx = localIdx % BT_LEAF_SIZE;
        const int ly = (localIdx / BT_LEAF_SIZE) % BT_LEAF_SIZE;
        const int lz = localIdx / (BT_LEAF_SIZE * BT_LEAF_SIZE);
        const auto gx = axisPos(lx);
        const auto gy = axisPos(ly);
        const auto gz = axisPos(lz);
        auto sampleNode = [&](int ix, int iy, int iz) -> float {
            const int ctrlIdx = ix + 4 * (iy + 4 * iz);
            return kfInterp(leaf.codebook[ctrlIdx], t);
        };
        const int x0 = static_cast<int>(gx[0]), x1 = static_cast<int>(gx[1]);
        const int y0 = static_cast<int>(gy[0]), y1 = static_cast<int>(gy[1]);
        const int z0 = static_cast<int>(gz[0]), z1 = static_cast<int>(gz[1]);
        const float ax = gx[2], ay = gy[2], az = gz[2];
        const float c000 = sampleNode(x0, y0, z0);
        const float c100 = sampleNode(x1, y0, z0);
        const float c010 = sampleNode(x0, y1, z0);
        const float c110 = sampleNode(x1, y1, z0);
        const float c001 = sampleNode(x0, y0, z1);
        const float c101 = sampleNode(x1, y0, z1);
        const float c011 = sampleNode(x0, y1, z1);
        const float c111 = sampleNode(x1, y1, z1);
        const float c00 = c000 + ax * (c100 - c000);
        const float c10 = c010 + ax * (c110 - c010);
        const float c01 = c001 + ax * (c101 - c001);
        const float c11 = c011 + ax * (c111 - c011);
        const float c0 = c00 + ay * (c10 - c00);
        const float c1 = c01 + ay * (c11 - c01);
        float base = c0 + az * (c1 - c0);
        const int wordIdx = localIdx >> 6;
        const int bitIdx = localIdx & 63;
        if (((leaf.residualMask[wordIdx] >> bitIdx) & 1ull) == 0ull) return base;
        int rank = 0;
        for (int i = 0; i < wordIdx; ++i) rank += __builtin_popcountll(leaf.residualMask[i]);
        if (bitIdx) rank += __builtin_popcountll(leaf.residualMask[wordIdx] & ((1ull << bitIdx) - 1ull));
        if (rank < 0 || rank >= static_cast<int>(leaf.residualCodebook.size())) return base;
        return base + kfInterp(leaf.residualCodebook[rank], t) * leaf.residualScale;
    }
    case LeafMode::POLY11: {
        const int lx = localIdx % BT_LEAF_SIZE;
        const int ly = (localIdx / BT_LEAF_SIZE) % BT_LEAF_SIZE;
        const int lz = localIdx / (BT_LEAF_SIZE * BT_LEAF_SIZE);
        const double x = (2.0 * lx / 7.0) - 1.0;
        const double y = (2.0 * ly / 7.0) - 1.0;
        const double z = (2.0 * lz / 7.0) - 1.0;
        const double basis[11] = {1.0, x, y, z, x * y, x * z, y * z, x * y * z, x * x, y * y, z * z};
        float sum = 0.0f;
        for (int i = 0; i < 11; ++i) sum += static_cast<float>(basis[i]) * kfInterp(leaf.codebook[static_cast<size_t>(i)], t);
        return sum;
    }
    case LeafMode::POLY11_RESIDUAL: {
        const int lx = localIdx % BT_LEAF_SIZE;
        const int ly = (localIdx / BT_LEAF_SIZE) % BT_LEAF_SIZE;
        const int lz = localIdx / (BT_LEAF_SIZE * BT_LEAF_SIZE);
        const double x = (2.0 * lx / 7.0) - 1.0;
        const double y = (2.0 * ly / 7.0) - 1.0;
        const double z = (2.0 * lz / 7.0) - 1.0;
        const double basis[11] = {1.0, x, y, z, x * y, x * z, y * z, x * y * z, x * x, y * y, z * z};
        float base = 0.0f;
        for (int i = 0; i < 11; ++i) base += static_cast<float>(basis[i]) * kfInterp(leaf.codebook[static_cast<size_t>(i)], t);
        const int wordIdx = localIdx >> 6;
        const int bitIdx = localIdx & 63;
        if (((leaf.residualMask[wordIdx] >> bitIdx) & 1ull) == 0ull) return base;
        int rank = 0;
        for (int i = 0; i < wordIdx; ++i) rank += __builtin_popcountll(leaf.residualMask[i]);
        if (bitIdx) rank += __builtin_popcountll(leaf.residualMask[wordIdx] & ((1ull << bitIdx) - 1ull));
        if (rank < 0 || rank >= static_cast<int>(leaf.residualCodebook.size())) return base;
        return base + kfInterp(leaf.residualCodebook[rank], t) * leaf.residualScale;
    }
    default:
        return kfInterp(leaf.codebook[localIdx], t);
    }
}

static float sampleGrid4FromControlValuesLocal(const std::array<float, 64>& controlValues, int localIdx) {
    auto axisPos = [](int localCoord) {
        const float u = (3.0f * localCoord) / 7.0f;
        int i0 = static_cast<int>(std::floor(u));
        i0 = std::clamp(i0, 0, 2);
        return std::array<float, 3>{static_cast<float>(i0), static_cast<float>(i0 + 1), u - i0};
    };
    const int lx = localIdx % BT_LEAF_SIZE;
    const int ly = (localIdx / BT_LEAF_SIZE) % BT_LEAF_SIZE;
    const int lz = localIdx / (BT_LEAF_SIZE * BT_LEAF_SIZE);
    const auto gx = axisPos(lx);
    const auto gy = axisPos(ly);
    const auto gz = axisPos(lz);
    auto sampleNode = [&](int ix, int iy, int iz) -> float {
        return controlValues[static_cast<size_t>(ix + 4 * (iy + 4 * iz))];
    };
    const int x0 = static_cast<int>(gx[0]), x1 = static_cast<int>(gx[1]);
    const int y0 = static_cast<int>(gy[0]), y1 = static_cast<int>(gy[1]);
    const int z0 = static_cast<int>(gz[0]), z1 = static_cast<int>(gz[1]);
    const float ax = gx[2], ay = gy[2], az = gz[2];
    const float c000 = sampleNode(x0, y0, z0);
    const float c100 = sampleNode(x1, y0, z0);
    const float c010 = sampleNode(x0, y1, z0);
    const float c110 = sampleNode(x1, y1, z0);
    const float c001 = sampleNode(x0, y0, z1);
    const float c101 = sampleNode(x1, y0, z1);
    const float c011 = sampleNode(x0, y1, z1);
    const float c111 = sampleNode(x1, y1, z1);
    const float c00 = c000 + ax * (c100 - c000);
    const float c10 = c010 + ax * (c110 - c010);
    const float c01 = c001 + ax * (c101 - c001);
    const float c11 = c011 + ax * (c111 - c011);
    const float c0 = c00 + ay * (c10 - c00);
    const float c1 = c01 + ay * (c11 - c01);
    return c0 + az * (c1 - c0);
}

static std::vector<std::array<float, 64>> buildGrid4ControlValueSeries(const LeafBlock& leaf, int frames) {
    std::vector<std::array<float, 64>> perFrame(static_cast<size_t>(frames));
    for (int t = 0; t < frames; ++t) {
        auto& values = perFrame[static_cast<size_t>(t)];
        for (int i = 0; i < 64; ++i) {
            values[static_cast<size_t>(i)] = kfInterp(leaf.codebook[static_cast<size_t>(i)], static_cast<float>(t));
        }
    }
    return perFrame;
}

static void applyPoly11SpatialAllLeaves(
    BlockTree& tree,
    const CompressedVolume4D& compressedVolume,
    int width, int height, int depth, int frames,
    const FieldProfile& fieldProfile)
{
    int changed = 0;
    const int leafNX = (width + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafNY = (height + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    for (int leafId = 0; leafId < static_cast<int>(tree.leafBlocks.size()); ++leafId) {
        LeafBlock& leaf = tree.leafBlocks[leafId];
        if (leaf.mode != LeafMode::UNIFORM &&
            leaf.mode != LeafMode::CLUSTERED &&
            leaf.mode != LeafMode::DENSE &&
            leaf.mode != LeafMode::GRID4 &&
            leaf.mode != LeafMode::GRID4_RESIDUAL) {
            continue;
        }
        LeafBlock denseLeaf;
        denseLeaf.mode = LeafMode::DENSE;
        denseLeaf.codebook.resize(BT_LEAF_VOXELS);
        const int bx = leafId % leafNX;
        const int by = (leafId / leafNX) % leafNY;
        const int bz = leafId / (leafNX * leafNY);
        const int ox = bx * BT_LEAF_SIZE;
        const int oy = by * BT_LEAF_SIZE;
        const int oz = bz * BT_LEAF_SIZE;
        int idx = 0;
        for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
            for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                for (int lx = 0; lx < BT_LEAF_SIZE; ++lx, ++idx) {
                    const int gx = ox + lx;
                    const int gy = oy + ly;
                    const int gz = oz + lz;
                    if (gx < width && gy < height && gz < depth) {
                        const auto& src = compressedVolume[gz][gy][gx];
                        KFSeq seq;
                        seq.reserve(src.size());
                        for (const auto& p : src) {
                            seq.push_back(KFPoint{static_cast<uint16_t>(p.index), f32_to_f16(p.value)});
                        }
                        denseLeaf.codebook[static_cast<size_t>(idx)] = std::move(seq);
                    }
                }
            }
        }
        LeafBlock polyLeaf;
        polyLeaf.mode = LeafMode::POLY11;
        polyLeaf.codebook.resize(11);
        std::array<std::vector<float>, 11> coeffSeries;
        for (auto& s : coeffSeries) s.resize(frames);
        for (int t = 0; t < frames; ++t) {
            std::array<float, BT_LEAF_VOXELS> vals{};
            for (int i = 0; i < BT_LEAF_VOXELS; ++i) {
                vals[static_cast<size_t>(i)] = kfInterp(denseLeaf.codebook[static_cast<size_t>(i)], static_cast<float>(t));
            }
            const auto coeffs = fitPoly11Coeffs(vals);
            for (int c = 0; c < 11; ++c) coeffSeries[static_cast<size_t>(c)][static_cast<size_t>(t)] = coeffs[static_cast<size_t>(c)];
        }
        for (int c = 0; c < 11; ++c) {
            KFSeq seq;
            seq.reserve(frames);
            for (int t = 0; t < frames; ++t) {
                seq.push_back(KFPoint{static_cast<uint16_t>(t), f32_to_f16(coeffSeries[static_cast<size_t>(c)][static_cast<size_t>(t)])});
            }
            polyLeaf.codebook[static_cast<size_t>(c)] = std::move(seq);
        }
        leaf = std::move(polyLeaf);
        ++changed;
    }
    printf("  POLY11 spatial     : applied=%d\n", changed);
}

static void applyPoly11ResidualHotspots(
    BlockTree& tree,
    const Volume4D& volumeSequence,
    const CompressedVolume4D& compressedVolume,
    const std::vector<LeafErrorStats>& leafStats,
    int width, int height, int depth, int frames,
    int topK,
    double residualThr,
    double residualDpEps)
{
    if (topK <= 0 || leafStats.empty()) return;
    std::vector<LeafErrorStats> sorted = leafStats;
    std::sort(sorted.begin(), sorted.end(),
              [](const LeafErrorStats& a, const LeafErrorStats& b) {
                  if (a.sse != b.sse) return a.sse > b.sse;
                  return a.rmse > b.rmse;
              });
    if (static_cast<int>(sorted.size()) > topK) sorted.resize(topK);

    FieldProfile residualProfile = FieldProfile::makeGeneric(
        DensityProfileParams{static_cast<float>(std::max(0.1, residualDpEps)), 0.0f, 0.0f, 0.0f});
    int changed = 0;
    long long totalActive = 0;

    for (const auto& s : sorted) {
        if (s.leafId < 0 || s.leafId >= static_cast<int>(tree.leafBlocks.size())) continue;
        LeafBlock& leaf = tree.leafBlocks[s.leafId];
        if (leaf.mode != LeafMode::POLY11) continue;

        std::array<uint64_t, 8> mask{};
        std::vector<std::vector<float>> residualSeries(BT_LEAF_VOXELS);
        float maxAbsResidual = 0.0f;
        int activeCount = 0;

        const int ox = s.bx * BT_LEAF_SIZE;
        const int oy = s.by * BT_LEAF_SIZE;
        const int oz = s.bz * BT_LEAF_SIZE;

        for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
            for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                    const int gx = ox + lx;
                    const int gy = oy + ly;
                    const int gz = oz + lz;
                    if (gx >= width || gy >= height || gz >= depth) continue;
                    const int localIdx = lx + BT_LEAF_SIZE * (ly + BT_LEAF_SIZE * lz);
                    float voxelMaxResidual = 0.0f;
                    residualSeries[localIdx].resize(frames);
                    for (int t = 0; t < frames; ++t) {
                        const float raw = volumeSequence[t][gz][gy][gx];
                        const float base = sampleLeafSequenceLocal(leaf, localIdx, static_cast<float>(t));
                        const float residual = raw - base;
                        residualSeries[localIdx][t] = residual;
                        voxelMaxResidual = std::max(voxelMaxResidual, std::abs(residual));
                    }
                    if (voxelMaxResidual > residualThr) {
                        mask[localIdx >> 6] |= (1ull << (localIdx & 63));
                        maxAbsResidual = std::max(maxAbsResidual, voxelMaxResidual);
                        ++activeCount;
                    } else {
                        residualSeries[localIdx].clear();
                    }
                }
            }
        }

        if (activeCount == 0 || maxAbsResidual <= 0.0f) continue;

        leaf.mode = LeafMode::POLY11_RESIDUAL;
        leaf.residualMask = mask;
        leaf.residualScale = maxAbsResidual / 127.0f;
        leaf.residualCodebook.clear();
        leaf.residualCodebook.reserve(activeCount);

        for (int localIdx = 0; localIdx < BT_LEAF_VOXELS; ++localIdx) {
            if (((mask[localIdx >> 6] >> (localIdx & 63)) & 1ull) == 0ull) continue;
            std::vector<float> qvals(frames);
            for (int t = 0; t < frames; ++t) {
                const float q = std::round(residualSeries[localIdx][t] / leaf.residualScale);
                qvals[t] = std::clamp(q, -127.0f, 127.0f);
            }
            std::vector<int> kfIdx = detectKeyFrames(qvals, 0.0, residualProfile);
            KFSeq seq;
            seq.reserve(kfIdx.size());
            for (int idx : kfIdx) {
                seq.push_back(KFPoint{static_cast<uint16_t>(idx), f32_to_f16(qvals[idx])});
            }
            leaf.residualCodebook.push_back(std::move(seq));
        }

        totalActive += activeCount;
        ++changed;
    }

    printf("  POLY11 residual    : requested=%d  applied=%d  avgActive=%.2f  dpEps=%.2f\n",
           topK, changed, changed > 0 ? (static_cast<double>(totalActive) / changed) : 0.0, residualDpEps);
}

static bool resolveLeafIdAtOrigin(const BlockTree& tree, int ox, int oy, int oz, int& leafIdOut) {
    int rx, ry, rz, ix, iy, iz, lx, ly, lz;
    BlockTree::decompose(ox, oy, oz, rx, ry, rz, ix, iy, iz, lx, ly, lz);
    if (rx < 0 || rx >= tree.rootDimX ||
        ry < 0 || ry >= tree.rootDimY ||
        rz < 0 || rz >= tree.rootDimZ) return false;

    const uint32_t rootEntry = tree.rootTable[rx + tree.rootDimX * (ry + tree.rootDimY * rz)];
    if (rootEntry == BT_CHILD_AIR || rootEntry == BT_CHILD_INTERIOR) return false;
    const int internalId = static_cast<int>(rootEntry - BT_CHILD_ID_BASE);
    const InternalNode& node = tree.internalNodes[internalId];
    const int childIdx = ix + BT_INTERNAL_SIZE * (iy + BT_INTERNAL_SIZE * iz);

    uint32_t childEntry;
    if (node.hasBit(childIdx)) {
        childEntry = tree.childList[node.childBase + static_cast<uint64_t>(node.rankBefore(childIdx))];
    } else {
        childEntry = node.defaultVal;
    }
    if (childEntry == BT_CHILD_AIR || childEntry == BT_CHILD_INTERIOR) return false;
    leafIdOut = static_cast<int>(childEntry - BT_CHILD_ID_BASE);
    return true;
}

static std::vector<LeafErrorStats> analyzeLeafErrors(
    const Volume4D& volumeSequence,
    const CompressedVolume4D& compressedVolume,
    const BlockTree& tree,
    int width, int height, int depth, int frames)
{
    const int leafCountX = (width + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafCountY = (height + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafCountZ = (depth + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int totalLeafCount = leafCountX * leafCountY * leafCountZ;

    std::vector<LeafErrorStats> perLeaf(tree.leafBlocks.size());
    std::vector<uint8_t> seen(tree.leafBlocks.size(), 0);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 4)
    #endif
    for (int leafFlat = 0; leafFlat < totalLeafCount; ++leafFlat) {
        const int bx = leafFlat % leafCountX;
        const int by = (leafFlat / leafCountX) % leafCountY;
        const int bz = leafFlat / (leafCountX * leafCountY);
        const int ox = bx * BT_LEAF_SIZE;
        const int oy = by * BT_LEAF_SIZE;
        const int oz = bz * BT_LEAF_SIZE;
        int leafId = -1;
        if (!resolveLeafIdAtOrigin(tree, ox, oy, oz, leafId)) continue;
        if (leafId < 0 || leafId >= static_cast<int>(tree.leafBlocks.size())) continue;

        const LeafBlock& leaf = tree.leafBlocks[leafId];
        LeafErrorStats stats;
        stats.leafId = leafId;
        stats.bx = bx;
        stats.by = by;
        stats.bz = bz;

        std::vector<float> errs;
        errs.reserve(BT_LEAF_VOXELS * frames);
        double errSum = 0.0;
        double errSum2 = 0.0;
        double temporalErr2 = 0.0;
        double rawMin = std::numeric_limits<double>::infinity();
        double rawMax = -std::numeric_limits<double>::infinity();
        float errMax = 0.0f;
        long long count = 0;

        for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
            for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                    const int gx = ox + lx;
                    const int gy = oy + ly;
                    const int gz = oz + lz;
                    if (gx >= width || gy >= height || gz >= depth) continue;
                    const int localIdx = lx + BT_LEAF_SIZE * (ly + BT_LEAF_SIZE * lz);
                    const auto& temporalSeq = compressedVolume[gz][gy][gx];
                    for (int t = 0; t < frames; ++t) {
                        const float raw = volumeSequence[t][gz][gy][gx];
                        const float temporal = sampleTemporalKfs(temporalSeq, t);
                        const float full = sampleLeafSequenceLocal(leaf, localIdx, static_cast<float>(t));
                        const float e = std::abs(full - raw);
                        const float te = std::abs(temporal - raw);
                        errs.push_back(e);
                        errSum += e;
                        errSum2 += static_cast<double>(e) * e;
                        temporalErr2 += static_cast<double>(te) * te;
                        errMax = std::max(errMax, e);
                        rawMin = std::min(rawMin, static_cast<double>(raw));
                        rawMax = std::max(rawMax, static_cast<double>(raw));
                        ++count;
                    }
                }
            }
        }

        if (count == 0) continue;
        std::sort(errs.begin(), errs.end());
        const size_t idx99 = std::min(errs.size() - 1, static_cast<size_t>(0.99 * static_cast<double>(errs.size() - 1)));
        const size_t idx999 = std::min(errs.size() - 1, static_cast<size_t>(0.999 * static_cast<double>(errs.size() - 1)));
        stats.samples = count;
        stats.range = std::max(1e-6, rawMax - rawMin);
        stats.mean = errSum / count;
        stats.max = errMax;
        stats.p99 = errs[idx99];
        stats.p999 = errs[idx999];
        stats.sse = errSum2;
        stats.rmse = std::sqrt(errSum2 / count);
        stats.temporalRmse = std::sqrt(temporalErr2 / count);
        stats.normRmse = stats.rmse / stats.range;
        stats.temporalNormRmse = stats.temporalRmse / stats.range;
        stats.ampVsTemporal = stats.rmse / std::max(1e-8, stats.temporalRmse);
        perLeaf[leafId] = stats;
        seen[leafId] = 1;
    }

    std::vector<LeafErrorStats> compact;
    compact.reserve(perLeaf.size());
    for (size_t i = 0; i < perLeaf.size(); ++i) {
        if (seen[i]) compact.push_back(perLeaf[i]);
    }
    return compact;
}

static std::vector<LeafErrorStats> analyzeLeafErrorsSampled(
    const Volume4D& volumeSequence,
    const CompressedVolume4D& compressedVolume,
    const BlockTree& tree,
    int width, int height, int depth, int frames,
    int sampleStep)
{
    const int step = std::max(1, sampleStep);
    if (step <= 1) {
        return analyzeLeafErrors(volumeSequence, compressedVolume, tree, width, height, depth, frames);
    }

    const int leafCountX = (width + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafCountY = (height + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafCountZ = (depth + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int totalLeafCount = leafCountX * leafCountY * leafCountZ;

    std::vector<LeafErrorStats> perLeaf(tree.leafBlocks.size());
    std::vector<uint8_t> seen(tree.leafBlocks.size(), 0);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int leafFlat = 0; leafFlat < totalLeafCount; ++leafFlat) {
        const int bx = leafFlat % leafCountX;
        const int by = (leafFlat / leafCountX) % leafCountY;
        const int bz = leafFlat / (leafCountX * leafCountY);
        const int ox = bx * BT_LEAF_SIZE;
        const int oy = by * BT_LEAF_SIZE;
        const int oz = bz * BT_LEAF_SIZE;
        int leafId = -1;
        if (!resolveLeafIdAtOrigin(tree, ox, oy, oz, leafId)) continue;
        if (leafId < 0 || leafId >= static_cast<int>(tree.leafBlocks.size())) continue;

        const LeafBlock& leaf = tree.leafBlocks[leafId];
        LeafErrorStats stats;
        stats.leafId = leafId;
        stats.bx = bx;
        stats.by = by;
        stats.bz = bz;

        std::vector<float> errs;
        errs.reserve(std::max(8, (BT_LEAF_VOXELS / step) * std::max(1, frames / step)));
        double errSum = 0.0;
        double errSum2 = 0.0;
        double temporalErr2 = 0.0;
        double rawMin = std::numeric_limits<double>::infinity();
        double rawMax = -std::numeric_limits<double>::infinity();
        float errMax = 0.0f;
        long long count = 0;

        for (int lz = 0; lz < BT_LEAF_SIZE; lz += step) {
            for (int ly = 0; ly < BT_LEAF_SIZE; ly += step) {
                for (int lx = 0; lx < BT_LEAF_SIZE; lx += step) {
                    const int gx = ox + lx;
                    const int gy = oy + ly;
                    const int gz = oz + lz;
                    if (gx >= width || gy >= height || gz >= depth) continue;
                    const int localIdx = lx + BT_LEAF_SIZE * (ly + BT_LEAF_SIZE * lz);
                    const auto& temporalSeq = compressedVolume[gz][gy][gx];
                    for (int t = 0; t < frames; t += step) {
                        const float raw = volumeSequence[t][gz][gy][gx];
                        const float temporal = temporalSeq.empty() ? raw : sampleTemporalKfs(temporalSeq, t);
                        const float full = sampleLeafSequenceLocal(leaf, localIdx, static_cast<float>(t));
                        const float e = std::abs(full - raw);
                        const float te = std::abs(temporal - raw);
                        errs.push_back(e);
                        errSum += e;
                        errSum2 += static_cast<double>(e) * e;
                        temporalErr2 += static_cast<double>(te) * te;
                        errMax = std::max(errMax, e);
                        rawMin = std::min(rawMin, static_cast<double>(raw));
                        rawMax = std::max(rawMax, static_cast<double>(raw));
                        ++count;
                    }
                }
            }
        }

        if (count == 0) continue;
        std::sort(errs.begin(), errs.end());
        const size_t idx99 = std::min(errs.size() - 1, static_cast<size_t>(0.99 * static_cast<double>(errs.size() - 1)));
        const size_t idx999 = std::min(errs.size() - 1, static_cast<size_t>(0.999 * static_cast<double>(errs.size() - 1)));
        stats.samples = count;
        stats.range = std::max(1e-6, rawMax - rawMin);
        stats.mean = errSum / count;
        stats.max = errMax;
        stats.p99 = errs[idx99];
        stats.p999 = errs[idx999];
        stats.sse = errSum2;
        stats.rmse = std::sqrt(errSum2 / count);
        stats.temporalRmse = std::sqrt(temporalErr2 / count);
        stats.normRmse = stats.rmse / stats.range;
        stats.temporalNormRmse = stats.temporalRmse / stats.range;
        stats.ampVsTemporal = stats.rmse / std::max(1e-8, stats.temporalRmse);
        perLeaf[leafId] = stats;
        seen[leafId] = 1;
    }

    std::vector<LeafErrorStats> compact;
    compact.reserve(perLeaf.size());
    for (size_t i = 0; i < perLeaf.size(); ++i) {
        if (seen[i]) compact.push_back(perLeaf[i]);
    }
    return compact;
}

static void writeLeafErrorReport(const std::string& inputFile,
                                 const std::vector<LeafErrorStats>& leafStats,
                                 int topK) {
    if (leafStats.empty()) return;
    namespace fs = std::filesystem;
    fs::create_directories("leaf_error_reports");

    const std::string stem = fs::path(inputFile).stem().string();
    const fs::path reportPath = fs::path("leaf_error_reports") / (stem + "_leaf_error_report.md");
    const fs::path csvPath = fs::path("leaf_error_reports") / (stem + "_leaf_error_report.csv");

    std::vector<LeafErrorStats> byRmse = leafStats;
    std::sort(byRmse.begin(), byRmse.end(),
              [](const LeafErrorStats& a, const LeafErrorStats& b) { return a.rmse > b.rmse; });
    std::vector<LeafErrorStats> bySse = leafStats;
    std::sort(bySse.begin(), bySse.end(),
              [](const LeafErrorStats& a, const LeafErrorStats& b) { return a.sse > b.sse; });

    double totalSse = 0.0;
    std::vector<float> rmses;
    rmses.reserve(leafStats.size());
    for (const auto& s : leafStats) {
        totalSse += s.sse;
        rmses.push_back(static_cast<float>(s.rmse));
    }
    std::sort(rmses.begin(), rmses.end());

    std::ofstream md(reportPath);
    md << "# Leaf Error Report\n\n";
    md << "- dataset: `" << stem << "`\n";
    md << "- leaf count: `" << leafStats.size() << "`\n";
    md << "- leaf RMSE p50/p90/p95/p99: `"
       << percentileFromSorted(rmses, 0.50) << " / "
       << percentileFromSorted(rmses, 0.90) << " / "
       << percentileFromSorted(rmses, 0.95) << " / "
       << percentileFromSorted(rmses, 0.99) << "`\n\n";

    md << "## Top Leaves By RMSE\n\n";
    md << "| Rank | Leaf | Block (bx,by,bz) | RMSE | NormRMSE | TempRMSE | Amp | P99 | P99.9 | Max | Range |\n";
    md << "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (int i = 0; i < std::min<int>(topK, byRmse.size()); ++i) {
        const auto& s = byRmse[i];
        md << "| " << (i + 1)
           << " | " << s.leafId
           << " | (" << s.bx << "," << s.by << "," << s.bz << ")"
           << " | " << s.rmse
           << " | " << s.normRmse
           << " | " << s.temporalRmse
           << " | " << s.ampVsTemporal
           << " | " << s.p99
           << " | " << s.p999
           << " | " << s.max
           << " | " << s.range
           << " |\n";
    }

    md << "\n## Top Leaves By SSE Share\n\n";
    md << "| Rank | Leaf | Block (bx,by,bz) | RMSE | SSE Share | Cum Share | Amp | Range |\n";
    md << "|---|---:|---|---:|---:|---:|---:|---:|\n";
    double cumShare = 0.0;
    for (int i = 0; i < std::min<int>(topK, bySse.size()); ++i) {
        const auto& s = bySse[i];
        const double share = s.sse / std::max(1e-12, totalSse);
        cumShare += share;
        md << "| " << (i + 1)
           << " | " << s.leafId
           << " | (" << s.bx << "," << s.by << "," << s.bz << ")"
           << " | " << s.rmse
           << " | " << share
           << " | " << cumShare
           << " | " << s.ampVsTemporal
           << " | " << s.range
           << " |\n";
    }

    std::ofstream csv(csvPath);
    csv << "leaf_id,bx,by,bz,samples,range,mean,max,p99,p999,rmse,temporal_rmse,norm_rmse,temporal_norm_rmse,amp_vs_temporal,sse\n";
    for (const auto& s : leafStats) {
        csv << s.leafId << ',' << s.bx << ',' << s.by << ',' << s.bz << ','
            << s.samples << ',' << s.range << ',' << s.mean << ',' << s.max << ','
            << s.p99 << ',' << s.p999 << ',' << s.rmse << ',' << s.temporalRmse << ','
            << s.normRmse << ',' << s.temporalNormRmse << ',' << s.ampVsTemporal << ','
            << s.sse << '\n';
    }

    std::cout << "\nLeaf error report written to: " << reportPath.string() << std::endl;
    std::cout << "Leaf error CSV written to   : " << csvPath.string() << std::endl;
}

static uint64_t leafCoordKey(int bx, int by, int bz) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(bx)) << 42) ^
           (static_cast<uint64_t>(static_cast<uint32_t>(by)) << 21) ^
           static_cast<uint64_t>(static_cast<uint32_t>(bz));
}

static std::vector<LeafErrorStats> selectResidualRegionLeaves(
    const std::vector<LeafErrorStats>& leafStats,
    int maxLeaves,
    int seedCount,
    int radius)
{
    if (maxLeaves <= 0 || leafStats.empty()) return {};

    std::vector<LeafErrorStats> sorted = leafStats;
    std::sort(sorted.begin(), sorted.end(),
              [](const LeafErrorStats& a, const LeafErrorStats& b) {
                  if (a.sse != b.sse) return a.sse > b.sse;
                  return a.rmse > b.rmse;
              });
    if (seedCount <= 0 || radius <= 0) {
        if (static_cast<int>(sorted.size()) > maxLeaves) sorted.resize(maxLeaves);
        return sorted;
    }

    std::unordered_map<uint64_t, size_t> coordToSortedIdx;
    coordToSortedIdx.reserve(sorted.size() * 2);
    for (size_t i = 0; i < sorted.size(); ++i) {
        coordToSortedIdx.emplace(leafCoordKey(sorted[i].bx, sorted[i].by, sorted[i].bz), i);
    }

    std::vector<size_t> selected;
    selected.reserve(std::min<int>(maxLeaves, static_cast<int>(sorted.size())));
    std::unordered_set<size_t> seen;
    seen.reserve(selected.capacity() * 2 + 1);

    int usedSeeds = 0;
    for (size_t i = 0; i < sorted.size() && usedSeeds < seedCount && static_cast<int>(selected.size()) < maxLeaves; ++i) {
        if (seen.find(i) != seen.end()) continue;
        ++usedSeeds;
        const LeafErrorStats& seed = sorted[i];
        for (int dist = 0; dist <= radius && static_cast<int>(selected.size()) < maxLeaves; ++dist) {
            for (int dz = -radius; dz <= radius && static_cast<int>(selected.size()) < maxLeaves; ++dz) {
                for (int dy = -radius; dy <= radius && static_cast<int>(selected.size()) < maxLeaves; ++dy) {
                    for (int dx = -radius; dx <= radius && static_cast<int>(selected.size()) < maxLeaves; ++dx) {
                        const int manhattan = std::abs(dx) + std::abs(dy) + std::abs(dz);
                        if (manhattan != dist) continue;
                        const auto it = coordToSortedIdx.find(
                            leafCoordKey(seed.bx + dx, seed.by + dy, seed.bz + dz));
                        if (it == coordToSortedIdx.end()) continue;
                        if (!seen.insert(it->second).second) continue;
                        selected.push_back(it->second);
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < sorted.size() && static_cast<int>(selected.size()) < maxLeaves; ++i) {
        if (!seen.insert(i).second) continue;
        selected.push_back(i);
    }

    std::sort(selected.begin(), selected.end(),
              [&sorted](size_t a, size_t b) {
                  if (sorted[a].sse != sorted[b].sse) return sorted[a].sse > sorted[b].sse;
                  return sorted[a].rmse > sorted[b].rmse;
              });

    std::vector<LeafErrorStats> result;
    result.reserve(selected.size());
    for (size_t idx : selected) result.push_back(sorted[idx]);
    return result;
}

static void applyGrid4DenseHotspotOverride(
    BlockTree& tree,
    const CompressedVolume4D& compressedVolume,
    const std::vector<LeafErrorStats>& leafStats,
    int width, int height, int depth,
    int topK)
{
    if (topK <= 0 || leafStats.empty()) return;

    std::vector<LeafErrorStats> sorted = leafStats;
    std::sort(sorted.begin(), sorted.end(),
              [](const LeafErrorStats& a, const LeafErrorStats& b) {
                  if (a.sse != b.sse) return a.sse > b.sse;
                  return a.rmse > b.rmse;
              });
    if (static_cast<int>(sorted.size()) > topK) sorted.resize(topK);

    int changed = 0;
    for (const auto& s : sorted) {
        if (s.leafId < 0 || s.leafId >= static_cast<int>(tree.leafBlocks.size())) continue;
        LeafBlock& leaf = tree.leafBlocks[s.leafId];
        if (leaf.mode != LeafMode::GRID4) continue;

        leaf.mode = LeafMode::DENSE;
        leaf.assign.clear();
        leaf.codebook.clear();
        leaf.codebook.reserve(BT_LEAF_VOXELS);

        const int ox = s.bx * BT_LEAF_SIZE;
        const int oy = s.by * BT_LEAF_SIZE;
        const int oz = s.bz * BT_LEAF_SIZE;
        for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
            for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                    const int gx = ox + lx;
                    const int gy = oy + ly;
                    const int gz = oz + lz;
                    if (gx < width && gy < height && gz < depth) {
                        const auto& src = compressedVolume[gz][gy][gx];
                        KFSeq seq;
                        seq.reserve(src.size());
                        for (const auto& p : src) {
                            seq.push_back(KFPoint{static_cast<uint16_t>(p.index), f32_to_f16(p.value)});
                        }
                        leaf.codebook.push_back(std::move(seq));
                    } else {
                        leaf.codebook.push_back(KFSeq{});
                    }
                }
            }
        }
        ++changed;
    }
    printf("  GRID4 dense hotspot override: requested=%d  applied=%d\n", topK, changed);
}

static void applyPoly11DenseHotspotOverride(
    BlockTree& tree,
    const CompressedVolume4D& compressedVolume,
    const std::vector<LeafErrorStats>& leafStats,
    int width, int height, int depth,
    int topK)
{
    if (topK <= 0 || leafStats.empty()) return;

    std::vector<LeafErrorStats> sorted = leafStats;
    std::sort(sorted.begin(), sorted.end(),
              [](const LeafErrorStats& a, const LeafErrorStats& b) {
                  if (a.sse != b.sse) return a.sse > b.sse;
                  return a.rmse > b.rmse;
              });
    if (static_cast<int>(sorted.size()) > topK) sorted.resize(topK);

    int changed = 0;
    for (const auto& s : sorted) {
        if (s.leafId < 0 || s.leafId >= static_cast<int>(tree.leafBlocks.size())) continue;
        LeafBlock& leaf = tree.leafBlocks[s.leafId];
        if (leaf.mode != LeafMode::POLY11) continue;

        leaf.mode = LeafMode::DENSE;
        leaf.assign.clear();
        leaf.codebook.clear();
        leaf.codebook.reserve(BT_LEAF_VOXELS);
        leaf.residualMask = {};
        leaf.residualScale = 0.0f;
        leaf.residualCodebook.clear();

        const int ox = s.bx * BT_LEAF_SIZE;
        const int oy = s.by * BT_LEAF_SIZE;
        const int oz = s.bz * BT_LEAF_SIZE;
        for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
            for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                    const int gx = ox + lx;
                    const int gy = oy + ly;
                    const int gz = oz + lz;
                    if (gx < width && gy < height && gz < depth) {
                        const auto& src = compressedVolume[gz][gy][gx];
                        KFSeq seq;
                        seq.reserve(src.size());
                        for (const auto& p : src) {
                            seq.push_back(KFPoint{static_cast<uint16_t>(p.index), f32_to_f16(p.value)});
                        }
                        leaf.codebook.push_back(std::move(seq));
                    } else {
                        leaf.codebook.push_back(KFSeq{});
                    }
                }
            }
        }
        ++changed;
    }
    printf("  POLY11 dense hotspot override: requested=%d  applied=%d\n", topK, changed);
}

static void applyGrid4ResidualHotspots(
    BlockTree& tree,
    const Volume4D& volumeSequence,
    const CompressedVolume4D& compressedVolume,
    const std::vector<LeafErrorStats>& leafStats,
    const FieldProfile& fieldProfile,
    int width, int height, int depth, int frames,
    int topK,
    double residualThr,
    double residualRelThr,
    double residualLocalFloor,
    double residualBandFactor,
    double residualKeepRel,
    bool rankNormalized,
    double residualDpEps)
{
    if (topK <= 0 || leafStats.empty()) return;

    struct ResidualCandidate {
        LeafErrorStats stat;
        double score = 0.0;
        float leafMin = 0.0f;
        float leafMax = 0.0f;
        float leafRange = 0.0f;
        bool skip = false;
    };

    const bool densitySemantic =
        fieldProfile.type == FieldType::DENSITY &&
        fieldProfile.den.render_cutoff > 0.0f;
    const float cutoff = densitySemantic ? fieldProfile.den.render_cutoff : 0.0f;
    const float bgZeroHi = densitySemantic
        ? std::max(1e-9f, fieldProfile.den.bg_zero_ratio * cutoff)
        : 0.0f;
    const float bgConstHi = densitySemantic
        ? std::max(bgZeroHi, fieldProfile.den.bg_const_ratio * cutoff)
        : 0.0f;
    const float visBandLo = densitySemantic
        ? std::max(0.0f, cutoff - fieldProfile.den.cutoff_band)
        : 0.0f;
    const float visBandHi = densitySemantic
        ? (cutoff + fieldProfile.den.cutoff_band)
        : 0.0f;
    const float voxelBandLo = densitySemantic
        ? std::max(bgConstHi, static_cast<float>(std::max(0.0, residualBandFactor)) * cutoff)
        : 0.0f;

    std::vector<ResidualCandidate> sorted;
    sorted.reserve(leafStats.size());
    int semanticLeafSkipped = 0;
    for (const auto& s : leafStats) {
        ResidualCandidate c;
        c.stat = s;
        c.score = rankNormalized && fieldProfile.type == FieldType::DENSITY
            ? std::max(s.normRmse, s.p999 / std::max(1e-6, s.range))
            : s.sse;

        if (densitySemantic) {
            const int ox = s.bx * BT_LEAF_SIZE;
            const int oy = s.by * BT_LEAF_SIZE;
            const int oz = s.bz * BT_LEAF_SIZE;
            float leafMin = std::numeric_limits<float>::infinity();
            float leafMax = -std::numeric_limits<float>::infinity();
            int visHits = 0;
            for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
                const int gz = oz + lz;
                if (gz >= depth) continue;
                for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                    const int gy = oy + ly;
                    if (gy >= height) continue;
                    for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                        const int gx = ox + lx;
                        if (gx >= width) continue;
                        for (int t = 0; t < frames; ++t) {
                            const float raw = volumeSequence[t][gz][gy][gx];
                            leafMin = std::min(leafMin, raw);
                            leafMax = std::max(leafMax, raw);
                            if (raw >= visBandLo && raw <= visBandHi) ++visHits;
                        }
                    }
                }
            }
            c.leafMin = std::isfinite(leafMin) ? leafMin : 0.0f;
            c.leafMax = std::isfinite(leafMax) ? leafMax : 0.0f;
            c.leafRange = std::max(0.0f, c.leafMax - c.leafMin);
            const bool trueBackground = c.leafMax <= bgZeroHi;
            const bool deadQuietNearBg =
                c.leafMax <= bgConstHi &&
                c.leafRange <= std::max(bgZeroHi, 0.25f * bgConstHi) &&
                visHits == 0;
            if (trueBackground || deadQuietNearBg) {
                c.skip = true;
                ++semanticLeafSkipped;
            } else if (c.leafMax <= bgConstHi && visHits == 0) {
                c.score *= 0.25;
            }
        }

        if (!c.skip) sorted.push_back(c);
    }

    std::sort(sorted.begin(), sorted.end(),
              [rankNormalized, &fieldProfile](const ResidualCandidate& a, const ResidualCandidate& b) {
                  if (rankNormalized && fieldProfile.type == FieldType::DENSITY) {
                      if (a.score != b.score) return a.score > b.score;
                      if (a.stat.p999 != b.stat.p999) return a.stat.p999 > b.stat.p999;
                      return a.stat.rmse > b.stat.rmse;
                  }
                  if (a.score != b.score) return a.score > b.score;
                  return a.stat.rmse > b.stat.rmse;
              });
    if (static_cast<int>(sorted.size()) > topK) sorted.resize(topK);

    FieldProfile residualProfile = FieldProfile::makeGeneric(
        DensityProfileParams{static_cast<float>(std::max(0.1, residualDpEps)), 0.0f, 0.0f, 0.0f});
    int changed = 0;
    long long totalActive = 0;
    long long semanticDeadVoxelSkipped = 0;
    long long semanticBandVoxelSkipped = 0;
    long long coarseGoodVoxelSkipped = 0;

    #ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) reduction(+:changed,totalActive,semanticDeadVoxelSkipped,semanticBandVoxelSkipped,coarseGoodVoxelSkipped)
    #endif
    for (int sortedIdx = 0; sortedIdx < static_cast<int>(sorted.size()); ++sortedIdx) {
        const auto& cand = sorted[sortedIdx];
        const auto& s = cand.stat;
        if (s.leafId < 0 || s.leafId >= static_cast<int>(tree.leafBlocks.size())) continue;
        LeafBlock& leaf = tree.leafBlocks[s.leafId];
        if (leaf.mode != LeafMode::GRID4) continue;

        std::array<uint64_t, 8> mask{};
        std::vector<int> activeLocals;
        activeLocals.reserve(BT_LEAF_VOXELS / 2);
        float maxAbsResidual = 0.0f;
        const auto controlSeries = buildGrid4ControlValueSeries(leaf, frames);

        const int ox = s.bx * BT_LEAF_SIZE;
        const int oy = s.by * BT_LEAF_SIZE;
        const int oz = s.bz * BT_LEAF_SIZE;

        for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
            for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                    const int gx = ox + lx;
                    const int gy = oy + ly;
                    const int gz = oz + lz;
                    if (gx >= width || gy >= height || gz >= depth) continue;
                    const int localIdx = lx + BT_LEAF_SIZE * (ly + BT_LEAF_SIZE * lz);
                    float voxelMaxResidual = 0.0f;
                    double voxelSse = 0.0;
                    float voxelRawMin = std::numeric_limits<float>::infinity();
                    float voxelRawMax = -std::numeric_limits<float>::infinity();
                    for (int t = 0; t < frames; ++t) {
                        const float raw = volumeSequence[t][gz][gy][gx];
                        const float base = sampleGrid4FromControlValuesLocal(controlSeries[static_cast<size_t>(t)], localIdx);
                        const float residual = raw - base;
                        voxelMaxResidual = std::max(voxelMaxResidual, std::abs(residual));
                        voxelSse += static_cast<double>(residual) * static_cast<double>(residual);
                        voxelRawMin = std::min(voxelRawMin, raw);
                        voxelRawMax = std::max(voxelRawMax, raw);
                    }
                    const float voxelLocalRange = std::max(0.0f, voxelRawMax - voxelRawMin);
                    const bool deadQuietVoxel = densitySemantic &&
                        voxelRawMax <= bgConstHi &&
                        voxelLocalRange <= std::max(static_cast<float>(residualLocalFloor), 0.25f * bgConstHi);
                    if (deadQuietVoxel) {
                        ++semanticDeadVoxelSkipped;
                        continue;
                    }
                    if (densitySemantic && voxelRawMax < voxelBandLo) {
                        ++semanticBandVoxelSkipped;
                        continue;
                    }
                    const float relDenom = std::max(static_cast<float>(std::max(1e-6, residualLocalFloor)), voxelLocalRange);
                    const float voxelRmse = static_cast<float>(std::sqrt(voxelSse / std::max(1, frames)));
                    const float relResidual = voxelMaxResidual / relDenom;
                    const float relRmse = voxelRmse / relDenom;
                    if (densitySemantic && relRmse <= static_cast<float>(std::max(0.0, residualKeepRel))) {
                        ++coarseGoodVoxelSkipped;
                        continue;
                    }
                    if (voxelMaxResidual > residualThr || relResidual > residualRelThr) {
                        mask[localIdx >> 6] |= (1ull << (localIdx & 63));
                        maxAbsResidual = std::max(maxAbsResidual, voxelMaxResidual);
                        activeLocals.push_back(localIdx);
                    }
                }
            }
        }

        const int activeCount = static_cast<int>(activeLocals.size());
        if (activeCount == 0 || maxAbsResidual <= 0.0f) continue;

        leaf.mode = LeafMode::GRID4_RESIDUAL;
        leaf.residualMask = mask;
        leaf.residualScale = maxAbsResidual / 127.0f;
        leaf.residualCodebook.clear();
        leaf.residualCodebook.reserve(activeCount);

        for (int localIdx : activeLocals) {
            const int lx = localIdx % BT_LEAF_SIZE;
            const int ly = (localIdx / BT_LEAF_SIZE) % BT_LEAF_SIZE;
            const int lz = localIdx / (BT_LEAF_SIZE * BT_LEAF_SIZE);
            const int gx = ox + lx;
            const int gy = oy + ly;
            const int gz = oz + lz;
            std::vector<float> qvals(frames);
            for (int t = 0; t < frames; ++t) {
                const float raw = volumeSequence[t][gz][gy][gx];
                const float base = sampleGrid4FromControlValuesLocal(controlSeries[static_cast<size_t>(t)], localIdx);
                const float residual = raw - base;
                const float q = std::round(residual / leaf.residualScale);
                qvals[t] = std::clamp(q, -127.0f, 127.0f);
            }
            std::vector<int> kfIdx = detectKeyFrames(qvals, 0.0, residualProfile);
            KFSeq seq;
            seq.reserve(kfIdx.size());
            for (int idx : kfIdx) {
                seq.push_back(KFPoint{static_cast<uint16_t>(idx), f32_to_f16(qvals[idx])});
            }
            leaf.residualCodebook.push_back(std::move(seq));
        }

        totalActive += activeCount;
        ++changed;
    }

    printf("  GRID4 residual hotspots: requested=%d  applied=%d  avgActive=%.2f  dpEps=%.2f\n",
           topK, changed, changed > 0 ? (static_cast<double>(totalActive) / changed) : 0.0, residualDpEps);
    if (rankNormalized || residualRelThr > 0.0) {
        printf("    residual rank/trigger : normRank=%s  absThr=%.6f  relThr=%.3f  localFloor=%.6f\n",
               rankNormalized ? "on" : "off",
               residualThr, residualRelThr, residualLocalFloor);
    }
    if (densitySemantic) {
        printf("    density semantic gate : skippedLeaves=%d  skippedDeadVoxels=%lld  skippedBandVoxels=%lld  skippedGoodVoxels=%lld\n",
               semanticLeafSkipped, semanticDeadVoxelSkipped, semanticBandVoxelSkipped, coarseGoodVoxelSkipped);
        printf("                         : bgZero<=%.6f  bgConst<=%.6f  bandLo=%.6f  keepRel<=%.3f\n",
               bgZeroHi, bgConstHi, voxelBandLo, residualKeepRel);
    }
}

static std::vector<std::string> splitCsvLineSimple(const std::string& line) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) cols.push_back(item);
    return cols;
}

static std::vector<HotspotLeafRow> loadHotspotLeafRows(const std::string& csvPath, int topK) {
    std::ifstream in(csvPath);
    if (!in) {
        throw std::runtime_error("cannot open leaf hotspot csv: " + csvPath);
    }
    std::string header;
    if (!std::getline(in, header)) {
        throw std::runtime_error("empty leaf hotspot csv: " + csvPath);
    }

    std::vector<HotspotLeafRow> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto cols = splitCsvLineSimple(line);
        if (cols.size() < 15) continue;
        HotspotLeafRow r;
        r.leafId = std::stoi(cols[0]);
        r.bx = std::stoi(cols[1]);
        r.by = std::stoi(cols[2]);
        r.bz = std::stoi(cols[3]);
        r.range = std::stod(cols[5]);
        r.baselineP99 = std::stod(cols[8]);
        r.baselineP999 = std::stod(cols[9]);
        r.baselineMax = std::stod(cols[7]);
        r.baselineRmse = std::stod(cols[10]);
        r.temporalRmse = std::stod(cols[11]);
        r.ampVsTemporal = std::stod(cols[14]);
        rows.push_back(r);
    }

    std::sort(rows.begin(), rows.end(), [](const HotspotLeafRow& a, const HotspotLeafRow& b) {
        if (a.ampVsTemporal != b.ampVsTemporal) return a.ampVsTemporal > b.ampVsTemporal;
        return a.baselineRmse > b.baselineRmse;
    });
    if (static_cast<int>(rows.size()) > topK) rows.resize(topK);
    return rows;
}

static std::array<int, 4> grid4ControlCoords() {
    return {0, 2, 5, 7};
}

static std::vector<int> regularControlCoords(int dim) {
    switch (dim) {
    case 4: return {0, 2, 5, 7};
    case 6: return {0, 1, 3, 4, 6, 7};
    case 8: return {0, 1, 2, 3, 4, 5, 6, 7};
    default: {
        std::vector<int> coords(static_cast<size_t>(dim), 0);
        if (dim <= 1) return coords;
        int prev = -1;
        for (int i = 0; i < dim; ++i) {
            int c = static_cast<int>(std::round((7.0 * i) / std::max(1, dim - 1)));
            c = std::clamp(c, 0, 7);
            if (c <= prev) c = std::min(7, prev + 1);
            coords[static_cast<size_t>(i)] = c;
            prev = c;
        }
        return coords;
    }
    }
}

static double normLeafCoord(int c) {
    return (2.0 * static_cast<double>(c) / 7.0) - 1.0;
}

static bool invertSmallMatrix(std::vector<double>& a, int n) {
    std::vector<double> inv(static_cast<size_t>(n * n), 0.0);
    for (int i = 0; i < n; ++i) inv[static_cast<size_t>(i * n + i)] = 1.0;
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double best = std::abs(a[static_cast<size_t>(col * n + col)]);
        for (int row = col + 1; row < n; ++row) {
            const double v = std::abs(a[static_cast<size_t>(row * n + col)]);
            if (v > best) {
                best = v;
                pivot = row;
            }
        }
        if (best < 1e-12) return false;
        if (pivot != col) {
            for (int k = 0; k < n; ++k) {
                std::swap(a[static_cast<size_t>(col * n + k)], a[static_cast<size_t>(pivot * n + k)]);
                std::swap(inv[static_cast<size_t>(col * n + k)], inv[static_cast<size_t>(pivot * n + k)]);
            }
        }
        const double diag = a[static_cast<size_t>(col * n + col)];
        for (int k = 0; k < n; ++k) {
            a[static_cast<size_t>(col * n + k)] /= diag;
            inv[static_cast<size_t>(col * n + k)] /= diag;
        }
        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            const double f = a[static_cast<size_t>(row * n + col)];
            if (std::abs(f) < 1e-18) continue;
            for (int k = 0; k < n; ++k) {
                a[static_cast<size_t>(row * n + k)] -= f * a[static_cast<size_t>(col * n + k)];
                inv[static_cast<size_t>(row * n + k)] -= f * inv[static_cast<size_t>(col * n + k)];
            }
        }
    }
    a.swap(inv);
    return true;
}

static const std::array<std::array<double, BT_LEAF_VOXELS>, 8>& poly8PseudoInverse() {
    static const std::array<std::array<double, BT_LEAF_VOXELS>, 8> pinv = [] {
        std::array<std::array<double, BT_LEAF_VOXELS>, 8> out{};
        std::array<std::array<double, 8>, BT_LEAF_VOXELS> basis{};
        int idx = 0;
        for (int z = 0; z < BT_LEAF_SIZE; ++z) {
            for (int y = 0; y < BT_LEAF_SIZE; ++y) {
                for (int x = 0; x < BT_LEAF_SIZE; ++x, ++idx) {
                    const double nx = normLeafCoord(x);
                    const double ny = normLeafCoord(y);
                    const double nz = normLeafCoord(z);
                    basis[idx] = {1.0, nx, ny, nz, nx * ny, nx * nz, ny * nz, nx * ny * nz};
                }
            }
        }
        std::vector<double> ata(8 * 8, 0.0);
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                double s = 0.0;
                for (int i = 0; i < BT_LEAF_VOXELS; ++i) s += basis[i][r] * basis[i][c];
                ata[static_cast<size_t>(r * 8 + c)] = s;
            }
        }
        if (!invertSmallMatrix(ata, 8)) {
            throw std::runtime_error("poly8PseudoInverse: singular system");
        }
        for (int r = 0; r < 8; ++r) {
            for (int i = 0; i < BT_LEAF_VOXELS; ++i) {
                double s = 0.0;
                for (int k = 0; k < 8; ++k) {
                    s += ata[static_cast<size_t>(r * 8 + k)] * basis[i][k];
                }
                out[r][i] = s;
            }
        }
        return out;
    }();
    return pinv;
}

static const std::array<std::array<double, BT_LEAF_VOXELS>, 11>& poly11PseudoInverse() {
    static const std::array<std::array<double, BT_LEAF_VOXELS>, 11> pinv = [] {
        std::array<std::array<double, BT_LEAF_VOXELS>, 11> out{};
        std::array<std::array<double, 11>, BT_LEAF_VOXELS> basis{};
        int idx = 0;
        for (int z = 0; z < BT_LEAF_SIZE; ++z) {
            for (int y = 0; y < BT_LEAF_SIZE; ++y) {
                for (int x = 0; x < BT_LEAF_SIZE; ++x, ++idx) {
                    const double nx = normLeafCoord(x);
                    const double ny = normLeafCoord(y);
                    const double nz = normLeafCoord(z);
                    basis[idx] = {
                        1.0, nx, ny, nz,
                        nx * ny, nx * nz, ny * nz,
                        nx * ny * nz,
                        nx * nx, ny * ny, nz * nz
                    };
                }
            }
        }
        std::vector<double> ata(11 * 11, 0.0);
        for (int r = 0; r < 11; ++r) {
            for (int c = 0; c < 11; ++c) {
                double s = 0.0;
                for (int i = 0; i < BT_LEAF_VOXELS; ++i) s += basis[i][r] * basis[i][c];
                ata[static_cast<size_t>(r * 11 + c)] = s;
            }
        }
        if (!invertSmallMatrix(ata, 11)) {
            throw std::runtime_error("poly11PseudoInverse: singular system");
        }
        for (int r = 0; r < 11; ++r) {
            for (int i = 0; i < BT_LEAF_VOXELS; ++i) {
                double s = 0.0;
                for (int k = 0; k < 11; ++k) {
                    s += ata[static_cast<size_t>(r * 11 + k)] * basis[i][k];
                }
                out[r][i] = s;
            }
        }
        return out;
    }();
    return pinv;
}

static std::array<float, 8> fitPoly8Coeffs(const std::array<float, BT_LEAF_VOXELS>& values) {
    const auto& pinv = poly8PseudoInverse();
    std::array<float, 8> coeffs{};
    for (int r = 0; r < 8; ++r) {
        double s = 0.0;
        for (int i = 0; i < BT_LEAF_VOXELS; ++i) s += pinv[r][i] * static_cast<double>(values[i]);
        coeffs[r] = static_cast<float>(s);
    }
    return coeffs;
}

static std::array<float, 11> fitPoly11Coeffs(const std::array<float, BT_LEAF_VOXELS>& values) {
    const auto& pinv = poly11PseudoInverse();
    std::array<float, 11> coeffs{};
    for (int r = 0; r < 11; ++r) {
        double s = 0.0;
        for (int i = 0; i < BT_LEAF_VOXELS; ++i) s += pinv[r][i] * static_cast<double>(values[i]);
        coeffs[r] = static_cast<float>(s);
    }
    return coeffs;
}

static const std::array<std::array<int, 3>, 11>& poly11ControlPoints() {
    static const std::array<std::array<int, 3>, 11> pts = {{
        {{0, 0, 0}}, {{7, 0, 0}}, {{0, 7, 0}}, {{7, 7, 0}},
        {{0, 0, 7}}, {{7, 0, 7}}, {{0, 7, 7}}, {{7, 7, 7}},
        {{3, 3, 0}}, {{3, 0, 3}}, {{0, 3, 3}}
    }};
    return pts;
}

static const std::array<std::array<double, 11>, 11>& poly11ControlInverse() {
    static const std::array<std::array<double, 11>, 11> inv = [] {
        std::vector<double> A(11 * 11, 0.0);
        const auto& pts = poly11ControlPoints();
        for (int r = 0; r < 11; ++r) {
            const double x = normLeafCoord(pts[static_cast<size_t>(r)][0]);
            const double y = normLeafCoord(pts[static_cast<size_t>(r)][1]);
            const double z = normLeafCoord(pts[static_cast<size_t>(r)][2]);
            const double basis[11] = {
                1.0, x, y, z, x * y, x * z, y * z, x * y * z, x * x, y * y, z * z
            };
            for (int c = 0; c < 11; ++c) {
                A[static_cast<size_t>(r * 11 + c)] = basis[c];
            }
        }
        if (!invertSmallMatrix(A, 11)) {
            throw std::runtime_error("poly11ControlInverse: singular control-point system");
        }
        std::array<std::array<double, 11>, 11> out{};
        for (int r = 0; r < 11; ++r) {
            for (int c = 0; c < 11; ++c) {
                out[static_cast<size_t>(r)][static_cast<size_t>(c)] = A[static_cast<size_t>(r * 11 + c)];
            }
        }
        return out;
    }();
    return inv;
}

static std::array<float, 11> recoverPoly11CoeffsFromControlValues(const std::array<float, 11>& ctrlVals) {
    const auto& inv = poly11ControlInverse();
    std::array<float, 11> coeffs{};
    for (int r = 0; r < 11; ++r) {
        double s = 0.0;
        for (int c = 0; c < 11; ++c) {
            s += inv[static_cast<size_t>(r)][static_cast<size_t>(c)] * static_cast<double>(ctrlVals[static_cast<size_t>(c)]);
        }
        coeffs[static_cast<size_t>(r)] = static_cast<float>(s);
    }
    return coeffs;
}

static float evalPoly8At(const std::array<float, 8>& c, int lx, int ly, int lz) {
    const double x = normLeafCoord(lx);
    const double y = normLeafCoord(ly);
    const double z = normLeafCoord(lz);
    return static_cast<float>(
        c[0] +
        c[1] * x +
        c[2] * y +
        c[3] * z +
        c[4] * x * y +
        c[5] * x * z +
        c[6] * y * z +
        c[7] * x * y * z);
}

static float evalPoly11At(const std::array<float, 11>& c, int lx, int ly, int lz) {
    const double x = normLeafCoord(lx);
    const double y = normLeafCoord(ly);
    const double z = normLeafCoord(lz);
    return static_cast<float>(
        c[0] +
        c[1] * x +
        c[2] * y +
        c[3] * z +
        c[4] * x * y +
        c[5] * x * z +
        c[6] * y * z +
        c[7] * x * y * z +
        c[8] * x * x +
        c[9] * y * y +
        c[10] * z * z);
}

static float sampleGrid4Trilinear(const std::vector<float>& controlValues, int frames,
                                  int t, int lx, int ly, int lz) {
    auto axisPos = [](int localCoord) {
        const float u = (3.0f * localCoord) / 7.0f;
        int i0 = static_cast<int>(std::floor(u));
        i0 = std::clamp(i0, 0, 2);
        return std::array<float, 3>{static_cast<float>(i0), static_cast<float>(i0 + 1), u - i0};
    };
    const auto gx = axisPos(lx);
    const auto gy = axisPos(ly);
    const auto gz = axisPos(lz);

    auto sampleNode = [&](int ix, int iy, int iz) -> float {
        const size_t idx = (((static_cast<size_t>(iz) * 4u + static_cast<size_t>(iy)) * 4u
                           + static_cast<size_t>(ix)) * static_cast<size_t>(frames))
                           + static_cast<size_t>(t);
        return controlValues[idx];
    };

    const int x0 = static_cast<int>(gx[0]), x1 = static_cast<int>(gx[1]);
    const int y0 = static_cast<int>(gy[0]), y1 = static_cast<int>(gy[1]);
    const int z0 = static_cast<int>(gz[0]), z1 = static_cast<int>(gz[1]);
    const float ax = gx[2], ay = gy[2], az = gz[2];

    const float c000 = sampleNode(x0, y0, z0);
    const float c100 = sampleNode(x1, y0, z0);
    const float c010 = sampleNode(x0, y1, z0);
    const float c110 = sampleNode(x1, y1, z0);
    const float c001 = sampleNode(x0, y0, z1);
    const float c101 = sampleNode(x1, y0, z1);
    const float c011 = sampleNode(x0, y1, z1);
    const float c111 = sampleNode(x1, y1, z1);

    const float c00 = c000 + ax * (c100 - c000);
    const float c10 = c010 + ax * (c110 - c010);
    const float c01 = c001 + ax * (c101 - c001);
    const float c11 = c011 + ax * (c111 - c011);
    const float c0 = c00 + ay * (c10 - c00);
    const float c1 = c01 + ay * (c11 - c01);
    return c0 + az * (c1 - c0);
}

static float sampleGrid4TrilinearSingleFrame(const std::array<float, 64>& controlValues,
                                             int lx, int ly, int lz) {
    auto axisPos = [](int localCoord) {
        const float u = (3.0f * localCoord) / 7.0f;
        int i0 = static_cast<int>(std::floor(u));
        i0 = std::clamp(i0, 0, 2);
        return std::array<float, 3>{static_cast<float>(i0), static_cast<float>(i0 + 1), u - i0};
    };
    const auto gx = axisPos(lx);
    const auto gy = axisPos(ly);
    const auto gz = axisPos(lz);
    auto sampleNode = [&](int ix, int iy, int iz) -> float {
        return controlValues[static_cast<size_t>((iz * 4 + iy) * 4 + ix)];
    };

    const int x0 = static_cast<int>(gx[0]), x1 = static_cast<int>(gx[1]);
    const int y0 = static_cast<int>(gy[0]), y1 = static_cast<int>(gy[1]);
    const int z0 = static_cast<int>(gz[0]), z1 = static_cast<int>(gz[1]);
    const float ax = gx[2], ay = gy[2], az = gz[2];

    const float c000 = sampleNode(x0, y0, z0);
    const float c100 = sampleNode(x1, y0, z0);
    const float c010 = sampleNode(x0, y1, z0);
    const float c110 = sampleNode(x1, y1, z0);
    const float c001 = sampleNode(x0, y0, z1);
    const float c101 = sampleNode(x1, y0, z1);
    const float c011 = sampleNode(x0, y1, z1);
    const float c111 = sampleNode(x1, y1, z1);

    const float c00 = c000 + ax * (c100 - c000);
    const float c10 = c010 + ax * (c110 - c010);
    const float c01 = c001 + ax * (c101 - c001);
    const float c11 = c011 + ax * (c111 - c011);
    const float c0 = c00 + ay * (c10 - c00);
    const float c1 = c01 + ay * (c11 - c01);
    return c0 + az * (c1 - c0);
}

static float sampleGrid4FromCompVolume(const CompressedVolume4D& compressedVolume,
                                       int width, int height, int depth,
                                       int bx, int by, int bz,
                                       int lx, int ly, int lz, int t) {
    const auto ctrl = grid4ControlCoords();
    auto axisPos = [](int localCoord) {
        const float u = (3.0f * localCoord) / 7.0f;
        int i0 = static_cast<int>(std::floor(u));
        i0 = std::clamp(i0, 0, 2);
        return std::array<float, 3>{static_cast<float>(i0), static_cast<float>(i0 + 1), u - i0};
    };
    const auto gx = axisPos(lx);
    const auto gy = axisPos(ly);
    const auto gz = axisPos(lz);

    auto sampleNode = [&](int ix, int iy, int iz) -> float {
        const int vx = std::min(bx * BT_LEAF_SIZE + ctrl[ix], width  - 1);
        const int vy = std::min(by * BT_LEAF_SIZE + ctrl[iy], height - 1);
        const int vz = std::min(bz * BT_LEAF_SIZE + ctrl[iz], depth  - 1);
        return sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
    };

    const int x0 = static_cast<int>(gx[0]), x1 = static_cast<int>(gx[1]);
    const int y0 = static_cast<int>(gy[0]), y1 = static_cast<int>(gy[1]);
    const int z0 = static_cast<int>(gz[0]), z1 = static_cast<int>(gz[1]);
    const float ax = gx[2], ay = gy[2], az = gz[2];

    const float c000 = sampleNode(x0, y0, z0);
    const float c100 = sampleNode(x1, y0, z0);
    const float c010 = sampleNode(x0, y1, z0);
    const float c110 = sampleNode(x1, y1, z0);
    const float c001 = sampleNode(x0, y0, z1);
    const float c101 = sampleNode(x1, y0, z1);
    const float c011 = sampleNode(x0, y1, z1);
    const float c111 = sampleNode(x1, y1, z1);

    const float c00 = c000 + ax * (c100 - c000);
    const float c10 = c010 + ax * (c110 - c010);
    const float c01 = c001 + ax * (c101 - c001);
    const float c11 = c011 + ax * (c111 - c011);
    const float c0 = c00 + ay * (c10 - c00);
    const float c1 = c01 + ay * (c11 - c01);
    return c0 + az * (c1 - c0);
}

static float sampleGrid4TrilinearSingleFrameAtFloat(const std::array<float, 64>& controlValues,
                                                    float x, float y, float z) {
    auto axisPos = [](float localCoord) {
        const float u = (3.0f * localCoord) / 7.0f;
        int i0 = static_cast<int>(std::floor(u));
        i0 = std::clamp(i0, 0, 2);
        return std::array<float, 3>{static_cast<float>(i0), static_cast<float>(i0 + 1), u - i0};
    };
    const auto gx = axisPos(x);
    const auto gy = axisPos(y);
    const auto gz = axisPos(z);
    auto sampleNode = [&](int ix, int iy, int iz) -> float {
        return controlValues[static_cast<size_t>((iz * 4 + iy) * 4 + ix)];
    };

    const int x0 = static_cast<int>(gx[0]), x1 = static_cast<int>(gx[1]);
    const int y0 = static_cast<int>(gy[0]), y1 = static_cast<int>(gy[1]);
    const int z0 = static_cast<int>(gz[0]), z1 = static_cast<int>(gz[1]);
    const float ax = gx[2], ay = gy[2], az = gz[2];

    const float c000 = sampleNode(x0, y0, z0);
    const float c100 = sampleNode(x1, y0, z0);
    const float c010 = sampleNode(x0, y1, z0);
    const float c110 = sampleNode(x1, y1, z0);
    const float c001 = sampleNode(x0, y0, z1);
    const float c101 = sampleNode(x1, y0, z1);
    const float c011 = sampleNode(x0, y1, z1);
    const float c111 = sampleNode(x1, y1, z1);

    const float c00 = c000 + ax * (c100 - c000);
    const float c10 = c010 + ax * (c110 - c010);
    const float c01 = c001 + ax * (c101 - c001);
    const float c11 = c011 + ax * (c111 - c011);
    const float c0 = c00 + ay * (c10 - c00);
    const float c1 = c01 + ay * (c11 - c01);
    return c0 + az * (c1 - c0);
}

static float sampleDenseLeafTrilinearSingleFrame(const std::array<float, BT_LEAF_VOXELS>& values,
                                                 float x, float y, float z) {
    const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, BT_LEAF_SIZE - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, BT_LEAF_SIZE - 1);
    const int z0 = std::clamp(static_cast<int>(std::floor(z)), 0, BT_LEAF_SIZE - 1);
    const int x1 = std::min(x0 + 1, BT_LEAF_SIZE - 1);
    const int y1 = std::min(y0 + 1, BT_LEAF_SIZE - 1);
    const int z1 = std::min(z0 + 1, BT_LEAF_SIZE - 1);
    const float ax = std::clamp(x - static_cast<float>(x0), 0.0f, 1.0f);
    const float ay = std::clamp(y - static_cast<float>(y0), 0.0f, 1.0f);
    const float az = std::clamp(z - static_cast<float>(z0), 0.0f, 1.0f);
    auto sampleNode = [&](int ix, int iy, int iz) -> float {
        return values[static_cast<size_t>((iz * BT_LEAF_SIZE + iy) * BT_LEAF_SIZE + ix)];
    };
    const float c000 = sampleNode(x0, y0, z0);
    const float c100 = sampleNode(x1, y0, z0);
    const float c010 = sampleNode(x0, y1, z0);
    const float c110 = sampleNode(x1, y1, z0);
    const float c001 = sampleNode(x0, y0, z1);
    const float c101 = sampleNode(x1, y0, z1);
    const float c011 = sampleNode(x0, y1, z1);
    const float c111 = sampleNode(x1, y1, z1);
    const float c00 = c000 + ax * (c100 - c000);
    const float c10 = c010 + ax * (c110 - c010);
    const float c01 = c001 + ax * (c101 - c001);
    const float c11 = c011 + ax * (c111 - c011);
    const float c0 = c00 + ay * (c10 - c00);
    const float c1 = c01 + ay * (c11 - c01);
    return c0 + az * (c1 - c0);
}

static std::vector<float> uniformControlCoords(int dim) {
    const std::vector<int> ints = regularControlCoords(dim);
    std::vector<float> coords(static_cast<size_t>(ints.size()), 0.0f);
    for (size_t i = 0; i < ints.size(); ++i) {
        coords[i] = static_cast<float>(ints[i]);
    }
    return coords;
}

static float sampleRegularControlGridSingleFrame(const std::vector<float>& controlValues,
                                                 int dim,
                                                 int lx, int ly, int lz) {
    const std::vector<int> coords = regularControlCoords(dim);
    auto axisPos = [&](int localCoord) {
        if (coords.size() <= 1) return std::array<float, 3>{0.0f, 0.0f, 0.0f};
        if (localCoord <= coords.front()) return std::array<float, 3>{0.0f, 1.0f, 0.0f};
        if (localCoord >= coords.back()) {
            const int last = static_cast<int>(coords.size()) - 1;
            return std::array<float, 3>{static_cast<float>(last - 1), static_cast<float>(last), 1.0f};
        }
        for (int i = 0; i + 1 < static_cast<int>(coords.size()); ++i) {
            const int c0 = coords[static_cast<size_t>(i)];
            const int c1 = coords[static_cast<size_t>(i + 1)];
            if (localCoord >= c0 && localCoord <= c1) {
                const float a = (c1 > c0)
                    ? (static_cast<float>(localCoord - c0) / static_cast<float>(c1 - c0))
                    : 0.0f;
                return std::array<float, 3>{static_cast<float>(i), static_cast<float>(i + 1), a};
            }
        }
        const int last = static_cast<int>(coords.size()) - 1;
        return std::array<float, 3>{static_cast<float>(last - 1), static_cast<float>(last), 1.0f};
    };
    const auto gx = axisPos(lx);
    const auto gy = axisPos(ly);
    const auto gz = axisPos(lz);
    auto sampleNode = [&](int ix, int iy, int iz) -> float {
        return controlValues[static_cast<size_t>((iz * dim + iy) * dim + ix)];
    };

    const int x0 = static_cast<int>(gx[0]), x1 = static_cast<int>(gx[1]);
    const int y0 = static_cast<int>(gy[0]), y1 = static_cast<int>(gy[1]);
    const int z0 = static_cast<int>(gz[0]), z1 = static_cast<int>(gz[1]);
    const float ax = gx[2], ay = gy[2], az = gz[2];

    const float c000 = sampleNode(x0, y0, z0);
    const float c100 = sampleNode(x1, y0, z0);
    const float c010 = sampleNode(x0, y1, z0);
    const float c110 = sampleNode(x1, y1, z0);
    const float c001 = sampleNode(x0, y0, z1);
    const float c101 = sampleNode(x1, y0, z1);
    const float c011 = sampleNode(x0, y1, z1);
    const float c111 = sampleNode(x1, y1, z1);

    const float c00 = c000 + ax * (c100 - c000);
    const float c10 = c010 + ax * (c110 - c010);
    const float c01 = c001 + ax * (c101 - c001);
    const float c11 = c011 + ax * (c111 - c011);
    const float c0 = c00 + ay * (c10 - c00);
    const float c1 = c01 + ay * (c11 - c01);
    return c0 + az * (c1 - c0);
}

static float sampleRegularGridFromCompVolumeAtTime(const CompressedVolume4D& compressedVolume,
                                                   int width, int height, int depth,
                                                   int bx, int by, int bz,
                                                   const std::vector<int>& coords,
                                                   int lx, int ly, int lz, int t) {
    const int dim = static_cast<int>(coords.size());
    auto axisPos = [&](int localCoord) {
        if (coords.size() <= 1) return std::array<float, 3>{0.0f, 0.0f, 0.0f};
        if (localCoord <= coords.front()) return std::array<float, 3>{0.0f, 1.0f, 0.0f};
        if (localCoord >= coords.back()) {
            const int last = static_cast<int>(coords.size()) - 1;
            return std::array<float, 3>{static_cast<float>(last - 1), static_cast<float>(last), 1.0f};
        }
        for (int i = 0; i + 1 < static_cast<int>(coords.size()); ++i) {
            const int c0 = coords[static_cast<size_t>(i)];
            const int c1 = coords[static_cast<size_t>(i + 1)];
            if (localCoord >= c0 && localCoord <= c1) {
                const float a = (c1 > c0)
                    ? (static_cast<float>(localCoord - c0) / static_cast<float>(c1 - c0))
                    : 0.0f;
                return std::array<float, 3>{static_cast<float>(i), static_cast<float>(i + 1), a};
            }
        }
        const int last = static_cast<int>(coords.size()) - 1;
        return std::array<float, 3>{static_cast<float>(last - 1), static_cast<float>(last), 1.0f};
    };
    auto sampleNode = [&](int ix, int iy, int iz) -> float {
        const int vx = std::min(bx * BT_LEAF_SIZE + coords[static_cast<size_t>(ix)], width - 1);
        const int vy = std::min(by * BT_LEAF_SIZE + coords[static_cast<size_t>(iy)], height - 1);
        const int vz = std::min(bz * BT_LEAF_SIZE + coords[static_cast<size_t>(iz)], depth - 1);
        return sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
    };
    const auto gx = axisPos(lx);
    const auto gy = axisPos(ly);
    const auto gz = axisPos(lz);
    const int x0 = static_cast<int>(gx[0]), x1 = static_cast<int>(gx[1]);
    const int y0 = static_cast<int>(gy[0]), y1 = static_cast<int>(gy[1]);
    const int z0 = static_cast<int>(gz[0]), z1 = static_cast<int>(gz[1]);
    const float ax = gx[2], ay = gy[2], az = gz[2];
    const float c000 = sampleNode(x0, y0, z0);
    const float c100 = sampleNode(x1, y0, z0);
    const float c010 = sampleNode(x0, y1, z0);
    const float c110 = sampleNode(x1, y1, z0);
    const float c001 = sampleNode(x0, y0, z1);
    const float c101 = sampleNode(x1, y0, z1);
    const float c011 = sampleNode(x0, y1, z1);
    const float c111 = sampleNode(x1, y1, z1);
    const float c00 = c000 + ax * (c100 - c000);
    const float c10 = c010 + ax * (c110 - c010);
    const float c01 = c001 + ax * (c101 - c001);
    const float c11 = c011 + ax * (c111 - c011);
    const float c0 = c00 + ay * (c10 - c00);
    const float c1 = c01 + ay * (c11 - c01);
    return c0 + az * (c1 - c0);
}

static int cutoffStateWithBand(float v, float cutoff, float band) {
    if (cutoff <= 0.0f) return 1;
    if (v < cutoff - band) return 0;
    if (v > cutoff + band) return 2;
    return 1;
}

static void applyGrid4MultiscaleAdaptive(const Volume4D& volumeSequence,
                                         const CompressedVolume4D& compressedVolume,
                                         BlockTree& tree,
                                         int width, int height, int depth, int frames,
                                         const FieldProfile& fieldProfile,
                                         const Grid4MultiscaleOptions& opt) {
    if (!opt.enabled) return;
    if (tree.leafBlocks.empty() || tree.leafOrigins.size() != tree.leafBlocks.size()) return;

    const auto ctrl = grid4ControlCoords();
    const std::vector<int> fine6Coords = regularControlCoords(6);
    const std::vector<int> fine8Coords = regularControlCoords(8);
    const int stepX = 4, stepY = 4, stepZ = 4, stepT = 4;
    DensityProfileParams fineDpParams = fieldProfile.den;
    fineDpParams.eps_abs = std::max(1e-5f, fineDpParams.eps_abs * static_cast<float>(std::max(0.01, opt.fineResidualDpEps)));
    fineDpParams.eps_rel = std::max(0.0f, fineDpParams.eps_rel * static_cast<float>(std::max(0.01, opt.fineResidualDpEps)));
    fineDpParams.gamma_delta = 0.0f;
    if (fieldProfile.type == FieldType::DENSITY && opt.fineResidualCutoffProtect) {
        fineDpParams.render_cutoff = static_cast<float>(opt.fineResidualCutoff);
        fineDpParams.cutoff_band = static_cast<float>(opt.fineResidualCutoffBand);
    }
    if (fineDpParams.render_cutoff > 0.0f && fineDpParams.cutoff_band > 0.0f) {
        fineDpParams.cutoff_temporal_protect = true;
    }
    FieldProfile fineResidualProfile =
        (fieldProfile.type == FieldType::GENERIC)
            ? FieldProfile::makeGeneric(fineDpParams)
            : FieldProfile::makeDensity(fineDpParams);

    std::vector<uint8_t> genericBudgetSelected(tree.leafBlocks.size(), 1);
    if (fieldProfile.type == FieldType::GENERIC &&
        (opt.genericBudgetTopK > 0 || opt.genericBudgetFraction > 0.0)) {
        struct GenericBudgetCand {
            int leafId = -1;
            double score = 0.0;
        };
        std::vector<GenericBudgetCand> cands;
        cands.reserve(tree.leafBlocks.size());
        const int stepX = 4, stepY = 4, stepZ = 4, stepT = 4;
        for (int leafId = 0; leafId < static_cast<int>(tree.leafBlocks.size()); ++leafId) {
            const LeafBlock& leaf = tree.leafBlocks[static_cast<size_t>(leafId)];
            if (leaf.mode != LeafMode::GRID4) continue;
            const auto origin = tree.leafOrigins[static_cast<size_t>(leafId)];
            const int ox = origin[0], oy = origin[1], oz = origin[2];

            double localGridSum2 = 0.0;
            std::vector<float> localGridAbsErrs;
            double rawSum = 0.0, rawSum2 = 0.0;
            int rawCount = 0;
            float localMin = std::numeric_limits<float>::infinity();
            float localMax = -std::numeric_limits<float>::infinity();

            for (int t = 0; t < frames; t += stepT) {
                std::array<float, 64> grid4Values{};
                int cidx = 0;
                for (int gz = 0; gz < 4; ++gz) {
                    for (int gy = 0; gy < 4; ++gy) {
                        for (int gx = 0; gx < 4; ++gx, ++cidx) {
                            const int vx = std::min(ox + ctrl[gx], width - 1);
                            const int vy = std::min(oy + ctrl[gy], height - 1);
                            const int vz = std::min(oz + ctrl[gz], depth - 1);
                            grid4Values[static_cast<size_t>(cidx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                        }
                    }
                }
                for (int lz = 0; lz < BT_LEAF_SIZE; lz += stepZ) {
                    for (int ly = 0; ly < BT_LEAF_SIZE; ly += stepY) {
                        for (int lx = 0; lx < BT_LEAF_SIZE; lx += stepX) {
                            const int vx = ox + lx;
                            const int vy = oy + ly;
                            const int vz = oz + lz;
                            if (vx >= width || vy >= height || vz >= depth) continue;
                            const float raw = volumeSequence[t][vz][vy][vx];
                            const float gridPred = sampleGrid4TrilinearSingleFrame(grid4Values, lx, ly, lz);
                            const double eg = static_cast<double>(gridPred) - raw;
                            localGridSum2 += eg * eg;
                            localGridAbsErrs.push_back(std::abs(gridPred - raw));
                            rawSum += raw;
                            rawSum2 += static_cast<double>(raw) * raw;
                            ++rawCount;
                            localMin = std::min(localMin, raw);
                            localMax = std::max(localMax, raw);
                        }
                    }
                }
            }

            if (rawCount == 0) continue;
            const double localGridRmse = std::sqrt(localGridSum2 / std::max(1, rawCount));
            const double localRange = std::max(1e-6, static_cast<double>(localMax - localMin));
            std::sort(localGridAbsErrs.begin(), localGridAbsErrs.end());
            const double localGridP99 = percentileFromSorted(localGridAbsErrs, 0.99);
            const double coarseRmseNorm = localGridRmse / localRange;
            const double coarseP99Norm = localGridP99 / localRange;
            if (coarseRmseNorm > opt.genericPromoteRmseNorm ||
                coarseP99Norm > opt.genericPromoteP99Norm) {
                const double rmseOver = coarseRmseNorm / std::max(1e-9, opt.genericPromoteRmseNorm);
                const double p99Over = coarseP99Norm / std::max(1e-9, opt.genericPromoteP99Norm);
                cands.push_back(GenericBudgetCand{leafId, std::max(rmseOver, p99Over)});
            }
        }

        std::fill(genericBudgetSelected.begin(), genericBudgetSelected.end(), 0);
        if (!cands.empty()) {
            std::sort(cands.begin(), cands.end(),
                      [](const GenericBudgetCand& a, const GenericBudgetCand& b) {
                          return a.score > b.score;
                      });
            int keep = opt.genericBudgetTopK;
            if (keep <= 0) {
                keep = static_cast<int>(std::ceil(opt.genericBudgetFraction * static_cast<double>(tree.leafBlocks.size())));
            }
            keep = std::clamp(keep, 0, static_cast<int>(cands.size()));
            for (int i = 0; i < keep; ++i) {
                genericBudgetSelected[static_cast<size_t>(cands[static_cast<size_t>(i)].leafId)] = 1;
            }
        }
    }

    long long coarseLeaves = 0, fine6Leaves = 0, fine8Leaves = 0;
    long long fine6CtrlKF = 0, fine8CtrlKF = 0;

#ifdef _OPENMP
    #pragma omp parallel for reduction(+:coarseLeaves,fine6Leaves,fine8Leaves,fine6CtrlKF,fine8CtrlKF) schedule(dynamic, 32)
#endif
    for (int leafId = 0; leafId < static_cast<int>(tree.leafBlocks.size()); ++leafId) {
        LeafBlock& leaf = tree.leafBlocks[static_cast<size_t>(leafId)];
        if (leaf.mode != LeafMode::GRID4) continue;

        const auto origin = tree.leafOrigins[static_cast<size_t>(leafId)];
        const int ox = origin[0], oy = origin[1], oz = origin[2];
        const int bx = ox / BT_LEAF_SIZE;
        const int by = oy / BT_LEAF_SIZE;
        const int bz = oz / BT_LEAF_SIZE;

        struct SampleRec {
            int lx, ly, lz, t;
            float raw;
            float basePred;
            float gridPred;
        };
        std::vector<SampleRec> localSamples;
        localSamples.reserve(static_cast<size_t>(BT_LEAF_VOXELS * frames) / 64 + 16);
        std::vector<float> localGridAbsErrs;
        double localBaseSum2 = 0.0;
        double localGridSum2 = 0.0;
        double rawSum = 0.0, rawSum2 = 0.0;
        int rawCount = 0, visCount = 0, visMismatchCount = 0;
        double gradAccum = 0.0;
        int gradCount = 0;
        float localMin = std::numeric_limits<float>::infinity();
        float localMax = -std::numeric_limits<float>::infinity();

        const bool useVisibilityAware =
            (fieldProfile.type == FieldType::DENSITY || fieldProfile.type == FieldType::GENERIC) &&
            fieldProfile.den.render_cutoff > 0.0f &&
            fieldProfile.den.cutoff_band > 0.0f;
        const float cutoff = fieldProfile.den.render_cutoff;
        const float cutoffBand = fieldProfile.den.cutoff_band;

        for (int t = 0; t < frames; t += stepT) {
            std::array<float, 64> grid4Values{};
            int cidx = 0;
            for (int gz = 0; gz < 4; ++gz) {
                for (int gy = 0; gy < 4; ++gy) {
                    for (int gx = 0; gx < 4; ++gx, ++cidx) {
                        const int vx = std::min(ox + ctrl[gx], width - 1);
                        const int vy = std::min(oy + ctrl[gy], height - 1);
                        const int vz = std::min(oz + ctrl[gz], depth - 1);
                        grid4Values[static_cast<size_t>(cidx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                    }
                }
            }
            for (int lz = 0; lz < BT_LEAF_SIZE; lz += stepZ) {
                for (int ly = 0; ly < BT_LEAF_SIZE; ly += stepY) {
                    for (int lx = 0; lx < BT_LEAF_SIZE; lx += stepX) {
                        const int vx = ox + lx;
                        const int vy = oy + ly;
                        const int vz = oz + lz;
                        if (vx >= width || vy >= height || vz >= depth) continue;
                        const float raw = volumeSequence[t][vz][vy][vx];
                        const float basePred = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                        const float gridPred = sampleGrid4TrilinearSingleFrame(grid4Values, lx, ly, lz);
                        localSamples.push_back({lx, ly, lz, t, raw, basePred, gridPred});
                        const double eb = static_cast<double>(basePred) - raw;
                        const double eg = static_cast<double>(gridPred) - raw;
                        localBaseSum2 += eb * eb;
                        localGridSum2 += eg * eg;
                        localGridAbsErrs.push_back(std::abs(gridPred - raw));
                        rawSum += raw;
                        rawSum2 += static_cast<double>(raw) * raw;
                        ++rawCount;
                        localMin = std::min(localMin, raw);
                        localMax = std::max(localMax, raw);
                        if (useVisibilityAware) {
                            if (raw >= cutoff - cutoffBand && raw <= cutoff + cutoffBand) ++visCount;
                            if (cutoffStateWithBand(raw, cutoff, cutoffBand) != cutoffStateWithBand(gridPred, cutoff, cutoffBand)) {
                                ++visMismatchCount;
                            }
                        }
                    }
                }
            }
            for (int lz = 0; lz + stepZ < BT_LEAF_SIZE; lz += stepZ) {
                for (int ly = 0; ly + stepY < BT_LEAF_SIZE; ly += stepY) {
                    for (int lx = 0; lx + stepX < BT_LEAF_SIZE; lx += stepX) {
                        const int vx = ox + lx;
                        const int vy = oy + ly;
                        const int vz = oz + lz;
                        if (vx + stepX >= width || vy + stepY >= height || vz + stepZ >= depth) continue;
                        const float v0 = volumeSequence[t][vz][vy][vx];
                        const float vx1 = volumeSequence[t][vz][vy][vx + stepX];
                        const float vy1 = volumeSequence[t][vz][vy + stepY][vx];
                        const float vz1 = volumeSequence[t][vz + stepZ][vy][vx];
                        gradAccum += std::abs(v0 - vx1);
                        gradAccum += std::abs(v0 - vy1);
                        gradAccum += std::abs(v0 - vz1);
                        gradCount += 3;
                    }
                }
            }
        }

        if (localSamples.empty()) {
            ++coarseLeaves;
            continue;
        }

        const double localBaseRmse = std::sqrt(localBaseSum2 / std::max<size_t>(1, localSamples.size()));
        const double localGridRmse = std::sqrt(localGridSum2 / std::max<size_t>(1, localSamples.size()));
        const double localRange = std::max(1e-6, static_cast<double>(localMax - localMin));
        std::sort(localGridAbsErrs.begin(), localGridAbsErrs.end());
        const double localGridP99 = percentileFromSorted(localGridAbsErrs, 0.99);
        const double rawMean = rawSum / std::max(1, rawCount);
        const double rawVar = std::max(0.0, rawSum2 / std::max(1, rawCount) - rawMean * rawMean);
        const double rawStd = std::sqrt(rawVar);
        const double visibilityFrac = static_cast<double>(visCount) / std::max<size_t>(1, localSamples.size());
        const double visibilityMismatchFrac = static_cast<double>(visMismatchCount) / std::max<size_t>(1, localSamples.size());
        const double gradMean = gradAccum / std::max(1, gradCount);

        double rmseTerm = 0.0;
        if (opt.scoreRmseTargetRel > 0.0) {
            rmseTerm = std::clamp((localGridRmse / localRange) /
                                      std::max(1e-6, opt.scoreRmseTargetRel),
                                  0.0, 1.0);
        } else {
            rmseTerm = std::clamp((localGridRmse / std::max(localBaseRmse, 1e-6) - 1.0) / 4.0, 0.0, 1.0);
        }
        const double p99Term = std::clamp((localGridP99 / localRange) / 0.25, 0.0, 1.0);
        const double visBandTerm = std::clamp(visibilityFrac / 0.25, 0.0, 1.0);
        const double visMismatchTerm = std::clamp(visibilityMismatchFrac / 0.10, 0.0, 1.0);
        const double visTerm = std::max(visBandTerm, visMismatchTerm);
        const double gradTerm = std::clamp((gradMean / localRange) / 0.25, 0.0, 1.0);
        const double varTerm = std::clamp((rawStd / localRange) / 0.5, 0.0, 1.0);
        const double score = opt.scoreWRmse * rmseTerm + opt.scoreWP99 * p99Term + opt.scoreWVis * visTerm + opt.scoreWGrad * gradTerm + opt.scoreWVar * varTerm;

        int chosenLevel = 0;
        std::vector<std::vector<float>> fine6ResidualSeries;
        std::vector<std::vector<float>> fine8ResidualSeries;
        double localFine6Rmse = std::numeric_limits<double>::infinity();
        double localFine8Rmse = std::numeric_limits<double>::infinity();
        double localFine6VisMismatch = std::numeric_limits<double>::infinity();
        double localFine8VisMismatch = std::numeric_limits<double>::infinity();

        auto buildResidualSeries = [&](const std::vector<int>& coords,
                                       std::vector<std::vector<float>>& outResidualSeries,
                                       std::vector<std::vector<float>>* outRawSeries) {
            const int dim = static_cast<int>(coords.size());
            outResidualSeries.assign(static_cast<size_t>(dim * dim * dim),
                                     std::vector<float>(static_cast<size_t>(frames), 0.0f));
            if (outRawSeries) {
                outRawSeries->assign(static_cast<size_t>(dim * dim * dim),
                                     std::vector<float>(static_cast<size_t>(frames), 0.0f));
            }
            for (int t = 0; t < frames; ++t) {
                std::array<float, 64> grid4Values{};
                int cidx = 0;
                for (int gz = 0; gz < 4; ++gz) {
                    for (int gy = 0; gy < 4; ++gy) {
                        for (int gx = 0; gx < 4; ++gx, ++cidx) {
                            const int vx = std::min(ox + ctrl[gx], width - 1);
                            const int vy = std::min(oy + ctrl[gy], height - 1);
                            const int vz = std::min(oz + ctrl[gz], depth - 1);
                            grid4Values[static_cast<size_t>(cidx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                        }
                    }
                }
                int fidx = 0;
                for (int gz = 0; gz < dim; ++gz) {
                    for (int gy = 0; gy < dim; ++gy) {
                        for (int gx = 0; gx < dim; ++gx, ++fidx) {
                            const int lx = coords[static_cast<size_t>(gx)];
                            const int ly = coords[static_cast<size_t>(gy)];
                            const int lz = coords[static_cast<size_t>(gz)];
                            const int vx = std::min(ox + lx, width - 1);
                            const int vy = std::min(oy + ly, height - 1);
                            const int vz = std::min(oz + lz, depth - 1);
                            const float raw = volumeSequence[t][vz][vy][vx];
                            const float coarse = sampleGrid4TrilinearSingleFrameAtFloat(
                                grid4Values, static_cast<float>(lx), static_cast<float>(ly), static_cast<float>(lz));
                            outResidualSeries[static_cast<size_t>(fidx)][static_cast<size_t>(t)] = raw - coarse;
                            if (outRawSeries) {
                                (*outRawSeries)[static_cast<size_t>(fidx)][static_cast<size_t>(t)] = raw;
                            }
                        }
                    }
                }
            }
        };

        auto evalResidualSeries = [&](const std::vector<std::vector<float>>& series,
                                      int fineDim,
                                      double& outRmse,
                                      double& outVisMismatch) {
            double sum2 = 0.0;
            int visMismatch = 0;
            std::vector<float> controlValues(static_cast<size_t>(fineDim * fineDim * fineDim), 0.0f);
            for (const auto& s : localSamples) {
                for (int i = 0; i < fineDim * fineDim * fineDim; ++i) {
                    controlValues[static_cast<size_t>(i)] = series[static_cast<size_t>(i)][static_cast<size_t>(s.t)];
                }
                const float pred = s.gridPred + sampleRegularControlGridSingleFrame(controlValues, fineDim, s.lx, s.ly, s.lz);
                const double e = static_cast<double>(pred) - s.raw;
                sum2 += e * e;
                if (useVisibilityAware &&
                    cutoffStateWithBand(s.raw, cutoff, cutoffBand) != cutoffStateWithBand(pred, cutoff, cutoffBand)) {
                    ++visMismatch;
                }
            }
            outRmse = std::sqrt(sum2 / std::max<size_t>(1, localSamples.size()));
            outVisMismatch = static_cast<double>(visMismatch) / std::max<size_t>(1, localSamples.size());
        };

        auto estimateResidualKfPerCtrl = [&](const std::vector<std::vector<float>>& rawSeries) {
            if (rawSeries.empty()) return 0.0;
            long long totalKf = 0;
            for (const auto& seq : rawSeries) {
                totalKf += static_cast<long long>(detectKeyFrames(seq, 0.0, fineResidualProfile).size());
            }
            return static_cast<double>(totalKf) / static_cast<double>(rawSeries.size());
        };

        const bool useGenericImprovementPolicy =
            (fieldProfile.type == FieldType::GENERIC);
        const bool genericBudgetAllowsFine =
            !useGenericImprovementPolicy ||
            genericBudgetSelected[static_cast<size_t>(leafId)] != 0;

        const double coarseRmseNorm = localGridRmse / localRange;
        const double coarseP99Norm = localGridP99 / localRange;
        const bool genericNeedsFine6 =
            genericBudgetAllowsFine &&
            (coarseRmseNorm > opt.genericPromoteRmseNorm ||
             coarseP99Norm > opt.genericPromoteP99Norm);

        const bool scoreNeedsFine6 =
            score >= opt.scoreT1 || visibilityMismatchFrac > opt.visPromoteFine6;

        if ((useGenericImprovementPolicy && genericNeedsFine6) ||
            (!useGenericImprovementPolicy && scoreNeedsFine6)) {
            std::vector<std::vector<float>> fine6RawSeries;
            buildResidualSeries(fine6Coords, fine6ResidualSeries, &fine6RawSeries);
            evalResidualSeries(fine6ResidualSeries, 6, localFine6Rmse, localFine6VisMismatch);
            const double fine6AbsErrsImprove =
                (localGridRmse > 1e-12) ? (localFine6Rmse / localGridRmse) : 1.0;
            const double fine6KfPerCtrl =
                useGenericImprovementPolicy ? estimateResidualKfPerCtrl(fine6RawSeries) : 0.0;
            const bool genericAcceptFine6 =
                !useGenericImprovementPolicy ||
                ((localFine6Rmse < localGridRmse * opt.genericFine6Improve) &&
                 (fine6KfPerCtrl <= opt.genericFineCostKfPerCtrl ||
                  localFine6Rmse < localGridRmse * opt.genericFineCostImprove));
            if (genericAcceptFine6) {
                chosenLevel = 6;
            }

            const bool smokeScoreNeedsFine8 =
                score >= opt.scoreT2 &&
                (localFine6VisMismatch > opt.visPromoteFine8 ||
                 localFine6Rmse < std::numeric_limits<double>::infinity());
            const bool scoreNeedsFine8 =
                smokeScoreNeedsFine8;
            const bool genericNeedsFine8 =
                useGenericImprovementPolicy &&
                chosenLevel == 6 &&
                fine6AbsErrsImprove > 0.92 &&
                (coarseRmseNorm > opt.genericPromoteRmseNorm * 1.8 ||
                 coarseP99Norm > opt.genericPromoteP99Norm * 1.4);

            if ((useGenericImprovementPolicy && genericNeedsFine8) ||
                (!useGenericImprovementPolicy && scoreNeedsFine8)) {
                std::vector<std::vector<float>> fine8RawSeries;
                buildResidualSeries(fine8Coords, fine8ResidualSeries, &fine8RawSeries);
                evalResidualSeries(fine8ResidualSeries, 8, localFine8Rmse, localFine8VisMismatch);
                const bool betterRmse = (localFine8Rmse < localFine6Rmse * opt.fine8Improve);
                const bool betterVisibility =
                    useVisibilityAware &&
                    localFine6VisMismatch > 0.0 &&
                    localFine8VisMismatch < localFine6VisMismatch * opt.visFine8Improve;
                const double fine8KfPerCtrl =
                    useGenericImprovementPolicy ? estimateResidualKfPerCtrl(fine8RawSeries) : 0.0;
                const bool genericAcceptFine8 =
                    useGenericImprovementPolicy &&
                    (localFine8Rmse < localFine6Rmse * opt.genericFine8Improve) &&
                    (fine8KfPerCtrl <= opt.genericFineCostKfPerCtrl ||
                     localFine8Rmse < localFine6Rmse * opt.genericFineCostImprove);
                if ((!useGenericImprovementPolicy && (betterRmse || betterVisibility)) ||
                    (useGenericImprovementPolicy && genericAcceptFine8)) {
                    chosenLevel = 8;
                }
            }
        }

        if (chosenLevel == 0) {
            ++coarseLeaves;
            continue;
        }

        leaf.mode = LeafMode::GRID4_MULTISCALE;
        leaf.fineGridDim = static_cast<uint8_t>(chosenLevel);
        leaf.assign.clear();
        leaf.residualMask = {};
        leaf.residualCodebook.clear();

        const auto& chosenSeries = (chosenLevel == 6) ? fine6ResidualSeries : fine8ResidualSeries;
        std::vector<std::vector<float>> chosenRawSeries;
        if (chosenLevel == 6) {
            buildResidualSeries(fine6Coords, fine6ResidualSeries, &chosenRawSeries);
        } else {
            buildResidualSeries(fine8Coords, fine8ResidualSeries, &chosenRawSeries);
        }
        const bool densityFineMask =
            fieldProfile.type == FieldType::DENSITY &&
            fieldProfile.den.render_cutoff > 0.0f &&
            (opt.fineCtrlBandFactor > 0.0 || opt.fineCtrlKeepRel > 0.0);
        const float fineCutoff = densityFineMask ? fieldProfile.den.render_cutoff : 0.0f;
        const float fineBandLo = densityFineMask
            ? std::max(
                  std::max(fieldProfile.den.bg_const_ratio * fineCutoff, 1e-9f),
                  static_cast<float>(std::max(0.0, opt.fineCtrlBandFactor)) * fineCutoff)
            : 0.0f;
        const float fineRelFloor = densityFineMask
            ? std::max(1e-6f, 0.25f * fineCutoff)
            : 1e-6f;
        float maxAbsResidual = 0.0f;
        std::vector<uint8_t> keepFlags(chosenSeries.size(), 1);
        if (densityFineMask) {
            for (size_t seqIdx = 0; seqIdx < chosenSeries.size(); ++seqIdx) {
                const auto& residualSeq = chosenSeries[seqIdx];
                const auto& rawSeq = chosenRawSeries[seqIdx];
                float rawMax = 0.0f;
                float rawMin = std::numeric_limits<float>::infinity();
                float residualSum2 = 0.0f;
                for (size_t ti = 0; ti < residualSeq.size(); ++ti) {
                    const float raw = rawSeq[ti];
                    rawMax = std::max(rawMax, raw);
                    rawMin = std::min(rawMin, raw);
                    residualSum2 += residualSeq[ti] * residualSeq[ti];
                }
                const float ctrlRange = std::max(fineRelFloor, rawMax - rawMin);
                const float ctrlRmse = std::sqrt(residualSum2 / std::max<size_t>(1, residualSeq.size()));
                const float ctrlRelRmse = ctrlRmse / ctrlRange;
                if (rawMax < fineBandLo || ctrlRelRmse <= static_cast<float>(opt.fineCtrlKeepRel)) {
                    keepFlags[seqIdx] = 0;
                    continue;
                }
            }
        }
        for (size_t seqIdx = 0; seqIdx < chosenSeries.size(); ++seqIdx) {
            if (!keepFlags[seqIdx]) continue;
            const auto& seq = chosenSeries[seqIdx];
            for (float v : seq) maxAbsResidual = std::max(maxAbsResidual, std::abs(v));
            const int wordIdx = static_cast<int>(seqIdx >> 6);
            const int bitIdx = static_cast<int>(seqIdx & 63);
            leaf.residualMask[static_cast<size_t>(wordIdx)] |= (1ull << bitIdx);
        }
        if (maxAbsResidual <= 1e-12f) {
            ++coarseLeaves;
            leaf.mode = LeafMode::GRID4;
            leaf.fineGridDim = 0;
            leaf.residualScale = 0.0f;
            leaf.residualMask = {};
            leaf.residualCodebook.clear();
            continue;
        }
        leaf.residualScale = (maxAbsResidual > 1e-12f) ? maxAbsResidual : 1.0f;
        leaf.residualCodebook.reserve(chosenSeries.size());
        for (size_t seqIdx = 0; seqIdx < chosenSeries.size(); ++seqIdx) {
            if (!keepFlags[seqIdx]) continue;
            const auto& seq = chosenSeries[seqIdx];
            std::vector<float> qvals(static_cast<size_t>(frames), 0.0f);
            for (int t = 0; t < frames; ++t) {
                const float q = std::clamp(seq[static_cast<size_t>(t)] / leaf.residualScale, -1.0f, 1.0f);
                qvals[static_cast<size_t>(t)] = q;
            }
            const std::vector<int> kfIdx = detectKeyFrames(chosenRawSeries[seqIdx], 0.0, fineResidualProfile);
            KFSeq kfSeq;
            kfSeq.reserve(kfIdx.size());
            for (int idx : kfIdx) {
                kfSeq.push_back(KFPoint{static_cast<uint16_t>(idx), f32_to_f16(qvals[static_cast<size_t>(idx)])});
            }
            leaf.residualCodebook.push_back(std::move(kfSeq));
        }

        if (chosenLevel == 6) {
            ++fine6Leaves;
            for (const auto& seq : leaf.residualCodebook) fine6CtrlKF += static_cast<long long>(seq.size());
        } else {
            ++fine8Leaves;
            for (const auto& seq : leaf.residualCodebook) fine8CtrlKF += static_cast<long long>(seq.size());
        }
    }

    printf("  GRID4 multiscale : coarse/fine6/fine8 = %lld / %lld / %lld\n",
           coarseLeaves, fine6Leaves, fine8Leaves);
    printf("  GRID4 multiscale : fine KF 6/8 = %lld / %lld\n", fine6CtrlKF, fine8CtrlKF);
}

static int runGrid4FineSampledFullProbe(const std::string& inputFile,
                                        const Volume4D& volumeSequence,
                                        const CompressedVolume4D& compressedVolume,
                                        int width, int height, int depth, int frames,
                                        const FieldProfile& fieldProfile,
                                        bool enableBackgroundElision,
                                        bool enableBlockAwareCluster,
                                        bool enableBudgetAwareCluster,
                                        bool enableGuardedMedoidCluster,
                                        bool enableValidateFallback,
                                        bool enableHotspotSecondPass,
                                        double clusterThr,
                                        double dataMin,
                                        double dataMax,
                                        bool useDataRangeForPsnr,
                                        const SpatialProbeOptions& spatialProbe,
                                        int fineDim,
                                        bool hotspotOnly) {
    namespace fs = std::filesystem;
    fs::create_directories("spatial_probe_reports");

    BlockTree grid4Tree;
    grid4Tree.build(compressedVolume, width, height, depth, frames,
                    -1.0, clusterThr, fieldProfile, enableBackgroundElision,
                    enableBlockAwareCluster, enableBudgetAwareCluster, enableGuardedMedoidCluster, true, false,
                    &volumeSequence, enableValidateFallback, enableHotspotSecondPass);
    grid4Tree.flattenLeaves();

    std::vector<uint8_t> hotspotLeaf;
    std::vector<LeafErrorStats> leafStats;
    if (hotspotOnly) {
        leafStats = analyzeLeafErrors(volumeSequence, compressedVolume, grid4Tree, width, height, depth, frames);
        std::sort(leafStats.begin(), leafStats.end(),
                  [](const LeafErrorStats& a, const LeafErrorStats& b) {
                      if (a.sse != b.sse) return a.sse > b.sse;
                      return a.rmse > b.rmse;
                  });
        if (static_cast<int>(leafStats.size()) > spatialProbe.topK) {
            leafStats.resize(spatialProbe.topK);
        }
        hotspotLeaf.assign(grid4Tree.leafBlocks.size(), 0);
        for (const auto& s : leafStats) {
            if (s.leafId >= 0 && s.leafId < static_cast<int>(hotspotLeaf.size())) {
                hotspotLeaf[static_cast<size_t>(s.leafId)] = 1;
            }
        }
    }

    const std::vector<float> fineCoords = uniformControlCoords(fineDim);
    const int stepX = 4, stepY = 4, stepZ = 4, stepT = 4;
    std::vector<float> baselineErrs;
    std::vector<float> gridErrs;
    std::vector<float> fineErrs;
    baselineErrs.reserve(static_cast<size_t>((width / stepX + 1) * (height / stepY + 1) * (depth / stepZ + 1) * (frames / stepT + 1)));
    gridErrs.reserve(baselineErrs.capacity());
    fineErrs.reserve(baselineErrs.capacity());
    double baseSum2 = 0.0, gridSum2 = 0.0, fineSum2 = 0.0;
    long long samples = 0;
    long long fineLeavesUsed = 0;
    long long coarseCtrlKF = 0;
    long long fineCtrlKF = 0;

    const int leafNX = (width + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafNY = (height + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafNZ = (depth + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const auto ctrl = grid4ControlCoords();

    for (int bz = 0; bz < leafNZ; ++bz) {
        for (int by = 0; by < leafNY; ++by) {
            for (int bx = 0; bx < leafNX; ++bx) {
                const int leafId = (bz * leafNY + by) * leafNX + bx;
                const bool useFine = !hotspotOnly || (leafId >= 0 && leafId < static_cast<int>(hotspotLeaf.size()) &&
                                                      hotspotLeaf[static_cast<size_t>(leafId)] != 0);
                if (useFine) ++fineLeavesUsed;
                std::vector<std::vector<float>> fineResidualSeries;
                if (useFine) {
                    fineResidualSeries.assign(static_cast<size_t>(fineDim * fineDim * fineDim),
                                              std::vector<float>(static_cast<size_t>(frames), 0.0f));
                }

                for (int t = 0; t < frames; t += stepT) {
                    std::array<float, BT_LEAF_VOXELS> leafValues{};
                    std::array<float, 64> grid4Values{};
                    int idx = 0;
                    for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; ++lx, ++idx) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx < width && vy < height && vz < depth) {
                                    leafValues[static_cast<size_t>(idx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                                } else {
                                    leafValues[static_cast<size_t>(idx)] = 0.0f;
                                }
                            }
                        }
                    }

                    int cidx = 0;
                    for (int gz = 0; gz < 4; ++gz) {
                        for (int gy = 0; gy < 4; ++gy) {
                            for (int gx = 0; gx < 4; ++gx, ++cidx) {
                                const int vx = std::min(bx * BT_LEAF_SIZE + ctrl[gx], width - 1);
                                const int vy = std::min(by * BT_LEAF_SIZE + ctrl[gy], height - 1);
                                const int vz = std::min(bz * BT_LEAF_SIZE + ctrl[gz], depth - 1);
                                grid4Values[static_cast<size_t>(cidx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                                if (t == 0 && vx < width && vy < height && vz < depth) {
                                    coarseCtrlKF += static_cast<long long>(compressedVolume[vz][vy][vx].size());
                                }
                            }
                        }
                    }

                    std::vector<float> fineResidualValues(static_cast<size_t>(fineDim * fineDim * fineDim), 0.0f);
                    if (useFine) {
                        int fidx = 0;
                        for (int gz = 0; gz < fineDim; ++gz) {
                            for (int gy = 0; gy < fineDim; ++gy) {
                                for (int gx = 0; gx < fineDim; ++gx, ++fidx) {
                                    const float fx = fineCoords[static_cast<size_t>(gx)];
                                    const float fy = fineCoords[static_cast<size_t>(gy)];
                                    const float fz = fineCoords[static_cast<size_t>(gz)];
                                    const float rawLike = sampleDenseLeafTrilinearSingleFrame(leafValues, fx, fy, fz);
                                    const float coarse = sampleGrid4TrilinearSingleFrameAtFloat(grid4Values, fx, fy, fz);
                                    const float residual = rawLike - coarse;
                                    fineResidualValues[static_cast<size_t>(fidx)] = residual;
                                    fineResidualSeries[static_cast<size_t>(fidx)][static_cast<size_t>(t)] = residual;
                                }
                            }
                        }
                    }

                    for (int lz = 0; lz < BT_LEAF_SIZE; lz += stepZ) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ly += stepY) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; lx += stepX) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx >= width || vy >= height || vz >= depth) continue;

                                const float raw = volumeSequence[t][vz][vy][vx];
                                const float basePred = leafValues[static_cast<size_t>((lz * BT_LEAF_SIZE + ly) * BT_LEAF_SIZE + lx)];
                                const float gridPred = sampleGrid4TrilinearSingleFrame(grid4Values, lx, ly, lz);
                                float finePred = gridPred;
                                if (useFine) {
                                    finePred += sampleRegularControlGridSingleFrame(fineResidualValues, fineDim, lx, ly, lz);
                                }
                                const float eb = std::abs(basePred - raw);
                                const float eg = std::abs(gridPred - raw);
                                const float ef = std::abs(finePred - raw);
                                baselineErrs.push_back(eb);
                                gridErrs.push_back(eg);
                                fineErrs.push_back(ef);
                                baseSum2 += static_cast<double>(eb) * eb;
                                gridSum2 += static_cast<double>(eg) * eg;
                                fineSum2 += static_cast<double>(ef) * ef;
                                ++samples;
                            }
                        }
                    }
                }

                if (useFine) {
                    for (int gz = 0; gz < fineDim; ++gz) {
                        for (int gy = 0; gy < fineDim; ++gy) {
                            for (int gx = 0; gx < fineDim; ++gx) {
                                const int fidx = (gz * fineDim + gy) * fineDim + gx;
                                const std::vector<float>& qvals = fineResidualSeries[static_cast<size_t>(fidx)];
                                std::vector<int> kfIdx = detectKeyFrames(qvals, 0.0, fieldProfile);
                                fineCtrlKF += static_cast<long long>(kfIdx.size());
                            }
                        }
                    }
                }
            }
        }
    }

    std::sort(baselineErrs.begin(), baselineErrs.end());
    std::sort(gridErrs.begin(), gridErrs.end());
    std::sort(fineErrs.begin(), fineErrs.end());
    const size_t idx999 = std::min(baselineErrs.size() - 1, static_cast<size_t>(0.999 * static_cast<double>(baselineErrs.size() - 1)));
    const double psnrPeak = useDataRangeForPsnr ? std::max(1e-6, dataMax - dataMin) : 255.0;
    const double baseRmse = std::sqrt(baseSum2 / std::max<long long>(1, samples));
    const double gridRmse = std::sqrt(gridSum2 / std::max<long long>(1, samples));
    const double fineRmse = std::sqrt(fineSum2 / std::max<long long>(1, samples));

    const std::string suffix = hotspotOnly ? ("_grid4_fine" + std::to_string(fineDim) + "_hf_sampled_full_probe.md")
                                           : ("_grid4_fine" + std::to_string(fineDim) + "_sampled_full_probe.md");
    const fs::path reportPath = fs::path("spatial_probe_reports") / (fs::path(inputFile).stem().string() + suffix);
    std::ofstream md(reportPath);
    md << "# Grid4 + Fine Grid Sampled Full-Volume Probe\n\n";
    md << "- fine grid dim: `" << fineDim << "x" << fineDim << "x" << fineDim << "`\n";
    md << "- mode: `" << (hotspotOnly ? "hotspot-only fine grid" : "global fine grid") << "`\n";
    if (hotspotOnly) md << "- hotspot topK: `" << leafStats.size() << "`\n";
    md << "- sampled stride: `x=" << stepX << ", y=" << stepY << ", z=" << stepZ << ", t=" << stepT << "`\n";
    md << "- samples: `" << samples << "`\n";
    md << "- fine leaves used: `" << fineLeavesUsed << "`\n";
    md << "- baseline RMSE: `" << baseRmse << "`\n";
    md << "- baseline P99.9: `" << baselineErrs[idx999] << "`\n";
    md << "- baseline PSNR: `" << computePsnr(baseRmse, psnrPeak) << "`\n";
    md << "- grid4 RMSE: `" << gridRmse << "`\n";
    md << "- grid4 P99.9: `" << gridErrs[idx999] << "`\n";
    md << "- grid4 PSNR: `" << computePsnr(gridRmse, psnrPeak) << "`\n";
    md << "- coarse+fine RMSE: `" << fineRmse << "`\n";
    md << "- coarse+fine P99.9: `" << fineErrs[idx999] << "`\n";
    md << "- coarse+fine PSNR: `" << computePsnr(fineRmse, psnrPeak) << "`\n";
    md << "- coarse control KF: `" << coarseCtrlKF << "`\n";
    md << "- fine control KF: `" << fineCtrlKF << "`\n";
    md << "- fine/full KF ratio (rough): `" << (coarseCtrlKF > 0 ? static_cast<double>(coarseCtrlKF + fineCtrlKF) / coarseCtrlKF : 0.0) << "`\n";
    md.close();

    printf("\n=== Spatial Probe: grid4 + fine%d sampled full-volume ===\n", fineDim);
    printf("  Mode              : %s\n", hotspotOnly ? "hotspot-only fine grid" : "global fine grid");
    if (hotspotOnly) printf("  Hotspot topK      : %zu\n", leafStats.size());
    printf("  Sample stride     : x=%d y=%d z=%d t=%d\n", stepX, stepY, stepZ, stepT);
    printf("  Samples           : %lld  fine-leaves=%lld\n", samples, fineLeavesUsed);
    printf("  Baseline RMSE     : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           baseRmse, baselineErrs[idx999], computePsnr(baseRmse, psnrPeak));
    printf("  Grid4 RMSE        : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           gridRmse, gridErrs[idx999], computePsnr(gridRmse, psnrPeak));
    printf("  Coarse+Fine RMSE  : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           fineRmse, fineErrs[idx999], computePsnr(fineRmse, psnrPeak));
    printf("  KF coarse/fine    : %lld / %lld\n", coarseCtrlKF, fineCtrlKF);
    printf("  Report            : %s\n", reportPath.string().c_str());
    return 0;
}

static int runGrid4FineAdaptiveSampledFullProbe(const std::string& inputFile,
                                                const Volume4D& volumeSequence,
                                                const CompressedVolume4D& compressedVolume,
                                                int width, int height, int depth, int frames,
                                                const FieldProfile& fieldProfile,
                                                bool enableBackgroundElision,
                                                bool enableBlockAwareCluster,
                                                bool enableBudgetAwareCluster,
                                                bool enableGuardedMedoidCluster,
                                                bool enableValidateFallback,
                                                bool enableHotspotSecondPass,
                                                double clusterThr,
                                                double dataMin,
                                                double dataMax,
                                                bool useDataRangeForPsnr,
                                                const SpatialProbeOptions& spatialProbe) {
    namespace fs = std::filesystem;
    fs::create_directories("spatial_probe_reports");

    BlockTree grid4Tree;
    grid4Tree.build(compressedVolume, width, height, depth, frames,
                    -1.0, clusterThr, fieldProfile, enableBackgroundElision,
                    enableBlockAwareCluster, enableBudgetAwareCluster, enableGuardedMedoidCluster, true, false,
                    &volumeSequence, enableValidateFallback, enableHotspotSecondPass);
    grid4Tree.flattenLeaves();

    const int stepX = 4, stepY = 4, stepZ = 4, stepT = 4;
    const double relFloor = (spatialProbe.threshold > 0.0) ? spatialProbe.threshold : 0.02;
    const double baseFactor = (spatialProbe.adaptiveBaseFactor > 0.0) ? spatialProbe.adaptiveBaseFactor : 1.5;
    const double fine8Trigger = (spatialProbe.adaptiveFine8Trigger > 0.0) ? spatialProbe.adaptiveFine8Trigger : 1.10;
    const int leafNX = (width + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafNY = (height + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafNZ = (depth + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const auto ctrl = grid4ControlCoords();
    const std::vector<float> fine6Coords = uniformControlCoords(6);
    const std::vector<float> fine8Coords = uniformControlCoords(8);

    std::vector<float> baselineErrs;
    std::vector<float> gridErrs;
    std::vector<float> adaptiveErrs;
    baselineErrs.reserve(static_cast<size_t>((width / stepX + 1) * (height / stepY + 1) * (depth / stepZ + 1) * (frames / stepT + 1)));
    gridErrs.reserve(baselineErrs.capacity());
    adaptiveErrs.reserve(baselineErrs.capacity());
    double baseSum2 = 0.0, gridSum2 = 0.0, adaptiveSum2 = 0.0;
    long long samples = 0;
    long long coarseLeaves = 0, fine6Leaves = 0, fine8Leaves = 0;
    long long coarseCtrlKF = 0, fine6CtrlKF = 0, fine8CtrlKF = 0;

    for (int bz = 0; bz < leafNZ; ++bz) {
        for (int by = 0; by < leafNY; ++by) {
            for (int bx = 0; bx < leafNX; ++bx) {
                std::vector<std::array<float, BT_LEAF_VOXELS>> leafValuesByT(static_cast<size_t>(frames));
                std::vector<std::array<float, BT_LEAF_VOXELS>> rawLeafValuesByT(static_cast<size_t>(frames));
                std::vector<std::array<float, 64>> grid4ValuesByT(static_cast<size_t>(frames));

                for (int t = 0; t < frames; ++t) {
                    auto& leafValues = leafValuesByT[static_cast<size_t>(t)];
                    auto& rawLeafValues = rawLeafValuesByT[static_cast<size_t>(t)];
                    auto& grid4Values = grid4ValuesByT[static_cast<size_t>(t)];
                    int idx = 0;
                    for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; ++lx, ++idx) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx < width && vy < height && vz < depth) {
                                    leafValues[static_cast<size_t>(idx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                                    rawLeafValues[static_cast<size_t>(idx)] = volumeSequence[t][vz][vy][vx];
                                } else {
                                    leafValues[static_cast<size_t>(idx)] = 0.0f;
                                    rawLeafValues[static_cast<size_t>(idx)] = 0.0f;
                                }
                            }
                        }
                    }
                    int cidx = 0;
                    for (int gz = 0; gz < 4; ++gz) {
                        for (int gy = 0; gy < 4; ++gy) {
                            for (int gx = 0; gx < 4; ++gx, ++cidx) {
                                const int vx = std::min(bx * BT_LEAF_SIZE + ctrl[gx], width - 1);
                                const int vy = std::min(by * BT_LEAF_SIZE + ctrl[gy], height - 1);
                                const int vz = std::min(bz * BT_LEAF_SIZE + ctrl[gz], depth - 1);
                                grid4Values[static_cast<size_t>(cidx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                                if (t == 0 && vx < width && vy < height && vz < depth) {
                                    coarseCtrlKF += static_cast<long long>(compressedVolume[vz][vy][vx].size());
                                }
                            }
                        }
                    }
                }

                struct SampleRec {
                    int lx, ly, lz, t;
                    float raw;
                    float basePred;
                    float gridPred;
                };
                std::vector<SampleRec> localSamples;
                localSamples.reserve(static_cast<size_t>(BT_LEAF_VOXELS * frames) / 64 + 16);
                double localBaseSum2 = 0.0;
                double localGridSum2 = 0.0;
                float localMin = std::numeric_limits<float>::infinity();
                float localMax = -std::numeric_limits<float>::infinity();

                for (int t = 0; t < frames; t += stepT) {
                    const auto& leafValues = leafValuesByT[static_cast<size_t>(t)];
                    const auto& grid4Values = grid4ValuesByT[static_cast<size_t>(t)];
                    for (int lz = 0; lz < BT_LEAF_SIZE; lz += stepZ) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ly += stepY) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; lx += stepX) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx >= width || vy >= height || vz >= depth) continue;
                                const float raw = volumeSequence[t][vz][vy][vx];
                                const float basePred = leafValues[static_cast<size_t>((lz * BT_LEAF_SIZE + ly) * BT_LEAF_SIZE + lx)];
                                const float gridPred = sampleGrid4TrilinearSingleFrame(grid4Values, lx, ly, lz);
                                localSamples.push_back({lx, ly, lz, t, raw, basePred, gridPred});
                                const double eb = static_cast<double>(basePred) - raw;
                                const double eg = static_cast<double>(gridPred) - raw;
                                localBaseSum2 += eb * eb;
                                localGridSum2 += eg * eg;
                                localMin = std::min(localMin, raw);
                                localMax = std::max(localMax, raw);
                            }
                        }
                    }
                }

                const double localBaseRmse = std::sqrt(localBaseSum2 / std::max<size_t>(1, localSamples.size()));
                const double localGridRmse = std::sqrt(localGridSum2 / std::max<size_t>(1, localSamples.size()));
                const double localRange = std::max(1e-6, static_cast<double>(localMax - localMin));
                const double acceptThr = std::max(localBaseRmse * baseFactor, localRange * relFloor);

                int chosenLevel = 0;
                std::vector<std::vector<float>> fine6Series;
                std::vector<std::vector<float>> fine8Series;
                double localFine6Rmse = std::numeric_limits<double>::infinity();
                double localFine8Rmse = std::numeric_limits<double>::infinity();

                if (localGridRmse > acceptThr) {
                    fine6Series.assign(static_cast<size_t>(6 * 6 * 6), std::vector<float>(static_cast<size_t>(frames), 0.0f));
                    for (int t = 0; t < frames; ++t) {
                        const auto& leafValues = leafValuesByT[static_cast<size_t>(t)];
                        const auto& grid4Values = grid4ValuesByT[static_cast<size_t>(t)];
                        int fidx = 0;
                        for (int gz = 0; gz < 6; ++gz) {
                            for (int gy = 0; gy < 6; ++gy) {
                                for (int gx = 0; gx < 6; ++gx, ++fidx) {
                                    const float fx = fine6Coords[static_cast<size_t>(gx)];
                                    const float fy = fine6Coords[static_cast<size_t>(gy)];
                                    const float fz = fine6Coords[static_cast<size_t>(gz)];
                                    const float rawLike = sampleDenseLeafTrilinearSingleFrame(leafValues, fx, fy, fz);
                                    const float coarse = sampleGrid4TrilinearSingleFrameAtFloat(grid4Values, fx, fy, fz);
                                    fine6Series[static_cast<size_t>(fidx)][static_cast<size_t>(t)] = rawLike - coarse;
                                }
                            }
                        }
                    }
                    double sum2 = 0.0;
                    for (const auto& s : localSamples) {
                        std::vector<float> controlValues(static_cast<size_t>(6 * 6 * 6), 0.0f);
                        for (int i = 0; i < 6 * 6 * 6; ++i) {
                            controlValues[static_cast<size_t>(i)] = fine6Series[static_cast<size_t>(i)][static_cast<size_t>(s.t)];
                        }
                        const float pred = s.gridPred + sampleRegularControlGridSingleFrame(controlValues, 6, s.lx, s.ly, s.lz);
                        const double e = static_cast<double>(pred) - s.raw;
                        sum2 += e * e;
                    }
                    localFine6Rmse = std::sqrt(sum2 / std::max<size_t>(1, localSamples.size()));
                    if (localFine6Rmse <= acceptThr) {
                        chosenLevel = 6;
                    } else if (localFine6Rmse > acceptThr * fine8Trigger) {
                        fine8Series.assign(static_cast<size_t>(8 * 8 * 8), std::vector<float>(static_cast<size_t>(frames), 0.0f));
                        for (int t = 0; t < frames; ++t) {
                            const auto& leafValues = leafValuesByT[static_cast<size_t>(t)];
                            const auto& grid4Values = grid4ValuesByT[static_cast<size_t>(t)];
                            int fidx = 0;
                            for (int gz = 0; gz < 8; ++gz) {
                                for (int gy = 0; gy < 8; ++gy) {
                                    for (int gx = 0; gx < 8; ++gx, ++fidx) {
                                        const float fx = fine8Coords[static_cast<size_t>(gx)];
                                        const float fy = fine8Coords[static_cast<size_t>(gy)];
                                        const float fz = fine8Coords[static_cast<size_t>(gz)];
                                        const float rawLike = sampleDenseLeafTrilinearSingleFrame(leafValues, fx, fy, fz);
                                        const float coarse = sampleGrid4TrilinearSingleFrameAtFloat(grid4Values, fx, fy, fz);
                                        fine8Series[static_cast<size_t>(fidx)][static_cast<size_t>(t)] = rawLike - coarse;
                                    }
                                }
                            }
                        }
                        double sum28 = 0.0;
                        for (const auto& s : localSamples) {
                            std::vector<float> controlValues(static_cast<size_t>(8 * 8 * 8), 0.0f);
                            for (int i = 0; i < 8 * 8 * 8; ++i) {
                                controlValues[static_cast<size_t>(i)] = fine8Series[static_cast<size_t>(i)][static_cast<size_t>(s.t)];
                            }
                            const float pred = s.gridPred + sampleRegularControlGridSingleFrame(controlValues, 8, s.lx, s.ly, s.lz);
                            const double e = static_cast<double>(pred) - s.raw;
                            sum28 += e * e;
                        }
                        localFine8Rmse = std::sqrt(sum28 / std::max<size_t>(1, localSamples.size()));
                        chosenLevel = (localFine8Rmse < localFine6Rmse) ? 8 : 6;
                    } else {
                        chosenLevel = 6;
                    }
                }

                if (chosenLevel == 0) ++coarseLeaves;
                else if (chosenLevel == 6) ++fine6Leaves;
                else ++fine8Leaves;

                if (chosenLevel == 6) {
                    for (const auto& seq : fine6Series) {
                        std::vector<int> kfIdx = detectKeyFrames(seq, 0.0, fieldProfile);
                        fine6CtrlKF += static_cast<long long>(kfIdx.size());
                    }
                } else if (chosenLevel == 8) {
                    for (const auto& seq : fine8Series) {
                        std::vector<int> kfIdx = detectKeyFrames(seq, 0.0, fieldProfile);
                        fine8CtrlKF += static_cast<long long>(kfIdx.size());
                    }
                }

                for (const auto& s : localSamples) {
                    float adaptivePred = s.gridPred;
                    if (chosenLevel == 6) {
                        std::vector<float> controlValues(static_cast<size_t>(6 * 6 * 6), 0.0f);
                        for (int i = 0; i < 6 * 6 * 6; ++i) {
                            controlValues[static_cast<size_t>(i)] = fine6Series[static_cast<size_t>(i)][static_cast<size_t>(s.t)];
                        }
                        adaptivePred += sampleRegularControlGridSingleFrame(controlValues, 6, s.lx, s.ly, s.lz);
                    } else if (chosenLevel == 8) {
                        std::vector<float> controlValues(static_cast<size_t>(8 * 8 * 8), 0.0f);
                        for (int i = 0; i < 8 * 8 * 8; ++i) {
                            controlValues[static_cast<size_t>(i)] = fine8Series[static_cast<size_t>(i)][static_cast<size_t>(s.t)];
                        }
                        adaptivePred += sampleRegularControlGridSingleFrame(controlValues, 8, s.lx, s.ly, s.lz);
                    }

                    const float eb = std::abs(s.basePred - s.raw);
                    const float eg = std::abs(s.gridPred - s.raw);
                    const float ea = std::abs(adaptivePred - s.raw);
                    baselineErrs.push_back(eb);
                    gridErrs.push_back(eg);
                    adaptiveErrs.push_back(ea);
                    baseSum2 += static_cast<double>(eb) * eb;
                    gridSum2 += static_cast<double>(eg) * eg;
                    adaptiveSum2 += static_cast<double>(ea) * ea;
                    ++samples;
                }
            }
        }
    }

    std::sort(baselineErrs.begin(), baselineErrs.end());
    std::sort(gridErrs.begin(), gridErrs.end());
    std::sort(adaptiveErrs.begin(), adaptiveErrs.end());
    const size_t idx999 = std::min(baselineErrs.size() - 1, static_cast<size_t>(0.999 * static_cast<double>(baselineErrs.size() - 1)));
    const double psnrPeak = useDataRangeForPsnr ? std::max(1e-6, dataMax - dataMin) : 255.0;
    const double baseRmse = std::sqrt(baseSum2 / std::max<long long>(1, samples));
    const double gridRmse = std::sqrt(gridSum2 / std::max<long long>(1, samples));
    const double adaptiveRmse = std::sqrt(adaptiveSum2 / std::max<long long>(1, samples));

    const fs::path reportPath = fs::path("spatial_probe_reports") / (fs::path(inputFile).stem().string() + "_grid4_fine_adaptive_sampled_full_probe.md");
    std::ofstream md(reportPath);
    md << "# Grid4 + Adaptive Fine Grid Sampled Full-Volume Probe\n\n";
    md << "- adaptive rule: `coarse -> fine6 -> fine8`\n";
    md << "- accept threshold: `max(baseRmse*" << baseFactor << ", leafRange*" << relFloor << ")`\n";
    md << "- fine8 trigger: `fine6Rmse > acceptThreshold * " << fine8Trigger << "`\n";
    md << "- sampled stride: `x=" << stepX << ", y=" << stepY << ", z=" << stepZ << ", t=" << stepT << "`\n";
    md << "- samples: `" << samples << "`\n";
    md << "- chosen leaves coarse/fine6/fine8: `" << coarseLeaves << " / " << fine6Leaves << " / " << fine8Leaves << "`\n";
    md << "- baseline RMSE: `" << baseRmse << "`\n";
    md << "- baseline P99.9: `" << baselineErrs[idx999] << "`\n";
    md << "- baseline PSNR: `" << computePsnr(baseRmse, psnrPeak) << "`\n";
    md << "- grid4 RMSE: `" << gridRmse << "`\n";
    md << "- grid4 P99.9: `" << gridErrs[idx999] << "`\n";
    md << "- grid4 PSNR: `" << computePsnr(gridRmse, psnrPeak) << "`\n";
    md << "- adaptive RMSE: `" << adaptiveRmse << "`\n";
    md << "- adaptive P99.9: `" << adaptiveErrs[idx999] << "`\n";
    md << "- adaptive PSNR: `" << computePsnr(adaptiveRmse, psnrPeak) << "`\n";
    md << "- KF coarse/fine6/fine8: `" << coarseCtrlKF << " / " << fine6CtrlKF << " / " << fine8CtrlKF << "`\n";
    md.close();

    printf("\n=== Spatial Probe: grid4 + adaptive fine sampled full-volume ===\n");
    printf("  Rule              : coarse -> fine6 -> fine8\n");
    printf("  Accept threshold  : max(baseRmse*%.3f, leafRange*%.4f)\n", baseFactor, relFloor);
    printf("  Fine8 trigger     : fine6Rmse > acceptThr*%.3f\n", fine8Trigger);
    printf("  Sample stride     : x=%d y=%d z=%d t=%d\n", stepX, stepY, stepZ, stepT);
    printf("  Samples           : %lld\n", samples);
    printf("  Leaves c/6/8      : %lld / %lld / %lld\n", coarseLeaves, fine6Leaves, fine8Leaves);
    printf("  Baseline RMSE     : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           baseRmse, baselineErrs[idx999], computePsnr(baseRmse, psnrPeak));
    printf("  Grid4 RMSE        : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           gridRmse, gridErrs[idx999], computePsnr(gridRmse, psnrPeak));
    printf("  Adaptive RMSE     : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           adaptiveRmse, adaptiveErrs[idx999], computePsnr(adaptiveRmse, psnrPeak));
    printf("  KF coarse/6/8     : %lld / %lld / %lld\n", coarseCtrlKF, fine6CtrlKF, fine8CtrlKF);
    printf("  Report            : %s\n", reportPath.string().c_str());
    return 0;
}

static int runGrid4SpatialFirstSampledFullProbe(const std::string& inputFile,
                                                const Volume4D& volumeSequence,
                                                const CompressedVolume4D& compressedVolume,
                                                int width, int height, int depth, int frames,
                                                const FieldProfile& fieldProfile,
                                                double dataMin,
                                                double dataMax,
                                                bool useDataRangeForPsnr,
                                                const SpatialProbeOptions& spatialProbe,
                                                bool weighted) {
    namespace fs = std::filesystem;
    fs::create_directories("spatial_probe_reports");

    const int fineDim = 6;
    const int stepX = 4, stepY = 4, stepZ = 4, stepT = 4;
    const int leafNX = (width + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafNY = (height + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafNZ = (depth + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const auto ctrl = grid4ControlCoords();
    const std::vector<float> fineCoords = uniformControlCoords(fineDim);
    auto semanticWeight = [&](float raw) -> float {
        if (!weighted) return 1.0f;
        if (fieldProfile.type == FieldType::DENSITY) {
            const float cutoff = fieldProfile.den.render_cutoff;
            const float band = std::max(1e-6f, fieldProfile.den.cutoff_band);
            if (cutoff > 0.0f) {
                const float d = std::abs(raw - cutoff);
                if (d <= band) return 4.0f;
                if (raw >= cutoff) return 2.0f;
                if (raw >= cutoff - 2.0f * band) return 1.5f;
            }
            return 1.0f;
        }
        if (fieldProfile.type == FieldType::SDF) {
            const float d = std::abs(raw - fieldProfile.sdf.iso);
            if (d <= fieldProfile.sdf.w_critical) return 4.0f;
            if (d <= fieldProfile.sdf.w_near) return 2.5f;
            if (d <= fieldProfile.sdf.w_band) return 1.5f;
            return 1.0f;
        }
        return 1.0f;
    };
    std::vector<float> baselineErrs;
    std::vector<float> coarseErrs;
    std::vector<float> fineErrs;
    baselineErrs.reserve(static_cast<size_t>((width / stepX + 1) * (height / stepY + 1) * (depth / stepZ + 1) * (frames / stepT + 1)));
    coarseErrs.reserve(baselineErrs.capacity());
    fineErrs.reserve(baselineErrs.capacity());
    double baseSum2 = 0.0, coarseSum2 = 0.0, fineSum2 = 0.0;
    long long samples = 0;
    long long coarseCtrlKF = 0;
    long long fineCtrlKF = 0;

    std::vector<float> exportFrameBuffer;
    if (spatialProbe.exportFrame >= 0 && spatialProbe.exportFrame < frames) {
        exportFrameBuffer.assign(static_cast<size_t>(width) * height * depth, 0.0f);
    }

    for (int bz = 0; bz < leafNZ; ++bz) {
        for (int by = 0; by < leafNY; ++by) {
            for (int bx = 0; bx < leafNX; ++bx) {
                std::vector<std::array<float, BT_LEAF_VOXELS>> rawLeafValuesByT(static_cast<size_t>(frames));
                for (int t = 0; t < frames; ++t) {
                    auto& rawLeafValues = rawLeafValuesByT[static_cast<size_t>(t)];
                    int idx = 0;
                    for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; ++lx, ++idx) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                rawLeafValues[static_cast<size_t>(idx)] =
                                    (vx < width && vy < height && vz < depth)
                                        ? volumeSequence[t][vz][vy][vx]
                                        : 0.0f;
                            }
                        }
                    }
                }

                std::vector<std::vector<float>> coarseSeries(64, std::vector<float>(static_cast<size_t>(frames), 0.0f));
                std::vector<std::vector<float>> fineSeries(static_cast<size_t>(fineDim * fineDim * fineDim),
                                                           std::vector<float>(static_cast<size_t>(frames), 0.0f));
                std::vector<std::vector<float>> coarseWeights;
                std::vector<std::vector<float>> fineWeights;
                if (weighted) {
                    coarseWeights.assign(64, std::vector<float>(static_cast<size_t>(frames), 1.0f));
                    fineWeights.assign(static_cast<size_t>(fineDim * fineDim * fineDim),
                                       std::vector<float>(static_cast<size_t>(frames), 1.0f));
                }

                for (int t = 0; t < frames; ++t) {
                    const auto& rawLeafValues = rawLeafValuesByT[static_cast<size_t>(t)];
                    std::array<float, 64> coarseFrame{};
                    int cidx = 0;
                    for (int gz = 0; gz < 4; ++gz) {
                        for (int gy = 0; gy < 4; ++gy) {
                            for (int gx = 0; gx < 4; ++gx, ++cidx) {
                                const float lx = static_cast<float>(ctrl[gx]);
                                const float ly = static_cast<float>(ctrl[gy]);
                                const float lz = static_cast<float>(ctrl[gz]);
                                const float v = sampleDenseLeafTrilinearSingleFrame(rawLeafValues, lx, ly, lz);
                                coarseFrame[static_cast<size_t>(cidx)] = v;
                                coarseSeries[static_cast<size_t>(cidx)][static_cast<size_t>(t)] = v;
                                if (weighted) {
                                    coarseWeights[static_cast<size_t>(cidx)][static_cast<size_t>(t)] = semanticWeight(v);
                                }
                            }
                        }
                    }

                    int fidx = 0;
                    for (int gz = 0; gz < fineDim; ++gz) {
                        for (int gy = 0; gy < fineDim; ++gy) {
                                for (int gx = 0; gx < fineDim; ++gx, ++fidx) {
                                    const float fx = fineCoords[static_cast<size_t>(gx)];
                                    const float fy = fineCoords[static_cast<size_t>(gy)];
                                    const float fz = fineCoords[static_cast<size_t>(gz)];
                                    const float rawLike = sampleDenseLeafTrilinearSingleFrame(rawLeafValues, fx, fy, fz);
                                    const float coarse = sampleGrid4TrilinearSingleFrameAtFloat(coarseFrame, fx, fy, fz);
                                    fineSeries[static_cast<size_t>(fidx)][static_cast<size_t>(t)] = rawLike - coarse;
                                    if (weighted) {
                                        fineWeights[static_cast<size_t>(fidx)][static_cast<size_t>(t)] = semanticWeight(rawLike);
                                    }
                                }
                            }
                        }
                }

                std::vector<std::vector<Point1D>> coarseKfs(64);
                for (int i = 0; i < 64; ++i) {
                    coarseKfs[static_cast<size_t>(i)] = weighted
                        ? buildTemporalPointKfsWeighted(coarseSeries[static_cast<size_t>(i)], fieldProfile, coarseWeights[static_cast<size_t>(i)])
                        : buildTemporalPointKfs(coarseSeries[static_cast<size_t>(i)], fieldProfile);
                    coarseCtrlKF += static_cast<long long>(coarseKfs[static_cast<size_t>(i)].size());
                }
                std::vector<std::vector<Point1D>> fineKfs(static_cast<size_t>(fineDim * fineDim * fineDim));
                for (int i = 0; i < fineDim * fineDim * fineDim; ++i) {
                    fineKfs[static_cast<size_t>(i)] = weighted
                        ? buildTemporalPointKfsWeighted(fineSeries[static_cast<size_t>(i)], fieldProfile, fineWeights[static_cast<size_t>(i)])
                        : buildTemporalPointKfs(fineSeries[static_cast<size_t>(i)], fieldProfile);
                    fineCtrlKF += static_cast<long long>(fineKfs[static_cast<size_t>(i)].size());
                }

                for (int t = 0; t < frames; t += stepT) {
                    std::array<float, 64> coarseFrame{};
                    for (int i = 0; i < 64; ++i) {
                        coarseFrame[static_cast<size_t>(i)] = sampleTemporalKfs(coarseKfs[static_cast<size_t>(i)], t);
                    }
                    std::vector<float> fineFrame(static_cast<size_t>(fineDim * fineDim * fineDim), 0.0f);
                    for (int i = 0; i < fineDim * fineDim * fineDim; ++i) {
                        fineFrame[static_cast<size_t>(i)] = sampleTemporalKfs(fineKfs[static_cast<size_t>(i)], t);
                    }

                    for (int lz = 0; lz < BT_LEAF_SIZE; lz += stepZ) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ly += stepY) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; lx += stepX) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx >= width || vy >= height || vz >= depth) continue;

                                const float raw = volumeSequence[t][vz][vy][vx];
                                const float basePred = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                                const float coarsePred = sampleGrid4TrilinearSingleFrame(coarseFrame, lx, ly, lz);
                                const float finePred = coarsePred + sampleRegularControlGridSingleFrame(fineFrame, fineDim, lx, ly, lz);
                                const float eb = std::abs(basePred - raw);
                                const float ec = std::abs(coarsePred - raw);
                                const float ef = std::abs(finePred - raw);
                                baselineErrs.push_back(eb);
                                coarseErrs.push_back(ec);
                                fineErrs.push_back(ef);
                                baseSum2 += static_cast<double>(eb) * eb;
                                coarseSum2 += static_cast<double>(ec) * ec;
                                fineSum2 += static_cast<double>(ef) * ef;
                                ++samples;
                            }
                        }
                    }

                    if (!exportFrameBuffer.empty() && t == spatialProbe.exportFrame) {
                        for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
                            for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                                for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                                    const int vx = bx * BT_LEAF_SIZE + lx;
                                    const int vy = by * BT_LEAF_SIZE + ly;
                                    const int vz = bz * BT_LEAF_SIZE + lz;
                                    if (vx >= width || vy >= height || vz >= depth) continue;
                                    const float pred = sampleGrid4TrilinearSingleFrame(coarseFrame, lx, ly, lz) +
                                                       sampleRegularControlGridSingleFrame(fineFrame, fineDim, lx, ly, lz);
                                    exportFrameBuffer[(static_cast<size_t>(vz) * height + static_cast<size_t>(vy)) * width + static_cast<size_t>(vx)] = pred;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::sort(baselineErrs.begin(), baselineErrs.end());
    std::sort(coarseErrs.begin(), coarseErrs.end());
    std::sort(fineErrs.begin(), fineErrs.end());
    const size_t idx999 = std::min(baselineErrs.size() - 1, static_cast<size_t>(0.999 * static_cast<double>(baselineErrs.size() - 1)));
    const double psnrPeak = useDataRangeForPsnr ? std::max(1e-6, dataMax - dataMin) : 255.0;
    const double baseRmse = std::sqrt(baseSum2 / std::max<long long>(1, samples));
    const double coarseRmse = std::sqrt(coarseSum2 / std::max<long long>(1, samples));
    const double fineRmse = std::sqrt(fineSum2 / std::max<long long>(1, samples));

    const std::string tag = weighted ? "grid4_spatialfirst_weighted" : "grid4_spatialfirst";
    const fs::path reportPath = fs::path("spatial_probe_reports") / (fs::path(inputFile).stem().string() + "_" + tag + "_sampled_full_probe.md");
    std::ofstream md(reportPath);
    md << "# Grid4 Spatial-First Probe\n\n";
    md << "- mode: `" << (weighted ? "weighted" : "unweighted") << "`\n";
    md << "- flow: `raw per-frame spatial fitting -> control-sequence temporal compression`\n";
    md << "- coarse grid: `4x4x4`\n";
    md << "- fine residual grid: `6x6x6`\n";
    md << "- sampled stride: `x=" << stepX << ", y=" << stepY << ", z=" << stepZ << ", t=" << stepT << "`\n";
    md << "- samples: `" << samples << "`\n";
    md << "- temporal baseline RMSE: `" << baseRmse << "`\n";
    md << "- temporal baseline P99.9: `" << baselineErrs[idx999] << "`\n";
    md << "- temporal baseline PSNR: `" << computePsnr(baseRmse, psnrPeak) << "`\n";
    md << "- spatial-first coarse RMSE: `" << coarseRmse << "`\n";
    md << "- spatial-first coarse P99.9: `" << coarseErrs[idx999] << "`\n";
    md << "- spatial-first coarse PSNR: `" << computePsnr(coarseRmse, psnrPeak) << "`\n";
    md << "- spatial-first coarse+fine RMSE: `" << fineRmse << "`\n";
    md << "- spatial-first coarse+fine P99.9: `" << fineErrs[idx999] << "`\n";
    md << "- spatial-first coarse+fine PSNR: `" << computePsnr(fineRmse, psnrPeak) << "`\n";
    md << "- coarse control KF: `" << coarseCtrlKF << "`\n";
    md << "- fine control KF: `" << fineCtrlKF << "`\n";
    md.close();

    if (!exportFrameBuffer.empty()) {
        const fs::path outDir = spatialProbe.exportDir.empty()
            ? (fs::path("smokeDate/render_ready/spatialfirst_probe") / fs::path(inputFile).stem())
            : fs::path(spatialProbe.exportDir);
        fs::create_directories(outDir);
        const std::string exportStem = spatialProbe.exportName.empty()
            ? (fs::path(inputFile).stem().string() + "_f" + std::to_string(spatialProbe.exportFrame) + (weighted ? "_grid4sfw" : "_grid4sf"))
            : spatialProbe.exportName;
        const fs::path rawPath = outDir / (exportStem + ".raw");
        const fs::path metaPath = outDir / (exportStem + ".metadata.json");

        std::ofstream rawOut(rawPath, std::ios::binary);
        rawOut.write(reinterpret_cast<const char*>(exportFrameBuffer.data()),
                     static_cast<std::streamsize>(exportFrameBuffer.size() * sizeof(float)));
        rawOut.close();

        std::array<int, 3> exportBboxMin{0, 0, 0};
        std::array<int, 3> exportBboxMax{width - 1, height - 1, depth - 1};
        try {
            const fs::path srcMetaPath = guessMetadataPathForRaw(inputFile);
            if (!srcMetaPath.empty()) {
                const auto srcMeta = vdbtools::loadFrameMetadata(srcMetaPath);
                if (srcMeta.width == width && srcMeta.height == height && srcMeta.depth == depth) {
                    exportBboxMin = srcMeta.bboxMin;
                    exportBboxMax = srcMeta.bboxMax;
                }
            }
        } catch (...) {
        }

        std::ofstream metaOut(metaPath);
        metaOut << "{\n";
        metaOut << "  \"source_dir\": \"spatialfirst_probe\",\n";
        metaOut << "  \"grid_name\": \"density\",\n";
        metaOut << "  \"width\": " << width << ",\n";
        metaOut << "  \"height\": " << height << ",\n";
        metaOut << "  \"depth\": " << depth << ",\n";
        metaOut << "  \"frames\": 1,\n";
        metaOut << "  \"bbox_min\": [" << exportBboxMin[0] << ", " << exportBboxMin[1] << ", " << exportBboxMin[2] << "],\n";
        metaOut << "  \"bbox_max\": [" << exportBboxMax[0] << ", " << exportBboxMax[1] << ", " << exportBboxMax[2] << "],\n";
        metaOut << "  \"data_min\": " << dataMin << ",\n";
        metaOut << "  \"data_max\": " << dataMax << "\n";
        metaOut << "}\n";
        metaOut.close();
    }

    printf("\n=== Spatial Probe: grid4 spatial-first %s sampled full-volume ===\n", weighted ? "weighted" : "unweighted");
    printf("  Sample stride     : x=%d y=%d z=%d t=%d\n", stepX, stepY, stepZ, stepT);
    printf("  Samples           : %lld\n", samples);
    printf("  Temporal RMSE     : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           baseRmse, baselineErrs[idx999], computePsnr(baseRmse, psnrPeak));
    printf("  SF coarse RMSE    : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           coarseRmse, coarseErrs[idx999], computePsnr(coarseRmse, psnrPeak));
    printf("  SF coarse+fine    : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           fineRmse, fineErrs[idx999], computePsnr(fineRmse, psnrPeak));
    printf("  KF coarse/fine    : %lld / %lld\n", coarseCtrlKF, fineCtrlKF);
    printf("  Report            : %s\n", reportPath.string().c_str());
    if (!exportFrameBuffer.empty()) {
        const fs::path outDir = spatialProbe.exportDir.empty()
            ? (fs::path("smokeDate/render_ready/spatialfirst_probe") / fs::path(inputFile).stem())
            : fs::path(spatialProbe.exportDir);
        const std::string exportStem = spatialProbe.exportName.empty()
            ? (fs::path(inputFile).stem().string() + "_f" + std::to_string(spatialProbe.exportFrame) + (weighted ? "_grid4sfw" : "_grid4sf"))
            : spatialProbe.exportName;
        printf("  Export           : %s\\%s.raw\n", outDir.string().c_str(), exportStem.c_str());
    }
    return 0;
}

static int runGrid4FineAdaptiveScoreSampledFullProbe(const std::string& inputFile,
                                                     const Volume4D& volumeSequence,
                                                     const CompressedVolume4D& compressedVolume,
                                                     int width, int height, int depth, int frames,
                                                     const FieldProfile& fieldProfile,
                                                     bool enableBackgroundElision,
                                                     bool enableBlockAwareCluster,
                                                     bool enableBudgetAwareCluster,
                                                     bool enableGuardedMedoidCluster,
                                                     bool enableValidateFallback,
                                                     bool enableHotspotSecondPass,
                                                     double clusterThr,
                                                     double dataMin,
                                                     double dataMax,
                                                     bool useDataRangeForPsnr,
                                                     const SpatialProbeOptions& spatialProbe) {
    namespace fs = std::filesystem;
    fs::create_directories("spatial_probe_reports");

    BlockTree grid4Tree;
    grid4Tree.build(compressedVolume, width, height, depth, frames,
                    -1.0, clusterThr, fieldProfile, enableBackgroundElision,
                    enableBlockAwareCluster, enableBudgetAwareCluster, enableGuardedMedoidCluster, true, false,
                    &volumeSequence, enableValidateFallback, enableHotspotSecondPass);
    grid4Tree.flattenLeaves();

    const int stepX = 4, stepY = 4, stepZ = 4, stepT = 4;
    const double scoreT1 = (spatialProbe.scoreT1 > 0.0) ? spatialProbe.scoreT1 : 0.35;
    const double scoreT2 = (spatialProbe.scoreT2 > 0.0) ? spatialProbe.scoreT2 : 0.65;
    const double wRmse = (spatialProbe.scoreWRmse > 0.0) ? spatialProbe.scoreWRmse : 0.40;
    const double wP99  = (spatialProbe.scoreWP99  > 0.0) ? spatialProbe.scoreWP99  : 0.20;
    const double wVis  = (spatialProbe.scoreWVis  > 0.0) ? spatialProbe.scoreWVis  : 0.20;
    const double wGrad = (spatialProbe.scoreWGrad > 0.0) ? spatialProbe.scoreWGrad : 0.10;
    const double wVar  = (spatialProbe.scoreWVar  > 0.0) ? spatialProbe.scoreWVar  : 0.10;
    const double fine8Improve = (spatialProbe.fine8Improve > 0.0) ? spatialProbe.fine8Improve : 0.95; // require at least 5% RMSE reduction over fine6 by default
    const int leafNX = (width + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafNY = (height + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafNZ = (depth + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const auto ctrl = grid4ControlCoords();
    const std::vector<float> fine6Coords = uniformControlCoords(6);
    const std::vector<float> fine8Coords = uniformControlCoords(8);

    std::vector<float> baselineErrs;
    std::vector<float> gridErrs;
    std::vector<float> adaptiveErrs;
    baselineErrs.reserve(static_cast<size_t>((width / stepX + 1) * (height / stepY + 1) * (depth / stepZ + 1) * (frames / stepT + 1)));
    gridErrs.reserve(baselineErrs.capacity());
    adaptiveErrs.reserve(baselineErrs.capacity());
    double baseSum2 = 0.0, gridSum2 = 0.0, adaptiveSum2 = 0.0;
    long long samples = 0;
    long long coarseLeaves = 0, fine6Leaves = 0, fine8Leaves = 0;
    long long coarseCtrlKF = 0, fine6CtrlKF = 0, fine8CtrlKF = 0;
    double scoreSum = 0.0;
    double scoreMax = 0.0;
    std::vector<float> exportFrameBuffer;
    if (spatialProbe.exportFrame >= 0 && spatialProbe.exportFrame < frames) {
        exportFrameBuffer.assign(static_cast<size_t>(width) * height * depth, 0.0f);
    }

    for (int bz = 0; bz < leafNZ; ++bz) {
        for (int by = 0; by < leafNY; ++by) {
            for (int bx = 0; bx < leafNX; ++bx) {
                std::vector<std::array<float, BT_LEAF_VOXELS>> leafValuesByT(static_cast<size_t>(frames));
                std::vector<std::array<float, BT_LEAF_VOXELS>> rawLeafValuesByT(static_cast<size_t>(frames));
                std::vector<std::array<float, 64>> grid4ValuesByT(static_cast<size_t>(frames));

                for (int t = 0; t < frames; ++t) {
                    auto& leafValues = leafValuesByT[static_cast<size_t>(t)];
                    auto& rawLeafValues = rawLeafValuesByT[static_cast<size_t>(t)];
                    auto& grid4Values = grid4ValuesByT[static_cast<size_t>(t)];
                    int idx = 0;
                    for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; ++lx, ++idx) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx < width && vy < height && vz < depth) {
                                    leafValues[static_cast<size_t>(idx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                                    rawLeafValues[static_cast<size_t>(idx)] = volumeSequence[t][vz][vy][vx];
                                } else {
                                    leafValues[static_cast<size_t>(idx)] = 0.0f;
                                    rawLeafValues[static_cast<size_t>(idx)] = 0.0f;
                                }
                            }
                        }
                    }
                    int cidx = 0;
                    for (int gz = 0; gz < 4; ++gz) {
                        for (int gy = 0; gy < 4; ++gy) {
                            for (int gx = 0; gx < 4; ++gx, ++cidx) {
                                const int vx = std::min(bx * BT_LEAF_SIZE + ctrl[gx], width - 1);
                                const int vy = std::min(by * BT_LEAF_SIZE + ctrl[gy], height - 1);
                                const int vz = std::min(bz * BT_LEAF_SIZE + ctrl[gz], depth - 1);
                                grid4Values[static_cast<size_t>(cidx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                                if (t == 0 && vx < width && vy < height && vz < depth) {
                                    coarseCtrlKF += static_cast<long long>(compressedVolume[vz][vy][vx].size());
                                }
                            }
                        }
                    }
                }

                struct SampleRec {
                    int lx, ly, lz, t;
                    float raw;
                    float basePred;
                    float gridPred;
                };
                std::vector<SampleRec> localSamples;
                localSamples.reserve(static_cast<size_t>(BT_LEAF_VOXELS * frames) / 64 + 16);
                std::vector<float> localGridAbsErrs;
                double localBaseSum2 = 0.0;
                double localGridSum2 = 0.0;
                double rawSum = 0.0;
                double rawSum2 = 0.0;
                int rawCount = 0;
                int visCount = 0;
                int visMismatchCount = 0;
                double gradAccum = 0.0;
                int gradCount = 0;
                float localMin = std::numeric_limits<float>::infinity();
                float localMax = -std::numeric_limits<float>::infinity();
                const bool useVisibilityAware =
                    (fieldProfile.type == FieldType::DENSITY || fieldProfile.type == FieldType::GENERIC) &&
                    fieldProfile.den.render_cutoff > 0.0f &&
                    fieldProfile.den.cutoff_band > 0.0f;
                const float cutoff = fieldProfile.den.render_cutoff;
                const float cutoffBand = fieldProfile.den.cutoff_band;

                for (int t = 0; t < frames; t += stepT) {
                    const auto& leafValues = leafValuesByT[static_cast<size_t>(t)];
                    const auto& grid4Values = grid4ValuesByT[static_cast<size_t>(t)];
                    for (int lz = 0; lz < BT_LEAF_SIZE; lz += stepZ) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ly += stepY) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; lx += stepX) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx >= width || vy >= height || vz >= depth) continue;
                                const float raw = volumeSequence[t][vz][vy][vx];
                                const float basePred = leafValues[static_cast<size_t>((lz * BT_LEAF_SIZE + ly) * BT_LEAF_SIZE + lx)];
                                const float gridPred = sampleGrid4TrilinearSingleFrame(grid4Values, lx, ly, lz);
                                localSamples.push_back({lx, ly, lz, t, raw, basePred, gridPred});
                                const double eb = static_cast<double>(basePred) - raw;
                                const double eg = static_cast<double>(gridPred) - raw;
                                localBaseSum2 += eb * eb;
                                localGridSum2 += eg * eg;
                                localGridAbsErrs.push_back(std::abs(gridPred - raw));
                                rawSum += raw;
                                rawSum2 += static_cast<double>(raw) * raw;
                                ++rawCount;
                                localMin = std::min(localMin, raw);
                                localMax = std::max(localMax, raw);
                                if (useVisibilityAware) {
                                    if (raw >= cutoff - cutoffBand &&
                                        raw <= cutoff + cutoffBand) {
                                        ++visCount;
                                    }
                                    if (cutoffStateWithBand(raw, cutoff, cutoffBand) !=
                                        cutoffStateWithBand(gridPred, cutoff, cutoffBand)) {
                                        ++visMismatchCount;
                                    }
                                }
                            }
                        }
                    }

                    for (int lz = 0; lz + stepZ < BT_LEAF_SIZE; lz += stepZ) {
                        for (int ly = 0; ly + stepY < BT_LEAF_SIZE; ly += stepY) {
                            for (int lx = 0; lx + stepX < BT_LEAF_SIZE; lx += stepX) {
                                const int idx0 = (lz * BT_LEAF_SIZE + ly) * BT_LEAF_SIZE + lx;
                                const int idxX = (lz * BT_LEAF_SIZE + ly) * BT_LEAF_SIZE + (lx + stepX);
                                const int idxY = (lz * BT_LEAF_SIZE + (ly + stepY)) * BT_LEAF_SIZE + lx;
                                const int idxZ = (((lz + stepZ) * BT_LEAF_SIZE) + ly) * BT_LEAF_SIZE + lx;
                                gradAccum += std::abs(leafValues[static_cast<size_t>(idx0)] - leafValues[static_cast<size_t>(idxX)]);
                                gradAccum += std::abs(leafValues[static_cast<size_t>(idx0)] - leafValues[static_cast<size_t>(idxY)]);
                                gradAccum += std::abs(leafValues[static_cast<size_t>(idx0)] - leafValues[static_cast<size_t>(idxZ)]);
                                gradCount += 3;
                            }
                        }
                    }
                }

                const double localBaseRmse = std::sqrt(localBaseSum2 / std::max<size_t>(1, localSamples.size()));
                const double localGridRmse = std::sqrt(localGridSum2 / std::max<size_t>(1, localSamples.size()));
                const double localRange = std::max(1e-6, static_cast<double>(localMax - localMin));
                std::sort(localGridAbsErrs.begin(), localGridAbsErrs.end());
                const double localGridP99 = percentileFromSorted(localGridAbsErrs, 0.99);
                const double rawMean = rawSum / std::max(1, rawCount);
                const double rawVar = std::max(0.0, rawSum2 / std::max(1, rawCount) - rawMean * rawMean);
                const double rawStd = std::sqrt(rawVar);
                const double visibilityFrac = static_cast<double>(visCount) / std::max<size_t>(1, localSamples.size());
                const double visibilityMismatchFrac = static_cast<double>(visMismatchCount) / std::max<size_t>(1, localSamples.size());
                const double gradMean = gradAccum / std::max(1, gradCount);

                const double rmseTerm = std::clamp((localGridRmse / std::max(localBaseRmse, 1e-6) - 1.0) / 4.0, 0.0, 1.0);
                const double p99Term = std::clamp((localGridP99 / localRange) / 0.25, 0.0, 1.0);
                const double visBandTerm = std::clamp(visibilityFrac / 0.25, 0.0, 1.0);
                const double visMismatchTerm = std::clamp(visibilityMismatchFrac / 0.10, 0.0, 1.0);
                const double visTerm = std::max(visBandTerm, visMismatchTerm);
                const double gradTerm = std::clamp((gradMean / localRange) / 0.25, 0.0, 1.0);
                const double varTerm = std::clamp((rawStd / localRange) / 0.5, 0.0, 1.0);
                const double score = wRmse * rmseTerm + wP99 * p99Term + wVis * visTerm + wGrad * gradTerm + wVar * varTerm;
                scoreSum += score;
                scoreMax = std::max(scoreMax, score);

                int chosenLevel = 0;
                std::vector<std::vector<float>> fine6Series;
                std::vector<std::vector<float>> fine8Series;
                double localFine6Rmse = std::numeric_limits<double>::infinity();
                double localFine8Rmse = std::numeric_limits<double>::infinity();
                double localFine6VisMismatch = std::numeric_limits<double>::infinity();
                double localFine8VisMismatch = std::numeric_limits<double>::infinity();

                if (score >= scoreT1 || visibilityMismatchFrac > 0.03) {
                    fine6Series.assign(static_cast<size_t>(6 * 6 * 6), std::vector<float>(static_cast<size_t>(frames), 0.0f));
                    for (int t = 0; t < frames; ++t) {
                        const auto& rawLeafValues = rawLeafValuesByT[static_cast<size_t>(t)];
                        const auto& grid4Values = grid4ValuesByT[static_cast<size_t>(t)];
                        int fidx = 0;
                        for (int gz = 0; gz < 6; ++gz) {
                            for (int gy = 0; gy < 6; ++gy) {
                                for (int gx = 0; gx < 6; ++gx, ++fidx) {
                                    const float fx = fine6Coords[static_cast<size_t>(gx)];
                                    const float fy = fine6Coords[static_cast<size_t>(gy)];
                                    const float fz = fine6Coords[static_cast<size_t>(gz)];
                                    const float rawLike = sampleDenseLeafTrilinearSingleFrame(rawLeafValues, fx, fy, fz);
                                    const float coarse = sampleGrid4TrilinearSingleFrameAtFloat(grid4Values, fx, fy, fz);
                                    fine6Series[static_cast<size_t>(fidx)][static_cast<size_t>(t)] = rawLike - coarse;
                                }
                            }
                        }
                    }
                    double sum2 = 0.0;
                    int visMismatch6 = 0;
                    for (const auto& s : localSamples) {
                        std::vector<float> controlValues(static_cast<size_t>(6 * 6 * 6), 0.0f);
                        for (int i = 0; i < 6 * 6 * 6; ++i) {
                            controlValues[static_cast<size_t>(i)] = fine6Series[static_cast<size_t>(i)][static_cast<size_t>(s.t)];
                        }
                        const float pred = s.gridPred + sampleRegularControlGridSingleFrame(controlValues, 6, s.lx, s.ly, s.lz);
                        const double e = static_cast<double>(pred) - s.raw;
                        sum2 += e * e;
                        if (useVisibilityAware &&
                            cutoffStateWithBand(s.raw, cutoff, cutoffBand) != cutoffStateWithBand(pred, cutoff, cutoffBand)) {
                            ++visMismatch6;
                        }
                    }
                    localFine6Rmse = std::sqrt(sum2 / std::max<size_t>(1, localSamples.size()));
                    localFine6VisMismatch = static_cast<double>(visMismatch6) / std::max<size_t>(1, localSamples.size());
                    chosenLevel = 6;

                    if (score >= scoreT2 || localFine6VisMismatch > 0.02) {
                        fine8Series.assign(static_cast<size_t>(8 * 8 * 8), std::vector<float>(static_cast<size_t>(frames), 0.0f));
                        for (int t = 0; t < frames; ++t) {
                            const auto& rawLeafValues = rawLeafValuesByT[static_cast<size_t>(t)];
                            const auto& grid4Values = grid4ValuesByT[static_cast<size_t>(t)];
                            int fidx = 0;
                            for (int gz = 0; gz < 8; ++gz) {
                                for (int gy = 0; gy < 8; ++gy) {
                                    for (int gx = 0; gx < 8; ++gx, ++fidx) {
                                        const float fx = fine8Coords[static_cast<size_t>(gx)];
                                        const float fy = fine8Coords[static_cast<size_t>(gy)];
                                        const float fz = fine8Coords[static_cast<size_t>(gz)];
                                        const float rawLike = sampleDenseLeafTrilinearSingleFrame(rawLeafValues, fx, fy, fz);
                                        const float coarse = sampleGrid4TrilinearSingleFrameAtFloat(grid4Values, fx, fy, fz);
                                        fine8Series[static_cast<size_t>(fidx)][static_cast<size_t>(t)] = rawLike - coarse;
                                    }
                                }
                            }
                        }
                        double sum28 = 0.0;
                        int visMismatch8 = 0;
                        for (const auto& s : localSamples) {
                            std::vector<float> controlValues(static_cast<size_t>(8 * 8 * 8), 0.0f);
                            for (int i = 0; i < 8 * 8 * 8; ++i) {
                                controlValues[static_cast<size_t>(i)] = fine8Series[static_cast<size_t>(i)][static_cast<size_t>(s.t)];
                            }
                            const float pred = s.gridPred + sampleRegularControlGridSingleFrame(controlValues, 8, s.lx, s.ly, s.lz);
                            const double e = static_cast<double>(pred) - s.raw;
                            sum28 += e * e;
                            if (useVisibilityAware &&
                                cutoffStateWithBand(s.raw, cutoff, cutoffBand) != cutoffStateWithBand(pred, cutoff, cutoffBand)) {
                                ++visMismatch8;
                            }
                        }
                        localFine8Rmse = std::sqrt(sum28 / std::max<size_t>(1, localSamples.size()));
                        localFine8VisMismatch = static_cast<double>(visMismatch8) / std::max<size_t>(1, localSamples.size());
                        const bool betterRmse = (localFine8Rmse < localFine6Rmse * fine8Improve);
                        const bool betterVisibility =
                            useVisibilityAware &&
                            localFine6VisMismatch > 0.0 &&
                            localFine8VisMismatch < localFine6VisMismatch * 0.80;
                        if (betterRmse || betterVisibility) {
                            chosenLevel = 8;
                        }
                    }
                }

                if (chosenLevel == 0) ++coarseLeaves;
                else if (chosenLevel == 6) ++fine6Leaves;
                else ++fine8Leaves;

                if (chosenLevel == 6) {
                    for (const auto& seq : fine6Series) {
                        std::vector<int> kfIdx = detectKeyFrames(seq, 0.0, fieldProfile);
                        fine6CtrlKF += static_cast<long long>(kfIdx.size());
                    }
                } else if (chosenLevel == 8) {
                    for (const auto& seq : fine8Series) {
                        std::vector<int> kfIdx = detectKeyFrames(seq, 0.0, fieldProfile);
                        fine8CtrlKF += static_cast<long long>(kfIdx.size());
                    }
                }

                if (!exportFrameBuffer.empty()) {
                    const int t = spatialProbe.exportFrame;
                    const auto& grid4Values = grid4ValuesByT[static_cast<size_t>(t)];
                    std::vector<float> fineControlValues;
                    if (chosenLevel == 6) {
                        fineControlValues.assign(static_cast<size_t>(6 * 6 * 6), 0.0f);
                        for (int i = 0; i < 6 * 6 * 6; ++i) {
                            fineControlValues[static_cast<size_t>(i)] = fine6Series[static_cast<size_t>(i)][static_cast<size_t>(t)];
                        }
                    } else if (chosenLevel == 8) {
                        fineControlValues.assign(static_cast<size_t>(8 * 8 * 8), 0.0f);
                        for (int i = 0; i < 8 * 8 * 8; ++i) {
                            fineControlValues[static_cast<size_t>(i)] = fine8Series[static_cast<size_t>(i)][static_cast<size_t>(t)];
                        }
                    }

                    for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx >= width || vy >= height || vz >= depth) continue;
                                float pred = sampleGrid4TrilinearSingleFrame(grid4Values, lx, ly, lz);
                                if (chosenLevel == 6) {
                                    pred += sampleRegularControlGridSingleFrame(fineControlValues, 6, lx, ly, lz);
                                } else if (chosenLevel == 8) {
                                    pred += sampleRegularControlGridSingleFrame(fineControlValues, 8, lx, ly, lz);
                                }
                                exportFrameBuffer[(static_cast<size_t>(vz) * height + static_cast<size_t>(vy)) * width + static_cast<size_t>(vx)] = pred;
                            }
                        }
                    }
                }

                for (const auto& s : localSamples) {
                    float adaptivePred = s.gridPred;
                    if (chosenLevel == 6) {
                        std::vector<float> controlValues(static_cast<size_t>(6 * 6 * 6), 0.0f);
                        for (int i = 0; i < 6 * 6 * 6; ++i) {
                            controlValues[static_cast<size_t>(i)] = fine6Series[static_cast<size_t>(i)][static_cast<size_t>(s.t)];
                        }
                        adaptivePred += sampleRegularControlGridSingleFrame(controlValues, 6, s.lx, s.ly, s.lz);
                    } else if (chosenLevel == 8) {
                        std::vector<float> controlValues(static_cast<size_t>(8 * 8 * 8), 0.0f);
                        for (int i = 0; i < 8 * 8 * 8; ++i) {
                            controlValues[static_cast<size_t>(i)] = fine8Series[static_cast<size_t>(i)][static_cast<size_t>(s.t)];
                        }
                        adaptivePred += sampleRegularControlGridSingleFrame(controlValues, 8, s.lx, s.ly, s.lz);
                    }
                    const float eb = std::abs(s.basePred - s.raw);
                    const float eg = std::abs(s.gridPred - s.raw);
                    const float ea = std::abs(adaptivePred - s.raw);
                    baselineErrs.push_back(eb);
                    gridErrs.push_back(eg);
                    adaptiveErrs.push_back(ea);
                    baseSum2 += static_cast<double>(eb) * eb;
                    gridSum2 += static_cast<double>(eg) * eg;
                    adaptiveSum2 += static_cast<double>(ea) * ea;
                    ++samples;
                }
            }
        }
    }

    std::sort(baselineErrs.begin(), baselineErrs.end());
    std::sort(gridErrs.begin(), gridErrs.end());
    std::sort(adaptiveErrs.begin(), adaptiveErrs.end());
    const size_t idx999 = std::min(baselineErrs.size() - 1, static_cast<size_t>(0.999 * static_cast<double>(baselineErrs.size() - 1)));
    const double psnrPeak = useDataRangeForPsnr ? std::max(1e-6, dataMax - dataMin) : 255.0;
    const double baseRmse = std::sqrt(baseSum2 / std::max<long long>(1, samples));
    const double gridRmse = std::sqrt(gridSum2 / std::max<long long>(1, samples));
    const double adaptiveRmse = std::sqrt(adaptiveSum2 / std::max<long long>(1, samples));

    const fs::path reportPath = fs::path("spatial_probe_reports") / (fs::path(inputFile).stem().string() + "_grid4_fine_adaptive_score_sampled_full_probe.md");
    std::ofstream md(reportPath);
    md << "# Grid4 + Adaptive Fine Grid Score-Based Sampled Full-Volume Probe\n\n";
    md << "- score rule: `coarse if score<T1`, `fine6 if T1<=score<T2`, `fine8 if score>=T2 and improves over fine6 enough`\n";
    md << "- visibility-aware override: `promote to fine6 if grid visibility mismatch is high; allow fine8 if it significantly reduces visibility mismatch`\n";
    md << "- T1/T2: `" << scoreT1 << " / " << scoreT2 << "`\n";
    md << "- fine8 improve factor: `" << fine8Improve << "` (require `fine8Rmse < fine6Rmse * factor`)\n";
    md << "- weights rmse/p99/vis/grad/var: `" << wRmse << " / " << wP99 << " / " << wVis << " / " << wGrad << " / " << wVar << "`\n";
    md << "- score mean/max: `" << (scoreSum / std::max(1, leafNX * leafNY * leafNZ)) << " / " << scoreMax << "`\n";
    md << "- sampled stride: `x=" << stepX << ", y=" << stepY << ", z=" << stepZ << ", t=" << stepT << "`\n";
    md << "- samples: `" << samples << "`\n";
    md << "- chosen leaves coarse/fine6/fine8: `" << coarseLeaves << " / " << fine6Leaves << " / " << fine8Leaves << "`\n";
    md << "- baseline RMSE: `" << baseRmse << "`\n";
    md << "- baseline P99.9: `" << baselineErrs[idx999] << "`\n";
    md << "- baseline PSNR: `" << computePsnr(baseRmse, psnrPeak) << "`\n";
    md << "- grid4 RMSE: `" << gridRmse << "`\n";
    md << "- grid4 P99.9: `" << gridErrs[idx999] << "`\n";
    md << "- grid4 PSNR: `" << computePsnr(gridRmse, psnrPeak) << "`\n";
    md << "- adaptive RMSE: `" << adaptiveRmse << "`\n";
    md << "- adaptive P99.9: `" << adaptiveErrs[idx999] << "`\n";
    md << "- adaptive PSNR: `" << computePsnr(adaptiveRmse, psnrPeak) << "`\n";
    md << "- KF coarse/fine6/fine8: `" << coarseCtrlKF << " / " << fine6CtrlKF << " / " << fine8CtrlKF << "`\n";
    md.close();

    if (!exportFrameBuffer.empty()) {
        const fs::path outDir = spatialProbe.exportDir.empty()
            ? (fs::path("smokeDate/render_ready/multiscale_probe") / fs::path(inputFile).stem())
            : fs::path(spatialProbe.exportDir);
        fs::create_directories(outDir);
        const std::string exportStem = spatialProbe.exportName.empty()
            ? (fs::path(inputFile).stem().string() + "_f" + std::to_string(spatialProbe.exportFrame) + "_grid4fas")
            : spatialProbe.exportName;
        const fs::path rawPath = outDir / (exportStem + ".raw");
        const fs::path metaPath = outDir / (exportStem + ".metadata.json");

        std::ofstream rawOut(rawPath, std::ios::binary);
        rawOut.write(reinterpret_cast<const char*>(exportFrameBuffer.data()),
                     static_cast<std::streamsize>(exportFrameBuffer.size() * sizeof(float)));
        rawOut.close();

        std::array<int, 3> exportBboxMin{0, 0, 0};
        std::array<int, 3> exportBboxMax{width - 1, height - 1, depth - 1};
        try {
            const fs::path srcMetaPath = guessMetadataPathForRaw(inputFile);
            if (!srcMetaPath.empty()) {
                const auto srcMeta = vdbtools::loadFrameMetadata(srcMetaPath);
                if (srcMeta.width == width && srcMeta.height == height && srcMeta.depth == depth) {
                    exportBboxMin = srcMeta.bboxMin;
                    exportBboxMax = srcMeta.bboxMax;
                }
            }
        } catch (...) {
            // Fall back to zero-based bbox if source metadata is unavailable.
        }

        std::ofstream metaOut(metaPath);
        metaOut << "{\n";
        metaOut << "  \"source_dir\": \"adaptive_multiscale_probe\",\n";
        metaOut << "  \"grid_name\": \"density\",\n";
        metaOut << "  \"width\": " << width << ",\n";
        metaOut << "  \"height\": " << height << ",\n";
        metaOut << "  \"depth\": " << depth << ",\n";
        metaOut << "  \"frames\": 1,\n";
        metaOut << "  \"bbox_min\": [" << exportBboxMin[0] << ", " << exportBboxMin[1] << ", " << exportBboxMin[2] << "],\n";
        metaOut << "  \"bbox_max\": [" << exportBboxMax[0] << ", " << exportBboxMax[1] << ", " << exportBboxMax[2] << "],\n";
        metaOut << "  \"data_min\": " << dataMin << ",\n";
        metaOut << "  \"data_max\": " << dataMax << "\n";
        metaOut << "}\n";
        metaOut.close();
    }

    printf("\n=== Spatial Probe: grid4 + adaptive fine (score) sampled full-volume ===\n");
    printf("  T1/T2             : %.3f / %.3f\n", scoreT1, scoreT2);
    printf("  Fine8 improve     : %.3f\n", fine8Improve);
    printf("  Weights           : rmse=%.2f p99=%.2f vis=%.2f grad=%.2f var=%.2f\n", wRmse, wP99, wVis, wGrad, wVar);
    printf("  Score mean/max    : %.4f / %.4f\n", scoreSum / std::max(1, leafNX * leafNY * leafNZ), scoreMax);
    printf("  Sample stride     : x=%d y=%d z=%d t=%d\n", stepX, stepY, stepZ, stepT);
    printf("  Samples           : %lld\n", samples);
    printf("  Leaves c/6/8      : %lld / %lld / %lld\n", coarseLeaves, fine6Leaves, fine8Leaves);
    printf("  Baseline RMSE     : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           baseRmse, baselineErrs[idx999], computePsnr(baseRmse, psnrPeak));
    printf("  Grid4 RMSE        : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           gridRmse, gridErrs[idx999], computePsnr(gridRmse, psnrPeak));
    printf("  Adaptive RMSE     : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           adaptiveRmse, adaptiveErrs[idx999], computePsnr(adaptiveRmse, psnrPeak));
    printf("  KF coarse/6/8     : %lld / %lld / %lld\n", coarseCtrlKF, fine6CtrlKF, fine8CtrlKF);
    printf("  Report            : %s\n", reportPath.string().c_str());
    if (!exportFrameBuffer.empty()) {
        const fs::path outDir = spatialProbe.exportDir.empty()
            ? (fs::path("smokeDate/render_ready/multiscale_probe") / fs::path(inputFile).stem())
            : fs::path(spatialProbe.exportDir);
        const std::string exportStem = spatialProbe.exportName.empty()
            ? (fs::path(inputFile).stem().string() + "_f" + std::to_string(spatialProbe.exportFrame) + "_grid4fas")
            : spatialProbe.exportName;
        printf("  Export           : %s\\%s.raw\n", outDir.string().c_str(), exportStem.c_str());
    }
    return 0;
}

static int runGrid4HotspotProbe(const std::string& inputFile,
                                const Volume4D& volumeSequence,
                                const CompressedVolume4D& compressedVolume,
                                int width, int height, int depth, int frames,
                                const SpatialProbeOptions& spatialProbe) {
    namespace fs = std::filesystem;
    fs::create_directories("spatial_probe_reports");

    const std::string stem = fs::path(inputFile).stem().string();
    std::string csvPath = spatialProbe.leafCsv;
    if (csvPath.empty()) {
        csvPath = (fs::path("leaf_error_reports") / (stem + "_leaf_error_report.csv")).string();
    }

    const auto hotspots = loadHotspotLeafRows(csvPath, spatialProbe.topK);
    if (hotspots.empty()) {
        std::cerr << "Spatial probe: no hotspot rows found in " << csvPath << std::endl;
        return 1;
    }

    std::vector<GridProbeLeafResult> results;
    results.reserve(hotspots.size());
    const auto ctrl = grid4ControlCoords();

    for (const auto& row : hotspots) {
        GridProbeLeafResult out;
        out.base = row;
        std::vector<float> controlValues(4u * 4u * 4u * static_cast<size_t>(frames), 0.0f);

        for (int gz = 0; gz < 4; ++gz) {
            for (int gy = 0; gy < 4; ++gy) {
                for (int gx = 0; gx < 4; ++gx) {
                    const int vx = row.bx * BT_LEAF_SIZE + ctrl[gx];
                    const int vy = row.by * BT_LEAF_SIZE + ctrl[gy];
                    const int vz = row.bz * BT_LEAF_SIZE + ctrl[gz];
                    if (vx >= width || vy >= height || vz >= depth) continue;
                    const auto& seq = compressedVolume[vz][vy][vx];
                    out.controlKF += static_cast<long long>(seq.size());
                    for (int t = 0; t < frames; ++t) {
                        const size_t idx = (((static_cast<size_t>(gz) * 4u + static_cast<size_t>(gy)) * 4u
                                           + static_cast<size_t>(gx)) * static_cast<size_t>(frames))
                                           + static_cast<size_t>(t);
                        controlValues[idx] = sampleTemporalKfs(seq, t);
                    }
                }
            }
        }

        std::vector<float> errs;
        errs.reserve(static_cast<size_t>(BT_LEAF_VOXELS) * static_cast<size_t>(frames));
        double errSum = 0.0;
        double errSum2 = 0.0;
        float errMax = 0.0f;
        long long samples = 0;

        for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
            for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                    const int vx = row.bx * BT_LEAF_SIZE + lx;
                    const int vy = row.by * BT_LEAF_SIZE + ly;
                    const int vz = row.bz * BT_LEAF_SIZE + lz;
                    if (vx >= width || vy >= height || vz >= depth) continue;
                    out.fullLeafKF += static_cast<long long>(compressedVolume[vz][vy][vx].size());
                    for (int t = 0; t < frames; ++t) {
                        const float pred = sampleGrid4Trilinear(controlValues, frames, t, lx, ly, lz);
                        const float raw = volumeSequence[t][vz][vy][vx];
                        const float e = std::abs(pred - raw);
                        errs.push_back(e);
                        errSum += e;
                        errSum2 += static_cast<double>(e) * e;
                        errMax = std::max(errMax, e);
                        ++samples;
                    }
                }
            }
        }

        std::sort(errs.begin(), errs.end());
        const size_t idx99 = std::min(errs.size() - 1, static_cast<size_t>(0.99 * static_cast<double>(errs.size() - 1)));
        const size_t idx999 = std::min(errs.size() - 1, static_cast<size_t>(0.999 * static_cast<double>(errs.size() - 1)));
        out.gridMean = errSum / std::max<long long>(1, samples);
        out.gridRmse = std::sqrt(errSum2 / std::max<long long>(1, samples));
        out.gridP99 = errs[idx99];
        out.gridP999 = errs[idx999];
        out.gridMax = errMax;
        results.push_back(out);
    }

    double baseRmseMean = 0.0, gridRmseMean = 0.0, baseP999Mean = 0.0, gridP999Mean = 0.0;
    long long controlKF = 0, fullLeafKF = 0;
    int betterCount = 0;
    for (const auto& r : results) {
        baseRmseMean += r.base.baselineRmse;
        gridRmseMean += r.gridRmse;
        baseP999Mean += r.base.baselineP999;
        gridP999Mean += r.gridP999;
        controlKF += r.controlKF;
        fullLeafKF += r.fullLeafKF;
        if (r.gridRmse < r.base.baselineRmse) ++betterCount;
    }
    const double denom = std::max<size_t>(1, results.size());
    baseRmseMean /= denom;
    gridRmseMean /= denom;
    baseP999Mean /= denom;
    gridP999Mean /= denom;

    const fs::path reportPath = fs::path("spatial_probe_reports") / (stem + "_grid4_hotspot_probe.md");
    std::ofstream md(reportPath);
    md << "# Grid4 Hotspot Probe\n\n";
    md << "- dataset: `" << stem << "`\n";
    md << "- mode: `grid4 control grid + trilinear interpolation`\n";
    md << "- hotspot csv: `" << csvPath << "`\n";
    md << "- topK: `" << results.size() << "`\n\n";
    md << "## Summary\n\n";
    md << "- baseline hotspot RMSE mean: `" << baseRmseMean << "`\n";
    md << "- grid4 hotspot RMSE mean: `" << gridRmseMean << "`\n";
    md << "- baseline hotspot P99.9 mean: `" << baseP999Mean << "`\n";
    md << "- grid4 hotspot P99.9 mean: `" << gridP999Mean << "`\n";
    md << "- improved leaf count: `" << betterCount << " / " << results.size() << "`\n";
    md << "- controlKF/fullLeafKF ratio: `" << (fullLeafKF > 0 ? static_cast<double>(controlKF) / fullLeafKF : 0.0) << "`\n\n";
    md << "| leaf | block | amp | baseline RMSE | grid4 RMSE | baseline P99.9 | grid4 P99.9 |\n";
    md << "|---:|---|---:|---:|---:|---:|---:|\n";
    for (const auto& r : results) {
        md << "| " << r.base.leafId
           << " | (" << r.base.bx << "," << r.base.by << "," << r.base.bz << ")"
           << " | " << r.base.ampVsTemporal
           << " | " << r.base.baselineRmse
           << " | " << r.gridRmse
           << " | " << r.base.baselineP999
           << " | " << r.gridP999
           << " |\n";
    }
    md.close();

    printf("\n=== Spatial Probe: grid4 hotspot leaf test ===\n");
    printf("  Hotspot CSV     : %s\n", csvPath.c_str());
    printf("  TopK            : %zu\n", results.size());
    printf("  Baseline RMSE   : %.6f\n", baseRmseMean);
    printf("  Grid4 RMSE      : %.6f\n", gridRmseMean);
    printf("  Baseline P99.9  : %.6f\n", baseP999Mean);
    printf("  Grid4 P99.9     : %.6f\n", gridP999Mean);
    printf("  Improved leaves : %d / %zu\n", betterCount, results.size());
    printf("  controlKF/fullKF ratio: %.4f\n",
           fullLeafKF > 0 ? static_cast<double>(controlKF) / fullLeafKF : 0.0);
    printf("  Report          : %s\n", reportPath.string().c_str());
    return 0;
}

static int runGrid4SampledFullProbe(const std::string& inputFile,
                                    const Volume4D& volumeSequence,
                                    const CompressedVolume4D& compressedVolume,
                                    int width, int height, int depth, int frames,
                                    const FieldProfile& fieldProfile,
                                    bool enableBackgroundElision,
                                    bool enableBlockAwareCluster,
                                    bool enableBudgetAwareCluster,
                                    bool enableGuardedMedoidCluster,
                                    bool enableValidateFallback,
                                    bool enableHotspotSecondPass,
                                    double clusterThr,
                                    double dataMin,
                                    double dataMax,
                                    bool useDataRangeForPsnr) {
    namespace fs = std::filesystem;
    fs::create_directories("spatial_probe_reports");

    BlockTree btree;
    btree.build(compressedVolume, width, height, depth, frames,
                -1.0, clusterThr, fieldProfile, enableBackgroundElision,
                enableBlockAwareCluster, enableBudgetAwareCluster, enableGuardedMedoidCluster, false, false,
                &volumeSequence, enableValidateFallback, enableHotspotSecondPass);
    btree.flattenLeaves();

    const int stepX = 4, stepY = 4, stepZ = 4, stepT = 4;
    std::vector<float> baselineErrs;
    std::vector<float> gridErrs;
    baselineErrs.reserve(static_cast<size_t>((width / stepX + 1) * (height / stepY + 1) * (depth / stepZ + 1) * (frames / stepT + 1)));
    gridErrs.reserve(baselineErrs.capacity());
    double baseSum = 0.0, baseSum2 = 0.0;
    double gridSum = 0.0, gridSum2 = 0.0;
    float baseMax = 0.0f, gridMax = 0.0f;
    long long samples = 0;

    for (int z = 0; z < depth; z += stepZ) {
        for (int y = 0; y < height; y += stepY) {
            for (int x = 0; x < width; x += stepX) {
                const int bx = x / BT_LEAF_SIZE;
                const int by = y / BT_LEAF_SIZE;
                const int bz = z / BT_LEAF_SIZE;
                const int lx = x % BT_LEAF_SIZE;
                const int ly = y % BT_LEAF_SIZE;
                const int lz = z % BT_LEAF_SIZE;
                for (int t = 0; t < frames; t += stepT) {
                    const float raw = volumeSequence[t][z][y][x];
                    const float basePred = btree.sampleFlat(x, y, z, static_cast<float>(t));
                    const float gridPred = sampleGrid4FromCompVolume(compressedVolume, width, height, depth,
                                                                     bx, by, bz, lx, ly, lz, t);
                    const float eb = std::abs(basePred - raw);
                    const float eg = std::abs(gridPred - raw);
                    baselineErrs.push_back(eb);
                    gridErrs.push_back(eg);
                    baseSum += eb;
                    baseSum2 += static_cast<double>(eb) * eb;
                    gridSum += eg;
                    gridSum2 += static_cast<double>(eg) * eg;
                    baseMax = std::max(baseMax, eb);
                    gridMax = std::max(gridMax, eg);
                    ++samples;
                }
            }
        }
    }

    std::sort(baselineErrs.begin(), baselineErrs.end());
    std::sort(gridErrs.begin(), gridErrs.end());
    const size_t idx99 = std::min(baselineErrs.size() - 1, static_cast<size_t>(0.99 * static_cast<double>(baselineErrs.size() - 1)));
    const size_t idx999 = std::min(baselineErrs.size() - 1, static_cast<size_t>(0.999 * static_cast<double>(baselineErrs.size() - 1)));

    long long controlKF = 0;
    long long temporalKF = 0;
    const auto ctrl = grid4ControlCoords();
    for (int bz = 0; bz < (depth + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE; ++bz) {
        for (int by = 0; by < (height + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE; ++by) {
            for (int bx = 0; bx < (width + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE; ++bx) {
                for (int gz = 0; gz < 4; ++gz) {
                    for (int gy = 0; gy < 4; ++gy) {
                        for (int gx = 0; gx < 4; ++gx) {
                            const int vx = std::min(bx * BT_LEAF_SIZE + ctrl[gx], width  - 1);
                            const int vy = std::min(by * BT_LEAF_SIZE + ctrl[gy], height - 1);
                            const int vz = std::min(bz * BT_LEAF_SIZE + ctrl[gz], depth  - 1);
                            controlKF += static_cast<long long>(compressedVolume[vz][vy][vx].size());
                        }
                    }
                }
            }
        }
    }
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                temporalKF += static_cast<long long>(compressedVolume[z][y][x].size());
            }
        }
    }

    const double baseRmse = std::sqrt(baseSum2 / std::max<long long>(1, samples));
    const double gridRmse = std::sqrt(gridSum2 / std::max<long long>(1, samples));
    const double psnrPeak = useDataRangeForPsnr ? std::max(1e-6, dataMax - dataMin) : 255.0;
    const fs::path reportPath = fs::path("spatial_probe_reports") / (fs::path(inputFile).stem().string() + "_grid4_sampled_full_probe.md");
    std::ofstream md(reportPath);
    md << "# Grid4 Sampled Full-Volume Probe\n\n";
    md << "- sampled stride: `x=" << stepX << ", y=" << stepY << ", z=" << stepZ << ", t=" << stepT << "`\n";
    md << "- samples: `" << samples << "`\n";
    md << "- baseline RMSE: `" << baseRmse << "`\n";
    md << "- baseline P99.9: `" << baselineErrs[idx999] << "`\n";
    md << "- baseline PSNR: `" << computePsnr(baseRmse, psnrPeak) << "`\n";
    md << "- grid4 RMSE: `" << gridRmse << "`\n";
    md << "- grid4 P99.9: `" << gridErrs[idx999] << "`\n";
    md << "- grid4 PSNR: `" << computePsnr(gridRmse, psnrPeak) << "`\n";
    md << "- controlKF/temporalKF: `" << controlKF << " / " << temporalKF << "` ratio=`"
       << (temporalKF > 0 ? static_cast<double>(controlKF) / temporalKF : 0.0) << "`\n";
    md.close();

    printf("\n=== Spatial Probe: grid4 sampled full-volume ===\n");
    printf("  Sample stride     : x=%d y=%d z=%d t=%d\n", stepX, stepY, stepZ, stepT);
    printf("  Samples           : %lld\n", samples);
    printf("  Baseline RMSE     : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           baseRmse, baselineErrs[idx999], computePsnr(baseRmse, psnrPeak));
    printf("  Grid4 RMSE        : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           gridRmse, gridErrs[idx999], computePsnr(gridRmse, psnrPeak));
    printf("  controlKF/temporalKF ratio: %.4f\n",
           temporalKF > 0 ? static_cast<double>(controlKF) / temporalKF : 0.0);
    printf("  Report            : %s\n", reportPath.string().c_str());
    return 0;
}

static int runPoly8SampledFullProbe(const std::string& inputFile,
                                    const Volume4D& volumeSequence,
                                    const CompressedVolume4D& compressedVolume,
                                    int width, int height, int depth, int frames,
                                    double dataMin,
                                    double dataMax,
                                    bool useDataRangeForPsnr) {
    namespace fs = std::filesystem;
    fs::create_directories("spatial_probe_reports");

    const int stepX = 4, stepY = 4, stepZ = 4, stepT = 4;
    std::vector<float> baselineErrs;
    std::vector<float> gridErrs;
    std::vector<float> polyErrs;
    baselineErrs.reserve(static_cast<size_t>((width / stepX + 1) * (height / stepY + 1) * (depth / stepZ + 1) * (frames / stepT + 1)));
    gridErrs.reserve(baselineErrs.capacity());
    polyErrs.reserve(baselineErrs.capacity());
    double baseSum2 = 0.0, gridSum2 = 0.0, polySum2 = 0.0;
    long long samples = 0;
    const auto ctrl = grid4ControlCoords();

    for (int bz = 0; bz < (depth + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE; ++bz) {
        for (int by = 0; by < (height + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE; ++by) {
            for (int bx = 0; bx < (width + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE; ++bx) {
                for (int t = 0; t < frames; t += stepT) {
                    std::array<float, BT_LEAF_VOXELS> leafValues{};
                    std::array<float, 64> grid4Values{};

                    int idx = 0;
                    for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; ++lx, ++idx) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx < width && vy < height && vz < depth) {
                                    leafValues[static_cast<size_t>(idx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                                } else {
                                    leafValues[static_cast<size_t>(idx)] = 0.0f;
                                }
                            }
                        }
                    }

                    int cidx = 0;
                    for (int gz = 0; gz < 4; ++gz) {
                        for (int gy = 0; gy < 4; ++gy) {
                            for (int gx = 0; gx < 4; ++gx, ++cidx) {
                                const int vx = std::min(bx * BT_LEAF_SIZE + ctrl[gx], width - 1);
                                const int vy = std::min(by * BT_LEAF_SIZE + ctrl[gy], height - 1);
                                const int vz = std::min(bz * BT_LEAF_SIZE + ctrl[gz], depth - 1);
                                grid4Values[static_cast<size_t>(cidx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                            }
                        }
                    }

                    const auto coeffs = fitPoly8Coeffs(leafValues);

                    for (int lz = 0; lz < BT_LEAF_SIZE; lz += stepZ) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ly += stepY) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; lx += stepX) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx >= width || vy >= height || vz >= depth) continue;

                                const float raw = volumeSequence[t][vz][vy][vx];
                                const int localIdx = (lz * BT_LEAF_SIZE + ly) * BT_LEAF_SIZE + lx;
                                const float basePred = leafValues[static_cast<size_t>(localIdx)];
                                const float gridPred = sampleGrid4TrilinearSingleFrame(grid4Values, lx, ly, lz);
                                const float polyPred = evalPoly8At(coeffs, lx, ly, lz);
                                const float eb = std::abs(basePred - raw);
                                const float eg = std::abs(gridPred - raw);
                                const float ep = std::abs(polyPred - raw);
                                baselineErrs.push_back(eb);
                                gridErrs.push_back(eg);
                                polyErrs.push_back(ep);
                                baseSum2 += static_cast<double>(eb) * eb;
                                gridSum2 += static_cast<double>(eg) * eg;
                                polySum2 += static_cast<double>(ep) * ep;
                                ++samples;
                            }
                        }
                    }
                }
            }
        }
    }

    std::sort(baselineErrs.begin(), baselineErrs.end());
    std::sort(gridErrs.begin(), gridErrs.end());
    std::sort(polyErrs.begin(), polyErrs.end());
    const size_t idx999 = std::min(baselineErrs.size() - 1, static_cast<size_t>(0.999 * static_cast<double>(baselineErrs.size() - 1)));
    const double psnrPeak = useDataRangeForPsnr ? std::max(1e-6, dataMax - dataMin) : 255.0;

    const double baseRmse = std::sqrt(baseSum2 / std::max<long long>(1, samples));
    const double gridRmse = std::sqrt(gridSum2 / std::max<long long>(1, samples));
    const double polyRmse = std::sqrt(polySum2 / std::max<long long>(1, samples));

    const fs::path reportPath = fs::path("spatial_probe_reports") / (fs::path(inputFile).stem().string() + "_poly8_sampled_full_probe.md");
    std::ofstream md(reportPath);
    md << "# Poly8 Sampled Full-Volume Probe\n\n";
    md << "- sampled stride: `x=" << stepX << ", y=" << stepY << ", z=" << stepZ << ", t=" << stepT << "`\n";
    md << "- samples: `" << samples << "`\n";
    md << "- baseline RMSE: `" << baseRmse << "`\n";
    md << "- baseline P99.9: `" << baselineErrs[idx999] << "`\n";
    md << "- baseline PSNR: `" << computePsnr(baseRmse, psnrPeak) << "`\n";
    md << "- grid4 RMSE: `" << gridRmse << "`\n";
    md << "- grid4 P99.9: `" << gridErrs[idx999] << "`\n";
    md << "- grid4 PSNR: `" << computePsnr(gridRmse, psnrPeak) << "`\n";
    md << "- poly8 RMSE: `" << polyRmse << "`\n";
    md << "- poly8 P99.9: `" << polyErrs[idx999] << "`\n";
    md << "- poly8 PSNR: `" << computePsnr(polyRmse, psnrPeak) << "`\n";
    md.close();

    printf("\n=== Spatial Probe: poly8 sampled full-volume ===\n");
    printf("  Sample stride     : x=%d y=%d z=%d t=%d\n", stepX, stepY, stepZ, stepT);
    printf("  Samples           : %lld\n", samples);
    printf("  Baseline RMSE     : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           baseRmse, baselineErrs[idx999], computePsnr(baseRmse, psnrPeak));
    printf("  Grid4 RMSE        : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           gridRmse, gridErrs[idx999], computePsnr(gridRmse, psnrPeak));
    printf("  Poly8 RMSE        : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           polyRmse, polyErrs[idx999], computePsnr(polyRmse, psnrPeak));
    printf("  Report            : %s\n", reportPath.string().c_str());
    return 0;
}

static int runPoly11SampledFullProbe(const std::string& inputFile,
                                     const Volume4D& volumeSequence,
                                     const CompressedVolume4D& compressedVolume,
                                     int width, int height, int depth, int frames,
                                     double dataMin,
                                     double dataMax,
                                     bool useDataRangeForPsnr) {
    namespace fs = std::filesystem;
    fs::create_directories("spatial_probe_reports");

    const int stepX = 4, stepY = 4, stepZ = 4, stepT = 4;
    std::vector<float> baselineErrs;
    std::vector<float> gridErrs;
    std::vector<float> poly8Errs;
    std::vector<float> poly11Errs;
    std::vector<float> poly11f16Errs;
    std::vector<float> poly11cpInvErrs;
    baselineErrs.reserve(static_cast<size_t>((width / stepX + 1) * (height / stepY + 1) * (depth / stepZ + 1) * (frames / stepT + 1)));
    gridErrs.reserve(baselineErrs.capacity());
    poly8Errs.reserve(baselineErrs.capacity());
    poly11Errs.reserve(baselineErrs.capacity());
    poly11f16Errs.reserve(baselineErrs.capacity());
    poly11cpInvErrs.reserve(baselineErrs.capacity());
    double baseSum2 = 0.0, gridSum2 = 0.0, poly8Sum2 = 0.0, poly11Sum2 = 0.0, poly11f16Sum2 = 0.0, poly11cpInvSum2 = 0.0;
    long long samples = 0;
    const auto ctrl = grid4ControlCoords();
    const auto& polyCtrlPts = poly11ControlPoints();

    for (int bz = 0; bz < (depth + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE; ++bz) {
        for (int by = 0; by < (height + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE; ++by) {
            for (int bx = 0; bx < (width + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE; ++bx) {
                for (int t = 0; t < frames; t += stepT) {
                    std::array<float, BT_LEAF_VOXELS> leafValues{};
                    std::array<float, 64> grid4Values{};

                    int idx = 0;
                    for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; ++lx, ++idx) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx < width && vy < height && vz < depth) {
                                    leafValues[static_cast<size_t>(idx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                                } else {
                                    leafValues[static_cast<size_t>(idx)] = 0.0f;
                                }
                            }
                        }
                    }

                    int cidx = 0;
                    for (int gz = 0; gz < 4; ++gz) {
                        for (int gy = 0; gy < 4; ++gy) {
                            for (int gx = 0; gx < 4; ++gx, ++cidx) {
                                const int vx = std::min(bx * BT_LEAF_SIZE + ctrl[gx], width - 1);
                                const int vy = std::min(by * BT_LEAF_SIZE + ctrl[gy], height - 1);
                                const int vz = std::min(bz * BT_LEAF_SIZE + ctrl[gz], depth - 1);
                                grid4Values[static_cast<size_t>(cidx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                            }
                        }
                    }

                    const auto coeffs8 = fitPoly8Coeffs(leafValues);
                    const auto coeffs11 = fitPoly11Coeffs(leafValues);
                    std::array<float, 11> coeffs11f16{};
                    std::array<float, 11> controlVals{};
                    std::array<float, 11> controlValsF16{};
                    for (int i = 0; i < 11; ++i) {
                        coeffs11f16[static_cast<size_t>(i)] = f16_to_f32(f32_to_f16(coeffs11[static_cast<size_t>(i)]));
                        const int cx = polyCtrlPts[static_cast<size_t>(i)][0];
                        const int cy = polyCtrlPts[static_cast<size_t>(i)][1];
                        const int cz = polyCtrlPts[static_cast<size_t>(i)][2];
                        controlVals[static_cast<size_t>(i)] = evalPoly11At(coeffs11, cx, cy, cz);
                        controlValsF16[static_cast<size_t>(i)] = f16_to_f32(f32_to_f16(controlVals[static_cast<size_t>(i)]));
                    }
                    const auto coeffs11cpInv = recoverPoly11CoeffsFromControlValues(controlValsF16);

                    for (int lz = 0; lz < BT_LEAF_SIZE; lz += stepZ) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ly += stepY) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; lx += stepX) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx >= width || vy >= height || vz >= depth) continue;

                                const float raw = volumeSequence[t][vz][vy][vx];
                                const int localIdx = (lz * BT_LEAF_SIZE + ly) * BT_LEAF_SIZE + lx;
                                const float basePred = leafValues[static_cast<size_t>(localIdx)];
                                const float gridPred = sampleGrid4TrilinearSingleFrame(grid4Values, lx, ly, lz);
                                const float poly8Pred = evalPoly8At(coeffs8, lx, ly, lz);
                                const float poly11Pred = evalPoly11At(coeffs11, lx, ly, lz);
                                const float poly11f16Pred = evalPoly11At(coeffs11f16, lx, ly, lz);
                                const float poly11cpInvPred = evalPoly11At(coeffs11cpInv, lx, ly, lz);
                                const float eb = std::abs(basePred - raw);
                                const float eg = std::abs(gridPred - raw);
                                const float ep8 = std::abs(poly8Pred - raw);
                                const float ep11 = std::abs(poly11Pred - raw);
                                const float ep11f16 = std::abs(poly11f16Pred - raw);
                                const float ep11cpInv = std::abs(poly11cpInvPred - raw);
                                baselineErrs.push_back(eb);
                                gridErrs.push_back(eg);
                                poly8Errs.push_back(ep8);
                                poly11Errs.push_back(ep11);
                                poly11f16Errs.push_back(ep11f16);
                                poly11cpInvErrs.push_back(ep11cpInv);
                                baseSum2 += static_cast<double>(eb) * eb;
                                gridSum2 += static_cast<double>(eg) * eg;
                                poly8Sum2 += static_cast<double>(ep8) * ep8;
                                poly11Sum2 += static_cast<double>(ep11) * ep11;
                                poly11f16Sum2 += static_cast<double>(ep11f16) * ep11f16;
                                poly11cpInvSum2 += static_cast<double>(ep11cpInv) * ep11cpInv;
                                ++samples;
                            }
                        }
                    }
                }
            }
        }
    }

    std::sort(baselineErrs.begin(), baselineErrs.end());
    std::sort(gridErrs.begin(), gridErrs.end());
    std::sort(poly8Errs.begin(), poly8Errs.end());
    std::sort(poly11Errs.begin(), poly11Errs.end());
    std::sort(poly11f16Errs.begin(), poly11f16Errs.end());
    std::sort(poly11cpInvErrs.begin(), poly11cpInvErrs.end());
    const size_t idx999 = std::min(baselineErrs.size() - 1, static_cast<size_t>(0.999 * static_cast<double>(baselineErrs.size() - 1)));
    const double psnrPeak = useDataRangeForPsnr ? std::max(1e-6, dataMax - dataMin) : 255.0;

    const double baseRmse = std::sqrt(baseSum2 / std::max<long long>(1, samples));
    const double gridRmse = std::sqrt(gridSum2 / std::max<long long>(1, samples));
    const double poly8Rmse = std::sqrt(poly8Sum2 / std::max<long long>(1, samples));
    const double poly11Rmse = std::sqrt(poly11Sum2 / std::max<long long>(1, samples));
    const double poly11f16Rmse = std::sqrt(poly11f16Sum2 / std::max<long long>(1, samples));
    const double poly11cpInvRmse = std::sqrt(poly11cpInvSum2 / std::max<long long>(1, samples));

    const fs::path reportPath = fs::path("spatial_probe_reports") / (fs::path(inputFile).stem().string() + "_poly11_sampled_full_probe.md");
    std::ofstream md(reportPath);
    md << "# Poly11 Sampled Full-Volume Probe\n\n";
    md << "- sampled stride: `x=" << stepX << ", y=" << stepY << ", z=" << stepZ << ", t=" << stepT << "`\n";
    md << "- samples: `" << samples << "`\n";
    md << "- baseline RMSE: `" << baseRmse << "`\n";
    md << "- baseline P99.9: `" << baselineErrs[idx999] << "`\n";
    md << "- baseline PSNR: `" << computePsnr(baseRmse, psnrPeak) << "`\n";
    md << "- grid4 RMSE: `" << gridRmse << "`\n";
    md << "- grid4 P99.9: `" << gridErrs[idx999] << "`\n";
    md << "- grid4 PSNR: `" << computePsnr(gridRmse, psnrPeak) << "`\n";
    md << "- poly8 RMSE: `" << poly8Rmse << "`\n";
    md << "- poly8 P99.9: `" << poly8Errs[idx999] << "`\n";
    md << "- poly8 PSNR: `" << computePsnr(poly8Rmse, psnrPeak) << "`\n";
    md << "- poly11 RMSE: `" << poly11Rmse << "`\n";
    md << "- poly11 P99.9: `" << poly11Errs[idx999] << "`\n";
    md << "- poly11 PSNR: `" << computePsnr(poly11Rmse, psnrPeak) << "`\n";
    md << "- poly11-f16coeff RMSE: `" << poly11f16Rmse << "`\n";
    md << "- poly11-f16coeff P99.9: `" << poly11f16Errs[idx999] << "`\n";
    md << "- poly11-f16coeff PSNR: `" << computePsnr(poly11f16Rmse, psnrPeak) << "`\n";
    md << "- poly11-cp-inv RMSE: `" << poly11cpInvRmse << "`\n";
    md << "- poly11-cp-inv P99.9: `" << poly11cpInvErrs[idx999] << "`\n";
    md << "- poly11-cp-inv PSNR: `" << computePsnr(poly11cpInvRmse, psnrPeak) << "`\n";
    md.close();

    printf("\n=== Spatial Probe: poly11 sampled full-volume ===\n");
    printf("  Sample stride     : x=%d y=%d z=%d t=%d\n", stepX, stepY, stepZ, stepT);
    printf("  Samples           : %lld\n", samples);
    printf("  Baseline RMSE     : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           baseRmse, baselineErrs[idx999], computePsnr(baseRmse, psnrPeak));
    printf("  Grid4 RMSE        : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           gridRmse, gridErrs[idx999], computePsnr(gridRmse, psnrPeak));
    printf("  Poly8 RMSE        : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           poly8Rmse, poly8Errs[idx999], computePsnr(poly8Rmse, psnrPeak));
    printf("  Poly11 RMSE       : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           poly11Rmse, poly11Errs[idx999], computePsnr(poly11Rmse, psnrPeak));
    printf("  Poly11-f16 RMSE   : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           poly11f16Rmse, poly11f16Errs[idx999], computePsnr(poly11f16Rmse, psnrPeak));
    printf("  Poly11-cp-inv RMSE: %.6f  P99.9=%.6f  PSNR=%.2f\n",
           poly11cpInvRmse, poly11cpInvErrs[idx999], computePsnr(poly11cpInvRmse, psnrPeak));
    printf("  Report            : %s\n", reportPath.string().c_str());
    return 0;
}

static int runPoly11ResidualSampledFullProbe(const std::string& inputFile,
                                             const Volume4D& volumeSequence,
                                             const CompressedVolume4D& compressedVolume,
                                             int width, int height, int depth, int frames,
                                             double dataMin,
                                             double dataMax,
                                             bool useDataRangeForPsnr,
                                             const SpatialProbeOptions& spatialProbe) {
    namespace fs = std::filesystem;
    fs::create_directories("spatial_probe_reports");

    const int stepX = 4, stepY = 4, stepZ = 4, stepT = 4;
    const int leafNX = (width + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafNY = (height + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafNZ = (depth + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafCount = leafNX * leafNY * leafNZ;
    const auto ctrl = grid4ControlCoords();

    std::vector<double> leafSse(static_cast<size_t>(leafCount), 0.0);

    for (int bz = 0; bz < leafNZ; ++bz) {
        for (int by = 0; by < leafNY; ++by) {
            for (int bx = 0; bx < leafNX; ++bx) {
                const int leafId = (bz * leafNY + by) * leafNX + bx;
                for (int t = 0; t < frames; t += stepT) {
                    std::array<float, BT_LEAF_VOXELS> leafValues{};
                    int idx = 0;
                    for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; ++lx, ++idx) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx < width && vy < height && vz < depth) {
                                    leafValues[static_cast<size_t>(idx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                                } else {
                                    leafValues[static_cast<size_t>(idx)] = 0.0f;
                                }
                            }
                        }
                    }
                    const auto coeffs11 = fitPoly11Coeffs(leafValues);
                    for (int lz = 0; lz < BT_LEAF_SIZE; lz += stepZ) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ly += stepY) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; lx += stepX) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx >= width || vy >= height || vz >= depth) continue;
                                const float raw = volumeSequence[t][vz][vy][vx];
                                const float pred = evalPoly11At(coeffs11, lx, ly, lz);
                                const double e = static_cast<double>(pred - raw);
                                leafSse[static_cast<size_t>(leafId)] += e * e;
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<int> order(leafCount);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (leafSse[static_cast<size_t>(a)] != leafSse[static_cast<size_t>(b)])
            return leafSse[static_cast<size_t>(a)] > leafSse[static_cast<size_t>(b)];
        return a < b;
    });
    if (static_cast<int>(order.size()) > spatialProbe.topK) {
        order.resize(spatialProbe.topK);
    }
    std::vector<uint8_t> hotspot(static_cast<size_t>(leafCount), 0);
    for (int id : order) hotspot[static_cast<size_t>(id)] = 1;

    std::vector<float> baselineErrs;
    std::vector<float> gridErrs;
    std::vector<float> poly11Errs;
    std::vector<float> gridResidualErrs;
    std::vector<float> residualErrs;
    baselineErrs.reserve(static_cast<size_t>((width / stepX + 1) * (height / stepY + 1) * (depth / stepZ + 1) * (frames / stepT + 1)));
    gridErrs.reserve(baselineErrs.capacity());
    poly11Errs.reserve(baselineErrs.capacity());
    gridResidualErrs.reserve(baselineErrs.capacity());
    residualErrs.reserve(baselineErrs.capacity());
    double baseSum2 = 0.0, gridSum2 = 0.0, poly11Sum2 = 0.0, gridResidualSum2 = 0.0, residualSum2 = 0.0;
    long long samples = 0;
    long long residualSamples = 0;

    for (int bz = 0; bz < leafNZ; ++bz) {
        for (int by = 0; by < leafNY; ++by) {
            for (int bx = 0; bx < leafNX; ++bx) {
                const int leafId = (bz * leafNY + by) * leafNX + bx;
                const bool useResidual = hotspot[static_cast<size_t>(leafId)] != 0;
                for (int t = 0; t < frames; t += stepT) {
                    std::array<float, BT_LEAF_VOXELS> leafValues{};
                    std::array<float, 64> grid4Values{};
                    int idx = 0;
                    for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; ++lx, ++idx) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx < width && vy < height && vz < depth) {
                                    leafValues[static_cast<size_t>(idx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                                } else {
                                    leafValues[static_cast<size_t>(idx)] = 0.0f;
                                }
                            }
                        }
                    }
                    int cidx = 0;
                    for (int gz = 0; gz < 4; ++gz) {
                        for (int gy = 0; gy < 4; ++gy) {
                            for (int gx = 0; gx < 4; ++gx, ++cidx) {
                                const int vx = std::min(bx * BT_LEAF_SIZE + ctrl[gx], width - 1);
                                const int vy = std::min(by * BT_LEAF_SIZE + ctrl[gy], height - 1);
                                const int vz = std::min(bz * BT_LEAF_SIZE + ctrl[gz], depth - 1);
                                grid4Values[static_cast<size_t>(cidx)] = sampleTemporalKfs(compressedVolume[vz][vy][vx], t);
                            }
                        }
                    }
                    const auto coeffs11 = fitPoly11Coeffs(leafValues);

                    for (int lz = 0; lz < BT_LEAF_SIZE; lz += stepZ) {
                        for (int ly = 0; ly < BT_LEAF_SIZE; ly += stepY) {
                            for (int lx = 0; lx < BT_LEAF_SIZE; lx += stepX) {
                                const int vx = bx * BT_LEAF_SIZE + lx;
                                const int vy = by * BT_LEAF_SIZE + ly;
                                const int vz = bz * BT_LEAF_SIZE + lz;
                                if (vx >= width || vy >= height || vz >= depth) continue;

                                const float raw = volumeSequence[t][vz][vy][vx];
                                const int localIdx = (lz * BT_LEAF_SIZE + ly) * BT_LEAF_SIZE + lx;
                                const float basePred = leafValues[static_cast<size_t>(localIdx)];
                                const float gridPred = sampleGrid4TrilinearSingleFrame(grid4Values, lx, ly, lz);
                                const float polyPred = evalPoly11At(coeffs11, lx, ly, lz);
                                const float gridResidualPred = useResidual ? basePred : gridPred;
                                const float residualPred = useResidual ? basePred : polyPred;
                                if (useResidual) ++residualSamples;

                                const float eb = std::abs(basePred - raw);
                                const float eg = std::abs(gridPred - raw);
                                const float ep = std::abs(polyPred - raw);
                                const float egr = std::abs(gridResidualPred - raw);
                                const float er = std::abs(residualPred - raw);
                                baselineErrs.push_back(eb);
                                gridErrs.push_back(eg);
                                poly11Errs.push_back(ep);
                                gridResidualErrs.push_back(egr);
                                residualErrs.push_back(er);
                                baseSum2 += static_cast<double>(eb) * eb;
                                gridSum2 += static_cast<double>(eg) * eg;
                                poly11Sum2 += static_cast<double>(ep) * ep;
                                gridResidualSum2 += static_cast<double>(egr) * egr;
                                residualSum2 += static_cast<double>(er) * er;
                                ++samples;
                            }
                        }
                    }
                }
            }
        }
    }

    std::sort(baselineErrs.begin(), baselineErrs.end());
    std::sort(gridErrs.begin(), gridErrs.end());
    std::sort(poly11Errs.begin(), poly11Errs.end());
    std::sort(gridResidualErrs.begin(), gridResidualErrs.end());
    std::sort(residualErrs.begin(), residualErrs.end());
    const size_t idx999 = std::min(baselineErrs.size() - 1, static_cast<size_t>(0.999 * static_cast<double>(baselineErrs.size() - 1)));
    const double psnrPeak = useDataRangeForPsnr ? std::max(1e-6, dataMax - dataMin) : 255.0;

    const double baseRmse = std::sqrt(baseSum2 / std::max<long long>(1, samples));
    const double gridRmse = std::sqrt(gridSum2 / std::max<long long>(1, samples));
    const double poly11Rmse = std::sqrt(poly11Sum2 / std::max<long long>(1, samples));
    const double gridResidualRmse = std::sqrt(gridResidualSum2 / std::max<long long>(1, samples));
    const double residualRmse = std::sqrt(residualSum2 / std::max<long long>(1, samples));

    const fs::path reportPath = fs::path("spatial_probe_reports") / (fs::path(inputFile).stem().string() + "_poly11_residual_sampled_full_probe.md");
    std::ofstream md(reportPath);
    md << "# Poly11 + Residual Sampled Full-Volume Probe\n\n";
    md << "- residual type: `hotspot leaves fallback to temporal-only`\n";
    md << "- hotspot topK: `" << order.size() << "` (ranked by poly11 leaf SSE)\n";
    md << "- sampled stride: `x=" << stepX << ", y=" << stepY << ", z=" << stepZ << ", t=" << stepT << "`\n";
    md << "- samples: `" << samples << "`\n";
    md << "- residual samples: `" << residualSamples << "`\n";
    md << "- baseline RMSE: `" << baseRmse << "`\n";
    md << "- baseline P99.9: `" << baselineErrs[idx999] << "`\n";
    md << "- baseline PSNR: `" << computePsnr(baseRmse, psnrPeak) << "`\n";
    md << "- grid4 RMSE: `" << gridRmse << "`\n";
    md << "- grid4 P99.9: `" << gridErrs[idx999] << "`\n";
    md << "- grid4 PSNR: `" << computePsnr(gridRmse, psnrPeak) << "`\n";
    md << "- poly11 RMSE: `" << poly11Rmse << "`\n";
    md << "- poly11 P99.9: `" << poly11Errs[idx999] << "`\n";
    md << "- poly11 PSNR: `" << computePsnr(poly11Rmse, psnrPeak) << "`\n";
    md << "- grid4+residual RMSE: `" << gridResidualRmse << "`\n";
    md << "- grid4+residual P99.9: `" << gridResidualErrs[idx999] << "`\n";
    md << "- grid4+residual PSNR: `" << computePsnr(gridResidualRmse, psnrPeak) << "`\n";
    md << "- poly11+residual RMSE: `" << residualRmse << "`\n";
    md << "- poly11+residual P99.9: `" << residualErrs[idx999] << "`\n";
    md << "- poly11+residual PSNR: `" << computePsnr(residualRmse, psnrPeak) << "`\n";
    md.close();

    printf("\n=== Spatial Probe: poly11 + residual sampled full-volume ===\n");
    printf("  Hotspot topK      : %zu\n", order.size());
    printf("  Sample stride     : x=%d y=%d z=%d t=%d\n", stepX, stepY, stepZ, stepT);
    printf("  Samples           : %lld  residual-samples=%lld\n", samples, residualSamples);
    printf("  Baseline RMSE     : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           baseRmse, baselineErrs[idx999], computePsnr(baseRmse, psnrPeak));
    printf("  Grid4 RMSE        : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           gridRmse, gridErrs[idx999], computePsnr(gridRmse, psnrPeak));
    printf("  Poly11 RMSE       : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           poly11Rmse, poly11Errs[idx999], computePsnr(poly11Rmse, psnrPeak));
    printf("  Grid4+Residual    : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           gridResidualRmse, gridResidualErrs[idx999], computePsnr(gridResidualRmse, psnrPeak));
    printf("  Poly11+Residual   : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           residualRmse, residualErrs[idx999], computePsnr(residualRmse, psnrPeak));
    printf("  Report            : %s\n", reportPath.string().c_str());
    return 0;
}

static int runGrid4HybridSampledFullProbe(const std::string& inputFile,
                                          const Volume4D& volumeSequence,
                                          const CompressedVolume4D& compressedVolume,
                                          int width, int height, int depth, int frames,
                                          const FieldProfile& fieldProfile,
                                          bool enableBackgroundElision,
                                          bool enableBlockAwareCluster,
                                          bool enableBudgetAwareCluster,
                                          bool enableGuardedMedoidCluster,
                                          bool enableValidateFallback,
                                          bool enableHotspotSecondPass,
                                          double clusterThr,
                                          double dataMin,
                                          double dataMax,
                                          bool useDataRangeForPsnr,
                                          const SpatialProbeOptions& spatialProbe) {
    namespace fs = std::filesystem;
    fs::create_directories("spatial_probe_reports");

    std::string csvPath = spatialProbe.leafCsv;
    if (csvPath.empty()) {
        csvPath = (fs::path("leaf_error_reports") / (fs::path(inputFile).stem().string() + "_leaf_error_report.csv")).string();
    }
    const auto hotspotRows = loadHotspotLeafRows(csvPath, spatialProbe.topK);
    std::unordered_map<int, bool> hotspotLeafs;
    hotspotLeafs.reserve(hotspotRows.size() * 2);
    for (const auto& r : hotspotRows) hotspotLeafs[r.leafId] = true;

    BlockTree btree;
    btree.build(compressedVolume, width, height, depth, frames,
                -1.0, clusterThr, fieldProfile, enableBackgroundElision,
                enableBlockAwareCluster, enableBudgetAwareCluster, enableGuardedMedoidCluster, false, false,
                &volumeSequence, enableValidateFallback, enableHotspotSecondPass);
    btree.flattenLeaves();

    const int stepX = 4, stepY = 4, stepZ = 4, stepT = 4;
    std::vector<float> baselineErrs;
    std::vector<float> hybridErrs;
    baselineErrs.reserve(static_cast<size_t>((width / stepX + 1) * (height / stepY + 1) * (depth / stepZ + 1) * (frames / stepT + 1)));
    hybridErrs.reserve(baselineErrs.capacity());
    double baseSum = 0.0, baseSum2 = 0.0;
    double hybridSum = 0.0, hybridSum2 = 0.0;
    float baseMax = 0.0f, hybridMax = 0.0f;
    long long samples = 0;

    for (int z = 0; z < depth; z += stepZ) {
        for (int y = 0; y < height; y += stepY) {
            for (int x = 0; x < width; x += stepX) {
                const int bx = x / BT_LEAF_SIZE;
                const int by = y / BT_LEAF_SIZE;
                const int bz = z / BT_LEAF_SIZE;
                const int lx = x % BT_LEAF_SIZE;
                const int ly = y % BT_LEAF_SIZE;
                const int lz = z % BT_LEAF_SIZE;
                int leafId = -1;
                const bool hasLeaf = resolveLeafIdAtOrigin(btree, bx * BT_LEAF_SIZE, by * BT_LEAF_SIZE, bz * BT_LEAF_SIZE, leafId);
                const bool useGrid4 = hasLeaf && hotspotLeafs.find(leafId) != hotspotLeafs.end();
                for (int t = 0; t < frames; t += stepT) {
                    const float raw = volumeSequence[t][z][y][x];
                    const float basePred = btree.sampleFlat(x, y, z, static_cast<float>(t));
                    const float hybridPred = useGrid4
                        ? sampleGrid4FromCompVolume(compressedVolume, width, height, depth, bx, by, bz, lx, ly, lz, t)
                        : basePred;
                    const float eb = std::abs(basePred - raw);
                    const float eh = std::abs(hybridPred - raw);
                    baselineErrs.push_back(eb);
                    hybridErrs.push_back(eh);
                    baseSum += eb;
                    baseSum2 += static_cast<double>(eb) * eb;
                    hybridSum += eh;
                    hybridSum2 += static_cast<double>(eh) * eh;
                    baseMax = std::max(baseMax, eb);
                    hybridMax = std::max(hybridMax, eh);
                    ++samples;
                }
            }
        }
    }

    std::sort(baselineErrs.begin(), baselineErrs.end());
    std::sort(hybridErrs.begin(), hybridErrs.end());
    const size_t idx99 = std::min(baselineErrs.size() - 1, static_cast<size_t>(0.99 * static_cast<double>(baselineErrs.size() - 1)));
    const size_t idx999 = std::min(baselineErrs.size() - 1, static_cast<size_t>(0.999 * static_cast<double>(baselineErrs.size() - 1)));
    const double psnrPeak = useDataRangeForPsnr ? std::max(1e-6, dataMax - dataMin) : 255.0;

    const double baseRmse = std::sqrt(baseSum2 / std::max<long long>(1, samples));
    const double hybridRmse = std::sqrt(hybridSum2 / std::max<long long>(1, samples));

    const fs::path reportPath = fs::path("spatial_probe_reports") / (fs::path(inputFile).stem().string() + "_grid4_hybrid_sampled_full_probe.md");
    std::ofstream md(reportPath);
    md << "# Grid4 Hybrid Sampled Full-Volume Probe\n\n";
    md << "- hotspot csv: `" << csvPath << "`\n";
    md << "- hotspot topK: `" << hotspotRows.size() << "`\n";
    md << "- sampled stride: `x=" << stepX << ", y=" << stepY << ", z=" << stepZ << ", t=" << stepT << "`\n";
    md << "- samples: `" << samples << "`\n";
    md << "- baseline RMSE: `" << baseRmse << "`\n";
    md << "- baseline P99.9: `" << baselineErrs[idx999] << "`\n";
    md << "- baseline PSNR: `" << computePsnr(baseRmse, psnrPeak) << "`\n";
    md << "- hybrid RMSE: `" << hybridRmse << "`\n";
    md << "- hybrid P99.9: `" << hybridErrs[idx999] << "`\n";
    md << "- hybrid PSNR: `" << computePsnr(hybridRmse, psnrPeak) << "`\n";
    md.close();

    printf("\n=== Spatial Probe: grid4 hybrid sampled full-volume ===\n");
    printf("  Hotspot CSV       : %s\n", csvPath.c_str());
    printf("  Hotspot topK      : %zu\n", hotspotRows.size());
    printf("  Sample stride     : x=%d y=%d z=%d t=%d\n", stepX, stepY, stepZ, stepT);
    printf("  Samples           : %lld\n", samples);
    printf("  Baseline RMSE     : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           baseRmse, baselineErrs[idx999], computePsnr(baseRmse, psnrPeak));
    printf("  Hybrid RMSE       : %.6f  P99.9=%.6f  PSNR=%.2f\n",
           hybridRmse, hybridErrs[idx999], computePsnr(hybridRmse, psnrPeak));
    printf("  Report            : %s\n", reportPath.string().c_str());
    return 0;
}

static int runGrid4ResidualSampledFullProbe(const std::string& inputFile,
                                            const Volume4D& volumeSequence,
                                            const CompressedVolume4D& compressedVolume,
                                            int width, int height, int depth, int frames,
                                            const FieldProfile& fieldProfile,
                                            bool enableBackgroundElision,
                                            bool enableBlockAwareCluster,
                                            bool enableBudgetAwareCluster,
                                            bool enableGuardedMedoidCluster,
                                            bool enableValidateFallback,
                                            bool enableHotspotSecondPass,
                                            double clusterThr,
                                            double dataMin,
                                            double dataMax,
                                            bool useDataRangeForPsnr,
                                            const SpatialProbeOptions& spatialProbe) {
    namespace fs = std::filesystem;
    fs::create_directories("spatial_probe_reports");
    try {
        printf("\n[grid4-residual] build GRID4 base tree...\n");
        BlockTree grid4Tree;
        grid4Tree.build(compressedVolume, width, height, depth, frames,
                        -1.0, clusterThr, fieldProfile, enableBackgroundElision,
                        enableBlockAwareCluster, enableBudgetAwareCluster, enableGuardedMedoidCluster, true, false,
                        &volumeSequence, enableValidateFallback, enableHotspotSecondPass);
        printf("[grid4-residual] flatten GRID4 base tree...\n");
        grid4Tree.flattenLeaves();

        printf("[grid4-residual] analyze GRID4 leaf errors...\n");
        std::vector<LeafErrorStats> leafStats = analyzeLeafErrors(
            volumeSequence, compressedVolume, grid4Tree, width, height, depth, frames);
        std::sort(leafStats.begin(), leafStats.end(),
                  [](const LeafErrorStats& a, const LeafErrorStats& b) {
                      if (a.sse != b.sse) return a.sse > b.sse;
                      return a.rmse > b.rmse;
                  });
        if (static_cast<int>(leafStats.size()) > spatialProbe.topK) {
            leafStats.resize(spatialProbe.topK);
        }

        printf("[grid4-residual] select hotspot leaves: %zu\n", leafStats.size());
        std::unordered_map<int, bool> hotspotLeafs;
        hotspotLeafs.reserve(leafStats.size() * 2);
        for (const auto& s : leafStats) hotspotLeafs[s.leafId] = true;

        const int stepX = 4, stepY = 4, stepZ = 4, stepT = 4;
        std::vector<float> baseErrs;
        std::vector<float> residualErrs;
        baseErrs.reserve(static_cast<size_t>((width / stepX + 1) * (height / stepY + 1) * (depth / stepZ + 1) * (frames / stepT + 1)));
        residualErrs.reserve(baseErrs.capacity());
        double baseSum2 = 0.0;
        double residualSum2 = 0.0;
        double baseSum = 0.0;
        double residualSum = 0.0;
        float baseMax = 0.0f;
        float residualMax = 0.0f;
        long long samples = 0;
        long long residualSamples = 0;

        printf("[grid4-residual] evaluate sampled full volume...\n");
        for (int z = 0; z < depth; z += stepZ) {
            for (int y = 0; y < height; y += stepY) {
                for (int x = 0; x < width; x += stepX) {
                    int leafId = -1;
                    const bool hasLeaf = resolveLeafIdAtOrigin(grid4Tree,
                        (x / BT_LEAF_SIZE) * BT_LEAF_SIZE,
                        (y / BT_LEAF_SIZE) * BT_LEAF_SIZE,
                        (z / BT_LEAF_SIZE) * BT_LEAF_SIZE,
                        leafId);
                    const bool useResidual = hasLeaf && hotspotLeafs.find(leafId) != hotspotLeafs.end();

                    for (int t = 0; t < frames; t += stepT) {
                        const float raw = volumeSequence[t][z][y][x];
                        const float basePred = grid4Tree.sampleFlat(x, y, z, static_cast<float>(t));
                        float pred = basePred;
                        if (useResidual) {
                            if (z < 0 || z >= static_cast<int>(compressedVolume.size()) ||
                                y < 0 || y >= static_cast<int>(compressedVolume[z].size()) ||
                                x < 0 || x >= static_cast<int>(compressedVolume[z][y].size())) {
                                fprintf(stderr, "[grid4-residual] compressedVolume OOB at x=%d y=%d z=%d\n", x, y, z);
                                return 1;
                            }
                            const auto& temporalSeq = compressedVolume[z][y][x];
                            if (temporalSeq.empty()) {
                                fprintf(stderr, "[grid4-residual] empty temporal seq at x=%d y=%d z=%d\n", x, y, z);
                                return 1;
                            }
                            pred = sampleTemporalKfs(temporalSeq, t);
                            ++residualSamples;
                        }
                        const float eb = std::abs(basePred - raw);
                        const float er = std::abs(pred - raw);
                        baseErrs.push_back(eb);
                        residualErrs.push_back(er);
                        baseSum += eb;
                        residualSum += er;
                        baseSum2 += static_cast<double>(eb) * eb;
                        residualSum2 += static_cast<double>(er) * er;
                        baseMax = std::max(baseMax, eb);
                        residualMax = std::max(residualMax, er);
                        ++samples;
                    }
                }
            }
        }

        std::sort(baseErrs.begin(), baseErrs.end());
        std::sort(residualErrs.begin(), residualErrs.end());
        const size_t idx99 = std::min(baseErrs.size() - 1, static_cast<size_t>(0.99 * static_cast<double>(baseErrs.size() - 1)));
        const size_t idx999 = std::min(baseErrs.size() - 1, static_cast<size_t>(0.999 * static_cast<double>(baseErrs.size() - 1)));
        const double psnrPeak = useDataRangeForPsnr ? std::max(1e-6, dataMax - dataMin) : 255.0;
        const double baseRmse = std::sqrt(baseSum2 / std::max<long long>(1, samples));
        const double residualRmse = std::sqrt(residualSum2 / std::max<long long>(1, samples));

        const fs::path reportPath = fs::path("spatial_probe_reports") / (fs::path(inputFile).stem().string() + "_grid4_residual_sampled_full_probe.md");
        std::ofstream md(reportPath);
        md << "# Grid4 + Residual Sampled Full-Volume Probe\n\n";
        md << "- residual type: `hotspot leaves fallback to temporal-only`\n";
        md << "- hotspot topK: `" << leafStats.size() << "` (ranked by SSE in GRID4 base)\n";
        md << "- sampled stride: `x=" << stepX << ", y=" << stepY << ", z=" << stepZ << ", t=" << stepT << "`\n";
        md << "- samples: `" << samples << "`\n";
        md << "- residual samples: `" << residualSamples << "`\n";
        md << "- base RMSE: `" << baseRmse << "`\n";
        md << "- base P99.9: `" << baseErrs[idx999] << "`\n";
        md << "- base PSNR: `" << computePsnr(baseRmse, psnrPeak) << "`\n";
        md << "- base Mean/Max: `" << (baseSum / std::max<long long>(1, samples)) << " / " << baseMax << "`\n";
        md << "- residual RMSE: `" << residualRmse << "`\n";
        md << "- residual P99.9: `" << residualErrs[idx999] << "`\n";
        md << "- residual PSNR: `" << computePsnr(residualRmse, psnrPeak) << "`\n";
        md << "- residual Mean/Max: `" << (residualSum / std::max<long long>(1, samples)) << " / " << residualMax << "`\n";
        md.close();

        printf("\n=== Spatial Probe: grid4 + residual sampled full-volume ===\n");
        printf("  Hotspot topK      : %zu\n", leafStats.size());
        printf("  Sample stride     : x=%d y=%d z=%d t=%d\n", stepX, stepY, stepZ, stepT);
        printf("  Samples           : %lld  residual-samples=%lld\n", samples, residualSamples);
        printf("  Base RMSE         : %.6f  P99.9=%.6f  PSNR=%.2f\n",
               baseRmse, baseErrs[idx999], computePsnr(baseRmse, psnrPeak));
        printf("  Residual RMSE     : %.6f  P99.9=%.6f  PSNR=%.2f\n",
               residualRmse, residualErrs[idx999], computePsnr(residualRmse, psnrPeak));
        printf("  Report            : %s\n", reportPath.string().c_str());
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[grid4-residual] exception: %s\n", e.what());
        return 1;
    } catch (...) {
        fprintf(stderr, "[grid4-residual] unknown exception\n");
        return 1;
    }
}

static int runGrid4ResidualProfiler(const std::string& inputFile,
                                    const Volume4D& volumeSequence,
                                    const CompressedVolume4D& compressedVolume,
                                    int width, int height, int depth, int frames,
                                    const FieldProfile& fieldProfile,
                                    bool enableBackgroundElision,
                                    bool enableBlockAwareCluster,
                                    bool enableBudgetAwareCluster,
                                    bool enableGuardedMedoidCluster,
                                    bool enableValidateFallback,
                                    bool enableHotspotSecondPass,
                                    double clusterThr,
                                    double dataMin,
                                    double dataMax,
                                    const SpatialProbeOptions& spatialProbe) {
    namespace fs = std::filesystem;
    fs::create_directories("spatial_probe_reports");

    printf("\n[grid4-res-prof] build GRID4 base tree...\n");
    BlockTree grid4Tree;
    grid4Tree.build(compressedVolume, width, height, depth, frames,
                    -1.0, clusterThr, fieldProfile, enableBackgroundElision,
                    enableBlockAwareCluster, enableBudgetAwareCluster, enableGuardedMedoidCluster, true, false,
                    &volumeSequence, enableValidateFallback, enableHotspotSecondPass);

    printf("[grid4-res-prof] analyze GRID4 leaf errors...\n");
    std::vector<LeafErrorStats> leafStats = analyzeLeafErrors(
        volumeSequence, compressedVolume, grid4Tree, width, height, depth, frames);
    std::sort(leafStats.begin(), leafStats.end(),
              [](const LeafErrorStats& a, const LeafErrorStats& b) {
                  if (a.sse != b.sse) return a.sse > b.sse;
                  return a.rmse > b.rmse;
              });
    if (static_cast<int>(leafStats.size()) > spatialProbe.topK) {
        leafStats.resize(spatialProbe.topK);
    }

    const double fullRange = std::max(1e-6, dataMax - dataMin);
    const double residualThr = (spatialProbe.threshold > 0.0)
        ? spatialProbe.threshold
        : std::max(0.005, 0.03 * fullRange);

    struct ResidualLeafProfile {
        LeafErrorStats base;
        int activeVoxels = 0;
        double sparsity = 0.0;
        double maxAbsResidual = 0.0;
        double int8Step = 0.0;
        double quantRmse = 0.0;
        double quantMax = 0.0;
    };
    std::vector<ResidualLeafProfile> results;
    results.reserve(leafStats.size());

    long long totalActive = 0;
    double sumSparsity = 0.0;
    double sumInt8Step = 0.0;
    double sumQuantRmse = 0.0;
    double sumMaxAbsResidual = 0.0;
    double globalQuantSse = 0.0;
    double globalResidualSse = 0.0;
    long long globalResidualSamples = 0;
    std::array<int, 10> hist{};

    for (const auto& leafStat : leafStats) {
        const LeafBlock& leaf = grid4Tree.leafBlocks[leafStat.leafId];
        ResidualLeafProfile out;
        out.base = leafStat;
        std::vector<uint8_t> active(BT_LEAF_VOXELS, 0);
        std::vector<double> voxelMaxResidual(BT_LEAF_VOXELS, 0.0);

        const int ox = leafStat.bx * BT_LEAF_SIZE;
        const int oy = leafStat.by * BT_LEAF_SIZE;
        const int oz = leafStat.bz * BT_LEAF_SIZE;

        for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
            for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                    const int gx = ox + lx;
                    const int gy = oy + ly;
                    const int gz = oz + lz;
                    if (gx >= width || gy >= height || gz >= depth) continue;
                    const int localIdx = lx + BT_LEAF_SIZE * (ly + BT_LEAF_SIZE * lz);
                    double voxelMax = 0.0;
                    for (int t = 0; t < frames; ++t) {
                        const float raw = volumeSequence[t][gz][gy][gx];
                        const float base = sampleLeafSequenceLocal(leaf, localIdx, static_cast<float>(t));
                        const double r = std::abs(static_cast<double>(raw) - base);
                        voxelMax = std::max(voxelMax, r);
                    }
                    voxelMaxResidual[localIdx] = voxelMax;
                    if (voxelMax > residualThr) {
                        active[localIdx] = 1;
                        out.activeVoxels++;
                        out.maxAbsResidual = std::max(out.maxAbsResidual, voxelMax);
                    }
                }
            }
        }

        out.sparsity = 1.0 - (static_cast<double>(out.activeVoxels) / BT_LEAF_VOXELS);
        out.int8Step = (out.activeVoxels > 0) ? (out.maxAbsResidual / 127.0) : 0.0;

        if (out.activeVoxels > 0) {
            for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
                for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                    for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                        const int gx = ox + lx;
                        const int gy = oy + ly;
                        const int gz = oz + lz;
                        if (gx >= width || gy >= height || gz >= depth) continue;
                        const int localIdx = lx + BT_LEAF_SIZE * (ly + BT_LEAF_SIZE * lz);
                        if (!active[localIdx]) continue;
                        for (int t = 0; t < frames; ++t) {
                            const float raw = volumeSequence[t][gz][gy][gx];
                            const float base = sampleLeafSequenceLocal(leaf, localIdx, static_cast<float>(t));
                            const double residual = static_cast<double>(raw) - base;
                            const double q = std::round(residual / out.int8Step);
                            const double qClamped = std::clamp(q, -127.0, 127.0);
                            const double restored = qClamped * out.int8Step;
                            const double qe = restored - residual;
                            out.quantRmse += qe * qe;
                            out.quantMax = std::max(out.quantMax, std::abs(qe));
                            globalQuantSse += qe * qe;
                            globalResidualSse += residual * residual;
                            globalResidualSamples++;
                        }
                    }
                }
            }
            out.quantRmse = std::sqrt(out.quantRmse / std::max(1, out.activeVoxels * frames));
        }

        totalActive += out.activeVoxels;
        sumSparsity += out.sparsity;
        sumInt8Step += out.int8Step;
        sumQuantRmse += out.quantRmse;
        sumMaxAbsResidual += out.maxAbsResidual;
        if (out.activeVoxels > 0) {
            const double scaled = out.int8Step * 10000.0;
            const int bin = std::min<int>(9, std::max<int>(0, static_cast<int>(scaled)));
            hist[bin]++;
        }

        results.push_back(out);
    }

    const double avgSparsity = results.empty() ? 0.0 : (sumSparsity / results.size());
    const double avgActive = results.empty() ? 0.0 : (static_cast<double>(totalActive) / results.size());
    const double avgStep = results.empty() ? 0.0 : (sumInt8Step / results.size());
    const double avgQuantRmse = results.empty() ? 0.0 : (sumQuantRmse / results.size());
    const double avgMaxResidual = results.empty() ? 0.0 : (sumMaxAbsResidual / results.size());
    const double globalResidualRmse = std::sqrt(globalResidualSse / std::max<long long>(1, globalResidualSamples));
    const double globalQuantRmse = std::sqrt(globalQuantSse / std::max<long long>(1, globalResidualSamples));
    const double quantOverResidual = globalResidualRmse > 1e-12 ? (globalQuantRmse / globalResidualRmse) : 0.0;

    const fs::path reportPath = fs::path("spatial_probe_reports") / (fs::path(inputFile).stem().string() + "_grid4_residual_profile.md");
    std::ofstream md(reportPath);
    md << "# GRID4 Residual Profiler\n\n";
    md << "- hotspot topK: `" << results.size() << "`\n";
    md << "- residual threshold: `" << residualThr << "`\n";
    md << "- avg active voxels / leaf: `" << avgActive << "` / 512\n";
    md << "- avg sparsity: `" << avgSparsity * 100.0 << "%`\n";
    md << "- avg max abs residual: `" << avgMaxResidual << "`\n";
    md << "- avg int8 step: `" << avgStep << "`\n";
    md << "- avg per-leaf quant RMSE: `" << avgQuantRmse << "`\n";
    md << "- global residual RMSE: `" << globalResidualRmse << "`\n";
    md << "- global quant RMSE: `" << globalQuantRmse << "`\n";
    md << "- quant/residual RMSE ratio: `" << quantOverResidual << "`\n";
    md << "\n## Step Histogram (`int8_step * 10000` bins)\n\n";
    for (int i = 0; i < 10; ++i) {
        md << "- bin[" << i << "]: `" << hist[i] << "`\n";
    }
    md << "\n## Top Leaves\n\n";
    md << "| leaf | block | base rmse | active voxels | sparsity | max residual | int8 step | quant rmse | quant max |\n";
    md << "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
    for (size_t i = 0; i < std::min<size_t>(results.size(), 32); ++i) {
        const auto& r = results[i];
        md << "| " << r.base.leafId
           << " | (" << r.base.bx << "," << r.base.by << "," << r.base.bz << ")"
           << " | " << r.base.rmse
           << " | " << r.activeVoxels
           << " | " << r.sparsity
           << " | " << r.maxAbsResidual
           << " | " << r.int8Step
           << " | " << r.quantRmse
           << " | " << r.quantMax
           << " |\n";
    }
    md.close();

    printf("\n=== Spatial Probe: GRID4 residual profiler ===\n");
    printf("  Hotspot topK      : %zu\n", results.size());
    printf("  Residual thr      : %.6f\n", residualThr);
    printf("  Avg active voxels : %.2f / 512\n", avgActive);
    printf("  Avg sparsity      : %.2f%%\n", avgSparsity * 100.0);
    printf("  Avg max residual  : %.6f\n", avgMaxResidual);
    printf("  Avg int8 step     : %.8f\n", avgStep);
    printf("  Global residual RMSE : %.6f\n", globalResidualRmse);
    printf("  Global quant RMSE    : %.6f\n", globalQuantRmse);
    printf("  Quant/Residual ratio : %.4f\n", quantOverResidual);
    printf("  Report               : %s\n", reportPath.string().c_str());
    return 0;
}

static FieldProfile makeFieldProfile(const ProfileOptions& opt) {
    if (opt.name == "DENSITY") {
        DensityProfileParams p{opt.epsAbs, opt.epsRel, opt.baseEps, opt.gammaDelta,
                               opt.renderCutoff, opt.cutoffBand,
                               opt.cutoffTemporalProtect, opt.cutoffClusterProtect,
                               opt.bgZeroRatio, opt.bgConstRatio};
        return FieldProfile::makeDensity(p);
    }
    if (opt.name == "GENERIC") {
        DensityProfileParams p{opt.epsAbs, opt.epsRel, opt.baseEps, opt.gammaDelta,
                               opt.renderCutoff, opt.cutoffBand,
                               opt.cutoffTemporalProtect, opt.cutoffClusterProtect,
                               opt.bgZeroRatio, opt.bgConstRatio};
        return FieldProfile::makeGeneric(p);
    }
    SdfProfileParams p{opt.iso, opt.wband, opt.nearBand, opt.criticalBand,
                       opt.epsFar, opt.epsNear, opt.epsCritical};
    return FieldProfile::makeSDF(p);
}

static int resolveStart(int total, int sub, int requested) {
    if (sub >= total) return 0;
    if (requested >= 0) {
        if (requested + sub > total) return std::max(0, total - sub);
        return requested;
    }
    return std::max(0, (total - sub) / 2);
}

static Volume4D extractSubvolume(
    const Volume4D& src,
    int startX, int startY, int startZ, int startT,
    int subW, int subH, int subD, int subF,
    float& outMin, float& outMax)
{
    Volume4D out(subF,
                 std::vector<std::vector<std::vector<float>>>(
                     subD,
                     std::vector<std::vector<float>>(
                         subH,
                         std::vector<float>(subW))));

    outMin = std::numeric_limits<float>::max();
    outMax = std::numeric_limits<float>::lowest();

    for (int t = 0; t < subF; ++t) {
        for (int z = 0; z < subD; ++z) {
            for (int y = 0; y < subH; ++y) {
                for (int x = 0; x < subW; ++x) {
                    float v = src[startT + t][startZ + z][startY + y][startX + x];
                    out[t][z][y][x] = v;
                    outMin = std::min(outMin, v);
                    outMax = std::max(outMax, v);
                }
            }
        }
    }
    return out;
}

static bool sameBlock(const SubvolumeSpec& a, const SubvolumeSpec& b) {
    return a.startX == b.startX && a.startY == b.startY &&
           a.startZ == b.startZ && a.startT == b.startT;
}

static BlockScore scoreBlock(
    const Volume4D& src,
    int startX, int startY, int startZ, int startT,
    int subW, int subH, int subD, int subF)
{
    const int stepX = std::max(1, subW / 4);
    const int stepY = std::max(1, subH / 4);
    const int stepZ = std::max(1, subD / 4);
    const int stepT = std::max(1, subF / 4);

    double sum = 0.0;
    double sum2 = 0.0;
    double absSum = 0.0;
    double deltaSum = 0.0;
    int count = 0;
    int deltaCount = 0;
    float vmin = std::numeric_limits<float>::max();
    float vmax = std::numeric_limits<float>::lowest();

    for (int t = 0; t < subF; t += stepT) {
        for (int z = 0; z < subD; z += stepZ) {
            for (int y = 0; y < subH; y += stepY) {
                for (int x = 0; x < subW; x += stepX) {
                    const float v = src[startT + t][startZ + z][startY + y][startX + x];
                    vmin = std::min(vmin, v);
                    vmax = std::max(vmax, v);
                    sum += v;
                    sum2 += static_cast<double>(v) * v;
                    absSum += std::abs(v);
                    ++count;
                    if (t + stepT < subF) {
                        const float vn = src[startT + t + stepT][startZ + z][startY + y][startX + x];
                        deltaSum += std::abs(vn - v);
                        ++deltaCount;
                    }
                }
            }
        }
    }

    const double mean = count > 0 ? sum / count : 0.0;
    const double var = count > 0 ? std::max(0.0, sum2 / count - mean * mean) : 0.0;
    BlockScore s;
    s.range = vmax - vmin;
    s.deltaMean = deltaCount > 0 ? deltaSum / deltaCount : 0.0;
    s.stddev = std::sqrt(var);
    s.absMean = count > 0 ? absSum / count : 0.0;
    return s;
}

static std::vector<SubvolumeSpec> selectCalibrationBlocks(
    const Volume4D& src,
    int width, int height, int depth, int frames,
    const CalibrationOptions& calib)
{
    const int subW = std::min(calib.subW, width);
    const int subH = std::min(calib.subH, height);
    const int subD = std::min(calib.subD, depth);
    const int subF = std::min(calib.subF, frames);

    std::vector<BlockScore> scores;
    const int stepX = std::max(1, subW);
    const int stepY = std::max(1, subH);
    const int stepZ = std::max(1, subD);
    const int stepT = std::max(1, subF / 2);

    for (int t = 0; t <= std::max(0, frames - subF); t += stepT) {
        for (int z = 0; z <= std::max(0, depth - subD); z += stepZ) {
            for (int y = 0; y <= std::max(0, height - subH); y += stepY) {
                for (int x = 0; x <= std::max(0, width - subW); x += stepX) {
                    BlockScore s = scoreBlock(src, x, y, z, t, subW, subH, subD, subF);
                    s.spec = {x, y, z, t, ""};
                    scores.push_back(s);
                }
            }
        }
    }

    std::vector<SubvolumeSpec> out;
    auto pushUnique = [&](SubvolumeSpec s) {
        for (const auto& existing : out) {
            if (sameBlock(existing, s)) return;
        }
        out.push_back(s);
    };

    pushUnique({resolveStart(width, subW, calib.startX),
                resolveStart(height, subH, calib.startY),
                resolveStart(depth, subD, calib.startZ),
                resolveStart(frames, subF, calib.startT),
                "center"});

    auto bestBy = [&](auto proj, const std::string& tag) {
        if (scores.empty()) return;
        const auto it = std::max_element(scores.begin(), scores.end(),
            [&](const BlockScore& a, const BlockScore& b) { return proj(a) < proj(b); });
        if (it != scores.end()) {
            SubvolumeSpec s = it->spec;
            s.tag = tag;
            pushUnique(s);
        }
    };

    bestBy([](const BlockScore& s) { return s.range; }, "max_range");
    bestBy([](const BlockScore& s) { return s.deltaMean; }, "max_delta");
    bestBy([](const BlockScore& s) { return s.stddev; }, "max_stddev");
    bestBy([](const BlockScore& s) { return s.absMean; }, "max_abs_mean");

    if (static_cast<int>(out.size()) > calib.numBlocks) {
        out.resize(calib.numBlocks);
    }
    return out;
}

static EvalMetrics aggregateMetrics(
    const std::vector<EvalMetrics>& perBlock,
    double psnrPeak)
{
    EvalMetrics out;
    if (perBlock.empty()) return out;

    double mseMean = 0.0;
    for (const auto& m : perBlock) {
        out.vbtBytes += m.vbtBytes;
        out.leaves += m.leaves;
        out.totalKF += m.totalKF;
        out.mean += m.mean;
        out.max = std::max(out.max, m.max);
        out.p99 = std::max(out.p99, m.p99);
        out.p999 = std::max(out.p999, m.p999);
        mseMean += m.rmse * m.rmse;
    }

    const double invN = 1.0 / perBlock.size();
    out.vbtBytes = static_cast<size_t>(out.vbtBytes * invN);
    out.leaves = static_cast<int>(out.leaves * invN);
    out.totalKF = static_cast<long long>(out.totalKF * invN);
    out.mean *= invN;
    out.rmse = std::sqrt(mseMean * invN);
    out.psnr = computePsnr(out.rmse, psnrPeak);
    return out;
}

static EvalMetrics evaluateBuiltTreeSampled(
    BlockTree& btree,
    const Volume4D& volumeSequence,
    int width, int height, int depth, int frames,
    double dataMin,
    double dataMax,
    bool useDataRangeForPsnr,
    int stepX,
    int stepY,
    int stepZ,
    int stepT)
{
    EvalMetrics metrics;
    metrics.vbtBytes = computeFlatVbtBytes(btree);
    metrics.leaves = static_cast<int>(btree.leafHeaders.size());
    metrics.totalKF = btree.getStats().totalKF;

    std::vector<float> errs;
    double errSum = 0.0;
    double errSum2 = 0.0;
    float errMax = 0.0f;
    long long sampleCount = 0;

#ifdef _OPENMP
    const int sweepThreads = std::max(1, omp_get_max_threads());
    std::vector<std::vector<float>> errsByThread(static_cast<size_t>(sweepThreads));
    std::vector<double> errSumsByThread(static_cast<size_t>(sweepThreads), 0.0);
    std::vector<double> errSum2ByThread(static_cast<size_t>(sweepThreads), 0.0);
    std::vector<float> errMaxByThread(static_cast<size_t>(sweepThreads), 0.0f);
    std::vector<long long> sampleCountByThread(static_cast<size_t>(sweepThreads), 0);
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        std::vector<float>& localErrs = errsByThread[static_cast<size_t>(tid)];
        double& localErrSum = errSumsByThread[static_cast<size_t>(tid)];
        double& localErrSum2 = errSum2ByThread[static_cast<size_t>(tid)];
        float& localErrMax = errMaxByThread[static_cast<size_t>(tid)];
        long long& localCount = sampleCountByThread[static_cast<size_t>(tid)];

        #pragma omp for collapse(3) nowait
        for (int z = 0; z < depth; z += stepZ) {
            for (int y = 0; y < height; y += stepY) {
                for (int x = 0; x < width; x += stepX) {
                    for (int t = 0; t < frames; t += stepT) {
                        const float got = btree.sampleFlat(x, y, z, static_cast<float>(t));
                        const float orig = volumeSequence[t][z][y][x];
                        const float e = std::abs(got - orig);
                        localErrs.push_back(e);
                        localErrSum += e;
                        localErrSum2 += static_cast<double>(e) * e;
                        localErrMax = std::max(localErrMax, e);
                        ++localCount;
                    }
                }
            }
        }
    }
    for (int tid = 0; tid < sweepThreads; ++tid) {
        std::vector<float>& localErrs = errsByThread[static_cast<size_t>(tid)];
        errs.insert(errs.end(), localErrs.begin(), localErrs.end());
        errSum += errSumsByThread[static_cast<size_t>(tid)];
        errSum2 += errSum2ByThread[static_cast<size_t>(tid)];
        errMax = std::max(errMax, errMaxByThread[static_cast<size_t>(tid)]);
        sampleCount += sampleCountByThread[static_cast<size_t>(tid)];
    }
#else
    for (int z = 0; z < depth; z += stepZ) {
        for (int y = 0; y < height; y += stepY) {
            for (int x = 0; x < width; x += stepX) {
                for (int t = 0; t < frames; t += stepT) {
                    const float got = btree.sampleFlat(x, y, z, static_cast<float>(t));
                    const float orig = volumeSequence[t][z][y][x];
                    const float e = std::abs(got - orig);
                    errs.push_back(e);
                    errSum += e;
                    errSum2 += static_cast<double>(e) * e;
                    errMax = std::max(errMax, e);
                    ++sampleCount;
                }
            }
        }
    }
#endif

    if (errs.empty()) {
        metrics.valid = false;
        metrics.rmse = std::numeric_limits<double>::infinity();
        metrics.psnr = -std::numeric_limits<double>::infinity();
        return metrics;
    }

    std::sort(errs.begin(), errs.end());
    const size_t idx99 = std::min(errs.size() - 1, static_cast<size_t>(sampleCount * 0.99));
    const size_t idx999 = std::min(errs.size() - 1, static_cast<size_t>(sampleCount * 0.999));
    metrics.mean = errSum / std::max(1LL, sampleCount);
    metrics.max = errMax;
    metrics.p99 = errs[idx99];
    metrics.p999 = errs[idx999];
    metrics.rmse = std::sqrt(errSum2 / std::max(1LL, sampleCount));
    const double psnrPeak = useDataRangeForPsnr ? std::max(1e-6, dataMax - dataMin) : 255.0;
    metrics.psnr = computePsnr(metrics.rmse, psnrPeak);
    return metrics;
}

static int chooseDensityProxyFrame(
    const Volume4D& volumeSequence,
    int width, int height, int depth, int frames,
    float cutoff,
    int stepX,
    int stepY,
    int stepZ)
{
    int bestFrame = 0;
    long long bestVisible = -1;
    for (int t = 0; t < frames; ++t) {
        long long visible = 0;
        for (int z = 0; z < depth; z += stepZ) {
            for (int y = 0; y < height; y += stepY) {
                for (int x = 0; x < width; x += stepX) {
                    if (volumeSequence[t][z][y][x] >= cutoff) ++visible;
                }
            }
        }
        if (visible > bestVisible) {
            bestVisible = visible;
            bestFrame = t;
        }
    }
    return bestFrame;
}

static std::vector<float> computeDensityProjectionProxyOrig(
    const Volume4D& volumeSequence,
    int width, int height, int depth,
    int frame,
    float cutoff,
    int stepX,
    int stepY,
    int stepZ)
{
    const int outW = (width + stepX - 1) / stepX;
    const int outH = (height + stepY - 1) / stepY;
    std::vector<float> img(static_cast<size_t>(outW * outH), 0.0f);
    const float scale = std::max(cutoff, 1e-6f);
    for (int oy = 0; oy < outH; ++oy) {
        const int y = std::min(height - 1, oy * stepY);
        for (int ox = 0; ox < outW; ++ox) {
            const int x = std::min(width - 1, ox * stepX);
            float accum = 0.0f;
            float trans = 1.0f;
            for (int z = 0; z < depth; z += stepZ) {
                const float v = volumeSequence[frame][z][y][x];
                const float dens = std::max(0.0f, v - cutoff);
                const float alpha = 1.0f - std::exp(-2.0f * dens / scale);
                accum += trans * alpha;
                trans *= (1.0f - alpha);
                if (trans < 1e-3f) break;
            }
            img[static_cast<size_t>(oy * outW + ox)] = accum;
        }
    }
    return img;
}

static std::vector<float> computeDensityProjectionProxyTree(
    BlockTree& btree,
    int width, int height, int depth,
    int frame,
    float cutoff,
    int stepX,
    int stepY,
    int stepZ)
{
    const int outW = (width + stepX - 1) / stepX;
    const int outH = (height + stepY - 1) / stepY;
    std::vector<float> img(static_cast<size_t>(outW * outH), 0.0f);
    const float scale = std::max(cutoff, 1e-6f);
    for (int oy = 0; oy < outH; ++oy) {
        const int y = std::min(height - 1, oy * stepY);
        for (int ox = 0; ox < outW; ++ox) {
            const int x = std::min(width - 1, ox * stepX);
            float accum = 0.0f;
            float trans = 1.0f;
            for (int z = 0; z < depth; z += stepZ) {
                const float v = btree.sampleFlat(x, y, z, static_cast<float>(frame));
                const float dens = std::max(0.0f, v - cutoff);
                const float alpha = 1.0f - std::exp(-2.0f * dens / scale);
                accum += trans * alpha;
                trans *= (1.0f - alpha);
                if (trans < 1e-3f) break;
            }
            img[static_cast<size_t>(oy * outW + ox)] = accum;
        }
    }
    return img;
}

static void computeProxyImageMetrics(
    const std::vector<float>& orig,
    const std::vector<float>& recon,
    double& outRmse,
    double& outPsnr)
{
    if (orig.empty() || recon.empty() || orig.size() != recon.size()) {
        outRmse = std::numeric_limits<double>::infinity();
        outPsnr = -std::numeric_limits<double>::infinity();
        return;
    }
    double sse = 0.0;
    double peak = 1e-6;
    for (size_t i = 0; i < orig.size(); ++i) {
        const double d = static_cast<double>(orig[i]) - recon[i];
        sse += d * d;
        peak = std::max(peak, static_cast<double>(orig[i]));
    }
    outRmse = std::sqrt(sse / static_cast<double>(orig.size()));
    outPsnr = computePsnr(outRmse, peak);
}

static EvalMetrics evaluateSpatialPolicyCandidate(
    const Volume4D& volumeSequence,
    const CompressedVolume4D& compressedVolume,
    int width, int height, int depth, int frames,
    const FieldProfile& fieldProfile,
    bool enableBackgroundElision,
    bool enableBlockAwareCluster,
    bool enableBudgetAwareCluster,
    bool enableGuardedMedoidCluster,
    bool enableGrid4Spatial,
    bool enableValidateFallback,
    bool enableHotspotSecondPass,
    double clusterThr,
    double dataMin,
    double dataMax,
    bool useDataRangeForPsnr,
    const Grid4MultiscaleOptions& grid4MsOpt,
    const ResidualOptions& residualOpt,
    const AutoPolicySearchOptions& autoPolicyOpt,
    int densityProxyFrame)
{
    EvalMetrics metrics;
    BlockTree btree;
    try {
        btree.build(compressedVolume, width, height, depth, frames,
                    -1.0, clusterThr, fieldProfile, enableBackgroundElision, enableBlockAwareCluster,
                    enableBudgetAwareCluster, enableGuardedMedoidCluster, enableGrid4Spatial,
                    false, &volumeSequence, enableValidateFallback, enableHotspotSecondPass);
        if (grid4MsOpt.enabled && enableGrid4Spatial) {
            applyGrid4MultiscaleAdaptive(
                volumeSequence, compressedVolume, btree, width, height, depth, frames,
                fieldProfile, grid4MsOpt);
        }
        if (residualOpt.grid4ResidualHotspots > 0 && enableGrid4Spatial) {
            std::vector<LeafErrorStats> leafStats =
                (residualOpt.grid4ResidualLeafSampleStep > 1)
                ? analyzeLeafErrorsSampled(
                    volumeSequence, compressedVolume, btree, width, height, depth, frames,
                    residualOpt.grid4ResidualLeafSampleStep)
                : analyzeLeafErrors(
                    volumeSequence, compressedVolume, btree, width, height, depth, frames);
            const double fullRange = std::max(1e-6, static_cast<double>(dataMax) - dataMin);
            const bool densityLike = (fieldProfile.type == FieldType::DENSITY);
            const double residualThr = (residualOpt.grid4ResidualThr > 0.0)
                ? residualOpt.grid4ResidualThr
                : (densityLike ? std::max(1e-6, 0.10 * fullRange)
                               : std::max(0.005, 0.03 * fullRange));
            const double residualRelThr = (residualOpt.grid4ResidualRelThr > 0.0)
                ? residualOpt.grid4ResidualRelThr
                : (densityLike ? 0.10 : std::numeric_limits<double>::infinity());
            const double residualLocalFloor = (residualOpt.grid4ResidualLocalFloor > 0.0)
                ? residualOpt.grid4ResidualLocalFloor
                : (densityLike ? std::max(1e-6, 0.10 * fullRange)
                               : std::max(1e-6, 0.01 * fullRange));
            applyGrid4ResidualHotspots(
                btree, volumeSequence, compressedVolume, leafStats, fieldProfile,
                width, height, depth, frames,
                residualOpt.grid4ResidualHotspots, residualThr, residualRelThr, residualLocalFloor,
                residualOpt.grid4ResidualBandFactor, residualOpt.grid4ResidualKeepRel,
                residualOpt.grid4ResidualRankNormalized || densityLike,
                residualOpt.grid4ResidualDpEps);
        }
        btree.flattenLeaves();
    } catch (const std::exception&) {
        metrics.valid = false;
        metrics.vbtBytes = std::numeric_limits<size_t>::max();
        metrics.rmse = std::numeric_limits<double>::infinity();
        metrics.psnr = -std::numeric_limits<double>::infinity();
        metrics.mean = std::numeric_limits<double>::infinity();
        metrics.max = std::numeric_limits<double>::infinity();
        metrics.p99 = std::numeric_limits<double>::infinity();
        metrics.p999 = std::numeric_limits<double>::infinity();
        return metrics;
    }
    metrics = evaluateBuiltTreeSampled(
        btree, volumeSequence, width, height, depth, frames,
        dataMin, dataMax, useDataRangeForPsnr,
        autoPolicyOpt.sampleStepX, autoPolicyOpt.sampleStepY,
        autoPolicyOpt.sampleStepZ, autoPolicyOpt.sampleStepT);
    if (fieldProfile.type == FieldType::DENSITY &&
        fieldProfile.den.render_cutoff > 0.0f &&
        densityProxyFrame >= 0) {
        const std::vector<float> origImg = computeDensityProjectionProxyOrig(
            volumeSequence, width, height, depth, densityProxyFrame,
            fieldProfile.den.render_cutoff,
            std::max(1, autoPolicyOpt.sampleStepX * 2),
            std::max(1, autoPolicyOpt.sampleStepY * 2),
            std::max(1, autoPolicyOpt.sampleStepZ));
        const std::vector<float> reconImg = computeDensityProjectionProxyTree(
            btree, width, height, depth, densityProxyFrame,
            fieldProfile.den.render_cutoff,
            std::max(1, autoPolicyOpt.sampleStepX * 2),
            std::max(1, autoPolicyOpt.sampleStepY * 2),
            std::max(1, autoPolicyOpt.sampleStepZ));
        computeProxyImageMetrics(origImg, reconImg, metrics.renderProxyRmse, metrics.renderProxyPsnr);
    }
    return metrics;
}

static std::vector<SpatialPolicyCandidate> makeDensityPolicyCandidates(
    const SampledFieldStats& sampledStats,
    const ProfileOptions& profileOpt,
    const Grid4MultiscaleOptions& currentMs,
    const ResidualOptions& currentResidual)
{
    std::vector<SpatialPolicyCandidate> out;
    const bool sparseLike =
        sampledStats.maxValue <= 0.01 &&
        sampledStats.lag1Autocorr <= 0.30;

    auto pushMs = [&](const std::string& name, double fineDp, double t1, double vis6) {
        SpatialPolicyCandidate c;
        c.name = name;
        c.mode = AutoSpatialRouteMode::GRID4_MULTISCALE;
        c.grid4Ms = currentMs;
        c.grid4Ms.enabled = true;
        c.grid4Ms.fineResidualDpEps = fineDp;
        c.grid4Ms.scoreT1 = t1;
        c.grid4Ms.visPromoteFine6 = vis6;
        out.push_back(c);
    };
    auto pushSparse = [&](const std::string& name, int topK) {
        SpatialPolicyCandidate c;
        c.name = name;
        c.mode = AutoSpatialRouteMode::GRID4_RESIDUAL;
        c.grid4Ms = currentMs;
        c.grid4Ms.enabled = false;
        c.residual = currentResidual;
        c.residual.grid4ResidualHotspots = topK;
        c.residual.grid4ResidualDpEps = 6.0;
        c.residual.grid4ResidualRankNormalized = true;
        c.residual.grid4ResidualRelThr = 0.10;
        c.residual.grid4ResidualLocalFloor =
            std::max(0.0003, profileOpt.renderCutoff > 0.0 ? 0.60 * profileOpt.renderCutoff : 0.0003);
        c.residual.grid4ResidualBandFactor = 0.80;
        c.residual.grid4ResidualKeepRel = 0.06;
        out.push_back(c);
    };

    if (sparseLike) {
        pushSparse("sparse_topk1024", 1024);
        pushSparse("sparse_topk1408", 1408);
        pushSparse("sparse_topk1536", 1536);
        pushMs("multiscale_ref", 2.8, 0.50, 0.05);
    } else {
        pushMs("multiscale_balanced", 2.8, 0.42, 0.04);
        pushMs("multiscale_lean", 2.8, 0.50, 0.05);
        pushMs("multiscale_compact", 2.8, 0.55, 0.06);
        pushSparse("sparse_topk1024", 1024);
    }
    return out;
}

static std::vector<SpatialPolicyCandidate> makeGenericPolicyCandidates(
    const SampledFieldStats& sampledStats,
    const Grid4MultiscaleOptions& currentMs,
    const ResidualOptions& currentResidual)
{
    std::vector<SpatialPolicyCandidate> out;

    auto pushGrid4 = [&](const std::string& name) {
        SpatialPolicyCandidate c;
        c.name = name;
        c.mode = AutoSpatialRouteMode::GRID4;
        c.grid4Ms = currentMs;
        c.grid4Ms.enabled = false;
        c.residual = currentResidual;
        c.residual.grid4ResidualHotspots = 0;
        out.push_back(c);
    };
    auto pushSparse = [&](const std::string& name, int topK) {
        SpatialPolicyCandidate c;
        c.name = name;
        c.mode = AutoSpatialRouteMode::GRID4_RESIDUAL;
        c.grid4Ms = currentMs;
        c.grid4Ms.enabled = false;
        c.residual = currentResidual;
        c.residual.grid4ResidualHotspots = topK;
        c.residual.grid4ResidualDpEps = 6.0;
        c.residual.grid4ResidualRankNormalized = false;
        c.residual.grid4ResidualThr = -1.0;
        c.residual.grid4ResidualRelThr = std::numeric_limits<double>::infinity();
        c.residual.grid4ResidualLocalFloor = -1.0;
        out.push_back(c);
    };

    pushGrid4("grid4_plain");
    pushSparse("sparse_topk8192", 8192);
    pushSparse("sparse_topk16384", 16384);
    pushSparse("sparse_topk23552", 23552);
    pushSparse("sparse_topk32256", 32256);
    return out;
}

static int pickBestDensityPolicyCandidate(
    const std::vector<SpatialPolicyEvalResult>& results,
    const SampledFieldStats& sampledStats)
{
    if (results.empty()) return -1;
    const bool sparseLike =
        sampledStats.maxValue <= 0.01 &&
        sampledStats.lag1Autocorr <= 0.30;
    const double targetPsnr = sparseLike ? 30.0 : 42.0;
    int bestIdx = -1;
    size_t bestBytes = std::numeric_limits<size_t>::max();
    double fallbackBestPsnr = -std::numeric_limits<double>::infinity();
    size_t fallbackBestBytes = std::numeric_limits<size_t>::max();
    int fallbackIdx = -1;
    for (int i = 0; i < static_cast<int>(results.size()); ++i) {
        const auto& r = results[static_cast<size_t>(i)];
        if (!r.metrics.valid) continue;
        const double proxyScore = r.metrics.renderProxyPsnr;
        if (r.metrics.psnr >= targetPsnr) {
            if (proxyScore > -std::numeric_limits<double>::infinity()) {
                if (bestIdx < 0 ||
                    proxyScore > results[static_cast<size_t>(bestIdx)].metrics.renderProxyPsnr + 0.20 ||
                    (std::abs(proxyScore - results[static_cast<size_t>(bestIdx)].metrics.renderProxyPsnr) <= 0.20 &&
                     r.metrics.vbtBytes < bestBytes)) {
                    bestBytes = r.metrics.vbtBytes;
                    bestIdx = i;
                }
            } else if (r.metrics.vbtBytes < bestBytes) {
                bestBytes = r.metrics.vbtBytes;
                bestIdx = i;
            }
        }
        const double fallbackScore = (proxyScore > -std::numeric_limits<double>::infinity()) ? proxyScore : r.metrics.psnr;
        if (fallbackScore > fallbackBestPsnr + 0.10 ||
            (std::abs(fallbackScore - fallbackBestPsnr) <= 0.10 && r.metrics.vbtBytes < fallbackBestBytes)) {
            fallbackBestPsnr = fallbackScore;
            fallbackBestBytes = r.metrics.vbtBytes;
            fallbackIdx = i;
        }
    }
    return bestIdx >= 0 ? bestIdx : fallbackIdx;
}

static int pickBestGenericPolicyCandidate(
    const std::vector<SpatialPolicyEvalResult>& results,
    double targetPsnr)
{
    if (results.empty()) return -1;
    int bestIdx = -1;
    size_t bestBytes = std::numeric_limits<size_t>::max();
    double fallbackBestPsnr = -std::numeric_limits<double>::infinity();
    size_t fallbackBestBytes = std::numeric_limits<size_t>::max();
    int fallbackIdx = -1;
    for (int i = 0; i < static_cast<int>(results.size()); ++i) {
        const auto& r = results[static_cast<size_t>(i)];
        if (!r.metrics.valid) continue;
        if (r.metrics.psnr >= targetPsnr) {
            if (r.metrics.vbtBytes < bestBytes) {
                bestBytes = r.metrics.vbtBytes;
                bestIdx = i;
            }
        }
        if (r.metrics.psnr > fallbackBestPsnr + 0.10 ||
            (std::abs(r.metrics.psnr - fallbackBestPsnr) <= 0.10 && r.metrics.vbtBytes < fallbackBestBytes)) {
            fallbackBestPsnr = r.metrics.psnr;
            fallbackBestBytes = r.metrics.vbtBytes;
            fallbackIdx = i;
        }
    }
    return bestIdx >= 0 ? bestIdx : fallbackIdx;
}

static CompressedVolume4D temporalCompress(
    const Volume4D& volumeSequence,
    int width, int height, int depth, int frames,
    const FieldProfile& fieldProfile,
    long long& totalOrig,
    long long& totalKF,
    bool verbose)
{
    CompressedVolume4D compressedVolume(depth);
    for (int z = 0; z < depth; ++z) {
        compressedVolume[z].resize(height);
        for (int y = 0; y < height; ++y) {
            compressedVolume[z][y].resize(width);
        }
    }

    totalOrig = 0;
    totalKF = 0;
    const int totalVoxels = width * height * depth;
    int processed = 0;
    const int progressStep = std::max(1, totalVoxels / 10);
    int actualThreads = 1;

#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp master
        {
            actualThreads = omp_get_num_threads();
        }
        #pragma omp for collapse(3) schedule(dynamic, 8) reduction(+:totalOrig,totalKF,processed)
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    std::vector<float> seq(frames);
                    for (int t = 0; t < frames; ++t) {
                        seq[t] = volumeSequence[t][z][y][x];
                    }

                    std::vector<int> kfIdx = detectKeyFrames(seq, 2.0, fieldProfile);
                    std::vector<Point1D> kfs;
                    kfs.reserve(kfIdx.size());
                    for (int idx : kfIdx) {
                        kfs.push_back(Point1D(idx, seq[idx]));
                    }

                    compressedVolume[z][y][x] = std::move(kfs);
                    totalOrig += frames;
                    totalKF += static_cast<long long>(kfIdx.size());
                    ++processed;
                }
            }
        }
    }
#else
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                std::vector<float> seq(frames);
                for (int t = 0; t < frames; ++t) {
                    seq[t] = volumeSequence[t][z][y][x];
                }

                std::vector<int> kfIdx = detectKeyFrames(seq, 2.0, fieldProfile);
                std::vector<Point1D> kfs;
                kfs.reserve(kfIdx.size());
                for (int idx : kfIdx) {
                    kfs.push_back(Point1D(idx, seq[idx]));
                }

                compressedVolume[z][y][x] = std::move(kfs);
                totalOrig += frames;
                totalKF += static_cast<long long>(kfIdx.size());
                ++processed;

                if (verbose && processed % progressStep == 0) {
                    printf("  %d%%\n", 100 * processed / totalVoxels);
                }
            }
        }
    }
#endif

#ifdef _OPENMP
    if (verbose) {
        printf("  [OpenMP] temporalCompress threads=%d\n", actualThreads);
        printf("  100%%\n");
    }
#endif

    return compressedVolume;
}

static size_t computeFlatVbtBytes(const BlockTree& tree) {
    return sizeof(VBTHeader)
         + tree.rootTable.size()   * sizeof(uint32_t)
         + tree.internalNodes.size() * 20u
         + tree.childList.size()   * sizeof(uint32_t)
         + tree.leafHeaders.size() * sizeof(LeafHeader)
         + tree.topoPool.size()    * sizeof(uint32_t)
         + tree.seqMetaPool.size() * sizeof(SeqMeta)
         + tree.seqPool.size()     * sizeof(uint32_t);
}

static EvalMetrics evaluateCandidate(
    const Volume4D& volumeSequence,
    int width, int height, int depth, int frames,
    const FieldProfile& fieldProfile,
    bool enableBackgroundElision,
    bool enableBlockAwareCluster,
    bool enableBudgetAwareCluster,
    bool enableGuardedMedoidCluster,
    bool enableGrid4Spatial,
    bool enableHotspotSecondPass,
    double clusterThr,
    double dataMin,
    double dataMax,
    bool useDataRangeForPsnr)
{
    long long totalOrig = 0;
    long long totalKF = 0;
    CompressedVolume4D compressedVolume = temporalCompress(
        volumeSequence, width, height, depth, frames, fieldProfile, totalOrig, totalKF, false);

    EvalMetrics metrics;
    BlockTree btree;
    try {
        btree.build(compressedVolume, width, height, depth, frames,
                    -1.0, clusterThr, fieldProfile, enableBackgroundElision, enableBlockAwareCluster,
                    enableBudgetAwareCluster, enableGuardedMedoidCluster, enableGrid4Spatial, false,
                    nullptr, false, enableHotspotSecondPass);
        btree.flattenLeaves();
    } catch (const std::exception&) {
        metrics.valid = false;
        metrics.vbtBytes = std::numeric_limits<size_t>::max();
        metrics.rmse = std::numeric_limits<double>::infinity();
        metrics.psnr = -std::numeric_limits<double>::infinity();
        metrics.mean = std::numeric_limits<double>::infinity();
        metrics.max = std::numeric_limits<double>::infinity();
        metrics.p99 = std::numeric_limits<double>::infinity();
        metrics.p999 = std::numeric_limits<double>::infinity();
        return metrics;
    }
    metrics.vbtBytes = computeFlatVbtBytes(btree);
    metrics.leaves = static_cast<int>(btree.leafHeaders.size());
    metrics.totalKF = btree.getStats().totalKF;

    const long long N = static_cast<long long>(width) * height * depth * frames;
    std::vector<float> errs;
    errs.reserve(static_cast<size_t>(N));
    double errSum = 0.0;
    double errSum2 = 0.0;
    float errMax = 0.0f;

    #ifdef _OPENMP
    #pragma omp parallel
    {
        std::vector<float> localErrs;
        double localErrSum = 0.0;
        double localErrSum2 = 0.0;
        float localErrMax = 0.0f;

        #pragma omp for collapse(3) nowait
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    for (int t = 0; t < frames; ++t) {
                        float got = btree.sampleFlat(x, y, z, static_cast<float>(t));
                        float orig = volumeSequence[t][z][y][x];
                        float e = std::abs(got - orig);
                        localErrs.push_back(e);
                        localErrSum += e;
                        localErrSum2 += static_cast<double>(e) * e;
                        localErrMax = std::max(localErrMax, e);
                    }
                }
            }
        }

        #pragma omp critical
        {
            errs.insert(errs.end(), localErrs.begin(), localErrs.end());
            errSum += localErrSum;
            errSum2 += localErrSum2;
            errMax = std::max(errMax, localErrMax);
        }
    }
    #else
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int t = 0; t < frames; ++t) {
                    float got = btree.sampleFlat(x, y, z, static_cast<float>(t));
                    float orig = volumeSequence[t][z][y][x];
                    float e = std::abs(got - orig);
                    errs.push_back(e);
                    errSum += e;
                    errSum2 += static_cast<double>(e) * e;
                    errMax = std::max(errMax, e);
                }
            }
        }
    }
    #endif

    std::sort(errs.begin(), errs.end());
    const size_t idx99 = std::min(errs.size() - 1, static_cast<size_t>(N * 0.99));
    const size_t idx999 = std::min(errs.size() - 1, static_cast<size_t>(N * 0.999));
    metrics.mean = errSum / N;
    metrics.max = errMax;
    metrics.p99 = errs[idx99];
    metrics.p999 = errs[idx999];
    metrics.rmse = std::sqrt(errSum2 / N);
    const double psnrPeak = useDataRangeForPsnr ? std::max(1e-6, dataMax - dataMin) : 255.0;
    metrics.psnr = computePsnr(metrics.rmse, psnrPeak);

    return metrics;
}

static std::vector<CalibrationCandidate> makeCalibrationCandidates(
    double dataRange,
    float currentAbs,
    float currentRel,
    float currentGamma,
    float currentClusterThr)
{
    const float legacyTemporalAbs = static_cast<float>(dataRange * (6.0 / 255.0));
    const float legacySpatialThr = 8.0f / 6.0f;
    const std::vector<float> absMultipliers = {0.5f, 0.75f, 1.0f, 1.5f, 2.0f};
    const std::vector<float> rels = {0.0f, 0.01f, 0.02f};
    const std::vector<float> gammas = {0.0f, 0.2f};
    const std::vector<float> clusterMultipliers = {0.5f, 0.75f, 1.0f, 1.5f, 2.0f};

    std::vector<CalibrationCandidate> out;
    auto pushUnique = [&](CalibrationCandidate c) {
        for (const auto& existing : out) {
            if (std::fabs(existing.epsAbs - c.epsAbs) < 1e-6f &&
                std::fabs(existing.epsRel - c.epsRel) < 1e-6f &&
                std::fabs(existing.gammaDelta - c.gammaDelta) < 1e-6f &&
                std::fabs(existing.clusterThr - c.clusterThr) < 1e-6f) {
                return;
            }
        }
        out.push_back(c);
    };

    pushUnique({currentAbs, currentRel, currentGamma, currentClusterThr});
    pushUnique({legacyTemporalAbs, currentRel, currentGamma, legacySpatialThr});
    for (float absMul : absMultipliers) {
        const float absValue = std::max(1e-6f, currentAbs * absMul);
        for (float r : rels) {
            for (float g : gammas) {
                for (float clusterMul : clusterMultipliers) {
                    const float clusterValue = std::max(0.1f, currentClusterThr * clusterMul);
                    pushUnique({absValue, r, g, clusterValue});
                }
            }
        }
    }
    return out;
}

static bool candidateMeetsTargets(const CalibrationOptions& calib, const EvalMetrics& m) {
    if (calib.targetP99 >= 0.0 && m.p99 > calib.targetP99) return false;
    if (calib.targetP999 >= 0.0 && m.p999 > calib.targetP999) return false;
    if (calib.targetPsnr >= 0.0 && m.psnr < calib.targetPsnr) return false;
    if (calib.targetRmse >= 0.0 && m.rmse > calib.targetRmse) return false;
    return true;
}

static bool calibrationHasTargets(const CalibrationOptions& calib) {
    return calib.targetP99 >= 0.0
        || calib.targetP999 >= 0.0
        || calib.targetPsnr >= 0.0
        || calib.targetRmse >= 0.0;
}

static int pickBalancedCandidate(const std::vector<CalibrationResult>& results) {
    if (results.empty()) return -1;
    double bestPsnr = results.front().metrics.psnr;
    for (const auto& r : results) {
        bestPsnr = std::max(bestPsnr, r.metrics.psnr);
    }

    int bestIdx = 0;
    size_t bestBytes = std::numeric_limits<size_t>::max();
    for (int i = 0; i < static_cast<int>(results.size()); ++i) {
        if (results[i].metrics.psnr + 0.25 < bestPsnr) continue;
        if (results[i].metrics.vbtBytes < bestBytes) {
            bestBytes = results[i].metrics.vbtBytes;
            bestIdx = i;
        }
    }
    return bestIdx;
}

static int pickBestTargetedCandidate(const CalibrationOptions& calib, const std::vector<CalibrationResult>& results) {
    int bestIdx = -1;
    size_t bestBytes = std::numeric_limits<size_t>::max();
    for (int i = 0; i < static_cast<int>(results.size()); ++i) {
        if (!candidateMeetsTargets(calib, results[i].metrics)) continue;
        if (results[i].metrics.vbtBytes < bestBytes) {
            bestBytes = results[i].metrics.vbtBytes;
            bestIdx = i;
        }
    }
    return bestIdx;
}

static int runCalibration(
    const Volume4D& volumeSequence,
    int width, int height, int depth, int frames,
    const ProfileOptions& profileOpt,
    bool enableBackgroundElision,
    bool enableBlockAwareCluster,
    bool enableBudgetAwareCluster,
    bool enableGuardedMedoidCluster,
    bool enableGrid4Spatial,
    bool enableHotspotSecondPass,
    double clusterThr,
    const CalibrationOptions& calib,
    double globalDataMin,
    double globalDataMax,
    bool useDataRangeForPsnr)
{
    if (profileOpt.name == "SDF") {
        std::cerr << "Calibration mode currently targets DENSITY/GENERIC scalar profiles." << std::endl;
        return 1;
    }

    const int subW = std::min(calib.subW, width);
    const int subH = std::min(calib.subH, height);
    const int subD = std::min(calib.subD, depth);
    const int subF = std::min(calib.subF, frames);
    const double globalRange = std::max(1e-6, globalDataMax - globalDataMin);
    const long long rawBytesF32 = static_cast<long long>(subW) * subH * subD * subF * 4LL;
    const std::vector<SubvolumeSpec> blocks = selectCalibrationBlocks(
        volumeSequence, width, height, depth, frames, calib);

    std::vector<CalibrationCandidate> candidates = makeCalibrationCandidates(
        globalRange, profileOpt.epsAbs, profileOpt.epsRel, profileOpt.gammaDelta, static_cast<float>(clusterThr));

    printf("\n=== Calibration Mode ===\n");
    printf("  Profile             : %s\n", profileOpt.name.c_str());
    printf("  Subvolume           : %dx%dx%d  frames=%d\n", subW, subH, subD, subF);
    printf("  Global Data Range   : [%.6f, %.6f]  span=%.6f\n", globalDataMin, globalDataMax, globalRange);
    printf("  Background Elision  : %s\n", enableBackgroundElision ? "enabled" : "disabled");
    printf("  Block-aware cluster : %s\n", enableBlockAwareCluster ? "enabled" : "disabled");
    printf("  Budget-aware cluster: %s\n", enableBudgetAwareCluster ? "enabled" : "disabled");
    printf("  Guarded medoid      : %s\n", enableGuardedMedoidCluster ? "enabled" : "disabled");
    printf("  GRID4 spatial       : %s\n", enableGrid4Spatial ? "enabled" : "disabled");
    printf("  Hotspot second pass : %s\n", enableHotspotSecondPass ? "enabled" : "disabled");
    printf("  Calibration Blocks  : %d\n", static_cast<int>(blocks.size()));
    printf("  Candidate Count     : %d\n", static_cast<int>(candidates.size()));
    printf("  eps_abs center      : current default = %.6f\n", profileOpt.epsAbs);
    printf("  eps_abs multipliers : {0.5, 0.75, 1.0, 1.5, 2.0}\n");
    printf("  eps_rel seeds       : {0, 0.01, 0.02}\n");
    printf("  gamma_delta seeds   : {0, 0.2}\n");
    printf("  cluster_thr center  : current default = %.6f\n", clusterThr);
    printf("  cluster multipliers : {0.5, 0.75, 1.0, 1.5, 2.0}\n");
    for (const auto& b : blocks) {
        printf("  Block %-12s : x=%d y=%d z=%d t=%d\n", b.tag.c_str(), b.startX, b.startY, b.startZ, b.startT);
    }

    std::vector<CalibrationResult> results;
    results.reserve(candidates.size());

    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        ProfileOptions tuned = profileOpt;
        tuned.epsAbs = candidates[i].epsAbs;
        tuned.epsRel = candidates[i].epsRel;
        tuned.gammaDelta = candidates[i].gammaDelta;
        FieldProfile fieldProfile = makeFieldProfile(tuned);

        printf("\n[Calib %d/%d] eps_abs=%.6f eps_rel=%.4f gamma=%.3f cluster=%.3f\n",
               i + 1, static_cast<int>(candidates.size()),
               tuned.epsAbs, tuned.epsRel, tuned.gammaDelta, candidates[i].clusterThr);

        CalibrationResult row;
        row.candidate = candidates[i];
        std::vector<EvalMetrics> perBlock;
        perBlock.reserve(blocks.size());
        for (const auto& b : blocks) {
            float localMin = 0.0f;
            float localMax = 0.0f;
            Volume4D subVolume = extractSubvolume(
                volumeSequence, b.startX, b.startY, b.startZ, b.startT,
                subW, subH, subD, subF, localMin, localMax);
            perBlock.push_back(evaluateCandidate(
                subVolume, subW, subH, subD, subF,
                fieldProfile, enableBackgroundElision, enableBlockAwareCluster, enableBudgetAwareCluster,
                enableGuardedMedoidCluster, enableGrid4Spatial, enableHotspotSecondPass,
                candidates[i].clusterThr,
                globalDataMin, globalDataMax, useDataRangeForPsnr));
        }
        row.metrics = aggregateMetrics(
            perBlock,
            useDataRangeForPsnr ? std::max(1e-6, globalDataMax - globalDataMin) : 255.0);
        results.push_back(row);

        printf("  -> AvgVBT=%.2f KB  RMSE=%.6f  PSNR=%.2f  P99=%.4f  P99.9=%.4f\n",
               row.metrics.vbtBytes / 1024.0,
               row.metrics.rmse, row.metrics.psnr, row.metrics.p99, row.metrics.p999);
    }

    std::vector<CalibrationResult> sorted = results;
    std::sort(sorted.begin(), sorted.end(), [](const CalibrationResult& a, const CalibrationResult& b) {
        if (a.metrics.vbtBytes != b.metrics.vbtBytes) return a.metrics.vbtBytes < b.metrics.vbtBytes;
        return a.metrics.psnr > b.metrics.psnr;
    });

    printf("\n=== Calibration Table (sorted by VBT size) ===\n");
    printf(" rank  eps_abs    eps_rel  gamma  cthr   VBT_KB   RMSE      PSNR    P99     P99.9   Leaves   KF\n");
    for (int i = 0; i < static_cast<int>(sorted.size()); ++i) {
        const auto& r = sorted[i];
        printf(" %4d  %8.6f  %7.4f  %5.3f  %4.2f  %7.2f  %8.6f  %6.2f  %6.4f  %7.4f  %6d  %lld\n",
               i + 1,
               r.candidate.epsAbs, r.candidate.epsRel, r.candidate.gammaDelta, r.candidate.clusterThr,
               r.metrics.vbtBytes / 1024.0,
               r.metrics.rmse, r.metrics.psnr, r.metrics.p99, r.metrics.p999,
               r.metrics.leaves, r.metrics.totalKF);
    }

    int bestTargeted = calibrationHasTargets(calib) ? pickBestTargetedCandidate(calib, results) : -1;
    if (bestTargeted >= 0) {
        const auto& r = results[bestTargeted];
        printf("\nCALIB_RECOMMENDED target_matched eps_abs=%.6f eps_rel=%.4f gamma=%.3f cluster=%.3f"
               " vbt_kb=%.2f rmse=%.6f psnr=%.2f p99=%.4f p999=%.4f\n",
               r.candidate.epsAbs, r.candidate.epsRel, r.candidate.gammaDelta, r.candidate.clusterThr,
               r.metrics.vbtBytes / 1024.0, r.metrics.rmse, r.metrics.psnr,
               r.metrics.p99, r.metrics.p999);
    } else {
        int balancedIdx = pickBalancedCandidate(results);
        if (balancedIdx >= 0) {
            const auto& r = results[balancedIdx];
            printf("\nCALIB_RECOMMENDED balanced eps_abs=%.6f eps_rel=%.4f gamma=%.3f cluster=%.3f"
                   " vbt_kb=%.2f rmse=%.6f psnr=%.2f p99=%.4f p999=%.4f\n",
                   r.candidate.epsAbs, r.candidate.epsRel, r.candidate.gammaDelta, r.candidate.clusterThr,
                   r.metrics.vbtBytes / 1024.0, r.metrics.rmse, r.metrics.psnr,
                   r.metrics.p99, r.metrics.p999);
        }
    }

    printf("CALIB_NOTE raw_subset_fp32_mb=%.2f candidate_count=%d blocks=%d\n",
           rawBytesF32 / (1024.0 * 1024.0), static_cast<int>(candidates.size()), static_cast<int>(blocks.size()));
    return 0;
}

int main(int argc, char* argv[]) {
    std::cout << "========== 3D Scalar Field Spatiotemporal Compression (Float16) ==========" << std::endl;

#ifdef _OPENMP
    omp_set_dynamic(0);
    const char* ompThreadsEnv = std::getenv("OMP_NUM_THREADS");
    if (ompThreadsEnv == nullptr || ompThreadsEnv[0] == '\0') {
        omp_set_num_threads(omp_get_num_procs());
    }
    printf("OpenMP: enabled  version=%d  max_threads=%d  num_procs=%d  dynamic=%d\n",
           _OPENMP, omp_get_max_threads(), omp_get_num_procs(), omp_get_dynamic());
#else
    printf("OpenMP: disabled\n");
#endif

    double v5ThreshCluster = 8.0;
    bool clusterSpecified = false;
    bool epsAbsSpecified = false;
    bool gammaSpecified = false;
    bool enableBlockAwareCluster = false;
    bool enableBudgetAwareCluster = false;
    bool enableGuardedMedoidCluster = false;
    bool enableGrid4Spatial = false;
    bool enablePoly11Spatial = false;
    Grid4EncoderOptions grid4EncoderOpt;
    bool autoGrid4Spatial = false;
    bool autoSpatialRouting = false;
    bool enableValidateFallback = false;
    bool enableHotspotSecondPass = false;
    ProfileOptions profileOpt;
    CalibrationOptions calib;
    LeafAnalysisOptions leafAnalysis;
    SpatialProbeOptions spatialProbe;
    ResidualOptions residualOpt;
    Grid4MultiscaleOptions grid4MsOpt;
    AutoPolicySearchOptions autoPolicyOpt;

    int backgroundElisionOverride = -1;
    bool timeOnly = false;
    bool skipFullSweep = false;
    bool sampledFullSweep = false;
    int sampledStepX = 8;
    int sampledStepY = 8;
    int sampledStepZ = 8;
    int sampledStepT = 8;
    int hintW = 0;
    int hintH = 0;
    int hintD = 0;
    int hintF = 0;
    std::string inputFile = "data/cube_to_sphere_64_91.raw";
    bool manualSpatialMode = false;
    bool manualResidualMode = false;
    bool manualGrid4MultiscaleMode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto eqval = [&](int skip) { return std::atof(arg.c_str() + skip); };
        auto eqint = [&](int skip) { return std::atoi(arg.c_str() + skip); };

        if      (arg.substr(0, 13) == "--v5-cluster=")     { v5ThreshCluster = eqval(13); clusterSpecified = true; }
        else if (arg == "--block-aware-cluster")           enableBlockAwareCluster = true;
        else if (arg == "--no-block-aware-cluster")        enableBlockAwareCluster = false;
        else if (arg == "--budget-aware-cluster")          enableBudgetAwareCluster = true;
        else if (arg == "--no-budget-aware-cluster")       enableBudgetAwareCluster = false;
        else if (arg == "--guarded-medoid-cluster")        enableGuardedMedoidCluster = true;
        else if (arg == "--no-guarded-medoid-cluster")     enableGuardedMedoidCluster = false;
        else if (arg == "--grid4-spatial")                 { enableGrid4Spatial = true; manualSpatialMode = true; }
        else if (arg == "--no-grid4-spatial")              { enableGrid4Spatial = false; manualSpatialMode = true; }
        else if (arg == "--grid4-multiscale")              { grid4MsOpt.enabled = true; manualGrid4MultiscaleMode = true; }
        else if (arg == "--no-grid4-multiscale")           { grid4MsOpt.enabled = false; manualGrid4MultiscaleMode = true; }
        else if (arg.rfind("--grid4-ms-fine-dp-eps=", 0) == 0) grid4MsOpt.fineResidualDpEps = std::stod(arg.substr(23));
        else if (arg.rfind("--grid4-ms-fine-cutoff=", 0) == 0) grid4MsOpt.fineResidualCutoff = std::stod(arg.substr(24));
        else if (arg.rfind("--grid4-ms-fine-cutoff-band=", 0) == 0) grid4MsOpt.fineResidualCutoffBand = std::stod(arg.substr(29));
        else if (arg == "--grid4-ms-fine-no-cutoff-protect") grid4MsOpt.fineResidualCutoffProtect = false;
        else if (arg == "--auto-grid4-spatial")            autoGrid4Spatial = true;
        else if (arg == "--auto-spatial-routing")          autoSpatialRouting = true;
        else if (arg == "--auto-policy-search")            autoPolicyOpt.enabled = true;
        else if (arg.rfind("--auto-policy-step-x=", 0) == 0) autoPolicyOpt.sampleStepX = std::max(1, std::atoi(arg.substr(21).c_str()));
        else if (arg.rfind("--auto-policy-step-y=", 0) == 0) autoPolicyOpt.sampleStepY = std::max(1, std::atoi(arg.substr(21).c_str()));
        else if (arg.rfind("--auto-policy-step-z=", 0) == 0) autoPolicyOpt.sampleStepZ = std::max(1, std::atoi(arg.substr(21).c_str()));
        else if (arg.rfind("--auto-policy-step-t=", 0) == 0) autoPolicyOpt.sampleStepT = std::max(1, std::atoi(arg.substr(21).c_str()));
        else if (arg == "--poly11-spatial")                { enablePoly11Spatial = true; manualSpatialMode = true; }
        else if (arg == "--no-poly11-spatial")             { enablePoly11Spatial = false; manualSpatialMode = true; }
        else if (arg == "--grid4-control-only-temporal")   grid4EncoderOpt.controlOnlyTemporal = true;
        else if (arg == "--no-grid4-control-only-temporal") grid4EncoderOpt.controlOnlyTemporal = false;
        else if (arg == "--validate-fallback")             enableValidateFallback = true;
        else if (arg == "--no-validate-fallback")          enableValidateFallback = false;
        else if (arg == "--hotspot-second-pass")           enableHotspotSecondPass = true;
        else if (arg == "--no-hotspot-second-pass")        enableHotspotSecondPass = false;
        else if (arg == "--analyze-leaf-errors")           leafAnalysis.enabled = true;
        else if (arg.substr(0, 18) == "--leaf-report-top=") leafAnalysis.topK = std::max(1, std::atoi(arg.substr(18).c_str()));
        else if (arg.substr(0, 16) == "--spatial-probe=") spatialProbe.mode = arg.substr(16);
        else if (arg.substr(0, 12) == "--probe-top=")     spatialProbe.topK = std::max(1, std::atoi(arg.substr(12).c_str()));
        else if (arg.substr(0, 17) == "--probe-leaf-csv=") spatialProbe.leafCsv = arg.substr(17);
        else if (arg.substr(0, 12) == "--probe-thr=")     spatialProbe.threshold = std::atof(arg.substr(12).c_str());
        else if (arg.substr(0, 20) == "--probe-base-factor=") spatialProbe.adaptiveBaseFactor = std::atof(arg.substr(20).c_str());
        else if (arg.substr(0, 23) == "--probe-fine8-trigger=") spatialProbe.adaptiveFine8Trigger = std::atof(arg.substr(23).c_str());
        else if (arg.substr(0, 17) == "--probe-score-t1=") spatialProbe.scoreT1 = std::atof(arg.substr(17).c_str());
        else if (arg.substr(0, 17) == "--probe-score-t2=") spatialProbe.scoreT2 = std::atof(arg.substr(17).c_str());
        else if (arg.substr(0, 15) == "--probe-w-rmse=") spatialProbe.scoreWRmse = std::atof(arg.substr(15).c_str());
        else if (arg.substr(0, 14) == "--probe-w-p99=")  spatialProbe.scoreWP99 = std::atof(arg.substr(14).c_str());
        else if (arg.substr(0, 14) == "--probe-w-vis=")  spatialProbe.scoreWVis = std::atof(arg.substr(14).c_str());
        else if (arg.substr(0, 15) == "--probe-w-grad=") spatialProbe.scoreWGrad = std::atof(arg.substr(15).c_str());
        else if (arg.substr(0, 14) == "--probe-w-var=")  spatialProbe.scoreWVar = std::atof(arg.substr(14).c_str());
        else if (arg.substr(0, 22) == "--probe-fine8-improve=") spatialProbe.fine8Improve = std::atof(arg.substr(22).c_str());
        else if (arg.rfind("--grid4-ms-t1=", 0) == 0) grid4MsOpt.scoreT1 = std::atof(arg.substr(std::string("--grid4-ms-t1=").size()).c_str());
        else if (arg.rfind("--grid4-ms-t2=", 0) == 0) grid4MsOpt.scoreT2 = std::atof(arg.substr(std::string("--grid4-ms-t2=").size()).c_str());
        else if (arg.rfind("--grid4-ms-fine8-improve=", 0) == 0) grid4MsOpt.fine8Improve = std::atof(arg.substr(std::string("--grid4-ms-fine8-improve=").size()).c_str());
        else if (arg.rfind("--grid4-ms-vis-fine6=", 0) == 0) grid4MsOpt.visPromoteFine6 = std::atof(arg.substr(std::string("--grid4-ms-vis-fine6=").size()).c_str());
        else if (arg.rfind("--grid4-ms-vis-fine8=", 0) == 0) grid4MsOpt.visPromoteFine8 = std::atof(arg.substr(std::string("--grid4-ms-vis-fine8=").size()).c_str());
        else if (arg.rfind("--grid4-ms-vis-improve=", 0) == 0) grid4MsOpt.visFine8Improve = std::atof(arg.substr(std::string("--grid4-ms-vis-improve=").size()).c_str());
        else if (arg.rfind("--grid4-ms-generic-rmse=", 0) == 0) grid4MsOpt.genericPromoteRmseNorm = std::atof(arg.substr(std::string("--grid4-ms-generic-rmse=").size()).c_str());
        else if (arg.rfind("--grid4-ms-generic-p99=", 0) == 0) grid4MsOpt.genericPromoteP99Norm = std::atof(arg.substr(std::string("--grid4-ms-generic-p99=").size()).c_str());
        else if (arg.rfind("--grid4-ms-generic-f6=", 0) == 0) grid4MsOpt.genericFine6Improve = std::atof(arg.substr(std::string("--grid4-ms-generic-f6=").size()).c_str());
        else if (arg.rfind("--grid4-ms-generic-f8=", 0) == 0) grid4MsOpt.genericFine8Improve = std::atof(arg.substr(std::string("--grid4-ms-generic-f8=").size()).c_str());
        else if (arg.rfind("--grid4-ms-generic-kf=", 0) == 0) grid4MsOpt.genericFineCostKfPerCtrl = std::atof(arg.substr(std::string("--grid4-ms-generic-kf=").size()).c_str());
        else if (arg.rfind("--grid4-ms-generic-cost-improve=", 0) == 0) grid4MsOpt.genericFineCostImprove = std::atof(arg.substr(std::string("--grid4-ms-generic-cost-improve=").size()).c_str());
        else if (arg.rfind("--grid4-ms-generic-budget-frac=", 0) == 0) grid4MsOpt.genericBudgetFraction = std::atof(arg.substr(std::string("--grid4-ms-generic-budget-frac=").size()).c_str());
        else if (arg.rfind("--grid4-ms-generic-topk=", 0) == 0) grid4MsOpt.genericBudgetTopK = std::atoi(arg.substr(std::string("--grid4-ms-generic-topk=").size()).c_str());
        else if (arg.rfind("--grid4-ms-w-rmse=", 0) == 0) grid4MsOpt.scoreWRmse = std::atof(arg.substr(std::string("--grid4-ms-w-rmse=").size()).c_str());
        else if (arg.rfind("--grid4-ms-w-p99=", 0) == 0) grid4MsOpt.scoreWP99 = std::atof(arg.substr(std::string("--grid4-ms-w-p99=").size()).c_str());
        else if (arg.rfind("--grid4-ms-w-vis=", 0) == 0) grid4MsOpt.scoreWVis = std::atof(arg.substr(std::string("--grid4-ms-w-vis=").size()).c_str());
        else if (arg.rfind("--grid4-ms-w-grad=", 0) == 0) grid4MsOpt.scoreWGrad = std::atof(arg.substr(std::string("--grid4-ms-w-grad=").size()).c_str());
        else if (arg.rfind("--grid4-ms-w-var=", 0) == 0) grid4MsOpt.scoreWVar = std::atof(arg.substr(std::string("--grid4-ms-w-var=").size()).c_str());
        else if (arg.rfind("--grid4-ms-rmse-rel=", 0) == 0) grid4MsOpt.scoreRmseTargetRel = std::atof(arg.substr(std::string("--grid4-ms-rmse-rel=").size()).c_str());
        else if (arg.rfind("--grid4-ms-fine-band-factor=", 0) == 0) grid4MsOpt.fineCtrlBandFactor = std::atof(arg.substr(std::string("--grid4-ms-fine-band-factor=").size()).c_str());
        else if (arg.rfind("--grid4-ms-fine-keep-rel=", 0) == 0) grid4MsOpt.fineCtrlKeepRel = std::atof(arg.substr(std::string("--grid4-ms-fine-keep-rel=").size()).c_str());
        else if (arg.substr(0, 21) == "--probe-export-frame=") spatialProbe.exportFrame = std::atoi(arg.substr(21).c_str());
        else if (arg.substr(0, 19) == "--probe-export-dir=") spatialProbe.exportDir = arg.substr(19);
        else if (arg.substr(0, 20) == "--probe-export-name=") spatialProbe.exportName = arg.substr(20);
        else if (arg.substr(0, 23) == "--grid4-dense-hotspots=") { residualOpt.grid4DenseHotspots = std::max(0, std::atoi(arg.substr(23).c_str())); manualResidualMode = true; }
        else if (arg.substr(0, 26) == "--grid4-residual-hotspots=") { residualOpt.grid4ResidualHotspots = std::max(0, std::atoi(arg.substr(26).c_str())); manualResidualMode = true; }
        else if (arg.substr(0, 30) == "--grid4-residual-region-seeds=") { residualOpt.grid4ResidualRegionSeeds = std::max(0, std::atoi(arg.substr(30).c_str())); manualResidualMode = true; }
        else if (arg.substr(0, 31) == "--grid4-residual-region-radius=") { residualOpt.grid4ResidualRegionRadius = std::max(0, std::atoi(arg.substr(31).c_str())); manualResidualMode = true; }
        else if (arg.substr(0, 29) == "--grid4-residual-leaf-sample=") { residualOpt.grid4ResidualLeafSampleStep = std::max(1, std::atoi(arg.substr(29).c_str())); manualResidualMode = true; }
        else if (arg.substr(0, 21) == "--grid4-residual-thr=") { residualOpt.grid4ResidualThr = std::atof(arg.substr(21).c_str()); manualResidualMode = true; }
        else if (arg.substr(0, 25) == "--grid4-residual-rel-thr=") { residualOpt.grid4ResidualRelThr = std::atof(arg.substr(25).c_str()); manualResidualMode = true; }
        else if (arg.substr(0, 28) == "--grid4-residual-local-floor=") { residualOpt.grid4ResidualLocalFloor = std::atof(arg.substr(28).c_str()); manualResidualMode = true; }
        else if (arg.rfind("--grid4-residual-band-factor=", 0) == 0) { residualOpt.grid4ResidualBandFactor = std::atof(arg.substr(std::string("--grid4-residual-band-factor=").size()).c_str()); manualResidualMode = true; }
        else if (arg.rfind("--grid4-residual-keep-rel=", 0) == 0) { residualOpt.grid4ResidualKeepRel = std::atof(arg.substr(std::string("--grid4-residual-keep-rel=").size()).c_str()); manualResidualMode = true; }
        else if (arg == "--grid4-residual-rank-norm") { residualOpt.grid4ResidualRankNormalized = true; manualResidualMode = true; }
        else if (arg.substr(0, 20) == "--grid4-residual-dp=") { residualOpt.grid4ResidualDpEps = std::atof(arg.substr(20).c_str()); manualResidualMode = true; }
        else if (arg.substr(0, 24) == "--poly11-dense-hotspots=") { residualOpt.poly11DenseHotspots = std::max(0, std::atoi(arg.substr(24).c_str())); manualResidualMode = true; }
        else if (arg.substr(0, 27) == "--poly11-residual-hotspots=") { residualOpt.poly11ResidualHotspots = std::max(0, std::atoi(arg.substr(27).c_str())); manualResidualMode = true; }
        else if (arg.substr(0, 22) == "--poly11-residual-thr=") { residualOpt.poly11ResidualThr = std::atof(arg.substr(22).c_str()); manualResidualMode = true; }
        else if (arg.substr(0, 21) == "--poly11-residual-dp=") { residualOpt.poly11ResidualDpEps = std::atof(arg.substr(21).c_str()); manualResidualMode = true; }
        else if (arg == "--profile=sdf")                   profileOpt.name = "SDF";
        else if (arg == "--profile=density")               profileOpt.name = "DENSITY";
        else if (arg == "--profile=generic")               profileOpt.name = "GENERIC";
        else if (arg == "--disable-elision")               backgroundElisionOverride = 0;
        else if (arg == "--enable-elision")                backgroundElisionOverride = 1;
        else if (arg.substr(0, 10) == "--eps-abs=")        { profileOpt.epsAbs = eqval(10); epsAbsSpecified = true; }
        else if (arg.substr(0, 10) == "--eps-rel=")        profileOpt.epsRel = eqval(10);
        else if (arg.substr(0, 14) == "--gamma-delta=")    { profileOpt.gammaDelta = eqval(14); gammaSpecified = true; }
        else if (arg.substr(0, 11) == "--base-eps=")       profileOpt.baseEps = eqval(11);
        else if (arg.substr(0, 16) == "--render-cutoff=")  profileOpt.renderCutoff = eqval(16);
        else if (arg.substr(0, 14) == "--cutoff-band=")    profileOpt.cutoffBand = eqval(14);
        else if (arg.substr(0, 16) == "--bg-zero-ratio=")  profileOpt.bgZeroRatio = eqval(16);
        else if (arg.substr(0, 17) == "--bg-const-ratio=") profileOpt.bgConstRatio = eqval(17);
        else if (arg == "--cutoff-aware")                  { profileOpt.cutoffTemporalProtect = true; profileOpt.cutoffClusterProtect = true; }
        else if (arg == "--cutoff-temporal-protect")       profileOpt.cutoffTemporalProtect = true;
        else if (arg == "--no-cutoff-temporal-protect")    profileOpt.cutoffTemporalProtect = false;
        else if (arg == "--cutoff-cluster-protect")        profileOpt.cutoffClusterProtect = true;
        else if (arg == "--no-cutoff-cluster-protect")     profileOpt.cutoffClusterProtect = false;
        else if (arg.substr(0, 6)  == "--iso=")            profileOpt.iso = eqval(6);
        else if (arg.substr(0, 8)  == "--wband=")          profileOpt.wband = eqval(8);
        else if (arg.substr(0, 13) == "--near-band=")      profileOpt.nearBand = eqval(13);
        else if (arg.substr(0, 17) == "--critical-band=")  profileOpt.criticalBand = eqval(17);
        else if (arg.substr(0, 11) == "--eps-near=")       profileOpt.epsNear = eqval(11);
        else if (arg.substr(0, 10) == "--eps-far=")        profileOpt.epsFar = eqval(10);
        else if (arg.substr(0, 15) == "--eps-critical=")   profileOpt.epsCritical = eqval(15);
        else if (arg == "--time-only")                     timeOnly = true;
        else if (arg == "--skip-full-sweep")               skipFullSweep = true;
        else if (arg == "--sampled-full-sweep")            sampledFullSweep = true;
        else if (arg.substr(0, 16) == "--sample-step-x=") sampledStepX = std::max(1, eqint(16));
        else if (arg.substr(0, 16) == "--sample-step-y=") sampledStepY = std::max(1, eqint(16));
        else if (arg.substr(0, 16) == "--sample-step-z=") sampledStepZ = std::max(1, eqint(16));
        else if (arg.substr(0, 16) == "--sample-step-t=") sampledStepT = std::max(1, eqint(16));
        else if (arg == "--calibrate")                     calib.enabled = true;
        else if (arg.substr(0, 14) == "--calib-width=")    calib.subW = eqint(14);
        else if (arg.substr(0, 15) == "--calib-height=")   calib.subH = eqint(15);
        else if (arg.substr(0, 14) == "--calib-depth=")    calib.subD = eqint(14);
        else if (arg.substr(0, 15) == "--calib-frames=")   calib.subF = eqint(15);
        else if (arg.substr(0, 15) == "--calib-blocks=")   calib.numBlocks = std::max(1, eqint(15));
        else if (arg.substr(0, 16) == "--calib-start-x=")  calib.startX = eqint(16);
        else if (arg.substr(0, 16) == "--calib-start-y=")  calib.startY = eqint(16);
        else if (arg.substr(0, 16) == "--calib-start-z=")  calib.startZ = eqint(16);
        else if (arg.substr(0, 16) == "--calib-start-t=")  calib.startT = eqint(16);
        else if (arg.substr(0, 14) == "--target-p99=")     calib.targetP99 = eqval(14);
        else if (arg.substr(0, 15) == "--target-p999=")    calib.targetP999 = eqval(15);
        else if (arg.substr(0, 14) == "--target-rmse=")    calib.targetRmse = eqval(14);
        else if (arg.substr(0, 14) == "--target-psnr=")    calib.targetPsnr = eqval(14);
        else if (arg.substr(0, 8)  == "--width=")          hintW = eqint(8);
        else if (arg.substr(0, 9)  == "--height=")         hintH = eqint(9);
        else if (arg.substr(0, 8)  == "--depth=")          hintD = eqint(8);
        else if (arg.substr(0, 9)  == "--frames=")         hintF = eqint(9);
        else if (arg.substr(0, 2)  != "--")                inputFile = arg;
    }
    const bool useDataRangeForPsnr = (profileOpt.name != "SDF");

    std::string stem = inputFile;
    {
        auto s = stem.rfind('/');
        if (s != std::string::npos) stem = stem.substr(s + 1);
        auto b = stem.rfind('\\');
        if (b != std::string::npos) stem = stem.substr(b + 1);
        auto d = stem.rfind('.');
        if (d != std::string::npos) stem = stem.substr(0, d);
    }

    std::error_code mkdirEc;
    std::filesystem::create_directories("results/vbt", mkdirEc);
    const std::string outVBT = "results/vbt/" + stem + ".vbt";

    std::cout << "\nStep 1: Loading volume..." << std::endl;
    int width = 0, height = 0, depth = 0, frames = 0;
    RawDataFormat detectedFormat = RawDataFormat::UNKNOWN;
    float dataMin = 0.0f;
    float dataMax = 0.0f;
    Volume4D volumeSequence;
    if (hintW <= 0 || hintH <= 0 || hintD <= 0 || hintF <= 0) {
        const auto metaPath = guessMetadataPathForRaw(inputFile);
        if (!metaPath.empty()) {
            try {
                const auto meta = vdbtools::loadFrameMetadata(metaPath);
                if (hintW <= 0) hintW = meta.width;
                if (hintH <= 0) hintH = meta.height;
                if (hintD <= 0) hintD = meta.depth;
                if (hintF <= 0) hintF = meta.frames;
                std::cout << "  Metadata hint: " << metaPath.string()
                          << " -> " << meta.width << "x" << meta.height << "x" << meta.depth
                          << " frames=" << meta.frames << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Warning: failed to load metadata hint: " << e.what() << std::endl;
            }
        }
    }
    if (!loadRawVolume(inputFile, width, height, depth, frames,
                       volumeSequence, detectedFormat, dataMin, dataMax,
                       hintW, hintH, hintD, hintF)) {
        std::cerr << "Loading failed!" << std::endl;
        return 1;
    }

    const double dataRange = std::max(1e-6f, dataMax - dataMin);
    const SampledFieldStats sampledStats = (profileOpt.name != "SDF")
        ? computeSampledFieldStats(volumeSequence, width, height, depth, frames)
        : SampledFieldStats{};
    if (profileOpt.name != "SDF") {
        if (!epsAbsSpecified) {
            profileOpt.epsAbs = chooseAdaptiveScalarEpsAbs(sampledStats, dataRange);
        }
        if (!clusterSpecified) {
            v5ThreshCluster = chooseAdaptiveScalarClusterThr(sampledStats);
        }
        if (!gammaSpecified) {
            const double normalizedDelta = sampledStats.deltaP90 / std::max(1e-6, sampledStats.robustRange);
            profileOpt.gammaDelta = static_cast<float>(std::clamp(0.05 + 0.60 * normalizedDelta, 0.05, 0.25));
        }
    }
    FieldProfile fieldProfile = makeFieldProfile(profileOpt);
    const bool enableBackgroundElision = (backgroundElisionOverride >= 0)
        ? (backgroundElisionOverride != 0)
        : fieldProfile.default_background_elision();

    const long long rawBytesOrig = static_cast<long long>(width) * height * depth * frames
                                 * (detectedFormat == RawDataFormat::UINT8 ? 1LL : 4LL);
    const long long rawBytesF32 = static_cast<long long>(width) * height * depth * frames * 4LL;
    printf("  %dx%dx%d  frames=%d  raw(%s)=%.2f MB  dataMin=%.4f  dataMax=%.4f\n",
           width, height, depth, frames,
           detectedFormat == RawDataFormat::UINT8 ? "U8" : "FP32",
           rawBytesOrig / (1024.0 * 1024.0), dataMin, dataMax);
    printf("  Profile: %s\n", profileOpt.name.c_str());
    if (profileOpt.name == "SDF") {
        printf("  SDF Bands: far=iso±%.1f  near=iso±%.1f  critical=iso±%.1f\n",
               profileOpt.wband, profileOpt.nearBand, profileOpt.criticalBand);
        printf("  SDF Eps  : far=%.3f  near=%.3f  critical=%.3f  iso=%.1f\n",
               profileOpt.epsFar, profileOpt.epsNear, profileOpt.epsCritical, profileOpt.iso);
    } else {
        printf("  Sampled Stats: q01=%.6f  q99=%.6f  robustRange=%.6f  tighterRange=%.6f\n",
               sampledStats.q01, sampledStats.q99, sampledStats.robustRange, sampledStats.tighterRange);
        printf("  Sampled Stats: temporalRange=%.6f (max(robustRange, 0.8 * fullRange))\n",
               chooseAdaptiveScalarTemporalRange(sampledStats, dataRange));
        printf("  Sampled Stats: deltaMean=%.6f  deltaP90=%.6f  deltaP99=%.6f  lag1=%.4f\n",
               sampledStats.deltaMean, sampledStats.deltaP90, sampledStats.deltaP99, sampledStats.lag1Autocorr);
        printf("  Sampled Stats: spatialMean=%.6f  spatialP90=%.6f  spatialP99=%.6f  sampleStep=%dx%dx%d t=%d\n",
               sampledStats.spatialMean, sampledStats.spatialP90, sampledStats.spatialP99,
               sampledStats.stepX, sampledStats.stepY, sampledStats.stepZ, sampledStats.stepT);
        printf("  Scalar Defaults: eps_abs=%.6f  gamma_delta=%.3f  cluster_thr=%.3f\n",
               profileOpt.epsAbs, profileOpt.gammaDelta, v5ThreshCluster);
        if (profileOpt.name == "DENSITY" && profileOpt.renderCutoff > 0.0f) {
            printf("  Smoke Background: bg_zero_ratio=%.3f  bg_const_ratio=%.3f  cutoff=%.6f  band=%.6f\n",
                   profileOpt.bgZeroRatio, profileOpt.bgConstRatio,
                   profileOpt.renderCutoff, profileOpt.cutoffBand);
        }
        if (!epsAbsSpecified) {
            printf("  [AutoDefault] temporal eps_abs <- 6/255 * max(robustRange(q01..q99), 0.8 * fullRange)\n");
        }
        if (!gammaSpecified) {
            printf("  [AutoDefault] gamma_delta <- clamp(0.05 + 0.60 * deltaP90/robustRange, 0.05, 0.25)\n");
        }
        if (!clusterSpecified) {
            printf("  [AutoDefault] spatial cluster_thr <- (8/6) * spatialSmoothnessScale(spatialP90/robustRange)\n");
        }
    }
    printf("  Background elision: %s", enableBackgroundElision ? "enabled" : "disabled");
    if (backgroundElisionOverride >= 0) {
        printf(" (CLI override)\n");
    } else if (!fieldProfile.default_background_elision()) {
        printf(" (profile default for GENERIC)\n");
    } else {
        printf(" (profile default)\n");
    }
    printf("  Block-aware cluster: %s\n", enableBlockAwareCluster ? "enabled" : "disabled");
    printf("  Budget-aware cluster: %s\n", enableBudgetAwareCluster ? "enabled" : "disabled");
    printf("  Guarded medoid     : %s\n", enableGuardedMedoidCluster ? "enabled" : "disabled");
    printf("  GRID4 spatial      : %s\n", enableGrid4Spatial ? "enabled" : "disabled");
    printf("  GRID4 multiscale   : %s\n", grid4MsOpt.enabled ? "enabled" : "disabled");
    printf("  GRID4 encoder      : %s\n", grid4EncoderOpt.controlOnlyTemporal ? "control-only temporal (experimental)" : "full-voxel temporal");
    printf("  POLY11 spatial     : %s\n", enablePoly11Spatial ? "enabled" : "disabled");
    printf("  Validate fallback : %s\n", enableValidateFallback ? "enabled" : "disabled");
    printf("  Hotspot second pass: %s\n", enableHotspotSecondPass ? "enabled" : "disabled");
    printf("  Leaf hotspot report: %s", leafAnalysis.enabled ? "enabled" : "disabled");
    if (leafAnalysis.enabled) printf("  (top=%d)", leafAnalysis.topK);
    printf("\n");
    if (!spatialProbe.mode.empty()) {
        printf("  Spatial probe      : %s  (top=%d", spatialProbe.mode.c_str(), spatialProbe.topK);
        if (spatialProbe.threshold > 0.0) printf("  thr=%.6f", spatialProbe.threshold);
        if (spatialProbe.adaptiveBaseFactor > 0.0) printf("  baseFactor=%.3f", spatialProbe.adaptiveBaseFactor);
        if (spatialProbe.adaptiveFine8Trigger > 0.0) printf("  fine8Trigger=%.3f", spatialProbe.adaptiveFine8Trigger);
        if (spatialProbe.scoreT1 > 0.0) printf("  T1=%.3f", spatialProbe.scoreT1);
        if (spatialProbe.scoreT2 > 0.0) printf("  T2=%.3f", spatialProbe.scoreT2);
        if (spatialProbe.fine8Improve > 0.0) printf("  fine8Improve=%.3f", spatialProbe.fine8Improve);
        if (spatialProbe.exportFrame >= 0) printf("  exportFrame=%d", spatialProbe.exportFrame);
        printf(")\n");
    }
    if (autoSpatialRouting) {
        printf("  Auto routing       : enabled\n");
    }
    if (autoPolicyOpt.enabled) {
        printf("  Auto policy search : enabled  (sampleStep=%d,%d,%d,%d)\n",
               autoPolicyOpt.sampleStepX, autoPolicyOpt.sampleStepY,
               autoPolicyOpt.sampleStepZ, autoPolicyOpt.sampleStepT);
    }
    if (residualOpt.grid4DenseHotspots > 0) {
        printf("  GRID4 dense hotfix : top=%d\n", residualOpt.grid4DenseHotspots);
    }
    if (residualOpt.grid4ResidualHotspots > 0) {
        printf("  GRID4 residual     : top=%d", residualOpt.grid4ResidualHotspots);
        if (residualOpt.grid4ResidualRegionSeeds > 0) {
            printf("  regionSeeds=%d  radius=%d",
                   residualOpt.grid4ResidualRegionSeeds,
                   residualOpt.grid4ResidualRegionRadius);
        }
        if (residualOpt.grid4ResidualLeafSampleStep > 1) {
            printf("  leafSample=%d", residualOpt.grid4ResidualLeafSampleStep);
        }
        if (residualOpt.grid4ResidualThr > 0.0) printf("  thr=%.6f", residualOpt.grid4ResidualThr);
        if (residualOpt.grid4ResidualRelThr > 0.0) printf("  relThr=%.3f", residualOpt.grid4ResidualRelThr);
        if (residualOpt.grid4ResidualLocalFloor > 0.0) printf("  localFloor=%.6f", residualOpt.grid4ResidualLocalFloor);
        if (residualOpt.grid4ResidualBandFactor > 0.0) printf("  bandFactor=%.3f", residualOpt.grid4ResidualBandFactor);
        if (residualOpt.grid4ResidualKeepRel > 0.0) printf("  keepRel=%.3f", residualOpt.grid4ResidualKeepRel);
        if (residualOpt.grid4ResidualRankNormalized) printf("  rank=norm");
        printf("  dp=%.2f", residualOpt.grid4ResidualDpEps);
        printf("\n");
    }
    if (residualOpt.poly11DenseHotspots > 0) {
        printf("  POLY11 dense hotfix: top=%d\n", residualOpt.poly11DenseHotspots);
    }
    if (residualOpt.poly11ResidualHotspots > 0) {
        printf("  POLY11 residual    : top=%d", residualOpt.poly11ResidualHotspots);
        if (residualOpt.poly11ResidualThr > 0.0) printf("  thr=%.6f", residualOpt.poly11ResidualThr);
        printf("  dp=%.2f", residualOpt.poly11ResidualDpEps);
        printf("\n");
    }

    if (calib.enabled) {
        int rc = runCalibration(volumeSequence, width, height, depth, frames,
                                profileOpt, enableBackgroundElision, enableBlockAwareCluster, enableBudgetAwareCluster,
                                enableGuardedMedoidCluster, enableGrid4Spatial, enableHotspotSecondPass, v5ThreshCluster,
                                calib, dataMin, dataMax, useDataRangeForPsnr);
        if (rc != 0) return rc;
        std::cout << "\n========== Calibration Done ==========" << std::endl;
        return 0;
    }

    std::cout << "\nStep 2: Temporal compression (weighted DP keyframes, float values)..." << std::endl;
    long long totalOrig = 0;
    long long totalKF = 0;
    CompressedVolume4D compressedVolume =
        (grid4EncoderOpt.controlOnlyTemporal && enableGrid4Spatial)
        ? temporalCompressGrid4ControlsOnly(
            volumeSequence, width, height, depth, frames, fieldProfile, totalOrig, totalKF, true)
        : temporalCompress(
            volumeSequence, width, height, depth, frames, fieldProfile, totalOrig, totalKF, true);
    printf("  KF ratio: %.2f%%  (%lld -> %lld)\n", 100.0 * totalKF / std::max(1LL, totalOrig), totalOrig, totalKF);

    if (spatialProbe.mode == "grid4") {
        int rc = runGrid4HotspotProbe(inputFile, volumeSequence, compressedVolume,
                                      width, height, depth, frames, spatialProbe);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }
    if (spatialProbe.mode == "grid4-full") {
        int rc = runGrid4SampledFullProbe(inputFile, volumeSequence, compressedVolume,
                                          width, height, depth, frames,
                                          fieldProfile, enableBackgroundElision,
                                          enableBlockAwareCluster, enableBudgetAwareCluster,
                                          enableGuardedMedoidCluster, enableValidateFallback,
                                          enableHotspotSecondPass, v5ThreshCluster,
                                          dataMin, dataMax, useDataRangeForPsnr);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }
    if (spatialProbe.mode == "grid4-fine6-full" || spatialProbe.mode == "grid4f6") {
        int rc = runGrid4FineSampledFullProbe(inputFile, volumeSequence, compressedVolume,
                                              width, height, depth, frames,
                                              fieldProfile, enableBackgroundElision,
                                              enableBlockAwareCluster, enableBudgetAwareCluster,
                                              enableGuardedMedoidCluster, enableValidateFallback,
                                              enableHotspotSecondPass, v5ThreshCluster,
                                              dataMin, dataMax, useDataRangeForPsnr, spatialProbe,
                                              6, false);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }
    if (spatialProbe.mode == "grid4-fine8-full" || spatialProbe.mode == "grid4f8") {
        int rc = runGrid4FineSampledFullProbe(inputFile, volumeSequence, compressedVolume,
                                              width, height, depth, frames,
                                              fieldProfile, enableBackgroundElision,
                                              enableBlockAwareCluster, enableBudgetAwareCluster,
                                              enableGuardedMedoidCluster, enableValidateFallback,
                                              enableHotspotSecondPass, v5ThreshCluster,
                                              dataMin, dataMax, useDataRangeForPsnr, spatialProbe,
                                              8, false);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }
    if (spatialProbe.mode == "grid4-fine6-hf-full" || spatialProbe.mode == "grid4f6hf") {
        int rc = runGrid4FineSampledFullProbe(inputFile, volumeSequence, compressedVolume,
                                              width, height, depth, frames,
                                              fieldProfile, enableBackgroundElision,
                                              enableBlockAwareCluster, enableBudgetAwareCluster,
                                              enableGuardedMedoidCluster, enableValidateFallback,
                                              enableHotspotSecondPass, v5ThreshCluster,
                                              dataMin, dataMax, useDataRangeForPsnr, spatialProbe,
                                              6, true);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }
    if (spatialProbe.mode == "grid4-fine8-hf-full" || spatialProbe.mode == "grid4f8hf") {
        int rc = runGrid4FineSampledFullProbe(inputFile, volumeSequence, compressedVolume,
                                              width, height, depth, frames,
                                              fieldProfile, enableBackgroundElision,
                                              enableBlockAwareCluster, enableBudgetAwareCluster,
                                              enableGuardedMedoidCluster, enableValidateFallback,
                                              enableHotspotSecondPass, v5ThreshCluster,
                                              dataMin, dataMax, useDataRangeForPsnr, spatialProbe,
                                              8, true);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }
    if (spatialProbe.mode == "grid4-fine-adaptive-full" || spatialProbe.mode == "grid4fa") {
        int rc = runGrid4FineAdaptiveSampledFullProbe(inputFile, volumeSequence, compressedVolume,
                                                      width, height, depth, frames,
                                                      fieldProfile, enableBackgroundElision,
                                                      enableBlockAwareCluster, enableBudgetAwareCluster,
                                                      enableGuardedMedoidCluster, enableValidateFallback,
                                                      enableHotspotSecondPass, v5ThreshCluster,
                                                      dataMin, dataMax, useDataRangeForPsnr, spatialProbe);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }
    if (spatialProbe.mode == "grid4-fine-adaptive-score-full" || spatialProbe.mode == "grid4fas") {
        int rc = runGrid4FineAdaptiveScoreSampledFullProbe(inputFile, volumeSequence, compressedVolume,
                                                           width, height, depth, frames,
                                                           fieldProfile, enableBackgroundElision,
                                                           enableBlockAwareCluster, enableBudgetAwareCluster,
                                                           enableGuardedMedoidCluster, enableValidateFallback,
                                                           enableHotspotSecondPass, v5ThreshCluster,
                                                           dataMin, dataMax, useDataRangeForPsnr, spatialProbe);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }
    if (spatialProbe.mode == "grid4-spatialfirst-full" || spatialProbe.mode == "grid4sf") {
        int rc = runGrid4SpatialFirstSampledFullProbe(inputFile, volumeSequence, compressedVolume,
                                                      width, height, depth, frames,
                                                      fieldProfile, dataMin, dataMax,
                                                      useDataRangeForPsnr, spatialProbe, false);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }
    if (spatialProbe.mode == "grid4-spatialfirst-weighted-full" || spatialProbe.mode == "grid4sfw") {
        int rc = runGrid4SpatialFirstSampledFullProbe(inputFile, volumeSequence, compressedVolume,
                                                      width, height, depth, frames,
                                                      fieldProfile, dataMin, dataMax,
                                                      useDataRangeForPsnr, spatialProbe, true);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }
    if (spatialProbe.mode == "poly8-full" || spatialProbe.mode == "poly8f") {
        int rc = runPoly8SampledFullProbe(inputFile, volumeSequence, compressedVolume,
                                          width, height, depth, frames,
                                          dataMin, dataMax, useDataRangeForPsnr);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }
    if (spatialProbe.mode == "poly11-full" || spatialProbe.mode == "poly11f") {
        int rc = runPoly11SampledFullProbe(inputFile, volumeSequence, compressedVolume,
                                           width, height, depth, frames,
                                           dataMin, dataMax, useDataRangeForPsnr);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }
    if (spatialProbe.mode == "poly11-residual-full" || spatialProbe.mode == "poly11rf") {
        int rc = runPoly11ResidualSampledFullProbe(inputFile, volumeSequence, compressedVolume,
                                                   width, height, depth, frames,
                                                   dataMin, dataMax, useDataRangeForPsnr, spatialProbe);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }
    if (spatialProbe.mode == "grid4-hybrid-full") {
        int rc = runGrid4HybridSampledFullProbe(inputFile, volumeSequence, compressedVolume,
                                                width, height, depth, frames,
                                                fieldProfile, enableBackgroundElision,
                                                enableBlockAwareCluster, enableBudgetAwareCluster,
                                                enableGuardedMedoidCluster, enableValidateFallback,
                                                enableHotspotSecondPass, v5ThreshCluster,
                                                dataMin, dataMax, useDataRangeForPsnr, spatialProbe);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }
    if (spatialProbe.mode == "grid4-residual-full" || spatialProbe.mode == "grid4rf") {
        int rc = runGrid4ResidualSampledFullProbe(inputFile, volumeSequence, compressedVolume,
                                                  width, height, depth, frames,
                                                  fieldProfile, enableBackgroundElision,
                                                  enableBlockAwareCluster, enableBudgetAwareCluster,
                                                  enableGuardedMedoidCluster, enableValidateFallback,
                                                  enableHotspotSecondPass, v5ThreshCluster,
                                                  dataMin, dataMax, useDataRangeForPsnr, spatialProbe);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }
    if (spatialProbe.mode == "grid4-residual-profile" || spatialProbe.mode == "grid4rp") {
        int rc = runGrid4ResidualProfiler(inputFile, volumeSequence, compressedVolume,
                                          width, height, depth, frames,
                                          fieldProfile, enableBackgroundElision,
                                          enableBlockAwareCluster, enableBudgetAwareCluster,
                                          enableGuardedMedoidCluster, enableValidateFallback,
                                          enableHotspotSecondPass, v5ThreshCluster,
                                          dataMin, dataMax, spatialProbe);
        std::cout << "\n========== Spatial Probe Done ==========" << std::endl;
        return rc;
    }

    double temporalMeanMeasured = -1.0;
    double temporalMaxMeasured = -1.0;
    double temporalPsnrMeasured = -1.0;
    double temporalRmseMeasured = -1.0;
    AutoSpatialRouteDecision autoRouteDecision;
    bool timeOnlyDeferredExit = false;

    if (grid4EncoderOpt.controlOnlyTemporal && enableGrid4Spatial) {
        std::cout << "\nStep 3: Temporal reconstruction error (RMSE/PSNR)..." << std::endl;
        printf("  [Skip] GRID4 control-only temporal mode does not build voxel-wise temporal KFs for all voxels.\n");
        if (timeOnly) {
            timeOnlyDeferredExit = true;
        }
    } else {
        std::cout << "\nStep 3: Temporal reconstruction error (RMSE/PSNR)..." << std::endl;
        {
        const long long N = static_cast<long long>(width) * height * depth * frames;
        double errSum2 = 0.0;
        double errSum = 0.0;
        float errMax = 0.0f;

        #ifdef _OPENMP
        #pragma omp parallel
        {
            double localErrSum = 0.0;
            double localErrSum2 = 0.0;
            float localErrMax = 0.0f;

            #pragma omp for collapse(3) nowait
            for (int z = 0; z < depth; ++z) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        const auto& kfs = compressedVolume[z][y][x];
                        for (int t = 0; t < frames; ++t) {
                            float orig = volumeSequence[t][z][y][x];
                            float recon = sampleTemporalKfs(kfs, t);
                            float e = std::abs(orig - recon);
                            localErrSum += e;
                            localErrSum2 += static_cast<double>(e) * e;
                            localErrMax = std::max(localErrMax, e);
                        }
                    }
                }
            }

            #pragma omp critical
            {
                errSum += localErrSum;
                errSum2 += localErrSum2;
                errMax = std::max(errMax, localErrMax);
            }
        }
        #else
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const auto& kfs = compressedVolume[z][y][x];
                    for (int t = 0; t < frames; ++t) {
                        float orig = volumeSequence[t][z][y][x];
                        float recon = sampleTemporalKfs(kfs, t);
                        float e = std::abs(orig - recon);
                        errSum += e;
                        errSum2 += static_cast<double>(e) * e;
                        errMax = std::max(errMax, e);
                    }
                }
            }
        }
        #endif

        const double rmse = std::sqrt(errSum2 / N);
        temporalMeanMeasured = errSum / N;
        temporalMaxMeasured = errMax;
        temporalRmseMeasured = rmse;
        const double psnrPeak = useDataRangeForPsnr ? std::max(1e-6, static_cast<double>(dataMax) - dataMin) : 255.0;
        const double psnr = computePsnr(rmse, psnrPeak);
        temporalPsnrMeasured = psnr;
        printf("  Temporal DP - Mean=%.4f  Max=%.4f  RMSE=%.6f  PSNR=%.2f dB  (peak=%.6f)\n",
               errSum / N, errMax, rmse, psnr, psnrPeak);
        if (timeOnly) {
            timeOnlyDeferredExit = true;
        }
    }
    }

    const double avgKfPerVoxel = static_cast<double>(totalKF) / std::max(1LL, static_cast<long long>(width) * height * depth);
    const double spatialSmoothNorm = sampledStats.spatialP90 / std::max(1e-6, sampledStats.robustRange);
    const bool grid4Recommended = recommendGrid4Spatial(profileOpt.name, width, height, depth, totalKF, sampledStats);
    printf("  Spatial mode hint: %s  (avgKF/voxel=%.2f  lag1=%.3f  spatialP90/robust=%.3f)\n",
           grid4Recommended ? "GRID4-family candidate" : "legacy clustered candidate",
           avgKfPerVoxel, sampledStats.lag1Autocorr, spatialSmoothNorm);
    if (autoGrid4Spatial && !autoSpatialRouting) {
        enableGrid4Spatial = grid4Recommended;
        printf("  Auto GRID4 spatial : selected=%s\n", enableGrid4Spatial ? "GRID4-family" : "legacy clustered");
    }
    if (autoSpatialRouting && temporalRmseMeasured >= 0.0) {
        autoRouteDecision = recommendSpatialRoute(profileOpt.name, width, height, depth, totalKF, sampledStats,
                                                  temporalRmseMeasured, dataMin, dataMax);
        if (!manualSpatialMode) {
            enableGrid4Spatial = (autoRouteDecision.mode != AutoSpatialRouteMode::CLUSTERED);
            enablePoly11Spatial = false;
        }
        if (!manualGrid4MultiscaleMode) {
            grid4MsOpt.enabled = (autoRouteDecision.mode == AutoSpatialRouteMode::GRID4_MULTISCALE);
        }
        if (!manualResidualMode) {
            residualOpt.grid4DenseHotspots = 0;
            residualOpt.grid4ResidualRegionSeeds = 0;
            residualOpt.grid4ResidualRegionRadius = 1;
            residualOpt.grid4ResidualLeafSampleStep = 1;
            residualOpt.grid4ResidualThr = -1.0;
            residualOpt.grid4ResidualHotspots =
                (autoRouteDecision.mode == AutoSpatialRouteMode::GRID4_RESIDUAL) ? autoRouteDecision.residualTopK : 0;
            residualOpt.grid4ResidualDpEps = autoRouteDecision.residualDp;
        }
        const char* routeName =
            autoRouteDecision.mode == AutoSpatialRouteMode::CLUSTERED ? "legacy clustered-seq" :
            autoRouteDecision.mode == AutoSpatialRouteMode::GRID4 ? "GRID4-family base" :
            autoRouteDecision.mode == AutoSpatialRouteMode::GRID4_MULTISCALE ? "GRID4 + fine residual grid" :
            "GRID4 + sparse residual";
        printf("  Auto routing       : selected=%s  (temporalNRMSE=%.4f  reason=%s)\n",
               routeName, autoRouteDecision.temporalNormRmse, autoRouteDecision.reason);
        if (autoRouteDecision.temporalLimited) {
            printf("  Auto routing note  : temporal-limited dataset\n");
        }
    }

    if (autoPolicyOpt.enabled && profileOpt.name == "DENSITY" && enableGrid4Spatial) {
        std::cout << "\nStep 3.5: Auto policy search (pre-save)..." << std::endl;
        const int densityProxyFrame = (profileOpt.renderCutoff > 0.0f)
            ? chooseDensityProxyFrame(
                volumeSequence, width, height, depth, frames,
                profileOpt.renderCutoff,
                std::max(1, autoPolicyOpt.sampleStepX * 2),
                std::max(1, autoPolicyOpt.sampleStepY * 2),
                std::max(1, autoPolicyOpt.sampleStepZ))
            : -1;
        if (densityProxyFrame >= 0) {
            printf("  Render proxy frame : %d\n", densityProxyFrame);
        }
        const std::vector<SpatialPolicyCandidate> candidates = makeDensityPolicyCandidates(
            sampledStats, profileOpt, grid4MsOpt, residualOpt);
        std::vector<SpatialPolicyEvalResult> results;
        results.reserve(candidates.size());
        for (const auto& cand : candidates) {
            SpatialPolicyEvalResult row;
            row.candidate = cand;
            row.metrics = evaluateSpatialPolicyCandidate(
                volumeSequence, compressedVolume, width, height, depth, frames,
                fieldProfile, enableBackgroundElision, enableBlockAwareCluster, enableBudgetAwareCluster,
                enableGuardedMedoidCluster, true, enableValidateFallback, enableHotspotSecondPass,
                v5ThreshCluster, dataMin, dataMax, useDataRangeForPsnr,
                cand.grid4Ms, cand.residual, autoPolicyOpt, densityProxyFrame);
            results.push_back(row);
            printf("  Candidate %-22s : %8.2f KB  RMSE=%.6f  PSNR=%.2f  P99=%.6f  P99.9=%.6f  proxyPSNR=%.2f\n",
                   cand.name.c_str(),
                   row.metrics.vbtBytes / 1024.0,
                   row.metrics.rmse,
                   row.metrics.psnr,
                    row.metrics.p99,
                   row.metrics.p999,
                   row.metrics.renderProxyPsnr);
        }
        const int bestIdx = pickBestDensityPolicyCandidate(results, sampledStats);
        if (bestIdx >= 0) {
            const auto& best = results[static_cast<size_t>(bestIdx)];
            grid4MsOpt = best.candidate.grid4Ms;
            residualOpt = best.candidate.residual;
            enableGrid4Spatial = true;
            enablePoly11Spatial = false;
            printf("  Auto policy winner : %s  -> %.2f KB  sampledPSNR=%.2f\n",
                   best.candidate.name.c_str(),
                   best.metrics.vbtBytes / 1024.0,
                   best.metrics.psnr);
            printf("  Selected structure : %s\n",
                   best.candidate.mode == AutoSpatialRouteMode::GRID4_RESIDUAL
                       ? "GRID4 + selective sparse residual"
                       : "GRID4 + fine residual grid");
        }
    } else if (autoPolicyOpt.enabled && profileOpt.name == "GENERIC" && enableGrid4Spatial) {
        std::cout << "\nStep 3.5: Auto policy search (pre-save)..." << std::endl;
        const std::vector<SpatialPolicyCandidate> candidates = makeGenericPolicyCandidates(
            sampledStats, grid4MsOpt, residualOpt);
        std::vector<SpatialPolicyEvalResult> results;
        results.reserve(candidates.size());
        for (const auto& cand : candidates) {
            SpatialPolicyEvalResult row;
            row.candidate = cand;
            row.metrics = evaluateSpatialPolicyCandidate(
                volumeSequence, compressedVolume, width, height, depth, frames,
                fieldProfile, enableBackgroundElision, enableBlockAwareCluster, enableBudgetAwareCluster,
                enableGuardedMedoidCluster, true, enableValidateFallback, enableHotspotSecondPass,
                v5ThreshCluster, dataMin, dataMax, useDataRangeForPsnr,
                cand.grid4Ms, cand.residual, autoPolicyOpt, -1);
            results.push_back(row);
            printf("  Candidate %-22s : %8.2f KB  RMSE=%.6f  PSNR=%.2f  P99=%.6f  P99.9=%.6f\n",
                   cand.name.c_str(),
                   row.metrics.vbtBytes / 1024.0,
                   row.metrics.rmse,
                   row.metrics.psnr,
                   row.metrics.p99,
                   row.metrics.p999);
        }
        const int bestIdx = pickBestGenericPolicyCandidate(results, 30.0);
        if (bestIdx >= 0) {
            const auto& best = results[static_cast<size_t>(bestIdx)];
            grid4MsOpt = best.candidate.grid4Ms;
            residualOpt = best.candidate.residual;
            enableGrid4Spatial = true;
            enablePoly11Spatial = false;
            printf("  Auto policy winner : %s  -> %.2f KB  sampledPSNR=%.2f\n",
                   best.candidate.name.c_str(),
                   best.metrics.vbtBytes / 1024.0,
                   best.metrics.psnr);
            printf("  Selected structure : %s\n",
                   best.candidate.mode == AutoSpatialRouteMode::GRID4_RESIDUAL
                       ? "GRID4 + selective sparse residual"
                       : "GRID4-family base");
        }
    }

    if (timeOnlyDeferredExit) {
        printf("TIME_ONLY_SUMMARY kf_ratio=%.2f mean=%.4f max=%.4f rmse=%.6f psnr=%.2f eps_abs=%.6f eps_rel=%.4f gamma=%.3f cluster_thr=%.3f\n",
               100.0 * totalKF / std::max(1LL, totalOrig), temporalMeanMeasured, temporalMaxMeasured,
               temporalRmseMeasured, temporalPsnrMeasured, profileOpt.epsAbs, profileOpt.epsRel,
               profileOpt.gammaDelta, v5ThreshCluster);
        std::cout << "\n========== Time-Only Done ==========" << std::endl;
        return 0;
    }

    printf("\nStep 4: BlockTree build  (cluster_thr=%.1f  profile=%s  elision=%s  blockAware=%s  guardedMedoid=%s  grid4=%s)...\n",
           v5ThreshCluster, profileOpt.name.c_str(), enableBackgroundElision ? "on" : "off",
           enableBlockAwareCluster ? "on" : "off",
           enableGuardedMedoidCluster ? "on" : "off",
           enableGrid4Spatial ? "on" : "off");
    BlockTree btree;
    const auto buildStart = std::chrono::steady_clock::now();
    try {
        btree.build(compressedVolume, width, height, depth, frames,
                    -1.0, v5ThreshCluster, fieldProfile, enableBackgroundElision, enableBlockAwareCluster,
                    enableBudgetAwareCluster, enableGuardedMedoidCluster, enableGrid4Spatial,
                    grid4EncoderOpt.controlOnlyTemporal,
                    &volumeSequence, enableValidateFallback, enableHotspotSecondPass);
    } catch (const std::exception& ex) {
        std::cerr << "Build failed: " << ex.what() << std::endl;
        return 2;
    }
    const auto buildElapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - buildStart).count();
    printf("  Build Time: %.3f s\n", buildElapsed);
    if (enablePoly11Spatial) {
        std::cout << "  Convert leaves to POLY11 base..." << std::endl;
        applyPoly11SpatialAllLeaves(
            btree, compressedVolume, width, height, depth, frames, fieldProfile);
    }
    if (grid4MsOpt.enabled && enableGrid4Spatial) {
        std::cout << "  Convert GRID4 leaves to adaptive multiscale..." << std::endl;
        applyGrid4MultiscaleAdaptive(
            volumeSequence, compressedVolume, btree, width, height, depth, frames,
            fieldProfile, grid4MsOpt);
    }
    if (residualOpt.grid4DenseHotspots > 0 && enableGrid4Spatial) {
        std::cout << "  Analyze GRID4 hotspot leaves for dense override..." << std::endl;
        std::vector<LeafErrorStats> leafStats = analyzeLeafErrors(
            volumeSequence, compressedVolume, btree, width, height, depth, frames);
        applyGrid4DenseHotspotOverride(
            btree, compressedVolume, leafStats, width, height, depth, residualOpt.grid4DenseHotspots);
    }
    if (residualOpt.grid4ResidualHotspots > 0 && enableGrid4Spatial) {
        std::cout << "  Analyze GRID4 hotspot leaves for sparse residual..." << std::endl;
        const auto residualStart = std::chrono::steady_clock::now();
        std::vector<LeafErrorStats> leafStats =
            (residualOpt.grid4ResidualLeafSampleStep > 1)
            ? analyzeLeafErrorsSampled(
                volumeSequence, compressedVolume, btree, width, height, depth, frames,
                residualOpt.grid4ResidualLeafSampleStep)
            : analyzeLeafErrors(
                volumeSequence, compressedVolume, btree, width, height, depth, frames);
        if (residualOpt.grid4ResidualRegionSeeds > 0) {
            const size_t beforeCount = leafStats.size();
            leafStats = selectResidualRegionLeaves(
                leafStats,
                residualOpt.grid4ResidualHotspots,
                residualOpt.grid4ResidualRegionSeeds,
                residualOpt.grid4ResidualRegionRadius);
            printf("  GRID4 residual regions: seeds=%d radius=%d selected=%zu/%zu\n",
                   residualOpt.grid4ResidualRegionSeeds,
                   residualOpt.grid4ResidualRegionRadius,
                   leafStats.size(),
                   beforeCount);
        }
        const double fullRange = std::max(1e-6, static_cast<double>(dataMax) - dataMin);
        const bool densityLike = (fieldProfile.type == FieldType::DENSITY);
        const double residualThr = (residualOpt.grid4ResidualThr > 0.0)
            ? residualOpt.grid4ResidualThr
            : (densityLike ? std::max(1e-6, 0.10 * fullRange)
                           : std::max(0.005, 0.03 * fullRange));
        const double residualRelThr = (residualOpt.grid4ResidualRelThr > 0.0)
            ? residualOpt.grid4ResidualRelThr
            : (densityLike ? 0.10 : std::numeric_limits<double>::infinity());
        const double residualLocalFloor = (residualOpt.grid4ResidualLocalFloor > 0.0)
            ? residualOpt.grid4ResidualLocalFloor
            : (densityLike ? std::max(1e-6, 0.10 * fullRange)
                           : std::max(1e-6, 0.01 * fullRange));
        applyGrid4ResidualHotspots(
            btree, volumeSequence, compressedVolume, leafStats, fieldProfile,
            width, height, depth, frames,
            residualOpt.grid4ResidualHotspots, residualThr, residualRelThr,
            residualLocalFloor, residualOpt.grid4ResidualBandFactor, residualOpt.grid4ResidualKeepRel,
            residualOpt.grid4ResidualRankNormalized || densityLike,
            residualOpt.grid4ResidualDpEps);
        const auto residualElapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - residualStart).count();
        printf("  Residual Stage Time: %.3f s\n", residualElapsed);
    }
    if (residualOpt.poly11ResidualHotspots > 0 && enablePoly11Spatial) {
        std::cout << "  Analyze POLY11 hotspot leaves for sparse residual..." << std::endl;
        std::vector<LeafErrorStats> leafStats = analyzeLeafErrors(
            volumeSequence, compressedVolume, btree, width, height, depth, frames);
        const double fullRange = std::max(1e-6, static_cast<double>(dataMax) - dataMin);
        const double residualThr = (residualOpt.poly11ResidualThr > 0.0)
            ? residualOpt.poly11ResidualThr
            : std::max(0.005, 0.03 * fullRange);
        applyPoly11ResidualHotspots(
            btree, volumeSequence, compressedVolume, leafStats, width, height, depth, frames,
            residualOpt.poly11ResidualHotspots, residualThr, residualOpt.poly11ResidualDpEps);
    }
    if (residualOpt.poly11DenseHotspots > 0 && enablePoly11Spatial) {
        std::cout << "  Analyze POLY11 hotspot leaves for dense override..." << std::endl;
        std::vector<LeafErrorStats> leafStats = analyzeLeafErrors(
            volumeSequence, compressedVolume, btree, width, height, depth, frames);
        applyPoly11DenseHotspotOverride(
            btree, compressedVolume, leafStats, width, height, depth,
            residualOpt.poly11DenseHotspots);
    }
    auto bs = btree.getStats();
    printf("  Leaves    : %d  (U=%d C=%d D=%d G=%d R=%d P=%d PR=%d GM=%d)\n",
           bs.totalLeafs, bs.mode0, bs.mode1, bs.mode2, bs.mode3, bs.mode4, bs.mode5, bs.mode6, bs.mode7);
    printf("  Total KF  : %lld  (%.2f%% of temporal-only)\n", bs.totalKF, 100.0 * bs.totalKF / totalKF);
    if (bs.mode1 > 0) {
        printf("  Cluster K : mean=%.1f  P95=%.0f  max=%.0f\n", bs.kMean, bs.kP95, bs.kMax);
    }
    if (bs.blockAwareCluster || bs.budgetAwareCluster) {
        printf("  Local cThr: min=%.3f  mean=%.3f  max=%.3f\n",
               bs.localClusterThrMin, bs.localClusterThrMean, bs.localClusterThrMax);
    }
    if (bs.guardedMedoidCluster) {
        printf("  Guarded medoid stats: gateRejects=%d  medoidChanges=%d  soloPromotions=%d\n",
               bs.guardedGateRejects, bs.medoidCenterChanges, bs.qualityGatePromotions);
    }
    if (bs.validateFallback) {
        printf("  Validate fallback blocks: %d\n", bs.validateFallbackBlocks);
        printf("    retry-cluster: %d  dense-fallback: %d\n",
               bs.validateRetryBlocks, bs.validateDenseFallbackBlocks);
    }
    if (bs.hotspotSecondPass) {
        printf("  Hotspot second-pass blocks: %d\n", bs.hotspotSecondPassBlocks);
        printf("    dense-fallback: %d\n", bs.hotspotSecondPassDenseBlocks);
        printf("    region-seeds: %d  region-retouched: %d  region-dense-fallback: %d\n",
               bs.hotspotRegionSeedBlocks, bs.hotspotRegionTouchedBlocks, bs.hotspotRegionDenseFallbackBlocks);
    }
    printf("  Est. size : %.1f KB\n", bs.fileSizeEstimate / 1024.0);

    std::cout << "\nStep 5: Flatten to A/B and save VBT (Float16)..." << std::endl;
    const auto flattenStart = std::chrono::steady_clock::now();
    try {
        btree.flattenLeaves();
    } catch (const std::exception& ex) {
        std::cerr << "Flatten/save preparation failed: " << ex.what() << std::endl;
        return 2;
    }
    const auto flattenElapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - flattenStart).count();
    printf("  Flatten Time: %.3f s\n", flattenElapsed);
    if (!saveToVBT(outVBT, btree)) {
        std::cerr << "  Save failed!" << std::endl;
        return 1;
    }

    std::cout << "\nStep 6: Load reloaded VBT and validate..." << std::endl;
    BlockTree loaded;
    if (!loadFromVBT(outVBT, loaded)) {
        std::cerr << "  Load failed!" << std::endl;
        return 1;
    }
    if (!validateVBT(loaded)) {
        std::cerr << "  validateVBT failed on reloaded VBT!" << std::endl;
        return 1;
    }

    if (!skipFullSweep) {
    const auto sweepStart = std::chrono::steady_clock::now();
    if (!sampledFullSweep) {
    std::cout << "\nStep 7: Full error sweep (reloaded VBT vs original FP32 raw)..." << std::endl;
    {
        const long long N = static_cast<long long>(width) * height * depth * frames;
        double errSum2 = 0.0;
        double errSum = 0.0;
        float errMax = 0.0f;

        #ifdef _OPENMP
        const int sweepThreads = std::max(1, omp_get_max_threads());
        std::vector<double> errSumsByThread(static_cast<size_t>(sweepThreads), 0.0);
        std::vector<double> errSum2ByThread(static_cast<size_t>(sweepThreads), 0.0);
        std::vector<float> errMaxByThread(static_cast<size_t>(sweepThreads), 0.0f);
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double& localErrSum = errSumsByThread[static_cast<size_t>(tid)];
            double& localErrSum2 = errSum2ByThread[static_cast<size_t>(tid)];
            float& localErrMax = errMaxByThread[static_cast<size_t>(tid)];

            #pragma omp for collapse(3) nowait
            for (int z = 0; z < depth; ++z) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        for (int t = 0; t < frames; ++t) {
                            float got = loaded.sampleFlat(x, y, z, static_cast<float>(t));
                            float orig = volumeSequence[t][z][y][x];
                            float e = std::abs(got - orig);
                            localErrSum += e;
                            localErrSum2 += static_cast<double>(e) * e;
                            localErrMax = std::max(localErrMax, e);
                        }
                    }
                }
            }
        }
        for (int tid = 0; tid < sweepThreads; ++tid) {
            errSum += errSumsByThread[static_cast<size_t>(tid)];
            errSum2 += errSum2ByThread[static_cast<size_t>(tid)];
            errMax = std::max(errMax, errMaxByThread[static_cast<size_t>(tid)]);
        }
        #else
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    for (int t = 0; t < frames; ++t) {
                        float got = loaded.sampleFlat(x, y, z, static_cast<float>(t));
                        float orig = volumeSequence[t][z][y][x];
                        float e = std::abs(got - orig);
                        errSum += e;
                        errSum2 += static_cast<double>(e) * e;
                        errMax = std::max(errMax, e);
                    }
                }
            }
        }
        #endif

        const double rmse = std::sqrt(errSum2 / N);
        const double psnrPeak = useDataRangeForPsnr ? std::max(1e-6, static_cast<double>(dataMax) - dataMin) : 255.0;
        const double psnr = computePsnr(rmse, psnrPeak);

        const size_t HIST_BINS = 1u << 18;
        std::vector<unsigned long long> hist(HIST_BINS, 0);
        const double invMaxErr = errMax > 0.0f ? (static_cast<double>(HIST_BINS - 1) / errMax) : 0.0;

        #ifdef _OPENMP
        std::vector<std::vector<unsigned long long>> histByThread(
            static_cast<size_t>(sweepThreads),
            std::vector<unsigned long long>(HIST_BINS, 0));
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            std::vector<unsigned long long>& localHist = histByThread[static_cast<size_t>(tid)];

            #pragma omp for collapse(3) nowait
            for (int z = 0; z < depth; ++z) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        for (int t = 0; t < frames; ++t) {
                            float got = loaded.sampleFlat(x, y, z, static_cast<float>(t));
                            float orig = volumeSequence[t][z][y][x];
                            float e = std::abs(got - orig);
                            size_t bin = (errMax > 0.0f)
                                ? static_cast<size_t>(std::min<double>(HIST_BINS - 1, e * invMaxErr))
                                : 0;
                            localHist[bin] += 1;
                        }
                    }
                }
            }
        }
        for (int tid = 0; tid < sweepThreads; ++tid) {
            const std::vector<unsigned long long>& localHist = histByThread[static_cast<size_t>(tid)];
            for (size_t i = 0; i < HIST_BINS; ++i) hist[i] += localHist[i];
        }
        #else
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    for (int t = 0; t < frames; ++t) {
                        float got = loaded.sampleFlat(x, y, z, static_cast<float>(t));
                        float orig = volumeSequence[t][z][y][x];
                        float e = std::abs(got - orig);
                        size_t bin = (errMax > 0.0f)
                            ? static_cast<size_t>(std::min<double>(HIST_BINS - 1, e * invMaxErr))
                            : 0;
                        hist[bin] += 1;
                    }
                }
            }
        }
        #endif

        auto quantileFromHist = [&](double q) -> double {
            const unsigned long long target = static_cast<unsigned long long>(std::floor(q * static_cast<double>(N - 1)));
            unsigned long long acc = 0;
            for (size_t i = 0; i < HIST_BINS; ++i) {
                acc += hist[i];
                if (acc > target) {
                    return errMax > 0.0f ? (static_cast<double>(i) / invMaxErr) : 0.0;
                }
            }
            return static_cast<double>(errMax);
        };

        const double q99 = quantileFromHist(0.99);
        const double q999 = quantileFromHist(0.999);

        printf("\n=== Full Error (reloaded VBT sampleFlat vs FP32 raw) ===\n");
        printf("  Mean=%.4f  Max=%.4f  P99=%.4f  P99.9=%.4f\n",
               errSum / N, errMax, q99, q999);
        printf("  Root Mean Square Error (RMSE) : %.6f\n", rmse);
        printf("  Peak Signal-to-Noise Ratio    : %.2f dB  (peak=%.6f)\n", psnr, psnrPeak);

        std::ifstream szCheck(outVBT, std::ios::binary | std::ios::ate);
        const size_t vbtBytes = szCheck ? static_cast<size_t>(szCheck.tellg()) : 0;
        printf("\n=== Compression Summary (reloaded VBT validated) ===\n");
        printf("  Original Format             : %s\n", detectedFormat == RawDataFormat::UINT8 ? "uint8" : "float32");
        printf("  Total Original Size (raw)   : %.2f MB\n", rawBytesOrig / (1024.0 * 1024.0));
        printf("  Total Original Size (FP32)  : %.2f MB\n", rawBytesF32 / (1024.0 * 1024.0));
        printf("  Data Range                  : [%.6f, %.6f]\n", dataMin, dataMax);
        printf("  VBT Size (FP16 Compressed)  : %.2f KB\n", vbtBytes / 1024.0);
        printf("  Compression Ratio (vs FP32) : %.2f%%\n", rawBytesF32 > 0 ? 100.0 * vbtBytes / rawBytesF32 : 0.0);
        printf("  Compression Ratio (vs raw)  : %.2f%%\n", rawBytesOrig > 0 ? 100.0 * vbtBytes / rawBytesOrig : 0.0);
        printf("  Root Mean Square Error (RMSE): %.6f\n", rmse);
        printf("  Peak Signal-to-Noise Ratio   : %.2f dB  (peak=%.6f)\n", psnr, psnrPeak);
        printf("VBT_SUMMARY_RELOADED leaves=%d kf=%lld cluster_thr=%.1f mean=%.4f max=%.4f rmse=%.6f psnr=%.2f\n",
               bs.totalLeafs, bs.totalKF, v5ThreshCluster, errSum / N, errMax, rmse, psnr);
    }
    } else {
        std::cout << "\nStep 7: Sampled full error sweep (reloaded VBT vs original FP32 raw)..." << std::endl;
        long long sampleCount = 0;
        double errSum2 = 0.0;
        double errSum = 0.0;
        float errMax = 0.0f;
        std::vector<float> errs;
        const long long estSamples =
            static_cast<long long>((depth + sampledStepZ - 1) / sampledStepZ) *
            static_cast<long long>((height + sampledStepY - 1) / sampledStepY) *
            static_cast<long long>((width + sampledStepX - 1) / sampledStepX) *
            static_cast<long long>((frames + sampledStepT - 1) / sampledStepT);
        errs.reserve(static_cast<size_t>(std::max<long long>(1, estSamples)));

        #ifdef _OPENMP
        const int sweepThreads = std::max(1, omp_get_max_threads());
        std::vector<std::vector<float>> errsByThread(static_cast<size_t>(sweepThreads));
        std::vector<double> errSumsByThread(static_cast<size_t>(sweepThreads), 0.0);
        std::vector<double> errSum2ByThread(static_cast<size_t>(sweepThreads), 0.0);
        std::vector<float> errMaxByThread(static_cast<size_t>(sweepThreads), 0.0f);
        std::vector<long long> sampleCountByThread(static_cast<size_t>(sweepThreads), 0);
        for (int tid = 0; tid < sweepThreads; ++tid) {
            errsByThread[static_cast<size_t>(tid)].reserve(
                static_cast<size_t>(std::max<long long>(1, estSamples / sweepThreads + 1024)));
        }
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            std::vector<float>& localErrs = errsByThread[static_cast<size_t>(tid)];
            double& localErrSum = errSumsByThread[static_cast<size_t>(tid)];
            double& localErrSum2 = errSum2ByThread[static_cast<size_t>(tid)];
            float& localErrMax = errMaxByThread[static_cast<size_t>(tid)];
            long long& localCount = sampleCountByThread[static_cast<size_t>(tid)];

            #pragma omp for collapse(3) nowait
            for (int z = 0; z < depth; z += sampledStepZ) {
                for (int y = 0; y < height; y += sampledStepY) {
                    for (int x = 0; x < width; x += sampledStepX) {
                        for (int t = 0; t < frames; t += sampledStepT) {
                            float got = loaded.sampleFlat(x, y, z, static_cast<float>(t));
                            float orig = volumeSequence[t][z][y][x];
                            float e = std::abs(got - orig);
                            localErrs.push_back(e);
                            localErrSum += e;
                            localErrSum2 += static_cast<double>(e) * e;
                            localErrMax = std::max(localErrMax, e);
                            ++localCount;
                        }
                    }
                }
            }
        }
        for (int tid = 0; tid < sweepThreads; ++tid) {
            std::vector<float>& localErrs = errsByThread[static_cast<size_t>(tid)];
            errs.insert(errs.end(), localErrs.begin(), localErrs.end());
            errSum += errSumsByThread[static_cast<size_t>(tid)];
            errSum2 += errSum2ByThread[static_cast<size_t>(tid)];
            errMax = std::max(errMax, errMaxByThread[static_cast<size_t>(tid)]);
            sampleCount += sampleCountByThread[static_cast<size_t>(tid)];
        }
        #else
        for (int z = 0; z < depth; z += sampledStepZ) {
            for (int y = 0; y < height; y += sampledStepY) {
                for (int x = 0; x < width; x += sampledStepX) {
                    for (int t = 0; t < frames; t += sampledStepT) {
                        float got = loaded.sampleFlat(x, y, z, static_cast<float>(t));
                        float orig = volumeSequence[t][z][y][x];
                        float e = std::abs(got - orig);
                        errs.push_back(e);
                        errSum += e;
                        errSum2 += static_cast<double>(e) * e;
                        errMax = std::max(errMax, e);
                        ++sampleCount;
                    }
                }
            }
        }
        #endif

        std::sort(errs.begin(), errs.end());
        const size_t idx99 = std::min(errs.size() - 1, static_cast<size_t>(sampleCount * 0.99));
        const size_t idx999 = std::min(errs.size() - 1, static_cast<size_t>(sampleCount * 0.999));
        const double rmse = std::sqrt(errSum2 / std::max(1LL, sampleCount));
        const double psnrPeak = useDataRangeForPsnr ? std::max(1e-6, static_cast<double>(dataMax) - dataMin) : 255.0;
        const double psnr = computePsnr(rmse, psnrPeak);

        printf("\n=== Sampled Full Error (reloaded VBT sampleFlat vs FP32 raw) ===\n");
        printf("  Sample count=%lld  step=(%d,%d,%d,%d)\n", sampleCount, sampledStepX, sampledStepY, sampledStepZ, sampledStepT);
        printf("  Mean=%.4f  Max=%.4f  P99=%.4f  P99.9=%.4f\n",
               errSum / std::max(1LL, sampleCount), errMax, errs[idx99], errs[idx999]);
        printf("  Root Mean Square Error (RMSE) : %.6f\n", rmse);
        printf("  Peak Signal-to-Noise Ratio    : %.2f dB  (peak=%.6f)\n", psnr, psnrPeak);

        std::ifstream szCheck(outVBT, std::ios::binary | std::ios::ate);
        const size_t vbtBytes = szCheck ? static_cast<size_t>(szCheck.tellg()) : 0;
        printf("\n=== Compression Summary (reloaded VBT validated, sampled full sweep) ===\n");
        printf("  Original Format             : %s\n", detectedFormat == RawDataFormat::UINT8 ? "uint8" : "float32");
        printf("  Total Original Size (raw)   : %.2f MB\n", rawBytesOrig / (1024.0 * 1024.0));
        printf("  Total Original Size (FP32)  : %.2f MB\n", rawBytesF32 / (1024.0 * 1024.0));
        printf("  Data Range                  : [%.6f, %.6f]\n", dataMin, dataMax);
        printf("  VBT Size (FP16 Compressed)  : %.2f KB\n", vbtBytes / 1024.0);
        printf("  Compression Ratio (vs FP32) : %.2f%%\n", rawBytesF32 > 0 ? 100.0 * vbtBytes / rawBytesF32 : 0.0);
        printf("  Compression Ratio (vs raw)  : %.2f%%\n", rawBytesOrig > 0 ? 100.0 * vbtBytes / rawBytesOrig : 0.0);
        printf("  Root Mean Square Error (RMSE): %.6f\n", rmse);
        printf("  Peak Signal-to-Noise Ratio   : %.2f dB  (peak=%.6f)\n", psnr, psnrPeak);
        printf("VBT_SUMMARY_RELOADED_SAMPLED leaves=%d kf=%lld cluster_thr=%.1f samples=%lld rmse=%.6f psnr=%.2f\n",
               bs.totalLeafs, bs.totalKF, v5ThreshCluster, sampleCount, rmse, psnr);
    }
    const auto sweepElapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - sweepStart).count();
    printf("  Step 7 Time: %.3f s\n", sweepElapsed);
    } else {
        std::cout << "\nStep 7: Full error sweep skipped (--skip-full-sweep)." << std::endl;
        std::ifstream szCheck(outVBT, std::ios::binary | std::ios::ate);
        const size_t vbtBytes = szCheck ? static_cast<size_t>(szCheck.tellg()) : 0;
        printf("\n=== Compression Summary (reloaded VBT validated, full sweep skipped) ===\n");
        printf("  Original Format             : %s\n", detectedFormat == RawDataFormat::UINT8 ? "uint8" : "float32");
        printf("  Total Original Size (raw)   : %.2f MB\n", rawBytesOrig / (1024.0 * 1024.0));
        printf("  Total Original Size (FP32)  : %.2f MB\n", rawBytesF32 / (1024.0 * 1024.0));
        printf("  Data Range                  : [%.6f, %.6f]\n", dataMin, dataMax);
        printf("  VBT Size (FP16 Compressed)  : %.2f KB\n", vbtBytes / 1024.0);
        printf("  Compression Ratio (vs FP32) : %.2f%%\n", rawBytesF32 > 0 ? 100.0 * vbtBytes / rawBytesF32 : 0.0);
        printf("  Compression Ratio (vs raw)  : %.2f%%\n", rawBytesOrig > 0 ? 100.0 * vbtBytes / rawBytesOrig : 0.0);
        printf("VBT_SUMMARY_RELOADED leaves=%d kf=%lld cluster_thr=%.1f full_sweep=skipped\n",
               bs.totalLeafs, bs.totalKF, v5ThreshCluster);
    }

    if (leafAnalysis.enabled) {
        std::cout << "\nStep 8: Leaf hotspot analysis..." << std::endl;
        std::vector<LeafErrorStats> leafStats = analyzeLeafErrors(
            volumeSequence, compressedVolume, btree, width, height, depth, frames);
        writeLeafErrorReport(inputFile, leafStats, leafAnalysis.topK);
    }

    std::cout << "\n========== Done ==========" << std::endl;
    return 0;
}
