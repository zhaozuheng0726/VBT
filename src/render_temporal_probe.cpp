#include "render_temporal_probe.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef VBT_USE_OPENMP
#include <omp.h>
#endif

namespace vbt {

namespace {

struct DensityTemporalProfile {
    float epsAbs = 1e-5f;
    float epsRel = 0.02f;
    float gammaDelta = 0.2f;
    float renderCutoff = -1.0f;
    float cutoffBand = 0.0f;
    bool cutoffTemporalProtect = true;
    float bgZeroRatio = 0.30f;
};

struct GenericTemporalStats {
    float mean = 0.0f;
    float stddev = 0.0f;
    float deltaScale = 1.0f;
};

double computePsnr(double rmse, double peak)
{
    if (rmse <= 0.0) return 120.0;
    return 20.0 * std::log10(std::max(peak, 1e-12) / rmse);
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
                float predVal,
                const DensityTemporalProfile& profile,
                const GenericTemporalStats& stats)
{
    float eps = profile.epsAbs + profile.epsRel * std::max(std::abs(values[t]), stats.stddev);
    if (profile.gammaDelta > 0.0f && t > 0) {
        const float delta = std::abs(values[t] - values[t - 1]) / std::max(1e-6f, stats.deltaScale);
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
        const double eps = epsilonAt(values, t, static_cast<float>(pred), profile, stats);
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
                                        bool* usedCutoffProtect)
{
    const int n = static_cast<int>(values.size());
    if (usedCutoffProtect) *usedCutoffProtect = false;
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
        if (var < static_cast<double>(epsFloor) * epsFloor) {
            return {0};
        }
    }

    std::vector<int> keys{0, n - 1};
    dpRecurse(values, 0, n - 1, profile, stats, keys);

    if (profile.cutoffTemporalProtect && profile.renderCutoff > 0.0f && profile.cutoffBand >= 0.0f) {
        bool added = false;
        const float cutoff = profile.renderCutoff;
        const float band = profile.cutoffBand;
        for (int t = 0; t < n; ++t) {
            const int s = cutoffState(values[static_cast<size_t>(t)], cutoff, band);
            if (s == 1) {
                keys.push_back(t);
                if (t > 0) keys.push_back(t - 1);
                if (t + 1 < n) keys.push_back(t + 1);
                added = true;
            }
            if (t > 0) {
                const int prev = cutoffState(values[static_cast<size_t>(t - 1)], cutoff, band);
                if (prev != s) {
                    keys.push_back(t - 1);
                    keys.push_back(t);
                    if (t + 1 < n) keys.push_back(t + 1);
                    added = true;
                }
            }
        }
        if (usedCutoffProtect) *usedCutoffProtect = added;
    }

    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
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
        while (seg + 1 < keys.size() && t > keys[seg + 1]) {
            ++seg;
        }
        if (seg + 1 >= keys.size()) {
            out[static_cast<size_t>(t)] = values[static_cast<size_t>(keys.back())];
            continue;
        }
        const int a = keys[seg];
        const int b = keys[seg + 1];
        if (b <= a) {
            out[static_cast<size_t>(t)] = values[static_cast<size_t>(a)];
            continue;
        }
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

} // namespace

RawVolume4D applyRenderTemporalFirstProbe(const RawVolume4D& source,
                                          const SpatialFirstOptions& options,
                                          RenderTemporalFirstProbeStats* outStats)
{
    if (options.profile.type != FieldType::DENSITY) {
        throw std::runtime_error("render temporal-first probe currently only supports density profile");
    }

    const float cutoff = options.profile.den.renderCutoff;
    const float band = std::max(0.0f, options.profile.den.cutoffBand);
    DensityTemporalProfile probeProfile;
    probeProfile.renderCutoff = cutoff;
    probeProfile.cutoffBand = band;
    probeProfile.cutoffTemporalProtect = options.renderTemporalProbeCutoffProtect;
    probeProfile.bgZeroRatio = options.renderTemporalProbeBgZeroRatio;
    if (options.renderTemporalProbeEpsAbs > 0.0f) {
        probeProfile.epsAbs = options.renderTemporalProbeEpsAbs;
    } else if (band > 0.0f) {
        probeProfile.epsAbs = std::max(1e-6f, band * 0.125f);
    } else if (cutoff > 0.0f) {
        probeProfile.epsAbs = std::max(1e-6f, cutoff * 0.05f);
    }
    probeProfile.epsRel = std::max(0.0f, options.renderTemporalProbeEpsRel);
    probeProfile.gammaDelta = std::max(0.0f, options.renderTemporalProbeGammaDelta);

    RawVolume4D recon = source;
    const size_t frameStride = source.frameVoxelCount();
    const size_t totalVoxels = frameStride;
    const int frames = source.meta.frames;
    const float bgThreshold = (cutoff > 0.0f)
        ? std::max(0.0f, probeProfile.bgZeroRatio * cutoff)
        : -1.0f;

    double errSum2 = 0.0;
    uint64_t bgZeroed = 0;
    uint64_t totalKeys = 0;
    uint64_t protectedSeries = 0;
    int maxKeys = 0;

#ifdef VBT_USE_OPENMP
#pragma omp parallel
#endif
    {
        std::vector<float> series(static_cast<size_t>(frames), 0.0f);
        std::vector<float> reconSeries;
        double localErr2 = 0.0;
        uint64_t localBgZeroed = 0;
        uint64_t localKeys = 0;
        uint64_t localProtected = 0;
        int localMaxKeys = 0;

#ifdef VBT_USE_OPENMP
#pragma omp for schedule(dynamic, 256)
#endif
        for (long long voxel = 0; voxel < static_cast<long long>(totalVoxels); ++voxel) {
            float maxVal = 0.0f;
            for (int t = 0; t < frames; ++t) {
                const float v = source.values[static_cast<size_t>(t) * frameStride + static_cast<size_t>(voxel)];
                series[static_cast<size_t>(t)] = v;
                maxVal = std::max(maxVal, v);
            }

            if (bgThreshold >= 0.0f && maxVal < bgThreshold) {
                for (int t = 0; t < frames; ++t) {
                    const float truth = series[static_cast<size_t>(t)];
                    const size_t idx = static_cast<size_t>(t) * frameStride + static_cast<size_t>(voxel);
                    recon.values[idx] = 0.0f;
                    const double diff = static_cast<double>(truth);
                    localErr2 += diff * diff;
                }
                localBgZeroed += 1;
                localKeys += 1;
                localMaxKeys = std::max(localMaxKeys, 1);
                continue;
            }

            bool usedCutoffProtect = false;
            const auto keys = detectDensityKeyFrames(series, probeProfile, &usedCutoffProtect);
            reconstructFromKeys(series, keys, reconSeries);
            localKeys += static_cast<uint64_t>(keys.size());
            localMaxKeys = std::max(localMaxKeys, static_cast<int>(keys.size()));
            if (usedCutoffProtect) localProtected += 1;

            for (int t = 0; t < frames; ++t) {
                const float truth = series[static_cast<size_t>(t)];
                const float pred = reconSeries[static_cast<size_t>(t)];
                const size_t idx = static_cast<size_t>(t) * frameStride + static_cast<size_t>(voxel);
                recon.values[idx] = pred;
                const double diff = static_cast<double>(truth) - static_cast<double>(pred);
                localErr2 += diff * diff;
            }
        }

#ifdef VBT_USE_OPENMP
#pragma omp critical
#endif
        {
            errSum2 += localErr2;
            bgZeroed += localBgZeroed;
            totalKeys += localKeys;
            protectedSeries += localProtected;
            maxKeys = std::max(maxKeys, localMaxKeys);
        }
    }

    if (outStats != nullptr) {
        outStats->voxelCount = static_cast<uint64_t>(totalVoxels);
        outStats->backgroundZeroedVoxels = bgZeroed;
        outStats->totalKeyframes = totalKeys;
        outStats->temporalProtectedSeries = protectedSeries;
        outStats->maxKeyframes = maxKeys;
        const double samples = static_cast<double>(source.totalVoxelCount());
        outStats->rmse = std::sqrt(errSum2 / std::max(1.0, samples));
        const double peak = std::max(1e-6f, source.meta.dataMax - source.meta.dataMin);
        outStats->psnr = computePsnr(outStats->rmse, peak);
    }

    return recon;
}

} // namespace vbt
