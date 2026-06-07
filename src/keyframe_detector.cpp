#include "keyframe_detector.h"

#include <algorithm>
#include <cmath>

namespace vbt {

namespace {

struct GenericTemporalStats {
    float mean = 0.0f;
    float stddev = 0.0f;
    float deltaScale = 1.0f;
};

GenericTemporalStats computeGenericStats(const std::vector<float>& values)
{
    GenericTemporalStats stats;
    if (values.empty()) return stats;

    double sum = 0.0;
    double sum2 = 0.0;
    for (float v : values) {
        sum += v;
        sum2 += static_cast<double>(v) * v;
    }
    stats.mean = static_cast<float>(sum / values.size());
    const double var = std::max(0.0, sum2 / values.size() - static_cast<double>(stats.mean) * stats.mean);
    stats.stddev = static_cast<float>(std::sqrt(var));

    if (values.size() >= 2) {
        std::vector<float> deltas;
        deltas.reserve(values.size() - 1);
        for (size_t i = 1; i < values.size(); ++i) {
            deltas.push_back(std::abs(values[i] - values[i - 1]));
        }
        std::sort(deltas.begin(), deltas.end());
        stats.deltaScale = std::max(1e-4f, deltas[(deltas.size() * 9) / 10]);
    }
    return stats;
}

float epsilonFloor(const FieldProfile& profile)
{
    if (profile.type == FieldType::SDF) return 0.5f;
    return std::max(1e-4f, profile.den.epsAbs * 0.1f);
}

float epsilonFunc(int t,
                  const std::vector<float>& values,
                  float predVal,
                  const FieldProfile& profile,
                  const GenericTemporalStats& genericStats,
                  float epsFloor)
{
    if (profile.type == FieldType::SDF) {
        const float truthDist = std::abs(values[t] - profile.sdf.iso);
        const float predDist = std::abs(predVal - profile.sdf.iso);
        const float minDist = std::min(truthDist, predDist);
        if (minDist <= profile.sdf.wCritical) return std::max(profile.sdf.epsCritical, epsFloor);
        if (minDist <= profile.sdf.wNear) return std::max(profile.sdf.epsNear, epsFloor);
        if (minDist <= profile.sdf.wBand) return std::max(0.5f * (profile.sdf.epsNear + profile.sdf.epsFar), epsFloor);
        return std::max(profile.sdf.epsFar, epsFloor);
    }

    float eps = profile.den.epsAbs + profile.den.epsRel * std::max(std::abs(values[t]), genericStats.stddev);
    if (profile.den.gammaDelta > 0.0f && t > 0) {
        const float delta = std::abs(values[t] - values[t - 1]) / genericStats.deltaScale;
        eps /= (1.0f + profile.den.gammaDelta * delta);
    }
    return std::max(eps, epsFloor);
}

void recurseDp(const std::vector<float>& values,
               int a,
               int b,
               std::vector<int>& keys,
               const FieldProfile& profile,
               const GenericTemporalStats& genericStats,
               float epsFloor)
{
    if (b - a <= 1) return;

    const double v0 = values[a];
    const double v1 = values[b];
    double maxErrNorm = 0.0;
    int split = -1;
    for (int t = a + 1; t < b; ++t) {
        const double alpha = static_cast<double>(t - a) / static_cast<double>(b - a);
        const double pred = v0 + alpha * (v1 - v0);
        const double eps = epsilonFunc(t, values, static_cast<float>(pred), profile, genericStats, epsFloor);
        const double errNorm = std::abs(static_cast<double>(values[t]) - pred) / eps;
        if (errNorm > maxErrNorm) {
            maxErrNorm = errNorm;
            split = t;
        }
    }

    if (maxErrNorm > 1.0 && split != -1) {
        keys.push_back(split);
        recurseDp(values, a, split, keys, profile, genericStats, epsFloor);
        recurseDp(values, split, b, keys, profile, genericStats, epsFloor);
    }
}

} // namespace

std::vector<int> detectKeyFrames(const std::vector<float>& values,
                                 const FieldProfile& profile)
{
    const int n = static_cast<int>(values.size());
    if (n == 0) return {};
    if (n == 1) return {0};

    const float epsFloor = epsilonFloor(profile);
    const GenericTemporalStats genericStats =
        (profile.type == FieldType::GENERIC || profile.type == FieldType::DENSITY)
            ? computeGenericStats(values)
            : GenericTemporalStats{};

    std::vector<int> keys = {0, n - 1};
    recurseDp(values, 0, n - 1, keys, profile, genericStats, epsFloor);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

} // namespace vbt
