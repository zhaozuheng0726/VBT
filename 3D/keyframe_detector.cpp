#include "keyframe_detector.h"

#include <algorithm>
#include <cmath>
#include <vector>

static const float KF_EPS_MIN_SDF = 0.5f;
static const int KF_MAX_DEPTH = 300;

struct GenericTemporalStats {
    float mean = 0.0f;
    float stddev = 0.0f;
    float deltaScale = 1.0f;
};

static float epsilon_floor(const FieldProfile& profile) {
    if (profile.type == FieldType::SDF) {
        return KF_EPS_MIN_SDF;
    }
    return std::max(1e-4f, profile.den.eps_abs * 0.1f);
}

static int cutoff_state(float v, float cutoff, float band) {
    if (cutoff <= 0.0f) return 1;
    if (v < cutoff - band) return 0;
    if (v > cutoff + band) return 2;
    return 1;
}

static GenericTemporalStats computeGenericStats(const std::vector<float>& values) {
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
        const size_t idx = std::min(deltas.size() - 1, (deltas.size() * 9) / 10);
        stats.deltaScale = std::max(1e-4f, deltas[idx]);
    }

    return stats;
}

static float epsilon_func(int t,
                          const std::vector<float>& values,
                          float predVal,
                          const FieldProfile& profile,
                          const GenericTemporalStats& genericStats,
                          float epsFloor,
                          const std::vector<float>* frameWeights) {
    float eps = 0.0f;
    if (profile.type == FieldType::SDF) {
        const float truthDist = std::abs(values[t] - profile.sdf.iso);
        const float predDist = std::abs(predVal - profile.sdf.iso);
        const float minDist = std::min(truthDist, predDist);
        if (minDist <= profile.sdf.w_critical) {
            eps = profile.sdf.eps_critical;
        } else if (minDist <= profile.sdf.w_near) {
            eps = profile.sdf.eps_near;
        } else if (minDist <= profile.sdf.w_band) {
            eps = 0.5f * (profile.sdf.eps_near + profile.sdf.eps_far);
        } else {
            eps = profile.sdf.eps_far;
        }
    } else if (profile.type == FieldType::GENERIC) {
        const float centeredMag = std::max(std::abs(values[t] - genericStats.mean), genericStats.stddev);
        eps = profile.den.eps_abs + profile.den.eps_rel * centeredMag;
        if (profile.den.gamma_delta > 0.0f && t > 0) {
            const float delta = std::abs(values[t] - values[t - 1]) / genericStats.deltaScale;
            eps = eps / (1.0f + profile.den.gamma_delta * delta);
        }
    } else {
        eps = profile.epsilon_at(values, t);
    }
    eps = std::max(eps, epsFloor);
    if (frameWeights != nullptr && static_cast<size_t>(t) < frameWeights->size()) {
        const float w = std::max(1.0f, (*frameWeights)[static_cast<size_t>(t)]);
        eps /= w;
    }
    return std::max(eps, epsFloor * 0.25f);
}

static void dp_recurse(const std::vector<float>& values,
                       int a,
                       int b,
                       std::vector<int>& keyOut,
                       int depth,
                       const FieldProfile& profile,
                       const GenericTemporalStats& genericStats,
                       float epsFloor,
                       const std::vector<float>* frameWeights) {
    if (b - a <= 1) return;
    if (depth >= KF_MAX_DEPTH) return;

    const double v0 = values[a];
    const double v1 = values[b];

    double maxErrNorm = 0.0;
    int tStar = -1;

    for (int t = a + 1; t < b; ++t) {
        const double alpha = static_cast<double>(t - a) / static_cast<double>(b - a);
        const double pred = v0 + alpha * (v1 - v0);
        const float eps = epsilon_func(t, values, static_cast<float>(pred), profile, genericStats, epsFloor, frameWeights);
        const double errNorm = std::abs(static_cast<double>(values[t]) - pred) / eps;

        if (errNorm > maxErrNorm) {
            maxErrNorm = errNorm;
            tStar = t;
        }
    }

    if (maxErrNorm > 1.0 && tStar != -1) {
        keyOut.push_back(tStar);
        dp_recurse(values, a, tStar, keyOut, depth + 1, profile, genericStats, epsFloor, frameWeights);
        dp_recurse(values, tStar, b, keyOut, depth + 1, profile, genericStats, epsFloor, frameWeights);
    }
}

std::vector<int> detectKeyFrames(const std::vector<float>& values,
                                 double /* antialiasWidth */,
                                 const FieldProfile& profile) {
    return detectKeyFrames(values, 0.0, profile, std::vector<float>{});
}

std::vector<int> detectKeyFrames(const std::vector<float>& values,
                                 double /* antialiasWidth */,
                                 const FieldProfile& profile,
                                 const std::vector<float>& frameWeights) {
    const int n = static_cast<int>(values.size());
    if (n == 0) return {};
    if (n == 1) return {0};

    const float epsFloor = epsilon_floor(profile);
    const GenericTemporalStats genericStats =
        (profile.type == FieldType::GENERIC) ? computeGenericStats(values) : GenericTemporalStats{};

    {
        double sum = 0.0;
        double sum2 = 0.0;
        for (float v : values) {
            sum += v;
            sum2 += static_cast<double>(v) * v;
        }
        const double mean = sum / n;
        const double var = std::max(0.0, sum2 / n - mean * mean);
        if (var < static_cast<double>(epsFloor) * epsFloor) {
            return {0};
        }
    }

    {
        const double v0 = values[0];
        const double vT = values[n - 1];
        double maxNorm = 0.0;
        for (int t = 1; t < n - 1; ++t) {
            const double alpha = static_cast<double>(t) / static_cast<double>(n - 1);
            const double pred = v0 + alpha * (vT - v0);
            const float eps = epsilon_func(t, values, static_cast<float>(pred), profile, genericStats, epsFloor,
                                           frameWeights.empty() ? nullptr : &frameWeights);
            const double errNorm = std::abs(static_cast<double>(values[t]) - pred) / eps;
            maxNorm = std::max(maxNorm, errNorm);
        }
        if (maxNorm <= 1.0) {
            return {0, n - 1};
        }
    }

    std::vector<int> keyOut;
    keyOut.reserve(32);
    keyOut.push_back(0);
    keyOut.push_back(n - 1);

    dp_recurse(values, 0, n - 1, keyOut, 0, profile, genericStats, epsFloor,
               frameWeights.empty() ? nullptr : &frameWeights);

    if (profile.type == FieldType::DENSITY &&
        profile.den.cutoff_temporal_protect &&
        profile.den.render_cutoff > 0.0f &&
        profile.den.cutoff_band >= 0.0f) {
        const float cutoff = profile.den.render_cutoff;
        const float band = profile.den.cutoff_band;
        for (int t = 0; t < n; ++t) {
            const int s = cutoff_state(values[t], cutoff, band);
            if (s == 1) {
                keyOut.push_back(t);
                if (t > 0) keyOut.push_back(t - 1);
                if (t + 1 < n) keyOut.push_back(t + 1);
            }
            if (t > 0) {
                const int prev = cutoff_state(values[t - 1], cutoff, band);
                if (prev != s) {
                    keyOut.push_back(t - 1);
                    keyOut.push_back(t);
                    if (t + 1 < n) keyOut.push_back(t + 1);
                }
            }
        }
    }

    std::sort(keyOut.begin(), keyOut.end());
    keyOut.erase(std::unique(keyOut.begin(), keyOut.end()), keyOut.end());
    return keyOut;
}
