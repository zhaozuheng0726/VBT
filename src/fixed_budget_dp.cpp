#include "fixed_budget_dp.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vbt {

namespace {

float segmentCost(const std::vector<float>& series, int a, int b)
{
    if (b <= a + 1) return 0.0f;
    const float va = series[a];
    const float vb = series[b];
    float sse = 0.0f;
    for (int t = a + 1; t < b; ++t) {
        const float alpha = static_cast<float>(t - a) / static_cast<float>(b - a);
        const float pred = va + alpha * (vb - va);
        const float diff = series[t] - pred;
        sse += diff * diff;
    }
    return sse;
}

float segmentCostWeighted(const std::vector<float>& series,
                          const std::vector<float>& weights,
                          int a, int b)
{
    if (b <= a + 1) return 0.0f;
    const float va = series[a];
    const float vb = series[b];
    float sse = 0.0f;
    for (int t = a + 1; t < b; ++t) {
        const float alpha = static_cast<float>(t - a) / static_cast<float>(b - a);
        const float pred = va + alpha * (vb - va);
        const float diff = series[t] - pred;
        const float w = std::max(1e-6f, weights.empty() ? 1.0f : weights[static_cast<size_t>(t)]);
        sse += w * diff * diff;
    }
    return sse;
}

FixedBudgetSegment encodeImpl(const std::vector<float>& series,
                              const std::vector<float>& weights,
                              int keyframes,
                              bool weighted)
{
    const int n = static_cast<int>(series.size());
    const int k = std::max(2, std::min(keyframes, n));
    if (n == 0) return {};
    if (n == 1) return {{0}, {series[0]}, 0.0f};

    std::vector<std::vector<float>> cost(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            cost[i][j] = weighted ? segmentCostWeighted(series, weights, i, j)
                                  : segmentCost(series, i, j);
        }
    }

    const float inf = std::numeric_limits<float>::infinity();
    std::vector<std::vector<float>> dp(k + 1, std::vector<float>(n, inf));
    std::vector<std::vector<int>> prev(k + 1, std::vector<int>(n, -1));

    dp[1][0] = 0.0f;
    for (int m = 2; m <= k; ++m) {
        for (int j = m - 1; j < n; ++j) {
            for (int i = m - 2; i < j; ++i) {
                if (!std::isfinite(dp[m - 1][i])) continue;
                const float candidate = dp[m - 1][i] + cost[i][j];
                if (candidate < dp[m][j]) {
                    dp[m][j] = candidate;
                    prev[m][j] = i;
                }
            }
        }
    }

    FixedBudgetSegment encoded;
    encoded.sse = dp[k][n - 1];
    encoded.times.resize(k);
    encoded.values.resize(k);

    int cur = n - 1;
    for (int m = k; m >= 1; --m) {
        encoded.times[m - 1] = cur;
        encoded.values[m - 1] = series[cur];
        cur = (m > 1) ? prev[m][cur] : -1;
    }
    return encoded;
}

} // namespace

FixedBudgetSegment encodeFixedBudgetLinear(const std::vector<float>& series, int keyframes)
{
    return encodeImpl(series, {}, keyframes, false);
}

FixedBudgetSegment encodeFixedBudgetLinearWeighted(const std::vector<float>& series,
                                                   const std::vector<float>& weights,
                                                   int keyframes)
{
    return encodeImpl(series, weights, keyframes, true);
}

float decodeFixedBudgetAt(const FixedBudgetSegment& encoded, int index)
{
    if (encoded.times.empty()) return 0.0f;
    if (encoded.times.size() == 1) return encoded.values[0];
    if (index <= encoded.times.front()) return encoded.values.front();
    if (index >= encoded.times.back()) return encoded.values.back();

    for (size_t i = 1; i < encoded.times.size(); ++i) {
        if (index <= encoded.times[i]) {
            const int t0 = encoded.times[i - 1];
            const int t1 = encoded.times[i];
            const float v0 = encoded.values[i - 1];
            const float v1 = encoded.values[i];
            const float alpha = static_cast<float>(index - t0) / static_cast<float>(t1 - t0);
            return v0 + alpha * (v1 - v0);
        }
    }
    return encoded.values.back();
}

} // namespace vbt
