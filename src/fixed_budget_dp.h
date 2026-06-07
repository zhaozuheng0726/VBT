#pragma once

#include <vector>

namespace vbt {

struct FixedBudgetSegment {
    std::vector<int> times;
    std::vector<float> values;
    float sse = 0.0f;
};

FixedBudgetSegment encodeFixedBudgetLinear(const std::vector<float>& series, int keyframes);
FixedBudgetSegment encodeFixedBudgetLinearWeighted(const std::vector<float>& series,
                                                   const std::vector<float>& weights,
                                                   int keyframes);
float decodeFixedBudgetAt(const FixedBudgetSegment& encoded, int index);

} // namespace vbt
