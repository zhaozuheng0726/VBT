#pragma once

#include <vector>

namespace vbt {

std::vector<float> dctEncodeKeep(const std::vector<float>& series, int keepCount);
std::vector<float> dctEncodeKeepWeighted(const std::vector<float>& series,
                                         const std::vector<float>& weights,
                                         int keepCount);
float dctDecodeAt(const std::vector<float>& coeffs, int totalLength, int index);

} // namespace vbt
