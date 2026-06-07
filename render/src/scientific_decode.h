#pragma once

#include "query_patterns.h"
#include "vbt_file.h"

#include <cstdint>
#include <vector>

namespace vbt::render {

float decodeScientificValueAtCpu(const VbtFile& file, const Query4D& query);
std::vector<float> reconstructScientificFrameCpu(const VbtFile& file, uint32_t frameIndex);

} // namespace vbt::render
