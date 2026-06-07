#pragma once

#include "vbt_file.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vbt::render {

struct Query4D {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
    uint32_t t = 0;
    uint32_t leafIndex = 0;
};

enum class QueryPattern {
    RandomFull,
    SameT,
    SameXYZ,
    CoherentTiles,
};

bool parseQueryPattern(const std::string& text, QueryPattern& outPattern);
std::vector<Query4D> generateQueries(const VbtFileHeader& header,
                                     QueryPattern pattern,
                                     uint32_t count,
                                     uint32_t seed);
void fillLeafIndices(const VbtFileHeader& header, std::vector<Query4D>& queries);
void sortQueriesByLeaf(std::vector<Query4D>& queries);

} // namespace vbt::render
