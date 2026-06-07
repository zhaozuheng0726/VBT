#include "query_patterns.h"

#include <algorithm>
#include <random>

namespace vbt::render {

bool parseQueryPattern(const std::string& text, QueryPattern& outPattern)
{
    if (text == "random") {
        outPattern = QueryPattern::RandomFull;
        return true;
    }
    if (text == "same-t") {
        outPattern = QueryPattern::SameT;
        return true;
    }
    if (text == "same-xyz") {
        outPattern = QueryPattern::SameXYZ;
        return true;
    }
    if (text == "coherent") {
        outPattern = QueryPattern::CoherentTiles;
        return true;
    }
    return false;
}

std::vector<Query4D> generateQueries(const VbtFileHeader& header,
                                     QueryPattern pattern,
                                     uint32_t count,
                                     uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> dx(0, std::max(1u, header.width) - 1u);
    std::uniform_int_distribution<uint32_t> dy(0, std::max(1u, header.height) - 1u);
    std::uniform_int_distribution<uint32_t> dz(0, std::max(1u, header.depth) - 1u);
    std::uniform_int_distribution<uint32_t> dt(0, std::max(1u, header.frames) - 1u);

    std::vector<Query4D> queries;
    queries.resize(count);

    const uint32_t fixedX = dx(rng);
    const uint32_t fixedY = dy(rng);
    const uint32_t fixedZ = dz(rng);
    const uint32_t fixedT = dt(rng);

    for (uint32_t i = 0; i < count; ++i) {
        Query4D q{};
        switch (pattern) {
        case QueryPattern::RandomFull:
            q.x = dx(rng);
            q.y = dy(rng);
            q.z = dz(rng);
            q.t = dt(rng);
            break;
        case QueryPattern::SameT:
            q.x = dx(rng);
            q.y = dy(rng);
            q.z = dz(rng);
            q.t = fixedT;
            break;
        case QueryPattern::SameXYZ:
            q.x = fixedX;
            q.y = fixedY;
            q.z = fixedZ;
            q.t = dt(rng);
            break;
        case QueryPattern::CoherentTiles: {
            const uint32_t baseX = (dx(rng) / 8u) * 8u;
            const uint32_t baseY = (dy(rng) / 8u) * 8u;
            const uint32_t baseZ = (dz(rng) / 8u) * 8u;
            q.x = std::min(header.width - 1u, baseX + (i % 8u));
            q.y = std::min(header.height - 1u, baseY + ((i / 8u) % 8u));
            q.z = std::min(header.depth - 1u, baseZ + ((i / 64u) % 8u));
            q.t = fixedT;
            break;
        }
        }
        queries[static_cast<size_t>(i)] = q;
    }
    return queries;
}

void fillLeafIndices(const VbtFileHeader& header, std::vector<Query4D>& queries)
{
    for (auto& q : queries) {
        q.leafIndex = leafIndexForVoxel(header, q.x, q.y, q.z);
    }
}

void sortQueriesByLeaf(std::vector<Query4D>& queries)
{
    std::stable_sort(queries.begin(), queries.end(), [](const Query4D& a, const Query4D& b) {
        if (a.leafIndex != b.leafIndex) return a.leafIndex < b.leafIndex;
        if (a.t != b.t) return a.t < b.t;
        if (a.z != b.z) return a.z < b.z;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });
}

} // namespace vbt::render
