#pragma once

#include "query_patterns.h"
#include "vbt_file.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vbt::render {

enum class GpuBenchMode {
    Vbt,
    Dense,
    Compare,
};

struct GpuQueryResult {
    uint32_t leafIndex = 0;
    uint32_t packedHeader = 0;
    uint32_t mode = 0;
    float value = 0.0f;
};

struct GpuQueryBenchStats {
    bool ok = false;
    std::string error;
    double uploadMs = 0.0;
    double gpuDispatchMs = 0.0;
    double readbackMs = 0.0;
    double endToEndMs = 0.0;
    double queriesPerSec = 0.0;
    uint32_t mismatchCount = 0;
    std::vector<GpuQueryResult> results;
};

bool runGpuQueryBench(const VbtFile& file,
                      const std::vector<Query4D>& queries,
                      const std::string& shaderPath,
                      GpuQueryBenchStats& outStats);

bool runDenseFrameCacheGpuBench(const VbtFile& file,
                                const std::vector<Query4D>& queries,
                                const std::string& shaderPath,
                                size_t maxDenseBytes,
                                GpuQueryBenchStats& outStats);

} // namespace vbt::render
