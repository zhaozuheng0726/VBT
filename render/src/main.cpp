#include "gpu_query_bench.h"
#include "query_patterns.h"
#include "vbt_file.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

namespace {

using namespace vbt::render;

void printUsage()
{
    std::cout
        << "Usage: vbt_query_bench --input-vbt <file> [--batch-size N] [--pattern random|same-t|same-xyz|coherent]\n"
        << "                       [--seed N] [--sort-by-leaf] [--gpu-bench]\n"
        << "                       [--bench-mode vbt|dense|compare] [--dense-max-mb N]\n"
        << "  --gpu-bench is shorthand for --bench-mode vbt.\n"
        << "  compare runs dense-predecoded baseline, VBT unsorted, and VBT sorted-by-leaf.\n";
}

std::string modeName(ScientificMode mode)
{
    switch (mode) {
    case ScientificMode::CoarseOnly: return "CoarseOnly";
    case ScientificMode::SparseEvents: return "SparseEvents";
    case ScientificMode::DenseGrid3: return "DenseGrid3";
    case ScientificMode::DenseGrid4: return "DenseGrid4";
    default: return "Unknown";
    }
}

} // namespace

int main(int argc, char** argv)
{
    std::string inputPath;
    uint32_t batchSize = 65536;
    uint32_t seed = 1;
    bool sortByLeaf = false;
    bool gpuBench = false;
    GpuBenchMode benchMode = GpuBenchMode::Vbt;
    uint32_t denseMaxMb = 1536;
    QueryPattern pattern = QueryPattern::RandomFull;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input-vbt" && i + 1 < argc) {
            inputPath = argv[++i];
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batchSize = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--pattern" && i + 1 < argc) {
            const std::string text = argv[++i];
            if (!parseQueryPattern(text, pattern)) {
                std::cerr << "Unknown pattern: " << text << "\n";
                printUsage();
                return 1;
            }
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--sort-by-leaf") {
            sortByLeaf = true;
        } else if (arg == "--gpu-bench") {
            gpuBench = true;
            benchMode = GpuBenchMode::Vbt;
        } else if (arg == "--bench-mode" && i + 1 < argc) {
            const std::string text = argv[++i];
            if (text == "vbt") benchMode = GpuBenchMode::Vbt;
            else if (text == "dense") benchMode = GpuBenchMode::Dense;
            else if (text == "compare") benchMode = GpuBenchMode::Compare;
            else {
                std::cerr << "Unknown bench mode: " << text << "\n";
                printUsage();
                return 1;
            }
            gpuBench = true;
        } else if (arg == "--dense-max-mb" && i + 1 < argc) {
            denseMaxMb = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage();
            return 1;
        }
    }

    if (inputPath.empty()) {
        printUsage();
        return 1;
    }

    VbtFile file;
    std::string error;
    if (!loadVbtFile(inputPath, file, error)) {
        std::cerr << error << "\n";
        return 1;
    }

    std::cout << "Loaded " << inputPath << "\n";
    std::cout << "  magic: " << std::string(file.header.magic, file.header.magic + 8) << "\n";
    std::cout << "  version: " << file.header.version << "\n";
    std::cout << "  dims: " << file.header.width << " x " << file.header.height << " x " << file.header.depth
              << " x " << file.header.frames << "\n";
    std::cout << "  leafSize: " << file.header.leafSize << "\n";
    std::cout << "  leafCount: " << file.header.leafCount << "\n";
    std::cout << "  coarseAcScales: " << file.header.coarseAcScaleCount << "\n";
    std::cout << "  payloadWords: " << file.payloadWords.size() << "\n";

    auto queries = generateQueries(file.header, pattern, batchSize, seed);
    fillLeafIndices(file.header, queries);

    if (sortByLeaf) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        sortQueriesByLeaf(queries);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Sorted queries by leaf in " << ms << " ms\n";
    }

    std::unordered_map<uint32_t, uint64_t> leafHistogram;
    std::unordered_map<std::string, uint64_t> modeHistogram;
    uint64_t totalSparseEvents = 0;

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (const auto& q : queries) {
        ++leafHistogram[q.leafIndex];
        const uint32_t packedHeader = readPackedHeaderWord(file, q.leafIndex);
        const auto decoded = decodeScientificHeaderV2(packedHeader);
        ++modeHistogram[modeName(decoded.mode)];
        if (decoded.mode == ScientificMode::SparseEvents) {
            totalSparseEvents += decoded.packedEventCount;
        }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    uint64_t maxQueriesPerLeaf = 0;
    for (const auto& kv : leafHistogram) {
        maxQueriesPerLeaf = std::max(maxQueriesPerLeaf, kv.second);
    }

    std::cout << "Query preparation summary\n";
    std::cout << "  batchSize: " << batchSize << "\n";
    std::cout << "  uniqueLeaves: " << leafHistogram.size() << "\n";
    std::cout << "  maxQueriesPerLeaf: " << maxQueriesPerLeaf << "\n";
    std::cout << "  inspectionTimeMs: " << ms << "\n";
    std::cout << "  avgNsPerQuery(host-inspect): " << (ms * 1.0e6 / static_cast<double>(std::max<uint32_t>(1u, batchSize))) << "\n";
    std::cout << "  avgSparseEventCountOnHits: "
              << ((modeHistogram["SparseEvents"] > 0)
                      ? (static_cast<double>(totalSparseEvents) / static_cast<double>(modeHistogram["SparseEvents"]))
                      : 0.0)
              << "\n";

    std::cout << "Mode histogram\n";
    std::cout << "  CoarseOnly: " << modeHistogram["CoarseOnly"] << "\n";
    std::cout << "  SparseEvents: " << modeHistogram["SparseEvents"] << "\n";
    std::cout << "  DenseGrid3: " << modeHistogram["DenseGrid3"] << "\n";
    std::cout << "  DenseGrid4: " << modeHistogram["DenseGrid4"] << "\n";

    if (gpuBench) {
        GpuQueryBenchStats gpuStats;
        auto printBenchStats = [](const std::string& name, const GpuQueryBenchStats& stats) {
            std::cout << "\n" << name << "\n";
            std::cout << "  uploadMs: " << stats.uploadMs << "\n";
            std::cout << "  gpuDispatchMs: " << stats.gpuDispatchMs << "\n";
            std::cout << "  readbackMs: " << stats.readbackMs << "\n";
            std::cout << "  endToEndMs: " << stats.endToEndMs << "\n";
            std::cout << "  queriesPerSec: " << stats.queriesPerSec << "\n";
            std::cout << "  mismatchCount: " << stats.mismatchCount << "\n";
        };

        if (benchMode == GpuBenchMode::Vbt) {
            if (!runGpuQueryBench(file, queries, VBT_QUERY_SHADER_SPV_PATH, gpuStats)) {
                std::cerr << "\nGPU benchmark failed: " << gpuStats.error << "\n";
                return 2;
            }
            printBenchStats("GPU benchmark (VBT)", gpuStats);
        } else if (benchMode == GpuBenchMode::Dense) {
            if (!runDenseFrameCacheGpuBench(file,
                                            queries,
                                            VBT_DENSE_QUERY_SHADER_SPV_PATH,
                                            static_cast<size_t>(denseMaxMb) * 1024ull * 1024ull,
                                            gpuStats)) {
                std::cerr << "\nDense GPU benchmark failed: " << gpuStats.error << "\n";
                return 3;
            }
            printBenchStats("GPU benchmark (dense predecoded frame cache)", gpuStats);
        } else {
            std::cout << "\nGPU benchmark comparison\n";

            GpuQueryBenchStats denseStats;
            if (!runDenseFrameCacheGpuBench(file,
                                            queries,
                                            VBT_DENSE_QUERY_SHADER_SPV_PATH,
                                            static_cast<size_t>(denseMaxMb) * 1024ull * 1024ull,
                                            denseStats)) {
                std::cout << "  dense baseline skipped: " << denseStats.error << "\n";
            } else {
                printBenchStats("Dense predecoded frame cache", denseStats);
            }

            if (!runGpuQueryBench(file, queries, VBT_QUERY_SHADER_SPV_PATH, gpuStats)) {
                std::cerr << "\nVBT unsorted benchmark failed: " << gpuStats.error << "\n";
                return 4;
            }
            printBenchStats("VBT unsorted", gpuStats);

            auto sortedQueries = queries;
            const auto sortStart = std::chrono::high_resolution_clock::now();
            sortQueriesByLeaf(sortedQueries);
            const auto sortEnd = std::chrono::high_resolution_clock::now();
            const double sortMs = std::chrono::duration<double, std::milli>(sortEnd - sortStart).count();
            std::cout << "\nSort-by-leaf overhead\n";
            std::cout << "  sortMs: " << sortMs << "\n";

            GpuQueryBenchStats sortedStats;
            if (!runGpuQueryBench(file, sortedQueries, VBT_QUERY_SHADER_SPV_PATH, sortedStats)) {
                std::cerr << "\nVBT sorted benchmark failed: " << sortedStats.error << "\n";
                return 5;
            }
            printBenchStats("VBT sorted-by-leaf", sortedStats);
        }
    } else {
        std::cout << "\nScaffold ready. Next step: upload queries and payload buffers to compute shader.\n";
    }

    return 0;
}
