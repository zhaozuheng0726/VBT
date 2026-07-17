#include "frame_metadata.h"
#include "query_patterns.h"
#include "scientific_decode.h"
#include "vbt_file.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <zfp/array4.hpp>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace {

using namespace vbt::render;

struct RawMap {
    const float* data = nullptr;
    size_t valueCount = 0;
#ifdef _WIN32
    HANDLE fileHandle = INVALID_HANDLE_VALUE;
    HANDLE mappingHandle = nullptr;
#endif
};

void printUsage()
{
    std::cout
        << "Usage: vbt_cpu_query_compare --input-vbt <file> --input-raw <raw> [--metadata <json>]\n"
        << "                             [--zfp-rate R] [--zfp-cache-mb N]\n"
        << "                             [--batch-size N] [--pattern random|same-t|same-xyz|coherent]\n"
        << "                             [--seed N]\n";
}

#ifdef _WIN32
bool mapRawFile(const std::string& path, size_t expectedBytes, RawMap& outMap, std::string& error)
{
    outMap = {};
    const HANDLE fileHandle = CreateFileA(path.c_str(),
                                          GENERIC_READ,
                                          FILE_SHARE_READ,
                                          nullptr,
                                          OPEN_EXISTING,
                                          FILE_ATTRIBUTE_NORMAL,
                                          nullptr);
    if (fileHandle == INVALID_HANDLE_VALUE) {
        error = "Failed to open raw file: " + path;
        return false;
    }

    LARGE_INTEGER fileSize{};
    if (!GetFileSizeEx(fileHandle, &fileSize)) {
        CloseHandle(fileHandle);
        error = "Failed to read raw file size: " + path;
        return false;
    }
    if (static_cast<uint64_t>(fileSize.QuadPart) != expectedBytes) {
        CloseHandle(fileHandle);
        error = "Raw file size does not match the expected byte count.";
        return false;
    }

    const HANDLE mappingHandle = CreateFileMappingA(fileHandle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mappingHandle) {
        CloseHandle(fileHandle);
        error = "Failed to create raw file mapping.";
        return false;
    }

    const void* view = MapViewOfFile(mappingHandle, FILE_MAP_READ, 0, 0, expectedBytes);
    if (!view) {
        CloseHandle(mappingHandle);
        CloseHandle(fileHandle);
        error = "Failed to map raw file view.";
        return false;
    }

    outMap.data = static_cast<const float*>(view);
    outMap.valueCount = expectedBytes / sizeof(float);
    outMap.fileHandle = fileHandle;
    outMap.mappingHandle = mappingHandle;
    return true;
}

void unmapRawFile(RawMap& map)
{
    if (map.data) {
        UnmapViewOfFile(map.data);
        map.data = nullptr;
    }
    if (map.mappingHandle) {
        CloseHandle(map.mappingHandle);
        map.mappingHandle = nullptr;
    }
    if (map.fileHandle != INVALID_HANDLE_VALUE) {
        CloseHandle(map.fileHandle);
        map.fileHandle = INVALID_HANDLE_VALUE;
    }
    map.valueCount = 0;
}
#else
bool mapRawFile(const std::string&, size_t, RawMap&, std::string& error)
{
    error = "Raw mapping is only implemented for Windows in this benchmark.";
    return false;
}

void unmapRawFile(RawMap&) {}
#endif

size_t rawLinearIndex(const VbtFileHeader& header,
                      const vbt::FrameMetadata& metadata,
                      uint32_t x,
                      uint32_t y,
                      uint32_t z,
                      uint32_t t)
{
    const uint64_t spatial =
        ((static_cast<uint64_t>(z) * header.height) + y) * header.width + x;
    if (metadata.rawStorageOrder == vbt::RawStorageOrder::TimeFastest) {
        return static_cast<size_t>(spatial * header.frames + t);
    }
    const uint64_t frameVoxels =
        static_cast<uint64_t>(header.width) * header.height * header.depth;
    return static_cast<size_t>(static_cast<uint64_t>(t) * frameVoxels + spatial);
}

struct CpuBenchStats {
    double ms = 0.0;
    double queriesPerSec = 0.0;
    double avgAbsError = 0.0;
    double rmse = 0.0;
    double checksum = 0.0;
};

template <class Fn>
CpuBenchStats runCpuBench(const std::vector<Query4D>& queries,
                          const VbtFileHeader& header,
                          const vbt::FrameMetadata& metadata,
                          const float* truth,
                          Fn&& eval)
{
    CpuBenchStats stats;
    double sumAbs = 0.0;
    double sumSq = 0.0;
    volatile double checksum = 0.0;

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (const auto& q : queries) {
        const float value = eval(q);
        const float gt = truth[rawLinearIndex(header, metadata, q.x, q.y, q.z, q.t)];
        const double diff = static_cast<double>(value) - static_cast<double>(gt);
        sumAbs += std::abs(diff);
        sumSq += diff * diff;
        checksum += static_cast<double>(value);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    stats.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats.queriesPerSec = queries.empty() ? 0.0 : static_cast<double>(queries.size()) / (stats.ms * 1.0e-3);
    stats.avgAbsError = queries.empty() ? 0.0 : sumAbs / static_cast<double>(queries.size());
    stats.rmse = queries.empty() ? 0.0 : std::sqrt(sumSq / static_cast<double>(queries.size()));
    stats.checksum = checksum;
    return stats;
}

void printStats(const std::string& name, const CpuBenchStats& stats)
{
    std::cout << "\n" << name << "\n";
    std::cout << "  ms: " << stats.ms << "\n";
    std::cout << "  queriesPerSec: " << stats.queriesPerSec << "\n";
    std::cout << "  avgAbsError: " << stats.avgAbsError << "\n";
    std::cout << "  rmse: " << stats.rmse << "\n";
    std::cout << "  checksum: " << stats.checksum << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    std::string inputVbt;
    std::string inputRaw;
    std::filesystem::path metadataPath;
    uint32_t batchSize = 65536;
    uint32_t seed = 1;
    uint32_t zfpCacheMb = 64;
    double zfpRate = 0.5;
    QueryPattern pattern = QueryPattern::SameT;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input-vbt" && i + 1 < argc) {
            inputVbt = argv[++i];
        } else if (arg == "--input-raw" && i + 1 < argc) {
            inputRaw = argv[++i];
        } else if (arg == "--metadata" && i + 1 < argc) {
            metadataPath = argv[++i];
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
        } else if (arg == "--zfp-rate" && i + 1 < argc) {
            zfpRate = std::stod(argv[++i]);
        } else if (arg == "--zfp-cache-mb" && i + 1 < argc) {
            zfpCacheMb = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage();
            return 1;
        }
    }

    if (inputVbt.empty() || inputRaw.empty()) {
        printUsage();
        return 1;
    }

    VbtFile file;
    std::string error;
    if (!loadVbtFile(inputVbt, file, error)) {
        std::cerr << error << "\n";
        return 2;
    }

    if (metadataPath.empty()) {
        metadataPath = std::filesystem::path(inputRaw);
        metadataPath.replace_extension(".metadata.json");
    }
    if (!std::filesystem::exists(metadataPath)) {
        std::cerr << "Raw metadata is required to establish storage order: "
                  << metadataPath.string() << "\n";
        return 3;
    }
    vbt::FrameMetadata metadata;
    try {
        metadata = vbt::loadFrameMetadata(metadataPath);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 3;
    }
    if (metadata.width != static_cast<int>(file.header.width) ||
        metadata.height != static_cast<int>(file.header.height) ||
        metadata.depth != static_cast<int>(file.header.depth) ||
        metadata.frames != static_cast<int>(file.header.frames)) {
        std::cerr << "Raw metadata dimensions do not match VBTPACK4 header.\n";
        return 3;
    }

    const uint64_t voxelCount64 = static_cast<uint64_t>(file.header.width) *
                                  static_cast<uint64_t>(file.header.height) *
                                  static_cast<uint64_t>(file.header.depth) *
                                  static_cast<uint64_t>(file.header.frames);
    if (voxelCount64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        std::cerr << "Volume too large for host index space.\n";
        return 3;
    }
    const size_t voxelCount = static_cast<size_t>(voxelCount64);
    const size_t rawBytes = voxelCount * sizeof(float);

    RawMap rawMap;
    if (!mapRawFile(inputRaw, rawBytes, rawMap, error)) {
        std::cerr << error << "\n";
        return 4;
    }

    auto queries = generateQueries(file.header, pattern, batchSize, seed);
    fillLeafIndices(file.header, queries);
    auto sortedQueries = queries;
    sortQueriesByLeaf(sortedQueries);

    std::cout << "Loaded VBT: " << inputVbt << "\n";
    std::cout << "Mapped raw: " << inputRaw << "\n";
    std::cout << "  rawLayout: "
              << (metadata.rawStorageOrder == vbt::RawStorageOrder::TimeFastest
                      ? "time-fastest"
                      : "frame-major")
              << "\n";
    std::cout << "  dims: " << file.header.width << " x " << file.header.height << " x "
              << file.header.depth << " x " << file.header.frames << "\n";
    std::cout << "  batchSize: " << batchSize << "\n";
    std::cout << "  zfpRate(bits/value): " << zfpRate << "\n";
    std::cout << "  zfpCacheMb: " << zfpCacheMb << "\n";

    const auto zfpStart = std::chrono::high_resolution_clock::now();
    const bool timeFastest = metadata.rawStorageOrder == vbt::RawStorageOrder::TimeFastest;
    auto zfpArray = timeFastest
        ? std::make_unique<zfp::array4f>(file.header.frames,
                                        file.header.width,
                                        file.header.height,
                                        file.header.depth,
                                        zfpRate,
                                        rawMap.data,
                                        static_cast<size_t>(zfpCacheMb) * 1024ull * 1024ull)
        : std::make_unique<zfp::array4f>(file.header.width,
                                        file.header.height,
                                        file.header.depth,
                                        file.header.frames,
                                        zfpRate,
                                        rawMap.data,
                                        static_cast<size_t>(zfpCacheMb) * 1024ull * 1024ull);
    zfpArray->flush_cache();
    const auto zfpEnd = std::chrono::high_resolution_clock::now();
    const double zfpBuildMs = std::chrono::duration<double, std::milli>(zfpEnd - zfpStart).count();
    const double zfpCompressedMb = static_cast<double>(zfpArray->compressed_size()) / 1.0e6;

    std::cout << "  zfpBuildMs: " << zfpBuildMs << "\n";
    std::cout << "  zfpCompressedMb: " << zfpCompressedMb << "\n";

    const auto vbtUnsorted = runCpuBench(queries, file.header, metadata, rawMap.data, [&](const Query4D& q) {
        return decodeScientificValueAtCpu(file, q);
    });
    printStats("VBT CPU unsorted", vbtUnsorted);

    const auto vbtSorted = runCpuBench(sortedQueries, file.header, metadata, rawMap.data, [&](const Query4D& q) {
        return decodeScientificValueAtCpu(file, q);
    });
    printStats("VBT CPU sorted-by-leaf", vbtSorted);

    zfpArray->clear_cache();
    const auto zfpUnsorted = runCpuBench(queries, file.header, metadata, rawMap.data, [&](const Query4D& q) {
        return timeFastest
            ? static_cast<float>((*zfpArray)(q.t, q.x, q.y, q.z))
            : static_cast<float>((*zfpArray)(q.x, q.y, q.z, q.t));
    });
    printStats("zfp CPU random-access array4", zfpUnsorted);

    unmapRawFile(rawMap);
    return 0;
}
