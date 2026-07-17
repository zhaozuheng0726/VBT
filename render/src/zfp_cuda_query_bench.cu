#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "zfp.h"
#include "zfp/bitstream.h"

#include "../../3D/compare/zfp-develop/src/cuda_zfp/shared.h"
#include "../../3D/compare/zfp-develop/src/cuda_zfp/decode.cuh"

namespace {

struct QueryPoint {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t z = 0;
  uint32_t t = 0;
  uint32_t blockIndex = 0;
  uint32_t localIndex = 0;
};

enum class QueryPattern {
  RandomFull,
  SameT,
  SameXYZ,
  CoherentTiles,
};

enum class RawLayout {
  FrameMajor,
  TimeFastest,
};

struct BenchResult {
  double requestedRate = 0.0;
  double actualRate = 0.0;
  uint32_t dimX = 0;
  uint32_t dimY = 0;
  uint32_t dimZ = 0;
  uint32_t frames = 0;
  uint32_t queryCount = 0;
  uint32_t repeats = 0;
  uint32_t maxbits = 0;
  size_t compressedBytes = 0;
  size_t compressedWords = 0;
  double uploadMs = 0.0;
  double kernelMs = 0.0;
  double queriesPerSec = 0.0;
  double maxAbsDiff = 0.0;
  double meanAbsDiff = 0.0;
};

void checkCuda(cudaError_t err, const char* what)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

uint32_t parseUint(const char* text)
{
  return static_cast<uint32_t>(std::stoul(text));
}

double parseDouble(const char* text)
{
  return std::stod(text);
}

void printUsage()
{
  std::cout
      << "Usage: zfp_cuda_query_bench --input-raw <path> --dim-x N --dim-y N --dim-z N --frames N\n"
      << "                           [--raw-layout frame-major|time-fastest]\n"
      << "                           --rate R [--query-count N] [--seed N] [--repeats N]\n"
      << "                           [--pattern random|same-t|same-xyz|coherent] [--fixed-frame N]\n"
      << "                           [--json-out path]\n";
}

std::vector<float> loadRawSequence(const std::string& rawPath,
                                   uint32_t dimX,
                                   uint32_t dimY,
                                   uint32_t dimZ,
                                   uint32_t frames,
                                   RawLayout rawLayout)
{
  const size_t voxelsPerFrame = static_cast<size_t>(dimX) * dimY * dimZ;
  const size_t totalVoxels = voxelsPerFrame * frames;
  const size_t totalBytes = totalVoxels * sizeof(float);

  std::ifstream file(rawPath, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open raw file: " + rawPath);
  }
  const size_t fileBytes = static_cast<size_t>(file.tellg());
  if (fileBytes != totalBytes) {
    throw std::runtime_error("Raw file size mismatch: " + rawPath);
  }
  file.seekg(0, std::ios::beg);

  std::vector<float> source(totalVoxels);
  file.read(reinterpret_cast<char*>(source.data()), static_cast<std::streamsize>(totalBytes));
  if (!file) {
    throw std::runtime_error("Failed to read raw sequence: " + rawPath);
  }
  if (rawLayout == RawLayout::FrameMajor) return source;

  std::vector<float> canonical(totalVoxels);
  for (uint32_t t = 0; t < frames; ++t) {
    const size_t outputBase = static_cast<size_t>(t) * voxelsPerFrame;
    for (size_t spatial = 0; spatial < voxelsPerFrame; ++spatial) {
      canonical[outputBase + spatial] =
          source[spatial * static_cast<size_t>(frames) + t];
    }
  }
  return canonical;
}

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

std::vector<QueryPoint> makeQueries(uint32_t dimX,
                                    uint32_t dimY,
                                    uint32_t dimZ,
                                    uint32_t frames,
                                    QueryPattern pattern,
                                    uint32_t fixedFrame,
                                    uint32_t queryCount,
                                    uint32_t seed)
{
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> dx(0, dimX - 1);
  std::uniform_int_distribution<uint32_t> dy(0, dimY - 1);
  std::uniform_int_distribution<uint32_t> dz(0, dimZ - 1);
  std::uniform_int_distribution<uint32_t> dt(0, frames - 1);

  const uint32_t blocksX = (dimX + 3u) / 4u;
  const uint32_t blocksY = (dimY + 3u) / 4u;
  const uint32_t fixedX = dx(rng);
  const uint32_t fixedY = dy(rng);
  const uint32_t fixedZ = dz(rng);

  std::vector<QueryPoint> queries(queryCount);
  for (uint32_t i = 0; i < queryCount; ++i) {
    QueryPoint q{};
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
      q.t = fixedFrame;
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
      q.x = std::min(dimX - 1u, baseX + (i % 8u));
      q.y = std::min(dimY - 1u, baseY + ((i / 8u) % 8u));
      q.z = std::min(dimZ - 1u, baseZ + ((i / 64u) % 8u));
      q.t = fixedFrame;
      break;
    }
    }
    const uint32_t bx = q.x >> 2u;
    const uint32_t by = q.y >> 2u;
    const uint32_t bz = q.z >> 2u;
    q.blockIndex = bx + by * blocksX + bz * blocksX * blocksY;
    q.localIndex = ((q.z & 3u) << 4u) | ((q.y & 3u) << 2u) | (q.x & 3u);
    queries[i] = q;
  }
  return queries;
}

void sortQueriesByFrameBlock(std::vector<QueryPoint>& queries)
{
  std::stable_sort(queries.begin(), queries.end(), [](const QueryPoint& a, const QueryPoint& b) {
    if (a.t != b.t) return a.t < b.t;
    if (a.blockIndex != b.blockIndex) return a.blockIndex < b.blockIndex;
    if (a.z != b.z) return a.z < b.z;
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
  });
}

void gatherExpectedForFrame(std::vector<float>& values,
                            const std::vector<float>& frame,
                                  uint32_t dimX,
                                  uint32_t dimY,
                            uint32_t frameIndex,
                                  const std::vector<QueryPoint>& queries)
{
  for (size_t i = 0; i < queries.size(); ++i) {
    const auto& q = queries[i];
    if (q.t != frameIndex) continue;
    const size_t index = static_cast<size_t>(q.x) +
                         static_cast<size_t>(dimX) *
                             (static_cast<size_t>(q.y) + static_cast<size_t>(dimY) * q.z);
    values[i] = frame[index];
  }
}

std::vector<float> compressAndDecompressCpu(const std::vector<float>& frame,
                                            uint32_t dimX,
                                            uint32_t dimY,
                                            uint32_t dimZ,
                                            double requestedRate,
                                            BenchResult& result,
                                            std::vector<Word>& compressedWords)
{
  zfp_field* field = zfp_field_3d(const_cast<float*>(frame.data()), zfp_type_float, dimX, dimY, dimZ);
  if (!field) {
    throw std::runtime_error("zfp_field_3d failed");
  }

  zfp_stream* zfp = zfp_stream_open(nullptr);
  if (!zfp) {
    zfp_field_free(field);
    throw std::runtime_error("zfp_stream_open failed");
  }

  const double actualRate = zfp_stream_set_rate(zfp, requestedRate, zfp_type_float, 3, zfp_false);
  const size_t maxBytes = zfp_stream_maximum_size(zfp, field);
  std::vector<unsigned char> compressed(maxBytes);

  bitstream* stream = stream_open(compressed.data(), compressed.size());
  if (!stream) {
    zfp_stream_close(zfp);
    zfp_field_free(field);
    throw std::runtime_error("stream_open failed");
  }

  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  const size_t compressedBytes = zfp_compress(zfp, field);
  if (!compressedBytes) {
    stream_close(stream);
    zfp_stream_close(zfp);
    zfp_field_free(field);
    throw std::runtime_error("zfp_compress failed");
  }

  uint minbits = 0;
  uint maxbits = 0;
  uint maxprec = 0;
  int minexp = 0;
  zfp_stream_params(zfp, &minbits, &maxbits, &maxprec, &minexp);

  compressed.resize(compressedBytes);
  compressedWords.assign((compressedBytes + sizeof(Word) - 1) / sizeof(Word), 0ull);
  std::memcpy(compressedWords.data(), compressed.data(), compressedBytes);

  std::vector<float> decoded(frame.size(), 0.0f);
  zfp_field_set_pointer(field, decoded.data());
  zfp_stream_rewind(zfp);
  if (!zfp_decompress(zfp, field)) {
    stream_close(stream);
    zfp_stream_close(zfp);
    zfp_field_free(field);
    throw std::runtime_error("zfp_decompress failed");
  }

  result.requestedRate = requestedRate;
  result.actualRate = actualRate;
  result.maxbits = maxbits;
  result.compressedBytes = compressedBytes;
  result.compressedWords = compressedWords.size();

  stream_close(stream);
  zfp_stream_close(zfp);
  zfp_field_free(field);
  return decoded;
}

__global__ void zfpQueryKernel(const Word* blocks,
                               uint32_t blocksPerFrame,
                               uint32_t maxbits,
                               const QueryPoint* queries,
                               uint32_t queryCount,
                               float* outValues)
{
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= queryCount) {
    return;
  }

  const QueryPoint q = queries[tid];
  float decoded[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    decoded[i] = 0.0f;
  }

  cuZFP::BlockReader<64> reader(const_cast<Word*>(blocks),
                                static_cast<int>(maxbits),
                                static_cast<int>(q.t * blocksPerFrame + q.blockIndex),
                                static_cast<int>(blocksPerFrame));
  cuZFP::zfp_decode<float, 64>(reader, decoded, maxbits);
  outValues[tid] = decoded[q.localIndex];
}

BenchResult runGpuBench(const std::vector<Word>& compressedWords,
                        uint32_t blocksPerFrame,
                        uint32_t dimX,
                        uint32_t dimY,
                        uint32_t dimZ,
                        uint32_t frames,
                        const std::vector<QueryPoint>& queries,
                        const std::vector<float>& expected,
                        double requestedRate,
                        double actualRate,
                        uint32_t maxbits,
                        uint32_t repeats)
{
  BenchResult result{};
  result.requestedRate = requestedRate;
  result.actualRate = actualRate;
  result.dimX = dimX;
  result.dimY = dimY;
  result.dimZ = dimZ;
  result.frames = frames;
  result.queryCount = static_cast<uint32_t>(queries.size());
  result.repeats = repeats;
  result.maxbits = maxbits;
  result.compressedBytes = compressedWords.size() * sizeof(Word);
  result.compressedWords = compressedWords.size();

  Word* dBlocks = nullptr;
  QueryPoint* dQueries = nullptr;
  float* dOut = nullptr;

  const size_t blockBytes = compressedWords.size() * sizeof(Word);
  const size_t queryBytes = queries.size() * sizeof(QueryPoint);
  const size_t outBytes = queries.size() * sizeof(float);

  checkCuda(cudaFree(nullptr), "cudaFree warmup");
  auto uploadStart = std::chrono::high_resolution_clock::now();
  checkCuda(cudaMalloc(&dBlocks, blockBytes), "cudaMalloc blocks");
  checkCuda(cudaMalloc(&dQueries, queryBytes), "cudaMalloc queries");
  checkCuda(cudaMalloc(&dOut, outBytes), "cudaMalloc outputs");
  checkCuda(cudaMemcpy(dBlocks, compressedWords.data(), blockBytes, cudaMemcpyHostToDevice), "cudaMemcpy blocks");
  checkCuda(cudaMemcpy(dQueries, queries.data(), queryBytes, cudaMemcpyHostToDevice), "cudaMemcpy queries");
  checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after upload");
  auto uploadEnd = std::chrono::high_resolution_clock::now();
  result.uploadMs = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();

  const uint32_t threads = 128u;
  const uint32_t blocks = (result.queryCount + threads - 1u) / threads;

  for (int i = 0; i < 3; ++i) {
    zfpQueryKernel<<<blocks, threads>>>(dBlocks, blocksPerFrame, maxbits, dQueries, result.queryCount, dOut);
  }
  checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warmup");

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
  checkCuda(cudaEventRecord(start), "cudaEventRecord start");
  for (uint32_t i = 0; i < repeats; ++i) {
    zfpQueryKernel<<<blocks, threads>>>(dBlocks, blocksPerFrame, maxbits, dQueries, result.queryCount, dOut);
  }
  checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
  checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

  float kernelMs = 0.0f;
  checkCuda(cudaEventElapsedTime(&kernelMs, start, stop), "cudaEventElapsedTime");
  result.kernelMs = kernelMs;
  result.queriesPerSec =
      (static_cast<double>(result.queryCount) * static_cast<double>(repeats)) / (static_cast<double>(kernelMs) * 1.0e-3);

  std::vector<float> out(result.queryCount, 0.0f);
  checkCuda(cudaMemcpy(out.data(), dOut, outBytes, cudaMemcpyDeviceToHost), "cudaMemcpy outputs");
  checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after readback");

  double sumAbs = 0.0;
  double maxAbs = 0.0;
  for (size_t i = 0; i < out.size(); ++i) {
    const double diff = std::abs(static_cast<double>(out[i]) - static_cast<double>(expected[i]));
    sumAbs += diff;
    maxAbs = std::max(maxAbs, diff);
  }
  result.meanAbsDiff = sumAbs / std::max<size_t>(1, out.size());
  result.maxAbsDiff = maxAbs;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dBlocks);
  cudaFree(dQueries);
  cudaFree(dOut);
  return result;
}

void writeJson(const std::string& path,
               RawLayout rawLayout,
               const std::string& patternName,
               const BenchResult& unsorted,
               const BenchResult& sorted)
{
  std::filesystem::path outPath(path);
  if (!outPath.parent_path().empty()) {
    std::filesystem::create_directories(outPath.parent_path());
  }
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to open json output: " + path);
  }
  out << "{\n";
  out << "  \"dataset\": \"zfp_cuda_query\",\n";
  out << "  \"raw_layout\": \""
      << (rawLayout == RawLayout::TimeFastest ? "time-fastest" : "frame-major")
      << "\",\n";
  out << "  \"dim_x\": " << unsorted.dimX << ",\n";
  out << "  \"dim_y\": " << unsorted.dimY << ",\n";
  out << "  \"dim_z\": " << unsorted.dimZ << ",\n";
  out << "  \"frames\": " << unsorted.frames << ",\n";
  out << "  \"pattern\": \"" << patternName << "\",\n";
  out << "  \"requested_rate\": " << unsorted.requestedRate << ",\n";
  out << "  \"actual_rate\": " << unsorted.actualRate << ",\n";
  out << "  \"compressed_bytes\": " << unsorted.compressedBytes << ",\n";
  out << "  \"maxbits\": " << unsorted.maxbits << ",\n";
  out << "  \"query_count\": " << unsorted.queryCount << ",\n";
  out << "  \"repeats\": " << unsorted.repeats << ",\n";
  out << "  \"unsorted\": {\n";
  out << "    \"upload_ms\": " << unsorted.uploadMs << ",\n";
  out << "    \"kernel_ms\": " << unsorted.kernelMs << ",\n";
  out << "    \"queries_per_sec\": " << unsorted.queriesPerSec << ",\n";
  out << "    \"mean_abs_diff\": " << unsorted.meanAbsDiff << ",\n";
  out << "    \"max_abs_diff\": " << unsorted.maxAbsDiff << "\n";
  out << "  },\n";
  out << "  \"sorted\": {\n";
  out << "    \"upload_ms\": " << sorted.uploadMs << ",\n";
  out << "    \"kernel_ms\": " << sorted.kernelMs << ",\n";
  out << "    \"queries_per_sec\": " << sorted.queriesPerSec << ",\n";
  out << "    \"mean_abs_diff\": " << sorted.meanAbsDiff << ",\n";
  out << "    \"max_abs_diff\": " << sorted.maxAbsDiff << "\n";
  out << "  }\n";
  out << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
  std::string inputRaw;
  std::string jsonOut;
  uint32_t dimX = 0;
  uint32_t dimY = 0;
  uint32_t dimZ = 0;
  uint32_t frames = 0;
  uint32_t fixedFrame = 64;
  uint32_t queryCount = 65536;
  uint32_t seed = 1;
  uint32_t repeats = 100;
  double rate = 0.5;
  QueryPattern pattern = QueryPattern::RandomFull;
  RawLayout rawLayout = RawLayout::FrameMajor;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input-raw" && i + 1 < argc) inputRaw = argv[++i];
    else if (arg == "--dim-x" && i + 1 < argc) dimX = parseUint(argv[++i]);
    else if (arg == "--dim-y" && i + 1 < argc) dimY = parseUint(argv[++i]);
    else if (arg == "--dim-z" && i + 1 < argc) dimZ = parseUint(argv[++i]);
    else if (arg == "--frames" && i + 1 < argc) frames = parseUint(argv[++i]);
    else if (arg == "--fixed-frame" && i + 1 < argc) fixedFrame = parseUint(argv[++i]);
    else if (arg == "--raw-layout" && i + 1 < argc) {
      const std::string text = argv[++i];
      if (text == "frame-major") rawLayout = RawLayout::FrameMajor;
      else if (text == "time-fastest") rawLayout = RawLayout::TimeFastest;
      else {
        printUsage();
        return 1;
      }
    }
    else if (arg == "--pattern" && i + 1 < argc) {
      if (!parseQueryPattern(argv[++i], pattern)) {
        printUsage();
        return 1;
      }
    }
    else if (arg == "--rate" && i + 1 < argc) rate = parseDouble(argv[++i]);
    else if (arg == "--query-count" && i + 1 < argc) queryCount = parseUint(argv[++i]);
    else if (arg == "--seed" && i + 1 < argc) seed = parseUint(argv[++i]);
    else if (arg == "--repeats" && i + 1 < argc) repeats = parseUint(argv[++i]);
    else if (arg == "--json-out" && i + 1 < argc) jsonOut = argv[++i];
    else {
      printUsage();
      return 1;
    }
  }

  if (inputRaw.empty() || !dimX || !dimY || !dimZ || !frames || fixedFrame >= frames) {
    printUsage();
    return 1;
  }

  try {
    auto unsortedQueries = makeQueries(dimX, dimY, dimZ, frames, pattern, fixedFrame, queryCount, seed);
    auto sortedQueries = unsortedQueries;
    sortQueriesByFrameBlock(sortedQueries);
    std::vector<float> expectedUnsorted(queryCount, 0.0f);
    std::vector<float> expectedSorted(queryCount, 0.0f);

    BenchResult base{};
    std::vector<Word> compressedWords;
    std::vector<Word> allCompressedWords;
    uint32_t blocksPerFrame = 0;
    size_t wordsPerFrame = 0;
    auto rawSequence = loadRawSequence(inputRaw, dimX, dimY, dimZ, frames, rawLayout);
    const size_t voxelsPerFrame = static_cast<size_t>(dimX) * dimY * dimZ;
    for (uint32_t frameIndex = 0; frameIndex < frames; ++frameIndex) {
      const auto frameBegin =
          rawSequence.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(frameIndex) * voxelsPerFrame);
      std::vector<float> originalFrame(frameBegin,
                                       frameBegin + static_cast<std::ptrdiff_t>(voxelsPerFrame));
      BenchResult frameBase{};
      compressedWords.clear();
      auto cpuDecoded = compressAndDecompressCpu(originalFrame, dimX, dimY, dimZ, rate, frameBase, compressedWords);
      if (frameIndex == 0) {
        base = frameBase;
        blocksPerFrame = ((dimX + 3u) & ~3u) * ((dimY + 3u) & ~3u) * ((dimZ + 3u) & ~3u) / 64u;
        wordsPerFrame = compressedWords.size();
        allCompressedWords.reserve(wordsPerFrame * static_cast<size_t>(frames));
      } else {
        if (frameBase.maxbits != base.maxbits || compressedWords.size() != wordsPerFrame) {
          throw std::runtime_error("Per-frame zfp stream layout mismatch across frames.");
        }
      }
      allCompressedWords.insert(allCompressedWords.end(), compressedWords.begin(), compressedWords.end());
      gatherExpectedForFrame(expectedUnsorted, cpuDecoded, dimX, dimY, frameIndex, unsortedQueries);
      gatherExpectedForFrame(expectedSorted, cpuDecoded, dimX, dimY, frameIndex, sortedQueries);
    }

    auto unsorted = runGpuBench(allCompressedWords,
                                blocksPerFrame,
                                dimX,
                                dimY,
                                dimZ,
                                frames,
                                unsortedQueries,
                                expectedUnsorted,
                                base.requestedRate,
                                base.actualRate,
                                base.maxbits,
                                repeats);
    auto sorted = runGpuBench(allCompressedWords,
                              blocksPerFrame,
                              dimX,
                              dimY,
                              dimZ,
                              frames,
                              sortedQueries,
                              expectedSorted,
                              base.requestedRate,
                              base.actualRate,
                              base.maxbits,
                              repeats);

    const std::string patternName = (pattern == QueryPattern::RandomFull ? "random" :
                                     pattern == QueryPattern::SameT ? "same-t" :
                                     pattern == QueryPattern::SameXYZ ? "same-xyz" : "coherent");
    std::cout << "zfp CUDA query benchmark\n";
    std::cout << "  dims: " << dimX << " x " << dimY << " x " << dimZ << "\n";
    std::cout << "  frames: " << frames << "\n";
    std::cout << "  rawLayout: "
              << (rawLayout == RawLayout::TimeFastest ? "time-fastest" : "frame-major")
              << "\n";
    std::cout << "  pattern: " << patternName << "\n";
    std::cout << "  requestedRate: " << base.requestedRate << "\n";
    std::cout << "  actualRate: " << base.actualRate << "\n";
    std::cout << "  compressedBytes: " << base.compressedBytes << "\n";
    std::cout << "  maxbits: " << base.maxbits << "\n";
    std::cout << "  queryCount: " << queryCount << "\n";
    std::cout << "  repeats: " << repeats << "\n";
    std::cout << "\nUnsorted\n";
    std::cout << "  uploadMs: " << unsorted.uploadMs << "\n";
    std::cout << "  kernelMs: " << unsorted.kernelMs << "\n";
    std::cout << "  queriesPerSec: " << unsorted.queriesPerSec << "\n";
    std::cout << "  meanAbsDiff(vs cpu zfp): " << unsorted.meanAbsDiff << "\n";
    std::cout << "  maxAbsDiff(vs cpu zfp): " << unsorted.maxAbsDiff << "\n";
    std::cout << "\nSortedByBlock\n";
    std::cout << "  uploadMs: " << sorted.uploadMs << "\n";
    std::cout << "  kernelMs: " << sorted.kernelMs << "\n";
    std::cout << "  queriesPerSec: " << sorted.queriesPerSec << "\n";
    std::cout << "  meanAbsDiff(vs cpu zfp): " << sorted.meanAbsDiff << "\n";
    std::cout << "  maxAbsDiff(vs cpu zfp): " << sorted.maxAbsDiff << "\n";

    if (!jsonOut.empty()) {
      writeJson(jsonOut, rawLayout, patternName, unsorted, sorted);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "zfp_cuda_query_bench failed: " << e.what() << "\n";
    return 2;
  }
}
