#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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
  uint32_t dimX = 0;
  uint32_t dimY = 0;
  uint32_t dimZ = 0;
  uint32_t frames = 0;
  uint32_t queryCount = 0;
  uint32_t repeats = 0;
  size_t residentBytes = 0;
  size_t gpuFreeBytesBefore = 0;
  size_t gpuTotalBytes = 0;
  double uploadMs = 0.0;
  double kernelMs = 0.0;
  double queriesPerSec = 0.0;
  double meanAbsDiff = 0.0;
  double maxAbsDiff = 0.0;
  std::string error;
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

void printUsage()
{
  std::cout
      << "Usage: dense_cuda_query_bench --input-raw <path> --dim-x N --dim-y N --dim-z N --frames N\n"
      << "                             [--raw-layout frame-major|time-fastest]\n"
      << "                             [--query-count N] [--seed N] [--repeats N]\n"
      << "                             [--pattern random|same-t|same-xyz|coherent] [--fixed-frame N]\n"
      << "                             [--json-out path]\n";
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

std::vector<float> loadRawSequence(const std::string& rawPath,
                                   uint32_t dimX,
                                   uint32_t dimY,
                                   uint32_t dimZ,
                                   uint32_t frames)
{
  const size_t voxelCount = static_cast<size_t>(dimX) * dimY * dimZ * frames;
  const size_t totalBytes = voxelCount * sizeof(float);
  std::ifstream file(rawPath, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open raw file: " + rawPath);
  }
  const size_t fileBytes = static_cast<size_t>(file.tellg());
  if (fileBytes != totalBytes) {
    throw std::runtime_error("Raw file size mismatch: " + rawPath);
  }
  file.seekg(0, std::ios::beg);
  std::vector<float> data(voxelCount);
  file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(totalBytes));
  if (!file) {
    throw std::runtime_error("Failed to read raw sequence: " + rawPath);
  }
  return data;
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

std::vector<float> gatherExpected(const std::vector<float>& dense,
                                  uint32_t dimX,
                                  uint32_t dimY,
                                  uint32_t dimZ,
                                  uint32_t frames,
                                  RawLayout rawLayout,
                                  const std::vector<QueryPoint>& queries)
{
  const size_t voxelsPerFrame = static_cast<size_t>(dimX) * dimY * dimZ;
  std::vector<float> expected(queries.size(), 0.0f);
  for (size_t i = 0; i < queries.size(); ++i) {
    const auto& q = queries[i];
    const size_t linear = static_cast<size_t>(q.x) +
                          static_cast<size_t>(dimX) *
                              (static_cast<size_t>(q.y) + static_cast<size_t>(dimY) * q.z);
    const size_t index = rawLayout == RawLayout::TimeFastest
        ? linear * static_cast<size_t>(frames) + q.t
        : static_cast<size_t>(q.t) * voxelsPerFrame + linear;
    expected[i] = dense[index];
  }
  return expected;
}

__global__ void denseQueryKernel(const float* dense,
                                 uint32_t dimX,
                                 uint32_t dimY,
                                 uint32_t dimZ,
                                 uint32_t voxelsPerFrame,
                                 uint32_t frames,
                                 bool timeFastest,
                                 const QueryPoint* queries,
                                 uint32_t queryCount,
                                 float* outValues)
{
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= queryCount) return;
  const QueryPoint q = queries[tid];
  const uint32_t linear = q.x + dimX * (q.y + dimY * q.z);
  const size_t index = timeFastest
      ? static_cast<size_t>(linear) * frames + q.t
      : static_cast<size_t>(q.t) * voxelsPerFrame + linear;
  outValues[tid] = dense[index];
}

BenchResult runGpuBench(const std::vector<float>& dense,
                        uint32_t dimX,
                        uint32_t dimY,
                        uint32_t dimZ,
                        uint32_t frames,
                        RawLayout rawLayout,
                        const std::vector<QueryPoint>& queries,
                        const std::vector<float>& expected,
                        uint32_t repeats)
{
  BenchResult result{};
  result.dimX = dimX;
  result.dimY = dimY;
  result.dimZ = dimZ;
  result.frames = frames;
  result.queryCount = static_cast<uint32_t>(queries.size());
  result.repeats = repeats;
  result.residentBytes = dense.size() * sizeof(float);

  const uint32_t voxelsPerFrame = dimX * dimY * dimZ;
  const size_t denseBytes = dense.size() * sizeof(float);
  const size_t queryBytes = queries.size() * sizeof(QueryPoint);
  const size_t outBytes = queries.size() * sizeof(float);

  size_t freeBytes = 0;
  size_t totalBytes = 0;
  checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes), "cudaMemGetInfo");
  result.gpuFreeBytesBefore = freeBytes;
  result.gpuTotalBytes = totalBytes;

  float* dDense = nullptr;
  QueryPoint* dQueries = nullptr;
  float* dOut = nullptr;

  checkCuda(cudaFree(nullptr), "cudaFree warmup");
  auto uploadStart = std::chrono::high_resolution_clock::now();
  checkCuda(cudaMalloc(&dDense, denseBytes), "cudaMalloc dense");
  checkCuda(cudaMalloc(&dQueries, queryBytes), "cudaMalloc queries");
  checkCuda(cudaMalloc(&dOut, outBytes), "cudaMalloc outputs");
  checkCuda(cudaMemcpy(dDense, dense.data(), denseBytes, cudaMemcpyHostToDevice), "cudaMemcpy dense");
  checkCuda(cudaMemcpy(dQueries, queries.data(), queryBytes, cudaMemcpyHostToDevice), "cudaMemcpy queries");
  checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after upload");
  auto uploadEnd = std::chrono::high_resolution_clock::now();
  result.uploadMs = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();

  const uint32_t threads = 128u;
  const uint32_t blocks = (result.queryCount + threads - 1u) / threads;

  for (int i = 0; i < 3; ++i) {
    denseQueryKernel<<<blocks, threads>>>(dDense,
                                         dimX,
                                         dimY,
                                         dimZ,
                                         voxelsPerFrame,
                                         frames,
                                         rawLayout == RawLayout::TimeFastest,
                                         dQueries,
                                         result.queryCount,
                                         dOut);
  }
  checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warmup");

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
  checkCuda(cudaEventRecord(start), "cudaEventRecord start");
  for (uint32_t i = 0; i < repeats; ++i) {
    denseQueryKernel<<<blocks, threads>>>(dDense,
                                         dimX,
                                         dimY,
                                         dimZ,
                                         voxelsPerFrame,
                                         frames,
                                         rawLayout == RawLayout::TimeFastest,
                                         dQueries,
                                         result.queryCount,
                                         dOut);
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
  cudaFree(dDense);
  cudaFree(dQueries);
  cudaFree(dOut);
  return result;
}

void writeJson(const std::string& path,
               const std::string& inputRaw,
               RawLayout rawLayout,
               const std::string& patternName,
               const BenchResult& unsorted,
               const BenchResult* sorted)
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
  out << "  \"dataset\": \"dense_cuda_query\",\n";
  out << "  \"input_raw\": \"" << inputRaw << "\",\n";
  out << "  \"raw_layout\": \""
      << (rawLayout == RawLayout::TimeFastest ? "time-fastest" : "frame-major")
      << "\",\n";
  out << "  \"dim_x\": " << unsorted.dimX << ",\n";
  out << "  \"dim_y\": " << unsorted.dimY << ",\n";
  out << "  \"dim_z\": " << unsorted.dimZ << ",\n";
  out << "  \"frames\": " << unsorted.frames << ",\n";
  out << "  \"pattern\": \"" << patternName << "\",\n";
  out << "  \"resident_bytes\": " << unsorted.residentBytes << ",\n";
  out << "  \"gpu_free_bytes_before\": " << unsorted.gpuFreeBytesBefore << ",\n";
  out << "  \"gpu_total_bytes\": " << unsorted.gpuTotalBytes << ",\n";
  out << "  \"query_count\": " << unsorted.queryCount << ",\n";
  out << "  \"repeats\": " << unsorted.repeats << ",\n";
  if (!unsorted.error.empty()) {
    out << "  \"error\": \"" << unsorted.error << "\"\n";
    out << "}\n";
    return;
  }
  out << "  \"unsorted\": {\n";
  out << "    \"upload_ms\": " << unsorted.uploadMs << ",\n";
  out << "    \"kernel_ms\": " << unsorted.kernelMs << ",\n";
  out << "    \"queries_per_sec\": " << unsorted.queriesPerSec << ",\n";
  out << "    \"mean_abs_diff\": " << unsorted.meanAbsDiff << ",\n";
  out << "    \"max_abs_diff\": " << unsorted.maxAbsDiff << "\n";
  out << "  }";
  if (sorted) {
    out << ",\n";
    out << "  \"sorted\": {\n";
    out << "    \"upload_ms\": " << sorted->uploadMs << ",\n";
    out << "    \"kernel_ms\": " << sorted->kernelMs << ",\n";
    out << "    \"queries_per_sec\": " << sorted->queriesPerSec << ",\n";
    out << "    \"mean_abs_diff\": " << sorted->meanAbsDiff << ",\n";
    out << "    \"max_abs_diff\": " << sorted->maxAbsDiff << "\n";
    out << "  }\n";
  } else {
    out << "\n";
  }
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
    } else if (arg == "--query-count" && i + 1 < argc) queryCount = parseUint(argv[++i]);
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

  const std::string patternName = (pattern == QueryPattern::RandomFull ? "random" :
                                   pattern == QueryPattern::SameT ? "same-t" :
                                   pattern == QueryPattern::SameXYZ ? "same-xyz" : "coherent");

  try {
    auto dense = loadRawSequence(inputRaw, dimX, dimY, dimZ, frames);
    auto unsortedQueries = makeQueries(dimX, dimY, dimZ, frames, pattern, fixedFrame, queryCount, seed);
    auto sortedQueries = unsortedQueries;
    sortQueriesByFrameBlock(sortedQueries);
    auto expectedUnsorted =
        gatherExpected(dense, dimX, dimY, dimZ, frames, rawLayout, unsortedQueries);
    auto expectedSorted =
        gatherExpected(dense, dimX, dimY, dimZ, frames, rawLayout, sortedQueries);

    auto unsorted =
        runGpuBench(dense, dimX, dimY, dimZ, frames, rawLayout, unsortedQueries, expectedUnsorted, repeats);
    auto sorted =
        runGpuBench(dense, dimX, dimY, dimZ, frames, rawLayout, sortedQueries, expectedSorted, repeats);

    std::cout << "dense CUDA query benchmark\n";
    std::cout << "  dims: " << dimX << " x " << dimY << " x " << dimZ << "\n";
    std::cout << "  frames: " << frames << "\n";
    std::cout << "  rawLayout: "
              << (rawLayout == RawLayout::TimeFastest ? "time-fastest" : "frame-major")
              << "\n";
    std::cout << "  residentBytes: " << unsorted.residentBytes << "\n";
    std::cout << "  gpuFreeBytesBefore: " << unsorted.gpuFreeBytesBefore << "\n";
    std::cout << "  gpuTotalBytes: " << unsorted.gpuTotalBytes << "\n";
    std::cout << "  queryCount: " << queryCount << "\n";
    std::cout << "  repeats: " << repeats << "\n";
    std::cout << "\nUnsorted\n";
    std::cout << "  uploadMs: " << unsorted.uploadMs << "\n";
    std::cout << "  kernelMs: " << unsorted.kernelMs << "\n";
    std::cout << "  queriesPerSec: " << unsorted.queriesPerSec << "\n";
    std::cout << "\nSortedByFrameBlock\n";
    std::cout << "  uploadMs: " << sorted.uploadMs << "\n";
    std::cout << "  kernelMs: " << sorted.kernelMs << "\n";
    std::cout << "  queriesPerSec: " << sorted.queriesPerSec << "\n";

    if (!jsonOut.empty()) {
      writeJson(jsonOut, inputRaw, rawLayout, patternName, unsorted, &sorted);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "dense_cuda_query_bench failed: " << e.what() << "\n";
    if (!jsonOut.empty()) {
      BenchResult failed{};
      failed.dimX = dimX;
      failed.dimY = dimY;
      failed.dimZ = dimZ;
      failed.frames = frames;
      failed.queryCount = queryCount;
      failed.repeats = repeats;
      failed.residentBytes = static_cast<size_t>(dimX) * dimY * dimZ * frames * sizeof(float);
      size_t freeBytes = 0;
      size_t totalBytes = 0;
      if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) {
        failed.gpuFreeBytesBefore = freeBytes;
        failed.gpuTotalBytes = totalBytes;
      }
      failed.error = e.what();
      try {
        writeJson(jsonOut, inputRaw, rawLayout, patternName, failed, nullptr);
      } catch (...) {
      }
    }
    return 2;
  }
}
