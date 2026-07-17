#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "scientific_decode.h"
#include "vbt_file.h"

namespace {

using vbt::render::Query4D;
using vbt::render::ScientificMode;
using vbt::render::VbtFile;
using vbt::render::VbtFileHeader;

struct BenchResult {
    uint32_t dimX = 0;
    uint32_t dimY = 0;
    uint32_t dimZ = 0;
    uint32_t frames = 0;
    uint32_t fixedFrame = 0;
    uint32_t queryCount = 0;
    uint32_t repeats = 0;
    size_t payloadBytes = 0;
    size_t offsetBytes = 0;
    size_t coarseScaleBytes = 0;
    double uploadMs = 0.0;
    double kernelMs = 0.0;
    double queriesPerSec = 0.0;
    double meanAbsDiff = 0.0;
    double maxAbsDiff = 0.0;
};

struct QuerySegment {
    uint32_t start = 0;
    uint32_t count = 0;
    uint32_t leafIndex = 0;
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
        << "Usage: vbt_cuda_query_bench --input-vbt <path>\n"
        << "                           [--query-count N] [--seed N] [--repeats N]\n"
        << "                           [--pattern random|same-t|same-xyz|coherent] [--fixed-frame N]\n"
        << "                           [--sort-by-leaf] [--json-out path]\n";
}

enum class QueryPattern {
    RandomFull,
    SameT,
    SameXYZ,
    CoherentTiles,
};

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

struct DeviceHeader {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
    uint32_t frames = 0;
    uint32_t leafSize = 0;
    uint32_t leafCountX = 0;
    uint32_t leafCountY = 0;
    uint32_t coarseAcScaleCount = 0;
};

__device__ __host__ inline uint32_t payloadByte(const uint32_t* payload, uint32_t byteOffset)
{
    const uint32_t word = payload[byteOffset >> 2u];
    const uint32_t shift = (byteOffset & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

__device__ __host__ inline uint32_t payloadU16(const uint32_t* payload, uint32_t byteOffset)
{
    return payloadByte(payload, byteOffset) | (payloadByte(payload, byteOffset + 1u) << 8u);
}

__device__ __host__ inline int payloadI8(const uint32_t* payload, uint32_t byteOffset)
{
    const uint32_t u = payloadByte(payload, byteOffset);
    return (u < 128u) ? static_cast<int>(u) : static_cast<int>(u) - 256;
}

__device__ __host__ inline float halfToFloatBits(uint16_t h)
{
    const uint32_t sign = (static_cast<uint32_t>(h) >> 15u) & 1u;
    const uint32_t exponent = (static_cast<uint32_t>(h) >> 10u) & 0x1Fu;
    const uint32_t mantissa = static_cast<uint32_t>(h) & 0x3FFu;
    uint32_t f = 0;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            f = sign << 31u;
        } else {
            uint32_t e = 1u;
            uint32_t m = mantissa;
            while ((m & 0x400u) == 0u) {
                m <<= 1u;
                --e;
            }
            m &= 0x3FFu;
            f = (sign << 31u) | ((e + 112u) << 23u) | (m << 13u);
        }
    } else if (exponent == 31u) {
        f = (sign << 31u) | 0x7F800000u | (mantissa << 13u);
    } else {
        f = (sign << 31u) | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }
    union {
        uint32_t u;
        float f;
    } cvt{};
    cvt.u = f;
    return cvt.f;
}

__device__ __host__ inline float payloadHalf(const uint32_t* payload, uint32_t byteOffset)
{
    return halfToFloatBits(static_cast<uint16_t>(payloadU16(payload, byteOffset)));
}

__device__ __host__ inline float dctBasis(uint32_t totalLength, uint32_t index, uint32_t k)
{
    constexpr float kPi = 3.14159265358979323846f;
    const float n = static_cast<float>(totalLength);
    const float alpha = (k == 0u) ? rsqrtf(n) : sqrtf(2.0f / n);
    return alpha * cosf((kPi / n) * (static_cast<float>(index) + 0.5f) * static_cast<float>(k));
}

__device__ __host__ inline uint32_t tierCapacityFromId(uint32_t tierId)
{
    if (tierId == 0u) return 64u;
    if (tierId == 1u) return 256u;
    if (tierId == 2u) return 768u;
    return 1024u;
}

__device__ inline float decodeCoarseControl(const uint32_t* payload,
                                            const float* coarseScales,
                                            uint32_t coarseScaleCount,
                                            uint32_t frames,
                                            uint32_t payloadByteBase,
                                            int keep,
                                            int controlIndex,
                                            uint32_t t)
{
    const uint32_t controlBase = payloadByteBase + 4u + static_cast<uint32_t>(controlIndex * (keep + 1));
    float sum = payloadHalf(payload, controlBase) * dctBasis(frames, t, 0u);
    for (int k = 1; k < keep; ++k) {
        float coeff = static_cast<float>(payloadI8(payload, controlBase + 2u + static_cast<uint32_t>(k - 1)));
        if (static_cast<uint32_t>(k - 1) < coarseScaleCount) {
            coeff *= coarseScales[k - 1];
        } else {
            coeff = 0.0f;
        }
        sum += coeff * dctBasis(frames, t, static_cast<uint32_t>(k));
    }
    return sum;
}

__device__ inline float sampleCoarseAt(const uint32_t* payload,
                                       const float* coarseScales,
                                       uint32_t coarseScaleCount,
                                       uint32_t frames,
                                       uint32_t payloadByteBase,
                                       int keep,
                                       uint32_t lx,
                                       uint32_t ly,
                                       uint32_t lz,
                                       uint32_t t)
{
    const float fx = (static_cast<float>(lx) / 7.0f) * 3.0f;
    const float fy = (static_cast<float>(ly) / 7.0f) * 3.0f;
    const float fz = (static_cast<float>(lz) / 7.0f) * 3.0f;
    const int x0 = max(0, min(2, static_cast<int>(floorf(fx))));
    const int y0 = max(0, min(2, static_cast<int>(floorf(fy))));
    const int z0 = max(0, min(2, static_cast<int>(floorf(fz))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);

    const int idx000 = (z0 * 4 + y0) * 4 + x0;
    const int idx100 = (z0 * 4 + y0) * 4 + x1;
    const int idx010 = (z0 * 4 + y1) * 4 + x0;
    const int idx110 = (z0 * 4 + y1) * 4 + x1;
    const int idx001 = (z1 * 4 + y0) * 4 + x0;
    const int idx101 = (z1 * 4 + y0) * 4 + x1;
    const int idx011 = (z1 * 4 + y1) * 4 + x0;
    const int idx111 = (z1 * 4 + y1) * 4 + x1;

    const float c00 = decodeCoarseControl(payload, coarseScales, coarseScaleCount, frames, payloadByteBase, keep, idx000, t) * (1.0f - tx) +
                      decodeCoarseControl(payload, coarseScales, coarseScaleCount, frames, payloadByteBase, keep, idx100, t) * tx;
    const float c10 = decodeCoarseControl(payload, coarseScales, coarseScaleCount, frames, payloadByteBase, keep, idx010, t) * (1.0f - tx) +
                      decodeCoarseControl(payload, coarseScales, coarseScaleCount, frames, payloadByteBase, keep, idx110, t) * tx;
    const float c01 = decodeCoarseControl(payload, coarseScales, coarseScaleCount, frames, payloadByteBase, keep, idx001, t) * (1.0f - tx) +
                      decodeCoarseControl(payload, coarseScales, coarseScaleCount, frames, payloadByteBase, keep, idx101, t) * tx;
    const float c11 = decodeCoarseControl(payload, coarseScales, coarseScaleCount, frames, payloadByteBase, keep, idx011, t) * (1.0f - tx) +
                      decodeCoarseControl(payload, coarseScales, coarseScaleCount, frames, payloadByteBase, keep, idx111, t) * tx;
    return ((c00 * (1.0f - ty) + c10 * ty) * (1.0f - tz)) + ((c01 * (1.0f - ty) + c11 * ty) * tz);
}

__device__ inline float sampleCoarseFromControls(const float* ctrl,
                                                 uint32_t lx,
                                                 uint32_t ly,
                                                 uint32_t lz)
{
    const float fx = (static_cast<float>(lx) / 7.0f) * 3.0f;
    const float fy = (static_cast<float>(ly) / 7.0f) * 3.0f;
    const float fz = (static_cast<float>(lz) / 7.0f) * 3.0f;
    const int x0 = max(0, min(2, static_cast<int>(floorf(fx))));
    const int y0 = max(0, min(2, static_cast<int>(floorf(fy))));
    const int z0 = max(0, min(2, static_cast<int>(floorf(fz))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);

    const int idx000 = (z0 * 4 + y0) * 4 + x0;
    const int idx100 = (z0 * 4 + y0) * 4 + x1;
    const int idx010 = (z0 * 4 + y1) * 4 + x0;
    const int idx110 = (z0 * 4 + y1) * 4 + x1;
    const int idx001 = (z1 * 4 + y0) * 4 + x0;
    const int idx101 = (z1 * 4 + y0) * 4 + x1;
    const int idx011 = (z1 * 4 + y1) * 4 + x0;
    const int idx111 = (z1 * 4 + y1) * 4 + x1;

    const float c00 = ctrl[idx000] * (1.0f - tx) + ctrl[idx100] * tx;
    const float c10 = ctrl[idx010] * (1.0f - tx) + ctrl[idx110] * tx;
    const float c01 = ctrl[idx001] * (1.0f - tx) + ctrl[idx101] * tx;
    const float c11 = ctrl[idx011] * (1.0f - tx) + ctrl[idx111] * tx;
    return ((c00 * (1.0f - ty) + c10 * ty) * (1.0f - tz)) + ((c01 * (1.0f - ty) + c11 * ty) * tz);
}

__device__ inline float decodeBfp(uint32_t q, float scale, uint32_t bits)
{
    if (bits == 0u) return 0.0f;
    const uint32_t levels = (1u << bits) - 1u;
    const float norm = (static_cast<float>(q) / static_cast<float>(levels)) * 2.0f - 1.0f;
    return norm * scale;
}

__device__ inline float readDenseQuantValue(const uint32_t* payload,
                                            uint32_t dataBase,
                                            uint32_t frameByteStride,
                                            uint32_t denseBits,
                                            uint32_t valuesPerFrame,
                                            uint32_t localFrame,
                                            uint32_t controlIndex)
{
    if (controlIndex >= valuesPerFrame || denseBits == 0u) return 0.0f;
    const uint32_t bitOffset = controlIndex * denseBits;
    const uint32_t byteIndex = dataBase + localFrame * frameByteStride + (bitOffset >> 3u);
    const uint32_t shift = bitOffset & 7u;
    uint32_t packed = payloadByte(payload, byteIndex) |
                      (payloadByte(payload, byteIndex + 1u) << 8u) |
                      (payloadByte(payload, byteIndex + 2u) << 16u) |
                      (payloadByte(payload, byteIndex + 3u) << 24u);
    const uint32_t mask = (1u << denseBits) - 1u;
    return static_cast<float>((packed >> shift) & mask);
}

__device__ inline float decodeDenseResidual(const uint32_t* payload,
                                            uint32_t frames,
                                            uint32_t payloadByteBase,
                                            int keep,
                                            uint32_t denseResolution,
                                            uint32_t denseBits,
                                            uint32_t lx,
                                            uint32_t ly,
                                            uint32_t lz,
                                            uint32_t t)
{
    const uint32_t valuesPerFrame = denseResolution * denseResolution * denseResolution;
    const uint32_t frameByteStride = (valuesPerFrame * denseBits + 7u) >> 3u;
    const uint32_t scalesBase = payloadByteBase + 4u + static_cast<uint32_t>(64 * (keep + 1));
    const uint32_t dataBase = scalesBase + frames * 2u;
    const float scale = payloadHalf(payload, scalesBase + t * 2u);

    const float fx = (static_cast<float>(lx) / 7.0f) * static_cast<float>(denseResolution - 1u);
    const float fy = (static_cast<float>(ly) / 7.0f) * static_cast<float>(denseResolution - 1u);
    const float fz = (static_cast<float>(lz) / 7.0f) * static_cast<float>(denseResolution - 1u);
    const int x0 = max(0, min(static_cast<int>(denseResolution) - 2, static_cast<int>(floorf(fx))));
    const int y0 = max(0, min(static_cast<int>(denseResolution) - 2, static_cast<int>(floorf(fy))));
    const int z0 = max(0, min(static_cast<int>(denseResolution) - 2, static_cast<int>(floorf(fz))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);

    const uint32_t idx000 = static_cast<uint32_t>((z0 * static_cast<int>(denseResolution) + y0) * static_cast<int>(denseResolution) + x0);
    const uint32_t idx100 = static_cast<uint32_t>((z0 * static_cast<int>(denseResolution) + y0) * static_cast<int>(denseResolution) + x1);
    const uint32_t idx010 = static_cast<uint32_t>((z0 * static_cast<int>(denseResolution) + y1) * static_cast<int>(denseResolution) + x0);
    const uint32_t idx110 = static_cast<uint32_t>((z0 * static_cast<int>(denseResolution) + y1) * static_cast<int>(denseResolution) + x1);
    const uint32_t idx001 = static_cast<uint32_t>((z1 * static_cast<int>(denseResolution) + y0) * static_cast<int>(denseResolution) + x0);
    const uint32_t idx101 = static_cast<uint32_t>((z1 * static_cast<int>(denseResolution) + y0) * static_cast<int>(denseResolution) + x1);
    const uint32_t idx011 = static_cast<uint32_t>((z1 * static_cast<int>(denseResolution) + y1) * static_cast<int>(denseResolution) + x0);
    const uint32_t idx111 = static_cast<uint32_t>((z1 * static_cast<int>(denseResolution) + y1) * static_cast<int>(denseResolution) + x1);

    const float c00 = decodeBfp(static_cast<uint32_t>(readDenseQuantValue(payload, dataBase, frameByteStride, denseBits, valuesPerFrame, t, idx000)), scale, denseBits) * (1.0f - tx) +
                      decodeBfp(static_cast<uint32_t>(readDenseQuantValue(payload, dataBase, frameByteStride, denseBits, valuesPerFrame, t, idx100)), scale, denseBits) * tx;
    const float c10 = decodeBfp(static_cast<uint32_t>(readDenseQuantValue(payload, dataBase, frameByteStride, denseBits, valuesPerFrame, t, idx010)), scale, denseBits) * (1.0f - tx) +
                      decodeBfp(static_cast<uint32_t>(readDenseQuantValue(payload, dataBase, frameByteStride, denseBits, valuesPerFrame, t, idx110)), scale, denseBits) * tx;
    const float c01 = decodeBfp(static_cast<uint32_t>(readDenseQuantValue(payload, dataBase, frameByteStride, denseBits, valuesPerFrame, t, idx001)), scale, denseBits) * (1.0f - tx) +
                      decodeBfp(static_cast<uint32_t>(readDenseQuantValue(payload, dataBase, frameByteStride, denseBits, valuesPerFrame, t, idx101)), scale, denseBits) * tx;
    const float c11 = decodeBfp(static_cast<uint32_t>(readDenseQuantValue(payload, dataBase, frameByteStride, denseBits, valuesPerFrame, t, idx011)), scale, denseBits) * (1.0f - tx) +
                      decodeBfp(static_cast<uint32_t>(readDenseQuantValue(payload, dataBase, frameByteStride, denseBits, valuesPerFrame, t, idx111)), scale, denseBits) * tx;
    return ((c00 * (1.0f - ty) + c10 * ty) * (1.0f - tz)) + ((c01 * (1.0f - ty) + c11 * ty) * tz);
}

__device__ inline float decodeDenseTemporalBasisResidual(const uint32_t* payload,
                                                         uint32_t frames,
                                                         uint32_t payloadByteBase,
                                                         int keep,
                                                         uint32_t denseResolution,
                                                         uint32_t denseBits,
                                                         uint32_t temporalKeep,
                                                         uint32_t lx,
                                                         uint32_t ly,
                                                         uint32_t lz,
                                                         uint32_t t)
{
    if (temporalKeep == 0u || denseBits == 0u) return 0.0f;
    const uint32_t valuesPerBasis = denseResolution * denseResolution * denseResolution;
    const uint32_t basisByteStride = (valuesPerBasis * denseBits + 7u) >> 3u;
    const uint32_t scalesBase = payloadByteBase + 4u + static_cast<uint32_t>(64 * (keep + 1));
    const uint32_t dataBase = scalesBase + temporalKeep * 2u;

    const float fx = (static_cast<float>(lx) / 7.0f) * static_cast<float>(denseResolution - 1u);
    const float fy = (static_cast<float>(ly) / 7.0f) * static_cast<float>(denseResolution - 1u);
    const float fz = (static_cast<float>(lz) / 7.0f) * static_cast<float>(denseResolution - 1u);
    const int x0 = max(0, min(static_cast<int>(denseResolution) - 2, static_cast<int>(floorf(fx))));
    const int y0 = max(0, min(static_cast<int>(denseResolution) - 2, static_cast<int>(floorf(fy))));
    const int z0 = max(0, min(static_cast<int>(denseResolution) - 2, static_cast<int>(floorf(fz))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);

    auto sampleBasis = [&](uint32_t basisIdx, int cx, int cy, int cz) -> float {
        const uint32_t controlIndex = static_cast<uint32_t>((cz * static_cast<int>(denseResolution) + cy) * static_cast<int>(denseResolution) + cx);
        const uint32_t basisBase = dataBase + basisIdx * basisByteStride;
        const uint32_t bitOffset = controlIndex * denseBits;
        const uint32_t byteIndex = basisBase + (bitOffset >> 3u);
        const uint32_t shift = bitOffset & 7u;
        uint32_t packed = payloadByte(payload, byteIndex) |
                          (payloadByte(payload, byteIndex + 1u) << 8u) |
                          (payloadByte(payload, byteIndex + 2u) << 16u) |
                          (payloadByte(payload, byteIndex + 3u) << 24u);
        const uint32_t mask = (1u << denseBits) - 1u;
        const float scale = payloadHalf(payload, scalesBase + basisIdx * 2u);
        return decodeBfp(static_cast<uint32_t>((packed >> shift) & mask), scale, denseBits);
    };

    float residual = 0.0f;
    for (uint32_t k = 0; k < temporalKeep; ++k) {
        const float c00 = sampleBasis(k, x0, y0, z0) * (1.0f - tx) + sampleBasis(k, x1, y0, z0) * tx;
        const float c10 = sampleBasis(k, x0, y1, z0) * (1.0f - tx) + sampleBasis(k, x1, y1, z0) * tx;
        const float c01 = sampleBasis(k, x0, y0, z1) * (1.0f - tx) + sampleBasis(k, x1, y0, z1) * tx;
        const float c11 = sampleBasis(k, x0, y1, z1) * (1.0f - tx) + sampleBasis(k, x1, y1, z1) * tx;
        residual += (((c00 * (1.0f - ty) + c10 * ty) * (1.0f - tz)) + ((c01 * (1.0f - ty) + c11 * ty) * tz))
                    * dctBasis(frames, t, k);
    }
    return residual;
}

__device__ inline float decodeSparseResidual(const uint32_t* payload,
                                             uint32_t coarseScaleCount,
                                             uint32_t payloadByteBase,
                                             uint32_t packedHeader,
                                             int coarseKeep,
                                             uint32_t localIndexValue,
                                             uint32_t t)
{
    (void)coarseScaleCount;
    const uint32_t eventCount = (packedHeader >> 16u) & 0x0FFFu;
    if (eventCount == 0u) return 0.0f;
    const uint32_t tierId = (packedHeader >> 8u) & 0x3u;
    const uint32_t tierCapacity = tierCapacityFromId(tierId);
    const uint32_t framesPerBin = ((packedHeader >> 10u) & 0xFu) + 1u;
    const uint32_t sparseBase = payloadByteBase + 4u + static_cast<uint32_t>(64 * (coarseKeep + 1));

    union {
        uint32_t u;
        float f;
    } eventScale{};
    eventScale.u = payload[sparseBase >> 2u];
    if (!(eventScale.f > 0.0f)) return 0.0f;

    const uint32_t maskWord = payload[(sparseBase + 4u + ((localIndexValue >> 5u) * 4u)) >> 2u];
    const uint32_t maskBit = 1u << (localIndexValue & 31u);
    if ((maskWord & maskBit) == 0u) return 0.0f;

    const uint32_t binIdx = min(15u, t / max(1u, framesPerBin));
    const uint32_t localTime = t - binIdx * max(1u, framesPerBin);
    const uint32_t binsBase = sparseBase + 4u + 64u;
    const uint32_t start = payloadU16(payload, binsBase + binIdx * 2u);
    const uint32_t end = (binIdx + 1u < 16u) ? payloadU16(payload, binsBase + (binIdx + 1u) * 2u) : eventCount;
    if (start >= end) return 0.0f;

    const uint32_t targetKey = ((localTime & 0xFu) << 9u) | (localIndexValue & 0x1FFu);
    const uint32_t coordsBase = binsBase + 32u;
    const uint32_t residualBase = coordsBase + tierCapacity * 2u;
    int left = static_cast<int>(start);
    int right = static_cast<int>(end) - 1;
    while (left <= right) {
        const int mid = (left + right) >> 1;
        const uint32_t key = payloadU16(payload, coordsBase + static_cast<uint32_t>(mid) * 2u);
        if (key == targetKey) {
            const uint32_t byte = payloadByte(payload, residualBase + static_cast<uint32_t>(mid >> 1));
            const uint32_t nibble = ((mid & 1) == 0) ? (byte & 0x0Fu) : ((byte >> 4u) & 0x0Fu);
            int q = static_cast<int>(nibble);
            if ((q & 0x8) != 0) q -= 16;
            return static_cast<float>(q) * eventScale.f;
        }
        if (key < targetKey) left = mid + 1;
        else right = mid - 1;
    }
    return 0.0f;
}

__global__ void vbtQueryKernel(const Query4D* queries,
                               uint32_t queryCount,
                               const float* coarseScales,
                               const uint32_t* offsets,
                               const uint32_t* payload,
                               DeviceHeader header,
                               float* outValues)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= queryCount) return;

    const Query4D q = queries[tid];
    const uint32_t bx = min(header.leafCountX - 1u, q.x / header.leafSize);
    const uint32_t by = min(header.leafCountY - 1u, q.y / header.leafSize);
    const uint32_t leafCountZ = (header.depth + header.leafSize - 1u) / header.leafSize;
    const uint32_t bz = min(leafCountZ - 1u, q.z / header.leafSize);
    const uint32_t leafIndex = (bz * header.leafCountY + by) * header.leafCountX + bx;
    const uint32_t wordOffset = offsets[leafIndex];
    const uint32_t packedHeader = payload[wordOffset];
    const uint32_t payloadByteBase = wordOffset * 4u;
    const uint32_t mode = packedHeader & 0x3u;

    const uint32_t lx = q.x % header.leafSize;
    const uint32_t ly = q.y % header.leafSize;
    const uint32_t lz = q.z % header.leafSize;
    const uint32_t localIndexValue = (lz * header.leafSize + ly) * header.leafSize + lx;

    const int coarseKeep = static_cast<int>(((packedHeader >> 2u) & 0x3Fu) + 1u);
    const uint32_t denseSubtype = (packedHeader >> 12u) & 0x7u;
    const uint32_t denseTemporalKeep = ((packedHeader >> 15u) & 0xFu) + 1u;
    float value = sampleCoarseAt(payload,
                                 coarseScales,
                                 header.coarseAcScaleCount,
                                 header.frames,
                                 payloadByteBase,
                                 coarseKeep,
                                 lx,
                                 ly,
                                 lz,
                                 q.t);

    if (mode == static_cast<uint32_t>(ScientificMode::SparseEvents)) {
        value += decodeSparseResidual(payload,
                                      header.coarseAcScaleCount,
                                      payloadByteBase,
                                      packedHeader,
                                      coarseKeep,
                                      localIndexValue,
                                      q.t);
    } else if (mode == static_cast<uint32_t>(ScientificMode::DenseGrid3)) {
        const uint32_t denseBits = (packedHeader >> 8u) & 0xFu;
        value += (denseSubtype == 1u)
            ? decodeDenseTemporalBasisResidual(payload, header.frames, payloadByteBase, coarseKeep, 3u, denseBits, denseTemporalKeep, lx, ly, lz, q.t)
            : decodeDenseResidual(payload, header.frames, payloadByteBase, coarseKeep, 3u, denseBits, lx, ly, lz, q.t);
    } else if (mode == static_cast<uint32_t>(ScientificMode::DenseGrid4)) {
        const uint32_t denseBits = (packedHeader >> 8u) & 0xFu;
        value += (denseSubtype == 1u)
            ? decodeDenseTemporalBasisResidual(payload, header.frames, payloadByteBase, coarseKeep, 4u, denseBits, denseTemporalKeep, lx, ly, lz, q.t)
            : decodeDenseResidual(payload, header.frames, payloadByteBase, coarseKeep, 4u, denseBits, lx, ly, lz, q.t);
    }

    outValues[tid] = value;
}

__global__ void vbtGroupedLeafQueryKernel(const Query4D* queries,
                                          const QuerySegment* segments,
                                          uint32_t segmentCount,
                                          const float* coarseScales,
                                          const uint32_t* offsets,
                                          const uint32_t* payload,
                                          DeviceHeader header,
                                          float* outValues)
{
    const uint32_t segmentIndex = blockIdx.x;
    if (segmentIndex >= segmentCount) return;

    const QuerySegment seg = segments[segmentIndex];
    const uint32_t leafIndex = seg.leafIndex;
    const uint32_t wordOffset = offsets[leafIndex];
    const uint32_t packedHeader = payload[wordOffset];
    const uint32_t payloadByteBase = wordOffset * 4u;
    const uint32_t mode = packedHeader & 0x3u;
    const int coarseKeep = static_cast<int>(((packedHeader >> 2u) & 0x3Fu) + 1u);
    const uint32_t denseSubtype = (packedHeader >> 12u) & 0x7u;
    const uint32_t denseTemporalKeep = ((packedHeader >> 15u) & 0xFu) + 1u;

    __shared__ float coarseCtrl[64];
    if (threadIdx.x < 64u) {
        coarseCtrl[threadIdx.x] = decodeCoarseControl(payload,
                                                      coarseScales,
                                                      header.coarseAcScaleCount,
                                                      header.frames,
                                                      payloadByteBase,
                                                      coarseKeep,
                                                      static_cast<int>(threadIdx.x),
                                                      queries[seg.start].t);
    }
    __syncthreads();

    for (uint32_t local = threadIdx.x; local < seg.count; local += blockDim.x) {
        const uint32_t queryIndex = seg.start + local;
        const Query4D q = queries[queryIndex];
        const uint32_t lx = q.x % header.leafSize;
        const uint32_t ly = q.y % header.leafSize;
        const uint32_t lz = q.z % header.leafSize;
        const uint32_t localIndexValue = (lz * header.leafSize + ly) * header.leafSize + lx;

        float value = sampleCoarseFromControls(coarseCtrl, lx, ly, lz);
        if (mode == static_cast<uint32_t>(ScientificMode::SparseEvents)) {
            value += decodeSparseResidual(payload,
                                          header.coarseAcScaleCount,
                                          payloadByteBase,
                                          packedHeader,
                                          coarseKeep,
                                          localIndexValue,
                                          q.t);
        } else if (mode == static_cast<uint32_t>(ScientificMode::DenseGrid3)) {
            const uint32_t denseBits = (packedHeader >> 8u) & 0xFu;
            value += (denseSubtype == 1u)
                ? decodeDenseTemporalBasisResidual(payload, header.frames, payloadByteBase, coarseKeep, 3u, denseBits, denseTemporalKeep, lx, ly, lz, q.t)
                : decodeDenseResidual(payload, header.frames, payloadByteBase, coarseKeep, 3u, denseBits, lx, ly, lz, q.t);
        } else if (mode == static_cast<uint32_t>(ScientificMode::DenseGrid4)) {
            const uint32_t denseBits = (packedHeader >> 8u) & 0xFu;
            value += (denseSubtype == 1u)
                ? decodeDenseTemporalBasisResidual(payload, header.frames, payloadByteBase, coarseKeep, 4u, denseBits, denseTemporalKeep, lx, ly, lz, q.t)
                : decodeDenseResidual(payload, header.frames, payloadByteBase, coarseKeep, 4u, denseBits, lx, ly, lz, q.t);
        }
        outValues[queryIndex] = value;
    }
}

std::vector<float> gatherExpected(const VbtFile& file, const std::vector<Query4D>& queries)
{
    std::vector<float> values(queries.size(), 0.0f);
    for (size_t i = 0; i < queries.size(); ++i) {
        values[i] = vbt::render::decodeScientificValueAtCpu(file, queries[i]);
    }
    return values;
}

std::vector<Query4D> makeQueries(const VbtFileHeader& header,
                                 QueryPattern pattern,
                                 uint32_t fixedFrame,
                                 uint32_t queryCount,
                                 uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> dx(0, header.width - 1u);
    std::uniform_int_distribution<uint32_t> dy(0, header.height - 1u);
    std::uniform_int_distribution<uint32_t> dz(0, header.depth - 1u);
    std::uniform_int_distribution<uint32_t> dt(0, header.frames - 1u);
    const uint32_t fixedX = dx(rng);
    const uint32_t fixedY = dy(rng);
    const uint32_t fixedZ = dz(rng);

    std::vector<Query4D> queries(queryCount);
    for (uint32_t i = 0; i < queryCount; ++i) {
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
            q.x = std::min(header.width - 1u, baseX + (i % 8u));
            q.y = std::min(header.height - 1u, baseY + ((i / 8u) % 8u));
            q.z = std::min(header.depth - 1u, baseZ + ((i / 64u) % 8u));
            q.t = fixedFrame;
            break;
        }
        }
        q.leafIndex = vbt::render::leafIndexForVoxel(header, q.x, q.y, q.z);
        queries[i] = q;
    }
    return queries;
}

void sortQueriesByLeafLocal(std::vector<Query4D>& queries)
{
    std::stable_sort(queries.begin(), queries.end(), [](const Query4D& a, const Query4D& b) {
        if (a.leafIndex != b.leafIndex) return a.leafIndex < b.leafIndex;
        if (a.t != b.t) return a.t < b.t;
        if (a.z != b.z) return a.z < b.z;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });
}

std::vector<QuerySegment> buildLeafSegments(const std::vector<Query4D>& sortedQueries)
{
    std::vector<QuerySegment> segments;
    if (sortedQueries.empty()) return segments;
    QuerySegment current{};
    current.start = 0;
    current.count = 1;
    current.leafIndex = sortedQueries[0].leafIndex;
    for (size_t i = 1; i < sortedQueries.size(); ++i) {
        if (sortedQueries[i].leafIndex == current.leafIndex) {
            current.count += 1u;
        } else {
            segments.push_back(current);
            current.start = static_cast<uint32_t>(i);
            current.count = 1;
            current.leafIndex = sortedQueries[i].leafIndex;
        }
    }
    segments.push_back(current);
    return segments;
}

BenchResult runGpuBench(const VbtFile& file,
                        uint32_t fixedFrame,
                        const std::vector<Query4D>& queries,
                        const std::vector<float>& expected,
                        uint32_t repeats,
                        bool groupedSorted)
{
    BenchResult result{};
    result.dimX = file.header.width;
    result.dimY = file.header.height;
    result.dimZ = file.header.depth;
    result.frames = file.header.frames;
    result.fixedFrame = fixedFrame;
    result.queryCount = static_cast<uint32_t>(queries.size());
    result.repeats = repeats;
    result.payloadBytes = file.payloadWords.size() * sizeof(uint32_t);
    result.offsetBytes = file.offsetsWords.size() * sizeof(uint32_t);
    result.coarseScaleBytes = file.coarseAcScales.size() * sizeof(float);

    Query4D* dQueries = nullptr;
    float* dOut = nullptr;
    float* dCoarseScales = nullptr;
    uint32_t* dOffsets = nullptr;
    uint32_t* dPayload = nullptr;
    QuerySegment* dSegments = nullptr;
    std::vector<QuerySegment> hostSegments;
    bool useGroupedSorted = false;

    const size_t queryBytes = queries.size() * sizeof(Query4D);
    const size_t outBytes = queries.size() * sizeof(float);

    checkCuda(cudaFree(nullptr), "cudaFree warmup");
    auto uploadStart = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMalloc(&dQueries, queryBytes), "cudaMalloc queries");
    checkCuda(cudaMalloc(&dOut, outBytes), "cudaMalloc outputs");
    if (!file.coarseAcScales.empty()) {
        checkCuda(cudaMalloc(&dCoarseScales, result.coarseScaleBytes), "cudaMalloc coarse scales");
        checkCuda(cudaMemcpy(dCoarseScales, file.coarseAcScales.data(), result.coarseScaleBytes, cudaMemcpyHostToDevice), "cudaMemcpy coarse scales");
    }
    checkCuda(cudaMalloc(&dOffsets, result.offsetBytes), "cudaMalloc offsets");
    checkCuda(cudaMalloc(&dPayload, result.payloadBytes), "cudaMalloc payload");
    checkCuda(cudaMemcpy(dQueries, queries.data(), queryBytes, cudaMemcpyHostToDevice), "cudaMemcpy queries");
    checkCuda(cudaMemcpy(dOffsets, file.offsetsWords.data(), result.offsetBytes, cudaMemcpyHostToDevice), "cudaMemcpy offsets");
    checkCuda(cudaMemcpy(dPayload, file.payloadWords.data(), result.payloadBytes, cudaMemcpyHostToDevice), "cudaMemcpy payload");
    if (groupedSorted) {
        hostSegments = buildLeafSegments(queries);
        const double avgSegmentSize =
            hostSegments.empty() ? 0.0 : static_cast<double>(queries.size()) / static_cast<double>(hostSegments.size());
        useGroupedSorted = avgSegmentSize >= 32.0;
        if (useGroupedSorted) {
            checkCuda(cudaMalloc(&dSegments, hostSegments.size() * sizeof(QuerySegment)), "cudaMalloc segments");
            checkCuda(cudaMemcpy(dSegments, hostSegments.data(), hostSegments.size() * sizeof(QuerySegment), cudaMemcpyHostToDevice), "cudaMemcpy segments");
        }
    }
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after upload");
    auto uploadEnd = std::chrono::high_resolution_clock::now();
    result.uploadMs = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();

    DeviceHeader deviceHeader{};
    deviceHeader.width = file.header.width;
    deviceHeader.height = file.header.height;
    deviceHeader.depth = file.header.depth;
    deviceHeader.frames = file.header.frames;
    deviceHeader.leafSize = file.header.leafSize;
    deviceHeader.leafCountX = (file.header.width + file.header.leafSize - 1u) / file.header.leafSize;
    deviceHeader.leafCountY = (file.header.height + file.header.leafSize - 1u) / file.header.leafSize;
    deviceHeader.coarseAcScaleCount = file.header.coarseAcScaleCount;

    const uint32_t threads = 128u;
    const uint32_t blocks = (result.queryCount + threads - 1u) / threads;
    for (int i = 0; i < 3; ++i) {
        if (useGroupedSorted) {
            vbtGroupedLeafQueryKernel<<<static_cast<uint32_t>(hostSegments.size()), threads>>>(dQueries,
                                                                                                dSegments,
                                                                                                static_cast<uint32_t>(hostSegments.size()),
                                                                                                dCoarseScales,
                                                                                                dOffsets,
                                                                                                dPayload,
                                                                                                deviceHeader,
                                                                                                dOut);
        } else {
            vbtQueryKernel<<<blocks, threads>>>(dQueries,
                                                result.queryCount,
                                                dCoarseScales,
                                                dOffsets,
                                                dPayload,
                                                deviceHeader,
                                                dOut);
        }
    }
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warmup");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    checkCuda(cudaEventRecord(start), "cudaEventRecord start");
    for (uint32_t i = 0; i < repeats; ++i) {
        if (useGroupedSorted) {
            vbtGroupedLeafQueryKernel<<<static_cast<uint32_t>(hostSegments.size()), threads>>>(dQueries,
                                                                                                dSegments,
                                                                                                static_cast<uint32_t>(hostSegments.size()),
                                                                                                dCoarseScales,
                                                                                                dOffsets,
                                                                                                dPayload,
                                                                                                deviceHeader,
                                                                                                dOut);
        } else {
            vbtQueryKernel<<<blocks, threads>>>(dQueries,
                                                result.queryCount,
                                                dCoarseScales,
                                                dOffsets,
                                                dPayload,
                                                deviceHeader,
                                                dOut);
        }
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
    cudaFree(dQueries);
    cudaFree(dOut);
    cudaFree(dCoarseScales);
    cudaFree(dOffsets);
    cudaFree(dPayload);
    cudaFree(dSegments);
    return result;
}

void writeJson(const std::string& path,
               const std::string& inputVbt,
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
    out << "  \"dataset\": \"vbt_cuda_query\",\n";
    out << "  \"input_vbt\": \"" << inputVbt << "\",\n";
    out << "  \"dim_x\": " << unsorted.dimX << ",\n";
    out << "  \"dim_y\": " << unsorted.dimY << ",\n";
    out << "  \"dim_z\": " << unsorted.dimZ << ",\n";
    out << "  \"frames\": " << unsorted.frames << ",\n";
    out << "  \"pattern\": \"" << patternName << "\",\n";
    out << "  \"query_count\": " << unsorted.queryCount << ",\n";
    out << "  \"repeats\": " << unsorted.repeats << ",\n";
    out << "  \"payload_bytes\": " << unsorted.payloadBytes << ",\n";
    out << "  \"offset_bytes\": " << unsorted.offsetBytes << ",\n";
    out << "  \"coarse_scale_bytes\": " << unsorted.coarseScaleBytes << ",\n";
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
    std::string inputVbt;
    std::string jsonOut;
    uint32_t fixedFrame = 64;
    uint32_t queryCount = 65536;
    uint32_t seed = 1;
    uint32_t repeats = 100;
    QueryPattern pattern = QueryPattern::RandomFull;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input-vbt" && i + 1 < argc) inputVbt = argv[++i];
        else if (arg == "--fixed-frame" && i + 1 < argc) fixedFrame = parseUint(argv[++i]);
        else if (arg == "--pattern" && i + 1 < argc) {
            if (!parseQueryPattern(argv[++i], pattern)) {
                printUsage();
                return 1;
            }
        }
        else if (arg == "--query-count" && i + 1 < argc) queryCount = parseUint(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = parseUint(argv[++i]);
        else if (arg == "--repeats" && i + 1 < argc) repeats = parseUint(argv[++i]);
        else if (arg == "--json-out" && i + 1 < argc) jsonOut = argv[++i];
        else {
            printUsage();
            return 1;
        }
    }

    if (inputVbt.empty()) {
        printUsage();
        return 1;
    }

    try {
        VbtFile file{};
        std::string error;
        if (!vbt::render::loadVbtFile(inputVbt, file, error)) {
            throw std::runtime_error(error);
        }
        if (fixedFrame >= file.header.frames) {
            throw std::runtime_error("Frame index out of range.");
        }

        auto unsortedQueries = makeQueries(file.header, pattern, fixedFrame, queryCount, seed);
        auto expectedUnsorted = gatherExpected(file, unsortedQueries);
        auto unsorted = runGpuBench(file, fixedFrame, unsortedQueries, expectedUnsorted, repeats, false);

        auto sortedQueries = unsortedQueries;
        sortQueriesByLeafLocal(sortedQueries);
        auto expectedSorted = gatherExpected(file, sortedQueries);
        auto sorted = runGpuBench(file, fixedFrame, sortedQueries, expectedSorted, repeats, true);

        std::cout << "VBT CUDA query benchmark\n";
        std::cout << "  input: " << inputVbt << "\n";
        std::cout << "  frames: " << file.header.frames << "\n";
        const std::string patternName = (pattern == QueryPattern::RandomFull ? "random" :
                                         pattern == QueryPattern::SameT ? "same-t" :
                                         pattern == QueryPattern::SameXYZ ? "same-xyz" : "coherent");
        std::cout << "  pattern: " << patternName << "\n";
        std::cout << "  query_count: " << queryCount << "\n";
        std::cout << "  unsorted q/s: " << unsorted.queriesPerSec << "\n";
        std::cout << "  sorted   q/s: " << sorted.queriesPerSec << "\n";
        std::cout << "  unsorted diff mean/max: " << unsorted.meanAbsDiff << " / " << unsorted.maxAbsDiff << "\n";
        std::cout << "  sorted   diff mean/max: " << sorted.meanAbsDiff << " / " << sorted.maxAbsDiff << "\n";

        if (!jsonOut.empty()) {
            writeJson(jsonOut, inputVbt, patternName, unsorted, sorted);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
