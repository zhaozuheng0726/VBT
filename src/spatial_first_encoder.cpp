#include "spatial_first_encoder.h"

#include "dct_codec.h"
#include "keyframe_detector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

#ifdef VBT_USE_OPENMP
#include <omp.h>
#endif

namespace vbt {

namespace {

float decodeDenseTemporalBasisAt(const DenseTemporalBasisEncoding& encoding,
                                 int x,
                                 int y,
                                 int z,
                                 int t,
                                 int totalFrames);
float decodeDenseTileTemporalBasisAt(const DenseTileTemporalBasisEncoding& encoding,
                                     int x,
                                     int y,
                                     int z,
                                     int t,
                                     int totalFrames);
float decodeDensePatchTemporalBasisAt(const DensePatchTemporalBasisEncoding& encoding,
                                      int x,
                                      int y,
                                      int z,
                                      int t,
                                      int totalFrames);

constexpr int kLeafVoxelCount = 8 * 8 * 8;
constexpr int kCoarseControlCount = 4 * 4 * 4;
constexpr uint32_t kPackedCoordSpatialBits = 9u;
constexpr uint32_t kPackedCoordLocalTimeBits = 4u;
constexpr uint16_t kMaskCoordSpatial = static_cast<uint16_t>((1u << kPackedCoordSpatialBits) - 1u);
constexpr uint16_t kMaskCoordLocalTime = static_cast<uint16_t>((1u << kPackedCoordLocalTimeBits) - 1u);
constexpr int kEventBinCount = 16;
constexpr uint32_t kScientificHeaderModeBits = 2u;
constexpr uint32_t kScientificHeaderKeepBits = 6u;
constexpr uint32_t kScientificHeaderModeMask = (1u << kScientificHeaderModeBits) - 1u;
constexpr uint32_t kScientificHeaderKeepMask = (1u << kScientificHeaderKeepBits) - 1u;

struct PseudoInverse4x4x4 {
    std::array<std::array<double, kLeafVoxelCount>, kCoarseControlCount> pinv{};
};

struct DynamicPseudoInverseGrid {
    int resolution = 0;
    int controlCount = 0;
    std::vector<double> pinv;
};

struct PatchPseudoInverseGrid {
    int patchSize = 0;
    int resolution = 0;
    int controlCount = 0;
    std::vector<double> pinv;
};

inline int localIndex(int x, int y, int z)
{
    return (z * 8 + y) * 8 + x;
}

inline int cellIndexOf(int x, int y, int z, int cellSize)
{
    const int cellRes = 8 / std::max(1, cellSize);
    const int cx = x / cellSize;
    const int cy = y / cellSize;
    const int cz = z / cellSize;
    return (cz * cellRes + cy) * cellRes + cx;
}

inline int popcount64(uint64_t value)
{
    int count = 0;
    while (value != 0ull) {
        value &= (value - 1ull);
        count += 1;
    }
    return count;
}

uint16_t chooseTierCapacity(size_t actualCount)
{
    if (actualCount == 0) return 0;
    if (actualCount <= 64u) return 64;
    if (actualCount <= 256u) return 256;
    if (actualCount <= 768u) return 768;
    if (actualCount <= 1024u) return 1024;
    return static_cast<uint16_t>(std::min<size_t>(actualCount, 0xFFFFu));
}

inline bool useScientificHeaderV2(const FieldProfile& profile, const SpatialFirstOptions& options)
{
    return options.splitScientificRenderModes && profile.type == FieldType::GENERIC;
}

uint32_t encodeScientificCoarseKeepMinus1(int dctKeep)
{
    const uint32_t keep = static_cast<uint32_t>(std::clamp(dctKeep, 1, 64));
    return (keep - 1u) & kScientificHeaderKeepMask;
}

int resolveCoarseKeep(const SpatialFirstOptions& options, const LeafEncoding& encoding)
{
    return std::max(1, encoding.coarseKeep > 0 ? static_cast<int>(encoding.coarseKeep) : options.dctKeep);
}

std::vector<int> buildScientificCoarseKeepLadder(int maxKeep)
{
    constexpr std::array<int, 8> kBaseKeeps{4, 6, 8, 10, 12, 14, 15, 16};
    const int requestedKeep = std::clamp(maxKeep, 1, 64);
    int clampedMaxKeep = kBaseKeeps.front();
    int bestDistance = std::abs(requestedKeep - clampedMaxKeep);
    for (int keep : kBaseKeeps) {
        const int distance = std::abs(requestedKeep - keep);
        if (distance < bestDistance || (distance == bestDistance && keep > clampedMaxKeep)) {
            clampedMaxKeep = keep;
            bestDistance = distance;
        }
    }
    std::vector<int> ladder;
    ladder.reserve(kBaseKeeps.size());
    for (int keep : kBaseKeeps) {
        if (keep <= clampedMaxKeep) ladder.push_back(keep);
    }
    return ladder;
}

int coarseKeepBucketIndex(int keep)
{
    switch (keep) {
    case 4: return 0;
    case 6: return 1;
    case 8: return 2;
    case 10: return 3;
    case 12: return 4;
    case 14: return 5;
    case 15: return 6;
    case 16: return 7;
    default: return -1;
    }
}

int computeMaxCoarseKeep(const std::vector<LeafEncoding>& encodings, const SpatialFirstOptions& options)
{
    int maxKeep = std::max(1, options.dctKeep);
    for (const auto& encoding : encodings) {
        maxKeep = std::max(maxKeep, resolveCoarseKeep(options, encoding));
    }
    return maxKeep;
}

uint32_t encodeScientificSparseTierId(uint16_t tierCapacity)
{
    if (tierCapacity <= 64u) return 0u;
    if (tierCapacity <= 256u) return 1u;
    if (tierCapacity <= 768u) return 2u;
    return 3u;
}

uint32_t encodeScientificEventQuantCode(const LeafEncoding&)
{
    // Current scientific mainline fixes sparse residuals to 4-bit signed
    // nibbles. Keep this explicit in the header so future quant modes can grow
    // without changing the outer block dictionary again.
    return 0u;
}

uint32_t encodeScientificDenseSubtype(const LeafEncoding& encoding)
{
    // Scientific dense payloads keep the outer mode ids stable and encode
    // format evolution through a small subtype field. Subtype 0 is the current
    // per-frame dense BFP payload; subtype 1 is reserved for temporal-basis
    // dense residuals so we can upgrade the scientific frontend without
    // changing the global block dictionary.
    if (encoding.useDensePatchTemporalBasis) return 3u;
    if (encoding.useDenseTileTemporalBasis) return 2u;
    if (encoding.useDenseTemporalBasis) return 1u;
    return 0u;
}

uint32_t packScientificHeaderV2(const SpatialFirstOptions& options, const LeafEncoding& encoding)
{
    const uint32_t keepMinus1 = encodeScientificCoarseKeepMinus1(resolveCoarseKeep(options, encoding));
    if (encoding.mode == BlockMode::SPARSE_IMPULSE) {
        const uint32_t mode = 1u;
        const uint32_t tierId = encodeScientificSparseTierId(encoding.packedTierCapacity) & 0x3u;
        const int safeEncodedFrames = std::max(1, encoding.encodedFrameCount);
        const uint32_t framesPerBin =
            static_cast<uint32_t>((safeEncodedFrames + kEventBinCount - 1) / kEventBinCount);
        const uint32_t framesPerBinMinus1 = std::min<uint32_t>(15u, framesPerBin - 1u);
        const uint32_t quantCode = encodeScientificEventQuantCode(encoding) & 0x3u;
        const uint32_t sparseSubtype = 0u;
        const uint32_t eventCount = static_cast<uint32_t>(encoding.packedEventCount) & 0x0FFFu;
        return mode |
               (keepMinus1 << 2u) |
               (tierId << 8u) |
               (framesPerBinMinus1 << 10u) |
               (quantCode << 14u) |
               (eventCount << 16u) |
               ((sparseSubtype & 0xFu) << 28u);
    }

    if (encoding.mode == BlockMode::DENSE_FINE) {
        const bool isDenseGrid3 =
            ((encoding.useDenseResidualBfp &&
              encoding.denseResidualBfp.sampleAsGrid &&
              encoding.denseResidualBfp.resolution <= 3) ||
             (encoding.useDenseTemporalBasis &&
              encoding.denseTemporalBasis.resolution <= 3) ||
             (encoding.useDensePatchTemporalBasis &&
              encoding.densePatchTemporalBasis.localResolution <= 2) ||
             (encoding.useDenseTileTemporalBasis &&
              encoding.denseTileTemporalBasis.tileSize <= 2));
        const uint32_t mode = isDenseGrid3 ? 2u : 3u;
        const uint32_t denseQuantBits =
            static_cast<uint32_t>(std::clamp(
                encoding.useDensePatchTemporalBasis ? encoding.densePatchTemporalBasis.bitsPerValue :
                encoding.useDenseTileTemporalBasis ? encoding.denseTileTemporalBasis.bitsPerValue :
                encoding.useDenseTemporalBasis ? encoding.denseTemporalBasis.bitsPerValue :
                (encoding.useDenseResidualBfp ? encoding.denseResidualBfp.bitsPerValue : 0),
                0, 15));
        const uint32_t denseSubtype = encodeScientificDenseSubtype(encoding) & 0x7u;
        const uint32_t denseTemporalKeepMinus1 =
            (encoding.useDenseTemporalBasis || encoding.useDenseTileTemporalBasis || encoding.useDensePatchTemporalBasis)
                ? static_cast<uint32_t>(std::clamp(
                      encoding.useDensePatchTemporalBasis ? encoding.densePatchTemporalBasis.temporalKeep :
                      encoding.useDenseTileTemporalBasis ? encoding.denseTileTemporalBasis.temporalKeep
                                                         : encoding.denseTemporalBasis.temporalKeep,
                      1, 16) - 1)
                : 0u;
        return (mode & kScientificHeaderModeMask) |
               (keepMinus1 << 2u) |
               ((denseQuantBits & 0xFu) << 8u) |
               (denseSubtype << 12u) |
               ((denseTemporalKeepMinus1 & 0xFu) << 15u);
    }

    // Scientific mode 00 covers the coarse/base path. Generic constant blocks
    // are currently disabled, so coarse-only and constant both map here.
    const uint32_t mode = 0u;
    const uint32_t coarseQuantClass = 0u;
    return mode | (keepMinus1 << 2u) | ((coarseQuantClass & 0x7u) << 8u);
}

constexpr int kMode2ResidualQMax = 7;

uint16_t packMode2CoordWord(uint16_t spatialIndex, uint8_t localTime)
{
    const uint16_t safeSpatial = static_cast<uint16_t>(spatialIndex & kMaskCoordSpatial);
    const uint16_t safeLocalTime = static_cast<uint16_t>(localTime & kMaskCoordLocalTime);
    return static_cast<uint16_t>((safeLocalTime << kPackedCoordSpatialBits) | safeSpatial);
}

uint16_t unpackMode2CoordSpatial(uint16_t word)
{
    return static_cast<uint16_t>(word & kMaskCoordSpatial);
}

uint8_t unpackMode2CoordLocalTime(uint16_t word)
{
    return static_cast<uint8_t>((word >> kPackedCoordSpatialBits) & kMaskCoordLocalTime);
}

uint16_t framesPerEventBin(int encodedFrameCount)
{
    return static_cast<uint16_t>(std::max(1, (encodedFrameCount + kEventBinCount - 1) / kEventBinCount));
}

std::array<uint16_t, kEventBinCount> buildEventTimeBins(const std::vector<uint8_t>& sortedBinIds, uint16_t actualCount)
{
    std::array<uint16_t, kEventBinCount> bins{};
    bins.fill(actualCount);
    if (actualCount == 0 || sortedBinIds.empty()) return bins;

    uint32_t currentBin = 0;
    bins[0] = 0;
    for (uint16_t i = 0; i < actualCount; ++i) {
        const uint32_t targetBin = std::min<uint32_t>(kEventBinCount - 1, sortedBinIds[static_cast<size_t>(i)]);
        while (currentBin < targetBin) {
            ++currentBin;
            bins[static_cast<size_t>(currentBin)] = i;
        }
    }
    while (currentBin + 1 < kEventBinCount) {
        ++currentBin;
        bins[static_cast<size_t>(currentBin)] = actualCount;
    }
    return bins;
}

std::vector<uint8_t> packMode2ResidualBytes(const std::vector<int16_t>& values, uint16_t tierCapacity)
{
    if (tierCapacity == 0) return {};
    std::vector<uint8_t> packed(static_cast<size_t>((static_cast<uint32_t>(tierCapacity) + 1u) / 2u), 0u);
    const size_t count = std::min<size_t>(values.size(), static_cast<size_t>(tierCapacity));
    for (size_t i = 0; i < count; ++i) {
        const int q = std::clamp<int>(values[i], -7, 7);
        const uint8_t nibble = static_cast<uint8_t>(q & 0x0F);
        const size_t byteIndex = i >> 1u;
        if ((i & 1u) == 0u) packed[byteIndex] |= nibble;
        else packed[byteIndex] |= static_cast<uint8_t>(nibble << 4u);
    }
    return packed;
}

int16_t unpackMode2ResidualValue(const LeafEncoding& encoding, size_t index)
{
    if (index >= static_cast<size_t>(encoding.packedEventCount) ||
        encoding.packedResiduals.empty()) {
        return 0;
    }
    const size_t byteIndex = index >> 1u;
    if (byteIndex >= encoding.packedResiduals.size()) return 0;
    const uint8_t byte = encoding.packedResiduals[byteIndex];
    const uint8_t nibble = ((index & 1u) == 0u)
        ? static_cast<uint8_t>(byte & 0x0F)
        : static_cast<uint8_t>((byte >> 4u) & 0x0F);
    int8_t q = static_cast<int8_t>(nibble);
    if ((q & 0x08) != 0) q = static_cast<int8_t>(q - 16);
    return static_cast<int16_t>(q);
}

uint64_t estimateMode2ResidualBytes(uint16_t tierCapacity)
{
    if (tierCapacity == 0) return 0ull;
    const uint64_t coordBytes = static_cast<uint64_t>(tierCapacity) * sizeof(uint16_t);
    const uint64_t residualBytes = static_cast<uint64_t>((static_cast<uint32_t>(tierCapacity) + 1u) / 2u);
    return 4ull + 64ull + 32ull + coordBytes + residualBytes;
}

template <typename T>
void writeBinaryValue(std::ofstream& out, const T& value)
{
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void appendBinaryValue(std::vector<uint8_t>& out, const T& value)
{
    const size_t offset = out.size();
    out.resize(offset + sizeof(T));
    std::memcpy(out.data() + offset, &value, sizeof(T));
}

void appendBinaryBytes(std::vector<uint8_t>& out, const void* data, size_t size)
{
    if (size == 0) return;
    const size_t offset = out.size();
    out.resize(offset + size);
    std::memcpy(out.data() + offset, data, size);
}

template <typename T, size_t N>
void appendBinaryArray(std::vector<uint8_t>& out, const std::array<T, N>& values)
{
    appendBinaryBytes(out, values.data(), sizeof(T) * N);
}

uint16_t floatToHalfBits(float value)
{
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        const uint32_t shifted = mantissa >> static_cast<uint32_t>(1 - exponent);
        const uint32_t rounded = (shifted + 0x1000u) >> 13;
        return static_cast<uint16_t>(sign | rounded);
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    const uint32_t roundedMantissa = (mantissa + 0x1000u) >> 13;
    if (roundedMantissa == 0x400u) {
        exponent += 1;
        if (exponent >= 31) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10));
    }
    return static_cast<uint16_t>(sign |
                                 (static_cast<uint32_t>(exponent) << 10) |
                                 (roundedMantissa & 0x3FFu));
}

void appendHalfFloat(std::vector<uint8_t>& out, float value)
{
    const uint16_t bits = floatToHalfBits(value);
    appendBinaryValue(out, bits);
}

void appendPackedDenseTemporalBasis(std::vector<uint8_t>& out,
                                    const DenseTemporalBasisEncoding& encoding)
{
    if (encoding.temporalKeep <= 0 || encoding.valuesPerBasis <= 0) return;
    for (float scale : encoding.basisScales) {
        appendHalfFloat(out, scale);
    }
    if (!encoding.packedBasisValues.empty()) {
        appendBinaryBytes(out,
                          encoding.packedBasisValues.data(),
                          encoding.packedBasisValues.size());
    }
}

void appendPackedDenseTileTemporalBasis(std::vector<uint8_t>& out,
                                        const DenseTileTemporalBasisEncoding& encoding)
{
    appendBinaryValue(out, encoding.activeTileMask);
    appendBinaryValue(out, static_cast<uint16_t>(encoding.tileSize));
    appendBinaryValue(out, static_cast<uint16_t>(encoding.activeTileCount));
    if (encoding.temporalKeep <= 0 || encoding.valuesPerBasis <= 0) return;
    for (float scale : encoding.basisScales) {
        appendHalfFloat(out, scale);
    }
    if (!encoding.packedBasisValues.empty()) {
        appendBinaryBytes(out,
                          encoding.packedBasisValues.data(),
                          encoding.packedBasisValues.size());
    }
}

void appendPackedDensePatchTemporalBasis(std::vector<uint8_t>& out,
                                         const DensePatchTemporalBasisEncoding& encoding)
{
    appendBinaryValue(out, encoding.activePatchMask);
    appendBinaryValue(out, static_cast<uint16_t>(encoding.patchSize));
    appendBinaryValue(out, static_cast<uint16_t>(encoding.localResolution));
    appendBinaryValue(out, static_cast<uint16_t>(encoding.activePatchCount));
    if (encoding.temporalKeep <= 0 || encoding.valuesPerBasis <= 0) return;
    for (float scale : encoding.basisScales) {
        appendHalfFloat(out, scale);
    }
    if (!encoding.packedBasisValues.empty()) {
        appendBinaryBytes(out,
                          encoding.packedBasisValues.data(),
                          encoding.packedBasisValues.size());
    }
}

std::vector<float> computeGlobalCoarseAcScales(const std::vector<LeafEncoding>& encodings, int dctKeep)
{
    std::vector<float> scales(static_cast<size_t>(std::max(0, dctKeep - 1)), 0.0f);
    if (dctKeep <= 1) return scales;
    for (const auto& encoding : encodings) {
        if (encoding.coarseCoeffs.empty()) continue;
        const int keep = std::min(dctKeep, std::max(1, encoding.coarseKeep > 0 ? static_cast<int>(encoding.coarseKeep) : dctKeep));
        for (int i = 0; i < kCoarseControlCount; ++i) {
            const size_t base = static_cast<size_t>(i * keep);
            for (int k = 1; k < keep; ++k) {
                scales[static_cast<size_t>(k - 1)] =
                    std::max(scales[static_cast<size_t>(k - 1)],
                             std::abs(encoding.coarseCoeffs[base + static_cast<size_t>(k)]));
            }
        }
    }
    for (float& s : scales) {
        s = (s <= 1e-12f) ? 0.0f : (s / 127.0f);
    }
    return scales;
}

void appendPackedDctBlock(std::vector<uint8_t>& out,
                          const std::vector<float>& coeffs,
                          int controlCount,
                          int dctKeep,
                          const std::vector<float>& acScales)
{
    if (controlCount <= 0 || dctKeep <= 0 || coeffs.empty()) return;
    for (int i = 0; i < controlCount; ++i) {
        const size_t base = static_cast<size_t>(i * dctKeep);
        appendHalfFloat(out, coeffs[base]);
        for (int k = 1; k < dctKeep; ++k) {
            const float scale = acScales[static_cast<size_t>(k - 1)];
            int8_t q = 0;
            if (scale > 1e-12f) {
                const int quant = std::lround(coeffs[base + static_cast<size_t>(k)] / scale);
                q = static_cast<int8_t>(std::clamp(quant, -127, 127));
            }
            appendBinaryValue(out, q);
        }
    }
}

void appendPackedSignedBitstream(std::vector<uint8_t>& out,
                                 const std::vector<int16_t>& values,
                                 int bits)
{
    if (bits <= 0 || values.empty()) return;
    const uint32_t mask = (bits >= 16) ? 0xFFFFu : ((1u << bits) - 1u);
    const size_t byteCount = static_cast<size_t>((static_cast<uint64_t>(values.size()) * static_cast<uint64_t>(bits) + 7ull) / 8ull);
    std::vector<uint8_t> packed(byteCount, 0u);
    uint64_t bitOffset = 0ull;
    for (int16_t v : values) {
        const uint32_t word = static_cast<uint32_t>(static_cast<int32_t>(v)) & mask;
        const size_t byteIndex = static_cast<size_t>(bitOffset >> 3u);
        const int shift = static_cast<int>(bitOffset & 7ull);
        packed[byteIndex] |= static_cast<uint8_t>(word << shift);
        if (shift + bits > 8 && byteIndex + 1 < packed.size()) {
            packed[byteIndex + 1] |= static_cast<uint8_t>(word >> (8 - shift));
            if (shift + bits > 16 && byteIndex + 2 < packed.size()) {
                packed[byteIndex + 2] |= static_cast<uint8_t>(word >> (16 - shift));
            }
        }
        bitOffset += static_cast<uint64_t>(bits);
    }
    appendBinaryBytes(out, packed.data(), packed.size());
}

std::vector<uint8_t> buildCompactLeafPayload(const LeafEncoding& encoding,
                                             const SpatialFirstOptions& options,
                                             const std::vector<float>& coarseAcScales)
{
    // The on-disk payload is intentionally much tighter than the in-memory probe
    // state: every block starts with the packed header, then appends only the
    // mode-specific data needed by the decoder.
    std::vector<uint8_t> payload;
    payload.reserve(256);
    appendBinaryValue(payload, encoding.packedHeader);

    if (encoding.mode == BlockMode::CONSTANT) {
        appendBinaryValue(payload, encoding.constantValue);
        return payload;
    }

    const int coarseKeep = resolveCoarseKeep(options, encoding);
    appendPackedDctBlock(payload,
                         encoding.coarseCoeffs,
                         kCoarseControlCount,
                         coarseKeep,
                         coarseAcScales);

    if (encoding.mode == BlockMode::SPARSE_IMPULSE) {
        // Mode 2 keeps a GPU-friendly lookup path: scale + spatial bitmask +
        // time bins + SoA payload (coords + residuals).
        appendBinaryValue(payload, encoding.eventScale);
        appendBinaryArray(payload, encoding.spatialBitmask);
        appendBinaryArray(payload, encoding.timeBins);
        if (!encoding.packedCoords.empty()) {
            const size_t count = std::min<size_t>(encoding.packedTierCapacity, encoding.packedCoords.size());
            appendBinaryBytes(payload, encoding.packedCoords.data(), count * sizeof(uint16_t));
        }
        if (!encoding.packedResiduals.empty()) {
            appendBinaryBytes(payload,
                              encoding.packedResiduals.data(),
                              encoding.packedResiduals.size());
        }
        return payload;
    }

    if (encoding.useFineResidualGrid) {
        if (encoding.fineResidualGrid.quantBits > 0 && !encoding.fineResidualGrid.quantizedCoeffs.empty()) {
            appendBinaryValue(payload, encoding.fineResidualGrid.blockScale);
            appendPackedSignedBitstream(payload,
                                        encoding.fineResidualGrid.quantizedCoeffs,
                                        encoding.fineResidualGrid.quantBits);
        } else {
            const int keep = encoding.fineResidualGrid.dctKeep;
            const int controlCount =
                encoding.fineResidualGrid.resolution *
                encoding.fineResidualGrid.resolution *
                encoding.fineResidualGrid.resolution;
            std::vector<float> fineAcScales(static_cast<size_t>(std::max(0, keep - 1)), 0.0f);
            for (int i = 0; i < controlCount; ++i) {
                const size_t base = static_cast<size_t>(i * keep);
                for (int k = 1; k < keep; ++k) {
                    fineAcScales[static_cast<size_t>(k - 1)] =
                        std::max(fineAcScales[static_cast<size_t>(k - 1)],
                                 std::abs(encoding.fineResidualGrid.coeffs[base + static_cast<size_t>(k)]));
                }
            }
            for (float& s : fineAcScales) s = (s <= 1e-12f) ? 0.0f : (s / 127.0f);
            appendPackedDctBlock(payload,
                                 encoding.fineResidualGrid.coeffs,
                                 controlCount,
                                 keep,
                                 fineAcScales);
        }
        return payload;
    }

    if (encoding.useDenseResidualBfp) {
        for (float scale : encoding.denseResidualBfp.frameScales) {
            appendHalfFloat(payload, scale);
        }
        if (!encoding.denseResidualBfp.packedValues.empty()) {
            appendBinaryBytes(payload,
                              encoding.denseResidualBfp.packedValues.data(),
                              encoding.denseResidualBfp.packedValues.size());
        }
        return payload;
    }

    if (encoding.useDenseTemporalBasis) {
        appendPackedDenseTemporalBasis(payload, encoding.denseTemporalBasis);
        return payload;
    }

    if (encoding.useDenseTileTemporalBasis) {
        appendPackedDenseTileTemporalBasis(payload, encoding.denseTileTemporalBasis);
        return payload;
    }

    if (encoding.useDensePatchTemporalBasis) {
        appendPackedDensePatchTemporalBasis(payload, encoding.densePatchTemporalBasis);
        return payload;
    }

    return payload;
}

uint64_t writeProbeFile(const std::filesystem::path& path,
                        const RawVolume4D& volume,
                        const SpatialFirstOptions& options,
                        const std::vector<LeafEncoding>& encodings)
{
    auto alignPayloadBytes = [](size_t bytes) -> size_t {
        return static_cast<size_t>(((static_cast<uint64_t>(bytes) + 3ull) / 4ull) * 4ull);
    };
    std::filesystem::create_directories(path.parent_path());
    const int maxCoarseKeep = computeMaxCoarseKeep(encodings, options);
    const std::vector<float> coarseAcScales = computeGlobalCoarseAcScales(encodings, maxCoarseKeep);
    std::vector<uint32_t> offsetsWords(encodings.size() + 1u, 0u);
    std::vector<uint8_t> payloadPool;
    payloadPool.reserve(encodings.size() * 256ull);

    // Build a classic offset table + payload pool so saved size matches the
    // logical payload estimate instead of dumping verbose per-leaf state.
    for (size_t i = 0; i < encodings.size(); ++i) {
        offsetsWords[i] = static_cast<uint32_t>(payloadPool.size() / 4u);
        auto payload = buildCompactLeafPayload(encodings[i], options, coarseAcScales);
        const size_t alignedSize = alignPayloadBytes(payload.size());
        payload.resize(alignedSize, 0u);
        appendBinaryBytes(payloadPool, payload.data(), payload.size());
    }
    offsetsWords.back() = static_cast<uint32_t>(payloadPool.size() / 4u);

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open output probe file: " + path.string());
    }

    const char magic[8] = {'V','B','T','P','A','C','K','4'};
    out.write(magic, sizeof(magic));
    const uint32_t version = 4;
    writeBinaryValue(out, version);
    writeBinaryValue(out, static_cast<uint32_t>(volume.meta.width));
    writeBinaryValue(out, static_cast<uint32_t>(volume.meta.height));
    writeBinaryValue(out, static_cast<uint32_t>(volume.meta.depth));
    writeBinaryValue(out, static_cast<uint32_t>(volume.meta.frames));
    writeBinaryValue(out, static_cast<uint32_t>(options.leafSize));
    writeBinaryValue(out, static_cast<uint32_t>(options.coarseResolution));
    writeBinaryValue(out, static_cast<uint32_t>(maxCoarseKeep));
    writeBinaryValue(out, static_cast<uint32_t>(encodings.size()));
    writeBinaryValue(out, static_cast<uint32_t>(static_cast<uint32_t>(options.profile.type)));
    writeBinaryValue(out, static_cast<uint32_t>(coarseAcScales.size()));
    if (!coarseAcScales.empty()) {
        out.write(reinterpret_cast<const char*>(coarseAcScales.data()),
                  static_cast<std::streamsize>(sizeof(float) * coarseAcScales.size()));
    }
    out.write(reinterpret_cast<const char*>(offsetsWords.data()),
              static_cast<std::streamsize>(sizeof(uint32_t) * offsetsWords.size()));
    if (!payloadPool.empty()) {
        out.write(reinterpret_cast<const char*>(payloadPool.data()),
                  static_cast<std::streamsize>(payloadPool.size()));
    }
    out.flush();
    out.close();
    return std::filesystem::file_size(path);
}

std::array<float, 4> fitCellBasis4(const std::array<float, 8>& values)
{
    // Least-squares fit for basis [1, sx, sy, sz] on a 2x2x2 cell,
    // where sx/sy/sz are in {-1,+1}.
    std::array<float, 4> coeffs{};
    for (int dz = 0; dz < 2; ++dz) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                const int idx = (dz * 2 + dy) * 2 + dx;
                const float sx = dx ? 1.0f : -1.0f;
                const float sy = dy ? 1.0f : -1.0f;
                const float sz = dz ? 1.0f : -1.0f;
                const float v = values[static_cast<size_t>(idx)];
                coeffs[0] += v;
                coeffs[1] += v * sx;
                coeffs[2] += v * sy;
                coeffs[3] += v * sz;
            }
        }
    }
    for (float& c : coeffs) c *= 0.125f;
    return coeffs;
}

float evalCellBasis4(const std::array<float, 4>& coeffs, int dx, int dy, int dz, int cellSize)
{
    const float sx = (cellSize <= 1) ? 0.0f : (dx ? 1.0f : -1.0f);
    const float sy = (cellSize <= 1) ? 0.0f : (dy ? 1.0f : -1.0f);
    const float sz = (cellSize <= 1) ? 0.0f : (dz ? 1.0f : -1.0f);
    return coeffs[0] + coeffs[1] * sx + coeffs[2] * sy + coeffs[3] * sz;
}

bool invertMatrix64(std::array<std::array<double, kCoarseControlCount>, kCoarseControlCount>& m)
{
    std::array<std::array<double, kCoarseControlCount>, kCoarseControlCount> inv{};
    for (int i = 0; i < kCoarseControlCount; ++i) inv[i][i] = 1.0;

    for (int c = 0; c < kCoarseControlCount; ++c) {
        int pivot = c;
        double best = std::abs(m[c][c]);
        for (int r = c + 1; r < kCoarseControlCount; ++r) {
            const double cand = std::abs(m[r][c]);
            if (cand > best) {
                best = cand;
                pivot = r;
            }
        }
        if (best < 1e-12) return false;
        if (pivot != c) {
            std::swap(m[pivot], m[c]);
            std::swap(inv[pivot], inv[c]);
        }
        const double diag = m[c][c];
        for (int j = 0; j < kCoarseControlCount; ++j) {
            m[c][j] /= diag;
            inv[c][j] /= diag;
        }
        for (int r = 0; r < kCoarseControlCount; ++r) {
            if (r == c) continue;
            const double factor = m[r][c];
            if (std::abs(factor) < 1e-18) continue;
            for (int j = 0; j < kCoarseControlCount; ++j) {
                m[r][j] -= factor * m[c][j];
                inv[r][j] -= factor * inv[c][j];
            }
        }
    }
    m = inv;
    return true;
}

PseudoInverse4x4x4 buildPseudoInverse4x4x4()
{
    std::array<std::array<double, kCoarseControlCount>, kLeafVoxelCount> A{};
    for (int z = 0; z < 8; ++z) {
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                const int row = localIndex(x, y, z);
                const double fx = (static_cast<double>(x) / 7.0) * 3.0;
                const double fy = (static_cast<double>(y) / 7.0) * 3.0;
                const double fz = (static_cast<double>(z) / 7.0) * 3.0;
                const int x0 = std::min(2, std::max(0, static_cast<int>(std::floor(fx))));
                const int y0 = std::min(2, std::max(0, static_cast<int>(std::floor(fy))));
                const int z0 = std::min(2, std::max(0, static_cast<int>(std::floor(fz))));
                const int x1 = x0 + 1;
                const int y1 = y0 + 1;
                const int z1 = z0 + 1;
                const double tx = fx - static_cast<double>(x0);
                const double ty = fy - static_cast<double>(y0);
                const double tz = fz - static_cast<double>(z0);

                auto add = [&](int cx, int cy, int cz, double w) {
                    A[row][(cz * 4 + cy) * 4 + cx] += w;
                };
                add(x0, y0, z0, (1.0 - tx) * (1.0 - ty) * (1.0 - tz));
                add(x1, y0, z0, tx * (1.0 - ty) * (1.0 - tz));
                add(x0, y1, z0, (1.0 - tx) * ty * (1.0 - tz));
                add(x1, y1, z0, tx * ty * (1.0 - tz));
                add(x0, y0, z1, (1.0 - tx) * (1.0 - ty) * tz);
                add(x1, y0, z1, tx * (1.0 - ty) * tz);
                add(x0, y1, z1, (1.0 - tx) * ty * tz);
                add(x1, y1, z1, tx * ty * tz);
            }
        }
    }

    std::array<std::array<double, kCoarseControlCount>, kCoarseControlCount> ata{};
    for (int r = 0; r < kCoarseControlCount; ++r) {
        for (int c = 0; c < kCoarseControlCount; ++c) {
            double sum = 0.0;
            for (int i = 0; i < kLeafVoxelCount; ++i) sum += A[i][r] * A[i][c];
            ata[r][c] = sum;
        }
    }
    if (!invertMatrix64(ata)) {
        throw std::runtime_error("Failed to invert GRID4 least-squares normal matrix");
    }

    PseudoInverse4x4x4 result;
    for (int r = 0; r < kCoarseControlCount; ++r) {
        for (int i = 0; i < kLeafVoxelCount; ++i) {
            double sum = 0.0;
            for (int k = 0; k < kCoarseControlCount; ++k) sum += ata[r][k] * A[i][k];
            result.pinv[r][i] = sum;
        }
    }
    return result;
}

const PseudoInverse4x4x4& coarsePseudoInverse()
{
    static const PseudoInverse4x4x4 pinv = buildPseudoInverse4x4x4();
    return pinv;
}

DynamicPseudoInverseGrid buildPseudoInverseGrid(int controlResolution)
{
    DynamicPseudoInverseGrid result;
    result.resolution = controlResolution;
    result.controlCount = controlResolution * controlResolution * controlResolution;
    result.pinv.resize(static_cast<size_t>(result.controlCount * kLeafVoxelCount), 0.0);

    std::vector<double> A(static_cast<size_t>(kLeafVoxelCount * result.controlCount), 0.0);
    for (int z = 0; z < 8; ++z) {
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                const int row = localIndex(x, y, z);
                const double fx = (static_cast<double>(x) / 7.0) * static_cast<double>(controlResolution - 1);
                const double fy = (static_cast<double>(y) / 7.0) * static_cast<double>(controlResolution - 1);
                const double fz = (static_cast<double>(z) / 7.0) * static_cast<double>(controlResolution - 1);
                const int x0 = std::min(controlResolution - 2, std::max(0, static_cast<int>(std::floor(fx))));
                const int y0 = std::min(controlResolution - 2, std::max(0, static_cast<int>(std::floor(fy))));
                const int z0 = std::min(controlResolution - 2, std::max(0, static_cast<int>(std::floor(fz))));
                const int x1 = x0 + 1;
                const int y1 = y0 + 1;
                const int z1 = z0 + 1;
                const double tx = fx - static_cast<double>(x0);
                const double ty = fy - static_cast<double>(y0);
                const double tz = fz - static_cast<double>(z0);

                auto add = [&](int cx, int cy, int cz, double w) {
                    const int col = (cz * controlResolution + cy) * controlResolution + cx;
                    A[static_cast<size_t>(row * result.controlCount + col)] += w;
                };
                add(x0, y0, z0, (1.0 - tx) * (1.0 - ty) * (1.0 - tz));
                add(x1, y0, z0, tx * (1.0 - ty) * (1.0 - tz));
                add(x0, y1, z0, (1.0 - tx) * ty * (1.0 - tz));
                add(x1, y1, z0, tx * ty * (1.0 - tz));
                add(x0, y0, z1, (1.0 - tx) * (1.0 - ty) * tz);
                add(x1, y0, z1, tx * (1.0 - ty) * tz);
                add(x0, y1, z1, (1.0 - tx) * ty * tz);
                add(x1, y1, z1, tx * ty * tz);
            }
        }
    }

    std::vector<double> ata(static_cast<size_t>(result.controlCount * result.controlCount), 0.0);
    for (int r = 0; r < result.controlCount; ++r) {
        for (int c = 0; c < result.controlCount; ++c) {
            double sum = 0.0;
            for (int i = 0; i < kLeafVoxelCount; ++i) {
                sum += A[static_cast<size_t>(i * result.controlCount + r)] * A[static_cast<size_t>(i * result.controlCount + c)];
            }
            ata[static_cast<size_t>(r * result.controlCount + c)] = sum;
        }
    }

    std::vector<double> inv = ata;
    std::vector<double> rhs(static_cast<size_t>(result.controlCount), 0.0);
    for (int col = 0; col < result.controlCount; ++col) {
        std::fill(inv.begin(), inv.end(), 0.0);
        for (int r = 0; r < result.controlCount; ++r) {
            for (int c = 0; c < result.controlCount; ++c) {
                double sum = 0.0;
                for (int i = 0; i < kLeafVoxelCount; ++i) {
                    sum += A[static_cast<size_t>(i * result.controlCount + r)] * A[static_cast<size_t>(i * result.controlCount + c)];
                }
                inv[static_cast<size_t>(r * result.controlCount + c)] = sum;
            }
        }
        std::fill(rhs.begin(), rhs.end(), 0.0);
        rhs[static_cast<size_t>(col)] = 1.0;
        for (int i = 0; i < result.controlCount; ++i) {
            int pivot = i;
            double best = std::abs(inv[static_cast<size_t>(i * result.controlCount + i)]);
            for (int r = i + 1; r < result.controlCount; ++r) {
                const double cand = std::abs(inv[static_cast<size_t>(r * result.controlCount + i)]);
                if (cand > best) {
                    best = cand;
                    pivot = r;
                }
            }
            if (best < 1e-12) {
                throw std::runtime_error("Failed to invert fine-grid least-squares normal matrix");
            }
            if (pivot != i) {
                for (int c = i; c < result.controlCount; ++c) {
                    std::swap(inv[static_cast<size_t>(i * result.controlCount + c)], inv[static_cast<size_t>(pivot * result.controlCount + c)]);
                }
                std::swap(rhs[static_cast<size_t>(i)], rhs[static_cast<size_t>(pivot)]);
            }
            const double diag = inv[static_cast<size_t>(i * result.controlCount + i)];
            for (int c = i; c < result.controlCount; ++c) inv[static_cast<size_t>(i * result.controlCount + c)] /= diag;
            rhs[static_cast<size_t>(i)] /= diag;
            for (int r = 0; r < result.controlCount; ++r) {
                if (r == i) continue;
                const double factor = inv[static_cast<size_t>(r * result.controlCount + i)];
                if (std::abs(factor) < 1e-18) continue;
                for (int c = i; c < result.controlCount; ++c) {
                    inv[static_cast<size_t>(r * result.controlCount + c)] -= factor * inv[static_cast<size_t>(i * result.controlCount + c)];
                }
                rhs[static_cast<size_t>(r)] -= factor * rhs[static_cast<size_t>(i)];
            }
        }

        for (int i = 0; i < kLeafVoxelCount; ++i) {
            double sum = 0.0;
            for (int k = 0; k < result.controlCount; ++k) {
                sum += rhs[static_cast<size_t>(k)] * A[static_cast<size_t>(i * result.controlCount + k)];
            }
            result.pinv[static_cast<size_t>(col * kLeafVoxelCount + i)] = sum;
        }
    }
    return result;
}

const DynamicPseudoInverseGrid& finePseudoInverseGrid(int resolution)
{
    switch (resolution) {
    case 2: {
        static const DynamicPseudoInverseGrid pinv = buildPseudoInverseGrid(2);
        return pinv;
    }
    case 3: {
        static const DynamicPseudoInverseGrid pinv = buildPseudoInverseGrid(3);
        return pinv;
    }
    case 4: {
        static const DynamicPseudoInverseGrid pinv = buildPseudoInverseGrid(4);
        return pinv;
    }
    case 5: {
        static const DynamicPseudoInverseGrid pinv = buildPseudoInverseGrid(5);
        return pinv;
    }
    case 6: {
        static const DynamicPseudoInverseGrid pinv = buildPseudoInverseGrid(6);
        return pinv;
    }
    case 7: {
        static const DynamicPseudoInverseGrid pinv = buildPseudoInverseGrid(7);
        return pinv;
    }
    case 8: {
        static const DynamicPseudoInverseGrid pinv = buildPseudoInverseGrid(8);
        return pinv;
    }
    default:
        throw std::runtime_error("Unsupported fine-grid resolution");
    }
}

PatchPseudoInverseGrid buildPatchPseudoInverseGrid(int patchSize, int controlResolution)
{
    PatchPseudoInverseGrid result;
    result.patchSize = patchSize;
    result.resolution = controlResolution;
    result.controlCount = controlResolution * controlResolution * controlResolution;
    const int voxelCount = patchSize * patchSize * patchSize;
    result.pinv.resize(static_cast<size_t>(result.controlCount * voxelCount), 0.0);

    std::vector<double> A(static_cast<size_t>(voxelCount * result.controlCount), 0.0);
    auto patchLocalIndex = [patchSize](int x, int y, int z) {
        return (z * patchSize + y) * patchSize + x;
    };
    for (int z = 0; z < patchSize; ++z) {
        for (int y = 0; y < patchSize; ++y) {
            for (int x = 0; x < patchSize; ++x) {
                const int row = patchLocalIndex(x, y, z);
                const double fx = (static_cast<double>(x) / static_cast<double>(patchSize - 1)) * static_cast<double>(controlResolution - 1);
                const double fy = (static_cast<double>(y) / static_cast<double>(patchSize - 1)) * static_cast<double>(controlResolution - 1);
                const double fz = (static_cast<double>(z) / static_cast<double>(patchSize - 1)) * static_cast<double>(controlResolution - 1);
                const int x0 = std::min(controlResolution - 2, std::max(0, static_cast<int>(std::floor(fx))));
                const int y0 = std::min(controlResolution - 2, std::max(0, static_cast<int>(std::floor(fy))));
                const int z0 = std::min(controlResolution - 2, std::max(0, static_cast<int>(std::floor(fz))));
                const int x1 = x0 + 1;
                const int y1 = y0 + 1;
                const int z1 = z0 + 1;
                const double tx = fx - static_cast<double>(x0);
                const double ty = fy - static_cast<double>(y0);
                const double tz = fz - static_cast<double>(z0);
                auto add = [&](int cx, int cy, int cz, double w) {
                    const int col = (cz * controlResolution + cy) * controlResolution + cx;
                    A[static_cast<size_t>(row * result.controlCount + col)] += w;
                };
                add(x0, y0, z0, (1.0 - tx) * (1.0 - ty) * (1.0 - tz));
                add(x1, y0, z0, tx * (1.0 - ty) * (1.0 - tz));
                add(x0, y1, z0, (1.0 - tx) * ty * (1.0 - tz));
                add(x1, y1, z0, tx * ty * (1.0 - tz));
                add(x0, y0, z1, (1.0 - tx) * (1.0 - ty) * tz);
                add(x1, y0, z1, tx * (1.0 - ty) * tz);
                add(x0, y1, z1, (1.0 - tx) * ty * tz);
                add(x1, y1, z1, tx * ty * tz);
            }
        }
    }

    std::vector<double> inv(static_cast<size_t>(result.controlCount * result.controlCount), 0.0);
    std::vector<double> rhs(static_cast<size_t>(result.controlCount), 0.0);
    for (int col = 0; col < result.controlCount; ++col) {
        for (int r = 0; r < result.controlCount; ++r) {
            for (int c = 0; c < result.controlCount; ++c) {
                double sum = 0.0;
                for (int i = 0; i < voxelCount; ++i) {
                    sum += A[static_cast<size_t>(i * result.controlCount + r)] * A[static_cast<size_t>(i * result.controlCount + c)];
                }
                inv[static_cast<size_t>(r * result.controlCount + c)] = sum;
            }
        }
        std::fill(rhs.begin(), rhs.end(), 0.0);
        rhs[static_cast<size_t>(col)] = 1.0;
        for (int i = 0; i < result.controlCount; ++i) {
            int pivot = i;
            double best = std::abs(inv[static_cast<size_t>(i * result.controlCount + i)]);
            for (int r = i + 1; r < result.controlCount; ++r) {
                const double cand = std::abs(inv[static_cast<size_t>(r * result.controlCount + i)]);
                if (cand > best) {
                    best = cand;
                    pivot = r;
                }
            }
            if (best < 1e-12) {
                throw std::runtime_error("Failed to invert patch-grid least-squares normal matrix");
            }
            if (pivot != i) {
                for (int c = i; c < result.controlCount; ++c) {
                    std::swap(inv[static_cast<size_t>(i * result.controlCount + c)], inv[static_cast<size_t>(pivot * result.controlCount + c)]);
                }
                std::swap(rhs[static_cast<size_t>(i)], rhs[static_cast<size_t>(pivot)]);
            }
            const double diag = inv[static_cast<size_t>(i * result.controlCount + i)];
            for (int c = i; c < result.controlCount; ++c) inv[static_cast<size_t>(i * result.controlCount + c)] /= diag;
            rhs[static_cast<size_t>(i)] /= diag;
            for (int r = 0; r < result.controlCount; ++r) {
                if (r == i) continue;
                const double factor = inv[static_cast<size_t>(r * result.controlCount + i)];
                if (std::abs(factor) < 1e-18) continue;
                for (int c = i; c < result.controlCount; ++c) {
                    inv[static_cast<size_t>(r * result.controlCount + c)] -= factor * inv[static_cast<size_t>(i * result.controlCount + c)];
                }
                rhs[static_cast<size_t>(r)] -= factor * rhs[static_cast<size_t>(i)];
            }
        }
        for (int i = 0; i < voxelCount; ++i) {
            double sum = 0.0;
            for (int k = 0; k < result.controlCount; ++k) {
                sum += rhs[static_cast<size_t>(k)] * A[static_cast<size_t>(i * result.controlCount + k)];
            }
            result.pinv[static_cast<size_t>(col * voxelCount + i)] = sum;
        }
    }
    return result;
}

const PatchPseudoInverseGrid& patchPseudoInverseGrid(int patchSize, int resolution)
{
    if (patchSize == 4 && resolution == 2) {
        static const PatchPseudoInverseGrid pinv = buildPatchPseudoInverseGrid(4, 2);
        return pinv;
    }
    throw std::runtime_error("Unsupported patch pseudo-inverse grid");
}

std::array<float, kLeafVoxelCount> loadLeafFrame(const RawVolume4D& volume, int bx, int by, int bz, int t)
{
    std::array<float, kLeafVoxelCount> leaf{};
    const int x0 = bx * 8;
    const int y0 = by * 8;
    const int z0 = bz * 8;
    for (int z = 0; z < 8; ++z) {
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                const int gx = std::clamp(x0 + x, 0, volume.meta.width - 1);
                const int gy = std::clamp(y0 + y, 0, volume.meta.height - 1);
                const int gz = std::clamp(z0 + z, 0, volume.meta.depth - 1);
                leaf[localIndex(x, y, z)] = volume.at(gx, gy, gz, t);
            }
        }
    }
    return leaf;
}

std::array<float, kCoarseControlCount> computeCoarseControls(const std::array<float, kLeafVoxelCount>& leaf)
{
    std::array<float, kCoarseControlCount> ctrl{};
    const auto& pinv = coarsePseudoInverse();
    for (int r = 0; r < kCoarseControlCount; ++r) {
        double sum = 0.0;
        for (int i = 0; i < kLeafVoxelCount; ++i) sum += pinv.pinv[r][i] * static_cast<double>(leaf[i]);
        ctrl[r] = static_cast<float>(sum);
    }
    return ctrl;
}

std::vector<float> computeGridControls(const std::array<float, kLeafVoxelCount>& leaf, int resolution)
{
    const auto& pinv = finePseudoInverseGrid(resolution);
    std::vector<float> ctrl(static_cast<size_t>(pinv.controlCount), 0.0f);
    for (int r = 0; r < pinv.controlCount; ++r) {
        double sum = 0.0;
        for (int i = 0; i < kLeafVoxelCount; ++i) {
            sum += pinv.pinv[static_cast<size_t>(r * kLeafVoxelCount + i)] * static_cast<double>(leaf[static_cast<size_t>(i)]);
        }
        ctrl[static_cast<size_t>(r)] = static_cast<float>(sum);
    }
    return ctrl;
}

float sampleTrilinear4(const std::array<float, kCoarseControlCount>& ctrl, int x, int y, int z)
{
    const float fx = (static_cast<float>(x) / 7.0f) * 3.0f;
    const float fy = (static_cast<float>(y) / 7.0f) * 3.0f;
    const float fz = (static_cast<float>(z) / 7.0f) * 3.0f;
    const int x0 = std::min(2, std::max(0, static_cast<int>(std::floor(fx))));
    const int y0 = std::min(2, std::max(0, static_cast<int>(std::floor(fy))));
    const int z0 = std::min(2, std::max(0, static_cast<int>(std::floor(fz))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);
    auto at = [&](int cx, int cy, int cz) -> float { return ctrl[(cz * 4 + cy) * 4 + cx]; };
    const float c00 = at(x0, y0, z0) * (1.0f - tx) + at(x1, y0, z0) * tx;
    const float c01 = at(x0, y0, z1) * (1.0f - tx) + at(x1, y0, z1) * tx;
    const float c10 = at(x0, y1, z0) * (1.0f - tx) + at(x1, y1, z0) * tx;
    const float c11 = at(x0, y1, z1) * (1.0f - tx) + at(x1, y1, z1) * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

float sampleTrilinearGrid(const std::vector<float>& ctrl, int resolution, int x, int y, int z)
{
    const float fx = (static_cast<float>(x) / 7.0f) * static_cast<float>(resolution - 1);
    const float fy = (static_cast<float>(y) / 7.0f) * static_cast<float>(resolution - 1);
    const float fz = (static_cast<float>(z) / 7.0f) * static_cast<float>(resolution - 1);
    const int x0 = std::min(resolution - 2, std::max(0, static_cast<int>(std::floor(fx))));
    const int y0 = std::min(resolution - 2, std::max(0, static_cast<int>(std::floor(fy))));
    const int z0 = std::min(resolution - 2, std::max(0, static_cast<int>(std::floor(fz))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);
    auto at = [&](int cx, int cy, int cz) -> float {
        return ctrl[static_cast<size_t>((cz * resolution + cy) * resolution + cx)];
    };
    const float c00 = at(x0, y0, z0) * (1.0f - tx) + at(x1, y0, z0) * tx;
    const float c01 = at(x0, y0, z1) * (1.0f - tx) + at(x1, y0, z1) * tx;
    const float c10 = at(x0, y1, z0) * (1.0f - tx) + at(x1, y1, z0) * tx;
    const float c11 = at(x0, y1, z1) * (1.0f - tx) + at(x1, y1, z1) * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

double computePsnr(double rmse, double peak)
{
    if (rmse <= 0.0) return 120.0;
    return 20.0 * std::log10(std::max(peak, 1e-12) / rmse);
}

uint32_t packHeader(const FieldProfile& profile, const SpatialFirstOptions& options, const LeafEncoding& encoding)
{
    if (useScientificHeaderV2(profile, options)) {
        return packScientificHeaderV2(options, encoding);
    }

    const uint32_t modeBits = static_cast<uint32_t>(encoding.mode) & 0x3u;
    const uint32_t startBits = static_cast<uint32_t>(encoding.startFrame) & 0x7Fu;
    const uint32_t endBits = static_cast<uint32_t>(encoding.endFrame) & 0x7Fu;
    const uint32_t configBits = static_cast<uint32_t>(encoding.config);
    return modeBits | (startBits << 2u) | (endBits << 9u) | (configBits << 16u);
}

uint64_t alignWords(uint64_t bytes)
{
    return (bytes + 3ull) / 4ull;
}

std::vector<int> samplePhasesForAxis(int step, int leafSize)
{
    if (step >= leafSize && (step % leafSize) == 0) {
        return {0, leafSize / 2};
    }
    return {0};
}

int clampToEncodedFrame(const LeafEncoding& encoding, int t);
std::vector<float> gatherSeriesAtSample(const RawVolume4D& volume, int x, int y, int z);
float decodeKeyframeSeriesAt(const std::vector<float>& series, const std::vector<int>& keys, int t);
float decodeDenseResidualBfpAt(const DenseResidualBfpEncoding& encoding, int x, int y, int z, int t);

bool hitsRegularSamplePhase(int coord, int step, const std::vector<int>& phases)
{
    if (step <= 0) return false;
    for (int phase : phases) {
        if (coord >= phase && ((coord - phase) % step) == 0) {
            return true;
        }
    }
    return false;
}

struct DecodedLeafSample {
    float coarse = 0.0f;
    float residual = 0.0f;
    float pred = 0.0f;
};

std::vector<float> buildDctDecodeRow(int totalLength, int keep, int index)
{
    constexpr double kPiLocal = 3.14159265358979323846;
    std::vector<float> row(static_cast<size_t>(keep), 0.0f);
    const double invN = 1.0 / static_cast<double>(totalLength);
    for (int k = 0; k < keep; ++k) {
        const double alpha = (k == 0) ? std::sqrt(invN) : std::sqrt(2.0 * invN);
        row[static_cast<size_t>(k)] = static_cast<float>(
            alpha * std::cos((kPiLocal / static_cast<double>(totalLength)) *
                             (static_cast<double>(index) + 0.5) *
                             static_cast<double>(k)));
    }
    return row;
}

void decodeDctCoeffBlockAt(const std::vector<float>& flatCoeffs,
                           int seriesCount,
                           int keep,
                           const std::vector<float>& row,
                           float* out)
{
    for (int i = 0; i < seriesCount; ++i) {
        const int off = i * keep;
        double sum = 0.0;
        for (int k = 0; k < keep; ++k) {
            sum += static_cast<double>(flatCoeffs[static_cast<size_t>(off + k)]) *
                   static_cast<double>(row[static_cast<size_t>(k)]);
        }
        out[i] = static_cast<float>(sum);
    }
}

void decodeQuantizedDctCoeffBlockAt(const std::vector<int16_t>& flatCoeffs,
                                    float blockScale,
                                    int seriesCount,
                                    int keep,
                                    const std::vector<float>& row,
                                    float* out)
{
    for (int i = 0; i < seriesCount; ++i) {
        const int off = i * keep;
        double sum = 0.0;
        for (int k = 0; k < keep; ++k) {
            sum += static_cast<double>(flatCoeffs[static_cast<size_t>(off + k)]) *
                   static_cast<double>(blockScale) *
                   static_cast<double>(row[static_cast<size_t>(k)]);
        }
        out[i] = static_cast<float>(sum);
    }
}

void buildSparseResidualFrameMap(const LeafEncoding& encoding,
                                 int localT,
                                 std::array<float, kLeafVoxelCount>& residualMap)
{
    residualMap.fill(0.0f);
    if (encoding.packedEventCount == 0 ||
        encoding.packedCoords.empty() ||
        encoding.packedResiduals.empty() ||
        encoding.eventScale <= 0.0f) return;

    const uint16_t fpb = framesPerEventBin(encoding.encodedFrameCount);
    const uint16_t binIdx = static_cast<uint16_t>(std::min<int>(kEventBinCount - 1, localT / std::max<int>(1, fpb)));
    const uint8_t localTimeInBin = static_cast<uint8_t>(localT - static_cast<int>(binIdx) * static_cast<int>(fpb));
    const uint16_t start = encoding.timeBins[static_cast<size_t>(binIdx)];
    const uint16_t end = (binIdx + 1u < kEventBinCount)
        ? encoding.timeBins[static_cast<size_t>(binIdx + 1u)]
        : encoding.packedEventCount;
    if (start >= end) return;

    for (uint16_t i = start; i < end; ++i) {
        const uint16_t coord = encoding.packedCoords[static_cast<size_t>(i)];
        const uint8_t evtLocalTime = unpackMode2CoordLocalTime(coord);
        if (evtLocalTime < localTimeInBin) continue;
        if (evtLocalTime > localTimeInBin) break;
        const uint16_t idx = unpackMode2CoordSpatial(coord);
        residualMap[static_cast<size_t>(idx)] +=
            static_cast<float>(unpackMode2ResidualValue(encoding, i)) * encoding.eventScale;
    }
}

DecodedLeafSample decodeLeafSampleAt(const SpatialFirstOptions& options,
                                     const LeafEncoding& encoding,
                                     int lx, int ly, int lz, int t)
{
    const int lidx = localIndex(lx, ly, lz);
    const int localT = clampToEncodedFrame(encoding, t);
    const bool withinWindow =
        t >= static_cast<int>(encoding.startFrame) &&
        t <= static_cast<int>(encoding.endFrame);

    if (encoding.mode == BlockMode::CONSTANT) {
        return {encoding.constantValue, 0.0f, encoding.constantValue};
    }

    const int coarseKeep = resolveCoarseKeep(options, encoding);
    std::array<float, kCoarseControlCount> ctrl{};
    for (int i = 0; i < kCoarseControlCount; ++i) {
        const int off = i * coarseKeep;
        std::vector<float> coeffs(static_cast<size_t>(coarseKeep), 0.0f);
        for (int k = 0; k < coarseKeep; ++k) {
            coeffs[static_cast<size_t>(k)] = encoding.coarseCoeffs[static_cast<size_t>(off + k)];
        }
        ctrl[i] = dctDecodeAt(coeffs, encoding.encodedFrameCount, localT);
    }

    const float coarse = sampleTrilinear4(ctrl, lx, ly, lz);
    float residual = 0.0f;
    if (encoding.useFineResidualGrid) {
        std::vector<float> fineCtrl(static_cast<size_t>(encoding.fineResidualGrid.resolution *
                                                        encoding.fineResidualGrid.resolution *
                                                        encoding.fineResidualGrid.resolution), 0.0f);
        for (size_t i = 0; i < fineCtrl.size(); ++i) {
            const int off = static_cast<int>(i) * encoding.fineResidualGrid.dctKeep;
            std::vector<float> coeffs(static_cast<size_t>(encoding.fineResidualGrid.dctKeep), 0.0f);
            if (!encoding.fineResidualGrid.quantizedCoeffs.empty()) {
                for (int k = 0; k < encoding.fineResidualGrid.dctKeep; ++k) {
                    coeffs[static_cast<size_t>(k)] =
                        static_cast<float>(encoding.fineResidualGrid.quantizedCoeffs[static_cast<size_t>(off + k)]) *
                        encoding.fineResidualGrid.blockScale;
                }
            } else {
                for (int k = 0; k < encoding.fineResidualGrid.dctKeep; ++k) {
                    coeffs[static_cast<size_t>(k)] = encoding.fineResidualGrid.coeffs[static_cast<size_t>(off + k)];
                }
            }
            fineCtrl[i] = dctDecodeAt(coeffs, encoding.encodedFrameCount, localT);
        }
        residual = sampleTrilinearGrid(fineCtrl, encoding.fineResidualGrid.resolution, lx, ly, lz);
    } else if (encoding.useDenseResidualBfp) {
        residual = decodeDenseResidualBfpAt(encoding.denseResidualBfp, lx, ly, lz, localT);
    } else if (encoding.useDenseTemporalBasis) {
        residual = decodeDenseTemporalBasisAt(encoding.denseTemporalBasis, lx, ly, lz, localT, encoding.encodedFrameCount);
    } else if (encoding.useDenseTileTemporalBasis) {
        residual = decodeDenseTileTemporalBasisAt(encoding.denseTileTemporalBasis, lx, ly, lz, localT, encoding.encodedFrameCount);
    } else if (encoding.useDensePatchTemporalBasis) {
        residual = decodeDensePatchTemporalBasisAt(encoding.densePatchTemporalBasis, lx, ly, lz, localT, encoding.encodedFrameCount);
    } else {
        if (withinWindow &&
            encoding.packedEventCount > 0 &&
            !encoding.packedCoords.empty() &&
            !encoding.packedResiduals.empty() &&
            encoding.eventScale > 0.0f) {
            // Mode 2 query path is: spatial bitmask reject -> time-bin narrow ->
            // binary search the packed (localTimeInBin, spatial) coord stream.
            const uint32_t bitWord = encoding.spatialBitmask[static_cast<size_t>(lidx >> 5)];
            const uint32_t bitMask = 1u << (lidx & 31);
            if ((bitWord & bitMask) != 0u) {
                const uint16_t fpb = framesPerEventBin(encoding.encodedFrameCount);
                const uint16_t binIdx = static_cast<uint16_t>(std::min<int>(kEventBinCount - 1, localT / std::max<int>(1, fpb)));
                const uint8_t localTimeInBin = static_cast<uint8_t>(localT - static_cast<int>(binIdx) * static_cast<int>(fpb));
                const uint16_t start = encoding.timeBins[static_cast<size_t>(binIdx)];
                const uint16_t end = (binIdx + 1u < kEventBinCount)
                    ? encoding.timeBins[static_cast<size_t>(binIdx + 1u)]
                    : encoding.packedEventCount;
                if (start < end) {
                    const uint16_t targetKey =
                        packMode2CoordWord(static_cast<uint16_t>(lidx), localTimeInBin);
                    int left = static_cast<int>(start);
                    int right = static_cast<int>(end) - 1;
                    while (left <= right) {
                        const int mid = (left + right) >> 1;
                        const uint16_t key = encoding.packedCoords[static_cast<size_t>(mid)];
                        if (key == targetKey) {
                            residual += static_cast<float>(unpackMode2ResidualValue(encoding, static_cast<size_t>(mid))) * encoding.eventScale;
                            break;
                        }
                        if (key < targetKey) left = mid + 1;
                        else right = mid - 1;
                    }
                }
            }
        }
    }

    return {coarse, residual, coarse + residual};
}

void accumulateSampleError(const RawVolume4D& volume,
                           const FieldProfile& profile,
                           const SpatialFirstOptions& options,
                           const LeafEncoding& encoding,
                           int x, int y, int z, int t,
                           std::vector<double>& errors,
                           double& errSum2,
                           uint64_t& samples,
                           ProbeSummary& summary)
{
    const int lx = x % 8;
    const int ly = y % 8;
    const int lz = z % 8;

    if (encoding.mode == BlockMode::CONSTANT) {
        const float pred = encoding.constantValue;
        const float truth = volume.at(x, y, z, t);
        const double err = std::abs(static_cast<double>(truth) - static_cast<double>(pred));
        errors.push_back(err);
        errSum2 += err * err;
        ++samples;

        if (options.compareTemporalBaseline) {
            const auto series = gatherSeriesAtSample(volume, x, y, z);
            const auto keys = detectKeyFrames(series, profile);
            const double baselinePred = decodeKeyframeSeriesAt(series, keys, t);
            const double berr = std::abs(static_cast<double>(truth) - baselinePred);
            summary.baselineRmse += berr * berr;
            summary.baselineSampledKeyframes += static_cast<uint64_t>(keys.size());
        }
        return;
    }

    const auto decoded = decodeLeafSampleAt(options, encoding, lx, ly, lz, t);
    const float pred = decoded.pred;
    const float truth = volume.at(x, y, z, t);
    const double err = std::abs(static_cast<double>(truth) - static_cast<double>(pred));
    errors.push_back(err);
    errSum2 += err * err;
    ++samples;

    if (options.compareTemporalBaseline) {
        const auto series = gatherSeriesAtSample(volume, x, y, z);
        const auto keys = detectKeyFrames(series, profile);
        const double baselinePred = decodeKeyframeSeriesAt(series, keys, t);
        const double berr = std::abs(static_cast<double>(truth) - baselinePred);
        summary.baselineRmse += berr * berr;
        summary.baselineSampledKeyframes += static_cast<uint64_t>(keys.size());
    }
}

std::pair<float, std::vector<int16_t>> quantizeFineCoefficients(const std::vector<float>& coeffs, int bits)
{
    if (bits <= 0 || coeffs.empty()) {
        return {0.0f, {}};
    }
    float maxAbs = 0.0f;
    for (float v : coeffs) maxAbs = std::max(maxAbs, std::abs(v));
    if (maxAbs <= 1e-12f) {
        return {0.0f, std::vector<int16_t>(coeffs.size(), 0)};
    }
    const int maxQ = std::max(1, (1 << (bits - 1)) - 1);
    const float scale = maxAbs / static_cast<float>(maxQ);
    std::vector<int16_t> q(coeffs.size(), 0);
    for (size_t i = 0; i < coeffs.size(); ++i) {
        const float scaled = coeffs[i] / scale;
        const int qi = std::clamp(static_cast<int>(std::lround(scaled)), -maxQ, maxQ);
        q[i] = static_cast<int16_t>(qi);
    }
    return {scale, std::move(q)};
}

std::pair<int, int> selectActiveWindow(const FieldProfile& profile,
                                       const std::vector<float>& frameMin,
                                       const std::vector<float>& frameMax)
{
    const int frames = static_cast<int>(frameMin.size());
    if (frames <= 0) return {0, 0};
    if (profile.type != FieldType::DENSITY || profile.den.renderCutoff <= 0.0f) {
        return {0, frames - 1};
    }

    const float cutoff = profile.den.renderCutoff;
    const float band = std::max(1e-6f, profile.den.cutoffBand);
    const float activeMaxThr = std::max(1e-6f, 0.25f * cutoff);
    const float activeRangeThr = std::max(1e-6f, 0.5f * band);
    int first = -1;
    int last = -1;
    for (int t = 0; t < frames; ++t) {
        const float range = frameMax[static_cast<size_t>(t)] - frameMin[static_cast<size_t>(t)];
        const bool active = frameMax[static_cast<size_t>(t)] > activeMaxThr || range > activeRangeThr;
        if (active) {
            if (first < 0) first = t;
            last = t;
        }
    }
    if (first < 0 || last < 0) return {0, 0};
    return {first, last};
}

float computeLeafRangeOverWindow(const std::vector<float>& frameMin,
                                 const std::vector<float>& frameMax,
                                 int startFrame, int endFrame)
{
    float leafMin = std::numeric_limits<float>::max();
    float leafMax = -std::numeric_limits<float>::max();
    for (int t = startFrame; t <= endFrame; ++t) {
        leafMin = std::min(leafMin, frameMin[static_cast<size_t>(t)]);
        leafMax = std::max(leafMax, frameMax[static_cast<size_t>(t)]);
    }
    return leafMax - leafMin;
}

std::vector<float> sliceSeries(const std::vector<float>& series, int stride, int baseIndex, int startFrame, int frameCount)
{
    std::vector<float> out(static_cast<size_t>(frameCount), 0.0f);
    for (int t = 0; t < frameCount; ++t) {
        out[static_cast<size_t>(t)] = series[static_cast<size_t>(baseIndex * stride + (startFrame + t))];
    }
    return out;
}

int clampToEncodedFrame(const LeafEncoding& encoding, int t)
{
    if (encoding.encodedFrameCount <= 0) return 0;
    const int local = t - static_cast<int>(encoding.startFrame);
    return std::clamp(local, 0, encoding.encodedFrameCount - 1);
}

std::vector<float> gatherSeriesAtSample(const RawVolume4D& volume, int x, int y, int z)
{
    std::vector<float> series(static_cast<size_t>(volume.meta.frames));
    for (int t = 0; t < volume.meta.frames; ++t) series[static_cast<size_t>(t)] = volume.at(x, y, z, t);
    return series;
}

std::vector<float> computeCoarseControlWeights(const RawVolume4D& volume,
                                               const FieldProfile& profile,
                                               int bx, int by, int bz, int cx, int cy, int cz, int resolution)
{
    std::vector<float> weights(static_cast<size_t>(volume.meta.frames), 1.0f);
    const float gx = static_cast<float>(bx * 8) + (static_cast<float>(cx) / static_cast<float>(std::max(1, resolution - 1))) * 7.0f;
    const float gy = static_cast<float>(by * 8) + (static_cast<float>(cy) / static_cast<float>(std::max(1, resolution - 1))) * 7.0f;
    const float gz = static_cast<float>(bz * 8) + (static_cast<float>(cz) / static_cast<float>(std::max(1, resolution - 1))) * 7.0f;
    const int x0 = static_cast<int>(std::floor(gx));
    const int y0 = static_cast<int>(std::floor(gy));
    const int z0 = static_cast<int>(std::floor(gz));
    for (int t = 0; t < volume.meta.frames; ++t) {
        float sum = 0.0f;
        for (int dz = 0; dz < 2; ++dz) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const int gx = std::clamp(x0 + dx, 0, volume.meta.width - 1);
                    const int gy = std::clamp(y0 + dy, 0, volume.meta.height - 1);
                    const int gz = std::clamp(z0 + dz, 0, volume.meta.depth - 1);
                    sum += semanticWeight(profile, volume.at(gx, gy, gz, t));
                }
            }
        }
        weights[static_cast<size_t>(t)] = sum / 8.0f;
    }
    return weights;
}

float decodeKeyframeSeriesAt(const std::vector<float>& series, const std::vector<int>& keys, int t)
{
    if (keys.empty()) return 0.0f;
    if (keys.size() == 1) return series[static_cast<size_t>(keys.front())];
    if (t <= keys.front()) return series[static_cast<size_t>(keys.front())];
    if (t >= keys.back()) return series[static_cast<size_t>(keys.back())];
    for (size_t i = 1; i < keys.size(); ++i) {
        if (t <= keys[i]) {
            const int t0 = keys[i - 1];
            const int t1 = keys[i];
            const float v0 = series[static_cast<size_t>(t0)];
            const float v1 = series[static_cast<size_t>(t1)];
            const float alpha = static_cast<float>(t - t0) / static_cast<float>(t1 - t0);
            return v0 + alpha * (v1 - v0);
        }
    }
    return series[static_cast<size_t>(keys.back())];
}

uint8_t quantizeBfp(float value, float scale, int bits)
{
    if (bits <= 2) {
        if (scale <= 1e-12f) return 1;
        const float norm = std::clamp(value / scale, -1.0f, 1.0f);
        if (norm < -0.6666667f) return 0;
        if (norm < 0.0f) return 1;
        if (norm < 0.6666667f) return 2;
        return 3;
    }
    if (scale <= 1e-12f) return 1;
    const float norm = std::clamp(value / scale, -1.0f, 1.0f);
    const int levels = (1 << bits) - 1;
    const float shifted = 0.5f * (norm + 1.0f);
    return static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(shifted * static_cast<float>(levels))), 0, levels));
}

float dequantizeBfp(uint8_t q, float scale, int bits)
{
    if (bits <= 2) {
        static constexpr std::array<float, 4> kLevels = {-1.0f, -0.3333333f, 0.3333333f, 1.0f};
        return kLevels[std::min<size_t>(q, kLevels.size() - 1)] * scale;
    }
    const int levels = (1 << bits) - 1;
    const float norm = (static_cast<float>(q) / static_cast<float>(levels)) * 2.0f - 1.0f;
    return norm * scale;
}

float temporalDctBasisAt(int totalLength, int index, int k)
{
    constexpr double kPi = 3.14159265358979323846;
    const double invN = 1.0 / static_cast<double>(totalLength);
    const double alpha = (k == 0) ? std::sqrt(invN) : std::sqrt(2.0 * invN);
    return static_cast<float>(alpha * std::cos((kPi / static_cast<double>(totalLength)) *
                                               (static_cast<double>(index) + 0.5) *
                                               static_cast<double>(k)));
}

DenseResidualBfpEncoding encodeDenseResidualBfp(const std::vector<float>& residualSeries,
                                                int frames,
                                                int bits,
                                                int valuesPerFrame,
                                                int resolution,
                                                bool sampleAsGrid = false,
                                                int originX = 0,
                                                int originY = 0,
                                                int originZ = 0,
                                                int extentX = 8,
                                                int extentY = 8,
                                                int extentZ = 8,
                                                int timeStartLocal = 0,
                                                bool usePatchMask = false,
                                                uint64_t patchMask = 0ull,
                                                int activePatchCount = 0,
                                                int patchSize = 4)
{
    DenseResidualBfpEncoding encoding;
    encoding.sampleAsGrid = sampleAsGrid;
    encoding.usePatchMask = usePatchMask;
    encoding.resolution = resolution;
    encoding.valuesPerFrame = valuesPerFrame;
    encoding.originX = originX;
    encoding.originY = originY;
    encoding.originZ = originZ;
    encoding.extentX = extentX;
    encoding.extentY = extentY;
    encoding.extentZ = extentZ;
    encoding.timeStartLocal = timeStartLocal;
    encoding.timeCount = frames;
    encoding.patchMask = patchMask;
    encoding.patchSize = std::max(1, patchSize);
    encoding.activePatchCount = activePatchCount;
    encoding.bitsPerValue = std::max(2, bits);
    encoding.frameBytes = (encoding.valuesPerFrame * encoding.bitsPerValue + 7) / 8;
    encoding.frameScales.resize(static_cast<size_t>(frames), 0.0f);
    encoding.packedValues.resize(static_cast<size_t>(frames * encoding.frameBytes), 0u);

    for (int t = 0; t < frames; ++t) {
        float maxAbs = 0.0f;
        for (int idx = 0; idx < encoding.valuesPerFrame; ++idx) {
            maxAbs = std::max(maxAbs, std::abs(residualSeries[static_cast<size_t>(idx * frames + t)]));
        }
        encoding.frameScales[static_cast<size_t>(t)] = maxAbs;
        auto* framePtr = encoding.packedValues.data() + static_cast<size_t>(t * encoding.frameBytes);
        for (int idx = 0; idx < encoding.valuesPerFrame; ++idx) {
            const uint8_t q = quantizeBfp(residualSeries[static_cast<size_t>(idx * frames + t)], maxAbs, encoding.bitsPerValue);
            const int bitOffset = idx * encoding.bitsPerValue;
            const int byteIndex = bitOffset / 8;
            const int shift = bitOffset % 8;
            framePtr[byteIndex] |= static_cast<uint8_t>(q << shift);
            if (shift + encoding.bitsPerValue > 8) {
                framePtr[byteIndex + 1] |= static_cast<uint8_t>(q >> (8 - shift));
            }
        }
    }
    return encoding;
}

uint64_t estimateDenseResidualBfpBytes(int frames, int bits, int valuesPerFrame)
{
    const int safeBits = std::max(2, bits);
    const int frameBytes = (std::max(1, valuesPerFrame) * safeBits + 7) / 8;
    return static_cast<uint64_t>(std::max(0, frames)) * static_cast<uint64_t>(2 + frameBytes);
}

DenseTemporalBasisEncoding encodeDenseTemporalBasis(const std::vector<float>& residualSeries,
                                                    int frames,
                                                    int bits,
                                                    int valuesPerBasis,
                                                    int resolution,
                                                    int temporalKeep)
{
    DenseTemporalBasisEncoding encoding;
    encoding.resolution = resolution;
    encoding.temporalKeep = std::max(1, std::min(temporalKeep, frames));
    encoding.bitsPerValue = std::max(2, bits);
    encoding.valuesPerBasis = valuesPerBasis;
    encoding.basisBytes = (std::max(1, valuesPerBasis) * encoding.bitsPerValue + 7) / 8;
    encoding.basisScales.resize(static_cast<size_t>(encoding.temporalKeep), 0.0f);
    encoding.packedBasisValues.resize(static_cast<size_t>(encoding.temporalKeep * encoding.basisBytes), 0u);

    std::vector<float> coeffBuffer(static_cast<size_t>(valuesPerBasis * encoding.temporalKeep), 0.0f);
    std::vector<float> series(static_cast<size_t>(frames), 0.0f);
    for (int idx = 0; idx < valuesPerBasis; ++idx) {
        for (int t = 0; t < frames; ++t) {
            series[static_cast<size_t>(t)] = residualSeries[static_cast<size_t>(idx * frames + t)];
        }
        const auto coeffs = dctEncodeKeep(series, encoding.temporalKeep);
        for (int k = 0; k < encoding.temporalKeep; ++k) {
            coeffBuffer[static_cast<size_t>(idx * encoding.temporalKeep + k)] = coeffs[static_cast<size_t>(k)];
            encoding.basisScales[static_cast<size_t>(k)] =
                std::max(encoding.basisScales[static_cast<size_t>(k)], std::abs(coeffs[static_cast<size_t>(k)]));
        }
    }

    for (int k = 0; k < encoding.temporalKeep; ++k) {
        const float scale = encoding.basisScales[static_cast<size_t>(k)];
        auto* basisPtr = encoding.packedBasisValues.data() + static_cast<size_t>(k * encoding.basisBytes);
        for (int idx = 0; idx < valuesPerBasis; ++idx) {
            const uint8_t q =
                quantizeBfp(coeffBuffer[static_cast<size_t>(idx * encoding.temporalKeep + k)], scale, encoding.bitsPerValue);
            const int bitOffset = idx * encoding.bitsPerValue;
            const int byteIndex = bitOffset / 8;
            const int shift = bitOffset % 8;
            basisPtr[byteIndex] |= static_cast<uint8_t>(q << shift);
            if (shift + encoding.bitsPerValue > 8) {
                basisPtr[byteIndex + 1] |= static_cast<uint8_t>(q >> (8 - shift));
            }
        }
    }
    return encoding;
}

uint64_t estimateDenseTemporalBasisBytes(int temporalKeep, int bits, int valuesPerBasis)
{
    const int safeKeep = std::max(1, temporalKeep);
    const int safeBits = std::max(2, bits);
    const int basisBytes = (std::max(1, valuesPerBasis) * safeBits + 7) / 8;
    return static_cast<uint64_t>(safeKeep) * static_cast<uint64_t>(2 + basisBytes);
}

DenseTileTemporalBasisEncoding encodeDenseTileTemporalBasis(const std::vector<float>& residualSeries,
                                                            int frames,
                                                            int bits,
                                                            int tileSize,
                                                            uint64_t activeTileMask,
                                                            int temporalKeep)
{
    DenseTileTemporalBasisEncoding encoding;
    encoding.tileSize = std::max(1, tileSize);
    encoding.activeTileMask = activeTileMask;
    encoding.activeTileCount = popcount64(activeTileMask);
    encoding.temporalKeep = std::max(1, std::min(temporalKeep, frames));
    encoding.bitsPerValue = std::max(2, bits);
    const int tileVoxelCount = encoding.tileSize * encoding.tileSize * encoding.tileSize;
    encoding.valuesPerBasis = encoding.activeTileCount * tileVoxelCount;
    encoding.basisBytes = (std::max(1, encoding.valuesPerBasis) * encoding.bitsPerValue + 7) / 8;
    encoding.basisScales.resize(static_cast<size_t>(encoding.temporalKeep), 0.0f);
    encoding.packedBasisValues.resize(static_cast<size_t>(encoding.temporalKeep * encoding.basisBytes), 0u);
    if (encoding.activeTileCount <= 0) return encoding;

    std::vector<float> compactSeries(static_cast<size_t>(encoding.valuesPerBasis * frames), 0.0f);
    const int tilesPerAxis = std::max(1, 8 / encoding.tileSize);
    int compactBase = 0;
    for (int tz = 0; tz < tilesPerAxis; ++tz) {
        for (int ty = 0; ty < tilesPerAxis; ++ty) {
            for (int tx = 0; tx < tilesPerAxis; ++tx) {
                const int tileIdx = (tz * tilesPerAxis + ty) * tilesPerAxis + tx;
                const uint64_t bit = (tileIdx < 64) ? (1ull << tileIdx) : 0ull;
                if (bit == 0ull || (activeTileMask & bit) == 0ull) continue;
                for (int lz = 0; lz < encoding.tileSize; ++lz) {
                    for (int ly = 0; ly < encoding.tileSize; ++ly) {
                        for (int lx = 0; lx < encoding.tileSize; ++lx) {
                            const int gx = tx * encoding.tileSize + lx;
                            const int gy = ty * encoding.tileSize + ly;
                            const int gz = tz * encoding.tileSize + lz;
                            const int localVoxel = (lz * encoding.tileSize + ly) * encoding.tileSize + lx;
                            const int compactIndex = compactBase + localVoxel;
                            const int sourceVoxel = localIndex(gx, gy, gz);
                            for (int t = 0; t < frames; ++t) {
                                compactSeries[static_cast<size_t>(compactIndex * frames + t)] =
                                    residualSeries[static_cast<size_t>(sourceVoxel * frames + t)];
                            }
                        }
                    }
                }
                compactBase += tileVoxelCount;
            }
        }
    }

    std::vector<float> coeffBuffer(static_cast<size_t>(encoding.valuesPerBasis * encoding.temporalKeep), 0.0f);
    std::vector<float> series(static_cast<size_t>(frames), 0.0f);
    for (int idx = 0; idx < encoding.valuesPerBasis; ++idx) {
        for (int t = 0; t < frames; ++t) {
            series[static_cast<size_t>(t)] = compactSeries[static_cast<size_t>(idx * frames + t)];
        }
        const auto coeffs = dctEncodeKeep(series, encoding.temporalKeep);
        for (int k = 0; k < encoding.temporalKeep; ++k) {
            coeffBuffer[static_cast<size_t>(idx * encoding.temporalKeep + k)] = coeffs[static_cast<size_t>(k)];
            encoding.basisScales[static_cast<size_t>(k)] =
                std::max(encoding.basisScales[static_cast<size_t>(k)], std::abs(coeffs[static_cast<size_t>(k)]));
        }
    }

    for (int k = 0; k < encoding.temporalKeep; ++k) {
        const float scale = encoding.basisScales[static_cast<size_t>(k)];
        auto* basisPtr = encoding.packedBasisValues.data() + static_cast<size_t>(k * encoding.basisBytes);
        for (int idx = 0; idx < encoding.valuesPerBasis; ++idx) {
            const uint8_t q =
                quantizeBfp(coeffBuffer[static_cast<size_t>(idx * encoding.temporalKeep + k)], scale, encoding.bitsPerValue);
            const int bitOffset = idx * encoding.bitsPerValue;
            const int byteIndex = bitOffset / 8;
            const int shift = bitOffset % 8;
            basisPtr[byteIndex] |= static_cast<uint8_t>(q << shift);
            if (shift + encoding.bitsPerValue > 8) {
                basisPtr[byteIndex + 1] |= static_cast<uint8_t>(q >> (8 - shift));
            }
        }
    }
    return encoding;
}

uint64_t estimateDenseTileTemporalBasisBytes(int temporalKeep, int bits, int activeTileCount, int tileSize)
{
    const int safeKeep = std::max(1, temporalKeep);
    const int safeBits = std::max(2, bits);
    const int safeTile = std::max(1, tileSize);
    const int valuesPerBasis = std::max(1, activeTileCount * safeTile * safeTile * safeTile);
    const int basisBytes = (valuesPerBasis * safeBits + 7) / 8;
    return 8ull + 4ull + static_cast<uint64_t>(safeKeep) * static_cast<uint64_t>(2 + basisBytes);
}

DensePatchTemporalBasisEncoding encodeDensePatchTemporalBasis(const std::vector<float>& residualSeries,
                                                              int frames,
                                                              int bits,
                                                              uint64_t activePatchMask,
                                                              int temporalKeep)
{
    DensePatchTemporalBasisEncoding encoding;
    encoding.activePatchMask = activePatchMask;
    encoding.activePatchCount = popcount64(activePatchMask);
    encoding.temporalKeep = std::max(1, std::min(temporalKeep, frames));
    encoding.bitsPerValue = std::max(2, bits);
    const int controlsPerPatch = encoding.localResolution * encoding.localResolution * encoding.localResolution;
    encoding.valuesPerBasis = encoding.activePatchCount * controlsPerPatch;
    encoding.basisBytes = (std::max(1, encoding.valuesPerBasis) * encoding.bitsPerValue + 7) / 8;
    encoding.basisScales.resize(static_cast<size_t>(encoding.temporalKeep), 0.0f);
    encoding.packedBasisValues.resize(static_cast<size_t>(encoding.temporalKeep * encoding.basisBytes), 0u);
    if (encoding.activePatchCount <= 0) return encoding;

    const auto& pinv = patchPseudoInverseGrid(encoding.patchSize, encoding.localResolution);
    std::vector<float> compactSeries(static_cast<size_t>(encoding.valuesPerBasis * frames), 0.0f);
    int patchBase = 0;
    for (int p = 0; p < 8; ++p) {
        const uint64_t bit = 1ull << p;
        if ((activePatchMask & bit) == 0ull) continue;
        const int baseX = (p & 1) * encoding.patchSize;
        const int baseY = ((p >> 1) & 1) * encoding.patchSize;
        const int baseZ = ((p >> 2) & 1) * encoding.patchSize;
        for (int t = 0; t < frames; ++t) {
            std::array<float, 64> patchValues{};
            for (int lz = 0; lz < encoding.patchSize; ++lz) {
                for (int ly = 0; ly < encoding.patchSize; ++ly) {
                    for (int lx = 0; lx < encoding.patchSize; ++lx) {
                        const int patchVoxel = (lz * encoding.patchSize + ly) * encoding.patchSize + lx;
                        const int gx = baseX + lx;
                        const int gy = baseY + ly;
                        const int gz = baseZ + lz;
                        const int voxel = localIndex(gx, gy, gz);
                        patchValues[static_cast<size_t>(patchVoxel)] =
                            residualSeries[static_cast<size_t>(voxel * frames + t)];
                    }
                }
            }
            for (int ctrl = 0; ctrl < controlsPerPatch; ++ctrl) {
                double sum = 0.0;
                for (int i = 0; i < 64; ++i) {
                    sum += pinv.pinv[static_cast<size_t>(ctrl * 64 + i)] *
                           static_cast<double>(patchValues[static_cast<size_t>(i)]);
                }
                const int compactIndex = patchBase * controlsPerPatch + ctrl;
                compactSeries[static_cast<size_t>(compactIndex * frames + t)] = static_cast<float>(sum);
            }
        }
        patchBase += 1;
    }

    std::vector<float> coeffBuffer(static_cast<size_t>(encoding.valuesPerBasis * encoding.temporalKeep), 0.0f);
    std::vector<float> series(static_cast<size_t>(frames), 0.0f);
    for (int idx = 0; idx < encoding.valuesPerBasis; ++idx) {
        for (int t = 0; t < frames; ++t) {
            series[static_cast<size_t>(t)] = compactSeries[static_cast<size_t>(idx * frames + t)];
        }
        const auto coeffs = dctEncodeKeep(series, encoding.temporalKeep);
        for (int k = 0; k < encoding.temporalKeep; ++k) {
            coeffBuffer[static_cast<size_t>(idx * encoding.temporalKeep + k)] = coeffs[static_cast<size_t>(k)];
            encoding.basisScales[static_cast<size_t>(k)] =
                std::max(encoding.basisScales[static_cast<size_t>(k)], std::abs(coeffs[static_cast<size_t>(k)]));
        }
    }

    for (int k = 0; k < encoding.temporalKeep; ++k) {
        const float scale = encoding.basisScales[static_cast<size_t>(k)];
        auto* basisPtr = encoding.packedBasisValues.data() + static_cast<size_t>(k * encoding.basisBytes);
        for (int idx = 0; idx < encoding.valuesPerBasis; ++idx) {
            const uint8_t q =
                quantizeBfp(coeffBuffer[static_cast<size_t>(idx * encoding.temporalKeep + k)], scale, encoding.bitsPerValue);
            const int bitOffset = idx * encoding.bitsPerValue;
            const int byteIndex = bitOffset / 8;
            const int shift = bitOffset % 8;
            basisPtr[byteIndex] |= static_cast<uint8_t>(q << shift);
            if (shift + encoding.bitsPerValue > 8) {
                basisPtr[byteIndex + 1] |= static_cast<uint8_t>(q >> (8 - shift));
            }
        }
    }
    return encoding;
}

uint64_t estimateDensePatchTemporalBasisBytes(int temporalKeep, int bits, int activePatchCount)
{
    const int safeKeep = std::max(1, temporalKeep);
    const int safeBits = std::max(2, bits);
    const int valuesPerBasis = std::max(1, activePatchCount * 8);
    const int basisBytes = (valuesPerBasis * safeBits + 7) / 8;
    return 8ull + 2ull + 2ull + 2ull + static_cast<uint64_t>(safeKeep) * static_cast<uint64_t>(2 + basisBytes);
}

float decodeDenseResidualBfpValueAt(const DenseResidualBfpEncoding& encoding, int controlIndex, int t)
{
    if (encoding.frameBytes <= 0 || encoding.packedValues.empty() || encoding.frameScales.empty()) return 0.0f;
    const size_t frameBase = static_cast<size_t>(t * encoding.frameBytes);
    const int bitOffset = controlIndex * encoding.bitsPerValue;
    const int byteIndex = bitOffset / 8;
    const int shift = bitOffset % 8;
    uint16_t packed = encoding.packedValues[frameBase + static_cast<size_t>(byteIndex)];
    if (shift + encoding.bitsPerValue > 8) {
        packed |= static_cast<uint16_t>(encoding.packedValues[frameBase + static_cast<size_t>(byteIndex + 1)]) << 8;
    }
    const uint8_t mask = static_cast<uint8_t>((1u << encoding.bitsPerValue) - 1u);
    const uint8_t q = static_cast<uint8_t>((packed >> shift) & mask);
    return dequantizeBfp(q, encoding.frameScales[static_cast<size_t>(t)], encoding.bitsPerValue);
}

float decodeDenseResidualBfpAt(const DenseResidualBfpEncoding& encoding, int x, int y, int z, int t)
{
    if (encoding.resolution <= 0 || encoding.valuesPerFrame <= 0) return 0.0f;
    if (t < encoding.timeStartLocal || t >= (encoding.timeStartLocal + encoding.timeCount)) return 0.0f;
    const int localFrame = t - encoding.timeStartLocal;
    if (encoding.usePatchMask) {
        const int patchSize = std::max(1, encoding.patchSize);
        const int tilesPerAxis = std::max(1, 8 / patchSize);
        const int tx = x / patchSize;
        const int ty = y / patchSize;
        const int tz = z / patchSize;
        const int patchIdx = (tz * tilesPerAxis + ty) * tilesPerAxis + tx;
        const uint64_t bit = (patchIdx >= 0 && patchIdx < 64) ? (1ull << patchIdx) : 0ull;
        if (bit == 0ull || (encoding.patchMask & bit) == 0ull) return 0.0f;
        const uint64_t beforeMask = encoding.patchMask & (bit - 1ull);
        const int activeBefore = popcount64(beforeMask);
        const int localX = x % patchSize;
        const int localY = y % patchSize;
        const int localZ = z % patchSize;
        const int patchVoxelCount = patchSize * patchSize * patchSize;
        const int localIdx = (localZ * patchSize + localY) * patchSize + localX;
        const int idx = activeBefore * patchVoxelCount + localIdx;
        return decodeDenseResidualBfpValueAt(encoding, idx, localFrame);
    }
    if (!encoding.sampleAsGrid) {
        if (x < encoding.originX || x >= (encoding.originX + encoding.extentX) ||
            y < encoding.originY || y >= (encoding.originY + encoding.extentY) ||
            z < encoding.originZ || z >= (encoding.originZ + encoding.extentZ)) {
            return 0.0f;
        }
        if (encoding.extentX == 8 && encoding.extentY == 8 && encoding.extentZ == 8 &&
            encoding.originX == 0 && encoding.originY == 0 && encoding.originZ == 0 &&
            encoding.valuesPerFrame == kLeafVoxelCount) {
            return decodeDenseResidualBfpValueAt(encoding, localIndex(x, y, z), localFrame);
        }
        const int lx = x - encoding.originX;
        const int ly = y - encoding.originY;
        const int lz = z - encoding.originZ;
        const int idx = (lz * encoding.extentY + ly) * encoding.extentX + lx;
        return decodeDenseResidualBfpValueAt(encoding, idx, localFrame);
    }

    const int resolution = encoding.resolution;
    const float fx = (static_cast<float>(x) / 7.0f) * static_cast<float>(resolution - 1);
    const float fy = (static_cast<float>(y) / 7.0f) * static_cast<float>(resolution - 1);
    const float fz = (static_cast<float>(z) / 7.0f) * static_cast<float>(resolution - 1);
    const int x0 = std::min(resolution - 2, std::max(0, static_cast<int>(std::floor(fx))));
    const int y0 = std::min(resolution - 2, std::max(0, static_cast<int>(std::floor(fy))));
    const int z0 = std::min(resolution - 2, std::max(0, static_cast<int>(std::floor(fz))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);
    auto at = [&](int cx, int cy, int cz) -> float {
        const int idx = (cz * resolution + cy) * resolution + cx;
        return decodeDenseResidualBfpValueAt(encoding, idx, localFrame);
    };
    const float c00 = at(x0, y0, z0) * (1.0f - tx) + at(x1, y0, z0) * tx;
    const float c01 = at(x0, y0, z1) * (1.0f - tx) + at(x1, y0, z1) * tx;
    const float c10 = at(x0, y1, z0) * (1.0f - tx) + at(x1, y1, z0) * tx;
    const float c11 = at(x0, y1, z1) * (1.0f - tx) + at(x1, y1, z1) * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

float decodeDenseTemporalBasisValueAt(const DenseTemporalBasisEncoding& encoding, int controlIndex, int basisIndex)
{
    if (encoding.temporalKeep <= 0 || encoding.basisBytes <= 0 || encoding.packedBasisValues.empty() || encoding.basisScales.empty()) return 0.0f;
    const size_t basisBase = static_cast<size_t>(basisIndex * encoding.basisBytes);
    const int bitOffset = controlIndex * encoding.bitsPerValue;
    const int byteIndex = bitOffset / 8;
    const int shift = bitOffset % 8;
    uint16_t packed = encoding.packedBasisValues[basisBase + static_cast<size_t>(byteIndex)];
    if (shift + encoding.bitsPerValue > 8) {
        packed |= static_cast<uint16_t>(encoding.packedBasisValues[basisBase + static_cast<size_t>(byteIndex + 1)]) << 8;
    }
    const uint8_t mask = static_cast<uint8_t>((1u << encoding.bitsPerValue) - 1u);
    const uint8_t q = static_cast<uint8_t>((packed >> shift) & mask);
    return dequantizeBfp(q, encoding.basisScales[static_cast<size_t>(basisIndex)], encoding.bitsPerValue);
}

float decodeDenseTemporalBasisAt(const DenseTemporalBasisEncoding& encoding, int x, int y, int z, int t, int totalFrames)
{
    if (encoding.resolution <= 0 || encoding.valuesPerBasis <= 0 || encoding.temporalKeep <= 0) return 0.0f;
    const int resolution = encoding.resolution;
    const float fx = (static_cast<float>(x) / 7.0f) * static_cast<float>(resolution - 1);
    const float fy = (static_cast<float>(y) / 7.0f) * static_cast<float>(resolution - 1);
    const float fz = (static_cast<float>(z) / 7.0f) * static_cast<float>(resolution - 1);
    const int x0 = std::min(resolution - 2, std::max(0, static_cast<int>(std::floor(fx))));
    const int y0 = std::min(resolution - 2, std::max(0, static_cast<int>(std::floor(fy))));
    const int z0 = std::min(resolution - 2, std::max(0, static_cast<int>(std::floor(fz))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);
    auto at = [&](int basisIndex, int cx, int cy, int cz) -> float {
        const int idx = (cz * resolution + cy) * resolution + cx;
        return decodeDenseTemporalBasisValueAt(encoding, idx, basisIndex);
    };

    float residual = 0.0f;
    for (int k = 0; k < encoding.temporalKeep; ++k) {
        const float c00 = at(k, x0, y0, z0) * (1.0f - tx) + at(k, x1, y0, z0) * tx;
        const float c01 = at(k, x0, y0, z1) * (1.0f - tx) + at(k, x1, y0, z1) * tx;
        const float c10 = at(k, x0, y1, z0) * (1.0f - tx) + at(k, x1, y1, z0) * tx;
        const float c11 = at(k, x0, y1, z1) * (1.0f - tx) + at(k, x1, y1, z1) * tx;
        const float c0 = c00 * (1.0f - ty) + c10 * ty;
        const float c1 = c01 * (1.0f - ty) + c11 * ty;
        residual += (c0 * (1.0f - tz) + c1 * tz) * temporalDctBasisAt(totalFrames, t, k);
    }
    return residual;
}

float decodeDenseTileTemporalBasisValueAt(const DenseTileTemporalBasisEncoding& encoding, int controlIndex, int basisIndex)
{
    if (encoding.temporalKeep <= 0 || encoding.basisBytes <= 0 || encoding.packedBasisValues.empty() || encoding.basisScales.empty()) return 0.0f;
    const size_t basisBase = static_cast<size_t>(basisIndex * encoding.basisBytes);
    const int bitOffset = controlIndex * encoding.bitsPerValue;
    const int byteIndex = bitOffset / 8;
    const int shift = bitOffset % 8;
    uint16_t packed = encoding.packedBasisValues[basisBase + static_cast<size_t>(byteIndex)];
    if (shift + encoding.bitsPerValue > 8) {
        packed |= static_cast<uint16_t>(encoding.packedBasisValues[basisBase + static_cast<size_t>(byteIndex + 1)]) << 8;
    }
    const uint8_t mask = static_cast<uint8_t>((1u << encoding.bitsPerValue) - 1u);
    const uint8_t q = static_cast<uint8_t>((packed >> shift) & mask);
    return dequantizeBfp(q, encoding.basisScales[static_cast<size_t>(basisIndex)], encoding.bitsPerValue);
}

float decodeDenseTileTemporalBasisAt(const DenseTileTemporalBasisEncoding& encoding, int x, int y, int z, int t, int totalFrames)
{
    if (encoding.temporalKeep <= 0 || encoding.activeTileCount <= 0 || encoding.tileSize <= 0) return 0.0f;
    const int tilesPerAxis = std::max(1, 8 / encoding.tileSize);
    const int tx = x / encoding.tileSize;
    const int ty = y / encoding.tileSize;
    const int tz = z / encoding.tileSize;
    const int tileIdx = (tz * tilesPerAxis + ty) * tilesPerAxis + tx;
    const uint64_t bit = (tileIdx >= 0 && tileIdx < 64) ? (1ull << tileIdx) : 0ull;
    if (bit == 0ull || (encoding.activeTileMask & bit) == 0ull) return 0.0f;
    const int tileVoxelCount = encoding.tileSize * encoding.tileSize * encoding.tileSize;
    const int activeBefore = popcount64(encoding.activeTileMask & (bit - 1ull));
    const int lx = x % encoding.tileSize;
    const int ly = y % encoding.tileSize;
    const int lz = z % encoding.tileSize;
    const int localVoxel = (lz * encoding.tileSize + ly) * encoding.tileSize + lx;
    const int controlIndex = activeBefore * tileVoxelCount + localVoxel;
    float residual = 0.0f;
    for (int k = 0; k < encoding.temporalKeep; ++k) {
        residual += decodeDenseTileTemporalBasisValueAt(encoding, controlIndex, k) *
                    temporalDctBasisAt(totalFrames, t, k);
    }
    return residual;
}

float decodeDensePatchTemporalBasisValueAt(const DensePatchTemporalBasisEncoding& encoding, int controlIndex, int basisIndex)
{
    if (encoding.temporalKeep <= 0 || encoding.basisBytes <= 0 || encoding.packedBasisValues.empty() || encoding.basisScales.empty()) return 0.0f;
    const size_t basisBase = static_cast<size_t>(basisIndex * encoding.basisBytes);
    const int bitOffset = controlIndex * encoding.bitsPerValue;
    const int byteIndex = bitOffset / 8;
    const int shift = bitOffset % 8;
    uint16_t packed = encoding.packedBasisValues[basisBase + static_cast<size_t>(byteIndex)];
    if (shift + encoding.bitsPerValue > 8) {
        packed |= static_cast<uint16_t>(encoding.packedBasisValues[basisBase + static_cast<size_t>(byteIndex + 1)]) << 8;
    }
    const uint8_t mask = static_cast<uint8_t>((1u << encoding.bitsPerValue) - 1u);
    const uint8_t q = static_cast<uint8_t>((packed >> shift) & mask);
    return dequantizeBfp(q, encoding.basisScales[static_cast<size_t>(basisIndex)], encoding.bitsPerValue);
}

float decodeDensePatchTemporalBasisAt(const DensePatchTemporalBasisEncoding& encoding, int x, int y, int z, int t, int totalFrames)
{
    if (encoding.temporalKeep <= 0 || encoding.activePatchCount <= 0 || encoding.patchSize <= 0 || encoding.localResolution <= 0) return 0.0f;
    const int px = x / encoding.patchSize;
    const int py = y / encoding.patchSize;
    const int pz = z / encoding.patchSize;
    const int patchIdx = (pz * 2 + py) * 2 + px;
    const uint64_t bit = (patchIdx >= 0 && patchIdx < 64) ? (1ull << patchIdx) : 0ull;
    if (bit == 0ull || (encoding.activePatchMask & bit) == 0ull) return 0.0f;
    const int activeBefore = popcount64(encoding.activePatchMask & (bit - 1ull));
    const int localX = x % encoding.patchSize;
    const int localY = y % encoding.patchSize;
    const int localZ = z % encoding.patchSize;
    const float fx = (static_cast<float>(localX) / static_cast<float>(encoding.patchSize - 1)) *
                     static_cast<float>(encoding.localResolution - 1);
    const float fy = (static_cast<float>(localY) / static_cast<float>(encoding.patchSize - 1)) *
                     static_cast<float>(encoding.localResolution - 1);
    const float fz = (static_cast<float>(localZ) / static_cast<float>(encoding.patchSize - 1)) *
                     static_cast<float>(encoding.localResolution - 1);
    const int x0 = std::min(encoding.localResolution - 2, std::max(0, static_cast<int>(std::floor(fx))));
    const int y0 = std::min(encoding.localResolution - 2, std::max(0, static_cast<int>(std::floor(fy))));
    const int z0 = std::min(encoding.localResolution - 2, std::max(0, static_cast<int>(std::floor(fz))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);
    const int controlsPerPatch = encoding.localResolution * encoding.localResolution * encoding.localResolution;
    auto at = [&](int basisIndex, int cx, int cy, int cz) -> float {
        const int localCtrl = (cz * encoding.localResolution + cy) * encoding.localResolution + cx;
        const int controlIndex = activeBefore * controlsPerPatch + localCtrl;
        return decodeDensePatchTemporalBasisValueAt(encoding, controlIndex, basisIndex);
    };

    float residual = 0.0f;
    for (int k = 0; k < encoding.temporalKeep; ++k) {
        const float c00 = at(k, x0, y0, z0) * (1.0f - tx) + at(k, x1, y0, z0) * tx;
        const float c01 = at(k, x0, y0, z1) * (1.0f - tx) + at(k, x1, y0, z1) * tx;
        const float c10 = at(k, x0, y1, z0) * (1.0f - tx) + at(k, x1, y1, z0) * tx;
        const float c11 = at(k, x0, y1, z1) * (1.0f - tx) + at(k, x1, y1, z1) * tx;
        const float c0 = c00 * (1.0f - ty) + c10 * ty;
        const float c1 = c01 * (1.0f - ty) + c11 * ty;
        residual += (c0 * (1.0f - tz) + c1 * tz) * temporalDctBasisAt(totalFrames, t, k);
    }
    return residual;
}

float decodeSparseEventResidualAt(const LeafEncoding& encoding, int lidx, int localT)
{
    if (encoding.packedEventCount == 0 ||
        encoding.packedCoords.empty() ||
        encoding.packedResiduals.empty() ||
        encoding.eventScale <= 0.0f) {
        return 0.0f;
    }
    const uint32_t bitWord = encoding.spatialBitmask[static_cast<size_t>(lidx >> 5)];
    const uint32_t bitMask = 1u << (lidx & 31);
    if ((bitWord & bitMask) == 0u) return 0.0f;
    const uint16_t fpb = framesPerEventBin(encoding.encodedFrameCount);
    const uint16_t binIdx = static_cast<uint16_t>(std::min<int>(kEventBinCount - 1, localT / std::max<int>(1, fpb)));
    const uint8_t localTimeInBin = static_cast<uint8_t>(localT - static_cast<int>(binIdx) * static_cast<int>(fpb));
    const uint16_t start = encoding.timeBins[static_cast<size_t>(binIdx)];
    const uint16_t end = (binIdx + 1u < kEventBinCount)
        ? encoding.timeBins[static_cast<size_t>(binIdx + 1u)]
        : encoding.packedEventCount;
    if (start >= end) return 0.0f;
    const uint16_t targetKey =
        packMode2CoordWord(static_cast<uint16_t>(lidx), localTimeInBin);
    int left = static_cast<int>(start);
    int right = static_cast<int>(end) - 1;
    while (left <= right) {
        const int mid = (left + right) >> 1;
        const uint16_t key = encoding.packedCoords[static_cast<size_t>(mid)];
        if (key == targetKey) {
            return static_cast<float>(unpackMode2ResidualValue(encoding, static_cast<size_t>(mid))) * encoding.eventScale;
        }
        if (key < targetKey) left = mid + 1;
        else right = mid - 1;
    }
    return 0.0f;
}

struct DistortionProxyStats {
    double rmseNorm = 0.0;
    double p99Norm = 0.0;
    double peakNorm = 0.0;
};

double percentileFromSortedD(const std::vector<double>& values, double q)
{
    if (values.empty()) return 0.0;
    const double clampedQ = std::clamp(q, 0.0, 1.0);
    const double pos = clampedQ * static_cast<double>(values.size() - 1);
    const size_t i0 = static_cast<size_t>(std::floor(pos));
    const size_t i1 = std::min(values.size() - 1, i0 + 1);
    const double t = pos - static_cast<double>(i0);
    return values[i0] + (values[i1] - values[i0]) * t;
}

DistortionProxyStats buildDistortionProxyStats(std::vector<double> absNormErrors,
                                               double sumSqNormErr)
{
    DistortionProxyStats stats{};
    if (absNormErrors.empty()) return stats;
    std::sort(absNormErrors.begin(), absNormErrors.end());
    stats.rmseNorm = std::sqrt(sumSqNormErr / static_cast<double>(absNormErrors.size()));
    stats.p99Norm = percentileFromSortedD(absNormErrors, 0.99);
    stats.peakNorm = absNormErrors.back();
    return stats;
}

DistortionProxyStats maxDistortionProxyStats(const DistortionProxyStats& a,
                                             const DistortionProxyStats& b)
{
    DistortionProxyStats combined{};
    combined.rmseNorm = std::max(a.rmseNorm, b.rmseNorm);
    combined.p99Norm = std::max(a.p99Norm, b.p99Norm);
    combined.peakNorm = std::max(a.peakNorm, b.peakNorm);
    return combined;
}

template <typename DecodeFn>
DistortionProxyStats estimateResidualProxySamples(const std::vector<float>& residualSeries,
                                                  int activeFrames,
                                                  int spatialStride,
                                                  int timeStride,
                                                  float normDenom,
                                                  DecodeFn&& decodeAt)
{
    const int safeSpatialStride = std::max(1, spatialStride);
    const int safeTimeStride = std::max(1, timeStride);
    const double safeDenom = std::max<double>(static_cast<double>(normDenom), 1e-12);
    std::vector<double> absNormErrors;
    double sumSqNormErr = 0.0;
    for (int lt = 0; lt < activeFrames; lt += safeTimeStride) {
        for (int z = 0; z < 8; z += safeSpatialStride) {
            for (int y = 0; y < 8; y += safeSpatialStride) {
                for (int x = 0; x < 8; x += safeSpatialStride) {
                    const int idx = localIndex(x, y, z);
                    const double truth = static_cast<double>(residualSeries[static_cast<size_t>(idx * activeFrames + lt)]);
                    const double pred = static_cast<double>(decodeAt(idx, x, y, z, lt));
                    const double absNormErr = std::abs(truth - pred) / safeDenom;
                    absNormErrors.push_back(absNormErr);
                    sumSqNormErr += absNormErr * absNormErr;
                }
            }
        }
    }
    return buildDistortionProxyStats(std::move(absNormErrors), sumSqNormErr);
}

template <typename CandidateRange, typename DecodeFn>
DistortionProxyStats estimateResidualProxyCandidates(const std::vector<float>& residualSeries,
                                                     int activeFrames,
                                                     float normDenom,
                                                     const CandidateRange& candidates,
                                                     DecodeFn&& decodeAt)
{
    const double safeDenom = std::max<double>(static_cast<double>(normDenom), 1e-12);
    std::vector<double> absNormErrors;
    absNormErrors.reserve(candidates.size());
    double sumSqNormErr = 0.0;
    for (const auto& candidate : candidates) {
        const int idx = static_cast<int>(candidate.spatialIndex);
        const int lt = static_cast<int>(candidate.timeTag);
        const int x = idx & 7;
        const int y = (idx >> 3) & 7;
        const int z = (idx >> 6) & 7;
        const double truth = static_cast<double>(residualSeries[static_cast<size_t>(idx * activeFrames + lt)]);
        const double pred = static_cast<double>(decodeAt(idx, x, y, z, lt));
        const double absNormErr = std::abs(truth - pred) / safeDenom;
        absNormErrors.push_back(absNormErr);
        sumSqNormErr += absNormErr * absNormErr;
    }
    return buildDistortionProxyStats(std::move(absNormErrors), sumSqNormErr);
}

double computeRdoScore(const DistortionProxyStats& stats,
                       uint64_t bytes,
                       float lambda,
                       float p99Weight,
                       float peakWeight,
                       bool useMaxEnvelope,
                       bool rmseOnly = false)
{
    double distortion = 0.0;
    if (rmseOnly) {
        distortion = stats.rmseNorm;
    } else if (useMaxEnvelope) {
        distortion = std::max({
            stats.rmseNorm,
            static_cast<double>(p99Weight) * stats.p99Norm,
            static_cast<double>(peakWeight) * stats.peakNorm,
        });
    } else {
        distortion =
            stats.rmseNorm +
            static_cast<double>(p99Weight) * stats.p99Norm +
            static_cast<double>(peakWeight) * stats.peakNorm;
    }
    return distortion + static_cast<double>(lambda) * (static_cast<double>(bytes) / (1024.0 * 1024.0));
}

double computeDistortionEnvelope(const DistortionProxyStats& stats,
                                 float p99Weight,
                                 float peakWeight,
                                 bool useMaxEnvelope,
                                 bool rmseOnly = false)
{
    if (rmseOnly) {
        return stats.rmseNorm;
    }
    if (useMaxEnvelope) {
        return std::max({
            stats.rmseNorm,
            static_cast<double>(p99Weight) * stats.p99Norm,
            static_cast<double>(peakWeight) * stats.peakNorm,
        });
    }
    return
        stats.rmseNorm +
        static_cast<double>(p99Weight) * stats.p99Norm +
        static_cast<double>(peakWeight) * stats.peakNorm;
}

struct ResidualDiagnosticCandidate {
    int leafId = 0;
    int bx = 0;
    int by = 0;
    int bz = 0;
    int activeFrames = 0;
    BlockMode mode = BlockMode::COARSE_ONLY;
    double normErr = 0.0;
    double peakErrNorm = 0.0;
    double topEnergyFrac = 0.0;
    double spatialOccupancy = 0.0;
    double timeOccupancy = 0.0;
    double score = 0.0;
};

struct CorrelationAccumulator {
    double sumA = 0.0;
    double sumB = 0.0;
    double sumAA = 0.0;
    double sumBB = 0.0;
    double sumAB = 0.0;
    uint64_t count = 0;

    void add(double a, double b)
    {
        sumA += a;
        sumB += b;
        sumAA += a * a;
        sumBB += b * b;
        sumAB += a * b;
        count += 1;
    }

    double corr() const
    {
        if (count < 2) return 0.0;
        const double n = static_cast<double>(count);
        const double cov = n * sumAB - sumA * sumB;
        const double varA = n * sumAA - sumA * sumA;
        const double varB = n * sumBB - sumB * sumB;
        if (varA <= 1e-18 || varB <= 1e-18) return 0.0;
        return cov / std::sqrt(varA * varB);
    }
};

std::array<double, 8> dctBasis8(int k)
{
    constexpr double kPi = 3.14159265358979323846;
    std::array<double, 8> basis{};
    const double alpha = (k == 0) ? std::sqrt(1.0 / 8.0) : std::sqrt(2.0 / 8.0);
    for (int n = 0; n < 8; ++n) {
        basis[static_cast<size_t>(n)] =
            alpha * std::cos((kPi / 8.0) * (static_cast<double>(n) + 0.5) * static_cast<double>(k));
    }
    return basis;
}

struct DctTopCoeff {
    int u = 0;
    int v = 0;
    int w = 0;
    double value = 0.0;
    double energy = 0.0;
};

std::array<DctTopCoeff, 12> computeTopDct8x8x8(const std::array<float, kLeafVoxelCount>& frame,
                                               double& dcEnergyFrac,
                                               double& lowFreqEnergyFrac,
                                               double& highFreqEnergyFrac)
{
    std::array<std::array<double, 8>, 8> basis{};
    for (int k = 0; k < 8; ++k) {
        basis[static_cast<size_t>(k)] = dctBasis8(k);
    }
    std::array<DctTopCoeff, 12> top{};
    std::fill(top.begin(), top.end(), DctTopCoeff{});
    double totalEnergy = 0.0;
    double dcEnergy = 0.0;
    double lowEnergy = 0.0;
    double highEnergy = 0.0;
    for (int w = 0; w < 8; ++w) {
        for (int v = 0; v < 8; ++v) {
            for (int u = 0; u < 8; ++u) {
                double coeff = 0.0;
                for (int z = 0; z < 8; ++z) {
                    const double bz = basis[static_cast<size_t>(w)][static_cast<size_t>(z)];
                    for (int y = 0; y < 8; ++y) {
                        const double by = basis[static_cast<size_t>(v)][static_cast<size_t>(y)];
                        for (int x = 0; x < 8; ++x) {
                            const double bx = basis[static_cast<size_t>(u)][static_cast<size_t>(x)];
                            coeff += static_cast<double>(frame[static_cast<size_t>(localIndex(x, y, z))]) * bx * by * bz;
                        }
                    }
                }
                const double energy = coeff * coeff;
                totalEnergy += energy;
                if (u == 0 && v == 0 && w == 0) dcEnergy += energy;
                if (u < 2 && v < 2 && w < 2) lowEnergy += energy;
                if (u >= 4 || v >= 4 || w >= 4) highEnergy += energy;
                DctTopCoeff candidate{u, v, w, coeff, energy};
                auto minIt = std::min_element(top.begin(), top.end(),
                                              [](const DctTopCoeff& a, const DctTopCoeff& b) {
                                                  return a.energy < b.energy;
                                              });
                if (candidate.energy > minIt->energy) {
                    *minIt = candidate;
                }
            }
        }
    }
    std::sort(top.begin(), top.end(), [](const DctTopCoeff& a, const DctTopCoeff& b) {
        return a.energy > b.energy;
    });
    const double denom = std::max(1e-18, totalEnergy);
    dcEnergyFrac = dcEnergy / denom;
    lowFreqEnergyFrac = lowEnergy / denom;
    highFreqEnergyFrac = highEnergy / denom;
    return top;
}

void writeFloatRaw(const std::filesystem::path& path, const std::vector<float>& values)
{
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
}

void writeResidualDiagnostics(const RawVolume4D& volume,
                              const SpatialFirstOptions& options,
                              const std::vector<LeafEncoding>& encodings,
                              std::vector<ResidualDiagnosticCandidate> candidates)
{
    if (!options.residualDiagnostics || options.residualDiagnosticDir.empty() || candidates.empty()) return;
    std::sort(candidates.begin(), candidates.end(), [](const ResidualDiagnosticCandidate& a,
                                                       const ResidualDiagnosticCandidate& b) {
        return a.score > b.score;
    });
    const int dumpCount = std::min<int>(options.residualDiagnosticBlocks, static_cast<int>(candidates.size()));
    if (dumpCount <= 0) return;

    const std::filesystem::path outDir(options.residualDiagnosticDir);
    std::filesystem::create_directories(outDir);
    const auto reportPath = outDir / "residual_diagnostics.md";
    std::ofstream report(reportPath);
    report << "# Residual Diagnostics\n\n";
    report << "- input dimensions: `" << volume.meta.width << "x" << volume.meta.height << "x" << volume.meta.depth
           << " x " << volume.meta.frames << "`\n";
    report << "- profile: `" << (options.profile.type == FieldType::GENERIC ? "generic" : "non-generic") << "`\n";
    report << "- dumped blocks: `" << dumpCount << "`\n\n";

    for (int rank = 0; rank < dumpCount; ++rank) {
        const auto& candidate = candidates[static_cast<size_t>(rank)];
        const auto& encoding = encodings[static_cast<size_t>(candidate.leafId)];
        const int activeFrames = encoding.encodedFrameCount;
        if (activeFrames <= 0) continue;

        std::vector<float> residualSeries(static_cast<size_t>(kLeafVoxelCount * activeFrames), 0.0f);
        std::vector<float> fullSeries(static_cast<size_t>(kLeafVoxelCount * activeFrames), 0.0f);
        std::vector<double> frameEnergy(static_cast<size_t>(activeFrames), 0.0);
        CorrelationAccumulator corrX, corrY, corrZ, corrT;
        double meanAbs = 0.0;
        double peakAbs = 0.0;
        std::array<float, kCoarseControlCount> coarseCtrl{};
        const int coarseKeep = resolveCoarseKeep(options, encoding);

        for (int lt = 0; lt < activeFrames; ++lt) {
            const auto coarseRow = buildDctDecodeRow(activeFrames, coarseKeep, lt);
            decodeDctCoeffBlockAt(encoding.coarseCoeffs, kCoarseControlCount, coarseKeep, coarseRow, coarseCtrl.data());
            const auto leaf = loadLeafFrame(volume, candidate.bx, candidate.by, candidate.bz, static_cast<int>(encoding.startFrame) + lt);
            for (int z = 0; z < 8; ++z) {
                for (int y = 0; y < 8; ++y) {
                    for (int x = 0; x < 8; ++x) {
                        const int idx = localIndex(x, y, z);
                        const float truth = leaf[static_cast<size_t>(idx)];
                        const float coarse = sampleTrilinear4(coarseCtrl, x, y, z);
                        const float residual = truth - coarse;
                        residualSeries[static_cast<size_t>(lt * kLeafVoxelCount + idx)] = residual;
                        fullSeries[static_cast<size_t>(idx * activeFrames + lt)] = residual;
                        const double absResidual = std::abs(static_cast<double>(residual));
                        meanAbs += absResidual;
                        peakAbs = std::max(peakAbs, absResidual);
                        frameEnergy[static_cast<size_t>(lt)] += static_cast<double>(residual) * static_cast<double>(residual);
                    }
                }
            }
        }

        for (int lt = 0; lt < activeFrames; ++lt) {
            for (int z = 0; z < 8; ++z) {
                for (int y = 0; y < 8; ++y) {
                    for (int x = 0; x < 8; ++x) {
                        const int idx = localIndex(x, y, z);
                        const double a = static_cast<double>(residualSeries[static_cast<size_t>(lt * kLeafVoxelCount + idx)]);
                        if (x + 1 < 8) corrX.add(a, static_cast<double>(residualSeries[static_cast<size_t>(lt * kLeafVoxelCount + localIndex(x + 1, y, z))]));
                        if (y + 1 < 8) corrY.add(a, static_cast<double>(residualSeries[static_cast<size_t>(lt * kLeafVoxelCount + localIndex(x, y + 1, z))]));
                        if (z + 1 < 8) corrZ.add(a, static_cast<double>(residualSeries[static_cast<size_t>(lt * kLeafVoxelCount + localIndex(x, y, z + 1))]));
                        if (lt + 1 < activeFrames) corrT.add(a, static_cast<double>(fullSeries[static_cast<size_t>(idx * activeFrames + (lt + 1))]));
                    }
                }
            }
        }

        const int repLt = static_cast<int>(std::distance(frameEnergy.begin(),
            std::max_element(frameEnergy.begin(), frameEnergy.end())));
        std::array<float, kLeafVoxelCount> repFrame{};
        std::vector<float> repFrameVec(static_cast<size_t>(kLeafVoxelCount), 0.0f);
        for (int idx = 0; idx < kLeafVoxelCount; ++idx) {
            const float v = residualSeries[static_cast<size_t>(repLt * kLeafVoxelCount + idx)];
            repFrame[static_cast<size_t>(idx)] = v;
            repFrameVec[static_cast<size_t>(idx)] = v;
        }
        double dcFrac = 0.0, lowFrac = 0.0, highFrac = 0.0;
        const auto topCoeffs = computeTopDct8x8x8(repFrame, dcFrac, lowFrac, highFrac);
        meanAbs /= std::max(1.0, static_cast<double>(kLeafVoxelCount * activeFrames));

        const std::string stem =
            "block" + std::to_string(rank) + "_b" +
            std::to_string(candidate.bx) + "_" +
            std::to_string(candidate.by) + "_" +
            std::to_string(candidate.bz);
        const auto seriesPath = outDir / (stem + "_residual_8x8x8x" + std::to_string(activeFrames) + "_f32.raw");
        const auto framePath = outDir / (stem + "_frame" + std::to_string(static_cast<int>(encoding.startFrame) + repLt) + "_8x8x8_f32.raw");
        writeFloatRaw(seriesPath, residualSeries);
        writeFloatRaw(framePath, repFrameVec);

        report << "## Block " << rank + 1 << "\n\n";
        report << "- leaf: `" << candidate.bx << "," << candidate.by << "," << candidate.bz << "`\n";
        report << "- mode: `" << static_cast<int>(candidate.mode) << "`\n";
        report << "- active frames: `" << candidate.activeFrames << "`\n";
        report << "- norm err: `" << candidate.normErr << "`\n";
        report << "- peak err norm: `" << candidate.peakErrNorm << "`\n";
        report << "- top energy frac: `" << candidate.topEnergyFrac << "`\n";
        report << "- spatial occupancy: `" << candidate.spatialOccupancy << "`\n";
        report << "- time occupancy: `" << candidate.timeOccupancy << "`\n";
        report << "- score: `" << candidate.score << "`\n";
        report << "- residual mean abs: `" << meanAbs << "`\n";
        report << "- residual peak abs: `" << peakAbs << "`\n";
        report << "- spatial corr x/y/z: `" << corrX.corr() << " / " << corrY.corr() << " / " << corrZ.corr() << "`\n";
        report << "- temporal lag1 corr: `" << corrT.corr() << "`\n";
        report << "- representative local frame: `" << repLt << "`\n";
        report << "- representative global frame: `" << (static_cast<int>(encoding.startFrame) + repLt) << "`\n";
        report << "- DCT energy fractions dc/low/high: `" << dcFrac << " / " << lowFrac << " / " << highFrac << "`\n";
        report << "- residual series raw: `" << seriesPath.string() << "`\n";
        report << "- representative frame raw: `" << framePath.string() << "`\n";
        report << "- top DCT coeffs:\n";
        for (const auto& coeff : topCoeffs) {
            if (coeff.energy <= 0.0) continue;
            report << "  - `(" << coeff.u << "," << coeff.v << "," << coeff.w << ")` value=`"
                   << coeff.value << "` energy=`" << coeff.energy << "`\n";
        }
        report << "\n";
    }
}

} // namespace

SpatialFirstHybridEncoder::SpatialFirstHybridEncoder(SpatialFirstOptions options) : m_options(std::move(options)) {}

ProbeSummary SpatialFirstHybridEncoder::run(const RawVolume4D& volume)
{
    ProbeSummary summary{};
    const int leafCountX = (volume.meta.width + m_options.leafSize - 1) / m_options.leafSize;
    const int leafCountY = (volume.meta.height + m_options.leafSize - 1) / m_options.leafSize;
    const int leafCountZ = (volume.meta.depth + m_options.leafSize - 1) / m_options.leafSize;
    const int leafCount = leafCountX * leafCountY * leafCountZ;
    summary.leafCount = static_cast<uint64_t>(leafCount);

    std::vector<LeafEncoding> encodings(static_cast<size_t>(leafCount));
    std::vector<ResidualDiagnosticCandidate> residualDiagnosticCandidates;

#ifdef VBT_USE_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int leafId = 0; leafId < leafCount; ++leafId) {
        const int bx = leafId % leafCountX;
        const int by = (leafId / leafCountX) % leafCountY;
        const int bz = leafId / (leafCountX * leafCountY);
        auto& encoding = encodings[static_cast<size_t>(leafId)];

                std::vector<float> coarseSeries(static_cast<size_t>(kCoarseControlCount * volume.meta.frames), 0.0f);
                std::vector<float> rawSeries(static_cast<size_t>(kLeafVoxelCount * volume.meta.frames), 0.0f);
                std::vector<float> residualSeries(static_cast<size_t>(kLeafVoxelCount * volume.meta.frames), 0.0f);
                std::vector<float> weightedEnergy(kLeafVoxelCount, 0.0f);
                std::vector<float> frameMin(static_cast<size_t>(volume.meta.frames), std::numeric_limits<float>::max());
                std::vector<float> frameMax(static_cast<size_t>(volume.meta.frames), -std::numeric_limits<float>::max());
                const bool useFineGrid =
                    m_options.fineResidualGridForRenderProfiles &&
                    m_options.profile.type != FieldType::GENERIC;
                const int fineControlCount = useFineGrid
                    ? (m_options.fineGridResolution * m_options.fineGridResolution * m_options.fineGridResolution)
                    : 0;
                std::vector<float> fineSeries(useFineGrid
                    ? static_cast<size_t>(fineControlCount * volume.meta.frames)
                    : 0u, 0.0f);

                // Spatial-first core: fit coarse controls per frame first, then
                // derive residual representations from raw - coarse.
                for (int t = 0; t < volume.meta.frames; ++t) {
                    const auto leaf = loadLeafFrame(volume, bx, by, bz, t);
                    const auto coarse = computeCoarseControls(leaf);
                    for (int i = 0; i < kCoarseControlCount; ++i) {
                        coarseSeries[static_cast<size_t>(i * volume.meta.frames + t)] = coarse[i];
                    }

                    for (int z = 0; z < 8; ++z) {
                        for (int y = 0; y < 8; ++y) {
                            for (int x = 0; x < 8; ++x) {
                                const int idx = localIndex(x, y, z);
                                rawSeries[static_cast<size_t>(idx * volume.meta.frames + t)] = leaf[idx];
                                frameMin[static_cast<size_t>(t)] = std::min(frameMin[static_cast<size_t>(t)], leaf[idx]);
                                frameMax[static_cast<size_t>(t)] = std::max(frameMax[static_cast<size_t>(t)], leaf[idx]);
                                const float recon = sampleTrilinear4(coarse, x, y, z);
                                const float residual = leaf[idx] - recon;
                                residualSeries[static_cast<size_t>(idx * volume.meta.frames + t)] = residual;
                                weightedEnergy[idx] += semanticWeight(m_options.profile, leaf[idx]) * residual * residual;
                            }
                        }
                    }

                    if (useFineGrid) {
                        std::array<float, kLeafVoxelCount> residualLeaf{};
                        for (int idx = 0; idx < kLeafVoxelCount; ++idx) {
                            residualLeaf[static_cast<size_t>(idx)] = residualSeries[static_cast<size_t>(idx * volume.meta.frames + t)];
                        }
                        const auto fine = computeGridControls(residualLeaf, m_options.fineGridResolution);
                        for (int i = 0; i < fineControlCount; ++i) {
                            fineSeries[static_cast<size_t>(i * volume.meta.frames + t)] = fine[static_cast<size_t>(i)];
                        }
                    }
                }

                const auto activeWindow = selectActiveWindow(m_options.profile, frameMin, frameMax);
                encoding.startFrame = static_cast<uint8_t>(activeWindow.first);
                encoding.endFrame = static_cast<uint8_t>(activeWindow.second);
                encoding.encodedFrameCount = std::max(1, activeWindow.second - activeWindow.first + 1);
                encoding.coarseKeep = static_cast<uint8_t>(std::clamp(m_options.dctKeep, 1, 64));

                const float leafRange = computeLeafRangeOverWindow(frameMin, frameMax, activeWindow.first, activeWindow.second);
                const float globalRange = std::max(1e-6f, volume.meta.dataMax - volume.meta.dataMin);
                const float constRangeThr = std::max(1e-6f, m_options.routeConstRangeRatio * globalRange);
                const bool allowConstantMode =
                    !(m_options.splitScientificRenderModes &&
                      m_options.genericDisableConstantMode &&
                      m_options.profile.type == FieldType::GENERIC);
                if (allowConstantMode && leafRange <= constRangeThr) {
                    encoding.mode = BlockMode::CONSTANT;
                    double mean = 0.0;
                    const int frameCount = encoding.encodedFrameCount;
                    const double denom = static_cast<double>(kLeafVoxelCount * frameCount);
                    for (int idx = 0; idx < kLeafVoxelCount; ++idx) {
                        for (int lt = 0; lt < frameCount; ++lt) {
                            mean += rawSeries[static_cast<size_t>(idx * volume.meta.frames + activeWindow.first + lt)];
                        }
                    }
                    encoding.constantValue = static_cast<float>(mean / std::max(1.0, denom));
                    encoding.config = 0;
                    encoding.packedHeader = packHeader(m_options.profile, m_options, encoding);
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                    summary.mode0Count += 1;
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                    summary.payloadWords += 2; // header + aligned float16 const payload
                    continue;
                }

                double coarseErr2 = 0.0;
                double coarsePeakAbs = 0.0;
                const int activeFrames = encoding.encodedFrameCount;
                for (int idx = 0; idx < kLeafVoxelCount; ++idx) {
                    for (int lt = 0; lt < activeFrames; ++lt) {
                        const int t = activeWindow.first + lt;
                        const double residual = static_cast<double>(residualSeries[static_cast<size_t>(idx * volume.meta.frames + t)]);
                        coarseErr2 += residual * residual;
                        coarsePeakAbs = std::max(coarsePeakAbs, std::abs(residual));
                    }
                }
                const double coarseRmse = std::sqrt(coarseErr2 / static_cast<double>(kLeafVoxelCount * activeFrames));
                const double safeLeafRange = std::max<double>(leafRange, 1e-6);
                const double safeGlobalRange = std::max<double>(globalRange, 1e-6);
                const double routeDenom =
                    (m_options.profile.type == FieldType::GENERIC) ? safeGlobalRange : safeLeafRange;
                double normErr = coarseRmse / routeDenom;
                double peakErrNorm = coarsePeakAbs / routeDenom;

                std::vector<float> genericTemporalResidualSeries;
                std::vector<float> genericTemporalSpatialEnergy;
                std::array<std::vector<float>, kCoarseControlCount> activeCoarseSeries;
                std::array<std::vector<float>, kCoarseControlCount> activeCoarseWeights;
                for (int i = 0; i < kCoarseControlCount; ++i) {
                    activeCoarseSeries[static_cast<size_t>(i)] =
                        sliceSeries(coarseSeries, volume.meta.frames, i, activeWindow.first, activeFrames);
                    const int cx = i % 4;
                    const int cy = (i / 4) % 4;
                    const int cz = i / 16;
                    auto weights = computeCoarseControlWeights(volume, m_options.profile, bx, by, bz, cx, cy, cz, 4);
                    weights.erase(weights.begin(), weights.begin() + activeWindow.first);
                    weights.resize(static_cast<size_t>(activeFrames));
                    activeCoarseWeights[static_cast<size_t>(i)] = std::move(weights);
                }

                auto encodeCoarseCoeffsForKeep = [&](int keep) {
                    std::vector<float> coeffFlat(static_cast<size_t>(kCoarseControlCount * keep), 0.0f);
                    for (int i = 0; i < kCoarseControlCount; ++i) {
                        const auto coeffs = dctEncodeKeepWeighted(
                            activeCoarseSeries[static_cast<size_t>(i)],
                            activeCoarseWeights[static_cast<size_t>(i)],
                            keep);
                        for (int k = 0; k < keep; ++k) {
                            coeffFlat[static_cast<size_t>(i * keep + k)] = coeffs[static_cast<size_t>(k)];
                        }
                    }
                    return coeffFlat;
                };

                auto evaluateTemporalCoarse = [&](const std::vector<float>& coeffFlat,
                                                  int keep,
                                                  bool storeResiduals,
                                                  std::vector<float>* outResiduals,
                                                  std::vector<float>* outSpatialEnergy,
                                                  double& outErr2,
                                                  double& outPeakAbs) {
                    if (storeResiduals && outResiduals != nullptr) {
                        outResiduals->assign(static_cast<size_t>(kLeafVoxelCount * activeFrames), 0.0f);
                    }
                    if (storeResiduals && outSpatialEnergy != nullptr) {
                        outSpatialEnergy->assign(static_cast<size_t>(kLeafVoxelCount), 0.0f);
                    }

                    outErr2 = 0.0;
                    outPeakAbs = 0.0;
                    std::array<float, kCoarseControlCount> ctrl{};
                    for (int lt = 0; lt < activeFrames; ++lt) {
                        const auto coarseRow = buildDctDecodeRow(activeFrames, keep, lt);
                        decodeDctCoeffBlockAt(coeffFlat, kCoarseControlCount, keep, coarseRow, ctrl.data());
                        for (int z = 0; z < 8; ++z) {
                            for (int y = 0; y < 8; ++y) {
                                for (int x = 0; x < 8; ++x) {
                                    const int idx = localIndex(x, y, z);
                                    const float recon = sampleTrilinear4(ctrl, x, y, z);
                                    const float truth =
                                        rawSeries[static_cast<size_t>(idx * volume.meta.frames + activeWindow.first + lt)];
                                    const float residual = truth - recon;
                                    if (storeResiduals && outResiduals != nullptr) {
                                        (*outResiduals)[static_cast<size_t>(idx * activeFrames + lt)] = residual;
                                    }
                                    if (storeResiduals && outSpatialEnergy != nullptr) {
                                        (*outSpatialEnergy)[static_cast<size_t>(idx)] += residual * residual;
                                    }
                                    const double residualD = static_cast<double>(residual);
                                    outErr2 += residualD * residualD;
                                    outPeakAbs = std::max(outPeakAbs, std::abs(residualD));
                                }
                            }
                        }
                    }
                };

                if (m_options.profile.type == FieldType::GENERIC && m_options.genericAdaptiveCoarseKeep) {
                    const auto keepLadder = buildScientificCoarseKeepLadder(m_options.dctKeep);
                    int chosenKeep = keepLadder.back();
                    std::vector<float> chosenCoeffs;
                    for (int keep : keepLadder) {
                        auto coeffFlat = encodeCoarseCoeffsForKeep(keep);
                        double candidateErr2 = 0.0;
                        double candidatePeakAbs = 0.0;
                        evaluateTemporalCoarse(coeffFlat,
                                               keep,
                                               false,
                                               nullptr,
                                               nullptr,
                                               candidateErr2,
                                               candidatePeakAbs);
                        const double candidateRmse =
                            std::sqrt(candidateErr2 / static_cast<double>(kLeafVoxelCount * activeFrames));
                        const double candidateNormErr = candidateRmse / routeDenom;
                        const double candidatePeakErrNorm = candidatePeakAbs / routeDenom;
                        chosenKeep = keep;
                        chosenCoeffs = std::move(coeffFlat);
                        if (candidateNormErr <= m_options.routeCoarseOnlyNormErrGeneric &&
                            candidatePeakErrNorm <= m_options.routeCoarseOnlyPeakErrGeneric) {
                            break;
                        }
                    }
                    encoding.coarseKeep = static_cast<uint8_t>(chosenKeep);
                    encoding.coarseCoeffs = std::move(chosenCoeffs);
                } else {
                    encoding.coarseCoeffs = encodeCoarseCoeffsForKeep(static_cast<int>(encoding.coarseKeep));
                }

                if (m_options.profile.type == FieldType::GENERIC) {
                    // Scientific routing is based on the temporally compressed
                    // coarse reconstruction, not the raw per-frame spatial fit.
                    evaluateTemporalCoarse(encoding.coarseCoeffs,
                                           static_cast<int>(encoding.coarseKeep),
                                           true,
                                           &genericTemporalResidualSeries,
                                           &genericTemporalSpatialEnergy,
                                           coarseErr2,
                                           coarsePeakAbs);
                }

                const double temporalCoarseRmse = std::sqrt(coarseErr2 / static_cast<double>(kLeafVoxelCount * activeFrames));
                if (m_options.profile.type == FieldType::GENERIC) {
                    normErr = temporalCoarseRmse / routeDenom;
                    peakErrNorm = coarsePeakAbs / routeDenom;
                }
                bool preferCoarseOnly =
                    (m_options.profile.type == FieldType::GENERIC)
                        ? (normErr <= m_options.routeCoarseOnlyNormErrGeneric &&
                           peakErrNorm <= m_options.routeCoarseOnlyPeakErrGeneric)
                        : (normErr <= m_options.routeCoarseOnlyNormErrRender);
                bool energyAmnestied = false;

                std::vector<int> order(kLeafVoxelCount);
                std::iota(order.begin(), order.end(), 0);
                const bool useFullResidual =
                    !useFineGrid &&
                    m_options.fullResidualForRenderProfiles &&
                    m_options.profile.type != FieldType::GENERIC;
                const int keptEnergyVoxels = useFullResidual
                    ? kLeafVoxelCount
                    : std::min(m_options.eventTopK, kLeafVoxelCount);
                if (!useFullResidual) {
                    std::partial_sort(order.begin(), order.begin() + keptEnergyVoxels, order.end(),
                                      [&](int a, int b) {
                                          return weightedEnergy[static_cast<size_t>(a)] > weightedEnergy[static_cast<size_t>(b)];
                                      });
                }
                const double totalWeightedEnergy = std::accumulate(weightedEnergy.begin(), weightedEnergy.end(), 0.0);
                double topWeightedEnergy = 0.0;
                for (int n = 0; n < keptEnergyVoxels; ++n) {
                    topWeightedEnergy += weightedEnergy[static_cast<size_t>(order[static_cast<size_t>(n)])];
                }
                const double topEnergyFrac = totalWeightedEnergy > 1e-18 ? (topWeightedEnergy / totalWeightedEnergy) : 0.0;
                double hotEnergyFrac = 0.0;
                if (m_options.profile.type == FieldType::GENERIC && m_options.genericEnergyAmnestyTopN > 0) {
                    const auto& energyForAmnesty = genericTemporalSpatialEnergy.empty()
                        ? weightedEnergy
                        : genericTemporalSpatialEnergy;
                    const double totalEnergyForAmnesty = std::accumulate(energyForAmnesty.begin(), energyForAmnesty.end(), 0.0);
                    if (totalEnergyForAmnesty > 1e-18) {
                        std::vector<float> sortedEnergy = energyForAmnesty;
                        std::sort(sortedEnergy.begin(), sortedEnergy.end(), std::greater<float>());
                        const int hotCount = std::min<int>(m_options.genericEnergyAmnestyTopN, static_cast<int>(sortedEnergy.size()));
                        double hotEnergy = 0.0;
                        for (int n = 0; n < hotCount; ++n) {
                            hotEnergy += static_cast<double>(sortedEnergy[static_cast<size_t>(n)]);
                        }
                        hotEnergyFrac = hotEnergy / totalEnergyForAmnesty;
                    }
                }
                // Energy amnesty keeps near-smooth generic blocks in Mode 1 even
                // when a few sharp samples would otherwise force sparse residuals.
                double thresholdSpatialOccupancy = 0.0;
                double thresholdTimeOccupancy = 0.0;
                if (!preferCoarseOnly &&
                    m_options.profile.type == FieldType::GENERIC &&
                    m_options.genericEnergyAmnestyNorm > 0.0f &&
                    m_options.genericEnergyAmnestyHotFrac > 0.0f) {
                    const double relaxedPeakThr =
                        static_cast<double>(m_options.routeCoarseOnlyPeakErrGeneric) *
                        static_cast<double>(std::max(1.0f, m_options.genericEnergyAmnestyPeakScale));
                    const bool lowEnergy = normErr <= static_cast<double>(m_options.genericEnergyAmnestyNorm);
                    const bool concentratedHotspots = hotEnergyFrac >= static_cast<double>(m_options.genericEnergyAmnestyHotFrac);
                    const bool peakNotTooExtreme = peakErrNorm <= relaxedPeakThr;
                    if (lowEnergy && concentratedHotspots && peakNotTooExtreme) {
                        preferCoarseOnly = true;
                        energyAmnestied = true;
                    }
                }
                if (preferCoarseOnly) {
                    encoding.mode = BlockMode::COARSE_ONLY;
                    encoding.config = static_cast<uint16_t>(((m_options.coarseResolution & 0xFF) << 8) | (encoding.coarseKeep & 0xFF));
                    if (energyAmnestied) {
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                        summary.mode1EnergyAmnestyCount += 1;
                    }
                } else if (useFineGrid) {
                    encoding.mode = BlockMode::DENSE_FINE;
                    encoding.useFineResidualGrid = true;
                    encoding.fineResidualGrid.resolution = m_options.fineGridResolution;
                    encoding.fineResidualGrid.dctKeep = m_options.fineGridDctKeep;
                    encoding.fineResidualGrid.quantBits = std::max(0, m_options.fineQuantBits);
                    encoding.fineResidualGrid.coeffs.resize(static_cast<size_t>(fineControlCount * m_options.fineGridDctKeep), 0.0f);
                    for (int i = 0; i < fineControlCount; ++i) {
                        auto series = sliceSeries(fineSeries, volume.meta.frames, i, activeWindow.first, activeFrames);
                        const int r = m_options.fineGridResolution;
                        const int cx = i % r;
                        const int cy = (i / r) % r;
                        const int cz = i / (r * r);
                        auto weights = computeCoarseControlWeights(volume, m_options.profile, bx, by, bz, cx, cy, cz, r);
                        weights.erase(weights.begin(), weights.begin() + activeWindow.first);
                        weights.resize(static_cast<size_t>(activeFrames));
                        const auto coeffs = dctEncodeKeepWeighted(series, weights, m_options.fineGridDctKeep);
                        for (int k = 0; k < m_options.fineGridDctKeep; ++k) {
                            encoding.fineResidualGrid.coeffs[static_cast<size_t>(i * m_options.fineGridDctKeep + k)] = coeffs[static_cast<size_t>(k)];
                        }
                    }
                    if (encoding.fineResidualGrid.quantBits > 0) {
                        auto quant = quantizeFineCoefficients(encoding.fineResidualGrid.coeffs, encoding.fineResidualGrid.quantBits);
                        encoding.fineResidualGrid.blockScale = quant.first;
                        encoding.fineResidualGrid.quantizedCoeffs = std::move(quant.second);
                    }
                    encoding.config = static_cast<uint16_t>(((m_options.fineGridResolution & 0x0F) << 12) |
                                                            ((m_options.fineGridDctKeep & 0x3F) << 6) |
                                                            (encoding.fineResidualGrid.quantBits & 0x3F));
                } else if (useFullResidual) {
                    encoding.mode = BlockMode::DENSE_FINE;
                    encoding.useDenseResidualBfp = true;
                    std::vector<float> croppedResidual(static_cast<size_t>(kLeafVoxelCount * activeFrames), 0.0f);
                    for (int idx = 0; idx < kLeafVoxelCount; ++idx) {
                        for (int lt = 0; lt < activeFrames; ++lt) {
                            croppedResidual[static_cast<size_t>(idx * activeFrames + lt)] =
                                residualSeries[static_cast<size_t>(idx * volume.meta.frames + activeWindow.first + lt)];
                        }
                    }
                    encoding.denseResidualBfp = encodeDenseResidualBfp(croppedResidual,
                                                                       activeFrames,
                                                                       m_options.fullResidualBits,
                                                                       kLeafVoxelCount,
                                                                       8,
                                                                       false,
                                                                       0, 0, 0,
                                                                       8, 8, 8,
                                                                       0);
                    encoding.config = static_cast<uint16_t>(((m_options.fullResidualBits & 0xFF) << 8) | 0x01);
                } else {
                    encoding.mode = BlockMode::SPARSE_IMPULSE;
                    struct EventCandidate {
                        uint16_t spatialIndex = 0;
                        uint8_t timeTag = 0;
                        float value = 0.0f;
                        float absValue = 0.0f;
                    };
                    std::vector<EventCandidate> candidates;
                    candidates.reserve(static_cast<size_t>(kLeafVoxelCount * activeFrames));
                    const float normDenom =
                        (m_options.profile.type == FieldType::GENERIC)
                            ? std::max(globalRange, 1e-6f)
                            : std::max(leafRange, 1e-6f);
                    // Threshold-first semantics: every residual event that
                    // exceeds the normalized error target is eligible, then
                    // eventTopK acts only as a safety cap.
                    for (int idx = 0; idx < kLeafVoxelCount; ++idx) {
                        for (int lt = 0; lt < activeFrames; ++lt) {
                            const int t = activeWindow.first + lt;
                            const float value =
                                (m_options.profile.type == FieldType::GENERIC)
                                    ? genericTemporalResidualSeries[static_cast<size_t>(idx * activeFrames + lt)]
                                    : residualSeries[static_cast<size_t>(idx * volume.meta.frames + t)];
                            const float absValue = std::abs(value);
                            if (absValue <= 1e-12f) continue;
                            const float normValue = absValue / normDenom;
                            if (normValue < m_options.eventThreshold) continue;
                            candidates.push_back(EventCandidate{
                                static_cast<uint16_t>(idx),
                                static_cast<uint8_t>(lt),
                                value,
                                absValue});
                        }
                    }
                    const auto thresholdCandidates = candidates;
                    if (m_options.eventTopK > 0) {
                        auto sortByMagnitude = [](const EventCandidate& a, const EventCandidate& b) {
                            if (a.absValue != b.absValue) return a.absValue > b.absValue;
                            if (a.timeTag != b.timeTag) return a.timeTag < b.timeTag;
                            return a.spatialIndex < b.spatialIndex;
                        };
                        if (m_options.profile.type == FieldType::GENERIC &&
                            m_options.genericStratifiedEvents &&
                            activeFrames > 1 &&
                            static_cast<int>(candidates.size()) > m_options.eventTopK) {
                            std::vector<std::vector<EventCandidate>> frameBuckets(static_cast<size_t>(activeFrames));
                            for (const auto& candidate : candidates) {
                                frameBuckets[static_cast<size_t>(candidate.timeTag)].push_back(candidate);
                            }
                            std::vector<int> nonEmptyFrames;
                            nonEmptyFrames.reserve(static_cast<size_t>(activeFrames));
                            for (int lt = 0; lt < activeFrames; ++lt) {
                                auto& bucket = frameBuckets[static_cast<size_t>(lt)];
                                if (bucket.empty()) continue;
                                std::sort(bucket.begin(), bucket.end(), sortByMagnitude);
                                nonEmptyFrames.push_back(lt);
                            }

                            std::vector<EventCandidate> selected;
                            selected.reserve(static_cast<size_t>(m_options.eventTopK));
                            if (!nonEmptyFrames.empty()) {
                                const int frameCountWithEvents = static_cast<int>(nonEmptyFrames.size());
                                if (m_options.eventTopK < frameCountWithEvents) {
                                    std::sort(nonEmptyFrames.begin(), nonEmptyFrames.end(),
                                              [&](int a, int b) {
                                                  return frameBuckets[static_cast<size_t>(a)].front().absValue >
                                                         frameBuckets[static_cast<size_t>(b)].front().absValue;
                                              });
                                    for (int i = 0; i < m_options.eventTopK; ++i) {
                                        selected.push_back(frameBuckets[static_cast<size_t>(nonEmptyFrames[static_cast<size_t>(i)])].front());
                                    }
                                } else {
                                    const int baseQuota = std::max(1, m_options.eventTopK / frameCountWithEvents);
                                    std::vector<EventCandidate> leftovers;
                                    leftovers.reserve(candidates.size());
                                    for (int lt : nonEmptyFrames) {
                                        auto& bucket = frameBuckets[static_cast<size_t>(lt)];
                                        const int take = std::min<int>(static_cast<int>(bucket.size()), baseQuota);
                                        selected.insert(selected.end(), bucket.begin(), bucket.begin() + take);
                                        leftovers.insert(leftovers.end(), bucket.begin() + take, bucket.end());
                                    }
                                    if (static_cast<int>(selected.size()) > m_options.eventTopK) {
                                        std::sort(selected.begin(), selected.end(), sortByMagnitude);
                                        selected.resize(static_cast<size_t>(m_options.eventTopK));
                                    } else if (static_cast<int>(selected.size()) < m_options.eventTopK && !leftovers.empty()) {
                                        std::sort(leftovers.begin(), leftovers.end(), sortByMagnitude);
                                        const int remaining = m_options.eventTopK - static_cast<int>(selected.size());
                                        const int fill = std::min<int>(remaining, static_cast<int>(leftovers.size()));
                                        selected.insert(selected.end(), leftovers.begin(), leftovers.begin() + fill);
                                    }
                                }
                            }
                            candidates = std::move(selected);
                        } else {
                            std::sort(candidates.begin(), candidates.end(), sortByMagnitude);
                            if (static_cast<int>(candidates.size()) > m_options.eventTopK) {
                                candidates.resize(static_cast<size_t>(m_options.eventTopK));
                            }
                        }
                    }
                    float maxAbs = 0.0f;
                    for (const auto& candidate : candidates) {
                        maxAbs = std::max(maxAbs, candidate.absValue);
                    }
                    encoding.events.clear();
                    encoding.packedCoords.clear();
                    encoding.packedResiduals.clear();
                    encoding.packedEventCount = 0;
                    encoding.packedTierCapacity = 0;
                    encoding.spatialBitmask.fill(0u);
                    encoding.timeBins.fill(0u);
                    if (maxAbs > 1e-12f) {
                        const uint16_t fpb = framesPerEventBin(activeFrames);
                        if (fpb > 16u) {
                            throw std::runtime_error("Mode 2 local-time packing currently supports at most 256 active frames");
                        }
                        const int maxQ = kMode2ResidualQMax;
                        encoding.eventScale = maxAbs / static_cast<float>(maxQ);
                        encoding.events.reserve(candidates.size());
                        struct PackedMode2Event {
                            uint8_t binIdx = 0;
                            uint16_t coord = 0;
                            int16_t q = 0;
                        };
                        std::vector<PackedMode2Event> packedMode2Events;
                        packedMode2Events.reserve(candidates.size());
                        for (const auto& candidate : candidates) {
                            const int q = std::clamp(
                                static_cast<int>(std::lround(candidate.value / encoding.eventScale)),
                                -maxQ, maxQ);
                            if (q == 0) continue;
                            encoding.events.push_back(SpatiotemporalEvent{
                                candidate.spatialIndex,
                                candidate.timeTag,
                                static_cast<int16_t>(q)});
                            encoding.spatialBitmask[static_cast<size_t>(candidate.spatialIndex >> 5u)] |=
                                (1u << (candidate.spatialIndex & 31u));
                            const uint8_t binIdx =
                                static_cast<uint8_t>(std::min<int>(kEventBinCount - 1, candidate.timeTag / std::max<int>(1, fpb)));
                            const uint8_t localTimeInBin =
                                static_cast<uint8_t>(candidate.timeTag - binIdx * fpb);
                            packedMode2Events.push_back(PackedMode2Event{
                                binIdx,
                                packMode2CoordWord(candidate.spatialIndex, localTimeInBin),
                                static_cast<int16_t>(q)});
                        }
                        std::sort(packedMode2Events.begin(), packedMode2Events.end(),
                                  [](const PackedMode2Event& a, const PackedMode2Event& b) {
                                      if (a.binIdx != b.binIdx) return a.binIdx < b.binIdx;
                                      return a.coord < b.coord;
                                  });
                        encoding.packedEventCount = static_cast<uint16_t>(std::min<size_t>(packedMode2Events.size(), 0xFFFFu));
                        encoding.packedTierCapacity = chooseTierCapacity(packedMode2Events.size());
                        std::vector<uint8_t> sortedBinIds;
                        std::vector<int16_t> sortedResiduals;
                        encoding.packedCoords.reserve(static_cast<size_t>(encoding.packedTierCapacity));
                        sortedBinIds.reserve(static_cast<size_t>(encoding.packedEventCount));
                        sortedResiduals.reserve(static_cast<size_t>(encoding.packedEventCount));
                        for (size_t i = 0; i < static_cast<size_t>(encoding.packedEventCount); ++i) {
                            const auto& evt = packedMode2Events[i];
                            encoding.packedCoords.push_back(evt.coord);
                            sortedBinIds.push_back(evt.binIdx);
                            sortedResiduals.push_back(evt.q);
                        }
                        encoding.timeBins = buildEventTimeBins(sortedBinIds, encoding.packedEventCount);
                        if (encoding.packedTierCapacity > 0 &&
                            encoding.packedCoords.size() < static_cast<size_t>(encoding.packedTierCapacity)) {
                            encoding.packedCoords.resize(static_cast<size_t>(encoding.packedTierCapacity), 0xFFFFu);
                        }
                        encoding.packedResiduals =
                            packMode2ResidualBytes(sortedResiduals,
                                                   encoding.packedTierCapacity);
                    }
                    if (encoding.events.empty() ||
                        static_cast<int>(encoding.events.size()) < std::max(1, m_options.eventMinCount)) {
                        encoding.mode = BlockMode::COARSE_ONLY;
                        encoding.config = static_cast<uint16_t>(((m_options.coarseResolution & 0xFF) << 8) | (encoding.coarseKeep & 0xFF));
                        if (!encoding.events.empty()) {
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                            summary.mode1EventFallbackCount += 1;
                        }
                        encoding.events.clear();
                        encoding.packedCoords.clear();
                        encoding.packedResiduals.clear();
                        encoding.packedEventCount = 0;
                        encoding.packedTierCapacity = 0;
                        encoding.spatialBitmask.fill(0u);
                        encoding.timeBins.fill(0u);
                        encoding.eventScale = 0.0f;
                    } else {
                        const uint64_t sparseResidualBytes =
                            estimateMode2ResidualBytes(encoding.packedTierCapacity);
                        const bool useGenericRdo =
                            m_options.profile.type == FieldType::GENERIC &&
                            m_options.genericDenseCrossover &&
                            m_options.genericRdoLambda > 0.0f;
                        const bool allowGenericDense =
                            m_options.profile.type == FieldType::GENERIC &&
                            m_options.genericDenseCrossover &&
                            m_options.genericDenseResidualBits > 0;
                        const bool allowGenericSubgrid =
                            m_options.profile.type == FieldType::GENERIC &&
                            m_options.genericSubgridCrossover &&
                            m_options.genericSubgridBits > 0 &&
                            !useGenericRdo &&
                            !thresholdCandidates.empty();
                        const bool allowGenericMultipatch =
                            m_options.profile.type == FieldType::GENERIC &&
                            m_options.genericMultipatchCrossover &&
                            m_options.genericMultipatchBits > 0 &&
                            !useGenericRdo &&
                            !thresholdCandidates.empty();
                        const bool allowGenericTile =
                            m_options.profile.type == FieldType::GENERIC &&
                            m_options.genericTileCrossover &&
                            m_options.genericTileBits > 0 &&
                            !useGenericRdo &&
                            !thresholdCandidates.empty();
                        const bool allowGenericTileTemporal =
                            m_options.profile.type == FieldType::GENERIC &&
                            (m_options.genericTileTemporalBasisCandidate || m_options.genericTileTemporalBasisForce) &&
                            m_options.genericTileBits > 0 &&
                            !thresholdCandidates.empty();
                        const bool allowGenericPatchTemporal =
                            m_options.profile.type == FieldType::GENERIC &&
                            (m_options.genericPatchTemporalBasisCandidate || m_options.genericPatchTemporalBasisForce) &&
                            m_options.genericMultipatchBits > 0 &&
                            !thresholdCandidates.empty();
                        // Scientific blocks compare sparse events and compact
                        // dense alternatives with an RDO score, not byte cost
                        // alone, so densegrid3 only wins when its distortion is
                        // low enough to justify the smoother representation.
                        enum class GenericDenseChoice {
                            Sparse,
                            DenseGrid,
                            Subgrid,
                            Multipatch,
                            Tile,
                            TileTemporal,
                            PatchTemporal,
                        };
                        GenericDenseChoice denseChoice = GenericDenseChoice::Sparse;
                        uint64_t bestResidualBytes = sparseResidualBytes;
                        if (!thresholdCandidates.empty()) {
                            std::array<uint8_t, kLeafVoxelCount> spatialFlags{};
                            std::vector<uint8_t> timeFlags(static_cast<size_t>(activeFrames), 0u);
                            int uniqueSpatial = 0;
                            int uniqueTime = 0;
                            for (const auto& candidate : thresholdCandidates) {
                                const int spatialIdx = static_cast<int>(candidate.spatialIndex);
                                const int localTime = static_cast<int>(candidate.timeTag);
                                if (spatialFlags[static_cast<size_t>(spatialIdx)] == 0u) {
                                    spatialFlags[static_cast<size_t>(spatialIdx)] = 1u;
                                    uniqueSpatial += 1;
                                }
                                if (localTime >= 0 && localTime < activeFrames &&
                                    timeFlags[static_cast<size_t>(localTime)] == 0u) {
                                    timeFlags[static_cast<size_t>(localTime)] = 1u;
                                    uniqueTime += 1;
                                }
                            }
                            thresholdSpatialOccupancy =
                                static_cast<double>(uniqueSpatial) / static_cast<double>(kLeafVoxelCount);
                            thresholdTimeOccupancy =
                                activeFrames > 0 ? (static_cast<double>(uniqueTime) / static_cast<double>(activeFrames)) : 0.0;
                        }
                        const bool genericChaosAdaptiveRdo =
                            useGenericRdo &&
                            m_options.genericChaosAdaptiveRdo &&
                            thresholdSpatialOccupancy >= static_cast<double>(m_options.genericChaosSpatialOccThreshold) &&
                            thresholdTimeOccupancy >= static_cast<double>(m_options.genericChaosTimeOccThreshold);
                        if (genericChaosAdaptiveRdo) {
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                            summary.genericChaosRdoCount += 1;
                        }
                        DistortionProxyStats sparseDistortionStats{};
                        if (useGenericRdo) {
                            const auto sparseUniformStats =
                                estimateResidualProxySamples(genericTemporalResidualSeries,
                                                             activeFrames,
                                                             m_options.genericRdoSpatialStride,
                                                             m_options.genericRdoTimeStride,
                                                             static_cast<float>(routeDenom),
                                                             [&](int idx, int, int, int, int lt) {
                                                                 return decodeSparseEventResidualAt(encoding, idx, lt);
                                                             });
                            const auto sparseHotStats =
                                estimateResidualProxyCandidates(genericTemporalResidualSeries,
                                                                activeFrames,
                                                                static_cast<float>(routeDenom),
                                                                thresholdCandidates,
                                                                [&](int idx, int, int, int, int lt) {
                                                                    return decodeSparseEventResidualAt(encoding, idx, lt);
                                                                });
                            sparseDistortionStats = maxDistortionProxyStats(sparseUniformStats, sparseHotStats);
                        }
                        double bestRdoScore = useGenericRdo
                            ? computeRdoScore(sparseDistortionStats,
                                              sparseResidualBytes,
                                              m_options.genericRdoLambda,
                                              m_options.genericRdoP99Weight,
                                              m_options.genericRdoPeakWeight,
                                              m_options.genericRdoUseMaxEnvelope,
                                              genericChaosAdaptiveRdo)
                            : std::numeric_limits<double>::infinity();
                        int bestDenseGridRes = 0;
                        bool bestDenseGridUseTemporalBasis = false;
                        DenseResidualBfpEncoding bestDenseGridEncoding;
                        DenseTemporalBasisEncoding bestDenseTemporalEncoding;
                        double denseGrid3DistEnvelope = std::numeric_limits<double>::infinity();
                        bool denseGrid3DistEnvelopeValid = false;
                        int subgridXMin = 0;
                        int subgridYMin = 0;
                        int subgridZMin = 0;
                        int subgridDx = 0;
                        int subgridDy = 0;
                        int subgridDz = 0;
                        int subgridTStart = 0;
                        int subgridDt = 0;
                        uint8_t multipatchMask = 0u;
                        int multipatchPatchCount = 0;
                        int multipatchTStart = 0;
                        int multipatchDt = 0;
                        uint64_t tileMask = 0ull;
                        int tileCount = 0;
                        int tileTStart = 0;
                        int tileDt = 0;
                        int tileSize = 2;
                        uint64_t tileTemporalMask = 0ull;
                        int tileTemporalCount = 0;
                        int tileTemporalSize = 2;
                        DenseTileTemporalBasisEncoding bestTileTemporalEncoding;
                        uint64_t patchTemporalMask = 0ull;
                        int patchTemporalCount = 0;
                        DensePatchTemporalBasisEncoding bestPatchTemporalEncoding;
                        if (allowGenericDense) {
                            std::vector<int> denseGridCandidates;
                            denseGridCandidates.push_back(std::clamp(m_options.genericDenseGridResolution, 2, 8));
                            if (m_options.splitScientificRenderModes &&
                                m_options.profile.type == FieldType::GENERIC &&
                                m_options.genericDenseGrid4Candidate) {
                                denseGridCandidates.push_back(4);
                            }
                            std::sort(denseGridCandidates.begin(), denseGridCandidates.end());
                            denseGridCandidates.erase(std::unique(denseGridCandidates.begin(), denseGridCandidates.end()),
                                                      denseGridCandidates.end());

                            for (const int denseGridRes : denseGridCandidates) {
                                const int denseControlCount = denseGridRes * denseGridRes * denseGridRes;
                                std::vector<float> denseGridSeries(static_cast<size_t>(denseControlCount * activeFrames), 0.0f);
                                for (int lt = 0; lt < activeFrames; ++lt) {
                                    std::array<float, kLeafVoxelCount> residualLeaf{};
                                    for (int idx = 0; idx < kLeafVoxelCount; ++idx) {
                                        residualLeaf[static_cast<size_t>(idx)] =
                                            genericTemporalResidualSeries[static_cast<size_t>(idx * activeFrames + lt)];
                                    }
                                    const auto denseGrid = computeGridControls(residualLeaf, denseGridRes);
                                    for (int i = 0; i < denseControlCount; ++i) {
                                        denseGridSeries[static_cast<size_t>(i * activeFrames + lt)] =
                                            denseGrid[static_cast<size_t>(i)];
                                    }
                                }
                                auto denseGridEncoding = encodeDenseResidualBfp(denseGridSeries,
                                                                                activeFrames,
                                                                                m_options.genericDenseResidualBits,
                                                                                denseControlCount,
                                                                                denseGridRes,
                                                                                true,
                                                                                0, 0, 0,
                                                                                8, 8, 8,
                                                                                0);
                                const uint64_t denseResidualBytes =
                                    estimateDenseResidualBfpBytes(activeFrames,
                                                                  m_options.genericDenseResidualBits,
                                                                  denseControlCount);
                                DistortionProxyStats denseDistortionStats{};
                                if (useGenericRdo) {
                                    const auto denseUniformStats =
                                        estimateResidualProxySamples(genericTemporalResidualSeries,
                                                                     activeFrames,
                                                                     m_options.genericRdoSpatialStride,
                                                                     m_options.genericRdoTimeStride,
                                                                     static_cast<float>(routeDenom),
                                                                     [&](int, int x, int y, int z, int lt) {
                                                                         return decodeDenseResidualBfpAt(denseGridEncoding, x, y, z, lt);
                                                                     });
                                    const auto denseHotStats =
                                        estimateResidualProxyCandidates(genericTemporalResidualSeries,
                                                                        activeFrames,
                                                                        static_cast<float>(routeDenom),
                                                                        thresholdCandidates,
                                                                        [&](int, int x, int y, int z, int lt) {
                                                                            return decodeDenseResidualBfpAt(denseGridEncoding, x, y, z, lt);
                                                                        });
                                    denseDistortionStats = maxDistortionProxyStats(denseUniformStats, denseHotStats);
                                }
                                const double denseRdoScore = useGenericRdo
                                    ? computeRdoScore(denseDistortionStats,
                                                      static_cast<uint64_t>(std::llround(
                                                          static_cast<double>(denseResidualBytes) *
                                                          ((denseGridRes >= 4)
                                                               ? static_cast<double>(m_options.genericDenseGrid4RateScale)
                                                               : 1.0))),
                                                      m_options.genericRdoLambda,
                                                      m_options.genericRdoP99Weight,
                                                      m_options.genericRdoPeakWeight,
                                                      m_options.genericRdoUseMaxEnvelope,
                                                      genericChaosAdaptiveRdo)
                                    : std::numeric_limits<double>::infinity();
                                if (useGenericRdo && denseGridRes == 3) {
                                    denseGrid3DistEnvelope =
                                        computeDistortionEnvelope(denseDistortionStats,
                                                                  m_options.genericRdoP99Weight,
                                                                  m_options.genericRdoPeakWeight,
                                                                  m_options.genericRdoUseMaxEnvelope,
                                                                  genericChaosAdaptiveRdo);
                                    denseGrid3DistEnvelopeValid = true;
                                }
                                const bool denseGrid4Allowed =
                                    !(useGenericRdo &&
                                      denseGridRes >= 4 &&
                                      m_options.genericDenseGrid4DistThreshold > 0.0f &&
                                      denseGrid3DistEnvelopeValid &&
                                      denseGrid3DistEnvelope <
                                          static_cast<double>(m_options.genericDenseGrid4DistThreshold));
                                const bool chooseDense = (!m_options.genericDenseTemporalBasisForce) &&
                                    (useGenericRdo
                                         ? (denseGrid4Allowed && denseRdoScore < bestRdoScore)
                                         : (denseResidualBytes < bestResidualBytes));
                                if (chooseDense) {
                                    denseChoice = GenericDenseChoice::DenseGrid;
                                    bestResidualBytes = denseResidualBytes;
                                    bestDenseGridRes = denseGridRes;
                                    bestDenseGridUseTemporalBasis = false;
                                    bestDenseGridEncoding = std::move(denseGridEncoding);
                                    if (useGenericRdo) bestRdoScore = denseRdoScore;
                                }

                                if (m_options.genericDenseTemporalBasisCandidate &&
                                    (denseGridRes <= 3 || m_options.genericDenseTemporalForGrid4)) {
                                    const int temporalKeep =
                                        std::max(1, std::min(m_options.genericDenseTemporalKeep, activeFrames));
                                    auto denseTemporalEncoding =
                                        encodeDenseTemporalBasis(denseGridSeries,
                                                                 activeFrames,
                                                                 m_options.genericDenseResidualBits,
                                                                 denseControlCount,
                                                                 denseGridRes,
                                                                 temporalKeep);
                                    const uint64_t denseTemporalBytes =
                                        estimateDenseTemporalBasisBytes(temporalKeep,
                                                                        m_options.genericDenseResidualBits,
                                                                        denseControlCount);
                                    DistortionProxyStats denseTemporalDistortionStats{};
                                    if (useGenericRdo) {
                                        const auto denseTemporalUniformStats =
                                            estimateResidualProxySamples(genericTemporalResidualSeries,
                                                                         activeFrames,
                                                                         m_options.genericRdoSpatialStride,
                                                                         m_options.genericRdoTimeStride,
                                                                         static_cast<float>(routeDenom),
                                                                         [&](int, int x, int y, int z, int lt) {
                                                                             return decodeDenseTemporalBasisAt(denseTemporalEncoding, x, y, z, lt, activeFrames);
                                                                         });
                                        const auto denseTemporalHotStats =
                                            estimateResidualProxyCandidates(genericTemporalResidualSeries,
                                                                            activeFrames,
                                                                            static_cast<float>(routeDenom),
                                                                            thresholdCandidates,
                                                                            [&](int, int x, int y, int z, int lt) {
                                                                                return decodeDenseTemporalBasisAt(denseTemporalEncoding, x, y, z, lt, activeFrames);
                                                                            });
                                        denseTemporalDistortionStats =
                                            maxDistortionProxyStats(denseTemporalUniformStats, denseTemporalHotStats);
                                    }
                                    const double denseTemporalRdoScore = useGenericRdo
                                        ? computeRdoScore(denseTemporalDistortionStats,
                                                          static_cast<uint64_t>(std::llround(
                                                              static_cast<double>(denseTemporalBytes) *
                                                              ((denseGridRes >= 4)
                                                                   ? static_cast<double>(m_options.genericDenseGrid4RateScale)
                                                                   : 1.0))),
                                                          m_options.genericRdoLambda,
                                                          m_options.genericRdoP99Weight,
                                                          m_options.genericRdoPeakWeight,
                                                          m_options.genericRdoUseMaxEnvelope,
                                                          genericChaosAdaptiveRdo)
                                        : std::numeric_limits<double>::infinity();
                                    const bool temporalAllowed =
                                        !(useGenericRdo &&
                                          denseGridRes >= 4 &&
                                          m_options.genericDenseGrid4DistThreshold > 0.0f &&
                                          denseGrid3DistEnvelopeValid &&
                                          denseGrid3DistEnvelope <
                                              static_cast<double>(m_options.genericDenseGrid4DistThreshold));
                                    const bool chooseTemporal = useGenericRdo
                                        ? (temporalAllowed && denseTemporalRdoScore < bestRdoScore)
                                        : (denseTemporalBytes < bestResidualBytes);
                                    if (chooseTemporal) {
                                        denseChoice = GenericDenseChoice::DenseGrid;
                                        bestResidualBytes = denseTemporalBytes;
                                        bestDenseGridRes = denseGridRes;
                                        bestDenseGridUseTemporalBasis = true;
                                        bestDenseTemporalEncoding = std::move(denseTemporalEncoding);
                                        if (useGenericRdo) bestRdoScore = denseTemporalRdoScore;
                                    }
                                }
                            }
                        }
                        if (allowGenericSubgrid) {
                            int xMin = 7, yMin = 7, zMin = 7;
                            int xMax = 0, yMax = 0, zMax = 0;
                            int tMin = activeFrames - 1;
                            int tMax = 0;
                            for (const auto& candidate : thresholdCandidates) {
                                const int x = candidate.spatialIndex & 7;
                                const int y = (candidate.spatialIndex >> 3) & 7;
                                const int z = (candidate.spatialIndex >> 6) & 7;
                                xMin = std::min(xMin, x); xMax = std::max(xMax, x);
                                yMin = std::min(yMin, y); yMax = std::max(yMax, y);
                                zMin = std::min(zMin, z); zMax = std::max(zMax, z);
                                tMin = std::min<int>(tMin, candidate.timeTag);
                                tMax = std::max<int>(tMax, candidate.timeTag);
                            }
                            const int dx = xMax - xMin + 1;
                            const int dy = yMax - yMin + 1;
                            const int dz = zMax - zMin + 1;
                            const int dt = tMax - tMin + 1;
                            const int valuesPerFrame = std::max(1, dx * dy * dz);
                            const uint64_t subgridResidualBytes =
                                8ull + estimateDenseResidualBfpBytes(dt, m_options.genericSubgridBits, valuesPerFrame);
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                            summary.genericSubgridCandidateCount += 1;
#ifdef VBT_USE_OPENMP
#pragma omp critical(vbt_subgrid_stats)
#endif
                            {
                                summary.genericSubgridVoxelCounts.push_back(static_cast<uint32_t>(valuesPerFrame));
                                summary.genericSubgridFrameCounts.push_back(static_cast<uint16_t>(dt));
                                summary.genericSubgridBytes.push_back(static_cast<uint32_t>(std::min<uint64_t>(subgridResidualBytes, 0xFFFFFFFFu)));
                                summary.genericSparseBytes.push_back(static_cast<uint32_t>(std::min<uint64_t>(sparseResidualBytes, 0xFFFFFFFFu)));
                            }
                            if (subgridResidualBytes < bestResidualBytes) {
                                denseChoice = GenericDenseChoice::Subgrid;
                                bestResidualBytes = subgridResidualBytes;
                                subgridXMin = xMin;
                                subgridYMin = yMin;
                                subgridZMin = zMin;
                                subgridDx = dx;
                                subgridDy = dy;
                                subgridDz = dz;
                                subgridTStart = tMin;
                                subgridDt = dt;
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                                summary.genericSubgridCheaperCount += 1;
                            }
                        }
                        if (allowGenericMultipatch) {
                            uint8_t patchMask = 0u;
                            int tMin = activeFrames - 1;
                            int tMax = 0;
                            for (const auto& candidate : thresholdCandidates) {
                                const int x = candidate.spatialIndex & 7;
                                const int y = (candidate.spatialIndex >> 3) & 7;
                                const int z = (candidate.spatialIndex >> 6) & 7;
                                const uint8_t patchIdx =
                                    static_cast<uint8_t>(((z >> 2) << 2) | ((y >> 2) << 1) | (x >> 2));
                                patchMask = static_cast<uint8_t>(patchMask | static_cast<uint8_t>(1u << patchIdx));
                                tMin = std::min<int>(tMin, candidate.timeTag);
                                tMax = std::max<int>(tMax, candidate.timeTag);
                            }
                            int activePatchCount = 0;
                            uint8_t tempMask = patchMask;
                            while (tempMask != 0u) {
                                tempMask = static_cast<uint8_t>(tempMask & static_cast<uint8_t>(tempMask - 1u));
                                activePatchCount += 1;
                            }
                            if (activePatchCount > 0) {
                                const int dt = tMax - tMin + 1;
                                const int valuesPerFrame = activePatchCount * 64;
                                const uint64_t multipatchResidualBytes =
                                    8ull + estimateDenseResidualBfpBytes(dt, m_options.genericMultipatchBits, valuesPerFrame);
                                if (multipatchResidualBytes < bestResidualBytes) {
                                    denseChoice = GenericDenseChoice::Multipatch;
                                    bestResidualBytes = multipatchResidualBytes;
                                    multipatchMask = patchMask;
                                    multipatchPatchCount = activePatchCount;
                                    multipatchTStart = tMin;
                                    multipatchDt = dt;
                                }
                            }
                        }
                        if (allowGenericPatchTemporal) {
                            uint64_t candidateMask = 0ull;
                            for (const auto& candidate : thresholdCandidates) {
                                const int x = candidate.spatialIndex & 7;
                                const int y = (candidate.spatialIndex >> 3) & 7;
                                const int z = (candidate.spatialIndex >> 6) & 7;
                                const uint32_t patchIdx =
                                    static_cast<uint32_t>(((z >> 2) << 2) | ((y >> 2) << 1) | (x >> 2));
                                candidateMask |= (1ull << patchIdx);
                            }
                            const int activePatchCount = popcount64(candidateMask);
                            if (activePatchCount > 0) {
                                const int temporalKeep =
                                    std::max(1, std::min(m_options.genericPatchTemporalKeep, activeFrames));
                                auto patchTemporalEncoding =
                                    encodeDensePatchTemporalBasis(genericTemporalResidualSeries,
                                                                  activeFrames,
                                                                  m_options.genericMultipatchBits,
                                                                  candidateMask,
                                                                  temporalKeep);
                                const uint64_t patchTemporalBytes =
                                    estimateDensePatchTemporalBasisBytes(temporalKeep,
                                                                         m_options.genericMultipatchBits,
                                                                         activePatchCount);
                                DistortionProxyStats patchTemporalDistortionStats{};
                                if (useGenericRdo) {
                                    const auto patchTemporalUniformStats =
                                        estimateResidualProxySamples(genericTemporalResidualSeries,
                                                                     activeFrames,
                                                                     m_options.genericRdoSpatialStride,
                                                                     m_options.genericRdoTimeStride,
                                                                     static_cast<float>(routeDenom),
                                                                     [&](int, int x, int y, int z, int lt) {
                                                                         return decodeDensePatchTemporalBasisAt(patchTemporalEncoding, x, y, z, lt, activeFrames);
                                                                     });
                                    const auto patchTemporalHotStats =
                                        estimateResidualProxyCandidates(genericTemporalResidualSeries,
                                                                        activeFrames,
                                                                        static_cast<float>(routeDenom),
                                                                        thresholdCandidates,
                                                                        [&](int, int x, int y, int z, int lt) {
                                                                            return decodeDensePatchTemporalBasisAt(patchTemporalEncoding, x, y, z, lt, activeFrames);
                                                                        });
                                    patchTemporalDistortionStats =
                                        maxDistortionProxyStats(patchTemporalUniformStats, patchTemporalHotStats);
                                }
                                const double patchTemporalRdoScore = useGenericRdo
                                    ? computeRdoScore(patchTemporalDistortionStats,
                                                      patchTemporalBytes,
                                                      m_options.genericRdoLambda,
                                                      m_options.genericRdoP99Weight,
                                                      m_options.genericRdoPeakWeight,
                                                      m_options.genericRdoUseMaxEnvelope,
                                                      genericChaosAdaptiveRdo)
                                    : std::numeric_limits<double>::infinity();
                                const bool choosePatchTemporal = m_options.genericPatchTemporalBasisForce
                                    ? true
                                    : (useGenericRdo
                                           ? (patchTemporalRdoScore < bestRdoScore)
                                           : (patchTemporalBytes < bestResidualBytes));
                                if (choosePatchTemporal) {
                                    denseChoice = GenericDenseChoice::PatchTemporal;
                                    bestResidualBytes = patchTemporalBytes;
                                    patchTemporalMask = candidateMask;
                                    patchTemporalCount = activePatchCount;
                                    bestPatchTemporalEncoding = std::move(patchTemporalEncoding);
                                    if (useGenericRdo) bestRdoScore = patchTemporalRdoScore;
                                }
                            }
                        }
                        if (allowGenericTile) {
                            const int candidateTileSize =
                                (m_options.genericTileSize <= 2) ? 2 : 4;
                            const int tilesPerAxis = 8 / candidateTileSize;
                            uint64_t candidateMask = 0ull;
                            int tMin = activeFrames - 1;
                            int tMax = 0;
                            for (const auto& candidate : thresholdCandidates) {
                                const int x = candidate.spatialIndex & 7;
                                const int y = (candidate.spatialIndex >> 3) & 7;
                                const int z = (candidate.spatialIndex >> 6) & 7;
                                const int tx = x / candidateTileSize;
                                const int ty = y / candidateTileSize;
                                const int tz = z / candidateTileSize;
                                const int tileIdx = (tz * tilesPerAxis + ty) * tilesPerAxis + tx;
                                candidateMask |= (1ull << tileIdx);
                                tMin = std::min<int>(tMin, candidate.timeTag);
                                tMax = std::max<int>(tMax, candidate.timeTag);
                            }
                            const int activeTileCount = popcount64(candidateMask);
                            if (activeTileCount > 0) {
                                const int dt = tMax - tMin + 1;
                                const int patchVoxelCount = candidateTileSize * candidateTileSize * candidateTileSize;
                                const int valuesPerFrame = activeTileCount * patchVoxelCount;
                                const uint64_t tileResidualBytes =
                                    8ull + estimateDenseResidualBfpBytes(dt, m_options.genericTileBits, valuesPerFrame);
                                if (tileResidualBytes < bestResidualBytes) {
                                    denseChoice = GenericDenseChoice::Tile;
                                    bestResidualBytes = tileResidualBytes;
                                    tileMask = candidateMask;
                                    tileCount = activeTileCount;
                                    tileTStart = tMin;
                                    tileDt = dt;
                                    tileSize = candidateTileSize;
                                }
                            }
                        }
                        if (allowGenericTileTemporal) {
                            const int candidateTileSize =
                                (m_options.genericTileSize <= 2) ? 2 : 4;
                            const int tilesPerAxis = 8 / candidateTileSize;
                            std::vector<std::pair<double, int>> tileEnergy;
                            tileEnergy.reserve(static_cast<size_t>(tilesPerAxis * tilesPerAxis * tilesPerAxis));
                            double totalTileEnergy = 0.0;
                            for (int tz = 0; tz < tilesPerAxis; ++tz) {
                                for (int ty = 0; ty < tilesPerAxis; ++ty) {
                                    for (int tx = 0; tx < tilesPerAxis; ++tx) {
                                        const int tileIdx = (tz * tilesPerAxis + ty) * tilesPerAxis + tx;
                                        double energy = 0.0;
                                        for (int lz = 0; lz < candidateTileSize; ++lz) {
                                            for (int ly = 0; ly < candidateTileSize; ++ly) {
                                                for (int lx = 0; lx < candidateTileSize; ++lx) {
                                                    const int gx = tx * candidateTileSize + lx;
                                                    const int gy = ty * candidateTileSize + ly;
                                                    const int gz = tz * candidateTileSize + lz;
                                                    const int voxel = localIndex(gx, gy, gz);
                                                    for (int lt = 0; lt < activeFrames; ++lt) {
                                                        const double v = static_cast<double>(
                                                            genericTemporalResidualSeries[static_cast<size_t>(voxel * activeFrames + lt)]);
                                                        energy += v * v;
                                                    }
                                                }
                                            }
                                        }
                                        totalTileEnergy += energy;
                                        tileEnergy.emplace_back(energy, tileIdx);
                                    }
                                }
                            }
                            std::sort(tileEnergy.begin(), tileEnergy.end(),
                                      [](const auto& a, const auto& b) { return a.first > b.first; });
                            uint64_t candidateMask = 0ull;
                            double coveredEnergy = 0.0;
                            const double targetCoverage = 0.995;
                            for (const auto& [energy, tileIdx] : tileEnergy) {
                                if (energy <= 0.0) continue;
                                candidateMask |= (1ull << tileIdx);
                                coveredEnergy += energy;
                                if (totalTileEnergy > 0.0 &&
                                    (coveredEnergy / totalTileEnergy) >= targetCoverage) {
                                    break;
                                }
                            }
                            const int activeTileCount = popcount64(candidateMask);
                            if (activeTileCount > 0) {
                                const int temporalKeep =
                                    std::max(1, std::min(m_options.genericTileTemporalKeep, activeFrames));
                                auto tileTemporalEncoding =
                                    encodeDenseTileTemporalBasis(genericTemporalResidualSeries,
                                                                 activeFrames,
                                                                 m_options.genericTileBits,
                                                                 candidateTileSize,
                                                                 candidateMask,
                                                                 temporalKeep);
                                const uint64_t tileTemporalBytes =
                                    estimateDenseTileTemporalBasisBytes(temporalKeep,
                                                                        m_options.genericTileBits,
                                                                        activeTileCount,
                                                                        candidateTileSize);
                                DistortionProxyStats tileTemporalDistortionStats{};
                                if (useGenericRdo) {
                                    const auto tileTemporalUniformStats =
                                        estimateResidualProxySamples(genericTemporalResidualSeries,
                                                                     activeFrames,
                                                                     m_options.genericRdoSpatialStride,
                                                                     m_options.genericRdoTimeStride,
                                                                     static_cast<float>(routeDenom),
                                                                     [&](int, int x, int y, int z, int lt) {
                                                                         return decodeDenseTileTemporalBasisAt(tileTemporalEncoding, x, y, z, lt, activeFrames);
                                                                     });
                                    const auto tileTemporalHotStats =
                                        estimateResidualProxyCandidates(genericTemporalResidualSeries,
                                                                        activeFrames,
                                                                        static_cast<float>(routeDenom),
                                                                        thresholdCandidates,
                                                                        [&](int, int x, int y, int z, int lt) {
                                                                            return decodeDenseTileTemporalBasisAt(tileTemporalEncoding, x, y, z, lt, activeFrames);
                                                                        });
                                    tileTemporalDistortionStats =
                                        maxDistortionProxyStats(tileTemporalUniformStats, tileTemporalHotStats);
                                }
                                const double tileTemporalRdoScore = useGenericRdo
                                    ? computeRdoScore(tileTemporalDistortionStats,
                                                      tileTemporalBytes,
                                                      m_options.genericRdoLambda,
                                                      m_options.genericRdoP99Weight,
                                                      m_options.genericRdoPeakWeight,
                                                      m_options.genericRdoUseMaxEnvelope,
                                                      genericChaosAdaptiveRdo)
                                    : std::numeric_limits<double>::infinity();
                                const bool chooseTileTemporal = m_options.genericTileTemporalBasisForce
                                    ? true
                                    : (useGenericRdo
                                           ? (tileTemporalRdoScore < bestRdoScore)
                                           : (tileTemporalBytes < bestResidualBytes));
                                if (chooseTileTemporal) {
                                    denseChoice = GenericDenseChoice::TileTemporal;
                                    bestResidualBytes = tileTemporalBytes;
                                    tileTemporalMask = candidateMask;
                                    tileTemporalCount = activeTileCount;
                                    tileTemporalSize = candidateTileSize;
                                    bestTileTemporalEncoding = std::move(tileTemporalEncoding);
                                    if (useGenericRdo) bestRdoScore = tileTemporalRdoScore;
                                }
                            }
                        }

                        if (denseChoice == GenericDenseChoice::DenseGrid) {
                            encoding.mode = BlockMode::DENSE_FINE;
                            encoding.useDenseResidualBfp = !bestDenseGridUseTemporalBasis;
                            encoding.useDenseTemporalBasis = bestDenseGridUseTemporalBasis;
                            encoding.useDenseTileTemporalBasis = false;
                            encoding.useDensePatchTemporalBasis = false;
                            if (bestDenseGridUseTemporalBasis) {
                                encoding.denseTemporalBasis = std::move(bestDenseTemporalEncoding);
                            } else {
                                encoding.denseResidualBfp = std::move(bestDenseGridEncoding);
                            }
                            encoding.config = static_cast<uint16_t>(((bestDenseGridRes & 0x0F) << 12) |
                                                                    ((m_options.genericDenseResidualBits & 0xFF) << 4) |
                                                                    0x01);
                            encoding.events.clear();
                            encoding.packedCoords.clear();
                            encoding.packedResiduals.clear();
                            encoding.packedEventCount = 0;
                            encoding.packedTierCapacity = 0;
                            encoding.spatialBitmask.fill(0u);
                            encoding.timeBins.fill(0u);
                            encoding.eventScale = 0.0f;
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                            summary.mode2DenseCrossoverCount += 1;
                            if (genericChaosAdaptiveRdo) {
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                                summary.genericChaosDenseWins += 1;
                            }
                            if (bestDenseGridRes <= 3) {
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                                summary.mode2DenseGrid3Count += 1;
                            } else {
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                                summary.mode2DenseGrid4Count += 1;
                            }
                            if (bestDenseGridUseTemporalBasis) {
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                                summary.mode2DenseTemporalBasisCount += 1;
                            }
                        } else if (denseChoice == GenericDenseChoice::TileTemporal) {
                            encoding.mode = BlockMode::DENSE_FINE;
                            encoding.useDenseResidualBfp = false;
                            encoding.useDenseTemporalBasis = false;
                            encoding.useDenseTileTemporalBasis = true;
                            encoding.useDensePatchTemporalBasis = false;
                            encoding.denseTileTemporalBasis = std::move(bestTileTemporalEncoding);
                            encoding.config = static_cast<uint16_t>(((m_options.genericTileBits & 0xFF) << 8) |
                                                                    ((tileTemporalSize & 0x0F) << 4) |
                                                                    0x06);
                            encoding.events.clear();
                            encoding.packedCoords.clear();
                            encoding.packedResiduals.clear();
                            encoding.packedEventCount = 0;
                            encoding.packedTierCapacity = 0;
                            encoding.spatialBitmask.fill(0u);
                            encoding.timeBins.fill(0u);
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                            summary.mode2DenseCrossoverCount += 1;
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                            summary.mode2TileTemporalBasisCount += 1;
                        } else if (denseChoice == GenericDenseChoice::PatchTemporal) {
                            encoding.mode = BlockMode::DENSE_FINE;
                            encoding.useDenseResidualBfp = false;
                            encoding.useDenseTemporalBasis = false;
                            encoding.useDenseTileTemporalBasis = false;
                            encoding.useDensePatchTemporalBasis = true;
                            encoding.densePatchTemporalBasis = std::move(bestPatchTemporalEncoding);
                            encoding.config = static_cast<uint16_t>(((m_options.genericMultipatchBits & 0xFF) << 8) | 0x07);
                            encoding.events.clear();
                            encoding.packedCoords.clear();
                            encoding.packedResiduals.clear();
                            encoding.packedEventCount = 0;
                            encoding.packedTierCapacity = 0;
                            encoding.spatialBitmask.fill(0u);
                            encoding.timeBins.fill(0u);
                            encoding.eventScale = 0.0f;
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                            summary.mode2DenseCrossoverCount += 1;
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                            summary.mode2PatchTemporalBasisCount += 1;
                        } else if (denseChoice == GenericDenseChoice::Multipatch) {
                            const int valuesPerFrame = multipatchPatchCount * 64;
                            std::vector<float> patchSeries(static_cast<size_t>(valuesPerFrame * multipatchDt), 0.0f);
                            int patchBase = 0;
                            for (int p = 0; p < 8; ++p) {
                                const uint8_t bit = static_cast<uint8_t>(1u << p);
                                if ((multipatchMask & bit) == 0u) continue;
                                const int baseX = (p & 1) * 4;
                                const int baseY = ((p >> 1) & 1) * 4;
                                const int baseZ = ((p >> 2) & 1) * 4;
                                for (int lt = 0; lt < multipatchDt; ++lt) {
                                    const int srcT = multipatchTStart + lt;
                                    for (int lz = 0; lz < 4; ++lz) {
                                        for (int ly = 0; ly < 4; ++ly) {
                                            for (int lx = 0; lx < 4; ++lx) {
                                                const int localIdx = (lz * 4 + ly) * 4 + lx;
                                                const int dstIdx = patchBase * 64 + localIdx;
                                                const int srcIdx = localIndex(baseX + lx, baseY + ly, baseZ + lz);
                                                patchSeries[static_cast<size_t>(dstIdx * multipatchDt + lt)] =
                                                    genericTemporalResidualSeries[static_cast<size_t>(srcIdx * activeFrames + srcT)];
                                            }
                                        }
                                    }
                                }
                                patchBase += 1;
                            }
                            encoding.mode = BlockMode::DENSE_FINE;
                            encoding.useDenseResidualBfp = true;
                            encoding.useDenseTemporalBasis = false;
                            encoding.useDenseTileTemporalBasis = false;
                            encoding.useDensePatchTemporalBasis = false;
                            encoding.denseResidualBfp = encodeDenseResidualBfp(patchSeries,
                                                                                multipatchDt,
                                                                                m_options.genericMultipatchBits,
                                                                                valuesPerFrame,
                                                                                4,
                                                                                false,
                                                                                0, 0, 0,
                                                                                8, 8, 8,
                                                                                multipatchTStart,
                                                                                true,
                                                                                multipatchMask,
                                                                                multipatchPatchCount);
                            encoding.config = static_cast<uint16_t>(((m_options.genericMultipatchBits & 0xFF) << 8) | 0x03);
                            encoding.events.clear();
                            encoding.packedCoords.clear();
                            encoding.packedResiduals.clear();
                            encoding.packedEventCount = 0;
                            encoding.packedTierCapacity = 0;
                            encoding.spatialBitmask.fill(0u);
                            encoding.timeBins.fill(0u);
                            encoding.eventScale = 0.0f;
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                            summary.mode2MultipatchCrossoverCount += 1;
                        } else if (denseChoice == GenericDenseChoice::Tile) {
                            const int patchVoxelCount = tileSize * tileSize * tileSize;
                            const int tilesPerAxis = 8 / tileSize;
                            const int valuesPerFrame = tileCount * patchVoxelCount;
                            std::vector<float> tileSeries(static_cast<size_t>(valuesPerFrame * tileDt), 0.0f);
                            int tileBase = 0;
                            const int totalTiles = tilesPerAxis * tilesPerAxis * tilesPerAxis;
                            for (int p = 0; p < totalTiles; ++p) {
                                const uint64_t bit = (1ull << p);
                                if ((tileMask & bit) == 0ull) continue;
                                const int baseX = (p % tilesPerAxis) * tileSize;
                                const int baseY = ((p / tilesPerAxis) % tilesPerAxis) * tileSize;
                                const int baseZ = (p / (tilesPerAxis * tilesPerAxis)) * tileSize;
                                for (int lt = 0; lt < tileDt; ++lt) {
                                    const int srcT = tileTStart + lt;
                                    for (int lz = 0; lz < tileSize; ++lz) {
                                        for (int ly = 0; ly < tileSize; ++ly) {
                                            for (int lx = 0; lx < tileSize; ++lx) {
                                                const int localIdx = (lz * tileSize + ly) * tileSize + lx;
                                                const int dstIdx = tileBase * patchVoxelCount + localIdx;
                                                const int srcIdx = localIndex(baseX + lx, baseY + ly, baseZ + lz);
                                                tileSeries[static_cast<size_t>(dstIdx * tileDt + lt)] =
                                                    genericTemporalResidualSeries[static_cast<size_t>(srcIdx * activeFrames + srcT)];
                                            }
                                        }
                                    }
                                }
                                tileBase += 1;
                            }
                            encoding.mode = BlockMode::DENSE_FINE;
                            encoding.useDenseResidualBfp = true;
                            encoding.useDenseTemporalBasis = false;
                            encoding.useDenseTileTemporalBasis = false;
                            encoding.useDensePatchTemporalBasis = false;
                            encoding.denseResidualBfp = encodeDenseResidualBfp(tileSeries,
                                                                                tileDt,
                                                                                m_options.genericTileBits,
                                                                                valuesPerFrame,
                                                                                tileSize,
                                                                                false,
                                                                                0, 0, 0,
                                                                                8, 8, 8,
                                                                                tileTStart,
                                                                                true,
                                                                                tileMask,
                                                                                tileCount,
                                                                                tileSize);
                            encoding.config = static_cast<uint16_t>(((m_options.genericTileBits & 0xFF) << 8) |
                                                                    ((tileSize & 0x0F) << 4) |
                                                                    0x04);
                            encoding.events.clear();
                            encoding.packedCoords.clear();
                            encoding.packedResiduals.clear();
                            encoding.packedEventCount = 0;
                            encoding.packedTierCapacity = 0;
                            encoding.spatialBitmask.fill(0u);
                            encoding.timeBins.fill(0u);
                            encoding.eventScale = 0.0f;
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                            summary.mode2TileCrossoverCount += 1;
                        } else if (denseChoice == GenericDenseChoice::Subgrid) {
                            const int valuesPerFrame = subgridDx * subgridDy * subgridDz;
                            std::vector<float> subgridSeries(static_cast<size_t>(valuesPerFrame * subgridDt), 0.0f);
                            for (int lt = 0; lt < subgridDt; ++lt) {
                                const int srcT = subgridTStart + lt;
                                for (int lz = 0; lz < subgridDz; ++lz) {
                                    for (int ly = 0; ly < subgridDy; ++ly) {
                                        for (int lx = 0; lx < subgridDx; ++lx) {
                                            const int dstIdx = (lz * subgridDy + ly) * subgridDx + lx;
                                            const int srcIdx = localIndex(subgridXMin + lx,
                                                                          subgridYMin + ly,
                                                                          subgridZMin + lz);
                                            subgridSeries[static_cast<size_t>(dstIdx * subgridDt + lt)] =
                                                genericTemporalResidualSeries[static_cast<size_t>(srcIdx * activeFrames + srcT)];
                                        }
                                    }
                                }
                            }
                            encoding.mode = BlockMode::DENSE_FINE;
                            encoding.useDenseResidualBfp = true;
                            encoding.useDenseTemporalBasis = false;
                            encoding.useDenseTileTemporalBasis = false;
                            encoding.useDensePatchTemporalBasis = false;
                            encoding.denseResidualBfp = encodeDenseResidualBfp(subgridSeries,
                                                                                subgridDt,
                                                                                m_options.genericSubgridBits,
                                                                                valuesPerFrame,
                                                                                std::max({subgridDx, subgridDy, subgridDz}),
                                                                                false,
                                                                                subgridXMin,
                                                                                subgridYMin,
                                                                                subgridZMin,
                                                                                subgridDx,
                                                                                subgridDy,
                                                                                subgridDz,
                                                                                subgridTStart);
                            encoding.config = static_cast<uint16_t>(((m_options.genericSubgridBits & 0xFF) << 8) | 0x02);
                            encoding.events.clear();
                            encoding.packedCoords.clear();
                            encoding.packedResiduals.clear();
                            encoding.packedEventCount = 0;
                            encoding.packedTierCapacity = 0;
                            encoding.spatialBitmask.fill(0u);
                            encoding.timeBins.fill(0u);
                            encoding.eventScale = 0.0f;
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                            summary.mode2DenseCrossoverCount += 1;
                        } else if (allowGenericDense) {
                            encoding.config = encoding.packedEventCount;
                        } else {
                            encoding.config = encoding.packedEventCount;
                        }
                    }
                }

                if (m_options.dumpRoutingStats) {
                    LeafRouteStat stat;
                    stat.bx = bx;
                    stat.by = by;
                    stat.bz = bz;
                    stat.leafRange = leafRange;
                    stat.coarseRmse = temporalCoarseRmse;
                    stat.normErr = normErr;
                    stat.peakErrNorm = peakErrNorm;
                    stat.topEnergyFrac = topEnergyFrac;
                    stat.activeFrames = activeFrames;
                    stat.mode = encoding.mode;
#ifdef VBT_USE_OPENMP
#pragma omp critical(vbt_route_stats)
#endif
                    summary.routeStats.push_back(stat);
                }
                if (m_options.residualDiagnostics &&
                    m_options.profile.type == FieldType::GENERIC &&
                    encoding.mode != BlockMode::CONSTANT) {
                    ResidualDiagnosticCandidate diag;
                    diag.leafId = leafId;
                    diag.bx = bx;
                    diag.by = by;
                    diag.bz = bz;
                    diag.activeFrames = activeFrames;
                    diag.mode = encoding.mode;
                    diag.normErr = normErr;
                    diag.peakErrNorm = peakErrNorm;
                    diag.topEnergyFrac = topEnergyFrac;
                    diag.spatialOccupancy = thresholdSpatialOccupancy;
                    diag.timeOccupancy = thresholdTimeOccupancy;
                    diag.score =
                        (0.25 + peakErrNorm + 0.5 * normErr) *
                        (0.5 + thresholdSpatialOccupancy) *
                        (0.5 + thresholdTimeOccupancy);
                    if (encoding.mode == BlockMode::SPARSE_IMPULSE) diag.score *= 1.25;
#ifdef VBT_USE_OPENMP
#pragma omp critical(vbt_residual_diag_candidates)
#endif
                    residualDiagnosticCandidates.push_back(diag);
                }

                encoding.packedHeader = packHeader(m_options.profile, m_options, encoding);

                if (m_options.profile.type == FieldType::GENERIC) {
                    const int bucket = coarseKeepBucketIndex(resolveCoarseKeep(m_options, encoding));
                    if (bucket >= 0) {
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                        summary.scientificCoarseKeepCounts[static_cast<size_t>(bucket)] += 1;
                    } else {
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                        summary.scientificCoarseKeepOtherCount += 1;
                    }
                }

#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                summary.coarseCoefficientCount +=
                    static_cast<uint64_t>(kCoarseControlCount * resolveCoarseKeep(m_options, encoding));
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                summary.fineCoefficientCount += static_cast<uint64_t>(encoding.useFineResidualGrid ? fineControlCount * m_options.fineGridDctKeep : 0);
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                summary.eventCount += static_cast<uint64_t>((encoding.useFineResidualGrid || encoding.useDenseResidualBfp || encoding.useDenseTemporalBasis || encoding.useDenseTileTemporalBasis || encoding.useDensePatchTemporalBasis) ? 0 : encoding.events.size());

                const uint64_t coarseBytes =
                    static_cast<uint64_t>(kCoarseControlCount) *
                    static_cast<uint64_t>(2 + std::max(0, resolveCoarseKeep(m_options, encoding) - 1));
                const uint64_t residualBytes = encoding.useFineResidualGrid
                    ? (encoding.fineResidualGrid.quantBits > 0
                    ? (4ull + alignWords((static_cast<uint64_t>(encoding.fineResidualGrid.quantizedCoeffs.size()) *
                                          static_cast<uint64_t>(encoding.fineResidualGrid.quantBits) + 7ull) / 8ull) * 4ull)
                        : static_cast<uint64_t>(fineControlCount) * static_cast<uint64_t>(2 + std::max(0, m_options.fineGridDctKeep - 1)))
                    : encoding.useDenseResidualBfp
                    ? (static_cast<uint64_t>(encoding.encodedFrameCount) * static_cast<uint64_t>(2 + encoding.denseResidualBfp.frameBytes))
                    : encoding.useDenseTemporalBasis
                    ? estimateDenseTemporalBasisBytes(encoding.denseTemporalBasis.temporalKeep,
                                                      encoding.denseTemporalBasis.bitsPerValue,
                                                      encoding.denseTemporalBasis.valuesPerBasis)
                    : encoding.useDenseTileTemporalBasis
                    ? estimateDenseTileTemporalBasisBytes(encoding.denseTileTemporalBasis.temporalKeep,
                                                          encoding.denseTileTemporalBasis.bitsPerValue,
                                                          encoding.denseTileTemporalBasis.activeTileCount,
                                                          encoding.denseTileTemporalBasis.tileSize)
                    : encoding.useDensePatchTemporalBasis
                    ? estimateDensePatchTemporalBasisBytes(encoding.densePatchTemporalBasis.temporalKeep,
                                                           encoding.densePatchTemporalBasis.bitsPerValue,
                                                           encoding.densePatchTemporalBasis.activePatchCount)
                    : (encoding.events.empty() ? 0ull
                        : estimateMode2ResidualBytes(encoding.packedTierCapacity));
                uint64_t leafBytes = 0;
                switch (encoding.mode) {
                case BlockMode::CONSTANT:
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                    summary.mode0Count += 1;
                    leafBytes = 8;
                    break;
                case BlockMode::COARSE_ONLY:
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                    summary.mode1Count += 1;
                    leafBytes = 4 + coarseBytes;
                    break;
                case BlockMode::SPARSE_IMPULSE:
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                    summary.mode2Count += 1;
                    {
                        std::array<uint64_t, 8> spatialMask{0, 0, 0, 0, 0, 0, 0, 0};
                        std::array<uint64_t, 2> timeMask{0, 0};
                        for (const auto& evt : encoding.events) {
                            const uint16_t idx = evt.spatialIndex;
                            spatialMask[static_cast<size_t>(idx >> 6u)] |= (1ull << (idx & 63u));
                            const uint16_t t = static_cast<uint16_t>(evt.timeTag);
                            timeMask[static_cast<size_t>(t >> 6u)] |= (1ull << (t & 63u));
                        }
                        uint32_t uniqueSpatial = 0;
                        for (uint64_t word : spatialMask) {
                            while (word != 0ull) {
                                word &= (word - 1ull);
                                uniqueSpatial += 1u;
                            }
                        }
                        uint32_t uniqueTime = 0;
                        for (uint64_t word : timeMask) {
                            while (word != 0ull) {
                                word &= (word - 1ull);
                                uniqueTime += 1u;
                            }
                        }
#ifdef VBT_USE_OPENMP
#pragma omp critical(vbt_mode2_stats)
#endif
                        {
                            summary.mode2EventCounts.push_back(static_cast<uint32_t>(encoding.events.size()));
                            summary.mode2UniqueSpatialCounts.push_back(static_cast<uint16_t>(uniqueSpatial));
                            summary.mode2UniqueTimeCounts.push_back(static_cast<uint16_t>(uniqueTime));
                            summary.mode2EncodedFrameCounts.push_back(static_cast<uint16_t>(encoding.encodedFrameCount));
                        }
                    }
                    leafBytes = 4 + coarseBytes + residualBytes;
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                    summary.eventPayloadBytes += residualBytes;
                    break;
                case BlockMode::DENSE_FINE:
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                    summary.mode3Count += 1;
                    leafBytes = 4 + coarseBytes + residualBytes;
                    break;
                }
#ifdef VBT_USE_OPENMP
#pragma omp atomic
#endif
                summary.payloadWords += alignWords(leafBytes);
    }

    summary.offsetTableWords = static_cast<uint64_t>(leafCount) + 1ull;
    summary.totalWords = summary.offsetTableWords + summary.payloadWords;
    summary.estimatedBytes = summary.totalWords * 4ull;
    if (!m_options.saveVbtPath.empty()) {
        summary.savedFileBytes = writeProbeFile(m_options.saveVbtPath, volume, m_options, encodings);
    }
    if (m_options.residualDiagnostics &&
        m_options.profile.type == FieldType::GENERIC &&
        !residualDiagnosticCandidates.empty()) {
        writeResidualDiagnostics(volume, m_options, encodings, std::move(residualDiagnosticCandidates));
    }

    if (m_options.fullEvaluation) {
        const int width = volume.meta.width;
        const int height = volume.meta.height;
        const int depth = volume.meta.depth;
        const int frames = volume.meta.frames;
        const int64_t totalVoxelSamples =
            static_cast<int64_t>(width) *
            static_cast<int64_t>(height) *
            static_cast<int64_t>(depth) *
            static_cast<int64_t>(frames);
        const double peak = std::max(1e-6f, volume.meta.dataMax - volume.meta.dataMin);
        const int totalLeaves = leafCountX * leafCountY * leafCountZ;

        double errSum2 = 0.0;
        double pairedCoarseErr2 = 0.0;
        double sparseDiagCoarseErr2 = 0.0;
        double sparseDiagModeErr2 = 0.0;
        double maxAbsErr = 0.0;
        uint64_t samples = 0;
        uint64_t sparseSamples = 0;
        uint64_t sparseImproved = 0;
        uint64_t sparseWorsened = 0;

#ifdef VBT_USE_OPENMP
#pragma omp parallel
#endif
        {
            double localErr2 = 0.0;
            double localPairedCoarseErr2 = 0.0;
            double localSparseDiagCoarseErr2 = 0.0;
            double localSparseDiagModeErr2 = 0.0;
            double localMaxAbsErr = 0.0;
            uint64_t localSamples = 0;
            uint64_t localSparseSamples = 0;
            uint64_t localSparseImproved = 0;
            uint64_t localSparseWorsened = 0;

#ifdef VBT_USE_OPENMP
#pragma omp for schedule(static)
#endif
            for (int leafId = 0; leafId < totalLeaves; ++leafId) {
                const int bz = leafId / (leafCountX * leafCountY);
                const int rem = leafId % (leafCountX * leafCountY);
                const int by = rem / leafCountX;
                const int bx = rem % leafCountX;
                const int x0 = bx * 8;
                const int y0 = by * 8;
                const int z0 = bz * 8;
                const int x1 = std::min(x0 + 8, width);
                const int y1 = std::min(y0 + 8, height);
                const int z1 = std::min(z0 + 8, depth);
                const auto& encoding = encodings[static_cast<size_t>(leafId)];
                std::array<float, kCoarseControlCount> coarseCtrl{};
                std::vector<float> fineCtrl;
                if (encoding.useFineResidualGrid) {
                    fineCtrl.resize(static_cast<size_t>(encoding.fineResidualGrid.resolution *
                                                        encoding.fineResidualGrid.resolution *
                                                        encoding.fineResidualGrid.resolution), 0.0f);
                }
                std::array<float, kLeafVoxelCount> sparseResidualMap{};
                const int coarseKeep = resolveCoarseKeep(m_options, encoding);

                for (int t = 0; t < frames; ++t) {
                    const int localT = clampToEncodedFrame(encoding, t);
                    const bool withinWindow =
                        t >= static_cast<int>(encoding.startFrame) &&
                        t <= static_cast<int>(encoding.endFrame);

                    if (encoding.mode != BlockMode::CONSTANT) {
                        // Exact evaluation decodes per-leaf/per-frame controls
                        // once, then reuses them for all 8x8x8 voxels.
                        const auto coarseRow = buildDctDecodeRow(encoding.encodedFrameCount, coarseKeep, localT);
                        decodeDctCoeffBlockAt(encoding.coarseCoeffs, kCoarseControlCount, coarseKeep, coarseRow, coarseCtrl.data());
                    }

                    if (encoding.useFineResidualGrid) {
                        const auto fineRow = buildDctDecodeRow(encoding.encodedFrameCount,
                                                               encoding.fineResidualGrid.dctKeep,
                                                               localT);
                        if (!encoding.fineResidualGrid.quantizedCoeffs.empty()) {
                            decodeQuantizedDctCoeffBlockAt(encoding.fineResidualGrid.quantizedCoeffs,
                                                           encoding.fineResidualGrid.blockScale,
                                                           static_cast<int>(fineCtrl.size()),
                                                           encoding.fineResidualGrid.dctKeep,
                                                           fineRow,
                                                           fineCtrl.data());
                        } else {
                            decodeDctCoeffBlockAt(encoding.fineResidualGrid.coeffs,
                                                  static_cast<int>(fineCtrl.size()),
                                                  encoding.fineResidualGrid.dctKeep,
                                                  fineRow,
                                                  fineCtrl.data());
                        }
                    } else if (!encoding.useDenseResidualBfp && !encoding.useDenseTemporalBasis && !encoding.useDenseTileTemporalBasis && !encoding.useDensePatchTemporalBasis && encoding.mode == BlockMode::SPARSE_IMPULSE && withinWindow) {
                        buildSparseResidualFrameMap(encoding, localT, sparseResidualMap);
                    } else {
                        sparseResidualMap.fill(0.0f);
                    }

                    for (int z = z0; z < z1; ++z) {
                        const int lz = z - z0;
                        for (int y = y0; y < y1; ++y) {
                            const int ly = y - y0;
                            for (int x = x0; x < x1; ++x) {
                                const int lx = x - x0;
                                const float truth = volume.at(x, y, z, t);
                                const float coarse = (encoding.mode == BlockMode::CONSTANT)
                                    ? encoding.constantValue
                                    : sampleTrilinear4(coarseCtrl, lx, ly, lz);

                                float residual = 0.0f;
                                if (encoding.mode == BlockMode::CONSTANT || !withinWindow) {
                                    residual = 0.0f;
                                } else if (encoding.useFineResidualGrid) {
                                    residual = sampleTrilinearGrid(fineCtrl, encoding.fineResidualGrid.resolution, lx, ly, lz);
                                } else if (encoding.useDenseResidualBfp) {
                                    residual = decodeDenseResidualBfpAt(encoding.denseResidualBfp, lx, ly, lz, localT);
                                } else if (encoding.useDenseTemporalBasis) {
                                    residual = decodeDenseTemporalBasisAt(encoding.denseTemporalBasis, lx, ly, lz, localT, encoding.encodedFrameCount);
                                } else if (encoding.useDenseTileTemporalBasis) {
                                    residual = decodeDenseTileTemporalBasisAt(encoding.denseTileTemporalBasis, lx, ly, lz, localT, encoding.encodedFrameCount);
                                } else if (encoding.useDensePatchTemporalBasis) {
                                    residual = decodeDensePatchTemporalBasisAt(encoding.densePatchTemporalBasis, lx, ly, lz, localT, encoding.encodedFrameCount);
                                } else if (encoding.mode == BlockMode::SPARSE_IMPULSE) {
                                    residual = sparseResidualMap[static_cast<size_t>(localIndex(lx, ly, lz))];
                                }
                                const float pred = coarse + residual;

                                const double coarseErr = std::abs(static_cast<double>(truth) - static_cast<double>(coarse));
                                const double modeErr = std::abs(static_cast<double>(truth) - static_cast<double>(pred));
                                localPairedCoarseErr2 += coarseErr * coarseErr;
                                localErr2 += modeErr * modeErr;
                                localMaxAbsErr = std::max(localMaxAbsErr, modeErr);
                                localSamples += 1;

                                if (encoding.mode == BlockMode::SPARSE_IMPULSE) {
                                    localSparseDiagCoarseErr2 += coarseErr * coarseErr;
                                    localSparseDiagModeErr2 += modeErr * modeErr;
                                    localSparseSamples += 1;
                                    if (modeErr < coarseErr) localSparseImproved += 1;
                                    else if (modeErr > coarseErr) localSparseWorsened += 1;
                                }
                            }
                        }
                    }
                }
            }

#ifdef VBT_USE_OPENMP
#pragma omp critical
#endif
            {
                errSum2 += localErr2;
                pairedCoarseErr2 += localPairedCoarseErr2;
                sparseDiagCoarseErr2 += localSparseDiagCoarseErr2;
                sparseDiagModeErr2 += localSparseDiagModeErr2;
                maxAbsErr = std::max(maxAbsErr, localMaxAbsErr);
                samples += localSamples;
                sparseSamples += localSparseSamples;
                sparseImproved += localSparseImproved;
                sparseWorsened += localSparseWorsened;
            }
        }

        summary.evaluatedSamples = samples;
        summary.rmse = std::sqrt(errSum2 / static_cast<double>(samples));
        summary.psnr = computePsnr(summary.rmse, peak);
        summary.p99 = 0.0;
        summary.p999 = 0.0;
        summary.maxAbsError = maxAbsErr;
        summary.p999Valid = false;
        summary.tailMetricsApproximate = false;
        summary.pairedCoarseRmse = std::sqrt(pairedCoarseErr2 / static_cast<double>(samples));
        summary.pairedCoarsePsnr = computePsnr(summary.pairedCoarseRmse, peak);
        summary.sparseDiagSamples = sparseSamples;
        summary.sparseDiagImproved = sparseImproved;
        summary.sparseDiagWorsened = sparseWorsened;
        if (sparseSamples > 0) {
            summary.sparseDiagCoarseRmse = std::sqrt(sparseDiagCoarseErr2 / static_cast<double>(sparseSamples));
            summary.sparseDiagModeRmse = std::sqrt(sparseDiagModeErr2 / static_cast<double>(sparseSamples));
            summary.sparseDiagCoarsePsnr = computePsnr(summary.sparseDiagCoarseRmse, peak);
            summary.sparseDiagModePsnr = computePsnr(summary.sparseDiagModeRmse, peak);
        }
        if (samples > 0 && maxAbsErr > 0.0) {
            constexpr size_t kTailBins = 262144;
            std::vector<uint64_t> histogram(kTailBins, 0ull);
            const double invMaxErr = 1.0 / maxAbsErr;

#ifdef VBT_USE_OPENMP
#pragma omp parallel
#endif
            {
                std::vector<uint64_t> localHist(kTailBins, 0ull);

#ifdef VBT_USE_OPENMP
#pragma omp for schedule(static)
#endif
                for (int leafId = 0; leafId < totalLeaves; ++leafId) {
                    const int bz = leafId / (leafCountX * leafCountY);
                    const int rem = leafId % (leafCountX * leafCountY);
                    const int by = rem / leafCountX;
                    const int bx = rem % leafCountX;
                    const int x0 = bx * 8;
                    const int y0 = by * 8;
                    const int z0 = bz * 8;
                    const int x1 = std::min(x0 + 8, width);
                    const int y1 = std::min(y0 + 8, height);
                    const int z1 = std::min(z0 + 8, depth);
                    const auto& encoding = encodings[static_cast<size_t>(leafId)];
                    std::array<float, kCoarseControlCount> coarseCtrl{};
                    std::vector<float> fineCtrl;
                    if (encoding.useFineResidualGrid) {
                        fineCtrl.resize(static_cast<size_t>(encoding.fineResidualGrid.resolution *
                                                            encoding.fineResidualGrid.resolution *
                                                            encoding.fineResidualGrid.resolution), 0.0f);
                    }
                    std::array<float, kLeafVoxelCount> sparseResidualMap{};
                    const int coarseKeep = resolveCoarseKeep(m_options, encoding);

                    for (int t = 0; t < frames; ++t) {
                        const int localT = clampToEncodedFrame(encoding, t);
                        const bool withinWindow =
                            t >= static_cast<int>(encoding.startFrame) &&
                            t <= static_cast<int>(encoding.endFrame);

                        if (encoding.mode != BlockMode::CONSTANT) {
                            const auto coarseRow = buildDctDecodeRow(encoding.encodedFrameCount, coarseKeep, localT);
                            decodeDctCoeffBlockAt(encoding.coarseCoeffs, kCoarseControlCount, coarseKeep, coarseRow, coarseCtrl.data());
                        }

                        if (encoding.useFineResidualGrid) {
                            const auto fineRow = buildDctDecodeRow(encoding.encodedFrameCount,
                                                                   encoding.fineResidualGrid.dctKeep,
                                                                   localT);
                            if (!encoding.fineResidualGrid.quantizedCoeffs.empty()) {
                                decodeQuantizedDctCoeffBlockAt(encoding.fineResidualGrid.quantizedCoeffs,
                                                               encoding.fineResidualGrid.blockScale,
                                                               static_cast<int>(fineCtrl.size()),
                                                               encoding.fineResidualGrid.dctKeep,
                                                               fineRow,
                                                               fineCtrl.data());
                            } else {
                                decodeDctCoeffBlockAt(encoding.fineResidualGrid.coeffs,
                                                      static_cast<int>(fineCtrl.size()),
                                                      encoding.fineResidualGrid.dctKeep,
                                                      fineRow,
                                                      fineCtrl.data());
                            }
                        } else if (!encoding.useDenseResidualBfp && !encoding.useDenseTemporalBasis && !encoding.useDenseTileTemporalBasis && !encoding.useDensePatchTemporalBasis && encoding.mode == BlockMode::SPARSE_IMPULSE && withinWindow) {
                            buildSparseResidualFrameMap(encoding, localT, sparseResidualMap);
                        } else {
                            sparseResidualMap.fill(0.0f);
                        }

                        for (int z = z0; z < z1; ++z) {
                            const int lz = z - z0;
                            for (int y = y0; y < y1; ++y) {
                                const int ly = y - y0;
                                for (int x = x0; x < x1; ++x) {
                                    const int lx = x - x0;
                                    const float truth = volume.at(x, y, z, t);
                                    const float coarse = (encoding.mode == BlockMode::CONSTANT)
                                        ? encoding.constantValue
                                        : sampleTrilinear4(coarseCtrl, lx, ly, lz);

                                    float residual = 0.0f;
                                    if (encoding.mode == BlockMode::CONSTANT || !withinWindow) {
                                        residual = 0.0f;
                                    } else if (encoding.useFineResidualGrid) {
                                        residual = sampleTrilinearGrid(fineCtrl, encoding.fineResidualGrid.resolution, lx, ly, lz);
                                    } else if (encoding.useDenseResidualBfp) {
                                        residual = decodeDenseResidualBfpAt(encoding.denseResidualBfp, lx, ly, lz, localT);
                                    } else if (encoding.useDenseTemporalBasis) {
                                        residual = decodeDenseTemporalBasisAt(encoding.denseTemporalBasis, lx, ly, lz, localT, encoding.encodedFrameCount);
                                    } else if (encoding.useDenseTileTemporalBasis) {
                                        residual = decodeDenseTileTemporalBasisAt(encoding.denseTileTemporalBasis, lx, ly, lz, localT, encoding.encodedFrameCount);
                                    } else if (encoding.useDensePatchTemporalBasis) {
                                        residual = decodeDensePatchTemporalBasisAt(encoding.densePatchTemporalBasis, lx, ly, lz, localT, encoding.encodedFrameCount);
                                    } else if (encoding.mode == BlockMode::SPARSE_IMPULSE) {
                                        residual = sparseResidualMap[static_cast<size_t>(localIndex(lx, ly, lz))];
                                    }

                                    const double modeErr = std::abs(static_cast<double>(truth) - static_cast<double>(coarse + residual));
                                    const size_t bin = static_cast<size_t>(std::min<double>(
                                        static_cast<double>(kTailBins - 1),
                                        std::floor(modeErr * invMaxErr * static_cast<double>(kTailBins - 1))));
                                    localHist[bin] += 1ull;
                                }
                            }
                        }
                    }
                }

#ifdef VBT_USE_OPENMP
#pragma omp critical
#endif
                {
                    for (size_t i = 0; i < kTailBins; ++i) {
                        histogram[i] += localHist[i];
                    }
                }
            }

            const uint64_t targetP99 = static_cast<uint64_t>(std::ceil(static_cast<double>(samples) * 0.99));
            const uint64_t targetP999 = static_cast<uint64_t>(std::ceil(static_cast<double>(samples) * 0.999));
            uint64_t prefix = 0;
            size_t p99Bin = kTailBins - 1;
            size_t p999Bin = kTailBins - 1;
            for (size_t i = 0; i < kTailBins; ++i) {
                prefix += histogram[i];
                if (prefix >= targetP99 && p99Bin == kTailBins - 1) p99Bin = i;
                if (prefix >= targetP999) {
                    p999Bin = i;
                    break;
                }
            }
            summary.p99 = maxAbsErr * (static_cast<double>(p99Bin + 1) / static_cast<double>(kTailBins));
            summary.p999 = maxAbsErr * (static_cast<double>(p999Bin + 1) / static_cast<double>(kTailBins));
            summary.p999Valid = true;
            summary.tailMetricsApproximate = true;
        }
        return summary;
    }

    std::vector<double> errors;
    errors.reserve(1 << 20);
    double errSum2 = 0.0;
    double pairedCoarseErr2 = 0.0;
    double sparseDiagCoarseErr2 = 0.0;
    double sparseDiagModeErr2 = 0.0;
    const double peak = std::max(1e-6f, volume.meta.dataMax - volume.meta.dataMin);
    uint64_t samples = 0;

    const auto samplePhasesX = samplePhasesForAxis(m_options.sampleStepX, m_options.leafSize);
    const auto samplePhasesY = samplePhasesForAxis(m_options.sampleStepY, m_options.leafSize);
    const auto samplePhasesZ = samplePhasesForAxis(m_options.sampleStepZ, m_options.leafSize);

    for (int t = 0; t < volume.meta.frames; t += m_options.sampleStepT) {
        for (int phaseZ : samplePhasesZ) {
            for (int z = phaseZ; z < volume.meta.depth; z += m_options.sampleStepZ) {
                for (int phaseY : samplePhasesY) {
                    for (int y = phaseY; y < volume.meta.height; y += m_options.sampleStepY) {
                        for (int phaseX : samplePhasesX) {
                            for (int x = phaseX; x < volume.meta.width; x += m_options.sampleStepX) {
                    const int bx = x / 8;
                    const int by = y / 8;
                    const int bz = z / 8;
                    const int leafId = (bz * leafCountY + by) * leafCountX + bx;
                    const auto& encoding = encodings[static_cast<size_t>(leafId)];
                    const auto decoded = decodeLeafSampleAt(m_options, encoding, x % 8, y % 8, z % 8, t);
                    const float truth = volume.at(x, y, z, t);
                    const double coarseErr = std::abs(static_cast<double>(truth) - static_cast<double>(decoded.coarse));
                    const double modeErr = std::abs(static_cast<double>(truth) - static_cast<double>(decoded.pred));
                    pairedCoarseErr2 += coarseErr * coarseErr;
                    errors.push_back(modeErr);
                    errSum2 += modeErr * modeErr;
                    ++samples;

                    if (m_options.compareTemporalBaseline) {
                        const auto series = gatherSeriesAtSample(volume, x, y, z);
                        const auto keys = detectKeyFrames(series, m_options.profile);
                        const double baselinePred = decodeKeyframeSeriesAt(series, keys, t);
                        const double berr = std::abs(static_cast<double>(truth) - baselinePred);
                        summary.baselineRmse += berr * berr;
                        summary.baselineSampledKeyframes += static_cast<uint64_t>(keys.size());
                    }
                            }
                        }
                    }
                }
            }
        }

    }

    // Residual-aware supplement for sparse blocks: explicitly evaluate stored
    // events so sampled metrics reflect Mode-2 gains at event locations.
    for (int bz = 0; bz < leafCountZ; ++bz) {
        for (int by = 0; by < leafCountY; ++by) {
            for (int bx = 0; bx < leafCountX; ++bx) {
                const int leafId = (bz * leafCountY + by) * leafCountX + bx;
                const auto& encoding = encodings[static_cast<size_t>(leafId)];
                if (encoding.mode != BlockMode::SPARSE_IMPULSE) continue;
                if (encoding.events.empty()) continue;

                const int gx0 = bx * 8;
                const int gy0 = by * 8;
                const int gz0 = bz * 8;
                for (const auto& evt : encoding.events) {
                    const int localT = static_cast<int>(evt.timeTag);
                    const int t = encoding.startFrame + localT;
                    if (t < 0 || t >= volume.meta.frames) continue;

                    const int lx = static_cast<int>(evt.spatialIndex % 8u);
                    const int ly = static_cast<int>((evt.spatialIndex / 8u) % 8u);
                    const int lz = static_cast<int>(evt.spatialIndex / 64u);
                    const int x = gx0 + lx;
                    const int y = gy0 + ly;
                    const int z = gz0 + lz;
                    if (x >= volume.meta.width || y >= volume.meta.height || z >= volume.meta.depth) continue;

                    const bool alreadyInRegularSamples =
                        (t % m_options.sampleStepT) == 0 &&
                        hitsRegularSamplePhase(x, m_options.sampleStepX, samplePhasesX) &&
                        hitsRegularSamplePhase(y, m_options.sampleStepY, samplePhasesY) &&
                        hitsRegularSamplePhase(z, m_options.sampleStepZ, samplePhasesZ);
                    if (alreadyInRegularSamples) continue;

                    const auto decoded = decodeLeafSampleAt(m_options, encoding, lx, ly, lz, t);
                    const float truth = volume.at(x, y, z, t);
                    const double coarseErr = std::abs(static_cast<double>(truth) - static_cast<double>(decoded.coarse));
                    const double modeErr = std::abs(static_cast<double>(truth) - static_cast<double>(decoded.pred));
                    pairedCoarseErr2 += coarseErr * coarseErr;
                    errors.push_back(modeErr);
                    errSum2 += modeErr * modeErr;
                    ++samples;
                    sparseDiagCoarseErr2 += coarseErr * coarseErr;
                    sparseDiagModeErr2 += modeErr * modeErr;
                    summary.sparseDiagSamples += 1;
                    if (modeErr < coarseErr) summary.sparseDiagImproved += 1;
                    else if (modeErr > coarseErr) summary.sparseDiagWorsened += 1;
                }
            }
        }
    }

    std::sort(errors.begin(), errors.end());
    summary.evaluatedSamples = samples;
    summary.rmse = std::sqrt(errSum2 / static_cast<double>(samples));
    summary.psnr = computePsnr(summary.rmse, peak);
    summary.p99 = errors[std::min(errors.size() - 1, (errors.size() * 99) / 100)];
    summary.p999 = errors[std::min(errors.size() - 1, (errors.size() * 999) / 1000)];
    summary.maxAbsError = errors.empty() ? 0.0 : errors.back();
    summary.p999Valid = true;
    summary.tailMetricsApproximate = false;
    if (samples > 0) {
        summary.pairedCoarseRmse = std::sqrt(pairedCoarseErr2 / static_cast<double>(samples));
        summary.pairedCoarsePsnr = computePsnr(summary.pairedCoarseRmse, peak);
    }

    if (summary.sparseDiagSamples > 0) {
        summary.sparseDiagCoarseRmse = std::sqrt(sparseDiagCoarseErr2 / static_cast<double>(summary.sparseDiagSamples));
        summary.sparseDiagModeRmse = std::sqrt(sparseDiagModeErr2 / static_cast<double>(summary.sparseDiagSamples));
        summary.sparseDiagCoarsePsnr = computePsnr(summary.sparseDiagCoarseRmse, peak);
        summary.sparseDiagModePsnr = computePsnr(summary.sparseDiagModeRmse, peak);
    }

    if (m_options.compareTemporalBaseline && samples > 0) {
        summary.baselineRmse = std::sqrt(summary.baselineRmse / static_cast<double>(samples));
        summary.baselinePsnr = computePsnr(summary.baselineRmse, peak);
    }
    return summary;
}

} // namespace vbt
