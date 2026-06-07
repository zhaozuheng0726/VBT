#include "scientific_decode.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace vbt::render {

namespace {

float halfToFloat(uint16_t h)
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
    float out = 0.0f;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

uint8_t readPayloadByte(const VbtFile& file, uint32_t byteOffset)
{
    const uint32_t word = file.payloadWords.at(byteOffset >> 2u);
    const uint32_t shift = (byteOffset & 3u) * 8u;
    return static_cast<uint8_t>((word >> shift) & 0xFFu);
}

uint16_t readPayloadU16(const VbtFile& file, uint32_t byteOffset)
{
    const uint16_t lo = static_cast<uint16_t>(readPayloadByte(file, byteOffset));
    const uint16_t hi = static_cast<uint16_t>(readPayloadByte(file, byteOffset + 1u));
    return static_cast<uint16_t>(lo | static_cast<uint16_t>(hi << 8u));
}

int8_t readPayloadI8(const VbtFile& file, uint32_t byteOffset)
{
    return static_cast<int8_t>(readPayloadByte(file, byteOffset));
}

float decodeBfp(uint32_t q, float scale, int bits)
{
    if (bits <= 2) {
        static constexpr std::array<float, 4> kLevels = {-1.0f, -0.3333333f, 0.3333333f, 1.0f};
        return kLevels[std::min<size_t>(q, kLevels.size() - 1)] * scale;
    }
    const int levels = (1 << bits) - 1;
    const float norm = (static_cast<float>(q) / static_cast<float>(levels)) * 2.0f - 1.0f;
    return norm * scale;
}

float dctBasis(int totalLength, int index, int k)
{
    constexpr double kPi = 3.14159265358979323846;
    const double invN = 1.0 / static_cast<double>(totalLength);
    const double alpha = (k == 0) ? std::sqrt(invN) : std::sqrt(2.0 * invN);
    return static_cast<float>(alpha * std::cos((kPi / static_cast<double>(totalLength)) *
                                               (static_cast<double>(index) + 0.5) *
                                               static_cast<double>(k)));
}

float decodeCoarseControlAt(const VbtFile& file,
                            uint32_t payloadByteBase,
                            int keep,
                            int controlIndex,
                            int timeIndex)
{
    const uint32_t controlBase = payloadByteBase + static_cast<uint32_t>(4 + controlIndex * (keep + 1));
    double sum = static_cast<double>(halfToFloat(readPayloadU16(file, controlBase))) *
                 static_cast<double>(dctBasis(file.header.frames, timeIndex, 0));
    for (int k = 1; k < keep; ++k) {
        const float scale = file.coarseAcScales.at(static_cast<size_t>(k - 1));
        const float coeff = static_cast<float>(readPayloadI8(file, controlBase + 2u + static_cast<uint32_t>(k - 1u))) * scale;
        sum += static_cast<double>(coeff) * static_cast<double>(dctBasis(file.header.frames, timeIndex, k));
    }
    return static_cast<float>(sum);
}

float sampleCoarseAt(const std::array<float, 64>& ctrl, uint32_t x, uint32_t y, uint32_t z)
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
    auto at = [&](int cx, int cy, int cz) -> float {
        return ctrl[static_cast<size_t>((cz * 4 + cy) * 4 + cx)];
    };
    const float c00 = at(x0, y0, z0) * (1.0f - tx) + at(x1, y0, z0) * tx;
    const float c01 = at(x0, y0, z1) * (1.0f - tx) + at(x1, y0, z1) * tx;
    const float c10 = at(x0, y1, z0) * (1.0f - tx) + at(x1, y1, z0) * tx;
    const float c11 = at(x0, y1, z1) * (1.0f - tx) + at(x1, y1, z1) * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

float decodeDenseValueAt(const VbtFile& file,
                         uint32_t payloadByteBase,
                         int coarseKeep,
                         int denseResolution,
                         int denseBits,
                         uint32_t x,
                         uint32_t y,
                         uint32_t z,
                         uint32_t t)
{
    const int valuesPerFrame = denseResolution * denseResolution * denseResolution;
    const int frameBytes = (valuesPerFrame * denseBits + 7) / 8;
    const uint32_t scalesBase = payloadByteBase + static_cast<uint32_t>(4 + 64 * (coarseKeep + 1));
    const uint32_t dataBase = scalesBase + static_cast<uint32_t>(file.header.frames * 2u);
    const float scale = halfToFloat(readPayloadU16(file, scalesBase + t * 2u));

    const float fx = (static_cast<float>(x) / 7.0f) * static_cast<float>(denseResolution - 1);
    const float fy = (static_cast<float>(y) / 7.0f) * static_cast<float>(denseResolution - 1);
    const float fz = (static_cast<float>(z) / 7.0f) * static_cast<float>(denseResolution - 1);
    const int x0 = std::min(denseResolution - 2, std::max(0, static_cast<int>(std::floor(fx))));
    const int y0 = std::min(denseResolution - 2, std::max(0, static_cast<int>(std::floor(fy))));
    const int z0 = std::min(denseResolution - 2, std::max(0, static_cast<int>(std::floor(fz))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);

    auto sample = [&](int cx, int cy, int cz) -> float {
        const int controlIndex = (cz * denseResolution + cy) * denseResolution + cx;
        const uint32_t frameBase = dataBase + static_cast<uint32_t>(t * frameBytes);
        const int bitOffset = controlIndex * denseBits;
        const uint32_t byteIndex = frameBase + static_cast<uint32_t>(bitOffset / 8);
        const int shift = bitOffset % 8;
        uint32_t packed = static_cast<uint32_t>(readPayloadByte(file, byteIndex));
        if (shift + denseBits > 8) {
            packed |= static_cast<uint32_t>(readPayloadByte(file, byteIndex + 1u)) << 8u;
        }
        const uint32_t mask = (1u << denseBits) - 1u;
        return decodeBfp((packed >> shift) & mask, scale, denseBits);
    };

    const float c00 = sample(x0, y0, z0) * (1.0f - tx) + sample(x1, y0, z0) * tx;
    const float c01 = sample(x0, y0, z1) * (1.0f - tx) + sample(x1, y0, z1) * tx;
    const float c10 = sample(x0, y1, z0) * (1.0f - tx) + sample(x1, y1, z0) * tx;
    const float c11 = sample(x0, y1, z1) * (1.0f - tx) + sample(x1, y1, z1) * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

float decodeDenseTemporalBasisAt(const VbtFile& file,
                                 uint32_t payloadByteBase,
                                 int coarseKeep,
                                 int denseResolution,
                                 int denseBits,
                                 int temporalKeep,
                                 uint32_t x,
                                 uint32_t y,
                                 uint32_t z,
                                 uint32_t t)
{
    if (temporalKeep <= 0 || denseBits <= 0) return 0.0f;
    const int valuesPerBasis = denseResolution * denseResolution * denseResolution;
    const int basisBytes = (valuesPerBasis * denseBits + 7) / 8;
    const uint32_t scalesBase = payloadByteBase + static_cast<uint32_t>(4 + 64 * (coarseKeep + 1));
    const uint32_t dataBase = scalesBase + static_cast<uint32_t>(temporalKeep * 2);

    const float fx = (static_cast<float>(x) / 7.0f) * static_cast<float>(denseResolution - 1);
    const float fy = (static_cast<float>(y) / 7.0f) * static_cast<float>(denseResolution - 1);
    const float fz = (static_cast<float>(z) / 7.0f) * static_cast<float>(denseResolution - 1);
    const int x0 = std::min(denseResolution - 2, std::max(0, static_cast<int>(std::floor(fx))));
    const int y0 = std::min(denseResolution - 2, std::max(0, static_cast<int>(std::floor(fy))));
    const int z0 = std::min(denseResolution - 2, std::max(0, static_cast<int>(std::floor(fz))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);

    auto sampleBasis = [&](int basisIdx, int cx, int cy, int cz) -> float {
        const int controlIndex = (cz * denseResolution + cy) * denseResolution + cx;
        const uint32_t basisBase = dataBase + static_cast<uint32_t>(basisIdx * basisBytes);
        const int bitOffset = controlIndex * denseBits;
        const uint32_t byteIndex = basisBase + static_cast<uint32_t>(bitOffset / 8);
        const int shift = bitOffset % 8;
        uint32_t packed = static_cast<uint32_t>(readPayloadByte(file, byteIndex));
        if (shift + denseBits > 8) {
            packed |= static_cast<uint32_t>(readPayloadByte(file, byteIndex + 1u)) << 8u;
        }
        const uint32_t mask = (1u << denseBits) - 1u;
        const float scale = halfToFloat(readPayloadU16(file, scalesBase + static_cast<uint32_t>(basisIdx * 2)));
        return decodeBfp((packed >> shift) & mask, scale, denseBits);
    };

    float residual = 0.0f;
    for (int k = 0; k < temporalKeep; ++k) {
        const float c00 = sampleBasis(k, x0, y0, z0) * (1.0f - tx) + sampleBasis(k, x1, y0, z0) * tx;
        const float c01 = sampleBasis(k, x0, y0, z1) * (1.0f - tx) + sampleBasis(k, x1, y0, z1) * tx;
        const float c10 = sampleBasis(k, x0, y1, z0) * (1.0f - tx) + sampleBasis(k, x1, y1, z0) * tx;
        const float c11 = sampleBasis(k, x0, y1, z1) * (1.0f - tx) + sampleBasis(k, x1, y1, z1) * tx;
        const float c0 = c00 * (1.0f - ty) + c10 * ty;
        const float c1 = c01 * (1.0f - ty) + c11 * ty;
        residual += (c0 * (1.0f - tz) + c1 * tz) * dctBasis(file.header.frames, static_cast<int>(t), k);
    }
    return residual;
}

float decodeSparseResidualAt(const VbtFile& file,
                             uint32_t payloadByteBase,
                             const ScientificHeaderV2& decoded,
                             int coarseKeep,
                             uint32_t localIndexValue,
                             uint32_t t)
{
    const uint32_t actualCount = decoded.packedEventCount;
    if (actualCount == 0u) return 0.0f;

    const uint32_t tierCapacity =
        (decoded.tierId == 0u) ? 64u :
        (decoded.tierId == 1u) ? 256u :
        (decoded.tierId == 2u) ? 768u : 1024u;
    const uint32_t framesPerBin = decoded.framesPerBinMinus1 + 1u;
    const uint32_t framesPerBinSafe = std::max<uint32_t>(1u, framesPerBin);
    const uint32_t binIdx = std::min<uint32_t>(15u, t / framesPerBinSafe);
    const uint8_t localTimeInBin = static_cast<uint8_t>(t - binIdx * framesPerBinSafe);
    const uint32_t sparseBase = payloadByteBase + 4u + static_cast<uint32_t>(64 * (coarseKeep + 1));

    float eventScale = 0.0f;
    const uint32_t eventScaleWord = file.payloadWords.at(sparseBase >> 2u);
    std::memcpy(&eventScale, &eventScaleWord, sizeof(float));
    if (!(eventScale > 0.0f)) return 0.0f;

    const uint32_t maskWordOffset = sparseBase + 4u + ((localIndexValue >> 5u) * 4u);
    const uint32_t maskWord = file.payloadWords.at(maskWordOffset >> 2u);
    const uint32_t maskBit = 1u << (localIndexValue & 31u);
    if ((maskWord & maskBit) == 0u) return 0.0f;

    const uint32_t binsBase = sparseBase + 4u + 64u;
    const uint16_t start = readPayloadU16(file, binsBase + binIdx * 2u);
    const uint16_t end = (binIdx + 1u < 16u)
        ? readPayloadU16(file, binsBase + (binIdx + 1u) * 2u)
        : static_cast<uint16_t>(actualCount);
    if (start >= end) return 0.0f;

    const uint32_t coordsBase = binsBase + 32u;
    const uint32_t residualBase = coordsBase + tierCapacity * 2u;
    const uint16_t targetKey = static_cast<uint16_t>((static_cast<uint16_t>(localTimeInBin) << 9u) |
                                                     static_cast<uint16_t>(localIndexValue & 0x1FFu));
    int left = static_cast<int>(start);
    int right = static_cast<int>(end) - 1;
    while (left <= right) {
        const int mid = (left + right) >> 1;
        const uint16_t key = readPayloadU16(file, coordsBase + static_cast<uint32_t>(mid) * 2u);
        if (key == targetKey) {
            const uint32_t byteIndex = residualBase + static_cast<uint32_t>(mid >> 1);
            const uint8_t byte = readPayloadByte(file, byteIndex);
            const uint8_t nibble = ((mid & 1) == 0)
                ? static_cast<uint8_t>(byte & 0x0Fu)
                : static_cast<uint8_t>((byte >> 4u) & 0x0Fu);
            int8_t q = static_cast<int8_t>(nibble);
            if ((q & 0x08) != 0) q = static_cast<int8_t>(q - 16);
            return static_cast<float>(q) * eventScale;
        }
        if (key < targetKey) left = mid + 1;
        else right = mid - 1;
    }
    return 0.0f;
}

} // namespace

float decodeScientificValueAtCpu(const VbtFile& file, const Query4D& query)
{
    const uint32_t leafIndex = query.leafIndex;
    const uint32_t packedHeader = readPackedHeaderWord(file, leafIndex);
    const ScientificHeaderV2 decoded = decodeScientificHeaderV2(packedHeader);
    const uint32_t payloadByteBase = file.offsetsWords.at(leafIndex) * 4u;

    const uint32_t leafSize = std::max<uint32_t>(1u, file.header.leafSize);
    const uint32_t lx = query.x % leafSize;
    const uint32_t ly = query.y % leafSize;
    const uint32_t lz = query.z % leafSize;
    const uint32_t lidx = (lz * leafSize + ly) * leafSize + lx;

    const int coarseKeep = static_cast<int>(decoded.coarseKeepMinus1 + 1u);

    std::array<float, 64> coarseCtrl{};
    for (int i = 0; i < 64; ++i) {
        coarseCtrl[static_cast<size_t>(i)] =
            decodeCoarseControlAt(file, payloadByteBase, coarseKeep, i, static_cast<int>(query.t));
    }
    const float coarse = sampleCoarseAt(coarseCtrl, lx, ly, lz);

    float residual = 0.0f;
    switch (decoded.mode) {
    case ScientificMode::CoarseOnly:
        residual = 0.0f;
        break;
    case ScientificMode::SparseEvents:
        residual = decodeSparseResidualAt(file, payloadByteBase, decoded, coarseKeep, lidx, query.t);
        break;
    case ScientificMode::DenseGrid3:
        residual = (decoded.denseSubtype == kScientificDenseSubtypeTemporalBasis)
            ? decodeDenseTemporalBasisAt(file,
                                         payloadByteBase,
                                         coarseKeep,
                                         3,
                                         static_cast<int>(decoded.denseQuantBits),
                                         static_cast<int>(decoded.denseTemporalKeepMinus1 + 1u),
                                         lx, ly, lz, query.t)
            : decodeDenseValueAt(file, payloadByteBase, coarseKeep, 3, static_cast<int>(decoded.denseQuantBits), lx, ly, lz, query.t);
        break;
    case ScientificMode::DenseGrid4:
        residual = (decoded.denseSubtype == kScientificDenseSubtypeTemporalBasis)
            ? decodeDenseTemporalBasisAt(file,
                                         payloadByteBase,
                                         coarseKeep,
                                         4,
                                         static_cast<int>(decoded.denseQuantBits),
                                         static_cast<int>(decoded.denseTemporalKeepMinus1 + 1u),
                                         lx, ly, lz, query.t)
            : decodeDenseValueAt(file, payloadByteBase, coarseKeep, 4, static_cast<int>(decoded.denseQuantBits), lx, ly, lz, query.t);
        break;
    default:
        residual = 0.0f;
        break;
    }
    return coarse + residual;
}

std::vector<float> reconstructScientificFrameCpu(const VbtFile& file, uint32_t frameIndex)
{
    const uint64_t voxelCount = static_cast<uint64_t>(file.header.width) *
                                static_cast<uint64_t>(file.header.height) *
                                static_cast<uint64_t>(file.header.depth);
    if (voxelCount > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("Frame too large to reconstruct into memory.");
    }
    std::vector<float> frame(static_cast<size_t>(voxelCount), 0.0f);
    for (uint32_t z = 0; z < file.header.depth; ++z) {
        for (uint32_t y = 0; y < file.header.height; ++y) {
            for (uint32_t x = 0; x < file.header.width; ++x) {
                Query4D q{};
                q.x = x;
                q.y = y;
                q.z = z;
                q.t = frameIndex;
                q.leafIndex = leafIndexForVoxel(file.header, x, y, z);
                const size_t idx = static_cast<size_t>((static_cast<uint64_t>(z) * file.header.height + y) * file.header.width + x);
                frame[idx] = decodeScientificValueAtCpu(file, q);
            }
        }
    }
    return frame;
}

} // namespace vbt::render
