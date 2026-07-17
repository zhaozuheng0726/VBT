#include "vbt_file.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>

namespace vbt::render {

namespace {

constexpr uint64_t kHeaderBytes = 8u + 11u * sizeof(uint32_t);
constexpr uint32_t kPack4Version = 4u;
constexpr uint32_t kPack4LeafSize = 8u;
constexpr uint32_t kMaxCoarseResolution = 8u;
constexpr uint32_t kMaxCoarseScaleCount = 64u;

template <typename T>
bool readValue(std::ifstream& in, T& value)
{
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(in);
}

bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) return false;
    result = lhs + rhs;
    return true;
}

bool checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0u && rhs > std::numeric_limits<uint64_t>::max() / lhs) return false;
    result = lhs * rhs;
    return true;
}

bool validateHeader(const VbtFileHeader& header, uint64_t fileBytes, std::string& error)
{
    if (std::strncmp(header.magic, "VBTPACK4", 8) != 0) {
        error = "Unsupported magic. Expected VBTPACK4.";
        return false;
    }
    if (header.version != kPack4Version) {
        error = "Unsupported VBTPACK4 version: " + std::to_string(header.version);
        return false;
    }
    if (header.width == 0u || header.height == 0u || header.depth == 0u || header.frames == 0u) {
        error = "VBTPACK4 dimensions and frame count must be non-zero.";
        return false;
    }
    if (header.leafSize != kPack4LeafSize) {
        error = "VBTPACK4 v4 requires leafSize=8.";
        return false;
    }
    if (header.coarseResolution == 0u || header.coarseResolution > kMaxCoarseResolution) {
        error = "Invalid VBTPACK4 coarse resolution.";
        return false;
    }
    if (header.profileType > 2u) {
        error = "Invalid VBTPACK4 profile type.";
        return false;
    }
    if (header.maxCoarseKeep > header.frames || header.coarseAcScaleCount > kMaxCoarseScaleCount ||
        header.coarseAcScaleCount > header.maxCoarseKeep) {
        error = "Invalid VBTPACK4 coarse temporal metadata.";
        return false;
    }

    const uint64_t leafCountX =
        (static_cast<uint64_t>(header.width) + header.leafSize - 1u) / header.leafSize;
    const uint64_t leafCountY =
        (static_cast<uint64_t>(header.height) + header.leafSize - 1u) / header.leafSize;
    const uint64_t leafCountZ =
        (static_cast<uint64_t>(header.depth) + header.leafSize - 1u) / header.leafSize;
    uint64_t expectedLeafCount = 0u;
    uint64_t leafCountXY = 0u;
    if (!checkedMultiply(leafCountX, leafCountY, leafCountXY) ||
        !checkedMultiply(leafCountXY, leafCountZ, expectedLeafCount) ||
        expectedLeafCount > std::numeric_limits<uint32_t>::max()) {
        error = "VBTPACK4 leaf count overflows the v4 header.";
        return false;
    }
    if (header.leafCount != expectedLeafCount) {
        error = "VBTPACK4 leafCount does not match dimensions and leafSize.";
        return false;
    }

    uint64_t scaleBytes = 0u;
    uint64_t offsetBytes = 0u;
    uint64_t minimumBytes = kHeaderBytes;
    if (!checkedMultiply(header.coarseAcScaleCount, sizeof(float), scaleBytes) ||
        !checkedMultiply(static_cast<uint64_t>(header.leafCount) + 1u, sizeof(uint32_t), offsetBytes) ||
        !checkedAdd(minimumBytes, scaleBytes, minimumBytes) ||
        !checkedAdd(minimumBytes, offsetBytes, minimumBytes)) {
        error = "VBTPACK4 table sizes overflow.";
        return false;
    }
    if (minimumBytes > fileBytes) {
        error = "VBTPACK4 file is truncated before its offset table.";
        return false;
    }
    return true;
}

} // namespace

bool loadVbtFile(const std::filesystem::path& path, VbtFile& outFile, std::string& error)
{
    error.clear();
    std::error_code sizeError;
    const uintmax_t fileSizeValue = std::filesystem::file_size(path, sizeError);
    if (sizeError) {
        error = "Failed to query file size: " + path.string();
        return false;
    }
    if (fileSizeValue < kHeaderBytes || fileSizeValue > std::numeric_limits<uint64_t>::max()) {
        error = "Invalid VBTPACK4 file size.";
        return false;
    }
    const uint64_t fileBytes = static_cast<uint64_t>(fileSizeValue);

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "Failed to open file: " + path.string();
        return false;
    }

    VbtFile loaded;
    if (!readValue(in, loaded.header.magic) ||
        !readValue(in, loaded.header.version) ||
        !readValue(in, loaded.header.width) ||
        !readValue(in, loaded.header.height) ||
        !readValue(in, loaded.header.depth) ||
        !readValue(in, loaded.header.frames) ||
        !readValue(in, loaded.header.leafSize) ||
        !readValue(in, loaded.header.coarseResolution) ||
        !readValue(in, loaded.header.maxCoarseKeep) ||
        !readValue(in, loaded.header.leafCount) ||
        !readValue(in, loaded.header.profileType) ||
        !readValue(in, loaded.header.coarseAcScaleCount)) {
        error = "Failed to read VBTPACK header.";
        return false;
    }

    if (!validateHeader(loaded.header, fileBytes, error)) return false;

    loaded.coarseAcScales.resize(loaded.header.coarseAcScaleCount);
    if (!loaded.coarseAcScales.empty()) {
        in.read(reinterpret_cast<char*>(loaded.coarseAcScales.data()),
                static_cast<std::streamsize>(sizeof(float) * loaded.coarseAcScales.size()));
        if (!in) {
            error = "Failed to read coarse AC scales.";
            return false;
        }
    }

    loaded.offsetsWords.resize(static_cast<size_t>(loaded.header.leafCount) + 1u);
    in.read(reinterpret_cast<char*>(loaded.offsetsWords.data()),
            static_cast<std::streamsize>(sizeof(uint32_t) * loaded.offsetsWords.size()));
    if (!in) {
        error = "Failed to read offset table.";
        return false;
    }
    if (loaded.offsetsWords.front() != 0u) {
        error = "VBTPACK4 offset table must begin at word zero.";
        return false;
    }
    for (size_t i = 0; i < static_cast<size_t>(loaded.header.leafCount); ++i) {
        if (loaded.offsetsWords[i] >= loaded.offsetsWords[i + 1u]) {
            error = "VBTPACK4 offsets must be strictly increasing for every leaf.";
            return false;
        }
    }

    const uint32_t payloadWordCount = loaded.offsetsWords.back();
    uint64_t scaleBytes = 0u;
    uint64_t offsetBytes = 0u;
    uint64_t payloadBytes = 0u;
    uint64_t expectedBytes = kHeaderBytes;
    if (!checkedMultiply(loaded.header.coarseAcScaleCount, sizeof(float), scaleBytes) ||
        !checkedMultiply(loaded.offsetsWords.size(), sizeof(uint32_t), offsetBytes) ||
        !checkedMultiply(payloadWordCount, sizeof(uint32_t), payloadBytes) ||
        !checkedAdd(expectedBytes, scaleBytes, expectedBytes) ||
        !checkedAdd(expectedBytes, offsetBytes, expectedBytes) ||
        !checkedAdd(expectedBytes, payloadBytes, expectedBytes)) {
        error = "VBTPACK4 payload size overflows.";
        return false;
    }
    if (expectedBytes != fileBytes) {
        error = expectedBytes > fileBytes
            ? "VBTPACK4 payload is truncated."
            : "VBTPACK4 file contains trailing bytes.";
        return false;
    }

    loaded.payloadWords.resize(payloadWordCount);
    if (payloadWordCount > 0) {
        in.read(reinterpret_cast<char*>(loaded.payloadWords.data()),
                static_cast<std::streamsize>(sizeof(uint32_t) * loaded.payloadWords.size()));
        if (!in) {
            error = "Failed to read payload pool.";
            return false;
        }
    }

    outFile = std::move(loaded);
    return true;
}

uint32_t readPackedHeaderWord(const VbtFile& file, uint32_t leafIndex)
{
    if (leafIndex >= file.header.leafCount) return 0u;
    const uint32_t wordOffset = file.offsetsWords[leafIndex];
    if (wordOffset >= file.payloadWords.size()) return 0u;
    return file.payloadWords[wordOffset];
}

ScientificHeaderV2 decodeScientificHeaderV2(uint32_t packedHeader)
{
    ScientificHeaderV2 decoded{};
    decoded.mode = static_cast<ScientificMode>(packedHeader & 0x3u);
    switch (decoded.mode) {
    case ScientificMode::CoarseOnly:
        decoded.coarseKeepMinus1 = (packedHeader >> 2u) & 0x3Fu;
        decoded.coarseQuantClass = (packedHeader >> 8u) & 0x7u;
        break;
    case ScientificMode::SparseEvents:
        decoded.coarseKeepMinus1 = (packedHeader >> 2u) & 0x3Fu;
        decoded.tierId = (packedHeader >> 8u) & 0x3u;
        decoded.framesPerBinMinus1 = (packedHeader >> 10u) & 0xFu;
        decoded.eventQuantCode = (packedHeader >> 14u) & 0x3u;
        decoded.packedEventCount = (packedHeader >> 16u) & 0x0FFFu;
        decoded.sparseSubtype = (packedHeader >> 28u) & 0xFu;
        break;
    case ScientificMode::DenseGrid3:
    case ScientificMode::DenseGrid4:
        decoded.coarseKeepMinus1 = (packedHeader >> 2u) & 0x3Fu;
        decoded.denseQuantBits = (packedHeader >> 8u) & 0xFu;
        decoded.denseSubtype = (packedHeader >> 12u) & 0x7u;
        decoded.denseTemporalKeepMinus1 = (packedHeader >> 15u) & 0xFu;
        break;
    default:
        break;
    }
    return decoded;
}

uint32_t leafIndexForVoxel(const VbtFileHeader& header, uint32_t x, uint32_t y, uint32_t z)
{
    if (header.width == 0u || header.height == 0u || header.depth == 0u || header.leafSize == 0u) {
        return 0u;
    }
    const uint32_t leafSize = header.leafSize;
    const uint32_t leafCountX = (header.width + leafSize - 1u) / leafSize;
    const uint32_t leafCountY = (header.height + leafSize - 1u) / leafSize;
    const uint32_t bx = std::min<uint32_t>(leafCountX - 1u, x / leafSize);
    const uint32_t by = std::min<uint32_t>(leafCountY - 1u, y / leafSize);
    const uint32_t bz = std::min<uint32_t>((header.depth + leafSize - 1u) / leafSize - 1u, z / leafSize);
    return (bz * leafCountY + by) * leafCountX + bx;
}

} // namespace vbt::render
