#include "vbt_file.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace vbt::render {

namespace {

template <typename T>
bool readValue(std::ifstream& in, T& value)
{
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(in);
}

} // namespace

bool loadVbtFile(const std::filesystem::path& path, VbtFile& outFile, std::string& error)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "Failed to open file: " + path.string();
        return false;
    }

    if (!readValue(in, outFile.header.magic) ||
        !readValue(in, outFile.header.version) ||
        !readValue(in, outFile.header.width) ||
        !readValue(in, outFile.header.height) ||
        !readValue(in, outFile.header.depth) ||
        !readValue(in, outFile.header.frames) ||
        !readValue(in, outFile.header.leafSize) ||
        !readValue(in, outFile.header.coarseResolution) ||
        !readValue(in, outFile.header.maxCoarseKeep) ||
        !readValue(in, outFile.header.leafCount) ||
        !readValue(in, outFile.header.profileType) ||
        !readValue(in, outFile.header.coarseAcScaleCount)) {
        error = "Failed to read VBTPACK header.";
        return false;
    }

    if (std::strncmp(outFile.header.magic, "VBTPACK4", 8) != 0) {
        error = "Unsupported magic. Expected VBTPACK4.";
        return false;
    }

    outFile.coarseAcScales.resize(outFile.header.coarseAcScaleCount);
    if (!outFile.coarseAcScales.empty()) {
        in.read(reinterpret_cast<char*>(outFile.coarseAcScales.data()),
                static_cast<std::streamsize>(sizeof(float) * outFile.coarseAcScales.size()));
        if (!in) {
            error = "Failed to read coarse AC scales.";
            return false;
        }
    }

    outFile.offsetsWords.resize(static_cast<size_t>(outFile.header.leafCount) + 1u);
    in.read(reinterpret_cast<char*>(outFile.offsetsWords.data()),
            static_cast<std::streamsize>(sizeof(uint32_t) * outFile.offsetsWords.size()));
    if (!in) {
        error = "Failed to read offset table.";
        return false;
    }

    const uint32_t payloadWordCount = outFile.offsetsWords.back();
    outFile.payloadWords.resize(payloadWordCount);
    if (payloadWordCount > 0) {
        in.read(reinterpret_cast<char*>(outFile.payloadWords.data()),
                static_cast<std::streamsize>(sizeof(uint32_t) * outFile.payloadWords.size()));
        if (!in) {
            error = "Failed to read payload pool.";
            return false;
        }
    }

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
    const uint32_t leafSize = std::max<uint32_t>(1u, header.leafSize);
    const uint32_t leafCountX = (header.width + leafSize - 1u) / leafSize;
    const uint32_t leafCountY = (header.height + leafSize - 1u) / leafSize;
    const uint32_t bx = std::min<uint32_t>(leafCountX - 1u, x / leafSize);
    const uint32_t by = std::min<uint32_t>(leafCountY - 1u, y / leafSize);
    const uint32_t bz = std::min<uint32_t>((header.depth + leafSize - 1u) / leafSize - 1u, z / leafSize);
    return (bz * leafCountY + by) * leafCountX + bx;
}

} // namespace vbt::render
