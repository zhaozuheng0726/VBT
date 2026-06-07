#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vbt::render {

struct VbtFileHeader {
    char magic[8]{};
    uint32_t version = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
    uint32_t frames = 0;
    uint32_t leafSize = 0;
    uint32_t coarseResolution = 0;
    uint32_t maxCoarseKeep = 0;
    uint32_t leafCount = 0;
    uint32_t profileType = 0;
    uint32_t coarseAcScaleCount = 0;
};

struct VbtFile {
    VbtFileHeader header;
    std::vector<float> coarseAcScales;
    std::vector<uint32_t> offsetsWords;
    std::vector<uint32_t> payloadWords;
};

enum class ScientificMode : uint32_t {
    CoarseOnly = 0,
    SparseEvents = 1,
    DenseGrid3 = 2,
    DenseGrid4 = 3,
};

struct ScientificHeaderV2 {
    ScientificMode mode = ScientificMode::CoarseOnly;
    uint32_t coarseKeepMinus1 = 0;
    uint32_t coarseQuantClass = 0;
    uint32_t tierId = 0;
    uint32_t framesPerBinMinus1 = 0;
    uint32_t eventQuantCode = 0;
    uint32_t sparseSubtype = 0;
    uint32_t packedEventCount = 0;
    uint32_t denseQuantBits = 0;
    uint32_t denseSubtype = 0;
    uint32_t denseTemporalKeepMinus1 = 0;
};

enum : uint32_t {
    kScientificDenseSubtypeLegacyPerFrame = 0,
    kScientificDenseSubtypeTemporalBasis = 1,
    kScientificDenseSubtypeTileTemporalBasis = 2,
    kScientificDenseSubtypePatchTemporalBasis = 3,
};

bool loadVbtFile(const std::filesystem::path& path, VbtFile& outFile, std::string& error);
uint32_t readPackedHeaderWord(const VbtFile& file, uint32_t leafIndex);
ScientificHeaderV2 decodeScientificHeaderV2(uint32_t packedHeader);
uint32_t leafIndexForVoxel(const VbtFileHeader& header, uint32_t x, uint32_t y, uint32_t z);

} // namespace vbt::render
