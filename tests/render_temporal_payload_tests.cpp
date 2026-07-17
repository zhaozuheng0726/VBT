#include "render_temporal_payload.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

template <class Fn>
void expectFailure(Fn&& fn, const std::string& expectedText)
{
    try {
        fn();
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find(expectedText) != std::string::npos,
                "Unexpected error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected operation to fail: " + expectedText);
}

vbt::RenderTemporalShellOccupancySection makeOccupancy()
{
    vbt::RenderTemporalShellOccupancySection occupancy{};
    occupancy.shellMask[0] = 1ull;
    occupancy.shellMask[2] = 1ull << 5u;
    occupancy.groupPrefix = {0u, 1u, 1u, 2u, 2u, 2u, 2u, 2u};
    return occupancy;
}

std::vector<uint8_t> makeValidMode2Leaf()
{
    vbt::RenderTemporalPackedHeaderFields header;
    header.mode = vbt::TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT;
    header.coarseCodec = vbt::RenderTemporalSequenceCodec::DP_KEYFRAME_FP16;
    header.fineCodec = vbt::RenderTemporalSequenceCodec::DP_KEYFRAME_FP16;
    header.coarseResolution = 1u;
    header.fineResolution = 0u;

    const auto prefix = vbt::makeRenderTemporalLeafPrefix(header.mode, 0u, 2u);
    const auto layout = vbt::computeRenderTemporalLeafLayout(header, prefix, true);
    std::vector<uint8_t> coarseBytes(layout.coarse.totalBytes, 0u);
    const auto occupancyBytes = vbt::packRenderTemporalShellOccupancySection(makeOccupancy());

    const uint32_t activeCount = 2u;
    const uint32_t descriptorBytes = activeCount * vbt::kRenderTemporalControlDescriptorBytes;
    const uint32_t binIndexBytes = activeCount * vbt::kRenderTemporalShellPackedTimeBinCount;
    const uint32_t keyframeBytes = 8u;
    std::vector<uint8_t> residualBytes(descriptorBytes + binIndexBytes + keyframeBytes, 0u);

    return vbt::buildRenderTemporalPackedLeafBytes(
        header, prefix, coarseBytes, occupancyBytes, residualBytes, true);
}

} // namespace

int main()
{
    require(vbt::renderTemporalLeafVoxelIndex(0u, 0u, 1u) == 64u,
            "Mode2 voxel indexing must retain the fixed 8x8x8 leaf stride");
    require(vbt::renderTemporalLeafVoxelIndex(2u, 2u, 1u) == 82u,
            "Mode2 boundary-leaf voxel indexing must not use cropped extents");

    const auto validBytes = makeValidMode2Leaf();
    const auto validView =
        vbt::parseRenderTemporalPackedLeaf(validBytes.data(), validBytes.size(), true);
    require(validView.header.mode == vbt::TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT,
            "Mode2 header was not preserved");
    require(validView.shellActiveVoxelCount == 2u, "Mode2 active voxel count mismatch");
    require(validView.shellResidualOffset + validView.shellResidualBytes == validBytes.size(),
            "Mode2 residual bounds mismatch");

    auto badPrefix = validBytes;
    const uint16_t corruptPrefix = 0u;
    const size_t groupOnePrefixOffset =
        validView.layout.fine.descriptorOffset + 64u + sizeof(uint16_t);
    std::memcpy(badPrefix.data() + groupOnePrefixOffset,
                &corruptPrefix,
                sizeof(corruptPrefix));
    expectFailure(
        [&] { vbt::parseRenderTemporalPackedLeaf(badPrefix.data(), badPrefix.size(), true); },
        "group prefix");

    auto truncatedResidual = validBytes;
    truncatedResidual.resize(truncatedResidual.size() - sizeof(uint32_t));
    expectFailure(
        [&] {
            vbt::parseRenderTemporalPackedLeaf(
                truncatedResidual.data(), truncatedResidual.size(), true);
        },
        "residual stream");

    auto trailingBytes = validBytes;
    trailingBytes.resize(trailingBytes.size() + sizeof(uint32_t), 0u);
    expectFailure(
        [&] { vbt::parseRenderTemporalPackedLeaf(trailingBytes.data(), trailingBytes.size(), true); },
        "trailing bytes");

    auto noActiveVoxels = validBytes;
    std::memset(noActiveVoxels.data() + validView.layout.fine.descriptorOffset, 0, 80u);
    expectFailure(
        [&] {
            vbt::parseRenderTemporalPackedLeaf(
                noActiveVoxels.data(), noActiveVoxels.size(), true);
        },
        "no active voxels");

    std::cout << "vbt_render_temporal_payload_tests: PASS\n";
    return 0;
}
