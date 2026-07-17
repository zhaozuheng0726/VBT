#pragma once

#include "temporal_first_encoder.h"

#include <array>
#include <cstdint>
#include <vector>

namespace vbt {

enum class RenderTemporalSequenceCodec : uint8_t {
    DP_KEYFRAME_FP16 = 0,
    DP_KEYFRAME_FP32 = 1,
};

// Render temporal-first keeps a flat, GPU-friendly block layout:
// [packedHeader][fixed prefix][coarse descriptors][coarse bin index]
// [coarse keyframes][fine descriptors][fine bin index][fine keyframes]
// The global offset table still locates the block; the header/prefix never
// store payload offsets.
struct RenderTemporalPackedHeaderFields {
    TemporalFirstPackedMode mode = TemporalFirstPackedMode::EMPTY;
    RenderTemporalSequenceCodec coarseCodec = RenderTemporalSequenceCodec::DP_KEYFRAME_FP16;
    RenderTemporalSequenceCodec fineCodec = RenderTemporalSequenceCodec::DP_KEYFRAME_FP16;
    uint8_t coarseResolution = 4;
    uint8_t fineResolution = 0;
};

// Fixed-size per-block prefix. After Header V2, route/meta/config all live in
// the 32-bit header; the prefix only carries the variable keyframe counts that
// are still genuinely per-leaf.
struct RenderTemporalPackedLeafPrefix {
    uint16_t coarseKeyframeCount = 0;
    uint16_t fineKeyframeCount = 0;
};

static_assert(sizeof(RenderTemporalPackedLeafPrefix) == 4,
              "RenderTemporalPackedLeafPrefix must stay 4 bytes");

struct RenderTemporalStreamLayout {
    uint32_t descriptorOffset = 0;
    uint32_t descriptorBytes = 0;
    uint32_t binIndexOffset = 0;
    uint32_t binIndexBytes = 0;
    uint32_t keyframeOffset = 0;
    uint32_t keyframeBytes = 0;
    uint32_t totalBytes = 0;
};

struct RenderTemporalPackedLeafLayout {
    uint32_t headerBytes = 4;
    uint32_t prefixOffset = 4;
    uint32_t prefixBytes = sizeof(RenderTemporalPackedLeafPrefix);
    uint32_t coarseOffset = 0;
    RenderTemporalStreamLayout coarse;
    uint32_t fineOffset = 0;
    RenderTemporalStreamLayout fine;
    uint32_t totalBytes = 0;
};

struct RenderTemporalPackedLeafView {
    RenderTemporalPackedHeaderFields header;
    RenderTemporalPackedLeafPrefix prefix;
    RenderTemporalPackedLeafLayout layout;
    uint16_t shellActiveVoxelCount = 0;
    uint32_t shellResidualOffset = 0;
    uint32_t shellResidualBytes = 0;
};

constexpr uint32_t kRenderTemporalHeaderBytes = 4;
constexpr uint32_t kRenderTemporalControlDescriptorBytes = 4;
constexpr uint8_t kRenderTemporalPackedTimeBinCount = 8;
constexpr uint8_t kRenderTemporalShellPackedTimeBinCount = 4;
constexpr uint16_t kRenderTemporalLeafSize = 8;

constexpr uint16_t renderTemporalLeafVoxelIndex(uint16_t lx,
                                                uint16_t ly,
                                                uint16_t lz,
                                                uint16_t leafSize = kRenderTemporalLeafSize)
{
    return static_cast<uint16_t>((lz * leafSize + ly) * leafSize + lx);
}

// Shell occupancy section for Mode2 = TemporalShellVoxel (liquid shell mode).
// Doc: VBT/实现策略/liquid_shell_packed_mode_design_2026-04-08.md section 7.2.
// Fixed 80 bytes: 64-byte shell mask + 16-byte group prefix table.
struct RenderTemporalShellOccupancySection {
    std::array<std::uint64_t, 8> shellMask{};
    std::array<std::uint16_t, 8> groupPrefix{};
};

static_assert(sizeof(RenderTemporalShellOccupancySection) == 80,
              "RenderTemporalShellOccupancySection must stay 80 bytes");

struct RenderTemporalShellSectionLayout {
    uint32_t occupancyOffset = 0;
    uint32_t occupancyBytes = 0;
    uint32_t residualOffset = 0;
    uint32_t residualBytes = 0;
    uint32_t totalBytes = 0;
};

uint32_t renderTemporalKeyframeBytes(RenderTemporalSequenceCodec codec);

uint32_t packRenderTemporalHeader(const RenderTemporalPackedHeaderFields& fields);
RenderTemporalPackedHeaderFields unpackRenderTemporalHeader(uint32_t packedHeader);

RenderTemporalPackedLeafPrefix makeRenderTemporalLeafPrefix(TemporalFirstPackedMode mode,
                                                            uint32_t coarseKeyframeCount,
                                                            uint32_t fineKeyframeCount);

bool validateRenderTemporalLeafPrefix(const RenderTemporalPackedLeafPrefix& prefix,
                                      const RenderTemporalPackedHeaderFields& header,
                                      const TemporalFirstEncoderOptions& options);

RenderTemporalPackedLeafLayout computeRenderTemporalLeafLayout(const RenderTemporalPackedHeaderFields& header,
                                                              const RenderTemporalPackedLeafPrefix& prefix,
                                                              bool alignTo4Bytes);

std::vector<uint8_t> buildRenderTemporalPackedLeafBytes(const RenderTemporalPackedHeaderFields& header,
                                                        const RenderTemporalPackedLeafPrefix& prefix,
                                                        const std::vector<uint8_t>& coarseStreamBytes,
                                                        const std::vector<uint8_t>& fineStreamBytes,
                                                        bool alignTo4Bytes);

// 6-arg version for Mode2 = TemporalShellVoxel:
// inserts shell occupancy section between coarse and shell residual streams.
std::vector<uint8_t> buildRenderTemporalPackedLeafBytes(const RenderTemporalPackedHeaderFields& header,
                                                        const RenderTemporalPackedLeafPrefix& prefix,
                                                        const std::vector<uint8_t>& coarseStreamBytes,
                                                        const std::vector<uint8_t>& shellOccupancyBytes,
                                                        const std::vector<uint8_t>& shellResidualStreamBytes,
                                                        bool alignTo4Bytes);

std::vector<uint8_t> packRenderTemporalShellOccupancySection(const RenderTemporalShellOccupancySection& section);
RenderTemporalShellOccupancySection unpackRenderTemporalShellOccupancySection(const uint8_t* data, std::uint64_t size);
std::uint16_t renderTemporalShellActiveVoxelCount(const RenderTemporalShellOccupancySection& section);

RenderTemporalPackedLeafView parseRenderTemporalPackedLeaf(const uint8_t* leafBase,
                                                           size_t leafBytes,
                                                           bool alignTo4Bytes);

} // namespace vbt
