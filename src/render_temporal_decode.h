#pragma once

#include "render_temporal_payload.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vbt {

struct RenderTemporalControlDescriptor {
    uint16_t keyStart = 0;
    uint8_t keyCount = 0;
    uint8_t reserved = 0;
};

struct RenderTemporalControlKeyframe {
    uint16_t frame = 0;
    float value = 0.0f;
};

static_assert(sizeof(RenderTemporalControlDescriptor) == 4,
              "RenderTemporalControlDescriptor must stay 4 bytes");

struct RenderTemporalControlStreamData {
    RenderTemporalSequenceCodec codec = RenderTemporalSequenceCodec::DP_KEYFRAME_FP16;
    uint16_t controlCount = 0;
    uint8_t timeBinCount = 0;
    uint8_t maxBinLocalKeys = 0;
    std::vector<RenderTemporalControlDescriptor> descriptors;
    std::vector<uint8_t> binIndex;
    std::vector<RenderTemporalControlKeyframe> keyframes;
};

uint16_t renderTemporalFloatToHalfBits(float value);
float renderTemporalHalfBitsToFloat(uint16_t bits);

bool validateRenderTemporalControlStreamData(const RenderTemporalControlStreamData& stream);
std::vector<uint8_t> packRenderTemporalControlStream(const RenderTemporalControlStreamData& stream,
                                                     bool alignTo4Bytes);

RenderTemporalControlStreamData unpackRenderTemporalControlStream(const uint8_t* streamBase,
                                                                 RenderTemporalSequenceCodec codec,
                                                                 uint16_t controlCount,
                                                                 uint8_t timeBinCount,
                                                                 uint8_t maxBinLocalKeys,
                                                                 uint32_t keyframeCount,
                                                                 uint32_t streamBytes,
                                                                 bool alignTo4Bytes);

float decodeRenderTemporalControlValue(const uint8_t* streamBase,
                                       RenderTemporalSequenceCodec codec,
                                       uint16_t controlCount,
                                       uint8_t timeBinCount,
                                       uint8_t maxBinLocalKeys,
                                       uint32_t keyframeCount,
                                       uint32_t streamBytes,
                                       bool alignTo4Bytes,
                                       int totalFrames,
                                       uint16_t controlIndex,
                                       int frameIndex);

float decodeRenderTemporalLeafControlValue(const uint8_t* leafBase,
                                           const RenderTemporalPackedHeaderFields& header,
                                           const RenderTemporalPackedLeafPrefix& prefix,
                                           const RenderTemporalPackedLeafLayout& layout,
                                           bool useFineStream,
                                           bool alignTo4Bytes,
                                           int totalFrames,
                                           uint16_t controlIndex,
                                           int frameIndex);

// Mode2 = TemporalShellVoxel decode entry.
// Reads the shell occupancy section right after the coarse stream and decodes
// the shell-active voxel residual when the voxel is masked active.
float decodeRenderTemporalLeafShellVoxelValue(const uint8_t* leafBase,
                                              const RenderTemporalPackedLeafView& leafView,
                                              bool alignTo4Bytes,
                                              int totalFrames,
                                              std::uint16_t voxelIndex,
                                              int frameIndex);

} // namespace vbt
