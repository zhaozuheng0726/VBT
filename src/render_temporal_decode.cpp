#include "render_temporal_decode.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace vbt {

namespace {

struct RenderTemporalPackedKeyframeFp32 {
    uint16_t frame = 0;
    uint16_t reserved = 0;
    float value = 0.0f;
};

uint32_t alignUp4(uint32_t bytes)
{
    return (bytes + 3u) & ~3u;
}

template <typename T>
T readPod(const uint8_t* base, size_t offset, size_t limit)
{
    if (offset + sizeof(T) > limit) {
        throw std::runtime_error("render temporal decode out of bounds");
    }
    T out{};
    std::memcpy(&out, base + offset, sizeof(T));
    return out;
}

RenderTemporalControlKeyframe readPackedKeyframe(const uint8_t* keyBase,
                                                 RenderTemporalSequenceCodec codec,
                                                 uint32_t keyIndex,
                                                 uint32_t keyframeCount)
{
    if (keyIndex >= keyframeCount) {
        throw std::runtime_error("render temporal keyframe index out of range");
    }
    RenderTemporalControlKeyframe out;
    const size_t offset = static_cast<size_t>(keyIndex) * renderTemporalKeyframeBytes(codec);
    switch (codec) {
    case RenderTemporalSequenceCodec::DP_KEYFRAME_FP32: {
        const auto key = readPod<RenderTemporalPackedKeyframeFp32>(
            keyBase, offset, static_cast<size_t>(keyframeCount) * renderTemporalKeyframeBytes(codec));
        out.frame = key.frame;
        out.value = key.value;
        break;
    }
    case RenderTemporalSequenceCodec::DP_KEYFRAME_FP16:
    default: {
        const size_t limit = static_cast<size_t>(keyframeCount) * renderTemporalKeyframeBytes(codec);
        if (offset + 3 > limit) {
            throw std::runtime_error("render temporal decode out of bounds");
        }
        const uint8_t frame = keyBase[offset];
        const uint16_t valueFp16Bits =
            static_cast<uint16_t>(keyBase[offset + 1]) |
            static_cast<uint16_t>(static_cast<uint16_t>(keyBase[offset + 2]) << 8u);
        out.frame = frame;
        out.value = renderTemporalHalfBitsToFloat(valueFp16Bits);
        break;
    }
    }
    return out;
}

uint16_t readPackedKeyframeFrame(const uint8_t* keyBase,
                                 RenderTemporalSequenceCodec codec,
                                 uint32_t keyIndex)
{
    const size_t offset = static_cast<size_t>(keyIndex) * renderTemporalKeyframeBytes(codec);
    return codec == RenderTemporalSequenceCodec::DP_KEYFRAME_FP32
        ? static_cast<uint16_t>(keyBase[offset] | (static_cast<uint16_t>(keyBase[offset + 1]) << 8u))
        : keyBase[offset];
}

float decodeFromSequence(const uint8_t* keyBase,
                         RenderTemporalSequenceCodec codec,
                         uint32_t keyframeCount,
                         uint32_t keyStart,
                         uint8_t keyCount,
                         const uint8_t* localBinIndex,
                         uint8_t timeBinCount,
                         uint8_t maxBinLocalKeys,
                         int totalFrames,
                         int frameIndex)
{
    (void)maxBinLocalKeys;
    if (keyCount == 0) return 0.0f;
    const int frame = std::max(0, std::min(std::max(0, totalFrames - 1), frameIndex));
    const auto timestamp = [&](uint32_t index) {
        return static_cast<int>(readPackedKeyframeFrame(keyBase, codec, keyStart + index));
    };
    const auto value = [&](uint32_t index) {
        return readPackedKeyframe(keyBase, codec, keyStart + index, keyframeCount).value;
    };
    const uint32_t last = keyCount - 1u;
    const uint32_t bin = timeBinCount == 0 ? 0u : std::min<uint32_t>(timeBinCount - 1u,
        static_cast<uint32_t>((static_cast<int64_t>(frame) * timeBinCount) / std::max(1, totalFrames)));
    uint32_t lo = timeBinCount == 0 ? 0u : std::min<uint32_t>(localBinIndex[bin], last);
    int f0 = timestamp(lo);
    if (f0 > frame) {
        lo = 0u;
        f0 = timestamp(lo);
    }
    if (f0 >= frame || lo == last) return value(lo);

    // Bins are hints: the next predecessor is not an interpolation boundary.
    uint32_t hi = bin + 1u < timeBinCount
        ? std::min<uint32_t>(localBinIndex[bin + 1u] + 1u, last) : last;
    hi = std::max(hi, lo + 1u);
    int f1 = timestamp(hi);
    if (f1 < frame) {
        hi = last;
        f1 = timestamp(hi);
    }
    if (f1 <= frame) return value(hi);
    while (hi > lo + 1u) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        const int fm = timestamp(mid);
        if (fm < frame) { lo = mid; f0 = fm; }
        else { hi = mid; f1 = fm; }
    }
    const float v1 = value(hi);
    if (f1 == frame) return v1;
    const float v0 = value(lo);
    const float alpha = static_cast<float>(frame - f0) / static_cast<float>(f1 - f0);
    return v0 + alpha * (v1 - v0);
}

} // namespace

uint16_t renderTemporalFloatToHalfBits(float value)
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

float renderTemporalHalfBitsToFloat(uint16_t bits)
{
    const uint32_t sign = (static_cast<uint32_t>(bits & 0x8000u)) << 16;
    uint32_t exponent = (bits >> 10) & 0x1Fu;
    uint32_t mantissa = bits & 0x03FFu;
    uint32_t outBits = 0u;

    if (exponent == 0) {
        if (mantissa == 0) {
            outBits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03FFu;
            outBits = sign |
                      ((exponent + 127 - 15) << 23) |
                      (mantissa << 13);
        }
    } else if (exponent == 31) {
        outBits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        outBits = sign |
                  ((exponent + 127 - 15) << 23) |
                  (mantissa << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &outBits, sizeof(out));
    return out;
}

bool validateRenderTemporalControlStreamData(const RenderTemporalControlStreamData& stream)
{
    if (stream.controlCount == 0 || stream.timeBinCount == 0 || stream.maxBinLocalKeys == 0) {
        return false;
    }
    if (stream.descriptors.size() != stream.controlCount) return false;
    if (stream.binIndex.size() != static_cast<size_t>(stream.controlCount) * stream.timeBinCount) {
        return false;
    }
    for (const auto& desc : stream.descriptors) {
        if (static_cast<size_t>(desc.keyStart) + static_cast<size_t>(desc.keyCount) > stream.keyframes.size()) {
            return false;
        }
        if (desc.keyCount > stream.timeBinCount * stream.maxBinLocalKeys) {
            return false;
        }
        if (desc.reserved != 0u && desc.reserved > stream.maxBinLocalKeys) {
            return false;
        }
    }
    return true;
}

std::vector<uint8_t> packRenderTemporalControlStream(const RenderTemporalControlStreamData& stream,
                                                     bool alignTo4Bytes)
{
    if (!validateRenderTemporalControlStreamData(stream)) {
        throw std::runtime_error("invalid render temporal control stream");
    }

    const uint32_t descriptorBytes =
        static_cast<uint32_t>(stream.descriptors.size() * sizeof(RenderTemporalControlDescriptor));
    uint32_t binIndexBytes = static_cast<uint32_t>(stream.binIndex.size());
    uint32_t keyframeBytes =
        static_cast<uint32_t>(stream.keyframes.size()) * renderTemporalKeyframeBytes(stream.codec);
    if (alignTo4Bytes) {
        binIndexBytes = alignUp4(binIndexBytes);
        keyframeBytes = alignUp4(keyframeBytes);
    }

    std::vector<uint8_t> out(static_cast<size_t>(descriptorBytes + binIndexBytes + keyframeBytes), 0u);
    std::memcpy(out.data(), stream.descriptors.data(), descriptorBytes);
    std::memcpy(out.data() + descriptorBytes, stream.binIndex.data(), stream.binIndex.size());
    uint8_t* keyOut = out.data() + descriptorBytes + binIndexBytes;
    for (size_t i = 0; i < stream.keyframes.size(); ++i) {
        const auto& key = stream.keyframes[i];
        const size_t offset = i * renderTemporalKeyframeBytes(stream.codec);
        if (stream.codec == RenderTemporalSequenceCodec::DP_KEYFRAME_FP32) {
            RenderTemporalPackedKeyframeFp32 packed{};
            packed.frame = key.frame;
            packed.value = key.value;
            std::memcpy(keyOut + offset, &packed, sizeof(packed));
        } else {
            if (key.frame > 255u) {
                throw std::runtime_error("render temporal fp16 keyframe frame exceeds uint8 range");
            }
            const uint16_t valueFp16Bits = renderTemporalFloatToHalfBits(key.value);
            keyOut[offset] = static_cast<uint8_t>(key.frame);
            keyOut[offset + 1] = static_cast<uint8_t>(valueFp16Bits & 0xFFu);
            keyOut[offset + 2] = static_cast<uint8_t>((valueFp16Bits >> 8u) & 0xFFu);
        }
    }
    return out;
}

RenderTemporalControlStreamData unpackRenderTemporalControlStream(const uint8_t* streamBase,
                                                                 RenderTemporalSequenceCodec codec,
                                                                 uint16_t controlCount,
                                                                 uint8_t timeBinCount,
                                                                 uint8_t maxBinLocalKeys,
                                                                 uint32_t keyframeCount,
                                                                 uint32_t streamBytes,
                                                                 bool alignTo4Bytes)
{
    RenderTemporalControlStreamData out;
    out.codec = codec;
    out.controlCount = controlCount;
    out.timeBinCount = timeBinCount;
    out.maxBinLocalKeys = maxBinLocalKeys;
    out.descriptors.resize(controlCount);
    out.binIndex.resize(static_cast<size_t>(controlCount) * timeBinCount);
    out.keyframes.resize(keyframeCount);

    const uint32_t descriptorBytes = static_cast<uint32_t>(controlCount) * sizeof(RenderTemporalControlDescriptor);
    uint32_t binIndexBytes = static_cast<uint32_t>(controlCount) * timeBinCount;
    uint32_t keyframeBytes = keyframeCount * renderTemporalKeyframeBytes(codec);
    if (alignTo4Bytes) {
        binIndexBytes = alignUp4(binIndexBytes);
        keyframeBytes = alignUp4(keyframeBytes);
    }
    if (descriptorBytes + binIndexBytes + keyframeBytes > streamBytes) {
        throw std::runtime_error("render temporal stream bytes too small");
    }

    std::memcpy(out.descriptors.data(), streamBase, descriptorBytes);
    std::memcpy(out.binIndex.data(), streamBase + descriptorBytes, out.binIndex.size());
    const uint8_t* keyBase = streamBase + descriptorBytes + binIndexBytes;
    for (uint32_t i = 0; i < keyframeCount; ++i) {
        out.keyframes[static_cast<size_t>(i)] = readPackedKeyframe(keyBase, codec, i, keyframeCount);
    }
    return out;
}

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
                                       int frameIndex)
{
    if (controlIndex >= controlCount) {
        throw std::runtime_error("render temporal control index out of range");
    }

    const uint32_t descriptorBytes = static_cast<uint32_t>(controlCount) * sizeof(RenderTemporalControlDescriptor);
    uint32_t binIndexBytes = static_cast<uint32_t>(controlCount) * timeBinCount;
    uint32_t keyframeBytes = keyframeCount * renderTemporalKeyframeBytes(codec);
    if (alignTo4Bytes) {
        binIndexBytes = alignUp4(binIndexBytes);
        keyframeBytes = alignUp4(keyframeBytes);
    }
    const uint32_t totalBytes = descriptorBytes + binIndexBytes + keyframeBytes;
    if (totalBytes > streamBytes) {
        throw std::runtime_error("render temporal stream decode out of bounds");
    }

    const auto desc =
        readPod<RenderTemporalControlDescriptor>(streamBase,
                                                static_cast<size_t>(controlIndex) * sizeof(RenderTemporalControlDescriptor),
                                                streamBytes);
    if (desc.keyCount == 0) return 0.0f;
    if (static_cast<uint32_t>(desc.keyStart) + desc.keyCount > keyframeCount) {
        throw std::runtime_error("render temporal keyframe range out of bounds");
    }

    const uint8_t* binBase = streamBase + descriptorBytes +
                             static_cast<size_t>(controlIndex) * timeBinCount;
    const uint8_t* keyBase = streamBase + descriptorBytes + binIndexBytes;
    const uint8_t localMaxBinLocalKeys =
        (desc.reserved != 0u) ? desc.reserved : maxBinLocalKeys;
    return decodeFromSequence(keyBase,
                              codec,
                              keyframeCount,
                              desc.keyStart,
                              desc.keyCount,
                              binBase,
                              timeBinCount,
                              localMaxBinLocalKeys,
                              totalFrames,
                              frameIndex);
}

float decodeRenderTemporalLeafControlValue(const uint8_t* leafBase,
                                           const RenderTemporalPackedHeaderFields& header,
                                           const RenderTemporalPackedLeafPrefix& prefix,
                                           const RenderTemporalPackedLeafLayout& layout,
                                           bool useFineStream,
                                           bool alignTo4Bytes,
                                           int totalFrames,
                                           uint16_t controlIndex,
                                           int frameIndex)
{
    const uint16_t coarseControlCount =
        static_cast<uint16_t>(header.coarseResolution * header.coarseResolution * header.coarseResolution);
    const uint16_t fineControlCount =
        (header.fineResolution == 0u)
            ? 0u
            : static_cast<uint16_t>(header.fineResolution * header.fineResolution * header.fineResolution);
    if (useFineStream) {
        return decodeRenderTemporalControlValue(leafBase + layout.fineOffset,
                                                header.fineCodec,
                                                fineControlCount,
                                                kRenderTemporalPackedTimeBinCount,
                                                0u,
                                                prefix.fineKeyframeCount,
                                                layout.fine.totalBytes,
                                                alignTo4Bytes,
                                                totalFrames,
                                                controlIndex,
                                                frameIndex);
    }
    return decodeRenderTemporalControlValue(leafBase + layout.coarseOffset,
                                            header.coarseCodec,
                                            coarseControlCount,
                                            kRenderTemporalPackedTimeBinCount,
                                            0u,
                                            prefix.coarseKeyframeCount,
                                            layout.coarse.totalBytes,
                                            alignTo4Bytes,
                                            totalFrames,
                                            controlIndex,
                                            frameIndex);
}

// Mode2 = TemporalShellVoxel decode.
// Doc: VBT/实现策略/liquid_shell_packed_mode_design_2026-04-08.md sections 7-8.
// Returns the additive shell residual contribution for the given (voxel, frame).
// Caller composes coarse + shellResidual to obtain final density.
float decodeRenderTemporalLeafShellVoxelValue(const uint8_t* leafBase,
                                              const RenderTemporalPackedLeafView& leafView,
                                              bool alignTo4Bytes,
                                              int totalFrames,
                                              std::uint16_t voxelIndex,
                                              int frameIndex)
{
    if (leafBase == nullptr || voxelIndex >= 512) {
        return 0.0f;
    }
    if (leafView.header.mode != TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT) {
        return 0.0f;
    }
    const uint32_t group = static_cast<uint32_t>(voxelIndex >> 6);
    const uint32_t bit = static_cast<uint32_t>(voxelIndex & 63);

    const uint32_t shellOffset = leafView.layout.fine.descriptorOffset;
    const auto section = unpackRenderTemporalShellOccupancySection(
        leafBase + shellOffset,
        leafView.layout.fine.descriptorBytes);
    const uint64_t mask = section.shellMask[group];
    if ((mask & (1ull << bit)) == 0ull) {
        return 0.0f;
    }
    const uint64_t maskBefore = mask & ((1ull << bit) - 1ull);
    uint16_t descIndex = section.groupPrefix[group];
    for (uint64_t w = maskBefore; w != 0; w &= (w - 1)) {
        descIndex += 1;
    }

    const uint32_t residualOffset = leafView.shellResidualOffset;
    const uint16_t activeCount = leafView.shellActiveVoxelCount;
    if (activeCount == 0 || leafView.prefix.fineKeyframeCount == 0) {
        return 0.0f;
    }
    return decodeRenderTemporalControlValue(leafBase + residualOffset,
                                            leafView.header.fineCodec,
                                            activeCount,
                                            kRenderTemporalShellPackedTimeBinCount,
                                            0u,
                                            leafView.prefix.fineKeyframeCount,
                                            leafView.shellResidualBytes,
                                            alignTo4Bytes,
                                            totalFrames,
                                            descIndex,
                                            frameIndex);
}

} // namespace vbt
