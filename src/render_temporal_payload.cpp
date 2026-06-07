#include "render_temporal_payload.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace vbt {

namespace {

uint32_t alignUp4(uint32_t bytes)
{
    return (bytes + 3u) & ~3u;
}

uint32_t codecToBits(RenderTemporalSequenceCodec codec)
{
    return static_cast<uint32_t>(codec) & 0x3u;
}

RenderTemporalSequenceCodec bitsToCodec(uint32_t bits)
{
    return static_cast<RenderTemporalSequenceCodec>(bits & 0x3u);
}

uint16_t controlCountForResolution(uint32_t resolution)
{
    if (resolution == 0u) return 0u;
    return static_cast<uint16_t>(resolution * resolution * resolution);
}

uint32_t keyframeBytesForCodec(RenderTemporalSequenceCodec codec)
{
    switch (codec) {
    case RenderTemporalSequenceCodec::DP_KEYFRAME_FP32:
        return 8u;
    case RenderTemporalSequenceCodec::DP_KEYFRAME_FP16:
    default:
        return 3u;
    }
}

uint32_t streamDescriptorBytes(uint32_t controlCount)
{
    return controlCount * kRenderTemporalControlDescriptorBytes;
}

uint32_t streamBinIndexBytes(uint32_t controlCount, uint32_t timeBinCount, bool alignTo4Bytes)
{
    uint32_t bytes = controlCount * timeBinCount;
    return alignTo4Bytes ? alignUp4(bytes) : bytes;
}

uint32_t streamKeyframeBytes(uint32_t keyframeCount,
                             RenderTemporalSequenceCodec codec,
                             bool alignTo4Bytes)
{
    uint32_t bytes = keyframeCount * keyframeBytesForCodec(codec);
    return alignTo4Bytes ? alignUp4(bytes) : bytes;
}

RenderTemporalStreamLayout computeStreamLayout(uint32_t startOffset,
                                               uint32_t controlCount,
                                               uint32_t timeBinCount,
                                               uint32_t keyframeCount,
                                               RenderTemporalSequenceCodec codec,
                                               bool alignTo4Bytes)
{
    RenderTemporalStreamLayout out;
    out.descriptorOffset = startOffset;
    out.descriptorBytes = streamDescriptorBytes(controlCount);
    out.binIndexOffset = out.descriptorOffset + out.descriptorBytes;
    out.binIndexBytes = streamBinIndexBytes(controlCount, timeBinCount, alignTo4Bytes);
    out.keyframeOffset = out.binIndexOffset + out.binIndexBytes;
    out.keyframeBytes = streamKeyframeBytes(keyframeCount, codec, alignTo4Bytes);
    out.totalBytes = out.descriptorBytes + out.binIndexBytes + out.keyframeBytes;
    return out;
}

} // namespace

uint32_t renderTemporalKeyframeBytes(RenderTemporalSequenceCodec codec)
{
    return keyframeBytesForCodec(codec);
}

uint32_t packRenderTemporalHeader(const RenderTemporalPackedHeaderFields& fields)
{
    uint32_t packed = 0;
    packed |= (static_cast<uint32_t>(fields.mode) & 0x3u);
    packed |= (codecToBits(fields.coarseCodec) & 0x3u) << 2;
    packed |= (codecToBits(fields.fineCodec) & 0x3u) << 4;
    packed |= (std::min<uint32_t>(fields.coarseResolution, 0x7u) & 0x7u) << 6;
    packed |= (std::min<uint32_t>(fields.fineResolution, 0x7u) & 0x7u) << 9;
    return packed;
}

RenderTemporalPackedHeaderFields unpackRenderTemporalHeader(uint32_t packedHeader)
{
    RenderTemporalPackedHeaderFields out;
    out.mode = static_cast<TemporalFirstPackedMode>(packedHeader & 0x3u);
    out.coarseCodec = bitsToCodec((packedHeader >> 2) & 0x3u);
    out.fineCodec = bitsToCodec((packedHeader >> 4) & 0x3u);
    out.coarseResolution = static_cast<uint8_t>((packedHeader >> 6) & 0x7u);
    out.fineResolution = static_cast<uint8_t>((packedHeader >> 9) & 0x7u);
    return out;
}

RenderTemporalPackedLeafPrefix makeRenderTemporalLeafPrefix(TemporalFirstPackedMode mode,
                                                            uint32_t coarseKeyframeCount,
                                                            uint32_t fineKeyframeCount)
{
    (void)mode;
    RenderTemporalPackedLeafPrefix out;
    if (coarseKeyframeCount > 0xFFFFu || fineKeyframeCount > 0xFFFFu) {
        throw std::runtime_error("render temporal prefix keyframe count exceeds uint16");
    }
    out.coarseKeyframeCount = static_cast<uint16_t>(coarseKeyframeCount);
    out.fineKeyframeCount = static_cast<uint16_t>(fineKeyframeCount);
    return out;
}

bool validateRenderTemporalLeafPrefix(const RenderTemporalPackedLeafPrefix& prefix,
                                      const RenderTemporalPackedHeaderFields& header,
                                      const TemporalFirstEncoderOptions& options)
{
    if (options.timeBinCount != kRenderTemporalPackedTimeBinCount) {
        return false;
    }
    const uint16_t coarseControlCount = controlCountForResolution(header.coarseResolution);
    const uint16_t fineControlCount = controlCountForResolution(header.fineResolution);
    if (coarseControlCount == 0 && prefix.coarseKeyframeCount != 0) {
        return false;
    }
    if ((header.mode == TemporalFirstPackedMode::TEMPORAL_FINE6 ||
         header.mode == TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT) &&
        fineControlCount == 0 && prefix.fineKeyframeCount != 0) {
        return false;
    }
    return true;
}

RenderTemporalPackedLeafLayout computeRenderTemporalLeafLayout(const RenderTemporalPackedHeaderFields& header,
                                                              const RenderTemporalPackedLeafPrefix& prefix,
                                                              bool alignTo4Bytes)
{
    RenderTemporalPackedLeafLayout out;
    out.coarseOffset = out.prefixOffset + out.prefixBytes;
    const uint16_t coarseControlCount = controlCountForResolution(header.coarseResolution);
    const uint16_t fineControlCount = controlCountForResolution(header.fineResolution);

    out.coarse = computeStreamLayout(out.coarseOffset,
                                     coarseControlCount,
                                     kRenderTemporalPackedTimeBinCount,
                                     prefix.coarseKeyframeCount,
                                     header.coarseCodec,
                                     alignTo4Bytes);

    out.fineOffset = out.coarseOffset + out.coarse.totalBytes;
    if (header.mode == TemporalFirstPackedMode::TEMPORAL_FINE6) {
        out.fine = computeStreamLayout(out.fineOffset,
                                       fineControlCount,
                                       kRenderTemporalPackedTimeBinCount,
                                       prefix.fineKeyframeCount,
                                       header.fineCodec,
                                       alignTo4Bytes);
        out.totalBytes = out.fineOffset + out.fine.totalBytes;
    } else if (header.mode == TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT) {
        // Mode2: layout holds only the 80-byte shell occupancy section plus the
        // shell residual stream's keyframe bytes (upper bound). The real shell
        // residual totalBytes depends on the active voxel count and must be
        // computed by the decoder after reading the occupancy mask.
        // Here we record:
        //   - fine.descriptorOffset = shell section offset
        //   - fine.totalBytes = 80 (occupancy) + keyframeBytes (upper bound on residual tail)
        out.fine = RenderTemporalStreamLayout{};
        out.fine.descriptorOffset = out.fineOffset;
        out.fine.descriptorBytes = 80u;
        out.fine.binIndexOffset = out.fineOffset + 80u;
        out.fine.binIndexBytes = 0u;
        out.fine.keyframeOffset = out.fineOffset + 80u;
        out.fine.keyframeBytes = streamKeyframeBytes(prefix.fineKeyframeCount, header.fineCodec, alignTo4Bytes);
        out.fine.totalBytes = 80u + out.fine.keyframeBytes;
        out.totalBytes = out.fineOffset + out.fine.totalBytes;
    } else {
        out.totalBytes = out.coarseOffset + out.coarse.totalBytes;
    }

    if (header.mode == TemporalFirstPackedMode::EMPTY) {
        out.coarseOffset = 0;
        out.fineOffset = 0;
        out.coarse = {};
        out.fine = {};
        out.totalBytes = out.headerBytes;
    }
    return out;
}

std::vector<uint8_t> buildRenderTemporalPackedLeafBytes(const RenderTemporalPackedHeaderFields& header,
                                                        const RenderTemporalPackedLeafPrefix& prefix,
                                                        const std::vector<uint8_t>& coarseStreamBytes,
                                                        const std::vector<uint8_t>& fineStreamBytes,
                                                        bool alignTo4Bytes)
{
    const auto layout = computeRenderTemporalLeafLayout(header, prefix, alignTo4Bytes);
    if (header.mode == TemporalFirstPackedMode::EMPTY) {
        std::vector<uint8_t> out(sizeof(uint32_t), 0u);
        const uint32_t packed = packRenderTemporalHeader(header);
        std::memcpy(out.data(), &packed, sizeof(packed));
        return out;
    }
    if (coarseStreamBytes.size() != layout.coarse.totalBytes) {
        throw std::runtime_error("render temporal coarse stream bytes do not match computed layout");
    }
    if (header.mode == TemporalFirstPackedMode::TEMPORAL_FINE6 &&
        fineStreamBytes.size() != layout.fine.totalBytes) {
        throw std::runtime_error("render temporal fine stream bytes do not match computed layout");
    }
    // Mode2 (TEMPORAL_FINE_COMPACT) uses the 6-arg overload below; the 5-arg version
    // does not validate against layout.fine because Mode2's fine section combines an
    // 80-byte occupancy table with a variable-length shell residual stream.

    std::vector<uint8_t> out(layout.totalBytes, 0u);
    const uint32_t packed = packRenderTemporalHeader(header);
    std::memcpy(out.data(), &packed, sizeof(packed));
    std::memcpy(out.data() + layout.prefixOffset, &prefix, sizeof(prefix));
    std::memcpy(out.data() + layout.coarseOffset, coarseStreamBytes.data(), coarseStreamBytes.size());
    if (!fineStreamBytes.empty()) {
        std::memcpy(out.data() + layout.fineOffset, fineStreamBytes.data(), fineStreamBytes.size());
    }
    return out;
}

RenderTemporalPackedLeafView parseRenderTemporalPackedLeaf(const uint8_t* leafBase,
                                                           size_t leafBytes,
                                                           bool alignTo4Bytes)
{
    if (leafBase == nullptr || leafBytes < sizeof(uint32_t)) {
        throw std::runtime_error("render temporal leaf bytes too small");
    }

    RenderTemporalPackedLeafView out;
    uint32_t packedHeader = 0u;
    std::memcpy(&packedHeader, leafBase, sizeof(packedHeader));
    out.header = unpackRenderTemporalHeader(packedHeader);

    if (out.header.mode == TemporalFirstPackedMode::EMPTY) {
        out.layout = computeRenderTemporalLeafLayout(out.header, out.prefix, alignTo4Bytes);
        return out;
    }
    if (leafBytes < sizeof(uint32_t) + sizeof(RenderTemporalPackedLeafPrefix)) {
        throw std::runtime_error("render temporal leaf prefix missing");
    }
    std::memcpy(&out.prefix, leafBase + sizeof(uint32_t), sizeof(RenderTemporalPackedLeafPrefix));
    out.layout = computeRenderTemporalLeafLayout(out.header, out.prefix, alignTo4Bytes);
    if (out.layout.totalBytes > leafBytes) {
        throw std::runtime_error("render temporal leaf shorter than computed layout");
    }
    return out;
}

// Mode2 = TemporalShellVoxel support.
// Doc: VBT/实现策略/liquid_shell_packed_mode_design_2026-04-08.md.

std::vector<uint8_t> packRenderTemporalShellOccupancySection(const RenderTemporalShellOccupancySection& section)
{
    std::vector<uint8_t> out(80, 0u);
    std::memcpy(out.data(), section.shellMask.data(), 64);
    std::memcpy(out.data() + 64, section.groupPrefix.data(), 16);
    return out;
}

RenderTemporalShellOccupancySection unpackRenderTemporalShellOccupancySection(const uint8_t* data, std::uint64_t size)
{
    RenderTemporalShellOccupancySection section{};
    if (data == nullptr || size < 80) {
        return section;
    }
    std::memcpy(section.shellMask.data(), data, 64);
    std::memcpy(section.groupPrefix.data(), data + 64, 16);
    return section;
}

std::uint16_t renderTemporalShellActiveVoxelCount(const RenderTemporalShellOccupancySection& section)
{
    std::uint16_t count = 0;
    for (auto word : section.shellMask) {
        for (uint64_t w = word; w != 0; w &= (w - 1)) {
            count += 1;
        }
    }
    return count;
}

// 6-arg overload: inserts shell occupancy section between coarse and shell residual streams.
// Layout: [header 4B][prefix 4B][coarse stream][80B occupancy][shell residual stream]
std::vector<uint8_t> buildRenderTemporalPackedLeafBytes(const RenderTemporalPackedHeaderFields& header,
                                                        const RenderTemporalPackedLeafPrefix& prefix,
                                                        const std::vector<uint8_t>& coarseStreamBytes,
                                                        const std::vector<uint8_t>& shellOccupancyBytes,
                                                        const std::vector<uint8_t>& shellResidualStreamBytes,
                                                        bool alignTo4Bytes)
{
    auto alignTo4 = [&](std::vector<uint8_t>& buf) {
        if (!alignTo4Bytes) return;
        while (buf.size() % 4 != 0) buf.push_back(0);
    };

    std::vector<uint8_t> out;
    out.reserve(8 + coarseStreamBytes.size() + shellOccupancyBytes.size() + shellResidualStreamBytes.size() + 16);

    out.resize(4);
    const uint32_t packed = packRenderTemporalHeader(header);
    std::memcpy(out.data(), &packed, sizeof(packed));

    out.resize(8);
    std::memcpy(out.data() + 4, &prefix, sizeof(RenderTemporalPackedLeafPrefix));

    if (!coarseStreamBytes.empty()) {
        out.insert(out.end(), coarseStreamBytes.begin(), coarseStreamBytes.end());
        alignTo4(out);
    }

    if (!shellOccupancyBytes.empty()) {
        out.insert(out.end(), shellOccupancyBytes.begin(), shellOccupancyBytes.end());
        alignTo4(out);
    }

    if (!shellResidualStreamBytes.empty()) {
        out.insert(out.end(), shellResidualStreamBytes.begin(), shellResidualStreamBytes.end());
        alignTo4(out);
    }

    return out;
}

} // namespace vbt
