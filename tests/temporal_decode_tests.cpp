#include "render_temporal_decode.h"
#include "vbt_cycles_sampler.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace vbt;

static void near(float actual, float expected) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 2e-6f) {
        throw std::runtime_error("temporal value mismatch: " + std::to_string(actual) + " vs " + std::to_string(expected));
    }
}

static RenderTemporalControlStreamData makeStream(int frames, int bins, int codec, bool dense) {
    RenderTemporalControlStreamData stream;
    stream.codec = static_cast<RenderTemporalSequenceCodec>(codec);
    stream.controlCount = 1;
    stream.timeBinCount = static_cast<uint8_t>(bins);
    stream.maxBinLocalKeys = 255;
    for (int t = 0; t < frames; ++t) {
        if (dense || t == 0 || t == frames - 1 || t % 13 == 7) {
            float value = std::sin(static_cast<float>(t) * .19f);
            if (codec == 0) value = renderTemporalHalfBitsToFloat(renderTemporalFloatToHalfBits(value));
            stream.keyframes.push_back({static_cast<uint16_t>(t), value});
        }
    }
    stream.descriptors.push_back({0, static_cast<uint8_t>(stream.keyframes.size()), 1});
    for (int bin = 0; bin < bins; ++bin) {
        const int boundary = bin * frames / bins;
        size_t index = 0;
        while (index < stream.keyframes.size() && stream.keyframes[index].frame < boundary) ++index;
        stream.binIndex.push_back(static_cast<uint8_t>(index == 0 ? 0 : index - 1));
    }
    return stream;
}

static float reference(const RenderTemporalControlStreamData& stream, int frames, int frame) {
    frame = std::clamp(frame, 0, frames - 1);
    const auto& keys = stream.keyframes;
    auto hi = std::lower_bound(keys.begin(), keys.end(), frame,
        [](const auto& key, int t) { return key.frame < t; });
    if (hi == keys.end()) return keys.back().value;
    if (hi == keys.begin() || hi->frame == frame) return hi->value;
    const auto lo = hi - 1;
    const float alpha = static_cast<float>(frame - lo->frame) / (hi->frame - lo->frame);
    return lo->value * (1.0f - alpha) + hi->value * alpha;
}

static void checkStream(RenderTemporalControlStreamData stream, int frames, int hint) {
    if (hint >= 0) std::fill(stream.binIndex.begin(), stream.binIndex.end(), static_cast<uint8_t>(hint));
    const auto packed = packRenderTemporalControlStream(stream, true);
    std::vector<uint32_t> words(packed.size() / 4);
    std::memcpy(words.data(), packed.data(), packed.size());
    for (int t = -2; t <= frames + 2; ++t) {
        const float expected = reference(stream, frames, t);
        near(decodeRenderTemporalControlValue(packed.data(), stream.codec, 1, stream.timeBinCount,
            1, static_cast<uint32_t>(stream.keyframes.size()), static_cast<uint32_t>(packed.size()), true, frames, 0, t), expected);
        if (stream.timeBinCount == 8) {
            near(vbt_cycles::decode_temporal_control_value(words.data(), 0, static_cast<uint32_t>(stream.codec),
                1, static_cast<uint32_t>(stream.keyframes.size()), 0, frames, t), expected);
        }
    }
}

static void shellTest() {
    auto coarse = makeStream(101, 8, 0, false);
    coarse.controlCount = 64;
    coarse.descriptors.resize(64, coarse.descriptors.front());
    coarse.binIndex.resize(64 * 8);
    for (int c = 1; c < 64; ++c) std::copy_n(coarse.binIndex.begin(), 8, coarse.binIndex.begin() + c * 8);
    auto residual = makeStream(101, 4, 1, true);
    const std::vector<int> voxels{0, 63, 64, 255, 511};
    residual.controlCount = static_cast<uint16_t>(voxels.size());
    residual.descriptors.resize(voxels.size(), residual.descriptors.front());
    residual.binIndex.resize(voxels.size() * 4);
    for (size_t c = 1; c < voxels.size(); ++c) std::copy_n(residual.binIndex.begin(), 4, residual.binIndex.begin() + c * 4);
    RenderTemporalShellOccupancySection occupancy;
    for (int v : voxels) occupancy.shellMask[v / 64] |= uint64_t(1) << (v % 64);
    for (int group = 0; group < 8; ++group) {
        occupancy.groupPrefix[group] = static_cast<uint16_t>(std::count_if(voxels.begin(), voxels.end(),
            [group](int v) { return v < group * 64; }));
    }
    RenderTemporalPackedHeaderFields header;
    header.mode = TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT;
    header.fineCodec = residual.codec;
    const auto prefix = makeRenderTemporalLeafPrefix(header.mode,
        static_cast<uint32_t>(coarse.keyframes.size()), static_cast<uint32_t>(residual.keyframes.size()));
    const auto leaf = buildRenderTemporalPackedLeafBytes(header, prefix, packRenderTemporalControlStream(coarse, true),
        packRenderTemporalShellOccupancySection(occupancy), packRenderTemporalControlStream(residual, true), true);
    const auto view = parseRenderTemporalPackedLeaf(leaf.data(), leaf.size(), true);
    for (int t = -1; t <= 101; ++t) {
        for (int v = 0; v < 512; ++v) {
            const bool active = std::find(voxels.begin(), voxels.end(), v) != voxels.end();
            near(decodeRenderTemporalLeafShellVoxelValue(leaf.data(), view, true, 101, static_cast<uint16_t>(v), t),
                active ? reference(residual, 101, t) : 0.0f);
        }
    }
}

int main() {
    try {
        for (int codec : {0, 1}) for (int bins : {4, 8}) for (int frames = 1; frames <= 255; ++frames) {
            checkStream(makeStream(frames, bins, codec, true), frames, -1);
            for (int hint : {-1, 0, 255}) checkStream(makeStream(frames, bins, codec, false), frames, hint);
        }
        shellTest();
        std::cout << "Host/native dense/sparse, 4/8 bins, invalid hints and Mode2 occupancy passed.\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
