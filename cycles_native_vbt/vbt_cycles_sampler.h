#pragma once

#include <stdint.h>

/*
 * Device-style render VBT sampler for native Cycles integration.
 *
 * This header is intentionally standalone. It mirrors the render-side shader
 * sampler in render/shaders/vbt_smoke_sample.comp and covers the main-paper
 * smoke/fire modes:
 *
 *   tag 0: Empty
 *   tag 1: TemporalGrid4
 *   tag 3: TemporalFine6
 *
 * Tag 2 is reserved for the compact shell-residual variant and is not sampled
 * in the native Cycles MVP.
 */

#ifndef VBT_CYCLES_DEVICE
#  define VBT_CYCLES_DEVICE inline
#endif

namespace vbt_cycles {

enum : uint32_t {
    VBT_MODE_EMPTY = 0u,
    VBT_MODE_TEMPORAL_GRID4 = 1u,
    VBT_MODE_TEMPORAL_FINE_COMPACT_RESERVED = 2u,
    VBT_MODE_TEMPORAL_FINE6 = 3u,
    VBT_TIME_BIN_COUNT = 8u,
};

struct Header {
    uint32_t dim_x = 0;
    uint32_t dim_y = 0;
    uint32_t dim_z = 0;
    uint32_t frames = 0;
    uint32_t leaf_size = 8;
    uint32_t leaf_count_x = 0;
    uint32_t leaf_count_y = 0;
    int frame_index = 0;
    float bbox_min_x = 0.0f;
    float bbox_min_y = 0.0f;
    float bbox_min_z = 0.0f;
};

struct PayloadView {
    Header header;
    const uint32_t *offset_words = nullptr;
    const uint32_t *payload_words = nullptr;
};

VBT_CYCLES_DEVICE uint32_t min_u32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

VBT_CYCLES_DEVICE uint32_t max_u32(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

VBT_CYCLES_DEVICE int clamp_i32(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

VBT_CYCLES_DEVICE float clamp_f32(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

VBT_CYCLES_DEVICE float mix_f32(float a, float b, float t)
{
    return a * (1.0f - t) + b * t;
}

VBT_CYCLES_DEVICE uint32_t align_up4(uint32_t bytes)
{
    return (bytes + 3u) & ~3u;
}

VBT_CYCLES_DEVICE float uint_as_float(uint32_t bits)
{
    union {
        uint32_t u;
        float f;
    } cvt;
    cvt.u = bits;
    return cvt.f;
}

VBT_CYCLES_DEVICE uint32_t read_payload_byte(const uint32_t *payload, uint32_t byte_offset)
{
    const uint32_t word = payload[byte_offset >> 2u];
    const uint32_t shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 0xffu;
}

VBT_CYCLES_DEVICE uint32_t read_payload_u16(const uint32_t *payload, uint32_t byte_offset)
{
    return read_payload_byte(payload, byte_offset) |
           (read_payload_byte(payload, byte_offset + 1u) << 8u);
}

VBT_CYCLES_DEVICE float half_to_float(uint32_t h)
{
    const uint32_t sign = (h >> 15u) & 1u;
    uint32_t exponent = (h >> 10u) & 0x1fu;
    uint32_t mantissa = h & 0x3ffu;
    uint32_t bits = 0u;

    if (exponent == 0u) {
        if (mantissa == 0u) {
            bits = sign << 31u;
        }
        else {
            int e = -14;
            while ((mantissa & 0x400u) == 0u) {
                mantissa <<= 1u;
                --e;
            }
            mantissa &= 0x3ffu;
            bits = (sign << 31u) | (static_cast<uint32_t>(e + 127) << 23u) | (mantissa << 13u);
        }
    }
    else if (exponent == 31u) {
        bits = (sign << 31u) | 0x7f800000u | (mantissa << 13u);
    }
    else {
        bits = (sign << 31u) | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }
    return uint_as_float(bits);
}

VBT_CYCLES_DEVICE float read_keyframe_value(const uint32_t *payload,
                                            uint32_t key_base,
                                            uint32_t codec,
                                            uint32_t key_index,
                                            uint32_t *key_frame)
{
    if (codec == 1u) {
        const uint32_t offset = key_base + key_index * 8u;
        *key_frame = read_payload_u16(payload, offset);
        const uint32_t bits = read_payload_byte(payload, offset + 4u) |
                              (read_payload_byte(payload, offset + 5u) << 8u) |
                              (read_payload_byte(payload, offset + 6u) << 16u) |
                              (read_payload_byte(payload, offset + 7u) << 24u);
        return uint_as_float(bits);
    }

    const uint32_t offset = key_base + key_index * 3u;
    *key_frame = read_payload_byte(payload, offset);
    return half_to_float(read_payload_u16(payload, offset + 1u));
}

VBT_CYCLES_DEVICE float decode_temporal_control_value(const uint32_t *payload,
                                                       uint32_t stream_base,
                                                       uint32_t codec,
                                                       uint32_t control_count,
                                                       uint32_t keyframe_count,
                                                       uint32_t control_index,
                                                       int total_frames,
                                                       int frame_index)
{
    (void)keyframe_count;
    const uint32_t descriptor_bytes = control_count * 4u;
    const uint32_t bin_index_bytes = align_up4(control_count * VBT_TIME_BIN_COUNT);
    const uint32_t key_base = stream_base + descriptor_bytes + bin_index_bytes;

    const uint32_t desc_offset = stream_base + control_index * 4u;
    const uint32_t key_start = read_payload_u16(payload, desc_offset);
    const uint32_t key_count = read_payload_byte(payload, desc_offset + 2u);
    const uint32_t reserved = read_payload_byte(payload, desc_offset + 3u);

    if (key_count == 0u) {
        return 0.0f;
    }
    if (key_count == 1u) {
        uint32_t only_frame = 0u;
        return read_keyframe_value(payload, key_base, codec, key_start, &only_frame);
    }

    const int clamped_frame = clamp_i32(frame_index, 0, total_frames > 0 ? total_frames - 1 : 0);
    const uint32_t safe_total_frames = max_u32(static_cast<uint32_t>(total_frames), 1u);
    const uint32_t bin = min_u32(VBT_TIME_BIN_COUNT - 1u,
                                 (static_cast<uint32_t>(clamped_frame) * VBT_TIME_BIN_COUNT) /
                                     safe_total_frames);
    const uint32_t bin_base = stream_base + descriptor_bytes + control_index * VBT_TIME_BIN_COUNT;
    int begin_local = static_cast<int>(read_payload_byte(payload, bin_base + bin));
    begin_local = clamp_i32(begin_local, 0, static_cast<int>(key_count) - 1);

    int end_local = static_cast<int>(key_count);
    for (int next = static_cast<int>(bin) + 1; next < static_cast<int>(VBT_TIME_BIN_COUNT); ++next) {
        const int next_local = static_cast<int>(read_payload_byte(payload, bin_base + static_cast<uint32_t>(next)));
        if (next_local > begin_local) {
            end_local = next_local < static_cast<int>(key_count) ? next_local : static_cast<int>(key_count);
            break;
        }
    }

    const int local_max_bin_keys = reserved != 0u ? static_cast<int>(reserved) : 8;
    const int bounded_end = begin_local + (local_max_bin_keys > 1 ? local_max_bin_keys : 1);
    end_local = end_local < bounded_end ? end_local : bounded_end;

    int prev_local = begin_local;
    int curr_local = begin_local;
    while (curr_local < end_local) {
        uint32_t curr_frame = 0u;
        const float curr_value = read_keyframe_value(payload,
                                                     key_base,
                                                     codec,
                                                     key_start + static_cast<uint32_t>(curr_local),
                                                     &curr_frame);
        if (static_cast<int>(curr_frame) >= clamped_frame) {
            if (static_cast<int>(curr_frame) <= clamped_frame || curr_local == prev_local) {
                return curr_value;
            }
            uint32_t prev_frame = 0u;
            const float prev_value = read_keyframe_value(payload,
                                                         key_base,
                                                         codec,
                                                         key_start + static_cast<uint32_t>(prev_local),
                                                         &prev_frame);
            const int f0 = static_cast<int>(prev_frame);
            const int f1 = static_cast<int>(curr_frame);
            if (f1 <= f0) {
                return prev_value;
            }
            const float alpha = static_cast<float>(clamped_frame - f0) / static_cast<float>(f1 - f0);
            return mix_f32(prev_value, curr_value, alpha);
        }
        prev_local = curr_local;
        ++curr_local;
    }

    curr_local = curr_local < static_cast<int>(key_count) ? curr_local : static_cast<int>(key_count) - 1;
    uint32_t curr_frame = 0u;
    const float curr_value = read_keyframe_value(payload,
                                                 key_base,
                                                 codec,
                                                 key_start + static_cast<uint32_t>(curr_local),
                                                 &curr_frame);
    if (static_cast<int>(curr_frame) <= clamped_frame || curr_local == prev_local) {
        return curr_value;
    }

    uint32_t prev_frame = 0u;
    const float prev_value = read_keyframe_value(payload,
                                                 key_base,
                                                 codec,
                                                 key_start + static_cast<uint32_t>(prev_local),
                                                 &prev_frame);
    const int f0 = static_cast<int>(prev_frame);
    const int f1 = static_cast<int>(curr_frame);
    if (f1 <= f0) {
        return prev_value;
    }
    const float alpha = static_cast<float>(clamped_frame - f0) / static_cast<float>(f1 - f0);
    return mix_f32(prev_value, curr_value, alpha);
}

VBT_CYCLES_DEVICE float control_coord(uint32_t index, uint32_t resolution)
{
    return resolution <= 1u ? 0.0f : static_cast<float>(index) * (7.0f / static_cast<float>(resolution - 1u));
}

VBT_CYCLES_DEVICE void locate_coord(float p,
                                    uint32_t extent,
                                    uint32_t resolution,
                                    uint32_t *i0,
                                    uint32_t *i1,
                                    float *w)
{
    const float max_p = static_cast<float>(extent > 0u ? extent - 1u : 0u);
    const float clamped = clamp_f32(p, 0.0f, max_p);
    uint32_t hi = 1u;
    while (hi < resolution - 1u && control_coord(hi, resolution) < clamped) {
        ++hi;
    }
    const uint32_t lo = hi > 0u ? hi - 1u : 0u;
    const float c0 = control_coord(lo, resolution);
    const float c1 = control_coord(hi, resolution);
    *i0 = lo;
    *i1 = hi;
    *w = c1 > c0 ? ((clamped - c0) / (c1 - c0)) : 0.0f;
}

VBT_CYCLES_DEVICE uint32_t control_index3(uint32_t x, uint32_t y, uint32_t z, uint32_t resolution)
{
    return (z * resolution + y) * resolution + x;
}

VBT_CYCLES_DEVICE float sample_control_stream(const PayloadView &vbt,
                                              uint32_t stream_base,
                                              uint32_t codec,
                                              uint32_t resolution,
                                              uint32_t keyframe_count,
                                              uint32_t leaf_width,
                                              uint32_t leaf_height,
                                              uint32_t leaf_depth,
                                              float lx,
                                              float ly,
                                              float lz)
{
    uint32_t x0 = 0u, x1 = 0u, y0 = 0u, y1 = 0u, z0 = 0u, z1 = 0u;
    float tx = 0.0f, ty = 0.0f, tz = 0.0f;
    locate_coord(lx, leaf_width, resolution, &x0, &x1, &tx);
    locate_coord(ly, leaf_height, resolution, &y0, &y1, &ty);
    locate_coord(lz, leaf_depth, resolution, &z0, &z1, &tz);

    const uint32_t control_count = resolution * resolution * resolution;
    const uint32_t frames = vbt.header.frames;
    const int frame = vbt.header.frame_index;
    const uint32_t *payload = vbt.payload_words;

    const float v000 = decode_temporal_control_value(payload, stream_base, codec, control_count, keyframe_count, control_index3(x0, y0, z0, resolution), static_cast<int>(frames), frame);
    const float v100 = decode_temporal_control_value(payload, stream_base, codec, control_count, keyframe_count, control_index3(x1, y0, z0, resolution), static_cast<int>(frames), frame);
    const float v010 = decode_temporal_control_value(payload, stream_base, codec, control_count, keyframe_count, control_index3(x0, y1, z0, resolution), static_cast<int>(frames), frame);
    const float v110 = decode_temporal_control_value(payload, stream_base, codec, control_count, keyframe_count, control_index3(x1, y1, z0, resolution), static_cast<int>(frames), frame);
    const float v001 = decode_temporal_control_value(payload, stream_base, codec, control_count, keyframe_count, control_index3(x0, y0, z1, resolution), static_cast<int>(frames), frame);
    const float v101 = decode_temporal_control_value(payload, stream_base, codec, control_count, keyframe_count, control_index3(x1, y0, z1, resolution), static_cast<int>(frames), frame);
    const float v011 = decode_temporal_control_value(payload, stream_base, codec, control_count, keyframe_count, control_index3(x0, y1, z1, resolution), static_cast<int>(frames), frame);
    const float v111 = decode_temporal_control_value(payload, stream_base, codec, control_count, keyframe_count, control_index3(x1, y1, z1, resolution), static_cast<int>(frames), frame);

    const float c00 = mix_f32(v000, v100, tx);
    const float c10 = mix_f32(v010, v110, tx);
    const float c01 = mix_f32(v001, v101, tx);
    const float c11 = mix_f32(v011, v111, tx);
    return mix_f32(mix_f32(c00, c10, ty), mix_f32(c01, c11, ty), tz);
}

VBT_CYCLES_DEVICE float sample_density_index(const PayloadView &vbt, float x, float y, float z)
{
    const Header &h = vbt.header;
    if (x < 0.0f || y < 0.0f || z < 0.0f ||
        x > static_cast<float>(h.dim_x - 1u) ||
        y > static_cast<float>(h.dim_y - 1u) ||
        z > static_cast<float>(h.dim_z - 1u)) {
        return 0.0f;
    }

    const uint32_t xi = min_u32(h.dim_x - 1u, static_cast<uint32_t>(x));
    const uint32_t yi = min_u32(h.dim_y - 1u, static_cast<uint32_t>(y));
    const uint32_t zi = min_u32(h.dim_z - 1u, static_cast<uint32_t>(z));
    const uint32_t bx = min_u32(h.leaf_count_x - 1u, xi / h.leaf_size);
    const uint32_t by = min_u32(h.leaf_count_y - 1u, yi / h.leaf_size);
    const uint32_t leaf_count_z = (h.dim_z + h.leaf_size - 1u) / h.leaf_size;
    const uint32_t bz = min_u32(leaf_count_z - 1u, zi / h.leaf_size);
    const uint32_t leaf_index = (bz * h.leaf_count_y + by) * h.leaf_count_x + bx;

    const uint32_t word_begin = vbt.offset_words[leaf_index];
    const uint32_t word_end = vbt.offset_words[leaf_index + 1u];
    if (word_end <= word_begin) {
        return 0.0f;
    }

    const uint32_t leaf_base_byte = word_begin * 4u;
    const uint32_t packed_header = vbt.payload_words[word_begin];
    const uint32_t mode = packed_header & 0x3u;
    if (mode == VBT_MODE_EMPTY) {
        return 0.0f;
    }

    const uint32_t coarse_codec = (packed_header >> 2u) & 0x3u;
    const uint32_t fine_codec = (packed_header >> 4u) & 0x3u;
    const uint32_t coarse_resolution = (packed_header >> 6u) & 0x7u;
    const uint32_t fine_resolution = (packed_header >> 9u) & 0x7u;
    const uint32_t coarse_keyframe_count = read_payload_u16(vbt.payload_words, leaf_base_byte + 4u);
    const uint32_t fine_keyframe_count = read_payload_u16(vbt.payload_words, leaf_base_byte + 6u);

    const uint32_t coarse_control_count = coarse_resolution * coarse_resolution * coarse_resolution;
    const uint32_t coarse_descriptor_bytes = coarse_control_count * 4u;
    const uint32_t coarse_bin_bytes = align_up4(coarse_control_count * VBT_TIME_BIN_COUNT);
    const uint32_t coarse_keyframe_stride = coarse_codec == 1u ? 8u : 3u;
    const uint32_t coarse_keyframe_bytes = align_up4(coarse_keyframe_count * coarse_keyframe_stride);
    const uint32_t coarse_stream_base = leaf_base_byte + 8u;
    const uint32_t fine_stream_base = coarse_stream_base + coarse_descriptor_bytes + coarse_bin_bytes + coarse_keyframe_bytes;

    const uint32_t base_x = bx * h.leaf_size;
    const uint32_t base_y = by * h.leaf_size;
    const uint32_t base_z = bz * h.leaf_size;
    const uint32_t leaf_width = min_u32(h.leaf_size, h.dim_x - base_x);
    const uint32_t leaf_height = min_u32(h.leaf_size, h.dim_y - base_y);
    const uint32_t leaf_depth = min_u32(h.leaf_size, h.dim_z - base_z);
    const float lx = x - static_cast<float>(base_x);
    const float ly = y - static_cast<float>(base_y);
    const float lz = z - static_cast<float>(base_z);

    float value = sample_control_stream(vbt,
                                        coarse_stream_base,
                                        coarse_codec,
                                        coarse_resolution,
                                        coarse_keyframe_count,
                                        leaf_width,
                                        leaf_height,
                                        leaf_depth,
                                        lx,
                                        ly,
                                        lz);

    if (mode == VBT_MODE_TEMPORAL_FINE6 && fine_resolution > 0u) {
        value += sample_control_stream(vbt,
                                       fine_stream_base,
                                       fine_codec,
                                       fine_resolution,
                                       fine_keyframe_count,
                                       leaf_width,
                                       leaf_height,
                                       leaf_depth,
                                       lx,
                                       ly,
                                       lz);
    }

    return value;
}

VBT_CYCLES_DEVICE float sample_density_world(const PayloadView &vbt, float x, float y, float z)
{
    return sample_density_index(vbt,
                                x - vbt.header.bbox_min_x,
                                y - vbt.header.bbox_min_y,
                                z - vbt.header.bbox_min_z);
}

}  // namespace vbt_cycles
