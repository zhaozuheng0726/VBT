#include "vbt_cycles_loader.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>

namespace vbt_cycles_host {

namespace {

constexpr uint64_t k_header_bytes = 8u + 11u * sizeof(uint32_t);
constexpr uint32_t k_pack4_version = 4u;
constexpr uint32_t k_pack4_leaf_size = 8u;
constexpr uint32_t k_max_coarse_resolution = 8u;
constexpr uint32_t k_max_coarse_scale_count = 64u;

template<typename T>
bool read_value(std::ifstream &in, T &value)
{
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    return static_cast<bool>(in);
}

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool checked_multiply(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0u && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool validate_header(const FileHeader &header, uint64_t file_bytes, std::string &error)
{
    if (std::strncmp(header.magic, "VBTPACK4", 8) != 0) {
        error = "Unsupported VBT magic. Expected VBTPACK4.";
        return false;
    }
    if (header.version != k_pack4_version) {
        error = "Unsupported VBTPACK4 version: " + std::to_string(header.version);
        return false;
    }
    if (header.width == 0u || header.height == 0u || header.depth == 0u || header.frames == 0u) {
        error = "VBTPACK4 dimensions and frame count must be non-zero.";
        return false;
    }
    if (header.leaf_size != k_pack4_leaf_size) {
        error = "VBTPACK4 v4 requires leaf_size=8.";
        return false;
    }
    if (header.coarse_resolution == 0u || header.coarse_resolution > k_max_coarse_resolution) {
        error = "Invalid VBTPACK4 coarse resolution.";
        return false;
    }
    if (header.profile_type > VBT_FIELD_GENERIC) {
        error = "Invalid VBTPACK4 profile type.";
        return false;
    }
    if (header.max_coarse_keep > header.frames ||
        header.coarse_ac_scale_count > k_max_coarse_scale_count ||
        header.coarse_ac_scale_count > header.max_coarse_keep) {
        error = "Invalid VBTPACK4 coarse temporal metadata.";
        return false;
    }

    const uint64_t leaf_count_x =
        (static_cast<uint64_t>(header.width) + header.leaf_size - 1u) / header.leaf_size;
    const uint64_t leaf_count_y =
        (static_cast<uint64_t>(header.height) + header.leaf_size - 1u) / header.leaf_size;
    const uint64_t leaf_count_z =
        (static_cast<uint64_t>(header.depth) + header.leaf_size - 1u) / header.leaf_size;
    uint64_t leaf_count_xy = 0u;
    uint64_t expected_leaf_count = 0u;
    if (!checked_multiply(leaf_count_x, leaf_count_y, leaf_count_xy) ||
        !checked_multiply(leaf_count_xy, leaf_count_z, expected_leaf_count) ||
        expected_leaf_count > std::numeric_limits<uint32_t>::max()) {
        error = "VBTPACK4 leaf count overflows the v4 header.";
        return false;
    }
    if (header.leaf_count != expected_leaf_count) {
        error = "VBTPACK4 leaf_count does not match dimensions and leaf_size.";
        return false;
    }

    uint64_t scale_bytes = 0u;
    uint64_t offset_bytes = 0u;
    uint64_t minimum_bytes = k_header_bytes;
    if (!checked_multiply(header.coarse_ac_scale_count, sizeof(float), scale_bytes) ||
        !checked_multiply(static_cast<uint64_t>(header.leaf_count) + 1u,
                          sizeof(uint32_t),
                          offset_bytes) ||
        !checked_add(minimum_bytes, scale_bytes, minimum_bytes) ||
        !checked_add(minimum_bytes, offset_bytes, minimum_bytes)) {
        error = "VBTPACK4 table sizes overflow.";
        return false;
    }
    if (minimum_bytes > file_bytes) {
        error = "VBTPACK4 file is truncated before its offset table.";
        return false;
    }
    return true;
}

}  // namespace

bool load_pack4(const std::filesystem::path &path, File &out_file, std::string &error)
{
    error.clear();
    std::error_code size_error;
    const uintmax_t file_size_value = std::filesystem::file_size(path, size_error);
    if (size_error) {
        error = "Failed to query VBT file size: " + path.string();
        return false;
    }
    if (file_size_value < k_header_bytes ||
        file_size_value > std::numeric_limits<uint64_t>::max()) {
        error = "Invalid VBTPACK4 file size.";
        return false;
    }
    const uint64_t file_bytes = static_cast<uint64_t>(file_size_value);

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "Failed to open VBT file: " + path.string();
        return false;
    }

    File loaded;
    if (!read_value(in, loaded.header.magic) ||
        !read_value(in, loaded.header.version) ||
        !read_value(in, loaded.header.width) ||
        !read_value(in, loaded.header.height) ||
        !read_value(in, loaded.header.depth) ||
        !read_value(in, loaded.header.frames) ||
        !read_value(in, loaded.header.leaf_size) ||
        !read_value(in, loaded.header.coarse_resolution) ||
        !read_value(in, loaded.header.max_coarse_keep) ||
        !read_value(in, loaded.header.leaf_count) ||
        !read_value(in, loaded.header.profile_type) ||
        !read_value(in, loaded.header.coarse_ac_scale_count)) {
        error = "Failed to read VBTPACK4 header.";
        return false;
    }

    if (!validate_header(loaded.header, file_bytes, error)) {
        return false;
    }

    loaded.coarse_ac_scales.resize(loaded.header.coarse_ac_scale_count);
    if (!loaded.coarse_ac_scales.empty()) {
        in.read(reinterpret_cast<char *>(loaded.coarse_ac_scales.data()),
                static_cast<std::streamsize>(sizeof(float) * loaded.coarse_ac_scales.size()));
        if (!in) {
            error = "Failed to read coarse AC scales.";
            return false;
        }
    }

    loaded.offsets_words.resize(static_cast<size_t>(loaded.header.leaf_count) + 1u);
    in.read(reinterpret_cast<char *>(loaded.offsets_words.data()),
            static_cast<std::streamsize>(sizeof(uint32_t) * loaded.offsets_words.size()));
    if (!in) {
        error = "Failed to read VBT offset table.";
        return false;
    }
    if (loaded.offsets_words.front() != 0u) {
        error = "VBTPACK4 offset table must begin at word zero.";
        return false;
    }
    for (size_t i = 0; i < static_cast<size_t>(loaded.header.leaf_count); ++i) {
        if (loaded.offsets_words[i] >= loaded.offsets_words[i + 1u]) {
            error = "VBTPACK4 offsets must be strictly increasing for every leaf.";
            return false;
        }
    }

    const uint32_t payload_word_count = loaded.offsets_words.back();
    uint64_t scale_bytes = 0u;
    uint64_t offset_bytes = 0u;
    uint64_t payload_bytes = 0u;
    uint64_t expected_bytes = k_header_bytes;
    if (!checked_multiply(loaded.header.coarse_ac_scale_count, sizeof(float), scale_bytes) ||
        !checked_multiply(loaded.offsets_words.size(), sizeof(uint32_t), offset_bytes) ||
        !checked_multiply(payload_word_count, sizeof(uint32_t), payload_bytes) ||
        !checked_add(expected_bytes, scale_bytes, expected_bytes) ||
        !checked_add(expected_bytes, offset_bytes, expected_bytes) ||
        !checked_add(expected_bytes, payload_bytes, expected_bytes)) {
        error = "VBTPACK4 payload size overflows.";
        return false;
    }
    if (expected_bytes != file_bytes) {
        error = expected_bytes > file_bytes ?
                    "VBTPACK4 payload is truncated." :
                    "VBTPACK4 file contains trailing bytes.";
        return false;
    }

    loaded.payload_words.resize(payload_word_count);
    if (payload_word_count > 0u) {
        in.read(reinterpret_cast<char *>(loaded.payload_words.data()),
                static_cast<std::streamsize>(sizeof(uint32_t) * loaded.payload_words.size()));
        if (!in) {
            error = "Failed to read VBT payload pool.";
            return false;
        }
    }

    out_file = std::move(loaded);
    return true;
}

bool is_density_payload(const File &file)
{
    return file.header.profile_type == VBT_FIELD_DENSITY;
}

}  // namespace vbt_cycles_host
