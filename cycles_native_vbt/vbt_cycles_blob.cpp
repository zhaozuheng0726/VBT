#include "vbt_cycles_blob.h"

#include <algorithm>
#include <cstring>

namespace vbt_cycles_host {

namespace {

uint32_t div_round_up(uint32_t value, uint32_t divisor)
{
    return divisor == 0u ? 0u : (value + divisor - 1u) / divisor;
}

size_t align_up4(size_t value)
{
    return (value + 3u) & ~size_t(3u);
}

}  // namespace

bool pack_blob(const File &file, int frame_index, std::vector<uint8_t> &out_blob, std::string &error)
{
    if (file.header.width == 0u || file.header.height == 0u || file.header.depth == 0u ||
        file.header.frames == 0u || file.header.leaf_count == 0u) {
        error = "Invalid empty VBT dimensions or frame count.";
        return false;
    }
    if (file.offsets_words.size() != static_cast<size_t>(file.header.leaf_count) + 1u) {
        error = "Offset table size does not match VBT leaf count.";
        return false;
    }
    if (!file.offsets_words.empty() && file.offsets_words.back() != file.payload_words.size()) {
        error = "Offset table payload size does not match payload word count.";
        return false;
    }

    const uint32_t leaf_size = file.header.leaf_size == 0u ? 8u : file.header.leaf_size;
    const uint32_t leaf_count_x = div_round_up(file.header.width, leaf_size);
    const uint32_t leaf_count_y = div_round_up(file.header.height, leaf_size);
    const uint32_t leaf_count_z = div_round_up(file.header.depth, leaf_size);
    const uint64_t expected_leaf_count = static_cast<uint64_t>(leaf_count_x) *
                                         static_cast<uint64_t>(leaf_count_y) *
                                         static_cast<uint64_t>(leaf_count_z);
    if (expected_leaf_count != file.header.leaf_count) {
        error = "Computed leaf count does not match VBT header.";
        return false;
    }

    const size_t offsets_byte_offset = align_up4(sizeof(BlobHeader));
    const size_t payload_byte_offset = offsets_byte_offset + file.offsets_words.size() * sizeof(uint32_t);
    const size_t blob_bytes = payload_byte_offset + file.payload_words.size() * sizeof(uint32_t);
    if (offsets_byte_offset > UINT32_MAX || payload_byte_offset > UINT32_MAX) {
        error = "VBT blob section offset exceeds the 32-bit header range.";
        return false;
    }

    out_blob.assign(blob_bytes, 0);

    BlobHeader header;
    header.dim_x = file.header.width;
    header.dim_y = file.header.height;
    header.dim_z = file.header.depth;
    header.frames = file.header.frames;
    header.leaf_size = leaf_size;
    header.leaf_count_x = leaf_count_x;
    header.leaf_count_y = leaf_count_y;
    header.leaf_count_z = leaf_count_z;
    header.leaf_count = file.header.leaf_count;
    header.frame_index = static_cast<uint32_t>(std::clamp(frame_index, 0, static_cast<int>(file.header.frames) - 1));
    header.offsets_word_count = static_cast<uint32_t>(file.offsets_words.size());
    header.payload_word_count = static_cast<uint32_t>(file.payload_words.size());
    header.offsets_byte_offset = static_cast<uint32_t>(offsets_byte_offset);
    header.payload_byte_offset = static_cast<uint32_t>(payload_byte_offset);

    std::memcpy(out_blob.data(), &header, sizeof(header));
    std::memcpy(out_blob.data() + offsets_byte_offset,
                file.offsets_words.data(),
                file.offsets_words.size() * sizeof(uint32_t));
    if (!file.payload_words.empty()) {
        std::memcpy(out_blob.data() + payload_byte_offset,
                    file.payload_words.data(),
                    file.payload_words.size() * sizeof(uint32_t));
    }

    return true;
}

bool blob_payload_view(const std::vector<uint8_t> &blob, vbt_cycles::PayloadView &out_view, std::string &error)
{
    if (blob.size() < sizeof(BlobHeader)) {
        error = "Blob is smaller than BlobHeader.";
        return false;
    }

    const auto *blob_header = reinterpret_cast<const BlobHeader *>(blob.data());
    if (blob_header->magic != kBlobMagic || blob_header->version != 1u) {
        error = "Invalid VBT blob header.";
        return false;
    }
    const size_t offsets_end = static_cast<size_t>(blob_header->offsets_byte_offset) +
                               static_cast<size_t>(blob_header->offsets_word_count) * sizeof(uint32_t);
    const size_t payload_end = static_cast<size_t>(blob_header->payload_byte_offset) +
                              static_cast<size_t>(blob_header->payload_word_count) * sizeof(uint32_t);
    if (offsets_end > blob.size() || payload_end > blob.size()) {
        error = "VBT blob offset ranges exceed blob size.";
        return false;
    }

    out_view.header.dim_x = blob_header->dim_x;
    out_view.header.dim_y = blob_header->dim_y;
    out_view.header.dim_z = blob_header->dim_z;
    out_view.header.frames = blob_header->frames;
    out_view.header.leaf_size = blob_header->leaf_size;
    out_view.header.leaf_count_x = blob_header->leaf_count_x;
    out_view.header.leaf_count_y = blob_header->leaf_count_y;
    out_view.header.frame_index = static_cast<int>(blob_header->frame_index);
    out_view.header.bbox_min_x = blob_header->bbox_min_x;
    out_view.header.bbox_min_y = blob_header->bbox_min_y;
    out_view.header.bbox_min_z = blob_header->bbox_min_z;
    out_view.offset_words = reinterpret_cast<const uint32_t *>(blob.data() + blob_header->offsets_byte_offset);
    out_view.payload_words = reinterpret_cast<const uint32_t *>(blob.data() + blob_header->payload_byte_offset);
    return true;
}

}  // namespace vbt_cycles_host
