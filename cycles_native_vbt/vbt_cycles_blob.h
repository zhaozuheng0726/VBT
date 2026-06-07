#pragma once

#include "vbt_cycles_loader.h"
#include "vbt_cycles_sampler.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vbt_cycles_host {

constexpr uint32_t kBlobMagic = 0x31544256u;  // "VBT1" in little-endian byte order.

struct BlobHeader {
    uint32_t magic = kBlobMagic;
    uint32_t version = 1;
    uint32_t dim_x = 0;
    uint32_t dim_y = 0;
    uint32_t dim_z = 0;
    uint32_t frames = 0;
    uint32_t leaf_size = 0;
    uint32_t leaf_count_x = 0;
    uint32_t leaf_count_y = 0;
    uint32_t leaf_count_z = 0;
    uint32_t leaf_count = 0;
    uint32_t frame_index = 0;
    uint32_t offsets_word_count = 0;
    uint32_t payload_word_count = 0;
    uint32_t offsets_byte_offset = 0;
    uint32_t payload_byte_offset = 0;
    float bbox_min_x = 0.0f;
    float bbox_min_y = 0.0f;
    float bbox_min_z = 0.0f;
};

bool pack_blob(const File &file, int frame_index, std::vector<uint8_t> &out_blob, std::string &error);
bool blob_payload_view(const std::vector<uint8_t> &blob, vbt_cycles::PayloadView &out_view, std::string &error);

}  // namespace vbt_cycles_host
