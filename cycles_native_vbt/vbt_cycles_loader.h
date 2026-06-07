#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vbt_cycles_host {

enum : uint32_t {
    VBT_FIELD_SDF = 0u,
    VBT_FIELD_DENSITY = 1u,
    VBT_FIELD_GENERIC = 2u,
};

struct FileHeader {
    char magic[8]{};
    uint32_t version = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
    uint32_t frames = 0;
    uint32_t leaf_size = 0;
    uint32_t coarse_resolution = 0;
    uint32_t max_coarse_keep = 0;
    uint32_t leaf_count = 0;
    uint32_t profile_type = 0;
    uint32_t coarse_ac_scale_count = 0;
};

struct File {
    FileHeader header;
    std::vector<float> coarse_ac_scales;
    std::vector<uint32_t> offsets_words;
    std::vector<uint32_t> payload_words;
};

bool load_pack4(const std::filesystem::path &path, File &out_file, std::string &error);
bool is_density_payload(const File &file);

}  // namespace vbt_cycles_host
