#include "vbt_cycles_loader.h"

#include <cstring>
#include <fstream>

namespace vbt_cycles_host {

namespace {

template<typename T>
bool read_value(std::ifstream &in, T &value)
{
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    return static_cast<bool>(in);
}

}  // namespace

bool load_pack4(const std::filesystem::path &path, File &out_file, std::string &error)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "Failed to open VBT file: " + path.string();
        return false;
    }

    if (!read_value(in, out_file.header.magic) ||
        !read_value(in, out_file.header.version) ||
        !read_value(in, out_file.header.width) ||
        !read_value(in, out_file.header.height) ||
        !read_value(in, out_file.header.depth) ||
        !read_value(in, out_file.header.frames) ||
        !read_value(in, out_file.header.leaf_size) ||
        !read_value(in, out_file.header.coarse_resolution) ||
        !read_value(in, out_file.header.max_coarse_keep) ||
        !read_value(in, out_file.header.leaf_count) ||
        !read_value(in, out_file.header.profile_type) ||
        !read_value(in, out_file.header.coarse_ac_scale_count)) {
        error = "Failed to read VBTPACK4 header.";
        return false;
    }

    if (std::strncmp(out_file.header.magic, "VBTPACK4", 8) != 0) {
        error = "Unsupported VBT magic. Expected VBTPACK4.";
        return false;
    }

    out_file.coarse_ac_scales.resize(out_file.header.coarse_ac_scale_count);
    if (!out_file.coarse_ac_scales.empty()) {
        in.read(reinterpret_cast<char *>(out_file.coarse_ac_scales.data()),
                static_cast<std::streamsize>(sizeof(float) * out_file.coarse_ac_scales.size()));
        if (!in) {
            error = "Failed to read coarse AC scales.";
            return false;
        }
    }

    out_file.offsets_words.resize(static_cast<size_t>(out_file.header.leaf_count) + 1u);
    in.read(reinterpret_cast<char *>(out_file.offsets_words.data()),
            static_cast<std::streamsize>(sizeof(uint32_t) * out_file.offsets_words.size()));
    if (!in) {
        error = "Failed to read VBT offset table.";
        return false;
    }

    const uint32_t payload_word_count = out_file.offsets_words.empty() ? 0u : out_file.offsets_words.back();
    out_file.payload_words.resize(payload_word_count);
    if (payload_word_count > 0u) {
        in.read(reinterpret_cast<char *>(out_file.payload_words.data()),
                static_cast<std::streamsize>(sizeof(uint32_t) * out_file.payload_words.size()));
        if (!in) {
            error = "Failed to read VBT payload pool.";
            return false;
        }
    }

    return true;
}

bool is_density_payload(const File &file)
{
    return file.header.profile_type == VBT_FIELD_DENSITY;
}

}  // namespace vbt_cycles_host
