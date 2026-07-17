#include "vbt_cycles_loader.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct TestFile {
    uint32_t version = 4;
    uint32_t width = 8;
    uint32_t height = 8;
    uint32_t depth = 8;
    uint32_t frames = 4;
    uint32_t leaf_size = 8;
    uint32_t coarse_resolution = 4;
    uint32_t max_coarse_keep = 0;
    uint32_t leaf_count = 1;
    uint32_t profile_type = vbt_cycles_host::VBT_FIELD_DENSITY;
    std::vector<float> scales;
    std::vector<uint32_t> offsets{0, 1};
    std::vector<uint32_t> payload{0};
    bool append_trailing_byte = false;
};

template<typename T>
void write_value(std::ofstream &out, const T &value)
{
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

void write_file(const fs::path &path, const TestFile &file)
{
    std::ofstream out(path, std::ios::binary);
    const char magic[8] = {'V', 'B', 'T', 'P', 'A', 'C', 'K', '4'};
    out.write(magic, sizeof(magic));
    write_value(out, file.version);
    write_value(out, file.width);
    write_value(out, file.height);
    write_value(out, file.depth);
    write_value(out, file.frames);
    write_value(out, file.leaf_size);
    write_value(out, file.coarse_resolution);
    write_value(out, file.max_coarse_keep);
    write_value(out, file.leaf_count);
    write_value(out, file.profile_type);
    const uint32_t scale_count = static_cast<uint32_t>(file.scales.size());
    write_value(out, scale_count);
    if (!file.scales.empty()) {
        out.write(reinterpret_cast<const char *>(file.scales.data()),
                  static_cast<std::streamsize>(file.scales.size() * sizeof(float)));
    }
    if (!file.offsets.empty()) {
        out.write(reinterpret_cast<const char *>(file.offsets.data()),
                  static_cast<std::streamsize>(file.offsets.size() * sizeof(uint32_t)));
    }
    if (!file.payload.empty()) {
        out.write(reinterpret_cast<const char *>(file.payload.data()),
                  static_cast<std::streamsize>(file.payload.size() * sizeof(uint32_t)));
    }
    if (file.append_trailing_byte) {
        out.put('\0');
    }
}

void require(bool condition, const std::string &message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_failure(const fs::path &path, const std::string &expected_text)
{
    vbt_cycles_host::File file;
    std::string error;
    require(!vbt_cycles_host::load_pack4(path, file, error), "Expected VBT load to fail");
    require(error.find(expected_text) != std::string::npos,
            "Unexpected VBT error: " + error + ", expected: " + expected_text);
}

}  // namespace

int main()
{
    const fs::path root = fs::temp_directory_path() / "vbt_cycles_loader_tests";
    std::error_code cleanup_error;
    fs::remove_all(root, cleanup_error);
    fs::create_directories(root);

    try {
        TestFile valid;
        const fs::path valid_path = root / "valid.vbtp";
        write_file(valid_path, valid);
        vbt_cycles_host::File loaded;
        std::string error;
        require(vbt_cycles_host::load_pack4(valid_path, loaded, error),
                "Valid VBT failed: " + error);
        require(loaded.header.version == 4u && loaded.header.leaf_count == 1u,
                "Valid VBT header mismatch");

        TestFile bad_version = valid;
        bad_version.version = 3;
        const fs::path bad_version_path = root / "bad_version.vbtp";
        write_file(bad_version_path, bad_version);
        expect_failure(bad_version_path, "Unsupported VBTPACK4 version");

        TestFile bad_leaf_count = valid;
        bad_leaf_count.width = 16;
        const fs::path bad_leaf_count_path = root / "bad_leaf_count.vbtp";
        write_file(bad_leaf_count_path, bad_leaf_count);
        expect_failure(bad_leaf_count_path, "leaf_count does not match");

        TestFile non_monotonic = valid;
        non_monotonic.width = 16;
        non_monotonic.leaf_count = 2;
        non_monotonic.offsets = {0, 2, 1};
        const fs::path non_monotonic_path = root / "non_monotonic.vbtp";
        write_file(non_monotonic_path, non_monotonic);
        expect_failure(non_monotonic_path, "strictly increasing");

        TestFile trailing = valid;
        trailing.append_trailing_byte = true;
        const fs::path trailing_path = root / "trailing.vbtp";
        write_file(trailing_path, trailing);
        expect_failure(trailing_path, "trailing bytes");

        TestFile truncated = valid;
        truncated.offsets = {0, 2};
        const fs::path truncated_path = root / "truncated.vbtp";
        write_file(truncated_path, truncated);
        expect_failure(truncated_path, "truncated");
    }
    catch (...) {
        fs::remove_all(root, cleanup_error);
        throw;
    }

    fs::remove_all(root, cleanup_error);
    std::cout << "vbt_cycles_loader_tests: PASS\n";
    return 0;
}
