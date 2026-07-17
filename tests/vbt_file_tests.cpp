#include "vbt_file.h"

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
    uint32_t leafSize = 8;
    uint32_t coarseResolution = 4;
    uint32_t maxCoarseKeep = 0;
    uint32_t leafCount = 1;
    uint32_t profileType = 1;
    std::vector<float> scales;
    std::vector<uint32_t> offsets{0, 1};
    std::vector<uint32_t> payload{0};
    bool appendTrailingByte = false;
};

template <typename T>
void writeValue(std::ofstream& out, const T& value)
{
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void writeFile(const fs::path& path, const TestFile& file)
{
    std::ofstream out(path, std::ios::binary);
    const char magic[8] = {'V', 'B', 'T', 'P', 'A', 'C', 'K', '4'};
    out.write(magic, sizeof(magic));
    writeValue(out, file.version);
    writeValue(out, file.width);
    writeValue(out, file.height);
    writeValue(out, file.depth);
    writeValue(out, file.frames);
    writeValue(out, file.leafSize);
    writeValue(out, file.coarseResolution);
    writeValue(out, file.maxCoarseKeep);
    writeValue(out, file.leafCount);
    writeValue(out, file.profileType);
    const uint32_t scaleCount = static_cast<uint32_t>(file.scales.size());
    writeValue(out, scaleCount);
    if (!file.scales.empty()) {
        out.write(reinterpret_cast<const char*>(file.scales.data()),
                  static_cast<std::streamsize>(file.scales.size() * sizeof(float)));
    }
    if (!file.offsets.empty()) {
        out.write(reinterpret_cast<const char*>(file.offsets.data()),
                  static_cast<std::streamsize>(file.offsets.size() * sizeof(uint32_t)));
    }
    if (!file.payload.empty()) {
        out.write(reinterpret_cast<const char*>(file.payload.data()),
                  static_cast<std::streamsize>(file.payload.size() * sizeof(uint32_t)));
    }
    if (file.appendTrailingByte) out.put('\0');
}

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void expectFailure(const fs::path& path, const std::string& expectedText)
{
    vbt::render::VbtFile file;
    std::string error;
    require(!vbt::render::loadVbtFile(path, file, error), "Expected VBT load to fail");
    require(error.find(expectedText) != std::string::npos,
            "Unexpected VBT error: " + error + ", expected: " + expectedText);
}

} // namespace

int main()
{
    const fs::path root = fs::temp_directory_path() / "vbt_file_tests";
    std::error_code cleanupError;
    fs::remove_all(root, cleanupError);
    fs::create_directories(root);

    try {
        TestFile valid;
        const fs::path validPath = root / "valid.vbtp";
        writeFile(validPath, valid);
        vbt::render::VbtFile loaded;
        std::string error;
        require(vbt::render::loadVbtFile(validPath, loaded, error), "Valid VBT failed: " + error);
        require(loaded.header.version == 4 && loaded.header.leafCount == 1,
                "Valid VBT header mismatch");
        require(loaded.offsetsWords == valid.offsets && loaded.payloadWords == valid.payload,
                "Valid VBT payload mismatch");

        TestFile badVersion = valid;
        badVersion.version = 3;
        const fs::path badVersionPath = root / "bad_version.vbtp";
        writeFile(badVersionPath, badVersion);
        expectFailure(badVersionPath, "Unsupported VBTPACK4 version");

        TestFile badLeafCount = valid;
        badLeafCount.width = 16;
        const fs::path badLeafCountPath = root / "bad_leaf_count.vbtp";
        writeFile(badLeafCountPath, badLeafCount);
        expectFailure(badLeafCountPath, "leafCount does not match");

        TestFile nonMonotonic = valid;
        nonMonotonic.width = 16;
        nonMonotonic.leafCount = 2;
        nonMonotonic.offsets = {0, 2, 1};
        const fs::path nonMonotonicPath = root / "non_monotonic.vbtp";
        writeFile(nonMonotonicPath, nonMonotonic);
        expectFailure(nonMonotonicPath, "strictly increasing");

        TestFile emptyLeaf = valid;
        emptyLeaf.offsets = {0, 0};
        emptyLeaf.payload.clear();
        const fs::path emptyLeafPath = root / "empty_leaf.vbtp";
        writeFile(emptyLeafPath, emptyLeaf);
        expectFailure(emptyLeafPath, "strictly increasing");

        TestFile trailing = valid;
        trailing.appendTrailingByte = true;
        const fs::path trailingPath = root / "trailing.vbtp";
        writeFile(trailingPath, trailing);
        expectFailure(trailingPath, "trailing bytes");

        TestFile truncated = valid;
        truncated.offsets = {0, 2};
        const fs::path truncatedPath = root / "truncated.vbtp";
        writeFile(truncatedPath, truncated);
        expectFailure(truncatedPath, "truncated");
    } catch (...) {
        fs::remove_all(root, cleanupError);
        throw;
    }

    fs::remove_all(root, cleanupError);
    std::cout << "vbt_file_tests: PASS\n";
    return 0;
}
