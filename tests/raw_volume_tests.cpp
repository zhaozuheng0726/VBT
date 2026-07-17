#include "raw_volume.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

float logicalValue(int x, int y, int z, int t)
{
    return static_cast<float>(1000 * t + 100 * z + 10 * y + x);
}

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void writeMetadata(const fs::path& path,
                   bool timeFastest,
                   const std::string& axisOrder = R"(["X","Y","Z","T"])")
{
    std::ofstream out(path, std::ios::binary);
    out << "{\n"
        << "  \"dimensions\": [3, 2, 2, 4],\n"
        << "  \"axis_order\": " << axisOrder << ",\n"
        << "  \"time_is_fastest_dimension\": " << (timeFastest ? "true" : "false") << ",\n"
        << "  \"data_type\": \"float32\",\n"
        << "  \"global_min\": 0,\n"
        << "  \"global_max\": 3102\n"
        << "}\n";
}

void writeRaw(const fs::path& path, bool timeFastest, bool appendExtra = false)
{
    std::vector<float> values;
    if (timeFastest) {
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 3; ++x)
                    for (int t = 0; t < 4; ++t)
                        values.push_back(logicalValue(x, y, z, t));
    } else {
        for (int t = 0; t < 4; ++t)
            for (int z = 0; z < 2; ++z)
                for (int y = 0; y < 2; ++y)
                    for (int x = 0; x < 3; ++x)
                        values.push_back(logicalValue(x, y, z, t));
    }
    if (appendExtra) values.push_back(-1.0f);

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
}

void verifyLogicalValues(const vbt::RawVolume4D& volume)
{
    require(volume.meta.width == 3 && volume.meta.height == 2 &&
                volume.meta.depth == 2 && volume.meta.frames == 4,
            "Unexpected loaded dimensions");
    for (int t = 0; t < 4; ++t)
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 3; ++x)
                    require(volume.at(x, y, z, t) == logicalValue(x, y, z, t),
                            "Logical coordinate mismatch");
}

void expectFailure(const std::function<void()>& fn, const std::string& expectedText)
{
    try {
        fn();
    } catch (const std::exception& e) {
        require(std::string(e.what()).find(expectedText) != std::string::npos,
                "Failure message did not contain '" + expectedText + "': " + e.what());
        return;
    }
    throw std::runtime_error("Expected operation to fail");
}

} // namespace

int main()
{
    const fs::path root = fs::temp_directory_path() / "vbt_raw_volume_tests";
    std::error_code cleanupError;
    fs::remove_all(root, cleanupError);
    fs::create_directories(root);

    try {
        const fs::path frameRaw = root / "frame_major.raw";
        const fs::path frameMeta = root / "frame_major.metadata.json";
        writeRaw(frameRaw, false);
        writeMetadata(frameMeta, false);
        const auto frameVolume = vbt::loadRawVolume(frameRaw, frameMeta);
        verifyLogicalValues(frameVolume);
        require(frameVolume.meta.rawStorageOrder == vbt::RawStorageOrder::FrameMajor,
                "Frame-major metadata was not retained");
        const auto mappedFrameVolume = vbt::loadRawVolumeMapped(frameRaw, frameMeta);
        verifyLogicalValues(mappedFrameVolume);
        require(mappedFrameVolume.values.empty() && mappedFrameVolume.mappedValues,
                "Frame-major mapped loader materialized the RAW file");

        const fs::path fastestRaw = root / "time_fastest.raw";
        const fs::path fastestMeta = root / "time_fastest.metadata.json";
        writeRaw(fastestRaw, true);
        writeMetadata(fastestMeta, true);
        const auto fastestVolume = vbt::loadRawVolume(fastestRaw, fastestMeta);
        verifyLogicalValues(fastestVolume);
        require(fastestVolume.meta.rawStorageOrder == vbt::RawStorageOrder::TimeFastest,
                "T-fastest metadata was not retained");
        const auto mappedFastestVolume = vbt::loadRawVolumeMapped(fastestRaw, fastestMeta);
        verifyLogicalValues(mappedFastestVolume);
        require(mappedFastestVolume.values.empty() && mappedFastestVolume.mappedValues,
                "T-fastest mapped loader materialized the RAW file");
        require(frameVolume.values == fastestVolume.values,
                "Source layouts did not normalize to one canonical in-memory layout");

        const auto cropped = vbt::cropRawVolume(fastestVolume, 1, 0, 0, 2, 2, 2);
        for (int t = 0; t < 4; ++t)
            for (int z = 0; z < 2; ++z)
                for (int y = 0; y < 2; ++y)
                    for (int x = 0; x < 2; ++x)
                        require(cropped.at(x, y, z, t) == logicalValue(x + 1, y, z, t),
                                "Cropped logical coordinate mismatch");

        const fs::path badAxisMeta = root / "bad_axis.metadata.json";
        writeMetadata(badAxisMeta, true, R"(["T","Z","Y","X"])");
        expectFailure([&] { (void)vbt::loadRawVolume(fastestRaw, badAxisMeta); },
                      "Unsupported metadata axis_order");

        const fs::path extraRaw = root / "extra.raw";
        writeRaw(extraRaw, true, true);
        expectFailure([&] { (void)vbt::loadRawVolume(extraRaw, fastestMeta); },
                      "Raw file size mismatch");

        expectFailure([&] { (void)vbt::cropRawVolume(fastestVolume, 2, 0, 0, 2, 2, 2); },
                      "Crop bounds exceed");
    } catch (...) {
        fs::remove_all(root, cleanupError);
        throw;
    }

    fs::remove_all(root, cleanupError);
    std::cout << "raw_volume_tests: PASS\n";
    return 0;
}
