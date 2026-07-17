#include "frame_metadata.h"
#include "raw_frame_reader.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void writeText(const fs::path& path, const std::string& text)
{
    std::ofstream output(path, std::ios::binary);
    output << text;
}

void writeFloats(const fs::path& path, const std::vector<float>& values)
{
    std::ofstream output(path, std::ios::binary);
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
}

void expectFailure(const std::function<void()>& operation, const std::string& expectedText)
{
    try {
        operation();
    } catch (const std::exception& error) {
        require(
            std::string(error.what()).find(expectedText) != std::string::npos,
            "Unexpected error: " + std::string(error.what()) +
                ", expected text: " + expectedText);
        return;
    }
    throw std::runtime_error("Expected operation to fail: " + expectedText);
}

std::string namedMetadata(const std::string& extra = {})
{
    return
        "{\n"
        "  \"source_dir\": \"source\",\n"
        "  \"grid_name\": \"surface\",\n"
        "  \"conversion_mode\": \"levelset\",\n"
        "  \"width\": 3,\n"
        "  \"height\": 2,\n"
        "  \"depth\": 2,\n"
        "  \"frames\": 4,\n"
        "  \"bbox_min\": [-1, 5, 9],\n"
        "  \"bbox_max\": [1, 6, 10],\n"
        "  \"data_min\": -3,\n"
        "  \"data_max\": 3,\n"
        "  \"voxel_size\": 0.5,\n"
        "  \"index_to_world\": [0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, -2, 3, 4, 1],\n"
        "  \"shell_width_voxels\": 1.5,\n"
        "  \"half_width_voxels\": 3" +
        extra +
        "\n}\n";
}

float logicalValue(int x, int y, int z, int t)
{
    return static_cast<float>(1000 * t + 100 * z + 10 * y + x);
}

void verifyFrame(const std::vector<float>& frame, int frameIndex)
{
    require(frame.size() == 12, "Unexpected frame voxel count");
    for (int z = 0; z < 2; ++z) {
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 3; ++x) {
                const size_t index =
                    (static_cast<size_t>(z) * 2u + static_cast<size_t>(y)) * 3u +
                    static_cast<size_t>(x);
                require(
                    frame[index] == logicalValue(x, y, z, frameIndex),
                    "Decoded RAW frame contains an incorrect logical value");
            }
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    const fs::path root = fs::temp_directory_path() / "vdb_frame_metadata_tests";
    std::error_code cleanupError;
    fs::remove_all(root, cleanupError);
    fs::create_directories(root);

    try {
        const fs::path namedPath = root / "named.json";
        writeText(
            namedPath,
            namedMetadata(
                ",\n  \"frame_files\": [\"a\", \"b\", \"c\", \"d\"]"));
        const auto named = vdbtools::loadFrameMetadata(namedPath);
        require(
            named.width == 3 && named.height == 2 && named.depth == 2 && named.frames == 4,
            "Named dimensions were not loaded");
        require(named.bboxMin == std::array<int, 3>{-1, 5, 9}, "bbox_min was not loaded");
        require(named.bboxMax == std::array<int, 3>{1, 6, 10}, "bbox_max was not loaded");
        require(named.conversionMode == "levelset", "conversion_mode was not loaded");
        require(named.indexToWorldDeclared, "index_to_world declaration was not retained");
        require(named.indexToWorld[12] == -2.0 && named.indexToWorld[13] == 3.0 &&
                    named.indexToWorld[14] == 4.0,
                "index_to_world translation was not loaded");

        const fs::path dimensionsPath = root / "dimensions.json";
        writeText(
            dimensionsPath,
            "{\n"
            "  \"dimensions\": [3, 2, 2, 4],\n"
            "  \"axis_order\": [\"x\", \"y\", \"z\", \"t\"],\n"
            "  \"time_is_fastest_dimension\": true,\n"
            "  \"data_type\": \"float32\",\n"
            "  \"global_min\": 0,\n"
            "  \"global_max\": 3102\n"
            "}\n");
        const auto dimensions = vdbtools::loadFrameMetadata(dimensionsPath);
        require(dimensions.axisOrderDeclared, "axis_order declaration was not retained");
        require(dimensions.timeFastestDeclared, "T-fastest declaration was not retained");
        require(
            dimensions.rawStorageOrder == vdbtools::RawStorageOrder::TimeFastest,
            "T-fastest storage order was not retained");
        require(dimensions.bboxMin == std::array<int, 3>{0, 0, 0}, "Default bbox_min is wrong");
        require(dimensions.bboxMax == std::array<int, 3>{2, 1, 1}, "Default bbox_max is wrong");

        const fs::path malformedPath = root / "malformed.json";
        writeText(malformedPath, "{\"width\":");
        expectFailure(
            [&] { (void)vdbtools::loadFrameMetadata(malformedPath); },
            "Failed to parse metadata JSON");

        const fs::path mismatchPath = root / "mismatch.json";
        writeText(
            mismatchPath,
            "{\"width\":4,\"dimensions\":[3,2,2,4]}");
        expectFailure(
            [&] { (void)vdbtools::loadFrameMetadata(mismatchPath); },
            "disagree with dimensions[]");

        const fs::path badBboxPath = root / "bad_bbox.json";
        writeText(
            badBboxPath,
            "{\"dimensions\":[3,2,2,4],\"bbox_min\":[0,0,0],\"bbox_max\":[3,1,1]}");
        expectFailure(
            [&] { (void)vdbtools::loadFrameMetadata(badBboxPath); },
            "bbox extent disagrees");

        const fs::path missingBboxPath = root / "missing_bbox.json";
        writeText(
            missingBboxPath,
            "{\"dimensions\":[3,2,2,4],\"bbox_min\":[0,0,0]}");
        expectFailure(
            [&] { (void)vdbtools::loadFrameMetadata(missingBboxPath); },
            "must be provided together");

        const fs::path badModePath = root / "bad_mode.json";
        writeText(
            badModePath,
            "{\"dimensions\":[3,2,2,4],\"conversion_mode\":\"unknown\"}");
        expectFailure(
            [&] { (void)vdbtools::loadFrameMetadata(badModePath); },
            "Unsupported metadata conversion_mode");

        const fs::path badHalfWidthPath = root / "bad_half_width.json";
        writeText(
            badHalfWidthPath,
            "{\"dimensions\":[3,2,2,4],\"half_width_voxels\":0}");
        expectFailure(
            [&] { (void)vdbtools::loadFrameMetadata(badHalfWidthPath); },
            "half_width_voxels must be positive");

        const fs::path badFramesPath = root / "bad_frame_files.json";
        writeText(
            badFramesPath,
            "{\"dimensions\":[3,2,2,4],\"frame_files\":[\"a\"]}");
        expectFailure(
            [&] { (void)vdbtools::loadFrameMetadata(badFramesPath); },
            "one entry per frame");

        std::vector<float> frameMajorValues;
        for (int t = 0; t < 4; ++t)
            for (int z = 0; z < 2; ++z)
                for (int y = 0; y < 2; ++y)
                    for (int x = 0; x < 3; ++x)
                        frameMajorValues.push_back(logicalValue(x, y, z, t));
        const fs::path frameMajorRaw = root / "frame_major.raw";
        writeFloats(frameMajorRaw, frameMajorValues);
        auto frameMajorMeta = dimensions;
        frameMajorMeta.rawStorageOrder = vdbtools::RawStorageOrder::FrameMajor;
        verifyFrame(vdbtools::loadRawFrame(frameMajorRaw, frameMajorMeta, 2), 2);

        std::vector<float> timeFastestValues;
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 3; ++x)
                    for (int t = 0; t < 4; ++t)
                        timeFastestValues.push_back(logicalValue(x, y, z, t));
        const fs::path timeFastestRaw = root / "time_fastest.raw";
        writeFloats(timeFastestRaw, timeFastestValues);
        verifyFrame(vdbtools::loadRawFrame(timeFastestRaw, dimensions, 2), 2);

        timeFastestValues.push_back(-1.0f);
        const fs::path trailingRaw = root / "trailing.raw";
        writeFloats(trailingRaw, timeFastestValues);
        expectFailure(
            [&] { (void)vdbtools::loadRawFrame(trailingRaw, dimensions, 0); },
            "RAW file size mismatch");
    } catch (...) {
        fs::remove_all(root, cleanupError);
        throw;
    }

    fs::remove_all(root, cleanupError);
    for (int i = 1; i < argc; ++i) {
        const fs::path metadataPath = argv[i];
        const auto metadata = vdbtools::loadFrameMetadata(metadataPath);
        const std::string filename = metadataPath.filename().string();
        constexpr const char* suffix = ".metadata.json";
        require(
            filename.size() > std::char_traits<char>::length(suffix) &&
                filename.compare(
                    filename.size() - std::char_traits<char>::length(suffix),
                    std::char_traits<char>::length(suffix),
                    suffix) == 0,
            "Repository metadata filename must end with .metadata.json");
        const fs::path rawPath =
            metadataPath.parent_path() /
            (filename.substr(
                 0, filename.size() - std::char_traits<char>::length(suffix)) +
             ".raw");
        vdbtools::validateRawFileSize(rawPath, metadata);
        std::cout << "validated: " << metadataPath.string()
                  << " (" << metadata.width << " x " << metadata.height
                  << " x " << metadata.depth << " x " << metadata.frames
                  << ", raw bytes " << vdbtools::rawByteCount(metadata) << ")\n";
    }
    std::cout << "frame_metadata_tests: PASS";
    if (argc > 1) {
        std::cout << " (" << (argc - 1) << " repository metadata files)";
    }
    std::cout << "\n";
    return 0;
}
