#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/Dense.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void traceLine(const std::string& s)
{
    std::ofstream trace("vdb_to_raw.trace.txt", std::ios::app | std::ios::binary);
    trace << s << "\n";
}

struct Options {
    fs::path inputDir;
    fs::path outputPrefix;
    std::string gridName = "density";
    bool listOnly = false;
    bool timeFastest = false;
};

struct FrameInfo {
    fs::path path;
    openvdb::CoordBBox bbox;
    float background = 0.0f;
};

struct FrameSortKey {
    std::string parent;
    std::string prefix;
    std::string filename;
    std::uint64_t frame = 0;
    bool hasFrame = false;
};

static FrameSortKey makeFrameSortKey(const fs::path& path)
{
    FrameSortKey key;
    key.parent = path.parent_path().generic_string();
    key.filename = path.filename().generic_string();

    const std::string stem = path.stem().string();
    size_t digitBegin = stem.size();
    while (digitBegin > 0 && std::isdigit(static_cast<unsigned char>(stem[digitBegin - 1]))) {
        --digitBegin;
    }
    key.prefix = stem.substr(0, digitBegin);
    if (digitBegin == stem.size()) return key;

    key.hasFrame = true;
    for (size_t i = digitBegin; i < stem.size(); ++i) {
        key.frame = key.frame * 10u + static_cast<std::uint64_t>(stem[i] - '0');
    }
    return key;
}

static bool framePathLess(const fs::path& lhs, const fs::path& rhs)
{
    const auto a = makeFrameSortKey(lhs);
    const auto b = makeFrameSortKey(rhs);
    if (a.parent != b.parent) return a.parent < b.parent;
    if (a.prefix != b.prefix) return a.prefix < b.prefix;
    if (a.hasFrame != b.hasFrame) return a.hasFrame;
    if (a.hasFrame && a.frame != b.frame) return a.frame < b.frame;
    return a.filename < b.filename;
}

static void printUsage()
{
    std::cout <<
        "Usage:\n"
        "  vdb_to_raw.exe --input-dir <dir> --output-prefix <prefix> [--grid density]\n"
        "                 [--time-fastest] [--list]\n\n"
        "Notes:\n"
        "  - Reads a directory of .vdb frames.\n"
        "  - Exports dense float32 raw in index space using the union active voxel bbox.\n"
        "  - Writes <prefix>.raw and <prefix>.metadata.json\n";
}

static bool parseArgs(int argc, char** argv, Options& opt)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--input-dir" && i + 1 < argc) {
            opt.inputDir = argv[++i];
        } else if (arg == "--output-prefix" && i + 1 < argc) {
            opt.outputPrefix = argv[++i];
        } else if (arg == "--grid" && i + 1 < argc) {
            opt.gridName = argv[++i];
        } else if (arg == "--list") {
            opt.listOnly = true;
        } else if (arg == "--time-fastest") {
            opt.timeFastest = true;
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            printUsage();
            return false;
        }
    }
    if (opt.inputDir.empty()) {
        std::cerr << "--input-dir is required\n";
        return false;
    }
    if (!opt.listOnly && opt.outputPrefix.empty()) {
        std::cerr << "--output-prefix is required unless --list is used\n";
        return false;
    }
    return true;
}

static std::vector<fs::path> collectFrames(const fs::path& inputDir)
{
    std::vector<fs::path> frames;
    for (const auto& entry : fs::recursive_directory_iterator(inputDir)) {
        if (!entry.is_regular_file()) continue;
        const auto ext = entry.path().extension().string();
        if (_stricmp(ext.c_str(), ".vdb") == 0) {
            frames.push_back(entry.path());
        }
    }
    std::sort(frames.begin(), frames.end(), framePathLess);
    return frames;
}

static std::string chooseGridName(openvdb::io::File& file, const std::string& requested)
{
    if (!requested.empty()) {
        for (auto it = file.beginName(); it != file.endName(); ++it) {
            const auto name = *it;
            if (name != requested) continue;
            auto base = file.readGridMetadata(name);
            if (base && base->isType<openvdb::FloatGrid>()) return name;
        }
    }
    for (auto it = file.beginName(); it != file.endName(); ++it) {
        const auto name = *it;
        auto base = file.readGridMetadata(name);
        if (base && base->isType<openvdb::FloatGrid>()) return name;
    }
    return {};
}

static std::vector<std::string> listFloatGridNames(openvdb::io::File& file)
{
    std::vector<std::string> names;
    for (auto it = file.beginName(); it != file.endName(); ++it) {
        const auto name = *it;
        auto base = file.readGridMetadata(name);
        if (base && base->isType<openvdb::FloatGrid>()) names.push_back(name);
    }
    return names;
}

static std::string jsonEscape(const std::string& s)
{
    std::ostringstream oss;
    for (char c : s) {
        switch (c) {
        case '\\': oss << "\\\\"; break;
        case '"':  oss << "\\\""; break;
        case '\n': oss << "\\n"; break;
        case '\r': oss << "\\r"; break;
        case '\t': oss << "\\t"; break;
        default:   oss << c; break;
        }
    }
    return oss.str();
}

static void transposeFrameMajorToTimeFastest(const fs::path& frameMajorPath,
                                             const fs::path& outputPath,
                                             size_t frameVoxelCount,
                                             int frames)
{
    constexpr size_t kTargetChunkBytes = 512ull * 1024ull * 1024ull;
    const size_t frameCount = static_cast<size_t>(frames);
    const size_t bytesPerSpatialVoxel = frameCount * sizeof(float);
    const size_t spatialChunk = std::max<size_t>(1, kTargetChunkBytes / bytesPerSpatialVoxel);

    std::ifstream input(frameMajorPath, std::ios::binary);
    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!input || !output) {
        throw std::runtime_error("Failed to open RAW transpose input/output");
    }

    std::vector<float> frameChunk;
    std::vector<float> timeFastestChunk;
    for (size_t spatialBase = 0; spatialBase < frameVoxelCount; spatialBase += spatialChunk) {
        const size_t spatialCount = std::min(spatialChunk, frameVoxelCount - spatialBase);
        frameChunk.resize(spatialCount);
        timeFastestChunk.resize(spatialCount * frameCount);

        for (size_t frame = 0; frame < frameCount; ++frame) {
            const size_t valueOffset = frame * frameVoxelCount + spatialBase;
            const size_t byteOffset = valueOffset * sizeof(float);
            if (byteOffset > static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
                throw std::runtime_error("RAW transpose offset exceeds streamoff");
            }
            input.seekg(static_cast<std::streamoff>(byteOffset), std::ios::beg);
            input.read(reinterpret_cast<char*>(frameChunk.data()),
                       static_cast<std::streamsize>(spatialCount * sizeof(float)));
            if (!input) {
                throw std::runtime_error("Failed to read frame-major RAW transpose chunk");
            }
            for (size_t local = 0; local < spatialCount; ++local) {
                timeFastestChunk[local * frameCount + frame] = frameChunk[local];
            }
        }

        output.write(reinterpret_cast<const char*>(timeFastestChunk.data()),
                     static_cast<std::streamsize>(timeFastestChunk.size() * sizeof(float)));
        if (!output) {
            throw std::runtime_error("Failed to write time-fastest RAW transpose chunk");
        }
        std::cout << "Transposed spatial voxels " << (spatialBase + spatialCount)
                  << "/" << frameVoxelCount << "\n";
    }
}

int main(int argc, char** argv)
{
    traceLine("main:begin");
    Options opt;
    if (!parseArgs(argc, argv, opt)) {
        traceLine("main:parseArgs_failed");
        return 1;
    }
    traceLine("main:parsed");

    auto framePaths = collectFrames(opt.inputDir);
    traceLine("main:frames=" + std::to_string(framePaths.size()));
    if (framePaths.empty()) {
        std::cerr << "No .vdb files found in: " << opt.inputDir << "\n";
        traceLine("main:no_frames");
        return 2;
    }

    openvdb::initialize();
    traceLine("main:openvdb_initialized");

    std::vector<FrameInfo> frames;
    frames.reserve(framePaths.size());

    openvdb::CoordBBox unionBBox;
    bool haveUnionBBox = false;
    float globalMin = std::numeric_limits<float>::infinity();
    float globalMax = -std::numeric_limits<float>::infinity();
    openvdb::math::Mat4d indexToWorld = openvdb::math::Mat4d::identity();
    bool haveIndexToWorld = false;

    for (const auto& path : framePaths) {
        traceLine("frame:open:" + path.string());
        openvdb::io::File file(path.string());
        file.open(false);
        traceLine("frame:opened");
        auto floatGridNames = listFloatGridNames(file);
        auto gridName = chooseGridName(file, opt.gridName);
        if (gridName.empty()) {
            std::cerr << "No matching float grid found in: " << path << "\n";
            if (!floatGridNames.empty()) {
                std::cerr << "Available FloatGrid names:";
                for (const auto& name : floatGridNames) {
                    std::cerr << " " << name;
                }
                std::cerr << "\n";
            }
            file.close();
            return 3;
        }

        traceLine("frame:grid=" + gridName);
        auto base = file.readGridMetadata(gridName);
        auto gridMetadata = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
        if (!gridMetadata) {
            std::cerr << "Selected grid is not FloatGrid in: " << path << "\n";
            file.close();
            return 4;
        }

        const auto frameIndexToWorld =
            gridMetadata->transform().baseMap()->getAffineMap()->getMat4();
        if (!haveIndexToWorld) {
            indexToWorld = frameIndexToWorld;
            haveIndexToWorld = true;
        } else {
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    if (std::abs(indexToWorld(row, column) - frameIndexToWorld(row, column)) > 1.0e-9) {
                        std::cerr << "Grid transform differs across frames: " << path << "\n";
                        file.close();
                        return 10;
                    }
                }
            }
        }

        openvdb::CoordBBox bbox;
        try {
            const auto bboxMin = base->metaValue<openvdb::Vec3i>(openvdb::GridBase::META_FILE_BBOX_MIN);
            const auto bboxMax = base->metaValue<openvdb::Vec3i>(openvdb::GridBase::META_FILE_BBOX_MAX);
            bbox = openvdb::CoordBBox(openvdb::Coord(bboxMin.x(), bboxMin.y(), bboxMin.z()),
                                      openvdb::Coord(bboxMax.x(), bboxMax.y(), bboxMax.z()));
            traceLine("frame:bbox_metadata");
        } catch (const openvdb::Exception&) {
            auto fullGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(gridName));
            if (!fullGrid) {
                std::cerr << "Failed to read FloatGrid in: " << path << "\n";
                file.close();
                return 4;
            }
            bbox = fullGrid->evalActiveVoxelBoundingBox();
            traceLine("frame:bbox_grid_fallback");
        }
        if (!bbox.empty()) {
            if (!haveUnionBBox) {
                unionBBox = bbox;
                haveUnionBBox = true;
            } else {
                unionBBox.expand(bbox);
            }
        }

        FrameInfo info;
        info.path = path;
        info.bbox = bbox;
        info.background = gridMetadata->background();
        frames.push_back(std::move(info));
        file.close();
        traceLine("frame:closed");
    }

    if (!haveUnionBBox) {
        std::cerr << "All frames are empty; no active bbox found.\n";
        return 5;
    }

    const auto min = unionBBox.min();
    const auto max = unionBBox.max();
    const int width = max.x() - min.x() + 1;
    const int height = max.y() - min.y() + 1;
    const int depth = max.z() - min.z() + 1;
    const int numFrames = static_cast<int>(frames.size());

    std::cout << "Frames: " << numFrames << "\n";
    std::cout << "Grid: " << opt.gridName << "\n";
    std::cout << "Union bbox: [(" << min.x() << "," << min.y() << "," << min.z() << ") -> ("
              << max.x() << "," << max.y() << "," << max.z() << ")]\n";
    std::cout << "Dense dims: " << width << " x " << height << " x " << depth << "\n";
    std::cout << "Index-to-world:";
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            std::cout << " " << std::setprecision(17) << indexToWorld(row, column);
        }
    }
    std::cout << "\n";

    if (opt.listOnly) {
        traceLine("main:list_only");
        {
            openvdb::io::File file(frames.front().path.string());
            file.open(false);
            auto names = listFloatGridNames(file);
            file.close();
            std::cout << "Float grids in first frame:";
            for (const auto& name : names) {
                std::cout << " " << name;
            }
            std::cout << "\n";
        }
        for (const auto& f : frames) {
            std::cout << "  " << f.path.filename().string()
                      << "  bg=" << f.background
                      << "  bbox=[(" << f.bbox.min().x() << "," << f.bbox.min().y() << "," << f.bbox.min().z()
                      << ") -> (" << f.bbox.max().x() << "," << f.bbox.max().y() << "," << f.bbox.max().z() << ")]\n";
        }
        traceLine("main:list_done");
        return 0;
    }

    const auto outputDir = opt.outputPrefix.parent_path();
    if (!outputDir.empty()) fs::create_directories(outputDir);
    const fs::path rawPath = opt.outputPrefix.string() + ".raw";
    const fs::path frameMajorPath = opt.timeFastest
        ? fs::path(rawPath.string() + ".frame_major.tmp")
        : rawPath;
    const fs::path metaPath = opt.outputPrefix.string() + ".metadata.json";

    std::ofstream raw(frameMajorPath, std::ios::binary);
    if (!raw) {
        std::cerr << "Failed to open raw output: " << frameMajorPath << "\n";
        return 6;
    }

    const size_t frameVoxelCount = static_cast<size_t>(width) * height * depth;
    std::vector<float> buffer(frameVoxelCount, 0.0f);
    openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> dense(unionBBox, buffer.data());

    for (int fi = 0; fi < numFrames; ++fi) {
        traceLine("write:frame=" + std::to_string(fi));
        openvdb::io::File file(frames[fi].path.string());
        file.open(false);
        const auto gridName = chooseGridName(file, opt.gridName);
        if (gridName.empty()) {
            std::cerr << "No matching float grid found while writing: " << frames[fi].path << "\n";
            file.close();
            return 8;
        }
        auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(gridName));
        file.close();
        if (!grid) {
            std::cerr << "Selected grid is not FloatGrid while writing: " << frames[fi].path << "\n";
            return 9;
        }
        openvdb::tools::copyToDense(*grid, dense);
        const auto frameMinMax = std::minmax_element(buffer.begin(), buffer.end());
        globalMin = std::min(globalMin, *frameMinMax.first);
        globalMax = std::max(globalMax, *frameMinMax.second);

        raw.write(reinterpret_cast<const char*>(buffer.data()),
                  static_cast<std::streamsize>(buffer.size() * sizeof(float)));
        std::cout << "Wrote frame " << (fi + 1) << "/" << numFrames << ": "
                  << frames[fi].path.filename().string() << "\n";
    }
    raw.close();

    if (opt.timeFastest) {
        transposeFrameMajorToTimeFastest(frameMajorPath, rawPath, frameVoxelCount, numFrames);
        std::error_code removeError;
        fs::remove(frameMajorPath, removeError);
        if (removeError) {
            throw std::runtime_error("Failed to remove frame-major temporary RAW: " +
                                     frameMajorPath.string());
        }
    }

    std::ofstream meta(metaPath, std::ios::binary);
    if (!meta) {
        std::cerr << "Failed to open metadata output: " << metaPath << "\n";
        return 7;
    }

    meta << "{\n";
    meta << "  \"source_dir\": \"" << jsonEscape(opt.inputDir.string()) << "\",\n";
    meta << "  \"grid_name\": \"" << jsonEscape(opt.gridName) << "\",\n";
    meta << "  \"format\": \"float32_raw_dense_index_space\",\n";
    meta << "  \"data_type\": \"float32\",\n";
    meta << "  \"axis_order\": [\"X\", \"Y\", \"Z\", \"T\"],\n";
    meta << "  \"time_is_fastest_dimension\": " << (opt.timeFastest ? "true" : "false") << ",\n";
    meta << "  \"index_to_world\": [";
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            if (row != 0 || column != 0) meta << ", ";
            meta << std::setprecision(17) << indexToWorld(row, column);
        }
    }
    meta << "],\n";
    meta << "  \"width\": " << width << ",\n";
    meta << "  \"height\": " << height << ",\n";
    meta << "  \"depth\": " << depth << ",\n";
    meta << "  \"frames\": " << numFrames << ",\n";
    meta << "  \"bbox_min\": [" << min.x() << ", " << min.y() << ", " << min.z() << "],\n";
    meta << "  \"bbox_max\": [" << max.x() << ", " << max.y() << ", " << max.z() << "],\n";
    meta << "  \"data_min\": " << std::setprecision(9) << globalMin << ",\n";
    meta << "  \"data_max\": " << std::setprecision(9) << globalMax << ",\n";
    meta << "  \"frame_files\": [\n";
    for (int i = 0; i < numFrames; ++i) {
        meta << "    \"" << jsonEscape(frames[i].path.filename().string()) << "\"";
        meta << (i + 1 == numFrames ? "\n" : ",\n");
    }
    meta << "  ]\n";
    meta << "}\n";
    meta.close();

    std::cout << "RAW written: " << rawPath << "\n";
    std::cout << "RAW storage order: " << (opt.timeFastest ? "time-fastest" : "frame-major") << "\n";
    std::cout << "Metadata written: " << metaPath << "\n";
    traceLine("main:done");
    return 0;
}
