#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/MeshToVolume.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "vdb_tools/frame_metadata.h"
#include "compare/openvdb-master/openvdb_cmd/vdb_tool/include/Geometry.h"

namespace fs = std::filesystem;
using Geometry = openvdb::vdb_tool::Geometry;

struct Options {
    fs::path inputDir;
    fs::path outputPrefix;
    int dim = 128;
    float halfWidth = 3.0f;
    std::string gridName = "density";
    std::string mode = "fog";
    float shellWidth = 1.5f;
};

struct FrameInfo {
    fs::path path;
    openvdb::CoordBBox bbox;
};

static void printUsage()
{
    std::cout
        << "Usage:\n"
        << "  objseq_to_raw.exe --input-dir <dir> --output-prefix <prefix> [--dim 128] [--half-width 3] [--grid density]\n"
        << "                    [--mode fog|interior|shell|levelset] [--shell-width 1.5]\n";
}

static bool parseArgs(int argc, char** argv, Options& opt)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--input-dir" && i + 1 < argc) {
            opt.inputDir = argv[++i];
        } else if (arg == "--output-prefix" && i + 1 < argc) {
            opt.outputPrefix = argv[++i];
        } else if (arg == "--dim" && i + 1 < argc) {
            opt.dim = std::stoi(argv[++i]);
        } else if (arg == "--half-width" && i + 1 < argc) {
            opt.halfWidth = std::stof(argv[++i]);
        } else if (arg == "--grid" && i + 1 < argc) {
            opt.gridName = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            opt.mode = argv[++i];
        } else if (arg == "--shell-width" && i + 1 < argc) {
            opt.shellWidth = std::stof(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            printUsage();
            return false;
        }
    }
    if (opt.inputDir.empty() || opt.outputPrefix.empty()) {
        printUsage();
        return false;
    }
    if (opt.dim <= 0 || opt.halfWidth <= 0.0f) {
        std::cerr << "Invalid --dim or --half-width\n";
        return false;
    }
    if (opt.mode != "fog" && opt.mode != "interior" && opt.mode != "shell" && opt.mode != "levelset") {
        std::cerr << "Invalid --mode, expected fog|interior|shell|levelset\n";
        return false;
    }
    if (opt.shellWidth <= 0.0f) {
        std::cerr << "Invalid --shell-width\n";
        return false;
    }
    return true;
}

static std::vector<fs::path> collectObjFrames(const fs::path& inputDir)
{
    std::vector<fs::path> frames;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (!entry.is_regular_file()) continue;
        const std::string ext = entry.path().extension().string();
        if (_stricmp(ext.c_str(), ".obj") == 0) {
            frames.push_back(entry.path());
        }
    }
    std::sort(frames.begin(), frames.end());
    return frames;
}

static std::string jsonEscape(const std::string& s)
{
    std::ostringstream oss;
    for (char c : s) {
        switch (c) {
        case '\\': oss << "\\\\"; break;
        case '"': oss << "\\\""; break;
        case '\n': oss << "\\n"; break;
        case '\r': oss << "\\r"; break;
        case '\t': oss << "\\t"; break;
        default: oss << c; break;
        }
    }
    return oss.str();
}

static float estimateVoxelSize(const openvdb::math::BBox<openvdb::Vec3s>& bbox, int maxDim, float halfWidth)
{
    const auto ext = bbox.extents();
    const float longest = std::max({ext.x(), ext.y(), ext.z()});
    const float usable = std::max(1.0f, static_cast<float>(maxDim) - 2.0f * halfWidth);
    return longest / usable;
}

static openvdb::FloatGrid::Ptr buildDensityGrid(const fs::path& objPath, const Options& opt, const openvdb::math::Transform& xform)
{
    Geometry geo;
    geo.read(objPath.string());
    auto sdf = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(xform, geo.vtx(), geo.tri(), geo.quad(), opt.halfWidth);
    sdf->setName(opt.gridName);

    if (opt.mode == "levelset") {
        sdf->setGridClass(openvdb::GRID_LEVEL_SET);
        return sdf;
    }

    if (opt.mode == "fog") {
        openvdb::tools::sdfToFogVolume(*sdf);
        sdf->setGridClass(openvdb::GRID_FOG_VOLUME);
        return sdf;
    }

    auto density = openvdb::FloatGrid::create(/*background=*/0.0f);
    density->setTransform(sdf->transform().copy());
    density->setName(opt.gridName);
    density->setGridClass(openvdb::GRID_FOG_VOLUME);

    if (opt.mode == "interior") {
        auto mask = openvdb::tools::sdfInteriorMask(*sdf, 0.0f);
        for (auto iter = mask->cbeginValueOn(); iter.test(); ++iter) {
            density->tree().setValueOn(iter.getCoord(), 1.0f);
        }
        return density;
    }

    const float voxelSize = static_cast<float>(xform.voxelSize()[0]);
    const float shellDistance = std::max(1.0e-6f, opt.shellWidth * voxelSize);
    for (auto iter = sdf->cbeginValueOn(); iter.test(); ++iter) {
        const float phi = *iter;
        if (phi > 0.0f) continue;
        const float dist = std::abs(phi);
        if (dist > shellDistance) continue;
        const float w = 1.0f - dist / shellDistance;
        density->tree().setValueOn(iter.getCoord(), std::max(0.0f, std::min(1.0f, w)));
    }
    return density;
}

int main(int argc, char** argv)
{
    Options opt;
    if (!parseArgs(argc, argv, opt)) return 1;

    const auto frames = collectObjFrames(opt.inputDir);
    if (frames.empty()) {
        std::cerr << "No OBJ frames found in: " << opt.inputDir << "\n";
        return 2;
    }

    openvdb::initialize();

    openvdb::math::BBox<openvdb::Vec3s> globalBBox;
    bool hasBBox = false;
    for (const auto& path : frames) {
        Geometry geo;
        geo.read(path.string());
        const auto& bbox = geo.bbox();
        if (!hasBBox) {
            globalBBox = bbox;
            hasBBox = true;
        } else {
            globalBBox.expand(bbox);
        }
    }
    if (!hasBBox) {
        std::cerr << "Failed to compute geometry bbox\n";
        return 3;
    }

    const float voxelSize = estimateVoxelSize(globalBBox, opt.dim, opt.halfWidth);
    auto xform = openvdb::math::Transform::createLinearTransform(voxelSize);

    std::vector<FrameInfo> infos;
    infos.reserve(frames.size());
    openvdb::CoordBBox unionBBox;
    bool haveUnionBBox = false;
    float globalMin = std::numeric_limits<float>::infinity();
    float globalMax = -std::numeric_limits<float>::infinity();

    for (const auto& path : frames) {
        auto density = buildDensityGrid(path, opt, *xform);
        const auto bbox = density->evalActiveVoxelBoundingBox();
        if (!bbox.empty()) {
            if (!haveUnionBBox) {
                unionBBox = bbox;
                haveUnionBBox = true;
            } else {
                unionBBox.expand(bbox);
            }
        }
        infos.push_back(FrameInfo{path, bbox});
        for (auto iter = density->cbeginValueOn(); iter.test(); ++iter) {
            const float v = *iter;
            globalMin = std::min(globalMin, v);
            globalMax = std::max(globalMax, v);
        }
        globalMin = std::min(globalMin, density->background());
        globalMax = std::max(globalMax, density->background());
        std::cout << "Scanned " << path.filename().string() << "\n";
    }

    if (!haveUnionBBox) {
        std::cerr << "All fog grids are empty\n";
        return 4;
    }

    const auto min = unionBBox.min();
    const auto max = unionBBox.max();
    const int width = max.x() - min.x() + 1;
    const int height = max.y() - min.y() + 1;
    const int depth = max.z() - min.z() + 1;
    const int numFrames = static_cast<int>(frames.size());

    fs::create_directories(opt.outputPrefix.parent_path());
    const fs::path rawPath = opt.outputPrefix.string() + ".raw";
    const fs::path metaPath = opt.outputPrefix.string() + ".metadata.json";

    std::ofstream raw(rawPath, std::ios::binary);
    if (!raw) {
        std::cerr << "Failed to open raw output: " << rawPath << "\n";
        return 5;
    }

    const size_t frameVoxelCount = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(depth);
    std::vector<float> buffer(frameVoxelCount, 0.0f);

    for (int fi = 0; fi < numFrames; ++fi) {
        auto density = buildDensityGrid(frames[fi], opt, *xform);
        std::fill(buffer.begin(), buffer.end(), density->background());
        for (int z = min.z(); z <= max.z(); ++z) {
            for (int y = min.y(); y <= max.y(); ++y) {
                const size_t zy = (static_cast<size_t>(z - min.z()) * static_cast<size_t>(height) + static_cast<size_t>(y - min.y())) * static_cast<size_t>(width);
                for (int x = min.x(); x <= max.x(); ++x) {
                    const size_t idx = zy + static_cast<size_t>(x - min.x());
                    buffer[idx] = density->tree().getValue(openvdb::Coord(x, y, z));
                }
            }
        }
        raw.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(float)));
        std::cout << "Wrote frame " << (fi + 1) << "/" << numFrames << ": " << frames[fi].filename().string() << "\n";
    }
    raw.close();

    std::ofstream meta(metaPath, std::ios::binary);
    if (!meta) {
        std::cerr << "Failed to open metadata output: " << metaPath << "\n";
        return 6;
    }

    meta << "{\n";
    meta << "  \"source_dir\": \"" << jsonEscape(opt.inputDir.string()) << "\",\n";
    meta << "  \"grid_name\": \"" << jsonEscape(opt.gridName) << "\",\n";
    meta << "  \"conversion_mode\": \"" << jsonEscape(opt.mode) << "\",\n";
    meta << "  \"shell_width_voxels\": " << std::setprecision(6) << opt.shellWidth << ",\n";
    meta << "  \"half_width_voxels\": " << std::setprecision(6) << opt.halfWidth << ",\n";
    meta << "  \"format\": \"float32_raw_dense_index_space\",\n";
    meta << "  \"width\": " << width << ",\n";
    meta << "  \"height\": " << height << ",\n";
    meta << "  \"depth\": " << depth << ",\n";
    meta << "  \"frames\": " << numFrames << ",\n";
    meta << "  \"bbox_min\": [" << min.x() << ", " << min.y() << ", " << min.z() << "],\n";
    meta << "  \"bbox_max\": [" << max.x() << ", " << max.y() << ", " << max.z() << "],\n";
    meta << "  \"data_min\": " << std::setprecision(9) << globalMin << ",\n";
    meta << "  \"data_max\": " << std::setprecision(9) << globalMax << ",\n";
    meta << "  \"voxel_size\": " << std::setprecision(9) << voxelSize << ",\n";
    meta << "  \"frame_files\": [\n";
    for (int i = 0; i < numFrames; ++i) {
        meta << "    \"" << jsonEscape(frames[i].filename().string()) << "\"";
        meta << (i + 1 == numFrames ? "\n" : ",\n");
    }
    meta << "  ]\n";
    meta << "}\n";
    meta.close();

    std::cout << "OBJ sequence -> RAW done\n"
              << "  input:      " << opt.inputDir << "\n"
              << "  output raw: " << rawPath << "\n"
              << "  output meta:" << metaPath << "\n"
              << "  mode:       " << opt.mode << "\n"
              << "  voxel_size: " << voxelSize << "\n"
              << "  dense dims: " << width << " x " << height << " x " << depth << " x " << numFrames << "\n";
    return 0;
}
