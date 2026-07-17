#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

#include <filesystem>
#include <iostream>
#include <string>

#include "block_tree.h"
#include "vdb_tools/frame_metadata.h"

namespace fs = std::filesystem;
using namespace vdbtools;

struct Options {
    fs::path inputVbt;
    fs::path metadataPath;
    fs::path outputVdb;
    int frameIndex = 0;
    float background = 0.0f;
    float sparseThreshold = 0.0f;
    std::string gridNameOverride;
};

static void printUsage()
{
    std::cout
        << "Usage:\n"
        << "  vbt_to_vdb.exe --input-vbt <file.vbt> --metadata <meta.json> --output-vdb <out.vdb> --frame <index>\n"
        << "                  [--background 0] [--sparse-threshold 0] [--grid-name density]\n";
}

static bool parseArgs(int argc, char** argv, Options& opt)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--input-vbt" && i + 1 < argc) {
            opt.inputVbt = argv[++i];
        } else if (arg == "--metadata" && i + 1 < argc) {
            opt.metadataPath = argv[++i];
        } else if (arg == "--output-vdb" && i + 1 < argc) {
            opt.outputVdb = argv[++i];
        } else if (arg == "--frame" && i + 1 < argc) {
            opt.frameIndex = std::stoi(argv[++i]);
        } else if (arg == "--background" && i + 1 < argc) {
            opt.background = std::stof(argv[++i]);
        } else if (arg == "--sparse-threshold" && i + 1 < argc) {
            opt.sparseThreshold = std::stof(argv[++i]);
        } else if (arg == "--grid-name" && i + 1 < argc) {
            opt.gridNameOverride = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            printUsage();
            return false;
        }
    }
    if (opt.inputVbt.empty() || opt.metadataPath.empty() || opt.outputVdb.empty()) {
        printUsage();
        return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    Options opt;
    if (!parseArgs(argc, argv, opt)) return 1;

    const FrameMetadata meta = loadFrameMetadata(opt.metadataPath);
    if (opt.frameIndex < 0 || opt.frameIndex >= meta.frames) {
        std::cerr << "Frame index out of range: " << opt.frameIndex << " / " << meta.frames << "\n";
        return 2;
    }

    BlockTree tree;
    if (!loadFromVBT(opt.inputVbt.string(), tree)) {
        std::cerr << "Failed to load VBT: " << opt.inputVbt << "\n";
        return 3;
    }

    if (tree.dimX != meta.width || tree.dimY != meta.height || tree.dimZ != meta.depth) {
        std::cerr << "Dimension mismatch between VBT and metadata\n";
        std::cerr << "  VBT:  " << tree.dimX << " x " << tree.dimY << " x " << tree.dimZ << "\n";
        std::cerr << "  Meta: " << meta.width << " x " << meta.height << " x " << meta.depth << "\n";
        return 4;
    }

    openvdb::initialize();
    auto grid = openvdb::FloatGrid::create(opt.background);
    grid->setGridClass(openvdb::GRID_FOG_VOLUME);
    grid->setName(opt.gridNameOverride.empty() ? meta.gridName : opt.gridNameOverride);

    auto accessor = grid->getAccessor();
    const int x0 = meta.bboxMin[0];
    const int y0 = meta.bboxMin[1];
    const int z0 = meta.bboxMin[2];
    const float t = static_cast<float>(opt.frameIndex);
    const int progressStride = std::max(1, tree.dimZ / 20);

    for (int z = 0; z < tree.dimZ; ++z) {
        if (z == 0 || z == tree.dimZ - 1 || (z % progressStride) == 0) {
            std::cout << "Export progress: z=" << z << " / " << (tree.dimZ - 1) << "\n";
        }
        for (int y = 0; y < tree.dimY; ++y) {
            for (int x = 0; x < tree.dimX; ++x) {
                const float v = tree.sampleFlat(x, y, z, t);
                if (std::abs(v - opt.background) <= opt.sparseThreshold) continue;
                accessor.setValue(openvdb::Coord(x0 + x, y0 + y, z0 + z), v);
            }
        }
    }

    grid->tree().prune();
    fs::create_directories(opt.outputVdb.parent_path());
    openvdb::io::File out(opt.outputVdb.string());
    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    out.write(grids);
    out.close();

    std::cout << "VBT -> VDB done\n"
              << "  vbt:    " << opt.inputVbt << "\n"
              << "  meta:   " << opt.metadataPath << "\n"
              << "  frame:  " << opt.frameIndex << "\n"
              << "  output: " << opt.outputVdb << "\n";
    return 0;
}
