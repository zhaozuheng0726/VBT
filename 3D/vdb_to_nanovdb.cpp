#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

#define NANOVDB_USE_OPENVDB
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path inputVdb;
    fs::path outputNvdb;
    std::string gridName;
};

void printUsage()
{
    std::cout
        << "Usage:\n"
        << "  vdb_to_nanovdb.exe --input-vdb <in.vdb> --output-nvdb <out.nvdb>\n"
        << "                     [--grid-name density]\n";
}

bool parseArgs(int argc, char** argv, Options& opt)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--input-vdb" && i + 1 < argc) {
            opt.inputVdb = argv[++i];
        } else if (arg == "--output-nvdb" && i + 1 < argc) {
            opt.outputNvdb = argv[++i];
        } else if (arg == "--grid-name" && i + 1 < argc) {
            opt.gridName = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            printUsage();
            return false;
        }
    }
    if (opt.inputVdb.empty() || opt.outputNvdb.empty()) {
        printUsage();
        return false;
    }
    return true;
}

openvdb::GridBase::Ptr loadGrid(const Options& opt)
{
    openvdb::io::File file(opt.inputVdb.string());
    file.open();

    openvdb::GridBase::Ptr selected;
    const auto grids = file.getGrids();
    if (!grids) {
        file.close();
        throw std::runtime_error("No grids found in VDB: " + opt.inputVdb.string());
    }

    if (!opt.gridName.empty()) {
        for (const auto& grid : *grids) {
            if (grid && grid->getName() == opt.gridName) {
                selected = grid;
                break;
            }
        }
        if (!selected) {
            file.close();
            throw std::runtime_error("Grid not found in VDB: " + opt.gridName);
        }
    } else {
        for (const auto& grid : *grids) {
            if (grid && openvdb::GridBase::grid<openvdb::FloatGrid>(grid)) {
                selected = grid;
                break;
            }
        }
        if (!selected && !grids->empty()) {
            selected = grids->front();
        }
    }

    file.close();
    if (!selected) {
        throw std::runtime_error("Failed to select a grid from: " + opt.inputVdb.string());
    }
    return selected;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        Options opt;
        if (!parseArgs(argc, argv, opt)) return 1;

        openvdb::initialize();
        const openvdb::GridBase::Ptr base = loadGrid(opt);
        auto handle = nanovdb::tools::openToNanoVDB(base,
                                                    nanovdb::tools::StatsMode::Default,
                                                    nanovdb::CheckMode::Default,
                                                    0);

        fs::create_directories(opt.outputNvdb.parent_path());
        nanovdb::io::writeGrid(opt.outputNvdb.string(), handle, nanovdb::io::Codec::NONE, 0);

        std::cout << "VDB -> NanoVDB done\n"
                  << "  input:  " << opt.inputVdb << "\n"
                  << "  grid:   " << base->getName() << "\n"
                  << "  output: " << opt.outputNvdb << "\n"
                  << "  bytes:  " << handle.bufferSize() << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "vdb_to_nanovdb failed: " << ex.what() << "\n";
        return 2;
    }
}
