#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

#define NANOVDB_USE_OPENVDB
#define NANOVDB_USE_ZIP
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path inputVdb;
    fs::path outputNvdb;
    fs::path outputNvdbZip;
    fs::path outputMatchedVdb;
    std::string gridName;
};

void printUsage()
{
    std::cout
        << "Usage:\n"
        << "  vdb_to_nanovdb.exe --input-vdb <in.vdb> [--output-nvdb <out.nvdb>]\n"
        << "                     [--output-nvdb-zip <out.zip.nvdb>]\n"
        << "                     [--output-matched-vdb <density.vdb>]\n"
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
        } else if (arg == "--output-nvdb-zip" && i + 1 < argc) {
            opt.outputNvdbZip = argv[++i];
        } else if (arg == "--output-matched-vdb" && i + 1 < argc) {
            opt.outputMatchedVdb = argv[++i];
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
    if (opt.inputVdb.empty() ||
        (opt.outputNvdb.empty() && opt.outputNvdbZip.empty() && opt.outputMatchedVdb.empty())) {
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
    if (!opt.gridName.empty()) {
        selected = file.readGrid(opt.gridName);
        if (!selected) {
            file.close();
            throw std::runtime_error("Grid not found in VDB: " + opt.gridName);
        }
    } else {
        const auto grids = file.getGrids();
        if (!grids) {
            file.close();
            throw std::runtime_error("No grids found in VDB: " + opt.inputVdb.string());
        }
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
        using Clock = std::chrono::steady_clock;
        const auto elapsedMs = [](const Clock::time_point& start) {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        };
        Options opt;
        if (!parseArgs(argc, argv, opt)) return 1;

        openvdb::initialize();
        auto timer = Clock::now();
        const openvdb::GridBase::Ptr base = loadGrid(opt);
        const double openvdbReadMs = elapsedMs(timer);
        std::cout << "VDB grid conversion done\n"
                  << "  input:  " << opt.inputVdb << "\n"
                  << "  grid:   " << base->getName() << "\n"
                  << "  openvdb-read-ms: " << openvdbReadMs << "\n";

        if (!opt.outputMatchedVdb.empty()) {
            if (!opt.outputMatchedVdb.parent_path().empty()) {
                fs::create_directories(opt.outputMatchedVdb.parent_path());
            }
            openvdb::GridPtrVec grids;
            grids.push_back(base);
            openvdb::io::File matchedFile(opt.outputMatchedVdb.string());
            timer = Clock::now();
            matchedFile.write(grids);
            matchedFile.close();
            std::cout << "  matched-vdb-output: " << opt.outputMatchedVdb << "\n"
                      << "  matched-vdb-bytes:  " << fs::file_size(opt.outputMatchedVdb) << "\n"
                      << "  matched-vdb-write-ms: " << elapsedMs(timer) << "\n";
        }

        if (!opt.outputNvdb.empty() || !opt.outputNvdbZip.empty()) {
            timer = Clock::now();
            auto handle = nanovdb::tools::openToNanoVDB(base,
                                                        nanovdb::tools::StatsMode::Default,
                                                        nanovdb::CheckMode::Default,
                                                        0);
            const double nanoConvertMs = elapsedMs(timer);
            std::cout << "  nanovdb-buffer-bytes: " << handle.bufferSize() << "\n"
                      << "  nanovdb-convert-ms: " << nanoConvertMs << "\n";
            if (!opt.outputNvdb.empty()) {
                if (!opt.outputNvdb.parent_path().empty()) {
                    fs::create_directories(opt.outputNvdb.parent_path());
                }
                timer = Clock::now();
                nanovdb::io::writeGrid(opt.outputNvdb.string(), handle, nanovdb::io::Codec::NONE, 0);
                std::cout << "  nanovdb-output: " << opt.outputNvdb << "\n"
                          << "  nanovdb-none-write-ms: " << elapsedMs(timer) << "\n";
            }
            if (!opt.outputNvdbZip.empty()) {
                if (!opt.outputNvdbZip.parent_path().empty()) {
                    fs::create_directories(opt.outputNvdbZip.parent_path());
                }
                timer = Clock::now();
                nanovdb::io::writeGrid(opt.outputNvdbZip.string(), handle, nanovdb::io::Codec::ZIP, 0);
                std::cout << "  nanovdb-zip-output: " << opt.outputNvdbZip << "\n"
                          << "  nanovdb-zip-write-ms: " << elapsedMs(timer) << "\n";
            }
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "vdb_to_nanovdb failed: " << ex.what() << "\n";
        return 2;
    }
}
