#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/tools/VolumeToMesh.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path inputVdb;
    fs::path outputObj;
    std::string gridName = "density";
    double isovalue = 0.08;
    double adaptivity = 0.0;
    bool triangulate = false;
    int smoothGaussianWidth = 0;
    int smoothGaussianIterations = 1;
    double smoothMix = 1.0;
};

void printUsage()
{
    std::cout
        << "Usage:\n"
        << "  vdb_to_obj.exe --input-vdb <in.vdb> --output-obj <out.obj>\n"
        << "                 [--grid-name density] [--isovalue 0.08] [--adaptivity 0.0] [--triangulate]\n"
        << "                 [--smooth-gaussian-width 0] [--smooth-gaussian-iterations 1] [--smooth-mix 1.0]\n";
}

bool parseArgs(int argc, char** argv, Options& opt)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--input-vdb" && i + 1 < argc) {
            opt.inputVdb = argv[++i];
        } else if (arg == "--output-obj" && i + 1 < argc) {
            opt.outputObj = argv[++i];
        } else if (arg == "--grid-name" && i + 1 < argc) {
            opt.gridName = argv[++i];
        } else if (arg == "--isovalue" && i + 1 < argc) {
            opt.isovalue = std::stod(argv[++i]);
        } else if (arg == "--adaptivity" && i + 1 < argc) {
            opt.adaptivity = std::stod(argv[++i]);
        } else if (arg == "--triangulate") {
            opt.triangulate = true;
        } else if (arg == "--smooth-gaussian-width" && i + 1 < argc) {
            opt.smoothGaussianWidth = std::stoi(argv[++i]);
        } else if (arg == "--smooth-gaussian-iterations" && i + 1 < argc) {
            opt.smoothGaussianIterations = std::stoi(argv[++i]);
        } else if (arg == "--smooth-mix" && i + 1 < argc) {
            opt.smoothMix = std::stod(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage();
            return false;
        }
    }
    if (opt.inputVdb.empty() || opt.outputObj.empty()) {
        printUsage();
        return false;
    }
    return true;
}

openvdb::GridBase::Ptr chooseGrid(openvdb::io::File& file, const std::string& requestedName)
{
    file.open();
    openvdb::GridBase::Ptr chosen;
    for (auto iter = file.beginName(); iter != file.endName(); ++iter) {
        if (iter.gridName() == requestedName) {
            chosen = file.readGrid(iter.gridName());
            break;
        }
    }
    if (!chosen) {
        for (auto iter = file.beginName(); iter != file.endName(); ++iter) {
            openvdb::GridBase::Ptr grid = file.readGrid(iter.gridName());
            if (grid->isType<openvdb::FloatGrid>() || grid->isType<openvdb::DoubleGrid>()) {
                chosen = grid;
                break;
            }
        }
    }
    file.close();
    return chosen;
}

template <typename GridT>
bool meshGrid(const GridT& grid, const Options& opt)
{
    auto gridPtr = grid.deepCopy();
    if (opt.smoothGaussianWidth > 0 && opt.smoothGaussianIterations > 0) {
        auto smoothed = grid.deepCopy();
        openvdb::tools::Filter<GridT> filter(*smoothed);
        filter.gaussian(opt.smoothGaussianWidth, opt.smoothGaussianIterations);
        const auto mix = std::clamp(opt.smoothMix, 0.0, 1.0);
        if (mix > 0.0) {
            for (auto iter = gridPtr->beginValueAll(); iter; ++iter) {
                const auto coord = iter.getCoord();
                const auto baseValue = grid.tree().getValue(coord);
                const auto smoothValue = smoothed->tree().getValue(coord);
                const auto blended = static_cast<typename GridT::ValueType>(
                    (1.0 - mix) * static_cast<double>(baseValue) + mix * static_cast<double>(smoothValue));
                iter.setValue(blended);
            }
        }
    }

    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    std::vector<openvdb::Vec4I> quads;
    openvdb::tools::volumeToMesh(*gridPtr, points, triangles, quads, opt.isovalue, opt.adaptivity, true);

    std::ofstream out(opt.outputObj, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output OBJ: " << opt.outputObj << "\n";
        return false;
    }

    out << "# generated from " << opt.inputVdb.string() << "\n";
    out << "# grid " << grid.getName() << "\n";
    out << "# isovalue " << opt.isovalue << " adaptivity " << opt.adaptivity
        << " smoothGaussianWidth " << opt.smoothGaussianWidth
        << " smoothGaussianIterations " << opt.smoothGaussianIterations << "\n";
    for (const auto& p : points) {
        out << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    for (const auto& tri : triangles) {
        out << "f " << (tri[0] + 1) << " " << (tri[1] + 1) << " " << (tri[2] + 1) << "\n";
    }
    for (const auto& quad : quads) {
        if (opt.triangulate) {
            out << "f " << (quad[0] + 1) << " " << (quad[1] + 1) << " " << (quad[2] + 1) << "\n";
            out << "f " << (quad[0] + 1) << " " << (quad[2] + 1) << " " << (quad[3] + 1) << "\n";
        } else {
            out << "f " << (quad[0] + 1) << " " << (quad[1] + 1) << " " << (quad[2] + 1) << " " << (quad[3] + 1)
                << "\n";
        }
    }
    out.close();

    std::cout << "VDB -> OBJ done\n";
    std::cout << "  input:      " << opt.inputVdb << "\n";
    std::cout << "  output:     " << opt.outputObj << "\n";
    std::cout << "  points:     " << points.size() << "\n";
    std::cout << "  triangles:  " << triangles.size() << "\n";
    std::cout << "  quads:      " << quads.size() << "\n";
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    Options opt;
    if (!parseArgs(argc, argv, opt)) return 1;

    openvdb::initialize();

    openvdb::io::File file(opt.inputVdb.string());
    openvdb::GridBase::Ptr baseGrid = chooseGrid(file, opt.gridName);
    if (!baseGrid) {
        std::cerr << "No compatible scalar grid found in: " << opt.inputVdb << "\n";
        return 2;
    }

    fs::create_directories(opt.outputObj.parent_path());

    if (baseGrid->isType<openvdb::FloatGrid>()) {
        return meshGrid(*openvdb::gridConstPtrCast<openvdb::FloatGrid>(baseGrid), opt) ? 0 : 3;
    }
    if (baseGrid->isType<openvdb::DoubleGrid>()) {
        return meshGrid(*openvdb::gridConstPtrCast<openvdb::DoubleGrid>(baseGrid), opt) ? 0 : 3;
    }

    std::cerr << "Unsupported grid type for meshing: " << baseGrid->type() << "\n";
    return 4;
}
