#include "vbt_file.h"

#include "../../src/frame_metadata.h"
#include "../../src/field_profile.h"

#include <algorithm>
#include <array>
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
using namespace vbt;
using namespace vbt::render;

namespace {

struct SmokeRayPacked {
    float ox = 0.0f;
    float oy = 0.0f;
    float oz = 0.0f;
    float tMin = 0.0f;
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 1.0f;
    float tMax = 0.0f;
};

struct Options {
    fs::path inputVbt;
    fs::path metadataPath;
    fs::path outputRayBin;
    fs::path outputSummaryJson;
    int frameIndex = 100;
    uint32_t imageWidth = 640;
    uint32_t imageHeight = 360;
    uint32_t stepCount = 192;
    float padXY = 0.05f;
    float padZ = 0.15f;
};

struct Bounds3f {
    float minX = 0.0f;
    float minY = 0.0f;
    float minZ = 0.0f;
    float maxX = 0.0f;
    float maxY = 0.0f;
    float maxZ = 0.0f;
};

void printUsage()
{
    std::cout
        << "Usage: vbt_render_smoke_probe --input-vbt <file.vbtp> --metadata <meta.json>\n"
        << "                              [--frame 100] [--image-width 640] [--image-height 360]\n"
        << "                              [--step-count 192] [--pad-xy 0.05] [--pad-z 0.15]\n"
        << "                              [--output-rays rays.bin] [--output-summary summary.json]\n";
}

bool parseArgs(int argc, char** argv, Options& opt)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input-vbt" && i + 1 < argc) {
            opt.inputVbt = argv[++i];
        } else if (arg == "--metadata" && i + 1 < argc) {
            opt.metadataPath = argv[++i];
        } else if (arg == "--frame" && i + 1 < argc) {
            opt.frameIndex = std::stoi(argv[++i]);
        } else if (arg == "--image-width" && i + 1 < argc) {
            opt.imageWidth = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--image-height" && i + 1 < argc) {
            opt.imageHeight = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--step-count" && i + 1 < argc) {
            opt.stepCount = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--pad-xy" && i + 1 < argc) {
            opt.padXY = std::stof(argv[++i]);
        } else if (arg == "--pad-z" && i + 1 < argc) {
            opt.padZ = std::stof(argv[++i]);
        } else if (arg == "--output-rays" && i + 1 < argc) {
            opt.outputRayBin = argv[++i];
        } else if (arg == "--output-summary" && i + 1 < argc) {
            opt.outputSummaryJson = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            printUsage();
            return false;
        }
    }

    if (opt.inputVbt.empty() || opt.metadataPath.empty()) {
        printUsage();
        return false;
    }
    if (opt.imageWidth == 0 || opt.imageHeight == 0 || opt.stepCount == 0) {
        std::cerr << "image size and step count must be > 0\n";
        return false;
    }
    return true;
}

Bounds3f computeBenchmarkBounds(const FrameMetadata& meta, float padXY, float padZ)
{
    const float minX = static_cast<float>(meta.bboxMin[0]);
    const float minY = static_cast<float>(meta.bboxMin[1]);
    const float minZ = static_cast<float>(meta.bboxMin[2]);
    const float maxX = static_cast<float>(meta.bboxMax[0]);
    const float maxY = static_cast<float>(meta.bboxMax[1]);
    const float maxZ = static_cast<float>(meta.bboxMax[2]);

    const float dx = std::max(1.0f, maxX - minX + 1.0f);
    const float dy = std::max(1.0f, maxY - minY + 1.0f);
    const float dz = std::max(1.0f, maxZ - minZ + 1.0f);

    Bounds3f out;
    out.minX = minX - dx * padXY;
    out.maxX = maxX + dx * padXY;
    out.minY = minY - dy * padXY;
    out.maxY = maxY + dy * padXY;
    out.minZ = minZ - dz * padZ;
    out.maxZ = maxZ + dz * padZ;
    return out;
}

std::vector<SmokeRayPacked> buildOrthographicRays(const Bounds3f& bounds,
                                                  uint32_t imageWidth,
                                                  uint32_t imageHeight)
{
    std::vector<SmokeRayPacked> rays;
    rays.resize(static_cast<size_t>(imageWidth) * static_cast<size_t>(imageHeight));

    const float width = std::max(1.0f, bounds.maxX - bounds.minX);
    const float height = std::max(1.0f, bounds.maxY - bounds.minY);
    const float depth = std::max(1.0f, bounds.maxZ - bounds.minZ);

    for (uint32_t py = 0; py < imageHeight; ++py) {
        for (uint32_t px = 0; px < imageWidth; ++px) {
            const float u = (static_cast<float>(px) + 0.5f) / static_cast<float>(imageWidth);
            const float v = (static_cast<float>(py) + 0.5f) / static_cast<float>(imageHeight);
            const float x = bounds.minX + u * width;
            const float y = bounds.minY + v * height;
            SmokeRayPacked ray;
            ray.ox = x;
            ray.oy = y;
            ray.oz = bounds.minZ;
            ray.tMin = 0.0f;
            ray.dx = 0.0f;
            ray.dy = 0.0f;
            ray.dz = 1.0f;
            ray.tMax = depth;
            rays[static_cast<size_t>(py) * imageWidth + px] = ray;
        }
    }
    return rays;
}

void writeBinary(const fs::path& path, const std::vector<SmokeRayPacked>& rays)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open ray output: " + path.string());
    }
    out.write(reinterpret_cast<const char*>(rays.data()),
              static_cast<std::streamsize>(rays.size() * sizeof(SmokeRayPacked)));
}

std::string jsonEscape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += c; break;
        }
    }
    return out;
}

void writeSummary(const fs::path& path,
                  const Options& opt,
                  const VbtFile& file,
                  const FrameMetadata& meta,
                  const Bounds3f& bounds,
                  size_t rayCount)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open summary output: " + path.string());
    }
    const double depth = static_cast<double>(std::max(1.0f, bounds.maxZ - bounds.minZ));
    const double dt = depth / static_cast<double>(opt.stepCount);
    out << "{\n";
    out << "  \"input_vbt\": \"" << jsonEscape(opt.inputVbt.string()) << "\",\n";
    out << "  \"metadata\": \"" << jsonEscape(opt.metadataPath.string()) << "\",\n";
    out << "  \"frame\": " << opt.frameIndex << ",\n";
    out << "  \"image_width\": " << opt.imageWidth << ",\n";
    out << "  \"image_height\": " << opt.imageHeight << ",\n";
    out << "  \"ray_count\": " << rayCount << ",\n";
    out << "  \"step_count\": " << opt.stepCount << ",\n";
    out << "  \"dt\": " << std::fixed << std::setprecision(6) << dt << ",\n";
    out << "  \"profile_type\": " << file.header.profileType << ",\n";
    out << "  \"leaf_size\": " << file.header.leafSize << ",\n";
    out << "  \"coarse_resolution\": " << file.header.coarseResolution << ",\n";
    out << "  \"bbox_min\": [" << bounds.minX << ", " << bounds.minY << ", " << bounds.minZ << "],\n";
    out << "  \"bbox_max\": [" << bounds.maxX << ", " << bounds.maxY << ", " << bounds.maxZ << "],\n";
    out << "  \"metadata_bbox_min\": [" << meta.bboxMin[0] << ", " << meta.bboxMin[1] << ", " << meta.bboxMin[2] << "],\n";
    out << "  \"metadata_bbox_max\": [" << meta.bboxMax[0] << ", " << meta.bboxMax[1] << ", " << meta.bboxMax[2] << "]\n";
    out << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        Options opt;
        if (!parseArgs(argc, argv, opt)) return 1;

        VbtFile file;
        std::string error;
        if (!loadVbtFile(opt.inputVbt, file, error)) {
            std::cerr << error << "\n";
            return 2;
        }
        const auto meta = loadFrameMetadata(opt.metadataPath);
        if (opt.frameIndex < 0 || opt.frameIndex >= meta.frames) {
            std::cerr << "Frame index out of range: " << opt.frameIndex << " / " << meta.frames << "\n";
            return 3;
        }
        if (file.header.profileType != static_cast<uint32_t>(FieldType::DENSITY)) {
            std::cerr << "Smoke probe only supports render density .vbtp\n";
            return 4;
        }

        const Bounds3f bounds = computeBenchmarkBounds(meta, opt.padXY, opt.padZ);
        const auto rays = buildOrthographicRays(bounds, opt.imageWidth, opt.imageHeight);

        std::cout << "Smoke sampling probe\n";
        std::cout << "  inputVbt: " << opt.inputVbt.string() << "\n";
        std::cout << "  metadata: " << opt.metadataPath.string() << "\n";
        std::cout << "  frame: " << opt.frameIndex << "\n";
        std::cout << "  image: " << opt.imageWidth << " x " << opt.imageHeight << "\n";
        std::cout << "  rays: " << rays.size() << "\n";
        std::cout << "  stepCount: " << opt.stepCount << "\n";
        std::cout << "  bboxMin: [" << bounds.minX << ", " << bounds.minY << ", " << bounds.minZ << "]\n";
        std::cout << "  bboxMax: [" << bounds.maxX << ", " << bounds.maxY << ", " << bounds.maxZ << "]\n";
        std::cout << "  note: this is Phase-0 smoke workload preparation for the render-profile GPU benchmark.\n";

        if (!opt.outputRayBin.empty()) {
            writeBinary(opt.outputRayBin, rays);
            std::cout << "  wrote rays: " << opt.outputRayBin.string() << "\n";
        }
        if (!opt.outputSummaryJson.empty()) {
            writeSummary(opt.outputSummaryJson, opt, file, meta, bounds, rays.size());
            std::cout << "  wrote summary: " << opt.outputSummaryJson.string() << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Smoke probe failed: " << ex.what() << "\n";
        return 10;
    }
}
