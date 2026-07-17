#include "../render/src/vbt_file.h"
#include "../src/field_profile.h"
#include "../src/render_temporal_decode.h"
#include "../src/render_temporal_payload.h"
#include "vdb_tools/frame_metadata.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using namespace vbt;
using namespace vbt::render;

namespace {

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }

float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3 cross(Vec3 a, Vec3 b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

float length(Vec3 v) { return std::sqrt(std::max(0.0f, dot(v, v))); }

Vec3 normalize(Vec3 v)
{
    const float len = length(v);
    if (len <= 1.0e-8f) return {0.0f, 0.0f, 1.0f};
    return v * (1.0f / len);
}

struct Options {
    fs::path inputVbt;
    fs::path metadataPath;
    fs::path outputImage;
    int frameIndex = 0;
    int width = 512;
    int height = 288;
    int steps = 128;
    float exposure = 0.045f;
    float densityScale = 1.0f;
    float gamma = 1.0f;
    float cameraOffsetX = 0.20f;
    float cameraOffsetY = 0.10f;
    float cameraOffsetZ = 3.15f;
    float targetOffsetX = 0.12f;
    float targetOffsetY = 0.22f;
    float targetOffsetZ = 0.0f;
    float orthoScale = 2.0f;
    float shellEmptyScale = 1.0f;
    float shellEmptyMax = std::numeric_limits<float>::quiet_NaN();
};

void printUsage()
{
    std::cout
        << "Usage:\n"
        << "  render_temporal_vbt_direct_preview.exe --input-vbt <file.vbtp> --metadata <meta.json>\n"
        << "      --output-bmp <out.bmp> --frame <index> [--width 512] [--height 288]\n"
        << "      [--output-ppm <out.ppm>]\n"
        << "      [--steps 128] [--exposure 0.045] [--density-scale 1.0]\n"
        << "      [--camera-offset x,y,z] [--target-offset x,y,z] [--ortho-scale 2.0]\n";
}

bool parseVec3(const std::string& text, float& x, float& y, float& z)
{
    const size_t a = text.find(',');
    if (a == std::string::npos) return false;
    const size_t b = text.find(',', a + 1);
    if (b == std::string::npos) return false;
    x = std::stof(text.substr(0, a));
    y = std::stof(text.substr(a + 1, b - a - 1));
    z = std::stof(text.substr(b + 1));
    return true;
}

bool parseArgs(int argc, char** argv, Options& opt)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--input-vbt" && i + 1 < argc) opt.inputVbt = argv[++i];
        else if (arg == "--metadata" && i + 1 < argc) opt.metadataPath = argv[++i];
        else if (arg == "--output-ppm" && i + 1 < argc) opt.outputImage = argv[++i];
        else if (arg == "--output-bmp" && i + 1 < argc) opt.outputImage = argv[++i];
        else if (arg == "--frame" && i + 1 < argc) opt.frameIndex = std::stoi(argv[++i]);
        else if (arg == "--width" && i + 1 < argc) opt.width = std::stoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) opt.height = std::stoi(argv[++i]);
        else if (arg == "--steps" && i + 1 < argc) opt.steps = std::stoi(argv[++i]);
        else if (arg == "--exposure" && i + 1 < argc) opt.exposure = std::stof(argv[++i]);
        else if (arg == "--density-scale" && i + 1 < argc) opt.densityScale = std::stof(argv[++i]);
        else if (arg == "--gamma" && i + 1 < argc) opt.gamma = std::stof(argv[++i]);
        else if (arg == "--ortho-scale" && i + 1 < argc) opt.orthoScale = std::stof(argv[++i]);
        else if (arg == "--shell-empty-scale" && i + 1 < argc) opt.shellEmptyScale = std::stof(argv[++i]);
        else if (arg == "--shell-empty-max" && i + 1 < argc) opt.shellEmptyMax = std::stof(argv[++i]);
        else if (arg == "--camera-offset" && i + 1 < argc) {
            if (!parseVec3(argv[++i], opt.cameraOffsetX, opt.cameraOffsetY, opt.cameraOffsetZ)) return false;
        } else if (arg == "--target-offset" && i + 1 < argc) {
            if (!parseVec3(argv[++i], opt.targetOffsetX, opt.targetOffsetY, opt.targetOffsetZ)) return false;
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            printUsage();
            return false;
        }
    }
    if (opt.inputVbt.empty() || opt.metadataPath.empty() || opt.outputImage.empty()) {
        printUsage();
        return false;
    }
    if (opt.width <= 0 || opt.height <= 0 || opt.steps <= 0) {
        std::cerr << "width, height, and steps must be positive\n";
        return false;
    }
    return true;
}

std::vector<float> controlCoords(int resolution)
{
    std::vector<float> coords(static_cast<size_t>(resolution), 0.0f);
    if (resolution <= 1) return coords;
    const float scale = 7.0f / static_cast<float>(resolution - 1);
    for (int i = 0; i < resolution; ++i) {
        coords[static_cast<size_t>(i)] = static_cast<float>(i) * scale;
    }
    return coords;
}

bool shellOccupancyContains(const RenderTemporalShellOccupancySection& occupancy, uint16_t voxelIndex)
{
    if (voxelIndex >= 512u) return false;
    const uint32_t group = voxelIndex >> 6u;
    const uint32_t bit = voxelIndex & 63u;
    return ((occupancy.shellMask[group] >> bit) & 1ull) != 0ull;
}

float samplePackedLeafGridValue(const uint8_t* leafBase,
                                const RenderTemporalPackedLeafView& leafView,
                                bool useFineStream,
                                int resolution,
                                int frames,
                                int leafWidth,
                                int leafHeight,
                                int leafDepth,
                                int frameIndex,
                                float fx,
                                float fy,
                                float fz)
{
    const auto coords = controlCoords(resolution);
    auto locate = [&](float p, int extent) {
        const float maxP = static_cast<float>(std::max(0, extent - 1));
        p = std::clamp(p, 0.0f, maxP);
        int i1 = 1;
        while (i1 < resolution - 1 && coords[static_cast<size_t>(i1)] < p) ++i1;
        const int i0 = std::max(0, i1 - 1);
        const float c0 = coords[static_cast<size_t>(i0)];
        const float c1 = coords[static_cast<size_t>(i1)];
        const float w = (c1 > c0) ? ((p - c0) / (c1 - c0)) : 0.0f;
        return std::array<float, 3>{static_cast<float>(i0), static_cast<float>(i1), w};
    };
    const auto lx = locate(fx, leafWidth);
    const auto ly = locate(fy, leafHeight);
    const auto lz = locate(fz, leafDepth);
    auto at = [&](int ix, int iy, int iz) {
        const int idx = (iz * resolution + iy) * resolution + ix;
        return decodeRenderTemporalLeafControlValue(leafBase,
                                                    leafView.header,
                                                    leafView.prefix,
                                                    leafView.layout,
                                                    useFineStream,
                                                    true,
                                                    frames,
                                                    static_cast<uint16_t>(idx),
                                                    frameIndex);
    };
    const int x0 = static_cast<int>(lx[0]);
    const int x1 = static_cast<int>(lx[1]);
    const int y0 = static_cast<int>(ly[0]);
    const int y1 = static_cast<int>(ly[1]);
    const int z0 = static_cast<int>(lz[0]);
    const int z1 = static_cast<int>(lz[1]);
    const float tx = lx[2];
    const float ty = ly[2];
    const float tz = lz[2];
    const float c00 = at(x0, y0, z0) * (1.0f - tx) + at(x1, y0, z0) * tx;
    const float c01 = at(x0, y0, z1) * (1.0f - tx) + at(x1, y0, z1) * tx;
    const float c10 = at(x0, y1, z0) * (1.0f - tx) + at(x1, y1, z0) * tx;
    const float c11 = at(x0, y1, z1) * (1.0f - tx) + at(x1, y1, z1) * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

float samplePackedValueAtWorld(const VbtFile& file,
                               const vdbtools::FrameMetadata& meta,
                               const Options& opt,
                               float xWorld,
                               float yWorld,
                               float zWorld)
{
    const float x = xWorld - static_cast<float>(meta.bboxMin[0]);
    const float y = yWorld - static_cast<float>(meta.bboxMin[1]);
    const float z = zWorld - static_cast<float>(meta.bboxMin[2]);
    if (x < 0.0f || y < 0.0f || z < 0.0f ||
        x > static_cast<float>(file.header.width - 1) ||
        y > static_cast<float>(file.header.height - 1) ||
        z > static_cast<float>(file.header.depth - 1)) {
        return 0.0f;
    }

    const int leafSize = static_cast<int>(file.header.leafSize);
    const int leafCountX = (static_cast<int>(file.header.width) + leafSize - 1) / leafSize;
    const int leafCountY = (static_cast<int>(file.header.height) + leafSize - 1) / leafSize;
    const int leafCountZ = (static_cast<int>(file.header.depth) + leafSize - 1) / leafSize;
    const int xi = std::clamp(static_cast<int>(std::floor(x)), 0, static_cast<int>(file.header.width) - 1);
    const int yi = std::clamp(static_cast<int>(std::floor(y)), 0, static_cast<int>(file.header.height) - 1);
    const int zi = std::clamp(static_cast<int>(std::floor(z)), 0, static_cast<int>(file.header.depth) - 1);
    const int bx = std::min(leafCountX - 1, xi / leafSize);
    const int by = std::min(leafCountY - 1, yi / leafSize);
    const int bz = std::min(leafCountZ - 1, zi / leafSize);
    const uint32_t leafIndex = static_cast<uint32_t>((bz * leafCountY + by) * leafCountX + bx);

    const uint32_t wordBegin = file.offsetsWords[leafIndex];
    const uint32_t wordEnd = file.offsetsWords[leafIndex + 1u];
    if (wordEnd <= wordBegin) return 0.0f;

    const uint8_t* payloadBase = reinterpret_cast<const uint8_t*>(file.payloadWords.data());
    const uint8_t* leafBase = payloadBase + static_cast<size_t>(wordBegin) * sizeof(uint32_t);
    const size_t leafBytes = static_cast<size_t>(wordEnd - wordBegin) * sizeof(uint32_t);
    const auto leafView = parseRenderTemporalPackedLeaf(leafBase, leafBytes, true);
    if (leafView.header.mode == TemporalFirstPackedMode::EMPTY) return 0.0f;

    const int baseX = bx * leafSize;
    const int baseY = by * leafSize;
    const int baseZ = bz * leafSize;
    const int leafWidth = std::min(leafSize, static_cast<int>(file.header.width) - baseX);
    const int leafHeight = std::min(leafSize, static_cast<int>(file.header.height) - baseY);
    const int leafDepth = std::min(leafSize, static_cast<int>(file.header.depth) - baseZ);
    const float fx = x - static_cast<float>(baseX);
    const float fy = y - static_cast<float>(baseY);
    const float fz = z - static_cast<float>(baseZ);

    float value = samplePackedLeafGridValue(leafBase,
                                            leafView,
                                            false,
                                            static_cast<int>(leafView.header.coarseResolution),
                                            static_cast<int>(file.header.frames),
                                            leafWidth,
                                            leafHeight,
                                            leafDepth,
                                            opt.frameIndex,
                                            fx,
                                            fy,
                                            fz);
    if (leafView.header.mode == TemporalFirstPackedMode::TEMPORAL_FINE6 &&
        leafView.header.fineResolution > 0u) {
        value += samplePackedLeafGridValue(leafBase,
                                           leafView,
                                           true,
                                           static_cast<int>(leafView.header.fineResolution),
                                           static_cast<int>(file.header.frames),
                                           leafWidth,
                                           leafHeight,
                                           leafDepth,
                                           opt.frameIndex,
                                           fx,
                                           fy,
                                           fz);
    } else if (leafView.header.mode == TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT) {
        const int lx = std::clamp(static_cast<int>(std::floor(fx)), 0, leafWidth - 1);
        const int ly = std::clamp(static_cast<int>(std::floor(fy)), 0, leafHeight - 1);
        const int lz = std::clamp(static_cast<int>(std::floor(fz)), 0, leafDepth - 1);
        const uint16_t voxelIndex = renderTemporalLeafVoxelIndex(
            static_cast<uint16_t>(lx),
            static_cast<uint16_t>(ly),
            static_cast<uint16_t>(lz),
            static_cast<uint16_t>(file.header.leafSize));
        value += decodeRenderTemporalLeafShellVoxelValue(leafBase,
                                                         leafView,
                                                         true,
                                                         static_cast<int>(file.header.frames),
                                                         voxelIndex,
                                                         opt.frameIndex);
        if (!std::isnan(opt.shellEmptyMax) &&
            opt.shellEmptyScale < 1.0f &&
            leafView.layout.fine.descriptorBytes > 0) {
            const auto occupancy = unpackRenderTemporalShellOccupancySection(
                leafBase + leafView.layout.fine.descriptorOffset,
                leafView.layout.fine.descriptorBytes);
            if (!shellOccupancyContains(occupancy, voxelIndex) &&
                value > 0.0f &&
                value <= opt.shellEmptyMax) {
                value *= opt.shellEmptyScale;
            }
        }
    }
    return std::max(0.0f, value) * opt.densityScale;
}

bool intersectBox(Vec3 origin, Vec3 dir, Vec3 bmin, Vec3 bmax, float& t0, float& t1)
{
    t0 = -std::numeric_limits<float>::infinity();
    t1 = std::numeric_limits<float>::infinity();
    const float o[3] = {origin.x, origin.y, origin.z};
    const float d[3] = {dir.x, dir.y, dir.z};
    const float mn[3] = {bmin.x, bmin.y, bmin.z};
    const float mx[3] = {bmax.x, bmax.y, bmax.z};
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(d[axis]) < 1.0e-8f) {
            if (o[axis] < mn[axis] || o[axis] > mx[axis]) return false;
            continue;
        }
        float a = (mn[axis] - o[axis]) / d[axis];
        float b = (mx[axis] - o[axis]) / d[axis];
        if (a > b) std::swap(a, b);
        t0 = std::max(t0, a);
        t1 = std::min(t1, b);
        if (t0 > t1) return false;
    }
    return t1 > std::max(t0, 0.0f);
}

void writePpm(const fs::path& path, int width, int height, const std::vector<uint8_t>& rgb)
{
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open output PPM: " + path.string());
    out << "P6\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    if (!out) throw std::runtime_error("Failed to write output PPM: " + path.string());
}

void writeBmp(const fs::path& path, int width, int height, const std::vector<uint8_t>& rgb)
{
    fs::create_directories(path.parent_path());
    const int rowStride = ((width * 3 + 3) / 4) * 4;
    const uint32_t pixelBytes = static_cast<uint32_t>(rowStride * height);
    const uint32_t fileBytes = 14u + 40u + pixelBytes;
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open output BMP: " + path.string());

    auto writeU16 = [&](uint16_t v) {
        char b[2] = {static_cast<char>(v & 0xffu), static_cast<char>((v >> 8u) & 0xffu)};
        out.write(b, 2);
    };
    auto writeU32 = [&](uint32_t v) {
        char b[4] = {
            static_cast<char>(v & 0xffu),
            static_cast<char>((v >> 8u) & 0xffu),
            static_cast<char>((v >> 16u) & 0xffu),
            static_cast<char>((v >> 24u) & 0xffu),
        };
        out.write(b, 4);
    };
    auto writeI32 = [&](int32_t v) {
        writeU32(static_cast<uint32_t>(v));
    };

    writeU16(0x4D42u);
    writeU32(fileBytes);
    writeU16(0u);
    writeU16(0u);
    writeU32(54u);
    writeU32(40u);
    writeI32(width);
    writeI32(height);
    writeU16(1u);
    writeU16(24u);
    writeU32(0u);
    writeU32(pixelBytes);
    writeI32(2835);
    writeI32(2835);
    writeU32(0u);
    writeU32(0u);

    std::vector<uint8_t> row(static_cast<size_t>(rowStride), 0u);
    for (int y = height - 1; y >= 0; --y) {
        std::fill(row.begin(), row.end(), 0u);
        for (int x = 0; x < width; ++x) {
            const size_t src = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3u;
            const size_t dst = static_cast<size_t>(x) * 3u;
            row[dst + 0] = rgb[src + 2];
            row[dst + 1] = rgb[src + 1];
            row[dst + 2] = rgb[src + 0];
        }
        out.write(reinterpret_cast<const char*>(row.data()), row.size());
    }
    if (!out) throw std::runtime_error("Failed to write output BMP: " + path.string());
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
        if (file.header.profileType != static_cast<uint32_t>(FieldType::DENSITY)) {
            std::cerr << "Direct preview currently supports render/density VBT payloads only.\n";
            return 3;
        }
        const vdbtools::FrameMetadata meta = vdbtools::loadFrameMetadata(opt.metadataPath);
        if (file.header.width != static_cast<uint32_t>(meta.width) ||
            file.header.height != static_cast<uint32_t>(meta.height) ||
            file.header.depth != static_cast<uint32_t>(meta.depth) ||
            file.header.frames != static_cast<uint32_t>(meta.frames)) {
            std::cerr << "Dimension mismatch between VBTPACK4 and metadata\n";
            return 4;
        }
        if (opt.frameIndex < 0 || opt.frameIndex >= static_cast<int>(file.header.frames)) {
            std::cerr << "Frame index out of range.\n";
            return 5;
        }

        const Vec3 bmin{
            static_cast<float>(meta.bboxMin[0]),
            static_cast<float>(meta.bboxMin[1]),
            static_cast<float>(meta.bboxMin[2]),
        };
        const Vec3 bmax{
            static_cast<float>(meta.bboxMin[0] + static_cast<int>(file.header.width) - 1),
            static_cast<float>(meta.bboxMin[1] + static_cast<int>(file.header.height) - 1),
            static_cast<float>(meta.bboxMin[2] + static_cast<int>(file.header.depth) - 1),
        };
        const Vec3 center = (bmin + bmax) * 0.5f;
        const Vec3 size = bmax - bmin;
        const float radius = std::max({size.x, size.y, size.z}) * 0.5f;
        const Vec3 target = center + Vec3{opt.targetOffsetX, opt.targetOffsetY, opt.targetOffsetZ} * radius;
        const Vec3 camera = center + Vec3{opt.cameraOffsetX, opt.cameraOffsetY, opt.cameraOffsetZ} * radius;
        const Vec3 forward = normalize(target - camera);
        Vec3 right = normalize(cross(forward, {0.0f, 1.0f, 0.0f}));
        if (length(right) < 1.0e-5f) right = {1.0f, 0.0f, 0.0f};
        const Vec3 up = normalize(cross(right, forward));

        const float aspect = static_cast<float>(opt.width) / static_cast<float>(opt.height);
        const float halfY = radius * opt.orthoScale * 0.5f;
        const float halfX = halfY * aspect;
        std::vector<uint8_t> rgb(static_cast<size_t>(opt.width) * static_cast<size_t>(opt.height) * 3u, 0);

        for (int py = 0; py < opt.height; ++py) {
            const float v = (1.0f - 2.0f * ((static_cast<float>(py) + 0.5f) / static_cast<float>(opt.height))) * halfY;
            for (int px = 0; px < opt.width; ++px) {
                const float u = (2.0f * ((static_cast<float>(px) + 0.5f) / static_cast<float>(opt.width)) - 1.0f) * halfX;
                const Vec3 origin = camera + right * u + up * v;
                float tEnter = 0.0f;
                float tExit = 0.0f;
                float transmittance = 1.0f;
                float radiance = 0.0f;
                if (intersectBox(origin, forward, bmin, bmax, tEnter, tExit)) {
                    tEnter = std::max(tEnter, 0.0f);
                    const float dt = (tExit - tEnter) / static_cast<float>(opt.steps);
                    for (int s = 0; s < opt.steps; ++s) {
                        const float t = tEnter + (static_cast<float>(s) + 0.5f) * dt;
                        const Vec3 p = origin + forward * t;
                        const float density = samplePackedValueAtWorld(file, meta, opt, p.x, p.y, p.z);
                        const float alpha = 1.0f - std::exp(-density * opt.exposure * dt);
                        radiance += transmittance * alpha;
                        transmittance *= (1.0f - alpha);
                        if (transmittance < 0.01f) break;
                    }
                }
                float tone = std::clamp(radiance, 0.0f, 1.0f);
                if (opt.gamma > 0.0f && opt.gamma != 1.0f) {
                    tone = std::pow(tone, 1.0f / opt.gamma);
                }
                const float bgR = 0.07f;
                const float bgG = 0.08f;
                const float bgB = 0.09f;
                const float smR = 0.72f;
                const float smG = 0.77f;
                const float smB = 0.84f;
                const uint8_t r = static_cast<uint8_t>(std::round(255.0f * std::clamp(bgR + tone * (smR - bgR), 0.0f, 1.0f)));
                const uint8_t g = static_cast<uint8_t>(std::round(255.0f * std::clamp(bgG + tone * (smG - bgG), 0.0f, 1.0f)));
                const uint8_t b = static_cast<uint8_t>(std::round(255.0f * std::clamp(bgB + tone * (smB - bgB), 0.0f, 1.0f)));
                const size_t idx = (static_cast<size_t>(py) * static_cast<size_t>(opt.width) + static_cast<size_t>(px)) * 3u;
                rgb[idx + 0] = r;
                rgb[idx + 1] = g;
                rgb[idx + 2] = b;
            }
            if ((py % 32) == 0) {
                std::cout << "row " << py << " / " << opt.height << "\n";
            }
        }

        const std::string ext = opt.outputImage.extension().string();
        if (ext == ".bmp" || ext == ".BMP") {
            writeBmp(opt.outputImage, opt.width, opt.height, rgb);
        } else {
            writePpm(opt.outputImage, opt.width, opt.height, rgb);
        }
        std::cout << "Direct VBT preview done\n"
                  << "  vbt:    " << opt.inputVbt << "\n"
                  << "  frame:  " << opt.frameIndex << "\n"
                  << "  output: " << opt.outputImage << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 10;
    }
}
