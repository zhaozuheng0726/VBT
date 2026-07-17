#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "../src/field_profile.h"
#include "../render/src/vbt_file.h"
#include "vdb_tools/frame_metadata.h"

namespace fs = std::filesystem;
using namespace vdbtools;
using namespace vbt::render;

namespace {

constexpr int kLeafSize = 8;
constexpr int kCoarseControlCount = 4 * 4 * 4;

enum class RenderMode : uint32_t {
    Constant = 0,
    CoarseOnly = 1,
    SparseImpulse = 2,
    DenseFine = 3,
};

struct Options {
    fs::path inputVbt;
    fs::path metadataPath;
    fs::path outputVdb;
    int frameIndex = 0;
    float background = 0.0f;
    float sparseThreshold = 0.0f;
    std::string gridNameOverride;
};

struct RenderHeader {
    RenderMode mode = RenderMode::CoarseOnly;
    uint32_t startFrame = 0;
    uint32_t endFrame = 0;
    uint32_t config = 0;
};

void printUsage()
{
    std::cout
        << "Usage:\n"
        << "  vbtpack4_to_vdb.exe --input-vbt <file.vbtp> --metadata <meta.json> --output-vdb <out.vdb> --frame <index>\n"
        << "                      [--background 0] [--sparse-threshold 0] [--grid-name density]\n";
}

bool parseArgs(int argc, char** argv, Options& opt)
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

float halfToFloat(uint16_t h)
{
    const uint32_t sign = (static_cast<uint32_t>(h) >> 15u) & 1u;
    const uint32_t exponent = (static_cast<uint32_t>(h) >> 10u) & 0x1Fu;
    const uint32_t mantissa = static_cast<uint32_t>(h) & 0x3FFu;
    uint32_t f = 0;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            f = sign << 31u;
        } else {
            uint32_t e = 1u;
            uint32_t m = mantissa;
            while ((m & 0x400u) == 0u) {
                m <<= 1u;
                --e;
            }
            m &= 0x3FFu;
            f = (sign << 31u) | ((e + 112u) << 23u) | (m << 13u);
        }
    } else if (exponent == 31u) {
        f = (sign << 31u) | 0x7F800000u | (mantissa << 13u);
    } else {
        f = (sign << 31u) | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }
    float out = 0.0f;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

uint8_t readPayloadByte(const VbtFile& file, uint32_t byteOffset)
{
    const uint32_t word = file.payloadWords[byteOffset >> 2u];
    const uint32_t shift = (byteOffset & 3u) * 8u;
    return static_cast<uint8_t>((word >> shift) & 0xFFu);
}

uint16_t readPayloadU16(const VbtFile& file, uint32_t byteOffset)
{
    const uint16_t lo = static_cast<uint16_t>(readPayloadByte(file, byteOffset));
    const uint16_t hi = static_cast<uint16_t>(readPayloadByte(file, byteOffset + 1u));
    return static_cast<uint16_t>(lo | static_cast<uint16_t>(hi << 8u));
}

int8_t readPayloadI8(const VbtFile& file, uint32_t byteOffset)
{
    return static_cast<int8_t>(readPayloadByte(file, byteOffset));
}

uint32_t readPackedBits(const VbtFile& file, uint32_t byteOffset, int bitOffset, int bits)
{
    uint32_t packed = static_cast<uint32_t>(readPayloadByte(file, byteOffset));
    if (bitOffset + bits > 8) packed |= static_cast<uint32_t>(readPayloadByte(file, byteOffset + 1u)) << 8u;
    if (bitOffset + bits > 16) packed |= static_cast<uint32_t>(readPayloadByte(file, byteOffset + 2u)) << 16u;
    const uint32_t mask = (bits >= 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    return (packed >> bitOffset) & mask;
}

int32_t readPackedSignedBits(const VbtFile& file, uint32_t bitBase, uint32_t index, int bits)
{
    const uint32_t absoluteBit = bitBase + index * static_cast<uint32_t>(bits);
    const uint32_t byteOffset = absoluteBit >> 3u;
    const int bitOffset = static_cast<int>(absoluteBit & 7u);
    const uint32_t raw = readPackedBits(file, byteOffset, bitOffset, bits);
    if (bits <= 0) return 0;
    const uint32_t signBit = 1u << (bits - 1);
    if ((raw & signBit) == 0u) return static_cast<int32_t>(raw);
    const uint32_t extendMask = ~((1u << bits) - 1u);
    return static_cast<int32_t>(raw | extendMask);
}

RenderHeader decodeRenderHeader(uint32_t packedHeader)
{
    RenderHeader decoded{};
    decoded.mode = static_cast<RenderMode>(packedHeader & 0x3u);
    decoded.startFrame = (packedHeader >> 2u) & 0x7Fu;
    decoded.endFrame = (packedHeader >> 9u) & 0x7Fu;
    decoded.config = (packedHeader >> 16u) & 0xFFFFu;
    return decoded;
}

float dctBasis(int totalLength, int index, int k)
{
    constexpr double kPi = 3.14159265358979323846;
    const double invN = 1.0 / static_cast<double>(totalLength);
    const double alpha = (k == 0) ? std::sqrt(invN) : std::sqrt(2.0 * invN);
    return static_cast<float>(alpha * std::cos((kPi / static_cast<double>(totalLength)) *
                                               (static_cast<double>(index) + 0.5) *
                                               static_cast<double>(k)));
}

float decodeCoarseControlAt(const VbtFile& file,
                            uint32_t payloadByteBase,
                            int keep,
                            int controlIndex,
                            int timeIndex,
                            int encodedFrameCount)
{
    const uint32_t controlBase = payloadByteBase + static_cast<uint32_t>(4 + controlIndex * (keep + 1));
    double sum = static_cast<double>(halfToFloat(readPayloadU16(file, controlBase))) *
                 static_cast<double>(dctBasis(encodedFrameCount, timeIndex, 0));
    for (int k = 1; k < keep; ++k) {
        const float scale = file.coarseAcScales[static_cast<size_t>(k - 1)];
        const float coeff = static_cast<float>(readPayloadI8(file, controlBase + 2u + static_cast<uint32_t>(k - 1u))) * scale;
        sum += static_cast<double>(coeff) * static_cast<double>(dctBasis(encodedFrameCount, timeIndex, k));
    }
    return static_cast<float>(sum);
}

float sampleTrilinear4(const std::array<float, 64>& ctrl, uint32_t x, uint32_t y, uint32_t z)
{
    const float fx = (static_cast<float>(x) / 7.0f) * 3.0f;
    const float fy = (static_cast<float>(y) / 7.0f) * 3.0f;
    const float fz = (static_cast<float>(z) / 7.0f) * 3.0f;
    const int x0 = std::min(2, std::max(0, static_cast<int>(std::floor(fx))));
    const int y0 = std::min(2, std::max(0, static_cast<int>(std::floor(fy))));
    const int z0 = std::min(2, std::max(0, static_cast<int>(std::floor(fz))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);
    auto at = [&](int cx, int cy, int cz) -> float {
        return ctrl[static_cast<size_t>((cz * 4 + cy) * 4 + cx)];
    };
    const float c00 = at(x0, y0, z0) * (1.0f - tx) + at(x1, y0, z0) * tx;
    const float c01 = at(x0, y0, z1) * (1.0f - tx) + at(x1, y0, z1) * tx;
    const float c10 = at(x0, y1, z0) * (1.0f - tx) + at(x1, y1, z0) * tx;
    const float c11 = at(x0, y1, z1) * (1.0f - tx) + at(x1, y1, z1) * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

float sampleTrilinearGrid(const std::vector<float>& ctrl, int resolution, uint32_t x, uint32_t y, uint32_t z)
{
    const float fx = (static_cast<float>(x) / 7.0f) * static_cast<float>(resolution - 1);
    const float fy = (static_cast<float>(y) / 7.0f) * static_cast<float>(resolution - 1);
    const float fz = (static_cast<float>(z) / 7.0f) * static_cast<float>(resolution - 1);
    const int x0 = std::min(resolution - 2, std::max(0, static_cast<int>(std::floor(fx))));
    const int y0 = std::min(resolution - 2, std::max(0, static_cast<int>(std::floor(fy))));
    const int z0 = std::min(resolution - 2, std::max(0, static_cast<int>(std::floor(fz))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);
    auto at = [&](int cx, int cy, int cz) -> float {
        return ctrl[static_cast<size_t>((cz * resolution + cy) * resolution + cx)];
    };
    const float c00 = at(x0, y0, z0) * (1.0f - tx) + at(x1, y0, z0) * tx;
    const float c01 = at(x0, y0, z1) * (1.0f - tx) + at(x1, y0, z1) * tx;
    const float c10 = at(x0, y1, z0) * (1.0f - tx) + at(x1, y1, z0) * tx;
    const float c11 = at(x0, y1, z1) * (1.0f - tx) + at(x1, y1, z1) * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

float decodeBfp(uint32_t q, float scale, int bits)
{
    if (bits <= 2) {
        static constexpr std::array<float, 4> kLevels = {-1.0f, -0.3333333f, 0.3333333f, 1.0f};
        return kLevels[std::min<size_t>(q, kLevels.size() - 1)] * scale;
    }
    const int levels = (1 << bits) - 1;
    const float norm = (static_cast<float>(q) / static_cast<float>(levels)) * 2.0f - 1.0f;
    return norm * scale;
}

float decodeDenseResidualBfpAt(const VbtFile& file,
                               uint32_t payloadByteBase,
                               int coarseKeep,
                               int bitsPerValue,
                               uint32_t lx,
                               uint32_t ly,
                               uint32_t lz,
                               int localT,
                               int encodedFrameCount)
{
    const int valuesPerFrame = 512;
    const int frameBytes = (valuesPerFrame * bitsPerValue + 7) / 8;
    const uint32_t scalesBase = payloadByteBase + static_cast<uint32_t>(4 + 64 * (coarseKeep + 1));
    const uint32_t dataBase = scalesBase + static_cast<uint32_t>(encodedFrameCount * 2u);
    const float scale = halfToFloat(readPayloadU16(file, scalesBase + static_cast<uint32_t>(localT) * 2u));
    const int controlIndex = static_cast<int>((lz * 8u + ly) * 8u + lx);
    const uint32_t bitOffset = static_cast<uint32_t>(controlIndex * bitsPerValue);
    const uint32_t byteIndex = dataBase + (bitOffset >> 3u);
    const int shift = static_cast<int>(bitOffset & 7u);
    uint32_t packed = static_cast<uint32_t>(readPayloadByte(file, byteIndex));
    if (shift + bitsPerValue > 8) {
        packed |= static_cast<uint32_t>(readPayloadByte(file, byteIndex + 1u)) << 8u;
    }
    const uint32_t mask = (1u << bitsPerValue) - 1u;
    return decodeBfp((packed >> shift) & mask, scale, bitsPerValue);
}

float decodeFineResidualGridAt(const VbtFile& file,
                               uint32_t payloadByteBase,
                               int coarseKeep,
                               int encodedFrameCount,
                               const RenderHeader& header,
                               uint32_t lx,
                               uint32_t ly,
                               uint32_t lz,
                               int localT)
{
    const int resolution = static_cast<int>((header.config >> 12u) & 0x0Fu);
    const int dctKeep = static_cast<int>((header.config >> 6u) & 0x3Fu);
    const int quantBits = static_cast<int>(header.config & 0x3Fu);
    if (resolution < 2 || dctKeep <= 0 || quantBits <= 0) {
        throw std::runtime_error("Unsupported render fine-grid payload in VBTPACK4 exporter");
    }

    const int controlCount = resolution * resolution * resolution;
    const uint32_t fineBase = payloadByteBase + static_cast<uint32_t>(4 + 64 * (coarseKeep + 1));
    const float blockScale = [] (const VbtFile& inFile, uint32_t byteBase) {
        uint32_t word = inFile.payloadWords[byteBase >> 2u];
        float value = 0.0f;
        std::memcpy(&value, &word, sizeof(float));
        return value;
    }(file, fineBase);
    const uint32_t coeffBitBase = (fineBase + 4u) * 8u;

    std::vector<float> fineCtrl(static_cast<size_t>(controlCount), 0.0f);
    for (int i = 0; i < controlCount; ++i) {
        double sum = 0.0;
        for (int k = 0; k < dctKeep; ++k) {
            const uint32_t flatIndex = static_cast<uint32_t>(i * dctKeep + k);
            const int32_t q = readPackedSignedBits(file, coeffBitBase, flatIndex, quantBits);
            const float coeff = static_cast<float>(q) * blockScale;
            sum += static_cast<double>(coeff) * static_cast<double>(dctBasis(encodedFrameCount, localT, k));
        }
        fineCtrl[static_cast<size_t>(i)] = static_cast<float>(sum);
    }
    return sampleTrilinearGrid(fineCtrl, resolution, lx, ly, lz);
}

float decodeRenderValueAt(const VbtFile& file, uint32_t x, uint32_t y, uint32_t z, uint32_t t)
{
    const uint32_t leafIndex = leafIndexForVoxel(file.header, x, y, z);
    const uint32_t payloadByteBase = file.offsetsWords[leafIndex] * 4u;
    const uint32_t packedHeader = readPackedHeaderWord(file, leafIndex);
    const RenderHeader header = decodeRenderHeader(packedHeader);
    const uint32_t lx = x % std::max<uint32_t>(1u, file.header.leafSize);
    const uint32_t ly = y % std::max<uint32_t>(1u, file.header.leafSize);
    const uint32_t lz = z % std::max<uint32_t>(1u, file.header.leafSize);

    if (header.mode == RenderMode::Constant) {
        uint32_t word = file.payloadWords[(payloadByteBase + 4u) >> 2u];
        float value = 0.0f;
        std::memcpy(&value, &word, sizeof(float));
        return value;
    }

    const int coarseKeep = static_cast<int>(file.header.maxCoarseKeep);
    const int encodedFrameCount = std::max<int>(1, static_cast<int>(header.endFrame) - static_cast<int>(header.startFrame) + 1);
    const int localT = std::clamp<int>(static_cast<int>(t) - static_cast<int>(header.startFrame), 0, encodedFrameCount - 1);

    std::array<float, kCoarseControlCount> coarseCtrl{};
    for (int i = 0; i < kCoarseControlCount; ++i) {
        coarseCtrl[static_cast<size_t>(i)] =
            decodeCoarseControlAt(file, payloadByteBase, coarseKeep, i, localT, encodedFrameCount);
    }
    float coarse = sampleTrilinear4(coarseCtrl, lx, ly, lz);
    if (header.mode == RenderMode::CoarseOnly) {
        return coarse;
    }
    if (header.mode != RenderMode::DenseFine) {
        throw std::runtime_error("Unsupported render mode in VBTPACK4 exporter");
    }

    float residual = 0.0f;
    if ((header.config & 0x00FFu) == 0x01u) {
        const int bitsPerValue = static_cast<int>((header.config >> 8u) & 0x00FFu);
        residual = decodeDenseResidualBfpAt(file, payloadByteBase, coarseKeep, bitsPerValue, lx, ly, lz, localT, encodedFrameCount);
    } else {
        residual = decodeFineResidualGridAt(file, payloadByteBase, coarseKeep, encodedFrameCount, header, lx, ly, lz, localT);
    }
    return coarse + residual;
}

} // namespace

int main(int argc, char** argv)
{
    Options opt;
    if (!parseArgs(argc, argv, opt)) return 1;

    const FrameMetadata meta = loadFrameMetadata(opt.metadataPath);
    if (opt.frameIndex < 0 || opt.frameIndex >= meta.frames) {
        std::cerr << "Frame index out of range: " << opt.frameIndex << " / " << meta.frames << "\n";
        return 2;
    }

    VbtFile file;
    std::string error;
    if (!loadVbtFile(opt.inputVbt, file, error)) {
        std::cerr << error << "\n";
        return 3;
    }

    if (file.header.width != static_cast<uint32_t>(meta.width) ||
        file.header.height != static_cast<uint32_t>(meta.height) ||
        file.header.depth != static_cast<uint32_t>(meta.depth) ||
        file.header.frames != static_cast<uint32_t>(meta.frames)) {
        std::cerr << "Dimension mismatch between VBTPACK4 and metadata\n";
        return 4;
    }

    if (file.header.profileType == static_cast<uint32_t>(vbt::FieldType::GENERIC)) {
        std::cerr << "This exporter is for render/density VBTPACK4 files, not scientific/generic files.\n";
        return 5;
    }

    openvdb::initialize();
    auto grid = openvdb::FloatGrid::create(opt.background);
    grid->setGridClass(openvdb::GRID_FOG_VOLUME);
    grid->setName(opt.gridNameOverride.empty() ? meta.gridName : opt.gridNameOverride);

    auto accessor = grid->getAccessor();
    const int x0 = meta.bboxMin[0];
    const int y0 = meta.bboxMin[1];
    const int z0 = meta.bboxMin[2];
    const int progressStride = std::max(1, meta.depth / 20);

    for (int z = 0; z < meta.depth; ++z) {
        if (z == 0 || z == meta.depth - 1 || (z % progressStride) == 0) {
            std::cout << "Export progress: z=" << z << " / " << (meta.depth - 1) << "\n";
        }
        for (int y = 0; y < meta.height; ++y) {
            for (int x = 0; x < meta.width; ++x) {
                const float v = decodeRenderValueAt(file,
                                                    static_cast<uint32_t>(x),
                                                    static_cast<uint32_t>(y),
                                                    static_cast<uint32_t>(z),
                                                    static_cast<uint32_t>(opt.frameIndex));
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

    std::cout << "VBTPACK4 -> VDB done\n"
              << "  vbt:    " << opt.inputVbt << "\n"
              << "  meta:   " << opt.metadataPath << "\n"
              << "  frame:  " << opt.frameIndex << "\n"
              << "  output: " << opt.outputVdb << "\n";
    return 0;
}
