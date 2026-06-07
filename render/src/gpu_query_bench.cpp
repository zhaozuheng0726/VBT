#include "gpu_query_bench.h"
#include "scientific_decode.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace vbt::render {

namespace {

struct PushConstants {
    uint32_t dimX = 0;
    uint32_t dimY = 0;
    uint32_t dimZ = 0;
    uint32_t frames = 0;
    uint32_t leafSize = 0;
    uint32_t leafCountX = 0;
    uint32_t leafCountY = 0;
    uint32_t queryCount = 0;
    uint32_t coarseAcScaleCount = 0;
};
static_assert(sizeof(PushConstants) <= 128, "Push constants too large");

struct DensePushConstants {
    uint32_t dimX = 0;
    uint32_t dimY = 0;
    uint32_t dimZ = 0;
    uint32_t frames = 0;
    uint32_t queryCount = 0;
};
static_assert(sizeof(DensePushConstants) <= 128, "Dense push constants too large");

struct GpuQueryPacked {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
    uint32_t t = 0;
};

struct DenseFrameCache {
    std::vector<uint32_t> frameToSlot;
    std::vector<float> denseValues;
    std::vector<float> expectedValues;
    uint32_t uniqueFrameCount = 0;
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

std::vector<char> readFileBytes(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open shader: " + path);
    }
    const size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> bytes(size);
    file.seekg(0);
    file.read(bytes.data(), static_cast<std::streamsize>(size));
    return bytes;
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
                            int timeIndex)
{
    const uint32_t controlBase = payloadByteBase + static_cast<uint32_t>(4 + controlIndex * (keep + 1));
    double sum = static_cast<double>(halfToFloat(readPayloadU16(file, controlBase))) *
                 static_cast<double>(dctBasis(file.header.frames, timeIndex, 0));
    for (int k = 1; k < keep; ++k) {
        const float scale = file.coarseAcScales[static_cast<size_t>(k - 1)];
        const float coeff = static_cast<float>(readPayloadI8(file, controlBase + 2u + static_cast<uint32_t>(k - 1u))) * scale;
        sum += static_cast<double>(coeff) * static_cast<double>(dctBasis(file.header.frames, timeIndex, k));
    }
    return static_cast<float>(sum);
}

float sampleCoarseAt(const std::array<float, 64>& ctrl, uint32_t x, uint32_t y, uint32_t z)
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

float decodeDenseValueAt(const VbtFile& file,
                         uint32_t payloadByteBase,
                         int coarseKeep,
                         int denseResolution,
                         int denseBits,
                         uint32_t x,
                         uint32_t y,
                         uint32_t z,
                         uint32_t t)
{
    const int valuesPerFrame = denseResolution * denseResolution * denseResolution;
    const int frameBytes = (valuesPerFrame * denseBits + 7) / 8;
    const uint32_t scalesBase = payloadByteBase + static_cast<uint32_t>(4 + 64 * (coarseKeep + 1));
    const uint32_t dataBase = scalesBase + static_cast<uint32_t>(file.header.frames * 2u);
    const float scale = halfToFloat(readPayloadU16(file, scalesBase + t * 2u));

    const float fx = (static_cast<float>(x) / 7.0f) * static_cast<float>(denseResolution - 1);
    const float fy = (static_cast<float>(y) / 7.0f) * static_cast<float>(denseResolution - 1);
    const float fz = (static_cast<float>(z) / 7.0f) * static_cast<float>(denseResolution - 1);
    const int x0 = std::min(denseResolution - 2, std::max(0, static_cast<int>(std::floor(fx))));
    const int y0 = std::min(denseResolution - 2, std::max(0, static_cast<int>(std::floor(fy))));
    const int z0 = std::min(denseResolution - 2, std::max(0, static_cast<int>(std::floor(fz))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);

    auto sample = [&](int cx, int cy, int cz) -> float {
        const int controlIndex = (cz * denseResolution + cy) * denseResolution + cx;
        const uint32_t frameBase = dataBase + static_cast<uint32_t>(t * frameBytes);
        const int bitOffset = controlIndex * denseBits;
        const uint32_t byteIndex = frameBase + static_cast<uint32_t>(bitOffset / 8);
        const int shift = bitOffset % 8;
        uint32_t packed = static_cast<uint32_t>(readPayloadByte(file, byteIndex));
        if (shift + denseBits > 8) {
            packed |= static_cast<uint32_t>(readPayloadByte(file, byteIndex + 1u)) << 8u;
        }
        const uint32_t mask = (1u << denseBits) - 1u;
        return decodeBfp((packed >> shift) & mask, scale, denseBits);
    };

    const float c00 = sample(x0, y0, z0) * (1.0f - tx) + sample(x1, y0, z0) * tx;
    const float c01 = sample(x0, y0, z1) * (1.0f - tx) + sample(x1, y0, z1) * tx;
    const float c10 = sample(x0, y1, z0) * (1.0f - tx) + sample(x1, y1, z0) * tx;
    const float c11 = sample(x0, y1, z1) * (1.0f - tx) + sample(x1, y1, z1) * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

float decodeDenseTemporalBasisAt(const VbtFile& file,
                                 uint32_t payloadByteBase,
                                 int coarseKeep,
                                 int denseResolution,
                                 int denseBits,
                                 int temporalKeep,
                                 uint32_t x,
                                 uint32_t y,
                                 uint32_t z,
                                 uint32_t t)
{
    if (temporalKeep <= 0 || denseBits <= 0) return 0.0f;
    const int valuesPerBasis = denseResolution * denseResolution * denseResolution;
    const int basisBytes = (valuesPerBasis * denseBits + 7) / 8;
    const uint32_t scalesBase = payloadByteBase + static_cast<uint32_t>(4 + 64 * (coarseKeep + 1));
    const uint32_t dataBase = scalesBase + static_cast<uint32_t>(temporalKeep * 2);

    const float fx = (static_cast<float>(x) / 7.0f) * static_cast<float>(denseResolution - 1);
    const float fy = (static_cast<float>(y) / 7.0f) * static_cast<float>(denseResolution - 1);
    const float fz = (static_cast<float>(z) / 7.0f) * static_cast<float>(denseResolution - 1);
    const int x0 = std::min(denseResolution - 2, std::max(0, static_cast<int>(std::floor(fx))));
    const int y0 = std::min(denseResolution - 2, std::max(0, static_cast<int>(std::floor(fy))));
    const int z0 = std::min(denseResolution - 2, std::max(0, static_cast<int>(std::floor(fz))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);

    auto sampleBasis = [&](int basisIdx, int cx, int cy, int cz) -> float {
        const int controlIndex = (cz * denseResolution + cy) * denseResolution + cx;
        const uint32_t basisBase = dataBase + static_cast<uint32_t>(basisIdx * basisBytes);
        const int bitOffset = controlIndex * denseBits;
        const uint32_t byteIndex = basisBase + static_cast<uint32_t>(bitOffset / 8);
        const int shift = bitOffset % 8;
        uint32_t packed = static_cast<uint32_t>(readPayloadByte(file, byteIndex));
        if (shift + denseBits > 8) {
            packed |= static_cast<uint32_t>(readPayloadByte(file, byteIndex + 1u)) << 8u;
        }
        const uint32_t mask = (1u << denseBits) - 1u;
        const float scale = halfToFloat(readPayloadU16(file, scalesBase + static_cast<uint32_t>(basisIdx * 2)));
        return decodeBfp((packed >> shift) & mask, scale, denseBits);
    };

    float residual = 0.0f;
    for (int k = 0; k < temporalKeep; ++k) {
        const float c00 = sampleBasis(k, x0, y0, z0) * (1.0f - tx) + sampleBasis(k, x1, y0, z0) * tx;
        const float c01 = sampleBasis(k, x0, y0, z1) * (1.0f - tx) + sampleBasis(k, x1, y0, z1) * tx;
        const float c10 = sampleBasis(k, x0, y1, z0) * (1.0f - tx) + sampleBasis(k, x1, y1, z0) * tx;
        const float c11 = sampleBasis(k, x0, y1, z1) * (1.0f - tx) + sampleBasis(k, x1, y1, z1) * tx;
        const float c0 = c00 * (1.0f - ty) + c10 * ty;
        const float c1 = c01 * (1.0f - ty) + c11 * ty;
        residual += (c0 * (1.0f - tz) + c1 * tz) * dctBasis(file.header.frames, static_cast<int>(t), k);
    }
    return residual;
}

float decodeSparseResidualAt(const VbtFile& file,
                             uint32_t payloadByteBase,
                             const ScientificHeaderV2& decoded,
                             int coarseKeep,
                             uint32_t localIndexValue,
                             uint32_t t)
{
    const uint32_t actualCount = decoded.packedEventCount;
    if (actualCount == 0u) return 0.0f;

    const uint32_t tierCapacity =
        (decoded.tierId == 0u) ? 64u :
        (decoded.tierId == 1u) ? 256u :
        (decoded.tierId == 2u) ? 768u : 1024u;
    const uint32_t framesPerBin = decoded.framesPerBinMinus1 + 1u;
    const uint32_t binIdx = std::min<uint32_t>(15u, t / std::max<uint32_t>(1u, framesPerBin));
    const uint8_t localTimeInBin = static_cast<uint8_t>(t - binIdx * std::max<uint32_t>(1u, framesPerBin));
    const uint32_t sparseBase = payloadByteBase + 4u + static_cast<uint32_t>(64 * (coarseKeep + 1));

    float eventScale = 0.0f;
    const uint32_t eventScaleWord = file.payloadWords[sparseBase >> 2u];
    std::memcpy(&eventScale, &eventScaleWord, sizeof(float));
    if (!(eventScale > 0.0f)) return 0.0f;

    const uint32_t maskWordOffset = sparseBase + 4u + ((localIndexValue >> 5u) * 4u);
    const uint32_t maskWord = file.payloadWords[maskWordOffset >> 2u];
    const uint32_t maskBit = 1u << (localIndexValue & 31u);
    if ((maskWord & maskBit) == 0u) return 0.0f;

    const uint32_t binsBase = sparseBase + 4u + 64u;
    const uint16_t start = readPayloadU16(file, binsBase + binIdx * 2u);
    const uint16_t end = (binIdx + 1u < 16u)
        ? readPayloadU16(file, binsBase + (binIdx + 1u) * 2u)
        : static_cast<uint16_t>(actualCount);
    if (start >= end) return 0.0f;

    const uint32_t coordsBase = binsBase + 32u;
    const uint32_t residualBase = coordsBase + tierCapacity * 2u;
    const uint16_t targetKey = static_cast<uint16_t>((static_cast<uint16_t>(localTimeInBin) << 9u) |
                                                     static_cast<uint16_t>(localIndexValue & 0x1FFu));
    int left = static_cast<int>(start);
    int right = static_cast<int>(end) - 1;
    while (left <= right) {
        const int mid = (left + right) >> 1;
        const uint16_t key = readPayloadU16(file, coordsBase + static_cast<uint32_t>(mid) * 2u);
        if (key == targetKey) {
            const uint32_t byteIndex = residualBase + static_cast<uint32_t>(mid >> 1);
            const uint8_t byte = readPayloadByte(file, byteIndex);
            const uint8_t nibble = ((mid & 1) == 0)
                ? static_cast<uint8_t>(byte & 0x0Fu)
                : static_cast<uint8_t>((byte >> 4u) & 0x0Fu);
            int8_t q = static_cast<int8_t>(nibble);
            if ((q & 0x08) != 0) q = static_cast<int8_t>(q - 16);
            return static_cast<float>(q) * eventScale;
        }
        if (key < targetKey) left = mid + 1;
        else right = mid - 1;
    }
    return 0.0f;
}

float decodeScientificValueAt(const VbtFile& file, const Query4D& query)
{
    const uint32_t leafIndex = query.leafIndex;
    const uint32_t packedHeader = readPackedHeaderWord(file, leafIndex);
    const ScientificHeaderV2 decoded = decodeScientificHeaderV2(packedHeader);
    const uint32_t payloadByteBase = file.offsetsWords[leafIndex] * 4u;

    const uint32_t leafSize = std::max<uint32_t>(1u, file.header.leafSize);
    const uint32_t lx = query.x % leafSize;
    const uint32_t ly = query.y % leafSize;
    const uint32_t lz = query.z % leafSize;
    const uint32_t lidx = (lz * leafSize + ly) * leafSize + lx;

    const int coarseKeep = static_cast<int>(decoded.coarseKeepMinus1 + 1u);

    std::array<float, 64> coarseCtrl{};
    for (int i = 0; i < 64; ++i) {
        coarseCtrl[static_cast<size_t>(i)] =
            decodeCoarseControlAt(file, payloadByteBase, coarseKeep, i, static_cast<int>(query.t));
    }
    const float coarse = sampleCoarseAt(coarseCtrl, lx, ly, lz);

    float residual = 0.0f;
    switch (decoded.mode) {
    case ScientificMode::CoarseOnly:
        residual = 0.0f;
        break;
    case ScientificMode::SparseEvents:
        residual = decodeSparseResidualAt(file, payloadByteBase, decoded, coarseKeep, lidx, query.t);
        break;
    case ScientificMode::DenseGrid3:
        residual = (decoded.denseSubtype == kScientificDenseSubtypeTemporalBasis)
            ? decodeDenseTemporalBasisAt(file,
                                         payloadByteBase,
                                         coarseKeep,
                                         3,
                                         static_cast<int>(decoded.denseQuantBits),
                                         static_cast<int>(decoded.denseTemporalKeepMinus1 + 1u),
                                         lx, ly, lz, query.t)
            : decodeDenseValueAt(file, payloadByteBase, coarseKeep, 3, static_cast<int>(decoded.denseQuantBits), lx, ly, lz, query.t);
        break;
    case ScientificMode::DenseGrid4:
        residual = (decoded.denseSubtype == kScientificDenseSubtypeTemporalBasis)
            ? decodeDenseTemporalBasisAt(file,
                                         payloadByteBase,
                                         coarseKeep,
                                         4,
                                         static_cast<int>(decoded.denseQuantBits),
                                         static_cast<int>(decoded.denseTemporalKeepMinus1 + 1u),
                                         lx, ly, lz, query.t)
            : decodeDenseValueAt(file, payloadByteBase, coarseKeep, 4, static_cast<int>(decoded.denseQuantBits), lx, ly, lz, query.t);
        break;
    default:
        residual = 0.0f;
        break;
    }
    return coarse + residual;
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeBits, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("No suitable Vulkan memory type found");
}

void createBuffer(VkPhysicalDevice physicalDevice,
                  VkDevice device,
                  VkDeviceSize size,
                  VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags properties,
                  Buffer& out)
{
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = std::max<VkDeviceSize>(size, 4);
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &out.buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan buffer");
    }

    VkMemoryRequirements memReq{};
    vkGetBufferMemoryRequirements(device, out.buffer, &memReq);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memReq.memoryTypeBits, properties);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &out.memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate Vulkan buffer memory");
    }

    vkBindBufferMemory(device, out.buffer, out.memory, 0);
    out.size = bufferInfo.size;
}

void destroyBuffer(VkDevice device, Buffer& buffer)
{
    if (buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
    }
    if (buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }
    buffer.size = 0;
}

void copyBuffer(VkDevice device,
                VkQueue queue,
                VkCommandPool commandPool,
                VkBuffer src,
                VkBuffer dst,
                VkDeviceSize size)
{
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &allocInfo, &cmd) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate copy command buffer");
    }

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copy{};
    copy.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &copy);

    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = dst;
    barrier.offset = 0;
    barrier.size = size;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0,
                         nullptr,
                         1,
                         &barrier,
                         0,
                         nullptr);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code)
{
    VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }
    return module;
}

bool buildDenseFrameCache(const VbtFile& file,
                          const std::vector<Query4D>& queries,
                          size_t maxDenseBytes,
                          DenseFrameCache& outCache,
                          std::string& outError)
{
    outCache = {};
    outCache.frameToSlot.assign(file.header.frames, std::numeric_limits<uint32_t>::max());
    outCache.expectedValues.resize(queries.size(), 0.0f);

    std::vector<uint32_t> uniqueFrames;
    uniqueFrames.reserve(file.header.frames);
    for (const auto& q : queries) {
        if (q.t >= file.header.frames) {
            outError = "Query frame index out of range for dense baseline.";
            return false;
        }
        if (outCache.frameToSlot[q.t] == std::numeric_limits<uint32_t>::max()) {
            outCache.frameToSlot[q.t] = static_cast<uint32_t>(uniqueFrames.size());
            uniqueFrames.push_back(q.t);
        }
    }

    const size_t frameVoxelCount = static_cast<size_t>(file.header.width) *
                                   static_cast<size_t>(file.header.height) *
                                   static_cast<size_t>(file.header.depth);
    const size_t denseBytes = uniqueFrames.size() * frameVoxelCount * sizeof(float);
    if (denseBytes > maxDenseBytes) {
        outError = "Dense frame cache exceeds memory cap (" +
                   std::to_string(denseBytes / (1024ull * 1024ull)) + " MB > " +
                   std::to_string(maxDenseBytes / (1024ull * 1024ull)) + " MB).";
        return false;
    }

    outCache.uniqueFrameCount = static_cast<uint32_t>(uniqueFrames.size());
    outCache.denseValues.resize(uniqueFrames.size() * frameVoxelCount);
    for (size_t slot = 0; slot < uniqueFrames.size(); ++slot) {
        const auto frame = reconstructScientificFrameCpu(file, uniqueFrames[slot]);
        std::copy(frame.begin(),
                  frame.end(),
                  outCache.denseValues.begin() + static_cast<std::ptrdiff_t>(slot * frameVoxelCount));
    }

    for (size_t i = 0; i < queries.size(); ++i) {
        const auto& q = queries[i];
        const uint32_t slot = outCache.frameToSlot[q.t];
        if (slot == std::numeric_limits<uint32_t>::max()) {
            outError = "Internal dense frame slot mapping failure.";
            return false;
        }
        const size_t linear = (static_cast<size_t>(q.z) * file.header.height + q.y) * file.header.width + q.x;
        outCache.expectedValues[i] =
            outCache.denseValues[static_cast<size_t>(slot) * frameVoxelCount + linear];
    }
    return true;
}

} // namespace

bool runGpuQueryBench(const VbtFile& file,
                      const std::vector<Query4D>& queries,
                      const std::string& shaderPath,
                      GpuQueryBenchStats& outStats)
{
    outStats = {};
    if (file.header.version != 4) {
        outStats.error = "GPU value benchmark requires VBTPACK4 files.";
        return false;
    }

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkQueryPool timestampPool = VK_NULL_HANDLE;
    Buffer queryBuffer{};
    Buffer resultBuffer{};
    Buffer coarseScaleBuffer{};
    Buffer offsetsBuffer{};
    Buffer payloadBuffer{};
    Buffer stagingCoarseScales{};
    Buffer stagingOffsets{};
    Buffer stagingPayload{};

    try {
        const auto totalStart = std::chrono::high_resolution_clock::now();

        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.pApplicationName = "VBT Query Bench";
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.pEngineName = "None";
        appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instanceInfo.pApplicationInfo = &appInfo;
        if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance");
        }

        uint32_t physicalDeviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);
        if (physicalDeviceCount == 0) {
            throw std::runtime_error("No Vulkan physical device found");
        }
        std::vector<VkPhysicalDevice> devices(physicalDeviceCount);
        vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, devices.data());

        uint32_t queueFamilyIndex = UINT32_MAX;
        VkPhysicalDeviceProperties physicalProps{};
        VkPhysicalDeviceLimits limits{};
        for (VkPhysicalDevice candidate : devices) {
            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, nullptr);
            std::vector<VkQueueFamilyProperties> families(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, families.data());
            for (uint32_t i = 0; i < queueFamilyCount; ++i) {
                if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    physicalDevice = candidate;
                    queueFamilyIndex = i;
                    vkGetPhysicalDeviceProperties(candidate, &physicalProps);
                    limits = physicalProps.limits;
                    break;
                }
            }
            if (physicalDevice != VK_NULL_HANDLE) {
                break;
            }
        }
        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("No Vulkan compute queue family found");
        }

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueInfo.queueFamilyIndex = queueFamilyIndex;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan device");
        }
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool");
        }

        VkCommandBufferAllocateInfo cmdAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cmdAlloc.commandPool = commandPool;
        cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAlloc.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &cmdAlloc, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffer");
        }

        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create fence");
        }

        const auto uploadStart = std::chrono::high_resolution_clock::now();

        std::vector<GpuQueryPacked> packedQueries(queries.size());
        for (size_t i = 0; i < queries.size(); ++i) {
            packedQueries[i].x = queries[i].x;
            packedQueries[i].y = queries[i].y;
            packedQueries[i].z = queries[i].z;
            packedQueries[i].t = queries[i].t;
        }

        createBuffer(physicalDevice,
                     device,
                     sizeof(GpuQueryPacked) * packedQueries.size(),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     queryBuffer);
        void* mapped = nullptr;
        vkMapMemory(device, queryBuffer.memory, 0, queryBuffer.size, 0, &mapped);
        std::memcpy(mapped, packedQueries.data(), sizeof(GpuQueryPacked) * packedQueries.size());
        vkUnmapMemory(device, queryBuffer.memory);

        createBuffer(physicalDevice,
                     device,
                     sizeof(GpuQueryResult) * queries.size(),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     resultBuffer);

        const VkDeviceSize offsetsBytes = sizeof(uint32_t) * file.offsetsWords.size();
        const VkDeviceSize payloadBytes = sizeof(uint32_t) * file.payloadWords.size();

        createBuffer(physicalDevice,
                     device,
                     sizeof(float) * file.coarseAcScales.size(),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     coarseScaleBuffer);
        createBuffer(physicalDevice,
                     device,
                     offsetsBytes,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     offsetsBuffer);
        createBuffer(physicalDevice,
                     device,
                     payloadBytes,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     payloadBuffer);

        createBuffer(physicalDevice,
                     device,
                     sizeof(float) * file.coarseAcScales.size(),
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingCoarseScales);
        createBuffer(physicalDevice,
                     device,
                     offsetsBytes,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingOffsets);
        createBuffer(physicalDevice,
                     device,
                     payloadBytes,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingPayload);

        if (!file.coarseAcScales.empty()) {
            vkMapMemory(device, stagingCoarseScales.memory, 0, stagingCoarseScales.size, 0, &mapped);
            std::memcpy(mapped, file.coarseAcScales.data(), sizeof(float) * file.coarseAcScales.size());
            vkUnmapMemory(device, stagingCoarseScales.memory);
        }
        vkMapMemory(device, stagingOffsets.memory, 0, stagingOffsets.size, 0, &mapped);
        std::memcpy(mapped, file.offsetsWords.data(), static_cast<size_t>(offsetsBytes));
        vkUnmapMemory(device, stagingOffsets.memory);
        vkMapMemory(device, stagingPayload.memory, 0, stagingPayload.size, 0, &mapped);
        std::memcpy(mapped, file.payloadWords.data(), static_cast<size_t>(payloadBytes));
        vkUnmapMemory(device, stagingPayload.memory);

        if (!file.coarseAcScales.empty()) {
            copyBuffer(device,
                       queue,
                       commandPool,
                       stagingCoarseScales.buffer,
                       coarseScaleBuffer.buffer,
                       sizeof(float) * file.coarseAcScales.size());
        }
        copyBuffer(device, queue, commandPool, stagingOffsets.buffer, offsetsBuffer.buffer, offsetsBytes);
        copyBuffer(device, queue, commandPool, stagingPayload.buffer, payloadBuffer.buffer, payloadBytes);

        VkDescriptorSetLayoutBinding bindings[5]{};
        for (uint32_t i = 0; i < 5; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = 5;
        layoutInfo.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout");
        }

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout");
        }

        const auto shaderCode = readFileBytes(shaderPath);
        shaderModule = createShaderModule(device, shaderCode);

        VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shaderModule;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = pipelineLayout;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute pipeline");
        }

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 5;
        VkDescriptorPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolCreateInfo.maxSets = 1;
        poolCreateInfo.poolSizeCount = 1;
        poolCreateInfo.pPoolSizes = &poolSize;
        if (vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool");
        }

        VkDescriptorSetAllocateInfo setAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        setAlloc.descriptorPool = descriptorPool;
        setAlloc.descriptorSetCount = 1;
        setAlloc.pSetLayouts = &descriptorSetLayout;
        if (vkAllocateDescriptorSets(device, &setAlloc, &descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor set");
        }

        VkDescriptorBufferInfo queryInfo{queryBuffer.buffer, 0, queryBuffer.size};
        VkDescriptorBufferInfo resultInfo{resultBuffer.buffer, 0, resultBuffer.size};
        VkDescriptorBufferInfo coarseInfo{coarseScaleBuffer.buffer, 0, coarseScaleBuffer.size};
        VkDescriptorBufferInfo offsetsInfo{offsetsBuffer.buffer, 0, offsetsBuffer.size};
        VkDescriptorBufferInfo payloadInfo{payloadBuffer.buffer, 0, payloadBuffer.size};

        std::array<VkWriteDescriptorSet, 5> writes{};
        const VkDescriptorBufferInfo infos[5] = {queryInfo, resultInfo, coarseInfo, offsetsInfo, payloadInfo};
        for (uint32_t i = 0; i < 5; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptorSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkQueryPoolCreateInfo queryPoolInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryPoolInfo.queryCount = 2;
        if (vkCreateQueryPool(device, &queryPoolInfo, nullptr, &timestampPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create timestamp query pool");
        }

        const auto uploadEnd = std::chrono::high_resolution_clock::now();
        outStats.uploadMs = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();

        const uint32_t leafCountX = (file.header.width + file.header.leafSize - 1u) / file.header.leafSize;
        const uint32_t leafCountY = (file.header.height + file.header.leafSize - 1u) / file.header.leafSize;
        PushConstants push{};
        push.dimX = file.header.width;
        push.dimY = file.header.height;
        push.dimZ = file.header.depth;
        push.frames = file.header.frames;
        push.leafSize = file.header.leafSize;
        push.leafCountX = leafCountX;
        push.leafCountY = leafCountY;
        push.queryCount = static_cast<uint32_t>(queries.size());
        push.coarseAcScaleCount = static_cast<uint32_t>(file.coarseAcScales.size());

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        vkCmdResetQueryPool(commandBuffer, timestampPool, 0, 2);
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampPool, 0);
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout,
                                0,
                                1,
                                &descriptorSet,
                                0,
                                nullptr);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(commandBuffer,
                           pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(PushConstants),
                           &push);
        const uint32_t groupSize = 32u;
        const uint32_t groupCount = (push.queryCount + groupSize - 1u) / groupSize;
        vkCmdDispatch(commandBuffer, groupCount, 1, 1);

        VkBufferMemoryBarrier resultBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        resultBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        resultBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        resultBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        resultBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        resultBarrier.buffer = resultBuffer.buffer;
        resultBarrier.offset = 0;
        resultBarrier.size = resultBuffer.size;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0,
                             0,
                             nullptr,
                             1,
                             &resultBarrier,
                             0,
                             nullptr);
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampPool, 1);
        vkEndCommandBuffer(commandBuffer);

        const auto dispatchStart = std::chrono::high_resolution_clock::now();
        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        vkQueueSubmit(queue, 1, &submitInfo, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        const auto dispatchEnd = std::chrono::high_resolution_clock::now();

        outStats.endToEndMs = std::chrono::duration<double, std::milli>(dispatchEnd - uploadStart).count();

        uint64_t timestamps[2]{};
        if (vkGetQueryPoolResults(device,
                                  timestampPool,
                                  0,
                                  2,
                                  sizeof(timestamps),
                                  timestamps,
                                  sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
            const double periodNs = static_cast<double>(limits.timestampPeriod);
            outStats.gpuDispatchMs = (static_cast<double>(timestamps[1] - timestamps[0]) * periodNs) * 1.0e-6;
        }

        const auto readbackStart = std::chrono::high_resolution_clock::now();
        outStats.results.resize(queries.size());
        vkMapMemory(device, resultBuffer.memory, 0, resultBuffer.size, 0, &mapped);
        std::memcpy(outStats.results.data(),
                    mapped,
                    sizeof(GpuQueryResult) * outStats.results.size());
        vkUnmapMemory(device, resultBuffer.memory);
        const auto readbackEnd = std::chrono::high_resolution_clock::now();
        outStats.readbackMs = std::chrono::duration<double, std::milli>(readbackEnd - readbackStart).count();

        uint32_t mismatchCount = 0;
        for (size_t i = 0; i < queries.size(); ++i) {
            const uint32_t expectedLeaf = queries[i].leafIndex;
            const uint32_t expectedHeader = readPackedHeaderWord(file, expectedLeaf);
            const uint32_t expectedMode = expectedHeader & 0x3u;
            const float expectedValue = decodeScientificValueAt(file, queries[i]);
            const auto& gpu = outStats.results[i];
            const float valueDiff = std::abs(gpu.value - expectedValue);
            if (gpu.leafIndex != expectedLeaf ||
                gpu.packedHeader != expectedHeader ||
                gpu.mode != expectedMode ||
                valueDiff > 1.0e-4f) {
                ++mismatchCount;
            }
        }
        outStats.mismatchCount = mismatchCount;
        outStats.queriesPerSec = (outStats.gpuDispatchMs > 0.0)
            ? (static_cast<double>(queries.size()) / (outStats.gpuDispatchMs * 1.0e-3))
            : 0.0;
        outStats.ok = true;

        vkDeviceWaitIdle(device);
        destroyBuffer(device, stagingCoarseScales);
        destroyBuffer(device, stagingOffsets);
        destroyBuffer(device, stagingPayload);
        destroyBuffer(device, coarseScaleBuffer);
        destroyBuffer(device, offsetsBuffer);
        destroyBuffer(device, payloadBuffer);
        destroyBuffer(device, queryBuffer);
        destroyBuffer(device, resultBuffer);
        if (timestampPool != VK_NULL_HANDLE) vkDestroyQueryPool(device, timestampPool, nullptr);
        if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
        if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        if (shaderModule != VK_NULL_HANDLE) vkDestroyShaderModule(device, shaderModule, nullptr);
        if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
        if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, nullptr);
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
        return true;
    } catch (const std::exception& ex) {
        outStats.error = ex.what();
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
        }
        destroyBuffer(device, stagingCoarseScales);
        destroyBuffer(device, stagingOffsets);
        destroyBuffer(device, stagingPayload);
        destroyBuffer(device, coarseScaleBuffer);
        destroyBuffer(device, offsetsBuffer);
        destroyBuffer(device, payloadBuffer);
        destroyBuffer(device, queryBuffer);
        destroyBuffer(device, resultBuffer);
        if (timestampPool != VK_NULL_HANDLE) vkDestroyQueryPool(device, timestampPool, nullptr);
        if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
        if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        if (shaderModule != VK_NULL_HANDLE) vkDestroyShaderModule(device, shaderModule, nullptr);
        if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
        if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, nullptr);
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
        return false;
    }
}

bool runDenseFrameCacheGpuBench(const VbtFile& file,
                                const std::vector<Query4D>& queries,
                                const std::string& shaderPath,
                                size_t maxDenseBytes,
                                GpuQueryBenchStats& outStats)
{
    outStats = {};
    if (file.header.version != 4) {
        outStats.error = "Dense GPU baseline requires VBTPACK4 files.";
        return false;
    }

    DenseFrameCache denseCache;
    if (!buildDenseFrameCache(file, queries, maxDenseBytes, denseCache, outStats.error)) {
        return false;
    }

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkQueryPool timestampPool = VK_NULL_HANDLE;
    Buffer queryBuffer{};
    Buffer resultBuffer{};
    Buffer frameLookupBuffer{};
    Buffer denseValuesBuffer{};
    Buffer stagingLookup{};
    Buffer stagingDense{};

    try {
        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.pApplicationName = "Dense Query Bench";
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.pEngineName = "None";
        appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instanceInfo.pApplicationInfo = &appInfo;
        if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance");
        }

        uint32_t physicalCount = 0;
        vkEnumeratePhysicalDevices(instance, &physicalCount, nullptr);
        if (physicalCount == 0) {
            throw std::runtime_error("No Vulkan physical device available");
        }
        std::vector<VkPhysicalDevice> physicalDevices(physicalCount);
        vkEnumeratePhysicalDevices(instance, &physicalCount, physicalDevices.data());

        uint32_t queueFamilyIndex = std::numeric_limits<uint32_t>::max();
        VkPhysicalDeviceLimits limits{};
        for (VkPhysicalDevice candidate : physicalDevices) {
            uint32_t queueCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueCount, nullptr);
            std::vector<VkQueueFamilyProperties> queueProps(queueCount);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueCount, queueProps.data());
            for (uint32_t i = 0; i < queueCount; ++i) {
                if ((queueProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    physicalDevice = candidate;
                    queueFamilyIndex = i;
                    VkPhysicalDeviceProperties properties{};
                    vkGetPhysicalDeviceProperties(candidate, &properties);
                    limits = properties.limits;
                    break;
                }
            }
            if (physicalDevice != VK_NULL_HANDLE) break;
        }
        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("No Vulkan compute queue family found");
        }

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueInfo.queueFamilyIndex = queueFamilyIndex;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan device");
        }
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool");
        }

        VkCommandBufferAllocateInfo cmdAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cmdAlloc.commandPool = commandPool;
        cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAlloc.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &cmdAlloc, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffer");
        }

        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create fence");
        }

        const auto uploadStart = std::chrono::high_resolution_clock::now();

        std::vector<GpuQueryPacked> packedQueries(queries.size());
        for (size_t i = 0; i < queries.size(); ++i) {
            packedQueries[i].x = queries[i].x;
            packedQueries[i].y = queries[i].y;
            packedQueries[i].z = queries[i].z;
            packedQueries[i].t = queries[i].t;
        }

        createBuffer(physicalDevice,
                     device,
                     sizeof(GpuQueryPacked) * packedQueries.size(),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     queryBuffer);
        void* mapped = nullptr;
        vkMapMemory(device, queryBuffer.memory, 0, queryBuffer.size, 0, &mapped);
        std::memcpy(mapped, packedQueries.data(), sizeof(GpuQueryPacked) * packedQueries.size());
        vkUnmapMemory(device, queryBuffer.memory);

        createBuffer(physicalDevice,
                     device,
                     sizeof(GpuQueryResult) * queries.size(),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     resultBuffer);

        const VkDeviceSize lookupBytes = sizeof(uint32_t) * denseCache.frameToSlot.size();
        const VkDeviceSize denseBytes = sizeof(float) * denseCache.denseValues.size();

        createBuffer(physicalDevice,
                     device,
                     lookupBytes,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     frameLookupBuffer);
        createBuffer(physicalDevice,
                     device,
                     denseBytes,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     denseValuesBuffer);

        createBuffer(physicalDevice,
                     device,
                     lookupBytes,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingLookup);
        createBuffer(physicalDevice,
                     device,
                     denseBytes,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingDense);

        vkMapMemory(device, stagingLookup.memory, 0, stagingLookup.size, 0, &mapped);
        std::memcpy(mapped, denseCache.frameToSlot.data(), static_cast<size_t>(lookupBytes));
        vkUnmapMemory(device, stagingLookup.memory);

        vkMapMemory(device, stagingDense.memory, 0, stagingDense.size, 0, &mapped);
        std::memcpy(mapped, denseCache.denseValues.data(), static_cast<size_t>(denseBytes));
        vkUnmapMemory(device, stagingDense.memory);

        copyBuffer(device, queue, commandPool, stagingLookup.buffer, frameLookupBuffer.buffer, lookupBytes);
        copyBuffer(device, queue, commandPool, stagingDense.buffer, denseValuesBuffer.buffer, denseBytes);

        VkDescriptorSetLayoutBinding bindings[4]{};
        for (uint32_t i = 0; i < 4; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = 4;
        layoutInfo.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout");
        }

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(DensePushConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout");
        }

        const auto shaderCode = readFileBytes(shaderPath);
        shaderModule = createShaderModule(device, shaderCode);

        VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shaderModule;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = pipelineLayout;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute pipeline");
        }

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 4;
        VkDescriptorPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolCreateInfo.maxSets = 1;
        poolCreateInfo.poolSizeCount = 1;
        poolCreateInfo.pPoolSizes = &poolSize;
        if (vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool");
        }

        VkDescriptorSetAllocateInfo setAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        setAlloc.descriptorPool = descriptorPool;
        setAlloc.descriptorSetCount = 1;
        setAlloc.pSetLayouts = &descriptorSetLayout;
        if (vkAllocateDescriptorSets(device, &setAlloc, &descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor set");
        }

        VkDescriptorBufferInfo queryInfo{queryBuffer.buffer, 0, queryBuffer.size};
        VkDescriptorBufferInfo resultInfo{resultBuffer.buffer, 0, resultBuffer.size};
        VkDescriptorBufferInfo lookupInfo{frameLookupBuffer.buffer, 0, frameLookupBuffer.size};
        VkDescriptorBufferInfo denseInfo{denseValuesBuffer.buffer, 0, denseValuesBuffer.size};

        std::array<VkWriteDescriptorSet, 4> writes{};
        const VkDescriptorBufferInfo infos[4] = {queryInfo, resultInfo, lookupInfo, denseInfo};
        for (uint32_t i = 0; i < 4; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptorSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkQueryPoolCreateInfo queryPoolInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryPoolInfo.queryCount = 2;
        if (vkCreateQueryPool(device, &queryPoolInfo, nullptr, &timestampPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create timestamp query pool");
        }

        const auto uploadEnd = std::chrono::high_resolution_clock::now();
        outStats.uploadMs = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();

        DensePushConstants push{};
        push.dimX = file.header.width;
        push.dimY = file.header.height;
        push.dimZ = file.header.depth;
        push.frames = file.header.frames;
        push.queryCount = static_cast<uint32_t>(queries.size());

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        vkCmdResetQueryPool(commandBuffer, timestampPool, 0, 2);
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampPool, 0);
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout,
                                0,
                                1,
                                &descriptorSet,
                                0,
                                nullptr);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(commandBuffer,
                           pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(DensePushConstants),
                           &push);
        const uint32_t groupSize = 32u;
        const uint32_t groupCount = (push.queryCount + groupSize - 1u) / groupSize;
        vkCmdDispatch(commandBuffer, groupCount, 1, 1);

        VkBufferMemoryBarrier resultBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        resultBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        resultBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        resultBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        resultBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        resultBarrier.buffer = resultBuffer.buffer;
        resultBarrier.offset = 0;
        resultBarrier.size = resultBuffer.size;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0,
                             0,
                             nullptr,
                             1,
                             &resultBarrier,
                             0,
                             nullptr);
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampPool, 1);
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        vkQueueSubmit(queue, 1, &submitInfo, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

        uint64_t timestamps[2]{};
        if (vkGetQueryPoolResults(device,
                                  timestampPool,
                                  0,
                                  2,
                                  sizeof(timestamps),
                                  timestamps,
                                  sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
            const double periodNs = static_cast<double>(limits.timestampPeriod);
            outStats.gpuDispatchMs = (static_cast<double>(timestamps[1] - timestamps[0]) * periodNs) * 1.0e-6;
        }

        const auto readbackStart = std::chrono::high_resolution_clock::now();
        outStats.results.resize(queries.size());
        vkMapMemory(device, resultBuffer.memory, 0, resultBuffer.size, 0, &mapped);
        std::memcpy(outStats.results.data(),
                    mapped,
                    sizeof(GpuQueryResult) * outStats.results.size());
        vkUnmapMemory(device, resultBuffer.memory);
        const auto readbackEnd = std::chrono::high_resolution_clock::now();
        outStats.readbackMs = std::chrono::duration<double, std::milli>(readbackEnd - readbackStart).count();

        uint32_t mismatchCount = 0;
        for (size_t i = 0; i < queries.size(); ++i) {
            const float valueDiff = std::abs(outStats.results[i].value - denseCache.expectedValues[i]);
            if (valueDiff > 1.0e-6f) {
                ++mismatchCount;
            }
        }
        outStats.mismatchCount = mismatchCount;
        outStats.endToEndMs = outStats.uploadMs + outStats.gpuDispatchMs + outStats.readbackMs;
        outStats.queriesPerSec = (outStats.gpuDispatchMs > 0.0)
            ? (static_cast<double>(queries.size()) / (outStats.gpuDispatchMs * 1.0e-3))
            : 0.0;
        outStats.ok = true;

        vkDeviceWaitIdle(device);
        destroyBuffer(device, stagingLookup);
        destroyBuffer(device, stagingDense);
        destroyBuffer(device, frameLookupBuffer);
        destroyBuffer(device, denseValuesBuffer);
        destroyBuffer(device, queryBuffer);
        destroyBuffer(device, resultBuffer);
        if (timestampPool != VK_NULL_HANDLE) vkDestroyQueryPool(device, timestampPool, nullptr);
        if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
        if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        if (shaderModule != VK_NULL_HANDLE) vkDestroyShaderModule(device, shaderModule, nullptr);
        if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
        if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, nullptr);
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
        return true;
    } catch (const std::exception& ex) {
        outStats.error = ex.what();
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
        }
        destroyBuffer(device, stagingLookup);
        destroyBuffer(device, stagingDense);
        destroyBuffer(device, frameLookupBuffer);
        destroyBuffer(device, denseValuesBuffer);
        destroyBuffer(device, queryBuffer);
        destroyBuffer(device, resultBuffer);
        if (timestampPool != VK_NULL_HANDLE) vkDestroyQueryPool(device, timestampPool, nullptr);
        if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
        if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        if (shaderModule != VK_NULL_HANDLE) vkDestroyShaderModule(device, shaderModule, nullptr);
        if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
        if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, nullptr);
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
        return false;
    }
}

} // namespace vbt::render
