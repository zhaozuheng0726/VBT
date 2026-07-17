#include "../../src/frame_metadata.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef VBT_USE_OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

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

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct ProbeSummary {
    uint32_t imageWidth = 0;
    uint32_t imageHeight = 0;
    uint32_t rayCount = 0;
    uint32_t stepCount = 0;
    int frame = 0;
};

struct Options {
    fs::path probeSummaryPath;
    fs::path rayBinPath;
    fs::path rawPath;
    fs::path metadataPath;
    fs::path outputSummaryJson;
    int frameIndex = -1;
    float densityScale = 1.0f;
    float compareTolerance = 1.0e-3f;
};

struct PushConstants {
    uint32_t dimX = 0;
    uint32_t dimY = 0;
    uint32_t dimZ = 0;
    uint32_t imageWidth = 0;
    uint32_t imageHeight = 0;
    uint32_t rayCount = 0;
    uint32_t stepCount = 0;
    uint32_t slotStrideFloats = 0;
    uint32_t activeSlot = 0;
    float bboxMinX = 0.0f;
    float bboxMinY = 0.0f;
    float bboxMinZ = 0.0f;
    float densityScale = 1.0f;
};
static_assert(sizeof(PushConstants) <= 128, "Push constants too large");

void printUsage()
{
    std::cout
        << "Usage: vbt_smoke_dense_bench --probe-summary <summary.json> --ray-bin <rays.bin>\n"
        << "                             --raw <density.raw> --metadata <meta.json>\n"
        << "                             [--frame 100] [--density-scale 1.0]\n"
        << "                             [--compare-tol 1e-3] [--output-summary dense_summary.json]\n";
}

bool parseArgs(int argc, char** argv, Options& opt)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--probe-summary" && i + 1 < argc) {
            opt.probeSummaryPath = argv[++i];
        } else if (arg == "--ray-bin" && i + 1 < argc) {
            opt.rayBinPath = argv[++i];
        } else if (arg == "--raw" && i + 1 < argc) {
            opt.rawPath = argv[++i];
        } else if (arg == "--metadata" && i + 1 < argc) {
            opt.metadataPath = argv[++i];
        } else if (arg == "--frame" && i + 1 < argc) {
            opt.frameIndex = std::stoi(argv[++i]);
        } else if (arg == "--density-scale" && i + 1 < argc) {
            opt.densityScale = std::stof(argv[++i]);
        } else if (arg == "--compare-tol" && i + 1 < argc) {
            opt.compareTolerance = std::stof(argv[++i]);
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

    if (opt.probeSummaryPath.empty() || opt.rayBinPath.empty() ||
        opt.rawPath.empty() || opt.metadataPath.empty()) {
        printUsage();
        return false;
    }
    return true;
}

ProbeSummary loadProbeSummary(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open probe summary: " + path.string());
    }
    nlohmann::json document;
    try {
        in >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Failed to parse probe summary '" + path.string() + "': " + error.what());
    }
    auto readInt = [&](const char* key) {
        const auto it = document.find(key);
        if (it == document.end() || !it->is_number_integer()) {
            throw std::runtime_error(
                "Probe summary field must be an integer: " + std::string(key));
        }
        return it->get<int>();
    };
    ProbeSummary summary;
    summary.imageWidth = static_cast<uint32_t>(readInt("image_width"));
    summary.imageHeight = static_cast<uint32_t>(readInt("image_height"));
    summary.rayCount = static_cast<uint32_t>(readInt("ray_count"));
    summary.stepCount = static_cast<uint32_t>(readInt("step_count"));
    summary.frame = readInt("frame");
    if (summary.imageWidth == 0 || summary.imageHeight == 0 ||
        summary.rayCount == 0 || summary.stepCount == 0) {
        throw std::runtime_error("Invalid probe summary: " + path.string());
    }
    return summary;
}

std::vector<SmokeRayPacked> loadRays(const fs::path& path, uint32_t expectedCount)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open ray bin: " + path.string());
    }
    const std::streamsize size = in.tellg();
    if (size < 0 || (size % static_cast<std::streamsize>(sizeof(SmokeRayPacked))) != 0) {
        throw std::runtime_error("Invalid ray bin size: " + path.string());
    }
    const size_t rayCount = static_cast<size_t>(size) / sizeof(SmokeRayPacked);
    if (expectedCount != 0 && rayCount != expectedCount) {
        throw std::runtime_error("Ray bin count mismatch with probe summary");
    }
    std::vector<SmokeRayPacked> rays(rayCount);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(rays.data()), size);
    return rays;
}

std::vector<float> loadDenseFrame(const fs::path& rawPath, const vbt::FrameMetadata& meta, int frameIndex)
{
    const size_t voxelCount = vbt::frameVoxelCount(meta);
    const uint64_t frameBytes = static_cast<uint64_t>(voxelCount) * sizeof(float);
    const uint64_t offset = static_cast<uint64_t>(frameIndex) * frameBytes;

    std::ifstream in(rawPath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open raw file: " + rawPath.string());
    }
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) {
        throw std::runtime_error("Failed to seek raw frame in: " + rawPath.string());
    }

    std::vector<float> frame(voxelCount, 0.0f);
    in.read(reinterpret_cast<char*>(frame.data()), static_cast<std::streamsize>(frameBytes));
    if (!in) {
        throw std::runtime_error("Failed to read dense frame from: " + rawPath.string());
    }
    return frame;
}

inline size_t denseIndex(const vbt::FrameMetadata& meta, int x, int y, int z)
{
    return (static_cast<size_t>(z) * static_cast<size_t>(meta.height) + static_cast<size_t>(y)) *
               static_cast<size_t>(meta.width) +
           static_cast<size_t>(x);
}

float sampleDenseTrilinear(const std::vector<float>& dense,
                           const vbt::FrameMetadata& meta,
                           float xWorld,
                           float yWorld,
                           float zWorld)
{
    const float x = xWorld - static_cast<float>(meta.bboxMin[0]);
    const float y = yWorld - static_cast<float>(meta.bboxMin[1]);
    const float z = zWorld - static_cast<float>(meta.bboxMin[2]);
    if (x < 0.0f || y < 0.0f || z < 0.0f ||
        x > static_cast<float>(meta.width - 1) ||
        y > static_cast<float>(meta.height - 1) ||
        z > static_cast<float>(meta.depth - 1)) {
        return 0.0f;
    }

    const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, meta.width - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, meta.height - 1);
    const int z0 = std::clamp(static_cast<int>(std::floor(z)), 0, meta.depth - 1);
    const int x1 = std::min(x0 + 1, meta.width - 1);
    const int y1 = std::min(y0 + 1, meta.height - 1);
    const int z1 = std::min(z0 + 1, meta.depth - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float tz = z - static_cast<float>(z0);

    auto at = [&](int sx, int sy, int sz) -> float {
        return dense[denseIndex(meta, sx, sy, sz)];
    };

    const float c00 = at(x0, y0, z0) * (1.0f - tx) + at(x1, y0, z0) * tx;
    const float c10 = at(x0, y1, z0) * (1.0f - tx) + at(x1, y1, z0) * tx;
    const float c01 = at(x0, y0, z1) * (1.0f - tx) + at(x1, y0, z1) * tx;
    const float c11 = at(x0, y1, z1) * (1.0f - tx) + at(x1, y1, z1) * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

std::vector<float> runCpuReference(const std::vector<SmokeRayPacked>& rays,
                                   const std::vector<float>& dense,
                                   const vbt::FrameMetadata& meta,
                                   uint32_t stepCount,
                                   float densityScale)
{
    std::vector<float> out(rays.size(), 0.0f);
#ifdef VBT_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(rays.size()); ++i) {
        const SmokeRayPacked& ray = rays[static_cast<size_t>(i)];
        const float dt = ray.tMax / std::max(1u, stepCount);
        float t = ray.tMin + 0.5f * dt;
        float accum = 0.0f;
        for (uint32_t s = 0; s < stepCount; ++s) {
            const float x = ray.ox + ray.dx * t;
            const float y = ray.oy + ray.dy * t;
            const float z = ray.oz + ray.dz * t;
            accum += sampleDenseTrilinear(dense, meta, x, y, z) * densityScale;
            t += dt;
        }
        out[static_cast<size_t>(i)] = accum;
    }
    return out;
}

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
        throw std::runtime_error("Failed to allocate Vulkan memory");
    }
    if (vkBindBufferMemory(device, out.buffer, out.memory, 0) != VK_SUCCESS) {
        throw std::runtime_error("Failed to bind Vulkan buffer memory");
    }
    out.size = bufferInfo.size;
}

void destroyBuffer(VkDevice device, Buffer& buffer)
{
    if (device == VK_NULL_HANDLE) return;
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
                  const ProbeSummary& probe,
                  const vbt::FrameMetadata& meta,
                  size_t frameVoxelCount,
                  double cpuMs,
                  double uploadMs,
                  double dispatchMs,
                  double readbackMs,
                  float minValue,
                  float maxValue,
                  float avgValue,
                  uint32_t mismatchCount,
                  float meanAbsDiff,
                  float maxAbsDiff)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open smoke dense summary output: " + path.string());
    }

    const double totalSamples = static_cast<double>(probe.rayCount) * static_cast<double>(probe.stepCount);
    const double samplesPerSec = (dispatchMs > 0.0) ? (totalSamples / (dispatchMs * 1.0e-3)) : 0.0;
    const double raysPerSec = (dispatchMs > 0.0) ? (static_cast<double>(probe.rayCount) / (dispatchMs * 1.0e-3)) : 0.0;

    out << "{\n";
    out << "  \"probe_summary\": \"" << jsonEscape(opt.probeSummaryPath.string()) << "\",\n";
    out << "  \"ray_bin\": \"" << jsonEscape(opt.rayBinPath.string()) << "\",\n";
    out << "  \"raw\": \"" << jsonEscape(opt.rawPath.string()) << "\",\n";
    out << "  \"metadata\": \"" << jsonEscape(opt.metadataPath.string()) << "\",\n";
    out << "  \"frame\": " << ((opt.frameIndex >= 0) ? opt.frameIndex : probe.frame) << ",\n";
    out << "  \"image_width\": " << probe.imageWidth << ",\n";
    out << "  \"image_height\": " << probe.imageHeight << ",\n";
    out << "  \"ray_count\": " << probe.rayCount << ",\n";
    out << "  \"step_count\": " << probe.stepCount << ",\n";
    out << "  \"frame_voxel_count\": " << frameVoxelCount << ",\n";
    out << "  \"frame_dense_mb\": " << std::fixed << std::setprecision(3)
        << (static_cast<double>(frameVoxelCount) * sizeof(float) / 1.0e6) << ",\n";
    out << "  \"density_scale\": " << opt.densityScale << ",\n";
    out << "  \"cpu_reference_ms\": " << cpuMs << ",\n";
    out << "  \"upload_ms\": " << uploadMs << ",\n";
    out << "  \"gpu_dispatch_ms\": " << dispatchMs << ",\n";
    out << "  \"readback_ms\": " << readbackMs << ",\n";
    out << "  \"samples_per_sec\": " << samplesPerSec << ",\n";
    out << "  \"rays_per_sec\": " << raysPerSec << ",\n";
    out << "  \"result_min\": " << minValue << ",\n";
    out << "  \"result_max\": " << maxValue << ",\n";
    out << "  \"result_avg\": " << avgValue << ",\n";
    out << "  \"compare_tolerance\": " << opt.compareTolerance << ",\n";
    out << "  \"mismatch_count\": " << mismatchCount << ",\n";
    out << "  \"mean_abs_diff\": " << meanAbsDiff << ",\n";
    out << "  \"max_abs_diff\": " << maxAbsDiff << "\n";
    out << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkQueryPool timestampPool = VK_NULL_HANDLE;
    Buffer rayBuffer{};
    Buffer resultBuffer{};
    Buffer denseBuffer{};
    Buffer stagingRayBuffer{};
    Buffer stagingDenseBuffer{};

    try {
        Options opt;
        if (!parseArgs(argc, argv, opt)) return 1;

        const ProbeSummary probe = loadProbeSummary(opt.probeSummaryPath);
        const vbt::FrameMetadata meta = vbt::loadFrameMetadata(opt.metadataPath);
        const int frameIndex = (opt.frameIndex >= 0) ? opt.frameIndex : probe.frame;
        if (frameIndex < 0 || frameIndex >= meta.frames) {
            throw std::runtime_error("Frame index out of range for dense smoke benchmark");
        }

        const std::vector<SmokeRayPacked> rays = loadRays(opt.rayBinPath, probe.rayCount);
        const auto cpuLoadStart = std::chrono::high_resolution_clock::now();
        const std::vector<float> denseFrame = loadDenseFrame(opt.rawPath, meta, frameIndex);
        const auto cpuRefStart = std::chrono::high_resolution_clock::now();
        const std::vector<float> cpuReference =
            runCpuReference(rays, denseFrame, meta, probe.stepCount, opt.densityScale);
        const auto cpuRefEnd = std::chrono::high_resolution_clock::now();
        const double cpuRefMs =
            std::chrono::duration<double, std::milli>(cpuRefEnd - cpuRefStart).count();
        (void)cpuLoadStart;

        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.pApplicationName = "Smoke Dense Bench";
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
        const VkDeviceSize rayBytes = static_cast<VkDeviceSize>(rays.size() * sizeof(SmokeRayPacked));
        const VkDeviceSize denseBytes = static_cast<VkDeviceSize>(denseFrame.size() * sizeof(float));
        const VkDeviceSize resultBytes = static_cast<VkDeviceSize>(rays.size() * sizeof(float));

        createBuffer(physicalDevice, device,
                     rayBytes,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     rayBuffer);
        createBuffer(physicalDevice, device,
                     denseBytes,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     denseBuffer);
        createBuffer(physicalDevice, device,
                     resultBytes,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     resultBuffer);
        createBuffer(physicalDevice, device,
                     rayBytes,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingRayBuffer);
        createBuffer(physicalDevice, device,
                     denseBytes,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingDenseBuffer);

        void* mapped = nullptr;
        vkMapMemory(device, stagingRayBuffer.memory, 0, stagingRayBuffer.size, 0, &mapped);
        std::memcpy(mapped, rays.data(), static_cast<size_t>(rayBytes));
        vkUnmapMemory(device, stagingRayBuffer.memory);

        vkMapMemory(device, stagingDenseBuffer.memory, 0, stagingDenseBuffer.size, 0, &mapped);
        std::memcpy(mapped, denseFrame.data(), static_cast<size_t>(denseBytes));
        vkUnmapMemory(device, stagingDenseBuffer.memory);

        copyBuffer(device, queue, commandPool, stagingRayBuffer.buffer, rayBuffer.buffer, rayBytes);
        copyBuffer(device, queue, commandPool, stagingDenseBuffer.buffer, denseBuffer.buffer, denseBytes);

        VkDescriptorSetLayoutBinding bindings[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = 3;
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

        const auto shaderCode = readFileBytes(VBT_DENSE_SMOKE_SHADER_SPV_PATH);
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
        poolSize.descriptorCount = 3;
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

        VkDescriptorBufferInfo rayInfo{rayBuffer.buffer, 0, rayBuffer.size};
        VkDescriptorBufferInfo denseInfo{denseBuffer.buffer, 0, denseBuffer.size};
        VkDescriptorBufferInfo resultInfo{resultBuffer.buffer, 0, resultBuffer.size};
        const VkDescriptorBufferInfo infos[3] = {rayInfo, denseInfo, resultInfo};
        std::array<VkWriteDescriptorSet, 3> writes{};
        for (uint32_t i = 0; i < 3; ++i) {
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
        const double uploadMs =
            std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();

        PushConstants push{};
        push.dimX = static_cast<uint32_t>(meta.width);
        push.dimY = static_cast<uint32_t>(meta.height);
        push.dimZ = static_cast<uint32_t>(meta.depth);
        push.imageWidth = probe.imageWidth;
        push.imageHeight = probe.imageHeight;
        push.rayCount = probe.rayCount;
        push.stepCount = probe.stepCount;
        push.slotStrideFloats = static_cast<uint32_t>(denseFrame.size());
        push.activeSlot = 0;
        push.bboxMinX = static_cast<float>(meta.bboxMin[0]);
        push.bboxMinY = static_cast<float>(meta.bboxMin[1]);
        push.bboxMinZ = static_cast<float>(meta.bboxMin[2]);
        push.densityScale = opt.densityScale;

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
        const uint32_t groupX = (probe.imageWidth + 7u) / 8u;
        const uint32_t groupY = (probe.imageHeight + 7u) / 8u;
        vkCmdDispatch(commandBuffer, groupX, groupY, 1);

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
        double dispatchMs = 0.0;
        if (vkGetQueryPoolResults(device,
                                  timestampPool,
                                  0,
                                  2,
                                  sizeof(timestamps),
                                  timestamps,
                                  sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
            const double periodNs = static_cast<double>(limits.timestampPeriod);
            dispatchMs = (static_cast<double>(timestamps[1] - timestamps[0]) * periodNs) * 1.0e-6;
        }

        const auto readbackStart = std::chrono::high_resolution_clock::now();
        std::vector<float> gpuResults(rays.size(), 0.0f);
        vkMapMemory(device, resultBuffer.memory, 0, resultBuffer.size, 0, &mapped);
        std::memcpy(gpuResults.data(), mapped, static_cast<size_t>(resultBytes));
        vkUnmapMemory(device, resultBuffer.memory);
        const auto readbackEnd = std::chrono::high_resolution_clock::now();
        const double readbackMs =
            std::chrono::duration<double, std::milli>(readbackEnd - readbackStart).count();

        uint32_t mismatchCount = 0;
        double diffSum = 0.0;
        float maxAbsDiff = 0.0f;
        float minValue = std::numeric_limits<float>::max();
        float maxValue = std::numeric_limits<float>::lowest();
        double avgAccum = 0.0;
        for (size_t i = 0; i < gpuResults.size(); ++i) {
            const float value = gpuResults[i];
            const float diff = std::abs(value - cpuReference[i]);
            diffSum += static_cast<double>(diff);
            maxAbsDiff = std::max(maxAbsDiff, diff);
            if (diff > opt.compareTolerance) {
                ++mismatchCount;
            }
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
            avgAccum += static_cast<double>(value);
        }
        const float meanAbsDiff = gpuResults.empty()
            ? 0.0f
            : static_cast<float>(diffSum / static_cast<double>(gpuResults.size()));
        const float avgValue = gpuResults.empty()
            ? 0.0f
            : static_cast<float>(avgAccum / static_cast<double>(gpuResults.size()));

        std::cout << "Dense smoke GPU benchmark\n";
        std::cout << "  frame: " << frameIndex << "\n";
        std::cout << "  rays: " << probe.rayCount << "\n";
        std::cout << "  image: " << probe.imageWidth << " x " << probe.imageHeight << "\n";
        std::cout << "  stepCount: " << probe.stepCount << "\n";
        std::cout << "  frameDenseMB: " << std::fixed << std::setprecision(3)
                  << (static_cast<double>(denseFrame.size()) * sizeof(float) / 1.0e6) << "\n";
        std::cout << "  cpuReferenceMs: " << cpuRefMs << "\n";
        std::cout << "  uploadMs: " << uploadMs << "\n";
        std::cout << "  gpuDispatchMs: " << dispatchMs << "\n";
        std::cout << "  readbackMs: " << readbackMs << "\n";
        std::cout << "  meanAbsDiff: " << meanAbsDiff << "\n";
        std::cout << "  maxAbsDiff: " << maxAbsDiff << "\n";
        std::cout << "  mismatchCount(>" << opt.compareTolerance << "): " << mismatchCount << "\n";

        if (!opt.outputSummaryJson.empty()) {
            writeSummary(opt.outputSummaryJson,
                         opt,
                         probe,
                         meta,
                         denseFrame.size(),
                         cpuRefMs,
                         uploadMs,
                         dispatchMs,
                         readbackMs,
                         minValue,
                         maxValue,
                         avgValue,
                         mismatchCount,
                         meanAbsDiff,
                         maxAbsDiff);
            std::cout << "  wrote summary: " << opt.outputSummaryJson.string() << "\n";
        }

        vkDeviceWaitIdle(device);
        destroyBuffer(device, stagingRayBuffer);
        destroyBuffer(device, stagingDenseBuffer);
        destroyBuffer(device, rayBuffer);
        destroyBuffer(device, denseBuffer);
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
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Dense smoke benchmark failed: " << ex.what() << "\n";
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
        }
        destroyBuffer(device, stagingRayBuffer);
        destroyBuffer(device, stagingDenseBuffer);
        destroyBuffer(device, rayBuffer);
        destroyBuffer(device, denseBuffer);
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
        return 10;
    }
}
