#include "../../src/frame_metadata.h"

#include <vulkan/vulkan.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef VBT_USE_OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

namespace {

using NanoHandle = nanovdb::GridHandle<nanovdb::HostBuffer>;

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
    fs::path nvdbDir;
    fs::path outputSummaryJson;
    std::string framePrefix = "industrial_chimney_smoke_";
    std::string frameSuffix = ".nvdb";
    uint32_t frameDigits = 4;
    int frameStart = 96;
    uint32_t frameCount = 6;
    uint32_t ringSlots = 3;
    float densityScale = 1.0f;
    float compareTolerance = 1.0e-3f;
};

struct PushConstants {
    uint32_t imageWidth = 0;
    uint32_t imageHeight = 0;
    uint32_t rayCount = 0;
    uint32_t stepCount = 0;
    uint32_t gridByteOffset = 0;
    float densityScale = 1.0f;
};
static_assert(sizeof(PushConstants) <= 128, "Push constants too large");

struct FrameNanoMeta {
    fs::path path;
    uint64_t fileBytes = 0;
};

void printUsage()
{
    std::cout
        << "Usage: vbt_smoke_nanovdb_sequence_bench --probe-summary <summary.json> --ray-bin <rays.bin>\n"
        << "                                        --nvdb-dir <dir>\n"
        << "                                        [--frame-prefix industrial_chimney_smoke_]\n"
        << "                                        [--frame-suffix .nvdb]\n"
        << "                                        [--frame-digits 4]\n"
        << "                                        [--frame-start 96] [--frame-count 6] [--ring-slots 3]\n"
        << "                                        [--density-scale 1.0] [--compare-tol 1e-3]\n"
        << "                                        [--output-summary nanovdb_seq.json]\n";
}

bool parseArgs(int argc, char** argv, Options& opt)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--probe-summary" && i + 1 < argc) {
            opt.probeSummaryPath = argv[++i];
        } else if (arg == "--ray-bin" && i + 1 < argc) {
            opt.rayBinPath = argv[++i];
        } else if (arg == "--nvdb-dir" && i + 1 < argc) {
            opt.nvdbDir = argv[++i];
        } else if (arg == "--frame-prefix" && i + 1 < argc) {
            opt.framePrefix = argv[++i];
        } else if (arg == "--frame-suffix" && i + 1 < argc) {
            opt.frameSuffix = argv[++i];
        } else if (arg == "--frame-digits" && i + 1 < argc) {
            opt.frameDigits = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--frame-start" && i + 1 < argc) {
            opt.frameStart = std::stoi(argv[++i]);
        } else if (arg == "--frame-count" && i + 1 < argc) {
            opt.frameCount = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--ring-slots" && i + 1 < argc) {
            opt.ringSlots = static_cast<uint32_t>(std::stoul(argv[++i]));
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

    if (opt.probeSummaryPath.empty() || opt.rayBinPath.empty() || opt.nvdbDir.empty()) {
        printUsage();
        return false;
    }
    if (opt.frameCount == 0 || opt.ringSlots == 0 || opt.frameDigits == 0) {
        std::cerr << "frame-count, ring-slots and frame-digits must be > 0\n";
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

std::string frameNumberString(int frameIndex, uint32_t digits)
{
    std::ostringstream oss;
    oss << std::setw(static_cast<int>(digits)) << std::setfill('0') << frameIndex;
    return oss.str();
}

fs::path makeFramePath(const Options& opt, int frameIndex)
{
    return opt.nvdbDir / (opt.framePrefix + frameNumberString(frameIndex, opt.frameDigits) + opt.frameSuffix);
}

std::vector<FrameNanoMeta> collectFrameMeta(const Options& opt)
{
    std::vector<FrameNanoMeta> metas;
    metas.reserve(opt.frameCount);
    for (uint32_t i = 0; i < opt.frameCount; ++i) {
        const int frameIndex = opt.frameStart + static_cast<int>(i);
        const fs::path path = makeFramePath(opt, frameIndex);
        if (!fs::exists(path)) {
            throw std::runtime_error("Missing NanoVDB frame: " + path.string());
        }
        metas.push_back(FrameNanoMeta{path, fs::file_size(path)});
    }
    return metas;
}

NanoHandle loadNanoHandle(const fs::path& path)
{
    auto handle = nanovdb::io::readGrid<nanovdb::HostBuffer>(path.string(), 0, 0);
    const auto* grid = handle.grid<float>();
    if (!grid) {
        throw std::runtime_error("NanoVDB grid is null or not float: " + path.string());
    }
    return handle;
}

template <typename AccessorT>
float sampleNanoVdbTrilinear(const nanovdb::NanoGrid<float>& grid,
                             AccessorT& accessor,
                             float xWorld,
                             float yWorld,
                             float zWorld)
{
    const nanovdb::Vec3f xyz = grid.worldToIndexF(nanovdb::Vec3f(xWorld, yWorld, zWorld));
    const int x0 = static_cast<int>(std::floor(xyz[0]));
    const int y0 = static_cast<int>(std::floor(xyz[1]));
    const int z0 = static_cast<int>(std::floor(xyz[2]));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = xyz[0] - static_cast<float>(x0);
    const float ty = xyz[1] - static_cast<float>(y0);
    const float tz = xyz[2] - static_cast<float>(z0);

    auto at = [&](int sx, int sy, int sz) -> float {
        return accessor.getValue(nanovdb::Coord(sx, sy, sz));
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
                                   const nanovdb::NanoGrid<float>& grid,
                                   uint32_t stepCount,
                                   float densityScale)
{
    std::vector<float> out(rays.size(), 0.0f);
#ifdef VBT_USE_OPENMP
#pragma omp parallel
    {
        auto accessor = grid.getAccessor();
#pragma omp for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(rays.size()); ++i) {
            const SmokeRayPacked& ray = rays[static_cast<size_t>(i)];
            const float dt = ray.tMax / std::max(1u, stepCount);
            float t = ray.tMin + 0.5f * dt;
            float accum = 0.0f;
            for (uint32_t s = 0; s < stepCount; ++s) {
                const float x = ray.ox + ray.dx * t;
                const float y = ray.oy + ray.dy * t;
                const float z = ray.oz + ray.dz * t;
                accum += sampleNanoVdbTrilinear(grid, accessor, x, y, z) * densityScale;
                t += dt;
            }
            out[static_cast<size_t>(i)] = accum;
        }
    }
#else
    auto accessor = grid.getAccessor();
    for (size_t i = 0; i < rays.size(); ++i) {
        const SmokeRayPacked& ray = rays[i];
        const float dt = ray.tMax / std::max(1u, stepCount);
        float t = ray.tMin + 0.5f * dt;
        float accum = 0.0f;
        for (uint32_t s = 0; s < stepCount; ++s) {
            const float x = ray.ox + ray.dx * t;
            const float y = ray.oy + ray.dy * t;
            const float z = ray.oz + ray.dz * t;
            accum += sampleNanoVdbTrilinear(grid, accessor, x, y, z) * densityScale;
            t += dt;
        }
        out[i] = accum;
    }
#endif
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

void copyBufferRegion(VkDevice device,
                      VkQueue queue,
                      VkCommandPool commandPool,
                      VkBuffer src,
                      VkBuffer dst,
                      VkDeviceSize srcOffset,
                      VkDeviceSize dstOffset,
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
    copy.srcOffset = srcOffset;
    copy.dstOffset = dstOffset;
    copy.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &copy);

    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = dst;
    barrier.offset = dstOffset;
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

double averageMs(const std::vector<double>& xs)
{
    if (xs.empty()) return 0.0;
    return std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
}

double minMs(const std::vector<double>& xs)
{
    return xs.empty() ? 0.0 : *std::min_element(xs.begin(), xs.end());
}

double maxMs(const std::vector<double>& xs)
{
    return xs.empty() ? 0.0 : *std::max_element(xs.begin(), xs.end());
}

void writeSummary(const fs::path& path,
                  const Options& opt,
                  const ProbeSummary& probe,
                  uint64_t slotStrideBytes,
                  uint32_t preloadCount,
                  double cpuValidateMs,
                  uint32_t mismatchCount,
                  float meanAbsDiff,
                  float maxAbsDiff,
                  const std::vector<double>& dispatchMs,
                  const std::vector<double>& uploadMs)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open NanoVDB sequence summary output: " + path.string());
    }

    const double totalSamplesPerFrame =
        static_cast<double>(probe.rayCount) * static_cast<double>(probe.stepCount);
    const double avgDispatch = averageMs(dispatchMs);
    const double avgUpload = averageMs(uploadMs);
    const double avgSamplesPerSec = (avgDispatch > 0.0) ? (totalSamplesPerFrame / (avgDispatch * 1.0e-3)) : 0.0;
    const double avgRaysPerSec = (avgDispatch > 0.0) ? (static_cast<double>(probe.rayCount) / (avgDispatch * 1.0e-3)) : 0.0;

    out << "{\n";
    out << "  \"probe_summary\": \"" << jsonEscape(opt.probeSummaryPath.string()) << "\",\n";
    out << "  \"ray_bin\": \"" << jsonEscape(opt.rayBinPath.string()) << "\",\n";
    out << "  \"nvdb_dir\": \"" << jsonEscape(opt.nvdbDir.string()) << "\",\n";
    out << "  \"frame_prefix\": \"" << jsonEscape(opt.framePrefix) << "\",\n";
    out << "  \"frame_suffix\": \"" << jsonEscape(opt.frameSuffix) << "\",\n";
    out << "  \"frame_digits\": " << opt.frameDigits << ",\n";
    out << "  \"frame_start\": " << opt.frameStart << ",\n";
    out << "  \"frame_count\": " << opt.frameCount << ",\n";
    out << "  \"ring_slots\": " << opt.ringSlots << ",\n";
    out << "  \"preload_count\": " << preloadCount << ",\n";
    out << "  \"image_width\": " << probe.imageWidth << ",\n";
    out << "  \"image_height\": " << probe.imageHeight << ",\n";
    out << "  \"ray_count\": " << probe.rayCount << ",\n";
    out << "  \"step_count\": " << probe.stepCount << ",\n";
    out << "  \"slot_stride_bytes\": " << slotStrideBytes << ",\n";
    out << "  \"slot_stride_mb\": " << std::fixed << std::setprecision(3)
        << (static_cast<double>(slotStrideBytes) / 1.0e6) << ",\n";
    out << "  \"resident_total_mb\": "
        << (static_cast<double>(slotStrideBytes) * static_cast<double>(opt.ringSlots) / 1.0e6) << ",\n";
    out << "  \"density_scale\": " << opt.densityScale << ",\n";
    out << "  \"cpu_validate_ms\": " << cpuValidateMs << ",\n";
    out << "  \"compare_tolerance\": " << opt.compareTolerance << ",\n";
    out << "  \"mismatch_count\": " << mismatchCount << ",\n";
    out << "  \"mean_abs_diff\": " << meanAbsDiff << ",\n";
    out << "  \"max_abs_diff\": " << maxAbsDiff << ",\n";
    out << "  \"dispatch_avg_ms\": " << avgDispatch << ",\n";
    out << "  \"dispatch_min_ms\": " << minMs(dispatchMs) << ",\n";
    out << "  \"dispatch_max_ms\": " << maxMs(dispatchMs) << ",\n";
    out << "  \"upload_avg_ms\": " << avgUpload << ",\n";
    out << "  \"upload_min_ms\": " << minMs(uploadMs) << ",\n";
    out << "  \"upload_max_ms\": " << maxMs(uploadMs) << ",\n";
    out << "  \"steady_state_samples_per_sec\": " << avgSamplesPerSec << ",\n";
    out << "  \"steady_state_rays_per_sec\": " << avgRaysPerSec << "\n";
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
    VkFence fence = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkQueryPool timestampPool = VK_NULL_HANDLE;
    Buffer rayBuffer{};
    Buffer nanoBuffer{};
    Buffer resultBuffer{};
    Buffer stagingRayBuffer{};
    Buffer stagingNanoBuffer{};

    try {
        Options opt;
        if (!parseArgs(argc, argv, opt)) return 1;

        const ProbeSummary probe = loadProbeSummary(opt.probeSummaryPath);
        const std::vector<SmokeRayPacked> rays = loadRays(opt.rayBinPath, probe.rayCount);
        const std::vector<FrameNanoMeta> frameMetas = collectFrameMeta(opt);

        uint64_t slotStrideBytes = 0;
        for (const FrameNanoMeta& meta : frameMetas) {
            slotStrideBytes = std::max(slotStrideBytes, meta.fileBytes);
        }
        if (slotStrideBytes == 0) {
            throw std::runtime_error("NanoVDB slot stride resolved to 0");
        }

        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.pApplicationName = "vbt_smoke_nanovdb_sequence_bench";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instanceInfo.pApplicationInfo = &appInfo;
        if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance");
        }

        uint32_t physicalCount = 0;
        vkEnumeratePhysicalDevices(instance, &physicalCount, nullptr);
        if (physicalCount == 0) {
            throw std::runtime_error("No Vulkan physical devices found");
        }
        std::vector<VkPhysicalDevice> physicalDevices(physicalCount);
        vkEnumeratePhysicalDevices(instance, &physicalCount, physicalDevices.data());

        uint32_t queueFamilyIndex = UINT32_MAX;
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

        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create fence");
        }

        const VkDeviceSize rayBytes = static_cast<VkDeviceSize>(rays.size() * sizeof(SmokeRayPacked));
        const VkDeviceSize resultBytes = static_cast<VkDeviceSize>(rays.size() * sizeof(float));
        createBuffer(physicalDevice, device,
                     rayBytes,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     rayBuffer);
        createBuffer(physicalDevice, device,
                     static_cast<VkDeviceSize>(opt.ringSlots) * static_cast<VkDeviceSize>(slotStrideBytes),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     nanoBuffer);
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
                     static_cast<VkDeviceSize>(slotStrideBytes),
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingNanoBuffer);

        void* mapped = nullptr;
        vkMapMemory(device, stagingRayBuffer.memory, 0, stagingRayBuffer.size, 0, &mapped);
        std::memcpy(mapped, rays.data(), static_cast<size_t>(rayBytes));
        vkUnmapMemory(device, stagingRayBuffer.memory);
        copyBufferRegion(device, queue, commandPool, stagingRayBuffer.buffer, rayBuffer.buffer, 0, 0, rayBytes);

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

        const auto shaderCode = readFileBytes(VBT_NANOVDB_SMOKE_SHADER_SPV_PATH);
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
        VkDescriptorBufferInfo nanoInfo{nanoBuffer.buffer, 0, nanoBuffer.size};
        VkDescriptorBufferInfo resultInfo{resultBuffer.buffer, 0, resultBuffer.size};
        const VkDescriptorBufferInfo infos[3] = {rayInfo, nanoInfo, resultInfo};
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

        auto uploadToSlot = [&](const fs::path& framePath, uint32_t slot) -> std::pair<uint32_t, double> {
            const auto upStart = std::chrono::high_resolution_clock::now();
            NanoHandle handle = loadNanoHandle(framePath);
            const auto* bufferBase = reinterpret_cast<const uint8_t*>(handle.data());
            const auto* gridBase = reinterpret_cast<const uint8_t*>(handle.gridData(0));
            if (!bufferBase || !gridBase || gridBase < bufferBase) {
                throw std::runtime_error("Invalid NanoVDB buffer pointers: " + framePath.string());
            }
            const uint32_t gridByteOffset = static_cast<uint32_t>(gridBase - bufferBase);
            if (handle.bufferSize() > slotStrideBytes) {
                throw std::runtime_error("NanoVDB frame larger than slot stride: " + framePath.string());
            }

            vkMapMemory(device, stagingNanoBuffer.memory, 0, stagingNanoBuffer.size, 0, &mapped);
            std::memset(mapped, 0, static_cast<size_t>(slotStrideBytes));
            std::memcpy(mapped, handle.data(), static_cast<size_t>(handle.bufferSize()));
            vkUnmapMemory(device, stagingNanoBuffer.memory);
            copyBufferRegion(device, queue, commandPool,
                             stagingNanoBuffer.buffer,
                             nanoBuffer.buffer,
                             0,
                             static_cast<VkDeviceSize>(slot) * static_cast<VkDeviceSize>(slotStrideBytes),
                             static_cast<VkDeviceSize>(slotStrideBytes));
            const auto upEnd = std::chrono::high_resolution_clock::now();
            const double uploadMs =
                std::chrono::duration<double, std::milli>(upEnd - upStart).count();
            return {gridByteOffset, uploadMs};
        };

        const uint32_t preloadCount = std::min<uint32_t>(opt.ringSlots, opt.frameCount);
        std::vector<int> slotToFrame(opt.ringSlots, -1);
        std::vector<uint32_t> slotGridOffsets(opt.ringSlots, 0);
        std::vector<double> uploadMs;
        uploadMs.reserve(opt.frameCount);

        for (uint32_t slot = 0; slot < preloadCount; ++slot) {
            const int frameIndex = opt.frameStart + static_cast<int>(slot);
            const auto [gridOffset, upload] = uploadToSlot(frameMetas[slot].path, slot);
            slotToFrame[slot] = frameIndex;
            slotGridOffsets[slot] = gridOffset;
            uploadMs.push_back(upload);
        }

        std::vector<double> dispatchMs;
        dispatchMs.reserve(opt.frameCount);

        uint32_t mismatchCount = 0;
        float meanAbsDiff = 0.0f;
        float maxAbsDiff = 0.0f;
        double cpuValidateMs = 0.0;
        bool validated = false;

        for (uint32_t i = 0; i < opt.frameCount; ++i) {
            const int currentFrame = opt.frameStart + static_cast<int>(i);
            const uint32_t activeSlot = i % opt.ringSlots;
            if (slotToFrame[activeSlot] != currentFrame) {
                throw std::runtime_error("NanoVDB ring slot mapping corrupted before dispatch");
            }

            VkCommandBufferAllocateInfo cmdAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cmdAlloc.commandPool = commandPool;
            cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAlloc.commandBufferCount = 1;
            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(device, &cmdAlloc, &commandBuffer) != VK_SUCCESS) {
                throw std::runtime_error("Failed to allocate command buffer");
            }

            PushConstants push{};
            push.imageWidth = probe.imageWidth;
            push.imageHeight = probe.imageHeight;
            push.rayCount = probe.rayCount;
            push.stepCount = probe.stepCount;
            push.gridByteOffset = activeSlot * static_cast<uint32_t>(slotStrideBytes) + slotGridOffsets[activeSlot];
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
            vkResetFences(device, 1, &fence);
            vkQueueSubmit(queue, 1, &submitInfo, fence);
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

            uint64_t timestamps[2]{};
            double dispatch = 0.0;
            if (vkGetQueryPoolResults(device,
                                      timestampPool,
                                      0,
                                      2,
                                      sizeof(timestamps),
                                      timestamps,
                                      sizeof(uint64_t),
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
                const double periodNs = static_cast<double>(limits.timestampPeriod);
                dispatch = (static_cast<double>(timestamps[1] - timestamps[0]) * periodNs) * 1.0e-6;
            }
            dispatchMs.push_back(dispatch);

            if (!validated) {
                NanoHandle handle = loadNanoHandle(frameMetas[i].path);
                const auto* grid = handle.grid<float>();
                const auto cpuRefStart = std::chrono::high_resolution_clock::now();
                const std::vector<float> cpuReference =
                    runCpuReference(rays, *grid, probe.stepCount, opt.densityScale);
                const auto cpuRefEnd = std::chrono::high_resolution_clock::now();
                cpuValidateMs =
                    std::chrono::duration<double, std::milli>(cpuRefEnd - cpuRefStart).count();

                std::vector<float> gpuResults(rays.size(), 0.0f);
                vkMapMemory(device, resultBuffer.memory, 0, resultBuffer.size, 0, &mapped);
                std::memcpy(gpuResults.data(), mapped, static_cast<size_t>(resultBytes));
                vkUnmapMemory(device, resultBuffer.memory);

                double diffSum = 0.0;
                for (size_t j = 0; j < gpuResults.size(); ++j) {
                    const float diff = std::abs(gpuResults[j] - cpuReference[j]);
                    diffSum += static_cast<double>(diff);
                    maxAbsDiff = std::max(maxAbsDiff, diff);
                    if (diff > opt.compareTolerance) {
                        ++mismatchCount;
                    }
                }
                meanAbsDiff = gpuResults.empty()
                    ? 0.0f
                    : static_cast<float>(diffSum / static_cast<double>(gpuResults.size()));
                validated = true;
            }

            vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

            const uint32_t nextIndex = i + opt.ringSlots;
            if (nextIndex < opt.frameCount) {
                const int nextFrame = opt.frameStart + static_cast<int>(nextIndex);
                const auto [gridOffset, upload] = uploadToSlot(frameMetas[nextIndex].path, activeSlot);
                slotToFrame[activeSlot] = nextFrame;
                slotGridOffsets[activeSlot] = gridOffset;
                uploadMs.push_back(upload);
            }
        }

        std::cout << "NanoVDB smoke sequence benchmark\n";
        std::cout << "  frameStart: " << opt.frameStart << "\n";
        std::cout << "  frameCount: " << opt.frameCount << "\n";
        std::cout << "  ringSlots: " << opt.ringSlots << "\n";
        std::cout << "  preloadCount: " << preloadCount << "\n";
        std::cout << "  slotStrideMB: " << std::fixed << std::setprecision(3)
                  << (static_cast<double>(slotStrideBytes) / 1.0e6) << "\n";
        std::cout << "  residentTotalMB: "
                  << (static_cast<double>(slotStrideBytes) * static_cast<double>(opt.ringSlots) / 1.0e6) << "\n";
        std::cout << "  dispatchAvgMs: " << averageMs(dispatchMs) << "\n";
        std::cout << "  uploadAvgMs: " << averageMs(uploadMs) << "\n";
        std::cout << "  mismatchCount(>" << opt.compareTolerance << "): " << mismatchCount << "\n";

        if (!opt.outputSummaryJson.empty()) {
            writeSummary(opt.outputSummaryJson,
                         opt,
                         probe,
                         slotStrideBytes,
                         preloadCount,
                         cpuValidateMs,
                         mismatchCount,
                         meanAbsDiff,
                         maxAbsDiff,
                         dispatchMs,
                         uploadMs);
            std::cout << "  wrote summary: " << opt.outputSummaryJson.string() << "\n";
        }

        vkDeviceWaitIdle(device);
        destroyBuffer(device, stagingRayBuffer);
        destroyBuffer(device, stagingNanoBuffer);
        destroyBuffer(device, rayBuffer);
        destroyBuffer(device, nanoBuffer);
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
        std::cerr << "NanoVDB smoke sequence benchmark failed: " << ex.what() << "\n";
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
        }
        destroyBuffer(device, stagingRayBuffer);
        destroyBuffer(device, stagingNanoBuffer);
        destroyBuffer(device, rayBuffer);
        destroyBuffer(device, nanoBuffer);
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
