#include "volume_renderer.h"

#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#include <wincodec.h>
#endif

namespace vbtstudio::frontend {
namespace {

struct RayPushConstants {
    std::uint32_t image_width = 0;
    std::uint32_t image_height = 0;
    float bbox_min_x = 0.0f;
    float bbox_min_y = 0.0f;
    float bbox_min_z = 0.0f;
    float bbox_max_x = 0.0f;
    float bbox_max_y = 0.0f;
    float bbox_max_z = 0.0f;
    float yaw = 0.0f;
    float pitch = 0.0f;
    float distance = 0.0f;
    float field_of_view = 0.0f;
    float target_offset_x = 0.0f;
    float target_offset_y = 0.0f;
    float target_offset_z = 0.0f;
    std::uint32_t up_axis = 1;
};

struct SamplePushConstants {
    std::uint32_t dim_x = 0;
    std::uint32_t dim_y = 0;
    std::uint32_t dim_z = 0;
    std::uint32_t frames = 0;
    std::uint32_t leaf_size = 0;
    std::uint32_t leaf_count_x = 0;
    std::uint32_t leaf_count_y = 0;
    std::uint32_t image_width = 0;
    std::uint32_t image_height = 0;
    std::uint32_t ray_count = 0;
    std::uint32_t step_count = 0;
    std::int32_t frame_index = 0;
    float bbox_min_x = 0.0f;
    float bbox_min_y = 0.0f;
    float bbox_min_z = 0.0f;
    std::uint32_t render_mode = 0;
    float voxel_size = 1.0f;
    float background_value = 0.0f;
    float surface_iso = 0.0f;
    float surface_epsilon_voxels = 0.08f;
    float surface_normal_step = 0.75f;
    float max_step_voxels = 4.0f;
    std::uint32_t surface_model = 0;
    float floor_height = 0.0f;
    std::uint32_t shadow_steps = 0;
    std::uint32_t padding = 0;
};

struct CachePushConstants {
    std::uint32_t frames = 0;
    std::uint32_t leaf_count = 0;
    std::int32_t frame_index = 0;
    std::uint32_t padding = 0;
};

struct ActiveIndexPushConstants {
    std::uint32_t leaf_count = 0;
};

struct ColorPushConstants {
    std::uint32_t image_width = 0;
    std::uint32_t image_height = 0;
    float density_scale = 0.0f;
    float density_threshold = 0.0f;
    float density_gamma = 1.0f;
    float exposure = 0.0f;
    std::uint32_t physical_mode = 0;
    std::uint32_t padding1 = 0;
    std::array<float, 4> smoke_color{};
    std::array<float, 4> background_color{};
    float sample_distance = 2.0f;
    std::uint32_t step_count = 0;
    float flame_strength = 0.0f;
    float flame_threshold = 0.0f;
    std::uint32_t has_flames = 0;
    float temperature_min = 0.0f;
    float temperature_max = 0.0f;
    std::uint32_t has_temperature = 0;
    std::uint32_t render_mode = 0;
    float surface_roughness = 0.0f;
    float surface_metallic = 0.0f;
    float surface_opacity = 1.0f;
    std::array<float, 4> surface_color{};
};

static_assert(sizeof(RayPushConstants) == 64);
static_assert(sizeof(SamplePushConstants) == 104);
static_assert(sizeof(CachePushConstants) == 16);
static_assert(sizeof(ActiveIndexPushConstants) == 4);
static_assert(sizeof(ColorPushConstants) == 128);

constexpr VkDeviceSize cached_controls_per_leaf = 576u;

std::vector<char> read_binary_file(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Unable to open shader: " + path.string());
    const std::streamsize size = input.tellg();
    if (size <= 0 || size % 4 != 0) throw std::runtime_error("Invalid SPIR-V shader: " + path.string());
    std::vector<char> bytes(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(bytes.data(), size);
    if (!input) throw std::runtime_error("Unable to read shader: " + path.string());
    return bytes;
}

VkShaderModule create_shader_module(VkDevice device, const std::filesystem::path& path)
{
    const std::vector<char> bytes = read_binary_file(path);
    VkShaderModuleCreateInfo create_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    create_info.codeSize = bytes.size();
    create_info.pCode = reinterpret_cast<const std::uint32_t*>(bytes.data());
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &create_info, nullptr, &module) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create shader module: " + path.string());
    }
    return module;
}

bool material_equal(const backend::MaterialState& left, const backend::MaterialState& right)
{
    return left.density_scale == right.density_scale &&
           left.density_threshold == right.density_threshold &&
           left.density_gamma == right.density_gamma && left.exposure == right.exposure &&
           left.flame_strength == right.flame_strength && left.flame_threshold == right.flame_threshold &&
           left.temperature_min == right.temperature_min && left.temperature_max == right.temperature_max &&
           left.sample_steps == right.sample_steps && left.volume_model == right.volume_model &&
           left.fire_scattering == right.fire_scattering &&
           left.fire_blackbody_mix == right.fire_blackbody_mix && left.fire_glow == right.fire_glow &&
           left.surface_model == right.surface_model &&
           left.surface_iso == right.surface_iso &&
           left.surface_epsilon_voxels == right.surface_epsilon_voxels &&
           left.surface_normal_step == right.surface_normal_step &&
           left.surface_roughness == right.surface_roughness &&
           left.surface_metallic == right.surface_metallic &&
           left.surface_opacity == right.surface_opacity &&
           left.water_ior == right.water_ior && left.absorption_density == right.absorption_density &&
           left.reflection_strength == right.reflection_strength &&
           left.environment_strength == right.environment_strength &&
           left.floor_offset == right.floor_offset && left.shadow_strength == right.shadow_strength &&
           left.shadow_steps == right.shadow_steps &&
           left.smoke_color == right.smoke_color && left.fire_tint == right.fire_tint &&
           left.surface_color == right.surface_color &&
           left.absorption_color == right.absorption_color &&
           left.background_color == right.background_color;
}

float half_to_float(std::uint16_t half)
{
    const std::uint32_t sign = (half >> 15u) & 1u;
    std::uint32_t exponent = (half >> 10u) & 0x1fu;
    std::uint32_t mantissa = half & 0x3ffu;
    std::uint32_t bits = 0;
    if (exponent == 0u) {
        if (mantissa != 0u) {
            int normalized_exponent = -14;
            while ((mantissa & 0x400u) == 0u) {
                mantissa <<= 1u;
                --normalized_exponent;
            }
            mantissa &= 0x3ffu;
            bits = (sign << 31u) | (static_cast<std::uint32_t>(normalized_exponent + 127) << 23u) |
                   (mantissa << 13u);
        }
        else {
            bits = sign << 31u;
        }
    }
    else if (exponent == 31u) {
        bits = (sign << 31u) | 0x7f800000u | (mantissa << 13u);
    }
    else {
        bits = (sign << 31u) | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

#ifdef _WIN32
bool write_png(const std::filesystem::path& path,
               const std::vector<std::uint8_t>& pixels,
               std::uint32_t width,
               std::uint32_t height)
{
    const HRESULT initialize_result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool uninitialize = SUCCEEDED(initialize_result);
    IWICImagingFactory* factory = nullptr;
    IWICStream* stream = nullptr;
    IWICBitmapEncoder* encoder = nullptr;
    IWICBitmapFrameEncode* frame = nullptr;
    IPropertyBag2* properties = nullptr;
    bool success = false;
    if (SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory,
                                   nullptr,
                                   CLSCTX_INPROC_SERVER,
                                   IID_PPV_ARGS(&factory))) &&
        SUCCEEDED(factory->CreateStream(&stream)) &&
        SUCCEEDED(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE)) &&
        SUCCEEDED(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder)) &&
        SUCCEEDED(encoder->Initialize(stream, WICBitmapEncoderNoCache)) &&
        SUCCEEDED(encoder->CreateNewFrame(&frame, &properties)) &&
        SUCCEEDED(frame->Initialize(properties)) && SUCCEEDED(frame->SetSize(width, height))) {
        WICPixelFormatGUID format = GUID_WICPixelFormat32bppBGRA;
        success = SUCCEEDED(frame->SetPixelFormat(&format)) && format == GUID_WICPixelFormat32bppBGRA &&
                  SUCCEEDED(frame->WritePixels(height,
                                               width * 4u,
                                               static_cast<UINT>(pixels.size()),
                                               const_cast<BYTE*>(pixels.data()))) &&
                  SUCCEEDED(frame->Commit()) && SUCCEEDED(encoder->Commit());
    }
    if (properties) properties->Release();
    if (frame) frame->Release();
    if (encoder) encoder->Release();
    if (stream) stream->Release();
    if (factory) factory->Release();
    if (uninitialize) CoUninitialize();
    return success;
}
#endif

} // namespace

VolumeRenderer::VolumeRenderer(VkPhysicalDevice physical_device,
                               VkDevice device,
                               VkQueue queue,
                               std::uint32_t queue_family)
    : physical_device_(physical_device), device_(device), queue_(queue), queue_family_(queue_family)
{
    create_pipeline_resources();
    initialized_ = true;
}

VolumeRenderer::~VolumeRenderer() { shutdown(); }

void VolumeRenderer::sync_asset(const std::optional<backend::VbtAssetInfo>& asset)
{
    if (!asset) {
        if (asset_) {
            vkDeviceWaitIdle(device_);
            destroy_viewport_resources();
            destroy_asset_resources();
            asset_.reset();
            status_ = "No resident VBT asset";
        }
        return;
    }
    if (asset_ && asset_->path == asset->path) return;

    try {
        load_asset(*asset);
    }
    catch (const std::exception& error) {
        vkDeviceWaitIdle(device_);
        destroy_viewport_resources();
        destroy_asset_resources();
        asset_.reset();
        status_ = error.what();
    }
}

void VolumeRenderer::sync_secondary_field(const std::optional<backend::VbtAssetInfo>& asset)
{
    if (!asset) {
        if (secondary_field_) {
            vkDeviceWaitIdle(device_);
            destroy_viewport_resources();
            destroy_secondary_resources();
            secondary_field_.reset();
            render_dirty_ = true;
        }
        return;
    }
    if (secondary_field_ && secondary_field_->path == asset->path) return;
    try {
        load_secondary_field(*asset);
    }
    catch (const std::exception& error) {
        vkDeviceWaitIdle(device_);
        destroy_viewport_resources();
        destroy_secondary_resources();
        secondary_field_.reset();
        status_ = error.what();
    }
}

void VolumeRenderer::sync_temperature_field(const std::optional<backend::VbtAssetInfo>& asset)
{
    if (!asset) {
        if (temperature_field_) {
            vkDeviceWaitIdle(device_);
            destroy_viewport_resources();
            destroy_temperature_resources();
            temperature_field_.reset();
            render_dirty_ = true;
        }
        return;
    }
    if (temperature_field_ && temperature_field_->path == asset->path) return;
    try {
        load_temperature_field(*asset);
    }
    catch (const std::exception& error) {
        vkDeviceWaitIdle(device_);
        destroy_viewport_resources();
        destroy_temperature_resources();
        temperature_field_.reset();
        status_ = error.what();
    }
}

void VolumeRenderer::ensure_viewport(std::uint32_t width, std::uint32_t height)
{
    if (!asset_ || width == 0 || height == 0) return;
    if (render_width_ == width && render_height_ == height && output_image_.handle != VK_NULL_HANDLE) return;
    try {
        vkDeviceWaitIdle(device_);
        destroy_viewport_resources();
        create_viewport_resources(width, height);
        update_descriptor_sets();
        render_dirty_ = true;
    }
    catch (const std::exception& error) {
        destroy_viewport_resources();
        status_ = error.what();
    }
}

void VolumeRenderer::set_render_state(std::uint32_t frame,
                                      const backend::MaterialState& material,
                                      const backend::CameraState& camera)
{
    if (frame_ != frame) {
        cache_dirty_ = true;
        render_dirty_ = true;
    }
    if (!material_equal(material_, material)) render_dirty_ = true;
    if (std::memcmp(&camera_, &camera, sizeof(camera)) != 0) {
        camera_ = camera;
        rays_dirty_ = true;
        render_dirty_ = true;
    }
    frame_ = frame;
    material_ = material;
}

void VolumeRenderer::invalidate_render() noexcept { render_dirty_ = true; }

bool VolumeRenderer::export_png(const std::filesystem::path& path)
{
    if (!ready()) return false;
    vkDeviceWaitIdle(device_);
    Buffer staging;
    const VkDeviceSize byte_count = static_cast<VkDeviceSize>(render_width_) * render_height_ * 8u;
    try {
        create_buffer(byte_count,
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      staging);
        VkCommandBufferAllocateInfo allocation_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation_info.commandPool = command_pool_;
        allocation_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation_info.commandBufferCount = 1;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device_, &allocation_info, &command_buffer) != VK_SUCCESS) {
            throw std::runtime_error("Unable to allocate export command buffer");
        }
        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(command_buffer, &begin_info);
        VkImageMemoryBarrier to_transfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_transfer.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        to_transfer.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.image = output_image_.handle;
        to_transfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_transfer.subresourceRange.levelCount = 1;
        to_transfer.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &to_transfer);
        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = {render_width_, render_height_, 1};
        vkCmdCopyImageToBuffer(command_buffer,
                               output_image_.handle,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging.handle,
                               1,
                               &copy);
        VkImageMemoryBarrier to_general = to_transfer;
        to_general.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        to_general.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        to_general.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &to_general);
        vkEndCommandBuffer(command_buffer);
        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        if (vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS ||
            vkQueueWaitIdle(queue_) != VK_SUCCESS) {
            vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
            throw std::runtime_error("Unable to read back the viewport image");
        }
        vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);

        void* mapped = nullptr;
        if (vkMapMemory(device_, staging.memory, 0, byte_count, 0, &mapped) != VK_SUCCESS) {
            throw std::runtime_error("Unable to map the exported viewport image");
        }
        const auto* half_pixels = static_cast<const std::uint16_t*>(mapped);
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(render_width_) * render_height_ * 4u);
        const std::size_t pixel_count = static_cast<std::size_t>(render_width_) * render_height_;
        for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
            const std::array<std::size_t, 4> source_channels{2, 1, 0, 3};
            for (std::size_t channel = 0; channel < 4; ++channel) {
                const float value = std::clamp(
                    half_to_float(half_pixels[pixel * 4u + source_channels[channel]]), 0.0f, 1.0f);
                pixels[pixel * 4u + channel] = static_cast<std::uint8_t>(std::lround(value * 255.0f));
            }
        }
        vkUnmapMemory(device_, staging.memory);
        destroy_buffer(staging);
#ifdef _WIN32
        if (!write_png(path, pixels, render_width_, render_height_)) {
            throw std::runtime_error("Windows Imaging Component could not encode the PNG");
        }
#else
        throw std::runtime_error("PNG export is not implemented on this platform");
#endif
        status_ = "Exported " + path.filename().string();
        return true;
    }
    catch (const std::exception& error) {
        destroy_buffer(staging);
        status_ = error.what();
        return false;
    }
}

void VolumeRenderer::collect_gpu_timings(std::uint32_t query_slot) noexcept
{
    if (timing_query_pool_ == VK_NULL_HANDLE) return;
    const std::uint32_t slot = query_slot % timing_slot_count_;
    if (!timing_pending_[slot]) return;

    std::array<std::uint64_t, timing_query_count_> timestamps{};
    const std::uint32_t first_query = slot * timing_query_count_;
    const VkResult result = vkGetQueryPoolResults(device_,
                                                   timing_query_pool_,
                                                   first_query,
                                                   timing_query_count_,
                                                   sizeof(timestamps),
                                                   timestamps.data(),
                                                   sizeof(std::uint64_t),
                                                   VK_QUERY_RESULT_64_BIT);
    timing_pending_[slot] = false;
    if (result != VK_SUCCESS) return;

    gpu_timings_.supported = true;
    gpu_timings_.valid = true;
    gpu_timings_.rays_regenerated = timing_rays_regenerated_[slot];
    gpu_timings_.frame = timing_frames_[slot];
    gpu_timings_.field_count = timing_field_counts_[slot];
    gpu_timings_.ray_milliseconds =
        timestamp_delta_milliseconds(timestamps[0], timestamps[1]);
    gpu_timings_.sample_milliseconds =
        timestamp_delta_milliseconds(timestamps[1], timestamps[2]);
    gpu_timings_.colorize_milliseconds =
        timestamp_delta_milliseconds(timestamps[2], timestamps[3]);
    gpu_timings_.total_milliseconds =
        timestamp_delta_milliseconds(timestamps[0], timestamps[3]);
}

std::optional<VolumeRenderer::PngExportResult>
VolumeRenderer::collect_png_export(std::uint32_t query_slot)
{
    const std::uint32_t slot = query_slot % timing_slot_count_;
    if (!export_pending_[slot]) return std::nullopt;

    PngExportResult result{};
    result.path = export_paths_[slot];
    result.frame = export_frames_[slot];
    result.width = export_widths_[slot];
    result.height = export_heights_[slot];
    export_pending_[slot] = false;

    Buffer& staging = export_staging_buffers_[slot];
    void* mapped = nullptr;
    if (vkMapMemory(device_, staging.memory, 0, staging.size, 0, &mapped) != VK_SUCCESS) {
        result.error = "Unable to map completed sequence frame";
        status_ = result.error;
        return result;
    }

    const auto* half_pixels = static_cast<const std::uint16_t*>(mapped);
    const std::size_t pixel_count = static_cast<std::size_t>(result.width) * result.height;
    std::vector<std::uint8_t> pixels(pixel_count * 4u);
    for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
        const std::array<std::size_t, 4> source_channels{2, 1, 0, 3};
        for (std::size_t channel = 0; channel < 4; ++channel) {
            const float value = std::clamp(
                half_to_float(half_pixels[pixel * 4u + source_channels[channel]]), 0.0f, 1.0f);
            pixels[pixel * 4u + channel] = static_cast<std::uint8_t>(std::lround(value * 255.0f));
        }
    }
    vkUnmapMemory(device_, staging.memory);

#ifdef _WIN32
    result.success = write_png(result.path, pixels, result.width, result.height);
    if (!result.success) result.error = "Windows Imaging Component could not encode the sequence frame";
#else
    result.error = "PNG export is not implemented on this platform";
#endif
    status_ = result.success ? "Exported " + result.path.filename().string() : result.error;
    return result;
}

bool VolumeRenderer::record(VkCommandBuffer command_buffer,
                            std::uint32_t query_slot,
                            const std::optional<std::filesystem::path>& export_path)
{
    if (!ready() || !render_dirty_) return false;

    const bool timing_enabled = timing_query_pool_ != VK_NULL_HANDLE;
    const std::uint32_t timing_slot = query_slot % timing_slot_count_;
    const std::uint32_t first_timing_query = timing_slot * timing_query_count_;
    const bool regenerating_rays = rays_dirty_;
    const std::uint32_t field_count =
        1u + (secondary_field_ ? 1u : 0u) + (temperature_field_ ? 1u : 0u);
    if (timing_enabled) {
        vkCmdResetQueryPool(command_buffer, timing_query_pool_, first_timing_query, timing_query_count_);
        vkCmdWriteTimestamp(command_buffer,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            timing_query_pool_,
                            first_timing_query);
    }
    if (export_path) {
        const VkDeviceSize byte_count = static_cast<VkDeviceSize>(render_width_) * render_height_ * 8u;
        ensure_export_staging(timing_slot, byte_count);
        export_paths_[timing_slot] = *export_path;
        export_frames_[timing_slot] = frame_;
        export_widths_[timing_slot] = render_width_;
        export_heights_[timing_slot] = render_height_;
    }

    std::array<VkBufferMemoryBarrier, 3> result_write_barriers{};
    const std::uint32_t result_write_count =
        1u + (secondary_field_ ? 1u : 0u) + (temperature_field_ ? 1u : 0u);
    for (std::uint32_t index = 0; index < result_write_count; ++index) {
        auto& barrier = result_write_barriers[index];
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        const bool is_secondary = index == 1u && secondary_field_;
        barrier.buffer = index == 0 ? result_buffer_.handle
                                    : (is_secondary ? secondary_result_buffer_.handle
                                                    : temperature_result_buffer_.handle);
        barrier.offset = 0;
        barrier.size = index == 0 ? result_buffer_.size
                                  : (is_secondary ? secondary_result_buffer_.size
                                                  : temperature_result_buffer_.size);
    }

    VkImageMemoryBarrier image_write_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    image_write_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_write_barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    image_write_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_write_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_write_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_write_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_write_barrier.image = output_image_.handle;
    image_write_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_write_barrier.subresourceRange.levelCount = 1;
    image_write_barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0,
                         nullptr,
                         result_write_count,
                         result_write_barriers.data(),
                         1,
                         &image_write_barrier);

    if (rays_dirty_) {
        RayPushConstants ray{};
        ray.image_width = render_width_;
        ray.image_height = render_height_;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            std::int32_t minimum_value = asset_->bbox_min[axis];
            std::int32_t maximum_value = asset_->bbox_max[axis];
            if (secondary_field_) {
                minimum_value = std::min(minimum_value, secondary_field_->bbox_min[axis]);
                maximum_value = std::max(maximum_value, secondary_field_->bbox_max[axis]);
            }
            if (temperature_field_) {
                minimum_value = std::min(minimum_value, temperature_field_->bbox_min[axis]);
                maximum_value = std::max(maximum_value, temperature_field_->bbox_max[axis]);
            }
            const float minimum = static_cast<float>(minimum_value);
            const float maximum = static_cast<float>(maximum_value);
            if (axis == 0) { ray.bbox_min_x = minimum; ray.bbox_max_x = maximum; }
            if (axis == 1) { ray.bbox_min_y = minimum; ray.bbox_max_y = maximum; }
            if (axis == 2) { ray.bbox_min_z = minimum; ray.bbox_max_z = maximum; }
        }
        ray.yaw = camera_.yaw;
        ray.pitch = camera_.pitch;
        ray.distance = camera_.distance;
        ray.field_of_view = camera_.field_of_view;
        ray.target_offset_x = camera_.target_offset[0];
        ray.target_offset_y = camera_.target_offset[1];
        ray.target_offset_z = camera_.target_offset[2];
        ray.up_axis = camera_.up_axis;
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ray_pipeline_);
        vkCmdBindDescriptorSets(command_buffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                ray_pipeline_layout_,
                                0,
                                1,
                                &ray_set_,
                                0,
                                nullptr);
        vkCmdPushConstants(command_buffer,
                           ray_pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(ray),
                           &ray);
        vkCmdDispatch(command_buffer, (render_width_ + 7u) / 8u, (render_height_ + 7u) / 8u, 1);

        VkBufferMemoryBarrier ray_read_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        ray_read_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        ray_read_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        ray_read_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ray_read_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ray_read_barrier.buffer = ray_buffer_.handle;
        ray_read_barrier.offset = 0;
        ray_read_barrier.size = ray_buffer_.size;
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             1,
                             &ray_read_barrier,
                             0,
                             nullptr);
    }
    if (timing_enabled) {
        vkCmdWriteTimestamp(command_buffer,
                            regenerating_rays ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                              : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            timing_query_pool_,
                            first_timing_query + 1u);
    }

    if (cache_dirty_) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, cache_pipeline_);
        const auto dispatch_cache = [&](const backend::VbtAssetInfo& field,
                                        VkDescriptorSet descriptor_set) {
            CachePushConstants cache{};
            cache.frames = field.frames;
            cache.leaf_count = field.leaf_count;
            cache.frame_index = static_cast<std::int32_t>(std::min(frame_, field.frames - 1u));
            vkCmdBindDescriptorSets(command_buffer,
                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                    cache_pipeline_layout_,
                                    0,
                                    1,
                                    &descriptor_set,
                                    0,
                                    nullptr);
            vkCmdPushConstants(command_buffer,
                               cache_pipeline_layout_,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0,
                               sizeof(cache),
                               &cache);
            const std::uint64_t group_count = field.leaf_count;
            const std::uint32_t groups_x = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(group_count, 65535u));
            const std::uint32_t groups_y = static_cast<std::uint32_t>(
                (group_count + groups_x - 1u) / groups_x);
            vkCmdDispatch(command_buffer, groups_x, groups_y, 1u);
        };
        dispatch_cache(*asset_, cache_set_);
        if (secondary_field_) dispatch_cache(*secondary_field_, secondary_cache_set_);
        if (temperature_field_) dispatch_cache(*temperature_field_, temperature_cache_set_);

        std::array<VkBufferMemoryBarrier, 3> cache_barriers{};
        std::array<const Buffer*, 3> cache_buffers{};
        std::uint32_t cache_barrier_count = 0u;
        cache_buffers[cache_barrier_count++] = &cache_buffer_;
        if (secondary_field_) cache_buffers[cache_barrier_count++] = &secondary_cache_buffer_;
        if (temperature_field_) cache_buffers[cache_barrier_count++] = &temperature_cache_buffer_;
        for (std::uint32_t index = 0; index < cache_barrier_count; ++index) {
            auto& barrier = cache_barriers[index];
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = cache_buffers[index]->handle;
            barrier.offset = 0;
            barrier.size = cache_buffers[index]->size;
        }
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             cache_barrier_count,
                             cache_barriers.data(),
                             0,
                             nullptr);
        cache_dirty_ = false;
    }

    const std::uint32_t step_count = std::clamp(material_.sample_steps, 32u, 1024u);
    SamplePushConstants sample{};
    sample.dim_x = asset_->width;
    sample.dim_y = asset_->height;
    sample.dim_z = asset_->depth;
    sample.frames = asset_->frames;
    sample.leaf_size = asset_->leaf_size;
    sample.leaf_count_x = (asset_->width + asset_->leaf_size - 1u) / asset_->leaf_size;
    sample.leaf_count_y = (asset_->height + asset_->leaf_size - 1u) / asset_->leaf_size;
    sample.image_width = render_width_;
    sample.image_height = render_height_;
    sample.ray_count = render_width_ * render_height_;
    sample.step_count = step_count;
    sample.frame_index = static_cast<std::int32_t>(std::min(frame_, asset_->frames - 1u));
    sample.bbox_min_x = static_cast<float>(asset_->bbox_min[0]);
    sample.bbox_min_y = static_cast<float>(asset_->bbox_min[1]);
    sample.bbox_min_z = static_cast<float>(asset_->bbox_min[2]);
    sample.render_mode = asset_->role == backend::FieldRole::LevelSet ? 1u : 0u;
    sample.voxel_size = std::max(asset_->voxel_size, 1e-6f);
    sample.background_value = std::max(asset_->background_value, sample.voxel_size);
    sample.surface_iso = material_.surface_iso;
    sample.surface_epsilon_voxels = material_.surface_epsilon_voxels;
    sample.surface_normal_step = material_.surface_normal_step;
    sample.surface_model = sample.render_mode != 0u ? material_.surface_model : material_.volume_model;
    const float extent = static_cast<float>(std::max({asset_->width, asset_->height, asset_->depth}));
    const std::size_t up_axis = camera_.up_axis == 2u ? 2u : 1u;
    sample.floor_height = static_cast<float>(asset_->bbox_min[up_axis]) - extent * material_.floor_offset;
    sample.shadow_steps = material_.shadow_steps;
    sample.padding = camera_.up_axis;

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, sample_pipeline_);
    vkCmdBindDescriptorSets(command_buffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            sample_pipeline_layout_,
                            0,
                            1,
                            &sample_set_,
                            0,
                            nullptr);
    vkCmdPushConstants(command_buffer,
                       sample_pipeline_layout_,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(sample),
                       &sample);
    vkCmdDispatch(command_buffer, (render_width_ + 7u) / 8u, (render_height_ + 7u) / 8u, 1);

    if (secondary_field_) {
        SamplePushConstants secondary = sample;
        secondary.dim_x = secondary_field_->width;
        secondary.dim_y = secondary_field_->height;
        secondary.dim_z = secondary_field_->depth;
        secondary.frames = secondary_field_->frames;
        secondary.leaf_size = secondary_field_->leaf_size;
        secondary.leaf_count_x =
            (secondary_field_->width + secondary_field_->leaf_size - 1u) / secondary_field_->leaf_size;
        secondary.leaf_count_y =
            (secondary_field_->height + secondary_field_->leaf_size - 1u) / secondary_field_->leaf_size;
        secondary.bbox_min_x = static_cast<float>(secondary_field_->bbox_min[0]);
        secondary.bbox_min_y = static_cast<float>(secondary_field_->bbox_min[1]);
        secondary.bbox_min_z = static_cast<float>(secondary_field_->bbox_min[2]);
        secondary.render_mode = 0u;
        secondary.surface_model = material_.volume_model;
        secondary.background_value = 0.0f;
        vkCmdBindDescriptorSets(command_buffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                sample_pipeline_layout_,
                                0,
                                1,
                                &secondary_sample_set_,
                                0,
                                nullptr);
        vkCmdPushConstants(command_buffer,
                           sample_pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(secondary),
                           &secondary);
        vkCmdDispatch(command_buffer, (render_width_ + 7u) / 8u, (render_height_ + 7u) / 8u, 1);
    }

    if (temperature_field_) {
        SamplePushConstants temperature = sample;
        temperature.dim_x = temperature_field_->width;
        temperature.dim_y = temperature_field_->height;
        temperature.dim_z = temperature_field_->depth;
        temperature.frames = temperature_field_->frames;
        temperature.leaf_size = temperature_field_->leaf_size;
        temperature.leaf_count_x =
            (temperature_field_->width + temperature_field_->leaf_size - 1u) / temperature_field_->leaf_size;
        temperature.leaf_count_y =
            (temperature_field_->height + temperature_field_->leaf_size - 1u) / temperature_field_->leaf_size;
        temperature.bbox_min_x = static_cast<float>(temperature_field_->bbox_min[0]);
        temperature.bbox_min_y = static_cast<float>(temperature_field_->bbox_min[1]);
        temperature.bbox_min_z = static_cast<float>(temperature_field_->bbox_min[2]);
        temperature.render_mode = 0u;
        temperature.surface_model = material_.volume_model;
        temperature.background_value = 0.0f;
        vkCmdBindDescriptorSets(command_buffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                sample_pipeline_layout_,
                                0,
                                1,
                                &temperature_sample_set_,
                                0,
                                nullptr);
        vkCmdPushConstants(command_buffer,
                           sample_pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(temperature),
                           &temperature);
        vkCmdDispatch(command_buffer, (render_width_ + 7u) / 8u, (render_height_ + 7u) / 8u, 1);
    }

    std::array<VkBufferMemoryBarrier, 3> result_read_barriers{};
    const std::uint32_t result_barrier_count =
        1u + (secondary_field_ ? 1u : 0u) + (temperature_field_ ? 1u : 0u);
    for (std::uint32_t index = 0; index < result_barrier_count; ++index) {
        auto& barrier = result_read_barriers[index];
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        const bool is_secondary = index == 1u && secondary_field_;
        barrier.buffer = index == 0 ? result_buffer_.handle
                                    : (is_secondary ? secondary_result_buffer_.handle
                                                    : temperature_result_buffer_.handle);
        barrier.offset = 0;
        barrier.size = index == 0 ? result_buffer_.size
                                  : (is_secondary ? secondary_result_buffer_.size
                                                  : temperature_result_buffer_.size);
    }
    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0,
                         nullptr,
                         result_barrier_count,
                         result_read_barriers.data(),
                         0,
                         nullptr);
    if (timing_enabled) {
        vkCmdWriteTimestamp(command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timing_query_pool_,
                            first_timing_query + 2u);
    }

    ColorPushConstants color{};
    color.image_width = render_width_;
    color.image_height = render_height_;
    color.density_scale = material_.density_scale;
    color.density_threshold = material_.density_threshold;
    color.density_gamma = material_.density_gamma;
    color.exposure = material_.exposure;
    std::copy(material_.smoke_color.begin(), material_.smoke_color.end(), color.smoke_color.begin());
    color.smoke_color[3] = 1.0f;
    std::copy(material_.background_color.begin(), material_.background_color.end(), color.background_color.begin());
    color.background_color[3] = 1.0f;
    color.step_count = step_count;
    color.flame_strength = material_.flame_strength;
    color.flame_threshold = material_.flame_threshold;
    color.has_flames = secondary_field_ ? 1u : 0u;
    color.temperature_min = material_.temperature_min;
    color.temperature_max = material_.temperature_max;
    color.has_temperature = temperature_field_ ? 1u : 0u;
    color.padding1 = camera_.up_axis;
    color.render_mode = asset_->role == backend::FieldRole::LevelSet ? 1u : 0u;
    color.physical_mode = color.render_mode != 0u ? material_.surface_model : material_.volume_model;
    if (color.render_mode == 0u && color.physical_mode != 0u) {
        color.sample_distance = material_.anisotropy;
        color.surface_roughness = material_.fire_scattering;
        color.surface_metallic = material_.fire_blackbody_mix;
        color.surface_opacity = material_.fire_glow;
        std::copy(material_.fire_tint.begin(), material_.fire_tint.end(), color.surface_color.begin());
        color.surface_color[3] = 1.0f;
    }
    if (color.render_mode != 0u) {
        color.physical_mode = material_.surface_model;
        color.density_scale = material_.water_ior;
        color.density_threshold = material_.absorption_density;
        color.density_gamma = sample.floor_height;
        color.sample_distance = material_.environment_strength;
        color.flame_strength = material_.shadow_strength;
        color.flame_threshold = material_.reflection_strength;
        std::copy(material_.absorption_color.begin(), material_.absorption_color.end(), color.smoke_color.begin());
        color.smoke_color[3] = 1.0f;
    }
    if (color.render_mode != 0u || color.physical_mode == 0u) {
        color.surface_roughness = material_.surface_roughness;
        color.surface_metallic = material_.surface_metallic;
        color.surface_opacity = material_.surface_opacity;
        std::copy(material_.surface_color.begin(), material_.surface_color.end(), color.surface_color.begin());
        color.surface_color[3] = 1.0f;
    }

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, color_pipeline_);
    vkCmdBindDescriptorSets(command_buffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            color_pipeline_layout_,
                            0,
                            1,
                            &color_set_,
                            0,
                            nullptr);
    vkCmdPushConstants(command_buffer,
                       color_pipeline_layout_,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(color),
                       &color);
    vkCmdDispatch(command_buffer, (render_width_ + 7u) / 8u, (render_height_ + 7u) / 8u, 1);
    if (timing_enabled) {
        vkCmdWriteTimestamp(command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timing_query_pool_,
                            first_timing_query + 3u);
        timing_pending_[timing_slot] = true;
        timing_rays_regenerated_[timing_slot] = regenerating_rays;
        timing_frames_[timing_slot] = frame_;
        timing_field_counts_[timing_slot] = field_count;
    }

    VkImageMemoryBarrier image_read_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    image_read_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    image_read_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_read_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_read_barrier.image = output_image_.handle;
    image_read_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_read_barrier.subresourceRange.levelCount = 1;
    image_read_barrier.subresourceRange.layerCount = 1;
    if (export_path) {
        image_read_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        image_read_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        image_read_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &image_read_barrier);
        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = {render_width_, render_height_, 1};
        vkCmdCopyImageToBuffer(command_buffer,
                               output_image_.handle,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               export_staging_buffers_[timing_slot].handle,
                               1,
                               &copy);
        image_read_barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        image_read_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        image_read_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        image_read_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &image_read_barrier);
        export_pending_[timing_slot] = true;
    }
    else {
        image_read_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        image_read_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        image_read_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &image_read_barrier);
    }
    render_dirty_ = false;
    rays_dirty_ = false;
    return true;
}

void VolumeRenderer::shutdown()
{
    if (!initialized_) return;
    vkDeviceWaitIdle(device_);
    destroy_viewport_resources();
    destroy_asset_resources();
    destroy_pipeline_resources();
    initialized_ = false;
}

bool VolumeRenderer::ready() const noexcept
{
    return asset_.has_value() && offset_buffer_.handle != VK_NULL_HANDLE &&
           payload_buffer_.handle != VK_NULL_HANDLE && cache_buffer_.handle != VK_NULL_HANDLE &&
           active_leaf_mapping_buffer_.handle != VK_NULL_HANDLE &&
           output_image_.handle != VK_NULL_HANDLE;
}

ImTextureID VolumeRenderer::texture_id() const noexcept
{
    return static_cast<ImTextureID>(reinterpret_cast<std::uintptr_t>(texture_descriptor_));
}
const std::string& VolumeRenderer::status() const noexcept { return status_; }
double VolumeRenderer::upload_milliseconds() const noexcept { return upload_milliseconds_; }
const VolumeRenderer::GpuTimings& VolumeRenderer::gpu_timings() const noexcept { return gpu_timings_; }
VolumeRenderer::CacheStats VolumeRenderer::cache_stats() const noexcept
{
    CacheStats stats{};
    const auto add_field = [&](const std::optional<backend::VbtAssetInfo>& field,
                               std::uint32_t active_leaves,
                               const Buffer& cache,
                               const Buffer& mapping) {
        if (!field) return;
        stats.active_leaves += active_leaves;
        stats.total_leaves += field->leaf_count;
        stats.cache_bytes += cache.size;
        stats.fixed_cache_bytes += static_cast<std::uint64_t>(field->leaf_count) *
                                   cached_controls_per_leaf * sizeof(float);
        stats.mapping_bytes += mapping.size;
    };
    add_field(asset_, active_leaf_count_, cache_buffer_, active_leaf_mapping_buffer_);
    add_field(secondary_field_,
              secondary_active_leaf_count_,
              secondary_cache_buffer_,
              secondary_active_leaf_mapping_buffer_);
    add_field(temperature_field_,
              temperature_active_leaf_count_,
              temperature_cache_buffer_,
              temperature_active_leaf_mapping_buffer_);
    return stats;
}
std::uint32_t VolumeRenderer::render_width() const noexcept { return render_width_; }
std::uint32_t VolumeRenderer::render_height() const noexcept { return render_height_; }

void VolumeRenderer::create_pipeline_resources()
{
    VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    command_pool_info.queueFamilyIndex = queue_family_;
    if (vkCreateCommandPool(device_, &command_pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create volume upload command pool");
    }
    create_buffer(sizeof(std::uint32_t),
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  active_index_counter_buffer_);
    create_buffer(sizeof(std::uint32_t),
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  active_index_readback_buffer_);

    VkPhysicalDeviceProperties physical_properties{};
    vkGetPhysicalDeviceProperties(physical_device_, &physical_properties);
    std::uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_properties(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_properties.data());
    if (physical_properties.limits.timestampComputeAndGraphics == VK_TRUE &&
        queue_family_ < queue_properties.size() &&
        queue_properties[queue_family_].timestampValidBits > 0) {
        VkQueryPoolCreateInfo query_pool_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_pool_info.queryCount = timing_query_count_ * timing_slot_count_;
        if (vkCreateQueryPool(device_, &query_pool_info, nullptr, &timing_query_pool_) == VK_SUCCESS) {
            timestamp_period_nanoseconds_ = physical_properties.limits.timestampPeriod;
            timestamp_valid_bits_ = queue_properties[queue_family_].timestampValidBits;
            gpu_timings_.supported = true;
        }
    }

    VkDescriptorSetLayoutBinding ray_binding{
        0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo ray_layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ray_layout_info.bindingCount = 1;
    ray_layout_info.pBindings = &ray_binding;
    if (vkCreateDescriptorSetLayout(device_, &ray_layout_info, nullptr, &ray_set_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create ray descriptor layout");
    }

    std::array<VkDescriptorSetLayoutBinding, 6> sample_bindings{};
    for (std::uint32_t index = 0; index < sample_bindings.size(); ++index) {
        sample_bindings[index].binding = index;
        sample_bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sample_bindings[index].descriptorCount = 1;
        sample_bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo sample_layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sample_layout_info.bindingCount = static_cast<std::uint32_t>(sample_bindings.size());
    sample_layout_info.pBindings = sample_bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &sample_layout_info, nullptr, &sample_set_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create VBT sample descriptor layout");
    }

    std::array<VkDescriptorSetLayoutBinding, 4> cache_bindings{};
    for (std::uint32_t index = 0; index < cache_bindings.size(); ++index) {
        cache_bindings[index].binding = index;
        cache_bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        cache_bindings[index].descriptorCount = 1;
        cache_bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo cache_layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    cache_layout_info.bindingCount = static_cast<std::uint32_t>(cache_bindings.size());
    cache_layout_info.pBindings = cache_bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &cache_layout_info, nullptr, &cache_set_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create VBT frame-cache descriptor layout");
    }

    std::array<VkDescriptorSetLayoutBinding, 4> active_index_bindings{};
    for (std::uint32_t index = 0; index < active_index_bindings.size(); ++index) {
        active_index_bindings[index].binding = index;
        active_index_bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        active_index_bindings[index].descriptorCount = 1;
        active_index_bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo active_index_layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    active_index_layout_info.bindingCount = static_cast<std::uint32_t>(active_index_bindings.size());
    active_index_layout_info.pBindings = active_index_bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &active_index_layout_info, nullptr, &active_index_set_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create VBT active-leaf index descriptor layout");
    }

    std::array<VkDescriptorSetLayoutBinding, 5> color_bindings{};
    color_bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    color_bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    color_bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    color_bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    color_bindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo color_layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    color_layout_info.bindingCount = static_cast<std::uint32_t>(color_bindings.size());
    color_layout_info.pBindings = color_bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &color_layout_info, nullptr, &color_set_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create VBT color descriptor layout");
    }

    const std::array<VkDescriptorPoolSize, 2> pool_sizes{{
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 48},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
    }};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 11;
    pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create volume descriptor pool");
    }

    const std::array<VkDescriptorSetLayout, 11> layouts{
        ray_set_layout_,
        sample_set_layout_, sample_set_layout_, sample_set_layout_,
        cache_set_layout_, cache_set_layout_, cache_set_layout_,
        active_index_set_layout_, active_index_set_layout_, active_index_set_layout_,
        color_set_layout_};
    std::array<VkDescriptorSet, 11> sets{};
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = descriptor_pool_;
    set_info.descriptorSetCount = static_cast<std::uint32_t>(layouts.size());
    set_info.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device_, &set_info, sets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Unable to allocate volume descriptor sets");
    }
    ray_set_ = sets[0];
    sample_set_ = sets[1];
    secondary_sample_set_ = sets[2];
    temperature_sample_set_ = sets[3];
    cache_set_ = sets[4];
    secondary_cache_set_ = sets[5];
    temperature_cache_set_ = sets[6];
    active_index_set_ = sets[7];
    secondary_active_index_set_ = sets[8];
    temperature_active_index_set_ = sets[9];
    color_set_ = sets[10];

    VkPushConstantRange ray_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RayPushConstants)};
    VkPipelineLayoutCreateInfo ray_pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    ray_pipeline_layout_info.setLayoutCount = 1;
    ray_pipeline_layout_info.pSetLayouts = &ray_set_layout_;
    ray_pipeline_layout_info.pushConstantRangeCount = 1;
    ray_pipeline_layout_info.pPushConstantRanges = &ray_range;
    if (vkCreatePipelineLayout(device_, &ray_pipeline_layout_info, nullptr, &ray_pipeline_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create ray pipeline layout");
    }

    VkPushConstantRange sample_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SamplePushConstants)};
    VkPipelineLayoutCreateInfo sample_pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    sample_pipeline_layout_info.setLayoutCount = 1;
    sample_pipeline_layout_info.pSetLayouts = &sample_set_layout_;
    sample_pipeline_layout_info.pushConstantRangeCount = 1;
    sample_pipeline_layout_info.pPushConstantRanges = &sample_range;
    if (vkCreatePipelineLayout(device_, &sample_pipeline_layout_info, nullptr, &sample_pipeline_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create VBT sample pipeline layout");
    }

    VkPushConstantRange cache_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CachePushConstants)};
    VkPipelineLayoutCreateInfo cache_pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    cache_pipeline_layout_info.setLayoutCount = 1;
    cache_pipeline_layout_info.pSetLayouts = &cache_set_layout_;
    cache_pipeline_layout_info.pushConstantRangeCount = 1;
    cache_pipeline_layout_info.pPushConstantRanges = &cache_range;
    if (vkCreatePipelineLayout(device_, &cache_pipeline_layout_info, nullptr, &cache_pipeline_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create VBT frame-cache pipeline layout");
    }

    VkPushConstantRange active_index_range{
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ActiveIndexPushConstants)};
    VkPipelineLayoutCreateInfo active_index_pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    active_index_pipeline_layout_info.setLayoutCount = 1;
    active_index_pipeline_layout_info.pSetLayouts = &active_index_set_layout_;
    active_index_pipeline_layout_info.pushConstantRangeCount = 1;
    active_index_pipeline_layout_info.pPushConstantRanges = &active_index_range;
    if (vkCreatePipelineLayout(device_,
                               &active_index_pipeline_layout_info,
                               nullptr,
                               &active_index_pipeline_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create VBT active-leaf index pipeline layout");
    }

    VkPushConstantRange color_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ColorPushConstants)};
    VkPipelineLayoutCreateInfo color_pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    color_pipeline_layout_info.setLayoutCount = 1;
    color_pipeline_layout_info.pSetLayouts = &color_set_layout_;
    color_pipeline_layout_info.pushConstantRangeCount = 1;
    color_pipeline_layout_info.pPushConstantRanges = &color_range;
    if (vkCreatePipelineLayout(device_, &color_pipeline_layout_info, nullptr, &color_pipeline_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create VBT color pipeline layout");
    }

    const VkShaderModule ray_shader = create_shader_module(device_, VBTSTUDIO_RAY_SHADER_SPV);
    const VkShaderModule sample_shader = create_shader_module(device_, VBTSTUDIO_VBT_SHADER_SPV);
    const VkShaderModule cache_shader = create_shader_module(device_, VBTSTUDIO_CACHE_SHADER_SPV);
    const VkShaderModule active_index_shader =
        create_shader_module(device_, VBTSTUDIO_ACTIVE_INDEX_SHADER_SPV);
    const VkShaderModule color_shader = create_shader_module(device_, VBTSTUDIO_COLOR_SHADER_SPV);
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.pName = "main";
    pipeline_info.stage.module = ray_shader;
    pipeline_info.layout = ray_pipeline_layout_;
    const VkResult ray_result =
        vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &ray_pipeline_);
    pipeline_info.stage.module = sample_shader;
    pipeline_info.layout = sample_pipeline_layout_;
    const VkResult sample_result =
        vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &sample_pipeline_);
    pipeline_info.stage.module = cache_shader;
    pipeline_info.layout = cache_pipeline_layout_;
    const VkResult cache_result =
        vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &cache_pipeline_);
    pipeline_info.stage.module = active_index_shader;
    pipeline_info.layout = active_index_pipeline_layout_;
    const VkResult active_index_result =
        vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &active_index_pipeline_);
    pipeline_info.stage.module = color_shader;
    pipeline_info.layout = color_pipeline_layout_;
    const VkResult color_result =
        vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &color_pipeline_);
    vkDestroyShaderModule(device_, color_shader, nullptr);
    vkDestroyShaderModule(device_, active_index_shader, nullptr);
    vkDestroyShaderModule(device_, cache_shader, nullptr);
    vkDestroyShaderModule(device_, sample_shader, nullptr);
    vkDestroyShaderModule(device_, ray_shader, nullptr);
    if (ray_result != VK_SUCCESS || sample_result != VK_SUCCESS ||
        cache_result != VK_SUCCESS || active_index_result != VK_SUCCESS ||
        color_result != VK_SUCCESS) {
        throw std::runtime_error("Unable to create VBT compute pipelines");
    }
}

void VolumeRenderer::destroy_pipeline_resources()
{
    for (auto& buffer : export_staging_buffers_) destroy_buffer(buffer);
    if (timing_query_pool_ != VK_NULL_HANDLE) vkDestroyQueryPool(device_, timing_query_pool_, nullptr);
    if (color_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, color_pipeline_, nullptr);
    if (active_index_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, active_index_pipeline_, nullptr);
    if (cache_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, cache_pipeline_, nullptr);
    if (sample_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, sample_pipeline_, nullptr);
    if (ray_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, ray_pipeline_, nullptr);
    if (color_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, color_pipeline_layout_, nullptr);
    if (active_index_pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, active_index_pipeline_layout_, nullptr);
    }
    if (cache_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, cache_pipeline_layout_, nullptr);
    if (sample_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, sample_pipeline_layout_, nullptr);
    if (ray_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, ray_pipeline_layout_, nullptr);
    if (descriptor_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    if (color_set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, color_set_layout_, nullptr);
    if (active_index_set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, active_index_set_layout_, nullptr);
    }
    if (cache_set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, cache_set_layout_, nullptr);
    if (sample_set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, sample_set_layout_, nullptr);
    if (ray_set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, ray_set_layout_, nullptr);
    destroy_buffer(active_index_readback_buffer_);
    destroy_buffer(active_index_counter_buffer_);
    if (command_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, command_pool_, nullptr);
}

double VolumeRenderer::timestamp_delta_milliseconds(std::uint64_t start,
                                                    std::uint64_t end) const noexcept
{
    std::uint64_t ticks = end - start;
    if (timestamp_valid_bits_ < 64u) {
        const std::uint64_t mask = (std::uint64_t{1} << timestamp_valid_bits_) - 1u;
        ticks = (end - start) & mask;
    }
    return static_cast<double>(ticks) * timestamp_period_nanoseconds_ / 1'000'000.0;
}

void VolumeRenderer::ensure_export_staging(std::uint32_t slot, VkDeviceSize byte_count)
{
    Buffer& staging = export_staging_buffers_[slot];
    if (staging.handle != VK_NULL_HANDLE && staging.size == byte_count) return;
    destroy_buffer(staging);
    create_buffer(byte_count,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  staging);
}

std::uint32_t VolumeRenderer::build_active_leaf_index(const backend::VbtAssetInfo& asset,
                                                      const Buffer& offsets,
                                                      const Buffer& payload,
                                                      Buffer& mapping,
                                                      VkDescriptorSet descriptor_set)
{
    const std::array<VkDescriptorBufferInfo, 4> infos{{
        {offsets.handle, 0, offsets.size},
        {payload.handle, 0, payload.size},
        {mapping.handle, 0, mapping.size},
        {active_index_counter_buffer_.handle, 0, active_index_counter_buffer_.size},
    }};
    std::array<VkWriteDescriptorSet, 4> writes{};
    for (std::uint32_t binding = 0; binding < writes.size(); ++binding) {
        writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[binding].dstSet = descriptor_set;
        writes[binding].dstBinding = binding;
        writes[binding].descriptorCount = 1;
        writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[binding].pBufferInfo = &infos[binding];
    }
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkCommandBufferAllocateInfo allocation_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation_info.commandPool = command_pool_;
    allocation_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_, &allocation_info, &command_buffer) != VK_SUCCESS) {
        throw std::runtime_error("Unable to allocate active-leaf index command buffer");
    }

    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command_buffer, &begin_info);
    vkCmdFillBuffer(command_buffer, active_index_counter_buffer_.handle, 0, sizeof(std::uint32_t), 0u);

    VkBufferMemoryBarrier counter_clear_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    counter_clear_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    counter_clear_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    counter_clear_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    counter_clear_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    counter_clear_barrier.buffer = active_index_counter_buffer_.handle;
    counter_clear_barrier.offset = 0;
    counter_clear_barrier.size = sizeof(std::uint32_t);
    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0,
                         nullptr,
                         1,
                         &counter_clear_barrier,
                         0,
                         nullptr);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, active_index_pipeline_);
    vkCmdBindDescriptorSets(command_buffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            active_index_pipeline_layout_,
                            0,
                            1,
                            &descriptor_set,
                            0,
                            nullptr);
    ActiveIndexPushConstants push{asset.leaf_count};
    vkCmdPushConstants(command_buffer,
                       active_index_pipeline_layout_,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(push),
                       &push);
    const std::uint64_t group_count = (static_cast<std::uint64_t>(asset.leaf_count) + 255u) / 256u;
    const std::uint32_t groups_x = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(group_count, 65535u));
    const std::uint32_t groups_y = static_cast<std::uint32_t>(
        (group_count + groups_x - 1u) / groups_x);
    vkCmdDispatch(command_buffer, groups_x, groups_y, 1u);

    std::array<VkBufferMemoryBarrier, 2> compute_barriers{};
    compute_barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    compute_barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    compute_barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    compute_barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    compute_barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    compute_barriers[0].buffer = mapping.handle;
    compute_barriers[0].offset = 0;
    compute_barriers[0].size = mapping.size;
    compute_barriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    compute_barriers[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    compute_barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    compute_barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    compute_barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    compute_barriers[1].buffer = active_index_counter_buffer_.handle;
    compute_barriers[1].offset = 0;
    compute_barriers[1].size = sizeof(std::uint32_t);
    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0,
                         nullptr,
                         static_cast<std::uint32_t>(compute_barriers.size()),
                         compute_barriers.data(),
                         0,
                         nullptr);

    VkBufferCopy counter_copy{0, 0, sizeof(std::uint32_t)};
    vkCmdCopyBuffer(command_buffer,
                    active_index_counter_buffer_.handle,
                    active_index_readback_buffer_.handle,
                    1,
                    &counter_copy);
    VkBufferMemoryBarrier host_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    host_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host_barrier.buffer = active_index_readback_buffer_.handle;
    host_barrier.offset = 0;
    host_barrier.size = sizeof(std::uint32_t);
    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0,
                         0,
                         nullptr,
                         1,
                         &host_barrier,
                         0,
                         nullptr);
    vkEndCommandBuffer(command_buffer);

    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    if (vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS ||
        vkQueueWaitIdle(queue_) != VK_SUCCESS) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
        throw std::runtime_error("Unable to build VBT active-leaf index");
    }
    vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);

    void* mapped = nullptr;
    if (vkMapMemory(device_,
                    active_index_readback_buffer_.memory,
                    0,
                    sizeof(std::uint32_t),
                    0,
                    &mapped) != VK_SUCCESS) {
        throw std::runtime_error("Unable to map active-leaf count readback");
    }
    const std::uint32_t active_count = *static_cast<const std::uint32_t*>(mapped);
    vkUnmapMemory(device_, active_index_readback_buffer_.memory);
    if (active_count > asset.leaf_count) {
        throw std::runtime_error("GPU active-leaf index returned an invalid count");
    }
    return active_count;
}

void VolumeRenderer::load_asset(const backend::VbtAssetInfo& asset)
{
    vkDeviceWaitIdle(device_);
    destroy_viewport_resources();
    destroy_asset_resources();

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device_, &properties);
    const VkDeviceSize offset_bytes = (static_cast<VkDeviceSize>(asset.leaf_count) + 1u) * sizeof(std::uint32_t);
    const VkDeviceSize mapping_bytes = static_cast<VkDeviceSize>(asset.leaf_count) * sizeof(std::uint32_t);
    if (offset_bytes > properties.limits.maxStorageBufferRange ||
        asset.payload_bytes > properties.limits.maxStorageBufferRange ||
        mapping_bytes > properties.limits.maxStorageBufferRange) {
        throw std::runtime_error("VBT payload or active-leaf mapping exceeds this GPU's maxStorageBufferRange");
    }

    const auto start = std::chrono::steady_clock::now();
    create_buffer(offset_bytes,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  offset_buffer_);
    create_buffer(asset.payload_bytes,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                   payload_buffer_);
    create_buffer(mapping_bytes,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  active_leaf_mapping_buffer_);
    upload_file_region(asset.path, asset.offset_table_offset, offset_bytes, offset_buffer_);
    upload_file_region(asset.path, asset.payload_offset, asset.payload_bytes, payload_buffer_);
    active_leaf_count_ = build_active_leaf_index(
        asset, offset_buffer_, payload_buffer_, active_leaf_mapping_buffer_, active_index_set_);
    const VkDeviceSize cache_bytes = static_cast<VkDeviceSize>(active_leaf_count_) *
                                     cached_controls_per_leaf * sizeof(float);
    if (cache_bytes > properties.limits.maxStorageBufferRange) {
        throw std::runtime_error("Compact current-frame cache exceeds this GPU's maxStorageBufferRange");
    }
    create_buffer(cache_bytes,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  cache_buffer_);
    const auto end = std::chrono::steady_clock::now();

    asset_ = asset;
    upload_milliseconds_ = std::chrono::duration<double, std::milli>(end - start).count();
    status_ = "Resident on GPU: " + std::to_string(active_leaf_count_) + "/" +
              std::to_string(asset.leaf_count) + " active leaves";
    cache_dirty_ = true;
    render_dirty_ = true;
}

void VolumeRenderer::load_secondary_field(const backend::VbtAssetInfo& asset)
{
    if (!asset_ || asset.frames != asset_->frames) {
        throw std::runtime_error("Secondary field is not timeline-compatible with the primary VBT");
    }
    vkDeviceWaitIdle(device_);
    destroy_viewport_resources();
    destroy_secondary_resources();
    const VkDeviceSize offset_bytes = (static_cast<VkDeviceSize>(asset.leaf_count) + 1u) * sizeof(std::uint32_t);
    const VkDeviceSize mapping_bytes = static_cast<VkDeviceSize>(asset.leaf_count) * sizeof(std::uint32_t);
    create_buffer(offset_bytes,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  secondary_offset_buffer_);
    create_buffer(asset.payload_bytes,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  secondary_payload_buffer_);
    create_buffer(mapping_bytes,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  secondary_active_leaf_mapping_buffer_);
    upload_file_region(asset.path, asset.offset_table_offset, offset_bytes, secondary_offset_buffer_);
    upload_file_region(asset.path, asset.payload_offset, asset.payload_bytes, secondary_payload_buffer_);
    secondary_active_leaf_count_ = build_active_leaf_index(asset,
                                                           secondary_offset_buffer_,
                                                           secondary_payload_buffer_,
                                                           secondary_active_leaf_mapping_buffer_,
                                                           secondary_active_index_set_);
    const VkDeviceSize cache_bytes = static_cast<VkDeviceSize>(secondary_active_leaf_count_) *
                                     cached_controls_per_leaf * sizeof(float);
    create_buffer(cache_bytes,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  secondary_cache_buffer_);
    secondary_field_ = asset;
    cache_dirty_ = true;
    status_ = "Density + " + backend::field_role_name(asset.role) + " resident on GPU";
    render_dirty_ = true;
    rays_dirty_ = true;
}

void VolumeRenderer::load_temperature_field(const backend::VbtAssetInfo& asset)
{
    if (!asset_ || asset.frames != asset_->frames) {
        throw std::runtime_error("Temperature field is not timeline-compatible with the primary VBT");
    }
    vkDeviceWaitIdle(device_);
    destroy_viewport_resources();
    destroy_temperature_resources();
    const VkDeviceSize offset_bytes = (static_cast<VkDeviceSize>(asset.leaf_count) + 1u) * sizeof(std::uint32_t);
    const VkDeviceSize mapping_bytes = static_cast<VkDeviceSize>(asset.leaf_count) * sizeof(std::uint32_t);
    create_buffer(offset_bytes,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  temperature_offset_buffer_);
    create_buffer(asset.payload_bytes,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  temperature_payload_buffer_);
    create_buffer(mapping_bytes,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  temperature_active_leaf_mapping_buffer_);
    upload_file_region(asset.path, asset.offset_table_offset, offset_bytes, temperature_offset_buffer_);
    upload_file_region(asset.path, asset.payload_offset, asset.payload_bytes, temperature_payload_buffer_);
    temperature_active_leaf_count_ = build_active_leaf_index(asset,
                                                             temperature_offset_buffer_,
                                                             temperature_payload_buffer_,
                                                             temperature_active_leaf_mapping_buffer_,
                                                             temperature_active_index_set_);
    const VkDeviceSize cache_bytes = static_cast<VkDeviceSize>(temperature_active_leaf_count_) *
                                     cached_controls_per_leaf * sizeof(float);
    create_buffer(cache_bytes,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  temperature_cache_buffer_);
    temperature_field_ = asset;
    cache_dirty_ = true;
    status_ = "Density + flames + temperature resident on GPU";
    render_dirty_ = true;
    rays_dirty_ = true;
}

void VolumeRenderer::destroy_asset_resources()
{
    destroy_secondary_resources();
    destroy_temperature_resources();
    destroy_buffer(payload_buffer_);
    destroy_buffer(cache_buffer_);
    destroy_buffer(active_leaf_mapping_buffer_);
    destroy_buffer(offset_buffer_);
    secondary_field_.reset();
    temperature_field_.reset();
    active_leaf_count_ = 0u;
}

void VolumeRenderer::destroy_secondary_resources()
{
    destroy_buffer(secondary_payload_buffer_);
    destroy_buffer(secondary_cache_buffer_);
    destroy_buffer(secondary_active_leaf_mapping_buffer_);
    destroy_buffer(secondary_offset_buffer_);
    secondary_active_leaf_count_ = 0u;
}

void VolumeRenderer::destroy_temperature_resources()
{
    destroy_buffer(temperature_payload_buffer_);
    destroy_buffer(temperature_cache_buffer_);
    destroy_buffer(temperature_active_leaf_mapping_buffer_);
    destroy_buffer(temperature_offset_buffer_);
    temperature_active_leaf_count_ = 0u;
}

void VolumeRenderer::create_viewport_resources(std::uint32_t width, std::uint32_t height)
{
    const std::uint64_t ray_count = static_cast<std::uint64_t>(width) * height;
    if (ray_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Viewport ray count is too large");
    }

    create_buffer(ray_count * sizeof(float) * 8u,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  ray_buffer_);
    create_buffer(ray_count * sizeof(float) * 8u,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  result_buffer_);
    if (secondary_field_) {
        create_buffer(ray_count * sizeof(float) * 8u,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      secondary_result_buffer_);
    }
    if (temperature_field_) {
        create_buffer(ray_count * sizeof(float) * 8u,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      temperature_result_buffer_);
    }
    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    image_info.extent = {width, height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device_, &image_info, nullptr, &output_image_.handle) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create RGBA16F viewport image");
    }
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device_, output_image_.handle, &requirements);
    VkMemoryAllocateInfo memory_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memory_info.allocationSize = requirements.size;
    memory_info.memoryTypeIndex = find_memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device_, &memory_info, nullptr, &output_image_.memory) != VK_SUCCESS ||
        vkBindImageMemory(device_, output_image_.handle, output_image_.memory, 0) != VK_SUCCESS) {
        throw std::runtime_error("Unable to allocate RGBA16F viewport image");
    }

    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = output_image_.handle;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = image_info.format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device_, &view_info, nullptr, &output_image_.view) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create viewport image view");
    }

    VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.maxLod = 1.0f;
    if (vkCreateSampler(device_, &sampler_info, nullptr, &output_image_.sampler) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create viewport sampler");
    }

    transition_output_image();
    texture_descriptor_ = ImGui_ImplVulkan_AddTexture(
        output_image_.sampler, output_image_.view, VK_IMAGE_LAYOUT_GENERAL);
    render_width_ = width;
    render_height_ = height;
    rays_dirty_ = true;
}

void VolumeRenderer::destroy_viewport_resources()
{
    if (texture_descriptor_ != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(texture_descriptor_);
        texture_descriptor_ = VK_NULL_HANDLE;
    }
    if (output_image_.sampler != VK_NULL_HANDLE) vkDestroySampler(device_, output_image_.sampler, nullptr);
    if (output_image_.view != VK_NULL_HANDLE) vkDestroyImageView(device_, output_image_.view, nullptr);
    if (output_image_.handle != VK_NULL_HANDLE) vkDestroyImage(device_, output_image_.handle, nullptr);
    if (output_image_.memory != VK_NULL_HANDLE) vkFreeMemory(device_, output_image_.memory, nullptr);
    output_image_ = {};
    destroy_buffer(result_buffer_);
    destroy_buffer(secondary_result_buffer_);
    destroy_buffer(temperature_result_buffer_);
    destroy_buffer(ray_buffer_);
    render_width_ = 0;
    render_height_ = 0;
}

void VolumeRenderer::update_descriptor_sets()
{
    const std::array<VkDescriptorBufferInfo, 6> primary_infos{{
        {ray_buffer_.handle, 0, ray_buffer_.size},
        {offset_buffer_.handle, 0, offset_buffer_.size},
        {payload_buffer_.handle, 0, payload_buffer_.size},
        {result_buffer_.handle, 0, result_buffer_.size},
        {cache_buffer_.handle, 0, cache_buffer_.size},
        {active_leaf_mapping_buffer_.handle, 0, active_leaf_mapping_buffer_.size},
    }};
    const std::array<VkDescriptorBufferInfo, 6> secondary_infos{{
        primary_infos[0],
        secondary_field_ ? VkDescriptorBufferInfo{secondary_offset_buffer_.handle, 0, secondary_offset_buffer_.size}
                         : primary_infos[1],
        secondary_field_ ? VkDescriptorBufferInfo{secondary_payload_buffer_.handle, 0, secondary_payload_buffer_.size}
                         : primary_infos[2],
        secondary_field_ ? VkDescriptorBufferInfo{secondary_result_buffer_.handle, 0, secondary_result_buffer_.size}
                          : primary_infos[3],
        secondary_field_ ? VkDescriptorBufferInfo{secondary_cache_buffer_.handle, 0, secondary_cache_buffer_.size}
                          : primary_infos[4],
        secondary_field_ ? VkDescriptorBufferInfo{secondary_active_leaf_mapping_buffer_.handle, 0, secondary_active_leaf_mapping_buffer_.size}
                          : primary_infos[5],
    }};
    const std::array<VkDescriptorBufferInfo, 6> temperature_infos{{
        primary_infos[0],
        temperature_field_ ? VkDescriptorBufferInfo{temperature_offset_buffer_.handle, 0, temperature_offset_buffer_.size}
                           : primary_infos[1],
        temperature_field_ ? VkDescriptorBufferInfo{temperature_payload_buffer_.handle, 0, temperature_payload_buffer_.size}
                           : primary_infos[2],
        temperature_field_ ? VkDescriptorBufferInfo{temperature_result_buffer_.handle, 0, temperature_result_buffer_.size}
                            : primary_infos[3],
        temperature_field_ ? VkDescriptorBufferInfo{temperature_cache_buffer_.handle, 0, temperature_cache_buffer_.size}
                            : primary_infos[4],
        temperature_field_ ? VkDescriptorBufferInfo{temperature_active_leaf_mapping_buffer_.handle, 0, temperature_active_leaf_mapping_buffer_.size}
                            : primary_infos[5],
    }};
    std::array<VkWriteDescriptorSet, 24> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = ray_set_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &primary_infos[0];
    for (std::uint32_t index = 0; index < 6; ++index) {
        writes[index + 1u].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index + 1u].dstSet = sample_set_;
        writes[index + 1u].dstBinding = index;
        writes[index + 1u].descriptorCount = 1;
        writes[index + 1u].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index + 1u].pBufferInfo = &primary_infos[index];
    }
    for (std::uint32_t index = 0; index < 6; ++index) {
        writes[index + 7u].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index + 7u].dstSet = secondary_sample_set_;
        writes[index + 7u].dstBinding = index;
        writes[index + 7u].descriptorCount = 1;
        writes[index + 7u].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index + 7u].pBufferInfo = &secondary_infos[index];
    }
    for (std::uint32_t index = 0; index < 6; ++index) {
        writes[index + 13u].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index + 13u].dstSet = temperature_sample_set_;
        writes[index + 13u].dstBinding = index;
        writes[index + 13u].descriptorCount = 1;
        writes[index + 13u].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index + 13u].pBufferInfo = &temperature_infos[index];
    }
    const std::array<const VkDescriptorBufferInfo*, 3> color_infos{
        &primary_infos[3], &secondary_infos[3], &temperature_infos[3]};
    for (std::uint32_t index = 0; index < 3; ++index) {
        writes[index + 19u].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index + 19u].dstSet = color_set_;
        writes[index + 19u].dstBinding = index;
        writes[index + 19u].descriptorCount = 1;
        writes[index + 19u].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index + 19u].pBufferInfo = color_infos[index];
    }
    VkDescriptorImageInfo image_info{};
    image_info.imageView = output_image_.view;
    image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    writes[22].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[22].dstSet = color_set_;
    writes[22].dstBinding = 3;
    writes[22].descriptorCount = 1;
    writes[22].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[22].pImageInfo = &image_info;
    writes[23].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[23].dstSet = color_set_;
    writes[23].dstBinding = 4;
    writes[23].descriptorCount = 1;
    writes[23].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[23].pBufferInfo = &primary_infos[0];
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

    const std::array<std::array<VkDescriptorBufferInfo, 4>, 3> cache_infos{{
        {primary_infos[1], primary_infos[2], primary_infos[4], primary_infos[5]},
        {secondary_infos[1], secondary_infos[2], secondary_infos[4], secondary_infos[5]},
        {temperature_infos[1], temperature_infos[2], temperature_infos[4], temperature_infos[5]},
    }};
    const std::array<VkDescriptorSet, 3> cache_sets{
        cache_set_, secondary_cache_set_, temperature_cache_set_};
    std::array<VkWriteDescriptorSet, 12> cache_writes{};
    for (std::uint32_t field = 0; field < cache_sets.size(); ++field) {
        for (std::uint32_t binding = 0; binding < 4u; ++binding) {
            auto& write = cache_writes[field * 4u + binding];
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = cache_sets[field];
            write.dstBinding = binding;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &cache_infos[field][binding];
        }
    }
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(cache_writes.size()), cache_writes.data(), 0, nullptr);
}

std::uint32_t VolumeRenderer::find_memory_type(std::uint32_t type_bits,
                                               VkMemoryPropertyFlags properties) const
{
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties);
    for (std::uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
        if ((type_bits & (1u << index)) != 0u &&
            (memory_properties.memoryTypes[index].propertyFlags & properties) == properties) {
            return index;
        }
    }
    throw std::runtime_error("No compatible Vulkan memory type");
}

void VolumeRenderer::create_buffer(VkDeviceSize size,
                                   VkBufferUsageFlags usage,
                                   VkMemoryPropertyFlags properties,
                                   Buffer& buffer)
{
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = std::max<VkDeviceSize>(size, 4);
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer.handle) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create Vulkan buffer");
    }
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, buffer.handle, &requirements);
    VkMemoryAllocateInfo memory_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memory_info.allocationSize = requirements.size;
    memory_info.memoryTypeIndex = find_memory_type(requirements.memoryTypeBits, properties);
    if (vkAllocateMemory(device_, &memory_info, nullptr, &buffer.memory) != VK_SUCCESS ||
        vkBindBufferMemory(device_, buffer.handle, buffer.memory, 0) != VK_SUCCESS) {
        throw std::runtime_error("Unable to allocate Vulkan buffer");
    }
    buffer.size = buffer_info.size;
}

void VolumeRenderer::destroy_buffer(Buffer& buffer)
{
    if (buffer.handle != VK_NULL_HANDLE) vkDestroyBuffer(device_, buffer.handle, nullptr);
    if (buffer.memory != VK_NULL_HANDLE) vkFreeMemory(device_, buffer.memory, nullptr);
    buffer = {};
}

void VolumeRenderer::upload_file_region(const std::filesystem::path& path,
                                        std::uint64_t file_offset,
                                        VkDeviceSize size,
                                        Buffer& destination)
{
    Buffer staging;
    create_buffer(size,
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  staging);
    void* mapped = nullptr;
    if (vkMapMemory(device_, staging.memory, 0, size, 0, &mapped) != VK_SUCCESS) {
        destroy_buffer(staging);
        throw std::runtime_error("Unable to map VBT upload buffer");
    }
    std::ifstream input(path, std::ios::binary);
    input.seekg(static_cast<std::streamoff>(file_offset));
    input.read(static_cast<char*>(mapped), static_cast<std::streamsize>(size));
    vkUnmapMemory(device_, staging.memory);
    if (!input) {
        destroy_buffer(staging);
        throw std::runtime_error("Unable to stream VBT data into the GPU upload buffer");
    }
    copy_buffer(staging, destination, size);
    destroy_buffer(staging);
}

void VolumeRenderer::copy_buffer(const Buffer& source, const Buffer& destination, VkDeviceSize size)
{
    VkCommandBufferAllocateInfo allocation_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation_info.commandPool = command_pool_;
    allocation_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_, &allocation_info, &command_buffer) != VK_SUCCESS) {
        throw std::runtime_error("Unable to allocate VBT upload command buffer");
    }
    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command_buffer, &begin_info);
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(command_buffer, source.handle, destination.handle, 1, &region);
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = destination.handle;
    barrier.offset = 0;
    barrier.size = size;
    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0,
                         nullptr,
                         1,
                         &barrier,
                         0,
                         nullptr);
    vkEndCommandBuffer(command_buffer);
    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    if (vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS ||
        vkQueueWaitIdle(queue_) != VK_SUCCESS) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
        throw std::runtime_error("Unable to upload VBT data to resident GPU memory");
    }
    vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
}

void VolumeRenderer::transition_output_image()
{
    VkCommandBufferAllocateInfo allocation_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation_info.commandPool = command_pool_;
    allocation_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device_, &allocation_info, &command_buffer);
    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command_buffer, &begin_info);
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = output_image_.handle;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &barrier);
    vkEndCommandBuffer(command_buffer);
    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue_);
    vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
}

} // namespace vbtstudio::frontend
