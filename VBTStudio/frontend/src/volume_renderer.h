#pragma once

#include "vbtstudio/backend/camera.h"
#include "vbtstudio/backend/material.h"
#include "vbtstudio/backend/vbt_asset.h"

#include <imgui.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace vbtstudio::frontend {

class VolumeRenderer {
public:
    struct GpuTimings {
        bool supported = false;
        bool valid = false;
        bool rays_regenerated = false;
        std::uint32_t frame = 0;
        std::uint32_t field_count = 0;
        double ray_milliseconds = 0.0;
        double sample_milliseconds = 0.0;
        double colorize_milliseconds = 0.0;
        double total_milliseconds = 0.0;
    };

    struct PngExportResult {
        std::filesystem::path path;
        std::string error;
        std::uint32_t frame = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        bool success = false;
    };

    struct CacheStats {
        std::uint32_t active_leaves = 0;
        std::uint32_t total_leaves = 0;
        std::uint64_t cache_bytes = 0;
        std::uint64_t fixed_cache_bytes = 0;
        std::uint64_t mapping_bytes = 0;
    };

    VolumeRenderer(VkPhysicalDevice physical_device,
                   VkDevice device,
                   VkQueue queue,
                   std::uint32_t queue_family);
    ~VolumeRenderer();

    VolumeRenderer(const VolumeRenderer&) = delete;
    VolumeRenderer& operator=(const VolumeRenderer&) = delete;

    void sync_asset(const std::optional<backend::VbtAssetInfo>& asset);
    void sync_secondary_field(const std::optional<backend::VbtAssetInfo>& asset);
    void sync_temperature_field(const std::optional<backend::VbtAssetInfo>& asset);
    void ensure_viewport(std::uint32_t width, std::uint32_t height);
    void set_render_state(std::uint32_t frame,
                          const backend::MaterialState& material,
                          const backend::CameraState& camera);
    void invalidate_render() noexcept;
    bool export_png(const std::filesystem::path& path);
    void collect_gpu_timings(std::uint32_t query_slot) noexcept;
    [[nodiscard]] std::optional<PngExportResult> collect_png_export(std::uint32_t query_slot);
    bool record(VkCommandBuffer command_buffer,
                std::uint32_t query_slot,
                const std::optional<std::filesystem::path>& export_path = std::nullopt);
    void shutdown();

    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] ImTextureID texture_id() const noexcept;
    [[nodiscard]] const std::string& status() const noexcept;
    [[nodiscard]] double upload_milliseconds() const noexcept;
    [[nodiscard]] const GpuTimings& gpu_timings() const noexcept;
    [[nodiscard]] CacheStats cache_stats() const noexcept;
    [[nodiscard]] std::uint32_t render_width() const noexcept;
    [[nodiscard]] std::uint32_t render_height() const noexcept;

private:
    struct Buffer {
        VkBuffer handle = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
    };

    struct Image {
        VkImage handle = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
    };

    void create_pipeline_resources();
    void destroy_pipeline_resources();
    void load_asset(const backend::VbtAssetInfo& asset);
    void load_secondary_field(const backend::VbtAssetInfo& asset);
    void load_temperature_field(const backend::VbtAssetInfo& asset);
    void destroy_asset_resources();
    void destroy_secondary_resources();
    void destroy_temperature_resources();
    void create_viewport_resources(std::uint32_t width, std::uint32_t height);
    void destroy_viewport_resources();
    void update_descriptor_sets();
    std::uint32_t build_active_leaf_index(const backend::VbtAssetInfo& asset,
                                          const Buffer& offsets,
                                          const Buffer& payload,
                                          Buffer& mapping,
                                          VkDescriptorSet descriptor_set);

    [[nodiscard]] std::uint32_t find_memory_type(std::uint32_t type_bits,
                                                 VkMemoryPropertyFlags properties) const;
    void create_buffer(VkDeviceSize size,
                       VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags properties,
                       Buffer& buffer);
    void destroy_buffer(Buffer& buffer);
    void upload_file_region(const std::filesystem::path& path,
                            std::uint64_t file_offset,
                            VkDeviceSize size,
                            Buffer& destination);
    void copy_buffer(const Buffer& source, const Buffer& destination, VkDeviceSize size);
    void transition_output_image();
    void ensure_export_staging(std::uint32_t slot, VkDeviceSize byte_count);
    [[nodiscard]] double timestamp_delta_milliseconds(std::uint64_t start,
                                                      std::uint64_t end) const noexcept;

    static constexpr std::uint32_t timing_query_count_ = 4;
    static constexpr std::uint32_t timing_slot_count_ = 16;

    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout ray_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout sample_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout cache_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout active_index_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout color_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet ray_set_ = VK_NULL_HANDLE;
    VkDescriptorSet sample_set_ = VK_NULL_HANDLE;
    VkDescriptorSet secondary_sample_set_ = VK_NULL_HANDLE;
    VkDescriptorSet temperature_sample_set_ = VK_NULL_HANDLE;
    VkDescriptorSet cache_set_ = VK_NULL_HANDLE;
    VkDescriptorSet secondary_cache_set_ = VK_NULL_HANDLE;
    VkDescriptorSet temperature_cache_set_ = VK_NULL_HANDLE;
    VkDescriptorSet active_index_set_ = VK_NULL_HANDLE;
    VkDescriptorSet secondary_active_index_set_ = VK_NULL_HANDLE;
    VkDescriptorSet temperature_active_index_set_ = VK_NULL_HANDLE;
    VkDescriptorSet color_set_ = VK_NULL_HANDLE;
    VkPipelineLayout ray_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout sample_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout cache_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout active_index_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout color_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline ray_pipeline_ = VK_NULL_HANDLE;
    VkPipeline sample_pipeline_ = VK_NULL_HANDLE;
    VkPipeline cache_pipeline_ = VK_NULL_HANDLE;
    VkPipeline active_index_pipeline_ = VK_NULL_HANDLE;
    VkPipeline color_pipeline_ = VK_NULL_HANDLE;
    VkQueryPool timing_query_pool_ = VK_NULL_HANDLE;

    Buffer offset_buffer_;
    Buffer payload_buffer_;
    Buffer cache_buffer_;
    Buffer active_leaf_mapping_buffer_;
    Buffer secondary_offset_buffer_;
    Buffer secondary_payload_buffer_;
    Buffer secondary_cache_buffer_;
    Buffer secondary_active_leaf_mapping_buffer_;
    Buffer temperature_offset_buffer_;
    Buffer temperature_payload_buffer_;
    Buffer temperature_cache_buffer_;
    Buffer temperature_active_leaf_mapping_buffer_;
    Buffer active_index_counter_buffer_;
    Buffer active_index_readback_buffer_;
    Buffer ray_buffer_;
    Buffer result_buffer_;
    Buffer secondary_result_buffer_;
    Buffer temperature_result_buffer_;
    std::array<Buffer, timing_slot_count_> export_staging_buffers_{};
    Image output_image_;
    VkDescriptorSet texture_descriptor_ = VK_NULL_HANDLE;

    std::optional<backend::VbtAssetInfo> asset_;
    std::optional<backend::VbtAssetInfo> secondary_field_;
    std::optional<backend::VbtAssetInfo> temperature_field_;
    std::string status_ = "No resident VBT asset";
    double upload_milliseconds_ = 0.0;
    std::uint32_t active_leaf_count_ = 0;
    std::uint32_t secondary_active_leaf_count_ = 0;
    std::uint32_t temperature_active_leaf_count_ = 0;
    double timestamp_period_nanoseconds_ = 0.0;
    std::uint32_t timestamp_valid_bits_ = 0;
    std::array<bool, timing_slot_count_> timing_pending_{};
    std::array<bool, timing_slot_count_> timing_rays_regenerated_{};
    std::array<std::uint32_t, timing_slot_count_> timing_frames_{};
    std::array<std::uint32_t, timing_slot_count_> timing_field_counts_{};
    std::array<bool, timing_slot_count_> export_pending_{};
    std::array<std::filesystem::path, timing_slot_count_> export_paths_{};
    std::array<std::uint32_t, timing_slot_count_> export_frames_{};
    std::array<std::uint32_t, timing_slot_count_> export_widths_{};
    std::array<std::uint32_t, timing_slot_count_> export_heights_{};
    GpuTimings gpu_timings_{};
    std::uint32_t render_width_ = 0;
    std::uint32_t render_height_ = 0;
    std::uint32_t frame_ = 0;
    backend::MaterialState material_;
    backend::CameraState camera_;
    bool rays_dirty_ = true;
    bool cache_dirty_ = true;
    bool render_dirty_ = false;
    bool initialized_ = false;
};

} // namespace vbtstudio::frontend
