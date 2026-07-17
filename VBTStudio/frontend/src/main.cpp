#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

#include <imgui.h>
#include <imgui_internal.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "vbtstudio/backend/studio_session.h"
#include "volume_renderer.h"

#ifdef _WIN32
#include <Windows.h>
#include <commdlg.h>
#include <ShlObj.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

VkAllocationCallbacks* allocator = nullptr;
VkInstance instance = VK_NULL_HANDLE;
VkPhysicalDevice physical_device = VK_NULL_HANDLE;
VkDevice device = VK_NULL_HANDLE;
std::uint32_t queue_family = std::numeric_limits<std::uint32_t>::max();
VkQueue queue = VK_NULL_HANDLE;
VkDebugReportCallbackEXT debug_report = VK_NULL_HANDLE;
VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
ImGui_ImplVulkanH_Window main_window;
int min_image_count = 2;
bool swapchain_rebuild = false;

struct WindowPlacement {
    int width;
    int height;
    int x;
    int y;
};

struct SequenceTimingRow {
    std::filesystem::path path;
    vbtstudio::frontend::VolumeRenderer::GpuTimings timings;
    std::uint32_t frame = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    bool success = false;
    std::string error;
};

struct SequenceExportState {
    bool active = false;
    bool scheduling_complete = false;
    bool close_when_complete = false;
    std::filesystem::path output_directory;
    std::filesystem::path timing_csv;
    std::filesystem::path video_path;
    std::filesystem::path ffmpeg_path;
    std::string file_prefix;
    std::string status;
    std::uint32_t start_frame = 0;
    std::uint32_t end_frame = 0;
    std::uint32_t frame_step = 1;
    std::uint32_t next_frame = 0;
    std::uint32_t scheduled = 0;
    std::uint32_t completed = 0;
    std::uint32_t total = 0;
    std::uint32_t render_width = 960;
    std::uint32_t render_height = 550;
    double frames_per_second = 24.0;
    std::vector<SequenceTimingRow> rows;
};

struct FrameRenderResult {
    std::optional<vbtstudio::frontend::VolumeRenderer::PngExportResult> completed_export;
    vbtstudio::frontend::VolumeRenderer::GpuTimings completed_timings;
    bool export_scheduled = false;
};

WindowPlacement initial_window_placement()
{
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if (!monitor) return {1440, 864, 80, 60};

    int work_x = 0;
    int work_y = 0;
    int work_width = 0;
    int work_height = 0;
    glfwGetMonitorWorkarea(monitor, &work_x, &work_y, &work_width, &work_height);

    const int width = std::max(960, static_cast<int>(std::round(static_cast<double>(work_width) * 0.90)));
    const int height = std::max(640, static_cast<int>(std::round(static_cast<double>(work_height) * 0.90)));
    return {
        std::min(width, work_width),
        std::min(height, work_height),
        work_x + (work_width - std::min(width, work_width)) / 2,
        work_y + (work_height - std::min(height, work_height)) / 2,
    };
}

void check_vk_result(VkResult error)
{
    if (error == VK_SUCCESS) return;
    std::fprintf(stderr, "Vulkan error: %d\n", error);
    if (error < 0) std::abort();
}

void setup_vulkan(const char** extensions, std::uint32_t extension_count)
{
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.enabledExtensionCount = extension_count;
    instance_info.ppEnabledExtensionNames = extensions;
    check_vk_result(vkCreateInstance(&instance_info, allocator, &instance));

    physical_device = ImGui_ImplVulkanH_SelectPhysicalDevice(instance);
    queue_family = ImGui_ImplVulkanH_SelectQueueFamilyIndex(physical_device);

    const std::array<const char*, 1> device_extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = static_cast<std::uint32_t>(device_extensions.size());
    device_info.ppEnabledExtensionNames = device_extensions.data();
    check_vk_result(vkCreateDevice(physical_device, &device_info, allocator, &device));
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    const std::array<VkDescriptorPoolSize, 11> pool_sizes{{
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000},
    }};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000 * static_cast<std::uint32_t>(pool_sizes.size());
    pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    check_vk_result(vkCreateDescriptorPool(device, &pool_info, allocator, &descriptor_pool));
}

void setup_vulkan_window(GLFWwindow* window, int width, int height)
{
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    check_vk_result(glfwCreateWindowSurface(instance, window, allocator, &surface));
    VkBool32 supported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, queue_family, surface, &supported);
    if (supported != VK_TRUE) throw std::runtime_error("Selected Vulkan queue cannot present to the GLFW surface.");

    main_window.Surface = surface;
    const VkFormat formats[] = {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
    main_window.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
        physical_device, surface, formats, static_cast<int>(std::size(formats)), VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);
    const VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_FIFO_KHR};
    main_window.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
        physical_device, surface, present_modes, static_cast<int>(std::size(present_modes)));
    ImGui_ImplVulkanH_CreateOrResizeWindow(
        instance,
        physical_device,
        device,
        &main_window,
        queue_family,
        allocator,
        width,
        height,
        min_image_count,
        0);
}

void cleanup_vulkan_window()
{
    ImGui_ImplVulkanH_DestroyWindow(instance, device, &main_window, allocator);
}

void cleanup_vulkan()
{
    vkDestroyDescriptorPool(device, descriptor_pool, allocator);
    vkDestroyDevice(device, allocator);
    vkDestroyInstance(instance, allocator);
}

FrameRenderResult frame_render(
    ImDrawData* draw_data,
    vbtstudio::frontend::VolumeRenderer& volume_renderer,
    const std::optional<std::filesystem::path>& export_path = std::nullopt)
{
    FrameRenderResult frame_result{};
    VkSemaphore image_acquired = main_window.FrameSemaphores[main_window.SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore render_complete = main_window.FrameSemaphores[main_window.SemaphoreIndex].RenderCompleteSemaphore;
    VkResult result = vkAcquireNextImageKHR(
        device, main_window.Swapchain, UINT64_MAX, image_acquired, VK_NULL_HANDLE, &main_window.FrameIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        swapchain_rebuild = true;
        return frame_result;
    }
    check_vk_result(result);

    ImGui_ImplVulkanH_Frame* frame = &main_window.Frames[main_window.FrameIndex];
    check_vk_result(vkWaitForFences(device, 1, &frame->Fence, VK_TRUE, UINT64_MAX));
    volume_renderer.collect_gpu_timings(main_window.FrameIndex);
    frame_result.completed_timings = volume_renderer.gpu_timings();
    frame_result.completed_export = volume_renderer.collect_png_export(main_window.FrameIndex);
    check_vk_result(vkResetFences(device, 1, &frame->Fence));
    check_vk_result(vkResetCommandPool(device, frame->CommandPool, 0));

    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk_result(vkBeginCommandBuffer(frame->CommandBuffer, &begin_info));

    frame_result.export_scheduled =
        volume_renderer.record(frame->CommandBuffer, main_window.FrameIndex, export_path) && export_path.has_value();

    const VkClearValue clear_value{{{0.055f, 0.055f, 0.052f, 1.0f}}};
    VkRenderPassBeginInfo render_pass_info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    render_pass_info.renderPass = main_window.RenderPass;
    render_pass_info.framebuffer = frame->Framebuffer;
    render_pass_info.renderArea.extent.width = main_window.Width;
    render_pass_info.renderArea.extent.height = main_window.Height;
    render_pass_info.clearValueCount = 1;
    render_pass_info.pClearValues = &clear_value;
    vkCmdBeginRenderPass(frame->CommandBuffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(draw_data, frame->CommandBuffer);
    vkCmdEndRenderPass(frame->CommandBuffer);
    check_vk_result(vkEndCommandBuffer(frame->CommandBuffer));

    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &image_acquired;
    submit_info.pWaitDstStageMask = &wait_stage;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &frame->CommandBuffer;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &render_complete;
    check_vk_result(vkQueueSubmit(queue, 1, &submit_info, frame->Fence));
    return frame_result;
}

void frame_present()
{
    if (swapchain_rebuild) return;
    VkSemaphore render_complete = main_window.FrameSemaphores[main_window.SemaphoreIndex].RenderCompleteSemaphore;
    VkPresentInfoKHR present_info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &render_complete;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &main_window.Swapchain;
    present_info.pImageIndices = &main_window.FrameIndex;
    const VkResult result = vkQueuePresentKHR(queue, &present_info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        swapchain_rebuild = true;
    }
    else {
        check_vk_result(result);
    }
    main_window.SemaphoreIndex = (main_window.SemaphoreIndex + 1) % main_window.SemaphoreCount;
}

void apply_style()
{
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 3.0f;
    style.ChildRounding = 3.0f;
    style.FrameRounding = 3.0f;
    style.PopupRounding = 3.0f;
    style.ScrollbarRounding = 3.0f;
    style.GrabRounding = 3.0f;
    style.TabRounding = 3.0f;
    style.FramePadding = ImVec2(8.0f, 5.0f);
    style.ItemSpacing = ImVec2(8.0f, 7.0f);
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.075f, 0.075f, 0.072f, 1.0f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.065f, 0.065f, 0.062f, 1.0f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.095f, 0.095f, 0.090f, 1.0f);
    colors[ImGuiCol_Border] = ImVec4(0.22f, 0.22f, 0.20f, 1.0f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.13f, 0.13f, 0.12f, 1.0f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.19f, 0.18f, 0.16f, 1.0f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.24f, 0.21f, 0.16f, 1.0f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.055f, 0.055f, 0.052f, 1.0f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.075f, 0.075f, 0.070f, 1.0f);
    colors[ImGuiCol_Header] = ImVec4(0.27f, 0.20f, 0.10f, 1.0f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.38f, 0.27f, 0.11f, 1.0f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.46f, 0.31f, 0.10f, 1.0f);
    colors[ImGuiCol_Button] = ImVec4(0.19f, 0.17f, 0.13f, 1.0f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.38f, 0.27f, 0.11f, 1.0f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.48f, 0.32f, 0.10f, 1.0f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.98f, 0.65f, 0.20f, 1.0f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.95f, 0.58f, 0.16f, 1.0f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(1.0f, 0.72f, 0.25f, 1.0f);
    colors[ImGuiCol_Tab] = ImVec4(0.10f, 0.10f, 0.095f, 1.0f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.34f, 0.24f, 0.11f, 1.0f);
    colors[ImGuiCol_TabSelected] = ImVec4(0.24f, 0.18f, 0.10f, 1.0f);
    colors[ImGuiCol_DockingPreview] = ImVec4(0.95f, 0.58f, 0.16f, 0.70f);
}

#ifdef _WIN32
std::optional<std::filesystem::path> open_vbt_dialog(GLFWwindow* window)
{
    wchar_t buffer[32768]{};
    const std::filesystem::path default_directory(VBTSTUDIO_DEFAULT_VBT_DIR);
    const std::wstring default_directory_text = default_directory.wstring();
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = glfwGetWin32Window(window);
    dialog.lpstrFile = buffer;
    dialog.nMaxFile = static_cast<DWORD>(std::size(buffer));
    dialog.lpstrFilter = L"VBT Packed Volume (*.vbtp)\0*.vbtp\0All Files (*.*)\0*.*\0";
    dialog.lpstrInitialDir = default_directory_text.c_str();
    dialog.nFilterIndex = 1;
    dialog.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameW(&dialog) == TRUE) return std::filesystem::path(buffer);
    return std::nullopt;
}

std::optional<std::filesystem::path> save_png_dialog(GLFWwindow* window, const std::string& suggested_name)
{
    std::array<wchar_t, 32768> buffer{};
    const std::wstring initial_name(suggested_name.begin(), suggested_name.end());
    std::copy(initial_name.begin(), initial_name.end(), buffer.begin());
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = glfwGetWin32Window(window);
    dialog.lpstrFile = buffer.data();
    dialog.nMaxFile = static_cast<DWORD>(buffer.size());
    dialog.lpstrFilter = L"PNG Image (*.png)\0*.png\0";
    dialog.lpstrDefExt = L"png";
    dialog.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetSaveFileNameW(&dialog) == TRUE) return std::filesystem::path(buffer.data());
    return std::nullopt;
}

std::optional<std::filesystem::path> select_folder_dialog(GLFWwindow* window)
{
    BROWSEINFOW browse{};
    browse.hwndOwner = glfwGetWin32Window(window);
    browse.lpszTitle = L"Select sequence output folder";
    browse.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
    PIDLIST_ABSOLUTE item = SHBrowseForFolderW(&browse);
    if (item == nullptr) return std::nullopt;
    std::array<wchar_t, 32768> buffer{};
    const bool success = SHGetPathFromIDListW(item, buffer.data()) == TRUE;
    CoTaskMemFree(item);
    if (success) return std::filesystem::path(buffer.data());
    return std::nullopt;
}

std::optional<std::filesystem::path> save_mp4_dialog(GLFWwindow* window, const std::string& suggested_name)
{
    std::array<wchar_t, 32768> buffer{};
    const std::wstring initial_name(suggested_name.begin(), suggested_name.end());
    std::copy(initial_name.begin(), initial_name.end(), buffer.begin());
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = glfwGetWin32Window(window);
    dialog.lpstrFile = buffer.data();
    dialog.nMaxFile = static_cast<DWORD>(buffer.size());
    dialog.lpstrFilter = L"MPEG-4 Video (*.mp4)\0*.mp4\0";
    dialog.lpstrDefExt = L"mp4";
    dialog.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetSaveFileNameW(&dialog) == TRUE) return std::filesystem::path(buffer.data());
    return std::nullopt;
}
#else
std::optional<std::filesystem::path> open_vbt_dialog(GLFWwindow*) { return std::nullopt; }
std::optional<std::filesystem::path> save_png_dialog(GLFWwindow*, const std::string&) { return std::nullopt; }
std::optional<std::filesystem::path> select_folder_dialog(GLFWwindow*) { return std::nullopt; }
std::optional<std::filesystem::path> save_mp4_dialog(GLFWwindow*, const std::string&) { return std::nullopt; }
#endif

std::filesystem::path sequence_frame_path(const SequenceExportState& sequence, std::uint32_t frame)
{
    std::ostringstream name;
    name << sequence.file_prefix << "_frame" << std::setfill('0') << std::setw(4) << frame << ".png";
    return sequence.output_directory / name.str();
}

bool begin_sequence_export(SequenceExportState& sequence,
                           vbtstudio::backend::StudioSession& session,
                           const std::filesystem::path& output_directory,
                           std::uint32_t start_frame,
                           std::uint32_t end_frame,
                           std::uint32_t frame_step,
                           double frames_per_second,
                           std::uint32_t render_width,
                           std::uint32_t render_height,
                           bool close_when_complete,
                           const std::optional<std::filesystem::path>& timing_csv = std::nullopt,
                           const std::optional<std::filesystem::path>& video_path = std::nullopt,
                           const std::optional<std::filesystem::path>& ffmpeg_path = std::nullopt)
{
    if (!session.asset()) {
        sequence.status = "No VBT asset loaded";
        return false;
    }
    const std::uint32_t last_frame = session.timeline().frame_count() - 1u;
    start_frame = std::min(start_frame, last_frame);
    end_frame = std::clamp(end_frame, start_frame, last_frame);
    frame_step = std::max(1u, frame_step);
    std::error_code error;
    std::filesystem::create_directories(output_directory, error);
    if (error) {
        sequence.status = "Unable to create sequence output directory: " + error.message();
        return false;
    }

    sequence = {};
    sequence.active = true;
    sequence.close_when_complete = close_when_complete;
    sequence.output_directory = output_directory;
    sequence.timing_csv = timing_csv.value_or(output_directory / "gpu_timings.csv");
    sequence.video_path = video_path.value_or(std::filesystem::path{});
    sequence.ffmpeg_path = ffmpeg_path.value_or(std::filesystem::path{});
    sequence.file_prefix = session.asset()->path.stem().string();
    sequence.start_frame = start_frame;
    sequence.end_frame = end_frame;
    sequence.frame_step = frame_step;
    sequence.next_frame = start_frame;
    sequence.total = (end_frame - start_frame) / frame_step + 1u;
    sequence.render_width = std::max(1u, render_width);
    sequence.render_height = std::max(1u, render_height);
    sequence.frames_per_second = std::clamp(frames_per_second, 1.0, 240.0);
    sequence.status = "Preparing sequence";
    session.timeline().pause();
    session.timeline().seek(start_frame);
    return true;
}

std::string csv_field(const std::string& value)
{
    std::string escaped = value;
    std::size_t position = 0;
    while ((position = escaped.find('"', position)) != std::string::npos) {
        escaped.insert(position, 1, '"');
        position += 2;
    }
    return '"' + escaped + '"';
}

bool write_sequence_csv(SequenceExportState& sequence)
{
    std::sort(sequence.rows.begin(), sequence.rows.end(), [](const auto& left, const auto& right) {
        return left.frame < right.frame;
    });
    std::ofstream output(sequence.timing_csv);
    if (!output) return false;
    output << "frame,file,width,height,success,fields,rays_regenerated,ray_ms,sample_ms,colorize_ms,total_ms,gpu_fps,error\n";
    output << std::fixed << std::setprecision(6);
    for (const auto& row : sequence.rows) {
        const double gpu_fps = row.timings.total_milliseconds > 0.0
                                   ? 1000.0 / row.timings.total_milliseconds
                                   : 0.0;
        output << row.frame << ',' << csv_field(row.path.filename().string()) << ',' << row.width << ','
               << row.height << ',' << (row.success ? 1 : 0) << ',' << row.timings.field_count << ','
               << (row.timings.rays_regenerated ? 1 : 0) << ',' << row.timings.ray_milliseconds << ','
               << row.timings.sample_milliseconds << ',' << row.timings.colorize_milliseconds << ','
               << row.timings.total_milliseconds << ',' << gpu_fps << ',' << csv_field(row.error) << '\n';
    }
    return static_cast<bool>(output);
}

std::filesystem::path write_ffconcat_manifest(const SequenceExportState& sequence)
{
    const std::filesystem::path manifest = sequence.output_directory / "sequence.ffconcat";
    std::ofstream output(manifest);
    if (!output) return {};
    output << "ffconcat version 1.0\n";
    const double duration = 1.0 / sequence.frames_per_second;
    for (const auto& row : sequence.rows) {
        if (!row.success) continue;
        std::string value = row.path.filename().generic_string();
        std::replace(value.begin(), value.end(), '\'', '_');
        output << "file '" << value << "'\n";
        output << std::fixed << std::setprecision(9) << "duration " << duration << '\n';
    }
    return static_cast<bool>(output) ? manifest : std::filesystem::path{};
}

#ifdef _WIN32
std::optional<std::filesystem::path> resolve_ffmpeg(const std::filesystem::path& requested)
{
    if (!requested.empty() && std::filesystem::is_regular_file(requested)) return requested;
    std::array<wchar_t, 32768> configured{};
    const DWORD configured_length = GetEnvironmentVariableW(
        L"IMAGEIO_FFMPEG_EXE", configured.data(), static_cast<DWORD>(configured.size()));
    if (configured_length > 0 && configured_length < configured.size() &&
        std::filesystem::is_regular_file(configured.data())) {
        return std::filesystem::path(configured.data());
    }
    std::array<wchar_t, 32768> buffer{};
    const DWORD length = SearchPathW(nullptr, L"ffmpeg.exe", nullptr, static_cast<DWORD>(buffer.size()),
                                     buffer.data(), nullptr);
    if (length > 0 && length < buffer.size()) return std::filesystem::path(buffer.data());

    std::array<wchar_t, 32768> appdata{};
    const DWORD appdata_length =
        GetEnvironmentVariableW(L"APPDATA", appdata.data(), static_cast<DWORD>(appdata.size()));
    if (appdata_length > 0 && appdata_length < appdata.size()) {
        const std::filesystem::path python_root = std::filesystem::path(appdata.data()) / "Python";
        std::error_code error;
        for (const auto& version : std::filesystem::directory_iterator(python_root, error)) {
            if (error || !version.is_directory()) continue;
            const std::filesystem::path binaries =
                version.path() / "site-packages" / "imageio_ffmpeg" / "binaries";
            std::error_code binaries_error;
            for (const auto& candidate : std::filesystem::directory_iterator(binaries, binaries_error)) {
                if (binaries_error || !candidate.is_regular_file()) continue;
                const std::wstring name = candidate.path().filename().wstring();
                if (name.starts_with(L"ffmpeg") && candidate.path().extension() == L".exe") {
                    return candidate.path();
                }
            }
        }
    }
    return std::nullopt;
}

bool encode_mp4(const std::filesystem::path& ffmpeg,
                const std::filesystem::path& manifest,
                const std::filesystem::path& output_path,
                double frames_per_second)
{
    if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
    std::wostringstream rate;
    rate << std::fixed << std::setprecision(6) << frames_per_second;
    std::wstring command = L"\"" + ffmpeg.wstring() +
                           L"\" -y -hide_banner -loglevel error -f concat -safe 0 -i \"" +
                           manifest.wstring() +
                           L"\" -vf \"setpts=N/(" + rate.str() + L"*TB)\" -r " + rate.str() +
                           L" -fps_mode cfr -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p "
                           L"-movflags +faststart \"" + output_path.wstring() + L"\"";
    std::vector<wchar_t> command_line(command.begin(), command.end());
    command_line.push_back(L'\0');
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (CreateProcessW(nullptr,
                       command_line.data(),
                       nullptr,
                       nullptr,
                       FALSE,
                       CREATE_NO_WINDOW,
                       nullptr,
                       nullptr,
                       &startup,
                       &process) == FALSE) {
        return false;
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return exit_code == 0;
}
#else
std::optional<std::filesystem::path> resolve_ffmpeg(const std::filesystem::path&) { return std::nullopt; }
bool encode_mp4(const std::filesystem::path&,
                const std::filesystem::path&,
                const std::filesystem::path&,
                double)
{
    return false;
}
#endif

void finish_sequence_export(SequenceExportState& sequence)
{
    const bool csv_written = write_sequence_csv(sequence);
    const std::filesystem::path manifest = write_ffconcat_manifest(sequence);
    const auto failed = std::count_if(sequence.rows.begin(), sequence.rows.end(), [](const auto& row) {
        return !row.success;
    });

    bool video_written = false;
    bool ffmpeg_missing = false;
    if (!sequence.video_path.empty() && !manifest.empty() && failed == 0) {
        const auto ffmpeg = resolve_ffmpeg(sequence.ffmpeg_path);
        if (ffmpeg) {
            video_written =
                encode_mp4(*ffmpeg, manifest, sequence.video_path, sequence.frames_per_second);
        }
        else {
            ffmpeg_missing = true;
        }
    }

    sequence.active = false;
    if (failed > 0) {
        sequence.status = "Sequence completed with " + std::to_string(failed) + " failed frame(s)";
    }
    else if (!csv_written || manifest.empty()) {
        sequence.status = "Frames exported, but report generation failed";
    }
    else if (!sequence.video_path.empty() && ffmpeg_missing) {
        sequence.status = "PNG sequence complete; FFmpeg was not found";
    }
    else if (!sequence.video_path.empty() && !video_written) {
        sequence.status = "PNG sequence complete; MP4 encoding failed";
    }
    else if (video_written) {
        sequence.status = "PNG sequence, timing CSV, and MP4 complete";
    }
    else {
        sequence.status = "PNG sequence and timing CSV complete";
    }
}

void record_sequence_result(
    SequenceExportState& sequence,
    const vbtstudio::frontend::VolumeRenderer::PngExportResult& export_result,
    const vbtstudio::frontend::VolumeRenderer::GpuTimings& timings)
{
    SequenceTimingRow row{};
    row.path = export_result.path;
    row.frame = export_result.frame;
    row.width = export_result.width;
    row.height = export_result.height;
    row.success = export_result.success;
    row.error = export_result.error;
    if (timings.valid && timings.frame == export_result.frame) row.timings = timings;
    sequence.rows.push_back(std::move(row));
    ++sequence.completed;
    sequence.status = "Exporting " + std::to_string(sequence.completed) + " / " +
                      std::to_string(sequence.total);
}

void advance_sequence_after_schedule(SequenceExportState& sequence)
{
    ++sequence.scheduled;
    if (sequence.frame_step > sequence.end_frame - sequence.next_frame) {
        sequence.scheduling_complete = true;
        return;
    }
    sequence.next_frame += sequence.frame_step;
}

bool sequence_ready_to_finish(const SequenceExportState& sequence)
{
    return sequence.active && sequence.scheduling_complete && sequence.scheduled == sequence.total &&
           sequence.completed == sequence.scheduled;
}

std::string size_text(std::uint64_t bytes)
{
    constexpr double mebibyte = 1024.0 * 1024.0;
    constexpr double gibibyte = 1024.0 * mebibyte;
    char text[64]{};
    if (bytes >= static_cast<std::uint64_t>(gibibyte)) {
        std::snprintf(text, sizeof(text), "%.2f GiB", static_cast<double>(bytes) / gibibyte);
    }
    else {
        std::snprintf(text, sizeof(text), "%.2f MiB", static_cast<double>(bytes) / mebibyte);
    }
    return text;
}

void build_default_dock_layout(ImGuiID dockspace_id, const ImVec2& size)
{
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, size);
    ImGuiID main_id = dockspace_id;
    const ImGuiID left_id = ImGui::DockBuilderSplitNode(main_id, ImGuiDir_Left, 0.20f, nullptr, &main_id);
    const ImGuiID right_id = ImGui::DockBuilderSplitNode(main_id, ImGuiDir_Right, 0.24f, nullptr, &main_id);
    const ImGuiID bottom_id = ImGui::DockBuilderSplitNode(main_id, ImGuiDir_Down, 0.19f, nullptr, &main_id);
    ImGui::DockBuilderDockWindow("Assets", left_id);
    ImGui::DockBuilderDockWindow("Inspector", right_id);
    ImGui::DockBuilderDockWindow("Timeline", bottom_id);
    ImGui::DockBuilderDockWindow("Viewport", main_id);
    ImGui::DockBuilderFinish(dockspace_id);
}

void draw_viewport(vbtstudio::backend::StudioSession& session,
                   vbtstudio::frontend::VolumeRenderer& volume_renderer,
                   const SequenceExportState& sequence)
{
    ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 available = ImGui::GetContentRegionAvail();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImU32 background = IM_COL32(18, 18, 17, 255);
    draw->AddRectFilled(origin, ImVec2(origin.x + available.x, origin.y + available.y), background);

    if (session.asset() && available.x > 1.0f && available.y > 1.0f) {
        if (sequence.active) {
            volume_renderer.ensure_viewport(sequence.render_width, sequence.render_height);
        }
        else {
            const float render_scale = std::min({1.0f, 960.0f / available.x, 720.0f / available.y});
            const auto render_width =
                static_cast<std::uint32_t>(std::max(1.0f, std::floor(available.x * render_scale)));
            const auto render_height =
                static_cast<std::uint32_t>(std::max(1.0f, std::floor(available.y * render_scale)));
            volume_renderer.ensure_viewport(render_width, render_height);
        }
    }

    if (volume_renderer.ready()) {
        ImGui::Image(volume_renderer.texture_id(), available);
        if (ImGui::IsItemHovered() && !sequence.active) {
            const ImGuiIO& io = ImGui::GetIO();
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                vbtstudio::backend::orbit_camera(
                    session.camera(), -io.MouseDelta.x * 0.005f, -io.MouseDelta.y * 0.005f);
            }
            if (io.MouseWheel != 0.0f) vbtstudio::backend::zoom_camera(session.camera(), io.MouseWheel);
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                vbtstudio::backend::reset_camera(session.camera());
            }
        }
        const auto& asset = *session.asset();
        const std::string label = vbtstudio::backend::field_role_name(asset.role) + "  " +
                                  std::to_string(asset.width) + " x " + std::to_string(asset.height) + " x " +
                                  std::to_string(asset.depth);
        draw->AddRectFilled(origin, ImVec2(origin.x + 280.0f, origin.y + 58.0f), IM_COL32(12, 12, 11, 190));
        draw->AddText(ImVec2(origin.x + 16.0f, origin.y + 12.0f), IM_COL32(232, 232, 226, 255), label.c_str());
        const std::string frame = "Frame " + std::to_string(session.timeline().frame()) + " / " +
                                  std::to_string(asset.frames - 1u) + "  |  " +
                                  std::to_string(volume_renderer.render_width()) + " x " +
                                  std::to_string(volume_renderer.render_height());
        draw->AddText(ImVec2(origin.x + 16.0f, origin.y + 34.0f), IM_COL32(106, 199, 204, 255), frame.c_str());
        ImGui::End();
        return;
    }

    const float grid = 48.0f;
    const ImU32 grid_color = IM_COL32(43, 43, 40, 160);
    for (float x = origin.x; x < origin.x + available.x; x += grid) {
        draw->AddLine(ImVec2(x, origin.y), ImVec2(x, origin.y + available.y), grid_color);
    }
    for (float y = origin.y; y < origin.y + available.y; y += grid) {
        draw->AddLine(ImVec2(origin.x, y), ImVec2(origin.x + available.x, y), grid_color);
    }

    if (session.asset()) {
        const auto& asset = *session.asset();
        const float max_dimension = static_cast<float>(std::max({asset.width, asset.height, asset.depth}));
        const float width = std::min(available.x * 0.46f, 420.0f) * static_cast<float>(asset.width) / max_dimension;
        const float height = std::min(available.y * 0.52f, 360.0f) * static_cast<float>(asset.depth) / max_dimension;
        const ImVec2 center(origin.x + available.x * 0.5f, origin.y + available.y * 0.5f);
        const ImVec2 minimum(center.x - width * 0.5f, center.y - height * 0.5f);
        const ImVec2 maximum(center.x + width * 0.5f, center.y + height * 0.5f);
        const ImVec2 offset(24.0f, -18.0f);
        const ImU32 bounds = IM_COL32(241, 157, 45, 230);
        const ImU32 back_bounds = IM_COL32(72, 178, 184, 170);
        draw->AddRect(minimum, maximum, bounds, 0.0f, 0, 2.0f);
        draw->AddRect(ImVec2(minimum.x + offset.x, minimum.y + offset.y),
                      ImVec2(maximum.x + offset.x, maximum.y + offset.y), back_bounds, 0.0f, 0, 1.5f);
        for (const ImVec2 corner : std::array<ImVec2, 4>{
                 minimum,
                 ImVec2(maximum.x, minimum.y),
                 maximum,
                 ImVec2(minimum.x, maximum.y),
             }) {
            draw->AddLine(corner, ImVec2(corner.x + offset.x, corner.y + offset.y), back_bounds, 1.5f);
        }

        const std::string label = vbtstudio::backend::field_role_name(asset.role) + "  " +
                                  std::to_string(asset.width) + " x " + std::to_string(asset.height) + " x " +
                                  std::to_string(asset.depth);
        draw->AddText(ImVec2(origin.x + 16.0f, origin.y + 14.0f), IM_COL32(230, 230, 224, 255), label.c_str());
        const std::string frame = "Frame " + std::to_string(session.timeline().frame()) + " / " +
                                  std::to_string(asset.frames - 1u);
        draw->AddText(ImVec2(origin.x + 16.0f, origin.y + 36.0f), IM_COL32(106, 199, 204, 255), frame.c_str());
    }
    ImGui::Dummy(available);
    ImGui::End();
}

void draw_assets(GLFWwindow* window,
                 vbtstudio::backend::StudioSession& session,
                 const vbtstudio::frontend::VolumeRenderer& volume_renderer,
                 const SequenceExportState& sequence)
{
    ImGui::Begin("Assets");
    ImGui::BeginDisabled(sequence.active);
    if (ImGui::Button("Open VBT...", ImVec2(-1.0f, 0.0f))) {
        if (const auto path = open_vbt_dialog(window)) session.open_asset(*path);
    }
    ImGui::BeginDisabled(!session.asset());
    if (ImGui::Button("Add Field...", ImVec2(-1.0f, 0.0f))) {
        if (const auto path = open_vbt_dialog(window)) session.add_field(*path);
    }
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    ImGui::Separator();
    if (session.asset()) {
        const auto& asset = *session.asset();
        ImGui::TextWrapped("%s", asset.path.filename().string().c_str());
        ImGui::Spacing();
        ImGui::TextDisabled("FIELD");
        ImGui::Text("%s", vbtstudio::backend::field_role_name(asset.role).c_str());
        ImGui::TextDisabled("DIMENSIONS");
        ImGui::Text("%u x %u x %u", asset.width, asset.height, asset.depth);
        ImGui::TextDisabled("FRAMES");
        ImGui::Text("%u", asset.frames);
        ImGui::TextDisabled("FILE SIZE");
        ImGui::Text("%s", size_text(asset.file_bytes).c_str());
        ImGui::TextDisabled("PAYLOAD");
        ImGui::Text("%s", size_text(asset.payload_bytes).c_str());
        ImGui::TextDisabled("LEAVES");
        ImGui::Text("%u", asset.leaf_count);
        ImGui::TextDisabled("BBOX");
        ImGui::Text("[%d %d %d] - [%d %d %d]",
                    asset.bbox_min[0], asset.bbox_min[1], asset.bbox_min[2],
                    asset.bbox_max[0], asset.bbox_max[1], asset.bbox_max[2]);
        if (asset.role == vbtstudio::backend::FieldRole::LevelSet) {
            ImGui::TextDisabled("VOXEL SIZE");
            ImGui::Text("%.6f", asset.voxel_size);
            ImGui::TextDisabled("SDF BACKGROUND");
            ImGui::Text("%.6f", asset.background_value);
        }
        if (session.secondary_field()) {
            const auto& field = *session.secondary_field();
            ImGui::SeparatorText("ADDITIONAL FIELD");
            ImGui::TextWrapped("%s", field.path.filename().string().c_str());
            ImGui::Text("%s | %s", vbtstudio::backend::field_role_name(field.role).c_str(),
                        size_text(field.file_bytes).c_str());
            ImGui::Text("[%d %d %d] - [%d %d %d]",
                        field.bbox_min[0], field.bbox_min[1], field.bbox_min[2],
                        field.bbox_max[0], field.bbox_max[1], field.bbox_max[2]);
        }
        if (session.temperature_field()) {
            const auto& field = *session.temperature_field();
            ImGui::SeparatorText("TEMPERATURE FIELD");
            ImGui::TextWrapped("%s", field.path.filename().string().c_str());
            ImGui::Text("%s | %s", vbtstudio::backend::field_role_name(field.role).c_str(),
                        size_text(field.file_bytes).c_str());
            ImGui::Text("[%d %d %d] - [%d %d %d]",
                        field.bbox_min[0], field.bbox_min[1], field.bbox_min[2],
                        field.bbox_max[0], field.bbox_max[1], field.bbox_max[2]);
        }
        ImGui::TextDisabled("GPU");
        ImGui::TextWrapped("%s", volume_renderer.status().c_str());
        if (volume_renderer.upload_milliseconds() > 0.0) {
            ImGui::Text("%.1f ms resident upload", volume_renderer.upload_milliseconds());
        }
    }
    else {
        ImGui::TextDisabled("No asset loaded");
    }
    if (!session.last_error().empty()) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.38f, 0.28f, 1.0f));
        ImGui::TextWrapped("%s", session.last_error().c_str());
        ImGui::PopStyleColor();
    }
    ImGui::End();
}

void draw_inspector(GLFWwindow* window,
                    vbtstudio::backend::StudioSession& session,
                    vbtstudio::frontend::VolumeRenderer& volume_renderer,
                    SequenceExportState& sequence)
{
    ImGui::Begin("Inspector");
    auto& material = session.material();
    if (ImGui::BeginTabBar("InspectorTabs")) {
        if (ImGui::BeginTabItem("Material")) {
            ImGui::BeginDisabled(sequence.active);
            const bool level_set = session.asset() &&
                                   session.asset()->role == vbtstudio::backend::FieldRole::LevelSet;
            const char* volume_presets[] = {
                "paper_gray", "charcoal", "soft_ash", "cool_steel", "fire_warm", "fire_physical"};
            const char* water_presets[] = {"water_studio", "water_clear", "water_blue", "water_silver"};
            const char* const* presets = level_set ? water_presets : volume_presets;
            const int preset_count = level_set ? static_cast<int>(std::size(water_presets))
                                               : static_cast<int>(std::size(volume_presets));
            int selected = 0;
            for (int index = 0; index < preset_count; ++index) {
                if (material.preset == presets[index]) selected = index;
            }
            if (ImGui::Combo("Preset", &selected, presets, preset_count)) {
                vbtstudio::backend::apply_material_preset(material, presets[selected]);
            }
            if (level_set) {
                int surface_model = static_cast<int>(material.surface_model);
                if (ImGui::Combo("Surface model", &surface_model, "Fast Surface\0Physical Water\0")) {
                    material.surface_model = static_cast<std::uint32_t>(surface_model);
                }
                ImGui::ColorEdit3("Surface", material.surface_color.data());
                ImGui::SliderFloat("Roughness", &material.surface_roughness, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Opacity", &material.surface_opacity, 0.0f, 1.0f, "%.2f");
                if (material.surface_model != 0u) {
                    ImGui::ColorEdit3("Absorption", material.absorption_color.data());
                    ImGui::SliderFloat("IOR", &material.water_ior, 1.01f, 2.0f, "%.3f");
                    ImGui::SliderFloat("Absorption density", &material.absorption_density, 0.0f, 0.25f, "%.3f");
                    ImGui::SliderFloat("Reflection", &material.reflection_strength, 0.0f, 2.0f, "%.2f");
                    ImGui::SliderFloat("Environment", &material.environment_strength, 0.0f, 3.0f, "%.2f");
                    ImGui::SliderFloat("Floor offset", &material.floor_offset, 0.0f, 0.5f, "%.3f extent");
                    ImGui::SliderFloat("Shadow", &material.shadow_strength, 0.0f, 1.0f, "%.2f");
                    int shadow_steps = static_cast<int>(material.shadow_steps);
                    if (ImGui::SliderInt("Shadow steps", &shadow_steps, 8, 128)) {
                        material.shadow_steps = static_cast<std::uint32_t>(shadow_steps);
                    }
                }
                else {
                    ImGui::SliderFloat("Metallic", &material.surface_metallic, 0.0f, 1.0f, "%.2f");
                }
                ImGui::SliderFloat("Iso value", &material.surface_iso, -1.0f, 1.0f, "%.4f");
                ImGui::SliderFloat("Hit epsilon", &material.surface_epsilon_voxels, 0.02f, 0.5f, "%.3f vox");
                ImGui::SliderFloat("Normal step", &material.surface_normal_step, 0.25f, 2.0f, "%.2f vox");
            }
            else {
                int volume_model = static_cast<int>(material.volume_model);
                if (ImGui::Combo("Volume model", &volume_model, "Fast Volume\0Physical Fire\0")) {
                    material.volume_model = static_cast<std::uint32_t>(volume_model);
                }
                ImGui::SliderFloat("Density", &material.density_scale, 0.0f, 128.0f, "%.2f");
                ImGui::SliderFloat("Threshold", &material.density_threshold, 0.0f, 1.0f, "%.4f");
                ImGui::SliderFloat("Gamma", &material.density_gamma, 0.1f, 3.0f, "%.2f");
                ImGui::SliderFloat("Anisotropy", &material.anisotropy, -0.9f, 0.9f, "%.2f");
                ImGui::ColorEdit3("Smoke", material.smoke_color.data());
                ImGui::SeparatorText("Fire");
                ImGui::SliderFloat("Strength", &material.flame_strength, 0.0f, 8.0f, "%.2f");
                ImGui::SliderFloat("Flame threshold", &material.flame_threshold, 0.0f, 2.0f, "%.3f");
                ImGui::SliderFloat("Temperature low", &material.temperature_min, 0.0f, 10000.0f, "%.0f");
                ImGui::SliderFloat("Temperature high", &material.temperature_max, 1000.0f, 25000.0f, "%.0f");
                if (material.volume_model != 0u) {
                    ImGui::ColorEdit3("Fire tint", material.fire_tint.data());
                    ImGui::SliderFloat("Scattering", &material.fire_scattering, 0.0f, 2.0f, "%.2f");
                    ImGui::SliderFloat("Blackbody mix", &material.fire_blackbody_mix, 0.0f, 1.0f, "%.2f");
                    ImGui::SliderFloat("Emission glow", &material.fire_glow, 0.0f, 2.0f, "%.2f");
                }
            }
            ImGui::ColorEdit3("Background", material.background_color.data());
            ImGui::SliderFloat("Exposure", &material.exposure, -5.0f, 5.0f, "%.2f EV");
            ImGui::EndDisabled();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Render")) {
            ImGui::BeginDisabled(sequence.active);
            int quality = material.sample_steps <= 160u ? 0 : (material.sample_steps <= 320u ? 1 : 2);
            if (ImGui::Combo("Quality", &quality, "Interactive\0Balanced\0Final Vulkan\0")) {
                material.sample_steps = std::array<std::uint32_t, 3>{160u, 320u, 512u}[quality];
                if (material.surface_model != 0u) {
                    material.shadow_steps = std::array<std::uint32_t, 3>{24u, 48u, 80u}[quality];
                }
            }
            int steps = static_cast<int>(material.sample_steps);
            if (ImGui::SliderInt("Steps", &steps, 32, 1024)) {
                material.sample_steps = static_cast<std::uint32_t>(steps);
            }
            ImGui::EndDisabled();
            const auto cache_stats = volume_renderer.cache_stats();
            if (cache_stats.total_leaves > 0u) {
                ImGui::SeparatorText("Frame Cache");
                ImGui::Text("Active leaves  %u / %u",
                            cache_stats.active_leaves,
                            cache_stats.total_leaves);
                ImGui::Text("Controls       %s",
                            size_text(cache_stats.cache_bytes).c_str());
                ImGui::Text("Leaf mapping   %s",
                            size_text(cache_stats.mapping_bytes).c_str());
                if (cache_stats.fixed_cache_bytes > 0u) {
                    const double saved = 100.0 *
                        (1.0 - static_cast<double>(cache_stats.cache_bytes) /
                                   static_cast<double>(cache_stats.fixed_cache_bytes));
                    ImGui::Text("VRAM saved     %.1f%%", saved);
                }
            }
            ImGui::SeparatorText("GPU Timing");
            const auto& timings = volume_renderer.gpu_timings();
            if (!timings.supported) {
                ImGui::TextDisabled("Timestamp queries unavailable");
            }
            else if (!timings.valid) {
                ImGui::TextDisabled("Waiting for first completed render");
            }
            else {
                ImGui::Text("Frame %u | %u field%s",
                            timings.frame,
                            timings.field_count,
                            timings.field_count == 1u ? "" : "s");
                ImGui::Text("Total       %7.3f ms", timings.total_milliseconds);
                ImGui::Text("Sampling    %7.3f ms", timings.sample_milliseconds);
                ImGui::Text("Colorize    %7.3f ms", timings.colorize_milliseconds);
                if (timings.rays_regenerated) {
                    ImGui::Text("Ray build   %7.3f ms", timings.ray_milliseconds);
                }
                else {
                    ImGui::TextDisabled("Ray build   cached");
                }
                if (timings.total_milliseconds > 0.0) {
                    ImGui::Text("Throughput  %7.1f FPS", 1000.0 / timings.total_milliseconds);
                }
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Camera")) {
            ImGui::BeginDisabled(sequence.active);
            auto& camera = session.camera();
            ImGui::SliderAngle("Yaw", &camera.yaw, -180.0f, 180.0f);
            ImGui::SliderAngle("Pitch", &camera.pitch, -83.0f, 83.0f);
            ImGui::SliderFloat("Distance", &camera.distance, 0.75f, 6.0f, "%.2f x");
            ImGui::SliderFloat("Field of view", &camera.field_of_view, 15.0f, 90.0f, "%.1f deg");
            int up_axis = camera.up_axis == 2u ? 1 : 0;
            if (ImGui::Combo("Up axis", &up_axis, "Y Up\0Z Up\0")) {
                camera.up_axis = up_axis == 1 ? 2u : 1u;
            }
            ImGui::DragFloat3("Target offset", camera.target_offset.data(), 0.002f, -1.0f, 1.0f, "%.3f");
            if (session.asset() && session.asset()->role == vbtstudio::backend::FieldRole::LevelSet &&
                ImGui::Button("Water Hero", ImVec2(-1.0f, 0.0f))) {
                camera.yaw = -0.62f;
                camera.pitch = 0.16f;
                camera.distance = 1.25f;
                camera.field_of_view = 34.0f;
                camera.target_offset = {0.0f, 0.0f, 0.0f};
                camera.up_axis = 1u;
            }
            if (session.asset() && session.asset()->role != vbtstudio::backend::FieldRole::LevelSet &&
                ImGui::Button("Fire Cycles", ImVec2(-1.0f, 0.0f))) {
                camera.yaw = 2.656f;
                camera.pitch = 0.080f;
                camera.distance = 1.56f;
                camera.field_of_view = 38.0f;
                camera.target_offset = {0.0f, 0.0f, 0.025f};
                camera.up_axis = 2u;
            }
            if (ImGui::Button("Fit Volume", ImVec2(-1.0f, 0.0f))) {
                vbtstudio::backend::reset_camera(camera);
            }
            ImGui::EndDisabled();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Export")) {
            ImGui::TextDisabled("FORMAT");
            ImGui::Text("PNG RGBA 8-bit");
            ImGui::TextDisabled("CURRENT RESOLUTION");
            ImGui::Text("%u x %u", volume_renderer.render_width(), volume_renderer.render_height());
            ImGui::BeginDisabled(!volume_renderer.ready() || sequence.active);
            if (ImGui::Button("Export Current Frame", ImVec2(-1.0f, 0.0f))) {
                const std::string name = session.asset()->path.stem().string() + "_frame" +
                                         std::to_string(session.timeline().frame()) + ".png";
                if (const auto path = save_png_dialog(window, name)) volume_renderer.export_png(*path);
            }
            ImGui::EndDisabled();
            ImGui::SeparatorText("Sequence");
            static int sequence_start = 0;
            static int sequence_end = 0;
            static int sequence_step = 1;
            static float sequence_fps = 24.0f;
            static std::uint32_t configured_frame_count = 0;
            if (session.asset()) {
                const int last_frame = static_cast<int>(session.timeline().frame_count() - 1u);
                if (configured_frame_count != session.timeline().frame_count()) {
                    sequence_start = 0;
                    sequence_end = last_frame;
                    configured_frame_count = session.timeline().frame_count();
                }
                sequence_start = std::clamp(sequence_start, 0, last_frame);
                sequence_end = std::clamp(sequence_end, sequence_start, last_frame);
            }
            ImGui::InputInt("Start frame", &sequence_start);
            ImGui::InputInt("End frame", &sequence_end);
            ImGui::InputInt("Frame step", &sequence_step);
            ImGui::DragFloat("Sequence FPS", &sequence_fps, 0.25f, 1.0f, 240.0f, "%.2f");
            sequence_step = std::max(1, sequence_step);
            ImGui::BeginDisabled(!session.asset() || sequence.active);
            if (ImGui::Button("Export PNG Sequence", ImVec2(-1.0f, 0.0f))) {
                if (const auto directory = select_folder_dialog(window)) {
                    begin_sequence_export(sequence,
                                          session,
                                          *directory,
                                          static_cast<std::uint32_t>(std::max(0, sequence_start)),
                                          static_cast<std::uint32_t>(std::max(sequence_start, sequence_end)),
                                          static_cast<std::uint32_t>(sequence_step),
                                          sequence_fps,
                                          volume_renderer.render_width() > 0 ? volume_renderer.render_width() : 960u,
                                          volume_renderer.render_height() > 0 ? volume_renderer.render_height() : 550u,
                                          false);
                }
            }
            if (ImGui::Button("Export MP4 Sequence", ImVec2(-1.0f, 0.0f))) {
                const std::string suggested = session.asset()->path.stem().string() + ".mp4";
                if (const auto video = save_mp4_dialog(window, suggested)) {
                    const std::filesystem::path directory =
                        video->parent_path() / (video->stem().string() + "_frames");
                    begin_sequence_export(sequence,
                                          session,
                                          directory,
                                          static_cast<std::uint32_t>(std::max(0, sequence_start)),
                                          static_cast<std::uint32_t>(std::max(sequence_start, sequence_end)),
                                          static_cast<std::uint32_t>(sequence_step),
                                          sequence_fps,
                                          volume_renderer.render_width() > 0 ? volume_renderer.render_width() : 960u,
                                          volume_renderer.render_height() > 0 ? volume_renderer.render_height() : 550u,
                                          false,
                                          std::nullopt,
                                          *video);
                }
            }
            ImGui::EndDisabled();
            if (sequence.active) {
                const float progress = sequence.total > 0
                                           ? static_cast<float>(sequence.completed) /
                                                 static_cast<float>(sequence.total)
                                           : 0.0f;
                const std::string overlay = std::to_string(sequence.completed) + " / " +
                                            std::to_string(sequence.total);
                ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), overlay.c_str());
            }
            if (!sequence.status.empty()) ImGui::TextWrapped("%s", sequence.status.c_str());
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void draw_timeline(vbtstudio::backend::StudioSession& session, const SequenceExportState& sequence)
{
    ImGui::Begin("Timeline", nullptr, ImGuiWindowFlags_NoScrollbar);
    auto& timeline = session.timeline();
    const bool enabled = session.asset().has_value() && !sequence.active;
    ImGui::BeginDisabled(!enabled);
    if (ImGui::Button("|<")) timeline.seek(0);
    ImGui::SameLine();
    if (ImGui::Button("<")) timeline.step(-1);
    ImGui::SameLine();
    if (ImGui::Button(timeline.playing() ? "Pause" : "Play")) timeline.toggle_playback();
    ImGui::SameLine();
    if (ImGui::Button(">")) timeline.step(1);
    ImGui::SameLine();
    if (ImGui::Button(">|")) timeline.seek(timeline.frame_count() - 1u);
    ImGui::SameLine();
    int frame = static_cast<int>(timeline.frame());
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::SliderInt("##Frame", &frame, 0, static_cast<int>(timeline.frame_count() - 1u), "Frame %d")) {
        timeline.seek(static_cast<std::uint32_t>(std::max(0, frame)));
    }
    float fps = static_cast<float>(timeline.fps());
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::DragFloat("FPS", &fps, 0.25f, 1.0f, 240.0f, "%.2f")) timeline.set_fps(fps);
    ImGui::SameLine();
    int loop_mode = static_cast<int>(timeline.loop_mode());
    ImGui::SetNextItemWidth(130.0f);
    if (ImGui::Combo("Loop", &loop_mode, "Once\0Loop\0Ping-Pong\0")) {
        timeline.set_loop_mode(static_cast<vbtstudio::backend::LoopMode>(loop_mode));
    }
    ImGui::EndDisabled();
    ImGui::End();
}

void draw_dockspace(GLFWwindow* window,
                    vbtstudio::backend::StudioSession& session,
                    vbtstudio::frontend::VolumeRenderer& volume_renderer,
                    SequenceExportState& sequence)
{
    volume_renderer.sync_asset(session.asset());
    volume_renderer.sync_secondary_field(session.secondary_field());
    volume_renderer.sync_temperature_field(session.temperature_field());
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking |
                                   ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                                   ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("VBTStudioHost", nullptr, flags);
    ImGui::PopStyleVar(3);

    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open VBT...", "Ctrl+O", false, !sequence.active)) {
                if (const auto path = open_vbt_dialog(window)) session.open_asset(*path);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit")) glfwSetWindowShouldClose(window, GLFW_TRUE);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Reset Layout")) {
                ImGui::DockBuilderRemoveNode(ImGui::GetID("VBTStudioDockspace"));
            }
            ImGui::EndMenu();
        }
        ImGui::SameLine(ImGui::GetWindowWidth() - 250.0f);
        ImGui::TextColored(ImVec4(0.42f, 0.78f, 0.80f, 1.0f), "Vulkan 1.3");
        ImGui::SameLine();
        ImGui::TextDisabled("%.1f FPS", ImGui::GetIO().Framerate);
        ImGui::EndMenuBar();
    }

    const ImGuiID dockspace_id = ImGui::GetID("VBTStudioDockspace");
    const ImVec2 dockspace_size = ImGui::GetContentRegionAvail();
    ImGuiDockNode* dockspace_node = ImGui::DockBuilderGetNode(dockspace_id);
    const bool invalid_saved_size = dockspace_node != nullptr &&
                                    (dockspace_node->Size.x > dockspace_size.x * 1.05f ||
                                     dockspace_node->Size.y > dockspace_size.y * 1.05f);
    if (dockspace_node == nullptr || invalid_saved_size) {
        build_default_dock_layout(dockspace_id, dockspace_size);
    }
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::End();

    draw_assets(window, session, volume_renderer, sequence);
    draw_viewport(session, volume_renderer, sequence);
    draw_inspector(window, session, volume_renderer, sequence);
    draw_timeline(session, sequence);
}

std::optional<std::filesystem::path> command_line_asset(int argc, char** argv)
{
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) == "--asset") return std::filesystem::path(argv[index + 1]);
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> command_line_export(int argc, char** argv)
{
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) == "--export-frame") return std::filesystem::path(argv[index + 1]);
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> command_line_path(int argc,
                                                       char** argv,
                                                       const std::string& option)
{
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) == option) return std::filesystem::path(argv[index + 1]);
    }
    return std::nullopt;
}

std::optional<std::uint32_t> command_line_uint(int argc,
                                               char** argv,
                                               const std::string& option)
{
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) != option) continue;
        try {
            return static_cast<std::uint32_t>(std::stoul(argv[index + 1]));
        }
        catch (...) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> command_line_field(int argc, char** argv)
{
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) == "--field") return std::filesystem::path(argv[index + 1]);
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> command_line_temperature(int argc, char** argv)
{
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) == "--temperature") return std::filesystem::path(argv[index + 1]);
    }
    return std::nullopt;
}

std::optional<std::uint32_t> command_line_frame(int argc, char** argv)
{
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) != "--frame") continue;
        try {
            return static_cast<std::uint32_t>(std::stoul(argv[index + 1]));
        }
        catch (...) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<float> command_line_float(int argc, char** argv, const std::string& option)
{
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) != option) continue;
        try {
            return std::stof(argv[index + 1]);
        }
        catch (...) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<std::string> command_line_preset(int argc, char** argv)
{
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) == "--preset") return std::string(argv[index + 1]);
    }
    return std::nullopt;
}

std::optional<std::string> command_line_string(int argc, char** argv, const std::string& option)
{
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) == option) return std::string(argv[index + 1]);
    }
    return std::nullopt;
}

} // namespace

int main(int argc, char** argv)
{
    if (!glfwInit()) return 1;
    if (!glfwVulkanSupported()) return 2;
    const auto command_sequence_directory = command_line_path(argc, argv, "--export-sequence");
    const auto command_video = command_line_path(argc, argv, "--export-video");
    const auto command_single_export = command_line_export(argc, argv);
    const bool batch_mode = command_single_export.has_value() || command_sequence_directory.has_value() ||
                            command_video.has_value();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_FALSE);
    if (batch_mode) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    const WindowPlacement placement = initial_window_placement();
    GLFWwindow* window = glfwCreateWindow(placement.width, placement.height, "VBT Studio", nullptr, nullptr);
    if (!window) return 3;
    glfwSetWindowPos(window, placement.x, placement.y);

    std::uint32_t extension_count = 0;
    const char** extensions = glfwGetRequiredInstanceExtensions(&extension_count);
    setup_vulkan(extensions, extension_count);
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    setup_vulkan_window(window, width, height);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = "vbtstudio_layout.ini";
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigWindowsMoveFromTitleBarOnly = true;
    apply_style();

    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.ApiVersion = VK_API_VERSION_1_0;
    init_info.Instance = instance;
    init_info.PhysicalDevice = physical_device;
    init_info.Device = device;
    init_info.QueueFamily = queue_family;
    init_info.Queue = queue;
    init_info.PipelineCache = pipeline_cache;
    init_info.DescriptorPool = descriptor_pool;
    init_info.PipelineInfoMain.RenderPass = main_window.RenderPass;
    init_info.PipelineInfoMain.Subpass = 0;
    init_info.MinImageCount = min_image_count;
    init_info.ImageCount = main_window.ImageCount;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator = allocator;
    init_info.CheckVkResultFn = check_vk_result;
    ImGui_ImplVulkan_Init(&init_info);

    vbtstudio::frontend::VolumeRenderer volume_renderer(physical_device, device, queue, queue_family);

    vbtstudio::backend::StudioSession session;
    if (const auto asset = command_line_asset(argc, argv)) session.open_asset(*asset);
    if (const auto field = command_line_field(argc, argv); field && session.asset()) session.add_field(*field);
    if (const auto field = command_line_temperature(argc, argv); field && session.asset()) session.add_field(*field);
    if (const auto preset = command_line_preset(argc, argv)) {
        vbtstudio::backend::apply_material_preset(session.material(), *preset);
    }
    if (const auto steps = command_line_uint(argc, argv, "--steps")) {
        session.material().sample_steps = std::clamp(*steps, 32u, 1024u);
    }
    if (const auto value = command_line_float(argc, argv, "--density-scale")) {
        session.material().density_scale = std::clamp(*value, 0.0f, 128.0f);
    }
    if (const auto value = command_line_float(argc, argv, "--flame-strength")) {
        session.material().flame_strength = std::clamp(*value, 0.0f, 8.0f);
    }
    if (const auto value = command_line_float(argc, argv, "--flame-threshold")) {
        session.material().flame_threshold = std::clamp(*value, 0.0f, 2.0f);
    }
    if (const auto value = command_line_float(argc, argv, "--temperature-min")) {
        session.material().temperature_min = std::clamp(*value, 0.0f, 25000.0f);
    }
    if (const auto value = command_line_float(argc, argv, "--temperature-max")) {
        session.material().temperature_max = std::clamp(*value, 1.0f, 25000.0f);
    }
    if (const auto value = command_line_float(argc, argv, "--anisotropy")) {
        session.material().anisotropy = std::clamp(*value, -0.9f, 0.9f);
    }
    if (const auto value = command_line_float(argc, argv, "--scattering")) {
        session.material().fire_scattering = std::clamp(*value, 0.0f, 2.0f);
    }
    if (const auto value = command_line_float(argc, argv, "--blackbody-mix")) {
        session.material().fire_blackbody_mix = std::clamp(*value, 0.0f, 1.0f);
    }
    if (const auto value = command_line_float(argc, argv, "--glow")) {
        session.material().fire_glow = std::clamp(*value, 0.0f, 2.0f);
    }
    if (const auto value = command_line_float(argc, argv, "--exposure")) {
        session.material().exposure = std::clamp(*value, -5.0f, 5.0f);
    }
    if (const auto frame = command_line_frame(argc, argv); frame && session.asset()) session.timeline().seek(*frame);
    constexpr float radians_per_degree = 3.1415926535f / 180.0f;
    if (const auto yaw = command_line_float(argc, argv, "--yaw-deg")) session.camera().yaw = *yaw * radians_per_degree;
    if (const auto pitch = command_line_float(argc, argv, "--pitch-deg")) {
        session.camera().pitch = std::clamp(*pitch * radians_per_degree, -1.45f, 1.45f);
    }
    if (const auto distance = command_line_float(argc, argv, "--distance")) {
        session.camera().distance = std::clamp(*distance, 0.75f, 6.0f);
    }
    if (const auto fov = command_line_float(argc, argv, "--fov")) {
        session.camera().field_of_view = std::clamp(*fov, 15.0f, 90.0f);
    }
    if (const auto value = command_line_float(argc, argv, "--target-x")) session.camera().target_offset[0] = *value;
    if (const auto value = command_line_float(argc, argv, "--target-y")) session.camera().target_offset[1] = *value;
    if (const auto value = command_line_float(argc, argv, "--target-z")) session.camera().target_offset[2] = *value;
    if (const auto axis = command_line_string(argc, argv, "--up-axis")) {
        if (*axis == "z" || *axis == "Z") session.camera().up_axis = 2u;
        if (*axis == "y" || *axis == "Y") session.camera().up_axis = 1u;
    }
    SequenceExportState sequence;
    bool batch_start_failed = false;
    if ((command_sequence_directory || command_video) && session.asset()) {
        std::filesystem::path output_directory;
        if (command_sequence_directory) {
            output_directory = *command_sequence_directory;
        }
        else {
            const std::filesystem::path parent = command_video->parent_path().empty()
                                                     ? std::filesystem::current_path()
                                                     : command_video->parent_path();
            output_directory = parent / (command_video->stem().string() + "_frames");
        }
        const std::uint32_t start_frame = command_line_uint(argc, argv, "--start-frame").value_or(0u);
        const std::uint32_t end_frame = command_line_uint(argc, argv, "--end-frame")
                                            .value_or(session.timeline().frame_count() - 1u);
        const std::uint32_t frame_step = command_line_uint(argc, argv, "--frame-step").value_or(1u);
        const double sequence_fps = command_line_float(argc, argv, "--sequence-fps")
                                        .value_or(static_cast<float>(session.timeline().fps()));
        const std::uint32_t render_width = command_line_uint(argc, argv, "--render-width").value_or(960u);
        const std::uint32_t render_height = command_line_uint(argc, argv, "--render-height").value_or(550u);
        batch_start_failed = !begin_sequence_export(sequence,
                                                    session,
                                                    output_directory,
                                                    start_frame,
                                                    end_frame,
                                                    frame_step,
                                                    sequence_fps,
                                                    render_width,
                                                    render_height,
                                                    true,
                                                    command_line_path(argc, argv, "--timing-csv"),
                                                    command_video,
                                                    command_line_path(argc, argv, "--ffmpeg"));
    }
    else if (command_sequence_directory || command_video) {
        batch_start_failed = true;
        sequence.status = "Sequence export requires a valid --asset";
    }
    const auto command_export = sequence.active ? std::optional<std::filesystem::path>{} : command_single_export;
    bool command_export_complete = false;

    auto previous_time = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        int framebuffer_width = 0;
        int framebuffer_height = 0;
        glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
        if (framebuffer_width > 0 && framebuffer_height > 0 &&
            (swapchain_rebuild || main_window.Width != framebuffer_width || main_window.Height != framebuffer_height)) {
            ImGui_ImplVulkan_SetMinImageCount(min_image_count);
            ImGui_ImplVulkanH_CreateOrResizeWindow(
                instance,
                physical_device,
                device,
                &main_window,
                queue_family,
                allocator,
                framebuffer_width,
                framebuffer_height,
                min_image_count,
                0);
            swapchain_rebuild = false;
        }

        const auto current_time = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(current_time - previous_time).count();
        previous_time = current_time;
        if (!sequence.active) session.update(std::min(elapsed, 0.25));
        std::optional<std::filesystem::path> sequence_request;
        if (sequence.active && !sequence.scheduling_complete) {
            session.timeline().seek(sequence.next_frame);
            volume_renderer.invalidate_render();
            sequence_request = sequence_frame_path(sequence, sequence.next_frame);
        }
        volume_renderer.set_render_state(session.timeline().frame(), session.material(), session.camera());

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        draw_dockspace(window, session, volume_renderer, sequence);
        ImGui::Render();
        const FrameRenderResult frame_result =
            frame_render(ImGui::GetDrawData(), volume_renderer, sequence_request);
        frame_present();
        if (frame_result.completed_export && sequence.active) {
            record_sequence_result(sequence, *frame_result.completed_export, frame_result.completed_timings);
        }
        if (frame_result.export_scheduled && sequence.active) {
            advance_sequence_after_schedule(sequence);
        }
        if (sequence_ready_to_finish(sequence)) {
            const bool close_when_complete = sequence.close_when_complete;
            finish_sequence_export(sequence);
            std::printf("VBT Studio sequence: %s\n", sequence.status.c_str());
            std::printf("VBT Studio sequence outputs: frames=%s timing_csv=%s\n",
                        sequence.output_directory.string().c_str(),
                        sequence.timing_csv.string().c_str());
            if (close_when_complete) glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (command_export && !command_export_complete && volume_renderer.ready()) {
            command_export_complete = true;
            const bool exported = volume_renderer.export_png(*command_export);
            volume_renderer.collect_gpu_timings(main_window.FrameIndex);
            std::printf("VBT Studio export: %s\n", exported ? "success" : volume_renderer.status().c_str());
            const auto& timings = volume_renderer.gpu_timings();
            const auto cache_stats = volume_renderer.cache_stats();
            std::printf("VBT Studio frame cache: active_leaves=%u total_leaves=%u "
                        "cache_bytes=%llu fixed_cache_bytes=%llu mapping_bytes=%llu\n",
                        cache_stats.active_leaves,
                        cache_stats.total_leaves,
                        static_cast<unsigned long long>(cache_stats.cache_bytes),
                        static_cast<unsigned long long>(cache_stats.fixed_cache_bytes),
                        static_cast<unsigned long long>(cache_stats.mapping_bytes));
            if (timings.valid) {
                std::printf("VBT Studio GPU timing: frame=%u fields=%u ray_ms=%.6f "
                            "sample_ms=%.6f colorize_ms=%.6f total_ms=%.6f\n",
                            timings.frame,
                            timings.field_count,
                            timings.ray_milliseconds,
                            timings.sample_milliseconds,
                            timings.colorize_milliseconds,
                            timings.total_milliseconds);
            }
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (batch_start_failed) {
            std::fprintf(stderr, "VBT Studio sequence error: %s\n", sequence.status.c_str());
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            batch_start_failed = false;
        }
    }

    check_vk_result(vkDeviceWaitIdle(device));
    if (sequence.active) {
        for (std::uint32_t slot = 0; slot < main_window.ImageCount; ++slot) {
            volume_renderer.collect_gpu_timings(slot);
            if (const auto completed = volume_renderer.collect_png_export(slot)) {
                record_sequence_result(sequence, *completed, volume_renderer.gpu_timings());
            }
        }
        if (sequence_ready_to_finish(sequence)) finish_sequence_export(sequence);
    }
    volume_renderer.shutdown();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    cleanup_vulkan_window();
    cleanup_vulkan();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
