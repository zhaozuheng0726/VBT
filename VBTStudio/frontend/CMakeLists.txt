find_package(Vulkan REQUIRED)
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Do not update fetched dependencies after checkout" FORCE)

find_program(VBTSTUDIO_GLSLANG_VALIDATOR
  NAMES glslangValidator
  HINTS "$ENV{VULKAN_SDK}/Bin"
  REQUIRED
)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG docking
  GIT_SHALLOW TRUE
)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
  FetchContent_Populate(imgui)
endif()

add_library(vbtstudio_imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

target_include_directories(vbtstudio_imgui
  PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(vbtstudio_imgui PUBLIC glfw Vulkan::Vulkan)

set(VBTSTUDIO_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(VBTSTUDIO_VBT_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vbt_sample.comp")
set(VBTSTUDIO_CACHE_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vbt_cache.comp")
set(VBTSTUDIO_ACTIVE_INDEX_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vbt_active_index.comp")
set(VBTSTUDIO_COLOR_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vbt_colorize.comp")
set(VBTSTUDIO_RAY_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ray_generate.comp")
set(VBTSTUDIO_VBT_SPV "${VBTSTUDIO_SHADER_DIR}/vbt_sample.spv")
set(VBTSTUDIO_CACHE_SPV "${VBTSTUDIO_SHADER_DIR}/vbt_cache.spv")
set(VBTSTUDIO_ACTIVE_INDEX_SPV "${VBTSTUDIO_SHADER_DIR}/vbt_active_index.spv")
set(VBTSTUDIO_COLOR_SPV "${VBTSTUDIO_SHADER_DIR}/vbt_colorize.spv")
set(VBTSTUDIO_RAY_SPV "${VBTSTUDIO_SHADER_DIR}/ray_generate.spv")

add_custom_command(
  OUTPUT "${VBTSTUDIO_VBT_SPV}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${VBTSTUDIO_SHADER_DIR}"
  COMMAND "${VBTSTUDIO_GLSLANG_VALIDATOR}" -V "${VBTSTUDIO_VBT_SHADER}" -o "${VBTSTUDIO_VBT_SPV}"
  DEPENDS "${VBTSTUDIO_VBT_SHADER}"
  VERBATIM
)

add_custom_command(
  OUTPUT "${VBTSTUDIO_CACHE_SPV}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${VBTSTUDIO_SHADER_DIR}"
  COMMAND "${VBTSTUDIO_GLSLANG_VALIDATOR}" -V "${VBTSTUDIO_CACHE_SHADER}" -o "${VBTSTUDIO_CACHE_SPV}"
  DEPENDS "${VBTSTUDIO_CACHE_SHADER}"
  VERBATIM
)

add_custom_command(
  OUTPUT "${VBTSTUDIO_ACTIVE_INDEX_SPV}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${VBTSTUDIO_SHADER_DIR}"
  COMMAND "${VBTSTUDIO_GLSLANG_VALIDATOR}" -V "${VBTSTUDIO_ACTIVE_INDEX_SHADER}" -o "${VBTSTUDIO_ACTIVE_INDEX_SPV}"
  DEPENDS "${VBTSTUDIO_ACTIVE_INDEX_SHADER}"
  VERBATIM
)

add_custom_command(
  OUTPUT "${VBTSTUDIO_COLOR_SPV}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${VBTSTUDIO_SHADER_DIR}"
  COMMAND "${VBTSTUDIO_GLSLANG_VALIDATOR}" -V "${VBTSTUDIO_COLOR_SHADER}" -o "${VBTSTUDIO_COLOR_SPV}"
  DEPENDS "${VBTSTUDIO_COLOR_SHADER}"
  VERBATIM
)

add_custom_command(
  OUTPUT "${VBTSTUDIO_RAY_SPV}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${VBTSTUDIO_SHADER_DIR}"
  COMMAND "${VBTSTUDIO_GLSLANG_VALIDATOR}" -V "${VBTSTUDIO_RAY_SHADER}" -o "${VBTSTUDIO_RAY_SPV}"
  DEPENDS "${VBTSTUDIO_RAY_SHADER}"
  VERBATIM
)

add_custom_target(vbtstudio_shaders DEPENDS
  "${VBTSTUDIO_VBT_SPV}"
  "${VBTSTUDIO_CACHE_SPV}"
  "${VBTSTUDIO_ACTIVE_INDEX_SPV}"
  "${VBTSTUDIO_COLOR_SPV}"
  "${VBTSTUDIO_RAY_SPV}"
)

add_executable(vbtstudio
  src/main.cpp
  src/volume_renderer.cpp
  src/volume_renderer.h
)

add_dependencies(vbtstudio vbtstudio_shaders)

target_link_libraries(vbtstudio
  PRIVATE
    vbtstudio_backend
    vbtstudio_imgui
    Vulkan::Vulkan
)

if(WIN32)
  target_link_libraries(vbtstudio PRIVATE comdlg32 ole32 windowscodecs)
  target_compile_definitions(vbtstudio PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

target_compile_definitions(vbtstudio PRIVATE
  VBTSTUDIO_DEFAULT_VBT_DIR="${VBT_SOURCE_DIR}/data"
  VBTSTUDIO_VBT_SHADER_SPV="${VBTSTUDIO_VBT_SPV}"
  VBTSTUDIO_CACHE_SHADER_SPV="${VBTSTUDIO_CACHE_SPV}"
  VBTSTUDIO_ACTIVE_INDEX_SHADER_SPV="${VBTSTUDIO_ACTIVE_INDEX_SPV}"
  VBTSTUDIO_COLOR_SHADER_SPV="${VBTSTUDIO_COLOR_SPV}"
  VBTSTUDIO_RAY_SHADER_SPV="${VBTSTUDIO_RAY_SPV}"
)

if(MSVC)
  target_compile_options(vbtstudio PRIVATE /W4 /permissive-)
else()
  target_compile_options(vbtstudio PRIVATE -Wall -Wextra -Wpedantic)
endif()
