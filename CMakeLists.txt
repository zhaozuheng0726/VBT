cmake_minimum_required(VERSION 3.20)
project(VBTMainline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VBT_BUILD_RENDER_TOOLS "Build Vulkan-based render/query tools" ON)
option(VBT_BUILD_CYCLES_NATIVE_STAGING "Build native Cycles staging probe" ON)
option(VBT_BUILD_BLENDER_BRIDGE_TOOLS "Build optional Blender bridge helper tools" OFF)

add_executable(vbt_spatialfirst_probe
    src/main.cpp
    src/raw_volume.cpp
    src/keyframe_detector.cpp
    src/render_temporal_probe.cpp
    src/render_temporal_kf_probe_main.cpp
    src/render_temporal_formal.cpp
    src/render_temporal_route.cpp
    src/render_temporal_payload.cpp
    src/render_temporal_decode.cpp
    src/temporal_first_encoder.cpp
    src/dct_codec.cpp
    src/fixed_budget_dp.cpp
    src/spatial_first_encoder.cpp
)

target_include_directories(vbt_spatialfirst_probe PRIVATE src)

add_executable(vbt_render_temporal_kf_probe
    src/render_temporal_kf_probe_main.cpp
    src/raw_volume.cpp
    src/render_temporal_probe.cpp
    src/render_temporal_formal.cpp
    src/render_temporal_route.cpp
    src/render_temporal_payload.cpp
    src/render_temporal_decode.cpp
    src/temporal_first_encoder.cpp
)

target_include_directories(vbt_render_temporal_kf_probe PRIVATE src)
target_compile_definitions(vbt_render_temporal_kf_probe PRIVATE
    VBT_RENDER_TEMPORAL_PROBE_STANDALONE=1
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    foreach(target vbt_spatialfirst_probe vbt_render_temporal_kf_probe)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(${target} PRIVATE VBT_USE_OPENMP=1)
    endforeach()
endif()

if(VBT_BUILD_RENDER_TOOLS)
    add_subdirectory(render)
endif()

if(VBT_BUILD_CYCLES_NATIVE_STAGING)
    add_subdirectory(cycles_native_vbt)
endif()

if(VBT_BUILD_BLENDER_BRIDGE_TOOLS)
    add_subdirectory(blender_bridge_tools)
endif()
