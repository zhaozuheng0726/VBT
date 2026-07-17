cmake_minimum_required(VERSION 3.20)
project(VBTSpatialFirst LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_VBT_RENDER "Build VBT render/query benchmark workspace" OFF)
option(VBT_BUILD_RENDER_TOOLS "Build VBT render/query benchmark workspace" OFF)
option(VBT_BUILD_CYCLES_NATIVE_STAGING "Build native Cycles staging probe" ON)
option(VBT_BUILD_BLENDER_BRIDGE_TOOLS "Build Blender bridge helper tools" OFF)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann")

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
target_compile_definitions(vbt_render_temporal_kf_probe PRIVATE VBT_RENDER_TEMPORAL_PROBE_STANDALONE=1)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vbt_spatialfirst_probe PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(vbt_spatialfirst_probe PRIVATE VBT_USE_OPENMP=1)
    target_link_libraries(vbt_render_temporal_kf_probe PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(vbt_render_temporal_kf_probe PRIVATE VBT_USE_OPENMP=1)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(vbt_raw_volume_tests
        tests/raw_volume_tests.cpp
        src/raw_volume.cpp
    )
    target_include_directories(vbt_raw_volume_tests PRIVATE src)
    add_test(NAME vbt_raw_volume_tests COMMAND vbt_raw_volume_tests)

    add_executable(vbt_file_tests
        tests/vbt_file_tests.cpp
        render/src/vbt_file.cpp
    )
    target_include_directories(vbt_file_tests PRIVATE render/src)
    add_test(NAME vbt_file_tests COMMAND vbt_file_tests)

    add_executable(vbt_render_temporal_payload_tests
        tests/render_temporal_payload_tests.cpp
        src/render_temporal_payload.cpp
    )
    target_include_directories(vbt_render_temporal_payload_tests PRIVATE src)
    add_test(NAME vbt_render_temporal_payload_tests COMMAND vbt_render_temporal_payload_tests)

    add_executable(vbt_cycles_loader_tests
        tests/vbt_cycles_loader_tests.cpp
        cycles_native_vbt/vbt_cycles_loader.cpp
    )
    target_include_directories(vbt_cycles_loader_tests PRIVATE cycles_native_vbt)
    add_test(NAME vbt_cycles_loader_tests COMMAND vbt_cycles_loader_tests)
endif()

if(BUILD_VBT_RENDER OR VBT_BUILD_RENDER_TOOLS)
    add_subdirectory(render)
endif()

if(VBT_BUILD_CYCLES_NATIVE_STAGING)
    add_subdirectory(cycles_native_vbt)
endif()

if(VBT_BUILD_BLENDER_BRIDGE_TOOLS)
    add_subdirectory(blender_bridge_tools)
endif()
