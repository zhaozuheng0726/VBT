cmake_minimum_required(VERSION 3.20)

project(VBTRender LANGUAGES CXX)

find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE NAMES glslc REQUIRED)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING_FULL OFF CACHE BOOL "" FORCE)
set(BUILD_UTILITIES OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_CFP OFF CACHE BOOL "" FORCE)
set(BUILD_ZFORP OFF CACHE BOOL "" FORCE)
set(BUILD_ZFPY OFF CACHE BOOL "" FORCE)
set(ZFP_WITH_CUDA OFF CACHE BOOL "" FORCE)
set(ZFP_WITH_OPENMP OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../3D/compare/zfp-develop
                 ${CMAKE_CURRENT_BINARY_DIR}/third_party/zfp
                 EXCLUDE_FROM_ALL)

set(VBT_QUERY_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vbt_query.comp)
set(VBT_QUERY_SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/vbt_query.comp.spv)
set(VBT_DENSE_QUERY_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/dense_query.comp)
set(VBT_DENSE_QUERY_SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/dense_query.comp.spv)
set(VBT_DENSE_SMOKE_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/dense_smoke_sample.comp)
set(VBT_DENSE_SMOKE_SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/dense_smoke_sample.comp.spv)
set(VBT_SMOKE_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vbt_smoke_sample.comp)
set(VBT_SMOKE_SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/vbt_smoke_sample.comp.spv)
set(VBT_NANOVDB_SMOKE_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/nanovdb_smoke_sample.comp)
set(VBT_NANOVDB_SMOKE_SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/nanovdb_smoke_sample.comp.spv)

add_custom_command(
    OUTPUT ${VBT_QUERY_SHADER_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMAND ${GLSLC_EXECUTABLE} -O ${VBT_QUERY_SHADER_SOURCE} -o ${VBT_QUERY_SHADER_SPV}
    DEPENDS ${VBT_QUERY_SHADER_SOURCE}
    COMMENT "Compiling VBT query compute shader"
    VERBATIM
)

add_custom_command(
    OUTPUT ${VBT_DENSE_QUERY_SHADER_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMAND ${GLSLC_EXECUTABLE} -O ${VBT_DENSE_QUERY_SHADER_SOURCE} -o ${VBT_DENSE_QUERY_SHADER_SPV}
    DEPENDS ${VBT_DENSE_QUERY_SHADER_SOURCE}
    COMMENT "Compiling dense query compute shader"
    VERBATIM
)

add_custom_command(
    OUTPUT ${VBT_DENSE_SMOKE_SHADER_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMAND ${GLSLC_EXECUTABLE} -O ${VBT_DENSE_SMOKE_SHADER_SOURCE} -o ${VBT_DENSE_SMOKE_SHADER_SPV}
    DEPENDS ${VBT_DENSE_SMOKE_SHADER_SOURCE}
    COMMENT "Compiling dense smoke compute shader"
    VERBATIM
)

add_custom_command(
    OUTPUT ${VBT_SMOKE_SHADER_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMAND ${GLSLC_EXECUTABLE} -O ${VBT_SMOKE_SHADER_SOURCE} -o ${VBT_SMOKE_SHADER_SPV}
    DEPENDS ${VBT_SMOKE_SHADER_SOURCE}
    COMMENT "Compiling VBT smoke compute shader"
    VERBATIM
)

add_custom_command(
    OUTPUT ${VBT_NANOVDB_SMOKE_SHADER_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMAND ${GLSLC_EXECUTABLE}
            -O
            -I${CMAKE_CURRENT_SOURCE_DIR}/shaders
            ${VBT_NANOVDB_SMOKE_SHADER_SOURCE}
            -o ${VBT_NANOVDB_SMOKE_SHADER_SPV}
    DEPENDS
        ${VBT_NANOVDB_SMOKE_SHADER_SOURCE}
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/nanovdb/PNanoVDB.h
    COMMENT "Compiling NanoVDB smoke compute shader"
    VERBATIM
)

add_executable(vbt_query_bench
    src/main.cpp
    src/vbt_file.cpp
    src/query_patterns.cpp
    src/gpu_query_bench.cpp
    src/scientific_decode.cpp
    ${VBT_QUERY_SHADER_SPV}
    ${VBT_DENSE_QUERY_SHADER_SPV}
)

target_compile_features(vbt_query_bench PRIVATE cxx_std_17)
target_include_directories(vbt_query_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(vbt_query_bench PRIVATE Vulkan::Vulkan)
target_compile_definitions(vbt_query_bench PRIVATE
    VBT_QUERY_SHADER_SPV_PATH="${VBT_QUERY_SHADER_SPV}"
    VBT_DENSE_QUERY_SHADER_SPV_PATH="${VBT_DENSE_QUERY_SHADER_SPV}"
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vbt_query_bench PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(vbt_query_bench PRIVATE VBT_USE_OPENMP=1)
endif()

add_executable(vbt_export_scientific_frame
    src/export_scientific_frame_main.cpp
    src/vbt_file.cpp
    src/query_patterns.cpp
    src/scientific_decode.cpp
)

target_compile_features(vbt_export_scientific_frame PRIVATE cxx_std_17)
target_include_directories(vbt_export_scientific_frame PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(vbt_cpu_query_compare
    src/cpu_query_bench_main.cpp
    src/vbt_file.cpp
    src/query_patterns.cpp
    src/scientific_decode.cpp
)

target_compile_features(vbt_cpu_query_compare PRIVATE cxx_std_17)
target_include_directories(vbt_cpu_query_compare PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(vbt_cpu_query_compare PRIVATE zfp)

add_executable(vbt_render_smoke_probe
    src/render_smoke_sampling_probe_main.cpp
    src/vbt_file.cpp
)

target_compile_features(vbt_render_smoke_probe PRIVATE cxx_std_17)
target_include_directories(vbt_render_smoke_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(vbt_smoke_dense_bench
    src/render_smoke_dense_bench_main.cpp
    ${VBT_DENSE_SMOKE_SHADER_SPV}
)

target_compile_features(vbt_smoke_dense_bench PRIVATE cxx_std_17)
target_include_directories(vbt_smoke_dense_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(vbt_smoke_dense_bench PRIVATE Vulkan::Vulkan)
target_compile_definitions(vbt_smoke_dense_bench PRIVATE
    VBT_DENSE_SMOKE_SHADER_SPV_PATH="${VBT_DENSE_SMOKE_SHADER_SPV}"
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(vbt_smoke_dense_bench PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(vbt_smoke_dense_bench PRIVATE VBT_USE_OPENMP=1)
endif()

add_executable(vbt_smoke_dense_sequence_bench
    src/render_smoke_dense_sequence_bench_main.cpp
    ${VBT_DENSE_SMOKE_SHADER_SPV}
)

target_compile_features(vbt_smoke_dense_sequence_bench PRIVATE cxx_std_17)
target_include_directories(vbt_smoke_dense_sequence_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(vbt_smoke_dense_sequence_bench PRIVATE Vulkan::Vulkan)
target_compile_definitions(vbt_smoke_dense_sequence_bench PRIVATE
    VBT_DENSE_SMOKE_SHADER_SPV_PATH="${VBT_DENSE_SMOKE_SHADER_SPV}"
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(vbt_smoke_dense_sequence_bench PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(vbt_smoke_dense_sequence_bench PRIVATE VBT_USE_OPENMP=1)
endif()

add_executable(vbt_smoke_vbt_bench
    src/render_smoke_vbt_bench_main.cpp
    src/vbt_file.cpp
    ../src/render_temporal_decode.cpp
    ../src/render_temporal_payload.cpp
    ${VBT_SMOKE_SHADER_SPV}
)

target_compile_features(vbt_smoke_vbt_bench PRIVATE cxx_std_17)
target_include_directories(vbt_smoke_vbt_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(vbt_smoke_vbt_bench PRIVATE Vulkan::Vulkan)
target_compile_definitions(vbt_smoke_vbt_bench PRIVATE
    VBT_SMOKE_SHADER_SPV_PATH="${VBT_SMOKE_SHADER_SPV}"
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(vbt_smoke_vbt_bench PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(vbt_smoke_vbt_bench PRIVATE VBT_USE_OPENMP=1)
endif()

set(VBT_NANOVDB_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/../3D/compare/openvdb-master/nanovdb
)

if(EXISTS ${VBT_NANOVDB_INCLUDE_DIR}/nanovdb/NanoVDB.h)
    add_executable(vbt_smoke_nanovdb_bench
        src/render_smoke_nanovdb_bench_main.cpp
        ${VBT_NANOVDB_SMOKE_SHADER_SPV}
    )

    add_executable(vbt_smoke_nanovdb_sequence_bench
        src/render_smoke_nanovdb_sequence_bench_main.cpp
        ${VBT_NANOVDB_SMOKE_SHADER_SPV}
    )

    foreach(target vbt_smoke_nanovdb_bench vbt_smoke_nanovdb_sequence_bench)
        target_compile_features(${target} PRIVATE cxx_std_17)
        target_include_directories(${target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../src
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${VBT_NANOVDB_INCLUDE_DIR}
        )
        target_link_libraries(${target} PRIVATE Vulkan::Vulkan)
        target_compile_definitions(${target} PRIVATE
            VBT_NANOVDB_SMOKE_SHADER_SPV_PATH="${VBT_NANOVDB_SMOKE_SHADER_SPV}"
        )
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
            target_compile_definitions(${target} PRIVATE VBT_USE_OPENMP=1)
        endif()
    endforeach()
else()
    message(STATUS "Skipping NanoVDB smoke targets because NanoVDB headers were not found.")
endif()
