set(VBT_RENDER_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(vbt_export_scientific_frame
    src/export_scientific_frame_main.cpp
    src/vbt_file.cpp
    src/query_patterns.cpp
    src/scientific_decode.cpp
)

target_compile_features(vbt_export_scientific_frame PRIVATE cxx_std_17)
target_include_directories(vbt_export_scientific_frame PRIVATE ${VBT_RENDER_INCLUDE_DIRS})

add_executable(vbt_render_smoke_probe
    src/render_smoke_sampling_probe_main.cpp
    src/vbt_file.cpp
)

target_compile_features(vbt_render_smoke_probe PRIVATE cxx_std_17)
target_include_directories(vbt_render_smoke_probe PRIVATE ${VBT_RENDER_INCLUDE_DIRS})

find_package(Vulkan QUIET)
find_program(GLSLC_EXECUTABLE NAMES glslc)

if(Vulkan_FOUND AND GLSLC_EXECUTABLE)
    set(VBT_QUERY_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vbt_query.comp)
    set(VBT_QUERY_SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/vbt_query.comp.spv)
    set(VBT_DENSE_QUERY_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/dense_query.comp)
    set(VBT_DENSE_QUERY_SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/dense_query.comp.spv)
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
    target_include_directories(vbt_query_bench PRIVATE ${VBT_RENDER_INCLUDE_DIRS})
    target_link_libraries(vbt_query_bench PRIVATE Vulkan::Vulkan)
    target_compile_definitions(vbt_query_bench PRIVATE
        VBT_QUERY_SHADER_SPV_PATH="${VBT_QUERY_SHADER_SPV}"
        VBT_DENSE_QUERY_SHADER_SPV_PATH="${VBT_DENSE_QUERY_SHADER_SPV}"
    )

    if(OpenMP_CXX_FOUND)
        foreach(target vbt_query_bench)
            target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
            target_compile_definitions(${target} PRIVATE VBT_USE_OPENMP=1)
        endforeach()
    endif()
else()
    message(STATUS "Skipping Vulkan render/query targets because Vulkan SDK or glslc was not found.")
endif()
