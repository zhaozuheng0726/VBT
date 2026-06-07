add_executable(render_temporal_vbt_direct_preview
    render_temporal_vbt_direct_preview.cpp
    ../render/src/vbt_file.cpp
    ../src/render_temporal_payload.cpp
    ../src/render_temporal_decode.cpp
)

target_compile_features(render_temporal_vbt_direct_preview PRIVATE cxx_std_17)
target_include_directories(render_temporal_vbt_direct_preview PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../render/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

find_package(OpenVDB QUIET CONFIG)
if(OpenVDB_FOUND)
    add_executable(render_temporal_vbt_to_vdb
        render_temporal_vbt_to_vdb.cpp
        ../render/src/vbt_file.cpp
        ../src/render_temporal_payload.cpp
        ../src/render_temporal_decode.cpp
    )

    target_compile_features(render_temporal_vbt_to_vdb PRIVATE cxx_std_17)
    target_include_directories(render_temporal_vbt_to_vdb PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../render/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
    )
    target_link_libraries(render_temporal_vbt_to_vdb PRIVATE OpenVDB::openvdb)
else()
    message(STATUS "OpenVDB not found: render_temporal_vbt_to_vdb will not be built.")
endif()
