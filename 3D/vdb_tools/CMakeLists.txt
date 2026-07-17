cmake_minimum_required(VERSION 3.24)
project(vdb_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

include(CTest)

if(EXISTS "C:/work/vcpkg/installed/x64-windows")
  set(TBB_ROOT "C:/work/vcpkg/installed/x64-windows" CACHE PATH "" FORCE)
  set(TBB_INCLUDEDIR "C:/work/vcpkg/installed/x64-windows/include" CACHE PATH "" FORCE)
  set(TBB_LIBRARYDIR "C:/work/vcpkg/installed/x64-windows/lib" CACHE PATH "" FORCE)
  set(BLOSC_ROOT "C:/work/vcpkg/installed/x64-windows" CACHE PATH "" FORCE)
  set(BLOSC_INCLUDEDIR "C:/work/vcpkg/installed/x64-windows/include" CACHE PATH "" FORCE)
  set(BLOSC_LIBRARYDIR "C:/work/vcpkg/installed/x64-windows/lib" CACHE PATH "" FORCE)
  set(ZLIB_INCLUDE_DIR "C:/work/vcpkg/installed/x64-windows/include" CACHE PATH "" FORCE)
  set(ZLIB_LIBRARY "C:/work/vcpkg/installed/x64-windows/lib/zlib.lib" CACHE FILEPATH "" FORCE)
endif()

set(OPENVDB_PREBUILT_DLL "${CMAKE_SOURCE_DIR}/../compare/openvdb-build-vcpkg/openvdb/openvdb/Release/openvdb.dll")
set(OPENVDB_PREBUILT_LIB "${CMAKE_SOURCE_DIR}/../compare/openvdb-build-vcpkg/openvdb/openvdb/Release/openvdb.lib")
set(OPENVDB_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../compare/openvdb-master/openvdb")

if(EXISTS "${OPENVDB_PREBUILT_DLL}" AND EXISTS "${OPENVDB_PREBUILT_LIB}")
  add_library(openvdb_shared SHARED IMPORTED GLOBAL)
  set_target_properties(openvdb_shared PROPERTIES
    IMPORTED_LOCATION "${OPENVDB_PREBUILT_DLL}"
    IMPORTED_IMPLIB "${OPENVDB_PREBUILT_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${OPENVDB_INCLUDE_DIR};${CMAKE_SOURCE_DIR}/../compare/openvdb-master/ext/imath;C:/work/vcpkg/installed/x64-windows/include"
  )
else()
  set(OPENVDB_BUILD_CORE ON CACHE BOOL "" FORCE)
  set(OPENVDB_BUILD_BINARIES OFF CACHE BOOL "" FORCE)
  set(OPENVDB_BUILD_PYTHON_MODULE OFF CACHE BOOL "" FORCE)
  set(OPENVDB_BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
  set(OPENVDB_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(OPENVDB_BUILD_HOUDINI_PLUGIN OFF CACHE BOOL "" FORCE)
  set(OPENVDB_BUILD_HOUDINI_ABITESTS OFF CACHE BOOL "" FORCE)
  set(OPENVDB_BUILD_MAYA_PLUGIN OFF CACHE BOOL "" FORCE)
  set(OPENVDB_BUILD_AX OFF CACHE BOOL "" FORCE)
  set(OPENVDB_BUILD_AX_UNITTESTS OFF CACHE BOOL "" FORCE)
  set(OPENVDB_BUILD_NANOVDB OFF CACHE BOOL "" FORCE)
  set(OPENVDB_CORE_SHARED ON CACHE BOOL "" FORCE)
  set(OPENVDB_CORE_STATIC OFF CACHE BOOL "" FORCE)
  set(USE_BLOSC ON CACHE BOOL "" FORCE)
  set(USE_ZLIB ON CACHE BOOL "" FORCE)
  set(USE_LOG4CPLUS OFF CACHE BOOL "" FORCE)
  set(USE_EXR OFF CACHE BOOL "" FORCE)
  set(USE_IMATH_HALF OFF CACHE BOOL "" FORCE)
  set(USE_PNG OFF CACHE BOOL "" FORCE)
  add_subdirectory("${CMAKE_SOURCE_DIR}/../compare/openvdb-master" "${CMAKE_BINARY_DIR}/openvdb_build")
endif()

add_executable(vdb_to_raw ../vdb_to_raw.cpp)
target_link_libraries(vdb_to_raw PRIVATE openvdb_shared)

add_executable(vdb_to_obj ../vdb_to_obj.cpp)
target_compile_options(vdb_to_obj PRIVATE /bigobj)
target_link_libraries(vdb_to_obj PRIVATE openvdb_shared)

add_executable(objseq_to_raw ../objseq_to_raw.cpp)
target_include_directories(objseq_to_raw PRIVATE
  "${CMAKE_SOURCE_DIR}/.."
  "${CMAKE_SOURCE_DIR}/../compare/openvdb-master/openvdb_cmd/vdb_tool/include"
)
target_compile_options(objseq_to_raw PRIVATE /bigobj)
target_link_libraries(objseq_to_raw PRIVATE openvdb_shared "C:/work/vcpkg/installed/x64-windows/lib/tbb12.lib")
add_custom_command(TARGET objseq_to_raw POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:openvdb_shared>"
    "$<TARGET_FILE_DIR:objseq_to_raw>/openvdb.dll"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/work/vcpkg/installed/x64-windows/bin/blosc.dll"
    "$<TARGET_FILE_DIR:objseq_to_raw>/blosc.dll"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/work/vcpkg/installed/x64-windows/bin/zlib1.dll"
    "$<TARGET_FILE_DIR:objseq_to_raw>/zlib1.dll"
)

add_executable(raw_to_vdb ../raw_to_vdb.cpp)
target_include_directories(raw_to_vdb PRIVATE "${CMAKE_SOURCE_DIR}/..")
target_link_libraries(raw_to_vdb PRIVATE openvdb_shared)

if(BUILD_TESTING)
  add_executable(frame_metadata_tests frame_metadata_tests.cpp)
  target_include_directories(frame_metadata_tests PRIVATE
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/.."
  )
  add_test(NAME frame_metadata_tests COMMAND frame_metadata_tests)
endif()

add_executable(vdb_to_nanovdb ../vdb_to_nanovdb.cpp)
target_include_directories(vdb_to_nanovdb PRIVATE
  "${CMAKE_SOURCE_DIR}/../compare/openvdb-master/nanovdb"
)
target_link_libraries(vdb_to_nanovdb PRIVATE openvdb_shared)

add_executable(vbt_to_vdb ../vbt_to_vdb.cpp ../block_tree.cpp ../keyframe_detector.cpp)
target_include_directories(vbt_to_vdb PRIVATE "${CMAKE_SOURCE_DIR}/..")
target_link_libraries(vbt_to_vdb PRIVATE openvdb_shared)

add_executable(vbtpack4_to_vdb
  ../vbtpack4_to_vdb.cpp
  ../../render/src/vbt_file.cpp
)
target_include_directories(vbtpack4_to_vdb PRIVATE
  "${CMAKE_SOURCE_DIR}/.."
  "${CMAKE_SOURCE_DIR}/../../render/src"
  "${CMAKE_SOURCE_DIR}/../../src"
)
target_link_libraries(vbtpack4_to_vdb PRIVATE openvdb_shared)

add_executable(render_temporal_vbt_to_vdb
  ../render_temporal_vbt_to_vdb.cpp
  ../../render/src/vbt_file.cpp
  ../../src/render_temporal_payload.cpp
  ../../src/render_temporal_decode.cpp
)
target_include_directories(render_temporal_vbt_to_vdb PRIVATE
  "${CMAKE_SOURCE_DIR}/.."
  "${CMAKE_SOURCE_DIR}/../../render/src"
  "${CMAKE_SOURCE_DIR}/../../src"
  "${CMAKE_SOURCE_DIR}/../../third_party/nlohmann"
)
target_link_libraries(render_temporal_vbt_to_vdb PRIVATE openvdb_shared)

add_executable(render_temporal_vbt_direct_preview
  ../render_temporal_vbt_direct_preview.cpp
  ../../render/src/vbt_file.cpp
  ../../src/render_temporal_payload.cpp
  ../../src/render_temporal_decode.cpp
)
target_include_directories(render_temporal_vbt_direct_preview PRIVATE
  "${CMAKE_SOURCE_DIR}/.."
  "${CMAKE_SOURCE_DIR}/../../render/src"
  "${CMAKE_SOURCE_DIR}/../../src"
  "${CMAKE_SOURCE_DIR}/../../third_party/nlohmann"
)

add_executable(render_temporal_leaf_diagnose
  ../render_temporal_leaf_diagnose.cpp
  ../../render/src/vbt_file.cpp
  ../../src/render_temporal_payload.cpp
  ../../src/render_temporal_decode.cpp
)
target_include_directories(render_temporal_leaf_diagnose PRIVATE
  "${CMAKE_SOURCE_DIR}/.."
  "${CMAKE_SOURCE_DIR}/../../render/src"
  "${CMAKE_SOURCE_DIR}/../../src"
  "${CMAKE_SOURCE_DIR}/../../third_party/nlohmann"
)
target_link_libraries(render_temporal_leaf_diagnose PRIVATE openvdb_shared)

add_executable(vdb_render_cli ../compare/openvdb-master/openvdb_cmd/vdb_render/main.cc)
target_include_directories(vdb_render_cli PRIVATE "${CMAKE_SOURCE_DIR}/..")
target_link_libraries(vdb_render_cli PRIVATE openvdb_shared "C:/work/vcpkg/installed/x64-windows/lib/tbb12.lib")
