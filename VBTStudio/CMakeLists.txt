cmake_minimum_required(VERSION 3.24)

project(VBTStudio VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VBTSTUDIO_BUILD_FRONTEND "Build the Vulkan desktop frontend" ON)
option(VBTSTUDIO_BUILD_TESTS "Build backend tests" ON)
set(VBT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "Path to the VBT repository")

add_subdirectory(backend)

if(VBTSTUDIO_BUILD_FRONTEND)
  add_subdirectory(frontend)
endif()

if(VBTSTUDIO_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
