add_library(vbtstudio_backend STATIC
  src/camera.cpp
  src/material.cpp
  src/vbt_asset.cpp
  src/timeline.cpp
  src/studio_session.cpp
)

target_include_directories(vbtstudio_backend
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${VBT_SOURCE_DIR}/third_party
)

target_compile_features(vbtstudio_backend PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(vbtstudio_backend PRIVATE /W4 /permissive-)
else()
  target_compile_options(vbtstudio_backend PRIVATE -Wall -Wextra -Wpedantic)
endif()
