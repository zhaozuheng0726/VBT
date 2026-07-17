#pragma once

#include <array>
#include <cstdint>

namespace vbtstudio::backend {

struct CameraState {
    float yaw = -0.62f;
    float pitch = 0.16f;
    float distance = 1.85f;
    float field_of_view = 38.0f;
    std::array<float, 3> target_offset{};
    std::uint32_t up_axis = 1;
};

void orbit_camera(CameraState& camera, float delta_yaw, float delta_pitch);
void zoom_camera(CameraState& camera, float wheel_delta);
void reset_camera(CameraState& camera);

} // namespace vbtstudio::backend
