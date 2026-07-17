#include "vbtstudio/backend/camera.h"

#include <algorithm>
#include <cmath>

namespace vbtstudio::backend {

void orbit_camera(CameraState& camera, float delta_yaw, float delta_pitch)
{
    camera.yaw += delta_yaw;
    camera.pitch = std::clamp(camera.pitch + delta_pitch, -1.45f, 1.45f);
}

void zoom_camera(CameraState& camera, float wheel_delta)
{
    camera.distance = std::clamp(camera.distance * std::exp(-wheel_delta * 0.12f), 0.75f, 6.0f);
}

void reset_camera(CameraState& camera) { camera = {}; }

} // namespace vbtstudio::backend
