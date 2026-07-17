#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace vbtstudio::backend {

struct MaterialState {
    std::string preset = "paper_gray";
    float density_scale = 10.0f;
    float density_threshold = 0.0f;
    float density_gamma = 0.9f;
    float anisotropy = 0.2f;
    float flame_strength = 0.9f;
    float flame_threshold = 0.0f;
    float temperature_min = 1000.0f;
    float temperature_max = 18000.0f;
    float exposure = 0.15f;
    std::uint32_t sample_steps = 192;
    std::uint32_t volume_model = 0;
    float fire_scattering = 0.45f;
    float fire_blackbody_mix = 0.35f;
    float fire_glow = 0.65f;
    std::uint32_t surface_model = 0;
    float surface_iso = 0.0f;
    float surface_epsilon_voxels = 0.08f;
    float surface_normal_step = 0.75f;
    float surface_roughness = 0.08f;
    float surface_metallic = 0.0f;
    float surface_opacity = 0.9f;
    float water_ior = 1.333f;
    float absorption_density = 0.035f;
    float reflection_strength = 1.0f;
    float environment_strength = 1.0f;
    float floor_offset = 0.08f;
    float shadow_strength = 0.7f;
    std::uint32_t shadow_steps = 48;
    std::array<float, 3> smoke_color{0.78f, 0.80f, 0.82f};
    std::array<float, 3> fire_tint{1.0f, 0.78f, 0.58f};
    std::array<float, 3> surface_color{0.16f, 0.48f, 0.68f};
    std::array<float, 3> absorption_color{0.78f, 0.92f, 0.98f};
    std::array<float, 3> background_color{0.025f, 0.027f, 0.03f};
};

void apply_material_preset(MaterialState& material, const std::string& preset);

} // namespace vbtstudio::backend
