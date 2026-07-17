#include "vbtstudio/backend/material.h"

namespace vbtstudio::backend {

void apply_material_preset(MaterialState& material, const std::string& preset)
{
    material.preset = preset;
    material.density_threshold = 0.0f;
    material.anisotropy = 0.2f;
    material.volume_model = 0;
    if (preset == "water_studio") {
        material.surface_model = 1;
        material.sample_steps = 320;
        material.shadow_steps = 48;
        material.surface_color = {0.66f, 0.76f, 0.88f};
        material.absorption_color = {0.45f, 0.68f, 0.85f};
        material.surface_roughness = 0.025f;
        material.surface_normal_step = 1.25f;
        material.surface_metallic = 0.0f;
        material.surface_opacity = 1.0f;
        material.water_ior = 1.333f;
        material.absorption_density = 0.18f;
        material.reflection_strength = 1.0f;
        material.environment_strength = 1.15f;
        material.floor_offset = 0.035f;
        material.shadow_strength = 0.72f;
        material.exposure = -0.15f;
        material.background_color = {0.13f, 0.16f, 0.20f};
    }
    else if (preset == "water_clear") {
        material.surface_model = 0;
        material.sample_steps = 256;
        material.surface_color = {0.10f, 0.43f, 0.62f};
        material.surface_roughness = 0.06f;
        material.surface_metallic = 0.0f;
        material.surface_opacity = 0.86f;
        material.exposure = 0.2f;
        material.background_color = {0.025f, 0.035f, 0.04f};
    }
    else if (preset == "water_blue") {
        material.surface_model = 0;
        material.sample_steps = 320;
        material.surface_color = {0.035f, 0.28f, 0.55f};
        material.surface_roughness = 0.12f;
        material.surface_metallic = 0.0f;
        material.surface_opacity = 0.96f;
        material.exposure = 0.35f;
        material.background_color = {0.012f, 0.018f, 0.024f};
    }
    else if (preset == "water_silver") {
        material.surface_model = 0;
        material.sample_steps = 256;
        material.surface_color = {0.55f, 0.67f, 0.72f};
        material.surface_roughness = 0.2f;
        material.surface_metallic = 0.35f;
        material.surface_opacity = 1.0f;
        material.exposure = 0.1f;
        material.background_color = {0.025f, 0.027f, 0.03f};
    }
    else if (preset == "charcoal") {
        material.density_scale = 14.0f;
        material.density_gamma = 0.85f;
        material.exposure = 0.10f;
        material.smoke_color = {0.56f, 0.58f, 0.60f};
        material.background_color = {0.006f, 0.007f, 0.008f};
    }
    else if (preset == "soft_ash") {
        material.density_scale = 9.0f;
        material.density_gamma = 1.0f;
        material.exposure = 0.12f;
        material.smoke_color = {0.76f, 0.68f, 0.58f};
        material.background_color = {0.035f, 0.028f, 0.023f};
    }
    else if (preset == "cool_steel") {
        material.density_scale = 10.0f;
        material.density_gamma = 0.95f;
        material.exposure = 0.18f;
        material.smoke_color = {0.42f, 0.64f, 0.72f};
        material.background_color = {0.008f, 0.014f, 0.018f};
    }
    else if (preset == "fire_physical") {
        material.volume_model = 1;
        material.sample_steps = 160;
        material.density_scale = 2.8f;
        material.density_gamma = 0.92f;
        material.exposure = -0.35f;
        material.flame_strength = 1.15f;
        material.flame_threshold = 0.015f;
        material.temperature_min = 500.0f;
        material.temperature_max = 20000.0f;
        material.anisotropy = 0.12f;
        material.fire_scattering = 0.62f;
        material.fire_blackbody_mix = 0.20f;
        material.fire_glow = 0.85f;
        material.smoke_color = {0.12f, 0.13f, 0.15f};
        material.fire_tint = {1.0f, 0.78f, 0.58f};
        material.background_color = {0.002f, 0.002f, 0.004f};
    }
    else if (preset == "fire_warm") {
        material.density_scale = 8.0f;
        material.density_gamma = 0.9f;
        material.exposure = 0.35f;
        material.flame_strength = 2.4f;
        material.flame_threshold = 0.02f;
        material.temperature_min = 500.0f;
        material.temperature_max = 10000.0f;
        material.smoke_color = {0.20f, 0.18f, 0.16f};
        material.fire_tint = {1.0f, 0.78f, 0.58f};
        material.background_color = {0.006f, 0.004f, 0.003f};
    }
    else {
        material.preset = "paper_gray";
        material.density_scale = 10.0f;
        material.density_gamma = 0.9f;
        material.exposure = 0.15f;
        material.smoke_color = {0.78f, 0.80f, 0.82f};
        material.background_color = {0.025f, 0.027f, 0.030f};
    }
}

} // namespace vbtstudio::backend
