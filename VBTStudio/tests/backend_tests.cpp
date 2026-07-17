#include "vbtstudio/backend/studio_session.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>

namespace {

void write_test_vbt(const std::filesystem::path& path)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    const std::array<char, 8> magic{'V', 'B', 'T', 'P', 'A', 'C', 'K', '4'};
    const std::array<std::uint32_t, 11> header{
        4, 8, 8, 8, 10, 8, 4, 0, 1, 1, 0,
    };
    const std::array<std::uint32_t, 2> offsets{0, 1};
    const std::uint32_t payload = 0;
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    output.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
    output.write(reinterpret_cast<const char*>(offsets.data()), sizeof(offsets));
    output.write(reinterpret_cast<const char*>(&payload), sizeof(payload));
}

} // namespace

int main()
{
    const auto directory = std::filesystem::temp_directory_path() / "vbtstudio_backend_tests";
    std::filesystem::create_directories(directory);
    const auto path = directory / "vbtstudio_density_test.vbtp";
    const auto flame_path = directory / "vbtstudio_flames_test.vbtp";
    const auto temperature_path = directory / "vbtstudio_temperature_test.vbtp";
    write_test_vbt(path);
    write_test_vbt(flame_path);
    write_test_vbt(temperature_path);

    vbtstudio::backend::StudioSession session;
    assert(session.open_asset(path));
    assert(session.asset().has_value());
    assert(session.asset()->width == 8);
    assert(session.asset()->frames == 10);
    assert(session.asset()->offset_table_offset == 52);
    assert(session.asset()->payload_offset == 60);
    assert(session.asset()->role == vbtstudio::backend::FieldRole::Density);
    assert((session.asset()->bbox_min == std::array<std::int32_t, 3>{0, 0, 0}));
    assert((session.asset()->bbox_max == std::array<std::int32_t, 3>{7, 7, 7}));
    assert(session.add_field(flame_path));
    assert(session.secondary_field().has_value());
    assert(session.add_field(temperature_path));
    assert(session.temperature_field().has_value());

    vbtstudio::backend::apply_material_preset(session.material(), "charcoal");
    assert(session.material().preset == "charcoal");
    assert(session.material().density_scale == 14.0f);
    vbtstudio::backend::apply_material_preset(session.material(), "fire_warm");
    assert(session.material().flame_strength == 2.4f);
    assert(session.material().volume_model == 0u);
    vbtstudio::backend::apply_material_preset(session.material(), "fire_physical");
    assert(session.material().volume_model == 1u);
    assert(session.material().sample_steps == 160u);
    assert(session.material().fire_scattering == 0.62f);
    assert(session.material().temperature_max == 20000.0f);

    const auto level_set_directory = directory / "level_set";
    std::filesystem::create_directories(level_set_directory);
    const auto level_set_path = level_set_directory / "water_levelset_mode2.vbtp";
    write_test_vbt(level_set_path);
    {
        std::ofstream metadata(level_set_directory / "water_levelset.metadata.json");
        metadata << R"({"bbox_min":[-2,3,7],"bbox_max":[5,10,14],)"
                    R"("voxel_size":0.5,"data_max":1.5,"conversion_mode":"levelset"})";
    }
    assert(session.open_asset(level_set_path));
    assert(session.asset()->role == vbtstudio::backend::FieldRole::LevelSet);
    assert(session.asset()->voxel_size == 0.5f);
    assert(session.asset()->background_value == 1.5f);
    assert((session.asset()->bbox_min == std::array<std::int32_t, 3>{-2, 3, 7}));
    assert(session.material().preset == "water_clear");
    assert(session.material().sample_steps == 256u);
    vbtstudio::backend::apply_material_preset(session.material(), "water_studio");
    assert(session.material().surface_model == 1u);
    assert(session.material().water_ior == 1.333f);
    assert(session.material().shadow_steps == 48u);
    vbtstudio::backend::orbit_camera(session.camera(), 0.5f, 2.0f);
    assert(session.camera().pitch == 1.45f);
    vbtstudio::backend::reset_camera(session.camera());
    assert(session.camera().distance == 1.85f);
    assert(session.camera().up_axis == 1u);
    assert((session.camera().target_offset == std::array<float, 3>{0.0f, 0.0f, 0.0f}));

    auto& timeline = session.timeline();
    timeline.seek(9);
    timeline.set_loop_mode(vbtstudio::backend::LoopMode::Loop);
    timeline.play();
    session.update(1.0 / timeline.fps());
    assert(timeline.frame() == 0);
    timeline.step(-1);
    assert(timeline.frame() == 9);

    std::filesystem::remove_all(directory);
    return 0;
}
