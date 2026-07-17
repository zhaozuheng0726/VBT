#include "vbtstudio/backend/studio_session.h"

namespace vbtstudio::backend {

bool StudioSession::open_asset(const std::filesystem::path& path)
{
    const AssetOpenResult result = inspect_vbt_asset(path);
    if (!result) {
        last_error_ = result.error;
        return false;
    }
    asset_ = result.asset;
    secondary_field_.reset();
    temperature_field_.reset();
    last_error_.clear();
    timeline_.configure(asset_->frames, 24.0);
    if (asset_->role == FieldRole::LevelSet) {
        apply_material_preset(material_, "water_clear");
        camera_.distance = 2.0f;
        camera_.field_of_view = 38.0f;
    }
    return true;
}

bool StudioSession::add_field(const std::filesystem::path& path)
{
    if (!asset_) {
        last_error_ = "Open a primary VBT before adding another field.";
        return false;
    }
    const AssetOpenResult result = inspect_vbt_asset(path);
    if (!result) {
        last_error_ = result.error;
        return false;
    }
    if (result.asset->frames != asset_->frames) {
        last_error_ = "Field frame count does not match the primary VBT.";
        return false;
    }
    if (result.asset->role == FieldRole::Flames) {
        secondary_field_ = result.asset;
    }
    else if (result.asset->role == FieldRole::Temperature) {
        temperature_field_ = result.asset;
    }
    else {
        last_error_ = "Additional fields must be flames or temperature VBT files.";
        return false;
    }
    last_error_.clear();
    return true;
}

void StudioSession::update(double elapsed_seconds) { timeline_.update(elapsed_seconds); }
const std::optional<VbtAssetInfo>& StudioSession::asset() const noexcept { return asset_; }
const std::string& StudioSession::last_error() const noexcept { return last_error_; }
const std::optional<VbtAssetInfo>& StudioSession::secondary_field() const noexcept { return secondary_field_; }
const std::optional<VbtAssetInfo>& StudioSession::temperature_field() const noexcept { return temperature_field_; }
Timeline& StudioSession::timeline() noexcept { return timeline_; }
const Timeline& StudioSession::timeline() const noexcept { return timeline_; }
MaterialState& StudioSession::material() noexcept { return material_; }
const MaterialState& StudioSession::material() const noexcept { return material_; }
CameraState& StudioSession::camera() noexcept { return camera_; }
const CameraState& StudioSession::camera() const noexcept { return camera_; }

} // namespace vbtstudio::backend
