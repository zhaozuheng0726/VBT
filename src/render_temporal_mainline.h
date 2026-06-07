#pragma once

#include "spatial_first_encoder.h"

#include <filesystem>

namespace vbt {

int runRenderTemporalMainline(const std::filesystem::path& inputRaw,
                              const std::filesystem::path& metadataPath,
                              const std::filesystem::path& reportPath,
                              const SpatialFirstOptions& options);

} // namespace vbt
