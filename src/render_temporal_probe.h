#pragma once

#include "raw_volume.h"
#include "spatial_first_encoder.h"

namespace vbt {

struct RenderTemporalFirstProbeStats {
    uint64_t voxelCount = 0;
    uint64_t backgroundZeroedVoxels = 0;
    uint64_t totalKeyframes = 0;
    uint64_t temporalProtectedSeries = 0;
    int maxKeyframes = 0;
    double rmse = 0.0;
    double psnr = 0.0;
};

RawVolume4D applyRenderTemporalFirstProbe(const RawVolume4D& source,
                                          const SpatialFirstOptions& options,
                                          RenderTemporalFirstProbeStats* outStats);

} // namespace vbt
