#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vbt {

enum class RenderTemporalFormalMode : uint8_t {
    EMPTY = 0,
    COARSE_ONLY = 1,
    FINE_COMPACT = 2,
    FINE6_FULL = 3,
};

struct RenderTemporalFormalOptions {
    int leafSize = 8;
    int coarseResolution = 4;
    int fine6Resolution = 6;
    float backgroundZeroRatio = 0.30f;
    bool cutoffProtect = true;
    float controlEpsScale = 8.0f;
};

struct RenderTemporalVariantSpec {
    std::string name;
    RenderTemporalFormalMode mode = RenderTemporalFormalMode::COARSE_ONLY;
    bool routeSelect = false;
    int fineResolution = 0;
    bool gatedFine = false;
    bool tileLocal = false;
    bool shellAwareTile = false;
    bool shellBandResidual = false;
    bool adaptiveShellBand = false;
    bool coarseShellWeightedResidual = false;
    bool temporalSmoothResidual = false;
    bool temporalSmoothGuidance = false;
    bool shellVoxelResidual = false;
    int tileSize = 0;
    int tileGridResolution = 0;
    float visibleFracThreshold = 0.0f;
    float shellFracThreshold = 0.0f;
    float shellPeakThreshold = 0.0f;
    float shellValueThreshold = 0.0f;
    float shellBandThreshold = 0.0f;
    float shellWeightLow = 0.0f;
    float shellWeightHigh = 0.0f;
    float shellMaxGain = 1.0f;
    float temporalSmoothAlpha = 0.0f;
    int temporalSmoothPasses = 0;
    float bandFracThreshold = 0.0f;
    float maskedRmseBandScale = 0.0f;
    float maskedPeakBandScale = 0.0f;
    bool shellVoxelNarrowBand = false;
    float coarseLeakMax = 0.0f;
    float coarseLeakScale = 1.0f;
    bool coarseLeakSmoothRamp = false;
    int shellVoxelDilateRadius = 0;
    float shellVoxelDilateScale = 1.0f;
};

RenderTemporalFormalOptions makeRenderTemporalFormalDefaults();
std::vector<RenderTemporalVariantSpec> buildRenderTemporalFormalProbeVariants();
const char* toString(RenderTemporalFormalMode mode);

} // namespace vbt
