#pragma once

#include "render_temporal_formal.h"
#include "render_temporal_route.h"
#include "spatial_first_encoder.h"

#include <array>
#include <cstdint>

namespace vbt {

// Temporal-first is currently a render/smoke-only frontend. It intentionally
// owns the render-specific mode dictionary so scientific and render can share
// a backend later without sharing the same frontend ordering.
enum class TemporalFirstPackedMode : uint8_t {
    EMPTY = 0,
    TEMPORAL_GRID4 = 1,
    TEMPORAL_FINE_COMPACT = 2,
    TEMPORAL_FINE6 = 3,
};

// This struct does not yet encode the payload itself. Its job is to freeze the
// render temporal-first contract: block size, mode family, time indexing and
// GPU-friendly constraints that later packed payload code must obey.
struct TemporalFirstEncoderOptions {
    RenderTemporalFormalOptions formal;
    RenderTemporalRouteOptions route;

    float cutoff = 0.0f;
    float cutoffBand = 0.0f;
    float temporalEpsAbs = 1e-5f;
    float temporalEpsRel = 0.02f;
    float temporalGammaDelta = 0.2f;
    bool cutoffProtect = true;

    // GPU-friendly constraints:
    // - offset table stays global
    // - header stores only mode + local metadata, never payload offsets
    // - time lookup uses a tiny bin index plus bounded bin-local search
    int timeBinCount = 8;
    int maxBinLocalKeys = 8;
    bool useGlobalOffsetTable = true;
    bool alignPayloadTo4Bytes = true;
};

struct TemporalFirstPackedLeafShape {
    TemporalFirstPackedMode mode = TemporalFirstPackedMode::EMPTY;
    bool hasTimeBinIndex = false;
    bool hasCoarseStream = false;
    bool hasFineStream = false;
    int coarseResolution = 0;
    int fineResolution = 0;
};

bool isTemporalFirstRenderProfile(const FieldProfile& profile);
TemporalFirstEncoderOptions makeTemporalFirstEncoderOptions(const SpatialFirstOptions& options);
std::array<TemporalFirstPackedLeafShape, 4> buildTemporalFirstPackedLeafShapes();
const char* toString(TemporalFirstPackedMode mode);

} // namespace vbt
