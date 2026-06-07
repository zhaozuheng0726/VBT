#include "temporal_first_encoder.h"

namespace vbt {

bool isTemporalFirstRenderProfile(const FieldProfile& profile)
{
    return profile.type == FieldType::DENSITY;
}

TemporalFirstEncoderOptions makeTemporalFirstEncoderOptions(const SpatialFirstOptions& options)
{
    TemporalFirstEncoderOptions out;
    out.formal = makeRenderTemporalFormalDefaults();
    out.route = makeRenderTemporalRouteDefaults(options.profile.den.renderCutoff,
                                                options.profile.den.cutoffBand,
                                                options.renderTemporalProbeBgZeroRatio);

    out.formal.leafSize = options.leafSize;
    out.formal.coarseResolution = options.coarseResolution;
    out.formal.backgroundZeroRatio = options.renderTemporalProbeBgZeroRatio;
    out.formal.cutoffProtect = options.renderTemporalProbeCutoffProtect;
    out.formal.controlEpsScale = 8.0f;

    out.cutoff = options.profile.den.renderCutoff;
    out.cutoffBand = options.profile.den.cutoffBand;
    if (options.renderTemporalProbeEpsAbs > 0.0f) {
        out.temporalEpsAbs = options.renderTemporalProbeEpsAbs;
    }
    out.temporalEpsRel = options.renderTemporalProbeEpsRel;
    out.temporalGammaDelta = options.renderTemporalProbeGammaDelta;
    out.cutoffProtect = options.renderTemporalProbeCutoffProtect;

    return out;
}

std::array<TemporalFirstPackedLeafShape, 4> buildTemporalFirstPackedLeafShapes()
{
    return {{
        {TemporalFirstPackedMode::EMPTY, false, false, false, 0, 0},
        // Grid4 is the low-cost mode. It still carries a time-bin index because
        // temporal-first render decode must avoid long per-sample keyframe scans.
        {TemporalFirstPackedMode::TEMPORAL_GRID4, true, true, false, 4, 0},
        // Reserved compact fine slot. Keep the mode and GPU constraints frozen
        // even though there is no winning compact basis yet.
        {TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT, true, true, true, 4, 0},
        {TemporalFirstPackedMode::TEMPORAL_FINE6, true, true, true, 4, 6},
    }};
}

const char* toString(TemporalFirstPackedMode mode)
{
    switch (mode) {
    case TemporalFirstPackedMode::EMPTY:
        return "empty";
    case TemporalFirstPackedMode::TEMPORAL_GRID4:
        return "temporal_grid4";
    case TemporalFirstPackedMode::TEMPORAL_FINE_COMPACT:
        return "temporal_fine_compact";
    case TemporalFirstPackedMode::TEMPORAL_FINE6:
        return "temporal_fine6";
    default:
        return "unknown";
    }
}

} // namespace vbt
