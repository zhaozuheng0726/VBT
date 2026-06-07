#include "render_temporal_formal.h"

namespace vbt {

RenderTemporalFormalOptions makeRenderTemporalFormalDefaults()
{
    RenderTemporalFormalOptions options;
    options.leafSize = 8;
    options.coarseResolution = 4;
    options.fine6Resolution = 6;
    options.backgroundZeroRatio = 0.30f;
    options.cutoffProtect = true;
    // Current best-known control-sequence setting from the render probes.
    options.controlEpsScale = 8.0f;
    return options;
}

std::vector<RenderTemporalVariantSpec> buildRenderTemporalFormalProbeVariants()
{
    return {
        {"coarse_only", RenderTemporalFormalMode::COARSE_ONLY, false, 0, false, false, false, false, false, false, false, false, false, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f, 0.0f, false, 0.0f, 1.0f, false, 0, 1.0f},
        {"fine6_full", RenderTemporalFormalMode::FINE6_FULL, false, 6, false, false, false, false, false, false, false, false, false, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f, 0.0f, false, 0.0f, 1.0f, false, 0, 1.0f},
        {"fine6_shellsoft20_50", RenderTemporalFormalMode::FINE6_FULL, false, 6, false, false, false, true, false, false, false, false, false, 0, 0, 0.0f, 0.0f, 0.0f, 0.50f, 0.20f, 0.0f, 0.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f, 0.0f, false, 0.0f, 1.0f, false, 0, 1.0f},
        {"fine6_shellsoft_local20_70", RenderTemporalFormalMode::FINE6_FULL, false, 6, false, false, false, true, true, false, false, false, false, 0, 0, 0.0f, 0.0f, 0.0f, 0.70f, 0.20f, 0.0f, 0.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f, 0.0f, false, 0.0f, 1.0f, false, 0, 1.0f},
        {"fine6_shellgain20_60_x4", RenderTemporalFormalMode::FINE6_FULL, false, 6, false, false, false, false, false, true, false, false, false, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.20f, 0.60f, 4.0f, 0.0f, 0, 0.0f, 0.0f, 0.0f, false, 0.0f, 1.0f, false, 0, 1.0f},
        {"fine6_shellgain_guidesmooth20_60_x4", RenderTemporalFormalMode::FINE6_FULL, false, 6, false, false, false, false, false, true, false, true, false, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.20f, 0.60f, 4.0f, 0.18f, 1, 0.0f, 0.0f, 0.0f, false, 0.0f, 1.0f, false, 0, 1.0f},
        {"shellvoxel_band25_100", RenderTemporalFormalMode::FINE6_FULL, false, 6, false, false, false, false, false, false, false, false, true, 0, 0, 0.0f, 0.0f, 0.0f, 1.0f, 0.25f, 0.0f, 0.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f, 0.0f, true, 0.0f, 1.0f, false, 0, 1.0f},
        {"shellvoxel_band25_100_dilate1_35", RenderTemporalFormalMode::FINE6_FULL, false, 6, false, false, false, false, false, false, false, false, true, 0, 0, 0.0f, 0.0f, 0.0f, 1.0f, 0.25f, 0.0f, 0.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f, 0.0f, true, 0.0f, 1.0f, false, 1, 0.35f},
        {"shellvoxel_band25_100_dilate1_50", RenderTemporalFormalMode::FINE6_FULL, false, 6, false, false, false, false, false, false, false, false, true, 0, 0, 0.0f, 0.0f, 0.0f, 1.0f, 0.25f, 0.0f, 0.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f, 0.0f, true, 0.0f, 1.0f, false, 1, 0.50f},
        {"shellvoxel_band25_100_leaksafe040", RenderTemporalFormalMode::FINE6_FULL, false, 6, false, false, false, false, false, false, false, false, true, 0, 0, 0.0f, 0.0f, 0.0f, 1.0f, 0.25f, 0.0f, 0.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f, 0.0f, true, 0.40f, 0.0f, false, 0, 1.0f},
        {"shellvoxel_band25_100_leaksmooth040_25", RenderTemporalFormalMode::FINE6_FULL, false, 6, false, false, false, false, false, false, false, false, true, 0, 0, 0.0f, 0.0f, 0.0f, 1.0f, 0.25f, 0.0f, 0.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f, 0.0f, true, 0.40f, 0.25f, true, 0, 1.0f},
        {"shellvoxel_band25_100_leaksmooth040_50", RenderTemporalFormalMode::FINE6_FULL, false, 6, false, false, false, false, false, false, false, false, true, 0, 0, 0.0f, 0.0f, 0.0f, 1.0f, 0.25f, 0.0f, 0.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f, 0.0f, true, 0.40f, 0.50f, true, 0, 1.0f},
        {"routed_empty_grid4_fine6", RenderTemporalFormalMode::COARSE_ONLY, true, 6, false, false, false, false, false, false, false, false, false, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f, 0.0f, false, 0.0f, 1.0f, false, 0, 1.0f},
        // Compact/local variants stay probe-only until a lighter basis becomes competitive.
        {"tile3_gate_visible20pct", RenderTemporalFormalMode::FINE_COMPACT, false, 0, true, true, false, false, false, false, false, false, false, 4, 3, 0.20f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0, 1000.0f, 1000.0f, 0.0f, false, 0.0f, 1.0f, false, 0, 1.0f},
        {"tile3_gate_visible50pct", RenderTemporalFormalMode::FINE_COMPACT, false, 0, true, true, false, false, false, false, false, false, false, 4, 3, 0.50f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0, 1000.0f, 1000.0f, 0.0f, false, 0.0f, 1.0f, false, 0, 1.0f},
        // Surface-aware local probes for thin shell liquids.
        {"tile3_gate_shell01pct", RenderTemporalFormalMode::FINE_COMPACT, false, 0, true, true, true, false, false, false, false, false, false, 4, 3, 0.0f, 0.01f, 0.45f, 0.35f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0, 1000.0f, 1000.0f, 0.0f, false, 0.0f, 1.0f, false, 0, 1.0f},
        {"tile3_gate_shell03pct", RenderTemporalFormalMode::FINE_COMPACT, false, 0, true, true, true, false, false, false, false, false, false, 4, 3, 0.0f, 0.03f, 0.45f, 0.35f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0, 1000.0f, 1000.0f, 0.0f, false, 0.0f, 1.0f, false, 0, 1.0f},
    };
}

const char* toString(RenderTemporalFormalMode mode)
{
    switch (mode) {
    case RenderTemporalFormalMode::EMPTY:
        return "empty";
    case RenderTemporalFormalMode::COARSE_ONLY:
        return "coarse_only";
    case RenderTemporalFormalMode::FINE_COMPACT:
        return "fine_compact";
    case RenderTemporalFormalMode::FINE6_FULL:
        return "fine6_full";
    default:
        return "unknown";
    }
}

} // namespace vbt
