#pragma once

#include "render_temporal_formal.h"

namespace vbt {

struct RenderTemporalRouteMetrics {
    double visibleFrac = 0.0;
    double bandFrac = 0.0;
    double shellFrac = 0.0;
    double coarseRmse = 0.0;
    double coarsePeak = 0.0;
    double coarseShellRmse = 0.0;
    double coarseShellPeak = 0.0;
    double fine6Rmse = 0.0;
    double fine6Peak = 0.0;
    double shellVoxelRmse = 0.0;
    double shellVoxelPeak = 0.0;
};

struct RenderTemporalRouteOptions {
    float cutoff = 0.0f;
    float cutoffBand = 0.0f;
    float backgroundZeroRatio = 0.30f;
    float emptyVisibleFracThreshold = 0.001f;
    float fineVisibleFracThreshold = 0.02f;
    float fineBandFracThreshold = 0.01f;
    float coarseRmseThreshold = 0.015f;
    float coarsePeakThreshold = 0.06f;
    float fineGainThreshold = 0.15f;
    float shellFracThreshold = 0.01f;
    float coarseShellRmseThreshold = 0.015f;
    float coarseShellPeakThreshold = 0.05f;
    float shellGainThreshold = 0.10f;
};

RenderTemporalRouteOptions makeRenderTemporalRouteDefaults(float cutoff, float cutoffBand, float bgZeroRatio);

RenderTemporalFormalMode routeRenderTemporalLeaf(bool emptyLeaf,
                                                 const RenderTemporalRouteMetrics& metrics,
                                                 const RenderTemporalRouteOptions& options);

} // namespace vbt
