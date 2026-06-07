#include "render_temporal_route.h"

#include <algorithm>

namespace vbt {

RenderTemporalRouteOptions makeRenderTemporalRouteDefaults(float cutoff, float cutoffBand, float bgZeroRatio)
{
    RenderTemporalRouteOptions options;
    options.cutoff = cutoff;
    options.cutoffBand = cutoffBand;
    options.backgroundZeroRatio = bgZeroRatio;
    options.emptyVisibleFracThreshold = 0.001f;
    // Current best-known probe thresholds from the temporal-first render route.
    options.fineVisibleFracThreshold = 0.005f;
    options.fineBandFracThreshold = 0.005f;
    options.coarseRmseThreshold = 0.008f;
    options.coarsePeakThreshold = 0.03f;
    options.fineGainThreshold = 0.05f;
    options.shellFracThreshold = 0.002f;
    options.coarseShellRmseThreshold = 0.008f;
    options.coarseShellPeakThreshold = 0.03f;
    options.shellGainThreshold = 0.02f;
    return options;
}

RenderTemporalFormalMode routeRenderTemporalLeaf(bool emptyLeaf,
                                                 const RenderTemporalRouteMetrics& metrics,
                                                 const RenderTemporalRouteOptions& options)
{
    (void)options;
    if (emptyLeaf) {
        return RenderTemporalFormalMode::EMPTY;
    }

    const bool coarseTooRough =
        (metrics.coarseRmse > static_cast<double>(options.coarseRmseThreshold)) ||
        (metrics.coarsePeak > static_cast<double>(options.coarsePeakThreshold));

    const bool visibilitySensitive =
        (metrics.visibleFrac >= static_cast<double>(options.fineVisibleFracThreshold)) ||
        (metrics.bandFrac >= static_cast<double>(options.fineBandFracThreshold));

    const double denom = std::max(metrics.coarseRmse, 1e-12);
    const double fineGain = (metrics.coarseRmse - metrics.fine6Rmse) / denom;

    const bool shellSensitive =
        metrics.shellFrac >= static_cast<double>(options.shellFracThreshold);
    const bool coarseShellTooRough =
        (metrics.coarseShellRmse > static_cast<double>(options.coarseShellRmseThreshold)) ||
        (metrics.coarseShellPeak > static_cast<double>(options.coarseShellPeakThreshold));
    const double shellDenom = std::max(metrics.coarseShellRmse, 1e-12);
    const double shellGain = (metrics.coarseShellRmse - metrics.shellVoxelRmse) / shellDenom;

    if (shellSensitive &&
        coarseShellTooRough &&
        shellGain >= static_cast<double>(options.shellGainThreshold)) {
        return RenderTemporalFormalMode::FINE_COMPACT;
    }

    if (coarseTooRough && visibilitySensitive && fineGain >= static_cast<double>(options.fineGainThreshold)) {
        return RenderTemporalFormalMode::FINE6_FULL;
    }

    return RenderTemporalFormalMode::COARSE_ONLY;
}

} // namespace vbt
