#include "field_profile.h"
#include "raw_volume.h"
#include "render_temporal_mainline.h"
#include "render_temporal_probe.h"
#include "spatial_first_encoder.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

#ifdef VBT_USE_OPENMP
#include <omp.h>
#endif

using namespace vbt;

namespace {

struct SubsetSpec {
    bool enabled = false;
    int x0 = 0;
    int y0 = 0;
    int z0 = 0;
    int sx = 0;
    int sy = 0;
    int sz = 0;
};

std::array<int, 3> parseInt3(const std::string& text)
{
    std::array<int, 3> out{0, 0, 0};
    std::stringstream ss(text);
    std::string token;
    for (int i = 0; i < 3; ++i) {
        if (!std::getline(ss, token, ',')) {
            throw std::runtime_error("Expected x,y,z triple");
        }
        out[static_cast<size_t>(i)] = std::stoi(token);
    }
    return out;
}

void printUsage()
{
    // Keep the CLI grouped by concern: routing/encoding first, then evaluation,
    // then reporting/output switches.
    std::cout
        << "Usage:\n"
        << "  vbt_spatialfirst_probe --input-raw <raw> [--metadata <meta.json>]\n"
        << "                         [--profile generic|density|sdf]\n"
        << "                         [--sample-step 8]\n"
        << "                         [--dct-keep 4]\n"
        << "                         [--event-top-k 128|0(unlimited)]\n"
        << "                         [--event-threshold 0.02]\n"
        << "                         [--event-min-count 3]\n"
        << "                         [--generic-stratified-events]\n"
        << "                         [--split-scientific-render-modes|--no-split-scientific-render-modes]\n"
        << "                         [--generic-adaptive-coarse-keep|--no-generic-adaptive-coarse-keep]\n"
        << "                         [--generic-dense-crossover|--no-generic-dense-crossover] [--generic-dense-grid-res 3] [--generic-dense-bits 4]\n"
        << "                         [--generic-dense-temporal-basis|--no-generic-dense-temporal-basis] [--generic-dense-temporal-keep 4]\n"
        << "                         [--generic-dense-temporal-force|--no-generic-dense-temporal-force]\n"
        << "                         [--generic-dense-temporal-for-grid4|--no-generic-dense-temporal-for-grid4]\n"
        << "                         [--generic-tile-temporal-basis|--no-generic-tile-temporal-basis] [--generic-tile-temporal-keep 4]\n"
        << "                         [--generic-tile-temporal-force|--no-generic-tile-temporal-force]\n"
        << "                         [--generic-patch-temporal-basis|--no-generic-patch-temporal-basis] [--generic-patch-temporal-keep 4]\n"
        << "                         [--generic-patch-temporal-force|--no-generic-patch-temporal-force]\n"
        << "                         [--generic-densegrid4-candidate|--no-generic-densegrid4-candidate] [--generic-densegrid4-rate-scale 1.0] [--generic-densegrid4-dist-threshold 0.3]\n"
        << "                         [--generic-rdo-lambda 1e-5] [--generic-rdo-spatial-stride 2] [--generic-rdo-time-stride 4]\n"
        << "                         [--generic-chaos-adaptive-rdo|--no-generic-chaos-adaptive-rdo]\n"
        << "                         [--generic-chaos-spatial-occ-thr 0.50] [--generic-chaos-time-occ-thr 0.45]\n"
        << "                         [--generic-subgrid-crossover] [--generic-subgrid-bits 4]\n"
        << "                         [--generic-multipatch-crossover] [--generic-multipatch-bits 4]\n"
        << "                         [--generic-tile-crossover] [--generic-tile-bits 4] [--generic-tile-size 2]\n"
          << "                         [--full-eval]\n"
          << "                         [--omp-threads N]\n"
        << "                         [--save-vbt <path>]\n"
        << "                         [--residual-diagnostics] [--residual-diagnostic-blocks 4] [--residual-diagnostic-dir <dir>]\n"
        << "                         [--fine-residual-grid] [--fine-grid-res 6] [--fine-dct-keep 2] [--fine-quant-bits 0|8]\n"
        << "                         [--render-temporal-first-probe] [--render-temporal-probe-eps-abs X] [--render-temporal-probe-eps-rel X]\n"
        << "                         [--render-temporal-probe-gamma-delta X] [--render-temporal-probe-bg-zero-ratio X]\n"
        << "                         [--render-temporal-probe-cutoff-protect|--no-render-temporal-probe-cutoff-protect]\n"
        << "                         [--render-unified-backbone-probe]\n"
        << "                         [--render-compact-fine-grid-res 6] [--render-compact-fine-dct-keep 4] [--render-compact-fine-quant-bits 8]\n"
        << "                         [--render-full-fine-grid-res 8] [--render-full-fine-dct-keep 8] [--render-full-fine-quant-bits 8]\n"
        << "                         [--render-compact-fine-norm-thr 0.20] [--render-compact-fine-peak-thr 0.60]\n"
        << "                         [--full-residual]\n"
        << "                         [--full-residual-bits 2|4]\n"
          << "                         [--render-disable-constant-mode]\n"
          << "                         [--render-norm-thr 0.08]\n"
          << "                         [--generic-norm-thr 0.03] [--generic-peak-thr 0.03]\n"
        << "                         [--generic-energy-amnesty-norm 0.0] [--generic-energy-amnesty-frac 0.0]\n"
        << "                         [--generic-energy-amnesty-peak-scale 1.5] [--generic-energy-amnesty-topn 16]\n"
        << "                         [--routing-stats <path.csv>]\n"
        << "                         [--subset-origin x,y,z] [--subset-size sx,sy,sz]\n"
        << "                         [--cutoff 0.005] [--cutoff-band 0.002]\n"
        << "                         [--iso 128] [--report <path>]\n"
        << "                         [--eval-mask-cutoff X] [--eval-mask-band X]\n"
        << "                         [--compare-temporal-baseline]\n";
}

double percentileFromSorted(const std::vector<uint32_t>& values, double q)
{
    if (values.empty()) return 0.0;
    const double clamped = std::clamp(q, 0.0, 1.0);
    const double pos = clamped * static_cast<double>(values.size() - 1);
    const size_t i0 = static_cast<size_t>(std::floor(pos));
    const size_t i1 = static_cast<size_t>(std::ceil(pos));
    const double t = pos - static_cast<double>(i0);
    const double v0 = static_cast<double>(values[i0]);
    const double v1 = static_cast<double>(values[i1]);
    return v0 + (v1 - v0) * t;
}

void writeReport(const std::filesystem::path& reportPath,
                 const std::filesystem::path& inputRaw,
                 const RawVolume4D& volume,
                 const SpatialFirstOptions& options,
                 const SubsetSpec& subset,
                 const ProbeSummary& summary)
{
    std::ofstream out(reportPath);
    out << "# VBT Spatial-First Hybrid Temporal Probe\n\n";
    out << "- input: `" << inputRaw.string() << "`\n";
    out << "- dimensions: `" << volume.meta.width << "x" << volume.meta.height << "x" << volume.meta.depth << " x " << volume.meta.frames << "`\n";
    out << "- profile: `" << (options.profile.type == FieldType::GENERIC ? "generic" : options.profile.type == FieldType::DENSITY ? "density" : "sdf") << "`\n";
    out << "- DCT keep: `" << options.dctKeep << "`\n";
    out << "- event top-K: `" << options.eventTopK << "`\n";
    out << "- event threshold: `" << options.eventThreshold << "`\n";
    out << "- event min count: `" << options.eventMinCount << "`\n";
    out << "- generic stratified events: `" << (options.genericStratifiedEvents ? "true" : "false") << "`\n";
    out << "- split scientific/render modes: `" << (options.splitScientificRenderModes ? "true" : "false") << "`\n";
    out << "- generic adaptive coarse keep: `" << (options.genericAdaptiveCoarseKeep ? "true" : "false") << "`\n";
    out << "- generic disable constant mode: `" << (options.genericDisableConstantMode ? "true" : "false") << "`\n";
    out << "- event quant bits: `4`\n";
    out << "- generic dense crossover: `" << (options.genericDenseCrossover ? "true" : "false") << "`\n";
    out << "- generic dense grid resolution: `" << options.genericDenseGridResolution << "`\n";
    out << "- generic dense temporal basis: `" << (options.genericDenseTemporalBasisCandidate ? "true" : "false") << "`\n";
    out << "- generic dense temporal force: `" << (options.genericDenseTemporalBasisForce ? "true" : "false") << "`\n";
    out << "- generic dense temporal keep: `" << options.genericDenseTemporalKeep << "`\n";
    out << "- generic dense temporal for grid4: `" << (options.genericDenseTemporalForGrid4 ? "true" : "false") << "`\n";
    out << "- generic tile temporal basis: `" << (options.genericTileTemporalBasisCandidate ? "true" : "false") << "`\n";
    out << "- generic tile temporal force: `" << (options.genericTileTemporalBasisForce ? "true" : "false") << "`\n";
    out << "- generic tile temporal keep: `" << options.genericTileTemporalKeep << "`\n";
    out << "- generic patch temporal basis: `" << (options.genericPatchTemporalBasisCandidate ? "true" : "false") << "`\n";
    out << "- generic patch temporal force: `" << (options.genericPatchTemporalBasisForce ? "true" : "false") << "`\n";
    out << "- generic patch temporal keep: `" << options.genericPatchTemporalKeep << "`\n";
    out << "- generic densegrid4 candidate: `" << (options.genericDenseGrid4Candidate ? "true" : "false") << "`\n";
    out << "- generic densegrid4 rate scale: `" << options.genericDenseGrid4RateScale << "`\n";
    out << "- generic densegrid4 dist threshold: `" << options.genericDenseGrid4DistThreshold << "`\n";
    out << "- generic dense bits: `" << options.genericDenseResidualBits << "`\n";
    out << "- generic RDO lambda: `" << options.genericRdoLambda << "`\n";
    out << "- generic RDO stride xyz/t: `" << options.genericRdoSpatialStride << "," << options.genericRdoTimeStride << "`\n";
    out << "- generic RDO p99 weight: `" << options.genericRdoP99Weight << "`\n";
    out << "- generic RDO peak weight: `" << options.genericRdoPeakWeight << "`\n";
    out << "- generic RDO envelope: `" << (options.genericRdoUseMaxEnvelope ? "max" : "sum") << "`\n";
    out << "- generic chaos adaptive RDO: `" << (options.genericChaosAdaptiveRdo ? "true" : "false") << "`\n";
    out << "- generic chaos spatial occupancy threshold: `" << options.genericChaosSpatialOccThreshold << "`\n";
    out << "- generic chaos time occupancy threshold: `" << options.genericChaosTimeOccThreshold << "`\n";
    out << "- generic subgrid crossover: `" << (options.genericSubgridCrossover ? "true" : "false") << "`\n";
    out << "- generic subgrid bits: `" << options.genericSubgridBits << "`\n";
    out << "- generic multipatch crossover: `" << (options.genericMultipatchCrossover ? "true" : "false") << "`\n";
    out << "- generic multipatch bits: `" << options.genericMultipatchBits << "`\n";
    out << "- generic tile crossover: `" << (options.genericTileCrossover ? "true" : "false") << "`\n";
    out << "- generic tile bits: `" << options.genericTileBits << "`\n";
    out << "- generic tile size: `" << options.genericTileSize << "`\n";
    out << "- fine residual grid for render profiles: `" << (options.fineResidualGridForRenderProfiles ? "true" : "false") << "`\n";
    out << "- fine grid resolution: `" << options.fineGridResolution << "`\n";
    out << "- fine grid DCT keep: `" << options.fineGridDctKeep << "`\n";
    out << "- fine quant bits: `" << options.fineQuantBits << "`\n";
    out << "- render unified backbone probe: `" << (options.renderUnifiedBackboneProbe ? "true" : "false") << "`\n";
    out << "- render compact fine grid resolution: `" << options.renderCompactFineGridResolution << "`\n";
    out << "- render compact fine DCT keep: `" << options.renderCompactFineDctKeep << "`\n";
    out << "- render compact fine quant bits: `" << options.renderCompactFineQuantBits << "`\n";
    out << "- render full fine grid resolution: `" << options.renderFullFineGridResolution << "`\n";
    out << "- render full fine DCT keep: `" << options.renderFullFineDctKeep << "`\n";
    out << "- render full fine quant bits: `" << options.renderFullFineQuantBits << "`\n";
    out << "- render compact fine norm threshold: `" << options.renderCompactFineNormErrThreshold << "`\n";
    out << "- render compact fine peak threshold: `" << options.renderCompactFinePeakErrThreshold << "`\n";
    out << "- full residual for render profiles: `" << (options.fullResidualForRenderProfiles ? "true" : "false") << "`\n";
    out << "- full residual bits: `" << options.fullResidualBits << "`\n";
    out << "- render disable constant mode: `" << (options.renderDisableConstantMode ? "true" : "false") << "`\n";
    out << "- render coarse-only norm threshold: `" << options.routeCoarseOnlyNormErrRender << "`\n";
    out << "- generic norm threshold: `" << options.routeCoarseOnlyNormErrGeneric << "`\n";
    out << "- generic peak threshold: `" << options.routeCoarseOnlyPeakErrGeneric << "`\n";
    out << "- generic energy amnesty norm: `" << options.genericEnergyAmnestyNorm << "`\n";
    out << "- generic energy amnesty hot frac: `" << options.genericEnergyAmnestyHotFrac << "`\n";
    out << "- generic energy amnesty peak scale: `" << options.genericEnergyAmnestyPeakScale << "`\n";
    out << "- generic energy amnesty top-N: `" << options.genericEnergyAmnestyTopN << "`\n";
    out << "- subset: `" << (subset.enabled
        ? (std::to_string(subset.x0) + "," + std::to_string(subset.y0) + "," + std::to_string(subset.z0) +
           " / " + std::to_string(subset.sx) + "," + std::to_string(subset.sy) + "," + std::to_string(subset.sz))
        : std::string("full")) << "`\n";
    // Reports always distinguish encoding scope from evaluation scope: the
    // payload is for the full encoded volume, while PSNR may be sampled or exact.
    out << "- evaluation: `" << (options.fullEvaluation ? "full-voxel" : "sampled") << "`\n";
    out << "- omp threads: `" << options.ompThreads << "`\n";
    out << "- residual diagnostics: `" << (options.residualDiagnostics ? "true" : "false") << "`\n";
    out << "- residual diagnostic blocks: `" << options.residualDiagnosticBlocks << "`\n";
    if (!options.residualDiagnosticDir.empty()) {
        out << "- residual diagnostic dir: `" << options.residualDiagnosticDir << "`\n";
    }
    out << "- sample step: `" << options.sampleStepX << "," << options.sampleStepY << "," << options.sampleStepZ << "," << options.sampleStepT << "`\n";
    out << "- eval mask cutoff: `" << options.evalMaskCutoff << "`\n";
    out << "- eval mask band: `" << options.evalMaskBand << "`\n";
    out << "- estimated payload bytes: `" << summary.estimatedBytes << "`\n";
    out << "- saved probe file bytes: `" << summary.savedFileBytes << "`\n";
    out << "- offset table words: `" << summary.offsetTableWords << "`\n";
    out << "- payload words: `" << summary.payloadWords << "`\n";
    out << "- total words: `" << summary.totalWords << "`\n";
    out << "- evaluated samples: `" << summary.evaluatedSamples << "`\n";
    out << "- voxel RMSE: `" << summary.rmse << "`\n";
    if (summary.maskedEvaluatedSamples > 0) {
        out << "- cutoff-masked samples: `" << summary.maskedEvaluatedSamples << "`\n";
        out << "- cutoff-masked threshold: `" << summary.maskedThreshold << "`\n";
        out << "- cutoff-masked RMSE: `" << summary.maskedRmse << "`\n";
        out << "- cutoff-masked PSNR: `" << summary.maskedPsnr << " dB`\n";
    }
    if (summary.p999Valid) {
        out << "- voxel P99" << (summary.tailMetricsApproximate ? " (hist)" : "") << ": `" << summary.p99 << "`\n";
        out << "- voxel P99.9" << (summary.tailMetricsApproximate ? " (hist)" : "") << ": `" << summary.p999 << "`\n";
        out << "- voxel MaxAbs" << (summary.tailMetricsApproximate ? " (exact)" : "") << ": `" << summary.maxAbsError << "`\n";
    }
    out << "- voxel PSNR: `" << summary.psnr << " dB`\n";
    if (summary.pairedCoarsePsnr > 0.0) {
        out << "- paired coarse RMSE: `" << summary.pairedCoarseRmse << "`\n";
        out << "- paired coarse PSNR: `" << summary.pairedCoarsePsnr << " dB`\n";
    }
    if (options.compareTemporalBaseline) {
        out << "- temporal baseline sampled RMSE: `" << summary.baselineRmse << "`\n";
        out << "- temporal baseline sampled PSNR: `" << summary.baselinePsnr << " dB`\n";
        out << "- temporal baseline sampled keyframes: `" << summary.baselineSampledKeyframes << "`\n";
    }
    out << "- coarse coefficient count: `" << summary.coarseCoefficientCount << "`\n";
    out << "- fine coefficient count: `" << summary.fineCoefficientCount << "`\n";
    out << "- event count: `" << summary.eventCount << "`\n";
    out << "- event payload bytes: `" << summary.eventPayloadBytes << "`\n";
    out << "- leaf count: `" << summary.leafCount << "`\n";
    out << "- mode0 blocks: `" << summary.mode0Count << "`\n";
    out << "- mode1 blocks: `" << summary.mode1Count << "`\n";
    out << "- mode2 blocks: `" << summary.mode2Count << "`\n";
    out << "- mode3 blocks: `" << summary.mode3Count << "`\n";
    out << "- render compact fine blocks: `" << summary.renderCompactFineCount << "`\n";
    out << "- render full fine blocks: `" << summary.renderFullFineCount << "`\n";
    if (options.profile.type == FieldType::GENERIC) {
        out << "- scientific coarse keep counts [4,6,8,10,12,14,15,16,other]: `"
            << summary.scientificCoarseKeepCounts[0] << ", "
            << summary.scientificCoarseKeepCounts[1] << ", "
            << summary.scientificCoarseKeepCounts[2] << ", "
            << summary.scientificCoarseKeepCounts[3] << ", "
            << summary.scientificCoarseKeepCounts[4] << ", "
            << summary.scientificCoarseKeepCounts[5] << ", "
            << summary.scientificCoarseKeepCounts[6] << ", "
            << summary.scientificCoarseKeepCounts[7] << ", "
            << summary.scientificCoarseKeepOtherCount << "`\n";
    }
    out << "- mode1 energy amnesty blocks: `" << summary.mode1EnergyAmnestyCount << "`\n";
    out << "- mode1 small-event fallback blocks: `" << summary.mode1EventFallbackCount << "`\n";
    out << "- mode2 dense crossover blocks: `" << summary.mode2DenseCrossoverCount << "`\n";
    out << "- mode2 densegrid3 crossover blocks: `" << summary.mode2DenseGrid3Count << "`\n";
    out << "- mode2 densegrid4 crossover blocks: `" << summary.mode2DenseGrid4Count << "`\n";
    out << "- mode2 dense temporal basis blocks: `" << summary.mode2DenseTemporalBasisCount << "`\n";
    out << "- mode2 tile temporal basis blocks: `" << summary.mode2TileTemporalBasisCount << "`\n";
    out << "- mode2 patch temporal basis blocks: `" << summary.mode2PatchTemporalBasisCount << "`\n";
    out << "- generic chaos-adaptive RDO blocks: `" << summary.genericChaosRdoCount << "`\n";
    out << "- generic chaos-adaptive dense wins: `" << summary.genericChaosDenseWins << "`\n";
    out << "- mode2 multipatch crossover blocks: `" << summary.mode2MultipatchCrossoverCount << "`\n";
    out << "- mode2 tile crossover blocks: `" << summary.mode2TileCrossoverCount << "`\n";
    out << "- generic subgrid candidate blocks: `" << summary.genericSubgridCandidateCount << "`\n";
    out << "- generic subgrid cheaper blocks: `" << summary.genericSubgridCheaperCount << "`\n";
    if (!summary.mode2EventCounts.empty()) {
        auto sorted = summary.mode2EventCounts;
        if (options.eventTopK > 0) {
            const uint32_t topCap = static_cast<uint32_t>(options.eventTopK);
            for (auto& k : sorted) {
                k = std::max<uint32_t>(1u, std::min<uint32_t>(k, topCap));
            }
        }
        std::sort(sorted.begin(), sorted.end());
        const double mean = static_cast<double>(summary.eventCount) / static_cast<double>(sorted.size());
        const auto p50 = percentileFromSorted(sorted, 0.50);
        const auto p90 = percentileFromSorted(sorted, 0.90);
        const auto p95 = percentileFromSorted(sorted, 0.95);
        const auto p99 = percentileFromSorted(sorted, 0.99);
        const uint32_t minK = sorted.front();
        const uint32_t maxK = sorted.back();
        std::array<uint64_t, 8> bins{0, 0, 0, 0, 0, 0, 0, 0};
        for (uint32_t k : sorted) {
            if (k == 0) bins[0] += 1;
            else if (k <= 32) bins[1] += 1;
            else if (k <= 64) bins[2] += 1;
            else if (k <= 128) bins[3] += 1;
            else if (k <= 256) bins[4] += 1;
            else if (k <= 512) bins[5] += 1;
            else if (k <= 1024) bins[6] += 1;
            else bins[7] += 1;
        }
        out << "- mode2 K min/mean/p50/p90/p95/p99/max: `"
            << minK << " / "
            << mean << " / "
            << p50 << " / "
            << p90 << " / "
            << p95 << " / "
            << p99 << " / "
            << maxK << "`\n";
        out << "- mode2 K bins [0,1-32,33-64,65-128,129-256,257-512,513-1024,1025+]: `"
            << bins[0] << ", "
            << bins[1] << ", "
            << bins[2] << ", "
            << bins[3] << ", "
            << bins[4] << ", "
            << bins[5] << ", "
            << bins[6] << ", "
            << bins[7] << "`\n";
    }
    if (!summary.mode2UniqueSpatialCounts.empty()) {
        auto sorted = summary.mode2UniqueSpatialCounts;
        std::sort(sorted.begin(), sorted.end());
        const double sumUniqueSpatial = std::accumulate(sorted.begin(), sorted.end(), 0.0);
        const double mean = sumUniqueSpatial / static_cast<double>(sorted.size());
        const double p50 = percentileFromSorted(std::vector<uint32_t>(sorted.begin(), sorted.end()), 0.50);
        const double p90 = percentileFromSorted(std::vector<uint32_t>(sorted.begin(), sorted.end()), 0.90);
        const double p95 = percentileFromSorted(std::vector<uint32_t>(sorted.begin(), sorted.end()), 0.95);
        const double p99 = percentileFromSorted(std::vector<uint32_t>(sorted.begin(), sorted.end()), 0.99);
        const uint32_t minCount = sorted.front();
        const uint32_t maxCount = sorted.back();
        std::array<uint64_t, 8> bins{0, 0, 0, 0, 0, 0, 0, 0};
        for (uint16_t n : sorted) {
            if (n == 0) bins[0] += 1;
            else if (n <= 8) bins[1] += 1;
            else if (n <= 16) bins[2] += 1;
            else if (n <= 32) bins[3] += 1;
            else if (n <= 64) bins[4] += 1;
            else if (n <= 128) bins[5] += 1;
            else if (n <= 256) bins[6] += 1;
            else bins[7] += 1;
        }
        const double meanOccupancy = mean / 512.0;
        const double meanEventsPerSpatial = summary.eventCount / std::max(1.0, sumUniqueSpatial);
        out << "- mode2 unique spatial voxels min/mean/p50/p90/p95/p99/max: `"
            << minCount << " / "
            << mean << " / "
            << p50 << " / "
            << p90 << " / "
            << p95 << " / "
            << p99 << " / "
            << maxCount << "`\n";
        out << "- mode2 unique spatial occupancy mean: `" << meanOccupancy << "`\n";
        out << "- mode2 mean events per active spatial voxel: `" << meanEventsPerSpatial << "`\n";
        out << "- mode2 unique spatial bins [0,1-8,9-16,17-32,33-64,65-128,129-256,257-512]: `"
            << bins[0] << ", "
            << bins[1] << ", "
            << bins[2] << ", "
            << bins[3] << ", "
            << bins[4] << ", "
            << bins[5] << ", "
            << bins[6] << ", "
            << bins[7] << "`\n";
    }
    if (!summary.mode2UniqueTimeCounts.empty() &&
        summary.mode2UniqueTimeCounts.size() == summary.mode2EncodedFrameCounts.size()) {
        auto sorted = summary.mode2UniqueTimeCounts;
        std::sort(sorted.begin(), sorted.end());
        std::vector<uint32_t> sorted32(sorted.begin(), sorted.end());
        const double sumUniqueTime = std::accumulate(sorted.begin(), sorted.end(), 0.0);
        const double mean = sumUniqueTime / static_cast<double>(sorted.size());
        double sumTimeOccupancy = 0.0;
        for (size_t i = 0; i < summary.mode2UniqueTimeCounts.size(); ++i) {
            const double denom = std::max<int>(1, summary.mode2EncodedFrameCounts[i]);
            sumTimeOccupancy += static_cast<double>(summary.mode2UniqueTimeCounts[i]) / denom;
        }
        const double meanTimeOccupancy = sumTimeOccupancy / static_cast<double>(summary.mode2UniqueTimeCounts.size());
        const double meanEventsPerTime = summary.eventCount / std::max(1.0, sumUniqueTime);
        const auto p50 = percentileFromSorted(sorted32, 0.50);
        const auto p90 = percentileFromSorted(sorted32, 0.90);
        const auto p95 = percentileFromSorted(sorted32, 0.95);
        const auto p99 = percentileFromSorted(sorted32, 0.99);
        const uint32_t minCount = sorted.front();
        const uint32_t maxCount = sorted.back();
        std::array<uint64_t, 8> bins{0, 0, 0, 0, 0, 0, 0, 0};
        for (uint16_t n : sorted) {
            if (n == 0) bins[0] += 1;
            else if (n <= 4) bins[1] += 1;
            else if (n <= 8) bins[2] += 1;
            else if (n <= 16) bins[3] += 1;
            else if (n <= 32) bins[4] += 1;
            else if (n <= 64) bins[5] += 1;
            else if (n <= 96) bins[6] += 1;
            else bins[7] += 1;
        }
        out << "- mode2 unique time frames min/mean/p50/p90/p95/p99/max: `"
            << minCount << " / "
            << mean << " / "
            << p50 << " / "
            << p90 << " / "
            << p95 << " / "
            << p99 << " / "
            << maxCount << "`\n";
        out << "- mode2 unique time occupancy mean: `" << meanTimeOccupancy << "`\n";
        out << "- mode2 mean events per active time frame: `" << meanEventsPerTime << "`\n";
        out << "- mode2 unique time bins [0,1-4,5-8,9-16,17-32,33-64,65-96,97+]: `"
            << bins[0] << ", "
            << bins[1] << ", "
            << bins[2] << ", "
            << bins[3] << ", "
            << bins[4] << ", "
            << bins[5] << ", "
            << bins[6] << ", "
            << bins[7] << "`\n";
    }
    if (!summary.genericSubgridVoxelCounts.empty()) {
        auto vox = summary.genericSubgridVoxelCounts;
        std::sort(vox.begin(), vox.end());
        const double meanVox = std::accumulate(vox.begin(), vox.end(), 0.0) / static_cast<double>(vox.size());
        out << "- generic subgrid voxels min/mean/p50/p90/p95/p99/max: `"
            << vox.front() << " / "
            << meanVox << " / "
            << percentileFromSorted(vox, 0.50) << " / "
            << percentileFromSorted(vox, 0.90) << " / "
            << percentileFromSorted(vox, 0.95) << " / "
            << percentileFromSorted(vox, 0.99) << " / "
            << vox.back() << "`\n";
    }
    if (!summary.genericSubgridFrameCounts.empty()) {
        std::vector<uint32_t> frames(summary.genericSubgridFrameCounts.begin(), summary.genericSubgridFrameCounts.end());
        std::sort(frames.begin(), frames.end());
        const double meanFrames = std::accumulate(frames.begin(), frames.end(), 0.0) / static_cast<double>(frames.size());
        out << "- generic subgrid dt min/mean/p50/p90/p95/p99/max: `"
            << frames.front() << " / "
            << meanFrames << " / "
            << percentileFromSorted(frames, 0.50) << " / "
            << percentileFromSorted(frames, 0.90) << " / "
            << percentileFromSorted(frames, 0.95) << " / "
            << percentileFromSorted(frames, 0.99) << " / "
            << frames.back() << "`\n";
    }
    if (!summary.genericSubgridBytes.empty() && !summary.genericSparseBytes.empty()) {
        auto denseBytes = summary.genericSubgridBytes;
        auto sparseBytes = summary.genericSparseBytes;
        std::sort(denseBytes.begin(), denseBytes.end());
        std::sort(sparseBytes.begin(), sparseBytes.end());
        const double meanDense = std::accumulate(denseBytes.begin(), denseBytes.end(), 0.0) / static_cast<double>(denseBytes.size());
        const double meanSparse = std::accumulate(sparseBytes.begin(), sparseBytes.end(), 0.0) / static_cast<double>(sparseBytes.size());
        out << "- generic subgrid bytes min/mean/p50/p90/p95/p99/max: `"
            << denseBytes.front() << " / "
            << meanDense << " / "
            << percentileFromSorted(denseBytes, 0.50) << " / "
            << percentileFromSorted(denseBytes, 0.90) << " / "
            << percentileFromSorted(denseBytes, 0.95) << " / "
            << percentileFromSorted(denseBytes, 0.99) << " / "
            << denseBytes.back() << "`\n";
        out << "- generic sparse bytes min/mean/p50/p90/p95/p99/max: `"
            << sparseBytes.front() << " / "
            << meanSparse << " / "
            << percentileFromSorted(sparseBytes, 0.50) << " / "
            << percentileFromSorted(sparseBytes, 0.90) << " / "
            << percentileFromSorted(sparseBytes, 0.95) << " / "
            << percentileFromSorted(sparseBytes, 0.99) << " / "
            << sparseBytes.back() << "`\n";
    }
    if (summary.sparseDiagSamples > 0) {
        out << "- sparse diag samples: `" << summary.sparseDiagSamples << "`\n";
        out << "- sparse diag coarse RMSE: `" << summary.sparseDiagCoarseRmse << "`\n";
        out << "- sparse diag coarse PSNR: `" << summary.sparseDiagCoarsePsnr << " dB`\n";
        out << "- sparse diag mode RMSE: `" << summary.sparseDiagModeRmse << "`\n";
        out << "- sparse diag mode PSNR: `" << summary.sparseDiagModePsnr << " dB`\n";
        out << "- sparse diag improved samples: `" << summary.sparseDiagImproved << "`\n";
        out << "- sparse diag worsened samples: `" << summary.sparseDiagWorsened << "`\n";
    }
}

void writeRoutingStatsCsv(const std::filesystem::path& csvPath, const ProbeSummary& summary)
{
    std::ofstream out(csvPath);
    out << "bx,by,bz,mode,active_frames,leaf_range,coarse_rmse,norm_err,peak_err_norm,top_energy_frac\n";
    out << std::setprecision(10);
    for (const auto& row : summary.routeStats) {
        out << row.bx << ','
            << row.by << ','
            << row.bz << ','
            << static_cast<int>(row.mode) << ','
            << row.activeFrames << ','
            << row.leafRange << ','
            << row.coarseRmse << ','
            << row.normErr << ','
            << row.peakErrNorm << ','
            << row.topEnergyFrac << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    std::filesystem::path inputRaw;
    std::filesystem::path metadataPath;
    std::filesystem::path reportPath;
    std::filesystem::path routingStatsPath;
    std::string profileName = "generic";
    SpatialFirstOptions options;
    SubsetSpec subset;
    bool warnResidualTopAlias = false;
    bool warnResidualKfDeprecated = false;
    bool warnResidualCellDeprecated = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        // Parsing is intentionally flat: routing/encoding knobs are forwarded
        // directly into SpatialFirstOptions so probe runs remain easy to script.
        if (arg == "--input-raw" && i + 1 < argc) inputRaw = argv[++i];
        else if (arg == "--metadata" && i + 1 < argc) metadataPath = argv[++i];
        else if (arg == "--profile" && i + 1 < argc) profileName = argv[++i];
        else if (arg == "--sample-step" && i + 1 < argc) {
            const int s = std::stoi(argv[++i]);
            options.sampleStepX = s;
            options.sampleStepY = s;
            options.sampleStepZ = s;
            options.sampleStepT = s;
        } else if (arg == "--dct-keep" && i + 1 < argc) options.dctKeep = std::stoi(argv[++i]);
        else if (arg == "--event-top-k" && i + 1 < argc) options.eventTopK = std::stoi(argv[++i]);
        else if (arg == "--event-quant-bits" && i + 1 < argc) {
            ++i;
            std::cerr << "[warn] --event-quant-bits is deprecated; Mode 2 now always uses 4-bit nibble residuals.\n";
        }
        else if (arg == "--event-threshold" && i + 1 < argc) options.eventThreshold = std::stof(argv[++i]);
        else if (arg == "--event-min-count" && i + 1 < argc) options.eventMinCount = std::stoi(argv[++i]);
        else if (arg == "--generic-stratified-events") options.genericStratifiedEvents = true;
        else if (arg == "--split-scientific-render-modes") options.splitScientificRenderModes = true;
        else if (arg == "--no-split-scientific-render-modes") options.splitScientificRenderModes = false;
        else if (arg == "--generic-adaptive-coarse-keep") options.genericAdaptiveCoarseKeep = true;
        else if (arg == "--no-generic-adaptive-coarse-keep") options.genericAdaptiveCoarseKeep = false;
        else if (arg == "--generic-dense-crossover") options.genericDenseCrossover = true;
        else if (arg == "--no-generic-dense-crossover") options.genericDenseCrossover = false;
        else if (arg == "--generic-dense-grid-res" && i + 1 < argc) options.genericDenseGridResolution = std::stoi(argv[++i]);
        else if (arg == "--generic-dense-temporal-basis") options.genericDenseTemporalBasisCandidate = true;
        else if (arg == "--no-generic-dense-temporal-basis") options.genericDenseTemporalBasisCandidate = false;
        else if (arg == "--generic-dense-temporal-force") options.genericDenseTemporalBasisForce = true;
        else if (arg == "--no-generic-dense-temporal-force") options.genericDenseTemporalBasisForce = false;
        else if (arg == "--generic-dense-temporal-keep" && i + 1 < argc) options.genericDenseTemporalKeep = std::stoi(argv[++i]);
        else if (arg == "--generic-dense-temporal-for-grid4") options.genericDenseTemporalForGrid4 = true;
        else if (arg == "--no-generic-dense-temporal-for-grid4") options.genericDenseTemporalForGrid4 = false;
        else if (arg == "--generic-tile-temporal-basis") options.genericTileTemporalBasisCandidate = true;
        else if (arg == "--no-generic-tile-temporal-basis") options.genericTileTemporalBasisCandidate = false;
        else if (arg == "--generic-tile-temporal-force") options.genericTileTemporalBasisForce = true;
        else if (arg == "--no-generic-tile-temporal-force") options.genericTileTemporalBasisForce = false;
        else if (arg == "--generic-tile-temporal-keep" && i + 1 < argc) options.genericTileTemporalKeep = std::stoi(argv[++i]);
        else if (arg == "--generic-patch-temporal-basis") options.genericPatchTemporalBasisCandidate = true;
        else if (arg == "--no-generic-patch-temporal-basis") options.genericPatchTemporalBasisCandidate = false;
        else if (arg == "--generic-patch-temporal-force") options.genericPatchTemporalBasisForce = true;
        else if (arg == "--no-generic-patch-temporal-force") options.genericPatchTemporalBasisForce = false;
        else if (arg == "--generic-patch-temporal-keep" && i + 1 < argc) options.genericPatchTemporalKeep = std::stoi(argv[++i]);
        else if (arg == "--generic-densegrid4-candidate") options.genericDenseGrid4Candidate = true;
        else if (arg == "--no-generic-densegrid4-candidate") options.genericDenseGrid4Candidate = false;
        else if (arg == "--generic-densegrid4-rate-scale" && i + 1 < argc) options.genericDenseGrid4RateScale = std::stof(argv[++i]);
        else if (arg == "--generic-densegrid4-dist-threshold" && i + 1 < argc) options.genericDenseGrid4DistThreshold = std::stof(argv[++i]);
        else if (arg == "--generic-dense-bits" && i + 1 < argc) options.genericDenseResidualBits = std::stoi(argv[++i]);
        else if (arg == "--generic-rdo-lambda" && i + 1 < argc) options.genericRdoLambda = std::stof(argv[++i]);
        else if (arg == "--generic-rdo-spatial-stride" && i + 1 < argc) options.genericRdoSpatialStride = std::stoi(argv[++i]);
        else if (arg == "--generic-rdo-time-stride" && i + 1 < argc) options.genericRdoTimeStride = std::stoi(argv[++i]);
        else if (arg == "--generic-rdo-p99-weight" && i + 1 < argc) options.genericRdoP99Weight = std::stof(argv[++i]);
        else if (arg == "--generic-rdo-peak-weight" && i + 1 < argc) options.genericRdoPeakWeight = std::stof(argv[++i]);
        else if (arg == "--generic-rdo-sum") options.genericRdoUseMaxEnvelope = false;
        else if (arg == "--generic-rdo-max-envelope") options.genericRdoUseMaxEnvelope = true;
        else if (arg == "--generic-chaos-adaptive-rdo") options.genericChaosAdaptiveRdo = true;
        else if (arg == "--no-generic-chaos-adaptive-rdo") options.genericChaosAdaptiveRdo = false;
        else if (arg == "--generic-chaos-spatial-occ-thr" && i + 1 < argc) options.genericChaosSpatialOccThreshold = std::stof(argv[++i]);
        else if (arg == "--generic-chaos-time-occ-thr" && i + 1 < argc) options.genericChaosTimeOccThreshold = std::stof(argv[++i]);
        else if (arg == "--generic-subgrid-crossover") options.genericSubgridCrossover = true;
        else if (arg == "--generic-subgrid-bits" && i + 1 < argc) options.genericSubgridBits = std::stoi(argv[++i]);
        else if (arg == "--generic-multipatch-crossover") options.genericMultipatchCrossover = true;
        else if (arg == "--generic-multipatch-bits" && i + 1 < argc) options.genericMultipatchBits = std::stoi(argv[++i]);
        else if (arg == "--generic-tile-crossover") options.genericTileCrossover = true;
        else if (arg == "--generic-tile-bits" && i + 1 < argc) options.genericTileBits = std::stoi(argv[++i]);
        else if (arg == "--generic-tile-size" && i + 1 < argc) options.genericTileSize = std::stoi(argv[++i]);
        else if (arg == "--full-eval") options.fullEvaluation = true;
        else if (arg == "--omp-threads" && i + 1 < argc) options.ompThreads = std::stoi(argv[++i]);
        else if (arg == "--save-vbt" && i + 1 < argc) options.saveVbtPath = argv[++i];
        else if (arg == "--residual-diagnostics") options.residualDiagnostics = true;
        else if (arg == "--residual-diagnostic-blocks" && i + 1 < argc) options.residualDiagnosticBlocks = std::stoi(argv[++i]);
        else if (arg == "--residual-diagnostic-dir" && i + 1 < argc) options.residualDiagnosticDir = argv[++i];
        else if (arg == "--residual-top" && i + 1 < argc) {
            options.eventTopK = std::stoi(argv[++i]);
            warnResidualTopAlias = true;
        }
        else if (arg == "--residual-kf" && i + 1 < argc) {
            ++i;
            warnResidualKfDeprecated = true;
        }
        else if (arg == "--residual-cell-size" && i + 1 < argc) {
            ++i;
            warnResidualCellDeprecated = true;
        }
        else if (arg == "--fine-residual-grid") options.fineResidualGridForRenderProfiles = true;
        else if (arg == "--fine-grid-res" && i + 1 < argc) options.fineGridResolution = std::stoi(argv[++i]);
        else if (arg == "--fine-dct-keep" && i + 1 < argc) options.fineGridDctKeep = std::stoi(argv[++i]);
        else if (arg == "--fine-quant-bits" && i + 1 < argc) options.fineQuantBits = std::stoi(argv[++i]);
        else if (arg == "--render-temporal-first-probe") options.renderTemporalFirstProbe = true;
        else if (arg == "--render-temporal-probe-eps-abs" && i + 1 < argc) options.renderTemporalProbeEpsAbs = std::stof(argv[++i]);
        else if (arg == "--render-temporal-probe-eps-rel" && i + 1 < argc) options.renderTemporalProbeEpsRel = std::stof(argv[++i]);
        else if (arg == "--render-temporal-probe-gamma-delta" && i + 1 < argc) options.renderTemporalProbeGammaDelta = std::stof(argv[++i]);
        else if (arg == "--render-temporal-probe-bg-zero-ratio" && i + 1 < argc) options.renderTemporalProbeBgZeroRatio = std::stof(argv[++i]);
        else if (arg == "--render-temporal-probe-cutoff-protect") options.renderTemporalProbeCutoffProtect = true;
        else if (arg == "--no-render-temporal-probe-cutoff-protect") options.renderTemporalProbeCutoffProtect = false;
        else if (arg == "--render-unified-backbone-probe") options.renderUnifiedBackboneProbe = true;
        else if (arg == "--render-compact-fine-grid-res" && i + 1 < argc) options.renderCompactFineGridResolution = std::stoi(argv[++i]);
        else if (arg == "--render-compact-fine-dct-keep" && i + 1 < argc) options.renderCompactFineDctKeep = std::stoi(argv[++i]);
        else if (arg == "--render-compact-fine-quant-bits" && i + 1 < argc) options.renderCompactFineQuantBits = std::stoi(argv[++i]);
        else if (arg == "--render-full-fine-grid-res" && i + 1 < argc) options.renderFullFineGridResolution = std::stoi(argv[++i]);
        else if (arg == "--render-full-fine-dct-keep" && i + 1 < argc) options.renderFullFineDctKeep = std::stoi(argv[++i]);
        else if (arg == "--render-full-fine-quant-bits" && i + 1 < argc) options.renderFullFineQuantBits = std::stoi(argv[++i]);
        else if (arg == "--render-compact-fine-norm-thr" && i + 1 < argc) options.renderCompactFineNormErrThreshold = std::stof(argv[++i]);
        else if (arg == "--render-compact-fine-peak-thr" && i + 1 < argc) options.renderCompactFinePeakErrThreshold = std::stof(argv[++i]);
        else if (arg == "--full-residual") options.fullResidualForRenderProfiles = true;
        else if (arg == "--full-residual-bits" && i + 1 < argc) options.fullResidualBits = std::stoi(argv[++i]);
        else if (arg == "--render-disable-constant-mode") options.renderDisableConstantMode = true;
        else if (arg == "--render-norm-thr" && i + 1 < argc) options.routeCoarseOnlyNormErrRender = std::stof(argv[++i]);
        else if (arg == "--generic-norm-thr" && i + 1 < argc) options.routeCoarseOnlyNormErrGeneric = std::stof(argv[++i]);
        else if (arg == "--generic-peak-thr" && i + 1 < argc) options.routeCoarseOnlyPeakErrGeneric = std::stof(argv[++i]);
        else if (arg == "--generic-energy-amnesty-norm" && i + 1 < argc) options.genericEnergyAmnestyNorm = std::stof(argv[++i]);
        else if (arg == "--generic-energy-amnesty-frac" && i + 1 < argc) options.genericEnergyAmnestyHotFrac = std::stof(argv[++i]);
        else if (arg == "--generic-energy-amnesty-peak-scale" && i + 1 < argc) options.genericEnergyAmnestyPeakScale = std::stof(argv[++i]);
        else if (arg == "--generic-energy-amnesty-topn" && i + 1 < argc) options.genericEnergyAmnestyTopN = std::stoi(argv[++i]);
        else if (arg == "--routing-stats" && i + 1 < argc) { routingStatsPath = argv[++i]; options.dumpRoutingStats = true; }
        else if (arg == "--subset-origin" && i + 1 < argc) {
            const auto v = parseInt3(argv[++i]);
            subset.enabled = true;
            subset.x0 = v[0];
            subset.y0 = v[1];
            subset.z0 = v[2];
        } else if (arg == "--subset-size" && i + 1 < argc) {
            const auto v = parseInt3(argv[++i]);
            subset.enabled = true;
            subset.sx = v[0];
            subset.sy = v[1];
            subset.sz = v[2];
        }
        else if (arg == "--cutoff" && i + 1 < argc) options.profile.den.renderCutoff = std::stof(argv[++i]);
        else if (arg == "--cutoff-band" && i + 1 < argc) options.profile.den.cutoffBand = std::stof(argv[++i]);
        else if (arg == "--eval-mask-cutoff" && i + 1 < argc) options.evalMaskCutoff = std::stof(argv[++i]);
        else if (arg == "--eval-mask-band" && i + 1 < argc) options.evalMaskBand = std::stof(argv[++i]);
        else if (arg == "--iso" && i + 1 < argc) options.profile.sdf.iso = std::stof(argv[++i]);
        else if (arg == "--compare-temporal-baseline") options.compareTemporalBaseline = true;
        else if (arg == "--report" && i + 1 < argc) reportPath = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage();
            return 1;
        }
    }

    if (inputRaw.empty()) {
        printUsage();
        return 1;
    }
    if (metadataPath.empty()) {
        metadataPath = guessMetadataPathForRaw(inputRaw);
        if (metadataPath.empty()) {
            std::cerr << "Metadata not found for raw: " << inputRaw << "\n";
            return 2;
        }
    }

    if (profileName == "generic") options.profile = FieldProfile::makeGeneric();
    else if (profileName == "density") options.profile = FieldProfile::makeDensity();
    else if (profileName == "sdf") options.profile = FieldProfile::makeSdf();
    else {
        std::cerr << "Unknown profile: " << profileName << "\n";
        return 3;
    }

    try {
#ifdef VBT_USE_OPENMP
        if (options.ompThreads > 0) {
            omp_set_num_threads(options.ompThreads);
        }
#endif
        if (warnResidualTopAlias) {
            std::cerr << "[warn] --residual-top is deprecated; use --event-top-k instead.\n";
        }
        if (warnResidualKfDeprecated) {
            std::cerr << "[warn] --residual-kf is deprecated and ignored for event-based Mode 2.\n";
        }
        if (warnResidualCellDeprecated) {
            std::cerr << "[warn] --residual-cell-size is deprecated and ignored for event-based Mode 2.\n";
        }
        auto volume = loadRawVolume(inputRaw, metadataPath);
        if (subset.enabled) {
            if (subset.sx <= 0 || subset.sy <= 0 || subset.sz <= 0) {
                throw std::runtime_error("Subset size must be set when subset probing is enabled");
            }
            volume = cropRawVolume(volume, subset.x0, subset.y0, subset.z0, subset.sx, subset.sy, subset.sz);
        }
        RenderTemporalFirstProbeStats temporalProbeStats{};
        if (options.profile.type == FieldType::DENSITY) {
            if (reportPath.empty()) {
                reportPath = std::filesystem::path("reports") / (inputRaw.stem().string() + "_render_temporal_mainline.md");
            }
            std::filesystem::create_directories(reportPath.parent_path());
            return runRenderTemporalMainline(inputRaw, metadataPath, reportPath, options);
        }
        if (options.renderTemporalFirstProbe) {
            volume = applyRenderTemporalFirstProbe(volume, options, &temporalProbeStats);
        }
        if (reportPath.empty()) {
            reportPath = std::filesystem::path("reports") / (inputRaw.stem().string() + "_spatialfirst_probe.md");
        }
        if (options.residualDiagnostics && options.residualDiagnosticDir.empty()) {
            options.residualDiagnosticDir =
                (reportPath.parent_path() / (inputRaw.stem().string() + "_residual_diagnostics")).string();
        }
        // Main execution is: load/crop -> encode whole volume -> evaluate ->
        // optionally save compact VBTPACK2 payload -> emit markdown/csv reports.
        SpatialFirstHybridEncoder encoder(options);
        const auto summary = encoder.run(volume);

        std::cout << "=== VBT Spatial-First Hybrid Temporal Probe ===\n";
        std::cout << "Input                 : " << inputRaw.string() << "\n";
        std::cout << "Dimensions            : " << volume.meta.width << "x" << volume.meta.height << "x" << volume.meta.depth << " x " << volume.meta.frames << "\n";
        if (options.renderTemporalFirstProbe) {
            const double avgKf =
                temporalProbeStats.voxelCount > 0
                    ? static_cast<double>(temporalProbeStats.totalKeyframes) / static_cast<double>(temporalProbeStats.voxelCount)
                    : 0.0;
            std::cout << "Temporal-first probe  : enabled\n";
            std::cout << "Temporal probe RMSE   : " << temporalProbeStats.rmse << "\n";
            std::cout << "Temporal probe PSNR   : " << temporalProbeStats.psnr << " dB\n";
            std::cout << "Temporal avg KFs      : " << avgKf << "\n";
            std::cout << "Temporal max KFs      : " << temporalProbeStats.maxKeyframes << "\n";
            std::cout << "Temporal bg-zero vox  : " << temporalProbeStats.backgroundZeroedVoxels
                      << " / " << temporalProbeStats.voxelCount << "\n";
            std::cout << "Temporal cutoff prot  : " << temporalProbeStats.temporalProtectedSeries << "\n";
        }
        std::cout << "Estimated payload     : " << summary.estimatedBytes << " bytes\n";
        if (!options.saveVbtPath.empty()) {
            std::cout << "Saved probe bytes     : " << summary.savedFileBytes << " bytes\n";
            std::cout << "Saved probe file      : " << options.saveVbtPath << "\n";
        }
        std::cout << "Offset table words    : " << summary.offsetTableWords << "\n";
        std::cout << "Payload words         : " << summary.payloadWords << "\n";
        std::cout << "Total words           : " << summary.totalWords << "\n";
        std::cout << "Evaluation            : " << (options.fullEvaluation ? "full-voxel" : "sampled") << "\n";
#ifdef VBT_USE_OPENMP
        std::cout << "OMP threads           : " << (options.ompThreads > 0 ? options.ompThreads : omp_get_max_threads()) << "\n";
#else
        std::cout << "OMP threads           : disabled\n";
#endif
        std::cout << "Evaluated samples     : " << summary.evaluatedSamples << "\n";
        std::cout << "Voxel RMSE            : " << summary.rmse << "\n";
        if (summary.maskedEvaluatedSamples > 0) {
            std::cout << "Cutoff-masked samples : " << summary.maskedEvaluatedSamples << "\n";
            std::cout << "Cutoff-masked thr     : " << summary.maskedThreshold << "\n";
            std::cout << "Cutoff-masked RMSE    : " << summary.maskedRmse << "\n";
            std::cout << "Cutoff-masked PSNR    : " << summary.maskedPsnr << " dB\n";
        }
        if (summary.p999Valid) {
            std::cout << "Voxel P99"
                      << (summary.tailMetricsApproximate ? " (hist)" : "")
                      << "            : " << summary.p99 << "\n";
            std::cout << "Voxel P99.9"
                      << (summary.tailMetricsApproximate ? " (hist)" : "")
                      << "          : " << summary.p999 << "\n";
            std::cout << "Voxel MaxAbs"
                      << (summary.tailMetricsApproximate ? " (exact)" : "")
                      << "        : " << summary.maxAbsError << "\n";
        }
        std::cout << "Voxel PSNR            : " << summary.psnr << " dB\n";
        std::cout << "Event top-K           : " << options.eventTopK << "\n";
        std::cout << "Event threshold       : " << options.eventThreshold << "\n";
        std::cout << "Dense temporal basis  : " << (options.genericDenseTemporalBasisCandidate ? "on" : "off")
                  << " force=" << (options.genericDenseTemporalBasisForce ? "on" : "off")
                  << " (keep=" << options.genericDenseTemporalKeep
                  << ", grid4=" << (options.genericDenseTemporalForGrid4 ? "on" : "off") << ")\n";
        std::cout << "Tile temporal basis   : " << (options.genericTileTemporalBasisCandidate ? "on" : "off")
                  << " force=" << (options.genericTileTemporalBasisForce ? "on" : "off")
                  << " (keep=" << options.genericTileTemporalKeep
                  << ", tile=" << options.genericTileSize << ")\n";
        std::cout << "Patch temporal basis  : " << (options.genericPatchTemporalBasisCandidate ? "on" : "off")
                  << " force=" << (options.genericPatchTemporalBasisForce ? "on" : "off")
                  << " (keep=" << options.genericPatchTemporalKeep << ", patch=4, local=2)\n";
        if (summary.pairedCoarsePsnr > 0.0) {
            std::cout << "Paired coarse RMSE    : " << summary.pairedCoarseRmse << "\n";
            std::cout << "Paired coarse PSNR    : " << summary.pairedCoarsePsnr << " dB\n";
        }
        if (options.compareTemporalBaseline) {
            std::cout << "Baseline sampled RMSE : " << summary.baselineRmse << "\n";
            std::cout << "Baseline sampled PSNR : " << summary.baselinePsnr << " dB\n";
            std::cout << "Baseline sampled KFs  : " << summary.baselineSampledKeyframes << "\n";
        }
        std::cout << "Coarse coeff count    : " << summary.coarseCoefficientCount << "\n";
        std::cout << "Fine coeff count      : " << summary.fineCoefficientCount << "\n";
        std::cout << "Event count           : " << summary.eventCount << "\n";
        std::cout << "Event payload bytes   : " << summary.eventPayloadBytes << "\n";
        std::cout << "Leaf count            : " << summary.leafCount << "\n";
        std::cout << "Mode counts           : M0=" << summary.mode0Count
                  << " M1=" << summary.mode1Count
                  << " M2=" << summary.mode2Count
                  << " M3=" << summary.mode3Count << "\n";
        std::cout << "Dense temporal wins   : " << summary.mode2DenseTemporalBasisCount << "\n";
        std::cout << "Tile temporal wins    : " << summary.mode2TileTemporalBasisCount << "\n";
        std::cout << "Patch temporal wins   : " << summary.mode2PatchTemporalBasisCount << "\n";
        if (summary.sparseDiagSamples > 0) {
            std::cout << "Sparse diag samples   : " << summary.sparseDiagSamples << "\n";
            std::cout << "Sparse coarse PSNR    : " << summary.sparseDiagCoarsePsnr << " dB\n";
            std::cout << "Sparse mode PSNR      : " << summary.sparseDiagModePsnr << " dB\n";
            std::cout << "Sparse improve/worse  : " << summary.sparseDiagImproved
                      << " / " << summary.sparseDiagWorsened << "\n";
        }

        std::filesystem::create_directories(reportPath.parent_path());
        writeReport(reportPath, inputRaw, volume, options, subset, summary);
        std::cout << "Report                : " << reportPath.string() << "\n";
        if (!routingStatsPath.empty()) {
            std::filesystem::create_directories(routingStatsPath.parent_path());
            writeRoutingStatsCsv(routingStatsPath, summary);
            std::cout << "Routing stats         : " << routingStatsPath.string() << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 4;
    }

    return 0;
}
