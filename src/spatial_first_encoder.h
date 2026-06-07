#pragma once

#include "field_profile.h"
#include "raw_volume.h"

#include <cstdint>
#include <array>
#include <string>
#include <vector>

namespace vbt {

enum class BlockMode : uint8_t {
    CONSTANT = 0,
    COARSE_ONLY = 1,
    SPARSE_IMPULSE = 2, // scientific/generic sparse residual events
    DENSE_FINE = 3,     // fine grid or compact dense residual, depending on profile
};

struct SpatialFirstOptions {
    int leafSize = 8;
    int coarseResolution = 4;
    int dctKeep = 4;
    int eventTopK = 128;
    float eventThreshold = 0.02f;
    int eventMinCount = 3;
    bool genericStratifiedEvents = false;
    bool splitScientificRenderModes = true;
    bool genericAdaptiveCoarseKeep = true;
    bool genericDisableConstantMode = true;
    bool genericDenseCrossover = true;
    int genericDenseGridResolution = 3;
    bool genericDenseGrid4Candidate = true;
    bool genericDenseTemporalBasisCandidate = false;
    bool genericDenseTemporalBasisForce = false;
    int genericDenseTemporalKeep = 4;
    bool genericDenseTemporalForGrid4 = true;
    bool genericTileTemporalBasisCandidate = false;
    bool genericTileTemporalBasisForce = false;
    int genericTileTemporalKeep = 4;
    bool genericPatchTemporalBasisCandidate = false;
    bool genericPatchTemporalBasisForce = false;
    int genericPatchTemporalKeep = 4;
    int genericDenseResidualBits = 4;
    float genericDenseGrid4RateScale = 1.0f;
    float genericDenseGrid4DistThreshold = 0.3f;
    float genericRdoLambda = 1e-5f;
    int genericRdoSpatialStride = 2;
    int genericRdoTimeStride = 4;
    float genericRdoP99Weight = 2.0f;
    float genericRdoPeakWeight = 4.0f;
    bool genericRdoUseMaxEnvelope = true;
    bool genericChaosAdaptiveRdo = false;
    float genericChaosSpatialOccThreshold = 0.50f;
    float genericChaosTimeOccThreshold = 0.45f;
    bool genericSubgridCrossover = false;
    int genericSubgridBits = 4;
    bool genericMultipatchCrossover = false;
    int genericMultipatchBits = 4;
    bool genericTileCrossover = false;
    int genericTileBits = 4;
    int genericTileSize = 2;
    bool fineResidualGridForRenderProfiles = false;
    int fineGridResolution = 6;
    int fineGridDctKeep = 2;
    int fineQuantBits = 0;
    bool renderTemporalFirstProbe = false;
    bool renderUnifiedBackboneProbe = false;
    int renderCompactFineGridResolution = 6;
    int renderCompactFineDctKeep = 4;
    int renderCompactFineQuantBits = 8;
    int renderFullFineGridResolution = 8;
    int renderFullFineDctKeep = 8;
    int renderFullFineQuantBits = 8;
    float renderCompactFineNormErrThreshold = 0.20f;
    float renderCompactFinePeakErrThreshold = 0.60f;
    bool fullResidualForRenderProfiles = false;
    int fullResidualBits = 2;
    bool renderDisableConstantMode = false;
    bool routeVbrModes = true;
    float routeConstRangeRatio = 1e-4f;
    float routeCoarseOnlyNormErrGeneric = 0.035f;
    float routeCoarseOnlyPeakErrGeneric = 0.20f;
    float routeSparseEnergyFracGeneric = 0.60f;
    float routeSparseNormErrGeneric = 0.010f;
    float routeCoarseOnlyNormErrRender = 0.080f;
    float genericEnergyAmnestyNorm = 0.0f;
    float genericEnergyAmnestyHotFrac = 0.0f;
    float genericEnergyAmnestyPeakScale = 1.5f;
    int genericEnergyAmnestyTopN = 16;
    int sampleStepX = 8;
    int sampleStepY = 8;
    int sampleStepZ = 8;
    int sampleStepT = 8;
    bool fullEvaluation = false;
    float evalMaskCutoff = 0.0f;
    float evalMaskBand = 0.0f;
    int ompThreads = 0;
    std::string saveVbtPath;
    bool residualDiagnostics = false;
    int residualDiagnosticBlocks = 4;
    std::string residualDiagnosticDir;
    bool compareTemporalBaseline = false;
    bool dumpRoutingStats = false;
    FieldProfile profile = FieldProfile::makeGeneric();
    float renderTemporalProbeEpsAbs = 1e-5f;
    float renderTemporalProbeEpsRel = 0.02f;
    float renderTemporalProbeGammaDelta = 0.2f;
    float renderTemporalProbeBgZeroRatio = 0.30f;
    bool renderTemporalProbeCutoffProtect = true;
};

struct LeafRouteStat {
    int bx = 0;
    int by = 0;
    int bz = 0;
    double leafRange = 0.0;
    double coarseRmse = 0.0;
    double normErr = 0.0;
    double peakErrNorm = 0.0;
    double topEnergyFrac = 0.0;
    int activeFrames = 0;
    BlockMode mode = BlockMode::COARSE_ONLY;
};

struct SpatiotemporalEvent {
    uint16_t spatialIndex = 0;
    uint8_t timeTag = 0;
    int16_t quantizedValue = 0;
};

struct DenseResidualBfpEncoding {
    bool sampleAsGrid = false;
    bool usePatchMask = false;
    int resolution = 8;
    int valuesPerFrame = 512;
    int originX = 0;
    int originY = 0;
    int originZ = 0;
    int extentX = 8;
    int extentY = 8;
    int extentZ = 8;
    int timeStartLocal = 0;
    int timeCount = 0;
    uint64_t patchMask = 0;
    int patchSize = 4;
    int activePatchCount = 0;
    int bitsPerValue = 2;
    int frameBytes = 0;
    std::vector<float> frameScales;
    std::vector<uint8_t> packedValues;
};

struct DenseTemporalBasisEncoding {
    int resolution = 3;
    int temporalKeep = 0;
    int bitsPerValue = 0;
    int valuesPerBasis = 0;
    int basisBytes = 0;
    std::vector<float> basisScales;
    std::vector<uint8_t> packedBasisValues;
};

struct DenseTileTemporalBasisEncoding {
    int tileSize = 2;
    uint64_t activeTileMask = 0ull;
    int activeTileCount = 0;
    int temporalKeep = 0;
    int bitsPerValue = 0;
    int valuesPerBasis = 0;
    int basisBytes = 0;
    std::vector<float> basisScales;
    std::vector<uint8_t> packedBasisValues;
};

struct DensePatchTemporalBasisEncoding {
    uint64_t activePatchMask = 0ull;
    int activePatchCount = 0;
    int patchSize = 4;
    int localResolution = 2;
    int temporalKeep = 0;
    int bitsPerValue = 0;
    int valuesPerBasis = 0;
    int basisBytes = 0;
    std::vector<float> basisScales;
    std::vector<uint8_t> packedBasisValues;
};

struct FineResidualGridEncoding {
    int resolution = 6;
    int dctKeep = 2;
    int quantBits = 0;
    float blockScale = 0.0f;
    std::vector<float> coeffs;
    std::vector<int16_t> quantizedCoeffs;
};

struct LeafEncoding {
    BlockMode mode = BlockMode::COARSE_ONLY;
    uint8_t startFrame = 0;
    uint8_t endFrame = 0;
    uint8_t coarseKeep = 0;
    uint16_t config = 0;
    uint32_t packedHeader = 0;
    int encodedFrameCount = 0;
    float constantValue = 0.0f;
    std::vector<float> coarseCoeffs;
    // Mode 2 payload: threshold-first residual events packed for fast lookup.
    std::vector<SpatiotemporalEvent> events;
    float eventScale = 0.0f;
    uint16_t packedEventCount = 0;
    uint16_t packedTierCapacity = 0;
    std::array<uint32_t, 16> spatialBitmask{};
    std::array<uint16_t, 16> timeBins{};
    std::vector<uint16_t> packedCoords;
    std::vector<uint8_t> packedResiduals;
    // Mode 3 payload: either render-oriented fine residual grid or a compact
    // dense residual used by scientific crossover.
    bool useFineResidualGrid = false;
    FineResidualGridEncoding fineResidualGrid;
    bool useDenseResidualBfp = false;
    DenseResidualBfpEncoding denseResidualBfp;
    bool useDenseTemporalBasis = false;
    DenseTemporalBasisEncoding denseTemporalBasis;
    bool useDenseTileTemporalBasis = false;
    DenseTileTemporalBasisEncoding denseTileTemporalBasis;
    bool useDensePatchTemporalBasis = false;
    DensePatchTemporalBasisEncoding densePatchTemporalBasis;
};

struct ProbeSummary {
    double rmse = 0.0;
    double psnr = 0.0;
    double p99 = 0.0;
    double p999 = 0.0;
    double maxAbsError = 0.0;
    bool p999Valid = true;
    bool tailMetricsApproximate = false;
    uint64_t evaluatedSamples = 0;
    uint64_t maskedEvaluatedSamples = 0;
    double maskedThreshold = 0.0;
    double maskedRmse = 0.0;
    double maskedPsnr = 0.0;
    double pairedCoarseRmse = 0.0;
    double pairedCoarsePsnr = 0.0;
    double sparseDiagCoarseRmse = 0.0;
    double sparseDiagCoarsePsnr = 0.0;
    double sparseDiagModeRmse = 0.0;
    double sparseDiagModePsnr = 0.0;
    uint64_t estimatedBytes = 0;
    uint64_t coarseCoefficientCount = 0;
    uint64_t fineCoefficientCount = 0;
    uint64_t eventCount = 0;
    uint64_t eventPayloadBytes = 0;
    uint64_t leafCount = 0;
    uint64_t mode0Count = 0;
    uint64_t mode1Count = 0;
    uint64_t mode2Count = 0;
    uint64_t mode3Count = 0;
    uint64_t renderCompactFineCount = 0;
    uint64_t renderFullFineCount = 0;
    uint64_t mode1EnergyAmnestyCount = 0;
    uint64_t mode1EventFallbackCount = 0;
    uint64_t mode2DenseCrossoverCount = 0;
    uint64_t mode2DenseGrid3Count = 0;
    uint64_t mode2DenseGrid4Count = 0;
    uint64_t mode2DenseTemporalBasisCount = 0;
    uint64_t mode2TileTemporalBasisCount = 0;
    uint64_t mode2PatchTemporalBasisCount = 0;
    uint64_t genericChaosRdoCount = 0;
    uint64_t genericChaosDenseWins = 0;
    uint64_t mode2MultipatchCrossoverCount = 0;
    uint64_t mode2TileCrossoverCount = 0;
    uint64_t genericSubgridCandidateCount = 0;
    uint64_t genericSubgridCheaperCount = 0;
    std::array<uint64_t, 8> scientificCoarseKeepCounts{0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t scientificCoarseKeepOtherCount = 0;
    uint64_t offsetTableWords = 0;
    uint64_t payloadWords = 0;
    uint64_t totalWords = 0;
    uint64_t savedFileBytes = 0;
    double baselineRmse = 0.0;
    double baselinePsnr = 0.0;
    uint64_t baselineSampledKeyframes = 0;
    uint64_t sparseDiagSamples = 0;
    uint64_t sparseDiagImproved = 0;
    uint64_t sparseDiagWorsened = 0;
    std::vector<LeafRouteStat> routeStats;
    std::vector<uint32_t> mode2EventCounts;
    std::vector<uint16_t> mode2UniqueSpatialCounts;
    std::vector<uint16_t> mode2UniqueTimeCounts;
    std::vector<uint16_t> mode2EncodedFrameCounts;
    std::vector<uint32_t> genericSubgridVoxelCounts;
    std::vector<uint16_t> genericSubgridFrameCounts;
    std::vector<uint32_t> genericSubgridBytes;
    std::vector<uint32_t> genericSparseBytes;
};

class SpatialFirstHybridEncoder {
public:
    explicit SpatialFirstHybridEncoder(SpatialFirstOptions options);
    ProbeSummary run(const RawVolume4D& volume);

private:
    SpatialFirstOptions m_options;
};

} // namespace vbt
