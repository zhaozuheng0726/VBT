#ifndef BLOCK_TREE_H
#define BLOCK_TREE_H

#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <array>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include "keyframe_detector.h"   // Point1D
#include "field_profile.h"       // FieldProfile

using RawVolume4D = std::vector<std::vector<std::vector<std::vector<float>>>>;

inline int bt_popcount32(uint32_t x) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt(x));
#else
    return __builtin_popcount(x);
#endif
}

inline int bt_popcount64(uint64_t x) {
#if defined(_MSC_VER) && defined(_M_X64)
    return static_cast<int>(__popcnt64(x));
#elif defined(_MSC_VER)
    return bt_popcount32(static_cast<uint32_t>(x)) + bt_popcount32(static_cast<uint32_t>(x >> 32));
#else
    return __builtin_popcountll(x);
#endif
}

// ============================================================
// IEEE 754 Float16 ↔ Float32 conversion (no third-party deps)
// ============================================================
inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign   = (x >> 31) & 0x1u;
    const uint32_t exp32  = (x >> 23) & 0xFFu;
    const uint32_t mant32 = x & 0x7FFFFFu;
    const uint16_t sign16 = static_cast<uint16_t>(sign << 15);

    if (exp32 == 0xFFu) return sign16 | 0x7C00u | (mant32 ? 0x200u : 0u);
    if (exp32 == 0u)    return sign16;

    const int exp16 = static_cast<int>(exp32) - 127 + 15;
    if (exp16 >= 31) return sign16 | 0x7C00u;
    if (exp16 <= 0) {
        if (exp16 < -10) return sign16;
        const uint32_t m = (mant32 | 0x800000u) >> (1 - exp16);
        return sign16 | static_cast<uint16_t>((m + 0x800u) >> 13);
    }
    uint32_t mant16 = (mant32 + 0x800u) >> 13;
    int exp16_out = exp16;
    if (mant16 >= 0x400u) {
        mant16 = 0u;
        exp16_out++;
        if (exp16_out >= 31) return sign16 | 0x7C00u;
    }
    return sign16 | static_cast<uint16_t>(exp16_out << 10)
                  | static_cast<uint16_t>(mant16 & 0x3FFu);
}

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign   = (h >> 15) & 0x1u;
    const uint32_t exp16  = (h >> 10) & 0x1Fu;
    const uint32_t mant16 = h & 0x3FFu;
    uint32_t x;
    if (exp16 == 0x1Fu) {
        x = (sign << 31) | 0x7F800000u | (mant16 << 13);
    } else if (exp16 == 0u) {
        if (mant16 == 0u) { x = sign << 31; }
        else {
            int e = -1; uint32_t m = mant16;
            while (!(m & 0x400u)) { m <<= 1; e--; }
            m &= 0x3FFu;
            x = (sign << 31) | (static_cast<uint32_t>(e + 127) << 23) | (m << 13);
        }
    } else {
        x = (sign << 31) | (static_cast<uint32_t>(exp16 - 15 + 127) << 23) | (mant16 << 13);
    }
    float fv; memcpy(&fv, &x, 4); return fv;
}

// ============================================================
// Fixed-depth brick tree (v5) — NanoVDB-inspired
//
// Bit decomposition per axis:
//   bits [2:0]  → leaf local index  (BT_LEAF_BITS=3  → 8 per axis)
//   bits [4:3]  → internal index    (BT_INTERNAL_BITS=2 → 4 per axis)
//   bits [31:5] → root index
//
// BLOCK_SPAN = 8 × 4 = 32  → each root cell covers 32 voxels per axis
// ============================================================

constexpr int BT_LEAF_BITS     = 3;
constexpr int BT_INTERNAL_BITS = 2;   // 4 per axis, BLOCK_SPAN = 32

constexpr int BT_LEAF_SIZE     = 1 << BT_LEAF_BITS;      // 8
constexpr int BT_INTERNAL_SIZE = 1 << BT_INTERNAL_BITS;  // 4
constexpr int BT_LEAF_VOXELS   = BT_LEAF_SIZE * BT_LEAF_SIZE * BT_LEAF_SIZE;   // 512
constexpr int BT_INT_CHILDREN  = BT_INTERNAL_SIZE * BT_INTERNAL_SIZE * BT_INTERNAL_SIZE; // 64

// Sentinel values in rootTable / childList
constexpr uint32_t BT_CHILD_AIR      = 0;
constexpr uint32_t BT_CHILD_INTERIOR = 1;
constexpr uint32_t BT_CHILD_ID_BASE  = 2;

// Max uint16 offset; if exceeded, use DENSE_32 escape hatch
constexpr uint32_t BT_OFFSET_THRESHOLD = 65535u;

// ---- Keyframe sequence ----
struct KFPoint {
    uint16_t t;  // frame index (0-65535)
    uint16_t v;  // float16 value bit pattern
};
static_assert(sizeof(KFPoint) == 4, "KFPoint must be 4 bytes");

using KFSeq = std::vector<KFPoint>;

// Piecewise-linear interpolation on a KFSeq
float kfInterp(const KFSeq& seq, float t);

// ---- Leaf block modes ----
enum class LeafMode : uint8_t {
    UNIFORM   = 0,  // 1 representative sequence
    CLUSTERED = 1,  // legacy shared-sequence codebook + assign[512]
    DENSE     = 2,  // 512 independent sequences
    DENSE_16  = 2,  // flat: 16-bit offset table (256 words)
    DENSE_32  = 3,  // flat: 32-bit offset table (512 words)
    GRID4     = 4,  // 4x4x4 control-grid sequences + trilinear spatial interpolation
    GRID4_RESIDUAL = 5, // GRID4 coarse base + selective sparse residual sequences
    POLY11    = 6,  // 11 polynomial coefficient sequences
    POLY11_RESIDUAL = 7, // POLY11 base + sparse additive residual sequences
    GRID4_MULTISCALE = 8, // GRID4 coarse base + regular fine residual control grid
};

// ---- GPU-friendly flat leaf layout (A表 + B表) ----
//
// SeqRef (uint32): bits[31:24]=N, bits[23:0]=absOffset into seqPool
//   N         — sequence length (max 255)
//   absOffset — word offset into BlockTree::seqPool (B表)
//
// LeafHeader points into topoPool (A表):
//   UNIFORM   → topoPool[topoOffset]          = 1 SeqRef
//   CLUSTERED → topoPool[topoOffset+0..127]   = assign (128 words)
//               topoPool[topoOffset+128..128+K-1] = K SeqRefs
//   DENSE     → topoPool[topoOffset+0..511]   = 512 SeqRefs
struct LeafHeader {
    uint8_t  mode;        // 0=UNIFORM, 1=CLUSTERED, 2=DENSE, 4=GRID4
    uint8_t  numCenters;  // K for CLUSTERED, 0 otherwise
    uint16_t reserved;    // padding to 4-byte boundary
    uint32_t topoOffset;  // word offset into BlockTree::topoPool (A表)
};

struct SeqMeta {
    uint32_t absOffset;
    uint16_t length;
    uint16_t flags;
};
static_assert(sizeof(SeqMeta) == 8, "SeqMeta must be 8 bytes");

// ---- Leaf block (build-time structure) ----
struct LeafBlock {
    LeafMode mode = LeafMode::DENSE;
    std::vector<KFSeq>    codebook;  // K or 512 sequences
    std::vector<uint8_t>  assign;    // 512 assignments (CLUSTERED only)
    std::array<uint64_t, 8> residualMask{};
    float residualScale = 0.0f;
    std::vector<KFSeq> residualCodebook;
    uint8_t fineGridDim = 0;
};

// ---- Sparse internal node ----
//
// childMask[64 bits]: bit i = 1 → non-default child at slot i
// defaultVal: implicit value for mask=0 slots
// Query: hasBit(i) → rankBefore(i) → childList[base+rank]
//
// On disk: 8 + 8 + 4 = 20 bytes
struct InternalNode {
    uint8_t  childMask[BT_INT_CHILDREN / 8];  // 8 bytes = 64 bits
    uint64_t childBase;    // offset into BlockTree::childList
    uint32_t defaultVal;   // BT_CHILD_AIR or BT_CHILD_INTERIOR

    InternalNode() {
        memset(childMask, 0, sizeof(childMask));
        childBase  = 0;
        defaultVal = BT_CHILD_AIR;
    }

    bool hasBit(int i) const { return (childMask[i >> 3] >> (i & 7)) & 1u; }

    int rankBefore(int i) const {
        int rank = 0;
        const int fullBytes = i >> 3;
        for (int b = 0; b < fullBytes; b++)
            rank += bt_popcount32(childMask[b]);
        const int rem = i & 7;
        if (rem) rank += bt_popcount32(
            childMask[fullBytes] & static_cast<uint8_t>((1u << rem) - 1u));
        return rank;
    }
};

// ---- Block tree ----
class BlockTree {
public:
    int dimX = 0, dimY = 0, dimZ = 0, numFrames = 0;
    int rootDimX = 0, rootDimY = 0, rootDimZ = 0;

    // For current uint8-biased SDF storage:
    // - semantic zero level set is encoded at 128
    // - the two SDF-side background extremes are typically 0 and 255
    float bgAir      = 0.0f;
    float bgInterior = 255.0f;

    std::vector<uint32_t>     rootTable;
    std::vector<InternalNode> internalNodes;
    std::vector<uint32_t>     childList;
    std::vector<LeafBlock>    leafBlocks;
    std::vector<std::array<int, 3>> leafOrigins;
    std::vector<std::array<int, 3>> leafGridCoords;
    bool   blockAwareCluster = false;
    bool   budgetAwareCluster = false;
    bool   guardedMedoidCluster = false;
    bool   grid4Spatial = false;
    double blockAwareClusterMin = 0.0;
    double blockAwareClusterMax = 0.0;
    double blockAwareClusterMean = 0.0;
    bool   validateFallback = false;
    bool   hotspotSecondPass = false;
    int    validateFallbackBlocks = 0;
    int    validateRetryBlocks = 0;
    int    validateDenseFallbackBlocks = 0;
    int    hotspotSecondPassBlocks = 0;
    int    hotspotSecondPassDenseBlocks = 0;
    int    hotspotRegionSeedBlocks = 0;
    int    hotspotRegionTouchedBlocks = 0;
    int    hotspotRegionDenseFallbackBlocks = 0;
    int    guardedGateRejects = 0;
    int    medoidCenterChanges = 0;
    int    qualityGatePromotions = 0;

    // Build from temporal-compressed volume (4D path)
    void build(
        const std::vector<std::vector<std::vector<std::vector<Point1D>>>>& compVol,
        int W, int H, int D, int T,
        double thresh_uniform = -1.0,
        double thresh_cluster = -1.0,
        const FieldProfile& profile = FieldProfile{},
        bool enableBackgroundElision = true,
        bool enableBlockAwareCluster = false,
        bool enableBudgetAwareCluster = false,
        bool enableGuardedMedoidCluster = false,
        bool enableGrid4Spatial = false,
        bool enableGrid4ControlOnlyTemporal = false,
        const RawVolume4D* rawVolume = nullptr,
        bool enableValidateFallback = false,
        bool enableHotspotSecondPass = false
    );

    // Sample value at (x,y,z) and temporal query t
    float sample(int x, int y, int z, float t) const;

    // Routing helper
    static void decompose(int x, int y, int z,
                          int& rx, int& ry, int& rz,
                          int& ix, int& iy, int& iz,
                          int& lx, int& ly, int& lz);

    // ---- GPU-friendly flat layout (A表 + B表) ----
    std::vector<LeafHeader> leafHeaders;
    std::vector<uint32_t>   topoPool;   // A表: topology (SeqRefs + assign tables)
    std::vector<uint32_t>   seqPool;    // B表: global deduplicated SoA seq data
    std::unordered_map<uint64_t, uint32_t> seqHashMap_;  // hash → absOffset in seqPool

    std::vector<SeqMeta>    seqMetaPool;

    void flattenLeaves();
    float sampleFlat(int x, int y, int z, float t) const;

    // B表 global sequence dedup helper (used by flattenLeaves)
    uint32_t findOrAddSeqToPool(const KFSeq& seq);

    // Outlier guard stats
    int outlierVoxels       = 0;
    int mode2FallbackBlocks = 0;

    struct Stats {
        int  totalLeafs;
        int  mode0, mode1, mode2, mode3, mode4, mode5, mode6, mode7;
        int  elidedAir, elidedInterior;
        long long totalKF;
        size_t memBytes;
        size_t fileSizeEstimate;
        float kMean, kP50, kP95, kMax;
        size_t childListSize;
        int outlierVoxels;
        int mode2FallbackBlocks;
        bool blockAwareCluster;
        bool budgetAwareCluster;
        bool guardedMedoidCluster;
        float localClusterThrMin;
        float localClusterThrMean;
        float localClusterThrMax;
        bool validateFallback;
        bool hotspotSecondPass;
        int validateFallbackBlocks;
        int validateRetryBlocks;
        int validateDenseFallbackBlocks;
        int hotspotSecondPassBlocks;
        int hotspotSecondPassDenseBlocks;
        int hotspotRegionSeedBlocks;
        int hotspotRegionTouchedBlocks;
        int hotspotRegionDenseFallbackBlocks;
        int guardedGateRejects;
        int medoidCenterChanges;
        int qualityGatePromotions;
    };
    Stats getStats() const;
};

// ---- Legacy ZSD5 serialization ----
bool writeBlockTree(const std::string& path, const BlockTree& tree);
bool readBlockTree (const std::string& path, BlockTree&       tree);

// ============================================================
// VBT3 file format — A表(topoPool) + B表(seqPool) binary format
//
// [128B Header]
// [Root Table]      rootTableCount × uint32
// [Internal Nodes]  internalCount  × 20 bytes each
// [Child List]      childListCount × uint32
// [Leaf Headers]    numLeafHeaders × 8 bytes
// [Topo Pool]       topoPoolWords  × uint32   (A表: SeqRefs + assign)
// [Seq Pool]        seqPoolWords   × uint32   (B表: SoA seq data, deduplicated)
// ============================================================

#pragma pack(push, 1)
struct VBTHeader {
    char     magic[4];         // "VBT2"
    uint32_t version;          // 3 or 4
    int32_t  dimX, dimY, dimZ;
    int32_t  numFrames;
    int32_t  rootDimX, rootDimY, rootDimZ;
    uint32_t rootTableCount;
    uint32_t internalCount;
    uint32_t childListCount;
    uint32_t numLeafHeaders;
    uint32_t topoPoolWords;    // A表 word count
    uint32_t seqPoolWords;     // B表 word count
    float    bgAir;
    float    bgInterior;
    float    voxelSize;
    uint32_t seqMetaCount;     // v4 only, 0 for v3
    uint64_t rootTableOff;
    uint64_t internalOff;
    uint64_t childListOff;
    uint64_t leafHeadersOff;
    uint64_t topoPoolOff;      // A表 offset
    uint64_t seqPoolOff;       // B表 offset
    uint8_t  _padding[4];
};
#pragma pack(pop)
static_assert(sizeof(VBTHeader) == 128, "VBTHeader must be 128 bytes");

bool saveToVBT(const std::string& filename, BlockTree& tree);
bool loadFromVBT(const std::string& filename, BlockTree& tree);
bool validateVBT(const BlockTree& tree, float tol = 1e-3f);

#endif // BLOCK_TREE_H
