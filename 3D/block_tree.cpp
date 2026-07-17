#include "block_tree.h"
#include "keyframe_detector.h"
#include "field_profile.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#ifdef _OPENMP
#include <omp.h>
#endif

static float sampleLeafGrid4(const LeafBlock& leaf, int lx, int ly, int lz, float t);
static float sampleLeafRegularControlGrid(const std::vector<KFSeq>& codebook, int gridDim, int lx, int ly, int lz, float t);
static float sampleLeafRegularControlGridMasked(const std::vector<KFSeq>& codebook,
                                                const std::array<uint64_t, 8>& mask,
                                                int gridDim, int lx, int ly, int lz, float t);

// ============================================================
// kfInterp — piecewise-linear interpolation on a KFSeq
// ============================================================
float kfInterp(const KFSeq& seq, float t) {
    if (seq.empty())           return 128.0f;
    if (seq.size() == 1)       return f16_to_f32(seq[0].v);
    if (t <= seq.front().t)    return f16_to_f32(seq.front().v);
    if (t >= seq.back().t)     return f16_to_f32(seq.back().v);

    int lo = 0, hi = static_cast<int>(seq.size()) - 1;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (seq[mid].t <= t) lo = mid; else hi = mid;
    }
    float t0 = seq[lo].t, t1 = seq[hi].t;
    float v0 = f16_to_f32(seq[lo].v), v1 = f16_to_f32(seq[hi].v);
    if (t1 == t0) return v0;
    return v0 + (t - t0) / (t1 - t0) * (v1 - v0);
}

// ============================================================
// kfInterpPtr — interpolation directly on a KFPoint array slice
// ============================================================
static float kfInterpPtr(const KFPoint* pts, int N, float t) {
    if (N == 0) return 128.0f;
    if (N == 1) return f16_to_f32(pts[0].v);
    if (t <= pts[0].t)     return f16_to_f32(pts[0].v);
    if (t >= pts[N - 1].t) return f16_to_f32(pts[N - 1].v);
    int lo = 0, hi = N - 1;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (pts[mid].t <= t) lo = mid; else hi = mid;
    }
    float t0 = pts[lo].t, t1 = pts[hi].t;
    float v0 = f16_to_f32(pts[lo].v), v1 = f16_to_f32(pts[hi].v);
    if (t1 == t0) return v0;
    return v0 + (t - t0) / (t1 - t0) * (v1 - v0);
}

// ============================================================
// BlockTree::decompose — per-axis bit decomposition
//
// BT_INTERNAL_BITS=2, BT_LEAF_BITS=3  → BLOCK_SPAN=32
//   bits [2:0]  → leaf local index   (3 bits, 8 per axis)
//   bits [4:3]  → internal index     (2 bits, 4 per axis)
//   bits [31:5] → root index
// ============================================================
void BlockTree::decompose(int x, int y, int z,
                          int& rx, int& ry, int& rz,
                          int& ix, int& iy, int& iz,
                          int& lx, int& ly, int& lz)
{
    lx = x & 0x7;           ix = (x >> 3) & 0x3;     rx = x >> 5;
    ly = y & 0x7;           iy = (y >> 3) & 0x3;     ry = y >> 5;
    lz = z & 0x7;           iz = (z >> 3) & 0x3;     rz = z >> 5;
}

// ============================================================
// Pending child entry — filled during Phase 1, sorted/compacted in Phase 2
// ============================================================
struct ChildEntry {
    uint32_t internalId;
    uint16_t childIdx;   // 0 .. BT_INT_CHILDREN-1
    uint32_t value;      // BT_CHILD_AIR / BT_CHILD_INTERIOR / leafId+BASE
};

// ============================================================
// Phase 2 helper — sort pendingEntries and build sparse InternalNodes
// ============================================================
static void phase2_compact(std::vector<ChildEntry>& pendingEntries,
                           std::vector<InternalNode>& internalNodes,
                           std::vector<uint32_t>& childList,
                           int firstNewNode, int lastNewNode)
{
    std::sort(pendingEntries.begin(), pendingEntries.end(),
        [](const ChildEntry& a, const ChildEntry& b) {
            if (a.internalId != b.internalId) return a.internalId < b.internalId;
            return a.childIdx < b.childIdx;
        });

    size_t pi = 0;
    const size_t nP = pendingEntries.size();

    for (int ni = firstNewNode; ni <= lastNewNode; ni++) {
        size_t start = pi;
        while (pi < nP && pendingEntries[pi].internalId == static_cast<uint32_t>(ni)) ++pi;
        if (start == pi) continue;

        InternalNode& node = internalNodes[ni];

        // M5.2: defaultVal = majority of AIR vs INTERIOR among non-leaf entries
        int airCnt = 0, intCnt = 0;
        for (size_t j = start; j < pi; j++) {
            const uint32_t v = pendingEntries[j].value;
            if (v == BT_CHILD_AIR)           ++airCnt;
            else if (v == BT_CHILD_INTERIOR) ++intCnt;
        }
        node.defaultVal = (intCnt > airCnt) ? BT_CHILD_INTERIOR : BT_CHILD_AIR;

        node.childBase = static_cast<uint64_t>(childList.size());
        memset(node.childMask, 0, sizeof(node.childMask));

        for (size_t j = start; j < pi; j++) {
            const uint32_t v = pendingEntries[j].value;
            if (v != node.defaultVal) {
                const int ci = pendingEntries[j].childIdx;
                node.childMask[ci >> 3] |= static_cast<uint8_t>(1u << (ci & 7));
                childList.push_back(v);
            }
        }
    }
}

// ============================================================
// clusterLeaf — greedy L∞ clustering with Outlier Guard
//
// For each voxel i in leaf.codebook (size=512), find existing
// cluster centre with min profile.metric_error ≤ cThr.
// If found: assign to it. Else: new cluster (i is centre).
// K=1 → UNIFORM, 2≤K≤255 → CLUSTERED, K≥256 → DENSE.
//
// Outlier Guard (Method B): after first pass, re-verify all
// voxels against their cluster centre. Any voxel exceeding
// cThr at any frame becomes its own solo cluster (unless K≥256
// → fall back to DENSE, Method A).
// ============================================================
static int densityCutoffState(float v, const DensityProfileParams& den)
{
    if (den.render_cutoff <= 0.0f) return 1;
    if (v < den.render_cutoff - den.cutoff_band) return 0;
    if (v > den.render_cutoff + den.cutoff_band) return 2;
    return 1;
}

static bool densityVisibilityCompatible(const KFSeq& a, const KFSeq& b, int T, const DensityProfileParams& den)
{
    if (!den.cutoff_cluster_protect || den.render_cutoff <= 0.0f) return true;

    std::vector<int> times;
    times.reserve(a.size() + b.size() + 2);
    for (const auto& kf : a) times.push_back(kf.t);
    for (const auto& kf : b) times.push_back(kf.t);
    times.push_back(0);
    times.push_back(T - 1);
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end()), times.end());

    for (int t : times) {
        const int sa = densityCutoffState(kfInterp(a, static_cast<float>(t)), den);
        const int sb = densityCutoffState(kfInterp(b, static_cast<float>(t)), den);
        if ((sa == 0 && sb == 2) || (sa == 2 && sb == 0)) return false;
    }
    return true;
}

static void clusterLeaf(LeafBlock& leaf, double cThr, int T,
                        const FieldProfile& profile,
                        int& outlierVoxels, int& mode2FallbackBlocks,
                        const std::array<uint8_t, BT_LEAF_VOXELS>* surfaceCriticalMask = nullptr)
{
    if (cThr <= 0.0) {
        // No clustering — keep DENSE mode
        leaf.mode = LeafMode::DENSE;
        leaf.assign.clear();
        return;
    }

    const int N = BT_LEAF_VOXELS;  // 512
    struct GenericSeqStats {
        float mean = 0.0f;
        float stddev = 0.0f;
    };
    std::vector<GenericSeqStats> genericStats;
    if (profile.type == FieldType::GENERIC) {
        genericStats.resize(N);
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            double sum2 = 0.0;
            for (int t = 0; t < T; ++t) {
                const float v = kfInterp(leaf.codebook[i], static_cast<float>(t));
                sum += v;
                sum2 += static_cast<double>(v) * v;
            }
            const double mean = sum / T;
            const double var = std::max(0.0, sum2 / T - mean * mean);
            genericStats[i].mean = static_cast<float>(mean);
            genericStats[i].stddev = static_cast<float>(std::sqrt(var));
        }
    }
    auto metricErr = [&](int seqIdx, float truth, float pred) {
        if (profile.type != FieldType::GENERIC) {
            return static_cast<double>(profile.metric_error(truth, pred));
        }
        const auto& st = genericStats[seqIdx];
        const float centeredMag = std::max(std::abs(truth - st.mean), st.stddev);
        const float denom = profile.den.eps_abs + profile.den.eps_rel * centeredMag;
        return static_cast<double>(std::abs(truth - pred) / std::max(denom, 1e-6f));
    };
    // Cluster centres (by voxel index into codebook)
    std::vector<int> centres;       // index into codebook[] of each centre
    std::vector<int> assign(N, -1);

    for (int i = 0; i < N; i++) {
        const KFSeq& si = leaf.codebook[i];

        if (surfaceCriticalMask && (*surfaceCriticalMask)[i]) {
            assign[i] = static_cast<int>(centres.size());
            centres.push_back(i);
            continue;
        }

        // Find best cluster
        int bestCluster = -1;
        for (int c = 0; c < static_cast<int>(centres.size()); c++) {
            if (surfaceCriticalMask && (*surfaceCriticalMask)[centres[c]]) continue;
            const KFSeq& sc = leaf.codebook[centres[c]];
            if (profile.type == FieldType::DENSITY &&
                !densityVisibilityCompatible(si, sc, T, profile.den)) {
                continue;
            }

            // Check L∞ metric_error over all frames (union of kf times)
            // Build union of t values
            std::vector<int> times;
            times.reserve(si.size() + sc.size());
            for (const auto& kf : si) times.push_back(kf.t);
            for (const auto& kf : sc) times.push_back(kf.t);
            times.push_back(0); times.push_back(T - 1);
            std::sort(times.begin(), times.end());
            times.erase(std::unique(times.begin(), times.end()), times.end());

            double maxErr = 0.0;
            for (int t : times) {
                float vi = kfInterp(si, static_cast<float>(t));
                float vc = kfInterp(sc, static_cast<float>(t));
                double err = metricErr(i, vi, vc);
                if (err > maxErr) maxErr = err;
                if (maxErr > cThr) break;  // early exit
            }

            if (maxErr <= cThr) {
                bestCluster = c;
                break;  // greedy: first fit
            }
        }

        if (bestCluster >= 0) {
            assign[i] = bestCluster;
        } else {
            assign[i] = static_cast<int>(centres.size());
            centres.push_back(i);
        }
    }

    const int K = static_cast<int>(centres.size());

    // --- Outlier Guard (Method B) ---
    // Re-verify all voxels against their assigned centre.
    // If any frame exceeds cThr, make it a solo cluster.
    int newOutliers = 0;
    for (int i = 0; i < N; i++) {
        int c = assign[i];
        if (c < 0 || centres[c] == i) continue;  // already a centre

        const KFSeq& si = leaf.codebook[i];
        const KFSeq& sc = leaf.codebook[centres[c]];

        std::vector<int> times;
        times.reserve(si.size() + sc.size());
        for (const auto& kf : si) times.push_back(kf.t);
        for (const auto& kf : sc) times.push_back(kf.t);
        times.push_back(0); times.push_back(T - 1);
        std::sort(times.begin(), times.end());
        times.erase(std::unique(times.begin(), times.end()), times.end());

        bool ok = true;
        for (int t : times) {
            float vi = kfInterp(si, static_cast<float>(t));
            float vc = kfInterp(sc, static_cast<float>(t));
            if (metricErr(i, vi, vc) > cThr) { ok = false; break; }
        }
        if (!ok) {
            // Method B: promote this voxel to a new solo cluster
            assign[i] = static_cast<int>(centres.size());
            centres.push_back(i);
            ++newOutliers;
        }
    }
    if (newOutliers > 0) outlierVoxels += newOutliers;

    const int Kfinal = static_cast<int>(centres.size());

    // Method A: if K≥256, fall back to DENSE
    if (Kfinal >= 256) {
        ++mode2FallbackBlocks;
        leaf.mode = LeafMode::DENSE;
        leaf.assign.clear();
        return;
    }

    if (Kfinal == 1) {
        // UNIFORM
        leaf.mode = LeafMode::UNIFORM;
        KFSeq centreSeq = leaf.codebook[centres[0]];  // save BEFORE resize
        leaf.codebook.resize(1);
        leaf.codebook[0] = std::move(centreSeq);
        leaf.assign.clear();
    } else {
        // CLUSTERED
        leaf.mode = LeafMode::CLUSTERED;
        // Build compact codebook from centres
        std::vector<KFSeq> compactCB(Kfinal);
        for (int c = 0; c < Kfinal; c++) compactCB[c] = leaf.codebook[centres[c]];
        leaf.codebook = std::move(compactCB);
        leaf.assign.resize(N);
        for (int i = 0; i < N; i++) leaf.assign[i] = static_cast<uint8_t>(assign[i]);
    }
}

static void clusterLeafGuarded(LeafBlock& leaf, double cThr, int T,
                               const FieldProfile& profile,
                               int& outlierVoxels, int& mode2FallbackBlocks,
                               int& guardedGateRejects,
                               int& medoidCenterChanges,
                               int& qualityGatePromotions,
                               const std::array<uint8_t, BT_LEAF_VOXELS>* surfaceCriticalMask = nullptr)
{
    if (profile.type == FieldType::SDF) {
        clusterLeaf(leaf, cThr, T, profile, outlierVoxels, mode2FallbackBlocks, surfaceCriticalMask);
        return;
    }
    if (cThr <= 0.0) {
        leaf.mode = LeafMode::DENSE;
        leaf.assign.clear();
        return;
    }

    const int N = BT_LEAF_VOXELS;
    struct GenericSeqStats {
        float mean = 0.0f;
        float stddev = 0.0f;
    };
    struct SeqSketch {
        std::array<float, 9> samples{};
        float minValue = 0.0f;
        float maxValue = 0.0f;
        int minIdx = 0;
        int maxIdx = 0;
        float stddev = 0.0f;
    };

    std::vector<GenericSeqStats> genericStats(N);
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        double sum2 = 0.0;
        for (int t = 0; t < T; ++t) {
            const float v = kfInterp(leaf.codebook[i], static_cast<float>(t));
            sum += v;
            sum2 += static_cast<double>(v) * v;
        }
        const double mean = sum / T;
        const double var = std::max(0.0, sum2 / T - mean * mean);
        genericStats[i].mean = static_cast<float>(mean);
        genericStats[i].stddev = static_cast<float>(std::sqrt(var));
    }

    std::vector<SeqSketch> seqSketches(N);
    std::array<int, 9> sampleTimes{};
    for (int s = 0; s < 9; ++s) sampleTimes[s] = (T <= 1) ? 0 : ((T - 1) * s) / 8;
    for (int i = 0; i < N; ++i) {
        SeqSketch sketch;
        double sum = 0.0;
        double sum2 = 0.0;
        sketch.minValue = std::numeric_limits<float>::infinity();
        sketch.maxValue = -std::numeric_limits<float>::infinity();
        for (int s = 0; s < 9; ++s) {
            const float v = kfInterp(leaf.codebook[i], static_cast<float>(sampleTimes[s]));
            sketch.samples[s] = v;
            sum += v;
            sum2 += static_cast<double>(v) * v;
            if (v < sketch.minValue) {
                sketch.minValue = v;
                sketch.minIdx = s;
            }
            if (v > sketch.maxValue) {
                sketch.maxValue = v;
                sketch.maxIdx = s;
            }
        }
        const double mean = sum / 9.0;
        const double var = std::max(0.0, sum2 / 9.0 - mean * mean);
        (void)mean;
        sketch.stddev = static_cast<float>(std::sqrt(var));
        seqSketches[i] = sketch;
    }

    auto metricErr = [&](int seqIdx, float truth, float pred) {
        const auto& st = genericStats[seqIdx];
        const float centeredMag = std::max(std::abs(truth - st.mean), st.stddev);
        const float denom = profile.den.eps_abs + profile.den.eps_rel * centeredMag;
        return static_cast<double>(std::abs(truth - pred) / std::max(denom, 1e-6f));
    };

    auto genericThreeGatePass = [&](int seqIdx, int centerIdx) {
        const auto& a = seqSketches[seqIdx];
        const auto& b = seqSketches[centerIdx];
        const float scaleA = std::max(1e-5f, std::max(a.stddev, a.maxValue - a.minValue));
        const float envelopeGap =
            std::max(std::abs(a.minValue - b.minValue), std::abs(a.maxValue - b.maxValue)) / scaleA;
        if (envelopeGap > 4.5f) return false;

        const int extremaSlack = 2;
        if (std::abs(a.minIdx - b.minIdx) > extremaSlack &&
            std::abs(a.maxIdx - b.maxIdx) > extremaSlack) {
            return false;
        }

        double meanA = 0.0, meanB = 0.0;
        for (int s = 0; s < 9; ++s) {
            meanA += a.samples[s];
            meanB += b.samples[s];
        }
        meanA /= 9.0;
        meanB /= 9.0;
        double varA = 0.0, varB = 0.0, cov = 0.0;
        int slopeCount = 0;
        int slopeMismatch = 0;
        for (int s = 0; s < 9; ++s) {
            const double da = a.samples[s] - meanA;
            const double db = b.samples[s] - meanB;
            varA += da * da;
            varB += db * db;
            cov += da * db;
            if (s > 0) {
                const float dsa = a.samples[s] - a.samples[s - 1];
                const float dsb = b.samples[s] - b.samples[s - 1];
                const int sa = (dsa > 1e-6f) ? 1 : ((dsa < -1e-6f) ? -1 : 0);
                const int sb = (dsb > 1e-6f) ? 1 : ((dsb < -1e-6f) ? -1 : 0);
                if (sa != 0 && sb != 0) {
                    ++slopeCount;
                    if (sa != sb) ++slopeMismatch;
                }
            }
        }
        const double denom = std::sqrt(std::max(1e-12, varA * varB));
        const double corr = (denom > 0.0) ? cov / denom : 1.0;
        if (corr < 0.60) return false;
        if (slopeCount > 0 && (static_cast<double>(slopeMismatch) / slopeCount) > 0.45) return false;
        return true;
    };

    auto sampledQualityStats = [&](int seqIdx, int centerIdx, double& p95Norm, double& maxNorm) {
        std::array<double, 9> errs{};
        for (int s = 0; s < 9; ++s) {
            const float truth = seqSketches[seqIdx].samples[s];
            const float pred = seqSketches[centerIdx].samples[s];
            errs[s] = metricErr(seqIdx, truth, pred);
        }
        std::sort(errs.begin(), errs.end());
        p95Norm = errs[8];
        maxNorm = errs[8];
    };

    auto sampleL1 = [&](int aIdx, int bIdx) {
        double acc = 0.0;
        for (int s = 0; s < 9; ++s) {
            acc += std::abs(static_cast<double>(seqSketches[aIdx].samples[s]) -
                            static_cast<double>(seqSketches[bIdx].samples[s]));
        }
        return acc;
    };

    auto passesFullPair = [&](int seqIdx, int centerIdx) {
        if (!genericThreeGatePass(seqIdx, centerIdx)) {
            ++guardedGateRejects;
            return false;
        }

        const KFSeq& si = leaf.codebook[seqIdx];
        const KFSeq& sc = leaf.codebook[centerIdx];
        if (!densityVisibilityCompatible(si, sc, T, profile.den)) {
            ++guardedGateRejects;
            return false;
        }
        std::vector<int> times;
        times.reserve(si.size() + sc.size());
        for (const auto& kf : si) times.push_back(kf.t);
        for (const auto& kf : sc) times.push_back(kf.t);
        times.push_back(0);
        times.push_back(T - 1);
        std::sort(times.begin(), times.end());
        times.erase(std::unique(times.begin(), times.end()), times.end());

        double maxErr = 0.0;
        for (int t : times) {
            const float vi = kfInterp(si, static_cast<float>(t));
            const float vc = kfInterp(sc, static_cast<float>(t));
            const double err = metricErr(seqIdx, vi, vc);
            if (err > maxErr) maxErr = err;
            if (maxErr > cThr) return false;
        }

        double p95Norm = 0.0, maxNorm = 0.0;
        sampledQualityStats(seqIdx, centerIdx, p95Norm, maxNorm);
        const double p95Gate = std::max(0.65, 0.80 * cThr);
        const double maxGate = std::max(0.85, 1.00 * cThr);
        if (p95Norm > p95Gate || maxNorm > maxGate) {
            ++guardedGateRejects;
            return false;
        }
        return true;
    };

    std::vector<int> centres;
    std::vector<int> assign(N, -1);
    for (int i = 0; i < N; ++i) {
        int bestCluster = -1;
        for (int c = 0; c < static_cast<int>(centres.size()); ++c) {
            if (passesFullPair(i, centres[c])) {
                bestCluster = c;
                break;
            }
        }
        if (bestCluster >= 0) assign[i] = bestCluster;
        else {
            assign[i] = static_cast<int>(centres.size());
            centres.push_back(i);
        }
    }

    int newOutliers = 0;
    for (int i = 0; i < N; ++i) {
        const int c = assign[i];
        if (c < 0 || centres[c] == i) continue;
        if (!passesFullPair(i, centres[c])) {
            assign[i] = static_cast<int>(centres.size());
            centres.push_back(i);
            ++newOutliers;
            ++qualityGatePromotions;
        }
    }
    if (newOutliers > 0) outlierVoxels += newOutliers;

    if (!centres.empty()) {
        std::vector<std::vector<int>> members(centres.size());
        for (int i = 0; i < N; ++i) {
            if (assign[i] >= 0) members[assign[i]].push_back(i);
        }
        for (int c = 0; c < static_cast<int>(centres.size()); ++c) {
            const auto& cmembers = members[c];
            if (cmembers.size() <= 1) continue;
            int bestMedoid = centres[c];
            double bestCost = std::numeric_limits<double>::infinity();
            for (int cand : cmembers) {
                double cost = 0.0;
                for (int m : cmembers) cost += sampleL1(cand, m);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestMedoid = cand;
                }
            }
            if (bestMedoid != centres[c]) {
                centres[c] = bestMedoid;
                ++medoidCenterChanges;
            }
        }

        std::vector<int> newAssign(N, -1);
        for (int i = 0; i < N; ++i) {
            int bestCluster = -1;
            double bestCost = std::numeric_limits<double>::infinity();
            for (int c = 0; c < static_cast<int>(centres.size()); ++c) {
                if (!passesFullPair(i, centres[c])) continue;
                const double cost = sampleL1(i, centres[c]);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestCluster = c;
                }
            }
            if (bestCluster >= 0) {
                newAssign[i] = bestCluster;
            } else {
                newAssign[i] = static_cast<int>(centres.size());
                centres.push_back(i);
                ++qualityGatePromotions;
            }
        }
        assign = std::move(newAssign);
    }

    const int Kfinal = static_cast<int>(centres.size());
    if (Kfinal >= 256) {
        ++mode2FallbackBlocks;
        leaf.mode = LeafMode::DENSE;
        leaf.assign.clear();
        return;
    }

    if (Kfinal == 1) {
        leaf.mode = LeafMode::UNIFORM;
        KFSeq centreSeq = leaf.codebook[centres[0]];
        leaf.codebook.resize(1);
        leaf.codebook[0] = std::move(centreSeq);
        leaf.assign.clear();
    } else {
        leaf.mode = LeafMode::CLUSTERED;
        std::vector<KFSeq> compactCB(Kfinal);
        for (int c = 0; c < Kfinal; ++c) compactCB[c] = leaf.codebook[centres[c]];
        leaf.codebook = std::move(compactCB);
        leaf.assign.resize(N);
        for (int i = 0; i < N; ++i) leaf.assign[i] = static_cast<uint8_t>(assign[i]);
    }
}

// ============================================================
// checkElide — returns BT_CHILD_AIR / BT_CHILD_INTERIOR / UINT32_MAX (no elide)
// Only elides UNIFORM leaves whose single sequence is constant
// at bgAir or bgInterior across all T frames.
// ============================================================
static uint32_t checkElide(const LeafBlock& leaf, float bgAir, float bgInterior, int T) {
    if (leaf.mode != LeafMode::UNIFORM || leaf.codebook.empty()) return UINT32_MAX;
    const KFSeq& seq = leaf.codebook[0];
    if (seq.empty()) return BT_CHILD_AIR;

    // Sample at 3 points to check constancy
    float v0  = kfInterp(seq, 0.0f);
    float vT  = kfInterp(seq, static_cast<float>(T - 1));
    float mid = kfInterp(seq, static_cast<float>((T - 1) / 2));

    bool allAir = (std::fabs(v0 - bgAir) < 0.5f &&
                   std::fabs(vT - bgAir) < 0.5f &&
                   std::fabs(mid - bgAir) < 0.5f);
    bool allInt = (std::fabs(v0 - bgInterior) < 0.5f &&
                   std::fabs(vT - bgInterior) < 0.5f &&
                   std::fabs(mid - bgInterior) < 0.5f);

    if (allAir) return BT_CHILD_AIR;
    if (allInt) return BT_CHILD_INTERIOR;
    return UINT32_MAX;
}

static double computeLeafAdaptiveClusterThreshold(
    const LeafBlock& leaf,
    double baseClusterThr)
{
    if (baseClusterThr <= 0.0) return baseClusterThr;

    std::vector<int> lengths;
    lengths.reserve(leaf.codebook.size());
    double sumLen = 0.0;
    for (const auto& seq : leaf.codebook) {
        const int n = static_cast<int>(seq.size());
        lengths.push_back(n);
        sumLen += n;
    }
    if (lengths.empty()) return baseClusterThr;

    std::sort(lengths.begin(), lengths.end());
    const double meanLen = sumLen / lengths.size();
    const double p90Len = static_cast<double>(lengths[std::min(lengths.size() - 1, (lengths.size() * 9) / 10)]);

    // Safe block-aware variant:
    // - keep complex blocks at the global baseline
    // - only slightly relax very smooth blocks to improve compression
    // This avoids the large full-pipeline regressions observed with the
    // earlier bidirectional scaling rule.
    const double meanNorm = std::clamp((meanLen - 2.0) / 10.0, 0.0, 1.0);
    const double p90Norm = std::clamp((p90Len - 4.0) / 12.0, 0.0, 1.0);
    const double complexity = std::max(meanNorm, p90Norm);
    const double smoothness = 1.0 - complexity;
    const double scale = 1.0 + 0.05 * std::pow(std::clamp(smoothness, 0.0, 1.0), 1.5);
    return baseClusterThr * scale;
}

struct LeafBudgetStats {
    double range = 0.0;
    double temporalNorm = 0.0;
    double spatialP90Norm = 0.0;
    double spatialP99Norm = 0.0;
    double frameSpanP90Norm = 0.0;
};

static LeafBudgetStats computeLeafBudgetStats(
    const LeafBlock& temporalLeaf,
    const RawVolume4D& rawVolume,
    int ox, int oy, int oz,
    int W, int H, int D, int T)
{
    LeafBudgetStats stats;
    double rawMin = std::numeric_limits<double>::infinity();
    double rawMax = -std::numeric_limits<double>::infinity();
    double temporalErr2 = 0.0;
    long long count = 0;
    std::vector<float> spatialDiffs;
    spatialDiffs.reserve(BT_LEAF_VOXELS * 4);
    std::vector<float> frameSpans;
    frameSpans.reserve(8);

    std::vector<int> sampleTimes;
    sampleTimes.push_back(0);
    sampleTimes.push_back(std::max(0, T / 4));
    sampleTimes.push_back(std::max(0, T / 2));
    sampleTimes.push_back(std::max(0, (3 * T) / 4));
    sampleTimes.push_back(std::max(0, T - 1));
    std::sort(sampleTimes.begin(), sampleTimes.end());
    sampleTimes.erase(std::unique(sampleTimes.begin(), sampleTimes.end()), sampleTimes.end());

    for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
        for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
            for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                const int gx = ox + lx;
                const int gy = oy + ly;
                const int gz = oz + lz;
                if (gx >= W || gy >= H || gz >= D) continue;
                const int localIdx = lx + BT_LEAF_SIZE * (ly + BT_LEAF_SIZE * lz);

                for (int t : sampleTimes) {
                    const float raw = rawVolume[t][gz][gy][gx];
                    const float temporal = kfInterp(temporalLeaf.codebook[localIdx], static_cast<float>(t));
                    const double te = static_cast<double>(raw) - temporal;
                    temporalErr2 += te * te;
                    rawMin = std::min(rawMin, static_cast<double>(raw));
                    rawMax = std::max(rawMax, static_cast<double>(raw));
                    ++count;

                    if (lx + 1 < BT_LEAF_SIZE && gx + 1 < W) {
                        spatialDiffs.push_back(std::abs(rawVolume[t][gz][gy][gx + 1] - raw));
                    }
                    if (ly + 1 < BT_LEAF_SIZE && gy + 1 < H) {
                        spatialDiffs.push_back(std::abs(rawVolume[t][gz][gy + 1][gx] - raw));
                    }
                    if (lz + 1 < BT_LEAF_SIZE && gz + 1 < D) {
                        spatialDiffs.push_back(std::abs(rawVolume[t][gz + 1][gy][gx] - raw));
                    }
                }
            }
        }
    }

    if (count == 0) return stats;
    stats.range = std::max(1e-6, rawMax - rawMin);
    stats.temporalNorm = std::sqrt(temporalErr2 / count) / stats.range;
    if (!spatialDiffs.empty()) {
        std::sort(spatialDiffs.begin(), spatialDiffs.end());
        const size_t idx90 = std::min(spatialDiffs.size() - 1, static_cast<size_t>(0.90 * static_cast<double>(spatialDiffs.size() - 1)));
        const size_t idx99 = std::min(spatialDiffs.size() - 1, static_cast<size_t>(0.99 * static_cast<double>(spatialDiffs.size() - 1)));
        stats.spatialP90Norm = spatialDiffs[idx90] / stats.range;
        stats.spatialP99Norm = spatialDiffs[idx99] / stats.range;
    }
    for (int t : sampleTimes) {
        double frameMin = std::numeric_limits<double>::infinity();
        double frameMax = -std::numeric_limits<double>::infinity();
        bool any = false;
        for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
            for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                    const int gx = ox + lx;
                    const int gy = oy + ly;
                    const int gz = oz + lz;
                    if (gx >= W || gy >= H || gz >= D) continue;
                    const float raw = rawVolume[t][gz][gy][gx];
                    frameMin = std::min(frameMin, static_cast<double>(raw));
                    frameMax = std::max(frameMax, static_cast<double>(raw));
                    any = true;
                }
            }
        }
        if (any) frameSpans.push_back(static_cast<float>((frameMax - frameMin) / stats.range));
    }
    if (!frameSpans.empty()) {
        std::sort(frameSpans.begin(), frameSpans.end());
        const size_t idx90 = std::min(frameSpans.size() - 1, static_cast<size_t>(0.90 * static_cast<double>(frameSpans.size() - 1)));
        stats.frameSpanP90Norm = frameSpans[idx90];
    }
    return stats;
}

static double computeLeafBudgetAwareClusterThreshold(
    const LeafBudgetStats& stats,
    double baseClusterThr)
{
    if (baseClusterThr <= 0.0) return baseClusterThr;
    const double temporalTightness = std::clamp((0.08 - stats.temporalNorm) / 0.06, 0.0, 1.0);
    const double spatialRisk = std::clamp((stats.spatialP90Norm - 0.10) / 0.25, 0.0, 1.0);
    const double edgeRisk = std::clamp((stats.spatialP99Norm - 0.25) / 0.50, 0.0, 1.0);
    const double frameSplitRisk = std::clamp((stats.frameSpanP90Norm - 0.55) / 0.35, 0.0, 1.0);
    const double risk = std::clamp(0.35 * spatialRisk + 0.20 * edgeRisk + 0.45 * frameSplitRisk, 0.0, 1.0) * temporalTightness;
    const double scale = 1.0 - 0.45 * risk;
    return baseClusterThr * std::clamp(scale, 0.55, 1.0);
}

static float sampleLeafLocal(const LeafBlock& leaf, int localIdx, float t) {
    const int lx = localIdx & 7;
    const int ly = (localIdx >> 3) & 7;
    const int lz = (localIdx >> 6) & 7;
    auto residualRank = [&](int idx) -> int {
        int rank = 0;
        const int wordIdx = idx >> 6;
        const int bitIdx = idx & 63;
        for (int i = 0; i < wordIdx; ++i) rank += bt_popcount64(leaf.residualMask[i]);
        if (bitIdx) rank += bt_popcount64(leaf.residualMask[wordIdx] & ((1ull << bitIdx) - 1ull));
        return rank;
    };
    switch (leaf.mode) {
    case LeafMode::UNIFORM:
        return kfInterp(leaf.codebook[0], t);
    case LeafMode::CLUSTERED:
        return kfInterp(leaf.codebook[leaf.assign[localIdx]], t);
    case LeafMode::GRID4:
        return sampleLeafGrid4(leaf, lx, ly, lz, t);
    case LeafMode::GRID4_RESIDUAL: {
        float base = sampleLeafGrid4(leaf, lx, ly, lz, t);
        const int wordIdx = localIdx >> 6;
        const int bitIdx = localIdx & 63;
        if (((leaf.residualMask[wordIdx] >> bitIdx) & 1ull) == 0ull) return base;
        const int rank = residualRank(localIdx);
        if (rank < 0 || rank >= static_cast<int>(leaf.residualCodebook.size())) return base;
        const float q = kfInterp(leaf.residualCodebook[rank], t);
        return base + q * leaf.residualScale;
    }
    case LeafMode::GRID4_MULTISCALE: {
        float base = sampleLeafGrid4(leaf, lx, ly, lz, t);
        if (leaf.fineGridDim <= 1 || leaf.residualCodebook.empty()) return base;
        return base +
               sampleLeafRegularControlGridMasked(
                   leaf.residualCodebook,
                   leaf.residualMask,
                   static_cast<int>(leaf.fineGridDim),
                   lx,
                   ly,
                   lz,
                   t) * leaf.residualScale;
    }
    default:
        return kfInterp(leaf.codebook[localIdx], t);
    }
}

struct LeafRawErrorStats {
    double range = 0.0;
    double temporalRmse = 0.0;
    double clusteredRmse = 0.0;
    double extraNorm = 0.0;
    double amplification = 1.0;
    double maxNorm = 0.0;
    double p95Norm = 0.0;
};

static LeafRawErrorStats computeLeafRawErrorStats(
    const LeafBlock& denseLeaf,
    const LeafBlock& clusteredLeaf,
    const RawVolume4D& rawVolume,
    int ox, int oy, int oz,
    int W, int H, int D, int T)
{
    LeafRawErrorStats stats;
    double rawMin = std::numeric_limits<double>::infinity();
    double rawMax = -std::numeric_limits<double>::infinity();
    double temporalErr2 = 0.0;
    double clusteredErr2 = 0.0;
    long long count = 0;
    std::vector<float> clusteredAbsErrs;
    clusteredAbsErrs.reserve(4 * 4 * 4 * T);

    std::vector<int> sampleTimes;
    sampleTimes.push_back(0);
    sampleTimes.push_back(std::max(0, T / 4));
    sampleTimes.push_back(std::max(0, T / 2));
    sampleTimes.push_back(std::max(0, (3 * T) / 4));
    sampleTimes.push_back(std::max(0, T - 1));
    std::sort(sampleTimes.begin(), sampleTimes.end());
    sampleTimes.erase(std::unique(sampleTimes.begin(), sampleTimes.end()), sampleTimes.end());

    for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
        for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
            for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                const int gx = ox + lx;
                const int gy = oy + ly;
                const int gz = oz + lz;
                if (gx >= W || gy >= H || gz >= D) continue;
                const int localIdx = lx + BT_LEAF_SIZE * (ly + BT_LEAF_SIZE * lz);
                for (int t : sampleTimes) {
                    const float raw = rawVolume[t][gz][gy][gx];
                    const float temporal = kfInterp(denseLeaf.codebook[localIdx], static_cast<float>(t));
                    const float clustered = sampleLeafLocal(clusteredLeaf, localIdx, static_cast<float>(t));
                    rawMin = std::min(rawMin, static_cast<double>(raw));
                    rawMax = std::max(rawMax, static_cast<double>(raw));
                    const double te = static_cast<double>(raw) - temporal;
                    const double ce = static_cast<double>(raw) - clustered;
                    temporalErr2 += te * te;
                    clusteredErr2 += ce * ce;
                    clusteredAbsErrs.push_back(static_cast<float>(std::abs(ce)));
                    ++count;
                }
            }
        }
    }

    if (count == 0) return stats;
    stats.range = std::max(1e-6, rawMax - rawMin);
    stats.temporalRmse = std::sqrt(temporalErr2 / count);
    stats.clusteredRmse = std::sqrt(clusteredErr2 / count);
    const double extraRmse = std::sqrt(std::max(0.0, stats.clusteredRmse * stats.clusteredRmse -
                                                     stats.temporalRmse * stats.temporalRmse));
    stats.extraNorm = extraRmse / stats.range;
    stats.amplification = stats.clusteredRmse / std::max(1e-8, stats.temporalRmse);
    if (!clusteredAbsErrs.empty()) {
        std::sort(clusteredAbsErrs.begin(), clusteredAbsErrs.end());
        stats.maxNorm = clusteredAbsErrs.back() / stats.range;
        const size_t idx95 = std::min(clusteredAbsErrs.size() - 1,
                                      static_cast<size_t>(0.95 * clusteredAbsErrs.size()));
        stats.p95Norm = clusteredAbsErrs[idx95] / stats.range;
    }
    return stats;
}

static double hotspotBudgetP95Norm(double temporalNorm) {
    return std::max(0.05, 2.5 * temporalNorm);
}

static double hotspotBudgetClusteredNorm(double temporalNorm) {
    return std::max(0.35, 6.0 * temporalNorm);
}

static double computeHotspotSeverity(
    const LeafRawErrorStats& stats,
    double temporalNorm)
{
    const double clusteredNorm = stats.clusteredRmse / std::max(1e-6, stats.range);
    const double budgetClustered = hotspotBudgetClusteredNorm(temporalNorm);
    const double budgetP95 = hotspotBudgetP95Norm(temporalNorm);
    const double clusteredOver = clusteredNorm / std::max(1e-6, budgetClustered);
    const double p95Over = stats.p95Norm / std::max(1e-6, budgetP95);
    const double maxOver = stats.maxNorm / std::max(0.20, 1.5 + 10.0 * temporalNorm);
    const double ampOver = stats.amplification / 8.0;
    return std::max(std::max(clusteredOver, p95Over), std::max(maxOver, ampOver));
}

static double hotspotScore(const LeafRawErrorStats& stats) {
    const double clusteredNorm = stats.clusteredRmse / std::max(1e-6, stats.range);
    return 1.5 * clusteredNorm + 1.0 * stats.p95Norm + 0.35 * stats.maxNorm +
           0.10 * std::log2(std::max(1.0, stats.amplification));
}

static LeafBlock buildDenseLeafFromCompVolume(
    const std::vector<std::vector<std::vector<std::vector<Point1D>>>>& compVol,
    int ox, int oy, int oz,
    int W, int H, int D,
    float bgAir)
{
    LeafBlock leaf;
    leaf.mode = LeafMode::DENSE;
    leaf.codebook.resize(BT_LEAF_VOXELS);

    for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
        for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
            for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                const int gx = ox + lx;
                const int gy = oy + ly;
                const int gz = oz + lz;
                const int localIdx = lx + BT_LEAF_SIZE * (ly + BT_LEAF_SIZE * lz);
                KFSeq seq;
                if (gx < W && gy < H && gz < D) {
                    const auto& pts = compVol[gz][gy][gx];
                    seq.reserve(pts.size());
                    for (const auto& p : pts) {
                        seq.push_back({static_cast<uint16_t>(p.index),
                                       f32_to_f16(static_cast<float>(p.value))});
                    }
                } else {
                    seq.push_back({0, f32_to_f16(bgAir)});
                }
                leaf.codebook[localIdx] = std::move(seq);
            }
        }
    }
    return leaf;
}

struct DensityBackgroundDecision {
    uint32_t elideValue = UINT32_MAX;
    bool makeUniform = false;
    float leafMin = 0.0f;
    float leafMax = 0.0f;
    float leafRange = 0.0f;
};

static DensityBackgroundDecision classifyDensityBackgroundLeaf(
    const LeafBlock& sourceLeaf,
    const FieldProfile& profile)
{
    DensityBackgroundDecision out;
    if (profile.type != FieldType::DENSITY) return out;
    if (profile.den.render_cutoff <= 0.0f) return out;
    if (sourceLeaf.codebook.empty()) return out;

    float leafMin = std::numeric_limits<float>::infinity();
    float leafMax = -std::numeric_limits<float>::infinity();
    for (const auto& seq : sourceLeaf.codebook) {
        for (const auto& kf : seq) {
            const float v = f16_to_f32(kf.v);
            leafMin = std::min(leafMin, v);
            leafMax = std::max(leafMax, v);
        }
    }
    if (!std::isfinite(leafMin) || !std::isfinite(leafMax)) return out;

    out.leafMin = leafMin;
    out.leafMax = leafMax;
    out.leafRange = std::max(0.0f, leafMax - leafMin);

    const float cutoff = profile.den.render_cutoff;
    const float cutoffBand = std::max(0.0f, profile.den.cutoff_band);
    const float bgZeroHi = std::max(1e-9f, profile.den.bg_zero_ratio * cutoff);
    const float bgConstHi = std::max(bgZeroHi, profile.den.bg_const_ratio * cutoff);
    const float nearVisibleFloor = std::max(bgZeroHi, 0.25f * std::max(cutoffBand, cutoff * 0.10f));

    if (leafMax <= bgZeroHi) {
        out.elideValue = BT_CHILD_AIR;
        return out;
    }

    if (leafMax <= bgConstHi &&
        leafMax < cutoff - cutoffBand &&
        out.leafRange <= nearVisibleFloor)
    {
        out.makeUniform = true;
    }

    return out;
}

static LeafBlock buildUniformLeafFromMeanSeries(
    const LeafBlock& sourceLeaf,
    int T,
    const FieldProfile& profile)
{
    LeafBlock leaf;
    leaf.mode = LeafMode::UNIFORM;
    leaf.codebook.resize(1);
    if (sourceLeaf.codebook.empty() || T <= 0) {
        leaf.codebook[0].push_back({0, f32_to_f16(0.0f)});
        return leaf;
    }

    std::vector<float> meanSeries(static_cast<size_t>(T), 0.0f);
    const float invN = 1.0f / std::max(1, static_cast<int>(sourceLeaf.codebook.size()));
    for (int t = 0; t < T; ++t) {
        double sum = 0.0;
        for (const auto& seq : sourceLeaf.codebook) {
            sum += kfInterp(seq, static_cast<float>(t));
        }
        meanSeries[static_cast<size_t>(t)] = static_cast<float>(sum * invN);
    }

    std::vector<int> kfIdx = detectKeyFrames(meanSeries, 0.0, profile);
    if (kfIdx.empty()) kfIdx.push_back(0);
    KFSeq seq;
    seq.reserve(kfIdx.size());
    for (int idx : kfIdx) {
        seq.push_back({static_cast<uint16_t>(idx),
                       f32_to_f16(meanSeries[static_cast<size_t>(idx)])});
    }
    leaf.codebook[0] = std::move(seq);
    return leaf;
}

static bool isSdfSurfaceCriticalSeq(const KFSeq& seq, const FieldProfile& profile)
{
    if (profile.type != FieldType::SDF || seq.empty()) return false;
    const float iso = profile.sdf.iso;
    const float critical = profile.sdf.w_critical;

    auto nearCritical = [&](float v) {
        return std::abs(v - iso) <= critical;
    };
    for (const auto& kf : seq) {
        if (nearCritical(f16_to_f32(kf.v))) return true;
    }
    for (size_t i = 1; i < seq.size(); ++i) {
        const float v0 = f16_to_f32(seq[i - 1].v) - iso;
        const float v1 = f16_to_f32(seq[i].v) - iso;
        if ((v0 <= 0.0f && v1 >= 0.0f) || (v0 >= 0.0f && v1 <= 0.0f)) return true;
    }
    return false;
}

static std::array<uint8_t, BT_LEAF_VOXELS> buildSdfSurfaceCriticalMask(
    const LeafBlock& denseLeaf,
    const FieldProfile& profile)
{
    std::array<uint8_t, BT_LEAF_VOXELS> mask{};

    auto idxOf = [](int lx, int ly, int lz) {
        return lx + BT_LEAF_SIZE * (ly + BT_LEAF_SIZE * lz);
    };

    for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
        for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
            for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                const int idx = idxOf(lx, ly, lz);
                if (isSdfSurfaceCriticalSeq(denseLeaf.codebook[static_cast<size_t>(idx)], profile)) {
                    mask[static_cast<size_t>(idx)] = 1;
                }
            }
        }
    }

    std::array<uint8_t, BT_LEAF_VOXELS> expanded = mask;
    const int dx[6] = {1, -1, 0, 0, 0, 0};
    const int dy[6] = {0, 0, 1, -1, 0, 0};
    const int dz[6] = {0, 0, 0, 0, 1, -1};
    for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
        for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
            for (int lx = 0; lx < BT_LEAF_SIZE; ++lx) {
                const int idx = idxOf(lx, ly, lz);
                if (!mask[static_cast<size_t>(idx)]) continue;
                for (int k = 0; k < 6; ++k) {
                    const int nx = lx + dx[k];
                    const int ny = ly + dy[k];
                    const int nz = lz + dz[k];
                    if (nx < 0 || ny < 0 || nz < 0 ||
                        nx >= BT_LEAF_SIZE || ny >= BT_LEAF_SIZE || nz >= BT_LEAF_SIZE) {
                        continue;
                    }
                    expanded[static_cast<size_t>(idxOf(nx, ny, nz))] = 1;
                }
            }
        }
    }

    return expanded;
}

static std::array<int, 4> grid4ControlCoordsLocal() {
    return {0, 2, 5, 7};
}

static LeafBlock buildGrid4LeafFromDenseLeaf(const LeafBlock& denseLeaf)
{
    LeafBlock leaf;
    leaf.mode = LeafMode::GRID4;
    leaf.codebook.reserve(64);
    const auto ctrl = grid4ControlCoordsLocal();
    for (int gz = 0; gz < 4; ++gz) {
        for (int gy = 0; gy < 4; ++gy) {
            for (int gx = 0; gx < 4; ++gx) {
                const int lx = ctrl[gx];
                const int ly = ctrl[gy];
                const int lz = ctrl[gz];
                const int localIdx = lx + BT_LEAF_SIZE * (ly + BT_LEAF_SIZE * lz);
                leaf.codebook.push_back(denseLeaf.codebook[localIdx]);
            }
        }
    }
    return leaf;
}

static LeafBlock buildGrid4LeafFromCompVolumeControls(
    const std::vector<std::vector<std::vector<std::vector<Point1D>>>>& compVol,
    int ox, int oy, int oz,
    int W, int H, int D,
    float bgAir)
{
    LeafBlock leaf;
    leaf.mode = LeafMode::GRID4;
    leaf.codebook.reserve(64);
    const auto ctrl = grid4ControlCoordsLocal();
    for (int gz = 0; gz < 4; ++gz) {
        for (int gy = 0; gy < 4; ++gy) {
            for (int gx = 0; gx < 4; ++gx) {
                const int lx = ctrl[gx];
                const int ly = ctrl[gy];
                const int lz = ctrl[gz];
                const int x = ox + lx;
                const int y = oy + ly;
                const int z = oz + lz;
                KFSeq seq;
                if (x < W && y < H && z < D) {
                    const auto& pts = compVol[z][y][x];
                    seq.reserve(std::max<size_t>(1, pts.size()));
                    if (pts.empty()) {
                        seq.push_back({0, f32_to_f16(bgAir)});
                    } else {
                        for (const auto& p : pts) {
                            seq.push_back({static_cast<uint16_t>(p.index),
                                           f32_to_f16(static_cast<float>(p.value))});
                        }
                    }
                } else {
                    seq.push_back({0, f32_to_f16(bgAir)});
                }
                leaf.codebook.push_back(std::move(seq));
            }
        }
    }
    return leaf;
}

static std::array<float, 3> grid4AxisPos(int localCoord) {
    const float u = (3.0f * static_cast<float>(localCoord)) / 7.0f;
    int i0 = static_cast<int>(std::floor(u));
    i0 = std::clamp(i0, 0, 2);
    return {static_cast<float>(i0), static_cast<float>(i0 + 1), u - static_cast<float>(i0)};
}

static const std::vector<int>& regularControlCoordsForDim(int gridDim) {
    static const std::vector<int> grid4 = {0, 2, 5, 7};
    static const std::vector<int> grid6 = {0, 1, 3, 4, 6, 7};
    static const std::vector<int> grid8 = {0, 1, 2, 3, 4, 5, 6, 7};
    switch (gridDim) {
    case 4: return grid4;
    case 6: return grid6;
    case 8: return grid8;
    default: return grid8;
    }
}

static std::array<float, 3> regularGridAxisPos(const std::vector<int>& coords, int localCoord) {
    if (coords.size() <= 1) return {0.0f, 0.0f, 0.0f};
    if (localCoord <= coords.front()) return {0.0f, 1.0f, 0.0f};
    if (localCoord >= coords.back()) {
        const int last = static_cast<int>(coords.size()) - 1;
        return {static_cast<float>(last - 1), static_cast<float>(last), 1.0f};
    }
    for (int i = 0; i + 1 < static_cast<int>(coords.size()); ++i) {
        const int c0 = coords[static_cast<size_t>(i)];
        const int c1 = coords[static_cast<size_t>(i + 1)];
        if (localCoord >= c0 && localCoord <= c1) {
            const float a = (c1 > c0)
                ? (static_cast<float>(localCoord - c0) / static_cast<float>(c1 - c0))
                : 0.0f;
            return {static_cast<float>(i), static_cast<float>(i + 1), a};
        }
    }
    const int last = static_cast<int>(coords.size()) - 1;
    return {static_cast<float>(last - 1), static_cast<float>(last), 1.0f};
}

static double polyCoordNorm(int localCoord) {
    return (2.0 * static_cast<double>(localCoord) / 7.0) - 1.0;
}

static std::array<double, 11> poly11BasisAt(int lx, int ly, int lz) {
    const double x = polyCoordNorm(lx);
    const double y = polyCoordNorm(ly);
    const double z = polyCoordNorm(lz);
    return {1.0, x, y, z, x * y, x * z, y * z, x * y * z, x * x, y * y, z * z};
}

static bool invertSmallMatrixLocal(std::vector<double>& a, int n) {
    std::vector<double> inv(static_cast<size_t>(n * n), 0.0);
    for (int i = 0; i < n; ++i) inv[static_cast<size_t>(i * n + i)] = 1.0;

    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double best = std::abs(a[static_cast<size_t>(col * n + col)]);
        for (int r = col + 1; r < n; ++r) {
            const double v = std::abs(a[static_cast<size_t>(r * n + col)]);
            if (v > best) { best = v; pivot = r; }
        }
        if (best < 1e-12) return false;
        if (pivot != col) {
            for (int c = 0; c < n; ++c) {
                std::swap(a[static_cast<size_t>(col * n + c)], a[static_cast<size_t>(pivot * n + c)]);
                std::swap(inv[static_cast<size_t>(col * n + c)], inv[static_cast<size_t>(pivot * n + c)]);
            }
        }
        const double diag = a[static_cast<size_t>(col * n + col)];
        for (int c = 0; c < n; ++c) {
            a[static_cast<size_t>(col * n + c)] /= diag;
            inv[static_cast<size_t>(col * n + c)] /= diag;
        }
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            const double factor = a[static_cast<size_t>(r * n + col)];
            if (std::abs(factor) < 1e-20) continue;
            for (int c = 0; c < n; ++c) {
                a[static_cast<size_t>(r * n + c)] -= factor * a[static_cast<size_t>(col * n + c)];
                inv[static_cast<size_t>(r * n + c)] -= factor * inv[static_cast<size_t>(col * n + c)];
            }
        }
    }
    a.swap(inv);
    return true;
}

static const std::array<std::array<double, BT_LEAF_VOXELS>, 11>& poly11PseudoInverseLocal() {
    static const std::array<std::array<double, BT_LEAF_VOXELS>, 11> pinv = []() {
        constexpr int M = BT_LEAF_VOXELS;
        constexpr int N = 11;
        std::array<std::array<double, M>, N> out{};
        std::array<std::array<double, N>, M> A{};
        int idx = 0;
        for (int lz = 0; lz < BT_LEAF_SIZE; ++lz) {
            for (int ly = 0; ly < BT_LEAF_SIZE; ++ly) {
                for (int lx = 0; lx < BT_LEAF_SIZE; ++lx, ++idx) {
                    A[static_cast<size_t>(idx)] = poly11BasisAt(lx, ly, lz);
                }
            }
        }
        std::vector<double> ata(static_cast<size_t>(N * N), 0.0);
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                double sum = 0.0;
                for (int i = 0; i < M; ++i) sum += A[static_cast<size_t>(i)][static_cast<size_t>(r)] * A[static_cast<size_t>(i)][static_cast<size_t>(c)];
                ata[static_cast<size_t>(r * N + c)] = sum;
            }
        }
        if (!invertSmallMatrixLocal(ata, N)) {
            throw std::runtime_error("poly11PseudoInverseLocal: singular system");
        }
        for (int r = 0; r < N; ++r) {
            for (int i = 0; i < M; ++i) {
                double sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += ata[static_cast<size_t>(r * N + k)] * A[static_cast<size_t>(i)][static_cast<size_t>(k)];
                }
                out[static_cast<size_t>(r)][static_cast<size_t>(i)] = sum;
            }
        }
        return out;
    }();
    return pinv;
}

static std::array<float, 11> fitPoly11CoeffsFromSamples(const std::array<float, BT_LEAF_VOXELS>& values) {
    std::array<float, 11> coeffs{};
    const auto& pinv = poly11PseudoInverseLocal();
    for (int r = 0; r < 11; ++r) {
        double sum = 0.0;
        for (int i = 0; i < BT_LEAF_VOXELS; ++i) {
            sum += pinv[static_cast<size_t>(r)][static_cast<size_t>(i)] * static_cast<double>(values[static_cast<size_t>(i)]);
        }
        coeffs[static_cast<size_t>(r)] = static_cast<float>(sum);
    }
    return coeffs;
}

static float evalPoly11CoeffsAt(const std::vector<KFSeq>& codebook, int lx, int ly, int lz, float t) {
    const auto basis = poly11BasisAt(lx, ly, lz);
    float sum = 0.0f;
    for (int i = 0; i < 11; ++i) {
        sum += static_cast<float>(basis[static_cast<size_t>(i)]) * kfInterp(codebook[static_cast<size_t>(i)], t);
    }
    return sum;
}

static LeafBlock buildPoly11LeafFromDenseLeaf(const LeafBlock& denseLeaf, int T, const FieldProfile& profile) {
    LeafBlock leaf;
    leaf.mode = LeafMode::POLY11;
    leaf.codebook.resize(11);
    std::array<std::vector<float>, 11> coeffSeries;
    for (auto& s : coeffSeries) s.resize(T);

    for (int t = 0; t < T; ++t) {
        std::array<float, BT_LEAF_VOXELS> vals{};
        for (int i = 0; i < BT_LEAF_VOXELS; ++i) {
            vals[static_cast<size_t>(i)] = kfInterp(denseLeaf.codebook[static_cast<size_t>(i)], static_cast<float>(t));
        }
        const auto coeffs = fitPoly11CoeffsFromSamples(vals);
        for (int c = 0; c < 11; ++c) coeffSeries[static_cast<size_t>(c)][static_cast<size_t>(t)] = coeffs[static_cast<size_t>(c)];
    }

    for (int c = 0; c < 11; ++c) {
        std::vector<int> kfIdx = detectKeyFrames(coeffSeries[static_cast<size_t>(c)], 0.0, profile);
        KFSeq seq;
        seq.reserve(kfIdx.size());
        for (int idx : kfIdx) {
            seq.push_back(KFPoint{static_cast<uint16_t>(idx), f32_to_f16(coeffSeries[static_cast<size_t>(c)][static_cast<size_t>(idx)])});
        }
        leaf.codebook[static_cast<size_t>(c)] = std::move(seq);
    }
    return leaf;
}

static float sampleLeafGrid4(const LeafBlock& leaf, int lx, int ly, int lz, float t) {
    auto sampleNode = [&](int ix, int iy, int iz) -> float {
        const int ctrlIdx = ix + 4 * (iy + 4 * iz);
        return kfInterp(leaf.codebook[ctrlIdx], t);
    };

    const auto gx = grid4AxisPos(lx);
    const auto gy = grid4AxisPos(ly);
    const auto gz = grid4AxisPos(lz);
    const int x0 = static_cast<int>(gx[0]), x1 = static_cast<int>(gx[1]);
    const int y0 = static_cast<int>(gy[0]), y1 = static_cast<int>(gy[1]);
    const int z0 = static_cast<int>(gz[0]), z1 = static_cast<int>(gz[1]);
    const float ax = gx[2], ay = gy[2], az = gz[2];

    const float c000 = sampleNode(x0, y0, z0);
    const float c100 = sampleNode(x1, y0, z0);
    const float c010 = sampleNode(x0, y1, z0);
    const float c110 = sampleNode(x1, y1, z0);
    const float c001 = sampleNode(x0, y0, z1);
    const float c101 = sampleNode(x1, y0, z1);
    const float c011 = sampleNode(x0, y1, z1);
    const float c111 = sampleNode(x1, y1, z1);

    const float c00 = c000 + ax * (c100 - c000);
    const float c10 = c010 + ax * (c110 - c010);
    const float c01 = c001 + ax * (c101 - c001);
    const float c11 = c011 + ax * (c111 - c011);
    const float c0 = c00 + ay * (c10 - c00);
    const float c1 = c01 + ay * (c11 - c01);
    return c0 + az * (c1 - c0);
}

static float sampleLeafRegularControlGrid(const std::vector<KFSeq>& codebook, int gridDim, int lx, int ly, int lz, float t) {
    if (gridDim <= 1 || static_cast<int>(codebook.size()) != gridDim * gridDim * gridDim) {
        return 0.0f;
    }
    const auto& coords = regularControlCoordsForDim(gridDim);
    auto sampleNode = [&](int ix, int iy, int iz) -> float {
        const int idx = ix + gridDim * (iy + gridDim * iz);
        return kfInterp(codebook[static_cast<size_t>(idx)], t);
    };
    const auto gx = regularGridAxisPos(coords, lx);
    const auto gy = regularGridAxisPos(coords, ly);
    const auto gz = regularGridAxisPos(coords, lz);
    const int x0 = static_cast<int>(gx[0]), x1 = static_cast<int>(gx[1]);
    const int y0 = static_cast<int>(gy[0]), y1 = static_cast<int>(gy[1]);
    const int z0 = static_cast<int>(gz[0]), z1 = static_cast<int>(gz[1]);
    const float ax = gx[2], ay = gy[2], az = gz[2];

    const float c000 = sampleNode(x0, y0, z0);
    const float c100 = sampleNode(x1, y0, z0);
    const float c010 = sampleNode(x0, y1, z0);
    const float c110 = sampleNode(x1, y1, z0);
    const float c001 = sampleNode(x0, y0, z1);
    const float c101 = sampleNode(x1, y0, z1);
    const float c011 = sampleNode(x0, y1, z1);
    const float c111 = sampleNode(x1, y1, z1);

    const float c00 = c000 + ax * (c100 - c000);
    const float c10 = c010 + ax * (c110 - c010);
    const float c01 = c001 + ax * (c101 - c001);
    const float c11 = c011 + ax * (c111 - c011);
    const float c0 = c00 + ay * (c10 - c00);
    const float c1 = c01 + ay * (c11 - c01);
    return c0 + az * (c1 - c0);
}

static float sampleLeafRegularControlGridMasked(
    const std::vector<KFSeq>& codebook,
    const std::array<uint64_t, 8>& mask,
    int gridDim,
    int lx,
    int ly,
    int lz,
    float t)
{
    if (gridDim <= 1 || codebook.empty()) {
        return 0.0f;
    }
    const auto& coords = regularControlCoordsForDim(gridDim);
    const int total = gridDim * gridDim * gridDim;
    auto sampleNode = [&](int ix, int iy, int iz) -> float {
        const int idx = ix + gridDim * (iy + gridDim * iz);
        if (idx < 0 || idx >= total) return 0.0f;
        const int wordIdx = idx >> 6;
        const int bitIdx = idx & 63;
        if (((mask[static_cast<size_t>(wordIdx)] >> bitIdx) & 1ull) == 0ull) return 0.0f;
        int rank = 0;
        for (int i = 0; i < wordIdx; ++i) rank += bt_popcount64(mask[static_cast<size_t>(i)]);
        if (bitIdx) rank += bt_popcount64(mask[static_cast<size_t>(wordIdx)] & ((1ull << bitIdx) - 1ull));
        if (rank < 0 || rank >= static_cast<int>(codebook.size())) return 0.0f;
        return kfInterp(codebook[static_cast<size_t>(rank)], t);
    };
    const auto gx = regularGridAxisPos(coords, lx);
    const auto gy = regularGridAxisPos(coords, ly);
    const auto gz = regularGridAxisPos(coords, lz);
    const int x0 = static_cast<int>(gx[0]), x1 = static_cast<int>(gx[1]);
    const int y0 = static_cast<int>(gy[0]), y1 = static_cast<int>(gy[1]);
    const int z0 = static_cast<int>(gz[0]), z1 = static_cast<int>(gz[1]);
    const float ax = gx[2], ay = gy[2], az = gz[2];

    const float c000 = sampleNode(x0, y0, z0);
    const float c100 = sampleNode(x1, y0, z0);
    const float c010 = sampleNode(x0, y1, z0);
    const float c110 = sampleNode(x1, y1, z0);
    const float c001 = sampleNode(x0, y0, z1);
    const float c101 = sampleNode(x1, y0, z1);
    const float c011 = sampleNode(x0, y1, z1);
    const float c111 = sampleNode(x1, y1, z1);

    const float c00 = c000 + ax * (c100 - c000);
    const float c10 = c010 + ax * (c110 - c010);
    const float c01 = c001 + ax * (c101 - c001);
    const float c11 = c011 + ax * (c111 - c011);
    const float c0 = c00 + ay * (c10 - c00);
    const float c1 = c01 + ay * (c11 - c01);
    return c0 + az * (c1 - c0);
}

// ============================================================
// BlockTree::build  (4D temporal path)
// ============================================================
void BlockTree::build(
    const std::vector<std::vector<std::vector<std::vector<Point1D>>>>& compVol,
    int W, int H, int D, int T,
    double thresh_uniform,
    double thresh_cluster,
    const FieldProfile& profile,
    bool enableBackgroundElision,
    bool enableBlockAwareCluster,
    bool enableBudgetAwareCluster,
    bool enableGuardedMedoidCluster,
    bool enableGrid4Spatial,
    bool enableGrid4ControlOnlyTemporal,
    const RawVolume4D* rawVolume,
    bool enableValidateFallback,
    bool enableHotspotSecondPass)
{
    dimX = W; dimY = H; dimZ = D; numFrames = T;

    constexpr int BLOCK_SPAN = BT_LEAF_SIZE * BT_INTERNAL_SIZE;  // 32

    rootDimX = (W + BLOCK_SPAN - 1) / BLOCK_SPAN;
    rootDimY = (H + BLOCK_SPAN - 1) / BLOCK_SPAN;
    rootDimZ = (D + BLOCK_SPAN - 1) / BLOCK_SPAN;
    rootTable.assign(rootDimX * rootDimY * rootDimZ, BT_CHILD_AIR);
    internalNodes.clear();
    leafBlocks.clear();
    childList.clear();
    outlierVoxels       = 0;
    mode2FallbackBlocks = 0;
    blockAwareCluster = enableBlockAwareCluster;
    budgetAwareCluster = enableBudgetAwareCluster;
    guardedMedoidCluster = enableGuardedMedoidCluster;
    grid4Spatial = enableGrid4Spatial;
    blockAwareClusterMin = std::numeric_limits<double>::infinity();
    blockAwareClusterMax = 0.0;
    blockAwareClusterMean = 0.0;
    validateFallback = enableValidateFallback;
    hotspotSecondPass = enableHotspotSecondPass;
    validateFallbackBlocks = 0;
    validateRetryBlocks = 0;
    validateDenseFallbackBlocks = 0;
    hotspotSecondPassBlocks = 0;
    hotspotSecondPassDenseBlocks = 0;
    hotspotRegionSeedBlocks = 0;
    hotspotRegionTouchedBlocks = 0;
    hotspotRegionDenseFallbackBlocks = 0;
    guardedGateRejects = 0;
    medoidCenterChanges = 0;
    qualityGatePromotions = 0;
    int blockAwareClusterCount = 0;

    const int leafCountX = (W + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafCountY = (H + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;
    const int leafCountZ = (D + BT_LEAF_SIZE - 1) / BT_LEAF_SIZE;

    std::vector<ChildEntry> pendingEntries;
    pendingEntries.reserve(leafCountX * leafCountY * leafCountZ);
    leafOrigins.clear();
    leafOrigins.reserve(leafCountX * leafCountY * leafCountZ);
    leafGridCoords.clear();
    leafGridCoords.reserve(leafCountX * leafCountY * leafCountZ);
    std::vector<double> leafClusterThresholds;
    leafClusterThresholds.reserve(leafCountX * leafCountY * leafCountZ);
    std::vector<int> leafGridToId(leafCountX * leafCountY * leafCountZ, -1);
    auto runClusterWithCounters = [&](LeafBlock& targetLeaf, double localThr,
                                      int& localOutlierVoxels,
                                      int& localMode2FallbackBlocks,
                                      int& localGuardedGateRejects,
                                      int& localMedoidCenterChanges,
                                      int& localQualityGatePromotions,
                                      const std::array<uint8_t, BT_LEAF_VOXELS>* surfaceCriticalMask = nullptr) {
        if (enableGuardedMedoidCluster && profile.type != FieldType::SDF) {
            clusterLeafGuarded(targetLeaf, localThr, T, profile,
                               localOutlierVoxels, localMode2FallbackBlocks,
                               localGuardedGateRejects,
                               localMedoidCenterChanges,
                               localQualityGatePromotions,
                               surfaceCriticalMask);
        } else {
            clusterLeaf(targetLeaf, localThr, T, profile,
                        localOutlierVoxels, localMode2FallbackBlocks,
                        surfaceCriticalMask);
        }
    };
    auto runCluster = [&](LeafBlock& targetLeaf, double localThr,
                          const std::array<uint8_t, BT_LEAF_VOXELS>* surfaceCriticalMask = nullptr) {
        runClusterWithCounters(targetLeaf, localThr,
                               outlierVoxels, mode2FallbackBlocks,
                               guardedGateRejects,
                               medoidCenterChanges,
                               qualityGatePromotions,
                               surfaceCriticalMask);
    };

    int elidedAirCnt = 0, elidedIntCnt = 0;
    const int firstNewNode = 0;
    const int totalLeafCount = leafCountX * leafCountY * leafCountZ;
    internalNodes.assign(rootDimX * rootDimY * rootDimZ, InternalNode{});
    for (int rootIdx = 0; rootIdx < static_cast<int>(rootTable.size()); ++rootIdx) {
        rootTable[rootIdx] = static_cast<uint32_t>(rootIdx) + BT_CHILD_ID_BASE;
    }
    struct LeafBuildResult {
        LeafBlock leaf;
        double clusterThr = 0.0;
        uint32_t elideValue = UINT32_MAX;
        bool hasLeaf = false;
    };
    std::vector<LeafBuildResult> leafResults(totalLeafCount);

#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        int localOutlierVoxels = 0;
        int localMode2FallbackBlocks = 0;
        int localGuardedGateRejects = 0;
        int localMedoidCenterChanges = 0;
        int localQualityGatePromotions = 0;
        int localElidedAirCnt = 0;
        int localElidedIntCnt = 0;
        int localValidateFallbackBlocks = 0;
        int localValidateRetryBlocks = 0;
        int localValidateDenseFallbackBlocks = 0;
        int localHotspotSecondPassBlocks = 0;
        int localHotspotSecondPassDenseBlocks = 0;
        double localBlockAwareMin = std::numeric_limits<double>::infinity();
        double localBlockAwareMax = 0.0;
        double localBlockAwareSum = 0.0;
        int localBlockAwareCount = 0;

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (int leafFlat = 0; leafFlat < totalLeafCount; ++leafFlat) {
            const int bx = leafFlat % leafCountX;
            const int by = (leafFlat / leafCountX) % leafCountY;
            const int bz = leafFlat / (leafCountX * leafCountY);
            const int ox = bx * BT_LEAF_SIZE;
            const int oy = by * BT_LEAF_SIZE;
            const int oz = bz * BT_LEAF_SIZE;

            const bool useGrid4Leaf =
                enableGrid4Spatial;
            LeafBlock leaf;
            LeafBlock denseLeafBeforeCluster;
            std::array<uint8_t, BT_LEAF_VOXELS> surfaceCriticalMask{};
            bool hasSurfaceCriticalMask = false;
            bool densityBackgroundMerged = false;
            if (useGrid4Leaf && enableGrid4ControlOnlyTemporal) {
                leaf = buildGrid4LeafFromCompVolumeControls(compVol, ox, oy, oz, W, H, D, bgAir);
            } else {
                leaf.mode = LeafMode::DENSE;
                leaf.codebook.resize(BT_LEAF_VOXELS);
                for (int lz = 0; lz < BT_LEAF_SIZE; lz++) {
                    for (int ly = 0; ly < BT_LEAF_SIZE; ly++) {
                        for (int lx = 0; lx < BT_LEAF_SIZE; lx++) {
                            const int gx = ox + lx, gy = oy + ly, gz = oz + lz;
                            const int localIdx = lx + BT_LEAF_SIZE * (ly + BT_LEAF_SIZE * lz);
                            KFSeq seq;
                            if (gx < W && gy < H && gz < D) {
                                const auto& pts = compVol[gz][gy][gx];
                                seq.reserve(std::max<size_t>(1, pts.size()));
                                if (pts.empty()) {
                                    seq.push_back({0, f32_to_f16(bgAir)});
                                } else {
                                    for (const auto& p : pts) {
                                        seq.push_back({static_cast<uint16_t>(p.index),
                                                       f32_to_f16(static_cast<float>(p.value))});
                                    }
                                }
                            } else {
                                seq.push_back({0, f32_to_f16(bgAir)});
                            }
                            leaf.codebook[localIdx] = std::move(seq);
                        }
                    }
                }
                denseLeafBeforeCluster = leaf;
                if (profile.type == FieldType::SDF) {
                    surfaceCriticalMask = buildSdfSurfaceCriticalMask(denseLeafBeforeCluster, profile);
                    hasSurfaceCriticalMask = true;
                }
            }

            double localClusterThr = thresh_cluster;
            if (enableBudgetAwareCluster &&
                rawVolume != nullptr &&
                profile.type != FieldType::SDF &&
                thresh_cluster > 0.0)
            {
                const LeafBudgetStats budgetStats = computeLeafBudgetStats(
                    denseLeafBeforeCluster, *rawVolume, ox, oy, oz, W, H, D, T);
                localClusterThr = computeLeafBudgetAwareClusterThreshold(budgetStats, thresh_cluster);
                localBlockAwareMin = std::min(localBlockAwareMin, localClusterThr);
                localBlockAwareMax = std::max(localBlockAwareMax, localClusterThr);
                localBlockAwareSum += localClusterThr;
                ++localBlockAwareCount;
            } else if (enableBlockAwareCluster && profile.type != FieldType::SDF && thresh_cluster > 0.0) {
                localClusterThr = computeLeafAdaptiveClusterThreshold(leaf, thresh_cluster);
                localBlockAwareMin = std::min(localBlockAwareMin, localClusterThr);
                localBlockAwareMax = std::max(localBlockAwareMax, localClusterThr);
                localBlockAwareSum += localClusterThr;
                ++localBlockAwareCount;
            }

            if (useGrid4Leaf) {
                if (!enableGrid4ControlOnlyTemporal) {
                    leaf = buildGrid4LeafFromDenseLeaf(denseLeafBeforeCluster);
                }
            } else {
                runClusterWithCounters(leaf, localClusterThr,
                                       localOutlierVoxels, localMode2FallbackBlocks,
                                       localGuardedGateRejects,
                                       localMedoidCenterChanges,
                                       localQualityGatePromotions,
                                       hasSurfaceCriticalMask ? &surfaceCriticalMask : nullptr);
            }

            if (enableBackgroundElision && profile.type == FieldType::DENSITY && profile.den.render_cutoff > 0.0f) {
                const LeafBlock& semanticLeaf =
                    (!denseLeafBeforeCluster.codebook.empty()) ? denseLeafBeforeCluster : leaf;
                const DensityBackgroundDecision bgDecision =
                    classifyDensityBackgroundLeaf(semanticLeaf, profile);
                if (bgDecision.elideValue == BT_CHILD_AIR) {
                    ++localElidedAirCnt;
                    leafResults[leafFlat].clusterThr = localClusterThr;
                    leafResults[leafFlat].elideValue = BT_CHILD_AIR;
                    leafResults[leafFlat].hasLeaf = false;
                    continue;
                }
                if (bgDecision.makeUniform) {
                    leaf = buildUniformLeafFromMeanSeries(semanticLeaf, T, profile);
                    densityBackgroundMerged = true;
                }
            }

            if (enableValidateFallback &&
                rawVolume != nullptr &&
                profile.type != FieldType::SDF &&
                leaf.mode != LeafMode::DENSE &&
                leaf.mode != LeafMode::GRID4 &&
                !densityBackgroundMerged)
            {
                LeafRawErrorStats stats = computeLeafRawErrorStats(
                    denseLeafBeforeCluster, leaf, *rawVolume, ox, oy, oz, W, H, D, T);
                const double temporalNorm = stats.temporalRmse / std::max(1e-6, stats.range);
                const double allowedP95Norm = std::max(0.05, 3.0 * temporalNorm);
                const double allowedMaxNorm = std::max(0.35, 14.0 * temporalNorm);
                const bool needsRetry =
                    (stats.maxNorm > allowedMaxNorm && stats.p95Norm > allowedP95Norm);
                if (needsRetry) {
                    LeafBlock retryLeaf = denseLeafBeforeCluster;
                    const double retryClusterThr = std::max(0.35, localClusterThr * 0.70);
                    runClusterWithCounters(retryLeaf, retryClusterThr,
                                           localOutlierVoxels, localMode2FallbackBlocks,
                                           localGuardedGateRejects,
                                           localMedoidCenterChanges,
                                           localQualityGatePromotions);
                    const LeafRawErrorStats retryStats = computeLeafRawErrorStats(
                        denseLeafBeforeCluster, retryLeaf, *rawVolume, ox, oy, oz, W, H, D, T);
                    const bool retryBetter =
                        (retryStats.p95Norm < stats.p95Norm * 0.85) ||
                        (retryStats.maxNorm < stats.maxNorm * 0.75) ||
                        (retryStats.clusteredRmse < stats.clusteredRmse * 0.97);
                    if (retryBetter) {
                        leaf = std::move(retryLeaf);
                        stats = retryStats;
                        ++localValidateRetryBlocks;
                    }
                    const bool stillCatastrophic =
                        (stats.maxNorm > std::max(1.50, 12.0 * temporalNorm)) &&
                        (stats.p95Norm > std::max(0.20, 3.0 * temporalNorm));
                    const bool safeToDenseFallback = temporalNorm < 0.12;
                    if (stillCatastrophic && safeToDenseFallback) {
                        leaf = denseLeafBeforeCluster;
                        ++localValidateDenseFallbackBlocks;
                    }
                    if (retryBetter || (stillCatastrophic && safeToDenseFallback)) {
                        ++localValidateFallbackBlocks;
                    }
                }
            }

            if (enableHotspotSecondPass &&
                rawVolume != nullptr &&
                profile.type != FieldType::SDF &&
                leaf.mode != LeafMode::DENSE &&
                leaf.mode != LeafMode::GRID4 &&
                !densityBackgroundMerged)
            {
                LeafRawErrorStats stats = computeLeafRawErrorStats(
                    denseLeafBeforeCluster, leaf, *rawVolume, ox, oy, oz, W, H, D, T);
                const double temporalNorm = stats.temporalRmse / std::max(1e-6, stats.range);
                const double clusteredNorm = stats.clusteredRmse / std::max(1e-6, stats.range);
                const double hotspotSeverity = computeHotspotSeverity(stats, temporalNorm);
                const bool hotspot =
                    (clusteredNorm > hotspotBudgetClusteredNorm(temporalNorm)) ||
                    (stats.amplification > 8.0 && stats.p95Norm > hotspotBudgetP95Norm(temporalNorm)) ||
                    (stats.maxNorm > 1.0 && stats.p95Norm > 0.20);
                if (hotspot) {
                    ++localHotspotSecondPassBlocks;
                    LeafBlock bestLeaf = leaf;
                    LeafRawErrorStats bestStats = stats;
                    double bestScore = hotspotScore(stats);
                    const double budgetP95 = hotspotBudgetP95Norm(temporalNorm);
                    const double baseFactor = std::clamp(
                        std::sqrt(budgetP95 / std::max(1e-6, stats.p95Norm)),
                        0.08, 0.70);
                    std::vector<double> retryFactors;
                    retryFactors.push_back(baseFactor);
                    retryFactors.push_back(std::max(0.05, baseFactor * 0.50));
                    if (hotspotSeverity > 1.5) retryFactors.push_back(std::max(0.05, baseFactor * 0.25));
                    if (hotspotSeverity > 2.5) retryFactors.push_back(std::max(0.05, baseFactor * 0.125));
                    std::sort(retryFactors.begin(), retryFactors.end());
                    retryFactors.erase(std::unique(retryFactors.begin(), retryFactors.end()), retryFactors.end());
                    for (double factor : retryFactors) {
                        LeafBlock retryLeaf = denseLeafBeforeCluster;
                        const double retryClusterThr = std::max(0.20, localClusterThr * factor);
                        runClusterWithCounters(retryLeaf, retryClusterThr,
                                               localOutlierVoxels, localMode2FallbackBlocks,
                                               localGuardedGateRejects,
                                               localMedoidCenterChanges,
                                               localQualityGatePromotions);
                        const LeafRawErrorStats retryStats = computeLeafRawErrorStats(
                            denseLeafBeforeCluster, retryLeaf, *rawVolume, ox, oy, oz, W, H, D, T);
                        const double retryScore = hotspotScore(retryStats);
                        if (retryScore < bestScore * 0.98 ||
                            retryStats.clusteredRmse < bestStats.clusteredRmse * 0.995 ||
                            retryStats.p95Norm < bestStats.p95Norm * 0.92 ||
                            retryStats.maxNorm < bestStats.maxNorm * 0.85) {
                            bestLeaf = std::move(retryLeaf);
                            bestStats = retryStats;
                            bestScore = retryScore;
                        }
                    }
                    const double bestSeverity = computeHotspotSeverity(bestStats, temporalNorm);
                    const bool stillHotspot =
                        (bestStats.clusteredRmse / std::max(1e-6, bestStats.range) > hotspotBudgetClusteredNorm(temporalNorm)) ||
                        (bestStats.amplification > 8.0 && bestStats.p95Norm > hotspotBudgetP95Norm(temporalNorm));
                    if (stillHotspot &&
                        temporalNorm < 0.10 &&
                        (bestSeverity > 3.5 || bestStats.p95Norm > 0.18 || bestStats.amplification > 24.0)) {
                        bestLeaf = denseLeafBeforeCluster;
                        ++localHotspotSecondPassDenseBlocks;
                    }
                    leaf = std::move(bestLeaf);
                }
            }

            const uint32_t elide = enableBackgroundElision
                ? checkElide(leaf, bgAir, bgInterior, T)
                : UINT32_MAX;
            if (elide == BT_CHILD_AIR) {
                ++localElidedAirCnt;
            } else if (elide == BT_CHILD_INTERIOR) {
                ++localElidedIntCnt;
            }
            leafResults[leafFlat].clusterThr = localClusterThr;
            leafResults[leafFlat].elideValue = elide;
            if (elide != BT_CHILD_AIR && elide != BT_CHILD_INTERIOR) {
                leafResults[leafFlat].leaf = std::move(leaf);
                leafResults[leafFlat].hasLeaf = true;
            }
        }
#ifdef _OPENMP
#pragma omp critical
        {
            outlierVoxels += localOutlierVoxels;
            mode2FallbackBlocks += localMode2FallbackBlocks;
            guardedGateRejects += localGuardedGateRejects;
            medoidCenterChanges += localMedoidCenterChanges;
            qualityGatePromotions += localQualityGatePromotions;
            elidedAirCnt += localElidedAirCnt;
            elidedIntCnt += localElidedIntCnt;
            validateFallbackBlocks += localValidateFallbackBlocks;
            validateRetryBlocks += localValidateRetryBlocks;
            validateDenseFallbackBlocks += localValidateDenseFallbackBlocks;
            hotspotSecondPassBlocks += localHotspotSecondPassBlocks;
            hotspotSecondPassDenseBlocks += localHotspotSecondPassDenseBlocks;
            if (localBlockAwareCount > 0) {
                blockAwareClusterMin = std::min(blockAwareClusterMin, localBlockAwareMin);
                blockAwareClusterMax = std::max(blockAwareClusterMax, localBlockAwareMax);
                blockAwareClusterMean += localBlockAwareSum;
                blockAwareClusterCount += localBlockAwareCount;
            }
        }
    }
#endif

    pendingEntries.reserve(totalLeafCount);
    for (int leafFlat = 0; leafFlat < totalLeafCount; ++leafFlat) {
        const int bx = leafFlat % leafCountX;
        const int by = (leafFlat / leafCountX) % leafCountY;
        const int bz = leafFlat / (leafCountX * leafCountY);
        const int ox = bx * BT_LEAF_SIZE;
        const int oy = by * BT_LEAF_SIZE;
        const int oz = bz * BT_LEAF_SIZE;
        int rx, ry, rz, ix, iy, iz, lx0, ly0, lz0;
        decompose(ox, oy, oz, rx, ry, rz, ix, iy, iz, lx0, ly0, lz0);
        const int rootIdx = rx + rootDimX * (ry + rootDimY * rz);
        const int childIdx = ix + BT_INTERNAL_SIZE * (iy + BT_INTERNAL_SIZE * iz);
        uint32_t childValue = 0;
        if (leafResults[leafFlat].elideValue == BT_CHILD_AIR) {
            childValue = BT_CHILD_AIR;
        } else if (leafResults[leafFlat].elideValue == BT_CHILD_INTERIOR) {
            childValue = BT_CHILD_INTERIOR;
        } else {
            const int leafId = static_cast<int>(leafBlocks.size());
            leafBlocks.push_back(std::move(leafResults[leafFlat].leaf));
            leafOrigins.push_back({ox, oy, oz});
            leafGridCoords.push_back({bx, by, bz});
            leafClusterThresholds.push_back(leafResults[leafFlat].clusterThr);
            leafGridToId[leafFlat] = leafId;
            childValue = static_cast<uint32_t>(leafId) + BT_CHILD_ID_BASE;
        }
        pendingEntries.push_back({static_cast<uint32_t>(rootIdx),
                                  static_cast<uint16_t>(childIdx), childValue});
    }

    if (blockAwareClusterCount > 0) {
        blockAwareClusterMean /= static_cast<double>(blockAwareClusterCount);
    } else {
        blockAwareClusterMin = 0.0;
    }

    if (enableHotspotSecondPass &&
        !enableGrid4Spatial &&
        rawVolume != nullptr &&
        profile.type != FieldType::SDF &&
        !leafBlocks.empty())
    {
        struct RegionSeed {
            int leafId = -1;
            int bx = 0;
            int by = 0;
            int bz = 0;
            double temporalNorm = 0.0;
            double severity = 0.0;
        };

        std::vector<RegionSeed> seeds;
        seeds.reserve(leafBlocks.size() / 32);
        std::vector<double> targetFactors(leafBlocks.size(), 1.0);
        std::vector<double> targetSeverity(leafBlocks.size(), 0.0);

        for (int leafId = 0; leafId < static_cast<int>(leafBlocks.size()); ++leafId) {
            const auto [ox, oy, oz] = leafOrigins[leafId];
            LeafBlock denseLeaf = buildDenseLeafFromCompVolume(compVol, ox, oy, oz, W, H, D, bgAir);
            const LeafRawErrorStats stats = computeLeafRawErrorStats(
                denseLeaf, leafBlocks[leafId], *rawVolume, ox, oy, oz, W, H, D, T);
            const double temporalNorm = stats.temporalRmse / std::max(1e-6, stats.range);
            const double severity = computeHotspotSeverity(stats, temporalNorm);
            const bool seed =
                (
                    stats.amplification > 16.0 &&
                    stats.p95Norm > std::max(0.12, 4.0 * hotspotBudgetP95Norm(temporalNorm))
                ) ||
                (
                    severity > 2.5 &&
                    stats.p95Norm > std::max(0.18, 5.0 * hotspotBudgetP95Norm(temporalNorm))
                ) ||
                (
                    stats.maxNorm > 1.50 &&
                    stats.p95Norm > 0.20
                );
            if (!seed) continue;

            const auto [gbx, gby, gbz] = leafGridCoords[leafId];
            seeds.push_back({leafId, gbx, gby, gbz, temporalNorm, severity});
        }
        hotspotRegionSeedBlocks = static_cast<int>(seeds.size());

        for (const RegionSeed& seed : seeds) {
            const double centerFactor =
                (seed.severity > 4.0) ? 0.08 :
                (seed.severity > 3.0) ? 0.15 :
                (seed.severity > 2.0) ? 0.25 :
                (seed.severity > 1.5) ? 0.35 : 0.50;

            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nbx = seed.bx + dx;
                        const int nby = seed.by + dy;
                        const int nbz = seed.bz + dz;
                        if (nbx < 0 || nbx >= leafCountX ||
                            nby < 0 || nby >= leafCountY ||
                            nbz < 0 || nbz >= leafCountZ) {
                            continue;
                        }
                        const int neighborLeafId = leafGridToId[nbx + leafCountX * (nby + leafCountY * nbz)];
                        if (neighborLeafId < 0) continue;

                        const int chebDist = std::max({std::abs(dx), std::abs(dy), std::abs(dz)});
                        const double factor =
                            (chebDist == 0) ? centerFactor :
                            std::min(0.75, centerFactor + 0.20 * chebDist);
                        targetFactors[neighborLeafId] = std::min(targetFactors[neighborLeafId], factor);
                        targetSeverity[neighborLeafId] = std::max(targetSeverity[neighborLeafId], seed.severity);
                    }
                }
            }
        }

        const int denseCap = std::max(256, leafCountX * leafCountY * leafCountZ / 24);
        for (int leafId = 0; leafId < static_cast<int>(leafBlocks.size()); ++leafId) {
            if (targetFactors[leafId] >= 0.999) continue;

            const auto [ox, oy, oz] = leafOrigins[leafId];
            LeafBlock denseLeaf = buildDenseLeafFromCompVolume(compVol, ox, oy, oz, W, H, D, bgAir);
            const LeafRawErrorStats beforeStats = computeLeafRawErrorStats(
                denseLeaf, leafBlocks[leafId], *rawVolume, ox, oy, oz, W, H, D, T);
            const double temporalNorm = beforeStats.temporalRmse / std::max(1e-6, beforeStats.range);

            LeafBlock retryLeaf = denseLeaf;
            const double retryClusterThr = std::max(0.15, leafClusterThresholds[leafId] * targetFactors[leafId]);
            runCluster(retryLeaf, retryClusterThr);
            const LeafRawErrorStats retryStats = computeLeafRawErrorStats(
                denseLeaf, retryLeaf, *rawVolume, ox, oy, oz, W, H, D, T);

            const double beforeScore = hotspotScore(beforeStats);
            const double retryScore = hotspotScore(retryStats);
            bool accepted = false;
            if (retryScore < beforeScore * 0.95 ||
                retryStats.p95Norm < beforeStats.p95Norm * 0.88 ||
                retryStats.maxNorm < beforeStats.maxNorm * 0.80 ||
                retryStats.clusteredRmse < beforeStats.clusteredRmse * 0.985) {
                leafBlocks[leafId] = std::move(retryLeaf);
                ++hotspotRegionTouchedBlocks;
                accepted = true;
            }

            const LeafRawErrorStats finalStats = accepted
                ? computeLeafRawErrorStats(denseLeaf, leafBlocks[leafId], *rawVolume, ox, oy, oz, W, H, D, T)
                : beforeStats;
            const double finalSeverity = computeHotspotSeverity(finalStats, temporalNorm);
            if (finalSeverity > 3.5 &&
                targetSeverity[leafId] > 2.0 &&
                temporalNorm < 0.08 &&
                hotspotRegionDenseFallbackBlocks < denseCap) {
                leafBlocks[leafId] = std::move(denseLeaf);
                ++hotspotRegionDenseFallbackBlocks;
            }
        }
    }

    // ---- Phase 2: sort + compact internal nodes ----
    const int lastNewNode = static_cast<int>(internalNodes.size()) - 1;
    phase2_compact(pendingEntries, internalNodes, childList, firstNewNode, lastNewNode);

    printf("  build4D: leaves=%zu  elided_air=%d  elided_int=%d\n",
           leafBlocks.size(), elidedAirCnt, elidedIntCnt);
}

// ============================================================
// BlockTree::sample  (sparse lookup + KFSeq interpolation)
// ============================================================
float BlockTree::sample(int x, int y, int z, float t) const {
    int rx, ry, rz, ix, iy, iz, lx, ly, lz;
    decompose(x, y, z, rx, ry, rz, ix, iy, iz, lx, ly, lz);

    if (rx < 0 || rx >= rootDimX ||
        ry < 0 || ry >= rootDimY ||
        rz < 0 || rz >= rootDimZ) return bgAir;

    const uint32_t rootEntry = rootTable[rx + rootDimX * (ry + rootDimY * rz)];
    if (rootEntry == BT_CHILD_AIR)      return bgAir;
    if (rootEntry == BT_CHILD_INTERIOR) return bgInterior;

    const int internalId = static_cast<int>(rootEntry - BT_CHILD_ID_BASE);
    const InternalNode& node = internalNodes[internalId];
    const int childIdx = ix + BT_INTERNAL_SIZE * (iy + BT_INTERNAL_SIZE * iz);

    uint32_t childEntry;
    if (node.hasBit(childIdx)) {
        childEntry = childList[node.childBase + static_cast<uint64_t>(node.rankBefore(childIdx))];
    } else {
        childEntry = node.defaultVal;
    }

    if (childEntry == BT_CHILD_AIR)      return bgAir;
    if (childEntry == BT_CHILD_INTERIOR) return bgInterior;

    const int leafId = static_cast<int>(childEntry - BT_CHILD_ID_BASE);
    const LeafBlock& leaf = leafBlocks[leafId];
    const int localIdx = lx + BT_LEAF_SIZE * (ly + BT_LEAF_SIZE * lz);

    switch (leaf.mode) {
    case LeafMode::UNIFORM:
        return kfInterp(leaf.codebook[0], t);

    case LeafMode::CLUSTERED: {
        const int k = static_cast<int>(leaf.assign[localIdx]);
        return kfInterp(leaf.codebook[k], t);
    }

    case LeafMode::GRID4:
        return sampleLeafGrid4(leaf, lx, ly, lz, t);

    case LeafMode::GRID4_RESIDUAL: {
        float base = sampleLeafGrid4(leaf, lx, ly, lz, t);
        const int wordIdx = localIdx >> 6;
        const int bitIdx = localIdx & 63;
        if (((leaf.residualMask[wordIdx] >> bitIdx) & 1ull) == 0ull) return base;
        int rank = 0;
        for (int i = 0; i < wordIdx; ++i) rank += bt_popcount64(leaf.residualMask[i]);
        if (bitIdx) rank += bt_popcount64(leaf.residualMask[wordIdx] & ((1ull << bitIdx) - 1ull));
        if (rank < 0 || rank >= static_cast<int>(leaf.residualCodebook.size())) return base;
        return base + kfInterp(leaf.residualCodebook[rank], t) * leaf.residualScale;
    }

    case LeafMode::GRID4_MULTISCALE: {
        float base = sampleLeafGrid4(leaf, lx, ly, lz, t);
        if (leaf.fineGridDim <= 1 || leaf.residualCodebook.empty()) return base;
        return base +
               sampleLeafRegularControlGridMasked(
                   leaf.residualCodebook,
                   leaf.residualMask,
                   static_cast<int>(leaf.fineGridDim),
                   lx,
                   ly,
                   lz,
                   t) * leaf.residualScale;
    }

    case LeafMode::POLY11:
        return evalPoly11CoeffsAt(leaf.codebook, lx, ly, lz, t);

    case LeafMode::POLY11_RESIDUAL: {
        float base = evalPoly11CoeffsAt(leaf.codebook, lx, ly, lz, t);
        const int wordIdx = localIdx >> 6;
        const int bitIdx = localIdx & 63;
        if (((leaf.residualMask[wordIdx] >> bitIdx) & 1ull) == 0ull) return base;
        int rank = 0;
        for (int i = 0; i < wordIdx; ++i) rank += bt_popcount64(leaf.residualMask[i]);
        if (bitIdx) rank += bt_popcount64(leaf.residualMask[wordIdx] & ((1ull << bitIdx) - 1ull));
        if (rank < 0 || rank >= static_cast<int>(leaf.residualCodebook.size())) return base;
        return base + kfInterp(leaf.residualCodebook[rank], t) * leaf.residualScale;
    }

    default:  // DENSE
        return kfInterp(leaf.codebook[localIdx], t);
    }
}

// ============================================================
// SeqRef helpers — uint32 encodes both length N and B表 offset
//   bits [31:24] = N  (sequence length, max 255)
//   bits [23: 0] = absOffset into seqPool (B表)
// ============================================================
static inline uint32_t encodeSeqRef(int N, uint32_t absOffset) {
    return (static_cast<uint32_t>(N) << 24) | (absOffset & 0x00FFFFFFu);
}
static inline void decodeSeqRef(uint32_t ref, int& N, uint32_t& absOffset) {
    N         = static_cast<int>(ref >> 24);
    absOffset = ref & 0x00FFFFFFu;
}

static KFSeq decodeSeqMetaToKFSeq(const std::vector<SeqMeta>& seqMetaPool,
                                  const std::vector<uint32_t>& seqPool,
                                  uint32_t seqId) {
    KFSeq seq;
    if (seqId >= seqMetaPool.size()) return seq;

    const SeqMeta meta = seqMetaPool[seqId];
    const int N = static_cast<int>(meta.length);
    if (N <= 0) return seq;

    seq.resize(N);
    const int tWords = (N + 1) / 2;
    const uint32_t* tArr = seqPool.data() + meta.absOffset;
    const uint32_t* vArr = tArr + tWords;

    for (int i = 0; i < N; i++) {
        const uint32_t tw = tArr[i >> 1];
        const uint32_t vw = vArr[i >> 1];
        seq[i].t = static_cast<uint16_t>((i & 1) ? (tw >> 16) : (tw & 0xFFFFu));
        seq[i].v = static_cast<uint16_t>((i & 1) ? (vw >> 16) : (vw & 0xFFFFu));
    }
    return seq;
}

static KFSeq decodeSeqRefToKFSeq(const std::vector<uint32_t>& seqPool, uint32_t seqRef) {
    int N = 0;
    uint32_t absOffset = 0;
    decodeSeqRef(seqRef, N, absOffset);

    KFSeq seq;
    if (N <= 0) return seq;

    seq.resize(N);
    const int tWords = (N + 1) / 2;
    const uint32_t* tArr = seqPool.data() + absOffset;
    const uint32_t* vArr = tArr + tWords;

    for (int i = 0; i < N; i++) {
        const uint32_t tw = tArr[i >> 1];
        const uint32_t vw = vArr[i >> 1];
        seq[i].t = static_cast<uint16_t>((i & 1) ? (tw >> 16) : (tw & 0xFFFFu));
        seq[i].v = static_cast<uint16_t>((i & 1) ? (vw >> 16) : (vw & 0xFFFFu));
    }
    return seq;
}

// ============================================================
// B表 global sequence pool — SoA layout WITHOUT N word:
//   seqPool[absOffset .. absOffset + tWords - 1]  = t-array
//   seqPool[absOffset + tWords .. +tWords-1]       = v-array
//   tWords = ceil(N/2)   (each word packs 2×uint16)
// ============================================================
static uint64_t hashKFSeq(const KFSeq& seq) {
    uint64_t h = static_cast<uint64_t>(seq.size()) * 0x9e3779b97f4a7c15ULL;
    for (const auto& kf : seq) {
        h ^= h * 0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(kf.t);
        h ^= h * 0x6c62272e07bb0142ULL + static_cast<uint64_t>(kf.v);
    }
    return h;
}

uint32_t BlockTree::findOrAddSeqToPool(const KFSeq& seq) {
    const uint64_t h = hashKFSeq(seq);
    auto it = seqHashMap_.find(h);
    if (it != seqHashMap_.end()) return it->second;

    const int N = static_cast<int>(seq.size());
    const uint32_t absOffset = static_cast<uint32_t>(seqPool.size());
    const int tWords = (N + 1) / 2;

    // t-array: pack pairs of uint16
    for (int w = 0; w < tWords; w++) {
        const int i0 = w * 2, i1 = w * 2 + 1;
        const uint32_t t0 = (i0 < N) ? static_cast<uint32_t>(seq[i0].t) : 0u;
        const uint32_t t1 = (i1 < N) ? static_cast<uint32_t>(seq[i1].t) : 0u;
        seqPool.push_back(t0 | (t1 << 16));
    }
    // v-array: pack pairs of float16
    for (int w = 0; w < tWords; w++) {
        const int i0 = w * 2, i1 = w * 2 + 1;
        const uint32_t v0 = (i0 < N) ? static_cast<uint32_t>(seq[i0].v) : 0u;
        const uint32_t v1 = (i1 < N) ? static_cast<uint32_t>(seq[i1].v) : 0u;
        seqPool.push_back(v0 | (v1 << 16));
    }

    const uint32_t seqId = static_cast<uint32_t>(seqMetaPool.size());
    seqMetaPool.push_back(SeqMeta{absOffset, static_cast<uint16_t>(N), 0u});
    seqHashMap_[h] = seqId;
    return seqId;
}

// ============================================================
// flattenLeaves — build topoPool (A表) + seqPool (B表)
//
// A表 layout per leaf (at topoPool[topoOffset]):
//   UNIFORM  (mode=0): 1 SeqRef
//   CLUSTERED(mode=1): 128 assign words (4 bytes/voxel) + K SeqRefs
//   DENSE    (mode=2): 512 SeqRefs
//
// SeqRef = encodeSeqRef(N, absOffset):
//   N         = sequence length
//   absOffset = word offset into seqPool (B表)
// ============================================================
void BlockTree::flattenLeaves() {
    leafHeaders.clear();
    topoPool.clear();
    seqMetaPool.clear();
    seqPool.clear();
    seqHashMap_.clear();
    leafHeaders.resize(leafBlocks.size());

    struct FlattenPlan {
        uint8_t mode = 0;
        uint8_t numCenters = 0;
        uint32_t topoWords = 0;
    };

    const size_t leafCount = leafBlocks.size();
    std::vector<FlattenPlan> plans(leafCount);
    std::vector<uint32_t> topoOffsets(leafCount, 0);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int leafId = 0; leafId < static_cast<int>(leafCount); ++leafId) {
        const LeafBlock& leaf = leafBlocks[static_cast<size_t>(leafId)];
        FlattenPlan plan;
        switch (leaf.mode) {
        case LeafMode::UNIFORM:
            plan.mode = 0;
            plan.numCenters = 0;
            plan.topoWords = 1;
            break;
        case LeafMode::CLUSTERED:
            plan.mode = 1;
            plan.numCenters = static_cast<uint8_t>(leaf.codebook.size());
            plan.topoWords = 128u + static_cast<uint32_t>(leaf.codebook.size());
            break;
        case LeafMode::GRID4:
            plan.mode = 4;
            plan.numCenters = 64;
            plan.topoWords = 64;
            break;
        case LeafMode::GRID4_RESIDUAL:
            plan.mode = 5;
            plan.numCenters = 64;
            plan.topoWords = 64u + 16u + 2u + static_cast<uint32_t>(leaf.residualCodebook.size());
            break;
        case LeafMode::GRID4_MULTISCALE:
            plan.mode = 8;
            plan.numCenters = 64;
            plan.topoWords = 64u + 16u + 3u + static_cast<uint32_t>(leaf.residualCodebook.size());
            break;
        case LeafMode::POLY11:
            plan.mode = 6;
            plan.numCenters = 11;
            plan.topoWords = 11;
            break;
        case LeafMode::POLY11_RESIDUAL:
            plan.mode = 7;
            plan.numCenters = 11;
            plan.topoWords = 11u + 16u + 2u + static_cast<uint32_t>(leaf.residualCodebook.size());
            break;
        default:
            plan.mode = 2;
            plan.numCenters = 0;
            plan.topoWords = BT_LEAF_VOXELS;
            break;
        }
        plans[static_cast<size_t>(leafId)] = plan;
    }

    uint32_t topoCursor = 0;
    for (size_t leafId = 0; leafId < leafCount; ++leafId) {
        topoOffsets[leafId] = topoCursor;
        topoCursor += plans[leafId].topoWords;
    }
    topoPool.assign(topoCursor, 0u);
    std::vector<const KFSeq*> seqPtrByTopoWord(topoCursor, nullptr);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int leafId = 0; leafId < static_cast<int>(leafCount); ++leafId) {
        const LeafBlock& leaf = leafBlocks[static_cast<size_t>(leafId)];
        const FlattenPlan& plan = plans[static_cast<size_t>(leafId)];
        LeafHeader hdr;
        hdr.mode = plan.mode;
        hdr.numCenters = plan.numCenters;
        hdr.reserved = 0;
        hdr.topoOffset = topoOffsets[static_cast<size_t>(leafId)];
        leafHeaders[static_cast<size_t>(leafId)] = hdr;

        uint32_t* topo = topoPool.data() + hdr.topoOffset;
        const uint32_t base = hdr.topoOffset;

        switch (leaf.mode) {
        case LeafMode::UNIFORM:
            seqPtrByTopoWord[base] = &leaf.codebook[0];
            break;

        case LeafMode::CLUSTERED:
            for (int w = 0; w < 128; ++w) {
                uint32_t word = 0;
                for (int b = 0; b < 4; ++b) {
                    const int i = w * 4 + b;
                    if (i < static_cast<int>(leaf.assign.size())) {
                        word |= static_cast<uint32_t>(leaf.assign[i]) << (b * 8);
                    }
                }
                topo[w] = word;
            }
            for (int c = 0; c < static_cast<int>(leaf.codebook.size()); ++c) {
                seqPtrByTopoWord[base + 128u + static_cast<uint32_t>(c)] = &leaf.codebook[static_cast<size_t>(c)];
            }
            break;

        case LeafMode::GRID4:
            for (int i = 0; i < 64; ++i) {
                seqPtrByTopoWord[base + static_cast<uint32_t>(i)] = &leaf.codebook[static_cast<size_t>(i)];
            }
            break;

        case LeafMode::GRID4_RESIDUAL: {
            for (int i = 0; i < 64; ++i) {
                seqPtrByTopoWord[base + static_cast<uint32_t>(i)] = &leaf.codebook[static_cast<size_t>(i)];
            }
            uint32_t offset = 64;
            for (int i = 0; i < 8; ++i) {
                topo[offset++] = static_cast<uint32_t>(leaf.residualMask[static_cast<size_t>(i)] & 0xFFFFFFFFu);
                topo[offset++] = static_cast<uint32_t>(leaf.residualMask[static_cast<size_t>(i)] >> 32);
            }
            uint32_t scaleBits = 0;
            std::memcpy(&scaleBits, &leaf.residualScale, sizeof(uint32_t));
            topo[offset++] = scaleBits;
            topo[offset++] = static_cast<uint32_t>(leaf.residualCodebook.size());
            for (size_t i = 0; i < leaf.residualCodebook.size(); ++i) {
                seqPtrByTopoWord[base + offset + static_cast<uint32_t>(i)] = &leaf.residualCodebook[i];
            }
            break;
        }

        case LeafMode::GRID4_MULTISCALE: {
            for (int i = 0; i < 64; ++i) {
                seqPtrByTopoWord[base + static_cast<uint32_t>(i)] = &leaf.codebook[static_cast<size_t>(i)];
            }
            uint32_t offset = 64;
            topo[offset++] = static_cast<uint32_t>(leaf.fineGridDim);
            for (int i = 0; i < 8; ++i) {
                topo[offset++] = static_cast<uint32_t>(leaf.residualMask[static_cast<size_t>(i)] & 0xFFFFFFFFu);
                topo[offset++] = static_cast<uint32_t>(leaf.residualMask[static_cast<size_t>(i)] >> 32);
            }
            uint32_t scaleBits = 0;
            std::memcpy(&scaleBits, &leaf.residualScale, sizeof(uint32_t));
            topo[offset++] = scaleBits;
            topo[offset++] = static_cast<uint32_t>(leaf.residualCodebook.size());
            for (size_t i = 0; i < leaf.residualCodebook.size(); ++i) {
                seqPtrByTopoWord[base + offset + static_cast<uint32_t>(i)] = &leaf.residualCodebook[i];
            }
            break;
        }

        case LeafMode::POLY11:
            for (int i = 0; i < 11; ++i) {
                seqPtrByTopoWord[base + static_cast<uint32_t>(i)] = &leaf.codebook[static_cast<size_t>(i)];
            }
            break;

        case LeafMode::POLY11_RESIDUAL: {
            for (int i = 0; i < 11; ++i) {
                seqPtrByTopoWord[base + static_cast<uint32_t>(i)] = &leaf.codebook[static_cast<size_t>(i)];
            }
            uint32_t offset = 11;
            for (int i = 0; i < 8; ++i) {
                topo[offset++] = static_cast<uint32_t>(leaf.residualMask[static_cast<size_t>(i)] & 0xFFFFFFFFu);
                topo[offset++] = static_cast<uint32_t>(leaf.residualMask[static_cast<size_t>(i)] >> 32);
            }
            uint32_t scaleBits = 0;
            std::memcpy(&scaleBits, &leaf.residualScale, sizeof(uint32_t));
            topo[offset++] = scaleBits;
            topo[offset++] = static_cast<uint32_t>(leaf.residualCodebook.size());
            for (size_t i = 0; i < leaf.residualCodebook.size(); ++i) {
                seqPtrByTopoWord[base + offset + static_cast<uint32_t>(i)] = &leaf.residualCodebook[i];
            }
            break;
        }

        default:
            for (int i = 0; i < BT_LEAF_VOXELS; ++i) {
                seqPtrByTopoWord[base + static_cast<uint32_t>(i)] = &leaf.codebook[static_cast<size_t>(i)];
            }
            break;
        }
    }

    for (uint32_t topoIdx = 0; topoIdx < topoCursor; ++topoIdx) {
        const KFSeq* seq = seqPtrByTopoWord[static_cast<size_t>(topoIdx)];
        if (seq != nullptr) {
            topoPool[static_cast<size_t>(topoIdx)] = findOrAddSeqToPool(*seq);
        }
    }

    printf("flattenLeaves: leaves=%zu  topoPool=%zu words (%.1f KB)  seqMeta=%zu entries (%.1f KB)  seqPool=%zu words (%.1f KB)\n",
           leafBlocks.size(),
           topoPool.size(), topoPool.size() * 4.0 / 1024.0,
           seqMetaPool.size(), seqMetaPool.size() * sizeof(SeqMeta) / 1024.0,
           seqPool.size(),  seqPool.size()  * 4.0 / 1024.0);
}

// ============================================================
// BlockTree::sampleFlat — sample using leafHeaders + topoPool (A表) + seqPool (B表)
// ============================================================
float BlockTree::sampleFlat(int x, int y, int z, float t) const {
    int rx, ry, rz, ix, iy, iz, lx, ly, lz;
    decompose(x, y, z, rx, ry, rz, ix, iy, iz, lx, ly, lz);

    if (rx < 0 || rx >= rootDimX ||
        ry < 0 || ry >= rootDimY ||
        rz < 0 || rz >= rootDimZ) return bgAir;

    const uint32_t rootEntry = rootTable[rx + rootDimX * (ry + rootDimY * rz)];
    if (rootEntry == BT_CHILD_AIR)      return bgAir;
    if (rootEntry == BT_CHILD_INTERIOR) return bgInterior;

    const int internalId = static_cast<int>(rootEntry - BT_CHILD_ID_BASE);
    const InternalNode& node = internalNodes[internalId];
    const int childIdx = ix + BT_INTERNAL_SIZE * (iy + BT_INTERNAL_SIZE * iz);

    uint32_t childEntry;
    if (node.hasBit(childIdx)) {
        childEntry = childList[node.childBase + static_cast<uint64_t>(node.rankBefore(childIdx))];
    } else {
        childEntry = node.defaultVal;
    }

    if (childEntry == BT_CHILD_AIR)      return bgAir;
    if (childEntry == BT_CHILD_INTERIOR) return bgInterior;

    const int leafId   = static_cast<int>(childEntry - BT_CHILD_ID_BASE);
    const LeafHeader& hdr = leafHeaders[leafId];
    const uint32_t* topo  = topoPool.data() + hdr.topoOffset;
    const int localIdx    = lx + BT_LEAF_SIZE * (ly + BT_LEAF_SIZE * lz);

    // Decode SeqRef → interpolate from seqPool (B表, no N word)
    // B表 layout at absOffset: [t-array: tWords words][v-array: tWords words]
    //   tWords = ceil(N/2); each word packs 2×uint16 (even→lo, odd→hi)
    auto interpSeqEntry = [&](uint32_t seqEntry) -> float {
        int N = 0;
        uint32_t absOffset = 0;
        if (!seqMetaPool.empty()) {
            if (seqEntry >= seqMetaPool.size()) return 128.0f;
            const SeqMeta meta = seqMetaPool[seqEntry];
            N = static_cast<int>(meta.length);
            absOffset = meta.absOffset;
        } else {
            decodeSeqRef(seqEntry, N, absOffset);
        }
        if (N == 0) return 128.0f;

        const int tWords          = (N + 1) / 2;
        const uint32_t* tArr      = seqPool.data() + absOffset;
        const uint32_t* vArr      = tArr + tWords;   // vBase = absOffset + tWords

        auto getT = [&](int i) -> float {
            const uint32_t w = tArr[i >> 1];
            return static_cast<float>((i & 1) ? (w >> 16) : (w & 0xFFFFu));
        };
        auto getV = [&](int i) -> float {
            const uint32_t w = vArr[i >> 1];
            return f16_to_f32(static_cast<uint16_t>((i & 1) ? (w >> 16) : (w & 0xFFFFu)));
        };

        const float tf = t;
        if (tf <= getT(0))     return getV(0);
        if (tf >= getT(N - 1)) return getV(N - 1);

        int lo = 0, hi = N - 1;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) >> 1;
            if (getT(mid) <= tf) lo = mid; else hi = mid;
        }

        const float ft0 = getT(lo), ft1 = getT(hi);
        const float fv0 = getV(lo), fv1 = getV(hi);
        if (ft1 == ft0) return fv0;
        return fv0 + (tf - ft0) / (ft1 - ft0) * (fv1 - fv0);
    };

    auto interpGrid4 = [&]() -> float {
        auto sampleNode = [&](int ix, int iy, int iz) -> float {
            return interpSeqEntry(topo[ix + 4 * (iy + 4 * iz)]);
        };

        const auto gx = grid4AxisPos(lx);
        const auto gy = grid4AxisPos(ly);
        const auto gz = grid4AxisPos(lz);
        const int x0 = static_cast<int>(gx[0]), x1 = static_cast<int>(gx[1]);
        const int y0 = static_cast<int>(gy[0]), y1 = static_cast<int>(gy[1]);
        const int z0 = static_cast<int>(gz[0]), z1 = static_cast<int>(gz[1]);
        const float ax = gx[2], ay = gy[2], az = gz[2];

        const float c000 = sampleNode(x0, y0, z0);
        const float c100 = sampleNode(x1, y0, z0);
        const float c010 = sampleNode(x0, y1, z0);
        const float c110 = sampleNode(x1, y1, z0);
        const float c001 = sampleNode(x0, y0, z1);
        const float c101 = sampleNode(x1, y0, z1);
        const float c011 = sampleNode(x0, y1, z1);
        const float c111 = sampleNode(x1, y1, z1);

        const float c00 = c000 + ax * (c100 - c000);
        const float c10 = c010 + ax * (c110 - c010);
        const float c01 = c001 + ax * (c101 - c001);
        const float c11 = c011 + ax * (c111 - c011);
        const float c0 = c00 + ay * (c10 - c00);
        const float c1 = c01 + ay * (c11 - c01);
        return c0 + az * (c1 - c0);
    };
    auto interpRegularGrid = [&](const uint32_t* seqRefs, int gridDim) -> float {
        if (gridDim <= 1) return 0.0f;
        const auto& coords = regularControlCoordsForDim(gridDim);
        auto sampleNode = [&](int ix, int iy, int iz) -> float {
            return interpSeqEntry(seqRefs[ix + gridDim * (iy + gridDim * iz)]);
        };
        const auto gx = regularGridAxisPos(coords, lx);
        const auto gy = regularGridAxisPos(coords, ly);
        const auto gz = regularGridAxisPos(coords, lz);
        const int x0 = static_cast<int>(gx[0]), x1 = static_cast<int>(gx[1]);
        const int y0 = static_cast<int>(gy[0]), y1 = static_cast<int>(gy[1]);
        const int z0 = static_cast<int>(gz[0]), z1 = static_cast<int>(gz[1]);
        const float ax = gx[2], ay = gy[2], az = gz[2];

        const float c000 = sampleNode(x0, y0, z0);
        const float c100 = sampleNode(x1, y0, z0);
        const float c010 = sampleNode(x0, y1, z0);
        const float c110 = sampleNode(x1, y1, z0);
        const float c001 = sampleNode(x0, y0, z1);
        const float c101 = sampleNode(x1, y0, z1);
        const float c011 = sampleNode(x0, y1, z1);
        const float c111 = sampleNode(x1, y1, z1);

        const float c00 = c000 + ax * (c100 - c000);
        const float c10 = c010 + ax * (c110 - c010);
        const float c01 = c001 + ax * (c101 - c001);
        const float c11 = c011 + ax * (c111 - c011);
        const float c0 = c00 + ay * (c10 - c00);
        const float c1 = c01 + ay * (c11 - c01);
        return c0 + az * (c1 - c0);
    };

    switch (hdr.mode) {
    case 0:  // UNIFORM: topo[topoOffset] = 1 SeqRef
        return interpSeqEntry(topo[0]);

    case 1: {  // CLUSTERED: topo[0..127]=assign words, topo[128..128+K-1]=SeqRefs
        const int k = static_cast<int>((topo[localIdx / 4] >> ((localIdx % 4) * 8)) & 0xFFu);
        return interpSeqEntry(topo[128 + k]);
    }

    case 4:  // GRID4: topo[0..63] = 64 control-grid SeqRefs
        return interpGrid4();

    case 5: { // GRID4_RESIDUAL: 64 base + 16 mask words + scale + count + residual seq refs
        auto interpResidual = [&]() -> float {
            float base = interpGrid4();
            const uint32_t* maskWords = topo + 64;
            const int wordIdx = localIdx >> 5;
            const int bitIdx = localIdx & 31;
            if (((maskWords[wordIdx] >> bitIdx) & 1u) == 0u) return base;
            int rank = 0;
            for (int i = 0; i < wordIdx; ++i) rank += bt_popcount32(maskWords[i]);
            if (bitIdx) rank += bt_popcount32(maskWords[wordIdx] & ((1u << bitIdx) - 1u));
            uint32_t scaleBits = topo[80];
            float scale = 0.0f;
            std::memcpy(&scale, &scaleBits, sizeof(float));
            const uint32_t residualCount = topo[81];
            if (rank < 0 || rank >= static_cast<int>(residualCount)) return base;
            const float rq = interpSeqEntry(topo[82 + rank]);
            return base + rq * scale;
        };
        return interpResidual();
    }

    case 8: { // GRID4_MULTISCALE: 64 coarse refs + fineDim + mask + scale + count + masked fine residual refs
        float base = interpGrid4();
        const int fineDim = static_cast<int>(topo[64]);
        if (fineDim <= 1) return base;
        const uint32_t* maskWords = topo + 65;
        auto interpMaskedRegularGrid = [&](const uint32_t* seqRefs, int dim) -> float {
            const auto& coords = regularControlCoordsForDim(dim);
            auto sampleNode = [&](int ix, int iy, int iz) -> float {
                const int idx = ix + dim * (iy + dim * iz);
                const int wordIdx = idx >> 5;
                const int bitIdx = idx & 31;
                if (((maskWords[wordIdx] >> bitIdx) & 1u) == 0u) return 0.0f;
                int rank = 0;
                for (int i = 0; i < wordIdx; ++i) rank += bt_popcount32(maskWords[i]);
                if (bitIdx) rank += bt_popcount32(maskWords[wordIdx] & ((1u << bitIdx) - 1u));
                return interpSeqEntry(seqRefs[rank]);
            };
            const auto gx = regularGridAxisPos(coords, lx);
            const auto gy = regularGridAxisPos(coords, ly);
            const auto gz = regularGridAxisPos(coords, lz);
            const int x0 = static_cast<int>(gx[0]), x1 = static_cast<int>(gx[1]);
            const int y0 = static_cast<int>(gy[0]), y1 = static_cast<int>(gy[1]);
            const int z0 = static_cast<int>(gz[0]), z1 = static_cast<int>(gz[1]);
            const float ax = gx[2], ay = gy[2], az = gz[2];

            const float c000 = sampleNode(x0, y0, z0);
            const float c100 = sampleNode(x1, y0, z0);
            const float c010 = sampleNode(x0, y1, z0);
            const float c110 = sampleNode(x1, y1, z0);
            const float c001 = sampleNode(x0, y0, z1);
            const float c101 = sampleNode(x1, y0, z1);
            const float c011 = sampleNode(x0, y1, z1);
            const float c111 = sampleNode(x1, y1, z1);

            const float c00 = c000 + ax * (c100 - c000);
            const float c10 = c010 + ax * (c110 - c010);
            const float c01 = c001 + ax * (c101 - c001);
            const float c11 = c011 + ax * (c111 - c011);
            const float c0 = c00 + ay * (c10 - c00);
            const float c1 = c01 + ay * (c11 - c01);
            return c0 + az * (c1 - c0);
        };
        float scale = 0.0f;
        const uint32_t scaleBits = topo[81];
        std::memcpy(&scale, &scaleBits, sizeof(float));
        const uint32_t residualCount = topo[82];
        if (residualCount == 0) return base;
        return base + interpMaskedRegularGrid(topo + 83, fineDim) * scale;
    }

    case 6: { // POLY11: topo[0..10] = coefficient SeqRefs
        float sum = 0.0f;
        const auto basis = poly11BasisAt(lx, ly, lz);
        for (int i = 0; i < 11; ++i) {
            sum += static_cast<float>(basis[static_cast<size_t>(i)]) * interpSeqEntry(topo[i]);
        }
        return sum;
    }

    case 7: { // POLY11_RESIDUAL: 11 base + 16 mask words + scale + count + residual seq refs
        float base = 0.0f;
        const auto basis = poly11BasisAt(lx, ly, lz);
        for (int i = 0; i < 11; ++i) {
            base += static_cast<float>(basis[static_cast<size_t>(i)]) * interpSeqEntry(topo[i]);
        }
        const uint32_t* maskWords = topo + 11;
        const int wordIdx = localIdx >> 5;
        const int bitIdx = localIdx & 31;
        if (((maskWords[wordIdx] >> bitIdx) & 1u) == 0u) return base;
        int rank = 0;
        for (int i = 0; i < wordIdx; ++i) rank += bt_popcount32(maskWords[i]);
        if (bitIdx) rank += bt_popcount32(maskWords[wordIdx] & ((1u << bitIdx) - 1u));
        uint32_t scaleBits = topo[27];
        float scale = 0.0f;
        std::memcpy(&scale, &scaleBits, sizeof(float));
        const uint32_t residualCount = topo[28];
        if (rank < 0 || rank >= static_cast<int>(residualCount)) return base;
        const float rq = interpSeqEntry(topo[29 + rank]);
        return base + rq * scale;
    }

    default:  // DENSE: topo[0..511] = 512 SeqRefs
        return interpSeqEntry(topo[localIdx]);
    }
}

// ============================================================
// BlockTree::getStats
// ============================================================
BlockTree::Stats BlockTree::getStats() const {
    Stats s{};
    s.totalLeafs          = static_cast<int>(leafBlocks.size());
    s.mode0 = s.mode1 = s.mode2 = s.mode3 = s.mode4 = s.mode5 = s.mode6 = s.mode7 = 0;
    s.childListSize       = childList.size();
    s.outlierVoxels       = outlierVoxels;
    s.mode2FallbackBlocks = mode2FallbackBlocks;
    s.blockAwareCluster   = blockAwareCluster;
    s.budgetAwareCluster  = budgetAwareCluster;
    s.guardedMedoidCluster = guardedMedoidCluster;
    s.localClusterThrMin  = static_cast<float>(blockAwareClusterMin);
    s.localClusterThrMean = static_cast<float>(blockAwareClusterMean);
    s.localClusterThrMax  = static_cast<float>(blockAwareClusterMax);
    s.validateFallback    = validateFallback;
    s.hotspotSecondPass = hotspotSecondPass;
    s.validateFallbackBlocks = validateFallbackBlocks;
    s.validateRetryBlocks = validateRetryBlocks;
    s.validateDenseFallbackBlocks = validateDenseFallbackBlocks;
    s.hotspotSecondPassBlocks = hotspotSecondPassBlocks;
    s.hotspotSecondPassDenseBlocks = hotspotSecondPassDenseBlocks;
    s.hotspotRegionSeedBlocks = hotspotRegionSeedBlocks;
    s.hotspotRegionTouchedBlocks = hotspotRegionTouchedBlocks;
    s.hotspotRegionDenseFallbackBlocks = hotspotRegionDenseFallbackBlocks;
    s.guardedGateRejects = guardedGateRejects;
    s.medoidCenterChanges = medoidCenterChanges;
    s.qualityGatePromotions = qualityGatePromotions;

    // Count elided entries in childList
    for (uint32_t v : childList) {
        if      (v == BT_CHILD_AIR)      ++s.elidedAir;
        else if (v == BT_CHILD_INTERIOR) ++s.elidedInterior;
    }
    // Also count from rootTable
    for (uint32_t v : rootTable) {
        if      (v == BT_CHILD_AIR)      ++s.elidedAir;
        else if (v == BT_CHILD_INTERIOR) ++s.elidedInterior;
    }

    // Mode counts + KF stats
    std::vector<int> kVals;
    kVals.reserve(leafBlocks.size());
    long long totalKF = 0;

    for (const LeafBlock& leaf : leafBlocks) {
        switch (leaf.mode) {
        case LeafMode::UNIFORM:
            ++s.mode0;
            kVals.push_back(1);
            totalKF += static_cast<long long>(leaf.codebook[0].size());
            break;
        case LeafMode::CLUSTERED: {
            ++s.mode1;
            const int K = static_cast<int>(leaf.codebook.size());
            kVals.push_back(K);
            for (const auto& seq : leaf.codebook) totalKF += static_cast<long long>(seq.size());
            break;
        }
        case LeafMode::GRID4:
            ++s.mode3;
            kVals.push_back(64);
            for (const auto& seq : leaf.codebook) totalKF += static_cast<long long>(seq.size());
            break;
        case LeafMode::GRID4_RESIDUAL:
            ++s.mode4;
            kVals.push_back(64 + static_cast<int>(leaf.residualCodebook.size()));
            for (const auto& seq : leaf.codebook) totalKF += static_cast<long long>(seq.size());
            for (const auto& seq : leaf.residualCodebook) totalKF += static_cast<long long>(seq.size());
            break;
        case LeafMode::GRID4_MULTISCALE:
            ++s.mode7;
            kVals.push_back(64 + static_cast<int>(leaf.residualCodebook.size()));
            for (const auto& seq : leaf.codebook) totalKF += static_cast<long long>(seq.size());
            for (const auto& seq : leaf.residualCodebook) totalKF += static_cast<long long>(seq.size());
            break;
        case LeafMode::POLY11:
            ++s.mode5;
            kVals.push_back(11);
            for (const auto& seq : leaf.codebook) totalKF += static_cast<long long>(seq.size());
            break;
        case LeafMode::POLY11_RESIDUAL:
            ++s.mode6;
            kVals.push_back(11 + static_cast<int>(leaf.residualCodebook.size()));
            for (const auto& seq : leaf.codebook) totalKF += static_cast<long long>(seq.size());
            for (const auto& seq : leaf.residualCodebook) totalKF += static_cast<long long>(seq.size());
            break;
        default:  // DENSE
            ++s.mode2;
            kVals.push_back(BT_LEAF_VOXELS);
            for (const auto& seq : leaf.codebook) totalKF += static_cast<long long>(seq.size());
            break;
        }
    }
    s.totalKF = totalKF;

    if (!kVals.empty()) {
        std::sort(kVals.begin(), kVals.end());
        double sum = 0;
        for (int k : kVals) sum += k;
        s.kMean = static_cast<float>(sum / kVals.size());
        s.kP50  = static_cast<float>(kVals[kVals.size() / 2]);
        s.kP95  = static_cast<float>(kVals[static_cast<size_t>(kVals.size() * 0.95)]);
        s.kMax  = static_cast<float>(kVals.back());
    }

    // Memory estimate (build-time)
    s.memBytes =
        rootTable.size()     * sizeof(uint32_t)
      + internalNodes.size() * sizeof(InternalNode)
      + childList.size()     * sizeof(uint32_t);
    for (const LeafBlock& leaf : leafBlocks) {
        s.memBytes += sizeof(LeafMode);
        for (const auto& seq : leaf.codebook) s.memBytes += seq.size() * sizeof(KFPoint);
        for (const auto& seq : leaf.residualCodebook) s.memBytes += seq.size() * sizeof(KFPoint);
        s.memBytes += leaf.assign.size();
    }

    // On-disk size estimate (VBT3 format)
    // InternalNode on disk: childMask[8B]+childBase[8B]+defaultVal[4B] = 20B
    s.fileSizeEstimate =
        128                              // header
      + rootTable.size()     * 4
      + internalNodes.size() * 20        // 8+8+4
      + childList.size()     * 4
      + leafHeaders.size()   * 8         // LeafHeader: 2×uint32 per header
      + topoPool.size()      * 4         // A表: topology (SeqRefs + assign)
      + seqPool.size()       * 4;        // B表: deduplicated SoA seq data

    return s;
}

// ============================================================
// ZSD5 serialization helpers
// ============================================================
static void writeKFSeq(std::ostream& out, const KFSeq& seq) {
    uint32_t n = static_cast<uint32_t>(seq.size());
    out.write(reinterpret_cast<const char*>(&n), 4);
    if (n > 0)
        out.write(reinterpret_cast<const char*>(seq.data()), n * sizeof(KFPoint));
}

static bool readKFSeq(std::istream& in, KFSeq& seq) {
    uint32_t n = 0;
    if (!in.read(reinterpret_cast<char*>(&n), 4)) return false;
    seq.resize(n);
    if (n > 0)
        if (!in.read(reinterpret_cast<char*>(seq.data()), n * sizeof(KFPoint))) return false;
    return true;
}

static void writeLeaf(std::ostream& out, const LeafBlock& leaf) {
    uint8_t mode = static_cast<uint8_t>(leaf.mode);
    out.write(reinterpret_cast<const char*>(&mode), 1);
    uint32_t K = static_cast<uint32_t>(leaf.codebook.size());
    out.write(reinterpret_cast<const char*>(&K), 4);
    for (const auto& seq : leaf.codebook) writeKFSeq(out, seq);
    uint32_t assignSz = static_cast<uint32_t>(leaf.assign.size());
    out.write(reinterpret_cast<const char*>(&assignSz), 4);
    if (assignSz > 0)
        out.write(reinterpret_cast<const char*>(leaf.assign.data()), assignSz);
    if (leaf.mode == LeafMode::GRID4_RESIDUAL || leaf.mode == LeafMode::POLY11_RESIDUAL) {
        out.write(reinterpret_cast<const char*>(leaf.residualMask.data()), sizeof(uint64_t) * leaf.residualMask.size());
        out.write(reinterpret_cast<const char*>(&leaf.residualScale), sizeof(float));
        uint32_t residualK = static_cast<uint32_t>(leaf.residualCodebook.size());
        out.write(reinterpret_cast<const char*>(&residualK), 4);
        for (const auto& seq : leaf.residualCodebook) writeKFSeq(out, seq);
    } else if (leaf.mode == LeafMode::GRID4_MULTISCALE) {
        out.write(reinterpret_cast<const char*>(&leaf.fineGridDim), sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(leaf.residualMask.data()), sizeof(uint64_t) * leaf.residualMask.size());
        out.write(reinterpret_cast<const char*>(&leaf.residualScale), sizeof(float));
        uint32_t residualK = static_cast<uint32_t>(leaf.residualCodebook.size());
        out.write(reinterpret_cast<const char*>(&residualK), 4);
        for (const auto& seq : leaf.residualCodebook) writeKFSeq(out, seq);
    }
}

static bool readLeaf(std::istream& in, LeafBlock& leaf) {
    uint8_t mode = 0;
    if (!in.read(reinterpret_cast<char*>(&mode), 1)) return false;
    leaf.mode = static_cast<LeafMode>(mode);
    uint32_t K = 0;
    if (!in.read(reinterpret_cast<char*>(&K), 4)) return false;
    leaf.codebook.resize(K);
    for (auto& seq : leaf.codebook) if (!readKFSeq(in, seq)) return false;
    uint32_t assignSz = 0;
    if (!in.read(reinterpret_cast<char*>(&assignSz), 4)) return false;
    leaf.assign.resize(assignSz);
    if (assignSz > 0)
        if (!in.read(reinterpret_cast<char*>(leaf.assign.data()), assignSz)) return false;
    if (leaf.mode == LeafMode::GRID4_RESIDUAL || leaf.mode == LeafMode::POLY11_RESIDUAL) {
        if (!in.read(reinterpret_cast<char*>(leaf.residualMask.data()), sizeof(uint64_t) * leaf.residualMask.size())) return false;
        if (!in.read(reinterpret_cast<char*>(&leaf.residualScale), sizeof(float))) return false;
        uint32_t residualK = 0;
        if (!in.read(reinterpret_cast<char*>(&residualK), 4)) return false;
        leaf.residualCodebook.resize(residualK);
        for (auto& seq : leaf.residualCodebook) if (!readKFSeq(in, seq)) return false;
    } else if (leaf.mode == LeafMode::GRID4_MULTISCALE) {
        if (!in.read(reinterpret_cast<char*>(&leaf.fineGridDim), sizeof(uint8_t))) return false;
        if (!in.read(reinterpret_cast<char*>(leaf.residualMask.data()), sizeof(uint64_t) * leaf.residualMask.size())) return false;
        if (!in.read(reinterpret_cast<char*>(&leaf.residualScale), sizeof(float))) return false;
        uint32_t residualK = 0;
        if (!in.read(reinterpret_cast<char*>(&residualK), 4)) return false;
        leaf.residualCodebook.resize(residualK);
        for (auto& seq : leaf.residualCodebook) if (!readKFSeq(in, seq)) return false;
    }
    return true;
}

// ============================================================
// ZSD5 file format
// Header: 128 bytes ("ZSD5" magic)
// InternalNode on disk: childMask[8B]+childBase[8B]+defaultVal[4B] = 20B
// ============================================================
#pragma pack(push, 1)
struct ZSDFV5Header {
    char     magic[4];         // "ZSD5"
    uint16_t version;          // 5
    uint8_t  leafBits;         // 3
    uint8_t  internalBits;     // 2
    int32_t  dimX, dimY, dimZ;
    int32_t  numFrames;
    int32_t  rootDimX, rootDimY, rootDimZ;
    uint32_t rootTableCount;
    uint32_t internalCount;
    uint32_t childListCount;
    uint32_t leafCount;
    uint64_t rootTableOff;
    uint64_t internalOff;
    uint64_t childListOff;
    uint64_t leafDataOff;
    float    bgAir;
    float    bgInterior;
    uint8_t  _padding[36];
};
#pragma pack(pop)
static_assert(sizeof(ZSDFV5Header) == 128, "ZSDFV5Header must be 128 bytes");

bool writeBlockTree(const std::string& path, const BlockTree& tree) {
    std::ofstream out(path, std::ios::binary);
    if (!out) { std::cerr << "writeBlockTree: cannot open " << path << std::endl; return false; }

    ZSDFV5Header hdr{};
    hdr.magic[0]='Z'; hdr.magic[1]='S'; hdr.magic[2]='D'; hdr.magic[3]='5';
    hdr.version        = 5;
    hdr.leafBits       = BT_LEAF_BITS;
    hdr.internalBits   = BT_INTERNAL_BITS;
    hdr.dimX           = tree.dimX; hdr.dimY = tree.dimY; hdr.dimZ = tree.dimZ;
    hdr.numFrames      = tree.numFrames;
    hdr.rootDimX       = tree.rootDimX; hdr.rootDimY = tree.rootDimY; hdr.rootDimZ = tree.rootDimZ;
    hdr.rootTableCount = static_cast<uint32_t>(tree.rootTable.size());
    hdr.internalCount  = static_cast<uint32_t>(tree.internalNodes.size());
    hdr.childListCount = static_cast<uint32_t>(tree.childList.size());
    hdr.leafCount      = static_cast<uint32_t>(tree.leafBlocks.size());
    hdr.bgAir          = tree.bgAir;
    hdr.bgInterior     = tree.bgInterior;

    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    hdr.rootTableOff = static_cast<uint64_t>(out.tellp());
    out.write(reinterpret_cast<const char*>(tree.rootTable.data()),
              tree.rootTable.size() * sizeof(uint32_t));

    hdr.internalOff = static_cast<uint64_t>(out.tellp());
    for (const InternalNode& node : tree.internalNodes) {
        out.write(reinterpret_cast<const char*>(node.childMask), sizeof(node.childMask));  // 8B
        out.write(reinterpret_cast<const char*>(&node.childBase), sizeof(node.childBase)); // 8B
        out.write(reinterpret_cast<const char*>(&node.defaultVal), sizeof(node.defaultVal)); // 4B
    }

    hdr.childListOff = static_cast<uint64_t>(out.tellp());
    if (!tree.childList.empty())
        out.write(reinterpret_cast<const char*>(tree.childList.data()),
                  tree.childList.size() * sizeof(uint32_t));

    hdr.leafDataOff = static_cast<uint64_t>(out.tellp());
    for (const LeafBlock& leaf : tree.leafBlocks) writeLeaf(out, leaf);

    // Rewrite header with offsets
    out.seekp(0);
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    out.close();

    std::ifstream in2(path, std::ios::binary | std::ios::ate);
    const size_t fileSize = in2 ? static_cast<size_t>(in2.tellg()) : 0;
    std::cout << "writeBlockTree: " << path
              << "  size=" << (fileSize / 1024.0) << " KB"
              << "  leaves=" << tree.leafBlocks.size()
              << "  internal=" << tree.internalNodes.size() << std::endl;
    return true;
}

bool readBlockTree(const std::string& path, BlockTree& tree) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "readBlockTree: cannot open " << path << std::endl; return false; }

    ZSDFV5Header hdr{};
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (hdr.magic[0]!='Z'||hdr.magic[1]!='S'||hdr.magic[2]!='D'||hdr.magic[3]!='5'||hdr.version!=5) {
        std::cerr << "readBlockTree: invalid magic or version" << std::endl; return false;
    }

    tree.dimX      = hdr.dimX; tree.dimY = hdr.dimY; tree.dimZ = hdr.dimZ;
    tree.numFrames = hdr.numFrames;
    tree.rootDimX  = hdr.rootDimX; tree.rootDimY = hdr.rootDimY; tree.rootDimZ = hdr.rootDimZ;
    tree.bgAir     = hdr.bgAir;    tree.bgInterior = hdr.bgInterior;

    in.seekg(hdr.rootTableOff);
    tree.rootTable.resize(hdr.rootTableCount);
    in.read(reinterpret_cast<char*>(tree.rootTable.data()),
            hdr.rootTableCount * sizeof(uint32_t));

    in.seekg(hdr.internalOff);
    tree.internalNodes.resize(hdr.internalCount);
    for (InternalNode& node : tree.internalNodes) {
        in.read(reinterpret_cast<char*>(node.childMask), sizeof(node.childMask));
        in.read(reinterpret_cast<char*>(&node.childBase), sizeof(node.childBase));
        in.read(reinterpret_cast<char*>(&node.defaultVal), sizeof(node.defaultVal));
    }

    in.seekg(hdr.childListOff);
    tree.childList.resize(hdr.childListCount);
    if (hdr.childListCount > 0)
        in.read(reinterpret_cast<char*>(tree.childList.data()),
                hdr.childListCount * sizeof(uint32_t));

    in.seekg(hdr.leafDataOff);
    tree.leafBlocks.resize(hdr.leafCount);
    for (LeafBlock& leaf : tree.leafBlocks)
        if (!readLeaf(in, leaf)) { std::cerr << "readBlockTree: leaf read failed\n"; return false; }

    std::cout << "readBlockTree: " << path
              << "  grid=" << tree.dimX << "x" << tree.dimY << "x" << tree.dimZ
              << "  leaves=" << tree.leafBlocks.size() << std::endl;
    return true;
}

// ============================================================
// saveToVBT / loadFromVBT  (VBT3 format — A表 topoPool + B表 seqPool)
// ============================================================
bool saveToVBT(const std::string& filename, BlockTree& tree) {
    if (tree.leafHeaders.size() != tree.leafBlocks.size()) {
        std::cerr << "saveToVBT: flattenLeaves() must be called first\n"; return false;
    }

    std::ofstream out(filename, std::ios::binary);
    if (!out) { std::cerr << "saveToVBT: cannot open " << filename << std::endl; return false; }

    VBTHeader hdr{};
    hdr.magic[0]='V'; hdr.magic[1]='B'; hdr.magic[2]='T'; hdr.magic[3]='2';
    hdr.version        = 4;
    hdr.dimX           = tree.dimX;  hdr.dimY = tree.dimY;  hdr.dimZ = tree.dimZ;
    hdr.numFrames      = tree.numFrames;
    hdr.rootDimX       = tree.rootDimX; hdr.rootDimY = tree.rootDimY; hdr.rootDimZ = tree.rootDimZ;
    hdr.rootTableCount = static_cast<uint32_t>(tree.rootTable.size());
    hdr.internalCount  = static_cast<uint32_t>(tree.internalNodes.size());
    hdr.childListCount = static_cast<uint32_t>(tree.childList.size());
    hdr.numLeafHeaders = static_cast<uint32_t>(tree.leafHeaders.size());
    hdr.topoPoolWords  = static_cast<uint32_t>(tree.topoPool.size());
    hdr.seqPoolWords   = static_cast<uint32_t>(tree.seqPool.size());
    hdr.bgAir          = tree.bgAir;
    hdr.bgInterior     = tree.bgInterior;
    hdr.voxelSize      = 1.0f;
    hdr.seqMetaCount   = static_cast<uint32_t>(tree.seqMetaPool.size());

    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    hdr.rootTableOff = static_cast<uint64_t>(out.tellp());
    out.write(reinterpret_cast<const char*>(tree.rootTable.data()),
              tree.rootTable.size() * sizeof(uint32_t));

    hdr.internalOff = static_cast<uint64_t>(out.tellp());
    for (const InternalNode& node : tree.internalNodes) {
        out.write(reinterpret_cast<const char*>(node.childMask), sizeof(node.childMask));
        out.write(reinterpret_cast<const char*>(&node.childBase), sizeof(node.childBase));
        out.write(reinterpret_cast<const char*>(&node.defaultVal), sizeof(node.defaultVal));
    }

    hdr.childListOff = static_cast<uint64_t>(out.tellp());
    if (!tree.childList.empty())
        out.write(reinterpret_cast<const char*>(tree.childList.data()),
                  tree.childList.size() * sizeof(uint32_t));

    hdr.leafHeadersOff = static_cast<uint64_t>(out.tellp());
    out.write(reinterpret_cast<const char*>(tree.leafHeaders.data()),
              tree.leafHeaders.size() * sizeof(LeafHeader));

    hdr.topoPoolOff = static_cast<uint64_t>(out.tellp());
    if (!tree.topoPool.empty())
        out.write(reinterpret_cast<const char*>(tree.topoPool.data()),
                  tree.topoPool.size() * sizeof(uint32_t));

    if (!tree.seqMetaPool.empty()) {
        out.write(reinterpret_cast<const char*>(tree.seqMetaPool.data()),
                  tree.seqMetaPool.size() * sizeof(SeqMeta));
    }

    hdr.seqPoolOff = static_cast<uint64_t>(out.tellp());
    if (!tree.seqPool.empty())
        out.write(reinterpret_cast<const char*>(tree.seqPool.data()),
                  tree.seqPool.size() * sizeof(uint32_t));

    out.seekp(0);
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    out.close();

    std::ifstream check(filename, std::ios::binary | std::ios::ate);
    const size_t fileBytes = check ? static_cast<size_t>(check.tellg()) : 0;
    const long long rawF32 = (long long)tree.dimX * tree.dimY * tree.dimZ * tree.numFrames * 4;
    const double comprRatio = rawF32 > 0 ? 100.0 * fileBytes / rawF32 : 0.0;

    printf("saveToVBT: %s\n"
           "  fileSize=%.1f KB  rawF32=%.1f KB  ratio=%.2f%%\n"
           "  leaves=%u  topoPool=%u words  seqMeta=%u  seqPool=%u words\n",
           filename.c_str(),
           fileBytes / 1024.0, rawF32 / 1024.0, comprRatio,
           hdr.numLeafHeaders, hdr.topoPoolWords, hdr.seqMetaCount, hdr.seqPoolWords);
    return true;
}

bool loadFromVBT(const std::string& filename, BlockTree& tree) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "loadFromVBT: cannot open " << filename << std::endl; return false; }

    VBTHeader hdr{};
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (hdr.magic[0]!='V'||hdr.magic[1]!='B'||hdr.magic[2]!='T'||hdr.magic[3]!='2'||(hdr.version!=3 && hdr.version!=4)) {
        std::cerr << "loadFromVBT: unsupported magic/version (expected VBT2 v3/v4)\n"; return false;
    }

    tree.dimX      = hdr.dimX; tree.dimY = hdr.dimY; tree.dimZ = hdr.dimZ;
    tree.numFrames = hdr.numFrames;
    tree.rootDimX  = hdr.rootDimX; tree.rootDimY = hdr.rootDimY; tree.rootDimZ = hdr.rootDimZ;
    tree.bgAir     = hdr.bgAir;    tree.bgInterior = hdr.bgInterior;
    tree.leafBlocks.clear();
    tree.seqHashMap_.clear();
    tree.seqMetaPool.clear();

    in.seekg(hdr.rootTableOff);
    tree.rootTable.resize(hdr.rootTableCount);
    in.read(reinterpret_cast<char*>(tree.rootTable.data()),
            hdr.rootTableCount * sizeof(uint32_t));

    in.seekg(hdr.internalOff);
    tree.internalNodes.resize(hdr.internalCount);
    for (InternalNode& node : tree.internalNodes) {
        in.read(reinterpret_cast<char*>(node.childMask), sizeof(node.childMask));
        in.read(reinterpret_cast<char*>(&node.childBase), sizeof(node.childBase));
        in.read(reinterpret_cast<char*>(&node.defaultVal), sizeof(node.defaultVal));
    }

    in.seekg(hdr.childListOff);
    tree.childList.resize(hdr.childListCount);
    if (hdr.childListCount > 0)
        in.read(reinterpret_cast<char*>(tree.childList.data()),
                hdr.childListCount * sizeof(uint32_t));

    in.seekg(hdr.leafHeadersOff);
    tree.leafHeaders.resize(hdr.numLeafHeaders);
    if (hdr.numLeafHeaders > 0)
        in.read(reinterpret_cast<char*>(tree.leafHeaders.data()),
                hdr.numLeafHeaders * sizeof(LeafHeader));

    in.seekg(hdr.topoPoolOff);
    tree.topoPool.resize(hdr.topoPoolWords);
    if (hdr.topoPoolWords > 0)
        in.read(reinterpret_cast<char*>(tree.topoPool.data()),
                hdr.topoPoolWords * sizeof(uint32_t));

    if (hdr.version >= 4 && hdr.seqMetaCount > 0) {
        const uint64_t seqMetaOff = hdr.topoPoolOff + static_cast<uint64_t>(hdr.topoPoolWords) * sizeof(uint32_t);
        in.seekg(seqMetaOff);
        tree.seqMetaPool.resize(hdr.seqMetaCount);
        in.read(reinterpret_cast<char*>(tree.seqMetaPool.data()),
                hdr.seqMetaCount * sizeof(SeqMeta));
    }

    in.seekg(hdr.seqPoolOff);
    tree.seqPool.resize(hdr.seqPoolWords);
    if (hdr.seqPoolWords > 0)
        in.read(reinterpret_cast<char*>(tree.seqPool.data()),
                hdr.seqPoolWords * sizeof(uint32_t));

    // Reconstruct build-time leafBlocks from the flat A/B tables so that
    // sample() and validateVBT() can run on the reloaded tree as well.
    tree.leafBlocks.resize(tree.leafHeaders.size());
    for (size_t leafId = 0; leafId < tree.leafHeaders.size(); leafId++) {
        const LeafHeader& hdrLeaf = tree.leafHeaders[leafId];
        const uint32_t* topo = tree.topoPool.data() + hdrLeaf.topoOffset;
        LeafBlock& leaf = tree.leafBlocks[leafId];
        leaf.assign.clear();
        leaf.codebook.clear();

        switch (hdrLeaf.mode) {
        case 0: {
            leaf.mode = LeafMode::UNIFORM;
            if (!tree.seqMetaPool.empty()) leaf.codebook.push_back(decodeSeqMetaToKFSeq(tree.seqMetaPool, tree.seqPool, topo[0]));
            else                           leaf.codebook.push_back(decodeSeqRefToKFSeq(tree.seqPool, topo[0]));
            break;
        }
        case 1: {
            leaf.mode = LeafMode::CLUSTERED;
            leaf.assign.resize(BT_LEAF_VOXELS);
            for (int w = 0; w < 128; w++) {
                const uint32_t word = topo[w];
                for (int b = 0; b < 4; b++) {
                    leaf.assign[w * 4 + b] = static_cast<uint8_t>((word >> (b * 8)) & 0xFFu);
                }
            }
            leaf.codebook.resize(hdrLeaf.numCenters);
            for (int c = 0; c < hdrLeaf.numCenters; c++) {
                leaf.codebook[c] = !tree.seqMetaPool.empty()
                    ? decodeSeqMetaToKFSeq(tree.seqMetaPool, tree.seqPool, topo[128 + c])
                    : decodeSeqRefToKFSeq(tree.seqPool, topo[128 + c]);
            }
            break;
        }
        case 4: {
            leaf.mode = LeafMode::GRID4;
            leaf.codebook.resize(64);
            for (int i = 0; i < 64; i++) {
                leaf.codebook[i] = !tree.seqMetaPool.empty()
                    ? decodeSeqMetaToKFSeq(tree.seqMetaPool, tree.seqPool, topo[i])
                    : decodeSeqRefToKFSeq(tree.seqPool, topo[i]);
            }
            break;
        }
        case 5: {
            leaf.mode = LeafMode::GRID4_RESIDUAL;
            leaf.codebook.resize(64);
            for (int i = 0; i < 64; i++) {
                leaf.codebook[i] = !tree.seqMetaPool.empty()
                    ? decodeSeqMetaToKFSeq(tree.seqMetaPool, tree.seqPool, topo[i])
                    : decodeSeqRefToKFSeq(tree.seqPool, topo[i]);
            }
            for (int i = 0; i < 8; ++i) {
                const uint64_t lo = static_cast<uint64_t>(topo[64 + i * 2]);
                const uint64_t hi = static_cast<uint64_t>(topo[64 + i * 2 + 1]);
                leaf.residualMask[i] = lo | (hi << 32);
            }
            uint32_t scaleBits = topo[80];
            std::memcpy(&leaf.residualScale, &scaleBits, sizeof(float));
            const uint32_t residualCount = topo[81];
            leaf.residualCodebook.resize(residualCount);
            for (uint32_t i = 0; i < residualCount; ++i) {
                leaf.residualCodebook[i] = !tree.seqMetaPool.empty()
                    ? decodeSeqMetaToKFSeq(tree.seqMetaPool, tree.seqPool, topo[82 + i])
                    : decodeSeqRefToKFSeq(tree.seqPool, topo[82 + i]);
            }
            break;
        }
        case 8: {
            leaf.mode = LeafMode::GRID4_MULTISCALE;
            leaf.codebook.resize(64);
            for (int i = 0; i < 64; ++i) {
                leaf.codebook[i] = !tree.seqMetaPool.empty()
                    ? decodeSeqMetaToKFSeq(tree.seqMetaPool, tree.seqPool, topo[i])
                    : decodeSeqRefToKFSeq(tree.seqPool, topo[i]);
            }
            leaf.fineGridDim = static_cast<uint8_t>(topo[64]);
            for (int i = 0; i < 8; ++i) {
                const uint64_t lo = static_cast<uint64_t>(topo[65 + i * 2]);
                const uint64_t hi = static_cast<uint64_t>(topo[65 + i * 2 + 1]);
                leaf.residualMask[i] = lo | (hi << 32);
            }
            uint32_t scaleBits = topo[81];
            std::memcpy(&leaf.residualScale, &scaleBits, sizeof(float));
            const uint32_t codeCount = topo[82];
            leaf.residualCodebook.resize(codeCount);
            for (uint32_t i = 0; i < codeCount; ++i) {
                leaf.residualCodebook[i] = !tree.seqMetaPool.empty()
                    ? decodeSeqMetaToKFSeq(tree.seqMetaPool, tree.seqPool, topo[83 + i])
                    : decodeSeqRefToKFSeq(tree.seqPool, topo[83 + i]);
            }
            break;
        }
        case 6: {
            leaf.mode = LeafMode::POLY11;
            leaf.codebook.resize(11);
            for (int i = 0; i < 11; i++) {
                leaf.codebook[i] = !tree.seqMetaPool.empty()
                    ? decodeSeqMetaToKFSeq(tree.seqMetaPool, tree.seqPool, topo[i])
                    : decodeSeqRefToKFSeq(tree.seqPool, topo[i]);
            }
            break;
        }
        case 7: {
            leaf.mode = LeafMode::POLY11_RESIDUAL;
            leaf.codebook.resize(11);
            for (int i = 0; i < 11; i++) {
                leaf.codebook[i] = !tree.seqMetaPool.empty()
                    ? decodeSeqMetaToKFSeq(tree.seqMetaPool, tree.seqPool, topo[i])
                    : decodeSeqRefToKFSeq(tree.seqPool, topo[i]);
            }
            for (int i = 0; i < 8; ++i) {
                const uint64_t lo = static_cast<uint64_t>(topo[11 + i * 2]);
                const uint64_t hi = static_cast<uint64_t>(topo[11 + i * 2 + 1]);
                leaf.residualMask[i] = lo | (hi << 32);
            }
            uint32_t scaleBits = topo[27];
            std::memcpy(&leaf.residualScale, &scaleBits, sizeof(float));
            const uint32_t residualCount = topo[28];
            leaf.residualCodebook.resize(residualCount);
            for (uint32_t i = 0; i < residualCount; ++i) {
                leaf.residualCodebook[i] = !tree.seqMetaPool.empty()
                    ? decodeSeqMetaToKFSeq(tree.seqMetaPool, tree.seqPool, topo[29 + i])
                    : decodeSeqRefToKFSeq(tree.seqPool, topo[29 + i]);
            }
            break;
        }
        default: {
            leaf.mode = LeafMode::DENSE;
            leaf.codebook.resize(BT_LEAF_VOXELS);
            for (int i = 0; i < BT_LEAF_VOXELS; i++) {
                leaf.codebook[i] = !tree.seqMetaPool.empty()
                    ? decodeSeqMetaToKFSeq(tree.seqMetaPool, tree.seqPool, topo[i])
                    : decodeSeqRefToKFSeq(tree.seqPool, topo[i]);
            }
            break;
        }
        }
    }

    std::cout << "loadFromVBT: " << filename
              << "  grid=" << tree.dimX << "x" << tree.dimY << "x" << tree.dimZ
              << "  leafHeaders=" << tree.leafHeaders.size()
              << "  topoPool=" << tree.topoPool.size() << " words"
              << "  seqMeta=" << tree.seqMetaPool.size()
              << "  seqPool=" << tree.seqPool.size() << " words" << std::endl;
    return true;
}

// ============================================================
// validateVBT — cross-check sampleFlat vs sample
// ============================================================
bool validateVBT(const BlockTree& tree, float tol) {
    if (tree.leafHeaders.empty()) {
        printf("validateVBT: no flat layout — call flattenLeaves() first\n");
        return false;
    }
    if (tree.leafBlocks.size() != tree.leafHeaders.size()) {
        printf("validateVBT: leafBlocks/leafHeaders size mismatch\n");
        return false;
    }

    // Sample a grid of points and compare sample() vs sampleFlat()
    const int stepX = std::max(1, tree.dimX / 16);
    const int stepY = std::max(1, tree.dimY / 16);
    const int stepZ = std::max(1, tree.dimZ / 16);
    const int stepT = std::max(1, tree.numFrames / 8);

    float maxErr = 0.0f;
    int checked = 0;

    for (int z = 0; z < tree.dimZ; z += stepZ) {
        for (int y = 0; y < tree.dimY; y += stepY) {
            for (int x = 0; x < tree.dimX; x += stepX) {
                for (int tf = 0; tf < tree.numFrames; tf += stepT) {
                    float v1 = tree.sample(x, y, z, static_cast<float>(tf));
                    float v2 = tree.sampleFlat(x, y, z, static_cast<float>(tf));
                    float err = std::fabs(v1 - v2);
                    if (err > maxErr) maxErr = err;
                    ++checked;
                }
            }
        }
    }

    if (maxErr > tol) {
        printf("validateVBT: FAILED  maxErr=%.4f > tol=%.4f  (checked %d samples)\n",
               maxErr, tol, checked);
        return false;
    }
    printf("validateVBT: OK  maxErr=%.6f  checked=%d samples\n", maxErr, checked);
    return true;
}
