#ifndef FIELD_PROFILE_H
#define FIELD_PROFILE_H

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

// ============================================================
// FieldProfile — Profile-driven scalar field abstraction
//
// Supports three field types:
//   SDF     — signed-distance field (uses ABS error + band oracle)
//   DENSITY — density/scalar field  (uses ABS+REL normalized error + delta oracle)
//   GENERIC — generic scalar field  (uses ABS+REL normalized error, no bg elision)
//
// Usage: create a FieldProfile, choose type, fill params.
//   SDF:     profile = FieldProfile::makeSDF();
//   DENSITY: profile = FieldProfile::makeDensity();
//
// Core interface:
//   metric_error(truth, pred)  — error measure for clustering/DP decisions
//   epsilon_at(values, t)      — DP tolerance budget at time t
//   passes_cluster(truth, pred)— whether voxel-vs-rep error is within budget
// ============================================================

enum class FieldType { SDF, DENSITY, GENERIC };

// ---- SDF profile parameters ----
struct SdfProfileParams {
    // Mathematical SDF semantics use zero level set = 0.
    // This project currently stores many SDF datasets in biased uint8 form,
    // so the semantic zero level set is encoded at storage value 128.
    float iso            = 128.0f;   // iso-surface value
    float w_band         = 16.0f;    // far-band half-width around surface
    float w_near         =  4.0f;    // near-band half-width
    float w_critical     =  2.0f;    // surface-critical half-width
    float eps_far        =  6.0f;    // DP tolerance outside far band
    float eps_near       =  2.0f;    // DP tolerance inside near band
    float eps_critical   =  0.5f;    // DP tolerance for surface-critical voxels
};

// ---- Density profile parameters ----
struct DensityProfileParams {
    float eps_abs    = 1.0f;   // absolute tolerance floor (uint8 units)
    float eps_rel    = 0.05f;  // relative tolerance (5% of truth magnitude)
    float base_eps   = 6.0f;   // DP budget baseline (used in epsilon_at)
    float gamma_delta= 0.2f;   // Δt oracle sensitivity (higher = more KF at edges)
    float render_cutoff = -1.0f; // optional visibility cutoff for smoke rendering
    float cutoff_band   = 0.0f;  // protect values near [cutoff-band, cutoff+band]
    bool cutoff_temporal_protect = false; // preserve KF near cutoff crossings
    bool cutoff_cluster_protect  = false; // reject clustering across visibility states
    float bg_zero_ratio = 0.30f; // values below bg_zero_ratio * cutoff are treated as true background
    float bg_const_ratio = 0.60f; // values below bg_const_ratio * cutoff may be merged to near-background uniform leaves
};

// ---- Unified profile ----
struct FieldProfile {
    FieldType type = FieldType::SDF;
    SdfProfileParams    sdf;
    DensityProfileParams den;

    // Factory helpers
    static FieldProfile makeSDF(SdfProfileParams p = {}) {
        FieldProfile fp; fp.type = FieldType::SDF; fp.sdf = p; return fp;
    }
    static FieldProfile makeDensity(DensityProfileParams p = {}) {
        FieldProfile fp; fp.type = FieldType::DENSITY; fp.den = p; return fp;
    }
    static FieldProfile makeGeneric(DensityProfileParams p = {}) {
        FieldProfile fp; fp.type = FieldType::GENERIC; fp.den = p; return fp;
    }

    // ---- metric_error(truth, pred) ----
    //
    // SDF:     abs(truth - pred)
    //   passes when metric_error <= cluster_threshold
    //
    // DENSITY: abs(truth - pred) / (eps_abs + eps_rel * abs(truth))
    //   normalized error; passes when metric_error <= 1.0
    //   (the eps_abs/eps_rel are baked into the denominator)
    //
    inline float metric_error(float truth, float pred) const {
        const float diff = std::abs(truth - pred);
        if (type == FieldType::SDF) {
            return diff;
        } else {
            const float denom = den.eps_abs + den.eps_rel * std::abs(truth);
            return diff / std::max(denom, 1e-6f);
        }
    }

    // ---- cluster_threshold() ----
    // The threshold that metric_error must be <= for cluster membership.
    // SDF:     absolute threshold in field units (e.g. 8.0)
    // DENSITY / GENERIC:
    //          normalized threshold in metric space.
    //          1.0 means "pass when normalized error <= 1".
    inline float cluster_threshold(float abs_threshold) const {
        return abs_threshold;
    }

    // ---- epsilon_at(values, t) — oracle for DP tolerance at frame t ----
    //
    // SDF:     band-oracle — near surface → eps_near, elsewhere → eps_far
    //   returns eps such that DP normalized error = abs_error / eps,
    //   so DP already expects the raw diff to be <= eps.
    //
    // DENSITY: delta-oracle — high temporal gradient → smaller eps (more KF)
    //   eps = base_eps / (1 + gamma_delta * |values[t] - values[t-1]|)
    //   This epsilon is raw (not normalized), and DP normalizes by it:
    //   err_norm = abs(truth - pred) / eps
    //
    // Note: epsilon_at is used by the DP loop.  For SDF, DP uses
    //   err_norm = abs_err / eps_near_or_far, passes when > 1.
    // For DENSITY we want DP to use the *same* normalized error as clustering:
    //   err_norm = abs_err / (eps_abs + eps_rel * abs(truth))
    // So we return eps = eps_abs + eps_rel * abs(values[t]) for DENSITY.
    // The delta-oracle further scales this by 1/(1+gamma_delta*|Δv|).
    //
    inline float epsilon_at(const std::vector<float>& values, int t) const {
        if (type == FieldType::SDF) {
            const float d = std::abs(values[t] - sdf.iso);
            if (d <= sdf.w_critical) return sdf.eps_critical;
            if (d <= sdf.w_near) return sdf.eps_near;
            if (d <= sdf.w_band) return 0.5f * (sdf.eps_near + sdf.eps_far);
            return sdf.eps_far;
        } else {
            // Base eps from the metric denominator at values[t]
            float eps = den.eps_abs + den.eps_rel * std::abs(values[t]);
            // Optionally scale by inverse delta-oracle
            if (den.gamma_delta > 0.0f && t > 0) {
                float delta = std::abs(values[t] - values[t - 1]);
                eps = eps / (1.0f + den.gamma_delta * delta);
            }
            return std::max(eps, 1e-4f);
        }
    }

    // ---- passes_cluster(truth, pred, abs_threshold) ----
    // Convenience: returns true if voxel-vs-rep error is within budget.
    inline bool passes_cluster(float truth, float pred, float abs_threshold) const {
        return metric_error(truth, pred) <= cluster_threshold(abs_threshold);
    }

    inline bool default_background_elision() const {
        return type != FieldType::GENERIC;
    }
};

#endif // FIELD_PROFILE_H
