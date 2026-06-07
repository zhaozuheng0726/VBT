#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace vbt {

enum class FieldType { SDF, DENSITY, GENERIC };

struct SdfProfileParams {
    float iso = 128.0f;
    float wBand = 16.0f;
    float wNear = 4.0f;
    float wCritical = 2.0f;
    float epsFar = 6.0f;
    float epsNear = 2.0f;
    float epsCritical = 0.5f;
};

struct DensityProfileParams {
    float epsAbs = 1.0f;
    float epsRel = 0.05f;
    float gammaDelta = 0.2f;
    float renderCutoff = -1.0f;
    float cutoffBand = 0.0f;
};

struct FieldProfile {
    FieldType type = FieldType::GENERIC;
    SdfProfileParams sdf;
    DensityProfileParams den;

    static FieldProfile makeGeneric(DensityProfileParams p = {}) { FieldProfile fp; fp.type = FieldType::GENERIC; fp.den = p; return fp; }
    static FieldProfile makeDensity(DensityProfileParams p = {}) { FieldProfile fp; fp.type = FieldType::DENSITY; fp.den = p; return fp; }
    static FieldProfile makeSdf(SdfProfileParams p = {}) { FieldProfile fp; fp.type = FieldType::SDF; fp.sdf = p; return fp; }

    float epsilonAt(const std::vector<float>& values, int t) const
    {
        if (type == FieldType::SDF) {
            const float d = std::abs(values[t] - sdf.iso);
            if (d <= sdf.wCritical) return sdf.epsCritical;
            if (d <= sdf.wNear) return sdf.epsNear;
            if (d <= sdf.wBand) return 0.5f * (sdf.epsNear + sdf.epsFar);
            return sdf.epsFar;
        }

        float eps = den.epsAbs + den.epsRel * std::abs(values[t]);
        if (den.gammaDelta > 0.0f && t > 0) {
            const float delta = std::abs(values[t] - values[t - 1]);
            eps /= (1.0f + den.gammaDelta * delta);
        }
        return std::max(eps, 1e-4f);
    }
};

inline float semanticWeight(const FieldProfile& profile, float value)
{
    if (profile.type == FieldType::DENSITY && profile.den.renderCutoff > 0.0f) {
        const float cutoff = profile.den.renderCutoff;
        const float band = std::max(1e-6f, profile.den.cutoffBand);
        const float d = std::abs(value - cutoff);
        if (d <= band) return 4.0f;
        if (value >= cutoff) return 2.0f;
        if (value >= cutoff - 2.0f * band) return 1.5f;
        return 1.0f;
    }
    if (profile.type == FieldType::SDF) {
        const float d = std::abs(value - profile.sdf.iso);
        if (d <= profile.sdf.wCritical) return 4.0f;
        if (d <= profile.sdf.wNear) return 2.5f;
        if (d <= profile.sdf.wBand) return 1.5f;
        return 1.0f;
    }
    return 1.0f;
}

} // namespace vbt
