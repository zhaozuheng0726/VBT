#include "dct_codec.h"

#include <algorithm>
#include <cmath>

namespace vbt {

namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<double> buildBasisRow(int n, int keep, int sampleIndex)
{
    std::vector<double> row(static_cast<size_t>(keep), 0.0);
    const double invN = 1.0 / static_cast<double>(n);
    for (int k = 0; k < keep; ++k) {
        const double alpha = (k == 0) ? std::sqrt(invN) : std::sqrt(2.0 * invN);
        row[static_cast<size_t>(k)] = alpha *
            std::cos((kPi / static_cast<double>(n)) * (static_cast<double>(sampleIndex) + 0.5) * static_cast<double>(k));
    }
    return row;
}

bool solveSymmetricSystem(std::vector<double>& A, std::vector<double>& b, int n)
{
    for (int i = 0; i < n; ++i) {
        int pivot = i;
        double best = std::abs(A[static_cast<size_t>(i * n + i)]);
        for (int r = i + 1; r < n; ++r) {
            const double cand = std::abs(A[static_cast<size_t>(r * n + i)]);
            if (cand > best) {
                best = cand;
                pivot = r;
            }
        }
        if (best < 1e-12) return false;
        if (pivot != i) {
            for (int c = i; c < n; ++c) {
                std::swap(A[static_cast<size_t>(i * n + c)], A[static_cast<size_t>(pivot * n + c)]);
            }
            std::swap(b[static_cast<size_t>(i)], b[static_cast<size_t>(pivot)]);
        }

        const double diag = A[static_cast<size_t>(i * n + i)];
        for (int c = i; c < n; ++c) A[static_cast<size_t>(i * n + c)] /= diag;
        b[static_cast<size_t>(i)] /= diag;

        for (int r = 0; r < n; ++r) {
            if (r == i) continue;
            const double factor = A[static_cast<size_t>(r * n + i)];
            if (std::abs(factor) < 1e-18) continue;
            for (int c = i; c < n; ++c) {
                A[static_cast<size_t>(r * n + c)] -= factor * A[static_cast<size_t>(i * n + c)];
            }
            b[static_cast<size_t>(r)] -= factor * b[static_cast<size_t>(i)];
        }
    }
    return true;
}

}

std::vector<float> dctEncodeKeep(const std::vector<float>& series, int keepCount)
{
    const int n = static_cast<int>(series.size());
    const int keep = std::max(1, std::min(keepCount, n));
    std::vector<float> coeffs(keep, 0.0f);
    const double invN = 1.0 / static_cast<double>(n);
    for (int k = 0; k < keep; ++k) {
        const double alpha = (k == 0) ? std::sqrt(invN) : std::sqrt(2.0 * invN);
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += static_cast<double>(series[i]) *
                   std::cos((kPi / static_cast<double>(n)) * (static_cast<double>(i) + 0.5) * static_cast<double>(k));
        }
        coeffs[k] = static_cast<float>(alpha * sum);
    }
    return coeffs;
}

std::vector<float> dctEncodeKeepWeighted(const std::vector<float>& series,
                                         const std::vector<float>& weights,
                                         int keepCount)
{
    const int n = static_cast<int>(series.size());
    const int keep = std::max(1, std::min(keepCount, n));
    if (weights.empty()) return dctEncodeKeep(series, keep);

    std::vector<double> normal(static_cast<size_t>(keep * keep), 0.0);
    std::vector<double> rhs(static_cast<size_t>(keep), 0.0);
    for (int i = 0; i < n; ++i) {
        const double w = std::max(1e-6, static_cast<double>(weights[static_cast<size_t>(i)]));
        const auto row = buildBasisRow(n, keep, i);
        for (int r = 0; r < keep; ++r) {
            rhs[static_cast<size_t>(r)] += w * row[static_cast<size_t>(r)] * static_cast<double>(series[static_cast<size_t>(i)]);
            for (int c = 0; c < keep; ++c) {
                normal[static_cast<size_t>(r * keep + c)] += w * row[static_cast<size_t>(r)] * row[static_cast<size_t>(c)];
            }
        }
    }

    if (!solveSymmetricSystem(normal, rhs, keep)) {
        return dctEncodeKeep(series, keep);
    }

    std::vector<float> coeffs(static_cast<size_t>(keep), 0.0f);
    for (int i = 0; i < keep; ++i) coeffs[static_cast<size_t>(i)] = static_cast<float>(rhs[static_cast<size_t>(i)]);
    return coeffs;
}

float dctDecodeAt(const std::vector<float>& coeffs, int totalLength, int index)
{
    const int keep = static_cast<int>(coeffs.size());
    const double invN = 1.0 / static_cast<double>(totalLength);
    double value = 0.0;
    for (int k = 0; k < keep; ++k) {
        const double alpha = (k == 0) ? std::sqrt(invN) : std::sqrt(2.0 * invN);
        value += static_cast<double>(coeffs[k]) * alpha *
                 std::cos((kPi / static_cast<double>(totalLength)) * (static_cast<double>(index) + 0.5) * static_cast<double>(k));
    }
    return static_cast<float>(value);
}

} // namespace vbt
