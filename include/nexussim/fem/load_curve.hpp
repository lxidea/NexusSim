#pragma once

/**
 * @file load_curve.hpp
 * @brief Time-dependent load curves for crash simulation
 *
 * GPU-compatible piecewise-linear interpolation with
 * configurable extrapolation modes.
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <cmath>
#include <vector>
#include <map>
#include <iostream>

namespace nxs {
namespace fem {

// ============================================================================
// Load Curve Extrapolation
// ============================================================================

enum class ExtrapolationMode {
    Constant,   ///< Hold last value
    Linear,     ///< Linear extrapolation from last segment
    Zero        ///< Return zero outside range
};

// ============================================================================
// Load Curve (GPU-compatible, fixed-size)
// ============================================================================

struct LoadCurve {
    static constexpr int MAX_POINTS = 128;

    int id;
    int num_points;
    Real time[MAX_POINTS];
    Real value[MAX_POINTS];
    ExtrapolationMode extrap_low;
    ExtrapolationMode extrap_high;

    LoadCurve()
        : id(0), num_points(0)
        , extrap_low(ExtrapolationMode::Constant)
        , extrap_high(ExtrapolationMode::Constant) {
        for (int i = 0; i < MAX_POINTS; ++i) { time[i] = 0.0; value[i] = 0.0; }
    }

    /// Add a point to the curve
    void add_point(Real t, Real v) {
        if (num_points < MAX_POINTS) {
            time[num_points] = t;
            value[num_points] = v;
            num_points++;
        }
    }

    /// Evaluate curve at time t (piecewise linear interpolation)
    KOKKOS_INLINE_FUNCTION
    Real evaluate(Real t) const {
        if (num_points <= 0) return 0.0;
        if (num_points == 1) return value[0];

        // Below range
        if (t <= time[0]) {
            switch (extrap_low) {
                case ExtrapolationMode::Zero: return 0.0;
                case ExtrapolationMode::Linear: {
                    Real slope = (value[1] - value[0]) / (time[1] - time[0] + 1.0e-30);
                    return value[0] + slope * (t - time[0]);
                }
                default: return value[0]; // Constant
            }
        }

        // Above range
        if (t >= time[num_points - 1]) {
            switch (extrap_high) {
                case ExtrapolationMode::Zero: return 0.0;
                case ExtrapolationMode::Linear: {
                    int n = num_points - 1;
                    Real slope = (value[n] - value[n-1]) / (time[n] - time[n-1] + 1.0e-30);
                    return value[n] + slope * (t - time[n]);
                }
                default: return value[num_points - 1]; // Constant
            }
        }

        // Interior: piecewise linear
        for (int i = 0; i < num_points - 1; ++i) {
            if (t >= time[i] && t <= time[i + 1]) {
                Real dt_seg = time[i + 1] - time[i];
                if (dt_seg < 1.0e-30) return value[i];
                Real alpha = (t - time[i]) / dt_seg;
                return value[i] + alpha * (value[i + 1] - value[i]);
            }
        }

        return value[num_points - 1];
    }
};

// ============================================================================
// Load Curve Manager
// ============================================================================

class LoadCurveManager {
public:
    LoadCurveManager() = default;

    LoadCurve& add_curve(int id) {
        curves_[id] = LoadCurve();
        curves_[id].id = id;
        return curves_[id];
    }

    LoadCurve* get_curve(int id) {
        auto it = curves_.find(id);
        return (it != curves_.end()) ? &it->second : nullptr;
    }

    const LoadCurve* get_curve(int id) const {
        auto it = curves_.find(id);
        return (it != curves_.end()) ? &it->second : nullptr;
    }

    Real evaluate(int curve_id, Real time) const {
        const auto* c = get_curve(curve_id);
        return c ? c->evaluate(time) : 1.0;  // Default: constant 1.0
    }

    std::size_t num_curves() const { return curves_.size(); }

    void print_summary() const {
        std::cout << "Load Curves: " << curves_.size() << "\n";
        for (const auto& [id, c] : curves_) {
            std::cout << "  Curve " << id << ": " << c.num_points << " points, t=["
                      << c.time[0] << ", " << c.time[c.num_points-1] << "]\n";
        }
    }

private:
    std::map<int, LoadCurve> curves_;
};

} // namespace fem
} // namespace nxs
