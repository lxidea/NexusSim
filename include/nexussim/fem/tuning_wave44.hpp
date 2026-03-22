#pragma once

#include <nexussim/core/types.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nxs::fem {

// ============================================================================
// HourglassMode — matches OpenRadioss /PROP IHQ values
// ============================================================================

enum class HourglassMode {
    IHQ1 = 1,  ///< Flanagan-Belytschko viscous (default)
    IHQ2 = 2,  ///< Flanagan-Belytschko stiffness
    IHQ3 = 3,  ///< Exact volume integration (Flanagan-Belytschko improved)
    IHQ4 = 4,  ///< Assumed strain co-rotational (Belytschko-Bindeman)
    IHQ5 = 5,  ///< Assumed deviatoric strain
    IHQ6 = 6,  ///< Type 1 + pressure smoothing
    IHQ7 = 7,  ///< Type 2 + pressure smoothing
    IHQ8 = 8,  ///< Full integration (no hourglass needed)
};

// ============================================================================
// HourglassCoefficients
// ============================================================================

struct HourglassCoefficients {
    nxs::Real qh{0.1};                ///< Hourglass coefficient (viscous component)
    nxs::Real qm{0.0};                ///< Hourglass coefficient (stiffness component)
    nxs::Real bulk_viscosity_q1{0.06}; ///< Linear bulk viscosity coefficient
    nxs::Real bulk_viscosity_q2{1.2};  ///< Quadratic bulk viscosity coefficient

    /// Return OpenRadioss defaults for the given IHQ mode.
    static HourglassCoefficients defaults(HourglassMode mode) {
        HourglassCoefficients c;
        // q1 and q2 are the same for all modes
        c.bulk_viscosity_q1 = 0.06;
        c.bulk_viscosity_q2 = 1.2;
        switch (mode) {
            case HourglassMode::IHQ1:
                c.qh = 0.1; c.qm = 0.0;
                break;
            case HourglassMode::IHQ2:
                c.qh = 0.0; c.qm = 0.1;
                break;
            case HourglassMode::IHQ3:
                c.qh = 0.1; c.qm = 0.0;
                break;
            case HourglassMode::IHQ4:
                c.qh = 0.0; c.qm = 0.05;
                break;
            case HourglassMode::IHQ5:
                c.qh = 0.0; c.qm = 0.05;
                break;
            case HourglassMode::IHQ6:
                c.qh = 0.1; c.qm = 0.0;
                break;
            case HourglassMode::IHQ7:
                c.qh = 0.0; c.qm = 0.1;
                break;
            case HourglassMode::IHQ8:
                c.qh = 0.0; c.qm = 0.0;
                break;
        }
        return c;
    }

    /// Viscous hourglass force scale: qh * rho * c * le
    nxs::Real effective_viscosity(nxs::Real rho, nxs::Real c, nxs::Real le) const {
        return qh * rho * c * le;
    }

    /// Stiffness hourglass force scale: qm * E / le
    nxs::Real effective_stiffness(nxs::Real E, nxs::Real le) const {
        return qm * E / le;
    }
};

// ============================================================================
// DrillingPenaltyCalibration
// ============================================================================

class DrillingPenaltyCalibration {
public:
    explicit DrillingPenaltyCalibration() : alpha_(1.0e-3), base_alpha_(1.0e-3) {}

    /// Estimate condition number from the diagonal of a stiffness matrix and
    /// return a calibrated drilling penalty alpha.
    ///
    /// @param stiffness_matrix  Row-major dense matrix of size ndof x ndof.
    /// @param ndof              Number of DOFs (matrix dimension).
    /// @param condition_threshold  Condition number threshold for "well conditioned".
    nxs::Real calibrate(const nxs::Real* stiffness_matrix, std::size_t ndof,
                        nxs::Real condition_threshold = 1.0e8) {
        if (ndof == 0) { alpha_ = base_alpha_; return alpha_; }

        // Extract diagonal
        nxs::Real min_diag = std::numeric_limits<nxs::Real>::max();
        nxs::Real max_diag = 0.0;
        for (std::size_t i = 0; i < ndof; ++i) {
            nxs::Real d = std::abs(stiffness_matrix[i * ndof + i]);
            if (d < min_diag) min_diag = d;
            if (d > max_diag) max_diag = d;
        }
        return calibrate_from_diagonal_impl(min_diag, max_diag, condition_threshold);
    }

    /// Same logic but takes the diagonal array directly (more efficient).
    nxs::Real calibrate_from_diagonal(const nxs::Real* diagonal, std::size_t n,
                                      nxs::Real condition_threshold = 1.0e8) {
        if (n == 0) { alpha_ = base_alpha_; return alpha_; }

        nxs::Real min_diag = std::numeric_limits<nxs::Real>::max();
        nxs::Real max_diag = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            nxs::Real d = std::abs(diagonal[i]);
            if (d < min_diag) min_diag = d;
            if (d > max_diag) max_diag = d;
        }
        return calibrate_from_diagonal_impl(min_diag, max_diag, condition_threshold);
    }

    /// Current alpha value.
    nxs::Real alpha() const { return alpha_; }

    /// Override the base value used in calibration.
    void set_base_alpha(nxs::Real a) { base_alpha_ = a; alpha_ = a; }

    /// Heuristic: alpha decreases for high aspect-ratio elements.
    /// aspect_ratio >= 1; alpha = base_alpha / max(1, aspect_ratio^2 / 100).
    static nxs::Real recommended_alpha(nxs::Real aspect_ratio) {
        constexpr nxs::Real base = 1.0e-3;
        if (aspect_ratio <= 1.0) return base;
        nxs::Real ar2 = aspect_ratio * aspect_ratio;
        nxs::Real denom = std::max(1.0, ar2 / 100.0);
        return base / denom;
    }

private:
    nxs::Real alpha_;
    nxs::Real base_alpha_;

    nxs::Real calibrate_from_diagonal_impl(nxs::Real min_diag, nxs::Real max_diag,
                                            nxs::Real condition_threshold) {
        if (max_diag <= 0.0) { alpha_ = base_alpha_; return alpha_; }

        nxs::Real cond = max_diag / std::max(min_diag, std::numeric_limits<nxs::Real>::min());

        if (cond < condition_threshold) {
            // Well-conditioned: use standard alpha
            alpha_ = base_alpha_;
        } else {
            // Poorly conditioned: scale alpha down to avoid ill-conditioning
            alpha_ = (min_diag / max_diag) * base_alpha_;
        }
        return alpha_;
    }
};

// ============================================================================
// FrictionType
// ============================================================================

enum class FrictionType {
    Coulomb,
    StaticDynamic,
    Exponential,
    TemperatureDependent,
};

// ============================================================================
// FrictionModelVariants
// ============================================================================

class FrictionModelVariants {
public:
    FrictionModelVariants(FrictionType type, nxs::Real mu_static)
        : type_(type), mu_static_(mu_static), mu_dynamic_(mu_static),
          decay_exponent_(1.0) {}

    /// Configure dynamic friction and decay (used by StaticDynamic and Exponential).
    void set_dynamic(nxs::Real mu_dynamic, nxs::Real decay_exponent = 1.0) {
        mu_dynamic_    = mu_dynamic;
        decay_exponent_ = decay_exponent;
    }

    /// Set lookup table for TemperatureDependent friction.
    /// @param temp_mu_pairs  Vector of (temperature, mu) pairs, sorted by temperature.
    void set_temperature_curve(const std::vector<std::pair<nxs::Real, nxs::Real>>& temp_mu_pairs) {
        temp_mu_pairs_ = temp_mu_pairs;
    }

    /// Compute friction coefficient.
    ///
    /// @param slip_velocity  Relative slip velocity magnitude.
    /// @param pressure       Contact pressure (not used in basic models but kept for interface parity).
    /// @param temperature    Contact temperature (only used for TemperatureDependent).
    nxs::Real compute_friction(nxs::Real slip_velocity, nxs::Real /*pressure*/,
                               nxs::Real temperature = 0.0) const {
        switch (type_) {
            case FrictionType::Coulomb:
                return mu_static_;

            case FrictionType::StaticDynamic:
            case FrictionType::Exponential: {
                nxs::Real v = std::abs(slip_velocity);
                return mu_dynamic_ + (mu_static_ - mu_dynamic_) *
                       std::exp(-decay_exponent_ * v);
            }

            case FrictionType::TemperatureDependent: {
                nxs::Real mu = interpolate_temperature(temperature);
                return mu;
            }
        }
        return mu_static_; // unreachable, but silences warning
    }

    nxs::Real mu_static() const { return mu_static_; }
    nxs::Real mu_dynamic() const { return mu_dynamic_; }
    FrictionType type() const { return type_; }

private:
    FrictionType type_;
    nxs::Real mu_static_;
    nxs::Real mu_dynamic_;
    nxs::Real decay_exponent_;
    std::vector<std::pair<nxs::Real, nxs::Real>> temp_mu_pairs_;

    /// Linear interpolation from temperature curve.
    nxs::Real interpolate_temperature(nxs::Real T) const {
        if (temp_mu_pairs_.empty()) return mu_static_;
        if (T <= temp_mu_pairs_.front().first) return temp_mu_pairs_.front().second;
        if (T >= temp_mu_pairs_.back().first) return temp_mu_pairs_.back().second;

        // Binary search for the interval
        std::size_t lo = 0, hi = temp_mu_pairs_.size() - 1;
        while (hi - lo > 1) {
            std::size_t mid = (lo + hi) / 2;
            if (temp_mu_pairs_[mid].first <= T) lo = mid;
            else hi = mid;
        }
        nxs::Real t0 = temp_mu_pairs_[lo].first;
        nxs::Real t1 = temp_mu_pairs_[hi].first;
        nxs::Real m0 = temp_mu_pairs_[lo].second;
        nxs::Real m1 = temp_mu_pairs_[hi].second;
        nxs::Real frac = (T - t0) / (t1 - t0);
        return m0 + frac * (m1 - m0);
    }
};

// ============================================================================
// TuningParameterSet — combines all tuning parameters
// ============================================================================

class TuningParameterSet {
public:
    HourglassCoefficients hourglass;
    DrillingPenaltyCalibration drilling;
    FrictionModelVariants friction;

    TuningParameterSet()
        : hourglass(HourglassCoefficients::defaults(HourglassMode::IHQ1)),
          drilling(),
          friction(FrictionType::Coulomb, 0.0) {}

    TuningParameterSet(HourglassCoefficients hg,
                       DrillingPenaltyCalibration dp,
                       FrictionModelVariants fr)
        : hourglass(hg), drilling(dp), friction(std::move(fr)) {}

    /// OpenRadioss default tuning: IHQ1 + standard alpha + zero Coulomb friction.
    static TuningParameterSet openradioss_defaults() {
        HourglassCoefficients hg = HourglassCoefficients::defaults(HourglassMode::IHQ1);
        DrillingPenaltyCalibration dp;
        FrictionModelVariants fr(FrictionType::Coulomb, 0.0);
        return TuningParameterSet(hg, dp, fr);
    }

    /// Crash analysis defaults: IHQ4 + standard alpha + StaticDynamic(0.3, 0.2).
    static TuningParameterSet crash_defaults() {
        HourglassCoefficients hg = HourglassCoefficients::defaults(HourglassMode::IHQ4);
        DrillingPenaltyCalibration dp;
        FrictionModelVariants fr(FrictionType::StaticDynamic, 0.3);
        fr.set_dynamic(0.2, 1.0);
        return TuningParameterSet(hg, dp, fr);
    }

    /// Sheet metal stamping defaults: IHQ2 + small alpha + Coulomb(0.15).
    static TuningParameterSet stamping_defaults() {
        HourglassCoefficients hg = HourglassCoefficients::defaults(HourglassMode::IHQ2);
        DrillingPenaltyCalibration dp;
        dp.set_base_alpha(1.0e-4);
        FrictionModelVariants fr(FrictionType::Coulomb, 0.15);
        return TuningParameterSet(hg, dp, fr);
    }
};

} // namespace nxs::fem
