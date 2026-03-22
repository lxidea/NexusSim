#pragma once

/**
 * @file failure_wave33.hpp
 * @brief 14 additional failure/damage models for ductile, composite, fatigue, and specialty
 *
 * Models implemented:
 *   1.  EMCFailure              - Extended Mohr-Coulomb ductile fracture
 *   2.  SahraeiFailure          - Battery cell short-circuit criterion
 *   3.  SyazwanFailure          - Strain-rate dependent ductile failure
 *   4.  VisualFailure           - Visual damage indicator (no deletion)
 *   5.  NXTFailure              - Next-gen composite (fiber/matrix damage)
 *   6.  Gene1Failure            - Polynomial stress-space failure
 *   7.  IniEvoFailure           - Initial damage + linear evolution
 *   8.  AlterFailure            - Alternating-load fatigue (Miner's rule)
 *   9.  BiquadFailure           - Biquadratic failure surface
 *   10. OrthBiquadFailure       - Orthotropic biquadratic failure
 *   11. OrthEnergFailure        - Orthotropic energy-based failure
 *   12. OrthStrainFailure       - Orthotropic max-strain failure
 *   13. TensStrainFailure       - Tensile strain criterion
 *   14. SnConnectFailure        - Spot-weld connection failure
 *
 * All models inherit from FailureModel, implement compute_damage(), and
 * provide KOKKOS_INLINE_FUNCTION helpers for GPU portability.
 */

#include <nexussim/physics/failure/failure_model.hpp>
#include <nexussim/physics/material.hpp>
#include <cmath>
#include <memory>
#include <algorithm>

namespace nxs {
namespace physics {
namespace failure {

// ============================================================================
// Wave 33 failure model type enumeration
// ============================================================================

enum class FailureModelTypeWave33 {
    EMC,
    Sahraei,
    Syazwan,
    Visual,
    NXT,
    Gene1,
    IniEvo,
    Alter,
    Biquad,
    OrthBiquad,
    OrthEnerg,
    OrthStrain,
    TensStrain,
    SnConnect
};

// ============================================================================
// Parameter Structures
// ============================================================================

/// Extended Mohr-Coulomb parameters
struct EMCFailureParams {
    Real c0;          ///< Cohesion (Pa)
    Real phi;         ///< Friction angle (radians)
    Real A_mc;        ///< MC calibration constant A
    Real B_mc;        ///< MC calibration constant B
    Real n_mc;        ///< MC exponent

    EMCFailureParams()
        : c0(100.0e6), phi(0.5236), A_mc(0.8), B_mc(0.6), n_mc(0.1) {}
};

/// Battery cell short-circuit parameters
struct SahraeiFailureParams {
    Real eps_crit;        ///< Critical effective strain
    Real pressure_crit;   ///< Critical pressure (Pa)
    Real combined_weight; ///< Weight alpha for combined criterion [0,1]

    SahraeiFailureParams()
        : eps_crit(0.15), pressure_crit(50.0e6), combined_weight(0.5) {}
};

/// Strain-rate dependent ductile failure parameters
struct SyazwanFailureParams {
    Real eps_f0;      ///< Reference failure strain (quasi-static)
    Real C_rate;      ///< Rate sensitivity coefficient
    Real n_rate;      ///< Rate sensitivity exponent
    Real eps_dot_ref; ///< Reference strain rate (1/s)

    SyazwanFailureParams()
        : eps_f0(0.5), C_rate(0.04), n_rate(1.0), eps_dot_ref(1.0) {}
};

/// Visual damage indicator parameters
struct VisualFailureParams {
    Real d_threshold_low;  ///< Low damage threshold
    Real d_threshold_med;  ///< Medium damage threshold
    Real d_threshold_high; ///< High damage threshold
    Real eps_ref;          ///< Reference strain for normalization

    VisualFailureParams()
        : d_threshold_low(0.25), d_threshold_med(0.5), d_threshold_high(0.75),
          eps_ref(0.1) {}
};

/// Next-gen composite failure parameters
struct NXTFailureParams {
    Real Xt;           ///< Fiber tensile strength (Pa)
    Real Xc;           ///< Fiber compressive strength (Pa)
    Real Yt;           ///< Matrix tensile strength (Pa)
    Real Yc;           ///< Matrix compressive strength (Pa)
    Real S12;          ///< In-plane shear strength (Pa)
    Real m_fiber;      ///< Fiber damage evolution exponent
    Real m_matrix;     ///< Matrix damage evolution exponent

    NXTFailureParams()
        : Xt(1.5e9), Xc(1.2e9), Yt(50.0e6), Yc(200.0e6),
          S12(75.0e6), m_fiber(2.0), m_matrix(1.5) {}
};

/// Polynomial stress-space failure parameters
struct Gene1FailureParams {
    Real a[6];         ///< 6 polynomial coefficients

    Gene1FailureParams() {
        // Default: Von Mises-like
        a[0] = 1.0e-18;  // s11^2
        a[1] = 1.0e-18;  // s22^2
        a[2] = 1.0e-18;  // s12^2 (cross)
        a[3] = 3.0e-18;  // s12^2 (shear)
        a[4] = 0.0;       // s11 linear
        a[5] = 0.0;       // s22 linear
    }
};

/// Initial damage + evolution parameters
struct IniEvoFailureParams {
    Real d0;           ///< Initial damage [0,1)
    Real eps_i;        ///< Initiation strain
    Real eps_f;        ///< Final failure strain

    IniEvoFailureParams()
        : d0(0.05), eps_i(0.01), eps_f(0.2) {}
};

/// Alternating-load fatigue parameters
struct AlterFailureParams {
    Real S_f;              ///< Fatigue limit (endurance, Pa)
    Real b_basquin;        ///< Basquin exponent (negative, e.g. -0.12)
    Real c_coffin;         ///< Coffin exponent (negative, e.g. -0.6)
    Real sigma_f_prime;    ///< Fatigue strength coefficient (Pa)
    Real eps_f_prime;      ///< Fatigue ductility coefficient

    AlterFailureParams()
        : S_f(200.0e6), b_basquin(-0.12), c_coffin(-0.6),
          sigma_f_prime(900.0e6), eps_f_prime(0.5) {}
};

/// Biquadratic failure surface parameters
struct BiquadFailureParams {
    Real F1;   ///< Coefficient for s1^2
    Real F2;   ///< Coefficient for s2^2
    Real F3;   ///< Coefficient for s1*s2 (interaction)
    Real F4;   ///< Coefficient for s12^2
    Real F5;   ///< Linear coefficient for s1
    Real F6;   ///< Linear coefficient for s2

    BiquadFailureParams()
        : F1(1.0e-18), F2(1.0e-18), F3(-0.5e-18), F4(3.0e-18),
          F5(0.0), F6(0.0) {}
};

/// Orthotropic biquadratic failure parameters
struct OrthBiquadFailureParams {
    Real Xt;   ///< Tensile strength, 1-direction (Pa)
    Real Xc;   ///< Compressive strength, 1-direction (Pa)
    Real Yt;   ///< Tensile strength, 2-direction (Pa)
    Real Yc;   ///< Compressive strength, 2-direction (Pa)
    Real S12;  ///< In-plane shear strength (Pa)
    Real S23;  ///< Transverse shear strength (Pa)

    OrthBiquadFailureParams()
        : Xt(1.5e9), Xc(1.2e9), Yt(50.0e6), Yc(200.0e6),
          S12(75.0e6), S23(50.0e6) {}
};

/// Orthotropic energy-based failure parameters
struct OrthEnergFailureParams {
    Real G1c;   ///< Critical energy, 1-direction (J/m^3)
    Real G2c;   ///< Critical energy, 2-direction (J/m^3)
    Real G12c;  ///< Critical energy, shear 1-2 (J/m^3)

    OrthEnergFailureParams()
        : G1c(1.0e6), G2c(0.5e6), G12c(0.8e6) {}
};

/// Orthotropic max-strain failure parameters
struct OrthStrainFailureParams {
    Real eps_1t;        ///< Max tensile strain, 1-direction
    Real eps_1c;        ///< Max compressive strain, 1-direction (positive value)
    Real eps_2t;        ///< Max tensile strain, 2-direction
    Real eps_2c;        ///< Max compressive strain, 2-direction (positive value)
    Real gamma_12_max;  ///< Max shear strain 1-2

    OrthStrainFailureParams()
        : eps_1t(0.02), eps_1c(0.015), eps_2t(0.005), eps_2c(0.01),
          gamma_12_max(0.025) {}
};

/// Tensile strain criterion parameters
struct TensStrainFailureParams {
    Real eps_crit;                ///< Critical tensile strain
    Real damage_evolution_exponent; ///< Exponent n for damage evolution

    TensStrainFailureParams()
        : eps_crit(0.01), damage_evolution_exponent(2.0) {}
};

/// Spot-weld connection failure parameters
struct SnConnectFailureParams {
    Real F_normal_max;  ///< Maximum normal force (N)
    Real F_shear_max;   ///< Maximum shear force (N)
    Real n_norm;        ///< Normal force exponent
    Real n_shear;       ///< Shear force exponent
    Real area;          ///< Connection area (m^2)

    SnConnectFailureParams()
        : F_normal_max(5000.0), F_shear_max(8000.0),
          n_norm(2.0), n_shear(2.0), area(2.0e-5) {}
};


// ============================================================================
// 1. EMCFailure - Extended Mohr-Coulomb
// ============================================================================

/**
 * @brief Extended Mohr-Coulomb ductile fracture model
 *
 * Modified MC criterion in (eta, theta_bar, eps_p) space.
 * Fracture strain: eps_f = f(eta, theta_bar) using EMC formulation.
 * Damage: D += d_eps_p / eps_f (incremental).
 *
 * History:
 *   [0] = accumulated damage D
 *   [1] = current fracture strain eps_f
 *   [2] = stress triaxiality eta
 *   [3] = normalized Lode angle theta_bar
 *
 * Reference: Bai & Wierzbicki (2010), Int. J. Fracture
 */
class EMCFailure : public FailureModel {
public:
    EMCFailure(const FailureModelParameters& base_params,
               const EMCFailureParams& emc_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , emc_(emc_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // Compute stress invariants
        Real s1 = mstate.stress[0], s2 = mstate.stress[1], s3 = mstate.stress[2];
        Real t12 = mstate.stress[3], t23 = mstate.stress[4], t13 = mstate.stress[5];

        Real p = -(s1 + s2 + s3) / 3.0; // hydrostatic pressure (sign convention)
        Real sigma_m = -p; // mean stress

        // Deviatoric stress components
        Real d1 = s1 - sigma_m, d2 = s2 - sigma_m, d3 = s3 - sigma_m;

        // Von Mises equivalent stress
        Real J2 = 0.5 * (d1*d1 + d2*d2 + d3*d3) + t12*t12 + t23*t23 + t13*t13;
        Real sigma_eq = Kokkos::sqrt(3.0 * J2);
        if (sigma_eq < 1.0e-30) sigma_eq = 1.0e-30;

        // Stress triaxiality
        Real eta = sigma_m / sigma_eq;

        // Third invariant J3
        Real J3 = d1*(d2*d3 - t23*t23) - t12*(t12*d3 - t13*t23) + t13*(t12*t23 - t13*d2);

        // Normalized Lode angle parameter theta_bar [-1, 1]
        Real r_val = 27.0 * J3 / (2.0 * sigma_eq * sigma_eq * sigma_eq);
        if (r_val > 1.0) r_val = 1.0;
        if (r_val < -1.0) r_val = -1.0;
        // theta_bar = 1 - (2/pi)*acos(r_val) -- but use simplified form
        Real theta_bar = 1.0 - (2.0 / 3.14159265358979) * Kokkos::acos(r_val);

        // EMC fracture strain
        Real A = emc_.A_mc;
        Real B = emc_.B_mc;
        Real n = emc_.n_mc;
        Real c0 = emc_.c0;

        // Modified MC fracture strain envelope
        // eps_f = [A/c0] * [ cos(theta_pi/6) / cos(theta_pi/6 - pi/6) ]
        //       * [ sqrt((1+c1^2)/3) / (cos(theta_pi/6) + c1*(eta + 1/3)*sin(phi)) ]^(1/n)
        // Simplified EMC form:
        Real sin_phi = Kokkos::sin(emc_.phi);
        Real theta_pi6 = (1.0 - theta_bar) * 3.14159265358979 / 6.0;
        Real cos_t = Kokkos::cos(theta_pi6);
        Real sin_t = Kokkos::sin(theta_pi6);
        if (cos_t < 1.0e-10) cos_t = 1.0e-10;

        Real bracket1 = (A / (c0 > 1.0e-20 ? c0 : 1.0e-20));
        Real term1 = cos_t + sin_phi * sin_t / Kokkos::sqrt(3.0);
        Real term2 = cos_t / 3.0 + sin_phi * (eta + 1.0 / 3.0);
        if (term2 < 1.0e-20) term2 = 1.0e-20;

        Real eps_f = (B + bracket1 * Kokkos::sqrt((1.0 + sin_phi*sin_phi) / 3.0)) /
                     Kokkos::pow(term1 * term2, n);
        if (eps_f < 1.0e-10) eps_f = 1.0e-10;
        if (eps_f > 10.0) eps_f = 10.0;

        // Damage accumulation
        Real d_eps_p = mstate.plastic_strain - fstate.history[4]; // incremental plastic strain
        if (d_eps_p < 0.0) d_eps_p = 0.0;

        fstate.history[0] += d_eps_p / eps_f;
        fstate.history[1] = eps_f;
        fstate.history[2] = eta;
        fstate.history[3] = theta_bar;
        fstate.history[4] = mstate.plastic_strain; // store current eps_p

        // Clamp damage
        if (fstate.history[0] > 1.0) fstate.history[0] = 1.0;
        fstate.damage = fstate.history[0];

        if (fstate.damage >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& state, Real damage) const {
        return damage >= 1.0;
    }

    const char* name() const { return "EMCFailure"; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const EMCFailureParams& p) {
        return std::make_unique<EMCFailure>(base, p);
    }

private:
    EMCFailureParams emc_;
};


// ============================================================================
// 2. SahraeiFailure - Battery cell short-circuit
// ============================================================================

/**
 * @brief Sahraei battery cell short-circuit failure criterion
 *
 * Combined strain-pressure criterion for lithium-ion battery cells.
 * Short-circuit occurs when:
 *   alpha * (eps_eff / eps_crit) + (1-alpha) * (P / P_crit) >= 1
 *
 * History:
 *   [0] = failure index F
 *   [1] = strain contribution
 *   [2] = pressure contribution
 *
 * Reference: Sahraei et al. (2012), J. Power Sources
 */
class SahraeiFailure : public FailureModel {
public:
    SahraeiFailure(const FailureModelParameters& base_params,
                   const SahraeiFailureParams& sah_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , sah_(sah_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real alpha = sah_.combined_weight;
        Real eps_eff = mstate.plastic_strain;
        Real eps_crit = sah_.eps_crit;
        if (eps_crit < 1.0e-30) eps_crit = 1.0e-30;

        // Hydrostatic pressure (positive in compression)
        Real P = -(mstate.stress[0] + mstate.stress[1] + mstate.stress[2]) / 3.0;
        Real P_crit = sah_.pressure_crit;
        if (P_crit < 1.0e-30) P_crit = 1.0e-30;

        // Only compressive pressure contributes
        Real P_pos = (P > 0.0) ? P : 0.0;

        Real strain_term = eps_eff / eps_crit;
        Real pressure_term = P_pos / P_crit;

        Real F = alpha * strain_term + (1.0 - alpha) * pressure_term;

        fstate.history[0] = F;
        fstate.history[1] = strain_term;
        fstate.history[2] = pressure_term;

        // Track maximum (irreversible)
        if (F > fstate.history[3]) {
            fstate.history[3] = F;
        }

        fstate.damage = (fstate.history[3] > 1.0) ? 1.0 : fstate.history[3];

        if (fstate.history[3] >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1; // short-circuit
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& /*state*/, Real damage) const {
        return damage >= 1.0;
    }

    const char* name() const { return "SahraeiFailure"; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const SahraeiFailureParams& p) {
        return std::make_unique<SahraeiFailure>(base, p);
    }

private:
    SahraeiFailureParams sah_;
};


// ============================================================================
// 3. SyazwanFailure - Strain-rate dependent ductile
// ============================================================================

/**
 * @brief Syazwan strain-rate dependent ductile failure model
 *
 * Failure strain depends on strain rate:
 *   eps_f = eps_f0 * (1 + C_rate * ln(eps_dot/eps_dot_ref))^n_rate
 *
 * Damage: D += d_eps_p / eps_f (incremental).
 *
 * History:
 *   [0] = accumulated damage D
 *   [1] = current fracture strain eps_f
 *   [2] = previous plastic strain
 *
 * Reference: Syazwan et al. (2019), Metals
 */
class SyazwanFailure : public FailureModel {
public:
    SyazwanFailure(const FailureModelParameters& base_params,
                   const SyazwanFailureParams& syz_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , syz_(syz_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real eps_dot = mstate.effective_strain_rate;
        Real eps_dot_ref = syz_.eps_dot_ref;
        if (eps_dot_ref < 1.0e-30) eps_dot_ref = 1.0e-30;

        // Compute rate-dependent failure strain
        Real ratio = eps_dot / eps_dot_ref;
        if (ratio < 1.0) ratio = 1.0; // clamp to quasi-static

        Real log_ratio = Kokkos::log(ratio);
        Real rate_factor = 1.0 + syz_.C_rate * log_ratio;
        if (rate_factor < 0.1) rate_factor = 0.1; // safety clamp

        Real eps_f = syz_.eps_f0 * Kokkos::pow(rate_factor, syz_.n_rate);
        if (eps_f < 1.0e-10) eps_f = 1.0e-10;

        // Incremental damage
        Real d_eps_p = mstate.plastic_strain - fstate.history[2];
        if (d_eps_p < 0.0) d_eps_p = 0.0;

        fstate.history[0] += d_eps_p / eps_f;
        fstate.history[1] = eps_f;
        fstate.history[2] = mstate.plastic_strain;

        if (fstate.history[0] > 1.0) fstate.history[0] = 1.0;
        fstate.damage = fstate.history[0];

        if (fstate.damage >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& /*state*/, Real damage) const {
        return damage >= 1.0;
    }

    const char* name() const { return "SyazwanFailure"; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const SyazwanFailureParams& p) {
        return std::make_unique<SyazwanFailure>(base, p);
    }

private:
    SyazwanFailureParams syz_;
};


// ============================================================================
// 4. VisualFailure - Visual damage indicator
// ============================================================================

/**
 * @brief Visual damage indicator (no element deletion)
 *
 * Purely visual: maps plastic strain to damage coloring levels.
 * Never triggers element deletion.
 *
 * damage = plastic_strain / eps_ref, clamped to [0, 1].
 * Levels:
 *   0 -> d_threshold_low:  green
 *   d_threshold_low -> d_threshold_med: yellow
 *   d_threshold_med -> d_threshold_high: orange
 *   d_threshold_high -> 1.0: red
 *
 * History:
 *   [0] = current damage level (0-3 for color band)
 *   [1] = raw damage ratio
 */
class VisualFailure : public FailureModel {
public:
    VisualFailure(const FailureModelParameters& base_params,
                  const VisualFailureParams& vis_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , vis_(vis_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real eps_ref = vis_.eps_ref;
        if (eps_ref < 1.0e-30) eps_ref = 1.0e-30;

        Real raw_damage = mstate.plastic_strain / eps_ref;
        if (raw_damage > 1.0) raw_damage = 1.0;
        if (raw_damage < 0.0) raw_damage = 0.0;

        fstate.history[1] = raw_damage;

        // Determine color level
        if (raw_damage < vis_.d_threshold_low) {
            fstate.history[0] = 0.0; // green/undamaged
        } else if (raw_damage < vis_.d_threshold_med) {
            fstate.history[0] = 1.0; // yellow/low
        } else if (raw_damage < vis_.d_threshold_high) {
            fstate.history[0] = 2.0; // orange/medium
        } else {
            fstate.history[0] = 3.0; // red/high
        }

        fstate.damage = raw_damage;
        // Visual model never triggers element deletion
        fstate.failed = false;
        fstate.failure_mode = 0;
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& /*state*/, Real /*damage*/) const {
        return false; // never fails elements
    }

    const char* name() const { return "VisualFailure"; }

    // Override to prevent deletion
    void degrade_stress(Real* /*sigma*/, const FailureState& /*fstate*/) const override {
        // No stress degradation - purely visual
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const VisualFailureParams& p) {
        return std::make_unique<VisualFailure>(base, p);
    }

private:
    VisualFailureParams vis_;
};


// ============================================================================
// 5. NXTFailure - Next-gen composite
// ============================================================================

/**
 * @brief Next-generation composite failure with separate fiber/matrix damage
 *
 * Four independent damage modes:
 *   fiber_tension:    f_ft = (sigma_1/Xt)^2 + (tau_12/S12)^2
 *   fiber_compression: f_fc = (sigma_1/Xc)^2
 *   matrix_tension:   f_mt = (sigma_2/Yt)^2 + (tau_12/S12)^2
 *   matrix_compression: f_mc = (sigma_2/Yc)^2 + (tau_12/S12)^2
 *
 * Damage evolves: d_i = 1 - 1/r_i * exp(-m_i*(r_i-1)) where r_i = sqrt(f_i)
 *
 * History:
 *   [0] = fiber tension damage
 *   [1] = fiber compression damage
 *   [2] = matrix tension damage
 *   [3] = matrix compression damage
 *   [4] = max fiber tension loading r
 *   [5] = max fiber compression loading r
 *   [6] = max matrix tension loading r
 *   [7] = max matrix compression loading r
 *
 * Reference: Maimi et al. (2007), Comp. Sci. Tech.
 */
class NXTFailure : public FailureModel {
public:
    NXTFailure(const FailureModelParameters& base_params,
               const NXTFailureParams& nxt_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , nxt_(nxt_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real s1 = mstate.stress[0];
        Real s2 = mstate.stress[1];
        Real t12 = mstate.stress[3];

        Real Xt = nxt_.Xt; if (Xt < 1.0e-20) Xt = 1.0e-20;
        Real Xc = nxt_.Xc; if (Xc < 1.0e-20) Xc = 1.0e-20;
        Real Yt = nxt_.Yt; if (Yt < 1.0e-20) Yt = 1.0e-20;
        Real Yc = nxt_.Yc; if (Yc < 1.0e-20) Yc = 1.0e-20;
        Real S12 = nxt_.S12; if (S12 < 1.0e-20) S12 = 1.0e-20;

        // Fiber tension
        Real f_ft = 0.0;
        if (s1 > 0.0) {
            f_ft = (s1/Xt)*(s1/Xt) + (t12/S12)*(t12/S12);
        }

        // Fiber compression
        Real f_fc = 0.0;
        if (s1 < 0.0) {
            f_fc = (s1/Xc)*(s1/Xc);
        }

        // Matrix tension
        Real f_mt = 0.0;
        if (s2 > 0.0) {
            f_mt = (s2/Yt)*(s2/Yt) + (t12/S12)*(t12/S12);
        }

        // Matrix compression
        Real f_mc = 0.0;
        if (s2 < 0.0) {
            f_mc = (s2/Yc)*(s2/Yc) + (t12/S12)*(t12/S12);
        }

        // Loading functions (max over time for irreversibility)
        Real r_ft = Kokkos::sqrt(f_ft);
        Real r_fc = Kokkos::sqrt(f_fc);
        Real r_mt = Kokkos::sqrt(f_mt);
        Real r_mc = Kokkos::sqrt(f_mc);

        if (r_ft > fstate.history[4]) fstate.history[4] = r_ft;
        if (r_fc > fstate.history[5]) fstate.history[5] = r_fc;
        if (r_mt > fstate.history[6]) fstate.history[6] = r_mt;
        if (r_mc > fstate.history[7]) fstate.history[7] = r_mc;

        // Damage evolution: d = 1 - (1/r)*exp(-m*(r-1))
        auto evolve_damage = [](Real r_max, Real m) -> Real {
            if (r_max < 1.0) return 0.0;
            Real d = 1.0 - (1.0 / r_max) * Kokkos::exp(-m * (r_max - 1.0));
            if (d < 0.0) d = 0.0;
            if (d > 1.0) d = 1.0;
            return d;
        };

        fstate.history[0] = evolve_damage(fstate.history[4], nxt_.m_fiber);
        fstate.history[1] = evolve_damage(fstate.history[5], nxt_.m_fiber);
        fstate.history[2] = evolve_damage(fstate.history[6], nxt_.m_matrix);
        fstate.history[3] = evolve_damage(fstate.history[7], nxt_.m_matrix);

        // Total damage = max of all modes
        Real d_max = fstate.history[0];
        if (fstate.history[1] > d_max) d_max = fstate.history[1];
        if (fstate.history[2] > d_max) d_max = fstate.history[2];
        if (fstate.history[3] > d_max) d_max = fstate.history[3];

        fstate.damage = d_max;

        if (d_max >= 1.0) {
            fstate.failed = true;
            // Identify dominant mode
            if (fstate.history[0] >= 1.0) fstate.failure_mode = 1;
            else if (fstate.history[1] >= 1.0) fstate.failure_mode = 2;
            else if (fstate.history[2] >= 1.0) fstate.failure_mode = 3;
            else fstate.failure_mode = 4;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& /*state*/, Real damage) const {
        return damage >= 1.0;
    }

    const char* name() const { return "NXTFailure"; }

    Real fiber_tension_damage(const FailureState& fs) const { return fs.history[0]; }
    Real fiber_compression_damage(const FailureState& fs) const { return fs.history[1]; }
    Real matrix_tension_damage(const FailureState& fs) const { return fs.history[2]; }
    Real matrix_compression_damage(const FailureState& fs) const { return fs.history[3]; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const NXTFailureParams& p) {
        return std::make_unique<NXTFailure>(base, p);
    }

private:
    NXTFailureParams nxt_;
};


// ============================================================================
// 6. Gene1Failure - Polynomial in stress space
// ============================================================================

/**
 * @brief Polynomial failure criterion in stress space
 *
 * F = a[0]*s1^2 + a[1]*s2^2 + a[2]*s1*s2 + a[3]*s12^2 + a[4]*s1 + a[5]*s2
 *
 * Damage evolves incrementally when F >= 1.
 *
 * History:
 *   [0] = max failure index F
 *   [1] = accumulated damage
 */
class Gene1Failure : public FailureModel {
public:
    Gene1Failure(const FailureModelParameters& base_params,
                 const Gene1FailureParams& gen_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , gen_(gen_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real s1 = mstate.stress[0];
        Real s2 = mstate.stress[1];
        Real s12 = mstate.stress[3];

        Real F = gen_.a[0]*s1*s1 + gen_.a[1]*s2*s2 + gen_.a[2]*s1*s2
               + gen_.a[3]*s12*s12 + gen_.a[4]*s1 + gen_.a[5]*s2;

        // Track maximum
        if (F > fstate.history[0]) {
            fstate.history[0] = F;
        }

        // Damage evolution: once F exceeds 1, damage grows
        if (fstate.history[0] >= 1.0) {
            // Incremental damage proportional to excess
            fstate.history[1] = 1.0; // immediate full damage on first exceedance
        }

        fstate.damage = fstate.history[1];
        if (fstate.damage > 1.0) fstate.damage = 1.0;

        if (fstate.damage >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& /*state*/, Real damage) const {
        return damage >= 1.0;
    }

    const char* name() const { return "Gene1Failure"; }

    Real failure_index(const FailureState& fs) const { return fs.history[0]; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const Gene1FailureParams& p) {
        return std::make_unique<Gene1Failure>(base, p);
    }

private:
    Gene1FailureParams gen_;
};


// ============================================================================
// 7. IniEvoFailure - Initial damage + evolution
// ============================================================================

/**
 * @brief Initial damage + linear evolution failure model
 *
 * Damage starts at d0, then evolves linearly from d0 to 1.0
 * between initiation strain eps_i and failure strain eps_f.
 *
 * D = d0                                          if eps_p < eps_i
 * D = d0 + (1-d0) * (eps_p - eps_i)/(eps_f - eps_i)  if eps_i <= eps_p < eps_f
 * D = 1.0                                         if eps_p >= eps_f
 *
 * History:
 *   [0] = current damage D
 *   [1] = max plastic strain seen
 */
class IniEvoFailure : public FailureModel {
public:
    IniEvoFailure(const FailureModelParameters& base_params,
                  const IniEvoFailureParams& ini_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , ini_(ini_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real eps_p = mstate.plastic_strain;
        Real d0 = ini_.d0;
        Real eps_i = ini_.eps_i;
        Real eps_f = ini_.eps_f;

        // Track max plastic strain (irreversible)
        if (eps_p > fstate.history[1]) {
            fstate.history[1] = eps_p;
        }
        Real eps_max = fstate.history[1];

        Real D;
        if (eps_max < eps_i) {
            D = d0;
        } else if (eps_max >= eps_f) {
            D = 1.0;
        } else {
            Real denom = eps_f - eps_i;
            if (denom < 1.0e-30) denom = 1.0e-30;
            D = d0 + (1.0 - d0) * (eps_max - eps_i) / denom;
        }

        if (D > 1.0) D = 1.0;
        if (D < d0) D = d0; // never below initial damage

        fstate.history[0] = D;
        fstate.damage = D;

        if (D >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& /*state*/, Real damage) const {
        return damage >= 1.0;
    }

    const char* name() const { return "IniEvoFailure"; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const IniEvoFailureParams& p) {
        return std::make_unique<IniEvoFailure>(base, p);
    }

private:
    IniEvoFailureParams ini_;
};


// ============================================================================
// 8. AlterFailure - Alternating-load fatigue
// ============================================================================

/**
 * @brief Alternating-load fatigue failure (Miner's rule)
 *
 * Uses simplified rainflow counting with Basquin-Coffin-Manson S-N relation.
 * N_f from total strain amplitude:
 *   eps_a = (sigma_f'/E) * (2*N_f)^b + eps_f' * (2*N_f)^c
 *
 * Simplified: for a given stress amplitude sigma_a,
 *   N_f = (sigma_a / sigma_f')^(1/b) / 2  (Basquin high-cycle)
 *
 * Miner's rule: D += 1/N_f per half-cycle.
 *
 * History:
 *   [0] = accumulated fatigue damage D
 *   [1] = previous stress (for cycle detection)
 *   [2] = cycle count
 *   [3] = sign of last stress change (for rainflow approx)
 *
 * Reference: Miner (1945), J. Applied Mechanics
 */
class AlterFailure : public FailureModel {
public:
    AlterFailure(const FailureModelParameters& base_params,
                 const AlterFailureParams& alt_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , alt_(alt_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // Current von Mises stress
        Real s1 = mstate.stress[0], s2 = mstate.stress[1], s3 = mstate.stress[2];
        Real t12 = mstate.stress[3], t23 = mstate.stress[4], t13 = mstate.stress[5];

        Real vm = Kokkos::sqrt(0.5*((s1-s2)*(s1-s2) + (s2-s3)*(s2-s3) + (s3-s1)*(s3-s1))
                  + 3.0*(t12*t12 + t23*t23 + t13*t13));

        // Signed stress (tension positive, compression negative based on hydrostatic)
        Real sigma_h = (s1 + s2 + s3) / 3.0;
        Real sigma_signed = (sigma_h >= 0.0) ? vm : -vm;

        Real prev_stress = fstate.history[1];
        Real prev_sign = fstate.history[3];

        // Detect sign change (half-cycle) via simplified rainflow
        Real current_sign = (sigma_signed - prev_stress > 0.0) ? 1.0 : -1.0;
        if (Kokkos::fabs(sigma_signed - prev_stress) < 1.0e-10) {
            current_sign = prev_sign;
        }

        bool half_cycle = (prev_sign != 0.0 && current_sign != prev_sign);

        if (half_cycle) {
            // Stress amplitude for this half-cycle
            Real sigma_a = Kokkos::fabs(sigma_signed - prev_stress) / 2.0;

            // Only accumulate if above endurance limit
            if (sigma_a > alt_.S_f) {
                // Basquin: N_f = 0.5 * (sigma_a / sigma_f')^(1/b)
                Real sigma_f_prime = alt_.sigma_f_prime;
                if (sigma_f_prime < 1.0e-20) sigma_f_prime = 1.0e-20;
                Real b = alt_.b_basquin;
                if (Kokkos::fabs(b) < 1.0e-20) b = -0.12;

                Real N_f = 0.5 * Kokkos::pow(sigma_a / sigma_f_prime, 1.0 / b);
                if (N_f < 1.0) N_f = 1.0;

                // Miner's rule: add 1/N_f for this half-cycle (count as 0.5 cycle)
                fstate.history[0] += 0.5 / N_f;
            }

            fstate.history[2] += 0.5; // half-cycle count
        }

        fstate.history[1] = sigma_signed;
        fstate.history[3] = current_sign;

        if (fstate.history[0] > 1.0) fstate.history[0] = 1.0;
        fstate.damage = fstate.history[0];

        if (fstate.damage >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& /*state*/, Real damage) const {
        return damage >= 1.0;
    }

    const char* name() const { return "AlterFailure"; }

    Real cycle_count(const FailureState& fs) const { return fs.history[2]; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const AlterFailureParams& p) {
        return std::make_unique<AlterFailure>(base, p);
    }

private:
    AlterFailureParams alt_;
};


// ============================================================================
// 9. BiquadFailure - Biquadratic failure surface
// ============================================================================

/**
 * @brief Biquadratic failure surface in stress space
 *
 * F = F1*s1^2 + F2*s2^2 + F3*s1*s2 + F4*s12^2 + F5*s1 + F6*s2
 *
 * Fails when F >= 1.
 *
 * History:
 *   [0] = max failure index F
 */
class BiquadFailure : public FailureModel {
public:
    BiquadFailure(const FailureModelParameters& base_params,
                  const BiquadFailureParams& biq_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , biq_(biq_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real s1 = mstate.stress[0];
        Real s2 = mstate.stress[1];
        Real s12 = mstate.stress[3];

        Real F = biq_.F1*s1*s1 + biq_.F2*s2*s2 + biq_.F3*s1*s2
               + biq_.F4*s12*s12 + biq_.F5*s1 + biq_.F6*s2;

        if (F > fstate.history[0]) {
            fstate.history[0] = F;
        }

        fstate.damage = (fstate.history[0] > 1.0) ? 1.0 : fstate.history[0];
        if (fstate.damage < 0.0) fstate.damage = 0.0;

        if (fstate.history[0] >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& /*state*/, Real damage) const {
        return damage >= 1.0;
    }

    const char* name() const { return "BiquadFailure"; }

    Real failure_index(const FailureState& fs) const { return fs.history[0]; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const BiquadFailureParams& p) {
        return std::make_unique<BiquadFailure>(base, p);
    }

private:
    BiquadFailureParams biq_;
};


// ============================================================================
// 10. OrthBiquadFailure - Orthotropic biquadratic
// ============================================================================

/**
 * @brief Orthotropic biquadratic failure criterion
 *
 * F = (s1/X)^2 + (s2/Y)^2 + (s12/S12)^2 + (s23/S23)^2 - (s1*s2)/(X*Y)
 *
 * X = Xt if s1 >= 0, Xc if s1 < 0 (similarly for Y).
 *
 * History:
 *   [0] = max failure index
 */
class OrthBiquadFailure : public FailureModel {
public:
    OrthBiquadFailure(const FailureModelParameters& base_params,
                      const OrthBiquadFailureParams& obq_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , obq_(obq_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real s1 = mstate.stress[0];
        Real s2 = mstate.stress[1];
        Real s12 = mstate.stress[3];
        Real s23 = mstate.stress[4];

        // Select strength based on sign
        Real X = (s1 >= 0.0) ? obq_.Xt : obq_.Xc;
        Real Y = (s2 >= 0.0) ? obq_.Yt : obq_.Yc;
        Real S12 = obq_.S12;
        Real S23 = obq_.S23;

        if (X < 1.0e-20) X = 1.0e-20;
        if (Y < 1.0e-20) Y = 1.0e-20;
        if (S12 < 1.0e-20) S12 = 1.0e-20;
        if (S23 < 1.0e-20) S23 = 1.0e-20;

        Real F = (s1/X)*(s1/X) + (s2/Y)*(s2/Y)
               + (s12/S12)*(s12/S12) + (s23/S23)*(s23/S23)
               - (s1*s2)/(X*Y);

        if (F > fstate.history[0]) {
            fstate.history[0] = F;
        }

        fstate.damage = (fstate.history[0] > 1.0) ? 1.0 : fstate.history[0];
        if (fstate.damage < 0.0) fstate.damage = 0.0;

        if (fstate.history[0] >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& /*state*/, Real damage) const {
        return damage >= 1.0;
    }

    const char* name() const { return "OrthBiquadFailure"; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const OrthBiquadFailureParams& p) {
        return std::make_unique<OrthBiquadFailure>(base, p);
    }

private:
    OrthBiquadFailureParams obq_;
};


// ============================================================================
// 11. OrthEnergFailure - Orthotropic energy-based
// ============================================================================

/**
 * @brief Orthotropic energy-based failure criterion
 *
 * Energy density per direction:
 *   W_1 = 0.5 * sigma_1 * eps_1  (1-direction)
 *   W_2 = 0.5 * sigma_2 * eps_2  (2-direction)
 *   W_12 = 0.5 * tau_12 * gamma_12 (shear)
 *
 * Damage per direction: D_i = W_i / G_ic
 * Total D = max(D_1, D_2, D_12)
 *
 * History:
 *   [0] = D_1 (1-direction damage)
 *   [1] = D_2 (2-direction damage)
 *   [2] = D_12 (shear damage)
 *   [3] = W_1 accumulated
 *   [4] = W_2 accumulated
 *   [5] = W_12 accumulated
 */
class OrthEnergFailure : public FailureModel {
public:
    OrthEnergFailure(const FailureModelParameters& base_params,
                     const OrthEnergFailureParams& oen_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , oen_(oen_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // Current energy densities (secant approximation)
        Real W_1 = 0.5 * Kokkos::fabs(mstate.stress[0] * mstate.strain[0]);
        Real W_2 = 0.5 * Kokkos::fabs(mstate.stress[1] * mstate.strain[1]);
        Real W_12 = 0.5 * Kokkos::fabs(mstate.stress[3] * mstate.strain[3]);

        // Track max energy (irreversible)
        if (W_1 > fstate.history[3]) fstate.history[3] = W_1;
        if (W_2 > fstate.history[4]) fstate.history[4] = W_2;
        if (W_12 > fstate.history[5]) fstate.history[5] = W_12;

        Real G1c = oen_.G1c; if (G1c < 1.0e-30) G1c = 1.0e-30;
        Real G2c = oen_.G2c; if (G2c < 1.0e-30) G2c = 1.0e-30;
        Real G12c = oen_.G12c; if (G12c < 1.0e-30) G12c = 1.0e-30;

        fstate.history[0] = fstate.history[3] / G1c;
        fstate.history[1] = fstate.history[4] / G2c;
        fstate.history[2] = fstate.history[5] / G12c;

        // Total damage = max of directional damages
        Real d_max = fstate.history[0];
        if (fstate.history[1] > d_max) d_max = fstate.history[1];
        if (fstate.history[2] > d_max) d_max = fstate.history[2];

        if (d_max > 1.0) d_max = 1.0;
        fstate.damage = d_max;

        if (d_max >= 1.0) {
            fstate.failed = true;
            // Identify dominant mode
            if (fstate.history[0] >= fstate.history[1] && fstate.history[0] >= fstate.history[2])
                fstate.failure_mode = 1;
            else if (fstate.history[1] >= fstate.history[2])
                fstate.failure_mode = 2;
            else
                fstate.failure_mode = 3;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& /*state*/, Real damage) const {
        return damage >= 1.0;
    }

    const char* name() const { return "OrthEnergFailure"; }

    Real direction_damage(const FailureState& fs, int dir) const {
        if (dir >= 0 && dir < 3) return fs.history[dir];
        return 0.0;
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const OrthEnergFailureParams& p) {
        return std::make_unique<OrthEnergFailure>(base, p);
    }

private:
    OrthEnergFailureParams oen_;
};


// ============================================================================
// 12. OrthStrainFailure - Orthotropic max-strain
// ============================================================================

/**
 * @brief Orthotropic maximum strain failure criterion
 *
 * Independent strain limits per direction:
 *   f_1 = eps_1/eps_1t (tension) or |eps_1|/eps_1c (compression)
 *   f_2 = eps_2/eps_2t (tension) or |eps_2|/eps_2c (compression)
 *   f_12 = |gamma_12| / gamma_12_max
 *
 * Failure when max(f_i) >= 1.
 *
 * History:
 *   [0] = f_1
 *   [1] = f_2
 *   [2] = f_12
 *   [3] = max f over time
 */
class OrthStrainFailure : public FailureModel {
public:
    OrthStrainFailure(const FailureModelParameters& base_params,
                      const OrthStrainFailureParams& ost_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , ost_(ost_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real e1 = mstate.strain[0];
        Real e2 = mstate.strain[1];
        Real g12 = mstate.strain[3];

        // Direction 1
        Real f1;
        if (e1 >= 0.0) {
            Real eps_1t = ost_.eps_1t; if (eps_1t < 1.0e-30) eps_1t = 1.0e-30;
            f1 = e1 / eps_1t;
        } else {
            Real eps_1c = ost_.eps_1c; if (eps_1c < 1.0e-30) eps_1c = 1.0e-30;
            f1 = Kokkos::fabs(e1) / eps_1c;
        }

        // Direction 2
        Real f2;
        if (e2 >= 0.0) {
            Real eps_2t = ost_.eps_2t; if (eps_2t < 1.0e-30) eps_2t = 1.0e-30;
            f2 = e2 / eps_2t;
        } else {
            Real eps_2c = ost_.eps_2c; if (eps_2c < 1.0e-30) eps_2c = 1.0e-30;
            f2 = Kokkos::fabs(e2) / eps_2c;
        }

        // Shear
        Real gmax = ost_.gamma_12_max; if (gmax < 1.0e-30) gmax = 1.0e-30;
        Real f12 = Kokkos::fabs(g12) / gmax;

        fstate.history[0] = f1;
        fstate.history[1] = f2;
        fstate.history[2] = f12;

        Real f_max = f1;
        if (f2 > f_max) f_max = f2;
        if (f12 > f_max) f_max = f12;

        // Track max (irreversible)
        if (f_max > fstate.history[3]) {
            fstate.history[3] = f_max;
        }

        fstate.damage = (fstate.history[3] > 1.0) ? 1.0 : fstate.history[3];

        if (fstate.history[3] >= 1.0) {
            fstate.failed = true;
            // Dominant mode
            if (f1 >= f2 && f1 >= f12) fstate.failure_mode = 1;
            else if (f2 >= f12) fstate.failure_mode = 2;
            else fstate.failure_mode = 3;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& /*state*/, Real damage) const {
        return damage >= 1.0;
    }

    const char* name() const { return "OrthStrainFailure"; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const OrthStrainFailureParams& p) {
        return std::make_unique<OrthStrainFailure>(base, p);
    }

private:
    OrthStrainFailureParams ost_;
};


// ============================================================================
// 13. TensStrainFailure - Tensile strain criterion
// ============================================================================

/**
 * @brief Tensile strain failure criterion
 *
 * D = (eps_max_principal / eps_crit)^n for eps_max_principal > eps_crit
 *
 * Uses max principal strain approximation from strain tensor.
 *
 * History:
 *   [0] = max principal strain seen
 *   [1] = current damage
 */
class TensStrainFailure : public FailureModel {
public:
    TensStrainFailure(const FailureModelParameters& base_params,
                      const TensStrainFailureParams& tns_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , tns_(tns_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // Approximate max principal strain from normal strains
        // For general case: eigenvalue of strain tensor
        // Simplified: max of normal strains (conservative for tensile)
        Real e1 = mstate.strain[0];
        Real e2 = mstate.strain[1];
        Real e3 = mstate.strain[2];

        // 2D principal strain approximation (plane stress)
        Real e_avg = 0.5 * (e1 + e2);
        Real e_diff = 0.5 * (e1 - e2);
        Real gamma_xy = 0.5 * mstate.strain[3]; // engineering -> tensor shear
        Real R = Kokkos::sqrt(e_diff*e_diff + gamma_xy*gamma_xy);
        Real eps_p1 = e_avg + R;

        // Also check e3
        Real eps_max = eps_p1;
        if (e3 > eps_max) eps_max = e3;

        // Track max (irreversible)
        if (eps_max > fstate.history[0]) {
            fstate.history[0] = eps_max;
        }

        Real eps_crit = tns_.eps_crit;
        if (eps_crit < 1.0e-30) eps_crit = 1.0e-30;
        Real n = tns_.damage_evolution_exponent;

        Real D = 0.0;
        if (fstate.history[0] > eps_crit) {
            D = Kokkos::pow(fstate.history[0] / eps_crit, n);
            // Normalize so D=1 at some reasonable multiple (use D directly)
            // Since eps/eps_crit > 1 and raised to power n, D > 1 immediately
            // Re-interpret: D = ((eps - eps_crit) / eps_crit)^n capped at 1
            Real excess = (fstate.history[0] - eps_crit) / eps_crit;
            D = Kokkos::pow(excess, n);
            if (D > 1.0) D = 1.0;
        }

        fstate.history[1] = D;
        fstate.damage = D;

        if (D >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& /*state*/, Real damage) const {
        return damage >= 1.0;
    }

    const char* name() const { return "TensStrainFailure"; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const TensStrainFailureParams& p) {
        return std::make_unique<TensStrainFailure>(base, p);
    }

private:
    TensStrainFailureParams tns_;
};


// ============================================================================
// 14. SnConnectFailure - Spot-weld connection
// ============================================================================

/**
 * @brief Spot-weld connection failure criterion
 *
 * Force-based criterion:
 *   (F_n/F_n_max)^n_norm + (F_s/F_s_max)^n_shear >= 1
 *
 * Forces derived from stress * area:
 *   F_n = |sigma_normal| * area
 *   F_s = |tau_shear| * area
 *
 * History:
 *   [0] = max failure index
 *   [1] = normal force ratio
 *   [2] = shear force ratio
 */
class SnConnectFailure : public FailureModel {
public:
    SnConnectFailure(const FailureModelParameters& base_params,
                     const SnConnectFailureParams& snc_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , snc_(snc_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real area = snc_.area;
        if (area < 1.0e-30) area = 1.0e-30;

        // Normal stress -> force (through-thickness = stress[2])
        // For spot welds: normal = sigma_zz, shear = sqrt(tau_xz^2 + tau_yz^2)
        Real sigma_n = mstate.stress[2];
        Real tau_xz = mstate.stress[5];
        Real tau_yz = mstate.stress[4];

        Real F_n = Kokkos::fabs(sigma_n) * area;
        Real F_s = Kokkos::sqrt(tau_xz*tau_xz + tau_yz*tau_yz) * area;

        Real F_n_max = snc_.F_normal_max;
        Real F_s_max = snc_.F_shear_max;
        if (F_n_max < 1.0e-30) F_n_max = 1.0e-30;
        if (F_s_max < 1.0e-30) F_s_max = 1.0e-30;

        Real r_n = F_n / F_n_max;
        Real r_s = F_s / F_s_max;

        Real FI = Kokkos::pow(r_n, snc_.n_norm) + Kokkos::pow(r_s, snc_.n_shear);

        fstate.history[1] = r_n;
        fstate.history[2] = r_s;

        if (FI > fstate.history[0]) {
            fstate.history[0] = FI;
        }

        fstate.damage = (fstate.history[0] > 1.0) ? 1.0 : fstate.history[0];

        if (fstate.history[0] >= 1.0) {
            fstate.failed = true;
            // Dominant mode
            if (Kokkos::pow(r_n, snc_.n_norm) >= Kokkos::pow(r_s, snc_.n_shear))
                fstate.failure_mode = 1; // normal
            else
                fstate.failure_mode = 2; // shear
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& /*state*/, Real damage) const {
        return damage >= 1.0;
    }

    const char* name() const { return "SnConnectFailure"; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const SnConnectFailureParams& p) {
        return std::make_unique<SnConnectFailure>(base, p);
    }

private:
    SnConnectFailureParams snc_;
};


} // namespace failure
} // namespace physics
} // namespace nxs
