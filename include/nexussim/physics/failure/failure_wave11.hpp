#pragma once

/**
 * @file failure_wave11.hpp
 * @brief 12 additional failure/damage models for metals, forming, and fabrics
 *
 * Models implemented:
 *   1.  JohnsonCookFailure   - Cumulative damage with triaxiality, rate, and temperature
 *   2.  CockcroftLathamFailure - Maximum principal stress work criterion
 *   3.  LemaitreCDMFailure   - Continuum damage mechanics (CDM)
 *   4.  PuckFailure          - Inter-fiber and fiber failure for composites
 *   5.  FLDFailure           - Forming Limit Diagram (tabulated)
 *   6.  WilkinsFailure       - Cumulative damage for ductile metals
 *   7.  TulerButcherFailure  - Spall failure under dynamic tension
 *   8.  MaxStressFailure     - Component-wise maximum stress criterion
 *   9.  MaxStrainFailure     - Component-wise maximum strain criterion
 *   10. EnergyFailure        - Strain energy density criterion
 *   11. WierzbickiFailure    - Modified Mohr-Coulomb (triaxiality + Lode angle)
 *   12. FabricFailure        - Biaxial failure for woven fabrics
 *
 * All models inherit from FailureModel, implement compute_damage(), and
 * provide a static create() factory method. KOKKOS_INLINE_FUNCTION is used
 * on helper routines for GPU portability.
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
// Extended failure model type enumeration
// ============================================================================

enum class FailureModelTypeExt {
    JohnsonCook,
    CockcroftLatham,
    LemaitreCDM,
    Puck,
    FLD,
    Wilkins,
    TulerButcher,
    MaxStress,
    MaxStrain,
    Energy,
    Wierzbicki,
    Fabric
};

// ============================================================================
// Parameter Structures (per-model)
// ============================================================================

/// Johnson-Cook failure parameters
struct JohnsonCookFailureParams {
    Real d1;              ///< Constant term
    Real d2;              ///< Exponential pre-factor
    Real d3;              ///< Triaxiality exponent (negative)
    Real d4;              ///< Strain-rate coefficient
    Real d5;              ///< Temperature coefficient
    Real eps_dot_ref;     ///< Reference strain rate (1/s)
    Real T_melt;          ///< Melting temperature (K)
    Real T_room;          ///< Reference (room) temperature (K)

    JohnsonCookFailureParams()
        : d1(0.05), d2(3.44), d3(-2.12), d4(0.002), d5(0.61)
        , eps_dot_ref(1.0), T_melt(1800.0), T_room(293.15) {}
};

/// Cockcroft-Latham parameters
struct CockcroftLathamParams {
    Real W_crit;          ///< Critical plastic work at fracture

    CockcroftLathamParams() : W_crit(1.0e6) {}
};

/// Lemaitre CDM parameters
struct LemaitreCDMParams {
    Real S;               ///< Damage strength parameter
    Real s_exp;           ///< Damage exponent
    Real E;               ///< Young's modulus (for Y calculation)
    Real D_crit;          ///< Critical damage (element erosion threshold)
    Real eps_D;           ///< Damage threshold strain (plastic)

    LemaitreCDMParams()
        : S(1.0e6), s_exp(1.0), E(2.1e11), D_crit(0.99), eps_D(0.0) {}
};

/// Puck failure parameters (composite)
struct PuckFailureParams {
    Real R_para_t;        ///< Fiber tensile strength (R_||t)
    Real R_para_c;        ///< Fiber compressive strength (R_||c)
    Real R_perp_t;        ///< Matrix tensile strength (R_perp_t)
    Real R_perp_c;        ///< Matrix compressive strength (R_perp_c)
    Real R_perp_para;     ///< In-plane shear strength (R_perp_||)
    Real eps_1t;          ///< Fiber tensile failure strain

    PuckFailureParams()
        : R_para_t(1.5e9), R_para_c(1.2e9)
        , R_perp_t(50.0e6), R_perp_c(200.0e6)
        , R_perp_para(75.0e6), eps_1t(0.02) {}
};

/// FLD failure parameters (forming)
struct FLDFailureParams {
    TabulatedCurve fld_curve;  ///< FLD curve: x = minor strain, y = major strain limit

    FLDFailureParams() {}
};

/// Wilkins failure parameters
struct WilkinsFailureParams {
    Real a_exp;           ///< Exponent on max principal stress
    Real b_exp;           ///< Exponent on asymmetry term
    Real f_crit;          ///< Critical integral value

    WilkinsFailureParams()
        : a_exp(1.0), b_exp(1.0), f_crit(1.0e6) {}
};

/// Tuler-Butcher spall parameters
struct TulerButcherParams {
    Real sigma_spall;     ///< Spall threshold stress (Pa)
    Real lambda;          ///< Exponent on overstress
    Real K_crit;          ///< Critical impulse integral

    TulerButcherParams()
        : sigma_spall(1.0e9), lambda(2.0), K_crit(1.0e3) {}
};

/// Maximum stress failure parameters
struct MaxStressFailureParams {
    Real sigma_limit[6];  ///< Stress limits for each Voigt component

    MaxStressFailureParams() {
        for (int i = 0; i < 6; ++i) sigma_limit[i] = 1.0e30;
    }
};

/// Maximum strain failure parameters
struct MaxStrainFailureParams {
    Real eps_limit[6];    ///< Strain limits for each Voigt component

    MaxStrainFailureParams() {
        for (int i = 0; i < 6; ++i) eps_limit[i] = 1.0e30;
    }
};

/// Energy failure parameters
struct EnergyFailureParams {
    Real W_crit;          ///< Critical strain energy density (J/m^3)

    EnergyFailureParams() : W_crit(1.0e8) {}
};

/// Wierzbicki (Modified Mohr-Coulomb) parameters
struct WierzbickiFailureParams {
    Real C1;              ///< Scaling constant
    Real C2;              ///< Lode-angle sensitivity
    Real C_theta_s;       ///< Lode angle parameter for shear (axisymmetric compression)
    Real C_theta_ax;      ///< Lode angle parameter for axisymmetric tension
    Real sigma_bar_0;     ///< Reference flow stress
    Real c_friction;      ///< Internal friction coefficient (Mohr-Coulomb)

    WierzbickiFailureParams()
        : C1(0.5), C2(600.0e6), C_theta_s(1.0), C_theta_ax(1.0)
        , sigma_bar_0(500.0e6), c_friction(0.1) {}
};

/// Fabric (biaxial) failure parameters
struct FabricFailureParams {
    Real eps_1f;          ///< Warp direction failure strain
    Real eps_2f;          ///< Weft direction failure strain
    Real gamma_12f;       ///< In-plane shear failure strain

    FabricFailureParams()
        : eps_1f(0.05), eps_2f(0.05), gamma_12f(0.10) {}
};


// ============================================================================
// 1. Johnson-Cook Failure
// ============================================================================

/**
 * @brief Johnson-Cook cumulative-damage failure model
 *
 * Damage accumulation:
 *   D = Sum(Delta_eps_p / eps_f)
 *
 * Where fracture strain:
 *   eps_f = [d1 + d2*exp(d3*eta)] * [1 + d4*ln(eps_dot*)] * [1 + d5*T*]
 *
 *   eta = sigma_m / sigma_eq  (stress triaxiality)
 *   eps_dot* = eps_dot / eps_dot_ref
 *   T* = (T - T_room) / (T_melt - T_room)
 *
 * History:
 *   [0] = accumulated damage D
 *   [1] = maximum triaxiality encountered
 *   [2] = previous plastic strain (for increment)
 *
 * Reference: Johnson & Cook (1985), Eng. Fract. Mech. 21(1)
 */
class JohnsonCookFailure : public FailureModel {
public:
    JohnsonCookFailure(const FailureModelParameters& base_params,
                       const JohnsonCookFailureParams& jc_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)  // reuse enum slot
        , jc_(jc_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // --- Triaxiality ---
        Real sigma_m = (mstate.stress[0] + mstate.stress[1] + mstate.stress[2]) / 3.0;
        Real sigma_eq = Material::von_mises_stress(mstate.stress);
        Real eta = (sigma_eq > 1.0e-20) ? sigma_m / sigma_eq : 0.0;

        // Track max triaxiality
        if (eta > fstate.history[1]) fstate.history[1] = eta;

        // --- Fracture strain ---
        // Triaxiality term
        Real term_triax = jc_.d1 + jc_.d2 * Kokkos::exp(jc_.d3 * eta);
        if (term_triax < 1.0e-10) term_triax = 1.0e-10;

        // Strain-rate term
        Real eps_dot = mstate.effective_strain_rate;
        Real eps_dot_star = eps_dot / jc_.eps_dot_ref;
        Real term_rate = 1.0;
        if (eps_dot_star > 1.0) {
            term_rate = 1.0 + jc_.d4 * Kokkos::log(eps_dot_star);
        }
        if (term_rate < 1.0e-10) term_rate = 1.0e-10;

        // Temperature term
        Real T = mstate.temperature;
        Real T_star = 0.0;
        if (T > jc_.T_room && jc_.T_melt > jc_.T_room) {
            T_star = (T - jc_.T_room) / (jc_.T_melt - jc_.T_room);
            if (T_star > 1.0) T_star = 1.0;
        }
        Real term_temp = 1.0 + jc_.d5 * T_star;
        if (term_temp < 1.0e-10) term_temp = 1.0e-10;

        Real eps_f = term_triax * term_rate * term_temp;
        if (eps_f < 1.0e-10) eps_f = 1.0e-10;

        // --- Plastic strain increment ---
        Real eps_p = mstate.plastic_strain;
        Real eps_p_old = fstate.history[2];
        Real delta_eps_p = eps_p - eps_p_old;
        if (delta_eps_p < 0.0) delta_eps_p = 0.0;
        fstate.history[2] = eps_p;

        // --- Damage increment ---
        Real dD = delta_eps_p / eps_f;
        fstate.history[0] += dD;

        fstate.damage = fstate.history[0];
        if (fstate.damage > 1.0) fstate.damage = 1.0;

        if (fstate.history[0] >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const JohnsonCookFailureParams& jc) {
        return std::make_unique<JohnsonCookFailure>(base, jc);
    }

private:
    JohnsonCookFailureParams jc_;
};


// ============================================================================
// 2. Cockcroft-Latham Failure
// ============================================================================

/**
 * @brief Cockcroft-Latham failure criterion
 *
 * Plastic work weighted by maximum principal stress:
 *   W = Integral[ max(sigma_1, 0) * d_eps_p ]
 *
 * Failed when W >= W_crit.
 *
 * sigma_1 approximated as max(sigma_xx, sigma_yy, sigma_zz) for efficiency.
 *
 * History:
 *   [0] = accumulated weighted plastic work W
 *   [1] = previous plastic strain
 *
 * Reference: Cockcroft & Latham (1968), J. Inst. Metals 96
 */
class CockcroftLathamFailure : public FailureModel {
public:
    CockcroftLathamFailure(const FailureModelParameters& base_params,
                           const CockcroftLathamParams& cl_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)
        , cl_(cl_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // Approximate maximum principal stress as max of normal components
        Real sigma1 = mstate.stress[0];
        if (mstate.stress[1] > sigma1) sigma1 = mstate.stress[1];
        if (mstate.stress[2] > sigma1) sigma1 = mstate.stress[2];

        // Only positive (tensile) principal stress contributes
        Real sigma1_pos = (sigma1 > 0.0) ? sigma1 : 0.0;

        // Plastic strain increment
        Real eps_p = mstate.plastic_strain;
        Real eps_p_old = fstate.history[1];
        Real delta_eps_p = eps_p - eps_p_old;
        if (delta_eps_p < 0.0) delta_eps_p = 0.0;
        fstate.history[1] = eps_p;

        // Accumulate weighted plastic work
        fstate.history[0] += sigma1_pos * delta_eps_p;

        // Damage as ratio to critical value
        fstate.damage = fstate.history[0] / cl_.W_crit;
        if (fstate.damage > 1.0) fstate.damage = 1.0;

        if (fstate.history[0] >= cl_.W_crit) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const CockcroftLathamParams& cl) {
        return std::make_unique<CockcroftLathamFailure>(base, cl);
    }

private:
    CockcroftLathamParams cl_;
};


// ============================================================================
// 3. Lemaitre CDM Failure
// ============================================================================

/**
 * @brief Lemaitre continuum damage mechanics (CDM) model
 *
 * Damage evolution coupled to plastic strain:
 *   dD/d_eps_p = (Y / S)^s / (1 - D)
 *
 * Where the strain energy release rate:
 *   Y = sigma_eq^2 / (2 * E * (1 - D)^2)
 *
 * Effective stress degradation:
 *   sigma_eff = sigma / (1 - D)
 *
 * History:
 *   [0] = damage D
 *   [1] = previous plastic strain
 *
 * Reference: Lemaitre (1985), J. Eng. Mat. Tech.
 */
class LemaitreCDMFailure : public FailureModel {
public:
    LemaitreCDMFailure(const FailureModelParameters& base_params,
                       const LemaitreCDMParams& lem_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)
        , lem_(lem_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real D = fstate.history[0];
        if (D >= lem_.D_crit) {
            fstate.damage = 1.0;
            fstate.failed = true;
            fstate.failure_mode = 1;
            return;
        }

        // Plastic strain increment
        Real eps_p = mstate.plastic_strain;
        Real eps_p_old = fstate.history[1];
        Real delta_eps_p = eps_p - eps_p_old;
        if (delta_eps_p < 0.0) delta_eps_p = 0.0;
        fstate.history[1] = eps_p;

        // Only accumulate damage above threshold strain
        if (eps_p < lem_.eps_D) {
            fstate.damage = D;
            return;
        }

        // Von Mises equivalent stress
        Real sigma_eq = Material::von_mises_stress(mstate.stress);

        // Strain energy release rate Y
        Real one_minus_D = 1.0 - D;
        if (one_minus_D < 1.0e-10) one_minus_D = 1.0e-10;
        Real Y = (sigma_eq * sigma_eq) / (2.0 * lem_.E * one_minus_D * one_minus_D);

        // Damage increment: dD = (Y/S)^s / (1-D) * d_eps_p
        Real Y_over_S = Y / lem_.S;
        if (Y_over_S < 0.0) Y_over_S = 0.0;

        Real dD = Kokkos::pow(Y_over_S, lem_.s_exp) / one_minus_D * delta_eps_p;

        D += dD;
        if (D > 1.0) D = 1.0;
        fstate.history[0] = D;

        fstate.damage = D;
        if (D >= lem_.D_crit) {
            fstate.failed = true;
            fstate.failure_mode = 1;
            fstate.damage = 1.0;
        }
    }

    void degrade_stress(Real* sigma, const FailureState& fstate) const override {
        // Effective stress concept: sigma_eff = sigma / (1 - D)
        // For output, we reduce the nominal stress by (1 - D)
        Real D = fstate.damage;
        if (D > 0.0 && D < 1.0) {
            Real factor = 1.0 - D;
            for (int i = 0; i < 6; ++i) sigma[i] *= factor;
        } else if (D >= 1.0) {
            for (int i = 0; i < 6; ++i) sigma[i] = 0.0;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const LemaitreCDMParams& lem) {
        return std::make_unique<LemaitreCDMFailure>(base, lem);
    }

private:
    LemaitreCDMParams lem_;
};


// ============================================================================
// 4. Puck Failure
// ============================================================================

/**
 * @brief Puck failure criterion for fiber-reinforced composites
 *
 * Evaluates two classes of failure:
 *
 * Fiber Failure (FF):
 *   Tension:    eps_1 / eps_1t >= 1
 *   Compression: |sigma_1| / R_||c >= 1
 *
 * Inter-Fiber Failure (IFF):
 *   (sigma_n / R_perp)^2 + (tau_nt / R_perp_||)^2 + (tau_n1 / R_perp_||)^2 >= 1
 *
 * For the IFF criterion, sigma_n is identified with sigma_22 (transverse normal),
 * tau_nt with tau_23, and tau_n1 with tau_12 in lamina coordinates.
 *
 * History:
 *   [0] = FF tension failure index
 *   [1] = FF compression failure index
 *   [2] = IFF failure index
 *   [3] = (reserved)
 *   [4] = FF tension damage flag
 *   [5] = FF compression damage flag
 *   [6] = IFF damage flag
 *   [7] = (reserved)
 *
 * Reference: Puck & Schurmann (2002), Comp. Sci. Tech. 62
 */
class PuckFailure : public FailureModel {
public:
    PuckFailure(const FailureModelParameters& base_params,
                const PuckFailureParams& puck_params)
        : FailureModel(FailureModelType::Hashin, base_params)  // composite-family
        , pk_(puck_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        const Real s11 = mstate.stress[0];  // Fiber direction
        const Real s22 = mstate.stress[1];  // Transverse normal
        const Real t12 = mstate.stress[3];  // In-plane shear
        const Real t23 = mstate.stress[4];  // Transverse shear

        const Real e11 = mstate.strain[0];  // Fiber strain

        // --- Fiber failure (FF) ---
        // Tension: strain-based
        Real ff_t = 0.0;
        if (e11 > 0.0 && pk_.eps_1t > 0.0) {
            ff_t = e11 / pk_.eps_1t;
        }
        fstate.history[0] = ff_t;

        // Compression: stress-based
        Real ff_c = 0.0;
        if (s11 < 0.0 && pk_.R_para_c > 0.0) {
            ff_c = Kokkos::fabs(s11) / pk_.R_para_c;
        }
        fstate.history[1] = ff_c;

        // --- Inter-fiber failure (IFF) ---
        // Using action-plane representation; sigma_n = s22, tau_nt = t23, tau_n1 = t12
        Real iff = 0.0;
        {
            Real R_perp = (s22 >= 0.0) ? pk_.R_perp_t : pk_.R_perp_c;
            if (R_perp < 1.0e-20) R_perp = 1.0e-20;
            Real R_pp = pk_.R_perp_para;
            if (R_pp < 1.0e-20) R_pp = 1.0e-20;

            Real term1 = (s22 / R_perp) * (s22 / R_perp);
            Real term2 = (t23 / R_pp) * (t23 / R_pp);
            Real term3 = (t12 / R_pp) * (t12 / R_pp);
            iff = term1 + term2 + term3;
        }
        fstate.history[2] = iff;

        // --- Update damage flags (irreversible) ---
        if (ff_t >= 1.0) fstate.history[4] = 1.0;
        if (ff_c >= 1.0) fstate.history[5] = 1.0;
        if (iff >= 1.0)  fstate.history[6] = 1.0;

        // Overall damage
        Real max_damage = 0.0;
        int fail_mode = 0;
        if (fstate.history[4] >= 1.0) { max_damage = 1.0; fail_mode = 1; }  // FF tension
        if (fstate.history[5] >= 1.0) { max_damage = 1.0; fail_mode = 2; }  // FF compression
        if (fstate.history[6] >= 1.0) { max_damage = 1.0; fail_mode = 3; }  // IFF

        fstate.damage = max_damage;
        if (max_damage >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = fail_mode;
        }
    }

    void degrade_stress(Real* sigma, const FailureState& fstate) const override {
        // Fiber failure: degrade all fiber-direction stresses and shear
        if (fstate.history[4] >= 1.0 || fstate.history[5] >= 1.0) {
            sigma[0] = 0.0;  // sigma_11
            sigma[3] = 0.0;  // tau_12
            sigma[5] = 0.0;  // tau_13
        }
        // Inter-fiber failure: degrade transverse and shear
        if (fstate.history[6] >= 1.0) {
            sigma[1] = 0.0;  // sigma_22
            sigma[3] = 0.0;  // tau_12
            sigma[4] = 0.0;  // tau_23
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const PuckFailureParams& pk) {
        return std::make_unique<PuckFailure>(base, pk);
    }

private:
    PuckFailureParams pk_;
};


// ============================================================================
// 5. FLD Failure (Forming Limit Diagram)
// ============================================================================

/**
 * @brief Forming Limit Diagram (FLD) failure criterion
 *
 * Compares in-plane principal strains (eps_major, eps_minor) against a
 * user-supplied FLD curve. The curve maps minor strain (x) to the
 * limiting major strain (y). If the current major strain exceeds the
 * limit for the current minor strain, failure occurs.
 *
 * The principal strains are computed from the in-plane strain components
 * (eps_xx, eps_yy, gamma_xy).
 *
 * History:
 *   [0] = proximity ratio to FLD (current_major / fld_limit)
 *   [1] = current major principal strain
 *   [2] = current minor principal strain
 *
 * Reference: Keeler & Backofen (1963), ASM Trans. Quarterly
 */
class FLDFailure : public FailureModel {
public:
    FLDFailure(const FailureModelParameters& base_params,
               const FLDFailureParams& fld_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)
        , fld_(fld_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // In-plane strain components
        Real exx = mstate.strain[0];
        Real eyy = mstate.strain[1];
        Real gxy = mstate.strain[3];  // Engineering shear strain

        // Principal strains (2D Mohr's circle)
        Real e_avg = 0.5 * (exx + eyy);
        Real e_diff = 0.5 * (exx - eyy);
        Real R = Kokkos::sqrt(e_diff * e_diff + 0.25 * gxy * gxy);

        Real e1 = e_avg + R;  // Major principal strain
        Real e2 = e_avg - R;  // Minor principal strain

        fstate.history[1] = e1;
        fstate.history[2] = e2;

        // Evaluate FLD curve: limiting major strain at current minor strain
        Real e_major_limit = 1.0e30;  // Default: no limit
        if (fld_.fld_curve.num_points > 0) {
            e_major_limit = fld_.fld_curve.evaluate(e2);
        }

        // Proximity ratio
        Real ratio = 0.0;
        if (e_major_limit > 1.0e-20) {
            ratio = e1 / e_major_limit;
        }
        fstate.history[0] = ratio;

        fstate.damage = (ratio > 0.0) ? ratio : 0.0;
        if (fstate.damage > 1.0) fstate.damage = 1.0;

        if (ratio >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const FLDFailureParams& fld) {
        return std::make_unique<FLDFailure>(base, fld);
    }

private:
    FLDFailureParams fld_;
};


// ============================================================================
// 6. Wilkins Failure
// ============================================================================

/**
 * @brief Wilkins cumulative-damage failure model for ductile metals
 *
 * Integral damage measure:
 *   f = Integral[ max(0, sigma_1)^a * (2 - A)^b * d_eps_p ]
 *
 * Where A = min(sigma_2/sigma_1, sigma_3/sigma_1) is an asymmetry measure.
 * Principal stresses are approximated from normal stress components.
 *
 * Failed when f >= f_crit.
 *
 * History:
 *   [0] = accumulated integral f
 *   [1] = previous plastic strain
 *
 * Reference: Wilkins et al. (1980), "Cumulative-strain-damage model of
 *            ductile fracture", UCRL-53058
 */
class WilkinsFailure : public FailureModel {
public:
    WilkinsFailure(const FailureModelParameters& base_params,
                   const WilkinsFailureParams& wk_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)
        , wk_(wk_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // Approximate principal stresses by sorting normal components
        Real s[3] = { mstate.stress[0], mstate.stress[1], mstate.stress[2] };

        // Simple sort descending (sigma_1 >= sigma_2 >= sigma_3)
        if (s[0] < s[1]) { Real t = s[0]; s[0] = s[1]; s[1] = t; }
        if (s[1] < s[2]) { Real t = s[1]; s[1] = s[2]; s[2] = t; }
        if (s[0] < s[1]) { Real t = s[0]; s[0] = s[1]; s[1] = t; }

        Real sigma_1 = s[0];
        Real sigma_2 = s[1];
        Real sigma_3 = s[2];

        // Max principal stress contribution (tensile only)
        Real sigma1_pos = (sigma_1 > 0.0) ? sigma_1 : 0.0;
        Real sigma1_term = Kokkos::pow(sigma1_pos, wk_.a_exp);

        // Asymmetry factor A = min(sigma_2/sigma_1, sigma_3/sigma_1)
        Real A = 0.0;
        if (Kokkos::fabs(sigma_1) > 1.0e-20) {
            Real r2 = sigma_2 / sigma_1;
            Real r3 = sigma_3 / sigma_1;
            A = (r2 < r3) ? r2 : r3;
        }
        // Clamp A to prevent negative (2-A) for extreme states
        if (A > 2.0) A = 2.0;

        Real asym_term = 2.0 - A;
        if (asym_term < 0.0) asym_term = 0.0;
        Real asym_factor = Kokkos::pow(asym_term, wk_.b_exp);

        // Plastic strain increment
        Real eps_p = mstate.plastic_strain;
        Real eps_p_old = fstate.history[1];
        Real delta_eps_p = eps_p - eps_p_old;
        if (delta_eps_p < 0.0) delta_eps_p = 0.0;
        fstate.history[1] = eps_p;

        // Accumulate damage integral
        fstate.history[0] += sigma1_term * asym_factor * delta_eps_p;

        fstate.damage = fstate.history[0] / wk_.f_crit;
        if (fstate.damage > 1.0) fstate.damage = 1.0;

        if (fstate.history[0] >= wk_.f_crit) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const WilkinsFailureParams& wk) {
        return std::make_unique<WilkinsFailure>(base, wk);
    }

private:
    WilkinsFailureParams wk_;
};


// ============================================================================
// 7. Tuler-Butcher Failure (Spall)
// ============================================================================

/**
 * @brief Tuler-Butcher dynamic spall failure criterion
 *
 * Impulse integral for dynamic tensile (spall) failure:
 *   K = Integral[ max(0, sigma_1 - sigma_spall)^lambda * dt ]
 *
 * Failed when K >= K_crit.
 *
 * sigma_1 approximated as max(sigma_xx, sigma_yy, sigma_zz).
 *
 * History:
 *   [0] = accumulated impulse K
 *
 * Reference: Tuler & Butcher (1968), Int. J. Fract. Mech.
 */
class TulerButcherFailure : public FailureModel {
public:
    TulerButcherFailure(const FailureModelParameters& base_params,
                        const TulerButcherParams& tb_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)
        , tb_(tb_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real dt,
                        Real /*element_size*/) const override {
        // Approximate maximum principal stress
        Real sigma1 = mstate.stress[0];
        if (mstate.stress[1] > sigma1) sigma1 = mstate.stress[1];
        if (mstate.stress[2] > sigma1) sigma1 = mstate.stress[2];

        // Overstress above spall threshold
        Real overstress = sigma1 - tb_.sigma_spall;
        if (overstress > 0.0) {
            Real increment = Kokkos::pow(overstress, tb_.lambda) * dt;
            fstate.history[0] += increment;
        }

        fstate.damage = fstate.history[0] / tb_.K_crit;
        if (fstate.damage > 1.0) fstate.damage = 1.0;

        if (fstate.history[0] >= tb_.K_crit) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const TulerButcherParams& tb) {
        return std::make_unique<TulerButcherFailure>(base, tb);
    }

private:
    TulerButcherParams tb_;
};


// ============================================================================
// 8. Maximum Stress Failure
// ============================================================================

/**
 * @brief Maximum stress failure criterion
 *
 * Element fails if |sigma_ij| exceeds the limit for any Voigt component.
 * Six independent limits allow anisotropic strength definitions.
 *
 * History:
 *   [0..5] = current stress ratios |sigma_i| / limit_i
 *
 * Reference: Classic maximum stress theory
 */
class MaxStressFailure : public FailureModel {
public:
    MaxStressFailure(const FailureModelParameters& base_params,
                     const MaxStressFailureParams& ms_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)
        , ms_(ms_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real max_ratio = 0.0;
        int fail_comp = -1;

        for (int i = 0; i < 6; ++i) {
            Real limit = ms_.sigma_limit[i];
            if (limit < 1.0e-20) limit = 1.0e-20;

            Real ratio = Kokkos::fabs(mstate.stress[i]) / limit;
            fstate.history[i] = ratio;

            if (ratio > max_ratio) {
                max_ratio = ratio;
                fail_comp = i;
            }
        }

        fstate.damage = (max_ratio > 1.0) ? 1.0 : max_ratio;

        if (max_ratio >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = fail_comp + 1;  // 1-indexed component
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const MaxStressFailureParams& ms) {
        return std::make_unique<MaxStressFailure>(base, ms);
    }

private:
    MaxStressFailureParams ms_;
};


// ============================================================================
// 9. Maximum Strain Failure
// ============================================================================

/**
 * @brief Maximum strain failure criterion
 *
 * Element fails if |epsilon_ij| exceeds the limit for any Voigt component.
 * Six independent limits allow anisotropic failure definitions.
 *
 * History:
 *   [0..5] = current strain ratios |eps_i| / limit_i
 *
 * Reference: Classic maximum strain theory
 */
class MaxStrainFailure : public FailureModel {
public:
    MaxStrainFailure(const FailureModelParameters& base_params,
                     const MaxStrainFailureParams& ms_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)
        , ms_(ms_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real max_ratio = 0.0;
        int fail_comp = -1;

        for (int i = 0; i < 6; ++i) {
            Real limit = ms_.eps_limit[i];
            if (limit < 1.0e-20) limit = 1.0e-20;

            Real ratio = Kokkos::fabs(mstate.strain[i]) / limit;
            fstate.history[i] = ratio;

            if (ratio > max_ratio) {
                max_ratio = ratio;
                fail_comp = i;
            }
        }

        fstate.damage = (max_ratio > 1.0) ? 1.0 : max_ratio;

        if (max_ratio >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = fail_comp + 1;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const MaxStrainFailureParams& ms) {
        return std::make_unique<MaxStrainFailure>(base, ms);
    }

private:
    MaxStrainFailureParams ms_;
};


// ============================================================================
// 10. Energy Failure
// ============================================================================

/**
 * @brief Strain energy density failure criterion
 *
 * Computes total strain energy density:
 *   W = 0.5 * sigma : epsilon
 *     = 0.5 * (sigma_xx*eps_xx + sigma_yy*eps_yy + sigma_zz*eps_zz
 *              + tau_xy*gamma_xy + tau_yz*gamma_yz + tau_xz*gamma_xz)
 *
 * Failed when W > W_crit.
 *
 * History:
 *   [0] = maximum strain energy density encountered (W_max)
 *
 * Reference: Beltrami (1885) strain energy criterion
 */
class EnergyFailure : public FailureModel {
public:
    EnergyFailure(const FailureModelParameters& base_params,
                  const EnergyFailureParams& en_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)
        , en_(en_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // Strain energy density: W = 0.5 * sigma : epsilon
        // Voigt contraction: normal terms + 2*shear terms (engineering strain)
        Real W = 0.5 * (mstate.stress[0] * mstate.strain[0]
                       + mstate.stress[1] * mstate.strain[1]
                       + mstate.stress[2] * mstate.strain[2]
                       + mstate.stress[3] * mstate.strain[3]
                       + mstate.stress[4] * mstate.strain[4]
                       + mstate.stress[5] * mstate.strain[5]);

        // Track maximum
        if (W > fstate.history[0]) {
            fstate.history[0] = W;
        }

        fstate.damage = fstate.history[0] / en_.W_crit;
        if (fstate.damage > 1.0) fstate.damage = 1.0;

        if (fstate.history[0] >= en_.W_crit) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const EnergyFailureParams& en) {
        return std::make_unique<EnergyFailure>(base, en);
    }

private:
    EnergyFailureParams en_;
};


// ============================================================================
// 11. Wierzbicki (Modified Mohr-Coulomb) Failure
// ============================================================================

/**
 * @brief Wierzbicki modified Mohr-Coulomb failure model
 *
 * Fracture strain as function of triaxiality eta and Lode angle parameter theta_bar:
 *
 *   eps_f(eta, theta_bar) = { C2 / sigma_bar_0 *
 *       [ C_theta * (1/sqrt(3)) * (1 + c_friction * I1/3) *
 *         sec(theta_bar * pi/6) - c_friction * (eta + 1/3) ] }^(-1/C1)
 *
 * Where:
 *   C_theta = C_theta_ax + (C_theta_s - C_theta_ax) * f(theta_bar)
 *   f(theta_bar) normalizes the Lode angle parameter (0=axisymmetric, 1=plane strain)
 *
 * Simplified implementation using incremental damage accumulation:
 *   D += delta_eps_p / eps_f(eta, theta_bar)
 *
 * History:
 *   [0] = accumulated damage D
 *   [1] = current triaxiality
 *   [2] = current Lode angle parameter
 *   [3] = previous plastic strain
 *
 * Reference: Bai & Wierzbicki (2010), Int. J. Fract. 161
 */
class WierzbickiFailure : public FailureModel {
public:
    WierzbickiFailure(const FailureModelParameters& base_params,
                      const WierzbickiFailureParams& wb_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)
        , wb_(wb_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // --- Stress invariants ---
        Real sigma_m = (mstate.stress[0] + mstate.stress[1] + mstate.stress[2]) / 3.0;
        Real sigma_eq = Material::von_mises_stress(mstate.stress);

        // Triaxiality
        Real eta = (sigma_eq > 1.0e-20) ? sigma_m / sigma_eq : 0.0;
        fstate.history[1] = eta;

        // Deviatoric stress for J3 (Lode angle)
        Real s[6];
        s[0] = mstate.stress[0] - sigma_m;
        s[1] = mstate.stress[1] - sigma_m;
        s[2] = mstate.stress[2] - sigma_m;
        s[3] = mstate.stress[3];
        s[4] = mstate.stress[4];
        s[5] = mstate.stress[5];

        // J2 = 0.5 * s:s
        Real J2 = 0.5 * (s[0]*s[0] + s[1]*s[1] + s[2]*s[2])
                 + s[3]*s[3] + s[4]*s[4] + s[5]*s[5];

        // J3 = det(s)  (deviatoric stress tensor determinant)
        // s is symmetric: s11, s22, s33, s12, s23, s13
        Real J3 = s[0] * (s[1]*s[2] - s[4]*s[4])
                 - s[3] * (s[3]*s[2] - s[4]*s[5])
                 + s[5] * (s[3]*s[4] - s[1]*s[5]);

        // Lode angle parameter: theta_bar = 1 - (2/pi)*acos(xi)
        // where xi = (27/2) * J3 / sigma_eq^3
        Real theta_bar = 0.0;
        if (sigma_eq > 1.0e-20) {
            Real xi = 13.5 * J3 / (sigma_eq * sigma_eq * sigma_eq);
            // Clamp xi to [-1, 1]
            if (xi > 1.0) xi = 1.0;
            if (xi < -1.0) xi = -1.0;
            theta_bar = 1.0 - (2.0 / 3.14159265358979323846) * Kokkos::acos(xi);
        }
        fstate.history[2] = theta_bar;

        // --- Lode-angle-dependent parameter C_theta ---
        // Interpolate between axisymmetric tension (theta_bar=1) and shear (theta_bar=0)
        Real gamma_theta = 0.0;
        if (Kokkos::fabs(theta_bar) < 1.0e-10) {
            gamma_theta = 1.0;  // Pure shear
        } else {
            // Weight: gamma = (1 - theta_bar) for linear interpolation
            Real abs_tb = Kokkos::fabs(theta_bar);
            gamma_theta = 1.0 - abs_tb;
        }
        Real C_theta = wb_.C_theta_ax + (wb_.C_theta_s - wb_.C_theta_ax) * gamma_theta;

        // --- Fracture strain eps_f ---
        // Modified Mohr-Coulomb formulation:
        // eps_f^(-C1) = (C2/sigma_bar_0) * [ C_theta/sqrt(3) * cos(theta_bar*pi/6)^(-1)
        //               * (1 + c*sin(theta_bar*pi/6)) + c*(eta + 1/3) ]
        //
        // Simplified: use secant form
        Real pi_over_6 = 3.14159265358979323846 / 6.0;
        Real tb_angle = theta_bar * pi_over_6;
        Real cos_tb = Kokkos::cos(tb_angle);
        if (Kokkos::fabs(cos_tb) < 1.0e-10) cos_tb = 1.0e-10;
        Real sec_tb = 1.0 / cos_tb;
        Real sin_tb = Kokkos::sin(tb_angle);

        Real bracket = (C_theta / 1.7320508075688772) * sec_tb
                      * (1.0 + wb_.c_friction * sin_tb)
                      + wb_.c_friction * (eta + 1.0/3.0);

        // Guard against zero or negative bracket
        if (bracket < 1.0e-20) bracket = 1.0e-20;

        Real ratio = (wb_.C2 / wb_.sigma_bar_0) * bracket;
        if (ratio < 1.0e-20) ratio = 1.0e-20;

        // eps_f = ratio^(-1/C1)
        Real eps_f = Kokkos::pow(ratio, -1.0 / wb_.C1);
        if (eps_f < 1.0e-10) eps_f = 1.0e-10;

        // --- Plastic strain increment ---
        Real eps_p = mstate.plastic_strain;
        Real eps_p_old = fstate.history[3];
        Real delta_eps_p = eps_p - eps_p_old;
        if (delta_eps_p < 0.0) delta_eps_p = 0.0;
        fstate.history[3] = eps_p;

        // --- Damage accumulation ---
        Real dD = delta_eps_p / eps_f;
        fstate.history[0] += dD;

        fstate.damage = fstate.history[0];
        if (fstate.damage > 1.0) fstate.damage = 1.0;

        if (fstate.history[0] >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const WierzbickiFailureParams& wb) {
        return std::make_unique<WierzbickiFailure>(base, wb);
    }

private:
    WierzbickiFailureParams wb_;
};


// ============================================================================
// 12. Fabric Failure (Biaxial)
// ============================================================================

/**
 * @brief Biaxial failure criterion for woven fabrics
 *
 * Interactive quadratic criterion in principal material directions:
 *   FI = (eps_1 / eps_1f)^2 + (eps_2 / eps_2f)^2 + (gamma_12 / gamma_12f)^2
 *
 * Failed when FI >= 1.
 *
 * eps_1 = warp direction strain, eps_2 = weft direction strain,
 * gamma_12 = in-plane shear strain.
 *
 * Different failure strains for warp and weft allow asymmetric weave behavior.
 *
 * History:
 *   [0] = failure index FI
 *   [1] = warp strain ratio
 *   [2] = weft strain ratio
 *   [3] = shear strain ratio
 *
 * Reference: Gasser et al. (2000), Comp. Meth. Appl. Mech. Eng.
 */
class FabricFailure : public FailureModel {
public:
    FabricFailure(const FailureModelParameters& base_params,
                  const FabricFailureParams& fab_params)
        : FailureModel(FailureModelType::Hashin, base_params)  // composite-family
        , fab_(fab_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // Strains in material coordinates
        Real e1 = mstate.strain[0];       // Warp direction
        Real e2 = mstate.strain[1];       // Weft direction
        Real g12 = mstate.strain[3];      // In-plane shear (engineering)

        // Failure strain denominators (guard against zero)
        Real e1f = fab_.eps_1f;
        Real e2f = fab_.eps_2f;
        Real g12f = fab_.gamma_12f;
        if (e1f < 1.0e-20) e1f = 1.0e-20;
        if (e2f < 1.0e-20) e2f = 1.0e-20;
        if (g12f < 1.0e-20) g12f = 1.0e-20;

        // Component ratios
        Real r1 = e1 / e1f;
        Real r2 = e2 / e2f;
        Real r12 = g12 / g12f;

        fstate.history[1] = r1;
        fstate.history[2] = r2;
        fstate.history[3] = r12;

        // Quadratic failure index
        Real FI = r1*r1 + r2*r2 + r12*r12;
        fstate.history[0] = FI;

        fstate.damage = (FI > 0.0) ? Kokkos::fmin(FI, 1.0) : 0.0;

        if (FI >= 1.0) {
            fstate.failed = true;

            // Identify dominant mode
            Real abs_r1 = Kokkos::fabs(r1);
            Real abs_r2 = Kokkos::fabs(r2);
            Real abs_r12 = Kokkos::fabs(r12);
            if (abs_r1 >= abs_r2 && abs_r1 >= abs_r12) {
                fstate.failure_mode = 1;  // Warp failure
            } else if (abs_r2 >= abs_r12) {
                fstate.failure_mode = 2;  // Weft failure
            } else {
                fstate.failure_mode = 3;  // Shear failure
            }
        }
    }

    void degrade_stress(Real* sigma, const FailureState& fstate) const override {
        if (!fstate.failed) return;

        // Mode-specific degradation
        switch (fstate.failure_mode) {
            case 1:  // Warp failure: degrade warp and shear
                sigma[0] = 0.0;
                sigma[3] = 0.0;
                break;
            case 2:  // Weft failure: degrade weft and shear
                sigma[1] = 0.0;
                sigma[3] = 0.0;
                break;
            case 3:  // Shear failure: degrade shear only
                sigma[3] = 0.0;
                break;
            default: // Total failure
                for (int i = 0; i < 6; ++i) sigma[i] = 0.0;
                break;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const FabricFailureParams& fab) {
        return std::make_unique<FabricFailure>(base, fab);
    }

private:
    FabricFailureParams fab_;
};


// ============================================================================
// Utility: to_string for extended failure model types
// ============================================================================

inline const char* to_string(FailureModelTypeExt type) {
    switch (type) {
        case FailureModelTypeExt::JohnsonCook:     return "JohnsonCook";
        case FailureModelTypeExt::CockcroftLatham:  return "CockcroftLatham";
        case FailureModelTypeExt::LemaitreCDM:      return "LemaitreCDM";
        case FailureModelTypeExt::Puck:             return "Puck";
        case FailureModelTypeExt::FLD:              return "FLD";
        case FailureModelTypeExt::Wilkins:          return "Wilkins";
        case FailureModelTypeExt::TulerButcher:     return "TulerButcher";
        case FailureModelTypeExt::MaxStress:        return "MaxStress";
        case FailureModelTypeExt::MaxStrain:        return "MaxStrain";
        case FailureModelTypeExt::Energy:           return "Energy";
        case FailureModelTypeExt::Wierzbicki:       return "Wierzbicki";
        case FailureModelTypeExt::Fabric:           return "Fabric";
        default:                                    return "Unknown";
    }
}

} // namespace failure
} // namespace physics
} // namespace nxs
