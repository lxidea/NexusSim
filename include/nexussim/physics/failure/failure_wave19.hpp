#pragma once

/**
 * @file failure_wave19.hpp
 * @brief 10 additional failure/damage models for composites, hyperelastics, and specialty
 *
 * Models implemented:
 *   1.  LaDevezeDelamination       - Composite interlaminar delamination (quadratic)
 *   2.  HoffmanFailure             - Anisotropic tension/compression failure
 *   3.  TsaiHillFailure            - Simplified interactive composite failure
 *   4.  RTClFailure                - Reinforced thermoplastic composite failure
 *   5.  MullinsEffect              - Hyperelastic softening/damage (Ogden-Roxburgh)
 *   6.  SpallingFailure            - Dynamic tensile spall fracture
 *   7.  HCDSSEFailure              - High-cycle damage for structural steels
 *   8.  AdhesiveJointFailure       - Adhesive/connection quadratic traction failure
 *   9.  WindshieldFailure          - Laminate glass + PVB progressive failure
 *   10. GeneralizedEnergyFailure   - Generalized plastic work / energy criterion
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
// Wave 19 failure model type enumeration
// ============================================================================

enum class FailureModelTypeWave19 {
    LaDevezeDelamination,
    Hoffman,
    TsaiHill,
    RTCl,
    MullinsEffect,
    Spalling,
    HCDSSE,
    AdhesiveJoint,
    Windshield,
    GeneralizedEnergy
};

// ============================================================================
// Parameter Structures (per-model)
// ============================================================================

/// LaDeveze delamination parameters
struct LaDevezeDelaminationParams {
    Real Z_t;             ///< Through-thickness tensile strength (Pa)
    Real S_13;            ///< Interlaminar shear strength 1-3 (Pa)
    Real S_23;            ///< Interlaminar shear strength 2-3 (Pa)

    LaDevezeDelaminationParams()
        : Z_t(50.0e6), S_13(60.0e6), S_23(60.0e6) {}
};

/// Hoffman failure parameters
struct HoffmanFailureParams {
    Real Xt;              ///< Tensile strength, 1-direction (Pa)
    Real Xc;              ///< Compressive strength, 1-direction (Pa)
    Real Yt;              ///< Tensile strength, 2-direction (Pa)
    Real Yc;              ///< Compressive strength, 2-direction (Pa)
    Real Zt;              ///< Tensile strength, 3-direction (Pa)
    Real Zc;              ///< Compressive strength, 3-direction (Pa)
    Real S12;             ///< Shear strength 1-2 (Pa)
    Real S23;             ///< Shear strength 2-3 (Pa)
    Real S13;             ///< Shear strength 1-3 (Pa)

    HoffmanFailureParams()
        : Xt(1.5e9), Xc(1.2e9), Yt(50.0e6), Yc(200.0e6)
        , Zt(50.0e6), Zc(200.0e6), S12(75.0e6), S23(50.0e6), S13(75.0e6) {}
};

/// Tsai-Hill failure parameters
struct TsaiHillFailureParams {
    Real X;               ///< Strength in 1-direction (Pa)
    Real Y;               ///< Strength in 2-direction (Pa)
    Real S;               ///< In-plane shear strength (Pa)

    TsaiHillFailureParams()
        : X(1.5e9), Y(50.0e6), S(75.0e6) {}
};

/// Reinforced thermoplastic composite failure parameters
struct RTClFailureParams {
    Real X_ft;            ///< Fiber tensile strength (Pa)
    Real X_fc;            ///< Fiber compressive strength (Pa)
    Real Y_mt;            ///< Matrix tensile strength (Pa)
    Real Y_mc;            ///< Matrix compressive strength (Pa)
    Real S_12;            ///< In-plane shear strength (Pa)
    Real beta;            ///< Shear-transverse interaction coefficient

    RTClFailureParams()
        : X_ft(1.5e9), X_fc(1.2e9), Y_mt(80.0e6), Y_mc(200.0e6)
        , S_12(75.0e6), beta(1.0) {}
};

/// Mullins effect (Ogden-Roxburgh) parameters
struct MullinsEffectParams {
    Real r;               ///< Damage saturation parameter
    Real m;               ///< Damage evolution parameter (Pa)
    Real beta;            ///< Damage rate parameter

    MullinsEffectParams()
        : r(1.5), m(0.1e6), beta(0.1) {}
};

/// Spalling failure parameters
struct SpallingFailureParams {
    Real P_spall;         ///< Spall threshold (hydrostatic tension, Pa)
    Real D_crit;          ///< Critical cumulative damage for failure
    Real exponent;        ///< Overstress exponent

    SpallingFailureParams()
        : P_spall(1.0e9), D_crit(1.0), exponent(2.0) {}
};

/// HCDSSE (high-cycle damage structural steels) parameters
struct HCDSSEFailureParams {
    Real sigma_ref;       ///< Reference stress amplitude on Wohler curve (Pa)
    Real b_exp;           ///< Basquin exponent (negative slope of S-N curve)
    Real D_crit;          ///< Critical accumulated damage for failure
    Real sigma_endurance; ///< Endurance limit below which no damage (Pa)

    HCDSSEFailureParams()
        : sigma_ref(500.0e6), b_exp(5.0), D_crit(1.0), sigma_endurance(200.0e6) {}
};

/// Adhesive joint failure parameters
struct AdhesiveJointFailureParams {
    Real t_n;             ///< Normal traction strength (Pa)
    Real t_s;             ///< Shear traction strength (Pa)
    Real G_Ic;            ///< Mode I critical energy release rate (J/m^2)
    Real G_IIc;           ///< Mode II critical energy release rate (J/m^2)
    Real eta;             ///< Benzeggagh-Kenane exponent for mixed mode

    AdhesiveJointFailureParams()
        : t_n(30.0e6), t_s(50.0e6), G_Ic(300.0), G_IIc(600.0), eta(1.45) {}
};

/// Windshield laminate failure parameters
struct WindshieldFailureParams {
    Real glass_weibull_m;      ///< Weibull modulus for glass
    Real glass_sigma0;         ///< Weibull characteristic strength (Pa)
    Real glass_V0;             ///< Weibull reference volume (m^3)
    Real pvb_stretch_limit;    ///< PVB interlayer ultimate stretch ratio
    Real pvb_tear_energy;      ///< PVB tear energy (J/m^2)

    WindshieldFailureParams()
        : glass_weibull_m(7.0), glass_sigma0(70.0e6), glass_V0(1.0e-6)
        , pvb_stretch_limit(3.0), pvb_tear_energy(5000.0) {}
};

/// Generalized energy failure parameters
struct GeneralizedEnergyFailureParams {
    Real G_c;             ///< Critical fracture energy (J/m^2)

    GeneralizedEnergyFailureParams() : G_c(1.0e4) {}
};


// ============================================================================
// 1. LaDeveze Delamination
// ============================================================================

/**
 * @brief LaDeveze composite interlaminar delamination model
 *
 * Quadratic interaction criterion for delamination:
 *   FI = (sigma_33/Z_t)^2 + (tau_13/S_13)^2 + (tau_23/S_23)^2
 *
 * Delamination initiates when FI >= 1.
 * Only tensile through-thickness stress (sigma_33 > 0) contributes;
 * compressive sigma_33 suppresses delamination.
 *
 * History:
 *   [0] = maximum failure index encountered
 *   [1] = current sigma_33 contribution
 *   [2] = current tau_13 contribution
 *   [3] = current tau_23 contribution
 *
 * Reference: Ladeveze & Le Dantec (1992), Comp. Sci. Tech.
 */
class LaDevezeDelamination : public FailureModel {
public:
    LaDevezeDelamination(const FailureModelParameters& base_params,
                         const LaDevezeDelaminationParams& delam_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , delam_(delam_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // Through-thickness normal stress (sigma_33 = stress[2])
        Real sigma_33 = mstate.stress[2];

        // Interlaminar shear stresses
        // tau_13 = stress[5] (tau_xz in Voigt)
        // tau_23 = stress[4] (tau_yz in Voigt)
        Real tau_13 = mstate.stress[5];
        Real tau_23 = mstate.stress[4];

        // Guard against zero strengths
        Real Z_t = delam_.Z_t;
        Real S_13 = delam_.S_13;
        Real S_23 = delam_.S_23;
        if (Z_t < 1.0e-20) Z_t = 1.0e-20;
        if (S_13 < 1.0e-20) S_13 = 1.0e-20;
        if (S_23 < 1.0e-20) S_23 = 1.0e-20;

        // Only tensile sigma_33 contributes to delamination
        Real sigma_33_pos = (sigma_33 > 0.0) ? sigma_33 : 0.0;

        // Quadratic failure index
        Real term_n = (sigma_33_pos / Z_t) * (sigma_33_pos / Z_t);
        Real term_13 = (tau_13 / S_13) * (tau_13 / S_13);
        Real term_23 = (tau_23 / S_23) * (tau_23 / S_23);

        Real FI = term_n + term_13 + term_23;

        // Store component contributions
        fstate.history[1] = term_n;
        fstate.history[2] = term_13;
        fstate.history[3] = term_23;

        // Track maximum failure index (irreversible)
        if (FI > fstate.history[0]) {
            fstate.history[0] = FI;
        }

        fstate.damage = (fstate.history[0] > 1.0) ? 1.0 : fstate.history[0];

        if (fstate.history[0] >= 1.0) {
            fstate.failed = true;
            // Identify dominant mode
            if (term_n >= term_13 && term_n >= term_23) {
                fstate.failure_mode = 1;  // Mode I opening
            } else if (term_13 >= term_23) {
                fstate.failure_mode = 2;  // Mode II shear (1-3)
            } else {
                fstate.failure_mode = 3;  // Mode III shear (2-3)
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& state) const {
        Real sigma_33 = state.stress[2];
        Real tau_13 = state.stress[5];
        Real tau_23 = state.stress[4];

        Real sigma_33_pos = (sigma_33 > 0.0) ? sigma_33 : 0.0;

        Real Z_t = delam_.Z_t;
        Real S_13 = delam_.S_13;
        Real S_23 = delam_.S_23;
        if (Z_t < 1.0e-20) Z_t = 1.0e-20;
        if (S_13 < 1.0e-20) S_13 = 1.0e-20;
        if (S_23 < 1.0e-20) S_23 = 1.0e-20;

        Real FI = (sigma_33_pos / Z_t) * (sigma_33_pos / Z_t)
                + (tau_13 / S_13) * (tau_13 / S_13)
                + (tau_23 / S_23) * (tau_23 / S_23);

        return FI >= 1.0;
    }

    const char* name() const { return "LaDevezeDelamination"; }

    void degrade_stress(Real* sigma, const FailureState& fstate) const override {
        if (!fstate.failed) return;
        // Zero through-thickness and interlaminar shear stresses
        sigma[2] = 0.0;  // sigma_33
        sigma[4] = 0.0;  // tau_23
        sigma[5] = 0.0;  // tau_13
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const LaDevezeDelaminationParams& delam) {
        return std::make_unique<LaDevezeDelamination>(base, delam);
    }

private:
    LaDevezeDelaminationParams delam_;
};


// ============================================================================
// 2. Hoffman Failure
// ============================================================================

/**
 * @brief Hoffman anisotropic failure criterion
 *
 * Accounts for different tensile and compressive strengths:
 *   F = C1*(s2-s3)^2 + C2*(s3-s1)^2 + C3*(s1-s2)^2
 *     + C4*s1 + C5*s2 + C6*s3
 *     + C7*s4^2 + C8*s5^2 + C9*s6^2
 *
 * Where:
 *   C1 = 1/(2*Yt*Yc) + 1/(2*Zt*Zc) - 1/(2*Xt*Xc)
 *   C2 = 1/(2*Xt*Xc) + 1/(2*Zt*Zc) - 1/(2*Yt*Yc)
 *   C3 = 1/(2*Xt*Xc) + 1/(2*Yt*Yc) - 1/(2*Zt*Zc)
 *   C4 = 1/Xt - 1/Xc,  C5 = 1/Yt - 1/Yc,  C6 = 1/Zt - 1/Zc
 *   C7 = 1/S12^2,  C8 = 1/S23^2,  C9 = 1/S13^2
 *
 * Failed when F >= 1.
 *
 * History:
 *   [0] = maximum failure index
 *
 * Reference: Hoffman (1967), J. Comp. Mat.
 */
class HoffmanFailure : public FailureModel {
public:
    HoffmanFailure(const FailureModelParameters& base_params,
                   const HoffmanFailureParams& hoff_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , hoff_(hoff_params)
    {
        // Precompute Hoffman coefficients
        Real XtXc = hoff_.Xt * hoff_.Xc;
        Real YtYc = hoff_.Yt * hoff_.Yc;
        Real ZtZc = hoff_.Zt * hoff_.Zc;

        if (XtXc < 1.0e-30) XtXc = 1.0e-30;
        if (YtYc < 1.0e-30) YtYc = 1.0e-30;
        if (ZtZc < 1.0e-30) ZtZc = 1.0e-30;

        C1_ = 0.5 / YtYc + 0.5 / ZtZc - 0.5 / XtXc;
        C2_ = 0.5 / XtXc + 0.5 / ZtZc - 0.5 / YtYc;
        C3_ = 0.5 / XtXc + 0.5 / YtYc - 0.5 / ZtZc;

        C4_ = 1.0 / hoff_.Xt - 1.0 / hoff_.Xc;
        C5_ = 1.0 / hoff_.Yt - 1.0 / hoff_.Yc;
        C6_ = 1.0 / hoff_.Zt - 1.0 / hoff_.Zc;

        Real S12_sq = hoff_.S12 * hoff_.S12;
        Real S23_sq = hoff_.S23 * hoff_.S23;
        Real S13_sq = hoff_.S13 * hoff_.S13;
        if (S12_sq < 1.0e-30) S12_sq = 1.0e-30;
        if (S23_sq < 1.0e-30) S23_sq = 1.0e-30;
        if (S13_sq < 1.0e-30) S13_sq = 1.0e-30;

        C7_ = 1.0 / S12_sq;
        C8_ = 1.0 / S23_sq;
        C9_ = 1.0 / S13_sq;
    }

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real s1 = mstate.stress[0];
        Real s2 = mstate.stress[1];
        Real s3 = mstate.stress[2];
        Real s4 = mstate.stress[3];  // tau_12
        Real s5 = mstate.stress[4];  // tau_23
        Real s6 = mstate.stress[5];  // tau_13

        Real FI = C1_ * (s2 - s3) * (s2 - s3)
                + C2_ * (s3 - s1) * (s3 - s1)
                + C3_ * (s1 - s2) * (s1 - s2)
                + C4_ * s1 + C5_ * s2 + C6_ * s3
                + C7_ * s4 * s4 + C8_ * s5 * s5 + C9_ * s6 * s6;

        if (FI > fstate.history[0]) {
            fstate.history[0] = FI;
        }

        fstate.damage = (fstate.history[0] > 1.0) ? 1.0 : fstate.history[0];

        if (fstate.history[0] >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& state) const {
        Real s1 = state.stress[0];
        Real s2 = state.stress[1];
        Real s3 = state.stress[2];
        Real s4 = state.stress[3];
        Real s5 = state.stress[4];
        Real s6 = state.stress[5];

        Real FI = C1_ * (s2 - s3) * (s2 - s3)
                + C2_ * (s3 - s1) * (s3 - s1)
                + C3_ * (s1 - s2) * (s1 - s2)
                + C4_ * s1 + C5_ * s2 + C6_ * s3
                + C7_ * s4 * s4 + C8_ * s5 * s5 + C9_ * s6 * s6;

        return FI >= 1.0;
    }

    const char* name() const { return "HoffmanFailure"; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const HoffmanFailureParams& hoff) {
        return std::make_unique<HoffmanFailure>(base, hoff);
    }

private:
    HoffmanFailureParams hoff_;
    Real C1_, C2_, C3_, C4_, C5_, C6_, C7_, C8_, C9_;
};


// ============================================================================
// 3. Tsai-Hill Failure
// ============================================================================

/**
 * @brief Tsai-Hill interactive composite failure criterion
 *
 * Simplified interactive quadratic criterion:
 *   FI = (s1/X)^2 - s1*s2/X^2 + (s2/Y)^2 + (s12/S)^2
 *
 * The interaction term (-s1*s2/X^2) distinguishes this from maximum stress.
 * Failed when FI >= 1.
 *
 * History:
 *   [0] = maximum failure index encountered
 *
 * Reference: Tsai (1968), NASA CR-71, Hill (1948)
 */
class TsaiHillFailure : public FailureModel {
public:
    TsaiHillFailure(const FailureModelParameters& base_params,
                    const TsaiHillFailureParams& th_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , th_(th_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real s1 = mstate.stress[0];   // Fiber direction stress
        Real s2 = mstate.stress[1];   // Transverse stress
        Real s12 = mstate.stress[3];  // In-plane shear

        Real X = th_.X;
        Real Y = th_.Y;
        Real S = th_.S;

        if (X < 1.0e-20) X = 1.0e-20;
        if (Y < 1.0e-20) Y = 1.0e-20;
        if (S < 1.0e-20) S = 1.0e-20;

        Real X2 = X * X;
        Real Y2 = Y * Y;
        Real S2 = S * S;

        Real FI = (s1 * s1) / X2
                - (s1 * s2) / X2
                + (s2 * s2) / Y2
                + (s12 * s12) / S2;

        if (FI > fstate.history[0]) {
            fstate.history[0] = FI;
        }

        fstate.damage = (fstate.history[0] > 1.0) ? 1.0 : fstate.history[0];

        if (fstate.history[0] >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& state) const {
        Real s1 = state.stress[0];
        Real s2 = state.stress[1];
        Real s12 = state.stress[3];

        Real X = th_.X;
        Real Y = th_.Y;
        Real S = th_.S;
        if (X < 1.0e-20) X = 1.0e-20;
        if (Y < 1.0e-20) Y = 1.0e-20;
        if (S < 1.0e-20) S = 1.0e-20;

        Real FI = (s1 * s1) / (X * X)
                - (s1 * s2) / (X * X)
                + (s2 * s2) / (Y * Y)
                + (s12 * s12) / (S * S);

        return FI >= 1.0;
    }

    const char* name() const { return "TsaiHillFailure"; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const TsaiHillFailureParams& th) {
        return std::make_unique<TsaiHillFailure>(base, th);
    }

private:
    TsaiHillFailureParams th_;
};


// ============================================================================
// 4. RTCl Failure (Reinforced Thermoplastic Composite)
// ============================================================================

/**
 * @brief Reinforced thermoplastic composite failure model
 *
 * Two independent failure modes:
 *
 * Fiber failure:
 *   Tension:    |sigma_1| / X_ft >= 1 when sigma_1 > 0
 *   Compression: |sigma_1| / X_fc >= 1 when sigma_1 < 0
 *
 * Matrix failure (Puck-like quadratic):
 *   (sigma_2 / Y_m)^2 + (tau_12 / S_12)^2 + beta * (sigma_2 / Y_m) >= 1
 *   where Y_m = Y_mt (tension) or Y_mc (compression) depending on sign of sigma_2
 *
 * History:
 *   [0] = fiber failure index (max)
 *   [1] = matrix failure index (max)
 *   [2] = fiber damage flag
 *   [3] = matrix damage flag
 *
 * Reference: Metha & Ishai; Ladeveze & LeDantec (1992)
 */
class RTClFailure : public FailureModel {
public:
    RTClFailure(const FailureModelParameters& base_params,
                const RTClFailureParams& rtcl_params)
        : FailureModel(FailureModelType::Hashin, base_params)
        , rtcl_(rtcl_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        Real s1 = mstate.stress[0];   // Fiber direction
        Real s2 = mstate.stress[1];   // Transverse
        Real t12 = mstate.stress[3];  // In-plane shear

        // --- Fiber failure ---
        Real ff = 0.0;
        if (s1 > 0.0) {
            Real X_ft = rtcl_.X_ft;
            if (X_ft < 1.0e-20) X_ft = 1.0e-20;
            ff = Kokkos::fabs(s1) / X_ft;
        } else {
            Real X_fc = rtcl_.X_fc;
            if (X_fc < 1.0e-20) X_fc = 1.0e-20;
            ff = Kokkos::fabs(s1) / X_fc;
        }

        if (ff > fstate.history[0]) fstate.history[0] = ff;

        // --- Matrix failure ---
        Real Y_m = (s2 >= 0.0) ? rtcl_.Y_mt : rtcl_.Y_mc;
        if (Y_m < 1.0e-20) Y_m = 1.0e-20;

        Real S_12 = rtcl_.S_12;
        if (S_12 < 1.0e-20) S_12 = 1.0e-20;

        Real mf = (s2 / Y_m) * (s2 / Y_m)
                + (t12 / S_12) * (t12 / S_12)
                + rtcl_.beta * (s2 / Y_m);

        if (mf > fstate.history[1]) fstate.history[1] = mf;

        // Update damage flags (irreversible)
        if (fstate.history[0] >= 1.0) fstate.history[2] = 1.0;
        if (fstate.history[1] >= 1.0) fstate.history[3] = 1.0;

        // Overall damage
        Real max_damage = 0.0;
        int fail_mode = 0;
        if (fstate.history[2] >= 1.0) { max_damage = 1.0; fail_mode = 1; }  // Fiber
        if (fstate.history[3] >= 1.0) { max_damage = 1.0; fail_mode = 2; }  // Matrix

        // If no failure yet, report proximity
        if (max_damage < 1.0) {
            Real fi_max = (fstate.history[0] > fstate.history[1]) ? fstate.history[0] : fstate.history[1];
            max_damage = (fi_max > 1.0) ? 1.0 : fi_max;
        }

        fstate.damage = max_damage;
        if (fstate.history[2] >= 1.0 || fstate.history[3] >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = fail_mode;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& state) const {
        Real s1 = state.stress[0];
        Real s2 = state.stress[1];
        Real t12 = state.stress[3];

        // Fiber failure check
        Real X_f = (s1 > 0.0) ? rtcl_.X_ft : rtcl_.X_fc;
        if (X_f < 1.0e-20) X_f = 1.0e-20;
        if (Kokkos::fabs(s1) / X_f >= 1.0) return true;

        // Matrix failure check
        Real Y_m = (s2 >= 0.0) ? rtcl_.Y_mt : rtcl_.Y_mc;
        if (Y_m < 1.0e-20) Y_m = 1.0e-20;
        Real S_12 = rtcl_.S_12;
        if (S_12 < 1.0e-20) S_12 = 1.0e-20;

        Real mf = (s2 / Y_m) * (s2 / Y_m)
                + (t12 / S_12) * (t12 / S_12)
                + rtcl_.beta * (s2 / Y_m);
        return mf >= 1.0;
    }

    const char* name() const { return "RTClFailure"; }

    void degrade_stress(Real* sigma, const FailureState& fstate) const override {
        // Fiber failure: degrade fiber-direction and shear
        if (fstate.history[2] >= 1.0) {
            sigma[0] = 0.0;
            sigma[3] = 0.0;
            sigma[5] = 0.0;
        }
        // Matrix failure: degrade transverse and shear
        if (fstate.history[3] >= 1.0) {
            sigma[1] = 0.0;
            sigma[3] = 0.0;
            sigma[4] = 0.0;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const RTClFailureParams& rtcl) {
        return std::make_unique<RTClFailure>(base, rtcl);
    }

private:
    RTClFailureParams rtcl_;
};


// ============================================================================
// 5. Mullins Effect (Ogden-Roxburgh Damage)
// ============================================================================

/**
 * @brief Mullins effect softening model for filled rubbers
 *
 * Ogden-Roxburgh pseudo-elastic damage:
 *   d = 1 - (1/r) * erf( (W_max - W) / (m + beta * W_max) )
 *
 * Stress reduction:
 *   sigma = (1 - d) * sigma_virgin
 *
 * W is the current strain energy density and W_max is the historical
 * maximum. Damage is only active during unloading (W < W_max).
 *
 * History:
 *   [0] = W_max (maximum strain energy density encountered)
 *   [1] = current damage variable d
 *   [2] = current strain energy density W
 *
 * Reference: Ogden & Roxburgh (1999), Proc. R. Soc. London A
 */
class MullinsEffect : public FailureModel {
public:
    MullinsEffect(const FailureModelParameters& base_params,
                  const MullinsEffectParams& mul_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)
        , mul_(mul_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // Current strain energy density: W = 0.5 * sigma : epsilon
        Real W = 0.0;
        for (int i = 0; i < 6; ++i) {
            W += 0.5 * mstate.stress[i] * mstate.strain[i];
        }
        if (W < 0.0) W = 0.0;
        fstate.history[2] = W;

        // Update maximum strain energy
        if (W > fstate.history[0]) {
            fstate.history[0] = W;
        }

        Real W_max = fstate.history[0];

        // Compute damage (only during unloading / reloading below W_max)
        Real d = 0.0;
        if (W_max > 1.0e-20 && W < W_max) {
            Real denom = mul_.m + mul_.beta * W_max;
            if (denom < 1.0e-20) denom = 1.0e-20;

            Real arg = (W_max - W) / denom;

            // erf approximation (Abramowitz & Stegun)
            Real erf_val = erf_approx(arg);

            Real r = mul_.r;
            if (r < 1.0e-20) r = 1.0e-20;

            d = 1.0 - (1.0 / r) * erf_val;
            if (d < 0.0) d = 0.0;
            if (d > 1.0) d = 1.0;
        }

        fstate.history[1] = d;
        fstate.damage = d;

        // Mullins effect is continuous softening, not a discrete failure
        // Mark as "failed" only if damage saturates
        if (d >= 0.999) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& state) const {
        // Mullins effect is a continuous softening; check if damage is significant
        Real W = 0.0;
        for (int i = 0; i < 6; ++i) {
            W += 0.5 * state.stress[i] * state.strain[i];
        }
        // This is an instantaneous check; always returns false as it needs history
        // True failure requires history tracking via compute_damage
        (void)W;
        return false;
    }

    const char* name() const { return "MullinsEffect"; }

    void degrade_stress(Real* sigma, const FailureState& fstate) const override {
        Real d = fstate.history[1];
        if (d > 0.0 && d < 1.0) {
            Real factor = 1.0 - d;
            for (int i = 0; i < 6; ++i) sigma[i] *= factor;
        } else if (d >= 1.0) {
            for (int i = 0; i < 6; ++i) sigma[i] = 0.0;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const MullinsEffectParams& mul) {
        return std::make_unique<MullinsEffect>(base, mul);
    }

private:
    MullinsEffectParams mul_;

    /// Abramowitz & Stegun erf approximation (max error ~1.5e-7)
    KOKKOS_INLINE_FUNCTION
    static Real erf_approx(Real x) {
        // Handle negative arguments
        Real sign = 1.0;
        if (x < 0.0) {
            sign = -1.0;
            x = -x;
        }

        // Constants
        constexpr Real a1 = 0.254829592;
        constexpr Real a2 = -0.284496736;
        constexpr Real a3 = 1.421413741;
        constexpr Real a4 = -1.453152027;
        constexpr Real a5 = 1.061405429;
        constexpr Real p  = 0.3275911;

        Real t = 1.0 / (1.0 + p * x);
        Real t2 = t * t;
        Real t3 = t2 * t;
        Real t4 = t3 * t;
        Real t5 = t4 * t;

        Real erf_val = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5)
                       * Kokkos::exp(-x * x);

        return sign * erf_val;
    }
};


// ============================================================================
// 6. Spalling Failure
// ============================================================================

/**
 * @brief Dynamic spall failure under tension waves
 *
 * Fails when hydrostatic tension exceeds the spall strength for
 * sufficient cumulative duration. Uses a power-law impulse criterion:
 *
 *   D += max(0, P_hydro - P_spall)^n * dt
 *
 * Where P_hydro = -(sigma_xx + sigma_yy + sigma_zz)/3 is the
 * hydrostatic tension (positive in tension). Failed when D >= D_crit.
 *
 * History:
 *   [0] = cumulative damage D
 *   [1] = maximum hydrostatic tension encountered
 *
 * Reference: Kanel (2010), Int. J. Fract.; Grady (1988)
 */
class SpallingFailure : public FailureModel {
public:
    SpallingFailure(const FailureModelParameters& base_params,
                    const SpallingFailureParams& spall_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)
        , spall_(spall_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real dt,
                        Real /*element_size*/) const override {
        // Hydrostatic tension: positive means tension
        // Pressure p = (s_xx + s_yy + s_zz) / 3
        // Hydrostatic tension = -p
        Real p = (mstate.stress[0] + mstate.stress[1] + mstate.stress[2]) / 3.0;
        Real P_hydro = -p;  // Positive in tension

        // Track maximum hydrostatic tension
        if (P_hydro > fstate.history[1]) {
            fstate.history[1] = P_hydro;
        }

        // Accumulate damage when tension exceeds spall threshold
        Real overstress = P_hydro - spall_.P_spall;
        if (overstress > 0.0 && dt > 0.0) {
            Real increment = Kokkos::pow(overstress, spall_.exponent) * dt;
            fstate.history[0] += increment;
        }

        Real D_crit = spall_.D_crit;
        if (D_crit < 1.0e-30) D_crit = 1.0e-30;

        fstate.damage = fstate.history[0] / D_crit;
        if (fstate.damage > 1.0) fstate.damage = 1.0;

        if (fstate.history[0] >= D_crit) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& state) const {
        Real p = (state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;
        Real P_hydro = -p;
        return P_hydro >= spall_.P_spall;
    }

    const char* name() const { return "SpallingFailure"; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const SpallingFailureParams& spall) {
        return std::make_unique<SpallingFailure>(base, spall);
    }

private:
    SpallingFailureParams spall_;
};


// ============================================================================
// 7. HCDSSE Failure (High-Cycle Damage for Structural Steels)
// ============================================================================

/**
 * @brief High-cycle fatigue damage model (Wohler/Basquin S-N curve)
 *
 * Cycles to failure from Basquin relation:
 *   N_f = (sigma_a / sigma_ref)^(-b)
 *
 * Palmgren-Miner damage accumulation:
 *   D += 1 / N_f  per cycle
 *
 * Stress amplitude sigma_a is computed from von Mises equivalent stress.
 * The model tracks loading reversals to count cycles using a simple
 * zero-crossing approach on the von Mises stress rate.
 *
 * History:
 *   [0] = accumulated damage D
 *   [1] = number of counted cycles
 *   [2] = previous von Mises stress (for reversal detection)
 *   [3] = previous stress rate sign
 *   [4] = peak stress in current half-cycle
 *   [5] = valley stress in current half-cycle
 *
 * Reference: Basquin (1910); Palmgren (1924); Miner (1945)
 */
class HCDSSEFailure : public FailureModel {
public:
    HCDSSEFailure(const FailureModelParameters& base_params,
                  const HCDSSEFailureParams& hcd_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)
        , hcd_(hcd_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // Current von Mises stress
        Real sigma_vm = Material::von_mises_stress(mstate.stress);

        Real prev_sigma = fstate.history[2];
        Real prev_sign = fstate.history[3];

        // Compute stress rate sign
        Real d_sigma = sigma_vm - prev_sigma;
        Real cur_sign = (d_sigma > 0.0) ? 1.0 : ((d_sigma < 0.0) ? -1.0 : prev_sign);

        // Detect reversal (sign change in stress rate)
        bool reversal = (prev_sign != 0.0 && cur_sign != 0.0 && cur_sign != prev_sign);

        // Track peaks and valleys
        if (cur_sign > 0.0) {
            // Loading: update peak
            if (sigma_vm > fstate.history[4]) fstate.history[4] = sigma_vm;
        } else {
            // Unloading: update valley
            if (sigma_vm < fstate.history[5] || fstate.history[5] < 1.0e-30) {
                fstate.history[5] = sigma_vm;
            }
        }

        // Count cycles and accumulate damage on reversals
        if (reversal && cur_sign < 0.0) {
            // Completed a half cycle (peak reached), count as half cycle
            Real peak = fstate.history[4];
            Real valley = fstate.history[5];

            // Stress amplitude
            Real sigma_a = 0.5 * (peak - valley);

            if (sigma_a > hcd_.sigma_endurance && sigma_a > 1.0e-20) {
                // Basquin: N_f = (sigma_a / sigma_ref)^(-b)
                Real ratio = sigma_a / hcd_.sigma_ref;
                if (ratio < 1.0e-20) ratio = 1.0e-20;

                Real N_f = Kokkos::pow(ratio, -hcd_.b_exp);
                if (N_f < 1.0) N_f = 1.0;

                // Palmgren-Miner: half cycle contributes 0.5/N_f
                fstate.history[0] += 0.5 / N_f;
            }

            fstate.history[1] += 0.5;  // Half-cycle count

            // Reset peak/valley for next half-cycle
            fstate.history[4] = sigma_vm;
            fstate.history[5] = sigma_vm;
        }

        // Update tracking variables
        fstate.history[2] = sigma_vm;
        fstate.history[3] = cur_sign;

        fstate.damage = fstate.history[0] / hcd_.D_crit;
        if (fstate.damage > 1.0) fstate.damage = 1.0;

        if (fstate.history[0] >= hcd_.D_crit) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& state) const {
        // Instantaneous check: stress amplitude exceeds endurance limit
        Real sigma_vm = Material::von_mises_stress(state.stress);
        return sigma_vm >= hcd_.sigma_ref;
    }

    const char* name() const { return "HCDSSEFailure"; }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const HCDSSEFailureParams& hcd) {
        return std::make_unique<HCDSSEFailure>(base, hcd);
    }

private:
    HCDSSEFailureParams hcd_;
};


// ============================================================================
// 8. Adhesive Joint Failure
// ============================================================================

/**
 * @brief Adhesive/cohesive joint failure model
 *
 * Two-phase failure process:
 *
 * Phase 1 - Damage initiation (quadratic traction criterion):
 *   (<sigma_n>+/t_n)^2 + (tau_s/t_s)^2 >= 1
 *
 * Phase 2 - Damage evolution (energy-based):
 *   D is computed from the Benzeggagh-Kenane mixed-mode fracture criterion:
 *   G_c = G_Ic + (G_IIc - G_Ic) * (G_shear / G_total)^eta
 *
 * History:
 *   [0] = damage variable D
 *   [1] = maximum displacement ratio (for irreversibility)
 *   [2] = accumulated mode I energy
 *   [3] = accumulated mode II energy
 *   [4] = initiation flag (0=not initiated, 1=initiated)
 *
 * Reference: Camanho & Davila (2002), NASA-TM-2002-211737
 */
class AdhesiveJointFailure : public FailureModel {
public:
    AdhesiveJointFailure(const FailureModelParameters& base_params,
                         const AdhesiveJointFailureParams& adh_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)
        , adh_(adh_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // Extract tractions from stress state
        // Normal traction (through-thickness): sigma_33
        Real sigma_n = mstate.stress[2];

        // Shear tractions: tau_13, tau_23
        Real tau_13 = mstate.stress[5];
        Real tau_23 = mstate.stress[4];
        Real tau_s = Kokkos::sqrt(tau_13 * tau_13 + tau_23 * tau_23);

        // Only tensile normal contributes (Macaulay bracket)
        Real sigma_n_pos = (sigma_n > 0.0) ? sigma_n : 0.0;

        Real t_n = adh_.t_n;
        Real t_s = adh_.t_s;
        if (t_n < 1.0e-20) t_n = 1.0e-20;
        if (t_s < 1.0e-20) t_s = 1.0e-20;

        // Phase 1: Initiation check
        Real initiation_index = (sigma_n_pos / t_n) * (sigma_n_pos / t_n)
                              + (tau_s / t_s) * (tau_s / t_s);

        if (initiation_index >= 1.0 && fstate.history[4] < 0.5) {
            fstate.history[4] = 1.0;  // Mark as initiated
        }

        // Phase 2: Damage evolution (only after initiation)
        if (fstate.history[4] >= 0.5) {
            // Equivalent displacement
            Real delta_n = mstate.strain[2];  // Normal opening
            Real delta_s = Kokkos::sqrt(mstate.strain[4] * mstate.strain[4]
                                       + mstate.strain[5] * mstate.strain[5]);

            Real delta_n_pos = (delta_n > 0.0) ? delta_n : 0.0;
            Real delta_eq = Kokkos::sqrt(delta_n_pos * delta_n_pos + delta_s * delta_s);

            // Track maximum displacement for irreversibility
            if (delta_eq > fstate.history[1]) {
                fstate.history[1] = delta_eq;
            }

            // Mode mixity for BK criterion
            Real G_n = 0.5 * sigma_n_pos * delta_n_pos;
            Real G_s = 0.5 * tau_s * delta_s;
            fstate.history[2] = G_n;
            fstate.history[3] = G_s;

            Real G_total = G_n + G_s;

            // Mixed-mode critical energy (Benzeggagh-Kenane)
            Real G_c = adh_.G_Ic;
            if (G_total > 1.0e-20) {
                Real mode_ratio = G_s / G_total;
                G_c = adh_.G_Ic + (adh_.G_IIc - adh_.G_Ic)
                    * Kokkos::pow(mode_ratio, adh_.eta);
            }

            // Damage as energy ratio
            if (G_c > 1.0e-20) {
                Real D = G_total / G_c;
                if (D > fstate.history[0]) {
                    fstate.history[0] = D;  // Irreversible
                }
            }
        }

        fstate.damage = fstate.history[0];
        if (fstate.damage > 1.0) fstate.damage = 1.0;

        if (fstate.history[0] >= 1.0) {
            fstate.failed = true;
            // Determine dominant mode
            Real G_n = fstate.history[2];
            Real G_s = fstate.history[3];
            fstate.failure_mode = (G_n >= G_s) ? 1 : 2;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& state) const {
        Real sigma_n = state.stress[2];
        Real tau_13 = state.stress[5];
        Real tau_23 = state.stress[4];
        Real tau_s = Kokkos::sqrt(tau_13 * tau_13 + tau_23 * tau_23);
        Real sigma_n_pos = (sigma_n > 0.0) ? sigma_n : 0.0;

        Real t_n = adh_.t_n;
        Real t_s = adh_.t_s;
        if (t_n < 1.0e-20) t_n = 1.0e-20;
        if (t_s < 1.0e-20) t_s = 1.0e-20;

        Real FI = (sigma_n_pos / t_n) * (sigma_n_pos / t_n)
                + (tau_s / t_s) * (tau_s / t_s);
        return FI >= 1.0;
    }

    const char* name() const { return "AdhesiveJointFailure"; }

    void degrade_stress(Real* sigma, const FailureState& fstate) const override {
        Real D = fstate.damage;
        if (D > 0.0 && D < 1.0) {
            Real factor = 1.0 - D;
            // Degrade through-thickness and interlaminar components
            sigma[2] *= factor;
            sigma[4] *= factor;
            sigma[5] *= factor;
        } else if (D >= 1.0) {
            sigma[2] = 0.0;
            sigma[4] = 0.0;
            sigma[5] = 0.0;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const AdhesiveJointFailureParams& adh) {
        return std::make_unique<AdhesiveJointFailure>(base, adh);
    }

private:
    AdhesiveJointFailureParams adh_;
};


// ============================================================================
// 9. Windshield Failure
// ============================================================================

/**
 * @brief Windshield laminate progressive failure model
 *
 * Models a glass-PVB-glass laminate with separate failure for each layer:
 *
 * Glass layers (Weibull statistical fracture):
 *   P_fail = 1 - exp( -(V/V0) * (sigma_1/sigma0)^m )
 *   Glass fails when max principal stress exceeds Weibull-derived threshold.
 *   Simplified deterministic check: sigma_1 >= sigma0 * (V0/V)^(1/m)
 *
 * PVB interlayer (stretch-based):
 *   Fails when principal stretch ratio lambda_1 >= lambda_limit
 *   or accumulated energy >= tear_energy
 *
 * History:
 *   [0] = glass layer 1 damage (0 or 1)
 *   [1] = glass layer 2 damage (0 or 1)
 *   [2] = PVB damage (0 to 1)
 *   [3] = max principal stress encountered
 *   [4] = max principal stretch encountered
 *   [5] = accumulated PVB stretch energy
 *
 * Reference: Timmel et al. (2007), Int. J. Impact Eng.
 */
class WindshieldFailure : public FailureModel {
public:
    WindshieldFailure(const FailureModelParameters& base_params,
                      const WindshieldFailureParams& ws_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)
        , ws_(ws_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real dt,
                        Real /*element_size*/) const override {
        // --- Glass layer assessment ---
        // Approximate max principal stress from normal components
        Real s1 = mstate.stress[0];
        Real s2 = mstate.stress[1];
        Real s3 = mstate.stress[2];

        // Sort to find max principal (approximate)
        Real sigma_max = s1;
        if (s2 > sigma_max) sigma_max = s2;
        if (s3 > sigma_max) sigma_max = s3;

        // Track max principal stress
        if (sigma_max > fstate.history[3]) {
            fstate.history[3] = sigma_max;
        }

        // Weibull fracture threshold (deterministic, unit volume)
        Real sigma_thresh = ws_.glass_sigma0;

        // Glass layer 1: fails if max principal stress exceeds threshold
        if (fstate.history[0] < 0.5 && sigma_max >= sigma_thresh) {
            fstate.history[0] = 1.0;  // Glass 1 cracked
        }

        // Glass layer 2: fails at slightly higher stress (crack on second layer)
        // Model second layer at 1.1x threshold (accounting for stress redistribution)
        if (fstate.history[1] < 0.5 && sigma_max >= 1.1 * sigma_thresh) {
            fstate.history[1] = 1.0;  // Glass 2 cracked
        }

        // --- PVB interlayer assessment ---
        // Approximate principal stretch from strain
        Real e1 = mstate.strain[0];
        Real e2 = mstate.strain[1];

        // Simple max in-plane stretch ratio
        Real lambda_1 = 1.0 + ((e1 > e2) ? e1 : e2);
        if (lambda_1 < 1.0) lambda_1 = 1.0;

        if (lambda_1 > fstate.history[4]) {
            fstate.history[4] = lambda_1;
        }

        // PVB energy accumulation (stretching energy)
        Real stretch_power = 0.0;
        for (int i = 0; i < 6; ++i) {
            stretch_power += Kokkos::fabs(mstate.stress[i] * mstate.strain_rate[i]);
        }
        fstate.history[5] += stretch_power * dt;

        // PVB damage based on stretch ratio
        Real pvb_damage = 0.0;
        if (ws_.pvb_stretch_limit > 1.0) {
            Real stretch_ratio = (lambda_1 - 1.0) / (ws_.pvb_stretch_limit - 1.0);
            if (stretch_ratio > 1.0) stretch_ratio = 1.0;
            pvb_damage = stretch_ratio;
        }

        // Also check energy criterion
        if (ws_.pvb_tear_energy > 1.0e-20) {
            Real energy_damage = fstate.history[5] / ws_.pvb_tear_energy;
            if (energy_damage > pvb_damage) pvb_damage = energy_damage;
        }

        if (pvb_damage > fstate.history[2]) {
            fstate.history[2] = pvb_damage;
        }

        // Overall damage: max of all layers
        Real overall = 0.0;
        if (fstate.history[0] >= 1.0 && fstate.history[1] >= 1.0) {
            // Both glass layers cracked
            overall = 0.5 + 0.5 * fstate.history[2];
        } else if (fstate.history[0] >= 1.0 || fstate.history[1] >= 1.0) {
            // One glass layer cracked
            overall = 0.25 + 0.25 * fstate.history[2];
        } else {
            overall = 0.0;
        }

        fstate.damage = (overall > 1.0) ? 1.0 : overall;

        // Full failure: both glass cracked AND PVB torn
        if (fstate.history[0] >= 1.0 && fstate.history[1] >= 1.0
            && fstate.history[2] >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 3;  // Full laminate through-failure
        } else if (fstate.history[0] >= 1.0 && fstate.history[1] >= 1.0) {
            fstate.failure_mode = 2;  // Both glass cracked, PVB holding
        } else if (fstate.history[0] >= 1.0 || fstate.history[1] >= 1.0) {
            fstate.failure_mode = 1;  // Single glass layer cracked
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& state) const {
        // Instantaneous check: max principal stress exceeds glass strength
        Real s1 = state.stress[0];
        Real s2 = state.stress[1];
        Real s3 = state.stress[2];
        Real sigma_max = s1;
        if (s2 > sigma_max) sigma_max = s2;
        if (s3 > sigma_max) sigma_max = s3;
        return sigma_max >= ws_.glass_sigma0;
    }

    const char* name() const { return "WindshieldFailure"; }

    void degrade_stress(Real* sigma, const FailureState& fstate) const override {
        // Progressive degradation based on damage state
        if (fstate.history[0] >= 1.0 && fstate.history[1] >= 1.0) {
            // Both glass layers cracked: only PVB carries load
            Real pvb_factor = 1.0 - fstate.history[2];
            if (pvb_factor < 0.0) pvb_factor = 0.0;
            // PVB carries ~10% of glass stiffness
            Real factor = 0.1 * pvb_factor;
            for (int i = 0; i < 6; ++i) sigma[i] *= factor;
        } else if (fstate.history[0] >= 1.0 || fstate.history[1] >= 1.0) {
            // One glass layer cracked: ~50% capacity
            for (int i = 0; i < 6; ++i) sigma[i] *= 0.5;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const WindshieldFailureParams& ws) {
        return std::make_unique<WindshieldFailure>(base, ws);
    }

private:
    WindshieldFailureParams ws_;
};


// ============================================================================
// 10. Generalized Energy Failure
// ============================================================================

/**
 * @brief Generalized energy failure criterion
 *
 * Cumulative plastic work criterion:
 *   D = Integral[ sigma_eq * d_eps_p ] / G_c
 *
 * Where sigma_eq is the von Mises equivalent stress, d_eps_p is the
 * effective plastic strain increment, and G_c is the critical fracture
 * energy.
 *
 * Failed when D >= 1.
 *
 * History:
 *   [0] = accumulated plastic dissipation W_p
 *   [1] = previous plastic strain
 *
 * Reference: Hillerborg et al. (1976), Cem. Concr. Res.
 */
class GeneralizedEnergyFailure : public FailureModel {
public:
    GeneralizedEnergyFailure(const FailureModelParameters& base_params,
                             const GeneralizedEnergyFailureParams& ge_params)
        : FailureModel(FailureModelType::TabulatedEnvelope, base_params)
        , ge_(ge_params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // Von Mises equivalent stress
        Real sigma_eq = Material::von_mises_stress(mstate.stress);

        // Plastic strain increment
        Real eps_p = mstate.plastic_strain;
        Real eps_p_old = fstate.history[1];
        Real delta_eps_p = eps_p - eps_p_old;
        if (delta_eps_p < 0.0) delta_eps_p = 0.0;
        fstate.history[1] = eps_p;

        // Accumulate plastic dissipation
        fstate.history[0] += sigma_eq * delta_eps_p;

        // Damage
        Real G_c = ge_.G_c;
        if (G_c < 1.0e-20) G_c = 1.0e-20;

        fstate.damage = fstate.history[0] / G_c;
        if (fstate.damage > 1.0) fstate.damage = 1.0;

        if (fstate.history[0] >= G_c) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool evaluate(const MaterialState& state) const {
        // Instantaneous check not meaningful for cumulative criterion
        // Return true if plastic strain is large (heuristic)
        return state.plastic_strain > 1.0;
    }

    const char* name() const { return "GeneralizedEnergyFailure"; }

    void degrade_stress(Real* sigma, const FailureState& fstate) const override {
        Real D = fstate.damage;
        if (D > 0.0 && D < 1.0) {
            Real factor = 1.0 - D;
            for (int i = 0; i < 6; ++i) sigma[i] *= factor;
        } else if (D >= 1.0) {
            for (int i = 0; i < 6; ++i) sigma[i] = 0.0;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& base,
                                                 const GeneralizedEnergyFailureParams& ge) {
        return std::make_unique<GeneralizedEnergyFailure>(base, ge);
    }

private:
    GeneralizedEnergyFailureParams ge_;
};


// ============================================================================
// Utility: to_string for Wave 19 failure model types
// ============================================================================

inline const char* to_string(FailureModelTypeWave19 type) {
    switch (type) {
        case FailureModelTypeWave19::LaDevezeDelamination: return "LaDevezeDelamination";
        case FailureModelTypeWave19::Hoffman:              return "Hoffman";
        case FailureModelTypeWave19::TsaiHill:             return "TsaiHill";
        case FailureModelTypeWave19::RTCl:                 return "RTCl";
        case FailureModelTypeWave19::MullinsEffect:        return "MullinsEffect";
        case FailureModelTypeWave19::Spalling:             return "Spalling";
        case FailureModelTypeWave19::HCDSSE:               return "HCDSSE";
        case FailureModelTypeWave19::AdhesiveJoint:        return "AdhesiveJoint";
        case FailureModelTypeWave19::Windshield:           return "Windshield";
        case FailureModelTypeWave19::GeneralizedEnergy:    return "GeneralizedEnergy";
        default:                                           return "Unknown";
    }
}

} // namespace failure
} // namespace physics
} // namespace nxs
