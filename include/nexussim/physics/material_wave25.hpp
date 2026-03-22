#pragma once

/**
 * @file material_wave25.hpp
 * @brief Wave 25 material models: 10 advanced constitutive models
 *
 * Models included:
 *   1.  OrthotropicBrittleMaterial      - Rankine cracking per direction (MAT125)
 *   2.  ExplosiveBurnExtMaterial        - Multi-point detonation (MAT42-49)
 *   3.  ExtendedSoilMaterial            - Pressure-dependent yield with tension cutoff (MAT63/69)
 *   4.  SoilAndCrushMaterial            - 3-invariant soil with crush curve (MAT72/79)
 *   5.  AdvancedPolymerMaterial         - Bergstrom-Boyce multi-network (MAT105-109)
 *   6.  SpecialHardeningMaterial        - Arbitrary curve + Bauschinger effect (MAT119)
 *   7.  MultiScaleMaterial              - Macro stress from micro RVE homogenization (MAT112)
 *   8.  ExtendedRateCompositeMaterial   - Rate-dependent with mode separation
 *   9.  HysteresisSpringExtMaterial     - Full hysteresis loop with energy dissipation
 *  10.  ConcreteDamPlastMaterial        - Lee-Fenves damage-plasticity
 */

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

// ============================================================================
// 1. OrthotropicBrittleMaterial - Rankine Cracking Per Direction (MAT125)
// ============================================================================

/**
 * @brief Orthotropic brittle material with independent cracking per direction
 *
 * Crack initiates when sigma_i > ft_i (tensile strength per direction).
 * After cracking, stiffness in that direction is reduced:
 *   E_i_cracked = E_i * (1 - d_i)^2
 * Exponential softening: d_i evolves with crack opening strain.
 *
 * History: [32]=d1, [33]=d2, [34]=d3 (damage per direction)
 */
class OrthotropicBrittleMaterial : public Material {
public:
    /**
     * @param props  Material properties (E1,E2,E3 for directional moduli)
     * @param ft1    Tensile strength in direction 1
     * @param ft2    Tensile strength in direction 2
     * @param ft3    Tensile strength in direction 3
     * @param Gf     Fracture energy (controls softening rate)
     * @param h      Characteristic element length
     */
    OrthotropicBrittleMaterial(const MaterialProperties& props,
                                Real ft1 = 3.0e6, Real ft2 = 3.0e6, Real ft3 = 3.0e6,
                                Real Gf = 100.0, Real h = 0.01)
        : Material(MaterialType::Custom, props)
        , ft_{ft1, ft2, ft3}, Gf_(Gf), h_(h)
    {
        // Softening slope: epsilon_f = 2*Gf / (ft * h)
        for (int i = 0; i < 3; ++i) {
            Real Ei = (i == 0) ? props.E1 : ((i == 1) ? props.E2 : props.E3);
            if (Ei < 1.0e-30) Ei = props.E;
            eps_cr_[i] = ft_[i] / Ei;
            eps_f_[i] = 2.0 * Gf_ / (ft_[i] * h_ + 1.0e-30);
            if (eps_f_[i] < eps_cr_[i] * 1.01) eps_f_[i] = eps_cr_[i] * 2.0;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E[3];
        E[0] = props_.E1 > 0.0 ? props_.E1 : props_.E;
        E[1] = props_.E2 > 0.0 ? props_.E2 : props_.E;
        E[2] = props_.E3 > 0.0 ? props_.E3 : props_.E;

        Real nu = props_.nu;
        Real G12 = props_.G12 > 0.0 ? props_.G12 : props_.G;
        Real G23 = props_.G23 > 0.0 ? props_.G23 : props_.G;
        Real G13 = props_.G13 > 0.0 ? props_.G13 : props_.G;

        // Retrieve damage per direction
        Real d[3];
        d[0] = state.history[32];
        d[1] = state.history[33];
        d[2] = state.history[34];

        // Update damage based on current strain
        for (int i = 0; i < 3; ++i) {
            Real eps_i = state.strain[i];
            if (eps_i > eps_cr_[i]) {
                // Exponential softening
                Real excess = eps_i - eps_cr_[i];
                Real range = eps_f_[i] - eps_cr_[i];
                Real d_new = 1.0 - Kokkos::exp(-3.0 * excess / (range + 1.0e-30));
                if (d_new < 0.0) d_new = 0.0;
                if (d_new > 1.0) d_new = 1.0;
                // Damage can only grow
                if (d_new > d[i]) d[i] = d_new;
            }
            state.history[32 + i] = d[i];
        }

        // Effective moduli with damage
        Real E_eff[3];
        for (int i = 0; i < 3; ++i) {
            Real factor = (1.0 - d[i]) * (1.0 - d[i]);
            E_eff[i] = E[i] * factor;
            if (E_eff[i] < 1.0) E_eff[i] = 1.0; // Floor to avoid singularity
        }

        // Simplified orthotropic compliance (uncoupled approximation for clarity)
        // sigma_i = E_eff_i * eps_i + nu * E_eff_i / (1-2nu) * ev (simplified)
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];

        // Use a simplified orthotropic stiffness:
        // For each normal direction: sigma_i = lambda_eff * ev + 2 * mu_eff_i * eps_i
        // where mu_eff_i = E_eff_i / (2*(1+nu)), lambda_eff = average
        Real E_avg = (E_eff[0] + E_eff[1] + E_eff[2]) / 3.0;
        Real lambda = E_avg * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        for (int i = 0; i < 3; ++i) {
            Real mu_i = E_eff[i] / (2.0 * (1.0 + nu));
            state.stress[i] = lambda * ev + 2.0 * mu_i * state.strain[i];
        }

        // Shear: use undamaged shear moduli (simplified)
        Real d_avg_12 = Kokkos::fmax(d[0], d[1]);
        Real d_avg_23 = Kokkos::fmax(d[1], d[2]);
        Real d_avg_13 = Kokkos::fmax(d[0], d[2]);
        state.stress[3] = G12 * (1.0 - d_avg_12) * state.strain[3];
        state.stress[4] = G23 * (1.0 - d_avg_23) * state.strain[4];
        state.stress[5] = G13 * (1.0 - d_avg_13) * state.strain[5];

        // Overall damage for state
        state.damage = Kokkos::fmax(d[0], Kokkos::fmax(d[1], d[2]));
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E[3];
        E[0] = props_.E1 > 0.0 ? props_.E1 : props_.E;
        E[1] = props_.E2 > 0.0 ? props_.E2 : props_.E;
        E[2] = props_.E3 > 0.0 ? props_.E3 : props_.E;

        Real nu = props_.nu;

        Real d[3] = {state.history[32], state.history[33], state.history[34]};
        Real E_eff[3];
        for (int i = 0; i < 3; ++i) {
            E_eff[i] = E[i] * (1.0 - d[i]) * (1.0 - d[i]);
            if (E_eff[i] < 1.0) E_eff[i] = 1.0;
        }

        Real E_avg = (E_eff[0] + E_eff[1] + E_eff[2]) / 3.0;
        Real lambda = E_avg * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        for (int i = 0; i < 3; ++i) {
            Real mu_i = E_eff[i] / (2.0 * (1.0 + nu));
            C[i * 6 + i] = lambda + 2.0 * mu_i;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda;
            }
        }

        Real G12 = props_.G12 > 0.0 ? props_.G12 : props_.G;
        Real G23 = props_.G23 > 0.0 ? props_.G23 : props_.G;
        Real G13 = props_.G13 > 0.0 ? props_.G13 : props_.G;

        C[21] = G12 * (1.0 - Kokkos::fmax(d[0], d[1]));
        C[28] = G23 * (1.0 - Kokkos::fmax(d[1], d[2]));
        C[35] = G13 * (1.0 - Kokkos::fmax(d[0], d[2]));
    }

    /// Get damage in a specific direction (0-indexed)
    KOKKOS_INLINE_FUNCTION
    Real directional_damage(const MaterialState& state, int dir) const {
        if (dir >= 0 && dir < 3) return state.history[32 + dir];
        return 0.0;
    }

private:
    Real ft_[3];       ///< Tensile strengths per direction
    Real Gf_;          ///< Fracture energy
    Real h_;           ///< Characteristic length
    Real eps_cr_[3];   ///< Cracking strains per direction
    Real eps_f_[3];    ///< Failure strains per direction
};

// ============================================================================
// 2. ExplosiveBurnExtMaterial - Multi-Point Detonation (MAT42-49)
// ============================================================================

/**
 * @brief Multi-point detonation with independent burn fractions
 *
 * Multiple detonation points, each with (x,y,z, t_light).
 * F_total = max(F_i) where each F_i = beta_burn from point i.
 * JWL EOS for products, unreacted EOS for unburnt.
 *
 * History: [32..35] = burn fractions for up to 4 detonation points
 */
class ExplosiveBurnExtMaterial : public Material {
public:
    struct DetonationPoint {
        Real x, y, z;
        Real t_light;
    };

    ExplosiveBurnExtMaterial(const MaterialProperties& props,
                              const DetonationPoint* det_points, int num_points,
                              Real A_jwl = 3.712e11, Real B_jwl = 3.231e9,
                              Real R1 = 4.15, Real R2 = 0.95, Real omega = 0.30,
                              Real D_cj = 6930.0)
        : Material(MaterialType::Custom, props)
        , num_det_points_(num_points > 4 ? 4 : num_points)
        , A_jwl_(A_jwl), B_jwl_(B_jwl), R1_(R1), R2_(R2), omega_(omega)
        , D_cj_(D_cj)
    {
        for (int i = 0; i < 4; ++i) {
            if (i < num_det_points_) {
                det_points_[i] = det_points[i];
            } else {
                det_points_[i] = {0.0, 0.0, 0.0, 1.0e30};
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real rho0 = props_.density;

        // Accumulate elapsed time
        Real t_elapsed = state.history[36] + state.dt;
        state.history[36] = t_elapsed;

        // Element position from history [37..39]
        Real ex = state.history[37];
        Real ey = state.history[38];
        Real ez = state.history[39];

        // Compute burn fraction for each detonation point
        Real F_total = 0.0;
        for (int i = 0; i < num_det_points_; ++i) {
            Real dx = ex - det_points_[i].x;
            Real dy = ey - det_points_[i].y;
            Real dz = ez - det_points_[i].z;
            Real dist = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
            Real t_arrive = det_points_[i].t_light + dist / (D_cj_ + 1.0e-30);

            Real F_i = state.history[32 + i];
            if (t_elapsed >= t_arrive && F_i < 1.0) {
                // Beta burn: ramp burn fraction
                Real dt_since = t_elapsed - t_arrive;
                Real char_len = 0.01; // Characteristic element size
                F_i = Kokkos::fmin(1.0, dt_since * D_cj_ / (char_len + 1.0e-30));
                if (F_i > state.history[32 + i]) {
                    state.history[32 + i] = F_i;
                }
            }
            F_total = Kokkos::fmax(F_total, state.history[32 + i]);
        }

        // Relative volume
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real V = 1.0 + ev;
        if (V < 0.1) V = 0.1;

        // JWL pressure for detonated products
        Real P_det = A_jwl_ * (1.0 - omega_ / (R1_ * V)) * Kokkos::exp(-R1_ * V)
                   + B_jwl_ * (1.0 - omega_ / (R2_ * V)) * Kokkos::exp(-R2_ * V);

        // Unreacted: simple bulk pressure
        Real K_unreact = rho0 * D_cj_ * D_cj_ * 0.25;
        Real P_unreact = -K_unreact * ev;

        // Mixed pressure
        Real P = F_total * P_det + (1.0 - F_total) * P_unreact;

        // Hydrostatic stress
        state.stress[0] = -P;
        state.stress[1] = -P;
        state.stress[2] = -P;
        state.stress[3] = 0.0;
        state.stress[4] = 0.0;
        state.stress[5] = 0.0;

        state.damage = F_total;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real K = props_.density * D_cj_ * D_cj_ * 0.25;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = K;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = K;
    }

    /// Get burn fraction for a specific detonation point
    KOKKOS_INLINE_FUNCTION
    Real burn_fraction(const MaterialState& state, int point = 0) const {
        if (point >= 0 && point < 4) return state.history[32 + point];
        return 0.0;
    }

    /// Get total (max) burn fraction
    KOKKOS_INLINE_FUNCTION
    Real total_burn_fraction(const MaterialState& state) const {
        Real F = 0.0;
        for (int i = 0; i < num_det_points_; ++i) {
            F = Kokkos::fmax(F, state.history[32 + i]);
        }
        return F;
    }

private:
    int num_det_points_;
    DetonationPoint det_points_[4];
    Real A_jwl_, B_jwl_, R1_, R2_, omega_;
    Real D_cj_;
};

// ============================================================================
// 3. ExtendedSoilMaterial - Pressure-Dependent Yield with Tension Cutoff (MAT63/69)
// ============================================================================

/**
 * @brief Extended soil model with Drucker-Prager-like yield and tension cutoff
 *
 * Yield: f = sqrt(J2) - (a0 + a1*P + a2*P^2) where P = -I1/3
 * Tension cutoff at P = -P_t
 * Associative flow for shear, non-associative for tension cutoff.
 *
 * History: [0]=plastic_strain, [32]=tension_cutoff_active
 */
class ExtendedSoilMaterial : public Material {
public:
    /**
     * @param props  Material properties
     * @param a0     Cohesion term
     * @param a1     Friction coefficient (linear pressure term)
     * @param a2     Quadratic pressure coefficient
     * @param P_t    Tension cutoff pressure (positive value)
     */
    ExtendedSoilMaterial(const MaterialProperties& props,
                          Real a0 = 1.0e6, Real a1 = 0.5, Real a2 = 0.0,
                          Real P_t = 0.5e6)
        : Material(MaterialType::Custom, props)
        , a0_(a0), a1_(a1), a2_(a2), P_t_(P_t) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;
        const Real K = props_.K;

        // Get accumulated plastic strain
        Real eps_p = state.history[0];

        // Compute trial elastic strain
        Real strain_e[6];
        for (int i = 0; i < 6; ++i) {
            strain_e[i] = state.strain[i] - state.history[i + 1];
        }

        // Trial stress
        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        // Invariants
        Real I1 = stress_trial[0] + stress_trial[1] + stress_trial[2];
        Real P = -I1 / 3.0; // Pressure (positive in compression)

        Real s[6];
        Real p_mean = I1 / 3.0;
        for (int i = 0; i < 3; ++i) s[i] = stress_trial[i] - p_mean;
        for (int i = 3; i < 6; ++i) s[i] = stress_trial[i];

        Real J2 = 0.5 * (s[0]*s[0] + s[1]*s[1] + s[2]*s[2])
                 + s[3]*s[3] + s[4]*s[4] + s[5]*s[5];
        Real sqrt_J2 = Kokkos::sqrt(J2 + 1.0e-30);

        // Tension cutoff check
        state.history[32] = 0.0;
        if (P < -P_t_) {
            // Tension cutoff: limit pressure to -P_t
            Real P_new = -P_t_;
            Real I1_new = -3.0 * P_new;
            Real dp = (I1_new - I1) / 3.0;
            for (int i = 0; i < 3; ++i) {
                stress_trial[i] += dp;
            }
            I1 = I1_new;
            P = P_new;
            state.history[32] = 1.0; // Tension cutoff active
        }

        // Yield surface: f = sqrt(J2) - F(P)
        Real F_P = a0_ + a1_ * P + a2_ * P * P;
        if (F_P < 0.0) F_P = 0.0;

        Real f_trial = sqrt_J2 - F_P;

        if (f_trial <= 0.0) {
            // Elastic
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            // Plastic return: radial return on deviatoric part
            Real dF_dP = a1_ + 2.0 * a2_ * P;
            Real H = 0.0; // No hardening in basic model
            Real denom = G + K * dF_dP * dF_dP / 3.0 + H;
            if (denom < 1.0e-30) denom = 1.0e-30;

            Real delta_gamma = f_trial / (G + H + 1.0e-30);

            // Update plastic strain
            eps_p += delta_gamma;
            state.history[0] = eps_p;

            // Radial return on deviatoric stress
            Real scale = 1.0 - G * delta_gamma / (sqrt_J2 + 1.0e-30);
            if (scale < 0.0) scale = 0.0;

            Real p_final = -P;
            state.stress[0] = scale * s[0] + p_final;
            state.stress[1] = scale * s[1] + p_final;
            state.stress[2] = scale * s[2] + p_final;
            state.stress[3] = scale * s[3];
            state.stress[4] = scale * s[4];
            state.stress[5] = scale * s[5];

            // Update plastic strain tensor
            if (sqrt_J2 > 1.0e-12) {
                Real factor = delta_gamma / (2.0 * sqrt_J2);
                for (int i = 0; i < 6; ++i) {
                    state.history[i + 1] += factor * s[i];
                }
            }
        }

        state.plastic_strain = eps_p;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const Real mu = E / (2.0 * (1.0 + nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lambda + 2.0 * mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

    /// Check if tension cutoff is active
    KOKKOS_INLINE_FUNCTION
    bool tension_cutoff_active(const MaterialState& state) const {
        return state.history[32] > 0.5;
    }

    /// Evaluate yield surface strength at given pressure
    KOKKOS_INLINE_FUNCTION
    Real yield_strength(Real P) const {
        Real F = a0_ + a1_ * P + a2_ * P * P;
        return F > 0.0 ? F : 0.0;
    }

private:
    Real a0_, a1_, a2_; ///< Yield surface coefficients
    Real P_t_;          ///< Tension cutoff pressure
};

// ============================================================================
// 4. SoilAndCrushMaterial - 3-Invariant Soil with Crush Curve (MAT72/79)
// ============================================================================

/**
 * @brief Modified soil model with Lode angle dependence and volumetric crush
 *
 * Yield: f = sqrt(J2)*r(theta) - F(P)
 * r(theta) accounts for Lode angle dependence (triangular deviatoric section).
 * F(P) = a0 + P/(a1 + a2*P) (meridional yield function).
 * Volumetric crush curve for compaction.
 *
 * History: [32]=max_pressure, [33]=crush_volumetric_strain
 */
class SoilAndCrushMaterial : public Material {
public:
    /**
     * @param props    Material properties
     * @param a0_m     Meridional yield intercept
     * @param a1_m     Meridional yield linear denominator
     * @param a2_m     Meridional yield quadratic denominator
     * @param e_lode   Eccentricity for Lode angle dependence (0.5 < e < 1.0)
     * @param P_crush  Crush pressure (onset of compaction)
     * @param K_crush  Crush bulk modulus (compacted)
     */
    SoilAndCrushMaterial(const MaterialProperties& props,
                          Real a0_m = 5.0e6, Real a1_m = 0.1, Real a2_m = 1.0e-9,
                          Real e_lode = 0.6,
                          Real P_crush = 50.0e6, Real K_crush = 20.0e9)
        : Material(MaterialType::Custom, props)
        , a0_m_(a0_m), a1_m_(a1_m), a2_m_(a2_m)
        , e_lode_(e_lode), P_crush_(P_crush), K_crush_(K_crush) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;

        // Trial stress
        Real strain_e[6];
        for (int i = 0; i < 6; ++i) {
            strain_e[i] = state.strain[i] - state.history[i + 1];
        }

        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        // Invariants
        Real I1 = stress_trial[0] + stress_trial[1] + stress_trial[2];
        Real P = -I1 / 3.0;

        // Track max pressure
        if (P > state.history[32]) state.history[32] = P;

        Real p_mean = I1 / 3.0;
        Real s[6];
        for (int i = 0; i < 3; ++i) s[i] = stress_trial[i] - p_mean;
        for (int i = 3; i < 6; ++i) s[i] = stress_trial[i];

        Real J2 = 0.5 * (s[0]*s[0] + s[1]*s[1] + s[2]*s[2])
                 + s[3]*s[3] + s[4]*s[4] + s[5]*s[5];
        Real sqrt_J2 = Kokkos::sqrt(J2 + 1.0e-30);

        // J3 for Lode angle
        Real J3 = s[0] * (s[1] * s[2] - s[4] * s[4])
                 - s[3] * (s[3] * s[2] - s[4] * s[5])
                 + s[5] * (s[3] * s[4] - s[1] * s[5]);

        // Lode angle: cos(3*theta) = (3*sqrt(3)/2) * J3 / J2^(3/2)
        Real cos3theta = 0.0;
        if (J2 > 1.0e-20) {
            Real J2_32 = Kokkos::pow(J2, 1.5);
            cos3theta = 1.5 * Kokkos::sqrt(3.0) * J3 / (J2_32 + 1.0e-30);
            if (cos3theta > 1.0) cos3theta = 1.0;
            if (cos3theta < -1.0) cos3theta = -1.0;
        }

        // r(theta) from Willam-Warnke
        Real e = e_lode_;
        Real cos_theta = Kokkos::sqrt((1.0 + cos3theta) / 2.0 + 1.0e-30);
        if (cos_theta < 0.01) cos_theta = 0.01;

        // Simplified r(theta) using the Willam-Warnke formula:
        // r = (2*(1-e^2)*cos(theta) + (2e-1)*sqrt(4*(1-e^2)*cos^2(theta) + 5*e^2-4*e))
        //     / (4*(1-e^2)*cos^2(theta) + (2e-1)^2)
        Real e2 = e * e;
        Real one_minus_e2 = 1.0 - e2;
        Real two_e_minus_1 = 2.0 * e - 1.0;
        Real cos2 = cos_theta * cos_theta;

        Real numer_sqrt = 4.0 * one_minus_e2 * cos2 + 5.0 * e2 - 4.0 * e;
        if (numer_sqrt < 0.0) numer_sqrt = 0.0;
        Real r_theta = (2.0 * one_minus_e2 * cos_theta + two_e_minus_1 * Kokkos::sqrt(numer_sqrt))
                      / (4.0 * one_minus_e2 * cos2 + two_e_minus_1 * two_e_minus_1 + 1.0e-30);
        if (r_theta < 0.5) r_theta = 0.5;
        if (r_theta > 1.0) r_theta = 1.0;

        // Meridional yield function: F(P) = a0 + P / (a1 + a2*P)
        Real F_P = a0_m_;
        if (P > 0.0) {
            Real denom = a1_m_ + a2_m_ * P;
            if (denom > 1.0e-30) {
                F_P += P / denom;
            }
        }

        // Yield check: f = sqrt(J2) * r(theta) - F(P)
        Real f_trial = sqrt_J2 * r_theta - F_P;

        // Crush curve: volumetric compaction
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        if (P > P_crush_ && ev < state.history[33]) {
            state.history[33] = ev; // Track crush volumetric strain
        }

        if (f_trial <= 0.0) {
            // Elastic
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            // Radial return
            Real delta_gamma = f_trial / (G * r_theta + 1.0e-30);

            state.history[0] += delta_gamma;

            Real scale = 1.0 - G * delta_gamma * r_theta / (sqrt_J2 * r_theta + 1.0e-30);
            if (scale < 0.0) scale = 0.0;

            state.stress[0] = scale * s[0] + p_mean;
            state.stress[1] = scale * s[1] + p_mean;
            state.stress[2] = scale * s[2] + p_mean;
            state.stress[3] = scale * s[3];
            state.stress[4] = scale * s[4];
            state.stress[5] = scale * s[5];

            if (sqrt_J2 > 1.0e-12) {
                Real factor = delta_gamma / (2.0 * sqrt_J2);
                for (int i = 0; i < 6; ++i) {
                    state.history[i + 1] += factor * s[i];
                }
            }
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const Real mu = E / (2.0 * (1.0 + nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lambda + 2.0 * mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

    /// Get max pressure reached
    KOKKOS_INLINE_FUNCTION
    Real max_pressure(const MaterialState& state) const { return state.history[32]; }

    /// Get crush volumetric strain
    KOKKOS_INLINE_FUNCTION
    Real crush_vol_strain(const MaterialState& state) const { return state.history[33]; }

    /// Evaluate r(theta) for a given Lode angle cosine
    KOKKOS_INLINE_FUNCTION
    Real lode_factor(Real cos_theta) const {
        Real e = e_lode_;
        Real e2 = e * e;
        Real one_minus_e2 = 1.0 - e2;
        Real two_e_minus_1 = 2.0 * e - 1.0;
        Real cos2 = cos_theta * cos_theta;
        Real numer_sqrt = 4.0 * one_minus_e2 * cos2 + 5.0 * e2 - 4.0 * e;
        if (numer_sqrt < 0.0) numer_sqrt = 0.0;
        Real r = (2.0 * one_minus_e2 * cos_theta + two_e_minus_1 * Kokkos::sqrt(numer_sqrt))
                / (4.0 * one_minus_e2 * cos2 + two_e_minus_1 * two_e_minus_1 + 1.0e-30);
        if (r < 0.5) r = 0.5;
        if (r > 1.0) r = 1.0;
        return r;
    }

private:
    Real a0_m_, a1_m_, a2_m_;
    Real e_lode_;
    Real P_crush_, K_crush_;
};

// ============================================================================
// 5. AdvancedPolymerMaterial - Bergstrom-Boyce Multi-Network (MAT105-109)
// ============================================================================

/**
 * @brief Bergstrom-Boyce multi-network polymer model
 *
 * Two parallel networks: A (equilibrium, hyperelastic) + B (viscous, rate-dependent).
 * Network A: 8-chain model.
 * Network B: same 8-chain + flow rule for viscous creep.
 *
 * History: [32..40] = viscous deformation gradient (3x3, row-major)
 */
class AdvancedPolymerMaterial : public Material {
public:
    /**
     * @param props      Material properties (E for initial modulus, nu for Poisson's)
     * @param mu_A       Shear modulus of network A (equilibrium)
     * @param lambda_L_A Locking stretch of network A
     * @param mu_B       Shear modulus of network B (viscous)
     * @param lambda_L_B Locking stretch of network B
     * @param gamma0     Reference creep rate
     * @param m_rate     Rate exponent (1/m)
     * @param C_creep    Chain stretch exponent for creep
     * @param xi_creep   Creep regularization (small positive)
     */
    AdvancedPolymerMaterial(const MaterialProperties& props,
                             Real mu_A = 1.0e6, Real lambda_L_A = 5.0,
                             Real mu_B = 2.0e6, Real lambda_L_B = 5.0,
                             Real gamma0 = 1.0e3, Real m_rate = 0.2,
                             Real C_creep = -1.0, Real xi_creep = 0.01)
        : Material(MaterialType::Custom, props)
        , mu_A_(mu_A), lambda_L_A_(lambda_L_A)
        , mu_B_(mu_B), lambda_L_B_(lambda_L_B)
        , gamma0_(gamma0), m_rate_(m_rate)
        , C_creep_(C_creep), xi_creep_(xi_creep) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        // Initialize viscous deformation gradient as identity if all zero
        bool all_zero = true;
        for (int i = 32; i < 41; ++i) {
            if (Kokkos::fabs(state.history[i]) > 1.0e-30) { all_zero = false; break; }
        }
        if (all_zero) {
            state.history[32] = 1.0; state.history[33] = 0.0; state.history[34] = 0.0;
            state.history[35] = 0.0; state.history[36] = 1.0; state.history[37] = 0.0;
            state.history[38] = 0.0; state.history[39] = 0.0; state.history[40] = 1.0;
        }

        // Total deformation gradient
        const Real* F = state.F;

        // Network A: 8-chain hyperelastic on total F
        Real sigma_A[6];
        compute_8chain_stress(F, mu_A_, lambda_L_A_, sigma_A);

        // Network B: elastic part is F_e_B = F * inv(F_v_B)
        Real F_v[9];
        for (int i = 0; i < 9; ++i) F_v[i] = state.history[32 + i];

        // Compute F_v inverse
        Real F_v_inv[9];
        invert_3x3(F_v, F_v_inv);

        // F_e_B = F * F_v^{-1}
        Real F_eB[9];
        mat_mult_3x3(F, F_v_inv, F_eB);

        // Stress from network B elastic part
        Real sigma_B[6];
        compute_8chain_stress(F_eB, mu_B_, lambda_L_B_, sigma_B);

        // Compute driving stress for viscous flow (deviatoric of sigma_B)
        Real tau_vm = von_mises_stress(sigma_B);

        // Chain stretch for viscous network
        Real B_eB[9];
        // B = F_eB * F_eB^T
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                B_eB[i*3+j] = 0.0;
                for (int k = 0; k < 3; ++k) {
                    B_eB[i*3+j] += F_eB[i*3+k] * F_eB[j*3+k];
                }
            }
        }
        Real I1_B = B_eB[0] + B_eB[4] + B_eB[8];
        Real lambda_chain_B = Kokkos::sqrt(I1_B / 3.0);

        // Viscous flow rate: gamma_dot = gamma0 * (tau/tau_hat)^(1/m) * (lambda_chain - 1 + xi)^C
        Real tau_hat = mu_B_ * 1.0; // Reference stress
        Real gamma_dot = 0.0;
        if (tau_vm > 1.0e-10 && tau_hat > 1.0e-10) {
            Real rate_term = Kokkos::pow(tau_vm / tau_hat, 1.0 / (m_rate_ + 1.0e-10));
            Real stretch_term = Kokkos::pow(
                Kokkos::fmax(lambda_chain_B - 1.0 + xi_creep_, 1.0e-10), C_creep_);
            gamma_dot = gamma0_ * rate_term * stretch_term;
        }

        // Update viscous deformation gradient: F_v_dot = gamma_dot * N * F_v
        // where N = deviatoric stress direction
        Real dt = state.dt;
        if (tau_vm > 1.0e-10 && gamma_dot > 1.0e-30 && dt > 0.0) {
            Real p_B = (sigma_B[0] + sigma_B[1] + sigma_B[2]) / 3.0;
            Real s_B[6];
            s_B[0] = sigma_B[0] - p_B;
            s_B[1] = sigma_B[1] - p_B;
            s_B[2] = sigma_B[2] - p_B;
            s_B[3] = sigma_B[3];
            s_B[4] = sigma_B[4];
            s_B[5] = sigma_B[5];

            // Direction tensor N (3x3 symmetric -> full)
            Real N[9];
            Real inv_tau = 1.0 / (tau_vm + 1.0e-30);
            N[0] = 1.5 * s_B[0] * inv_tau; N[1] = 1.5 * s_B[3] * inv_tau; N[2] = 1.5 * s_B[5] * inv_tau;
            N[3] = N[1]; N[4] = 1.5 * s_B[1] * inv_tau; N[5] = 1.5 * s_B[4] * inv_tau;
            N[6] = N[2]; N[7] = N[5]; N[8] = 1.5 * s_B[2] * inv_tau;

            // L_v = gamma_dot * N (velocity gradient for viscous part)
            // F_v_new = (I + dt * L_v) * F_v_old (forward Euler)
            Real I_plus_dtL[9];
            for (int i = 0; i < 9; ++i) I_plus_dtL[i] = gamma_dot * dt * N[i];
            I_plus_dtL[0] += 1.0; I_plus_dtL[4] += 1.0; I_plus_dtL[8] += 1.0;

            Real F_v_new[9];
            mat_mult_3x3(I_plus_dtL, F_v, F_v_new);

            for (int i = 0; i < 9; ++i) state.history[32 + i] = F_v_new[i];
        }

        // Total stress = sigma_A + sigma_B
        for (int i = 0; i < 6; ++i) {
            state.stress[i] = sigma_A[i] + sigma_B[i];
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        // Approximate tangent: use initial modulus
        Real mu = mu_A_ + mu_B_;
        Real K = props_.K;
        Real lambda = K - 2.0 * mu / 3.0;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lambda + 2.0 * mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

private:
    Real mu_A_, lambda_L_A_;
    Real mu_B_, lambda_L_B_;
    Real gamma0_, m_rate_, C_creep_, xi_creep_;

    /// Inverse Langevin approximation: L^{-1}(x) ~ x*(3-x^2)/(1-x^2)
    KOKKOS_INLINE_FUNCTION
    static Real langevin_inv(Real x) {
        if (x >= 0.99) x = 0.99;
        if (x <= -0.99) x = -0.99;
        return x * (3.0 - x * x) / (1.0 - x * x + 1.0e-30);
    }

    /// Compute 8-chain stress from deformation gradient
    KOKKOS_INLINE_FUNCTION
    static void compute_8chain_stress(const Real* F, Real mu, Real lambda_L,
                                       Real* sigma) {
        // B = F * F^T
        Real B[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                B[i*3+j] = 0.0;
                for (int k = 0; k < 3; ++k) {
                    B[i*3+j] += F[i*3+k] * F[j*3+k];
                }
            }
        }

        // J = det(F)
        Real J = F[0]*(F[4]*F[8] - F[5]*F[7])
               - F[1]*(F[3]*F[8] - F[5]*F[6])
               + F[2]*(F[3]*F[7] - F[4]*F[6]);
        if (J < 1.0e-10) J = 1.0e-10;

        Real I1 = B[0] + B[4] + B[8];
        Real lambda_chain = Kokkos::sqrt(I1 / 3.0);
        if (lambda_chain < 1.0e-10) lambda_chain = 1.0e-10;

        Real x = lambda_chain / lambda_L;
        Real L_inv = langevin_inv(x);

        // Stress: sigma = (mu / (3*J)) * (L_inv / lambda_chain) * B_dev
        Real I1_third = I1 / 3.0;
        Real coeff = mu / (3.0 * J) * L_inv / lambda_chain;

        sigma[0] = coeff * (B[0] - I1_third);
        sigma[1] = coeff * (B[4] - I1_third);
        sigma[2] = coeff * (B[8] - I1_third);
        sigma[3] = coeff * B[1];
        sigma[4] = coeff * B[5];
        sigma[5] = coeff * B[2];
    }

    /// 3x3 matrix multiply: C = A * B
    KOKKOS_INLINE_FUNCTION
    static void mat_mult_3x3(const Real* A, const Real* B, Real* C_out) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                C_out[i*3+j] = 0.0;
                for (int k = 0; k < 3; ++k) {
                    C_out[i*3+j] += A[i*3+k] * B[k*3+j];
                }
            }
        }
    }

    /// 3x3 matrix inversion
    KOKKOS_INLINE_FUNCTION
    static void invert_3x3(const Real* M, Real* Minv) {
        Real det = M[0]*(M[4]*M[8] - M[5]*M[7])
                 - M[1]*(M[3]*M[8] - M[5]*M[6])
                 + M[2]*(M[3]*M[7] - M[4]*M[6]);
        if (Kokkos::fabs(det) < 1.0e-30) det = 1.0e-30;
        Real inv_det = 1.0 / det;

        Minv[0] = (M[4]*M[8] - M[5]*M[7]) * inv_det;
        Minv[1] = (M[2]*M[7] - M[1]*M[8]) * inv_det;
        Minv[2] = (M[1]*M[5] - M[2]*M[4]) * inv_det;
        Minv[3] = (M[5]*M[6] - M[3]*M[8]) * inv_det;
        Minv[4] = (M[0]*M[8] - M[2]*M[6]) * inv_det;
        Minv[5] = (M[2]*M[3] - M[0]*M[5]) * inv_det;
        Minv[6] = (M[3]*M[7] - M[4]*M[6]) * inv_det;
        Minv[7] = (M[1]*M[6] - M[0]*M[7]) * inv_det;
        Minv[8] = (M[0]*M[4] - M[1]*M[3]) * inv_det;
    }
};

// ============================================================================
// 6. SpecialHardeningMaterial - Arbitrary Curve + Bauschinger Effect (MAT119)
// ============================================================================

/**
 * @brief Tabulated yield curve with kinematic hardening for Bauschinger effect
 *
 * sigma_y = (1-beta)*sigma_iso(eps_p) + beta*sigma_kin
 * where sigma_kin follows Armstrong-Frederick backstress evolution.
 *
 * History: [0]=plastic_strain, [1..6]=backstress tensor
 */
class SpecialHardeningMaterial : public Material {
public:
    /**
     * @param props   Material properties
     * @param curve   Tabulated yield stress vs plastic strain
     * @param beta    Kinematic hardening fraction (0=iso, 1=kin)
     * @param C_kin   Kinematic hardening modulus
     * @param gamma_kin  Dynamic recovery parameter
     */
    SpecialHardeningMaterial(const MaterialProperties& props,
                              const TabulatedCurve& curve,
                              Real beta = 0.3, Real C_kin = 1.0e9,
                              Real gamma_kin = 10.0)
        : Material(MaterialType::Custom, props)
        , curve_(curve), beta_(beta), C_kin_(C_kin), gamma_kin_(gamma_kin) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;

        Real eps_p = state.history[0];

        // Backstress tensor
        Real alpha[6];
        for (int i = 0; i < 6; ++i) alpha[i] = state.history[1 + i];

        // Trial elastic strain (subtract plastic strain stored in history[7..12])
        Real strain_e[6];
        for (int i = 0; i < 6; ++i) {
            strain_e[i] = state.strain[i] - state.history[7 + i];
        }

        // Trial stress
        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        // Shifted stress: xi = s_trial - alpha
        Real p_trial = (stress_trial[0] + stress_trial[1] + stress_trial[2]) / 3.0;
        Real s_trial[6];
        for (int i = 0; i < 3; ++i) s_trial[i] = stress_trial[i] - p_trial;
        for (int i = 3; i < 6; ++i) s_trial[i] = stress_trial[i];

        Real xi[6];
        for (int i = 0; i < 6; ++i) xi[i] = s_trial[i] - alpha[i];

        // Effective shifted stress
        Real xi_sq = xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2]
                   + 2.0*(xi[3]*xi[3] + xi[4]*xi[4] + xi[5]*xi[5]);
        Real xi_vm = Kokkos::sqrt(1.5 * xi_sq);

        // Current yield stress from tabulated curve (isotropic part)
        Real sigma_iso = curve_.evaluate(eps_p);

        // Combined yield: radius of yield surface
        Real sigma_y = (1.0 - beta_) * sigma_iso + beta_ * sigma_iso;
        // (In this formulation, beta controls the split between iso and kin,
        //  but the yield surface radius is always sigma_iso for initial yield)
        sigma_y = sigma_iso;

        Real f_trial = xi_vm - sigma_y;

        if (f_trial <= 0.0) {
            // Elastic
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            // Plastic return
            Real denom = 3.0 * G + C_kin_ + 1.0e-30;
            // Approximate hardening slope from curve
            Real H_iso = 0.0;
            if (eps_p < 1.0) {
                Real sy1 = curve_.evaluate(eps_p);
                Real sy2 = curve_.evaluate(eps_p + 0.001);
                H_iso = (sy2 - sy1) / 0.001;
            }
            denom = 3.0 * G + (1.0 - beta_) * H_iso + beta_ * C_kin_;
            if (denom < 1.0e-30) denom = 1.0e-30;

            Real delta_gamma = f_trial / denom;

            // Update plastic strain
            eps_p += delta_gamma;
            state.history[0] = eps_p;

            // Flow direction
            Real n[6];
            if (xi_vm > 1.0e-12) {
                Real factor = 1.5 / xi_vm;
                for (int i = 0; i < 6; ++i) n[i] = factor * xi[i];
            } else {
                for (int i = 0; i < 6; ++i) n[i] = 0.0;
            }

            // Update backstress (Armstrong-Frederick):
            // d_alpha = (2/3)*C_kin * d_eps_p - gamma_kin * alpha * d_gamma
            for (int i = 0; i < 6; ++i) {
                Real d_alpha = (2.0 / 3.0) * C_kin_ * delta_gamma * n[i]
                             - gamma_kin_ * alpha[i] * delta_gamma;
                alpha[i] += d_alpha;
                state.history[1 + i] = alpha[i];
            }

            // Update plastic strain tensor
            for (int i = 0; i < 6; ++i) {
                state.history[7 + i] += delta_gamma * n[i];
            }

            // Return stress
            Real scale = 1.0 - 3.0 * G * delta_gamma / (xi_vm + 1.0e-30);
            if (scale < 0.0) scale = 0.0;
            for (int i = 0; i < 3; ++i) {
                state.stress[i] = scale * xi[i] + alpha[i] + p_trial;
            }
            for (int i = 3; i < 6; ++i) {
                state.stress[i] = scale * xi[i] + alpha[i];
            }
        }

        state.plastic_strain = eps_p;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const Real mu = E / (2.0 * (1.0 + nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lambda + 2.0 * mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

    /// Get current yield stress from curve
    KOKKOS_INLINE_FUNCTION
    Real yield_stress(Real eps_p) const {
        return curve_.evaluate(eps_p);
    }

    /// Get backstress magnitude
    KOKKOS_INLINE_FUNCTION
    Real backstress_vm(const MaterialState& state) const {
        Real alpha[6];
        for (int i = 0; i < 6; ++i) alpha[i] = state.history[1 + i];
        Real sq = alpha[0]*alpha[0] + alpha[1]*alpha[1] + alpha[2]*alpha[2]
                + 2.0*(alpha[3]*alpha[3] + alpha[4]*alpha[4] + alpha[5]*alpha[5]);
        return Kokkos::sqrt(1.5 * sq);
    }

private:
    TabulatedCurve curve_;
    Real beta_;
    Real C_kin_;
    Real gamma_kin_;
};

// ============================================================================
// 7. MultiScaleMaterial - Macro Stress from Micro RVE Homogenization (MAT112)
// ============================================================================

/**
 * @brief Simple homogenization: macro_sigma = (1-f)*sigma_matrix + f*sigma_inclusion
 *
 * Matrix: J2 plasticity with hardening.
 * Inclusion: linear elastic (stiffer).
 * Voigt assumption: strain compatibility (equal strain in both phases).
 *
 * History: [0]=matrix_plastic_strain, [32]=inclusion_stress_vm
 */
class MultiScaleMaterial : public Material {
public:
    /**
     * @param props     Material properties (matrix properties)
     * @param E_incl    Inclusion Young's modulus
     * @param nu_incl   Inclusion Poisson's ratio
     * @param f_vol     Inclusion volume fraction (0 to 1)
     */
    MultiScaleMaterial(const MaterialProperties& props,
                        Real E_incl = 400.0e9, Real nu_incl = 0.2,
                        Real f_vol = 0.3)
        : Material(MaterialType::Custom, props)
        , E_incl_(E_incl), nu_incl_(nu_incl), f_vol_(f_vol)
    {
        G_incl_ = E_incl / (2.0 * (1.0 + nu_incl));
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E_m = props_.E;
        const Real nu_m = props_.nu;
        const Real G_m = props_.G;
        const Real sigma_y0 = props_.yield_stress;
        const Real H_m = props_.hardening_modulus;

        // Voigt assumption: both phases see same strain
        Real eps_p_m = state.history[0];

        // ---- Matrix phase: J2 plasticity ----
        Real strain_e_m[6];
        for (int i = 0; i < 6; ++i) {
            strain_e_m[i] = state.strain[i] - state.history[i + 1];
        }

        Real stress_m[6];
        elastic_stress(strain_e_m, E_m, nu_m, stress_m);

        // Deviatoric trial
        Real p_m = (stress_m[0] + stress_m[1] + stress_m[2]) / 3.0;
        Real s_m[6];
        for (int i = 0; i < 3; ++i) s_m[i] = stress_m[i] - p_m;
        for (int i = 3; i < 6; ++i) s_m[i] = stress_m[i];

        Real s_sq = s_m[0]*s_m[0] + s_m[1]*s_m[1] + s_m[2]*s_m[2]
                  + 2.0*(s_m[3]*s_m[3] + s_m[4]*s_m[4] + s_m[5]*s_m[5]);
        Real sigma_vm_m = Kokkos::sqrt(1.5 * s_sq);

        Real sigma_y = sigma_y0 + H_m * eps_p_m;
        Real f_trial = sigma_vm_m - sigma_y;

        if (f_trial > 0.0) {
            Real delta_gamma = f_trial / (3.0 * G_m + H_m);
            eps_p_m += delta_gamma;
            state.history[0] = eps_p_m;

            Real scale = 1.0 - 3.0 * G_m * delta_gamma / (sigma_vm_m + 1.0e-30);
            if (scale < 0.0) scale = 0.0;

            stress_m[0] = scale * s_m[0] + p_m;
            stress_m[1] = scale * s_m[1] + p_m;
            stress_m[2] = scale * s_m[2] + p_m;
            stress_m[3] = scale * s_m[3];
            stress_m[4] = scale * s_m[4];
            stress_m[5] = scale * s_m[5];

            if (sigma_vm_m > 1.0e-12) {
                Real factor = 1.5 * delta_gamma / sigma_vm_m;
                for (int i = 0; i < 6; ++i) {
                    state.history[i + 1] += factor * s_m[i];
                }
            }
        }

        // ---- Inclusion phase: linear elastic ----
        Real stress_incl[6];
        elastic_stress(state.strain, E_incl_, nu_incl_, stress_incl);

        // Store inclusion VM stress
        state.history[32] = von_mises_stress(stress_incl);

        // ---- Homogenization: Voigt rule of mixtures ----
        Real f = f_vol_;
        for (int i = 0; i < 6; ++i) {
            state.stress[i] = (1.0 - f) * stress_m[i] + f * stress_incl[i];
        }

        state.plastic_strain = eps_p_m;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        // Voigt bound: C_eff = (1-f)*C_matrix + f*C_inclusion
        const Real E_m = props_.E;
        const Real nu_m = props_.nu;
        const Real f = f_vol_;

        Real C_m[36], C_i[36];

        // Matrix elastic tangent
        {
            Real lam = E_m * nu_m / ((1.0 + nu_m) * (1.0 - 2.0 * nu_m));
            Real mu = E_m / (2.0 * (1.0 + nu_m));
            for (int i = 0; i < 36; ++i) C_m[i] = 0.0;
            C_m[0] = C_m[7] = C_m[14] = lam + 2.0 * mu;
            C_m[1] = C_m[2] = C_m[6] = C_m[8] = C_m[12] = C_m[13] = lam;
            C_m[21] = C_m[28] = C_m[35] = mu;
        }

        // Inclusion elastic tangent
        {
            Real lam = E_incl_ * nu_incl_ / ((1.0 + nu_incl_) * (1.0 - 2.0 * nu_incl_));
            Real mu = E_incl_ / (2.0 * (1.0 + nu_incl_));
            for (int i = 0; i < 36; ++i) C_i[i] = 0.0;
            C_i[0] = C_i[7] = C_i[14] = lam + 2.0 * mu;
            C_i[1] = C_i[2] = C_i[6] = C_i[8] = C_i[12] = C_i[13] = lam;
            C_i[21] = C_i[28] = C_i[35] = mu;
        }

        for (int i = 0; i < 36; ++i) {
            C[i] = (1.0 - f) * C_m[i] + f * C_i[i];
        }
        (void)state;
    }

    /// Get composite effective modulus (Voigt bound)
    KOKKOS_INLINE_FUNCTION
    Real effective_E() const {
        return (1.0 - f_vol_) * props_.E + f_vol_ * E_incl_;
    }

    /// Get inclusion VM stress
    KOKKOS_INLINE_FUNCTION
    Real inclusion_stress_vm(const MaterialState& state) const {
        return state.history[32];
    }

private:
    Real E_incl_, nu_incl_, G_incl_;
    Real f_vol_;
};

// ============================================================================
// 8. ExtendedRateCompositeMaterial - Rate-Dependent with Mode Separation
// ============================================================================

/**
 * @brief Rate-dependent composite with separate fiber and matrix modes
 *
 * Fiber: sigma_f = E_f * eps_f * (1 + C_f * ln(eps_dot/eps_dot_0))
 * Matrix: sigma_m = sigma_y_m * (1 + C_m * ln(eps_dot/eps_dot_0)) after yield
 * Hashin-like failure modes with rate-enhanced strengths.
 *
 * History: [0]=matrix_plastic_strain, [32]=fiber_damage, [33]=matrix_damage
 */
class ExtendedRateCompositeMaterial : public Material {
public:
    /**
     * @param props       Material properties (E1=fiber modulus, E2=matrix modulus)
     * @param sigma_y_m   Matrix yield stress
     * @param C_f         Fiber rate sensitivity
     * @param C_m         Matrix rate sensitivity
     * @param eps_dot_0   Reference strain rate
     * @param Xt          Fiber tensile strength
     * @param Xc          Fiber compressive strength
     * @param Yt          Matrix tensile strength
     * @param S12         Shear strength
     */
    ExtendedRateCompositeMaterial(const MaterialProperties& props,
                                    Real sigma_y_m = 80.0e6,
                                    Real C_f = 0.02, Real C_m = 0.05,
                                    Real eps_dot_0 = 1.0,
                                    Real Xt = 1500.0e6, Real Xc = 1200.0e6,
                                    Real Yt = 50.0e6, Real S12 = 70.0e6)
        : Material(MaterialType::Custom, props)
        , sigma_y_m_(sigma_y_m), C_f_(C_f), C_m_(C_m), eps_dot_0_(eps_dot_0)
        , Xt_(Xt), Xc_(Xc), Yt_(Yt), S12_(S12) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E_f = props_.E1 > 0.0 ? props_.E1 : props_.E;
        Real E_m = props_.E2 > 0.0 ? props_.E2 : props_.E * 0.1;
        Real G12 = props_.G12 > 0.0 ? props_.G12 : props_.G;
        Real nu12 = props_.nu12 > 0.0 ? props_.nu12 : props_.nu;

        // Rate enhancement factors
        Real eps_dot = state.effective_strain_rate;
        Real rate_f = 1.0;
        Real rate_m = 1.0;
        if (eps_dot > eps_dot_0_) {
            rate_f = 1.0 + C_f_ * Kokkos::log(eps_dot / eps_dot_0_);
            rate_m = 1.0 + C_m_ * Kokkos::log(eps_dot / eps_dot_0_);
        }

        // Current damage
        Real d_f = state.history[32];
        Real d_m = state.history[33];

        // Fiber direction stress (1-direction)
        Real sigma_11 = (1.0 - d_f) * E_f * state.strain[0] * rate_f;

        // Matrix direction stress (2-direction)
        Real eps_m = state.strain[1];
        Real eps_p_m = state.history[0];
        Real sigma_22_trial = E_m * (eps_m - eps_p_m) * rate_m;

        // Matrix plasticity
        Real sigma_y_eff = sigma_y_m_ * rate_m;
        if (Kokkos::fabs(sigma_22_trial) > sigma_y_eff) {
            Real excess = Kokkos::fabs(sigma_22_trial) - sigma_y_eff;
            Real H_m = props_.hardening_modulus > 0.0 ? props_.hardening_modulus : E_m * 0.05;
            Real d_eps_p = excess / (E_m + H_m);
            eps_p_m += d_eps_p;
            state.history[0] = eps_p_m;
            Real sign = sigma_22_trial >= 0.0 ? 1.0 : -1.0;
            sigma_22_trial = sign * (sigma_y_eff + H_m * d_eps_p);
        }
        Real sigma_22 = (1.0 - d_m) * sigma_22_trial;

        // 3-direction (transverse)
        Real sigma_33 = (1.0 - d_m) * E_m * state.strain[2];

        // Shear
        Real sigma_12 = (1.0 - Kokkos::fmax(d_f, d_m)) * G12 * state.strain[3];
        Real sigma_23 = (1.0 - d_m) * G12 * state.strain[4];
        Real sigma_13 = (1.0 - Kokkos::fmax(d_f, d_m)) * G12 * state.strain[5];

        // Hashin-like failure check with rate-enhanced strengths
        Real Xt_eff = Xt_ * rate_f;
        Real Xc_eff = Xc_ * rate_f;
        Real Yt_eff = Yt_ * rate_m;
        Real S12_eff = S12_ * rate_m;

        // Fiber tension failure
        if (state.strain[0] > 0.0) {
            Real f_ft = (sigma_11 / (Xt_eff + 1.0e-30));
            f_ft = f_ft * f_ft;
            if (f_ft > 1.0 && d_f < 1.0) {
                d_f = Kokkos::fmin(1.0, d_f + 0.1);
                state.history[32] = d_f;
            }
        }
        // Fiber compression failure
        if (state.strain[0] < 0.0) {
            Real f_fc = (sigma_11 / (Xc_eff + 1.0e-30));
            f_fc = f_fc * f_fc;
            if (f_fc > 1.0 && d_f < 1.0) {
                d_f = Kokkos::fmin(1.0, d_f + 0.1);
                state.history[32] = d_f;
            }
        }

        // Matrix tension/compression failure
        Real sigma_22_abs = Kokkos::fabs(sigma_22);
        Real f_mt = (sigma_22_abs / (Yt_eff + 1.0e-30));
        f_mt = f_mt * f_mt + (sigma_12 / (S12_eff + 1.0e-30)) * (sigma_12 / (S12_eff + 1.0e-30));
        if (f_mt > 1.0 && d_m < 1.0) {
            d_m = Kokkos::fmin(1.0, d_m + 0.1);
            state.history[33] = d_m;
        }

        // Apply updated damage
        state.stress[0] = (1.0 - d_f) * E_f * state.strain[0] * rate_f;
        state.stress[1] = (1.0 - d_m) * sigma_22_trial;
        state.stress[2] = (1.0 - d_m) * E_m * state.strain[2];
        state.stress[3] = (1.0 - Kokkos::fmax(d_f, d_m)) * G12 * state.strain[3];
        state.stress[4] = (1.0 - d_m) * G12 * state.strain[4];
        state.stress[5] = (1.0 - Kokkos::fmax(d_f, d_m)) * G12 * state.strain[5];

        state.plastic_strain = eps_p_m;
        state.damage = Kokkos::fmax(d_f, d_m);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E_f = props_.E1 > 0.0 ? props_.E1 : props_.E;
        Real E_m = props_.E2 > 0.0 ? props_.E2 : props_.E * 0.1;
        Real G12 = props_.G12 > 0.0 ? props_.G12 : props_.G;
        Real d_f = state.history[32];
        Real d_m = state.history[33];

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = (1.0 - d_f) * E_f;
        C[7] = (1.0 - d_m) * E_m;
        C[14] = (1.0 - d_m) * E_m;
        C[21] = (1.0 - Kokkos::fmax(d_f, d_m)) * G12;
        C[28] = (1.0 - d_m) * G12;
        C[35] = (1.0 - Kokkos::fmax(d_f, d_m)) * G12;
    }

    /// Get fiber damage
    KOKKOS_INLINE_FUNCTION
    Real fiber_damage(const MaterialState& state) const { return state.history[32]; }

    /// Get matrix damage
    KOKKOS_INLINE_FUNCTION
    Real matrix_damage(const MaterialState& state) const { return state.history[33]; }

private:
    Real sigma_y_m_;
    Real C_f_, C_m_, eps_dot_0_;
    Real Xt_, Xc_, Yt_, S12_;
};

// ============================================================================
// 9. HysteresisSpringExtMaterial - Full Hysteresis Loop (Bouc-Wen)
// ============================================================================

/**
 * @brief Nonlinear spring with Bouc-Wen hysteresis model
 *
 * F_total = alpha*k*x + (1-alpha)*k*z
 * dz/dx = 1 - |z|^n * (gamma_bw + beta_bw*sign(dz*z))
 *
 * History: [32]=z (hysteretic variable), [33]=prev_displacement, [34]=energy_dissipated
 */
class HysteresisSpringExtMaterial : public Material {
public:
    /**
     * @param props      Material properties (E = linear stiffness k)
     * @param alpha      Post-yield stiffness ratio (0 < alpha < 1)
     * @param k3         Cubic stiffness coefficient
     * @param n_bw       Bouc-Wen exponent
     * @param beta_bw    Bouc-Wen beta parameter
     * @param gamma_bw   Bouc-Wen gamma parameter
     */
    HysteresisSpringExtMaterial(const MaterialProperties& props,
                                  Real alpha = 0.1, Real k3 = 0.0,
                                  Real n_bw = 1.0,
                                  Real beta_bw = 0.5, Real gamma_bw = 0.5)
        : Material(MaterialType::Custom, props)
        , alpha_(alpha), k3_(k3), n_bw_(n_bw)
        , beta_bw_(beta_bw), gamma_bw_(gamma_bw) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real k = props_.E; // Linear stiffness

        // Current and previous displacement (strain[0] = displacement for 1D spring)
        Real x = state.strain[0];
        Real x_prev = state.history[33];
        Real z = state.history[32];
        Real dx = x - x_prev;

        // Bouc-Wen ODE: dz/dx = 1 - |z|^n * (gamma + beta*sign(dz*z))
        // Explicit Euler integration
        if (Kokkos::fabs(dx) > 1.0e-30) {
            Real z_abs_n = Kokkos::pow(Kokkos::fabs(z) + 1.0e-30, n_bw_);
            Real sign_dxz = (dx * z >= 0.0) ? 1.0 : -1.0;
            Real dz_dx = 1.0 - z_abs_n * (gamma_bw_ + beta_bw_ * sign_dxz);
            Real dz = dz_dx * dx;

            z += dz;

            // Clamp z to reasonable range
            Real z_max = 1.0 / (gamma_bw_ + beta_bw_ + 1.0e-30);
            if (z > z_max) z = z_max;
            if (z < -z_max) z = -z_max;
        }

        state.history[32] = z;
        state.history[33] = x;

        // Total force: F = alpha*k*x + k3*x^3 + (1-alpha)*k*z
        Real F_elastic = alpha_ * k * x + k3_ * x * x * x;
        Real F_hysteretic = (1.0 - alpha_) * k * z;
        Real F_total = F_elastic + F_hysteretic;

        // Store as 1D stress (only xx component)
        state.stress[0] = F_total;
        state.stress[1] = 0.0;
        state.stress[2] = 0.0;
        state.stress[3] = 0.0;
        state.stress[4] = 0.0;
        state.stress[5] = 0.0;

        // Energy dissipation: integral of (1-alpha)*k*z*dx
        Real dW = Kokkos::fabs(F_hysteretic * dx);
        state.history[34] += dW;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real k = props_.E;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = k; // Only xx stiffness
    }

    /// Get hysteretic variable z
    KOKKOS_INLINE_FUNCTION
    Real hysteretic_z(const MaterialState& state) const { return state.history[32]; }

    /// Get energy dissipated
    KOKKOS_INLINE_FUNCTION
    Real energy_dissipated(const MaterialState& state) const { return state.history[34]; }

private:
    Real alpha_;
    Real k3_;
    Real n_bw_;
    Real beta_bw_, gamma_bw_;
};

// ============================================================================
// 10. ConcreteDamPlastMaterial - Lee-Fenves Damage-Plasticity
// ============================================================================

/**
 * @brief Concrete damage-plasticity model (Lee-Fenves)
 *
 * Yield: F = (1/(1-alpha)) * (alpha*I1 + sqrt(3*J2) + beta*<sigma_max>) - sigma_c(eps_p)
 * alpha = (fb0/fc0 - 1) / (2*fb0/fc0 - 1)
 * beta = sigma_c/sigma_t * (1-alpha) - (1+alpha)
 * Damage: d = 1 - (1-d_t)*(1-d_c)
 *
 * History: [0]=plastic_strain, [32]=kappa_t, [33]=kappa_c, [34]=d_t, [35]=d_c
 */
class ConcreteDamPlastMaterial : public Material {
public:
    /**
     * @param props   Material properties (E, nu, yield_stress = fc0)
     * @param ft0     Uniaxial tensile strength
     * @param fc0     Uniaxial compressive strength
     * @param fb0     Biaxial compressive strength
     * @param Gf_t    Tensile fracture energy
     * @param Gf_c    Compressive fracture energy
     * @param h       Characteristic element length
     */
    ConcreteDamPlastMaterial(const MaterialProperties& props,
                               Real ft0 = 3.0e6, Real fc0 = 30.0e6,
                               Real fb0 = 36.0e6,
                               Real Gf_t = 100.0, Real Gf_c = 15000.0,
                               Real h = 0.05)
        : Material(MaterialType::Custom, props)
        , ft0_(ft0), fc0_(fc0), fb0_(fb0)
        , Gf_t_(Gf_t), Gf_c_(Gf_c), h_(h)
    {
        // alpha from biaxial ratio
        Real ratio = fb0 / fc0;
        alpha_cdp_ = (ratio - 1.0) / (2.0 * ratio - 1.0);

        // Softening parameters
        eps_t_peak_ = ft0 / props.E;
        eps_c_peak_ = fc0 / props.E;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;

        // Current damage variables
        Real kappa_t = state.history[32];
        Real kappa_c = state.history[33];
        Real d_t = state.history[34];
        Real d_c = state.history[35];

        // Effective (undamaged) elastic strain
        Real strain_e[6];
        for (int i = 0; i < 6; ++i) {
            strain_e[i] = state.strain[i] - state.history[i + 1];
        }

        // Effective stress (undamaged)
        Real sigma_eff[6];
        elastic_stress(strain_e, E, nu, sigma_eff);

        // Stress invariants of effective stress
        Real I1 = sigma_eff[0] + sigma_eff[1] + sigma_eff[2];
        Real p_mean = I1 / 3.0;
        Real s[6];
        for (int i = 0; i < 3; ++i) s[i] = sigma_eff[i] - p_mean;
        for (int i = 3; i < 6; ++i) s[i] = sigma_eff[i];

        Real J2 = 0.5 * (s[0]*s[0] + s[1]*s[1] + s[2]*s[2])
                 + s[3]*s[3] + s[4]*s[4] + s[5]*s[5];

        // Maximum principal stress (approximate: largest normal)
        Real sigma_max = sigma_eff[0];
        if (sigma_eff[1] > sigma_max) sigma_max = sigma_eff[1];
        if (sigma_eff[2] > sigma_max) sigma_max = sigma_eff[2];
        Real sigma_max_pos = sigma_max > 0.0 ? sigma_max : 0.0; // Macaulay bracket

        // Current strengths
        Real sigma_t = tensile_strength(kappa_t);
        Real sigma_c = compressive_strength(kappa_c);

        // beta parameter
        Real beta_cdp = 0.0;
        if (sigma_t > 1.0e-10) {
            beta_cdp = (sigma_c / sigma_t) * (1.0 - alpha_cdp_) - (1.0 + alpha_cdp_);
        }

        // Yield function
        Real f_yield = 0.0;
        if ((1.0 - alpha_cdp_) > 1.0e-10) {
            f_yield = (1.0 / (1.0 - alpha_cdp_))
                    * (alpha_cdp_ * I1 + Kokkos::sqrt(3.0 * J2 + 1.0e-30) + beta_cdp * sigma_max_pos)
                    - sigma_c;
        }

        if (f_yield <= 0.0) {
            // Elastic
            // Apply damage to get nominal stress
            Real d = 1.0 - (1.0 - d_t) * (1.0 - d_c);
            for (int i = 0; i < 6; ++i) {
                state.stress[i] = (1.0 - d) * sigma_eff[i];
            }
        } else {
            // Plastic return
            // Simplified: proportional return with damage evolution
            Real sqrt3J2 = Kokkos::sqrt(3.0 * J2 + 1.0e-30);
            Real denom = G * 3.0 + 1.0e-30;

            // Determine tension vs compression
            Real r_weight = 0.0;  // Weight for tension (0=pure compression, 1=pure tension)
            Real sigma_sum_abs = Kokkos::fabs(sigma_eff[0]) + Kokkos::fabs(sigma_eff[1])
                               + Kokkos::fabs(sigma_eff[2]);
            if (sigma_sum_abs > 1.0e-10) {
                Real sigma_sum_pos = (sigma_eff[0] > 0.0 ? sigma_eff[0] : 0.0)
                                   + (sigma_eff[1] > 0.0 ? sigma_eff[1] : 0.0)
                                   + (sigma_eff[2] > 0.0 ? sigma_eff[2] : 0.0);
                r_weight = sigma_sum_pos / sigma_sum_abs;
            }

            // Plastic multiplier (simplified)
            Real delta_gamma = f_yield / (3.0 * G + 1.0e-30);

            // Update internal variables
            Real eps_p_new = state.history[0] + delta_gamma;
            state.history[0] = eps_p_new;

            // Update kappa (hardening/softening variables)
            kappa_t += r_weight * delta_gamma;
            kappa_c += (1.0 - r_weight) * delta_gamma;
            state.history[32] = kappa_t;
            state.history[33] = kappa_c;

            // Update damage
            d_t = tensile_damage(kappa_t);
            d_c = compressive_damage(kappa_c);
            state.history[34] = d_t;
            state.history[35] = d_c;

            // Radial return on deviatoric
            Real scale = 1.0 - G * delta_gamma / (sqrt3J2 / Kokkos::sqrt(3.0) + 1.0e-30);
            if (scale < 0.0) scale = 0.0;

            Real sigma_returned[6];
            sigma_returned[0] = scale * s[0] + p_mean;
            sigma_returned[1] = scale * s[1] + p_mean;
            sigma_returned[2] = scale * s[2] + p_mean;
            sigma_returned[3] = scale * s[3];
            sigma_returned[4] = scale * s[4];
            sigma_returned[5] = scale * s[5];

            // Update plastic strain tensor
            if (sqrt3J2 > 1.0e-12) {
                Real factor = 1.5 * delta_gamma / (sqrt3J2 + 1.0e-30);
                for (int i = 0; i < 6; ++i) {
                    state.history[i + 1] += factor * s[i];
                }
            }

            // Apply damage
            Real d = 1.0 - (1.0 - d_t) * (1.0 - d_c);
            for (int i = 0; i < 6; ++i) {
                state.stress[i] = (1.0 - d) * sigma_returned[i];
            }
        }

        state.plastic_strain = state.history[0];
        state.damage = 1.0 - (1.0 - d_t) * (1.0 - d_c);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const Real mu = E / (2.0 * (1.0 + nu));

        Real d = state.damage;
        Real factor = 1.0 - d;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = factor * (lambda + 2.0 * mu);
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = factor * lambda;
        C[21] = C[28] = C[35] = factor * mu;
    }

    /// Tensile strength evolution
    KOKKOS_INLINE_FUNCTION
    Real tensile_strength(Real kappa_t) const {
        if (kappa_t <= 0.0) return ft0_;
        // Exponential softening
        Real eps_f = 2.0 * Gf_t_ / (ft0_ * h_ + 1.0e-30);
        Real ft = ft0_ * Kokkos::exp(-kappa_t / (eps_f + 1.0e-30));
        if (ft < ft0_ * 0.001) ft = ft0_ * 0.001;
        return ft;
    }

    /// Compressive strength evolution
    KOKKOS_INLINE_FUNCTION
    Real compressive_strength(Real kappa_c) const {
        if (kappa_c <= 0.0) return fc0_;
        // Hardening then softening
        Real eps_peak = eps_c_peak_ * 2.0;
        if (kappa_c < eps_peak) {
            // Hardening: fc increases to peak
            Real ratio = kappa_c / eps_peak;
            return fc0_ * (1.0 + 0.3 * ratio);  // 30% hardening at peak
        } else {
            // Softening
            Real eps_f = 2.0 * Gf_c_ / (fc0_ * h_ + 1.0e-30);
            Real fc = fc0_ * 1.3 * Kokkos::exp(-(kappa_c - eps_peak) / (eps_f + 1.0e-30));
            if (fc < fc0_ * 0.01) fc = fc0_ * 0.01;
            return fc;
        }
    }

    /// Tensile damage
    KOKKOS_INLINE_FUNCTION
    Real tensile_damage(Real kappa_t) const {
        if (kappa_t <= 0.0) return 0.0;
        Real eps_f = 2.0 * Gf_t_ / (ft0_ * h_ + 1.0e-30);
        Real dt = 1.0 - Kokkos::exp(-2.0 * kappa_t / (eps_f + 1.0e-30));
        if (dt < 0.0) dt = 0.0;
        if (dt > 0.99) dt = 0.99;
        return dt;
    }

    /// Compressive damage
    KOKKOS_INLINE_FUNCTION
    Real compressive_damage(Real kappa_c) const {
        if (kappa_c <= 0.0) return 0.0;
        Real eps_peak = eps_c_peak_ * 2.0;
        if (kappa_c < eps_peak) return 0.0;
        Real eps_f = 2.0 * Gf_c_ / (fc0_ * h_ + 1.0e-30);
        Real dc = 1.0 - Kokkos::exp(-2.0 * (kappa_c - eps_peak) / (eps_f + 1.0e-30));
        if (dc < 0.0) dc = 0.0;
        if (dc > 0.99) dc = 0.99;
        return dc;
    }

    /// Get biaxial ratio alpha
    KOKKOS_INLINE_FUNCTION
    Real alpha_param() const { return alpha_cdp_; }

private:
    Real ft0_, fc0_, fb0_;
    Real Gf_t_, Gf_c_, h_;
    Real alpha_cdp_;
    Real eps_t_peak_, eps_c_peak_;
};

} // namespace physics
} // namespace nxs
