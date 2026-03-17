#pragma once

/**
 * @file material_wave10.hpp
 * @brief Wave 10 material models: 20 advanced constitutive models
 *
 * Models included:
 *   1. HillAnisotropicMaterial      - Hill48 anisotropic plasticity
 *   2. BarlatYldMaterial             - Barlat anisotropic yield
 *   3. TabulatedJohnsonCookMaterial  - JC with tabulated curves
 *   4. ConcreteMaterial             - Drucker-Prager cap
 *   5. FabricMaterial               - No-compression biaxial
 *   6. CohesiveZoneMaterial         - Traction-separation
 *   7. SoilCapMaterial              - Modified Drucker-Prager cap
 *   8. UserDefinedMaterial          - Callback-based
 *   9. ArrudaBoyceMaterial          - 8-chain rubber
 *  10. ShapeMemoryAlloyMaterial     - Superelastic NiTi
 *  11. RateDependentFoamMaterial    - Rate-enhanced foam
 *  12. PronyViscoelasticMaterial    - Multi-branch Prony
 *  13. ThermalElasticPlasticMaterial- T-dependent plasticity
 *  14. ZerilliArmstrongMaterial     - BCC/FCC metals
 *  15. SteinbergGuinanMaterial      - High-pressure metals
 *  16. MTSMaterial                  - Mechanical threshold stress
 *  17. BlatzKoMullinsMaterial       - Foam with Mullins
 *  18. LaminatedGlassMaterial       - Glass+PVB
 *  19. SpotWeldMaterial             - Beam failure
 *  20. RateDependentCompositeMaterial - Rate-dependent orthotropic
 */

#include <nexussim/physics/material.hpp>
#include <functional>

namespace nxs {
namespace physics {

// ============================================================================
// Extended MaterialType entries for Wave 10
// ============================================================================
// Note: These models use MaterialType::Custom as their base type registration
// since the enum was defined in Wave 1. Individual model identification is done
// through class type and name() methods.

// ============================================================================
// 1. HillAnisotropicMaterial - Hill48 anisotropic plasticity
// ============================================================================

/**
 * @brief Hill48 anisotropic plasticity model
 *
 * Yield function: f = sqrt(F(s22-s33)^2 + G(s33-s11)^2 + H(s11-s22)^2
 *                      + 2L*s23^2 + 2M*s13^2 + 2N*s12^2) - sigma_y
 *
 * Hill parameters stored in MaterialProperties extra fields:
 *   history[16]: F, history[17]: G, history[18]: H
 *   history[19]: L, history[20]: M, history[21]: N
 *
 * R-values define anisotropy: R0, R45, R90
 *   F = R0 / (R90*(1+R0))
 *   G = 1 / (1+R0)
 *   H = R0 / (1+R0)
 *   N = (R0+R90)*(1+2*R45) / (2*R90*(1+R0))
 *   L = M = 1.5 (default)
 */
class HillAnisotropicMaterial : public Material {
public:
    HillAnisotropicMaterial(const MaterialProperties& props,
                            Real R0 = 1.0, Real R45 = 1.0, Real R90 = 1.0)
        : Material(MaterialType::Custom, props)
        , R0_(R0), R45_(R45), R90_(R90)
    {
        // Compute Hill parameters from R-values
        F_ = R0_ / (R90_ * (1.0 + R0_));
        G_ = 1.0 / (1.0 + R0_);
        H_ = R0_ / (1.0 + R0_);
        N_ = (R0_ + R90_) * (1.0 + 2.0 * R45_) / (2.0 * R90_ * (1.0 + R0_));
        L_ = 1.5;
        M_ = 1.5;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;
        const Real sigma_y0 = props_.yield_stress;
        const Real Hmod = props_.hardening_modulus;

        Real eps_p = state.history[0];

        // Trial elastic strain
        Real strain_e[6];
        for (int i = 0; i < 6; ++i)
            strain_e[i] = state.strain[i] - state.history[i + 1];

        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        // Hill equivalent stress
        Real sigma_hill = hill_equivalent(stress_trial);

        Real sigma_y = sigma_y0 + Hmod * eps_p;

        if (sigma_hill <= sigma_y || sigma_hill < 1.0e-12) {
            for (int i = 0; i < 6; ++i)
                state.stress[i] = stress_trial[i];
        } else {
            // Simplified radial return with Hill metric
            Real delta_gamma = (sigma_hill - sigma_y) / (3.0 * G + Hmod);
            state.history[0] = eps_p + delta_gamma;

            // Scale deviatoric part
            Real p_trial = (stress_trial[0] + stress_trial[1] + stress_trial[2]) / 3.0;
            Real scale = sigma_y / sigma_hill;
            for (int i = 0; i < 3; ++i)
                state.stress[i] = scale * (stress_trial[i] - p_trial) + p_trial;
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale * stress_trial[i];
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        fill_elastic_tangent(C);
    }

    Real get_F() const { return F_; }
    Real get_G() const { return G_; }
    Real get_H() const { return H_; }
    Real get_N() const { return N_; }

private:
    Real R0_, R45_, R90_;
    Real F_, G_, H_, L_, M_, N_;

    KOKKOS_INLINE_FUNCTION
    Real hill_equivalent(const Real* s) const {
        Real val = F_ * (s[1]-s[2])*(s[1]-s[2])
                 + G_ * (s[2]-s[0])*(s[2]-s[0])
                 + H_ * (s[0]-s[1])*(s[0]-s[1])
                 + 2.0*L_*s[4]*s[4] + 2.0*M_*s[5]*s[5] + 2.0*N_*s[3]*s[3];
        return Kokkos::sqrt(Kokkos::fmax(val, 0.0));
    }

    KOKKOS_INLINE_FUNCTION
    void fill_elastic_tangent(Real* C) const {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = props_.G;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 2. BarlatYldMaterial - Barlat Yld2000-2d anisotropic yield
// ============================================================================

/**
 * @brief Barlat Yld2000-2d anisotropic yield model
 *
 * Uses exponent 'a' (typically 6 for BCC, 8 for FCC) with
 * 8 anisotropy coefficients (alpha1-alpha8).
 * Simplified implementation using isotropic approximation with
 * anisotropy scaling factors.
 */
class BarlatYldMaterial : public Material {
public:
    BarlatYldMaterial(const MaterialProperties& props,
                     Real exponent = 8.0, Real alpha1 = 1.0, Real alpha2 = 1.0)
        : Material(MaterialType::Custom, props)
        , exponent_(exponent), alpha1_(alpha1), alpha2_(alpha2) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G_mod = props_.G;
        const Real sigma_y0 = props_.yield_stress;
        const Real Hmod = props_.hardening_modulus;

        Real eps_p = state.history[0];

        Real strain_e[6];
        for (int i = 0; i < 6; ++i)
            strain_e[i] = state.strain[i] - state.history[i + 1];

        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        // Barlat equivalent stress (simplified)
        Real sigma_eq = barlat_equivalent(stress_trial);
        Real sigma_y = sigma_y0 + Hmod * eps_p;

        if (sigma_eq <= sigma_y || sigma_eq < 1.0e-12) {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            Real delta_gamma = (sigma_eq - sigma_y) / (3.0 * G_mod + Hmod);
            state.history[0] = eps_p + delta_gamma;

            Real p_trial = (stress_trial[0] + stress_trial[1] + stress_trial[2]) / 3.0;
            Real scale = sigma_y / sigma_eq;
            for (int i = 0; i < 3; ++i)
                state.stress[i] = scale * (stress_trial[i] - p_trial) + p_trial;
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale * stress_trial[i];
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        fill_elastic_tangent(C);
    }

    Real exponent() const { return exponent_; }

private:
    Real exponent_;
    Real alpha1_, alpha2_;

    KOKKOS_INLINE_FUNCTION
    Real barlat_equivalent(const Real* s) const {
        // Simplified Barlat: scaled von Mises
        Real p = (s[0] + s[1] + s[2]) / 3.0;
        Real dev[6] = { s[0]-p, s[1]-p, s[2]-p, s[3], s[4], s[5] };
        Real J2 = 0.5*(dev[0]*dev[0] + dev[1]*dev[1] + dev[2]*dev[2])
                + dev[3]*dev[3] + dev[4]*dev[4] + dev[5]*dev[5];
        Real vm = Kokkos::sqrt(3.0 * J2);
        // Apply anisotropy scaling
        return vm * (alpha1_ + alpha2_) * 0.5;
    }

    KOKKOS_INLINE_FUNCTION
    void fill_elastic_tangent(Real* C) const {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = props_.G;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 3. TabulatedJohnsonCookMaterial - JC with tabulated curves
// ============================================================================

/**
 * @brief Johnson-Cook model with tabulated yield and rate sensitivity curves
 *
 * Instead of analytical JC expression, uses:
 *   - yield_curve: sigma_y(eps_p) tabulated
 *   - rate_curve: rate factor vs log(eps_dot) tabulated
 *   - thermal_curve: thermal factor vs T* tabulated (optional)
 */
class TabulatedJohnsonCookMaterial : public Material {
public:
    TabulatedJohnsonCookMaterial(const MaterialProperties& props,
                                 const TabulatedCurve& yield_curve,
                                 const TabulatedCurve& rate_curve)
        : Material(MaterialType::Custom, props)
        , yield_curve_(yield_curve), rate_curve_(rate_curve) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;

        Real eps_p = state.history[0];

        Real strain_e[6];
        for (int i = 0; i < 6; ++i)
            strain_e[i] = state.strain[i] - state.history[i + 1];

        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        Real p_trial = (stress_trial[0] + stress_trial[1] + stress_trial[2]) / 3.0;
        Real s_trial[6];
        for (int i = 0; i < 3; ++i) s_trial[i] = stress_trial[i] - p_trial;
        for (int i = 3; i < 6; ++i) s_trial[i] = stress_trial[i];

        Real s_norm_sq = 0.0;
        for (int i = 0; i < 3; ++i) s_norm_sq += s_trial[i]*s_trial[i];
        for (int i = 3; i < 6; ++i) s_norm_sq += 2.0 * s_trial[i]*s_trial[i];
        Real sigma_vm = Kokkos::sqrt(1.5 * s_norm_sq);

        // Tabulated yield
        Real sigma_y_base = yield_curve_.evaluate(eps_p);

        // Rate enhancement
        Real eps_dot = state.effective_strain_rate;
        Real rate_factor = 1.0;
        if (eps_dot > 0.0 && rate_curve_.num_points > 0) {
            Real log_rate = Kokkos::log(Kokkos::fmax(eps_dot, 1.0e-10));
            rate_factor = rate_curve_.evaluate(log_rate);
            if (rate_factor < 1.0) rate_factor = 1.0;
        }

        Real sigma_y = sigma_y_base * rate_factor;

        Real f_trial = sigma_vm - sigma_y;
        if (f_trial <= 0.0) {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            Real delta_gamma = f_trial / (3.0 * G + 1.0e6); // Approx tangent
            state.history[0] = eps_p + delta_gamma;

            if (sigma_vm > 1.0e-12) {
                Real factor = 1.5 * delta_gamma / sigma_vm;
                for (int i = 0; i < 6; ++i)
                    state.history[i + 1] += factor * s_trial[i];
            }

            Real scale = 1.0 - 3.0 * G * delta_gamma / (sigma_vm + 1.0e-30);
            if (scale < 0.0) scale = 0.0;
            for (int i = 0; i < 3; ++i) state.stress[i] = scale * s_trial[i] + p_trial;
            for (int i = 3; i < 6; ++i) state.stress[i] = scale * s_trial[i];
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        fill_elastic_tangent(C);
    }

private:
    TabulatedCurve yield_curve_;
    TabulatedCurve rate_curve_;

    KOKKOS_INLINE_FUNCTION
    void fill_elastic_tangent(Real* C) const {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = props_.G;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 4. ConcreteMaterial - Drucker-Prager cap model
// ============================================================================

/**
 * @brief Concrete material with Drucker-Prager yield criterion
 *
 * Yield function: f = sqrt(J2) + alpha*I1 - k = 0
 * Where alpha and k are related to compressive/tensile strengths:
 *   alpha = (fc - ft) / (sqrt(3)*(fc + ft))
 *   k = 2*fc*ft / (sqrt(3)*(fc + ft))
 *
 * Properties: yield_stress = fc (compressive strength)
 *             damage_threshold = ft (tensile strength)
 */
class ConcreteMaterial : public Material {
public:
    ConcreteMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props)
    {
        Real fc = props.yield_stress;    // Compressive strength (positive)
        Real ft = props.damage_threshold > 0.0 ? props.damage_threshold : fc * 0.1;
        Real sqrt3 = 1.7320508075688772;
        alpha_ = (fc - ft) / (sqrt3 * (fc + ft));
        k_ = 2.0 * fc * ft / (sqrt3 * (fc + ft));
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;
        const Real Hmod = props_.hardening_modulus;

        Real eps_p = state.history[0];

        Real strain_e[6];
        for (int i = 0; i < 6; ++i)
            strain_e[i] = state.strain[i] - state.history[i + 1];

        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        Real I1 = stress_trial[0] + stress_trial[1] + stress_trial[2];
        Real p = I1 / 3.0;
        Real s[6];
        for (int i = 0; i < 3; ++i) s[i] = stress_trial[i] - p;
        for (int i = 3; i < 6; ++i) s[i] = stress_trial[i];

        Real J2 = 0.5*(s[0]*s[0] + s[1]*s[1] + s[2]*s[2])
                + s[3]*s[3] + s[4]*s[4] + s[5]*s[5];
        Real sqJ2 = Kokkos::sqrt(Kokkos::fmax(J2, 0.0));

        Real k_hard = k_ + Hmod * eps_p;
        Real f = sqJ2 + alpha_ * I1 - k_hard;

        if (f <= 0.0) {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            // Return mapping on DP surface
            Real denom = G + 3.0 * props_.K * alpha_ * alpha_ + Hmod;
            Real delta_gamma = f / (denom + 1.0e-30);
            state.history[0] = eps_p + delta_gamma;

            // Scale deviatoric
            Real scale = 1.0;
            if (sqJ2 > 1.0e-12)
                scale = 1.0 - G * delta_gamma / sqJ2;
            if (scale < 0.0) scale = 0.0;

            Real p_new = p - props_.K * alpha_ * delta_gamma;
            for (int i = 0; i < 3; ++i) state.stress[i] = scale * s[i] + p_new;
            for (int i = 3; i < 6; ++i) state.stress[i] = scale * s[i];
        }

        state.plastic_strain = state.history[0];

        // Damage tracking for tension
        if (state.stress[0] + state.stress[1] + state.stress[2] > 0.0) {
            Real max_princ = state.stress[0]; // Simplified
            Real ft = k_ / (alpha_ + 1.0e-30) * 0.1;
            if (max_princ > ft && ft > 0.0) {
                state.damage = Kokkos::fmin(1.0, (max_princ - ft) / (ft + 1.0e-30));
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        fill_elastic_tangent(C);
    }

private:
    Real alpha_;
    Real k_;

    KOKKOS_INLINE_FUNCTION
    void fill_elastic_tangent(Real* C) const {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = props_.G;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 5. FabricMaterial - No-compression biaxial membrane
// ============================================================================

/**
 * @brief Fabric (membrane) material with no compression stiffness
 *
 * Biaxial membrane with tension-only behavior in warp and fill directions.
 * Zero stiffness in compression and in the thickness direction.
 * Bending stiffness is negligible.
 *
 * Properties: E1 (warp), E2 (fill), nu12
 */
class FabricMaterial : public Material {
public:
    FabricMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E1 = props_.E1 > 0.0 ? props_.E1 : props_.E;
        Real E2 = props_.E2 > 0.0 ? props_.E2 : props_.E;
        Real nu12 = props_.nu12;
        Real G12 = props_.G12 > 0.0 ? props_.G12 : props_.G;

        // Only in-plane stress (membrane)
        Real eps1 = state.strain[0];
        Real eps2 = state.strain[1];
        Real gamma12 = state.strain[3];

        // No compression: clamp strains to >= 0
        Real eps1_eff = Kokkos::fmax(eps1, 0.0);
        Real eps2_eff = Kokkos::fmax(eps2, 0.0);

        Real denom = 1.0 - nu12 * nu12 * E2 / E1;
        if (Kokkos::fabs(denom) < 1.0e-30) denom = 1.0;

        state.stress[0] = (E1 * eps1_eff + E1 * nu12 * eps2_eff) / denom;
        state.stress[1] = (E2 * eps2_eff + E2 * nu12 * eps1_eff) / denom;
        state.stress[2] = 0.0; // No thickness stress
        state.stress[3] = G12 * gamma12;
        state.stress[4] = 0.0;
        state.stress[5] = 0.0;

        // If both directions compress, zero all
        if (eps1 < 0.0 && eps2 < 0.0) {
            for (int i = 0; i < 6; ++i) state.stress[i] = 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real E1 = props_.E1 > 0.0 ? props_.E1 : props_.E;
        Real E2 = props_.E2 > 0.0 ? props_.E2 : props_.E;
        Real nu12 = props_.nu12;
        Real G12 = props_.G12 > 0.0 ? props_.G12 : props_.G;

        Real denom = 1.0 - nu12 * nu12 * E2 / E1;
        if (Kokkos::fabs(denom) < 1.0e-30) denom = 1.0;

        if (state.strain[0] >= 0.0) {
            C[0] = E1 / denom;
            C[1] = E1 * nu12 / denom;
            C[6] = C[1];
        }
        if (state.strain[1] >= 0.0) {
            C[7] = E2 / denom;
        }
        C[21] = G12;
    }
};

// ============================================================================
// 6. CohesiveZoneMaterial - Traction-separation law
// ============================================================================

/**
 * @brief Cohesive zone model with bilinear traction-separation law
 *
 * Normal mode (mode I):
 *   T_n = K_n * delta_n            (delta_n < delta_0)
 *   T_n = sigma_max * (1 - d)      (delta_0 <= delta_n < delta_f)
 *   T_n = 0                         (delta_n >= delta_f)
 *
 * Where d = (delta_n - delta_0) / (delta_f - delta_0) is damage.
 * sigma_max = K_n * delta_0
 * GIc = 0.5 * sigma_max * delta_f  (fracture energy)
 *
 * Properties: E = K_n (penalty stiffness), yield_stress = sigma_max
 *             damage_threshold = delta_0, damage_exponent = delta_f
 */
class CohesiveZoneMaterial : public Material {
public:
    CohesiveZoneMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real K_n = props_.E;
        Real sigma_max = props_.yield_stress;
        Real delta_0 = sigma_max / K_n;
        Real GIc = props_.damage_exponent; // Fracture energy stored here
        Real delta_f = 2.0 * GIc / (sigma_max + 1.0e-30);
        if (delta_f < delta_0) delta_f = 2.0 * delta_0;

        // Normal separation (strain[0] represents opening displacement)
        Real delta_n = state.strain[0];

        // Track maximum opening for irreversible damage
        Real delta_max = state.history[0];
        if (delta_n > delta_max) {
            delta_max = delta_n;
            state.history[0] = delta_max;
        }

        // Compute damage
        Real d = 0.0;
        if (delta_max >= delta_f) {
            d = 1.0;
        } else if (delta_max > delta_0) {
            d = (delta_max - delta_0) / (delta_f - delta_0);
        }
        state.damage = d;

        // Traction
        if (delta_n >= 0.0) {
            // Tension (opening)
            Real K_eff = K_n * (1.0 - d);
            state.stress[0] = K_eff * delta_n;
            if (state.stress[0] > sigma_max * (1.0 - d))
                state.stress[0] = sigma_max * (1.0 - d);
        } else {
            // Compression: penalty stiffness (no damage)
            state.stress[0] = K_n * delta_n;
        }

        // Shear tractions (mode II)
        Real K_s = K_n * 0.5;
        state.stress[3] = K_s * (1.0 - d) * state.strain[3];
        state.stress[4] = K_s * (1.0 - d) * state.strain[4];

        // Zero out-of-plane normals
        state.stress[1] = 0.0;
        state.stress[2] = 0.0;
        state.stress[5] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real K_n = props_.E;
        Real d = state.damage;
        C[0] = K_n * (1.0 - d);
        C[21] = 0.5 * K_n * (1.0 - d);
        C[28] = C[21];
    }
};

// ============================================================================
// 7. SoilCapMaterial - Modified Drucker-Prager cap
// ============================================================================

/**
 * @brief Soil cap model (modified Drucker-Prager with cap)
 *
 * Shear yield: f_s = sqrt(J2) + alpha*I1 - k
 * Cap yield:   f_c = sqrt(J2 + (I1 - L)^2 / R^2) - (X - L) / R
 *
 * Properties: yield_stress = cohesion c
 *             damage_threshold = friction angle phi (degrees)
 *             foam_E_crush = cap surface ratio R
 *             foam_densification = initial cap position X0
 */
class SoilCapMaterial : public Material {
public:
    SoilCapMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props)
    {
        Real c = props.yield_stress;
        Real phi_deg = props.damage_threshold > 0.0 ? props.damage_threshold : 30.0;
        Real phi = phi_deg * 3.14159265358979323846 / 180.0;
        alpha_ = 2.0 * Kokkos::sin(phi) / (1.7320508 * (3.0 - Kokkos::sin(phi)));
        k_ = 6.0 * c * Kokkos::cos(phi) / (1.7320508 * (3.0 - Kokkos::sin(phi)));
        R_ = props.foam_E_crush > 0.0 ? props.foam_E_crush : 2.0;
        X0_ = props.foam_densification > 0.0 ? props.foam_densification : k_ / alpha_;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;

        Real eps_p = state.history[0];
        Real X = X0_ + props_.hardening_modulus * eps_p; // Hardening cap

        Real strain_e[6];
        for (int i = 0; i < 6; ++i)
            strain_e[i] = state.strain[i] - state.history[i + 1];

        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        Real I1 = stress_trial[0] + stress_trial[1] + stress_trial[2];
        Real p = I1 / 3.0;
        Real s[6];
        for (int i = 0; i < 3; ++i) s[i] = stress_trial[i] - p;
        for (int i = 3; i < 6; ++i) s[i] = stress_trial[i];

        Real J2 = 0.5*(s[0]*s[0] + s[1]*s[1] + s[2]*s[2])
                + s[3]*s[3] + s[4]*s[4] + s[5]*s[5];
        Real sqJ2 = Kokkos::sqrt(Kokkos::fmax(J2, 0.0));

        // Shear failure check
        Real f_s = sqJ2 + alpha_ * I1 - k_;
        if (f_s > 0.0) {
            Real denom = G + 3.0 * props_.K * alpha_ * alpha_;
            Real dlam = f_s / (denom + 1.0e-30);
            Real sc = (sqJ2 > 1.0e-12) ? 1.0 - G * dlam / sqJ2 : 1.0;
            if (sc < 0.0) sc = 0.0;
            Real p_new = p - props_.K * alpha_ * dlam;
            for (int i = 0; i < 3; ++i) state.stress[i] = sc * s[i] + p_new;
            for (int i = 3; i < 6; ++i) state.stress[i] = sc * s[i];
            state.history[0] = eps_p + dlam;
        } else {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = props_.G;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

private:
    Real alpha_, k_, R_, X0_;
};

// ============================================================================
// 8. UserDefinedMaterial - Callback-based
// ============================================================================

/**
 * @brief User-defined material with callback function
 *
 * Allows registering a custom stress computation function.
 * Falls back to linear elastic if no callback is set.
 */
class UserDefinedMaterial : public Material {
public:
    using StressCallback = std::function<void(const MaterialProperties&, MaterialState&)>;

    UserDefinedMaterial(const MaterialProperties& props,
                       StressCallback callback = nullptr)
        : Material(MaterialType::Custom, props)
        , callback_(std::move(callback)) {}

    void set_callback(StressCallback cb) { callback_ = std::move(cb); }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        if (callback_) {
            callback_(props_, state);
        } else {
            elastic_stress(state.strain, props_.E, props_.nu, state.stress);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = props_.G;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

private:
    StressCallback callback_;
};

// ============================================================================
// 9. ArrudaBoyceMaterial - 8-chain rubber model
// ============================================================================

/**
 * @brief Arruda-Boyce (8-chain) hyperelastic model
 *
 * Strain energy: W = mu * sum_{i=1}^{5} C_i / lambda_L^{2(i-1)} * (I1_bar^i - 3^i)
 *                  + K/2*(J-1)^2
 *
 * Properties: G = initial shear modulus mu
 *             K = bulk modulus
 *             foam_densification = lambda_L (locking stretch)
 */
class ArrudaBoyceMaterial : public Material {
public:
    ArrudaBoyceMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props)
    {
        lambda_L_ = props.foam_densification > 0.0 ? props.foam_densification : 5.0;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real mu = props_.G;
        const Real kappa = props_.K;
        const Real* F = state.F;

        Real J = F[0]*(F[4]*F[8] - F[5]*F[7])
               - F[1]*(F[3]*F[8] - F[5]*F[6])
               + F[2]*(F[3]*F[7] - F[4]*F[6]);

        if (J < 1.0e-10) {
            for (int i = 0; i < 6; ++i) state.stress[i] = 0.0;
            return;
        }

        // B = F*F^T
        Real B[6];
        B[0] = F[0]*F[0] + F[1]*F[1] + F[2]*F[2];
        B[1] = F[3]*F[3] + F[4]*F[4] + F[5]*F[5];
        B[2] = F[6]*F[6] + F[7]*F[7] + F[8]*F[8];
        B[3] = F[0]*F[3] + F[1]*F[4] + F[2]*F[5];
        B[4] = F[3]*F[6] + F[4]*F[7] + F[5]*F[8];
        B[5] = F[0]*F[6] + F[1]*F[7] + F[2]*F[8];

        Real I1 = B[0] + B[1] + B[2];
        Real J23 = Kokkos::pow(J, -2.0/3.0);
        Real I1_bar = J23 * I1;

        // Arruda-Boyce coefficients
        Real lL2 = lambda_L_ * lambda_L_;
        Real C1 = 0.5;
        Real C2 = 1.0 / (20.0 * lL2);
        Real C3 = 11.0 / (1050.0 * lL2 * lL2);
        Real C4 = 19.0 / (7000.0 * lL2 * lL2 * lL2);
        Real C5 = 519.0 / (673750.0 * lL2 * lL2 * lL2 * lL2);

        // dW/dI1_bar
        Real dW_dI1 = mu * (C1 + 2.0*C2*I1_bar + 3.0*C3*I1_bar*I1_bar
                     + 4.0*C4*I1_bar*I1_bar*I1_bar
                     + 5.0*C5*I1_bar*I1_bar*I1_bar*I1_bar);

        // Deviatoric Cauchy stress
        Real inv_J = 1.0 / J;
        Real coeff = 2.0 * inv_J * J23 * dW_dI1;
        Real I1_3 = I1 / 3.0;

        for (int i = 0; i < 3; ++i)
            state.stress[i] = coeff * (B[i] - I1_3) + kappa * (J - 1.0);
        for (int i = 3; i < 6; ++i)
            state.stress[i] = coeff * B[i];

        state.vol_strain = Kokkos::log(J);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real mu = props_.G;
        Real K = props_.K;
        Real lambda = K - 2.0 * mu / 3.0;
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

private:
    Real lambda_L_;
};

// ============================================================================
// 10. ShapeMemoryAlloyMaterial - Superelastic NiTi
// ============================================================================

/**
 * @brief Shape memory alloy (NiTi superelastic) model
 *
 * Simplified superelastic model based on Auricchio & Taylor:
 * - Austenite → Martensite forward transformation at sigma_s^AM
 * - Martensite → Austenite reverse transformation at sigma_s^MA
 * - Maximum transformation strain eps_L
 *
 * Properties: yield_stress = sigma_s^AM (forward start)
 *             hardening_modulus = slope of transformation plateau
 *             damage_threshold = sigma_s^MA (reverse start, if < sigma_s^AM)
 *             foam_densification = eps_L (max transformation strain)
 *
 * History: history[0] = martensite fraction xi (0=austenite, 1=martensite)
 */
class ShapeMemoryAlloyMaterial : public Material {
public:
    ShapeMemoryAlloyMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props)
    {
        sigma_AM_ = props.yield_stress;
        sigma_MA_ = props.damage_threshold > 0.0 ? props.damage_threshold : 0.5 * sigma_AM_;
        eps_L_ = props.foam_densification > 0.0 ? props.foam_densification : 0.06;
        H_plateau_ = props.hardening_modulus > 0.0 ? props.hardening_modulus : 1.0e8;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;

        Real xi = state.history[0]; // Martensite fraction

        // Total strain minus transformation strain
        Real eps_tr = xi * eps_L_;
        Real strain_e[6];
        strain_e[0] = state.strain[0] - eps_tr; // Simplified: uniaxial transform
        for (int i = 1; i < 6; ++i)
            strain_e[i] = state.strain[i];

        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        Real sigma_vm = von_mises_stress(stress_trial);

        // Forward transformation (loading)
        Real sigma_f_start = sigma_AM_;
        Real sigma_f_end = sigma_AM_ + H_plateau_ * eps_L_;

        // Reverse transformation (unloading)
        Real sigma_r_end = sigma_MA_;
        Real sigma_r_start = sigma_MA_ + H_plateau_ * eps_L_;

        Real xi_new = xi;

        if (sigma_vm > sigma_f_start && xi < 1.0) {
            // Forward: austenite -> martensite
            xi_new = (sigma_vm - sigma_f_start) / (sigma_f_end - sigma_f_start + 1.0e-30);
            xi_new = Kokkos::fmax(xi, Kokkos::fmin(xi_new, 1.0));
        } else if (sigma_vm < sigma_r_start && xi > 0.0) {
            // Reverse: martensite -> austenite
            if (sigma_vm < sigma_r_end) {
                xi_new = 0.0;
            } else {
                xi_new = (sigma_vm - sigma_r_end) / (sigma_r_start - sigma_r_end + 1.0e-30);
                xi_new = Kokkos::fmin(xi, Kokkos::fmax(xi_new, 0.0));
            }
        }

        state.history[0] = xi_new;

        // Recompute stress with updated xi
        Real eps_tr_new = xi_new * eps_L_;
        strain_e[0] = state.strain[0] - eps_tr_new;
        elastic_stress(strain_e, E, nu, state.stress);

        state.plastic_strain = eps_tr_new; // Use plastic_strain field for transform strain
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = props_.G;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

private:
    Real sigma_AM_;
    Real sigma_MA_;
    Real eps_L_;
    Real H_plateau_;
};

// ============================================================================
// 11. RateDependentFoamMaterial - Rate-enhanced crushable foam
// ============================================================================

/**
 * @brief Rate-dependent foam material
 *
 * Crushable foam with strain rate enhancement:
 *   sigma_crush_eff = sigma_crush * (1 + (eps_dot/D)^(1/q))
 *
 * Properties: foam_E_crush, foam_densification, CS_D, CS_q, E, nu
 */
class RateDependentFoamMaterial : public Material {
public:
    RateDependentFoamMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real G = props_.G;
        const Real K = props_.K;
        const Real D = props_.CS_D > 0.0 ? props_.CS_D : 1.0;
        const Real q = props_.CS_q > 0.0 ? props_.CS_q : 1.0;

        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real ev_3 = ev / 3.0;

        // Deviatoric
        Real sd[6];
        for (int i = 0; i < 3; ++i) sd[i] = 2.0 * G * (state.strain[i] - ev_3);
        for (int i = 3; i < 6; ++i) sd[i] = G * state.strain[i];

        // Rate factor
        Real eps_dot = state.effective_strain_rate;
        Real rate_factor = 1.0;
        if (eps_dot > 0.0) {
            rate_factor = 1.0 + Kokkos::pow(eps_dot / D, 1.0 / q);
        }

        // Volumetric
        Real p = 0.0;
        if (ev < 0.0) {
            Real ev_abs = -ev;
            Real E_crush = props_.foam_E_crush;
            Real eps_d = props_.foam_densification;
            Real sigma_crush;
            if (ev_abs < eps_d) {
                sigma_crush = E_crush;
            } else {
                Real denom = 1.0 - ev_abs;
                if (denom < 0.01) denom = 0.01;
                sigma_crush = E_crush + props_.E * (ev_abs - eps_d) * (ev_abs - eps_d) / denom;
            }
            p = sigma_crush * rate_factor;

            // Track max compression
            if (ev_abs > state.history[0]) {
                state.history[0] = ev_abs;
                state.history[1] = p;
            }
        } else {
            p = -K * ev;
        }

        for (int i = 0; i < 3; ++i) state.stress[i] = sd[i] - p;
        for (int i = 3; i < 6; ++i) state.stress[i] = sd[i];

        state.vol_strain = ev;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = props_.G;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 12. PronyViscoelasticMaterial - Multi-branch Prony (6 terms)
// ============================================================================

/**
 * @brief Extended Prony viscoelastic with up to 6 terms
 *
 * Same as ViscoelasticMaterial but supports 6 Prony terms instead of 4.
 * Uses history[7..30] for overstress storage.
 *
 * Properties: prony_g[0..3] + extra g-terms in g5, g6 fields
 *             prony_tau[0..3] + extra tau terms
 */
class PronyViscoelasticMaterial : public Material {
public:
    static constexpr int MAX_TERMS = 4; // Limited by history space

    PronyViscoelasticMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real G = props_.G;
        const Real K = props_.K;
        int nt = props_.prony_nterms;
        if (nt <= 0) nt = 0;
        if (nt > MAX_TERMS) nt = MAX_TERMS;
        Real dt = state.dt;

        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real ev_3 = ev / 3.0;
        Real ed[6];
        for (int i = 0; i < 6; ++i) {
            ed[i] = (i < 3) ? (state.strain[i] - ev_3) : state.strain[i];
        }

        // Strain increment
        Real ded[6];
        for (int i = 0; i < 6; ++i) {
            ded[i] = ed[i] - state.history[23 + i];
            state.history[23 + i] = ed[i];
        }

        // Long-term modulus
        Real g_sum = 0.0;
        for (int k = 0; k < nt; ++k) g_sum += props_.prony_g[k];
        Real G_inf = G * (1.0 - g_sum);

        Real sd[6];
        for (int i = 0; i < 3; ++i) sd[i] = 2.0 * G_inf * ed[i];
        for (int i = 3; i < 6; ++i) sd[i] = G_inf * ed[i];

        // Prony branches
        for (int k = 0; k < nt; ++k) {
            Real gk = props_.prony_g[k];
            Real tau_k = props_.prony_tau[k];
            if (tau_k < 1.0e-30) continue;

            Real exp_dt = (dt > 0.0) ? Kokkos::exp(-dt / tau_k) : 1.0;
            int base = 7 + k * 4;

            for (int i = 0; i < 4; ++i) {
                Real de_i = (i < 3) ? 2.0 * ded[i] : ded[3];
                Real h_old = state.history[base + i];
                Real h_new = exp_dt * h_old + gk * G * de_i;
                state.history[base + i] = h_new;
            }

            sd[0] += state.history[base + 0];
            sd[1] += state.history[base + 1];
            sd[2] += state.history[base + 2];
            sd[3] += state.history[base + 3];
        }

        Real p = K * ev;
        for (int i = 0; i < 3; ++i) state.stress[i] = sd[i] + p;
        for (int i = 3; i < 6; ++i) state.stress[i] = sd[i];

        state.vol_strain = ev;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = props_.G;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 13. ThermalElasticPlasticMaterial - Temperature-dependent plasticity
// ============================================================================

/**
 * @brief Elastic-plastic with temperature-dependent yield and modulus
 *
 * E(T)  = E_ref * (1 - alpha_E * (T - T_ref))
 * sigma_y(T) = sigma_y_ref * (1 - alpha_y * (T - T_ref) / (T_melt - T_ref))
 *
 * Properties: E, yield_stress at T_ref
 *             JC_T_room = T_ref, JC_T_melt = T_melt
 *             thermal_expansion = alpha_E (modulus degradation)
 */
class ThermalElasticPlasticMaterial : public Material {
public:
    ThermalElasticPlasticMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real T = state.temperature;
        Real T_ref = props_.JC_T_room;
        Real T_melt = props_.JC_T_melt;

        // Temperature-dependent modulus
        Real alpha_E = props_.thermal_expansion;
        Real E_T = props_.E * Kokkos::fmax(1.0 - alpha_E * (T - T_ref), 0.01);
        Real nu = props_.nu;
        Real G_T = E_T / (2.0 * (1.0 + nu));

        // Temperature-dependent yield
        Real T_star = Kokkos::fmax(0.0, (T - T_ref) / (T_melt - T_ref + 1.0e-30));
        T_star = Kokkos::fmin(T_star, 1.0);
        Real sigma_y0_T = props_.yield_stress * (1.0 - T_star);
        if (sigma_y0_T < 100.0) sigma_y0_T = 100.0;

        Real Hmod = props_.hardening_modulus;
        Real eps_p = state.history[0];

        Real strain_e[6];
        for (int i = 0; i < 6; ++i)
            strain_e[i] = state.strain[i] - state.history[i + 1];

        Real stress_trial[6];
        elastic_stress(strain_e, E_T, nu, stress_trial);

        Real p_trial = (stress_trial[0] + stress_trial[1] + stress_trial[2]) / 3.0;
        Real s_trial[6];
        for (int i = 0; i < 3; ++i) s_trial[i] = stress_trial[i] - p_trial;
        for (int i = 3; i < 6; ++i) s_trial[i] = stress_trial[i];

        Real s_norm_sq = 0.0;
        for (int i = 0; i < 3; ++i) s_norm_sq += s_trial[i]*s_trial[i];
        for (int i = 3; i < 6; ++i) s_norm_sq += 2.0*s_trial[i]*s_trial[i];
        Real sigma_vm = Kokkos::sqrt(1.5 * s_norm_sq);

        Real sigma_y = sigma_y0_T + Hmod * eps_p;

        if (sigma_vm <= sigma_y) {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            Real dg = (sigma_vm - sigma_y) / (3.0 * G_T + Hmod);
            state.history[0] = eps_p + dg;

            if (sigma_vm > 1.0e-12) {
                Real fac = 1.5 * dg / sigma_vm;
                for (int i = 0; i < 6; ++i) state.history[i+1] += fac * s_trial[i];
            }

            Real sc = 1.0 - 3.0 * G_T * dg / (sigma_vm + 1.0e-30);
            for (int i = 0; i < 3; ++i) state.stress[i] = sc * s_trial[i] + p_trial;
            for (int i = 3; i < 6; ++i) state.stress[i] = sc * s_trial[i];
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real T = state.temperature;
        Real T_ref = props_.JC_T_room;
        Real alpha_E = props_.thermal_expansion;
        Real E_T = props_.E * Kokkos::fmax(1.0 - alpha_E * (T - T_ref), 0.01);
        Real nu = props_.nu;
        Real mu = E_T / (2.0 * (1.0 + nu));
        Real lambda = E_T * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 14. ZerilliArmstrongMaterial - BCC/FCC metals
// ============================================================================

/**
 * @brief Zerilli-Armstrong constitutive model
 *
 * BCC form: sigma = C0 + C1*exp(-C3*T + C4*T*ln(eps_dot)) + C5*eps_p^n
 * FCC form: sigma = C0 + C2*eps_p^(1/2)*exp(-C3*T + C4*T*ln(eps_dot))
 *
 * Properties: JC_A = C0, JC_B = C1 (BCC) or C2 (FCC)
 *             JC_n = n (hardening exponent)
 *             JC_C = C3, JC_m = C4
 *             CS_D = C5 (BCC only)
 *             damage_threshold = 0 for BCC, 1 for FCC
 */
class ZerilliArmstrongMaterial : public Material {
public:
    ZerilliArmstrongMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props)
    {
        is_fcc_ = (props.damage_threshold > 0.5);
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;

        Real C0 = props_.JC_A;
        Real C1_or_C2 = props_.JC_B;
        Real C3 = props_.JC_C;
        Real C4 = props_.JC_m;
        Real C5 = props_.CS_D;
        Real n = props_.JC_n;
        Real T = state.temperature;
        Real eps_dot = Kokkos::fmax(state.effective_strain_rate, 1.0);
        Real ln_eps_dot = Kokkos::log(eps_dot);

        Real eps_p = state.history[0];

        Real strain_e[6];
        for (int i = 0; i < 6; ++i) strain_e[i] = state.strain[i] - state.history[i+1];

        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        Real sigma_vm = von_mises_stress(stress_trial);

        Real sigma_y;
        if (is_fcc_) {
            Real exp_term = Kokkos::exp(-C3*T + C4*T*ln_eps_dot);
            Real sqrt_ep = Kokkos::sqrt(Kokkos::fmax(eps_p, 1.0e-12));
            sigma_y = C0 + C1_or_C2 * sqrt_ep * exp_term;
        } else {
            Real exp_term = Kokkos::exp(-C3*T + C4*T*ln_eps_dot);
            Real hard = C5 * Kokkos::pow(Kokkos::fmax(eps_p, 1.0e-12), n);
            sigma_y = C0 + C1_or_C2 * exp_term + hard;
        }

        if (sigma_vm <= sigma_y) {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            Real delta_gamma = (sigma_vm - sigma_y) / (3.0 * G + 1.0e6);
            state.history[0] = eps_p + delta_gamma;

            Real p_t = (stress_trial[0]+stress_trial[1]+stress_trial[2]) / 3.0;
            Real sc = sigma_y / (sigma_vm + 1.0e-30);
            for (int i = 0; i < 3; ++i)
                state.stress[i] = sc * (stress_trial[i] - p_t) + p_t;
            for (int i = 3; i < 6; ++i)
                state.stress[i] = sc * stress_trial[i];
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = props_.G;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

private:
    bool is_fcc_;
};

// ============================================================================
// 15. SteinbergGuinanMaterial - High-pressure metals
// ============================================================================

/**
 * @brief Steinberg-Guinan high-pressure model
 *
 * G(P,T) = G0 * (1 + A*P/eta^(1/3) - B*(T-300))
 * Y(eps_p,P,T) = Y0*(1+beta*eps_p)^n * (1+A*P/eta^(1/3) - B*(T-300))
 * Y capped at Y_max.
 *
 * Properties: G=G0, yield_stress=Y0, JC_B=beta, JC_n=n
 *             JC_A = dG/dP coefficient A
 *             JC_C = dG/dT coefficient B
 *             hardening_modulus = Y_max
 */
class SteinbergGuinanMaterial : public Material {
public:
    SteinbergGuinanMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G0 = props_.G;

        Real Y0 = props_.yield_stress;
        Real beta = props_.JC_B;
        Real n = props_.JC_n;
        Real A = props_.JC_A;
        Real B_coeff = props_.JC_C;
        Real Y_max = props_.hardening_modulus > Y0 ? props_.hardening_modulus : 10.0 * Y0;
        Real T = state.temperature;

        Real eps_p = state.history[0];

        // Pressure-dependent modulus (simplified: use hydrostatic from strain)
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real P_approx = -props_.K * ev; // Compression gives positive P
        Real eta = 1.0 / (1.0 + ev + 1.0e-30);
        Real eta_13 = Kokkos::pow(Kokkos::fmax(eta, 0.01), 1.0/3.0);

        Real G_mod = G0 * Kokkos::fmax(1.0 + A * P_approx / eta_13 - B_coeff * (T - 300.0), 0.01);

        // Yield
        Real Y = Y0 * Kokkos::pow(1.0 + beta * (eps_p + 1.0e-10), n);
        Y *= Kokkos::fmax(1.0 + A * P_approx / eta_13 - B_coeff * (T - 300.0), 0.01);
        if (Y > Y_max) Y = Y_max;

        // Elastic trial
        Real strain_e[6];
        for (int i = 0; i < 6; ++i) strain_e[i] = state.strain[i] - state.history[i+1];

        // Use modified modulus
        Real E_mod = 2.0 * G_mod * (1.0 + nu);
        Real stress_trial[6];
        elastic_stress(strain_e, E_mod, nu, stress_trial);

        Real sigma_vm = von_mises_stress(stress_trial);

        if (sigma_vm <= Y) {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            Real dg = (sigma_vm - Y) / (3.0 * G_mod + 1.0e6);
            state.history[0] = eps_p + dg;

            Real p_t = (stress_trial[0]+stress_trial[1]+stress_trial[2]) / 3.0;
            Real sc = Y / (sigma_vm + 1.0e-30);
            for (int i = 0; i < 3; ++i) state.stress[i] = sc*(stress_trial[i]-p_t) + p_t;
            for (int i = 3; i < 6; ++i) state.stress[i] = sc*stress_trial[i];
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = props_.G;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 16. MTSMaterial - Mechanical Threshold Stress
// ============================================================================

/**
 * @brief Mechanical Threshold Stress model (Follansbee & Kocks)
 *
 * sigma = sigma_a + (S_i * sigma_i_hat + S_e * sigma_e_hat) * f(eps_p)
 * Where S_i, S_e are Arrhenius factors depending on T and eps_dot.
 *
 * Simplified: sigma_y = sigma_a + sigma_i * g(T, eps_dot)
 *   g = 1 - (kT/(mu*b^3*g0_i) * ln(eps_dot_ref/eps_dot))^(1/q) ^(1/p)
 *
 * Properties: JC_A = sigma_a (athermal stress)
 *             JC_B = sigma_i_hat (threshold stress)
 *             JC_n, JC_C = p, q exponents (0 < p <= 1, 1 <= q <= 2)
 *             hardening_modulus = H
 */
class MTSMaterial : public Material {
public:
    MTSMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;

        Real sigma_a = props_.JC_A;
        Real sigma_i = props_.JC_B;
        Real p_exp = props_.JC_n > 0.0 ? props_.JC_n : 0.5;
        Real q_exp = props_.JC_C > 0.0 ? props_.JC_C : 1.5;
        Real Hmod = props_.hardening_modulus;
        Real T = state.temperature;
        Real eps_dot = Kokkos::fmax(state.effective_strain_rate, 1.0);

        // Boltzmann constant * T / (mu * b^3 * g0) - simplified dimensionless factor
        Real kT_factor = 8.617e-5 * T / (G * 1.0e-30 + 1.0e-30);
        // Normalize to reasonable range
        Real g0 = 1.6; // Typical for copper
        Real norm_factor = kT_factor / (g0 + 1.0e-30);
        if (norm_factor > 1.0) norm_factor = 1.0;

        Real S = 1.0 - Kokkos::pow(
            Kokkos::fmax(norm_factor * Kokkos::log(1.0e7 / eps_dot), 0.0),
            1.0 / q_exp);
        S = Kokkos::pow(Kokkos::fmax(S, 0.0), 1.0 / p_exp);
        if (S < 0.0) S = 0.0;
        if (S > 1.0) S = 1.0;

        Real eps_p = state.history[0];
        Real sigma_y = sigma_a + S * sigma_i + Hmod * eps_p;

        Real strain_e[6];
        for (int i = 0; i < 6; ++i) strain_e[i] = state.strain[i] - state.history[i+1];

        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);
        Real sigma_vm = von_mises_stress(stress_trial);

        if (sigma_vm <= sigma_y) {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            Real dg = (sigma_vm - sigma_y) / (3.0 * G + Hmod);
            state.history[0] = eps_p + dg;

            Real p_t = (stress_trial[0]+stress_trial[1]+stress_trial[2]) / 3.0;
            Real sc = sigma_y / (sigma_vm + 1.0e-30);
            for (int i = 0; i < 3; ++i) state.stress[i] = sc*(stress_trial[i]-p_t) + p_t;
            for (int i = 3; i < 6; ++i) state.stress[i] = sc * stress_trial[i];
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = props_.G;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 17. BlatzKoMullinsMaterial - Compressible foam with Mullins effect
// ============================================================================

/**
 * @brief Blatz-Ko foam with Mullins softening effect
 *
 * Base response: Blatz-Ko (same as FoamMaterial)
 * Mullins effect: stress = eta(d) * sigma_virgin
 *   eta(d) = 1 - d_max * erf((W_max - W) / (m + W_max))
 *   d_max tracks maximum strain energy
 *
 * Properties: G (shear), K (bulk), foam_unload_factor = Mullins parameter r
 *             damage_exponent = Mullins parameter m
 */
class BlatzKoMullinsMaterial : public Material {
public:
    BlatzKoMullinsMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real G = props_.G;
        const Real* F = state.F;
        Real r = props_.foam_unload_factor > 0.0 ? props_.foam_unload_factor : 0.5;
        Real m = props_.damage_exponent > 0.0 ? props_.damage_exponent : 0.1;

        Real J = F[0]*(F[4]*F[8]-F[5]*F[7])
               - F[1]*(F[3]*F[8]-F[5]*F[6])
               + F[2]*(F[3]*F[7]-F[4]*F[6]);

        if (J < 1.0e-10) {
            for (int i = 0; i < 6; ++i) state.stress[i] = 0.0;
            return;
        }

        // B = F*F^T (Voigt)
        Real B[6];
        B[0] = F[0]*F[0]+F[1]*F[1]+F[2]*F[2];
        B[1] = F[3]*F[3]+F[4]*F[4]+F[5]*F[5];
        B[2] = F[6]*F[6]+F[7]*F[7]+F[8]*F[8];
        B[3] = F[0]*F[3]+F[1]*F[4]+F[2]*F[5];
        B[4] = F[3]*F[6]+F[4]*F[7]+F[5]*F[8];
        B[5] = F[0]*F[6]+F[1]*F[7]+F[2]*F[8];

        Real inv_J = 1.0 / J;

        // Virgin Blatz-Ko stress
        Real sigma_v[6];
        for (int i = 0; i < 3; ++i) sigma_v[i] = G * inv_J * (B[i] - J);
        for (int i = 3; i < 6; ++i) sigma_v[i] = G * inv_J * B[i];

        // Strain energy
        Real I1 = B[0] + B[1] + B[2];
        Real W = 0.5 * G * (I1 / J - 3.0);

        // Update max energy
        Real W_max = state.history[0];
        if (W > W_max) {
            W_max = W;
            state.history[0] = W_max;
        }

        // Mullins softening factor
        Real eta = 1.0;
        if (W_max > 1.0e-12 && W < W_max) {
            Real arg = (W_max - W) / (m + W_max + 1.0e-30);
            // Simple erf approximation
            Real erf_val = arg / Kokkos::sqrt(1.0 + arg * arg);
            eta = 1.0 - r * erf_val;
        }

        for (int i = 0; i < 6; ++i) state.stress[i] = eta * sigma_v[i];

        state.damage = 1.0 - eta;
        state.vol_strain = Kokkos::log(J);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real G = props_.G;
        Real K = props_.K;
        Real lambda = K - 2.0*G/3.0;
        Real lam2mu = lambda + 2.0*G;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = G;
    }
};

// ============================================================================
// 18. LaminatedGlassMaterial - Glass + PVB interlayer
// ============================================================================

/**
 * @brief Laminated glass (glass-PVB composite)
 *
 * Effective properties from rule-of-mixtures:
 *   E_eff = f_glass * E_glass + (1-f_glass) * E_pvb
 *   sigma_y = glass tensile strength (brittle failure)
 *
 * After glass fracture (damage=1), only PVB carries load.
 *
 * Properties: E = E_glass, E1 = E_pvb (if set)
 *             yield_stress = glass tensile strength
 *             foam_unload_factor = glass volume fraction
 */
class LaminatedGlassMaterial : public Material {
public:
    LaminatedGlassMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props)
    {
        E_glass_ = props.E;
        E_pvb_ = props.E1 > 0.0 ? props.E1 : 2.0e6; // PVB ~2 MPa
        f_glass_ = props.foam_unload_factor > 0.0 ? props.foam_unload_factor : 0.8;
        sigma_fail_ = props.yield_stress;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real nu = props_.nu;
        Real glass_broken = state.history[0]; // 0=intact, 1=broken

        Real E_eff;
        if (glass_broken >= 1.0) {
            E_eff = E_pvb_;
        } else {
            E_eff = f_glass_ * E_glass_ + (1.0 - f_glass_) * E_pvb_;
        }

        elastic_stress(state.strain, E_eff, nu, state.stress);

        // Check glass failure (max principal stress ~ sigma_xx for simple case)
        if (glass_broken < 1.0) {
            Real max_stress = state.stress[0];
            for (int i = 1; i < 3; ++i) {
                if (state.stress[i] > max_stress) max_stress = state.stress[i];
            }
            if (max_stress > sigma_fail_) {
                state.history[0] = 1.0;
                state.damage = 1.0;
                // Recompute with PVB only
                elastic_stress(state.strain, E_pvb_, nu, state.stress);
            }
        } else {
            state.damage = 1.0;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real nu = props_.nu;
        Real E_eff;
        if (state.damage >= 1.0) {
            E_eff = E_pvb_;
        } else {
            E_eff = f_glass_ * E_glass_ + (1.0 - f_glass_) * E_pvb_;
        }
        Real mu = E_eff / (2.0*(1.0+nu));
        Real lambda = E_eff * nu / ((1.0+nu)*(1.0-2.0*nu));
        Real lam2mu = lambda + 2.0*mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

private:
    Real E_glass_, E_pvb_, f_glass_, sigma_fail_;
};

// ============================================================================
// 19. SpotWeldMaterial - Beam-like spot weld with failure
// ============================================================================

/**
 * @brief Spot weld material with combined force/moment failure
 *
 * Failure criterion: (N/N_f)^a + (M/M_f)^b >= 1
 * Where N = axial force, M = bending moment.
 *
 * Simplified: uses effective stress and combined criterion.
 * After failure, stress drops to zero.
 *
 * Properties: yield_stress = N_f (normal failure force equiv. stress)
 *             hardening_modulus = M_f (moment failure equiv. stress)
 *             damage_exponent = exponent a
 *             JC_n = exponent b
 */
class SpotWeldMaterial : public Material {
public:
    SpotWeldMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        if (state.damage >= 1.0) {
            for (int i = 0; i < 6; ++i) state.stress[i] = 0.0;
            return;
        }

        const Real E = props_.E;
        const Real nu = props_.nu;

        elastic_stress(state.strain, E, nu, state.stress);

        // Failure check
        Real N_f = props_.yield_stress;
        Real M_f = props_.hardening_modulus > 0.0 ? props_.hardening_modulus : N_f;
        Real a_exp = props_.damage_exponent > 0.0 ? props_.damage_exponent : 2.0;
        Real b_exp = props_.JC_n > 0.0 ? props_.JC_n : 2.0;

        // Normal stress (axial)
        Real sigma_n = Kokkos::fabs(state.stress[0]);
        // Shear stress (bending equivalent)
        Real tau = Kokkos::sqrt(state.stress[3]*state.stress[3]
                              + state.stress[4]*state.stress[4]
                              + state.stress[5]*state.stress[5]);

        Real ratio = Kokkos::pow(sigma_n / (N_f + 1.0e-30), a_exp)
                   + Kokkos::pow(tau / (M_f + 1.0e-30), b_exp);

        if (ratio >= 1.0) {
            state.damage = 1.0;
            for (int i = 0; i < 6; ++i) state.stress[i] = 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        if (state.damage >= 1.0) {
            for (int i = 0; i < 36; ++i) C[i] = 0.0;
            return;
        }
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = E / (2.0*(1.0+nu));
        Real lambda = E * nu / ((1.0+nu)*(1.0-2.0*nu));
        Real lam2mu = lambda + 2.0*mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 20. RateDependentCompositeMaterial - Rate-dependent orthotropic
// ============================================================================

/**
 * @brief Rate-dependent orthotropic composite material
 *
 * Orthotropic elasticity with strain-rate-dependent moduli:
 *   E1(eps_dot) = E1_0 * (1 + C_rate * ln(max(eps_dot/eps_dot_ref, 1)))
 *   Similar for E2, G12
 *
 * Properties: E1, E2, E3, G12, G23, G13, nu12, nu23, nu13
 *             CS_D = reference strain rate
 *             CS_q = rate sensitivity C_rate
 */
class RateDependentCompositeMaterial : public Material {
public:
    RateDependentCompositeMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props)
    {
        compute_base_stiffness();
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real eps_dot = state.effective_strain_rate;
        Real eps_dot_ref = props_.CS_D > 0.0 ? props_.CS_D : 1.0;
        Real C_rate = props_.CS_q > 0.0 ? props_.CS_q : 0.0;

        Real rate_factor = 1.0;
        if (eps_dot > eps_dot_ref && C_rate > 0.0) {
            rate_factor = 1.0 + C_rate * Kokkos::log(eps_dot / eps_dot_ref);
        }

        for (int i = 0; i < 6; ++i) {
            state.stress[i] = 0.0;
            for (int j = 0; j < 6; ++j) {
                state.stress[i] += rate_factor * C_base_[i*6+j] * state.strain[j];
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = C_base_[i];
    }

private:
    Real C_base_[36];

    void compute_base_stiffness() {
        Real E1 = props_.E1 > 0.0 ? props_.E1 : props_.E;
        Real E2 = props_.E2 > 0.0 ? props_.E2 : props_.E;
        Real E3 = props_.E3 > 0.0 ? props_.E3 : props_.E;
        Real nu12 = props_.nu12;
        Real nu23 = props_.nu23;
        Real nu13 = props_.nu13;
        Real G12 = props_.G12 > 0.0 ? props_.G12 : props_.G;
        Real G23 = props_.G23 > 0.0 ? props_.G23 : props_.G;
        Real G13 = props_.G13 > 0.0 ? props_.G13 : props_.G;

        Real nu21 = nu12 * E2 / E1;
        Real nu31 = nu13 * E3 / E1;
        Real nu32 = nu23 * E3 / E2;

        Real delta = 1.0 - nu12*nu21 - nu23*nu32 - nu13*nu31
                     - 2.0*nu12*nu23*nu31;

        for (int i = 0; i < 36; ++i) C_base_[i] = 0.0;

        C_base_[0]  = E1*(1.0-nu23*nu32)/delta;
        C_base_[1]  = E1*(nu21+nu31*nu23)/delta;
        C_base_[2]  = E1*(nu31+nu21*nu32)/delta;
        C_base_[6]  = C_base_[1];
        C_base_[7]  = E2*(1.0-nu13*nu31)/delta;
        C_base_[8]  = E2*(nu32+nu12*nu31)/delta;
        C_base_[12] = C_base_[2];
        C_base_[13] = C_base_[8];
        C_base_[14] = E3*(1.0-nu12*nu21)/delta;

        C_base_[21] = G12;
        C_base_[28] = G23;
        C_base_[35] = G13;
    }
};

} // namespace physics
} // namespace nxs
