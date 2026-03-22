#pragma once

/**
 * @file material_wave18.hpp
 * @brief Wave 18 material models: 20 Tier 2 constitutive models
 *
 * Models included:
 *   1.  ExplosiveBurnMaterial        - Programmed burn / detonation
 *   2.  PorousElasticMaterial        - Void ratio dependent
 *   3.  BrittleFractureMaterial      - Elastic with cracking (Rankine)
 *   4.  CreepMaterial                - Time-dependent (Norton power law)
 *   5.  KinematicHardeningMaterial   - Prager / Chaboche backstress
 *   6.  DruckerPragerMaterial        - Pressure-dependent yield
 *   7.  TabulatedCompositeMaterial   - Tabulated laminate response
 *   8.  PlyDegradationMaterial       - Composite with ply-level degradation
 *   9.  OrthotropicPlasticMaterial   - Orthotropic elastic-plastic
 *  10.  PinchingMaterial             - Cyclic loading with pinching
 *  11.  FrequencyViscoelasticMaterial- Frequency-dependent VE
 *  12.  GeneralizedViscoelasticMaterial - Multi-branch Maxwell
 *  13.  PhaseTransformationMaterial  - Austenite / martensite
 *  14.  PolynomialHardeningMaterial  - Polynomial yield curve
 *  15.  ViscoplasticThermalMaterial  - Viscoplastic with T dependence
 *  16.  PorousBrittleMaterial        - Porous elastic with cracking
 *  17.  AnisotropicCrushFoamMaterial - Anisotropic crushable foam
 *  18.  SpringHysteresisMaterial     - Spring with hysteretic loop
 *  19.  ProgrammedDetonationMaterial - CJ detonation with lighting time
 *  20.  BondedInterfaceMaterial      - Cohesive bonded contact
 */

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

// ============================================================================
// 1. ExplosiveBurnMaterial - Programmed burn / detonation
// ============================================================================

/**
 * @brief Explosive material with programmed burn model
 *
 * Burn fraction F evolves from 0 to 1 based on detonation wave arrival.
 * Pressure from a JWL-like EOS is scaled by burn fraction:
 *   P = F * P_eos + (1 - F) * P_unreacted
 *
 * Properties: E = detonation velocity (D_cj),
 *   yield_stress = Chapman-Jouguet pressure (P_cj),
 *   density = initial density, damage_threshold = lighting_time
 */
class ExplosiveBurnMaterial : public Material {
public:
    ExplosiveBurnMaterial(const MaterialProperties& props,
                          Real A_jwl = 3.712e11, Real B_jwl = 3.231e9,
                          Real R1 = 4.15, Real R2 = 0.95, Real omega = 0.30)
        : Material(MaterialType::Custom, props)
        , A_jwl_(A_jwl), B_jwl_(B_jwl), R1_(R1), R2_(R2), omega_(omega) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real D_cj = props_.E;           // Detonation velocity
        Real P_cj = props_.yield_stress; // CJ pressure
        Real rho0 = props_.density;
        Real t_light = props_.damage_threshold; // Lighting time

        // Burn fraction from history[0]
        Real F_burn = state.history[0];

        // Evolve burn fraction based on time (using dt accumulator in history[1])
        Real t_elapsed = state.history[1] + state.dt;
        state.history[1] = t_elapsed;

        if (t_elapsed >= t_light && F_burn < 1.0) {
            // Simple beta burn: F = min(1, (t - t_light) * D_cj / char_length)
            Real char_length = 0.01; // Characteristic element size
            F_burn = Kokkos::fmin(1.0, (t_elapsed - t_light) * D_cj / (char_length + 1.0e-30));
            state.history[0] = F_burn;
        }

        // Relative volume V = rho0/rho (using volumetric strain)
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real V = 1.0 + ev; // V = 1 + volumetric strain
        if (V < 0.1) V = 0.1;

        // JWL pressure for detonated products
        Real P_det = A_jwl_ * (1.0 - omega_ / (R1_ * V)) * Kokkos::exp(-R1_ * V)
                   + B_jwl_ * (1.0 - omega_ / (R2_ * V)) * Kokkos::exp(-R2_ * V);

        // Unreacted: simple bulk pressure
        Real K_unreact = rho0 * D_cj * D_cj * 0.25; // Approximate
        Real P_unreact = -K_unreact * ev;

        // Mixed pressure
        Real P = F_burn * P_det + (1.0 - F_burn) * P_unreact;

        // Hydrostatic stress (no deviatoric for explosive)
        state.stress[0] = -P;
        state.stress[1] = -P;
        state.stress[2] = -P;
        state.stress[3] = 0.0;
        state.stress[4] = 0.0;
        state.stress[5] = 0.0;

        state.damage = F_burn; // Use damage field for burn fraction
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real K = props_.density * props_.E * props_.E * 0.25;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = K;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = K;
    }

    Real burn_fraction(const MaterialState& state) const { return state.history[0]; }

private:
    Real A_jwl_, B_jwl_, R1_, R2_, omega_;
};

// ============================================================================
// 2. PorousElasticMaterial - Void ratio dependent
// ============================================================================

/**
 * @brief Porous elastic material with void ratio dependence
 *
 * Bulk modulus depends on void ratio e:
 *   K(e) = (1 + e) * p / kappa
 * where kappa is the logarithmic bulk modulus slope.
 *
 * Properties: E = reference modulus, nu = Poisson's ratio,
 *   foam_E_crush = kappa (log bulk modulus), foam_densification = initial void ratio
 */
class PorousElasticMaterial : public Material {
public:
    PorousElasticMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props)
    {
        kappa_ = props.foam_E_crush > 0.0 ? props.foam_E_crush : 0.05;
        e0_ = props.foam_densification > 0.0 ? props.foam_densification : 0.8;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];

        // Current void ratio
        Real e = e0_ + (1.0 + e0_) * ev;
        if (e < 0.01) e = 0.01;
        state.history[0] = e;

        // Effective bulk modulus
        Real p_ref = props_.E / (3.0 * (1.0 - 2.0 * props_.nu));
        Real K_eff = (1.0 + e) * p_ref / (kappa_ + 1.0e-30) * 0.01;
        if (K_eff < 1.0e3) K_eff = 1.0e3;

        Real G_eff = 3.0 * K_eff * (1.0 - 2.0 * props_.nu) / (2.0 * (1.0 + props_.nu));

        // Isotropic elastic with updated moduli
        Real lambda = K_eff - 2.0 * G_eff / 3.0;
        state.stress[0] = lambda * ev + 2.0 * G_eff * state.strain[0];
        state.stress[1] = lambda * ev + 2.0 * G_eff * state.strain[1];
        state.stress[2] = lambda * ev + 2.0 * G_eff * state.strain[2];
        state.stress[3] = G_eff * state.strain[3];
        state.stress[4] = G_eff * state.strain[4];
        state.stress[5] = G_eff * state.strain[5];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        fill_elastic_tangent(C, props_.E, props_.nu);
    }

    Real void_ratio(const MaterialState& state) const { return state.history[0]; }

private:
    Real kappa_, e0_;

    KOKKOS_INLINE_FUNCTION
    void fill_elastic_tangent(Real* C, Real E, Real nu) const {
        Real mu = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 3. BrittleFractureMaterial - Elastic with Rankine cracking
// ============================================================================

/**
 * @brief Brittle material with Rankine (max principal stress) cracking
 *
 * Elastic until max principal stress exceeds tensile strength f_t.
 * Upon cracking, stress in that direction drops to zero (smeared crack).
 *
 * Properties: yield_stress = tensile strength f_t,
 *   damage_threshold = fracture energy G_f
 */
class BrittleFractureMaterial : public Material {
public:
    BrittleFractureMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real ft = props_.yield_stress;
        Real Gf = props_.damage_threshold > 0.0 ? props_.damage_threshold : ft * ft / (2.0 * E) * 100.0;

        // Characteristic length (for mesh regularization)
        Real h = 0.01;
        Real eps_f = 2.0 * Gf / (ft * h + 1.0e-30);

        // Elastic trial stress
        elastic_stress(state.strain, E, nu, state.stress);

        // Simplified principal stress check (using sigma_xx as proxy)
        Real sigma_max = state.stress[0];
        for (int i = 1; i < 3; ++i)
            if (state.stress[i] > sigma_max) sigma_max = state.stress[i];

        // Crack state: history[0] = max reached strain, history[1-3] = crack flags
        Real eps_max = state.history[0];
        Real curr_strain_max = state.strain[0];
        for (int i = 1; i < 3; ++i)
            if (state.strain[i] > curr_strain_max) curr_strain_max = state.strain[i];

        if (curr_strain_max > eps_max) {
            eps_max = curr_strain_max;
            state.history[0] = eps_max;
        }

        Real eps_cr = ft / E;
        if (eps_max > eps_cr) {
            // Softening: damage based on strain
            Real d = (eps_max - eps_cr) / (eps_f - eps_cr + 1.0e-30);
            d = Kokkos::fmin(d, 1.0);
            if (d < 0.0) d = 0.0;
            state.damage = d;

            // Reduce tensile stresses
            for (int i = 0; i < 3; ++i) {
                if (state.stress[i] > 0.0)
                    state.stress[i] *= (1.0 - d);
            }
            // Reduce shear proportionally
            for (int i = 3; i < 6; ++i)
                state.stress[i] *= (1.0 - d);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        Real scale = 1.0 - state.damage;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = scale * lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = scale * lambda;
        C[21] = C[28] = C[35] = scale * mu;
    }
};

// ============================================================================
// 4. CreepMaterial - Norton power law creep
// ============================================================================

/**
 * @brief Norton power-law creep material
 *
 * Creep strain rate: eps_dot_cr = A * sigma^n * exp(-Q / (R*T))
 *
 * Properties: yield_stress = reference stress,
 *   JC_A = creep coefficient A, JC_n = stress exponent n,
 *   JC_C = activation energy Q, JC_T_room = gas constant R
 */
class CreepMaterial : public Material {
public:
    CreepMaterial(const MaterialProperties& props,
                  Real A_cr = 1.0e-12, Real n_cr = 3.0, Real Q_act = 0.0)
        : Material(MaterialType::Custom, props)
        , A_cr_(A_cr), n_cr_(n_cr), Q_act_(Q_act) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = props_.G;

        // Creep strain from history[7-12]
        Real strain_e[6];
        for (int i = 0; i < 6; ++i)
            strain_e[i] = state.strain[i] - state.history[i + 7];

        // Elastic trial
        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        // Von Mises equivalent
        Real sigma_vm = von_mises_stress(stress_trial);

        // Creep strain rate
        Real T = state.temperature;
        Real R_gas = 8.314;
        Real thermal_factor = 1.0;
        if (Q_act_ > 0.0 && T > 0.0)
            thermal_factor = Kokkos::exp(-Q_act_ / (R_gas * T));

        Real eps_cr_dot = A_cr_ * Kokkos::pow(Kokkos::fmax(sigma_vm, 1.0), n_cr_) * thermal_factor;

        // Creep strain increment
        Real dt = state.dt;
        Real d_eps_cr = eps_cr_dot * dt;

        // Update creep strain (deviatoric direction)
        if (sigma_vm > 1.0e-10) {
            Real p = (stress_trial[0] + stress_trial[1] + stress_trial[2]) / 3.0;
            Real factor = 1.5 * d_eps_cr / sigma_vm;
            for (int i = 0; i < 3; ++i)
                state.history[i + 7] += factor * (stress_trial[i] - p);
            for (int i = 3; i < 6; ++i)
                state.history[i + 7] += factor * stress_trial[i];
        }

        state.history[0] += d_eps_cr; // Accumulated creep strain

        // Recompute elastic strain and stress
        for (int i = 0; i < 6; ++i)
            strain_e[i] = state.strain[i] - state.history[i + 7];
        elastic_stress(strain_e, E, nu, state.stress);

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

    Real creep_strain(const MaterialState& state) const { return state.history[0]; }

private:
    Real A_cr_, n_cr_, Q_act_;
};

// ============================================================================
// 5. KinematicHardeningMaterial - Chaboche backstress
// ============================================================================

/**
 * @brief Chaboche nonlinear kinematic hardening
 *
 * Yield: f = |sigma - alpha| - sigma_y = 0
 * Backstress evolution: d_alpha = (2/3)*C*d_eps_p - gamma*alpha*d_eps_p_eq
 *
 * Properties: yield_stress = initial yield,
 *   hardening_modulus = C (kinematic modulus),
 *   tangent_modulus = gamma (recall parameter)
 */
class KinematicHardeningMaterial : public Material {
public:
    KinematicHardeningMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = props_.G;
        Real sigma_y0 = props_.yield_stress;
        Real C_kin = props_.hardening_modulus;
        Real gamma_kin = props_.tangent_modulus;

        Real eps_p_eq = state.history[0]; // Accumulated plastic strain
        // Backstress alpha stored in history[7-12]

        // Elastic trial
        Real strain_e[6];
        for (int i = 0; i < 6; ++i)
            strain_e[i] = state.strain[i] - state.history[i + 1];

        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        // Relative stress (trial) = sigma_trial - alpha
        Real xi[6];
        for (int i = 0; i < 6; ++i)
            xi[i] = stress_trial[i] - state.history[i + 7];

        // Deviatoric of relative stress
        Real p_xi = (xi[0] + xi[1] + xi[2]) / 3.0;
        Real s_xi[6];
        for (int i = 0; i < 3; ++i) s_xi[i] = xi[i] - p_xi;
        for (int i = 3; i < 6; ++i) s_xi[i] = xi[i];

        Real J2_xi = 0.5 * (s_xi[0]*s_xi[0] + s_xi[1]*s_xi[1] + s_xi[2]*s_xi[2])
                   + s_xi[3]*s_xi[3] + s_xi[4]*s_xi[4] + s_xi[5]*s_xi[5];
        Real sigma_eq = Kokkos::sqrt(3.0 * J2_xi);

        Real f = sigma_eq - sigma_y0;
        if (f <= 0.0) {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            Real denom = 3.0 * G + C_kin;
            Real delta_gamma = f / (denom + 1.0e-30);
            state.history[0] = eps_p_eq + delta_gamma;

            // Flow direction
            Real n_dir[6];
            if (sigma_eq > 1.0e-12) {
                for (int i = 0; i < 6; ++i) n_dir[i] = 1.5 * s_xi[i] / sigma_eq;
            } else {
                for (int i = 0; i < 6; ++i) n_dir[i] = 0.0;
            }

            // Update plastic strain
            for (int i = 0; i < 6; ++i)
                state.history[i + 1] += delta_gamma * n_dir[i];

            // Update backstress (Chaboche rule)
            for (int i = 0; i < 6; ++i) {
                Real d_alpha = (2.0 / 3.0) * C_kin * delta_gamma * n_dir[i]
                             - gamma_kin * state.history[i + 7] * delta_gamma;
                state.history[i + 7] += d_alpha;
            }

            // Corrected stress
            Real p_trial = (stress_trial[0] + stress_trial[1] + stress_trial[2]) / 3.0;
            Real scale = 1.0 - 3.0 * G * delta_gamma / (sigma_eq + 1.0e-30);
            if (scale < 0.0) scale = 0.0;

            Real s_trial[6];
            for (int i = 0; i < 3; ++i) s_trial[i] = stress_trial[i] - p_trial;
            for (int i = 3; i < 6; ++i) s_trial[i] = stress_trial[i];

            for (int i = 0; i < 3; ++i) state.stress[i] = scale * s_trial[i] + p_trial;
            for (int i = 3; i < 6; ++i) state.stress[i] = scale * s_trial[i];
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        fill_elastic_tangent(C);
    }

    Real backstress(const MaterialState& state, int component) const {
        return (component >= 0 && component < 6) ? state.history[component + 7] : 0.0;
    }

private:
    KOKKOS_INLINE_FUNCTION
    void fill_elastic_tangent(Real* C) const {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 6. DruckerPragerMaterial - Pressure-dependent yield
// ============================================================================

/**
 * @brief Drucker-Prager plasticity
 *
 * Yield: f = sqrt(J2) + alpha*I1 - k = 0
 * alpha = 2*sin(phi) / (sqrt(3)*(3-sin(phi)))
 * k = 6*c*cos(phi) / (sqrt(3)*(3-sin(phi)))
 *
 * Properties: yield_stress = cohesion c,
 *   damage_threshold = friction angle (deg)
 */
class DruckerPragerMaterial : public Material {
public:
    DruckerPragerMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props)
    {
        Real c = props.yield_stress;
        Real phi_deg = props.damage_threshold > 0.0 ? props.damage_threshold : 30.0;
        Real phi = phi_deg * 3.14159265358979 / 180.0;
        Real sqrt3 = 1.7320508075688772;
        alpha_ = 2.0 * Kokkos::sin(phi) / (sqrt3 * (3.0 - Kokkos::sin(phi)));
        k_ = 6.0 * c * Kokkos::cos(phi) / (sqrt3 * (3.0 - Kokkos::sin(phi)));
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = props_.G;

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

        Real J2 = 0.5 * (s[0]*s[0] + s[1]*s[1] + s[2]*s[2])
                + s[3]*s[3] + s[4]*s[4] + s[5]*s[5];
        Real sqJ2 = Kokkos::sqrt(Kokkos::fmax(J2, 0.0));

        Real k_h = k_ + props_.hardening_modulus * eps_p;
        Real f = sqJ2 + alpha_ * I1 - k_h;

        if (f <= 0.0) {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            Real denom = G + 3.0 * props_.K * alpha_ * alpha_ + props_.hardening_modulus;
            Real dlam = f / (denom + 1.0e-30);
            state.history[0] = eps_p + dlam;

            Real sc = (sqJ2 > 1.0e-12) ? 1.0 - G * dlam / sqJ2 : 1.0;
            if (sc < 0.0) sc = 0.0;
            Real p_new = p - props_.K * alpha_ * dlam;
            for (int i = 0; i < 3; ++i) state.stress[i] = sc * s[i] + p_new;
            for (int i = 3; i < 6; ++i) state.stress[i] = sc * s[i];
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        fill_elastic_tangent(C);
    }

    Real get_alpha() const { return alpha_; }
    Real get_k() const { return k_; }

private:
    Real alpha_, k_;

    KOKKOS_INLINE_FUNCTION
    void fill_elastic_tangent(Real* C) const {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 7. TabulatedCompositeMaterial - General tabulated laminate
// ============================================================================

/**
 * @brief Tabulated composite: strain-stress curves for each direction
 *
 * Uses tabulated curves for warp (1-dir) and fill (2-dir) responses.
 * Shear response can also be tabulated.
 */
class TabulatedCompositeMaterial : public Material {
public:
    TabulatedCompositeMaterial(const MaterialProperties& props,
                               const TabulatedCurve& warp_curve,
                               const TabulatedCurve& fill_curve,
                               const TabulatedCurve& shear_curve)
        : Material(MaterialType::Custom, props)
        , warp_(warp_curve), fill_(fill_curve), shear_(shear_curve) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        // Tabulated response in each direction
        state.stress[0] = warp_.evaluate(state.strain[0]);
        state.stress[1] = fill_.evaluate(state.strain[1]);
        state.stress[2] = 0.0; // Thin laminate: no through-thickness stress
        state.stress[3] = shear_.evaluate(state.strain[3]);
        state.stress[4] = 0.0;
        state.stress[5] = 0.0;

        // Track max strain for damage
        Real max_strain = Kokkos::fmax(Kokkos::fabs(state.strain[0]),
                          Kokkos::fmax(Kokkos::fabs(state.strain[1]),
                                       Kokkos::fabs(state.strain[3])));
        state.history[0] = Kokkos::fmax(state.history[0], max_strain);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        // Approximate tangent from curves at zero
        C[0] = warp_.num_points > 1 ?
               (warp_.y[1] - warp_.y[0]) / (warp_.x[1] - warp_.x[0] + 1.0e-30) : props_.E;
        C[7] = fill_.num_points > 1 ?
               (fill_.y[1] - fill_.y[0]) / (fill_.x[1] - fill_.x[0] + 1.0e-30) : props_.E;
        C[21] = shear_.num_points > 1 ?
                (shear_.y[1] - shear_.y[0]) / (shear_.x[1] - shear_.x[0] + 1.0e-30) : props_.G;
    }

private:
    TabulatedCurve warp_, fill_, shear_;
};

// ============================================================================
// 8. PlyDegradationMaterial - Composite with ply-level degradation
// ============================================================================

/**
 * @brief Composite with progressive ply degradation
 *
 * Each ply can degrade independently in fiber (1-dir) and matrix (2-dir).
 * Degradation factor d_i reduces stiffness: E_i_eff = (1 - d_i) * E_i
 *
 * History: [0] = fiber damage, [1] = matrix damage, [2] = shear damage
 */
class PlyDegradationMaterial : public Material {
public:
    PlyDegradationMaterial(const MaterialProperties& props,
                           Real fiber_strength = 0.0, Real matrix_strength = 0.0,
                           Real shear_strength = 0.0)
        : Material(MaterialType::Custom, props)
    {
        fiber_str_ = fiber_strength > 0.0 ? fiber_strength : props.yield_stress;
        matrix_str_ = matrix_strength > 0.0 ? matrix_strength : props.yield_stress * 0.3;
        shear_str_ = shear_strength > 0.0 ? shear_strength : props.yield_stress * 0.2;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E1 = props_.E1 > 0.0 ? props_.E1 : props_.E;
        Real E2 = props_.E2 > 0.0 ? props_.E2 : props_.E * 0.1;
        Real G12 = props_.G12 > 0.0 ? props_.G12 : props_.G;
        Real nu12 = props_.nu12;

        // Current damage
        Real d_fib = state.history[0];
        Real d_mat = state.history[1];
        Real d_shr = state.history[2];

        // Reduced stiffnesses
        Real E1_eff = E1 * (1.0 - d_fib);
        Real E2_eff = E2 * (1.0 - d_mat);
        Real G12_eff = G12 * (1.0 - d_shr);

        Real denom = 1.0 - nu12 * nu12 * E2_eff / (E1_eff + 1.0e-30);
        if (Kokkos::fabs(denom) < 1.0e-30) denom = 1.0;

        state.stress[0] = (E1_eff * state.strain[0] + E1_eff * nu12 * state.strain[1]) / denom;
        state.stress[1] = (E2_eff * state.strain[1] + E2_eff * nu12 * state.strain[0]) / denom;
        state.stress[2] = 0.0;
        state.stress[3] = G12_eff * state.strain[3];
        state.stress[4] = 0.0;
        state.stress[5] = 0.0;

        // Check for damage initiation/evolution
        Real sigma1 = Kokkos::fabs(state.stress[0]);
        Real sigma2 = Kokkos::fabs(state.stress[1]);
        Real tau12 = Kokkos::fabs(state.stress[3]);

        // Fiber damage
        if (sigma1 > fiber_str_ * (1.0 - d_fib + 1.0e-30)) {
            Real d_new = 1.0 - fiber_str_ / (sigma1 / (1.0 - d_fib) + 1.0e-30);
            d_new = Kokkos::fmin(Kokkos::fmax(d_new, d_fib), 0.99);
            state.history[0] = d_new;
        }
        // Matrix damage
        if (sigma2 > matrix_str_ * (1.0 - d_mat + 1.0e-30)) {
            Real d_new = 1.0 - matrix_str_ / (sigma2 / (1.0 - d_mat) + 1.0e-30);
            d_new = Kokkos::fmin(Kokkos::fmax(d_new, d_mat), 0.99);
            state.history[1] = d_new;
        }
        // Shear damage
        if (tau12 > shear_str_ * (1.0 - d_shr + 1.0e-30)) {
            Real d_new = 1.0 - shear_str_ / (tau12 / (1.0 - d_shr) + 1.0e-30);
            d_new = Kokkos::fmin(Kokkos::fmax(d_new, d_shr), 0.99);
            state.history[2] = d_new;
        }

        state.damage = Kokkos::fmax(state.history[0],
                       Kokkos::fmax(state.history[1], state.history[2]));
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        Real E1 = (props_.E1 > 0.0 ? props_.E1 : props_.E) * (1.0 - state.history[0]);
        Real E2 = (props_.E2 > 0.0 ? props_.E2 : props_.E * 0.1) * (1.0 - state.history[1]);
        Real G12 = (props_.G12 > 0.0 ? props_.G12 : props_.G) * (1.0 - state.history[2]);
        C[0] = E1;
        C[7] = E2;
        C[21] = G12;
    }

private:
    Real fiber_str_, matrix_str_, shear_str_;
};

// ============================================================================
// 9. OrthotropicPlasticMaterial - Orthotropic elastic-plastic
// ============================================================================

/**
 * @brief Orthotropic elastic-plastic material
 *
 * Elastic response is orthotropic, plasticity uses Hill48.
 * R-values from properties or default isotropic.
 */
class OrthotropicPlasticMaterial : public Material {
public:
    OrthotropicPlasticMaterial(const MaterialProperties& props,
                               Real R0 = 1.0, Real R45 = 1.0, Real R90 = 1.0)
        : Material(MaterialType::Custom, props)
        , R0_(R0), R45_(R45), R90_(R90)
    {
        F_ = R0_ / (R90_ * (1.0 + R0_));
        G_ = 1.0 / (1.0 + R0_);
        H_ = R0_ / (1.0 + R0_);
        N_ = (R0_ + R90_) * (1.0 + 2.0 * R45_) / (2.0 * R90_ * (1.0 + R0_));
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E1 = props_.E1 > 0.0 ? props_.E1 : props_.E;
        Real E2 = props_.E2 > 0.0 ? props_.E2 : props_.E;
        Real G12 = props_.G12 > 0.0 ? props_.G12 : props_.G;
        Real nu12 = props_.nu12 > 0.0 ? props_.nu12 : props_.nu;
        Real sigma_y0 = props_.yield_stress;
        Real Hmod = props_.hardening_modulus;

        Real eps_p = state.history[0];

        // Orthotropic elastic trial
        Real denom = 1.0 - nu12 * nu12 * E2 / (E1 + 1.0e-30);
        if (Kokkos::fabs(denom) < 1.0e-30) denom = 1.0;

        Real stress_trial[6];
        stress_trial[0] = (E1 * state.strain[0] + E1 * nu12 * state.strain[1]) / denom;
        stress_trial[1] = (E2 * state.strain[1] + E2 * nu12 * state.strain[0]) / denom;
        stress_trial[2] = props_.E * state.strain[2];
        stress_trial[3] = G12 * state.strain[3];
        stress_trial[4] = props_.G * state.strain[4];
        stress_trial[5] = props_.G * state.strain[5];

        // Hill equivalent stress
        Real sh = hill_eq(stress_trial);
        Real sy = sigma_y0 + Hmod * eps_p;

        if (sh <= sy || sh < 1.0e-12) {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            Real dg = (sh - sy) / (3.0 * props_.G + Hmod);
            state.history[0] = eps_p + dg;

            Real p = (stress_trial[0] + stress_trial[1] + stress_trial[2]) / 3.0;
            Real scale = sy / sh;
            for (int i = 0; i < 3; ++i)
                state.stress[i] = scale * (stress_trial[i] - p) + p;
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale * stress_trial[i];
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = props_.E1 > 0.0 ? props_.E1 : props_.E;
        C[7] = props_.E2 > 0.0 ? props_.E2 : props_.E;
        C[14] = props_.E;
        C[21] = props_.G12 > 0.0 ? props_.G12 : props_.G;
        C[28] = C[35] = props_.G;
    }

private:
    Real R0_, R45_, R90_;
    Real F_, G_, H_, N_;

    KOKKOS_INLINE_FUNCTION
    Real hill_eq(const Real* s) const {
        Real val = F_ * (s[1]-s[2])*(s[1]-s[2])
                 + G_ * (s[2]-s[0])*(s[2]-s[0])
                 + H_ * (s[0]-s[1])*(s[0]-s[1])
                 + 2.0 * 1.5 * s[4]*s[4] + 2.0 * 1.5 * s[5]*s[5] + 2.0 * N_ * s[3]*s[3];
        return Kokkos::sqrt(Kokkos::fmax(val, 0.0));
    }
};

// ============================================================================
// 10. PinchingMaterial - Cyclic loading with pinching
// ============================================================================

/**
 * @brief Material for cyclic loading with pinching behavior
 *
 * Models hysteretic behavior with pinching (reduced stiffness near zero
 * during reloading). Common for RC structures.
 *
 * History: [0] = max tensile strain, [1] = max compressive strain,
 *          [2] = loading direction flag
 */
class PinchingMaterial : public Material {
public:
    PinchingMaterial(const MaterialProperties& props, Real pinch_factor = 0.5)
        : Material(MaterialType::Custom, props)
        , pinch_(pinch_factor) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real sigma_y = props_.yield_stress;

        // Track peak strains
        Real eps_max_t = state.history[0]; // Max tensile
        Real eps_max_c = state.history[1]; // Max compressive
        Real eps_xx = state.strain[0];

        if (eps_xx > eps_max_t) { eps_max_t = eps_xx; state.history[0] = eps_max_t; }
        if (eps_xx < eps_max_c) { eps_max_c = eps_xx; state.history[1] = eps_max_c; }

        // Full elastic stress
        Real stress_full[6];
        elastic_stress(state.strain, E, nu, stress_full);

        // Determine pinching zone
        Real sigma_peak_t = E * eps_max_t;
        Real sigma_peak_c = E * eps_max_c;

        // Pinching: if reloading from opposite direction and near zero
        Real pinch_scale = 1.0;
        if (eps_max_t > 1.0e-10 && eps_xx < 0.0 && eps_xx > eps_max_c * 0.5) {
            // Unloading from tension, near zero: pinch
            Real ratio = Kokkos::fabs(eps_xx) / (Kokkos::fabs(eps_max_t) + 1.0e-30);
            pinch_scale = pinch_ + (1.0 - pinch_) * ratio;
        } else if (eps_max_c < -1.0e-10 && eps_xx > 0.0 && eps_xx < eps_max_t * 0.5) {
            // Unloading from compression, near zero: pinch
            Real ratio = eps_xx / (Kokkos::fabs(eps_max_c) + 1.0e-30);
            pinch_scale = pinch_ + (1.0 - pinch_) * ratio;
        }

        for (int i = 0; i < 6; ++i)
            state.stress[i] = stress_full[i] * pinch_scale;

        // Cap at yield
        Real vm = von_mises_stress(state.stress);
        if (vm > sigma_y) {
            Real scale = sigma_y / vm;
            Real p = (state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;
            for (int i = 0; i < 3; ++i)
                state.stress[i] = scale * (state.stress[i] - p) + p;
            for (int i = 3; i < 6; ++i)
                state.stress[i] *= scale;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        fill_elastic_tangent(C);
    }

private:
    Real pinch_;

    KOKKOS_INLINE_FUNCTION
    void fill_elastic_tangent(Real* C) const {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 11. FrequencyViscoelasticMaterial - Frequency-dependent VE
// ============================================================================

/**
 * @brief Frequency-dependent viscoelastic material
 *
 * Storage and loss moduli depend on frequency:
 *   E'(omega) = E_inf + sum_i (E_i * omega^2 * tau_i^2) / (1 + omega^2 * tau_i^2)
 *   E''(omega) = sum_i (E_i * omega * tau_i) / (1 + omega^2 * tau_i^2)
 *
 * Properties: prony_g[] and prony_tau[] for up to 4 branches
 */
class FrequencyViscoelasticMaterial : public Material {
public:
    FrequencyViscoelasticMaterial(const MaterialProperties& props, Real omega = 1.0)
        : Material(MaterialType::Custom, props)
        , omega_(omega) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E_inf = props_.E;
        int n = props_.prony_nterms;
        if (n <= 0) n = 0;
        if (n > 4) n = 4;

        // Compute effective modulus at current frequency
        Real E_storage = E_inf;
        Real E_loss = 0.0;
        for (int i = 0; i < n; ++i) {
            Real Ei = props_.prony_g[i] * E_inf;
            Real tau_i = props_.prony_tau[i];
            Real ot2 = omega_ * omega_ * tau_i * tau_i;
            E_storage += Ei * ot2 / (1.0 + ot2);
            E_loss += Ei * omega_ * tau_i / (1.0 + ot2);
        }

        Real E_eff = Kokkos::sqrt(E_storage * E_storage + E_loss * E_loss);
        Real nu = props_.nu;

        // Elastic stress with effective modulus
        Real lambda = E_eff * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E_eff / (2.0 * (1.0 + nu));
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];

        state.stress[0] = lambda * ev + 2.0 * mu * state.strain[0];
        state.stress[1] = lambda * ev + 2.0 * mu * state.strain[1];
        state.stress[2] = lambda * ev + 2.0 * mu * state.strain[2];
        state.stress[3] = mu * state.strain[3];
        state.stress[4] = mu * state.strain[4];
        state.stress[5] = mu * state.strain[5];

        // Loss factor
        state.history[0] = (E_storage > 1.0e-30) ? E_loss / E_storage : 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        fill_elastic_tangent(C);
    }

    void set_frequency(Real omega) { omega_ = omega; }
    Real loss_factor(const MaterialState& state) const { return state.history[0]; }

private:
    Real omega_;

    KOKKOS_INLINE_FUNCTION
    void fill_elastic_tangent(Real* C) const {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 12. GeneralizedViscoelasticMaterial - Multi-branch Maxwell
// ============================================================================

/**
 * @brief Generalized Maxwell model with multiple branches
 *
 * Stress = sigma_eq + sum_i sigma_i
 * where sigma_eq is equilibrium (elastic) and sigma_i are viscous branch stresses.
 * Each branch: d(sigma_i)/dt = G_i * d(eps)/dt - sigma_i / tau_i
 *
 * History: [i*6 .. i*6+5] = viscous stress tensor for branch i (up to 4 branches)
 * Uses history[0-23] for 4 branches of 6 components each
 */
class GeneralizedViscoelasticMaterial : public Material {
public:
    GeneralizedViscoelasticMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E_inf = props_.E;
        Real nu = props_.nu;
        int n = props_.prony_nterms;
        if (n <= 0) n = 0;
        if (n > 4) n = 4;

        // Equilibrium elastic stress
        Real stress_eq[6];
        elastic_stress(state.strain, E_inf, nu, stress_eq);

        Real dt = state.dt;

        // Total stress = equilibrium + sum of branch stresses
        for (int i = 0; i < 6; ++i) state.stress[i] = stress_eq[i];

        for (int b = 0; b < n; ++b) {
            Real Gi = props_.prony_g[b] * E_inf / (2.0 * (1.0 + nu));
            Real tau_i = props_.prony_tau[b];
            Real exp_dt = Kokkos::exp(-dt / (tau_i + 1.0e-30));

            for (int i = 0; i < 6; ++i) {
                int idx = b * 6 + i;
                // Recursive update: sigma_i_new = exp(-dt/tau) * sigma_i_old + Gi * (1 - exp(-dt/tau)) * 2 * strain_i
                Real sigma_old = state.history[idx];
                Real sigma_new = exp_dt * sigma_old + Gi * (1.0 - exp_dt) * 2.0 * state.strain[i];
                state.history[idx] = sigma_new;
                state.stress[i] += sigma_new;
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 13. PhaseTransformationMaterial - Austenite / martensite
// ============================================================================

/**
 * @brief Shape memory alloy with austenite-martensite phase transformation
 *
 * Simplified Auricchio model:
 *   - Austenite: high temperature, high stiffness
 *   - Martensite: transformation plateau, lower stiffness
 *   - Transformation strain eps_t ~ 0.04-0.08
 *
 * Properties: yield_stress = start of transformation stress,
 *   JC_T_melt = austenite finish (Af),
 *   JC_T_room = martensite start (Ms),
 *   damage_threshold = max transformation strain
 */
class PhaseTransformationMaterial : public Material {
public:
    PhaseTransformationMaterial(const MaterialProperties& props,
                                 Real sigma_start = 0.0, Real sigma_finish = 0.0,
                                 Real eps_max_trans = 0.06)
        : Material(MaterialType::Custom, props)
    {
        sigma_start_ = sigma_start > 0.0 ? sigma_start : props.yield_stress;
        sigma_finish_ = sigma_finish > 0.0 ? sigma_finish : sigma_start_ * 1.5;
        eps_trans_max_ = eps_max_trans;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E_a = props_.E;                    // Austenite modulus
        Real E_m = props_.E * 0.3;              // Martensite modulus
        Real nu = props_.nu;
        Real T = state.temperature;
        Real Af = props_.JC_T_melt > 0.0 ? props_.JC_T_melt : 350.0;
        Real Ms = props_.JC_T_room > 0.0 ? props_.JC_T_room : 280.0;

        // Phase fraction xi (martensite fraction) from history[0]
        Real xi = state.history[0]; // 0 = full austenite, 1 = full martensite

        // Effective modulus
        Real E_eff = (1.0 - xi) * E_a + xi * E_m;

        // Elastic trial
        Real strain_e[6];
        for (int i = 0; i < 6; ++i)
            strain_e[i] = state.strain[i] - xi * state.history[i + 1];

        Real stress_trial[6];
        elastic_stress(strain_e, E_eff, nu, stress_trial);

        Real vm = von_mises_stress(stress_trial);

        // Temperature-dependent transformation stress
        Real Cm = 8.0e6; // Clausius-Clapeyron coefficient
        Real sigma_trans = sigma_start_ + Cm * (T - Ms);
        if (sigma_trans < 0.0) sigma_trans = 0.0;

        // Forward transformation (austenite -> martensite)
        if (vm > sigma_trans && xi < 1.0) {
            Real dxi = (vm - sigma_trans) / (sigma_finish_ - sigma_start_ + 1.0e-30);
            dxi = Kokkos::fmin(dxi, 1.0 - xi);
            xi += dxi * 0.1; // Rate-limited
            if (xi > 1.0) xi = 1.0;
            state.history[0] = xi;

            // Transformation strain direction
            if (vm > 1.0e-10) {
                Real p = (stress_trial[0] + stress_trial[1] + stress_trial[2]) / 3.0;
                for (int i = 0; i < 3; ++i)
                    state.history[i + 1] = eps_trans_max_ * 1.5 * (stress_trial[i] - p) / vm;
                for (int i = 3; i < 6; ++i)
                    state.history[i + 1] = eps_trans_max_ * 1.5 * stress_trial[i] / vm;
            }
        }

        // Reverse transformation (martensite -> austenite) at high T
        if (T > Af && xi > 0.0) {
            Real dxi = (T - Af) / (Af - Ms + 1.0e-30) * 0.1;
            xi -= Kokkos::fmin(dxi, xi);
            if (xi < 0.0) xi = 0.0;
            state.history[0] = xi;
        }

        // Recompute stress with current xi
        E_eff = (1.0 - xi) * E_a + xi * E_m;
        for (int i = 0; i < 6; ++i)
            strain_e[i] = state.strain[i] - xi * state.history[i + 1];
        elastic_stress(strain_e, E_eff, nu, state.stress);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

    Real martensite_fraction(const MaterialState& state) const { return state.history[0]; }

private:
    Real sigma_start_, sigma_finish_, eps_trans_max_;
};

// ============================================================================
// 14. PolynomialHardeningMaterial - Polynomial yield curve
// ============================================================================

/**
 * @brief Polynomial hardening: sigma_y = a0 + a1*eps + a2*eps^2 + a3*eps^3
 *
 * Up to cubic polynomial for yield curve. Coefficients from properties.
 */
class PolynomialHardeningMaterial : public Material {
public:
    PolynomialHardeningMaterial(const MaterialProperties& props,
                                Real a0 = 0.0, Real a1 = 0.0,
                                Real a2 = 0.0, Real a3 = 0.0)
        : Material(MaterialType::Custom, props)
    {
        a_[0] = a0 > 0.0 ? a0 : props.yield_stress;
        a_[1] = a1 > 0.0 ? a1 : props.hardening_modulus;
        a_[2] = a2;
        a_[3] = a3;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = props_.G;

        Real eps_p = state.history[0];

        Real strain_e[6];
        for (int i = 0; i < 6; ++i)
            strain_e[i] = state.strain[i] - state.history[i + 1];

        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        Real vm = von_mises_stress(stress_trial);
        Real sy = yield_stress(eps_p);

        if (vm <= sy || vm < 1.0e-12) {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            Real H_tan = yield_stress_deriv(eps_p);
            Real dg = (vm - sy) / (3.0 * G + H_tan + 1.0e-30);
            state.history[0] = eps_p + dg;

            Real p = (stress_trial[0] + stress_trial[1] + stress_trial[2]) / 3.0;
            Real scale = 1.0 - 3.0 * G * dg / (vm + 1.0e-30);
            if (scale < 0.0) scale = 0.0;
            for (int i = 0; i < 3; ++i)
                state.stress[i] = scale * (stress_trial[i] - p) + p;
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale * stress_trial[i];
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        fill_elastic_tangent(C);
    }

    KOKKOS_INLINE_FUNCTION
    Real yield_stress(Real eps_p) const {
        return a_[0] + a_[1] * eps_p + a_[2] * eps_p * eps_p + a_[3] * eps_p * eps_p * eps_p;
    }

private:
    Real a_[4];

    KOKKOS_INLINE_FUNCTION
    Real yield_stress_deriv(Real eps_p) const {
        return a_[1] + 2.0 * a_[2] * eps_p + 3.0 * a_[3] * eps_p * eps_p;
    }

    KOKKOS_INLINE_FUNCTION
    void fill_elastic_tangent(Real* C) const {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 15. ViscoplasticThermalMaterial - Viscoplastic with T dependence
// ============================================================================

/**
 * @brief Viscoplastic material with temperature-dependent yield
 *
 * Yield: sigma_y(T) = sigma_y0 * (1 - beta * (T - T_ref))
 * Rate: sigma_flow = sigma_y + eta * eps_dot^m_rate
 *
 * Properties: yield_stress, JC_T_room = T_ref,
 *   CS_D = viscosity eta, CS_q = rate exponent m_rate,
 *   thermal_expansion = beta (thermal softening)
 */
class ViscoplasticThermalMaterial : public Material {
public:
    ViscoplasticThermalMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = props_.G;
        Real sigma_y0 = props_.yield_stress;
        Real Hmod = props_.hardening_modulus;
        Real T = state.temperature;
        Real T_ref = props_.JC_T_room;
        Real eta = props_.CS_D > 0.0 ? props_.CS_D : 0.0;
        Real m_rate = props_.CS_q > 0.0 ? props_.CS_q : 1.0;
        Real beta = props_.thermal_expansion;

        Real eps_p = state.history[0];

        // Temperature-dependent yield
        Real thermal_soft = 1.0 - beta * (T - T_ref);
        if (thermal_soft < 0.01) thermal_soft = 0.01;

        Real strain_e[6];
        for (int i = 0; i < 6; ++i)
            strain_e[i] = state.strain[i] - state.history[i + 1];

        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        Real vm = von_mises_stress(stress_trial);

        // Rate-dependent flow stress
        Real eps_dot = state.effective_strain_rate;
        Real sigma_rate = 0.0;
        if (eta > 0.0 && eps_dot > 0.0)
            sigma_rate = eta * Kokkos::pow(eps_dot, m_rate);

        Real sy = (sigma_y0 + Hmod * eps_p) * thermal_soft + sigma_rate;

        if (vm <= sy || vm < 1.0e-12) {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            Real dg = (vm - sy) / (3.0 * G + Hmod * thermal_soft + 1.0e-30);
            state.history[0] = eps_p + dg;

            Real p = (stress_trial[0] + stress_trial[1] + stress_trial[2]) / 3.0;
            Real scale = 1.0 - 3.0 * G * dg / (vm + 1.0e-30);
            if (scale < 0.0) scale = 0.0;
            for (int i = 0; i < 3; ++i)
                state.stress[i] = scale * (stress_trial[i] - p) + p;
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale * stress_trial[i];

            // Update plastic strain tensor
            if (vm > 1.0e-12) {
                Real factor = 1.5 * dg / vm;
                Real s[6];
                for (int j = 0; j < 3; ++j) s[j] = stress_trial[j] - p;
                for (int j = 3; j < 6; ++j) s[j] = stress_trial[j];
                for (int j = 0; j < 6; ++j)
                    state.history[j + 1] += factor * s[j];
            }
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        fill_elastic_tangent(C);
    }

private:
    KOKKOS_INLINE_FUNCTION
    void fill_elastic_tangent(Real* C) const {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// 16. PorousBrittleMaterial - Porous elastic with cracking
// ============================================================================

/**
 * @brief Porous brittle material combining void-ratio effects with cracking
 *
 * Bulk modulus depends on porosity. Tensile cracking via Rankine criterion.
 */
class PorousBrittleMaterial : public Material {
public:
    PorousBrittleMaterial(const MaterialProperties& props, Real porosity = 0.2)
        : Material(MaterialType::Custom, props)
        , porosity_(porosity) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real ft = props_.yield_stress;

        // Reduced modulus due to porosity (self-consistent scheme)
        Real E_eff = E * (1.0 - porosity_) * (1.0 - porosity_);

        elastic_stress(state.strain, E_eff, nu, state.stress);

        // Damage from cracking
        Real d = state.history[0];
        Real sigma_max = state.stress[0];
        for (int i = 1; i < 3; ++i)
            if (state.stress[i] > sigma_max) sigma_max = state.stress[i];

        if (sigma_max > ft * (1.0 - d)) {
            Real d_new = 1.0 - ft / (sigma_max + 1.0e-30);
            d_new = Kokkos::fmin(Kokkos::fmax(d_new, d), 0.99);
            state.history[0] = d_new;
            d = d_new;

            for (int i = 0; i < 3; ++i)
                if (state.stress[i] > 0.0) state.stress[i] *= (1.0 - d);
            for (int i = 3; i < 6; ++i)
                state.stress[i] *= (1.0 - d);
        }

        state.damage = d;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E_eff = props_.E * (1.0 - porosity_) * (1.0 - porosity_) * (1.0 - state.damage);
        Real nu = props_.nu;
        Real mu = E_eff / (2.0 * (1.0 + nu));
        Real lambda = E_eff * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

    Real porosity() const { return porosity_; }

private:
    Real porosity_;
};

// ============================================================================
// 17. AnisotropicCrushFoamMaterial - Anisotropic crushable foam
// ============================================================================

/**
 * @brief Anisotropic crushable foam with directional crush strengths
 *
 * Different crush plateau stresses in each direction (x, y, z).
 * Densification strain is common.
 *
 * Properties: foam_E_crush = x-crush stress,
 *   prony_g[0] = y-crush, prony_g[1] = z-crush
 */
class AnisotropicCrushFoamMaterial : public Material {
public:
    AnisotropicCrushFoamMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props)
    {
        crush_x_ = props.foam_E_crush > 0.0 ? props.foam_E_crush : 1.0e6;
        crush_y_ = props.prony_g[0] > 0.0 ? props.prony_g[0] : crush_x_;
        crush_z_ = props.prony_g[1] > 0.0 ? props.prony_g[1] : crush_x_;
        eps_d_ = props.foam_densification;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;

        // Elastic trial
        elastic_stress(state.strain, E, nu, state.stress);

        // Crush in each direction independently
        Real crush[3] = { crush_x_, crush_y_, crush_z_ };
        for (int i = 0; i < 3; ++i) {
            if (state.strain[i] < 0.0) { // Compression
                Real ev_abs = -state.strain[i];
                Real denom = 1.0 - ev_abs / (eps_d_ + 1.0e-30);
                if (denom < 0.01) denom = 0.01;

                Real sigma_plateau = -crush[i] * ev_abs / denom;
                // Cap elastic stress at crush plateau
                if (state.stress[i] < sigma_plateau)
                    state.stress[i] = sigma_plateau;
            }
        }

        // Track max compressive strain
        Real ev = -(state.strain[0] + state.strain[1] + state.strain[2]) / 3.0;
        if (ev > 0.0) state.history[0] = Kokkos::fmax(state.history[0], ev);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real mu = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lam2mu = lambda + 2.0 * mu;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lam2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

private:
    Real crush_x_, crush_y_, crush_z_, eps_d_;
};

// ============================================================================
// 18. SpringHysteresisMaterial - Spring with hysteretic loop
// ============================================================================

/**
 * @brief 1D spring material with hysteretic energy dissipation
 *
 * Loading/unloading follows different stiffness paths creating a loop.
 * Hysteresis ratio = area of loop / max elastic energy.
 *
 * Properties: E = loading stiffness,
 *   foam_unload_factor = unloading stiffness ratio (< 1)
 */
class SpringHysteresisMaterial : public Material {
public:
    SpringHysteresisMaterial(const MaterialProperties& props, Real unload_ratio = 0.5)
        : Material(MaterialType::Custom, props)
        , unload_ratio_(unload_ratio) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real K_load = props_.E;
        Real K_unload = K_load * unload_ratio_;

        Real eps = state.strain[0];
        Real eps_max = state.history[0]; // Max positive strain
        Real eps_min = state.history[1]; // Min negative strain
        Real eps_prev = state.history[2]; // Previous strain

        // Determine loading direction
        Real d_eps = eps - eps_prev;
        state.history[2] = eps;

        // Update peak strains
        if (eps > eps_max) { eps_max = eps; state.history[0] = eps_max; }
        if (eps < eps_min) { eps_min = eps; state.history[1] = eps_min; }

        Real sigma;
        if (d_eps >= 0.0) {
            // Loading (increasing strain)
            if (eps >= 0.0) {
                sigma = K_load * eps;
            } else {
                // Reloading from compression
                Real sigma_min = K_load * eps_min;
                sigma = sigma_min + K_unload * (eps - eps_min);
            }
        } else {
            // Unloading (decreasing strain)
            if (eps <= 0.0) {
                sigma = K_load * eps;
            } else {
                // Unloading from tension
                Real sigma_max = K_load * eps_max;
                sigma = sigma_max + K_unload * (eps - eps_max);
            }
        }

        state.stress[0] = sigma;
        for (int i = 1; i < 6; ++i) state.stress[i] = 0.0;

        // Energy dissipated
        state.history[3] += Kokkos::fabs(sigma * d_eps * (1.0 - unload_ratio_));
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = props_.E;
    }

    Real energy_dissipated(const MaterialState& state) const { return state.history[3]; }

private:
    Real unload_ratio_;
};

// ============================================================================
// 19. ProgrammedDetonationMaterial - CJ detonation with lighting time
// ============================================================================

/**
 * @brief Programmed detonation with Chapman-Jouguet conditions
 *
 * Point-initiation with spherical detonation wave propagation.
 * Lighting time = distance_to_detonation_point / D_cj
 *
 * Properties: E = D_cj (detonation velocity), yield_stress = P_cj,
 *   damage_exponent = detonation energy e_det
 */
class ProgrammedDetonationMaterial : public Material {
public:
    ProgrammedDetonationMaterial(const MaterialProperties& props,
                                  Real x_det = 0.0, Real y_det = 0.0, Real z_det = 0.0)
        : Material(MaterialType::Custom, props)
        , x_det_(x_det), y_det_(y_det), z_det_(z_det) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real D_cj = props_.E;
        Real P_cj = props_.yield_stress;
        Real rho0 = props_.density;
        Real e_det = props_.damage_exponent > 0.0 ? props_.damage_exponent : P_cj / rho0;

        // Time elapsed
        Real t = state.history[1] + state.dt;
        state.history[1] = t;

        // Distance to detonation point stored in history[2]
        Real dist = state.history[2]; // Must be set externally
        Real t_arrive = dist / (D_cj + 1.0e-30);

        // Burn fraction
        Real F = state.history[0];
        if (t >= t_arrive && F < 1.0) {
            Real dt_burn = t - t_arrive;
            F = Kokkos::fmin(1.0, dt_burn * D_cj / (0.01 + 1.0e-30));
            state.history[0] = F;
        }

        // Volumetric strain
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real V = 1.0 + ev;
        if (V < 0.1) V = 0.1;

        // CJ pressure for detonated products
        Real gamma = 2.0; // Gamma for product gases
        Real P_det = rho0 * e_det * (gamma - 1.0) / V;

        // Unreacted solid: bulk response
        Real K_solid = rho0 * D_cj * D_cj * 0.25;
        Real P_unreact = -K_solid * ev;

        Real P = F * P_det + (1.0 - F) * P_unreact;

        state.stress[0] = -P;
        state.stress[1] = -P;
        state.stress[2] = -P;
        state.stress[3] = 0.0;
        state.stress[4] = 0.0;
        state.stress[5] = 0.0;

        state.damage = F;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real K = props_.density * props_.E * props_.E * 0.25;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = K;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = K;
    }

    Real burn_fraction(const MaterialState& state) const { return state.history[0]; }

private:
    Real x_det_, y_det_, z_det_;
};

// ============================================================================
// 20. BondedInterfaceMaterial - Cohesive bonded contact
// ============================================================================

/**
 * @brief Bonded interface (adhesive joint) material
 *
 * Normal and tangential tractions with mixed-mode failure.
 * Quadratic interaction criterion: (T_n/T_n_max)^2 + (T_s/T_s_max)^2 = 1
 *
 * Properties: E = normal stiffness K_n, G = shear stiffness K_s,
 *   yield_stress = normal strength T_n_max,
 *   damage_threshold = shear strength T_s_max,
 *   damage_exponent = fracture energy G_c
 */
class BondedInterfaceMaterial : public Material {
public:
    BondedInterfaceMaterial(const MaterialProperties& props)
        : Material(MaterialType::Custom, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real K_n = props_.E;
        Real K_s = props_.G > 0.0 ? props_.G : K_n * 0.5;
        Real Tn_max = props_.yield_stress;
        Real Ts_max = props_.damage_threshold > 0.0 ? props_.damage_threshold : Tn_max * 0.7;
        Real Gc = props_.damage_exponent > 0.0 ? props_.damage_exponent : Tn_max * 0.001;

        Real d = state.history[0]; // Damage variable

        // Normal separation
        Real delta_n = state.strain[0];
        // Tangential separation
        Real delta_s = Kokkos::sqrt(state.strain[3]*state.strain[3] +
                                     state.strain[4]*state.strain[4]);

        // Effective opening
        Real delta_n_pos = Kokkos::fmax(delta_n, 0.0);
        Real delta_eff = Kokkos::sqrt(delta_n_pos * delta_n_pos + delta_s * delta_s);

        // Critical openings
        Real delta_n0 = Tn_max / K_n;
        Real delta_s0 = Ts_max / K_s;
        Real delta_0 = Kokkos::sqrt(delta_n0 * delta_n0 + delta_s0 * delta_s0);
        Real delta_f = 2.0 * Gc / (Tn_max + 1.0e-30);
        if (delta_f < delta_0 * 2.0) delta_f = delta_0 * 2.0;

        // Track max opening
        Real delta_max = state.history[1];
        if (delta_eff > delta_max) {
            delta_max = delta_eff;
            state.history[1] = delta_max;
        }

        // Compute damage
        if (delta_max >= delta_f) {
            d = 1.0;
        } else if (delta_max > delta_0) {
            d = (delta_max - delta_0) * delta_f / ((delta_f - delta_0) * delta_max + 1.0e-30);
        }
        state.history[0] = d;
        state.damage = d;

        // Tractions
        if (delta_n >= 0.0) {
            state.stress[0] = K_n * (1.0 - d) * delta_n;
        } else {
            state.stress[0] = K_n * delta_n; // Compression: no damage
        }
        state.stress[1] = 0.0;
        state.stress[2] = 0.0;
        state.stress[3] = K_s * (1.0 - d) * state.strain[3];
        state.stress[4] = K_s * (1.0 - d) * state.strain[4];
        state.stress[5] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        Real d = state.damage;
        C[0] = props_.E * (1.0 - d);
        Real Ks = props_.G > 0.0 ? props_.G : props_.E * 0.5;
        C[21] = Ks * (1.0 - d);
        C[28] = Ks * (1.0 - d);
    }
};

} // namespace physics
} // namespace nxs
