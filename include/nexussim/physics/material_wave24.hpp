#pragma once

/**
 * @file material_wave24.hpp
 * @brief Wave 24 material models: 10 advanced constitutive models
 *
 * Models included:
 *   1.  JohnsonHolmquist1Material      - JH1 ceramics/glass impact
 *   2.  JohnsonHolmquist2Material      - JH2 improved continuous damage
 *   3.  MultiSurfaceConcreteMaterial   - 3-surface plasticity + damage
 *   4.  GranularSoilCapMaterial        - Drucker-Prager with cap hardening
 *   5.  Barlat2000Material             - Yld2000 8-parameter anisotropic yield
 *   6.  ChabocheKinHardeningMaterial   - Multi-backstress cyclic plasticity
 *   7.  ScaledCrushFoamMaterial        - Density-dependent crushable foam
 *   8.  ThermoplasticPolymerMaterial   - Rate/temperature dependent polymer
 *   9.  UnifiedCreepMaterial           - Plasticity + Norton creep + thermal
 *  10.  AdvancedFabricMaterial         - Biaxial fabric with permanent set
 */

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

// ============================================================================
// 1. JohnsonHolmquist1Material (JH1) - Ceramics/Glass Impact
// ============================================================================

/**
 * @brief Johnson-Holmquist 1 ceramic/glass impact model
 *
 * Intact strength: sigma_i = A*(P* + T*)^N (normalized)
 * Fracture strength: sigma_f = B*(P*)^M
 * Damage: D = sum(delta_eps_p / eps_f(P)), eps_f(P) = D1*(P* + T*)^D2
 * P* = P/P_HEL, T* = T/P_HEL
 *
 * History: [32]=damage, [33]=bulking pressure
 */
class JohnsonHolmquist1Material : public Material {
public:
    JohnsonHolmquist1Material(const MaterialProperties& props,
                               Real A = 0.93, Real N = 0.77,
                               Real B = 0.31, Real M = 0.85,
                               Real T_norm = 0.15, Real P_HEL = 1.46e9,
                               Real D1 = 0.005, Real D2 = 1.0,
                               Real sigma_HEL = 2.0e9)
        : Material(MaterialType::Custom, props)
        , A_(A), N_(N), B_(B), M_(M), T_norm_(T_norm)
        , P_HEL_(P_HEL), D1_(D1), D2_(D2), sigma_HEL_(sigma_HEL) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));

        // Volumetric strain and pressure
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real P = -K_bulk * ev;

        // Deviatoric strain
        Real e_dev[6];
        Real ev_third = ev / 3.0;
        e_dev[0] = state.strain[0] - ev_third;
        e_dev[1] = state.strain[1] - ev_third;
        e_dev[2] = state.strain[2] - ev_third;
        e_dev[3] = state.strain[3];
        e_dev[4] = state.strain[4];
        e_dev[5] = state.strain[5];

        // Deviatoric stress (trial)
        Real s_dev[6];
        s_dev[0] = 2.0 * G * e_dev[0];
        s_dev[1] = 2.0 * G * e_dev[1];
        s_dev[2] = 2.0 * G * e_dev[2];
        s_dev[3] = 2.0 * G * e_dev[3];
        s_dev[4] = 2.0 * G * e_dev[4];
        s_dev[5] = 2.0 * G * e_dev[5];

        // Von Mises equivalent stress (trial)
        Real J2 = 0.5 * (s_dev[0] * s_dev[0] + s_dev[1] * s_dev[1] + s_dev[2] * s_dev[2])
                 + s_dev[3] * s_dev[3] + s_dev[4] * s_dev[4] + s_dev[5] * s_dev[5];
        Real sigma_vm = Kokkos::sqrt(3.0 * J2 + 1.0e-30);

        // Normalized pressure and tensile strength
        Real P_star = P / (P_HEL_ + 1.0e-30);
        Real T_star = T_norm_;

        // Intact strength (normalized, then scaled by sigma_HEL)
        Real arg_intact = P_star + T_star;
        if (arg_intact < 0.0) arg_intact = 0.0;
        Real sigma_i = A_ * Kokkos::pow(arg_intact + 1.0e-30, N_) * sigma_HEL_;

        // Fracture strength
        Real P_star_pos = Kokkos::fmax(P_star, 0.0);
        Real sigma_f = B_ * Kokkos::pow(P_star_pos + 1.0e-30, M_) * sigma_HEL_;

        // Get current damage
        Real D = state.history[32];

        // Current strength: switch between intact and fractured
        Real sigma_strength;
        if (D < 1.0) {
            sigma_strength = sigma_i * (1.0 - D) + sigma_f * D;
        } else {
            sigma_strength = sigma_f;
        }
        if (sigma_strength < 0.0) sigma_strength = 0.0;

        // Plastic strain increment for damage
        Real delta_eps_p = 0.0;
        if (sigma_vm > sigma_strength && sigma_strength > 0.0) {
            // Scale deviatoric stress to strength
            Real scale = sigma_strength / sigma_vm;
            for (int i = 0; i < 6; ++i) s_dev[i] *= scale;
            delta_eps_p = (sigma_vm - sigma_strength) / (3.0 * G + 1.0e-30);
            sigma_vm = sigma_strength;
        }

        // Fracture strain
        Real arg_frac = P_star + T_star;
        if (arg_frac < 0.0) arg_frac = 0.0;
        Real eps_f = D1_ * Kokkos::pow(arg_frac + 1.0e-30, D2_);
        if (eps_f < 1.0e-10) eps_f = 1.0e-10;

        // Damage accumulation
        D += delta_eps_p / eps_f;
        if (D > 1.0) D = 1.0;
        state.history[32] = D;

        // Bulking pressure: energy from damage goes to pressure increase
        Real dU_bulking = delta_eps_p * sigma_vm * 0.5;
        Real bulking_P = state.history[33];
        bulking_P += dU_bulking / (K_bulk * 1.0e-6 + 1.0e-30);
        state.history[33] = bulking_P;
        P += bulking_P;

        // Assemble total stress = deviatoric + hydrostatic
        state.stress[0] = s_dev[0] - P;
        state.stress[1] = s_dev[1] - P;
        state.stress[2] = s_dev[2] - P;
        state.stress[3] = s_dev[3];
        state.stress[4] = s_dev[4];
        state.stress[5] = s_dev[5];

        state.damage = D;
        state.plastic_strain += delta_eps_p;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = lam + 2.0 * mu; C[1] = lam;           C[2] = lam;
        C[6] = lam;            C[7] = lam + 2.0 * mu; C[8] = lam;
        C[12] = lam;           C[13] = lam;           C[14] = lam + 2.0 * mu;
        C[21] = mu; C[28] = mu; C[35] = mu;
    }

    Real get_damage(const MaterialState& state) const { return state.history[32]; }

private:
    Real A_, N_, B_, M_, T_norm_, P_HEL_, D1_, D2_, sigma_HEL_;
};

// ============================================================================
// 2. JohnsonHolmquist2Material (JH2) - Improved Continuous Damage
// ============================================================================

/**
 * @brief Johnson-Holmquist 2 ceramic model with continuous damage
 *
 * Continuous damage: sigma* = sigma_i* - D*(sigma_i* - sigma_f*)
 * Same damage law as JH1 but with continuous softening path.
 * Bulking energy: dP = -K1*mu + sqrt((K1*mu)^2 + 2*K1*dU)
 *
 * History: [32]=damage, [33]=bulking_energy
 */
class JohnsonHolmquist2Material : public Material {
public:
    JohnsonHolmquist2Material(const MaterialProperties& props,
                               Real A = 0.93, Real N = 0.77,
                               Real B = 0.31, Real M = 0.85,
                               Real T_norm = 0.15, Real P_HEL = 1.46e9,
                               Real D1 = 0.005, Real D2 = 1.0,
                               Real sigma_HEL = 2.0e9,
                               Real K1 = 130.0e9, Real K2 = 0.0, Real K3 = 0.0)
        : Material(MaterialType::Custom, props)
        , A_(A), N_(N), B_(B), M_(M), T_norm_(T_norm)
        , P_HEL_(P_HEL), D1_(D1), D2_(D2), sigma_HEL_(sigma_HEL)
        , K1_(K1), K2_(K2), K3_(K3) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));

        // Volumetric strain and pressure
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real mu_comp = -ev; // Compression positive
        Real P = K_bulk * mu_comp;

        // Add polynomial EOS terms for high pressure
        if (mu_comp > 0.0) {
            P = K1_ * mu_comp + K2_ * mu_comp * mu_comp + K3_ * mu_comp * mu_comp * mu_comp;
        }

        // Deviatoric strain and stress
        Real ev_third = ev / 3.0;
        Real s_dev[6];
        s_dev[0] = 2.0 * G * (state.strain[0] - ev_third);
        s_dev[1] = 2.0 * G * (state.strain[1] - ev_third);
        s_dev[2] = 2.0 * G * (state.strain[2] - ev_third);
        s_dev[3] = 2.0 * G * state.strain[3];
        s_dev[4] = 2.0 * G * state.strain[4];
        s_dev[5] = 2.0 * G * state.strain[5];

        Real J2 = 0.5 * (s_dev[0] * s_dev[0] + s_dev[1] * s_dev[1] + s_dev[2] * s_dev[2])
                 + s_dev[3] * s_dev[3] + s_dev[4] * s_dev[4] + s_dev[5] * s_dev[5];
        Real sigma_vm = Kokkos::sqrt(3.0 * J2 + 1.0e-30);

        // Normalized quantities
        Real P_star = P / (P_HEL_ + 1.0e-30);
        Real T_star = T_norm_;

        // Intact strength (normalized)
        Real arg_intact = Kokkos::fmax(P_star + T_star, 0.0);
        Real sigma_i_star = A_ * Kokkos::pow(arg_intact + 1.0e-30, N_);

        // Fracture strength (normalized)
        Real P_star_pos = Kokkos::fmax(P_star, 0.0);
        Real sigma_f_star = B_ * Kokkos::pow(P_star_pos + 1.0e-30, M_);

        // Current damage
        Real D = state.history[32];

        // JH2 continuous damage: sigma* = sigma_i* - D*(sigma_i* - sigma_f*)
        Real sigma_star = sigma_i_star - D * (sigma_i_star - sigma_f_star);
        if (sigma_star < 0.0) sigma_star = 0.0;
        Real sigma_strength = sigma_star * sigma_HEL_;

        // Plastic strain increment
        Real delta_eps_p = 0.0;
        if (sigma_vm > sigma_strength && sigma_strength > 0.0) {
            Real scale = sigma_strength / sigma_vm;
            for (int i = 0; i < 6; ++i) s_dev[i] *= scale;
            delta_eps_p = (sigma_vm - sigma_strength) / (3.0 * G + 1.0e-30);
            sigma_vm = sigma_strength;
        }

        // Fracture strain
        Real arg_frac = Kokkos::fmax(P_star + T_star, 0.0);
        Real eps_f = D1_ * Kokkos::pow(arg_frac + 1.0e-30, D2_);
        if (eps_f < 1.0e-10) eps_f = 1.0e-10;

        // Damage accumulation
        D += delta_eps_p / eps_f;
        if (D > 1.0) D = 1.0;
        state.history[32] = D;

        // Bulking energy contribution
        Real dU = 0.5 * delta_eps_p * sigma_vm;
        Real U_bulk = state.history[33] + dU;
        state.history[33] = U_bulk;

        // Bulking pressure correction
        Real dP_bulk = 0.0;
        if (U_bulk > 0.0 && K1_ > 0.0) {
            Real inner = K1_ * mu_comp * K1_ * mu_comp + 2.0 * K1_ * U_bulk;
            if (inner > 0.0) {
                dP_bulk = -K1_ * mu_comp + Kokkos::sqrt(inner);
            }
        }
        P += dP_bulk;

        // Assemble total stress
        state.stress[0] = s_dev[0] - P;
        state.stress[1] = s_dev[1] - P;
        state.stress[2] = s_dev[2] - P;
        state.stress[3] = s_dev[3];
        state.stress[4] = s_dev[4];
        state.stress[5] = s_dev[5];

        state.damage = D;
        state.plastic_strain += delta_eps_p;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = lam + 2.0 * mu; C[1] = lam;           C[2] = lam;
        C[6] = lam;            C[7] = lam + 2.0 * mu; C[8] = lam;
        C[12] = lam;           C[13] = lam;           C[14] = lam + 2.0 * mu;
        C[21] = mu; C[28] = mu; C[35] = mu;
    }

    Real get_damage(const MaterialState& state) const { return state.history[32]; }
    Real get_bulking_energy(const MaterialState& state) const { return state.history[33]; }

private:
    Real A_, N_, B_, M_, T_norm_, P_HEL_, D1_, D2_, sigma_HEL_;
    Real K1_, K2_, K3_;
};

// ============================================================================
// 3. MultiSurfaceConcreteMaterial - 3-Surface Plasticity + Damage
// ============================================================================

/**
 * @brief Multi-surface concrete model with tensile/shear/compressive yield
 *
 * Three yield surfaces:
 *   - Tensile cap (Rankine): f_t = sigma_max - f_t
 *   - Shear (Drucker-Prager): f_s = sqrt(J2) + alpha*I1 - k
 *   - Compressive cap: f_c = sqrt(J2) + alpha_c*(I1 - L) - beta_c
 *
 * Damage: d = 1 - (1-d_t)*(1-d_c)
 * History: [32]=damage_tension, [33]=damage_compression,
 *          [34]=max_tensile_strain, [35]=max_compress_strain
 */
class MultiSurfaceConcreteMaterial : public Material {
public:
    MultiSurfaceConcreteMaterial(const MaterialProperties& props,
                                  Real ft = 3.0e6, Real fc = 30.0e6,
                                  Real Gf_t = 100.0, Real Gf_c = 10000.0,
                                  Real alpha_dp = 0.2, Real lchar = 0.05)
        : Material(MaterialType::Custom, props)
        , ft_(ft), fc_(fc), Gf_t_(Gf_t), Gf_c_(Gf_c)
        , alpha_dp_(alpha_dp), lchar_(lchar) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real lam = K_bulk - 2.0 * G / 3.0;

        // Elastic trial stress
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        Real ev_third = ev / 3.0;
        for (int i = 0; i < 3; ++i)
            trial[i] = lam * ev + 2.0 * G * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = 2.0 * G * state.strain[i];

        // Invariants
        Real I1 = trial[0] + trial[1] + trial[2];
        Real s_dev[6];
        Real mean_stress = I1 / 3.0;
        s_dev[0] = trial[0] - mean_stress;
        s_dev[1] = trial[1] - mean_stress;
        s_dev[2] = trial[2] - mean_stress;
        s_dev[3] = trial[3];
        s_dev[4] = trial[4];
        s_dev[5] = trial[5];

        Real J2 = 0.5 * (s_dev[0] * s_dev[0] + s_dev[1] * s_dev[1] + s_dev[2] * s_dev[2])
                 + s_dev[3] * s_dev[3] + s_dev[4] * s_dev[4] + s_dev[5] * s_dev[5];
        Real sqrt_J2 = Kokkos::sqrt(J2 + 1.0e-30);

        // Maximum principal stress (approximate for Rankine check)
        Real sigma_max = trial[0];
        if (trial[1] > sigma_max) sigma_max = trial[1];
        if (trial[2] > sigma_max) sigma_max = trial[2];

        // Get damage state
        Real d_t = state.history[32];
        Real d_c = state.history[33];
        Real max_tens_strain = state.history[34];
        Real max_comp_strain = state.history[35];

        // --- Tensile damage (Rankine) ---
        Real eps_tens = Kokkos::fmax(state.strain[0], Kokkos::fmax(state.strain[1], state.strain[2]));
        Real eps_t0 = ft_ / E; // Cracking strain
        Real eps_tu = eps_t0 + 2.0 * Gf_t_ / (ft_ * lchar_ + 1.0e-30); // Ultimate tensile strain

        if (eps_tens > max_tens_strain) max_tens_strain = eps_tens;
        state.history[34] = max_tens_strain;

        if (max_tens_strain > eps_t0 && eps_tu > eps_t0) {
            d_t = (max_tens_strain - eps_t0) / (eps_tu - eps_t0);
            if (d_t > 1.0) d_t = 1.0;
            if (d_t < 0.0) d_t = 0.0;
        }
        state.history[32] = d_t;

        // --- Compressive damage ---
        Real eps_comp = -Kokkos::fmin(state.strain[0], Kokkos::fmin(state.strain[1], state.strain[2]));
        Real eps_c0 = fc_ / E;
        Real eps_cu = eps_c0 + 2.0 * Gf_c_ / (fc_ * lchar_ + 1.0e-30);

        if (eps_comp > max_comp_strain) max_comp_strain = eps_comp;
        state.history[35] = max_comp_strain;

        if (max_comp_strain > eps_c0 && eps_cu > eps_c0) {
            d_c = (max_comp_strain - eps_c0) / (eps_cu - eps_c0);
            if (d_c > 1.0) d_c = 1.0;
            if (d_c < 0.0) d_c = 0.0;
        }
        state.history[33] = d_c;

        // --- Drucker-Prager shear return ---
        Real k_dp = fc_ * (1.0 - alpha_dp_) / Kokkos::sqrt(3.0);
        Real f_s = sqrt_J2 + alpha_dp_ * I1 - k_dp;
        if (f_s > 0.0 && sqrt_J2 > 1.0e-10) {
            // Return to yield surface
            Real dlam = f_s / (G + K_bulk * alpha_dp_ * alpha_dp_ * 3.0 + 1.0e-30);
            Real scale = 1.0 - G * dlam / sqrt_J2;
            if (scale < 0.0) scale = 0.0;
            for (int i = 0; i < 6; ++i) s_dev[i] *= scale;
            mean_stress -= K_bulk * alpha_dp_ * dlam;
        }

        // Combined damage factor
        Real d = 1.0 - (1.0 - d_t) * (1.0 - d_c);
        Real omega = 1.0 - d;
        if (omega < 0.0) omega = 0.0;

        // Apply damage to stress
        // Tensile damage only applies to tensile stress components
        for (int i = 0; i < 3; ++i) {
            Real sig_i = s_dev[i] + mean_stress;
            if (sig_i > 0.0) {
                // Tensile stress affected by tensile damage
                state.stress[i] = sig_i * (1.0 - d_t);
            } else {
                // Compressive stress affected by compressive damage
                state.stress[i] = sig_i * (1.0 - d_c);
            }
        }
        for (int i = 3; i < 6; ++i) {
            state.stress[i] = s_dev[i] * omega;
        }

        state.damage = d;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real d = state.damage;
        Real omega = 1.0 - d;
        Real lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) * omega;
        Real mu = E / (2.0 * (1.0 + nu)) * omega;
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = lam + 2.0 * mu; C[1] = lam;           C[2] = lam;
        C[6] = lam;            C[7] = lam + 2.0 * mu; C[8] = lam;
        C[12] = lam;           C[13] = lam;           C[14] = lam + 2.0 * mu;
        C[21] = mu; C[28] = mu; C[35] = mu;
    }

    Real get_tensile_damage(const MaterialState& state) const { return state.history[32]; }
    Real get_compressive_damage(const MaterialState& state) const { return state.history[33]; }

private:
    Real ft_;       // Tensile strength
    Real fc_;       // Compressive strength
    Real Gf_t_;     // Tensile fracture energy (per unit area)
    Real Gf_c_;     // Compressive fracture energy
    Real alpha_dp_; // Drucker-Prager friction parameter
    Real lchar_;    // Characteristic element length (regularization)
};

// ============================================================================
// 4. GranularSoilCapMaterial - Drucker-Prager with Cap Hardening
// ============================================================================

/**
 * @brief Granular soil model with Drucker-Prager shear + cap surface
 *
 * Shear yield: F_s = sqrt(J2) - alpha*I1 - k = 0
 * Cap surface: F_c = sqrt(J2 + (I1 - L)^2/R^2) - (a - c*exp(-b*kappa)) = 0
 * Cap position L moves with plastic volumetric strain kappa.
 *
 * History: [32]=cap_position (L), [33]=kappa (plastic vol strain)
 */
class GranularSoilCapMaterial : public Material {
public:
    GranularSoilCapMaterial(const MaterialProperties& props,
                             Real alpha = 0.25, Real k = 5.0e6,
                             Real R = 2.0,
                             Real a = 20.0e6, Real b = 0.01, Real c = 18.0e6,
                             Real L_init = -40.0e6)
        : Material(MaterialType::Custom, props)
        , alpha_(alpha), k_(k), R_(R)
        , a_(a), b_(b), c_(c), L_init_(L_init) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real lam = K_bulk - 2.0 * G / 3.0;

        // Elastic trial stress
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lam * ev + 2.0 * G * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = 2.0 * G * state.strain[i];

        Real I1 = trial[0] + trial[1] + trial[2];
        Real mean_s = I1 / 3.0;
        Real s_dev[6];
        s_dev[0] = trial[0] - mean_s;
        s_dev[1] = trial[1] - mean_s;
        s_dev[2] = trial[2] - mean_s;
        s_dev[3] = trial[3];
        s_dev[4] = trial[4];
        s_dev[5] = trial[5];

        Real J2 = 0.5 * (s_dev[0] * s_dev[0] + s_dev[1] * s_dev[1] + s_dev[2] * s_dev[2])
                 + s_dev[3] * s_dev[3] + s_dev[4] * s_dev[4] + s_dev[5] * s_dev[5];
        Real sqrt_J2 = Kokkos::sqrt(J2 + 1.0e-30);

        // Initialize cap position if first call
        Real L = state.history[32];
        Real kappa = state.history[33];
        if (L == 0.0 && kappa == 0.0) {
            L = L_init_;
            state.history[32] = L;
        }

        // Cap hardening function: cap_yield = a - c*exp(-b*kappa)
        Real cap_yield = a_ - c_ * Kokkos::exp(-b_ * kappa);

        // --- Check Drucker-Prager shear yield ---
        // F_s = sqrt(J2) - alpha*I1 - k (compressive I1 is negative)
        Real f_s = sqrt_J2 - alpha_ * I1 - k_;

        if (f_s > 0.0 && sqrt_J2 > 1.0e-10) {
            // Shear return mapping
            Real dlam = f_s / (G + 3.0 * K_bulk * alpha_ * alpha_ + 1.0e-30);
            Real scale = 1.0 - G * dlam / sqrt_J2;
            if (scale < 0.0) scale = 0.0;
            for (int i = 0; i < 6; ++i) s_dev[i] *= scale;
            mean_s += K_bulk * alpha_ * dlam; // Pressure correction (volumetric return)
            I1 = 3.0 * mean_s;
        }

        // --- Check cap surface ---
        // Cap only active in compression (I1 < L means very compressive)
        if (I1 < L) {
            Real diff = (I1 - L) / (R_ + 1.0e-30);
            Real f_c = Kokkos::sqrt(J2 + diff * diff) - cap_yield;

            if (f_c > 0.0) {
                // Simplified cap return: project pressure onto cap
                Real delta_kappa = f_c / (K_bulk * 3.0 / (R_ * R_) + 1.0e-30);
                kappa += Kokkos::fabs(delta_kappa);
                state.history[33] = kappa;

                // Update cap position
                cap_yield = a_ - c_ * Kokkos::exp(-b_ * kappa);
                L = L_init_ - (cap_yield - (a_ - c_)) * R_;
                state.history[32] = L;

                // Scale deviatoric to cap
                Real new_J2_cap = cap_yield * cap_yield - diff * diff;
                if (new_J2_cap < 0.0) new_J2_cap = 0.0;
                Real new_sqrt_J2 = Kokkos::sqrt(new_J2_cap + 1.0e-30);
                if (sqrt_J2 > 1.0e-10) {
                    Real cap_scale = new_sqrt_J2 / sqrt_J2;
                    for (int i = 0; i < 6; ++i) s_dev[i] *= cap_scale;
                }

                // Pressure correction
                mean_s = I1 / 3.0 + K_bulk * delta_kappa;
            }
        }

        // Assemble stress
        state.stress[0] = s_dev[0] + mean_s;
        state.stress[1] = s_dev[1] + mean_s;
        state.stress[2] = s_dev[2] + mean_s;
        state.stress[3] = s_dev[3];
        state.stress[4] = s_dev[4];
        state.stress[5] = s_dev[5];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = lam + 2.0 * mu; C[1] = lam;           C[2] = lam;
        C[6] = lam;            C[7] = lam + 2.0 * mu; C[8] = lam;
        C[12] = lam;           C[13] = lam;           C[14] = lam + 2.0 * mu;
        C[21] = mu; C[28] = mu; C[35] = mu;
    }

    Real get_cap_position(const MaterialState& state) const { return state.history[32]; }
    Real get_kappa(const MaterialState& state) const { return state.history[33]; }

private:
    Real alpha_;   // DP friction coefficient
    Real k_;       // DP cohesion
    Real R_;       // Cap aspect ratio
    Real a_, b_, c_; // Cap hardening parameters
    Real L_init_;  // Initial cap position (pressure)
};

// ============================================================================
// 5. Barlat2000Material (Yld2000) - 8-Parameter Anisotropic Yield
// ============================================================================

/**
 * @brief Barlat Yld2000-2d anisotropic yield criterion (plane stress)
 *
 * phi = |X'1 - X'2|^a + |2*X''2 + X''1|^a + |2*X''1 + X''2|^a = 2*sigma_y^a
 * Linear transformations X' = L'*sigma, X'' = L''*sigma
 * 8 alpha parameters for anisotropy
 *
 * History: [0]=plastic_strain (shared), [32..39] available for backstress
 */
class Barlat2000Material : public Material {
public:
    /**
     * @param alpha Array of 8 anisotropy coefficients (alpha1..alpha8)
     * @param a Yield exponent (6 for BCC, 8 for FCC)
     */
    Barlat2000Material(const MaterialProperties& props,
                        const Real alpha[8], Real a = 8.0)
        : Material(MaterialType::Custom, props), a_(a) {
        for (int i = 0; i < 8; ++i) alpha_[i] = alpha[i];
        compute_L_matrices();
    }

    // Convenience: isotropic (all alpha = 1)
    Barlat2000Material(const MaterialProperties& props, Real a = 8.0)
        : Material(MaterialType::Custom, props), a_(a) {
        for (int i = 0; i < 8; ++i) alpha_[i] = 1.0;
        compute_L_matrices();
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));

        // Plane stress elastic trial (sigma_zz = 0)
        Real eps_xx = state.strain[0];
        Real eps_yy = state.strain[1];
        Real eps_xy = state.strain[3];

        Real factor = E / (1.0 - nu * nu);
        Real trial_xx = factor * (eps_xx + nu * eps_yy);
        Real trial_yy = factor * (eps_yy + nu * eps_xx);
        Real trial_xy = G * eps_xy;

        // Von Mises equivalent for initial check
        Real vm_sq = trial_xx * trial_xx - trial_xx * trial_yy + trial_yy * trial_yy
                    + 3.0 * trial_xy * trial_xy;
        Real sigma_vm = Kokkos::sqrt(Kokkos::fmax(vm_sq, 0.0));

        // Current yield stress with isotropic hardening
        Real eps_p = state.history[0];
        Real sigma_y = props_.yield_stress + props_.hardening_modulus * eps_p;

        // Compute Barlat yield function
        Real phi = compute_phi(trial_xx, trial_yy, trial_xy);
        Real phi_yield = 2.0 * Kokkos::pow(sigma_y, a_);

        if (phi > phi_yield && sigma_vm > 1.0e-10) {
            // Radial return (simplified): scale stress to yield surface
            // Newton iteration for consistency
            Real scale = 1.0;
            for (int iter = 0; iter < 20; ++iter) {
                Real sig_xx = trial_xx * scale;
                Real sig_yy = trial_yy * scale;
                Real sig_xy = trial_xy * scale;
                Real phi_trial = compute_phi(sig_xx, sig_yy, sig_xy);

                // delta_eps_p from consistency
                Real dep = (1.0 - scale) * sigma_vm / (3.0 * G + 1.0e-30);
                Real sy = props_.yield_stress + props_.hardening_modulus * (eps_p + dep);
                Real phi_y = 2.0 * Kokkos::pow(sy, a_);

                Real residual = phi_trial - phi_y;
                if (Kokkos::fabs(residual) < 1.0e-6 * phi_y + 1.0e-30) break;

                // Approximate derivative
                Real dphi_ds = a_ * Kokkos::pow(scale * sigma_vm + 1.0e-30, a_ - 1.0) * 2.0;
                Real dsy_dep = props_.hardening_modulus;
                Real dphi_y_dep = 2.0 * a_ * Kokkos::pow(sy + 1.0e-30, a_ - 1.0) * dsy_dep;
                Real denom = dphi_ds * sigma_vm / (3.0 * G + 1.0e-30) + dphi_y_dep * sigma_vm / (3.0 * G + 1.0e-30);
                if (Kokkos::fabs(denom) < 1.0e-30) break;

                Real dscale = -residual / (dphi_ds * sigma_vm + dphi_y_dep * sigma_vm / (3.0 * G) + 1.0e-30);
                scale += dscale;
                if (scale < 0.0) scale = 0.01;
                if (scale > 1.0) scale = 1.0;
            }

            Real delta_ep = (1.0 - scale) * sigma_vm / (3.0 * G + 1.0e-30);
            if (delta_ep < 0.0) delta_ep = 0.0;
            state.history[0] = eps_p + delta_ep;
            state.plastic_strain = eps_p + delta_ep;

            trial_xx *= scale;
            trial_yy *= scale;
            trial_xy *= scale;
        }

        state.stress[0] = trial_xx;
        state.stress[1] = trial_yy;
        state.stress[2] = 0.0; // Plane stress
        state.stress[3] = trial_xy;
        state.stress[4] = 0.0;
        state.stress[5] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real factor = E / (1.0 - nu * nu);
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = factor;          C[1] = factor * nu;
        C[6] = factor * nu;     C[7] = factor;
        C[21] = E / (2.0 * (1.0 + nu));
    }

private:
    Real alpha_[8];
    Real a_; // Yield exponent
    // L' matrix coefficients (2x3 for plane stress: maps [sxx, syy, sxy] to principal)
    Real Lp_[4]; // L' 2x2 for normal part
    Real Lpp_[4]; // L'' 2x2 for normal part

    void compute_L_matrices() {
        // L' depends on alpha1, alpha2, alpha7
        // L'' depends on alpha3, alpha4, alpha5, alpha6, alpha8
        // Simplified: for isotropic (all alpha=1), both reduce to identity
        Lp_[0] = (2.0 * alpha_[0] + alpha_[1]) / 3.0;
        Lp_[1] = (alpha_[1] - alpha_[0]) / 3.0;
        Lp_[2] = (alpha_[0] - alpha_[1]) / 3.0;
        Lp_[3] = (2.0 * alpha_[1] + alpha_[0]) / 3.0;

        Lpp_[0] = (8.0 * alpha_[4] - 2.0 * alpha_[2] - 2.0 * alpha_[3] + 2.0 * alpha_[5]) / 9.0;
        Lpp_[1] = (4.0 * alpha_[5] - 4.0 * alpha_[4] - 4.0 * alpha_[3] + alpha_[2]) / 9.0;
        Lpp_[2] = (4.0 * alpha_[2] - 4.0 * alpha_[4] - 4.0 * alpha_[5] + alpha_[3]) / 9.0;
        Lpp_[3] = (8.0 * alpha_[4] - 2.0 * alpha_[2] - 2.0 * alpha_[3] + 2.0 * alpha_[5]) / 9.0;
    }

    KOKKOS_INLINE_FUNCTION
    Real compute_phi(Real sxx, Real syy, Real sxy) const {
        // Transform through L'
        Real Xp_xx = Lp_[0] * sxx + Lp_[1] * syy;
        Real Xp_yy = Lp_[2] * sxx + Lp_[3] * syy;
        Real Xp_xy = alpha_[6] * sxy; // alpha7

        // Principal values of X'
        Real avg = 0.5 * (Xp_xx + Xp_yy);
        Real diff = 0.5 * (Xp_xx - Xp_yy);
        Real rad = Kokkos::sqrt(diff * diff + Xp_xy * Xp_xy + 1.0e-30);
        Real Xp1 = avg + rad;
        Real Xp2 = avg - rad;

        // Transform through L''
        Real Xpp_xx = Lpp_[0] * sxx + Lpp_[1] * syy;
        Real Xpp_yy = Lpp_[2] * sxx + Lpp_[3] * syy;
        Real Xpp_xy = alpha_[7] * sxy; // alpha8

        avg = 0.5 * (Xpp_xx + Xpp_yy);
        diff = 0.5 * (Xpp_xx - Xpp_yy);
        rad = Kokkos::sqrt(diff * diff + Xpp_xy * Xpp_xy + 1.0e-30);
        Real Xpp1 = avg + rad;
        Real Xpp2 = avg - rad;

        // phi = |X'1 - X'2|^a + |2*X''2 + X''1|^a + |2*X''1 + X''2|^a
        Real term1 = Kokkos::pow(Kokkos::fabs(Xp1 - Xp2), a_);
        Real term2 = Kokkos::pow(Kokkos::fabs(2.0 * Xpp2 + Xpp1), a_);
        Real term3 = Kokkos::pow(Kokkos::fabs(2.0 * Xpp1 + Xpp2), a_);

        return term1 + term2 + term3;
    }
};

// ============================================================================
// 6. ChabocheKinHardeningMaterial - Multi-Backstress Cyclic Plasticity
// ============================================================================

/**
 * @brief Chaboche nonlinear kinematic hardening with multiple backstress terms
 *
 * Yield: f = sigma_vm(sigma - sum(alpha_i)) - R(p) = 0
 * Backstress: dalpha_i = (2/3)*C_i*deps_p - gamma_i*alpha_i*dp
 * Isotropic: R(p) = R_inf*(1 - exp(-b*p))
 *
 * History: [0]=p, [1-6]=total backstress, [7-12]=alpha_1,
 *          [13-18]=alpha_2, [19-24]=alpha_3, [25-30]=alpha_4
 */
class ChabocheKinHardeningMaterial : public Material {
public:
    /**
     * @param n_backstress Number of backstress terms (1-4)
     * @param C_i Kinematic hardening moduli
     * @param gamma_i Dynamic recovery coefficients
     * @param R_inf Isotropic hardening saturation
     * @param b_iso Isotropic hardening rate
     */
    ChabocheKinHardeningMaterial(const MaterialProperties& props,
                                  int n_backstress,
                                  const Real* C_i, const Real* gamma_i,
                                  Real R_inf = 0.0, Real b_iso = 0.0)
        : Material(MaterialType::Custom, props)
        , n_back_(n_backstress > 4 ? 4 : n_backstress)
        , R_inf_(R_inf), b_iso_(b_iso) {
        for (int i = 0; i < 4; ++i) {
            C_[i] = (i < n_back_) ? C_i[i] : 0.0;
            gamma_[i] = (i < n_back_) ? gamma_i[i] : 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real lam = K_bulk - 2.0 * G / 3.0;

        // Elastic trial stress
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lam * ev + 2.0 * G * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = 2.0 * G * state.strain[i];

        // Get backstress from history
        Real alpha_total[6];
        for (int i = 0; i < 6; ++i)
            alpha_total[i] = state.history[1 + i];

        // Relative stress: xi = trial - alpha
        Real xi[6];
        for (int i = 0; i < 6; ++i)
            xi[i] = trial[i] - alpha_total[i];

        // Deviatoric part of xi
        Real xi_mean = (xi[0] + xi[1] + xi[2]) / 3.0;
        Real xi_dev[6];
        xi_dev[0] = xi[0] - xi_mean;
        xi_dev[1] = xi[1] - xi_mean;
        xi_dev[2] = xi[2] - xi_mean;
        xi_dev[3] = xi[3];
        xi_dev[4] = xi[4];
        xi_dev[5] = xi[5];

        Real J2_xi = 0.5 * (xi_dev[0] * xi_dev[0] + xi_dev[1] * xi_dev[1] + xi_dev[2] * xi_dev[2])
                    + xi_dev[3] * xi_dev[3] + xi_dev[4] * xi_dev[4] + xi_dev[5] * xi_dev[5];
        Real sigma_vm_xi = Kokkos::sqrt(3.0 * J2_xi + 1.0e-30);

        // Current yield stress
        Real p = state.history[0]; // Accumulated plastic strain
        Real R_p = R_inf_ * (1.0 - Kokkos::exp(-b_iso_ * p));
        Real sigma_y = props_.yield_stress + R_p;

        // Yield check
        Real f = sigma_vm_xi - sigma_y;

        if (f > 0.0) {
            // Plastic correction with Newton iteration
            Real dp = 0.0;
            Real sqrt_J2_xi = Kokkos::sqrt(J2_xi + 1.0e-30);

            // Flow direction (normalized deviatoric relative stress)
            Real n_flow[6];
            for (int i = 0; i < 6; ++i)
                n_flow[i] = xi_dev[i] / (2.0 * sqrt_J2_xi + 1.0e-30);

            // Simplified radial return: find dp such that f(dp) = 0
            for (int iter = 0; iter < 25; ++iter) {
                Real R_new = R_inf_ * (1.0 - Kokkos::exp(-b_iso_ * (p + dp)));
                Real sy_new = props_.yield_stress + R_new;

                // Sum of kinematic hardening contributions
                Real H_kin = 0.0;
                for (int k = 0; k < n_back_; ++k) {
                    H_kin += C_[k] - gamma_[k] * backstress_norm(state, k) * 1.5;
                }
                // This is approximate; use 2/3 * sum(C_i)
                Real H_kin_approx = 0.0;
                for (int k = 0; k < n_back_; ++k)
                    H_kin_approx += (2.0 / 3.0) * C_[k];

                Real f_dp = sigma_vm_xi - 3.0 * G * dp - sy_new - H_kin_approx * dp;
                if (Kokkos::fabs(f_dp) < 1.0e-10 * sigma_y + 1.0e-30) break;

                Real dR = R_inf_ * b_iso_ * Kokkos::exp(-b_iso_ * (p + dp));
                Real df_ddp = -3.0 * G - dR - H_kin_approx;
                dp -= f_dp / (df_ddp - 1.0e-30);
                if (dp < 0.0) dp = 0.0;
            }

            // Update accumulated plastic strain
            state.history[0] = p + dp;
            state.plastic_strain = p + dp;

            // Update each backstress
            Real alpha_new_total[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (int k = 0; k < n_back_; ++k) {
                int offset = 7 + k * 6;
                for (int i = 0; i < 6; ++i) {
                    Real alpha_k_old = state.history[offset + i];
                    // Armstrong-Frederick: dalpha = (2/3)*C*n*dp - gamma*alpha*dp
                    Real factor = (i < 3) ? 1.0 : 1.0; // Voigt factor handled by n_flow
                    Real dalpha = (2.0 / 3.0) * C_[k] * n_flow[i] * Kokkos::sqrt(2.0 / 3.0) * dp
                                - gamma_[k] * alpha_k_old * dp;
                    Real alpha_k_new = alpha_k_old + dalpha;
                    state.history[offset + i] = alpha_k_new;
                    alpha_new_total[i] += alpha_k_new;
                }
            }
            for (int i = 0; i < 6; ++i)
                state.history[1 + i] = alpha_new_total[i];

            // Scale trial stress to yield surface
            Real scale = 1.0 - 3.0 * G * dp / (sigma_vm_xi + 1.0e-30);
            if (scale < 0.0) scale = 0.0;

            Real mean_trial = (trial[0] + trial[1] + trial[2]) / 3.0;
            Real s_trial[6];
            s_trial[0] = trial[0] - mean_trial;
            s_trial[1] = trial[1] - mean_trial;
            s_trial[2] = trial[2] - mean_trial;
            s_trial[3] = trial[3];
            s_trial[4] = trial[4];
            s_trial[5] = trial[5];

            for (int i = 0; i < 3; ++i)
                state.stress[i] = s_trial[i] * scale + mean_trial + alpha_new_total[i] * (1.0 - scale);
            for (int i = 3; i < 6; ++i)
                state.stress[i] = trial[i] - 2.0 * G * dp * n_flow[i] * Kokkos::sqrt(6.0);
        } else {
            // Elastic
            for (int i = 0; i < 6; ++i)
                state.stress[i] = trial[i];
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = lam + 2.0 * mu; C[1] = lam;           C[2] = lam;
        C[6] = lam;            C[7] = lam + 2.0 * mu; C[8] = lam;
        C[12] = lam;           C[13] = lam;           C[14] = lam + 2.0 * mu;
        C[21] = mu; C[28] = mu; C[35] = mu;
    }

    Real get_accumulated_plastic_strain(const MaterialState& state) const { return state.history[0]; }

private:
    int n_back_;
    Real C_[4];
    Real gamma_[4];
    Real R_inf_;
    Real b_iso_;

    KOKKOS_INLINE_FUNCTION
    Real backstress_norm(const MaterialState& state, int k) const {
        int offset = 7 + k * 6;
        Real sum = 0.0;
        for (int i = 0; i < 3; ++i)
            sum += state.history[offset + i] * state.history[offset + i];
        for (int i = 3; i < 6; ++i)
            sum += 2.0 * state.history[offset + i] * state.history[offset + i];
        return Kokkos::sqrt(1.5 * sum + 1.0e-30);
    }
};

// ============================================================================
// 7. ScaledCrushFoamMaterial - Density-Dependent Crushable Foam
// ============================================================================

/**
 * @brief Density-scaled crushable foam model
 *
 * Base response from tabulated crush curve.
 * Stress scaled by relative density: sigma = sigma_base * (rho/rho0)^n
 * Tension cutoff with damage.
 *
 * History: [0]=max_volumetric_strain, [32]=density_ratio
 */
class ScaledCrushFoamMaterial : public Material {
public:
    ScaledCrushFoamMaterial(const MaterialProperties& props,
                             const TabulatedCurve& crush_curve,
                             Real density_exponent = 2.0,
                             Real tension_cutoff = 1.0e6,
                             Real initial_density_ratio = 1.0)
        : Material(MaterialType::Custom, props)
        , crush_curve_(crush_curve)
        , density_exp_(density_exponent)
        , tension_cutoff_(tension_cutoff)
        , init_density_ratio_(initial_density_ratio) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));

        // Volumetric strain (positive = compression)
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real ev_comp = -ev; // Positive in compression

        // Track max volumetric strain
        Real ev_max = state.history[0];
        if (ev_comp > ev_max) {
            ev_max = ev_comp;
            state.history[0] = ev_max;
        }

        // Density ratio evolution
        Real rho_ratio = state.history[32];
        if (rho_ratio < 1.0e-10) rho_ratio = init_density_ratio_;
        // Density increases with compression: rho/rho0 = 1/(1 - ev_comp)
        Real new_rho_ratio = init_density_ratio_ / (1.0 - ev_comp + 1.0e-10);
        if (new_rho_ratio > rho_ratio) rho_ratio = new_rho_ratio;
        state.history[32] = rho_ratio;

        // Density scaling factor
        Real scale = Kokkos::pow(rho_ratio, density_exp_);

        // Pressure from crush curve (evaluated at max volumetric strain for unloading)
        Real P_crush;
        if (ev_comp >= 0.0) {
            // Compression: follow crush curve
            if (ev_comp >= ev_max - 1.0e-10) {
                // Loading
                P_crush = crush_curve_.evaluate(ev_comp) * scale;
            } else {
                // Unloading: elastic from max point
                Real P_max = crush_curve_.evaluate(ev_max) * scale;
                Real unload_stiffness = E;
                P_crush = P_max - unload_stiffness * (ev_max - ev_comp);
                if (P_crush < 0.0) P_crush = 0.0;
            }
        } else {
            // Tension
            P_crush = E * ev_comp; // Linear in tension (ev_comp < 0 means tension)
            if (-P_crush > tension_cutoff_) {
                P_crush = -tension_cutoff_;
                state.damage = 1.0;
            }
        }

        // Deviatoric stress (elastic)
        Real ev_third = ev / 3.0;
        Real s_dev[6];
        s_dev[0] = 2.0 * G * (state.strain[0] - ev_third);
        s_dev[1] = 2.0 * G * (state.strain[1] - ev_third);
        s_dev[2] = 2.0 * G * (state.strain[2] - ev_third);
        s_dev[3] = 2.0 * G * state.strain[3];
        s_dev[4] = 2.0 * G * state.strain[4];
        s_dev[5] = 2.0 * G * state.strain[5];

        // Assemble: stress = deviatoric - P (pressure positive in compression)
        state.stress[0] = s_dev[0] - P_crush;
        state.stress[1] = s_dev[1] - P_crush;
        state.stress[2] = s_dev[2] - P_crush;
        state.stress[3] = s_dev[3];
        state.stress[4] = s_dev[4];
        state.stress[5] = s_dev[5];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = lam + 2.0 * mu; C[1] = lam;           C[2] = lam;
        C[6] = lam;            C[7] = lam + 2.0 * mu; C[8] = lam;
        C[12] = lam;           C[13] = lam;           C[14] = lam + 2.0 * mu;
        C[21] = mu; C[28] = mu; C[35] = mu;
    }

    Real get_density_ratio(const MaterialState& state) const { return state.history[32]; }

private:
    TabulatedCurve crush_curve_;
    Real density_exp_;
    Real tension_cutoff_;
    Real init_density_ratio_;
};

// ============================================================================
// 8. ThermoplasticPolymerMaterial - Rate/Temperature Dependent Polymer
// ============================================================================

/**
 * @brief Thermoplastic polymer with rate and temperature dependence
 *
 * sigma_y = [sigma_0 + H*eps_p + K_poly*eps_p^n] * [1 + C*ln(edot/edot_0)]
 *         * [1 - ((T-T_ref)/(T_melt-T_ref))^m]
 *
 * Combines polynomial hardening + logarithmic rate sensitivity + thermal softening
 *
 * History: [0]=plastic_strain (shared)
 */
class ThermoplasticPolymerMaterial : public Material {
public:
    ThermoplasticPolymerMaterial(const MaterialProperties& props,
                                  Real sigma_0 = 50.0e6,
                                  Real H = 100.0e6, Real K_poly = 200.0e6,
                                  Real n = 0.4, Real C_rate = 0.02,
                                  Real eps_dot_0 = 1.0,
                                  Real T_ref = 293.15, Real T_melt = 500.0,
                                  Real m_thermal = 1.0)
        : Material(MaterialType::Custom, props)
        , sigma_0_(sigma_0), H_(H), K_poly_(K_poly), n_(n)
        , C_rate_(C_rate), eps_dot_0_(eps_dot_0)
        , T_ref_(T_ref), T_melt_(T_melt), m_thermal_(m_thermal) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real lam = K_bulk - 2.0 * G / 3.0;

        // Elastic trial
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lam * ev + 2.0 * G * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = 2.0 * G * state.strain[i];

        // Von Mises equivalent
        Real mean = (trial[0] + trial[1] + trial[2]) / 3.0;
        Real s_dev[6];
        s_dev[0] = trial[0] - mean;
        s_dev[1] = trial[1] - mean;
        s_dev[2] = trial[2] - mean;
        s_dev[3] = trial[3];
        s_dev[4] = trial[4];
        s_dev[5] = trial[5];

        Real J2 = 0.5 * (s_dev[0] * s_dev[0] + s_dev[1] * s_dev[1] + s_dev[2] * s_dev[2])
                 + s_dev[3] * s_dev[3] + s_dev[4] * s_dev[4] + s_dev[5] * s_dev[5];
        Real sigma_vm = Kokkos::sqrt(3.0 * J2 + 1.0e-30);

        // Current plastic strain
        Real eps_p = state.history[0];

        // Compute yield stress with rate and temperature effects
        Real sigma_y = compute_yield(eps_p, state.effective_strain_rate, state.temperature);

        // Yield check
        Real f = sigma_vm - sigma_y;
        if (f > 0.0 && sigma_vm > 1.0e-10) {
            // Radial return
            Real dp = f / (3.0 * G + hardening_slope(eps_p, state.effective_strain_rate, state.temperature));
            if (dp < 0.0) dp = 0.0;

            state.history[0] = eps_p + dp;
            state.plastic_strain = eps_p + dp;

            Real scale = 1.0 - 3.0 * G * dp / sigma_vm;
            if (scale < 0.0) scale = 0.0;
            for (int i = 0; i < 6; ++i)
                s_dev[i] *= scale;
        }

        // Assemble stress
        for (int i = 0; i < 3; ++i)
            state.stress[i] = s_dev[i] + mean;
        for (int i = 3; i < 6; ++i)
            state.stress[i] = s_dev[i];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = lam + 2.0 * mu; C[1] = lam;           C[2] = lam;
        C[6] = lam;            C[7] = lam + 2.0 * mu; C[8] = lam;
        C[12] = lam;           C[13] = lam;           C[14] = lam + 2.0 * mu;
        C[21] = mu; C[28] = mu; C[35] = mu;
    }

    KOKKOS_INLINE_FUNCTION
    Real compute_yield(Real eps_p, Real eps_dot, Real T) const {
        // Base hardening
        Real sigma_base = sigma_0_ + H_ * eps_p + K_poly_ * Kokkos::pow(eps_p + 1.0e-30, n_);

        // Rate sensitivity
        Real rate_factor = 1.0;
        if (eps_dot > eps_dot_0_) {
            rate_factor = 1.0 + C_rate_ * Kokkos::log(eps_dot / eps_dot_0_);
        }

        // Thermal softening
        Real T_star = (T - T_ref_) / (T_melt_ - T_ref_ + 1.0e-30);
        if (T_star < 0.0) T_star = 0.0;
        if (T_star > 1.0) T_star = 1.0;
        Real thermal_factor = 1.0 - Kokkos::pow(T_star, m_thermal_);

        return sigma_base * rate_factor * thermal_factor;
    }

private:
    Real sigma_0_, H_, K_poly_, n_;
    Real C_rate_, eps_dot_0_;
    Real T_ref_, T_melt_, m_thermal_;

    KOKKOS_INLINE_FUNCTION
    Real hardening_slope(Real eps_p, Real eps_dot, Real T) const {
        // Derivative of yield w.r.t. eps_p (clamp to avoid singularity for n < 1)
        Real eps_p_safe = Kokkos::fmax(eps_p, 1.0e-6);
        Real dbase = H_ + K_poly_ * n_ * Kokkos::pow(eps_p_safe, n_ - 1.0);

        Real rate_factor = 1.0;
        if (eps_dot > eps_dot_0_) {
            rate_factor = 1.0 + C_rate_ * Kokkos::log(eps_dot / eps_dot_0_);
        }

        Real T_star = (T - T_ref_) / (T_melt_ - T_ref_ + 1.0e-30);
        if (T_star < 0.0) T_star = 0.0;
        if (T_star > 1.0) T_star = 1.0;
        Real thermal_factor = 1.0 - Kokkos::pow(T_star, m_thermal_);

        return dbase * rate_factor * thermal_factor;
    }
};

// ============================================================================
// 9. UnifiedCreepMaterial - Plasticity + Norton Creep + Thermal
// ============================================================================

/**
 * @brief Unified plasticity-creep model with thermal dependence
 *
 * Total: eps_dot = eps_dot_elastic + eps_dot_plastic + eps_dot_creep
 * Creep rate: eps_dot_cr = A_cr * sigma_vm^n_cr * exp(-Q/(R_gas*T))
 * Plasticity: standard J2 with isotropic hardening
 *
 * History: [0]=plastic_strain, [32]=creep_strain, [33]=total_time
 */
class UnifiedCreepMaterial : public Material {
public:
    UnifiedCreepMaterial(const MaterialProperties& props,
                          Real A_cr = 1.0e-12, Real n_cr = 3.0,
                          Real Q_act = 200.0e3, Real R_gas = 8.314)
        : Material(MaterialType::Custom, props)
        , A_cr_(A_cr), n_cr_(n_cr), Q_act_(Q_act), R_gas_(R_gas) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real lam = K_bulk - 2.0 * G / 3.0;

        // Time tracking
        state.history[33] += state.dt;

        // Elastic trial stress
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lam * ev + 2.0 * G * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = 2.0 * G * state.strain[i];

        Real mean = (trial[0] + trial[1] + trial[2]) / 3.0;
        Real s_dev[6];
        for (int i = 0; i < 3; ++i) s_dev[i] = trial[i] - mean;
        for (int i = 3; i < 6; ++i) s_dev[i] = trial[i];

        Real J2 = 0.5 * (s_dev[0] * s_dev[0] + s_dev[1] * s_dev[1] + s_dev[2] * s_dev[2])
                 + s_dev[3] * s_dev[3] + s_dev[4] * s_dev[4] + s_dev[5] * s_dev[5];
        Real sigma_vm = Kokkos::sqrt(3.0 * J2 + 1.0e-30);

        Real eps_p = state.history[0];
        Real eps_cr = state.history[32];

        // --- Plasticity (J2 with hardening) ---
        Real sigma_y = props_.yield_stress + props_.hardening_modulus * eps_p;
        Real f = sigma_vm - sigma_y;
        Real dp = 0.0;

        if (f > 0.0 && sigma_vm > 1.0e-10) {
            dp = f / (3.0 * G + props_.hardening_modulus);
            if (dp < 0.0) dp = 0.0;
            eps_p += dp;
            state.history[0] = eps_p;
            state.plastic_strain = eps_p;

            Real scale_p = 1.0 - 3.0 * G * dp / sigma_vm;
            if (scale_p < 0.0) scale_p = 0.0;
            for (int i = 0; i < 6; ++i) s_dev[i] *= scale_p;

            // Recompute sigma_vm after plastic return
            J2 = 0.5 * (s_dev[0] * s_dev[0] + s_dev[1] * s_dev[1] + s_dev[2] * s_dev[2])
                + s_dev[3] * s_dev[3] + s_dev[4] * s_dev[4] + s_dev[5] * s_dev[5];
            sigma_vm = Kokkos::sqrt(3.0 * J2 + 1.0e-30);
        }

        // --- Creep (Norton power law) ---
        Real T = state.temperature;
        Real eps_dot_cr = A_cr_ * Kokkos::pow(sigma_vm + 1.0e-30, n_cr_)
                        * Kokkos::exp(-Q_act_ / (R_gas_ * T + 1.0e-30));

        Real d_eps_cr = eps_dot_cr * state.dt;
        eps_cr += d_eps_cr;
        state.history[32] = eps_cr;

        // Apply creep strain as stress relaxation
        if (sigma_vm > 1.0e-10 && d_eps_cr > 0.0) {
            Real scale_cr = 1.0 - 3.0 * G * d_eps_cr / sigma_vm;
            if (scale_cr < 0.0) scale_cr = 0.0;
            for (int i = 0; i < 6; ++i) s_dev[i] *= scale_cr;
        }

        // Assemble stress
        for (int i = 0; i < 3; ++i)
            state.stress[i] = s_dev[i] + mean;
        for (int i = 3; i < 6; ++i)
            state.stress[i] = s_dev[i];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = lam + 2.0 * mu; C[1] = lam;           C[2] = lam;
        C[6] = lam;            C[7] = lam + 2.0 * mu; C[8] = lam;
        C[12] = lam;           C[13] = lam;           C[14] = lam + 2.0 * mu;
        C[21] = mu; C[28] = mu; C[35] = mu;
    }

    Real get_creep_strain(const MaterialState& state) const { return state.history[32]; }
    Real get_total_time(const MaterialState& state) const { return state.history[33]; }

    KOKKOS_INLINE_FUNCTION
    Real creep_rate(Real sigma_vm, Real T) const {
        return A_cr_ * Kokkos::pow(sigma_vm + 1.0e-30, n_cr_)
             * Kokkos::exp(-Q_act_ / (R_gas_ * T + 1.0e-30));
    }

private:
    Real A_cr_;  // Creep coefficient
    Real n_cr_;  // Creep exponent
    Real Q_act_; // Activation energy (J/mol)
    Real R_gas_; // Gas constant (J/(mol*K))
};

// ============================================================================
// 10. AdvancedFabricMaterial - Biaxial Fabric with Permanent Set
// ============================================================================

/**
 * @brief Advanced biaxial fabric model with shear locking and permanent set
 *
 * Biaxial: separate warp/weft stiffness (E_warp, E_weft)
 * Shear resistance: tau = G*gamma + G_lock*(gamma/gamma_lock)^3
 * Permanent set: residual strain after exceeding threshold
 *
 * History: [32]=perm_set_warp, [33]=perm_set_weft,
 *          [34]=max_warp_strain, [35]=max_weft_strain
 */
class AdvancedFabricMaterial : public Material {
public:
    AdvancedFabricMaterial(const MaterialProperties& props,
                            Real E_warp = 1.0e9, Real E_weft = 0.5e9,
                            Real G_shear = 1.0e6, Real G_lock = 10.0e6,
                            Real gamma_lock = 0.5,
                            Real perm_set_threshold = 0.02,
                            Real perm_set_fraction = 0.3,
                            Real thickness = 0.001)
        : Material(MaterialType::Custom, props)
        , E_warp_(E_warp), E_weft_(E_weft)
        , G_shear_(G_shear), G_lock_(G_lock), gamma_lock_(gamma_lock)
        , perm_thresh_(perm_set_threshold), perm_frac_(perm_set_fraction)
        , thickness_(thickness) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        // Warp direction = xx, Weft direction = yy, shear = xy
        Real eps_warp = state.strain[0];
        Real eps_weft = state.strain[1];
        Real gamma_xy = state.strain[3];

        // Get permanent set state
        Real perm_warp = state.history[32];
        Real perm_weft = state.history[33];
        Real max_warp = state.history[34];
        Real max_weft = state.history[35];

        // Effective strain (subtract permanent set)
        Real eff_warp = eps_warp - perm_warp;
        Real eff_weft = eps_weft - perm_weft;

        // Update max strains
        if (eps_warp > max_warp) max_warp = eps_warp;
        if (eps_weft > max_weft) max_weft = eps_weft;
        state.history[34] = max_warp;
        state.history[35] = max_weft;

        // Permanent set evolution: on unloading after exceeding threshold
        if (max_warp > perm_thresh_ && eps_warp < max_warp) {
            Real new_perm = perm_frac_ * max_warp;
            if (new_perm > perm_warp) {
                perm_warp = new_perm;
                state.history[32] = perm_warp;
                eff_warp = eps_warp - perm_warp;
            }
        }
        if (max_weft > perm_thresh_ && eps_weft < max_weft) {
            Real new_perm = perm_frac_ * max_weft;
            if (new_perm > perm_weft) {
                perm_weft = new_perm;
                state.history[33] = perm_weft;
                eff_weft = eps_weft - perm_weft;
            }
        }

        // Warp stress: no-compression fabric (tension only)
        Real sigma_warp = 0.0;
        if (eff_warp > 0.0) {
            sigma_warp = E_warp_ * eff_warp;
        }

        // Weft stress
        Real sigma_weft = 0.0;
        if (eff_weft > 0.0) {
            sigma_weft = E_weft_ * eff_weft;
        }

        // Shear stress with locking
        Real gamma_abs = Kokkos::fabs(gamma_xy);
        Real tau = G_shear_ * gamma_xy;
        if (gamma_abs > 1.0e-10) {
            // Add locking term: G_lock * (gamma/gamma_lock)^3
            Real ratio = gamma_abs / (gamma_lock_ + 1.0e-30);
            Real lock_term = G_lock_ * ratio * ratio * ratio;
            // Sign follows gamma_xy
            Real sign = (gamma_xy > 0.0) ? 1.0 : -1.0;
            tau += sign * lock_term;
        }

        // Through-thickness: assume zero (membrane)
        state.stress[0] = sigma_warp;
        state.stress[1] = sigma_weft;
        state.stress[2] = 0.0;
        state.stress[3] = tau;
        state.stress[4] = 0.0;
        state.stress[5] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        // Only tension stiffness
        Real eff_warp = state.strain[0] - state.history[32];
        Real eff_weft = state.strain[1] - state.history[33];

        C[0] = (eff_warp > 0.0) ? E_warp_ : 0.0;
        C[7] = (eff_weft > 0.0) ? E_weft_ : 0.0;
        C[21] = G_shear_; // Linear part only
    }

    Real get_perm_set_warp(const MaterialState& state) const { return state.history[32]; }
    Real get_perm_set_weft(const MaterialState& state) const { return state.history[33]; }

private:
    Real E_warp_, E_weft_;
    Real G_shear_, G_lock_, gamma_lock_;
    Real perm_thresh_, perm_frac_;
    Real thickness_;
};

} // namespace physics
} // namespace nxs
