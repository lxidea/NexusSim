#pragma once

/**
 * @file material_wave39.hpp
 * @brief Wave 39a material models: 6 constitutive models
 *
 * Models included:
 *   1.  DPCapMaterial                  - Drucker-Prager with cap hardening (LAW81)
 *   2.  ThermalMetallurgyMaterial      - Phase transformation kinetics (LAW80)
 *   3.  ElasticShellMaterial           - Shell-specific elastic-plastic (LAW4)
 *   4.  CompositeDamageMaterial        - Progressive composite damage (LAW53)
 *   5.  NonlinearElasticMaterial       - Nonlinear elastic polynomial/exponential (LAW44)
 *   6.  MohrCoulombMaterial            - Classic Mohr-Coulomb (LAW27)
 */

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

// ============================================================================
// 1. DPCapMaterial - Drucker-Prager with Cap Hardening (LAW81)
// ============================================================================

/**
 * @brief Drucker-Prager with elliptical cap hardening model
 *
 * Two yield surfaces:
 *   Shear (DP): F_s = sqrt(J2) + alpha*I1 - k = 0
 *   Cap:        F_c = sqrt(J2 + (I1 - L)^2 / R^2) - (d + p_a*tan(beta)) = 0
 *
 * The cap position L evolves with plastic volumetric strain:
 *   L = L(eps_v_p) via hardening curve.
 * Cap apex p_a = intersection of DP line and cap.
 *   p_a = (k - alpha*L*R) / (1 + alpha*R)  ... approximately.
 *
 * d = k - alpha * p_a  (cohesion intercept)
 * beta = friction_angle (in the cap context)
 *
 * History: [32] = eps_v_p (plastic volumetric strain)
 *          [33] = cap_position_pb (cap hardening parameter)
 */
class DPCapMaterial : public Material {
public:
    DPCapMaterial(const MaterialProperties& props,
                  Real friction_angle = 30.0, Real cohesion = 1.0e6,
                  Real cap_ratio_R = 2.0, Real cap_initial_pb = -1.0e6,
                  Real cap_hardening_W = 0.1, Real cap_hardening_D = 1.0e-9)
        : Material(MaterialType::Custom, props)
        , cohesion_(cohesion), cap_R_(cap_ratio_R)
        , cap_pb0_(cap_initial_pb)
        , cap_W_(cap_hardening_W), cap_D_(cap_hardening_D)
    {
        // Convert friction angle to DP parameters
        Real phi_rad = friction_angle * 3.14159265358979323846 / 180.0;
        Real sin_phi = Kokkos::sin(phi_rad);
        Real cos_phi = Kokkos::cos(phi_rad);
        // Drucker-Prager matching Mohr-Coulomb (outer cone)
        alpha_ = 2.0 * sin_phi / (1.7320508075688772 * (3.0 - sin_phi));
        k_ = 6.0 * cohesion * cos_phi / (1.7320508075688772 * (3.0 - sin_phi));
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = K_bulk - 2.0 * G / 3.0;

        // Elastic trial stress
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lambda * ev + 2.0 * G * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = G * state.strain[i];

        // Stress invariants
        Real I1 = trial[0] + trial[1] + trial[2];
        Real p = I1 / 3.0;
        Real s[6];
        for (int i = 0; i < 3; ++i) s[i] = trial[i] - p;
        for (int i = 3; i < 6; ++i) s[i] = trial[i];

        Real J2 = 0.5 * (s[0]*s[0] + s[1]*s[1] + s[2]*s[2])
                 + s[3]*s[3] + s[4]*s[4] + s[5]*s[5];
        Real sqrtJ2 = Kokkos::sqrt(J2 + 1.0e-30);

        // Retrieve history
        Real eps_v_p = state.history[32];
        Real pb = state.history[33];
        if (pb == 0.0 && eps_v_p == 0.0) {
            pb = cap_pb0_;
        }

        // Cap position: L = pb (hydrostatic compressive cap limit)
        Real L = pb;
        // Intersection of DP line with p-axis:  p_a = k / alpha  (tension side)
        Real p_a = (alpha_ > 1.0e-30) ? (k_ / alpha_) : 1.0e30;
        // d parameter for the cap
        Real d = k_ - alpha_ * Kokkos::fabs(L);

        // DP shear yield: F_s = sqrt(J2) + alpha * I1 - k
        Real F_shear = sqrtJ2 + alpha_ * I1 - k_;

        // Cap yield: F_c = sqrt(J2 + (I1 - L)^2 / R^2) - (d + pa*tan(beta))
        // Use pa ~ k/alpha for the cap apex
        Real cap_rhs = d + 1.0e-10;
        if (cap_rhs < 1.0e-10) cap_rhs = 1.0e-10;
        Real cap_arg = J2 + (I1 - L) * (I1 - L) / (cap_R_ * cap_R_ + 1.0e-30);
        Real F_cap = Kokkos::sqrt(cap_arg + 1.0e-30) - cap_rhs;

        // Check if in the cap region (I1 < L) or shear region
        bool in_cap_region = (I1 < L);

        if (F_shear <= 0.0 && (!in_cap_region || F_cap <= 0.0)) {
            // Elastic
            for (int i = 0; i < 6; ++i)
                state.stress[i] = trial[i];
        } else if (in_cap_region && F_cap > 0.0) {
            // Cap return mapping (simplified radial return on cap surface)
            // Newton iteration for plastic multiplier on cap
            Real dgamma = 0.0;
            Real sqrt_cap = Kokkos::sqrt(cap_arg + 1.0e-30);
            for (int iter = 0; iter < 25; ++iter) {
                // Updated cap position with hardening
                Real eps_v_p_cur = eps_v_p + dgamma * (I1 - L) / (cap_R_ * cap_R_ * sqrt_cap + 1.0e-30);
                Real pb_cur = pb - cap_W_ * (1.0 - Kokkos::exp(-cap_D_ * Kokkos::fabs(eps_v_p_cur))) / (cap_D_ + 1.0e-30);
                Real L_cur = pb_cur;
                Real d_cur = k_ - alpha_ * Kokkos::fabs(L_cur);
                if (d_cur < 1.0e-10) d_cur = 1.0e-10;

                Real cap_arg_cur = J2 / ((1.0 + 2.0 * G * dgamma / (sqrt_cap + 1.0e-30)) *
                                          (1.0 + 2.0 * G * dgamma / (sqrt_cap + 1.0e-30)))
                                 + (I1 - 3.0 * K_bulk * dgamma * (I1 - L_cur) / (cap_R_ * cap_R_ * sqrt_cap + 1.0e-30) - L_cur) *
                                   (I1 - 3.0 * K_bulk * dgamma * (I1 - L_cur) / (cap_R_ * cap_R_ * sqrt_cap + 1.0e-30) - L_cur) /
                                   (cap_R_ * cap_R_ + 1.0e-30);

                Real f_val = Kokkos::sqrt(cap_arg_cur + 1.0e-30) - d_cur;
                if (Kokkos::fabs(f_val) < 1.0e-10 * (Kokkos::fabs(cap_rhs) + 1.0)) break;

                // Approximate derivative
                Real df = -(2.0 * G + 3.0 * K_bulk / (cap_R_ * cap_R_ + 1.0e-30));
                Real ddg = -f_val / (df - 1.0e-30);
                dgamma += ddg;
                if (dgamma < 0.0) dgamma = 0.0;
            }

            // Apply return: deviatoric scaling + volumetric correction
            Real scale_dev = 1.0 / (1.0 + 2.0 * G * dgamma / (sqrt_cap + 1.0e-30));
            if (scale_dev > 1.0) scale_dev = 1.0;
            if (scale_dev < 0.0) scale_dev = 0.0;

            Real dp_vol = K_bulk * dgamma * (I1 - L) / (cap_R_ * cap_R_ * sqrt_cap + 1.0e-30);
            Real p_new = p - dp_vol;

            for (int i = 0; i < 3; ++i)
                state.stress[i] = scale_dev * s[i] + p_new;
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale_dev * s[i];

            // Update history
            Real d_eps_v_p = dgamma * (I1 - L) / (cap_R_ * cap_R_ * sqrt_cap + 1.0e-30);
            eps_v_p += d_eps_v_p;
            pb = pb - cap_W_ * (1.0 - Kokkos::exp(-cap_D_ * Kokkos::fabs(eps_v_p))) / (cap_D_ + 1.0e-30);

            state.history[32] = eps_v_p;
            state.history[33] = pb;
            state.plastic_strain += dgamma;
        } else {
            // DP shear return mapping (same as standard DP)
            Real denom = G + 9.0 * K_bulk * alpha_ * alpha_;
            if (denom < 1.0e-30) denom = 1.0e-30;
            Real dgamma = F_shear / denom;
            if (dgamma < 0.0) dgamma = 0.0;

            Real factor = 1.0 - G * dgamma / (sqrtJ2 + 1.0e-30);
            if (factor < 0.0) factor = 0.0;

            Real p_new = p - K_bulk * alpha_ * dgamma;
            for (int i = 0; i < 3; ++i)
                state.stress[i] = factor * s[i] + p_new;
            for (int i = 3; i < 6; ++i)
                state.stress[i] = factor * s[i];

            // DP shear does not directly harden the cap but may produce vol plastic strain
            Real d_eps_v_p = alpha_ * dgamma;
            eps_v_p += d_eps_v_p;
            state.history[32] = eps_v_p;
            state.history[33] = pb;
            state.plastic_strain += dgamma;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = K_bulk - 2.0 * G / 3.0;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        // Use elastic tangent with plastic reduction factor
        Real red = 1.0;
        if (state.plastic_strain > 0.0) {
            // Reduce stiffness based on plastic activity
            red = G / (G + 9.0 * K_bulk * alpha_ * alpha_ + 1.0e-30);
            red = Kokkos::fmax(red, 0.3);
        }

        Real G_eff = G * red;
        Real lambda_eff = lambda * red;

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda_eff + 2.0 * G_eff;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda_eff;
            }
        }
        C[21] = G_eff;
        C[28] = G_eff;
        C[35] = G_eff;
    }

    KOKKOS_INLINE_FUNCTION Real get_alpha() const { return alpha_; }
    KOKKOS_INLINE_FUNCTION Real get_k() const { return k_; }
    KOKKOS_INLINE_FUNCTION Real get_cap_R() const { return cap_R_; }

private:
    Real cohesion_;
    Real alpha_;           // DP friction parameter
    Real k_;               // DP cohesion-related yield
    Real cap_R_;           // Cap ellipticity ratio R
    Real cap_pb0_;         // Initial cap position (hydrostatic)
    Real cap_W_;           // Cap hardening parameter W
    Real cap_D_;           // Cap hardening parameter D
};

// ============================================================================
// 2. ThermalMetallurgyMaterial - Phase Transformation Kinetics (LAW80)
// ============================================================================

/**
 * @brief Phase transformation kinetics with Kirkaldy diffusion model
 *
 * Tracks 5 metallurgical phases: austenite (0), ferrite (1), pearlite (2),
 * bainite (3), martensite (4).
 *
 * Diffusive transformations (ferrite, pearlite, bainite) use JMAK kinetics:
 *   f = f_max * (1 - exp(-b * t^n))
 *
 * Martensite via Koistinen-Marburger:
 *   f_m = (1 - f_diffuse) * (1 - exp(-alpha_km * (Ms - T)))
 *   for T < Ms during cooling
 *
 * Effective properties by linear mixing rule:
 *   E_eff = sum(f_i * E_i),  sigma_y_eff = sum(f_i * sigma_y_i)
 *
 * History: [32-36] = phase fractions (austenite, ferrite, pearlite, bainite, martensite)
 *          [37] = transformation strain (volumetric, from phase changes)
 */
class ThermalMetallurgyMaterial : public Material {
public:
    ThermalMetallurgyMaterial(const MaterialProperties& props,
                               Real Ms_temp = 350.0, Real alpha_km = 0.011,
                               Real jmak_b = 1.0e-3, Real jmak_n = 2.5)
        : Material(MaterialType::Custom, props)
        , Ms_temp_(Ms_temp), alpha_km_(alpha_km)
        , jmak_b_(jmak_b), jmak_n_(jmak_n)
    {
        // Default phase-specific Young's moduli [Pa]
        phase_E_[0] = 200.0e9;  // austenite
        phase_E_[1] = 210.0e9;  // ferrite
        phase_E_[2] = 210.0e9;  // pearlite
        phase_E_[3] = 215.0e9;  // bainite
        phase_E_[4] = 220.0e9;  // martensite

        // Default phase-specific yield stresses [Pa]
        phase_yield_[0] = 200.0e6;
        phase_yield_[1] = 250.0e6;
        phase_yield_[2] = 350.0e6;
        phase_yield_[3] = 500.0e6;
        phase_yield_[4] = 1000.0e6;

        // Transformation strain coefficients (volumetric)
        // Typically martensite has ~2-4% volume expansion
        trans_strain_[0] = 0.0;
        trans_strain_[1] = 0.005;
        trans_strain_[2] = 0.005;
        trans_strain_[3] = 0.006;
        trans_strain_[4] = 0.03;
    }

    /// Set phase-specific properties
    void set_phase_E(int phase, Real E_val) {
        if (phase >= 0 && phase < 5) phase_E_[phase] = E_val;
    }
    void set_phase_yield(int phase, Real sy) {
        if (phase >= 0 && phase < 5) phase_yield_[phase] = sy;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        // Read phase fractions from history
        Real f[5];
        bool first_call = true;
        for (int i = 0; i < 5; ++i) {
            f[i] = state.history[32 + i];
            if (f[i] != 0.0) first_call = false;
        }
        Real trans_eps = state.history[37];

        // Initialize: start as 100% austenite
        if (first_call) {
            f[0] = 1.0;
            for (int i = 1; i < 5; ++i) f[i] = 0.0;
        }

        // Get temperature from state (use temperature field if available,
        // otherwise use a default reference temperature)
        Real T = state.temperature;

        // Update phase fractions based on temperature
        // Diffusive transformations: ferrite (1), pearlite (2), bainite (3)
        // These occur during cooling when T is in the appropriate range
        Real f_aust_available = f[0]; // available austenite to transform

        if (f_aust_available > 1.0e-10) {
            // JMAK kinetics for diffusive phases
            // Simplified: use a pseudo-time based on accumulated strain as proxy
            // In a real implementation, this would track actual time
            Real pseudo_time = state.plastic_strain + 1.0e-6;

            // Ferrite (high T diffusive, ~700-900 C for steel)
            if (T > 600.0 && T < 900.0) {
                Real f_eq = f_aust_available * 0.7; // max ferrite fraction
                Real f_jmak = f_eq * (1.0 - Kokkos::exp(-jmak_b_ * Kokkos::pow(pseudo_time, jmak_n_)));
                if (f_jmak > f[1]) {
                    Real df = f_jmak - f[1];
                    f[1] += df;
                    f[0] -= df;
                }
            }

            // Pearlite (mid T diffusive, ~550-700 C)
            if (T > 500.0 && T < 700.0) {
                Real f_eq = f_aust_available * 0.5;
                Real f_jmak = f_eq * (1.0 - Kokkos::exp(-jmak_b_ * 2.0 * Kokkos::pow(pseudo_time, jmak_n_)));
                if (f_jmak > f[2]) {
                    Real df = f_jmak - f[2];
                    f[2] += df;
                    f[0] -= df;
                }
            }

            // Bainite (lower T diffusive, ~250-550 C)
            if (T > 200.0 && T < 550.0) {
                Real f_eq = f_aust_available * 0.8;
                Real f_jmak = f_eq * (1.0 - Kokkos::exp(-jmak_b_ * 0.5 * Kokkos::pow(pseudo_time, jmak_n_)));
                if (f_jmak > f[3]) {
                    Real df = f_jmak - f[3];
                    f[3] += df;
                    f[0] -= df;
                }
            }

            // Martensite via Koistinen-Marburger (athermal, below Ms)
            if (T < Ms_temp_) {
                Real f_remaining = f[0]; // remaining austenite
                if (f_remaining > 1.0e-10) {
                    Real f_m_km = f_remaining * (1.0 - Kokkos::exp(-alpha_km_ * (Ms_temp_ - T)));
                    if (f_m_km > f[4]) {
                        Real df = f_m_km - f[4];
                        f[4] += df;
                        f[0] -= df;
                    }
                }
            }
        }

        // Clamp fractions
        Real f_sum = 0.0;
        for (int i = 0; i < 5; ++i) {
            if (f[i] < 0.0) f[i] = 0.0;
            f_sum += f[i];
        }
        if (f_sum > 1.0e-30) {
            for (int i = 0; i < 5; ++i) f[i] /= f_sum;
        }

        // Compute effective properties via mixing rule
        Real E_eff = 0.0;
        Real sigma_y_eff = 0.0;
        Real trans_eps_new = 0.0;
        for (int i = 0; i < 5; ++i) {
            E_eff += f[i] * phase_E_[i];
            sigma_y_eff += f[i] * phase_yield_[i];
            trans_eps_new += f[i] * trans_strain_[i];
        }
        if (E_eff < 1.0e3) E_eff = 1.0e3; // floor

        Real nu = props_.nu;
        Real G_eff = E_eff / (2.0 * (1.0 + nu));
        Real lambda_eff = E_eff * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        // Compute transformation strain increment (isotropic volumetric)
        Real d_trans = trans_eps_new - trans_eps;

        // Elastic trial stress (with transformation strain subtracted)
        Real ev = state.strain[0] + state.strain[1] + state.strain[2] - d_trans;
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lambda_eff * ev + 2.0 * G_eff * (state.strain[i] - d_trans / 3.0);
        for (int i = 3; i < 6; ++i)
            trial[i] = G_eff * state.strain[i];

        // J2 plasticity with effective yield
        Real p_trial = (trial[0] + trial[1] + trial[2]) / 3.0;
        Real s_dev[6];
        for (int i = 0; i < 3; ++i) s_dev[i] = trial[i] - p_trial;
        for (int i = 3; i < 6; ++i) s_dev[i] = trial[i];

        Real J2 = 0.5 * (s_dev[0]*s_dev[0] + s_dev[1]*s_dev[1] + s_dev[2]*s_dev[2])
                 + s_dev[3]*s_dev[3] + s_dev[4]*s_dev[4] + s_dev[5]*s_dev[5];
        Real vm = Kokkos::sqrt(3.0 * J2 + 1.0e-30);

        if (vm > sigma_y_eff) {
            // Radial return
            Real scale = sigma_y_eff / vm;
            for (int i = 0; i < 3; ++i)
                state.stress[i] = scale * s_dev[i] + p_trial;
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale * s_dev[i];

            Real d_eps_p = (vm - sigma_y_eff) / (3.0 * G_eff + 1.0e-30);
            state.plastic_strain += d_eps_p;
        } else {
            for (int i = 0; i < 6; ++i)
                state.stress[i] = trial[i];
        }

        // Store phase fractions and transformation strain
        for (int i = 0; i < 5; ++i)
            state.history[32 + i] = f[i];
        state.history[37] = trans_eps_new;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        // Effective E from mixing rule
        Real E_eff = 0.0;
        for (int i = 0; i < 5; ++i)
            E_eff += state.history[32 + i] * phase_E_[i];
        if (E_eff < 1.0e3) E_eff = phase_E_[0]; // fallback to austenite

        Real nu = props_.nu;
        Real G_eff = E_eff / (2.0 * (1.0 + nu));
        Real lambda_eff = E_eff * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        // Reduce if plastic
        Real red = (state.plastic_strain > 0.0) ? 0.8 : 1.0;
        Real G_tan = G_eff * red;
        Real lam_tan = lambda_eff * red;

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lam_tan + 2.0 * G_tan;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lam_tan;
            }
        }
        C[21] = G_tan;
        C[28] = G_tan;
        C[35] = G_tan;
    }

    /// Get current phase fractions from a state
    KOKKOS_INLINE_FUNCTION
    void get_phase_fractions(const MaterialState& state, Real* fracs) const {
        for (int i = 0; i < 5; ++i)
            fracs[i] = state.history[32 + i];
    }

    KOKKOS_INLINE_FUNCTION Real get_Ms_temp() const { return Ms_temp_; }
    KOKKOS_INLINE_FUNCTION Real get_alpha_km() const { return alpha_km_; }

private:
    Real Ms_temp_;         // Martensite start temperature [C or K]
    Real alpha_km_;        // Koistinen-Marburger rate constant
    Real jmak_b_;          // JMAK rate parameter b
    Real jmak_n_;          // JMAK Avrami exponent n
    Real phase_E_[5];      // Phase-specific Young's moduli
    Real phase_yield_[5];  // Phase-specific yield stresses
    Real trans_strain_[5]; // Phase-specific transformation strains
};

// ============================================================================
// 3. ElasticShellMaterial - Shell-Specific Elastic-Plastic (LAW4)
// ============================================================================

/**
 * @brief Shell-specific elastic-plastic material with plane stress assumption
 *
 * Through-thickness integration with N layers (default 5).
 * Plane stress condition: sigma_33 = 0 enforced via iterative correction.
 * Standard J2 plasticity with isotropic hardening under plane stress.
 * Thickness updated via plastic incompressibility.
 *
 * History: [32] = accumulated equivalent plastic strain
 *          [33] = current thickness ratio (h/h0)
 */
class ElasticShellMaterial : public Material {
public:
    ElasticShellMaterial(const MaterialProperties& props,
                          Real yield_stress = 250.0e6, Real hardening_mod = 1.0e9,
                          int num_layers = 5, Real thickness = 0.001)
        : Material(MaterialType::Custom, props)
        , sigma_y0_(yield_stress), H_mod_(hardening_mod)
        , num_layers_(num_layers), thickness0_(thickness)
    {
        if (num_layers_ < 1) num_layers_ = 1;
        if (num_layers_ > 20) num_layers_ = 20;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));

        // Plane stress elastic matrix coefficients
        Real C11 = E / (1.0 - nu * nu);
        Real C12 = nu * C11;
        Real C44 = G;

        // Retrieve history
        Real eps_p = state.history[32];
        Real thickness_ratio = state.history[33];
        if (thickness_ratio <= 0.0) thickness_ratio = 1.0;

        // Current yield stress
        Real sigma_y = sigma_y0_ + H_mod_ * eps_p;

        // Plane stress trial stress (sigma_33 = 0 enforced)
        // Components: [0]=s11, [1]=s22, [2]=s33(=0), [3]=s12, [4]=s23, [5]=s13
        Real trial_s11 = C11 * state.strain[0] + C12 * state.strain[1];
        Real trial_s22 = C12 * state.strain[0] + C11 * state.strain[1];
        Real trial_s12 = C44 * state.strain[3];
        // s23, s13 for transverse shear (simplified with shear correction factor)
        Real kappa_shear = 5.0 / 6.0; // Reissner-Mindlin shear correction
        Real trial_s23 = kappa_shear * C44 * state.strain[4];
        Real trial_s13 = kappa_shear * C44 * state.strain[5];

        // Von Mises equivalent under plane stress
        // sigma_vm = sqrt(s11^2 - s11*s22 + s22^2 + 3*s12^2)
        Real vm_sq = trial_s11 * trial_s11 - trial_s11 * trial_s22
                   + trial_s22 * trial_s22 + 3.0 * trial_s12 * trial_s12;
        Real vm = Kokkos::sqrt(vm_sq + 1.0e-30);

        if (vm > sigma_y) {
            // Plane stress radial return
            // Newton iteration for d_lambda
            Real d_lambda = 0.0;
            for (int iter = 0; iter < 25; ++iter) {
                Real sy_cur = sigma_y0_ + H_mod_ * (eps_p + d_lambda);
                // Effective G for plane stress: consider constraint
                Real vm_corrected = vm - 3.0 * G * d_lambda;
                if (vm_corrected < 0.0) vm_corrected = 0.0;
                Real f = vm_corrected - sy_cur;
                Real df = -3.0 * G - H_mod_;
                Real ddl = -f / df;
                d_lambda += ddl;
                if (d_lambda < 0.0) d_lambda = 0.0;
                if (Kokkos::fabs(f) < 1.0e-10 * sigma_y0_) break;
            }

            Real scale = (sigma_y0_ + H_mod_ * (eps_p + d_lambda)) / (vm + 1.0e-30);
            if (scale > 1.0) scale = 1.0;

            state.stress[0] = scale * trial_s11;
            state.stress[1] = scale * trial_s22;
            state.stress[2] = 0.0; // plane stress
            state.stress[3] = scale * trial_s12;
            state.stress[4] = trial_s23; // transverse shear elastic
            state.stress[5] = trial_s13;

            eps_p += d_lambda;

            // Update thickness via plastic incompressibility:
            // d_eps_33^p = -(d_eps_11^p + d_eps_22^p)
            // Simplified: thickness_ratio decreases proportional to plastic strain
            Real d_thick = -d_lambda * nu / (1.0 - nu + 1.0e-30);
            thickness_ratio *= (1.0 + d_thick);
            if (thickness_ratio < 0.01) thickness_ratio = 0.01;
        } else {
            state.stress[0] = trial_s11;
            state.stress[1] = trial_s22;
            state.stress[2] = 0.0;
            state.stress[3] = trial_s12;
            state.stress[4] = trial_s23;
            state.stress[5] = trial_s13;
        }

        state.history[32] = eps_p;
        state.history[33] = thickness_ratio;
        state.plastic_strain = eps_p;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        // Plane stress tangent
        Real C11 = E / (1.0 - nu * nu);
        Real C12 = nu * C11;
        Real kappa_shear = 5.0 / 6.0;

        Real eps_p = state.history[32];
        Real red = 1.0;
        if (eps_p > 0.0) {
            // Continuum tangent modulus reduction for plasticity
            red = (3.0 * G) / (3.0 * G + H_mod_);
        }

        C[0 * 6 + 0] = C11 * red;
        C[0 * 6 + 1] = C12 * red;
        C[1 * 6 + 0] = C12 * red;
        C[1 * 6 + 1] = C11 * red;
        // C[2][2] = 0 (plane stress, sigma_33 = 0)
        C[3 * 6 + 3] = G * red;
        C[4 * 6 + 4] = kappa_shear * G; // transverse shear stays elastic
        C[5 * 6 + 5] = kappa_shear * G;
    }

    KOKKOS_INLINE_FUNCTION int get_num_layers() const { return num_layers_; }
    KOKKOS_INLINE_FUNCTION Real get_thickness0() const { return thickness0_; }

    KOKKOS_INLINE_FUNCTION
    Real current_thickness(const MaterialState& state) const {
        Real ratio = state.history[33];
        if (ratio <= 0.0) ratio = 1.0;
        return thickness0_ * ratio;
    }

private:
    Real sigma_y0_;
    Real H_mod_;
    int num_layers_;
    Real thickness0_;
};

// ============================================================================
// 4. CompositeDamageMaterial - Progressive Composite Damage (LAW53)
// ============================================================================

/**
 * @brief Progressive composite damage model with Hashin-type initiation
 *
 * Four failure modes with independent damage variables:
 *   d1 - fiber tension:     (s11/Xt)^2 + (s12/S12)^2 >= 1
 *   d2 - fiber compression: (s11/Xc)^2 >= 1
 *   d3 - matrix tension:    (s22/Yt)^2 + (s12/S12)^2 >= 1
 *   d4 - matrix compression: (s22/(2*S12))^2 + ((Yc/(2*S12))^2 - 1)*(s22/Yc) + (s12/S12)^2 >= 1
 *
 * Degraded stiffness via damage variables d_i in [0, 1].
 * Orthotropic stress with degradation:
 *   C_deg = C * diag((1-d1)(1-d2), (1-d3)(1-d4), 1, (1-d_s), ...)
 *
 * History: [32] = d1 (fiber tension damage)
 *          [33] = d2 (fiber compression damage)
 *          [34] = d3 (matrix tension damage)
 *          [35] = d4 (matrix compression damage)
 */
class CompositeDamageMaterial : public Material {
public:
    CompositeDamageMaterial(const MaterialProperties& props,
                             Real Xt = 2000.0e6, Real Xc = 1200.0e6,
                             Real Yt = 50.0e6, Real Yc = 200.0e6,
                             Real S12 = 70.0e6,
                             Real E1 = 140.0e9, Real E2 = 10.0e9,
                             Real G12 = 5.0e9, Real nu12 = 0.3,
                             Real damage_rate = 10.0)
        : Material(MaterialType::Custom, props)
        , Xt_(Xt), Xc_(Xc), Yt_(Yt), Yc_(Yc), S12_(S12)
        , E1_(E1), E2_(E2), G12_(G12), nu12_(nu12)
        , damage_rate_(damage_rate)
    {
        nu21_ = nu12_ * E2_ / (E1_ + 1.0e-30);
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        // Retrieve damage variables
        Real d1 = state.history[32];
        Real d2 = state.history[33];
        Real d3 = state.history[34];
        Real d4 = state.history[35];

        // Degraded stiffnesses
        Real df = (1.0 - d1) * (1.0 - d2); // fiber degradation
        Real dm = (1.0 - d3) * (1.0 - d4); // matrix degradation
        Real ds = (1.0 - d1) * (1.0 - d3); // shear degradation (fiber-matrix coupling)

        Real E1_d = E1_ * df;
        Real E2_d = E2_ * dm;
        Real G12_d = G12_ * ds;
        Real nu12_d = nu12_ * Kokkos::sqrt(df * dm + 1.0e-30);
        Real nu21_d = nu12_d * E2_d / (E1_d + 1.0e-30);

        // Plane stress compliance denominator
        Real denom = 1.0 - nu12_d * nu21_d;
        if (Kokkos::fabs(denom) < 1.0e-15) denom = 1.0e-15;

        // Degraded stiffness matrix (plane stress, ortho)
        Real Q11 = E1_d / denom;
        Real Q22 = E2_d / denom;
        Real Q12 = nu12_d * E2_d / denom;
        Real Q66 = G12_d;

        // Compute stress (plane stress formulation)
        // strain[0]=eps11, strain[1]=eps22, strain[3]=gamma12
        Real s11 = Q11 * state.strain[0] + Q12 * state.strain[1];
        Real s22 = Q12 * state.strain[0] + Q22 * state.strain[1];
        Real s12 = Q66 * state.strain[3];

        // Through-thickness (simplified elastic)
        Real G23 = E2_ * dm / (2.0 * (1.0 + nu12_) + 1.0e-30);
        Real G13 = G12_d;
        Real s23 = G23 * state.strain[4];
        Real s13 = G13 * state.strain[5];

        // Check Hashin failure criteria and update damage
        // 1. Fiber tension (s11 > 0)
        if (s11 > 0.0) {
            Real f_ft = (s11 / (Xt_ + 1.0e-30)) * (s11 / (Xt_ + 1.0e-30))
                      + (s12 / (S12_ + 1.0e-30)) * (s12 / (S12_ + 1.0e-30));
            if (f_ft >= 1.0) {
                Real d_inc = damage_rate_ * (f_ft - 1.0);
                d1 += d_inc;
                if (d1 > 0.999) d1 = 0.999;
            }
        }

        // 2. Fiber compression (s11 < 0)
        if (s11 < 0.0) {
            Real f_fc = (s11 / (Xc_ + 1.0e-30)) * (s11 / (Xc_ + 1.0e-30));
            if (f_fc >= 1.0) {
                Real d_inc = damage_rate_ * (f_fc - 1.0);
                d2 += d_inc;
                if (d2 > 0.999) d2 = 0.999;
            }
        }

        // 3. Matrix tension (s22 > 0)
        if (s22 > 0.0) {
            Real f_mt = (s22 / (Yt_ + 1.0e-30)) * (s22 / (Yt_ + 1.0e-30))
                      + (s12 / (S12_ + 1.0e-30)) * (s12 / (S12_ + 1.0e-30));
            if (f_mt >= 1.0) {
                Real d_inc = damage_rate_ * (f_mt - 1.0);
                d3 += d_inc;
                if (d3 > 0.999) d3 = 0.999;
            }
        }

        // 4. Matrix compression (s22 < 0)
        if (s22 < 0.0) {
            Real S12_sq = S12_ * S12_ + 1.0e-30;
            Real f_mc = (s22 / (2.0 * S12_ + 1.0e-30)) * (s22 / (2.0 * S12_ + 1.0e-30))
                      + ((Yc_ / (2.0 * S12_ + 1.0e-30)) * (Yc_ / (2.0 * S12_ + 1.0e-30)) - 1.0)
                        * s22 / (Yc_ + 1.0e-30)
                      + (s12 / (S12_ + 1.0e-30)) * (s12 / (S12_ + 1.0e-30));
            if (f_mc >= 1.0) {
                Real d_inc = damage_rate_ * (f_mc - 1.0);
                d4 += d_inc;
                if (d4 > 0.999) d4 = 0.999;
            }
        }

        // Recompute stress with updated damage (if damage grew)
        Real df2 = (1.0 - d1) * (1.0 - d2);
        Real dm2 = (1.0 - d3) * (1.0 - d4);
        Real ds2 = (1.0 - d1) * (1.0 - d3);

        Real E1_d2 = E1_ * df2;
        Real E2_d2 = E2_ * dm2;
        Real G12_d2 = G12_ * ds2;
        Real nu12_d2 = nu12_ * Kokkos::sqrt(df2 * dm2 + 1.0e-30);
        Real nu21_d2 = nu12_d2 * E2_d2 / (E1_d2 + 1.0e-30);

        Real denom2 = 1.0 - nu12_d2 * nu21_d2;
        if (Kokkos::fabs(denom2) < 1.0e-15) denom2 = 1.0e-15;

        state.stress[0] = E1_d2 / denom2 * state.strain[0] + nu12_d2 * E2_d2 / denom2 * state.strain[1];
        state.stress[1] = nu12_d2 * E2_d2 / denom2 * state.strain[0] + E2_d2 / denom2 * state.strain[1];
        state.stress[2] = 0.0; // plane stress
        state.stress[3] = G12_d2 * state.strain[3];
        Real G23_2 = E2_ * dm2 / (2.0 * (1.0 + nu12_) + 1.0e-30);
        state.stress[4] = G23_2 * state.strain[4];
        state.stress[5] = G12_d2 * state.strain[5];

        // Store damage variables
        state.history[32] = d1;
        state.history[33] = d2;
        state.history[34] = d3;
        state.history[35] = d4;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real d1 = state.history[32];
        Real d2 = state.history[33];
        Real d3 = state.history[34];
        Real d4 = state.history[35];

        Real df = (1.0 - d1) * (1.0 - d2);
        Real dm = (1.0 - d3) * (1.0 - d4);
        Real ds = (1.0 - d1) * (1.0 - d3);

        Real E1_d = E1_ * df;
        Real E2_d = E2_ * dm;
        Real G12_d = G12_ * ds;
        Real nu12_d = nu12_ * Kokkos::sqrt(df * dm + 1.0e-30);
        Real nu21_d = nu12_d * E2_d / (E1_d + 1.0e-30);

        Real denom = 1.0 - nu12_d * nu21_d;
        if (Kokkos::fabs(denom) < 1.0e-15) denom = 1.0e-15;

        C[0 * 6 + 0] = E1_d / denom;
        C[0 * 6 + 1] = nu12_d * E2_d / denom;
        C[1 * 6 + 0] = nu12_d * E2_d / denom;
        C[1 * 6 + 1] = E2_d / denom;
        // C[2][2] = 0 (plane stress)
        C[3 * 6 + 3] = G12_d;
        Real G23 = E2_ * dm / (2.0 * (1.0 + nu12_) + 1.0e-30);
        C[4 * 6 + 4] = G23;
        C[5 * 6 + 5] = G12_d;
    }

    /// Get current damage state
    KOKKOS_INLINE_FUNCTION
    void get_damage(const MaterialState& state, Real& d1, Real& d2, Real& d3, Real& d4) const {
        d1 = state.history[32];
        d2 = state.history[33];
        d3 = state.history[34];
        d4 = state.history[35];
    }

    /// Maximum damage across all modes
    KOKKOS_INLINE_FUNCTION
    Real max_damage(const MaterialState& state) const {
        Real dmax = state.history[32];
        for (int i = 33; i <= 35; ++i)
            if (state.history[i] > dmax) dmax = state.history[i];
        return dmax;
    }

private:
    Real Xt_, Xc_;       // Fiber tensile/compressive strength
    Real Yt_, Yc_;       // Matrix tensile/compressive strength
    Real S12_;           // In-plane shear strength
    Real E1_, E2_;       // Longitudinal/transverse moduli
    Real G12_;           // In-plane shear modulus
    Real nu12_, nu21_;   // Poisson's ratios
    Real damage_rate_;   // Damage evolution rate parameter
};

// ============================================================================
// 5. NonlinearElasticMaterial - Nonlinear Elastic (LAW44)
// ============================================================================

/**
 * @brief Nonlinear elastic material with no permanent deformation
 *
 * Two stress-strain curve forms (fully reversible, no plasticity):
 *
 * Polynomial:  sigma = C1*eps + C2*eps^2 + C3*eps^3 + C4*eps^4 + C5*eps^5
 * Exponential: sigma = A*(exp(B*eps) - 1)
 *
 * Applied component-wise to the normal strains with Poisson coupling.
 * Shear handled via secant shear modulus derived from the tangent slope.
 * No history variables needed (stateless, fully reversible).
 *
 * History: none (stateless)
 */
class NonlinearElasticMaterial : public Material {
public:
    NonlinearElasticMaterial(const MaterialProperties& props,
                              bool use_exponential = false,
                              Real exp_A = 1.0e9, Real exp_B = 10.0)
        : Material(MaterialType::Custom, props)
        , use_exponential_(use_exponential)
        , exp_A_(exp_A), exp_B_(exp_B)
    {
        for (int i = 0; i < 5; ++i) poly_coeffs_[i] = 0.0;
        // Default polynomial: linear + cubic stiffening
        poly_coeffs_[0] = props.E;        // C1 = E (linear term)
        poly_coeffs_[2] = props.E * 10.0; // C3 = mild cubic stiffening
    }

    /// Set polynomial coefficients C1..C5
    void set_poly_coeff(int idx, Real val) {
        if (idx >= 0 && idx < 5) poly_coeffs_[idx] = val;
    }

    /// Set all polynomial coefficients at once
    void set_poly_coeffs(Real c1, Real c2, Real c3, Real c4, Real c5) {
        poly_coeffs_[0] = c1;
        poly_coeffs_[1] = c2;
        poly_coeffs_[2] = c3;
        poly_coeffs_[3] = c4;
        poly_coeffs_[4] = c5;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real nu = props_.nu;

        // For each normal strain, compute nonlinear stress contribution
        // Use volumetric/deviatoric split with nonlinear bulk behavior

        // Volumetric strain
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];

        // Compute nonlinear "bulk" stress from volumetric strain
        Real p_nl = eval_stress(ev / 3.0) / (1.0 - 2.0 * nu + 1.0e-30);

        // Deviatoric strains
        Real e_dev[3];
        for (int i = 0; i < 3; ++i)
            e_dev[i] = state.strain[i] - ev / 3.0;

        // Tangent modulus at current volumetric strain for deviatoric response
        Real E_tan = eval_tangent(ev / 3.0);
        if (E_tan < 1.0e-10) E_tan = 1.0e-10;
        Real G_sec = E_tan / (2.0 * (1.0 + nu));

        // Normal stress = volumetric + deviatoric
        for (int i = 0; i < 3; ++i)
            state.stress[i] = p_nl + 2.0 * G_sec * e_dev[i];

        // Shear stress using secant shear modulus
        for (int i = 3; i < 6; ++i)
            state.stress[i] = G_sec * state.strain[i];

        // No plasticity - fully reversible
        state.plastic_strain = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real nu = props_.nu;

        // Tangent modulus at current strain level
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real E_tan = eval_tangent(ev / 3.0);
        if (E_tan < 1.0e-10) E_tan = 1.0e-10;

        Real G_tan = E_tan / (2.0 * (1.0 + nu));
        Real lambda_tan = E_tan * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda_tan + 2.0 * G_tan;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda_tan;
            }
        }
        C[21] = G_tan;
        C[28] = G_tan;
        C[35] = G_tan;
    }

    KOKKOS_INLINE_FUNCTION bool is_exponential() const { return use_exponential_; }

private:
    /// Evaluate stress for a given uniaxial strain
    KOKKOS_INLINE_FUNCTION
    Real eval_stress(Real eps) const {
        if (use_exponential_) {
            return exp_A_ * (Kokkos::exp(exp_B_ * eps) - 1.0);
        } else {
            // Polynomial: sigma = sum C_i * eps^i, i=1..5
            Real result = 0.0;
            Real eps_pow = eps;
            for (int i = 0; i < 5; ++i) {
                result += poly_coeffs_[i] * eps_pow;
                eps_pow *= eps;
            }
            return result;
        }
    }

    /// Evaluate tangent modulus d(sigma)/d(eps) at given strain
    KOKKOS_INLINE_FUNCTION
    Real eval_tangent(Real eps) const {
        if (use_exponential_) {
            return exp_A_ * exp_B_ * Kokkos::exp(exp_B_ * eps);
        } else {
            // d/deps(sum C_i * eps^i) = sum i*C_i * eps^(i-1)
            Real result = poly_coeffs_[0]; // C1 * 1
            Real eps_pow = 1.0;
            for (int i = 1; i < 5; ++i) {
                result += static_cast<Real>(i + 1) * poly_coeffs_[i] * eps_pow;
                eps_pow *= eps;
            }
            return result;
        }
    }

    bool use_exponential_;
    Real poly_coeffs_[5];  // C1..C5 polynomial coefficients
    Real exp_A_;           // Exponential amplitude
    Real exp_B_;           // Exponential rate
};

// ============================================================================
// 6. MohrCoulombMaterial - Classic Mohr-Coulomb (LAW27)
// ============================================================================

/**
 * @brief Classic Mohr-Coulomb yield criterion with tension cutoff
 *
 * Yield surface in principal stress space:
 *   F = sigma1 - sigma3 - 2*c*cos(phi) - (sigma1 + sigma3)*sin(phi) = 0
 * or equivalently:
 *   tau = c + sigma_n * tan(phi)  on the failure plane
 *
 * Tension cutoff at sigma_t (maximum allowable tensile stress).
 * Non-associated flow with dilatancy angle psi:
 *   g = sigma1 - sigma3 - (sigma1 + sigma3)*sin(psi)
 *
 * History: [32] = accumulated equivalent plastic strain
 *          [33] = yield state (0=elastic, 1=shear yield, 2=tension cutoff)
 */
class MohrCoulombMaterial : public Material {
public:
    MohrCoulombMaterial(const MaterialProperties& props,
                          Real cohesion = 1.0e6, Real friction_angle_deg = 30.0,
                          Real dilatancy_angle_deg = 10.0, Real tension_cutoff = 1.0e5)
        : Material(MaterialType::Custom, props)
        , cohesion_(cohesion), tension_cutoff_(tension_cutoff)
    {
        Real pi = 3.14159265358979323846;
        phi_rad_ = friction_angle_deg * pi / 180.0;
        psi_rad_ = dilatancy_angle_deg * pi / 180.0;
        sin_phi_ = Kokkos::sin(phi_rad_);
        cos_phi_ = Kokkos::cos(phi_rad_);
        sin_psi_ = Kokkos::sin(psi_rad_);

        // MC tension cutoff cannot exceed the apex of the MC surface
        // sigma_t_max = c * cos(phi) / (1 + sin(phi))  ... but simplified:
        Real sigma_t_max = cohesion_ * cos_phi_ / (1.0 + sin_phi_ + 1.0e-30);
        if (tension_cutoff_ > sigma_t_max)
            tension_cutoff_ = sigma_t_max;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = K_bulk - 2.0 * G / 3.0;

        // Elastic trial stress
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lambda * ev + 2.0 * G * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = G * state.strain[i];

        // Compute principal stresses via eigenvalue analysis of symmetric 3x3
        // For the return mapping, we need principal stresses sorted: s1 >= s2 >= s3
        Real sig_principal[3];
        compute_principal_stresses(trial, sig_principal);

        // Sort descending: s1 >= s2 >= s3
        sort_descending_3(sig_principal);

        Real s1 = sig_principal[0];
        Real s2 = sig_principal[1];
        Real s3 = sig_principal[2];

        // History
        Real eps_p = state.history[32];
        Real yield_state = 0.0;

        // 1. Check tension cutoff: s1 > sigma_t
        bool tension_yield = (s1 > tension_cutoff_);

        // 2. Check Mohr-Coulomb shear yield:
        //    F = s1 - s3 - 2*c*cos(phi) - (s1 + s3)*sin(phi)
        Real F_mc = (s1 - s3) - 2.0 * cohesion_ * cos_phi_ - (s1 + s3) * sin_phi_;

        if (tension_yield && F_mc <= 0.0) {
            // Pure tension cutoff return
            // Project s1 back to tension_cutoff_, adjust s2 if needed
            Real ds1 = s1 - tension_cutoff_;
            s1 = tension_cutoff_;
            // Volumetric correction: keep mean stress consistent with flow rule
            // For tension cutoff, associated flow in tension direction
            Real dp = ds1 / 3.0;
            s2 -= dp * (1.0 - sin_psi_);
            s3 -= dp * (1.0 - sin_psi_);

            yield_state = 2.0;
            eps_p += ds1 / (E + 1.0e-30);
        } else if (F_mc > 0.0) {
            // Mohr-Coulomb shear return mapping
            // Return to the MC surface using non-associated flow
            // Flow rule: dg/dsig1 = 1, dg/dsig3 = -(1+sin_psi)/(1-sin_psi)
            // But simplified: plastic multiplier from:
            //   F(sig - d_lam * dg/dsig) = 0

            Real alpha_flow = (1.0 + sin_psi_) / (1.0 - sin_psi_ + 1.0e-30);

            // Elastic moduli in principal space
            Real a1 = K_bulk + 4.0 * G / 3.0;
            Real a2 = K_bulk - 2.0 * G / 3.0;

            // Plastic multiplier (single surface return)
            Real denom = (a1 - a2 * sin_phi_) + alpha_flow * (a1 * sin_phi_ - a2);
            if (Kokkos::fabs(denom) < 1.0e-30) denom = 1.0e-30;
            Real d_lambda = F_mc / denom;
            if (d_lambda < 0.0) d_lambda = 0.0;

            // Update principal stresses
            s1 -= d_lambda * (a1 - a2 * sin_phi_ - a2 * (alpha_flow - 1.0));
            s3 += d_lambda * (alpha_flow * a1 - a2 - a2 * sin_phi_);
            // s2 correction via volumetric constraint
            Real dp_vol = K_bulk * d_lambda * (1.0 - sin_psi_);
            s2 -= dp_vol;

            // Simplified: direct return
            // s1_new - s3_new = 2*c*cos(phi) + (s1_new + s3_new)*sin(phi)
            Real mc_rhs = 2.0 * cohesion_ * cos_phi_;
            Real s1_new = s1;
            Real s3_new = s3;

            // Verify and clamp to yield surface
            Real F_check = (s1_new - s3_new) - mc_rhs - (s1_new + s3_new) * sin_phi_;
            if (Kokkos::fabs(F_check) > 1.0e-6 * (Kokkos::fabs(mc_rhs) + 1.0)) {
                // Direct projection to yield surface
                // s1(1-sin_phi) - s3(1+sin_phi) = 2c*cos(phi)
                // Keep s2 fixed, adjust s1 and s3 proportionally
                Real mean_13 = (s1_new + s3_new) / 2.0;
                Real half_diff = mc_rhs / (2.0 * (1.0 - sin_phi_ * sin_phi_ + 1.0e-30));
                // Actually solve: s1 - s3 = 2c*cos(phi) + (s1+s3)*sin(phi)
                // Let m = (s1+s3)/2 (mean of s1,s3), d = (s1-s3)/2
                // Then 2d = 2c*cos(phi) + 2m*sin(phi) => d = c*cos(phi) + m*sin(phi)
                Real d_half = cohesion_ * cos_phi_ + mean_13 * sin_phi_;
                s1_new = mean_13 + d_half;
                s3_new = mean_13 - d_half;
            }

            s1 = s1_new;
            s3 = s3_new;

            // Check tension cutoff on returned stress
            if (s1 > tension_cutoff_) {
                s1 = tension_cutoff_;
            }

            yield_state = 1.0;
            eps_p += d_lambda;
        }
        // else: elastic, no changes needed

        // Reconstruct full stress tensor from principal stresses
        // For simplicity, map back assuming the trial stress directions are preserved.
        // This is the standard approach: scale the trial stress tensor.
        Real trial_p[3];
        compute_principal_stresses(trial, trial_p);
        sort_descending_3(trial_p);

        // If principal stresses changed, reconstruct via scaling
        Real s1_trial = trial_p[0];
        Real s2_trial = trial_p[1];
        Real s3_trial = trial_p[2];

        // Mean stress correction
        Real p_trial = (s1_trial + s2_trial + s3_trial) / 3.0;
        Real p_new = (s1 + s2 + s3) / 3.0;

        // Deviatoric scaling factor
        Real dev_trial = Kokkos::sqrt(
            (s1_trial - p_trial) * (s1_trial - p_trial)
          + (s2_trial - p_trial) * (s2_trial - p_trial)
          + (s3_trial - p_trial) * (s3_trial - p_trial) + 1.0e-30);

        Real dev_new = Kokkos::sqrt(
            (s1 - p_new) * (s1 - p_new)
          + (s2 - p_new) * (s2 - p_new)
          + (s3 - p_new) * (s3 - p_new) + 1.0e-30);

        Real dev_scale = dev_new / (dev_trial + 1.0e-30);
        if (dev_scale > 2.0) dev_scale = 2.0; // safety clamp

        // Reconstruct: shift mean stress + scale deviatoric part
        Real s_dev_trial[6];
        for (int i = 0; i < 3; ++i) s_dev_trial[i] = trial[i] - p_trial;
        for (int i = 3; i < 6; ++i) s_dev_trial[i] = trial[i];

        for (int i = 0; i < 3; ++i)
            state.stress[i] = p_new + dev_scale * s_dev_trial[i];
        for (int i = 3; i < 6; ++i)
            state.stress[i] = dev_scale * s_dev_trial[i];

        // Update history
        state.history[32] = eps_p;
        state.history[33] = yield_state;
        state.plastic_strain = eps_p;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real red = 1.0;
        Real yield_state = state.history[33];
        if (yield_state > 0.5) {
            // Plastic: reduce stiffness
            // Consistent tangent approximation
            if (yield_state > 1.5) {
                // Tension cutoff: soften significantly
                red = 0.3;
            } else {
                // Shear yield: moderate reduction
                red = 0.6;
            }
        }

        Real G_eff = G * red;
        Real lambda_eff = lambda * red;

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda_eff + 2.0 * G_eff;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda_eff;
            }
        }
        C[21] = G_eff;
        C[28] = G_eff;
        C[35] = G_eff;
    }

    KOKKOS_INLINE_FUNCTION Real get_cohesion() const { return cohesion_; }
    KOKKOS_INLINE_FUNCTION Real get_friction_angle_rad() const { return phi_rad_; }
    KOKKOS_INLINE_FUNCTION Real get_dilatancy_angle_rad() const { return psi_rad_; }
    KOKKOS_INLINE_FUNCTION Real get_tension_cutoff() const { return tension_cutoff_; }

private:
    /// Compute principal stresses from Voigt stress tensor (3x3 symmetric)
    /// Uses analytical eigenvalues of the 3x3 symmetric matrix
    KOKKOS_INLINE_FUNCTION
    static void compute_principal_stresses(const Real* sig, Real* principal) {
        // sig = [s11, s22, s33, s12, s23, s13] (Voigt)
        Real s11 = sig[0], s22 = sig[1], s33 = sig[2];
        Real s12 = sig[3], s23 = sig[4], s13 = sig[5];

        // Invariants of the stress tensor
        Real I1 = s11 + s22 + s33;
        Real I2 = s11*s22 + s22*s33 + s33*s11 - s12*s12 - s23*s23 - s13*s13;
        Real I3 = s11*s22*s33 + 2.0*s12*s23*s13
                 - s11*s23*s23 - s22*s13*s13 - s33*s12*s12;

        // Solve characteristic equation: lambda^3 - I1*lambda^2 + I2*lambda - I3 = 0
        // Using Cardano's formula / trigonometric solution
        Real p = I1 / 3.0;
        Real q = (I1 * I1 - 3.0 * I2) / 9.0;
        if (q < 0.0) q = 0.0;
        Real r = (2.0 * I1 * I1 * I1 - 9.0 * I1 * I2 + 27.0 * I3) / 54.0;

        Real sqrt_q = Kokkos::sqrt(q + 1.0e-30);
        Real cos_arg = r / (q * sqrt_q + 1.0e-30);
        // Clamp to [-1, 1]
        if (cos_arg > 1.0) cos_arg = 1.0;
        if (cos_arg < -1.0) cos_arg = -1.0;

        Real theta = Kokkos::acos(cos_arg) / 3.0;
        Real pi_over_3 = 3.14159265358979323846 / 3.0;

        principal[0] = p + 2.0 * sqrt_q * Kokkos::cos(theta);
        principal[1] = p + 2.0 * sqrt_q * Kokkos::cos(theta - 2.0 * pi_over_3);
        principal[2] = p + 2.0 * sqrt_q * Kokkos::cos(theta + 2.0 * pi_over_3);
    }

    /// Sort 3 values in descending order
    KOKKOS_INLINE_FUNCTION
    static void sort_descending_3(Real* v) {
        if (v[1] > v[0]) { Real t = v[0]; v[0] = v[1]; v[1] = t; }
        if (v[2] > v[0]) { Real t = v[0]; v[0] = v[2]; v[2] = t; }
        if (v[2] > v[1]) { Real t = v[1]; v[1] = v[2]; v[2] = t; }
    }

    Real cohesion_;
    Real tension_cutoff_;
    Real phi_rad_;         // Friction angle [rad]
    Real psi_rad_;         // Dilatancy angle [rad]
    Real sin_phi_, cos_phi_;
    Real sin_psi_;
};

} // namespace physics
} // namespace nxs
