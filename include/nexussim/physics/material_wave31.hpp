#pragma once

/**
 * @file material_wave31.hpp
 * @brief Wave 31 material models: 15 constitutive models
 *
 * Models included:
 *   1.  DruckerPragerExtMaterial       - DP with associated/non-associated flow (LAW21)
 *   2.  ConcreteMaterial               - 3-surface concrete model (LAW24)
 *   3.  SesameTabMaterial              - Tabulated EOS material (LAW26)
 *   4.  BiphasicMaterial               - Solid skeleton + pore fluid (LAW37)
 *   5.  ViscousTabMaterial             - Generalized Maxwell with tabulated relaxation (LAW38)
 *   6.  KelvinMaxwellMaterial          - Kelvin-Maxwell chain (LAW40)
 *   7.  LeeTarverReactiveMaterial      - Lee-Tarver ignition & growth (LAW41)
 *   8.  FluffMaterial                  - Ultra-soft padding (LAW45)
 *   9.  LESFluidMaterial               - Large eddy simulation fluid (LAW46)
 *  10.  MultiMaterialMaterial          - Multi-material ALE (LAW51)
 *  11.  PlasticTriangleMaterial        - Triangle plastic with hourglass (LAW60)
 *  12.  HanselHotFormMaterial          - Hot forming with recrystallization (LAW63)
 *  13.  UgineALZMaterial              - Stainless steel forming (LAW64)
 *  14.  CosseratMaterial               - Micropolar continuum (LAW68)
 *  15.  YuModelMaterial                - Yu unified strength for geomaterials (LAW78)
 */

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

// ============================================================================
// 1. DruckerPragerExtMaterial - DP with associated/non-associated flow (LAW21)
// ============================================================================

/**
 * @brief Drucker-Prager with associated/non-associated flow rule
 *
 * Yield: F = sqrt(J2) + alpha*I1 - k
 * Flow potential: g = sqrt(J2) + beta*I1
 * where alpha = 2*sin(phi)/(sqrt(3)*(3-sin(phi))),
 *       beta  = 2*sin(psi)/(sqrt(3)*(3-sin(psi))),
 *       k depends on cohesion and hardening.
 *
 * History: [32]=equivalent plastic strain, [33]=accumulated volumetric plastic strain
 */
class DruckerPragerExtMaterial : public Material {
public:
    DruckerPragerExtMaterial(const MaterialProperties& props,
                              Real cohesion = 1.0e6, Real friction_angle = 30.0,
                              Real dilation_angle = 15.0, Real hardening_modulus = 0.0)
        : Material(MaterialType::Custom, props)
        , cohesion_(cohesion), hardening_mod_(hardening_modulus)
    {
        // Convert degrees to radians
        Real phi_rad = friction_angle * 3.14159265358979323846 / 180.0;
        Real psi_rad = dilation_angle * 3.14159265358979323846 / 180.0;
        Real sin_phi = Kokkos::sin(phi_rad);
        Real sin_psi = Kokkos::sin(psi_rad);
        alpha_ = 2.0 * sin_phi / (1.7320508075688772 * (3.0 - sin_phi));
        beta_ = 2.0 * sin_psi / (1.7320508075688772 * (3.0 - sin_psi));
        Real cos_phi = Kokkos::cos(phi_rad);
        k0_ = 6.0 * cohesion_ * cos_phi / (1.7320508075688772 * (3.0 - sin_phi));
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

        // Invariants
        Real I1 = trial[0] + trial[1] + trial[2];
        Real p = I1 / 3.0;
        Real s[6];
        for (int i = 0; i < 3; ++i) s[i] = trial[i] - p;
        for (int i = 3; i < 6; ++i) s[i] = trial[i];

        Real J2 = 0.5 * (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]) +
                   s[3]*s[3] + s[4]*s[4] + s[5]*s[5];
        Real sqrtJ2 = Kokkos::sqrt(J2 + 1.0e-30);

        // Current hardening
        Real eps_p = state.history[32];
        Real k = k0_ + hardening_mod_ * eps_p;

        // Yield function
        Real F_yield = sqrtJ2 + alpha_ * I1 - k;

        if (F_yield <= 0.0) {
            // Elastic
            for (int i = 0; i < 6; ++i)
                state.stress[i] = trial[i];
        } else {
            // Plastic return mapping
            // d_gamma from: F(sigma - d_gamma * dg/dsigma) = 0
            Real denom = G + K_bulk * alpha_ * beta_ + hardening_mod_ / 3.0;
            if (denom < 1.0e-30) denom = 1.0e-30;
            Real d_gamma = F_yield / denom;

            // Update stress: deviatoric correction + volumetric correction
            Real factor = 1.0 - G * d_gamma / (sqrtJ2 + 1.0e-30);
            if (factor < 0.0) factor = 0.0;
            for (int i = 0; i < 3; ++i)
                state.stress[i] = factor * s[i] + (p - K_bulk * beta_ * d_gamma);
            for (int i = 3; i < 6; ++i)
                state.stress[i] = factor * trial[i];

            // Update history
            Real d_eps_p = d_gamma * (1.0 / (1.7320508075688772 + 1.0e-30));
            state.history[32] = eps_p + d_eps_p;
            state.history[33] += beta_ * d_gamma;
            state.plastic_strain = state.history[32];
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * mu;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda;
            }
        }
        C[21] = mu;
        C[28] = mu;
        C[35] = mu;

        // If plastic, reduce stiffness
        if (state.plastic_strain > 0.0) {
            Real red = 0.8;
            for (int i = 0; i < 36; ++i) C[i] *= red;
        }
    }

    KOKKOS_INLINE_FUNCTION Real get_alpha() const { return alpha_; }
    KOKKOS_INLINE_FUNCTION Real get_beta() const { return beta_; }
    KOKKOS_INLINE_FUNCTION Real get_k0() const { return k0_; }

private:
    Real cohesion_;
    Real hardening_mod_;
    Real alpha_;       // Friction parameter for yield
    Real beta_;        // Dilation parameter for flow
    Real k0_;          // Initial cohesion-related yield parameter
};

// ============================================================================
// 2. ConcreteMaterial - 3-surface concrete (LAW24)
// ============================================================================

/**
 * @brief 3-surface concrete model with tension cutoff, DP compression cap, softening
 *
 * Tension cutoff at ft. Drucker-Prager compression surface.
 * Cap surface for confinement. Damage accumulation in tension and compression.
 *
 * History: [32]=damage_t (tensile damage), [33]=damage_c (compressive damage),
 *          [34]=kappa_t (tensile hardening variable), [35]=kappa_c (compressive hardening variable)
 */
class ConcreteMaterial : public Material {
public:
    ConcreteMaterial(const MaterialProperties& props,
                      Real fc = 30.0e6, Real ft = 3.0e6, Real crush_strain = 0.003)
        : Material(MaterialType::Custom, props)
        , fc_(fc), ft_(ft), crush_strain_(crush_strain)
    {
        // Softening slopes
        Real E = props.E;
        eps_t0_ = ft_ / (E + 1.0e-30);
        eps_c0_ = fc_ / (E + 1.0e-30);
        // Fracture energy based softening
        Gf_t_ = 2.0 * ft_ * eps_t0_ * 0.01; // Characteristic fracture energy
        Gf_c_ = 2.0 * fc_ * crush_strain_;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = K_bulk - 2.0 * G / 3.0;

        // Elastic trial
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lambda * ev + 2.0 * G * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = G * state.strain[i];

        // Principal stress approximation (use I1 and J2 for efficiency)
        Real I1 = trial[0] + trial[1] + trial[2];
        Real p = I1 / 3.0;
        Real s[6];
        for (int i = 0; i < 3; ++i) s[i] = trial[i] - p;
        for (int i = 3; i < 6; ++i) s[i] = trial[i];
        Real J2 = 0.5 * (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]) +
                   s[3]*s[3] + s[4]*s[4] + s[5]*s[5];
        Real sqrtJ2 = Kokkos::sqrt(J2 + 1.0e-30);

        // Retrieve damage
        Real d_t = state.history[32];
        Real d_c = state.history[33];
        Real kappa_t = state.history[34];
        Real kappa_c = state.history[35];

        // Tension cutoff surface check
        Real sigma_max = p + sqrtJ2; // Approximate max principal
        if (sigma_max > ft_ * (1.0 - d_t)) {
            // Tensile damage evolution
            Real eps_eq_t = sigma_max / (E + 1.0e-30);
            if (eps_eq_t > kappa_t) {
                kappa_t = eps_eq_t;
                Real excess = kappa_t - eps_t0_;
                if (excess > 0.0) {
                    Real eps_f_t = 10.0 * eps_t0_;
                    Real ratio = excess / (eps_f_t + 1.0e-30);
                    d_t = 1.0 - Kokkos::exp(-3.0 * ratio);
                    if (d_t > 0.99) d_t = 0.99;
                    if (d_t < 0.0) d_t = 0.0;
                }
            }
        }

        // Compression cap check
        Real sigma_min = p - sqrtJ2; // Approximate min principal
        if (sigma_min < -fc_ * (1.0 - d_c)) {
            // Compressive damage evolution
            Real eps_eq_c = Kokkos::fabs(sigma_min) / (E + 1.0e-30);
            if (eps_eq_c > kappa_c) {
                kappa_c = eps_eq_c;
                Real excess = kappa_c - eps_c0_;
                if (excess > 0.0) {
                    Real ratio = excess / (crush_strain_ + 1.0e-30);
                    d_c = 1.0 - Kokkos::exp(-2.0 * ratio);
                    if (d_c > 0.99) d_c = 0.99;
                    if (d_c < 0.0) d_c = 0.0;
                }
            }
        }

        // Update history
        state.history[32] = d_t;
        state.history[33] = d_c;
        state.history[34] = kappa_t;
        state.history[35] = kappa_c;

        // Apply damage: split stress into positive/negative parts
        // Simplified: damage_t acts on tensile components, damage_c on compressive
        for (int i = 0; i < 3; ++i) {
            if (trial[i] >= 0.0)
                state.stress[i] = (1.0 - d_t) * trial[i];
            else
                state.stress[i] = (1.0 - d_c) * trial[i];
        }
        Real d_shear = Kokkos::fmax(d_t, d_c);
        for (int i = 3; i < 6; ++i)
            state.stress[i] = (1.0 - d_shear) * trial[i];

        state.damage = Kokkos::fmax(d_t, d_c);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));

        Real d = state.damage;
        Real factor = (1.0 - d);

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = factor * (lambda + 2.0 * mu);
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = factor * lambda;
            }
        }
        C[21] = factor * mu;
        C[28] = factor * mu;
        C[35] = factor * mu;
    }

    KOKKOS_INLINE_FUNCTION Real get_fc() const { return fc_; }
    KOKKOS_INLINE_FUNCTION Real get_ft() const { return ft_; }

private:
    Real fc_;             // Compressive strength
    Real ft_;             // Tensile strength
    Real crush_strain_;   // Crushing strain
    Real eps_t0_;         // Tensile cracking strain
    Real eps_c0_;         // Compressive yield strain
    Real Gf_t_;           // Tensile fracture energy
    Real Gf_c_;           // Compressive fracture energy
};

// ============================================================================
// 3. SesameTabMaterial - Tabulated EOS (LAW26)
// ============================================================================

/**
 * @brief Tabulated EOS material with bilinear interpolation in (rho, e) space
 *
 * Pressure is looked up from a table P(rho, e). Returns hydrostatic stress only.
 * The table is stored as a flattened 2D array with rho and e grid points.
 *
 * History: [32]=current_density, [33]=internal_energy
 */
class SesameTabMaterial : public Material {
public:
    static constexpr int MAX_TAB = 16;

    SesameTabMaterial(const MaterialProperties& props,
                       Real rho0 = 1000.0)
        : Material(MaterialType::Custom, props)
        , rho0_(rho0), n_rho_(0), n_e_(0)
    {
        for (int i = 0; i < MAX_TAB; ++i) {
            rho_grid_[i] = 0.0;
            e_grid_[i] = 0.0;
        }
        for (int i = 0; i < MAX_TAB * MAX_TAB; ++i)
            p_table_[i] = 0.0;
    }

    /// Set up the density grid points
    void set_rho_grid(const Real* rho_pts, int n) {
        n_rho_ = (n < MAX_TAB) ? n : MAX_TAB;
        for (int i = 0; i < n_rho_; ++i) rho_grid_[i] = rho_pts[i];
    }

    /// Set up the energy grid points
    void set_e_grid(const Real* e_pts, int n) {
        n_e_ = (n < MAX_TAB) ? n : MAX_TAB;
        for (int i = 0; i < n_e_; ++i) e_grid_[i] = e_pts[i];
    }

    /// Set pressure table value at (i_rho, i_e)
    void set_pressure(int i_rho, int i_e, Real p_val) {
        if (i_rho < MAX_TAB && i_e < MAX_TAB)
            p_table_[i_rho * MAX_TAB + i_e] = p_val;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        // Compute current density from volumetric strain
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real rho_cur = rho0_ / (1.0 + ev + 1.0e-30);
        if (rho_cur < rho0_ * 0.01) rho_cur = rho0_ * 0.01;

        // Internal energy approximation from temperature
        Real e_cur = props_.specific_heat * state.temperature;
        if (e_cur < 0.0) e_cur = 0.0;

        state.history[32] = rho_cur;
        state.history[33] = e_cur;

        // Bilinear interpolation
        Real pressure = interpolate_pressure(rho_cur, e_cur);

        // Hydrostatic stress: sigma_ii = -P (compression positive in EOS)
        for (int i = 0; i < 3; ++i)
            state.stress[i] = -pressure;
        for (int i = 3; i < 6; ++i)
            state.stress[i] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        // Approximate bulk modulus from table
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real rho_cur = rho0_ / (1.0 + ev + 1.0e-30);
        Real e_cur = props_.specific_heat * state.temperature;
        if (e_cur < 0.0) e_cur = 0.0;

        Real drho = rho0_ * 0.001;
        Real p1 = interpolate_pressure(rho_cur + drho, e_cur);
        Real p0 = interpolate_pressure(rho_cur - drho, e_cur);
        Real K_eff = rho_cur * (p1 - p0) / (2.0 * drho + 1.0e-30);
        if (K_eff < 1.0) K_eff = 1.0;

        Real lambda = K_eff - 0.0; // No deviatoric for EOS-only
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = K_eff;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = K_eff - 2.0 * K_eff / 3.0;
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    Real interpolate_pressure(Real rho, Real e) const {
        if (n_rho_ < 2 || n_e_ < 2) {
            // Fallback: linear EOS
            Real mu_val = rho / rho0_ - 1.0;
            return props_.K * mu_val;
        }

        // Find rho bracket
        int ir = 0;
        for (int i = 0; i < n_rho_ - 1; ++i) {
            if (rho >= rho_grid_[i]) ir = i;
        }
        if (ir >= n_rho_ - 1) ir = n_rho_ - 2;

        // Find e bracket
        int ie = 0;
        for (int i = 0; i < n_e_ - 1; ++i) {
            if (e >= e_grid_[i]) ie = i;
        }
        if (ie >= n_e_ - 1) ie = n_e_ - 2;

        // Bilinear weights
        Real dr = rho_grid_[ir + 1] - rho_grid_[ir];
        Real de = e_grid_[ie + 1] - e_grid_[ie];
        Real tr = (rho - rho_grid_[ir]) / (dr + 1.0e-30);
        Real te = (e - e_grid_[ie]) / (de + 1.0e-30);
        tr = Kokkos::fmax(0.0, Kokkos::fmin(1.0, tr));
        te = Kokkos::fmax(0.0, Kokkos::fmin(1.0, te));

        Real p00 = p_table_[ir * MAX_TAB + ie];
        Real p10 = p_table_[(ir + 1) * MAX_TAB + ie];
        Real p01 = p_table_[ir * MAX_TAB + (ie + 1)];
        Real p11 = p_table_[(ir + 1) * MAX_TAB + (ie + 1)];

        return (1.0 - tr) * (1.0 - te) * p00 +
               tr * (1.0 - te) * p10 +
               (1.0 - tr) * te * p01 +
               tr * te * p11;
    }

private:
    Real rho0_;
    Real rho_grid_[MAX_TAB];
    Real e_grid_[MAX_TAB];
    Real p_table_[MAX_TAB * MAX_TAB];
    int n_rho_;
    int n_e_;
};

// ============================================================================
// 4. BiphasicMaterial - Solid skeleton + pore fluid (LAW37)
// ============================================================================

/**
 * @brief Biphasic material: solid skeleton + pore fluid (Terzaghi principle)
 *
 * Effective stress = total stress - pore pressure * identity
 * sigma'_ij = sigma_ij + p_f * delta_ij
 * Pore pressure evolves based on fluid bulk modulus and volumetric strain.
 *
 * History: [32]=pore_pressure, [33]=porosity_current
 */
class BiphasicMaterial : public Material {
public:
    BiphasicMaterial(const MaterialProperties& props,
                      Real E_skel = 10.0e9, Real nu_skel = 0.25,
                      Real K_fluid = 2.2e9, Real porosity = 0.3,
                      Real permeability = 1.0e-12)
        : Material(MaterialType::Custom, props)
        , E_skel_(E_skel), nu_skel_(nu_skel)
        , K_fluid_(K_fluid), porosity_(porosity)
        , permeability_(permeability)
    {
        G_skel_ = E_skel_ / (2.0 * (1.0 + nu_skel_));
        K_skel_ = E_skel_ / (3.0 * (1.0 - 2.0 * nu_skel_));
        // Biot coefficient (simplified): alpha_b ~ 1 - K_skel / K_grain
        // For simplicity assume grain modulus >> skeleton modulus
        alpha_b_ = 1.0;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real lambda = K_skel_ - 2.0 * G_skel_ / 3.0;

        // Volumetric strain
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];

        // Pore pressure update: p_f = K_fluid / porosity * (fluid volumetric strain)
        // Fluid volumetric strain ~ -alpha_b * ev (undrained)
        Real p_f = state.history[32];
        Real dp_f = -(K_fluid_ / (porosity_ + 1.0e-30)) * alpha_b_ * ev * state.dt;
        // For quasi-static, use incremental form; simplified:
        p_f = (K_fluid_ / (porosity_ + 1.0e-30)) * (-alpha_b_ * ev);
        state.history[32] = p_f;
        state.history[33] = porosity_ * (1.0 + ev); // Updated porosity

        // Effective stress (skeleton): sigma'_ij = lambda * ev * delta_ij + 2G * eps_ij
        Real sigma_eff[6];
        for (int i = 0; i < 3; ++i)
            sigma_eff[i] = lambda * ev + 2.0 * G_skel_ * state.strain[i];
        for (int i = 3; i < 6; ++i)
            sigma_eff[i] = G_skel_ * state.strain[i];

        // Total stress = effective stress - alpha_b * p_f * I
        // (tension positive convention: sigma = sigma' - p_f * delta)
        for (int i = 0; i < 3; ++i)
            state.stress[i] = sigma_eff[i] - alpha_b_ * p_f;
        for (int i = 3; i < 6; ++i)
            state.stress[i] = sigma_eff[i];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real lambda = K_skel_ - 2.0 * G_skel_ / 3.0;
        // Undrained tangent includes fluid contribution
        Real K_u = K_skel_ + alpha_b_ * alpha_b_ * K_fluid_ / (porosity_ + 1.0e-30);
        Real lambda_u = K_u - 2.0 * G_skel_ / 3.0;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda_u + 2.0 * G_skel_;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda_u;
            }
        }
        C[21] = G_skel_;
        C[28] = G_skel_;
        C[35] = G_skel_;
    }

    KOKKOS_INLINE_FUNCTION Real get_pore_pressure(const MaterialState& state) const {
        return state.history[32];
    }

private:
    Real E_skel_;
    Real nu_skel_;
    Real K_fluid_;
    Real porosity_;
    Real permeability_;
    Real G_skel_;
    Real K_skel_;
    Real alpha_b_;   // Biot coefficient
};

// ============================================================================
// 5. ViscousTabMaterial - Generalized Maxwell tabulated relaxation (LAW38)
// ============================================================================

/**
 * @brief Generalized Maxwell model with Prony series relaxation
 *
 * G(t) = g_inf + sum(g_i * exp(-t/tau_i)), i=1..nterms
 * Deviatoric stress via hereditary integral approximation.
 *
 * History: [32..39] = deviatoric viscous strains (2 per Maxwell arm, up to 4 arms)
 */
class ViscousTabMaterial : public Material {
public:
    static constexpr int MAX_ARMS = 4;

    ViscousTabMaterial(const MaterialProperties& props,
                        Real g_inf = 0.1, Real nterms = 4)
        : Material(MaterialType::Custom, props)
        , g_inf_(g_inf), nterms_(static_cast<int>(nterms))
    {
        if (nterms_ > MAX_ARMS) nterms_ = MAX_ARMS;
        if (nterms_ < 0) nterms_ = 0;
        for (int i = 0; i < MAX_ARMS; ++i) {
            g_i_[i] = 0.0;
            tau_i_[i] = 1.0;
        }
    }

    /// Set Prony series parameters for arm i
    void set_arm(int i, Real g, Real tau) {
        if (i >= 0 && i < MAX_ARMS) {
            g_i_[i] = g;
            tau_i_[i] = (tau > 1.0e-30) ? tau : 1.0e-30;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real G0 = E / (2.0 * (1.0 + nu));

        // Volumetric response (elastic)
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real pressure = K_bulk * ev;

        // Deviatoric strain
        Real e_dev[6];
        for (int i = 0; i < 3; ++i)
            e_dev[i] = state.strain[i] - ev / 3.0;
        for (int i = 3; i < 6; ++i)
            e_dev[i] = state.strain[i] / 2.0; // Engineering to tensor shear

        Real dt = state.dt;
        if (dt < 1.0e-30) dt = 1.0e-30;

        // Equilibrium (long-term) contribution
        Real G_eq = G0 * g_inf_;

        // Viscous overstress from each Maxwell arm
        Real s_visc[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int k = 0; k < nterms_; ++k) {
            Real G_k = G0 * g_i_[k];
            Real tau_k = tau_i_[k];
            Real exp_factor = Kokkos::exp(-dt / tau_k);

            for (int c = 0; c < 6; ++c) {
                // Recursive update of viscous internal variable
                int hist_idx = 32 + k * 6 + c;
                if (hist_idx < 64) {
                    Real h_old = state.history[hist_idx];
                    // Simo-Hughes recursive formula:
                    // h_new = exp(-dt/tau) * h_old + G_k * (1 - exp(-dt/tau)) * e_dev
                    Real h_new = exp_factor * h_old + G_k * (1.0 - exp_factor) * e_dev[c];
                    state.history[hist_idx] = h_new;
                    s_visc[c] += h_new;
                }
            }
        }

        // Total deviatoric stress: s = 2*G_eq*e_dev + 2*sum(h_k)
        for (int i = 0; i < 3; ++i)
            state.stress[i] = pressure / 3.0 + 2.0 * G_eq * e_dev[i] + 2.0 * s_visc[i];
        for (int i = 3; i < 6; ++i)
            state.stress[i] = 2.0 * G_eq * e_dev[i] + 2.0 * s_visc[i];

        // Fix: add hydrostatic part correctly
        for (int i = 0; i < 3; ++i)
            state.stress[i] = pressure + 2.0 * G_eq * e_dev[i] + 2.0 * s_visc[i];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real G0 = E / (2.0 * (1.0 + nu));

        // Instantaneous modulus (all arms fully active)
        Real G_inst = G0 * g_inf_;
        for (int k = 0; k < nterms_; ++k)
            G_inst += G0 * g_i_[k];

        Real lambda = K_bulk - 2.0 * G_inst / 3.0;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * G_inst;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda;
            }
        }
        C[21] = G_inst;
        C[28] = G_inst;
        C[35] = G_inst;
    }

    KOKKOS_INLINE_FUNCTION int get_nterms() const { return nterms_; }

private:
    Real g_inf_;
    Real g_i_[MAX_ARMS];
    Real tau_i_[MAX_ARMS];
    int nterms_;
};

// ============================================================================
// 6. KelvinMaxwellMaterial - Kelvin-Maxwell chain (LAW40)
// ============================================================================

/**
 * @brief Kelvin-Maxwell chain: Kelvin element (parallel E+eta) in series with Maxwell (series E+eta)
 *
 * The Kelvin element provides delayed elasticity; the Maxwell element provides viscous flow.
 * Total strain = Kelvin strain + Maxwell strain
 * Kelvin: sigma = E_kv * eps_kv + eta_kv * d(eps_kv)/dt
 * Maxwell: d(eps_mx)/dt = sigma / eta_mx - (E_mx / eta_mx) * eps_mx_elastic part
 *
 * Simplified 1D rheological model extended to 3D via deviatoric/volumetric split.
 *
 * History: [32]=Kelvin strain (scalar equiv), [33]=Maxwell dashpot strain (scalar equiv),
 *          [34..39]=Kelvin deviatoric strains, [40..45]=Maxwell deviatoric strains
 */
class KelvinMaxwellMaterial : public Material {
public:
    KelvinMaxwellMaterial(const MaterialProperties& props,
                           Real E_inf = 1.0e9, Real E_kv = 5.0e8,
                           Real eta_kv = 1.0e7, Real E_mx = 5.0e8,
                           Real eta_mx = 1.0e8)
        : Material(MaterialType::Custom, props)
        , E_inf_(E_inf), E_kv_(E_kv), eta_kv_(eta_kv)
        , E_mx_(E_mx), eta_mx_(eta_mx)
    {
        tau_kv_ = eta_kv_ / (E_kv_ + 1.0e-30);
        tau_mx_ = eta_mx_ / (E_mx_ + 1.0e-30);
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real nu = props_.nu;
        Real K_bulk = E_inf_ / (3.0 * (1.0 - 2.0 * nu));
        Real G_inf = E_inf_ / (2.0 * (1.0 + nu));
        Real G_kv = E_kv_ / (2.0 * (1.0 + nu));
        Real G_mx = E_mx_ / (2.0 * (1.0 + nu));
        Real eta_G_kv = eta_kv_ / (2.0 * (1.0 + nu));
        Real eta_G_mx = eta_mx_ / (2.0 * (1.0 + nu));

        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real pressure = K_bulk * ev;

        Real dt = state.dt;
        if (dt < 1.0e-30) dt = 1.0e-30;

        // Deviatoric total strain
        Real e_dev[6];
        for (int i = 0; i < 3; ++i)
            e_dev[i] = state.strain[i] - ev / 3.0;
        for (int i = 3; i < 6; ++i)
            e_dev[i] = state.strain[i] / 2.0;

        // Kelvin element: eps_kv evolves to match applied strain
        // sigma = G_kv * eps_kv + eta_G_kv * d(eps_kv)/dt
        // => d(eps_kv)/dt = (e_dev - eps_kv) / tau_kv  (equilibrium driven)
        Real exp_kv = Kokkos::exp(-dt / tau_kv_);
        for (int c = 0; c < 6; ++c) {
            int idx_kv = 34 + c;
            if (idx_kv < 64) {
                Real eps_kv_old = state.history[idx_kv];
                // Kelvin: exponential approach to total dev strain
                Real eps_kv_new = e_dev[c] - (e_dev[c] - eps_kv_old) * exp_kv;
                state.history[idx_kv] = eps_kv_new;
            }
        }

        // Maxwell element: viscous flow
        // d(eps_mx)/dt = s_dev / eta_G_mx
        // Approximate: eps_mx_new = exp(-dt/tau_mx)*eps_mx_old + (1-exp(-dt/tau_mx))*e_dev
        Real exp_mx = Kokkos::exp(-dt / tau_mx_);
        for (int c = 0; c < 6; ++c) {
            int idx_mx = 40 + c;
            if (idx_mx < 64) {
                Real eps_mx_old = state.history[idx_mx];
                Real eps_mx_new = exp_mx * eps_mx_old + (1.0 - exp_mx) * e_dev[c];
                state.history[idx_mx] = eps_mx_new;
            }
        }

        // Scalar histories for tracking
        Real eps_kv_scalar = 0.0, eps_mx_scalar = 0.0;
        for (int c = 0; c < 6; ++c) {
            int idx_kv = 34 + c;
            int idx_mx = 40 + c;
            if (idx_kv < 64) eps_kv_scalar += state.history[idx_kv] * state.history[idx_kv];
            if (idx_mx < 64) eps_mx_scalar += state.history[idx_mx] * state.history[idx_mx];
        }
        state.history[32] = Kokkos::sqrt(eps_kv_scalar + 1.0e-30);
        state.history[33] = Kokkos::sqrt(eps_mx_scalar + 1.0e-30);

        // Stress: spring in parallel (G_inf) + Kelvin contribution + Maxwell contribution
        // s_dev = 2*G_inf*e_dev + 2*G_kv*eps_kv + Kelvin dashpot + Maxwell spring residual
        // Simplified: total shear modulus at current state
        for (int i = 0; i < 3; ++i) {
            Real s_inf = 2.0 * G_inf * e_dev[i];
            Real s_kv = 2.0 * G_kv * ((34 + i < 64) ? state.history[34 + i] : 0.0);
            Real s_mx = 2.0 * G_mx * (e_dev[i] - ((40 + i < 64) ? state.history[40 + i] : 0.0));
            state.stress[i] = pressure + s_inf + s_kv + s_mx;
        }
        for (int i = 3; i < 6; ++i) {
            Real s_inf = 2.0 * G_inf * e_dev[i];
            Real s_kv = 2.0 * G_kv * ((34 + i < 64) ? state.history[34 + i] : 0.0);
            Real s_mx = 2.0 * G_mx * (e_dev[i] - ((40 + i < 64) ? state.history[40 + i] : 0.0));
            state.stress[i] = s_inf + s_kv + s_mx;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real nu = props_.nu;
        Real K_bulk = E_inf_ / (3.0 * (1.0 - 2.0 * nu));
        Real G_inst = E_inf_ / (2.0 * (1.0 + nu)) + E_kv_ / (2.0 * (1.0 + nu)) + E_mx_ / (2.0 * (1.0 + nu));
        Real lambda = K_bulk - 2.0 * G_inst / 3.0;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * G_inst;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda;
            }
        }
        C[21] = G_inst;
        C[28] = G_inst;
        C[35] = G_inst;
    }

    KOKKOS_INLINE_FUNCTION Real kelvin_strain(const MaterialState& state) const {
        return state.history[32];
    }
    KOKKOS_INLINE_FUNCTION Real maxwell_strain(const MaterialState& state) const {
        return state.history[33];
    }

private:
    Real E_inf_;
    Real E_kv_;
    Real eta_kv_;
    Real E_mx_;
    Real eta_mx_;
    Real tau_kv_;
    Real tau_mx_;
};

// ============================================================================
// 7. LeeTarverReactiveMaterial - Lee-Tarver ignition & growth (LAW41)
// ============================================================================

/**
 * @brief Lee-Tarver ignition and growth reactive burn model
 *
 * dF/dt = I*(1-F)^b*(rho/rho0 - 1 - a)^x [ignition]
 *       + G1*(1-F)^e*F^c*P^y             [growth term 1]
 *       + G2*(1-F)^g*F^d*P^z             [growth term 2]
 * JWL EOS for products, linear EOS for unreacted.
 *
 * History: [32]=burn_fraction, [33]=elapsed_time
 */
class LeeTarverReactiveMaterial : public Material {
public:
    LeeTarverReactiveMaterial(const MaterialProperties& props,
                               Real I_ig = 4.0e6, Real a_ig = 0.0,
                               Real b_ig = 0.667, Real x_ig = 7.0,
                               Real G1 = 140.0, Real c_g1 = 0.667,
                               Real d_g2 = 1.0, Real y_g1 = 2.0,
                               Real G2 = 0.0, Real e_g1 = 0.667,
                               Real g_g2 = 1.0, Real z_g2 = 3.0,
                               Real D_cj = 8000.0)
        : Material(MaterialType::Custom, props)
        , I_ig_(I_ig), a_ig_(a_ig), b_ig_(b_ig), x_ig_(x_ig)
        , G1_(G1), c_g1_(c_g1), d_g2_(d_g2), y_g1_(y_g1)
        , G2_(G2), e_g1_(e_g1), g_g2_(g_g2), z_g2_(z_g2)
        , D_cj_(D_cj)
    {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real rho0 = props_.density;
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real rho = rho0 / (1.0 + ev + 1.0e-30);
        Real mu = rho / rho0 - 1.0;
        Real dt = state.dt;
        if (dt < 1.0e-30) dt = 1.0e-30;

        Real F = state.history[32]; // Burn fraction
        Real t_elapsed = state.history[33];
        t_elapsed += dt;

        // Current pressure estimate (unreacted + partially reacted)
        Real P_unr = props_.K * mu; // Linear EOS for unreacted
        if (P_unr < 0.0) P_unr = 0.0;

        // JWL for products: P_jwl = A*exp(-R1*V) + B*exp(-R2*V) + omega*e/V
        // Simplified: P_jwl proportional to burn fraction and compression
        Real V = 1.0 / (1.0 + mu + 1.0e-30);
        Real P_jwl = rho0 * D_cj_ * D_cj_ * 0.25 * (1.0 - V);
        if (P_jwl < 0.0) P_jwl = 0.0;

        // Mixture pressure
        Real P_mix = (1.0 - F) * P_unr + F * P_jwl;

        // Lee-Tarver reaction rate
        Real dFdt = 0.0;

        // Ignition term: I*(1-F)^b * max(rho/rho0 - 1 - a, 0)^x
        if (F < 1.0) {
            Real comp_excess = mu - a_ig_;
            if (comp_excess > 0.0) {
                Real ign = I_ig_ * Kokkos::pow(1.0 - F + 1.0e-30, b_ig_)
                         * Kokkos::pow(comp_excess, x_ig_);
                dFdt += ign;
            }

            // Growth term 1: G1*(1-F)^e*F^c*P^y
            if (F > 1.0e-10 && P_mix > 0.0) {
                Real grow1 = G1_ * Kokkos::pow(1.0 - F + 1.0e-30, e_g1_)
                           * Kokkos::pow(F, c_g1_)
                           * Kokkos::pow(P_mix, y_g1_);
                dFdt += grow1;
            }

            // Growth term 2: G2*(1-F)^g*F^d*P^z
            if (F > 1.0e-10 && P_mix > 0.0 && G2_ > 0.0) {
                Real grow2 = G2_ * Kokkos::pow(1.0 - F + 1.0e-30, g_g2_)
                           * Kokkos::pow(F, d_g2_)
                           * Kokkos::pow(P_mix, z_g2_);
                dFdt += grow2;
            }
        }

        // Update burn fraction
        F += dFdt * dt;
        if (F > 1.0) F = 1.0;
        if (F < 0.0) F = 0.0;
        state.history[32] = F;
        state.history[33] = t_elapsed;

        // Final mixture pressure with updated F
        P_mix = (1.0 - F) * P_unr + F * P_jwl;

        // Hydrostatic stress (compression positive in EOS, tension positive in stress)
        for (int i = 0; i < 3; ++i)
            state.stress[i] = -P_mix;
        for (int i = 3; i < 6; ++i)
            state.stress[i] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real K_eff = props_.K;
        Real F = state.history[32];
        // Mix of unreacted and product bulk moduli
        Real K_prod = props_.density * D_cj_ * D_cj_ * 0.25;
        Real K_mix = (1.0 - F) * K_eff + F * K_prod;
        if (K_mix < 1.0) K_mix = 1.0;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = K_mix;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = K_mix - 2.0 * K_mix / 3.0;
            }
        }
    }

    KOKKOS_INLINE_FUNCTION Real burn_fraction(const MaterialState& state) const {
        return state.history[32];
    }
    KOKKOS_INLINE_FUNCTION Real elapsed_time(const MaterialState& state) const {
        return state.history[33];
    }

private:
    Real I_ig_, a_ig_, b_ig_, x_ig_;
    Real G1_, c_g1_, d_g2_, y_g1_;
    Real G2_, e_g1_, g_g2_, z_g2_;
    Real D_cj_;
};

// ============================================================================
// 8. FluffMaterial - Ultra-soft padding (LAW45)
// ============================================================================

/**
 * @brief Ultra-soft padding material with plateau stress and lockup
 *
 * Below lockup strain: constant plateau stress (like foam).
 * Above lockup: exponential stiffening.
 * Hysteretic unloading at reduced stiffness.
 *
 * History: [32]=max_vol_strain, [33]=unloading_flag
 */
class FluffMaterial : public Material {
public:
    FluffMaterial(const MaterialProperties& props,
                   Real E_plateau = 1.0e5, Real strain_lock = 0.7,
                   Real unload_factor = 0.1)
        : Material(MaterialType::Custom, props)
        , E_plateau_(E_plateau), strain_lock_(strain_lock)
        , unload_factor_(unload_factor)
    {
        // Stiffening modulus beyond lockup
        E_lock_ = E_plateau_ * 100.0;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real ev_comp = -ev; // Positive in compression
        if (ev_comp < 0.0) ev_comp = 0.0;

        Real max_ev = state.history[32];
        if (ev_comp > max_ev) {
            max_ev = ev_comp;
            state.history[32] = max_ev;
        }

        // Determine if loading or unloading
        Real is_unloading = (ev_comp < max_ev - 1.0e-10) ? 1.0 : 0.0;
        state.history[33] = is_unloading;

        Real sigma_vol = 0.0;
        if (ev_comp > 0.0) {
            if (ev_comp < strain_lock_) {
                // Plateau region
                sigma_vol = E_plateau_ * ev_comp;
            } else {
                // Lockup region: exponential stiffening
                Real excess = ev_comp - strain_lock_;
                Real denom = 1.0 - ev_comp;
                if (denom < 0.01) denom = 0.01;
                sigma_vol = E_plateau_ * strain_lock_ + E_lock_ * excess / denom;
            }

            // Hysteretic unloading
            if (is_unloading > 0.5) {
                // Unload at reduced stiffness from max point
                Real sigma_max = 0.0;
                if (max_ev < strain_lock_) {
                    sigma_max = E_plateau_ * max_ev;
                } else {
                    Real excess = max_ev - strain_lock_;
                    Real denom = 1.0 - max_ev;
                    if (denom < 0.01) denom = 0.01;
                    sigma_max = E_plateau_ * strain_lock_ + E_lock_ * excess / denom;
                }
                Real ratio = ev_comp / (max_ev + 1.0e-30);
                sigma_vol = sigma_max * ratio * (unload_factor_ + (1.0 - unload_factor_) * ratio);
            }
        }

        // Apply as compressive hydrostatic stress
        Real G = props_.G > 0.0 ? props_.G : E_plateau_ * 0.1;
        for (int i = 0; i < 3; ++i) {
            Real dev_strain = state.strain[i] - ev / 3.0;
            state.stress[i] = -sigma_vol / 3.0 + 2.0 * G * dev_strain;
        }
        for (int i = 3; i < 6; ++i)
            state.stress[i] = G * state.strain[i];

        // For compression, stress should be negative (compressive)
        // sigma_vol is magnitude, apply correctly
        for (int i = 0; i < 3; ++i)
            state.stress[i] = -sigma_vol + 2.0 * G * (state.strain[i] - ev / 3.0);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real ev = -(state.strain[0] + state.strain[1] + state.strain[2]);
        Real K_eff = E_plateau_;
        if (ev > strain_lock_) {
            Real denom = 1.0 - ev;
            if (denom < 0.01) denom = 0.01;
            K_eff = E_lock_ / (denom * denom);
        }
        Real G = props_.G > 0.0 ? props_.G : E_plateau_ * 0.1;
        Real lambda = K_eff - 2.0 * G / 3.0;
        if (lambda < 0.0) lambda = 0.0;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * G;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda;
            }
        }
        C[21] = G;
        C[28] = G;
        C[35] = G;
    }

    KOKKOS_INLINE_FUNCTION Real max_vol_strain(const MaterialState& state) const {
        return state.history[32];
    }

private:
    Real E_plateau_;
    Real strain_lock_;
    Real unload_factor_;
    Real E_lock_;
};

// ============================================================================
// 9. LESFluidMaterial - Large Eddy Simulation fluid (LAW46)
// ============================================================================

/**
 * @brief LES fluid material with Smagorinsky turbulence model
 *
 * Pressure: p = rho0 * c^2 * ev (artificial compressibility)
 * Deviatoric: tau_ij = 2*(mu_lam + mu_t)*S_ij
 * mu_t = rho*(Cs*delta)^2*|S|  (Smagorinsky)
 *
 * History: [32]=turbulent_viscosity, [33]=strain_rate_magnitude
 */
class LESFluidMaterial : public Material {
public:
    LESFluidMaterial(const MaterialProperties& props,
                      Real rho0 = 1000.0, Real c_sound = 1500.0,
                      Real mu_lam = 1.0e-3, Real C_s = 0.1)
        : Material(MaterialType::Custom, props)
        , rho0_(rho0), c_sound_(c_sound), mu_lam_(mu_lam), C_s_(C_s)
    {
        // Characteristic element size (user should set, default)
        delta_ = 0.01;
    }

    /// Set characteristic element size for LES filter
    void set_delta(Real delta) { delta_ = delta; }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];

        // Pressure from artificial compressibility
        Real pressure = rho0_ * c_sound_ * c_sound_ * ev;

        // Strain rate tensor (use strain_rate field if available, else approximate)
        Real S[6];
        for (int i = 0; i < 6; ++i) {
            S[i] = state.strain_rate[i];
        }

        // Strain rate magnitude: |S| = sqrt(2*S_ij*S_ij)
        Real S_mag = Kokkos::sqrt(2.0 * (S[0]*S[0] + S[1]*S[1] + S[2]*S[2] +
                                   2.0 * (S[3]*S[3] + S[4]*S[4] + S[5]*S[5])) + 1.0e-30);

        // Smagorinsky turbulent viscosity
        Real mu_t = rho0_ * (C_s_ * delta_) * (C_s_ * delta_) * S_mag;

        state.history[32] = mu_t;
        state.history[33] = S_mag;

        // Effective viscosity
        Real mu_eff = mu_lam_ + mu_t;

        // Deviatoric strain rate
        Real S_vol = (S[0] + S[1] + S[2]) / 3.0;

        // Stress: sigma_ij = -p*delta_ij + 2*mu_eff*S_dev_ij
        for (int i = 0; i < 3; ++i)
            state.stress[i] = -pressure + 2.0 * mu_eff * (S[i] - S_vol);
        for (int i = 3; i < 6; ++i)
            state.stress[i] = 2.0 * mu_eff * S[i];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real K_eff = rho0_ * c_sound_ * c_sound_;
        Real mu_eff = mu_lam_ + state.history[32];
        Real lambda = K_eff - 2.0 * mu_eff / 3.0;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * mu_eff;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda;
            }
        }
        C[21] = mu_eff;
        C[28] = mu_eff;
        C[35] = mu_eff;
    }

    KOKKOS_INLINE_FUNCTION Real get_turb_viscosity(const MaterialState& state) const {
        return state.history[32];
    }

private:
    Real rho0_;
    Real c_sound_;
    Real mu_lam_;
    Real C_s_;
    Real delta_;
};

// ============================================================================
// 10. MultiMaterialMaterial - Multi-material ALE (LAW51)
// ============================================================================

/**
 * @brief Multi-material ALE with pressure equilibrium among sub-materials
 *
 * Up to 4 sub-materials with volume fractions. Pressure equilibrium assumption:
 * P = sum(f_i * K_i * ev_i) with constraint sum(f_i * ev_i) = ev_total.
 * Deviatoric: mixture rule sigma_dev = sum(f_i * sigma_dev_i).
 *
 * History: [32..35]=volume fractions, [36..39]=sub-material pressures
 */
class MultiMaterialMaterial : public Material {
public:
    static constexpr int MAX_SUB = 4;

    MultiMaterialMaterial(const MaterialProperties& props, int num_sub = 2)
        : Material(MaterialType::Custom, props)
        , num_sub_(num_sub)
    {
        if (num_sub_ > MAX_SUB) num_sub_ = MAX_SUB;
        if (num_sub_ < 1) num_sub_ = 1;
        for (int i = 0; i < MAX_SUB; ++i) {
            K_sub_[i] = props.K;
            G_sub_[i] = props.G;
            f_vol_[i] = 0.0;
        }
        // Default: equal fractions
        for (int i = 0; i < num_sub_; ++i)
            f_vol_[i] = 1.0 / num_sub_;
    }

    /// Set sub-material properties
    void set_sub_material(int idx, Real K_val, Real G_val, Real f_vol) {
        if (idx >= 0 && idx < MAX_SUB) {
            K_sub_[idx] = K_val;
            G_sub_[idx] = G_val;
            f_vol_[idx] = f_vol;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];

        // Volume fractions from history or defaults
        Real f[MAX_SUB];
        Real f_sum = 0.0;
        for (int i = 0; i < num_sub_; ++i) {
            f[i] = (state.history[32 + i] > 1.0e-30) ? state.history[32 + i] : f_vol_[i];
            f_sum += f[i];
        }
        // Normalize
        if (f_sum > 1.0e-30) {
            for (int i = 0; i < num_sub_; ++i) f[i] /= f_sum;
        }

        // Pressure equilibrium: K_eff = sum(f_i * K_i)
        Real K_eff = 0.0, G_eff = 0.0;
        for (int i = 0; i < num_sub_; ++i) {
            K_eff += f[i] * K_sub_[i];
            G_eff += f[i] * G_sub_[i];
        }

        Real pressure = K_eff * ev;

        // Store sub-material pressures
        for (int i = 0; i < num_sub_; ++i) {
            state.history[36 + i] = K_sub_[i] * ev;
        }

        // Update volume fractions in history
        for (int i = 0; i < num_sub_; ++i)
            state.history[32 + i] = f[i];

        // Deviatoric stress with mixture shear
        Real lambda = K_eff - 2.0 * G_eff / 3.0;
        for (int i = 0; i < 3; ++i)
            state.stress[i] = lambda * ev + 2.0 * G_eff * state.strain[i];
        for (int i = 3; i < 6; ++i)
            state.stress[i] = G_eff * state.strain[i];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real f[MAX_SUB];
        Real f_sum = 0.0;
        for (int i = 0; i < num_sub_; ++i) {
            f[i] = (state.history[32 + i] > 1.0e-30) ? state.history[32 + i] : f_vol_[i];
            f_sum += f[i];
        }
        if (f_sum > 1.0e-30) {
            for (int i = 0; i < num_sub_; ++i) f[i] /= f_sum;
        }

        Real K_eff = 0.0, G_eff = 0.0;
        for (int i = 0; i < num_sub_; ++i) {
            K_eff += f[i] * K_sub_[i];
            G_eff += f[i] * G_sub_[i];
        }

        Real lambda = K_eff - 2.0 * G_eff / 3.0;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * G_eff;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda;
            }
        }
        C[21] = G_eff;
        C[28] = G_eff;
        C[35] = G_eff;
    }

    KOKKOS_INLINE_FUNCTION int get_num_sub() const { return num_sub_; }

private:
    int num_sub_;
    Real K_sub_[MAX_SUB];
    Real G_sub_[MAX_SUB];
    Real f_vol_[MAX_SUB];
};

// ============================================================================
// 11. PlasticTriangleMaterial - Triangle plastic with hourglass (LAW60)
// ============================================================================

/**
 * @brief J2 plasticity with Flanagan-Belytschko hourglass control
 *
 * Standard von Mises yield with linear hardening, plus
 * hourglass stabilization force F_hg = coeff * G * A * q_hg.
 *
 * History: [32]=equiv plastic strain, [33]=hourglass energy
 */
class PlasticTriangleMaterial : public Material {
public:
    PlasticTriangleMaterial(const MaterialProperties& props,
                              Real yield_stress = 250.0e6,
                              Real hardening = 1.0e9,
                              Real hourglass_coeff = 0.1)
        : Material(MaterialType::Custom, props)
        , sigma_y0_(yield_stress), H_(hardening), hg_coeff_(hourglass_coeff)
    {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real lambda = K_bulk - 2.0 * G / 3.0;

        // Elastic trial
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lambda * ev + 2.0 * G * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = G * state.strain[i];

        // Deviatoric trial
        Real p = (trial[0] + trial[1] + trial[2]) / 3.0;
        Real s[6];
        for (int i = 0; i < 3; ++i) s[i] = trial[i] - p;
        for (int i = 3; i < 6; ++i) s[i] = trial[i];

        // J2
        Real J2 = 0.5 * (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]) +
                   s[3]*s[3] + s[4]*s[4] + s[5]*s[5];
        Real sigma_eq = Kokkos::sqrt(3.0 * J2 + 1.0e-30);

        // Current yield stress
        Real eps_p = state.history[32];
        Real sigma_y = sigma_y0_ + H_ * eps_p;

        if (sigma_eq > sigma_y) {
            // Radial return
            Real d_gamma = (sigma_eq - sigma_y) / (3.0 * G + H_);
            Real scale = 1.0 - 3.0 * G * d_gamma / (sigma_eq + 1.0e-30);
            for (int i = 0; i < 3; ++i) s[i] *= scale;
            for (int i = 3; i < 6; ++i) s[i] *= scale;

            state.history[32] = eps_p + d_gamma;
            state.plastic_strain = state.history[32];
        }

        for (int i = 0; i < 3; ++i)
            state.stress[i] = s[i] + p;
        for (int i = 3; i < 6; ++i)
            state.stress[i] = s[i];

        // Hourglass energy tracking
        Real hg_energy = hg_coeff_ * G * ev * ev;
        state.history[33] = hg_energy;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * mu;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda;
            }
        }
        C[21] = mu;
        C[28] = mu;
        C[35] = mu;

        // Reduce for plastic state (consistent tangent approximation)
        if (state.plastic_strain > 0.0) {
            Real factor = 3.0 * mu / (3.0 * mu + H_);
            // Scale deviatoric part
            for (int i = 0; i < 36; ++i) C[i] *= (0.5 + 0.5 * factor);
        }
    }

    KOKKOS_INLINE_FUNCTION Real get_yield(const MaterialState& state) const {
        return sigma_y0_ + H_ * state.history[32];
    }

private:
    Real sigma_y0_;
    Real H_;
    Real hg_coeff_;
};

// ============================================================================
// 12. HanselHotFormMaterial - Hot forming with recrystallization (LAW63)
// ============================================================================

/**
 * @brief Hansel hot forming model with dynamic recrystallization
 *
 * Flow stress: sigma = A * exp(m1*T) * eps^m2 * exp(m3/eps) * eps_dot^(m4*T+m5) * (1+eps)^(m6*T)
 * Dynamic recrystallization: softening factor when accumulated strain exceeds critical.
 *
 * History: [32]=recrystallized_fraction, [33]=peak_stress, [34]=accumulated_strain
 */
class HanselHotFormMaterial : public Material {
public:
    HanselHotFormMaterial(const MaterialProperties& props,
                           Real A = 1000.0,
                           Real m1 = -0.003, Real m2 = 0.15,
                           Real m3 = -0.01, Real m4 = 0.0001,
                           Real m5 = 0.01, Real m6 = -0.001,
                           Real m7 = 0.0, Real m8 = 0.0, Real m9 = 0.0)
        : Material(MaterialType::Custom, props)
        , A_(A), m1_(m1), m2_(m2), m3_(m3)
        , m4_(m4), m5_(m5), m6_(m6)
        , m7_(m7), m8_(m8), m9_(m9)
    {
        // Critical strain for recrystallization
        eps_crit_ = 0.5;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real lambda = K_bulk - 2.0 * G / 3.0;

        // Elastic trial
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lambda * ev + 2.0 * G * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = G * state.strain[i];

        // Deviatoric trial
        Real p = (trial[0] + trial[1] + trial[2]) / 3.0;
        Real s[6];
        for (int i = 0; i < 3; ++i) s[i] = trial[i] - p;
        for (int i = 3; i < 6; ++i) s[i] = trial[i];

        Real J2 = 0.5 * (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]) +
                   s[3]*s[3] + s[4]*s[4] + s[5]*s[5];
        Real sigma_eq = Kokkos::sqrt(3.0 * J2 + 1.0e-30);

        // Temperature (Kelvin)
        Real T = state.temperature;
        if (T < 1.0) T = 293.15;

        // Equivalent plastic strain
        Real eps_p = state.history[34];
        if (eps_p < 1.0e-6) eps_p = 1.0e-6; // Clamp for pow safety

        // Strain rate
        Real eps_dot = state.effective_strain_rate;
        if (eps_dot < 1.0e-6) eps_dot = 1.0e-6;

        // Hansel flow stress
        Real sigma_flow = A_ * Kokkos::exp(m1_ * T)
                        * Kokkos::pow(eps_p, m2_)
                        * Kokkos::exp(m3_ / (eps_p + 1.0e-10))
                        * Kokkos::pow(eps_dot, m4_ * T + m5_)
                        * Kokkos::pow(1.0 + eps_p, m6_ * T);

        if (sigma_flow < 1.0e3) sigma_flow = 1.0e3; // Floor

        // Dynamic recrystallization softening
        Real X_rex = state.history[32];
        Real sigma_peak = state.history[33];
        if (sigma_eq > sigma_peak) sigma_peak = sigma_eq;
        state.history[33] = sigma_peak;

        if (eps_p > eps_crit_) {
            // Avrami-type recrystallization kinetics
            Real eps_excess = eps_p - eps_crit_;
            Real X_new = 1.0 - Kokkos::exp(-0.693 * Kokkos::pow(eps_excess / (0.1 + 1.0e-30), 2.0));
            if (X_new > X_rex) X_rex = X_new;
            if (X_rex > 1.0) X_rex = 1.0;
            state.history[32] = X_rex;

            // Softening: reduce flow stress
            Real sigma_ss = sigma_flow * 0.6; // Steady-state stress
            sigma_flow = sigma_flow - X_rex * (sigma_flow - sigma_ss);
        }

        // Radial return
        if (sigma_eq > sigma_flow) {
            Real d_gamma = (sigma_eq - sigma_flow) / (3.0 * G + 1.0e-30);
            Real scale = sigma_flow / (sigma_eq + 1.0e-30);
            for (int i = 0; i < 3; ++i) s[i] *= scale;
            for (int i = 3; i < 6; ++i) s[i] *= scale;

            state.history[34] = eps_p + d_gamma;
            state.plastic_strain = state.history[34];
        }

        for (int i = 0; i < 3; ++i)
            state.stress[i] = s[i] + p;
        for (int i = 3; i < 6; ++i)
            state.stress[i] = s[i];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * mu;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda;
            }
        }
        C[21] = mu;
        C[28] = mu;
        C[35] = mu;

        if (state.plastic_strain > 0.0) {
            Real factor = 0.7; // Approximate consistent tangent reduction
            for (int i = 0; i < 36; ++i) C[i] *= factor;
        }
    }

    KOKKOS_INLINE_FUNCTION Real recrystallized_fraction(const MaterialState& state) const {
        return state.history[32];
    }

private:
    Real A_, m1_, m2_, m3_, m4_, m5_, m6_, m7_, m8_, m9_;
    Real eps_crit_;
};

// ============================================================================
// 13. UgineALZMaterial - Stainless steel forming (LAW64)
// ============================================================================

/**
 * @brief Ugine ALZ stainless steel forming model
 *
 * Flow stress: sigma_y = K * (eps_p + eps0)^n * (1 + m*ln(eps_dot/eps_dot_ref)) * temp_factor
 * Temperature factor: (T_ref/T)^alpha_T or exponential softening.
 *
 * History: [32]=equiv plastic strain, [33]=temperature_history
 */
class UgineALZMaterial : public Material {
public:
    UgineALZMaterial(const MaterialProperties& props,
                      Real K_str = 1500.0e6, Real n = 0.5,
                      Real eps0 = 0.01, Real m = 0.02,
                      Real T_ref = 293.15)
        : Material(MaterialType::Custom, props)
        , K_str_(K_str), n_(n), eps0_(eps0), m_(m), T_ref_(T_ref)
    {
        eps_dot_ref_ = 1.0;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real lambda = K_bulk - 2.0 * G / 3.0;

        // Elastic trial
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lambda * ev + 2.0 * G * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = G * state.strain[i];

        Real p_trial = (trial[0] + trial[1] + trial[2]) / 3.0;
        Real s[6];
        for (int i = 0; i < 3; ++i) s[i] = trial[i] - p_trial;
        for (int i = 3; i < 6; ++i) s[i] = trial[i];

        Real J2 = 0.5 * (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]) +
                   s[3]*s[3] + s[4]*s[4] + s[5]*s[5];
        Real sigma_eq = Kokkos::sqrt(3.0 * J2 + 1.0e-30);

        // Equivalent plastic strain
        Real eps_p = state.history[32];

        // Strain rate effect
        Real eps_dot = state.effective_strain_rate;
        if (eps_dot < eps_dot_ref_) eps_dot = eps_dot_ref_;
        Real rate_factor = 1.0 + m_ * Kokkos::log(eps_dot / eps_dot_ref_);
        if (rate_factor < 0.1) rate_factor = 0.1;

        // Temperature effect
        Real T = state.temperature;
        Real temp_factor = 1.0;
        if (T > T_ref_) {
            temp_factor = Kokkos::exp(-0.001 * (T - T_ref_));
            if (temp_factor < 0.01) temp_factor = 0.01;
        }

        // Flow stress
        Real eps_eff = eps_p + eps0_;
        if (eps_eff < 1.0e-10) eps_eff = 1.0e-10;
        Real sigma_y = K_str_ * Kokkos::pow(eps_eff, n_) * rate_factor * temp_factor;

        if (sigma_eq > sigma_y) {
            // Radial return
            // Hardening slope: H = d(sigma_y)/d(eps_p) = K * n * (eps_p+eps0)^(n-1) * rate * temp
            Real H = K_str_ * n_ * Kokkos::pow(eps_eff, n_ - 1.0) * rate_factor * temp_factor;
            if (H < 0.0) H = 0.0;

            Real d_gamma = (sigma_eq - sigma_y) / (3.0 * G + H);
            Real scale = 1.0 - 3.0 * G * d_gamma / (sigma_eq + 1.0e-30);
            if (scale < 0.0) scale = 0.0;
            for (int i = 0; i < 3; ++i) s[i] *= scale;
            for (int i = 3; i < 6; ++i) s[i] *= scale;

            state.history[32] = eps_p + d_gamma;
            state.plastic_strain = state.history[32];
        }

        state.history[33] = T;

        for (int i = 0; i < 3; ++i)
            state.stress[i] = s[i] + p_trial;
        for (int i = 3; i < 6; ++i)
            state.stress[i] = s[i];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * mu;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda;
            }
        }
        C[21] = mu;
        C[28] = mu;
        C[35] = mu;

        if (state.plastic_strain > 0.0) {
            Real eps_eff = state.plastic_strain + eps0_;
            Real H = K_str_ * n_ * Kokkos::pow(eps_eff, n_ - 1.0);
            Real factor = (3.0 * mu + H) > 0.0 ? H / (3.0 * mu + H) : 0.0;
            // Consistent tangent: elastic - plastic correction (simplified)
            Real red = 0.5 + 0.5 * factor;
            for (int i = 0; i < 36; ++i) C[i] *= red;
        }
    }

private:
    Real K_str_;
    Real n_;
    Real eps0_;
    Real m_;
    Real T_ref_;
    Real eps_dot_ref_;
};

// ============================================================================
// 14. CosseratMaterial - Micropolar continuum (LAW68)
// ============================================================================

/**
 * @brief Cosserat micropolar continuum with couple stresses
 *
 * Asymmetric stress tensor with additional rotational DOF.
 * Force stress: sigma_ij = lambda*e_kk*delta_ij + (mu+mu_c)*e_ij + (mu-mu_c)*e_ji
 * Couple stress: m_ij = alpha*kappa_kk*delta_ij + beta*kappa_ij + gamma*kappa_ji
 * where gamma = mu*l_c^2, alpha = beta = gamma*(nu/(1-2*nu))
 *
 * Symmetric part stored in stress[0..5], asymmetric + couple in history.
 *
 * History: [32..34]=couple stress m_11,m_22,m_33, [35..37]=couple stress m_12,m_23,m_13,
 *          [38]=asymmetric_energy
 */
class CosseratMaterial : public Material {
public:
    CosseratMaterial(const MaterialProperties& props,
                      Real E_val = 10.0e9, Real nu_val = 0.25,
                      Real l_c = 0.001, Real N_couple = 0.5,
                      Real mu_c = 0.0)
        : Material(MaterialType::Custom, props)
        , E_(E_val), nu_(nu_val), l_c_(l_c), N_(N_couple)
    {
        mu_ = E_ / (2.0 * (1.0 + nu_));
        lambda_ = E_ * nu_ / ((1.0 + nu_) * (1.0 - 2.0 * nu_));

        // Cosserat coupling modulus
        // N = sqrt(mu_c / (mu + mu_c)) => mu_c = mu * N^2 / (1 - N^2)
        if (mu_c > 0.0) {
            mu_c_ = mu_c;
        } else {
            Real N2 = N_ * N_;
            if (N2 >= 1.0) N2 = 0.99;
            mu_c_ = mu_ * N2 / (1.0 - N2);
        }

        // Couple stress moduli
        gamma_ = mu_ * l_c_ * l_c_;
        beta_ = gamma_;
        alpha_c_ = gamma_ * nu_ / (1.0 - 2.0 * nu_ + 1.0e-30);
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];

        // Symmetric force stress (standard part)
        for (int i = 0; i < 3; ++i)
            state.stress[i] = lambda_ * ev + 2.0 * mu_ * state.strain[i];
        for (int i = 3; i < 6; ++i)
            state.stress[i] = mu_ * state.strain[i];

        // Cosserat asymmetric contribution: adds mu_c term
        // For symmetric loading, the additional mu_c scales the shear
        for (int i = 3; i < 6; ++i)
            state.stress[i] += mu_c_ * state.strain[i];

        // Couple stress: m_ij = (alpha_c + beta + gamma) * kappa_ij  (simplified symmetric)
        // Curvature tensor kappa is rotation gradient; approximate from strain gradient
        // In FEM this comes from micro-rotation DOFs; here store the couple stress stiffness
        Real kappa_scale = l_c_ * l_c_;
        Real G_couple = alpha_c_ + beta_ + gamma_;

        // Store couple stresses (proportional to curvature; here approximate as zero
        // unless curvature is provided via history variables)
        for (int i = 0; i < 6; ++i) {
            if (32 + i < 64) {
                // Couple stress = G_couple * curvature (stored in history as input or zero)
                Real curvature_i = 0.0; // Would come from element formulation
                state.history[32 + i] = G_couple * curvature_i;
            }
        }
        state.history[38] = kappa_scale * G_couple; // Store couple stiffness for reference
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda_ + 2.0 * mu_;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda_;
            }
        }
        // Shear includes Cosserat coupling
        C[21] = mu_ + mu_c_;
        C[28] = mu_ + mu_c_;
        C[35] = mu_ + mu_c_;
    }

    KOKKOS_INLINE_FUNCTION Real get_mu_c() const { return mu_c_; }
    KOKKOS_INLINE_FUNCTION Real get_lc() const { return l_c_; }
    KOKKOS_INLINE_FUNCTION Real couple_stiffness() const {
        return alpha_c_ + beta_ + gamma_;
    }

private:
    Real E_, nu_;
    Real l_c_;         // Characteristic length
    Real N_;           // Coupling number
    Real mu_;          // Shear modulus
    Real lambda_;      // Lame parameter
    Real mu_c_;        // Cosserat coupling modulus
    Real gamma_;       // Couple stress modulus
    Real beta_;        // Couple stress modulus
    Real alpha_c_;     // Couple stress modulus
};

// ============================================================================
// 15. YuModelMaterial - Yu unified strength for geomaterials (LAW78)
// ============================================================================

/**
 * @brief Yu unified strength theory for geomaterials
 *
 * Yu criterion: (sigma1-sigma3)/(1+b) + b*(sigma2-sigma3)/(1+b)
 *   = 2*c*cos(phi)/(1-sin(phi)) + (sigma1+b*sigma2+sigma3)*sin(phi)/(1+b)
 *
 * The parameter b (0 <= b <= 1) controls the effect of intermediate principal stress:
 * - b=0 reduces to Mohr-Coulomb
 * - b=1 reduces to twin-shear theory
 *
 * Return mapping in principal stress space with spectral decomposition.
 *
 * History: [32]=equiv plastic strain, [33]=mean stress, [34]=yield function value
 */
class YuModelMaterial : public Material {
public:
    YuModelMaterial(const MaterialProperties& props,
                     Real c = 1.0e6, Real phi = 30.0,
                     Real b = 0.5)
        : Material(MaterialType::Custom, props)
        , c_(c), b_(b)
    {
        Real phi_rad = phi * 3.14159265358979323846 / 180.0;
        sin_phi_ = Kokkos::sin(phi_rad);
        cos_phi_ = Kokkos::cos(phi_rad);
        // Yield constant: k_yu = 2*c*cos(phi)/(1-sin(phi))
        k_yu_ = 2.0 * c_ * cos_phi_ / (1.0 - sin_phi_ + 1.0e-30);
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real lambda = K_bulk - 2.0 * G / 3.0;

        // Elastic trial
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lambda * ev + 2.0 * G * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = G * state.strain[i];

        // Approximate principal stresses from normal components
        // (Full eigenvalue decomposition is expensive; use simplified approach)
        // Sort trial normal stresses as principal stress approximation
        Real sigma_n[3] = {trial[0], trial[1], trial[2]};

        // Simple bubble sort for 3 values (descending: sigma1 >= sigma2 >= sigma3)
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < 2 - pass; ++i) {
                if (sigma_n[i] < sigma_n[i + 1]) {
                    Real tmp = sigma_n[i];
                    sigma_n[i] = sigma_n[i + 1];
                    sigma_n[i + 1] = tmp;
                }
            }
        }

        Real s1 = sigma_n[0]; // Max principal
        Real s2 = sigma_n[1]; // Mid principal
        Real s3 = sigma_n[2]; // Min principal

        // Yu criterion: F = (s1-s3)/(1+b) + b*(s2-s3)/(1+b) - k_yu - sin(phi)*(s1+b*s2+s3)/(1+b)
        // Rewritten: F = (s1 + b*s2 - (1+b)*s3)/(1+b) - k_yu - sin(phi)*(s1+b*s2+s3)/(1+b)
        // = [(s1+b*s2-(1+b)*s3) - sin(phi)*(s1+b*s2+s3)] / (1+b) - k_yu
        // = [(1-sin_phi)*s1 + b*(1-sin_phi)*s2 - (1+b+sin_phi+b*sin_phi)*s3/? ]...

        // Direct form:
        Real bp1 = 1.0 + b_;
        Real lhs = (s1 - s3) / bp1 + b_ * (s2 - s3) / bp1;
        Real rhs = k_yu_ + sin_phi_ * (s1 + b_ * s2 + s3) / bp1;

        Real F_yield = lhs - rhs;
        state.history[34] = F_yield;
        state.history[33] = (s1 + s2 + s3) / 3.0;

        if (F_yield > 0.0) {
            // Return mapping: project back to yield surface
            // Use associated flow rule in principal space
            Real eps_p = state.history[32];

            // Flow direction in principal stress space (simplified)
            // dg/ds1 = (1-sin_phi)/(1+b), dg/ds2 = b*(1-sin_phi)/(1+b), dg/ds3 = -(1+b*sin_phi+sin_phi)/(1+b)
            Real n1 = (1.0 - sin_phi_) / bp1;
            Real n2 = b_ * (1.0 - sin_phi_) / bp1;
            Real n3 = -(1.0 + sin_phi_) / bp1;

            // Plastic multiplier
            Real A_coeff = n1 * n1 + n2 * n2 + n3 * n3;
            Real d_lambda = F_yield / (2.0 * G * A_coeff + K_bulk * (n1 + n2 + n3) * (n1 + n2 + n3) / 3.0 + 1.0e-30);

            // Correct principal stresses
            Real ds[3] = {
                2.0 * G * n1 * d_lambda + K_bulk * (n1 + n2 + n3) * d_lambda / 3.0,
                2.0 * G * n2 * d_lambda + K_bulk * (n1 + n2 + n3) * d_lambda / 3.0,
                2.0 * G * n3 * d_lambda + K_bulk * (n1 + n2 + n3) * d_lambda / 3.0
            };

            sigma_n[0] -= ds[0];
            sigma_n[1] -= ds[1];
            sigma_n[2] -= ds[2];

            // Equivalent plastic strain increment
            Real d_eps_p = d_lambda * Kokkos::sqrt(2.0 * (n1*n1 + n2*n2 + n3*n3) / 3.0 + 1.0e-30);
            state.history[32] = eps_p + d_eps_p;
            state.plastic_strain = state.history[32];
        }

        // Map corrected principal stresses back to original directions
        // (Simplified: apply proportional correction to original trial stresses)
        Real trial_max = Kokkos::fmax(Kokkos::fabs(s1), Kokkos::fmax(Kokkos::fabs(s2), Kokkos::fabs(s3)));
        if (trial_max > 1.0e-30 && F_yield > 0.0) {
            Real scale1 = sigma_n[0] / (s1 + 1.0e-30);
            Real scale2 = sigma_n[1] / (s2 + 1.0e-30);
            Real scale3 = sigma_n[2] / (s3 + 1.0e-30);
            Real avg_scale = (Kokkos::fabs(scale1) + Kokkos::fabs(scale2) + Kokkos::fabs(scale3)) / 3.0;
            if (avg_scale < 0.0) avg_scale = 0.0;
            if (avg_scale > 2.0) avg_scale = 2.0;
            for (int i = 0; i < 6; ++i)
                state.stress[i] = trial[i] * avg_scale;
        } else {
            for (int i = 0; i < 6; ++i)
                state.stress[i] = trial[i];
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * mu;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda;
            }
        }
        C[21] = mu;
        C[28] = mu;
        C[35] = mu;

        if (state.plastic_strain > 0.0) {
            Real red = 0.75;
            for (int i = 0; i < 36; ++i) C[i] *= red;
        }
    }

    KOKKOS_INLINE_FUNCTION Real get_k_yu() const { return k_yu_; }
    KOKKOS_INLINE_FUNCTION Real get_b() const { return b_; }

private:
    Real c_;           // Cohesion
    Real b_;           // Intermediate stress parameter
    Real sin_phi_;
    Real cos_phi_;
    Real k_yu_;        // Yield constant
};

} // namespace physics
} // namespace nxs
