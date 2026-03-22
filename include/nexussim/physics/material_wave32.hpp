#pragma once

/**
 * @file material_wave32.hpp
 * @brief Wave 32 material models: 20 constitutive models (15 main + 5 thin wrappers)
 *
 * Models included:
 *   1.  OrthotropicHillMaterial        - Orthotropic Hill yield with CONVERSE algorithm (LAW93)
 *   2.  VegterYieldMaterial            - Multi-point yield surface with Bezier interpolation (LAW110)
 *   3.  MarlowHyperelasticMaterial     - Test-data-driven hyperelastic (LAW111)
 *   4.  DeshpandeFleckMaterial         - Metal foam yield surface (LAW115)
 *   5.  ModifiedLaDevezeMaterial       - Modified composite mesomodel (LAW122)
 *   6.  CDPM2ConcreteMaterial          - Two-surface concrete damage-plasticity (LAW124)
 *   7.  JHConcreteMaterial             - Johnson-Holmquist concrete (LAW126)
 *   8.  EnhancedCompositeMaterial      - Fiber/matrix decomposition (LAW127)
 *   9.  GranularMaterial               - Granular with rolling resistance (LAW133)
 *  10.  ViscousFoamMaterial            - Rate-dependent foam (LAW134)
 *  11.  FabricNLMaterial               - Nonlinear fabric with locking (LAW158)
 *  12.  ARUPAdhesiveMaterial           - Mixed-mode adhesive (LAW169)
 *  13.  FoamDuboisMaterial             - Dubois foam with pressure coupling (LAW190)
 *  14.  HenselSpittelMaterial          - Hot rolling flow stress (LAW103)
 *  15.  PaperLightMaterial             - Paper/lightweight board (LAW107)
 *  16.  JWLBMaterial                   - JWL + afterburn (LAW97)
 *  17.  PPPolymerMaterial              - Polypropylene tabulated yield + creep (LAW101)
 *  18.  DruckerPrager3Material         - DP variant 3 with tension cutoff (LAW102)
 *  19.  JCookAluminumMaterial          - JC aluminum with thermal softening (LAW106)
 *  20.  SpringGeneralizedMaterial      - 6-DOF spring (LAW108)
 */

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

// ============================================================================
// 1. OrthotropicHillMaterial - Orthotropic Hill Yield (LAW93)
// ============================================================================

/**
 * @brief Orthotropic Hill yield criterion with CONVERSE return mapping
 *
 * Hill yield:
 *   F(sig22-sig33)^2 + G(sig33-sig11)^2 + H(sig11-sig22)^2
 *   + 2L*tau23^2 + 2M*tau13^2 + 2N*tau12^2 = sigma_y^2
 *
 * Return mapping via CONVERSE (consistent tangent, radial return to
 * Hill surface). Isotropic hardening: sigma_y = sigma_y0 + H_mod * eps_p.
 *
 * History: [32]=equivalent plastic strain, [33]=yield stress evolution
 */
class OrthotropicHillMaterial : public Material {
public:
    OrthotropicHillMaterial(const MaterialProperties& props,
                             Real F_hill = 0.5, Real G_hill = 0.5, Real H_hill = 0.5,
                             Real L_hill = 1.5, Real M_hill = 1.5, Real N_hill = 1.5,
                             Real yield_stress = 250.0e6, Real hardening_mod = 1.0e9)
        : Material(MaterialType::Custom, props)
        , F_(F_hill), G_(G_hill), H_(H_hill)
        , L_(L_hill), M_(M_hill), N_(N_hill)
        , sigma_y0_(yield_stress), H_mod_(hardening_mod)
    {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G_shear = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        // Elastic trial stress
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lambda * ev + 2.0 * G_shear * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = G_shear * state.strain[i];

        // Retrieve history
        Real eps_p = state.history[32];
        Real sigma_y = sigma_y0_ + H_mod_ * eps_p;

        // Hill equivalent stress
        Real s11 = trial[0], s22 = trial[1], s33 = trial[2];
        Real t12 = trial[3], t23 = trial[4], t13 = trial[5];

        Real hill_sq = F_ * (s22 - s33) * (s22 - s33)
                     + G_ * (s33 - s11) * (s33 - s11)
                     + H_ * (s11 - s22) * (s11 - s22)
                     + 2.0 * L_ * t23 * t23
                     + 2.0 * M_ * t13 * t13
                     + 2.0 * N_ * t12 * t12;

        Real sigma_hill = Kokkos::sqrt(Kokkos::fmax(hill_sq, 1.0e-30));

        if (sigma_hill > sigma_y) {
            // Radial return mapping (CONVERSE algorithm)
            // Newton iteration for plastic multiplier
            Real dgamma = 0.0;
            for (int iter = 0; iter < 20; ++iter) {
                Real sigma_y_cur = sigma_y0_ + H_mod_ * (eps_p + dgamma);
                Real f = sigma_hill - 3.0 * G_shear * dgamma - sigma_y_cur;
                Real df = -3.0 * G_shear - H_mod_;
                Real ddg = -f / df;
                dgamma += ddg;
                if (dgamma < 0.0) dgamma = 0.0;
                if (Kokkos::fabs(f) < 1.0e-10 * sigma_y0_) break;
            }

            // Scale back stresses
            Real scale = (sigma_y0_ + H_mod_ * (eps_p + dgamma)) / (sigma_hill + 1.0e-30);
            if (scale > 1.0) scale = 1.0;

            // Apply radial return to deviatoric part
            Real p_trial = (trial[0] + trial[1] + trial[2]) / 3.0;
            for (int i = 0; i < 3; ++i)
                state.stress[i] = p_trial + scale * (trial[i] - p_trial);
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale * trial[i];

            eps_p += dgamma;
            state.plastic_strain = eps_p;
        } else {
            for (int i = 0; i < 6; ++i)
                state.stress[i] = trial[i];
        }

        state.history[32] = eps_p;
        state.history[33] = sigma_y0_ + H_mod_ * eps_p;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G_shear = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real eps_p = state.history[32];
        Real sigma_y = sigma_y0_ + H_mod_ * eps_p;

        // Compute Hill equivalent from current stress
        Real s11 = state.stress[0], s22 = state.stress[1], s33 = state.stress[2];
        Real hill_sq = F_ * (s22 - s33) * (s22 - s33)
                     + G_ * (s33 - s11) * (s33 - s11)
                     + H_ * (s11 - s22) * (s11 - s22)
                     + 2.0 * L_ * state.stress[4] * state.stress[4]
                     + 2.0 * M_ * state.stress[5] * state.stress[5]
                     + 2.0 * N_ * state.stress[3] * state.stress[3];
        Real sigma_hill = Kokkos::sqrt(Kokkos::fmax(hill_sq, 1.0e-30));

        // Elastic tangent (default or if not yielded)
        Real factor = (sigma_hill > sigma_y && sigma_hill > 1.0e-20)
                    ? (3.0 * G_shear) / (3.0 * G_shear + H_mod_) : 1.0;

        Real Geff = G_shear * factor;
        Real lambda_eff = lambda;

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda_eff + 2.0 * Geff;
            for (int j = 0; j < 3; ++j) {
                if (j != i) C[i * 6 + j] = lambda_eff;
            }
        }
        C[21] = Geff;
        C[28] = Geff;
        C[35] = Geff;
    }

    /// Get Hill equivalent stress from a state
    KOKKOS_INLINE_FUNCTION
    Real hill_equivalent(const MaterialState& state) const {
        Real s11 = state.stress[0], s22 = state.stress[1], s33 = state.stress[2];
        Real hill_sq = F_ * (s22 - s33) * (s22 - s33)
                     + G_ * (s33 - s11) * (s33 - s11)
                     + H_ * (s11 - s22) * (s11 - s22)
                     + 2.0 * L_ * state.stress[4] * state.stress[4]
                     + 2.0 * M_ * state.stress[5] * state.stress[5]
                     + 2.0 * N_ * state.stress[3] * state.stress[3];
        return Kokkos::sqrt(Kokkos::fmax(hill_sq, 0.0));
    }

    /// Current yield stress
    KOKKOS_INLINE_FUNCTION
    Real current_yield(const MaterialState& state) const {
        return sigma_y0_ + H_mod_ * state.history[32];
    }

private:
    Real F_, G_, H_, L_, M_, N_;
    Real sigma_y0_, H_mod_;
};

// ============================================================================
// 2. VegterYieldMaterial - Multi-Point Yield Surface (LAW110)
// ============================================================================

/**
 * @brief Multi-point yield surface with Bezier interpolation
 *
 * Yield locus defined by reference points at angles theta around the
 * stress space. Bezier curves interpolate between reference points.
 * Plane-stress return mapping. Isotropic hardening shifts the locus.
 *
 * History: [32]=equivalent plastic strain, [33]=yield locus scale
 */
class VegterYieldMaterial : public Material {
public:
    struct YieldPoint {
        Real theta;   ///< Angle in stress space (radians)
        Real sigma;   ///< Yield stress at this angle
    };

    VegterYieldMaterial(const MaterialProperties& props,
                         const YieldPoint* yield_points = nullptr, int npoints = 0,
                         Real hardening_mod = 1.0e9)
        : Material(MaterialType::Custom, props)
        , npts_(npoints > 8 ? 8 : npoints)
        , H_mod_(hardening_mod)
    {
        Real sigma_y0 = props.yield_stress;
        for (int i = 0; i < 8; ++i) {
            if (i < npts_ && yield_points) {
                pts_[i].theta = yield_points[i].theta;
                pts_[i].sigma = yield_points[i].sigma;
            } else {
                // Default: circular yield surface
                pts_[i].theta = i * 3.14159265358979323846 / 4.0;
                pts_[i].sigma = sigma_y0;
            }
        }
        if (npts_ == 0) npts_ = 8;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G_shear = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        // Trial elastic stress
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lambda * ev + 2.0 * G_shear * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = G_shear * state.strain[i];

        // Plane-stress deviatoric: compute angle in (sig11, sig22) space
        Real s11_dev = trial[0] - (trial[0] + trial[1]) / 2.0;
        Real s22_dev = trial[1] - (trial[0] + trial[1]) / 2.0;
        Real tau = trial[3];

        // Angle in stress space
        Real r_stress = Kokkos::sqrt(s11_dev * s11_dev + s22_dev * s22_dev + tau * tau);
        Real theta = 0.0;
        if (r_stress > 1.0e-30)
            theta = Kokkos::atan2(s22_dev, s11_dev + 1.0e-30);
        if (theta < 0.0) theta += 2.0 * 3.14159265358979323846;

        // Interpolate yield stress at this angle via Bezier
        Real sigma_y_theta = interpolate_yield(theta);

        // Hardening
        Real eps_p = state.history[32];
        Real sigma_y = sigma_y_theta + H_mod_ * eps_p;

        // Von Mises equivalent for return mapping
        Real p_trial = (trial[0] + trial[1] + trial[2]) / 3.0;
        Real dev[6];
        for (int i = 0; i < 3; ++i) dev[i] = trial[i] - p_trial;
        for (int i = 3; i < 6; ++i) dev[i] = trial[i];
        Real J2 = 0.0;
        for (int i = 0; i < 3; ++i) J2 += dev[i] * dev[i];
        for (int i = 3; i < 6; ++i) J2 += 2.0 * dev[i] * dev[i];
        J2 *= 0.5;
        Real vm = Kokkos::sqrt(3.0 * J2);

        if (vm > sigma_y) {
            // Radial return
            Real dgamma = 0.0;
            for (int iter = 0; iter < 20; ++iter) {
                Real sy = sigma_y_theta + H_mod_ * (eps_p + dgamma);
                Real f = vm - 3.0 * G_shear * dgamma - sy;
                Real df = -3.0 * G_shear - H_mod_;
                dgamma -= f / df;
                if (dgamma < 0.0) dgamma = 0.0;
                if (Kokkos::fabs(f) < 1.0e-10 * sigma_y_theta) break;
            }

            Real scale = 1.0 - 3.0 * G_shear * dgamma / (vm + 1.0e-30);
            if (scale < 0.0) scale = 0.0;

            for (int i = 0; i < 3; ++i)
                state.stress[i] = p_trial + scale * dev[i];
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale * dev[i];

            eps_p += dgamma;
        } else {
            for (int i = 0; i < 6; ++i)
                state.stress[i] = trial[i];
        }

        state.history[32] = eps_p;
        state.history[33] = sigma_y;
        state.plastic_strain = eps_p;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G_shear = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real eps_p = state.history[32];
        bool yielded = (eps_p > 0.0);
        Real factor = yielded ? (3.0 * G_shear) / (3.0 * G_shear + H_mod_) : 1.0;
        Real Geff = G_shear * factor;

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * Geff;
            for (int j = 0; j < 3; ++j)
                if (j != i) C[i * 6 + j] = lambda;
        }
        C[21] = Geff;
        C[28] = Geff;
        C[35] = Geff;
    }

    /// Interpolate yield stress at given angle (Bezier between reference points)
    KOKKOS_INLINE_FUNCTION
    Real interpolate_yield(Real theta) const {
        // Normalize theta to [0, 2*pi)
        Real two_pi = 2.0 * 3.14159265358979323846;
        while (theta < 0.0) theta += two_pi;
        while (theta >= two_pi) theta -= two_pi;

        // Find bracketing points
        int i0 = 0, i1 = 1;
        for (int i = 0; i < npts_ - 1; ++i) {
            if (theta >= pts_[i].theta && theta < pts_[i + 1].theta) {
                i0 = i;
                i1 = i + 1;
                break;
            }
        }

        Real dtheta = pts_[i1].theta - pts_[i0].theta;
        if (dtheta < 1.0e-30) return pts_[i0].sigma;
        Real t = (theta - pts_[i0].theta) / dtheta;

        // Quadratic Bezier: midpoint = average
        Real s0 = pts_[i0].sigma;
        Real s1 = pts_[i1].sigma;
        Real sm = 0.5 * (s0 + s1); // Bezier control point
        Real result = (1.0 - t) * (1.0 - t) * s0
                    + 2.0 * (1.0 - t) * t * sm
                    + t * t * s1;
        return result;
    }

    /// Get number of yield points
    int num_points() const { return npts_; }

private:
    YieldPoint pts_[8];
    int npts_;
    Real H_mod_;
};

// ============================================================================
// 3. MarlowHyperelasticMaterial - Test-Data-Driven Hyperelastic (LAW111)
// ============================================================================

/**
 * @brief Marlow hyperelastic material from test data
 *
 * Strain energy W(lambda) interpolated from stress-strain test data.
 * Stress = dW/dlambda. Neo-Hookean fallback outside data range.
 *
 * History: [32]=max_stretch, [33]=strain_energy
 */
class MarlowHyperelasticMaterial : public Material {
public:
    struct StressStrainPoint {
        Real strain;
        Real stress;
    };

    MarlowHyperelasticMaterial(const MaterialProperties& props,
                                const StressStrainPoint* data = nullptr, int npoints = 0)
        : Material(MaterialType::Hyperelastic, props)
        , npts_(npoints > 32 ? 32 : npoints)
    {
        for (int i = 0; i < 32; ++i) {
            if (i < npts_ && data) {
                data_[i].strain = data[i].strain;
                data_[i].stress = data[i].stress;
            } else {
                data_[i].strain = 0.0;
                data_[i].stress = 0.0;
            }
        }
        // Fallback Neo-Hookean modulus
        mu_nh_ = props.E / (2.0 * (1.0 + props.nu));
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));

        // Compute principal stretches from strain
        // lambda_i = 1 + epsilon_i (small strain approx)
        Real lam[3];
        for (int i = 0; i < 3; ++i)
            lam[i] = 1.0 + state.strain[i];

        Real J = lam[0] * lam[1] * lam[2];
        if (J < 0.01) J = 0.01;

        // Uniaxial stress from test data interpolation
        Real eng_strain_1 = state.strain[0];
        Real sigma_data = interpolate_stress(eng_strain_1);

        // For 3D: use data-driven response in direction 1,
        // Neo-Hookean for transverse + volumetric response
        Real p = K_bulk * (J - 1.0); // volumetric pressure

        // Principal Cauchy stresses
        // Direction 1: from test data (if available)
        if (npts_ > 1) {
            state.stress[0] = sigma_data + p;
            // Directions 2,3: Neo-Hookean transverse
            for (int i = 1; i < 3; ++i) {
                Real lam_sq = lam[i] * lam[i];
                state.stress[i] = mu * (lam_sq - 1.0 / (J * J + 1.0e-30)) / J + p;
            }
        } else {
            // Pure Neo-Hookean fallback
            for (int i = 0; i < 3; ++i) {
                Real lam_sq = lam[i] * lam[i];
                state.stress[i] = mu * (lam_sq - 1.0 / (J * J + 1.0e-30)) / J + p;
            }
        }

        // Shear
        Real G_shear = mu;
        for (int i = 3; i < 6; ++i)
            state.stress[i] = G_shear * state.strain[i];

        // Track max stretch and energy
        Real max_lam = Kokkos::fmax(lam[0], Kokkos::fmax(lam[1], lam[2]));
        if (max_lam > state.history[32]) state.history[32] = max_lam;

        // Approximate strain energy
        Real W = 0.0;
        for (int i = 0; i < 6; ++i)
            W += 0.5 * state.stress[i] * state.strain[i];
        state.history[33] = W;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G_shear = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        // Use tangent from test data for C11, elastic for rest
        Real C11_data = tangent_from_data(state.strain[0]);
        if (C11_data < 1.0e-10) C11_data = lambda + 2.0 * G_shear;

        C[0] = C11_data;
        C[7] = lambda + 2.0 * G_shear;
        C[14] = lambda + 2.0 * G_shear;
        C[1] = lambda; C[2] = lambda;
        C[6] = lambda; C[8] = lambda;
        C[12] = lambda; C[13] = lambda;
        C[21] = G_shear;
        C[28] = G_shear;
        C[35] = G_shear;
    }

    /// Interpolate stress from test data
    KOKKOS_INLINE_FUNCTION
    Real interpolate_stress(Real eps) const {
        if (npts_ < 2) return mu_nh_ * eps; // Neo-Hookean fallback

        // Below range
        if (eps <= data_[0].strain) {
            Real E_init = (data_[1].stress - data_[0].stress)
                        / (data_[1].strain - data_[0].strain + 1.0e-30);
            return data_[0].stress + E_init * (eps - data_[0].strain);
        }
        // Above range: Neo-Hookean extrapolation
        if (eps >= data_[npts_ - 1].strain) {
            return data_[npts_ - 1].stress + mu_nh_ * (eps - data_[npts_ - 1].strain);
        }

        // Linear interpolation
        for (int i = 0; i < npts_ - 1; ++i) {
            if (eps >= data_[i].strain && eps <= data_[i + 1].strain) {
                Real t = (eps - data_[i].strain)
                       / (data_[i + 1].strain - data_[i].strain + 1.0e-30);
                return data_[i].stress + t * (data_[i + 1].stress - data_[i].stress);
            }
        }
        return mu_nh_ * eps;
    }

    /// Tangent modulus from data (finite difference)
    KOKKOS_INLINE_FUNCTION
    Real tangent_from_data(Real eps) const {
        Real h = 1.0e-7;
        Real s1 = interpolate_stress(eps + h);
        Real s0 = interpolate_stress(eps - h);
        return (s1 - s0) / (2.0 * h);
    }

    int num_data_points() const { return npts_; }

private:
    StressStrainPoint data_[32];
    int npts_;
    Real mu_nh_;
};

// ============================================================================
// 4. DeshpandeFleckMaterial - Metal Foam Yield (LAW115)
// ============================================================================

/**
 * @brief Deshpande-Fleck isotropic metal foam yield
 *
 * Yield: sigma_e^2 + alpha^2 * sigma_m^2 = sigma_y^2(eps_p)
 * where sigma_e = von Mises, sigma_m = mean stress,
 * alpha controls shape (alpha=0 => von Mises).
 * Densification at eps_d causes stiffening.
 *
 * History: [32]=equivalent plastic strain, [33]=volumetric plastic strain
 */
class DeshpandeFleckMaterial : public Material {
public:
    DeshpandeFleckMaterial(const MaterialProperties& props,
                            Real sigma_p = 5.0e6, Real alpha_foam = 1.5,
                            Real densification_strain = 0.7)
        : Material(MaterialType::Custom, props)
        , sigma_p_(sigma_p), alpha_(alpha_foam), eps_d_(densification_strain)
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

        // Pressure and deviatoric
        Real p = (trial[0] + trial[1] + trial[2]) / 3.0;
        Real dev[6];
        for (int i = 0; i < 3; ++i) dev[i] = trial[i] - p;
        for (int i = 3; i < 6; ++i) dev[i] = trial[i];

        Real J2 = 0.0;
        for (int i = 0; i < 3; ++i) J2 += dev[i] * dev[i];
        for (int i = 3; i < 6; ++i) J2 += 2.0 * dev[i] * dev[i];
        J2 *= 0.5;
        Real vm = Kokkos::sqrt(3.0 * J2);

        // Deshpande-Fleck equivalent stress
        Real sigma_df = Kokkos::sqrt(vm * vm + alpha_ * alpha_ * p * p + 1.0e-30);

        // Current yield with hardening + densification
        Real eps_p = state.history[32];
        Real ev_p = state.history[33];
        Real ev_abs = Kokkos::fabs(ev_p);
        Real densification_factor = 1.0;
        if (ev_abs > eps_d_ * 0.5) {
            Real ratio = ev_abs / (eps_d_ + 1.0e-30);
            densification_factor = 1.0 + 10.0 * ratio * ratio;
        }
        Real sigma_y = sigma_p_ * densification_factor + props_.hardening_modulus * eps_p;

        if (sigma_df > sigma_y) {
            // Return mapping
            Real dgamma = (sigma_df - sigma_y) / (3.0 * G + K_bulk * alpha_ * alpha_ + props_.hardening_modulus + 1.0e-30);
            if (dgamma < 0.0) dgamma = 0.0;

            // Scale deviatoric
            Real scale_dev = 1.0 - 3.0 * G * dgamma / (vm + 1.0e-30);
            if (scale_dev < 0.0) scale_dev = 0.0;

            // Scale pressure
            Real p_corr = p - K_bulk * alpha_ * alpha_ * dgamma * p / (sigma_df + 1.0e-30);

            for (int i = 0; i < 3; ++i)
                state.stress[i] = p_corr + scale_dev * dev[i];
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale_dev * dev[i];

            eps_p += dgamma;
            ev_p -= dgamma * alpha_ * alpha_ * p / (sigma_df + 1.0e-30);
        } else {
            for (int i = 0; i < 6; ++i)
                state.stress[i] = trial[i];
        }

        state.history[32] = eps_p;
        state.history[33] = ev_p;
        state.plastic_strain = eps_p;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real eps_p = state.history[32];
        bool yielded = (eps_p > 0.0);
        Real denom = 3.0 * G + props_.hardening_modulus;
        Real factor = yielded ? (3.0 * G) / (denom + 1.0e-30) : 1.0;
        Real Geff = G * factor;

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * Geff;
            for (int j = 0; j < 3; ++j)
                if (j != i) C[i * 6 + j] = lambda;
        }
        C[21] = Geff;
        C[28] = Geff;
        C[35] = Geff;
    }

    /// Get Deshpande-Fleck equivalent stress
    KOKKOS_INLINE_FUNCTION
    Real df_equivalent(const Real* stress) const {
        Real p = (stress[0] + stress[1] + stress[2]) / 3.0;
        Real dev[6];
        for (int i = 0; i < 3; ++i) dev[i] = stress[i] - p;
        for (int i = 3; i < 6; ++i) dev[i] = stress[i];
        Real J2 = 0.0;
        for (int i = 0; i < 3; ++i) J2 += dev[i] * dev[i];
        for (int i = 3; i < 6; ++i) J2 += 2.0 * dev[i] * dev[i];
        J2 *= 0.5;
        Real vm = Kokkos::sqrt(3.0 * J2);
        return Kokkos::sqrt(vm * vm + alpha_ * alpha_ * p * p);
    }

    Real plateau_stress() const { return sigma_p_; }
    Real alpha() const { return alpha_; }

private:
    Real sigma_p_;
    Real alpha_;
    Real eps_d_;
};

// ============================================================================
// 5. ModifiedLaDevezeMaterial - Modified Composite Mesomodel (LAW122)
// ============================================================================

/**
 * @brief Modified LaDev`eze mesomodel for composite ply damage
 *
 * Damage variables:
 *   d12: shear damage, evolves with Y12 = tau12^2 / (2*G12*(1-d12)^2)
 *   d22: transverse damage, evolves with Y22 = sig22^2 / (2*E22*(1-d22)^2)
 * Fiber failure: when eps_11 > eps_fiber_max
 *
 * History: [32]=d12, [33]=d22, [34]=fiber_failed, [35]=Y12_max, [36]=Y22_max
 */
class ModifiedLaDevezeMaterial : public Material {
public:
    ModifiedLaDevezeMaterial(const MaterialProperties& props,
                              Real E11 = 140.0e9, Real E22 = 10.0e9, Real G12 = 5.0e9,
                              Real Yt = 50.0e6, Real S12 = 80.0e6,
                              Real d_fiber_crit = 0.99)
        : Material(MaterialType::Composite, props)
        , E11_(E11), E22_(E22), G12_(G12)
        , Yt_(Yt), S12_(S12), d_fiber_crit_(d_fiber_crit)
    {
        // Thermodynamic force thresholds
        Y12_0_ = S12 * S12 / (2.0 * G12);
        Y22_0_ = Yt * Yt / (2.0 * E22);
        eps_fiber_max_ = Yt / E11 * 10.0; // Fiber failure strain
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real nu12 = props_.nu12 > 0.0 ? props_.nu12 : 0.3;
        Real nu21 = nu12 * E22_ / E11_;

        // Retrieve damage
        Real d12 = state.history[32];
        Real d22 = state.history[33];
        Real fiber_failed = state.history[34];
        Real Y12_max = state.history[35];
        Real Y22_max = state.history[36];

        // Check fiber failure
        if (state.strain[0] > eps_fiber_max_ && fiber_failed < 0.5) {
            fiber_failed = 1.0;
        }

        Real d_fiber = fiber_failed > 0.5 ? d_fiber_crit_ : 0.0;

        // Effective moduli
        Real E11_eff = E11_ * (1.0 - d_fiber);
        Real E22_eff = E22_ * (1.0 - d22) * (1.0 - d22);
        Real G12_eff = G12_ * (1.0 - d12) * (1.0 - d12);

        if (E11_eff < 1.0) E11_eff = 1.0;
        if (E22_eff < 1.0) E22_eff = 1.0;
        if (G12_eff < 1.0) G12_eff = 1.0;

        // Plane stress compliance
        Real denom = 1.0 - nu12 * nu21 * (1.0 - d_fiber) * (1.0 - d22) * (1.0 - d22);
        if (Kokkos::fabs(denom) < 1.0e-20) denom = 1.0e-20;

        // Stress calculation (plane stress in 1-2 plane, 3D extension)
        Real Q11 = E11_eff / denom;
        Real Q22 = E22_eff / denom;
        Real Q12 = nu12 * E22_eff * (1.0 - d_fiber) / denom;

        state.stress[0] = Q11 * state.strain[0] + Q12 * state.strain[1];
        state.stress[1] = Q12 * state.strain[0] + Q22 * state.strain[1];
        state.stress[2] = 0.0; // Plane stress
        state.stress[3] = G12_eff * state.strain[3];
        state.stress[4] = 0.0;
        state.stress[5] = 0.0;

        // Update thermodynamic forces
        Real Y12_cur = state.stress[3] * state.stress[3] / (2.0 * G12_ + 1.0e-30);
        Real Y22_cur = 0.0;
        if (state.stress[1] > 0.0)
            Y22_cur = state.stress[1] * state.stress[1] / (2.0 * E22_ + 1.0e-30);

        // Max thermodynamic forces (damage driving)
        if (Y12_cur > Y12_max) Y12_max = Y12_cur;
        if (Y22_cur > Y22_max) Y22_max = Y22_cur;

        // Damage evolution (sqrt law for shear, linear for transverse)
        if (Y12_max > Y12_0_) {
            Real d12_new = Kokkos::sqrt(Y12_max / Y12_0_) - 1.0;
            if (d12_new > d12) d12 = d12_new;
            if (d12 > 0.99) d12 = 0.99;
        }
        if (Y22_max > Y22_0_) {
            Real d22_new = (Y22_max - Y22_0_) / (Y22_0_ * 5.0 + 1.0e-30);
            if (d22_new > d22) d22 = d22_new;
            if (d22 > 0.99) d22 = 0.99;
        }

        state.history[32] = d12;
        state.history[33] = d22;
        state.history[34] = fiber_failed;
        state.history[35] = Y12_max;
        state.history[36] = Y22_max;
        state.damage = Kokkos::fmax(d12, Kokkos::fmax(d22, d_fiber));
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real nu12 = props_.nu12 > 0.0 ? props_.nu12 : 0.3;
        Real nu21 = nu12 * E22_ / E11_;

        Real d12 = state.history[32];
        Real d22 = state.history[33];
        Real d_fiber = state.history[34] > 0.5 ? d_fiber_crit_ : 0.0;

        Real E11_eff = E11_ * (1.0 - d_fiber);
        Real E22_eff = E22_ * (1.0 - d22) * (1.0 - d22);
        Real G12_eff = G12_ * (1.0 - d12) * (1.0 - d12);
        if (E11_eff < 1.0) E11_eff = 1.0;
        if (E22_eff < 1.0) E22_eff = 1.0;
        if (G12_eff < 1.0) G12_eff = 1.0;

        Real denom = 1.0 - nu12 * nu21 * (1.0 - d_fiber) * (1.0 - d22) * (1.0 - d22);
        if (Kokkos::fabs(denom) < 1.0e-20) denom = 1.0e-20;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        C[0] = E11_eff / denom;
        C[1] = nu12 * E22_eff * (1.0 - d_fiber) / denom;
        C[6] = C[1];
        C[7] = E22_eff / denom;
        C[14] = E22_eff * 0.1; // Minimal out-of-plane
        C[21] = G12_eff;
        C[28] = G12_eff * 0.5;
        C[35] = G12_eff * 0.5;
    }

    /// Get shear damage
    KOKKOS_INLINE_FUNCTION
    Real shear_damage(const MaterialState& state) const { return state.history[32]; }

    /// Get transverse damage
    KOKKOS_INLINE_FUNCTION
    Real transverse_damage(const MaterialState& state) const { return state.history[33]; }

    /// Check fiber failure
    KOKKOS_INLINE_FUNCTION
    bool fiber_failed(const MaterialState& state) const { return state.history[34] > 0.5; }

private:
    Real E11_, E22_, G12_;
    Real Yt_, S12_;
    Real d_fiber_crit_;
    Real Y12_0_, Y22_0_;
    Real eps_fiber_max_;
};

// ============================================================================
// 6. CDPM2ConcreteMaterial - Two-Surface Concrete (LAW124)
// ============================================================================

/**
 * @brief CDPM2: two-surface concrete damage-plasticity model
 *
 * Yield surface: Drucker-Prager cone (compression) intersected with
 * tension cap (Rankine). Plastic + damage coupling:
 *   sigma_eff = (1-d_t)*(1-d_c) * C : (eps - eps_p)
 *
 * History: [32]=kappa_t, [33]=kappa_c, [34]=dam_t, [35]=dam_c,
 *          [36..41]=plastic_strain[6]
 */
class CDPM2ConcreteMaterial : public Material {
public:
    CDPM2ConcreteMaterial(const MaterialProperties& props,
                           Real fc = 30.0e6, Real ft = 3.0e6,
                           Real Gf_t = 100.0, Real Gf_c = 15000.0,
                           Real e_cc = 0.002, Real wf = 0.5)
        : Material(MaterialType::Custom, props)
        , fc_(fc), ft_(ft), Gf_t_(Gf_t), Gf_c_(Gf_c)
        , e_cc_(e_cc), wf_(wf)
    {
        // DP parameters from fc and ft
        Real ratio = fc / (ft + 1.0e-30);
        alpha_dp_ = (ratio - 1.0) / (2.0 * ratio - 1.0);
        k_dp_ = fc * (1.0 - alpha_dp_) / 3.0;
        h_elem_ = 0.01; // characteristic element length
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real K_bulk = E / (3.0 * (1.0 - 2.0 * nu));
        Real lambda = K_bulk - 2.0 * G / 3.0;

        // Retrieve plastic strain
        Real eps_p[6];
        for (int i = 0; i < 6; ++i)
            eps_p[i] = state.history[36 + i];

        // Elastic strain
        Real eps_e[6];
        for (int i = 0; i < 6; ++i)
            eps_e[i] = state.strain[i] - eps_p[i];

        // Trial stress
        Real ev_e = eps_e[0] + eps_e[1] + eps_e[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lambda * ev_e + 2.0 * G * eps_e[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = G * eps_e[i];

        // Invariants
        Real p_trial = (trial[0] + trial[1] + trial[2]) / 3.0;
        Real dev[6];
        for (int i = 0; i < 3; ++i) dev[i] = trial[i] - p_trial;
        for (int i = 3; i < 6; ++i) dev[i] = trial[i];

        Real J2 = 0.0;
        for (int i = 0; i < 3; ++i) J2 += dev[i] * dev[i];
        for (int i = 3; i < 6; ++i) J2 += 2.0 * dev[i] * dev[i];
        J2 *= 0.5;
        Real sqrt_J2 = Kokkos::sqrt(Kokkos::fmax(J2, 0.0));

        Real I1 = trial[0] + trial[1] + trial[2];

        // Retrieve damage
        Real kappa_t = state.history[32];
        Real kappa_c = state.history[33];
        Real d_t = state.history[34];
        Real d_c = state.history[35];

        // --- Compression yield (DP) ---
        Real f_dp = sqrt_J2 + alpha_dp_ * I1 - k_dp_ - props_.hardening_modulus * kappa_c;

        if (f_dp > 0.0) {
            // Return to DP surface
            Real dgamma = f_dp / (G + K_bulk * alpha_dp_ * alpha_dp_ + props_.hardening_modulus + 1.0e-30);
            if (dgamma < 0.0) dgamma = 0.0;

            // Update plastic strain
            Real norm_dev = sqrt_J2 + 1.0e-30;
            for (int i = 0; i < 3; ++i)
                eps_p[i] += dgamma * (dev[i] / (2.0 * norm_dev) + alpha_dp_ / 3.0);
            for (int i = 3; i < 6; ++i)
                eps_p[i] += dgamma * dev[i] / (2.0 * norm_dev);

            kappa_c += dgamma;

            // Recompute elastic strain and trial stress
            for (int i = 0; i < 6; ++i) eps_e[i] = state.strain[i] - eps_p[i];
            ev_e = eps_e[0] + eps_e[1] + eps_e[2];
            for (int i = 0; i < 3; ++i)
                trial[i] = lambda * ev_e + 2.0 * G * eps_e[i];
            for (int i = 3; i < 6; ++i)
                trial[i] = G * eps_e[i];
        }

        // --- Tension cap (Rankine) ---
        Real sigma_max = trial[0];
        for (int i = 1; i < 3; ++i)
            if (trial[i] > sigma_max) sigma_max = trial[i];

        Real ft_eff = ft_ * (1.0 - d_t);
        if (sigma_max > ft_eff && ft_eff > 0.0) {
            // Tension plastic update
            Real d_eps_t = (sigma_max - ft_eff) / (E + 1.0e-30);
            kappa_t += d_eps_t;

            // Scale back tensile stresses
            for (int i = 0; i < 3; ++i) {
                if (trial[i] > ft_eff)
                    trial[i] = ft_eff;
            }
        }

        // Damage evolution
        // Tensile damage: exponential softening
        if (kappa_t > 0.0) {
            Real eps_f_t = 2.0 * Gf_t_ / (ft_ * h_elem_ + 1.0e-30);
            Real d_t_new = 1.0 - Kokkos::exp(-3.0 * kappa_t / (eps_f_t + 1.0e-30));
            if (d_t_new > d_t) d_t = d_t_new;
            if (d_t > 0.99) d_t = 0.99;
        }

        // Compressive damage: parabolic-exponential
        if (kappa_c > e_cc_) {
            Real d_c_new = 1.0 - Kokkos::exp(-2.0 * (kappa_c - e_cc_) / (wf_ + 1.0e-30));
            if (d_c_new > d_c) d_c = d_c_new;
            if (d_c > 0.99) d_c = 0.99;
        }

        // Apply damage to effective stress
        Real omega = (1.0 - d_t) * (1.0 - d_c);
        for (int i = 0; i < 6; ++i)
            state.stress[i] = omega * trial[i];

        // Store history
        state.history[32] = kappa_t;
        state.history[33] = kappa_c;
        state.history[34] = d_t;
        state.history[35] = d_c;
        for (int i = 0; i < 6; ++i)
            state.history[36 + i] = eps_p[i];
        state.damage = 1.0 - omega;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        Real d_t = state.history[34];
        Real d_c = state.history[35];
        Real omega = (1.0 - d_t) * (1.0 - d_c);

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real Geff = G * omega;
        Real lam_eff = lambda * omega;

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lam_eff + 2.0 * Geff;
            for (int j = 0; j < 3; ++j)
                if (j != i) C[i * 6 + j] = lam_eff;
        }
        C[21] = Geff;
        C[28] = Geff;
        C[35] = Geff;
    }

    /// Get tensile damage
    KOKKOS_INLINE_FUNCTION
    Real tensile_damage(const MaterialState& state) const { return state.history[34]; }

    /// Get compressive damage
    KOKKOS_INLINE_FUNCTION
    Real compressive_damage(const MaterialState& state) const { return state.history[35]; }

    Real fc() const { return fc_; }
    Real ft() const { return ft_; }

private:
    Real fc_, ft_, Gf_t_, Gf_c_, e_cc_, wf_;
    Real alpha_dp_, k_dp_, h_elem_;
};

// ============================================================================
// 7. JHConcreteMaterial - Johnson-Holmquist Concrete (LAW126)
// ============================================================================

/**
 * @brief Johnson-Holmquist model for concrete/ceramics
 *
 * Intact strength:   sigma_i = A*(P* + T*)^N  (normalized)
 * Damaged strength:  sigma_f = B*(P*)^M
 * Current strength:  sigma = sigma_i - D*(sigma_i - sigma_f)
 * Damage: D = sum(delta_eps_p / eps_f(P)),  eps_f = D1*(P+T)^D2
 *
 * History: [32]=damage_D, [33]=accumulated_eps_p, [34]=pressure_state
 */
class JHConcreteMaterial : public Material {
public:
    JHConcreteMaterial(const MaterialProperties& props,
                        Real fc = 48.0e6, Real ft = 4.0e6,
                        Real A_jh = 0.79, Real B_jh = 1.60,
                        Real C_jh = 0.007, Real N_jh = 0.61,
                        Real SFMAX = 7.0)
        : Material(MaterialType::Custom, props)
        , fc_(fc), ft_(ft), A_(A_jh), B_(B_jh), C_(C_jh), N_(N_jh)
        , SFMAX_(SFMAX)
    {
        M_ = 0.61; // Damaged strength exponent
        D1_ = 0.04;
        D2_ = 1.0;
        T_star_ = ft / fc; // Normalized tensile strength
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

        // Pressure and deviatoric
        Real P = -(trial[0] + trial[1] + trial[2]) / 3.0; // positive in compression
        Real dev[6];
        Real p_hydro = -P;
        for (int i = 0; i < 3; ++i) dev[i] = trial[i] - p_hydro;
        for (int i = 3; i < 6; ++i) dev[i] = trial[i];

        Real J2 = 0.0;
        for (int i = 0; i < 3; ++i) J2 += dev[i] * dev[i];
        for (int i = 3; i < 6; ++i) J2 += 2.0 * dev[i] * dev[i];
        J2 *= 0.5;
        Real vm = Kokkos::sqrt(3.0 * J2);

        // Normalized pressure
        Real P_star = P / (fc_ + 1.0e-30);

        // Retrieve damage
        Real D = state.history[32];

        // Intact strength (normalized)
        Real P_plus_T = P_star + T_star_;
        Real sigma_i_star = 0.0;
        if (P_plus_T > 0.0) {
            sigma_i_star = A_ * Kokkos::pow(P_plus_T, N_);
            if (sigma_i_star > SFMAX_) sigma_i_star = SFMAX_;
        }

        // Rate effect
        Real eps_dot = state.effective_strain_rate;
        Real rate_factor = 1.0 + C_ * Kokkos::log(Kokkos::fmax(eps_dot, 1.0) + 1.0e-30);
        if (rate_factor < 1.0) rate_factor = 1.0;

        // Damaged strength (normalized)
        Real sigma_f_star = 0.0;
        if (P_star > 0.0) {
            sigma_f_star = B_ * Kokkos::pow(P_star, M_);
            if (sigma_f_star > SFMAX_) sigma_f_star = SFMAX_;
        }

        // Current strength
        Real sigma_star = sigma_i_star - D * (sigma_i_star - sigma_f_star);
        sigma_star *= rate_factor;
        Real sigma_limit = sigma_star * fc_;

        // Return mapping
        if (vm > sigma_limit && sigma_limit > 0.0) {
            Real scale = sigma_limit / (vm + 1.0e-30);
            for (int i = 0; i < 3; ++i)
                state.stress[i] = p_hydro + scale * dev[i];
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale * dev[i];

            // Plastic strain increment
            Real deps_p = (vm - sigma_limit) / (3.0 * G + 1.0e-30);
            Real eps_p_acc = state.history[33] + deps_p;

            // Fracture strain
            Real eps_f = D1_ * Kokkos::pow(Kokkos::fmax(P_star + T_star_, 1.0e-10), D2_);
            if (eps_f < 1.0e-6) eps_f = 1.0e-6;

            // Damage update
            D += deps_p / eps_f;
            if (D > 1.0) D = 1.0;

            state.history[33] = eps_p_acc;
        } else {
            for (int i = 0; i < 6; ++i)
                state.stress[i] = trial[i];
        }

        state.history[32] = D;
        state.history[34] = P;
        state.damage = D;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real D = state.history[32];
        Real omega = 1.0 - D;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real Geff = G * omega;
        Real lam_eff = lambda * omega;

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lam_eff + 2.0 * Geff;
            for (int j = 0; j < 3; ++j)
                if (j != i) C[i * 6 + j] = lam_eff;
        }
        C[21] = Geff;
        C[28] = Geff;
        C[35] = Geff;
    }

    /// Get damage parameter
    KOKKOS_INLINE_FUNCTION
    Real jh_damage(const MaterialState& state) const { return state.history[32]; }

    /// Get current pressure state
    KOKKOS_INLINE_FUNCTION
    Real pressure_state(const MaterialState& state) const { return state.history[34]; }

    Real fc() const { return fc_; }
    Real ft() const { return ft_; }

private:
    Real fc_, ft_;
    Real A_, B_, C_, N_, M_;
    Real SFMAX_;
    Real D1_, D2_;
    Real T_star_;
};

// ============================================================================
// 8. EnhancedCompositeMaterial - Fiber/Matrix Decomposition (LAW127)
// ============================================================================

/**
 * @brief Enhanced composite with fiber/matrix stress decomposition
 *
 * Fiber stress: sigma_f = Ef * Vf * eps_1 (longitudinal)
 * Matrix stress: sigma_m = Em * Vm * eps (transverse + shear)
 * Independent failure: fiber tensile/compressive, matrix tensile/compressive
 *
 * History: [32]=fiber_damage, [33]=matrix_damage, [34]=fiber_strain_max,
 *          [35]=matrix_strain_max
 */
class EnhancedCompositeMaterial : public Material {
public:
    EnhancedCompositeMaterial(const MaterialProperties& props,
                               Real Ef = 230.0e9, Real Em = 3.5e9,
                               Real Vf = 0.6, Real Vm = 0.4,
                               Real Xft = 3500.0e6, Real Xfc = 1500.0e6,
                               Real Ymt = 80.0e6, Real Ymc = 200.0e6)
        : Material(MaterialType::Composite, props)
        , Ef_(Ef), Em_(Em), Vf_(Vf), Vm_(Vm)
        , Xft_(Xft), Xfc_(Xfc), Ymt_(Ymt), Ymc_(Ymc)
    {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real nu_m = props_.nu > 0.0 ? props_.nu : 0.3;

        Real d_f = state.history[32];
        Real d_m = state.history[33];

        // Fiber contribution (longitudinal direction 1)
        Real sigma_f1 = Ef_ * Vf_ * (1.0 - d_f) * state.strain[0];

        // Matrix contribution (all directions)
        Real Gm = Em_ / (2.0 * (1.0 + nu_m));
        Real lam_m = Em_ * nu_m / ((1.0 + nu_m) * (1.0 - 2.0 * nu_m));
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];

        Real sigma_m[6];
        for (int i = 0; i < 3; ++i)
            sigma_m[i] = Vm_ * (1.0 - d_m) * (lam_m * ev + 2.0 * Gm * state.strain[i]);
        for (int i = 3; i < 6; ++i)
            sigma_m[i] = Vm_ * (1.0 - d_m) * Gm * state.strain[i];

        // Total stress
        state.stress[0] = sigma_f1 + sigma_m[0];
        for (int i = 1; i < 6; ++i)
            state.stress[i] = sigma_m[i];

        // Fiber failure check
        Real fiber_stress = Ef_ * Vf_ * state.strain[0];
        if (fiber_stress > Xft_ || fiber_stress < -Xfc_) {
            Real d_f_new = 0.99;
            if (d_f_new > d_f) d_f = d_f_new;
        } else {
            // Progressive damage
            Real eps_f1 = Kokkos::fabs(state.strain[0]);
            Real eps_f_crit = (state.strain[0] > 0.0) ? Xft_ / (Ef_ * Vf_ + 1.0e-30)
                                                        : Xfc_ / (Ef_ * Vf_ + 1.0e-30);
            if (eps_f1 > eps_f_crit * 0.8) {
                Real ratio = (eps_f1 - eps_f_crit * 0.8) / (eps_f_crit * 0.2 + 1.0e-30);
                Real d_f_new = ratio * 0.99;
                if (d_f_new > 1.0) d_f_new = 0.99;
                if (d_f_new > d_f) d_f = d_f_new;
            }
        }

        // Matrix failure check (transverse + shear)
        Real sigma_m22 = Em_ * Vm_ * state.strain[1];
        if (sigma_m22 > Ymt_ || sigma_m22 < -Ymc_) {
            Real d_m_new = 0.99;
            if (d_m_new > d_m) d_m = d_m_new;
        } else {
            // Shear failure
            Real tau12 = Gm * Vm_ * state.strain[3];
            Real shear_limit = 0.5 * (Ymt_ + Ymc_); // Approximate
            if (Kokkos::fabs(tau12) > shear_limit * 0.8) {
                Real ratio = (Kokkos::fabs(tau12) - shear_limit * 0.8) / (shear_limit * 0.2 + 1.0e-30);
                Real d_m_new = ratio * 0.99;
                if (d_m_new > 1.0) d_m_new = 0.99;
                if (d_m_new > d_m) d_m = d_m_new;
            }
        }

        state.history[32] = d_f;
        state.history[33] = d_m;
        state.history[34] = Kokkos::fmax(state.history[34], Kokkos::fabs(state.strain[0]));
        state.history[35] = Kokkos::fmax(state.history[35], Kokkos::fabs(state.strain[1]));
        state.damage = Kokkos::fmax(d_f, d_m);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real nu_m = props_.nu > 0.0 ? props_.nu : 0.3;
        Real Gm = Em_ / (2.0 * (1.0 + nu_m));
        Real lam_m = Em_ * nu_m / ((1.0 + nu_m) * (1.0 - 2.0 * nu_m));

        Real d_f = state.history[32];
        Real d_m = state.history[33];

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        // Fiber + matrix in direction 1
        C[0] = Ef_ * Vf_ * (1.0 - d_f) + Vm_ * (1.0 - d_m) * (lam_m + 2.0 * Gm);
        // Matrix only in directions 2,3
        C[7] = Vm_ * (1.0 - d_m) * (lam_m + 2.0 * Gm);
        C[14] = C[7];

        Real lam_eff = Vm_ * (1.0 - d_m) * lam_m;
        C[1] = lam_eff; C[2] = lam_eff;
        C[6] = lam_eff; C[8] = lam_eff;
        C[12] = lam_eff; C[13] = lam_eff;

        Real Geff = Vm_ * (1.0 - d_m) * Gm;
        C[21] = Geff;
        C[28] = Geff;
        C[35] = Geff;
    }

    /// Get fiber damage
    KOKKOS_INLINE_FUNCTION
    Real fiber_damage(const MaterialState& state) const { return state.history[32]; }

    /// Get matrix damage
    KOKKOS_INLINE_FUNCTION
    Real matrix_damage(const MaterialState& state) const { return state.history[33]; }

private:
    Real Ef_, Em_, Vf_, Vm_;
    Real Xft_, Xfc_, Ymt_, Ymc_;
};

// ============================================================================
// 9. GranularMaterial - Granular with Rolling Resistance (LAW133)
// ============================================================================

/**
 * @brief Granular material with Drucker-Prager yield + rolling resistance
 *
 * DP yield: F = sqrt(J2) + alpha*I1 - k
 * Rolling resistance: additional couple stress mu_roll * sigma_n * d_grain
 * Dilation angle psi controls volumetric plastic strain.
 *
 * History: [32]=plastic_shear_strain, [33]=volumetric_plastic_strain,
 *          [34]=rolling_dissipation
 */
class GranularMaterial : public Material {
public:
    GranularMaterial(const MaterialProperties& props,
                      Real phi = 30.0, Real psi = 15.0,
                      Real mu_roll = 0.1, Real d_grain = 0.001)
        : Material(MaterialType::Custom, props)
        , mu_roll_(mu_roll), d_grain_(d_grain)
    {
        Real pi = 3.14159265358979323846;
        Real phi_rad = phi * pi / 180.0;
        Real psi_rad = psi * pi / 180.0;
        Real sin_phi = Kokkos::sin(phi_rad);
        Real cos_phi = Kokkos::cos(phi_rad);
        Real sin_psi = Kokkos::sin(psi_rad);

        alpha_ = 2.0 * sin_phi / (1.7320508075688772 * (3.0 - sin_phi));
        beta_ = 2.0 * sin_psi / (1.7320508075688772 * (3.0 - sin_psi));
        // Cohesion: derive from yield_stress
        Real cohesion = props.yield_stress / (2.0 * cos_phi / (1.0 - sin_phi) + 1.0e-30);
        k0_ = 6.0 * cohesion * cos_phi / (1.7320508075688772 * (3.0 - sin_phi));
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

        // Invariants
        Real I1 = trial[0] + trial[1] + trial[2];
        Real p = I1 / 3.0;
        Real dev[6];
        for (int i = 0; i < 3; ++i) dev[i] = trial[i] - p;
        for (int i = 3; i < 6; ++i) dev[i] = trial[i];

        Real J2 = 0.0;
        for (int i = 0; i < 3; ++i) J2 += dev[i] * dev[i];
        for (int i = 3; i < 6; ++i) J2 += 2.0 * dev[i] * dev[i];
        J2 *= 0.5;
        Real sqrt_J2 = Kokkos::sqrt(Kokkos::fmax(J2, 0.0));

        // Hardening
        Real eps_ps = state.history[32];
        Real k_cur = k0_ + props_.hardening_modulus * eps_ps;

        // DP yield
        Real f = sqrt_J2 + alpha_ * I1 - k_cur;

        if (f > 0.0) {
            // Non-associated return mapping
            Real denom = G + K_bulk * alpha_ * beta_ + props_.hardening_modulus;
            Real dgamma = f / (denom + 1.0e-30);
            if (dgamma < 0.0) dgamma = 0.0;

            // Update deviatoric
            Real scale = 1.0 - G * dgamma / (sqrt_J2 + 1.0e-30);
            if (scale < 0.0) scale = 0.0;

            Real p_corr = p - K_bulk * beta_ * dgamma;

            for (int i = 0; i < 3; ++i)
                state.stress[i] = p_corr + scale * dev[i];
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale * dev[i];

            eps_ps += dgamma;

            // Rolling resistance dissipation
            Real sigma_n = Kokkos::fabs(p_corr);
            Real rolling_diss = mu_roll_ * sigma_n * d_grain_ * dgamma;
            state.history[34] += rolling_diss;
        } else {
            for (int i = 0; i < 6; ++i)
                state.stress[i] = trial[i];
        }

        // Tension cutoff (granular cannot sustain tension)
        Real p_cur = (state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;
        if (p_cur > 0.0) {
            // Limit tension
            Real t_limit = k0_ * 0.1; // Small tension allowed
            if (p_cur > t_limit) {
                Real shift = p_cur - t_limit;
                for (int i = 0; i < 3; ++i)
                    state.stress[i] -= shift;
            }
        }

        state.history[32] = eps_ps;
        state.history[33] += (state.stress[0] + state.stress[1] + state.stress[2]) / (3.0 * K_bulk + 1.0e-30);
        state.plastic_strain = eps_ps;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real eps_ps = state.history[32];
        Real factor = (eps_ps > 0.0) ? G / (G + props_.hardening_modulus + 1.0e-30) : 1.0;
        Real Geff = G * factor;

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * Geff;
            for (int j = 0; j < 3; ++j)
                if (j != i) C[i * 6 + j] = lambda;
        }
        C[21] = Geff;
        C[28] = Geff;
        C[35] = Geff;
    }

    /// Get rolling dissipation energy
    KOKKOS_INLINE_FUNCTION
    Real rolling_dissipation(const MaterialState& state) const { return state.history[34]; }

    Real friction_alpha() const { return alpha_; }

private:
    Real alpha_, beta_, k0_;
    Real mu_roll_, d_grain_;
};

// ============================================================================
// 10. ViscousFoamMaterial - Rate-Dependent Foam (LAW134)
// ============================================================================

/**
 * @brief Rate-dependent foam with hysteretic unloading
 *
 * sigma = E_plateau * (1 + C_rate * ln(1 + eps_dot/eps_dot_ref)) * f(eps)
 * where f(eps) = eps/(1 - eps/eps_lock) (densification function).
 * Unloading follows a reduced-stiffness path: sigma_unload = unload_frac * sigma_load
 *
 * History: [32]=max_strain_reached, [33]=is_unloading (0 or 1),
 *          [34]=max_stress_reached, [35]=energy_absorbed
 */
class ViscousFoamMaterial : public Material {
public:
    ViscousFoamMaterial(const MaterialProperties& props,
                         Real E_plateau = 1.0e6, Real eps_lock = 0.8,
                         Real C_rate = 0.05, Real unload_frac = 0.1)
        : Material(MaterialType::Custom, props)
        , E_p_(E_plateau), eps_lock_(eps_lock), C_rate_(C_rate)
        , unload_frac_(unload_frac)
    {
        eps_dot_ref_ = 1.0; // reference strain rate
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));

        // Volumetric strain (compression = negative)
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real eps_comp = -ev; // Positive in compression

        // Rate enhancement
        Real eps_dot = Kokkos::fabs(state.effective_strain_rate);
        Real rate_factor = 1.0 + C_rate_ * Kokkos::log(1.0 + eps_dot / eps_dot_ref_);
        if (rate_factor < 1.0) rate_factor = 1.0;

        // Densification function: f(eps) = eps / (1 - eps/eps_lock)
        Real eps_ratio = Kokkos::fabs(eps_comp) / (eps_lock_ + 1.0e-30);
        if (eps_ratio > 0.99) eps_ratio = 0.99;
        Real f_dens = eps_comp / (1.0 - eps_ratio);

        // Loading stress
        Real sigma_load = E_p_ * rate_factor * f_dens;

        // Track max strain
        Real max_strain = state.history[32];
        Real is_unloading = state.history[33];
        Real max_stress = state.history[34];

        // Detect loading/unloading
        if (eps_comp >= max_strain) {
            // Loading
            is_unloading = 0.0;
            max_strain = eps_comp;
            if (Kokkos::fabs(sigma_load) > Kokkos::fabs(max_stress))
                max_stress = sigma_load;
        } else {
            is_unloading = 1.0;
        }

        Real sigma_vol;
        if (is_unloading > 0.5) {
            // Hysteretic unloading
            Real unload_stress = unload_frac_ * max_stress * (eps_comp / (max_strain + 1.0e-30));
            sigma_vol = unload_stress;
        } else {
            sigma_vol = sigma_load;
        }

        // Apply hydrostatic stress (compression = negative stress)
        Real p = -sigma_vol / 3.0;
        for (int i = 0; i < 3; ++i)
            state.stress[i] = p + 2.0 * G * 0.1 * (state.strain[i] - ev / 3.0);
        // Reduced deviatoric: foam has low shear stiffness
        for (int i = 3; i < 6; ++i)
            state.stress[i] = G * 0.1 * state.strain[i];

        // Energy absorbed
        Real energy = state.history[35] + Kokkos::fabs(sigma_vol * ev) * state.dt;

        state.history[32] = max_strain;
        state.history[33] = is_unloading;
        state.history[34] = max_stress;
        state.history[35] = energy;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu)) * 0.1; // Reduced shear
        Real K_eff = E_p_; // Approximate bulk tangent

        if (state.history[33] > 0.5) {
            // Unloading stiffness
            K_eff *= unload_frac_;
        }

        Real lambda = K_eff - 2.0 * G / 3.0;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * G;
            for (int j = 0; j < 3; ++j)
                if (j != i) C[i * 6 + j] = lambda;
        }
        C[21] = G;
        C[28] = G;
        C[35] = G;
    }

    /// Get max strain reached
    KOKKOS_INLINE_FUNCTION
    Real max_strain(const MaterialState& state) const { return state.history[32]; }

    /// Is currently unloading?
    KOKKOS_INLINE_FUNCTION
    bool is_unloading(const MaterialState& state) const { return state.history[33] > 0.5; }

    /// Energy absorbed
    KOKKOS_INLINE_FUNCTION
    Real energy_absorbed(const MaterialState& state) const { return state.history[35]; }

    Real plateau_modulus() const { return E_p_; }

private:
    Real E_p_, eps_lock_, C_rate_, unload_frac_;
    Real eps_dot_ref_;
};

// ============================================================================
// 11. FabricNLMaterial - Nonlinear Fabric with Locking (LAW158)
// ============================================================================

/**
 * @brief Nonlinear fabric with shear locking angle
 *
 * Biaxial: E_warp in direction 1, E_weft in direction 2.
 * Shear: at low angles G is small; when gamma > lock_angle, shear
 * stiffness increases sharply (fabric "locking").
 *
 * History: [32]=max_shear_angle, [33]=fabric_tension_1, [34]=fabric_tension_2
 */
class FabricNLMaterial : public Material {
public:
    FabricNLMaterial(const MaterialProperties& props,
                      Real E_warp = 1.0e9, Real E_weft = 0.5e9,
                      Real E_lock = 50.0e9, Real shear_lock_angle = 0.5)
        : Material(MaterialType::Custom, props)
        , E_warp_(E_warp), E_weft_(E_weft), E_lock_(E_lock)
        , lock_angle_(shear_lock_angle)
    {
        // Low shear stiffness before locking
        G_low_ = props.E / (2.0 * (1.0 + props.nu)) * 0.01;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        // Direction 1 (warp): tension only
        Real sigma_1 = 0.0;
        if (state.strain[0] > 0.0) {
            sigma_1 = E_warp_ * state.strain[0];
        }

        // Direction 2 (weft): tension only
        Real sigma_2 = 0.0;
        if (state.strain[1] > 0.0) {
            sigma_2 = E_weft_ * state.strain[1];
        }

        // Direction 3: very low stiffness (out-of-plane)
        Real E3 = Kokkos::fmin(E_warp_, E_weft_) * 0.01;
        Real sigma_3 = E3 * state.strain[2];

        // Shear: locking behavior
        Real gamma = Kokkos::fabs(state.strain[3]);
        Real G_eff = G_low_;
        if (gamma > lock_angle_) {
            // Sharp increase: cubic stiffening
            Real excess = (gamma - lock_angle_) / (lock_angle_ + 1.0e-30);
            G_eff = G_low_ + E_lock_ * excess * excess * excess;
        } else if (gamma > lock_angle_ * 0.8) {
            // Transition zone
            Real t = (gamma - lock_angle_ * 0.8) / (lock_angle_ * 0.2 + 1.0e-30);
            G_eff = G_low_ * (1.0 + 10.0 * t * t);
        }

        state.stress[0] = sigma_1;
        state.stress[1] = sigma_2;
        state.stress[2] = sigma_3;
        state.stress[3] = G_eff * state.strain[3];
        state.stress[4] = G_low_ * state.strain[4];
        state.stress[5] = G_low_ * state.strain[5];

        // Track max shear angle
        if (gamma > state.history[32]) state.history[32] = gamma;
        state.history[33] = sigma_1;
        state.history[34] = sigma_2;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real E1_eff = (state.strain[0] > 0.0) ? E_warp_ : 1.0;
        Real E2_eff = (state.strain[1] > 0.0) ? E_weft_ : 1.0;
        Real E3 = Kokkos::fmin(E_warp_, E_weft_) * 0.01;

        C[0] = E1_eff;
        C[7] = E2_eff;
        C[14] = E3;

        Real gamma = Kokkos::fabs(state.strain[3]);
        Real G_eff = G_low_;
        if (gamma > lock_angle_) {
            Real excess = (gamma - lock_angle_) / (lock_angle_ + 1.0e-30);
            G_eff = G_low_ + E_lock_ * excess * excess * excess;
        }

        C[21] = G_eff;
        C[28] = G_low_;
        C[35] = G_low_;
    }

    /// Get max shear angle reached
    KOKKOS_INLINE_FUNCTION
    Real max_shear_angle(const MaterialState& state) const { return state.history[32]; }

    /// Check if fabric is locked
    KOKKOS_INLINE_FUNCTION
    bool is_locked(const MaterialState& state) const { return state.history[32] > lock_angle_; }

    Real lock_angle() const { return lock_angle_; }

private:
    Real E_warp_, E_weft_, E_lock_;
    Real lock_angle_;
    Real G_low_;
};

// ============================================================================
// 12. ARUPAdhesiveMaterial - Mixed-Mode Adhesive (LAW169)
// ============================================================================

/**
 * @brief ARUP-style mixed-mode adhesive with traction-separation
 *
 * Normal: T_n = sigma_n_max * f(delta_n / delta_n0)
 * Shear:  T_s = tau_s_max * f(delta_s / delta_s0)
 * Mixed-mode: (G_I/GIc)^alpha + (G_II/GIIc)^alpha = 1
 * f(x) = x*(2-x) for x < 1, then softening to 0 at delta_f
 *
 * History: [32]=max_delta_n, [33]=max_delta_s, [34]=G_I_accum,
 *          [35]=G_II_accum, [36]=damage
 */
class ARUPAdhesiveMaterial : public Material {
public:
    ARUPAdhesiveMaterial(const MaterialProperties& props,
                          Real GIc = 300.0, Real GIIc = 600.0,
                          Real sigma_n_max = 30.0e6, Real tau_s_max = 40.0e6,
                          Real delta_n0 = 0.001, Real delta_s0 = 0.002)
        : Material(MaterialType::Custom, props)
        , GIc_(GIc), GIIc_(GIIc)
        , sigma_n_max_(sigma_n_max), tau_s_max_(tau_s_max)
        , delta_n0_(delta_n0), delta_s0_(delta_s0)
    {
        // Failure opening (from energy: GIc = 0.5 * sigma_max * delta_f)
        delta_nf_ = 2.0 * GIc / (sigma_n_max + 1.0e-30);
        delta_sf_ = 2.0 * GIIc / (tau_s_max + 1.0e-30);
        alpha_mm_ = 2.0; // Power law exponent (BK criterion)
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        // Interface opening: use strain[2] as normal opening, strain[3] as shear
        Real delta_n = state.strain[2];
        Real delta_s = Kokkos::sqrt(state.strain[3] * state.strain[3]
                                  + state.strain[4] * state.strain[4] + 1.0e-30);

        // Track maximum openings (damage irreversibility)
        Real max_dn = state.history[32];
        Real max_ds = state.history[33];
        if (delta_n > max_dn) max_dn = delta_n;
        if (delta_s > max_ds) max_ds = delta_s;

        Real D = state.history[36];

        // Normal traction
        Real T_n = 0.0;
        if (delta_n > 0.0) {
            if (delta_n < delta_n0_) {
                // Linear rise
                Real ratio = delta_n / delta_n0_;
                T_n = sigma_n_max_ * ratio;
            } else if (delta_n < delta_nf_) {
                // Linear softening
                Real t = (delta_n - delta_n0_) / (delta_nf_ - delta_n0_ + 1.0e-30);
                T_n = sigma_n_max_ * (1.0 - t);
            } else {
                T_n = 0.0;
            }
        } else {
            // Compression: penalty stiffness
            Real K_pen = sigma_n_max_ / delta_n0_;
            T_n = K_pen * delta_n * 10.0; // Strong penalty in compression
        }

        // Shear traction
        Real T_s = 0.0;
        if (delta_s < delta_s0_) {
            Real ratio = delta_s / delta_s0_;
            T_s = tau_s_max_ * ratio;
        } else if (delta_s < delta_sf_) {
            Real t = (delta_s - delta_s0_) / (delta_sf_ - delta_s0_ + 1.0e-30);
            T_s = tau_s_max_ * (1.0 - t);
        }

        // Mixed-mode damage
        Real G_I = 0.5 * Kokkos::fmax(T_n, 0.0) * delta_n;
        Real G_II = 0.5 * T_s * delta_s;
        state.history[34] = G_I;
        state.history[35] = G_II;

        // Power law failure criterion
        Real ratio_I = G_I / (GIc_ + 1.0e-30);
        Real ratio_II = G_II / (GIIc_ + 1.0e-30);
        Real D_new = Kokkos::pow(ratio_I, alpha_mm_) + Kokkos::pow(ratio_II, alpha_mm_);
        if (D_new > D) D = D_new;
        if (D > 1.0) D = 1.0;

        // Apply damage
        Real omega = 1.0 - D;
        if (omega < 0.0) omega = 0.0;

        // Apply as stress (interface element interpretation)
        state.stress[0] = 0.0;
        state.stress[1] = 0.0;
        state.stress[2] = T_n * omega;
        if (delta_n < 0.0) state.stress[2] = T_n; // No damage in compression

        // Shear direction
        if (delta_s > 1.0e-30) {
            state.stress[3] = T_s * omega * state.strain[3] / delta_s;
            state.stress[4] = T_s * omega * state.strain[4] / delta_s;
        } else {
            state.stress[3] = 0.0;
            state.stress[4] = 0.0;
        }
        state.stress[5] = 0.0;

        state.history[32] = max_dn;
        state.history[33] = max_ds;
        state.history[36] = D;
        state.damage = D;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real D = state.history[36];
        Real omega = 1.0 - D;
        if (omega < 0.0) omega = 0.0;

        Real K_n = sigma_n_max_ / (delta_n0_ + 1.0e-30);
        Real K_s = tau_s_max_ / (delta_s0_ + 1.0e-30);

        C[14] = K_n * omega;
        C[21] = K_s * omega;
        C[28] = K_s * omega;
    }

    /// Get mixed-mode damage
    KOKKOS_INLINE_FUNCTION
    Real adhesive_damage(const MaterialState& state) const { return state.history[36]; }

    /// Get mode I energy release
    KOKKOS_INLINE_FUNCTION
    Real mode_I_energy(const MaterialState& state) const { return state.history[34]; }

    /// Get mode II energy release
    KOKKOS_INLINE_FUNCTION
    Real mode_II_energy(const MaterialState& state) const { return state.history[35]; }

    Real GIc() const { return GIc_; }
    Real GIIc() const { return GIIc_; }

private:
    Real GIc_, GIIc_;
    Real sigma_n_max_, tau_s_max_;
    Real delta_n0_, delta_s0_;
    Real delta_nf_, delta_sf_;
    Real alpha_mm_;
};

// ============================================================================
// 13. FoamDuboisMaterial - Dubois Foam (LAW190)
// ============================================================================

/**
 * @brief Dubois foam model with pressure coupling
 *
 * sigma = sigma_p * (1 + C_rate*eps_dot) * (1 + p_coeff*P/sigma_p) * h(eps_v)
 * where h(eps_v) is the densification function:
 *   h = eps_v / (1 - eps_v/eps_d) for compression
 * Pressure coupling: foam stiffens under hydrostatic compression.
 *
 * History: [32]=max_volumetric_strain, [33]=energy_absorbed,
 *          [34]=current_plateau_stress
 */
class FoamDuboisMaterial : public Material {
public:
    FoamDuboisMaterial(const MaterialProperties& props,
                        Real sigma_p = 2.0e6, Real eps_d = 0.7,
                        Real rho_ratio = 0.1, Real C_rate = 0.05,
                        Real p_coeff = 0.3)
        : Material(MaterialType::Custom, props)
        , sigma_p_(sigma_p), eps_d_(eps_d), rho_ratio_(rho_ratio)
        , C_rate_(C_rate), p_coeff_(p_coeff)
    {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));

        // Volumetric strain
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real eps_comp = -ev; // Positive in compression

        // Rate enhancement
        Real eps_dot = Kokkos::fabs(state.effective_strain_rate);
        Real rate_factor = 1.0 + C_rate_ * eps_dot;

        // Pressure (from external or self)
        Real P = -ev * E / (3.0 * (1.0 - 2.0 * nu));
        Real pressure_factor = 1.0 + p_coeff_ * Kokkos::fabs(P) / (sigma_p_ + 1.0e-30);

        // Densification function
        Real h_val = 0.0;
        if (eps_comp > 0.0) {
            Real ratio = eps_comp / (eps_d_ + 1.0e-30);
            if (ratio > 0.99) ratio = 0.99;
            h_val = eps_comp / (1.0 - ratio);
        } else {
            // Tension: linear
            h_val = eps_comp;
        }

        // Effective stress
        Real sigma_eff = sigma_p_ * rate_factor * pressure_factor * h_val;

        // Apply as hydrostatic + deviatoric
        Real sigma_h = -sigma_eff / 3.0;
        Real G_dev = G * rho_ratio_; // Reduced deviatoric stiffness

        for (int i = 0; i < 3; ++i)
            state.stress[i] = sigma_h + 2.0 * G_dev * (state.strain[i] - ev / 3.0);
        for (int i = 3; i < 6; ++i)
            state.stress[i] = G_dev * state.strain[i];

        // Track history
        if (eps_comp > state.history[32]) state.history[32] = eps_comp;
        state.history[33] += Kokkos::fabs(sigma_eff * ev) * state.dt;
        state.history[34] = sigma_p_ * rate_factor * pressure_factor;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu)) * rho_ratio_;

        // Effective bulk modulus
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real eps_comp = -ev;
        Real ratio = Kokkos::fabs(eps_comp) / (eps_d_ + 1.0e-30);
        if (ratio > 0.99) ratio = 0.99;
        Real K_eff = sigma_p_ / ((1.0 - ratio) * (1.0 - ratio) + 1.0e-30);

        Real lambda = K_eff - 2.0 * G / 3.0;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * G;
            for (int j = 0; j < 3; ++j)
                if (j != i) C[i * 6 + j] = lambda;
        }
        C[21] = G;
        C[28] = G;
        C[35] = G;
    }

    /// Get current plateau stress (with rate and pressure enhancement)
    KOKKOS_INLINE_FUNCTION
    Real current_plateau(const MaterialState& state) const { return state.history[34]; }

    /// Get energy absorbed
    KOKKOS_INLINE_FUNCTION
    Real energy_absorbed(const MaterialState& state) const { return state.history[33]; }

    Real sigma_p() const { return sigma_p_; }

private:
    Real sigma_p_, eps_d_, rho_ratio_, C_rate_, p_coeff_;
};

// ============================================================================
// 14. HenselSpittelMaterial - Hot Rolling Flow Stress (LAW103)
// ============================================================================

/**
 * @brief Hensel-Spittel hot forming flow stress model
 *
 * sigma_y = A * exp(m1*T) * T^m4 * eps^m2 * exp(m3/eps) * eps_dot^m5
 * where T = temperature, eps = effective plastic strain,
 * eps_dot = effective strain rate.
 *
 * Standard isotropic J2 plasticity with this flow stress.
 *
 * History: [32]=effective_plastic_strain, [33]=current_flow_stress
 */
class HenselSpittelMaterial : public Material {
public:
    HenselSpittelMaterial(const MaterialProperties& props,
                           Real A_hs = 1000.0, Real m1 = -0.003, Real m2 = 0.15,
                           Real m3 = 0.05, Real m4 = -0.002, Real m5 = 0.01)
        : Material(MaterialType::Custom, props)
        , A_(A_hs), m1_(m1), m2_(m2), m3_(m3), m4_(m4), m5_(m5)
    {}

    KOKKOS_INLINE_FUNCTION
    Real flow_stress(Real eps_p, Real eps_dot, Real T) const {
        // Guard against log/pow singularities
        Real eps_safe = Kokkos::fmax(eps_p, 1.0e-6);
        Real eps_dot_safe = Kokkos::fmax(eps_dot, 1.0e-3);
        Real T_safe = Kokkos::fmax(T, 293.0);

        // Clamp m3/eps to avoid overflow: |m3/eps| <= 20
        Real m3_over_eps = m3_ / (eps_safe + 1.0e-10);
        if (m3_over_eps > 20.0) m3_over_eps = 20.0;
        if (m3_over_eps < -20.0) m3_over_eps = -20.0;

        Real sigma = A_ * Kokkos::exp(m1_ * T_safe)
                        * Kokkos::pow(T_safe, m4_)
                        * Kokkos::pow(eps_safe, m2_)
                        * Kokkos::exp(m3_over_eps)
                        * Kokkos::pow(eps_dot_safe, m5_);
        if (sigma < 1.0) sigma = 1.0;
        return sigma;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        // Elastic trial
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lambda * ev + 2.0 * G * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = G * state.strain[i];

        // Deviatoric trial
        Real p = (trial[0] + trial[1] + trial[2]) / 3.0;
        Real dev[6];
        for (int i = 0; i < 3; ++i) dev[i] = trial[i] - p;
        for (int i = 3; i < 6; ++i) dev[i] = trial[i];

        Real J2 = 0.0;
        for (int i = 0; i < 3; ++i) J2 += dev[i] * dev[i];
        for (int i = 3; i < 6; ++i) J2 += 2.0 * dev[i] * dev[i];
        J2 *= 0.5;
        Real vm = Kokkos::sqrt(3.0 * J2);

        // Current yield
        Real eps_p = state.history[32];
        Real eps_dot = state.effective_strain_rate;
        Real T = state.temperature;
        Real sigma_y = flow_stress(eps_p, eps_dot, T);

        if (vm > sigma_y) {
            // Radial return
            Real dgamma = 0.0;
            for (int iter = 0; iter < 15; ++iter) {
                Real sy = flow_stress(eps_p + dgamma, eps_dot, T);
                Real f = vm - 3.0 * G * dgamma - sy;
                // Numerical tangent
                Real sy_plus = flow_stress(eps_p + dgamma + 1.0e-8, eps_dot, T);
                Real H_tan = (sy_plus - sy) / 1.0e-8;
                Real df = -3.0 * G - H_tan;
                dgamma -= f / (df - 1.0e-30);
                if (dgamma < 0.0) dgamma = 0.0;
                if (Kokkos::fabs(f) < 1.0e-8 * sigma_y) break;
            }

            Real scale = 1.0 - 3.0 * G * dgamma / (vm + 1.0e-30);
            if (scale < 0.0) scale = 0.0;

            for (int i = 0; i < 3; ++i)
                state.stress[i] = p + scale * dev[i];
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale * dev[i];

            eps_p += dgamma;
        } else {
            for (int i = 0; i < 6; ++i)
                state.stress[i] = trial[i];
        }

        state.history[32] = eps_p;
        state.history[33] = flow_stress(eps_p, eps_dot, T);
        state.plastic_strain = eps_p;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real eps_p = state.history[32];
        bool yielded = (eps_p > 1.0e-6);

        Real Geff = G;
        if (yielded) {
            Real sy = state.history[33];
            Real sy_plus = flow_stress(eps_p + 1.0e-8, state.effective_strain_rate, state.temperature);
            Real H_tan = (sy_plus - sy) / 1.0e-8;
            Geff = G * H_tan / (3.0 * G + H_tan + 1.0e-30);
        }

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * Geff;
            for (int j = 0; j < 3; ++j)
                if (j != i) C[i * 6 + j] = lambda;
        }
        C[21] = Geff;
        C[28] = Geff;
        C[35] = Geff;
    }

    /// Get current flow stress
    KOKKOS_INLINE_FUNCTION
    Real current_flow(const MaterialState& state) const { return state.history[33]; }

private:
    Real A_, m1_, m2_, m3_, m4_, m5_;
};

// ============================================================================
// 15. PaperLightMaterial - Paper/Lightweight Board (LAW107)
// ============================================================================

/**
 * @brief Paper/lightweight board material with orthotropic tension/compression
 *
 * Machine direction (MD), cross direction (CD), thickness (ZD).
 * Different moduli in tension vs compression.
 * Low stiffness, large deformation regime.
 *
 * History: [32]=max_md_strain, [33]=max_cd_strain, [34]=damage
 */
class PaperLightMaterial : public Material {
public:
    PaperLightMaterial(const MaterialProperties& props,
                        Real E_md = 5.0e9, Real E_cd = 2.5e9, Real E_zd = 0.5e9,
                        Real sigma_md = 50.0e6, Real sigma_cd = 25.0e6)
        : Material(MaterialType::Custom, props)
        , E_md_(E_md), E_cd_(E_cd), E_zd_(E_zd)
        , sigma_md_(sigma_md), sigma_cd_(sigma_cd)
    {
        // Compression moduli: typically 50% of tension
        E_md_c_ = E_md * 0.5;
        E_cd_c_ = E_cd * 0.5;
        // Shear modulus
        G_md_cd_ = Kokkos::sqrt(E_md * E_cd) / (2.0 * (1.0 + props.nu));
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real d = state.history[34];

        // MD direction (tension/compression)
        Real E1_eff = (state.strain[0] >= 0.0) ? E_md_ : E_md_c_;
        // Yield cap in MD
        Real sigma_1 = E1_eff * (1.0 - d) * state.strain[0];
        if (sigma_1 > sigma_md_) sigma_1 = sigma_md_;
        if (sigma_1 < -sigma_md_ * 0.5) sigma_1 = -sigma_md_ * 0.5;

        // CD direction (tension/compression)
        Real E2_eff = (state.strain[1] >= 0.0) ? E_cd_ : E_cd_c_;
        Real sigma_2 = E2_eff * (1.0 - d) * state.strain[1];
        if (sigma_2 > sigma_cd_) sigma_2 = sigma_cd_;
        if (sigma_2 < -sigma_cd_ * 0.5) sigma_2 = -sigma_cd_ * 0.5;

        // ZD direction
        Real sigma_3 = E_zd_ * (1.0 - d) * state.strain[2];

        state.stress[0] = sigma_1;
        state.stress[1] = sigma_2;
        state.stress[2] = sigma_3;

        // Shear
        state.stress[3] = G_md_cd_ * (1.0 - d) * state.strain[3];
        state.stress[4] = E_zd_ * 0.3 * (1.0 - d) * state.strain[4];
        state.stress[5] = E_zd_ * 0.3 * (1.0 - d) * state.strain[5];

        // Progressive damage (based on strain energy)
        Real W = 0.0;
        for (int i = 0; i < 6; ++i)
            W += 0.5 * Kokkos::fabs(state.stress[i] * state.strain[i]);

        Real W_crit = 0.5 * sigma_md_ * sigma_md_ / E_md_;
        if (W > W_crit) {
            Real d_new = 1.0 - Kokkos::exp(-(W - W_crit) / (W_crit + 1.0e-30));
            if (d_new > d) d = d_new;
            if (d > 0.99) d = 0.99;
        }

        state.history[32] = Kokkos::fmax(state.history[32], state.strain[0]);
        state.history[33] = Kokkos::fmax(state.history[33], state.strain[1]);
        state.history[34] = d;
        state.damage = d;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real d = state.history[34];
        Real omega = 1.0 - d;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real E1_eff = (state.strain[0] >= 0.0) ? E_md_ : E_md_c_;
        Real E2_eff = (state.strain[1] >= 0.0) ? E_cd_ : E_cd_c_;

        C[0] = E1_eff * omega;
        C[7] = E2_eff * omega;
        C[14] = E_zd_ * omega;
        C[21] = G_md_cd_ * omega;
        C[28] = E_zd_ * 0.3 * omega;
        C[35] = E_zd_ * 0.3 * omega;
    }

    /// Paper damage
    KOKKOS_INLINE_FUNCTION
    Real paper_damage(const MaterialState& state) const { return state.history[34]; }

private:
    Real E_md_, E_cd_, E_zd_;
    Real E_md_c_, E_cd_c_;
    Real sigma_md_, sigma_cd_;
    Real G_md_cd_;
};

// ============================================================================
// 16. JWLBMaterial - JWL + Afterburn (LAW97) [Thin Wrapper]
// ============================================================================

/**
 * @brief JWL equation of state with afterburn energy
 *
 * P = P_jwl(V) + (gamma - 1) * rho * Q(t)
 * where P_jwl = A*exp(-R1*V) + B*exp(-R2*V) + omega*E0/V
 * Q(t) = Q_max * (1 - exp(-t/tau_burn)) afterburn energy
 *
 * History: [32]=afterburn_energy, [33]=elapsed_time
 */
class JWLBMaterial : public Material {
public:
    JWLBMaterial(const MaterialProperties& props,
                  Real A_jwl = 3.712e11, Real B_jwl = 3.231e9,
                  Real R1 = 4.15, Real R2 = 0.95, Real omega = 0.30,
                  Real E0 = 7.0e9, Real Q_max = 2.0e9, Real tau_burn = 1.0e-4,
                  Real gamma_gas = 1.3)
        : Material(MaterialType::Custom, props)
        , A_(A_jwl), B_(B_jwl), R1_(R1), R2_(R2), omega_(omega)
        , E0_(E0), Q_max_(Q_max), tau_burn_(tau_burn), gamma_(gamma_gas)
    {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real rho0 = props_.density;

        // Relative volume: V = rho0/rho = 1/(1+ev_comp)
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real V = 1.0 / (1.0 - ev + 1.0e-30);
        if (V < 0.1) V = 0.1;
        if (V > 10.0) V = 10.0;

        // JWL pressure
        Real P_jwl = A_ * Kokkos::exp(-R1_ * V) + B_ * Kokkos::exp(-R2_ * V)
                   + omega_ * E0_ / (V + 1.0e-30);

        // Afterburn
        Real t_elapsed = state.history[33] + state.dt;
        Real Q = Q_max_ * (1.0 - Kokkos::exp(-t_elapsed / (tau_burn_ + 1.0e-30)));
        Real P_afterburn = (gamma_ - 1.0) * rho0 * Q / (V + 1.0e-30);

        Real P_total = P_jwl + P_afterburn;

        // Hydrostatic stress (pressure = -sigma_hydro)
        for (int i = 0; i < 3; ++i)
            state.stress[i] = -P_total;
        for (int i = 3; i < 6; ++i)
            state.stress[i] = 0.0;

        state.history[32] = Q;
        state.history[33] = t_elapsed;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real V = 1.0 / (1.0 - ev + 1.0e-30);
        if (V < 0.1) V = 0.1;

        // Approximate bulk modulus from JWL tangent
        Real K_eff = A_ * R1_ * Kokkos::exp(-R1_ * V) / (V * V)
                   + B_ * R2_ * Kokkos::exp(-R2_ * V) / (V * V)
                   + omega_ * E0_ / (V * V + 1.0e-30);

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = K_eff;
            for (int j = 0; j < 3; ++j)
                if (j != i) C[i * 6 + j] = K_eff;
        }
    }

    /// Get afterburn energy
    KOKKOS_INLINE_FUNCTION
    Real afterburn_energy(const MaterialState& state) const { return state.history[32]; }

private:
    Real A_, B_, R1_, R2_, omega_;
    Real E0_, Q_max_, tau_burn_, gamma_;
};

// ============================================================================
// 17. PPPolymerMaterial - Polypropylene (LAW101) [Thin Wrapper]
// ============================================================================

/**
 * @brief Polypropylene with tabulated yield + creep
 *
 * sigma_y = f(eps_p, T) + creep_rate * t^n_creep
 * Tabulated temperature-dependent yield + power-law creep.
 *
 * History: [32]=plastic_strain, [33]=creep_strain, [34]=creep_stress_contrib
 */
class PPPolymerMaterial : public Material {
public:
    PPPolymerMaterial(const MaterialProperties& props,
                       Real yield_20C = 35.0e6, Real yield_80C = 20.0e6,
                       Real creep_rate = 1.0e-8, Real n_creep = 0.3)
        : Material(MaterialType::Custom, props)
        , yield_20C_(yield_20C), yield_80C_(yield_80C)
        , creep_rate_(creep_rate), n_creep_(n_creep)
    {
        // Linear interpolation between 20C and 80C
        T_ref_ = 293.15;
        dsy_dT_ = (yield_80C - yield_20C) / (60.0); // per degree C
    }

    KOKKOS_INLINE_FUNCTION
    Real yield_at_temp(Real T) const {
        Real dT = T - T_ref_;
        Real sy = yield_20C_ + dsy_dT_ * dT;
        if (sy < 1.0e6) sy = 1.0e6; // Floor
        return sy;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        // Elastic trial
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lambda * ev + 2.0 * G * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = G * state.strain[i];

        // Von Mises
        Real p = (trial[0] + trial[1] + trial[2]) / 3.0;
        Real dev[6];
        for (int i = 0; i < 3; ++i) dev[i] = trial[i] - p;
        for (int i = 3; i < 6; ++i) dev[i] = trial[i];

        Real J2 = 0.0;
        for (int i = 0; i < 3; ++i) J2 += dev[i] * dev[i];
        for (int i = 3; i < 6; ++i) J2 += 2.0 * dev[i] * dev[i];
        J2 *= 0.5;
        Real vm = Kokkos::sqrt(3.0 * J2);

        // Temperature-dependent yield + creep
        Real eps_p = state.history[32];
        Real eps_creep = state.history[33];
        Real T = state.temperature;
        Real sigma_y = yield_at_temp(T) + props_.hardening_modulus * eps_p;

        // Creep strain increment
        Real t_total = eps_creep / (creep_rate_ + 1.0e-30);
        if (t_total < 1.0e-10) t_total = 1.0e-10;
        Real deps_creep = creep_rate_ * n_creep_ * Kokkos::pow(t_total, n_creep_ - 1.0) * state.dt;
        if (deps_creep < 0.0) deps_creep = 0.0;
        eps_creep += deps_creep;

        // Effective yield accounts for creep relaxation
        Real sigma_y_eff = sigma_y - E * eps_creep;
        if (sigma_y_eff < sigma_y * 0.3) sigma_y_eff = sigma_y * 0.3;

        if (vm > sigma_y_eff) {
            Real dgamma = (vm - sigma_y_eff) / (3.0 * G + props_.hardening_modulus + 1.0e-30);
            if (dgamma < 0.0) dgamma = 0.0;

            Real scale = 1.0 - 3.0 * G * dgamma / (vm + 1.0e-30);
            if (scale < 0.0) scale = 0.0;

            for (int i = 0; i < 3; ++i)
                state.stress[i] = p + scale * dev[i];
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale * dev[i];

            eps_p += dgamma;
        } else {
            for (int i = 0; i < 6; ++i)
                state.stress[i] = trial[i];
        }

        state.history[32] = eps_p;
        state.history[33] = eps_creep;
        state.history[34] = E * eps_creep;
        state.plastic_strain = eps_p;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real eps_p = state.history[32];
        Real factor = (eps_p > 0.0) ? (3.0 * G) / (3.0 * G + props_.hardening_modulus + 1.0e-30) : 1.0;
        Real Geff = G * factor;

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * Geff;
            for (int j = 0; j < 3; ++j)
                if (j != i) C[i * 6 + j] = lambda;
        }
        C[21] = Geff;
        C[28] = Geff;
        C[35] = Geff;
    }

    /// Get creep strain
    KOKKOS_INLINE_FUNCTION
    Real creep_strain(const MaterialState& state) const { return state.history[33]; }

private:
    Real yield_20C_, yield_80C_;
    Real creep_rate_, n_creep_;
    Real T_ref_, dsy_dT_;
};

// ============================================================================
// 18. DruckerPrager3Material - DP Variant 3 with Tension Cutoff (LAW102)
// ============================================================================

/**
 * @brief Drucker-Prager variant 3 with explicit tension cutoff
 *
 * Same as standard DP but with a tension cap: if I1 > 3*sigma_t_cutoff,
 * the stress is clamped. This prevents unrealistic tensile states.
 *
 * History: [32]=plastic_strain, [33]=volumetric_plastic_strain
 */
class DruckerPrager3Material : public Material {
public:
    DruckerPrager3Material(const MaterialProperties& props,
                            Real cohesion = 1.0e6, Real friction_angle = 30.0,
                            Real sigma_t_cutoff = 0.5e6)
        : Material(MaterialType::Custom, props)
        , cohesion_(cohesion), sigma_t_cutoff_(sigma_t_cutoff)
    {
        Real pi = 3.14159265358979323846;
        Real phi_rad = friction_angle * pi / 180.0;
        Real sin_phi = Kokkos::sin(phi_rad);
        Real cos_phi = Kokkos::cos(phi_rad);
        alpha_ = 2.0 * sin_phi / (1.7320508075688772 * (3.0 - sin_phi));
        k0_ = 6.0 * cohesion * cos_phi / (1.7320508075688772 * (3.0 - sin_phi));
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

        // Invariants
        Real I1 = trial[0] + trial[1] + trial[2];
        Real p = I1 / 3.0;
        Real dev[6];
        for (int i = 0; i < 3; ++i) dev[i] = trial[i] - p;
        for (int i = 3; i < 6; ++i) dev[i] = trial[i];

        Real J2 = 0.0;
        for (int i = 0; i < 3; ++i) J2 += dev[i] * dev[i];
        for (int i = 3; i < 6; ++i) J2 += 2.0 * dev[i] * dev[i];
        J2 *= 0.5;
        Real sqrt_J2 = Kokkos::sqrt(Kokkos::fmax(J2, 0.0));

        Real eps_p = state.history[32];
        Real k_cur = k0_ + props_.hardening_modulus * eps_p;

        // DP yield
        Real f = sqrt_J2 + alpha_ * I1 - k_cur;

        if (f > 0.0) {
            Real dgamma = f / (G + K_bulk * alpha_ * alpha_ + props_.hardening_modulus + 1.0e-30);
            if (dgamma < 0.0) dgamma = 0.0;

            Real scale = 1.0 - G * dgamma / (sqrt_J2 + 1.0e-30);
            if (scale < 0.0) scale = 0.0;

            Real p_corr = p - K_bulk * alpha_ * dgamma;
            for (int i = 0; i < 3; ++i)
                state.stress[i] = p_corr + scale * dev[i];
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale * dev[i];

            eps_p += dgamma;
        } else {
            for (int i = 0; i < 6; ++i)
                state.stress[i] = trial[i];
        }

        // Tension cutoff
        Real p_cur = (state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;
        if (p_cur > sigma_t_cutoff_) {
            Real shift = p_cur - sigma_t_cutoff_;
            for (int i = 0; i < 3; ++i)
                state.stress[i] -= shift;
        }

        state.history[32] = eps_p;
        state.plastic_strain = eps_p;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real eps_p = state.history[32];
        Real factor = (eps_p > 0.0) ? G / (G + props_.hardening_modulus + 1.0e-30) : 1.0;
        Real Geff = G * factor;

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * Geff;
            for (int j = 0; j < 3; ++j)
                if (j != i) C[i * 6 + j] = lambda;
        }
        C[21] = Geff;
        C[28] = Geff;
        C[35] = Geff;
    }

    /// Get tension cutoff
    Real tension_cutoff() const { return sigma_t_cutoff_; }

private:
    Real cohesion_, sigma_t_cutoff_;
    Real alpha_, k0_;
};

// ============================================================================
// 19. JCookAluminumMaterial - JC Aluminum with Thermal (LAW106) [Thin Wrapper]
// ============================================================================

/**
 * @brief Johnson-Cook specialized for aluminum alloys with thermal softening
 *
 * sigma_y = (A + B*eps_p^n) * (1 + C*ln(eps_dot*)) * (1 - T*^m)
 * Taylor-Quinney coefficient beta = 0.9 for adiabatic heating.
 * Temperature update: dT = beta * sigma * deps_p / (rho * Cp)
 *
 * History: [32]=plastic_strain, [33]=temperature_rise
 */
class JCookAluminumMaterial : public Material {
public:
    JCookAluminumMaterial(const MaterialProperties& props,
                           Real A = 324.0e6, Real B = 114.0e6, Real n = 0.42,
                           Real C = 0.002, Real m = 1.34,
                           Real T_melt = 925.0, Real T_room = 293.15)
        : Material(MaterialType::Custom, props)
        , A_(A), B_(B), n_(n), C_(C), m_(m)
        , T_melt_(T_melt), T_room_(T_room)
    {
        beta_tq_ = 0.9; // Taylor-Quinney
        eps_dot_ref_ = 1.0;
    }

    KOKKOS_INLINE_FUNCTION
    Real jc_yield(Real eps_p, Real eps_dot, Real T) const {
        Real eps_safe = Kokkos::fmax(eps_p, 1.0e-6);
        Real strain_term = A_ + B_ * Kokkos::pow(eps_safe, n_);
        Real rate_term = 1.0 + C_ * Kokkos::log(Kokkos::fmax(eps_dot / eps_dot_ref_, 1.0) + 1.0e-30);
        if (rate_term < 1.0) rate_term = 1.0;
        Real T_star = (T - T_room_) / (T_melt_ - T_room_ + 1.0e-30);
        if (T_star < 0.0) T_star = 0.0;
        if (T_star > 1.0) T_star = 1.0;
        Real thermal_term = 1.0 - Kokkos::pow(T_star, m_);
        if (thermal_term < 0.01) thermal_term = 0.01;
        return strain_term * rate_term * thermal_term;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        // Elastic trial
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real trial[6];
        for (int i = 0; i < 3; ++i)
            trial[i] = lambda * ev + 2.0 * G * state.strain[i];
        for (int i = 3; i < 6; ++i)
            trial[i] = G * state.strain[i];

        // Deviatoric
        Real p = (trial[0] + trial[1] + trial[2]) / 3.0;
        Real dev[6];
        for (int i = 0; i < 3; ++i) dev[i] = trial[i] - p;
        for (int i = 3; i < 6; ++i) dev[i] = trial[i];

        Real J2 = 0.0;
        for (int i = 0; i < 3; ++i) J2 += dev[i] * dev[i];
        for (int i = 3; i < 6; ++i) J2 += 2.0 * dev[i] * dev[i];
        J2 *= 0.5;
        Real vm = Kokkos::sqrt(3.0 * J2);

        Real eps_p = state.history[32];
        Real T = state.temperature + state.history[33];
        Real eps_dot = Kokkos::fmax(state.effective_strain_rate, eps_dot_ref_);
        Real sigma_y = jc_yield(eps_p, eps_dot, T);

        if (vm > sigma_y) {
            Real dgamma = (vm - sigma_y) / (3.0 * G + 1.0e-30);
            // Iterative correction
            for (int iter = 0; iter < 10; ++iter) {
                Real sy = jc_yield(eps_p + dgamma, eps_dot, T);
                Real f = vm - 3.0 * G * dgamma - sy;
                Real sy_p = jc_yield(eps_p + dgamma + 1.0e-8, eps_dot, T);
                Real H = (sy_p - sy) / 1.0e-8;
                dgamma -= f / (-3.0 * G - H + 1.0e-30);
                if (dgamma < 0.0) dgamma = 0.0;
                if (Kokkos::fabs(f) < 1.0e-8 * sigma_y) break;
            }

            Real scale = 1.0 - 3.0 * G * dgamma / (vm + 1.0e-30);
            if (scale < 0.0) scale = 0.0;

            for (int i = 0; i < 3; ++i)
                state.stress[i] = p + scale * dev[i];
            for (int i = 3; i < 6; ++i)
                state.stress[i] = scale * dev[i];

            eps_p += dgamma;

            // Adiabatic heating
            Real dT = beta_tq_ * sigma_y * dgamma / (props_.density * props_.specific_heat + 1.0e-30);
            state.history[33] += dT;
        } else {
            for (int i = 0; i < 6; ++i)
                state.stress[i] = trial[i];
        }

        state.history[32] = eps_p;
        state.plastic_strain = eps_p;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = E / (2.0 * (1.0 + nu));
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        Real eps_p = state.history[32];
        Real Geff = G;
        if (eps_p > 1.0e-6) {
            Real T = state.temperature + state.history[33];
            Real sy = jc_yield(eps_p, state.effective_strain_rate, T);
            Real sy_p = jc_yield(eps_p + 1.0e-8, state.effective_strain_rate, T);
            Real H = (sy_p - sy) / 1.0e-8;
            Geff = G * H / (3.0 * G + H + 1.0e-30);
        }

        for (int i = 0; i < 3; ++i) {
            C[i * 6 + i] = lambda + 2.0 * Geff;
            for (int j = 0; j < 3; ++j)
                if (j != i) C[i * 6 + j] = lambda;
        }
        C[21] = Geff;
        C[28] = Geff;
        C[35] = Geff;
    }

    /// Temperature rise from adiabatic heating
    KOKKOS_INLINE_FUNCTION
    Real temperature_rise(const MaterialState& state) const { return state.history[33]; }

private:
    Real A_, B_, n_, C_, m_;
    Real T_melt_, T_room_;
    Real beta_tq_, eps_dot_ref_;
};

// ============================================================================
// 20. SpringGeneralizedMaterial - 6-DOF Spring (LAW108) [Thin Wrapper]
// ============================================================================

/**
 * @brief Generalized 6-DOF spring element material
 *
 * Force_i = K_i * f_curve(displacement_i) for each DOF independently.
 * Translational: directions 0,1,2 with K_trans[3]
 * Rotational: directions 3,4,5 with K_rot[3]
 * Each DOF can have its own force-displacement curve.
 *
 * History: [32..34]=max_displacement_trans, [35..37]=max_displacement_rot
 */
class SpringGeneralizedMaterial : public Material {
public:
    SpringGeneralizedMaterial(const MaterialProperties& props,
                               const Real* K_trans = nullptr,
                               const Real* K_rot = nullptr)
        : Material(MaterialType::Custom, props)
    {
        for (int i = 0; i < 3; ++i) {
            K_trans_[i] = (K_trans) ? K_trans[i] : 1.0e6;
            K_rot_[i] = (K_rot) ? K_rot[i] : 1.0e3;
        }
        // Initialize curves (linear by default)
        for (int i = 0; i < 6; ++i) {
            curves_[i].add_point(-1.0, -1.0);
            curves_[i].add_point(0.0, 0.0);
            curves_[i].add_point(1.0, 1.0);
        }
    }

    /// Set a custom force-displacement curve for a specific DOF
    void set_curve(int dof, const TabulatedCurve& curve) {
        if (dof >= 0 && dof < 6) curves_[dof] = curve;
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        // Use strain[0..2] as translational displacements
        // Use strain[3..5] as rotational displacements
        for (int i = 0; i < 3; ++i) {
            Real disp = state.strain[i];
            Real f_curve = curves_[i].evaluate(disp);
            state.stress[i] = K_trans_[i] * f_curve;

            // Track max displacement
            Real abs_d = Kokkos::fabs(disp);
            if (abs_d > state.history[32 + i]) state.history[32 + i] = abs_d;
        }

        for (int i = 0; i < 3; ++i) {
            Real rot = state.strain[3 + i];
            Real f_curve = curves_[3 + i].evaluate(rot);
            state.stress[3 + i] = K_rot_[i] * f_curve;

            Real abs_r = Kokkos::fabs(rot);
            if (abs_r > state.history[35 + i]) state.history[35 + i] = abs_r;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        // Diagonal stiffness matrix
        for (int i = 0; i < 3; ++i) {
            // Numerical tangent from curve
            Real disp = state.strain[i];
            Real h = 1.0e-7;
            Real f_plus = curves_[i].evaluate(disp + h);
            Real f_minus = curves_[i].evaluate(disp - h);
            Real slope = (f_plus - f_minus) / (2.0 * h);
            C[i * 6 + i] = K_trans_[i] * slope;
        }

        for (int i = 0; i < 3; ++i) {
            Real rot = state.strain[3 + i];
            Real h = 1.0e-7;
            Real f_plus = curves_[3 + i].evaluate(rot + h);
            Real f_minus = curves_[3 + i].evaluate(rot - h);
            Real slope = (f_plus - f_minus) / (2.0 * h);
            C[(3 + i) * 6 + (3 + i)] = K_rot_[i] * slope;
        }
    }

    /// Get translational stiffness for a DOF
    Real trans_stiffness(int i) const { return (i >= 0 && i < 3) ? K_trans_[i] : 0.0; }

    /// Get rotational stiffness for a DOF
    Real rot_stiffness(int i) const { return (i >= 0 && i < 3) ? K_rot_[i] : 0.0; }

private:
    Real K_trans_[3];
    Real K_rot_[3];
    TabulatedCurve curves_[6];
};

} // namespace physics
} // namespace nxs
