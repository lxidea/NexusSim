#pragma once

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

/**
 * @brief Piecewise-linear plasticity (equivalent to LS-DYNA MAT_024)
 *
 * J2 plasticity with a tabulated yield stress vs. effective plastic strain curve.
 * Uses radial return mapping with the tabulated hardening law.
 *
 * The yield curve is defined via a TabulatedCurve where:
 *   x = effective plastic strain
 *   y = yield stress at that strain level
 *
 * History variables:
 *   history[0]: accumulated effective plastic strain
 *   history[1-6]: plastic strain tensor (Voigt)
 */
class PiecewiseLinearMaterial : public Material {
public:
    PiecewiseLinearMaterial(const MaterialProperties& props,
                           const TabulatedCurve& yield_curve)
        : Material(MaterialType::PiecewiseLinear, props)
        , yield_curve_(yield_curve) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;

        Real eps_p = state.history[0];

        // Trial elastic strain
        Real strain_e[6];
        for (int i = 0; i < 6; ++i) {
            strain_e[i] = state.strain[i] - state.history[i + 1];
        }

        // Trial stress
        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        // Deviatoric trial stress
        const Real p_trial = (stress_trial[0] + stress_trial[1] + stress_trial[2]) / 3.0;
        Real s_trial[6];
        for (int i = 0; i < 3; ++i) s_trial[i] = stress_trial[i] - p_trial;
        for (int i = 3; i < 6; ++i) s_trial[i] = stress_trial[i];

        // Von Mises trial
        Real s_norm_sq = s_trial[0]*s_trial[0] + s_trial[1]*s_trial[1] + s_trial[2]*s_trial[2]
                       + 2.0*(s_trial[3]*s_trial[3] + s_trial[4]*s_trial[4] + s_trial[5]*s_trial[5]);
        Real sigma_vm_trial = Kokkos::sqrt(1.5 * s_norm_sq);

        // Current yield stress from curve
        Real sigma_y = yield_curve_.evaluate(eps_p);
        if (sigma_y < 1.0e-10) sigma_y = props_.yield_stress;

        Real f_trial = sigma_vm_trial - sigma_y;

        if (f_trial <= 0.0) {
            // Elastic
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            // Plastic: Newton iteration for nonlinear hardening
            Real delta_gamma = 0.0;
            Real eps_p_new = eps_p;

            for (int iter = 0; iter < 20; ++iter) {
                Real sy = yield_curve_.evaluate(eps_p_new);
                if (sy < 1.0e-10) sy = props_.yield_stress;

                Real sigma_vm_new = sigma_vm_trial - 3.0 * G * delta_gamma;
                Real R = sigma_vm_new - sy;

                if (Kokkos::fabs(R) < 1.0e-10 * sigma_y) break;

                // Numerical tangent for hardening slope
                Real deps = 1.0e-8 * (eps_p_new + 1.0e-10);
                Real sy_plus = yield_curve_.evaluate(eps_p_new + deps);
                Real H_tang = (sy_plus - sy) / deps;

                Real dR = -3.0 * G - H_tang;
                delta_gamma -= R / dR;
                if (delta_gamma < 0.0) delta_gamma = 0.0;
                eps_p_new = eps_p + delta_gamma;
            }

            state.history[0] = eps_p_new;

            // Update plastic strain tensor
            if (sigma_vm_trial > 1.0e-12) {
                Real factor = 1.5 * delta_gamma / sigma_vm_trial;
                for (int i = 0; i < 6; ++i) {
                    state.history[i + 1] += factor * s_trial[i];
                }
            }

            // Update stress
            Real scale = 1.0 - 3.0 * G * delta_gamma / sigma_vm_trial;
            for (int i = 0; i < 3; ++i) state.stress[i] = scale * s_trial[i] + p_trial;
            for (int i = 3; i < 6; ++i) state.stress[i] = scale * s_trial[i];
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;
        const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const Real lambda_2mu = lambda + 2.0 * G;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lambda_2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = G;

        // Reduce shear stiffness if yielding
        Real sigma_vm = von_mises_stress(state.stress);
        Real sigma_y = yield_curve_.evaluate(state.plastic_strain);
        if (sigma_y < 1.0e-10) sigma_y = props_.yield_stress;

        if (sigma_vm >= 0.99 * sigma_y && sigma_y > 0.0) {
            Real H = 0.0;
            Real deps = 1.0e-8;
            Real sy_plus = yield_curve_.evaluate(state.plastic_strain + deps);
            H = (sy_plus - sigma_y) / deps;
            Real theta = 3.0 * G / (3.0 * G + Kokkos::fmax(H, 0.0));
            C[21] *= (1.0 - theta);
            C[28] *= (1.0 - theta);
            C[35] *= (1.0 - theta);
        }
    }

    const TabulatedCurve& yield_curve() const { return yield_curve_; }

private:
    TabulatedCurve yield_curve_;
};

} // namespace physics
} // namespace nxs
