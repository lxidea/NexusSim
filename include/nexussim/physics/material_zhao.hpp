#pragma once

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

/**
 * @brief Zhao tabulated rate-dependent plasticity
 *
 * Similar to Cowper-Symonds but uses tabulated rate sensitivity:
 *   σ_y(ε_p, ε̇) = σ_y0(ε_p) * R(ε̇)
 *
 * Where R(ε̇) is a tabulated rate factor curve.
 * The yield curve σ_y0(ε_p) can also be tabulated.
 *
 * History variables: same as VonMises (history[0-6])
 */
class ZhaoMaterial : public Material {
public:
    ZhaoMaterial(const MaterialProperties& props,
                 const TabulatedCurve& yield_curve,
                 const TabulatedCurve& rate_curve)
        : Material(MaterialType::Zhao, props)
        , yield_curve_(yield_curve)
        , rate_curve_(rate_curve) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;

        Real eps_p = state.history[0];

        Real strain_e[6];
        for (int i = 0; i < 6; ++i) {
            strain_e[i] = state.strain[i] - state.history[i + 1];
        }

        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        const Real p_trial = (stress_trial[0] + stress_trial[1] + stress_trial[2]) / 3.0;
        Real s_trial[6];
        for (int i = 0; i < 3; ++i) s_trial[i] = stress_trial[i] - p_trial;
        for (int i = 3; i < 6; ++i) s_trial[i] = stress_trial[i];

        Real s_norm_sq = s_trial[0]*s_trial[0] + s_trial[1]*s_trial[1] + s_trial[2]*s_trial[2]
                       + 2.0*(s_trial[3]*s_trial[3] + s_trial[4]*s_trial[4] + s_trial[5]*s_trial[5]);
        Real sigma_vm_trial = Kokkos::sqrt(1.5 * s_norm_sq);

        // Yield stress from curve
        Real sigma_y_static = yield_curve_.num_points > 0
            ? yield_curve_.evaluate(eps_p)
            : (props_.yield_stress + props_.hardening_modulus * eps_p);

        // Rate factor from curve
        Real eps_dot = state.effective_strain_rate;
        Real rate_factor = rate_curve_.num_points > 0
            ? rate_curve_.evaluate(eps_dot)
            : 1.0;
        if (rate_factor < 1.0) rate_factor = 1.0;

        Real sigma_y = sigma_y_static * rate_factor;
        Real f_trial = sigma_vm_trial - sigma_y;

        if (f_trial <= 0.0) {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            // Newton iteration for nonlinear hardening
            Real delta_gamma = 0.0;
            Real eps_p_new = eps_p;

            for (int iter = 0; iter < 20; ++iter) {
                Real sy = yield_curve_.num_points > 0
                    ? yield_curve_.evaluate(eps_p_new) * rate_factor
                    : (props_.yield_stress + props_.hardening_modulus * eps_p_new) * rate_factor;

                Real sigma_vm_new = sigma_vm_trial - 3.0 * G * delta_gamma;
                Real R = sigma_vm_new - sy;
                if (Kokkos::fabs(R) < 1.0e-10 * sigma_y) break;

                Real deps = 1.0e-8 * (eps_p_new + 1.0e-10);
                Real sy_plus = yield_curve_.num_points > 0
                    ? yield_curve_.evaluate(eps_p_new + deps) * rate_factor
                    : (props_.yield_stress + props_.hardening_modulus * (eps_p_new + deps)) * rate_factor;
                Real H_tang = (sy_plus - sy) / deps;

                delta_gamma -= R / (-3.0 * G - H_tang);
                if (delta_gamma < 0.0) delta_gamma = 0.0;
                eps_p_new = eps_p + delta_gamma;
            }

            state.history[0] = eps_p_new;
            if (sigma_vm_trial > 1.0e-12) {
                Real factor = 1.5 * delta_gamma / sigma_vm_trial;
                for (int i = 0; i < 6; ++i) state.history[i + 1] += factor * s_trial[i];
            }

            Real scale = 1.0 - 3.0 * G * delta_gamma / sigma_vm_trial;
            for (int i = 0; i < 3; ++i) state.stress[i] = scale * s_trial[i] + p_trial;
            for (int i = 3; i < 6; ++i) state.stress[i] = scale * s_trial[i];
        }

        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E;
        Real nu = props_.nu;
        Real G = props_.G;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lambda_2mu = lambda + 2.0 * G;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lambda_2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = G;
    }

private:
    TabulatedCurve yield_curve_;
    TabulatedCurve rate_curve_;
};

} // namespace physics
} // namespace nxs
