#pragma once

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

/**
 * @brief Cowper-Symonds rate-dependent plasticity
 *
 * J2 plasticity with strain-rate enhanced yield stress:
 *   σ_y(ε_p, ε̇) = σ_y0(ε_p) * (1 + (ε̇/D)^(1/q))
 *
 * Where D and q are the Cowper-Symonds parameters.
 * Linear isotropic hardening: σ_y0 = σ_y0_initial + H * ε_p
 *
 * History variables: same as VonMises (history[0-6])
 * Properties used: yield_stress, hardening_modulus, CS_D, CS_q
 */
class CowperSymondsMaterial : public Material {
public:
    CowperSymondsMaterial(const MaterialProperties& props)
        : Material(MaterialType::CowperSymonds, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;
        const Real sigma_y0 = props_.yield_stress;
        const Real H = props_.hardening_modulus;
        const Real D = props_.CS_D;
        const Real q = props_.CS_q;

        Real eps_p = state.history[0];

        // Trial elastic strain
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

        // Static yield + hardening
        Real sigma_y_static = sigma_y0 + H * eps_p;

        // Rate enhancement
        Real eps_dot = state.effective_strain_rate;
        Real rate_factor = 1.0;
        if (D > 0.0 && q > 0.0 && eps_dot > 0.0) {
            rate_factor = 1.0 + Kokkos::pow(eps_dot / D, 1.0 / q);
        }

        Real sigma_y = sigma_y_static * rate_factor;

        Real f_trial = sigma_vm_trial - sigma_y;

        if (f_trial <= 0.0) {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            // Radial return (linear hardening → closed form)
            Real denom = 3.0 * G + H * rate_factor;
            Real delta_gamma = f_trial / denom;

            state.history[0] = eps_p + delta_gamma;

            if (sigma_vm_trial > 1.0e-12) {
                Real factor = 1.5 * delta_gamma / sigma_vm_trial;
                for (int i = 0; i < 6; ++i) {
                    state.history[i + 1] += factor * s_trial[i];
                }
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

    static std::unique_ptr<Material> create(const MaterialProperties& props) {
        return std::make_unique<CowperSymondsMaterial>(props);
    }
};

} // namespace physics
} // namespace nxs
