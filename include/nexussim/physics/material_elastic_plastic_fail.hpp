#pragma once

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

/**
 * @brief Elastic-plastic material with integrated failure
 *
 * J2 plasticity with linear hardening and a critical plastic strain
 * for element deletion. When ε_p >= failure_plastic_strain, the
 * damage flag is set and stress is reduced to zero.
 *
 * History variables: same as VonMises (history[0-6])
 * Properties used: yield_stress, hardening_modulus, failure_plastic_strain
 */
class ElasticPlasticFailMaterial : public Material {
public:
    ElasticPlasticFailMaterial(const MaterialProperties& props)
        : Material(MaterialType::ElasticPlasticFail, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        // Check if already failed
        if (state.damage >= 1.0) {
            for (int i = 0; i < 6; ++i) state.stress[i] = 0.0;
            return;
        }

        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;
        const Real sigma_y0 = props_.yield_stress;
        const Real H = props_.hardening_modulus;

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

        Real sigma_y = sigma_y0 + H * eps_p;
        Real f_trial = sigma_vm_trial - sigma_y;

        if (f_trial <= 0.0) {
            for (int i = 0; i < 6; ++i) state.stress[i] = stress_trial[i];
        } else {
            Real denom = 3.0 * G + H;
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

        // Check failure criterion
        if (state.plastic_strain >= props_.failure_plastic_strain) {
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
        Real G = props_.G;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real lambda_2mu = lambda + 2.0 * G;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lambda_2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = G;
    }

    static std::unique_ptr<Material> create(const MaterialProperties& props) {
        return std::make_unique<ElasticPlasticFailMaterial>(props);
    }
};

} // namespace physics
} // namespace nxs
