#pragma once

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

/**
 * @brief Viscoelastic material (Maxwell/Prony series)
 *
 * Generalized Maxwell model with Prony series relaxation:
 *   G(t) = G_inf + sum_k { G_k * exp(-t / tau_k) }
 *
 * Where G_inf = G * (1 - sum(g_k)) is the long-term shear modulus,
 * G_k = G * g_k are the relaxation moduli, and tau_k are the
 * relaxation times.
 *
 * History variables:
 *   history[7..10]: Internal overstress for Prony term 1 (h_xx, h_yy, h_zz, h_xy)
 *   history[11..14]: Internal overstress for Prony term 2
 *   history[15..18]: Internal overstress for Prony term 3
 *   history[19..22]: Internal overstress for Prony term 4
 *   history[23..28]: Previous deviatoric strain (for increment computation)
 *
 * Properties used: prony_g[4], prony_tau[4], prony_nterms, G, K
 */
class ViscoelasticMaterial : public Material {
public:
    ViscoelasticMaterial(const MaterialProperties& props)
        : Material(MaterialType::Viscoelastic, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real G = props_.G;
        const Real K = props_.K;
        int nt = props_.prony_nterms;
        if (nt <= 0) nt = 0;
        if (nt > 4) nt = 4;
        Real dt = state.dt;

        // Compute deviatoric strain
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];
        Real ev_3 = ev / 3.0;
        Real ed[6];
        ed[0] = state.strain[0] - ev_3;
        ed[1] = state.strain[1] - ev_3;
        ed[2] = state.strain[2] - ev_3;
        ed[3] = state.strain[3];
        ed[4] = state.strain[4];
        ed[5] = state.strain[5];

        // Retrieve old deviatoric strain and compute increment
        Real ded[6];
        for (int i = 0; i < 6; ++i) {
            ded[i] = ed[i] - state.history[23 + i];
            state.history[23 + i] = ed[i]; // Store for next step
        }

        // Long-term shear modulus
        Real g_sum = 0.0;
        for (int k = 0; k < nt; ++k) g_sum += props_.prony_g[k];
        Real G_inf = G * (1.0 - g_sum);

        // Long-term deviatoric stress
        Real sd[6];
        for (int i = 0; i < 3; ++i) sd[i] = 2.0 * G_inf * ed[i];
        for (int i = 3; i < 6; ++i) sd[i] = G_inf * ed[i];

        // Add Prony series overstress terms
        // h_k tracks the overstress from each Prony branch:
        //   h_k^{n+1} = exp(-dt/tau) * h_k^n + gk * G * 2 * d(ed)
        // At loading: overstress jumps up by gk * G * 2 * d(ed)
        // During hold: overstress decays exponentially
        for (int k = 0; k < nt; ++k) {
            Real gk = props_.prony_g[k];
            Real tau_k = props_.prony_tau[k];
            if (tau_k < 1.0e-30) continue;

            Real exp_dt = (dt > 0.0) ? Kokkos::exp(-dt / tau_k) : 1.0;
            int base = 7 + k * 4;

            for (int i = 0; i < 4; ++i) {
                Real de_i;
                if (i < 3) de_i = 2.0 * ded[i];   // Normal: 2G factor
                else de_i = ded[3];                  // Shear: G factor

                Real h_old = state.history[base + i];
                Real h_new = exp_dt * h_old + gk * G * de_i;
                state.history[base + i] = h_new;
            }

            // Add overstress to deviatoric stress
            sd[0] += state.history[base + 0];
            sd[1] += state.history[base + 1];
            sd[2] += state.history[base + 2];
            sd[3] += state.history[base + 3];
        }

        // Volumetric stress (elastic, no relaxation)
        Real p = K * ev;

        // Total stress
        for (int i = 0; i < 3; ++i) state.stress[i] = sd[i] + p;
        for (int i = 3; i < 6; ++i) state.stress[i] = sd[i];

        state.vol_strain = ev;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        // Use instantaneous (elastic) stiffness
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
        return std::make_unique<ViscoelasticMaterial>(props);
    }
};

} // namespace physics
} // namespace nxs
