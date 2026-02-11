#pragma once

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

/**
 * @brief Crushable foam material (equivalent to LS-DYNA MAT_063)
 *
 * Low-density foam with:
 * - Volumetric crush with tabulated stress-strain response
 * - Elastic unloading with reduced stiffness
 * - Tension cutoff
 * - Densification at high compression
 *
 * History variables:
 *   history[0]: maximum volumetric strain reached (for unloading)
 *   history[1]: current crush stress (for hysteresis)
 *   history[7]: volumetric plastic strain
 *
 * Properties used: foam_E_crush, foam_densification, foam_unload_factor, E, nu
 */
class CrushableFoamMaterial : public Material {
public:
    CrushableFoamMaterial(const MaterialProperties& props)
        : Material(MaterialType::CrushableFoam, props) {}

    CrushableFoamMaterial(const MaterialProperties& props,
                          const TabulatedCurve& crush_curve)
        : Material(MaterialType::CrushableFoam, props)
        , crush_curve_(crush_curve) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;

        // Volumetric strain (negative = compression)
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];

        // Deviatoric strain
        Real ev_3 = ev / 3.0;
        Real ed[6];
        ed[0] = state.strain[0] - ev_3;
        ed[1] = state.strain[1] - ev_3;
        ed[2] = state.strain[2] - ev_3;
        ed[3] = state.strain[3];
        ed[4] = state.strain[4];
        ed[5] = state.strain[5];

        // Deviatoric stress (elastic)
        Real sd[6];
        for (int i = 0; i < 3; ++i) sd[i] = 2.0 * G * ed[i];
        for (int i = 3; i < 6; ++i) sd[i] = G * ed[i];

        // Volumetric response
        Real p = 0.0;  // Pressure (positive = compression)

        if (ev < 0.0) {
            // Compression
            Real ev_abs = -ev;

            // Get crush stress from curve or analytical
            Real sigma_crush;
            if (crush_curve_.num_points > 0) {
                sigma_crush = crush_curve_.evaluate(ev_abs);
            } else {
                // Analytical: plateau + densification
                Real E_crush = props_.foam_E_crush;
                Real eps_d = props_.foam_densification;
                if (ev_abs < eps_d) {
                    sigma_crush = E_crush;
                } else {
                    // Exponential densification (prevent division by zero)
                    Real excess = ev_abs - eps_d;
                    Real denom = 1.0 - ev_abs;
                    if (denom < 0.01) denom = 0.01; // Cap near full compression
                    sigma_crush = E_crush + E * excess * excess / denom;
                }
            }

            // Track maximum compression for unloading
            Real ev_max = state.history[0]; // Already stored as positive
            if (ev_abs > ev_max) {
                // Loading: on the crush curve
                p = sigma_crush;
                state.history[0] = ev_abs;
                state.history[1] = sigma_crush;
            } else {
                // Unloading: reduced stiffness
                Real p_max = state.history[1];
                Real unload_E = props_.foam_unload_factor * E;
                Real delta_ev = ev_max - ev_abs;
                p = p_max - unload_E * delta_ev;
                if (p < 0.0) p = 0.0;
            }
        } else {
            // Tension: linear elastic (limited by tension cutoff)
            Real K = props_.K;
            p = -K * ev;  // Negative pressure = tension
            Real p_cutoff = -0.1 * props_.foam_E_crush; // 10% of crush stress
            if (p < p_cutoff) p = p_cutoff;
        }

        // Total stress = deviatoric + volumetric
        state.stress[0] = sd[0] - p;
        state.stress[1] = sd[1] - p;
        state.stress[2] = sd[2] - p;
        state.stress[3] = sd[3];
        state.stress[4] = sd[4];
        state.stress[5] = sd[5];

        state.vol_strain = ev;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        // Use elastic stiffness (conservative for implicit)
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
        return std::make_unique<CrushableFoamMaterial>(props);
    }

private:
    TabulatedCurve crush_curve_;
};

} // namespace physics
} // namespace nxs
