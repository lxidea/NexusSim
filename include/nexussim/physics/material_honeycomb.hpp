#pragma once

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

/**
 * @brief Orthotropic honeycomb crush material (equivalent to LS-DYNA MAT_026)
 *
 * Models orthotropic crush with different stress-strain responses in
 * three principal directions. Used for aluminum honeycomb, Nomex, and
 * similar energy absorbers.
 *
 * Features:
 * - Independent crush curves in each direction
 * - Elastic unloading with configurable stiffness
 * - Densification response
 *
 * History variables:
 *   history[7]: max strain in direction 1
 *   history[8]: max strain in direction 2
 *   history[9]: max strain in direction 3
 *   history[10]: crush stress dir 1
 *   history[11]: crush stress dir 2
 *   history[12]: crush stress dir 3
 *
 * Properties used: E1/E2/E3 (directional moduli), G12/G23/G13 (shear),
 *                  foam_E_crush (plateau stress), foam_densification
 */
class HoneycombMaterial : public Material {
public:
    HoneycombMaterial(const MaterialProperties& props)
        : Material(MaterialType::Honeycomb, props) {}

    HoneycombMaterial(const MaterialProperties& props,
                      const TabulatedCurve& crush1,
                      const TabulatedCurve& crush2,
                      const TabulatedCurve& crush3)
        : Material(MaterialType::Honeycomb, props)
        , crush_curve1_(crush1), crush_curve2_(crush2), crush_curve3_(crush3) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real E1 = props_.E1 > 0.0 ? props_.E1 : props_.E;
        Real E2 = props_.E2 > 0.0 ? props_.E2 : props_.E;
        Real E3 = props_.E3 > 0.0 ? props_.E3 : props_.E;
        Real G12 = props_.G12 > 0.0 ? props_.G12 : props_.G;
        Real G23 = props_.G23 > 0.0 ? props_.G23 : props_.G;
        Real G13 = props_.G13 > 0.0 ? props_.G13 : props_.G;
        Real E_crush = props_.foam_E_crush;
        Real eps_d = props_.foam_densification;

        // Process each normal direction independently
        for (int dir = 0; dir < 3; ++dir) {
            Real eps = state.strain[dir];
            Real E_dir = (dir == 0) ? E1 : ((dir == 1) ? E2 : E3);

            if (eps < 0.0) {
                // Compression
                Real eps_abs = -eps;
                Real sigma_crush;

                const TabulatedCurve& curve = (dir == 0) ? crush_curve1_ :
                                              ((dir == 1) ? crush_curve2_ : crush_curve3_);

                if (curve.num_points > 0) {
                    sigma_crush = curve.evaluate(eps_abs);
                } else {
                    // Default: plateau then densification
                    if (eps_abs < eps_d) {
                        sigma_crush = E_crush;
                    } else {
                        Real excess = eps_abs - eps_d;
                        sigma_crush = E_crush + E_dir * excess * excess / (1.0 - eps_abs + 1.0e-10);
                    }
                }

                // Unloading check
                Real eps_max = state.history[7 + dir];
                if (eps_abs > eps_max) {
                    state.stress[dir] = -sigma_crush;
                    state.history[7 + dir] = eps_abs;
                    state.history[10 + dir] = sigma_crush;
                } else {
                    Real sigma_max = state.history[10 + dir];
                    Real unload_E = props_.foam_unload_factor * E_dir;
                    Real delta = eps_max - eps_abs;
                    Real sigma = sigma_max - unload_E * delta;
                    if (sigma < 0.0) sigma = 0.0;
                    state.stress[dir] = -sigma;
                }
            } else {
                // Tension: linear elastic with low cutoff
                Real sigma_t = E_dir * eps;
                Real cutoff = 0.1 * E_crush;
                if (sigma_t > cutoff) sigma_t = cutoff;
                state.stress[dir] = sigma_t;
            }
        }

        // Shear: linear elastic
        state.stress[3] = G12 * state.strain[3];
        state.stress[4] = G23 * state.strain[4];
        state.stress[5] = G13 * state.strain[5];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E1 = props_.E1 > 0.0 ? props_.E1 : props_.E;
        Real E2 = props_.E2 > 0.0 ? props_.E2 : props_.E;
        Real E3 = props_.E3 > 0.0 ? props_.E3 : props_.E;
        Real G12 = props_.G12 > 0.0 ? props_.G12 : props_.G;
        Real G23 = props_.G23 > 0.0 ? props_.G23 : props_.G;
        Real G13 = props_.G13 > 0.0 ? props_.G13 : props_.G;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = E1; C[7] = E2; C[14] = E3;
        C[21] = G12; C[28] = G23; C[35] = G13;
    }

    static std::unique_ptr<Material> create(const MaterialProperties& props) {
        return std::make_unique<HoneycombMaterial>(props);
    }

private:
    TabulatedCurve crush_curve1_, crush_curve2_, crush_curve3_;
};

} // namespace physics
} // namespace nxs
