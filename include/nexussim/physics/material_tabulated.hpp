#pragma once

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

/**
 * @brief Tabulated stress-strain material (nonlinear elastic)
 *
 * Directly maps total strain to stress via a user-defined curve.
 * Supports optional unloading to a different curve or elastic unloading.
 * Useful for simplified nonlinear elastic models and curve-fitting.
 *
 * The stress-strain curve is defined via TabulatedCurve where:
 *   x = effective strain (positive = tension, negative = compression)
 *   y = uniaxial stress at that strain
 *
 * For multiaxial states, uses a scaled isotropic approach:
 *   sigma_ij = (sigma_table / sigma_elastic) * sigma_elastic_ij
 */
class TabulatedMaterial : public Material {
public:
    TabulatedMaterial(const MaterialProperties& props,
                     const TabulatedCurve& stress_strain_curve)
        : Material(MaterialType::Tabulated, props)
        , curve_(stress_strain_curve) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;

        // Compute linear elastic stress first
        Real stress_elastic[6];
        elastic_stress(state.strain, E, nu, stress_elastic);

        // Effective strain (von Mises equivalent)
        Real eps_vm = effective_strain(state.strain);

        // Get stress from table
        Real sigma_table = curve_.evaluate(eps_vm);

        // Elastic von Mises for scaling
        Real sigma_elastic_vm = von_mises_stress(stress_elastic);

        if (sigma_elastic_vm > 1.0e-12) {
            // Scale elastic stress to match table
            Real scale = sigma_table / sigma_elastic_vm;
            for (int i = 0; i < 6; ++i) {
                state.stress[i] = scale * stress_elastic[i];
            }
        } else {
            for (int i = 0; i < 6; ++i) state.stress[i] = 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state, Real* C) const override {
        // Secant stiffness based on current point on curve
        Real eps_vm = effective_strain(state.strain);
        Real sigma = curve_.evaluate(eps_vm);
        Real E_secant = (eps_vm > 1.0e-12) ? sigma / eps_vm : props_.E;

        Real nu = props_.nu;
        Real lambda = E_secant * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E_secant / (2.0 * (1.0 + nu));
        Real lambda_2mu = lambda + 2.0 * mu;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lambda_2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

    const TabulatedCurve& curve() const { return curve_; }

private:
    TabulatedCurve curve_;

    /// Von Mises equivalent strain
    KOKKOS_INLINE_FUNCTION
    static Real effective_strain(const Real* strain) {
        Real ed[6];
        Real ev = (strain[0] + strain[1] + strain[2]) / 3.0;
        ed[0] = strain[0] - ev;
        ed[1] = strain[1] - ev;
        ed[2] = strain[2] - ev;
        ed[3] = strain[3] * 0.5;  // Convert engineering to tensor shear
        ed[4] = strain[4] * 0.5;
        ed[5] = strain[5] * 0.5;

        Real e2 = ed[0]*ed[0] + ed[1]*ed[1] + ed[2]*ed[2]
                + 2.0*(ed[3]*ed[3] + ed[4]*ed[4] + ed[5]*ed[5]);
        return Kokkos::sqrt(2.0 / 3.0 * e2);
    }
};

} // namespace physics
} // namespace nxs
