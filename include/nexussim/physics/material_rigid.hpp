#pragma once

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

/**
 * @brief Rigid material (equivalent to LS-DYNA MAT_020)
 *
 * Elements with this material are treated as infinitely stiff — no deformation.
 * Stress is computed as very high elastic to resist any strain.
 * Used for rigid parts (barriers, impactors, fixtures).
 *
 * In practice, rigid parts should be modeled via the RigidBody system (Wave 3).
 * This material is a fallback for element-level rigidity.
 */
class RigidMaterial : public Material {
public:
    RigidMaterial(const MaterialProperties& props)
        : Material(MaterialType::Rigid, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        // Very high stiffness elastic response
        Real E_rigid = props_.E * 1000.0;  // 1000x stiffer
        Real nu = props_.nu;
        elastic_stress(state.strain, E_rigid, nu, state.stress);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real E = props_.E * 1000.0;
        Real nu = props_.nu;
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));
        Real lambda_2mu = lambda + 2.0 * mu;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lambda_2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

    static std::unique_ptr<Material> create(const MaterialProperties& props) {
        return std::make_unique<RigidMaterial>(props);
    }
};

} // namespace physics
} // namespace nxs
