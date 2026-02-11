#pragma once

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

/**
 * @brief Null material (pressure only, no deviatoric strength)
 *
 * Used for fluid-like elements where only volumetric (pressure) response
 * is needed. The deviatoric stress is zero. Typically paired with an
 * Equation of State (EOS) in Wave 5 for the volumetric part.
 *
 * Without EOS: uses linear bulk modulus for pressure.
 * Properties used: K (bulk modulus)
 */
class NullMaterial : public Material {
public:
    NullMaterial(const MaterialProperties& props)
        : Material(MaterialType::Null, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        Real K = props_.K;

        // Volumetric strain
        Real ev = state.strain[0] + state.strain[1] + state.strain[2];

        // Pressure only (negative sign: compression = positive pressure)
        Real p = K * ev;

        // Hydrostatic stress, no deviatoric
        state.stress[0] = p;
        state.stress[1] = p;
        state.stress[2] = p;
        state.stress[3] = 0.0;
        state.stress[4] = 0.0;
        state.stress[5] = 0.0;

        state.vol_strain = ev;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real K = props_.K;
        // Only volumetric stiffness, no shear
        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = K;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = K;
        // No shear terms (G = 0)
    }

    static std::unique_ptr<Material> create(const MaterialProperties& props) {
        return std::make_unique<NullMaterial>(props);
    }
};

} // namespace physics
} // namespace nxs
