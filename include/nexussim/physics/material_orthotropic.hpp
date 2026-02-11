#pragma once

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

/**
 * @brief Orthotropic elastic material model
 *
 * 9-constant orthotropic elasticity for composite shells and layered materials.
 * Uses directional moduli (E1, E2, E3), shear moduli (G12, G23, G13),
 * and Poisson's ratios (nu12, nu23, nu13).
 *
 * Compliance matrix S (inverse of stiffness C):
 *   S = [1/E1    -nu12/E1  -nu13/E1  0      0      0    ]
 *       [-nu12/E1  1/E2    -nu23/E2  0      0      0    ]
 *       [-nu13/E1 -nu23/E2  1/E3     0      0      0    ]
 *       [0         0        0        1/G12  0      0    ]
 *       [0         0        0        0      1/G23  0    ]
 *       [0         0        0        0      0      1/G13]
 */
class OrthotropicMaterial : public Material {
public:
    OrthotropicMaterial(const MaterialProperties& props)
        : Material(MaterialType::Orthotropic, props) {
        compute_stiffness_matrix();
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        for (int i = 0; i < 6; ++i) {
            state.stress[i] = 0.0;
            for (int j = 0; j < 6; ++j) {
                state.stress[i] += C_[i * 6 + j] * state.strain[j];
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/,
                          Real* C) const override {
        for (int i = 0; i < 36; ++i) C[i] = C_[i];
    }

    static std::unique_ptr<Material> create(const MaterialProperties& props) {
        return std::make_unique<OrthotropicMaterial>(props);
    }

private:
    Real C_[36];  ///< Precomputed 6x6 stiffness matrix

    void compute_stiffness_matrix() {
        Real E1 = props_.E1 > 0.0 ? props_.E1 : props_.E;
        Real E2 = props_.E2 > 0.0 ? props_.E2 : props_.E;
        Real E3 = props_.E3 > 0.0 ? props_.E3 : props_.E;
        Real nu12 = props_.nu12;
        Real nu23 = props_.nu23;
        Real nu13 = props_.nu13;
        Real G12 = props_.G12 > 0.0 ? props_.G12 : props_.G;
        Real G23 = props_.G23 > 0.0 ? props_.G23 : props_.G;
        Real G13 = props_.G13 > 0.0 ? props_.G13 : props_.G;

        // Reciprocal relations
        Real nu21 = nu12 * E2 / E1;
        Real nu31 = nu13 * E3 / E1;
        Real nu32 = nu23 * E3 / E2;

        // Determinant of compliance submatrix
        Real delta = 1.0 - nu12 * nu21 - nu23 * nu32 - nu13 * nu31
                     - 2.0 * nu12 * nu23 * nu31;

        for (int i = 0; i < 36; ++i) C_[i] = 0.0;

        // Normal stiffness terms
        C_[0]  = E1 * (1.0 - nu23 * nu32) / delta;
        C_[1]  = E1 * (nu21 + nu31 * nu23) / delta;
        C_[2]  = E1 * (nu31 + nu21 * nu32) / delta;
        C_[6]  = C_[1];
        C_[7]  = E2 * (1.0 - nu13 * nu31) / delta;
        C_[8]  = E2 * (nu32 + nu12 * nu31) / delta;
        C_[12] = C_[2];
        C_[13] = C_[8];
        C_[14] = E3 * (1.0 - nu12 * nu21) / delta;

        // Shear terms
        C_[21] = G12;
        C_[28] = G23;
        C_[35] = G13;
    }
};

} // namespace physics
} // namespace nxs
