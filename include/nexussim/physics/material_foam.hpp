#pragma once

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

/**
 * @brief Blatz-Ko compressible foam (hyperelastic)
 *
 * Strain energy density:
 *   W = (G/2) * (I2/I3 + 2*sqrt(I3) - 5)
 *
 * Where I2, I3 are invariants of the right Cauchy-Green tensor.
 * Suitable for open-cell foams under moderate deformation.
 *
 * Properties used: G (shear modulus), K (bulk modulus)
 */
class FoamMaterial : public Material {
public:
    FoamMaterial(const MaterialProperties& props)
        : Material(MaterialType::Foam, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real G = props_.G;
        const Real* F = state.F;

        // J = det(F)
        Real J = F[0] * (F[4] * F[8] - F[5] * F[7])
               - F[1] * (F[3] * F[8] - F[5] * F[6])
               + F[2] * (F[3] * F[7] - F[4] * F[6]);

        if (J < 1.0e-10) {
            for (int i = 0; i < 6; ++i) state.stress[i] = 0.0;
            return;
        }

        // Left Cauchy-Green B = F*F^T (Voigt)
        Real B[6];
        B[0] = F[0]*F[0] + F[1]*F[1] + F[2]*F[2];
        B[1] = F[3]*F[3] + F[4]*F[4] + F[5]*F[5];
        B[2] = F[6]*F[6] + F[7]*F[7] + F[8]*F[8];
        B[3] = F[0]*F[3] + F[1]*F[4] + F[2]*F[5];
        B[4] = F[3]*F[6] + F[4]*F[7] + F[5]*F[8];
        B[5] = F[0]*F[6] + F[1]*F[7] + F[2]*F[8];

        Real I1 = B[0] + B[1] + B[2];
        Real J2 = J * J;

        // Blatz-Ko Cauchy stress:
        // sigma = (G/J) * (B - I3^(1/2) * I) where I3 = J^2
        Real sqrtI3 = J;
        Real inv_J = 1.0 / J;

        // Deviatoric + volumetric
        for (int i = 0; i < 3; ++i) {
            state.stress[i] = G * inv_J * (B[i] - sqrtI3);
        }
        for (int i = 3; i < 6; ++i) {
            state.stress[i] = G * inv_J * B[i];
        }

        state.vol_strain = Kokkos::log(J);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/, Real* C) const override {
        Real G = props_.G;
        Real K = props_.K;
        Real lambda = K - 2.0 * G / 3.0;
        Real lambda_2mu = lambda + 2.0 * G;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lambda_2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = G;
    }

    static std::unique_ptr<Material> create(const MaterialProperties& props) {
        return std::make_unique<FoamMaterial>(props);
    }
};

} // namespace physics
} // namespace nxs
