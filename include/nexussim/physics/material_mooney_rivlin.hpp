#pragma once

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

/**
 * @brief Mooney-Rivlin hyperelastic material
 *
 * Strain energy density:
 *   W = C10*(I1_bar - 3) + C01*(I2_bar - 3) + K/2*(J - 1)^2
 *
 * Where I1_bar, I2_bar are isochoric invariants of the left Cauchy-Green tensor.
 * Commonly used for rubber and elastomeric materials.
 *
 * Properties used: C10, C01, K (bulk modulus)
 * Initial shear modulus: mu = 2*(C10 + C01)
 */
class MooneyRivlinMaterial : public Material {
public:
    MooneyRivlinMaterial(const MaterialProperties& props)
        : Material(MaterialType::MooneyRivlin, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        compute_stress_large_deform(state);
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress_large_deform(MaterialState& state) const override {
        const Real c10 = props_.C10;
        const Real c01 = props_.C01;
        const Real kappa = props_.K;
        const Real* F = state.F;

        // Determinant J = det(F)
        Real J = F[0] * (F[4] * F[8] - F[5] * F[7])
               - F[1] * (F[3] * F[8] - F[5] * F[6])
               + F[2] * (F[3] * F[7] - F[4] * F[6]);

        if (J < 1.0e-10) {
            for (int i = 0; i < 6; ++i) state.stress[i] = 0.0;
            return;
        }

        // Left Cauchy-Green B = F*F^T
        Real B[6]; // Voigt: Bxx, Byy, Bzz, Bxy, Byz, Bxz
        B[0] = F[0]*F[0] + F[1]*F[1] + F[2]*F[2];
        B[1] = F[3]*F[3] + F[4]*F[4] + F[5]*F[5];
        B[2] = F[6]*F[6] + F[7]*F[7] + F[8]*F[8];
        B[3] = F[0]*F[3] + F[1]*F[4] + F[2]*F[5];
        B[4] = F[3]*F[6] + F[4]*F[7] + F[5]*F[8];
        B[5] = F[0]*F[6] + F[1]*F[7] + F[2]*F[8];

        Real I1 = B[0] + B[1] + B[2];

        // B^2 diagonal and off-diagonal (for I2 computation)
        // I2 = 0.5*(I1^2 - tr(B^2))
        Real trB2 = B[0]*B[0] + B[1]*B[1] + B[2]*B[2]
                  + 2.0*(B[3]*B[3] + B[4]*B[4] + B[5]*B[5]);
        Real I2 = 0.5 * (I1 * I1 - trB2);

        // Isochoric invariants
        Real J23 = Kokkos::pow(J, -2.0/3.0);
        Real J43 = J23 * J23;
        (void)I2; // I2_bar used implicitly through c01 term

        // Cauchy stress for Mooney-Rivlin:
        // sigma = (2/J) * [ (c10 + c01*I1_bar)*B_dev - c01*B^2_dev ] + kappa*(J-1)*I
        // Simplified for nearly-incompressible form:
        Real inv_J = 1.0 / J;
        Real p = kappa * (J - 1.0);

        // Deviatoric B
        Real I1_3 = I1 / 3.0;
        Real Bd[6];
        Bd[0] = B[0] - I1_3;
        Bd[1] = B[1] - I1_3;
        Bd[2] = B[2] - I1_3;
        Bd[3] = B[3];
        Bd[4] = B[4];
        Bd[5] = B[5];

        // For C01 term, compute B^2 deviatoric
        Real B2[6];
        B2[0] = B[0]*B[0] + B[3]*B[3] + B[5]*B[5];
        B2[1] = B[3]*B[3] + B[1]*B[1] + B[4]*B[4];
        B2[2] = B[5]*B[5] + B[4]*B[4] + B[2]*B[2];
        B2[3] = B[0]*B[3] + B[3]*B[1] + B[5]*B[4];
        B2[4] = B[3]*B[5] + B[1]*B[4] + B[4]*B[2];
        B2[5] = B[0]*B[5] + B[3]*B[4] + B[5]*B[2];

        Real trB2_3 = trB2 / 3.0;
        Real B2d[6];
        B2d[0] = B2[0] - trB2_3;
        B2d[1] = B2[1] - trB2_3;
        B2d[2] = B2[2] - trB2_3;
        B2d[3] = B2[3];
        B2d[4] = B2[4];
        B2d[5] = B2[5];

        Real coeff1 = 2.0 * inv_J * J23 * c10;
        Real coeff2 = 2.0 * inv_J * J43 * c01;

        // Stress = coeff1*Bd + coeff2*(I1*Bd - B2d) + p*I
        // Simplified: use c10 on Bd and c01 on (I1*Bd - B2d)
        for (int i = 0; i < 3; ++i) {
            state.stress[i] = coeff1 * Bd[i] + coeff2 * (I1 * Bd[i] - B2d[i]) + p;
        }
        for (int i = 3; i < 6; ++i) {
            state.stress[i] = coeff1 * Bd[i] + coeff2 * (I1 * Bd[i] - B2d[i]);
        }

        state.vol_strain = Kokkos::log(J);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/,
                          Real* C) const override {
        // Use initial elastic approximation: mu = 2*(C10 + C01)
        Real mu = 2.0 * (props_.C10 + props_.C01);
        Real K = props_.K;
        Real lambda = K - 2.0 * mu / 3.0;
        Real lambda_2mu = lambda + 2.0 * mu;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lambda_2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

    static std::unique_ptr<Material> create(const MaterialProperties& props) {
        return std::make_unique<MooneyRivlinMaterial>(props);
    }
};

} // namespace physics
} // namespace nxs
