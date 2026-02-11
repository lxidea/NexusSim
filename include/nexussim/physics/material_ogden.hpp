#pragma once

#include <nexussim/physics/material.hpp>

namespace nxs {
namespace physics {

/**
 * @brief Ogden hyperelastic material model
 *
 * Strain energy density (isochoric part):
 *   W = sum_k { mu_k/alpha_k * (lambda1^alpha_k + lambda2^alpha_k + lambda3^alpha_k - 3) }
 *     + K/2 * (J - 1)^2
 *
 * Where lambda_i are the principal stretches of the isochoric deformation.
 * Supports up to 3 terms. Common for rubber with large deformations.
 *
 * Properties used: ogden_mu[3], ogden_alpha[3], ogden_nterms, K
 */
class OgdenMaterial : public Material {
public:
    OgdenMaterial(const MaterialProperties& props)
        : Material(MaterialType::Ogden, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        compute_stress_large_deform(state);
    }

    KOKKOS_INLINE_FUNCTION
    void compute_stress_large_deform(MaterialState& state) const override {
        const Real kappa = props_.K;
        const Real* F = state.F;

        // J = det(F)
        Real J = F[0] * (F[4] * F[8] - F[5] * F[7])
               - F[1] * (F[3] * F[8] - F[5] * F[6])
               + F[2] * (F[3] * F[7] - F[4] * F[6]);

        if (J < 1.0e-10) {
            for (int i = 0; i < 6; ++i) state.stress[i] = 0.0;
            return;
        }

        // Right Cauchy-Green C = F^T * F
        Real C[9];
        C[0] = F[0]*F[0] + F[3]*F[3] + F[6]*F[6];
        C[1] = F[0]*F[1] + F[3]*F[4] + F[6]*F[7];
        C[2] = F[0]*F[2] + F[3]*F[5] + F[6]*F[8];
        C[3] = C[1];
        C[4] = F[1]*F[1] + F[4]*F[4] + F[7]*F[7];
        C[5] = F[1]*F[2] + F[4]*F[5] + F[7]*F[8];
        C[6] = C[2];
        C[7] = C[5];
        C[8] = F[2]*F[2] + F[5]*F[5] + F[8]*F[8];

        // Principal stretches via eigenvalues of C
        // For 3D, compute eigenvalues of symmetric 3x3 matrix
        Real lambda_sq[3];
        symmetric_eigenvalues_3x3(C, lambda_sq);

        Real J_m13 = Kokkos::pow(J, -1.0/3.0);
        Real lambda_bar[3];
        for (int i = 0; i < 3; ++i) {
            lambda_bar[i] = J_m13 * Kokkos::sqrt(Kokkos::fmax(lambda_sq[i], 1.0e-20));
        }

        // Principal Cauchy stresses from Ogden model
        Real sigma_princ[3] = {0.0, 0.0, 0.0};
        int nt = props_.ogden_nterms > 0 ? props_.ogden_nterms : 1;
        if (nt > 3) nt = 3;

        for (int k = 0; k < nt; ++k) {
            Real mu_k = props_.ogden_mu[k];
            Real alpha_k = props_.ogden_alpha[k];
            if (Kokkos::fabs(alpha_k) < 1.0e-12) continue;

            for (int i = 0; i < 3; ++i) {
                Real lb_a = Kokkos::pow(lambda_bar[i], alpha_k);
                Real avg = 0.0;
                for (int j = 0; j < 3; ++j) {
                    avg += Kokkos::pow(lambda_bar[j], alpha_k);
                }
                avg /= 3.0;
                sigma_princ[i] += mu_k / J * (lb_a - avg);
            }
        }

        // Add volumetric: p = K*(J - 1)
        Real p = kappa * (J - 1.0);
        for (int i = 0; i < 3; ++i) sigma_princ[i] += p;

        // For simplicity, assume principal directions aligned with coordinate axes
        // (accurate for small rotations; full rotation tracking would need eigenvectors)
        state.stress[0] = sigma_princ[0];
        state.stress[1] = sigma_princ[1];
        state.stress[2] = sigma_princ[2];
        state.stress[3] = 0.0;
        state.stress[4] = 0.0;
        state.stress[5] = 0.0;

        state.vol_strain = Kokkos::log(J);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& /*state*/,
                          Real* C) const override {
        // Initial shear modulus: mu = sum(mu_k * alpha_k) / 2
        Real mu = 0.0;
        int nt = props_.ogden_nterms > 0 ? props_.ogden_nterms : 1;
        if (nt > 3) nt = 3;
        for (int k = 0; k < nt; ++k) {
            mu += props_.ogden_mu[k] * props_.ogden_alpha[k];
        }
        mu *= 0.5;
        if (mu < 1.0e-6) mu = props_.G;

        Real K = props_.K;
        Real lambda = K - 2.0 * mu / 3.0;
        Real lambda_2mu = lambda + 2.0 * mu;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;
        C[0] = C[7] = C[14] = lambda_2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }

    static std::unique_ptr<Material> create(const MaterialProperties& props) {
        return std::make_unique<OgdenMaterial>(props);
    }

private:
    /// Compute eigenvalues of a symmetric 3x3 matrix (Cardano's formula)
    KOKKOS_INLINE_FUNCTION
    static void symmetric_eigenvalues_3x3(const Real* A, Real* eigenvalues) {
        // A is 3x3 row-major symmetric
        Real a11 = A[0], a22 = A[4], a33 = A[8];
        Real a12 = A[1], a13 = A[2], a23 = A[5];

        Real p1 = a12*a12 + a13*a13 + a23*a23;
        if (p1 < 1.0e-30) {
            // Already diagonal
            eigenvalues[0] = a11;
            eigenvalues[1] = a22;
            eigenvalues[2] = a33;
            return;
        }

        Real q = (a11 + a22 + a33) / 3.0;
        Real p2 = (a11 - q)*(a11 - q) + (a22 - q)*(a22 - q) + (a33 - q)*(a33 - q)
                 + 2.0 * p1;
        Real p = Kokkos::sqrt(p2 / 6.0);
        Real inv_p = 1.0 / (p + 1.0e-30);

        // B = (1/p) * (A - q*I)
        Real b11 = inv_p * (a11 - q);
        Real b22 = inv_p * (a22 - q);
        Real b33 = inv_p * (a33 - q);
        Real b12 = inv_p * a12;
        Real b13 = inv_p * a13;
        Real b23 = inv_p * a23;

        Real detB = b11*(b22*b33 - b23*b23) - b12*(b12*b33 - b23*b13) + b13*(b12*b23 - b22*b13);
        Real r = detB * 0.5;
        r = Kokkos::fmax(-1.0, Kokkos::fmin(1.0, r));

        Real phi = Kokkos::acos(r) / 3.0;
        Real two_p = 2.0 * p;

        eigenvalues[0] = q + two_p * Kokkos::cos(phi);
        eigenvalues[2] = q + two_p * Kokkos::cos(phi + 2.0 * M_PI / 3.0);
        eigenvalues[1] = 3.0 * q - eigenvalues[0] - eigenvalues[2];
    }
};

} // namespace physics
} // namespace nxs
