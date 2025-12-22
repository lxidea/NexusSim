#pragma once

/**
 * @file large_deformation.hpp
 * @brief Large deformation mechanics - Updated and Total Lagrangian formulations
 *
 * This module provides:
 * - Deformation gradient computation (F = I + ∂u/∂X)
 * - Strain measures (Green-Lagrange, Euler-Almansi, logarithmic)
 * - Stress transformations (Cauchy, 1st/2nd Piola-Kirchhoff)
 * - Updated Lagrangian formulation (geometry update each step)
 * - Total Lagrangian formulation (reference configuration fixed)
 *
 * Reference: Belytschko et al., "Nonlinear Finite Elements for Continua and Structures"
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <cmath>

namespace nxs {
namespace physics {

// ============================================================================
// Formulation Types
// ============================================================================

enum class LagrangianFormulation {
    TotalLagrangian,    ///< Reference config fixed (material description)
    UpdatedLagrangian   ///< Reference updated each step (spatial description)
};

enum class StrainMeasure {
    SmallStrain,        ///< Linear ε = 0.5(∇u + ∇u^T)
    GreenLagrange,      ///< E = 0.5(F^T·F - I)
    EulerAlmansi,       ///< e = 0.5(I - F^{-T}·F^{-1})
    Logarithmic,        ///< ε_log = 0.5*ln(F^T·F) (Hencky)
    RateOfDeformation   ///< D = 0.5(L + L^T) where L = Ḟ·F^{-1}
};

// ============================================================================
// Tensor Utilities (GPU-compatible)
// ============================================================================

namespace tensor {

/**
 * @brief Compute 3x3 matrix determinant
 */
KOKKOS_INLINE_FUNCTION
Real determinant3x3(const Real* A) {
    return A[0]*(A[4]*A[8] - A[5]*A[7])
         - A[1]*(A[3]*A[8] - A[5]*A[6])
         + A[2]*(A[3]*A[7] - A[4]*A[6]);
}

/**
 * @brief Compute 3x3 matrix inverse
 * @return Determinant
 */
KOKKOS_INLINE_FUNCTION
Real inverse3x3(const Real* A, Real* A_inv) {
    const Real det = determinant3x3(A);
    if (Kokkos::fabs(det) < 1.0e-30) {
        // Return identity for singular matrix
        for (int i = 0; i < 9; ++i) A_inv[i] = (i == 0 || i == 4 || i == 8) ? 1.0 : 0.0;
        return 0.0;
    }

    const Real inv_det = 1.0 / det;

    A_inv[0] = (A[4]*A[8] - A[5]*A[7]) * inv_det;
    A_inv[1] = (A[2]*A[7] - A[1]*A[8]) * inv_det;
    A_inv[2] = (A[1]*A[5] - A[2]*A[4]) * inv_det;
    A_inv[3] = (A[5]*A[6] - A[3]*A[8]) * inv_det;
    A_inv[4] = (A[0]*A[8] - A[2]*A[6]) * inv_det;
    A_inv[5] = (A[2]*A[3] - A[0]*A[5]) * inv_det;
    A_inv[6] = (A[3]*A[7] - A[4]*A[6]) * inv_det;
    A_inv[7] = (A[1]*A[6] - A[0]*A[7]) * inv_det;
    A_inv[8] = (A[0]*A[4] - A[1]*A[3]) * inv_det;

    return det;
}

/**
 * @brief Matrix transpose (3x3)
 */
KOKKOS_INLINE_FUNCTION
void transpose3x3(const Real* A, Real* A_T) {
    A_T[0] = A[0]; A_T[1] = A[3]; A_T[2] = A[6];
    A_T[3] = A[1]; A_T[4] = A[4]; A_T[5] = A[7];
    A_T[6] = A[2]; A_T[7] = A[5]; A_T[8] = A[8];
}

/**
 * @brief Matrix multiplication C = A * B (3x3)
 */
KOKKOS_INLINE_FUNCTION
void multiply3x3(const Real* A, const Real* B, Real* C) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            C[i*3 + j] = A[i*3 + 0]*B[0*3 + j]
                       + A[i*3 + 1]*B[1*3 + j]
                       + A[i*3 + 2]*B[2*3 + j];
        }
    }
}

/**
 * @brief Matrix trace
 */
KOKKOS_INLINE_FUNCTION
Real trace3x3(const Real* A) {
    return A[0] + A[4] + A[8];
}

/**
 * @brief Convert 3x3 symmetric tensor to Voigt notation
 * A -> [A11, A22, A33, A12, A23, A13]
 */
KOKKOS_INLINE_FUNCTION
void symmetric_to_voigt(const Real* A, Real* v) {
    v[0] = A[0];  // A11
    v[1] = A[4];  // A22
    v[2] = A[8];  // A33
    v[3] = A[1];  // A12 = A21
    v[4] = A[5];  // A23 = A32
    v[5] = A[2];  // A13 = A31
}

/**
 * @brief Convert Voigt notation to 3x3 symmetric tensor
 */
KOKKOS_INLINE_FUNCTION
void voigt_to_symmetric(const Real* v, Real* A) {
    A[0] = v[0]; A[1] = v[3]; A[2] = v[5];
    A[3] = v[3]; A[4] = v[1]; A[5] = v[4];
    A[6] = v[5]; A[7] = v[4]; A[8] = v[2];
}

} // namespace tensor

// ============================================================================
// Deformation Gradient Computation
// ============================================================================

/**
 * @brief Compute deformation gradient F = I + ∂u/∂X
 *
 * For Updated Lagrangian: F_n+1 = I + ∂Δu/∂x_n (incremental)
 * For Total Lagrangian: F = I + ∂u/∂X (total)
 *
 * @param dN_dX Shape function derivatives w.r.t. reference coords [num_nodes x 3]
 * @param disp Nodal displacements [num_nodes x 3]
 * @param num_nodes Number of nodes
 * @param F Output: Deformation gradient [3x3, row-major]
 */
KOKKOS_INLINE_FUNCTION
void compute_deformation_gradient(const Real* dN_dX, const Real* disp,
                                   int num_nodes, Real* F) {
    // Initialize to identity
    F[0] = 1.0; F[1] = 0.0; F[2] = 0.0;
    F[3] = 0.0; F[4] = 1.0; F[5] = 0.0;
    F[6] = 0.0; F[7] = 0.0; F[8] = 1.0;

    // Add displacement gradient: F_ij = δ_ij + ∂u_i/∂X_j
    for (int n = 0; n < num_nodes; ++n) {
        const Real ux = disp[n*3 + 0];
        const Real uy = disp[n*3 + 1];
        const Real uz = disp[n*3 + 2];

        // dN/dX, dN/dY, dN/dZ for this node
        const Real dNdX = dN_dX[n*3 + 0];
        const Real dNdY = dN_dX[n*3 + 1];
        const Real dNdZ = dN_dX[n*3 + 2];

        // ∂u/∂X
        F[0] += ux * dNdX;  // ∂ux/∂X
        F[1] += ux * dNdY;  // ∂ux/∂Y
        F[2] += ux * dNdZ;  // ∂ux/∂Z

        F[3] += uy * dNdX;  // ∂uy/∂X
        F[4] += uy * dNdY;  // ∂uy/∂Y
        F[5] += uy * dNdZ;  // ∂uy/∂Z

        F[6] += uz * dNdX;  // ∂uz/∂X
        F[7] += uz * dNdY;  // ∂uz/∂Y
        F[8] += uz * dNdZ;  // ∂uz/∂Z
    }
}

/**
 * @brief Compute incremental deformation gradient for Updated Lagrangian
 * F_inc = I + ∂Δu/∂x where x is current config
 */
KOKKOS_INLINE_FUNCTION
void compute_incremental_deformation_gradient(
    const Real* dN_dx, const Real* delta_disp,
    int num_nodes, Real* F_inc) {
    compute_deformation_gradient(dN_dx, delta_disp, num_nodes, F_inc);
}

// ============================================================================
// Strain Measures
// ============================================================================

/**
 * @brief Compute Green-Lagrange strain: E = 0.5*(F^T·F - I)
 * @param F Deformation gradient [3x3]
 * @param E Output: Green-Lagrange strain [6, Voigt]
 */
KOKKOS_INLINE_FUNCTION
void green_lagrange_strain(const Real* F, Real* E) {
    // Compute right Cauchy-Green tensor C = F^T · F
    Real C[9];
    C[0] = F[0]*F[0] + F[3]*F[3] + F[6]*F[6];  // C11
    C[1] = F[0]*F[1] + F[3]*F[4] + F[6]*F[7];  // C12
    C[2] = F[0]*F[2] + F[3]*F[5] + F[6]*F[8];  // C13
    C[3] = C[1];                                // C21
    C[4] = F[1]*F[1] + F[4]*F[4] + F[7]*F[7];  // C22
    C[5] = F[1]*F[2] + F[4]*F[5] + F[7]*F[8];  // C23
    C[6] = C[2];                                // C31
    C[7] = C[5];                                // C32
    C[8] = F[2]*F[2] + F[5]*F[5] + F[8]*F[8];  // C33

    // E = 0.5*(C - I) in Voigt notation
    E[0] = 0.5 * (C[0] - 1.0);  // E11
    E[1] = 0.5 * (C[4] - 1.0);  // E22
    E[2] = 0.5 * (C[8] - 1.0);  // E33
    E[3] = C[1];                 // 2*E12 (engineering shear)
    E[4] = C[5];                 // 2*E23
    E[5] = C[2];                 // 2*E13
}

/**
 * @brief Compute Euler-Almansi strain: e = 0.5*(I - F^{-T}·F^{-1})
 * @param F Deformation gradient [3x3]
 * @param e Output: Euler-Almansi strain [6, Voigt]
 */
KOKKOS_INLINE_FUNCTION
void euler_almansi_strain(const Real* F, Real* e) {
    // Compute F inverse
    Real F_inv[9];
    tensor::inverse3x3(F, F_inv);

    // Compute b^{-1} = F^{-T} · F^{-1}
    Real F_inv_T[9];
    tensor::transpose3x3(F_inv, F_inv_T);

    Real b_inv[9];
    tensor::multiply3x3(F_inv_T, F_inv, b_inv);

    // e = 0.5*(I - b^{-1})
    e[0] = 0.5 * (1.0 - b_inv[0]);
    e[1] = 0.5 * (1.0 - b_inv[4]);
    e[2] = 0.5 * (1.0 - b_inv[8]);
    e[3] = -b_inv[1];  // 2*e12
    e[4] = -b_inv[5];  // 2*e23
    e[5] = -b_inv[2];  // 2*e13
}

/**
 * @brief Compute rate of deformation tensor: D = 0.5*(L + L^T)
 * where L = Ḟ·F^{-1} is the velocity gradient
 *
 * @param dN_dx Shape function derivatives w.r.t. current coords
 * @param velocity Nodal velocities [num_nodes x 3]
 * @param num_nodes Number of nodes
 * @param D Output: Rate of deformation [6, Voigt]
 */
KOKKOS_INLINE_FUNCTION
void rate_of_deformation(const Real* dN_dx, const Real* velocity,
                          int num_nodes, Real* D) {
    // Compute velocity gradient L_ij = ∂v_i/∂x_j
    Real L[9] = {0};

    for (int n = 0; n < num_nodes; ++n) {
        const Real vx = velocity[n*3 + 0];
        const Real vy = velocity[n*3 + 1];
        const Real vz = velocity[n*3 + 2];

        const Real dNdx = dN_dx[n*3 + 0];
        const Real dNdy = dN_dx[n*3 + 1];
        const Real dNdz = dN_dx[n*3 + 2];

        L[0] += vx * dNdx;  L[1] += vx * dNdy;  L[2] += vx * dNdz;
        L[3] += vy * dNdx;  L[4] += vy * dNdy;  L[5] += vy * dNdz;
        L[6] += vz * dNdx;  L[7] += vz * dNdy;  L[8] += vz * dNdz;
    }

    // D = 0.5*(L + L^T) in Voigt notation
    D[0] = L[0];                  // D11
    D[1] = L[4];                  // D22
    D[2] = L[8];                  // D33
    D[3] = L[1] + L[3];           // 2*D12 = L12 + L21
    D[4] = L[5] + L[7];           // 2*D23 = L23 + L32
    D[5] = L[2] + L[6];           // 2*D13 = L13 + L31
}

/**
 * @brief Compute spin tensor: W = 0.5*(L - L^T)
 * For objective stress rates (Jaumann, Green-Naghdi)
 */
KOKKOS_INLINE_FUNCTION
void spin_tensor(const Real* dN_dx, const Real* velocity,
                  int num_nodes, Real* W) {
    // Compute velocity gradient
    Real L[9] = {0};

    for (int n = 0; n < num_nodes; ++n) {
        const Real vx = velocity[n*3 + 0];
        const Real vy = velocity[n*3 + 1];
        const Real vz = velocity[n*3 + 2];

        const Real dNdx = dN_dx[n*3 + 0];
        const Real dNdy = dN_dx[n*3 + 1];
        const Real dNdz = dN_dx[n*3 + 2];

        L[0] += vx * dNdx;  L[1] += vx * dNdy;  L[2] += vx * dNdz;
        L[3] += vy * dNdx;  L[4] += vy * dNdy;  L[5] += vy * dNdz;
        L[6] += vz * dNdx;  L[7] += vz * dNdy;  L[8] += vz * dNdz;
    }

    // W = 0.5*(L - L^T) (antisymmetric)
    W[0] = 0.0;
    W[1] = 0.5 * (L[1] - L[3]);  // W12
    W[2] = 0.5 * (L[2] - L[6]);  // W13
    W[3] = -W[1];                 // W21 = -W12
    W[4] = 0.0;
    W[5] = 0.5 * (L[5] - L[7]);  // W23
    W[6] = -W[2];                 // W31 = -W13
    W[7] = -W[5];                 // W32 = -W23
    W[8] = 0.0;
}

// ============================================================================
// Stress Transformations
// ============================================================================

/**
 * @brief Convert Cauchy stress to 1st Piola-Kirchhoff stress
 * P = J * σ · F^{-T}
 *
 * @param sigma Cauchy stress [6, Voigt]
 * @param F Deformation gradient [3x3]
 * @param P Output: 1st PK stress [3x3, NOT symmetric]
 */
KOKKOS_INLINE_FUNCTION
void cauchy_to_first_pk(const Real* sigma, const Real* F, Real* P) {
    Real J = tensor::determinant3x3(F);

    // Convert Voigt to matrix
    Real sigma_mat[9];
    tensor::voigt_to_symmetric(sigma, sigma_mat);

    // Compute F^{-T}
    Real F_inv[9], F_inv_T[9];
    tensor::inverse3x3(F, F_inv);
    tensor::transpose3x3(F_inv, F_inv_T);

    // P = J * σ · F^{-T}
    Real tmp[9];
    tensor::multiply3x3(sigma_mat, F_inv_T, tmp);
    for (int i = 0; i < 9; ++i) {
        P[i] = J * tmp[i];
    }
}

/**
 * @brief Convert Cauchy stress to 2nd Piola-Kirchhoff stress
 * S = J * F^{-1} · σ · F^{-T}
 *
 * @param sigma Cauchy stress [6, Voigt]
 * @param F Deformation gradient [3x3]
 * @param S Output: 2nd PK stress [6, Voigt] (symmetric)
 */
KOKKOS_INLINE_FUNCTION
void cauchy_to_second_pk(const Real* sigma, const Real* F, Real* S) {
    Real J = tensor::determinant3x3(F);

    Real sigma_mat[9];
    tensor::voigt_to_symmetric(sigma, sigma_mat);

    Real F_inv[9], F_inv_T[9];
    tensor::inverse3x3(F, F_inv);
    tensor::transpose3x3(F_inv, F_inv_T);

    // S = J * F^{-1} · σ · F^{-T}
    Real tmp[9], S_mat[9];
    tensor::multiply3x3(sigma_mat, F_inv_T, tmp);
    tensor::multiply3x3(F_inv, tmp, S_mat);

    for (int i = 0; i < 9; ++i) {
        S_mat[i] *= J;
    }

    tensor::symmetric_to_voigt(S_mat, S);
}

/**
 * @brief Convert 2nd Piola-Kirchhoff stress to Cauchy stress
 * σ = (1/J) * F · S · F^T
 *
 * @param S 2nd PK stress [6, Voigt]
 * @param F Deformation gradient [3x3]
 * @param sigma Output: Cauchy stress [6, Voigt]
 */
KOKKOS_INLINE_FUNCTION
void second_pk_to_cauchy(const Real* S, const Real* F, Real* sigma) {
    Real J = tensor::determinant3x3(F);

    Real S_mat[9];
    tensor::voigt_to_symmetric(S, S_mat);

    Real F_T[9];
    tensor::transpose3x3(F, F_T);

    // σ = (1/J) * F · S · F^T
    Real tmp[9], sigma_mat[9];
    tensor::multiply3x3(S_mat, F_T, tmp);
    tensor::multiply3x3(F, tmp, sigma_mat);

    Real inv_J = 1.0 / J;
    for (int i = 0; i < 9; ++i) {
        sigma_mat[i] *= inv_J;
    }

    tensor::symmetric_to_voigt(sigma_mat, sigma);
}

// ============================================================================
// Internal Force Computation (Large Deformation)
// ============================================================================

/**
 * @brief Compute internal forces using Total Lagrangian formulation
 *
 * f_int = ∫ B_0^T · S · dV_0
 *
 * where B_0 is the B-matrix in reference config and S is 2nd PK stress
 *
 * @param dN_dX Shape function derivatives w.r.t. reference coords [num_nodes x 3]
 * @param S 2nd Piola-Kirchhoff stress [6, Voigt]
 * @param F Deformation gradient [3x3]
 * @param detJ0 Reference Jacobian determinant
 * @param weight Gauss weight
 * @param num_nodes Number of nodes
 * @param f_int Output: Internal force contribution [num_nodes x 3]
 */
KOKKOS_INLINE_FUNCTION
void internal_force_total_lagrangian(
    const Real* dN_dX, const Real* S, const Real* F,
    Real detJ0, Real weight, int num_nodes, Real* f_int) {

    // For Total Lagrangian: f = ∫ B_NL^T · S · dV_0
    // where B_NL incorporates geometric nonlinearity

    // Compute 1st Piola-Kirchhoff stress P = F · S
    Real S_mat[9], P[9];
    tensor::voigt_to_symmetric(S, S_mat);
    tensor::multiply3x3(F, S_mat, P);

    // f_I = ∫ P · (∂N_I/∂X)^T dV_0
    Real dV = detJ0 * weight;

    for (int n = 0; n < num_nodes; ++n) {
        const Real dNdX = dN_dX[n*3 + 0];
        const Real dNdY = dN_dX[n*3 + 1];
        const Real dNdZ = dN_dX[n*3 + 2];

        // f_x = P_11*dN/dX + P_12*dN/dY + P_13*dN/dZ
        f_int[n*3 + 0] += (P[0]*dNdX + P[1]*dNdY + P[2]*dNdZ) * dV;
        f_int[n*3 + 1] += (P[3]*dNdX + P[4]*dNdY + P[5]*dNdZ) * dV;
        f_int[n*3 + 2] += (P[6]*dNdX + P[7]*dNdY + P[8]*dNdZ) * dV;
    }
}

/**
 * @brief Compute internal forces using Updated Lagrangian formulation
 *
 * f_int = ∫ B^T · σ · dv
 *
 * where B is B-matrix in current config and σ is Cauchy stress
 *
 * @param dN_dx Shape function derivatives w.r.t. current coords [num_nodes x 3]
 * @param sigma Cauchy stress [6, Voigt]
 * @param detj Current Jacobian determinant
 * @param weight Gauss weight
 * @param num_nodes Number of nodes
 * @param f_int Output: Internal force contribution [num_nodes x 3]
 */
KOKKOS_INLINE_FUNCTION
void internal_force_updated_lagrangian(
    const Real* dN_dx, const Real* sigma,
    Real detj, Real weight, int num_nodes, Real* f_int) {

    Real dv = detj * weight;

    // Stress components (Voigt: σxx, σyy, σzz, τxy, τyz, τxz)
    const Real sxx = sigma[0];
    const Real syy = sigma[1];
    const Real szz = sigma[2];
    const Real sxy = sigma[3];
    const Real syz = sigma[4];
    const Real sxz = sigma[5];

    for (int n = 0; n < num_nodes; ++n) {
        const Real dNdx = dN_dx[n*3 + 0];
        const Real dNdy = dN_dx[n*3 + 1];
        const Real dNdz = dN_dx[n*3 + 2];

        // f = B^T · σ
        f_int[n*3 + 0] += (sxx*dNdx + sxy*dNdy + sxz*dNdz) * dv;
        f_int[n*3 + 1] += (sxy*dNdx + syy*dNdy + syz*dNdz) * dv;
        f_int[n*3 + 2] += (sxz*dNdx + syz*dNdy + szz*dNdz) * dv;
    }
}

// ============================================================================
// Geometry Update (Updated Lagrangian)
// ============================================================================

/**
 * @brief Update nodal coordinates for Updated Lagrangian formulation
 *
 * x_new = x_old + Δu
 *
 * @param coords Current coordinates [num_nodes x 3]
 * @param delta_disp Displacement increment [num_nodes x 3]
 * @param num_nodes Number of nodes
 * @param coords_new Output: Updated coordinates [num_nodes x 3]
 */
KOKKOS_INLINE_FUNCTION
void update_geometry(const Real* coords, const Real* delta_disp,
                      int num_nodes, Real* coords_new) {
    for (int i = 0; i < num_nodes * 3; ++i) {
        coords_new[i] = coords[i] + delta_disp[i];
    }
}

// ============================================================================
// Objective Stress Rates
// ============================================================================

/**
 * @brief Jaumann rate of Cauchy stress
 *
 * σ̊^J = σ̇ - W·σ + σ·W
 *
 * Used for hypoelastic materials in Updated Lagrangian formulation
 *
 * @param sigma_dot Stress rate (material derivative) [6, Voigt]
 * @param sigma Current stress [6, Voigt]
 * @param W Spin tensor [3x3]
 * @param sigma_jaumann Output: Jaumann stress rate [6, Voigt]
 */
KOKKOS_INLINE_FUNCTION
void jaumann_stress_rate(const Real* sigma_dot, const Real* sigma,
                          const Real* W, Real* sigma_jaumann) {
    // Convert to matrix form
    Real sigma_mat[9];
    tensor::voigt_to_symmetric(sigma, sigma_mat);

    // Compute W·σ and σ·W
    Real W_sigma[9], sigma_W[9];
    tensor::multiply3x3(W, sigma_mat, W_sigma);
    tensor::multiply3x3(sigma_mat, W, sigma_W);

    // σ̊ = σ̇ - W·σ + σ·W
    Real result_mat[9];
    for (int i = 0; i < 9; ++i) {
        result_mat[i] = -W_sigma[i] + sigma_W[i];
    }

    // Add σ̇ (input as Voigt)
    Real sigma_dot_mat[9];
    tensor::voigt_to_symmetric(sigma_dot, sigma_dot_mat);
    for (int i = 0; i < 9; ++i) {
        result_mat[i] += sigma_dot_mat[i];
    }

    tensor::symmetric_to_voigt(result_mat, sigma_jaumann);
}

/**
 * @brief Inverse Jaumann rate: update stress from rate
 *
 * σ_{n+1} = σ_n + Δt * (σ̊^J + W·σ - σ·W)
 *
 * @param sigma Current stress [6, Voigt]
 * @param sigma_rate_objective Objective stress rate [6, Voigt]
 * @param W Spin tensor [3x3]
 * @param dt Time step
 * @param sigma_new Output: Updated stress [6, Voigt]
 */
KOKKOS_INLINE_FUNCTION
void update_stress_jaumann(const Real* sigma, const Real* sigma_rate_objective,
                            const Real* W, Real dt, Real* sigma_new) {
    Real sigma_mat[9];
    tensor::voigt_to_symmetric(sigma, sigma_mat);

    // W·σ and σ·W
    Real W_sigma[9], sigma_W[9];
    tensor::multiply3x3(W, sigma_mat, W_sigma);
    tensor::multiply3x3(sigma_mat, W, sigma_W);

    // Material rate = objective rate + W·σ - σ·W
    Real sigma_dot_mat[9];
    tensor::voigt_to_symmetric(sigma_rate_objective, sigma_dot_mat);

    Real sigma_new_mat[9];
    for (int i = 0; i < 9; ++i) {
        Real sigma_dot = sigma_dot_mat[i] + W_sigma[i] - sigma_W[i];
        sigma_new_mat[i] = sigma_mat[i] + dt * sigma_dot;
    }

    tensor::symmetric_to_voigt(sigma_new_mat, sigma_new);
}

// ============================================================================
// Updated Lagrangian Element Processor
// ============================================================================

/**
 * @brief Configuration for large deformation analysis
 */
struct LargeDeformConfig {
    LagrangianFormulation formulation;
    StrainMeasure strain_measure;
    bool use_jaumann_rate;      ///< Use Jaumann objective rate
    bool geometry_update;        ///< Update geometry each step (UL)
    Real min_jacobian;          ///< Minimum allowed Jacobian (element inversion check)

    LargeDeformConfig()
        : formulation(LagrangianFormulation::UpdatedLagrangian)
        , strain_measure(StrainMeasure::RateOfDeformation)
        , use_jaumann_rate(true)
        , geometry_update(true)
        , min_jacobian(0.01)
    {}
};

/**
 * @brief Check for element inversion
 * @return true if element is valid (J > min_jacobian)
 */
KOKKOS_INLINE_FUNCTION
bool check_element_valid(const Real* F, Real min_jacobian) {
    Real J = tensor::determinant3x3(F);
    return J > min_jacobian;
}

/**
 * @brief Compute volume ratio from deformation gradient
 */
KOKKOS_INLINE_FUNCTION
Real volume_ratio(const Real* F) {
    return tensor::determinant3x3(F);
}

} // namespace physics
} // namespace nxs
