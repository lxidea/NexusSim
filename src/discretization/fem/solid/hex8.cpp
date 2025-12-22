/**
 * @file hex8.cpp
 * @brief Implementation of 8-node hexahedral element
 */

#include <nexussim/discretization/hex8.hpp>
#include <cmath>
#include <algorithm>

namespace nxs {
namespace fem {

// ============================================================================
// Shape Functions
// ============================================================================

KOKKOS_INLINE_FUNCTION
void Hex8Element::shape_functions(const Real xi[3], Real* N) const {
    const Real xi_val = xi[0];
    const Real eta = xi[1];
    const Real zeta = xi[2];

    // Trilinear shape functions
    // N_i = (1 + ξ_i*ξ)(1 + η_i*η)(1 + ζ_i*ζ) / 8
    N[0] = 0.125 * (1.0 - xi_val) * (1.0 - eta) * (1.0 - zeta);  // Node 0: (-1,-1,-1)
    N[1] = 0.125 * (1.0 + xi_val) * (1.0 - eta) * (1.0 - zeta);  // Node 1: (+1,-1,-1)
    N[2] = 0.125 * (1.0 + xi_val) * (1.0 + eta) * (1.0 - zeta);  // Node 2: (+1,+1,-1)
    N[3] = 0.125 * (1.0 - xi_val) * (1.0 + eta) * (1.0 - zeta);  // Node 3: (-1,+1,-1)
    N[4] = 0.125 * (1.0 - xi_val) * (1.0 - eta) * (1.0 + zeta);  // Node 4: (-1,-1,+1)
    N[5] = 0.125 * (1.0 + xi_val) * (1.0 - eta) * (1.0 + zeta);  // Node 5: (+1,-1,+1)
    N[6] = 0.125 * (1.0 + xi_val) * (1.0 + eta) * (1.0 + zeta);  // Node 6: (+1,+1,+1)
    N[7] = 0.125 * (1.0 - xi_val) * (1.0 + eta) * (1.0 + zeta);  // Node 7: (-1,+1,+1)
}

KOKKOS_INLINE_FUNCTION
void Hex8Element::shape_derivatives(const Real xi[3], Real* dN) const {
    const Real xi_val = xi[0];
    const Real eta = xi[1];
    const Real zeta = xi[2];

    // Derivatives: dN[i*3 + j] = ∂N_i/∂ξ_j
    // where j=0:ξ, j=1:η, j=2:ζ

    // Node 0: (-1,-1,-1)
    dN[0*3 + 0] = -0.125 * (1.0 - eta) * (1.0 - zeta);  // dN0/dξ
    dN[0*3 + 1] = -0.125 * (1.0 - xi_val) * (1.0 - zeta);  // dN0/dη
    dN[0*3 + 2] = -0.125 * (1.0 - xi_val) * (1.0 - eta);  // dN0/dζ

    // Node 1: (+1,-1,-1)
    dN[1*3 + 0] =  0.125 * (1.0 - eta) * (1.0 - zeta);
    dN[1*3 + 1] = -0.125 * (1.0 + xi_val) * (1.0 - zeta);
    dN[1*3 + 2] = -0.125 * (1.0 + xi_val) * (1.0 - eta);

    // Node 2: (+1,+1,-1)
    dN[2*3 + 0] =  0.125 * (1.0 + eta) * (1.0 - zeta);
    dN[2*3 + 1] =  0.125 * (1.0 + xi_val) * (1.0 - zeta);
    dN[2*3 + 2] = -0.125 * (1.0 + xi_val) * (1.0 + eta);

    // Node 3: (-1,+1,-1)
    dN[3*3 + 0] = -0.125 * (1.0 + eta) * (1.0 - zeta);
    dN[3*3 + 1] =  0.125 * (1.0 - xi_val) * (1.0 - zeta);
    dN[3*3 + 2] = -0.125 * (1.0 - xi_val) * (1.0 + eta);

    // Node 4: (-1,-1,+1)
    dN[4*3 + 0] = -0.125 * (1.0 - eta) * (1.0 + zeta);
    dN[4*3 + 1] = -0.125 * (1.0 - xi_val) * (1.0 + zeta);
    dN[4*3 + 2] =  0.125 * (1.0 - xi_val) * (1.0 - eta);

    // Node 5: (+1,-1,+1)
    dN[5*3 + 0] =  0.125 * (1.0 - eta) * (1.0 + zeta);
    dN[5*3 + 1] = -0.125 * (1.0 + xi_val) * (1.0 + zeta);
    dN[5*3 + 2] =  0.125 * (1.0 + xi_val) * (1.0 - eta);

    // Node 6: (+1,+1,+1)
    dN[6*3 + 0] =  0.125 * (1.0 + eta) * (1.0 + zeta);
    dN[6*3 + 1] =  0.125 * (1.0 + xi_val) * (1.0 + zeta);
    dN[6*3 + 2] =  0.125 * (1.0 + xi_val) * (1.0 + eta);

    // Node 7: (-1,+1,+1)
    dN[7*3 + 0] = -0.125 * (1.0 + eta) * (1.0 + zeta);
    dN[7*3 + 1] =  0.125 * (1.0 - xi_val) * (1.0 + zeta);
    dN[7*3 + 2] =  0.125 * (1.0 - xi_val) * (1.0 + eta);
}

// ============================================================================
// Gauss Quadrature
// ============================================================================

void Hex8Element::compute_gauss_points_1pt(Real* points, Real* weights) const {
    // 1-point Gauss quadrature (reduced integration)
    // Point at element center
    points[0] = 0.0;  // ξ
    points[1] = 0.0;  // η
    points[2] = 0.0;  // ζ
    weights[0] = 8.0;  // Weight = 2^3 for [-1,1]^3 domain
}

void Hex8Element::compute_gauss_points_8pt(Real* points, Real* weights) const {
    // 8-point Gauss quadrature (full integration)
    // Gauss point location: ±1/√3
    const Real a = 1.0 / std::sqrt(3.0);  // ≈ 0.577350269

    // 8 corner points in natural coordinates
    const Real gp[8][3] = {
        {-a, -a, -a},
        { a, -a, -a},
        { a,  a, -a},
        {-a,  a, -a},
        {-a, -a,  a},
        { a, -a,  a},
        { a,  a,  a},
        {-a,  a,  a}
    };

    for (int i = 0; i < 8; ++i) {
        points[i*3 + 0] = gp[i][0];
        points[i*3 + 1] = gp[i][1];
        points[i*3 + 2] = gp[i][2];
        weights[i] = 1.0;  // Each point has weight 1
    }
}

void Hex8Element::gauss_quadrature(Real* points, Real* weights) const {
    // Default to 1-point integration for explicit dynamics
    compute_gauss_points_1pt(points, weights);
}

// ============================================================================
// Jacobian Computation
// ============================================================================

KOKKOS_INLINE_FUNCTION
Real Hex8Element::jacobian(const Real xi[3], const Real* coords, Real* J) const {
    // Compute Jacobian matrix: J[i][j] = ∂x_i/∂ξ_j
    // J = [∂x/∂ξ  ∂x/∂η  ∂x/∂ζ]
    //     [∂y/∂ξ  ∂y/∂η  ∂y/∂ζ]
    //     [∂z/∂ξ  ∂z/∂η  ∂z/∂ζ]

    // Get shape function derivatives
    Real dN[NUM_NODES * NUM_DIMS];
    shape_derivatives(xi, dN);

    // Initialize Jacobian to zero
    for (int i = 0; i < 9; ++i) {
        J[i] = 0.0;
    }

    // J = Σ (∂N_i/∂ξ_j) * x_i
    for (int i = 0; i < NUM_NODES; ++i) {
        const Real x = coords[i*3 + 0];
        const Real y = coords[i*3 + 1];
        const Real z = coords[i*3 + 2];

        const Real dNdxi   = dN[i*3 + 0];
        const Real dNdeta  = dN[i*3 + 1];
        const Real dNdzeta = dN[i*3 + 2];

        // First column: ∂/∂ξ
        J[0] += dNdxi * x;  // ∂x/∂ξ
        J[3] += dNdxi * y;  // ∂y/∂ξ
        J[6] += dNdxi * z;  // ∂z/∂ξ

        // Second column: ∂/∂η
        J[1] += dNdeta * x;  // ∂x/∂η
        J[4] += dNdeta * y;  // ∂y/∂η
        J[7] += dNdeta * z;  // ∂z/∂η

        // Third column: ∂/∂ζ
        J[2] += dNdzeta * x;  // ∂x/∂ζ
        J[5] += dNdzeta * y;  // ∂y/∂ζ
        J[8] += dNdzeta * z;  // ∂z/∂ζ
    }

    // Compute determinant: det(J) = J[0]*(J[4]*J[8] - J[5]*J[7]) - J[1]*(J[3]*J[8] - J[5]*J[6]) + J[2]*(J[3]*J[7] - J[4]*J[6])
    const Real det_J = J[0] * (J[4] * J[8] - J[5] * J[7])
                     - J[1] * (J[3] * J[8] - J[5] * J[6])
                     + J[2] * (J[3] * J[7] - J[4] * J[6]);

    return det_J;
}

void Hex8Element::inverse_jacobian(const Real* J, Real* J_inv, Real det_J) const {
    // Compute inverse Jacobian using cofactor method
    // J_inv = adj(J) / det(J)

    const Real inv_det = 1.0 / det_J;

    // Cofactor matrix (transposed = adjugate)
    J_inv[0] = inv_det * (J[4] * J[8] - J[5] * J[7]);  // (J22*J33 - J23*J32)
    J_inv[1] = inv_det * (J[2] * J[7] - J[1] * J[8]);  // (J13*J32 - J12*J33)
    J_inv[2] = inv_det * (J[1] * J[5] - J[2] * J[4]);  // (J12*J23 - J13*J22)

    J_inv[3] = inv_det * (J[5] * J[6] - J[3] * J[8]);  // (J23*J31 - J21*J33)
    J_inv[4] = inv_det * (J[0] * J[8] - J[2] * J[6]);  // (J11*J33 - J13*J31)
    J_inv[5] = inv_det * (J[2] * J[3] - J[0] * J[5]);  // (J13*J21 - J11*J23)

    J_inv[6] = inv_det * (J[3] * J[7] - J[4] * J[6]);  // (J21*J32 - J22*J31)
    J_inv[7] = inv_det * (J[1] * J[6] - J[0] * J[7]);  // (J12*J31 - J11*J32)
    J_inv[8] = inv_det * (J[0] * J[4] - J[1] * J[3]);  // (J11*J22 - J12*J21)
}

Real Hex8Element::shape_derivatives_global(const Real xi[3], const Real* coords, Real* dNdx) const {
    // Get shape function derivatives w.r.t. natural coordinates
    Real dN[NUM_NODES * NUM_DIMS];
    shape_derivatives(xi, dN);

    // Compute Jacobian
    Real J[9];
    const Real det_J = jacobian(xi, coords, J);

    // Compute inverse Jacobian
    Real J_inv[9];
    inverse_jacobian(J, J_inv, det_J);

    // Transform derivatives: dN/dx = J^{-1} * dN/dξ
    for (int i = 0; i < NUM_NODES; ++i) {
        const Real dNdxi   = dN[i*3 + 0];
        const Real dNdeta  = dN[i*3 + 1];
        const Real dNdzeta = dN[i*3 + 2];

        dNdx[i*3 + 0] = J_inv[0] * dNdxi + J_inv[1] * dNdeta + J_inv[2] * dNdzeta;  // dN/dx
        dNdx[i*3 + 1] = J_inv[3] * dNdxi + J_inv[4] * dNdeta + J_inv[5] * dNdzeta;  // dN/dy
        dNdx[i*3 + 2] = J_inv[6] * dNdxi + J_inv[7] * dNdeta + J_inv[8] * dNdzeta;  // dN/dz
    }

    return det_J;
}

// ============================================================================
// B-Matrix (Strain-Displacement)
// ============================================================================

KOKKOS_INLINE_FUNCTION
void Hex8Element::strain_displacement_matrix(const Real xi[3], const Real* coords, Real* B) const {
    // B-matrix: 6 x 24 (6 strain components, 24 DOFs)
    // Relates nodal displacements to strains: {ε} = [B]{u}
    //
    // Strain vector (Voigt notation): {ε} = [εxx, εyy, εzz, γxy, γyz, γxz]^T
    // Displacement vector: {u} = [u0x, u0y, u0z, u1x, u1y, u1z, ..., u7x, u7y, u7z]^T
    //
    // B-matrix structure for node i:
    //   [dNi/dx    0       0   ]
    //   [  0     dNi/dy    0   ]
    //   [  0       0     dNi/dz]
    //   [dNi/dy  dNi/dx    0   ]
    //   [  0     dNi/dz  dNi/dy]
    //   [dNi/dz    0     dNi/dx]

    // Initialize B to zero
    for (int i = 0; i < 6 * NUM_DOF; ++i) {
        B[i] = 0.0;
    }

    // Get shape function derivatives w.r.t. global coordinates
    Real dNdx[NUM_NODES * NUM_DIMS];
    shape_derivatives_global(xi, coords, dNdx);

    // Fill B-matrix for each node
    for (int i = 0; i < NUM_NODES; ++i) {
        const Real dNdx_i = dNdx[i*3 + 0];
        const Real dNdy_i = dNdx[i*3 + 1];
        const Real dNdz_i = dNdx[i*3 + 2];

        const int col = i * 3;  // Column offset for node i

        // εxx row (row 0)
        B[0 * NUM_DOF + col + 0] = dNdx_i;

        // εyy row (row 1)
        B[1 * NUM_DOF + col + 1] = dNdy_i;

        // εzz row (row 2)
        B[2 * NUM_DOF + col + 2] = dNdz_i;

        // γxy row (row 3)
        B[3 * NUM_DOF + col + 0] = dNdy_i;
        B[3 * NUM_DOF + col + 1] = dNdx_i;

        // γyz row (row 4)
        B[4 * NUM_DOF + col + 1] = dNdz_i;
        B[4 * NUM_DOF + col + 2] = dNdy_i;

        // γxz row (row 5)
        B[5 * NUM_DOF + col + 0] = dNdz_i;
        B[5 * NUM_DOF + col + 2] = dNdx_i;
    }
}

// ============================================================================
// Constitutive Matrix
// ============================================================================

void Hex8Element::constitutive_matrix(Real E, Real nu, Real* C) const {
    // Elastic constitutive matrix (Voigt notation)
    // C: 6x6 matrix for 3D isotropic linear elasticity
    //
    // {σ} = [C]{ε}
    // where {σ} = [σxx, σyy, σzz, τxy, τyz, τxz]^T
    //       {ε} = [εxx, εyy, εzz, γxy, γyz, γxz]^T

    // Initialize to zero
    for (int i = 0; i < 36; ++i) {
        C[i] = 0.0;
    }

    const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const Real mu = E / (2.0 * (1.0 + nu));

    // Diagonal terms
    const Real diag = lambda + 2.0 * mu;
    C[0*6 + 0] = diag;  // C11
    C[1*6 + 1] = diag;  // C22
    C[2*6 + 2] = diag;  // C33
    C[3*6 + 3] = mu;    // C44 (shear)
    C[4*6 + 4] = mu;    // C55 (shear)
    C[5*6 + 5] = mu;    // C66 (shear)

    // Off-diagonal terms (coupling between normal strains)
    C[0*6 + 1] = lambda;  // C12
    C[0*6 + 2] = lambda;  // C13
    C[1*6 + 0] = lambda;  // C21
    C[1*6 + 2] = lambda;  // C23
    C[2*6 + 0] = lambda;  // C31
    C[2*6 + 1] = lambda;  // C32
}

// ============================================================================
// Mass Matrix
// ============================================================================

void Hex8Element::lumped_mass_matrix(const Real* coords, Real density, Real* M) const {
    // Lumped mass matrix (diagonal)
    // Total element mass distributed equally to nodes

    // Compute element volume using 1-point Gauss quadrature
    Real xi_center[3] = {0.0, 0.0, 0.0};
    Real J[9];
    const Real det_J = jacobian(xi_center, coords, J);
    const Real volume_elem = det_J * 8.0;  // Weight for full domain [-1,1]^3

    const Real total_mass = density * volume_elem;
    const Real nodal_mass = total_mass / NUM_NODES;

    // Distribute mass equally to all DOFs at each node
    for (int i = 0; i < NUM_DOF; ++i) {
        M[i] = nodal_mass;
    }
}

void Hex8Element::mass_matrix(const Real* coords, Real density, Real* M) const {
    // Consistent mass matrix: M = ∫ ρ N^T N dV
    // M: 24x24 matrix

    // Initialize to zero
    for (int i = 0; i < NUM_DOF * NUM_DOF; ++i) {
        M[i] = 0.0;
    }

    // Use 8-point Gauss quadrature for accurate integration
    Real gp[8*3], gw[8];
    compute_gauss_points_8pt(gp, gw);

    // Integrate over all Gauss points
    for (int ig = 0; ig < 8; ++ig) {
        Real xi[3] = {gp[ig*3 + 0], gp[ig*3 + 1], gp[ig*3 + 2]};
        const Real weight = gw[ig];

        // Compute shape functions
        Real N[NUM_NODES];
        shape_functions(xi, N);

        // Compute Jacobian determinant
        Real J[9];
        const Real det_J = jacobian(xi, coords, J);

        const Real factor = density * weight * det_J;

        // Assemble: M[i,j] += ρ * N_i * N_j * det(J) * w
        for (int i = 0; i < NUM_NODES; ++i) {
            for (int j = 0; j < NUM_NODES; ++j) {
                const Real mass_ij = factor * N[i] * N[j];

                // Add to 3x3 block for node pair (i,j)
                for (int d = 0; d < NUM_DIMS; ++d) {
                    const int row = i * NUM_DIMS + d;
                    const int col = j * NUM_DIMS + d;
                    M[row * NUM_DOF + col] += mass_ij;
                }
            }
        }
    }
}

// ============================================================================
// Stiffness Matrix
// ============================================================================

void Hex8Element::stiffness_matrix(const Real* coords, Real E, Real nu, Real* K) const {
    // Element stiffness matrix: K = ∫ B^T C B dV
    // K: 24x24 matrix

    // Initialize to zero
    for (int i = 0; i < NUM_DOF * NUM_DOF; ++i) {
        K[i] = 0.0;
    }

    // Compute constitutive matrix
    Real C[36];
    constitutive_matrix(E, nu, C);

    // Use 8-point Gauss quadrature
    Real gp[8*3], gw[8];
    compute_gauss_points_8pt(gp, gw);

    // Temporary arrays
    Real B[6 * NUM_DOF];   // 6 x 24
    Real CB[6 * NUM_DOF];  // C * B

    // Integrate over all Gauss points
    for (int ig = 0; ig < 8; ++ig) {
        Real xi[3] = {gp[ig*3 + 0], gp[ig*3 + 1], gp[ig*3 + 2]};
        const Real weight = gw[ig];

        // Compute B-matrix
        strain_displacement_matrix(xi, coords, B);

        // Compute Jacobian determinant
        Real J[9];
        const Real det_J = jacobian(xi, coords, J);

        // CB = C * B (6x6 * 6x24 = 6x24)
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < NUM_DOF; ++j) {
                CB[i * NUM_DOF + j] = 0.0;
                for (int k = 0; k < 6; ++k) {
                    CB[i * NUM_DOF + j] += C[i * 6 + k] * B[k * NUM_DOF + j];
                }
            }
        }

        const Real factor = weight * det_J;

        // K += B^T * CB * det(J) * w
        for (int i = 0; i < NUM_DOF; ++i) {
            for (int j = 0; j < NUM_DOF; ++j) {
                Real sum = 0.0;
                for (int k = 0; k < 6; ++k) {
                    sum += B[k * NUM_DOF + i] * CB[k * NUM_DOF + j];
                }
                K[i * NUM_DOF + j] += factor * sum;
            }
        }
    }
}

// ============================================================================
// Strain and Stress Computation
// ============================================================================

KOKKOS_INLINE_FUNCTION
void Hex8Element::compute_strain(const Real* coords, const Real* disp, Real* strain) const {
    // Compute strain from displacement: ε = B * u
    // Using 1-point Gauss integration at element center

    // Element center in natural coordinates
    Real xi[3] = {0.0, 0.0, 0.0};

    // Compute B-matrix at center
    Real B[6 * NUM_DOF];
    strain_displacement_matrix(xi, coords, B);

    // Initialize strain to zero
    for (int i = 0; i < 6; ++i) {
        strain[i] = 0.0;
    }

    // Compute strain: ε = B * u
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < NUM_DOF; ++j) {
            strain[i] += B[i * NUM_DOF + j] * disp[j];
        }
    }
}

KOKKOS_INLINE_FUNCTION
void Hex8Element::compute_stress(const Real* strain, Real E, Real nu, Real* stress) const {
    // Compute stress from strain: σ = C * ε
    // Using linear elastic constitutive model (Voigt notation)

    // Compute Lamé parameters
    const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const Real mu = E / (2.0 * (1.0 + nu));

    // Normal stresses: σ_ii = (λ + 2μ) * ε_ii + λ * (ε_jj + ε_kk)
    const Real trace_strain = strain[0] + strain[1] + strain[2];  // εxx + εyy + εzz

    stress[0] = (lambda + 2.0 * mu) * strain[0] + lambda * (strain[1] + strain[2]);  // σxx
    stress[1] = (lambda + 2.0 * mu) * strain[1] + lambda * (strain[0] + strain[2]);  // σyy
    stress[2] = (lambda + 2.0 * mu) * strain[2] + lambda * (strain[0] + strain[1]);  // σzz

    // Shear stresses: τ_ij = μ * γ_ij
    stress[3] = mu * strain[3];  // τxy
    stress[4] = mu * strain[4];  // τyz
    stress[5] = mu * strain[5];  // τxz
}

// ============================================================================
// Internal Force
// ============================================================================

KOKKOS_INLINE_FUNCTION
void Hex8Element::internal_force(const Real* coords, const Real* disp,
                                  const Real* stress, Real* fint) const {
    // Internal force: f_int = ∫ B^T σ dV
    // Note: This signature assumes stress is passed in from outside (1-point scheme)
    // For better accuracy in bending, we compute stress at each Gauss point here

    // Initialize to zero
    for (int i = 0; i < NUM_DOF; ++i) {
        fint[i] = 0.0;
    }

    // Use 1-point Gauss quadrature (stress at element center)
    // This is retained for explicit dynamics where reduced integration
    // is preferred for computational efficiency and to avoid locking
    Real xi[3] = {0.0, 0.0, 0.0};

    // Compute B-matrix at center
    Real B[6 * NUM_DOF];
    strain_displacement_matrix(xi, coords, B);

    // Compute Jacobian determinant
    Real J[9];
    const Real det_J = jacobian(xi, coords, J);

    const Real factor = det_J * 8.0;  // Weight for full domain

    // f_int = B^T * σ * det(J) * w
    for (int i = 0; i < NUM_DOF; ++i) {
        Real sum = 0.0;
        for (int j = 0; j < 6; ++j) {
            sum += B[j * NUM_DOF + i] * stress[j];
        }
        fint[i] = factor * sum;
    }
}

// ============================================================================
// Hourglass Control
// ============================================================================

void Hex8Element::hourglass_forces(const Real* coords, const Real* disp,
                                    Real hourglass_stiffness, Real* fhg) const {
    // Flanagan-Belytschko hourglass control for 1-point integration
    // Computes anti-hourglass forces to stabilize spurious zero-energy modes

    // Initialize to zero
    for (int i = 0; i < NUM_DOF; ++i) {
        fhg[i] = 0.0;
    }

    // Hourglass mode vectors (4 modes for hex8)
    // These are orthogonal to rigid body modes and constant strain modes
    const Real gamma[4][8] = {
        { 1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0},  // Mode 1
        { 1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0},  // Mode 2
        { 1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0},  // Mode 3
        {-1.0,  1.0,  1.0, -1.0,  1.0, -1.0, -1.0,  1.0}   // Mode 4
    };

    // Compute element center coordinates
    Real x_center[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < NUM_NODES; ++i) {
        x_center[0] += coords[i*3 + 0];
        x_center[1] += coords[i*3 + 1];
        x_center[2] += coords[i*3 + 2];
    }
    x_center[0] /= NUM_NODES;
    x_center[1] /= NUM_NODES;
    x_center[2] /= NUM_NODES;

    // Compute Jacobian at element center for characteristic length
    Real xi_center[3] = {0.0, 0.0, 0.0};
    Real J[9];
    const Real det_J = jacobian(xi_center, coords, J);
    const Real char_length = std::cbrt(det_J * 8.0);  // Cube root of volume

    // Loop over 4 hourglass modes
    for (int mode = 0; mode < 4; ++mode) {
        // Compute hourglass base vectors (gradient of mode shape)
        Real hg_base[3] = {0.0, 0.0, 0.0};

        for (int i = 0; i < NUM_NODES; ++i) {
            const Real dx = coords[i*3 + 0] - x_center[0];
            const Real dy = coords[i*3 + 1] - x_center[1];
            const Real dz = coords[i*3 + 2] - x_center[2];

            hg_base[0] += gamma[mode][i] * dx;
            hg_base[1] += gamma[mode][i] * dy;
            hg_base[2] += gamma[mode][i] * dz;
        }

        // Compute hourglass displacement (generalized hourglass strain)
        Real hg_disp[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < NUM_NODES; ++i) {
            hg_disp[0] += gamma[mode][i] * disp[i*3 + 0];
            hg_disp[1] += gamma[mode][i] * disp[i*3 + 1];
            hg_disp[2] += gamma[mode][i] * disp[i*3 + 2];
        }

        // Compute hourglass resistance coefficient
        // Use characteristic length and stiffness parameter
        const Real hg_coef = hourglass_stiffness * char_length;

        // Add hourglass resistance forces
        for (int i = 0; i < NUM_NODES; ++i) {
            fhg[i*3 + 0] -= hg_coef * gamma[mode][i] * hg_disp[0];
            fhg[i*3 + 1] -= hg_coef * gamma[mode][i] * hg_disp[1];
            fhg[i*3 + 2] -= hg_coef * gamma[mode][i] * hg_disp[2];
        }
    }
}

// ============================================================================
// Geometric Queries
// ============================================================================

bool Hex8Element::contains_point(const Real* coords, const Real* point, Real* xi) const {
    // Newton-Raphson iteration to find natural coordinates
    // Given global point p, solve: p = Σ N_i(ξ) * x_i

    const int max_iter = 20;
    const Real tol = 1.0e-6;

    // Initial guess: element center
    xi[0] = 0.0;
    xi[1] = 0.0;
    xi[2] = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Compute current position
        Real N[NUM_NODES];
        shape_functions(xi, N);

        Real x_current[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < NUM_NODES; ++i) {
            x_current[0] += N[i] * coords[i*3 + 0];
            x_current[1] += N[i] * coords[i*3 + 1];
            x_current[2] += N[i] * coords[i*3 + 2];
        }

        // Residual
        Real r[3];
        r[0] = point[0] - x_current[0];
        r[1] = point[1] - x_current[1];
        r[2] = point[2] - x_current[2];

        const Real r_norm = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        if (r_norm < tol) {
            // Converged - check if inside element
            return (std::abs(xi[0]) <= 1.0 + tol &&
                    std::abs(xi[1]) <= 1.0 + tol &&
                    std::abs(xi[2]) <= 1.0 + tol);
        }

        // Compute Jacobian
        Real J[9];
        jacobian(xi, coords, J);

        // Solve J * Δξ = r using Cramer's rule
        const Real det_J = J[0] * (J[4] * J[8] - J[5] * J[7])
                         - J[1] * (J[3] * J[8] - J[5] * J[6])
                         + J[2] * (J[3] * J[7] - J[4] * J[6]);

        if (std::abs(det_J) < 1.0e-12) {
            return false;  // Singular Jacobian
        }

        // Compute J^{-1} * r
        const Real inv_det = 1.0 / det_J;
        Real J_inv_r[3];
        J_inv_r[0] = inv_det * ((J[4]*J[8] - J[5]*J[7])*r[0] + (J[2]*J[7] - J[1]*J[8])*r[1] + (J[1]*J[5] - J[2]*J[4])*r[2]);
        J_inv_r[1] = inv_det * ((J[5]*J[6] - J[3]*J[8])*r[0] + (J[0]*J[8] - J[2]*J[6])*r[1] + (J[2]*J[3] - J[0]*J[5])*r[2]);
        J_inv_r[2] = inv_det * ((J[3]*J[7] - J[4]*J[6])*r[0] + (J[1]*J[6] - J[0]*J[7])*r[1] + (J[0]*J[4] - J[1]*J[3])*r[2]);

        // Update
        xi[0] += J_inv_r[0];
        xi[1] += J_inv_r[1];
        xi[2] += J_inv_r[2];
    }

    return false;  // Did not converge
}

Real Hex8Element::volume(const Real* coords) const {
    // Compute volume using 8-point Gauss quadrature
    Real gp[8*3], gw[8];
    compute_gauss_points_8pt(gp, gw);

    Real vol = 0.0;
    for (int ig = 0; ig < 8; ++ig) {
        Real xi[3] = {gp[ig*3 + 0], gp[ig*3 + 1], gp[ig*3 + 2]};
        const Real weight = gw[ig];

        Real J[9];
        const Real det_J = jacobian(xi, coords, J);

        vol += weight * det_J;
    }

    return vol;
}

Real Hex8Element::characteristic_length(const Real* coords) const {
    // Compute minimum edge length
    // Hexahedron has 12 edges

    const int edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Bottom face
        {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Top face
        {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Vertical edges
    };

    Real min_length = 1.0e30;

    for (int ie = 0; ie < 12; ++ie) {
        const int n0 = edges[ie][0];
        const int n1 = edges[ie][1];

        const Real dx = coords[n1*3 + 0] - coords[n0*3 + 0];
        const Real dy = coords[n1*3 + 1] - coords[n0*3 + 1];
        const Real dz = coords[n1*3 + 2] - coords[n0*3 + 2];

        const Real length = std::sqrt(dx*dx + dy*dy + dz*dz);
        min_length = std::min(min_length, length);
    }

    return min_length;
}

} // namespace fem
} // namespace nxs
