/**
 * @file tet4.cpp
 * @brief Implementation of 4-node tetrahedral element
 */

#include <nexussim/discretization/tet4.hpp>
#include <cmath>
#include <algorithm>

namespace nxs {
namespace fem {

// ============================================================================
// Shape Functions
// ============================================================================

KOKKOS_INLINE_FUNCTION
void Tet4Element::shape_functions(const Real xi[3], Real* N) const {
    // Linear shape functions (volume coordinates)
    // N1 = ξ
    // N2 = η
    // N3 = ζ
    // N4 = 1 - ξ - η - ζ

    N[0] = 1.0 - xi[0] - xi[1] - xi[2];  // Node 0 (origin)
    N[1] = xi[0];                         // Node 1 (ξ direction)
    N[2] = xi[1];                         // Node 2 (η direction)
    N[3] = xi[2];                         // Node 3 (ζ direction)
}

KOKKOS_INLINE_FUNCTION
void Tet4Element::shape_derivatives(const Real xi[3], Real* dN) const {
    // Shape function derivatives are CONSTANT for linear tetrahedron
    // This is one of the key advantages of Tet4 - derivatives don't depend on position

    // dN/dξ
    dN[0*3 + 0] = -1.0;  // dN0/dξ
    dN[1*3 + 0] =  1.0;  // dN1/dξ
    dN[2*3 + 0] =  0.0;  // dN2/dξ
    dN[3*3 + 0] =  0.0;  // dN3/dξ

    // dN/dη
    dN[0*3 + 1] = -1.0;  // dN0/dη
    dN[1*3 + 1] =  0.0;  // dN1/dη
    dN[2*3 + 1] =  1.0;  // dN2/dη
    dN[3*3 + 1] =  0.0;  // dN3/dη

    // dN/dζ
    dN[0*3 + 2] = -1.0;  // dN0/dζ
    dN[1*3 + 2] =  0.0;  // dN1/dζ
    dN[2*3 + 2] =  0.0;  // dN2/dζ
    dN[3*3 + 2] =  1.0;  // dN3/dζ
}

// ============================================================================
// Gauss Quadrature
// ============================================================================

void Tet4Element::gauss_quadrature(Real* points, Real* weights) const {
    // 1-point Gauss quadrature at centroid
    // Centroid in volume coordinates: (1/4, 1/4, 1/4, 1/4)
    // In (ξ, η, ζ) coordinates: (1/4, 1/4, 1/4)

    points[0] = 0.25;  // ξ
    points[1] = 0.25;  // η
    points[2] = 0.25;  // ζ
    weights[0] = 1.0/6.0;  // Weight for volume integral over reference tet
}

// ============================================================================
// Jacobian Computation
// ============================================================================

KOKKOS_INLINE_FUNCTION
Real Tet4Element::jacobian(const Real xi[3], const Real* coords, Real* J) const {
    // For linear tetrahedron, Jacobian is CONSTANT throughout element
    // J[i][j] = ∂x_i/∂ξ_j

    // Get shape function derivatives (constant for Tet4)
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
        J[0] += dNdxi * x;
        J[3] += dNdxi * y;
        J[6] += dNdxi * z;

        // Second column: ∂/∂η
        J[1] += dNdeta * x;
        J[4] += dNdeta * y;
        J[7] += dNdeta * z;

        // Third column: ∂/∂ζ
        J[2] += dNdzeta * x;
        J[5] += dNdzeta * y;
        J[8] += dNdzeta * z;
    }

    // Compute determinant
    const Real det_J = J[0] * (J[4] * J[8] - J[5] * J[7])
                     - J[1] * (J[3] * J[8] - J[5] * J[6])
                     + J[2] * (J[3] * J[7] - J[4] * J[6]);

    return det_J;
}

void Tet4Element::inverse_jacobian(const Real* J, Real* J_inv, Real det_J) const {
    const Real inv_det = 1.0 / det_J;

    // Cofactor matrix (transposed = adjugate)
    J_inv[0] = inv_det * (J[4] * J[8] - J[5] * J[7]);
    J_inv[1] = inv_det * (J[2] * J[7] - J[1] * J[8]);
    J_inv[2] = inv_det * (J[1] * J[5] - J[2] * J[4]);

    J_inv[3] = inv_det * (J[5] * J[6] - J[3] * J[8]);
    J_inv[4] = inv_det * (J[0] * J[8] - J[2] * J[6]);
    J_inv[5] = inv_det * (J[2] * J[3] - J[0] * J[5]);

    J_inv[6] = inv_det * (J[3] * J[7] - J[4] * J[6]);
    J_inv[7] = inv_det * (J[1] * J[6] - J[0] * J[7]);
    J_inv[8] = inv_det * (J[0] * J[4] - J[1] * J[3]);
}

Real Tet4Element::shape_derivatives_global(const Real xi[3], const Real* coords, Real* dNdx) const {
    // Get shape function derivatives w.r.t. natural coordinates (constant for Tet4)
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

        // dN/dx_i = Σ_j (J^{-1})_{ji} * dN/dξ_j  (use columns of J_inv = J^{-T})
        dNdx[i*3 + 0] = J_inv[0] * dNdxi + J_inv[3] * dNdeta + J_inv[6] * dNdzeta;  // dN/dx
        dNdx[i*3 + 1] = J_inv[1] * dNdxi + J_inv[4] * dNdeta + J_inv[7] * dNdzeta;  // dN/dy
        dNdx[i*3 + 2] = J_inv[2] * dNdxi + J_inv[5] * dNdeta + J_inv[8] * dNdzeta;  // dN/dz
    }

    return det_J;
}

// ============================================================================
// B-Matrix (Strain-Displacement)
// ============================================================================

KOKKOS_INLINE_FUNCTION
void Tet4Element::strain_displacement_matrix(const Real xi[3], const Real* coords, Real* B) const {
    // B-matrix: 6 x 12 (6 strain components, 12 DOFs)

    // Initialize B to zero
    for (int i = 0; i < 6 * NUM_DOF; ++i) {
        B[i] = 0.0;
    }

    // Get shape function derivatives w.r.t. global coordinates
    // For Tet4, these are constant throughout the element!
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

void Tet4Element::constitutive_matrix(Real E, Real nu, Real* C) const {
    // Same as Hex8 - 3D isotropic linear elasticity

    // Initialize to zero
    for (int i = 0; i < 36; ++i) {
        C[i] = 0.0;
    }

    const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const Real mu = E / (2.0 * (1.0 + nu));

    // Diagonal terms
    const Real diag = lambda + 2.0 * mu;
    C[0*6 + 0] = diag;
    C[1*6 + 1] = diag;
    C[2*6 + 2] = diag;
    C[3*6 + 3] = mu;
    C[4*6 + 4] = mu;
    C[5*6 + 5] = mu;

    // Off-diagonal terms
    C[0*6 + 1] = lambda;
    C[0*6 + 2] = lambda;
    C[1*6 + 0] = lambda;
    C[1*6 + 2] = lambda;
    C[2*6 + 0] = lambda;
    C[2*6 + 1] = lambda;
}

// ============================================================================
// Mass Matrix
// ============================================================================

void Tet4Element::lumped_mass_matrix(const Real* coords, Real density, Real* M) const {
    // Compute element volume
    const Real vol = volume(coords);
    const Real total_mass = density * vol;
    const Real nodal_mass = total_mass / NUM_NODES;

    // Distribute mass equally to all DOFs
    for (int i = 0; i < NUM_DOF; ++i) {
        M[i] = nodal_mass;
    }
}

void Tet4Element::mass_matrix(const Real* coords, Real density, Real* M) const {
    // Consistent mass matrix: M = ∫ ρ N^T N dV
    // For linear tetrahedron, this can be computed analytically

    // Initialize to zero
    for (int i = 0; i < NUM_DOF * NUM_DOF; ++i) {
        M[i] = 0.0;
    }

    const Real vol = volume(coords);
    const Real total_mass = density * vol;

    // For linear tet, consistent mass matrix has this pattern:
    // Diagonal:     total_mass / 10
    // Off-diagonal: total_mass / 20
    // Each value goes into a 3x3 diagonal block (one per DOF)

    const Real diag_mass = total_mass / 10.0;
    const Real off_diag_mass = total_mass / 20.0;

    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < NUM_NODES; ++j) {
            const Real mass_ij = (i == j) ? diag_mass : off_diag_mass;

            // Add to 3x3 diagonal block for node pair (i,j)
            // Each DOF gets the same mass value (mass per DOF, not total mass)
            for (int d = 0; d < NUM_DIMS; ++d) {
                const int row = i * NUM_DIMS + d;
                const int col = j * NUM_DIMS + d;
                M[row * NUM_DOF + col] = mass_ij / NUM_DIMS;
            }
        }
    }
}

// ============================================================================
// Stiffness Matrix
// ============================================================================

void Tet4Element::stiffness_matrix(const Real* coords, Real E, Real nu, Real* K) const {
    // Element stiffness matrix: K = V * B^T * C * B
    // Since B is constant for Tet4, we only need one evaluation!

    // Initialize to zero
    for (int i = 0; i < NUM_DOF * NUM_DOF; ++i) {
        K[i] = 0.0;
    }

    // Compute constitutive matrix
    Real C[36];
    constitutive_matrix(E, nu, C);

    // Compute B-matrix at centroid (constant everywhere for Tet4)
    Real xi[3] = {0.25, 0.25, 0.25};
    Real B[6 * NUM_DOF];
    strain_displacement_matrix(xi, coords, B);

    // Get element volume
    const Real vol = volume(coords);

    // Temporary array for C * B
    Real CB[6 * NUM_DOF];

    // CB = C * B (6x6 * 6x12 = 6x12)
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < NUM_DOF; ++j) {
            CB[i * NUM_DOF + j] = 0.0;
            for (int k = 0; k < 6; ++k) {
                CB[i * NUM_DOF + j] += C[i * 6 + k] * B[k * NUM_DOF + j];
            }
        }
    }

    // K = B^T * CB * V
    for (int i = 0; i < NUM_DOF; ++i) {
        for (int j = 0; j < NUM_DOF; ++j) {
            Real sum = 0.0;
            for (int k = 0; k < 6; ++k) {
                sum += B[k * NUM_DOF + i] * CB[k * NUM_DOF + j];
            }
            K[i * NUM_DOF + j] = vol * sum;
        }
    }
}

// ============================================================================
// Internal Force
// ============================================================================

KOKKOS_INLINE_FUNCTION
void Tet4Element::internal_force(const Real* coords, const Real* disp,
                                  const Real* stress, Real* fint) const {
    // Internal force: f_int = V * B^T * σ
    // B is constant for Tet4!

    // Initialize to zero
    for (int i = 0; i < NUM_DOF; ++i) {
        fint[i] = 0.0;
    }

    // Compute B-matrix (constant for Tet4)
    Real xi[3] = {0.25, 0.25, 0.25};
    Real B[6 * NUM_DOF];
    strain_displacement_matrix(xi, coords, B);

    // Compute volume
    Real J[9];
    const Real det_J = jacobian(xi, coords, J);
    const Real vol = det_J;  // For reference tetrahedron, V = det(J)

    // f_int = B^T * σ * V
    for (int i = 0; i < NUM_DOF; ++i) {
        Real sum = 0.0;
        for (int j = 0; j < 6; ++j) {
            sum += B[j * NUM_DOF + i] * stress[j];
        }
        fint[i] = vol * sum;
    }
}

// ============================================================================
// Geometric Queries
// ============================================================================

bool Tet4Element::contains_point(const Real* coords, const Real* point, Real* xi) const {
    // Use volume coordinate method
    // A point is inside if all volume coordinates are >= 0

    // Compute volume coordinates directly
    // Volume of tetrahedron: V = det(J) / 6

    const Real* x0 = &coords[0];  // Node 0
    const Real* x1 = &coords[3];  // Node 1
    const Real* x2 = &coords[6];  // Node 2
    const Real* x3 = &coords[9];  // Node 3

    // Total volume
    const Real v0[3] = {x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]};
    const Real v1[3] = {x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]};
    const Real v2[3] = {x3[0] - x0[0], x3[1] - x0[1], x3[2] - x0[2]};

    const Real total_vol = std::abs(
        v0[0] * (v1[1] * v2[2] - v1[2] * v2[1]) -
        v0[1] * (v1[0] * v2[2] - v1[2] * v2[0]) +
        v0[2] * (v1[0] * v2[1] - v1[1] * v2[0])
    ) / 6.0;

    if (total_vol < 1.0e-12) {
        return false;  // Degenerate element
    }

    // Compute volume coordinates (L1, L2, L3, L4)
    // This is done by computing sub-tetrahedron volumes

    // For now, use simple inside test: compute natural coordinates
    // and check if they're in [0,1] and sum <= 1

    // This is a simplified version - full implementation would compute
    // barycentric coordinates properly
    const Real tol = 1.0e-6;

    // Use Newton-Raphson to find natural coordinates
    xi[0] = 0.25;
    xi[1] = 0.25;
    xi[2] = 0.25;

    for (int iter = 0; iter < 10; ++iter) {
        Real N[NUM_NODES];
        shape_functions(xi, N);

        Real x_current[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < NUM_NODES; ++i) {
            x_current[0] += N[i] * coords[i*3 + 0];
            x_current[1] += N[i] * coords[i*3 + 1];
            x_current[2] += N[i] * coords[i*3 + 2];
        }

        Real r[3];
        r[0] = point[0] - x_current[0];
        r[1] = point[1] - x_current[1];
        r[2] = point[2] - x_current[2];

        const Real r_norm = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        if (r_norm < tol) {
            // Converged - check if inside
            const Real L4 = 1.0 - xi[0] - xi[1] - xi[2];
            return (xi[0] >= -tol && xi[1] >= -tol && xi[2] >= -tol && L4 >= -tol);
        }

        // Compute Jacobian and solve
        Real J[9];
        jacobian(xi, coords, J);

        const Real det_J = J[0] * (J[4] * J[8] - J[5] * J[7])
                         - J[1] * (J[3] * J[8] - J[5] * J[6])
                         + J[2] * (J[3] * J[7] - J[4] * J[6]);

        if (std::abs(det_J) < 1.0e-12) {
            return false;
        }

        Real J_inv[9];
        inverse_jacobian(J, J_inv, det_J);

        // Δξ = J^{-1} * r
        const Real dxi0 = J_inv[0]*r[0] + J_inv[1]*r[1] + J_inv[2]*r[2];
        const Real dxi1 = J_inv[3]*r[0] + J_inv[4]*r[1] + J_inv[5]*r[2];
        const Real dxi2 = J_inv[6]*r[0] + J_inv[7]*r[1] + J_inv[8]*r[2];

        xi[0] += dxi0;
        xi[1] += dxi1;
        xi[2] += dxi2;
    }

    return false;  // Did not converge
}

Real Tet4Element::volume(const Real* coords) const {
    // Volume of tetrahedron = |det(J)| / 6
    // where J is Jacobian from natural to physical coordinates
    // The factor of 6 comes from the transformation from natural coords to physical

    Real xi[3] = {0.25, 0.25, 0.25};  // Arbitrary point (Jacobian is constant)
    Real J[9];
    const Real det_J = jacobian(xi, coords, J);

    return std::abs(det_J) / 6.0;  // Volume = |det(J)| / 6 for Tet4
}

Real Tet4Element::characteristic_length(const Real* coords) const {
    // Compute minimum edge length
    // Tetrahedron has 6 edges

    const int edges[6][2] = {
        {0, 1}, {0, 2}, {0, 3},  // Edges from node 0
        {1, 2}, {1, 3}, {2, 3}   // Remaining edges
    };

    Real min_length = 1.0e30;

    for (int ie = 0; ie < 6; ++ie) {
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
