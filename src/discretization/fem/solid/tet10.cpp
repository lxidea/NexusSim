/**
 * @file tet10.cpp
 * @brief Implementation of 10-node tetrahedral element
 */

#include <nexussim/discretization/tet10.hpp>
#include <cmath>
#include <algorithm>

namespace nxs {
namespace fem {

KOKKOS_INLINE_FUNCTION
void Tet10Element::shape_functions(const Real xi[3], Real* N) const {
    // Natural coordinates (barycentric): L1, L2, L3, L4
    // where xi[0]=L1, xi[1]=L2, xi[2]=L3, and L4 = 1-L1-L2-L3
    const Real L1 = xi[0];
    const Real L2 = xi[1];
    const Real L3 = xi[2];
    const Real L4 = 1.0 - L1 - L2 - L3;

    // Corner nodes: N_i = L_i(2*L_i - 1)
    N[0] = L1 * (2.0 * L1 - 1.0);  // Node 0
    N[1] = L2 * (2.0 * L2 - 1.0);  // Node 1
    N[2] = L3 * (2.0 * L3 - 1.0);  // Node 2
    N[3] = L4 * (2.0 * L4 - 1.0);  // Node 3

    // Edge nodes: N_i = 4*L_j*L_k
    N[4] = 4.0 * L1 * L2;  // Edge 0-1
    N[5] = 4.0 * L2 * L3;  // Edge 1-2
    N[6] = 4.0 * L3 * L1;  // Edge 2-0
    N[7] = 4.0 * L1 * L4;  // Edge 0-3
    N[8] = 4.0 * L2 * L4;  // Edge 1-3
    N[9] = 4.0 * L3 * L4;  // Edge 2-3
}

KOKKOS_INLINE_FUNCTION
void Tet10Element::shape_derivatives(const Real xi[3], Real* dN) const {
    const Real L1 = xi[0];
    const Real L2 = xi[1];
    const Real L3 = xi[2];
    const Real L4 = 1.0 - L1 - L2 - L3;

    // dN/dL1, dN/dL2, dN/dL3
    // Corner nodes
    dN[0*3 + 0] = 4.0 * L1 - 1.0;
    dN[0*3 + 1] = 0.0;
    dN[0*3 + 2] = 0.0;

    dN[1*3 + 0] = 0.0;
    dN[1*3 + 1] = 4.0 * L2 - 1.0;
    dN[1*3 + 2] = 0.0;

    dN[2*3 + 0] = 0.0;
    dN[2*3 + 1] = 0.0;
    dN[2*3 + 2] = 4.0 * L3 - 1.0;

    dN[3*3 + 0] = -(4.0 * L4 - 1.0);
    dN[3*3 + 1] = -(4.0 * L4 - 1.0);
    dN[3*3 + 2] = -(4.0 * L4 - 1.0);

    // Edge nodes
    dN[4*3 + 0] = 4.0 * L2;
    dN[4*3 + 1] = 4.0 * L1;
    dN[4*3 + 2] = 0.0;

    dN[5*3 + 0] = 0.0;
    dN[5*3 + 1] = 4.0 * L3;
    dN[5*3 + 2] = 4.0 * L2;

    dN[6*3 + 0] = 4.0 * L3;
    dN[6*3 + 1] = 0.0;
    dN[6*3 + 2] = 4.0 * L1;

    dN[7*3 + 0] = 4.0 * (L4 - L1);
    dN[7*3 + 1] = -4.0 * L1;
    dN[7*3 + 2] = -4.0 * L1;

    dN[8*3 + 0] = -4.0 * L2;
    dN[8*3 + 1] = 4.0 * (L4 - L2);
    dN[8*3 + 2] = -4.0 * L2;

    dN[9*3 + 0] = -4.0 * L3;
    dN[9*3 + 1] = -4.0 * L3;
    dN[9*3 + 2] = 4.0 * (L4 - L3);
}

void Tet10Element::compute_gauss_points_4pt(Real* points, Real* weights) const {
    // 4-point Gauss quadrature for quadratic tetrahedron
    const Real a = 0.585410196624969;  // (5 + 3*sqrt(5))/20
    const Real b = 0.138196601125011;  // (5 - sqrt(5))/20

    points[0*3 + 0] = a; points[0*3 + 1] = b; points[0*3 + 2] = b;
    points[1*3 + 0] = b; points[1*3 + 1] = a; points[1*3 + 2] = b;
    points[2*3 + 0] = b; points[2*3 + 1] = b; points[2*3 + 2] = a;
    points[3*3 + 0] = b; points[3*3 + 1] = b; points[3*3 + 2] = b;

    weights[0] = 0.25;
    weights[1] = 0.25;
    weights[2] = 0.25;
    weights[3] = 0.25;
}

void Tet10Element::gauss_quadrature(Real* points, Real* weights) const {
    compute_gauss_points_4pt(points, weights);
}

KOKKOS_INLINE_FUNCTION
Real Tet10Element::jacobian(const Real xi[3], const Real* coords, Real* J) const {
    Real dN[NUM_NODES * NUM_DIMS];
    shape_derivatives(xi, dN);

    for (int i = 0; i < 9; ++i) J[i] = 0.0;

    for (int i = 0; i < NUM_NODES; ++i) {
        const Real x = coords[i*3 + 0];
        const Real y = coords[i*3 + 1];
        const Real z = coords[i*3 + 2];

        const Real dNdxi   = dN[i*3 + 0];
        const Real dNdeta  = dN[i*3 + 1];
        const Real dNdzeta = dN[i*3 + 2];

        J[0] += dNdxi * x; J[1] += dNdeta * x; J[2] += dNdzeta * x;
        J[3] += dNdxi * y; J[4] += dNdeta * y; J[5] += dNdzeta * y;
        J[6] += dNdxi * z; J[7] += dNdeta * z; J[8] += dNdzeta * z;
    }

    return J[0] * (J[4] * J[8] - J[5] * J[7])
         - J[1] * (J[3] * J[8] - J[5] * J[6])
         + J[2] * (J[3] * J[7] - J[4] * J[6]);
}

void Tet10Element::inverse_jacobian(const Real* J, Real* J_inv, Real det_J) const {
    const Real inv_det = 1.0 / det_J;
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

Real Tet10Element::shape_derivatives_global(const Real xi[3], const Real* coords, Real* dNdx) const {
    Real dN[NUM_NODES * NUM_DIMS];
    shape_derivatives(xi, dN);

    Real J[9];
    const Real det_J = jacobian(xi, coords, J);

    Real J_inv[9];
    inverse_jacobian(J, J_inv, det_J);

    for (int i = 0; i < NUM_NODES; ++i) {
        const Real dNdxi   = dN[i*3 + 0];
        const Real dNdeta  = dN[i*3 + 1];
        const Real dNdzeta = dN[i*3 + 2];

        // dN/dx_i = Σ_j (J^{-1})_{ji} * dN/dξ_j  (use columns of J_inv = J^{-T})
        dNdx[i*3 + 0] = J_inv[0] * dNdxi + J_inv[3] * dNdeta + J_inv[6] * dNdzeta;
        dNdx[i*3 + 1] = J_inv[1] * dNdxi + J_inv[4] * dNdeta + J_inv[7] * dNdzeta;
        dNdx[i*3 + 2] = J_inv[2] * dNdxi + J_inv[5] * dNdeta + J_inv[8] * dNdzeta;
    }

    return det_J;
}

KOKKOS_INLINE_FUNCTION
void Tet10Element::strain_displacement_matrix(const Real xi[3], const Real* coords, Real* B) const {
    for (int i = 0; i < 6 * NUM_DOF; ++i) B[i] = 0.0;

    Real dNdx[NUM_NODES * NUM_DIMS];
    shape_derivatives_global(xi, coords, dNdx);

    for (int i = 0; i < NUM_NODES; ++i) {
        const Real dNdx_i = dNdx[i*3 + 0];
        const Real dNdy_i = dNdx[i*3 + 1];
        const Real dNdz_i = dNdx[i*3 + 2];
        const int col = i * 3;

        B[0 * NUM_DOF + col + 0] = dNdx_i;
        B[1 * NUM_DOF + col + 1] = dNdy_i;
        B[2 * NUM_DOF + col + 2] = dNdz_i;
        B[3 * NUM_DOF + col + 0] = dNdy_i;
        B[3 * NUM_DOF + col + 1] = dNdx_i;
        B[4 * NUM_DOF + col + 1] = dNdz_i;
        B[4 * NUM_DOF + col + 2] = dNdy_i;
        B[5 * NUM_DOF + col + 0] = dNdz_i;
        B[5 * NUM_DOF + col + 2] = dNdx_i;
    }
}

void Tet10Element::constitutive_matrix(Real E, Real nu, Real* C) const {
    for (int i = 0; i < 36; ++i) C[i] = 0.0;

    const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const Real mu = E / (2.0 * (1.0 + nu));
    const Real diag = lambda + 2.0 * mu;

    C[0*6 + 0] = diag; C[1*6 + 1] = diag; C[2*6 + 2] = diag;
    C[3*6 + 3] = mu; C[4*6 + 4] = mu; C[5*6 + 5] = mu;
    C[0*6 + 1] = lambda; C[0*6 + 2] = lambda;
    C[1*6 + 0] = lambda; C[1*6 + 2] = lambda;
    C[2*6 + 0] = lambda; C[2*6 + 1] = lambda;
}

void Tet10Element::mass_matrix(const Real* coords, Real density, Real* M) const {
    for (int i = 0; i < NUM_DOF * NUM_DOF; ++i) M[i] = 0.0;

    Real gp[4*3], gw[4];
    compute_gauss_points_4pt(gp, gw);

    for (int ig = 0; ig < 4; ++ig) {
        Real xi[3] = {gp[ig*3 + 0], gp[ig*3 + 1], gp[ig*3 + 2]};
        const Real weight = gw[ig];

        Real N[NUM_NODES];
        shape_functions(xi, N);

        Real J[9];
        const Real det_J = jacobian(xi, coords, J);
        // Factor of 1/6 for tetrahedral barycentric coordinates
        // Use abs(det_J) so mass is positive regardless of node orientation
        const Real factor = density * weight * std::abs(det_J) / 6.0;

        for (int i = 0; i < NUM_NODES; ++i) {
            for (int j = 0; j < NUM_NODES; ++j) {
                const Real mass_ij = factor * N[i] * N[j];
                // Distribute mass equally across 3 spatial DOFs per node
                for (int d = 0; d < NUM_DIMS; ++d) {
                    const int row = i * NUM_DIMS + d;
                    const int col = j * NUM_DIMS + d;
                    M[row * NUM_DOF + col] += mass_ij / NUM_DIMS;
                }
            }
        }
    }
}

void Tet10Element::stiffness_matrix(const Real* coords, Real E, Real nu, Real* K) const {
    for (int i = 0; i < NUM_DOF * NUM_DOF; ++i) K[i] = 0.0;

    Real C[36];
    constitutive_matrix(E, nu, C);

    Real gp[4*3], gw[4];
    compute_gauss_points_4pt(gp, gw);

    Real B[6 * NUM_DOF];
    Real CB[6 * NUM_DOF];

    for (int ig = 0; ig < 4; ++ig) {
        Real xi[3] = {gp[ig*3 + 0], gp[ig*3 + 1], gp[ig*3 + 2]};
        const Real weight = gw[ig];

        strain_displacement_matrix(xi, coords, B);

        Real J[9];
        const Real det_J = jacobian(xi, coords, J);

        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < NUM_DOF; ++j) {
                CB[i * NUM_DOF + j] = 0.0;
                for (int k = 0; k < 6; ++k) {
                    CB[i * NUM_DOF + j] += C[i * 6 + k] * B[k * NUM_DOF + j];
                }
            }
        }

        // Factor of 1/6 for tetrahedral barycentric coordinates
        // Use abs(det_J) so stiffness is positive regardless of node orientation
        const Real factor = weight * std::abs(det_J) / 6.0;

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

KOKKOS_INLINE_FUNCTION
void Tet10Element::internal_force(const Real* coords, const Real* disp,
                                   const Real* stress, Real* fint) const {
    for (int i = 0; i < NUM_DOF; ++i) fint[i] = 0.0;

    Real xi[3] = {0.25, 0.25, 0.25};  // Centroid

    Real B[6 * NUM_DOF];
    strain_displacement_matrix(xi, coords, B);

    Real J[9];
    const Real det_J = jacobian(xi, coords, J);
    // Factor of 1/6 for tetrahedral barycentric coordinates
    const Real factor = det_J / 6.0;

    for (int i = 0; i < NUM_DOF; ++i) {
        Real sum = 0.0;
        for (int j = 0; j < 6; ++j) {
            sum += B[j * NUM_DOF + i] * stress[j];
        }
        fint[i] = factor * sum;
    }
}

bool Tet10Element::contains_point(const Real* coords, const Real* point, Real* xi) const {
    // Newton-Raphson iteration
    xi[0] = 0.25; xi[1] = 0.25; xi[2] = 0.25;

    for (int iter = 0; iter < 20; ++iter) {
        Real N[NUM_NODES];
        shape_functions(xi, N);

        Real x_current[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < NUM_NODES; ++i) {
            x_current[0] += N[i] * coords[i*3 + 0];
            x_current[1] += N[i] * coords[i*3 + 1];
            x_current[2] += N[i] * coords[i*3 + 2];
        }

        Real r[3] = {point[0] - x_current[0], point[1] - x_current[1], point[2] - x_current[2]};
        const Real r_norm = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        if (r_norm < 1.0e-6) {
            const Real L4 = 1.0 - xi[0] - xi[1] - xi[2];
            return (xi[0] >= -1e-6 && xi[1] >= -1e-6 && xi[2] >= -1e-6 && L4 >= -1e-6);
        }

        Real J[9];
        jacobian(xi, coords, J);
        const Real det_J = J[0] * (J[4] * J[8] - J[5] * J[7])
                         - J[1] * (J[3] * J[8] - J[5] * J[6])
                         + J[2] * (J[3] * J[7] - J[4] * J[6]);

        if (std::abs(det_J) < 1.0e-12) return false;

        const Real inv_det = 1.0 / det_J;
        xi[0] += inv_det * ((J[4]*J[8] - J[5]*J[7])*r[0] + (J[2]*J[7] - J[1]*J[8])*r[1] + (J[1]*J[5] - J[2]*J[4])*r[2]);
        xi[1] += inv_det * ((J[5]*J[6] - J[3]*J[8])*r[0] + (J[0]*J[8] - J[2]*J[6])*r[1] + (J[2]*J[3] - J[0]*J[5])*r[2]);
        xi[2] += inv_det * ((J[3]*J[7] - J[4]*J[6])*r[0] + (J[1]*J[6] - J[0]*J[7])*r[1] + (J[0]*J[4] - J[1]*J[3])*r[2]);
    }
    return false;
}

Real Tet10Element::volume(const Real* coords) const {
    // Volume = |det(J)| / 6 for tetrahedron
    Real gp[4*3], gw[4];
    compute_gauss_points_4pt(gp, gw);

    Real vol = 0.0;
    for (int ig = 0; ig < 4; ++ig) {
        Real xi[3] = {gp[ig*3 + 0], gp[ig*3 + 1], gp[ig*3 + 2]};
        Real J[9];
        const Real det_J = jacobian(xi, coords, J);
        vol += gw[ig] * det_J;
    }

    return std::abs(vol) / 6.0;  // Factor of 6 for tetrahedral natural coordinates
}

Real Tet10Element::characteristic_length(const Real* coords) const {
    // For Tet10 quadratic elements, mid-edge nodes split each edge in two.
    // The CFL condition requires the time step to be based on the MINIMUM
    // distance between any two adjacent nodes (corner to mid-edge).
    //
    // Mid-edge node mapping:
    // Node 4: edge 0-1    Node 5: edge 1-2    Node 6: edge 0-2
    // Node 7: edge 0-3    Node 8: edge 1-3    Node 9: edge 2-3

    // Check corner-to-midedge distances (12 half-edges)
    const int half_edges[12][2] = {
        {0, 4}, {4, 1},    // edge 0-1
        {1, 5}, {5, 2},    // edge 1-2
        {0, 6}, {6, 2},    // edge 0-2
        {0, 7}, {7, 3},    // edge 0-3
        {1, 8}, {8, 3},    // edge 1-3
        {2, 9}, {9, 3}     // edge 2-3
    };

    Real min_length = 1.0e30;

    for (int ie = 0; ie < 12; ++ie) {
        const int n0 = half_edges[ie][0];
        const int n1 = half_edges[ie][1];
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
