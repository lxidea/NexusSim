/**
 * @file wedge6.cpp
 * @brief Implementation of 6-node wedge/prism element
 */

#include <nexussim/discretization/wedge6.hpp>
#include <cmath>
#include <algorithm>

namespace nxs {
namespace fem {

KOKKOS_INLINE_FUNCTION
void Wedge6Element::shape_functions(const Real xi[3], Real* N) const {
    // xi[0] = ξ, xi[1] = η, xi[2] = ζ
    // Triangular coordinates in base: L1 = 1-ξ-η, L2 = ξ, L3 = η
    // Axial direction: ζ ∈ [-1, 1]

    const Real L1 = 1.0 - xi[0] - xi[1];
    const Real L2 = xi[0];
    const Real L3 = xi[1];
    const Real zeta = xi[2];

    // Bottom triangle (ζ = -1)
    N[0] = L1 * 0.5 * (1.0 - zeta);
    N[1] = L2 * 0.5 * (1.0 - zeta);
    N[2] = L3 * 0.5 * (1.0 - zeta);

    // Top triangle (ζ = +1)
    N[3] = L1 * 0.5 * (1.0 + zeta);
    N[4] = L2 * 0.5 * (1.0 + zeta);
    N[5] = L3 * 0.5 * (1.0 + zeta);
}

KOKKOS_INLINE_FUNCTION
void Wedge6Element::shape_derivatives(const Real xi[3], Real* dN) const {
    const Real zeta = xi[2];

    // dN/dξ, dN/dη, dN/dζ
    // Bottom nodes
    dN[0*3 + 0] = -0.5 * (1.0 - zeta);
    dN[0*3 + 1] = -0.5 * (1.0 - zeta);
    dN[0*3 + 2] = -0.5 * (1.0 - xi[0] - xi[1]);

    dN[1*3 + 0] =  0.5 * (1.0 - zeta);
    dN[1*3 + 1] =  0.0;
    dN[1*3 + 2] = -0.5 * xi[0];

    dN[2*3 + 0] =  0.0;
    dN[2*3 + 1] =  0.5 * (1.0 - zeta);
    dN[2*3 + 2] = -0.5 * xi[1];

    // Top nodes
    dN[3*3 + 0] = -0.5 * (1.0 + zeta);
    dN[3*3 + 1] = -0.5 * (1.0 + zeta);
    dN[3*3 + 2] =  0.5 * (1.0 - xi[0] - xi[1]);

    dN[4*3 + 0] =  0.5 * (1.0 + zeta);
    dN[4*3 + 1] =  0.0;
    dN[4*3 + 2] =  0.5 * xi[0];

    dN[5*3 + 0] =  0.0;
    dN[5*3 + 1] =  0.5 * (1.0 + zeta);
    dN[5*3 + 2] =  0.5 * xi[1];
}

void Wedge6Element::compute_gauss_points_6pt(Real* points, Real* weights) const {
    // 6-point Gauss quadrature for wedge: 3-point in triangle × 2-point in ζ
    // Triangle: 1/3, 1/3, 1/3 at vertices
    const Real a = 1.0 / 6.0;  // Triangle Gauss point
    const Real b = 2.0 / 3.0;
    const Real gz = 1.0 / std::sqrt(3.0);  // ±1/√3 for ζ direction

    // 3 points in triangle × 2 points in ζ = 6 points
    int idx = 0;
    Real tri_pts[3][2] = {{a, a}, {b, a}, {a, b}};
    Real tri_w = 1.0 / 6.0;  // Triangle weight
    Real z_pts[2] = {-gz, gz};
    Real z_w = 1.0;

    for (int iz = 0; iz < 2; ++iz) {
        for (int it = 0; it < 3; ++it) {
            points[idx*3 + 0] = tri_pts[it][0];
            points[idx*3 + 1] = tri_pts[it][1];
            points[idx*3 + 2] = z_pts[iz];
            weights[idx] = tri_w * z_w;
            idx++;
        }
    }
}

void Wedge6Element::gauss_quadrature(Real* points, Real* weights) const {
    compute_gauss_points_6pt(points, weights);
}

KOKKOS_INLINE_FUNCTION
Real Wedge6Element::jacobian(const Real xi[3], const Real* coords, Real* J) const {
    Real dN[NUM_NODES * NUM_DIMS];
    shape_derivatives(xi, dN);

    for (int i = 0; i < 9; ++i) J[i] = 0.0;

    for (int i = 0; i < NUM_NODES; ++i) {
        const Real x = coords[i*3 + 0];
        const Real y = coords[i*3 + 1];
        const Real z = coords[i*3 + 2];

        J[0] += dN[i*3 + 0] * x; J[1] += dN[i*3 + 1] * x; J[2] += dN[i*3 + 2] * x;
        J[3] += dN[i*3 + 0] * y; J[4] += dN[i*3 + 1] * y; J[5] += dN[i*3 + 2] * y;
        J[6] += dN[i*3 + 0] * z; J[7] += dN[i*3 + 1] * z; J[8] += dN[i*3 + 2] * z;
    }

    return J[0] * (J[4] * J[8] - J[5] * J[7])
         - J[1] * (J[3] * J[8] - J[5] * J[6])
         + J[2] * (J[3] * J[7] - J[4] * J[6]);
}

void Wedge6Element::inverse_jacobian(const Real* J, Real* J_inv, Real det_J) const {
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

Real Wedge6Element::shape_derivatives_global(const Real xi[3], const Real* coords, Real* dNdx) const {
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

        dNdx[i*3 + 0] = J_inv[0] * dNdxi + J_inv[1] * dNdeta + J_inv[2] * dNdzeta;
        dNdx[i*3 + 1] = J_inv[3] * dNdxi + J_inv[4] * dNdeta + J_inv[5] * dNdzeta;
        dNdx[i*3 + 2] = J_inv[6] * dNdxi + J_inv[7] * dNdeta + J_inv[8] * dNdzeta;
    }

    return det_J;
}

KOKKOS_INLINE_FUNCTION
void Wedge6Element::strain_displacement_matrix(const Real xi[3], const Real* coords, Real* B) const {
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

void Wedge6Element::constitutive_matrix(Real E, Real nu, Real* C) const {
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

void Wedge6Element::mass_matrix(const Real* coords, Real density, Real* M) const {
    for (int i = 0; i < NUM_DOF * NUM_DOF; ++i) M[i] = 0.0;

    Real gp[6*3], gw[6];
    compute_gauss_points_6pt(gp, gw);

    for (int ig = 0; ig < 6; ++ig) {
        Real xi[3] = {gp[ig*3 + 0], gp[ig*3 + 1], gp[ig*3 + 2]};
        const Real weight = gw[ig];

        Real N[NUM_NODES];
        shape_functions(xi, N);

        Real J[9];
        const Real det_J = jacobian(xi, coords, J);
        const Real factor = density * weight * det_J;

        for (int i = 0; i < NUM_NODES; ++i) {
            for (int j = 0; j < NUM_NODES; ++j) {
                const Real mass_ij = factor * N[i] * N[j];
                for (int d = 0; d < NUM_DIMS; ++d) {
                    const int row = i * NUM_DIMS + d;
                    const int col = j * NUM_DIMS + d;
                    M[row * NUM_DOF + col] += mass_ij;
                }
            }
        }
    }
}

void Wedge6Element::stiffness_matrix(const Real* coords, Real E, Real nu, Real* K) const {
    for (int i = 0; i < NUM_DOF * NUM_DOF; ++i) K[i] = 0.0;

    Real C[36];
    constitutive_matrix(E, nu, C);

    Real gp[6*3], gw[6];
    compute_gauss_points_6pt(gp, gw);

    Real B[6 * NUM_DOF];
    Real CB[6 * NUM_DOF];

    for (int ig = 0; ig < 6; ++ig) {
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

        const Real factor = weight * det_J;

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
void Wedge6Element::internal_force(const Real* coords, const Real* disp,
                                    const Real* stress, Real* fint) const {
    for (int i = 0; i < NUM_DOF; ++i) fint[i] = 0.0;

    Real xi[3] = {1.0/3.0, 1.0/3.0, 0.0};  // Centroid

    Real B[6 * NUM_DOF];
    strain_displacement_matrix(xi, coords, B);

    Real J[9];
    const Real det_J = jacobian(xi, coords, J);
    const Real factor = det_J;

    for (int i = 0; i < NUM_DOF; ++i) {
        Real sum = 0.0;
        for (int j = 0; j < 6; ++j) {
            sum += B[j * NUM_DOF + i] * stress[j];
        }
        fint[i] = factor * sum;
    }
}

bool Wedge6Element::contains_point(const Real* coords, const Real* point, Real* xi) const {
    xi[0] = 1.0/3.0; xi[1] = 1.0/3.0; xi[2] = 0.0;

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
            return (xi[0] >= -1e-6 && xi[1] >= -1e-6 &&
                    xi[0] + xi[1] <= 1.0 + 1e-6 &&
                    std::abs(xi[2]) <= 1.0 + 1e-6);
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

Real Wedge6Element::volume(const Real* coords) const {
    Real gp[6*3], gw[6];
    compute_gauss_points_6pt(gp, gw);

    Real vol = 0.0;
    for (int ig = 0; ig < 6; ++ig) {
        Real xi[3] = {gp[ig*3 + 0], gp[ig*3 + 1], gp[ig*3 + 2]};
        Real J[9];
        const Real det_J = jacobian(xi, coords, J);
        vol += gw[ig] * det_J;
    }
    return vol;
}

Real Wedge6Element::characteristic_length(const Real* coords) const {
    // Minimum edge length
    const int edges[9][2] = {
        {0, 1}, {1, 2}, {2, 0},  // Bottom triangle
        {3, 4}, {4, 5}, {5, 3},  // Top triangle
        {0, 3}, {1, 4}, {2, 5}   // Vertical edges
    };
    Real min_length = 1.0e30;

    for (int ie = 0; ie < 9; ++ie) {
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
