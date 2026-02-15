/**
 * @file shell4.cpp
 * @brief Implementation of 4-node quadrilateral shell element
 *
 * Note: This is a simplified flat shell formulation suitable for
 * thin plates and shells without large rotations.
 */

#include <nexussim/discretization/shell4.hpp>
#include <cmath>
#include <algorithm>

namespace nxs {
namespace fem {

// ============================================================================
// Shape Functions
// ============================================================================

KOKKOS_INLINE_FUNCTION
void Shell4Element::shape_functions(const Real xi[3], Real* N) const {
    // Bilinear shape functions (same as 2D quad)
    const Real xi_val = xi[0];
    const Real eta = xi[1];

    N[0] = 0.25 * (1.0 - xi_val) * (1.0 - eta);  // Node 0: (-1,-1)
    N[1] = 0.25 * (1.0 + xi_val) * (1.0 - eta);  // Node 1: (+1,-1)
    N[2] = 0.25 * (1.0 + xi_val) * (1.0 + eta);  // Node 2: (+1,+1)
    N[3] = 0.25 * (1.0 - xi_val) * (1.0 + eta);  // Node 3: (-1,+1)
}

KOKKOS_INLINE_FUNCTION
void Shell4Element::shape_derivatives(const Real xi[3], Real* dN) const {
    // Derivatives w.r.t. ξ and η
    const Real xi_val = xi[0];
    const Real eta = xi[1];

    // dN/dξ
    dN[0*3 + 0] = -0.25 * (1.0 - eta);
    dN[1*3 + 0] =  0.25 * (1.0 - eta);
    dN[2*3 + 0] =  0.25 * (1.0 + eta);
    dN[3*3 + 0] = -0.25 * (1.0 + eta);

    // dN/dη
    dN[0*3 + 1] = -0.25 * (1.0 - xi_val);
    dN[1*3 + 1] = -0.25 * (1.0 + xi_val);
    dN[2*3 + 1] =  0.25 * (1.0 + xi_val);
    dN[3*3 + 1] =  0.25 * (1.0 - xi_val);

    // dN/dζ (not used for shells, set to zero)
    dN[0*3 + 2] = 0.0;
    dN[1*3 + 2] = 0.0;
    dN[2*3 + 2] = 0.0;
    dN[3*3 + 2] = 0.0;
}

// ============================================================================
// Gauss Quadrature
// ============================================================================

void Shell4Element::compute_gauss_points_2x2(Real* points, Real* weights) const {
    // 2x2 Gauss quadrature
    const Real a = 1.0 / std::sqrt(3.0);  // ±0.577350269

    const Real gp[4][2] = {
        {-a, -a},
        { a, -a},
        { a,  a},
        {-a,  a}
    };

    for (int i = 0; i < 4; ++i) {
        points[i*3 + 0] = gp[i][0];  // ξ
        points[i*3 + 1] = gp[i][1];  // η
        points[i*3 + 2] = 0.0;       // ζ (not used)
        weights[i] = 1.0;
    }
}

void Shell4Element::gauss_quadrature(Real* points, Real* weights) const {
    compute_gauss_points_2x2(points, weights);
}

// ============================================================================
// Jacobian Computation
// ============================================================================

KOKKOS_INLINE_FUNCTION
Real Shell4Element::jacobian(const Real xi[3], const Real* coords, Real* J) const {
    // Compute 2D Jacobian for in-plane mapping
    Real dN[NUM_NODES * 3];
    shape_derivatives(xi, dN);

    // Initialize Jacobian
    for (int i = 0; i < 9; ++i) {
        J[i] = 0.0;
    }

    // J = [∂x/∂ξ  ∂x/∂η]
    //     [∂y/∂ξ  ∂y/∂η]
    //     [∂z/∂ξ  ∂z/∂η]

    for (int i = 0; i < NUM_NODES; ++i) {
        const Real x = coords[i*3 + 0];
        const Real y = coords[i*3 + 1];
        const Real z = coords[i*3 + 2];

        const Real dNdxi  = dN[i*3 + 0];
        const Real dNdeta = dN[i*3 + 1];

        // Column 1: ∂/∂ξ
        J[0] += dNdxi * x;
        J[3] += dNdxi * y;
        J[6] += dNdxi * z;

        // Column 2: ∂/∂η
        J[1] += dNdeta * x;
        J[4] += dNdeta * y;
        J[7] += dNdeta * z;
    }

    // For shell, determinant is area element
    // det = |∂r/∂ξ × ∂r/∂η|
    const Real dx_dxi  = J[0];
    const Real dy_dxi  = J[3];
    const Real dz_dxi  = J[6];
    const Real dx_deta = J[1];
    const Real dy_deta = J[4];
    const Real dz_deta = J[7];

    // Cross product
    const Real nx = dy_dxi * dz_deta - dz_dxi * dy_deta;
    const Real ny = dz_dxi * dx_deta - dx_dxi * dz_deta;
    const Real nz = dx_dxi * dy_deta - dy_dxi * dx_deta;

    const Real det_J = std::sqrt(nx*nx + ny*ny + nz*nz);

    // Store normal in third column
    if (det_J > 1.0e-12) {
        J[2] = nx / det_J;
        J[5] = ny / det_J;
        J[8] = nz / det_J;
    }

    return det_J;
}

// ============================================================================
// B-Matrix (Simplified for flat shells)
// ============================================================================

KOKKOS_INLINE_FUNCTION
void Shell4Element::strain_displacement_matrix(const Real xi[3], const Real* coords, Real* B) const {
    // Simplified B-matrix for flat shell
    // Full shell would need membrane + bending + shear components
    // This implementation provides membrane part only

    // Initialize B to zero
    for (int i = 0; i < 6 * NUM_DOF; ++i) {
        B[i] = 0.0;
    }

    // Get shape function derivatives
    Real dN[NUM_NODES * 3];
    shape_derivatives(xi, dN);

    // Compute Jacobian
    Real J[9];
    const Real det_J = jacobian(xi, coords, J);

    if (det_J < 1.0e-12) return;

    // For membrane behavior (simplified - in-plane only)
    // Would need coordinate transformation for general shells

    for (int i = 0; i < NUM_NODES; ++i) {
        const Real dNdxi  = dN[i*3 + 0];
        const Real dNdeta = dN[i*3 + 1];

        // Simplified: assumes local coords align with global
        // Full implementation would transform to shell local system

        const int col = i * DOF_PER_NODE;  // 6 DOFs per node

        // In-plane strains (εxx, εyy, γxy) - using first 3 translational DOFs
        B[0 * NUM_DOF + col + 0] = dNdxi;   // εxx from ux
        B[1 * NUM_DOF + col + 1] = dNdeta;  // εyy from uy
        B[3 * NUM_DOF + col + 0] = dNdeta;  // γxy from ux
        B[3 * NUM_DOF + col + 1] = dNdxi;   // γxy from uy
    }
}

// ============================================================================
// Local Coordinate System
// ============================================================================

void Shell4Element::local_coordinate_system(const Real* coords, Real* e1, Real* e2, Real* e3) const {
    // Compute local coordinate system
    // e1: along edge 0-1
    // e3: normal to shell (cross product)
    // e2: e3 × e1

    // e1 direction: node 0 to node 1
    e1[0] = coords[3] - coords[0];
    e1[1] = coords[4] - coords[1];
    e1[2] = coords[5] - coords[2];

    Real len_e1 = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
    if (len_e1 > 1.0e-12) {
        e1[0] /= len_e1;
        e1[1] /= len_e1;
        e1[2] /= len_e1;
    }

    // Temporary vector: node 0 to node 3
    Real v[3];
    v[0] = coords[9] - coords[0];
    v[1] = coords[10] - coords[1];
    v[2] = coords[11] - coords[2];

    // e3 = e1 × v (normal)
    e3[0] = e1[1] * v[2] - e1[2] * v[1];
    e3[1] = e1[2] * v[0] - e1[0] * v[2];
    e3[2] = e1[0] * v[1] - e1[1] * v[0];

    Real len_e3 = std::sqrt(e3[0]*e3[0] + e3[1]*e3[1] + e3[2]*e3[2]);
    if (len_e3 > 1.0e-12) {
        e3[0] /= len_e3;
        e3[1] /= len_e3;
        e3[2] /= len_e3;
    }

    // e2 = e3 × e1
    e2[0] = e3[1] * e1[2] - e3[2] * e1[1];
    e2[1] = e3[2] * e1[0] - e3[0] * e1[2];
    e2[2] = e3[0] * e1[1] - e3[1] * e1[0];
}

// ============================================================================
// Mass Matrix
// ============================================================================

void Shell4Element::lumped_mass_matrix(const Real* coords, Real density, Real* M) const {
    const Real area = volume(coords);  // Actually area for shells
    const Real total_mass = density * area * thickness_;
    const Real nodal_mass = total_mass / NUM_NODES;

    // Distribute mass to translational DOFs only
    for (int i = 0; i < NUM_NODES; ++i) {
        // Each translational DOF gets equal share of nodal mass
        for (int d = 0; d < 3; ++d) {  // Only translational DOFs get mass
            M[i * DOF_PER_NODE + d] = nodal_mass / 3.0;
        }
        // Rotational DOFs get rotational inertia
        for (int d = 3; d < 6; ++d) {
            M[i * DOF_PER_NODE + d] = nodal_mass * thickness_ * thickness_ / 12.0 / 3.0;
        }
    }
}

void Shell4Element::mass_matrix(const Real* coords, Real density, Real* M) const {
    // Simplified: use lumped mass on diagonal
    for (int i = 0; i < NUM_DOF * NUM_DOF; ++i) {
        M[i] = 0.0;
    }

    Real M_lumped[NUM_DOF];
    lumped_mass_matrix(coords, density, M_lumped);

    for (int i = 0; i < NUM_DOF; ++i) {
        M[i * NUM_DOF + i] = M_lumped[i];
    }
}

// ============================================================================
// Stiffness Matrix
// ============================================================================

void Shell4Element::stiffness_matrix(const Real* coords, Real E, Real nu, Real* K) const {
    // Flat shell stiffness: membrane + bending + drilling DOF
    // DOFs per node: (u, v, w, θx, θy, θz) in local coordinates
    // Membrane: u, v (in-plane translations)
    // Bending: w (out-of-plane), θx, θy (rotations)
    // Drilling: θz (in-plane rotation) - small stiffness to avoid singularity

    for (int i = 0; i < NUM_DOF * NUM_DOF; ++i) {
        K[i] = 0.0;
    }

    // Compute local coordinate system
    Real e1[3], e2[3], e3[3];
    local_coordinate_system(coords, e1, e2, e3);

    // Transform nodal coordinates to local system
    // Compute centroid
    Real cx = 0.0, cy = 0.0, cz = 0.0;
    for (int i = 0; i < NUM_NODES; ++i) {
        cx += coords[i*3 + 0];
        cy += coords[i*3 + 1];
        cz += coords[i*3 + 2];
    }
    cx /= NUM_NODES; cy /= NUM_NODES; cz /= NUM_NODES;

    // Local 2D coordinates of each node
    Real xl[4], yl[4];
    for (int i = 0; i < NUM_NODES; ++i) {
        Real dx = coords[i*3+0] - cx;
        Real dy = coords[i*3+1] - cy;
        Real dz = coords[i*3+2] - cz;
        xl[i] = e1[0]*dx + e1[1]*dy + e1[2]*dz;
        yl[i] = e2[0]*dx + e2[1]*dy + e2[2]*dz;
    }

    // Material matrices
    const Real t = thickness_;
    const Real fac = E / (1.0 - nu * nu);

    // Plane stress D_m (3x3): for membrane [σxx, σyy, τxy]
    Real Dm[9] = {0};
    Dm[0] = fac * t;            // D11
    Dm[1] = fac * nu * t;       // D12
    Dm[3] = fac * nu * t;       // D21
    Dm[4] = fac * t;            // D22
    Dm[8] = fac * (1.0 - nu) / 2.0 * t;  // D33 (shear)

    // Bending D_b (3x3): [Mxx, Myy, Mxy]
    Real Db[9] = {0};
    const Real bfac = fac * t * t * t / 12.0;
    Db[0] = bfac;               // D11
    Db[1] = bfac * nu;          // D12
    Db[3] = bfac * nu;          // D21
    Db[4] = bfac;               // D22
    Db[8] = bfac * (1.0 - nu) / 2.0;  // D33

    // 2x2 Gauss quadrature
    Real gp[12], gw[4];
    gauss_quadrature(gp, gw);

    // Shear modulus and area for drilling stiffness
    const Real G = E / (2.0 * (1.0 + nu));
    const Real area = volume(coords);

    for (int ig = 0; ig < 4; ++ig) {
        const Real xi = gp[ig*3 + 0];
        const Real eta = gp[ig*3 + 1];

        // Shape function derivatives w.r.t. natural coordinates
        Real dNdxi[4], dNdeta[4];
        dNdxi[0] = -0.25 * (1.0 - eta);
        dNdxi[1] =  0.25 * (1.0 - eta);
        dNdxi[2] =  0.25 * (1.0 + eta);
        dNdxi[3] = -0.25 * (1.0 + eta);

        dNdeta[0] = -0.25 * (1.0 - xi);
        dNdeta[1] = -0.25 * (1.0 + xi);
        dNdeta[2] =  0.25 * (1.0 + xi);
        dNdeta[3] =  0.25 * (1.0 - xi);

        // 2D Jacobian in local coordinates: J = [dx/dxi dx/deta; dy/dxi dy/deta]
        Real J11 = 0, J12 = 0, J21 = 0, J22 = 0;
        for (int i = 0; i < NUM_NODES; ++i) {
            J11 += dNdxi[i] * xl[i];
            J12 += dNdeta[i] * xl[i];
            J21 += dNdxi[i] * yl[i];
            J22 += dNdeta[i] * yl[i];
        }

        Real det_J = J11 * J22 - J12 * J21;
        if (std::abs(det_J) < 1.0e-20) continue;

        Real inv_det = 1.0 / det_J;

        // Shape function derivatives w.r.t. local physical coordinates
        Real dNdx[4], dNdy[4];
        for (int i = 0; i < NUM_NODES; ++i) {
            dNdx[i] = inv_det * ( J22 * dNdxi[i] - J21 * dNdeta[i]);
            dNdy[i] = inv_det * (-J12 * dNdxi[i] + J11 * dNdeta[i]);
        }

        Real weight = gw[ig] * std::abs(det_J);

        // ---- Membrane part: K_m = integral(B_m^T * D_m * B_m) ----
        // B_m is 3x(2*4) = 3x8: maps (u1,v1, u2,v2, ...) to (εxx, εyy, γxy)
        // But in 6-DOF layout, u maps to dof 0, v to dof 1 of each node
        for (int i = 0; i < NUM_NODES; ++i) {
            for (int j = 0; j < NUM_NODES; ++j) {
                // B_m^T * D_m * B_m contribution
                // Row/Col indices in the membrane sub-matrix
                Real Bi_T_D_Bj[2][2] = {0};

                // εxx row: dNdx  -> u
                // εyy row: dNdy  -> v
                // γxy row: dNdy,dNdx -> u,v

                // k_uu = dNi/dx * D11 * dNj/dx + dNi/dy * D33 * dNj/dy
                Bi_T_D_Bj[0][0] = dNdx[i] * Dm[0] * dNdx[j] + dNdy[i] * Dm[8] * dNdy[j];
                // k_uv = dNi/dx * D12 * dNj/dy + dNi/dy * D33 * dNj/dx
                Bi_T_D_Bj[0][1] = dNdx[i] * Dm[1] * dNdy[j] + dNdy[i] * Dm[8] * dNdx[j];
                // k_vu = dNi/dy * D21 * dNj/dx + dNi/dx * D33 * dNj/dy
                Bi_T_D_Bj[1][0] = dNdy[i] * Dm[3] * dNdx[j] + dNdx[i] * Dm[8] * dNdy[j];
                // k_vv = dNi/dy * D22 * dNj/dy + dNi/dx * D33 * dNj/dx
                Bi_T_D_Bj[1][1] = dNdy[i] * Dm[4] * dNdy[j] + dNdx[i] * Dm[8] * dNdx[j];

                // Map to global DOF layout (6 DOFs/node):
                // local u -> DOF 0, local v -> DOF 1
                for (int di = 0; di < 2; ++di) {
                    for (int dj = 0; dj < 2; ++dj) {
                        int row = i * DOF_PER_NODE + di;
                        int col = j * DOF_PER_NODE + dj;
                        K[row * NUM_DOF + col] += weight * Bi_T_D_Bj[di][dj];
                    }
                }
            }
        }

        // ---- Bending part: K_b = integral(B_b^T * D_b * B_b) ----
        // For Kirchhoff plate (thin shell), bending curvatures:
        //   κxx = -∂²w/∂x² ≈ dNdx * θy  (using small rotation: θy ≈ ∂w/∂x)
        //   κyy = -∂²w/∂y² ≈ -dNdy * θx
        //   κxy = -2*∂²w/∂x∂y ≈ dNdy * θy - dNdx * θx
        // B_b maps (θx, θy) per node to (κxx, κyy, 2*κxy)
        // Using DKQ-like approach with bilinear interpolation of rotations:
        //   κxx = Σ dNi/dx * θyi
        //   κyy = -Σ dNi/dy * θxi
        //   κxy = Σ (dNi/dy * θyi - dNi/dx * θxi)
        for (int i = 0; i < NUM_NODES; ++i) {
            for (int j = 0; j < NUM_NODES; ++j) {
                // B_b for node i:
                //   [  0      dNi/dx  ]   (κxx)
                //   [ -dNi/dy   0     ]   (κyy)
                //   [ -dNi/dx  dNi/dy ]   (κxy)
                // columns correspond to θx, θy

                Real Bi_00 = 0.0,       Bi_01 = dNdx[i];   // row 0 (κxx)
                Real Bi_10 = -dNdy[i],  Bi_11 = 0.0;       // row 1 (κyy)
                Real Bi_20 = -dNdx[i],  Bi_21 = dNdy[i];   // row 2 (κxy)

                Real Bj_00 = 0.0,       Bj_01 = dNdx[j];
                Real Bj_10 = -dNdy[j],  Bj_11 = 0.0;
                Real Bj_20 = -dNdx[j],  Bj_21 = dNdy[j];

                // Compute B_i^T * D_b * B_j  (2x2)
                // First column of D*Bj: D * [Bj_00, Bj_10, Bj_20]^T
                Real DBj_00 = Db[0]*Bj_00 + Db[1]*Bj_10 + 0;
                Real DBj_10 = Db[3]*Bj_00 + Db[4]*Bj_10 + 0;
                Real DBj_20 = 0            + 0            + Db[8]*Bj_20;

                // Second column of D*Bj: D * [Bj_01, Bj_11, Bj_21]^T
                Real DBj_01 = Db[0]*Bj_01 + Db[1]*Bj_11 + 0;
                Real DBj_11 = Db[3]*Bj_01 + Db[4]*Bj_11 + 0;
                Real DBj_21 = 0            + 0            + Db[8]*Bj_21;

                // B_i^T * (D*B_j)
                Real k_txi_txj = Bi_00*DBj_00 + Bi_10*DBj_10 + Bi_20*DBj_20;
                Real k_txi_tyj = Bi_00*DBj_01 + Bi_10*DBj_11 + Bi_20*DBj_21;
                Real k_tyi_txj = Bi_01*DBj_00 + Bi_11*DBj_10 + Bi_21*DBj_20;
                Real k_tyi_tyj = Bi_01*DBj_01 + Bi_11*DBj_11 + Bi_21*DBj_21;

                // Map to global DOFs: θx -> DOF 3, θy -> DOF 4
                int r3 = i * DOF_PER_NODE + 3;
                int r4 = i * DOF_PER_NODE + 4;
                int c3 = j * DOF_PER_NODE + 3;
                int c4 = j * DOF_PER_NODE + 4;

                K[r3 * NUM_DOF + c3] += weight * k_txi_txj;
                K[r3 * NUM_DOF + c4] += weight * k_txi_tyj;
                K[r4 * NUM_DOF + c3] += weight * k_tyi_txj;
                K[r4 * NUM_DOF + c4] += weight * k_tyi_tyj;
            }
        }

        // ---- Bending-transverse coupling: w relates to θx, θy ----
        // For thin shells, w (DOF 2) couples to rotations through shear
        // Use a shear correction approach: K_shear = κ*G*t * integral(B_s^T * B_s)
        // where κ = 5/6 is the shear correction factor
        // B_s maps (w, θx, θy) to (γxz, γyz):
        //   γxz = ∂w/∂x + θy   -> dNdx * w + N * θy
        //   γyz = ∂w/∂y - θx   -> dNdy * w - N * θx
        // Using selective reduced integration would be better, but
        // for simplicity use full integration with a scaled shear stiffness
        Real N_val[4];
        N_val[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
        N_val[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
        N_val[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
        N_val[3] = 0.25 * (1.0 - xi) * (1.0 + eta);

        const Real kappa = 5.0 / 6.0;
        const Real Ds = kappa * G * t;

        for (int i = 0; i < NUM_NODES; ++i) {
            for (int j = 0; j < NUM_NODES; ++j) {
                // B_s for node i: (for γxz and γyz)
                // γxz: [dNi/dx, 0, Ni]     maps (w, θx, θy)
                // γyz: [dNi/dy, -Ni, 0]     maps (w, θx, θy)

                // k_ww = Ds * (dNi/dx*dNj/dx + dNi/dy*dNj/dy)
                Real k_ww = Ds * (dNdx[i]*dNdx[j] + dNdy[i]*dNdy[j]);
                // k_w_tx = Ds * (-dNi/dy*Nj)   (from γyz)
                Real k_w_tx = Ds * (-dNdy[i]*N_val[j]);
                // k_w_ty = Ds * (dNi/dx*Nj)    (from γxz)
                Real k_w_ty = Ds * (dNdx[i]*N_val[j]);
                // k_tx_w = Ds * (-Ni*dNj/dy)
                Real k_tx_w = Ds * (-N_val[i]*dNdy[j]);
                // k_tx_tx = Ds * (Ni*Nj)         (from γyz*γyz)
                Real k_tx_tx = Ds * (N_val[i]*N_val[j]);
                // k_ty_w = Ds * (Ni*dNj/dx)
                Real k_ty_w = Ds * (N_val[i]*dNdx[j]);
                // k_ty_ty = Ds * (Ni*Nj)          (from γxz*γxz)
                Real k_ty_ty = Ds * (N_val[i]*N_val[j]);

                int rw = i * DOF_PER_NODE + 2;
                int rtx = i * DOF_PER_NODE + 3;
                int rty = i * DOF_PER_NODE + 4;
                int cw = j * DOF_PER_NODE + 2;
                int ctx = j * DOF_PER_NODE + 3;
                int cty = j * DOF_PER_NODE + 4;

                K[rw * NUM_DOF + cw]   += weight * k_ww;
                K[rw * NUM_DOF + ctx]  += weight * k_w_tx;
                K[rw * NUM_DOF + cty]  += weight * k_w_ty;
                K[rtx * NUM_DOF + cw]  += weight * k_tx_w;
                K[rtx * NUM_DOF + ctx] += weight * k_tx_tx;
                K[rty * NUM_DOF + cw]  += weight * k_ty_w;
                K[rty * NUM_DOF + cty] += weight * k_ty_ty;
            }
        }
    }

    // ---- Drilling DOF stiffness: prevent singularity in θz ----
    // Small stiffness α*G*t*A/4 per node
    const Real alpha_drill = 1.0e-3;
    const Real k_drill = alpha_drill * G * t * area / 4.0;
    for (int i = 0; i < NUM_NODES; ++i) {
        int dof = i * DOF_PER_NODE + 5;  // θz
        K[dof * NUM_DOF + dof] += k_drill;
    }
}

// ============================================================================
// Internal Force
// ============================================================================

KOKKOS_INLINE_FUNCTION
void Shell4Element::internal_force(const Real* coords, const Real* disp,
                                    const Real* stress, Real* fint) const {
    // Initialize
    for (int i = 0; i < NUM_DOF; ++i) {
        fint[i] = 0.0;
    }

    // Simplified: use B-matrix at center
    Real xi[3] = {0.0, 0.0, 0.0};
    Real B[6 * NUM_DOF];
    strain_displacement_matrix(xi, coords, B);

    // Compute area inline (simple cross product for quad)
    Real v1[3], v2[3];
    v1[0] = coords[3] - coords[0];
    v1[1] = coords[4] - coords[1];
    v1[2] = coords[5] - coords[2];
    v2[0] = coords[6] - coords[0];
    v2[1] = coords[7] - coords[1];
    v2[2] = coords[8] - coords[2];
    Real cross[3];
    cross[0] = v1[1] * v2[2] - v1[2] * v2[1];
    cross[1] = v1[2] * v2[0] - v1[0] * v2[2];
    cross[2] = v1[0] * v2[1] - v1[1] * v2[0];
    const Real area = std::sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);

    // f_int = B^T * σ * area
    for (int i = 0; i < NUM_DOF; ++i) {
        Real sum = 0.0;
        for (int j = 0; j < 6; ++j) {
            sum += B[j * NUM_DOF + i] * stress[j];
        }
        fint[i] = area * thickness_ * sum;
    }
}

// ============================================================================
// Geometric Queries
// ============================================================================

bool Shell4Element::contains_point(const Real* coords, const Real* point, Real* xi) const {
    // Project point onto shell plane and check if inside quad
    // Simplified implementation

    xi[0] = 0.0;
    xi[1] = 0.0;
    xi[2] = 0.0;

    // Use Newton-Raphson in 2D
    for (int iter = 0; iter < 20; ++iter) {
        Real N[4];
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
        if (r_norm < 1.0e-6) {
            return (std::abs(xi[0]) <= 1.0 && std::abs(xi[1]) <= 1.0);
        }

        // Update (simplified)
        xi[0] += r[0] * 0.1;
        xi[1] += r[1] * 0.1;
    }

    return false;
}

Real Shell4Element::volume(const Real* coords) const {
    // For shell, "volume" is actually area
    // Compute as area of two triangles

    // Triangle 0-1-2
    Real v1[3], v2[3];
    v1[0] = coords[3] - coords[0];
    v1[1] = coords[4] - coords[1];
    v1[2] = coords[5] - coords[2];

    v2[0] = coords[6] - coords[0];
    v2[1] = coords[7] - coords[1];
    v2[2] = coords[8] - coords[2];

    Real cross1[3];
    cross1[0] = v1[1] * v2[2] - v1[2] * v2[1];
    cross1[1] = v1[2] * v2[0] - v1[0] * v2[2];
    cross1[2] = v1[0] * v2[1] - v1[1] * v2[0];

    const Real area1 = 0.5 * std::sqrt(cross1[0]*cross1[0] + cross1[1]*cross1[1] + cross1[2]*cross1[2]);

    // Triangle 0-2-3
    v2[0] = coords[9] - coords[0];
    v2[1] = coords[10] - coords[1];
    v2[2] = coords[11] - coords[2];

    Real cross2[3];
    cross2[0] = (coords[6]-coords[0]) * v2[2] - (coords[8]-coords[2]) * v2[1];
    cross2[1] = (coords[8]-coords[2]) * v2[0] - (coords[6]-coords[0]) * v2[2];
    cross2[2] = (coords[6]-coords[0]) * v2[1] - (coords[7]-coords[1]) * v2[0];

    const Real area2 = 0.5 * std::sqrt(cross2[0]*cross2[0] + cross2[1]*cross2[1] + cross2[2]*cross2[2]);

    return area1 + area2;
}

Real Shell4Element::characteristic_length(const Real* coords) const {
    // Minimum edge length
    const int edges[4][2] = {{0,1}, {1,2}, {2,3}, {3,0}};

    Real min_length = 1.0e30;

    for (int ie = 0; ie < 4; ++ie) {
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
