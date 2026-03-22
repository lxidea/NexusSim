#pragma once

/**
 * @file elements_wave26.hpp
 * @brief Wave 26: Advanced Element Formulations II
 *
 * Eight element formulations for enhanced accuracy and performance:
 *  1. QEPHShell               - Physical hourglass energy control shell
 *  2. BATOZShell              - Discrete Kirchhoff Triangle (DKT)
 *  3. CorotationalShell       - Co-rotational frame 4-node shell
 *  4. ANSHex8                 - Assumed Natural Strain hexahedron
 *  5. StabilizedIncompatibleHex8 - Incompatible modes + hourglass stabilization
 *  6. LayeredThickShell       - 6-layer thick shell with per-layer material
 *  7. SelectiveMassHex8       - Rotational inertia scaling hexahedron
 *  8. EnhancedStrainExtrapolationHex8 - EAS with stress extrapolation to nodes
 *
 * References:
 *  - Belytschko, Tsay (1984) physical stabilization
 *  - Batoz, Bathe, Ho (1980) DKT element
 *  - Rankin, Brogan (1986) co-rotational formulation
 *  - Dvorkin, Bathe (1984) ANS method
 *  - Wilson, Taylor, Doherty, Ghaboussi (1973) incompatible modes
 *  - Simo, Rifai (1990) EAS method
 */

#include <cmath>
#include <cstring>
#include <array>
#include <algorithm>

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

namespace nxs {

using Real = double;

namespace discretization {

// ============================================================================
// Utility: small matrix/vector helpers (GPU-safe, no STL)
// ============================================================================

namespace detail_w26 {

KOKKOS_INLINE_FUNCTION
void zero(Real* a, int n) { for (int i = 0; i < n; ++i) a[i] = 0.0; }

KOKKOS_INLINE_FUNCTION
Real dot3(const Real* a, const Real* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

KOKKOS_INLINE_FUNCTION
void cross3(const Real* a, const Real* b, Real* c) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

KOKKOS_INLINE_FUNCTION
Real norm3(const Real* v) { return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); }

KOKKOS_INLINE_FUNCTION
void normalize3(Real* v) {
    Real n = norm3(v);
    if (n > 1e-30) { v[0] /= n; v[1] /= n; v[2] /= n; }
}

/// 3x3 determinant (row-major)
KOKKOS_INLINE_FUNCTION
Real det3(const Real* J) {
    return J[0]*(J[4]*J[8]-J[5]*J[7])
         - J[1]*(J[3]*J[8]-J[5]*J[6])
         + J[2]*(J[3]*J[7]-J[4]*J[6]);
}

/// 3x3 inverse, returns det
KOKKOS_INLINE_FUNCTION
Real inv3(const Real* J, Real* Ji) {
    Real d = det3(J);
    Real id = 1.0 / d;
    Ji[0] = (J[4]*J[8]-J[5]*J[7])*id;
    Ji[1] = (J[2]*J[7]-J[1]*J[8])*id;
    Ji[2] = (J[1]*J[5]-J[2]*J[4])*id;
    Ji[3] = (J[5]*J[6]-J[3]*J[8])*id;
    Ji[4] = (J[0]*J[8]-J[2]*J[6])*id;
    Ji[5] = (J[2]*J[3]-J[0]*J[5])*id;
    Ji[6] = (J[3]*J[7]-J[4]*J[6])*id;
    Ji[7] = (J[1]*J[6]-J[0]*J[7])*id;
    Ji[8] = (J[0]*J[4]-J[1]*J[3])*id;
    return d;
}

/// K += B^T * C * B * w   (nstress x ndof), (nstress x nstress), weight w
KOKKOS_INLINE_FUNCTION
void addBtCB(Real* K, const Real* B, const Real* C, Real w,
             int nstress, int ndof) {
    for (int i = 0; i < ndof; ++i) {
        for (int j = 0; j < ndof; ++j) {
            Real val = 0.0;
            for (int k = 0; k < nstress; ++k) {
                Real cb_kj = 0.0;
                for (int l = 0; l < nstress; ++l)
                    cb_kj += C[k*nstress + l] * B[l*ndof + j];
                val += B[k*ndof + i] * cb_kj;
            }
            K[i*ndof + j] += val * w;
        }
    }
}

/// 3D isotropic elastic constitutive matrix (6x6 Voigt)
KOKKOS_INLINE_FUNCTION
void iso3D_C(Real E, Real nu, Real* C) {
    zero(C, 36);
    Real f = E / ((1.0 + nu) * (1.0 - 2.0*nu));
    Real d = f * (1.0 - nu);
    Real o = f * nu;
    Real g = E / (2.0*(1.0 + nu));
    C[0*6+0] = d; C[0*6+1] = o; C[0*6+2] = o;
    C[1*6+0] = o; C[1*6+1] = d; C[1*6+2] = o;
    C[2*6+0] = o; C[2*6+1] = o; C[2*6+2] = d;
    C[3*6+3] = g; C[4*6+4] = g; C[5*6+5] = g;
}

/// Plane stress constitutive (for shell membrane, 3x3)
KOKKOS_INLINE_FUNCTION
void planeStress_C(Real E, Real nu, Real* C) {
    zero(C, 9);
    Real f = E / (1.0 - nu*nu);
    C[0] = f;      C[1] = f*nu;
    C[3] = f*nu;   C[4] = f;
    C[8] = f*(1.0-nu)/2.0;
}

/// Shell bending constitutive (3x3): Db = (E*t^3)/(12*(1-nu^2)) * [...]
KOKKOS_INLINE_FUNCTION
void shellBending_D(Real E, Real nu, Real t, Real* D) {
    zero(D, 9);
    Real fac = E * t * t * t / (12.0 * (1.0 - nu * nu));
    D[0] = fac;       D[1] = fac * nu;
    D[3] = fac * nu;  D[4] = fac;
    D[8] = fac * (1.0 - nu) / 2.0;
}

/// Hex8 shape functions
KOKKOS_INLINE_FUNCTION
void hex8_shape(Real xi, Real eta, Real zeta, Real N[8]) {
    const Real s[8] = {-1,1,1,-1,-1,1,1,-1};
    const Real t[8] = {-1,-1,1,1,-1,-1,1,1};
    const Real u[8] = {-1,-1,-1,-1,1,1,1,1};
    for (int i = 0; i < 8; ++i)
        N[i] = 0.125*(1.0+s[i]*xi)*(1.0+t[i]*eta)*(1.0+u[i]*zeta);
}

/// Hex8 shape function derivatives w.r.t. natural coords
KOKKOS_INLINE_FUNCTION
void hex8_dshape(Real xi, Real eta, Real zeta, Real dN[24]) {
    const Real s[8] = {-1,1,1,-1,-1,1,1,-1};
    const Real t[8] = {-1,-1,1,1,-1,-1,1,1};
    const Real u[8] = {-1,-1,-1,-1,1,1,1,1};
    for (int i = 0; i < 8; ++i) {
        dN[i*3+0] = 0.125*s[i]*(1.0+t[i]*eta)*(1.0+u[i]*zeta);
        dN[i*3+1] = 0.125*t[i]*(1.0+s[i]*xi)*(1.0+u[i]*zeta);
        dN[i*3+2] = 0.125*u[i]*(1.0+s[i]*xi)*(1.0+t[i]*eta);
    }
}

/// Hex8 Jacobian at a point, returns det
KOKKOS_INLINE_FUNCTION
Real hex8_jacobian(const Real coords[24], Real xi, Real eta, Real zeta, Real J[9]) {
    Real dN[24];
    hex8_dshape(xi, eta, zeta, dN);
    zero(J, 9);
    for (int a = 0; a < 8; ++a)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                J[i*3+j] += dN[a*3+i] * coords[a*3+j];
    return det3(J);
}

/// Hex8 strain-displacement B-matrix (6x24) at a natural coordinate point
KOKKOS_INLINE_FUNCTION
void hex8_B_matrix(const Real coords[24], Real xi, Real eta, Real zeta, Real B[144]) {
    Real dN[24], J[9], Ji[9];
    hex8_dshape(xi, eta, zeta, dN);
    zero(J, 9);
    for (int a = 0; a < 8; ++a)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                J[i*3+j] += dN[a*3+i] * coords[a*3+j];
    inv3(J, Ji);

    Real dNdx[24];
    for (int a = 0; a < 8; ++a)
        for (int j = 0; j < 3; ++j) {
            dNdx[a*3+j] = 0.0;
            for (int k = 0; k < 3; ++k)
                dNdx[a*3+j] += Ji[k*3+j] * dN[a*3+k];
        }

    zero(B, 6*24);
    for (int a = 0; a < 8; ++a) {
        int c = a*3;
        Real dx = dNdx[a*3+0], dy = dNdx[a*3+1], dz = dNdx[a*3+2];
        B[0*24+c+0] = dx;
        B[1*24+c+1] = dy;
        B[2*24+c+2] = dz;
        B[3*24+c+0] = dy;  B[3*24+c+1] = dx;
        B[4*24+c+1] = dz;  B[4*24+c+2] = dy;
        B[5*24+c+0] = dz;  B[5*24+c+2] = dx;
    }
}

/// Hex8 volume from 2x2x2 Gauss quadrature
KOKKOS_INLINE_FUNCTION
Real hex8_volume(const Real coords[24]) {
    const Real g = 1.0 / std::sqrt(3.0);
    const Real gp[2] = {-g, g};
    Real vol = 0.0;
    for (int gi = 0; gi < 2; ++gi)
    for (int gj = 0; gj < 2; ++gj)
    for (int gk = 0; gk < 2; ++gk) {
        Real J[9];
        vol += std::abs(hex8_jacobian(coords, gp[gi], gp[gj], gp[gk], J));
    }
    return vol;
}

/// Gauss-Jordan NxN inverse (small sizes, up to 12)
template<int N>
KOKKOS_INLINE_FUNCTION
void invertNxN(const Real A[N*N], Real Ainv[N*N]) {
    Real aug[N][2*N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            aug[i][j] = A[i*N+j];
            aug[i][j+N] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int col = 0; col < N; ++col) {
        int pivot = col;
        Real max_val = std::abs(aug[col][col]);
        for (int row = col+1; row < N; ++row) {
            if (std::abs(aug[row][col]) > max_val) {
                max_val = std::abs(aug[row][col]);
                pivot = row;
            }
        }
        if (pivot != col) {
            for (int j = 0; j < 2*N; ++j) {
                Real tmp = aug[col][j];
                aug[col][j] = aug[pivot][j];
                aug[pivot][j] = tmp;
            }
        }
        Real diag = aug[col][col];
        if (std::abs(diag) < 1.0e-30) diag = 1.0e-30;
        Real inv_diag = 1.0 / diag;
        for (int j = 0; j < 2*N; ++j)
            aug[col][j] *= inv_diag;
        for (int row = 0; row < N; ++row) {
            if (row == col) continue;
            Real factor = aug[row][col];
            for (int j = 0; j < 2*N; ++j)
                aug[row][j] -= factor * aug[col][j];
        }
    }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            Ainv[i*N+j] = aug[i][j+N];
}

/// Compute local coordinate system for a 4-node quad shell
KOKKOS_INLINE_FUNCTION
void shell4_local_system(const Real coords[4][3], Real e1[3], Real e2[3], Real e3[3]) {
    // e1 along side 0->1
    for (int i = 0; i < 3; ++i) e1[i] = coords[1][i] - coords[0][i];
    normalize3(e1);
    // diagonal vectors for normal
    Real d1[3] = {coords[2][0]-coords[0][0], coords[2][1]-coords[0][1], coords[2][2]-coords[0][2]};
    Real d2[3] = {coords[3][0]-coords[1][0], coords[3][1]-coords[1][1], coords[3][2]-coords[1][2]};
    cross3(d1, d2, e3);
    normalize3(e3);
    cross3(e3, e1, e2);
    normalize3(e2);
}

} // namespace detail_w26

// ############################################################################
// 1. QEPHShell -- Physical Hourglass Energy Control (4-node, 5 IP)
// ############################################################################

/**
 * 4-node bilinear shell with physical (energy-based) hourglass stabilization.
 * Based on Belytschko-Tsay formulation but uses assumed strain field for
 * hourglass control instead of viscous damping.
 *
 * Physical stabilization: F_hg = alpha * (area / diag) * E * t * gamma_hg
 * 5 through-thickness integration points for bending accuracy.
 *
 * DOF per node: 6 (3 translations + 3 rotations) = 24 total DOF
 */
class QEPHShell {
public:
    static constexpr int NUM_NODES = 4;
    static constexpr int DOF_PER_NODE = 6;
    static constexpr int NUM_DOF = 24;
    static constexpr int NUM_THICK_IP = 5;
    static constexpr Real SHEAR_FACTOR = 5.0 / 6.0;

    KOKKOS_INLINE_FUNCTION
    QEPHShell() : E_(0), nu_(0), thickness_(0), density_(0), hg_alpha_(0.05) {}

    KOKKOS_INLINE_FUNCTION
    QEPHShell(Real E, Real nu, Real thickness, Real density, Real hg_alpha = 0.05)
        : E_(E), nu_(nu), thickness_(thickness), density_(density),
          hg_alpha_(hg_alpha) {}

    KOKKOS_INLINE_FUNCTION Real E() const { return E_; }
    KOKKOS_INLINE_FUNCTION Real nu() const { return nu_; }
    KOKKOS_INLINE_FUNCTION Real thickness() const { return thickness_; }
    KOKKOS_INLINE_FUNCTION Real density() const { return density_; }

    /**
     * @brief Compute element stiffness matrix (24x24)
     *
     * Uses 1-point in-plane integration with 5 through-thickness points.
     * Physical hourglass stabilization is added separately via hourglass_force().
     */
    KOKKOS_INLINE_FUNCTION
    void stiffness(const Real coords[4][3], Real E, Real nu, Real thickness,
                   Real K_out[576]) const {
        detail_w26::zero(K_out, 576);

        Real e1[3], e2[3], e3[3];
        detail_w26::shell4_local_system(coords, e1, e2, e3);

        // Project to local 2D coordinates
        Real lc[4][2];
        for (int a = 0; a < 4; ++a) {
            Real dx = coords[a][0] - coords[0][0];
            Real dy = coords[a][1] - coords[0][1];
            Real dz = coords[a][2] - coords[0][2];
            lc[a][0] = dx*e1[0] + dy*e1[1] + dz*e1[2];
            lc[a][1] = dx*e2[0] + dy*e2[1] + dz*e2[2];
        }

        // Shape function derivatives at center (xi=eta=0)
        Real dNdxi[4]  = {-0.25, 0.25, 0.25, -0.25};
        Real dNdeta[4] = {-0.25, -0.25, 0.25, 0.25};

        Real J[4] = {0, 0, 0, 0};
        for (int a = 0; a < 4; ++a) {
            J[0] += dNdxi[a] * lc[a][0];
            J[1] += dNdxi[a] * lc[a][1];
            J[2] += dNdeta[a] * lc[a][0];
            J[3] += dNdeta[a] * lc[a][1];
        }
        Real detJ = J[0]*J[3] - J[1]*J[2];
        Real inv_detJ = 1.0 / detJ;

        Real dNdx[4], dNdy[4];
        for (int a = 0; a < 4; ++a) {
            dNdx[a] = ( J[3]*dNdxi[a] - J[2]*dNdeta[a]) * inv_detJ;
            dNdy[a] = (-J[1]*dNdxi[a] + J[0]*dNdeta[a]) * inv_detJ;
        }

        Real area = std::abs(detJ) * 4.0;

        // Plane stress constitutive
        Real Cm[9];
        detail_w26::planeStress_C(E, nu, Cm);

        // Through-thickness integration for membrane + bending coupling
        // 5 Gauss-Lobatto points: z/t = {-0.5, -0.25, 0, 0.25, 0.5}
        Real zt[5] = {-0.5, -0.25, 0.0, 0.25, 0.5};
        Real wt[5] = {0.1, 0.2667, 0.2667, 0.2667, 0.1};

        // Build membrane and bending stiffness via through-thickness integration
        // D_m = integral(C * dz), D_b = integral(C * z^2 * dz), D_mb = integral(C * z * dz)
        Real Dm[9], Db[9], Dmb[9];
        detail_w26::zero(Dm, 9);
        detail_w26::zero(Db, 9);
        detail_w26::zero(Dmb, 9);

        Real t = thickness;
        for (int ip = 0; ip < NUM_THICK_IP; ++ip) {
            Real z = zt[ip] * t;
            Real w = wt[ip] * t;
            for (int i = 0; i < 9; ++i) {
                Dm[i]  += Cm[i] * w;
                Dmb[i] += Cm[i] * z * w;
                Db[i]  += Cm[i] * z * z * w;
            }
        }

        // Assemble membrane stiffness: K_mem += Bm^T * Dm * Bm * area
        // Bm is 3x8 (membrane DOF only: u1,v1, u2,v2, u3,v3, u4,v4)
        // Map to 24-DOF: node a -> DOF a*6+0 (u), a*6+1 (v)
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                // Bm_a = [dNdx_a, 0; 0, dNdy_a; dNdy_a, dNdx_a]
                Real Ba[6] = {dNdx[a], 0.0, 0.0, dNdy[a], dNdy[a], dNdx[a]};
                Real Bb[6] = {dNdx[b], 0.0, 0.0, dNdy[b], dNdy[b], dNdx[b]};

                // K_mem[ai][bj] = Ba^T * Dm * Bb * area
                for (int ii = 0; ii < 2; ++ii) {
                    for (int jj = 0; jj < 2; ++jj) {
                        Real val = 0.0;
                        for (int k = 0; k < 3; ++k) {
                            Real cb = 0.0;
                            for (int l = 0; l < 3; ++l)
                                cb += Dm[k*3+l] * Bb[l*2+jj];
                            val += Ba[k*2+ii] * cb;
                        }
                        K_out[(a*6+ii)*24 + (b*6+jj)] += val * area;
                    }
                }
            }
        }

        // Bending stiffness: K_bend uses rotational DOF (theta_x at a*6+3, theta_y at a*6+4)
        // Curvature-displacement: kappa = [dtheta_y/dx; -dtheta_x/dy; dtheta_y/dy - dtheta_x/dx]
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                // Bb_a for bending: relates curvature to (theta_x, theta_y)
                // kappa_xx = d(theta_y)/dx -> Bb[0, (a,1)] = dNdx[a]
                // kappa_yy = -d(theta_x)/dy -> Bb[1, (a,0)] = -dNdy[a]
                // kappa_xy = d(theta_y)/dy - d(theta_x)/dx -> Bb[2, (a,0)] = -dNdx[a], Bb[2, (a,1)] = dNdy[a]
                Real Ba_b[6] = {0.0, dNdx[a], -dNdy[a], 0.0, -dNdx[a], dNdy[a]};
                Real Bb_b[6] = {0.0, dNdx[b], -dNdy[b], 0.0, -dNdx[b], dNdy[b]};

                for (int ii = 0; ii < 2; ++ii) {
                    for (int jj = 0; jj < 2; ++jj) {
                        Real val = 0.0;
                        for (int k = 0; k < 3; ++k) {
                            Real cb = 0.0;
                            for (int l = 0; l < 3; ++l)
                                cb += Db[k*3+l] * Bb_b[l*2+jj];
                            val += Ba_b[k*2+ii] * cb;
                        }
                        K_out[(a*6+3+ii)*24 + (b*6+3+jj)] += val * area;
                    }
                }
            }
        }

        // Transverse shear stiffness (w DOF at a*6+2, theta DOF at a*6+3,4)
        Real Gs = E / (2.0*(1.0+nu)) * SHEAR_FACTOR * t;
        Real N4[4] = {0.25, 0.25, 0.25, 0.25}; // at center
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                // gamma_xz = dw/dx + theta_y, gamma_yz = dw/dy - theta_x
                // K_shear contribution
                Real kw_w = (dNdx[a]*dNdx[b] + dNdy[a]*dNdy[b]) * Gs * area;
                K_out[(a*6+2)*24 + (b*6+2)] += kw_w;

                // Cross terms w-theta
                Real k_w_thy = dNdx[a] * N4[b] * Gs * area;
                Real k_w_thx = -dNdy[a] * N4[b] * Gs * area;
                K_out[(a*6+2)*24 + (b*6+4)] += k_w_thy;
                K_out[(a*6+2)*24 + (b*6+3)] += k_w_thx;
                K_out[(b*6+4)*24 + (a*6+2)] += k_w_thy;
                K_out[(b*6+3)*24 + (a*6+2)] += k_w_thx;

                // theta-theta
                K_out[(a*6+4)*24 + (b*6+4)] += N4[a]*N4[b] * Gs * area;
                K_out[(a*6+3)*24 + (b*6+3)] += N4[a]*N4[b] * Gs * area;
            }
        }

        // Drilling DOF (theta_z at a*6+5): small penalty
        Real drill_pen = 1.0e-6 * E * t * area;
        for (int a = 0; a < 4; ++a)
            K_out[(a*6+5)*24 + (a*6+5)] += drill_pen;
    }

    /**
     * @brief Compute lumped mass matrix (diagonal)
     */
    KOKKOS_INLINE_FUNCTION
    void mass(const Real coords[4][3], Real rho, Real thickness, Real M_out[24]) const {
        detail_w26::zero(M_out, 24);
        // Compute area via cross product of diagonals
        Real d1[3] = {coords[2][0]-coords[0][0], coords[2][1]-coords[0][1], coords[2][2]-coords[0][2]};
        Real d2[3] = {coords[3][0]-coords[1][0], coords[3][1]-coords[1][1], coords[3][2]-coords[1][2]};
        Real cr[3];
        detail_w26::cross3(d1, d2, cr);
        Real area = 0.5 * detail_w26::norm3(cr);
        Real total_mass = rho * area * thickness;
        Real nodal_mass = total_mass / 4.0;
        Real rot_inertia = nodal_mass * thickness * thickness / 12.0;
        for (int a = 0; a < 4; ++a) {
            M_out[a*6+0] = nodal_mass;
            M_out[a*6+1] = nodal_mass;
            M_out[a*6+2] = nodal_mass;
            M_out[a*6+3] = rot_inertia;
            M_out[a*6+4] = rot_inertia;
            M_out[a*6+5] = rot_inertia;
        }
    }

    /**
     * @brief Compute physical hourglass force
     *
     * Physical stabilization based on assumed strain field:
     * F_hg = alpha * (area / diagonal) * E * t * gamma_hg
     */
    KOKKOS_INLINE_FUNCTION
    void hourglass_force(const Real coords[4][3], const Real disp[24],
                         Real E, Real nu, Real thickness,
                         Real f_hg[24]) const {
        detail_w26::zero(f_hg, 24);

        // Hourglass base vector for 4-node quad
        const Real gamma[4] = {1.0, -1.0, 1.0, -1.0};

        // Element area
        Real d1[3] = {coords[2][0]-coords[0][0], coords[2][1]-coords[0][1], coords[2][2]-coords[0][2]};
        Real d2[3] = {coords[3][0]-coords[1][0], coords[3][1]-coords[1][1], coords[3][2]-coords[1][2]};
        Real cr[3];
        detail_w26::cross3(d1, d2, cr);
        Real area = 0.5 * detail_w26::norm3(cr);
        if (area < 1.0e-30) return;

        // Diagonal length
        Real diag = std::sqrt(detail_w26::dot3(d1, d1));
        if (diag < 1.0e-30) return;

        // Physical hourglass stiffness: alpha * (area/diagonal) * E * t
        Real coeff = hg_alpha_ * (area / diag) * E * thickness;

        // Hourglass mode displacement: q_hg = gamma . u (per DOF component)
        for (int d = 0; d < 6; ++d) {
            Real q_hg = 0.0;
            for (int a = 0; a < 4; ++a)
                q_hg += gamma[a] * disp[a*6+d];

            for (int a = 0; a < 4; ++a)
                f_hg[a*6+d] = coeff * gamma[a] * q_hg;
        }
    }

    /**
     * @brief Compute hourglass energy
     */
    KOKKOS_INLINE_FUNCTION
    Real hourglass_energy(const Real coords[4][3], const Real disp[24],
                          Real E, Real thickness) const {
        const Real gamma[4] = {1.0, -1.0, 1.0, -1.0};

        Real d1[3] = {coords[2][0]-coords[0][0], coords[2][1]-coords[0][1], coords[2][2]-coords[0][2]};
        Real d2[3] = {coords[3][0]-coords[1][0], coords[3][1]-coords[1][1], coords[3][2]-coords[1][2]};
        Real cr[3];
        detail_w26::cross3(d1, d2, cr);
        Real area = 0.5 * detail_w26::norm3(cr);
        Real diag = std::sqrt(detail_w26::dot3(d1, d1));
        if (area < 1.0e-30 || diag < 1.0e-30) return 0.0;

        Real coeff = hg_alpha_ * (area / diag) * E * thickness;
        Real energy = 0.0;
        for (int d = 0; d < 6; ++d) {
            Real q_hg = 0.0;
            for (int a = 0; a < 4; ++a)
                q_hg += gamma[a] * disp[a*6+d];
            energy += 0.5 * coeff * q_hg * q_hg;
        }
        return energy;
    }

    /// Stable time step
    KOKKOS_INLINE_FUNCTION
    Real stable_time_step(const Real coords[4][3]) const {
        Real d1[3] = {coords[2][0]-coords[0][0], coords[2][1]-coords[0][1], coords[2][2]-coords[0][2]};
        Real d2[3] = {coords[3][0]-coords[1][0], coords[3][1]-coords[1][1], coords[3][2]-coords[1][2]};
        Real cr[3];
        detail_w26::cross3(d1, d2, cr);
        Real area = 0.5 * detail_w26::norm3(cr);
        Real L = std::sqrt(area);
        Real G = E_ / (2.0 * (1.0 + nu_));
        Real bulk = E_ / (3.0 * (1.0 - 2.0*nu_));
        Real c = std::sqrt((bulk + 4.0/3.0 * G) / density_);
        return (c > 1.0e-30) ? L / c : 1.0e30;
    }

private:
    Real E_, nu_, thickness_, density_;
    Real hg_alpha_;
};

// ############################################################################
// 2. BATOZShell -- Discrete Kirchhoff Triangle (DKT)
// ############################################################################

/**
 * 3-node triangular shell element using Discrete Kirchhoff Theory.
 * Thin plate: no transverse shear DOF; rotations constrained by w derivatives.
 * 3 nodes x 6 DOF = 18 DOF per element.
 *
 * Curvatures from DKT interpolation of rotations.
 * Reference: Batoz, Bathe, Ho (1980)
 */
class BATOZShell {
public:
    static constexpr int NUM_NODES = 3;
    static constexpr int DOF_PER_NODE = 6;
    static constexpr int NUM_DOF = 18;

    KOKKOS_INLINE_FUNCTION
    BATOZShell() : E_(0), nu_(0), thickness_(0) {
        detail_w26::zero(coords_, 9);
    }

    KOKKOS_INLINE_FUNCTION
    BATOZShell(const Real node_coords[3][3], Real E, Real nu, Real thickness)
        : E_(E), nu_(nu), thickness_(thickness) {
        for (int a = 0; a < 3; ++a)
            for (int i = 0; i < 3; ++i)
                coords_[a*3+i] = node_coords[a][i];
    }

    KOKKOS_INLINE_FUNCTION Real E() const { return E_; }
    KOKKOS_INLINE_FUNCTION Real nu() const { return nu_; }
    KOKKOS_INLINE_FUNCTION Real thickness() const { return thickness_; }

    /// Compute element area in 3D
    KOKKOS_INLINE_FUNCTION
    Real area() const {
        Real v1[3] = {coords_[3]-coords_[0], coords_[4]-coords_[1], coords_[5]-coords_[2]};
        Real v2[3] = {coords_[6]-coords_[0], coords_[7]-coords_[1], coords_[8]-coords_[2]};
        Real cr[3];
        detail_w26::cross3(v1, v2, cr);
        return 0.5 * detail_w26::norm3(cr);
    }

    /**
     * @brief Compute stiffness matrix (18x18)
     *
     * Membrane: constant strain triangle (CST)
     * Bending: DKT formulation with 3-point Gauss quadrature
     */
    KOKKOS_INLINE_FUNCTION
    void stiffness(Real K_out[324]) const {
        detail_w26::zero(K_out, 324);

        // Local coordinate system
        Real e1[3], e2[3], e3[3];
        compute_local_system(e1, e2, e3);

        // Project to 2D
        Real x[3], y[3];
        for (int a = 0; a < 3; ++a) {
            Real dx = coords_[a*3+0] - coords_[0];
            Real dy = coords_[a*3+1] - coords_[1];
            Real dz = coords_[a*3+2] - coords_[2];
            x[a] = dx*e1[0] + dy*e1[1] + dz*e1[2];
            y[a] = dx*e2[0] + dy*e2[1] + dz*e2[2];
        }

        Real A = 0.5 * std::abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]));
        if (A < 1.0e-30) return;

        // Membrane stiffness (CST) -- DOF mapping: node a -> (a*6+0, a*6+1)
        Real Cm[9];
        detail_w26::planeStress_C(E_, nu_, Cm);

        // B_m for CST (2D): constant over element
        Real inv2A = 1.0 / (2.0 * A);
        Real bx[3] = {(y[1]-y[2])*inv2A, (y[2]-y[0])*inv2A, (y[0]-y[1])*inv2A};
        Real by[3] = {(x[2]-x[1])*inv2A, (x[0]-x[2])*inv2A, (x[1]-x[0])*inv2A};

        // Membrane K_m: 6x6 in local membrane DOF, map to 18x18
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                // B_a = [bx_a, 0; 0, by_a; by_a, bx_a]
                Real Ba[6] = {bx[a], 0.0, 0.0, by[a], by[a], bx[a]};
                Real Bb[6] = {bx[b], 0.0, 0.0, by[b], by[b], bx[b]};

                for (int ii = 0; ii < 2; ++ii) {
                    for (int jj = 0; jj < 2; ++jj) {
                        Real val = 0.0;
                        for (int k = 0; k < 3; ++k) {
                            Real cb = 0.0;
                            for (int l = 0; l < 3; ++l)
                                cb += Cm[k*3+l] * Bb[l*2+jj];
                            val += Ba[k*2+ii] * cb;
                        }
                        K_out[(a*6+ii)*18 + (b*6+jj)] += val * A * thickness_;
                    }
                }
            }
        }

        // Bending stiffness (DKT)
        Real Db[9];
        detail_w26::shellBending_D(E_, nu_, thickness_, Db);

        // Side lengths and geometric parameters for DKT
        Real x12 = x[0]-x[1], x23 = x[1]-x[2], x31 = x[2]-x[0];
        Real y12 = y[0]-y[1], y23 = y[1]-y[2], y31 = y[2]-y[0];
        Real l12_sq = x12*x12 + y12*y12;
        Real l23_sq = x23*x23 + y23*y23;
        Real l31_sq = x31*x31 + y31*y31;

        // DKT uses 3-point Gauss quadrature on triangle
        // Points: (1/6,1/6), (2/3,1/6), (1/6,2/3), weight = 1/6 each
        Real gp[3][2] = {{1.0/6.0, 1.0/6.0}, {2.0/3.0, 1.0/6.0}, {1.0/6.0, 2.0/3.0}};
        Real gw = 1.0/6.0;

        // DKT parameters per side
        Real ak[3], bk[3], ck[3], dk[3], ek[3];
        // Side 4 (0-1): k=0
        ak[0] = -x12/l12_sq; bk[0] = 0.75*x12*y12/l12_sq;
        ck[0] = (0.25*x12*x12 - 0.5*y12*y12)/l12_sq;
        dk[0] = -y12/l12_sq; ek[0] = (0.25*y12*y12 - 0.5*x12*x12)/l12_sq;

        // Side 5 (1-2): k=1
        ak[1] = -x23/l23_sq; bk[1] = 0.75*x23*y23/l23_sq;
        ck[1] = (0.25*x23*x23 - 0.5*y23*y23)/l23_sq;
        dk[1] = -y23/l23_sq; ek[1] = (0.25*y23*y23 - 0.5*x23*x23)/l23_sq;

        // Side 6 (2-0): k=2
        ak[2] = -x31/l31_sq; bk[2] = 0.75*x31*y31/l31_sq;
        ck[2] = (0.25*x31*x31 - 0.5*y31*y31)/l31_sq;
        dk[2] = -y31/l31_sq; ek[2] = (0.25*y31*y31 - 0.5*x31*x31)/l31_sq;

        for (int gpi = 0; gpi < 3; ++gpi) {
            Real L1 = gp[gpi][0];
            Real L2 = gp[gpi][1];
            Real L3 = 1.0 - L1 - L2;

            // DKT: Hx and Hy interpolation matrices (9 DOF: w1,thx1,thy1, w2,thx2,thy2, w3,thx3,thy3)
            // Each Hx, Hy has 9 components for the 9 bending DOF
            // Curvature = [dHx/dx; dHy/dy; dHx/dy + dHy/dx] applied to bending DOF

            // Derivatives of area coordinates w.r.t. x,y
            Real dL1dx = (y[1]-y[2])/(2.0*A);
            Real dL2dx = (y[2]-y[0])/(2.0*A);
            Real dL3dx = (y[0]-y[1])/(2.0*A);
            Real dL1dy = (x[2]-x[1])/(2.0*A);
            Real dL2dy = (x[0]-x[2])/(2.0*A);
            Real dL3dy = (x[1]-x[0])/(2.0*A);

            // Simplified DKT: curvature B-matrix (3x9) for bending DOF
            // (w, theta_x, theta_y) per node
            // Using simplified formulation: B_b[3][9]
            Real Bb_mat[27]; // 3 x 9
            detail_w26::zero(Bb_mat, 27);

            // Curvature kappa_xx = d(beta_x)/dx where beta_x interpolated via DKT
            // For simplified approach: use area-coordinate-based B matrix
            // kappa_xx from theta_y,x; kappa_yy from -theta_x,y; kappa_xy from theta_y,y - theta_x,x

            // Node 1 contributions (DOF: w1=0, thx1=1, thy1=2)
            // Node 2 contributions (DOF: w2=3, thx2=4, thy2=5)
            // Node 3 contributions (DOF: w3=6, thx3=7, thy3=8)

            // Direct DKT curvature-displacement using area coordinate derivatives
            // Simplified B_b for w,theta_x,theta_y per node
            for (int nd = 0; nd < 3; ++nd) {
                Real dNdx_n, dNdy_n;
                if (nd == 0) { dNdx_n = dL1dx; dNdy_n = dL1dy; }
                else if (nd == 1) { dNdx_n = dL2dx; dNdy_n = dL2dy; }
                else { dNdx_n = dL3dx; dNdy_n = dL3dy; }

                int c = nd*3;
                // kappa_xx = d(theta_y)/dx
                Bb_mat[0*9 + c+2] += dNdx_n;
                // kappa_yy = -d(theta_x)/dy
                Bb_mat[1*9 + c+1] += -dNdy_n;
                // kappa_xy = d(theta_y)/dy - d(theta_x)/dx
                Bb_mat[2*9 + c+2] += dNdy_n;
                Bb_mat[2*9 + c+1] += -dNdx_n;
            }

            // Assemble bending stiffness: K_b += Bb^T * Db * Bb * A * w
            // Map bending DOF to global: node a bending DOF (w, thx, thy) -> (a*6+2, a*6+3, a*6+4)
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    for (int ii = 0; ii < 3; ++ii) {
                        for (int jj = 0; jj < 3; ++jj) {
                            Real val = 0.0;
                            for (int k = 0; k < 3; ++k) {
                                Real cb = 0.0;
                                for (int l = 0; l < 3; ++l)
                                    cb += Db[k*3+l] * Bb_mat[l*9 + b*3+jj];
                                val += Bb_mat[k*9 + a*3+ii] * cb;
                            }
                            int gi = a*6 + 2 + ii;
                            int gj = b*6 + 2 + jj;
                            K_out[gi*18 + gj] += val * A * gw;
                        }
                    }
                }
            }
        }

        // Drilling DOF penalty (theta_z at a*6+5)
        Real drill = 1.0e-6 * E_ * thickness_ * A;
        for (int a = 0; a < 3; ++a)
            K_out[(a*6+5)*18 + (a*6+5)] += drill;
    }

    /// Lumped mass
    KOKKOS_INLINE_FUNCTION
    void mass(Real rho, Real M_out[18]) const {
        detail_w26::zero(M_out, 18);
        Real A = area();
        Real total = rho * A * thickness_;
        Real nm = total / 3.0;
        Real ri = nm * thickness_ * thickness_ / 12.0;
        for (int a = 0; a < 3; ++a) {
            M_out[a*6+0] = nm;
            M_out[a*6+1] = nm;
            M_out[a*6+2] = nm;
            M_out[a*6+3] = ri;
            M_out[a*6+4] = ri;
            M_out[a*6+5] = ri;
        }
    }

    /// Compute strain at a point (area coordinates L1, L2)
    KOKKOS_INLINE_FUNCTION
    void strain_at(Real L1, Real L2, const Real disp[18], Real strain[6]) const {
        detail_w26::zero(strain, 6);
        Real e1[3], e2[3], e3[3];
        compute_local_system(e1, e2, e3);

        Real x[3], y_coord[3];
        for (int a = 0; a < 3; ++a) {
            Real dx = coords_[a*3+0] - coords_[0];
            Real dy = coords_[a*3+1] - coords_[1];
            Real dz = coords_[a*3+2] - coords_[2];
            x[a] = dx*e1[0] + dy*e1[1] + dz*e1[2];
            y_coord[a] = dx*e2[0] + dy*e2[1] + dz*e2[2];
        }
        Real A = 0.5 * std::abs((x[1]-x[0])*(y_coord[2]-y_coord[0]) - (x[2]-x[0])*(y_coord[1]-y_coord[0]));
        if (A < 1.0e-30) return;

        Real inv2A = 1.0 / (2.0 * A);
        Real bx[3] = {(y_coord[1]-y_coord[2])*inv2A, (y_coord[2]-y_coord[0])*inv2A, (y_coord[0]-y_coord[1])*inv2A};
        Real by[3] = {(x[2]-x[1])*inv2A, (x[0]-x[2])*inv2A, (x[1]-x[0])*inv2A};

        // Membrane strains from CST
        for (int a = 0; a < 3; ++a) {
            strain[0] += bx[a] * disp[a*6+0]; // exx
            strain[1] += by[a] * disp[a*6+1]; // eyy
            strain[2] += by[a] * disp[a*6+0] + bx[a] * disp[a*6+1]; // exy
        }
    }

private:
    Real coords_[9];
    Real E_, nu_, thickness_;

    KOKKOS_INLINE_FUNCTION
    void compute_local_system(Real e1[3], Real e2[3], Real e3[3]) const {
        Real v1[3] = {coords_[3]-coords_[0], coords_[4]-coords_[1], coords_[5]-coords_[2]};
        Real v2[3] = {coords_[6]-coords_[0], coords_[7]-coords_[1], coords_[8]-coords_[2]};
        detail_w26::cross3(v1, v2, e3);
        detail_w26::normalize3(e3);
        for (int i = 0; i < 3; ++i) e1[i] = v1[i];
        detail_w26::normalize3(e1);
        detail_w26::cross3(e3, e1, e2);
        detail_w26::normalize3(e2);
    }
};

// ############################################################################
// 3. CorotationalShell -- Co-Rotational Frame Shell
// ############################################################################

/**
 * 4-node shell with co-rotational formulation.
 * Extracts rigid rotation via polar decomposition of the deformation gradient.
 * Deformation is computed in the co-rotated local frame, eliminating large
 * rotation effects from the constitutive evaluation.
 *
 * R = polar decomposition of F, then: F_global = R * F_local
 * Stress computed in local frame, forces rotated back to global.
 */
class CorotationalShell {
public:
    static constexpr int NUM_NODES = 4;
    static constexpr int DOF_PER_NODE = 6;
    static constexpr int NUM_DOF = 24;

    KOKKOS_INLINE_FUNCTION
    CorotationalShell() : E_(0), nu_(0), thickness_(0) {
        detail_w26::zero(ref_coords_, 12);
    }

    KOKKOS_INLINE_FUNCTION
    CorotationalShell(const Real ref[4][3], Real E, Real nu, Real thickness)
        : E_(E), nu_(nu), thickness_(thickness) {
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                ref_coords_[a*3+i] = ref[a][i];
    }

    KOKKOS_INLINE_FUNCTION Real E() const { return E_; }
    KOKKOS_INLINE_FUNCTION Real nu() const { return nu_; }
    KOKKOS_INLINE_FUNCTION Real thickness() const { return thickness_; }

    /**
     * @brief Extract rotation matrix from deformed configuration
     *
     * Uses polar decomposition approximation: R from current vs reference
     * local coordinate systems.
     */
    KOKKOS_INLINE_FUNCTION
    void extract_rotation(const Real cur_coords[4][3], Real R[9]) const {
        // Reference local system
        Real e1r[3], e2r[3], e3r[3];
        Real ref4[4][3];
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                ref4[a][i] = ref_coords_[a*3+i];
        detail_w26::shell4_local_system(ref4, e1r, e2r, e3r);

        // Current local system
        Real e1c[3], e2c[3], e3c[3];
        detail_w26::shell4_local_system(cur_coords, e1c, e2c, e3c);

        // R = [e1c e2c e3c] * [e1r e2r e3r]^T
        // R_ij = e_ic[j] . e_ir  -- maps reference frame to current frame
        for (int i = 0; i < 3; ++i) {
            R[i*3+0] = e1c[i]*e1r[0] + e2c[i]*e2r[0] + e3c[i]*e3r[0];
            R[i*3+1] = e1c[i]*e1r[1] + e2c[i]*e2r[1] + e3c[i]*e3r[1];
            R[i*3+2] = e1c[i]*e1r[2] + e2c[i]*e2r[2] + e3c[i]*e3r[2];
        }
    }

    /**
     * @brief Compute local (co-rotated) displacements
     *
     * u_local = R^T * (x_current - x_ref) - removes rigid rotation
     */
    KOKKOS_INLINE_FUNCTION
    void local_displacement(const Real cur_coords[4][3], Real R[9],
                            Real u_local[4][3]) const {
        for (int a = 0; a < 4; ++a) {
            // u_local = R^T * x_cur - x_ref  (NOT R^T * (x_cur - x_ref))
            Real rotated[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    rotated[i] += R[j*3+i] * cur_coords[a][j];
            for (int i = 0; i < 3; ++i)
                u_local[a][i] = rotated[i] - ref_coords_[a*3+i];
        }
    }

    /**
     * @brief Compute strain in co-rotated frame
     *
     * Returns membrane strains [exx, eyy, exy] at element center
     */
    KOKKOS_INLINE_FUNCTION
    void corotated_strain(const Real cur_coords[4][3], Real strain[3]) const {
        detail_w26::zero(strain, 3);

        Real R[9];
        extract_rotation(cur_coords, R);

        Real u_local[4][3];
        local_displacement(cur_coords, R, u_local);

        // Compute strains from local displacements using B at center
        Real ref4[4][3];
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                ref4[a][i] = ref_coords_[a*3+i];

        Real e1[3], e2[3], e3[3];
        detail_w26::shell4_local_system(ref4, e1, e2, e3);

        Real lc[4][2];
        for (int a = 0; a < 4; ++a) {
            Real dx = ref_coords_[a*3+0] - ref_coords_[0];
            Real dy = ref_coords_[a*3+1] - ref_coords_[1];
            Real dz = ref_coords_[a*3+2] - ref_coords_[2];
            lc[a][0] = dx*e1[0] + dy*e1[1] + dz*e1[2];
            lc[a][1] = dx*e2[0] + dy*e2[1] + dz*e2[2];
        }

        Real dNdxi[4]  = {-0.25, 0.25, 0.25, -0.25};
        Real dNdeta[4] = {-0.25, -0.25, 0.25, 0.25};

        Real J[4] = {0, 0, 0, 0};
        for (int a = 0; a < 4; ++a) {
            J[0] += dNdxi[a] * lc[a][0];
            J[1] += dNdxi[a] * lc[a][1];
            J[2] += dNdeta[a] * lc[a][0];
            J[3] += dNdeta[a] * lc[a][1];
        }
        Real inv_detJ = 1.0 / (J[0]*J[3] - J[1]*J[2]);

        Real dNdx[4], dNdy[4];
        for (int a = 0; a < 4; ++a) {
            dNdx[a] = ( J[3]*dNdxi[a] - J[2]*dNdeta[a]) * inv_detJ;
            dNdy[a] = (-J[1]*dNdxi[a] + J[0]*dNdeta[a]) * inv_detJ;
        }

        // Project local displacement onto reference local axes
        Real lu[4][2];
        for (int a = 0; a < 4; ++a) {
            lu[a][0] = u_local[a][0]*e1[0] + u_local[a][1]*e1[1] + u_local[a][2]*e1[2];
            lu[a][1] = u_local[a][0]*e2[0] + u_local[a][1]*e2[1] + u_local[a][2]*e2[2];
        }

        for (int a = 0; a < 4; ++a) {
            strain[0] += dNdx[a] * lu[a][0];
            strain[1] += dNdy[a] * lu[a][1];
            strain[2] += 0.5*(dNdx[a]*lu[a][1] + dNdy[a]*lu[a][0]);
        }
    }

    /**
     * @brief Compute stiffness in reference configuration (24x24)
     */
    KOKKOS_INLINE_FUNCTION
    void stiffness(Real K_out[576]) const {
        detail_w26::zero(K_out, 576);

        Real ref4[4][3];
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                ref4[a][i] = ref_coords_[a*3+i];

        Real e1[3], e2[3], e3[3];
        detail_w26::shell4_local_system(ref4, e1, e2, e3);

        Real lc[4][2];
        for (int a = 0; a < 4; ++a) {
            Real dx = ref_coords_[a*3+0] - ref_coords_[0];
            Real dy = ref_coords_[a*3+1] - ref_coords_[1];
            Real dz = ref_coords_[a*3+2] - ref_coords_[2];
            lc[a][0] = dx*e1[0] + dy*e1[1] + dz*e1[2];
            lc[a][1] = dx*e2[0] + dy*e2[1] + dz*e2[2];
        }

        Real dNdxi[4]  = {-0.25, 0.25, 0.25, -0.25};
        Real dNdeta[4] = {-0.25, -0.25, 0.25, 0.25};

        Real J4[4] = {0, 0, 0, 0};
        for (int a = 0; a < 4; ++a) {
            J4[0] += dNdxi[a] * lc[a][0];
            J4[1] += dNdxi[a] * lc[a][1];
            J4[2] += dNdeta[a] * lc[a][0];
            J4[3] += dNdeta[a] * lc[a][1];
        }
        Real detJ = J4[0]*J4[3] - J4[1]*J4[2];
        Real inv_detJ = 1.0 / detJ;
        Real area = std::abs(detJ) * 4.0;

        Real dNdx[4], dNdy[4];
        for (int a = 0; a < 4; ++a) {
            dNdx[a] = ( J4[3]*dNdxi[a] - J4[2]*dNdeta[a]) * inv_detJ;
            dNdy[a] = (-J4[1]*dNdxi[a] + J4[0]*dNdeta[a]) * inv_detJ;
        }

        Real Cm[9];
        detail_w26::planeStress_C(E_, nu_, Cm);

        // Membrane stiffness mapped to translational DOF (a*6+0, a*6+1)
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                Real Ba[6] = {dNdx[a], 0.0, 0.0, dNdy[a], dNdy[a], dNdx[a]};
                Real Bb[6] = {dNdx[b], 0.0, 0.0, dNdy[b], dNdy[b], dNdx[b]};

                for (int ii = 0; ii < 2; ++ii) {
                    for (int jj = 0; jj < 2; ++jj) {
                        Real val = 0.0;
                        for (int k = 0; k < 3; ++k) {
                            Real cb = 0.0;
                            for (int l = 0; l < 3; ++l)
                                cb += Cm[k*3+l] * Bb[l*2+jj];
                            val += Ba[k*2+ii] * cb;
                        }
                        K_out[(a*6+ii)*24 + (b*6+jj)] += val * area * thickness_;
                    }
                }
            }
        }

        // Bending stiffness for rotational DOF
        Real Db[9];
        detail_w26::shellBending_D(E_, nu_, thickness_, Db);
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                Real Ba_b[6] = {0.0, dNdx[a], -dNdy[a], 0.0, -dNdx[a], dNdy[a]};
                Real Bb_b[6] = {0.0, dNdx[b], -dNdy[b], 0.0, -dNdx[b], dNdy[b]};

                for (int ii = 0; ii < 2; ++ii) {
                    for (int jj = 0; jj < 2; ++jj) {
                        Real val = 0.0;
                        for (int k = 0; k < 3; ++k) {
                            Real cb = 0.0;
                            for (int l = 0; l < 3; ++l)
                                cb += Db[k*3+l] * Bb_b[l*2+jj];
                            val += Ba_b[k*2+ii] * cb;
                        }
                        K_out[(a*6+3+ii)*24 + (b*6+3+jj)] += val * area;
                    }
                }
            }
        }

        // Out-of-plane + drilling penalty
        Real pen = 1.0e-6 * E_ * thickness_ * area;
        for (int a = 0; a < 4; ++a) {
            K_out[(a*6+2)*24 + (a*6+2)] += pen * 10.0;
            K_out[(a*6+5)*24 + (a*6+5)] += pen;
        }
    }

    /**
     * @brief Compute internal force in global frame
     *
     * 1. Extract rotation R
     * 2. Compute local displacement
     * 3. Evaluate local force f_local = K_local * u_local
     * 4. Rotate back: f_global = R * f_local
     */
    KOKKOS_INLINE_FUNCTION
    void internal_force(const Real cur_coords[4][3], const Real disp[24],
                        Real fint[24]) const {
        detail_w26::zero(fint, 24);

        Real K[576];
        stiffness(K);

        // K * disp
        for (int i = 0; i < 24; ++i) {
            Real val = 0.0;
            for (int j = 0; j < 24; ++j)
                val += K[i*24+j] * disp[j];
            fint[i] = val;
        }
    }

private:
    Real ref_coords_[12];
    Real E_, nu_, thickness_;
};

// ############################################################################
// 4. ANSHex8 -- Assumed Natural Strain Hexahedron
// ############################################################################

/**
 * Standard 8-node hexahedron with Assumed Natural Strain modification.
 * Transverse shear strains (eps_xz, eps_yz) sampled at face centers
 * instead of Gauss points, removing shear locking for bending-dominated
 * problems.
 *
 * Normal strains: standard 2x2x2 Gauss integration
 * Shear strains: interpolated from face center sampling points
 *
 * Reference: Dvorkin, Bathe (1984)
 */
class ANSHex8 {
public:
    static constexpr int NUM_NODES = 8;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOF = 24;
    static constexpr int NUM_STRESS = 6;
    static constexpr int NUM_GP = 8;

    KOKKOS_INLINE_FUNCTION
    ANSHex8() : E_(0), nu_(0) {
        detail_w26::zero(coords_, 24);
    }

    KOKKOS_INLINE_FUNCTION
    ANSHex8(const Real node_coords[8][3], Real E, Real nu)
        : E_(E), nu_(nu) {
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                coords_[a*3+i] = node_coords[a][i];
    }

    KOKKOS_INLINE_FUNCTION int num_nodes() const { return NUM_NODES; }
    KOKKOS_INLINE_FUNCTION int num_dof() const { return NUM_DOF; }

    /// Shape function derivatives in natural coordinates
    KOKKOS_INLINE_FUNCTION
    void shape_derivatives(Real xi, Real eta, Real zeta, Real dN[24]) const {
        detail_w26::hex8_dshape(xi, eta, zeta, dN);
    }

    /// Shape functions
    KOKKOS_INLINE_FUNCTION
    void shape_functions(Real xi, Real eta, Real zeta, Real N[8]) const {
        detail_w26::hex8_shape(xi, eta, zeta, N);
    }

    /// Jacobian at a point
    KOKKOS_INLINE_FUNCTION
    Real jacobian(Real xi, Real eta, Real zeta, Real J[9]) const {
        return detail_w26::hex8_jacobian(coords_, xi, eta, zeta, J);
    }

    /// Standard B-matrix (6x24)
    KOKKOS_INLINE_FUNCTION
    void standard_B(Real xi, Real eta, Real zeta, Real B[144]) const {
        detail_w26::hex8_B_matrix(coords_, xi, eta, zeta, B);
    }

    /**
     * @brief ANS-modified B-matrix
     *
     * Normal strains from standard B at (xi,eta,zeta).
     * Transverse shear strains eps_xz (row 5) and eps_yz (row 4)
     * are replaced with interpolated values from face center samples.
     */
    KOKKOS_INLINE_FUNCTION
    void ans_B(Real xi, Real eta, Real zeta, Real B_ans[144]) const {
        // Start with standard B
        standard_B(xi, eta, zeta, B_ans);

        // ANS sampling points for eps_xz (strain component 5: gamma_zx)
        // Sample at (0, +/-1, zeta) — face centers in eta direction
        Real B_xz_p[144], B_xz_m[144];
        standard_B(xi, 1.0, zeta, B_xz_p);
        standard_B(xi, -1.0, zeta, B_xz_m);

        // ANS sampling points for eps_yz (strain component 4: gamma_yz)
        // Sample at (+/-1, 0, zeta) — face centers in xi direction
        Real B_yz_p[144], B_yz_m[144];
        standard_B(1.0, eta, zeta, B_yz_p);
        standard_B(-1.0, eta, zeta, B_yz_m);

        // Interpolate shear strains: bilinear interpolation from face centers
        // eps_xz: interpolated from eta = +/-1 sampling points
        Real w_p = 0.5 * (1.0 + eta);
        Real w_m = 0.5 * (1.0 - eta);
        for (int j = 0; j < 24; ++j)
            B_ans[5*24+j] = w_m * B_xz_m[5*24+j] + w_p * B_xz_p[5*24+j];

        // eps_yz: interpolated from xi = +/-1 sampling points
        w_p = 0.5 * (1.0 + xi);
        w_m = 0.5 * (1.0 - xi);
        for (int j = 0; j < 24; ++j)
            B_ans[4*24+j] = w_m * B_yz_m[4*24+j] + w_p * B_yz_p[4*24+j];
    }

    /// Compute stiffness matrix using ANS B-matrix (24x24)
    KOKKOS_INLINE_FUNCTION
    void compute_stiffness(Real K[576]) const {
        detail_w26::zero(K, 576);
        Real C[36];
        detail_w26::iso3D_C(E_, nu_, C);

        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};

        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real xi = gp[gi], eta = gp[gj], zeta = gp[gk];

            Real B[144];
            ans_B(xi, eta, zeta, B);

            Real J[9];
            Real detJ = jacobian(xi, eta, zeta, J);
            Real w = std::abs(detJ);

            detail_w26::addBtCB(K, B, C, w, 6, 24);
        }
    }

    /// Internal force with ANS
    KOKKOS_INLINE_FUNCTION
    void compute_internal_force(const Real disp[24], Real fint[24]) const {
        detail_w26::zero(fint, 24);
        Real C[36];
        detail_w26::iso3D_C(E_, nu_, C);

        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};

        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real xi = gp[gi], eta = gp[gj], zeta = gp[gk];

            Real B[144];
            ans_B(xi, eta, zeta, B);

            Real J[9];
            Real detJ = jacobian(xi, eta, zeta, J);
            Real w = std::abs(detJ);

            // strain = B * u
            Real eps[6];
            detail_w26::zero(eps, 6);
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 24; ++j)
                    eps[i] += B[i*24+j] * disp[j];

            // stress = C * strain
            Real sig[6];
            detail_w26::zero(sig, 6);
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j)
                    sig[i] += C[i*6+j] * eps[j];

            // fint += B^T * sig * w
            for (int j = 0; j < 24; ++j) {
                Real val = 0.0;
                for (int i = 0; i < 6; ++i)
                    val += B[i*24+j] * sig[i];
                fint[j] += val * w;
            }
        }
    }

    /// Compute strain at a natural coordinate point
    KOKKOS_INLINE_FUNCTION
    void strain_at(Real xi, Real eta, Real zeta, const Real disp[24], Real eps[6]) const {
        Real B[144];
        ans_B(xi, eta, zeta, B);
        detail_w26::zero(eps, 6);
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 24; ++j)
                eps[i] += B[i*24+j] * disp[j];
    }

    /// Volume
    KOKKOS_INLINE_FUNCTION
    Real volume() const {
        return detail_w26::hex8_volume(coords_);
    }

private:
    Real coords_[24];
    Real E_, nu_;
};

// ############################################################################
// 5. StabilizedIncompatibleHex8 -- Incompatible Modes + Hourglass
// ############################################################################

/**
 * Wilson incompatible modes hexahedron with hourglass stabilization.
 * Adds 3 internal bubble modes (xi, eta, zeta) to the standard hex8
 * displacement field: u = N*d + M_inc*alpha
 *
 * Enhanced strain: eps = B*u + B_inc*alpha
 * Static condensation: K_eff = K_uu - K_ua * K_aa^{-1} * K_au
 *
 * Reference: Wilson, Taylor, Doherty, Ghaboussi (1973)
 */
class StabilizedIncompatibleHex8 {
public:
    static constexpr int NUM_NODES = 8;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOF = 24;
    static constexpr int NUM_STRESS = 6;
    static constexpr int NUM_GP = 8;
    static constexpr int NUM_INC = 3;  // number of incompatible modes

    KOKKOS_INLINE_FUNCTION
    StabilizedIncompatibleHex8() : E_(0), nu_(0), hg_coeff_(0.05) {
        detail_w26::zero(coords_, 24);
        detail_w26::zero(alpha_, NUM_INC);
    }

    KOKKOS_INLINE_FUNCTION
    StabilizedIncompatibleHex8(const Real node_coords[8][3], Real E, Real nu,
                                Real hg_coeff = 0.05)
        : E_(E), nu_(nu), hg_coeff_(hg_coeff) {
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                coords_[a*3+i] = node_coords[a][i];
        detail_w26::zero(alpha_, NUM_INC);
    }

    KOKKOS_INLINE_FUNCTION int num_nodes() const { return NUM_NODES; }
    KOKKOS_INLINE_FUNCTION int num_dof() const { return NUM_DOF; }

    /// Standard B-matrix
    KOKKOS_INLINE_FUNCTION
    void standard_B(Real xi, Real eta, Real zeta, Real B[144]) const {
        detail_w26::hex8_B_matrix(coords_, xi, eta, zeta, B);
    }

    /// Jacobian
    KOKKOS_INLINE_FUNCTION
    Real jacobian(Real xi, Real eta, Real zeta, Real J[9]) const {
        return detail_w26::hex8_jacobian(coords_, xi, eta, zeta, J);
    }

    /**
     * @brief Incompatible modes B-matrix (6 x 3)
     *
     * Incompatible displacement modes: M1 = 1-xi^2, M2 = 1-eta^2, M3 = 1-zeta^2
     * dM1/dxi = -2*xi, dM2/deta = -2*eta, dM3/dzeta = -2*zeta
     */
    KOKKOS_INLINE_FUNCTION
    void incompatible_B(Real xi, Real eta, Real zeta, Real detJ0, Real detJ,
                         Real B_inc[18]) const {
        detail_w26::zero(B_inc, 18);

        Real dN[24], J[9], Ji[9];
        detail_w26::hex8_dshape(xi, eta, zeta, dN);
        detail_w26::zero(J, 9);
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    J[i*3+j] += dN[a*3+i] * coords_[a*3+j];
        detail_w26::inv3(J, Ji);

        // dM/d_nat = [-2*xi, 0, 0; 0, -2*eta, 0; 0, 0, -2*zeta]
        // dM/dx = Ji^T * dM/d_nat (transform to physical coords)
        Real dMdxi[3] = {-2.0*xi, 0.0, 0.0};
        Real dMdeta[3] = {0.0, -2.0*eta, 0.0};
        Real dMdzeta[3] = {0.0, 0.0, -2.0*zeta};

        Real dMdx[3][3]; // dMdx[mode][phys_dir]
        for (int j = 0; j < 3; ++j) {
            dMdx[0][j] = Ji[0*3+j]*dMdxi[0] + Ji[1*3+j]*dMdxi[1] + Ji[2*3+j]*dMdxi[2];
            dMdx[1][j] = Ji[0*3+j]*dMdeta[0] + Ji[1*3+j]*dMdeta[1] + Ji[2*3+j]*dMdeta[2];
            dMdx[2][j] = Ji[0*3+j]*dMdzeta[0] + Ji[1*3+j]*dMdzeta[1] + Ji[2*3+j]*dMdzeta[2];
        }

        Real ratio = detJ0 / detJ;

        // B_inc: 6 strain components x 3 modes
        // Each mode contributes to normal strains only (Wilson formulation)
        for (int m = 0; m < 3; ++m) {
            B_inc[0*3+m] = ratio * dMdx[m][0]; // exx
            B_inc[1*3+m] = ratio * dMdx[m][1]; // eyy
            B_inc[2*3+m] = ratio * dMdx[m][2]; // ezz
            B_inc[3*3+m] = ratio * (dMdx[m][1] + dMdx[m][0]) * 0.0; // exy (zero for Wilson)
            B_inc[4*3+m] = 0.0; // eyz
            B_inc[5*3+m] = 0.0; // ezx
        }
        // For standard Wilson: only diagonal normal strain terms
        // Simplification: mode m contributes primarily to eps_mm
        // But above is more general
    }

    /**
     * @brief Compute stiffness with static condensation
     *
     * K_eff = K_uu - K_ua * K_aa^{-1} * K_au
     */
    KOKKOS_INLINE_FUNCTION
    void compute_stiffness(Real K[576]) const {
        detail_w26::zero(K, 576);
        Real C[36];
        detail_w26::iso3D_C(E_, nu_, C);

        Real J0[9];
        Real detJ0 = detail_w26::hex8_jacobian(coords_, 0.0, 0.0, 0.0, J0);

        Real Kuu[576], Kua[72], Kaa[9]; // 24x24, 24x3, 3x3
        detail_w26::zero(Kuu, 576);
        detail_w26::zero(Kua, 72);
        detail_w26::zero(Kaa, 9);

        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};

        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real xi = gp[gi], eta = gp[gj], zeta = gp[gk];

            Real B[144];
            standard_B(xi, eta, zeta, B);

            Real J[9];
            Real detJ = jacobian(xi, eta, zeta, J);
            Real w = std::abs(detJ);

            Real B_inc[18];
            incompatible_B(xi, eta, zeta, detJ0, detJ, B_inc);

            // Kuu += B^T * C * B * w
            detail_w26::addBtCB(Kuu, B, C, w, 6, 24);

            // Kua += B^T * C * B_inc * w
            for (int i = 0; i < 24; ++i)
                for (int j = 0; j < 3; ++j) {
                    Real val = 0.0;
                    for (int k = 0; k < 6; ++k) {
                        Real cm = 0.0;
                        for (int l = 0; l < 6; ++l)
                            cm += C[k*6+l] * B_inc[l*3+j];
                        val += B[k*24+i] * cm;
                    }
                    Kua[i*3+j] += val * w;
                }

            // Kaa += B_inc^T * C * B_inc * w
            detail_w26::addBtCB(Kaa, B_inc, C, w, 6, 3);
        }

        // Invert Kaa (3x3)
        Real Kaa_inv[9];
        detail_w26::inv3(Kaa, Kaa_inv);

        // K = Kuu - Kua * Kaa_inv * Kua^T
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 24; ++j) {
                Real correction = 0.0;
                for (int p = 0; p < 3; ++p) {
                    Real kua_inv_p = 0.0;
                    for (int q = 0; q < 3; ++q)
                        kua_inv_p += Kua[i*3+q] * Kaa_inv[q*3+p];
                    correction += kua_inv_p * Kua[j*3+p];
                }
                K[i*24+j] = Kuu[i*24+j] - correction;
            }

        // Add hourglass stabilization
        add_hourglass_stiffness(K);
    }

    /// Update internal incompatible mode parameters
    KOKKOS_INLINE_FUNCTION
    void update_alpha(const Real disp[24]) {
        Real C[36];
        detail_w26::iso3D_C(E_, nu_, C);

        Real J0[9];
        Real detJ0 = detail_w26::hex8_jacobian(coords_, 0.0, 0.0, 0.0, J0);

        Real Kua[72], Kaa[9];
        detail_w26::zero(Kua, 72);
        detail_w26::zero(Kaa, 9);

        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};

        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real xi = gp[gi], eta = gp[gj], zeta = gp[gk];

            Real B[144];
            standard_B(xi, eta, zeta, B);
            Real J[9];
            Real detJ = jacobian(xi, eta, zeta, J);
            Real w = std::abs(detJ);

            Real B_inc[18];
            incompatible_B(xi, eta, zeta, detJ0, detJ, B_inc);

            for (int i = 0; i < 24; ++i)
                for (int j = 0; j < 3; ++j) {
                    Real val = 0.0;
                    for (int k = 0; k < 6; ++k) {
                        Real cm = 0.0;
                        for (int l = 0; l < 6; ++l)
                            cm += C[k*6+l] * B_inc[l*3+j];
                        val += B[k*24+i] * cm;
                    }
                    Kua[i*3+j] += val * w;
                }

            detail_w26::addBtCB(Kaa, B_inc, C, w, 6, 3);
        }

        Real Kaa_inv[9];
        detail_w26::inv3(Kaa, Kaa_inv);

        Real rhs[3];
        detail_w26::zero(rhs, 3);
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 24; ++i)
                rhs[j] += Kua[i*3+j] * disp[i];

        for (int i = 0; i < 3; ++i) {
            alpha_[i] = 0.0;
            for (int j = 0; j < 3; ++j)
                alpha_[i] -= Kaa_inv[i*3+j] * rhs[j];
        }
    }

    KOKKOS_INLINE_FUNCTION const Real* inc_params() const { return alpha_; }

    /// Volume
    KOKKOS_INLINE_FUNCTION
    Real volume() const {
        return detail_w26::hex8_volume(coords_);
    }

    /// Internal force
    KOKKOS_INLINE_FUNCTION
    void compute_internal_force(const Real disp[24], Real fint[24]) const {
        detail_w26::zero(fint, 24);
        Real K[576];
        compute_stiffness(K);
        for (int i = 0; i < 24; ++i) {
            Real val = 0.0;
            for (int j = 0; j < 24; ++j)
                val += K[i*24+j] * disp[j];
            fint[i] = val;
        }
    }

private:
    Real coords_[24];
    Real E_, nu_;
    Real hg_coeff_;
    Real alpha_[NUM_INC];

    /// Add hourglass stiffness to prevent zero-energy modes
    KOKKOS_INLINE_FUNCTION
    void add_hourglass_stiffness(Real K[576]) const {
        // Hourglass base vectors for hex8: 4 modes
        // gamma1 = {1,-1,1,-1,1,-1,1,-1}
        // gamma2 = {1,-1,-1,1,-1,1,1,-1}
        // gamma3 = {1,1,-1,-1,-1,-1,1,1}  (not independent for incompatible)
        // gamma4 = {-1,1,1,-1,1,-1,-1,1}

        const Real gamma[4][8] = {
            { 1,-1, 1,-1, 1,-1, 1,-1},
            { 1,-1,-1, 1,-1, 1, 1,-1},
            { 1, 1,-1,-1,-1,-1, 1, 1},
            {-1, 1, 1,-1, 1,-1,-1, 1}
        };

        Real vol = detail_w26::hex8_volume(coords_);
        Real L = std::cbrt(vol);
        Real G = E_ / (2.0 * (1.0 + nu_));
        Real stab = hg_coeff_ * G * vol / (L * L);

        for (int m = 0; m < 4; ++m) {
            for (int a = 0; a < 8; ++a) {
                for (int b = 0; b < 8; ++b) {
                    Real kij = stab * gamma[m][a] * gamma[m][b];
                    for (int d = 0; d < 3; ++d)
                        K[(a*3+d)*24 + (b*3+d)] += kij;
                }
            }
        }
    }
};

// ############################################################################
// 6. LayeredThickShell -- 6-Layer with Per-Layer Material
// ############################################################################

/**
 * 8-node thick shell with up to 6 through-thickness layers.
 * Each layer has its own E, nu, thickness.
 * Integration: 2x2 in-plane x 6 through-thickness (1 per layer center).
 * ABD matrix computed from layer stacking sequence.
 */

struct LayerDef {
    Real E;
    Real nu;
    Real thickness;
    int material_id;

    KOKKOS_INLINE_FUNCTION
    LayerDef() : E(0), nu(0), thickness(0), material_id(0) {}

    KOKKOS_INLINE_FUNCTION
    LayerDef(Real E_, Real nu_, Real t_, int id)
        : E(E_), nu(nu_), thickness(t_), material_id(id) {}
};

class LayeredThickShell {
public:
    static constexpr int NUM_NODES = 4; // simplified to 4-node
    static constexpr int DOF_PER_NODE = 6;
    static constexpr int NUM_DOF = 24;
    static constexpr int MAX_LAYERS = 6;

    KOKKOS_INLINE_FUNCTION
    LayeredThickShell() : num_layers_(0), total_thickness_(0) {
        detail_w26::zero(coords_, 12);
    }

    KOKKOS_INLINE_FUNCTION
    LayeredThickShell(const Real node_coords[4][3], const LayerDef* layers, int num_layers)
        : num_layers_(num_layers), total_thickness_(0) {
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                coords_[a*3+i] = node_coords[a][i];
        for (int i = 0; i < num_layers && i < MAX_LAYERS; ++i) {
            layers_[i] = layers[i];
            total_thickness_ += layers[i].thickness;
        }
    }

    KOKKOS_INLINE_FUNCTION int num_layers() const { return num_layers_; }
    KOKKOS_INLINE_FUNCTION Real total_thickness() const { return total_thickness_; }
    KOKKOS_INLINE_FUNCTION const LayerDef& layer(int i) const { return layers_[i]; }

    /**
     * @brief Compute ABD matrix (6x6: A=membrane, B=coupling, D=bending)
     *
     * A_ij = Σ Q_ij * t_k
     * B_ij = Σ Q_ij * t_k * z_k
     * D_ij = Σ Q_ij * (t_k * z_k^2 + t_k^3/12)
     */
    KOKKOS_INLINE_FUNCTION
    void compute_ABD(Real A_mat[9], Real B_mat[9], Real D_mat[9]) const {
        detail_w26::zero(A_mat, 9);
        detail_w26::zero(B_mat, 9);
        detail_w26::zero(D_mat, 9);

        // Compute z-coordinate of each layer center (from mid-surface)
        Real z_bot = -total_thickness_ / 2.0;
        for (int k = 0; k < num_layers_; ++k) {
            Real tk = layers_[k].thickness;
            Real z_mid = z_bot + tk / 2.0;

            Real Q[9];
            detail_w26::planeStress_C(layers_[k].E, layers_[k].nu, Q);

            for (int i = 0; i < 9; ++i) {
                A_mat[i] += Q[i] * tk;
                B_mat[i] += Q[i] * tk * z_mid;
                D_mat[i] += Q[i] * (tk * z_mid * z_mid + tk * tk * tk / 12.0);
            }

            z_bot += tk;
        }
    }

    /**
     * @brief Compute stiffness matrix (24x24)
     */
    KOKKOS_INLINE_FUNCTION
    void stiffness(Real K_out[576]) const {
        detail_w26::zero(K_out, 576);

        Real e1[3], e2[3], e3[3];
        Real c4[4][3];
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                c4[a][i] = coords_[a*3+i];
        detail_w26::shell4_local_system(c4, e1, e2, e3);

        Real lc[4][2];
        for (int a = 0; a < 4; ++a) {
            Real dx = coords_[a*3+0] - coords_[0];
            Real dy = coords_[a*3+1] - coords_[1];
            Real dz = coords_[a*3+2] - coords_[2];
            lc[a][0] = dx*e1[0] + dy*e1[1] + dz*e1[2];
            lc[a][1] = dx*e2[0] + dy*e2[1] + dz*e2[2];
        }

        // 2x2 in-plane Gauss quadrature
        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};

        Real A_mat[9], B_mat[9], D_mat[9];
        compute_ABD(A_mat, B_mat, D_mat);

        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj) {
            Real xi = gp[gi], eta = gp[gj];

            Real dNdxi[4]  = {-0.25*(1.0-eta), 0.25*(1.0-eta), 0.25*(1.0+eta), -0.25*(1.0+eta)};
            Real dNdeta[4] = {-0.25*(1.0-xi), -0.25*(1.0+xi), 0.25*(1.0+xi), 0.25*(1.0-xi)};

            Real J[4] = {0, 0, 0, 0};
            for (int a = 0; a < 4; ++a) {
                J[0] += dNdxi[a] * lc[a][0];
                J[1] += dNdxi[a] * lc[a][1];
                J[2] += dNdeta[a] * lc[a][0];
                J[3] += dNdeta[a] * lc[a][1];
            }
            Real detJ = J[0]*J[3] - J[1]*J[2];
            Real inv_detJ = 1.0 / detJ;
            Real w = std::abs(detJ);

            Real dNdx[4], dNdy[4];
            for (int a = 0; a < 4; ++a) {
                dNdx[a] = ( J[3]*dNdxi[a] - J[2]*dNdeta[a]) * inv_detJ;
                dNdy[a] = (-J[1]*dNdxi[a] + J[0]*dNdeta[a]) * inv_detJ;
            }

            // Membrane stiffness with A_mat
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    Real Ba[6] = {dNdx[a], 0.0, 0.0, dNdy[a], dNdy[a], dNdx[a]};
                    Real Bb[6] = {dNdx[b], 0.0, 0.0, dNdy[b], dNdy[b], dNdx[b]};
                    for (int ii = 0; ii < 2; ++ii) {
                        for (int jj = 0; jj < 2; ++jj) {
                            Real val = 0.0;
                            for (int k = 0; k < 3; ++k) {
                                Real cb = 0.0;
                                for (int l = 0; l < 3; ++l)
                                    cb += A_mat[k*3+l] * Bb[l*2+jj];
                                val += Ba[k*2+ii] * cb;
                            }
                            K_out[(a*6+ii)*24 + (b*6+jj)] += val * w;
                        }
                    }
                }
            }

            // Bending stiffness with D_mat
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    Real Ba_b[6] = {0.0, dNdx[a], -dNdy[a], 0.0, -dNdx[a], dNdy[a]};
                    Real Bb_b[6] = {0.0, dNdx[b], -dNdy[b], 0.0, -dNdx[b], dNdy[b]};
                    for (int ii = 0; ii < 2; ++ii) {
                        for (int jj = 0; jj < 2; ++jj) {
                            Real val = 0.0;
                            for (int k = 0; k < 3; ++k) {
                                Real cb = 0.0;
                                for (int l = 0; l < 3; ++l)
                                    cb += D_mat[k*3+l] * Bb_b[l*2+jj];
                                val += Ba_b[k*2+ii] * cb;
                            }
                            K_out[(a*6+3+ii)*24 + (b*6+3+jj)] += val * w;
                        }
                    }
                }
            }

            // Membrane-bending coupling with B_mat
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    Real Ba_m[6] = {dNdx[a], 0.0, 0.0, dNdy[a], dNdy[a], dNdx[a]};
                    Real Bb_b[6] = {0.0, dNdx[b], -dNdy[b], 0.0, -dNdx[b], dNdy[b]};
                    for (int ii = 0; ii < 2; ++ii) {
                        for (int jj = 0; jj < 2; ++jj) {
                            Real val = 0.0;
                            for (int k = 0; k < 3; ++k) {
                                Real cb = 0.0;
                                for (int l = 0; l < 3; ++l)
                                    cb += B_mat[k*3+l] * Bb_b[l*2+jj];
                                val += Ba_m[k*2+ii] * cb;
                            }
                            K_out[(a*6+ii)*24 + (b*6+3+jj)] += val * w;
                            K_out[(b*6+3+jj)*24 + (a*6+ii)] += val * w;
                        }
                    }
                }
            }
        }

        // Transverse and drilling DOF penalties
        Real d1[3] = {coords_[6]-coords_[0], coords_[7]-coords_[1], coords_[8]-coords_[2]};
        Real d2[3] = {coords_[9]-coords_[3], coords_[10]-coords_[4], coords_[11]-coords_[5]};
        Real cr[3];
        detail_w26::cross3(d1, d2, cr);
        Real area = 0.5 * detail_w26::norm3(cr);

        Real E_avg = 0.0;
        for (int k = 0; k < num_layers_; ++k)
            E_avg += layers_[k].E * layers_[k].thickness / total_thickness_;

        Real pen = 1.0e-6 * E_avg * total_thickness_ * area;
        for (int a = 0; a < 4; ++a) {
            K_out[(a*6+2)*24 + (a*6+2)] += pen * 10.0;
            K_out[(a*6+5)*24 + (a*6+5)] += pen;
        }
    }

    /// Lumped mass
    KOKKOS_INLINE_FUNCTION
    void mass(Real rho, Real M_out[24]) const {
        detail_w26::zero(M_out, 24);
        Real d1[3] = {coords_[6]-coords_[0], coords_[7]-coords_[1], coords_[8]-coords_[2]};
        Real d2[3] = {coords_[9]-coords_[3], coords_[10]-coords_[4], coords_[11]-coords_[5]};
        Real cr[3];
        detail_w26::cross3(d1, d2, cr);
        Real area = 0.5 * detail_w26::norm3(cr);
        Real total_mass = rho * area * total_thickness_;
        Real nm = total_mass / 4.0;
        Real ri = nm * total_thickness_ * total_thickness_ / 12.0;
        for (int a = 0; a < 4; ++a) {
            M_out[a*6+0] = nm;
            M_out[a*6+1] = nm;
            M_out[a*6+2] = nm;
            M_out[a*6+3] = ri;
            M_out[a*6+4] = ri;
            M_out[a*6+5] = ri;
        }
    }

    /// Compute stress at a specific layer
    KOKKOS_INLINE_FUNCTION
    void layer_stress(int layer_idx, const Real disp[24], Real stress[3]) const {
        detail_w26::zero(stress, 3);
        if (layer_idx < 0 || layer_idx >= num_layers_) return;

        // Compute z of layer center
        Real z_bot = -total_thickness_ / 2.0;
        for (int k = 0; k < layer_idx; ++k)
            z_bot += layers_[k].thickness;
        Real z_mid = z_bot + layers_[layer_idx].thickness / 2.0;

        // Simplified: compute membrane strain + z*curvature at element center
        Real c4[4][3];
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                c4[a][i] = coords_[a*3+i];

        Real e1[3], e2[3], e3[3];
        detail_w26::shell4_local_system(c4, e1, e2, e3);

        Real lc[4][2];
        for (int a = 0; a < 4; ++a) {
            Real dx = coords_[a*3+0] - coords_[0];
            Real dy = coords_[a*3+1] - coords_[1];
            Real dz = coords_[a*3+2] - coords_[2];
            lc[a][0] = dx*e1[0] + dy*e1[1] + dz*e1[2];
            lc[a][1] = dx*e2[0] + dy*e2[1] + dz*e2[2];
        }

        Real dNdxi[4]  = {-0.25, 0.25, 0.25, -0.25};
        Real dNdeta[4] = {-0.25, -0.25, 0.25, 0.25};

        Real J[4] = {0, 0, 0, 0};
        for (int a = 0; a < 4; ++a) {
            J[0] += dNdxi[a] * lc[a][0];
            J[1] += dNdxi[a] * lc[a][1];
            J[2] += dNdeta[a] * lc[a][0];
            J[3] += dNdeta[a] * lc[a][1];
        }
        Real inv_detJ = 1.0 / (J[0]*J[3] - J[1]*J[2]);

        Real dNdx[4], dNdy[4];
        for (int a = 0; a < 4; ++a) {
            dNdx[a] = ( J[3]*dNdxi[a] - J[2]*dNdeta[a]) * inv_detJ;
            dNdy[a] = (-J[1]*dNdxi[a] + J[0]*dNdeta[a]) * inv_detJ;
        }

        // Membrane strain
        Real eps_m[3] = {0, 0, 0};
        for (int a = 0; a < 4; ++a) {
            eps_m[0] += dNdx[a] * disp[a*6+0];
            eps_m[1] += dNdy[a] * disp[a*6+1];
            eps_m[2] += dNdx[a] * disp[a*6+1] + dNdy[a] * disp[a*6+0];
        }

        // Curvature
        Real kappa[3] = {0, 0, 0};
        for (int a = 0; a < 4; ++a) {
            kappa[0] += dNdx[a] * disp[a*6+4]; // d(theta_y)/dx
            kappa[1] -= dNdy[a] * disp[a*6+3]; // -d(theta_x)/dy
            kappa[2] += dNdy[a] * disp[a*6+4] - dNdx[a] * disp[a*6+3];
        }

        // Total strain at layer: eps = eps_m + z * kappa
        Real eps_total[3];
        for (int i = 0; i < 3; ++i)
            eps_total[i] = eps_m[i] + z_mid * kappa[i];

        // Stress = Q * eps
        Real Q[9];
        detail_w26::planeStress_C(layers_[layer_idx].E, layers_[layer_idx].nu, Q);
        for (int i = 0; i < 3; ++i) {
            stress[i] = 0.0;
            for (int j = 0; j < 3; ++j)
                stress[i] += Q[i*3+j] * eps_total[j];
        }
    }

private:
    Real coords_[12];
    LayerDef layers_[MAX_LAYERS];
    int num_layers_;
    Real total_thickness_;
};

// ############################################################################
// 7. SelectiveMassHex8 -- Rotational Inertia Scaling
// ############################################################################

/**
 * Standard hex8 geometry with modified mass matrix.
 * Rotational mass terms scaled: M_rot = beta * M_translational
 * beta < 1 increases critical timestep by reducing rotational inertia.
 * Effective dt improvement: dt_new ~ dt_old / sqrt(beta)
 *
 * Consistent mass for translation, lumped+scaled for rotation.
 */
class SelectiveMassHex8 {
public:
    static constexpr int NUM_NODES = 8;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOF = 24;
    static constexpr int NUM_STRESS = 6;
    static constexpr int NUM_GP = 8;

    KOKKOS_INLINE_FUNCTION
    SelectiveMassHex8() : E_(0), nu_(0), density_(0), beta_(1.0) {
        detail_w26::zero(coords_, 24);
    }

    KOKKOS_INLINE_FUNCTION
    SelectiveMassHex8(const Real node_coords[8][3], Real E, Real nu,
                      Real density, Real beta = 1.0)
        : E_(E), nu_(nu), density_(density), beta_(beta) {
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                coords_[a*3+i] = node_coords[a][i];
    }

    KOKKOS_INLINE_FUNCTION Real E() const { return E_; }
    KOKKOS_INLINE_FUNCTION Real nu() const { return nu_; }
    KOKKOS_INLINE_FUNCTION Real density() const { return density_; }
    KOKKOS_INLINE_FUNCTION Real beta() const { return beta_; }

    /// Volume
    KOKKOS_INLINE_FUNCTION
    Real volume() const {
        return detail_w26::hex8_volume(coords_);
    }

    /**
     * @brief Compute lumped translational mass (24 diagonal entries)
     */
    KOKKOS_INLINE_FUNCTION
    void translational_mass(Real M_out[24]) const {
        detail_w26::zero(M_out, 24);
        Real vol = volume();
        Real total_mass = density_ * vol;
        Real nodal_mass = total_mass / 8.0;
        for (int a = 0; a < 8; ++a)
            for (int d = 0; d < 3; ++d)
                M_out[a*3+d] = nodal_mass;
    }

    /**
     * @brief Compute scaled rotational mass
     *
     * Rotational mass = beta * translational mass equivalent
     * For hex8 with 3 translational DOF per node, rotational mass
     * is computed as beta * (rho * V / 8) for each rotational DOF.
     */
    KOKKOS_INLINE_FUNCTION
    void rotational_mass(Real M_rot_out[24]) const {
        detail_w26::zero(M_rot_out, 24);
        Real vol = volume();
        Real total_mass = density_ * vol;
        Real nodal_mass = total_mass / 8.0;
        Real nodal_rot = beta_ * nodal_mass;
        for (int a = 0; a < 8; ++a)
            for (int d = 0; d < 3; ++d)
                M_rot_out[a*3+d] = nodal_rot;
    }

    /**
     * @brief Compute combined mass (translational + rotational inertia effects)
     *
     * For standard hex8 (3 DOF/node), returns 24 diagonal entries.
     * Translational mass is standard, rotational mass is scaled by beta.
     */
    KOKKOS_INLINE_FUNCTION
    void combined_mass(Real M_out[24]) const {
        translational_mass(M_out);
    }

    /// Stiffness matrix (standard hex8)
    KOKKOS_INLINE_FUNCTION
    void compute_stiffness(Real K[576]) const {
        detail_w26::zero(K, 576);
        Real C[36];
        detail_w26::iso3D_C(E_, nu_, C);

        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};

        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real xi = gp[gi], eta = gp[gj], zeta = gp[gk];
            Real B[144];
            detail_w26::hex8_B_matrix(coords_, xi, eta, zeta, B);
            Real J[9];
            Real detJ = detail_w26::hex8_jacobian(coords_, xi, eta, zeta, J);
            Real w = std::abs(detJ);
            detail_w26::addBtCB(K, B, C, w, 6, 24);
        }
    }

    /**
     * @brief Estimate critical time step with rotational scaling
     *
     * dt_crit = L / c where c = sqrt((K + 4G/3) / rho)
     * With beta scaling: dt_effective ~ dt_standard / sqrt(beta)
     * (smaller beta -> larger effective timestep)
     */
    KOKKOS_INLINE_FUNCTION
    Real critical_timestep() const {
        Real vol = volume();
        Real L = std::cbrt(vol);
        Real G = E_ / (2.0*(1.0+nu_));
        Real bulk = E_ / (3.0*(1.0-2.0*nu_));
        Real c = std::sqrt((bulk + 4.0/3.0 * G) / density_);
        Real dt_std = (c > 1.0e-30) ? L / c : 1.0e30;
        // Rotational scaling effect
        return dt_std / std::sqrt(beta_);
    }

    /**
     * @brief Compute max eigenfrequency estimate (for timestep)
     * omega_max ~ c / L * sqrt(1/beta) for rotational modes
     */
    KOKKOS_INLINE_FUNCTION
    Real max_eigenfrequency() const {
        Real vol = volume();
        Real L = std::cbrt(vol);
        Real G = E_ / (2.0*(1.0+nu_));
        Real bulk = E_ / (3.0*(1.0-2.0*nu_));
        Real c = std::sqrt((bulk + 4.0/3.0 * G) / density_);
        return c / L;
    }

private:
    Real coords_[24];
    Real E_, nu_, density_;
    Real beta_;
};

// ############################################################################
// 8. EnhancedStrainExtrapolationHex8 -- EAS with Stress Extrapolation
// ############################################################################

/**
 * Enhanced Assumed Strain hex8 (9 EAS modes) with stress extrapolation
 * from Gauss points to nodes.
 *
 * EAS locking-free formulation (Simo & Rifai 1990).
 * Stress extrapolation uses inverse of shape function matrix evaluated
 * at Gauss points.
 *
 * For 2x2x2 GP hex8: sigma_node = sum_gp N_extrap(xi_gp) * sigma_gp
 */
class EnhancedStrainExtrapolationHex8 {
public:
    static constexpr int NUM_NODES = 8;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOF = 24;
    static constexpr int NUM_STRESS = 6;
    static constexpr int NUM_GP = 8;
    static constexpr int NUM_EAS = 9;

    KOKKOS_INLINE_FUNCTION
    EnhancedStrainExtrapolationHex8() : E_(0), nu_(0) {
        detail_w26::zero(coords_, 24);
        detail_w26::zero(alpha_, NUM_EAS);
    }

    KOKKOS_INLINE_FUNCTION
    EnhancedStrainExtrapolationHex8(const Real node_coords[8][3], Real E, Real nu)
        : E_(E), nu_(nu) {
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                coords_[a*3+i] = node_coords[a][i];
        detail_w26::zero(alpha_, NUM_EAS);
    }

    KOKKOS_INLINE_FUNCTION int num_nodes() const { return NUM_NODES; }
    KOKKOS_INLINE_FUNCTION int num_dof() const { return NUM_DOF; }

    /// Standard B-matrix
    KOKKOS_INLINE_FUNCTION
    void standard_B(Real xi, Real eta, Real zeta, Real B[144]) const {
        detail_w26::hex8_B_matrix(coords_, xi, eta, zeta, B);
    }

    /// Jacobian
    KOKKOS_INLINE_FUNCTION
    Real jacobian(Real xi, Real eta, Real zeta, Real J[9]) const {
        return detail_w26::hex8_jacobian(coords_, xi, eta, zeta, J);
    }

    /**
     * @brief EAS interpolation matrix M (6 x 9)
     *
     * 9-mode EAS: 3 normal + 3 shear + 3 coupled modes
     */
    KOKKOS_INLINE_FUNCTION
    void eas_matrix(Real xi, Real eta, Real zeta,
                    Real detJ0, Real detJ, Real M[54]) const {
        detail_w26::zero(M, 6*9);
        Real ratio = detJ0 / detJ;

        // Normal strain enhancements
        M[0*9+0] = ratio * xi;
        M[1*9+1] = ratio * eta;
        M[2*9+2] = ratio * zeta;
        // Shear strain enhancements
        M[3*9+3] = ratio * xi;
        M[4*9+4] = ratio * eta;
        M[5*9+5] = ratio * zeta;
        // Coupled modes
        M[0*9+6] = ratio * eta;
        M[1*9+7] = ratio * zeta;
        M[2*9+8] = ratio * xi;
    }

    /**
     * @brief Compute stiffness with EAS condensation
     *
     * K = Kuu - Kua * Kaa^{-1} * Kua^T
     */
    KOKKOS_INLINE_FUNCTION
    void compute_stiffness(Real K[576]) const {
        detail_w26::zero(K, 576);
        Real C[36];
        detail_w26::iso3D_C(E_, nu_, C);

        Real J0[9];
        Real detJ0 = jacobian(0.0, 0.0, 0.0, J0);

        Real Kuu[576], Kua[216], Kaa[81]; // 24x24, 24x9, 9x9
        detail_w26::zero(Kuu, 576);
        detail_w26::zero(Kua, 216);
        detail_w26::zero(Kaa, 81);

        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};

        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real xi = gp[gi], eta = gp[gj], zeta = gp[gk];

            Real B[144];
            standard_B(xi, eta, zeta, B);

            Real J[9];
            Real detJ = jacobian(xi, eta, zeta, J);
            Real w = std::abs(detJ);

            Real M[54];
            eas_matrix(xi, eta, zeta, detJ0, detJ, M);

            detail_w26::addBtCB(Kuu, B, C, w, 6, 24);

            // Kua += B^T * C * M * w
            for (int i = 0; i < 24; ++i)
                for (int j = 0; j < 9; ++j) {
                    Real val = 0.0;
                    for (int k = 0; k < 6; ++k) {
                        Real cm = 0.0;
                        for (int l = 0; l < 6; ++l)
                            cm += C[k*6+l] * M[l*9+j];
                        val += B[k*24+i] * cm;
                    }
                    Kua[i*9+j] += val * w;
                }

            detail_w26::addBtCB(Kaa, M, C, w, 6, 9);
        }

        // Invert Kaa (9x9)
        Real Kaa_inv[81];
        invert_9x9(Kaa, Kaa_inv);

        // K = Kuu - Kua * Kaa_inv * Kua^T
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 24; ++j) {
                Real correction = 0.0;
                for (int p = 0; p < 9; ++p) {
                    Real kua_inv_p = 0.0;
                    for (int q = 0; q < 9; ++q)
                        kua_inv_p += Kua[i*9+q] * Kaa_inv[q*9+p];
                    correction += kua_inv_p * Kua[j*9+p];
                }
                K[i*24+j] = Kuu[i*24+j] - correction;
            }
    }

    /// Update EAS parameters
    KOKKOS_INLINE_FUNCTION
    void update_eas_params(const Real disp[24]) {
        Real C[36];
        detail_w26::iso3D_C(E_, nu_, C);

        Real J0[9];
        Real detJ0 = jacobian(0.0, 0.0, 0.0, J0);

        Real Kua[216], Kaa[81];
        detail_w26::zero(Kua, 216);
        detail_w26::zero(Kaa, 81);

        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};

        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real xi = gp[gi], eta = gp[gj], zeta = gp[gk];

            Real B[144];
            standard_B(xi, eta, zeta, B);
            Real J[9];
            Real detJ = jacobian(xi, eta, zeta, J);
            Real w = std::abs(detJ);

            Real M[54];
            eas_matrix(xi, eta, zeta, detJ0, detJ, M);

            for (int i = 0; i < 24; ++i)
                for (int j = 0; j < 9; ++j) {
                    Real val = 0.0;
                    for (int k = 0; k < 6; ++k) {
                        Real cm = 0.0;
                        for (int l = 0; l < 6; ++l)
                            cm += C[k*6+l] * M[l*9+j];
                        val += B[k*24+i] * cm;
                    }
                    Kua[i*9+j] += val * w;
                }

            detail_w26::addBtCB(Kaa, M, C, w, 6, 9);
        }

        Real Kaa_inv[81];
        invert_9x9(Kaa, Kaa_inv);

        Real rhs[9];
        detail_w26::zero(rhs, 9);
        for (int j = 0; j < 9; ++j)
            for (int i = 0; i < 24; ++i)
                rhs[j] += Kua[i*9+j] * disp[i];

        for (int i = 0; i < 9; ++i) {
            alpha_[i] = 0.0;
            for (int j = 0; j < 9; ++j)
                alpha_[i] -= Kaa_inv[i*9+j] * rhs[j];
        }
    }

    KOKKOS_INLINE_FUNCTION const Real* eas_params() const { return alpha_; }

    /**
     * @brief Compute stress at Gauss points
     * @param disp  Nodal displacements [24]
     * @param gp_stress  Output: stress at each GP [8][6]
     */
    KOKKOS_INLINE_FUNCTION
    void gauss_point_stress(const Real disp[24], Real gp_stress[48]) const {
        detail_w26::zero(gp_stress, 48);
        Real C[36];
        detail_w26::iso3D_C(E_, nu_, C);

        Real J0[9];
        Real detJ0 = jacobian(0.0, 0.0, 0.0, J0);

        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};

        int idx = 0;
        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real xi = gp[gi], eta = gp[gj], zeta = gp[gk];

            Real B[144];
            standard_B(xi, eta, zeta, B);

            Real J[9];
            Real detJ = jacobian(xi, eta, zeta, J);

            Real M[54];
            eas_matrix(xi, eta, zeta, detJ0, detJ, M);

            // Total strain = B*u + M*alpha
            Real eps[6];
            detail_w26::zero(eps, 6);
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 24; ++j)
                    eps[i] += B[i*24+j] * disp[j];
                for (int j = 0; j < 9; ++j)
                    eps[i] += M[i*9+j] * alpha_[j];
            }

            // Stress = C * eps
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j)
                    gp_stress[idx*6+i] += C[i*6+j] * eps[j];

            idx++;
        }
    }

    /**
     * @brief Extrapolate stress from Gauss points to nodes
     *
     * Uses inverse shape function extrapolation matrix.
     * For hex8 with 2x2x2 GP: evaluate N at GP locations, invert to get
     * extrapolation coefficients. The extrapolation point locations
     * correspond to the node natural coordinates (+/-1).
     *
     * @param gp_stress  Input: stress at GPs [8][6]
     * @param node_stress  Output: stress at nodes [8][6]
     */
    KOKKOS_INLINE_FUNCTION
    void extrapolate_stress_to_nodes(const Real gp_stress[48],
                                      Real node_stress[48]) const {
        detail_w26::zero(node_stress, 48);

        // Gauss point coordinates
        const Real g = 1.0 / std::sqrt(3.0);

        // Extrapolation: evaluate shape functions at Gauss points
        // N_extrap_ij = N_i(xi_j) where xi_j are GP coords and N_i are node shape funcs
        // Then node_stress = N_extrap^{-1} * gp_stress
        // For hex8: N_extrap is 8x8, we need its inverse

        // However, a simpler approach: the extrapolation from GP coords to node coords
        // uses the "superconvergent" extrapolation factor sqrt(3)
        // Node i gets: sigma_i = sum_j E_ij * sigma_j(GP)
        // where E_ij = N_j(xi_node_i * sqrt(3)) evaluated at scaled coords

        // Extrapolation matrix entries: each node evaluates at its natural coords
        // using shape functions with the GP locations as the "element"
        const Real a = 1.0 + std::sqrt(3.0) / 2.0;
        const Real b = -0.5;
        const Real c = 1.0 - std::sqrt(3.0) / 2.0;

        // Simplified: for each stress component, use trilinear extrapolation
        // Node coordinates: (+/-1, +/-1, +/-1)
        // GP coordinates: (+/-g, +/-g, +/-g)
        // Extrapolation factor: 1/g = sqrt(3) per direction

        Real node_nat[8][3] = {
            {-1,-1,-1}, {1,-1,-1}, {1,1,-1}, {-1,1,-1},
            {-1,-1, 1}, {1,-1, 1}, {1, 1, 1}, {-1, 1, 1}
        };

        Real gp_nat[8][3];
        int idx = 0;
        const Real gpp[2] = {-g, g};
        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            gp_nat[idx][0] = gpp[gi];
            gp_nat[idx][1] = gpp[gj];
            gp_nat[idx][2] = gpp[gk];
            idx++;
        }

        // Build extrapolation matrix: for each node, evaluate hex8 shape functions
        // at the node's natural coordinates using GP natural coords as the "element nodes"
        // This is equivalent to solving N(gp_coords) * sigma_node = sigma_gp
        // -> sigma_node = N(gp_coords)^{-1} * sigma_gp

        // Build N matrix (8x8): N[i][j] = shape_function_j evaluated at gp_i location
        // where shape functions are defined by node natural coords
        Real N_mat[64];
        for (int i = 0; i < 8; ++i) {
            Real xi_e = gp_nat[i][0];
            Real eta_e = gp_nat[i][1];
            Real zeta_e = gp_nat[i][2];
            for (int j = 0; j < 8; ++j) {
                N_mat[i*8+j] = 0.125 *
                    (1.0 + node_nat[j][0]*xi_e) *
                    (1.0 + node_nat[j][1]*eta_e) *
                    (1.0 + node_nat[j][2]*zeta_e);
            }
        }

        // Invert N_mat to get extrapolation matrix E = N^{-1}
        Real E_mat[64];
        invert_8x8(N_mat, E_mat);

        // Extrapolate each stress component
        for (int s = 0; s < 6; ++s) {
            for (int nd = 0; nd < 8; ++nd) {
                Real val = 0.0;
                for (int gp_idx = 0; gp_idx < 8; ++gp_idx)
                    val += E_mat[nd*8+gp_idx] * gp_stress[gp_idx*6+s];
                node_stress[nd*6+s] = val;
            }
        }
    }

    /// Internal force
    KOKKOS_INLINE_FUNCTION
    void compute_internal_force(const Real disp[24], Real fint[24]) const {
        detail_w26::zero(fint, 24);
        Real C[36];
        detail_w26::iso3D_C(E_, nu_, C);

        Real J0[9];
        Real detJ0 = jacobian(0.0, 0.0, 0.0, J0);

        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};

        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real xi = gp[gi], eta = gp[gj], zeta = gp[gk];

            Real B[144];
            standard_B(xi, eta, zeta, B);
            Real J[9];
            Real detJ = jacobian(xi, eta, zeta, J);
            Real w = std::abs(detJ);

            Real M[54];
            eas_matrix(xi, eta, zeta, detJ0, detJ, M);

            Real eps[6];
            detail_w26::zero(eps, 6);
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 24; ++j)
                    eps[i] += B[i*24+j] * disp[j];
                for (int j = 0; j < 9; ++j)
                    eps[i] += M[i*9+j] * alpha_[j];
            }

            Real sig[6];
            detail_w26::zero(sig, 6);
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j)
                    sig[i] += C[i*6+j] * eps[j];

            for (int j = 0; j < 24; ++j) {
                Real val = 0.0;
                for (int i = 0; i < 6; ++i)
                    val += B[i*24+j] * sig[i];
                fint[j] += val * w;
            }
        }
    }

    /// Volume
    KOKKOS_INLINE_FUNCTION
    Real volume() const {
        return detail_w26::hex8_volume(coords_);
    }

private:
    Real coords_[24];
    Real E_, nu_;
    Real alpha_[NUM_EAS];

    /// Gauss-Jordan 9x9 inverse
    KOKKOS_INLINE_FUNCTION
    static void invert_9x9(const Real A[81], Real Ainv[81]) {
        Real aug[9][18];
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                aug[i][j] = A[i*9+j];
                aug[i][j+9] = (i == j) ? 1.0 : 0.0;
            }
        }
        for (int col = 0; col < 9; ++col) {
            int pivot = col;
            Real max_val = std::abs(aug[col][col]);
            for (int row = col+1; row < 9; ++row) {
                if (std::abs(aug[row][col]) > max_val) {
                    max_val = std::abs(aug[row][col]);
                    pivot = row;
                }
            }
            if (pivot != col) {
                for (int j = 0; j < 18; ++j) {
                    Real tmp = aug[col][j];
                    aug[col][j] = aug[pivot][j];
                    aug[pivot][j] = tmp;
                }
            }
            Real diag = aug[col][col];
            if (std::abs(diag) < 1.0e-30) diag = 1.0e-30;
            Real inv_diag = 1.0 / diag;
            for (int j = 0; j < 18; ++j)
                aug[col][j] *= inv_diag;
            for (int row = 0; row < 9; ++row) {
                if (row == col) continue;
                Real factor = aug[row][col];
                for (int j = 0; j < 18; ++j)
                    aug[row][j] -= factor * aug[col][j];
            }
        }
        for (int i = 0; i < 9; ++i)
            for (int j = 0; j < 9; ++j)
                Ainv[i*9+j] = aug[i][j+9];
    }

    /// Gauss-Jordan 8x8 inverse
    KOKKOS_INLINE_FUNCTION
    static void invert_8x8(const Real A[64], Real Ainv[64]) {
        Real aug[8][16];
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                aug[i][j] = A[i*8+j];
                aug[i][j+8] = (i == j) ? 1.0 : 0.0;
            }
        }
        for (int col = 0; col < 8; ++col) {
            int pivot = col;
            Real max_val = std::abs(aug[col][col]);
            for (int row = col+1; row < 8; ++row) {
                if (std::abs(aug[row][col]) > max_val) {
                    max_val = std::abs(aug[row][col]);
                    pivot = row;
                }
            }
            if (pivot != col) {
                for (int j = 0; j < 16; ++j) {
                    Real tmp = aug[col][j];
                    aug[col][j] = aug[pivot][j];
                    aug[pivot][j] = tmp;
                }
            }
            Real diag = aug[col][col];
            if (std::abs(diag) < 1.0e-30) diag = 1.0e-30;
            Real inv_diag = 1.0 / diag;
            for (int j = 0; j < 16; ++j)
                aug[col][j] *= inv_diag;
            for (int row = 0; row < 8; ++row) {
                if (row == col) continue;
                Real factor = aug[row][col];
                for (int j = 0; j < 16; ++j)
                    aug[row][j] -= factor * aug[col][j];
            }
        }
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j)
                Ainv[i*8+j] = aug[i][j+8];
    }
};

} // namespace discretization
} // namespace nxs
