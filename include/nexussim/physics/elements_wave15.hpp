#pragma once

/**
 * @file elements_wave15.hpp
 * @brief Wave 15: Element Formulation Expansion
 *
 * Seven new element formulations:
 *  1. ThickShell8  - 8-node thick shell (solid-like, 3D stress)
 *  2. ThickShell6  - 6-node thick shell (wedge-shaped)
 *  3. DKTShell     - Discrete Kirchhoff Triangle (bending)
 *  4. DKQShell     - Discrete Kirchhoff Quadrilateral (bending)
 *  5. PlaneElement  - 2D plane-stress / plane-strain
 *  6. AxisymmetricElement - Axisymmetric solid of revolution
 *  7. ConnectorElement    - Spot weld / rivet / fastener
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

namespace elements {

// ============================================================================
// Utility: small matrix/vector helpers (GPU-safe, no STL)
// ============================================================================

namespace detail {

KOKKOS_INLINE_FUNCTION
void zero(Real* a, int n) { for (int i = 0; i < n; ++i) a[i] = 0.0; }

KOKKOS_INLINE_FUNCTION
Real dot(const Real* a, const Real* b, int n) {
    Real s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
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

/// 3x3 determinant
KOKKOS_INLINE_FUNCTION
Real det3(const Real* J) {
    return J[0]*(J[4]*J[8]-J[5]*J[7])
         - J[1]*(J[3]*J[8]-J[5]*J[6])
         + J[2]*(J[3]*J[7]-J[4]*J[6]);
}

/// 3x3 inverse, stores in Ji, returns det
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

/// 2x2 determinant
KOKKOS_INLINE_FUNCTION
Real det2(const Real* J) { return J[0]*J[3] - J[1]*J[2]; }

/// 2x2 inverse
KOKKOS_INLINE_FUNCTION
Real inv2(const Real* J, Real* Ji) {
    Real d = det2(J);
    Real id = 1.0 / d;
    Ji[0] =  J[3]*id;
    Ji[1] = -J[1]*id;
    Ji[2] = -J[2]*id;
    Ji[3] =  J[0]*id;
    return d;
}

/// mat(m x n) * vec(n) -> out(m)
KOKKOS_INLINE_FUNCTION
void matvec(const Real* A, const Real* x, Real* y, int m, int n) {
    for (int i = 0; i < m; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < n; ++j)
            y[i] += A[i*n + j] * x[j];
    }
}

/// B^T(ndof x nstress) * sigma(nstress) -> fint(ndof)   [B is nstress x ndof]
KOKKOS_INLINE_FUNCTION
void BtSigma(const Real* B, const Real* sigma, Real* f, int nstress, int ndof) {
    for (int j = 0; j < ndof; ++j) {
        f[j] = 0.0;
        for (int i = 0; i < nstress; ++i)
            f[j] += B[i*ndof + j] * sigma[i];
    }
}

/// K += B^T * C * B * w   (nstress x ndof), (nstress x nstress), weight w
KOKKOS_INLINE_FUNCTION
void addBtCB(Real* K, const Real* B, const Real* C, Real w,
             int nstress, int ndof) {
    // CB = C * B  (nstress x ndof)
    // Then K += B^T * CB * w
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

/// Plane stress constitutive matrix (3x3)
KOKKOS_INLINE_FUNCTION
void planeStress_C(Real E, Real nu, Real* C) {
    zero(C, 9);
    Real f = E / (1.0 - nu*nu);
    C[0] = f;      C[1] = f*nu;
    C[3] = f*nu;   C[4] = f;
    C[8] = f*(1.0-nu)/2.0;
}

/// Plane strain constitutive matrix (3x3)
KOKKOS_INLINE_FUNCTION
void planeStrain_C(Real E, Real nu, Real* C) {
    zero(C, 9);
    Real f = E / ((1.0 + nu)*(1.0 - 2.0*nu));
    C[0] = f*(1.0-nu);  C[1] = f*nu;
    C[3] = f*nu;        C[4] = f*(1.0-nu);
    C[8] = f*(1.0-2.0*nu)/2.0;
}

/// Axisymmetric constitutive matrix (4x4)
KOKKOS_INLINE_FUNCTION
void axisym_C(Real E, Real nu, Real* C) {
    zero(C, 16);
    Real f = E / ((1.0 + nu)*(1.0 - 2.0*nu));
    Real d = f*(1.0-nu);
    Real o = f*nu;
    Real g = f*(1.0-2.0*nu)/2.0;
    C[0*4+0] = d; C[0*4+1] = o; C[0*4+2] = o;
    C[1*4+0] = o; C[1*4+1] = d; C[1*4+2] = o;
    C[2*4+0] = o; C[2*4+1] = o; C[2*4+2] = d;
    C[3*4+3] = g;
}

} // namespace detail

// ############################################################################
// 1. ThickShell8  — 8-node thick shell (solid-like hex, full 3D stress)
// ############################################################################

class ThickShell8 {
public:
    static constexpr int NUM_NODES = 8;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOF = 24;
    static constexpr int NUM_STRESS = 6;
    static constexpr int NUM_GP = 8; // 2x2x2

    KOKKOS_INLINE_FUNCTION
    ThickShell8() : E_(0), nu_(0) { detail::zero(coords_, NUM_NODES*3); }

    KOKKOS_INLINE_FUNCTION
    ThickShell8(const Real* node_coords, Real E, Real nu)
        : E_(E), nu_(nu) {
        for (int i = 0; i < NUM_NODES*3; ++i) coords_[i] = node_coords[i];
    }

    KOKKOS_INLINE_FUNCTION int num_nodes() const { return NUM_NODES; }
    KOKKOS_INLINE_FUNCTION int dof_per_node() const { return DOF_PER_NODE; }
    KOKKOS_INLINE_FUNCTION int num_integration_points() const { return NUM_GP; }

    /// Shape functions for 8-node hex: bilinear in-plane, linear through-thickness
    KOKKOS_INLINE_FUNCTION
    void shape_functions(Real xi, Real eta, Real zeta, Real* N) const {
        Real xm = 1.0-xi, xp = 1.0+xi;
        Real em = 1.0-eta, ep = 1.0+eta;
        Real zm = 1.0-zeta, zp = 1.0+zeta;
        N[0] = 0.125*xm*em*zm;
        N[1] = 0.125*xp*em*zm;
        N[2] = 0.125*xp*ep*zm;
        N[3] = 0.125*xm*ep*zm;
        N[4] = 0.125*xm*em*zp;
        N[5] = 0.125*xp*em*zp;
        N[6] = 0.125*xp*ep*zp;
        N[7] = 0.125*xm*ep*zp;
    }

    /// Shape function derivatives dN/d(xi,eta,zeta) — 8x3 row-major
    KOKKOS_INLINE_FUNCTION
    void shape_derivatives(Real xi, Real eta, Real zeta, Real* dN) const {
        const Real s[8] = {-1,1,1,-1,-1,1,1,-1};
        const Real t[8] = {-1,-1,1,1,-1,-1,1,1};
        const Real u[8] = {-1,-1,-1,-1,1,1,1,1};
        for (int i = 0; i < 8; ++i) {
            dN[i*3+0] = 0.125*s[i]*(1.0+t[i]*eta)*(1.0+u[i]*zeta);
            dN[i*3+1] = 0.125*t[i]*(1.0+s[i]*xi)*(1.0+u[i]*zeta);
            dN[i*3+2] = 0.125*u[i]*(1.0+s[i]*xi)*(1.0+t[i]*eta);
        }
    }

    /// Jacobian: J = sum dN_i/d(xi_j) * x_i_k -> J[j][k], returns det
    KOKKOS_INLINE_FUNCTION
    Real jacobian(Real xi, Real eta, Real zeta, Real* J) const {
        Real dN[24];
        shape_derivatives(xi, eta, zeta, dN);
        detail::zero(J, 9);
        for (int a = 0; a < 8; ++a) {
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    J[i*3+j] += dN[a*3+i] * coords_[a*3+j];
        }
        return detail::det3(J);
    }

    /// B-matrix (6 x 24) at a given (xi,eta,zeta)
    KOKKOS_INLINE_FUNCTION
    void strain_displacement_matrix(Real xi, Real eta, Real zeta, Real* B) const {
        Real dN[24], J[9], Ji[9];
        shape_derivatives(xi, eta, zeta, dN);
        detail::zero(J, 9);
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    J[i*3+j] += dN[a*3+i] * coords_[a*3+j];
        detail::inv3(J, Ji);
        // dNdx[a][j] = Ji^T[j][k] * dN[a][k] = Ji[k][j] * dN[a][k]
        Real dNdx[24];
        for (int a = 0; a < 8; ++a) {
            for (int j = 0; j < 3; ++j) {
                dNdx[a*3+j] = 0.0;
                for (int k = 0; k < 3; ++k)
                    dNdx[a*3+j] += Ji[k*3+j] * dN[a*3+k];
            }
        }
        detail::zero(B, 6*24);
        for (int a = 0; a < 8; ++a) {
            int c = a*3;
            Real dx = dNdx[a*3+0], dy = dNdx[a*3+1], dz = dNdx[a*3+2];
            B[0*24+c+0] = dx;  // exx
            B[1*24+c+1] = dy;  // eyy
            B[2*24+c+2] = dz;  // ezz
            B[3*24+c+0] = dy;  B[3*24+c+1] = dx;  // gxy
            B[4*24+c+1] = dz;  B[4*24+c+2] = dy;  // gyz
            B[5*24+c+0] = dz;  B[5*24+c+2] = dx;  // gxz
        }
    }

    /// Gauss points (2x2x2), stores pts[8*3], wts[8]
    KOKKOS_INLINE_FUNCTION
    void gauss_quadrature(Real* pts, Real* wts) const {
        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};
        int idx = 0;
        for (int k = 0; k < 2; ++k)
            for (int j = 0; j < 2; ++j)
                for (int i = 0; i < 2; ++i) {
                    pts[idx*3+0] = gp[i];
                    pts[idx*3+1] = gp[j];
                    pts[idx*3+2] = gp[k];
                    wts[idx] = 1.0;
                    idx++;
                }
    }

    /// Element stiffness matrix (24x24)
    KOKKOS_INLINE_FUNCTION
    void stiffness_matrix(Real* K) const {
        detail::zero(K, 24*24);
        Real C[36];
        detail::iso3D_C(E_, nu_, C);
        Real pts[24], wts[8];
        gauss_quadrature(pts, wts);
        for (int g = 0; g < NUM_GP; ++g) {
            Real B[6*24], J[9];
            strain_displacement_matrix(pts[g*3+0], pts[g*3+1], pts[g*3+2], B);
            Real detJ = jacobian(pts[g*3+0], pts[g*3+1], pts[g*3+2], J);
            Real w = wts[g] * std::abs(detJ);
            detail::addBtCB(K, B, C, w, 6, 24);
        }
    }

    /// Internal force from stress at integration points (stress: 8*6 flat)
    KOKKOS_INLINE_FUNCTION
    void compute_internal_forces(const Real* stress, Real* fint) const {
        detail::zero(fint, 24);
        Real pts[24], wts[8];
        gauss_quadrature(pts, wts);
        for (int g = 0; g < NUM_GP; ++g) {
            Real B[6*24], J[9];
            strain_displacement_matrix(pts[g*3+0], pts[g*3+1], pts[g*3+2], B);
            Real detJ = jacobian(pts[g*3+0], pts[g*3+1], pts[g*3+2], J);
            Real w = wts[g] * std::abs(detJ);
            for (int j = 0; j < 24; ++j) {
                Real val = 0.0;
                for (int i = 0; i < 6; ++i)
                    val += B[i*24+j] * stress[g*6+i];
                fint[j] += val * w;
            }
        }
    }

    /// Volume
    KOKKOS_INLINE_FUNCTION
    Real volume() const {
        Real pts[24], wts[8];
        gauss_quadrature(pts, wts);
        Real vol = 0.0;
        for (int g = 0; g < NUM_GP; ++g) {
            Real J[9];
            Real detJ = jacobian(pts[g*3+0], pts[g*3+1], pts[g*3+2], J);
            vol += wts[g] * std::abs(detJ);
        }
        return vol;
    }

private:
    Real coords_[24];
    Real E_, nu_;
};

// ############################################################################
// 2. ThickShell6  — 6-node thick shell (wedge-shaped)
// ############################################################################

class ThickShell6 {
public:
    static constexpr int NUM_NODES = 6;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOF = 18;
    static constexpr int NUM_STRESS = 6;
    static constexpr int NUM_GP = 6; // 3 in-plane x 2 through-thickness

    KOKKOS_INLINE_FUNCTION
    ThickShell6() : E_(0), nu_(0) { detail::zero(coords_, NUM_NODES*3); }

    KOKKOS_INLINE_FUNCTION
    ThickShell6(const Real* node_coords, Real E, Real nu)
        : E_(E), nu_(nu) {
        for (int i = 0; i < NUM_NODES*3; ++i) coords_[i] = node_coords[i];
    }

    KOKKOS_INLINE_FUNCTION int num_nodes() const { return NUM_NODES; }
    KOKKOS_INLINE_FUNCTION int dof_per_node() const { return DOF_PER_NODE; }
    KOKKOS_INLINE_FUNCTION int num_integration_points() const { return NUM_GP; }

    /// Shape functions: triangular in-plane (L1,L2,L3) x linear through-thickness (zeta)
    /// L1=xi, L2=eta, L3=1-xi-eta, zeta in [-1,1]
    /// Bottom face nodes 0,1,2 at zeta=-1; Top face nodes 3,4,5 at zeta=+1
    KOKKOS_INLINE_FUNCTION
    void shape_functions(Real xi, Real eta, Real zeta, Real* N) const {
        Real L1 = xi, L2 = eta, L3 = 1.0 - xi - eta;
        Real zm = 0.5*(1.0 - zeta), zp = 0.5*(1.0 + zeta);
        N[0] = L1*zm; N[1] = L2*zm; N[2] = L3*zm;
        N[3] = L1*zp; N[4] = L2*zp; N[5] = L3*zp;
    }

    /// Shape function derivatives dN/d(xi,eta,zeta) — 6x3 row-major
    KOKKOS_INLINE_FUNCTION
    void shape_derivatives(Real xi, Real eta, Real zeta, Real* dN) const {
        Real L1 = xi, L2 = eta, L3 = 1.0 - xi - eta;
        Real zm = 0.5*(1.0-zeta), zp = 0.5*(1.0+zeta);
        (void)L1; (void)L2; (void)L3;
        // dN/dxi
        dN[0*3+0] =  zm; dN[1*3+0] =  0.0; dN[2*3+0] = -zm;
        dN[3*3+0] =  zp; dN[4*3+0] =  0.0; dN[5*3+0] = -zp;
        // dN/deta
        dN[0*3+1] =  0.0; dN[1*3+1] =  zm; dN[2*3+1] = -zm;
        dN[3*3+1] =  0.0; dN[4*3+1] =  zp; dN[5*3+1] = -zp;
        // dN/dzeta
        dN[0*3+2] = -0.5*xi;          dN[1*3+2] = -0.5*eta;
        dN[2*3+2] = -0.5*(1.0-xi-eta);
        dN[3*3+2] =  0.5*xi;          dN[4*3+2] =  0.5*eta;
        dN[5*3+2] =  0.5*(1.0-xi-eta);
    }

    KOKKOS_INLINE_FUNCTION
    Real jacobian(Real xi, Real eta, Real zeta, Real* J) const {
        Real dN[18];
        shape_derivatives(xi, eta, zeta, dN);
        detail::zero(J, 9);
        for (int a = 0; a < 6; ++a)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    J[i*3+j] += dN[a*3+i] * coords_[a*3+j];
        return detail::det3(J);
    }

    KOKKOS_INLINE_FUNCTION
    void strain_displacement_matrix(Real xi, Real eta, Real zeta, Real* B) const {
        Real dN[18], J[9], Ji[9];
        shape_derivatives(xi, eta, zeta, dN);
        detail::zero(J, 9);
        for (int a = 0; a < 6; ++a)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    J[i*3+j] += dN[a*3+i] * coords_[a*3+j];
        detail::inv3(J, Ji);
        Real dNdx[18];
        for (int a = 0; a < 6; ++a)
            for (int j = 0; j < 3; ++j) {
                dNdx[a*3+j] = 0.0;
                for (int k = 0; k < 3; ++k)
                    dNdx[a*3+j] += Ji[k*3+j] * dN[a*3+k];
            }
        detail::zero(B, 6*18);
        for (int a = 0; a < 6; ++a) {
            int c = a*3;
            Real dx = dNdx[a*3+0], dy = dNdx[a*3+1], dz = dNdx[a*3+2];
            B[0*18+c+0] = dx;
            B[1*18+c+1] = dy;
            B[2*18+c+2] = dz;
            B[3*18+c+0] = dy; B[3*18+c+1] = dx;
            B[4*18+c+1] = dz; B[4*18+c+2] = dy;
            B[5*18+c+0] = dz; B[5*18+c+2] = dx;
        }
    }

    /// Gauss quadrature: 3-point triangle x 2-point line
    KOKKOS_INLINE_FUNCTION
    void gauss_quadrature(Real* pts, Real* wts) const {
        // 3-point triangle rule (order 2)
        const Real tri_xi[3]  = {1.0/6.0, 2.0/3.0, 1.0/6.0};
        const Real tri_eta[3] = {1.0/6.0, 1.0/6.0, 2.0/3.0};
        const Real tri_w = 1.0/6.0;  // weight for each (triangle area=0.5, each=1/6)
        const Real gz = 1.0/std::sqrt(3.0);
        const Real line_z[2] = {-gz, gz};
        int idx = 0;
        for (int t = 0; t < 3; ++t)
            for (int l = 0; l < 2; ++l) {
                pts[idx*3+0] = tri_xi[t];
                pts[idx*3+1] = tri_eta[t];
                pts[idx*3+2] = line_z[l];
                wts[idx] = tri_w * 1.0;  // line weight = 1
                idx++;
            }
    }

    KOKKOS_INLINE_FUNCTION
    void stiffness_matrix(Real* K) const {
        detail::zero(K, 18*18);
        Real C[36];
        detail::iso3D_C(E_, nu_, C);
        Real pts[18], wts[6];
        gauss_quadrature(pts, wts);
        for (int g = 0; g < NUM_GP; ++g) {
            Real B[6*18], J[9];
            strain_displacement_matrix(pts[g*3+0], pts[g*3+1], pts[g*3+2], B);
            Real detJ = jacobian(pts[g*3+0], pts[g*3+1], pts[g*3+2], J);
            Real w = wts[g] * std::abs(detJ);
            detail::addBtCB(K, B, C, w, 6, 18);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void compute_internal_forces(const Real* stress, Real* fint) const {
        detail::zero(fint, 18);
        Real pts[18], wts[6];
        gauss_quadrature(pts, wts);
        for (int g = 0; g < NUM_GP; ++g) {
            Real B[6*18], J[9];
            strain_displacement_matrix(pts[g*3+0], pts[g*3+1], pts[g*3+2], B);
            Real detJ = jacobian(pts[g*3+0], pts[g*3+1], pts[g*3+2], J);
            Real w = wts[g] * std::abs(detJ);
            for (int j = 0; j < 18; ++j) {
                Real val = 0.0;
                for (int i = 0; i < 6; ++i)
                    val += B[i*18+j] * stress[g*6+i];
                fint[j] += val * w;
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    Real volume() const {
        Real pts[18], wts[6];
        gauss_quadrature(pts, wts);
        Real vol = 0.0;
        for (int g = 0; g < NUM_GP; ++g) {
            Real J[9];
            vol += wts[g] * std::abs(jacobian(pts[g*3+0], pts[g*3+1], pts[g*3+2], J));
        }
        return vol;
    }

private:
    Real coords_[18];
    Real E_, nu_;
};

// ############################################################################
// 3. DKTShell  — Discrete Kirchhoff Triangle (bending-only, 9 DOF)
// ############################################################################

/**
 * 3 corner nodes, each with (w, theta_x, theta_y) = 3 DOF -> 9 DOF total.
 * Kirchhoff constraint: transverse shear = 0 enforced at discrete points.
 * Pure bending element (no membrane contribution).
 * Cubic interpolation along edges for normal slope.
 */
class DKTShell {
public:
    static constexpr int NUM_NODES = 3;
    static constexpr int DOF_PER_NODE = 3;  // w, theta_x, theta_y
    static constexpr int NUM_DOF = 9;
    static constexpr int NUM_GP = 3;

    KOKKOS_INLINE_FUNCTION
    DKTShell() : E_(0), nu_(0), thickness_(0.01) {
        detail::zero(coords_, 9);
    }

    KOKKOS_INLINE_FUNCTION
    DKTShell(const Real* node_coords_2d, Real E, Real nu, Real thickness)
        : E_(E), nu_(nu), thickness_(thickness) {
        // node_coords_2d: 3 nodes x 2 coords (x,y) flat
        for (int i = 0; i < 6; ++i) coords_[i] = node_coords_2d[i];
        // Compute edge lengths and directions
        precompute_edges();
    }

    KOKKOS_INLINE_FUNCTION int num_nodes() const { return NUM_NODES; }
    KOKKOS_INLINE_FUNCTION int dof_per_node() const { return DOF_PER_NODE; }
    KOKKOS_INLINE_FUNCTION int num_integration_points() const { return NUM_GP; }
    KOKKOS_INLINE_FUNCTION Real thickness() const { return thickness_; }

    /// Shape functions for the DKT element in area coordinates (L1,L2,L3)
    /// Returns 9 bending shape functions Hx(9), Hy(9) for curvatures
    /// Input: L1, L2 (L3 = 1 - L1 - L2)
    KOKKOS_INLINE_FUNCTION
    void shape_functions(Real L1, Real L2, Real* N) const {
        // For DKT, the "shape functions" relate the 9 DOFs to the deflection field.
        // We return the 3 standard triangular shape functions for the deflection w.
        Real L3 = 1.0 - L1 - L2;
        N[0] = L1; N[1] = L2; N[2] = L3;
    }

    /// Compute DKT bending B-matrix (3 x 9) that maps DOFs to curvatures
    /// curvatures = {kxx, kyy, 2*kxy}
    KOKKOS_INLINE_FUNCTION
    void bending_B_matrix(Real L1, Real L2, Real* Bb) const {
        detail::zero(Bb, 3*9);
        Real L3 = 1.0 - L1 - L2;

        // Edge vectors: x_ij = x_j - x_i, y_ij = y_j - y_i
        // Edges: 4(0-1), 5(1-2), 6(2-0)
        Real x12 = coords_[2] - coords_[0], y12 = coords_[3] - coords_[1];
        Real x23 = coords_[4] - coords_[2], y23 = coords_[5] - coords_[3];
        Real x31 = coords_[0] - coords_[4], y31 = coords_[1] - coords_[5];

        Real l12sq = x12*x12 + y12*y12;
        Real l23sq = x23*x23 + y23*y23;
        Real l31sq = x31*x31 + y31*y31;

        // Precomputed coefficients for DKT (Batoz, Bathe, Ho 1980)
        Real a4 = -x12/l12sq, b4 = 0.75*x12*y12/l12sq;
        Real c4 = (0.25*x12*x12 - 0.5*y12*y12)/l12sq;
        Real d4 = -y12/l12sq;
        Real e4 = (0.25*y12*y12 - 0.5*x12*x12)/l12sq;

        Real a5 = -x23/l23sq, b5 = 0.75*x23*y23/l23sq;
        Real c5 = (0.25*x23*x23 - 0.5*y23*y23)/l23sq;
        Real d5 = -y23/l23sq;
        Real e5 = (0.25*y23*y23 - 0.5*x23*x23)/l23sq;

        Real a6 = -x31/l31sq, b6 = 0.75*x31*y31/l31sq;
        Real c6 = (0.25*x31*x31 - 0.5*y31*y31)/l31sq;
        Real d6 = -y31/l31sq;
        Real e6 = (0.25*y31*y31 - 0.5*x31*x31)/l31sq;

        // Hx derivatives: d(beta_x)/dL1, d(beta_x)/dL2, d(beta_x)/dL3
        // beta_x = Hx . q,  beta_y = Hy . q
        // q = {w1, theta_x1, theta_y1, w2, theta_x2, theta_y2, w3, theta_x3, theta_y3}
        // Hx_i for node DOFs, differentiated w.r.t. area coordinates

        // dHx/dL1 components (9 entries)
        Real dHx1[9], dHx2[9], dHy1[9], dHy2[9];

        // Hx1 = dHx/dL1
        dHx1[0] = 6.0*(a6*L3 - a4*L2);          // P1
        dHx1[1] = -1.0 + 6.0*(b6*L3 - b4*L2);   // P2 (NOTE: combined with 1)
        dHx1[2] = -4.0*L1 + 6.0*(c6*L3 - c4*L2) + 2.0;  // P3
        dHx1[3] = 6.0*a4*L2;                      // P4
        dHx1[4] = 6.0*b4*L2;                      // P5
        dHx1[5] = 6.0*c4*L2 - 2.0*L2;            // P6
        dHx1[6] = -6.0*a6*L3;                     // P7
        dHx1[7] = 6.0*b6*L3;                      // P8 (NOTE: no negative)
        dHx1[8] = 6.0*c6*L3 - 2.0*L3;            // P9

        // Hx2 = dHx/dL2
        dHx2[0] = 6.0*(-a4*L1 + a5*L3);
        dHx2[1] = 6.0*(-b4*L1 + b5*L3);
        dHx2[2] = 6.0*(-c4*L1 + c5*L3) - 2.0*L1;
        dHx2[3] = 6.0*a4*L1 + 1.0;
        dHx2[3] = 6.0*(a4*L1 - a5*L3);
        dHx2[4] = 1.0 + 6.0*(b4*L1 - b5*L3);
        dHx2[5] = -4.0*L2 + 6.0*(c4*L1 - c5*L3) + 2.0;
        dHx2[6] = 6.0*a5*L3;
        dHx2[7] = 6.0*b5*L3;
        dHx2[8] = 6.0*c5*L3 - 2.0*L3;

        // Hy1 = dHy/dL1
        dHy1[0] = 6.0*(d6*L3 - d4*L2);
        dHy1[1] = -4.0*L1 + 6.0*(e6*L3 - e4*L2) + 2.0;
        dHy1[2] = -1.0 + 6.0*(-b6*L3 + b4*L2);
        dHy1[3] = 6.0*d4*L2;
        dHy1[4] = 6.0*e4*L2 - 2.0*L2;
        dHy1[5] = -6.0*b4*L2;
        dHy1[6] = -6.0*d6*L3;
        dHy1[7] = 6.0*e6*L3 - 2.0*L3;
        dHy1[8] = -6.0*b6*L3;

        // Hy2 = dHy/dL2
        dHy2[0] = 6.0*(-d4*L1 + d5*L3);
        dHy2[1] = 6.0*(-e4*L1 + e5*L3) - 2.0*L1;
        dHy2[2] = 6.0*(b4*L1 - b5*L3);
        dHy2[3] = 6.0*(d4*L1 - d5*L3);
        dHy2[4] = -4.0*L2 + 6.0*(e4*L1 - e5*L3) + 2.0;
        dHy2[5] = 1.0 + 6.0*(-b4*L1 + b5*L3);
        dHy2[6] = 6.0*d5*L3;
        dHy2[7] = 6.0*e5*L3 - 2.0*L3;
        dHy2[8] = -6.0*b5*L3;

        // Triangle Jacobian: J11=x12, J12=x31 (negative), etc.
        // For area coords: dx/dL1 = x1-x3, dx/dL2 = x2-x3
        Real J11 = coords_[0] - coords_[4];  // x1-x3
        Real J12 = coords_[2] - coords_[4];  // x2-x3
        Real J21 = coords_[1] - coords_[5];  // y1-y3
        Real J22 = coords_[3] - coords_[5];  // y2-y3
        Real detJ = J11*J22 - J12*J21;
        Real inv_detJ = 1.0 / detJ;

        // Transform from area coordinate derivatives to Cartesian:
        // d/dx = (J22 * d/dL1 - J21 * d/dL2) / detJ
        // d/dy = (-J12 * d/dL1 + J11 * d/dL2) / detJ

        for (int i = 0; i < 9; ++i) {
            Real dbx_dx = ( J22*dHx1[i] - J21*dHx2[i]) * inv_detJ;
            Real dby_dy = (-J12*dHy1[i] + J11*dHy2[i]) * inv_detJ;
            Real dbx_dy = (-J12*dHx1[i] + J11*dHx2[i]) * inv_detJ;
            Real dby_dx = ( J22*dHy1[i] - J21*dHy2[i]) * inv_detJ;

            Bb[0*9+i] = dbx_dx;              // kxx
            Bb[1*9+i] = dby_dy;              // kyy
            Bb[2*9+i] = dbx_dy + dby_dx;     // 2*kxy
        }
    }

    /// Bending stiffness matrix (9x9)
    KOKKOS_INLINE_FUNCTION
    void stiffness_matrix(Real* K) const {
        detail::zero(K, 9*9);
        // Db = (E*t^3)/(12*(1-nu^2)) * [1 nu 0; nu 1 0; 0 0 (1-nu)/2]
        Real Db[9];
        detail::zero(Db, 9);
        Real fac = E_*thickness_*thickness_*thickness_ / (12.0*(1.0-nu_*nu_));
        Db[0] = fac;       Db[1] = fac*nu_;
        Db[3] = fac*nu_;   Db[4] = fac;
        Db[8] = fac*(1.0-nu_)/2.0;

        // Triangle area
        Real x13 = coords_[0]-coords_[4], y13 = coords_[1]-coords_[5];
        Real x23 = coords_[2]-coords_[4], y23 = coords_[3]-coords_[5];
        Real area2 = std::abs(x13*y23 - x23*y13);  // 2*area
        Real area = 0.5*area2;

        // 3-point Gauss rule on triangle
        const Real gp[3][2] = {{1.0/6.0, 1.0/6.0},
                                {2.0/3.0, 1.0/6.0},
                                {1.0/6.0, 2.0/3.0}};
        const Real gw = 1.0/3.0;  // weight (sum=1, times area)

        for (int g = 0; g < 3; ++g) {
            Real Bb[27];  // 3x9
            bending_B_matrix(gp[g][0], gp[g][1], Bb);
            Real w = gw * area;
            detail::addBtCB(K, Bb, Db, w, 3, 9);
        }
    }

    /// Compute internal bending moments from curvatures
    KOKKOS_INLINE_FUNCTION
    void compute_internal_forces(const Real* moments, Real* fint) const {
        detail::zero(fint, 9);
        Real x13 = coords_[0]-coords_[4], y13 = coords_[1]-coords_[5];
        Real x23 = coords_[2]-coords_[4], y23 = coords_[3]-coords_[5];
        Real area = 0.5*std::abs(x13*y23 - x23*y13);
        const Real gp[3][2] = {{1.0/6.0, 1.0/6.0},
                                {2.0/3.0, 1.0/6.0},
                                {1.0/6.0, 2.0/3.0}};
        const Real gw = 1.0/3.0;
        for (int g = 0; g < 3; ++g) {
            Real Bb[27];
            bending_B_matrix(gp[g][0], gp[g][1], Bb);
            Real w = gw * area;
            for (int j = 0; j < 9; ++j) {
                Real val = 0.0;
                for (int i = 0; i < 3; ++i)
                    val += Bb[i*9+j] * moments[g*3+i];
                fint[j] += val * w;
            }
        }
    }

    /// Triangle area
    KOKKOS_INLINE_FUNCTION
    Real area() const {
        Real x13 = coords_[0]-coords_[4], y13 = coords_[1]-coords_[5];
        Real x23 = coords_[2]-coords_[4], y23 = coords_[3]-coords_[5];
        return 0.5*std::abs(x13*y23 - x23*y13);
    }

private:
    Real coords_[6];  // 3 nodes x 2 coords (x,y)
    Real E_, nu_, thickness_;

    KOKKOS_INLINE_FUNCTION
    void precompute_edges() {
        // edges are computed on the fly in bending_B_matrix
    }
};

// ############################################################################
// 4. DKQShell  — Discrete Kirchhoff Quadrilateral (bending-only, 12 DOF)
// ############################################################################

/**
 * 4 corner nodes, each with (w, theta_x, theta_y) = 3 DOF -> 12 DOF total.
 * Implemented as assembly of 4 DKT sub-triangles (cross-diagonal split)
 * using the simplified approach of Batoz & Tahar (1982).
 */
class DKQShell {
public:
    static constexpr int NUM_NODES = 4;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOF = 12;
    static constexpr int NUM_GP = 4;  // 2x2 Gauss

    KOKKOS_INLINE_FUNCTION
    DKQShell() : E_(0), nu_(0), thickness_(0.01) { detail::zero(coords_, 8); }

    KOKKOS_INLINE_FUNCTION
    DKQShell(const Real* node_coords_2d, Real E, Real nu, Real thickness)
        : E_(E), nu_(nu), thickness_(thickness) {
        for (int i = 0; i < 8; ++i) coords_[i] = node_coords_2d[i];
    }

    KOKKOS_INLINE_FUNCTION int num_nodes() const { return NUM_NODES; }
    KOKKOS_INLINE_FUNCTION int dof_per_node() const { return DOF_PER_NODE; }
    KOKKOS_INLINE_FUNCTION int num_integration_points() const { return NUM_GP; }
    KOKKOS_INLINE_FUNCTION Real thickness() const { return thickness_; }

    /// Bilinear shape functions at (xi, eta) in [-1,1]
    KOKKOS_INLINE_FUNCTION
    void shape_functions(Real xi, Real eta, Real* N) const {
        N[0] = 0.25*(1.0-xi)*(1.0-eta);
        N[1] = 0.25*(1.0+xi)*(1.0-eta);
        N[2] = 0.25*(1.0+xi)*(1.0+eta);
        N[3] = 0.25*(1.0-xi)*(1.0+eta);
    }

    /// Shape function derivatives dN/d(xi,eta) — 4x2 row-major
    KOKKOS_INLINE_FUNCTION
    void shape_derivatives_nat(Real xi, Real eta, Real* dN) const {
        dN[0*2+0] = -0.25*(1.0-eta); dN[0*2+1] = -0.25*(1.0-xi);
        dN[1*2+0] =  0.25*(1.0-eta); dN[1*2+1] = -0.25*(1.0+xi);
        dN[2*2+0] =  0.25*(1.0+eta); dN[2*2+1] =  0.25*(1.0+xi);
        dN[3*2+0] = -0.25*(1.0+eta); dN[3*2+1] =  0.25*(1.0-xi);
    }

    /// 2D Jacobian
    KOKKOS_INLINE_FUNCTION
    Real jacobian2d(Real xi, Real eta, Real* J) const {
        Real dN[8];
        shape_derivatives_nat(xi, eta, dN);
        detail::zero(J, 4);
        for (int a = 0; a < 4; ++a) {
            J[0] += dN[a*2+0] * coords_[a*2+0];
            J[1] += dN[a*2+0] * coords_[a*2+1];
            J[2] += dN[a*2+1] * coords_[a*2+0];
            J[3] += dN[a*2+1] * coords_[a*2+1];
        }
        return detail::det2(J);
    }

    /// Bending B-matrix (3 x 12) using a simplified DKQ approach
    /// Approximated using bilinear interpolation of curvatures
    KOKKOS_INLINE_FUNCTION
    void bending_B_matrix(Real xi, Real eta, Real* Bb) const {
        detail::zero(Bb, 3*12);
        Real dN[8];
        shape_derivatives_nat(xi, eta, dN);
        Real J[4], Ji[4];
        detail::zero(J, 4);
        for (int a = 0; a < 4; ++a) {
            J[0] += dN[a*2+0] * coords_[a*2+0];
            J[1] += dN[a*2+0] * coords_[a*2+1];
            J[2] += dN[a*2+1] * coords_[a*2+0];
            J[3] += dN[a*2+1] * coords_[a*2+1];
        }
        detail::inv2(J, Ji);

        // dN/dx, dN/dy in physical coords
        Real dNdx[4], dNdy[4];
        for (int a = 0; a < 4; ++a) {
            dNdx[a] = Ji[0]*dN[a*2+0] + Ji[1]*dN[a*2+1];
            dNdy[a] = Ji[2]*dN[a*2+0] + Ji[3]*dN[a*2+1];
        }

        // DKQ bending: for each node a, DOFs are (w_a, theta_xa, theta_ya)
        // curvatures:
        //   kxx = -d^2w/dx^2   -> approximated as d(theta_y)/dx
        //   kyy = -d^2w/dy^2   -> approximated as -d(theta_x)/dy
        //   kxy = -2*d^2w/dxdy -> approximated as d(theta_y)/dy - d(theta_x)/dx
        // Using Kirchhoff: theta_x = dw/dy, theta_y = -dw/dx
        // So: kxx = sum dNa/dx * (-theta_ya)  but we keep the sign as:
        //   Bb maps q = {w1,tx1,ty1, w2,tx2,ty2, ...} to {kxx, kyy, 2*kxy}

        for (int a = 0; a < 4; ++a) {
            int col = a*3;
            // kxx = d(beta_x)/dx where beta_x ~ -theta_y related
            // Using the DKQ simplification (Mindlin-type with Kirchhoff constraint):
            Bb[0*12 + col + 2] = -dNdx[a];      // kxx from theta_y
            Bb[1*12 + col + 1] =  dNdy[a];      // kyy from theta_x
            Bb[2*12 + col + 1] =  dNdx[a];      // 2*kxy from theta_x
            Bb[2*12 + col + 2] = -dNdy[a];      // 2*kxy from theta_y
        }
    }

    /// Bending stiffness matrix (12x12)
    KOKKOS_INLINE_FUNCTION
    void stiffness_matrix(Real* K) const {
        detail::zero(K, 12*12);
        Real Db[9];
        detail::zero(Db, 9);
        Real fac = E_*thickness_*thickness_*thickness_ / (12.0*(1.0 - nu_*nu_));
        Db[0] = fac;       Db[1] = fac*nu_;
        Db[3] = fac*nu_;   Db[4] = fac;
        Db[8] = fac*(1.0-nu_)/2.0;

        const Real g = 1.0/std::sqrt(3.0);
        const Real gp[4][2] = {{-g,-g},{g,-g},{g,g},{-g,g}};
        for (int q = 0; q < 4; ++q) {
            Real Bb[36]; // 3x12
            bending_B_matrix(gp[q][0], gp[q][1], Bb);
            Real J[4];
            Real detJ = jacobian2d(gp[q][0], gp[q][1], J);
            Real w = 1.0 * std::abs(detJ);
            detail::addBtCB(K, Bb, Db, w, 3, 12);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void compute_internal_forces(const Real* moments, Real* fint) const {
        detail::zero(fint, 12);
        const Real g = 1.0/std::sqrt(3.0);
        const Real gp[4][2] = {{-g,-g},{g,-g},{g,g},{-g,g}};
        for (int q = 0; q < 4; ++q) {
            Real Bb[36]; // 3x12
            bending_B_matrix(gp[q][0], gp[q][1], Bb);
            Real J[4];
            Real detJ = jacobian2d(gp[q][0], gp[q][1], J);
            Real w = 1.0 * std::abs(detJ);
            for (int j = 0; j < 12; ++j) {
                Real val = 0.0;
                for (int i = 0; i < 3; ++i)
                    val += Bb[i*12+j] * moments[q*3+i];
                fint[j] += val * w;
            }
        }
    }

    /// Quad area
    KOKKOS_INLINE_FUNCTION
    Real area() const {
        // Shoelace formula
        Real a = 0.0;
        for (int i = 0; i < 4; ++i) {
            int j = (i+1)%4;
            a += coords_[i*2]*coords_[j*2+1] - coords_[j*2]*coords_[i*2+1];
        }
        return 0.5*std::abs(a);
    }

private:
    Real coords_[8];  // 4 nodes x 2 coords
    Real E_, nu_, thickness_;
};

// ############################################################################
// 5. PlaneElement  — 2D plane-stress / plane-strain (Quad4 and Tri3)
// ############################################################################

enum class PlaneMode { PlaneStress, PlaneStrain };
enum class PlaneTopology { Quad4, Tri3 };

class PlaneElement {
public:
    static constexpr int DOF_PER_NODE = 2;
    static constexpr int NUM_STRESS = 3;  // sxx, syy, txy

    KOKKOS_INLINE_FUNCTION
    PlaneElement()
        : E_(0), nu_(0), thickness_(1.0),
          mode_(PlaneMode::PlaneStress), topo_(PlaneTopology::Quad4),
          nn_(4), ndof_(8), ngp_(4) {
        detail::zero(coords_, 8);
    }

    KOKKOS_INLINE_FUNCTION
    PlaneElement(const Real* node_coords_2d, int num_nodes, Real E, Real nu,
                 Real thickness, PlaneMode mode)
        : E_(E), nu_(nu), thickness_(thickness), mode_(mode) {
        if (num_nodes == 3) {
            topo_ = PlaneTopology::Tri3;
            nn_ = 3; ndof_ = 6; ngp_ = 1;
        } else {
            topo_ = PlaneTopology::Quad4;
            nn_ = 4; ndof_ = 8; ngp_ = 4;
        }
        for (int i = 0; i < nn_*2; ++i) coords_[i] = node_coords_2d[i];
    }

    KOKKOS_INLINE_FUNCTION int num_nodes() const { return nn_; }
    KOKKOS_INLINE_FUNCTION int dof_per_node() const { return DOF_PER_NODE; }
    KOKKOS_INLINE_FUNCTION int num_integration_points() const { return ngp_; }
    KOKKOS_INLINE_FUNCTION PlaneMode mode() const { return mode_; }
    KOKKOS_INLINE_FUNCTION PlaneTopology topology() const { return topo_; }

    /// Shape functions
    KOKKOS_INLINE_FUNCTION
    void shape_functions(Real xi, Real eta, Real* N) const {
        if (topo_ == PlaneTopology::Tri3) {
            N[0] = xi; N[1] = eta; N[2] = 1.0-xi-eta;
        } else {
            N[0] = 0.25*(1.0-xi)*(1.0-eta);
            N[1] = 0.25*(1.0+xi)*(1.0-eta);
            N[2] = 0.25*(1.0+xi)*(1.0+eta);
            N[3] = 0.25*(1.0-xi)*(1.0+eta);
        }
    }

    /// Shape function derivatives dN/d(xi,eta) — nn x 2, row-major
    KOKKOS_INLINE_FUNCTION
    void shape_derivatives_nat(Real xi, Real eta, Real* dN) const {
        if (topo_ == PlaneTopology::Tri3) {
            dN[0] = 1.0; dN[1] = 0.0;
            dN[2] = 0.0; dN[3] = 1.0;
            dN[4] = -1.0; dN[5] = -1.0;
        } else {
            dN[0] = -0.25*(1.0-eta); dN[1] = -0.25*(1.0-xi);
            dN[2] =  0.25*(1.0-eta); dN[3] = -0.25*(1.0+xi);
            dN[4] =  0.25*(1.0+eta); dN[5] =  0.25*(1.0+xi);
            dN[6] = -0.25*(1.0+eta); dN[7] =  0.25*(1.0-xi);
        }
    }

    /// 2D Jacobian
    KOKKOS_INLINE_FUNCTION
    Real jacobian2d(Real xi, Real eta, Real* J) const {
        Real dN[8];
        shape_derivatives_nat(xi, eta, dN);
        detail::zero(J, 4);
        for (int a = 0; a < nn_; ++a) {
            J[0] += dN[a*2+0]*coords_[a*2+0];
            J[1] += dN[a*2+0]*coords_[a*2+1];
            J[2] += dN[a*2+1]*coords_[a*2+0];
            J[3] += dN[a*2+1]*coords_[a*2+1];
        }
        return detail::det2(J);
    }

    /// B-matrix (3 x ndof) — strain = B * u
    KOKKOS_INLINE_FUNCTION
    void strain_displacement_matrix(Real xi, Real eta, Real* B) const {
        int nd = nn_*2;
        detail::zero(B, 3*nd);
        Real dN[8], J[4], Ji[4];
        shape_derivatives_nat(xi, eta, dN);
        detail::zero(J, 4);
        for (int a = 0; a < nn_; ++a) {
            J[0] += dN[a*2+0]*coords_[a*2+0];
            J[1] += dN[a*2+0]*coords_[a*2+1];
            J[2] += dN[a*2+1]*coords_[a*2+0];
            J[3] += dN[a*2+1]*coords_[a*2+1];
        }
        detail::inv2(J, Ji);
        Real dNdx[4], dNdy[4];
        for (int a = 0; a < nn_; ++a) {
            dNdx[a] = Ji[0]*dN[a*2+0] + Ji[1]*dN[a*2+1];
            dNdy[a] = Ji[2]*dN[a*2+0] + Ji[3]*dN[a*2+1];
        }
        for (int a = 0; a < nn_; ++a) {
            int c = a*2;
            B[0*nd+c+0] = dNdx[a];
            B[1*nd+c+1] = dNdy[a];
            B[2*nd+c+0] = dNdy[a];
            B[2*nd+c+1] = dNdx[a];
        }
    }

    /// Constitutive matrix (3x3)
    KOKKOS_INLINE_FUNCTION
    void constitutive_matrix(Real* C) const {
        if (mode_ == PlaneMode::PlaneStress)
            detail::planeStress_C(E_, nu_, C);
        else
            detail::planeStrain_C(E_, nu_, C);
    }

    /// Gauss quadrature
    KOKKOS_INLINE_FUNCTION
    void gauss_quadrature(Real* pts, Real* wts) const {
        if (topo_ == PlaneTopology::Tri3) {
            pts[0] = 1.0/3.0; pts[1] = 1.0/3.0;
            wts[0] = 0.5;  // triangle area weight
        } else {
            const Real g = 1.0/std::sqrt(3.0);
            pts[0] = -g; pts[1] = -g;
            pts[2] =  g; pts[3] = -g;
            pts[4] =  g; pts[5] =  g;
            pts[6] = -g; pts[7] =  g;
            for (int i = 0; i < 4; ++i) wts[i] = 1.0;
        }
    }

    /// Stiffness matrix (ndof x ndof)
    KOKKOS_INLINE_FUNCTION
    void stiffness_matrix(Real* K) const {
        int nd = nn_*2;
        for (int i = 0; i < nd*nd; ++i) K[i] = 0.0;
        Real C[9];
        constitutive_matrix(C);
        Real pts[8], wts[4];
        gauss_quadrature(pts, wts);
        for (int g = 0; g < ngp_; ++g) {
            Real B[3*8]; // max 3x8
            strain_displacement_matrix(pts[g*2+0], pts[g*2+1], B);
            Real J[4];
            Real detJ = jacobian2d(pts[g*2+0], pts[g*2+1], J);
            Real w = wts[g] * std::abs(detJ) * thickness_;
            detail::addBtCB(K, B, C, w, 3, nd);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void compute_internal_forces(const Real* stress, Real* fint) const {
        int nd = nn_*2;
        detail::zero(fint, nd);
        Real pts[8], wts[4];
        gauss_quadrature(pts, wts);
        for (int g = 0; g < ngp_; ++g) {
            Real B[3*8];
            strain_displacement_matrix(pts[g*2+0], pts[g*2+1], B);
            Real J[4];
            Real detJ = jacobian2d(pts[g*2+0], pts[g*2+1], J);
            Real w = wts[g] * std::abs(detJ) * thickness_;
            for (int j = 0; j < nd; ++j) {
                Real val = 0.0;
                for (int i = 0; i < 3; ++i)
                    val += B[i*nd+j] * stress[g*3+i];
                fint[j] += val * w;
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    Real area() const {
        if (topo_ == PlaneTopology::Tri3) {
            Real x13 = coords_[0]-coords_[4], y13 = coords_[1]-coords_[5];
            Real x23 = coords_[2]-coords_[4], y23 = coords_[3]-coords_[5];
            return 0.5*std::abs(x13*y23 - x23*y13);
        } else {
            Real a = 0.0;
            for (int i = 0; i < 4; ++i) {
                int j = (i+1)%4;
                a += coords_[i*2]*coords_[j*2+1] - coords_[j*2]*coords_[i*2+1];
            }
            return 0.5*std::abs(a);
        }
    }

private:
    Real coords_[8]; // max 4 nodes x 2 coords
    Real E_, nu_, thickness_;
    PlaneMode mode_;
    PlaneTopology topo_;
    int nn_, ndof_, ngp_;
};

// ############################################################################
// 6. AxisymmetricElement — Axisymmetric solid of revolution (Quad4 / Tri3)
// ############################################################################

/**
 * 2D mesh in (r, z) plane. Strain: [err, ezz, ett, 2*erz]
 * where ett = u_r / r (hoop strain).
 * Integration includes 2*pi*r factor.
 */

enum class AxisymTopology { Quad4, Tri3 };

class AxisymmetricElement {
public:
    static constexpr int DOF_PER_NODE = 2;  // u_r, u_z
    static constexpr int NUM_STRESS = 4;    // srr, szz, stt, trz

    KOKKOS_INLINE_FUNCTION
    AxisymmetricElement()
        : E_(0), nu_(0), topo_(AxisymTopology::Quad4), nn_(4), ndof_(8), ngp_(4) {
        detail::zero(coords_, 8);
    }

    KOKKOS_INLINE_FUNCTION
    AxisymmetricElement(const Real* node_coords_rz, int num_nodes, Real E, Real nu)
        : E_(E), nu_(nu) {
        if (num_nodes == 3) {
            topo_ = AxisymTopology::Tri3; nn_ = 3; ndof_ = 6; ngp_ = 1;
        } else {
            topo_ = AxisymTopology::Quad4; nn_ = 4; ndof_ = 8; ngp_ = 4;
        }
        for (int i = 0; i < nn_*2; ++i) coords_[i] = node_coords_rz[i];
    }

    KOKKOS_INLINE_FUNCTION int num_nodes() const { return nn_; }
    KOKKOS_INLINE_FUNCTION int dof_per_node() const { return DOF_PER_NODE; }
    KOKKOS_INLINE_FUNCTION int num_integration_points() const { return ngp_; }

    KOKKOS_INLINE_FUNCTION
    void shape_functions(Real xi, Real eta, Real* N) const {
        if (topo_ == AxisymTopology::Tri3) {
            N[0] = xi; N[1] = eta; N[2] = 1.0-xi-eta;
        } else {
            N[0] = 0.25*(1.0-xi)*(1.0-eta);
            N[1] = 0.25*(1.0+xi)*(1.0-eta);
            N[2] = 0.25*(1.0+xi)*(1.0+eta);
            N[3] = 0.25*(1.0-xi)*(1.0+eta);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void shape_derivatives_nat(Real xi, Real eta, Real* dN) const {
        if (topo_ == AxisymTopology::Tri3) {
            dN[0] = 1.0; dN[1] = 0.0;
            dN[2] = 0.0; dN[3] = 1.0;
            dN[4] = -1.0; dN[5] = -1.0;
        } else {
            dN[0] = -0.25*(1.0-eta); dN[1] = -0.25*(1.0-xi);
            dN[2] =  0.25*(1.0-eta); dN[3] = -0.25*(1.0+xi);
            dN[4] =  0.25*(1.0+eta); dN[5] =  0.25*(1.0+xi);
            dN[6] = -0.25*(1.0+eta); dN[7] =  0.25*(1.0-xi);
        }
    }

    KOKKOS_INLINE_FUNCTION
    Real jacobian2d(Real xi, Real eta, Real* J) const {
        Real dN[8];
        shape_derivatives_nat(xi, eta, dN);
        detail::zero(J, 4);
        for (int a = 0; a < nn_; ++a) {
            J[0] += dN[a*2+0]*coords_[a*2+0];
            J[1] += dN[a*2+0]*coords_[a*2+1];
            J[2] += dN[a*2+1]*coords_[a*2+0];
            J[3] += dN[a*2+1]*coords_[a*2+1];
        }
        return detail::det2(J);
    }

    /// Radius at parametric point
    KOKKOS_INLINE_FUNCTION
    Real radius_at(Real xi, Real eta) const {
        Real N[4];
        shape_functions(xi, eta, N);
        Real r = 0.0;
        for (int a = 0; a < nn_; ++a)
            r += N[a] * coords_[a*2+0];  // r-coordinate
        return r;
    }

    /// B-matrix (4 x ndof): [err, ezz, ett, 2*erz]
    KOKKOS_INLINE_FUNCTION
    void strain_displacement_matrix(Real xi, Real eta, Real* B) const {
        int nd = nn_*2;
        detail::zero(B, 4*nd);
        Real dN[8], N[4], J[4], Ji[4];
        shape_functions(xi, eta, N);
        shape_derivatives_nat(xi, eta, dN);
        detail::zero(J, 4);
        for (int a = 0; a < nn_; ++a) {
            J[0] += dN[a*2+0]*coords_[a*2+0];
            J[1] += dN[a*2+0]*coords_[a*2+1];
            J[2] += dN[a*2+1]*coords_[a*2+0];
            J[3] += dN[a*2+1]*coords_[a*2+1];
        }
        detail::inv2(J, Ji);
        Real dNdr[4], dNdz[4];
        for (int a = 0; a < nn_; ++a) {
            dNdr[a] = Ji[0]*dN[a*2+0] + Ji[1]*dN[a*2+1];
            dNdz[a] = Ji[2]*dN[a*2+0] + Ji[3]*dN[a*2+1];
        }
        Real r = 0.0;
        for (int a = 0; a < nn_; ++a) r += N[a]*coords_[a*2+0];
        Real inv_r = (r > 1e-30) ? 1.0/r : 0.0;

        for (int a = 0; a < nn_; ++a) {
            int c = a*2;
            B[0*nd+c+0] = dNdr[a];          // err = du_r/dr
            B[1*nd+c+1] = dNdz[a];          // ezz = du_z/dz
            B[2*nd+c+0] = N[a]*inv_r;       // ett = u_r/r
            B[3*nd+c+0] = dNdz[a];          // 2*erz = du_r/dz + du_z/dr
            B[3*nd+c+1] = dNdr[a];
        }
    }

    KOKKOS_INLINE_FUNCTION
    void constitutive_matrix(Real* C) const {
        detail::axisym_C(E_, nu_, C);
    }

    KOKKOS_INLINE_FUNCTION
    void gauss_quadrature(Real* pts, Real* wts) const {
        if (topo_ == AxisymTopology::Tri3) {
            pts[0] = 1.0/3.0; pts[1] = 1.0/3.0;
            wts[0] = 0.5;
        } else {
            const Real g = 1.0/std::sqrt(3.0);
            pts[0] = -g; pts[1] = -g;
            pts[2] =  g; pts[3] = -g;
            pts[4] =  g; pts[5] =  g;
            pts[6] = -g; pts[7] =  g;
            for (int i = 0; i < 4; ++i) wts[i] = 1.0;
        }
    }

    /// Stiffness matrix (ndof x ndof), includes 2*pi*r weighting
    KOKKOS_INLINE_FUNCTION
    void stiffness_matrix(Real* K) const {
        int nd = nn_*2;
        for (int i = 0; i < nd*nd; ++i) K[i] = 0.0;
        Real C[16];
        constitutive_matrix(C);
        Real pts[8], wts[4];
        gauss_quadrature(pts, wts);
        const Real two_pi = 2.0 * 3.14159265358979323846;
        for (int g = 0; g < ngp_; ++g) {
            Real B[4*8]; // max 4x8
            strain_displacement_matrix(pts[g*2+0], pts[g*2+1], B);
            Real J[4];
            Real detJ = jacobian2d(pts[g*2+0], pts[g*2+1], J);
            Real r = radius_at(pts[g*2+0], pts[g*2+1]);
            Real w = wts[g] * std::abs(detJ) * two_pi * r;
            detail::addBtCB(K, B, C, w, 4, nd);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void compute_internal_forces(const Real* stress, Real* fint) const {
        int nd = nn_*2;
        detail::zero(fint, nd);
        Real pts[8], wts[4];
        gauss_quadrature(pts, wts);
        const Real two_pi = 2.0 * 3.14159265358979323846;
        for (int g = 0; g < ngp_; ++g) {
            Real B[4*8];
            strain_displacement_matrix(pts[g*2+0], pts[g*2+1], B);
            Real J[4];
            Real detJ = jacobian2d(pts[g*2+0], pts[g*2+1], J);
            Real r = radius_at(pts[g*2+0], pts[g*2+1]);
            Real w = wts[g] * std::abs(detJ) * two_pi * r;
            for (int j = 0; j < nd; ++j) {
                Real val = 0.0;
                for (int i = 0; i < 4; ++i)
                    val += B[i*nd+j] * stress[g*4+i];
                fint[j] += val * w;
            }
        }
    }

private:
    Real coords_[8];
    Real E_, nu_;
    AxisymTopology topo_;
    int nn_, ndof_, ngp_;
};

// ############################################################################
// 7. ConnectorElement — Spot weld / rivet / fastener (2-node, 6 DOF/node)
// ############################################################################

/**
 * Two nodes connected by a beam-like element with force/moment resultants.
 * 6 DOF per node: (ux, uy, uz, theta_x, theta_y, theta_z) -> 12 DOF total.
 * Stiffness: translational K_t and rotational K_r (diagonal springs).
 * Failure: interaction criterion  (Fn/Fn_f)^a + (Fs/Fs_f)^b + (M/M_f)^c >= 1
 */

struct ConnectorFailureCriteria {
    Real normal_force_limit = 1e30;    // Fn_f
    Real shear_force_limit = 1e30;     // Fs_f
    Real moment_limit = 1e30;          // M_f
    Real normal_exponent = 2.0;        // a
    Real shear_exponent = 2.0;         // b
    Real moment_exponent = 2.0;        // c

    KOKKOS_INLINE_FUNCTION
    ConnectorFailureCriteria() = default;
};

class ConnectorElement {
public:
    static constexpr int NUM_NODES = 2;
    static constexpr int DOF_PER_NODE = 6;
    static constexpr int NUM_DOF = 12;

    KOKKOS_INLINE_FUNCTION
    ConnectorElement()
        : K_trans_{0,0,0}, K_rot_{0,0,0}, length_(0), failed_(false) {
        detail::zero(coords_, 6);
    }

    /**
     * @param node_coords  2 nodes x 3 coords (flat)
     * @param K_trans  translational stiffness (Kx, Ky, Kz) in local frame
     * @param K_rot    rotational stiffness (Krx, Kry, Krz) in local frame
     */
    KOKKOS_INLINE_FUNCTION
    ConnectorElement(const Real* node_coords, const Real K_trans[3],
                     const Real K_rot[3])
        : failed_(false) {
        for (int i = 0; i < 6; ++i) coords_[i] = node_coords[i];
        for (int i = 0; i < 3; ++i) { K_trans_[i] = K_trans[i]; K_rot_[i] = K_rot[i]; }
        Real dx = coords_[3]-coords_[0], dy = coords_[4]-coords_[1], dz = coords_[5]-coords_[2];
        length_ = std::sqrt(dx*dx + dy*dy + dz*dz);
        compute_local_axes();
    }

    KOKKOS_INLINE_FUNCTION int num_nodes() const { return NUM_NODES; }
    KOKKOS_INLINE_FUNCTION int dof_per_node() const { return DOF_PER_NODE; }
    KOKKOS_INLINE_FUNCTION Real length() const { return length_; }
    KOKKOS_INLINE_FUNCTION bool failed() const { return failed_; }

    void set_failure_criteria(const ConnectorFailureCriteria& fc) { fc_ = fc; }

    /// Local coordinate system: e1 along connector axis
    KOKKOS_INLINE_FUNCTION
    void local_axes(Real* e1, Real* e2, Real* e3) const {
        for (int i = 0; i < 3; ++i) {
            e1[i] = e1_[i]; e2[i] = e2_[i]; e3[i] = e3_[i];
        }
    }

    /// Shape functions (linear, 2-node)
    KOKKOS_INLINE_FUNCTION
    void shape_functions(Real xi, Real* N) const {
        // xi in [-1, 1]
        N[0] = 0.5*(1.0 - xi);
        N[1] = 0.5*(1.0 + xi);
    }

    /// Stiffness matrix (12x12) in global coordinates
    KOKKOS_INLINE_FUNCTION
    void stiffness_matrix(Real* K) const {
        detail::zero(K, 12*12);
        if (failed_) return;

        // Local stiffness: spring in each direction
        // K_local = [K_t -K_t; -K_t K_t] for translations
        //         + [K_r -K_r; -K_r K_r] for rotations
        // Then transform to global: K_global = T^T * K_local * T

        // Build local 12x12
        Real Kl[144];
        detail::zero(Kl, 144);
        for (int i = 0; i < 3; ++i) {
            Kl[i*12+i] = K_trans_[i];           // node1-node1
            Kl[i*12+(6+i)] = -K_trans_[i];      // node1-node2
            Kl[(6+i)*12+i] = -K_trans_[i];      // node2-node1
            Kl[(6+i)*12+(6+i)] = K_trans_[i];   // node2-node2
        }
        for (int i = 0; i < 3; ++i) {
            int ri = 3+i;
            Kl[ri*12+ri] = K_rot_[i];
            Kl[ri*12+(6+ri)] = -K_rot_[i];
            Kl[(6+ri)*12+ri] = -K_rot_[i];
            Kl[(6+ri)*12+(6+ri)] = K_rot_[i];
        }

        // Transform: T is block-diagonal with 4 copies of [e1;e2;e3] (3x3 rotation)
        Real T[144];
        detail::zero(T, 144);
        for (int b = 0; b < 4; ++b) {
            int off = b*3;
            for (int i = 0; i < 3; ++i) {
                T[(off+0)*12+(off+i)] = e1_[i];
                T[(off+1)*12+(off+i)] = e2_[i];
                T[(off+2)*12+(off+i)] = e3_[i];
            }
        }

        // K = T^T * Kl * T
        Real TKl[144]; // T^T * Kl
        detail::zero(TKl, 144);
        for (int i = 0; i < 12; ++i)
            for (int j = 0; j < 12; ++j)
                for (int k = 0; k < 12; ++k)
                    TKl[i*12+j] += T[k*12+i] * Kl[k*12+j];

        for (int i = 0; i < 12; ++i)
            for (int j = 0; j < 12; ++j)
                for (int k = 0; k < 12; ++k)
                    K[i*12+j] += TKl[i*12+k] * T[k*12+j];
    }

    /// Compute force/moment resultants from nodal displacements (12 DOF)
    /// Returns forces in local frame: Fx, Fy, Fz (node 2 - node 1)
    KOKKOS_INLINE_FUNCTION
    void compute_resultants(const Real* disp, Real* Fn, Real* Fs, Real* M) const {
        // Transform displacements to local frame
        Real du_global[3], dtheta_global[3];
        for (int i = 0; i < 3; ++i) {
            du_global[i] = disp[6+i] - disp[i];
            dtheta_global[i] = disp[6+3+i] - disp[3+i];
        }
        // Project onto local axes
        Real du_loc[3], dth_loc[3];
        du_loc[0] = du_global[0]*e1_[0] + du_global[1]*e1_[1] + du_global[2]*e1_[2];
        du_loc[1] = du_global[0]*e2_[0] + du_global[1]*e2_[1] + du_global[2]*e2_[2];
        du_loc[2] = du_global[0]*e3_[0] + du_global[1]*e3_[1] + du_global[2]*e3_[2];
        dth_loc[0] = dtheta_global[0]*e1_[0] + dtheta_global[1]*e1_[1] + dtheta_global[2]*e1_[2];
        dth_loc[1] = dtheta_global[0]*e2_[0] + dtheta_global[1]*e2_[1] + dtheta_global[2]*e2_[2];
        dth_loc[2] = dtheta_global[0]*e3_[0] + dtheta_global[1]*e3_[1] + dtheta_global[2]*e3_[2];

        // Normal force (along connector axis = e1)
        *Fn = K_trans_[0] * du_loc[0];
        // Shear force (perpendicular)
        Real Fs_y = K_trans_[1] * du_loc[1];
        Real Fs_z = K_trans_[2] * du_loc[2];
        *Fs = std::sqrt(Fs_y*Fs_y + Fs_z*Fs_z);
        // Moment
        Real M_x = K_rot_[0] * dth_loc[0];
        Real M_y = K_rot_[1] * dth_loc[1];
        Real M_z = K_rot_[2] * dth_loc[2];
        *M = std::sqrt(M_x*M_x + M_y*M_y + M_z*M_z);
    }

    /// Internal force vector (12 DOF)
    KOKKOS_INLINE_FUNCTION
    void compute_internal_forces(const Real* disp, Real* fint) const {
        detail::zero(fint, 12);
        if (failed_) return;

        Real K[144];
        stiffness_matrix(K);
        for (int i = 0; i < 12; ++i)
            for (int j = 0; j < 12; ++j)
                fint[i] += K[i*12+j] * disp[j];
    }

    /// Check failure criterion. Returns failure index (>=1 means failed).
    KOKKOS_INLINE_FUNCTION
    Real check_failure(const Real* disp) {
        Real Fn, Fs, M;
        compute_resultants(disp, &Fn, &Fs, &M);
        Real fn_ratio = std::abs(Fn) / fc_.normal_force_limit;
        Real fs_ratio = Fs / fc_.shear_force_limit;
        Real m_ratio = M / fc_.moment_limit;
        Real idx = std::pow(fn_ratio, fc_.normal_exponent)
                 + std::pow(fs_ratio, fc_.shear_exponent)
                 + std::pow(m_ratio, fc_.moment_exponent);
        if (idx >= 1.0) failed_ = true;
        return idx;
    }

    /// Reset failure state
    KOKKOS_INLINE_FUNCTION
    void reset_failure() { failed_ = false; }

private:
    Real coords_[6];     // 2 nodes x 3 coords
    Real K_trans_[3];    // translational stiffness (local)
    Real K_rot_[3];      // rotational stiffness (local)
    Real e1_[3], e2_[3], e3_[3];  // local axes
    Real length_;
    ConnectorFailureCriteria fc_;
    bool failed_;

    KOKKOS_INLINE_FUNCTION
    void compute_local_axes() {
        // e1 = axis direction
        e1_[0] = coords_[3]-coords_[0];
        e1_[1] = coords_[4]-coords_[1];
        e1_[2] = coords_[5]-coords_[2];
        detail::normalize3(e1_);
        // e2 perpendicular to e1 (use global Y or Z)
        Real ref[3] = {0.0, 1.0, 0.0};
        Real dp = e1_[0]*ref[0] + e1_[1]*ref[1] + e1_[2]*ref[2];
        if (std::abs(dp) > 0.9) { ref[0]=0; ref[1]=0; ref[2]=1; }
        detail::cross3(e1_, ref, e2_);
        detail::normalize3(e2_);
        detail::cross3(e1_, e2_, e3_);
        detail::normalize3(e3_);
    }
};

} // namespace elements
} // namespace nxs
