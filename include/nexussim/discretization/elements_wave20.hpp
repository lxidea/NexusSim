#pragma once

/**
 * @file elements_wave20.hpp
 * @brief Wave 20: Advanced Element Formulations
 *
 * Six element formulations for locking-free, high-performance analysis:
 *  1. BelytschkoTsayShell  - 1-point integration shell with hourglass control
 *  2. Pyramid5Element      - 5-node pyramid for hex-tet mesh transitions
 *  3. MITC4Shell           - Mixed Interpolation of Tensorial Components shell
 *  4. EASHex8              - Enhanced Assumed Strain 8-node hexahedron
 *  5. BBarHex8             - B-bar 8-node hexahedron (volumetric locking free)
 *  6. IsogeometricShell    - NURBS-based isogeometric shell element
 *
 * References:
 *  - Belytschko, Lin, Tsay (1984) CMAME
 *  - Flanagan, Belytschko (1981) hourglass control
 *  - Bathe, Dvorkin (1986) MITC4
 *  - Simo, Rifai (1990) EAS method
 *  - Hughes (1980) B-bar method
 *  - Kiendl et al. (2009) isogeometric shells
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

namespace detail {

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

} // namespace detail

// ############################################################################
// 1. BelytschkoTsayShell -- 1-point integration shell with hourglass control
// ############################################################################

/**
 * 4-node bilinear shell with single-point in-plane integration.
 * Velocity-strain formulation:
 *   - B-matrix evaluated at element center (xi=eta=0)
 *   - Hourglass control via Flanagan-Belytschko perturbation stiffness
 *   - Membrane + bending using Mindlin-Reissner theory
 *   - Shear correction factor kappa = 5/6
 *
 * DOF per node: 3 translations + 3 rotations = 6
 * Hourglass modes: gamma = h - N(0) . q (one per DOF direction)
 */
class BelytschkoTsayShell {
public:
    static constexpr int NUM_NODES = 4;
    static constexpr int DOF_PER_NODE = 6;
    static constexpr int NUM_DOF = 24;
    static constexpr Real SHEAR_FACTOR = 5.0 / 6.0;

    KOKKOS_INLINE_FUNCTION
    BelytschkoTsayShell() : E_(0), nu_(0), thickness_(0), density_(0),
                            hg_coeff_(0.1) {}

    KOKKOS_INLINE_FUNCTION
    BelytschkoTsayShell(Real E, Real nu, Real thickness, Real density,
                        Real hg_coefficient = 0.1)
        : E_(E), nu_(nu), thickness_(thickness), density_(density),
          hg_coeff_(hg_coefficient) {}

    /**
     * @brief Compute internal force vector from current configuration
     * @param coords  Node coordinates [4][3] (current configuration)
     * @param vel     Node velocities [4][3] (translational only)
     * @param dt      Time step size
     * @param stress_in   Input: previous stress state [5] (sxx,syy,sxy,sxz,syz)
     * @param stress_out  Output: updated stress [5]
     * @param force   Output: internal force [4][3] (translational contributions)
     */
    KOKKOS_INLINE_FUNCTION
    void compute_internal_force(const Real coords[4][3],
                                const Real vel[4][3],
                                Real dt,
                                const Real stress_in[5],
                                Real stress_out[5],
                                Real force[4][3]) const {
        // Zero output
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                force[a][i] = 0.0;

        // Compute local coordinate system from element geometry
        Real e1[3], e2[3], e3[3];
        compute_local_system(coords, e1, e2, e3);

        // Project node coordinates and velocities onto local frame
        Real lc[4][2], lv[4][2];
        for (int a = 0; a < 4; ++a) {
            Real dx = coords[a][0] - coords[0][0];
            Real dy = coords[a][1] - coords[0][1];
            Real dz = coords[a][2] - coords[0][2];
            lc[a][0] = dx*e1[0] + dy*e1[1] + dz*e1[2];
            lc[a][1] = dx*e2[0] + dy*e2[1] + dz*e2[2];

            lv[a][0] = vel[a][0]*e1[0] + vel[a][1]*e1[1] + vel[a][2]*e1[2];
            lv[a][1] = vel[a][0]*e2[0] + vel[a][1]*e2[1] + vel[a][2]*e2[2];
        }

        // B-matrix at center (xi=eta=0) for 4-node quad
        Real dNdxi[4]  = {-0.25, 0.25, 0.25, -0.25};
        Real dNdeta[4] = {-0.25, -0.25, 0.25, 0.25};

        // Jacobian at center
        Real J[4];
        J[0] = J[1] = J[2] = J[3] = 0.0;
        for (int a = 0; a < 4; ++a) {
            J[0] += dNdxi[a]  * lc[a][0];
            J[1] += dNdxi[a]  * lc[a][1];
            J[2] += dNdeta[a] * lc[a][0];
            J[3] += dNdeta[a] * lc[a][1];
        }
        Real detJ = J[0]*J[3] - J[1]*J[2];
        Real inv_detJ = 1.0 / detJ;

        // dN/dx, dN/dy at center
        Real dNdx[4], dNdy[4];
        for (int a = 0; a < 4; ++a) {
            dNdx[a] = ( J[3]*dNdxi[a] - J[2]*dNdeta[a]) * inv_detJ;
            dNdy[a] = (-J[1]*dNdxi[a] + J[0]*dNdeta[a]) * inv_detJ;
        }

        // Velocity strain rates at center
        Real eps_dot_xx = 0.0, eps_dot_yy = 0.0, eps_dot_xy = 0.0;
        for (int a = 0; a < 4; ++a) {
            eps_dot_xx += dNdx[a] * lv[a][0];
            eps_dot_yy += dNdy[a] * lv[a][1];
            eps_dot_xy += 0.5 * (dNdx[a] * lv[a][1] + dNdy[a] * lv[a][0]);
        }

        // Update stress using elastic constitutive relation (rate form)
        Real G = E_ / (2.0 * (1.0 + nu_));
        Real f = E_ / (1.0 - nu_ * nu_);

        stress_out[0] = stress_in[0] + (f * eps_dot_xx + f * nu_ * eps_dot_yy) * dt;
        stress_out[1] = stress_in[1] + (f * nu_ * eps_dot_xx + f * eps_dot_yy) * dt;
        stress_out[2] = stress_in[2] + (2.0 * G * eps_dot_xy) * dt;
        stress_out[3] = stress_in[3]; // transverse shear (sxz)
        stress_out[4] = stress_in[4]; // transverse shear (syz)

        // Internal force: f_a = B_a^T * sigma * A * t (single-point)
        Real area = std::abs(detJ) * 4.0; // integration over [-1,1]^2
        Real t = thickness_;

        for (int a = 0; a < 4; ++a) {
            Real fx_loc = (dNdx[a] * stress_out[0] + dNdy[a] * stress_out[2]) * area * t;
            Real fy_loc = (dNdy[a] * stress_out[1] + dNdx[a] * stress_out[2]) * area * t;

            // Transform back to global
            force[a][0] = fx_loc * e1[0] + fy_loc * e2[0];
            force[a][1] = fx_loc * e1[1] + fy_loc * e2[1];
            force[a][2] = fx_loc * e1[2] + fy_loc * e2[2];
        }
    }

    /**
     * @brief Compute hourglass force to suppress zero-energy modes
     * @param coords  Node coordinates [4][3]
     * @param vel     Node velocities [4][3]
     * @param hg_force Output: hourglass resistance force [4][3]
     */
    KOKKOS_INLINE_FUNCTION
    void hourglass_force(const Real coords[4][3],
                         const Real vel[4][3],
                         Real hg_force[4][3]) const {
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                hg_force[a][i] = 0.0;

        // Hourglass base vector for 4-node quad (Flanagan-Belytschko)
        const Real gamma[4] = {1.0, -1.0, 1.0, -1.0};

        // Compute element area using cross product of diagonals
        Real d1[3] = {coords[2][0]-coords[0][0], coords[2][1]-coords[0][1], coords[2][2]-coords[0][2]};
        Real d2[3] = {coords[3][0]-coords[1][0], coords[3][1]-coords[1][1], coords[3][2]-coords[1][2]};
        Real cr[3];
        detail::cross3(d1, d2, cr);
        Real area = 0.5 * detail::norm3(cr);
        if (area < 1.0e-30) return;

        // Hourglass velocity = gamma . v
        Real qhg[3] = {0.0, 0.0, 0.0};
        for (int a = 0; a < 4; ++a) {
            qhg[0] += gamma[a] * vel[a][0];
            qhg[1] += gamma[a] * vel[a][1];
            qhg[2] += gamma[a] * vel[a][2];
        }

        // Hourglass stiffness: Qhg = alpha * rho * c * A / (4 * L)
        Real G = E_ / (2.0 * (1.0 + nu_));
        Real bulk = E_ / (3.0 * (1.0 - 2.0 * nu_));
        Real c_sound = std::sqrt((bulk + 4.0/3.0 * G) / density_);
        Real char_len = std::sqrt(area);
        Real coeff = hg_coeff_ * density_ * c_sound * area / (4.0 * char_len);

        for (int a = 0; a < 4; ++a) {
            hg_force[a][0] = coeff * gamma[a] * qhg[0];
            hg_force[a][1] = coeff * gamma[a] * qhg[1];
            hg_force[a][2] = coeff * gamma[a] * qhg[2];
        }
    }

    /// Compute stable time step estimate
    KOKKOS_INLINE_FUNCTION
    Real stable_time_step(const Real coords[4][3]) const {
        Real d1[3] = {coords[2][0]-coords[0][0], coords[2][1]-coords[0][1], coords[2][2]-coords[0][2]};
        Real d2[3] = {coords[3][0]-coords[1][0], coords[3][1]-coords[1][1], coords[3][2]-coords[1][2]};
        Real cr[3];
        detail::cross3(d1, d2, cr);
        Real area = 0.5 * detail::norm3(cr);
        Real L = std::sqrt(area);

        Real G = E_ / (2.0 * (1.0 + nu_));
        Real bulk = E_ / (3.0 * (1.0 - 2.0 * nu_));
        Real c = std::sqrt((bulk + 4.0/3.0 * G) / density_);
        return (c > 1.0e-30) ? L / c : 1.0e30;
    }

    KOKKOS_INLINE_FUNCTION Real E() const { return E_; }
    KOKKOS_INLINE_FUNCTION Real nu() const { return nu_; }
    KOKKOS_INLINE_FUNCTION Real thickness() const { return thickness_; }

private:
    Real E_, nu_, thickness_, density_;
    Real hg_coeff_;

    KOKKOS_INLINE_FUNCTION
    void compute_local_system(const Real coords[4][3],
                              Real e1[3], Real e2[3], Real e3[3]) const {
        Real v1[3] = {coords[2][0]-coords[0][0], coords[2][1]-coords[0][1], coords[2][2]-coords[0][2]};
        Real v2[3] = {coords[3][0]-coords[1][0], coords[3][1]-coords[1][1], coords[3][2]-coords[1][2]};
        detail::cross3(v1, v2, e3);
        detail::normalize3(e3);
        for (int i = 0; i < 3; ++i) e1[i] = v1[i];
        detail::normalize3(e1);
        detail::cross3(e3, e1, e2);
        detail::normalize3(e2);
    }
};

// ############################################################################
// 2. Pyramid5Element -- 5-node pyramid for hex-tet transition meshes
// ############################################################################

/**
 * 5-node pyramid with quadrilateral base (nodes 0-3) and apex (node 4).
 * Natural coordinates: (xi, eta, zeta) in [-1,1]^2 x [0,1].
 *
 * Shape functions:
 *   N_0 = (1-xi)(1-eta)(1-zeta)/4
 *   N_1 = (1+xi)(1-eta)(1-zeta)/4
 *   N_2 = (1+xi)(1+eta)(1-zeta)/4
 *   N_3 = (1-xi)(1+eta)(1-zeta)/4
 *   N_4 = zeta (apex)
 */
class Pyramid5Element {
public:
    static constexpr int NUM_NODES = 5;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOF = 15;
    static constexpr int NUM_STRESS = 6;
    static constexpr int NUM_GP = 5;

    KOKKOS_INLINE_FUNCTION
    Pyramid5Element() : E_(0), nu_(0) {
        detail::zero(coords_, NUM_NODES * 3);
    }

    KOKKOS_INLINE_FUNCTION
    Pyramid5Element(const Real node_coords[5][3], Real E, Real nu)
        : E_(E), nu_(nu) {
        for (int a = 0; a < 5; ++a)
            for (int i = 0; i < 3; ++i)
                coords_[a*3+i] = node_coords[a][i];
    }

    KOKKOS_INLINE_FUNCTION int num_nodes() const { return NUM_NODES; }
    KOKKOS_INLINE_FUNCTION int num_dof() const { return NUM_DOF; }

    /// Shape functions
    KOKKOS_INLINE_FUNCTION
    void shape_functions(Real xi, Real eta, Real zeta, Real N[5]) const {
        Real oz = 1.0 - zeta;
        if (oz < 1.0e-12) oz = 1.0e-12;
        N[0] = 0.25 * (1.0 - xi) * (1.0 - eta) * oz;
        N[1] = 0.25 * (1.0 + xi) * (1.0 - eta) * oz;
        N[2] = 0.25 * (1.0 + xi) * (1.0 + eta) * oz;
        N[3] = 0.25 * (1.0 - xi) * (1.0 + eta) * oz;
        N[4] = zeta;
    }

    /// Shape function derivatives dN/d(xi,eta,zeta): 5 x 3, row-major
    KOKKOS_INLINE_FUNCTION
    void shape_derivatives(Real xi, Real eta, Real zeta, Real dN[15]) const {
        Real oz = 1.0 - zeta;
        if (oz < 1.0e-12) oz = 1.0e-12;

        dN[0*3+0] = -0.25 * (1.0 - eta) * oz;
        dN[1*3+0] =  0.25 * (1.0 - eta) * oz;
        dN[2*3+0] =  0.25 * (1.0 + eta) * oz;
        dN[3*3+0] = -0.25 * (1.0 + eta) * oz;
        dN[4*3+0] =  0.0;

        dN[0*3+1] = -0.25 * (1.0 - xi) * oz;
        dN[1*3+1] = -0.25 * (1.0 + xi) * oz;
        dN[2*3+1] =  0.25 * (1.0 + xi) * oz;
        dN[3*3+1] =  0.25 * (1.0 - xi) * oz;
        dN[4*3+1] =  0.0;

        dN[0*3+2] = -0.25 * (1.0 - xi) * (1.0 - eta);
        dN[1*3+2] = -0.25 * (1.0 + xi) * (1.0 - eta);
        dN[2*3+2] = -0.25 * (1.0 + xi) * (1.0 + eta);
        dN[3*3+2] = -0.25 * (1.0 - xi) * (1.0 + eta);
        dN[4*3+2] =  1.0;
    }

    /// Jacobian
    KOKKOS_INLINE_FUNCTION
    Real jacobian(Real xi, Real eta, Real zeta, Real J[9]) const {
        Real dN[15];
        shape_derivatives(xi, eta, zeta, dN);
        detail::zero(J, 9);
        for (int a = 0; a < 5; ++a)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    J[i*3+j] += dN[a*3+i] * coords_[a*3+j];
        return detail::det3(J);
    }

    /// B-matrix (6 x 15) strain-displacement
    KOKKOS_INLINE_FUNCTION
    void strain_displacement_matrix(Real xi, Real eta, Real zeta, Real B[90]) const {
        Real dN[15], J[9], Ji[9];
        shape_derivatives(xi, eta, zeta, dN);
        detail::zero(J, 9);
        for (int a = 0; a < 5; ++a)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    J[i*3+j] += dN[a*3+i] * coords_[a*3+j];
        detail::inv3(J, Ji);

        Real dNdx[15];
        for (int a = 0; a < 5; ++a)
            for (int j = 0; j < 3; ++j) {
                dNdx[a*3+j] = 0.0;
                for (int k = 0; k < 3; ++k)
                    dNdx[a*3+j] += Ji[k*3+j] * dN[a*3+k];
            }

        detail::zero(B, 6*15);
        for (int a = 0; a < 5; ++a) {
            int c = a*3;
            Real dx = dNdx[a*3+0], dy = dNdx[a*3+1], dz = dNdx[a*3+2];
            B[0*15+c+0] = dx;
            B[1*15+c+1] = dy;
            B[2*15+c+2] = dz;
            B[3*15+c+0] = dy;  B[3*15+c+1] = dx;
            B[4*15+c+1] = dz;  B[4*15+c+2] = dy;
            B[5*15+c+0] = dz;  B[5*15+c+2] = dx;
        }
    }

    /// 5-point quadrature for pyramid (Bedrosian-type)
    KOKKOS_INLINE_FUNCTION
    void gauss_quadrature(Real pts[15], Real wts[5]) const {
        Real a = 0.584237394;
        Real h = 0.1666666667;
        Real w_base = 81.0 / 100.0 * (1.0/3.0);
        Real w_apex = 125.0 / 27.0 * (1.0/3.0) * 0.064;

        pts[0*3+0] = -a; pts[0*3+1] = -a; pts[0*3+2] = h; wts[0] = w_base;
        pts[1*3+0] =  a; pts[1*3+1] = -a; pts[1*3+2] = h; wts[1] = w_base;
        pts[2*3+0] =  a; pts[2*3+1] =  a; pts[2*3+2] = h; wts[2] = w_base;
        pts[3*3+0] = -a; pts[3*3+1] =  a; pts[3*3+2] = h; wts[3] = w_base;
        pts[4*3+0] = 0.0; pts[4*3+1] = 0.0; pts[4*3+2] = 0.6; wts[4] = w_apex;
    }

    /// Element stiffness matrix (15 x 15)
    KOKKOS_INLINE_FUNCTION
    void compute_stiffness(Real K[225]) const {
        detail::zero(K, 15*15);
        Real C[36];
        detail::iso3D_C(E_, nu_, C);

        Real pts[15], wts[5];
        gauss_quadrature(pts, wts);

        for (int g = 0; g < NUM_GP; ++g) {
            Real B[90];
            strain_displacement_matrix(pts[g*3+0], pts[g*3+1], pts[g*3+2], B);
            Real J[9];
            Real detJ = jacobian(pts[g*3+0], pts[g*3+1], pts[g*3+2], J);
            Real w = wts[g] * std::abs(detJ);
            detail::addBtCB(K, B, C, w, 6, 15);
        }
    }

    /// Internal force from element stress [NUM_GP * 6]
    KOKKOS_INLINE_FUNCTION
    void compute_internal_forces(const Real stress[30], Real fint[15]) const {
        detail::zero(fint, 15);
        Real pts[15], wts[5];
        gauss_quadrature(pts, wts);
        for (int g = 0; g < NUM_GP; ++g) {
            Real B[90];
            strain_displacement_matrix(pts[g*3+0], pts[g*3+1], pts[g*3+2], B);
            Real J[9];
            Real detJ = jacobian(pts[g*3+0], pts[g*3+1], pts[g*3+2], J);
            Real w = wts[g] * std::abs(detJ);
            for (int j = 0; j < 15; ++j) {
                Real val = 0.0;
                for (int i = 0; i < 6; ++i)
                    val += B[i*15+j] * stress[g*6+i];
                fint[j] += val * w;
            }
        }
    }

    /// Volume
    KOKKOS_INLINE_FUNCTION
    Real volume() const {
        Real pts[15], wts[5];
        gauss_quadrature(pts, wts);
        Real vol = 0.0;
        for (int g = 0; g < NUM_GP; ++g) {
            Real J[9];
            vol += wts[g] * std::abs(jacobian(pts[g*3+0], pts[g*3+1], pts[g*3+2], J));
        }
        return vol;
    }

private:
    Real coords_[15];
    Real E_, nu_;
};

// ############################################################################
// 3. MITC4Shell -- Mixed Interpolation of Tensorial Components (locking-free)
// ############################################################################

/**
 * 4-node shell with assumed transverse shear strain field.
 * The MITC approach ties shear strain values at edge midpoints
 * to eliminate shear locking in thin-shell limits.
 *
 * DOF per node: (u, v, w, theta_x, theta_y) = 5, total = 20.
 * Membrane: bilinear interpolation.
 * Shear: assumed strain field with tying points at edge midpoints.
 *
 * Reference: Bathe, Dvorkin (1986) IJNME
 */
class MITC4Shell {
public:
    static constexpr int NUM_NODES = 4;
    static constexpr int DOF_PER_NODE = 5;
    static constexpr int NUM_DOF = 20;
    static constexpr int NUM_GP = 4;

    KOKKOS_INLINE_FUNCTION
    MITC4Shell() : E_(0), nu_(0), thickness_(0) {
        detail::zero(coords_, 12);
    }

    KOKKOS_INLINE_FUNCTION
    MITC4Shell(const Real node_coords[4][3], Real E, Real nu, Real thickness)
        : E_(E), nu_(nu), thickness_(thickness) {
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                coords_[a*3+i] = node_coords[a][i];
    }

    KOKKOS_INLINE_FUNCTION int num_nodes() const { return NUM_NODES; }
    KOKKOS_INLINE_FUNCTION int num_dof() const { return NUM_DOF; }
    KOKKOS_INLINE_FUNCTION Real thickness() const { return thickness_; }

    KOKKOS_INLINE_FUNCTION
    void shape_functions(Real xi, Real eta, Real N[4]) const {
        N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
        N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
        N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
        N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);
    }

    KOKKOS_INLINE_FUNCTION
    void shape_derivatives_nat(Real xi, Real eta, Real dN[8]) const {
        dN[0*2+0] = -0.25*(1.0-eta); dN[0*2+1] = -0.25*(1.0-xi);
        dN[1*2+0] =  0.25*(1.0-eta); dN[1*2+1] = -0.25*(1.0+xi);
        dN[2*2+0] =  0.25*(1.0+eta); dN[2*2+1] =  0.25*(1.0+xi);
        dN[3*2+0] = -0.25*(1.0+eta); dN[3*2+1] =  0.25*(1.0-xi);
    }

    /// Compute strain [exx, eyy, 2exy, kxx, kyy, 2kxy] at (xi, eta)
    KOKKOS_INLINE_FUNCTION
    void compute_strain(const Real disp[4][5], Real xi, Real eta,
                        Real strain[6]) const {
        detail::zero(strain, 6);

        Real e1[3], e2[3], e3[3];
        compute_local_system(e1, e2, e3);

        Real dN[8];
        shape_derivatives_nat(xi, eta, dN);

        Real lc[4][2];
        for (int a = 0; a < 4; ++a) {
            Real dx = coords_[a*3+0] - coords_[0];
            Real dy = coords_[a*3+1] - coords_[1];
            Real dz = coords_[a*3+2] - coords_[2];
            lc[a][0] = dx*e1[0] + dy*e1[1] + dz*e1[2];
            lc[a][1] = dx*e2[0] + dy*e2[1] + dz*e2[2];
        }

        Real J[4] = {0,0,0,0};
        for (int a = 0; a < 4; ++a) {
            J[0] += dN[a*2+0] * lc[a][0];
            J[1] += dN[a*2+0] * lc[a][1];
            J[2] += dN[a*2+1] * lc[a][0];
            J[3] += dN[a*2+1] * lc[a][1];
        }
        Real inv_detJ = 1.0 / (J[0]*J[3] - J[1]*J[2]);

        Real dNdx[4], dNdy[4];
        for (int a = 0; a < 4; ++a) {
            dNdx[a] = ( J[3]*dN[a*2+0] - J[2]*dN[a*2+1]) * inv_detJ;
            dNdy[a] = (-J[1]*dN[a*2+0] + J[0]*dN[a*2+1]) * inv_detJ;
        }

        Real u_loc[4][2], th[4][2];
        for (int a = 0; a < 4; ++a) {
            u_loc[a][0] = disp[a][0]*e1[0] + disp[a][1]*e1[1] + disp[a][2]*e1[2];
            u_loc[a][1] = disp[a][0]*e2[0] + disp[a][1]*e2[1] + disp[a][2]*e2[2];
            th[a][0] = disp[a][3];
            th[a][1] = disp[a][4];
        }

        // Membrane strains
        for (int a = 0; a < 4; ++a) {
            strain[0] += dNdx[a] * u_loc[a][0];
            strain[1] += dNdy[a] * u_loc[a][1];
            strain[2] += dNdx[a] * u_loc[a][1] + dNdy[a] * u_loc[a][0];
        }

        // Bending curvatures
        for (int a = 0; a < 4; ++a) {
            strain[3] += dNdx[a] * th[a][1];
            strain[4] -= dNdy[a] * th[a][0];
            strain[5] += dNdy[a] * th[a][1] - dNdx[a] * th[a][0];
        }
    }

    /// Compute MITC assumed transverse shear strains [gamma_xz, gamma_yz]
    KOKKOS_INLINE_FUNCTION
    void compute_mitc_shear(const Real disp[4][5], Real xi, Real eta,
                            Real gamma[2]) const {
        gamma[0] = gamma[1] = 0.0;

        Real e1[3], e2[3], e3[3];
        compute_local_system(e1, e2, e3);

        // Evaluate shear at 4 tying points
        Real gxz_A = eval_shear_at(disp, e1, e2, e3, 0.0, -1.0, 0);
        Real gxz_B = eval_shear_at(disp, e1, e2, e3, 0.0,  1.0, 0);
        Real gyz_C = eval_shear_at(disp, e1, e2, e3, -1.0, 0.0, 1);
        Real gyz_D = eval_shear_at(disp, e1, e2, e3,  1.0, 0.0, 1);

        // MITC interpolation
        gamma[0] = 0.5 * (1.0 - eta) * gxz_A + 0.5 * (1.0 + eta) * gxz_B;
        gamma[1] = 0.5 * (1.0 - xi)  * gyz_C + 0.5 * (1.0 + xi)  * gyz_D;
    }

    /// Internal force vector (20 DOFs)
    KOKKOS_INLINE_FUNCTION
    void compute_internal_force(const Real disp[4][5], Real fint[20]) const {
        detail::zero(fint, 20);

        Real Cm[9], Db[9];
        detail::planeStress_C(E_, nu_, Cm);
        detail::shellBending_D(E_, nu_, thickness_, Db);

        Real G = E_ / (2.0 * (1.0 + nu_));
        Real kappa = 5.0 / 6.0;
        Real shear_stiff = kappa * G * thickness_;

        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[4][2] = {{-g,-g},{g,-g},{g,g},{-g,g}};

        Real e1[3], e2[3], e3[3];
        compute_local_system(e1, e2, e3);

        for (int q = 0; q < 4; ++q) {
            Real xi = gp[q][0], eta_val = gp[q][1];

            Real eps[6];
            compute_strain(disp, xi, eta_val, eps);

            // Membrane resultants: N = Cm * eps_membrane * t
            Real Nm[3];
            for (int i = 0; i < 3; ++i) {
                Nm[i] = 0.0;
                for (int j = 0; j < 3; ++j)
                    Nm[i] += Cm[i*3+j] * eps[j];
                Nm[i] *= thickness_;
            }

            // Bending moments
            Real Mb[3];
            for (int i = 0; i < 3; ++i) {
                Mb[i] = 0.0;
                for (int j = 0; j < 3; ++j)
                    Mb[i] += Db[i*3+j] * eps[3+j];
            }

            // MITC shear
            Real gamma[2];
            compute_mitc_shear(disp, xi, eta_val, gamma);
            Real Qs[2] = {shear_stiff * gamma[0], shear_stiff * gamma[1]};

            // Jacobian for weight
            Real dN[8];
            shape_derivatives_nat(xi, eta_val, dN);
            Real lc[4][2];
            for (int a = 0; a < 4; ++a) {
                Real dx = coords_[a*3+0] - coords_[0];
                Real dy = coords_[a*3+1] - coords_[1];
                Real dz = coords_[a*3+2] - coords_[2];
                lc[a][0] = dx*e1[0] + dy*e1[1] + dz*e1[2];
                lc[a][1] = dx*e2[0] + dy*e2[1] + dz*e2[2];
            }
            Real J2[4] = {0,0,0,0};
            for (int a = 0; a < 4; ++a) {
                J2[0] += dN[a*2+0] * lc[a][0];
                J2[1] += dN[a*2+0] * lc[a][1];
                J2[2] += dN[a*2+1] * lc[a][0];
                J2[3] += dN[a*2+1] * lc[a][1];
            }
            Real detJ = J2[0]*J2[3] - J2[1]*J2[2];
            Real inv_detJ = 1.0 / detJ;
            Real dNdx[4], dNdy[4];
            for (int a = 0; a < 4; ++a) {
                dNdx[a] = ( J2[3]*dN[a*2+0] - J2[2]*dN[a*2+1]) * inv_detJ;
                dNdy[a] = (-J2[1]*dN[a*2+0] + J2[0]*dN[a*2+1]) * inv_detJ;
            }

            Real w = std::abs(detJ);  // Gauss weight = 1 for 2x2
            Real N_a[4];
            shape_functions(xi, eta_val, N_a);

            for (int a = 0; a < 4; ++a) {
                int base = a * 5;

                Real fx = (dNdx[a] * Nm[0] + dNdy[a] * Nm[2]) * w;
                Real fy = (dNdy[a] * Nm[1] + dNdx[a] * Nm[2]) * w;

                fint[base+0] += fx * e1[0] + fy * e2[0];
                fint[base+1] += fx * e1[1] + fy * e2[1];
                fint[base+2] += fx * e1[2] + fy * e2[2];

                // Bending
                fint[base+3] += (-dNdy[a] * Mb[1] - dNdx[a] * Mb[2]) * w;
                fint[base+4] += ( dNdx[a] * Mb[0] + dNdy[a] * Mb[2]) * w;

                // Shear
                fint[base+2] += (dNdx[a] * Qs[0] + dNdy[a] * Qs[1]) * w;
                fint[base+3] -= N_a[a] * Qs[1] * w;
                fint[base+4] += N_a[a] * Qs[0] * w;
            }
        }
    }

    /// Shell area
    KOKKOS_INLINE_FUNCTION
    Real area() const {
        Real d1[3] = {coords_[6]-coords_[0], coords_[7]-coords_[1], coords_[8]-coords_[2]};
        Real d2[3] = {coords_[9]-coords_[3], coords_[10]-coords_[4], coords_[11]-coords_[5]};
        Real cr[3];
        detail::cross3(d1, d2, cr);
        return 0.5 * detail::norm3(cr);
    }

private:
    Real coords_[12];
    Real E_, nu_, thickness_;

    KOKKOS_INLINE_FUNCTION
    void compute_local_system(Real e1[3], Real e2[3], Real e3[3]) const {
        Real v1[3] = {coords_[6]-coords_[0], coords_[7]-coords_[1], coords_[8]-coords_[2]};
        Real v2[3] = {coords_[9]-coords_[3], coords_[10]-coords_[4], coords_[11]-coords_[5]};
        detail::cross3(v1, v2, e3);
        detail::normalize3(e3);
        for (int i = 0; i < 3; ++i) e1[i] = v1[i];
        detail::normalize3(e1);
        detail::cross3(e3, e1, e2);
        detail::normalize3(e2);
    }

    /// Evaluate one transverse shear component at a tying point
    KOKKOS_INLINE_FUNCTION
    Real eval_shear_at(const Real disp[4][5],
                       const Real e1[3], const Real e2[3], const Real e3[3],
                       Real xi, Real eta, int component) const {
        Real dN[8];
        shape_derivatives_nat(xi, eta, dN);

        Real lc[4][2];
        for (int a = 0; a < 4; ++a) {
            Real dx = coords_[a*3+0] - coords_[0];
            Real dy = coords_[a*3+1] - coords_[1];
            Real dz = coords_[a*3+2] - coords_[2];
            lc[a][0] = dx*e1[0] + dy*e1[1] + dz*e1[2];
            lc[a][1] = dx*e2[0] + dy*e2[1] + dz*e2[2];
        }

        Real J[4] = {0,0,0,0};
        for (int a = 0; a < 4; ++a) {
            J[0] += dN[a*2+0] * lc[a][0];
            J[1] += dN[a*2+0] * lc[a][1];
            J[2] += dN[a*2+1] * lc[a][0];
            J[3] += dN[a*2+1] * lc[a][1];
        }
        Real inv_detJ = 1.0 / (J[0]*J[3] - J[1]*J[2]);

        Real dNdx[4], dNdy[4];
        for (int a = 0; a < 4; ++a) {
            dNdx[a] = ( J[3]*dN[a*2+0] - J[2]*dN[a*2+1]) * inv_detJ;
            dNdy[a] = (-J[1]*dN[a*2+0] + J[0]*dN[a*2+1]) * inv_detJ;
        }

        Real N[4];
        shape_functions(xi, eta, N);

        Real w_loc[4], th[4][2];
        for (int a = 0; a < 4; ++a) {
            w_loc[a] = disp[a][0]*e3[0] + disp[a][1]*e3[1] + disp[a][2]*e3[2];
            th[a][0] = disp[a][3];
            th[a][1] = disp[a][4];
        }

        if (component == 0) {
            Real dwdx = 0.0, th_y = 0.0;
            for (int a = 0; a < 4; ++a) {
                dwdx += dNdx[a] * w_loc[a];
                th_y += N[a] * th[a][1];
            }
            return dwdx + th_y;
        } else {
            Real dwdy = 0.0, th_x = 0.0;
            for (int a = 0; a < 4; ++a) {
                dwdy += dNdy[a] * w_loc[a];
                th_x += N[a] * th[a][0];
            }
            return dwdy - th_x;
        }
    }
};

// ############################################################################
// 4. EASHex8 -- Enhanced Assumed Strain 8-node hexahedron
// ############################################################################

/**
 * Standard 8-node hex with 7 EAS modes for volumetric/shear locking relief.
 * Strain = B*u + M*alpha. Internal alpha condensed via static condensation.
 *
 * Reference: Simo & Rifai (1990), Andelfinger & Ramm (1993)
 */
class EASHex8 {
public:
    static constexpr int NUM_NODES = 8;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOF = 24;
    static constexpr int NUM_STRESS = 6;
    static constexpr int NUM_GP = 8;
    static constexpr int NUM_EAS = 7;

    KOKKOS_INLINE_FUNCTION
    EASHex8() : E_(0), nu_(0) {
        detail::zero(coords_, 24);
        detail::zero(alpha_, NUM_EAS);
    }

    KOKKOS_INLINE_FUNCTION
    EASHex8(const Real node_coords[8][3], Real E, Real nu)
        : E_(E), nu_(nu) {
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                coords_[a*3+i] = node_coords[a][i];
        detail::zero(alpha_, NUM_EAS);
    }

    KOKKOS_INLINE_FUNCTION int num_nodes() const { return NUM_NODES; }
    KOKKOS_INLINE_FUNCTION int num_dof() const { return NUM_DOF; }

    KOKKOS_INLINE_FUNCTION
    void shape_derivatives(Real xi, Real eta, Real zeta, Real dN[24]) const {
        const Real s[8] = {-1,1,1,-1,-1,1,1,-1};
        const Real t[8] = {-1,-1,1,1,-1,-1,1,1};
        const Real u[8] = {-1,-1,-1,-1,1,1,1,1};
        for (int i = 0; i < 8; ++i) {
            dN[i*3+0] = 0.125*s[i]*(1.0+t[i]*eta)*(1.0+u[i]*zeta);
            dN[i*3+1] = 0.125*t[i]*(1.0+s[i]*xi)*(1.0+u[i]*zeta);
            dN[i*3+2] = 0.125*u[i]*(1.0+s[i]*xi)*(1.0+t[i]*eta);
        }
    }

    KOKKOS_INLINE_FUNCTION
    Real jacobian(Real xi, Real eta, Real zeta, Real J[9]) const {
        Real dN[24];
        shape_derivatives(xi, eta, zeta, dN);
        detail::zero(J, 9);
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    J[i*3+j] += dN[a*3+i] * coords_[a*3+j];
        return detail::det3(J);
    }

    /// Standard B-matrix (6 x 24)
    KOKKOS_INLINE_FUNCTION
    void strain_displacement_matrix(Real xi, Real eta, Real zeta, Real B[144]) const {
        Real dN[24], J[9], Ji[9];
        shape_derivatives(xi, eta, zeta, dN);
        detail::zero(J, 9);
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    J[i*3+j] += dN[a*3+i] * coords_[a*3+j];
        detail::inv3(J, Ji);

        Real dNdx[24];
        for (int a = 0; a < 8; ++a)
            for (int j = 0; j < 3; ++j) {
                dNdx[a*3+j] = 0.0;
                for (int k = 0; k < 3; ++k)
                    dNdx[a*3+j] += Ji[k*3+j] * dN[a*3+k];
            }

        detail::zero(B, 6*24);
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

    /// EAS interpolation matrix M (6 x 7)
    KOKKOS_INLINE_FUNCTION
    void eas_matrix(Real xi, Real eta, Real zeta,
                    Real detJ0, Real detJ, Real M[42]) const {
        detail::zero(M, 6*7);
        Real ratio = detJ0 / detJ;

        M[0*7+0] = ratio * xi;
        M[1*7+1] = ratio * eta;
        M[2*7+2] = ratio * zeta;
        M[3*7+3] = ratio * xi;
        M[4*7+4] = ratio * eta;
        M[5*7+5] = ratio * zeta;
        M[3*7+6] = ratio * xi * eta;
    }

    /// Stiffness with EAS condensation: K = Kuu - Kua * Kaa^{-1} * Kua^T
    KOKKOS_INLINE_FUNCTION
    void compute_stiffness(Real K[576]) const {
        detail::zero(K, 24*24);
        Real C[36];
        detail::iso3D_C(E_, nu_, C);

        Real J0[9];
        Real detJ0 = jacobian(0.0, 0.0, 0.0, J0);

        Real Kuu[576], Kua[168], Kaa[49];
        detail::zero(Kuu, 576);
        detail::zero(Kua, 168);
        detail::zero(Kaa, 49);

        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};

        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real xi = gp[gi], eta = gp[gj], zeta = gp[gk];

            Real B[144];
            strain_displacement_matrix(xi, eta, zeta, B);

            Real J[9];
            Real detJ = jacobian(xi, eta, zeta, J);
            Real w = std::abs(detJ);

            Real M[42];
            eas_matrix(xi, eta, zeta, detJ0, detJ, M);

            detail::addBtCB(Kuu, B, C, w, 6, 24);

            // Kua += B^T * C * M * w
            for (int i = 0; i < 24; ++i)
                for (int j = 0; j < 7; ++j) {
                    Real val = 0.0;
                    for (int k = 0; k < 6; ++k) {
                        Real cm = 0.0;
                        for (int l = 0; l < 6; ++l)
                            cm += C[k*6+l] * M[l*7+j];
                        val += B[k*24+i] * cm;
                    }
                    Kua[i*7+j] += val * w;
                }

            detail::addBtCB(Kaa, M, C, w, 6, 7);
        }

        // Invert Kaa (7x7)
        Real Kaa_inv[49];
        invert_7x7(Kaa, Kaa_inv);

        // K = Kuu - Kua * Kaa_inv * Kua^T
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 24; ++j) {
                Real correction = 0.0;
                for (int p = 0; p < 7; ++p) {
                    Real kua_inv_p = 0.0;
                    for (int q = 0; q < 7; ++q)
                        kua_inv_p += Kua[i*7+q] * Kaa_inv[q*7+p];
                    correction += kua_inv_p * Kua[j*7+p];
                }
                K[i*24+j] = Kuu[i*24+j] - correction;
            }
    }

    /// Update EAS internal parameters: alpha = -Kaa^{-1} * Kua^T * u
    KOKKOS_INLINE_FUNCTION
    void update_eas_params(const Real disp[24]) {
        Real C[36];
        detail::iso3D_C(E_, nu_, C);

        Real J0[9];
        Real detJ0 = jacobian(0.0, 0.0, 0.0, J0);

        Real Kua[168], Kaa[49];
        detail::zero(Kua, 168);
        detail::zero(Kaa, 49);

        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};

        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real xi = gp[gi], eta = gp[gj], zeta = gp[gk];

            Real B[144];
            strain_displacement_matrix(xi, eta, zeta, B);
            Real J[9];
            Real detJ = jacobian(xi, eta, zeta, J);
            Real w = std::abs(detJ);

            Real M[42];
            eas_matrix(xi, eta, zeta, detJ0, detJ, M);

            for (int i = 0; i < 24; ++i)
                for (int j = 0; j < 7; ++j) {
                    Real val = 0.0;
                    for (int k = 0; k < 6; ++k) {
                        Real cm = 0.0;
                        for (int l = 0; l < 6; ++l)
                            cm += C[k*6+l] * M[l*7+j];
                        val += B[k*24+i] * cm;
                    }
                    Kua[i*7+j] += val * w;
                }

            detail::addBtCB(Kaa, M, C, w, 6, 7);
        }

        Real Kaa_inv[49];
        invert_7x7(Kaa, Kaa_inv);

        Real rhs[7];
        detail::zero(rhs, 7);
        for (int j = 0; j < 7; ++j)
            for (int i = 0; i < 24; ++i)
                rhs[j] += Kua[i*7+j] * disp[i];

        for (int i = 0; i < 7; ++i) {
            alpha_[i] = 0.0;
            for (int j = 0; j < 7; ++j)
                alpha_[i] -= Kaa_inv[i*7+j] * rhs[j];
        }
    }

    KOKKOS_INLINE_FUNCTION const Real* eas_params() const { return alpha_; }

    KOKKOS_INLINE_FUNCTION
    Real volume() const {
        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};
        Real vol = 0.0;
        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real J[9];
            vol += std::abs(jacobian(gp[gi], gp[gj], gp[gk], J));
        }
        return vol;
    }

private:
    Real coords_[24];
    Real E_, nu_;
    Real alpha_[NUM_EAS];

    /// Gauss-Jordan 7x7 inverse
    KOKKOS_INLINE_FUNCTION
    static void invert_7x7(const Real A[49], Real Ainv[49]) {
        Real aug[7][14];
        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j < 7; ++j) {
                aug[i][j] = A[i*7+j];
                aug[i][j+7] = (i == j) ? 1.0 : 0.0;
            }
        }

        for (int col = 0; col < 7; ++col) {
            int pivot = col;
            Real max_val = std::abs(aug[col][col]);
            for (int row = col+1; row < 7; ++row) {
                if (std::abs(aug[row][col]) > max_val) {
                    max_val = std::abs(aug[row][col]);
                    pivot = row;
                }
            }
            if (pivot != col) {
                for (int j = 0; j < 14; ++j) {
                    Real tmp = aug[col][j];
                    aug[col][j] = aug[pivot][j];
                    aug[pivot][j] = tmp;
                }
            }

            Real diag = aug[col][col];
            if (std::abs(diag) < 1.0e-30) diag = 1.0e-30;
            Real inv_diag = 1.0 / diag;

            for (int j = 0; j < 14; ++j)
                aug[col][j] *= inv_diag;

            for (int row = 0; row < 7; ++row) {
                if (row == col) continue;
                Real factor = aug[row][col];
                for (int j = 0; j < 14; ++j)
                    aug[row][j] -= factor * aug[col][j];
            }
        }

        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                Ainv[i*7+j] = aug[i][j+7];
    }
};

// ############################################################################
// 5. BBarHex8 -- B-bar 8-node hexahedron (volumetric locking free)
// ############################################################################

/**
 * Hughes' B-bar: B_bar = B_dev(xi) + B_vol(0,0,0).
 * Volumetric part evaluated at element center, preventing locking.
 *
 * Reference: Hughes (1980)
 */
class BBarHex8 {
public:
    static constexpr int NUM_NODES = 8;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOF = 24;
    static constexpr int NUM_STRESS = 6;
    static constexpr int NUM_GP = 8;

    KOKKOS_INLINE_FUNCTION
    BBarHex8() : E_(0), nu_(0) {
        detail::zero(coords_, 24);
    }

    KOKKOS_INLINE_FUNCTION
    BBarHex8(const Real node_coords[8][3], Real E, Real nu)
        : E_(E), nu_(nu) {
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                coords_[a*3+i] = node_coords[a][i];
    }

    KOKKOS_INLINE_FUNCTION int num_nodes() const { return NUM_NODES; }
    KOKKOS_INLINE_FUNCTION int num_dof() const { return NUM_DOF; }

    KOKKOS_INLINE_FUNCTION
    void shape_derivatives(Real xi, Real eta, Real zeta, Real dN[24]) const {
        const Real s[8] = {-1,1,1,-1,-1,1,1,-1};
        const Real t[8] = {-1,-1,1,1,-1,-1,1,1};
        const Real u[8] = {-1,-1,-1,-1,1,1,1,1};
        for (int i = 0; i < 8; ++i) {
            dN[i*3+0] = 0.125*s[i]*(1.0+t[i]*eta)*(1.0+u[i]*zeta);
            dN[i*3+1] = 0.125*t[i]*(1.0+s[i]*xi)*(1.0+u[i]*zeta);
            dN[i*3+2] = 0.125*u[i]*(1.0+s[i]*xi)*(1.0+t[i]*eta);
        }
    }

    KOKKOS_INLINE_FUNCTION
    Real jacobian(Real xi, Real eta, Real zeta, Real J[9]) const {
        Real dN[24];
        shape_derivatives(xi, eta, zeta, dN);
        detail::zero(J, 9);
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    J[i*3+j] += dN[a*3+i] * coords_[a*3+j];
        return detail::det3(J);
    }

    /// Physical shape function derivatives at a point
    KOKKOS_INLINE_FUNCTION
    void physical_derivatives(Real xi, Real eta, Real zeta, Real dNdx[24]) const {
        Real dN[24], J[9], Ji[9];
        shape_derivatives(xi, eta, zeta, dN);
        detail::zero(J, 9);
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    J[i*3+j] += dN[a*3+i] * coords_[a*3+j];
        detail::inv3(J, Ji);

        for (int a = 0; a < 8; ++a)
            for (int j = 0; j < 3; ++j) {
                dNdx[a*3+j] = 0.0;
                for (int k = 0; k < 3; ++k)
                    dNdx[a*3+j] += Ji[k*3+j] * dN[a*3+k];
            }
    }

    /// B-bar matrix (6 x 24): B_dev(gp) + B_vol(center)
    KOKKOS_INLINE_FUNCTION
    void compute_bbar(Real xi, Real eta, Real zeta, Real B[144]) const {
        Real dNdx_gp[24];
        physical_derivatives(xi, eta, zeta, dNdx_gp);

        Real dNdx_c[24];
        physical_derivatives(0.0, 0.0, 0.0, dNdx_c);

        detail::zero(B, 6*24);

        for (int a = 0; a < 8; ++a) {
            int c = a * 3;

            Real bx = dNdx_gp[a*3+0];
            Real by = dNdx_gp[a*3+1];
            Real bz = dNdx_gp[a*3+2];
            Real div_gp = (bx + by + bz) / 3.0;

            Real bx_c = dNdx_c[a*3+0];
            Real by_c = dNdx_c[a*3+1];
            Real bz_c = dNdx_c[a*3+2];
            Real div_c = (bx_c + by_c + bz_c) / 3.0;

            Real vol_corr = -div_gp + div_c;

            // Normal strains: B_dev(gp) + B_vol(center)
            B[0*24+c+0] = bx + vol_corr;
            B[1*24+c+0] = vol_corr;
            B[2*24+c+0] = vol_corr;

            B[0*24+c+1] = vol_corr;
            B[1*24+c+1] = by + vol_corr;
            B[2*24+c+1] = vol_corr;

            B[0*24+c+2] = vol_corr;
            B[1*24+c+2] = vol_corr;
            B[2*24+c+2] = bz + vol_corr;

            // Shear strains (purely deviatoric, unchanged)
            B[3*24+c+0] = by;  B[3*24+c+1] = bx;
            B[4*24+c+1] = bz;  B[4*24+c+2] = by;
            B[5*24+c+0] = bz;  B[5*24+c+2] = bx;
        }
    }

    /// Stiffness matrix (24x24)
    KOKKOS_INLINE_FUNCTION
    void compute_stiffness(Real K[576]) const {
        detail::zero(K, 24*24);
        Real C[36];
        detail::iso3D_C(E_, nu_, C);

        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};

        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real B[144];
            compute_bbar(gp[gi], gp[gj], gp[gk], B);
            Real J[9];
            Real detJ = jacobian(gp[gi], gp[gj], gp[gk], J);
            Real w = std::abs(detJ);
            detail::addBtCB(K, B, C, w, 6, 24);
        }
    }

    /// Internal force using B-bar
    KOKKOS_INLINE_FUNCTION
    void compute_internal_force(const Real disp[24], Real fint[24]) const {
        detail::zero(fint, 24);
        Real C[36];
        detail::iso3D_C(E_, nu_, C);

        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};

        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real B[144];
            compute_bbar(gp[gi], gp[gj], gp[gk], B);
            Real J[9];
            Real detJ = jacobian(gp[gi], gp[gj], gp[gk], J);
            Real w = std::abs(detJ);

            Real eps[6];
            detail::zero(eps, 6);
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 24; ++j)
                    eps[i] += B[i*24+j] * disp[j];

            Real sig[6];
            detail::zero(sig, 6);
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
        const Real g = 1.0 / std::sqrt(3.0);
        const Real gp[2] = {-g, g};
        Real vol = 0.0;
        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real J[9];
            vol += std::abs(jacobian(gp[gi], gp[gj], gp[gk], J));
        }
        return vol;
    }

private:
    Real coords_[24];
    Real E_, nu_;
};

// ############################################################################
// 6. IsogeometricShell -- NURBS-based isogeometric analysis shell element
// ############################################################################

/**
 * NURBS-based Kirchhoff-Love shell for isogeometric analysis.
 * Control points with weights, B-spline/NURBS basis functions.
 * Supports up to degree 3, up to 4x4 = 16 control points per patch.
 *
 * Reference: Kiendl et al. (2009)
 */
class IsogeometricShell {
public:
    static constexpr int MAX_CP_PER_DIR = 4;
    static constexpr int MAX_CP = MAX_CP_PER_DIR * MAX_CP_PER_DIR;
    static constexpr int MAX_KNOTS = 10;
    static constexpr int DOF_PER_CP = 3;

    struct NURBSPatch {
        int n_u, n_v;
        int p_u, p_v;
        Real knots_u[MAX_KNOTS];
        Real knots_v[MAX_KNOTS];
        Real control_points[MAX_CP][3];
        Real weights[MAX_CP];
        int num_knots_u, num_knots_v;

        KOKKOS_INLINE_FUNCTION
        NURBSPatch() : n_u(0), n_v(0), p_u(0), p_v(0),
                       num_knots_u(0), num_knots_v(0) {
            for (int i = 0; i < MAX_KNOTS; ++i) { knots_u[i] = 0; knots_v[i] = 0; }
            for (int i = 0; i < MAX_CP; ++i) {
                control_points[i][0] = control_points[i][1] = control_points[i][2] = 0;
                weights[i] = 1.0;
            }
        }

        KOKKOS_INLINE_FUNCTION
        int total_cp() const { return n_u * n_v; }
    };

    KOKKOS_INLINE_FUNCTION
    IsogeometricShell() : E_(0), nu_(0), thickness_(0) {}

    KOKKOS_INLINE_FUNCTION
    IsogeometricShell(const NURBSPatch& patch, Real E, Real nu, Real thickness)
        : patch_(patch), E_(E), nu_(nu), thickness_(thickness) {}

    KOKKOS_INLINE_FUNCTION int num_control_points() const { return patch_.total_cp(); }
    KOKKOS_INLINE_FUNCTION Real thickness() const { return thickness_; }

    /// NURBS basis functions and derivatives at (xi, eta)
    KOKKOS_INLINE_FUNCTION
    void compute_basis(Real xi, Real eta,
                       Real R[MAX_CP], Real dRdxi[MAX_CP], Real dRdeta[MAX_CP]) const {
        int nu = patch_.n_u, nv = patch_.n_v;
        int pu = patch_.p_u, pv = patch_.p_v;

        Real Nu[MAX_CP_PER_DIR], dNu[MAX_CP_PER_DIR];
        Real Nv[MAX_CP_PER_DIR], dNv[MAX_CP_PER_DIR];

        bspline_basis_and_deriv(xi, pu, patch_.knots_u, patch_.num_knots_u, nu, Nu, dNu);
        bspline_basis_and_deriv(eta, pv, patch_.knots_v, patch_.num_knots_v, nv, Nv, dNv);

        Real w_sum = 0.0, dw_dxi = 0.0, dw_deta = 0.0;

        for (int j = 0; j < nv; ++j)
            for (int i = 0; i < nu; ++i) {
                int idx = j * nu + i;
                Real w = patch_.weights[idx];
                Real Nij = Nu[i] * Nv[j];
                w_sum += Nij * w;
                dw_dxi += dNu[i] * Nv[j] * w;
                dw_deta += Nu[i] * dNv[j] * w;
            }

        if (w_sum < 1.0e-30) w_sum = 1.0e-30;
        Real inv_w = 1.0 / w_sum;

        for (int j = 0; j < nv; ++j)
            for (int i = 0; i < nu; ++i) {
                int idx = j * nu + i;
                Real w = patch_.weights[idx];
                Real Nij = Nu[i] * Nv[j];
                Real dNij_dxi = dNu[i] * Nv[j];
                Real dNij_deta = Nu[i] * dNv[j];

                R[idx] = Nij * w * inv_w;
                dRdxi[idx] = (dNij_dxi * w - Nij * w * dw_dxi * inv_w) * inv_w;
                dRdeta[idx] = (dNij_deta * w - Nij * w * dw_deta * inv_w) * inv_w;
            }

        int ncp = nu * nv;
        for (int i = ncp; i < MAX_CP; ++i) {
            R[i] = 0.0; dRdxi[i] = 0.0; dRdeta[i] = 0.0;
        }
    }

    /// Evaluate surface point and tangent vectors
    KOKKOS_INLINE_FUNCTION
    void evaluate_surface(Real xi, Real eta,
                          Real x[3], Real a1[3], Real a2[3]) const {
        Real R[MAX_CP], dRdxi[MAX_CP], dRdeta[MAX_CP];
        compute_basis(xi, eta, R, dRdxi, dRdeta);

        x[0] = x[1] = x[2] = 0.0;
        a1[0] = a1[1] = a1[2] = 0.0;
        a2[0] = a2[1] = a2[2] = 0.0;

        int ncp = patch_.total_cp();
        for (int i = 0; i < ncp; ++i)
            for (int d = 0; d < 3; ++d) {
                x[d]  += R[i] * patch_.control_points[i][d];
                a1[d] += dRdxi[i] * patch_.control_points[i][d];
                a2[d] += dRdeta[i] * patch_.control_points[i][d];
            }
    }

    /// Stiffness matrix (membrane + bending) integrated with 3x3 Gauss
    KOKKOS_INLINE_FUNCTION
    void compute_stiffness(Real* K) const {
        int ncp = patch_.total_cp();
        int ndof = ncp * 3;
        for (int i = 0; i < ndof * ndof; ++i) K[i] = 0.0;

        Real Cm[9], Db[9];
        detail::planeStress_C(E_, nu_, Cm);
        detail::shellBending_D(E_, nu_, thickness_, Db);

        const Real gp3[3] = {0.5 - 0.5*std::sqrt(3.0/5.0),
                             0.5,
                             0.5 + 0.5*std::sqrt(3.0/5.0)};
        const Real gw3[3] = {5.0/18.0, 8.0/18.0, 5.0/18.0};

        for (int gi = 0; gi < 3; ++gi)
        for (int gj = 0; gj < 3; ++gj) {
            Real xi = gp3[gi], eta = gp3[gj];
            Real w = gw3[gi] * gw3[gj];

            Real R[MAX_CP], dRdxi[MAX_CP], dRdeta[MAX_CP];
            compute_basis(xi, eta, R, dRdxi, dRdeta);

            Real a1[3] = {0,0,0}, a2[3] = {0,0,0};
            for (int i = 0; i < ncp; ++i)
                for (int d = 0; d < 3; ++d) {
                    a1[d] += dRdxi[i] * patch_.control_points[i][d];
                    a2[d] += dRdeta[i] * patch_.control_points[i][d];
                }

            Real a3[3];
            detail::cross3(a1, a2, a3);
            Real j_surf = detail::norm3(a3);
            if (j_surf < 1.0e-30) continue;
            a3[0] /= j_surf; a3[1] /= j_surf; a3[2] /= j_surf;

            Real wt = w * j_surf;

            // Membrane stiffness contribution
            for (int a = 0; a < ncp; ++a)
                for (int b = 0; b < ncp; ++b) {
                    Real dRa_d1 = dRdxi[a], dRa_d2 = dRdeta[a];
                    Real dRb_d1 = dRdxi[b], dRb_d2 = dRdeta[b];

                    for (int di = 0; di < 3; ++di)
                        for (int dj = 0; dj < 3; ++dj) {
                            Real Ba[3] = {
                                dRa_d1 * a1[di],
                                dRa_d2 * a2[di],
                                dRa_d1 * a2[di] + dRa_d2 * a1[di]
                            };
                            Real Bb[3] = {
                                dRb_d1 * a1[dj],
                                dRb_d2 * a2[dj],
                                dRb_d1 * a2[dj] + dRb_d2 * a1[dj]
                            };

                            Real val = 0.0;
                            for (int p = 0; p < 3; ++p) {
                                Real cb = 0.0;
                                for (int q = 0; q < 3; ++q)
                                    cb += Cm[p*3+q] * Bb[q];
                                val += Ba[p] * cb;
                            }

                            K[(a*3+di)*ndof + (b*3+dj)] += val * thickness_ * wt;
                        }

                    // Bending contribution (simplified penalty)
                    Real g11 = detail::dot3(a1, a1);
                    Real g22 = detail::dot3(a2, a2);
                    Real g12 = detail::dot3(a1, a2);
                    Real det_g = g11 * g22 - g12 * g12;
                    if (det_g < 1.0e-30) continue;
                    Real inv_det_g = 1.0 / det_g;
                    Real g11c =  g22 * inv_det_g;
                    Real g22c =  g11 * inv_det_g;
                    Real g12c = -g12 * inv_det_g;

                    Real bend_val = dRa_d1*dRb_d1*g11c + dRa_d2*dRb_d2*g22c +
                                    (dRa_d1*dRb_d2 + dRa_d2*dRb_d1)*g12c;

                    for (int d = 0; d < 3; ++d)
                        K[(a*3+d)*ndof + (b*3+d)] += bend_val * a3[d]*a3[d] * Db[0] * wt;
                }
        }

        // Symmetrize
        for (int i = 0; i < ndof; ++i)
            for (int j = i+1; j < ndof; ++j) {
                Real avg = 0.5 * (K[i*ndof+j] + K[j*ndof+i]);
                K[i*ndof+j] = avg;
                K[j*ndof+i] = avg;
            }
    }

    /// Surface area
    KOKKOS_INLINE_FUNCTION
    Real area() const {
        const Real gp3[3] = {0.5 - 0.5*std::sqrt(3.0/5.0), 0.5,
                             0.5 + 0.5*std::sqrt(3.0/5.0)};
        const Real gw3[3] = {5.0/18.0, 8.0/18.0, 5.0/18.0};
        Real total = 0.0;
        int ncp = patch_.total_cp();

        for (int gi = 0; gi < 3; ++gi)
        for (int gj = 0; gj < 3; ++gj) {
            Real R[MAX_CP], dRdxi[MAX_CP], dRdeta[MAX_CP];
            compute_basis(gp3[gi], gp3[gj], R, dRdxi, dRdeta);

            Real a1[3] = {0,0,0}, a2[3] = {0,0,0};
            for (int i = 0; i < ncp; ++i)
                for (int d = 0; d < 3; ++d) {
                    a1[d] += dRdxi[i] * patch_.control_points[i][d];
                    a2[d] += dRdeta[i] * patch_.control_points[i][d];
                }

            Real a3[3];
            detail::cross3(a1, a2, a3);
            total += detail::norm3(a3) * gw3[gi] * gw3[gj];
        }
        return total;
    }

    KOKKOS_INLINE_FUNCTION const NURBSPatch& patch() const { return patch_; }

private:
    NURBSPatch patch_;
    Real E_, nu_, thickness_;

    /// Cox-de Boor B-spline basis and first derivative
    KOKKOS_INLINE_FUNCTION
    static void bspline_basis_and_deriv(Real t, int p,
                                         const Real* knots, int n_knots,
                                         int n_basis,
                                         Real* N, Real* dN) {
        for (int i = 0; i < n_basis; ++i) { N[i] = 0.0; dN[i] = 0.0; }

        // Find knot span
        int span = p;
        for (int i = p; i < n_knots - p - 2; ++i) {
            if (t >= knots[i] && t < knots[i+1]) { span = i; break; }
        }
        if (t >= knots[n_knots - p - 1]) span = n_knots - p - 2;

        // De Boor recursion for degree p
        Real Nloc[MAX_CP_PER_DIR + 1];
        for (int i = 0; i <= p; ++i) Nloc[i] = 0.0;
        Nloc[0] = 1.0;

        for (int j = 1; j <= p; ++j) {
            Real saved = 0.0;
            for (int r = 0; r < j; ++r) {
                Real rl = knots[span + 1 + r] - t;
                Real ll = t - knots[span + 1 - j + r];
                Real denom = rl + ll;
                Real temp = (denom > 1.0e-30) ? Nloc[r] / denom : 0.0;
                Nloc[r] = saved + rl * temp;
                saved = ll * temp;
            }
            Nloc[j] = saved;
        }

        for (int i = 0; i <= p; ++i) {
            int global_idx = span - p + i;
            if (global_idx >= 0 && global_idx < n_basis)
                N[global_idx] = Nloc[i];
        }

        // Derivative via degree p-1 basis
        if (p >= 1) {
            Real Npm1[MAX_CP_PER_DIR + 1];
            for (int i = 0; i <= p; ++i) Npm1[i] = 0.0;
            Npm1[0] = 1.0;

            for (int j = 1; j <= p - 1; ++j) {
                Real saved2 = 0.0;
                for (int r = 0; r < j; ++r) {
                    Real rl2 = knots[span + 1 + r] - t;
                    Real ll2 = t - knots[span + 1 - j + r];
                    Real denom2 = rl2 + ll2;
                    Real temp2 = (denom2 > 1.0e-30) ? Npm1[r] / denom2 : 0.0;
                    Npm1[r] = saved2 + rl2 * temp2;
                    saved2 = ll2 * temp2;
                }
                Npm1[j] = saved2;
            }

            for (int i = 0; i <= p; ++i) {
                int ki = span - p + i;
                if (ki < 0 || ki >= n_basis) continue;

                Real left_term = 0.0, right_term = 0.0;

                Real d1 = knots[ki + p] - knots[ki];
                if (d1 > 1.0e-30 && i > 0 && (i-1) <= p-1)
                    left_term = static_cast<Real>(p) * Npm1[i - 1] / d1;

                Real d2 = knots[ki + p + 1] - knots[ki + 1];
                if (d2 > 1.0e-30 && i <= p-1)
                    right_term = static_cast<Real>(p) * Npm1[i] / d2;

                dN[ki] = left_term - right_term;
            }
        }
    }
};

} // namespace discretization
} // namespace nxs
