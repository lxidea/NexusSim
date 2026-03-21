#pragma once

/**
 * @file assembly_wave43.hpp
 * @brief Wave 43: Per-element-type implicit assemblers and element buffer system
 *
 * Components:
 * (a) Per-Element-Type Implicit Assemblers
 *   1.  ElementAssemblerBase  - Abstract base with virtual assemble_element_stiffness
 *   2.  Hex8Assembler         - 8-node brick (24x24), 2x2x2 Gauss quadrature
 *   3.  Hex20Assembler        - 20-node brick (60x60), 3x3x3 Gauss quadrature
 *   4.  Tet4Assembler         - 4-node tet (12x12), single-point exact integration
 *   5.  Tet10Assembler        - 10-node tet (30x30), 4-point Gauss quadrature
 *   6.  Shell4Assembler       - 4-node shell (24x24), membrane + bending
 *   7.  Shell3Assembler       - 3-node triangle shell (18x18)
 *   8.  Beam2Assembler        - 2-node Euler-Bernoulli beam (12x12), analytical
 *   9.  SpringAssembler       - Spring element, direct stiffness
 *  10.  AssemblyDispatcher    - Factory/cache returning correct assembler by type
 *
 * (b) Element Buffer System
 *   1.  IntegrationPointState  - Per-IP state: stress, strain, history, scalars
 *   2.  ElementBuffer          - Per-element storage with variable IP/layer count
 *   3.  ElementBufferManager   - Collection manager with MaterialState conversion
 *
 * References:
 *  - Hughes (2000) "The Finite Element Method"
 *  - Belytschko, Liu, Moran (2000) "Nonlinear Finite Elements"
 *  - Zienkiewicz & Taylor (2000) "The Finite Element Method", Vol. 1-2
 */

#include <nexussim/core/types.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/physics/element.hpp>
#include <vector>
#include <array>
#include <memory>
#include <cmath>
#include <cstring>
#include <cassert>
#include <stdexcept>
#include <unordered_map>

namespace nxs {
namespace solver {

using Real = nxs::Real;

// ============================================================================
// Forward declarations
// ============================================================================

class ElementAssemblerBase;

// ============================================================================
// Namespace-local utility helpers
// ============================================================================

namespace detail43 {

/// Zero a flat matrix K[n*n]
inline void mat_zero(Real* K, int n) {
    std::memset(K, 0, static_cast<std::size_t>(n * n) * sizeof(Real));
}

/// K[i*n+j] += val
inline void mat_add(Real* K, int n, int i, int j, Real val) {
    K[i * n + j] += val;
}

/// Symmetric add: K[i,j] += val  and  K[j,i] += val  (i != j)
inline void mat_sym_add(Real* K, int n, int i, int j, Real val) {
    K[i * n + j] += val;
    K[j * n + i] += val;
}

/// C = A^T * B   (A: m x k,  B: m x p -> C: k x p), all row-major
inline void matTmul(const Real* A, const Real* B, Real* C, int m, int k, int p) {
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < p; ++j) {
            Real s = 0.0;
            for (int r = 0; r < m; ++r) s += A[r * k + i] * B[r * p + j];
            C[i * p + j] = s;
        }
}

/// C = A * B  (A: m x k,  B: k x p -> C: m x p), all row-major, accumulate
inline void matmul_add(const Real* A, const Real* B, Real* C, int m, int k, int p) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j) {
            Real s = 0.0;
            for (int r = 0; r < k; ++r) s += A[i * k + r] * B[r * p + j];
            C[i * p + j] += s;
        }
}

/// 3x3 determinant
inline Real det3(const Real* J) {
    return J[0]*(J[4]*J[8]-J[5]*J[7])
          -J[1]*(J[3]*J[8]-J[5]*J[6])
          +J[2]*(J[3]*J[7]-J[4]*J[6]);
}

/// Inverse of 3x3 matrix (overwrites Jinv), returns det
inline Real inv3(const Real* J, Real* Jinv) {
    Real d = det3(J);
    Real id = 1.0 / d;
    Jinv[0] =  (J[4]*J[8]-J[5]*J[7])*id;
    Jinv[1] = -(J[1]*J[8]-J[2]*J[7])*id;
    Jinv[2] =  (J[1]*J[5]-J[2]*J[4])*id;
    Jinv[3] = -(J[3]*J[8]-J[5]*J[6])*id;
    Jinv[4] =  (J[0]*J[8]-J[2]*J[6])*id;
    Jinv[5] = -(J[0]*J[5]-J[2]*J[3])*id;
    Jinv[6] =  (J[3]*J[7]-J[4]*J[6])*id;
    Jinv[7] = -(J[0]*J[7]-J[1]*J[6])*id;
    Jinv[8] =  (J[0]*J[4]-J[1]*J[3])*id;
    return d;
}

/// 2x2 determinant
inline Real det2(const Real* J) {
    return J[0]*J[3] - J[1]*J[2];
}

/// Build 6x3n B-matrix for a 3D solid element at given natural coords.
/// dN[n*3]: dN/dxi, dN/deta, dN/dzeta for each node.
/// coords[n*3]: nodal coordinates.
/// B[6*(3*n)]: output B-matrix.
/// Returns det(J).
inline Real build_B_3D(const Real* dN_nat, const Real* coords, int n_nodes, Real* B) {
    // Compute Jacobian J[3x3]
    Real J[9] = {};
    for (int a = 0; a < n_nodes; ++a) {
        for (int i = 0; i < 3; ++i)       // param direction
            for (int j = 0; j < 3; ++j)   // spatial direction
                J[i*3+j] += dN_nat[a*3+i] * coords[a*3+j];
    }
    Real Jinv[9];
    Real detJ = inv3(J, Jinv);

    // dN/dx = Jinv^T * dN/dxi  (Jinv^T means Jinv transposed)
    // dNdx[a][j] = sum_i Jinv[j,i] * dN_nat[a,i]  (Jinv^T applied)
    // Note: Jinv is already Jinv, so Jinv^T[j,i] = Jinv[i,j]
    // dNdx[a][j] = sum_i Jinv[i,j] * dN_nat[a,i]  -- correct physical gradient
    // Using J^{-T} * dN/dxi convention (correct formula for isoparametric):
    //   dN/dx_j = sum_i (J^{-T})_{j,i} * dN/dxi_i = sum_i Jinv[i,j] * dN/dxi_i
    int ndof = 3 * n_nodes;
    std::memset(B, 0, static_cast<std::size_t>(6 * ndof) * sizeof(Real));
    for (int a = 0; a < n_nodes; ++a) {
        Real dNdx = 0.0, dNdy = 0.0, dNdz = 0.0;
        for (int i = 0; i < 3; ++i) {
            dNdx += Jinv[i*3+0] * dN_nat[a*3+i];
            dNdy += Jinv[i*3+1] * dN_nat[a*3+i];
            dNdz += Jinv[i*3+2] * dN_nat[a*3+i];
        }
        int col = 3 * a;
        // Row 0: eps_xx = dNdx * u
        B[0 * ndof + col + 0] = dNdx;
        // Row 1: eps_yy = dNdy * v
        B[1 * ndof + col + 1] = dNdy;
        // Row 2: eps_zz = dNdz * w
        B[2 * ndof + col + 2] = dNdz;
        // Row 3: gamma_xy = dNdy*u + dNdx*v
        B[3 * ndof + col + 0] = dNdy;
        B[3 * ndof + col + 1] = dNdx;
        // Row 4: gamma_yz = dNdz*v + dNdy*w
        B[4 * ndof + col + 1] = dNdz;
        B[4 * ndof + col + 2] = dNdy;
        // Row 5: gamma_xz = dNdz*u + dNdx*w
        B[5 * ndof + col + 0] = dNdz;
        B[5 * ndof + col + 2] = dNdx;
    }
    return detJ;
}

/// Build isotropic elasticity matrix D[6x6] from E, nu.
/// Voigt ordering: xx, yy, zz, xy, yz, xz.
inline void build_D_isotropic(Real E, Real nu, Real* D) {
    std::memset(D, 0, 36 * sizeof(Real));
    Real lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Real mu  = E / (2.0 * (1.0 + nu));
    Real c   = lam + 2.0 * mu;
    // Normal-normal coupling
    D[0*6+0] = c;    D[0*6+1] = lam;  D[0*6+2] = lam;
    D[1*6+0] = lam;  D[1*6+1] = c;    D[1*6+2] = lam;
    D[2*6+0] = lam;  D[2*6+1] = lam;  D[2*6+2] = c;
    // Shear (engineering shear strain convention, factor mu not 2mu)
    D[3*6+3] = mu;
    D[4*6+4] = mu;
    D[5*6+5] = mu;
}

/// Accumulate K_e += w * detJ * B^T * D * B
/// B[6 x ndof], D[6 x 6], K[ndof x ndof], all row-major.
inline void accum_BtDB(const Real* B, const Real* D, Real* K, int ndof, Real w_detJ) {
    // Tmp = D * B  [6 x ndof]
    std::vector<Real> DB(6 * ndof, 0.0);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < ndof; ++j)
            for (int k = 0; k < 6; ++k)
                DB[i * ndof + j] += D[i * 6 + k] * B[k * ndof + j];
    // K += w_detJ * B^T * DB  [ndof x ndof]
    for (int i = 0; i < ndof; ++i)
        for (int j = 0; j < ndof; ++j)
            for (int k = 0; k < 6; ++k)
                K[i * ndof + j] += w_detJ * B[k * ndof + i] * DB[k * ndof + j];
}

/// Accumulate K_e += w * B^T * D * B for plane-stress/shell elements.
/// B[3 x ndof], D[3 x 3] (plane-stress), K[ndof x ndof], all row-major.
inline void accum_BtDB_ps(const Real* B, const Real* D, Real* K, int ndof, Real w_detJ) {
    // DB = D * B  [3 x ndof]
    std::vector<Real> DB(3 * ndof, 0.0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < ndof; ++j)
            for (int k = 0; k < 3; ++k)
                DB[i * ndof + j] += D[i * 3 + k] * B[k * ndof + j];
    // K += w_detJ * B^T * DB  [ndof x ndof]
    for (int i = 0; i < ndof; ++i)
        for (int j = 0; j < ndof; ++j)
            for (int k = 0; k < 3; ++k)
                K[i * ndof + j] += w_detJ * B[k * ndof + i] * DB[k * ndof + j];
}

} // namespace detail43

// ============================================================================
// (a) Per-Element-Type Implicit Assemblers
// ============================================================================

/**
 * @brief Abstract base for element stiffness assemblers.
 *
 * Subclasses implement assemble_element_stiffness() for a specific element
 * topology. Stiffness is returned as a flat row-major array K[ndof*ndof].
 */
class ElementAssemblerBase {
public:
    virtual ~ElementAssemblerBase() = default;

    /**
     * @brief Compute element stiffness matrix.
     *
     * @param elem_id  Element index (for error reporting).
     * @param coords   Nodal coordinates, layout [num_nodes * 3] (x0,y0,z0, x1,y1,z1, ...).
     * @param E        Young's modulus.
     * @param nu       Poisson's ratio.
     * @param K_local  Output stiffness [ndof * ndof], row-major, pre-zeroed by caller.
     */
    virtual void assemble_element_stiffness(
        std::size_t elem_id,
        const Real* coords,
        Real E,
        Real nu,
        Real* K_local) const = 0;

    /// Number of DOFs per element.
    virtual int ndof() const = 0;

    /// Element type identifier.
    virtual nxs::ElementType element_type() const = 0;
};

// ============================================================================
// Hex8Assembler  (8-node brick, 24x24, 2x2x2 Gauss)
// ============================================================================

/**
 * @brief Assembler for 8-node hexahedral elements.
 *
 * Uses 2x2x2 Gauss quadrature (8 integration points, weights = 1 each).
 * Natural coordinate range [-1, 1]^3.
 * Node ordering: right-hand rule, counter-clockwise bottom then top face.
 */
class Hex8Assembler : public ElementAssemblerBase {
public:
    static constexpr int NNODES = 8;
    static constexpr int NDOF   = 24; // 3 * 8

    int ndof() const override { return NDOF; }
    nxs::ElementType element_type() const override { return nxs::ElementType::Hex8; }

    void assemble_element_stiffness(
        std::size_t /*elem_id*/,
        const Real* coords,
        Real E,
        Real nu,
        Real* K_local) const override
    {
        detail43::mat_zero(K_local, NDOF);

        Real D[36];
        detail43::build_D_isotropic(E, nu, D);

        // 2x2x2 Gauss points and weights
        static const Real gp = 1.0 / std::sqrt(3.0);
        const Real xi_gp[2]  = {-gp, gp};
        const Real w_gp[2]   = {1.0, 1.0};

        Real dN_nat[NNODES * 3]; // dN/dxi, dN/deta, dN/dzeta
        Real B[6 * NDOF];

        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj)
        for (int gk = 0; gk < 2; ++gk) {
            Real xi   = xi_gp[gi];
            Real eta  = xi_gp[gj];
            Real zeta = xi_gp[gk];
            Real w    = w_gp[gi] * w_gp[gj] * w_gp[gk];

            // Shape function natural derivatives for Hex8
            // Node ordering: (xi,eta,zeta) = (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
            //                                (-1,-1,1),  (1,-1,1),  (1,1,1),  (-1,1,1)
            const Real xi_n[8]   = {-1,1,1,-1,-1,1,1,-1};
            const Real eta_n[8]  = {-1,-1,1,1,-1,-1,1,1};
            const Real zeta_n[8] = {-1,-1,-1,-1,1,1,1,1};
            for (int a = 0; a < 8; ++a) {
                Real xia = xi_n[a], etaa = eta_n[a], zetaa = zeta_n[a];
                dN_nat[a*3+0] = 0.125 * xia   * (1.0 + etaa*eta)  * (1.0 + zetaa*zeta);
                dN_nat[a*3+1] = 0.125 * etaa  * (1.0 + xia*xi)    * (1.0 + zetaa*zeta);
                dN_nat[a*3+2] = 0.125 * zetaa * (1.0 + xia*xi)    * (1.0 + etaa*eta);
            }

            Real detJ = detail43::build_B_3D(dN_nat, coords, NNODES, B);
            detail43::accum_BtDB(B, D, K_local, NDOF, w * std::abs(detJ));
        }
    }
};

// ============================================================================
// Hex20Assembler  (20-node serendipity brick, 60x60, 3x3x3 Gauss)
// ============================================================================

/**
 * @brief Assembler for 20-node hexahedral elements (serendipity).
 *
 * Uses 3x3x3 Gauss quadrature (27 integration points).
 * Nodes: 8 corners + 12 mid-edge nodes.
 * Natural coordinate range [-1, 1]^3.
 */
class Hex20Assembler : public ElementAssemblerBase {
public:
    static constexpr int NNODES = 20;
    static constexpr int NDOF   = 60; // 3 * 20

    int ndof() const override { return NDOF; }
    nxs::ElementType element_type() const override { return nxs::ElementType::Hex20; }

    void assemble_element_stiffness(
        std::size_t /*elem_id*/,
        const Real* coords,
        Real E,
        Real nu,
        Real* K_local) const override
    {
        detail43::mat_zero(K_local, NDOF);

        Real D[36];
        detail43::build_D_isotropic(E, nu, D);

        // 3-point Gauss rule: nodes at ±sqrt(3/5), 0; weights 5/9, 8/9, 5/9
        static const Real gp3[3]  = {-std::sqrt(0.6), 0.0, std::sqrt(0.6)};
        static const Real wt3[3]  = {5.0/9.0, 8.0/9.0, 5.0/9.0};

        Real dN_nat[NNODES * 3];
        Real B[6 * NDOF];

        for (int gi = 0; gi < 3; ++gi)
        for (int gj = 0; gj < 3; ++gj)
        for (int gk = 0; gk < 3; ++gk) {
            Real xi   = gp3[gi];
            Real eta  = gp3[gj];
            Real zeta = gp3[gk];
            Real w    = wt3[gi] * wt3[gj] * wt3[gk];

            compute_hex20_dN(xi, eta, zeta, dN_nat);

            Real detJ = detail43::build_B_3D(dN_nat, coords, NNODES, B);
            detail43::accum_BtDB(B, D, K_local, NDOF, w * std::abs(detJ));
        }
    }

private:
    /// Compute dN/d(xi,eta,zeta) for 20-node serendipity hex.
    /// Node ordering: 8 corners (same as Hex8) + 12 mid-edge nodes.
    static void compute_hex20_dN(Real xi, Real eta, Real zeta, Real* dN) {
        // Corner nodes (1-8): same sign convention as Hex8
        // xi_c, eta_c, zeta_c in {-1,+1}
        const Real xc[8]   = {-1, 1, 1,-1,-1, 1, 1,-1};
        const Real ec[8]   = {-1,-1, 1, 1,-1,-1, 1, 1};
        const Real zc[8]   = {-1,-1,-1,-1, 1, 1, 1, 1};

        // Mid-edge nodes 8-19
        // Parametric coords of mid-edge nodes:
        // 8:(0,-1,-1), 9:(1,0,-1), 10:(0,1,-1), 11:(-1,0,-1)
        // 12:(0,-1,1), 13:(1,0,1), 14:(0,1,1), 15:(-1,0,1)
        // 16:(-1,-1,0), 17:(1,-1,0), 18:(1,1,0), 19:(-1,1,0)

        // Corner node shape functions:
        // N_a = 1/8*(1+xi_a*xi)*(1+eta_a*eta)*(1+zeta_a*zeta)*(xi_a*xi+eta_a*eta+zeta_a*zeta-2)
        for (int a = 0; a < 8; ++a) {
            Real x0 = xc[a]*xi, e0 = ec[a]*eta, z0 = zc[a]*zeta;
            Real f = (1.0+x0)*(1.0+e0)*(1.0+z0);
            Real s = x0 + e0 + z0 - 2.0;
            // dN/dxi
            dN[a*3+0] = 0.125 * xc[a] * (1.0+e0)*(1.0+z0)*s
                      + 0.125 * f * xc[a];
            // dN/deta
            dN[a*3+1] = 0.125 * ec[a] * (1.0+x0)*(1.0+z0)*s
                      + 0.125 * f * ec[a];
            // dN/dzeta
            dN[a*3+2] = 0.125 * zc[a] * (1.0+x0)*(1.0+e0)*s
                      + 0.125 * f * zc[a];
        }

        // Mid-edge nodes dN/dxi, dN/deta, dN/dzeta
        // Node 8: (0,-1,-1): N = 1/4*(1-xi^2)*(1-eta)*(1-zeta)
        dN[8*3+0] = 0.25 * (-2.0*xi)*(1.0-eta)*(1.0-zeta);
        dN[8*3+1] = 0.25 * (1.0-xi*xi)*(-1.0)*(1.0-zeta);
        dN[8*3+2] = 0.25 * (1.0-xi*xi)*(1.0-eta)*(-1.0);

        // Node 9: (1,0,-1): N = 1/4*(1+xi)*(1-eta^2)*(1-zeta)
        dN[9*3+0] = 0.25 * (1.0)*(1.0-eta*eta)*(1.0-zeta);
        dN[9*3+1] = 0.25 * (1.0+xi)*(-2.0*eta)*(1.0-zeta);
        dN[9*3+2] = 0.25 * (1.0+xi)*(1.0-eta*eta)*(-1.0);

        // Node 10: (0,1,-1): N = 1/4*(1-xi^2)*(1+eta)*(1-zeta)
        dN[10*3+0] = 0.25 * (-2.0*xi)*(1.0+eta)*(1.0-zeta);
        dN[10*3+1] = 0.25 * (1.0-xi*xi)*(1.0)*(1.0-zeta);
        dN[10*3+2] = 0.25 * (1.0-xi*xi)*(1.0+eta)*(-1.0);

        // Node 11: (-1,0,-1): N = 1/4*(1-xi)*(1-eta^2)*(1-zeta)
        dN[11*3+0] = 0.25 * (-1.0)*(1.0-eta*eta)*(1.0-zeta);
        dN[11*3+1] = 0.25 * (1.0-xi)*(-2.0*eta)*(1.0-zeta);
        dN[11*3+2] = 0.25 * (1.0-xi)*(1.0-eta*eta)*(-1.0);

        // Node 12: (0,-1,1): N = 1/4*(1-xi^2)*(1-eta)*(1+zeta)
        dN[12*3+0] = 0.25 * (-2.0*xi)*(1.0-eta)*(1.0+zeta);
        dN[12*3+1] = 0.25 * (1.0-xi*xi)*(-1.0)*(1.0+zeta);
        dN[12*3+2] = 0.25 * (1.0-xi*xi)*(1.0-eta)*(1.0);

        // Node 13: (1,0,1): N = 1/4*(1+xi)*(1-eta^2)*(1+zeta)
        dN[13*3+0] = 0.25 * (1.0)*(1.0-eta*eta)*(1.0+zeta);
        dN[13*3+1] = 0.25 * (1.0+xi)*(-2.0*eta)*(1.0+zeta);
        dN[13*3+2] = 0.25 * (1.0+xi)*(1.0-eta*eta)*(1.0);

        // Node 14: (0,1,1): N = 1/4*(1-xi^2)*(1+eta)*(1+zeta)
        dN[14*3+0] = 0.25 * (-2.0*xi)*(1.0+eta)*(1.0+zeta);
        dN[14*3+1] = 0.25 * (1.0-xi*xi)*(1.0)*(1.0+zeta);
        dN[14*3+2] = 0.25 * (1.0-xi*xi)*(1.0+eta)*(1.0);

        // Node 15: (-1,0,1): N = 1/4*(1-xi)*(1-eta^2)*(1+zeta)
        dN[15*3+0] = 0.25 * (-1.0)*(1.0-eta*eta)*(1.0+zeta);
        dN[15*3+1] = 0.25 * (1.0-xi)*(-2.0*eta)*(1.0+zeta);
        dN[15*3+2] = 0.25 * (1.0-xi)*(1.0-eta*eta)*(1.0);

        // Node 16: (-1,-1,0): N = 1/4*(1-xi)*(1-eta)*(1-zeta^2)
        dN[16*3+0] = 0.25 * (-1.0)*(1.0-eta)*(1.0-zeta*zeta);
        dN[16*3+1] = 0.25 * (1.0-xi)*(-1.0)*(1.0-zeta*zeta);
        dN[16*3+2] = 0.25 * (1.0-xi)*(1.0-eta)*(-2.0*zeta);

        // Node 17: (1,-1,0): N = 1/4*(1+xi)*(1-eta)*(1-zeta^2)
        dN[17*3+0] = 0.25 * (1.0)*(1.0-eta)*(1.0-zeta*zeta);
        dN[17*3+1] = 0.25 * (1.0+xi)*(-1.0)*(1.0-zeta*zeta);
        dN[17*3+2] = 0.25 * (1.0+xi)*(1.0-eta)*(-2.0*zeta);

        // Node 18: (1,1,0): N = 1/4*(1+xi)*(1+eta)*(1-zeta^2)
        dN[18*3+0] = 0.25 * (1.0)*(1.0+eta)*(1.0-zeta*zeta);
        dN[18*3+1] = 0.25 * (1.0+xi)*(1.0)*(1.0-zeta*zeta);
        dN[18*3+2] = 0.25 * (1.0+xi)*(1.0+eta)*(-2.0*zeta);

        // Node 19: (-1,1,0): N = 1/4*(1-xi)*(1+eta)*(1-zeta^2)
        dN[19*3+0] = 0.25 * (-1.0)*(1.0+eta)*(1.0-zeta*zeta);
        dN[19*3+1] = 0.25 * (1.0-xi)*(1.0)*(1.0-zeta*zeta);
        dN[19*3+2] = 0.25 * (1.0-xi)*(1.0+eta)*(-2.0*zeta);
    }
};

// ============================================================================
// Tet4Assembler  (4-node tet, 12x12, constant-strain single-point integration)
// ============================================================================

/**
 * @brief Assembler for 4-node tetrahedral elements.
 *
 * Exact single-point integration (constant-strain element).
 * B-matrix is constant; K_e = V * B^T * D * B  where V = det(J)/6.
 * Node ordering: right-hand rule (outward normals).
 */
class Tet4Assembler : public ElementAssemblerBase {
public:
    static constexpr int NNODES = 4;
    static constexpr int NDOF   = 12;

    int ndof() const override { return NDOF; }
    nxs::ElementType element_type() const override { return nxs::ElementType::Tet4; }

    void assemble_element_stiffness(
        std::size_t /*elem_id*/,
        const Real* coords,
        Real E,
        Real nu,
        Real* K_local) const override
    {
        detail43::mat_zero(K_local, NDOF);

        Real D[36];
        detail43::build_D_isotropic(E, nu, D);

        // For Tet4 the Jacobian is constant.
        // dN/dxi for standard tet reference (L1,L2,L3, L4=1-L1-L2-L3):
        // N1=L1, N2=L2, N3=L3, N4=1-L1-L2-L3
        // dN1/dxi=1, dN1/deta=0, dN1/dzeta=0
        // dN2/dxi=0, dN2/deta=1, dN2/dzeta=0
        // dN3/dxi=0, dN3/deta=0, dN3/dzeta=1
        // dN4/dxi=-1, dN4/deta=-1, dN4/dzeta=-1
        Real dN_nat[NNODES * 3] = {
            1,0,0,
            0,1,0,
            0,0,1,
           -1,-1,-1
        };

        Real B[6 * NDOF];
        Real detJ = detail43::build_B_3D(dN_nat, coords, NNODES, B);
        Real vol = std::abs(detJ) / 6.0; // Tet volume = det(J)/6

        detail43::accum_BtDB(B, D, K_local, NDOF, vol);
    }
};

// ============================================================================
// Tet10Assembler  (10-node tet, 30x30, 4-point Gauss)
// ============================================================================

/**
 * @brief Assembler for 10-node tetrahedral elements (quadratic).
 *
 * Uses 4-point symmetric Gauss quadrature (exact for degree-3 polynomials).
 * Nodes: 4 corners + 6 mid-edge nodes.
 */
class Tet10Assembler : public ElementAssemblerBase {
public:
    static constexpr int NNODES = 10;
    static constexpr int NDOF   = 30;

    int ndof() const override { return NDOF; }
    nxs::ElementType element_type() const override { return nxs::ElementType::Tet10; }

    void assemble_element_stiffness(
        std::size_t /*elem_id*/,
        const Real* coords,
        Real E,
        Real nu,
        Real* K_local) const override
    {
        detail43::mat_zero(K_local, NDOF);

        Real D[36];
        detail43::build_D_isotropic(E, nu, D);

        // 4-point Gauss quadrature for tetrahedron
        // Points (a,b,b,b) and permutations with a=0.58541..., b=0.13819...
        static const Real a = 0.5854101966249685;
        static const Real b = 0.1381966011250105;
        // 4-point symmetric rule: each point gets weight 1/4 of the unit tet volume.
        // Volume = |detJ|/6, so effective w_detJ = (1/4) * |detJ|/6 per Gauss point.
        const Real gp[4][3] = {
            {a, b, b},
            {b, a, b},
            {b, b, a},
            {b, b, b}
        };
        // Weight for 4-point tet rule: each weight = 1/4 (barycentric), total = 1
        // Volume element = |detJ|/6, so w_detJ_eff = (1/4) * |detJ|/6 * ...
        // Actually accum_BtDB uses: K += w_detJ * B^T*D*B
        // For tet: volume integral = |detJ|/6 * sum_i w_i * f(xi_i), with sum w_i = 1
        // => pass w_detJ = w_i * |detJ| / 6

        Real dN_nat[NNODES * 3];
        Real B[6 * NDOF];

        for (int g = 0; g < 4; ++g) {
            Real L1 = gp[g][0], L2 = gp[g][1], L3 = gp[g][2];
            Real L4 = 1.0 - L1 - L2 - L3;
            compute_tet10_dN(L1, L2, L3, L4, dN_nat);
            Real detJ = detail43::build_B_3D(dN_nat, coords, NNODES, B);
            // Each point gets equal weight 1/4, volume factor = |detJ|/6
            Real w_detJ = (1.0 / 4.0) * std::abs(detJ) / 6.0;
            detail43::accum_BtDB(B, D, K_local, NDOF, w_detJ);
        }
    }

private:
    /// Natural derivatives for Tet10 in barycentric coords.
    /// dN/d(L1,L2,L3) — L4 is dependent: L4 = 1-L1-L2-L3, dL4/dLi = -1.
    static void compute_tet10_dN(Real L1, Real L2, Real L3, Real L4, Real* dN) {
        // Corner nodes: N_a = L_a*(2*L_a - 1)
        // dN1/dL1 = 4*L1-1, dN1/dL2 = 0, dN1/dL3 = 0 (using chain rule for L4)
        // Mid-edge nodes: N_5 = 4*L1*L2, etc.

        // Map: dN/dxi = dN/dL1, dN/deta = dN/dL2, dN/dzeta = dN/dL3
        // with dL4/dLi = -1 where needed.

        // Corner node 0 (L1): N0 = L1*(2L1-1)
        dN[0*3+0] = 4.0*L1-1.0;  dN[0*3+1] = 0.0;          dN[0*3+2] = 0.0;
        // Corner node 1 (L2): N1 = L2*(2L2-1)
        dN[1*3+0] = 0.0;          dN[1*3+1] = 4.0*L2-1.0;  dN[1*3+2] = 0.0;
        // Corner node 2 (L3): N2 = L3*(2L3-1)
        dN[2*3+0] = 0.0;          dN[2*3+1] = 0.0;          dN[2*3+2] = 4.0*L3-1.0;
        // Corner node 3 (L4): N3 = L4*(2L4-1), dL4/dLi = -1
        Real dN3_dL4 = 4.0*L4-1.0;
        dN[3*3+0] = -dN3_dL4;   dN[3*3+1] = -dN3_dL4;   dN[3*3+2] = -dN3_dL4;

        // Mid-edge node 4 (L1-L2 edge): N4 = 4*L1*L2
        dN[4*3+0] = 4.0*L2;  dN[4*3+1] = 4.0*L1;  dN[4*3+2] = 0.0;
        // Mid-edge node 5 (L2-L3 edge): N5 = 4*L2*L3
        dN[5*3+0] = 0.0;     dN[5*3+1] = 4.0*L3;  dN[5*3+2] = 4.0*L2;
        // Mid-edge node 6 (L1-L3 edge): N6 = 4*L1*L3
        dN[6*3+0] = 4.0*L3;  dN[6*3+1] = 0.0;     dN[6*3+2] = 4.0*L1;
        // Mid-edge node 7 (L1-L4 edge): N7 = 4*L1*L4, dL4/dLi = -1
        dN[7*3+0] = 4.0*(L4-L1); dN[7*3+1] = -4.0*L1; dN[7*3+2] = -4.0*L1;
        // Mid-edge node 8 (L2-L4 edge): N8 = 4*L2*L4
        dN[8*3+0] = -4.0*L2; dN[8*3+1] = 4.0*(L4-L2); dN[8*3+2] = -4.0*L2;
        // Mid-edge node 9 (L3-L4 edge): N9 = 4*L3*L4
        dN[9*3+0] = -4.0*L3; dN[9*3+1] = -4.0*L3; dN[9*3+2] = 4.0*(L4-L3);
    }
};

// ============================================================================
// Shell4Assembler  (4-node shell, 24x24 = 4 nodes * 6 DOF)
// ============================================================================

/**
 * @brief Assembler for 4-node Mindlin-Reissner shell elements.
 *
 * 6 DOF per node: (u, v, w, theta_x, theta_y, theta_z).
 * Membrane (plane-stress) + bending (plate) contributions.
 * 2x2 Gauss quadrature in plane; through-thickness integration collapsed
 * into the bending stiffness coefficients (D_b = E*h^3/(12*(1-nu^2))).
 * Shell thickness default h = 1.0 (caller should scale D_b by h^3).
 *
 * coords[4*3]: (x,y,0) for each node (shell assumed in xy-plane).
 * Thickness is passed as the 5th "material" parameter via E_nu convention:
 * thickness = 1.0 (fixed default — override by subclass if needed).
 */
class Shell4Assembler : public ElementAssemblerBase {
public:
    static constexpr int NNODES = 4;
    static constexpr int NDOF   = 24; // 4 nodes * 6 DOF

    int ndof() const override { return NDOF; }
    nxs::ElementType element_type() const override { return nxs::ElementType::Shell4; }

    /// Thickness (default 1.0). Caller can override.
    Real thickness = 1.0;

    void assemble_element_stiffness(
        std::size_t /*elem_id*/,
        const Real* coords,
        Real E,
        Real nu,
        Real* K_local) const override
    {
        detail43::mat_zero(K_local, NDOF);

        Real h = thickness;

        // Plane-stress membrane stiffness D_m [3x3]
        Real c = E / (1.0 - nu*nu);
        Real Dm[9] = {
            c,      c*nu,   0.0,
            c*nu,   c,      0.0,
            0.0,    0.0,    c*(1.0-nu)*0.5
        };
        // Bending stiffness D_b = h^2/12 * D_m (thin plate theory, h factor)
        Real hb = h * h / 12.0;
        Real Db[9];
        for (int i = 0; i < 9; ++i) Db[i] = hb * Dm[i];

        // 2x2 Gauss quadrature in (xi, eta)
        static const Real gp = 1.0 / std::sqrt(3.0);
        const Real xi_gp[2]  = {-gp, gp};
        const Real w_gp[2]   = {1.0, 1.0};

        for (int gi = 0; gi < 2; ++gi)
        for (int gj = 0; gj < 2; ++gj) {
            Real xi  = xi_gp[gi];
            Real eta = xi_gp[gj];
            Real w   = w_gp[gi] * w_gp[gj];

            // Shape functions and derivatives for Q4 in 2D
            // N1=(1-xi)(1-eta)/4, N2=(1+xi)(1-eta)/4, N3=(1+xi)(1+eta)/4, N4=(1-xi)(1+eta)/4
            Real N[4]    = {0.25*(1-xi)*(1-eta), 0.25*(1+xi)*(1-eta),
                            0.25*(1+xi)*(1+eta), 0.25*(1-xi)*(1+eta)};
            Real dNdxi[4] = {-0.25*(1-eta), 0.25*(1-eta), 0.25*(1+eta), -0.25*(1+eta)};
            Real dNdet[4] = {-0.25*(1-xi), -0.25*(1+xi), 0.25*(1+xi),  0.25*(1-xi)};
            (void)N; // shape functions not needed for K

            // 2D Jacobian
            Real J2[4] = {};
            for (int a = 0; a < 4; ++a) {
                J2[0] += dNdxi[a] * coords[a*3+0]; // dX/dxi
                J2[1] += dNdxi[a] * coords[a*3+1]; // dY/dxi
                J2[2] += dNdet[a] * coords[a*3+0]; // dX/deta
                J2[3] += dNdet[a] * coords[a*3+1]; // dY/deta
            }
            Real detJ2 = detail43::det2(J2);
            Real J2inv[4];
            Real idetJ2 = 1.0 / detJ2;
            J2inv[0] =  J2[3]*idetJ2;
            J2inv[1] = -J2[1]*idetJ2;
            J2inv[2] = -J2[2]*idetJ2;
            J2inv[3] =  J2[0]*idetJ2;

            // Physical derivatives dN/dX, dN/dY
            Real dNdX[4], dNdY[4];
            for (int a = 0; a < 4; ++a) {
                dNdX[a] = J2inv[0]*dNdxi[a] + J2inv[1]*dNdet[a];
                dNdY[a] = J2inv[2]*dNdxi[a] + J2inv[3]*dNdet[a];
            }

            // Membrane B_m [3 x 8] (DOFs: u0,v0,u1,v1,u2,v2,u3,v3)
            // eps_xx = sum dNdX_a * u_a
            // eps_yy = sum dNdY_a * v_a
            // gamma_xy = sum (dNdY_a*u_a + dNdX_a*v_a)
            Real Bm[3 * 8] = {};
            for (int a = 0; a < 4; ++a) {
                Bm[0*8 + 2*a]   = dNdX[a];
                Bm[1*8 + 2*a+1] = dNdY[a];
                Bm[2*8 + 2*a]   = dNdY[a];
                Bm[2*8 + 2*a+1] = dNdX[a];
            }

            // Bending B_b [3 x 8] (DOFs: theta_x_0,theta_y_0,...)
            // kappa_xx = sum dNdX_a * theta_y_a
            // kappa_yy = -sum dNdY_a * theta_x_a
            // 2*kappa_xy = -sum dNdY_a * theta_y_a + sum dNdX_a * theta_x_a  (Kirchhoff)
            Real Bb[3 * 8] = {};
            for (int a = 0; a < 4; ++a) {
                // DOF order: theta_x_a, theta_y_a
                Bb[0*8 + 2*a+1] =  dNdX[a]; // kappa_xx from theta_y
                Bb[1*8 + 2*a]   = -dNdY[a]; // kappa_yy from theta_x
                Bb[2*8 + 2*a]   =  dNdX[a]; // kappa_xy from theta_x
                Bb[2*8 + 2*a+1] = -dNdY[a]; // kappa_xy from theta_y
            }

            // Membrane contribution: K_m (DOFs 0,1 of each node: u,v)
            // Assemble into global 24x24 K (DOF mapping: node a -> DOF 6*a, 6*a+1)
            Real Km[8*8] = {};
            detail43::accum_BtDB_ps(Bm, Dm, Km, 8, w * std::abs(detJ2) * h);
            for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
            for (int ia = 0; ia < 2; ++ia)
            for (int ib = 0; ib < 2; ++ib)
                K_local[(6*a+ia)*NDOF + (6*b+ib)] += Km[(2*a+ia)*8 + (2*b+ib)];

            // Bending contribution: K_b (DOFs 3,4 of each node: theta_x, theta_y)
            Real Kb[8*8] = {};
            detail43::accum_BtDB_ps(Bb, Db, Kb, 8, w * std::abs(detJ2));
            for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
            for (int ia = 0; ia < 2; ++ia)
            for (int ib = 0; ib < 2; ++ib)
                K_local[(6*a+3+ia)*NDOF + (6*b+3+ib)] += Kb[(2*a+ia)*8 + (2*b+ib)];

            // Transverse shear via symmetric B-matrix: k_s * B_s^T * B_s
            // gamma_xz = dw/dX - theta_y  =>  B_xz = [dN_a/dX, 0, -N_a] per node (w, tx, ty)
            // gamma_yz = dw/dY + theta_x  =>  B_yz = [dN_a/dY, N_a, 0]  per node
            // DOF local index: w=0, tx=1, ty=2 in the 3-vector per node
            // Global DOF: w -> 6*a+2, tx -> 6*a+3, ty -> 6*a+4
            Real G = E / (2.0*(1.0+nu));
            Real ks = (5.0/6.0) * G * h;
            Real ww = w * std::abs(detJ2);
            // Build Bs_xz [4*3] and Bs_yz [4*3] per node, rows = nodes, cols = (w,tx,ty)
            // K_shear += ks * ww * sum_a sum_b [ Bs_xz[a]^T*Bs_xz[b] + Bs_yz[a]^T*Bs_yz[b] ]
            for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) {
                // B_xz contribution: (dNdX_a * dNdX_b) for w-w,
                //                    (dNdX_a * (-N_b)) for w-ty,
                //                    ((-N_a) * dNdX_b) for ty-w,
                //                    ((-N_a) * (-N_b)) = N_a*N_b for ty-ty
                Real xz_ww   =  dNdX[a] * dNdX[b];
                Real xz_wty  = -dNdX[a] * N[b];   // w_a, ty_b
                Real xz_tyw  = -N[a] * dNdX[b];   // ty_a, w_b  (== xz_wty^T)
                Real xz_tyty =  N[a] * N[b];

                K_local[(6*a+2)*NDOF + (6*b+2)] += ks * ww * xz_ww;
                K_local[(6*a+2)*NDOF + (6*b+4)] += ks * ww * xz_wty;
                K_local[(6*a+4)*NDOF + (6*b+2)] += ks * ww * xz_tyw;
                K_local[(6*a+4)*NDOF + (6*b+4)] += ks * ww * xz_tyty;

                // B_yz contribution: gamma_yz = dw/dY + theta_x
                Real yz_ww   =  dNdY[a] * dNdY[b];
                Real yz_wtx  =  dNdY[a] * N[b];   // w_a, tx_b
                Real yz_txw  =  N[a] * dNdY[b];   // tx_a, w_b
                Real yz_txtx =  N[a] * N[b];

                K_local[(6*a+2)*NDOF + (6*b+2)] += ks * ww * yz_ww;
                K_local[(6*a+2)*NDOF + (6*b+3)] += ks * ww * yz_wtx;
                K_local[(6*a+3)*NDOF + (6*b+2)] += ks * ww * yz_txw;
                K_local[(6*a+3)*NDOF + (6*b+3)] += ks * ww * yz_txtx;
            }

            // Drilling DOF penalty (theta_z): symmetric N_a*N_b mass-like term
            Real k_drill = 1.0e-4 * E * h;
            for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
                K_local[(6*a+5)*NDOF + (6*b+5)] += k_drill * ww * N[a] * N[b];
        }
    }
};

// ============================================================================
// Shell3Assembler  (3-node triangle shell, 18x18)
// ============================================================================

/**
 * @brief Assembler for 3-node triangular shell elements.
 *
 * 6 DOF per node. Constant-strain (CST) membrane + discrete Kirchhoff bending.
 * Single-point integration (centroid) for membrane; exact for linear field.
 */
class Shell3Assembler : public ElementAssemblerBase {
public:
    static constexpr int NNODES = 3;
    static constexpr int NDOF   = 18; // 3 nodes * 6 DOF

    int ndof() const override { return NDOF; }
    nxs::ElementType element_type() const override { return nxs::ElementType::Shell3; }

    Real thickness = 1.0;

    void assemble_element_stiffness(
        std::size_t /*elem_id*/,
        const Real* coords,
        Real E,
        Real nu,
        Real* K_local) const override
    {
        detail43::mat_zero(K_local, NDOF);

        Real h = thickness;

        // Triangle area from vertices (x1,y1), (x2,y2), (x3,y3)
        Real x1 = coords[0], y1 = coords[1];
        Real x2 = coords[3], y2 = coords[4];
        Real x3 = coords[6], y3 = coords[7];
        Real area = 0.5 * std::abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1));
        if (area < 1.0e-14) return;

        // CST B-matrix (constant)
        Real b1 = y2 - y3, b2 = y3 - y1, b3 = y1 - y2;
        Real c1 = x3 - x2, c2 = x1 - x3, c3 = x2 - x1;
        Real inv2A = 1.0 / (2.0 * area);

        // B_m [3 x 6] (DOFs: u1,v1,u2,v2,u3,v3)
        Real Bm[3*6] = {
            b1*inv2A, 0,        b2*inv2A, 0,        b3*inv2A, 0,
            0,        c1*inv2A, 0,        c2*inv2A, 0,        c3*inv2A,
            c1*inv2A, b1*inv2A, c2*inv2A, b2*inv2A, c3*inv2A, b3*inv2A
        };

        // Plane-stress D_m
        Real cf = E / (1.0 - nu*nu);
        Real Dm[9] = {
            cf,     cf*nu, 0.0,
            cf*nu,  cf,    0.0,
            0.0,    0.0,   cf*(1.0-nu)*0.5
        };

        // Membrane stiffness K_m [6x6] = h*A * B^T * D * B
        Real Km[6*6] = {};
        detail43::accum_BtDB_ps(Bm, Dm, Km, 6, h * area);

        // Map into global 18x18 K (DOFs 0,1 of each node: u,v)
        for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        for (int ia = 0; ia < 2; ++ia)
        for (int ib = 0; ib < 2; ++ib)
            K_local[(6*a+ia)*NDOF + (6*b+ib)] += Km[(2*a+ia)*6 + (2*b+ib)];

        // Bending stiffness (DKT-style, simplified as constant curvature plate)
        Real Db_scale = E * h*h*h / (12.0*(1.0-nu*nu));
        // Use same B-matrix structure scaled for bending, applied to rotation DOFs
        Real Bb[3*6] = {
            b1*inv2A, 0,        b2*inv2A, 0,        b3*inv2A, 0,
            0,        c1*inv2A, 0,        c2*inv2A, 0,        c3*inv2A,
            c1*inv2A, b1*inv2A, c2*inv2A, b2*inv2A, c3*inv2A, b3*inv2A
        };
        Real Db[9] = {
            Db_scale,      Db_scale*nu, 0.0,
            Db_scale*nu,   Db_scale,    0.0,
            0.0,           0.0,         Db_scale*(1.0-nu)*0.5
        };
        Real Kb[6*6] = {};
        detail43::accum_BtDB_ps(Bb, Db, Kb, 6, area);

        // Map into global K (DOFs 3,4 of each node: theta_x, theta_y)
        for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        for (int ia = 0; ia < 2; ++ia)
        for (int ib = 0; ib < 2; ++ib)
            K_local[(6*a+3+ia)*NDOF + (6*b+3+ib)] += Kb[(2*a+ia)*6 + (2*b+ib)];

        // Drilling DOF (theta_z): small penalty
        Real k_drill = 1.0e-4 * E * h;
        Real k_diag = k_drill * area / 3.0; // lumped
        for (int a = 0; a < 3; ++a)
            K_local[(6*a+5)*NDOF + (6*a+5)] += k_diag;

        // Transverse shear: penalty for w DOF
        Real G  = E / (2.0*(1.0+nu));
        Real ks = (5.0/6.0) * G * h * area / 3.0; // lumped
        for (int a = 0; a < 3; ++a)
            K_local[(6*a+2)*NDOF + (6*a+2)] += ks;
    }
};

// ============================================================================
// Beam2Assembler  (2-node Euler-Bernoulli beam, 12x12, analytical)
// ============================================================================

/**
 * @brief Assembler for 2-node Euler-Bernoulli beam elements.
 *
 * 6 DOF per node: (u, v, w, theta_x, theta_y, theta_z).
 * Beam axis along x by convention. Length L = |x2 - x1|.
 * Stiffness: axial (EA/L), bending about z (EI_z), bending about y (EI_y),
 * torsion (GJ), shear (no shear deformation in classical E-B).
 *
 * Default: circular cross-section A=1, I=1/12 (unit square).
 * Caller provides E, nu; G = E/(2*(1+nu)).
 * Section properties: A, Iz, Iy, J (second moments of area) — use defaults.
 */
class Beam2Assembler : public ElementAssemblerBase {
public:
    static constexpr int NNODES = 2;
    static constexpr int NDOF   = 12;

    int ndof() const override { return NDOF; }
    nxs::ElementType element_type() const override { return nxs::ElementType::Beam2; }

    // Section properties (can be set by caller)
    Real A  = 1.0;    ///< Cross-sectional area
    Real Iz = 1.0/12.0; ///< Second moment of area about z-axis
    Real Iy = 1.0/12.0; ///< Second moment of area about y-axis
    Real J  = 1.0/6.0;  ///< Torsional constant (circular: pi*r^4/2)

    void assemble_element_stiffness(
        std::size_t /*elem_id*/,
        const Real* coords,
        Real E,
        Real nu,
        Real* K_local) const override
    {
        detail43::mat_zero(K_local, NDOF);

        // Beam length
        Real dx = coords[3] - coords[0];
        Real dy = coords[4] - coords[1];
        Real dz = coords[5] - coords[2];
        Real L  = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (L < 1.0e-14) return;

        Real G = E / (2.0 * (1.0 + nu));

        // Stiffness coefficients
        Real EA_L  = E * A  / L;
        Real GJ_L  = G * J  / L;
        Real EIz_L = E * Iz / L;
        Real EIy_L = E * Iy / L;

        // Analytical Euler-Bernoulli beam stiffness in local frame.
        // DOF ordering (local): u1, v1, w1, theta_x1, theta_y1, theta_z1,
        //                        u2, v2, w2, theta_x2, theta_y2, theta_z2
        // Indexed 0..11.

        // Axial (DOFs 0, 6)
        K_local[0*12+0]  =  EA_L;  K_local[0*12+6]  = -EA_L;
        K_local[6*12+0]  = -EA_L;  K_local[6*12+6]  =  EA_L;

        // Torsion (DOFs 3, 9)
        K_local[3*12+3]  =  GJ_L;  K_local[3*12+9]  = -GJ_L;
        K_local[9*12+3]  = -GJ_L;  K_local[9*12+9]  =  GJ_L;

        // Bending about z-axis (DOFs v: 1, 7 and theta_z: 5, 11)
        Real a1 = 12.0*EIz_L/(L*L);
        Real a2 =  6.0*EIz_L/L;
        Real a3 =  4.0*EIz_L;
        Real a4 =  2.0*EIz_L;
        // [v1, tz1, v2, tz2] at positions [1, 5, 7, 11]
        int dv[4] = {1, 5, 7, 11};
        Real Kbz[4][4] = {
            { a1,  a2, -a1,  a2},
            { a2,  a3, -a2,  a4},
            {-a1, -a2,  a1, -a2},
            { a2,  a4, -a2,  a3}
        };
        for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            K_local[dv[i]*12 + dv[j]] += Kbz[i][j];

        // Bending about y-axis (DOFs w: 2, 8 and theta_y: 4, 10)
        Real b1 = 12.0*EIy_L/(L*L);
        Real b2 =  6.0*EIy_L/L;
        Real b3 =  4.0*EIy_L;
        Real b4 =  2.0*EIy_L;
        // [w1, ty1, w2, ty2] at positions [2, 4, 8, 10]
        // Note: sign convention for y-bending: theta_y positive per right-hand rule
        int dw[4] = {2, 4, 8, 10};
        Real Kby[4][4] = {
            { b1, -b2, -b1, -b2},
            {-b2,  b3,  b2,  b4},
            {-b1,  b2,  b1,  b2},
            {-b2,  b4,  b2,  b3}
        };
        for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            K_local[dw[i]*12 + dw[j]] += Kby[i][j];
    }
};

// ============================================================================
// SpringAssembler  (spring element, direct stiffness)
// ============================================================================

/**
 * @brief Assembler for 2-node spring elements.
 *
 * Spring stiffness k applied along the line connecting the two nodes.
 * DOF per node: 3 (translational). Total: 6x6 stiffness.
 * k is passed as E (Young's modulus parameter); nu is ignored.
 */
class SpringAssembler : public ElementAssemblerBase {
public:
    // ndof can vary — expose for tests. Default: 6 (2 nodes * 3 DOF).
    int ndof() const override { return 6; }
    nxs::ElementType element_type() const override { return nxs::ElementType::Spring; }

    void assemble_element_stiffness(
        std::size_t /*elem_id*/,
        const Real* coords,
        Real k,    // spring stiffness (passed as E)
        Real /*nu*/,
        Real* K_local) const override
    {
        detail43::mat_zero(K_local, 6);

        // Direction cosines
        Real dx = coords[3] - coords[0];
        Real dy = coords[4] - coords[1];
        Real dz = coords[5] - coords[2];
        Real L  = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (L < 1.0e-14) {
            // Degenerate: apply stiffness along x
            K_local[0*6+0] =  k;  K_local[0*6+3] = -k;
            K_local[3*6+0] = -k;  K_local[3*6+3] =  k;
            return;
        }
        Real l = dx/L, m = dy/L, n = dz/L;

        // K = k * [e e^T, -e e^T; -e e^T, e e^T]  where e = (l, m, n)^T
        Real e[3] = {l, m, n};
        for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            Real val = k * e[i] * e[j];
            K_local[i*6+j]     =  val;
            K_local[i*6+(j+3)] = -val;
            K_local[(i+3)*6+j] = -val;
            K_local[(i+3)*6+(j+3)] = val;
        }
    }
};

// ============================================================================
// AssemblyDispatcher  (factory + cache)
// ============================================================================

/**
 * @brief Factory that returns per-type assembler instances (cached).
 *
 * Uses nxs::ElementType (from core/types.hpp).
 * Assemblers are lazily constructed and cached on first request.
 */
class AssemblyDispatcher {
public:
    AssemblyDispatcher() = default;

    /**
     * @brief Get (or create) the assembler for the given element type.
     * @return Pointer to the cached assembler (owned by this dispatcher).
     */
    ElementAssemblerBase* get_assembler(nxs::ElementType type) {
        auto it = cache_.find(static_cast<int>(type));
        if (it != cache_.end()) return it->second.get();

        std::unique_ptr<ElementAssemblerBase> asm_ptr;
        switch (type) {
            case nxs::ElementType::Hex8:
                asm_ptr = std::make_unique<Hex8Assembler>(); break;
            case nxs::ElementType::Hex20:
                asm_ptr = std::make_unique<Hex20Assembler>(); break;
            case nxs::ElementType::Tet4:
                asm_ptr = std::make_unique<Tet4Assembler>(); break;
            case nxs::ElementType::Tet10:
                asm_ptr = std::make_unique<Tet10Assembler>(); break;
            case nxs::ElementType::Shell4:
                asm_ptr = std::make_unique<Shell4Assembler>(); break;
            case nxs::ElementType::Shell3:
                asm_ptr = std::make_unique<Shell3Assembler>(); break;
            case nxs::ElementType::Beam2:
                asm_ptr = std::make_unique<Beam2Assembler>(); break;
            case nxs::ElementType::Spring:
                asm_ptr = std::make_unique<SpringAssembler>(); break;
            default:
                return nullptr; // Unsupported type
        }

        auto* raw = asm_ptr.get();
        cache_.emplace(static_cast<int>(type), std::move(asm_ptr));
        return raw;
    }

    /// Clear cached assemblers.
    void clear() { cache_.clear(); }

private:
    std::unordered_map<int, std::unique_ptr<ElementAssemblerBase>> cache_;
};

// ============================================================================
// (b) Element Buffer System
// ============================================================================

/**
 * @brief State at a single integration point.
 *
 * Mirrors nxs::physics::MaterialState but owned per element buffer.
 * Designed for efficient per-element access without indirection through
 * a global MaterialState array.
 */
struct IntegrationPointState {
    Real stress[6]  = {};  ///< Cauchy stress (Voigt: xx,yy,zz,xy,yz,xz)
    Real strain[6]  = {};  ///< Engineering strain
    Real history[64] = {}; ///< Internal history variables (NEXUSSIM_HISTORY_SIZE)
    Real plastic_strain = 0.0; ///< Effective plastic strain
    Real damage         = 0.0; ///< Damage parameter [0,1]
    Real temperature    = 293.15; ///< Temperature (K), default room temp

    /// Zero all fields (reset to reference state, keep default temperature).
    void clear() {
        for (int i = 0; i < 6; ++i) { stress[i] = 0.0; strain[i] = 0.0; }
        for (int i = 0; i < 64; ++i) history[i] = 0.0;
        plastic_strain = 0.0;
        damage         = 0.0;
        temperature    = 293.15;
    }
};

/**
 * @brief Per-element state storage with variable integration points and layers.
 *
 * Supports composite elements (multi-layer through-thickness integration)
 * as well as isotropic elements (1 layer).
 *
 * Storage layout: ip_states[ip * num_layers + layer]
 */
struct ElementBuffer {
    int num_integration_points = 0; ///< Number of in-plane / volume IPs
    int num_layers             = 1; ///< Through-thickness layers (composites)
    nxs::ElementType element_type = nxs::ElementType::Hex8;
    std::vector<IntegrationPointState> ip_states; ///< [num_ip * num_layers]

    ElementBuffer() = default;

    ElementBuffer(int num_ip, nxs::ElementType etype, int layers = 1)
        : num_integration_points(num_ip)
        , num_layers(layers)
        , element_type(etype)
        , ip_states(static_cast<std::size_t>(num_ip * layers))
    {}

    /**
     * @brief Access state for a given IP and layer.
     * @param ip    Integration point index [0, num_integration_points).
     * @param layer Layer index [0, num_layers).
     */
    IntegrationPointState& state(int ip, int layer = 0) {
        return ip_states[static_cast<std::size_t>(ip * num_layers + layer)];
    }

    const IntegrationPointState& state(int ip, int layer = 0) const {
        return ip_states[static_cast<std::size_t>(ip * num_layers + layer)];
    }

    /// Zero all IP states.
    void clear() {
        for (auto& s : ip_states) s.clear();
    }

    /// Total number of IP states stored.
    std::size_t total_states() const { return ip_states.size(); }
};

/**
 * @brief Default number of integration points per element type.
 *
 * Used by ElementBufferManager when no override is provided.
 */
inline int default_num_ip(nxs::ElementType type) {
    switch (type) {
        case nxs::ElementType::Tet4:   return 1;
        case nxs::ElementType::Tet10:  return 4;
        case nxs::ElementType::Hex8:   return 8;
        case nxs::ElementType::Hex20:  return 27;
        case nxs::ElementType::Hex27:  return 27;
        case nxs::ElementType::Wedge6: return 6;
        case nxs::ElementType::Shell4: return 4;  // 2x2 in-plane
        case nxs::ElementType::Shell3: return 1;
        case nxs::ElementType::Beam2:  return 2;
        case nxs::ElementType::Spring: return 1;
        default:                        return 1;
    }
}

/**
 * @brief Manager for a collection of element buffers.
 *
 * Allocates uniform element buffers of a single type, then provides
 * per-element access and bidirectional conversion to/from MaterialState.
 */
class ElementBufferManager {
public:
    ElementBufferManager() = default;

    /**
     * @brief Allocate buffers for num_elements elements of a given type.
     *
     * If num_ip <= 0, uses the default for element_type.
     * Resets any existing allocation.
     *
     * @param num_elements  Number of elements.
     * @param element_type  Element topology.
     * @param num_ip        Integration points per element (0 = use default).
     * @param num_layers    Through-thickness layers (1 = isotropic).
     */
    void allocate(std::size_t num_elements,
                  nxs::ElementType element_type,
                  int num_ip = 0,
                  int num_layers = 1)
    {
        element_type_  = element_type;
        int nip = (num_ip > 0) ? num_ip : default_num_ip(element_type);
        buffers_.clear();
        buffers_.reserve(num_elements);
        for (std::size_t i = 0; i < num_elements; ++i)
            buffers_.emplace_back(nip, element_type, num_layers);
    }

    /**
     * @brief Access element buffer by index.
     */
    ElementBuffer& get_buffer(std::size_t elem_id) {
        return buffers_.at(elem_id);
    }

    const ElementBuffer& get_buffer(std::size_t elem_id) const {
        return buffers_.at(elem_id);
    }

    /**
     * @brief Copy buffer state for (elem_id, ip) into a MaterialState.
     *
     * Transfers: stress, strain, history, plastic_strain, damage, temperature.
     */
    void copy_to_material_state(std::size_t elem_id,
                                int ip,
                                nxs::physics::MaterialState& ms,
                                int layer = 0) const
    {
        const IntegrationPointState& s = buffers_.at(elem_id).state(ip, layer);
        for (int i = 0; i < 6; ++i) ms.stress[i] = s.stress[i];
        for (int i = 0; i < 6; ++i) ms.strain[i] = s.strain[i];
        for (int i = 0; i < 64; ++i) ms.history[i] = s.history[i];
        ms.plastic_strain = s.plastic_strain;
        ms.damage         = s.damage;
        ms.temperature    = s.temperature;
    }

    /**
     * @brief Store MaterialState results back into the buffer for (elem_id, ip).
     *
     * Transfers: stress, strain, history, plastic_strain, damage, temperature.
     */
    void copy_from_material_state(std::size_t elem_id,
                                  int ip,
                                  const nxs::physics::MaterialState& ms,
                                  int layer = 0)
    {
        IntegrationPointState& s = buffers_.at(elem_id).state(ip, layer);
        for (int i = 0; i < 6; ++i) s.stress[i] = ms.stress[i];
        for (int i = 0; i < 6; ++i) s.strain[i] = ms.strain[i];
        for (int i = 0; i < 64; ++i) s.history[i] = ms.history[i];
        s.plastic_strain = ms.plastic_strain;
        s.damage         = ms.damage;
        s.temperature    = ms.temperature;
    }

    /// Zero all element buffers.
    void clear() {
        for (auto& buf : buffers_) buf.clear();
    }

    /// Number of allocated elements.
    std::size_t num_elements() const { return buffers_.size(); }

    /// Element type for this manager's allocation.
    nxs::ElementType element_type() const { return element_type_; }

private:
    std::vector<ElementBuffer> buffers_;
    nxs::ElementType element_type_ = nxs::ElementType::Hex8;
};

} // namespace solver
} // namespace nxs
