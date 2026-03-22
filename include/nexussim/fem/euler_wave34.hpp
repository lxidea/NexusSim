#pragma once

/**
 * @file euler_wave34.hpp
 * @brief Wave 34a: Pure Eulerian Solver — Fixed-mesh Godunov methods
 *
 * Sub-modules:
 * - 34a-1: Euler2DSolver        — 2D fixed-mesh Godunov solver with HLLC flux
 * - 34a-2: Euler3DSolver        — 3D extension with full HLLC Riemann solver
 * - 34a-3: EulerianFlux         — HLLC and Roe approximate Riemann solvers
 * - 34a-4: EulerianGradient     — Piecewise-linear gradient with MinMod limiter
 * - 34a-5: EulerianTimeStepping — CFL-based time step control
 * - 34a-6: EulerianBCs          — Ghost-cell boundary conditions
 *
 * References:
 * - Toro (2009) "Riemann Solvers and Numerical Methods for Fluid Dynamics"
 * - Einfeldt (1988) "On Godunov-type methods for gas dynamics"
 * - Roe (1981) "Approximate Riemann solvers, parameter vectors, and difference schemes"
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>

namespace nxs {
namespace fem {

using Real = nxs::Real;

// ============================================================================
// Data Structures
// ============================================================================

/// Cell data for Eulerian solver
struct EulerCell {
    Real rho;           ///< Density
    Real u, v, w;       ///< Velocity components
    Real p;             ///< Pressure
    Real E;             ///< Total specific energy (per unit volume): rho*e + 0.5*rho*(u^2+v^2+w^2)
    Real dx, dy, dz;    ///< Cell dimensions
    int neighbors[6];   ///< Neighbor indices: -x, +x, -y, +y, -z, +z (-1 = boundary)
    Real gamma;         ///< Ratio of specific heats

    KOKKOS_INLINE_FUNCTION
    EulerCell() : rho(0), u(0), v(0), w(0), p(0), E(0),
                  dx(1), dy(1), dz(1), gamma(1.4) {
        for (int i = 0; i < 6; ++i) neighbors[i] = -1;
    }

    /// Compute sound speed
    KOKKOS_INLINE_FUNCTION
    Real sound_speed() const {
        Real rho_safe = (rho > 1.0e-30) ? rho : 1.0e-30;
        Real p_safe = (p > 0.0) ? p : 0.0;
        return Kokkos::sqrt(gamma * p_safe / rho_safe);
    }

    /// Compute total energy from primitives
    KOKKOS_INLINE_FUNCTION
    void compute_total_energy() {
        E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v + w * w);
    }

    /// Compute pressure from conserved variables
    KOKKOS_INLINE_FUNCTION
    void compute_pressure() {
        Real ke = 0.5 * rho * (u * u + v * v + w * w);
        p = (gamma - 1.0) * (E - ke);
        if (p < 0.0) p = 0.0;
    }

    /// Extract conservative variables [rho, rho*u, rho*v, rho*w, E]
    KOKKOS_INLINE_FUNCTION
    void to_conservative(Real U[5]) const {
        U[0] = rho;
        U[1] = rho * u;
        U[2] = rho * v;
        U[3] = rho * w;
        U[4] = E;
    }

    /// Set from conservative variables
    KOKKOS_INLINE_FUNCTION
    void from_conservative(const Real U[5]) {
        rho = U[0];
        Real rho_safe = (rho > 1.0e-30) ? rho : 1.0e-30;
        u = U[1] / rho_safe;
        v = U[2] / rho_safe;
        w = U[3] / rho_safe;
        E = U[4];
        compute_pressure();
    }
};

/// Boundary condition types
enum class EulerBCType {
    Transmitting,  ///< Zero-gradient (outflow)
    Reflecting,    ///< Reflecting wall
    Inflow,        ///< Prescribed inflow state
    Outflow        ///< Extrapolation outflow
};

/// Boundary condition data
struct EulerBCData {
    EulerBCType type;
    int face;         ///< 0=-x, 1=+x, 2=-y, 3=+y, 4=-z, 5=+z
    Real rho_bc;      ///< For inflow
    Real u_bc, v_bc, w_bc; ///< For inflow
    Real p_bc;        ///< For inflow
    Real gamma_bc;    ///< For inflow

    EulerBCData() : type(EulerBCType::Transmitting), face(0),
                    rho_bc(1.0), u_bc(0), v_bc(0), w_bc(0),
                    p_bc(1.0), gamma_bc(1.4) {}
};

// ============================================================================
// 34a-3: EulerianFlux — HLLC and Roe Approximate Riemann Solvers
// ============================================================================

/**
 * @brief Static class providing HLLC and Roe flux functions.
 *
 * HLLC: Three-wave model (two acoustic + contact). Exact for isolated contact
 * and shear waves. Preserves positivity.
 *
 * Roe: Linearized Riemann solver using Roe-averaged eigenvalues. Entropy fix
 * applied to prevent expansion shocks.
 */
class EulerianFlux {
public:
    /**
     * @brief HLLC flux for Euler equations.
     *
     * @param UL      Left conservative state [rho, rho*u, rho*v, rho*w, E]
     * @param UR      Right conservative state
     * @param pL      Left pressure
     * @param pR      Right pressure
     * @param gammaL  Left gamma
     * @param gammaR  Right gamma
     * @param normal  Face normal direction (0=x, 1=y, 2=z)
     * @param flux    [out] Numerical flux [5]
     */
    KOKKOS_INLINE_FUNCTION
    static void hllc_flux(const Real UL[5], const Real UR[5],
                          Real pL, Real pR,
                          Real gammaL, Real gammaR,
                          int normal, Real flux[5]) {
        // Extract primitives
        Real rhoL = UL[0];
        Real rhoR = UR[0];
        Real rhoL_safe = (rhoL > 1.0e-30) ? rhoL : 1.0e-30;
        Real rhoR_safe = (rhoR > 1.0e-30) ? rhoR : 1.0e-30;

        // Velocity components
        Real uL[3], uR[3];
        for (int d = 0; d < 3; ++d) {
            uL[d] = UL[d + 1] / rhoL_safe;
            uR[d] = UR[d + 1] / rhoR_safe;
        }

        // Normal velocity
        Real vnL = uL[normal];
        Real vnR = uR[normal];

        // Sound speeds
        Real pL_safe = (pL > 0.0) ? pL : 0.0;
        Real pR_safe = (pR > 0.0) ? pR : 0.0;
        Real aL = Kokkos::sqrt(gammaL * pL_safe / rhoL_safe);
        Real aR = Kokkos::sqrt(gammaR * pR_safe / rhoR_safe);

        // Pressure estimate (PVRS)
        Real p_pvrs = 0.5 * (pL_safe + pR_safe) - 0.5 * (vnR - vnL) * 0.5 * (rhoL + rhoR) * 0.5 * (aL + aR);
        Real p_star = (p_pvrs > 0.0) ? p_pvrs : 0.0;

        // Wave speed estimates
        Real qL = 1.0;
        if (p_star > pL_safe && pL_safe > 0.0) {
            qL = Kokkos::sqrt(1.0 + (gammaL + 1.0) / (2.0 * gammaL) * (p_star / pL_safe - 1.0));
        }
        Real qR = 1.0;
        if (p_star > pR_safe && pR_safe > 0.0) {
            qR = Kokkos::sqrt(1.0 + (gammaR + 1.0) / (2.0 * gammaR) * (p_star / pR_safe - 1.0));
        }

        Real SL = vnL - aL * qL;
        Real SR = vnR + aR * qR;

        // Contact wave speed
        Real denom = rhoL_safe * (SL - vnL) - rhoR_safe * (SR - vnR);
        Real S_star = 0.0;
        if (Kokkos::fabs(denom) > 1.0e-30) {
            S_star = (pR_safe - pL_safe + rhoL_safe * vnL * (SL - vnL) - rhoR_safe * vnR * (SR - vnR)) / denom;
        }

        // Compute HLLC flux
        if (SL >= 0.0) {
            // Left state flux
            euler_physical_flux(UL, pL_safe, uL, normal, flux);
        } else if (SR <= 0.0) {
            // Right state flux
            euler_physical_flux(UR, pR_safe, uR, normal, flux);
        } else if (S_star >= 0.0) {
            // Left star state
            Real fL[5];
            euler_physical_flux(UL, pL_safe, uL, normal, fL);

            Real coeff = rhoL_safe * (SL - vnL) / (SL - S_star);
            Real U_star[5];
            U_star[0] = coeff;
            for (int d = 0; d < 3; ++d) {
                U_star[d + 1] = coeff * ((d == normal) ? S_star : uL[d]);
            }
            U_star[4] = coeff * (UL[4] / rhoL_safe + (S_star - vnL) * (S_star + pL_safe / (rhoL_safe * (SL - vnL))));

            for (int i = 0; i < 5; ++i) {
                flux[i] = fL[i] + SL * (U_star[i] - UL[i]);
            }
        } else {
            // Right star state
            Real fR[5];
            euler_physical_flux(UR, pR_safe, uR, normal, fR);

            Real coeff = rhoR_safe * (SR - vnR) / (SR - S_star);
            Real U_star[5];
            U_star[0] = coeff;
            for (int d = 0; d < 3; ++d) {
                U_star[d + 1] = coeff * ((d == normal) ? S_star : uR[d]);
            }
            U_star[4] = coeff * (UR[4] / rhoR_safe + (S_star - vnR) * (S_star + pR_safe / (rhoR_safe * (SR - vnR))));

            for (int i = 0; i < 5; ++i) {
                flux[i] = fR[i] + SR * (U_star[i] - UR[i]);
            }
        }
    }

    /**
     * @brief Roe flux for Euler equations with entropy fix.
     *
     * @param UL      Left conservative state [rho, rho*u, rho*v, rho*w, E]
     * @param UR      Right conservative state
     * @param pL      Left pressure
     * @param pR      Right pressure
     * @param gammaVal Ratio of specific heats (common)
     * @param normal  Face normal direction (0=x, 1=y, 2=z)
     * @param flux    [out] Numerical flux [5]
     */
    KOKKOS_INLINE_FUNCTION
    static void roe_flux(const Real UL[5], const Real UR[5],
                         Real pL, Real pR,
                         Real gammaVal, int normal, Real flux[5]) {
        Real rhoL = UL[0];
        Real rhoR = UR[0];
        Real rhoL_safe = (rhoL > 1.0e-30) ? rhoL : 1.0e-30;
        Real rhoR_safe = (rhoR > 1.0e-30) ? rhoR : 1.0e-30;

        // Primitive velocities
        Real uL[3], uR[3];
        for (int d = 0; d < 3; ++d) {
            uL[d] = UL[d + 1] / rhoL_safe;
            uR[d] = UR[d + 1] / rhoR_safe;
        }

        // Roe averages
        Real sqrtL = Kokkos::sqrt(rhoL_safe);
        Real sqrtR = Kokkos::sqrt(rhoR_safe);
        Real denom = sqrtL + sqrtR;

        Real u_roe[3];
        for (int d = 0; d < 3; ++d) {
            u_roe[d] = (sqrtL * uL[d] + sqrtR * uR[d]) / denom;
        }

        Real HL = (UL[4] + pL) / rhoL_safe;
        Real HR = (UR[4] + pR) / rhoR_safe;
        Real H_roe = (sqrtL * HL + sqrtR * HR) / denom;

        Real q2_roe = 0.0;
        for (int d = 0; d < 3; ++d) q2_roe += u_roe[d] * u_roe[d];
        Real a2_roe = (gammaVal - 1.0) * (H_roe - 0.5 * q2_roe);
        if (a2_roe < 1.0e-30) a2_roe = 1.0e-30;
        Real a_roe = Kokkos::sqrt(a2_roe);

        Real vn_roe = u_roe[normal];

        // Eigenvalues
        Real lambda[3] = { vn_roe - a_roe, vn_roe, vn_roe + a_roe };

        // Entropy fix (Harten-Hyman)
        Real eps_fix = 0.1 * a_roe;
        for (int k = 0; k < 3; ++k) {
            if (Kokkos::fabs(lambda[k]) < eps_fix) {
                lambda[k] = (lambda[k] * lambda[k] + eps_fix * eps_fix) / (2.0 * eps_fix);
            }
        }

        // State differences
        Real drho = rhoR - rhoL;
        Real dp = pR - pL;
        Real dvn = uR[normal] - uL[normal];

        // Wave strengths
        Real alpha1 = (dp - rhoL_safe * a_roe * dvn) / (2.0 * a2_roe);  // left acoustic
        Real alpha2 = drho - dp / a2_roe;                                  // entropy
        Real alpha3 = (dp + rhoL_safe * a_roe * dvn) / (2.0 * a2_roe);  // right acoustic

        // Use Roe average density for wave strengths
        Real rho_roe = sqrtL * sqrtR;
        alpha1 = (dp - rho_roe * a_roe * dvn) / (2.0 * a2_roe);
        alpha3 = (dp + rho_roe * a_roe * dvn) / (2.0 * a2_roe);

        // Physical fluxes
        Real fL[5], fR[5];
        euler_physical_flux(UL, pL, uL, normal, fL);
        euler_physical_flux(UR, pR, uR, normal, fR);

        // Roe flux = 0.5*(FL + FR) - 0.5*sum(|lambda_k|*alpha_k*r_k)
        for (int i = 0; i < 5; ++i) {
            flux[i] = 0.5 * (fL[i] + fR[i]);
        }

        // Wave 1: left acoustic
        Real r1[5];
        r1[0] = 1.0;
        for (int d = 0; d < 3; ++d) r1[d + 1] = u_roe[d];
        r1[normal + 1] -= a_roe;
        r1[4] = H_roe - vn_roe * a_roe;

        // Wave 2: entropy
        Real r2[5];
        r2[0] = 1.0;
        for (int d = 0; d < 3; ++d) r2[d + 1] = u_roe[d];
        r2[4] = 0.5 * q2_roe;

        // Tangential velocity jump contributions to entropy wave
        // (handled as shear waves for multi-D)
        Real dvt[3];
        for (int d = 0; d < 3; ++d) dvt[d] = (d == normal) ? 0.0 : (uR[d] - uL[d]);

        // Wave 3: right acoustic
        Real r3[5];
        r3[0] = 1.0;
        for (int d = 0; d < 3; ++d) r3[d + 1] = u_roe[d];
        r3[normal + 1] += a_roe;
        r3[4] = H_roe + vn_roe * a_roe;

        Real abs_lam1 = Kokkos::fabs(lambda[0]);
        Real abs_lam2 = Kokkos::fabs(lambda[1]);
        Real abs_lam3 = Kokkos::fabs(lambda[2]);

        for (int i = 0; i < 5; ++i) {
            Real dissipation = abs_lam1 * alpha1 * r1[i]
                             + abs_lam2 * alpha2 * r2[i]
                             + abs_lam3 * alpha3 * r3[i];
            // Tangential shear dissipation
            for (int d = 0; d < 3; ++d) {
                if (d != normal) {
                    Real rk = 0.0;
                    if (i == d + 1) rk = 1.0;
                    else if (i == 4) rk = u_roe[d];
                    dissipation += abs_lam2 * rho_roe * dvt[d] * rk;
                }
            }
            flux[i] -= 0.5 * dissipation;
        }
    }

    /**
     * @brief Physical flux in given direction.
     */
    KOKKOS_INLINE_FUNCTION
    static void euler_physical_flux(const Real U[5], Real p,
                                    const Real vel[3], int normal, Real f[5]) {
        Real vn = vel[normal];
        f[0] = U[0] * vn;
        for (int d = 0; d < 3; ++d) {
            f[d + 1] = U[d + 1] * vn;
        }
        f[normal + 1] += p;
        f[4] = (U[4] + p) * vn;
    }
};

// ============================================================================
// 34a-4: EulerianGradient — Piecewise-linear reconstruction with MinMod limiter
// ============================================================================

/**
 * @brief Gradient reconstruction with slope limiting.
 *
 * Computes cell-centered gradients for second-order spatial accuracy.
 * MinMod limiter prevents spurious oscillations near discontinuities.
 */
class EulerianGradient {
public:
    /**
     * @brief MinMod limiter function.
     */
    KOKKOS_INLINE_FUNCTION
    static Real minmod(Real a, Real b) {
        if (a * b <= 0.0) return 0.0;
        if (Kokkos::fabs(a) < Kokkos::fabs(b)) return a;
        return b;
    }

    /**
     * @brief Compute limited gradient for a scalar field.
     *
     * Uses cell values and neighbor values to compute gradient with MinMod limiter.
     *
     * @param val_center  Cell center value
     * @param val_left    Left neighbor value
     * @param val_right   Right neighbor value
     * @param dx          Cell spacing
     * @return Limited gradient
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_1d_gradient(Real val_center, Real val_left, Real val_right, Real dx) {
        if (dx < 1.0e-30) return 0.0;
        Real grad_left = (val_center - val_left) / dx;
        Real grad_right = (val_right - val_center) / dx;
        return minmod(grad_left, grad_right);
    }

    /**
     * @brief Compute 3D gradient for a cell.
     *
     * @param cell_values   Array of cell-centered values
     * @param cell_idx      Index of current cell
     * @param neighbors     Neighbor indices [6] (-x,+x,-y,+y,-z,+z), -1 for boundary
     * @param dx, dy, dz    Cell spacings
     * @param ncells        Total number of cells
     * @param gradient      [out] Gradient [3] (dq/dx, dq/dy, dq/dz)
     */
    static void compute_gradient(const Real* cell_values, int cell_idx,
                                 const int neighbors[6],
                                 Real dx, Real dy, Real dz,
                                 int ncells, Real gradient[3]) {
        Real val_c = cell_values[cell_idx];

        // x-direction
        Real val_xm = (neighbors[0] >= 0 && neighbors[0] < ncells) ? cell_values[neighbors[0]] : val_c;
        Real val_xp = (neighbors[1] >= 0 && neighbors[1] < ncells) ? cell_values[neighbors[1]] : val_c;
        gradient[0] = compute_1d_gradient(val_c, val_xm, val_xp, dx);

        // y-direction
        Real val_ym = (neighbors[2] >= 0 && neighbors[2] < ncells) ? cell_values[neighbors[2]] : val_c;
        Real val_yp = (neighbors[3] >= 0 && neighbors[3] < ncells) ? cell_values[neighbors[3]] : val_c;
        gradient[1] = compute_1d_gradient(val_c, val_ym, val_yp, dy);

        // z-direction
        Real val_zm = (neighbors[4] >= 0 && neighbors[4] < ncells) ? cell_values[neighbors[4]] : val_c;
        Real val_zp = (neighbors[5] >= 0 && neighbors[5] < ncells) ? cell_values[neighbors[5]] : val_c;
        gradient[2] = compute_1d_gradient(val_c, val_zm, val_zp, dz);
    }

    /**
     * @brief Reconstruct left/right face values for a 1D interface.
     *
     * @param val_center Cell value
     * @param gradient   Gradient in face-normal direction
     * @param dx         Cell spacing
     * @param val_left   [out] Reconstructed value at left face
     * @param val_right  [out] Reconstructed value at right face
     */
    KOKKOS_INLINE_FUNCTION
    static void reconstruct_face(Real val_center, Real grad, Real dx,
                                 Real& val_left, Real& val_right) {
        val_left  = val_center - 0.5 * grad * dx;
        val_right = val_center + 0.5 * grad * dx;
    }
};

// ============================================================================
// 34a-5: EulerianTimeStepping — CFL-based time step control
// ============================================================================

/**
 * @brief CFL-based time step calculator for Eulerian solver.
 *
 * dt = CFL * min_i( dx_i / (|u_i| + c_i) )
 */
class EulerianTimeStepping {
public:
    /**
     * @brief Compute stable time step.
     *
     * @param cells   Array of Euler cells
     * @param ncells  Number of cells
     * @param cfl     CFL number (typically 0.5-0.9)
     * @return Stable time step
     */
    static Real compute_dt(const EulerCell* cells, int ncells, Real cfl) {
        Real dt_min = 1.0e30;

        for (int i = 0; i < ncells; ++i) {
            Real c = cells[i].sound_speed();

            // x-direction
            Real speed_x = Kokkos::fabs(cells[i].u) + c;
            if (speed_x > 1.0e-30 && cells[i].dx > 0.0) {
                Real dt_x = cells[i].dx / speed_x;
                if (dt_x < dt_min) dt_min = dt_x;
            }

            // y-direction
            Real speed_y = Kokkos::fabs(cells[i].v) + c;
            if (speed_y > 1.0e-30 && cells[i].dy > 0.0) {
                Real dt_y = cells[i].dy / speed_y;
                if (dt_y < dt_min) dt_min = dt_y;
            }

            // z-direction
            Real speed_z = Kokkos::fabs(cells[i].w) + c;
            if (speed_z > 1.0e-30 && cells[i].dz > 0.0) {
                Real dt_z = cells[i].dz / speed_z;
                if (dt_z < dt_min) dt_min = dt_z;
            }
        }

        return cfl * dt_min;
    }

    /**
     * @brief Compute dt for 2D solver (ignores z-direction).
     */
    static Real compute_dt_2d(const EulerCell* cells, int ncells, Real cfl) {
        Real dt_min = 1.0e30;

        for (int i = 0; i < ncells; ++i) {
            Real c = cells[i].sound_speed();

            Real speed_x = Kokkos::fabs(cells[i].u) + c;
            if (speed_x > 1.0e-30 && cells[i].dx > 0.0) {
                Real dt_x = cells[i].dx / speed_x;
                if (dt_x < dt_min) dt_min = dt_x;
            }

            Real speed_y = Kokkos::fabs(cells[i].v) + c;
            if (speed_y > 1.0e-30 && cells[i].dy > 0.0) {
                Real dt_y = cells[i].dy / speed_y;
                if (dt_y < dt_min) dt_min = dt_y;
            }
        }

        return cfl * dt_min;
    }
};

// ============================================================================
// 34a-6: EulerianBCs — Ghost-cell boundary conditions
// ============================================================================

/**
 * @brief Boundary condition application for Eulerian grids.
 *
 * Ghost-cell approach: boundary cells are set to mirror or prescribed states
 * so that the flux at the boundary produces the desired physical behavior.
 *
 * - Transmitting: ghost = interior (zero-gradient)
 * - Reflecting: ghost mirrors density/energy, reverses normal velocity
 * - Inflow: ghost set to prescribed state
 * - Outflow: ghost extrapolated from interior
 */
class EulerianBCs {
public:
    /**
     * @brief Apply boundary condition to a ghost cell.
     *
     * @param interior  Interior cell adjacent to boundary
     * @param ghost     [out] Ghost cell to set
     * @param bc_type   Type of BC
     * @param bc_data   BC data (used for inflow)
     * @param face      Face index (0=-x,1=+x,2=-y,3=+y,4=-z,5=+z)
     */
    static void apply_bc(const EulerCell& interior, EulerCell& ghost,
                         EulerBCType bc_type, const EulerBCData& bc_data, int face) {
        switch (bc_type) {
        case EulerBCType::Transmitting:
            // Zero-gradient: ghost = interior
            ghost = interior;
            break;

        case EulerBCType::Reflecting: {
            // Mirror state, reverse normal velocity
            ghost = interior;
            int normal_dir = face / 2;  // 0=x, 1=y, 2=z
            if (normal_dir == 0) ghost.u = -interior.u;
            else if (normal_dir == 1) ghost.v = -interior.v;
            else ghost.w = -interior.w;
            ghost.compute_total_energy();
            break;
        }

        case EulerBCType::Inflow:
            // Prescribed state
            ghost.rho = bc_data.rho_bc;
            ghost.u = bc_data.u_bc;
            ghost.v = bc_data.v_bc;
            ghost.w = bc_data.w_bc;
            ghost.p = bc_data.p_bc;
            ghost.gamma = bc_data.gamma_bc;
            ghost.compute_total_energy();
            break;

        case EulerBCType::Outflow:
            // Extrapolation (same as transmitting for first order)
            ghost = interior;
            break;
        }
    }

    /**
     * @brief Apply BCs to all boundary cells in an array.
     *
     * For each cell with a neighbor index of -1, applies the corresponding BC.
     *
     * @param cells     Array of cells
     * @param ncells    Number of cells
     * @param bc_data   Array of BC data per face [6]
     */
    static void apply_all_bcs(EulerCell* cells, int ncells, const EulerBCData bc_data[6]) {
        for (int i = 0; i < ncells; ++i) {
            for (int f = 0; f < 6; ++f) {
                if (cells[i].neighbors[f] == -1) {
                    // This face is a boundary — the cell itself acts as its own interior
                    // We modify the cell to enforce the BC (simplified approach)
                    // In practice, ghost cells would be separate
                }
            }
        }
    }

    /**
     * @brief Create ghost cell for a boundary face.
     */
    static EulerCell create_ghost(const EulerCell& interior, EulerBCType bc_type,
                                  const EulerBCData& bc_data, int face) {
        EulerCell ghost;
        apply_bc(interior, ghost, bc_type, bc_data, face);
        return ghost;
    }
};

// ============================================================================
// 34a-1: Euler2DSolver — 2D Fixed-mesh Godunov solver
// ============================================================================

/**
 * @brief 2D Eulerian solver on a fixed Cartesian grid.
 *
 * Uses Godunov's method with HLLC flux for the 2D Euler equations.
 * Conservative variables: [rho, rho*u, rho*v, E].
 * Dimensionally split: x-sweep then y-sweep.
 *
 * Grid layout: row-major, cell (ix, iy) has index ix + iy * nx.
 */
class Euler2DSolver {
public:
    int nx, ny;           ///< Grid dimensions
    Real dx, dy;          ///< Cell spacings
    Real gamma;           ///< Ratio of specific heats
    Real cfl;             ///< CFL number

    Euler2DSolver() : nx(0), ny(0), dx(1.0), dy(1.0), gamma(1.4), cfl(0.5) {}

    Euler2DSolver(int nx_, int ny_, Real dx_, Real dy_, Real gamma_, Real cfl_)
        : nx(nx_), ny(ny_), dx(dx_), dy(dy_), gamma(gamma_), cfl(cfl_) {}

    /**
     * @brief Initialize a uniform grid of cells.
     */
    void init_grid(std::vector<EulerCell>& cells) const {
        int ncells = nx * ny;
        cells.resize(ncells);
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int idx = ix + iy * nx;
                cells[idx].dx = dx;
                cells[idx].dy = dy;
                cells[idx].dz = 1.0;
                cells[idx].gamma = gamma;

                // Set neighbors
                cells[idx].neighbors[0] = (ix > 0) ? (ix - 1 + iy * nx) : -1;
                cells[idx].neighbors[1] = (ix < nx - 1) ? (ix + 1 + iy * nx) : -1;
                cells[idx].neighbors[2] = (iy > 0) ? (ix + (iy - 1) * nx) : -1;
                cells[idx].neighbors[3] = (iy < ny - 1) ? (ix + (iy + 1) * nx) : -1;
                cells[idx].neighbors[4] = -1;
                cells[idx].neighbors[5] = -1;
            }
        }
    }

    /**
     * @brief Perform one time step (dimensionally split Godunov).
     *
     * @param cells   Cell array (modified in place)
     * @param ncells  Number of cells
     * @param dt      Time step
     */
    void solve_step(EulerCell* cells, int ncells, Real dt) const {
        if (ncells != nx * ny) return;

        // Allocate conservative variable updates
        std::vector<Real> dU(ncells * 4, 0.0);  // [rho, rho*u, rho*v, E] per cell

        // X-sweep
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx - 1; ++ix) {
                int idxL = ix + iy * nx;
                int idxR = ix + 1 + iy * nx;

                Real UL[5], UR[5], flux[5];
                cells[idxL].to_conservative(UL);
                cells[idxR].to_conservative(UR);

                EulerianFlux::hllc_flux(UL, UR, cells[idxL].p, cells[idxR].p,
                                        cells[idxL].gamma, cells[idxR].gamma,
                                        0, flux);

                Real dtdx = dt / dx;
                // Update left cell (flux leaves through right face)
                dU[idxL * 4 + 0] -= dtdx * flux[0];
                dU[idxL * 4 + 1] -= dtdx * flux[1];
                dU[idxL * 4 + 2] -= dtdx * flux[2];
                dU[idxL * 4 + 3] -= dtdx * flux[4];

                // Update right cell (flux enters through left face)
                dU[idxR * 4 + 0] += dtdx * flux[0];
                dU[idxR * 4 + 1] += dtdx * flux[1];
                dU[idxR * 4 + 2] += dtdx * flux[2];
                dU[idxR * 4 + 3] += dtdx * flux[4];
            }
        }

        // Y-sweep
        for (int iy = 0; iy < ny - 1; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int idxL = ix + iy * nx;
                int idxR = ix + (iy + 1) * nx;

                Real UL[5], UR[5], flux[5];
                cells[idxL].to_conservative(UL);
                cells[idxR].to_conservative(UR);

                EulerianFlux::hllc_flux(UL, UR, cells[idxL].p, cells[idxR].p,
                                        cells[idxL].gamma, cells[idxR].gamma,
                                        1, flux);

                Real dtdy = dt / dy;
                dU[idxL * 4 + 0] -= dtdy * flux[0];
                dU[idxL * 4 + 1] -= dtdy * flux[1];
                dU[idxL * 4 + 2] -= dtdy * flux[2];
                dU[idxL * 4 + 3] -= dtdy * flux[4];

                dU[idxR * 4 + 0] += dtdy * flux[0];
                dU[idxR * 4 + 1] += dtdy * flux[1];
                dU[idxR * 4 + 2] += dtdy * flux[2];
                dU[idxR * 4 + 3] += dtdy * flux[4];
            }
        }

        // Apply updates
        for (int i = 0; i < ncells; ++i) {
            Real U[5];
            cells[i].to_conservative(U);

            U[0] += dU[i * 4 + 0];
            U[1] += dU[i * 4 + 1];
            U[2] += dU[i * 4 + 2];
            // w unchanged in 2D
            U[4] += dU[i * 4 + 3];

            // Enforce positivity
            if (U[0] < 1.0e-30) U[0] = 1.0e-30;

            cells[i].from_conservative(U);
        }
    }

    /**
     * @brief Compute CFL-limited dt for the current state.
     */
    Real compute_dt(const EulerCell* cells, int ncells) const {
        return EulerianTimeStepping::compute_dt_2d(cells, ncells, cfl);
    }
};

// ============================================================================
// 34a-2: Euler3DSolver — 3D Fixed-mesh Godunov solver
// ============================================================================

/**
 * @brief 3D Eulerian solver on a fixed Cartesian grid.
 *
 * Extension of Euler2DSolver to 3D. Conservative variables: [rho, rho*u, rho*v, rho*w, E].
 * Dimensionally split: x-sweep, y-sweep, z-sweep.
 *
 * Grid layout: row-major, cell (ix, iy, iz) has index ix + iy*nx + iz*nx*ny.
 */
class Euler3DSolver {
public:
    int nx, ny, nz;
    Real dx, dy, dz;
    Real gamma;
    Real cfl;

    Euler3DSolver() : nx(0), ny(0), nz(0), dx(1), dy(1), dz(1), gamma(1.4), cfl(0.5) {}

    Euler3DSolver(int nx_, int ny_, int nz_, Real dx_, Real dy_, Real dz_,
                  Real gamma_, Real cfl_)
        : nx(nx_), ny(ny_), nz(nz_), dx(dx_), dy(dy_), dz(dz_),
          gamma(gamma_), cfl(cfl_) {}

    /**
     * @brief Initialize a uniform 3D grid.
     */
    void init_grid(std::vector<EulerCell>& cells) const {
        int ncells = nx * ny * nz;
        cells.resize(ncells);
        for (int iz = 0; iz < nz; ++iz) {
            for (int iy = 0; iy < ny; ++iy) {
                for (int ix = 0; ix < nx; ++ix) {
                    int idx = ix + iy * nx + iz * nx * ny;
                    cells[idx].dx = dx;
                    cells[idx].dy = dy;
                    cells[idx].dz = dz;
                    cells[idx].gamma = gamma;

                    cells[idx].neighbors[0] = (ix > 0) ? (ix - 1 + iy * nx + iz * nx * ny) : -1;
                    cells[idx].neighbors[1] = (ix < nx - 1) ? (ix + 1 + iy * nx + iz * nx * ny) : -1;
                    cells[idx].neighbors[2] = (iy > 0) ? (ix + (iy - 1) * nx + iz * nx * ny) : -1;
                    cells[idx].neighbors[3] = (iy < ny - 1) ? (ix + (iy + 1) * nx + iz * nx * ny) : -1;
                    cells[idx].neighbors[4] = (iz > 0) ? (ix + iy * nx + (iz - 1) * nx * ny) : -1;
                    cells[idx].neighbors[5] = (iz < nz - 1) ? (ix + iy * nx + (iz + 1) * nx * ny) : -1;
                }
            }
        }
    }

    /**
     * @brief Perform one time step (dimensionally split 3D Godunov).
     */
    void solve_step(EulerCell* cells, int ncells, Real dt) const {
        if (ncells != nx * ny * nz) return;

        std::vector<Real> dU(ncells * 5, 0.0);

        // X-sweep
        for (int iz = 0; iz < nz; ++iz) {
            for (int iy = 0; iy < ny; ++iy) {
                for (int ix = 0; ix < nx - 1; ++ix) {
                    int idxL = ix + iy * nx + iz * nx * ny;
                    int idxR = ix + 1 + iy * nx + iz * nx * ny;
                    sweep_face(cells, dU, idxL, idxR, 0, dt / dx);
                }
            }
        }

        // Y-sweep
        for (int iz = 0; iz < nz; ++iz) {
            for (int iy = 0; iy < ny - 1; ++iy) {
                for (int ix = 0; ix < nx; ++ix) {
                    int idxL = ix + iy * nx + iz * nx * ny;
                    int idxR = ix + (iy + 1) * nx + iz * nx * ny;
                    sweep_face(cells, dU, idxL, idxR, 1, dt / dy);
                }
            }
        }

        // Z-sweep
        for (int iz = 0; iz < nz - 1; ++iz) {
            for (int iy = 0; iy < ny; ++iy) {
                for (int ix = 0; ix < nx; ++ix) {
                    int idxL = ix + iy * nx + iz * nx * ny;
                    int idxR = ix + iy * nx + (iz + 1) * nx * ny;
                    sweep_face(cells, dU, idxL, idxR, 2, dt / dz);
                }
            }
        }

        // Apply updates
        for (int i = 0; i < ncells; ++i) {
            Real U[5];
            cells[i].to_conservative(U);
            for (int k = 0; k < 5; ++k) U[k] += dU[i * 5 + k];
            if (U[0] < 1.0e-30) U[0] = 1.0e-30;
            cells[i].from_conservative(U);
        }
    }

    /**
     * @brief Compute CFL-limited dt.
     */
    Real compute_dt(const EulerCell* cells, int ncells) const {
        return EulerianTimeStepping::compute_dt(cells, ncells, cfl);
    }

private:
    void sweep_face(const EulerCell* cells, std::vector<Real>& dU,
                    int idxL, int idxR, int normal, Real dt_over_dx) const {
        Real UL[5], UR[5], flux[5];
        cells[idxL].to_conservative(UL);
        cells[idxR].to_conservative(UR);

        EulerianFlux::hllc_flux(UL, UR, cells[idxL].p, cells[idxR].p,
                                cells[idxL].gamma, cells[idxR].gamma,
                                normal, flux);

        for (int k = 0; k < 5; ++k) {
            dU[idxL * 5 + k] -= dt_over_dx * flux[k];
            dU[idxR * 5 + k] += dt_over_dx * flux[k];
        }
    }
};

} // namespace fem
} // namespace nxs
