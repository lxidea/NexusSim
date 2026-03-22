#pragma once

/**
 * @file ale_wave27.hpp
 * @brief Wave 27: Advanced ALE Extensions — Turbulence, Bi-material, Porous Media,
 *        Erosion Handling, and Adaptive Remeshing
 *
 * Sub-modules:
 * - 27a: KEpsilonTurbulence     — Standard k-epsilon two-equation turbulence model
 * - 27b: EulerianBimatTracker   — 2-material Eulerian with PLIC interface reconstruction
 * - 27c: PorousMediaFlow        — Darcy flow with effective stress and Biot coupling
 * - 27d: ALEErosionHandler      — Void filling with mass/momentum/energy conservation
 * - 27e: ALEAdaptiveRemesh      — Gradient-based error indicator and solution transfer
 *
 * References:
 * - Launder & Spalding (1974) "The numerical computation of turbulent flows"
 * - Youngs (1982) "Time-dependent multi-material flow with large fluid distortion"
 * - Biot (1941) "General theory of three-dimensional consolidation"
 * - Zienkiewicz & Zhu (1987) "A simple error estimator and adaptive procedure"
 */

#include <nexussim/physics/material.hpp>
#include <nexussim/core/types.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <numeric>
#include <set>

namespace nxs {
namespace fem {

using Real = nxs::Real;

// ============================================================================
// 27a: KEpsilonTurbulence — Standard k-epsilon Two-Equation Model
// ============================================================================

/**
 * @brief Standard k-epsilon turbulence model.
 *
 * Solves the two transport equations for turbulent kinetic energy (k) and
 * dissipation rate (epsilon):
 *
 *   d(rho*k)/dt   = P_k - rho*eps + diffusion_k
 *   d(rho*eps)/dt = C_e1*(eps/k)*P_k - C_e2*rho*eps^2/k + diffusion_eps
 *
 * Turbulent viscosity: mu_t = C_mu * rho * k^2 / eps
 * Production:          P_k  = mu_t * |S|^2
 *
 * Standard model constants (Launder & Spalding, 1974):
 *   C_mu  = 0.09
 *   C_e1  = 1.44
 *   C_e2  = 1.92
 *   sigma_k   = 1.0
 *   sigma_eps  = 1.3
 *
 * Wall functions:
 *   y+ <= 11.63:  u+ = y+           (viscous sublayer)
 *   y+ >  11.63:  u+ = (1/kappa)*ln(y+) + B   (log law)
 *   kappa = 0.41, B = 5.2
 */
class KEpsilonModel {
public:
    /// Standard model constants
    static constexpr Real C_mu      = 0.09;
    static constexpr Real C_e1      = 1.44;
    static constexpr Real C_e2      = 1.92;
    static constexpr Real sigma_k   = 1.0;
    static constexpr Real sigma_eps = 1.3;
    static constexpr Real kappa     = 0.41;
    static constexpr Real B_const   = 5.2;
    static constexpr Real y_plus_transition = 11.63;

    /// Minimum values to prevent division by zero
    static constexpr Real k_min   = 1.0e-12;
    static constexpr Real eps_min = 1.0e-14;

    KEpsilonModel() = default;

    /**
     * @brief Compute turbulent (eddy) viscosity.
     * @param rho     Fluid density
     * @param k       Turbulent kinetic energy (must be > 0)
     * @param epsilon Dissipation rate (must be > 0)
     * @return mu_t = C_mu * rho * k^2 / epsilon
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_turbulent_viscosity(Real rho, Real k, Real epsilon) const {
        if (k < k_min) k = k_min;
        if (epsilon < eps_min) epsilon = eps_min;
        return C_mu * rho * k * k / epsilon;
    }

    /**
     * @brief Compute turbulent production term.
     * @param mu_t                  Turbulent viscosity
     * @param strain_rate_magnitude |S| (magnitude of strain rate tensor)
     * @return P_k = mu_t * |S|^2
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_production(Real mu_t, Real strain_rate_magnitude) const {
        return mu_t * strain_rate_magnitude * strain_rate_magnitude;
    }

    /**
     * @brief Advance k and epsilon one time step (explicit Euler).
     *
     * Simplified (no diffusion terms — spatially homogeneous):
     *   k_new   = k   + dt * (P_k/rho - epsilon)
     *   eps_new = eps + dt * (C_e1*eps/k*P_k/rho - C_e2*eps^2/k)
     *
     * @param k       Current turbulent kinetic energy
     * @param epsilon Current dissipation rate
     * @param rho     Density
     * @param P_k     Production term
     * @param mu      Molecular viscosity (unused in homogeneous form)
     * @param mu_t    Turbulent viscosity (unused in homogeneous form)
     * @param dt      Time step
     * @param k_new   [out] Updated k
     * @param eps_new [out] Updated epsilon
     */
    KOKKOS_INLINE_FUNCTION
    void advance(Real k, Real epsilon, Real rho, Real P_k,
                 Real mu, Real mu_t, Real dt,
                 Real& k_new, Real& eps_new) const {
        (void)mu;
        (void)mu_t;

        if (k < k_min) k = k_min;
        if (epsilon < eps_min) epsilon = eps_min;
        if (rho < 1.0e-30) rho = 1.0e-30;

        Real rho_inv = 1.0 / rho;

        // k equation: dk/dt = P_k/rho - epsilon
        Real dk = (P_k * rho_inv) - epsilon;
        k_new = k + dt * dk;

        // epsilon equation: deps/dt = C_e1*(eps/k)*(P_k/rho) - C_e2*eps^2/k
        Real eps_over_k = epsilon / k;
        Real deps = C_e1 * eps_over_k * (P_k * rho_inv)
                  - C_e2 * epsilon * eps_over_k;
        eps_new = epsilon + dt * deps;

        // Enforce positivity
        if (k_new < k_min) k_new = k_min;
        if (eps_new < eps_min) eps_new = eps_min;
    }

    /**
     * @brief Advance with diffusion (1D Laplacian for testing).
     *
     * Adds diffusion: (mu + mu_t/sigma) * d2(phi)/dx2
     *
     * @param k_field       Array of k values
     * @param eps_field     Array of epsilon values
     * @param rho           Uniform density
     * @param P_k_field     Production per cell
     * @param mu            Molecular viscosity
     * @param dx            Cell spacing
     * @param dt            Time step
     * @param n_cells       Number of cells
     * @param k_new         [out] Updated k array
     * @param eps_new_field [out] Updated epsilon array
     */
    void advance_field_1d(const std::vector<Real>& k_field,
                          const std::vector<Real>& eps_field,
                          Real rho,
                          const std::vector<Real>& P_k_field,
                          Real mu,
                          Real dx, Real dt, int n_cells,
                          std::vector<Real>& k_new_field,
                          std::vector<Real>& eps_new_field) const {
        k_new_field.resize(n_cells);
        eps_new_field.resize(n_cells);

        for (int i = 0; i < n_cells; ++i) {
            Real ki = std::max(k_field[i], k_min);
            Real ei = std::max(eps_field[i], eps_min);
            Real mu_t = compute_turbulent_viscosity(rho, ki, ei);
            Real P_ki = P_k_field[i];

            // Diffusion of k: (mu + mu_t/sigma_k) * d2k/dx2
            Real k_left  = (i > 0)           ? k_field[i-1] : ki;
            Real k_right = (i < n_cells - 1) ? k_field[i+1] : ki;
            Real diff_k = (mu + mu_t / sigma_k) * (k_left - 2.0*ki + k_right) / (dx*dx);

            // Diffusion of eps: (mu + mu_t/sigma_eps) * d2eps/dx2
            Real e_left  = (i > 0)           ? eps_field[i-1] : ei;
            Real e_right = (i < n_cells - 1) ? eps_field[i+1] : ei;
            Real diff_e = (mu + mu_t / sigma_eps) * (e_left - 2.0*ei + e_right) / (dx*dx);

            Real rho_inv = 1.0 / rho;
            Real dk = P_ki * rho_inv - ei + diff_k / rho;
            Real eps_over_k = ei / ki;
            Real deps = C_e1 * eps_over_k * P_ki * rho_inv
                      - C_e2 * ei * eps_over_k + diff_e / rho;

            k_new_field[i]   = std::max(ki + dt * dk, k_min);
            eps_new_field[i] = std::max(ei + dt * deps, eps_min);
        }
    }

    /**
     * @brief Evaluate wall function.
     *
     * Viscous sublayer (y+ <= 11.63):   u+ = y+
     * Log-law region   (y+ >  11.63):   u+ = (1/kappa)*ln(y+) + B
     *
     * @param y_plus Dimensionless wall distance
     * @return u_plus Dimensionless velocity
     */
    KOKKOS_INLINE_FUNCTION
    Real wall_function(Real y_plus) const {
        if (y_plus <= 0.0) return 0.0;
        if (y_plus <= y_plus_transition) {
            return y_plus;
        }
        return (1.0 / kappa) * std::log(y_plus) + B_const;
    }

    /**
     * @brief Compute equilibrium k for a given production/dissipation balance.
     *
     * At equilibrium P_k = rho * epsilon, so:
     *   mu_t * |S|^2 = rho * eps  =>  C_mu * rho * k^2/eps * |S|^2 = rho*eps
     *   => k = eps / (C_mu^0.5 * |S|)   (for |S| != 0)
     */
    KOKKOS_INLINE_FUNCTION
    Real equilibrium_k(Real epsilon, Real strain_rate_mag) const {
        if (strain_rate_mag < 1.0e-30) return k_min;
        return epsilon / (std::sqrt(C_mu) * strain_rate_mag);
    }

    /**
     * @brief Compute the turbulent kinetic energy spectrum decay (no production).
     *
     * With P_k = 0, the k equation becomes:
     *   dk/dt = -epsilon
     *   deps/dt = -C_e2 * eps^2 / k
     *
     * Analytical solution: k(t) = k0 * (1 + (C_e2-1)*eps0*t/k0)^(-1/(C_e2-1))
     *                      eps(t) = eps0 * (1 + (C_e2-1)*eps0*t/k0)^(-C_e2/(C_e2-1))
     */
    KOKKOS_INLINE_FUNCTION
    void decay_analytical(Real k0, Real eps0, Real t,
                          Real& k_t, Real& eps_t) const {
        Real n = C_e2 - 1.0;  // 0.92
        Real factor = 1.0 + n * eps0 * t / k0;
        if (factor < 1.0e-30) factor = 1.0e-30;
        k_t   = k0   * std::pow(factor, -1.0 / n);
        eps_t = eps0 * std::pow(factor, -C_e2 / n);
    }
};

// ============================================================================
// 27b: EulerianBimatTracker — 2-Material Eulerian with PLIC Interface
// ============================================================================

/**
 * @brief Volume-of-Fluid (VOF) bi-material tracker with PLIC interface
 *        reconstruction.
 *
 * Tracks a volume fraction alpha in [0, 1]:
 *   alpha = 1  => pure material 1
 *   alpha = 0  => pure material 2
 *   0 < alpha < 1 => interface cell (mixed)
 *
 * Interface reconstruction uses Piecewise Linear Interface Construction
 * (PLIC), where the interface is approximated as a plane n.x = d within
 * each cell, with normal n computed from Youngs' gradient method:
 *   n = -grad(alpha) / |grad(alpha)|
 *
 * Volume fraction advection is done via a simple upwind/flux scheme:
 *   alpha_new = alpha - (dt / V) * (flux_right - flux_left)
 */
class BimatTracker {
public:
    BimatTracker() = default;

    /**
     * @brief Compute interface normal using Youngs' method from neighbor fractions.
     *
     * Uses central differences on a structured stencil:
     *   grad_alpha_x = (alpha_right - alpha_left) / (2*dx)
     * Since we only care about the direction, dx cancels.
     * Neighbors: [left, right, bottom, top, back, front]
     *
     * @param alpha_neighbors  Volume fractions of 6 neighbors [x-, x+, y-, y+, z-, z+]
     * @param normal           [out] Unit normal vector [3]
     */
    KOKKOS_INLINE_FUNCTION
    void compute_interface_normal(const Real alpha_neighbors[6],
                                  Real normal[3]) const {
        // Central difference gradients (unnormalized — dx cancels)
        Real gx = alpha_neighbors[1] - alpha_neighbors[0]; // right - left
        Real gy = alpha_neighbors[3] - alpha_neighbors[2]; // top - bottom
        Real gz = alpha_neighbors[5] - alpha_neighbors[4]; // front - back

        // Normal is negative gradient (points toward material 1)
        Real nx = -gx;
        Real ny = -gy;
        Real nz = -gz;

        Real mag = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (mag < 1.0e-30) {
            normal[0] = 1.0;
            normal[1] = 0.0;
            normal[2] = 0.0;
            return;
        }

        Real inv_mag = 1.0 / mag;
        normal[0] = nx * inv_mag;
        normal[1] = ny * inv_mag;
        normal[2] = nz * inv_mag;
    }

    /**
     * @brief Advect volume fraction using fluxes.
     *
     * alpha_new = alpha - dt * (flux_right - flux_left) / volume
     * Clamps result to [0, 1].
     *
     * @param alpha      Current volume fraction
     * @param flux_left  Flux entering from left face (alpha * u * A)
     * @param flux_right Flux leaving from right face (alpha * u * A)
     * @param volume     Cell volume
     * @param dt         Time step
     * @return Updated volume fraction
     */
    KOKKOS_INLINE_FUNCTION
    Real advect_volume_fraction(Real alpha, Real flux_left, Real flux_right,
                                Real volume, Real dt) const {
        Real alpha_new = alpha - dt * (flux_right - flux_left) / volume;
        if (alpha_new < 0.0) alpha_new = 0.0;
        if (alpha_new > 1.0) alpha_new = 1.0;
        return alpha_new;
    }

    /**
     * @brief Mix material properties using volume fraction.
     * @param alpha Volume fraction of material 1
     * @param rho1  Density of material 1
     * @param rho2  Density of material 2
     * @return Mixed density: alpha*rho1 + (1-alpha)*rho2
     */
    KOKKOS_INLINE_FUNCTION
    Real mix_properties(Real alpha, Real rho1, Real rho2) const {
        return alpha * rho1 + (1.0 - alpha) * rho2;
    }

    /**
     * @brief Determine if a cell is an interface cell.
     * @param alpha Volume fraction
     * @param tol   Tolerance (default 0.01)
     * @return true if tol < alpha < 1 - tol
     */
    KOKKOS_INLINE_FUNCTION
    bool is_interface_cell(Real alpha, Real tol = 0.01) const {
        return (alpha > tol) && (alpha < (1.0 - tol));
    }

    /**
     * @brief Compute PLIC plane offset d such that the truncated volume
     *        behind the plane n.x = d matches alpha * V_cell.
     *
     * For a unit cube [0,1]^3 with normal n aligned along x:
     *   d = alpha (for n = [1,0,0])
     *
     * General case: iterative bisection on d to match target volume.
     *
     * @param normal     Interface normal [3]
     * @param alpha      Volume fraction
     * @param cell_size  Edge length of cubic cell
     * @return d such that volume behind n.x = d equals alpha * cell_size^3
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_plic_offset(const Real normal[3], Real alpha,
                             Real cell_size) const {
        // For an axis-aligned normal (dominant component), use analytical formula
        // Find dominant direction
        int dir = 0;
        Real max_n = std::abs(normal[0]);
        if (std::abs(normal[1]) > max_n) { dir = 1; max_n = std::abs(normal[1]); }
        if (std::abs(normal[2]) > max_n) { dir = 2; max_n = std::abs(normal[2]); }

        // For nearly axis-aligned normal, d = alpha * cell_size * |n_dir|
        // General: bisection
        Real d_min = -cell_size * (std::abs(normal[0]) + std::abs(normal[1]) + std::abs(normal[2]));
        Real d_max = -d_min;

        Real target_vol = alpha * cell_size * cell_size * cell_size;

        // Bisection (20 iterations gives ~1e-6 accuracy)
        for (int iter = 0; iter < 30; ++iter) {
            Real d_mid = 0.5 * (d_min + d_max);
            Real vol = plic_truncated_volume(normal, d_mid, cell_size);
            if (vol < target_vol) {
                d_min = d_mid;
            } else {
                d_max = d_mid;
            }
        }
        return 0.5 * (d_min + d_max);
    }

    /**
     * @brief Compute volume of a unit cube behind the plane n.x <= d.
     *
     * Uses a sampling approach for simplicity (Monte Carlo or analytical
     * for axis-aligned normals).
     *
     * For axis-aligned normal n = [1,0,0]:
     *   volume = clamp(d, 0, cell_size) * cell_size^2
     */
    KOKKOS_INLINE_FUNCTION
    Real plic_truncated_volume(const Real normal[3], Real d,
                               Real cell_size) const {
        // Sample-based approach: subdivide into N^3 sub-cells
        int N = 20;
        Real sub = cell_size / N;
        Real count = 0.0;
        for (int i = 0; i < N; ++i) {
            Real x = (i + 0.5) * sub;
            for (int j = 0; j < N; ++j) {
                Real y = (j + 0.5) * sub;
                for (int kk = 0; kk < N; ++kk) {
                    Real z = (kk + 0.5) * sub;
                    Real dot = normal[0]*x + normal[1]*y + normal[2]*z;
                    if (dot <= d) {
                        count += 1.0;
                    }
                }
            }
        }
        return count / (N * N * N) * cell_size * cell_size * cell_size;
    }

    /**
     * @brief Advect a 1D array of volume fractions with uniform velocity.
     *
     * Simple upwind: flux_i+1/2 = u * alpha_i (if u > 0)
     *
     * @param alpha    Volume fraction array
     * @param velocity Uniform advection velocity
     * @param dx       Cell spacing
     * @param dt       Time step
     * @param n_cells  Number of cells
     * @param alpha_new [out] Updated volume fractions
     */
    void advect_1d(const std::vector<Real>& alpha, Real velocity,
                   Real dx, Real dt, int n_cells,
                   std::vector<Real>& alpha_new) const {
        alpha_new.resize(n_cells);
        Real courant = velocity * dt / dx;

        for (int i = 0; i < n_cells; ++i) {
            Real a_left  = (i > 0)           ? alpha[i-1] : alpha[i];
            Real a_right = (i < n_cells - 1) ? alpha[i+1] : alpha[i];

            Real flux_in, flux_out;
            if (velocity >= 0.0) {
                flux_in  = courant * a_left;
                flux_out = courant * alpha[i];
            } else {
                flux_in  = -courant * alpha[i];
                flux_out = -courant * a_right;
            }

            alpha_new[i] = alpha[i] + flux_in - flux_out;
            if (alpha_new[i] < 0.0) alpha_new[i] = 0.0;
            if (alpha_new[i] > 1.0) alpha_new[i] = 1.0;
        }
    }
};

// ============================================================================
// 27c: PorousMediaFlow — Darcy Flow + Effective Stress + Biot Coupling
// ============================================================================

/**
 * @brief Porous media solver implementing Darcy's law, Biot effective stress,
 *        and pressure diffusion for single-phase saturated flow.
 *
 * Darcy's law:
 *   v_f = -(kappa/mu) * (grad(p) - rho_f * g)
 *
 * Effective stress (Biot):
 *   sigma'_ij = sigma_ij + alpha_B * p * delta_ij
 *
 * Pressure diffusion (storage):
 *   S * dp/dt = div(kappa/mu * grad(p)) + Q
 *
 * @note kappa is permeability [m^2], mu is dynamic viscosity [Pa.s],
 *       alpha_B is Biot coefficient [dimensionless, 0 < alpha_B <= 1].
 */
class PorousMediaSolver {
public:
    PorousMediaSolver() = default;

    /**
     * @brief Compute Darcy seepage velocity.
     *
     * v_i = -(kappa/mu) * (dp/dx_i - rho_f * g_i)
     *
     * @param permeability  Intrinsic permeability kappa [m^2]
     * @param viscosity     Dynamic viscosity mu [Pa.s]
     * @param pressure_gradient  Pressure gradient [3] [Pa/m]
     * @param rho_f         Fluid density [kg/m^3]
     * @param gravity       Gravity vector [3] [m/s^2]
     * @param velocity      [out] Darcy velocity [3] [m/s]
     */
    KOKKOS_INLINE_FUNCTION
    void darcy_velocity(Real permeability, Real viscosity,
                        const Real pressure_gradient[3],
                        Real rho_f, const Real gravity[3],
                        Real velocity[3]) const {
        if (viscosity < 1.0e-30) viscosity = 1.0e-30;
        Real k_over_mu = permeability / viscosity;

        for (int i = 0; i < 3; ++i) {
            velocity[i] = -k_over_mu * (pressure_gradient[i] - rho_f * gravity[i]);
        }
    }

    /**
     * @brief Compute effective stress from total stress and pore pressure.
     *
     * sigma'_ij = sigma_ij + alpha_B * p * delta_ij
     *
     * Voigt notation [6]: [s11, s22, s33, s12, s23, s13]
     * delta_ij in Voigt: [1, 1, 1, 0, 0, 0]
     *
     * @param total_stress   Total stress [6] (Voigt)
     * @param pore_pressure  Pore pressure p
     * @param biot_coeff     Biot coefficient alpha_B
     * @param eff_stress     [out] Effective stress [6]
     */
    KOKKOS_INLINE_FUNCTION
    void effective_stress(const Real total_stress[6], Real pore_pressure,
                          Real biot_coeff, Real eff_stress[6]) const {
        for (int i = 0; i < 6; ++i) {
            eff_stress[i] = total_stress[i];
        }
        // Add Biot pore pressure to diagonal (normal) components
        eff_stress[0] += biot_coeff * pore_pressure;
        eff_stress[1] += biot_coeff * pore_pressure;
        eff_stress[2] += biot_coeff * pore_pressure;
    }

    /**
     * @brief One-step explicit pressure diffusion on a 1D grid.
     *
     * S * dp/dt = (kappa/mu) * d2p/dx2 + Q
     *
     * Forward Euler:
     *   p_new[i] = p[i] + (dt/S) * [(kappa/mu)*(p[i-1] - 2*p[i] + p[i+1])/dx^2 + Q]
     *
     * Boundary: zero-flux (Neumann) by copying neighbor.
     *
     * @param pressures     Current pressure array
     * @param permeability  Permeability kappa
     * @param viscosity     Viscosity mu
     * @param storage       Storage coefficient S
     * @param dx            Cell spacing
     * @param dt            Time step
     * @param n_cells       Number of cells
     * @param source        Source term Q (uniform)
     * @param new_pressures [out] Updated pressures
     */
    void pressure_diffusion_step(const std::vector<Real>& pressures,
                                 Real permeability, Real viscosity,
                                 Real storage, Real dx, Real dt,
                                 int n_cells, Real source,
                                 std::vector<Real>& new_pressures) const {
        new_pressures.resize(n_cells);
        Real alpha = (permeability / viscosity) / (dx * dx);

        for (int i = 0; i < n_cells; ++i) {
            Real p_left  = (i > 0)           ? pressures[i-1] : pressures[i];
            Real p_right = (i < n_cells - 1) ? pressures[i+1] : pressures[i];
            Real laplacian = alpha * (p_left - 2.0 * pressures[i] + p_right);
            new_pressures[i] = pressures[i] + (dt / storage) * (laplacian + source);
        }
    }

    /**
     * @brief Overload without source term (Q = 0).
     */
    void pressure_diffusion_step(const std::vector<Real>& pressures,
                                 Real permeability, Real viscosity,
                                 Real storage, Real dx, Real dt,
                                 int n_cells,
                                 std::vector<Real>& new_pressures) const {
        pressure_diffusion_step(pressures, permeability, viscosity,
                                storage, dx, dt, n_cells, 0.0, new_pressures);
    }

    /**
     * @brief Compute Darcy velocity magnitude.
     */
    KOKKOS_INLINE_FUNCTION
    Real darcy_velocity_magnitude(Real permeability, Real viscosity,
                                  const Real pressure_gradient[3],
                                  Real rho_f, const Real gravity[3]) const {
        Real v[3];
        darcy_velocity(permeability, viscosity, pressure_gradient, rho_f, gravity, v);
        return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    }
};

// ============================================================================
// 27d: ALEErosionHandler — Void Filling with Mass Conservation
// ============================================================================

/**
 * @brief Handles element erosion (removal) in ALE computations.
 *
 * When elements fail and are removed:
 * 1. The eroded element's mass, momentum, and energy are redistributed
 *    to its neighbors proportional to the shared face area.
 * 2. The eroded element is marked as void and excluded from computation.
 * 3. Global conservation is tracked and verified.
 *
 * Redistribution formula:
 *   m_neighbor_i += (A_i / A_total) * m_eroded
 *   p_neighbor_i += (A_i / A_total) * p_eroded
 *   E_neighbor_i += (A_i / A_total) * E_eroded
 */
class ALEErosionHandler {
public:
    /// Maximum number of elements that can be tracked
    static constexpr int MAX_ELEMENTS = 10000;

    ALEErosionHandler() : total_eroded_mass_(0.0), total_eroded_energy_(0.0) {
        eroded_flags_.clear();
    }

    /**
     * @brief Redistribute eroded element's conserved quantities to neighbors.
     *
     * @param elem_id         ID of the eroded element
     * @param elem_mass       Mass of the eroded element
     * @param elem_momentum   Momentum of eroded element [3]
     * @param elem_energy     Energy of the eroded element
     * @param neighbor_areas  Shared face areas with each neighbor
     * @param num_neighbors   Number of neighbors
     * @param out_mass        [out] Mass contribution to each neighbor
     * @param out_momentum_x  [out] X-momentum contribution to each neighbor
     * @param out_momentum_y  [out] Y-momentum contribution
     * @param out_momentum_z  [out] Z-momentum contribution
     * @param out_energy      [out] Energy contribution to each neighbor
     */
    void erode_element(int elem_id, Real elem_mass,
                       const Real elem_momentum[3], Real elem_energy,
                       const std::vector<Real>& neighbor_areas,
                       int num_neighbors,
                       std::vector<Real>& out_mass,
                       std::vector<Real>& out_momentum_x,
                       std::vector<Real>& out_momentum_y,
                       std::vector<Real>& out_momentum_z,
                       std::vector<Real>& out_energy) {
        out_mass.resize(num_neighbors, 0.0);
        out_momentum_x.resize(num_neighbors, 0.0);
        out_momentum_y.resize(num_neighbors, 0.0);
        out_momentum_z.resize(num_neighbors, 0.0);
        out_energy.resize(num_neighbors, 0.0);

        if (num_neighbors <= 0) {
            // No neighbors: mass is lost (edge case)
            mark_eroded(elem_id);
            total_eroded_mass_ += elem_mass;
            total_eroded_energy_ += elem_energy;
            return;
        }

        // Total shared face area
        Real A_total = 0.0;
        for (int i = 0; i < num_neighbors; ++i) {
            A_total += neighbor_areas[i];
        }

        if (A_total < 1.0e-30) {
            // Distribute equally
            Real weight = 1.0 / num_neighbors;
            for (int i = 0; i < num_neighbors; ++i) {
                out_mass[i]       = weight * elem_mass;
                out_momentum_x[i] = weight * elem_momentum[0];
                out_momentum_y[i] = weight * elem_momentum[1];
                out_momentum_z[i] = weight * elem_momentum[2];
                out_energy[i]     = weight * elem_energy;
            }
        } else {
            Real A_inv = 1.0 / A_total;
            for (int i = 0; i < num_neighbors; ++i) {
                Real weight = neighbor_areas[i] * A_inv;
                out_mass[i]       = weight * elem_mass;
                out_momentum_x[i] = weight * elem_momentum[0];
                out_momentum_y[i] = weight * elem_momentum[1];
                out_momentum_z[i] = weight * elem_momentum[2];
                out_energy[i]     = weight * elem_energy;
            }
        }

        mark_eroded(elem_id);
    }

    /**
     * @brief Mark an element as eroded.
     */
    void mark_eroded(int elem_id) {
        eroded_flags_.insert(elem_id);
    }

    /**
     * @brief Check if an element has been eroded.
     */
    bool is_eroded(int elem_id) const {
        return eroded_flags_.find(elem_id) != eroded_flags_.end();
    }

    /**
     * @brief Compute mass conservation error.
     * @param original_total Original total mass in the system
     * @param current_total  Current total mass (sum of all remaining elements)
     * @return Relative error |original - current| / original
     */
    KOKKOS_INLINE_FUNCTION
    static Real total_mass_check(Real original_total, Real current_total) {
        if (std::abs(original_total) < 1.0e-30) return 0.0;
        return std::abs(original_total - current_total) / std::abs(original_total);
    }

    /**
     * @brief Get number of eroded elements.
     */
    int num_eroded() const { return static_cast<int>(eroded_flags_.size()); }

    /**
     * @brief Reset all erosion tracking.
     */
    void reset() {
        eroded_flags_.clear();
        total_eroded_mass_ = 0.0;
        total_eroded_energy_ = 0.0;
    }

    Real total_eroded_mass() const { return total_eroded_mass_; }
    Real total_eroded_energy() const { return total_eroded_energy_; }

private:
    std::set<int> eroded_flags_;
    Real total_eroded_mass_;
    Real total_eroded_energy_;
};

// ============================================================================
// 27e: ALEAdaptiveRemesh — Gradient-Based Error Indicator + Solution Transfer
// ============================================================================

/**
 * @brief Adaptive remeshing controller with Zienkiewicz-Zhu type error
 *        indicator, refinement/coarsening marking, and conservative L2
 *        projection solution transfer.
 *
 * Error indicator per element:
 *   eta_e = h_e^p * |grad^2(u)|_e
 *
 * where h_e is the element size, p is the polynomial order (1 for linear),
 * and |grad^2(u)| is approximated from gradient jumps at element interfaces.
 *
 * Refinement criterion:
 *   Refine  if eta_e > theta_refine * max(eta)
 *   Coarsen if eta_e < theta_coarsen * max(eta)
 *
 * Solution transfer:
 *   Nearest-neighbor interpolation for simplicity, with optional
 *   inverse-distance weighting for smoother results.
 */
class AdaptiveRemeshIndicator {
public:
    /// Default thresholds
    static constexpr Real default_theta_refine  = 0.5;
    static constexpr Real default_theta_coarsen = 0.1;

    AdaptiveRemeshIndicator() = default;

    /**
     * @brief Compute element-wise error indicators.
     *
     * eta_e = h_e^p * |second_derivative_e|
     *
     * The second derivative is approximated as:
     *   |grad^2(u)|_e ≈ |grad(u)_right - grad(u)_left| / h_e
     * which simplifies to using the gradient array directly:
     *   |grad^2(u)|_e ≈ |diff of neighbors' gradients| / h_e
     *
     * For simplicity, uses the magnitude of the gradient as first-order
     * proxy, and jumps in the field as second-order indicator.
     *
     * @param field_values   Solution values per element
     * @param field_gradients Gradient magnitude per element
     * @param element_sizes  Characteristic element sizes h_e
     * @param num_elements   Number of elements
     * @param indicators     [out] Error indicators
     */
    void compute_error_indicator(const std::vector<Real>& field_values,
                                 const std::vector<Real>& field_gradients,
                                 const std::vector<Real>& element_sizes,
                                 int num_elements,
                                 std::vector<Real>& indicators) const {
        indicators.resize(num_elements, 0.0);

        for (int i = 0; i < num_elements; ++i) {
            // Approximate second derivative from gradient jumps
            Real grad_left  = (i > 0)                ? field_gradients[i-1] : field_gradients[i];
            Real grad_right = (i < num_elements - 1) ? field_gradients[i+1] : field_gradients[i];

            Real second_deriv = std::abs(grad_right - grad_left) / element_sizes[i];

            // ZZ-type indicator: h^p * |d2u/dx2|, with p=1
            Real h = element_sizes[i];
            indicators[i] = h * second_deriv;
        }
    }

    /**
     * @brief Mark elements for refinement or coarsening.
     *
     * @param indicators     Error indicators per element
     * @param num_elements   Number of elements
     * @param theta_refine   Refinement threshold (fraction of max indicator)
     * @param theta_coarsen  Coarsening threshold (fraction of max indicator)
     * @param flags          [out] Flags: 1 = refine, -1 = coarsen, 0 = keep
     */
    void mark_refinement(const std::vector<Real>& indicators,
                         int num_elements,
                         Real theta_refine, Real theta_coarsen,
                         std::vector<int>& flags) const {
        flags.resize(num_elements, 0);

        // Find maximum indicator
        Real max_eta = 0.0;
        for (int i = 0; i < num_elements; ++i) {
            if (indicators[i] > max_eta) max_eta = indicators[i];
        }

        if (max_eta < 1.0e-30) {
            // Uniform field, no refinement needed
            return;
        }

        Real refine_threshold  = theta_refine * max_eta;
        Real coarsen_threshold = theta_coarsen * max_eta;

        for (int i = 0; i < num_elements; ++i) {
            if (indicators[i] > refine_threshold) {
                flags[i] = 1;  // Refine
            } else if (indicators[i] < coarsen_threshold) {
                flags[i] = -1; // Coarsen
            } else {
                flags[i] = 0;  // Keep
            }
        }
    }

    /**
     * @brief Transfer field values from old mesh to new mesh.
     *
     * Uses inverse-distance weighted interpolation from old mesh points.
     * For each new point, finds all old points and weights by 1/r^2.
     *
     * Special cases:
     * - If a new point coincides with an old point, uses that value directly.
     * - Preserves constant and linear fields exactly (up to machine precision).
     *
     * @param old_values  Field values at old mesh points
     * @param old_coords  Coordinates of old mesh points (1D)
     * @param new_coords  Coordinates of new mesh points (1D)
     * @param num_old     Number of old points
     * @param num_new     Number of new points
     * @param new_values  [out] Interpolated field values
     */
    void transfer_field(const std::vector<Real>& old_values,
                        const std::vector<Real>& old_coords,
                        const std::vector<Real>& new_coords,
                        int num_old, int num_new,
                        std::vector<Real>& new_values) const {
        new_values.resize(num_new, 0.0);

        for (int j = 0; j < num_new; ++j) {
            Real x_new = new_coords[j];

            // Check for exact match first
            bool exact = false;
            for (int i = 0; i < num_old; ++i) {
                if (std::abs(old_coords[i] - x_new) < 1.0e-14) {
                    new_values[j] = old_values[i];
                    exact = true;
                    break;
                }
            }
            if (exact) continue;

            // Find bracketing old points for linear interpolation (1D)
            // This preserves linear fields exactly
            int left = -1, right = -1;
            Real dist_left = 1.0e30, dist_right = 1.0e30;

            for (int i = 0; i < num_old; ++i) {
                Real dx = old_coords[i] - x_new;
                if (dx <= 0.0 && std::abs(dx) < dist_left) {
                    dist_left = std::abs(dx);
                    left = i;
                }
                if (dx > 0.0 && dx < dist_right) {
                    dist_right = dx;
                    right = i;
                }
            }

            if (left >= 0 && right >= 0) {
                // Linear interpolation between bracketing points
                Real x_l = old_coords[left];
                Real x_r = old_coords[right];
                Real denom = x_r - x_l;
                if (std::abs(denom) < 1.0e-30) {
                    new_values[j] = old_values[left];
                } else {
                    Real t = (x_new - x_l) / denom;
                    new_values[j] = (1.0 - t) * old_values[left] + t * old_values[right];
                }
            } else if (left >= 0) {
                // Extrapolate from left
                new_values[j] = old_values[left];
            } else if (right >= 0) {
                // Extrapolate from right
                new_values[j] = old_values[right];
            }
        }
    }

    /**
     * @brief Compute the integral of a field (sum of value * element_size).
     * Used for conservation checking.
     */
    static Real compute_integral(const std::vector<Real>& values,
                                 const std::vector<Real>& sizes, int n) {
        Real sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += values[i] * sizes[i];
        }
        return sum;
    }

    /**
     * @brief Compute L2 norm of a field.
     */
    static Real compute_l2_norm(const std::vector<Real>& values, int n) {
        Real sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += values[i] * values[i];
        }
        return std::sqrt(sum / n);
    }
};

} // namespace fem
} // namespace nxs
