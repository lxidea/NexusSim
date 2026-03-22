#pragma once

/**
 * @file sph_wave19.hpp
 * @brief 7 SPH enrichment features for Wave 19
 *
 * Features implemented:
 *   1.  TensileInstabilityCorrection  - Artificial stress for tensile instability
 *   2.  MultiPhaseSPH                 - Multi-phase density discontinuity handling
 *   3.  SPHBoundaryTreatment          - Ghost/dummy particle boundary enforcement
 *   4.  SPHContactHandler             - SPH-to-SPH body contact with penalty
 *   5.  VerletNeighborList             - Verlet list with skin distance optimization
 *   6.  SPHThermalCoupling            - Heat conduction between SPH particles
 *   7.  SPHMUSCLReconstruction        - Second-order MUSCL gradient reconstruction
 *
 * All classes use Real type, KOKKOS_INLINE_FUNCTION where applicable.
 * Namespace: nxs::sph
 */

#include <nexussim/physics/material.hpp>
#include <nexussim/sph/sph_kernel.hpp>
#include <nexussim/sph/neighbor_search.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <array>

namespace nxs {
namespace sph {

// ============================================================================
// 1. Tensile Instability Correction
// ============================================================================

/**
 * @brief Artificial stress method for SPH tensile instability correction
 *
 * When particles are in tension, SPH can develop spurious clumping.
 * The artificial stress method adds a repulsive stress to counteract:
 *
 *   sigma_art_ij = epsilon * (W(r_ij, h) / W(delta, h))^n * R_ij
 *
 * Where:
 *   epsilon = artificial stress coefficient (typically 0.3)
 *   n = exponent controlling repulsion decay (typically 4)
 *   delta = initial particle spacing
 *   R_ij = tensile part of stress tensor interpolated to pair midpoint
 *
 * The correction is added to the momentum equation as an additional
 * inter-particle force only when particles are in tension.
 *
 * Reference: Monaghan (2000), J. Comp. Phys. 159; Gray et al. (2001)
 */
class TensileInstabilityCorrection {
public:
    /**
     * @brief Construct with artificial stress parameters
     * @param epsilon Artificial stress coefficient (default 0.3)
     * @param n Exponent on kernel ratio (default 4.0)
     * @param delta Initial particle spacing (m)
     */
    TensileInstabilityCorrection(Real epsilon = 0.3, Real n = 4.0, Real delta = 0.01)
        : epsilon_(epsilon), n_(n), delta_(delta) {}

    /**
     * @brief Compute artificial stress tensor for a particle in tension
     *
     * Extracts principal stresses and adds repulsive artificial stress
     * for any tensile component.
     *
     * @param stress Cauchy stress tensor [6] in Voigt notation
     * @param R_out Output artificial stress tensor [6]
     */
    KOKKOS_INLINE_FUNCTION
    void compute_artificial_stress(const Real* stress, Real* R_out) const {
        // Extract principal stress approximation (diagonal dominance)
        // For each normal component, add artificial stress if tensile
        for (int i = 0; i < 6; ++i) R_out[i] = 0.0;

        // Normal components: add repulsive stress for tension
        for (int i = 0; i < 3; ++i) {
            if (stress[i] > 0.0) {
                // Tensile: R = epsilon * |sigma|
                R_out[i] = epsilon_ * stress[i];
            }
        }
    }

    /**
     * @brief Compute the kernel-ratio scaling factor for a particle pair
     *
     * @param r_ij Distance between particles i and j
     * @param h Smoothing length
     * @param kernel SPH kernel for evaluation
     * @return Scaling factor (W(r_ij)/W(delta))^n
     */
    KOKKOS_INLINE_FUNCTION
    Real kernel_ratio_factor(Real r_ij, Real h, const SPHKernel& kernel) const {
        Real W_rij = kernel.W(r_ij, h);
        Real W_delta = kernel.W(delta_, h);

        if (W_delta < 1.0e-30) return 0.0;

        Real ratio = W_rij / W_delta;
        // Compute ratio^n efficiently
        Real factor = 1.0;
        for (int k = 0; k < static_cast<int>(n_); ++k) {
            factor *= ratio;
        }
        // Handle fractional exponent
        if (n_ != static_cast<Real>(static_cast<int>(n_))) {
            factor = Kokkos::pow(ratio, n_);
        }
        return factor;
    }

    /**
     * @brief Compute artificial stress acceleration for a particle pair
     *
     * @param R_i Artificial stress tensor of particle i [6]
     * @param R_j Artificial stress tensor of particle j [6]
     * @param grad_W Kernel gradient vector (gWx, gWy, gWz)
     * @param mass_j Mass of particle j
     * @param rho_i Density of particle i
     * @param rho_j Density of particle j
     * @param f Factor from kernel_ratio_factor
     * @param acc_x Output x-acceleration contribution
     * @param acc_y Output y-acceleration contribution
     * @param acc_z Output z-acceleration contribution
     */
    KOKKOS_INLINE_FUNCTION
    void compute_pair_acceleration(const Real* R_i, const Real* R_j,
                                   Real gWx, Real gWy, Real gWz,
                                   Real mass_j, Real rho_i, Real rho_j,
                                   Real f,
                                   Real& acc_x, Real& acc_y, Real& acc_z) const {
        if (rho_i < 1.0e-20 || rho_j < 1.0e-20) {
            acc_x = acc_y = acc_z = 0.0;
            return;
        }

        Real rho_i2 = rho_i * rho_i;
        Real rho_j2 = rho_j * rho_j;

        // Compute artificial stress contribution to acceleration
        // a_art = -sum_j m_j * f * (R_i/rho_i^2 + R_j/rho_j^2) . grad_W
        // Simplified: use trace of R as scalar artificial pressure
        Real R_i_trace = (R_i[0] + R_i[1] + R_i[2]) / 3.0;
        Real R_j_trace = (R_j[0] + R_j[1] + R_j[2]) / 3.0;

        Real coeff = -mass_j * f * (R_i_trace / rho_i2 + R_j_trace / rho_j2);

        acc_x = coeff * gWx;
        acc_y = coeff * gWy;
        acc_z = coeff * gWz;
    }

    void set_epsilon(Real eps) { epsilon_ = eps; }
    void set_exponent(Real n) { n_ = n; }
    void set_delta(Real delta) { delta_ = delta; }

    Real epsilon() const { return epsilon_; }
    Real exponent() const { return n_; }
    Real delta() const { return delta_; }
    const char* name() const { return "TensileInstabilityCorrection"; }

private:
    Real epsilon_;    ///< Artificial stress coefficient
    Real n_;          ///< Kernel ratio exponent
    Real delta_;      ///< Reference particle spacing
};


// ============================================================================
// 2. Multi-Phase SPH
// ============================================================================

/**
 * @brief Multi-phase SPH with density discontinuity handling
 *
 * Handles interfaces between fluids of different densities (e.g., water/air)
 * using a color function for interface detection and pressure smoothing
 * across the interface to prevent spurious fragmentation.
 *
 * Color function:
 *   c_i = sum_j (m_j / rho_j) * C_j * W_ij
 *
 * Where C_j is the phase identifier (0 or 1). The interface is located
 * where 0 < c_i < 1. Pressure is smoothed at the interface using a
 * weighted average to prevent density ratio instabilities.
 *
 * Reference: Colagrossi & Landrini (2003), J. Comp. Phys.;
 *            Hu & Adams (2006), J. Comp. Phys.
 */
class MultiPhaseSPH {
public:
    /**
     * @brief Construct with phase properties
     * @param num_phases Number of distinct phases (default 2)
     * @param interface_width Smoothed interface width in units of h (default 1.5)
     */
    MultiPhaseSPH(int num_phases = 2, Real interface_width = 1.5)
        : num_phases_(num_phases)
        , interface_width_(interface_width)
    {
        phase_density_.resize(num_phases, 1000.0);
        phase_viscosity_.resize(num_phases, 0.001);
    }

    /**
     * @brief Set material properties for a phase
     * @param phase Phase index (0-based)
     * @param density Reference density (kg/m^3)
     * @param viscosity Dynamic viscosity (Pa.s)
     */
    void set_phase_properties(int phase, Real density, Real viscosity) {
        if (phase >= 0 && phase < num_phases_) {
            phase_density_[phase] = density;
            phase_viscosity_[phase] = viscosity;
        }
    }

    /**
     * @brief Compute color function for interface detection
     *
     * @param particle_idx Current particle index
     * @param phase_id Phase identifier array (0 or 1 for two phases)
     * @param mass Mass array
     * @param rho Density array
     * @param pos_x X-position array
     * @param pos_y Y-position array
     * @param pos_z Z-position array
     * @param neighbor_list Neighbor indices
     * @param num_neighbors Number of neighbors
     * @param h Smoothing length
     * @param kernel SPH kernel
     * @return Color function value (0 = phase 0, 1 = phase 1, intermediate = interface)
     */
    Real compute_color_function(Index particle_idx,
                                const int* phase_id,
                                const Real* mass,
                                const Real* rho,
                                const Real* pos_x,
                                const Real* pos_y,
                                const Real* pos_z,
                                const Index* neighbor_list,
                                size_t num_neighbors,
                                Real h,
                                const SPHKernel& kernel) const {
        Real color = 0.0;

        // Self contribution
        Real W_self = kernel.W(0.0, h);
        Real self_vol = mass[particle_idx] / rho[particle_idx];
        color += self_vol * static_cast<Real>(phase_id[particle_idx]) * W_self;

        // Neighbor contributions
        for (size_t k = 0; k < num_neighbors; ++k) {
            Index j = neighbor_list[k];
            Real rx = pos_x[particle_idx] - pos_x[j];
            Real ry = pos_y[particle_idx] - pos_y[j];
            Real rz = pos_z[particle_idx] - pos_z[j];
            Real r = std::sqrt(rx * rx + ry * ry + rz * rz);

            Real W = kernel.W(r, h);
            Real vol_j = mass[j] / rho[j];
            color += vol_j * static_cast<Real>(phase_id[j]) * W;
        }

        return color;
    }

    /**
     * @brief Compute interface-smoothed pressure for a particle pair
     *
     * At interfaces, density discontinuities cause pressure jumps.
     * This method smooths the pressure using a harmonic average
     * to maintain hydrostatic equilibrium across the interface.
     *
     * @param p_i Pressure of particle i
     * @param p_j Pressure of particle j
     * @param rho_i Density of particle i
     * @param rho_j Density of particle j
     * @param phase_i Phase of particle i
     * @param phase_j Phase of particle j
     * @return Smoothed pressure term for the pair
     */
    KOKKOS_INLINE_FUNCTION
    Real smoothed_pressure_term(Real p_i, Real p_j,
                                Real rho_i, Real rho_j,
                                int phase_i, int phase_j) const {
        if (phase_i == phase_j) {
            // Same phase: standard SPH pressure formulation
            return p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j);
        } else {
            // Different phases: use weighted average to maintain equilibrium
            // Hu & Adams (2006) formulation
            Real rho_avg = 0.5 * (rho_i + rho_j);
            Real p_avg = (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j);
            return 2.0 * p_avg / (rho_avg * rho_avg);
        }
    }

    /**
     * @brief Compute inter-phase viscous stress
     *
     * Uses harmonic mean of viscosities at the interface for smooth
     * viscous force transition.
     *
     * @param phase_i Phase index of particle i
     * @param phase_j Phase index of particle j
     * @return Effective viscosity for the pair
     */
    KOKKOS_INLINE_FUNCTION
    Real effective_viscosity(int phase_i, int phase_j) const {
        if (phase_i == phase_j) {
            return phase_viscosity_[phase_i];
        }
        // Harmonic mean for interface viscosity
        Real mu_i = phase_viscosity_[phase_i];
        Real mu_j = phase_viscosity_[phase_j];
        if (mu_i + mu_j < 1.0e-30) return 0.0;
        return 2.0 * mu_i * mu_j / (mu_i + mu_j);
    }

    /**
     * @brief Check if a particle is at an interface
     * @param color Color function value
     * @return true if particle is in the interface region
     */
    KOKKOS_INLINE_FUNCTION
    bool is_interface(Real color) const {
        return color > 0.01 && color < 0.99;
    }

    int num_phases() const { return num_phases_; }
    Real interface_width() const { return interface_width_; }
    const char* name() const { return "MultiPhaseSPH"; }

private:
    int num_phases_;
    Real interface_width_;
    std::vector<Real> phase_density_;
    std::vector<Real> phase_viscosity_;
};


// ============================================================================
// 3. SPH Boundary Treatment
// ============================================================================

/**
 * @brief Boundary treatment for SPH using ghost/dummy particles
 *
 * Implements two complementary boundary methods:
 *
 * 1. Lennard-Jones repulsive force (Monaghan 1994):
 *    F_b = D * [(r0/r)^p1 - (r0/r)^p2] * (x/r^2)  for r < r0
 *    where D = c0^2, p1 = 12, p2 = 4 (short-range repulsion)
 *
 * 2. Mirror particle method:
 *    Boundary particles mirror interior particle properties with
 *    reflected velocity for no-slip or free-slip conditions.
 *
 * Reference: Monaghan (1994), J. Comp. Phys.; Adami et al. (2012)
 */
class SPHBoundaryTreatment {
public:
    /// Boundary condition type
    enum class BoundaryType {
        Repulsive,      ///< Lennard-Jones repulsive force
        Mirror,         ///< Mirror/ghost particle
        RepulsiveMirror ///< Combined approach
    };

    /// Wall definition (infinite plane)
    struct Wall {
        Real normal[3];    ///< Outward normal of wall
        Real point[3];     ///< Point on the wall
        BoundaryType type; ///< Boundary enforcement method

        Wall() : normal{0, 0, 1}, point{0, 0, 0}, type(BoundaryType::Repulsive) {}
    };

    /**
     * @brief Construct boundary handler
     * @param c0 Reference speed of sound (for LJ force magnitude)
     * @param r0 Cutoff distance for repulsive force (typically dx)
     */
    SPHBoundaryTreatment(Real c0 = 1480.0, Real r0 = 0.01)
        : c0_(c0), r0_(r0), p1_(12), p2_(4)
    {
        D_ = c0_ * c0_;
    }

    /**
     * @brief Add a wall boundary
     */
    void add_wall(const Wall& wall) {
        walls_.push_back(wall);
    }

    /**
     * @brief Add a planar wall from normal and offset
     */
    void add_planar_wall(Real nx, Real ny, Real nz,
                         Real px, Real py, Real pz,
                         BoundaryType type = BoundaryType::Repulsive) {
        Wall w;
        Real len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len < 1.0e-20) len = 1.0e-20;
        w.normal[0] = nx / len;
        w.normal[1] = ny / len;
        w.normal[2] = nz / len;
        w.point[0] = px;
        w.point[1] = py;
        w.point[2] = pz;
        w.type = type;
        walls_.push_back(w);
    }

    /**
     * @brief Compute repulsive boundary force on a particle
     *
     * @param px, py, pz Particle position
     * @param fx, fy, fz Output boundary force (accumulated)
     */
    KOKKOS_INLINE_FUNCTION
    void compute_repulsive_force(Real px, Real py, Real pz,
                                 Real& fx, Real& fy, Real& fz) const {
        fx = fy = fz = 0.0;

        for (size_t w = 0; w < walls_.size(); ++w) {
            const Wall& wall = walls_[w];
            if (wall.type != BoundaryType::Repulsive &&
                wall.type != BoundaryType::RepulsiveMirror) continue;

            // Signed distance from particle to wall
            Real dx = px - wall.point[0];
            Real dy = py - wall.point[1];
            Real dz = pz - wall.point[2];
            Real dist = dx * wall.normal[0] + dy * wall.normal[1] + dz * wall.normal[2];

            // Only apply when particle is close to wall (within r0)
            if (dist > 0.0 && dist < r0_) {
                Real q = dist / r0_;  // Normalized distance (0 < q < 1)
                if (q < 1.0e-10) q = 1.0e-10;

                // Lennard-Jones-like force
                Real q_p1 = 1.0;
                Real q_p2 = 1.0;
                for (int k = 0; k < p1_; ++k) q_p1 *= (1.0 / q);
                for (int k = 0; k < p2_; ++k) q_p2 *= (1.0 / q);

                Real force_mag = D_ * (q_p1 - q_p2) / (dist * dist);

                // Force is in the wall normal direction (pushes particle away)
                fx += force_mag * wall.normal[0];
                fy += force_mag * wall.normal[1];
                fz += force_mag * wall.normal[2];
            }
        }
    }

    /**
     * @brief Generate mirror particle position for boundary enforcement
     *
     * @param px, py, pz Interior particle position
     * @param wall_idx Wall index
     * @param mx, my, mz Output mirror particle position
     */
    KOKKOS_INLINE_FUNCTION
    void mirror_position(Real px, Real py, Real pz,
                         size_t wall_idx,
                         Real& mx, Real& my, Real& mz) const {
        if (wall_idx >= walls_.size()) {
            mx = px; my = py; mz = pz;
            return;
        }

        const Wall& wall = walls_[wall_idx];

        // Distance from point to wall
        Real dx = px - wall.point[0];
        Real dy = py - wall.point[1];
        Real dz = pz - wall.point[2];
        Real dist = dx * wall.normal[0] + dy * wall.normal[1] + dz * wall.normal[2];

        // Mirror position: reflect across wall
        mx = px - 2.0 * dist * wall.normal[0];
        my = py - 2.0 * dist * wall.normal[1];
        mz = pz - 2.0 * dist * wall.normal[2];
    }

    /**
     * @brief Compute mirror velocity for no-slip boundary
     *
     * @param vx, vy, vz Interior particle velocity
     * @param wall_idx Wall index
     * @param mvx, mvy, mvz Output mirror particle velocity
     * @param no_slip If true, reverse tangential velocity; if false, free-slip
     */
    KOKKOS_INLINE_FUNCTION
    void mirror_velocity(Real vx, Real vy, Real vz,
                         size_t wall_idx, bool no_slip,
                         Real& mvx, Real& mvy, Real& mvz) const {
        if (wall_idx >= walls_.size()) {
            mvx = -vx; mvy = -vy; mvz = -vz;
            return;
        }

        const Wall& wall = walls_[wall_idx];

        // Normal velocity component
        Real v_n = vx * wall.normal[0] + vy * wall.normal[1] + vz * wall.normal[2];

        // Tangential velocity
        Real vtx = vx - v_n * wall.normal[0];
        Real vty = vy - v_n * wall.normal[1];
        Real vtz = vz - v_n * wall.normal[2];

        if (no_slip) {
            // No-slip: reverse both normal and tangential
            mvx = -vx;
            mvy = -vy;
            mvz = -vz;
        } else {
            // Free-slip: reverse only normal component
            mvx = vtx - v_n * wall.normal[0];
            mvy = vty - v_n * wall.normal[1];
            mvz = vtz - v_n * wall.normal[2];
        }
    }

    /**
     * @brief Check if particle is within boundary influence zone
     */
    KOKKOS_INLINE_FUNCTION
    bool near_boundary(Real px, Real py, Real pz, Real influence_radius) const {
        for (size_t w = 0; w < walls_.size(); ++w) {
            Real dx = px - walls_[w].point[0];
            Real dy = py - walls_[w].point[1];
            Real dz = pz - walls_[w].point[2];
            Real dist = dx * walls_[w].normal[0] + dy * walls_[w].normal[1]
                      + dz * walls_[w].normal[2];
            if (dist >= 0.0 && dist < influence_radius) return true;
        }
        return false;
    }

    void set_r0(Real r0) { r0_ = r0; }
    void set_c0(Real c0) { c0_ = c0; D_ = c0 * c0; }

    size_t num_walls() const { return walls_.size(); }
    const char* name() const { return "SPHBoundaryTreatment"; }

private:
    Real c0_;         ///< Reference speed of sound
    Real r0_;         ///< Cutoff distance for repulsive force
    Real D_;          ///< Force magnitude = c0^2
    int p1_;          ///< First LJ exponent (repulsive)
    int p2_;          ///< Second LJ exponent (attractive cutoff)
    std::vector<Wall> walls_;
};


// ============================================================================
// 4. SPH Contact Handler
// ============================================================================

/**
 * @brief SPH-to-SPH contact between different bodies
 *
 * Handles contact between distinct SPH bodies (e.g., impactor vs. target)
 * using a penalty-based approach. Bodies are identified by a body ID tag
 * on each particle. Contact forces are computed between particles of
 * different bodies that are within a contact search radius.
 *
 * Contact force:
 *   F_c = k_c * max(0, d_pen) * n_ij
 *
 * Where:
 *   d_pen = h_contact - |r_ij| (penetration distance)
 *   n_ij = (r_i - r_j) / |r_ij| (contact normal)
 *   k_c = penalty stiffness
 *
 * Friction can be added via Coulomb model:
 *   F_t = min(mu * |F_c|, k_t * delta_t) * t_ij
 *
 * Reference: Campbell et al. (2000); Vignjevic et al. (2006)
 */
class SPHContactHandler {
public:
    /**
     * @brief Construct contact handler
     * @param penalty_stiffness Contact penalty stiffness (N/m)
     * @param friction_coeff Coulomb friction coefficient
     * @param contact_distance Contact activation distance (m)
     */
    SPHContactHandler(Real penalty_stiffness = 1.0e9,
                      Real friction_coeff = 0.3,
                      Real contact_distance = 0.01)
        : k_c_(penalty_stiffness)
        , mu_(friction_coeff)
        , h_contact_(contact_distance)
    {}

    /**
     * @brief Compute contact force between two particles of different bodies
     *
     * @param rx, ry, rz Position vector r_i - r_j
     * @param r_mag Distance |r_ij|
     * @param vx, vy, vz Relative velocity v_i - v_j
     * @param fx, fy, fz Output contact force on particle i
     * @return true if contact is active
     */
    KOKKOS_INLINE_FUNCTION
    bool compute_contact_force(Real rx, Real ry, Real rz, Real r_mag,
                               Real vx, Real vy, Real vz,
                               Real& fx, Real& fy, Real& fz) const {
        fx = fy = fz = 0.0;

        // Check penetration
        Real d_pen = h_contact_ - r_mag;
        if (d_pen <= 0.0) return false;

        // Contact normal (from j to i)
        if (r_mag < 1.0e-20) return false;
        Real inv_r = 1.0 / r_mag;
        Real nx = rx * inv_r;
        Real ny = ry * inv_r;
        Real nz = rz * inv_r;

        // Normal force (penalty)
        Real Fn = k_c_ * d_pen;

        fx = Fn * nx;
        fy = Fn * ny;
        fz = Fn * nz;

        // Friction: tangential component of relative velocity
        if (mu_ > 0.0) {
            Real v_n = vx * nx + vy * ny + vz * nz;
            Real vtx = vx - v_n * nx;
            Real vty = vy - v_n * ny;
            Real vtz = vz - v_n * nz;
            Real vt_mag = Kokkos::sqrt(vtx * vtx + vty * vty + vtz * vtz);

            if (vt_mag > 1.0e-20) {
                Real Ft = mu_ * Fn;
                // Tangential force opposes sliding
                fx -= Ft * vtx / vt_mag;
                fy -= Ft * vty / vt_mag;
                fz -= Ft * vtz / vt_mag;
            }
        }

        return true;
    }

    /**
     * @brief Compute all contact forces between bodies
     *
     * @param body_id Body identifier for each particle
     * @param pos_x, pos_y, pos_z Particle positions
     * @param vel_x, vel_y, vel_z Particle velocities
     * @param mass Particle masses
     * @param num_particles Total number of particles
     * @param neighbor_pairs Neighbor pair list
     * @param acc_x, acc_y, acc_z Output: acceleration contributions (accumulated)
     */
    void compute_all_contact(const int* body_id,
                             const Real* pos_x, const Real* pos_y, const Real* pos_z,
                             const Real* vel_x, const Real* vel_y, const Real* vel_z,
                             const Real* mass,
                             size_t num_particles,
                             const std::vector<NeighborPair>& neighbor_pairs,
                             Real* acc_x, Real* acc_y, Real* acc_z) const {
        (void)num_particles;

        for (const auto& pair : neighbor_pairs) {
            Index i = pair.i;
            Index j = pair.j;

            // Only compute contact between different bodies
            if (body_id[i] == body_id[j]) continue;

            Real rx = pos_x[i] - pos_x[j];
            Real ry = pos_y[i] - pos_y[j];
            Real rz = pos_z[i] - pos_z[j];

            Real vx = vel_x[i] - vel_x[j];
            Real vy = vel_y[i] - vel_y[j];
            Real vz = vel_z[i] - vel_z[j];

            Real fx, fy, fz;
            bool active = compute_contact_force(rx, ry, rz, pair.r,
                                                vx, vy, vz,
                                                fx, fy, fz);

            if (active) {
                // Apply to particle i (action)
                if (mass[i] > 1.0e-30) {
                    acc_x[i] += fx / mass[i];
                    acc_y[i] += fy / mass[i];
                    acc_z[i] += fz / mass[i];
                }
                // Apply to particle j (reaction)
                if (mass[j] > 1.0e-30) {
                    acc_x[j] -= fx / mass[j];
                    acc_y[j] -= fy / mass[j];
                    acc_z[j] -= fz / mass[j];
                }
            }
        }
    }

    void set_penalty_stiffness(Real k) { k_c_ = k; }
    void set_friction(Real mu) { mu_ = mu; }
    void set_contact_distance(Real h) { h_contact_ = h; }

    Real penalty_stiffness() const { return k_c_; }
    Real friction_coefficient() const { return mu_; }
    Real contact_distance() const { return h_contact_; }
    const char* name() const { return "SPHContactHandler"; }

private:
    Real k_c_;          ///< Penalty stiffness
    Real mu_;           ///< Coulomb friction coefficient
    Real h_contact_;    ///< Contact activation distance
};


// ============================================================================
// 5. Verlet Neighbor List
// ============================================================================

/**
 * @brief Verlet list with skin distance for reduced neighbor rebuilds
 *
 * Standard neighbor lists must be rebuilt every timestep. The Verlet list
 * adds a skin distance r_skin beyond the support radius. Neighbors are
 * searched within (r_support + r_skin), and the list is only rebuilt when
 * any particle has moved more than r_skin/2 since the last build.
 *
 * This can reduce the frequency of expensive neighbor searches by a
 * factor of 5-20x for typical SPH simulations.
 *
 * Reference: Verlet (1967), Phys. Rev.; Dominguez et al. (2011)
 */
class VerletNeighborList {
public:
    /**
     * @brief Construct Verlet list
     * @param support_radius SPH support radius (h * kernel_support)
     * @param skin_distance Additional search distance (default 0.1 * support)
     */
    VerletNeighborList(Real support_radius = 0.02, Real skin_distance = 0.0)
        : support_radius_(support_radius)
        , skin_distance_(skin_distance > 0 ? skin_distance : 0.1 * support_radius)
        , total_radius_(support_radius + (skin_distance > 0 ? skin_distance : 0.1 * support_radius))
        , build_count_(0)
        , check_count_(0)
        , needs_rebuild_(true)
    {}

    /**
     * @brief Store reference positions from which displacements are measured
     */
    void store_reference_positions(const Real* pos_x, const Real* pos_y,
                                   const Real* pos_z, size_t n) {
        ref_x_.resize(n);
        ref_y_.resize(n);
        ref_z_.resize(n);
        max_displacement_.resize(n, 0.0);

        for (size_t i = 0; i < n; ++i) {
            ref_x_[i] = pos_x[i];
            ref_y_[i] = pos_y[i];
            ref_z_[i] = pos_z[i];
        }
    }

    /**
     * @brief Check if any particle has moved enough to require a rebuild
     *
     * The list needs rebuilding if max displacement > skin/2.
     * We check the maximum displacement of any particle.
     *
     * @param pos_x, pos_y, pos_z Current positions
     * @param n Number of particles
     * @return true if rebuild is needed
     */
    bool needs_rebuild(const Real* pos_x, const Real* pos_y,
                       const Real* pos_z, size_t n) {
        check_count_++;

        if (needs_rebuild_) return true;
        if (ref_x_.size() != n) return true;

        Real half_skin = 0.5 * skin_distance_;
        Real max_disp = 0.0;

        for (size_t i = 0; i < n; ++i) {
            Real dx = pos_x[i] - ref_x_[i];
            Real dy = pos_y[i] - ref_y_[i];
            Real dz = pos_z[i] - ref_z_[i];
            Real disp = std::sqrt(dx * dx + dy * dy + dz * dz);
            max_displacement_[i] = disp;
            if (disp > max_disp) max_disp = disp;
        }

        // Need rebuild if any particle moved more than half the skin distance
        // (two particles each moving half_skin could miss each other)
        return max_disp >= half_skin;
    }

    /**
     * @brief Build the Verlet neighbor list
     *
     * Searches with the extended radius (support + skin) and stores
     * all pairs. Subsequent queries filter by the actual support radius.
     *
     * @param pos_x, pos_y, pos_z Particle positions
     * @param n Number of particles
     * @param grid Spatial hash grid for search
     */
    void build(const Real* pos_x, const Real* pos_y, const Real* pos_z,
               size_t n, SpatialHashGrid& grid) {
        // Update grid cell size for extended search
        grid.set_cell_size(total_radius_);
        grid.build(pos_x, pos_y, pos_z, n);

        // Find neighbors within extended radius
        grid.find_neighbors(pos_x, pos_y, pos_z, total_radius_, verlet_pairs_);

        // Store reference positions
        store_reference_positions(pos_x, pos_y, pos_z, n);

        needs_rebuild_ = false;
        build_count_++;
    }

    /**
     * @brief Get current neighbor pairs (within support radius, filtered)
     *
     * Filters the Verlet list to only include pairs within the actual
     * support radius. Positions must be current.
     *
     * @param pos_x, pos_y, pos_z Current positions
     * @param active_pairs Output filtered pairs
     */
    void get_active_pairs(const Real* pos_x, const Real* pos_y, const Real* pos_z,
                          std::vector<NeighborPair>& active_pairs) const {
        active_pairs.clear();
        Real r2_max = support_radius_ * support_radius_;

        for (const auto& pair : verlet_pairs_) {
            Real rx = pos_x[pair.i] - pos_x[pair.j];
            Real ry = pos_y[pair.i] - pos_y[pair.j];
            Real rz = pos_z[pair.i] - pos_z[pair.j];
            Real r2 = rx * rx + ry * ry + rz * rz;

            if (r2 < r2_max && r2 > 1.0e-20) {
                NeighborPair active;
                active.i = pair.i;
                active.j = pair.j;
                active.r = std::sqrt(r2);
                active.rx = rx;
                active.ry = ry;
                active.rz = rz;
                active_pairs.push_back(active);
            }
        }
    }

    /**
     * @brief Conditionally rebuild if needed, otherwise return cached pairs
     *
     * @param pos_x, pos_y, pos_z Current positions
     * @param n Number of particles
     * @param grid Spatial hash grid
     * @param active_pairs Output active neighbor pairs
     */
    void update_and_get_pairs(const Real* pos_x, const Real* pos_y, const Real* pos_z,
                              size_t n, SpatialHashGrid& grid,
                              std::vector<NeighborPair>& active_pairs) {
        if (needs_rebuild(pos_x, pos_y, pos_z, n)) {
            build(pos_x, pos_y, pos_z, n, grid);
        }
        get_active_pairs(pos_x, pos_y, pos_z, active_pairs);
    }

    void set_support_radius(Real r) {
        support_radius_ = r;
        total_radius_ = r + skin_distance_;
    }

    void set_skin_distance(Real s) {
        skin_distance_ = s;
        total_radius_ = support_radius_ + s;
        needs_rebuild_ = true;
    }

    Real support_radius() const { return support_radius_; }
    Real skin_distance() const { return skin_distance_; }
    Real total_radius() const { return total_radius_; }
    size_t build_count() const { return build_count_; }
    size_t check_count() const { return check_count_; }
    size_t verlet_pair_count() const { return verlet_pairs_.size(); }

    /**
     * @brief Rebuild savings ratio: 1 - (builds / checks)
     */
    Real savings_ratio() const {
        if (check_count_ == 0) return 0.0;
        return 1.0 - static_cast<Real>(build_count_) / static_cast<Real>(check_count_);
    }

    const char* name() const { return "VerletNeighborList"; }

private:
    Real support_radius_;
    Real skin_distance_;
    Real total_radius_;

    size_t build_count_;
    size_t check_count_;
    bool needs_rebuild_;

    std::vector<NeighborPair> verlet_pairs_;
    std::vector<Real> ref_x_, ref_y_, ref_z_;
    std::vector<Real> max_displacement_;
};


// ============================================================================
// 6. SPH Thermal Coupling
// ============================================================================

/**
 * @brief Heat conduction between SPH particles
 *
 * Discretizes the heat equation using SPH:
 *   dT_i/dt = (k / rho / cp) * sum_j (m_j / rho_j) * (T_j - T_i)
 *             * (2 * |nabla W|) / |r_ij|
 *
 * This is the Brookshaw (1985) / Cleary & Monaghan (1999) Laplacian
 * approximation which gives second-order accuracy for the heat flux
 * across particle pairs.
 *
 * For multi-material problems, the interface conductivity uses the
 * harmonic mean: k_ij = 2 * k_i * k_j / (k_i + k_j).
 *
 * Reference: Cleary & Monaghan (1999), J. Comp. Phys.;
 *            Brookshaw (1985), Proc. Astro. Soc. Aust.
 */
class SPHThermalCoupling {
public:
    /**
     * @brief Construct thermal solver
     * @param conductivity Thermal conductivity (W/m/K)
     * @param specific_heat Specific heat capacity (J/kg/K)
     */
    SPHThermalCoupling(Real conductivity = 50.0, Real specific_heat = 500.0)
        : k_(conductivity), cp_(specific_heat) {}

    /**
     * @brief Compute rate of temperature change for a single particle
     *
     * @param idx Particle index
     * @param temperature Temperature array
     * @param mass Mass array
     * @param rho Density array
     * @param pos_x, pos_y, pos_z Position arrays
     * @param neighbor_list Neighbor indices for this particle
     * @param num_neighbors Number of neighbors
     * @param h Smoothing length
     * @param kernel SPH kernel
     * @return dT/dt for the particle
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_dTdt(Index idx,
                      const Real* temperature,
                      const Real* mass,
                      const Real* rho,
                      const Real* pos_x,
                      const Real* pos_y,
                      const Real* pos_z,
                      const Index* neighbor_list,
                      size_t num_neighbors,
                      Real h,
                      const SPHKernel& kernel) const {
        Real dTdt = 0.0;

        Real rho_i = rho[idx];
        Real T_i = temperature[idx];

        if (rho_i < 1.0e-20 || cp_ < 1.0e-20) return 0.0;

        Real alpha = k_ / (rho_i * cp_);  // Thermal diffusivity

        for (size_t k_idx = 0; k_idx < num_neighbors; ++k_idx) {
            Index j = neighbor_list[k_idx];

            Real rx = pos_x[idx] - pos_x[j];
            Real ry = pos_y[idx] - pos_y[j];
            Real rz = pos_z[idx] - pos_z[j];
            Real r = Kokkos::sqrt(rx * rx + ry * ry + rz * rz);

            if (r < 1.0e-20) continue;

            Real dWdr = kernel.grad_W(r, h);
            Real vol_j = mass[j] / rho[j];

            // Brookshaw Laplacian: 2 * dW/dr / r
            Real laplacian_coeff = 2.0 * dWdr / r;

            dTdt += vol_j * (temperature[j] - T_i) * laplacian_coeff;
        }

        return alpha * dTdt;
    }

    /**
     * @brief Compute rate of temperature change for a particle pair (symmetric)
     *
     * Used in pair-wise iteration. Accumulates dT/dt for both particles.
     *
     * @param T_i, T_j Temperatures
     * @param mass_i, mass_j Masses
     * @param rho_i, rho_j Densities
     * @param k_i, k_j Thermal conductivities
     * @param cp_i, cp_j Specific heats
     * @param r Distance between particles
     * @param grad_W_mag Kernel gradient magnitude |dW/dr|
     * @param dTdt_i Output dT/dt contribution for particle i
     * @param dTdt_j Output dT/dt contribution for particle j
     */
    KOKKOS_INLINE_FUNCTION
    static void compute_pair_heat_flux(Real T_i, Real T_j,
                                       Real mass_i, Real mass_j,
                                       Real rho_i, Real rho_j,
                                       Real k_i, Real k_j,
                                       Real cp_i, Real cp_j,
                                       Real r, Real grad_W_mag,
                                       Real& dTdt_i, Real& dTdt_j) {
        dTdt_i = dTdt_j = 0.0;

        if (r < 1.0e-20) return;
        if (rho_i < 1.0e-20 || rho_j < 1.0e-20) return;

        // Harmonic mean conductivity at interface
        Real k_ij = 0.0;
        if (k_i + k_j > 1.0e-30) {
            k_ij = 2.0 * k_i * k_j / (k_i + k_j);
        }

        // Laplacian coefficient (Brookshaw)
        Real laplacian = 2.0 * grad_W_mag / r;

        Real dT = T_j - T_i;

        // dT/dt_i = (k_ij / (rho_i * cp_i)) * (m_j / rho_j) * dT * laplacian
        Real vol_j = mass_j / rho_j;
        Real vol_i = mass_i / rho_i;

        if (cp_i > 1.0e-20) {
            dTdt_i = (k_ij / (rho_i * cp_i)) * vol_j * dT * laplacian;
        }
        if (cp_j > 1.0e-20) {
            dTdt_j = -(k_ij / (rho_j * cp_j)) * vol_i * dT * laplacian;
        }
    }

    /**
     * @brief Update temperatures for all particles over a timestep
     *
     * @param temperature Temperature array (updated in place)
     * @param mass Mass array
     * @param rho Density array
     * @param pos_x, pos_y, pos_z Position arrays
     * @param neighbor_pairs Neighbor pair list
     * @param h Smoothing length
     * @param kernel SPH kernel
     * @param dt Timestep
     * @param n Number of particles
     */
    void advance_temperatures(Real* temperature,
                              const Real* mass,
                              const Real* rho,
                              const Real* pos_x,
                              const Real* pos_y,
                              const Real* pos_z,
                              const std::vector<NeighborPair>& neighbor_pairs,
                              Real h,
                              const SPHKernel& kernel,
                              Real dt,
                              size_t n) const {
        // Compute dT/dt for each particle
        std::vector<Real> dTdt(n, 0.0);

        for (const auto& pair : neighbor_pairs) {
            Index i = pair.i;
            Index j = pair.j;

            Real grad_W_mag = std::abs(kernel.grad_W(pair.r, h));

            Real dTdt_i, dTdt_j;
            compute_pair_heat_flux(
                temperature[i], temperature[j],
                mass[i], mass[j],
                rho[i], rho[j],
                k_, k_,
                cp_, cp_,
                pair.r, grad_W_mag,
                dTdt_i, dTdt_j
            );

            dTdt[i] += dTdt_i;
            dTdt[j] += dTdt_j;
        }

        // Forward Euler update
        for (size_t i = 0; i < n; ++i) {
            temperature[i] += dTdt[i] * dt;
        }
    }

    void set_conductivity(Real k) { k_ = k; }
    void set_specific_heat(Real cp) { cp_ = cp; }

    Real conductivity() const { return k_; }
    Real specific_heat() const { return cp_; }
    const char* name() const { return "SPHThermalCoupling"; }

private:
    Real k_;   ///< Thermal conductivity (W/m/K)
    Real cp_;  ///< Specific heat capacity (J/kg/K)
};


// ============================================================================
// 7. SPH MUSCL Reconstruction
// ============================================================================

/**
 * @brief Second-order MUSCL gradient reconstruction for SPH
 *
 * Reconstructs left and right states at particle pair midpoints using
 * gradient information, enabling Riemann-like flux computation for
 * improved accuracy and shock capturing.
 *
 * For a scalar field phi:
 *   phi_L = phi_i + 0.5 * psi(r_L) * (grad_phi_i . r_ij)
 *   phi_R = phi_j - 0.5 * psi(r_R) * (grad_phi_j . r_ij)
 *
 * Where psi(r) is a slope limiter (minmod, van Leer, superbee, etc.)
 * and grad_phi is the SPH gradient estimate.
 *
 * Reference: Inutsuka (2002), J. Comp. Phys.;
 *            Vila (1999), Math. Models Meth. Appl. Sci.
 */
class SPHMUSCLReconstruction {
public:
    /// Slope limiter type
    enum class LimiterType {
        Minmod,      ///< Most diffusive, TVD
        VanLeer,     ///< Good compromise
        Superbee,    ///< Least diffusive, TVD
        MonotonizedCentral  ///< MC limiter
    };

    /**
     * @brief Construct MUSCL reconstructor
     * @param limiter Slope limiter type
     */
    SPHMUSCLReconstruction(LimiterType limiter = LimiterType::Minmod)
        : limiter_(limiter) {}

    /**
     * @brief Minmod slope limiter
     * psi(r) = max(0, min(1, r))
     */
    KOKKOS_INLINE_FUNCTION
    static Real minmod(Real r) {
        if (r <= 0.0) return 0.0;
        return (r < 1.0) ? r : 1.0;
    }

    /**
     * @brief Van Leer slope limiter
     * psi(r) = (r + |r|) / (1 + |r|)
     */
    KOKKOS_INLINE_FUNCTION
    static Real van_leer(Real r) {
        Real abs_r = (r > 0.0) ? r : -r;
        return (r + abs_r) / (1.0 + abs_r);
    }

    /**
     * @brief Superbee slope limiter
     * psi(r) = max(0, max(min(2r,1), min(r,2)))
     */
    KOKKOS_INLINE_FUNCTION
    static Real superbee(Real r) {
        if (r <= 0.0) return 0.0;
        Real a = (2.0 * r < 1.0) ? 2.0 * r : 1.0;
        Real b = (r < 2.0) ? r : 2.0;
        return (a > b) ? a : b;
    }

    /**
     * @brief Monotonized Central (MC) slope limiter
     * psi(r) = max(0, min(2r, (1+r)/2, 2))
     */
    KOKKOS_INLINE_FUNCTION
    static Real monotonized_central(Real r) {
        if (r <= 0.0) return 0.0;
        Real a = 2.0 * r;
        Real b = 0.5 * (1.0 + r);
        Real c = 2.0;
        Real m = (a < b) ? a : b;
        m = (m < c) ? m : c;
        return m;
    }

    /**
     * @brief Evaluate the selected slope limiter
     */
    KOKKOS_INLINE_FUNCTION
    Real limiter(Real r) const {
        switch (limiter_) {
            case LimiterType::Minmod:             return minmod(r);
            case LimiterType::VanLeer:            return van_leer(r);
            case LimiterType::Superbee:           return superbee(r);
            case LimiterType::MonotonizedCentral: return monotonized_central(r);
            default: return minmod(r);
        }
    }

    /**
     * @brief Compute SPH gradient of a scalar field at a particle
     *
     * grad_phi_i = sum_j (m_j / rho_j) * (phi_j - phi_i) * nabla_W_ij
     *
     * @param idx Particle index
     * @param phi Scalar field array
     * @param mass Mass array
     * @param rho Density array
     * @param pos_x, pos_y, pos_z Position arrays
     * @param neighbor_list Neighbor indices
     * @param num_neighbors Number of neighbors
     * @param h Smoothing length
     * @param kernel SPH kernel
     * @param grad_x, grad_y, grad_z Output gradient components
     */
    void compute_gradient(Index idx,
                          const Real* phi,
                          const Real* mass,
                          const Real* rho,
                          const Real* pos_x,
                          const Real* pos_y,
                          const Real* pos_z,
                          const Index* neighbor_list,
                          size_t num_neighbors,
                          Real h,
                          const SPHKernel& kernel,
                          Real& grad_x, Real& grad_y, Real& grad_z) const {
        grad_x = grad_y = grad_z = 0.0;

        Real phi_i = phi[idx];

        for (size_t k = 0; k < num_neighbors; ++k) {
            Index j = neighbor_list[k];

            Real rx = pos_x[idx] - pos_x[j];
            Real ry = pos_y[idx] - pos_y[j];
            Real rz = pos_z[idx] - pos_z[j];

            Real gWx, gWy, gWz;
            kernel.grad_W_vec(rx, ry, rz, h, gWx, gWy, gWz);

            Real vol_j = mass[j] / rho[j];
            Real dphi = phi[j] - phi_i;

            grad_x += vol_j * dphi * gWx;
            grad_y += vol_j * dphi * gWy;
            grad_z += vol_j * dphi * gWz;
        }
    }

    /**
     * @brief Reconstruct left and right states for a particle pair
     *
     * Given gradients at particles i and j, reconstruct the values
     * at the pair midpoint using MUSCL with slope limiting.
     *
     * @param phi_i, phi_j Field values at particles i and j
     * @param grad_i Gradient at particle i [3]
     * @param grad_j Gradient at particle j [3]
     * @param rx, ry, rz Vector r_i - r_j
     * @param phi_L Output left state (from particle i)
     * @param phi_R Output right state (from particle j)
     */
    KOKKOS_INLINE_FUNCTION
    void reconstruct(Real phi_i, Real phi_j,
                     const Real* grad_i, const Real* grad_j,
                     Real rx, Real ry, Real rz,
                     Real& phi_L, Real& phi_R) const {
        // Half vector for midpoint reconstruction
        Real hx = 0.5 * rx;
        Real hy = 0.5 * ry;
        Real hz = 0.5 * rz;

        // Unlimited extrapolation from i to midpoint
        Real delta_i = grad_i[0] * hx + grad_i[1] * hy + grad_i[2] * hz;

        // Unlimited extrapolation from j to midpoint
        Real delta_j = grad_j[0] * (-hx) + grad_j[1] * (-hy) + grad_j[2] * (-hz);

        // Central difference for limiter ratio
        Real dphi = phi_j - phi_i;

        // Compute slope ratios for limiting
        Real r_L = 1.0;
        Real r_R = 1.0;

        if (Kokkos::fabs(delta_i) > 1.0e-30) {
            r_L = dphi / (2.0 * delta_i);
        }
        if (Kokkos::fabs(delta_j) > 1.0e-30) {
            r_R = dphi / (2.0 * delta_j);
        }

        // Apply slope limiter
        Real psi_L = limiter(r_L);
        Real psi_R = limiter(r_R);

        // Reconstructed states
        phi_L = phi_i + psi_L * delta_i;
        phi_R = phi_j + psi_R * delta_j;
    }

    /**
     * @brief Compute MUSCL-reconstructed density for a particle pair
     *
     * Convenience wrapper for density reconstruction used in
     * Riemann-based SPH flux computation.
     *
     * @param rho_i, rho_j Densities at particles i and j
     * @param grad_rho_i Density gradient at i [3]
     * @param grad_rho_j Density gradient at j [3]
     * @param rx, ry, rz Pair vector
     * @param rho_L Output left-state density
     * @param rho_R Output right-state density
     */
    KOKKOS_INLINE_FUNCTION
    void reconstruct_density(Real rho_i, Real rho_j,
                             const Real* grad_rho_i, const Real* grad_rho_j,
                             Real rx, Real ry, Real rz,
                             Real& rho_L, Real& rho_R) const {
        reconstruct(rho_i, rho_j, grad_rho_i, grad_rho_j,
                    rx, ry, rz, rho_L, rho_R);

        // Enforce positivity
        if (rho_L < 1.0e-10) rho_L = 1.0e-10;
        if (rho_R < 1.0e-10) rho_R = 1.0e-10;
    }

    /**
     * @brief Compute MUSCL-reconstructed velocity for a particle pair
     *
     * Performs reconstruction on each velocity component independently.
     *
     * @param vel_i Velocity of particle i [3]
     * @param vel_j Velocity of particle j [3]
     * @param grad_vx_i, grad_vy_i, grad_vz_i Velocity gradients at i
     * @param grad_vx_j, grad_vy_j, grad_vz_j Velocity gradients at j
     * @param rx, ry, rz Pair vector
     * @param vel_L Output left-state velocity [3]
     * @param vel_R Output right-state velocity [3]
     */
    void reconstruct_velocity(const Real* vel_i, const Real* vel_j,
                              const Real* grad_vx_i, const Real* grad_vy_i,
                              const Real* grad_vz_i,
                              const Real* grad_vx_j, const Real* grad_vy_j,
                              const Real* grad_vz_j,
                              Real rx, Real ry, Real rz,
                              Real* vel_L, Real* vel_R) const {
        reconstruct(vel_i[0], vel_j[0], grad_vx_i, grad_vx_j,
                    rx, ry, rz, vel_L[0], vel_R[0]);
        reconstruct(vel_i[1], vel_j[1], grad_vy_i, grad_vy_j,
                    rx, ry, rz, vel_L[1], vel_R[1]);
        reconstruct(vel_i[2], vel_j[2], grad_vz_i, grad_vz_j,
                    rx, ry, rz, vel_L[2], vel_R[2]);
    }

    void set_limiter(LimiterType type) { limiter_ = type; }
    LimiterType limiter_type() const { return limiter_; }
    const char* name() const { return "SPHMUSCLReconstruction"; }

private:
    LimiterType limiter_;
};

} // namespace sph
} // namespace nxs
