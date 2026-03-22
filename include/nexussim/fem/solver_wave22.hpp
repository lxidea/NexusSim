#pragma once

/**
 * @file solver_wave22.hpp
 * @brief Wave 22: Solver hardening features for explicit dynamics
 *
 * Components:
 * 1. MassScaling          - Selective mass scaling for explicit timestep efficiency
 * 2. Subcycling           - Multi-rate time integration (fast/slow element groups)
 * 3. AddedMassFluid       - Virtual mass for fluid-structure interaction
 * 4. DynamicRelaxation    - Quasi-static analysis via damped explicit dynamics
 * 5. SmoothParticleContact - Node-to-surface contact with smoothed normals
 *
 * References:
 * - Olovsson et al. (2005) "Selective Mass Scaling for Explicit FE Analyses"
 * - Belytschko et al. (2000) "Nonlinear Finite Elements for Continua and Structures"
 * - Cundall (1971) "A Computer Model for Simulating Progressive Large Scale Movements"
 * - Day & Pothen (1995) "Multi-rate Time Integration Methods"
 */

#include <nexussim/core/types.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <limits>
#include <functional>

namespace nxs {
namespace fem {

using Real = nxs::Real;

// ============================================================================
// 1. MassScaling - Selective Mass Scaling for Explicit Solver Efficiency
// ============================================================================

/**
 * @brief Selective mass scaling for explicit dynamics
 *
 * In explicit time integration, the critical timestep is governed by the smallest
 * element: dt_crit = L_min / c. Small elements force tiny global timesteps.
 * Mass scaling artificially increases the mass of small elements to raise their
 * critical timestep to a target dt, without affecting larger elements.
 *
 * The scaling factor for element e is:
 *   alpha_e = max(1, (c * dt_target / L_e)^2)
 *   m_scaled_e = alpha_e * m_e
 *
 * The total momentum is preserved: only mass increases, velocity is not changed.
 * Warning: excessive mass scaling (added mass ratio > 5%) can corrupt inertial response.
 */
class MassScaling {
public:
    /// Result of mass scaling computation
    struct ScaleResult {
        std::vector<Real> scale_factors;   ///< Per-element scale factor (>= 1.0)
        Real total_added_mass;             ///< Total mass added across all elements
        Real total_original_mass;          ///< Sum of original element masses
        Real added_mass_ratio;             ///< total_added_mass / total_original_mass
        int num_scaled_elements;           ///< Number of elements that needed scaling
    };

    /**
     * @brief Compute mass scaling factors to achieve a target timestep
     *
     * For each element, if its characteristic timestep L_e/c_e < dt_target,
     * increase its mass by factor alpha = (c_e * dt_target / L_e)^2.
     *
     * @param element_sizes Characteristic length for each element
     * @param sound_speeds Sound speed in each element
     * @param densities Current density of each element
     * @param volumes Element volumes
     * @param target_dt Target critical timestep
     * @param num_elements Number of elements
     * @param mass_increase Output: per-element mass increase (size = num_elements)
     * @return ScaleResult with per-element factors and statistics
     */
    static ScaleResult compute_scaled_mass(
        const Real* element_sizes, const Real* sound_speeds,
        const Real* densities, const Real* volumes,
        Real target_dt, int num_elements,
        Real* mass_increase)
    {
        ScaleResult res;
        res.scale_factors.resize(num_elements, 1.0);
        res.total_added_mass = 0.0;
        res.total_original_mass = 0.0;
        res.num_scaled_elements = 0;

        for (int e = 0; e < num_elements; ++e) {
            Real L = element_sizes[e];
            Real c = sound_speeds[e];
            Real rho = densities[e];
            Real vol = volumes[e];
            Real m_orig = rho * vol;
            res.total_original_mass += m_orig;

            mass_increase[e] = 0.0;

            if (L <= 0.0 || c <= 0.0) {
                res.scale_factors[e] = 1.0;
                continue;
            }

            // Element critical timestep (CFL = 1.0)
            Real dt_elem = L / c;

            if (dt_elem < target_dt) {
                // Scale mass: alpha = (dt_target / dt_elem)^2
                // Because c = sqrt(K/rho), scaling rho by alpha^2 makes
                // c_new = c/alpha, so dt_new = L/c_new = L*alpha/c
                // We want dt_new = dt_target, so alpha = (c * dt_target / L)
                // But mass scales as alpha^2 of the density ratio:
                Real ratio = target_dt / dt_elem;
                Real alpha = ratio * ratio;
                res.scale_factors[e] = alpha;
                Real added = m_orig * (alpha - 1.0);
                mass_increase[e] = added;
                res.total_added_mass += added;
                res.num_scaled_elements++;
            } else {
                res.scale_factors[e] = 1.0;
            }
        }

        res.added_mass_ratio = (res.total_original_mass > 1e-30)
            ? res.total_added_mass / res.total_original_mass : 0.0;

        return res;
    }

    /**
     * @brief Apply mass scaling factors to a lumped mass array
     *
     * Distributes the added mass equally among the element's DOFs.
     *
     * @param masses Lumped mass array (modified in place), size = num_dof
     * @param scale_factors Per-element scale factors
     * @param elem_to_dof Flat connectivity: elem_to_dof[e*ndpe + d] = global dof index
     * @param num_dof_per_elem Number of DOFs per element
     * @param num_elements Number of elements
     * @param original_elem_mass Original element mass (before scaling)
     */
    static void apply_mass_scaling(
        Real* masses, const Real* scale_factors,
        const int* elem_to_dof, int num_dof_per_elem, int num_elements,
        const Real* original_elem_mass)
    {
        for (int e = 0; e < num_elements; ++e) {
            Real factor = scale_factors[e];
            if (factor <= 1.0) continue;

            Real added = original_elem_mass[e] * (factor - 1.0);
            Real per_dof = added / static_cast<Real>(num_dof_per_elem);

            for (int d = 0; d < num_dof_per_elem; ++d) {
                int dof = elem_to_dof[e * num_dof_per_elem + d];
                if (dof >= 0) {
                    masses[dof] += per_dof;
                }
            }
        }
    }

    /**
     * @brief Compute the ratio of added mass to original mass
     * @param original_mass Sum of all original element masses
     * @param scaled_mass Sum of all scaled element masses
     * @return Added mass fraction (0 = no scaling)
     */
    static Real total_added_mass_ratio(Real original_mass, Real scaled_mass) {
        if (original_mass <= 1e-30) return 0.0;
        return (scaled_mass - original_mass) / original_mass;
    }
};

// ============================================================================
// 2. Subcycling - Multi-Rate Time Integration
// ============================================================================

/**
 * @brief Multi-rate time integration (subcycling)
 *
 * Elements are grouped by their critical timestep ratio relative to the global dt.
 * Fast elements (small dt) take multiple sub-steps per global step; slow elements
 * take one step. Groups are synchronized at each global timestep.
 *
 * Group assignment: ratio = nearest power-of-2 of ceil(dt_global / dt_elem).
 *
 * At each global step:
 *   Group 1 (ratio=1): 1 step of dt_global
 *   Group 2 (ratio=2): 2 steps of dt_global/2
 *   Group 4 (ratio=4): 4 steps of dt_global/4
 *
 * Inter-group coupling uses velocity interpolation at synchronization points.
 */
class Subcycling {
public:
    /// A group of elements sharing the same subcycle ratio
    struct ElementGroup {
        int ratio;                    ///< Number of sub-steps per global step (power of 2)
        std::vector<int> element_ids; ///< Elements in this group
        Real sub_dt;                  ///< Sub-step = dt_global / ratio
    };

    /**
     * @brief Group elements by their timestep ratio
     *
     * @param element_dt Critical timestep per element
     * @param global_dt Global timestep
     * @param num_elements Number of elements
     * @return Vector of element groups, sorted by ratio (ascending)
     */
    static std::vector<ElementGroup> group_elements(
        const Real* element_dt, int num_elements, Real global_dt)
    {
        std::map<int, std::vector<int>> ratio_map;

        for (int e = 0; e < num_elements; ++e) {
            if (element_dt[e] <= 0.0 || global_dt <= 0.0) {
                ratio_map[1].push_back(e);
                continue;
            }

            Real raw = global_dt / element_dt[e];
            if (raw <= 1.0) {
                ratio_map[1].push_back(e);
                continue;
            }

            // Round up to nearest power of 2
            int r = 1;
            while (r < static_cast<int>(std::ceil(raw))) {
                r *= 2;
                if (r > 256) { r = 256; break; }
            }
            ratio_map[r].push_back(e);
        }

        std::vector<ElementGroup> groups;
        for (auto& [ratio, ids] : ratio_map) {
            ElementGroup g;
            g.ratio = ratio;
            g.element_ids = std::move(ids);
            g.sub_dt = global_dt / static_cast<Real>(ratio);
            groups.push_back(g);
        }

        std::sort(groups.begin(), groups.end(),
                  [](const ElementGroup& a, const ElementGroup& b) {
                      return a.ratio < b.ratio;
                  });

        return groups;
    }

    /**
     * @brief Execute one global timestep with subcycling
     *
     * Each group advances through (ratio) sub-steps. The finest group determines
     * the resolution; coarser groups participate only on their synchronization points.
     *
     * Central difference per sub-step:
     *   v^{n+1/2} = v^{n-1/2} + (dt_sub / m) * f^n
     *   x^{n+1}   = x^n + dt_sub * v^{n+1/2}
     *
     * @param groups Element groups from group_elements()
     * @param positions Node positions (flat 3*N), modified
     * @param velocities Node velocities (flat 3*N), modified
     * @param masses Lumped masses (flat 3*N)
     * @param num_nodes Number of nodes
     * @param dt_global Global timestep
     * @param compute_forces Callback: given element IDs, current positions,
     *        computes and accumulates forces.
     *        Signature: void(const vector<int>& elem_ids, const Real* pos, Real* forces, int ndof)
     */
    static void subcycle_step(
        const std::vector<ElementGroup>& groups,
        Real* positions, Real* velocities, const Real* masses,
        int num_nodes, Real dt_global,
        std::function<void(const std::vector<int>&, const Real*, Real*, int)> compute_forces)
    {
        int ndof = num_nodes * 3;
        std::vector<Real> forces(ndof, 0.0);

        // Find the maximum ratio (finest resolution)
        int max_ratio = 1;
        for (const auto& g : groups) {
            max_ratio = std::max(max_ratio, g.ratio);
        }

        Real dt_fine = dt_global / static_cast<Real>(max_ratio);

        // Execute max_ratio sub-steps
        for (int s = 0; s < max_ratio; ++s) {
            std::fill(forces.begin(), forces.end(), 0.0);

            // Accumulate forces from groups active at this sub-step
            for (const auto& g : groups) {
                int stride = max_ratio / g.ratio;
                if ((s % stride) != 0) continue;
                compute_forces(g.element_ids, positions, forces.data(), ndof);
            }

            // Update velocities and positions
            for (int d = 0; d < ndof; ++d) {
                if (masses[d] > 1e-30) {
                    velocities[d] += (dt_fine / masses[d]) * forces[d];
                }
                positions[d] += dt_fine * velocities[d];
            }
        }
    }
};

// ============================================================================
// 3. AddedMassFluid - Added Mass for Fluid-Structure Interaction
// ============================================================================

/**
 * @brief Added mass (virtual mass) for fluid-structure interaction
 *
 * When a structure accelerates through fluid, it entrains surrounding fluid,
 * creating an effective mass increase:
 *
 *   M_eff = M_struct + C_a * rho_fluid * V_displaced
 *
 * where C_a is the added mass coefficient:
 *   Sphere:       C_a = 0.5
 *   Long cylinder: C_a = 1.0
 *   Flat plate:   C_a ~ 0 (parallel) to ~1 (perpendicular)
 */
class AddedMassFluid {
public:
    /// Per-node added mass computation result
    struct Result {
        std::vector<Real> added_mass;  ///< Per-node added mass
        Real total_added;              ///< Sum of all added mass
        Real total_struct;             ///< Sum of structural mass
        Real ratio;                    ///< total_added / total_struct
    };

    /**
     * @brief Compute added mass for structural nodes in a fluid
     *
     * @param struct_mass Structural mass at each node (num_nodes)
     * @param displaced_volume Fluid volume displaced per node (num_nodes)
     * @param fluid_density Density of the surrounding fluid
     * @param Ca Added mass coefficient
     * @param num_nodes Number of structural nodes
     * @return Result with per-node added mass
     */
    static Result compute_added_mass(
        const Real* struct_mass, const Real* displaced_volume,
        Real fluid_density, Real Ca, int num_nodes)
    {
        Result res;
        res.added_mass.resize(num_nodes, 0.0);
        res.total_added = 0.0;
        res.total_struct = 0.0;

        for (int i = 0; i < num_nodes; ++i) {
            res.total_struct += struct_mass[i];
            res.added_mass[i] = Ca * fluid_density * displaced_volume[i];
            res.total_added += res.added_mass[i];
        }

        res.ratio = (res.total_struct > 1e-30) ? res.total_added / res.total_struct : 0.0;
        return res;
    }

    /**
     * @brief Apply added mass to the lumped mass matrix
     *
     * For 3 translational DOFs per node, the same added mass applies to all three.
     *
     * @param mass_matrix Lumped mass (3*num_nodes), modified in place
     * @param added_mass Per-node added mass (num_nodes)
     * @param num_nodes Number of nodes
     */
    static void apply_to_mass_matrix(Real* mass_matrix, const Real* added_mass, int num_nodes) {
        for (int n = 0; n < num_nodes; ++n) {
            Real am = added_mass[n];
            mass_matrix[3 * n + 0] += am;
            mass_matrix[3 * n + 1] += am;
            mass_matrix[3 * n + 2] += am;
        }
    }

    /**
     * @brief Estimate displaced volume for a surface mesh node
     *
     * Uses a simple area*length approximation: V ~ A_node * L_char
     *
     * @param node_surface_area Surface area contribution of this node
     * @param char_length Characteristic length (e.g. local element size)
     * @return Estimated displaced volume
     */
    static Real estimate_displaced_volume(Real node_surface_area, Real char_length) {
        return node_surface_area * char_length;
    }
};

// ============================================================================
// 4. DynamicRelaxation - Quasi-Static via Damped Explicit Dynamics
// ============================================================================

/**
 * @brief Dynamic relaxation for quasi-static analysis
 *
 * Achieves static equilibrium through explicit dynamics with damping.
 * Kinetic energy is monitored; when it peaks (velocity reversal), all
 * velocities are zeroed and damping is adjusted. Convergence is reached
 * when KE/IE < tolerance.
 *
 * Algorithm:
 * 1. Start with v=0, apply loads
 * 2. March forward with central difference
 * 3. Apply viscous damping: v *= (1 - damping_factor)
 * 4. Monitor KE. On KE peak: if KE/IE < tol -> converged; else reset v=0
 * 5. Adaptive damping: increase if KE oscillates, decrease if monotonic
 */
class DynamicRelaxation {
public:
    /// Convergence status
    struct ConvergenceInfo {
        bool converged;
        Real kinetic_energy;
        Real internal_energy;
        Real ke_ie_ratio;
        int iteration;
        Real damping_factor;
    };

    DynamicRelaxation()
        : damping_(0.9), tol_(1e-6), max_iter_(100000),
          hist_size_(10), adaptive_(true), iter_(0) {}

    void set_damping(Real d) { damping_ = d; }
    void set_tolerance(Real t) { tol_ = t; }
    void set_max_iterations(int m) { max_iter_ = m; }
    void set_adaptive(bool a) { adaptive_ = a; }

    /**
     * @brief Apply viscous damping to velocities
     * @param velocities Velocity array (modified in place)
     * @param num_dof Number of DOFs
     * @param damping_factor Factor in [0,1]; 0=no damping, 1=full stop
     */
    static void apply_damping(Real* velocities, int num_dof, Real damping_factor) {
        Real f = 1.0 - damping_factor;
        for (int i = 0; i < num_dof; ++i) {
            velocities[i] *= f;
        }
    }

    /**
     * @brief Check if converged: KE/IE < tolerance
     * @param kinetic_energy Current kinetic energy
     * @param internal_energy Current strain energy
     * @param tolerance Convergence threshold
     * @return true if converged
     */
    static bool check_convergence(Real kinetic_energy, Real internal_energy, Real tolerance) {
        if (internal_energy < 1e-30) {
            return (kinetic_energy < 1e-30);
        }
        return (kinetic_energy / internal_energy) < tolerance;
    }

    /**
     * @brief Compute adaptive damping factor from KE history
     *
     * Counts sign changes in KE differences to detect oscillation.
     * High oscillation -> more damping; monotonic decrease -> less damping.
     *
     * @param KE_history Recent KE values (newest last)
     * @return Adjusted damping factor
     */
    static Real adaptive_damping(const std::vector<Real>& KE_history) {
        if (KE_history.size() < 3) return 0.9;

        int n = static_cast<int>(KE_history.size());
        int sign_changes = 0;
        for (int i = 2; i < n; ++i) {
            Real d1 = KE_history[i - 1] - KE_history[i - 2];
            Real d2 = KE_history[i] - KE_history[i - 1];
            if (d1 * d2 < 0.0) sign_changes++;
        }

        Real osc_ratio = static_cast<Real>(sign_changes) / static_cast<Real>(n - 2);

        // High oscillation: increase damping toward 0.99
        // Low oscillation: decrease damping toward 0.5
        if (osc_ratio > 0.5) {
            return std::min(0.99, 0.9 + 0.1 * osc_ratio);
        } else {
            return std::max(0.5, 0.9 - 0.4 * (1.0 - osc_ratio));
        }
    }

    /**
     * @brief Execute one relaxation step
     *
     * Updates velocities (with damping), positions, checks for KE peak,
     * and optionally adjusts damping.
     *
     * @param velocities Velocity array (3*N), modified
     * @param forces Force array (3*N): internal + external
     * @param masses Lumped mass array (3*N)
     * @param positions Position array (3*N), modified
     * @param num_nodes Number of nodes
     * @param dt Timestep
     * @param internal_energy Current strain energy for convergence check
     * @return ConvergenceInfo
     */
    ConvergenceInfo step(Real* velocities, const Real* forces, const Real* masses,
                          Real* positions, int num_nodes, Real dt,
                          Real internal_energy) {
        int ndof = num_nodes * 3;
        iter_++;

        // Central difference update
        for (int i = 0; i < ndof; ++i) {
            if (masses[i] > 1e-30) {
                velocities[i] += (dt / masses[i]) * forces[i];
            }
        }

        // Viscous damping
        apply_damping(velocities, ndof, damping_);

        // Position update
        for (int i = 0; i < ndof; ++i) {
            positions[i] += dt * velocities[i];
        }

        // Compute KE
        Real ke = 0.0;
        for (int i = 0; i < ndof; ++i) {
            ke += 0.5 * masses[i] * velocities[i] * velocities[i];
        }

        // KE history for adaptive damping
        ke_hist_.push_back(ke);
        if (static_cast<int>(ke_hist_.size()) > hist_size_) {
            ke_hist_.erase(ke_hist_.begin());
        }

        // Kinetic damping: detect KE peak and reset velocities
        if (ke_hist_.size() >= 3) {
            size_t s = ke_hist_.size();
            Real prev2 = ke_hist_[s - 3];
            Real prev1 = ke_hist_[s - 2];
            Real curr  = ke_hist_[s - 1];
            if (prev1 > prev2 && prev1 > curr) {
                // Peak at prev1: zero velocities
                std::fill(velocities, velocities + ndof, 0.0);
                ke = 0.0;
            }
        }

        // Adaptive damping
        if (adaptive_ && ke_hist_.size() >= 3) {
            damping_ = adaptive_damping(ke_hist_);
        }

        ConvergenceInfo info;
        info.converged = check_convergence(ke, internal_energy, tol_);
        info.kinetic_energy = ke;
        info.internal_energy = internal_energy;
        info.ke_ie_ratio = (internal_energy > 1e-30) ? ke / internal_energy : 0.0;
        info.iteration = iter_;
        info.damping_factor = damping_;
        return info;
    }

    void reset() { iter_ = 0; ke_hist_.clear(); }
    int iteration() const { return iter_; }
    Real damping_factor() const { return damping_; }

private:
    Real damping_;
    Real tol_;
    int max_iter_;
    int hist_size_;
    bool adaptive_;
    int iter_;
    std::vector<Real> ke_hist_;
};

// ============================================================================
// 5. SmoothParticleContact - Contact with Smoothed Surface Normals
// ============================================================================

/**
 * @brief Smooth particle contact for large sliding
 *
 * Standard node-to-surface contact uses facet normals, which are discontinuous
 * at element edges. This causes chattering as a node slides across edges.
 *
 * Smoothed normals are the area-weighted average of surrounding face normals
 * at each surface node. The contact gap and penalty force use these smooth
 * normals, preventing discontinuous force jumps.
 *
 * Algorithm:
 * 1. For each surface node, average adjacent face normals (area-weighted)
 * 2. Project contacting node onto the smooth surface
 * 3. Compute gap using smooth normal
 * 4. Apply penalty: F = -k * gap * n_smooth  (if gap < 0)
 */
class SmoothParticleContact {
public:
    /// Surface face
    struct Face {
        int node_ids[4];  ///< node_ids[3]=0 for triangle
        int num_nodes;    ///< 3 or 4
        Real normal[3];   ///< Unit face normal
        Real area;        ///< Face area
        Real centroid[3]; ///< Face centroid
    };

    /// Contact result for one node
    struct ContactResult {
        bool in_contact;
        Real gap;
        Real force[3];
        Real smooth_normal[3];
    };

    /**
     * @brief Compute smoothed normal at a surface node
     *
     * n_smooth = sum(A_f * n_f) / |sum(A_f * n_f)|
     * where sum is over all faces sharing the node.
     *
     * @param node_id The surface node
     * @param surface_faces Faces on the contact surface
     * @param num_faces Number of faces
     * @param smooth_normal Output: smoothed unit normal [3]
     */
    static void compute_smooth_normal(
        int node_id, const Face* surface_faces, int num_faces,
        Real smooth_normal[3])
    {
        smooth_normal[0] = smooth_normal[1] = smooth_normal[2] = 0.0;

        for (int f = 0; f < num_faces; ++f) {
            const auto& face = surface_faces[f];
            bool adj = false;
            for (int k = 0; k < face.num_nodes; ++k) {
                if (face.node_ids[k] == node_id) { adj = true; break; }
            }
            if (!adj) continue;

            smooth_normal[0] += face.area * face.normal[0];
            smooth_normal[1] += face.area * face.normal[1];
            smooth_normal[2] += face.area * face.normal[2];
        }

        Real mag = std::sqrt(smooth_normal[0] * smooth_normal[0]
                           + smooth_normal[1] * smooth_normal[1]
                           + smooth_normal[2] * smooth_normal[2]);
        if (mag > 1e-30) {
            smooth_normal[0] /= mag;
            smooth_normal[1] /= mag;
            smooth_normal[2] /= mag;
        }
    }

    /**
     * @brief Project a node onto the smooth surface
     *
     * Finds the closest face centroid, then projects along the smooth normal.
     * Gap = signed distance (negative = penetration).
     *
     * @param node_pos Contacting node position [3]
     * @param surface_faces Surface faces
     * @param num_faces Number of faces
     * @param smooth_normal Smoothed normal [3]
     * @param projected_pos Output: projected point on surface [3]
     * @return Signed gap distance
     */
    static Real project_to_smooth_surface(
        const Real node_pos[3],
        const Face* surface_faces, int num_faces,
        const Real smooth_normal[3],
        Real projected_pos[3])
    {
        // Find closest face by centroid distance
        Real best_dist = std::numeric_limits<Real>::max();
        int best = -1;
        for (int f = 0; f < num_faces; ++f) {
            Real dx = node_pos[0] - surface_faces[f].centroid[0];
            Real dy = node_pos[1] - surface_faces[f].centroid[1];
            Real dz = node_pos[2] - surface_faces[f].centroid[2];
            Real d = dx * dx + dy * dy + dz * dz;
            if (d < best_dist) { best_dist = d; best = f; }
        }

        if (best < 0) {
            projected_pos[0] = node_pos[0];
            projected_pos[1] = node_pos[1];
            projected_pos[2] = node_pos[2];
            return 1.0;
        }

        const auto& face = surface_faces[best];
        Real dx = node_pos[0] - face.centroid[0];
        Real dy = node_pos[1] - face.centroid[1];
        Real dz = node_pos[2] - face.centroid[2];

        Real gap = dx * smooth_normal[0] + dy * smooth_normal[1] + dz * smooth_normal[2];

        projected_pos[0] = node_pos[0] - gap * smooth_normal[0];
        projected_pos[1] = node_pos[1] - gap * smooth_normal[1];
        projected_pos[2] = node_pos[2] - gap * smooth_normal[2];

        return gap;
    }

    /**
     * @brief Compute penalty contact force
     *
     * F = -penalty * gap * n_smooth  (when gap < 0)
     * F = 0                          (when gap >= 0)
     *
     * @param gap Signed gap from project_to_smooth_surface
     * @param smooth_normal Smoothed normal [3]
     * @param penalty Penalty stiffness (force / length)
     * @return ContactResult
     */
    static ContactResult compute_contact_force(
        Real gap, const Real smooth_normal[3], Real penalty)
    {
        ContactResult res;
        res.gap = gap;
        res.smooth_normal[0] = smooth_normal[0];
        res.smooth_normal[1] = smooth_normal[1];
        res.smooth_normal[2] = smooth_normal[2];

        if (gap < 0.0) {
            res.in_contact = true;
            Real f_mag = -penalty * gap;
            res.force[0] = f_mag * smooth_normal[0];
            res.force[1] = f_mag * smooth_normal[1];
            res.force[2] = f_mag * smooth_normal[2];
        } else {
            res.in_contact = false;
            res.force[0] = res.force[1] = res.force[2] = 0.0;
        }

        return res;
    }

    /**
     * @brief Compute face properties (normal, area, centroid) from coordinates
     *
     * @param face Face struct with node_ids and num_nodes set
     * @param node_x, node_y, node_z Coordinate arrays
     * @param node_map node_id -> array index
     */
    static void compute_face_properties(
        Face& face,
        const Real* node_x, const Real* node_y, const Real* node_z,
        const std::map<int, int>& node_map)
    {
        face.normal[0] = face.normal[1] = face.normal[2] = 0.0;
        face.area = 0.0;
        face.centroid[0] = face.centroid[1] = face.centroid[2] = 0.0;

        Real px[4], py[4], pz[4];
        int n = face.num_nodes;
        for (int k = 0; k < n; ++k) {
            auto it = node_map.find(face.node_ids[k]);
            if (it == node_map.end()) return;
            int idx = it->second;
            px[k] = node_x[idx]; py[k] = node_y[idx]; pz[k] = node_z[idx];
        }

        // Centroid
        for (int k = 0; k < n; ++k) {
            face.centroid[0] += px[k];
            face.centroid[1] += py[k];
            face.centroid[2] += pz[k];
        }
        face.centroid[0] /= n; face.centroid[1] /= n; face.centroid[2] /= n;

        // Normal via cross product of edges 0-1 and 0-2
        Real ax = px[1] - px[0], ay = py[1] - py[0], az = pz[1] - pz[0];
        Real bx = px[2] - px[0], by = py[2] - py[0], bz = pz[2] - pz[0];

        Real nx = ay * bz - az * by;
        Real ny = az * bx - ax * bz;
        Real nz = ax * by - ay * bx;

        Real mag = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (mag > 1e-30) {
            face.normal[0] = nx / mag;
            face.normal[1] = ny / mag;
            face.normal[2] = nz / mag;
        }

        // Area
        if (n == 3) {
            face.area = 0.5 * mag;
        } else if (n == 4) {
            // Two triangles: 0-1-2 and 0-2-3
            Real cx = px[2] - px[0], cy = py[2] - py[0], cz = pz[2] - pz[0];
            Real dx = px[3] - px[0], dy = py[3] - py[0], dz = pz[3] - pz[0];
            Real n2x = cy * dz - cz * dy;
            Real n2y = cz * dx - cx * dz;
            Real n2z = cx * dy - cy * dx;
            Real mag2 = std::sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
            face.area = 0.5 * (mag + mag2);
        }
    }

    /**
     * @brief Process all slave nodes against a master surface
     *
     * For each slave node: find closest master face, compute smooth normal
     * from faces sharing that face's nodes, project, and compute penalty force.
     *
     * @param slave_pos Slave node positions (3*N flat)
     * @param num_slaves Number of slave nodes
     * @param master_faces Master surface faces with precomputed properties
     * @param num_master Number of master faces
     * @param penalty Penalty stiffness
     * @param slave_forces Output (3*N flat)
     * @return Number of slave nodes in contact
     */
    static int process_contact(
        const Real* slave_pos, int num_slaves,
        const Face* master_faces, int num_master,
        Real penalty, Real* slave_forces)
    {
        int count = 0;

        for (int s = 0; s < num_slaves; ++s) {
            const Real* pos = slave_pos + 3 * s;
            Real* frc = slave_forces + 3 * s;
            frc[0] = frc[1] = frc[2] = 0.0;

            // Find closest master face
            Real min_d = std::numeric_limits<Real>::max();
            int closest = -1;
            for (int f = 0; f < num_master; ++f) {
                Real dx = pos[0] - master_faces[f].centroid[0];
                Real dy = pos[1] - master_faces[f].centroid[1];
                Real dz = pos[2] - master_faces[f].centroid[2];
                Real d = dx * dx + dy * dy + dz * dz;
                if (d < min_d) { min_d = d; closest = f; }
            }
            if (closest < 0) continue;

            const auto& cf = master_faces[closest];

            // Collect face nodes
            std::set<int> fn;
            for (int k = 0; k < cf.num_nodes; ++k) fn.insert(cf.node_ids[k]);

            // Smooth normal: area-weighted average of faces sharing any node with closest
            Real sn[3] = {0.0, 0.0, 0.0};
            for (int f = 0; f < num_master; ++f) {
                bool shares = false;
                for (int k = 0; k < master_faces[f].num_nodes; ++k) {
                    if (fn.count(master_faces[f].node_ids[k])) { shares = true; break; }
                }
                if (shares) {
                    sn[0] += master_faces[f].area * master_faces[f].normal[0];
                    sn[1] += master_faces[f].area * master_faces[f].normal[1];
                    sn[2] += master_faces[f].area * master_faces[f].normal[2];
                }
            }

            Real mag = std::sqrt(sn[0] * sn[0] + sn[1] * sn[1] + sn[2] * sn[2]);
            if (mag > 1e-30) {
                sn[0] /= mag; sn[1] /= mag; sn[2] /= mag;
            } else {
                sn[0] = cf.normal[0]; sn[1] = cf.normal[1]; sn[2] = cf.normal[2];
            }

            // Gap
            Real dx = pos[0] - cf.centroid[0];
            Real dy = pos[1] - cf.centroid[1];
            Real dz = pos[2] - cf.centroid[2];
            Real gap = dx * sn[0] + dy * sn[1] + dz * sn[2];

            if (gap < 0.0) {
                Real fm = -penalty * gap;
                frc[0] = fm * sn[0];
                frc[1] = fm * sn[1];
                frc[2] = fm * sn[2];
                count++;
            }
        }

        return count;
    }
};

} // namespace fem
} // namespace nxs
