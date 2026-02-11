#pragma once

/**
 * @file ale_solver.hpp
 * @brief Arbitrary Lagrangian-Eulerian (ALE) mesh management
 *
 * ALE decouples mesh motion from material motion to:
 * - Prevent mesh tangling in large deformation problems
 * - Allow fluid-structure interaction
 * - Enable multi-material element handling
 *
 * ALE Step:
 *   1. Lagrangian step (standard FEM update)
 *   2. Mesh smoothing (relocate nodes to improve element quality)
 *   3. Advection (remap state variables to new mesh configuration)
 *
 * Smoothing methods:
 * - Laplacian: x_new = average of neighbor positions
 * - Equipotential: solve Laplace equation on mesh
 * - Weighted: boundary-aware with element quality weighting
 *
 * Advection methods:
 * - First-order donor cell (simple, diffusive)
 * - Van Leer (second-order, less diffusive)
 *
 * Reference:
 * - Donea et al., "Arbitrary Lagrangian-Eulerian Methods" (2004)
 * - LS-DYNA *ALE_SMOOTHING, *ALE_MULTI-MATERIAL_GROUP
 */

#include <nexussim/core/types.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>

namespace nxs {
namespace physics {

// ============================================================================
// ALE Configuration
// ============================================================================

enum class SmoothingMethod {
    Laplacian,       ///< Simple average of neighbors
    Weighted,        ///< Quality-weighted averaging
    Equipotential    ///< Laplace equation-based
};

enum class AdvectionMethod {
    DonorCell,       ///< First-order upwind
    VanLeer          ///< Second-order (Van Leer limiter)
};

struct ALEConfig {
    SmoothingMethod smoothing;
    AdvectionMethod advection;

    Real smoothing_weight;       ///< Relaxation factor (0=no smoothing, 1=full)
    int smoothing_iterations;    ///< Number of smoothing passes per step
    bool boundary_fixed;         ///< Fix boundary nodes during smoothing

    // Quality thresholds
    Real min_jacobian;           ///< Minimum acceptable Jacobian ratio
    Real target_aspect_ratio;    ///< Target element aspect ratio

    ALEConfig()
        : smoothing(SmoothingMethod::Laplacian)
        , advection(AdvectionMethod::DonorCell)
        , smoothing_weight(0.5)
        , smoothing_iterations(1)
        , boundary_fixed(true)
        , min_jacobian(0.1)
        , target_aspect_ratio(3.0) {}
};

// ============================================================================
// Mesh Connectivity for ALE
// ============================================================================

/**
 * @brief Node-to-node adjacency for mesh smoothing
 */
class MeshAdjacency {
public:
    MeshAdjacency() = default;

    /**
     * @brief Build adjacency from element connectivity
     * @param num_nodes Total number of nodes
     * @param num_elements Total number of elements
     * @param connectivity Element connectivity (flat: nodes_per_elem * num_elements)
     * @param nodes_per_elem Nodes per element
     */
    void build(std::size_t num_nodes, std::size_t num_elements,
               const Index* connectivity, int nodes_per_elem) {
        neighbors_.clear();
        neighbors_.resize(num_nodes);

        for (std::size_t e = 0; e < num_elements; ++e) {
            const Index* elem = connectivity + e * nodes_per_elem;
            for (int i = 0; i < nodes_per_elem; ++i) {
                for (int j = 0; j < nodes_per_elem; ++j) {
                    if (i != j && elem[i] < num_nodes && elem[j] < num_nodes) {
                        neighbors_[elem[i]].insert(elem[j]);
                    }
                }
            }
        }
    }

    const std::set<Index>& neighbors(Index node) const {
        return neighbors_[node];
    }

    std::size_t num_neighbors(Index node) const {
        return neighbors_[node].size();
    }

    std::size_t num_nodes() const { return neighbors_.size(); }

private:
    std::vector<std::set<Index>> neighbors_;
};

// ============================================================================
// ALE Solver
// ============================================================================

class ALESolver {
public:
    ALESolver() = default;

    void set_config(const ALEConfig& cfg) { config_ = cfg; }
    const ALEConfig& config() const { return config_; }

    /**
     * @brief Initialize ALE solver with mesh connectivity
     */
    void initialize(std::size_t num_nodes, std::size_t num_elements,
                    const Index* connectivity, int nodes_per_elem,
                    const std::set<Index>& boundary_nodes = {}) {
        num_nodes_ = num_nodes;
        adjacency_.build(num_nodes, num_elements, connectivity, nodes_per_elem);
        boundary_nodes_ = boundary_nodes;
    }

    /**
     * @brief Set boundary nodes (fixed during smoothing)
     */
    void set_boundary_nodes(const std::set<Index>& nodes) {
        boundary_nodes_ = nodes;
    }

    // --- Mesh Smoothing ---

    /**
     * @brief Smooth mesh coordinates (in-place)
     *
     * @param coordinates Node coordinates (3*num_nodes, [x0,y0,z0,x1,...])
     * @return Maximum node displacement from smoothing
     */
    Real smooth(Real* coordinates) {
        Real max_disp = 0.0;

        for (int iter = 0; iter < config_.smoothing_iterations; ++iter) {
            Real iter_disp = 0.0;
            switch (config_.smoothing) {
                case SmoothingMethod::Laplacian:
                    iter_disp = smooth_laplacian(coordinates);
                    break;
                case SmoothingMethod::Weighted:
                    iter_disp = smooth_weighted(coordinates);
                    break;
                case SmoothingMethod::Equipotential:
                    iter_disp = smooth_laplacian(coordinates);
                    break;
            }
            if (iter_disp > max_disp) max_disp = iter_disp;
        }

        return max_disp;
    }

    // --- State Advection ---

    /**
     * @brief Advect scalar field from old mesh to new mesh
     *
     * Uses mesh velocity (new_coords - old_coords) / dt to determine
     * advection direction and amount.
     *
     * @param old_coords Previous node coordinates (3*num_nodes)
     * @param new_coords New (smoothed) node coordinates (3*num_nodes)
     * @param field Scalar field values per node (modified in-place)
     * @param dt Time step
     */
    void advect_scalar(const Real* old_coords, const Real* new_coords,
                        Real* field, Real dt) {
        if (dt < 1.0e-30) return;

        std::vector<Real> field_new(num_nodes_);

        for (std::size_t i = 0; i < num_nodes_; ++i) {
            // Mesh velocity at node i
            Real vmx = (new_coords[3*i+0] - old_coords[3*i+0]) / dt;
            Real vmy = (new_coords[3*i+1] - old_coords[3*i+1]) / dt;
            Real vmz = (new_coords[3*i+2] - old_coords[3*i+2]) / dt;

            Real vm_mag = std::sqrt(vmx*vmx + vmy*vmy + vmz*vmz);

            if (vm_mag < 1.0e-30) {
                field_new[i] = field[i];
                continue;
            }

            switch (config_.advection) {
                case AdvectionMethod::DonorCell:
                    field_new[i] = advect_donor_cell(i, vmx, vmy, vmz,
                                                      old_coords, field, dt);
                    break;
                case AdvectionMethod::VanLeer:
                    field_new[i] = advect_van_leer(i, vmx, vmy, vmz,
                                                    old_coords, field, dt);
                    break;
            }
        }

        // Copy back
        for (std::size_t i = 0; i < num_nodes_; ++i) {
            field[i] = field_new[i];
        }
    }

    /**
     * @brief Advect vector field (3 components per node)
     */
    void advect_vector(const Real* old_coords, const Real* new_coords,
                        Real* field, Real dt) {
        // Advect each component separately
        std::vector<Real> comp(num_nodes_);

        for (int d = 0; d < 3; ++d) {
            for (std::size_t i = 0; i < num_nodes_; ++i)
                comp[i] = field[3*i+d];

            advect_scalar(old_coords, new_coords, comp.data(), dt);

            for (std::size_t i = 0; i < num_nodes_; ++i)
                field[3*i+d] = comp[i];
        }
    }

    // --- Full ALE Step ---

    /**
     * @brief Perform complete ALE step: smooth + advect
     *
     * @param coordinates Node coordinates (modified by smoothing)
     * @param velocities Nodal velocities (advected)
     * @param scalars Scalar fields to advect (vector of pointers)
     * @param dt Time step
     * @return Mesh smoothing displacement
     */
    Real ale_step(Real* coordinates, Real* velocities,
                   const std::vector<Real*>& scalars, Real dt) {
        // Save old coordinates
        std::vector<Real> old_coords(3 * num_nodes_);
        for (std::size_t i = 0; i < 3 * num_nodes_; ++i)
            old_coords[i] = coordinates[i];

        // Smooth mesh
        Real max_disp = smooth(coordinates);

        // Advect fields
        if (max_disp > 1.0e-30) {
            advect_vector(old_coords.data(), coordinates, velocities, dt);

            for (Real* field : scalars) {
                if (field) {
                    advect_scalar(old_coords.data(), coordinates, field, dt);
                }
            }
        }

        total_smoothing_disp_ += max_disp;
        ale_steps_++;

        return max_disp;
    }

    // --- Quality Metrics ---

    /**
     * @brief Compute element quality metric (Jacobian ratio)
     *
     * For a hex8 element, computes J_min/J_max at corners.
     * Returns 1.0 for perfect element, 0.0 for degenerate.
     */
    static Real element_quality(const Real* coords, const Index* elem_nodes,
                                  int nodes_per_elem) {
        if (nodes_per_elem < 4) return 1.0;

        // Simplified: compute volume-based quality for tetrahedra/hexahedra
        // Use ratio of min edge length to max edge length as proxy
        Real min_edge = 1.0e30;
        Real max_edge = 0.0;

        for (int i = 0; i < nodes_per_elem; ++i) {
            for (int j = i+1; j < nodes_per_elem; ++j) {
                Index ni = elem_nodes[i];
                Index nj = elem_nodes[j];
                Real dx = coords[3*ni+0] - coords[3*nj+0];
                Real dy = coords[3*ni+1] - coords[3*nj+1];
                Real dz = coords[3*ni+2] - coords[3*nj+2];
                Real len = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (len < min_edge) min_edge = len;
                if (len > max_edge) max_edge = len;
            }
        }

        return (max_edge > 1.0e-30) ? min_edge / max_edge : 0.0;
    }

    /**
     * @brief Compute average element quality over entire mesh
     */
    Real average_quality(const Real* coords, const Index* connectivity,
                          std::size_t num_elements, int nodes_per_elem) const {
        if (num_elements == 0) return 1.0;
        Real sum = 0.0;
        for (std::size_t e = 0; e < num_elements; ++e) {
            sum += element_quality(coords, connectivity + e * nodes_per_elem,
                                    nodes_per_elem);
        }
        return sum / num_elements;
    }

    // --- Statistics ---

    Real total_smoothing_displacement() const { return total_smoothing_disp_; }
    int ale_step_count() const { return ale_steps_; }

    void print_summary() const {
        std::cout << "ALE Solver: " << ale_steps_ << " steps, total smoothing="
                  << total_smoothing_disp_ << "\n";
    }

private:
    // --- Laplacian Smoothing ---

    Real smooth_laplacian(Real* coords) {
        Real max_disp = 0.0;
        Real w = config_.smoothing_weight;

        // Compute new positions
        std::vector<Real> new_coords(3 * num_nodes_);
        for (std::size_t i = 0; i < 3 * num_nodes_; ++i)
            new_coords[i] = coords[i];

        for (std::size_t i = 0; i < num_nodes_; ++i) {
            // Skip boundary nodes
            if (config_.boundary_fixed && boundary_nodes_.count(i) > 0) continue;

            const auto& nbrs = adjacency_.neighbors(i);
            if (nbrs.empty()) continue;

            // Average of neighbor positions
            Real avg_x = 0.0, avg_y = 0.0, avg_z = 0.0;
            for (Index j : nbrs) {
                avg_x += coords[3*j+0];
                avg_y += coords[3*j+1];
                avg_z += coords[3*j+2];
            }
            Real n = static_cast<Real>(nbrs.size());
            avg_x /= n;
            avg_y /= n;
            avg_z /= n;

            // Relaxed update
            new_coords[3*i+0] = (1.0-w) * coords[3*i+0] + w * avg_x;
            new_coords[3*i+1] = (1.0-w) * coords[3*i+1] + w * avg_y;
            new_coords[3*i+2] = (1.0-w) * coords[3*i+2] + w * avg_z;

            // Track maximum displacement
            Real dx = new_coords[3*i+0] - coords[3*i+0];
            Real dy = new_coords[3*i+1] - coords[3*i+1];
            Real dz = new_coords[3*i+2] - coords[3*i+2];
            Real disp = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (disp > max_disp) max_disp = disp;
        }

        // Copy back
        for (std::size_t i = 0; i < 3 * num_nodes_; ++i)
            coords[i] = new_coords[i];

        return max_disp;
    }

    // --- Weighted Smoothing ---

    Real smooth_weighted(Real* coords) {
        Real max_disp = 0.0;
        Real w = config_.smoothing_weight;

        std::vector<Real> new_coords(3 * num_nodes_);
        for (std::size_t i = 0; i < 3 * num_nodes_; ++i)
            new_coords[i] = coords[i];

        for (std::size_t i = 0; i < num_nodes_; ++i) {
            if (config_.boundary_fixed && boundary_nodes_.count(i) > 0) continue;

            const auto& nbrs = adjacency_.neighbors(i);
            if (nbrs.empty()) continue;

            // Distance-weighted average (closer neighbors have more influence)
            Real wsum_x = 0.0, wsum_y = 0.0, wsum_z = 0.0, wsum = 0.0;
            for (Index j : nbrs) {
                Real dx = coords[3*j+0] - coords[3*i+0];
                Real dy = coords[3*j+1] - coords[3*i+1];
                Real dz = coords[3*j+2] - coords[3*i+2];
                Real dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                Real weight = (dist > 1.0e-30) ? 1.0 / dist : 1.0;

                wsum_x += weight * coords[3*j+0];
                wsum_y += weight * coords[3*j+1];
                wsum_z += weight * coords[3*j+2];
                wsum += weight;
            }

            if (wsum > 1.0e-30) {
                Real avg_x = wsum_x / wsum;
                Real avg_y = wsum_y / wsum;
                Real avg_z = wsum_z / wsum;

                new_coords[3*i+0] = (1.0-w)*coords[3*i+0] + w*avg_x;
                new_coords[3*i+1] = (1.0-w)*coords[3*i+1] + w*avg_y;
                new_coords[3*i+2] = (1.0-w)*coords[3*i+2] + w*avg_z;

                Real dx = new_coords[3*i+0] - coords[3*i+0];
                Real dy = new_coords[3*i+1] - coords[3*i+1];
                Real dz = new_coords[3*i+2] - coords[3*i+2];
                Real disp = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (disp > max_disp) max_disp = disp;
            }
        }

        for (std::size_t i = 0; i < 3 * num_nodes_; ++i)
            coords[i] = new_coords[i];

        return max_disp;
    }

    // --- First-Order Donor Cell Advection ---

    Real advect_donor_cell(Index i, Real vmx, Real vmy, Real vmz,
                            const Real* old_coords, const Real* field,
                            Real dt) {
        // Find upstream neighbor (donor)
        const auto& nbrs = adjacency_.neighbors(i);
        if (nbrs.empty()) return field[i];

        // Find neighbor most aligned with negative mesh velocity (upstream)
        Real best_dot = -1.0e30;
        Index donor = i;

        for (Index j : nbrs) {
            Real dx = old_coords[3*j+0] - old_coords[3*i+0];
            Real dy = old_coords[3*j+1] - old_coords[3*i+1];
            Real dz = old_coords[3*j+2] - old_coords[3*i+2];
            Real dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dist < 1.0e-30) continue;

            // Dot product of direction to neighbor with negative mesh velocity
            Real dot = -(vmx*dx + vmy*dy + vmz*dz) / dist;
            if (dot > best_dot) {
                best_dot = dot;
                donor = j;
            }
        }

        // Courant number (approximate)
        Real vm = std::sqrt(vmx*vmx + vmy*vmy + vmz*vmz);
        Real dx = 0.0;
        if (donor != i) {
            Real ddx = old_coords[3*donor+0] - old_coords[3*i+0];
            Real ddy = old_coords[3*donor+1] - old_coords[3*i+1];
            Real ddz = old_coords[3*donor+2] - old_coords[3*i+2];
            dx = std::sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
        }
        Real courant = (dx > 1.0e-30) ? vm * dt / dx : 0.0;
        courant = std::min(courant, 1.0);  // Stability limit

        // Donor cell: f_new = f + courant * (f_donor - f)
        return field[i] + courant * (field[donor] - field[i]);
    }

    // --- Van Leer Advection ---

    Real advect_van_leer(Index i, Real vmx, Real vmy, Real vmz,
                          const Real* old_coords, const Real* field,
                          Real dt) {
        // Van Leer is donor cell with a slope limiter
        // For simplicity, use minmod limiter

        const auto& nbrs = adjacency_.neighbors(i);
        if (nbrs.empty()) return field[i];

        // Find donor and downstream
        Real best_up = -1.0e30;
        Real best_down = -1.0e30;
        Index donor = i, downstream = i;

        for (Index j : nbrs) {
            Real dx = old_coords[3*j+0] - old_coords[3*i+0];
            Real dy = old_coords[3*j+1] - old_coords[3*i+1];
            Real dz = old_coords[3*j+2] - old_coords[3*i+2];
            Real dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dist < 1.0e-30) continue;

            Real dot_neg = -(vmx*dx + vmy*dy + vmz*dz) / dist;
            Real dot_pos = (vmx*dx + vmy*dy + vmz*dz) / dist;

            if (dot_neg > best_up) { best_up = dot_neg; donor = j; }
            if (dot_pos > best_down) { best_down = dot_pos; downstream = j; }
        }

        // Gradients
        Real grad_up = field[i] - field[donor];        // Upwind gradient
        Real grad_down = field[downstream] - field[i];  // Downwind gradient

        // Minmod limiter
        Real slope = minmod(grad_up, grad_down);

        // Courant number
        Real vm = std::sqrt(vmx*vmx + vmy*vmy + vmz*vmz);
        Real dx = 0.0;
        if (donor != i) {
            Real ddx = old_coords[3*donor+0] - old_coords[3*i+0];
            Real ddy = old_coords[3*donor+1] - old_coords[3*i+1];
            Real ddz = old_coords[3*donor+2] - old_coords[3*i+2];
            dx = std::sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
        }
        Real courant = (dx > 1.0e-30) ? vm * dt / dx : 0.0;
        courant = std::min(courant, 1.0);

        return field[i] + courant * (field[donor] - field[i])
                        + 0.5 * courant * (1.0 - courant) * slope;
    }

    static Real minmod(Real a, Real b) {
        if (a * b <= 0.0) return 0.0;
        return (std::fabs(a) < std::fabs(b)) ? a : b;
    }

    // Members
    ALEConfig config_;
    MeshAdjacency adjacency_;
    std::set<Index> boundary_nodes_;
    std::size_t num_nodes_ = 0;

    // Statistics
    Real total_smoothing_disp_ = 0.0;
    int ale_steps_ = 0;
};

} // namespace physics
} // namespace nxs
