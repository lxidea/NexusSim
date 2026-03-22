#pragma once

/**
 * @file coupling_wave41.hpp
 * @brief Wave 41: Coupling & Acoustics Production Hardening — 4 Features
 *
 * Features:
 *   9.  CouplingSubIteration    - Implicit sub-iteration for FSI with Aitken relaxation
 *   10. CouplingFieldSmoothing  - Laplacian smoothing for transferred interface fields
 *   11. AcousticFMM             - Fast Multipole Method for BEM acoustics
 *   12. AcousticStructuralModes - Acoustic-structural modal coupling
 *
 * References:
 * - Kuttler & Wall (2008) "Fixed-point FSI solvers with dynamic relaxation"
 * - Greengard & Rokhlin (1987) "A fast algorithm for particle simulations"
 * - Everstine & Henderson (1990) "Coupled FE/BE methods for fluid-structure"
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>

namespace nxs {
namespace coupling {

using Real = nxs::Real;

// ============================================================================
// Utility functions
// ============================================================================

namespace coupling41_detail {

inline Real dot(const std::vector<Real>& a, const std::vector<Real>& b) {
    Real s = 0.0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline Real norm(const std::vector<Real>& v) {
    return std::sqrt(dot(v, v));
}

inline Real distance3(const Real* a, const Real* b) {
    Real d[3] = {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
    return std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
}

inline Real clamp(Real x, Real lo, Real hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

} // namespace coupling41_detail

// ============================================================================
// 9. CouplingSubIteration — Implicit FSI sub-iteration with Aitken relaxation
// ============================================================================

/**
 * @brief Implicit sub-iteration coupling for partitioned FSI.
 *
 * Fixed-point iteration with Aitken dynamic relaxation:
 *   omega_{k+1} = -omega_k * (r_{k-1} . (r_k - r_{k-1})) / ||r_k - r_{k-1}||^2
 *
 * The user provides callable solvers for fluid and structure.
 */
class CouplingSubIteration {
public:
    struct Params {
        Real omega_init   = 0.5;   ///< Initial relaxation factor
        int  max_sub_iter = 10;    ///< Maximum sub-iterations
        Real tolerance    = 1.0e-6; ///< Convergence tolerance on residual norm
        Real omega_min    = 0.01;  ///< Minimum relaxation factor
        Real omega_max    = 1.5;   ///< Maximum relaxation factor
    };

    struct Result {
        bool converged = false;
        int  iterations = 0;
        Real final_residual = 0.0;
        Real final_omega = 0.0;
    };

    CouplingSubIteration() = default;
    explicit CouplingSubIteration(const Params& p) : params_(p) {}

    /**
     * @brief Run sub-iteration loop until convergence or max iterations.
     *
     * @param fluid_solve   Callable: takes displacement, returns interface forces (vector<Real>)
     * @param struct_solve  Callable: takes forces, returns interface displacements (vector<Real>)
     * @param transfer      Callable: maps between fluid/structural interface representations
     * @param initial_disp  Initial guess for interface displacement
     * @return Result with convergence info
     */
    Result iterate(
        const std::function<std::vector<Real>(const std::vector<Real>&)>& fluid_solve,
        const std::function<std::vector<Real>(const std::vector<Real>&)>& struct_solve,
        const std::function<std::vector<Real>(const std::vector<Real>&)>& transfer,
        const std::vector<Real>& initial_disp) const
    {
        Result result;
        size_t n = initial_disp.size();
        if (n == 0) return result;

        std::vector<Real> disp = initial_disp;
        std::vector<Real> disp_prev(n, 0.0);
        std::vector<Real> residual(n, 0.0);
        std::vector<Real> residual_prev(n, 0.0);
        Real omega = params_.omega_init;

        for (int iter = 0; iter < params_.max_sub_iter; ++iter) {
            // 1. Solve fluid with current displacement -> forces
            auto forces = fluid_solve(disp);

            // 2. Transfer forces (e.g., interpolation)
            auto transferred_forces = transfer(forces);

            // 3. Solve structure with forces -> new displacement
            auto disp_new = struct_solve(transferred_forces);

            // 4. Compute residual: r = d_new - d
            if (disp_new.size() != n) {
                disp_new.resize(n, 0.0);
            }

            for (size_t i = 0; i < n; ++i) {
                residual[i] = disp_new[i] - disp[i];
            }

            Real res_norm = coupling41_detail::norm(residual);
            Real disp_norm = coupling41_detail::norm(disp);
            Real rel_res = (disp_norm > 1.0e-30) ? res_norm / disp_norm : res_norm;

            result.iterations = iter + 1;
            result.final_residual = rel_res;
            result.final_omega = omega;

            // 5. Check convergence
            if (rel_res < params_.tolerance) {
                result.converged = true;
                // Apply final update
                for (size_t i = 0; i < n; ++i) {
                    disp[i] = disp_new[i];
                }
                break;
            }

            // 6. Aitken relaxation update (after first iteration)
            if (iter > 0) {
                std::vector<Real> dr(n);
                for (size_t i = 0; i < n; ++i) {
                    dr[i] = residual[i] - residual_prev[i];
                }

                Real dr_norm_sq = coupling41_detail::dot(dr, dr);
                if (dr_norm_sq > 1.0e-30) {
                    Real dot_r_dr = coupling41_detail::dot(residual_prev, dr);
                    omega = -omega * dot_r_dr / dr_norm_sq;
                    omega = coupling41_detail::clamp(omega,
                                                     params_.omega_min,
                                                     params_.omega_max);
                }
            }

            // 7. Relaxed update: d_{k+1} = d_k + omega * r_k
            for (size_t i = 0; i < n; ++i) {
                disp_prev[i] = disp[i];
                disp[i] += omega * residual[i];
            }

            residual_prev = residual;
        }

        return result;
    }

    /**
     * @brief Simplified iterate with direct fluid-structure coupling.
     *
     * @param solve Combined F-S solver: takes displacement, returns new displacement
     * @param initial_disp Initial guess
     * @return Result
     */
    Result iterate_simple(
        const std::function<std::vector<Real>(const std::vector<Real>&)>& solve,
        const std::vector<Real>& initial_disp) const
    {
        auto identity = [](const std::vector<Real>& v) { return v; };
        return iterate(solve, identity, identity, initial_disp);
    }

    const Params& params() const { return params_; }
    Params& params() { return params_; }

private:
    Params params_{};
};

// ============================================================================
// 10. CouplingFieldSmoothing — Laplacian smoothing for interface fields
// ============================================================================

/**
 * @brief Laplacian smoothing for fields transferred across coupling interfaces.
 *
 * Smooths oscillatory artifacts that arise from non-matching interface meshes.
 * f_i_new = (1 - alpha) * f_i + alpha * mean(f_neighbors)
 */
class CouplingFieldSmoothing {
public:
    CouplingFieldSmoothing() = default;

    /**
     * @brief Smooth a scalar field on a mesh using Laplacian smoothing.
     *
     * @param field Field values at each node (modified in-place)
     * @param connectivity Adjacency: for each node, list of neighbor indices
     * @param n_passes Number of smoothing passes
     * @param alpha Smoothing factor in (0, 1). 0 = no smoothing, 1 = full average.
     */
    void smooth(std::vector<Real>& field,
                const std::vector<std::vector<int>>& connectivity,
                int n_passes = 3,
                Real alpha = 0.5) const
    {
        if (field.empty() || connectivity.empty()) return;

        size_t n = field.size();
        std::vector<Real> temp(n);

        for (int pass = 0; pass < n_passes; ++pass) {
            for (size_t i = 0; i < n; ++i) {
                if (i >= connectivity.size() || connectivity[i].empty()) {
                    temp[i] = field[i];
                    continue;
                }

                // Compute mean of neighbors
                Real sum = 0.0;
                int count = 0;
                for (int nbr : connectivity[i]) {
                    if (nbr >= 0 && static_cast<size_t>(nbr) < n) {
                        sum += field[static_cast<size_t>(nbr)];
                        ++count;
                    }
                }

                if (count > 0) {
                    Real mean = sum / static_cast<Real>(count);
                    temp[i] = (1.0 - alpha) * field[i] + alpha * mean;
                } else {
                    temp[i] = field[i];
                }
            }
            field = temp;
        }
    }

    /**
     * @brief Smooth a vector field (3 components per node).
     *
     * @param field Vector field values (3 per node, flattened)
     * @param connectivity Adjacency lists
     * @param n_passes Number of smoothing passes
     * @param alpha Smoothing factor
     */
    void smooth_vector(std::vector<Real>& field,
                       const std::vector<std::vector<int>>& connectivity,
                       int n_passes = 3,
                       Real alpha = 0.5) const
    {
        size_t n = field.size() / 3;
        if (n == 0 || connectivity.empty()) return;

        // Smooth each component independently
        for (int comp = 0; comp < 3; ++comp) {
            std::vector<Real> component(n);
            for (size_t i = 0; i < n; ++i) {
                component[i] = field[i * 3 + static_cast<size_t>(comp)];
            }
            smooth(component, connectivity, n_passes, alpha);
            for (size_t i = 0; i < n; ++i) {
                field[i * 3 + static_cast<size_t>(comp)] = component[i];
            }
        }
    }

    /**
     * @brief Build connectivity from element definitions (triangle mesh).
     *
     * @param elements Element connectivity (3 per triangle, flattened)
     * @param n_elements Number of elements
     * @param n_nodes Number of nodes
     * @return Adjacency lists for each node
     */
    static std::vector<std::vector<int>> build_connectivity(
        const std::vector<int>& elements,
        int n_elements,
        int n_nodes)
    {
        std::vector<std::vector<int>> conn(static_cast<size_t>(n_nodes));

        for (int e = 0; e < n_elements; ++e) {
            size_t base = static_cast<size_t>(e) * 3;
            if (base + 2 >= elements.size()) break;

            int nodes[3] = {elements[base], elements[base+1], elements[base+2]};

            // Each node is a neighbor of the other two
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    if (i == j) continue;
                    int ni = nodes[i], nj = nodes[j];
                    if (ni < 0 || ni >= n_nodes || nj < 0 || nj >= n_nodes) continue;
                    auto& nbrs = conn[static_cast<size_t>(ni)];
                    if (std::find(nbrs.begin(), nbrs.end(), nj) == nbrs.end()) {
                        nbrs.push_back(nj);
                    }
                }
            }
        }
        return conn;
    }
};

// ============================================================================
// 11. AcousticFMM — Fast Multipole Method for BEM acoustics
// ============================================================================

/**
 * @brief Fast Multipole Method (FMM) for acoustic BEM.
 *
 * Hierarchical octree decomposition separates near-field (direct) and
 * far-field (multipole expansion) interactions for O(N log N) BEM solves.
 */
class AcousticFMM {
public:
    struct FMMNode {
        Real center[3] = {0.0, 0.0, 0.0};
        Real radius    = 0.0;
        int  children[8] = {-1,-1,-1,-1,-1,-1,-1,-1}; ///< Octree child indices
        std::vector<int> source_indices;  ///< Sources in this leaf
        bool is_leaf   = true;
        int  level     = 0;
    };

    struct FMMResult {
        std::vector<Real> pressure_real;  ///< Real part of acoustic pressure
        std::vector<Real> pressure_imag;  ///< Imaginary part of pressure
        int near_field_evals = 0;
        int far_field_evals  = 0;
    };

    AcousticFMM() = default;

    /**
     * @brief Build octree from source positions.
     *
     * @param nodes Source node positions (3 per node, flattened)
     * @param n_nodes Number of nodes
     * @param max_leaf Maximum sources per leaf node
     */
    void build_tree(const std::vector<Real>& nodes, int n_nodes, int max_leaf = 32) {
        tree_.clear();
        max_leaf_ = max_leaf;

        if (n_nodes == 0) return;

        // Compute bounding box
        Real min_c[3] = { 1e30,  1e30,  1e30};
        Real max_c[3] = {-1e30, -1e30, -1e30};
        for (int i = 0; i < n_nodes; ++i) {
            size_t idx = static_cast<size_t>(i) * 3;
            if (idx + 2 >= nodes.size()) break;
            for (int d = 0; d < 3; ++d) {
                Real v = nodes[idx + static_cast<size_t>(d)];
                if (v < min_c[d]) min_c[d] = v;
                if (v > max_c[d]) max_c[d] = v;
            }
        }

        // Root node
        FMMNode root;
        for (int d = 0; d < 3; ++d) {
            root.center[d] = 0.5 * (min_c[d] + max_c[d]);
        }
        Real half[3] = {
            0.5 * (max_c[0] - min_c[0]),
            0.5 * (max_c[1] - min_c[1]),
            0.5 * (max_c[2] - min_c[2])
        };
        root.radius = std::max({half[0], half[1], half[2]}) * 1.01;
        root.source_indices.resize(static_cast<size_t>(n_nodes));
        std::iota(root.source_indices.begin(), root.source_indices.end(), 0);
        root.level = 0;
        tree_.push_back(root);

        // Recursive subdivision
        subdivide(0, nodes, 0);
    }

    /**
     * @brief Compute acoustic pressure at target points from monopole sources.
     *
     * Green's function: G(r) = exp(i*k*r) / (4*pi*r)
     * Uses FMM for far-field, direct evaluation for near-field.
     *
     * @param source_positions Source positions (3 per source, flattened)
     * @param source_strengths Source strengths (complex: 2 per source, [re,im])
     * @param target_positions Target positions (3 per target, flattened)
     * @param n_sources Number of sources
     * @param n_targets Number of targets
     * @param k_wave Wavenumber (omega/c)
     * @return FMMResult with pressure at each target
     */
    FMMResult compute_pressure(
        const std::vector<Real>& source_positions,
        const std::vector<Real>& source_strengths,
        const std::vector<Real>& target_positions,
        int n_sources,
        int n_targets,
        Real k_wave) const
    {
        FMMResult result;
        result.pressure_real.resize(static_cast<size_t>(n_targets), 0.0);
        result.pressure_imag.resize(static_cast<size_t>(n_targets), 0.0);

        if (n_sources == 0 || n_targets == 0) return result;

        const Real four_pi = 4.0 * M_PI;

        // If tree is not built or small problem, use direct evaluation
        if (tree_.empty() || n_sources * n_targets < 10000) {
            // Direct O(N*M) evaluation
            for (int t = 0; t < n_targets; ++t) {
                size_t tidx = static_cast<size_t>(t) * 3;
                if (tidx + 2 >= target_positions.size()) break;

                Real p_re = 0.0, p_im = 0.0;

                for (int s = 0; s < n_sources; ++s) {
                    size_t sidx = static_cast<size_t>(s) * 3;
                    size_t stidx = static_cast<size_t>(s) * 2;
                    if (sidx + 2 >= source_positions.size()) break;
                    if (stidx + 1 >= source_strengths.size()) break;

                    Real dx = target_positions[tidx]   - source_positions[sidx];
                    Real dy = target_positions[tidx+1] - source_positions[sidx+1];
                    Real dz = target_positions[tidx+2] - source_positions[sidx+2];
                    Real r = std::sqrt(dx*dx + dy*dy + dz*dz);

                    if (r < 1.0e-15) continue;

                    // G = exp(ikr) / (4*pi*r)
                    Real kr = k_wave * r;
                    Real g_re = std::cos(kr) / (four_pi * r);
                    Real g_im = std::sin(kr) / (four_pi * r);

                    Real q_re = source_strengths[stidx];
                    Real q_im = source_strengths[stidx + 1];

                    // p += G * q  (complex multiply)
                    p_re += g_re * q_re - g_im * q_im;
                    p_im += g_re * q_im + g_im * q_re;

                    ++result.near_field_evals;
                }

                result.pressure_real[static_cast<size_t>(t)] = p_re;
                result.pressure_imag[static_cast<size_t>(t)] = p_im;
            }
            return result;
        }

        // FMM evaluation using tree
        for (int t = 0; t < n_targets; ++t) {
            size_t tidx = static_cast<size_t>(t) * 3;
            if (tidx + 2 >= target_positions.size()) break;

            Real target[3] = {
                target_positions[tidx],
                target_positions[tidx+1],
                target_positions[tidx+2]
            };

            Real p_re = 0.0, p_im = 0.0;
            evaluate_node(0, target, source_positions, source_strengths,
                          k_wave, p_re, p_im,
                          result.near_field_evals, result.far_field_evals);

            result.pressure_real[static_cast<size_t>(t)] = p_re;
            result.pressure_imag[static_cast<size_t>(t)] = p_im;
        }

        return result;
    }

    size_t num_nodes() const { return tree_.size(); }
    const FMMNode& node(int i) const { return tree_[static_cast<size_t>(i)]; }

private:
    std::vector<FMMNode> tree_;
    int max_leaf_ = 32;

    void subdivide(int node_idx, const std::vector<Real>& positions, int depth) {
        if (depth > 20) return; // safety limit
        auto& nd = tree_[static_cast<size_t>(node_idx)];
        if (static_cast<int>(nd.source_indices.size()) <= max_leaf_) return;

        nd.is_leaf = false;

        // Create 8 children
        for (int oct = 0; oct < 8; ++oct) {
            FMMNode child;
            Real half_r = nd.radius * 0.5;
            child.center[0] = nd.center[0] + ((oct & 1) ? half_r : -half_r) * 0.5;
            child.center[1] = nd.center[1] + ((oct & 2) ? half_r : -half_r) * 0.5;
            child.center[2] = nd.center[2] + ((oct & 4) ? half_r : -half_r) * 0.5;
            child.radius = half_r;
            child.level = depth + 1;

            // Assign sources to this octant
            for (int si : nd.source_indices) {
                size_t idx = static_cast<size_t>(si) * 3;
                if (idx + 2 >= positions.size()) continue;
                Real px = positions[idx];
                Real py = positions[idx + 1];
                Real pz = positions[idx + 2];

                int octant = 0;
                if (px >= nd.center[0]) octant |= 1;
                if (py >= nd.center[1]) octant |= 2;
                if (pz >= nd.center[2]) octant |= 4;

                if (octant == oct) {
                    child.source_indices.push_back(si);
                }
            }

            if (!child.source_indices.empty()) {
                int child_idx = static_cast<int>(tree_.size());
                nd.children[oct] = child_idx;
                tree_.push_back(child);
                subdivide(child_idx, positions, depth + 1);
            }
        }

        // Clear parent source list (sources are now in children)
        // Keep the list for reference but mark as non-leaf
    }

    void evaluate_node(int node_idx, const Real target[3],
                       const std::vector<Real>& source_positions,
                       const std::vector<Real>& source_strengths,
                       Real k_wave,
                       Real& p_re, Real& p_im,
                       int& near_evals, int& far_evals) const
    {
        const auto& nd = tree_[static_cast<size_t>(node_idx)];

        Real dist = coupling41_detail::distance3(target, nd.center);
        const Real four_pi = 4.0 * M_PI;

        // Well-separated criterion: dist > 2 * radius (far-field)
        if (!nd.is_leaf && dist > 2.0 * nd.radius) {
            // Far-field: use monopole approximation (P=0 expansion)
            Real total_q_re = 0.0, total_q_im = 0.0;
            for (int si : nd.source_indices) {
                size_t stidx = static_cast<size_t>(si) * 2;
                if (stidx + 1 >= source_strengths.size()) continue;
                total_q_re += source_strengths[stidx];
                total_q_im += source_strengths[stidx + 1];
            }

            if (dist > 1.0e-15) {
                Real kr = k_wave * dist;
                Real g_re = std::cos(kr) / (four_pi * dist);
                Real g_im = std::sin(kr) / (four_pi * dist);
                p_re += g_re * total_q_re - g_im * total_q_im;
                p_im += g_re * total_q_im + g_im * total_q_re;
            }
            ++far_evals;
            return;
        }

        if (nd.is_leaf) {
            // Near-field: direct evaluation
            for (int si : nd.source_indices) {
                size_t sidx = static_cast<size_t>(si) * 3;
                size_t stidx = static_cast<size_t>(si) * 2;
                if (sidx + 2 >= source_positions.size()) continue;
                if (stidx + 1 >= source_strengths.size()) continue;

                Real dx = target[0] - source_positions[sidx];
                Real dy = target[1] - source_positions[sidx+1];
                Real dz = target[2] - source_positions[sidx+2];
                Real r = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (r < 1.0e-15) continue;

                Real kr = k_wave * r;
                Real g_re = std::cos(kr) / (four_pi * r);
                Real g_im = std::sin(kr) / (four_pi * r);

                Real q_re = source_strengths[stidx];
                Real q_im = source_strengths[stidx + 1];

                p_re += g_re * q_re - g_im * q_im;
                p_im += g_re * q_im + g_im * q_re;

                ++near_evals;
            }
            return;
        }

        // Recurse into children
        for (int c = 0; c < 8; ++c) {
            if (nd.children[c] >= 0) {
                evaluate_node(nd.children[c], target, source_positions,
                              source_strengths, k_wave, p_re, p_im,
                              near_evals, far_evals);
            }
        }
    }
};

// ============================================================================
// 12. AcousticStructuralModes — Acoustic-structural modal coupling
// ============================================================================

/**
 * @brief Coupled acoustic-structural modal analysis.
 *
 * Solves the coupled eigenvalue problem:
 *   [K_s, -C; -C^T, K_a] * {u; p} = omega^2 * [M_s, 0; 0, M_a] * {u; p}
 *
 * where C is the coupling matrix (integral of N_s . n . N_a over interface).
 * Uses a perturbation approach for small coupling.
 */
class AcousticStructuralModes {
public:
    struct CoupledMode {
        Real frequency = 0.0;                  ///< Coupled frequency [Hz]
        Real structural_participation = 0.0;   ///< Structural participation factor
        Real acoustic_participation = 0.0;     ///< Acoustic participation factor
        Real frequency_shift = 0.0;            ///< Shift from uncoupled frequency
    };

    struct ModeData {
        Real frequency = 0.0;           ///< Uncoupled frequency [Hz]
        std::vector<Real> mode_shape;   ///< Mode shape vector
        Real modal_mass = 1.0;          ///< Modal mass (generalized)
        Real modal_stiffness = 0.0;     ///< Modal stiffness (generalized)
    };

    AcousticStructuralModes() = default;

    /**
     * @brief Couple structural and acoustic modes through a coupling matrix.
     *
     * Uses first-order perturbation theory for moderate coupling.
     *
     * @param structural_modes Structural modal data (n_s modes)
     * @param acoustic_modes Acoustic modal data (n_a modes)
     * @param coupling_matrix C matrix (n_s x n_a), row-major, coupling structural
     *                        DOFs to acoustic DOFs via interface integral
     * @return Vector of coupled modes (n_s + n_a entries)
     */
    std::vector<CoupledMode> couple(
        const std::vector<ModeData>& structural_modes,
        const std::vector<ModeData>& acoustic_modes,
        const std::vector<Real>& coupling_matrix) const
    {
        size_t n_s = structural_modes.size();
        size_t n_a = acoustic_modes.size();
        std::vector<CoupledMode> result;

        if (n_s == 0 && n_a == 0) return result;

        // Compute modal coupling coefficients
        // c_ij = phi_s_i^T * C * phi_a_j  (projected coupling)
        // For simplicity, if mode shapes are not provided, use the coupling
        // matrix directly as modal coupling terms.

        // Process structural modes
        for (size_t i = 0; i < n_s; ++i) {
            CoupledMode cm;
            Real omega_s = 2.0 * M_PI * structural_modes[i].frequency;
            Real omega_s_sq = omega_s * omega_s;
            Real M_s = structural_modes[i].modal_mass;

            // First-order perturbation from acoustic coupling
            Real delta_omega_sq = 0.0;
            for (size_t j = 0; j < n_a; ++j) {
                Real omega_a = 2.0 * M_PI * acoustic_modes[j].frequency;
                Real omega_a_sq = omega_a * omega_a;
                Real M_a = acoustic_modes[j].modal_mass;

                // Get coupling coefficient
                Real c_ij = 0.0;
                size_t idx = i * n_a + j;
                if (idx < coupling_matrix.size()) {
                    c_ij = coupling_matrix[idx];
                }

                // Perturbation: delta_omega_s^2 = -c_ij^2 / (M_s * M_a * (omega_a^2 - omega_s^2))
                Real denom = omega_a_sq - omega_s_sq;
                if (std::abs(denom) > 1.0e-10) {
                    Real prod = M_s * M_a;
                    if (prod > 1.0e-30) {
                        delta_omega_sq += c_ij * c_ij / (prod * denom);
                    }
                }
            }

            Real omega_coupled_sq = omega_s_sq + delta_omega_sq;
            if (omega_coupled_sq < 0.0) omega_coupled_sq = 0.0;
            Real omega_coupled = std::sqrt(omega_coupled_sq);

            cm.frequency = omega_coupled / (2.0 * M_PI);
            cm.frequency_shift = cm.frequency - structural_modes[i].frequency;

            // Participation factors
            Real total_energy = omega_coupled_sq * M_s;
            cm.structural_participation = (total_energy > 1.0e-30)
                ? omega_s_sq * M_s / total_energy : 1.0;
            cm.acoustic_participation = 1.0 - cm.structural_participation;
            cm.structural_participation = coupling41_detail::clamp(
                cm.structural_participation, 0.0, 1.0);
            cm.acoustic_participation = coupling41_detail::clamp(
                cm.acoustic_participation, 0.0, 1.0);

            result.push_back(cm);
        }

        // Process acoustic modes
        for (size_t j = 0; j < n_a; ++j) {
            CoupledMode cm;
            Real omega_a = 2.0 * M_PI * acoustic_modes[j].frequency;
            Real omega_a_sq = omega_a * omega_a;
            Real M_a = acoustic_modes[j].modal_mass;

            Real delta_omega_sq = 0.0;
            for (size_t i = 0; i < n_s; ++i) {
                Real omega_s = 2.0 * M_PI * structural_modes[i].frequency;
                Real omega_s_sq = omega_s * omega_s;
                Real M_s = structural_modes[i].modal_mass;

                Real c_ij = 0.0;
                size_t idx = i * n_a + j;
                if (idx < coupling_matrix.size()) {
                    c_ij = coupling_matrix[idx];
                }

                Real denom = omega_s_sq - omega_a_sq;
                if (std::abs(denom) > 1.0e-10) {
                    Real prod = M_s * M_a;
                    if (prod > 1.0e-30) {
                        delta_omega_sq += c_ij * c_ij / (prod * denom);
                    }
                }
            }

            Real omega_coupled_sq = omega_a_sq + delta_omega_sq;
            if (omega_coupled_sq < 0.0) omega_coupled_sq = 0.0;
            Real omega_coupled = std::sqrt(omega_coupled_sq);

            cm.frequency = omega_coupled / (2.0 * M_PI);
            cm.frequency_shift = cm.frequency - acoustic_modes[j].frequency;

            Real total_energy = omega_coupled_sq * M_a;
            cm.acoustic_participation = (total_energy > 1.0e-30)
                ? omega_a_sq * M_a / total_energy : 1.0;
            cm.structural_participation = 1.0 - cm.acoustic_participation;
            cm.structural_participation = coupling41_detail::clamp(
                cm.structural_participation, 0.0, 1.0);
            cm.acoustic_participation = coupling41_detail::clamp(
                cm.acoustic_participation, 0.0, 1.0);

            result.push_back(cm);
        }

        // Sort by frequency
        std::sort(result.begin(), result.end(),
                  [](const CoupledMode& a, const CoupledMode& b) {
                      return a.frequency < b.frequency;
                  });

        return result;
    }

    /**
     * @brief Build coupling matrix from interface mesh.
     *
     * C_ij = integral of N_s_i * n_k * N_a_j dS over the interface.
     * Simplified: assumes matching interface nodes, so C is diagonal-like.
     *
     * @param interface_areas Area associated with each interface node
     * @param interface_normals Normal vectors (3 per node, flattened)
     * @param n_struct_modes Number of structural modes
     * @param n_acoustic_modes Number of acoustic modes
     * @param struct_shapes Structural mode shapes at interface (n_modes * n_interface)
     * @param acoustic_shapes Acoustic mode shapes at interface (n_modes * n_interface)
     * @param n_interface Number of interface nodes
     * @return Coupling matrix (n_struct_modes x n_acoustic_modes, row-major)
     */
    std::vector<Real> build_coupling_matrix(
        const std::vector<Real>& interface_areas,
        const std::vector<Real>& interface_normals,
        int n_struct_modes,
        int n_acoustic_modes,
        const std::vector<Real>& struct_shapes,
        const std::vector<Real>& acoustic_shapes,
        int n_interface) const
    {
        size_t ns = static_cast<size_t>(n_struct_modes);
        size_t na = static_cast<size_t>(n_acoustic_modes);
        size_t ni = static_cast<size_t>(n_interface);

        std::vector<Real> C(ns * na, 0.0);

        for (size_t i = 0; i < ns; ++i) {
            for (size_t j = 0; j < na; ++j) {
                Real c_ij = 0.0;
                for (size_t k = 0; k < ni; ++k) {
                    if (k >= interface_areas.size()) break;
                    size_t nidx = k * 3;
                    if (nidx + 2 >= interface_normals.size()) break;

                    // Structural shape: normal component
                    Real phi_s = 0.0;
                    size_t sidx = i * ni + k;
                    if (sidx < struct_shapes.size()) {
                        phi_s = struct_shapes[sidx];
                    }

                    // Acoustic shape
                    Real phi_a = 0.0;
                    size_t aidx = j * ni + k;
                    if (aidx < acoustic_shapes.size()) {
                        phi_a = acoustic_shapes[aidx];
                    }

                    // Normal magnitude (assuming shapes are scalar projections)
                    Real n_mag = std::sqrt(
                        interface_normals[nidx] * interface_normals[nidx] +
                        interface_normals[nidx+1] * interface_normals[nidx+1] +
                        interface_normals[nidx+2] * interface_normals[nidx+2]);

                    c_ij += phi_s * phi_a * interface_areas[k] * n_mag;
                }
                C[i * na + j] = c_ij;
            }
        }
        return C;
    }
};

} // namespace coupling
} // namespace nxs
