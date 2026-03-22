#pragma once

/**
 * @file xfem_wave41.hpp
 * @brief Wave 41: XFEM Production Hardening — 4 Features
 *
 * Features:
 *   1. XFEMFatigueCrack   - Paris-law fatigue crack growth with cycle counting
 *   2. XFEMMultiCrack     - Multiple simultaneous cracks with interaction
 *   3. XFEMAdaptiveMesh   - h-adaptive refinement around crack tips (ZZ error)
 *   4. XFEMOutputFields   - Crack path, SIF, COD extraction and VTK output
 *
 * References:
 * - Paris & Erdogan (1963) "A critical analysis of crack propagation laws"
 * - Budiansky & Rice (1973) "Conservation laws and energy-release rates"
 * - Zienkiewicz & Zhu (1987) "A simple error estimator for the FEM"
 * - Moes, Dolbow, Belytschko (1999) "A finite element method for crack growth"
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <string>
#include <cstring>
#include <limits>
#include <numeric>
#include <fstream>

namespace nxs {
namespace fem {

using Real = nxs::Real;

// ============================================================================
// Utility functions for XFEM Wave 41
// ============================================================================

namespace xfem41_detail {

inline Real dot3(const Real* a, const Real* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline Real norm3(const Real* v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

inline void normalize3(Real* v) {
    Real n = norm3(v);
    if (n > 1.0e-30) { v[0] /= n; v[1] /= n; v[2] /= n; }
}

inline Real distance3(const Real* a, const Real* b) {
    Real d[3] = {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
    return norm3(d);
}

inline Real clamp(Real x, Real lo, Real hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

} // namespace xfem41_detail

// ============================================================================
// 1. XFEMFatigueCrack — Paris-law fatigue crack growth
// ============================================================================

/**
 * @brief Paris-law fatigue crack growth with SIF computation and cycle counting.
 *
 * Paris law: da/dN = C * (DeltaK)^m
 * SIF: K = sigma * sqrt(pi * a) * F(a/W)  with geometry factor F.
 * Supports threshold and critical SIF range limits.
 */
class XFEMFatigueCrack {
public:
    struct Params {
        Real C_paris    = 1.0e-11;  ///< Paris law coefficient C
        Real m_paris    = 3.0;      ///< Paris law exponent m
        Real threshold_K = 5.0;    ///< Threshold SIF range (no growth below)
        Real critical_K  = 100.0;  ///< Critical SIF range (fracture above)
        Real initial_length = 0.001; ///< Initial crack length [m]
        Real specimen_width = 0.1;   ///< Specimen width W for geometry factor
    };

    XFEMFatigueCrack() = default;

    explicit XFEMFatigueCrack(const Params& p)
        : params_(p), crack_length_(p.initial_length), total_cycles_(0.0) {}

    /**
     * @brief Geometry correction factor F(a/W) for edge crack in finite plate.
     * Tada, Paris & Irwin approximation for single-edge notch.
     */
    Real geometry_factor(Real a, Real W) const {
        Real ratio = a / W;
        ratio = xfem41_detail::clamp(ratio, 0.0, 0.95);
        // Tada-Paris-Irwin correction for SEN specimen
        Real f = 1.12 - 0.231 * ratio + 10.55 * ratio * ratio
               - 21.72 * ratio * ratio * ratio
               + 30.39 * ratio * ratio * ratio * ratio;
        return f;
    }

    /**
     * @brief Compute SIF from remote stress and crack length.
     * K = sigma * sqrt(pi * a) * F(a/W)
     */
    Real compute_sif(Real sigma, Real a) const {
        if (a <= 0.0) return 0.0;
        Real F = geometry_factor(a, params_.specimen_width);
        return sigma * std::sqrt(M_PI * a) * F;
    }

    /**
     * @brief Compute Paris-law crack growth rate da/dN for given SIF range.
     * Returns 0 if delta_K < threshold, very large if >= critical.
     */
    Real compute_growth_rate(Real delta_K) const {
        if (delta_K <= params_.threshold_K) return 0.0;
        if (delta_K >= params_.critical_K) return 1.0e30; // rapid fracture
        return params_.C_paris * std::pow(delta_K, params_.m_paris);
    }

    /**
     * @brief Advance crack by a given number of cycles at specified SIF range.
     * @param cycles Number of load cycles
     * @param sif_range Delta-K (stress intensity factor range)
     * @return true if crack is still growing (not yet critical)
     */
    bool advance_crack(Real cycles, Real sif_range) {
        Real da_dN = compute_growth_rate(sif_range);
        if (da_dN >= 1.0e30) {
            crack_length_ = params_.specimen_width; // fracture
            return false;
        }
        Real da = da_dN * cycles;
        crack_length_ += da;
        total_cycles_ += cycles;
        growth_history_.push_back({total_cycles_, crack_length_, sif_range});

        // Check if crack has exceeded critical size
        if (crack_length_ >= 0.95 * params_.specimen_width) {
            return false;
        }
        return true;
    }

    /**
     * @brief Simple cycle counting: extract stress amplitude from a load history.
     *
     * Identifies peaks and valleys, returns vector of stress ranges.
     * (Simplified rainflow — counts adjacent reversals.)
     */
    std::vector<Real> count_cycles(const std::vector<Real>& stress_history) const {
        std::vector<Real> ranges;
        if (stress_history.size() < 3) return ranges;

        // Extract turning points (peaks and valleys)
        std::vector<Real> turning;
        turning.push_back(stress_history[0]);
        for (size_t i = 1; i + 1 < stress_history.size(); ++i) {
            Real prev = stress_history[i-1];
            Real curr = stress_history[i];
            Real next = stress_history[i+1];
            bool is_peak   = (curr >= prev && curr >= next);
            bool is_valley = (curr <= prev && curr <= next);
            if (is_peak || is_valley) {
                turning.push_back(curr);
            }
        }
        turning.push_back(stress_history.back());

        // Simple range counting: consecutive pairs of turning points
        for (size_t i = 0; i + 1 < turning.size(); ++i) {
            Real range = std::abs(turning[i+1] - turning[i]);
            if (range > 1.0e-15) {
                ranges.push_back(range);
            }
        }
        return ranges;
    }

    Real get_crack_length() const { return crack_length_; }
    Real get_total_cycles() const { return total_cycles_; }
    const Params& params() const { return params_; }

    struct GrowthRecord {
        Real cycles;
        Real length;
        Real sif_range;
    };

    const std::vector<GrowthRecord>& growth_history() const { return growth_history_; }

private:
    Params params_{};
    Real crack_length_ = 0.001;
    Real total_cycles_  = 0.0;
    std::vector<GrowthRecord> growth_history_;
};

// ============================================================================
// 2. XFEMMultiCrack — Multiple simultaneous cracks with interaction
// ============================================================================

/**
 * @brief Manages multiple simultaneous cracks with interaction effects.
 *
 * Each crack is tracked independently with its own level set.
 * Stress shielding reduces SIF when cracks are close.
 * Coalescence merges cracks when tips approach within tolerance.
 */
class XFEMMultiCrack {
public:
    struct CrackData {
        Real tip[3]       = {0.0, 0.0, 0.0};
        Real direction[3] = {1.0, 0.0, 0.0};
        Real length       = 0.0;
        Real sif_I        = 0.0;  ///< Mode I SIF
        Real sif_II       = 0.0;  ///< Mode II SIF
        bool active       = true;
        int  id           = 0;
    };

    XFEMMultiCrack() = default;

    /**
     * @brief Add a new crack to the system.
     * @return crack index
     */
    int add_crack(const Real tip[3], const Real direction[3], Real length) {
        CrackData c;
        std::memcpy(c.tip, tip, 3 * sizeof(Real));
        std::memcpy(c.direction, direction, 3 * sizeof(Real));
        xfem41_detail::normalize3(c.direction);
        c.length = length;
        c.id = static_cast<int>(cracks_.size());
        cracks_.push_back(c);
        return c.id;
    }

    /**
     * @brief Update all cracks given a uniform far-field stress.
     *
     * Computes SIF for each crack and applies stress shielding corrections
     * based on inter-crack distances.
     * @param stress_field 6-component stress tensor (Voigt: xx,yy,zz,xy,yz,xz)
     */
    void update_all(const Real stress_field[6]) {
        const Real sigma_xx = stress_field[0];
        const Real sigma_yy = stress_field[1];

        for (size_t i = 0; i < cracks_.size(); ++i) {
            if (!cracks_[i].active) continue;

            // Base SIF (Mode I from sigma_yy for a horizontal crack)
            Real a = cracks_[i].length * 0.5;
            if (a < 1.0e-15) continue;

            Real K_I_base  = sigma_yy * std::sqrt(M_PI * a);
            Real K_II_base = stress_field[3] * std::sqrt(M_PI * a); // tau_xy

            // Stress shielding: reduce SIF when other cracks are nearby
            Real shielding = 1.0;
            for (size_t j = 0; j < cracks_.size(); ++j) {
                if (i == j || !cracks_[j].active) continue;
                Real dist = xfem41_detail::distance3(cracks_[i].tip, cracks_[j].tip);
                Real char_len = std::max(cracks_[i].length, cracks_[j].length);
                if (char_len > 1.0e-15 && dist < 3.0 * char_len) {
                    // Shielding factor: closer cracks reduce SIF more
                    Real ratio = dist / char_len;
                    Real shield_factor = 1.0 - 0.3 * std::exp(-ratio);
                    shielding *= shield_factor;
                }
            }
            shielding = xfem41_detail::clamp(shielding, 0.1, 1.0);

            cracks_[i].sif_I  = K_I_base * shielding;
            cracks_[i].sif_II = K_II_base * shielding;
        }
    }

    /**
     * @brief Check for crack coalescence and merge cracks whose tips are
     *        within the given tolerance distance.
     * @return Number of coalescence events.
     */
    int check_coalescence(Real tol) {
        int merges = 0;
        for (size_t i = 0; i < cracks_.size(); ++i) {
            if (!cracks_[i].active) continue;
            for (size_t j = i + 1; j < cracks_.size(); ++j) {
                if (!cracks_[j].active) continue;
                Real dist = xfem41_detail::distance3(cracks_[i].tip, cracks_[j].tip);
                if (dist < tol) {
                    // Merge: extend crack i to include crack j
                    Real new_len = cracks_[i].length + cracks_[j].length + dist;
                    cracks_[i].length = new_len;
                    // Move tip to the farther extent
                    for (int d = 0; d < 3; ++d) {
                        cracks_[i].tip[d] = cracks_[j].tip[d]
                            + cracks_[j].direction[d] * cracks_[j].length * 0.5;
                    }
                    // Recompute direction from crack i start to new tip
                    cracks_[j].active = false;
                    ++merges;
                }
            }
        }
        return merges;
    }

    /**
     * @brief Propagate all active cracks by given increment using their SIFs.
     * @param da Crack growth increment
     */
    void propagate_all(Real da) {
        for (auto& c : cracks_) {
            if (!c.active) continue;
            if (c.sif_I <= 0.0 && c.sif_II <= 0.0) continue;

            // Maximum tangential stress criterion for direction
            Real theta = 0.0;
            if (std::abs(c.sif_II) > 1.0e-15) {
                Real ratio = c.sif_I / c.sif_II;
                theta = 2.0 * std::atan(0.25 * (ratio
                    - std::sqrt(ratio * ratio + 8.0)));
            }

            // Rotate direction by theta in 2D (about z-axis)
            Real cos_t = std::cos(theta);
            Real sin_t = std::sin(theta);
            Real new_dir[3] = {
                cos_t * c.direction[0] - sin_t * c.direction[1],
                sin_t * c.direction[0] + cos_t * c.direction[1],
                c.direction[2]
            };
            xfem41_detail::normalize3(new_dir);

            // Advance tip
            c.tip[0] += new_dir[0] * da;
            c.tip[1] += new_dir[1] * da;
            c.tip[2] += new_dir[2] * da;
            std::memcpy(c.direction, new_dir, 3 * sizeof(Real));
            c.length += da;
        }
    }

    size_t num_cracks() const { return cracks_.size(); }
    size_t num_active() const {
        return static_cast<size_t>(std::count_if(cracks_.begin(), cracks_.end(),
            [](const CrackData& c) { return c.active; }));
    }
    const CrackData& crack(int i) const { return cracks_[static_cast<size_t>(i)]; }
    CrackData& crack(int i) { return cracks_[static_cast<size_t>(i)]; }
    const std::vector<CrackData>& cracks() const { return cracks_; }

private:
    std::vector<CrackData> cracks_;
};

// ============================================================================
// 3. XFEMAdaptiveMesh — h-adaptive mesh refinement around crack tips
// ============================================================================

/**
 * @brief h-adaptive mesh refinement using Zienkiewicz-Zhu error indicator.
 *
 * Computes element-wise error based on stress recovery, marks elements
 * for refinement or coarsening, and produces a refined mesh description.
 */
class XFEMAdaptiveMesh {
public:
    struct RefinementInfo {
        std::vector<int> refined_elements;   ///< Elements that were refined
        std::vector<int> coarsened_elements; ///< Elements that were coarsened
        int new_nodes    = 0;  ///< Number of new nodes created
        int new_elements = 0;  ///< Number of new elements created
    };

    struct ElementData {
        int id = 0;
        Real centroid[3] = {0.0, 0.0, 0.0};
        Real size = 0.0;    ///< Characteristic element size (e.g., edge length)
        int  level = 0;     ///< Refinement level (0 = original)
    };

    XFEMAdaptiveMesh() = default;

    /**
     * @brief Compute Zienkiewicz-Zhu error indicator for each element.
     *
     * Error = ||sigma_h - sigma*||_L2 / ||sigma*||_L2
     * where sigma_h is the FE stress and sigma* is the recovered (smoothed) stress.
     *
     * @param elements Element data (centroids, sizes)
     * @param stresses Per-element stress tensors (6 components each, Voigt)
     * @return Per-element error indicators in [0, inf)
     */
    std::vector<Real> compute_error(const std::vector<ElementData>& elements,
                                     const std::vector<std::array<Real,6>>& stresses) const
    {
        size_t n = elements.size();
        std::vector<Real> errors(n, 0.0);
        if (n == 0 || stresses.size() != n) return errors;

        // Step 1: Compute global average stress (nodal recovery approximation)
        // In a real implementation, this would do patch-based SPR recovery.
        // Here we use a distance-weighted average as a simplified ZZ estimator.
        for (size_t i = 0; i < n; ++i) {
            Real sigma_star[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            Real weight_sum = 0.0;

            for (size_t j = 0; j < n; ++j) {
                Real dist = xfem41_detail::distance3(elements[i].centroid,
                                                      elements[j].centroid);
                Real h = elements[i].size;
                if (h < 1.0e-15) h = 1.0;
                Real w = std::exp(-dist * dist / (h * h));
                for (int k = 0; k < 6; ++k) {
                    sigma_star[k] += w * stresses[j][k];
                }
                weight_sum += w;
            }

            if (weight_sum > 1.0e-30) {
                for (int k = 0; k < 6; ++k) {
                    sigma_star[k] /= weight_sum;
                }
            }

            // Error = ||sigma_h - sigma*|| / ||sigma*||
            Real diff_norm_sq = 0.0;
            Real star_norm_sq = 0.0;
            for (int k = 0; k < 6; ++k) {
                Real diff = stresses[i][k] - sigma_star[k];
                diff_norm_sq += diff * diff;
                star_norm_sq += sigma_star[k] * sigma_star[k];
            }

            if (star_norm_sq > 1.0e-30) {
                errors[i] = std::sqrt(diff_norm_sq / star_norm_sq);
            } else {
                errors[i] = std::sqrt(diff_norm_sq);
            }
        }

        return errors;
    }

    /**
     * @brief Mark elements for refinement and coarsening based on error.
     *
     * @param errors Per-element error indicators
     * @param refine_threshold Elements with error > this are marked for refinement
     * @param coarsen_threshold Elements with error < this are marked for coarsening
     */
    void mark_refine(const std::vector<Real>& errors,
                     Real refine_threshold,
                     Real coarsen_threshold = -1.0)
    {
        refine_list_.clear();
        coarsen_list_.clear();

        for (size_t i = 0; i < errors.size(); ++i) {
            if (errors[i] > refine_threshold) {
                refine_list_.push_back(static_cast<int>(i));
            } else if (coarsen_threshold > 0.0 && errors[i] < coarsen_threshold) {
                coarsen_list_.push_back(static_cast<int>(i));
            }
        }
    }

    /**
     * @brief Perform mesh refinement by subdividing marked elements.
     *
     * Each marked quad is split into 4 sub-elements (bisection).
     * Each marked triangle is split into 4 sub-triangles.
     *
     * @param elements Current element data
     * @param max_level Maximum refinement level allowed
     * @return RefinementInfo describing the changes
     */
    RefinementInfo refine(std::vector<ElementData>& elements, int max_level = 5) const {
        RefinementInfo info;

        for (int idx : refine_list_) {
            if (idx < 0 || idx >= static_cast<int>(elements.size())) continue;
            if (elements[static_cast<size_t>(idx)].level >= max_level) continue;

            info.refined_elements.push_back(idx);
            // Each element splits into 4 children
            Real h_new = elements[static_cast<size_t>(idx)].size * 0.5;
            int new_level = elements[static_cast<size_t>(idx)].level + 1;
            Real cx = elements[static_cast<size_t>(idx)].centroid[0];
            Real cy = elements[static_cast<size_t>(idx)].centroid[1];
            Real cz = elements[static_cast<size_t>(idx)].centroid[2];

            // Create 4 sub-element centroids (2x2 subdivision)
            Real offsets[4][2] = {
                {-0.25 * elements[static_cast<size_t>(idx)].size,
                 -0.25 * elements[static_cast<size_t>(idx)].size},
                { 0.25 * elements[static_cast<size_t>(idx)].size,
                 -0.25 * elements[static_cast<size_t>(idx)].size},
                {-0.25 * elements[static_cast<size_t>(idx)].size,
                  0.25 * elements[static_cast<size_t>(idx)].size},
                { 0.25 * elements[static_cast<size_t>(idx)].size,
                  0.25 * elements[static_cast<size_t>(idx)].size}
            };

            for (int c = 0; c < 4; ++c) {
                ElementData child;
                child.id = static_cast<int>(elements.size()) + info.new_elements;
                child.centroid[0] = cx + offsets[c][0];
                child.centroid[1] = cy + offsets[c][1];
                child.centroid[2] = cz;
                child.size = h_new;
                child.level = new_level;
                elements.push_back(child);
                ++info.new_elements;
            }
            // Count new nodes: each subdivision adds ~5 new mid-edge/mid-face nodes
            info.new_nodes += 5;
        }

        for (int idx : coarsen_list_) {
            if (idx >= 0 && idx < static_cast<int>(elements.size())) {
                if (elements[static_cast<size_t>(idx)].level > 0) {
                    info.coarsened_elements.push_back(idx);
                }
            }
        }

        return info;
    }

    const std::vector<int>& refine_list() const { return refine_list_; }
    const std::vector<int>& coarsen_list() const { return coarsen_list_; }

private:
    std::vector<int> refine_list_;
    std::vector<int> coarsen_list_;
};

// ============================================================================
// 4. XFEMOutputFields — Crack path, SIF, COD extraction and VTK output
// ============================================================================

/**
 * @brief Extract and export XFEM crack data for post-processing.
 *
 * Extracts crack path from level-set zero contour, computes crack opening
 * displacement (COD), and writes VTK polydata files.
 */
class XFEMOutputFields {
public:
    struct Point3 {
        Real x = 0.0, y = 0.0, z = 0.0;
    };

    struct CrackPathData {
        std::vector<Point3> path_points;       ///< Polyline vertices
        std::vector<Real>   sif_along_front;   ///< SIF at each front point
        std::vector<Real>   cod_values;        ///< COD at each point
    };

    XFEMOutputFields() = default;

    /**
     * @brief Extract crack path as polyline from level-set zero contour.
     *
     * Given a structured grid of level-set values, find the zero iso-surface
     * using linear interpolation on element edges.
     *
     * @param level_set Level-set values at grid nodes
     * @param node_coords Node coordinates (3 per node, flattened)
     * @param connectivity Element connectivity (4 per quad, flattened)
     * @param n_elements Number of elements
     * @return Ordered list of crack path points
     */
    std::vector<Point3> extract_crack_path(
        const std::vector<Real>& level_set,
        const std::vector<Real>& node_coords,
        const std::vector<int>&  connectivity,
        int n_elements) const
    {
        std::vector<Point3> path;

        for (int e = 0; e < n_elements; ++e) {
            int base = e * 4;
            if (base + 3 >= static_cast<int>(connectivity.size())) break;

            int nodes[4];
            for (int i = 0; i < 4; ++i) {
                nodes[i] = connectivity[static_cast<size_t>(base + i)];
            }

            // Check each edge for zero crossing
            int edges[4][2] = {{0,1},{1,2},{2,3},{3,0}};
            std::vector<Point3> crossings;

            for (auto& edge : edges) {
                int n0 = nodes[edge[0]];
                int n1 = nodes[edge[1]];
                if (n0 < 0 || n1 < 0) continue;
                if (static_cast<size_t>(n0) >= level_set.size() ||
                    static_cast<size_t>(n1) >= level_set.size()) continue;

                Real phi0 = level_set[static_cast<size_t>(n0)];
                Real phi1 = level_set[static_cast<size_t>(n1)];

                if (phi0 * phi1 < 0.0) {
                    // Zero crossing on this edge
                    Real t = phi0 / (phi0 - phi1);
                    t = xfem41_detail::clamp(t, 0.0, 1.0);
                    size_t i0 = static_cast<size_t>(n0) * 3;
                    size_t i1 = static_cast<size_t>(n1) * 3;
                    if (i0 + 2 < node_coords.size() && i1 + 2 < node_coords.size()) {
                        Point3 p;
                        p.x = (1.0 - t) * node_coords[i0]     + t * node_coords[i1];
                        p.y = (1.0 - t) * node_coords[i0 + 1] + t * node_coords[i1 + 1];
                        p.z = (1.0 - t) * node_coords[i0 + 2] + t * node_coords[i1 + 2];
                        crossings.push_back(p);
                    }
                }
            }

            // Add midpoint of crossing pair to path
            if (crossings.size() == 2) {
                Point3 mid;
                mid.x = 0.5 * (crossings[0].x + crossings[1].x);
                mid.y = 0.5 * (crossings[0].y + crossings[1].y);
                mid.z = 0.5 * (crossings[0].z + crossings[1].z);
                path.push_back(crossings[0]);
                path.push_back(crossings[1]);
            } else if (crossings.size() == 1) {
                path.push_back(crossings[0]);
            }
        }

        // Sort path points by x then y for a reasonable polyline ordering
        std::sort(path.begin(), path.end(), [](const Point3& a, const Point3& b) {
            if (std::abs(a.x - b.x) > 1.0e-12) return a.x < b.x;
            return a.y < b.y;
        });

        // Remove duplicates (within tolerance)
        std::vector<Point3> unique_path;
        for (const auto& p : path) {
            bool duplicate = false;
            for (const auto& q : unique_path) {
                Real dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
                if (std::sqrt(dx*dx + dy*dy + dz*dz) < 1.0e-10) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) unique_path.push_back(p);
        }

        return unique_path;
    }

    /**
     * @brief Compute crack opening displacement (COD) along the crack path.
     *
     * COD is the jump in displacement across the crack.
     * Approximated as 2 * |u_n| at each path point using nearest-node interpolation.
     *
     * @param path_points Crack path points
     * @param displacements Nodal displacement field (3 per node, flattened)
     * @param node_coords Node coordinates (3 per node, flattened)
     * @param n_nodes Number of nodes
     * @param crack_normal Normal to crack plane [3]
     * @return COD value at each path point
     */
    std::vector<Real> compute_cod(
        const std::vector<Point3>& path_points,
        const std::vector<Real>& displacements,
        const std::vector<Real>& node_coords,
        int n_nodes,
        const Real crack_normal[3] = nullptr) const
    {
        Real default_normal[3] = {0.0, 1.0, 0.0};
        const Real* normal = (crack_normal != nullptr) ? crack_normal : default_normal;

        std::vector<Real> cod(path_points.size(), 0.0);

        for (size_t i = 0; i < path_points.size(); ++i) {
            // Find nearest node
            Real min_dist = std::numeric_limits<Real>::max();
            int nearest = -1;
            for (int n = 0; n < n_nodes; ++n) {
                size_t idx = static_cast<size_t>(n) * 3;
                if (idx + 2 >= node_coords.size()) break;
                Real dx = path_points[i].x - node_coords[idx];
                Real dy = path_points[i].y - node_coords[idx + 1];
                Real dz = path_points[i].z - node_coords[idx + 2];
                Real dist = dx*dx + dy*dy + dz*dz;
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest = n;
                }
            }

            if (nearest >= 0) {
                size_t uidx = static_cast<size_t>(nearest) * 3;
                if (uidx + 2 < displacements.size()) {
                    // COD approximated as 2 * normal component of displacement
                    Real u_n = displacements[uidx] * normal[0]
                             + displacements[uidx + 1] * normal[1]
                             + displacements[uidx + 2] * normal[2];
                    cod[i] = 2.0 * std::abs(u_n);
                }
            }
        }
        return cod;
    }

    /**
     * @brief Write crack path as VTK polydata file for ParaView visualization.
     *
     * @param filename Output VTK filename
     * @param path Crack path data (points, SIF, COD)
     * @return true on success
     */
    bool write_vtk(const std::string& filename, const CrackPathData& path) const {
        std::ofstream out(filename);
        if (!out.is_open()) return false;

        size_t n = path.path_points.size();

        out << "# vtk DataFile Version 3.0\n";
        out << "XFEM Crack Path\n";
        out << "ASCII\n";
        out << "DATASET POLYDATA\n";

        // Points
        out << "POINTS " << n << " double\n";
        for (const auto& p : path.path_points) {
            out << p.x << " " << p.y << " " << p.z << "\n";
        }

        // Lines (polyline connecting all points)
        if (n > 1) {
            out << "LINES 1 " << (n + 1) << "\n";
            out << n;
            for (size_t i = 0; i < n; ++i) {
                out << " " << i;
            }
            out << "\n";
        }

        // Point data
        out << "POINT_DATA " << n << "\n";

        // SIF field
        if (path.sif_along_front.size() == n) {
            out << "SCALARS SIF double 1\n";
            out << "LOOKUP_TABLE default\n";
            for (const auto& v : path.sif_along_front) {
                out << v << "\n";
            }
        }

        // COD field
        if (path.cod_values.size() == n) {
            out << "SCALARS COD double 1\n";
            out << "LOOKUP_TABLE default\n";
            for (const auto& v : path.cod_values) {
                out << v << "\n";
            }
        }

        out.close();
        return true;
    }
};

} // namespace fem
} // namespace nxs
