#pragma once

/**
 * @file contact_wave44.hpp
 * @brief Wave 44a: Contact Gap Evolution
 *
 * Provides composable wrappers for per-node gap tracking in contact detection.
 * Does NOT modify any existing ContactMechanics class.
 *
 * Sub-modules:
 * - GapMode                — IGAP=0/1/2 enum (Constant / Variable / VariableScaled)
 * - ContactGapEvolution    — Per-slave-node gap arrays, update logic, deletion flags
 * - GapAwarePairFilter     — Composable filter for ContactBucketSearch pair lists
 * - BilateralGapHandler    — Symmetric surface gap computation
 *
 * References:
 * - OpenRadioss IGAP contact gap definition (Starter manual §14.x)
 * - Hallquist (2006) "LS-DYNA Theory Manual" §26 contact thickness
 */

#include <nexussim/core/types.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

namespace nxs {
namespace fem {

using Real = nxs::Real;

// ============================================================================
// GapMode — corresponds to OpenRadioss IGAP parameter
// ============================================================================

/**
 * @brief Controls how per-node contact gaps evolve during the simulation.
 *
 * Constant      (IGAP=0): gap is fixed at its initial value throughout.
 * Variable      (IGAP=1): gap scales with (current thickness / initial thickness).
 * VariableScaled(IGAP=2): Variable + an additional user-supplied scale factor.
 */
enum class GapMode : int {
    Constant       = 0,  ///< IGAP=0 — gap never changes
    Variable       = 1,  ///< IGAP=1 — gap tracks shell thinning
    VariableScaled = 2   ///< IGAP=2 — Variable with extra scale factor
};

// ============================================================================
// Internal helpers
// ============================================================================

namespace wave44_detail {

KOKKOS_INLINE_FUNCTION
Real w44_max(Real a, Real b) { return a > b ? a : b; }

KOKKOS_INLINE_FUNCTION
Real w44_min(Real a, Real b) { return a < b ? a : b; }

KOKKOS_INLINE_FUNCTION
Real w44_clamp(Real v, Real lo, Real hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/// Sentinel value used to mark deleted nodes
static constexpr Real DELETED_GAP = std::numeric_limits<Real>::infinity();

/// Minimum thickness allowed in the denominator to prevent division by zero
static constexpr Real MIN_THICKNESS = Real(1.0e-30);

} // namespace wave44_detail

// ============================================================================
// ContactGapEvolution
// ============================================================================

/**
 * @brief Tracks per-slave-node contact gaps and updates them as shells thin.
 *
 * This class is a pure data/algorithm container; it does not hold references
 * to any existing contact or mesh objects so it can be composed freely.
 *
 * Usage pattern:
 * @code
 *   ContactGapEvolution cge(num_nodes, initial_thickness, GapMode::Variable);
 *   // ... time step ...
 *   cge.update_gaps(positions, new_thicknesses, num_nodes);
 *   Real g = cge.get_effective_gap(slave_id);
 * @endcode
 */
class ContactGapEvolution {
public:
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /**
     * @brief Construct gap evolution tracker.
     *
     * @param num_nodes         Number of slave nodes to track.
     * @param shell_thickness   Uniform initial shell half-thickness (gap = half-thickness).
     * @param mode              Gap evolution mode (IGAP=0/1/2).
     */
    ContactGapEvolution(std::size_t num_nodes,
                        Real        shell_thickness,
                        GapMode     mode = GapMode::Constant)
        : mode_(mode)
        , scale_factor_(Real(1))
        , gap_min_limit_(Real(0))
        , gap_max_limit_(wave44_detail::DELETED_GAP)
    {
        const Real t0 = std::max(shell_thickness, wave44_detail::MIN_THICKNESS);
        gap_s0_.assign(num_nodes, t0);
        gap_s_.assign(num_nodes, t0);
        thknod0_.assign(num_nodes, t0);
        deleted_.assign(num_nodes, false);
    }

    // ------------------------------------------------------------------
    // Core update
    // ------------------------------------------------------------------

    /**
     * @brief Recompute per-node gaps from current shell thicknesses.
     *
     * For GapMode::Constant: no-op (gaps stay at initial values).
     * For GapMode::Variable: gap_s[i] = gap_s0[i] * (thk[i] / thknod0[i]).
     * For GapMode::VariableScaled: same as Variable, then multiply by scale_factor_.
     *
     * Deleted nodes are left at DELETED_GAP.
     * Results are clamped to [gap_min_limit_, gap_max_limit_] if set.
     *
     * @param positions         Unused (reserved for future position-dependent gap).
     * @param shell_thicknesses Current per-node shell thickness array (length >= num_nodes).
     * @param num_nodes         Number of entries to process.
     */
    void update_gaps(const Real* /*positions*/,
                     const Real* shell_thicknesses,
                     std::size_t num_nodes)
    {
        const std::size_t n = std::min(num_nodes, gap_s_.size());

        if (mode_ == GapMode::Constant) {
            // Nothing to do — gaps stay at their initial values
            return;
        }

        for (std::size_t i = 0; i < n; ++i) {
            if (deleted_[i]) continue;

            const Real t0  = std::max(thknod0_[i], wave44_detail::MIN_THICKNESS);
            const Real t   = std::max(shell_thicknesses[i], wave44_detail::MIN_THICKNESS);
            Real g         = gap_s0_[i] * (t / t0);

            if (mode_ == GapMode::VariableScaled) {
                g *= scale_factor_;
            }

            // Apply clamping if limits are set
            g = wave44_detail::w44_clamp(g, gap_min_limit_, gap_max_limit_);

            gap_s_[i] = g;
        }
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /**
     * @brief Return the effective contact gap for a given slave node.
     *
     * Deleted nodes return +infinity so they are never selected as candidates.
     */
    KOKKOS_INLINE_FUNCTION
    Real get_effective_gap(std::size_t slave_node) const {
        if (slave_node >= gap_s_.size()) return wave44_detail::DELETED_GAP;
        if (deleted_[slave_node])        return wave44_detail::DELETED_GAP;
        return gap_s_[slave_node];
    }

    /// Maximum gap across all non-deleted nodes (useful for BucketSort3D search radius).
    Real max_gap() const {
        Real mx = Real(0);
        for (std::size_t i = 0; i < gap_s_.size(); ++i) {
            if (!deleted_[i]) mx = wave44_detail::w44_max(mx, gap_s_[i]);
        }
        return mx;
    }

    /// Minimum gap across all non-deleted nodes.
    Real min_gap() const {
        Real mn = wave44_detail::DELETED_GAP;
        for (std::size_t i = 0; i < gap_s_.size(); ++i) {
            if (!deleted_[i])
                mn = wave44_detail::w44_min(mn, gap_s_[i]);
        }
        return mn;
    }

    std::size_t num_nodes() const { return gap_s_.size(); }
    GapMode     gap_mode()  const { return mode_; }

    // ------------------------------------------------------------------
    // Deletion
    // ------------------------------------------------------------------

    /// Mark a node as deleted (eroded element). Gap returns infinity.
    void set_deleted(std::size_t node) {
        if (node < deleted_.size()) {
            deleted_[node] = true;
            gap_s_[node]   = wave44_detail::DELETED_GAP;
        }
    }

    bool is_deleted(std::size_t node) const {
        if (node >= deleted_.size()) return false;
        return deleted_[node];
    }

    // ------------------------------------------------------------------
    // Configuration setters
    // ------------------------------------------------------------------

    /**
     * @brief Set the additional scale factor for IGAP=2 mode.
     *
     * Has no effect in Constant or Variable mode.
     */
    void apply_gap_scale(Real scale_factor) {
        scale_factor_ = scale_factor;
    }

    /**
     * @brief Clamp gap values to [min_gap, max_gap] on every update_gaps() call.
     *
     * Pass 0.0 and +inf to disable clamping (default).
     */
    void set_gap_limits(Real min_g, Real max_g) {
        gap_min_limit_ = min_g;
        gap_max_limit_ = max_g;
    }

    /**
     * @brief Reset all nodes to a uniform thickness / gap state.
     *
     * Clears deletion flags. Does not change the mode or scale factor.
     *
     * @param uniform_thickness New uniform shell half-thickness.
     */
    void reset(Real uniform_thickness) {
        const Real t = std::max(uniform_thickness, wave44_detail::MIN_THICKNESS);
        const std::size_t n = gap_s_.size();
        for (std::size_t i = 0; i < n; ++i) {
            gap_s0_[i]   = t;
            gap_s_[i]    = t;
            thknod0_[i]  = t;
            deleted_[i]  = false;
        }
        // Re-apply max limit if set
        if (gap_max_limit_ < wave44_detail::DELETED_GAP) {
            for (std::size_t i = 0; i < n; ++i)
                gap_s_[i] = wave44_detail::w44_clamp(gap_s_[i], gap_min_limit_, gap_max_limit_);
        }
    }

    // ------------------------------------------------------------------
    // Raw array accessors (for testing / advanced use)
    // ------------------------------------------------------------------

    const std::vector<Real>& gap_s()     const { return gap_s_; }
    const std::vector<Real>& gap_s0()    const { return gap_s0_; }
    const std::vector<Real>& thknod0()   const { return thknod0_; }
    const std::vector<bool>& deleted()   const { return deleted_; }

private:
    GapMode              mode_;
    Real                 scale_factor_;
    Real                 gap_min_limit_;
    Real                 gap_max_limit_;

    std::vector<Real>    gap_s_;      ///< Current per-node gap
    std::vector<Real>    gap_s0_;     ///< Initial per-node gap (reference)
    std::vector<Real>    thknod0_;    ///< Initial per-node thickness (reference)
    std::vector<bool>    deleted_;    ///< Deletion flags
};

// ============================================================================
// GapAwarePairFilter
// ============================================================================

/**
 * @brief Composable filter that wraps ContactGapEvolution for pair filtering.
 *
 * Intended to be applied after ContactBucketSearch broad-phase produces a
 * candidate pair list.  For each (slave_node, penetration) pair, the filter
 * decides whether the pair should be kept based on whether the penetration
 * depth exceeds the node's effective gap.
 *
 * Usage:
 * @code
 *   GapAwarePairFilter filter(cge, /*tolerance=*\/ 0.0);
 *   bool keep = filter.filter_pair(slave_id, penetration, tolerance);
 *   Real search_dist = filter.adjusted_search_distance();
 * @endcode
 */
class GapAwarePairFilter {
public:
    /**
     * @brief Construct filter.
     *
     * @param cge        Reference to the ContactGapEvolution that owns node gaps.
     * @param margin     Additional tolerance added to adjusted_search_distance().
     */
    explicit GapAwarePairFilter(const ContactGapEvolution& cge,
                                Real margin = Real(0))
        : cge_(cge), margin_(margin)
    {}

    /**
     * @brief Decide whether a candidate pair should be retained.
     *
     * A pair is kept when:
     *   penetration  >  get_effective_gap(slave_node) - gap_tolerance
     *
     * i.e., the surfaces are closer than the gap threshold.
     *
     * @param slave_node    Index of the slave node in the ContactGapEvolution.
     * @param penetration   Signed penetration depth (positive = overlap).
     * @param gap_tolerance Additional tolerance subtracted from the gap.
     * @return true if the pair should be retained for narrow-phase processing.
     */
    bool filter_pair(std::size_t slave_node,
                     Real        penetration,
                     Real        gap_tolerance) const
    {
        if (cge_.is_deleted(slave_node)) return false;
        const Real effective = cge_.get_effective_gap(slave_node) - gap_tolerance;
        return penetration > effective;
    }

    /**
     * @brief Return the search distance that should be fed to BucketSort3D.
     *
     * Equals max_gap() + margin_ so that the broad-phase bucket search
     * captures all nodes that could possibly be within gap distance.
     */
    Real adjusted_search_distance() const {
        return cge_.max_gap() + margin_;
    }

    const ContactGapEvolution& gap_evolution() const { return cge_; }

private:
    const ContactGapEvolution& cge_;
    Real                       margin_;
};

// ============================================================================
// BilateralGapHandler
// ============================================================================

/**
 * @brief Handles symmetric contacts where either surface can be slave/master.
 *
 * In bilateral (two-pass) contact, both surfaces carry gap information.
 * The effective gap for a given pair is the sum of the individual node gaps,
 * which represents the total combined thickness contribution from both sides.
 *
 * Usage:
 * @code
 *   BilateralGapHandler bgh;
 *   Real g = bgh.compute_bilateral_gap(node_a, node_b, gap_evo_a, gap_evo_b);
 * @endcode
 */
class BilateralGapHandler {
public:
    BilateralGapHandler() = default;

    /**
     * @brief Compute the combined gap for a bilateral contact pair.
     *
     * Returns the sum of the effective gaps from both surfaces.
     * If either node is deleted, returns +infinity.
     *
     * @param node_a   Slave node index on surface A.
     * @param node_b   Slave node index on surface B (treated symmetrically).
     * @param gap_a    ContactGapEvolution for surface A.
     * @param gap_b    ContactGapEvolution for surface B.
     * @return Combined gap = gap_a[node_a] + gap_b[node_b].
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_bilateral_gap(std::size_t                node_a,
                               std::size_t                node_b,
                               const ContactGapEvolution& gap_a,
                               const ContactGapEvolution& gap_b) const
    {
        if (gap_a.is_deleted(node_a) || gap_b.is_deleted(node_b))
            return wave44_detail::DELETED_GAP;
        return gap_a.get_effective_gap(node_a) + gap_b.get_effective_gap(node_b);
    }

    /**
     * @brief Check whether a bilateral pair is in contact.
     *
     * A pair is considered "in contact" when the distance between the nodes
     * is less than their combined bilateral gap.
     *
     * @param distance   Current surface-to-surface distance (non-negative).
     * @param node_a     Slave node index on surface A.
     * @param node_b     Slave node index on surface B.
     * @param gap_a      ContactGapEvolution for surface A.
     * @param gap_b      ContactGapEvolution for surface B.
     * @return true if distance < bilateral gap.
     */
    bool in_contact(Real                       distance,
                    std::size_t                node_a,
                    std::size_t                node_b,
                    const ContactGapEvolution& gap_a,
                    const ContactGapEvolution& gap_b) const
    {
        const Real bg = compute_bilateral_gap(node_a, node_b, gap_a, gap_b);
        if (bg >= wave44_detail::DELETED_GAP) return false;
        return distance < bg;
    }
};

} // namespace fem
} // namespace nxs
