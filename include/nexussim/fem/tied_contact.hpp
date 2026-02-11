#pragma once

/**
 * @file tied_contact.hpp
 * @brief Tied contact for spot welds, adhesive joints, and mesh tying
 *
 * Supports:
 * - TiedAllDOF: All DOFs constrained (welded)
 * - TiedSlidingOnly: Normal tied, tangential free (glued)
 * - TiedWithFailure: Tied until force limit reached (spot weld rupture)
 *
 * Reference: OpenRadioss TYPE2 contact (tied interface)
 */

#include <nexussim/core/types.hpp>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>

namespace nxs {
namespace fem {

// ============================================================================
// Tied Contact Types
// ============================================================================

enum class TiedContactType {
    TiedAllDOF,       ///< All DOFs constrained
    TiedSlidingOnly,  ///< Normal tied, tangential sliding allowed
    TiedWithFailure   ///< Tied with force-based rupture
};

// ============================================================================
// Tied Pair (one slave node to one master segment)
// ============================================================================

struct TiedPair {
    Index slave_node;          ///< Slave node index
    Index master_nodes[4];     ///< Master segment nodes (up to 4 for quad)
    Real phi[4];               ///< Interpolation weights on master
    int num_master_nodes;      ///< Number of master nodes (3=tri, 4=quad)
    Real gap_initial[3];       ///< Initial gap vector
    bool active;               ///< Is this pair still tied?

    // Failure tracking
    Real accumulated_force;    ///< Accumulated normal force magnitude
    Real max_force;            ///< Maximum force experienced

    TiedPair() : slave_node(0), num_master_nodes(0), active(true)
               , accumulated_force(0.0), max_force(0.0) {
        for (int i = 0; i < 4; ++i) { master_nodes[i] = 0; phi[i] = 0.0; }
        gap_initial[0] = gap_initial[1] = gap_initial[2] = 0.0;
    }
};

// ============================================================================
// Tied Contact Configuration
// ============================================================================

struct TiedContactConfig {
    TiedContactType type;
    int id;
    std::string name;

    Real penalty_stiffness;    ///< Penalty stiffness for constraint
    Real failure_force;        ///< Force limit for TiedWithFailure
    Real failure_moment;       ///< Moment limit for TiedWithFailure

    TiedContactConfig()
        : type(TiedContactType::TiedAllDOF), id(0)
        , penalty_stiffness(1.0e10)
        , failure_force(1.0e20)
        , failure_moment(1.0e20) {}
};

// ============================================================================
// Tied Contact Manager
// ============================================================================

class TiedContact {
public:
    TiedContact() = default;

    void set_config(const TiedContactConfig& cfg) { config_ = cfg; }
    const TiedContactConfig& config() const { return config_; }

    // --- Setup ---

    /**
     * @brief Add a tied pair manually
     */
    TiedPair& add_pair() {
        pairs_.emplace_back();
        return pairs_.back();
    }

    /**
     * @brief Create tied pairs by proximity search
     * Find closest master node for each slave node within tolerance
     */
    void find_tied_pairs(const std::vector<Index>& slave_nodes,
                          const std::vector<Index>& master_nodes,
                          const Real* positions,
                          Real tolerance = 1.0e-3) {
        for (Index s : slave_nodes) {
            Real min_dist = tolerance;
            Index best_master = 0;
            bool found = false;

            for (Index m : master_nodes) {
                Real dx = positions[3*s+0] - positions[3*m+0];
                Real dy = positions[3*s+1] - positions[3*m+1];
                Real dz = positions[3*s+2] - positions[3*m+2];
                Real dist = std::sqrt(dx*dx + dy*dy + dz*dz);

                if (dist < min_dist) {
                    min_dist = dist;
                    best_master = m;
                    found = true;
                }
            }

            if (found) {
                TiedPair pair;
                pair.slave_node = s;
                pair.master_nodes[0] = best_master;
                pair.num_master_nodes = 1;
                pair.phi[0] = 1.0;
                pair.gap_initial[0] = positions[3*s+0] - positions[3*best_master+0];
                pair.gap_initial[1] = positions[3*s+1] - positions[3*best_master+1];
                pair.gap_initial[2] = positions[3*s+2] - positions[3*best_master+2];
                pair.active = true;
                pairs_.push_back(pair);
            }
        }
    }

    std::size_t num_pairs() const { return pairs_.size(); }
    std::size_t num_active_pairs() const {
        std::size_t count = 0;
        for (const auto& p : pairs_) { if (p.active) count++; }
        return count;
    }

    TiedPair& pair(std::size_t i) { return pairs_[i]; }
    const TiedPair& pair(std::size_t i) const { return pairs_[i]; }

    // --- Constraint Application ---

    /**
     * @brief Apply tied constraints (penalty-based)
     * @param positions Node positions
     * @param velocities Node velocities (modified for tied nodes)
     * @param forces Force vector (penalty forces added)
     * @param dt Time step
     */
    void apply_tied_constraints(const Real* positions, Real* velocities,
                                 Real* forces, Real dt) {
        Real k = config_.penalty_stiffness;

        for (auto& pair : pairs_) {
            if (!pair.active) continue;

            Index s = pair.slave_node;

            // Compute master point position (weighted average)
            Real xm[3] = {0, 0, 0};
            Real vm[3] = {0, 0, 0};
            for (int j = 0; j < pair.num_master_nodes; ++j) {
                Index m = pair.master_nodes[j];
                Real w = pair.phi[j];
                xm[0] += w * positions[3*m+0];
                xm[1] += w * positions[3*m+1];
                xm[2] += w * positions[3*m+2];
                vm[0] += w * velocities[3*m+0];
                vm[1] += w * velocities[3*m+1];
                vm[2] += w * velocities[3*m+2];
            }

            // Gap = slave - master - initial_gap
            Real gap[3];
            gap[0] = positions[3*s+0] - xm[0] - pair.gap_initial[0];
            gap[1] = positions[3*s+1] - xm[1] - pair.gap_initial[1];
            gap[2] = positions[3*s+2] - xm[2] - pair.gap_initial[2];

            // Penalty force
            Real fn = std::sqrt(gap[0]*gap[0] + gap[1]*gap[1] + gap[2]*gap[2]);

            if (config_.type == TiedContactType::TiedAllDOF) {
                // Constrain all DOFs
                for (int d = 0; d < 3; ++d) {
                    Real f = -k * gap[d];
                    forces[3*s+d] += f;
                    // Distribute reaction to master nodes
                    for (int j = 0; j < pair.num_master_nodes; ++j) {
                        Index m = pair.master_nodes[j];
                        forces[3*m+d] -= pair.phi[j] * f;
                    }
                }
            } else if (config_.type == TiedContactType::TiedSlidingOnly) {
                // Only constrain normal gap (simplified: use initial gap direction)
                Real gap_n = gap[0]*pair.gap_initial[0] + gap[1]*pair.gap_initial[1]
                           + gap[2]*pair.gap_initial[2];
                Real n_mag = std::sqrt(pair.gap_initial[0]*pair.gap_initial[0] +
                                        pair.gap_initial[1]*pair.gap_initial[1] +
                                        pair.gap_initial[2]*pair.gap_initial[2]);
                if (n_mag > 1.0e-20) {
                    Real nx = pair.gap_initial[0]/n_mag;
                    Real ny = pair.gap_initial[1]/n_mag;
                    Real nz = pair.gap_initial[2]/n_mag;
                    Real f = -k * gap_n / n_mag;
                    forces[3*s+0] += f * nx;
                    forces[3*s+1] += f * ny;
                    forces[3*s+2] += f * nz;
                }
            }

            // Track force for failure check
            Real force_mag = k * fn;
            pair.accumulated_force += force_mag * dt;
            if (force_mag > pair.max_force) pair.max_force = force_mag;

            // Check failure
            if (config_.type == TiedContactType::TiedWithFailure) {
                if (force_mag > config_.failure_force) {
                    pair.active = false;
                }
            }
        }
    }

    // --- Statistics ---

    struct TiedStats {
        std::size_t total_pairs;
        std::size_t active_pairs;
        std::size_t failed_pairs;
        Real max_force;
    };

    TiedStats get_stats() const {
        TiedStats stats = {pairs_.size(), 0, 0, 0.0};
        for (const auto& p : pairs_) {
            if (p.active) stats.active_pairs++;
            else stats.failed_pairs++;
            if (p.max_force > stats.max_force) stats.max_force = p.max_force;
        }
        return stats;
    }

    void print_summary() const {
        auto s = get_stats();
        std::cout << "Tied Contact [" << config_.id << "]: "
                  << s.active_pairs << "/" << s.total_pairs << " active, "
                  << s.failed_pairs << " failed, max_force=" << s.max_force << "\n";
    }

private:
    TiedContactConfig config_;
    std::vector<TiedPair> pairs_;
};

} // namespace fem
} // namespace nxs
