#pragma once

/**
 * @file force_exchange_wave45.hpp
 * @brief Wave 45b: Production Force Assembly + Ghost Exchange
 *
 * C++ equivalent of OpenRadioss IAD_ELEM/FR_ELEM frontier exchange.
 *
 * Classes:
 *   - FrontierPattern:          Per-neighbor send/recv frontier with pre-allocated buffers
 *   - ForceExchanger:           Non-blocking force accumulate and field scatter
 *   - DistributedTimeStep:      Global minimum dt via MPI_Allreduce
 *   - DistributedEnergyMonitor: Global energy summation via MPI_Allreduce
 *
 * All classes have serial fallbacks when NEXUSSIM_HAVE_MPI is not defined.
 */

#include <nexussim/core/types.hpp>
#include <nexussim/discretization/mesh_partition.hpp>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <cassert>

#ifdef NEXUSSIM_HAVE_MPI
#include <mpi.h>
#endif

namespace nxs {
namespace parallel {

// ============================================================================
// FrontierPattern — Per-neighbor send/recv frontier (IAD_ELEM/FR_ELEM analog)
// ============================================================================

/**
 * @brief Per-neighbor frontier communication pattern with pre-allocated buffers.
 *
 * In OpenRadioss, IAD_ELEM / FR_ELEM describe which element-adjacent nodes
 * must exchange forces across partition boundaries.  FrontierPattern is the
 * C++ equivalent: it stores, per neighbor rank, the local indices of nodes
 * whose force contributions must be sent (ghost -> owner accumulate) or
 * received (owner -> ghost scatter), together with pre-allocated double
 * buffers sized for `dofs_per_node` DOFs.
 */
struct FrontierPattern {
    /// Per-neighbor frontier data
    struct NeighborFrontier {
        int neighbor_rank = -1;

        /// Local node indices whose forces this rank must SEND to neighbor
        /// (these are ghost nodes on this rank whose owner is neighbor)
        std::vector<Index> send_frontier;

        /// Local node indices whose forces this rank will RECEIVE from neighbor
        /// (these are owned nodes on this rank that are ghosts on neighbor)
        std::vector<Index> recv_frontier;

        /// Pre-allocated contiguous send buffer (size = send_frontier.size() * dofs)
        std::vector<Real> send_buffer;

        /// Pre-allocated contiguous recv buffer (size = recv_frontier.size() * dofs)
        std::vector<Real> recv_buffer;

        /// Number of DOFs per node used for buffer sizing
        int dofs_per_node = 3;

        /// Allocate buffers based on frontier sizes and dofs
        void allocate_buffers(int dofs) {
            dofs_per_node = dofs;
            send_buffer.resize(send_frontier.size() * static_cast<std::size_t>(dofs), 0.0);
            recv_buffer.resize(recv_frontier.size() * static_cast<std::size_t>(dofs), 0.0);
        }

        /// Number of send values
        std::size_t send_count() const {
            return send_frontier.size() * static_cast<std::size_t>(dofs_per_node);
        }

        /// Number of recv values
        std::size_t recv_count() const {
            return recv_frontier.size() * static_cast<std::size_t>(dofs_per_node);
        }
    };

    /// All neighbor frontiers
    std::vector<NeighborFrontier> neighbors;

    /// Total DOFs per node
    int dofs_per_node = 3;

    /// This rank
    int owner_rank = 0;

    /// Number of neighbors
    std::size_t num_neighbors() const { return neighbors.size(); }

    /// Total send frontier nodes (sum over all neighbors)
    std::size_t total_send_nodes() const {
        std::size_t total = 0;
        for (const auto& n : neighbors) total += n.send_frontier.size();
        return total;
    }

    /// Total recv frontier nodes (sum over all neighbors)
    std::size_t total_recv_nodes() const {
        std::size_t total = 0;
        for (const auto& n : neighbors) total += n.recv_frontier.size();
        return total;
    }

    /**
     * @brief Build frontier pattern from a MeshPartition's CommPattern.
     *
     * For force accumulate (ghost -> owner):
     *   send_frontier = CommPattern::recv_nodes (our ghosts, send their forces to owner)
     *   recv_frontier = CommPattern::send_nodes (our shared nodes, receive ghost forces)
     *
     * For field scatter (owner -> ghost):
     *   Uses the same pattern but in the opposite direction (handled by
     *   ForceExchanger's scatter methods).
     *
     * @param partition  The mesh partition with comm_patterns
     * @param dofs       DOFs per node (default 3 for 3D forces)
     */
    void build(const discretization::MeshPartition& partition, int dofs = 3) {
        dofs_per_node = dofs;
        owner_rank = partition.owner_rank;
        neighbors.clear();
        neighbors.reserve(partition.comm_patterns.size());

        for (const auto& cp : partition.comm_patterns) {
            NeighborFrontier nf;
            nf.neighbor_rank = cp.neighbor_rank;

            // For accumulate: we SEND our ghost node contributions back to the
            // owner rank.  Our ghost nodes are the recv_nodes in CommPattern
            // (we receive data into those during a forward scatter).
            nf.send_frontier = cp.recv_nodes;

            // For accumulate: we RECEIVE ghost contributions from the neighbor
            // into our shared (owned) nodes.  Our shared nodes are the
            // send_nodes in CommPattern (we send data from those during a
            // forward scatter).
            nf.recv_frontier = cp.send_nodes;

            // Sort for deterministic ordering
            std::sort(nf.send_frontier.begin(), nf.send_frontier.end());
            std::sort(nf.recv_frontier.begin(), nf.recv_frontier.end());

            nf.allocate_buffers(dofs);
            neighbors.push_back(std::move(nf));
        }
    }

    /**
     * @brief Reset all buffers to zero
     */
    void clear_buffers() {
        for (auto& n : neighbors) {
            std::fill(n.send_buffer.begin(), n.send_buffer.end(), 0.0);
            std::fill(n.recv_buffer.begin(), n.recv_buffer.end(), 0.0);
        }
    }
};

// ============================================================================
// ForceExchanger — Production non-blocking force exchange
// ============================================================================

/**
 * @brief Non-blocking force accumulation and field scatter across MPI ranks.
 *
 * Two main operations:
 *   1. Accumulate: ghost nodes contribute partial forces back to owner nodes
 *      (begin_accumulate / finish_accumulate)
 *   2. Scatter: owner nodes broadcast updated fields to ghost copies
 *      (begin_scatter / finish_scatter)
 *
 * Uses MPI_Isend / MPI_Irecv for overlap with computation.
 * Serial fallback: all methods are no-ops.
 */
class ForceExchanger {
public:
    ForceExchanger() = default;

    /**
     * @brief Initialize from a mesh partition
     * @param partition  Mesh partition with comm_patterns
     * @param dofs_per_node  DOFs per node (3 for forces, 6 for forces+moments)
     */
    void setup(const discretization::MeshPartition& partition, int dofs_per_node = 3) {
        frontier_.build(partition, dofs_per_node);
        partition_ = &partition;
        dofs_ = dofs_per_node;
        initialized_ = true;

#ifdef NEXUSSIM_HAVE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
#else
        rank_ = 0;
        size_ = 1;
#endif
    }

    /**
     * @brief Begin non-blocking force accumulation (ghost -> owner).
     *
     * 1. Posts MPI_Irecv for incoming ghost contributions on each neighbor.
     * 2. Packs ghost node forces into contiguous send buffers.
     * 3. Posts MPI_Isend to send ghost contributions to owner ranks.
     *
     * After this call, the ghost entries in `forces` are cleared to zero
     * (their contributions have been packed into send buffers).
     *
     * @param forces  Force vector, size = (num_local + num_ghost) * dofs_per_node
     */
    void begin_accumulate(std::vector<Real>& forces) {
        if (!initialized_) return;

#ifdef NEXUSSIM_HAVE_MPI
        if (size_ <= 1) return;

        frontier_.clear_buffers();
        requests_.clear();

        const int tag_accum = 100;

        for (auto& nf : frontier_.neighbors) {
            // Post Irecv: receive ghost contributions from neighbor into our
            // recv buffer (will be accumulated into our owned nodes)
            if (!nf.recv_frontier.empty()) {
                MPI_Request req;
                MPI_Irecv(nf.recv_buffer.data(),
                          static_cast<int>(nf.recv_count()),
                          MPI_DOUBLE, nf.neighbor_rank, tag_accum,
                          MPI_COMM_WORLD, &req);
                requests_.push_back(req);
            }

            // Pack ghost node forces into send buffer
            for (std::size_t j = 0; j < nf.send_frontier.size(); ++j) {
                Index local = nf.send_frontier[j];
                for (int d = 0; d < dofs_; ++d) {
                    nf.send_buffer[j * dofs_ + d] = forces[local * dofs_ + d];
                    forces[local * dofs_ + d] = 0.0;  // Clear ghost contribution
                }
            }

            // Post Isend: send our ghost contributions to the owner
            if (!nf.send_frontier.empty()) {
                MPI_Request req;
                MPI_Isend(nf.send_buffer.data(),
                          static_cast<int>(nf.send_count()),
                          MPI_DOUBLE, nf.neighbor_rank, tag_accum,
                          MPI_COMM_WORLD, &req);
                requests_.push_back(req);
            }
        }
#else
        (void)forces;
#endif
    }

    /**
     * @brief Finish force accumulation — wait for all comms, unpack and add.
     *
     * Waits for all non-blocking operations posted by begin_accumulate(),
     * then accumulates received ghost contributions into the owned node forces.
     *
     * @param forces  Force vector (same one passed to begin_accumulate)
     */
    void finish_accumulate(std::vector<Real>& forces) {
        if (!initialized_) return;

#ifdef NEXUSSIM_HAVE_MPI
        if (size_ <= 1) return;
        if (requests_.empty()) return;

        // Wait for all sends and receives
        MPI_Waitall(static_cast<int>(requests_.size()),
                     requests_.data(), MPI_STATUSES_IGNORE);
        requests_.clear();

        // Unpack: accumulate received contributions into owned nodes
        for (const auto& nf : frontier_.neighbors) {
            for (std::size_t j = 0; j < nf.recv_frontier.size(); ++j) {
                Index local = nf.recv_frontier[j];
                for (int d = 0; d < dofs_; ++d) {
                    forces[local * dofs_ + d] += nf.recv_buffer[j * dofs_ + d];
                }
            }
        }
#else
        (void)forces;
#endif
    }

    /**
     * @brief Begin non-blocking forward scatter (owner -> ghost).
     *
     * Sends owned node values to neighbor ranks that hold ghost copies.
     * Used for accelerations, velocities, positions after the time integration step.
     *
     * For scatter, the direction is reversed compared to accumulate:
     *   - We SEND from recv_frontier (our owned shared nodes)
     *   - We RECEIVE into send_frontier (our ghost nodes)
     *
     * @param field  Field vector, size = (num_local + num_ghost) * dofs_per_node
     */
    void begin_scatter(std::vector<Real>& field) {
        if (!initialized_) return;

#ifdef NEXUSSIM_HAVE_MPI
        if (size_ <= 1) return;

        frontier_.clear_buffers();
        requests_.clear();

        const int tag_scatter = 200;

        for (auto& nf : frontier_.neighbors) {
            // Post Irecv: receive into our ghost nodes (send_frontier in
            // accumulate direction = ghost nodes on this rank)
            if (!nf.send_frontier.empty()) {
                // Reuse send_buffer as the recv target for scatter
                MPI_Request req;
                MPI_Irecv(nf.send_buffer.data(),
                          static_cast<int>(nf.send_count()),
                          MPI_DOUBLE, nf.neighbor_rank, tag_scatter,
                          MPI_COMM_WORLD, &req);
                requests_.push_back(req);
            }

            // Pack owned shared node values into recv_buffer (used as send
            // buffer for scatter direction)
            for (std::size_t j = 0; j < nf.recv_frontier.size(); ++j) {
                Index local = nf.recv_frontier[j];
                for (int d = 0; d < dofs_; ++d) {
                    nf.recv_buffer[j * dofs_ + d] = field[local * dofs_ + d];
                }
            }

            // Post Isend: send our owned values to the neighbor's ghosts
            if (!nf.recv_frontier.empty()) {
                MPI_Request req;
                MPI_Isend(nf.recv_buffer.data(),
                          static_cast<int>(nf.recv_count()),
                          MPI_DOUBLE, nf.neighbor_rank, tag_scatter,
                          MPI_COMM_WORLD, &req);
                requests_.push_back(req);
            }
        }
#else
        (void)field;
#endif
    }

    /**
     * @brief Finish forward scatter — wait and overwrite ghost values.
     *
     * @param field  Field vector (same one passed to begin_scatter)
     */
    void finish_scatter(std::vector<Real>& field) {
        if (!initialized_) return;

#ifdef NEXUSSIM_HAVE_MPI
        if (size_ <= 1) return;
        if (requests_.empty()) return;

        MPI_Waitall(static_cast<int>(requests_.size()),
                     requests_.data(), MPI_STATUSES_IGNORE);
        requests_.clear();

        // Unpack: overwrite ghost node values
        for (const auto& nf : frontier_.neighbors) {
            for (std::size_t j = 0; j < nf.send_frontier.size(); ++j) {
                Index local = nf.send_frontier[j];
                for (int d = 0; d < dofs_; ++d) {
                    field[local * dofs_ + d] = nf.send_buffer[j * dofs_ + d];
                }
            }
        }
#else
        (void)field;
#endif
    }

    /// Access the underlying frontier pattern (read-only)
    const FrontierPattern& frontier() const { return frontier_; }

    /// Check if setup has been called
    bool is_initialized() const { return initialized_; }

    /// Get DOFs per node
    int dofs_per_node() const { return dofs_; }

private:
    FrontierPattern frontier_;
    const discretization::MeshPartition* partition_ = nullptr;
    int dofs_ = 3;
    int rank_ = 0;
    int size_ = 1;
    bool initialized_ = false;

#ifdef NEXUSSIM_HAVE_MPI
    std::vector<MPI_Request> requests_;
#endif
};

// ============================================================================
// DistributedTimeStep — Global minimum time step
// ============================================================================

/**
 * @brief Computes the global minimum stable time step across all MPI ranks.
 *
 * In explicit dynamics, each element proposes a local stable dt.  The global
 * simulation dt is the minimum across all elements on all ranks.
 *
 * Serial fallback: returns local_dt unchanged.
 */
class DistributedTimeStep {
public:
    DistributedTimeStep() = default;

    /**
     * @brief Compute global minimum time step
     * @param local_dt  Local minimum time step on this rank
     * @return Global minimum time step across all ranks
     */
    Real compute_global_dt(Real local_dt) const {
        Real global_dt = local_dt;

#ifdef NEXUSSIM_HAVE_MPI
        MPI_Allreduce(&local_dt, &global_dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

        return global_dt;
    }

    /**
     * @brief Compute global minimum time step with a safety factor
     * @param local_dt  Local minimum time step on this rank
     * @param safety    Safety factor (typically 0.9)
     * @return safety * global_min_dt
     */
    Real compute_global_dt(Real local_dt, Real safety) const {
        return safety * compute_global_dt(local_dt);
    }

    /**
     * @brief Compute and report which rank controls the time step
     * @param local_dt  Local minimum time step
     * @param rank      This rank's MPI rank
     * @return Global minimum time step
     */
    Real compute_global_dt_verbose(Real local_dt, int rank) const {
        Real global_dt = local_dt;

#ifdef NEXUSSIM_HAVE_MPI
        MPI_Allreduce(&local_dt, &global_dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        // Find which rank has the controlling dt
        int controlling_rank = -1;
        int is_mine = (std::abs(local_dt - global_dt) < 1e-30) ? rank : -1;
        MPI_Allreduce(&is_mine, &controlling_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "  Global dt = " << global_dt
                      << " (controlled by rank " << controlling_rank << ")\n";
        }
#else
        if (rank == 0) {
            std::cout << "  Global dt = " << global_dt << " (serial)\n";
        }
#endif

        return global_dt;
    }
};

// ============================================================================
// DistributedEnergyMonitor — Global energy reduction
// ============================================================================

/**
 * @brief Reduces local energy contributions to global totals across MPI ranks.
 *
 * Tracks kinetic energy (KE), internal energy (IE), and external work (EE).
 * Uses MPI_Allreduce(MPI_SUM) so all ranks have the global totals.
 *
 * Serial fallback: uses local values directly.
 */
class DistributedEnergyMonitor {
public:
    DistributedEnergyMonitor() = default;

    /// Global energy totals (available on all ranks after reduce)
    struct GlobalEnergies {
        Real kinetic  = 0.0;   ///< Global kinetic energy
        Real internal_ = 0.0;  ///< Global internal energy
        Real external_ = 0.0;  ///< Global external work
        Real total() const { return kinetic + internal_ + external_; }
    };

    /**
     * @brief Reduce local energies to global totals
     * @param local_ke  Local kinetic energy on this rank
     * @param local_ie  Local internal energy on this rank
     * @param local_ee  Local external work on this rank
     * @return GlobalEnergies with summed values across all ranks
     */
    GlobalEnergies reduce_energies(Real local_ke, Real local_ie, Real local_ee) const {
        GlobalEnergies ge;

#ifdef NEXUSSIM_HAVE_MPI
        Real local_vals[3] = {local_ke, local_ie, local_ee};
        Real global_vals[3] = {0.0, 0.0, 0.0};
        MPI_Allreduce(local_vals, global_vals, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        ge.kinetic   = global_vals[0];
        ge.internal_ = global_vals[1];
        ge.external_ = global_vals[2];
#else
        ge.kinetic   = local_ke;
        ge.internal_ = local_ie;
        ge.external_ = local_ee;
#endif

        last_energies_ = ge;
        return ge;
    }

    /**
     * @brief Print energy summary (only on rank 0)
     * @param rank  This rank's MPI rank
     */
    void report(int rank) const {
        if (rank != 0) return;
        std::cout << "  Energy Summary:\n";
        std::cout << "    Kinetic  = " << last_energies_.kinetic << "\n";
        std::cout << "    Internal = " << last_energies_.internal_ << "\n";
        std::cout << "    External = " << last_energies_.external_ << "\n";
        std::cout << "    Total    = " << last_energies_.total() << "\n";
    }

    /// Get last reduced energies (cached)
    const GlobalEnergies& last() const { return last_energies_; }

private:
    mutable GlobalEnergies last_energies_;
};

// ============================================================================
// Convenience: combined accumulate + scatter in one call
// ============================================================================

/**
 * @brief Perform a complete force exchange cycle:
 *        accumulate ghost forces, then scatter updated accelerations.
 *
 * This is the typical pattern in an explicit time step:
 *   1. Compute element forces (including ghost-element contributions)
 *   2. Accumulate ghost forces back to owners
 *   3. Compute accelerations a = F/m on owned nodes
 *   4. Scatter accelerations to ghost copies
 */
inline void full_force_exchange_cycle(
    ForceExchanger& exchanger,
    std::vector<Real>& forces,
    std::vector<Real>& accelerations)
{
    // Phase 1: accumulate ghost -> owner forces
    exchanger.begin_accumulate(forces);
    exchanger.finish_accumulate(forces);

    // Phase 2: scatter owner -> ghost accelerations
    exchanger.begin_scatter(accelerations);
    exchanger.finish_scatter(accelerations);
}

} // namespace parallel
} // namespace nxs
