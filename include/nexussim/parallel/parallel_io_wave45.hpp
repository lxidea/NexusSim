#pragma once

/**
 * @file parallel_io_wave45.hpp
 * @brief Wave 45d: Parallel I/O + Load Rebalancing
 *
 * Provides rank-0 centralized output gathering, parallel animation/TH writers,
 * migration execution, and dynamic repartitioning. All MPI code is behind
 * NEXUSSIM_HAVE_MPI guards with serial fallbacks.
 *
 * Classes:
 *   - OutputGatherer:        MPI_Gatherv-based data centralization to root
 *   - ParallelAnimWriter:    Gather + write animation frames (rank 0)
 *   - ParallelTHWriter:      Gather + write time-history data (rank 0)
 *   - MigrationExecutor:     Pack/send/receive element data for load migration
 *   - DynamicRepartitioner:  Trigger repartition when imbalance exceeds threshold
 */

#include <vector>
#include <array>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <map>
#include <unordered_map>
#include <cassert>
#include <iostream>
#include <functional>

#ifdef NEXUSSIM_HAVE_MPI
#include <mpi.h>
#endif

#include <nexussim/parallel/mpi_wave17.hpp>
#include <nexussim/discretization/mesh_partition.hpp>

namespace nxs {
namespace parallel {

using Real = double;
using Index = std::size_t;

// ============================================================================
// OutputGatherer — Rank-0 centralization via MPI_Gatherv
// ============================================================================

/**
 * @brief Gathers distributed per-node/element/scalar data to root rank.
 *
 * Each rank contributes a local subset of data with associated global IDs.
 * Root assembles these into a single contiguous array indexed by global ID.
 * Serial fallback: returns input data directly (wrapped into global-sized array).
 */
class OutputGatherer {
public:
    OutputGatherer() = default;

    /**
     * @brief Gather per-node data from all ranks to root.
     *
     * @param local_data     Local node values (size = local_count * dofs_per_node)
     * @param dofs_per_node  Number of DOFs (values) per node
     * @param global_ids     Global node IDs for local nodes (size = local_count)
     * @param num_global_nodes Total number of global nodes
     * @param root           Root rank that receives assembled data
     * @return On root: global array of size num_global_nodes * dofs_per_node;
     *         on other ranks: empty vector
     */
    std::vector<Real> gather_node_data(
        const std::vector<Real>& local_data,
        int dofs_per_node,
        const std::vector<Index>& global_ids,
        Index num_global_nodes,
        int root = 0) const
    {
        Index local_count = global_ids.size();

#ifdef NEXUSSIM_HAVE_MPI
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Each rank sends: local_count (int), then global_ids, then local_data
        int local_n = static_cast<int>(local_count);

        // Gather counts from all ranks
        std::vector<int> all_counts(size, 0);
        MPI_Gather(&local_n, 1, MPI_INT,
                   all_counts.data(), 1, MPI_INT,
                   root, MPI_COMM_WORLD);

        // Compute displacements for data and IDs
        std::vector<int> data_counts(size, 0);
        std::vector<int> data_displs(size, 0);
        std::vector<int> id_displs(size, 0);

        if (rank == root) {
            for (int r = 0; r < size; ++r) {
                data_counts[r] = all_counts[r] * dofs_per_node;
            }
            data_displs[0] = 0;
            id_displs[0] = 0;
            for (int r = 1; r < size; ++r) {
                data_displs[r] = data_displs[r - 1] + data_counts[r - 1];
                id_displs[r] = id_displs[r - 1] + all_counts[r - 1];
            }
        }

        // Gather global IDs
        int total_ids = 0;
        if (rank == root) {
            for (int r = 0; r < size; ++r) total_ids += all_counts[r];
        }

        // Convert Index to unsigned long for MPI
        std::vector<unsigned long> local_ids_ul(local_count);
        for (Index i = 0; i < local_count; ++i) {
            local_ids_ul[i] = static_cast<unsigned long>(global_ids[i]);
        }

        std::vector<unsigned long> all_ids(total_ids);
        MPI_Gatherv(local_ids_ul.data(), local_n, MPI_UNSIGNED_LONG,
                    all_ids.data(), all_counts.data(), id_displs.data(),
                    MPI_UNSIGNED_LONG, root, MPI_COMM_WORLD);

        // Gather data
        int local_data_count = local_n * dofs_per_node;
        int total_data = 0;
        if (rank == root) {
            for (int r = 0; r < size; ++r) total_data += data_counts[r];
        }

        std::vector<Real> all_data(total_data);
        MPI_Gatherv(local_data.data(), local_data_count, MPI_DOUBLE,
                    all_data.data(), data_counts.data(), data_displs.data(),
                    MPI_DOUBLE, root, MPI_COMM_WORLD);

        // Root assembles by global ID
        if (rank == root) {
            std::vector<Real> global_data(num_global_nodes * dofs_per_node, 0.0);
            int offset = 0;
            for (int r = 0; r < size; ++r) {
                for (int i = 0; i < all_counts[r]; ++i) {
                    Index gid = static_cast<Index>(all_ids[id_displs[r] + i]);
                    if (gid < num_global_nodes) {
                        for (int d = 0; d < dofs_per_node; ++d) {
                            global_data[gid * dofs_per_node + d] =
                                all_data[data_displs[r] + i * dofs_per_node + d];
                        }
                    }
                }
            }
            return global_data;
        }
        return {};

#else
        // Serial fallback: place local data into global-sized array
        (void)root;
        std::vector<Real> global_data(num_global_nodes * dofs_per_node, 0.0);
        for (Index i = 0; i < local_count; ++i) {
            Index gid = global_ids[i];
            if (gid < num_global_nodes) {
                for (int d = 0; d < dofs_per_node; ++d) {
                    global_data[gid * dofs_per_node + d] =
                        local_data[i * dofs_per_node + d];
                }
            }
        }
        return global_data;
#endif
    }

    /**
     * @brief Gather per-element data from all ranks to root.
     *
     * @param local_data       Local element values (size = local_count * vals_per_elem)
     * @param vals_per_elem    Number of values per element
     * @param global_ids       Global element IDs for local elements
     * @param num_global_elements Total number of global elements
     * @param root             Root rank
     * @return On root: global array; on others: empty
     */
    std::vector<Real> gather_element_data(
        const std::vector<Real>& local_data,
        int vals_per_elem,
        const std::vector<Index>& global_ids,
        Index num_global_elements,
        int root = 0) const
    {
        // Element gather uses the same logic as node gather
        return gather_node_data(local_data, vals_per_elem, global_ids,
                                num_global_elements, root);
    }

    /**
     * @brief Gather scalar data (1 value per entity) from all ranks to root.
     *
     * @param local_values  Local scalar values
     * @param global_ids    Global IDs for local entities
     * @param num_global    Total number of global entities
     * @param root          Root rank
     * @return On root: global scalar array; on others: empty
     */
    std::vector<Real> gather_scalar(
        const std::vector<Real>& local_values,
        const std::vector<Index>& global_ids,
        Index num_global,
        int root = 0) const
    {
        return gather_node_data(local_values, 1, global_ids, num_global, root);
    }

    /**
     * @brief Get current MPI rank (0 in serial mode)
     */
    int current_rank() const {
        int rank = 0;
#ifdef NEXUSSIM_HAVE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        return rank;
    }

    /**
     * @brief Get total number of MPI ranks (1 in serial mode)
     */
    int num_ranks() const {
        int size = 1;
#ifdef NEXUSSIM_HAVE_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
        return size;
    }
};

// ============================================================================
// ParallelAnimWriter — Gather + rank-0 animation write
// ============================================================================

/**
 * @brief Parallel animation frame writer.
 *
 * Each rank contributes its local node positions (x,y,z).
 * OutputGatherer assembles the global field on root, which then writes
 * the animation frame. Demonstrates the gather pattern without
 * instantiating RadiossAnimWriter directly.
 */
class ParallelAnimWriter {
public:
    ParallelAnimWriter() = default;

    /**
     * @brief Gathered frame data produced by write_frame().
     */
    struct FrameResult {
        bool written = false;           ///< true if this rank wrote the frame
        std::vector<Real> global_positions; ///< 3*N global positions (root only)
        Real timestep = 0.0;
        Index num_global_nodes = 0;
    };

    /**
     * @brief Write an animation frame.
     *
     * Gathers local_node_positions from all ranks to root, then root
     * "writes" (stores the result). Non-root ranks get an empty result.
     *
     * @param local_node_positions  Local positions (3 * local_node_count)
     * @param local_node_ids        Global IDs of local nodes
     * @param num_global_nodes      Total global node count
     * @param timestep              Current simulation time
     * @param root                  Root rank
     * @return FrameResult with global data on root
     */
    FrameResult write_frame(
        const std::vector<Real>& local_node_positions,
        const std::vector<Index>& local_node_ids,
        Index num_global_nodes,
        Real timestep,
        int root = 0) const
    {
        OutputGatherer gatherer;
        auto global_pos = gatherer.gather_node_data(
            local_node_positions, 3, local_node_ids,
            num_global_nodes, root);

        FrameResult result;
        result.timestep = timestep;
        result.num_global_nodes = num_global_nodes;

        int rank = gatherer.current_rank();
        if (rank == root) {
            result.written = true;
            result.global_positions = std::move(global_pos);
            // In production: RadiossAnimWriter::write_frame(result.global_positions, ...)
        }

        return result;
    }

    /**
     * @brief Write an animation frame using MeshPartition for ID mapping.
     *
     * @param local_node_positions  Local positions (3 * partition.local_nodes.size())
     * @param partition             MeshPartition with local_nodes (global IDs)
     * @param timestep              Current simulation time
     * @param root                  Root rank
     * @return FrameResult
     */
    FrameResult write_frame(
        const std::vector<Real>& local_node_positions,
        const nxs::discretization::MeshPartition& partition,
        Real timestep,
        int root = 0) const
    {
        // Compute total global nodes: max global ID + 1
        Index max_gid = 0;
        for (Index nid : partition.local_nodes) {
            max_gid = std::max(max_gid, nid);
        }
        for (Index nid : partition.ghost_nodes) {
            max_gid = std::max(max_gid, nid);
        }

#ifdef NEXUSSIM_HAVE_MPI
        unsigned long local_max = static_cast<unsigned long>(max_gid);
        unsigned long global_max = local_max;
        MPI_Allreduce(&local_max, &global_max, 1, MPI_UNSIGNED_LONG,
                       MPI_MAX, MPI_COMM_WORLD);
        Index num_global_nodes = static_cast<Index>(global_max) + 1;
#else
        Index num_global_nodes = max_gid + 1;
#endif

        return write_frame(local_node_positions, partition.local_nodes,
                           num_global_nodes, timestep, root);
    }

    /// Number of frames written so far
    Index frames_written() const { return frames_written_; }

private:
    mutable Index frames_written_ = 0;
};

// ============================================================================
// ParallelTHWriter — Gather + rank-0 time-history write
// ============================================================================

/**
 * @brief Parallel time-history writer.
 *
 * Each rank contributes scalar time-history values for its local nodes/elements.
 * OutputGatherer assembles the global field on root for writing.
 */
class ParallelTHWriter {
public:
    ParallelTHWriter() = default;

    /**
     * @brief Gathered TH data produced by write_th().
     */
    struct THResult {
        bool written = false;
        std::vector<Real> global_values;
        Real time = 0.0;
        Index num_global_entities = 0;
    };

    /**
     * @brief Write time-history data.
     *
     * Gathers local_values from all ranks to root, then root stores/writes.
     *
     * @param local_values     Local scalar values (1 per entity)
     * @param local_node_ids   Global IDs of local entities
     * @param num_global       Total number of global entities
     * @param time             Current simulation time
     * @param root             Root rank
     * @return THResult with global data on root
     */
    THResult write_th(
        const std::vector<Real>& local_values,
        const std::vector<Index>& local_node_ids,
        Index num_global,
        Real time,
        int root = 0) const
    {
        OutputGatherer gatherer;
        auto global_vals = gatherer.gather_scalar(
            local_values, local_node_ids, num_global, root);

        THResult result;
        result.time = time;
        result.num_global_entities = num_global;

        int rank = gatherer.current_rank();
        if (rank == root) {
            result.written = true;
            result.global_values = std::move(global_vals);
            // In production: RadiossTHWriter::write(result.global_values, ...)
        }

        return result;
    }

    /**
     * @brief Write time-history using MeshPartition.
     */
    THResult write_th(
        const std::vector<Real>& local_values,
        const nxs::discretization::MeshPartition& partition,
        Real time,
        int root = 0) const
    {
        Index max_gid = 0;
        for (Index nid : partition.local_nodes) {
            max_gid = std::max(max_gid, nid);
        }

#ifdef NEXUSSIM_HAVE_MPI
        unsigned long local_max = static_cast<unsigned long>(max_gid);
        unsigned long global_max = local_max;
        MPI_Allreduce(&local_max, &global_max, 1, MPI_UNSIGNED_LONG,
                       MPI_MAX, MPI_COMM_WORLD);
        Index num_global = static_cast<Index>(global_max) + 1;
#else
        Index num_global = max_gid + 1;
#endif

        return write_th(local_values, partition.local_nodes, num_global,
                        time, root);
    }
};

// ============================================================================
// MigrationExecutor — Pack/send/receive element data for load migration
// ============================================================================

/**
 * @brief Executes element migration plans produced by LoadBalancer.
 *
 * Packs element data on source ranks, sends via non-blocking MPI to
 * destination ranks, receives, and updates local element lists.
 * Serial fallback: shuffles data between local arrays (no-op if single rank).
 */
class MigrationExecutor {
public:
    MigrationExecutor() = default;

    /**
     * @brief Result of migration execution.
     */
    struct MigrationResult {
        bool executed = false;
        Index elements_sent = 0;
        Index elements_received = 0;
        std::vector<std::vector<Real>> new_element_data;   ///< Per-rank element data after migration
        std::vector<std::vector<Index>> new_elements;      ///< Per-rank element IDs after migration
    };

    /**
     * @brief Execute a migration plan, redistributing element data.
     *
     * @param plan            Migration plan from LoadBalancer
     * @param element_data    Per-element data (vals_per_elem values per element),
     *                        indexed by element_id. Map from global element ID to data.
     * @param elements_per_rank  Current element IDs owned by each rank
     * @return MigrationResult with updated element distribution
     */
    MigrationResult execute(
        const MigrationPlan& plan,
        const std::map<Index, std::vector<Real>>& element_data,
        std::vector<std::vector<Index>>& elements_per_rank) const
    {
        MigrationResult result;

        if (!plan.should_migrate || plan.entries.empty()) {
            result.executed = false;
            result.new_elements = elements_per_rank;
            return result;
        }

        result.executed = true;
        int num_ranks = static_cast<int>(elements_per_rank.size());

#ifdef NEXUSSIM_HAVE_MPI
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Count elements to send and receive for this rank
        std::vector<Index> to_send;     // element IDs we send away
        std::vector<Index> to_receive;  // element IDs we receive
        std::map<int, std::vector<Index>> send_to;   // dest_rank -> elem_ids
        std::map<int, std::vector<Index>> recv_from;  // src_rank -> elem_ids

        for (const auto& entry : plan.entries) {
            if (entry.from_rank == rank) {
                to_send.push_back(entry.element_id);
                send_to[entry.to_rank].push_back(entry.element_id);
                result.elements_sent++;
            }
            if (entry.to_rank == rank) {
                to_receive.push_back(entry.element_id);
                recv_from[entry.from_rank].push_back(entry.element_id);
                result.elements_received++;
            }
        }

        // Non-blocking sends: pack element data and send
        std::vector<MPI_Request> requests;
        std::vector<std::vector<Real>> send_buffers;

        for (auto& [dest, eids] : send_to) {
            std::vector<Real> buf;
            for (Index eid : eids) {
                auto it = element_data.find(eid);
                if (it != element_data.end()) {
                    buf.insert(buf.end(), it->second.begin(), it->second.end());
                }
            }
            send_buffers.push_back(std::move(buf));
            MPI_Request req;
            int tag = rank * 1000 + dest;
            MPI_Isend(send_buffers.back().data(),
                      static_cast<int>(send_buffers.back().size()),
                      MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }

        // Non-blocking receives
        std::vector<std::vector<Real>> recv_buffers;
        for (auto& [src, eids] : recv_from) {
            // Probe for message size
            MPI_Status status;
            int tag = src * 1000 + rank;
            MPI_Probe(src, tag, MPI_COMM_WORLD, &status);
            int count;
            MPI_Get_count(&status, MPI_DOUBLE, &count);

            std::vector<Real> buf(count);
            MPI_Request req;
            MPI_Irecv(buf.data(), count, MPI_DOUBLE, src, tag,
                      MPI_COMM_WORLD, &req);
            recv_buffers.push_back(std::move(buf));
            requests.push_back(req);
        }

        // Wait for all
        if (!requests.empty()) {
            std::vector<MPI_Status> statuses(requests.size());
            MPI_Waitall(static_cast<int>(requests.size()),
                        requests.data(), statuses.data());
        }

        MPI_Barrier(MPI_COMM_WORLD);
#else
        // Serial fallback: just shuffle elements between rank arrays
        (void)element_data;
        result.elements_sent = 0;
        result.elements_received = 0;
#endif

        // Update elements_per_rank bookkeeping
        // Build set of elements to remove per rank
        std::map<int, std::set<Index>> remove_from_rank;
        std::map<int, std::vector<Index>> add_to_rank;

        for (const auto& entry : plan.entries) {
            if (entry.from_rank >= 0 && entry.from_rank < num_ranks) {
                remove_from_rank[entry.from_rank].insert(entry.element_id);
            }
            if (entry.to_rank >= 0 && entry.to_rank < num_ranks) {
                add_to_rank[entry.to_rank].push_back(entry.element_id);
            }
#ifndef NEXUSSIM_HAVE_MPI
            result.elements_sent++;
#endif
        }

        result.new_elements.resize(num_ranks);
        for (int r = 0; r < num_ranks; ++r) {
            auto& old_elems = elements_per_rank[r];
            auto& removals = remove_from_rank[r];
            for (Index eid : old_elems) {
                if (removals.find(eid) == removals.end()) {
                    result.new_elements[r].push_back(eid);
                }
            }
            auto& additions = add_to_rank[r];
            result.new_elements[r].insert(result.new_elements[r].end(),
                                           additions.begin(), additions.end());
            std::sort(result.new_elements[r].begin(), result.new_elements[r].end());
        }

        // Update the input
        elements_per_rank = result.new_elements;

        return result;
    }

    /**
     * @brief Rebuild MeshPartition after migration.
     *
     * Re-partitions the mesh using RCBPartitioner based on current
     * element distribution.
     *
     * @param partition      Partition to rebuild (in/out)
     * @param connectivity   Element connectivity (nodes_per_elem * num_elements)
     * @param coords         Node coordinates (3 * num_nodes)
     * @param num_nodes      Total node count
     * @param nodes_per_elem Nodes per element
     * @param num_parts      Number of partitions
     */
    void rebuild_partition(
        nxs::discretization::MeshPartition& partition,
        const std::vector<Index>& connectivity,
        const std::vector<Real>& coords,
        Index num_nodes,
        int nodes_per_elem,
        int num_parts) const
    {
        Index num_elements = connectivity.size() / nodes_per_elem;

#ifdef NEXUSSIM_HAVE_MPI
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
        int rank = partition.owner_rank;
#endif

        nxs::discretization::RCBPartitioner partitioner;
        partition = partitioner.create_partition(
            coords.data(), connectivity.data(),
            num_nodes, num_elements,
            nodes_per_elem, num_parts, rank);
    }
};

// ============================================================================
// DynamicRepartitioner — Trigger repartition on imbalance
// ============================================================================

/**
 * @brief Monitors load imbalance and triggers repartitioning.
 *
 * When the ratio max_weight / avg_weight exceeds the threshold,
 * generates a migration plan via LoadBalancer, executes it, and
 * rebuilds the partition.
 */
class DynamicRepartitioner {
public:
    /**
     * @brief Constructor.
     * @param imbalance_threshold  Ratio threshold (default 1.2 = 20% overload)
     */
    explicit DynamicRepartitioner(Real imbalance_threshold = 1.2)
        : threshold_(imbalance_threshold) {}

    /**
     * @brief Check whether rebalancing is needed.
     *
     * @param rank_weights  Computational weight per rank
     * @return true if max/avg > threshold
     */
    bool check_rebalance(const std::vector<Real>& rank_weights) const {
#ifdef NEXUSSIM_HAVE_MPI
        // In MPI mode, each rank provides its local weight;
        // use Allreduce to compute global max and sum
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        Real local_w = 0.0;
        if (rank < static_cast<int>(rank_weights.size())) {
            local_w = rank_weights[rank];
        }

        Real total_w = 0.0, max_w = 0.0;
        MPI_Allreduce(&local_w, &total_w, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_w, &max_w, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (size <= 1) return false;
        Real avg = total_w / size;
        if (avg < 1e-30) return false;
        return (max_w / avg) > threshold_;
#else
        // Serial fallback: never needs rebalancing
        if (rank_weights.size() <= 1) return false;

        Real total = 0.0;
        Real max_w = 0.0;
        for (Real w : rank_weights) {
            total += w;
            max_w = std::max(max_w, w);
        }
        Real avg = total / static_cast<Real>(rank_weights.size());
        if (avg < 1e-30) return false;
        return (max_w / avg) > threshold_;
#endif
    }

    /**
     * @brief Result of a repartition operation.
     */
    struct RepartitionResult {
        bool repartitioned = false;
        Real imbalance_before = 1.0;
        Real imbalance_after = 1.0;
        Index elements_migrated = 0;
        MigrationPlan plan;
    };

    /**
     * @brief Perform load rebalancing.
     *
     * Uses LoadBalancer to generate a migration plan, then MigrationExecutor
     * to apply it. Updates elements_per_rank in place.
     *
     * @param element_weights    Weight per element (indexed by global elem ID)
     * @param element_ranks      Current rank assignment per element
     * @param elements_per_rank  Current elements per rank (updated in place)
     * @param num_ranks          Number of ranks
     * @return RepartitionResult
     */
    RepartitionResult rebalance(
        const std::map<Index, Real>& element_weights,
        const std::map<Index, int>& element_ranks,
        std::vector<std::vector<Index>>& elements_per_rank,
        int num_ranks) const
    {
        RepartitionResult result;

        // Build element lists for LoadBalancer
        std::vector<Index> elem_ids;
        std::vector<Real> weights;
        std::vector<int> ranks;
        elem_ids.reserve(element_weights.size());
        weights.reserve(element_weights.size());
        ranks.reserve(element_weights.size());

        for (const auto& [eid, w] : element_weights) {
            elem_ids.push_back(eid);
            weights.push_back(w);
            auto it = element_ranks.find(eid);
            ranks.push_back(it != element_ranks.end() ? it->second : 0);
        }

        // Configure load balancer
        LoadBalancer lb;
        lb.initialize(num_ranks, threshold_);
        lb.register_elements(elem_ids, weights, ranks);

        result.imbalance_before = lb.compute_imbalance();

        // Generate migration plan
        result.plan = lb.generate_migration_plan();

        if (!result.plan.should_migrate) {
            result.repartitioned = false;
            result.imbalance_after = result.imbalance_before;
            return result;
        }

        // Execute migration
        MigrationExecutor executor;
        std::map<Index, std::vector<Real>> elem_data;
        for (const auto& [eid, w] : element_weights) {
            elem_data[eid] = {w};  // Pack weight as element data
        }

        executor.execute(result.plan, elem_data, elements_per_rank);

        // Execute migration on the load balancer's bookkeeping too
        lb.execute_migration(result.plan);

        result.repartitioned = true;
        result.imbalance_after = lb.compute_imbalance();
        result.elements_migrated = result.plan.total_migrations();

        return result;
    }

    /**
     * @brief Combined check + rebalance convenience method.
     *
     * @param rank_weights       Computational weight per rank
     * @param element_weights    Weight per element
     * @param element_ranks      Current rank per element
     * @param elements_per_rank  Elements per rank (updated in place)
     * @return RepartitionResult
     */
    RepartitionResult check_and_rebalance(
        const std::vector<Real>& rank_weights,
        const std::map<Index, Real>& element_weights,
        const std::map<Index, int>& element_ranks,
        std::vector<std::vector<Index>>& elements_per_rank) const
    {
        if (!check_rebalance(rank_weights)) {
            RepartitionResult result;
            result.repartitioned = false;
            return result;
        }

        int num_ranks = static_cast<int>(rank_weights.size());
        return rebalance(element_weights, element_ranks,
                         elements_per_rank, num_ranks);
    }

    /// Get threshold
    Real threshold() const { return threshold_; }

    /// Set threshold
    void set_threshold(Real t) { threshold_ = t; }

private:
    Real threshold_ = 1.2;
};

} // namespace parallel
} // namespace nxs
