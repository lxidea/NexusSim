#pragma once

/**
 * @file mesh_partition.hpp
 * @brief Parallel mesh partitioning for MPI domain decomposition
 *
 * Features:
 * - Recursive Coordinate Bisection (RCB) partitioning
 * - Element-based and node-based partitioning
 * - Ghost layer generation for parallel communication
 * - Load balancing metrics
 * - Partition quality analysis
 *
 * Designed for explicit FEM where element-based decomposition is preferred.
 */

#include <nexussim/core/types.hpp>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <unordered_set>
#include <unordered_map>

#ifdef NEXUSSIM_HAVE_MPI
#include <mpi.h>
#endif

namespace nxs {
namespace discretization {

// ============================================================================
// Partition Data Structures
// ============================================================================

/**
 * @brief Partition assignment for mesh entities
 */
struct MeshPartition {
    // Element ownership
    std::vector<int> element_owner;      // element_owner[elem] = rank
    std::vector<Index> local_elements;   // Elements owned by this rank

    // Node ownership and ghosts
    std::vector<int> node_owner;         // node_owner[node] = rank
    std::vector<Index> local_nodes;      // Nodes owned by this rank
    std::vector<Index> ghost_nodes;      // Ghost nodes needed from other ranks
    std::vector<Index> shared_nodes;     // Local nodes needed by other ranks

    // Mapping global <-> local
    std::unordered_map<Index, Index> global_to_local_elem;
    std::unordered_map<Index, Index> global_to_local_node;
    std::vector<Index> local_to_global_elem;
    std::vector<Index> local_to_global_node;

    // Communication patterns
    struct CommPattern {
        int neighbor_rank;
        std::vector<Index> send_nodes;   // Local node indices to send
        std::vector<Index> recv_nodes;   // Local ghost indices to receive into
    };
    std::vector<CommPattern> comm_patterns;

    // Statistics
    Index num_local_elements() const { return local_elements.size(); }
    Index num_local_nodes() const { return local_nodes.size(); }
    Index num_ghost_nodes() const { return ghost_nodes.size(); }
    int owner_rank;
};

/**
 * @brief Quality metrics for partition evaluation
 */
struct PartitionQuality {
    int num_partitions;
    Index total_elements;
    Index total_nodes;

    // Load balance
    Index min_elements;
    Index max_elements;
    Real element_imbalance;   // max/avg ratio

    // Communication
    Index total_interface_nodes;
    Index max_interface_nodes;
    Real communication_ratio;  // interface/total nodes

    void print(std::ostream& os = std::cout) const {
        os << "Partition Quality Metrics:\n";
        os << "  Partitions: " << num_partitions << "\n";
        os << "  Total elements: " << total_elements << "\n";
        os << "  Total nodes: " << total_nodes << "\n";
        os << "  Elements per partition: " << min_elements << " - " << max_elements << "\n";
        os << "  Element imbalance: " << element_imbalance << "x\n";
        os << "  Interface nodes: " << total_interface_nodes
           << " (" << (100.0 * communication_ratio) << "%)\n";
    }
};

// ============================================================================
// Recursive Coordinate Bisection (RCB) Partitioner
// ============================================================================

class RCBPartitioner {
public:
    /**
     * @brief Configuration for RCB partitioning
     */
    struct Config {
        bool balance_elements;     // Balance by element count (vs nodes)
        Real tolerance;            // Load imbalance tolerance
        bool use_centroids;        // Use element centroids for coordinates
        int max_recursion;         // Maximum recursion depth

        Config()
            : balance_elements(true)
            , tolerance(0.05)
            , use_centroids(true)
            , max_recursion(30)
        {}
    };

    RCBPartitioner(const Config& config = Config()) : config_(config) {}

    /**
     * @brief Partition mesh elements using RCB
     * @param coords Node coordinates (3 × num_nodes)
     * @param connectivity Element connectivity (nodes_per_elem × num_elements)
     * @param num_nodes Number of nodes
     * @param num_elements Number of elements
     * @param nodes_per_elem Nodes per element
     * @param num_parts Number of partitions
     * @return Element partition assignment
     */
    std::vector<int> partition_elements(
        const Real* coords,
        const Index* connectivity,
        Index num_nodes,
        Index num_elements,
        int nodes_per_elem,
        int num_parts)
    {
        std::vector<int> partition(num_elements, 0);

        if (num_parts <= 1) {
            return partition;
        }

        // Compute element centroids
        std::vector<Vec3r> centroids(num_elements);
        for (Index e = 0; e < num_elements; ++e) {
            centroids[e] = {0, 0, 0};
            for (int n = 0; n < nodes_per_elem; ++n) {
                Index node = connectivity[e * nodes_per_elem + n];
                for (int d = 0; d < 3; ++d) {
                    centroids[e][d] += coords[node * 3 + d];
                }
            }
            for (int d = 0; d < 3; ++d) {
                centroids[e][d] /= nodes_per_elem;
            }
        }

        // Initial element list
        std::vector<Index> elements(num_elements);
        std::iota(elements.begin(), elements.end(), 0);

        // Recursive bisection
        partition_recursive(centroids, elements, partition, 0, num_parts);

        return partition;
    }

    /**
     * @brief Generate full partition data including ghosts
     */
    MeshPartition create_partition(
        const Real* coords,
        const Index* connectivity,
        Index num_nodes,
        Index num_elements,
        int nodes_per_elem,
        int num_parts,
        int my_rank)
    {
        MeshPartition part;
        part.owner_rank = my_rank;

        // Get element partition
        auto element_partition = partition_elements(
            coords, connectivity, num_nodes, num_elements,
            nodes_per_elem, num_parts);

        part.element_owner = element_partition;

        // Find local elements
        for (Index e = 0; e < num_elements; ++e) {
            if (element_partition[e] == my_rank) {
                part.local_elements.push_back(e);
            }
        }

        // Determine node ownership based on element usage count per partition
        // For interface nodes (used by multiple partitions with equal counts),
        // distribute ownership evenly using node index
        part.node_owner.assign(num_nodes, -1);
        std::vector<std::vector<int>> node_elem_count(num_nodes, std::vector<int>(num_parts, 0));

        for (Index e = 0; e < num_elements; ++e) {
            int elem_rank = element_partition[e];
            for (int n = 0; n < nodes_per_elem; ++n) {
                Index node = connectivity[e * nodes_per_elem + n];
                node_elem_count[node][elem_rank]++;
            }
        }

        for (Index node = 0; node < num_nodes; ++node) {
            // Find all ranks with maximum element count
            int max_count = 0;
            for (int r = 0; r < num_parts; ++r) {
                max_count = std::max(max_count, node_elem_count[node][r]);
            }

            std::vector<int> candidate_ranks;
            for (int r = 0; r < num_parts; ++r) {
                if (node_elem_count[node][r] == max_count) {
                    candidate_ranks.push_back(r);
                }
            }

            if (candidate_ranks.size() == 1) {
                // Clear winner
                part.node_owner[node] = candidate_ranks[0];
            } else {
                // Tie: distribute ownership based on node index for load balance
                int owner_idx = static_cast<int>(node % candidate_ranks.size());
                part.node_owner[node] = candidate_ranks[owner_idx];
            }
        }

        // Find local and ghost nodes
        std::unordered_set<Index> needed_nodes;
        for (Index e : part.local_elements) {
            for (int n = 0; n < nodes_per_elem; ++n) {
                Index node = connectivity[e * nodes_per_elem + n];
                needed_nodes.insert(node);
            }
        }

        for (Index node : needed_nodes) {
            if (part.node_owner[node] == my_rank) {
                part.local_nodes.push_back(node);
            } else {
                part.ghost_nodes.push_back(node);
            }
        }

        std::sort(part.local_nodes.begin(), part.local_nodes.end());
        std::sort(part.ghost_nodes.begin(), part.ghost_nodes.end());

        // Find shared nodes (local nodes needed by neighbors)
        std::unordered_set<Index> shared_set;
        for (Index e = 0; e < num_elements; ++e) {
            if (element_partition[e] != my_rank) {
                for (int n = 0; n < nodes_per_elem; ++n) {
                    Index node = connectivity[e * nodes_per_elem + n];
                    if (part.node_owner[node] == my_rank) {
                        shared_set.insert(node);
                    }
                }
            }
        }
        part.shared_nodes.assign(shared_set.begin(), shared_set.end());
        std::sort(part.shared_nodes.begin(), part.shared_nodes.end());

        // Build global-to-local mappings
        Index local_idx = 0;
        for (Index elem : part.local_elements) {
            part.global_to_local_elem[elem] = local_idx;
            part.local_to_global_elem.push_back(elem);
            local_idx++;
        }

        local_idx = 0;
        for (Index node : part.local_nodes) {
            part.global_to_local_node[node] = local_idx;
            part.local_to_global_node.push_back(node);
            local_idx++;
        }
        for (Index node : part.ghost_nodes) {
            part.global_to_local_node[node] = local_idx;
            part.local_to_global_node.push_back(node);
            local_idx++;
        }

        // Build communication patterns
        build_comm_patterns(part, num_nodes, num_parts, connectivity,
                            num_elements, nodes_per_elem, element_partition);

        return part;
    }

    /**
     * @brief Analyze partition quality
     */
    PartitionQuality analyze_quality(
        const std::vector<int>& element_partition,
        const Index* connectivity,
        Index num_elements,
        Index num_nodes,
        int nodes_per_elem,
        int num_parts)
    {
        PartitionQuality quality;
        quality.num_partitions = num_parts;
        quality.total_elements = num_elements;
        quality.total_nodes = num_nodes;

        // Count elements per partition
        std::vector<Index> elem_counts(num_parts, 0);
        for (Index e = 0; e < num_elements; ++e) {
            elem_counts[element_partition[e]]++;
        }

        quality.min_elements = *std::min_element(elem_counts.begin(), elem_counts.end());
        quality.max_elements = *std::max_element(elem_counts.begin(), elem_counts.end());
        Real avg_elements = static_cast<Real>(num_elements) / num_parts;
        quality.element_imbalance = static_cast<Real>(quality.max_elements) / avg_elements;

        // Count interface nodes
        std::vector<std::unordered_set<int>> node_partitions(num_nodes);
        for (Index e = 0; e < num_elements; ++e) {
            int part = element_partition[e];
            for (int n = 0; n < nodes_per_elem; ++n) {
                Index node = connectivity[e * nodes_per_elem + n];
                node_partitions[node].insert(part);
            }
        }

        quality.total_interface_nodes = 0;
        std::vector<Index> interface_per_part(num_parts, 0);
        for (Index n = 0; n < num_nodes; ++n) {
            if (node_partitions[n].size() > 1) {
                quality.total_interface_nodes++;
                for (int p : node_partitions[n]) {
                    interface_per_part[p]++;
                }
            }
        }
        quality.max_interface_nodes = *std::max_element(
            interface_per_part.begin(), interface_per_part.end());
        quality.communication_ratio = static_cast<Real>(quality.total_interface_nodes) / num_nodes;

        return quality;
    }

private:
    Config config_;

    void partition_recursive(
        const std::vector<Vec3r>& centroids,
        std::vector<Index>& elements,
        std::vector<int>& partition,
        int start_rank,
        int num_parts)
    {
        if (num_parts <= 1 || elements.size() <= 1) {
            for (Index e : elements) {
                partition[e] = start_rank;
            }
            return;
        }

        // Find longest dimension
        Vec3r bbox_min = {1e30, 1e30, 1e30};
        Vec3r bbox_max = {-1e30, -1e30, -1e30};
        for (Index e : elements) {
            for (int d = 0; d < 3; ++d) {
                bbox_min[d] = std::min(bbox_min[d], centroids[e][d]);
                bbox_max[d] = std::max(bbox_max[d], centroids[e][d]);
            }
        }

        int split_dim = 0;
        Real max_extent = 0;
        for (int d = 0; d < 3; ++d) {
            Real extent = bbox_max[d] - bbox_min[d];
            if (extent > max_extent) {
                max_extent = extent;
                split_dim = d;
            }
        }

        // Sort elements by coordinate in split dimension
        std::sort(elements.begin(), elements.end(),
            [&](Index a, Index b) {
                return centroids[a][split_dim] < centroids[b][split_dim];
            });

        // Split into two groups
        int parts_left = num_parts / 2;
        int parts_right = num_parts - parts_left;
        Real ratio = static_cast<Real>(parts_left) / num_parts;

        std::size_t split_idx = static_cast<std::size_t>(elements.size() * ratio);
        split_idx = std::max(std::size_t(1), std::min(split_idx, elements.size() - 1));

        std::vector<Index> left(elements.begin(), elements.begin() + split_idx);
        std::vector<Index> right(elements.begin() + split_idx, elements.end());

        // Recurse
        partition_recursive(centroids, left, partition, start_rank, parts_left);
        partition_recursive(centroids, right, partition, start_rank + parts_left, parts_right);
    }

    void build_comm_patterns(
        MeshPartition& part,
        Index num_nodes,
        int num_parts,
        const Index* connectivity,
        Index num_elements,
        int nodes_per_elem,
        const std::vector<int>& element_partition)
    {
        // Find which ranks need which ghost nodes
        std::unordered_map<int, std::unordered_set<Index>> send_to_rank;
        std::unordered_map<int, std::unordered_set<Index>> recv_from_rank;

        for (Index node : part.shared_nodes) {
            // Find all ranks that need this node
            std::unordered_set<int> needing_ranks;
            for (Index e = 0; e < num_elements; ++e) {
                int elem_rank = element_partition[e];
                if (elem_rank == part.owner_rank) continue;

                for (int n = 0; n < nodes_per_elem; ++n) {
                    if (connectivity[e * nodes_per_elem + n] == node) {
                        needing_ranks.insert(elem_rank);
                        break;
                    }
                }
            }

            for (int rank : needing_ranks) {
                send_to_rank[rank].insert(node);
            }
        }

        for (Index node : part.ghost_nodes) {
            int owner = part.node_owner[node];
            recv_from_rank[owner].insert(node);
        }

        // Create comm patterns
        std::unordered_set<int> neighbor_ranks;
        for (const auto& kv : send_to_rank) {
            neighbor_ranks.insert(kv.first);
        }
        for (const auto& kv : recv_from_rank) {
            neighbor_ranks.insert(kv.first);
        }

        for (int rank : neighbor_ranks) {
            MeshPartition::CommPattern pattern;
            pattern.neighbor_rank = rank;

            // Convert global to local indices
            if (send_to_rank.count(rank)) {
                for (Index global_node : send_to_rank[rank]) {
                    Index local = part.global_to_local_node[global_node];
                    pattern.send_nodes.push_back(local);
                }
                std::sort(pattern.send_nodes.begin(), pattern.send_nodes.end());
            }

            if (recv_from_rank.count(rank)) {
                for (Index global_node : recv_from_rank[rank]) {
                    Index local = part.global_to_local_node[global_node];
                    pattern.recv_nodes.push_back(local);
                }
                std::sort(pattern.recv_nodes.begin(), pattern.recv_nodes.end());
            }

            part.comm_patterns.push_back(pattern);
        }
    }
};

// ============================================================================
// Parallel Communication (MPI)
// ============================================================================

#ifdef NEXUSSIM_HAVE_MPI

/**
 * @brief Exchange ghost node data between MPI ranks
 */
class GhostExchange {
public:
    /**
     * @brief Initialize exchange with partition info
     */
    void initialize(const MeshPartition& partition, MPI_Comm comm = MPI_COMM_WORLD) {
        partition_ = &partition;
        comm_ = comm;
        MPI_Comm_rank(comm, &rank_);
        MPI_Comm_size(comm, &size_);
    }

    /**
     * @brief Exchange scalar field (e.g., pressure)
     */
    void exchange_scalar(std::vector<Real>& field) const {
        // Non-blocking sends and receives
        std::vector<MPI_Request> requests;
        std::vector<std::vector<Real>> send_buffers(partition_->comm_patterns.size());
        std::vector<std::vector<Real>> recv_buffers(partition_->comm_patterns.size());

        for (std::size_t i = 0; i < partition_->comm_patterns.size(); ++i) {
            const auto& pattern = partition_->comm_patterns[i];

            // Pack send buffer
            send_buffers[i].resize(pattern.send_nodes.size());
            for (std::size_t j = 0; j < pattern.send_nodes.size(); ++j) {
                send_buffers[i][j] = field[pattern.send_nodes[j]];
            }

            // Allocate receive buffer
            recv_buffers[i].resize(pattern.recv_nodes.size());

            // Post non-blocking operations
            MPI_Request req;

            if (!pattern.recv_nodes.empty()) {
                MPI_Irecv(recv_buffers[i].data(), pattern.recv_nodes.size(),
                          MPI_DOUBLE, pattern.neighbor_rank, 0, comm_, &req);
                requests.push_back(req);
            }

            if (!pattern.send_nodes.empty()) {
                MPI_Isend(send_buffers[i].data(), pattern.send_nodes.size(),
                          MPI_DOUBLE, pattern.neighbor_rank, 0, comm_, &req);
                requests.push_back(req);
            }
        }

        // Wait for all communications
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        // Unpack receive buffers
        for (std::size_t i = 0; i < partition_->comm_patterns.size(); ++i) {
            const auto& pattern = partition_->comm_patterns[i];
            for (std::size_t j = 0; j < pattern.recv_nodes.size(); ++j) {
                field[pattern.recv_nodes[j]] = recv_buffers[i][j];
            }
        }
    }

    /**
     * @brief Exchange vector field (e.g., displacement)
     */
    void exchange_vector(std::vector<Real>& field, int components = 3) const {
        std::vector<MPI_Request> requests;
        std::vector<std::vector<Real>> send_buffers(partition_->comm_patterns.size());
        std::vector<std::vector<Real>> recv_buffers(partition_->comm_patterns.size());

        for (std::size_t i = 0; i < partition_->comm_patterns.size(); ++i) {
            const auto& pattern = partition_->comm_patterns[i];

            // Pack send buffer
            send_buffers[i].resize(pattern.send_nodes.size() * components);
            for (std::size_t j = 0; j < pattern.send_nodes.size(); ++j) {
                Index local = pattern.send_nodes[j];
                for (int c = 0; c < components; ++c) {
                    send_buffers[i][j * components + c] = field[local * components + c];
                }
            }

            // Allocate receive buffer
            recv_buffers[i].resize(pattern.recv_nodes.size() * components);

            // Post non-blocking operations
            MPI_Request req;

            if (!pattern.recv_nodes.empty()) {
                MPI_Irecv(recv_buffers[i].data(), pattern.recv_nodes.size() * components,
                          MPI_DOUBLE, pattern.neighbor_rank, 0, comm_, &req);
                requests.push_back(req);
            }

            if (!pattern.send_nodes.empty()) {
                MPI_Isend(send_buffers[i].data(), pattern.send_nodes.size() * components,
                          MPI_DOUBLE, pattern.neighbor_rank, 0, comm_, &req);
                requests.push_back(req);
            }
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        // Unpack receive buffers
        for (std::size_t i = 0; i < partition_->comm_patterns.size(); ++i) {
            const auto& pattern = partition_->comm_patterns[i];
            for (std::size_t j = 0; j < pattern.recv_nodes.size(); ++j) {
                Index local = pattern.recv_nodes[j];
                for (int c = 0; c < components; ++c) {
                    field[local * components + c] = recv_buffers[i][j * components + c];
                }
            }
        }
    }

    /**
     * @brief Accumulate contributions from ghost nodes (e.g., for internal forces)
     */
    void accumulate_vector(std::vector<Real>& field, int components = 3) const {
        // Similar to exchange but accumulates instead of overwrites
        std::vector<MPI_Request> requests;
        std::vector<std::vector<Real>> send_buffers(partition_->comm_patterns.size());
        std::vector<std::vector<Real>> recv_buffers(partition_->comm_patterns.size());

        // Send ghost contributions back to owners
        for (std::size_t i = 0; i < partition_->comm_patterns.size(); ++i) {
            const auto& pattern = partition_->comm_patterns[i];

            // Send contributions from our ghosts to owner
            send_buffers[i].resize(pattern.recv_nodes.size() * components);
            for (std::size_t j = 0; j < pattern.recv_nodes.size(); ++j) {
                Index local = pattern.recv_nodes[j];
                for (int c = 0; c < components; ++c) {
                    send_buffers[i][j * components + c] = field[local * components + c];
                    field[local * components + c] = 0.0;  // Clear ghost after send
                }
            }

            // Receive contributions to our shared nodes
            recv_buffers[i].resize(pattern.send_nodes.size() * components);

            MPI_Request req;

            if (!pattern.send_nodes.empty()) {
                MPI_Irecv(recv_buffers[i].data(), pattern.send_nodes.size() * components,
                          MPI_DOUBLE, pattern.neighbor_rank, 1, comm_, &req);
                requests.push_back(req);
            }

            if (!pattern.recv_nodes.empty()) {
                MPI_Isend(send_buffers[i].data(), pattern.recv_nodes.size() * components,
                          MPI_DOUBLE, pattern.neighbor_rank, 1, comm_, &req);
                requests.push_back(req);
            }
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        // Accumulate received contributions
        for (std::size_t i = 0; i < partition_->comm_patterns.size(); ++i) {
            const auto& pattern = partition_->comm_patterns[i];
            for (std::size_t j = 0; j < pattern.send_nodes.size(); ++j) {
                Index local = pattern.send_nodes[j];
                for (int c = 0; c < components; ++c) {
                    field[local * components + c] += recv_buffers[i][j * components + c];
                }
            }
        }
    }

private:
    const MeshPartition* partition_;
    MPI_Comm comm_;
    int rank_, size_;
};

#else // !NEXUSSIM_HAVE_MPI

// Stub implementation when MPI not available
class GhostExchange {
public:
    void initialize(const MeshPartition&, int = 0) {}
    void exchange_scalar(std::vector<Real>&) const {}
    void exchange_vector(std::vector<Real>&, int = 3) const {}
    void accumulate_vector(std::vector<Real>&, int = 3) const {}
};

#endif // NEXUSSIM_HAVE_MPI

} // namespace discretization
} // namespace nxs
