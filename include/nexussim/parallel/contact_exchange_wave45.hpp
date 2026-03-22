#pragma once

/**
 * @file contact_exchange_wave45.hpp
 * @brief Wave 45c: Parallel Contact Search
 *
 * Provides distributed broad-phase contact detection, node data exchange
 * for narrow-phase evaluation, and parallel bucket sort with halo exchange.
 *
 * All MPI code is behind #ifdef NEXUSSIM_HAVE_MPI guards.
 * Serial fallbacks provide correct single-rank behavior.
 */

#include <nexussim/parallel/mpi_wave17.hpp>
#include <nexussim/parallel/mpi_wave45.hpp>

#include <vector>
#include <array>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <functional>
#include <limits>
#include <cassert>

#ifdef NEXUSSIM_HAVE_MPI
#include <mpi.h>
#endif

namespace nxs {
namespace parallel {

// ============================================================================
// 1. DistributedBroadPhase
// ============================================================================

/**
 * @brief Multi-stage parallel broad-phase contact detection.
 *
 * Workflow:
 *   1. Each rank computes rank-level AABB from local surface elements.
 *   2. MPI_Allgather rank-level AABBs (6 doubles per rank per surface).
 *   3. Test local surface-B vs all remote surface-A AABBs to find overlapping rank pairs.
 *   4. For overlapping pairs, exchange element-level AABBs via Isend/Irecv.
 *   5. Local AABB intersection on received remote boxes produces cross-rank ContactPair vector.
 *
 * Serial fallback performs local-only broad phase (no MPI calls).
 */
class DistributedBroadPhase {
public:
    DistributedBroadPhase() = default;

    /**
     * @brief Initialize with local surface data
     * @param local_a_boxes AABBs for surface A elements on this rank
     * @param local_b_boxes AABBs for surface B elements on this rank
     * @param local_a_ids   Element IDs for surface A elements
     * @param local_b_ids   Element IDs for surface B elements
     * @param rank          This rank's ID
     * @param n_ranks       Total number of ranks
     */
    void initialize(const std::vector<AABB>& local_a_boxes,
                    const std::vector<AABB>& local_b_boxes,
                    const std::vector<Index>& local_a_ids,
                    const std::vector<Index>& local_b_ids,
                    int rank, int n_ranks) {
        local_a_boxes_ = local_a_boxes;
        local_b_boxes_ = local_b_boxes;
        local_a_ids_ = local_a_ids;
        local_b_ids_ = local_b_ids;
        rank_ = rank;
        n_ranks_ = n_ranks;

        // Compute rank-level bounding boxes
        rank_a_box_ = compute_rank_aabb(local_a_boxes_);
        rank_b_box_ = compute_rank_aabb(local_b_boxes_);

        initialized_ = true;
    }

    /**
     * @brief Detect cross-rank contact candidate pairs
     * @return Vector of ContactPair with cross-rank (and local) candidates
     */
    std::vector<ContactPair> detect_cross_rank_pairs() {
        if (!initialized_) return {};

        std::vector<ContactPair> result;

        // Step 1: Gather rank-level AABBs from all ranks
        std::vector<AABB> all_a_rank_boxes(n_ranks_);
        std::vector<AABB> all_b_rank_boxes(n_ranks_);

        gather_rank_aabbs(all_a_rank_boxes, all_b_rank_boxes);

        // Step 2: Determine overlapping rank pairs
        // For each remote rank r, check if our surface-B rank box overlaps
        // their surface-A rank box (and vice versa)
        std::set<int> overlapping_ranks;
        for (int r = 0; r < n_ranks_; ++r) {
            if (r == rank_) continue;
            // Our B vs their A
            if (rank_b_box_.is_valid() && all_a_rank_boxes[r].is_valid() &&
                rank_b_box_.intersects(all_a_rank_boxes[r])) {
                overlapping_ranks.insert(r);
            }
            // Our A vs their B
            if (rank_a_box_.is_valid() && all_b_rank_boxes[r].is_valid() &&
                rank_a_box_.intersects(all_b_rank_boxes[r])) {
                overlapping_ranks.insert(r);
            }
        }

        overlapping_ranks_ = overlapping_ranks;

        // Step 3: Exchange element-level AABBs with overlapping ranks
        std::map<int, std::vector<AABB>> remote_a_boxes;
        std::map<int, std::vector<Index>> remote_a_ids;
        std::map<int, std::vector<AABB>> remote_b_boxes;
        std::map<int, std::vector<Index>> remote_b_ids;

        exchange_element_aabbs(overlapping_ranks,
                               remote_a_boxes, remote_a_ids,
                               remote_b_boxes, remote_b_ids);

        // Step 4: Local broad-phase: our A vs our B
        add_local_pairs(result);

        // Step 5: Cross-rank pairs
        // Our B elements vs remote A elements
        for (auto& [remote_rank, r_a_boxes] : remote_a_boxes) {
            const auto& r_a_ids_vec = remote_a_ids[remote_rank];
            for (size_t ib = 0; ib < local_b_boxes_.size(); ++ib) {
                for (size_t ia = 0; ia < r_a_boxes.size(); ++ia) {
                    if (local_b_boxes_[ib].intersects(r_a_boxes[ia])) {
                        ContactPair cp;
                        cp.surface_a_id = r_a_ids_vec[ia];
                        cp.surface_b_id = local_b_ids_[ib];
                        cp.rank_a = remote_rank;
                        cp.rank_b = rank_;
                        cp.gap = estimate_gap(r_a_boxes[ia], local_b_boxes_[ib]);
                        auto mid = midpoint(r_a_boxes[ia], local_b_boxes_[ib]);
                        cp.contact_point[0] = mid[0];
                        cp.contact_point[1] = mid[1];
                        cp.contact_point[2] = mid[2];
                        result.push_back(cp);
                    }
                }
            }
        }

        // Our A elements vs remote B elements
        for (auto& [remote_rank, r_b_boxes] : remote_b_boxes) {
            const auto& r_b_ids_vec = remote_b_ids[remote_rank];
            for (size_t ia = 0; ia < local_a_boxes_.size(); ++ia) {
                for (size_t ib = 0; ib < r_b_boxes.size(); ++ib) {
                    if (local_a_boxes_[ia].intersects(r_b_boxes[ib])) {
                        ContactPair cp;
                        cp.surface_a_id = local_a_ids_[ia];
                        cp.surface_b_id = r_b_ids_vec[ib];
                        cp.rank_a = rank_;
                        cp.rank_b = remote_rank;
                        cp.gap = estimate_gap(local_a_boxes_[ia], r_b_boxes[ib]);
                        auto mid = midpoint(local_a_boxes_[ia], r_b_boxes[ib]);
                        cp.contact_point[0] = mid[0];
                        cp.contact_point[1] = mid[1];
                        cp.contact_point[2] = mid[2];
                        result.push_back(cp);
                    }
                }
            }
        }

        return result;
    }

    /// Get the set of overlapping ranks determined during last detection
    const std::set<int>& overlapping_ranks() const { return overlapping_ranks_; }

    /// Get the rank-level AABB for surface A
    const AABB& rank_a_box() const { return rank_a_box_; }

    /// Get the rank-level AABB for surface B
    const AABB& rank_b_box() const { return rank_b_box_; }

private:
    std::vector<AABB> local_a_boxes_;
    std::vector<AABB> local_b_boxes_;
    std::vector<Index> local_a_ids_;
    std::vector<Index> local_b_ids_;
    int rank_ = 0;
    int n_ranks_ = 1;
    bool initialized_ = false;

    AABB rank_a_box_;
    AABB rank_b_box_;
    std::set<int> overlapping_ranks_;

    /// Compute the union AABB of a set of boxes
    static AABB compute_rank_aabb(const std::vector<AABB>& boxes) {
        AABB result;
        for (const auto& box : boxes) {
            if (box.is_valid()) {
                result.expand(box);
            }
        }
        return result;
    }

    /// Gather rank-level AABBs from all ranks via Allgather
    void gather_rank_aabbs(std::vector<AABB>& all_a, std::vector<AABB>& all_b) {
#ifdef NEXUSSIM_HAVE_MPI
        // Pack local rank boxes: 6 doubles each (min_pt[3], max_pt[3])
        std::vector<double> send_a(6), send_b(6);
        for (int d = 0; d < 3; ++d) {
            send_a[d]     = rank_a_box_.min_pt[d];
            send_a[d + 3] = rank_a_box_.max_pt[d];
            send_b[d]     = rank_b_box_.min_pt[d];
            send_b[d + 3] = rank_b_box_.max_pt[d];
        }

        std::vector<double> recv_a(6 * n_ranks_), recv_b(6 * n_ranks_);
        MPI_Allgather(send_a.data(), 6, MPI_DOUBLE,
                      recv_a.data(), 6, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(send_b.data(), 6, MPI_DOUBLE,
                      recv_b.data(), 6, MPI_DOUBLE, MPI_COMM_WORLD);

        for (int r = 0; r < n_ranks_; ++r) {
            for (int d = 0; d < 3; ++d) {
                all_a[r].min_pt[d] = recv_a[r * 6 + d];
                all_a[r].max_pt[d] = recv_a[r * 6 + d + 3];
                all_b[r].min_pt[d] = recv_b[r * 6 + d];
                all_b[r].max_pt[d] = recv_b[r * 6 + d + 3];
            }
        }
#else
        // Serial: only one rank
        all_a[0] = rank_a_box_;
        all_b[0] = rank_b_box_;
#endif
    }

    /// Exchange element-level AABBs with overlapping ranks
    void exchange_element_aabbs(
        const std::set<int>& overlapping,
        std::map<int, std::vector<AABB>>& remote_a_boxes,
        std::map<int, std::vector<Index>>& remote_a_ids,
        std::map<int, std::vector<AABB>>& remote_b_boxes,
        std::map<int, std::vector<Index>>& remote_b_ids)
    {
#ifdef NEXUSSIM_HAVE_MPI
        // Phase 1: Exchange counts
        // For each overlapping rank, send our element counts, receive theirs
        std::vector<MPI_Request> count_reqs;

        // We send our A count and B count to each overlapping rank
        struct CountMsg {
            int a_count;
            int b_count;
        };

        std::map<int, CountMsg> send_counts;
        std::map<int, CountMsg> recv_counts;

        for (int r : overlapping) {
            send_counts[r] = {static_cast<int>(local_a_boxes_.size()),
                              static_cast<int>(local_b_boxes_.size())};
            recv_counts[r] = {0, 0};
        }

        // Non-blocking sends/receives for counts
        std::vector<MPI_Request> reqs;
        for (int r : overlapping) {
            MPI_Request req;
            MPI_Isend(&send_counts[r], 2, MPI_INT, r, 100, MPI_COMM_WORLD, &req);
            reqs.push_back(req);
            MPI_Irecv(&recv_counts[r], 2, MPI_INT, r, 100, MPI_COMM_WORLD, &req);
            reqs.push_back(req);
        }

        if (!reqs.empty()) {
            MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        }
        reqs.clear();

        // Phase 2: Exchange element-level data
        // Pack AABBs as flat arrays of 6 doubles each, IDs as size_t array
        std::map<int, std::vector<double>> send_a_data, send_b_data;
        std::map<int, std::vector<size_t>> send_a_id_data, send_b_id_data;

        for (int r : overlapping) {
            // Pack A boxes
            send_a_data[r].resize(local_a_boxes_.size() * 6);
            send_a_id_data[r] = local_a_ids_;
            for (size_t i = 0; i < local_a_boxes_.size(); ++i) {
                for (int d = 0; d < 3; ++d) {
                    send_a_data[r][i * 6 + d]     = local_a_boxes_[i].min_pt[d];
                    send_a_data[r][i * 6 + d + 3] = local_a_boxes_[i].max_pt[d];
                }
            }
            // Pack B boxes
            send_b_data[r].resize(local_b_boxes_.size() * 6);
            send_b_id_data[r] = local_b_ids_;
            for (size_t i = 0; i < local_b_boxes_.size(); ++i) {
                for (int d = 0; d < 3; ++d) {
                    send_b_data[r][i * 6 + d]     = local_b_boxes_[i].min_pt[d];
                    send_b_data[r][i * 6 + d + 3] = local_b_boxes_[i].max_pt[d];
                }
            }
        }

        // Allocate receive buffers
        std::map<int, std::vector<double>> recv_a_data, recv_b_data;
        std::map<int, std::vector<size_t>> recv_a_id_data, recv_b_id_data;

        for (int r : overlapping) {
            recv_a_data[r].resize(recv_counts[r].a_count * 6);
            recv_a_id_data[r].resize(recv_counts[r].a_count);
            recv_b_data[r].resize(recv_counts[r].b_count * 6);
            recv_b_id_data[r].resize(recv_counts[r].b_count);
        }

        // Non-blocking exchange of A boxes, B boxes, A ids, B ids
        for (int r : overlapping) {
            MPI_Request req;
            int na = static_cast<int>(local_a_boxes_.size());
            int nb = static_cast<int>(local_b_boxes_.size());
            int rna = recv_counts[r].a_count;
            int rnb = recv_counts[r].b_count;

            if (na > 0) {
                MPI_Isend(send_a_data[r].data(), na * 6, MPI_DOUBLE, r, 200,
                          MPI_COMM_WORLD, &req);
                reqs.push_back(req);
                MPI_Isend(send_a_id_data[r].data(), na,
                          mpi_type<size_t>(), r, 201,
                          MPI_COMM_WORLD, &req);
                reqs.push_back(req);
            }
            if (rna > 0) {
                MPI_Irecv(recv_a_data[r].data(), rna * 6, MPI_DOUBLE, r, 200,
                          MPI_COMM_WORLD, &req);
                reqs.push_back(req);
                MPI_Irecv(recv_a_id_data[r].data(), rna,
                          mpi_type<size_t>(), r, 201,
                          MPI_COMM_WORLD, &req);
                reqs.push_back(req);
            }
            if (nb > 0) {
                MPI_Isend(send_b_data[r].data(), nb * 6, MPI_DOUBLE, r, 300,
                          MPI_COMM_WORLD, &req);
                reqs.push_back(req);
                MPI_Isend(send_b_id_data[r].data(), nb,
                          mpi_type<size_t>(), r, 301,
                          MPI_COMM_WORLD, &req);
                reqs.push_back(req);
            }
            if (rnb > 0) {
                MPI_Irecv(recv_b_data[r].data(), rnb * 6, MPI_DOUBLE, r, 300,
                          MPI_COMM_WORLD, &req);
                reqs.push_back(req);
                MPI_Irecv(recv_b_id_data[r].data(), rnb,
                          mpi_type<size_t>(), r, 301,
                          MPI_COMM_WORLD, &req);
                reqs.push_back(req);
            }
        }

        if (!reqs.empty()) {
            MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        }

        // Unpack received data
        for (int r : overlapping) {
            int rna = recv_counts[r].a_count;
            int rnb = recv_counts[r].b_count;

            remote_a_boxes[r].resize(rna);
            remote_a_ids[r] = recv_a_id_data[r];
            for (int i = 0; i < rna; ++i) {
                for (int d = 0; d < 3; ++d) {
                    remote_a_boxes[r][i].min_pt[d] = recv_a_data[r][i * 6 + d];
                    remote_a_boxes[r][i].max_pt[d] = recv_a_data[r][i * 6 + d + 3];
                }
            }

            remote_b_boxes[r].resize(rnb);
            remote_b_ids[r] = recv_b_id_data[r];
            for (int i = 0; i < rnb; ++i) {
                for (int d = 0; d < 3; ++d) {
                    remote_b_boxes[r][i].min_pt[d] = recv_b_data[r][i * 6 + d];
                    remote_b_boxes[r][i].max_pt[d] = recv_b_data[r][i * 6 + d + 3];
                }
            }
        }
#else
        // Serial: no remote data to exchange
        (void)overlapping;
        (void)remote_a_boxes;
        (void)remote_a_ids;
        (void)remote_b_boxes;
        (void)remote_b_ids;
#endif
    }

    /// Add local A-vs-B pairs (same rank)
    void add_local_pairs(std::vector<ContactPair>& result) {
        for (size_t ia = 0; ia < local_a_boxes_.size(); ++ia) {
            for (size_t ib = 0; ib < local_b_boxes_.size(); ++ib) {
                if (local_a_boxes_[ia].intersects(local_b_boxes_[ib])) {
                    ContactPair cp;
                    cp.surface_a_id = local_a_ids_[ia];
                    cp.surface_b_id = local_b_ids_[ib];
                    cp.rank_a = rank_;
                    cp.rank_b = rank_;
                    cp.gap = estimate_gap(local_a_boxes_[ia], local_b_boxes_[ib]);
                    auto mid = midpoint(local_a_boxes_[ia], local_b_boxes_[ib]);
                    cp.contact_point[0] = mid[0];
                    cp.contact_point[1] = mid[1];
                    cp.contact_point[2] = mid[2];
                    result.push_back(cp);
                }
            }
        }
    }

    /// Estimate gap between two overlapping AABBs (negative = overlap)
    static Real estimate_gap(const AABB& a, const AABB& b) {
        Real max_gap = -std::numeric_limits<Real>::max();
        for (int d = 0; d < 3; ++d) {
            Real gap_d = std::max(a.min_pt[d] - b.max_pt[d],
                                  b.min_pt[d] - a.max_pt[d]);
            max_gap = std::max(max_gap, gap_d);
        }
        return max_gap;
    }

    /// Midpoint between centers of two AABBs
    static std::array<Real, 3> midpoint(const AABB& a, const AABB& b) {
        std::array<Real, 3> mid;
        Real ca[3], cb[3];
        ca[0] = 0.5 * (a.min_pt[0] + a.max_pt[0]);
        ca[1] = 0.5 * (a.min_pt[1] + a.max_pt[1]);
        ca[2] = 0.5 * (a.min_pt[2] + a.max_pt[2]);
        cb[0] = 0.5 * (b.min_pt[0] + b.max_pt[0]);
        cb[1] = 0.5 * (b.min_pt[1] + b.max_pt[1]);
        cb[2] = 0.5 * (b.min_pt[2] + b.max_pt[2]);
        mid[0] = 0.5 * (ca[0] + cb[0]);
        mid[1] = 0.5 * (ca[1] + cb[1]);
        mid[2] = 0.5 * (ca[2] + cb[2]);
        return mid;
    }
};

// ============================================================================
// 2. ContactDataExchanger
// ============================================================================

/**
 * @brief Exchange node positions and velocities for narrow-phase contact.
 *
 * After broad phase identifies cross-rank contact candidate pairs,
 * this class exchanges the actual node data needed for narrow-phase
 * evaluation (positions, velocities, node IDs).
 *
 * Serial fallback: no-op (all data is local).
 */
class ContactDataExchanger {
public:
    /// Per-node data packet for narrow-phase
    struct NodeData {
        Index node_id = 0;
        Real position[3] = {0.0, 0.0, 0.0};
        Real velocity[3] = {0.0, 0.0, 0.0};
    };

    ContactDataExchanger() = default;

    /**
     * @brief Configure which rank pairs need data exchange
     * @param rank_pairs Set of remote ranks that overlap with this rank
     * @param my_rank    This rank's ID
     */
    void setup(const std::set<int>& rank_pairs, int my_rank) {
        remote_ranks_ = rank_pairs;
        my_rank_ = my_rank;
        remote_data_.clear();
        configured_ = true;
    }

    /**
     * @brief Exchange node data with configured remote ranks
     * @param positions  Local node positions (3*n_nodes)
     * @param velocities Local node velocities (3*n_nodes)
     * @param node_ids   Local node IDs
     */
    void exchange_node_data(const std::vector<Real>& positions,
                            const std::vector<Real>& velocities,
                            const std::vector<Index>& node_ids) {
        if (!configured_) return;

        remote_data_.clear();

#ifdef NEXUSSIM_HAVE_MPI
        size_t n_local = node_ids.size();
        std::vector<MPI_Request> reqs;

        // Phase 1: Exchange node counts
        int local_count = static_cast<int>(n_local);
        std::map<int, int> remote_counts;
        for (int r : remote_ranks_) {
            remote_counts[r] = 0;
        }

        for (int r : remote_ranks_) {
            MPI_Request req;
            MPI_Isend(&local_count, 1, MPI_INT, r, 400, MPI_COMM_WORLD, &req);
            reqs.push_back(req);
            MPI_Irecv(&remote_counts[r], 1, MPI_INT, r, 400, MPI_COMM_WORLD, &req);
            reqs.push_back(req);
        }
        if (!reqs.empty()) {
            MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        }
        reqs.clear();

        // Phase 2: Exchange actual node data (positions, velocities, ids)
        // Pack local data into flat arrays
        // positions already in flat [x0,y0,z0, x1,y1,z1, ...] format
        std::map<int, std::vector<Real>> recv_pos, recv_vel;
        std::map<int, std::vector<size_t>> recv_ids;

        for (int r : remote_ranks_) {
            int rc = remote_counts[r];
            recv_pos[r].resize(rc * 3);
            recv_vel[r].resize(rc * 3);
            recv_ids[r].resize(rc);
        }

        for (int r : remote_ranks_) {
            MPI_Request req;
            int rc = remote_counts[r];

            // Send positions
            if (n_local > 0) {
                MPI_Isend(positions.data(), static_cast<int>(n_local * 3),
                          MPI_DOUBLE, r, 410, MPI_COMM_WORLD, &req);
                reqs.push_back(req);
                MPI_Isend(velocities.data(), static_cast<int>(n_local * 3),
                          MPI_DOUBLE, r, 411, MPI_COMM_WORLD, &req);
                reqs.push_back(req);
                MPI_Isend(node_ids.data(), static_cast<int>(n_local),
                          mpi_type<size_t>(), r, 412, MPI_COMM_WORLD, &req);
                reqs.push_back(req);
            }
            if (rc > 0) {
                MPI_Irecv(recv_pos[r].data(), rc * 3,
                          MPI_DOUBLE, r, 410, MPI_COMM_WORLD, &req);
                reqs.push_back(req);
                MPI_Irecv(recv_vel[r].data(), rc * 3,
                          MPI_DOUBLE, r, 411, MPI_COMM_WORLD, &req);
                reqs.push_back(req);
                MPI_Irecv(recv_ids[r].data(), rc,
                          mpi_type<size_t>(), r, 412, MPI_COMM_WORLD, &req);
                reqs.push_back(req);
            }
        }

        if (!reqs.empty()) {
            MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        }

        // Unpack into NodeData structs
        for (int r : remote_ranks_) {
            int rc = remote_counts[r];
            std::vector<NodeData> nodes(rc);
            for (int i = 0; i < rc; ++i) {
                nodes[i].node_id = recv_ids[r][i];
                for (int d = 0; d < 3; ++d) {
                    nodes[i].position[d] = recv_pos[r][i * 3 + d];
                    nodes[i].velocity[d] = recv_vel[r][i * 3 + d];
                }
            }
            remote_data_[r] = std::move(nodes);
        }
#else
        // Serial: no remote data to exchange
        (void)positions;
        (void)velocities;
        (void)node_ids;
#endif
    }

    /**
     * @brief Get received node data from a specific source rank
     * @param from_rank The remote rank to get data from
     * @return Vector of NodeData received from that rank (empty if none)
     */
    const std::vector<NodeData>& get_remote_data(int from_rank) const {
        static const std::vector<NodeData> empty;
        auto it = remote_data_.find(from_rank);
        if (it != remote_data_.end()) {
            return it->second;
        }
        return empty;
    }

    /// Check if data has been received from a specific rank
    bool has_data_from(int from_rank) const {
        return remote_data_.count(from_rank) > 0;
    }

    /// Get all remote ranks from which data was received
    std::set<int> received_from_ranks() const {
        std::set<int> result;
        for (const auto& [r, _] : remote_data_) {
            result.insert(r);
        }
        return result;
    }

private:
    std::set<int> remote_ranks_;
    int my_rank_ = 0;
    bool configured_ = false;
    std::map<int, std::vector<NodeData>> remote_data_;
};

// ============================================================================
// 3. ParallelBucketSort
// ============================================================================

/**
 * @brief Distributed bucket sort for spatial contact search with halo exchange.
 *
 * Extends the bucket sort concept for distributed domains:
 * - Each rank builds a local grid of buckets covering its domain
 * - Boundary cells are exchanged with neighboring ranks (halo)
 * - Queries can search local + received halo cells
 *
 * Serial fallback delegates to a simple local bucket sort.
 */
class ParallelBucketSort {
public:
    /// Entry stored in a bucket: element AABB + ID + owning rank
    struct BucketEntry {
        AABB box;
        Index elem_id = 0;
        int   owner_rank = 0;
    };

    /**
     * @brief Construct with domain bounds and bucket size
     * @param domain      Domain AABB
     * @param bucket_size Edge length of each bucket cell
     * @param rank        This rank's ID
     * @param n_ranks     Total number of ranks
     */
    ParallelBucketSort(const AABB& domain, Real bucket_size,
                       int rank = 0, int n_ranks = 1)
        : domain_(domain), bucket_size_(bucket_size),
          rank_(rank), n_ranks_(n_ranks)
    {
        // Compute grid dimensions
        for (int d = 0; d < 3; ++d) {
            Real extent = domain_.max_pt[d] - domain_.min_pt[d];
            dims_[d] = std::max(1, static_cast<int>(std::ceil(extent / bucket_size_)));
        }
        total_cells_ = static_cast<size_t>(dims_[0]) * dims_[1] * dims_[2];
        cells_.resize(total_cells_);
    }

    /**
     * @brief Insert local elements into the bucket grid
     * @param local_boxes AABBs of local elements
     * @param local_ids   IDs of local elements
     */
    void build(const std::vector<AABB>& local_boxes,
               const std::vector<Index>& local_ids) {
        // Clear existing entries
        for (auto& cell : cells_) {
            cell.clear();
        }
        halo_entries_.clear();

        for (size_t i = 0; i < local_boxes.size(); ++i) {
            insert(local_boxes[i], local_ids[i], rank_);
        }
    }

    /**
     * @brief Exchange boundary cell contents with neighboring ranks
     * @param boundary_depth Number of cell layers to exchange (default 1)
     */
    void exchange_halo(int boundary_depth = 1) {
#ifdef NEXUSSIM_HAVE_MPI
        if (n_ranks_ <= 1) return;

        // Collect boundary entries: elements in cells within boundary_depth
        // of the domain edge
        std::vector<BucketEntry> boundary_entries;
        for (int ix = 0; ix < dims_[0]; ++ix) {
            for (int iy = 0; iy < dims_[1]; ++iy) {
                for (int iz = 0; iz < dims_[2]; ++iz) {
                    bool is_boundary =
                        ix < boundary_depth || ix >= dims_[0] - boundary_depth ||
                        iy < boundary_depth || iy >= dims_[1] - boundary_depth ||
                        iz < boundary_depth || iz >= dims_[2] - boundary_depth;

                    if (is_boundary) {
                        size_t idx = cell_index(ix, iy, iz);
                        for (const auto& entry : cells_[idx]) {
                            boundary_entries.push_back(entry);
                        }
                    }
                }
            }
        }

        // Exchange with all other ranks (could be optimized for neighbor-only)
        // Phase 1: Exchange counts
        int local_count = static_cast<int>(boundary_entries.size());
        std::vector<int> all_counts(n_ranks_);
        MPI_Allgather(&local_count, 1, MPI_INT,
                      all_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // Phase 2: Pack entries as flat doubles: [min_pt(3), max_pt(3), elem_id, owner_rank]
        // = 8 doubles per entry
        std::vector<double> send_buf(boundary_entries.size() * 8);
        for (size_t i = 0; i < boundary_entries.size(); ++i) {
            const auto& e = boundary_entries[i];
            for (int d = 0; d < 3; ++d) {
                send_buf[i * 8 + d]     = e.box.min_pt[d];
                send_buf[i * 8 + d + 3] = e.box.max_pt[d];
            }
            send_buf[i * 8 + 6] = static_cast<double>(e.elem_id);
            send_buf[i * 8 + 7] = static_cast<double>(e.owner_rank);
        }

        // Compute displacements for Allgatherv
        std::vector<int> recv_counts(n_ranks_);
        std::vector<int> displs(n_ranks_, 0);
        int total_recv = 0;
        for (int r = 0; r < n_ranks_; ++r) {
            recv_counts[r] = all_counts[r] * 8;
            displs[r] = total_recv;
            total_recv += recv_counts[r];
        }

        std::vector<double> recv_buf(total_recv);
        MPI_Allgatherv(send_buf.data(), static_cast<int>(send_buf.size()), MPI_DOUBLE,
                       recv_buf.data(), recv_counts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Unpack received entries (skip our own)
        for (int r = 0; r < n_ranks_; ++r) {
            if (r == rank_) continue;
            int offset = displs[r];
            int count = all_counts[r];
            for (int i = 0; i < count; ++i) {
                BucketEntry entry;
                for (int d = 0; d < 3; ++d) {
                    entry.box.min_pt[d] = recv_buf[offset + i * 8 + d];
                    entry.box.max_pt[d] = recv_buf[offset + i * 8 + d + 3];
                }
                entry.elem_id = static_cast<Index>(recv_buf[offset + i * 8 + 6]);
                entry.owner_rank = static_cast<int>(recv_buf[offset + i * 8 + 7]);
                halo_entries_.push_back(entry);

                // Also insert into local grid if within domain
                insert_if_in_domain(entry);
            }
        }
#else
        (void)boundary_depth;
        // Serial: nothing to exchange
#endif
    }

    /**
     * @brief Query local + halo cells for elements overlapping a query box
     * @param query_box  AABB to search for
     * @param callback   Called with each matching BucketEntry
     */
    void query_with_remote(const AABB& query_box,
                           const std::function<void(const BucketEntry&)>& callback) const {
        // Determine cell range covered by query box
        int lo[3], hi[3];
        for (int d = 0; d < 3; ++d) {
            lo[d] = cell_coord(query_box.min_pt[d], d);
            hi[d] = cell_coord(query_box.max_pt[d], d);
            lo[d] = std::max(0, lo[d]);
            hi[d] = std::min(dims_[d] - 1, hi[d]);
        }

        // Search local cells
        for (int ix = lo[0]; ix <= hi[0]; ++ix) {
            for (int iy = lo[1]; iy <= hi[1]; ++iy) {
                for (int iz = lo[2]; iz <= hi[2]; ++iz) {
                    size_t idx = cell_index(ix, iy, iz);
                    for (const auto& entry : cells_[idx]) {
                        if (entry.box.intersects(query_box)) {
                            callback(entry);
                        }
                    }
                }
            }
        }

        // Also search halo entries that may not fit in the grid
        for (const auto& entry : halo_entries_) {
            if (entry.box.intersects(query_box)) {
                // Check we haven't already reported this from grid cells
                // (entries inserted via insert_if_in_domain are in both)
                // To avoid duplicates, only report entries outside our domain
                bool in_domain = true;
                for (int d = 0; d < 3; ++d) {
                    Real center_d = 0.5 * (entry.box.min_pt[d] + entry.box.max_pt[d]);
                    if (center_d < domain_.min_pt[d] || center_d > domain_.max_pt[d]) {
                        in_domain = false;
                        break;
                    }
                }
                if (!in_domain) {
                    callback(entry);
                }
            }
        }
    }

    /// Get number of local entries
    size_t local_entry_count() const {
        size_t count = 0;
        for (const auto& cell : cells_) {
            for (const auto& e : cell) {
                if (e.owner_rank == rank_) count++;
            }
        }
        return count;
    }

    /// Get number of halo entries received
    size_t halo_entry_count() const {
        return halo_entries_.size();
    }

    /// Get grid dimensions
    const int* dims() const { return dims_; }

    /// Get total number of grid cells
    size_t total_cells() const { return total_cells_; }

private:
    AABB domain_;
    Real bucket_size_;
    int rank_;
    int n_ranks_;
    int dims_[3] = {1, 1, 1};
    size_t total_cells_ = 1;

    std::vector<std::vector<BucketEntry>> cells_;
    std::vector<BucketEntry> halo_entries_;  ///< Remote entries outside our grid

    /// Convert coordinate to cell index along dimension d
    int cell_coord(Real val, int d) const {
        Real rel = val - domain_.min_pt[d];
        return static_cast<int>(std::floor(rel / bucket_size_));
    }

    /// Convert 3D cell coordinates to flat index
    size_t cell_index(int ix, int iy, int iz) const {
        ix = std::max(0, std::min(ix, dims_[0] - 1));
        iy = std::max(0, std::min(iy, dims_[1] - 1));
        iz = std::max(0, std::min(iz, dims_[2] - 1));
        return static_cast<size_t>(ix) * dims_[1] * dims_[2] +
               static_cast<size_t>(iy) * dims_[2] +
               static_cast<size_t>(iz);
    }

    /// Insert an element into the grid
    void insert(const AABB& box, Index elem_id, int owner_rank) {
        int lo[3], hi[3];
        for (int d = 0; d < 3; ++d) {
            lo[d] = cell_coord(box.min_pt[d], d);
            hi[d] = cell_coord(box.max_pt[d], d);
            lo[d] = std::max(0, lo[d]);
            hi[d] = std::min(dims_[d] - 1, hi[d]);
        }

        BucketEntry entry;
        entry.box = box;
        entry.elem_id = elem_id;
        entry.owner_rank = owner_rank;

        for (int ix = lo[0]; ix <= hi[0]; ++ix) {
            for (int iy = lo[1]; iy <= hi[1]; ++iy) {
                for (int iz = lo[2]; iz <= hi[2]; ++iz) {
                    size_t idx = cell_index(ix, iy, iz);
                    cells_[idx].push_back(entry);
                }
            }
        }
    }

    /// Insert a remote entry into the grid if its center is within domain
    void insert_if_in_domain(const BucketEntry& entry) {
        Real cx = 0.5 * (entry.box.min_pt[0] + entry.box.max_pt[0]);
        Real cy = 0.5 * (entry.box.min_pt[1] + entry.box.max_pt[1]);
        Real cz = 0.5 * (entry.box.min_pt[2] + entry.box.max_pt[2]);

        if (cx >= domain_.min_pt[0] && cx <= domain_.max_pt[0] &&
            cy >= domain_.min_pt[1] && cy <= domain_.max_pt[1] &&
            cz >= domain_.min_pt[2] && cz <= domain_.max_pt[2]) {
            insert(entry.box, entry.elem_id, entry.owner_rank);
        }
    }
};

} // namespace parallel
} // namespace nxs
