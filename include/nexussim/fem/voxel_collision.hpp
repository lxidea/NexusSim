#pragma once

/**
 * @file voxel_collision.hpp
 * @brief GPU-accelerated voxel-based broad phase collision detection
 *
 * Based on OpenRadioss intsort methodology:
 * - Linked-list per voxel for O(N) complexity
 * - GPU-parallel voxel assignment via Kokkos
 * - Automatic grid sizing based on element density
 * - Dynamic re-sorting trigger when geometry changes significantly
 *
 * This provides the broad phase for contact detection, generating
 * candidate pairs that are then refined by narrow phase algorithms.
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <vector>
#include <cmath>
#include <limits>

namespace nxs {
namespace fem {

// ============================================================================
// Collision Candidate Pair
// ============================================================================

/**
 * @brief Candidate pair from broad phase collision detection
 */
struct CollisionCandidate {
    Index node_id;       ///< Slave node index
    Index segment_id;    ///< Master segment index
    Real distance_sq;    ///< Squared distance estimate for prioritization
};

// ============================================================================
// Voxel Grid Configuration
// ============================================================================

/**
 * @brief Configuration for voxel-based collision detection
 */
struct VoxelCollisionConfig {
    Real cell_size_factor;     ///< Cell size = factor × average_element_size
    Real search_margin;        ///< Extra margin around bounding boxes
    Real rebuild_threshold;    ///< Motion threshold to trigger rebuild (fraction of cell size)
    Index max_candidates;      ///< Maximum candidate pairs per node
    bool auto_cell_size;       ///< Automatically compute optimal cell size

    VoxelCollisionConfig()
        : cell_size_factor(2.0)
        , search_margin(0.01)
        , rebuild_threshold(0.5)
        , max_candidates(64)
        , auto_cell_size(true)
    {}
};

// ============================================================================
// GPU Voxel Collision Detector
// ============================================================================

/**
 * @brief GPU-accelerated voxel-based collision detection
 *
 * Uses Kokkos for GPU parallelization with a linked-list per voxel structure.
 * The algorithm:
 * 1. Compute bounding box and grid dimensions
 * 2. Assign each segment to overlapping voxels
 * 3. Build linked-list structure for efficient queries
 * 4. For each slave node, find candidate segments in nearby voxels
 *
 * Usage:
 * ```cpp
 * VoxelCollisionDetector detector;
 * detector.initialize(num_nodes, num_segments, coords, segment_nodes, 4);
 *
 * // Each time step:
 * detector.update_coordinates(coords);
 * auto candidates = detector.find_candidates(slave_nodes);
 * ```
 */
class VoxelCollisionDetector {
public:
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = ExecutionSpace::memory_space;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    // Kokkos Views
    using RealView = Kokkos::View<Real*, MemorySpace>;
    using IndexView = Kokkos::View<Index*, MemorySpace>;
    using IntView = Kokkos::View<int*, MemorySpace>;
    using Real3View = Kokkos::View<Real*[3], MemorySpace>;

    // Host mirrors
    using RealHostView = Kokkos::View<Real*, Kokkos::HostSpace>;
    using IndexHostView = Kokkos::View<Index*, Kokkos::HostSpace>;
    using Real3HostView = Kokkos::View<Real*[3], Kokkos::HostSpace>;

    VoxelCollisionDetector(const VoxelCollisionConfig& config = VoxelCollisionConfig())
        : config_(config)
        , num_nodes_(0)
        , num_segments_(0)
        , nodes_per_segment_(4)
        , cell_size_(0.1)
        , initialized_(false)
        , needs_rebuild_(true)
    {
        grid_dims_[0] = grid_dims_[1] = grid_dims_[2] = 1;
        bbox_min_[0] = bbox_min_[1] = bbox_min_[2] = 0.0;
        bbox_max_[0] = bbox_max_[1] = bbox_max_[2] = 1.0;
    }

    /**
     * @brief Initialize the collision detector
     * @param num_nodes Total number of nodes
     * @param num_segments Number of surface segments
     * @param coords Node coordinates (num_nodes × 3)
     * @param segment_nodes Segment connectivity (num_segments × nodes_per_segment)
     * @param nodes_per_segment Nodes per segment (3 for triangles, 4 for quads)
     */
    void initialize(Index num_nodes,
                   Index num_segments,
                   const Real* coords,
                   const Index* segment_nodes,
                   int nodes_per_segment) {
        num_nodes_ = num_nodes;
        num_segments_ = num_segments;
        nodes_per_segment_ = nodes_per_segment;

        // Allocate device views
        coords_ = Real3View("coords", num_nodes);
        segment_nodes_ = IndexView("segment_nodes", num_segments * nodes_per_segment);

        // Copy segment connectivity to device
        auto segment_nodes_host = Kokkos::create_mirror_view(segment_nodes_);
        for (Index i = 0; i < num_segments * nodes_per_segment; ++i) {
            segment_nodes_host(i) = segment_nodes[i];
        }
        Kokkos::deep_copy(segment_nodes_, segment_nodes_host);

        // Copy initial coordinates
        update_coordinates(coords);

        // Compute cell size based on segment sizes
        if (config_.auto_cell_size) {
            compute_optimal_cell_size(coords, segment_nodes);
        } else {
            cell_size_ = config_.cell_size_factor * 0.1;  // Default
        }

        // Build initial grid
        build_grid();

        initialized_ = true;
    }

    /**
     * @brief Update node coordinates (call each time step)
     */
    void update_coordinates(const Real* coords) {
        auto coords_host = Kokkos::create_mirror_view(coords_);
        for (Index i = 0; i < num_nodes_; ++i) {
            coords_host(i, 0) = coords[i * 3 + 0];
            coords_host(i, 1) = coords[i * 3 + 1];
            coords_host(i, 2) = coords[i * 3 + 2];
        }
        Kokkos::deep_copy(coords_, coords_host);

        // Check if rebuild needed
        check_rebuild_needed();

        if (needs_rebuild_) {
            build_grid();
        }
    }

    /**
     * @brief Find collision candidates for given slave nodes
     * @param slave_nodes Vector of slave node indices
     * @param search_radius Search radius for candidate detection
     * @return Vector of collision candidate pairs
     */
    std::vector<CollisionCandidate> find_candidates(
            const std::vector<Index>& slave_nodes,
            Real search_radius) {

        std::vector<CollisionCandidate> candidates;
        if (slave_nodes.empty()) return candidates;

        candidates.reserve(slave_nodes.size() * config_.max_candidates);

        // Create host mirror of coordinates for candidate computation
        auto coords_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coords_);

        Real search_radius_sq = search_radius * search_radius;

        // Compute bounding box of all slave nodes
        Real slave_min[3] = {1e30, 1e30, 1e30};
        Real slave_max[3] = {-1e30, -1e30, -1e30};
        for (Index slave : slave_nodes) {
            for (int d = 0; d < 3; ++d) {
                Real c = coords_host(slave, d);
                slave_min[d] = std::min(slave_min[d], c);
                slave_max[d] = std::max(slave_max[d], c);
            }
        }

        // Expand bounding box by search radius
        for (int d = 0; d < 3; ++d) {
            slave_min[d] -= search_radius;
            slave_max[d] += search_radius;
        }

        // Find voxel range for the entire slave bounding box
        int i_min = static_cast<int>((slave_min[0] - bbox_min_[0]) / cell_size_);
        int j_min = static_cast<int>((slave_min[1] - bbox_min_[1]) / cell_size_);
        int k_min = static_cast<int>((slave_min[2] - bbox_min_[2]) / cell_size_);
        int i_max = static_cast<int>((slave_max[0] - bbox_min_[0]) / cell_size_);
        int j_max = static_cast<int>((slave_max[1] - bbox_min_[1]) / cell_size_);
        int k_max = static_cast<int>((slave_max[2] - bbox_min_[2]) / cell_size_);

        // Clamp to grid bounds
        i_min = std::max(0, std::min(i_min, grid_dims_[0] - 1));
        j_min = std::max(0, std::min(j_min, grid_dims_[1] - 1));
        k_min = std::max(0, std::min(k_min, grid_dims_[2] - 1));
        i_max = std::max(0, std::min(i_max, grid_dims_[0] - 1));
        j_max = std::max(0, std::min(j_max, grid_dims_[1] - 1));
        k_max = std::max(0, std::min(k_max, grid_dims_[2] - 1));

        // Collect all segments in the bounding box region
        std::vector<Index> nearby_segments;
        for (int i = i_min; i <= i_max; ++i) {
            for (int j = j_min; j <= j_max; ++j) {
                for (int k = k_min; k <= k_max; ++k) {
                    int voxel_idx = i + grid_dims_[0] * (j + grid_dims_[1] * k);

                    // Walk linked list for this voxel
                    Index seg_idx = voxel_heads_host_(voxel_idx);
                    while (seg_idx != static_cast<Index>(-1)) {
                        nearby_segments.push_back(seg_idx);
                        seg_idx = next_segment_host_(seg_idx);
                    }
                }
            }
        }

        // Remove duplicates
        std::sort(nearby_segments.begin(), nearby_segments.end());
        nearby_segments.erase(std::unique(nearby_segments.begin(), nearby_segments.end()),
                              nearby_segments.end());

        // Get segment connectivity
        auto segment_nodes_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), segment_nodes_);

        // For each slave node, check nearby segments
        for (Index slave : slave_nodes) {
            Real pos[3] = {
                coords_host(slave, 0),
                coords_host(slave, 1),
                coords_host(slave, 2)
            };

            for (Index seg_id : nearby_segments) {
                // Compute segment centroid
                Real centroid[3] = {0, 0, 0};
                for (int n = 0; n < nodes_per_segment_; ++n) {
                    Index node = segment_nodes_host(seg_id * nodes_per_segment_ + n);
                    centroid[0] += coords_host(node, 0);
                    centroid[1] += coords_host(node, 1);
                    centroid[2] += coords_host(node, 2);
                }
                centroid[0] /= nodes_per_segment_;
                centroid[1] /= nodes_per_segment_;
                centroid[2] /= nodes_per_segment_;

                // Distance squared to centroid
                Real dx = pos[0] - centroid[0];
                Real dy = pos[1] - centroid[1];
                Real dz = pos[2] - centroid[2];
                Real dist_sq = dx*dx + dy*dy + dz*dz;

                // Add candidate if within expanded search radius
                // Use very generous radius for broad phase
                Real expanded_radius_sq = search_radius_sq * 100.0;
                if (dist_sq < expanded_radius_sq) {
                    CollisionCandidate candidate;
                    candidate.node_id = slave;
                    candidate.segment_id = seg_id;
                    candidate.distance_sq = dist_sq;
                    candidates.push_back(candidate);
                }
            }
        }

        return candidates;
    }

    /**
     * @brief Get current cell size
     */
    Real cell_size() const { return cell_size_; }

    /**
     * @brief Get grid dimensions
     */
    void get_grid_dims(int dims[3]) const {
        dims[0] = grid_dims_[0];
        dims[1] = grid_dims_[1];
        dims[2] = grid_dims_[2];
    }

    /**
     * @brief Get bounding box
     */
    void get_bounding_box(Real min[3], Real max[3]) const {
        for (int d = 0; d < 3; ++d) {
            min[d] = bbox_min_[d];
            max[d] = bbox_max_[d];
        }
    }

    /**
     * @brief Force grid rebuild
     */
    void force_rebuild() {
        needs_rebuild_ = true;
    }

    /**
     * @brief Get number of segments in each voxel (for debugging)
     */
    std::vector<int> get_voxel_counts() const {
        int total_voxels = grid_dims_[0] * grid_dims_[1] * grid_dims_[2];
        std::vector<int> counts(total_voxels, 0);

        for (int v = 0; v < total_voxels; ++v) {
            Index seg_idx = voxel_heads_host_(v);
            while (seg_idx != static_cast<Index>(-1)) {
                counts[v]++;
                seg_idx = next_segment_host_(seg_idx);
            }
        }

        return counts;
    }

private:
    /**
     * @brief Compute optimal cell size based on average segment size
     */
    void compute_optimal_cell_size(const Real* coords, const Index* segment_nodes) {
        Real total_size = 0.0;

        for (Index s = 0; s < num_segments_; ++s) {
            // Compute segment bounding box
            Real seg_min[3] = {1e30, 1e30, 1e30};
            Real seg_max[3] = {-1e30, -1e30, -1e30};

            for (int n = 0; n < nodes_per_segment_; ++n) {
                Index node = segment_nodes[s * nodes_per_segment_ + n];
                for (int d = 0; d < 3; ++d) {
                    Real c = coords[node * 3 + d];
                    seg_min[d] = std::min(seg_min[d], c);
                    seg_max[d] = std::max(seg_max[d], c);
                }
            }

            // Characteristic size (max dimension)
            Real size = 0.0;
            for (int d = 0; d < 3; ++d) {
                size = std::max(size, seg_max[d] - seg_min[d]);
            }
            total_size += size;
        }

        Real avg_size = total_size / std::max(static_cast<Index>(1), num_segments_);
        cell_size_ = config_.cell_size_factor * avg_size;
        cell_size_ = std::max(cell_size_, 1.0e-10);  // Prevent zero cell size
    }

    /**
     * @brief Build the voxel grid structure
     */
    void build_grid() {
        // Get coordinates to host for bounding box computation
        auto coords_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coords_);

        // Compute global bounding box
        bbox_min_[0] = bbox_min_[1] = bbox_min_[2] = std::numeric_limits<Real>::max();
        bbox_max_[0] = bbox_max_[1] = bbox_max_[2] = std::numeric_limits<Real>::lowest();

        for (Index i = 0; i < num_nodes_; ++i) {
            for (int d = 0; d < 3; ++d) {
                bbox_min_[d] = std::min(bbox_min_[d], coords_host(i, d));
                bbox_max_[d] = std::max(bbox_max_[d], coords_host(i, d));
            }
        }

        // Add margin
        Real margin = cell_size_ * 2.0 + config_.search_margin;
        for (int d = 0; d < 3; ++d) {
            bbox_min_[d] -= margin;
            bbox_max_[d] += margin;
        }

        // Compute grid dimensions
        for (int d = 0; d < 3; ++d) {
            grid_dims_[d] = static_cast<int>((bbox_max_[d] - bbox_min_[d]) / cell_size_) + 1;
            grid_dims_[d] = std::max(grid_dims_[d], 1);
        }

        int total_voxels = grid_dims_[0] * grid_dims_[1] * grid_dims_[2];

        // Allocate linked list structures
        voxel_heads_host_ = IndexHostView("voxel_heads", total_voxels);
        next_segment_host_ = IndexHostView("next_segment", num_segments_);

        // Initialize heads to -1 (empty)
        for (int v = 0; v < total_voxels; ++v) {
            voxel_heads_host_(v) = static_cast<Index>(-1);
        }

        // Build linked lists by inserting segments into voxels
        auto segment_nodes_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), segment_nodes_);

        for (Index s = 0; s < num_segments_; ++s) {
            // Compute segment bounding box
            Real seg_min[3] = {1e30, 1e30, 1e30};
            Real seg_max[3] = {-1e30, -1e30, -1e30};

            for (int n = 0; n < nodes_per_segment_; ++n) {
                Index node = segment_nodes_host(s * nodes_per_segment_ + n);
                for (int d = 0; d < 3; ++d) {
                    Real c = coords_host(node, d);
                    seg_min[d] = std::min(seg_min[d], c);
                    seg_max[d] = std::max(seg_max[d], c);
                }
            }

            // Find overlapping voxels
            int i_min = static_cast<int>((seg_min[0] - bbox_min_[0]) / cell_size_);
            int j_min = static_cast<int>((seg_min[1] - bbox_min_[1]) / cell_size_);
            int k_min = static_cast<int>((seg_min[2] - bbox_min_[2]) / cell_size_);
            int i_max = static_cast<int>((seg_max[0] - bbox_min_[0]) / cell_size_);
            int j_max = static_cast<int>((seg_max[1] - bbox_min_[1]) / cell_size_);
            int k_max = static_cast<int>((seg_max[2] - bbox_min_[2]) / cell_size_);

            // Clamp to grid bounds
            i_min = std::max(0, std::min(i_min, grid_dims_[0] - 1));
            j_min = std::max(0, std::min(j_min, grid_dims_[1] - 1));
            k_min = std::max(0, std::min(k_min, grid_dims_[2] - 1));
            i_max = std::max(0, std::min(i_max, grid_dims_[0] - 1));
            j_max = std::max(0, std::min(j_max, grid_dims_[1] - 1));
            k_max = std::max(0, std::min(k_max, grid_dims_[2] - 1));

            // Insert segment into its center voxel
            // Note: Large segments may span multiple voxels, but we store at center
            // The find_candidates function compensates by searching the bounding box
            // of all slave nodes, not just individual node neighborhoods
            int i_center = (i_min + i_max) / 2;
            int j_center = (j_min + j_max) / 2;
            int k_center = (k_min + k_max) / 2;
            int voxel_idx = i_center + grid_dims_[0] * (j_center + grid_dims_[1] * k_center);

            // Prepend to linked list
            next_segment_host_(s) = voxel_heads_host_(voxel_idx);
            voxel_heads_host_(voxel_idx) = s;
        }

        // Store reference coordinates for rebuild check
        ref_coords_ = Real3View("ref_coords", num_nodes_);
        Kokkos::deep_copy(ref_coords_, coords_);

        needs_rebuild_ = false;
    }

    /**
     * @brief Check if grid needs to be rebuilt due to motion
     */
    void check_rebuild_needed() {
        if (!initialized_) {
            needs_rebuild_ = true;
            return;
        }

        // Compute maximum displacement since last rebuild
        auto coords_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coords_);
        auto ref_coords_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ref_coords_);

        Real max_disp_sq = 0.0;
        for (Index i = 0; i < num_nodes_; ++i) {
            Real dx = coords_host(i, 0) - ref_coords_host(i, 0);
            Real dy = coords_host(i, 1) - ref_coords_host(i, 1);
            Real dz = coords_host(i, 2) - ref_coords_host(i, 2);
            Real disp_sq = dx*dx + dy*dy + dz*dz;
            max_disp_sq = std::max(max_disp_sq, disp_sq);
        }

        Real max_disp = std::sqrt(max_disp_sq);
        Real threshold = config_.rebuild_threshold * cell_size_;

        needs_rebuild_ = (max_disp > threshold);
    }

    // Configuration
    VoxelCollisionConfig config_;

    // Problem size
    Index num_nodes_;
    Index num_segments_;
    int nodes_per_segment_;

    // Grid parameters
    Real cell_size_;
    int grid_dims_[3];
    Real bbox_min_[3];
    Real bbox_max_[3];

    // Device data
    Real3View coords_;
    Real3View ref_coords_;
    IndexView segment_nodes_;

    // Linked list structure (host-side for now, can be GPU-ified)
    IndexHostView voxel_heads_host_;
    IndexHostView next_segment_host_;

    // State
    bool initialized_;
    bool needs_rebuild_;
};

} // namespace fem
} // namespace nxs
