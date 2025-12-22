#pragma once

/**
 * @file neighbor_search.hpp
 * @brief Spatial hashing for efficient neighbor search in SPH
 *
 * Implements cell-linked list with spatial hashing for O(N) neighbor search.
 *
 * Algorithm:
 * 1. Hash particles into cells of size h (smoothing length)
 * 2. For each particle, check only neighboring cells (3x3x3 = 27 cells)
 * 3. Pairs within support radius are stored in neighbor list
 */

#include <nexussim/core/core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

namespace nxs {
namespace sph {

// ============================================================================
// Cell Index for Spatial Hashing
// ============================================================================

struct CellIndex {
    int i, j, k;

    KOKKOS_INLINE_FUNCTION
    CellIndex() : i(0), j(0), k(0) {}

    KOKKOS_INLINE_FUNCTION
    CellIndex(int ii, int jj, int kk) : i(ii), j(jj), k(kk) {}

    KOKKOS_INLINE_FUNCTION
    bool operator==(const CellIndex& other) const {
        return i == other.i && j == other.j && k == other.k;
    }
};

// Hash function for CellIndex
struct CellHash {
    KOKKOS_INLINE_FUNCTION
    uint32_t operator()(const CellIndex& cell) const {
        // Large primes for good distribution
        constexpr uint32_t p1 = 73856093;
        constexpr uint32_t p2 = 19349663;
        constexpr uint32_t p3 = 83492791;
        return static_cast<uint32_t>(cell.i * p1 ^ cell.j * p2 ^ cell.k * p3);
    }
};

// ============================================================================
// Neighbor List Entry
// ============================================================================

struct NeighborPair {
    Index i;      ///< First particle
    Index j;      ///< Second particle
    Real r;       ///< Distance |r_ij|
    Real rx, ry, rz;  ///< Vector r_ij = r_i - r_j
};

// ============================================================================
// Spatial Hash Grid (CPU version)
// ============================================================================

/**
 * @brief Cell-linked list neighbor search using spatial hashing
 */
class SpatialHashGrid {
public:
    SpatialHashGrid(Real cell_size = 1.0, size_t initial_capacity = 1024)
        : cell_size_(cell_size)
        , inv_cell_size_(1.0 / cell_size)
    {
        cell_start_.reserve(initial_capacity);
        cell_count_.reserve(initial_capacity);
        particle_order_.reserve(initial_capacity);
    }

    /**
     * @brief Set cell size (should match smoothing length)
     */
    void set_cell_size(Real h) {
        cell_size_ = h;
        inv_cell_size_ = 1.0 / h;
    }

    /**
     * @brief Build spatial hash from particle positions
     * @param x, y, z Particle coordinates
     * @param n Number of particles
     */
    void build(const Real* x, const Real* y, const Real* z, size_t n) {
        num_particles_ = n;

        // Clear previous data
        cell_map_.clear();
        cell_start_.clear();
        cell_count_.clear();
        particle_order_.resize(n);
        particle_cells_.resize(n);

        // Compute cell for each particle
        for (size_t p = 0; p < n; ++p) {
            CellIndex cell = get_cell(x[p], y[p], z[p]);
            particle_cells_[p] = cell;

            // Get or create cell entry
            uint32_t hash = CellHash()(cell);
            if (cell_map_.find(hash) == cell_map_.end()) {
                cell_map_[hash] = cell_start_.size();
                cell_start_.push_back(0);
                cell_count_.push_back(0);
            }
            cell_count_[cell_map_[hash]]++;
        }

        // Compute cell start indices (prefix sum)
        size_t offset = 0;
        for (size_t c = 0; c < cell_count_.size(); ++c) {
            size_t count = cell_count_[c];
            cell_start_[c] = offset;
            offset += count;
            cell_count_[c] = 0;  // Reset for insertion
        }

        // Insert particles into cells
        for (size_t p = 0; p < n; ++p) {
            uint32_t hash = CellHash()(particle_cells_[p]);
            size_t cell_id = cell_map_[hash];
            size_t pos = cell_start_[cell_id] + cell_count_[cell_id];
            particle_order_[pos] = p;
            cell_count_[cell_id]++;
        }
    }

    /**
     * @brief Find all neighbor pairs within support radius
     * @param x, y, z Particle coordinates
     * @param support_radius Maximum distance for neighbors
     * @param pairs Output neighbor pairs
     */
    void find_neighbors(const Real* x, const Real* y, const Real* z,
                        Real support_radius, std::vector<NeighborPair>& pairs) const {
        pairs.clear();
        Real r2_max = support_radius * support_radius;

        // For each particle
        for (size_t p = 0; p < num_particles_; ++p) {
            CellIndex cell_p = get_cell(x[p], y[p], z[p]);

            // Check 27 neighboring cells (including self)
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    for (int dk = -1; dk <= 1; ++dk) {
                        CellIndex neighbor_cell(cell_p.i + di, cell_p.j + dj, cell_p.k + dk);
                        uint32_t hash = CellHash()(neighbor_cell);

                        auto it = cell_map_.find(hash);
                        if (it == cell_map_.end()) continue;

                        size_t cell_id = it->second;
                        size_t start = cell_start_[cell_id];
                        size_t count = cell_count_[cell_id];

                        // Check particles in this cell
                        for (size_t idx = start; idx < start + count; ++idx) {
                            Index q = particle_order_[idx];

                            // Only store each pair once (p < q)
                            if (q <= p) continue;

                            Real rx = x[p] - x[q];
                            Real ry = y[p] - y[q];
                            Real rz = z[p] - z[q];
                            Real r2 = rx * rx + ry * ry + rz * rz;

                            if (r2 < r2_max && r2 > 1e-20) {
                                NeighborPair pair;
                                pair.i = p;
                                pair.j = q;
                                pair.r = std::sqrt(r2);
                                pair.rx = rx;
                                pair.ry = ry;
                                pair.rz = rz;
                                pairs.push_back(pair);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Find neighbors of a single particle
     */
    void find_particle_neighbors(Index p, const Real* x, const Real* y, const Real* z,
                                  Real support_radius, std::vector<Index>& neighbors) const {
        neighbors.clear();
        Real r2_max = support_radius * support_radius;

        CellIndex cell_p = get_cell(x[p], y[p], z[p]);

        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                for (int dk = -1; dk <= 1; ++dk) {
                    CellIndex neighbor_cell(cell_p.i + di, cell_p.j + dj, cell_p.k + dk);
                    uint32_t hash = CellHash()(neighbor_cell);

                    auto it = cell_map_.find(hash);
                    if (it == cell_map_.end()) continue;

                    size_t cell_id = it->second;
                    size_t start = cell_start_[cell_id];
                    size_t count = cell_count_[cell_id];

                    for (size_t idx = start; idx < start + count; ++idx) {
                        Index q = particle_order_[idx];
                        if (q == p) continue;

                        Real rx = x[p] - x[q];
                        Real ry = y[p] - y[q];
                        Real rz = z[p] - z[q];
                        Real r2 = rx * rx + ry * ry + rz * rz;

                        if (r2 < r2_max) {
                            neighbors.push_back(q);
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Get number of unique cells
     */
    size_t num_cells() const { return cell_map_.size(); }

    /**
     * @brief Get cell size
     */
    Real cell_size() const { return cell_size_; }

private:
    KOKKOS_INLINE_FUNCTION
    CellIndex get_cell(Real x, Real y, Real z) const {
        return CellIndex(
            static_cast<int>(std::floor(x * inv_cell_size_)),
            static_cast<int>(std::floor(y * inv_cell_size_)),
            static_cast<int>(std::floor(z * inv_cell_size_))
        );
    }

    Real cell_size_;
    Real inv_cell_size_;
    size_t num_particles_ = 0;

    // Spatial hash map: hash -> cell index
    std::unordered_map<uint32_t, size_t> cell_map_;

    // Cell data
    std::vector<size_t> cell_start_;    // Start index in particle_order
    std::vector<size_t> cell_count_;    // Number of particles in cell

    // Particle ordering
    std::vector<Index> particle_order_; // Sorted by cell
    std::vector<CellIndex> particle_cells_;  // Cell for each particle
};

// ============================================================================
// Compact Neighbor List
// ============================================================================

/**
 * @brief Compact neighbor list storage for efficient iteration
 *
 * Stores neighbors in CSR-like format:
 * - neighbor_offset[i] = start of i's neighbors in neighbor_list
 * - neighbor_list[offset:offset+count] = neighbors of particle i
 */
class CompactNeighborList {
public:
    CompactNeighborList() = default;

    /**
     * @brief Build from neighbor pairs
     */
    void build_from_pairs(const std::vector<NeighborPair>& pairs, size_t num_particles) {
        num_particles_ = num_particles;

        // Count neighbors per particle
        std::vector<size_t> counts(num_particles, 0);
        for (const auto& pair : pairs) {
            counts[pair.i]++;
            counts[pair.j]++;  // Symmetric
        }

        // Compute offsets (prefix sum)
        neighbor_offset_.resize(num_particles + 1);
        neighbor_offset_[0] = 0;
        for (size_t i = 0; i < num_particles; ++i) {
            neighbor_offset_[i + 1] = neighbor_offset_[i] + counts[i];
        }

        // Allocate neighbor list
        size_t total_neighbors = neighbor_offset_[num_particles];
        neighbor_list_.resize(total_neighbors);
        neighbor_dist_.resize(total_neighbors);

        // Reset counts for insertion
        std::fill(counts.begin(), counts.end(), 0);

        // Insert neighbors
        for (const auto& pair : pairs) {
            // Add j to i's neighbors
            size_t pos_i = neighbor_offset_[pair.i] + counts[pair.i];
            neighbor_list_[pos_i] = pair.j;
            neighbor_dist_[pos_i] = pair.r;
            counts[pair.i]++;

            // Add i to j's neighbors
            size_t pos_j = neighbor_offset_[pair.j] + counts[pair.j];
            neighbor_list_[pos_j] = pair.i;
            neighbor_dist_[pos_j] = pair.r;
            counts[pair.j]++;
        }
    }

    /**
     * @brief Get neighbors of particle i
     */
    std::pair<const Index*, size_t> neighbors(Index i) const {
        size_t start = neighbor_offset_[i];
        size_t count = neighbor_offset_[i + 1] - start;
        return {neighbor_list_.data() + start, count};
    }

    /**
     * @brief Get number of neighbors for particle i
     */
    size_t num_neighbors(Index i) const {
        return neighbor_offset_[i + 1] - neighbor_offset_[i];
    }

    /**
     * @brief Get total number of neighbor entries
     */
    size_t total_neighbors() const {
        return neighbor_list_.size();
    }

    /**
     * @brief Average neighbors per particle
     */
    Real avg_neighbors() const {
        if (num_particles_ == 0) return 0.0;
        return static_cast<Real>(neighbor_list_.size()) / num_particles_;
    }

    size_t num_particles() const { return num_particles_; }

private:
    size_t num_particles_ = 0;
    std::vector<size_t> neighbor_offset_;  // CSR offsets
    std::vector<Index> neighbor_list_;     // Neighbor indices
    std::vector<Real> neighbor_dist_;      // Distance to neighbor
};

} // namespace sph
} // namespace nxs
