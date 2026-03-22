#pragma once

/**
 * @file sph_wave29.hpp
 * @brief 5 Production SPH Features for Wave 29
 *
 * Features implemented:
 *   1.  ParticleSplitMerge          - Adaptive 1->4 (2D) / 1->8 (3D) with conservation
 *   2.  SPHFracturePropagation      - Bond-breaking crack propagation
 *   3.  KokkosHashNeighborSearch    - GPU-optimized spatial hash neighbor search
 *   4.  AdvancedSPHBoundary         - Rigid wall, free-slip, no-slip, porous boundaries
 *   5.  SPHStressPoints             - Dual particle method for tensile instability
 *
 * All classes use Real type, KOKKOS_INLINE_FUNCTION where applicable.
 * Namespace: nxs::sph
 */

#include <nexussim/physics/material.hpp>
#include <nexussim/sph/sph_kernel.hpp>
#include <nexussim/sph/neighbor_search.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <array>
#include <unordered_map>
#include <utility>
#include <cstdint>

namespace nxs {
namespace sph {

// ============================================================================
// 1. ParticleSplitMerge — Adaptive 1→4 (2D) / 1→8 (3D) with Conservation
// ============================================================================

/**
 * @brief Adaptive particle splitting and merging for SPH refinement
 *
 * Implements h-adaptivity through particle split/merge operations:
 * - Split: when a particle's smoothing length exceeds h_max_ratio * h_initial,
 *   it is split into N children (4 in 2D, 8 in 3D) placed symmetrically around
 *   the parent position.
 * - Merge: when a particle's smoothing length drops below h_min_ratio * h_initial
 *   and the local neighbor density is high enough, two particles merge into one.
 *
 * Conservation properties:
 * - Mass: m_child = m_parent / N_children
 * - Momentum: v_child = v_parent (velocity inherited)
 * - Energy: e_child = e_parent (specific energy inherited)
 * - Smoothing length: h_child = h_parent / N_children^(1/dim)
 *
 * Reference: Feldman & Bonet (2007), Dynamic refinement and coarsening in SPH
 */
class ParticleSplitMerge {
public:
    /**
     * @brief Construct with adaptive refinement parameters
     * @param h_max_ratio Ratio above which split is triggered (default 2.0)
     * @param h_min_ratio Ratio below which merge is triggered (default 0.5)
     * @param dim Spatial dimension (2 or 3, default 3)
     */
    ParticleSplitMerge(Real h_max_ratio = 2.0, Real h_min_ratio = 0.5, int dim = 3)
        : h_max_ratio_(h_max_ratio)
        , h_min_ratio_(h_min_ratio)
        , dim_(dim)
    {}

    /**
     * @brief Check whether a particle should be split
     *
     * A particle should be split when its current smoothing length exceeds
     * the maximum ratio times the initial smoothing length. This indicates
     * the particle has expanded into a region requiring finer resolution.
     *
     * @param h_current Current smoothing length of the particle
     * @param h_initial Initial (reference) smoothing length
     * @return true if the particle should be split
     */
    KOKKOS_INLINE_FUNCTION
    bool should_split(Real h_current, Real h_initial) const {
        return h_current > h_max_ratio_ * h_initial;
    }

    /**
     * @brief Check whether a particle should be merged
     *
     * A particle should be merged when its current smoothing length is below
     * the minimum ratio times the initial value AND the local neighbor count
     * is sufficient (indicating redundant resolution).
     *
     * @param h_current Current smoothing length
     * @param h_initial Initial (reference) smoothing length
     * @param num_neighbors Current number of neighbors
     * @param min_neighbors Minimum neighbors required for merge (default 8)
     * @return true if the particle should be merged
     */
    KOKKOS_INLINE_FUNCTION
    bool should_merge(Real h_current, Real h_initial,
                      int num_neighbors, int min_neighbors = 8) const {
        return (h_current < h_min_ratio_ * h_initial) &&
               (num_neighbors >= min_neighbors);
    }

    /**
     * @brief Split a 3D particle into 8 children placed at cube corners
     *
     * Children are placed at (x ± delta, y ± delta, z ± delta) where
     * delta = h_parent / 4. Each child inherits the parent velocity
     * and receives 1/8 of the parent mass. The child smoothing length
     * is h_parent / 8^(1/3) = h_parent / 2.
     *
     * @param parent_pos Parent position [3]
     * @param parent_vel Parent velocity [3]
     * @param parent_mass Parent mass
     * @param parent_h Parent smoothing length
     * @param child_pos Output child positions [8][3]
     * @param child_vel Output child velocities [8][3]
     * @param child_mass Output child masses [8]
     * @param child_h Output child smoothing lengths [8]
     * @param num_children Output number of children created (always 8)
     */
    KOKKOS_INLINE_FUNCTION
    void split_3d(const Real parent_pos[3], const Real parent_vel[3],
                  Real parent_mass, Real parent_h,
                  Real child_pos[][3], Real child_vel[][3],
                  Real child_mass[], Real child_h[],
                  int& num_children) const {
        num_children = 8;
        Real delta = parent_h / 4.0;

        // 8 children at corners of a cube centered on parent
        // Signs: (-,-,-), (-,-,+), (-,+,-), (-,+,+),
        //        (+,-,-), (+,-,+), (+,+,-), (+,+,+)
        int signs[8][3] = {
            {-1, -1, -1}, {-1, -1, +1}, {-1, +1, -1}, {-1, +1, +1},
            {+1, -1, -1}, {+1, -1, +1}, {+1, +1, -1}, {+1, +1, +1}
        };

        // h_child = h_parent / 8^(1/3) = h_parent / 2
        Real h_child = parent_h / 2.0;
        Real m_child = parent_mass / 8.0;

        for (int c = 0; c < 8; ++c) {
            for (int d = 0; d < 3; ++d) {
                child_pos[c][d] = parent_pos[d] + signs[c][d] * delta;
                child_vel[c][d] = parent_vel[d];  // Inherit velocity
            }
            child_mass[c] = m_child;
            child_h[c] = h_child;
        }
    }

    /**
     * @brief Split a 2D particle into 4 children placed at square corners
     *
     * Children are placed at (x ± delta, y ± delta) where
     * delta = h_parent / 4. Each child inherits the parent velocity
     * and receives 1/4 of the parent mass. The child smoothing length
     * is h_parent / 4^(1/2) = h_parent / 2.
     *
     * @param parent_pos Parent position [2]
     * @param parent_vel Parent velocity [2]
     * @param parent_mass Parent mass
     * @param parent_h Parent smoothing length
     * @param child_pos Output child positions [4][2]
     * @param child_vel Output child velocities [4][2]
     * @param child_mass Output child masses [4]
     * @param child_h Output child smoothing lengths [4]
     * @param num_children Output number of children created (always 4)
     */
    KOKKOS_INLINE_FUNCTION
    void split_2d(const Real parent_pos[2], const Real parent_vel[2],
                  Real parent_mass, Real parent_h,
                  Real child_pos[][2], Real child_vel[][2],
                  Real child_mass[], Real child_h[],
                  int& num_children) const {
        num_children = 4;
        Real delta = parent_h / 4.0;

        // 4 children at corners of a square centered on parent
        int signs[4][2] = {
            {-1, -1}, {-1, +1}, {+1, -1}, {+1, +1}
        };

        // h_child = h_parent / 4^(1/2) = h_parent / 2
        Real h_child = parent_h / 2.0;
        Real m_child = parent_mass / 4.0;

        for (int c = 0; c < 4; ++c) {
            for (int d = 0; d < 2; ++d) {
                child_pos[c][d] = parent_pos[d] + signs[c][d] * delta;
                child_vel[c][d] = parent_vel[d];  // Inherit velocity
            }
            child_mass[c] = m_child;
            child_h[c] = h_child;
        }
    }

    /**
     * @brief Merge two particles into one, conserving mass and momentum
     *
     * The merged particle is placed at the center of mass of the two parents.
     * Its velocity is the momentum-weighted average (conservation of momentum).
     * Mass is summed.
     *
     * @param pos_a Position of particle a [3]
     * @param pos_b Position of particle b [3]
     * @param vel_a Velocity of particle a [3]
     * @param vel_b Velocity of particle b [3]
     * @param mass_a Mass of particle a
     * @param mass_b Mass of particle b
     * @param merged_pos Output merged position [3]
     * @param merged_vel Output merged velocity [3]
     * @param merged_mass Output merged mass
     */
    KOKKOS_INLINE_FUNCTION
    void merge_particles(const Real pos_a[3], const Real pos_b[3],
                         const Real vel_a[3], const Real vel_b[3],
                         Real mass_a, Real mass_b,
                         Real merged_pos[3], Real merged_vel[3],
                         Real& merged_mass) const {
        merged_mass = mass_a + mass_b;
        Real inv_total = 1.0 / merged_mass;

        for (int d = 0; d < 3; ++d) {
            // Center of mass position
            merged_pos[d] = (mass_a * pos_a[d] + mass_b * pos_b[d]) * inv_total;
            // Momentum-conserving velocity
            merged_vel[d] = (mass_a * vel_a[d] + mass_b * vel_b[d]) * inv_total;
        }
    }

    /**
     * @brief Check that mass conservation holds within tolerance
     *
     * @param total_mass_before Total mass before split/merge
     * @param total_mass_after Total mass after split/merge
     * @param tol Relative tolerance (default 1e-12)
     * @return true if conservation holds
     */
    KOKKOS_INLINE_FUNCTION
    bool check_conservation(Real total_mass_before, Real total_mass_after,
                            Real tol = 1.0e-12) const {
        if (total_mass_before < 1.0e-30) return total_mass_after < 1.0e-30;
        Real rel_err = std::abs(total_mass_after - total_mass_before) / total_mass_before;
        return rel_err <= tol;
    }

    // Getters
    Real h_max_ratio() const { return h_max_ratio_; }
    Real h_min_ratio() const { return h_min_ratio_; }
    int dimension() const { return dim_; }
    const char* name() const { return "ParticleSplitMerge"; }

    // Setters
    void set_h_max_ratio(Real ratio) { h_max_ratio_ = ratio; }
    void set_h_min_ratio(Real ratio) { h_min_ratio_ = ratio; }
    void set_dimension(int dim) { dim_ = dim; }

private:
    Real h_max_ratio_;   ///< Split threshold ratio
    Real h_min_ratio_;   ///< Merge threshold ratio
    int dim_;            ///< Spatial dimension (2 or 3)
};


// ============================================================================
// 2. SPHFracturePropagation — Bond-Breaking Crack Propagation
// ============================================================================

/**
 * @brief Bond-based fracture propagation for SPH
 *
 * Implements a peridynamics-inspired bond-based fracture model for SPH:
 * - Bonds connect pairs of particles within a horizon distance
 * - Bond strain: s_ij = (|x_j - x_i| - |X_j - X_i|) / |X_j - X_i|
 *   where x = current position, X = reference position
 * - A bond breaks irreversibly when s_ij > s_critical
 * - Damage at particle i: d_i = 1 - (active_bonds / initial_bonds)
 * - Crack tip identified where 0.3 < d < 0.7
 *
 * Critical stretch formulas (Silling & Askari 2005):
 *   2D: s_c = sqrt(G_c / (3 * mu * delta))
 *   3D: s_c = sqrt(5 * G_c / (9 * kappa * delta))
 *
 * where G_c = fracture energy, mu = shear modulus, kappa = bulk modulus,
 * delta = horizon (neighborhood radius).
 *
 * Reference: Silling (2000), Reformulation of elasticity theory;
 *            Silling & Askari (2005), Meshfree method based on peridynamic model
 */
class SPHFracturePropagation {
public:
    /**
     * @brief Construct with fracture parameters
     * @param s_critical Critical bond stretch for failure (default 0.01)
     * @param horizon Peridynamic horizon / neighborhood radius (default 3.0)
     */
    SPHFracturePropagation(Real s_critical = 0.01, Real horizon = 3.0)
        : s_critical_(s_critical)
        , horizon_(horizon)
    {}

    /**
     * @brief Compute bond strain between two particles
     *
     * The bond strain is defined as the relative elongation of the bond:
     *   s = (|x_j - x_i| - |X_j - X_i|) / |X_j - X_i|
     *
     * Positive strain indicates tension (stretching), negative indicates
     * compression. Only tensile strain leads to bond failure.
     *
     * @param x_i Current position of particle i [3]
     * @param x_j Current position of particle j [3]
     * @param X_i Reference position of particle i [3]
     * @param X_j Reference position of particle j [3]
     * @return Bond strain (positive = tension)
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_bond_strain(const Real x_i[3], const Real x_j[3],
                             const Real X_i[3], const Real X_j[3]) const {
        // Current distance
        Real dx_c = x_j[0] - x_i[0];
        Real dy_c = x_j[1] - x_i[1];
        Real dz_c = x_j[2] - x_i[2];
        Real r_current = std::sqrt(dx_c * dx_c + dy_c * dy_c + dz_c * dz_c);

        // Reference distance
        Real dx_r = X_j[0] - X_i[0];
        Real dy_r = X_j[1] - X_i[1];
        Real dz_r = X_j[2] - X_i[2];
        Real r_ref = std::sqrt(dx_r * dx_r + dy_r * dy_r + dz_r * dz_r);

        if (r_ref < 1.0e-30) return 0.0;  // Coincident particles

        return (r_current - r_ref) / r_ref;
    }

    /**
     * @brief Check if a bond is intact (has not exceeded critical strain)
     *
     * @param strain Current bond strain
     * @param s_crit Critical stretch threshold
     * @return true if the bond is intact (strain <= s_crit)
     */
    KOKKOS_INLINE_FUNCTION
    bool check_bond(Real strain, Real s_crit) const {
        return strain <= s_crit;
    }

    /**
     * @brief Compute damage at a particle from active/initial bond counts
     *
     * Damage is defined as:
     *   d = 1 - (active_bonds / initial_bonds)
     *
     * d = 0: fully intact
     * d = 1: fully failed (all bonds broken)
     *
     * @param active_bonds Number of currently intact bonds
     * @param initial_bonds Number of bonds at initialization
     * @return Damage value in [0, 1]
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_damage(int active_bonds, int initial_bonds) const {
        if (initial_bonds <= 0) return 0.0;
        Real ratio = static_cast<Real>(active_bonds) / static_cast<Real>(initial_bonds);
        Real damage = 1.0 - ratio;
        // Clamp to [0, 1]
        if (damage < 0.0) damage = 0.0;
        if (damage > 1.0) damage = 1.0;
        return damage;
    }

    /**
     * @brief Check if a particle is at a crack tip
     *
     * A crack tip is identified by intermediate damage values,
     * indicating a transition between intact and fully damaged regions.
     *
     * @param damage Current damage value
     * @param low Lower threshold for crack tip (default 0.3)
     * @param high Upper threshold for crack tip (default 0.7)
     * @return true if particle is at a crack tip
     */
    KOKKOS_INLINE_FUNCTION
    bool is_crack_tip(Real damage, Real low = 0.3, Real high = 0.7) const {
        return (damage > low) && (damage < high);
    }

    /**
     * @brief Compute critical stretch from material properties
     *
     * For 2D: s_c = sqrt(G_c / (3 * mu * delta))
     * For 3D: s_c = sqrt(5 * G_c / (9 * kappa * delta))
     *
     * @param G_c Fracture energy (J/m^2)
     * @param modulus Shear modulus (2D) or bulk modulus (3D) (Pa)
     * @param horizon Peridynamic horizon distance
     * @param dim Spatial dimension (2 or 3)
     * @return Critical stretch threshold
     */
    KOKKOS_INLINE_FUNCTION
    Real critical_stretch(Real G_c, Real modulus, Real horizon, int dim) const {
        if (modulus < 1.0e-30 || horizon < 1.0e-30) return 0.0;
        if (dim == 2) {
            // 2D: s_c = sqrt(G_c / (3 * mu * delta))
            return std::sqrt(G_c / (3.0 * modulus * horizon));
        } else {
            // 3D: s_c = sqrt(5 * G_c / (9 * kappa * delta))
            return std::sqrt(5.0 * G_c / (9.0 * modulus * horizon));
        }
    }

    /**
     * @brief Propagate fracture by checking all bonds and breaking failed ones
     *
     * Iterates through all bonds, checks strain against critical stretch,
     * and deactivates bonds that have exceeded the threshold. Bond breaking
     * is irreversible — once broken, a bond cannot be restored.
     *
     * @param bond_strains Array of current bond strains
     * @param bond_active Array of bond status (true = intact, modified in place)
     * @param num_bonds Total number of bonds
     * @param s_crit Critical stretch for failure
     * @return Number of bonds broken in this propagation step
     */
    int propagate(const Real* bond_strains, bool* bond_active,
                  int num_bonds, Real s_crit) const {
        int num_broken = 0;
        for (int b = 0; b < num_bonds; ++b) {
            if (bond_active[b]) {
                if (!check_bond(bond_strains[b], s_crit)) {
                    bond_active[b] = false;
                    ++num_broken;
                }
            }
        }
        return num_broken;
    }

    /**
     * @brief Count active bonds in an array
     *
     * @param bond_active Bond status array
     * @param num_bonds Total number of bonds
     * @return Number of active (intact) bonds
     */
    int count_active(const bool* bond_active, int num_bonds) const {
        int count = 0;
        for (int b = 0; b < num_bonds; ++b) {
            if (bond_active[b]) ++count;
        }
        return count;
    }

    // Getters
    Real s_critical() const { return s_critical_; }
    Real horizon() const { return horizon_; }
    const char* name() const { return "SPHFracturePropagation"; }

    // Setters
    void set_s_critical(Real sc) { s_critical_ = sc; }
    void set_horizon(Real h) { horizon_ = h; }

private:
    Real s_critical_;   ///< Critical bond stretch
    Real horizon_;      ///< Peridynamic horizon distance
};


// ============================================================================
// 3. KokkosHashNeighborSearch — GPU-Optimized Spatial Hash
// ============================================================================

/**
 * @brief GPU-optimized spatial hash-based neighbor search
 *
 * Implements a hash-grid spatial data structure for efficient neighbor
 * search in SPH simulations. The hash function maps 3D cell indices
 * to a flat hash table using large prime numbers:
 *
 *   hash(ix, iy, iz) = (ix * P1 xor iy * P2 xor iz * P3) mod TABLE_SIZE
 *
 * where P1 = 73856093, P2 = 19349663, P3 = 83492791.
 *
 * Grid cell size is set to 2*h (covering the kernel support), so each
 * particle needs to search at most 27 neighboring cells (3x3x3).
 *
 * This implementation provides a host-side reference using std::unordered_map
 * and std::vector for correctness testing and CPU fallback. A GPU-native
 * Kokkos::UnorderedMap version can be substituted for production runs.
 *
 * Reference: Teschner et al. (2003), Optimized spatial hashing for collision
 *            detection of deformable objects
 */
class KokkosHashNeighborSearch {
public:
    /// Hash table prime constants
    static constexpr uint64_t P1 = 73856093ULL;
    static constexpr uint64_t P2 = 19349663ULL;
    static constexpr uint64_t P3 = 83492791ULL;

    /**
     * @brief Construct with cell size and hash table size
     * @param cell_size Grid cell size (typically 2*h)
     * @param table_size Hash table size (default 65536, should be power of 2)
     */
    KokkosHashNeighborSearch(Real cell_size = 1.0, int table_size = 65536)
        : cell_size_(cell_size)
        , table_size_(table_size)
    {
        hash_table_.resize(table_size);
    }

    /**
     * @brief Compute hash value for a cell index triplet
     *
     * Uses XOR-based spatial hashing with large primes:
     *   hash = ((ix * P1) ^ (iy * P2) ^ (iz * P3)) % TABLE_SIZE
     *
     * The use of XOR minimizes correlation between adjacent cells
     * while the prime multiplication spreads keys uniformly.
     *
     * @param ix Cell x-index
     * @param iy Cell y-index
     * @param iz Cell z-index
     * @return Hash value in [0, table_size)
     */
    KOKKOS_INLINE_FUNCTION
    int hash_cell(int ix, int iy, int iz) const {
        // Use unsigned arithmetic to handle negative indices
        uint64_t uix = static_cast<uint64_t>(static_cast<int64_t>(ix));
        uint64_t uiy = static_cast<uint64_t>(static_cast<int64_t>(iy));
        uint64_t uiz = static_cast<uint64_t>(static_cast<int64_t>(iz));

        uint64_t hash_val = (uix * P1) ^ (uiy * P2) ^ (uiz * P3);
        return static_cast<int>(hash_val % static_cast<uint64_t>(table_size_));
    }

    /**
     * @brief Compute cell indices from world-space position
     *
     * @param x X-coordinate
     * @param y Y-coordinate
     * @param z Z-coordinate
     * @param ix Output cell x-index
     * @param iy Output cell y-index
     * @param iz Output cell z-index
     */
    KOKKOS_INLINE_FUNCTION
    void cell_index(Real x, Real y, Real z,
                    int& ix, int& iy, int& iz) const {
        ix = static_cast<int>(std::floor(x / cell_size_));
        iy = static_cast<int>(std::floor(y / cell_size_));
        iz = static_cast<int>(std::floor(z / cell_size_));
    }

    /**
     * @brief Build hash table from particle positions
     *
     * Clears existing table and inserts all particles into their
     * corresponding hash cells. Each hash bucket stores a list of
     * particle indices that fall into that cell.
     *
     * @param positions Particle positions (flat array of [x,y,z] triplets)
     * @param num_particles Number of particles
     */
    void build(const Real positions[][3], int num_particles) {
        // Clear hash table
        for (int i = 0; i < table_size_; ++i) {
            hash_table_[i].clear();
        }

        // Store all positions for later queries
        num_particles_ = num_particles;
        particle_pos_.resize(num_particles);
        for (int p = 0; p < num_particles; ++p) {
            particle_pos_[p] = {positions[p][0], positions[p][1], positions[p][2]};
        }

        // Insert each particle into hash table
        for (int p = 0; p < num_particles; ++p) {
            int ix, iy, iz;
            cell_index(positions[p][0], positions[p][1], positions[p][2],
                       ix, iy, iz);
            int h = hash_cell(ix, iy, iz);
            hash_table_[h].push_back(p);
        }
    }

    /**
     * @brief Build hash table from std::vector of positions
     *
     * Convenience overload for host-side use.
     *
     * @param positions Vector of position arrays [3]
     */
    void build(const std::vector<std::array<Real, 3>>& positions) {
        int n = static_cast<int>(positions.size());
        // Clear hash table
        for (int i = 0; i < table_size_; ++i) {
            hash_table_[i].clear();
        }

        num_particles_ = n;
        particle_pos_ = positions;

        for (int p = 0; p < n; ++p) {
            int ix, iy, iz;
            cell_index(positions[p][0], positions[p][1], positions[p][2],
                       ix, iy, iz);
            int h = hash_cell(ix, iy, iz);
            hash_table_[h].push_back(p);
        }
    }

    /**
     * @brief Find all neighbors of a particle within a maximum distance
     *
     * Searches the 27 cells (3x3x3) surrounding the particle's cell.
     * For each candidate, computes the actual Euclidean distance and
     * includes only those within max_dist.
     *
     * @param particle_id Index of the query particle
     * @param max_dist Maximum search distance
     * @return Vector of neighbor particle indices (excluding self)
     */
    std::vector<int> find_neighbors(int particle_id, Real max_dist) const {
        std::vector<int> neighbors;
        if (particle_id < 0 || particle_id >= num_particles_) return neighbors;

        Real px = particle_pos_[particle_id][0];
        Real py = particle_pos_[particle_id][1];
        Real pz = particle_pos_[particle_id][2];

        int cix, ciy, ciz;
        cell_index(px, py, pz, cix, ciy, ciz);

        Real max_dist2 = max_dist * max_dist;

        // Search 27 neighboring cells
        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                for (int dk = -1; dk <= 1; ++dk) {
                    int h = hash_cell(cix + di, ciy + dj, ciz + dk);
                    const auto& bucket = hash_table_[h];
                    for (int cand : bucket) {
                        if (cand == particle_id) continue;  // Skip self
                        Real dx = particle_pos_[cand][0] - px;
                        Real dy = particle_pos_[cand][1] - py;
                        Real dz = particle_pos_[cand][2] - pz;
                        Real dist2 = dx * dx + dy * dy + dz * dz;
                        if (dist2 <= max_dist2) {
                            neighbors.push_back(cand);
                        }
                    }
                }
            }
        }

        // Remove duplicates (hash collisions can cause particles from
        // distant cells to appear in the same bucket)
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                        neighbors.end());

        return neighbors;
    }

    /**
     * @brief Count neighbors of a particle within maximum distance
     *
     * @param particle_id Index of the query particle
     * @param max_dist Maximum search distance
     * @return Number of neighbors within max_dist
     */
    int count_neighbors(int particle_id, Real max_dist) const {
        return static_cast<int>(find_neighbors(particle_id, max_dist).size());
    }

    // Getters
    Real cell_size() const { return cell_size_; }
    int table_size() const { return table_size_; }
    int num_particles() const { return num_particles_; }
    const char* name() const { return "KokkosHashNeighborSearch"; }

    // Setters
    void set_cell_size(Real cs) { cell_size_ = cs; }

private:
    Real cell_size_;              ///< Grid cell size (typically 2*h)
    int table_size_;              ///< Size of the hash table
    int num_particles_ = 0;      ///< Number of particles in current build

    /// Hash table: bucket index → list of particle IDs
    std::vector<std::vector<int>> hash_table_;

    /// Stored particle positions for distance queries
    std::vector<std::array<Real, 3>> particle_pos_;
};


// ============================================================================
// 4. AdvancedSPHBoundary — Rigid Wall, Free-Slip, No-Slip, Porous
// ============================================================================

/**
 * @brief Advanced SPH boundary treatment with multiple wall types
 *
 * Implements ghost/mirror particle boundary enforcement for SPH:
 *
 * - Rigid Wall (No-Slip): ghost particles mirror fluid positions across
 *   the wall plane, with reversed velocity (all components).
 *   v_ghost = -v_fluid
 *
 * - Free-Slip: ghost position mirrored, only normal velocity component
 *   reversed. Tangential component preserved.
 *   v_ghost = v_fluid - 2*(v_fluid . n)*n
 *
 * - No-Slip: all velocity components reversed relative to wall.
 *   v_ghost = -v_fluid
 *
 * - Porous: partial reflection with porosity coefficient alpha_p.
 *   v_ghost = (1 - alpha_p) * v_reflected + alpha_p * v_fluid
 *   alpha_p = 0 → impermeable (rigid), alpha_p = 1 → fully permeable
 *
 * Wall plane: n . x = d (where n is the outward normal pointing into fluid)
 *
 * Reference: Morris et al. (1997), Modeling low Reynolds number
 *            incompressible flows using SPH;
 *            Adami et al. (2012), Generalized wall boundary condition
 */

/// Boundary type enumeration
enum class BoundaryType {
    RigidWall,   ///< Impermeable, no-slip wall
    FreeSlip,    ///< Impermeable, free-slip (tangential velocity preserved)
    NoSlip,      ///< Same as RigidWall (all velocity reversed)
    Porous       ///< Partially permeable wall with porosity factor
};

class AdvancedSPHBoundary {
public:
    /**
     * @brief Construct a boundary with specified type and geometry
     * @param type Boundary type (RigidWall, FreeSlip, NoSlip, Porous)
     * @param normal Outward unit normal [3] (pointing into fluid domain)
     * @param d Signed distance of wall plane from origin (n . x = d)
     * @param porosity Porosity factor for Porous walls (0 = impermeable, 1 = fully open)
     */
    AdvancedSPHBoundary(BoundaryType type, const Real normal[3], Real d,
                        Real porosity = 0.0)
        : type_(type)
        , d_(d)
        , porosity_(porosity)
    {
        // Copy and normalize the normal vector
        Real mag = std::sqrt(normal[0] * normal[0] +
                             normal[1] * normal[1] +
                             normal[2] * normal[2]);
        if (mag < 1.0e-30) mag = 1.0;
        normal_[0] = normal[0] / mag;
        normal_[1] = normal[1] / mag;
        normal_[2] = normal[2] / mag;
    }

    /**
     * @brief Reflect a fluid particle position across the wall to create a ghost
     *
     * Ghost position is the mirror image of the fluid particle:
     *   x_ghost = x_fluid - 2 * (x_fluid . n - d) * n
     *
     * @param fluid_pos Fluid particle position [3]
     * @param ghost_pos Output ghost particle position [3]
     */
    KOKKOS_INLINE_FUNCTION
    void reflect_position(const Real fluid_pos[3], Real ghost_pos[3]) const {
        Real dot = fluid_pos[0] * normal_[0] +
                   fluid_pos[1] * normal_[1] +
                   fluid_pos[2] * normal_[2];
        Real dist_to_wall = dot - d_;

        for (int d = 0; d < 3; ++d) {
            ghost_pos[d] = fluid_pos[d] - 2.0 * dist_to_wall * normal_[d];
        }
    }

    /**
     * @brief Reflect a fluid particle velocity for the ghost particle
     *
     * The reflection depends on boundary type:
     * - NoSlip/RigidWall: v_ghost = -v_fluid
     * - FreeSlip: v_ghost = v_fluid - 2*(v_fluid . n)*n
     * - Porous: v_ghost = (1-alpha) * v_reflected + alpha * v_fluid
     *   where v_reflected is the no-slip reflection
     *
     * @param fluid_vel Fluid particle velocity [3]
     * @param ghost_vel Output ghost particle velocity [3]
     */
    KOKKOS_INLINE_FUNCTION
    void reflect_velocity(const Real fluid_vel[3], Real ghost_vel[3]) const {
        switch (type_) {
            case BoundaryType::NoSlip:
            case BoundaryType::RigidWall: {
                // Full velocity reversal
                for (int d = 0; d < 3; ++d) {
                    ghost_vel[d] = -fluid_vel[d];
                }
                break;
            }
            case BoundaryType::FreeSlip: {
                // Only reverse the normal component
                Real v_dot_n = fluid_vel[0] * normal_[0] +
                               fluid_vel[1] * normal_[1] +
                               fluid_vel[2] * normal_[2];
                for (int d = 0; d < 3; ++d) {
                    ghost_vel[d] = fluid_vel[d] - 2.0 * v_dot_n * normal_[d];
                }
                break;
            }
            case BoundaryType::Porous: {
                // Blend between no-slip reflection and pass-through
                Real v_reflected[3];
                for (int d = 0; d < 3; ++d) {
                    v_reflected[d] = -fluid_vel[d];
                }
                for (int d = 0; d < 3; ++d) {
                    ghost_vel[d] = (1.0 - porosity_) * v_reflected[d] +
                                   porosity_ * fluid_vel[d];
                }
                break;
            }
        }
    }

    /**
     * @brief Compute signed distance from a point to the wall plane
     *
     * Positive distance = fluid side (same side as normal direction)
     * Negative distance = wall/solid side
     *
     * @param point Position to evaluate [3]
     * @return Signed distance (positive = fluid side)
     */
    KOKKOS_INLINE_FUNCTION
    Real signed_distance(const Real point[3]) const {
        return point[0] * normal_[0] +
               point[1] * normal_[1] +
               point[2] * normal_[2] - d_;
    }

    /**
     * @brief Check if a point is near the boundary (within threshold)
     *
     * @param point Position to check [3]
     * @param threshold Distance threshold
     * @return true if |signed_distance| <= threshold
     */
    KOKKOS_INLINE_FUNCTION
    bool is_near_boundary(const Real point[3], Real threshold) const {
        Real sd = signed_distance(point);
        return std::abs(sd) <= threshold;
    }

    // Getters
    BoundaryType type() const { return type_; }
    Real distance() const { return d_; }
    Real porosity() const { return porosity_; }
    const Real* normal() const { return normal_; }
    const char* name() const { return "AdvancedSPHBoundary"; }

    // Setters
    void set_porosity(Real p) { porosity_ = p; }
    void set_type(BoundaryType t) { type_ = t; }

private:
    BoundaryType type_;   ///< Boundary condition type
    Real normal_[3];      ///< Outward unit normal (into fluid)
    Real d_;              ///< Wall plane distance from origin (n.x = d)
    Real porosity_;       ///< Porosity factor [0,1] (Porous type only)
};


// ============================================================================
// 5. SPHStressPoints — Dual Particle Method for Tensile Instability
// ============================================================================

/**
 * @brief Dual particle method using stress points to avoid tensile instability
 *
 * The standard SPH method suffers from tensile instability when particles
 * are in tension. The stress-point method alleviates this by introducing
 * a second set of points (stress points) located at midpoints between
 * velocity particles. Stress is computed at these stress points and
 * interpolated back to velocity points for the momentum equation.
 *
 * This dual-point approach:
 * - Eliminates the rank-deficiency that causes tensile instability
 * - Improves stress field smoothness
 * - Maintains conservation properties
 *
 * Stress point placement: x_sp = (x_i + x_j) / 2 for selected neighbor pairs.
 * Each stress point stores:
 * - Position (midpoint of parent pair)
 * - Stress tensor (computed from local deformation)
 * - Parent pair indices for updates
 *
 * Reference: Dyka & Ingel (1995), An approach for tension instability;
 *            Randles & Libersky (2000), Normalized SPH with stress points
 */

/// Stress point parent pair information
struct StressPointPair {
    int particle_a;    ///< First parent particle index
    int particle_b;    ///< Second parent particle index
};

class SPHStressPoints {
public:
    /**
     * @brief Default constructor
     */
    SPHStressPoints() = default;

    /**
     * @brief Generate stress points from particle positions and neighbor lists
     *
     * For each particle, creates a stress point at the midpoint between
     * the particle and each of its neighbors. To avoid duplicates (since
     * if i neighbors j, then j neighbors i), only pairs where i < j are
     * created.
     *
     * @param particle_pos Particle positions [num_particles][3]
     * @param neighbor_lists Neighbor indices for each particle
     * @param num_particles Number of velocity particles
     * @param sp_pos Output stress point positions
     * @param sp_pairs Output parent pair information
     */
    void generate_stress_points(const std::vector<std::array<Real, 3>>& particle_pos,
                                const std::vector<std::vector<int>>& neighbor_lists,
                                int num_particles,
                                std::vector<std::array<Real, 3>>& sp_pos,
                                std::vector<StressPointPair>& sp_pairs) const {
        sp_pos.clear();
        sp_pairs.clear();

        for (int i = 0; i < num_particles; ++i) {
            for (int j_idx : neighbor_lists[i]) {
                // Only create stress point for i < j to avoid duplicates
                if (j_idx <= i) continue;
                if (j_idx >= num_particles) continue;

                // Midpoint
                std::array<Real, 3> mid;
                mid[0] = 0.5 * (particle_pos[i][0] + particle_pos[j_idx][0]);
                mid[1] = 0.5 * (particle_pos[i][1] + particle_pos[j_idx][1]);
                mid[2] = 0.5 * (particle_pos[i][2] + particle_pos[j_idx][2]);

                sp_pos.push_back(mid);
                sp_pairs.push_back({i, j_idx});
            }
        }
    }

    /**
     * @brief Compute stress at a stress point from surrounding particles
     *
     * Simple SPH interpolation of stress from nearby particles.
     * Uses a uniform-weight average for simplicity (production code
     * would use kernel-weighted interpolation).
     *
     * @param pos Stress point position [3]
     * @param neighbor_stresses Stress tensors of nearby particles [N][6]
     * @param num_neighbors Number of contributing particles
     * @param stress_out Output stress tensor [6] (Voigt: xx,yy,zz,xy,yz,xz)
     */
    KOKKOS_INLINE_FUNCTION
    void compute_stress_at_point(const Real pos[3],
                                 const Real neighbor_stresses[][6],
                                 int num_neighbors,
                                 Real stress_out[6]) const {
        for (int c = 0; c < 6; ++c) stress_out[c] = 0.0;

        if (num_neighbors <= 0) return;

        // Uniform-weight average of neighbor stresses
        for (int n = 0; n < num_neighbors; ++n) {
            for (int c = 0; c < 6; ++c) {
                stress_out[c] += neighbor_stresses[n][c];
            }
        }

        Real inv_n = 1.0 / static_cast<Real>(num_neighbors);
        for (int c = 0; c < 6; ++c) {
            stress_out[c] *= inv_n;
        }
    }

    /**
     * @brief Interpolate stress from stress points back to a velocity point
     *
     * Averages the stress values from all stress points associated with
     * this velocity particle (i.e., stress points where this particle
     * is one of the parents).
     *
     * @param particle_id ID of the velocity particle
     * @param sp_pairs Array of stress point parent pairs
     * @param sp_stress Array of stress tensors at stress points [N][6]
     * @param num_sp Total number of stress points
     * @param stress_out Output interpolated stress [6]
     */
    void interpolate_to_velocity_point(int particle_id,
                                       const std::vector<StressPointPair>& sp_pairs,
                                       const std::vector<std::array<Real, 6>>& sp_stress,
                                       int num_sp,
                                       Real stress_out[6]) const {
        for (int c = 0; c < 6; ++c) stress_out[c] = 0.0;

        int count = 0;
        for (int s = 0; s < num_sp; ++s) {
            if (sp_pairs[s].particle_a == particle_id ||
                sp_pairs[s].particle_b == particle_id) {
                for (int c = 0; c < 6; ++c) {
                    stress_out[c] += sp_stress[s][c];
                }
                ++count;
            }
        }

        if (count > 0) {
            Real inv = 1.0 / static_cast<Real>(count);
            for (int c = 0; c < 6; ++c) {
                stress_out[c] *= inv;
            }
        }
    }

    /**
     * @brief Update stress point positions after velocity points have moved
     *
     * Recomputes each stress point position as the midpoint of its parent
     * pair's current positions.
     *
     * @param particle_pos Current velocity particle positions
     * @param sp_pairs Parent pair information
     * @param sp_pos Stress point positions to update (modified in place)
     * @param num_sp Number of stress points
     */
    void update_stress_point_positions(
            const std::vector<std::array<Real, 3>>& particle_pos,
            const std::vector<StressPointPair>& sp_pairs,
            std::vector<std::array<Real, 3>>& sp_pos,
            int num_sp) const {
        for (int s = 0; s < num_sp; ++s) {
            int a = sp_pairs[s].particle_a;
            int b = sp_pairs[s].particle_b;
            sp_pos[s][0] = 0.5 * (particle_pos[a][0] + particle_pos[b][0]);
            sp_pos[s][1] = 0.5 * (particle_pos[a][1] + particle_pos[b][1]);
            sp_pos[s][2] = 0.5 * (particle_pos[a][2] + particle_pos[b][2]);
        }
    }

    /**
     * @brief Count stress points associated with a given velocity particle
     *
     * @param particle_id Velocity particle index
     * @param sp_pairs Parent pair information
     * @param num_sp Total number of stress points
     * @return Number of stress points connected to this particle
     */
    int count_stress_points_for_particle(int particle_id,
                                         const std::vector<StressPointPair>& sp_pairs,
                                         int num_sp) const {
        int count = 0;
        for (int s = 0; s < num_sp; ++s) {
            if (sp_pairs[s].particle_a == particle_id ||
                sp_pairs[s].particle_b == particle_id) {
                ++count;
            }
        }
        return count;
    }

    const char* name() const { return "SPHStressPoints"; }
};


} // namespace sph
} // namespace nxs
