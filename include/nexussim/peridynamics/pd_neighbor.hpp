#pragma once

/**
 * @file pd_neighbor.hpp
 * @brief Peridynamics neighbor list with Kokkos GPU support
 *
 * Uses spatial hashing for O(N) neighbor search on GPU.
 * CSR format for efficient GPU access.
 */

#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_particle.hpp>
#include <vector>
#include <cmath>

namespace nxs {
namespace pd {

/**
 * @brief Influence function types for bond weighting
 */
enum class InfluenceFunction {
    Constant,       ///< w(xi) = 1
    Linear,         ///< w(xi) = 1 - |xi|/delta
    Gaussian,       ///< w(xi) = exp(-|xi|^2 / delta^2)
    Conical         ///< w(xi) = (1 - |xi|/delta)^2
};

/**
 * @brief Compute influence function weight
 */
KOKKOS_INLINE_FUNCTION
Real influence_weight(Real distance, Real horizon, Real horizon_factor = 3.015) {
    Real r = distance / horizon;
    if (r > 1.0) return 0.0;

    // Conical influence function (from PeriSys)
    Real one_minus_r = 1.0 - r;
    return one_minus_r * one_minus_r * Kokkos::exp(-r * horizon_factor);
}

/**
 * @brief Neighbor list for peridynamics (CSR format)
 *
 * Uses GPU-accelerated spatial hashing for O(N) neighbor search.
 */
class PDNeighborList {
public:
    PDNeighborList() = default;

    /**
     * @brief Build neighbor list using GPU spatial hashing
     *
     * Algorithm:
     * 1. Compute bounding box and cell grid
     * 2. Assign particles to cells (GPU parallel)
     * 3. Build cell lists with atomic operations
     * 4. Count neighbors per particle (GPU parallel, check 27 cells)
     * 5. Prefix sum for CSR offsets
     * 6. Fill neighbor list (GPU parallel)
     *
     * @param particles Particle system
     * @param horizon_factor Factor for influence function (default 3.015)
     */
    void build(PDParticleSystem& particles, Real horizon_factor = 3.015) {
        Index num_particles = particles.num_particles();

        // Get particle data on device
        auto x0 = particles.x0();
        auto horizon = particles.horizon();
        auto body_id = particles.body_id();
        auto material_id = particles.material_id();
        auto active = particles.active();

        // Step 1: Find bounding box and max horizon (reduction on GPU)
        Real min_x, min_y, min_z, max_x, max_y, max_z, max_horizon;

        Kokkos::parallel_reduce("find_bounds", num_particles,
            KOKKOS_LAMBDA(const Index i, Real& lmin_x, Real& lmin_y, Real& lmin_z,
                          Real& lmax_x, Real& lmax_y, Real& lmax_z, Real& lmax_h) {
                if (active(i)) {
                    lmin_x = Kokkos::fmin(lmin_x, x0(i, 0));
                    lmin_y = Kokkos::fmin(lmin_y, x0(i, 1));
                    lmin_z = Kokkos::fmin(lmin_z, x0(i, 2));
                    lmax_x = Kokkos::fmax(lmax_x, x0(i, 0));
                    lmax_y = Kokkos::fmax(lmax_y, x0(i, 1));
                    lmax_z = Kokkos::fmax(lmax_z, x0(i, 2));
                    lmax_h = Kokkos::fmax(lmax_h, horizon(i));
                }
            },
            Kokkos::Min<Real>(min_x), Kokkos::Min<Real>(min_y), Kokkos::Min<Real>(min_z),
            Kokkos::Max<Real>(max_x), Kokkos::Max<Real>(max_y), Kokkos::Max<Real>(max_z),
            Kokkos::Max<Real>(max_horizon));

        // Cell size = max horizon (ensures all neighbors are in adjacent cells)
        Real cell_size = max_horizon * 1.001;  // Small margin for floating point

        // Grid dimensions
        Index nx = static_cast<Index>((max_x - min_x) / cell_size) + 3;
        Index ny = static_cast<Index>((max_y - min_y) / cell_size) + 3;
        Index nz = static_cast<Index>((max_z - min_z) / cell_size) + 3;
        Index num_cells = nx * ny * nz;

        // Adjust grid origin to include margin
        Real grid_min_x = min_x - cell_size;
        Real grid_min_y = min_y - cell_size;
        Real grid_min_z = min_z - cell_size;

        // Step 2: Count particles per cell
        Kokkos::View<Index*> cell_count("cell_count", num_cells);
        Kokkos::View<Index*> particle_cell("particle_cell", num_particles);

        Kokkos::parallel_for("assign_cells", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i)) {
                    particle_cell(i) = num_cells;  // Invalid cell
                    return;
                }

                Index cx = static_cast<Index>((x0(i, 0) - grid_min_x) / cell_size);
                Index cy = static_cast<Index>((x0(i, 1) - grid_min_y) / cell_size);
                Index cz = static_cast<Index>((x0(i, 2) - grid_min_z) / cell_size);

                cx = Kokkos::max(Index(0), Kokkos::min(cx, nx - 1));
                cy = Kokkos::max(Index(0), Kokkos::min(cy, ny - 1));
                cz = Kokkos::max(Index(0), Kokkos::min(cz, nz - 1));

                Index cell_idx = cx + cy * nx + cz * nx * ny;
                particle_cell(i) = cell_idx;
                Kokkos::atomic_increment(&cell_count(cell_idx));
            });
        Kokkos::fence();

        // Step 3: Prefix sum for cell offsets
        Kokkos::View<Index*> cell_offset("cell_offset", num_cells + 1);
        Kokkos::parallel_scan("cell_prefix_sum", num_cells,
            KOKKOS_LAMBDA(const Index i, Index& sum, const bool final) {
                if (final) {
                    cell_offset(i) = sum;
                }
                sum += cell_count(i);
            });

        // Set final offset
        Index total_in_cells = 0;
        Kokkos::parallel_reduce("sum_cells", num_cells,
            KOKKOS_LAMBDA(const Index i, Index& sum) {
                sum += cell_count(i);
            }, total_in_cells);

        auto cell_offset_host = Kokkos::create_mirror_view(cell_offset);
        Kokkos::deep_copy(cell_offset_host, cell_offset);
        cell_offset_host(num_cells) = total_in_cells;
        Kokkos::deep_copy(cell_offset, cell_offset_host);

        // Step 4: Fill cell lists (sorted particle indices)
        Kokkos::View<Index*> cell_list("cell_list", num_particles);
        Kokkos::View<Index*> cell_fill("cell_fill", num_cells);  // Current fill position

        Kokkos::parallel_for("fill_cell_list", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i)) return;

                Index cell_idx = particle_cell(i);
                if (cell_idx >= num_cells) return;

                Index pos = cell_offset(cell_idx) + Kokkos::atomic_fetch_add(&cell_fill(cell_idx), 1);
                cell_list(pos) = i;
            });
        Kokkos::fence();

        // Step 5: Count neighbors per particle (checking 27 neighboring cells)
        neighbor_count_ = PDIndexView("neighbor_count", num_particles);
        auto neighbor_count = neighbor_count_;

        Kokkos::parallel_for("count_neighbors", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i)) {
                    neighbor_count(i) = 0;
                    return;
                }

                Real xi[3] = {x0(i, 0), x0(i, 1), x0(i, 2)};
                Real delta_i = horizon(i);
                Index body_i = body_id(i);
                Index mat_i = material_id(i);

                Index cx = static_cast<Index>((xi[0] - grid_min_x) / cell_size);
                Index cy = static_cast<Index>((xi[1] - grid_min_y) / cell_size);
                Index cz = static_cast<Index>((xi[2] - grid_min_z) / cell_size);

                Index count = 0;

                // Check 27 neighboring cells
                for (int dz = -1; dz <= 1; ++dz) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            Index ncx = cx + dx;
                            Index ncy = cy + dy;
                            Index ncz = cz + dz;

                            if (ncx < 0 || ncx >= nx || ncy < 0 || ncy >= ny || ncz < 0 || ncz >= nz)
                                continue;

                            Index neighbor_cell = ncx + ncy * nx + ncz * nx * ny;
                            Index cell_start = cell_offset(neighbor_cell);
                            Index cell_end = cell_offset(neighbor_cell + 1);

                            for (Index k = cell_start; k < cell_end; ++k) {
                                Index j = cell_list(k);
                                if (j <= i) continue;  // Only count once (i < j)
                                if (!active(j)) continue;
                                if (body_id(j) != body_i) continue;
                                if (material_id(j) != mat_i) continue;

                                Real xj[3] = {x0(j, 0), x0(j, 1), x0(j, 2)};
                                Real delta_j = horizon(j);
                                Real delta_ij = 0.5 * (delta_i + delta_j);

                                Real dx = xj[0] - xi[0];
                                Real dy = xj[1] - xi[1];
                                Real dz = xj[2] - xi[2];
                                Real dist = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);

                                if (dist <= delta_ij) {
                                    count++;
                                }
                            }
                        }
                    }
                }
                neighbor_count(i) = count;
            });
        Kokkos::fence();

        // Also count reverse bonds (j -> i where j < i)
        Kokkos::parallel_for("count_reverse_neighbors", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i)) return;

                Real xi[3] = {x0(i, 0), x0(i, 1), x0(i, 2)};
                Real delta_i = horizon(i);
                Index body_i = body_id(i);
                Index mat_i = material_id(i);

                Index cx = static_cast<Index>((xi[0] - grid_min_x) / cell_size);
                Index cy = static_cast<Index>((xi[1] - grid_min_y) / cell_size);
                Index cz = static_cast<Index>((xi[2] - grid_min_z) / cell_size);

                for (int dz = -1; dz <= 1; ++dz) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            Index ncx = cx + dx;
                            Index ncy = cy + dy;
                            Index ncz = cz + dz;

                            if (ncx < 0 || ncx >= nx || ncy < 0 || ncy >= ny || ncz < 0 || ncz >= nz)
                                continue;

                            Index neighbor_cell = ncx + ncy * nx + ncz * nx * ny;
                            Index cell_start = cell_offset(neighbor_cell);
                            Index cell_end = cell_offset(neighbor_cell + 1);

                            for (Index k = cell_start; k < cell_end; ++k) {
                                Index j = cell_list(k);
                                if (j >= i) continue;  // Only j < i for reverse
                                if (!active(j)) continue;
                                if (body_id(j) != body_i) continue;
                                if (material_id(j) != mat_i) continue;

                                Real xj[3] = {x0(j, 0), x0(j, 1), x0(j, 2)};
                                Real delta_j = horizon(j);
                                Real delta_ij = 0.5 * (delta_i + delta_j);

                                Real ddx = xj[0] - xi[0];
                                Real ddy = xj[1] - xi[1];
                                Real ddz = xj[2] - xi[2];
                                Real dist = Kokkos::sqrt(ddx*ddx + ddy*ddy + ddz*ddz);

                                if (dist <= delta_ij) {
                                    Kokkos::atomic_increment(&neighbor_count(i));
                                }
                            }
                        }
                    }
                }
            });
        Kokkos::fence();

        // Step 6: Prefix sum for neighbor offsets
        neighbor_offset_ = PDNeighborOffsetView("neighbor_offset", num_particles + 1);
        auto neighbor_offset = neighbor_offset_;

        Kokkos::parallel_scan("neighbor_prefix_sum", num_particles,
            KOKKOS_LAMBDA(const Index i, Index& sum, const bool final) {
                if (final) {
                    neighbor_offset(i) = sum;
                }
                sum += neighbor_count(i);
            });

        // Get total bonds
        Kokkos::parallel_reduce("sum_bonds", num_particles,
            KOKKOS_LAMBDA(const Index i, Index& sum) {
                sum += neighbor_count(i);
            }, total_bonds_);

        // Set final offset
        auto neighbor_offset_host = Kokkos::create_mirror_view(neighbor_offset);
        Kokkos::deep_copy(neighbor_offset_host, neighbor_offset);
        neighbor_offset_host(num_particles) = total_bonds_;
        Kokkos::deep_copy(neighbor_offset, neighbor_offset_host);

        // Step 7: Allocate neighbor data
        neighbor_list_ = PDNeighborListView("neighbor_list", total_bonds_);
        bond_weight_ = PDBondWeightView("bond_weight", total_bonds_);
        bond_intact_ = PDBondIntactView("bond_intact", total_bonds_);
        bond_xi_ = Kokkos::View<Real*[3]>("bond_xi", total_bonds_);
        bond_length_ = PDScalarView("bond_length", total_bonds_);

        auto neighbor_list = neighbor_list_;
        auto bond_weight = bond_weight_;
        auto bond_intact = bond_intact_;
        auto bond_xi = bond_xi_;
        auto bond_length = bond_length_;

        // Step 8: Fill neighbor list (forward bonds: i < j)
        Kokkos::View<Index*> fill_pos("fill_pos", num_particles);

        Kokkos::parallel_for("fill_forward_neighbors", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i)) return;

                Real xi[3] = {x0(i, 0), x0(i, 1), x0(i, 2)};
                Real delta_i = horizon(i);
                Index body_i = body_id(i);
                Index mat_i = material_id(i);

                Index cx = static_cast<Index>((xi[0] - grid_min_x) / cell_size);
                Index cy = static_cast<Index>((xi[1] - grid_min_y) / cell_size);
                Index cz = static_cast<Index>((xi[2] - grid_min_z) / cell_size);

                Index offset = neighbor_offset(i);
                Index local_count = 0;

                for (int ddz = -1; ddz <= 1; ++ddz) {
                    for (int ddy = -1; ddy <= 1; ++ddy) {
                        for (int ddx = -1; ddx <= 1; ++ddx) {
                            Index ncx = cx + ddx;
                            Index ncy = cy + ddy;
                            Index ncz = cz + ddz;

                            if (ncx < 0 || ncx >= nx || ncy < 0 || ncy >= ny || ncz < 0 || ncz >= nz)
                                continue;

                            Index neighbor_cell = ncx + ncy * nx + ncz * nx * ny;
                            Index cell_start = cell_offset(neighbor_cell);
                            Index cell_end = cell_offset(neighbor_cell + 1);

                            for (Index k = cell_start; k < cell_end; ++k) {
                                Index j = cell_list(k);
                                if (j <= i) continue;
                                if (!active(j)) continue;
                                if (body_id(j) != body_i) continue;
                                if (material_id(j) != mat_i) continue;

                                Real xj[3] = {x0(j, 0), x0(j, 1), x0(j, 2)};
                                Real delta_j = horizon(j);
                                Real delta_ij = 0.5 * (delta_i + delta_j);

                                Real dx = xj[0] - xi[0];
                                Real dy = xj[1] - xi[1];
                                Real dz = xj[2] - xi[2];
                                Real dist = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);

                                if (dist <= delta_ij) {
                                    Index bond_idx = offset + local_count;
                                    neighbor_list(bond_idx) = j;
                                    bond_weight(bond_idx) = influence_weight(dist, delta_ij, horizon_factor);
                                    bond_intact(bond_idx) = true;
                                    bond_xi(bond_idx, 0) = dx;
                                    bond_xi(bond_idx, 1) = dy;
                                    bond_xi(bond_idx, 2) = dz;
                                    bond_length(bond_idx) = dist;
                                    local_count++;
                                }
                            }
                        }
                    }
                }
                fill_pos(i) = local_count;
            });
        Kokkos::fence();

        // Step 9: Fill reverse bonds (j < i)
        Kokkos::parallel_for("fill_reverse_neighbors", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i)) return;

                Real xi[3] = {x0(i, 0), x0(i, 1), x0(i, 2)};
                Real delta_i = horizon(i);
                Index body_i = body_id(i);
                Index mat_i = material_id(i);

                Index cx = static_cast<Index>((xi[0] - grid_min_x) / cell_size);
                Index cy = static_cast<Index>((xi[1] - grid_min_y) / cell_size);
                Index cz = static_cast<Index>((xi[2] - grid_min_z) / cell_size);

                Index offset = neighbor_offset(i) + fill_pos(i);
                Index local_count = 0;

                for (int ddz = -1; ddz <= 1; ++ddz) {
                    for (int ddy = -1; ddy <= 1; ++ddy) {
                        for (int ddx = -1; ddx <= 1; ++ddx) {
                            Index ncx = cx + ddx;
                            Index ncy = cy + ddy;
                            Index ncz = cz + ddz;

                            if (ncx < 0 || ncx >= nx || ncy < 0 || ncy >= ny || ncz < 0 || ncz >= nz)
                                continue;

                            Index neighbor_cell = ncx + ncy * nx + ncz * nx * ny;
                            Index cell_start_idx = cell_offset(neighbor_cell);
                            Index cell_end_idx = cell_offset(neighbor_cell + 1);

                            for (Index k = cell_start_idx; k < cell_end_idx; ++k) {
                                Index j = cell_list(k);
                                if (j >= i) continue;
                                if (!active(j)) continue;
                                if (body_id(j) != body_i) continue;
                                if (material_id(j) != mat_i) continue;

                                Real xj[3] = {x0(j, 0), x0(j, 1), x0(j, 2)};
                                Real delta_j = horizon(j);
                                Real delta_ij = 0.5 * (delta_i + delta_j);

                                Real dx = xj[0] - xi[0];
                                Real dy = xj[1] - xi[1];
                                Real dz = xj[2] - xi[2];
                                Real dist = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);

                                if (dist <= delta_ij) {
                                    Index bond_idx = offset + local_count;
                                    neighbor_list(bond_idx) = j;
                                    bond_weight(bond_idx) = influence_weight(dist, delta_ij, horizon_factor);
                                    bond_intact(bond_idx) = true;
                                    bond_xi(bond_idx, 0) = dx;
                                    bond_xi(bond_idx, 1) = dy;
                                    bond_xi(bond_idx, 2) = dz;
                                    bond_length(bond_idx) = dist;
                                    local_count++;
                                }
                            }
                        }
                    }
                }
            });
        Kokkos::fence();

        // Compute statistics
        Index max_neighbors = 0;
        Real avg_neighbors = 0.0;

        Kokkos::parallel_reduce("neighbor_stats", num_particles,
            KOKKOS_LAMBDA(const Index i, Index& lmax, Real& lsum) {
                if (active(i)) {
                    lmax = Kokkos::max(lmax, neighbor_count(i));
                    lsum += neighbor_count(i);
                }
            },
            Kokkos::Max<Index>(max_neighbors), avg_neighbors);

        Index active_count = 0;
        Kokkos::parallel_reduce("count_active", num_particles,
            KOKKOS_LAMBDA(const Index i, Index& cnt) {
                if (active(i)) cnt++;
            }, active_count);

        if (active_count > 0) {
            avg_neighbors /= active_count;
        }

        NXS_LOG_INFO("PDNeighborList (GPU): {} particles, {} bonds, avg: {:.1f}, max: {}, cells: {}x{}x{}",
                     num_particles, total_bonds_, avg_neighbors, max_neighbors, nx, ny, nz);
    }

    /**
     * @brief Mark a bond as broken
     */
    void break_bond(Index bond_idx) {
        auto bond_intact = bond_intact_;
        Kokkos::parallel_for("break_bond", 1,
            KOKKOS_LAMBDA(const Index) {
                bond_intact(bond_idx) = false;
            });
    }

    /**
     * @brief Update damage for all particles based on broken bonds
     */
    void update_damage(PDParticleSystem& particles) {
        auto damage = particles.damage();
        auto neighbor_offset = neighbor_offset_;
        auto neighbor_count = neighbor_count_;
        auto bond_intact = bond_intact_;

        Kokkos::parallel_for("update_damage", particles.num_particles(),
            KOKKOS_LAMBDA(const Index i) {
                Index offset = neighbor_offset(i);
                Index count = neighbor_count(i);

                if (count == 0) {
                    damage(i) = 0.0;
                    return;
                }

                Index broken = 0;
                for (Index k = 0; k < count; ++k) {
                    if (!bond_intact(offset + k)) {
                        broken++;
                    }
                }

                damage(i) = static_cast<Real>(broken) / static_cast<Real>(count);
            });
    }

    /**
     * @brief Count total broken bonds
     */
    Index count_broken_bonds() const {
        Index broken_count = 0;
        auto bond_intact = bond_intact_;

        Kokkos::parallel_reduce("count_broken", total_bonds_,
            KOKKOS_LAMBDA(const Index i, Index& count) {
                if (!bond_intact(i)) {
                    count++;
                }
            }, broken_count);

        return broken_count;
    }

    // Accessors
    Index total_bonds() const { return total_bonds_; }

    PDNeighborOffsetView& neighbor_offset() { return neighbor_offset_; }
    PDNeighborListView& neighbor_list() { return neighbor_list_; }
    PDIndexView& neighbor_count() { return neighbor_count_; }
    PDBondWeightView& bond_weight() { return bond_weight_; }
    PDBondIntactView& bond_intact() { return bond_intact_; }
    Kokkos::View<Real*[3]>& bond_xi() { return bond_xi_; }
    PDScalarView& bond_length() { return bond_length_; }

private:
    Index total_bonds_ = 0;

    PDNeighborOffsetView neighbor_offset_;  ///< CSR row pointers [num_particles + 1]
    PDNeighborListView neighbor_list_;      ///< Neighbor indices [total_bonds]
    PDIndexView neighbor_count_;            ///< Neighbors per particle [num_particles]
    PDBondWeightView bond_weight_;          ///< Bond influence weights [total_bonds]
    PDBondIntactView bond_intact_;          ///< Bond intact status [total_bonds]
    Kokkos::View<Real*[3]> bond_xi_;        ///< Reference bond vectors [total_bonds][3]
    PDScalarView bond_length_;              ///< Reference bond lengths [total_bonds]
};

} // namespace pd
} // namespace nxs
