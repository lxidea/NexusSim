#pragma once

/**
 * @file pd_neighbor.hpp
 * @brief Peridynamics neighbor list with Kokkos support
 *
 * Ported from PeriSys-Haoran JBuildNeighborList.cu
 * Uses CSR format for efficient GPU access
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
 * @param distance Bond length |xi|
 * @param horizon Horizon delta
 * @param horizon_factor Factor for influence function
 */
KOKKOS_INLINE_FUNCTION
Real influence_weight(Real distance, Real horizon, Real horizon_factor = 3.015) {
    Real r = distance / horizon;
    if (r > 1.0) return 0.0;

    // Conical influence function (from PeriSys)
    // w = (1 - r)^2 * exp(-r * horizon_factor)
    Real one_minus_r = 1.0 - r;
    return one_minus_r * one_minus_r * std::exp(-r * horizon_factor);
}

/**
 * @brief Neighbor list for peridynamics (CSR format)
 *
 * Stores neighbor relationships in compressed sparse row format
 * for efficient GPU access.
 */
class PDNeighborList {
public:
    PDNeighborList() = default;

    /**
     * @brief Build neighbor list from particle positions
     *
     * For each particle, finds all neighbors within the horizon.
     * Uses O(N²) algorithm suitable for moderate particle counts.
     *
     * @param particles Particle system
     * @param horizon_factor Factor for influence function (default 3.015)
     */
    void build(PDParticleSystem& particles, Real horizon_factor = 3.015) {
        Index num_particles = particles.num_particles();

        // Sync to host for building
        particles.sync_to_host();

        auto x0_host = particles.x0_host();
        auto horizon_host = particles.horizon_host();
        auto body_id = particles.body_id();
        auto material_id = particles.material_id();
        auto active = particles.active();

        // First pass: count neighbors
        std::vector<Index> neighbor_count(num_particles, 0);
        std::vector<std::vector<Index>> neighbor_lists(num_particles);
        std::vector<std::vector<Real>> weight_lists(num_particles);

        // Reserve space (typical PD has ~100-250 neighbors)
        for (Index i = 0; i < num_particles; ++i) {
            neighbor_lists[i].reserve(250);
            weight_lists[i].reserve(250);
        }

        // Build neighbor list (O(N²) for now, TODO: spatial hashing)
        for (Index i = 0; i < num_particles - 1; ++i) {
            if (!active(i)) continue;

            Real xi[3] = {x0_host(i, 0), x0_host(i, 1), x0_host(i, 2)};
            Real delta_i = horizon_host(i);

            for (Index j = i + 1; j < num_particles; ++j) {
                if (!active(j)) continue;

                // Skip if different bodies (from PeriSys)
                if (body_id(i) != body_id(j)) continue;

                // Skip if different material types
                if (material_id(i) != material_id(j)) continue;

                Real xj[3] = {x0_host(j, 0), x0_host(j, 1), x0_host(j, 2)};
                Real delta_j = horizon_host(j);

                // Average horizon
                Real delta_ij = 0.5 * (delta_i + delta_j);

                // Distance
                Real dx = xj[0] - xi[0];
                Real dy = xj[1] - xi[1];
                Real dz = xj[2] - xi[2];
                Real distance = std::sqrt(dx * dx + dy * dy + dz * dz);

                // Check if within horizon
                Real r = distance / delta_ij;
                if (r <= 1.0) {
                    Real w = influence_weight(distance, delta_ij, horizon_factor);

                    // Add symmetric bond (i -> j and j -> i)
                    neighbor_lists[i].push_back(j);
                    neighbor_lists[j].push_back(i);
                    weight_lists[i].push_back(w);
                    weight_lists[j].push_back(w);
                    neighbor_count[i]++;
                    neighbor_count[j]++;
                }
            }
        }

        // Compute total neighbors and offsets
        total_bonds_ = 0;
        std::vector<Index> offsets(num_particles);
        for (Index i = 0; i < num_particles; ++i) {
            offsets[i] = total_bonds_;
            total_bonds_ += neighbor_count[i];
        }

        // Allocate device views
        neighbor_offset_ = PDNeighborOffsetView("neighbor_offset", num_particles + 1);
        neighbor_list_ = PDNeighborListView("neighbor_list", total_bonds_);
        neighbor_count_ = PDIndexView("neighbor_count", num_particles);
        bond_weight_ = PDBondWeightView("bond_weight", total_bonds_);
        bond_intact_ = PDBondIntactView("bond_intact", total_bonds_);
        bond_xi_ = Kokkos::View<Real*[3]>("bond_xi", total_bonds_);
        bond_length_ = PDScalarView("bond_length", total_bonds_);

        // Create host mirrors
        auto neighbor_offset_host = Kokkos::create_mirror_view(neighbor_offset_);
        auto neighbor_list_host = Kokkos::create_mirror_view(neighbor_list_);
        auto neighbor_count_host = Kokkos::create_mirror_view(neighbor_count_);
        auto bond_weight_host = Kokkos::create_mirror_view(bond_weight_);
        auto bond_intact_host = Kokkos::create_mirror_view(bond_intact_);
        auto bond_xi_host = Kokkos::create_mirror_view(bond_xi_);
        auto bond_length_host = Kokkos::create_mirror_view(bond_length_);

        // Fill host arrays
        for (Index i = 0; i < num_particles; ++i) {
            neighbor_offset_host(i) = offsets[i];
            neighbor_count_host(i) = neighbor_count[i];

            for (Index k = 0; k < neighbor_count[i]; ++k) {
                Index bond_idx = offsets[i] + k;
                Index j = neighbor_lists[i][k];

                neighbor_list_host(bond_idx) = j;
                bond_weight_host(bond_idx) = weight_lists[i][k];
                bond_intact_host(bond_idx) = true;

                // Reference bond vector xi = xj - xi
                Real xi_vec[3] = {
                    x0_host(j, 0) - x0_host(i, 0),
                    x0_host(j, 1) - x0_host(i, 1),
                    x0_host(j, 2) - x0_host(i, 2)
                };
                Real length = std::sqrt(
                    xi_vec[0] * xi_vec[0] +
                    xi_vec[1] * xi_vec[1] +
                    xi_vec[2] * xi_vec[2]
                );

                bond_xi_host(bond_idx, 0) = xi_vec[0];
                bond_xi_host(bond_idx, 1) = xi_vec[1];
                bond_xi_host(bond_idx, 2) = xi_vec[2];
                bond_length_host(bond_idx) = length;
            }
        }
        neighbor_offset_host(num_particles) = total_bonds_;

        // Copy to device
        Kokkos::deep_copy(neighbor_offset_, neighbor_offset_host);
        Kokkos::deep_copy(neighbor_list_, neighbor_list_host);
        Kokkos::deep_copy(neighbor_count_, neighbor_count_host);
        Kokkos::deep_copy(bond_weight_, bond_weight_host);
        Kokkos::deep_copy(bond_intact_, bond_intact_host);
        Kokkos::deep_copy(bond_xi_, bond_xi_host);
        Kokkos::deep_copy(bond_length_, bond_length_host);

        // Compute statistics
        Index max_neighbors = 0;
        Real avg_neighbors = 0.0;
        Index active_count = 0;
        for (Index i = 0; i < num_particles; ++i) {
            if (active(i) && neighbor_count[i] > 0) {
                max_neighbors = std::max(max_neighbors, neighbor_count[i]);
                avg_neighbors += neighbor_count[i];
                active_count++;
            }
        }
        if (active_count > 0) {
            avg_neighbors /= active_count;
        }

        NXS_LOG_INFO("PDNeighborList: {} particles, {} bonds, avg neighbors: {:.1f}, max: {}",
                     num_particles, total_bonds_, avg_neighbors, max_neighbors);
    }

    /**
     * @brief Mark a bond as broken
     */
    void break_bond(Index bond_idx) {
        Kokkos::parallel_for("break_bond", 1,
            KOKKOS_LAMBDA(const Index) {
                bond_intact_(bond_idx) = false;
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
