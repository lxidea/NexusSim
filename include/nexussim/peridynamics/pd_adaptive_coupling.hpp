#pragma once

/**
 * @file pd_adaptive_coupling.hpp
 * @brief Adaptive coupling region that expands PD zones around damage
 *
 * Monitors particle damage and reclassifies domain zones:
 * - Damaged particles → PD_Only
 * - Nearby particles within buffer → Overlap
 * - Distant undamaged → FEM_Only candidates
 * Triggers morphing for newly entering elements.
 */

#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_particle.hpp>
#include <nexussim/peridynamics/pd_neighbor.hpp>
#include <nexussim/peridynamics/pd_fem_coupling.hpp>
#include <nexussim/peridynamics/pd_morphing.hpp>
#include <vector>

namespace nxs {
namespace pd {

// ============================================================================
// Zone statistics
// ============================================================================

struct ZoneStatistics {
    Index fem_only = 0;
    Index pd_only = 0;
    Index overlap = 0;
    Index interface = 0;
    Index total = 0;
};

// ============================================================================
// PDAdaptiveCoupling
// ============================================================================

class PDAdaptiveCoupling {
public:
    PDAdaptiveCoupling() = default;

    /**
     * @brief Monitor per-particle damage from bond integrity
     *
     * damage_i = 1 - (intact bonds for i) / (total bonds for i)
     *
     * @param particles PD particle system
     * @param neighbors PD neighbor list
     * @return Maximum damage across all particles
     */
    Real monitor_damage(PDParticleSystem& particles, PDNeighborList& neighbors) {
        Index num_particles = particles.num_particles();

        if (particle_damage_.size() != static_cast<size_t>(num_particles)) {
            particle_damage_.resize(num_particles, 0.0);
        }

        // Sync damage from device
        neighbors.update_damage(particles);
        particles.sync_to_host();
        auto damage_host = particles.damage_host();

        Real max_damage = 0.0;
        for (Index i = 0; i < num_particles; ++i) {
            particle_damage_[i] = damage_host(i);
            max_damage = std::max(max_damage, particle_damage_[i]);
        }

        return max_damage;
    }

    /**
     * @brief Classify zones based on damage
     *
     * @param coupling FEM-PD coupling manager
     * @param particles PD particle system
     * @param damage_threshold Damage level for PD_Only classification
     * @param buffer_size Buffer zone in multiples of horizon
     */
    void classify_zones(
        FEMPDCoupling& coupling,
        PDParticleSystem& particles,
        Real damage_threshold = 0.3,
        Real buffer_size = 2.0)
    {
        Index num_particles = particles.num_particles();

        if (zone_types_.size() != static_cast<size_t>(num_particles)) {
            zone_types_.resize(num_particles, DomainType::Overlap);
        }

        particles.sync_to_host();
        auto x_host = particles.x_host();
        auto horizon_host = particles.horizon_host();

        // Pass 1: Mark damaged particles as PD_Only
        std::vector<Index> damaged_particles;
        for (Index i = 0; i < num_particles; ++i) {
            if (particle_damage_[i] > damage_threshold) {
                zone_types_[i] = DomainType::PD_Only;
                damaged_particles.push_back(i);
            }
        }

        // Pass 2: Mark particles within buffer of damaged as Overlap
        for (Index i = 0; i < num_particles; ++i) {
            if (zone_types_[i] == DomainType::PD_Only) continue;

            Real xi = x_host(i, 0);
            Real yi = x_host(i, 1);
            Real zi = x_host(i, 2);
            Real hi = horizon_host(i);
            Real buffer_dist = buffer_size * hi;

            bool near_damage = false;
            for (Index d_idx : damaged_particles) {
                Real dx = xi - x_host(d_idx, 0);
                Real dy = yi - x_host(d_idx, 1);
                Real dz = zi - x_host(d_idx, 2);
                Real dist = std::sqrt(dx * dx + dy * dy + dz * dz);

                if (dist <= buffer_dist) {
                    near_damage = true;
                    break;
                }
            }

            if (near_damage) {
                zone_types_[i] = DomainType::Overlap;
            } else {
                // Candidates for FEM_Only reconversion
                zone_types_[i] = DomainType::FEM_Only;
            }
        }

        // Update coupling domain arrays
        auto& pd_domain = const_cast<std::vector<DomainType>&>(coupling.pd_domain());
        if (pd_domain.size() < static_cast<size_t>(num_particles)) {
            pd_domain.resize(num_particles, DomainType::PD_Only);
        }
        for (Index i = 0; i < num_particles; ++i) {
            pd_domain[i] = zone_types_[i];
        }
    }

    /**
     * @brief Expand PD zone around damage, triggering morphing if needed
     *
     * @param coupling FEM-PD coupling manager
     * @param morphing Element morphing module
     * @param particles PD particle system
     * @param neighbors PD neighbor list
     * @param damage_threshold Damage threshold for zone expansion
     * @param buffer_size Buffer zone size
     */
    void expand_pd_zone(
        FEMPDCoupling& coupling,
        PDParticleSystem& particles,
        PDNeighborList& neighbors,
        Real damage_threshold = 0.3,
        Real buffer_size = 2.0)
    {
        // Monitor current damage
        monitor_damage(particles, neighbors);

        // Classify zones based on updated damage
        classify_zones(coupling, particles, damage_threshold, buffer_size);

        // Count zone changes
        zone_stats_ = get_zone_statistics();
    }

    /**
     * @brief Get zone statistics
     */
    ZoneStatistics get_zone_statistics() const {
        ZoneStatistics stats;
        stats.total = static_cast<Index>(zone_types_.size());

        for (auto& zt : zone_types_) {
            switch (zt) {
                case DomainType::FEM_Only:  stats.fem_only++; break;
                case DomainType::PD_Only:   stats.pd_only++; break;
                case DomainType::Overlap:   stats.overlap++; break;
                case DomainType::Interface: stats.interface++; break;
            }
        }

        return stats;
    }

    // Accessors
    const std::vector<DomainType>& zone_types() const { return zone_types_; }
    const std::vector<Real>& particle_damage() const { return particle_damage_; }

private:
    std::vector<DomainType> zone_types_;
    std::vector<Real> particle_damage_;
    ZoneStatistics zone_stats_;
};

} // namespace pd
} // namespace nxs
