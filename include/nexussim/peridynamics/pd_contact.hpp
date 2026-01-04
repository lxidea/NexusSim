#pragma once

/**
 * @file pd_contact.hpp
 * @brief Peridynamics contact algorithm
 *
 * Implements short-range repulsive force contact for PD particles.
 * Based on PeriSys-Haoran JContact_force.cu
 *
 * Contact force model:
 *   f_c = k_c * δ * n  (when δ < 0, i.e., penetration)
 *
 * where:
 *   δ = |x_i - x_j| - (r_i + r_j)  (gap)
 *   n = (x_j - x_i) / |x_j - x_i|  (normal direction)
 *   k_c = contact stiffness
 */

#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_particle.hpp>
#include <vector>
#include <algorithm>

namespace nxs {
namespace pd {

/**
 * @brief Contact configuration
 */
struct PDContactConfig {
    Real contact_stiffness = 1e12;      ///< Contact stiffness (N/m³)
    Real friction_coefficient = 0.3;    ///< Coulomb friction coefficient
    Real damping_ratio = 0.1;           ///< Contact damping ratio
    Real search_factor = 1.5;           ///< Search radius = factor * max_radius
    bool enable_friction = true;        ///< Enable friction
    bool enable_self_contact = false;   ///< Enable self-contact within same body
};

/**
 * @brief Contact pair
 */
struct ContactPair {
    Index i;            ///< Particle i
    Index j;            ///< Particle j
    Real gap;           ///< Gap distance (negative = penetration)
    Real normal[3];     ///< Contact normal (j to i)
};

/**
 * @brief Cell for spatial hashing
 */
struct SpatialCell {
    std::vector<Index> particles;
};

/**
 * @brief PD contact handler with spatial hashing
 */
class PDContact {
public:
    PDContact() = default;

    /**
     * @brief Initialize contact
     */
    void initialize(const PDContactConfig& config) {
        config_ = config;
        NXS_LOG_INFO("PDContact initialized: k={:.2e}, μ={:.2f}",
                     config_.contact_stiffness, config_.friction_coefficient);
    }

    /**
     * @brief Build spatial hash for contact detection
     *
     * @param particles Particle system
     */
    void build_spatial_hash(PDParticleSystem& particles) {
        particles.sync_to_host();

        Index np = particles.num_particles();
        auto x = particles.x_host();
        auto horizon = particles.horizon_host();
        auto active = particles.active_host();

        // Find bounding box and max radius
        Real min_coord[3] = {1e30, 1e30, 1e30};
        Real max_coord[3] = {-1e30, -1e30, -1e30};
        Real max_radius = 0.0;

        for (Index i = 0; i < np; ++i) {
            if (!active(i)) continue;
            for (int d = 0; d < 3; ++d) {
                min_coord[d] = std::min(min_coord[d], x(i, d));
                max_coord[d] = std::max(max_coord[d], x(i, d));
            }
            // Particle radius ~ horizon / 3
            Real r = horizon(i) / 3.0;
            max_radius = std::max(max_radius, r);
        }

        // Cell size = search_factor * 2 * max_radius
        cell_size_ = config_.search_factor * 2.0 * max_radius;
        if (cell_size_ < 1e-10) cell_size_ = 1.0;

        // Grid dimensions
        for (int d = 0; d < 3; ++d) {
            grid_min_[d] = min_coord[d] - cell_size_;
            grid_max_[d] = max_coord[d] + cell_size_;
            grid_dims_[d] = static_cast<Index>(
                std::ceil((grid_max_[d] - grid_min_[d]) / cell_size_)
            );
            grid_dims_[d] = std::max(grid_dims_[d], Index(1));
        }

        // Allocate cells
        Index total_cells = grid_dims_[0] * grid_dims_[1] * grid_dims_[2];
        cells_.resize(total_cells);
        for (auto& cell : cells_) {
            cell.particles.clear();
        }

        // Insert particles into cells
        for (Index i = 0; i < np; ++i) {
            if (!active(i)) continue;

            Index cell_idx = get_cell_index(x(i, 0), x(i, 1), x(i, 2));
            if (cell_idx < total_cells) {
                cells_[cell_idx].particles.push_back(i);
            }
        }

        NXS_LOG_DEBUG("PDContact: spatial hash built, {} cells, max_radius={}",
                      total_cells, max_radius);
    }

    /**
     * @brief Detect contact pairs
     *
     * @param particles Particle system
     * @return Vector of contact pairs
     */
    std::vector<ContactPair> detect_contacts(PDParticleSystem& particles) {
        std::vector<ContactPair> pairs;

        particles.sync_to_host();

        Index np = particles.num_particles();
        auto x = particles.x_host();
        auto horizon = particles.horizon_host();
        auto body_id = particles.body_id_host();  // Use host view
        auto active = particles.active_host();

        // For each particle, check neighboring cells
        for (Index i = 0; i < np; ++i) {
            if (!active(i)) continue;

            Real xi[3] = {x(i, 0), x(i, 1), x(i, 2)};
            Real ri = horizon(i) / 3.0;  // Approximate particle radius
            Index body_i = body_id(i);

            // Get cell indices to search
            int ci[3], cj_min[3], cj_max[3];
            get_cell_coords(xi[0], xi[1], xi[2], ci);

            for (int d = 0; d < 3; ++d) {
                cj_min[d] = std::max(0, ci[d] - 1);
                cj_max[d] = std::min(static_cast<int>(grid_dims_[d]) - 1, ci[d] + 1);
            }

            // Search neighboring cells
            for (int cz = cj_min[2]; cz <= cj_max[2]; ++cz) {
                for (int cy = cj_min[1]; cy <= cj_max[1]; ++cy) {
                    for (int cx = cj_min[0]; cx <= cj_max[0]; ++cx) {
                        Index cell_idx = cx + cy * grid_dims_[0] +
                                        cz * grid_dims_[0] * grid_dims_[1];

                        for (Index j : cells_[cell_idx].particles) {
                            if (j <= i) continue;  // Avoid duplicates
                            if (!active(j)) continue;

                            // Skip same body unless self-contact enabled
                            Index body_j = body_id(j);
                            if (body_i == body_j && !config_.enable_self_contact) {
                                continue;
                            }

                            Real xj[3] = {x(j, 0), x(j, 1), x(j, 2)};
                            Real rj = horizon(j) / 3.0;

                            // Distance between particles
                            Real dx = xj[0] - xi[0];
                            Real dy = xj[1] - xi[1];
                            Real dz = xj[2] - xi[2];
                            Real dist = std::sqrt(dx*dx + dy*dy + dz*dz);

                            // Gap (negative = penetration)
                            Real gap = dist - (ri + rj);

                            if (gap < 0.0) {
                                ContactPair pair;
                                pair.i = i;
                                pair.j = j;
                                pair.gap = gap;

                                // Normal from i to j
                                Real inv_dist = 1.0 / (dist + 1e-20);
                                pair.normal[0] = dx * inv_dist;
                                pair.normal[1] = dy * inv_dist;
                                pair.normal[2] = dz * inv_dist;

                                pairs.push_back(pair);
                            }
                        }
                    }
                }
            }
        }

        return pairs;
    }

    /**
     * @brief Compute contact forces
     *
     * @param particles Particle system
     * @param pairs Contact pairs
     */
    void compute_forces(PDParticleSystem& particles,
                        const std::vector<ContactPair>& pairs) {
        if (pairs.empty()) return;

        auto f = particles.f();
        auto v = particles.v();
        auto mass = particles.mass();

        Real k_c = config_.contact_stiffness;
        Real mu = config_.friction_coefficient;
        Real zeta = config_.damping_ratio;
        bool use_friction = config_.enable_friction;

        // Copy pairs to device
        Index num_pairs = pairs.size();
        Kokkos::View<ContactPair*> pairs_device("contact_pairs", num_pairs);
        auto pairs_host = Kokkos::create_mirror_view(pairs_device);

        for (Index i = 0; i < num_pairs; ++i) {
            pairs_host(i) = pairs[i];
        }
        Kokkos::deep_copy(pairs_device, pairs_host);

        // Compute contact forces
        Kokkos::parallel_for("compute_contact_forces", num_pairs,
            KOKKOS_LAMBDA(const Index p) {
                const auto& pair = pairs_device(p);
                Index i = pair.i;
                Index j = pair.j;
                Real gap = pair.gap;

                if (gap >= 0.0) return;  // No contact

                Real n[3] = {pair.normal[0], pair.normal[1], pair.normal[2]};

                // Normal contact force (penalty)
                Real delta = -gap;  // Penetration depth
                Real fn_mag = k_c * delta;

                // Damping
                Real vi[3] = {v(i, 0), v(i, 1), v(i, 2)};
                Real vj[3] = {v(j, 0), v(j, 1), v(j, 2)};
                Real vrel[3] = {vj[0] - vi[0], vj[1] - vi[1], vj[2] - vi[2]};

                // Normal relative velocity
                Real vn = vrel[0] * n[0] + vrel[1] * n[1] + vrel[2] * n[2];

                // Effective mass
                Real mi = mass(i);
                Real mj = mass(j);
                Real m_eff = (mi * mj) / (mi + mj);

                // Critical damping
                Real c_crit = 2.0 * Kokkos::sqrt(k_c * m_eff);
                Real c_n = zeta * c_crit;

                // Total normal force
                Real fn = fn_mag - c_n * vn;
                if (fn < 0.0) fn = 0.0;  // No tensile contact

                // Normal force vector
                Real fn_vec[3] = {fn * n[0], fn * n[1], fn * n[2]};

                // Friction force
                Real ft_vec[3] = {0.0, 0.0, 0.0};
                if (use_friction && mu > 0.0) {
                    // Tangential relative velocity
                    Real vt[3] = {
                        vrel[0] - vn * n[0],
                        vrel[1] - vn * n[1],
                        vrel[2] - vn * n[2]
                    };

                    Real vt_mag = Kokkos::sqrt(vt[0]*vt[0] + vt[1]*vt[1] + vt[2]*vt[2]);

                    if (vt_mag > 1e-10) {
                        // Coulomb friction (sliding)
                        Real ft_max = mu * fn;
                        Real ft_mag = Kokkos::fmin(ft_max, c_n * vt_mag);

                        ft_vec[0] = -ft_mag * vt[0] / vt_mag;
                        ft_vec[1] = -ft_mag * vt[1] / vt_mag;
                        ft_vec[2] = -ft_mag * vt[2] / vt_mag;
                    }
                }

                // Apply forces (Newton's third law)
                // Force on i (from j)
                Kokkos::atomic_add(&f(i, 0), fn_vec[0] + ft_vec[0]);
                Kokkos::atomic_add(&f(i, 1), fn_vec[1] + ft_vec[1]);
                Kokkos::atomic_add(&f(i, 2), fn_vec[2] + ft_vec[2]);

                // Force on j (from i)
                Kokkos::atomic_add(&f(j, 0), -(fn_vec[0] + ft_vec[0]));
                Kokkos::atomic_add(&f(j, 1), -(fn_vec[1] + ft_vec[1]));
                Kokkos::atomic_add(&f(j, 2), -(fn_vec[2] + ft_vec[2]));
            });
    }

    /**
     * @brief Full contact step: detect and compute forces
     */
    void apply_contact(PDParticleSystem& particles) {
        build_spatial_hash(particles);
        auto pairs = detect_contacts(particles);

        if (!pairs.empty()) {
            compute_forces(particles, pairs);
            NXS_LOG_DEBUG("PDContact: {} contact pairs, max_penetration={}",
                         pairs.size(),
                         -std::min_element(pairs.begin(), pairs.end(),
                             [](const auto& a, const auto& b) {
                                 return a.gap < b.gap;
                             })->gap);
        }
    }

    // Accessors
    const PDContactConfig& config() const { return config_; }
    Index num_active_contacts() const { return num_contacts_; }

private:
    /**
     * @brief Get cell index from position
     */
    Index get_cell_index(Real x, Real y, Real z) const {
        int ci[3];
        get_cell_coords(x, y, z, ci);

        for (int d = 0; d < 3; ++d) {
            ci[d] = std::max(0, std::min(ci[d], static_cast<int>(grid_dims_[d]) - 1));
        }

        return ci[0] + ci[1] * grid_dims_[0] + ci[2] * grid_dims_[0] * grid_dims_[1];
    }

    /**
     * @brief Get cell coordinates from position
     */
    void get_cell_coords(Real x, Real y, Real z, int* ci) const {
        ci[0] = static_cast<int>((x - grid_min_[0]) / cell_size_);
        ci[1] = static_cast<int>((y - grid_min_[1]) / cell_size_);
        ci[2] = static_cast<int>((z - grid_min_[2]) / cell_size_);
    }

    PDContactConfig config_;
    Index num_contacts_ = 0;

    // Spatial hash
    Real cell_size_ = 1.0;
    Real grid_min_[3] = {0, 0, 0};
    Real grid_max_[3] = {1, 1, 1};
    Index grid_dims_[3] = {1, 1, 1};
    std::vector<SpatialCell> cells_;
};

} // namespace pd
} // namespace nxs
