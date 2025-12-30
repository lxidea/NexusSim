#pragma once

/**
 * @file pd_force.hpp
 * @brief Peridynamics bond force calculation
 *
 * Ported from PeriSys-Haoran JParticle_stress.cu
 * Implements bond-based and state-based peridynamics force calculations
 */

#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_particle.hpp>
#include <nexussim/peridynamics/pd_neighbor.hpp>
#include <vector>

namespace nxs {
namespace pd {

/**
 * @brief Bond-based peridynamics force calculator
 *
 * Computes pairwise bond forces based on bond stretch.
 * Implements prototype microelastic brittle (PMB) model.
 */
class PDBondForce {
public:
    PDBondForce() = default;

    /**
     * @brief Initialize with materials
     */
    void initialize(const std::vector<PDMaterial>& materials) {
        num_materials_ = materials.size();

        // Allocate material arrays on device
        c_ = PDScalarView("c", num_materials_);
        s_critical_ = PDScalarView("s_critical", num_materials_);

        auto c_host = Kokkos::create_mirror_view(c_);
        auto s_crit_host = Kokkos::create_mirror_view(s_critical_);

        for (Index i = 0; i < num_materials_; ++i) {
            c_host(i) = materials[i].c;
            s_crit_host(i) = materials[i].s_critical;
        }

        Kokkos::deep_copy(c_, c_host);
        Kokkos::deep_copy(s_critical_, s_crit_host);

        NXS_LOG_INFO("PDBondForce: {} materials initialized", num_materials_);
    }

    /**
     * @brief Compute bond-based PD forces (PMB model)
     *
     * Force density from bond (i,j):
     *   f_{ij} = c * s * w * e_{ij}
     *
     * where:
     *   c = micromodulus (18K / pi delta^4 for 3D)
     *   s = stretch = (|xi + eta| - |xi|) / |xi|
     *   w = influence function weight
     *   e_{ij} = unit vector in deformed bond direction
     *
     * @param particles Particle system
     * @param neighbors Neighbor list
     */
    void compute_forces(PDParticleSystem& particles, PDNeighborList& neighbors) {
        // Zero forces first
        particles.zero_forces();

        auto x = particles.x();
        auto x0 = particles.x0();
        auto u = particles.u();
        auto f = particles.f();
        auto volume = particles.volume();
        auto material_id = particles.material_id();
        auto active = particles.active();

        auto neighbor_offset = neighbors.neighbor_offset();
        auto neighbor_list = neighbors.neighbor_list();
        auto neighbor_count = neighbors.neighbor_count();
        auto bond_weight = neighbors.bond_weight();
        auto bond_intact = neighbors.bond_intact();
        auto bond_xi = neighbors.bond_xi();
        auto bond_length = neighbors.bond_length();

        auto c = c_;
        auto s_critical = s_critical_;

        Index num_particles = particles.num_particles();

        // Compute forces
        Kokkos::parallel_for("compute_pd_forces", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i)) return;

                Index offset = neighbor_offset(i);
                Index count = neighbor_count(i);
                Index mat_i = material_id(i);

                Real c_i = c(mat_i);
                Real s_crit_i = s_critical(mat_i);

                Real fi[3] = {0.0, 0.0, 0.0};

                for (Index k = 0; k < count; ++k) {
                    Index bond_idx = offset + k;

                    if (!bond_intact(bond_idx)) continue;

                    Index j = neighbor_list(bond_idx);
                    if (!active(j)) continue;

                    // Get bond properties
                    Real w = bond_weight(bond_idx);
                    Real xi_len = bond_length(bond_idx);

                    // Reference bond vector
                    Real xi[3] = {
                        bond_xi(bond_idx, 0),
                        bond_xi(bond_idx, 1),
                        bond_xi(bond_idx, 2)
                    };

                    // Relative displacement eta = uj - ui
                    Real eta[3] = {
                        u(j, 0) - u(i, 0),
                        u(j, 1) - u(i, 1),
                        u(j, 2) - u(i, 2)
                    };

                    // Deformed bond vector xi + eta
                    Real xi_eta[3] = {
                        xi[0] + eta[0],
                        xi[1] + eta[1],
                        xi[2] + eta[2]
                    };

                    // Deformed bond length
                    Real xi_eta_len = Kokkos::sqrt(
                        xi_eta[0] * xi_eta[0] +
                        xi_eta[1] * xi_eta[1] +
                        xi_eta[2] * xi_eta[2]
                    );

                    // Bond stretch
                    Real s = (xi_eta_len - xi_len) / xi_len;

                    // Unit vector in deformed direction
                    Real inv_len = 1.0 / (xi_eta_len + 1e-20);
                    Real e[3] = {
                        xi_eta[0] * inv_len,
                        xi_eta[1] * inv_len,
                        xi_eta[2] * inv_len
                    };

                    // Bond force magnitude: t = c * s * w
                    // Force density contribution: f += t * e * Vj
                    Real t = c_i * s * w;
                    Real Vj = volume(j);

                    fi[0] += t * e[0] * Vj;
                    fi[1] += t * e[1] * Vj;
                    fi[2] += t * e[2] * Vj;
                }

                // Atomic add to force (for thread safety)
                Kokkos::atomic_add(&f(i, 0), fi[0]);
                Kokkos::atomic_add(&f(i, 1), fi[1]);
                Kokkos::atomic_add(&f(i, 2), fi[2]);
            });
    }

    /**
     * @brief Check and break bonds exceeding critical stretch
     *
     * @param particles Particle system
     * @param neighbors Neighbor list
     * @return Number of newly broken bonds
     */
    Index check_bond_failure(PDParticleSystem& particles, PDNeighborList& neighbors) {
        auto u = particles.u();
        auto material_id = particles.material_id();
        auto active = particles.active();

        auto neighbor_offset = neighbors.neighbor_offset();
        auto neighbor_list = neighbors.neighbor_list();
        auto neighbor_count = neighbors.neighbor_count();
        auto bond_intact = neighbors.bond_intact();
        auto bond_xi = neighbors.bond_xi();
        auto bond_length = neighbors.bond_length();

        auto s_critical = s_critical_;

        Index num_particles = particles.num_particles();
        Index new_broken = 0;

        Kokkos::parallel_reduce("check_failure", num_particles,
            KOKKOS_LAMBDA(const Index i, Index& broken) {
                if (!active(i)) return;

                Index offset = neighbor_offset(i);
                Index count = neighbor_count(i);
                Index mat_i = material_id(i);
                Real s_crit = s_critical(mat_i);

                for (Index k = 0; k < count; ++k) {
                    Index bond_idx = offset + k;

                    if (!bond_intact(bond_idx)) continue;

                    Index j = neighbor_list(bond_idx);
                    if (!active(j)) continue;

                    Real xi_len = bond_length(bond_idx);

                    // Relative displacement
                    Real eta[3] = {
                        u(j, 0) - u(i, 0),
                        u(j, 1) - u(i, 1),
                        u(j, 2) - u(i, 2)
                    };

                    // Deformed bond vector
                    Real xi_eta[3] = {
                        bond_xi(bond_idx, 0) + eta[0],
                        bond_xi(bond_idx, 1) + eta[1],
                        bond_xi(bond_idx, 2) + eta[2]
                    };

                    Real xi_eta_len = Kokkos::sqrt(
                        xi_eta[0] * xi_eta[0] +
                        xi_eta[1] * xi_eta[1] +
                        xi_eta[2] * xi_eta[2]
                    );

                    Real s = (xi_eta_len - xi_len) / xi_len;

                    // Check failure criterion
                    if (s > s_crit) {
                        bond_intact(bond_idx) = false;
                        broken++;
                    }
                }
            }, new_broken);

        return new_broken;
    }

    /**
     * @brief Compute strain energy density for all particles
     *
     * W = 0.5 * c * integral(s^2 * |xi| * w * dV)
     */
    void compute_strain_energy(PDParticleSystem& particles, PDNeighborList& neighbors,
                               PDScalarView& strain_energy) {
        auto u = particles.u();
        auto volume = particles.volume();
        auto material_id = particles.material_id();
        auto active = particles.active();

        auto neighbor_offset = neighbors.neighbor_offset();
        auto neighbor_list = neighbors.neighbor_list();
        auto neighbor_count = neighbors.neighbor_count();
        auto bond_weight = neighbors.bond_weight();
        auto bond_intact = neighbors.bond_intact();
        auto bond_xi = neighbors.bond_xi();
        auto bond_length = neighbors.bond_length();

        auto c = c_;
        Index num_particles = particles.num_particles();

        Kokkos::parallel_for("strain_energy", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i)) {
                    strain_energy(i) = 0.0;
                    return;
                }

                Index offset = neighbor_offset(i);
                Index count = neighbor_count(i);
                Index mat_i = material_id(i);
                Real c_i = c(mat_i);

                Real W = 0.0;

                for (Index k = 0; k < count; ++k) {
                    Index bond_idx = offset + k;

                    if (!bond_intact(bond_idx)) continue;

                    Index j = neighbor_list(bond_idx);
                    if (!active(j)) continue;

                    Real w = bond_weight(bond_idx);
                    Real xi_len = bond_length(bond_idx);

                    // Relative displacement
                    Real eta[3] = {
                        u(j, 0) - u(i, 0),
                        u(j, 1) - u(i, 1),
                        u(j, 2) - u(i, 2)
                    };

                    Real xi_eta[3] = {
                        bond_xi(bond_idx, 0) + eta[0],
                        bond_xi(bond_idx, 1) + eta[1],
                        bond_xi(bond_idx, 2) + eta[2]
                    };

                    Real xi_eta_len = Kokkos::sqrt(
                        xi_eta[0] * xi_eta[0] +
                        xi_eta[1] * xi_eta[1] +
                        xi_eta[2] * xi_eta[2]
                    );

                    Real s = (xi_eta_len - xi_len) / xi_len;
                    Real Vj = volume(j);

                    // Strain energy density: W += 0.5 * c * s^2 * |xi| * w * Vj
                    W += 0.5 * c_i * s * s * xi_len * w * Vj;
                }

                strain_energy(i) = W;
            });
    }

private:
    Index num_materials_ = 0;
    PDScalarView c_;            ///< Micromodulus for each material
    PDScalarView s_critical_;   ///< Critical stretch for each material
};

/**
 * @brief Apply gravity body force
 */
inline void apply_gravity(PDParticleSystem& particles, Real gx, Real gy, Real gz) {
    auto f_ext = particles.f_ext();
    auto mass = particles.mass();
    auto volume = particles.volume();
    auto active = particles.active();
    Index num_particles = particles.num_particles();

    Kokkos::parallel_for("apply_gravity", num_particles,
        KOKKOS_LAMBDA(const Index i) {
            if (active(i)) {
                Real rho = mass(i) / volume(i);
                f_ext(i, 0) = rho * gx;
                f_ext(i, 1) = rho * gy;
                f_ext(i, 2) = rho * gz;
            }
        });
}

} // namespace pd
} // namespace nxs
