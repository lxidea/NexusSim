#pragma once

/**
 * @file pd_state_based.hpp
 * @brief State-based peridynamics implementation
 *
 * Implements ordinary state-based peridynamics (OSB-PD) which:
 * - Handles arbitrary Poisson's ratio (unlike bond-based ν=0.25)
 * - Uses dilatation (volume change) and deviatoric deformation
 * - Provides more accurate material response
 *
 * Reference: Silling et al. (2007) "Peridynamic States and Constitutive Modeling"
 */

#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_particle.hpp>
#include <nexussim/peridynamics/pd_neighbor.hpp>

namespace nxs {
namespace pd {

/**
 * @brief State-based peridynamics material
 *
 * Extends PDMaterial with state-based specific parameters
 */
struct PDStateMaterial {
    // Basic properties
    Real E = 2.0e11;        ///< Young's modulus (Pa)
    Real nu = 0.3;          ///< Poisson's ratio (can be any valid value)
    Real rho = 7800.0;      ///< Density (kg/m³)

    // Derived properties
    Real K = 0.0;           ///< Bulk modulus
    Real G = 0.0;           ///< Shear modulus

    // State-based parameters
    Real kappa = 0.0;       ///< Bulk modulus coefficient (3K)
    Real alpha = 0.0;       ///< Shear modulus coefficient (15G/m)

    // Failure
    Real s_critical = 0.01; ///< Critical stretch
    Real Gc = 100.0;        ///< Fracture energy (J/m²)

    /**
     * @brief Compute derived properties
     */
    void compute_derived() {
        K = E / (3.0 * (1.0 - 2.0 * nu));
        G = E / (2.0 * (1.0 + nu));
        kappa = 3.0 * K;
    }
};

/**
 * @brief Ordinary state-based peridynamics force calculator
 *
 * Force state decomposition:
 *   t = t_iso + t_dev
 *
 * where:
 *   t_iso = (3K θ / m) ω |ξ| e  (isotropic/volumetric)
 *   t_dev = (15G / m) ω e_d     (deviatoric)
 *
 * θ = dilatation = (3/m) ∫ω |ξ| s dV
 * s = stretch = (|ξ+η| - |ξ|) / |ξ|
 * e_d = deviatoric extension = e - (θ/3) |ξ|
 * e = scalar extension = |ξ+η| - |ξ|
 */
class PDStateForce {
public:
    PDStateForce() = default;

    /**
     * @brief Initialize with materials
     */
    void initialize(const std::vector<PDStateMaterial>& materials) {
        num_materials_ = materials.size();

        // Allocate material arrays
        K_ = PDScalarView("K", num_materials_);
        G_ = PDScalarView("G", num_materials_);
        kappa_ = PDScalarView("kappa", num_materials_);
        s_critical_ = PDScalarView("s_critical", num_materials_);

        auto K_host = Kokkos::create_mirror_view(K_);
        auto G_host = Kokkos::create_mirror_view(G_);
        auto kappa_host = Kokkos::create_mirror_view(kappa_);
        auto s_crit_host = Kokkos::create_mirror_view(s_critical_);

        for (Index i = 0; i < num_materials_; ++i) {
            K_host(i) = materials[i].K;
            G_host(i) = materials[i].G;
            kappa_host(i) = materials[i].kappa;
            s_crit_host(i) = materials[i].s_critical;
        }

        Kokkos::deep_copy(K_, K_host);
        Kokkos::deep_copy(G_, G_host);
        Kokkos::deep_copy(kappa_, kappa_host);
        Kokkos::deep_copy(s_critical_, s_crit_host);

        NXS_LOG_INFO("PDStateForce: {} materials initialized", num_materials_);
    }

    /**
     * @brief Compute weighted volume for all particles
     *
     * m_i = Σ_j ω(|ξ|) |ξ|² V_j
     */
    void compute_weighted_volume(PDParticleSystem& particles, PDNeighborList& neighbors) {
        Index num_particles = particles.num_particles();

        // Allocate if needed
        if (weighted_volume_.extent(0) != num_particles) {
            weighted_volume_ = PDScalarView("weighted_volume", num_particles);
        }

        auto volume = particles.volume();
        auto active = particles.active();

        auto neighbor_offset = neighbors.neighbor_offset();
        auto neighbor_list = neighbors.neighbor_list();
        auto neighbor_count = neighbors.neighbor_count();
        auto bond_weight = neighbors.bond_weight();
        auto bond_intact = neighbors.bond_intact();
        auto bond_length = neighbors.bond_length();

        auto m = weighted_volume_;

        Kokkos::parallel_for("compute_weighted_volume", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i)) {
                    m(i) = 0.0;
                    return;
                }

                Index offset = neighbor_offset(i);
                Index count = neighbor_count(i);

                Real m_i = 0.0;
                for (Index k = 0; k < count; ++k) {
                    Index bond_idx = offset + k;
                    if (!bond_intact(bond_idx)) continue;

                    Index j = neighbor_list(bond_idx);
                    if (!active(j)) continue;

                    Real w = bond_weight(bond_idx);
                    Real xi_len = bond_length(bond_idx);
                    Real Vj = volume(j);

                    // m += ω |ξ|² V_j
                    m_i += w * xi_len * xi_len * Vj;
                }

                m(i) = m_i;
            });
    }

    /**
     * @brief Compute dilatation for all particles
     *
     * θ_i = (3/m_i) Σ_j ω(|ξ|) |ξ| (|ξ+η| - |ξ|) V_j
     *     = (3/m_i) Σ_j ω |ξ| e V_j
     */
    void compute_dilatation(PDParticleSystem& particles, PDNeighborList& neighbors) {
        Index num_particles = particles.num_particles();

        // Allocate if needed
        if (dilatation_.extent(0) != num_particles) {
            dilatation_ = PDScalarView("dilatation", num_particles);
        }

        auto u = particles.u();
        auto volume = particles.volume();
        auto active = particles.active();

        auto neighbor_offset = neighbors.neighbor_offset();
        auto neighbor_list = neighbors.neighbor_list();
        auto neighbor_count = neighbors.neighbor_count();
        auto bond_weight = neighbors.bond_weight();
        auto bond_intact = neighbors.bond_intact();
        auto bond_xi = neighbors.bond_xi();
        auto bond_length = neighbors.bond_length();

        auto m = weighted_volume_;
        auto theta = dilatation_;

        Kokkos::parallel_for("compute_dilatation", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i) || m(i) < 1e-30) {
                    theta(i) = 0.0;
                    return;
                }

                Index offset = neighbor_offset(i);
                Index count = neighbor_count(i);

                Real theta_i = 0.0;
                for (Index k = 0; k < count; ++k) {
                    Index bond_idx = offset + k;
                    if (!bond_intact(bond_idx)) continue;

                    Index j = neighbor_list(bond_idx);
                    if (!active(j)) continue;

                    Real w = bond_weight(bond_idx);
                    Real xi_len = bond_length(bond_idx);
                    Real Vj = volume(j);

                    // Relative displacement
                    Real eta[3] = {
                        u(j, 0) - u(i, 0),
                        u(j, 1) - u(i, 1),
                        u(j, 2) - u(i, 2)
                    };

                    // Deformed bond
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

                    // Extension e = |ξ+η| - |ξ|
                    Real e = xi_eta_len - xi_len;

                    // θ += (3/m) ω |ξ| e V_j
                    theta_i += w * xi_len * e * Vj;
                }

                theta(i) = 3.0 * theta_i / m(i);
            });
    }

    /**
     * @brief Compute state-based PD forces
     *
     * Force state: t = t_iso + t_dev
     *   t_iso = (κ θ / m) ω |ξ| ê  (κ = 3K)
     *   t_dev = (α / m) ω e_d ê    (α = 15G)
     *
     * where e_d = e - (θ/3)|ξ| is deviatoric extension
     */
    void compute_forces(PDParticleSystem& particles, PDNeighborList& neighbors) {
        // First compute weighted volume and dilatation
        compute_weighted_volume(particles, neighbors);
        compute_dilatation(particles, neighbors);

        // Zero forces
        particles.zero_forces();

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

        auto m = weighted_volume_;
        auto theta = dilatation_;
        auto K = K_;
        auto G = G_;

        Index num_particles = particles.num_particles();

        Kokkos::parallel_for("compute_state_forces", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                if (!active(i) || m(i) < 1e-30) return;

                Index offset = neighbor_offset(i);
                Index count = neighbor_count(i);
                Index mat_i = material_id(i);

                Real K_i = K(mat_i);
                Real G_i = G(mat_i);
                Real m_i = m(i);
                Real theta_i = theta(i);

                // Coefficients
                Real kappa_coef = 3.0 * K_i / m_i;  // κ/m
                Real alpha_coef = 15.0 * G_i / m_i; // α/m

                Real fi[3] = {0.0, 0.0, 0.0};

                for (Index k = 0; k < count; ++k) {
                    Index bond_idx = offset + k;
                    if (!bond_intact(bond_idx)) continue;

                    Index j = neighbor_list(bond_idx);
                    if (!active(j)) continue;

                    Real w = bond_weight(bond_idx);
                    Real xi_len = bond_length(bond_idx);
                    Real Vj = volume(j);

                    // Get neighbor's dilatation
                    Real theta_j = theta(j);
                    Real m_j = m(j);

                    // Relative displacement
                    Real eta[3] = {
                        u(j, 0) - u(i, 0),
                        u(j, 1) - u(i, 1),
                        u(j, 2) - u(i, 2)
                    };

                    // Deformed bond
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

                    // Extension
                    Real e = xi_eta_len - xi_len;

                    // Deviatoric extension (average of i and j contributions)
                    Real e_d_i = e - theta_i * xi_len / 3.0;
                    Real e_d_j = e - theta_j * xi_len / 3.0;

                    // Unit vector in deformed direction
                    Real inv_len = 1.0 / (xi_eta_len + 1e-20);
                    Real e_hat[3] = {
                        xi_eta[0] * inv_len,
                        xi_eta[1] * inv_len,
                        xi_eta[2] * inv_len
                    };

                    // Force state from i
                    // t_i = (κ θ_i / m_i) ω |ξ| + (α / m_i) ω e_d_i
                    Real t_iso_i = kappa_coef * theta_i * w * xi_len;
                    Real t_dev_i = alpha_coef * w * e_d_i;
                    Real t_i = t_iso_i + t_dev_i;

                    // Force state from j (for symmetry)
                    Real kappa_coef_j = (m_j > 1e-30) ? 3.0 * K(material_id(j)) / m_j : 0.0;
                    Real alpha_coef_j = (m_j > 1e-30) ? 15.0 * G(material_id(j)) / m_j : 0.0;
                    Real t_iso_j = kappa_coef_j * theta_j * w * xi_len;
                    Real t_dev_j = alpha_coef_j * w * e_d_j;
                    Real t_j = t_iso_j + t_dev_j;

                    // Net force density (average of both contributions)
                    Real t_avg = 0.5 * (t_i + t_j);

                    // Add to force
                    fi[0] += t_avg * e_hat[0] * Vj;
                    fi[1] += t_avg * e_hat[1] * Vj;
                    fi[2] += t_avg * e_hat[2] * Vj;
                }

                Kokkos::atomic_add(&f(i, 0), fi[0]);
                Kokkos::atomic_add(&f(i, 1), fi[1]);
                Kokkos::atomic_add(&f(i, 2), fi[2]);
            });
    }

    /**
     * @brief Check and break bonds exceeding critical stretch
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

        Kokkos::parallel_reduce("check_state_failure", num_particles,
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

                    if (s > s_crit) {
                        bond_intact(bond_idx) = false;
                        broken++;
                    }
                }
            }, new_broken);

        return new_broken;
    }

    // Accessors
    PDScalarView& weighted_volume() { return weighted_volume_; }
    PDScalarView& dilatation() { return dilatation_; }

private:
    Index num_materials_ = 0;
    PDScalarView K_;
    PDScalarView G_;
    PDScalarView kappa_;
    PDScalarView s_critical_;

    PDScalarView weighted_volume_;  ///< m_i for each particle
    PDScalarView dilatation_;       ///< θ_i for each particle
};

/**
 * @brief State-based PD solver
 */
class PDStateSolver {
public:
    PDStateSolver() = default;

    /**
     * @brief Initialize solver
     */
    void initialize(const PDSolverConfig& config) {
        config_ = config;
        time_ = 0.0;
        step_ = 0;
        NXS_LOG_INFO("PDStateSolver initialized: dt={}, steps={}", config_.dt, config_.total_steps);
    }

    /**
     * @brief Set materials
     */
    void set_materials(const std::vector<PDStateMaterial>& materials) {
        materials_ = materials;
        force_.initialize(materials);
    }

    /**
     * @brief Set particle system
     */
    void set_particles(std::shared_ptr<PDParticleSystem> particles) {
        particles_ = particles;
    }

    /**
     * @brief Build neighbor list
     */
    void build_neighbors() {
        if (!particles_) {
            NXS_LOG_ERROR("PDStateSolver: particles not set");
            return;
        }
        neighbors_.build(*particles_, config_.horizon_factor);
    }

    /**
     * @brief Perform single time step
     */
    void step() {
        if (!particles_) return;

        Real dt = config_.dt;

        // Velocity-Verlet: first half
        particles_->verlet_first_half(dt);

        // Compute state-based forces
        force_.compute_forces(*particles_, neighbors_);

        // Add external forces
        particles_->add_external_forces();

        // Apply damping
        if (config_.damping_coefficient > 0.0) {
            apply_damping(config_.damping_coefficient);
        }

        // Compute acceleration
        particles_->compute_acceleration();

        // Velocity-Verlet: second half
        particles_->verlet_second_half(dt);

        // Check bond failure
        if (config_.check_damage) {
            Index new_broken = force_.check_bond_failure(*particles_, neighbors_);
            if (new_broken > 0) {
                neighbors_.update_damage(*particles_);
            }
        }

        time_ += dt;
        step_++;
    }

    /**
     * @brief Run simulation
     */
    void run() {
        NXS_LOG_INFO("PDStateSolver: Starting simulation, {} steps", config_.total_steps);

        for (step_ = 0; step_ < config_.total_steps; ++step_) {
            step();

            if (step_ % config_.output_interval == 0 || step_ == 0) {
                Real KE = particles_->compute_kinetic_energy();
                Real avg_damage = particles_->compute_average_damage();
                Index broken = neighbors_.count_broken_bonds();

                NXS_LOG_INFO("Step {}: time={:.2e}, KE={:.2e}, damage={:.3f}, broken={}",
                            step_, time_, KE, avg_damage, broken);
            }
        }

        NXS_LOG_INFO("PDStateSolver: Complete, final time={:.2e}", time_);
    }

    // Accessors
    Real time() const { return time_; }
    Index current_step() const { return step_; }
    PDParticleSystem& particles() { return *particles_; }
    PDNeighborList& neighbors() { return neighbors_; }
    PDStateForce& force() { return force_; }

private:
    void apply_damping(Real c) {
        auto f = particles_->f();
        auto v = particles_->v();
        auto mass = particles_->mass();
        auto active = particles_->active();

        Kokkos::parallel_for("apply_damping", particles_->num_particles(),
            KOKKOS_LAMBDA(const Index i) {
                if (active(i)) {
                    Real m = mass(i);
                    f(i, 0) -= c * m * v(i, 0);
                    f(i, 1) -= c * m * v(i, 1);
                    f(i, 2) -= c * m * v(i, 2);
                }
            });
    }

    PDSolverConfig config_;
    Real time_ = 0.0;
    Index step_ = 0;

    std::shared_ptr<PDParticleSystem> particles_;
    PDNeighborList neighbors_;
    PDStateForce force_;
    std::vector<PDStateMaterial> materials_;
};

} // namespace pd
} // namespace nxs
