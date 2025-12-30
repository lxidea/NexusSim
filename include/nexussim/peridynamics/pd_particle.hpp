#pragma once

/**
 * @file pd_particle.hpp
 * @brief Peridynamics particle system with Kokkos views
 *
 * Ported from PeriSys-Haoran (ZHR) with Kokkos support
 * Manages particle data on host and device
 */

#include <nexussim/peridynamics/pd_types.hpp>
#include <vector>
#include <memory>

namespace nxs {
namespace pd {

/**
 * @brief Particle system for peridynamics simulation
 *
 * Stores particle state in Kokkos views for GPU acceleration.
 * Provides methods for initialization, time stepping, and data access.
 */
class PDParticleSystem {
public:
    PDParticleSystem() = default;

    /**
     * @brief Initialize particle system
     * @param num_particles Number of particles
     */
    void initialize(Index num_particles) {
        num_particles_ = num_particles;

        // Allocate device views
        x_ = PDPositionView("x", num_particles);
        x0_ = PDPositionView("x0", num_particles);
        u_ = PDPositionView("u", num_particles);
        v_ = PDVelocityView("v", num_particles);
        a_ = PDVelocityView("a", num_particles);
        f_ = PDForceView("f", num_particles);
        f_ext_ = PDForceView("f_ext", num_particles);

        volume_ = PDScalarView("volume", num_particles);
        horizon_ = PDScalarView("horizon", num_particles);
        mass_ = PDScalarView("mass", num_particles);
        damage_ = PDScalarView("damage", num_particles);
        theta_ = PDScalarView("theta", num_particles);
        temperature_ = PDScalarView("temperature", num_particles);

        material_id_ = PDIndexView("material_id", num_particles);
        body_id_ = PDIndexView("body_id", num_particles);
        bc_type_ = PDIndexView("bc_type", num_particles);
        active_ = PDBoolView("active", num_particles);

        // Create host mirrors
        x_host_ = Kokkos::create_mirror_view(x_);
        x0_host_ = Kokkos::create_mirror_view(x0_);
        u_host_ = Kokkos::create_mirror_view(u_);
        v_host_ = Kokkos::create_mirror_view(v_);
        a_host_ = Kokkos::create_mirror_view(a_);
        f_host_ = Kokkos::create_mirror_view(f_);
        volume_host_ = Kokkos::create_mirror_view(volume_);
        horizon_host_ = Kokkos::create_mirror_view(horizon_);
        mass_host_ = Kokkos::create_mirror_view(mass_);
        damage_host_ = Kokkos::create_mirror_view(damage_);
        active_host_ = Kokkos::create_mirror_view(active_);

        // Initialize all particles as active
        Kokkos::parallel_for("init_active", num_particles,
            KOKKOS_LAMBDA(const Index i) {
                active_(i) = true;
                damage_(i) = 0.0;
                temperature_(i) = 300.0;
                theta_(i) = 0.0;
            });
    }

    /**
     * @brief Set particle position (host)
     */
    void set_position(Index i, Real x, Real y, Real z) {
        x_host_(i, 0) = x;
        x_host_(i, 1) = y;
        x_host_(i, 2) = z;
        x0_host_(i, 0) = x;
        x0_host_(i, 1) = y;
        x0_host_(i, 2) = z;
    }

    /**
     * @brief Set particle velocity (host)
     */
    void set_velocity(Index i, Real vx, Real vy, Real vz) {
        v_host_(i, 0) = vx;
        v_host_(i, 1) = vy;
        v_host_(i, 2) = vz;
    }

    /**
     * @brief Set particle properties (host)
     */
    void set_properties(Index i, Real vol, Real delta, Real m) {
        volume_host_(i) = vol;
        horizon_host_(i) = delta;
        mass_host_(i) = m;
    }

    /**
     * @brief Set material and body ID (host)
     */
    void set_ids(Index i, Index mat_id, Index bod_id) {
        material_id_(i) = mat_id;
        body_id_(i) = bod_id;
    }

    /**
     * @brief Sync host data to device
     */
    void sync_to_device() {
        Kokkos::deep_copy(x_, x_host_);
        Kokkos::deep_copy(x0_, x0_host_);
        Kokkos::deep_copy(u_, u_host_);
        Kokkos::deep_copy(v_, v_host_);
        Kokkos::deep_copy(a_, a_host_);
        Kokkos::deep_copy(f_, f_host_);
        Kokkos::deep_copy(volume_, volume_host_);
        Kokkos::deep_copy(horizon_, horizon_host_);
        Kokkos::deep_copy(mass_, mass_host_);
        Kokkos::deep_copy(damage_, damage_host_);
        Kokkos::deep_copy(active_, active_host_);
    }

    /**
     * @brief Sync device data to host
     */
    void sync_to_host() {
        Kokkos::deep_copy(x_host_, x_);
        Kokkos::deep_copy(u_host_, u_);
        Kokkos::deep_copy(v_host_, v_);
        Kokkos::deep_copy(a_host_, a_);
        Kokkos::deep_copy(f_host_, f_);
        Kokkos::deep_copy(damage_host_, damage_);
        Kokkos::deep_copy(active_host_, active_);
    }

    /**
     * @brief Update positions from displacements (device)
     */
    void update_positions() {
        auto x = x_;
        auto x0 = x0_;
        auto u = u_;

        Kokkos::parallel_for("update_positions", num_particles_,
            KOKKOS_LAMBDA(const Index i) {
                x(i, 0) = x0(i, 0) + u(i, 0);
                x(i, 1) = x0(i, 1) + u(i, 1);
                x(i, 2) = x0(i, 2) + u(i, 2);
            });
    }

    /**
     * @brief Zero internal forces (device)
     */
    void zero_forces() {
        auto f = f_;

        Kokkos::parallel_for("zero_forces", num_particles_,
            KOKKOS_LAMBDA(const Index i) {
                f(i, 0) = 0.0;
                f(i, 1) = 0.0;
                f(i, 2) = 0.0;
            });
    }

    /**
     * @brief Add external forces to internal forces (device)
     */
    void add_external_forces() {
        auto f = f_;
        auto f_ext = f_ext_;

        Kokkos::parallel_for("add_external", num_particles_,
            KOKKOS_LAMBDA(const Index i) {
                f(i, 0) += f_ext(i, 0);
                f(i, 1) += f_ext(i, 1);
                f(i, 2) += f_ext(i, 2);
            });
    }

    /**
     * @brief Compute acceleration from forces: a = f / (rho * V)
     */
    void compute_acceleration() {
        auto a = a_;
        auto f = f_;
        auto volume = volume_;
        auto mass = mass_;
        auto active = active_;

        Kokkos::parallel_for("compute_accel", num_particles_,
            KOKKOS_LAMBDA(const Index i) {
                if (active(i) && mass(i) > 1e-20) {
                    Real inv_mass = 1.0 / mass(i);
                    a(i, 0) = f(i, 0) * volume(i) * inv_mass;
                    a(i, 1) = f(i, 1) * volume(i) * inv_mass;
                    a(i, 2) = f(i, 2) * volume(i) * inv_mass;
                } else {
                    a(i, 0) = 0.0;
                    a(i, 1) = 0.0;
                    a(i, 2) = 0.0;
                }
            });
    }

    /**
     * @brief Velocity-Verlet first half: v^{n+1/2} = v^n + 0.5*dt*a^n
     *                                    u^{n+1} = u^n + dt*v^{n+1/2}
     */
    void verlet_first_half(Real dt) {
        auto u = u_;
        auto v = v_;
        auto a = a_;
        auto active = active_;

        Kokkos::parallel_for("verlet_half1", num_particles_,
            KOKKOS_LAMBDA(const Index i) {
                if (active(i)) {
                    // v^{n+1/2} = v^n + 0.5*dt*a^n
                    v(i, 0) += 0.5 * dt * a(i, 0);
                    v(i, 1) += 0.5 * dt * a(i, 1);
                    v(i, 2) += 0.5 * dt * a(i, 2);

                    // u^{n+1} = u^n + dt*v^{n+1/2}
                    u(i, 0) += dt * v(i, 0);
                    u(i, 1) += dt * v(i, 1);
                    u(i, 2) += dt * v(i, 2);
                }
            });

        update_positions();
    }

    /**
     * @brief Velocity-Verlet second half: v^{n+1} = v^{n+1/2} + 0.5*dt*a^{n+1}
     */
    void verlet_second_half(Real dt) {
        auto v = v_;
        auto a = a_;
        auto active = active_;

        Kokkos::parallel_for("verlet_half2", num_particles_,
            KOKKOS_LAMBDA(const Index i) {
                if (active(i)) {
                    v(i, 0) += 0.5 * dt * a(i, 0);
                    v(i, 1) += 0.5 * dt * a(i, 1);
                    v(i, 2) += 0.5 * dt * a(i, 2);
                }
            });
    }

    /**
     * @brief Apply velocity boundary condition
     */
    void apply_velocity_bc(Index node, int dof, Real value) {
        Kokkos::parallel_for("apply_vel_bc", 1,
            KOKKOS_LAMBDA(const Index) {
                v_(node, dof) = value;
            });
    }

    /**
     * @brief Compute total kinetic energy
     */
    Real compute_kinetic_energy() const {
        Real KE = 0.0;
        auto v = v_;
        auto mass = mass_;
        auto active = active_;

        Kokkos::parallel_reduce("kinetic_energy", num_particles_,
            KOKKOS_LAMBDA(const Index i, Real& ke_sum) {
                if (active(i)) {
                    Real v_sq = v(i, 0) * v(i, 0) + v(i, 1) * v(i, 1) + v(i, 2) * v(i, 2);
                    ke_sum += 0.5 * mass(i) * v_sq;
                }
            }, KE);

        return KE;
    }

    /**
     * @brief Compute total damage (fraction of broken bonds)
     */
    Real compute_average_damage() const {
        Real total_damage = 0.0;
        Index active_count = 0;
        auto damage = damage_;
        auto active = active_;

        Kokkos::parallel_reduce("avg_damage", num_particles_,
            KOKKOS_LAMBDA(const Index i, Real& dmg_sum, Index& cnt) {
                if (active(i)) {
                    dmg_sum += damage(i);
                    cnt += 1;
                }
            }, total_damage, active_count);

        return (active_count > 0) ? total_damage / active_count : 0.0;
    }

    // Accessors
    Index num_particles() const { return num_particles_; }

    // Device view accessors
    PDPositionView& x() { return x_; }
    PDPositionView& x0() { return x0_; }
    PDPositionView& u() { return u_; }
    PDVelocityView& v() { return v_; }
    PDVelocityView& a() { return a_; }
    PDForceView& f() { return f_; }
    PDForceView& f_ext() { return f_ext_; }
    PDScalarView& volume() { return volume_; }
    PDScalarView& horizon() { return horizon_; }
    PDScalarView& mass() { return mass_; }
    PDScalarView& damage() { return damage_; }
    PDScalarView& theta() { return theta_; }
    PDScalarView& temperature() { return temperature_; }
    PDIndexView& material_id() { return material_id_; }
    PDIndexView& body_id() { return body_id_; }
    PDIndexView& bc_type() { return bc_type_; }
    PDBoolView& active() { return active_; }

    // Host view accessors
    PDPositionHostView& x_host() { return x_host_; }
    PDPositionHostView& x0_host() { return x0_host_; }
    PDPositionHostView& u_host() { return u_host_; }
    PDVelocityHostView& v_host() { return v_host_; }
    PDForceHostView& f_host() { return f_host_; }
    PDScalarHostView& volume_host() { return volume_host_; }
    PDScalarHostView& horizon_host() { return horizon_host_; }
    PDScalarHostView& mass_host() { return mass_host_; }
    PDScalarHostView& damage_host() { return damage_host_; }
    PDBoolHostView& active_host() { return active_host_; }

private:
    Index num_particles_ = 0;

    // Device views
    PDPositionView x_;          ///< Current position
    PDPositionView x0_;         ///< Reference position
    PDPositionView u_;          ///< Displacement
    PDVelocityView v_;          ///< Velocity
    PDVelocityView a_;          ///< Acceleration
    PDForceView f_;             ///< Force density
    PDForceView f_ext_;         ///< External force density

    PDScalarView volume_;       ///< Particle volume
    PDScalarView horizon_;      ///< Horizon
    PDScalarView mass_;         ///< Mass
    PDScalarView damage_;       ///< Damage
    PDScalarView theta_;        ///< Dilatation
    PDScalarView temperature_;  ///< Temperature

    PDIndexView material_id_;   ///< Material ID
    PDIndexView body_id_;       ///< Body ID
    PDIndexView bc_type_;       ///< BC type
    PDBoolView active_;         ///< Active flag

    // Host mirrors
    PDPositionHostView x_host_;
    PDPositionHostView x0_host_;
    PDPositionHostView u_host_;
    PDVelocityHostView v_host_;
    PDVelocityHostView a_host_;
    PDForceHostView f_host_;
    PDScalarHostView volume_host_;
    PDScalarHostView horizon_host_;
    PDScalarHostView mass_host_;
    PDScalarHostView damage_host_;
    PDBoolHostView active_host_;
};

} // namespace pd
} // namespace nxs
