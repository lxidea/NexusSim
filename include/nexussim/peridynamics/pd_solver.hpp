#pragma once

/**
 * @file pd_solver.hpp
 * @brief Peridynamics solver
 *
 * Ported from PeriSys-Haoran JSolve.cu
 * Integrates all PD components for time stepping
 */

#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_particle.hpp>
#include <nexussim/peridynamics/pd_neighbor.hpp>
#include <nexussim/peridynamics/pd_force.hpp>
#include <nexussim/io/vtk_writer.hpp>
#include <memory>
#include <functional>

namespace nxs {
namespace pd {

/**
 * @brief Peridynamics solver configuration
 */
struct PDSolverConfig {
    Real dt = 1e-7;                     ///< Time step (s)
    Index total_steps = 1000;           ///< Total time steps
    Index output_interval = 100;        ///< Output interval

    Real horizon_factor = 3.015;        ///< Influence function factor
    bool check_damage = true;           ///< Enable damage/fracture

    Real gravity[3] = {0.0, 0.0, 0.0};  ///< Gravity vector

    // Damping
    Real damping_coefficient = 0.0;     ///< Viscous damping (0 = none)

    // ADR (Adaptive Dynamic Relaxation)
    bool use_adr = false;               ///< Use ADR for quasi-static
};

/**
 * @brief Main peridynamics solver
 */
class PDSolver {
public:
    using OutputCallback = std::function<void(Index step, Real time)>;

    PDSolver() = default;

    /**
     * @brief Initialize solver
     * @param config Solver configuration
     */
    void initialize(const PDSolverConfig& config) {
        config_ = config;
        time_ = 0.0;
        step_ = 0;

        NXS_LOG_INFO("PDSolver initialized: dt={}, steps={}", config_.dt, config_.total_steps);
    }

    /**
     * @brief Set materials
     */
    void set_materials(const std::vector<PDMaterial>& materials) {
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
            NXS_LOG_ERROR("PDSolver: particles not set");
            return;
        }

        neighbors_.build(*particles_, config_.horizon_factor);
    }

    /**
     * @brief Set output callback
     */
    void set_output_callback(OutputCallback callback) {
        output_callback_ = callback;
    }

    /**
     * @brief Perform single time step
     */
    void step() {
        if (!particles_) return;

        Real dt = config_.dt;

        // Velocity-Verlet: first half
        particles_->verlet_first_half(dt);

        // Compute internal forces
        force_.compute_forces(*particles_, neighbors_);

        // Add external forces (gravity)
        if (std::abs(config_.gravity[0]) > 1e-20 ||
            std::abs(config_.gravity[1]) > 1e-20 ||
            std::abs(config_.gravity[2]) > 1e-20) {
            apply_gravity(*particles_,
                         config_.gravity[0],
                         config_.gravity[1],
                         config_.gravity[2]);
        }
        particles_->add_external_forces();

        // Apply damping (if any)
        if (config_.damping_coefficient > 0.0) {
            apply_damping(config_.damping_coefficient);
        }

        // Compute acceleration
        particles_->compute_acceleration();

        // Velocity-Verlet: second half
        particles_->verlet_second_half(dt);

        // Check for bond failure
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
        NXS_LOG_INFO("PDSolver: Starting simulation, {} steps", config_.total_steps);

        for (step_ = 0; step_ < config_.total_steps; ++step_) {
            step();

            // Output at intervals
            if (step_ % config_.output_interval == 0 || step_ == 0) {
                if (output_callback_) {
                    output_callback_(step_, time_);
                }

                Real KE = particles_->compute_kinetic_energy();
                Real avg_damage = particles_->compute_average_damage();
                Index broken = neighbors_.count_broken_bonds();

                NXS_LOG_INFO("Step {}: time={:.2e}, KE={:.2e}, damage={:.3f}, broken_bonds={}",
                            step_, time_, KE, avg_damage, broken);
            }
        }

        NXS_LOG_INFO("PDSolver: Simulation complete, final time={:.2e}", time_);
    }

    /**
     * @brief Write VTK output for visualization
     */
    void write_vtk(const std::string& filename) {
        if (!particles_) return;

        // Sync to host
        particles_->sync_to_host();

        Index np = particles_->num_particles();

        // Write VTK file
        std::ofstream file(filename);
        if (!file.is_open()) {
            NXS_LOG_ERROR("Cannot open file: {}", filename);
            return;
        }

        file << "# vtk DataFile Version 3.0\n";
        file << "Peridynamics Output\n";
        file << "ASCII\n";
        file << "DATASET UNSTRUCTURED_GRID\n";

        // Points
        file << "POINTS " << np << " double\n";
        auto x_host = particles_->x_host();
        for (Index i = 0; i < np; ++i) {
            file << x_host(i, 0) << " " << x_host(i, 1) << " " << x_host(i, 2) << "\n";
        }

        // Cells (each point is a vertex)
        file << "CELLS " << np << " " << (2 * np) << "\n";
        for (Index i = 0; i < np; ++i) {
            file << "1 " << i << "\n";
        }

        file << "CELL_TYPES " << np << "\n";
        for (Index i = 0; i < np; ++i) {
            file << "1\n";  // VTK_VERTEX
        }

        // Point data
        file << "POINT_DATA " << np << "\n";

        // Displacement
        file << "VECTORS displacement double\n";
        auto u_host = particles_->u_host();
        for (Index i = 0; i < np; ++i) {
            file << u_host(i, 0) << " " << u_host(i, 1) << " " << u_host(i, 2) << "\n";
        }

        // Velocity
        file << "VECTORS velocity double\n";
        auto v_host = particles_->v_host();
        for (Index i = 0; i < np; ++i) {
            file << v_host(i, 0) << " " << v_host(i, 1) << " " << v_host(i, 2) << "\n";
        }

        // Damage
        file << "SCALARS damage double 1\n";
        file << "LOOKUP_TABLE default\n";
        auto damage_host = particles_->damage_host();
        for (Index i = 0; i < np; ++i) {
            file << damage_host(i) << "\n";
        }

        // Volume
        file << "SCALARS volume double 1\n";
        file << "LOOKUP_TABLE default\n";
        auto volume_host = particles_->volume_host();
        for (Index i = 0; i < np; ++i) {
            file << volume_host(i) << "\n";
        }

        file.close();
        NXS_LOG_INFO("Wrote VTK file: {}", filename);
    }

    // Accessors
    Real time() const { return time_; }
    Index step() const { return step_; }
    PDParticleSystem& particles() { return *particles_; }
    PDNeighborList& neighbors() { return neighbors_; }
    const PDSolverConfig& config() const { return config_; }

private:
    /**
     * @brief Apply viscous damping
     */
    void apply_damping(Real c) {
        auto f = particles_->f();
        auto v = particles_->v();
        auto mass = particles_->mass();
        auto active = particles_->active();

        Kokkos::parallel_for("apply_damping", particles_->num_particles(),
            KOKKOS_LAMBDA(const Index i) {
                if (active(i)) {
                    // f -= c * m * v
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
    PDBondForce force_;
    std::vector<PDMaterial> materials_;

    OutputCallback output_callback_;
};

/**
 * @brief Compute stable time step for PD
 *
 * dt <= sqrt(2 * rho / (pi * c * delta * max_neighbors))
 */
inline Real compute_stable_dt(const PDMaterial& mat, Real horizon, Index max_neighbors) {
    Real pi = 3.14159265358979323846;
    Real factor = mat.rho / (pi * mat.c * horizon * max_neighbors);
    return 0.8 * std::sqrt(2.0 * factor);  // Safety factor 0.8
}

} // namespace pd
} // namespace nxs
