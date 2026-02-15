#pragma once

/**
 * @file peridynamics.hpp
 * @brief Main include file for NexusSim Peridynamics module
 *
 * Includes all peridynamics components:
 * - pd_types.hpp: Core types and Kokkos views
 * - pd_particle.hpp: Particle system with Velocity-Verlet
 * - pd_neighbor.hpp: CSR neighbor list with influence functions
 * - pd_force.hpp: Bond-based force calculation
 * - pd_solver.hpp: Main PD solver
 * - pd_fem_coupling.hpp: FEM-PD coupling interface
 *
 * Usage:
 *   #include <nexussim/peridynamics/peridynamics.hpp>
 *
 *   // Create a simple PD simulation
 *   nxs::pd::PDSolver solver;
 *   nxs::pd::PDSolverConfig config;
 *   config.dt = 1e-7;
 *   solver.initialize(config);
 *
 *   auto particles = std::make_shared<nxs::pd::PDParticleSystem>();
 *   particles->initialize(num_particles);
 *   // ... set particle positions, properties ...
 *
 *   nxs::pd::PDMaterial mat;
 *   mat.E = 2e11;
 *   mat.nu = 0.25;
 *   mat.rho = 7800;
 *   mat.compute_derived(horizon);
 *
 *   solver.set_materials({mat});
 *   solver.set_particles(particles);
 *   solver.build_neighbors();
 *   solver.run();
 */

#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_particle.hpp>
#include <nexussim/peridynamics/pd_neighbor.hpp>
#include <nexussim/peridynamics/pd_force.hpp>
#include <nexussim/peridynamics/pd_solver.hpp>
#include <nexussim/peridynamics/pd_fem_coupling.hpp>
#include <nexussim/peridynamics/pd_state_based.hpp>
#include <nexussim/peridynamics/pd_materials.hpp>
#include <nexussim/peridynamics/pd_contact.hpp>
#include <nexussim/peridynamics/pd_correspondence.hpp>
#include <nexussim/peridynamics/pd_bond_models.hpp>
#include <nexussim/peridynamics/pd_morphing.hpp>
#include <nexussim/peridynamics/pd_mortar_coupling.hpp>
#include <nexussim/peridynamics/pd_adaptive_coupling.hpp>

namespace nxs {

/**
 * @brief Convenience namespace alias for peridynamics
 *
 * Allows using nxs::pd:: or nxs::peridynamics::
 */
namespace peridynamics = pd;

} // namespace nxs
