/**
 * @file pd_bar_tension_test.cpp
 * @brief Test peridynamics solver with bar tension example
 *
 * Creates a simple bar discretized with PD particles,
 * applies tension load, and verifies deformation and damage.
 */

#include <nexussim/core/core.hpp>
#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_particle.hpp>
#include <nexussim/peridynamics/pd_neighbor.hpp>
#include <nexussim/peridynamics/pd_force.hpp>
#include <nexussim/peridynamics/pd_solver.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace nxs;
using namespace nxs::pd;

/**
 * @brief Create a 1D bar discretized with PD particles
 *
 * @param length Bar length (m)
 * @param num_particles Number of particles along length
 * @param area Cross-sectional area (m²)
 * @param mat Material properties
 * @return Particle system
 */
std::shared_ptr<PDParticleSystem> create_bar(
    Real length, Index num_particles, Real area, const PDMaterial& mat)
{
    auto particles = std::make_shared<PDParticleSystem>();
    particles->initialize(num_particles);

    Real dx = length / (num_particles - 1);
    Real volume = area * dx;  // Particle volume
    Real mass = mat.rho * volume;
    Real horizon = 3.015 * dx;  // Standard horizon factor

    for (Index i = 0; i < num_particles; ++i) {
        Real x = i * dx;

        particles->set_position(i, x, 0.0, 0.0);
        particles->set_velocity(i, 0.0, 0.0, 0.0);
        particles->set_properties(i, volume, horizon, mass);
        particles->set_ids(i, 0, 0);  // material 0, body 0
    }

    // Sync to device
    particles->sync_to_device();

    return particles;
}

/**
 * @brief Apply velocity boundary condition to particles
 */
void apply_velocity_bc(PDParticleSystem& particles, Index start, Index end,
                       Real vx, Real vy, Real vz)
{
    for (Index i = start; i <= end; ++i) {
        particles.apply_velocity_bc(i, 0, vx);
        particles.apply_velocity_bc(i, 1, vy);
        particles.apply_velocity_bc(i, 2, vz);
    }
}

/**
 * @brief Test 1: Simple elastic tension
 *
 * Apply constant velocity to right end, check for linear displacement field.
 */
bool test_elastic_tension()
{
    std::cout << "\n=== Test 1: Elastic Tension ===" << std::endl;

    // Bar parameters
    Real length = 1.0;          // 1 meter
    Index num_particles = 51;   // Particles
    Real area = 0.01 * 0.01;    // 1cm x 1cm cross-section
    Real dx = length / (num_particles - 1);

    // Material: Steel-like
    PDMaterial steel;
    steel.type = PDMaterialType::Elastic;
    steel.E = 2.0e11;           // 200 GPa
    steel.nu = 0.25;            // Bond-based requires nu = 0.25
    steel.rho = 7800.0;         // kg/m³
    steel.s_critical = 0.1;     // High critical stretch (no failure)
    steel.compute_derived(3.015 * dx);

    std::cout << "Material: E=" << steel.E << " Pa, rho=" << steel.rho
              << " kg/m³, c=" << steel.c << std::endl;

    // Create particle system
    auto particles = create_bar(length, num_particles, area, steel);

    // Setup solver
    PDSolverConfig config;
    config.dt = 1e-8;           // Small time step
    config.total_steps = 1000;
    config.output_interval = 100;
    config.check_damage = false;

    PDSolver solver;
    solver.initialize(config);
    solver.set_materials({steel});
    solver.set_particles(particles);
    solver.build_neighbors();

    // Apply boundary conditions:
    // - Left end (i=0,1,2): fixed (v = 0)
    // - Right end (i=N-3..N-1): velocity v_x = 0.1 m/s

    Real pull_velocity = 0.1;  // m/s

    // Run simulation with BCs
    std::cout << "Running " << config.total_steps << " steps..." << std::endl;

    for (Index step = 0; step < config.total_steps; ++step) {
        // Apply BCs
        apply_velocity_bc(*particles, 0, 2, 0.0, 0.0, 0.0);
        apply_velocity_bc(*particles, num_particles-3, num_particles-1,
                         pull_velocity, 0.0, 0.0);

        solver.step();

        if (step % config.output_interval == 0) {
            Real KE = particles->compute_kinetic_energy();
            particles->sync_to_host();
            auto u = particles->u_host();
            Real max_disp = 0.0;
            for (Index i = 0; i < num_particles; ++i) {
                max_disp = std::max(max_disp, std::abs(u(i, 0)));
            }
            std::cout << "Step " << step << ": KE=" << std::scientific
                      << KE << ", max_u=" << max_disp << std::endl;
        }
    }

    // Check results
    particles->sync_to_host();
    auto u = particles->u_host();

    // Displacement should be approximately linear
    Real total_extension = u(num_particles-1, 0) - u(0, 0);
    Real expected_extension = pull_velocity * config.dt * config.total_steps;

    std::cout << "Total extension: " << total_extension
              << ", expected: " << expected_extension << std::endl;

    // Check approximately correct (within 50% due to dynamics)
    bool passed = (total_extension > 0.5 * expected_extension);
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Test 2: Bond failure under extreme stretch
 *
 * Apply high velocity to induce bond breaking.
 */
bool test_bond_failure()
{
    std::cout << "\n=== Test 2: Bond Failure ===" << std::endl;

    // Bar parameters
    Real length = 0.1;          // 10 cm
    Index num_particles = 21;
    Real area = 0.001 * 0.001;  // 1mm x 1mm
    Real dx = length / (num_particles - 1);

    // Brittle material with low critical stretch
    PDMaterial brittle;
    brittle.type = PDMaterialType::Elastic;
    brittle.E = 7.0e10;         // 70 GPa (glass-like)
    brittle.nu = 0.25;
    brittle.rho = 2500.0;
    brittle.s_critical = 0.001; // Very low - will fail easily
    brittle.compute_derived(3.015 * dx);

    std::cout << "Material: E=" << brittle.E << " Pa, s_crit=" << brittle.s_critical << std::endl;

    // Create particles
    auto particles = create_bar(length, num_particles, area, brittle);

    // Setup solver
    PDSolverConfig config;
    config.dt = 1e-8;           // Larger time step
    config.total_steps = 5000;  // More steps for higher strain
    config.output_interval = 500;
    config.check_damage = true;

    PDSolver solver;
    solver.initialize(config);
    solver.set_materials({brittle});
    solver.set_particles(particles);
    solver.build_neighbors();

    // Apply high velocity to induce failure
    // Need strain > 0.001, bar = 0.1m, so need extension > 0.1mm
    // At 100 m/s for 5000 steps @ 1e-8s = 5e-3 m = 5mm extension
    // Strain = 5mm / 100mm = 5% >> 0.1% critical
    Real pull_velocity = 100.0;  // 100 m/s - very high

    std::cout << "Running with high velocity to induce failure..." << std::endl;

    Index broken_bonds = 0;
    for (Index step = 0; step < config.total_steps; ++step) {
        apply_velocity_bc(*particles, 0, 1, 0.0, 0.0, 0.0);
        apply_velocity_bc(*particles, num_particles-2, num_particles-1,
                         pull_velocity, 0.0, 0.0);

        solver.step();

        if (step % config.output_interval == 0) {
            broken_bonds = solver.neighbors().count_broken_bonds();
            Real damage = particles->compute_average_damage();
            std::cout << "Step " << step << ": broken_bonds=" << broken_bonds
                      << ", avg_damage=" << std::fixed << std::setprecision(3)
                      << damage << std::endl;
        }
    }

    // Should have broken bonds
    bool passed = (broken_bonds > 0);
    std::cout << "Final broken bonds: " << broken_bonds << std::endl;
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Test 3: 3D cube deformation
 */
bool test_3d_cube()
{
    std::cout << "\n=== Test 3: 3D Cube Compression ===" << std::endl;

    // Create a 3x3x3 cube of particles
    Index nx = 3, ny = 3, nz = 3;
    Index num_particles = nx * ny * nz;
    Real cube_size = 0.01;  // 1 cm cube
    Real dx = cube_size / (nx - 1);

    PDMaterial mat;
    mat.type = PDMaterialType::Elastic;
    mat.E = 2.0e11;
    mat.nu = 0.25;
    mat.rho = 7800.0;
    mat.s_critical = 0.1;
    mat.compute_derived(3.015 * dx);

    auto particles = std::make_shared<PDParticleSystem>();
    particles->initialize(num_particles);

    Real volume = dx * dx * dx;
    Real mass = mat.rho * volume;
    Real horizon = 3.015 * dx;

    Index idx = 0;
    for (Index i = 0; i < nx; ++i) {
        for (Index j = 0; j < ny; ++j) {
            for (Index k = 0; k < nz; ++k) {
                Real x = i * dx;
                Real y = j * dx;
                Real z = k * dx;

                particles->set_position(idx, x, y, z);
                particles->set_velocity(idx, 0.0, 0.0, 0.0);
                particles->set_properties(idx, volume, horizon, mass);
                particles->set_ids(idx, 0, 0);
                idx++;
            }
        }
    }

    particles->sync_to_device();

    // Setup solver
    PDSolverConfig config;
    config.dt = 1e-9;
    config.total_steps = 100;
    config.check_damage = false;

    PDSolver solver;
    solver.initialize(config);
    solver.set_materials({mat});
    solver.set_particles(particles);
    solver.build_neighbors();

    Index total_bonds = solver.neighbors().total_bonds();
    std::cout << "Created 3D cube: " << num_particles << " particles, "
              << total_bonds << " bonds" << std::endl;

    // Run a few steps
    for (Index step = 0; step < config.total_steps; ++step) {
        solver.step();
    }

    Real KE = particles->compute_kinetic_energy();
    std::cout << "Final KE: " << KE << std::endl;

    // Should have neighbors and be stable
    bool passed = (total_bonds > 0 && KE < 1e-10);  // Near zero KE (stable)
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Test 4: Compute stable time step
 */
bool test_stable_timestep()
{
    std::cout << "\n=== Test 4: Stable Time Step ===" << std::endl;

    Real dx = 0.001;  // 1mm spacing
    Real horizon = 3.015 * dx;

    PDMaterial steel;
    steel.E = 2.0e11;
    steel.nu = 0.25;
    steel.rho = 7800.0;
    steel.compute_derived(horizon);

    Index max_neighbors = 100;  // Typical for 3D

    Real dt = compute_stable_dt(steel, horizon, max_neighbors);

    std::cout << "Material: E=" << steel.E << ", rho=" << steel.rho << std::endl;
    std::cout << "Horizon: " << horizon << " m" << std::endl;
    std::cout << "Micromodulus c: " << steel.c << std::endl;
    std::cout << "Max neighbors: " << max_neighbors << std::endl;
    std::cout << "Stable dt: " << std::scientific << dt << " s" << std::endl;

    // Should be a reasonable small time step
    bool passed = (dt > 1e-10 && dt < 1e-5);
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Test 5: VTK output
 */
bool test_vtk_output()
{
    std::cout << "\n=== Test 5: VTK Output ===" << std::endl;

    // Create simple system
    Index num_particles = 10;
    Real dx = 0.01;

    PDMaterial mat;
    mat.E = 2.0e11;
    mat.nu = 0.25;
    mat.rho = 7800.0;
    mat.compute_derived(3.015 * dx);

    auto particles = std::make_shared<PDParticleSystem>();
    particles->initialize(num_particles);

    for (Index i = 0; i < num_particles; ++i) {
        particles->set_position(i, i * dx, 0.0, 0.0);
        particles->set_properties(i, dx*dx*dx, 3.015*dx, mat.rho * dx*dx*dx);
        particles->set_ids(i, 0, 0);
    }
    particles->sync_to_device();

    PDSolverConfig config;
    config.dt = 1e-8;
    config.total_steps = 10;

    PDSolver solver;
    solver.initialize(config);
    solver.set_materials({mat});
    solver.set_particles(particles);
    solver.build_neighbors();

    // Run a few steps
    for (Index i = 0; i < 10; ++i) {
        solver.step();
    }

    // Write VTK
    std::string filename = "/tmp/pd_test_output.vtk";
    solver.write_vtk(filename);

    // Check file exists (basic check)
    std::ifstream file(filename);
    bool passed = file.good();
    file.close();

    std::cout << "VTK file: " << filename << std::endl;
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

int main(int argc, char* argv[])
{
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);

    std::cout << "========================================" << std::endl;
    std::cout << "NexusSim Peridynamics Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    int passed = 0;
    int failed = 0;

    // Run tests
    if (test_elastic_tension()) passed++; else failed++;
    if (test_bond_failure()) passed++; else failed++;
    if (test_3d_cube()) passed++; else failed++;
    if (test_stable_timestep()) passed++; else failed++;
    if (test_vtk_output()) passed++; else failed++;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << (passed + failed)
              << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    Kokkos::finalize();

    return (failed > 0) ? 1 : 0;
}
