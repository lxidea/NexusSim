/**
 * Large-scale Peridynamics benchmark
 * Tests realistic simulation performance on GPU
 */
#include <nexussim/core/core.hpp>
#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_particle.hpp>
#include <nexussim/peridynamics/pd_neighbor.hpp>
#include <nexussim/peridynamics/pd_force.hpp>
#include <nexussim/peridynamics/pd_solver.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

using namespace nxs;
using namespace nxs::pd;

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    std::cout << "============================================================\n";
    std::cout << "NexusSim Large-Scale Peridynamics Benchmark\n";
    std::cout << "============================================================\n\n";
    
    // Problem sizes to test
    std::vector<int> sizes = {10, 15, 20, 25};  // 3D grids: n x n x n
    
    for (int n : sizes) {
        Index num_particles = n * n * n;
        
        std::cout << "--- " << n << "x" << n << "x" << n << " grid (" 
                  << num_particles << " particles) ---\n";
        
        // Create particle system
        auto particles = std::make_shared<PDParticleSystem>();
        particles->initialize(num_particles);
        
        // Material properties (steel)
        PDMaterial steel;
        steel.E = 2.0e11;
        steel.nu = 0.25;
        steel.rho = 7800.0;
        steel.s_critical = 0.1;
        
        Real L = 0.1;  // 10cm cube
        Real dx = L / (n - 1);
        Real volume = dx * dx * dx;
        Real horizon = 3.015 * dx;
        Real mass = steel.rho * volume;
        steel.compute_derived(horizon);
        
        // Initialize particles in 3D grid
        Index idx = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
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
        
        // Build neighbor list
        PDNeighborList neighbors;
        auto t1 = std::chrono::high_resolution_clock::now();
        neighbors.build(*particles);
        auto t2 = std::chrono::high_resolution_clock::now();
        double neighbor_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        std::cout << "  Neighbor list: " << neighbors.total_bonds() << " bonds, "
                  << neighbor_time << " ms\n";
        
        // Force computation
        PDBondForce force;
        force.initialize({steel});
        
        // Warm up
        force.compute_forces(*particles, neighbors);
        Kokkos::fence();
        
        // Benchmark force computation
        int num_iterations = 100;
        t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            force.compute_forces(*particles, neighbors);
        }
        Kokkos::fence();
        t2 = std::chrono::high_resolution_clock::now();
        double force_time = std::chrono::duration<double, std::milli>(t2 - t1).count() / num_iterations;
        
        // Full time step benchmark
        Real dt = 1e-9;
        t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            particles->verlet_first_half(dt);
            particles->update_positions();
            force.compute_forces(*particles, neighbors);
            particles->compute_acceleration();
            particles->verlet_second_half(dt);
        }
        Kokkos::fence();
        t2 = std::chrono::high_resolution_clock::now();
        double step_time = std::chrono::duration<double, std::milli>(t2 - t1).count() / num_iterations;
        
        double bonds_per_sec = neighbors.total_bonds() / (force_time / 1000.0);
        double particles_per_sec = num_particles / (step_time / 1000.0);
        
        std::cout << "  Force computation: " << force_time << " ms/iter\n";
        std::cout << "  Full time step: " << step_time << " ms/iter\n";
        std::cout << "  Throughput: " << bonds_per_sec / 1e6 << " M bonds/sec, "
                  << particles_per_sec / 1e6 << " M particles/sec\n\n";
    }
    
    std::cout << "============================================================\n";
    std::cout << "Benchmark Complete\n";
    std::cout << "============================================================\n";

    Kokkos::finalize();
    return 0;
}
