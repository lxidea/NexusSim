/**
 * @file hex20_debug_forces.cpp
 * @brief Debug Hex20 element force calculation
 *
 * Runs a few steps and dumps detailed diagnostics to identify NaN source.
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/fem/fem_solver.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/data/state.hpp>
#include <iostream>
#include <cmath>
#include <fstream>

using namespace nxs;
using namespace nxs::fem;

int main() {
    nxs::InitOptions options;
    options.log_level = nxs::Logger::Level::Debug;
    nxs::Context context(options);

    NXS_LOG_INFO("=================================================");
    NXS_LOG_INFO("Hex20 Force Calculation Debug");
    NXS_LOG_INFO("=================================================\n");

    try {
        // Create a single Hex20 element (20 nodes, no orphans!)
        const int n_nodes = 20;
        const int n_elems = 1;

        auto mesh = std::make_shared<Mesh>(n_nodes);

        // Define a simple beam element
        const Real L = 1.0;  // Length
        const Real W = 0.2;  // Width
        const Real H = 0.2;  // Height

        // Hex20 node positions
        std::vector<std::array<Real, 3>> node_coords = {
            // Corner nodes (0-7)
            {0.0, 0.0, 0.0},  {L,   0.0, 0.0},  {L,   W,   0.0},  {0.0, W,   0.0},
            {0.0, 0.0, H},    {L,   0.0, H},    {L,   W,   H},    {0.0, W,   H},
            // Bottom face mid-edge nodes (8-11)
            {L/2, 0.0, 0.0},  {L,   W/2, 0.0},  {L/2, W,   0.0},  {0.0, W/2, 0.0},
            // Vertical mid-edge nodes (12-15)
            {0.0, 0.0, H/2},  {L,   0.0, H/2},  {L,   W,   H/2},  {0.0, W,   H/2},
            // Top face mid-edge nodes (16-19)
            {L/2, 0.0, H},    {L,   W/2, H},    {L/2, W,   H},    {0.0, W/2, H}
        };

        for (int i = 0; i < n_nodes; ++i) {
            mesh->set_node_coordinates(i, node_coords[i]);
        }

        mesh->add_element_block("element", ElementType::Hex20, n_elems, 20);
        auto& block = mesh->element_block(0);

        auto elem_nodes = block.element_nodes(0);
        for (int i = 0; i < 20; ++i) {
            elem_nodes[i] = i;
        }

        std::vector<Index> connectivity;
        for (int i = 0; i < 20; ++i) {
            connectivity.push_back(i);
        }

        // Setup solver
        auto state = std::make_shared<State>(*mesh);
        FEMSolver solver("Hex20DebugForces");

        physics::MaterialProperties steel;
        steel.density = 7850.0;
        steel.E = 210.0e9;
        steel.nu = 0.3;
        steel.G = steel.E / (2.0 * (1.0 + steel.nu));
        steel.K = steel.E / (3.0 * (1.0 - 2.0 * steel.nu));

        std::vector<Index> elem_ids = {0};
        solver.add_element_group("element", physics::ElementType::Hex20,
                               elem_ids, connectivity, steel);

        // Add damping
        solver.set_damping(30.0);

        // Boundary conditions: fix left face (x=0)
        std::vector<Index> fixed_nodes = {0, 3, 4, 7, 11, 12, 15, 19};
        for (int dof = 0; dof < 3; ++dof) {
            BoundaryCondition bc_disp(BCType::Displacement, fixed_nodes, dof, 0.0);
            solver.add_boundary_condition(bc_disp);
        }

        // Apply load at right face (x=L)
        std::vector<Index> loaded_nodes = {1, 2, 5, 6, 9, 13, 14, 17};
        const Real total_force = -1000.0;
        const Real force_per_node = total_force / loaded_nodes.size();
        BoundaryCondition bc_force(BCType::Force, loaded_nodes, 2, force_per_node);
        solver.add_boundary_condition(bc_force);

        solver.initialize(mesh, state);

        const Real dt = solver.compute_stable_dt() * 0.9;

        NXS_LOG_INFO("Running debug simulation (340 steps to pass NaN point):");
        NXS_LOG_INFO("  Time step: {:.6e} s", dt);
        NXS_LOG_INFO("  Target: step 340 (NaN typically at 319)\n");

        // Run simulation with detailed monitoring
        std::ofstream log_file("hex20_debug.txt");
        log_file << "Step,Time,TipUz,TipUz_Vel,TipUz_Acc,MaxDisp,MinDisp,MaxForce\n";

        for (int step = 0; step < 340; ++step) {
            solver.step(dt);

            const Real tip_uz = solver.displacement()[1 * 3 + 2];
            const Real tip_vz = solver.velocity()[1 * 3 + 2];
            const Real tip_az = solver.acceleration()[1 * 3 + 2];

            // Check for NaN/Inf
            if (std::isnan(tip_uz) || std::isinf(tip_uz)) {
                NXS_LOG_ERROR("NaN/Inf detected at step {}!", step);
                NXS_LOG_ERROR("  Tip displacement: {}", tip_uz);
                NXS_LOG_ERROR("  Tip velocity: {}", tip_vz);
                NXS_LOG_ERROR("  Tip acceleration: {}", tip_az);

                // Dump all displacements
                NXS_LOG_ERROR("\nAll node displacements (Z-component):");
                for (int n = 0; n < 20; ++n) {
                    const Real uz = solver.displacement()[n * 3 + 2];
                    NXS_LOG_ERROR("  Node {}: uz = {}", n, uz);
                }

                log_file.close();
                return 1;
            }

            // Log every 10 steps around the critical zone
            if (step >= 310 || step % 10 == 0) {
                // Find max/min displacement
                Real max_disp = -1e30, min_disp = 1e30;
                for (int i = 0; i < 60; ++i) {
                    Real val = solver.displacement()[i];
                    max_disp = std::max(max_disp, std::abs(val));
                    min_disp = std::min(min_disp, val);
                }

                log_file << step << "," << step*dt << "," << tip_uz << ","
                         << tip_vz << "," << tip_az << ","
                         << max_disp << "," << min_disp << "," << 0.0 << "\n";

                NXS_LOG_INFO("Step {:3d}: tip_uz={:+.6e}, vel={:+.6e}, acc={:+.6e}, max_disp={:.6e}",
                            step, tip_uz, tip_vz, tip_az, max_disp);
            }
        }

        log_file.close();

        NXS_LOG_INFO("\n=================================================");
        NXS_LOG_INFO("DEBUG COMPLETE: Simulation reached step 340");
        NXS_LOG_INFO("Expected NaN at step 319, but completed successfully!");
        NXS_LOG_INFO("Check hex20_debug.txt for detailed log");
        NXS_LOG_INFO("=================================================");

        return 0;

    } catch (const std::exception& e) {
        NXS_LOG_ERROR("Test failed with exception: {}", e.what());
        return 1;
    }
}
