/**
 * @file hex20_single_element_bending.cpp
 * @brief Simple single-element Hex20 test
 *
 * Tests a single Hex20 element under bending to verify element implementation.
 * This avoids mesh generation complexities by manually creating just one element.
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/fem/fem_solver.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/data/state.hpp>
#include <iostream>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;

int main() {
    // Initialize NexusSim
    nxs::InitOptions options;
    options.log_level = nxs::Logger::Level::Info;
    nxs::Context context(options);

    NXS_LOG_INFO("=================================================");
    NXS_LOG_INFO("Hex20 Single Element Test");
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

        // Hex20 node positions (corner nodes first, then mid-edge nodes)
        std::vector<std::array<Real, 3>> node_coords = {
            // Corner nodes (0-7)
            {0.0, 0.0, 0.0},  // 0
            {L,   0.0, 0.0},  // 1
            {L,   W,   0.0},  // 2
            {0.0, W,   0.0},  // 3
            {0.0, 0.0, H},    // 4
            {L,   0.0, H},    // 5
            {L,   W,   H},    // 6
            {0.0, W,   H},    // 7
            // Bottom face mid-edge nodes (8-11)
            {L/2, 0.0, 0.0},  // 8  (edge 0-1)
            {L,   W/2, 0.0},  // 9  (edge 1-2)
            {L/2, W,   0.0},  // 10 (edge 2-3)
            {0.0, W/2, 0.0},  // 11 (edge 3-0)
            // Vertical mid-edge nodes (12-15)
            {0.0, 0.0, H/2},  // 12 (edge 0-4)
            {L,   0.0, H/2},  // 13 (edge 1-5)
            {L,   W,   H/2},  // 14 (edge 2-6)
            {0.0, W,   H/2},  // 15 (edge 3-7)
            // Top face mid-edge nodes (16-19)
            {L/2, 0.0, H},    // 16 (edge 4-5)
            {L,   W/2, H},    // 17 (edge 5-6)
            {L/2, W,   H},    // 18 (edge 6-7)
            {0.0, W/2, H}     // 19 (edge 7-4)
        };

        // Set node coordinates
        for (int i = 0; i < n_nodes; ++i) {
            mesh->set_node_coordinates(i, node_coords[i]);
        }

        NXS_LOG_INFO("Created {} nodes for single Hex20 element", n_nodes);

        // Create element block
        mesh->add_element_block("element", ElementType::Hex20, n_elems, 20);
        auto& block = mesh->element_block(0);

        // Set connectivity (all 20 nodes in order)
        auto elem_nodes = block.element_nodes(0);
        for (int i = 0; i < 20; ++i) {
            elem_nodes[i] = i;
        }

        std::vector<Index> connectivity;
        for (int i = 0; i < 20; ++i) {
            connectivity.push_back(i);
        }

        NXS_LOG_INFO("Created 1 Hex20 element\n");

        // Setup solver
        auto state = std::make_shared<State>(*mesh);
        FEMSolver solver("Hex20SingleElementTest");

        // Material properties (steel)
        physics::MaterialProperties steel;
        steel.density = 7850.0;
        steel.E = 210.0e9;
        steel.nu = 0.3;
        steel.G = steel.E / (2.0 * (1.0 + steel.nu));
        steel.K = steel.E / (3.0 * (1.0 - 2.0 * steel.nu));

        std::vector<Index> elem_ids = {0};
        solver.add_element_group("element", physics::ElementType::Hex20,
                               elem_ids, connectivity, steel);

        // Add damping for stability
        const Real damping_ratio = 0.15;
        const Real natural_freq_est = 100.0;  // Rough estimate
        const Real damping_alpha = 2.0 * damping_ratio * natural_freq_est;
        solver.set_damping(damping_alpha);
        NXS_LOG_INFO("Set Rayleigh damping: α = {} (ξ = {})", damping_alpha, damping_ratio);

        // Boundary conditions: fix left face (x=0)
        std::vector<Index> fixed_nodes = {0, 3, 4, 7, 11, 12, 15, 19};  // Left face nodes
        for (int dof = 0; dof < 3; ++dof) {
            BoundaryCondition bc_disp(BCType::Displacement, fixed_nodes, dof, 0.0);
            solver.add_boundary_condition(bc_disp);
        }

        // Apply load at right face (x=L)
        std::vector<Index> loaded_nodes = {1, 2, 5, 6, 9, 13, 14, 17};  // Right face nodes
        const Real total_force = -1000.0;  // Downward force in Z
        const Real force_per_node = total_force / loaded_nodes.size();
        BoundaryCondition bc_force(BCType::Force, loaded_nodes, 2, force_per_node);
        solver.add_boundary_condition(bc_force);

        NXS_LOG_INFO("Applied BCs: {} nodes fixed, {} nodes loaded",
                    fixed_nodes.size(), loaded_nodes.size());

        // Initialize and run
        solver.initialize(mesh, state);

        const Real dt = solver.compute_stable_dt() * 0.1;  // Very conservative for stability test
        const Real total_time = 0.1;
        const int n_steps = static_cast<int>(total_time / dt);

        NXS_LOG_INFO("Running simulation:");
        NXS_LOG_INFO("  Time step: {:.6e} s", dt);
        NXS_LOG_INFO("  Total time: {} s", total_time);
        NXS_LOG_INFO("  Number of steps: {}\n", n_steps);

        // Run simulation
        for (int step = 0; step < n_steps; ++step) {
            solver.step(dt);

            // Output every 10% or every step near failure point (step 300-330)
            bool should_log = (step % (n_steps / 10) == 0) ||
                             (step >= 300 && step <= 330) ||
                             (step % 50 == 0) ||
                             (step < 5);  // Always log first 5 steps

            if (should_log) {
                // Check tip displacement (node 1, corner at free end)
                const Real tip_uz = solver.displacement()[1 * 3 + 2];
                const Real tip_uy = solver.displacement()[1 * 3 + 1];
                const Real tip_ux = solver.displacement()[1 * 3 + 0];

                // Check internal force at tip
                const Real tip_fz = solver.force_internal()[1 * 3 + 2];
                const Real ext_fz = solver.force_external()[1 * 3 + 2];

                const Real progress = 100.0 * step / n_steps;
                NXS_LOG_INFO("  [{:.1f}%] Step {}: u = ({:.3e}, {:.3e}, {:.3e}), f_int_z = {:.3e}, f_ext_z = {:.3e}",
                            progress, step, tip_ux, tip_uy, tip_uz, tip_fz, ext_fz);

                // Check for NaN
                if (std::isnan(tip_uz) || std::isnan(tip_ux) || std::isnan(tip_uy)) {
                    NXS_LOG_ERROR("NaN detected at step {}!", step);
                    return 1;
                }

                // Check if force is in wrong direction
                if (step > 0 && tip_uz * tip_fz > 0 && std::abs(tip_uz) > 1e-10) {
                    NXS_LOG_WARN("Step {}: Force and displacement have SAME sign! uz={:.3e}, fz_int={:.3e}",
                                step, tip_uz, tip_fz);
                }
            }
        }

        // Final results
        const Real tip_uz = solver.displacement()[1 * 3 + 2];

        NXS_LOG_INFO("\n=================================================");
        NXS_LOG_INFO("Final tip deflection: {:.6e} m ({:.3f} mm)", tip_uz, tip_uz * 1000.0);

        if (std::isnan(tip_uz)) {
            NXS_LOG_ERROR("HEX20 SINGLE ELEMENT TEST: FAILED (NaN detected)");
            return 1;
        } else {
            NXS_LOG_INFO("HEX20 SINGLE ELEMENT TEST: PASSED (no NaN)");
        }
        NXS_LOG_INFO("=================================================");

        return 0;

    } catch (const std::exception& e) {
        NXS_LOG_ERROR("Test failed with exception: {}", e.what());
        return 1;
    }
}
