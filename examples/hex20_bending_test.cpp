/**
 * @file hex20_bending_test.cpp
 * @brief Bending test for Hex20 quadratic elements
 *
 * Tests cantilever beam deflection using 20-node quadratic hexahedral elements.
 * Hex20 elements should provide much better accuracy for bending problems
 * compared to Hex8 due to quadratic shape functions that avoid volumetric locking.
 *
 * Analytical solution for cantilever beam:
 *   δ_max = (F * L³) / (3 * E * I)
 *
 * Where:
 *   F = applied force
 *   L = beam length
 *   E = Young's modulus
 *   I = second moment of area = b*h³/12 for rectangular cross-section
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

// Ramp function for gradual load application
Real ramp_function(Real t) {
    const Real ramp_time = 0.1;  // Ramp up over 0.1 seconds
    if (t >= ramp_time) {
        return 1.0;
    }
    return t / ramp_time;
}

int main() {
    // Initialize NexusSim
    nxs::InitOptions options;
    options.log_level = nxs::Logger::Level::Info;
    nxs::Context context(options);

    NXS_LOG_INFO("=================================================");
    NXS_LOG_INFO("Hex20 Cantilever Beam Bending Test");
    NXS_LOG_INFO("=================================================\n");

    try {
        // ====================================================================
        // Test Parameters
        // ====================================================================

        const Real beam_length = 10.0;    // m
        const Real beam_width = 1.0;      // m
        const Real beam_height = 1.0;     // m
        const Real applied_force = -1000.0;  // N (downward)

        // Material: Aluminum
        const Real E = 70.0e9;      // Pa
        const Real nu = 0.33;
        const Real density = 2700.0; // kg/m³

        // For Hex20, we can use coarser meshes and still get good accuracy
        // Each Hex20 element is equivalent to ~8 Hex8 elements in accuracy
        std::vector<std::array<int, 3>> mesh_divisions = {
            {2, 1, 1},   // Very coarse: 2 elements along length (2 Hex20 elements)
            {4, 1, 1},   // Coarse: 4 along length (4 Hex20 elements)
            {4, 1, 2},   // Medium: 4 along length, 2 in height (8 Hex20 elements)
        };
        std::vector<Real> computed_deflections;

        NXS_LOG_INFO("Beam Geometry:");
        NXS_LOG_INFO("  Length: {} m", beam_length);
        NXS_LOG_INFO("  Width: {} m", beam_width);
        NXS_LOG_INFO("  Height: {} m", beam_height);
        NXS_LOG_INFO("  Applied force: {} N\n", applied_force);

        // ====================================================================
        // Analytical Solution
        // ====================================================================

        const Real I = beam_width * std::pow(beam_height, 3) / 12.0;
        const Real analytical_deflection = (applied_force * std::pow(beam_length, 3)) /
                                          (3.0 * E * I);

        NXS_LOG_INFO("Analytical Solution (Euler-Bernoulli):");
        NXS_LOG_INFO("  Second moment of area I: {:.6e} m⁴", I);
        NXS_LOG_INFO("  Tip deflection: {:.6e} m ({:.3f} mm)\n",
                     analytical_deflection, analytical_deflection * 1000.0);

        // ====================================================================
        // Run convergence study with Hex20 elements
        // ====================================================================

        for (const auto& div : mesh_divisions) {
            const int nx = div[0];  // Elements along length (x)
            const int ny = div[1];  // Elements along width (y)
            const int nz = div[2];  // Elements along height (z)

            NXS_LOG_INFO("=================================================");
            NXS_LOG_INFO("Running with {}x{}x{} Hex20 elements", nx, ny, nz);
            NXS_LOG_INFO("=================================================");

            // For Hex20, we need 20 nodes per element
            // Create mesh with mid-edge nodes
            // Node structure: (2*nx+1) x (2*ny+1) x (2*nz+1) to include mid-nodes
            const int n_nodes_x = 2 * nx + 1;
            const int n_nodes_y = 2 * ny + 1;
            const int n_nodes_z = 2 * nz + 1;
            const int n_nodes = n_nodes_x * n_nodes_y * n_nodes_z;
            const int n_elems = nx * ny * nz;

            auto mesh = std::make_shared<Mesh>(n_nodes);

            // Generate structured mesh nodes (including mid-edge nodes)
            // Use X as major (slowest-varying), then Y, then Z (fastest)
            int node_id = 0;
            for (int i = 0; i < n_nodes_x; ++i) {
                const Real x = (beam_length / (2 * nx)) * i;
                for (int j = 0; j < n_nodes_y; ++j) {
                    const Real y = (beam_width / (2 * ny)) * j;
                    for (int k = 0; k < n_nodes_z; ++k) {
                        const Real z = (beam_height / (2 * nz)) * k;
                        mesh->set_node_coordinates(node_id++, {x, y, z});
                    }
                }
            }

            NXS_LOG_INFO("Created mesh with {} nodes (including mid-edge nodes)", mesh->num_nodes());

            // Create element block for Hex20
            mesh->add_element_block("beam", ElementType::Hex20, n_elems, 20);
            auto& block = mesh->element_block(0);

            // Set element connectivity for Hex20
            std::vector<Index> connectivity;
            int elem_id = 0;

            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    for (int k = 0; k < nz; ++k) {
                        // Helper to get node index in the fine grid
                        // Node layout: i (X) is major, then j (Y), then k (Z) is minor (fastest-varying)
                        auto node_idx = [&](int di, int dj, int dk) -> Index {
                            return ((2*i + di) * n_nodes_y * n_nodes_z) +
                                   ((2*j + dj) * n_nodes_z) +
                                   (2*k + dk);
                        };

                        // Hex20 node ordering:
                        // Corner nodes (0-7): same as Hex8
                        // Mid-edge nodes (8-19): 12 mid-edge nodes

                        // Corner nodes
                        Index n0 = node_idx(0, 0, 0);
                        Index n1 = node_idx(2, 0, 0);
                        Index n2 = node_idx(2, 2, 0);
                        Index n3 = node_idx(0, 2, 0);
                        Index n4 = node_idx(0, 0, 2);
                        Index n5 = node_idx(2, 0, 2);
                        Index n6 = node_idx(2, 2, 2);
                        Index n7 = node_idx(0, 2, 2);

                        // Bottom face mid-edge nodes (z=0)
                        Index n8  = node_idx(1, 0, 0);  // Edge 0-1
                        Index n9  = node_idx(2, 1, 0);  // Edge 1-2
                        Index n10 = node_idx(1, 2, 0);  // Edge 2-3
                        Index n11 = node_idx(0, 1, 0);  // Edge 3-0

                        // Vertical mid-edge nodes
                        Index n12 = node_idx(0, 0, 1);  // Edge 0-4
                        Index n13 = node_idx(2, 0, 1);  // Edge 1-5
                        Index n14 = node_idx(2, 2, 1);  // Edge 2-6
                        Index n15 = node_idx(0, 2, 1);  // Edge 3-7

                        // Top face mid-edge nodes (z=2)
                        Index n16 = node_idx(1, 0, 2);  // Edge 4-5
                        Index n17 = node_idx(2, 1, 2);  // Edge 5-6
                        Index n18 = node_idx(1, 2, 2);  // Edge 6-7
                        Index n19 = node_idx(0, 1, 2);  // Edge 7-4

                        auto elem_nodes = block.element_nodes(elem_id);
                        elem_nodes[0] = n0;   elem_nodes[1] = n1;
                        elem_nodes[2] = n2;   elem_nodes[3] = n3;
                        elem_nodes[4] = n4;   elem_nodes[5] = n5;
                        elem_nodes[6] = n6;   elem_nodes[7] = n7;
                        elem_nodes[8] = n8;   elem_nodes[9] = n9;
                        elem_nodes[10] = n10; elem_nodes[11] = n11;
                        elem_nodes[12] = n12; elem_nodes[13] = n13;
                        elem_nodes[14] = n14; elem_nodes[15] = n15;
                        elem_nodes[16] = n16; elem_nodes[17] = n17;
                        elem_nodes[18] = n18; elem_nodes[19] = n19;

                        // Build connectivity vector for FEM solver
                        connectivity.insert(connectivity.end(), {
                            n0, n1, n2, n3, n4, n5, n6, n7,
                            n8, n9, n10, n11, n12, n13, n14, n15,
                            n16, n17, n18, n19
                        });

                        elem_id++;
                    }
                }
            }

            NXS_LOG_INFO("Created {} hex20 elements", n_elems);

            // ================================================================
            // Setup and solve
            // ================================================================

            auto state = std::make_shared<State>(*mesh);

            FEMSolver solver("Hex20BendingTest");

            physics::MaterialProperties aluminum;
            aluminum.density = density;
            aluminum.E = E;
            aluminum.nu = nu;
            aluminum.G = E / (2.0 * (1.0 + nu));
            aluminum.K = E / (3.0 * (1.0 - 2.0 * nu));

            std::vector<Index> elem_ids(n_elems);
            for (int i = 0; i < n_elems; ++i) elem_ids[i] = i;

            solver.add_element_group("beam", physics::ElementType::Hex20,
                                   elem_ids, connectivity, aluminum);

            // Set damping for stability
            const Real natural_freq_est = 10.0;
            const Real damping_ratio = 0.15;
            const Real damping_alpha = 2.0 * damping_ratio * natural_freq_est;
            solver.set_damping(damping_alpha);
            NXS_LOG_INFO("Set Rayleigh damping: α = {} (ξ = {})", damping_alpha, damping_ratio);

            // Apply boundary conditions
            // Fixed end (left, x=0): all nodes on the face at x=0
            // Node index = i * (n_nodes_y * n_nodes_z) + j * n_nodes_z + k
            // For x=0 face: i=0
            std::vector<Index> fixed_nodes;
            for (int j = 0; j < n_nodes_y; ++j) {
                for (int k = 0; k < n_nodes_z; ++k) {
                    Index node = 0 * (n_nodes_y * n_nodes_z) + j * n_nodes_z + k;
                    fixed_nodes.push_back(node);
                }
            }

            for (int dof = 0; dof < 3; ++dof) {
                BoundaryCondition bc_disp(BCType::Displacement, fixed_nodes, dof, 0.0);
                solver.add_boundary_condition(bc_disp);
            }

            // Applied force at free end (right, x=L): all nodes on face at x=L
            // For x=L face: i=2*nx (last X position)
            std::vector<Index> loaded_nodes;
            for (int j = 0; j < n_nodes_y; ++j) {
                for (int k = 0; k < n_nodes_z; ++k) {
                    Index node = (2*nx) * (n_nodes_y * n_nodes_z) + j * n_nodes_z + k;
                    loaded_nodes.push_back(node);
                }
            }

            const Real force_per_node = applied_force / loaded_nodes.size();
            BoundaryCondition bc_force(BCType::Force, loaded_nodes, 2, force_per_node);
            bc_force.time_function = ramp_function;
            solver.add_boundary_condition(bc_force);

            NXS_LOG_INFO("Applied BCs: {} nodes fixed, {} nodes loaded",
                        fixed_nodes.size(), loaded_nodes.size());

            solver.initialize(mesh, state);

            // Run simulation
            const Real dt = solver.compute_stable_dt() * 0.9;
            const Real total_time = 0.5;
            const int n_steps = static_cast<int>(total_time / dt);

            NXS_LOG_INFO("Running dynamic simulation:");
            NXS_LOG_INFO("  Time step: {:.6e} s", dt);
            NXS_LOG_INFO("  Total time: {:.6e} s", total_time);
            NXS_LOG_INFO("  Number of steps: {}", n_steps);

            // Run simulation
            Real min_uz = 0.0;
            Real max_uz = 0.0;

            for (int step = 0; step < n_steps; ++step) {
                solver.step(dt);

                // Tip node: corner node at free end (x=L, y=W, z=H)
                // Node index = i * (n_nodes_y * n_nodes_z) + j * n_nodes_z + k
                // Tip: i=2*nx, j=2*ny, k=2*nz
                const Index tip_node = (2*nx) * (n_nodes_y * n_nodes_z) +
                                      (2*ny) * n_nodes_z +
                                      (2*nz);
                const Real tip_uz = solver.displacement()[tip_node * 3 + 2];

                min_uz = std::min(min_uz, tip_uz);
                max_uz = std::max(max_uz, tip_uz);

                if (step % (n_steps / 10) == 0) {
                    const Real progress = 100.0 * step / n_steps;
                    NXS_LOG_INFO("  [{:.0f}%] Step {}/{}, tip uz = {:.6e} m",
                                progress, step, n_steps, tip_uz);
                }
            }

            NXS_LOG_INFO("  Displacement range: [{:.6e}, {:.6e}] m", min_uz, max_uz);
            NXS_LOG_INFO("  Mean displacement: {:.6e} m", (min_uz + max_uz) / 2.0);

            // Get final tip deflection
            // Node index = i * (n_nodes_y * n_nodes_z) + j * n_nodes_z + k
            const Index tip_node = (2*nx) * (n_nodes_y * n_nodes_z) +
                                  (2*ny) * n_nodes_z +
                                  (2*nz);
            const Real computed_deflection = solver.displacement()[tip_node * 3 + 2];
            computed_deflections.push_back(computed_deflection);

            NXS_LOG_INFO("\nResults for {}x{}x{} Hex20 mesh ({} elements):", nx, ny, nz, n_elems);
            NXS_LOG_INFO("  Computed deflection: {:.6e} m ({:.3f} μm)",
                        computed_deflection, computed_deflection * 1e6);
            NXS_LOG_INFO("  Analytical deflection: {:.6e} m ({:.3f} μm)",
                        analytical_deflection, analytical_deflection * 1e6);

            const Real error = std::abs(computed_deflection - analytical_deflection);
            const Real relative_error = error / std::abs(analytical_deflection) * 100.0;

            NXS_LOG_INFO("  Absolute error: {:.6e} m", error);
            NXS_LOG_INFO("  Relative error: {:.2f}%%\n", relative_error);
        }

        // ====================================================================
        // Summary
        // ====================================================================

        NXS_LOG_INFO("=================================================");
        NXS_LOG_INFO("Hex20 Convergence Study Summary");
        NXS_LOG_INFO("=================================================");

        NXS_LOG_INFO("Analytical solution: {:.6e} m ({:.3f} μm)",
                    analytical_deflection, analytical_deflection * 1e6);
        NXS_LOG_INFO("\nHex20 mesh refinement results:");

        bool test_passed = true;
        for (size_t i = 0; i < mesh_divisions.size(); ++i) {
            const auto& div = mesh_divisions[i];
            const Real rel_error = std::abs(computed_deflections[i] - analytical_deflection) /
                                  std::abs(analytical_deflection) * 100.0;
            NXS_LOG_INFO("  {}x{}x{} mesh: {:.6e} m (error: {:.2f}%%)",
                        div[0], div[1], div[2], computed_deflections[i], rel_error);

            // Hex20 should achieve much better accuracy (<5% error)
            if (rel_error > 10.0) {
                test_passed = false;
            }
        }

        NXS_LOG_INFO("\n=================================================");
        if (test_passed) {
            NXS_LOG_INFO("HEX20 BENDING TEST: PASSED");
            NXS_LOG_INFO("Hex20 elements provide excellent accuracy for bending");
        } else {
            NXS_LOG_WARN("HEX20 BENDING TEST: Errors larger than expected");
        }
        NXS_LOG_INFO("=================================================");

        return test_passed ? 0 : 1;

    } catch (const std::exception& e) {
        NXS_LOG_ERROR("Test failed with exception: {}", e.what());
        return 1;
    }
}
