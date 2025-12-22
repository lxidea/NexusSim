/**
 * @file mesh_reader_test.cpp
 * @brief Test mesh reader with cantilever beam simulation
 *
 * Tests:
 * - Loading mesh from file
 * - Using node sets for boundary conditions
 * - Running FEM simulation
 * - Outputting VTK visualization
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/fem/fem_solver.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/io/mesh_reader.hpp>
#include <nexussim/io/vtk_writer.hpp>
#include <iostream>

using namespace nxs;
using namespace nxs::fem;

int main() {
    // Initialize NexusSim
    nxs::InitOptions options;
    options.log_level = nxs::Logger::Level::Info;
    nxs::Context context(options);

    NXS_LOG_INFO("=================================================");
    NXS_LOG_INFO("Mesh Reader Test - Cantilever Beam");
    NXS_LOG_INFO("=================================================\n");

    try {
        // ====================================================================
        // Load mesh from file
        // ====================================================================

        io::SimpleMeshReader reader;
        auto mesh = reader.read("../examples/meshes/cantilever_beam.mesh");

        NXS_LOG_INFO("Mesh loaded: {} nodes, {} element blocks",
                     mesh->num_nodes(), mesh->num_element_blocks());

        // Get node sets
        const auto& node_sets = reader.node_sets();

        NXS_LOG_INFO("\nNode sets:");
        for (const auto& [name, nodes] : node_sets) {
            NXS_LOG_INFO("  {}: {} nodes", name, nodes.size());
        }

        // ====================================================================
        // Create state
        // ====================================================================

        auto state = std::make_shared<State>(*mesh);

        // ====================================================================
        // Create FEM solver
        // ====================================================================

        FEMSolver solver("ExplicitDynamics");

        // Material properties (aluminum)
        physics::MaterialProperties aluminum;
        aluminum.density = 2700.0;          // kg/m³
        aluminum.E = 70.0e9;                // Pa (70 GPa)
        aluminum.nu = 0.33;                 // Poisson's ratio
        aluminum.G = aluminum.E / (2.0 * (1.0 + aluminum.nu));
        aluminum.K = aluminum.E / (3.0 * (1.0 - 2.0 * aluminum.nu));

        NXS_LOG_INFO("Material: Aluminum (E={:.2e} Pa, ρ={:.1f} kg/m³, ν={:.2f})",
                     aluminum.E, aluminum.density, aluminum.nu);

        // Add element group from loaded mesh
        auto& block = mesh->element_block(0);
        std::vector<Index> elem_ids(block.num_elements());
        for (std::size_t i = 0; i < block.num_elements(); ++i) {
            elem_ids[i] = i;
        }

        // Convert ElementType to physics::ElementType
        physics::ElementType phys_type = physics::ElementType::Hex8;  // Default to Hex8 for this example

        // Convert Field<Index> connectivity to std::vector<Index>
        std::vector<Index> connectivity_vec;
        const std::size_t conn_size = block.num_elements() * block.num_nodes_per_elem;
        connectivity_vec.reserve(conn_size);
        for (std::size_t i = 0; i < conn_size; ++i) {
            connectivity_vec.push_back(block.connectivity[i]);
        }

        solver.add_element_group(block.name, phys_type,
                                elem_ids, connectivity_vec, aluminum);

        // ====================================================================
        // Boundary conditions using node sets
        // ====================================================================

        // Fixed BC on clamped end (from "fixed" node set)
        if (node_sets.count("fixed")) {
            const auto& fixed_nodes = node_sets.at("fixed");
            for (Index dof = 0; dof < 3; ++dof) {
                BoundaryCondition bc_disp(BCType::Displacement, fixed_nodes, dof, 0.0);
                solver.add_boundary_condition(bc_disp);
            }
            NXS_LOG_INFO("Applied fixed BC on {} nodes", fixed_nodes.size());
        }

        // Apply downward force on free end (from "free" node set)
        if (node_sets.count("free")) {
            const auto& free_nodes = node_sets.at("free");
            const Real force_per_node = -1000.0;  // N (downward in z)

            BoundaryCondition bc_force(BCType::Force, free_nodes, 2, force_per_node);
            solver.add_boundary_condition(bc_force);

            NXS_LOG_INFO("Applied force BC: {:.1f} N on {} nodes",
                        force_per_node, free_nodes.size());
        }

        // ====================================================================
        // Initialize solver
        // ====================================================================

        solver.initialize(mesh, state);

        // Compute stable time step
        const Real dt_stable = solver.compute_stable_dt();
        NXS_LOG_INFO("Stable time step: {:.6e} s", dt_stable);

        // ====================================================================
        // Time integration with VTK output
        // ====================================================================

        const Real dt = dt_stable;
        const Real t_final = 0.002;  // 2 ms
        const int num_steps = static_cast<int>(t_final / dt);
        const int output_interval = std::max(1, num_steps / 10);

        NXS_LOG_INFO("\nStarting time integration:");
        NXS_LOG_INFO("  Time step: {:.6e} s", dt);
        NXS_LOG_INFO("  Final time: {:.6e} s", t_final);
        NXS_LOG_INFO("  Number of steps: {}\n", num_steps);

        // Create VTK writer
        io::VTKWriter vtk_writer("cantilever_test");
        vtk_writer.set_output_directory(".");

        // Write initial state
        vtk_writer.write_time_step(*mesh, *state, 0.0, 0);

        // Time integration loop
        for (int step = 0; step < num_steps; ++step) {
            solver.step(dt);

            // Update state with solver results
            const auto& disp = solver.displacement();
            const auto& vel = solver.velocity();
            const auto& acc = solver.acceleration();

            auto& state_disp = state->field("displacement");
            auto& state_vel = state->field("velocity");
            auto& state_acc = state->field("acceleration");

            for (std::size_t node = 0; node < mesh->num_nodes(); ++node) {
                for (int comp = 0; comp < 3; ++comp) {
                    state_disp.at(node, comp) = disp[node * 3 + comp];
                    state_vel.at(node, comp) = vel[node * 3 + comp];
                    state_acc.at(node, comp) = acc[node * 3 + comp];
                }
            }

            // Write VTK output
            if (step % output_interval == 0 || step == num_steps - 1) {
                vtk_writer.write_time_step(*mesh, *state, solver.current_time(), step + 1);

                // Get displacement of tip node (node 11 - top right corner)
                const Real ux = disp[11 * 3 + 0];
                const Real uy = disp[11 * 3 + 1];
                const Real uz = disp[11 * 3 + 2];
                const Real umag = std::sqrt(ux*ux + uy*uy + uz*uz);

                NXS_LOG_INFO("Step {}/{}: t={:.6e} s, tip displacement={:.6e} m",
                            step, num_steps, solver.current_time(), umag);
            }
        }

        // Finalize VTK output
        vtk_writer.finalize_time_series();

        // ====================================================================
        // Final results
        // ====================================================================

        NXS_LOG_INFO("\n=================================================");
        NXS_LOG_INFO("Simulation Complete!");
        NXS_LOG_INFO("=================================================");

        const auto& disp = solver.displacement();

        // Check tip deflection (nodes 2, 5, 8, 11 - free end)
        NXS_LOG_INFO("\nFree end displacements:");
        for (Index node : {2, 5, 8, 11}) {
            const Real ux = disp[node * 3 + 0];
            const Real uy = disp[node * 3 + 1];
            const Real uz = disp[node * 3 + 2];
            const Real umag = std::sqrt(ux*ux + uy*uy + uz*uz);
            NXS_LOG_INFO("  Node {}: u = ({:.6e}, {:.6e}, {:.6e}) m, |u| = {:.6e} m",
                        node, ux, uy, uz, umag);
        }

        NXS_LOG_INFO("\n=================================================");
        NXS_LOG_INFO("Test PASSED!");
        NXS_LOG_INFO("=================================================");

    } catch (const std::exception& e) {
        NXS_LOG_ERROR("Test failed with exception: {}", e.what());
        return 1;
    }

    return 0;
}
