/**
 * @file fem_solver_test.cpp
 * @brief Test FEM solver with a simple dynamics problem
 *
 * Tests:
 * - Single Hex8 element under gravity
 * - Explicit time integration
 * - Boundary conditions
 * - CFL-based time stepping
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/fem/fem_solver.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/io/vtk_writer.hpp>
#include <iostream>
#include <vector>

using namespace nxs;
using namespace nxs::fem;

int main() {
    // Initialize NexusSim
    nxs::InitOptions options;
    options.log_level = nxs::Logger::Level::Info;
    nxs::Context context(options);

    NXS_LOG_INFO("=================================================");
    NXS_LOG_INFO("FEM Solver Test - Single Element Dynamics");
    NXS_LOG_INFO("=================================================\n");

    try {
        // ====================================================================
        // Create mesh - Single Hex8 element (unit cube)
        // ====================================================================

        auto mesh = std::make_shared<Mesh>(8);  // 8 nodes

        // Set node coordinates (unit cube)
        mesh->set_node_coordinates(0, {0.0, 0.0, 0.0});
        mesh->set_node_coordinates(1, {1.0, 0.0, 0.0});
        mesh->set_node_coordinates(2, {1.0, 1.0, 0.0});
        mesh->set_node_coordinates(3, {0.0, 1.0, 0.0});
        mesh->set_node_coordinates(4, {0.0, 0.0, 1.0});
        mesh->set_node_coordinates(5, {1.0, 0.0, 1.0});
        mesh->set_node_coordinates(6, {1.0, 1.0, 1.0});
        mesh->set_node_coordinates(7, {0.0, 1.0, 1.0});

        NXS_LOG_INFO("Created mesh with {} nodes", mesh->num_nodes());

        // Add element block to mesh
        mesh->add_element_block("block1", ElementType::Hex8, 1, 8);
        auto& block = mesh->element_block(0);

        // Set connectivity for the single hex element
        for (Index i = 0; i < 8; ++i) {
            block.connectivity[i] = i;
        }

        NXS_LOG_INFO("Added element block with {} elements", block.num_elements());

        // ====================================================================
        // Create state
        // ====================================================================

        auto state = std::make_shared<State>(*mesh);

        // ====================================================================
        // Create FEM solver
        // ====================================================================

        FEMSolver solver("ExplicitDynamics");

        // Material properties (steel)
        physics::MaterialProperties steel;
        steel.density = 7850.0;          // kg/m³
        steel.E = 200.0e9;               // Pa (200 GPa)
        steel.nu = 0.3;                  // Poisson's ratio
        steel.G = steel.E / (2.0 * (1.0 + steel.nu));
        steel.K = steel.E / (3.0 * (1.0 - 2.0 * steel.nu));

        NXS_LOG_INFO("Material: Steel (E={:.2e} Pa, ρ={:.1f} kg/m³, ν={:.2f})",
                     steel.E, steel.density, steel.nu);

        // Element group (single Hex8 element)
        std::vector<Index> elem_ids = {0};
        std::vector<Index> connectivity = {0, 1, 2, 3, 4, 5, 6, 7};

        solver.add_element_group("block1", physics::ElementType::Hex8,
                                elem_ids, connectivity, steel);

        // ====================================================================
        // Boundary conditions
        // ====================================================================

        // Fix nodes at z=0 (nodes 0,1,2,3)
        std::vector<Index> fixed_nodes = {0, 1, 2, 3};

        // Fix all DOFs for these nodes
        for (Index dof = 0; dof < 3; ++dof) {
            BoundaryCondition bc_disp(BCType::Displacement, fixed_nodes, dof, 0.0);
            solver.add_boundary_condition(bc_disp);
        }

        NXS_LOG_INFO("Applied fixed BC on {} nodes", fixed_nodes.size());

        // Apply gravity force on all nodes
        std::vector<Index> all_nodes = {0, 1, 2, 3, 4, 5, 6, 7};
        const Real gravity = -9.81;  // m/s²
        const Real node_mass = steel.density * 1.0 / 8.0;  // Volume / 8 nodes
        const Real gravity_force = node_mass * gravity;

        BoundaryCondition bc_gravity(BCType::Force, all_nodes, 2, gravity_force);  // z-direction
        solver.add_boundary_condition(bc_gravity);

        NXS_LOG_INFO("Applied gravity force: {:.3e} N per node", gravity_force);

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
        const Real t_final = 0.001;  // 1 ms
        const int num_steps = static_cast<int>(t_final / dt);
        const int output_interval = std::max(1, num_steps / 10);  // Avoid division by zero

        NXS_LOG_INFO("\nStarting time integration:");
        NXS_LOG_INFO("  Time step: {:.6e} s", dt);
        NXS_LOG_INFO("  Final time: {:.6e} s", t_final);
        NXS_LOG_INFO("  Number of steps: {}\n", num_steps);

        // Create VTK writer for visualization
        io::VTKWriter vtk_writer("fem_test");
        vtk_writer.set_output_directory(".");

        // Write initial state
        vtk_writer.write_time_step(*mesh, *state, 0.0, 0);

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

            // Write VTK output every step
            vtk_writer.write_time_step(*mesh, *state, solver.current_time(), step + 1);

            // Output progress
            if (step % output_interval == 0 || step == num_steps - 1) {
                const auto& disp = solver.displacement();
                const auto& vel = solver.velocity();

                // Check displacement of top node (node 4)
                const Real ux = disp[4 * 3 + 0];
                const Real uy = disp[4 * 3 + 1];
                const Real uz = disp[4 * 3 + 2];
                const Real vmag = std::sqrt(vel[4 * 3 + 0] * vel[4 * 3 + 0] +
                                           vel[4 * 3 + 1] * vel[4 * 3 + 1] +
                                           vel[4 * 3 + 2] * vel[4 * 3 + 2]);

                NXS_LOG_INFO("Step {}/{}: t={:.6e} s, uz={:.6e} m, |v|={:.6e} m/s",
                            step, num_steps, solver.current_time(), uz, vmag);
            }
        }

        // ====================================================================
        // Finalize VTK output
        // ====================================================================

        vtk_writer.finalize_time_series();

        // ====================================================================
        // Final results
        // ====================================================================

        NXS_LOG_INFO("\n=================================================");
        NXS_LOG_INFO("Simulation Complete!");
        NXS_LOG_INFO("=================================================");

        const auto& disp = solver.displacement();
        const auto& vel = solver.velocity();

        NXS_LOG_INFO("\nFinal displacements (top nodes):");
        for (Index node : {4, 5, 6, 7}) {
            const Real ux = disp[node * 3 + 0];
            const Real uy = disp[node * 3 + 1];
            const Real uz = disp[node * 3 + 2];
            NXS_LOG_INFO("  Node {}: u = ({:.6e}, {:.6e}, {:.6e}) m",
                        node, ux, uy, uz);
        }

        NXS_LOG_INFO("\nFinal velocities (top nodes):");
        for (Index node : {4, 5, 6, 7}) {
            const Real vx = vel[node * 3 + 0];
            const Real vy = vel[node * 3 + 1];
            const Real vz = vel[node * 3 + 2];
            NXS_LOG_INFO("  Node {}: v = ({:.6e}, {:.6e}, {:.6e}) m/s",
                        node, vx, vy, vz);
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
