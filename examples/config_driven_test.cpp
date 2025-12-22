/**
 * @file config_driven_test.cpp
 * @brief Test config-driven simulation workflow
 *
 * Tests:
 * - Reading configuration from YAML file
 * - Setting up simulation from config
 * - Running FEM solver with config parameters
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/fem/fem_solver.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/io/config_reader.hpp>
#include <nexussim/io/mesh_reader.hpp>
#include <nexussim/io/vtk_writer.hpp>
#include <iostream>

using namespace nxs;
using namespace nxs::fem;

int main(int argc, char** argv) {
    // Initialize NexusSim
    nxs::InitOptions options;
    options.log_level = nxs::Logger::Level::Info;
    nxs::Context context(options);

    NXS_LOG_INFO("=================================================");
    NXS_LOG_INFO("Config-Driven Simulation Test");
    NXS_LOG_INFO("=================================================\n");

    try {
        // ====================================================================
        // Read configuration file
        // ====================================================================

        std::string config_file = "../examples/configs/cantilever.yaml";
        if (argc > 1) {
            config_file = argv[1];
        }

        io::ConfigReader config_reader;
        auto config = config_reader.read(config_file);

        // ====================================================================
        // Parse simulation parameters
        // ====================================================================

        auto& sim_config = config.subsection("simulation");
        std::string sim_name = sim_config.get_string("name", "simulation");
        Real t_final = sim_config.get_real("final_time", 0.001);
        Real cfl_factor = sim_config.get_real("cfl_factor", 0.9);

        NXS_LOG_INFO("Simulation: {}", sim_name);
        NXS_LOG_INFO("Description: {}", sim_config.get_string("description", "N/A"));
        NXS_LOG_INFO("Final time: {:.6e} s", t_final);
        NXS_LOG_INFO("CFL factor: {:.2f}\n", cfl_factor);

        // ====================================================================
        // Load mesh
        // ====================================================================

        auto& mesh_config = config.subsection("mesh");
        std::string mesh_file = mesh_config.get_string("file", "../examples/meshes/cantilever_beam.mesh");

        NXS_LOG_INFO("Mesh file from config: '{}'", mesh_file);

        io::SimpleMeshReader reader;
        auto mesh = reader.read(mesh_file);

        NXS_LOG_INFO("Mesh loaded: {} nodes, {} element blocks\n",
                     mesh->num_nodes(), mesh->num_element_blocks());

        // ====================================================================
        // Create state
        // ====================================================================

        auto state = std::make_shared<State>(*mesh);

        // ====================================================================
        // Parse materials
        // ====================================================================

        auto& materials_config = config.subsection("materials");

        std::map<std::string, physics::MaterialProperties> materials;

        for (const auto& mat_name : materials_config.subsection_names()) {
            auto& mat_config = materials_config.subsection(mat_name);

            physics::MaterialProperties mat;
            mat.density = mat_config.get_real("density", 1000.0);
            mat.E = mat_config.get_real("E", 1.0e9);
            mat.nu = mat_config.get_real("nu", 0.3);
            mat.G = mat.E / (2.0 * (1.0 + mat.nu));
            mat.K = mat.E / (3.0 * (1.0 - 2.0 * mat.nu));

            materials[mat_name] = mat;

            NXS_LOG_INFO("Material '{}': E={:.2e} Pa, ρ={:.1f} kg/m³, ν={:.2f}",
                        mat_name, mat.E, mat.density, mat.nu);
        }

        // ====================================================================
        // Create FEM solver
        // ====================================================================

        FEMSolver solver("ExplicitDynamics");
        solver.set_cfl_factor(cfl_factor);

        // Add element groups from config
        auto& block = mesh->element_block(0);
        std::vector<Index> elem_ids(block.num_elements());
        for (std::size_t i = 0; i < block.num_elements(); ++i) {
            elem_ids[i] = i;
        }

        // Get material name from config (default to first material)
        std::string mat_name = materials.begin()->first;

        // Convert connectivity
        std::vector<Index> connectivity_vec;
        const std::size_t conn_size = block.num_elements() * block.num_nodes_per_elem;
        connectivity_vec.reserve(conn_size);
        for (std::size_t i = 0; i < conn_size; ++i) {
            connectivity_vec.push_back(block.connectivity[i]);
        }

        solver.add_element_group(block.name, physics::ElementType::Hex8,
                                elem_ids, connectivity_vec, materials[mat_name]);

        // ====================================================================
        // Apply boundary conditions from config
        // ====================================================================

        const auto& node_sets = reader.node_sets();

        NXS_LOG_INFO("Root config subsections:");
        for (const auto& name : config.subsection_names()) {
            NXS_LOG_INFO("  - {}", name);
            auto& subsec = config.subsection(name);
            for (const auto& subname : subsec.subsection_names()) {
                NXS_LOG_INFO("    - {}/{}", name, subname);
            }
        }

        if (!config.has_subsection("boundary_conditions")) {
            NXS_LOG_ERROR("ERROR: No boundary_conditions subsection found!");
            return 1;
        }

        auto& bc_config = config.subsection("boundary_conditions");

        NXS_LOG_INFO("Boundary conditions subsections: {}",
                     bc_config.subsection_names().size());

        // Displacement BCs
        if (bc_config.has_subsection("displacement")) {
            auto& disp_bc_config = bc_config.subsection("displacement");

            NXS_LOG_INFO("Displacement BC subsections: {}",
                         disp_bc_config.subsection_names().size());

            // Iterate through numbered list items (0, 1, 2, ...)
            for (const auto& bc_name : disp_bc_config.subsection_names()) {
                NXS_LOG_INFO("  Processing BC item: '{}'", bc_name);
                auto& bc = disp_bc_config.subsection(bc_name);

                std::string nodeset = bc.get_string("nodeset");
                NXS_LOG_INFO("    nodeset: '{}'", nodeset);
                auto dof_array = bc.get_int_array("dof");
                NXS_LOG_INFO("    dof_array size: {}", dof_array.size());
                Real value = bc.get_real("value", 0.0);
                NXS_LOG_INFO("    value: {}", value);

                if (node_sets.count(nodeset)) {
                    const auto& nodes = node_sets.at(nodeset);

                    for (Int dof : dof_array) {
                        BoundaryCondition bc_disp(BCType::Displacement, nodes, dof, value);
                        solver.add_boundary_condition(bc_disp);
                    }

                    NXS_LOG_INFO("Applied displacement BC on nodeset '{}': {} nodes, DOFs: {}",
                                nodeset, nodes.size(), dof_array.size());
                }
            }
        }

        // Force BCs
        if (bc_config.has_subsection("force")) {
            auto& force_bc_config = bc_config.subsection("force");

            // Iterate through numbered list items (0, 1, 2, ...)
            for (const auto& bc_name : force_bc_config.subsection_names()) {
                auto& bc = force_bc_config.subsection(bc_name);

                std::string nodeset = bc.get_string("nodeset");
                Int dof = bc.get_int("dof", 2);
                Real value = bc.get_real("value", 0.0);

                if (node_sets.count(nodeset)) {
                    const auto& nodes = node_sets.at(nodeset);

                    BoundaryCondition bc_force(BCType::Force, nodes, dof, value);
                    solver.add_boundary_condition(bc_force);

                    NXS_LOG_INFO("Applied force BC on nodeset '{}': {:.1f} N on {} nodes (DOF {})",
                                nodeset, value, nodes.size(), dof);
                }
            }
        }

        // ====================================================================
        // Initialize solver
        // ====================================================================

        solver.initialize(mesh, state);

        const Real dt_stable = solver.compute_stable_dt();
        NXS_LOG_INFO("Stable time step: {:.6e} s\n", dt_stable);

        // ====================================================================
        // Parse output configuration
        // ====================================================================

        if (!config.has_subsection("output")) {
            NXS_LOG_ERROR("ERROR: No output subsection found!");
            return 1;
        }

        auto& output_config = config.subsection("output");
        std::string output_format = output_config.get_string("format", "vtk");
        int output_freq = output_config.get_int("frequency", 1);
        std::string output_dir = output_config.get_string("directory", ".");
        std::string base_name = output_config.get_string("base_name", sim_name);

        // ====================================================================
        // Time integration
        // ====================================================================

        const Real dt = dt_stable;
        const int num_steps = static_cast<int>(t_final / dt);

        NXS_LOG_INFO("Starting time integration:");
        NXS_LOG_INFO("  Time step: {:.6e} s", dt);
        NXS_LOG_INFO("  Final time: {:.6e} s", t_final);
        NXS_LOG_INFO("  Number of steps: {}\n", num_steps);

        // Create VTK writer
        io::VTKWriter vtk_writer(base_name);
        vtk_writer.set_output_directory(output_dir);

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
            if ((step + 1) % output_freq == 0 || step == num_steps - 1) {
                vtk_writer.write_time_step(*mesh, *state, solver.current_time(), step + 1);

                // Output progress
                const Real uz = disp[11 * 3 + 2];  // Tip node displacement
                NXS_LOG_INFO("Step {}/{}: t={:.6e} s, tip uz={:.6e} m",
                            step + 1, num_steps, solver.current_time(), uz);
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
        NXS_LOG_INFO("Test PASSED!");
        NXS_LOG_INFO("=================================================");

    } catch (const std::exception& e) {
        NXS_LOG_ERROR("Test failed with exception: {}", e.what());
        return 1;
    }

    return 0;
}
