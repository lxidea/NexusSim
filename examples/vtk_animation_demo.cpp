/**
 * @file vtk_animation_demo.cpp
 * @brief Demonstrates VTK time series output for ParaView animation
 *
 * Creates a cantilever beam simulation and outputs VTK files at regular
 * intervals to visualize the dynamic response in ParaView.
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/fem/fem_solver.hpp>
#include <nexussim/io/vtk_writer.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/data/state.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sys/stat.h>

using namespace nxs;
using namespace nxs::fem;

int main(int argc, char* argv[]) {
    // Initialize NexusSim
    InitOptions options;
    options.log_level = Logger::Level::Info;
    options.enable_mpi = false;
    options.enable_gpu = true;

    Context context(&argc, &argv, options);

    std::cout << "=================================================" << std::endl;
    std::cout << "VTK Animation Demo - Cantilever Beam Impact" << std::endl;
    std::cout << "=================================================" << std::endl;

    // Create output directory
    mkdir("vtk_output", 0755);

    // Mesh parameters
    const int nx = 20, ny = 4, nz = 4;  // 320 elements
    const Real Lx = 1.0, Ly = 0.1, Lz = 0.1;  // 1m x 10cm x 10cm beam
    const Real dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;

    const int num_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    const int num_elements = nx * ny * nz;

    std::cout << "\nMesh:" << std::endl;
    std::cout << "  Elements: " << num_elements << " Hex8" << std::endl;
    std::cout << "  Nodes: " << num_nodes << std::endl;
    std::cout << "  Dimensions: " << Lx << " x " << Ly << " x " << Lz << " m" << std::endl;

    // Create mesh
    auto mesh = std::make_shared<Mesh>(num_nodes);

    int node_id = 0;
    for (int i = 0; i <= nx; ++i) {
        for (int j = 0; j <= ny; ++j) {
            for (int k = 0; k <= nz; ++k) {
                mesh->set_node_coordinates(node_id++, {i*dx, j*dy, k*dz});
            }
        }
    }

    // Create element block
    mesh->add_element_block("beam", ElementType::Hex8, num_elements, 8);
    auto& block = mesh->element_block(0);

    std::vector<Index> connectivity;
    int elem_id = 0;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                auto node_idx = [&](int di, int dj, int dk) -> Index {
                    return (i+di)*(ny+1)*(nz+1) + (j+dj)*(nz+1) + (k+dk);
                };

                auto elem_nodes = block.element_nodes(elem_id);
                elem_nodes[0] = node_idx(0,0,0);
                elem_nodes[1] = node_idx(1,0,0);
                elem_nodes[2] = node_idx(1,1,0);
                elem_nodes[3] = node_idx(0,1,0);
                elem_nodes[4] = node_idx(0,0,1);
                elem_nodes[5] = node_idx(1,0,1);
                elem_nodes[6] = node_idx(1,1,1);
                elem_nodes[7] = node_idx(0,1,1);

                for (int n = 0; n < 8; ++n) {
                    connectivity.push_back(elem_nodes[n]);
                }
                elem_id++;
            }
        }
    }

    // Create state
    auto state = std::make_shared<State>(*mesh);

    // Setup solver
    FEMSolver solver("CantileverBeam");

    // Material: Steel
    physics::MaterialProperties steel;
    steel.density = 7850.0;  // kg/m³
    steel.E = 210.0e9;       // Pa
    steel.nu = 0.3;
    steel.G = steel.E / (2.0 * (1.0 + steel.nu));
    steel.K = steel.E / (3.0 * (1.0 - 2.0 * steel.nu));

    std::vector<Index> elem_ids(num_elements);
    for (int i = 0; i < num_elements; ++i) elem_ids[i] = i;

    solver.add_element_group("beam", physics::ElementType::Hex8,
                            elem_ids, connectivity, steel);

    // Fix left end (x = 0)
    std::vector<Index> fixed_nodes;
    for (int j = 0; j <= ny; ++j) {
        for (int k = 0; k <= nz; ++k) {
            fixed_nodes.push_back(0*(ny+1)*(nz+1) + j*(nz+1) + k);
        }
    }

    for (int dof = 0; dof < 3; ++dof) {
        BoundaryCondition bc(BCType::Displacement, fixed_nodes, dof, 0.0);
        solver.add_boundary_condition(bc);
    }

    // Apply impulse load at free end (will be applied for first few steps)
    std::vector<Index> loaded_nodes;
    for (int j = 0; j <= ny; ++j) {
        for (int k = 0; k <= nz; ++k) {
            loaded_nodes.push_back(nx*(ny+1)*(nz+1) + j*(nz+1) + k);
        }
    }

    const Real total_force = -50000.0;  // 50 kN downward
    const Real force_per_node = total_force / loaded_nodes.size();

    // Initialize solver
    solver.initialize(mesh, state);

    // Time stepping parameters
    const Real dt = solver.compute_stable_dt() * 0.8;
    const Real t_end = 0.01;  // 10 ms simulation
    const int output_interval = 10;  // Output every 10 steps
    const Real load_duration = 0.001;  // Load applied for 1 ms

    std::cout << "\nSimulation:" << std::endl;
    std::cout << "  Time step: " << dt * 1e6 << " µs" << std::endl;
    std::cout << "  End time: " << t_end * 1000 << " ms" << std::endl;
    std::cout << "  Load: " << std::abs(total_force) / 1000 << " kN for " << load_duration * 1000 << " ms" << std::endl;
    std::cout << "  Output interval: every " << output_interval << " steps" << std::endl;

    // VTK writer for time series
    io::VTKWriter vtk_writer("cantilever");
    vtk_writer.set_output_directory("vtk_output");

    // Run simulation
    int step = 0;
    int output_count = 0;
    Real time = 0.0;

    // Track tip displacement
    const Index tip_node = nx*(ny+1)*(nz+1) + (ny/2)*(nz+1) + nz/2;

    std::cout << "\nRunning simulation..." << std::endl;
    std::cout << std::setw(10) << "Step" << std::setw(15) << "Time (ms)"
              << std::setw(18) << "Tip Disp (mm)" << std::endl;
    std::cout << std::string(43, '-') << std::endl;

    // Get reference to force field
    auto& force_field = state->force();

    while (time < t_end) {
        // Apply/remove load
        if (time < load_duration) {
            // Apply impulse load
            for (auto node : loaded_nodes) {
                force_field.at(node, 2) = force_per_node;
            }
        } else {
            // Remove load
            for (auto node : loaded_nodes) {
                force_field.at(node, 2) = 0.0;
            }
        }

        // Time step
        solver.step(dt);
        time += dt;
        step++;

        // Output VTK
        if (step % output_interval == 0) {
            vtk_writer.write_time_step(*mesh, *state, time, output_count);
            output_count++;

            // Print progress
            const auto& disp_field = state->displacement();
            Real tip_disp = disp_field.at(tip_node, 2) * 1000.0;  // mm
            std::cout << std::setw(10) << step
                      << std::setw(15) << std::fixed << std::setprecision(3) << time * 1000
                      << std::setw(18) << std::setprecision(4) << tip_disp << std::endl;
        }
    }

    // Finalize time series (create .pvd file)
    vtk_writer.finalize_time_series();

    std::cout << "\n=================================================" << std::endl;
    std::cout << "Simulation complete!" << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "\nOutput files:" << std::endl;
    std::cout << "  VTK files: vtk_output/cantilever_*.vtk" << std::endl;
    std::cout << "  PVD file: vtk_output/cantilever.pvd" << std::endl;
    std::cout << "\nTo view animation in ParaView:" << std::endl;
    std::cout << "  1. Open ParaView" << std::endl;
    std::cout << "  2. File -> Open -> vtk_output/cantilever.pvd" << std::endl;
    std::cout << "  3. Click 'Apply'" << std::endl;
    std::cout << "  4. Use the play button to animate" << std::endl;
    std::cout << "  5. Color by 'displacement_magnitude' or 'velocity_magnitude'" << std::endl;

    return 0;
}
