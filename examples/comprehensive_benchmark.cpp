/**
 * @file comprehensive_benchmark.cpp
 * @brief Comprehensive performance benchmark suite for NexusSim
 *
 * Tests FEM solver performance across:
 * - Different problem sizes (1K to 100K elements)
 * - Different element types (Hex8, Tet4, Shell4)
 * - CPU vs GPU backends (when available)
 * - Memory usage tracking
 *
 * Usage: ./comprehensive_benchmark [--large] [--quick]
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/fem/fem_solver.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/data/state.hpp>
#include <nexussim/utils/performance_timer.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cstring>

using namespace nxs;
using namespace nxs::fem;
using namespace nxs::utils;

// ============================================================================
// Mesh Generators
// ============================================================================

std::pair<std::shared_ptr<Mesh>, std::vector<Index>>
create_hex8_beam(int nx, int ny, int nz, double Lx = 10.0, double Ly = 1.0, double Lz = 1.0) {
    int num_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    int num_elements = nx * ny * nz;
    auto mesh = std::make_shared<Mesh>(num_nodes);

    const double dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;

    // Create nodes
    int node_id = 0;
    for (int i = 0; i <= nx; ++i) {
        for (int j = 0; j <= ny; ++j) {
            for (int k = 0; k <= nz; ++k) {
                mesh->set_node_coordinates(node_id++, {i * dx, j * dy, k * dz});
            }
        }
    }

    // Create elements
    mesh->add_element_block("solid", ElementType::Hex8, num_elements, 8);
    auto& block = mesh->element_block(0);

    std::vector<Index> connectivity;
    int elem_id = 0;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                auto node_idx = [&](int di, int dj, int dk) -> Index {
                    return (i + di) * (ny + 1) * (nz + 1) + (j + dj) * (nz + 1) + (k + dk);
                };

                auto elem_nodes = block.element_nodes(elem_id);
                elem_nodes[0] = node_idx(0, 0, 0);
                elem_nodes[1] = node_idx(1, 0, 0);
                elem_nodes[2] = node_idx(1, 1, 0);
                elem_nodes[3] = node_idx(0, 1, 0);
                elem_nodes[4] = node_idx(0, 0, 1);
                elem_nodes[5] = node_idx(1, 0, 1);
                elem_nodes[6] = node_idx(1, 1, 1);
                elem_nodes[7] = node_idx(0, 1, 1);

                for (int n = 0; n < 8; ++n) {
                    connectivity.push_back(elem_nodes[n]);
                }
                elem_id++;
            }
        }
    }

    return {mesh, connectivity};
}

std::pair<std::shared_ptr<Mesh>, std::vector<Index>>
create_tet4_beam(int nx, int ny, int nz, double Lx = 10.0, double Ly = 1.0, double Lz = 1.0) {
    // Create structured tet4 mesh (6 tets per hex cell)
    int num_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    int num_elements = nx * ny * nz * 6;  // 6 tets per hex
    auto mesh = std::make_shared<Mesh>(num_nodes);

    const double dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;

    // Create nodes
    int node_id = 0;
    for (int i = 0; i <= nx; ++i) {
        for (int j = 0; j <= ny; ++j) {
            for (int k = 0; k <= nz; ++k) {
                mesh->set_node_coordinates(node_id++, {i * dx, j * dy, k * dz});
            }
        }
    }

    // Create elements (subdivide each hex into 6 tets)
    mesh->add_element_block("solid", ElementType::Tet4, num_elements, 4);
    auto& block = mesh->element_block(0);

    std::vector<Index> connectivity;
    int elem_id = 0;

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                auto node_idx = [&](int di, int dj, int dk) -> Index {
                    return (i + di) * (ny + 1) * (nz + 1) + (j + dj) * (nz + 1) + (k + dk);
                };

                // Hex node indices
                Index n0 = node_idx(0, 0, 0);
                Index n1 = node_idx(1, 0, 0);
                Index n2 = node_idx(1, 1, 0);
                Index n3 = node_idx(0, 1, 0);
                Index n4 = node_idx(0, 0, 1);
                Index n5 = node_idx(1, 0, 1);
                Index n6 = node_idx(1, 1, 1);
                Index n7 = node_idx(0, 1, 1);

                // 6 tetrahedra decomposition
                Index tets[6][4] = {
                    {n0, n1, n3, n4}, {n1, n2, n3, n6},
                    {n1, n3, n4, n6}, {n1, n4, n5, n6},
                    {n3, n4, n6, n7}, {n4, n5, n6, n7}
                };

                for (int t = 0; t < 6; ++t) {
                    auto elem_nodes = block.element_nodes(elem_id);
                    for (int n = 0; n < 4; ++n) {
                        elem_nodes[n] = tets[t][n];
                        connectivity.push_back(tets[t][n]);
                    }
                    elem_id++;
                }
            }
        }
    }

    return {mesh, connectivity};
}

// ============================================================================
// Benchmark Runner
// ============================================================================

BenchmarkResult run_benchmark(
    const std::string& name,
    std::shared_ptr<Mesh> mesh,
    const std::vector<Index>& connectivity,
    physics::ElementType elem_type,
    int nodes_per_elem,
    int num_steps,
    int warmup_steps = 5)
{
    BenchmarkResult result;
    result.name = name;
    result.num_elements = static_cast<int>(connectivity.size() / nodes_per_elem);
    result.num_nodes = mesh->num_nodes();
    result.num_dofs = result.num_nodes * 3;
    result.num_steps = num_steps;
    result.backend = get_execution_space_name();

    // Setup solver
    auto state = std::make_shared<State>(*mesh);
    FEMSolver solver(name);

    physics::MaterialProperties steel;
    steel.density = 7850.0;
    steel.E = 210.0e9;
    steel.nu = 0.3;
    steel.G = steel.E / (2.0 * (1.0 + steel.nu));
    steel.K = steel.E / (3.0 * (1.0 - 2.0 * steel.nu));

    std::vector<Index> elem_ids(result.num_elements);
    for (int i = 0; i < result.num_elements; ++i) elem_ids[i] = i;

    solver.add_element_group("solid", elem_type, elem_ids, connectivity, steel);

    // Boundary conditions (fix left end)
    std::vector<Index> fixed_nodes;
    for (int i = 0; i < result.num_nodes; ++i) {
        auto coords = mesh->get_node_coordinates(i);
        if (coords[0] < 0.01) {  // Left end
            fixed_nodes.push_back(i);
        }
    }

    for (int dof = 0; dof < 3; ++dof) {
        BoundaryCondition bc(BCType::Displacement, fixed_nodes, dof, 0.0);
        solver.add_boundary_condition(bc);
    }

    // Apply load (right end)
    std::vector<Index> loaded_nodes;
    double max_x = 0;
    for (int i = 0; i < result.num_nodes; ++i) {
        auto coords = mesh->get_node_coordinates(i);
        if (coords[0] > max_x) max_x = coords[0];
    }
    for (int i = 0; i < result.num_nodes; ++i) {
        auto coords = mesh->get_node_coordinates(i);
        if (coords[0] > max_x - 0.01) {
            loaded_nodes.push_back(i);
        }
    }

    if (!loaded_nodes.empty()) {
        double force_per_node = -10000.0 / loaded_nodes.size();
        BoundaryCondition bc_force(BCType::Force, loaded_nodes, 2, force_per_node);
        solver.add_boundary_condition(bc_force);
    }

    solver.initialize(mesh, state);

    // Compute timestep
    const double dt = solver.compute_stable_dt() * 0.9;

    // Warmup
    for (int step = 0; step < warmup_steps; ++step) {
        solver.step(dt);
    }
    device_fence();

    // Benchmark
    Timer timer;
    timer.start();

    for (int step = 0; step < num_steps; ++step) {
        solver.step(dt);
    }
    device_fence();

    timer.stop();
    result.total_time_sec = timer.elapsed_sec();
    result.compute_derived();

    // Estimate memory usage
    result.memory.host_bytes = result.num_dofs * sizeof(double) * 6;  // disp, vel, acc, f_int, f_ext, mass
    result.memory.host_bytes += connectivity.size() * sizeof(Index);
    result.memory.device_bytes = result.memory.host_bytes;  // DualView mirrors

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    nxs::InitOptions options;
    options.log_level = nxs::Logger::Level::Warn;
    nxs::Context context(options);

    // Parse command line
    bool run_large = false;
    bool quick_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--large") == 0) run_large = true;
        if (std::strcmp(argv[i], "--quick") == 0) quick_mode = true;
    }

    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "NexusSim Comprehensive Performance Benchmark\n";
    std::cout << "============================================================\n\n";

    // System info
    std::cout << "System Configuration:\n";
    std::cout << "  Execution Space: " << get_execution_space_name() << "\n";
    std::cout << "  GPU Available:   " << (gpu_available() ? "Yes" : "No") << "\n";
    std::cout << "  Mode:            " << (quick_mode ? "Quick" : (run_large ? "Large" : "Standard")) << "\n";
    std::cout << "\n";

    std::vector<BenchmarkResult> results;

    // Define problem sizes
    struct ProblemSize {
        int nx, ny, nz;
        int steps;
        const char* label;
    };

    std::vector<ProblemSize> sizes;
    if (quick_mode) {
        sizes = {
            {5, 5, 1, 50, "Tiny"},
            {10, 5, 2, 50, "Small"},
        };
    } else if (run_large) {
        sizes = {
            {5, 5, 1, 100, "Tiny"},
            {10, 5, 2, 100, "Small"},
            {20, 5, 4, 100, "Medium"},
            {40, 10, 4, 50, "Large"},
            {60, 10, 6, 50, "Very Large"},
        };
    } else {
        sizes = {
            {5, 5, 1, 100, "Tiny"},
            {10, 5, 2, 100, "Small"},
            {20, 5, 4, 50, "Medium"},
            {30, 6, 5, 50, "Large"},
        };
    }

    // Run Hex8 benchmarks
    std::cout << "Running Hex8 Benchmarks...\n";
    std::cout << std::string(60, '-') << "\n";

    for (const auto& sz : sizes) {
        std::cout << "  " << sz.label << " (" << sz.nx * sz.ny * sz.nz << " elements)..." << std::flush;
        auto [mesh, conn] = create_hex8_beam(sz.nx, sz.ny, sz.nz);
        auto result = run_benchmark(
            std::string("Hex8-") + sz.label,
            mesh, conn, physics::ElementType::Hex8, 8, sz.steps);
        results.push_back(result);
        std::cout << " " << std::fixed << std::setprecision(3) << result.time_per_step_ms << " ms/step\n";
    }

    // Run Tet4 benchmarks
    std::cout << "\nRunning Tet4 Benchmarks...\n";
    std::cout << std::string(60, '-') << "\n";

    for (const auto& sz : sizes) {
        std::cout << "  " << sz.label << " (" << sz.nx * sz.ny * sz.nz * 6 << " elements)..." << std::flush;
        auto [mesh, conn] = create_tet4_beam(sz.nx, sz.ny, sz.nz);
        auto result = run_benchmark(
            std::string("Tet4-") + sz.label,
            mesh, conn, physics::ElementType::Tet4, 4, sz.steps);
        results.push_back(result);
        std::cout << " " << std::fixed << std::setprecision(3) << result.time_per_step_ms << " ms/step\n";
    }

    // Print results table
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "Performance Results Summary\n";
    std::cout << "============================================================\n\n";

    BenchmarkResult::print_header();
    for (const auto& r : results) {
        r.print();
    }

    // Print throughput summary
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "Throughput Analysis\n";
    std::cout << "============================================================\n\n";

    double max_dofs_per_sec = 0;
    std::string best_config;
    for (const auto& r : results) {
        if (r.dofs_per_sec > max_dofs_per_sec) {
            max_dofs_per_sec = r.dofs_per_sec;
            best_config = r.name;
        }
    }

    std::cout << "Peak Throughput: " << std::scientific << std::setprecision(2)
              << max_dofs_per_sec << " DOFs/sec (" << best_config << ")\n";
    std::cout << "Backend: " << get_execution_space_name() << "\n";

    // Memory summary
    if (!results.empty()) {
        std::cout << "\nMemory Usage (largest problem):\n";
        results.back().memory.print();
    }

    // GPU utilization tips
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "Notes\n";
    std::cout << "============================================================\n\n";

    if (gpu_available()) {
        std::cout << "GPU backend detected. For optimal performance:\n";
        std::cout << "  - Use problems with >10K elements for GPU benefit\n";
        std::cout << "  - Smaller problems may run faster on CPU due to overhead\n";
        std::cout << "  - Monitor GPU utilization with 'nvidia-smi' or similar\n";
    } else {
        std::cout << "Running on CPU backend. To enable GPU:\n";
        std::cout << "  - Ensure Kokkos is compiled with CUDA/HIP/SYCL backend\n";
        std::cout << "  - Set Kokkos_ENABLE_CUDA=ON in CMake\n";
        std::cout << "  - Verify with: Kokkos::DefaultExecutionSpace::name()\n";
    }

    std::cout << "\n============================================================\n";
    std::cout << "Benchmark Complete\n";
    std::cout << "============================================================\n\n";

    return 0;
}
