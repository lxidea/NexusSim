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
#include <nexussim/solver/fem_static_solver.hpp>
#include <nexussim/data/mesh.hpp>
#include <iostream>
#include <cmath>
#include <limits>

using namespace nxs;
using namespace nxs::solver;

/**
 * @brief Create a Hex20 serendipity bar mesh (corners + edge midpoints only)
 *
 * Unlike a full tensor product grid (2n+1)^3 which includes unused face/body
 * centers, this creates only the nodes that Hex20 elements actually reference.
 */
Mesh create_hex20_beam(int nx, int ny, int nz, Real Lx, Real Ly, Real Lz) {
    size_t n_corners = (nx+1) * (ny+1) * (nz+1);
    size_t n_xmid = nx * (ny+1) * (nz+1);
    size_t n_ymid = (nx+1) * ny * (nz+1);
    size_t n_zmid = (nx+1) * (ny+1) * nz;
    size_t num_nodes = n_corners + n_xmid + n_ymid + n_zmid;
    size_t num_elems = nx * ny * nz;

    Mesh mesh(num_nodes);

    Real dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;

    auto corner_id = [&](int i, int j, int k) -> Index {
        return k*(ny+1)*(nx+1) + j*(nx+1) + i;
    };
    auto xmid_id = [&](int i, int j, int k) -> Index {
        return n_corners + k*(ny+1)*nx + j*nx + i;
    };
    auto ymid_id = [&](int i, int j, int k) -> Index {
        return n_corners + n_xmid + k*ny*(nx+1) + j*(nx+1) + i;
    };
    auto zmid_id = [&](int i, int j, int k) -> Index {
        return n_corners + n_xmid + n_ymid + k*(ny+1)*(nx+1) + j*(nx+1) + i;
    };

    // Corner nodes
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i <= nx; ++i)
                mesh.set_node_coordinates(corner_id(i,j,k), {i*dx, j*dy, k*dz});

    // X-direction midedge nodes
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i < nx; ++i)
                mesh.set_node_coordinates(xmid_id(i,j,k), {(i+0.5)*dx, j*dy, k*dz});

    // Y-direction midedge nodes
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i <= nx; ++i)
                mesh.set_node_coordinates(ymid_id(i,j,k), {i*dx, (j+0.5)*dy, k*dz});

    // Z-direction midedge nodes
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i <= nx; ++i)
                mesh.set_node_coordinates(zmid_id(i,j,k), {i*dx, j*dy, (k+0.5)*dz});

    Index bid = mesh.add_element_block("beam", ElementType::Hex20, num_elems, 20);
    auto& block = mesh.element_block(bid);

    size_t eidx = 0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                auto en = block.element_nodes(eidx++);
                // Corner nodes (0-7)
                en[0]  = corner_id(i,   j,   k);
                en[1]  = corner_id(i+1, j,   k);
                en[2]  = corner_id(i+1, j+1, k);
                en[3]  = corner_id(i,   j+1, k);
                en[4]  = corner_id(i,   j,   k+1);
                en[5]  = corner_id(i+1, j,   k+1);
                en[6]  = corner_id(i+1, j+1, k+1);
                en[7]  = corner_id(i,   j+1, k+1);
                // Bottom edges (8-11)
                en[8]  = xmid_id(i,   j,   k);
                en[9]  = ymid_id(i+1, j,   k);
                en[10] = xmid_id(i,   j+1, k);
                en[11] = ymid_id(i,   j,   k);
                // Vertical edges (12-15)
                en[12] = zmid_id(i,   j,   k);
                en[13] = zmid_id(i+1, j,   k);
                en[14] = zmid_id(i+1, j+1, k);
                en[15] = zmid_id(i,   j+1, k);
                // Top edges (16-19)
                en[16] = xmid_id(i,   j,   k+1);
                en[17] = ymid_id(i+1, j,   k+1);
                en[18] = xmid_id(i,   j+1, k+1);
                en[19] = ymid_id(i,   j,   k+1);
            }

    return mesh;
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
            const int nx = div[0];
            const int ny = div[1];
            const int nz = div[2];
            const int n_elems = nx * ny * nz;

            NXS_LOG_INFO("=================================================");
            NXS_LOG_INFO("Running with {}x{}x{} Hex20 elements", nx, ny, nz);
            NXS_LOG_INFO("=================================================");

            auto mesh = create_hex20_beam(nx, ny, nz, beam_length, beam_width, beam_height);
            NXS_LOG_INFO("Created mesh with {} nodes, {} elements", mesh.num_nodes(), n_elems);

            // ================================================================
            // Setup and solve using implicit static solver
            // ================================================================

            FEMStaticSolver solver;
            solver.set_mesh(mesh);

            ElasticMaterial mat;
            mat.E = E;
            mat.nu = nu;
            mat.rho = density;
            solver.set_material(mat);

            // Fixed end (x=0)
            auto fixed_nodes = get_nodes_at_x(mesh, 0.0);
            for (auto n : fixed_nodes) solver.fix_node(n);

            // Applied force at free end (x=L)
            auto loaded_nodes = get_nodes_at_x(mesh, beam_length);
            const Real force_per_node = applied_force / static_cast<Real>(loaded_nodes.size());
            for (auto n : loaded_nodes) solver.add_force(n, 2, force_per_node);

            NXS_LOG_INFO("Applied BCs: {} nodes fixed, {} nodes loaded",
                        fixed_nodes.size(), loaded_nodes.size());

            // Solve static problem: K*u = F
            NXS_LOG_INFO("Solving static problem...");
            auto result = solver.solve_linear();

            if (!result.converged) {
                NXS_LOG_WARN("Static solver did not converge (residual: {:.2e}, iterations: {})",
                            result.residual, result.iterations);
                computed_deflections.push_back(std::numeric_limits<Real>::quiet_NaN());
            } else {
                NXS_LOG_INFO("Converged in {} iterations (residual: {:.2e})",
                            result.iterations, result.residual);

                // Find max deflection at loaded end
                Real max_w = 0.0;
                for (auto n : loaded_nodes) {
                    Real w = result.displacement[n * 3 + 2];
                    if (std::abs(w) > std::abs(max_w)) max_w = w;
                }
                computed_deflections.push_back(max_w);
            }

            const Real computed_deflection = computed_deflections.back();

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

            // Hex20 should achieve good accuracy; 15% tolerance allows coarse meshes
            // NaN detection: NaN > 15.0 evaluates false in C++, so check explicitly
            if (std::isnan(computed_deflections[i]) || rel_error > 15.0) {
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
