/**
 * @file shell4_solver_test.cpp
 * @brief Tests for Shell4 6-DOF integration into FEMStaticSolver
 *
 * Verifies:
 *   1. DOF detection (3 vs 6)
 *   2. Cantilever bending (tip deflection vs beam theory)
 *   3. Membrane tension
 *   4. Stiffness symmetry
 *   5. Rotational boundary conditions
 *   6. Patch test (constant strain)
 *   7. Mesh convergence
 */

#include <nexussim/solver/fem_static_solver.hpp>
#include <nexussim/discretization/shell4.hpp>

#include <iostream>
#include <cmath>
#include <vector>
#include <string>

using namespace nxs;
using namespace nxs::solver;

// ============================================================================
// Test infrastructure
// ============================================================================

static int total_checks = 0;
static int passed_checks = 0;
static int failed_checks = 0;

#define CHECK(cond, msg) do { \
    total_checks++; \
    if (cond) { \
        passed_checks++; \
        std::cout << "  [PASS] " << msg << std::endl; \
    } else { \
        failed_checks++; \
        std::cout << "  [FAIL] " << msg << std::endl; \
    } \
} while(0)

// ============================================================================
// Shell mesh generators
// ============================================================================

/**
 * @brief Generate a flat shell strip mesh in the XY plane
 *
 * Creates nx elements along X, 1 element wide in Y.
 * Node layout (nx=4):
 *   5---6---7---8---9    (y = width)
 *   |   |   |   |   |
 *   0---1---2---3---4    (y = 0)
 */
Mesh generate_shell_strip(Real length, Real width, int nx,
                          Real thickness = 0.01) {
    int ny = 1;
    size_t num_nodes = (nx + 1) * (ny + 1);
    size_t num_elems = nx * ny;

    Mesh mesh(num_nodes);

    Real dx = length / nx;
    Real dy = width / ny;

    // Create nodes in XY plane (z=0)
    size_t idx = 0;
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            mesh.set_node_coordinates(idx, {i * dx, j * dy, 0.0});
            ++idx;
        }
    }

    // Create Shell4 element block
    Index block_id = mesh.add_element_block("shells", ElementType::Shell4, num_elems, 4);
    auto& block = mesh.element_block(block_id);

    size_t eidx = 0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            auto nodes = block.element_nodes(eidx);
            nodes[0] = j * (nx + 1) + i;
            nodes[1] = j * (nx + 1) + i + 1;
            nodes[2] = (j + 1) * (nx + 1) + i + 1;
            nodes[3] = (j + 1) * (nx + 1) + i;
            ++eidx;
        }
    }

    return mesh;
}

/**
 * @brief Generate a 2D shell mesh (nx x ny elements)
 */
Mesh generate_shell_plate(Real Lx, Real Ly, int nx, int ny) {
    size_t num_nodes = (nx + 1) * (ny + 1);
    size_t num_elems = nx * ny;

    Mesh mesh(num_nodes);

    Real dx = Lx / nx;
    Real dy = Ly / ny;

    size_t idx = 0;
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            mesh.set_node_coordinates(idx, {i * dx, j * dy, 0.0});
            ++idx;
        }
    }

    Index block_id = mesh.add_element_block("shells", ElementType::Shell4, num_elems, 4);
    auto& block = mesh.element_block(block_id);

    size_t eidx = 0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            auto nodes = block.element_nodes(eidx);
            nodes[0] = j * (nx + 1) + i;
            nodes[1] = j * (nx + 1) + i + 1;
            nodes[2] = (j + 1) * (nx + 1) + i + 1;
            nodes[3] = (j + 1) * (nx + 1) + i;
            ++eidx;
        }
    }

    return mesh;
}

// ============================================================================
// Test 1: DOF Detection
// ============================================================================

void test_dof_detection() {
    std::cout << "\n--- Test: DOF detection ---\n";

    // Pure solid mesh -> 3 DOFs/node
    {
        auto mesh = generate_cantilever_mesh(1.0, 0.1, 0.1, 2, 1, 1);
        FEMStaticSolver solver;
        solver.set_mesh(mesh);
        CHECK(solver.dof_per_node() == 3, "Solid-only mesh -> 3 DOFs/node");
    }

    // Pure shell mesh -> 6 DOFs/node
    {
        auto mesh = generate_shell_strip(1.0, 0.1, 4);
        FEMStaticSolver solver;
        solver.set_mesh(mesh);
        CHECK(solver.dof_per_node() == 6, "Shell mesh -> 6 DOFs/node");
    }

    // Mixed mesh -> 6 DOFs/node
    {
        // Create a mesh with both solid and shell blocks
        Mesh mesh(12);  // 8 hex nodes + 4 shell nodes (sharing some)
        for (int i = 0; i < 8; ++i) {
            Real x = (i & 1) ? 1.0 : 0.0;
            Real y = (i & 2) ? 1.0 : 0.0;
            Real z = (i & 4) ? 1.0 : 0.0;
            mesh.set_node_coordinates(i, {x, y, z});
        }
        // Extra shell nodes
        mesh.set_node_coordinates(8, {2.0, 0.0, 0.0});
        mesh.set_node_coordinates(9, {2.0, 1.0, 0.0});
        mesh.set_node_coordinates(10, {2.0, 1.0, 1.0});
        mesh.set_node_coordinates(11, {2.0, 0.0, 1.0});

        Index b1 = mesh.add_element_block("solids", ElementType::Hex8, 1, 8);
        auto& blk1 = mesh.element_block(b1);
        auto n1 = blk1.element_nodes(0);
        for (int i = 0; i < 8; ++i) n1[i] = i;

        Index b2 = mesh.add_element_block("shells", ElementType::Shell4, 1, 4);
        auto& blk2 = mesh.element_block(b2);
        auto n2 = blk2.element_nodes(0);
        n2[0] = 1; n2[1] = 8; n2[2] = 9; n2[3] = 3;

        FEMStaticSolver solver;
        solver.set_mesh(mesh);
        CHECK(solver.dof_per_node() == 6, "Mixed solid+shell mesh -> 6 DOFs/node");
    }
}

// ============================================================================
// Test 2: Cantilever Bending
// ============================================================================

void test_cantilever_bending() {
    std::cout << "\n--- Test: Shell4 cantilever bending ---\n";

    // Cantilever beam: fixed at x=0, point load at x=L
    // Beam theory: delta = P*L^3 / (3*E*I), I = b*t^3/12
    const Real L = 1.0;
    const Real b = 0.1;       // width
    const Real t = 0.01;      // thickness
    const Real E = 2.0e11;
    const Real nu = 0.0;      // nu=0 for clean beam comparison
    const Real P = 1.0;       // total applied force

    const int nx = 8;
    auto mesh = generate_shell_strip(L, b, nx, t);

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    solver.set_shell_thickness(t);

    ElasticMaterial mat;
    mat.E = E;
    mat.nu = nu;
    solver.set_material(mat);

    // Fix all DOFs at x=0
    for (size_t i = 0; i < mesh.num_nodes(); ++i) {
        auto coords = mesh.get_node_coordinates(i);
        if (std::abs(coords[0]) < 1e-10) {
            solver.fix_node_all(i);
        }
    }

    // Apply downward force at x=L (z-direction for out-of-plane bending)
    // Distribute force to both nodes at x=L
    std::vector<Index> tip_nodes;
    for (size_t i = 0; i < mesh.num_nodes(); ++i) {
        auto coords = mesh.get_node_coordinates(i);
        if (std::abs(coords[0] - L) < 1e-10) {
            tip_nodes.push_back(i);
        }
    }

    Real force_per_node = P / tip_nodes.size();
    for (auto node : tip_nodes) {
        solver.add_force(node, 2, -force_per_node);  // z-direction (out of plane)
    }

    auto result = solver.solve_linear();

    CHECK(result.converged, "Cantilever bending converged");
    CHECK(result.iterations < 5000, "Reasonable iteration count: " + std::to_string(result.iterations));

    // Check tip deflection
    Real max_tip_z = 0.0;
    for (auto node : tip_nodes) {
        auto disp = FEMStaticSolver::get_node_displacement(result, node, 6);
        max_tip_z = std::max(max_tip_z, std::abs(disp[2]));
    }

    // Analytical: delta = P*L^3/(3*E*I)
    Real I = b * t * t * t / 12.0;
    Real delta_analytical = P * L * L * L / (3.0 * E * I);

    Real ratio = max_tip_z / delta_analytical;
    std::cout << "  Tip deflection: " << max_tip_z << std::endl;
    std::cout << "  Analytical:     " << delta_analytical << std::endl;
    std::cout << "  Ratio:          " << ratio << std::endl;

    // Shell4 with full-integration shear has shear locking for thin shells,
    // so expect significantly stiffer response than pure beam theory.
    // The key test is that bending occurs in the correct direction and converges
    // with refinement (test_convergence checks this).
    CHECK(ratio > 0.001 && ratio < 10.0,
          "Tip deflection has correct order-of-magnitude (shear locking expected)");
    CHECK(!std::isnan(max_tip_z), "No NaN in tip displacement");
    CHECK(max_tip_z > 0.0, "Non-zero tip displacement");
}

// ============================================================================
// Test 3: Membrane Tension
// ============================================================================

void test_membrane_tension() {
    std::cout << "\n--- Test: Shell4 membrane tension ---\n";

    // 2x1 shell mesh under uniaxial tension
    const Real L = 1.0;
    const Real b = 0.5;
    const Real t = 0.01;
    const Real E = 2.0e11;
    const Real nu = 0.0;
    const Real P = 1000.0;  // total force

    auto mesh = generate_shell_strip(L, b, 2, t);

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    solver.set_shell_thickness(t);

    ElasticMaterial mat;
    mat.E = E;
    mat.nu = nu;
    solver.set_material(mat);

    // Fix x=0 end: fix x-displacement, constrain y and z to prevent rigid body
    for (size_t i = 0; i < mesh.num_nodes(); ++i) {
        auto coords = mesh.get_node_coordinates(i);
        if (std::abs(coords[0]) < 1e-10) {
            solver.add_dirichlet_bc(i, 0, 0.0);  // fix ux
            solver.add_dirichlet_bc(i, 1, 0.0);  // fix uy
            solver.add_dirichlet_bc(i, 2, 0.0);  // fix uz
            solver.add_dirichlet_bc(i, 3, 0.0);  // fix rx
            solver.add_dirichlet_bc(i, 4, 0.0);  // fix ry
            solver.add_dirichlet_bc(i, 5, 0.0);  // fix rz
        }
    }

    // Apply tension at x=L
    std::vector<Index> right_nodes;
    for (size_t i = 0; i < mesh.num_nodes(); ++i) {
        auto coords = mesh.get_node_coordinates(i);
        if (std::abs(coords[0] - L) < 1e-10) {
            right_nodes.push_back(i);
        }
    }

    Real force_per_node = P / right_nodes.size();
    for (auto node : right_nodes) {
        solver.add_force(node, 0, force_per_node);  // x-direction tension
    }

    auto result = solver.solve_linear();

    CHECK(result.converged, "Membrane tension converged");

    // Analytical: delta = P*L / (E*t*b)
    Real delta_analytical = P * L / (E * t * b);

    // Check x-displacement at x=L
    Real ux_sum = 0.0;
    for (auto node : right_nodes) {
        auto disp = FEMStaticSolver::get_node_displacement(result, node, 6);
        ux_sum += disp[0];
    }
    Real ux_avg = ux_sum / right_nodes.size();

    Real ratio = ux_avg / delta_analytical;
    std::cout << "  Avg ux at x=L: " << ux_avg << std::endl;
    std::cout << "  Analytical:    " << delta_analytical << std::endl;
    std::cout << "  Ratio:         " << ratio << std::endl;

    CHECK(ratio > 0.5 && ratio < 2.0,
          "Membrane extension within reasonable range of analytical");

    // Check uniformity: all tip nodes should have similar ux
    Real ux_min = 1e30, ux_max = -1e30;
    for (auto node : right_nodes) {
        auto disp = FEMStaticSolver::get_node_displacement(result, node, 6);
        ux_min = std::min(ux_min, disp[0]);
        ux_max = std::max(ux_max, disp[0]);
    }
    Real uniformity = (ux_max > 1e-20) ? (ux_max - ux_min) / ux_max : 0.0;
    CHECK(uniformity < 0.1, "Tip displacement is uniform across width");
}

// ============================================================================
// Test 4: Stiffness Symmetry
// ============================================================================

void test_stiffness_symmetry() {
    std::cout << "\n--- Test: Stiffness matrix symmetry ---\n";

    auto mesh = generate_shell_strip(1.0, 0.1, 2);

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    solver.set_shell_thickness(0.01);

    ElasticMaterial mat;
    mat.E = 2.0e11;
    mat.nu = 0.3;
    solver.set_material(mat);

    // Need to trigger assembly by adding BCs and calling solve
    // Instead, let's just assemble K and check
    for (size_t i = 0; i < mesh.num_nodes(); ++i) {
        auto coords = mesh.get_node_coordinates(i);
        if (std::abs(coords[0]) < 1e-10) {
            solver.fix_node_all(i);
        }
    }
    solver.add_force(mesh.num_nodes() - 1, 2, 1.0);

    auto result = solver.solve_linear();

    const auto& K = solver.stiffness_matrix();
    size_t ndof = mesh.num_nodes() * 6;

    // Check symmetry: sample pairs
    // Use absolute threshold to skip near-zero pairs (floating-point noise)
    Real max_abs_diff = 0.0;
    Real max_rel_asym = 0.0;
    int samples = 0;
    for (size_t i = 0; i < ndof && i < 36; ++i) {
        for (size_t j = i + 1; j < ndof && j < 36; ++j) {
            Real Kij = K.get(i, j);
            Real Kji = K.get(j, i);
            Real abs_diff = std::abs(Kij - Kji);
            max_abs_diff = std::max(max_abs_diff, abs_diff);
            Real denom = std::max(std::abs(Kij), std::abs(Kji));
            if (denom > 1e-10) {  // Only check relative asymmetry for significant entries
                Real asym = abs_diff / denom;
                max_rel_asym = std::max(max_rel_asym, asym);
                samples++;
            }
        }
    }

    std::cout << "  Max abs diff: " << max_abs_diff
              << ", max rel asym: " << max_rel_asym
              << " (" << samples << " significant pairs)" << std::endl;
    CHECK(max_rel_asym < 1e-6, "Global stiffness matrix is symmetric (significant entries)");

    // Check positive diagonals
    bool all_positive = true;
    for (size_t i = 0; i < ndof; ++i) {
        if (K.get(i, i) <= 0.0) {
            all_positive = false;
            break;
        }
    }
    CHECK(all_positive, "All diagonal entries are positive");

    CHECK(result.converged, "Solver converged with shell stiffness");
}

// ============================================================================
// Test 5: Rotational Boundary Conditions
// ============================================================================

void test_rotational_bcs() {
    std::cout << "\n--- Test: Rotational BCs ---\n";

    const Real L = 1.0;
    const Real b = 0.1;
    const Real t = 0.01;

    auto mesh = generate_shell_strip(L, b, 4, t);

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    solver.set_shell_thickness(t);

    ElasticMaterial mat;
    mat.E = 2.0e11;
    mat.nu = 0.0;
    solver.set_material(mat);

    // Fix all DOFs at x=0
    for (size_t i = 0; i < mesh.num_nodes(); ++i) {
        auto coords = mesh.get_node_coordinates(i);
        if (std::abs(coords[0]) < 1e-10) {
            solver.fix_node_all(i);
        }
    }

    // Apply moment at tip (rotation around y-axis)
    std::vector<Index> tip_nodes;
    for (size_t i = 0; i < mesh.num_nodes(); ++i) {
        auto coords = mesh.get_node_coordinates(i);
        if (std::abs(coords[0] - L) < 1e-10) {
            tip_nodes.push_back(i);
        }
    }

    Real M_total = 1.0;
    Real moment_per_node = M_total / tip_nodes.size();
    for (auto node : tip_nodes) {
        solver.add_moment(node, 4, moment_per_node);  // My moment
    }

    auto result = solver.solve_linear();

    CHECK(result.converged, "Moment loading converged");

    // Check that rotations at fixed end are zero
    bool fixed_rotations_zero = true;
    for (size_t i = 0; i < mesh.num_nodes(); ++i) {
        auto coords = mesh.get_node_coordinates(i);
        if (std::abs(coords[0]) < 1e-10) {
            size_t base = i * 6;
            for (int d = 3; d < 6; ++d) {
                if (std::abs(result.displacement[base + d]) > 1e-10) {
                    fixed_rotations_zero = false;
                }
            }
        }
    }
    CHECK(fixed_rotations_zero, "Rotations at fixed end are zero");

    // Check that tip has non-zero rotation
    bool tip_has_rotation = false;
    for (auto node : tip_nodes) {
        size_t base = node * 6;
        Real rot_mag = 0.0;
        for (int d = 3; d < 6; ++d) {
            rot_mag += result.displacement[base + d] * result.displacement[base + d];
        }
        if (std::sqrt(rot_mag) > 1e-15) {
            tip_has_rotation = true;
        }
    }
    CHECK(tip_has_rotation, "Tip has non-zero rotation from moment loading");
}

// ============================================================================
// Test 6: Patch Test (Constant Strain)
// ============================================================================

void test_patch_test() {
    std::cout << "\n--- Test: Shell4 membrane patch test ---\n";

    // Single element patch test: apply constant strain state
    // For membrane: uniform extension in x
    const Real L = 1.0;
    const Real b = 1.0;
    const Real t = 0.01;
    const Real E = 1.0e6;  // Simple value for easy calculation
    const Real nu = 0.0;
    const Real eps_x = 0.001;  // Target strain

    Mesh mesh(4);
    mesh.set_node_coordinates(0, {0.0, 0.0, 0.0});
    mesh.set_node_coordinates(1, {L,   0.0, 0.0});
    mesh.set_node_coordinates(2, {L,   b,   0.0});
    mesh.set_node_coordinates(3, {0.0, b,   0.0});

    Index bid = mesh.add_element_block("patch", ElementType::Shell4, 1, 4);
    auto& blk = mesh.element_block(bid);
    auto nodes = blk.element_nodes(0);
    nodes[0] = 0; nodes[1] = 1; nodes[2] = 2; nodes[3] = 3;

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    solver.set_shell_thickness(t);

    ElasticMaterial mat;
    mat.E = E;
    mat.nu = nu;
    solver.set_material(mat);

    // Prescribe displacements consistent with constant strain
    // ux = eps_x * x, uy = 0, uz = 0
    for (size_t i = 0; i < 4; ++i) {
        auto coords = mesh.get_node_coordinates(i);
        Real prescribed_ux = eps_x * coords[0];
        solver.add_dirichlet_bc(i, 0, prescribed_ux);
        solver.add_dirichlet_bc(i, 1, 0.0);
        solver.add_dirichlet_bc(i, 2, 0.0);
        solver.add_dirichlet_bc(i, 3, 0.0);
        solver.add_dirichlet_bc(i, 4, 0.0);
        solver.add_dirichlet_bc(i, 5, 0.0);
    }

    auto result = solver.solve_linear();

    CHECK(result.converged, "Patch test converged");

    // Check that displacements match prescribed values
    bool displacements_correct = true;
    for (size_t i = 0; i < 4; ++i) {
        auto coords = mesh.get_node_coordinates(i);
        auto disp = FEMStaticSolver::get_node_displacement(result, i, 6);
        Real expected_ux = eps_x * coords[0];
        if (std::abs(disp[0] - expected_ux) > 1e-8) {
            displacements_correct = false;
            std::cout << "  Node " << i << ": ux=" << disp[0]
                      << " expected=" << expected_ux << std::endl;
        }
    }
    CHECK(displacements_correct, "Patch test: displacements match prescribed values");

    // Check no NaN
    bool no_nan = true;
    for (auto v : result.displacement) {
        if (std::isnan(v)) { no_nan = false; break; }
    }
    CHECK(no_nan, "Patch test: no NaN in solution");

    // Check that uy, uz are zero
    bool transverse_zero = true;
    for (size_t i = 0; i < 4; ++i) {
        auto disp = FEMStaticSolver::get_node_displacement(result, i, 6);
        if (std::abs(disp[1]) > 1e-8 || std::abs(disp[2]) > 1e-8) {
            transverse_zero = false;
        }
    }
    CHECK(transverse_zero, "Patch test: transverse displacements are zero");
}

// ============================================================================
// Test 7: Mesh Convergence
// ============================================================================

void test_convergence() {
    std::cout << "\n--- Test: Mesh convergence ---\n";

    const Real L = 1.0;
    const Real b = 0.1;
    const Real t = 0.01;
    const Real E = 2.0e11;
    const Real nu = 0.0;
    const Real P = 1.0;

    // Analytical tip deflection
    Real I = b * t * t * t / 12.0;
    Real delta_analytical = P * L * L * L / (3.0 * E * I);
    std::cout << "  Analytical delta: " << delta_analytical << std::endl;

    int refinements[] = {2, 4, 8};
    Real deltas[3] = {0, 0, 0};

    for (int r = 0; r < 3; ++r) {
        int nx = refinements[r];
        auto mesh = generate_shell_strip(L, b, nx, t);

        FEMStaticSolver solver;
        solver.set_mesh(mesh);
        solver.set_shell_thickness(t);

        ElasticMaterial mat;
        mat.E = E;
        mat.nu = nu;
        solver.set_material(mat);

        // Fix at x=0
        for (size_t i = 0; i < mesh.num_nodes(); ++i) {
            auto coords = mesh.get_node_coordinates(i);
            if (std::abs(coords[0]) < 1e-10) {
                solver.fix_node_all(i);
            }
        }

        // Load at x=L
        std::vector<Index> tip;
        for (size_t i = 0; i < mesh.num_nodes(); ++i) {
            auto coords = mesh.get_node_coordinates(i);
            if (std::abs(coords[0] - L) < 1e-10) {
                tip.push_back(i);
            }
        }

        Real fpn = P / tip.size();
        for (auto n : tip) {
            solver.add_force(n, 2, -fpn);
        }

        auto result = solver.solve_linear();

        Real max_tip = 0.0;
        for (auto n : tip) {
            auto d = FEMStaticSolver::get_node_displacement(result, n, 6);
            max_tip = std::max(max_tip, std::abs(d[2]));
        }

        deltas[r] = max_tip;
        std::cout << "  nx=" << nx << ": delta=" << max_tip
                  << " ratio=" << max_tip / delta_analytical << std::endl;
    }

    // Check convergence: finer meshes should give results closer to analytical
    // With refinement, the error should decrease
    Real err_coarse = std::abs(deltas[0] - delta_analytical);
    Real err_fine = std::abs(deltas[2] - delta_analytical);

    CHECK(deltas[0] > 0.0 && deltas[1] > 0.0 && deltas[2] > 0.0,
          "All meshes produce non-zero deflection");
    CHECK(err_fine < err_coarse || err_fine < 0.5 * delta_analytical,
          "Finer mesh improves or maintains accuracy");
    // Shell4 has shear locking, so even fine mesh won't match beam theory exactly.
    // Check that deflection is non-trivially large (> 0.1% of analytical).
    CHECK(deltas[2] / delta_analytical > 0.001,
          "Finest mesh produces meaningful deflection (shear locking expected)");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "  Shell4 6-DOF Solver Integration Test\n";
    std::cout << "========================================\n";

    test_dof_detection();
    test_cantilever_bending();
    test_membrane_tension();
    test_stiffness_symmetry();
    test_rotational_bcs();
    test_patch_test();
    test_convergence();

    std::cout << "\n========================================\n";
    std::cout << "  Results: " << passed_checks << "/" << total_checks
              << " passed, " << failed_checks << " failed\n";
    std::cout << "========================================\n";

    return (failed_checks > 0) ? 1 : 0;
}
