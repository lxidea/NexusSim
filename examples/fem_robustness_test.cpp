/**
 * @file fem_robustness_test.cpp
 * @brief Tests for FEM solver robustness guards
 *
 * Covers three categories:
 *   1. Singular / ill-conditioned systems
 *   2. Degenerate elements
 *   3. NaN / Inf propagation
 */

#include <nexussim/solver/fem_static_solver.hpp>
#include <nexussim/io/mesh_validator.hpp>
#include <nexussim/discretization/tet4.hpp>
#include <nexussim/discretization/hex8.hpp>

#include <iostream>
#include <cmath>
#include <limits>
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

static bool has_nan(const std::vector<Real>& v) {
    for (auto x : v) {
        if (std::isnan(x) || std::isinf(x)) return true;
    }
    return false;
}

// ============================================================================
// Category 1: Singular / ill-conditioned systems
// ============================================================================

void test_floating_structure() {
    std::cout << "\n--- Floating structure (no BCs) ---\n";

    auto mesh = generate_cantilever_mesh(1.0, 0.1, 0.1, 2, 1, 1);

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    ElasticMaterial mat;
    mat.E = 2.0e11;
    mat.nu = 0.3;
    solver.set_material(mat);

    // Apply load but no boundary conditions at all
    auto tip = get_nodes_at_x_max(mesh);
    for (auto n : tip) solver.add_force(n, 1, -1000.0);

    auto result = solver.solve_linear();

    CHECK(!result.converged || !has_nan(result.displacement),
          "Floating structure: no silent NaN (converged=" +
          std::to_string(result.converged) + ")");
    CHECK(!result.converged,
          "Floating structure: solver detects singularity (converged=" +
          std::to_string(result.converged) + ")");
}

void test_insufficient_bcs() {
    std::cout << "\n--- Insufficient BCs (only 1 node fixed) ---\n";

    auto mesh = generate_cantilever_mesh(1.0, 0.1, 0.1, 2, 1, 1);

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    ElasticMaterial mat;
    mat.E = 2.0e11;
    mat.nu = 0.3;
    solver.set_material(mat);

    // Fix only 1 node (3 DOFs) — not enough to prevent rigid body rotation
    solver.fix_node(0);

    auto tip = get_nodes_at_x_max(mesh);
    for (auto n : tip) solver.add_force(n, 1, -1000.0);

    auto result = solver.solve_linear();

    // Should either fail to converge or produce finite results
    CHECK(!has_nan(result.displacement),
          "Insufficient BCs: no NaN in displacement");
    // With penalty method, even insufficient BCs may converge,
    // but displacement should be finite
    if (result.converged) {
        Real max_d = FEMStaticSolver::max_displacement(result);
        CHECK(std::isfinite(max_d),
              "Insufficient BCs: finite displacement (max=" +
              std::to_string(max_d) + ")");
    } else {
        CHECK(true, "Insufficient BCs: solver correctly reports non-convergence");
    }
}

void test_near_incompressible() {
    std::cout << "\n--- Near-incompressible material (nu=0.499) ---\n";

    auto mesh = generate_cantilever_mesh(1.0, 0.1, 0.1, 2, 1, 1);

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    ElasticMaterial mat;
    mat.E = 2.0e11;
    mat.nu = 0.499;  // Nearly incompressible — ill-conditioned
    solver.set_material(mat);

    auto fixed = get_nodes_at_x(mesh, 0.0);
    for (auto n : fixed) solver.fix_node(n);

    auto tip = get_nodes_at_x_max(mesh);
    for (auto n : tip) solver.add_force(n, 1, -1000.0);

    auto result = solver.solve_linear();

    // Near-incompressible may or may not converge depending on element/solver;
    // the key requirement is no silent NaN corruption
    CHECK(!has_nan(result.displacement),
          "Near-incompressible: no NaN in displacement (converged=" +
          std::to_string(result.converged) + ")");
    if (result.converged) {
        Real max_d = FEMStaticSolver::max_displacement(result);
        CHECK(std::isfinite(max_d) && max_d > 0.0,
              "Near-incompressible: finite positive displacement (max=" +
              std::to_string(max_d) + ")");
    } else {
        CHECK(true, "Near-incompressible: graceful non-convergence (no NaN)");
    }
}

// ============================================================================
// Category 2: Degenerate elements
// ============================================================================

void test_coplanar_tet4() {
    std::cout << "\n--- Coplanar Tet4 (zero volume) ---\n";

    // 4 coplanar nodes in the z=0 plane
    Real coords[12] = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.5, 1.0, 0.0,
        0.25, 0.5, 0.0  // Coplanar!
    };

    fem::Tet4Element tet4;
    Real vol = tet4.volume(coords);
    CHECK(std::abs(vol) < 1e-12,
          "Coplanar Tet4: volume ~ 0 (got " + std::to_string(vol) + ")");

    // Stiffness matrix should be all zeros (or near-zero) for degenerate element
    std::vector<Real> ke(12 * 12, 0.0);
    tet4.stiffness_matrix(coords, 2.0e11, 0.3, ke.data());

    Real ke_max = 0.0;
    for (auto v : ke) ke_max = std::max(ke_max, std::abs(v));

    // For coplanar tet, volume is 0 → stiffness should be 0 or NaN
    // Either outcome is acceptable as long as we don't silently assemble garbage
    CHECK(ke_max < 1e-6 || has_nan(ke),
          "Coplanar Tet4: ke is zero or NaN (max=" + std::to_string(ke_max) + ")");
}

void test_collapsed_hex8() {
    std::cout << "\n--- Collapsed Hex8 (zero thickness) ---\n";

    // Hex8 with zero height (z=0 for all nodes)
    Real coords[24] = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,  // Same as bottom face!
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0
    };

    fem::Hex8Element hex8;
    std::vector<Real> ke(24 * 24, 0.0);
    hex8.stiffness_matrix(coords, 2.0e11, 0.3, ke.data());

    // Should produce NaN/Inf (singular Jacobian) or all zeros
    bool ke_nan = has_nan(ke);
    Real ke_max = 0.0;
    for (auto v : ke) {
        if (std::isfinite(v)) ke_max = std::max(ke_max, std::abs(v));
    }

    CHECK(ke_nan || ke_max < 1e-6,
          "Collapsed Hex8: ke contains NaN or is zero (nan=" +
          std::to_string(ke_nan) + ", max_finite=" + std::to_string(ke_max) + ")");

    // Assembly guard should catch this
    Mesh mesh(8);
    for (int i = 0; i < 8; ++i) {
        mesh.set_node_coordinates(i, {coords[i*3], coords[i*3+1], coords[i*3+2]});
    }
    Index bid = mesh.add_element_block("collapsed", ElementType::Hex8, 1, 8);
    auto& block = mesh.element_block(bid);
    auto en = block.element_nodes(0);
    for (int i = 0; i < 8; ++i) en[i] = i;

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    ElasticMaterial mat;
    mat.E = 2.0e11;
    mat.nu = 0.3;
    solver.set_material(mat);
    solver.fix_node(0);
    solver.fix_node(1);
    solver.fix_node(2);
    solver.fix_node(3);
    solver.add_force(4, 2, 1000.0);

    auto result = solver.solve_linear();

    // If the element produces NaN, the assembly guard should skip it
    if (ke_nan) {
        CHECK(solver.nan_element_count() > 0,
              "Collapsed Hex8: assembly NaN guard triggered (count=" +
              std::to_string(solver.nan_element_count()) + ")");
    } else {
        CHECK(true, "Collapsed Hex8: element produced finite (zero) stiffness, no NaN guard needed");
    }
}

void test_coincident_hex8_nodes() {
    std::cout << "\n--- Coincident Hex8 nodes (collapsed edge) ---\n";

    // Build a mesh where two nodes share the same coordinates
    Mesh mesh(8);
    mesh.set_node_coordinates(0, {0.0, 0.0, 0.0});
    mesh.set_node_coordinates(1, {1.0, 0.0, 0.0});
    mesh.set_node_coordinates(2, {1.0, 1.0, 0.0});
    mesh.set_node_coordinates(3, {0.0, 1.0, 0.0});
    mesh.set_node_coordinates(4, {0.0, 0.0, 1.0});
    mesh.set_node_coordinates(5, {0.0, 0.0, 1.0}); // Same as node 4!
    mesh.set_node_coordinates(6, {1.0, 1.0, 1.0});
    mesh.set_node_coordinates(7, {0.0, 1.0, 1.0});

    Index bid = mesh.add_element_block("collapsed_edge", ElementType::Hex8, 1, 8);
    auto& block = mesh.element_block(bid);
    auto en = block.element_nodes(0);
    for (int i = 0; i < 8; ++i) en[i] = i;

    io::MeshValidator validator;
    auto summary = validator.validate(mesh);

    CHECK(summary.duplicate_nodes > 0,
          "Coincident nodes: mesh validator detects duplicates (count=" +
          std::to_string(summary.duplicate_nodes) + ")");
}

void test_near_zero_volume_tet4() {
    std::cout << "\n--- Nearly flat Tet4 (poor quality) ---\n";

    // Very flat tet - one very long edge, one very short
    Real coords[12] = {
        0.0,  0.0, 0.0,
        100.0, 0.0, 0.0,   // Long edge
        50.0,  0.5, 0.0,
        50.0,  0.25, 1e-6  // Barely above the plane
    };

    fem::Tet4Element tet4;
    Real vol = tet4.volume(coords);

    CHECK(vol > 0.0 && vol < 1e-4,
          "Nearly flat Tet4: small positive volume (vol=" + std::to_string(vol) + ")");

    // Build mesh and check quality
    Mesh mesh(4);
    for (int i = 0; i < 4; ++i) {
        mesh.set_node_coordinates(i, {coords[i*3], coords[i*3+1], coords[i*3+2]});
    }
    Index bid = mesh.add_element_block("flat_tet", ElementType::Tet4, 1, 4);
    auto& block = mesh.element_block(bid);
    auto en = block.element_nodes(0);
    for (int i = 0; i < 4; ++i) en[i] = i;

    io::MeshValidator validator;
    auto qualities = validator.compute_all_quality(mesh);
    CHECK(qualities.size() > 0,
          "Nearly flat Tet4: quality computed");
    if (qualities.size() > 0) {
        CHECK(qualities[0].aspect_ratio > 10.0,
              "Nearly flat Tet4: high aspect ratio (ar=" +
              std::to_string(qualities[0].aspect_ratio) + ")");
    }
}

// ============================================================================
// Category 3: NaN / Inf propagation
// ============================================================================

void test_negative_youngs_modulus() {
    std::cout << "\n--- Negative Young's modulus ---\n";

    auto mesh = generate_cantilever_mesh(1.0, 0.1, 0.1, 1, 1, 1);

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    ElasticMaterial mat;
    mat.E = -2.0e11;  // Negative!
    mat.nu = 0.3;
    solver.set_material(mat);

    auto fixed = get_nodes_at_x(mesh, 0.0);
    for (auto n : fixed) solver.fix_node(n);

    auto tip = get_nodes_at_x_max(mesh);
    for (auto n : tip) solver.add_force(n, 1, -1000.0);

    auto result = solver.solve_linear();

    // Negative E → negative stiffness → CG should fail (not SPD)
    // Either non-convergence or NaN detection
    CHECK(!result.converged || !has_nan(result.displacement),
          "Negative E: no silent NaN (converged=" +
          std::to_string(result.converged) + ")");
}

void test_nan_in_force_vector() {
    std::cout << "\n--- NaN in force vector ---\n";

    auto mesh = generate_cantilever_mesh(1.0, 0.1, 0.1, 1, 1, 1);

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    ElasticMaterial mat;
    solver.set_material(mat);

    auto fixed = get_nodes_at_x(mesh, 0.0);
    for (auto n : fixed) solver.fix_node(n);

    // Apply NaN force
    auto tip = get_nodes_at_x_max(mesh);
    for (auto n : tip) {
        solver.add_force(n, 1, std::numeric_limits<Real>::quiet_NaN());
    }

    auto result = solver.solve_linear();

    CHECK(!result.converged,
          "NaN force: solver reports non-convergence");
    // Even if it "converges" with penalty method, displacement should not be all NaN
    // The CG solver should detect the NaN RHS
}

void test_nan_in_coords() {
    std::cout << "\n--- NaN in node coordinates ---\n";

    // Create mesh with a NaN coordinate
    Mesh mesh(8);
    mesh.set_node_coordinates(0, {0.0, 0.0, 0.0});
    mesh.set_node_coordinates(1, {1.0, 0.0, 0.0});
    mesh.set_node_coordinates(2, {1.0, 1.0, 0.0});
    mesh.set_node_coordinates(3, {0.0, 1.0, 0.0});
    mesh.set_node_coordinates(4, {0.0, 0.0, 1.0});
    mesh.set_node_coordinates(5, {std::numeric_limits<Real>::quiet_NaN(), 0.0, 1.0}); // NaN!
    mesh.set_node_coordinates(6, {1.0, 1.0, 1.0});
    mesh.set_node_coordinates(7, {0.0, 1.0, 1.0});

    Index bid = mesh.add_element_block("nan_elem", ElementType::Hex8, 1, 8);
    auto& block = mesh.element_block(bid);
    auto en = block.element_nodes(0);
    for (int i = 0; i < 8; ++i) en[i] = i;

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    ElasticMaterial mat;
    solver.set_material(mat);

    solver.fix_node(0);
    solver.fix_node(1);
    solver.fix_node(2);
    solver.fix_node(3);
    solver.add_force(4, 2, 1000.0);

    auto result = solver.solve_linear();

    // Assembly guard should catch the NaN element stiffness
    CHECK(solver.nan_element_count() > 0,
          "NaN coords: assembly NaN guard caught it (count=" +
          std::to_string(solver.nan_element_count()) + ")");
    CHECK(!has_nan(result.displacement),
          "NaN coords: no NaN propagated to solution");
}

void test_solution_nan_free_normal_case() {
    std::cout << "\n--- Normal case: verify solution is clean ---\n";

    auto mesh = generate_cantilever_mesh(1.0, 0.1, 0.1, 4, 1, 1);

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    ElasticMaterial mat;
    mat.E = 2.0e11;
    mat.nu = 0.3;
    solver.set_material(mat);

    auto fixed = get_nodes_at_x(mesh, 0.0);
    for (auto n : fixed) solver.fix_node(n);

    auto tip = get_nodes_at_x_max(mesh);
    for (auto n : tip) solver.add_force(n, 1, -1000.0);

    auto result = solver.solve_linear();

    CHECK(result.converged, "Normal case: solver converged");
    CHECK(!has_nan(result.displacement), "Normal case: no NaN in solution");
    CHECK(solver.nan_element_count() == 0,
          "Normal case: no NaN elements");
    CHECK(solver.zero_diagonal_count() == 0,
          "Normal case: no zero diagonals");

    Real max_d = FEMStaticSolver::max_displacement(result);
    CHECK(max_d > 0.0 && std::isfinite(max_d),
          "Normal case: positive finite displacement (max=" +
          std::to_string(max_d) + ")");
}

void test_inf_modulus() {
    std::cout << "\n--- Infinite Young's modulus ---\n";

    auto mesh = generate_cantilever_mesh(1.0, 0.1, 0.1, 1, 1, 1);

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    ElasticMaterial mat;
    mat.E = std::numeric_limits<Real>::infinity();
    mat.nu = 0.3;
    solver.set_material(mat);

    auto fixed = get_nodes_at_x(mesh, 0.0);
    for (auto n : fixed) solver.fix_node(n);

    auto tip = get_nodes_at_x_max(mesh);
    for (auto n : tip) solver.add_force(n, 1, -1000.0);

    auto result = solver.solve_linear();

    // Inf modulus → Inf stiffness → assembly guard should catch NaN/Inf in ke
    CHECK(solver.nan_element_count() > 0 || !result.converged,
          "Inf modulus: NaN guard or non-convergence (nan_elems=" +
          std::to_string(solver.nan_element_count()) +
          ", converged=" + std::to_string(result.converged) + ")");
    if (result.converged) {
        CHECK(!has_nan(result.displacement),
              "Inf modulus: no NaN in displacement if converged");
    }
}

void test_cg_solver_nan_rhs() {
    std::cout << "\n--- CG solver: NaN in RHS directly ---\n";

    // Build a small 3x3 SPD matrix
    SparseMatrix A;
    std::vector<std::vector<size_t>> pattern = {{0, 1}, {0, 1, 2}, {1, 2}};
    A.create_pattern(3, 3, pattern);
    A.set(0, 0, 4.0); A.set(0, 1, 1.0);
    A.set(1, 0, 1.0); A.set(1, 1, 5.0); A.set(1, 2, 1.0);
    A.set(2, 1, 1.0); A.set(2, 2, 3.0);

    std::vector<Real> b = {1.0, std::numeric_limits<Real>::quiet_NaN(), 1.0};
    std::vector<Real> x;

    CGSolver cg;
    cg.set_tolerance(1e-10);
    cg.set_max_iterations(100);

    auto result = cg.solve(A, b, x);

    CHECK(!result.converged,
          "CG NaN RHS: solver reports non-convergence");
    CHECK(result.diagnostic.find("NaN") != std::string::npos,
          "CG NaN RHS: diagnostic mentions NaN (\"" + result.diagnostic + "\")");
}

void test_cg_solver_singular() {
    std::cout << "\n--- CG solver: singular matrix (pAp guard) ---\n";

    // Build a rank-deficient symmetric matrix
    // Row 0 == Row 1 → singular
    SparseMatrix A;
    std::vector<std::vector<size_t>> pattern = {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
    A.create_pattern(3, 3, pattern);
    A.set(0, 0, 1.0); A.set(0, 1, 1.0); A.set(0, 2, 0.0);
    A.set(1, 0, 1.0); A.set(1, 1, 1.0); A.set(1, 2, 0.0);  // Same as row 0
    A.set(2, 0, 0.0); A.set(2, 1, 0.0); A.set(2, 2, 1.0);

    std::vector<Real> b = {1.0, 2.0, 1.0};  // Inconsistent for singular rows
    std::vector<Real> x;

    CGSolver cg;
    cg.set_tolerance(1e-10);
    cg.set_max_iterations(100);
    cg.set_preconditioner(false);

    auto result = cg.solve(A, b, x);

    // Singular matrix → should fail (pAp guard or max iterations)
    CHECK(!result.converged,
          "CG singular: solver reports non-convergence");
    // Should have a diagnostic (pAp near zero or divergence)
    bool has_diag = !result.diagnostic.empty();
    bool hit_max_iter = (result.iterations >= 100);
    CHECK(has_diag || hit_max_iter,
          "CG singular: diagnostic or max iterations reached (diag=\"" +
          result.diagnostic + "\", iters=" + std::to_string(result.iterations) + ")");
}

void test_direct_solver_nan_scan() {
    std::cout << "\n--- Direct solver: NaN from singular matrix ---\n";

    // Build a singular 2x2 matrix
    SparseMatrix A;
    std::vector<std::vector<size_t>> pattern = {{0, 1}, {0, 1}};
    A.create_pattern(2, 2, pattern);
    A.set(0, 0, 1.0); A.set(0, 1, 1.0);
    A.set(1, 0, 1.0); A.set(1, 1, 1.0);  // Singular!

    std::vector<Real> b = {1.0, 2.0};  // Inconsistent
    std::vector<Real> x;

    DirectSolver ds;
    auto result = ds.solve(A, b, x);

    CHECK(!result.converged,
          "Direct singular: solver reports non-convergence");
}

void test_zero_force_zero_displacement() {
    std::cout << "\n--- Zero force → zero displacement ---\n";

    auto mesh = generate_cantilever_mesh(1.0, 0.1, 0.1, 2, 1, 1);

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    ElasticMaterial mat;
    solver.set_material(mat);

    auto fixed = get_nodes_at_x(mesh, 0.0);
    for (auto n : fixed) solver.fix_node(n);

    // No forces applied
    auto result = solver.solve_linear();

    CHECK(result.converged, "Zero force: solver converged");
    if (result.converged) {
        Real max_d = FEMStaticSolver::max_displacement(result);
        CHECK(max_d < 1e-15,
              "Zero force: displacement ~ 0 (max=" + std::to_string(max_d) + ")");
    }
}

void test_very_soft_material() {
    std::cout << "\n--- Very soft material (E=1.0) ---\n";

    auto mesh = generate_cantilever_mesh(1.0, 0.1, 0.1, 2, 1, 1);

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    ElasticMaterial mat;
    mat.E = 1.0;  // Very soft
    mat.nu = 0.3;
    solver.set_material(mat);

    auto fixed = get_nodes_at_x(mesh, 0.0);
    for (auto n : fixed) solver.fix_node(n);

    auto tip = get_nodes_at_x_max(mesh);
    for (auto n : tip) solver.add_force(n, 1, -1.0);

    auto result = solver.solve_linear();

    CHECK(result.converged, "Very soft: solver converged");
    CHECK(!has_nan(result.displacement), "Very soft: no NaN in displacement");
}

void test_very_stiff_material() {
    std::cout << "\n--- Very stiff material (E=1e15) ---\n";

    auto mesh = generate_cantilever_mesh(1.0, 0.1, 0.1, 2, 1, 1);

    FEMStaticSolver solver;
    solver.set_mesh(mesh);
    ElasticMaterial mat;
    mat.E = 1.0e15;  // Very stiff
    mat.nu = 0.3;
    solver.set_material(mat);

    auto fixed = get_nodes_at_x(mesh, 0.0);
    for (auto n : fixed) solver.fix_node(n);

    auto tip = get_nodes_at_x_max(mesh);
    for (auto n : tip) solver.add_force(n, 1, -1e6);

    auto result = solver.solve_linear();

    CHECK(result.converged, "Very stiff: solver converged");
    CHECK(!has_nan(result.displacement), "Very stiff: no NaN in displacement");
    if (result.converged) {
        Real max_d = FEMStaticSolver::max_displacement(result);
        CHECK(std::isfinite(max_d),
              "Very stiff: finite displacement (max=" + std::to_string(max_d) + ")");
    }
}

// ============================================================================
// main
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "  FEM Robustness Guards Test Suite\n";
    std::cout << "========================================\n";

    // Category 1: Singular / ill-conditioned
    test_floating_structure();
    test_insufficient_bcs();
    test_near_incompressible();

    // Category 2: Degenerate elements
    test_coplanar_tet4();
    test_collapsed_hex8();
    test_coincident_hex8_nodes();
    test_near_zero_volume_tet4();

    // Category 3: NaN / Inf propagation
    test_negative_youngs_modulus();
    test_nan_in_force_vector();
    test_nan_in_coords();
    test_solution_nan_free_normal_case();
    test_inf_modulus();
    test_cg_solver_nan_rhs();
    test_cg_solver_singular();
    test_direct_solver_nan_scan();
    test_zero_force_zero_displacement();
    test_very_soft_material();
    test_very_stiff_material();

    // Summary
    std::cout << "\n========================================\n";
    std::cout << "  Results: " << passed_checks << "/" << total_checks
              << " passed, " << failed_checks << " failed\n";
    std::cout << "========================================\n";

    return (failed_checks > 0) ? 1 : 0;
}
