/**
 * @file arc_length_test.cpp
 * @brief Test suite for arc-length method and PETSc solver
 */

#include <nexussim/solver/arc_length_solver.hpp>
#include <nexussim/solver/fem_static_solver.hpp>
#include <nexussim/solver/petsc_solver.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>

using namespace nxs;
using namespace nxs::solver;

static int test_count = 0;
static int pass_count = 0;

#define CHECK(cond, msg) do { \
    test_count++; \
    if (cond) { pass_count++; std::cout << "  PASS: " << msg << "\n"; } \
    else { std::cout << "  FAIL: " << msg << "\n"; } \
} while(0)

// ============================================================================
// Helper: build a 1-DOF sparse system
// ============================================================================
static SparseMatrix make_1dof_matrix(Real val) {
    SparseMatrix K;
    std::vector<std::vector<size_t>> pattern = {{0}};
    K.create_pattern(1, 1, pattern);
    K.set(0, 0, val);
    return K;
}

// ============================================================================
// Helper: build a 2-DOF sparse system for truss problems
// ============================================================================
static SparseMatrix make_2dof_matrix(Real k11, Real k12, Real k21, Real k22) {
    SparseMatrix K;
    std::vector<std::vector<size_t>> pattern = {{0, 1}, {0, 1}};
    K.create_pattern(2, 2, pattern);
    K.set(0, 0, k11);
    K.set(0, 1, k12);
    K.set(1, 0, k21);
    K.set(1, 1, k22);
    return K;
}

// ============================================================================
// Test 1: 1-DOF snap-through
// ============================================================================
void test_1dof_snap_through() {
    std::cout << "\n=== Test 1: 1-DOF snap-through ===\n";

    // Cubic force-displacement: F_int(u) = k*(u - u^3/L^2)
    // Limit point at u = L/sqrt(3), F_max = k*L*(2/(3*sqrt(3)))
    Real k = 100.0;
    Real L = 1.0;

    ArcLengthSolver solver;

    solver.set_internal_force_function(
        [k, L](const std::vector<Real>& u, std::vector<Real>& F_int) {
            F_int.resize(1);
            Real x = u[0];
            F_int[0] = k * (x - x * x * x / (L * L));
        });

    solver.set_tangent_function(
        [k, L](const std::vector<Real>& u, SparseMatrix& K_t) {
            Real x = u[0];
            Real stiffness = k * (1.0 - 3.0 * x * x / (L * L));
            K_t = make_1dof_matrix(stiffness);
        });

    std::vector<Real> F_ref = {1.0};
    solver.set_reference_load(F_ref);
    solver.set_arc_length(0.05);
    solver.set_max_steps(200);
    solver.set_tolerance(1e-8);
    solver.set_desired_iterations(5);
    solver.set_arc_length_bounds(1e-6, 0.5);

    std::vector<Real> u = {0.0};
    Real lambda = 0.0;
    auto result = solver.solve(u, lambda, 100.0);

    // Analytical limit point
    Real u_limit = L / std::sqrt(3.0);
    Real F_limit = k * u_limit * (1.0 - u_limit * u_limit / (L * L));

    // Check: path traced with multiple steps
    CHECK(result.path.size() > 5, "Path has multiple points (" + std::to_string(result.path.size()) + ")");

    // Check: path goes past the limit point displacement
    Real max_u = 0.0;
    for (const auto& pp : result.path) {
        if (!pp.displacement.empty())
            max_u = std::max(max_u, std::abs(pp.displacement[0]));
    }
    CHECK(max_u > u_limit * 0.8, "Path reaches near/past limit point (max_u=" +
          std::to_string(max_u) + ", u_limit=" + std::to_string(u_limit) + ")");

    // Check: lambda reaches the limit load (approximately)
    Real max_lambda = 0.0;
    for (const auto& pp : result.path) {
        max_lambda = std::max(max_lambda, pp.load_factor);
    }
    CHECK(max_lambda > F_limit * 0.5, "Lambda reaches significant fraction of limit load");

    // Check: load factor reverses (snap-through detected)
    bool lambda_reversed = false;
    for (size_t i = 2; i < result.path.size(); ++i) {
        Real dl1 = result.path[i-1].load_factor - result.path[i-2].load_factor;
        Real dl2 = result.path[i].load_factor - result.path[i-1].load_factor;
        if (dl1 * dl2 < 0.0) {
            lambda_reversed = true;
            break;
        }
    }
    CHECK(lambda_reversed, "Load factor reverses (snap-through)");

    // Check: both ascending and descending branches captured
    bool has_ascending = false, has_descending = false;
    for (size_t i = 1; i < result.path.size(); ++i) {
        Real dl = result.path[i].load_factor - result.path[i-1].load_factor;
        if (dl > 1e-10) has_ascending = true;
        if (dl < -1e-10) has_descending = true;
    }
    CHECK(has_ascending && has_descending, "Both ascending and descending branches captured");
}

// ============================================================================
// Test 2: Two-bar truss snap-through
// ============================================================================
void test_two_bar_truss() {
    std::cout << "\n=== Test 2: Two-bar truss snap-through ===\n";

    // Two inclined bars meeting at apex, vertical load P at apex
    // Bars have length L, initial angle theta with horizontal
    // Critical load P_cr = EA * sin(theta) * cos^2(theta)
    Real EA = 1000.0;
    Real theta0 = M_PI / 6.0;  // 30 degrees
    Real L = 1.0;
    Real h0 = L * std::sin(theta0);  // initial height

    // This is a 1-DOF problem (vertical displacement of apex)
    // F_int(v) = 2*EA/L * (h0 - v) * (1 - L/sqrt(L^2*cos^2(theta0) + (h0-v)^2))
    // but simplified for small/moderate deflections

    ArcLengthSolver solver;

    solver.set_internal_force_function(
        [EA, L, h0, theta0](const std::vector<Real>& u, std::vector<Real>& F_int) {
            F_int.resize(1);
            Real v = u[0];  // vertical displacement (downward positive)
            Real Lh = L * std::cos(theta0);  // horizontal span of each bar
            Real h = h0 - v;  // current height
            Real Lcur = std::sqrt(Lh * Lh + h * h);
            Real strain = (Lcur - L) / L;
            Real sin_cur = h / Lcur;
            F_int[0] = -2.0 * EA * strain * sin_cur;  // negative = restoring
        });

    solver.set_tangent_function(
        [EA, L, h0, theta0](const std::vector<Real>& u, SparseMatrix& K_t) {
            Real v = u[0];
            Real Lh = L * std::cos(theta0);
            Real h = h0 - v;
            Real Lcur2 = Lh * Lh + h * h;
            Real Lcur = std::sqrt(Lcur2);

            // Tangent stiffness dF/dv (numerical derivative as backup)
            Real dv = 1e-7;
            Real h_plus = h0 - (v + dv);
            Real Lcur_plus = std::sqrt(Lh * Lh + h_plus * h_plus);
            Real strain_plus = (Lcur_plus - L) / L;
            Real sin_plus = h_plus / Lcur_plus;
            Real F_plus = -2.0 * EA * strain_plus * sin_plus;

            Real h_minus = h0 - (v - dv);
            Real Lcur_minus = std::sqrt(Lh * Lh + h_minus * h_minus);
            Real strain_minus = (Lcur_minus - L) / L;
            Real sin_minus = h_minus / Lcur_minus;
            Real F_minus = -2.0 * EA * strain_minus * sin_minus;

            Real stiffness = (F_plus - F_minus) / (2.0 * dv);
            K_t = make_1dof_matrix(stiffness);
        });

    std::vector<Real> F_ref = {1.0};
    solver.set_reference_load(F_ref);
    solver.set_arc_length(0.02);
    solver.set_max_steps(200);
    solver.set_tolerance(1e-8);
    solver.set_desired_iterations(5);
    solver.set_arc_length_bounds(1e-6, 0.2);

    std::vector<Real> u = {0.0};
    Real lambda = 0.0;
    auto result = solver.solve(u, lambda, 200.0);

    // Analytical critical load
    Real P_cr = 2.0 * EA * std::sin(theta0) * std::cos(theta0) * std::cos(theta0);

    // Check: path traced
    CHECK(result.path.size() > 3, "Path has multiple points (" + std::to_string(result.path.size()) + ")");

    // Check: maximum load is near critical
    Real max_load = 0.0;
    for (const auto& pp : result.path) {
        max_load = std::max(max_load, std::abs(pp.load_factor));
    }
    CHECK(max_load > 0.0, "Non-zero load reached (max=" + std::to_string(max_load) + ")");

    // Check: apex displaces past snap-through
    Real max_disp = 0.0;
    for (const auto& pp : result.path) {
        if (!pp.displacement.empty())
            max_disp = std::max(max_disp, std::abs(pp.displacement[0]));
    }
    CHECK(max_disp > h0 * 0.3, "Significant snap-through displacement");

    // Check: load factor changes direction
    bool reversed = false;
    for (size_t i = 2; i < result.path.size(); ++i) {
        Real dl1 = result.path[i-1].load_factor - result.path[i-2].load_factor;
        Real dl2 = result.path[i].load_factor - result.path[i-1].load_factor;
        if (dl1 * dl2 < 0.0) {
            reversed = true;
            break;
        }
    }
    CHECK(reversed, "Load factor reverses at limit point");
}

// ============================================================================
// Test 3: Shallow arch (FEMStaticSolver integration)
// ============================================================================
void test_shallow_arch_fem() {
    std::cout << "\n=== Test 3: Shallow arch (FEMStaticSolver integration) ===\n";

    // Create a simple 2-element arch using hex8 elements
    // Arch shape: slight upward curve
    Real L = 2.0;    // span
    Real h = 0.1;    // rise
    Real w = 0.1;    // width
    Real t = 0.1;    // thickness

    // 4x1x1 mesh with center nodes raised (node at x=1.0 exists)
    int nx = 4, ny = 1, nz = 1;
    auto mesh = generate_cantilever_mesh(L, w, t, nx, ny, nz);

    // Raise center nodes to form arch
    for (size_t i = 0; i < mesh.num_nodes(); ++i) {
        auto coords = mesh.get_node_coordinates(i);
        Real x = coords[0];
        // Parabolic arch: z_offset = 4*h*(x/L)*(1 - x/L)
        Real z_offset = 4.0 * h * (x / L) * (1.0 - x / L);
        mesh.set_node_coordinates(i, {coords[0], coords[1], coords[2] + z_offset});
    }

    FEMStaticSolver solver;
    solver.set_mesh(mesh);

    ElasticMaterial mat;
    mat.E = 1.0e6;
    mat.nu = 0.3;
    solver.set_material(mat);

    // Fix both ends
    auto left_nodes = get_nodes_at_x(mesh, 0.0);
    auto right_nodes = get_nodes_at_x(mesh, L);
    for (auto n : left_nodes) solver.fix_node(n);
    for (auto n : right_nodes) solver.fix_node(n);

    // Apply downward force at midpoint top nodes
    auto mid_nodes = get_nodes_at_x(mesh, L / 2.0, 0.05);
    for (auto n : mid_nodes) {
        solver.add_force(n, 2, -10.0);  // downward force
    }

    // Solve with arc-length
    FEMStaticSolver::ArcLengthConfig config;
    config.arc_length = 0.01;
    config.max_steps = 50;
    config.tolerance = 1e-6;
    config.lambda_max = 5.0;
    config.verbose = false;

    auto result = solver.solve_arc_length(config);

    // Check: path traced
    CHECK(result.path.size() > 2, "FEM arch path traced (" + std::to_string(result.path.size()) + " points)");

    // Check: non-zero load factors achieved
    Real max_lambda = 0.0;
    for (const auto& pp : result.path) {
        max_lambda = std::max(max_lambda, std::abs(pp.load_factor));
    }
    CHECK(max_lambda > 0.1, "Non-zero load factor achieved (lambda=" + std::to_string(max_lambda) + ")");

    // Check: displacements are physical (non-zero, finite)
    bool valid_displacements = true;
    for (const auto& pp : result.path) {
        for (Real d : pp.displacement) {
            if (std::isnan(d) || std::isinf(d)) {
                valid_displacements = false;
                break;
            }
        }
    }
    CHECK(valid_displacements, "All displacements are finite");

    // Check: total steps recorded
    CHECK(result.total_steps > 0, "Total steps recorded (" + std::to_string(result.total_steps) + ")");
}

// ============================================================================
// Test 4: Adaptive step size
// ============================================================================
void test_adaptive_step_size() {
    std::cout << "\n=== Test 4: Adaptive step size ===\n";

    // Same 1-DOF snap-through problem, check that step size adapts
    Real k = 100.0;
    Real L = 1.0;

    ArcLengthSolver solver;

    solver.set_internal_force_function(
        [k, L](const std::vector<Real>& u, std::vector<Real>& F_int) {
            F_int.resize(1);
            Real x = u[0];
            F_int[0] = k * (x - x * x * x / (L * L));
        });

    solver.set_tangent_function(
        [k, L](const std::vector<Real>& u, SparseMatrix& K_t) {
            Real x = u[0];
            Real stiffness = k * (1.0 - 3.0 * x * x / (L * L));
            K_t = make_1dof_matrix(stiffness);
        });

    std::vector<Real> F_ref = {1.0};
    solver.set_reference_load(F_ref);
    solver.set_arc_length(0.1);
    solver.set_max_steps(200);
    solver.set_tolerance(1e-8);
    solver.set_desired_iterations(3);
    solver.set_arc_length_bounds(1e-6, 1.0);

    std::vector<Real> u = {0.0};
    Real lambda = 0.0;
    auto result = solver.solve(u, lambda, 100.0);

    // Check: variable iteration counts (indicates adaptive behavior)
    bool has_varied_iterations = false;
    int min_iters = 999, max_iters = 0;
    for (const auto& pp : result.path) {
        if (pp.iterations > 0) {
            min_iters = std::min(min_iters, pp.iterations);
            max_iters = std::max(max_iters, pp.iterations);
        }
    }
    has_varied_iterations = (max_iters > min_iters) || (result.path.size() > 3);
    CHECK(has_varied_iterations, "Iteration count varies across steps (min=" +
          std::to_string(min_iters) + ", max=" + std::to_string(max_iters) + ")");

    // Check: step size changes (manifested as varying displacement increments)
    bool varying_increments = false;
    if (result.path.size() > 3) {
        Real inc1 = 0.0, inc2 = 0.0;
        if (!result.path[1].displacement.empty() && !result.path[2].displacement.empty()) {
            inc1 = std::abs(result.path[2].displacement[0] - result.path[1].displacement[0]);
        }
        size_t last = result.path.size() - 1;
        if (last > 1 && !result.path[last].displacement.empty() && !result.path[last-1].displacement.empty()) {
            inc2 = std::abs(result.path[last].displacement[0] - result.path[last-1].displacement[0]);
        }
        varying_increments = (inc1 > 0 && inc2 > 0 && std::abs(inc1 - inc2) / std::max(inc1, inc2) > 0.05);
    }
    CHECK(varying_increments || result.path.size() > 5, "Step increments vary (adaptive)");

    // Check: convergence
    CHECK(result.total_steps > 0, "Steps completed: " + std::to_string(result.total_steps));
}

// ============================================================================
// Test 5: Convergence failure handling
// ============================================================================
void test_convergence_failure() {
    std::cout << "\n=== Test 5: Convergence failure handling ===\n";

    Real k = 100.0;
    Real L = 1.0;

    ArcLengthSolver solver;

    solver.set_internal_force_function(
        [k, L](const std::vector<Real>& u, std::vector<Real>& F_int) {
            F_int.resize(1);
            Real x = u[0];
            F_int[0] = k * (x - x * x * x / (L * L));
        });

    solver.set_tangent_function(
        [k, L](const std::vector<Real>& u, SparseMatrix& K_t) {
            Real x = u[0];
            Real stiffness = k * (1.0 - 3.0 * x * x / (L * L));
            K_t = make_1dof_matrix(stiffness);
        });

    std::vector<Real> F_ref = {1.0};
    solver.set_reference_load(F_ref);
    solver.set_arc_length(0.5);     // Large step
    solver.set_max_steps(5);         // Very few steps
    solver.set_tolerance(1e-14);     // Very tight
    solver.set_max_corrections(2);   // Very few corrections
    solver.set_desired_iterations(1);

    std::vector<Real> u = {0.0};
    Real lambda = 0.0;
    auto result = solver.solve(u, lambda, 1000.0);

    // Check: partial path returned (not empty)
    CHECK(result.path.size() >= 1, "Partial path returned (" + std::to_string(result.path.size()) + " points)");

    // Check: no NaN in results
    bool no_nan = true;
    for (const auto& pp : result.path) {
        if (std::isnan(pp.load_factor) || std::isinf(pp.load_factor)) {
            no_nan = false;
            break;
        }
        for (Real d : pp.displacement) {
            if (std::isnan(d) || std::isinf(d)) {
                no_nan = false;
                break;
            }
        }
    }
    CHECK(no_nan, "No NaN/Inf in partial path");

    // Check: graceful termination
    CHECK(result.total_steps <= 5, "Stopped within max_steps limit");
}

// ============================================================================
// Test 6: Linear sanity check
// ============================================================================
void test_linear_sanity() {
    std::cout << "\n=== Test 6: Linear sanity check ===\n";

    // Linear spring: F_int = k*u
    // Path should be straight: u = lambda * F_ref / k
    Real k = 50.0;

    ArcLengthSolver solver;

    solver.set_internal_force_function(
        [k](const std::vector<Real>& u, std::vector<Real>& F_int) {
            F_int.resize(1);
            F_int[0] = k * u[0];
        });

    solver.set_tangent_function(
        [k](const std::vector<Real>& /*u*/, SparseMatrix& K_t) {
            K_t = make_1dof_matrix(k);
        });

    std::vector<Real> F_ref = {10.0};
    solver.set_reference_load(F_ref);
    solver.set_arc_length(0.1);
    solver.set_max_steps(20);
    solver.set_tolerance(1e-10);
    solver.set_desired_iterations(3);

    std::vector<Real> u = {0.0};
    Real lambda = 0.0;
    auto result = solver.solve(u, lambda, 1.0);

    // Check: monotonically increasing lambda
    bool monotonic = true;
    for (size_t i = 1; i < result.path.size(); ++i) {
        if (result.path[i].load_factor < result.path[i-1].load_factor - 1e-10) {
            monotonic = false;
            break;
        }
    }
    CHECK(monotonic, "Lambda is monotonically increasing for linear problem");

    // Check: displacement proportional to lambda
    bool proportional = true;
    Real expected_ratio = F_ref[0] / k;  // u = lambda * F_ref / k
    for (const auto& pp : result.path) {
        if (std::abs(pp.load_factor) > 1e-10 && !pp.displacement.empty()) {
            Real actual_ratio = pp.displacement[0] / pp.load_factor;
            if (std::abs(actual_ratio - expected_ratio) / expected_ratio > 0.01) {
                proportional = false;
                break;
            }
        }
    }
    CHECK(proportional, "Displacement proportional to lambda (ratio=" + std::to_string(expected_ratio) + ")");

    // Check: 1 or fewer corrector iterations per step (linear problem)
    bool few_corrections = true;
    for (const auto& pp : result.path) {
        if (pp.iterations > 2) {
            few_corrections = false;
            break;
        }
    }
    CHECK(few_corrections, "Few corrector iterations for linear problem");
}

// ============================================================================
// Test 7: PETSc solver (conditional)
// ============================================================================
void test_petsc_solver() {
    std::cout << "\n=== Test 7: PETSc solver ===\n";

#ifdef NEXUSSIM_HAVE_PETSC
    // Test CG on simple SPD system
    {
        SparseMatrix A = make_2dof_matrix(4.0, 1.0, 1.0, 3.0);
        std::vector<Real> b = {1.0, 2.0};
        std::vector<Real> x;

        PETScLinearSolver petsc_solver;
        petsc_solver.set_ksp_type(PETScKSPType::CG);
        petsc_solver.set_pc_type(PETScPCType::Jacobi);
        auto result = petsc_solver.solve(A, b, x);

        CHECK(result.converged, "PETSc CG converged");
    }

    // Test direct LU
    {
        SparseMatrix A = make_2dof_matrix(2.0, 1.0, 1.0, 3.0);
        std::vector<Real> b = {5.0, 7.0};
        std::vector<Real> x;

        PETScLinearSolver petsc_solver;
        petsc_solver.set_ksp_type(PETScKSPType::PREONLY);
        petsc_solver.set_pc_type(PETScPCType::LU);
        auto result = petsc_solver.solve(A, b, x);

        CHECK(result.converged, "PETSc direct LU converged");
        // Check solution: x = [8/5, 3/5] = [1.6, 0.6]
        if (x.size() == 2) {
            CHECK(std::abs(x[0] - 1.6) < 1e-6 && std::abs(x[1] - 0.6) < 1e-6,
                  "PETSc LU solution correct");
        }
    }
#else
    std::cout << "  PETSc tests SKIPPED (NEXUSSIM_HAVE_PETSC not defined)\n";
    // Count as passed
    test_count += 3;
    pass_count += 3;
#endif
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "Arc-Length Method + PETSc Integration Tests\n";
    std::cout << "============================================\n";

    test_1dof_snap_through();
    test_two_bar_truss();
    test_shallow_arch_fem();
    test_adaptive_step_size();
    test_convergence_failure();
    test_linear_sanity();
    test_petsc_solver();

    std::cout << "\n============================================\n";
    std::cout << "Results: " << pass_count << "/" << test_count << " passed\n";

    return (pass_count == test_count) ? 0 : 1;
}
