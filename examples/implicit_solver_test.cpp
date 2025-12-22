/**
 * @file implicit_solver_test.cpp
 * @brief Test implicit solver components
 *
 * Tests:
 * 1. Sparse matrix operations (CSR)
 * 2. Conjugate Gradient solver
 * 3. Direct LU solver
 * 4. Newton-Raphson nonlinear solver
 * 5. Static analysis (cantilever beam)
 * 6. Newmark-β implicit dynamics
 */

#include <nexussim/solver/implicit_solver.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace nxs;
using namespace nxs::solver;

// Test counter
static int test_count = 0;
static int pass_count = 0;

void check(bool condition, const std::string& test_name) {
    test_count++;
    if (condition) {
        pass_count++;
        std::cout << "[PASS] " << test_name << "\n";
    } else {
        std::cout << "[FAIL] " << test_name << "\n";
    }
}

// ============================================================================
// Test 1: Sparse Matrix Operations
// ============================================================================
void test_sparse_matrix() {
    std::cout << "\n=== Test 1: Sparse Matrix Operations ===\n";

    // Create 3x3 sparse matrix from COO format
    // A = [4 1 0]
    //     [1 5 2]
    //     [0 2 6]
    std::vector<size_t> rows = {0, 0, 1, 1, 1, 2, 2};
    std::vector<size_t> cols = {0, 1, 0, 1, 2, 1, 2};
    std::vector<Real> vals = {4, 1, 1, 5, 2, 2, 6};

    SparseMatrix A;
    A.from_coo(3, 3, rows, cols, vals);

    check(A.rows() == 3, "Matrix rows");
    check(A.cols() == 3, "Matrix cols");
    check(A.nnz() == 7, "Matrix nnz");

    // Check values
    check(std::abs(A.get(0, 0) - 4.0) < 1e-10, "A(0,0) = 4");
    check(std::abs(A.get(0, 1) - 1.0) < 1e-10, "A(0,1) = 1");
    check(std::abs(A.get(1, 1) - 5.0) < 1e-10, "A(1,1) = 5");
    check(std::abs(A.get(2, 2) - 6.0) < 1e-10, "A(2,2) = 6");
    check(std::abs(A.get(0, 2) - 0.0) < 1e-10, "A(0,2) = 0 (not in pattern)");

    // Matrix-vector product: y = A*x
    std::vector<Real> x = {1, 2, 3};
    std::vector<Real> y;
    A.multiply(x, y);

    // Expected: y = [4*1 + 1*2, 1*1 + 5*2 + 2*3, 2*2 + 6*3] = [6, 17, 22]
    check(std::abs(y[0] - 6.0) < 1e-10, "A*x[0] = 6");
    check(std::abs(y[1] - 17.0) < 1e-10, "A*x[1] = 17");
    check(std::abs(y[2] - 22.0) < 1e-10, "A*x[2] = 22");

    // Test add and set
    A.add(1, 1, 1.0);
    check(std::abs(A.get(1, 1) - 6.0) < 1e-10, "After add: A(1,1) = 6");

    A.set(1, 1, 5.0);
    check(std::abs(A.get(1, 1) - 5.0) < 1e-10, "After set: A(1,1) = 5");

    // Test diagonal extraction
    std::vector<Real> diag;
    A.get_diagonal(diag);
    check(std::abs(diag[0] - 4.0) < 1e-10, "Diagonal[0] = 4");
    check(std::abs(diag[1] - 5.0) < 1e-10, "Diagonal[1] = 5");
    check(std::abs(diag[2] - 6.0) < 1e-10, "Diagonal[2] = 6");
}

// ============================================================================
// Test 2: Conjugate Gradient Solver
// ============================================================================
void test_cg_solver() {
    std::cout << "\n=== Test 2: Conjugate Gradient Solver ===\n";

    // Create SPD matrix (3x3)
    // A = [4 1 0]
    //     [1 5 2]
    //     [0 2 6]
    std::vector<size_t> rows = {0, 0, 1, 1, 1, 2, 2};
    std::vector<size_t> cols = {0, 1, 0, 1, 2, 1, 2};
    std::vector<Real> vals = {4, 1, 1, 5, 2, 2, 6};

    SparseMatrix A;
    A.from_coo(3, 3, rows, cols, vals);

    // Solve A*x = b where b = A*[1,2,3]
    std::vector<Real> b_exact = {1, 2, 3};
    std::vector<Real> b;
    A.multiply(b_exact, b);

    std::vector<Real> x;

    CGSolver cg;
    cg.set_tolerance(1e-10);
    cg.set_max_iterations(100);
    cg.set_preconditioner(true);

    auto result = cg.solve(A, b, x);

    std::cout << "  CG converged: " << (result.converged ? "yes" : "no") << "\n";
    std::cout << "  Iterations: " << result.iterations << "\n";
    std::cout << "  Residual: " << result.residual << "\n";
    std::cout << "  Solution: [" << x[0] << ", " << x[1] << ", " << x[2] << "]\n";

    check(result.converged, "CG converged");
    check(result.iterations < 10, "CG converged in few iterations");
    check(std::abs(x[0] - 1.0) < 1e-6, "x[0] = 1");
    check(std::abs(x[1] - 2.0) < 1e-6, "x[1] = 2");
    check(std::abs(x[2] - 3.0) < 1e-6, "x[2] = 3");
}

// ============================================================================
// Test 3: Direct LU Solver
// ============================================================================
void test_direct_solver() {
    std::cout << "\n=== Test 3: Direct LU Solver ===\n";

    // Same matrix as CG test
    std::vector<size_t> rows = {0, 0, 1, 1, 1, 2, 2};
    std::vector<size_t> cols = {0, 1, 0, 1, 2, 1, 2};
    std::vector<Real> vals = {4, 1, 1, 5, 2, 2, 6};

    SparseMatrix A;
    A.from_coo(3, 3, rows, cols, vals);

    std::vector<Real> b_exact = {1, 2, 3};
    std::vector<Real> b;
    A.multiply(b_exact, b);

    std::vector<Real> x;

    DirectSolver direct;
    auto result = direct.solve(A, b, x);

    std::cout << "  Direct solver converged: " << (result.converged ? "yes" : "no") << "\n";
    std::cout << "  Solution: [" << x[0] << ", " << x[1] << ", " << x[2] << "]\n";

    check(result.converged, "Direct solver converged");
    check(std::abs(x[0] - 1.0) < 1e-6, "x[0] = 1");
    check(std::abs(x[1] - 2.0) < 1e-6, "x[1] = 2");
    check(std::abs(x[2] - 3.0) < 1e-6, "x[2] = 3");
}

// ============================================================================
// Test 4: Newton-Raphson Solver (Simple Nonlinear)
// ============================================================================
void test_newton_raphson() {
    std::cout << "\n=== Test 4: Newton-Raphson Solver ===\n";

    // Solve: x³ - x - 2 = 0 (solution: x ≈ 1.5214)
    // Residual: R(x) = x³ - x - 2
    // Jacobian: J = 3x² - 1

    // For vector formulation with 1 DOF
    NewtonRaphsonSolver newton;

    // Create single-entry stiffness pattern
    SparseMatrix K;
    std::vector<std::vector<size_t>> pattern = {{0}};
    K.create_pattern(1, 1, pattern);

    newton.set_residual_function(
        [](const std::vector<Real>& u, std::vector<Real>& R) {
            Real x = u[0];
            R[0] = x * x * x - x - 2.0;
        });

    newton.set_tangent_function(
        [&K](const std::vector<Real>& u, SparseMatrix& K_out) {
            Real x = u[0];
            K.set(0, 0, 3.0 * x * x - 1.0);
            K_out = K;
        });

    newton.set_tolerance(1e-10, 1e-8);
    newton.set_max_iterations(20);
    newton.set_verbose(true);

    std::vector<Real> x = {1.0};  // Initial guess
    auto result = newton.solve(x);

    std::cout << "  Newton-Raphson converged: " << (result.converged ? "yes" : "no") << "\n";
    std::cout << "  Iterations: " << result.iterations << "\n";
    std::cout << "  Solution: x = " << x[0] << "\n";
    std::cout << "  Expected: x ≈ 1.5214\n";

    check(result.converged, "Newton-Raphson converged");
    check(result.iterations <= 10, "Converged in reasonable iterations");
    check(std::abs(x[0] - 1.5214) < 0.001, "Solution correct (x ≈ 1.5214)");
}

// ============================================================================
// Test 5: Static Analysis - Simple Spring System
// ============================================================================
void test_static_spring() {
    std::cout << "\n=== Test 5: Static Analysis - Spring System ===\n";

    // Two springs in series: k1 = 1000, k2 = 2000 N/m
    // Fixed at left (node 0), force F = 100 N at right (node 2)
    //
    // [Fixed]---k1---[Node1]---k2---[Node2]---> F
    //
    // Stiffness matrix (after applying BC at node 0):
    // K = [k1+k2, -k2  ]   u1   F1=0
    //     [-k2,   k2   ] * u2 = F2=100
    //
    // Solution: u2 = F/k_equiv = 100 / (k1*k2/(k1+k2)) = 100 * (k1+k2)/(k1*k2)
    //         = 100 * 3000 / 2000000 = 0.15 m
    //         u1 = (k2/(k1+k2)) * u2 = (2000/3000) * 0.15 = 0.1 m

    Real k1 = 1000.0, k2 = 2000.0;
    Real F = 100.0;

    // Create stiffness matrix (2 free DOFs)
    SparseMatrix K;
    std::vector<size_t> rows = {0, 0, 1, 1};
    std::vector<size_t> cols = {0, 1, 0, 1};
    std::vector<Real> vals = {k1 + k2, -k2, -k2, k2};
    K.from_coo(2, 2, rows, cols, vals);

    std::vector<Real> F_ext = {0.0, F};
    std::vector<Real> u;

    StaticSolver solver;
    auto result = solver.solve_linear(K, F_ext, u);

    std::cout << "  Static solver converged: " << (result.converged ? "yes" : "no") << "\n";
    std::cout << "  u1 = " << u[0] << " m (expected 0.1)\n";
    std::cout << "  u2 = " << u[1] << " m (expected 0.15)\n";

    check(result.converged, "Static solver converged");
    check(std::abs(u[0] - 0.1) < 1e-6, "u1 = 0.1 m");
    check(std::abs(u[1] - 0.15) < 1e-6, "u2 = 0.15 m");
}

// ============================================================================
// Test 6: Nonlinear Static - Softening Spring
// ============================================================================
void test_nonlinear_static() {
    std::cout << "\n=== Test 6: Nonlinear Static - Softening Spring ===\n";

    // Nonlinear spring: F_int = k*u - c*u³
    // k = 1000 N/m, c = 100 N/m³
    // Apply F_ext = 50 N
    //
    // Solve: k*u - c*u³ = F_ext
    // 1000*u - 100*u³ = 50

    Real k = 1000.0, c = 100.0;
    Real F_ext_val = 50.0;

    SparseMatrix K;
    std::vector<std::vector<size_t>> pattern = {{0}};
    K.create_pattern(1, 1, pattern);

    StaticSolver solver;

    solver.set_stiffness_function(
        [k, c, &K](const std::vector<Real>& u, SparseMatrix& K_out) {
            // dF_int/du = k - 3*c*u²
            Real K_tangent = k - 3.0 * c * u[0] * u[0];
            K.set(0, 0, K_tangent);
            K_out = K;
        });

    solver.set_internal_force_function(
        [k, c](const std::vector<Real>& u, std::vector<Real>& F_int) {
            F_int[0] = k * u[0] - c * u[0] * u[0] * u[0];
        });

    solver.set_verbose(true);

    std::vector<Real> F_ext = {F_ext_val};
    std::vector<Real> u;

    auto result = solver.solve(F_ext, u, 1);  // Single load step

    // Check: 1000*u - 100*u³ = 50
    Real residual = k * u[0] - c * u[0] * u[0] * u[0] - F_ext_val;

    std::cout << "  Nonlinear static converged: " << (result.converged ? "yes" : "no") << "\n";
    std::cout << "  u = " << u[0] << " m\n";
    std::cout << "  Residual check: " << residual << "\n";

    check(result.converged, "Nonlinear static converged");
    check(std::abs(residual) < 1e-6, "Solution satisfies equilibrium");
    check(u[0] > 0.04 && u[0] < 0.06, "Displacement in reasonable range");
}

// ============================================================================
// Test 7: Larger Sparse System (Tridiagonal)
// ============================================================================
void test_larger_system() {
    std::cout << "\n=== Test 7: Larger Sparse System (100x100) ===\n";

    // Tridiagonal system from 1D finite difference
    // -u_{i-1} + 2*u_i - u_{i+1} = h² * f_i
    size_t n = 100;

    std::vector<size_t> rows, cols;
    std::vector<Real> vals;

    for (size_t i = 0; i < n; ++i) {
        // Diagonal
        rows.push_back(i);
        cols.push_back(i);
        vals.push_back(2.0);

        // Off-diagonals
        if (i > 0) {
            rows.push_back(i);
            cols.push_back(i - 1);
            vals.push_back(-1.0);
        }
        if (i < n - 1) {
            rows.push_back(i);
            cols.push_back(i + 1);
            vals.push_back(-1.0);
        }
    }

    SparseMatrix A;
    A.from_coo(n, n, rows, cols, vals);

    std::cout << "  Matrix size: " << n << "x" << n << "\n";
    std::cout << "  Non-zeros: " << A.nnz() << "\n";

    // RHS: all ones
    std::vector<Real> b(n, 1.0);
    std::vector<Real> x;

    CGSolver cg;
    cg.set_tolerance(1e-10);
    cg.set_max_iterations(200);
    cg.set_preconditioner(true);

    auto result = cg.solve(A, b, x);

    std::cout << "  CG converged: " << (result.converged ? "yes" : "no") << "\n";
    std::cout << "  Iterations: " << result.iterations << "\n";
    std::cout << "  Residual: " << result.residual << "\n";

    // Verify A*x ≈ b
    std::vector<Real> Ax;
    A.multiply(x, Ax);
    Real max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::abs(Ax[i] - b[i]));
    }
    std::cout << "  Max error |Ax - b|: " << max_err << "\n";

    check(result.converged, "Large system CG converged");
    check(max_err < 1e-8, "Solution accurate");
}

// ============================================================================
// Test 8: Newmark-β Integrator Setup
// ============================================================================
void test_newmark_setup() {
    std::cout << "\n=== Test 8: Newmark-β Integrator Setup ===\n";

    // Test that integrator can be initialized
    NewmarkIntegrator newmark(0.25, 0.5);  // Average acceleration

    size_t ndof = 10;
    newmark.initialize(ndof);

    // Set mass (uniform)
    std::vector<Real> M_diag(ndof, 1.0);
    newmark.set_mass(M_diag);

    // Set damping (Rayleigh)
    newmark.set_damping(0.01, 0.001);

    check(newmark.displacement().size() == ndof, "Displacement initialized");
    check(newmark.velocity().size() == ndof, "Velocity initialized");
    check(newmark.acceleration().size() == ndof, "Acceleration initialized");
    check(newmark.time() == 0.0, "Time starts at zero");

    // Set initial conditions
    newmark.displacement()[0] = 1.0;
    check(newmark.displacement()[0] == 1.0, "Can set initial displacement");
}

// ============================================================================
// Test 9: CG without Preconditioner
// ============================================================================
void test_cg_no_precond() {
    std::cout << "\n=== Test 9: CG without Preconditioner ===\n";

    // Same test as Test 2 but without Jacobi preconditioner
    std::vector<size_t> rows = {0, 0, 1, 1, 1, 2, 2};
    std::vector<size_t> cols = {0, 1, 0, 1, 2, 1, 2};
    std::vector<Real> vals = {4, 1, 1, 5, 2, 2, 6};

    SparseMatrix A;
    A.from_coo(3, 3, rows, cols, vals);

    std::vector<Real> b_exact = {1, 2, 3};
    std::vector<Real> b;
    A.multiply(b_exact, b);

    std::vector<Real> x;

    CGSolver cg;
    cg.set_tolerance(1e-10);
    cg.set_max_iterations(100);
    cg.set_preconditioner(false);  // No preconditioner

    auto result = cg.solve(A, b, x);

    std::cout << "  CG (no precond) converged: " << (result.converged ? "yes" : "no") << "\n";
    std::cout << "  Iterations: " << result.iterations << "\n";

    check(result.converged, "CG without preconditioner converged");
    check(std::abs(x[0] - 1.0) < 1e-6, "Solution correct");
}

// ============================================================================
// Test 10: Stiffness Assembly Helper
// ============================================================================
void test_stiffness_assembly() {
    std::cout << "\n=== Test 10: Element Stiffness Assembly ===\n";

    // Create 4x4 global stiffness with known pattern
    SparseMatrix K;
    std::vector<std::vector<size_t>> pattern = {
        {0, 1},     // Row 0
        {0, 1, 2},  // Row 1
        {1, 2, 3},  // Row 2
        {2, 3}      // Row 3
    };
    K.create_pattern(4, 4, pattern);

    // Element 1: nodes [0, 1], ke = [2, -1; -1, 2]
    std::vector<Index> dof_map1 = {0, 1};
    std::vector<Real> ke1 = {2, -1, -1, 2};
    K.add_element_matrix(dof_map1, ke1);

    // Element 2: nodes [1, 2], ke = [2, -1; -1, 2]
    std::vector<Index> dof_map2 = {1, 2};
    std::vector<Real> ke2 = {2, -1, -1, 2};
    K.add_element_matrix(dof_map2, ke2);

    // Element 3: nodes [2, 3], ke = [2, -1; -1, 2]
    std::vector<Index> dof_map3 = {2, 3};
    std::vector<Real> ke3 = {2, -1, -1, 2};
    K.add_element_matrix(dof_map3, ke3);

    // Check assembled stiffness
    // K = [2  -1   0   0]
    //     [-1  4  -1   0]
    //     [0  -1   4  -1]
    //     [0   0  -1   2]

    check(std::abs(K.get(0, 0) - 2.0) < 1e-10, "K(0,0) = 2");
    check(std::abs(K.get(1, 1) - 4.0) < 1e-10, "K(1,1) = 4");
    check(std::abs(K.get(0, 1) - (-1.0)) < 1e-10, "K(0,1) = -1");
    check(std::abs(K.get(1, 2) - (-1.0)) < 1e-10, "K(1,2) = -1");
    check(std::abs(K.get(3, 3) - 2.0) < 1e-10, "K(3,3) = 2");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=================================================\n";
    std::cout << "Implicit Solver Test Suite\n";
    std::cout << "=================================================\n";

    Kokkos::initialize();

    {
        test_sparse_matrix();
        test_cg_solver();
        test_direct_solver();
        test_newton_raphson();
        test_static_spring();
        test_nonlinear_static();
        test_larger_system();
        test_newmark_setup();
        test_cg_no_precond();
        test_stiffness_assembly();
    }

    Kokkos::finalize();

    std::cout << "\n=================================================\n";
    std::cout << "Results: " << pass_count << "/" << test_count << " tests passed\n";
    std::cout << "=================================================\n";

    return (pass_count == test_count) ? 0 : 1;
}
