/**
 * @file sparse_solver_test.cpp
 * @brief Test suite for sparse matrix and CG solver
 *
 * Tests:
 * 1. CSR matrix construction
 * 2. Matrix-vector multiplication
 * 3. Jacobi preconditioner
 * 4. CG solver (simple system)
 * 5. CG solver (3D Laplacian)
 * 6. Stiffness assembly
 * 7. Boundary condition application
 * 8. FEM-like system solve
 */

#include <nexussim/solver/sparse_matrix.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

using namespace nxs;
using namespace nxs::solver;

constexpr Real TOL = 1.0e-8;

int tests_passed = 0;
int tests_total = 0;

void check(bool condition, const std::string& test_name) {
    tests_total++;
    if (condition) {
        tests_passed++;
        std::cout << "  [PASS] " << test_name << "\n";
    } else {
        std::cout << "  [FAIL] " << test_name << "\n";
    }
}

// Test 1: CSR matrix construction
void test_csr_construction() {
    std::cout << "\n=== Test 1: CSR Matrix Construction ===\n";

    // Create 4x4 matrix:
    // [ 4 -1  0 -1]
    // [-1  4 -1  0]
    // [ 0 -1  4 -1]
    // [-1  0 -1  4]

    std::vector<Index> rows = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
    std::vector<Index> cols = {0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3};
    std::vector<Real> vals = {4, -1, -1, -1, 4, -1, -1, 4, -1, -1, -1, 4};

    SparseMatrixCSR A(4, 4);
    A.build_from_coo(rows, cols, vals);

    check(A.num_rows() == 4, "Correct number of rows");
    check(A.num_cols() == 4, "Correct number of columns");
    check(A.nnz() == 12, "Correct NNZ count");

    // Check specific entries
    check(std::abs(A.get(0, 0) - 4.0) < TOL, "A[0,0] = 4");
    check(std::abs(A.get(0, 1) - (-1.0)) < TOL, "A[0,1] = -1");
    check(std::abs(A.get(0, 2)) < TOL, "A[0,2] = 0");
    check(std::abs(A.get(1, 0) - (-1.0)) < TOL, "A[1,0] = -1");

    std::cout << "  Matrix structure:\n";
    A.print_stats();
}

// Test 2: Matrix-vector multiplication
void test_spmv() {
    std::cout << "\n=== Test 2: Matrix-Vector Multiplication ===\n";

    // Simple 3x3 diagonal matrix
    std::vector<Index> rows = {0, 1, 2};
    std::vector<Index> cols = {0, 1, 2};
    std::vector<Real> vals = {2.0, 3.0, 4.0};

    SparseMatrixCSR A(3, 3);
    A.build_from_coo(rows, cols, vals);

    std::vector<Real> x = {1.0, 2.0, 3.0};
    std::vector<Real> y;
    A.multiply(x, y);

    check(std::abs(y[0] - 2.0) < TOL, "y[0] = 2*1 = 2");
    check(std::abs(y[1] - 6.0) < TOL, "y[1] = 3*2 = 6");
    check(std::abs(y[2] - 12.0) < TOL, "y[2] = 4*3 = 12");

    // Test with identity matrix
    std::vector<Index> rows_I = {0, 1, 2, 3};
    std::vector<Index> cols_I = {0, 1, 2, 3};
    std::vector<Real> vals_I = {1, 1, 1, 1};

    SparseMatrixCSR I(4, 4);
    I.build_from_coo(rows_I, cols_I, vals_I);

    std::vector<Real> x4 = {1.0, 2.0, 3.0, 4.0};
    std::vector<Real> y4;
    I.multiply(x4, y4);

    bool identity_correct = true;
    for (int i = 0; i < 4; ++i) {
        if (std::abs(y4[i] - x4[i]) > TOL) identity_correct = false;
    }
    check(identity_correct, "Identity matrix: I*x = x");
}

// Test 3: Jacobi preconditioner
void test_jacobi_precond() {
    std::cout << "\n=== Test 3: Jacobi Preconditioner ===\n";

    // Diagonal matrix: diag(2, 4, 8)
    std::vector<Index> rows = {0, 1, 2};
    std::vector<Index> cols = {0, 1, 2};
    std::vector<Real> vals = {2.0, 4.0, 8.0};

    SparseMatrixCSR A(3, 3);
    A.build_from_coo(rows, cols, vals);

    JacobiPreconditioner precond;
    precond.setup(A);

    std::vector<Real> r = {2.0, 8.0, 16.0};
    std::vector<Real> z;
    precond.apply(r, z);

    // z = D^{-1} * r
    check(std::abs(z[0] - 1.0) < TOL, "z[0] = 2/2 = 1");
    check(std::abs(z[1] - 2.0) < TOL, "z[1] = 8/4 = 2");
    check(std::abs(z[2] - 2.0) < TOL, "z[2] = 16/8 = 2");
}

// Test 4: CG solver (simple system)
void test_cg_simple() {
    std::cout << "\n=== Test 4: CG Solver (Simple System) ===\n";

    // Solve: [2 0; 0 3] * x = [4; 9]
    // Solution: x = [2; 3]
    std::vector<Index> rows = {0, 1};
    std::vector<Index> cols = {0, 1};
    std::vector<Real> vals = {2.0, 3.0};

    SparseMatrixCSR A(2, 2);
    A.build_from_coo(rows, cols, vals);

    std::vector<Real> b = {4.0, 9.0};
    std::vector<Real> x(2, 0.0);

    ConjugateGradientSolver::Config config;
    config.tolerance = 1.0e-12;
    config.max_iterations = 100;
    config.verbose = false;

    ConjugateGradientSolver cg(config);
    JacobiPreconditioner precond;
    precond.setup(A);

    SolverResult result = cg.solve(A, b, x, &precond);

    check(result.converged, "CG converged");
    check(result.iterations <= 5, "Converged quickly (<=5 iterations)");
    check(std::abs(x[0] - 2.0) < 1e-10, "x[0] = 2");
    check(std::abs(x[1] - 3.0) < 1e-10, "x[1] = 3");

    std::cout << "  Iterations: " << result.iterations << "\n";
    std::cout << "  Residual: " << result.residual_norm << "\n";
}

// Test 5: CG solver (3D Laplacian)
void test_cg_laplacian() {
    std::cout << "\n=== Test 5: CG Solver (3D Laplacian) ===\n";

    // Create 1D Laplacian: -d²u/dx² with Dirichlet BC
    // [-2  1  0  0  0]
    // [ 1 -2  1  0  0]
    // [ 0  1 -2  1  0]
    // [ 0  0  1 -2  1]
    // [ 0  0  0  1 -2]

    const int n = 100;  // Grid size
    std::vector<Index> rows, cols;
    std::vector<Real> vals;

    for (int i = 0; i < n; ++i) {
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

    SparseMatrixCSR A(n, n);
    A.build_from_coo(rows, cols, vals);

    // RHS: constant source
    std::vector<Real> b(n, 1.0);
    std::vector<Real> x(n, 0.0);

    ConjugateGradientSolver::Config config;
    config.tolerance = 1.0e-10;
    config.max_iterations = 500;
    config.verbose = false;

    ConjugateGradientSolver cg(config);

    // Test with Jacobi preconditioner
    JacobiPreconditioner jacobi;
    jacobi.setup(A);

    auto start = std::chrono::high_resolution_clock::now();
    SolverResult result = cg.solve(A, b, x, &jacobi);
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    check(result.converged, "CG converged for Laplacian");

    // Verify solution: A*x should equal b
    std::vector<Real> Ax;
    A.multiply(x, Ax);
    Real residual = 0.0;
    for (int i = 0; i < n; ++i) {
        residual += (Ax[i] - b[i]) * (Ax[i] - b[i]);
    }
    residual = std::sqrt(residual);
    check(residual < 1e-8, "Solution satisfies A*x = b");

    std::cout << "  Grid size: " << n << "\n";
    std::cout << "  Iterations: " << result.iterations << "\n";
    std::cout << "  Final residual: " << result.residual_norm << "\n";
    std::cout << "  Time: " << time_ms << " ms\n";

    // Test with SSOR preconditioner
    std::fill(x.begin(), x.end(), 0.0);
    SSORPreconditioner ssor(1.5);
    ssor.setup(A);

    start = std::chrono::high_resolution_clock::now();
    SolverResult result_ssor = cg.solve(A, b, x, &ssor);
    end = std::chrono::high_resolution_clock::now();
    time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    check(result_ssor.converged, "CG with SSOR converged");
    std::cout << "  SSOR iterations: " << result_ssor.iterations << "\n";
    std::cout << "  SSOR time: " << time_ms << " ms\n";
}

// Test 6: Stiffness assembly
void test_stiffness_assembly() {
    std::cout << "\n=== Test 6: Stiffness Assembly ===\n";

    // Simulate assembling 2-element bar (3 nodes, 3 DOFs)
    // Elements: [0,1] and [1,2]
    // Element stiffness (1D bar): k * [1, -1; -1, 1]

    const Index num_dof = 3;
    StiffnessAssembler assembler(num_dof);

    Real k = 1000.0;  // Stiffness
    Real K_elem[4] = {k, -k, -k, k};

    // Element 1: nodes 0, 1
    std::vector<Index> elem1_dofs = {0, 1};
    assembler.add_element(elem1_dofs, K_elem, 2);

    // Element 2: nodes 1, 2
    std::vector<Index> elem2_dofs = {1, 2};
    assembler.add_element(elem2_dofs, K_elem, 2);

    SparseMatrixCSR K = assembler.build();

    // Expected global stiffness:
    // [ k   -k    0 ]
    // [-k   2k   -k ]
    // [ 0   -k    k ]

    check(std::abs(K.get(0, 0) - k) < TOL, "K[0,0] = k");
    check(std::abs(K.get(0, 1) - (-k)) < TOL, "K[0,1] = -k");
    check(std::abs(K.get(0, 2)) < TOL, "K[0,2] = 0");
    check(std::abs(K.get(1, 1) - 2*k) < TOL, "K[1,1] = 2k");
    check(std::abs(K.get(2, 2) - k) < TOL, "K[2,2] = k");

    std::cout << "  Assembled stiffness matrix:\n";
    K.print_dense(std::cout);
}

// Test 7: Boundary condition application
void test_boundary_conditions() {
    std::cout << "\n=== Test 7: Boundary Condition Application ===\n";

    // 3-DOF system with BC: u[0] = 1.0

    // Initial system: K * u = f
    // K = [2, -1, 0; -1, 2, -1; 0, -1, 2]
    // f = [0, 0, 0]

    std::vector<Index> rows = {0, 0, 1, 1, 1, 2, 2};
    std::vector<Index> cols = {0, 1, 0, 1, 2, 1, 2};
    std::vector<Real> vals = {2, -1, -1, 2, -1, -1, 2};

    SparseMatrixCSR K(3, 3);
    K.build_from_coo(rows, cols, vals);

    std::vector<Real> f = {0, 0, 0};

    // Apply BC: u[0] = 1.0
    std::vector<Index> bc_dofs = {0};
    std::vector<Real> bc_vals = {1.0};

    BoundaryConditionApplicator::apply_dirichlet(K, f, bc_dofs, bc_vals);

    // After BC: row 0 should be [2, 0, 0] with f[0] = 2*1 = 2
    // Row 1 should be [0, 2, -1] with f[1] = -(-1)*1 = 1
    check(std::abs(K.get(0, 0) - 2.0) < TOL, "K[0,0] preserved diagonal");
    check(std::abs(K.get(0, 1)) < TOL, "K[0,1] zeroed");
    check(std::abs(K.get(1, 0)) < TOL, "K[1,0] zeroed (symmetry)");
    check(std::abs(f[0] - 2.0) < TOL, "f[0] = diag * bc_val");
    check(std::abs(f[1] - 1.0) < TOL, "f[1] modified for BC");

    // Solve the modified system
    std::vector<Real> u(3, 0.0);
    ConjugateGradientSolver cg;
    JacobiPreconditioner precond;
    precond.setup(K);

    SolverResult result = cg.solve(K, f, u, &precond);

    check(result.converged, "System solved after BC");
    check(std::abs(u[0] - 1.0) < 1e-8, "u[0] = 1.0 (BC enforced)");

    std::cout << "  Solution: u = [" << u[0] << ", " << u[1] << ", " << u[2] << "]\n";
}

// Test 8: FEM-like system solve
void test_fem_system() {
    std::cout << "\n=== Test 8: FEM-Like System Solve ===\n";

    // 1D bar under tension
    // Fixed at x=0, force F at x=L
    // n elements, n+1 nodes

    const int n_elem = 10;
    const int n_nodes = n_elem + 1;
    const Real L = 1.0;
    const Real E = 200e9;  // Steel, Pa
    const Real A = 0.01;   // 100 cm², m²
    const Real F = 1000.0; // 1 kN

    const Real k_elem = E * A / (L / n_elem);  // Element stiffness

    // Assemble stiffness
    StiffnessAssembler assembler(n_nodes);
    Real K_e[4] = {k_elem, -k_elem, -k_elem, k_elem};

    for (int e = 0; e < n_elem; ++e) {
        std::vector<Index> elem_dofs = {static_cast<Index>(e), static_cast<Index>(e + 1)};
        assembler.add_element(elem_dofs, K_e, 2);
    }

    SparseMatrixCSR K = assembler.build();

    // Force vector: F at last node
    std::vector<Real> f(n_nodes, 0.0);
    f[n_nodes - 1] = F;

    // Apply BC: u[0] = 0
    std::vector<Index> bc_dofs = {0};
    std::vector<Real> bc_vals = {0.0};
    BoundaryConditionApplicator::apply_dirichlet(K, f, bc_dofs, bc_vals);

    // Solve
    std::vector<Real> u(n_nodes, 0.0);
    ConjugateGradientSolver::Config config;
    config.tolerance = 1.0e-12;
    config.max_iterations = 100;

    ConjugateGradientSolver cg(config);
    JacobiPreconditioner precond;
    precond.setup(K);

    SolverResult result = cg.solve(K, f, u, &precond);

    check(result.converged, "FEM system converged");

    // Analytical solution: u(x) = F*x/(E*A)
    Real u_tip_analytical = F * L / (E * A);
    Real u_tip_computed = u[n_nodes - 1];

    Real error = std::abs(u_tip_computed - u_tip_analytical) / u_tip_analytical;
    check(error < 1e-10, "Tip displacement matches analytical");

    std::cout << "  Elements: " << n_elem << "\n";
    std::cout << "  Iterations: " << result.iterations << "\n";
    std::cout << "  u_tip (computed):   " << u_tip_computed * 1e6 << " μm\n";
    std::cout << "  u_tip (analytical): " << u_tip_analytical * 1e6 << " μm\n";
    std::cout << "  Relative error: " << error << "\n";
}

// Test 9: Large sparse system performance
void test_performance() {
    std::cout << "\n=== Test 9: Performance (Large System) ===\n";

    // 3D Laplacian on nx × ny × nz grid
    const int nx = 20, ny = 20, nz = 20;
    const int n = nx * ny * nz;

    auto idx = [&](int i, int j, int k) { return i + nx * (j + ny * k); };

    std::vector<Index> rows, cols;
    std::vector<Real> vals;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int node = idx(i, j, k);

                // Diagonal
                rows.push_back(node);
                cols.push_back(node);
                vals.push_back(6.0);

                // Neighbors
                if (i > 0) { rows.push_back(node); cols.push_back(idx(i-1, j, k)); vals.push_back(-1.0); }
                if (i < nx-1) { rows.push_back(node); cols.push_back(idx(i+1, j, k)); vals.push_back(-1.0); }
                if (j > 0) { rows.push_back(node); cols.push_back(idx(i, j-1, k)); vals.push_back(-1.0); }
                if (j < ny-1) { rows.push_back(node); cols.push_back(idx(i, j+1, k)); vals.push_back(-1.0); }
                if (k > 0) { rows.push_back(node); cols.push_back(idx(i, j, k-1)); vals.push_back(-1.0); }
                if (k < nz-1) { rows.push_back(node); cols.push_back(idx(i, j, k+1)); vals.push_back(-1.0); }
            }
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    SparseMatrixCSR A(n, n);
    A.build_from_coo(rows, cols, vals);
    auto end = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "  Grid: " << nx << " x " << ny << " x " << nz << " = " << n << " DOFs\n";
    std::cout << "  NNZ: " << A.nnz() << "\n";
    std::cout << "  Build time: " << build_time << " ms\n";

    // Solve
    std::vector<Real> b(n, 1.0);
    std::vector<Real> x(n, 0.0);

    ConjugateGradientSolver::Config config;
    config.tolerance = 1.0e-6;
    config.max_iterations = 500;

    ConjugateGradientSolver cg(config);
    JacobiPreconditioner precond;
    precond.setup(A);

    start = std::chrono::high_resolution_clock::now();
    SolverResult result = cg.solve(A, b, x, &precond);
    end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double, std::milli>(end - start).count();

    check(result.converged, "3D Laplacian solved");
    std::cout << "  Iterations: " << result.iterations << "\n";
    std::cout << "  Solve time: " << solve_time << " ms\n";
    std::cout << "  Time per iteration: " << solve_time / result.iterations << " ms\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Sparse Matrix Solver Tests\n";
    std::cout << "========================================\n";

    Kokkos::initialize();
    {
        test_csr_construction();
        test_spmv();
        test_jacobi_precond();
        test_cg_simple();
        test_cg_laplacian();
        test_stiffness_assembly();
        test_boundary_conditions();
        test_fem_system();
        test_performance();
    }
    Kokkos::finalize();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << tests_total << " tests passed\n";
    std::cout << "========================================\n";

    return (tests_passed == tests_total) ? 0 : 1;
}
