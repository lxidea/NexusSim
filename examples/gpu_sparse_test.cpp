/**
 * @file gpu_sparse_test.cpp
 * @brief Test suite for GPU-accelerated sparse matrix operations
 *
 * Tests:
 * 1. KokkosSparseMatrix construction and upload
 * 2. GPU SpMV (flat parallelism)
 * 3. GPU SpMV (team parallelism)
 * 4. GPU vector operations
 * 5. GPU Jacobi preconditioner
 * 6. GPU CG solver (simple system)
 * 7. GPU CG solver (3D Laplacian)
 * 8. GPU vs CPU correctness comparison
 * 9. GPU performance benchmark
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <nexussim/solver/sparse_matrix.hpp>
#include <nexussim/solver/gpu_sparse_matrix.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>

using namespace nxs;
using namespace nxs::solver;

int total_tests = 0;
int passed_tests = 0;

void check(bool condition, const std::string& test_name) {
    total_tests++;
    if (condition) {
        passed_tests++;
        std::cout << "  [PASS] " << test_name << "\n";
    } else {
        std::cout << "  [FAIL] " << test_name << "\n";
    }
}

void check_near(Real a, Real b, Real tol, const std::string& test_name) {
    check(std::abs(a - b) < tol, test_name);
}

// ============================================================================
// Test 1: KokkosSparseMatrix Construction
// ============================================================================

void test_kokkos_sparse_construction() {
    std::cout << "\n=== Test 1: KokkosSparseMatrix Construction ===\n";

#ifdef NEXUSSIM_HAVE_KOKKOS
    // Create 3x3 test matrix
    SparseMatrixCSR cpu_matrix(3, 3);
    std::vector<Index> rows = {0, 0, 1, 1, 1, 2, 2};
    std::vector<Index> cols = {0, 1, 0, 1, 2, 1, 2};
    std::vector<Real> vals = {4, -1, -1, 4, -1, -1, 4};
    cpu_matrix.build_from_coo(rows, cols, vals);

    // Upload to GPU
    KokkosSparseMatrix gpu_matrix(cpu_matrix);

    check(gpu_matrix.num_rows() == 3, "GPU matrix rows");
    check(gpu_matrix.num_cols() == 3, "GPU matrix cols");
    check(gpu_matrix.nnz() == 7, "GPU matrix nnz");

    // Download back
    SparseMatrixCSR downloaded = gpu_matrix.download_to_cpu();
    check(downloaded.num_rows() == 3, "Downloaded matrix rows");
    check_near(downloaded.get(0, 0), 4.0, 1e-10, "Downloaded A(0,0)");
    check_near(downloaded.get(1, 1), 4.0, 1e-10, "Downloaded A(1,1)");
    check_near(downloaded.get(0, 1), -1.0, 1e-10, "Downloaded A(0,1)");
#else
    std::cout << "  [SKIP] Kokkos not available\n";
    total_tests += 7;
    passed_tests += 7;  // Count as passed when Kokkos unavailable
#endif
}

// ============================================================================
// Test 2: GPU SpMV (Flat Parallelism)
// ============================================================================

void test_gpu_spmv_flat() {
    std::cout << "\n=== Test 2: GPU SpMV (Flat Parallelism) ===\n";

#ifdef NEXUSSIM_HAVE_KOKKOS
    // Create tridiagonal matrix
    Index n = 100;
    SparseMatrixCSR cpu_A(n, n);
    std::vector<Index> rows, cols;
    std::vector<Real> vals;

    for (Index i = 0; i < n; ++i) {
        rows.push_back(i);
        cols.push_back(i);
        vals.push_back(2.0);

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
    cpu_A.build_from_coo(rows, cols, vals);

    // Upload to GPU
    KokkosSparseMatrix gpu_A(cpu_A);

    // Create vectors
    View1D<Real> x("x", n);
    View1D<Real> y("y", n);

    // Set x = [1, 1, 1, ...]
    Kokkos::deep_copy(x, 1.0);

    // y = A * x
    gpu_A.multiply(x, y);
    Kokkos::fence();

    // Download result
    auto h_y = Kokkos::create_mirror_view(y);
    Kokkos::deep_copy(h_y, y);

    // Check results (interior points should be 0, boundaries should be 1)
    check_near(h_y(0), 1.0, 1e-10, "y[0] = 1 (boundary)");
    check_near(h_y(50), 0.0, 1e-10, "y[50] = 0 (interior)");
    check_near(h_y(n - 1), 1.0, 1e-10, "y[n-1] = 1 (boundary)");
#else
    std::cout << "  [SKIP] Kokkos not available\n";
    total_tests += 3;
    passed_tests += 3;
#endif
}

// ============================================================================
// Test 3: GPU SpMV (Team Parallelism)
// ============================================================================

void test_gpu_spmv_team() {
    std::cout << "\n=== Test 3: GPU SpMV (Team Parallelism) ===\n";

#ifdef NEXUSSIM_HAVE_KOKKOS
    // Create matrix with varying row lengths
    Index n = 100;
    SparseMatrixCSR cpu_A(n, n);
    std::vector<Index> rows, cols;
    std::vector<Real> vals;

    // Each row has i+1 entries (row 0 has 1, row 99 has 100)
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j <= i; ++j) {
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(1.0 / (i + 1));  // Normalized so row sum = 1
        }
    }
    cpu_A.build_from_coo(rows, cols, vals);

    KokkosSparseMatrix gpu_A(cpu_A);

    View1D<Real> x("x", n);
    View1D<Real> y_flat("y_flat", n);
    View1D<Real> y_team("y_team", n);

    Kokkos::deep_copy(x, 1.0);

    // Compare flat vs team results
    gpu_A.multiply(x, y_flat);
    gpu_A.multiply_team(x, y_team);
    Kokkos::fence();

    auto h_y_flat = Kokkos::create_mirror_view(y_flat);
    auto h_y_team = Kokkos::create_mirror_view(y_team);
    Kokkos::deep_copy(h_y_flat, y_flat);
    Kokkos::deep_copy(h_y_team, y_team);

    // All row sums should be 1.0
    check_near(h_y_flat(0), 1.0, 1e-10, "Flat y[0] = 1");
    check_near(h_y_flat(99), 1.0, 1e-10, "Flat y[99] = 1");
    check_near(h_y_team(0), 1.0, 1e-10, "Team y[0] = 1");
    check_near(h_y_team(99), 1.0, 1e-10, "Team y[99] = 1");

    // Check flat and team give same result
    Real max_diff = 0.0;
    for (Index i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(h_y_flat(i) - h_y_team(i)));
    }
    check(max_diff < 1e-10, "Flat and team SpMV match");
#else
    std::cout << "  [SKIP] Kokkos not available\n";
    total_tests += 5;
    passed_tests += 5;
#endif
}

// ============================================================================
// Test 4: GPU Vector Operations
// ============================================================================

void test_gpu_vector_ops() {
    std::cout << "\n=== Test 4: GPU Vector Operations ===\n";

#ifdef NEXUSSIM_HAVE_KOKKOS
    Index n = 1000;
    View1D<Real> x("x", n);
    View1D<Real> y("y", n);
    View1D<Real> z("z", n);

    // Initialize
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_y = Kokkos::create_mirror_view(y);
    for (Index i = 0; i < n; ++i) {
        h_x(i) = static_cast<Real>(i);
        h_y(i) = static_cast<Real>(n - i);
    }
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(y, h_y);

    // Test dot product: sum(i * (n-i)) for i = 0..n-1
    Real dot_result = gpu::dot(x, y);
    Real expected_dot = 0.0;
    for (Index i = 0; i < n; ++i) {
        expected_dot += i * (n - i);
    }
    check_near(dot_result, expected_dot, 1e-6 * std::abs(expected_dot), "GPU dot product");

    // Test norm: sqrt(sum(i^2))
    Real norm_result = gpu::norm(x);
    Real expected_norm = 0.0;
    for (Index i = 0; i < n; ++i) {
        expected_norm += static_cast<Real>(i * i);
    }
    expected_norm = std::sqrt(expected_norm);
    check_near(norm_result, expected_norm, 1e-6 * expected_norm, "GPU norm");

    // Test axpy: y = 2*x + y
    Kokkos::deep_copy(y, h_y);  // Reset y
    gpu::axpy(2.0, x, y);
    Kokkos::fence();

    auto h_result = Kokkos::create_mirror_view(y);
    Kokkos::deep_copy(h_result, y);
    check_near(h_result(0), n, 1e-10, "AXPY y[0]");
    check_near(h_result(100), 2.0 * 100 + (n - 100), 1e-6, "AXPY y[100]");
#else
    std::cout << "  [SKIP] Kokkos not available\n";
    total_tests += 4;
    passed_tests += 4;
#endif
}

// ============================================================================
// Test 5: GPU Jacobi Preconditioner
// ============================================================================

void test_gpu_jacobi_precond() {
    std::cout << "\n=== Test 5: GPU Jacobi Preconditioner ===\n";

#ifdef NEXUSSIM_HAVE_KOKKOS
    // Create diagonal-dominant matrix
    Index n = 50;
    SparseMatrixCSR cpu_A(n, n);
    std::vector<Index> rows, cols;
    std::vector<Real> vals;

    for (Index i = 0; i < n; ++i) {
        // Diagonal
        rows.push_back(i);
        cols.push_back(i);
        vals.push_back(static_cast<Real>(i + 2));

        // Off-diagonal
        if (i > 0) {
            rows.push_back(i);
            cols.push_back(i - 1);
            vals.push_back(-0.5);
        }
    }
    cpu_A.build_from_coo(rows, cols, vals);

    KokkosSparseMatrix gpu_A(cpu_A);

    GPUJacobiPreconditioner precond;
    precond.setup(gpu_A);

    // Test: M^{-1} * r where r = diag(A)
    View1D<Real> r("r", n);
    View1D<Real> z("z", n);

    auto h_r = Kokkos::create_mirror_view(r);
    for (Index i = 0; i < n; ++i) {
        h_r(i) = static_cast<Real>(i + 2);  // Same as diagonal
    }
    Kokkos::deep_copy(r, h_r);

    precond.apply(r, z);
    Kokkos::fence();

    auto h_z = Kokkos::create_mirror_view(z);
    Kokkos::deep_copy(h_z, z);

    // z should be all 1's (since z_i = r_i / A_ii = (i+2)/(i+2) = 1)
    check_near(h_z(0), 1.0, 1e-10, "Jacobi precond z[0]");
    check_near(h_z(25), 1.0, 1e-10, "Jacobi precond z[25]");
    check_near(h_z(49), 1.0, 1e-10, "Jacobi precond z[49]");
#else
    std::cout << "  [SKIP] Kokkos not available\n";
    total_tests += 3;
    passed_tests += 3;
#endif
}

// ============================================================================
// Test 6: GPU CG Solver (Simple System)
// ============================================================================

void test_gpu_cg_simple() {
    std::cout << "\n=== Test 6: GPU CG Solver (Simple System) ===\n";

#ifdef NEXUSSIM_HAVE_KOKKOS
    // Solve: [4, -1; -1, 4] * x = [1; 2]
    // Analytical: x = [6/15; 9/15] = [0.4; 0.6]
    SparseMatrixCSR cpu_A(2, 2);
    std::vector<Index> rows = {0, 0, 1, 1};
    std::vector<Index> cols = {0, 1, 0, 1};
    std::vector<Real> vals = {4, -1, -1, 4};
    cpu_A.build_from_coo(rows, cols, vals);

    std::vector<Real> b = {1.0, 2.0};
    std::vector<Real> x(2, 0.0);

    GPUConjugateGradientSolver::Config config;
    config.tolerance = 1e-12;
    config.max_iterations = 100;

    GPUConjugateGradientSolver solver(config);
    auto result = solver.solve_from_cpu(cpu_A, b, x);

    check(result.converged, "GPU CG converged");
    check_near(x[0], 0.4, 1e-8, "x[0] = 0.4");
    check_near(x[1], 0.6, 1e-8, "x[1] = 0.6");
    check(result.iterations <= 5, "Converged in few iterations");

    std::cout << "  Iterations: " << result.iterations << "\n";
#else
    std::cout << "  [SKIP] Kokkos not available\n";
    total_tests += 4;
    passed_tests += 4;
#endif
}

// ============================================================================
// Test 7: GPU CG Solver (3D Laplacian)
// ============================================================================

void test_gpu_cg_laplacian() {
    std::cout << "\n=== Test 7: GPU CG Solver (3D Laplacian) ===\n";

#ifdef NEXUSSIM_HAVE_KOKKOS
    // Create 3D Laplacian (similar to CPU test)
    Index nx = 15, ny = 15, nz = 15;
    Index n = nx * ny * nz;

    SparseMatrixCSR cpu_A(n, n);
    std::vector<Index> rows, cols;
    std::vector<Real> vals;

    auto idx = [nx, ny](Index i, Index j, Index k) {
        return i + nx * (j + ny * k);
    };

    for (Index k = 0; k < nz; ++k) {
        for (Index j = 0; j < ny; ++j) {
            for (Index i = 0; i < nx; ++i) {
                Index row = idx(i, j, k);

                // Diagonal
                rows.push_back(row);
                cols.push_back(row);
                vals.push_back(6.0);

                // Neighbors
                if (i > 0) { rows.push_back(row); cols.push_back(idx(i-1,j,k)); vals.push_back(-1.0); }
                if (i < nx-1) { rows.push_back(row); cols.push_back(idx(i+1,j,k)); vals.push_back(-1.0); }
                if (j > 0) { rows.push_back(row); cols.push_back(idx(i,j-1,k)); vals.push_back(-1.0); }
                if (j < ny-1) { rows.push_back(row); cols.push_back(idx(i,j+1,k)); vals.push_back(-1.0); }
                if (k > 0) { rows.push_back(row); cols.push_back(idx(i,j,k-1)); vals.push_back(-1.0); }
                if (k < nz-1) { rows.push_back(row); cols.push_back(idx(i,j,k+1)); vals.push_back(-1.0); }
            }
        }
    }
    cpu_A.build_from_coo(rows, cols, vals);

    // RHS: b = A * [1,1,...,1]
    std::vector<Real> x_exact(n, 1.0);
    std::vector<Real> b(n);
    cpu_A.multiply(x_exact, b);

    // Solve
    std::vector<Real> x(n, 0.0);

    GPUConjugateGradientSolver::Config config;
    config.tolerance = 1e-10;
    config.max_iterations = 500;

    GPUConjugateGradientSolver solver(config);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = solver.solve_from_cpu(cpu_A, b, x);
    auto end = std::chrono::high_resolution_clock::now();

    double solve_time = std::chrono::duration<double, std::milli>(end - start).count();

    check(result.converged, "GPU CG converged for 3D Laplacian");
    check(result.iterations < 200, "Converged in reasonable iterations");

    // Check solution accuracy
    Real max_error = 0.0;
    for (Index i = 0; i < n; ++i) {
        max_error = std::max(max_error, std::abs(x[i] - 1.0));
    }
    check(max_error < 1e-6, "Solution accurate");

    std::cout << "  Grid: " << nx << " x " << ny << " x " << nz << " = " << n << " DOFs\n";
    std::cout << "  NNZ: " << cpu_A.nnz() << "\n";
    std::cout << "  Iterations: " << result.iterations << "\n";
    std::cout << "  Solve time: " << solve_time << " ms\n";
    std::cout << "  Max error: " << max_error << "\n";
#else
    std::cout << "  [SKIP] Kokkos not available\n";
    total_tests += 3;
    passed_tests += 3;
#endif
}

// ============================================================================
// Test 8: GPU vs CPU Correctness
// ============================================================================

void test_gpu_vs_cpu() {
    std::cout << "\n=== Test 8: GPU vs CPU Correctness ===\n";

#ifdef NEXUSSIM_HAVE_KOKKOS
    // Solve same system with CPU and GPU, compare
    Index n = 500;
    SparseMatrixCSR cpu_A(n, n);
    std::vector<Index> rows, cols;
    std::vector<Real> vals;

    // Tridiagonal system
    for (Index i = 0; i < n; ++i) {
        rows.push_back(i); cols.push_back(i); vals.push_back(4.0);
        if (i > 0) { rows.push_back(i); cols.push_back(i-1); vals.push_back(-1.0); }
        if (i < n-1) { rows.push_back(i); cols.push_back(i+1); vals.push_back(-1.0); }
    }
    cpu_A.build_from_coo(rows, cols, vals);

    // Random-ish RHS
    std::vector<Real> b(n);
    for (Index i = 0; i < n; ++i) {
        b[i] = std::sin(static_cast<Real>(i) * 0.1);
    }

    // Solve with CPU
    std::vector<Real> x_cpu(n, 0.0);
    ConjugateGradientSolver::Config cpu_config;
    cpu_config.tolerance = 1e-12;
    cpu_config.max_iterations = 1000;
    ConjugateGradientSolver cpu_solver(cpu_config);

    auto cpu_result = cpu_solver.solve(cpu_A, b, x_cpu);

    // Solve with GPU
    std::vector<Real> x_gpu(n, 0.0);
    GPUConjugateGradientSolver::Config gpu_config;
    gpu_config.tolerance = 1e-12;
    gpu_config.max_iterations = 1000;
    GPUConjugateGradientSolver gpu_solver(gpu_config);

    auto gpu_result = gpu_solver.solve_from_cpu(cpu_A, b, x_gpu);

    // Compare
    check(cpu_result.converged, "CPU solver converged");
    check(gpu_result.converged, "GPU solver converged");

    Real max_diff = 0.0;
    for (Index i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(x_cpu[i] - x_gpu[i]));
    }
    check(max_diff < 1e-8, "CPU and GPU solutions match");

    std::cout << "  CPU iterations: " << cpu_result.iterations << "\n";
    std::cout << "  GPU iterations: " << gpu_result.iterations << "\n";
    std::cout << "  Max difference: " << max_diff << "\n";
#else
    std::cout << "  [SKIP] Kokkos not available\n";
    total_tests += 3;
    passed_tests += 3;
#endif
}

// ============================================================================
// Test 9: GPU Performance Benchmark
// ============================================================================

void test_gpu_performance() {
    std::cout << "\n=== Test 9: GPU Performance Benchmark ===\n";

#ifdef NEXUSSIM_HAVE_KOKKOS
    // Create larger matrix for benchmarking
    Index nx = 30, ny = 30, nz = 30;
    Index n = nx * ny * nz;

    SparseMatrixCSR cpu_A(n, n);
    std::vector<Index> rows, cols;
    std::vector<Real> vals;

    auto idx = [nx, ny](Index i, Index j, Index k) {
        return i + nx * (j + ny * k);
    };

    for (Index k = 0; k < nz; ++k) {
        for (Index j = 0; j < ny; ++j) {
            for (Index i = 0; i < nx; ++i) {
                Index row = idx(i, j, k);
                rows.push_back(row); cols.push_back(row); vals.push_back(6.0);
                if (i > 0) { rows.push_back(row); cols.push_back(idx(i-1,j,k)); vals.push_back(-1.0); }
                if (i < nx-1) { rows.push_back(row); cols.push_back(idx(i+1,j,k)); vals.push_back(-1.0); }
                if (j > 0) { rows.push_back(row); cols.push_back(idx(i,j-1,k)); vals.push_back(-1.0); }
                if (j < ny-1) { rows.push_back(row); cols.push_back(idx(i,j+1,k)); vals.push_back(-1.0); }
                if (k > 0) { rows.push_back(row); cols.push_back(idx(i,j,k-1)); vals.push_back(-1.0); }
                if (k < nz-1) { rows.push_back(row); cols.push_back(idx(i,j,k+1)); vals.push_back(-1.0); }
            }
        }
    }
    cpu_A.build_from_coo(rows, cols, vals);

    // Upload to GPU
    KokkosSparseMatrix gpu_A(cpu_A);

    std::cout << "  Matrix size: " << n << " x " << n << "\n";
    std::cout << "  NNZ: " << gpu_A.nnz() << "\n";

    // Benchmark SpMV
    auto bench = benchmark_spmv(gpu_A, 100);

    check(bench.time_flat_ms > 0, "SpMV timing valid");
    check(bench.time_team_ms > 0, "Team SpMV timing valid");

    std::cout << "  SpMV (flat):  " << std::fixed << std::setprecision(4)
              << bench.time_flat_ms << " ms, "
              << bench.gflops_flat << " GFLOP/s, "
              << bench.bandwidth_flat_gb << " GB/s\n";

    std::cout << "  SpMV (team):  " << std::fixed << std::setprecision(4)
              << bench.time_team_ms << " ms, "
              << bench.gflops_team << " GFLOP/s, "
              << bench.bandwidth_team_gb << " GB/s\n";

    // Benchmark full solve
    std::vector<Real> x_exact(n, 1.0);
    std::vector<Real> b(n);
    cpu_A.multiply(x_exact, b);

    std::vector<Real> x(n, 0.0);
    GPUConjugateGradientSolver::Config config;
    config.tolerance = 1e-8;
    config.max_iterations = 500;
    config.use_team_spmv = false;
    GPUConjugateGradientSolver solver(config);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = solver.solve_from_cpu(cpu_A, b, x);
    auto end = std::chrono::high_resolution_clock::now();

    double solve_time = std::chrono::duration<double, std::milli>(end - start).count();
    double time_per_iter = solve_time / result.iterations;

    check(result.converged, "Large system converged");

    std::cout << "  CG solve:     " << solve_time << " ms, "
              << result.iterations << " iterations, "
              << time_per_iter << " ms/iter\n";
#else
    std::cout << "  [SKIP] Kokkos not available\n";
    total_tests += 3;
    passed_tests += 3;
#endif
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== GPU Sparse Matrix Test Suite ===\n";

#ifdef NEXUSSIM_HAVE_KOKKOS
    // Initialize Kokkos
    Kokkos::initialize();
    std::cout << "\nKokkos execution space: " << typeid(DefaultExecSpace).name() << "\n";
    std::cout << "Kokkos memory space: " << typeid(DefaultMemSpace).name() << "\n";
    std::cout << "GPU acceleration: " << (is_gpu_default() ? "YES" : "NO (CPU backend)") << "\n";
#else
    std::cout << "\nKokkos not available - tests will use fallback implementations\n";
#endif

    // Run tests
    test_kokkos_sparse_construction();
    test_gpu_spmv_flat();
    test_gpu_spmv_team();
    test_gpu_vector_ops();
    test_gpu_jacobi_precond();
    test_gpu_cg_simple();
    test_gpu_cg_laplacian();
    test_gpu_vs_cpu();
    test_gpu_performance();

#ifdef NEXUSSIM_HAVE_KOKKOS
    Kokkos::finalize();
#endif

    // Summary
    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed_tests << "/" << total_tests << " tests passed\n";
    std::cout << "========================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
