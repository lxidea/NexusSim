#pragma once

/**
 * @file gpu_sparse_matrix.hpp
 * @brief GPU-accelerated sparse matrix operations using Kokkos
 *
 * This module provides:
 * - KokkosSparseMatrix: CSR matrix stored on GPU
 * - GPU SpMV (Sparse Matrix-Vector multiplication)
 * - GPU-compatible PCG solver
 * - GPU vector operations (dot, axpy, norm)
 *
 * Portable across CUDA, HIP, SYCL via Kokkos abstraction.
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <nexussim/solver/sparse_matrix.hpp>
#include <vector>
#include <cmath>
#include <iostream>

namespace nxs {
namespace solver {

#ifdef NEXUSSIM_HAVE_KOKKOS

// ============================================================================
// GPU Sparse Matrix (CSR Format)
// ============================================================================

/**
 * @brief GPU-resident sparse matrix in CSR format
 *
 * Stores matrix data in Kokkos Views for GPU execution.
 * Provides high-performance SpMV using team-based parallelism.
 */
class KokkosSparseMatrix {
public:
    // View types for GPU storage
    using ValuesView = View1D<Real>;
    using IndicesView = View1D<Index>;
    using HostValuesView = HostView1D<Real>;
    using HostIndicesView = HostView1D<Index>;

    KokkosSparseMatrix() : num_rows_(0), num_cols_(0), nnz_(0) {}

    /**
     * @brief Construct GPU matrix from CPU CSR matrix
     */
    explicit KokkosSparseMatrix(const SparseMatrixCSR& cpu_matrix)
        : num_rows_(cpu_matrix.num_rows())
        , num_cols_(cpu_matrix.num_cols())
        , nnz_(cpu_matrix.nnz())
    {
        upload_from_cpu(cpu_matrix);
    }

    /**
     * @brief Upload CPU matrix data to GPU
     */
    void upload_from_cpu(const SparseMatrixCSR& cpu_matrix) {
        num_rows_ = cpu_matrix.num_rows();
        num_cols_ = cpu_matrix.num_cols();
        nnz_ = cpu_matrix.nnz();

        // Allocate GPU views
        values_ = ValuesView("SparseMatrix::values", nnz_);
        col_indices_ = IndicesView("SparseMatrix::col_indices", nnz_);
        row_ptr_ = IndicesView("SparseMatrix::row_ptr", num_rows_ + 1);

        // Create host mirrors
        auto h_values = Kokkos::create_mirror_view(values_);
        auto h_col_indices = Kokkos::create_mirror_view(col_indices_);
        auto h_row_ptr = Kokkos::create_mirror_view(row_ptr_);

        // Copy data
        const auto& cpu_vals = cpu_matrix.values();
        const auto& cpu_cols = cpu_matrix.col_indices();
        const auto& cpu_rows = cpu_matrix.row_ptr();

        for (Index i = 0; i < nnz_; ++i) {
            h_values(i) = cpu_vals[i];
            h_col_indices(i) = cpu_cols[i];
        }
        for (Index i = 0; i <= num_rows_; ++i) {
            h_row_ptr(i) = cpu_rows[i];
        }

        // Copy to GPU
        Kokkos::deep_copy(values_, h_values);
        Kokkos::deep_copy(col_indices_, h_col_indices);
        Kokkos::deep_copy(row_ptr_, h_row_ptr);
    }

    /**
     * @brief Download GPU matrix data to CPU
     */
    SparseMatrixCSR download_to_cpu() const {
        SparseMatrixCSR cpu_matrix(num_rows_, num_cols_);

        // Create host mirrors and copy
        auto h_values = Kokkos::create_mirror_view(values_);
        auto h_col_indices = Kokkos::create_mirror_view(col_indices_);
        auto h_row_ptr = Kokkos::create_mirror_view(row_ptr_);

        Kokkos::deep_copy(h_values, values_);
        Kokkos::deep_copy(h_col_indices, col_indices_);
        Kokkos::deep_copy(h_row_ptr, row_ptr_);

        // Copy to CPU vectors
        std::vector<Index> row_ptr_vec(num_rows_ + 1);
        std::vector<Index> col_indices_vec(nnz_);
        for (Index i = 0; i <= num_rows_; ++i) {
            row_ptr_vec[i] = h_row_ptr(i);
        }
        for (Index i = 0; i < nnz_; ++i) {
            col_indices_vec[i] = h_col_indices(i);
        }

        cpu_matrix.set_structure(row_ptr_vec, col_indices_vec);

        // Copy values
        for (Index i = 0; i < nnz_; ++i) {
            cpu_matrix.values()[i] = h_values(i);
        }

        return cpu_matrix;
    }

    // ========================================================================
    // Properties
    // ========================================================================

    Index num_rows() const { return num_rows_; }
    Index num_cols() const { return num_cols_; }
    Index nnz() const { return nnz_; }

    const ValuesView& values() const { return values_; }
    const IndicesView& col_indices() const { return col_indices_; }
    const IndicesView& row_ptr() const { return row_ptr_; }

    ValuesView& values() { return values_; }

    // ========================================================================
    // SpMV: y = A * x
    // ========================================================================

    /**
     * @brief GPU Sparse Matrix-Vector multiplication
     *
     * Uses flat parallelism (one thread per row).
     * For large matrices, team-based version may be faster.
     */
    void multiply(const View1D<Real>& x, View1D<Real>& y) const {
        auto values = values_;
        auto col_indices = col_indices_;
        auto row_ptr = row_ptr_;
        Index n = num_rows_;

        Kokkos::parallel_for("SpMV", n, KOKKOS_LAMBDA(const Index i) {
            Real sum = 0.0;
            for (Index k = row_ptr(i); k < row_ptr(i + 1); ++k) {
                sum += values(k) * x(col_indices(k));
            }
            y(i) = sum;
        });
    }

    /**
     * @brief GPU SpMV with team parallelism (better for long rows)
     *
     * Uses hierarchical parallelism:
     * - Teams: one per row
     * - Threads within team: parallel reduction over row entries
     */
    void multiply_team(const View1D<Real>& x, View1D<Real>& y) const {
        auto values = values_;
        auto col_indices = col_indices_;
        auto row_ptr = row_ptr_;
        Index n = num_rows_;

        // Team policy: one team per row
        using TeamPolicy = Kokkos::TeamPolicy<>;
        using MemberType = typename TeamPolicy::member_type;

        // Estimate team size based on average row length
        int avg_row_len = (nnz_ > 0) ? static_cast<int>((nnz_ + n - 1) / n) : 1;
        int team_size = std::min(std::max(avg_row_len, 1), 256);

        Kokkos::parallel_for("SpMV_team", TeamPolicy(n, team_size),
            KOKKOS_LAMBDA(const MemberType& team) {
                const Index row = team.league_rank();
                const Index row_start = row_ptr(row);
                const Index row_end = row_ptr(row + 1);

                Real sum = 0.0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, row_start, row_end),
                    [&](const Index k, Real& local_sum) {
                        local_sum += values(k) * x(col_indices(k));
                    },
                    sum
                );

                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    y(row) = sum;
                });
            }
        );
    }

    /**
     * @brief Extract diagonal elements to GPU view
     */
    void get_diagonal(View1D<Real>& diag) const {
        auto values = values_;
        auto col_indices = col_indices_;
        auto row_ptr = row_ptr_;
        Index n = num_rows_;

        Kokkos::parallel_for("GetDiagonal", n, KOKKOS_LAMBDA(const Index i) {
            diag(i) = 0.0;
            for (Index k = row_ptr(i); k < row_ptr(i + 1); ++k) {
                if (col_indices(k) == i) {
                    diag(i) = values(k);
                    break;
                }
            }
        });
    }

private:
    Index num_rows_;
    Index num_cols_;
    Index nnz_;
    ValuesView values_;
    IndicesView col_indices_;
    IndicesView row_ptr_;
};

// ============================================================================
// GPU Vector Operations
// ============================================================================

namespace gpu {

/**
 * @brief GPU dot product: result = sum(x[i] * y[i])
 */
inline Real dot(const View1D<Real>& x, const View1D<Real>& y) {
    Real result = 0.0;
    Index n = x.extent(0);

    Kokkos::parallel_reduce("dot", n,
        KOKKOS_LAMBDA(const Index i, Real& sum) {
            sum += x(i) * y(i);
        },
        result
    );

    return result;
}

/**
 * @brief GPU vector norm: result = sqrt(sum(x[i]^2))
 */
inline Real norm(const View1D<Real>& x) {
    return std::sqrt(dot(x, x));
}

/**
 * @brief GPU AXPY: y = alpha * x + y
 */
inline void axpy(Real alpha, const View1D<Real>& x, View1D<Real>& y) {
    Index n = x.extent(0);

    Kokkos::parallel_for("axpy", n, KOKKOS_LAMBDA(const Index i) {
        y(i) = alpha * x(i) + y(i);
    });
}

/**
 * @brief GPU vector scale: y = alpha * x
 */
inline void scale(Real alpha, const View1D<Real>& x, View1D<Real>& y) {
    Index n = x.extent(0);

    Kokkos::parallel_for("scale", n, KOKKOS_LAMBDA(const Index i) {
        y(i) = alpha * x(i);
    });
}

/**
 * @brief GPU vector copy: y = x
 */
inline void copy(const View1D<Real>& x, View1D<Real>& y) {
    Kokkos::deep_copy(y, x);
}

/**
 * @brief GPU vector fill: x = val
 */
inline void fill(View1D<Real>& x, Real val) {
    Kokkos::deep_copy(x, val);
}

/**
 * @brief GPU vector subtraction: z = x - y
 */
inline void subtract(const View1D<Real>& x, const View1D<Real>& y, View1D<Real>& z) {
    Index n = x.extent(0);

    Kokkos::parallel_for("subtract", n, KOKKOS_LAMBDA(const Index i) {
        z(i) = x(i) - y(i);
    });
}

/**
 * @brief GPU element-wise multiplication: z = x * y (for Jacobi preconditioner)
 */
inline void element_multiply(const View1D<Real>& x, const View1D<Real>& y, View1D<Real>& z) {
    Index n = x.extent(0);

    Kokkos::parallel_for("element_multiply", n, KOKKOS_LAMBDA(const Index i) {
        z(i) = x(i) * y(i);
    });
}

} // namespace gpu

// ============================================================================
// GPU Preconditioners
// ============================================================================

/**
 * @brief GPU Jacobi preconditioner
 */
class GPUJacobiPreconditioner {
public:
    void setup(const KokkosSparseMatrix& A) {
        Index n = A.num_rows();
        diag_inv_ = View1D<Real>("JacobiPrecond::diag_inv", n);

        auto diag_inv = diag_inv_;
        A.get_diagonal(diag_inv);

        // Invert diagonal
        Kokkos::parallel_for("JacobiSetup", n, KOKKOS_LAMBDA(const Index i) {
            Real d = diag_inv(i);
            diag_inv(i) = (Kokkos::abs(d) > 1.0e-30) ? 1.0 / d : 1.0;
        });
    }

    void apply(const View1D<Real>& r, View1D<Real>& z) const {
        gpu::element_multiply(diag_inv_, r, z);
    }

private:
    View1D<Real> diag_inv_;
};

// ============================================================================
// GPU Conjugate Gradient Solver
// ============================================================================

/**
 * @brief GPU-accelerated Preconditioned Conjugate Gradient solver
 *
 * All computations performed on GPU. Only residual checks transfer small
 * scalars back to host.
 */
class GPUConjugateGradientSolver {
public:
    struct Config {
        int max_iterations;
        Real tolerance;
        bool verbose;
        bool use_team_spmv;  // Use team-based SpMV for long rows

        Config()
            : max_iterations(1000)
            , tolerance(1.0e-10)
            , verbose(false)
            , use_team_spmv(false)
        {}
    };

    GPUConjugateGradientSolver(const Config& config = Config())
        : config_(config)
    {}

    /**
     * @brief Solve A * x = b on GPU
     */
    SolverResult solve(const KokkosSparseMatrix& A,
                       const View1D<Real>& b,
                       View1D<Real>& x,
                       const GPUJacobiPreconditioner* precond = nullptr) {
        const Index n = A.num_rows();
        SolverResult result;
        result.converged = false;
        result.iterations = 0;

        // Default preconditioner if not provided
        GPUJacobiPreconditioner default_precond;
        if (!precond) {
            default_precond.setup(A);
            precond = &default_precond;
        }

        // Allocate GPU work vectors
        View1D<Real> r("CG::r", n);
        View1D<Real> z("CG::z", n);
        View1D<Real> p("CG::p", n);
        View1D<Real> Ap("CG::Ap", n);

        // r = b - A*x
        if (config_.use_team_spmv) {
            A.multiply_team(x, Ap);
        } else {
            A.multiply(x, Ap);
        }
        gpu::subtract(b, Ap, r);

        // Initial residual
        result.initial_residual = gpu::norm(r);
        if (result.initial_residual < config_.tolerance) {
            result.converged = true;
            result.residual_norm = result.initial_residual;
            result.message = "Already converged";
            return result;
        }

        // z = M^{-1} * r
        precond->apply(r, z);

        // p = z
        gpu::copy(z, p);

        // r·z
        Real rz = gpu::dot(r, z);

        for (int iter = 0; iter < config_.max_iterations; ++iter) {
            // Ap = A * p
            if (config_.use_team_spmv) {
                A.multiply_team(p, Ap);
            } else {
                A.multiply(p, Ap);
            }

            // α = (r·z) / (p·Ap)
            Real pAp = gpu::dot(p, Ap);
            if (std::abs(pAp) < 1.0e-30) {
                result.message = "Breakdown: p·Ap = 0";
                result.iterations = iter;
                result.residual_norm = gpu::norm(r);
                return result;
            }
            Real alpha = rz / pAp;

            // x = x + α*p
            gpu::axpy(alpha, p, x);

            // r_new = r - α*Ap
            gpu::axpy(-alpha, Ap, r);

            // Check convergence
            Real res_norm = gpu::norm(r);
            Real rel_res = res_norm / result.initial_residual;

            if (config_.verbose && iter % 10 == 0) {
                std::cout << "  GPU CG iter " << iter << ": |r| = " << res_norm
                          << ", rel = " << rel_res << "\n";
            }

            if (res_norm < config_.tolerance || rel_res < config_.tolerance) {
                result.converged = true;
                result.iterations = iter + 1;
                result.residual_norm = res_norm;
                result.message = "Converged";
                return result;
            }

            // z_new = M^{-1} * r_new
            precond->apply(r, z);

            // β = (r_new·z_new) / (r·z)
            Real rz_new = gpu::dot(r, z);
            Real beta = rz_new / rz;
            rz = rz_new;

            // p = z + β*p
            // Implemented as: p = β*p + z (requires different axpy)
            Index num = n;
            auto p_view = p;
            auto z_view = z;
            Kokkos::parallel_for("CG_update_p", num, KOKKOS_LAMBDA(const Index i) {
                p_view(i) = z_view(i) + beta * p_view(i);
            });
        }

        result.iterations = config_.max_iterations;
        result.residual_norm = gpu::norm(r);
        result.message = "Max iterations reached";
        return result;
    }

    /**
     * @brief Convenience solve from CPU data
     */
    SolverResult solve_from_cpu(const SparseMatrixCSR& A_cpu,
                                 const std::vector<Real>& b_cpu,
                                 std::vector<Real>& x_cpu) {
        // Upload to GPU
        KokkosSparseMatrix A_gpu(A_cpu);

        Index n = A_cpu.num_rows();
        View1D<Real> b_gpu("b", n);
        View1D<Real> x_gpu("x", n);

        // Copy b to GPU
        auto h_b = Kokkos::create_mirror_view(b_gpu);
        for (Index i = 0; i < n; ++i) {
            h_b(i) = b_cpu[i];
        }
        Kokkos::deep_copy(b_gpu, h_b);

        // Copy initial x to GPU
        auto h_x = Kokkos::create_mirror_view(x_gpu);
        if (x_cpu.size() == static_cast<std::size_t>(n)) {
            for (Index i = 0; i < n; ++i) {
                h_x(i) = x_cpu[i];
            }
        } else {
            x_cpu.resize(n);
            for (Index i = 0; i < n; ++i) {
                h_x(i) = 0.0;
            }
        }
        Kokkos::deep_copy(x_gpu, h_x);

        // Solve on GPU
        auto result = solve(A_gpu, b_gpu, x_gpu);

        // Download solution
        Kokkos::deep_copy(h_x, x_gpu);
        for (Index i = 0; i < n; ++i) {
            x_cpu[i] = h_x(i);
        }

        return result;
    }

private:
    Config config_;
};

// ============================================================================
// Performance Benchmarking
// ============================================================================

/**
 * @brief Benchmark SpMV performance
 */
struct SpMVBenchmarkResult {
    double time_flat_ms;      // Flat parallelism time
    double time_team_ms;      // Team parallelism time
    double gflops_flat;       // GFLOP/s for flat
    double gflops_team;       // GFLOP/s for team
    double bandwidth_flat_gb; // Memory bandwidth for flat
    double bandwidth_team_gb; // Memory bandwidth for team
};

inline SpMVBenchmarkResult benchmark_spmv(const KokkosSparseMatrix& A, int num_runs = 100) {
    SpMVBenchmarkResult result;
    Index n = A.num_rows();
    Index nnz = A.nnz();

    View1D<Real> x("bench_x", n);
    View1D<Real> y("bench_y", n);

    // Initialize x to 1
    Kokkos::deep_copy(x, 1.0);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        A.multiply(x, y);
    }
    Kokkos::fence();

    // Benchmark flat SpMV
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        A.multiply(x, y);
    }
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();

    double flat_time = std::chrono::duration<double, std::milli>(end - start).count();
    result.time_flat_ms = flat_time / num_runs;

    // Warmup team
    for (int i = 0; i < 5; ++i) {
        A.multiply_team(x, y);
    }
    Kokkos::fence();

    // Benchmark team SpMV
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        A.multiply_team(x, y);
    }
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();

    double team_time = std::chrono::duration<double, std::milli>(end - start).count();
    result.time_team_ms = team_time / num_runs;

    // Calculate GFLOP/s (2 ops per NNZ: multiply + add)
    double gflops = 2.0 * nnz / 1.0e9;
    result.gflops_flat = gflops / (result.time_flat_ms / 1000.0);
    result.gflops_team = gflops / (result.time_team_ms / 1000.0);

    // Calculate memory bandwidth
    // Reads: values (nnz), col_indices (nnz), row_ptr (n+1), x (n)
    // Writes: y (n)
    double bytes_read = nnz * sizeof(Real) + nnz * sizeof(Index) +
                        (n + 1) * sizeof(Index) + n * sizeof(Real);
    double bytes_written = n * sizeof(Real);
    double total_bytes = bytes_read + bytes_written;

    result.bandwidth_flat_gb = (total_bytes / 1.0e9) / (result.time_flat_ms / 1000.0);
    result.bandwidth_team_gb = (total_bytes / 1.0e9) / (result.time_team_ms / 1000.0);

    return result;
}

#else // !NEXUSSIM_HAVE_KOKKOS

// Stub implementations when Kokkos is not available
class KokkosSparseMatrix {
public:
    KokkosSparseMatrix() {}
    explicit KokkosSparseMatrix(const SparseMatrixCSR&) {}
    Index num_rows() const { return 0; }
    Index num_cols() const { return 0; }
    Index nnz() const { return 0; }
};

class GPUJacobiPreconditioner {
public:
    void setup(const KokkosSparseMatrix&) {}
};

class GPUConjugateGradientSolver {
public:
    struct Config {
        int max_iterations;
        Real tolerance;
        bool verbose;
        bool use_team_spmv;

        Config()
            : max_iterations(1000)
            , tolerance(1.0e-10)
            , verbose(false)
            , use_team_spmv(false)
        {}
    };

    GPUConjugateGradientSolver(const Config& = Config()) {}

    SolverResult solve_from_cpu(const SparseMatrixCSR& A_cpu,
                                 const std::vector<Real>& b_cpu,
                                 std::vector<Real>& x_cpu) {
        // Fall back to CPU solver
        ConjugateGradientSolver::Config cpu_config;
        cpu_config.max_iterations = 1000;
        cpu_config.tolerance = 1.0e-10;
        ConjugateGradientSolver cpu_solver(cpu_config);
        return cpu_solver.solve(A_cpu, b_cpu, x_cpu);
    }
};

struct SpMVBenchmarkResult {
    double time_flat_ms = 0;
    double time_team_ms = 0;
    double gflops_flat = 0;
    double gflops_team = 0;
    double bandwidth_flat_gb = 0;
    double bandwidth_team_gb = 0;
};

#endif // NEXUSSIM_HAVE_KOKKOS

} // namespace solver
} // namespace nxs
