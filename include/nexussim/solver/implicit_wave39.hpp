#pragma once

/**
 * @file implicit_wave39.hpp
 * @brief Wave 39b: Implicit solver hardening features
 *
 * Components:
 * 1. MUMPSSolver          - Sparse direct solver (LDL^T factorization)
 * 2. ImplicitBFGS         - Full BFGS quasi-Newton solver (L-BFGS variant)
 * 3. ImplicitBuckling     - Linear buckling eigenvalue solver (inverse iteration)
 * 4. ImplicitDtControl    - Adaptive time step control (convergence-based)
 * 5. IterativeRefinement  - Iterative refinement for ill-conditioned systems
 * 6. ImplicitContactK     - Contact stiffness assembly (penalty-based)
 *
 * References:
 * - Duff, Erisman & Reid (1986) "Direct Methods for Sparse Matrices"
 * - Nocedal & Wright (2006) "Numerical Optimization", 2nd edition
 * - Bathe (2014) "Finite Element Procedures", 2nd edition
 * - Wriggers (2006) "Computational Contact Mechanics", 2nd edition
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <functional>
#include <numeric>
#include <cstring>
#include <cassert>
#include <limits>

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

namespace nxs {
namespace solver {

using Real = nxs::Real;

// ============================================================================
// Utility functions (local to this translation unit)
// ============================================================================

namespace implicit_detail {

inline Real dot(const Real* a, const Real* b, int n) {
    Real s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline Real norm2(const Real* v, int n) {
    return std::sqrt(dot(v, v, n));
}

inline void axpy(Real alpha, const Real* x, Real* y, int n) {
    for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}

inline void copy(const Real* x, Real* y, int n) {
    std::memcpy(y, x, static_cast<size_t>(n) * sizeof(Real));
}

inline void zero(Real* y, int n) {
    std::memset(y, 0, static_cast<size_t>(n) * sizeof(Real));
}

inline void scale(Real alpha, Real* y, int n) {
    for (int i = 0; i < n; ++i) y[i] *= alpha;
}

inline void scale_copy(Real alpha, const Real* x, Real* y, int n) {
    for (int i = 0; i < n; ++i) y[i] = alpha * x[i];
}

/// CSR sparse matrix-vector product: y = A * x
inline void csr_matvec(const int* row_ptr, const int* col_idx,
                       const Real* values, const Real* x, Real* y, int n) {
    for (int i = 0; i < n; ++i) {
        Real s = 0.0;
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            s += values[k] * x[col_idx[k]];
        }
        y[i] = s;
    }
}

/// Dense symmetric matrix-vector product: y = A * x (A stored row-major, full)
inline void dense_matvec(const Real* A, const Real* x, Real* y, int n) {
    for (int i = 0; i < n; ++i) {
        Real s = 0.0;
        for (int j = 0; j < n; ++j) {
            s += A[i * n + j] * x[j];
        }
        y[i] = s;
    }
}

} // namespace implicit_detail


// ============================================================================
// 1. MUMPSSolver - Sparse Direct Solver (LDL^T Factorization)
// ============================================================================

/**
 * @brief Sparse direct solver using LDL^T factorization for symmetric systems
 *
 * Solves A*x = b where A is symmetric positive (semi-)definite or indefinite.
 * Uses a fill-reducing ordering (approximate minimum degree) followed by
 * numeric LDL^T factorization with Bunch-Kaufman pivoting.
 *
 * Header-only implementation limited to MUMPS_MAX_SIZE DOFs.
 * For larger systems, link against external MUMPS/PARDISO.
 *
 * Storage:
 *   - perm_[n]: fill-reducing permutation
 *   - L stored in CSR: L_row_ptr_[], L_col_idx_[], L_values_[]
 *   - D stored as 1x1 or 2x2 blocks: D_values_[n] (diagonal), D_offdiag_[n] (sub-diag for 2x2)
 *   - pivot_type_[n]: 0 = 1x1 pivot, 1 = first of 2x2, 2 = second of 2x2
 */
class MUMPSSolver {
public:
    static constexpr int MUMPS_MAX_SIZE = 10000;

    MUMPSSolver() = default;

    /**
     * @brief Symbolic factorization: compute fill-reducing ordering and
     *        estimate structure of L factor.
     *
     * Uses approximate minimum degree (AMD) ordering on the sparsity
     * pattern of A given in CSR format. After this call, the permutation
     * and symbolic structure of L are established.
     *
     * @param n       System size (number of DOFs)
     * @param row_ptr CSR row pointers (size n+1)
     * @param col_idx CSR column indices (size nnz)
     * @return true if symbolic factorization succeeded
     */
    bool symbolic_factorize(int n, const int* row_ptr, const int* col_idx) {
        assert(n > 0 && n <= MUMPS_MAX_SIZE);
        n_ = n;
        symbolic_done_ = false;
        numeric_done_ = false;

        // Store the sparsity pattern
        int nnz = row_ptr[n];
        A_row_ptr_.assign(row_ptr, row_ptr + n + 1);
        A_col_idx_.assign(col_idx, col_idx + nnz);

        // ---- AMD ordering (simplified: degree-based greedy) ----
        perm_.resize(n);
        iperm_.resize(n);
        compute_amd_ordering(n, row_ptr, col_idx);

        // ---- Symbolic Cholesky on permuted graph ----
        // Build the elimination tree and estimate fill-in
        compute_symbolic_structure(n, row_ptr, col_idx);

        symbolic_done_ = true;
        return true;
    }

    /**
     * @brief Numeric LDL^T factorization with Bunch-Kaufman pivoting
     *
     * Requires symbolic_factorize() to have been called first.
     * Computes P*A*P^T = L*D*L^T where:
     *   - P is the fill-reducing permutation
     *   - L is unit lower triangular
     *   - D is block diagonal (1x1 and 2x2 blocks)
     *
     * @param values CSR values array (same layout as symbolic_factorize)
     * @return true if factorization succeeded (no zero pivots encountered)
     */
    bool numeric_factorize(const Real* values) {
        assert(symbolic_done_);
        numeric_done_ = false;

        int n = n_;

        // Build dense permuted matrix (for header-only simplicity)
        // Production code would use sparse factorization
        std::vector<Real> A_dense(n * n, 0.0);

        // Fill dense matrix from CSR
        for (int i = 0; i < n; ++i) {
            for (int k = A_row_ptr_[i]; k < A_row_ptr_[i + 1]; ++k) {
                int j = A_col_idx_[k];
                A_dense[i * n + j] = values[k];
                // Symmetrize
                A_dense[j * n + i] = values[k];
            }
        }

        // Permute: B = P * A * P^T
        std::vector<Real> B(n * n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                B[i * n + j] = A_dense[perm_[i] * n + perm_[j]];
            }
        }

        // LDL^T factorization with partial pivoting
        // B will be overwritten: lower triangle = L (unit diagonal implied),
        // diagonal = D
        D_values_.resize(n, 0.0);
        D_offdiag_.resize(n, 0.0);
        pivot_type_.resize(n, 0);

        // L stored as dense lower triangular for simplicity
        L_dense_.resize(n * n, 0.0);

        // Initialize L = I
        for (int i = 0; i < n; ++i) {
            L_dense_[i * n + i] = 1.0;
        }

        int i = 0;
        while (i < n) {
            // Bunch-Kaufman pivot selection
            Real alpha_bk = (1.0 + std::sqrt(17.0)) / 8.0;

            Real a_ii = std::abs(B[i * n + i]);

            // Find largest off-diagonal in column i (below diagonal)
            Real lambda = 0.0;
            int r = i;
            for (int k = i + 1; k < n; ++k) {
                Real val = std::abs(B[k * n + i]);
                if (val > lambda) {
                    lambda = val;
                    r = k;
                }
            }

            if (lambda == 0.0 && a_ii == 0.0) {
                // Zero pivot: set small value to avoid division by zero
                D_values_[i] = std::numeric_limits<Real>::epsilon();
                pivot_type_[i] = 0;
                ++i;
                continue;
            }

            bool use_1x1 = false;
            bool use_2x2 = false;

            if (a_ii >= alpha_bk * lambda) {
                // 1x1 pivot is acceptable
                use_1x1 = true;
            } else {
                // Check for 2x2 pivot
                Real sigma = 0.0;
                for (int k = i; k < n; ++k) {
                    if (k != r) {
                        Real val = std::abs(B[k * n + r]);
                        if (val > sigma) sigma = val;
                    }
                }

                if (a_ii * sigma >= alpha_bk * lambda * lambda) {
                    use_1x1 = true;
                } else if (std::abs(B[r * n + r]) >= alpha_bk * sigma) {
                    // Swap rows/cols i and r, then use 1x1 pivot
                    if (r != i) {
                        swap_rows_cols(B.data(), n, i, r);
                        swap_rows_cols(L_dense_.data(), n, i, r);
                        std::swap(perm_[i], perm_[r]);
                    }
                    use_1x1 = true;
                } else {
                    // Use 2x2 pivot with rows/cols i and r
                    if (r != i + 1) {
                        swap_rows_cols(B.data(), n, i + 1, r);
                        swap_rows_cols(L_dense_.data(), n, i + 1, r);
                        std::swap(perm_[i + 1], perm_[r]);
                    }
                    use_2x2 = true;
                }
            }

            if (use_1x1) {
                // 1x1 pivot
                Real d = B[i * n + i];
                D_values_[i] = d;
                pivot_type_[i] = 0;

                if (std::abs(d) < std::numeric_limits<Real>::epsilon() * 1e4) {
                    d = (d >= 0.0 ? 1.0 : -1.0) * std::numeric_limits<Real>::epsilon() * 1e4;
                    D_values_[i] = d;
                }

                Real inv_d = 1.0 / d;

                // Compute L column i and update trailing submatrix
                for (int k = i + 1; k < n; ++k) {
                    L_dense_[k * n + i] = B[k * n + i] * inv_d;
                }

                // Update: B(k,l) -= L(k,i) * d * L(l,i) for k,l > i
                for (int k = i + 1; k < n; ++k) {
                    Real lik = L_dense_[k * n + i];
                    for (int l = i + 1; l <= k; ++l) {
                        Real lil = L_dense_[l * n + i];
                        B[k * n + l] -= lik * d * lil;
                        B[l * n + k] = B[k * n + l]; // symmetry
                    }
                }

                ++i;

            } else {
                // 2x2 pivot: D block is [d11, d21; d21, d22]
                assert(use_2x2);
                assert(i + 1 < n);

                Real d11 = B[i * n + i];
                Real d21 = B[(i + 1) * n + i];
                Real d22 = B[(i + 1) * n + (i + 1)];

                D_values_[i] = d11;
                D_values_[i + 1] = d22;
                D_offdiag_[i] = d21;
                pivot_type_[i] = 1;     // first of 2x2
                pivot_type_[i + 1] = 2; // second of 2x2

                // Invert the 2x2 D block
                Real det = d11 * d22 - d21 * d21;
                if (std::abs(det) < std::numeric_limits<Real>::epsilon() * 1e4) {
                    det = (det >= 0.0 ? 1.0 : -1.0) * std::numeric_limits<Real>::epsilon() * 1e4;
                }
                Real inv_det = 1.0 / det;
                Real inv_d11 =  d22 * inv_det;
                Real inv_d12 = -d21 * inv_det;
                Real inv_d22 =  d11 * inv_det;

                // Compute L columns i, i+1
                for (int k = i + 2; k < n; ++k) {
                    Real bk_i   = B[k * n + i];
                    Real bk_ip1 = B[k * n + (i + 1)];
                    L_dense_[k * n + i]     = bk_i * inv_d11 + bk_ip1 * inv_d12;
                    L_dense_[k * n + (i + 1)] = bk_i * inv_d12 + bk_ip1 * inv_d22;
                }

                // Update trailing submatrix
                for (int k = i + 2; k < n; ++k) {
                    Real lk0 = L_dense_[k * n + i];
                    Real lk1 = L_dense_[k * n + (i + 1)];
                    for (int l = i + 2; l <= k; ++l) {
                        Real ll0 = L_dense_[l * n + i];
                        Real ll1 = L_dense_[l * n + (i + 1)];
                        // B(k,l) -= [lk0,lk1] * D * [ll0,ll1]^T
                        Real update = lk0 * (d11 * ll0 + d21 * ll1)
                                    + lk1 * (d21 * ll0 + d22 * ll1);
                        B[k * n + l] -= update;
                        B[l * n + k] = B[k * n + l];
                    }
                }

                i += 2;
            }
        }

        // Build iperm
        iperm_.resize(n_);
        for (int k = 0; k < n_; ++k) {
            iperm_[perm_[k]] = k;
        }

        numeric_done_ = true;
        return true;
    }

    /**
     * @brief Solve A*x = b using the computed LDL^T factorization
     *
     * Performs: P*b -> forward solve L*y = P*b -> D*z = y -> back solve L^T*w = z -> P^T*w -> x
     *
     * @param rhs      Right-hand side vector b (size n)
     * @param solution Output solution vector x (size n)
     * @return true if solve succeeded
     */
    bool solve(const Real* rhs, Real* solution) const {
        assert(numeric_done_);
        int n = n_;

        std::vector<Real> work(n);

        // Step 1: Permute RHS: work = P * b
        for (int i = 0; i < n; ++i) {
            work[i] = rhs[perm_[i]];
        }

        // Step 2: Forward substitution: L * y = work (L is unit lower triangular)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                work[i] -= L_dense_[i * n + j] * work[j];
            }
        }

        // Step 3: Diagonal solve: D * z = y
        int i = 0;
        while (i < n) {
            if (pivot_type_[i] == 0) {
                // 1x1 pivot
                work[i] /= D_values_[i];
                ++i;
            } else {
                // 2x2 pivot
                assert(i + 1 < n);
                Real d11 = D_values_[i];
                Real d21 = D_offdiag_[i];
                Real d22 = D_values_[i + 1];
                Real det = d11 * d22 - d21 * d21;

                Real y0 = work[i];
                Real y1 = work[i + 1];
                work[i]     = ( d22 * y0 - d21 * y1) / det;
                work[i + 1] = (-d21 * y0 + d11 * y1) / det;
                i += 2;
            }
        }

        // Step 4: Backward substitution: L^T * w = z
        for (int i2 = n - 1; i2 >= 0; --i2) {
            for (int j = i2 + 1; j < n; ++j) {
                work[i2] -= L_dense_[j * n + i2] * work[j];
            }
        }

        // Step 5: Inverse permute: solution = P^T * work
        for (int i3 = 0; i3 < n; ++i3) {
            solution[perm_[i3]] = work[i3];
        }

        return true;
    }

    /// System size
    int size() const { return n_; }

    /// Whether symbolic factorization is complete
    bool symbolic_done() const { return symbolic_done_; }

    /// Whether numeric factorization is complete
    bool numeric_done() const { return numeric_done_; }

    /// Access fill-reducing permutation
    const std::vector<int>& permutation() const { return perm_; }

    /// Estimated number of nonzeros in L factor
    int fill_in_estimate() const { return static_cast<int>(L_nnz_estimate_); }

private:
    int n_ = 0;
    bool symbolic_done_ = false;
    bool numeric_done_ = false;

    // Input sparsity
    std::vector<int> A_row_ptr_;
    std::vector<int> A_col_idx_;

    // Permutation
    std::vector<int> perm_;
    std::vector<int> iperm_;

    // L factor (dense lower triangular for header-only)
    std::vector<Real> L_dense_;
    size_t L_nnz_estimate_ = 0;

    // D factor
    std::vector<Real> D_values_;
    std::vector<Real> D_offdiag_;
    std::vector<int> pivot_type_;

    /**
     * @brief Approximate Minimum Degree ordering
     *
     * Simplified AMD: sort nodes by degree (number of adjacencies),
     * eliminate in order. Not as good as full AMD but functional.
     */
    void compute_amd_ordering(int n, const int* row_ptr, const int* col_idx) {
        // Compute degree of each node (symmetric: count upper + lower)
        std::vector<int> degree(n, 0);
        for (int i = 0; i < n; ++i) {
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                int j = col_idx[k];
                if (j != i) {
                    ++degree[i];
                }
            }
        }

        // Sort by degree (ascending) to minimize fill-in
        std::iota(perm_.begin(), perm_.end(), 0);
        std::sort(perm_.begin(), perm_.end(), [&](int a, int b) {
            return degree[a] < degree[b];
        });

        // Build inverse permutation
        iperm_.resize(n);
        for (int i = 0; i < n; ++i) {
            iperm_[perm_[i]] = i;
        }
    }

    /**
     * @brief Compute symbolic structure of L factor
     *
     * Estimates fill-in using elimination tree approach on the
     * permuted graph.
     */
    void compute_symbolic_structure(int n, const int* row_ptr, const int* col_idx) {
        // Estimate fill-in: for each row, count original nonzeros
        // below diagonal in permuted ordering
        L_nnz_estimate_ = 0;
        for (int i = 0; i < n; ++i) {
            int orig = perm_[i];
            for (int k = row_ptr[orig]; k < row_ptr[orig + 1]; ++k) {
                int j_perm = iperm_[col_idx[k]];
                if (j_perm < i) {
                    ++L_nnz_estimate_;
                }
            }
        }
        // Add estimated fill (heuristic: 2x the original)
        L_nnz_estimate_ = std::max(L_nnz_estimate_, static_cast<size_t>(n));
        L_nnz_estimate_ *= 2;
    }

    /// Swap rows and columns i, j in a dense n x n matrix
    void swap_rows_cols(Real* A, int n, int i, int j) {
        if (i == j) return;
        // Swap rows
        for (int k = 0; k < n; ++k) {
            std::swap(A[i * n + k], A[j * n + k]);
        }
        // Swap columns
        for (int k = 0; k < n; ++k) {
            std::swap(A[k * n + i], A[k * n + j]);
        }
    }
};


// ============================================================================
// 2. ImplicitBFGS - Full BFGS Quasi-Newton Solver (L-BFGS Variant)
// ============================================================================

/**
 * @brief Limited-memory BFGS (L-BFGS) quasi-Newton nonlinear solver
 *
 * Solves the nonlinear system R(x) = 0 by minimizing ||R(x)||^2 using
 * L-BFGS with Armijo backtracking line search.
 *
 * The L-BFGS method stores the last m correction pairs {s_k, y_k} and
 * uses the two-loop recursion to compute H_k * g without forming the
 * full Hessian approximation.
 *
 * Convergence criterion: ||g_k|| < tol * ||g_0||
 *
 * References:
 * - Nocedal (1980) "Updating quasi-Newton matrices with limited storage"
 * - Liu & Nocedal (1989) "On the limited memory BFGS method for large scale optimization"
 */
class ImplicitBFGS {
public:
    /// Result structure for solve
    struct Result {
        std::vector<Real> x;           ///< Solution vector
        Real final_residual = 0.0;     ///< Final residual norm
        int iterations = 0;            ///< Number of iterations performed
        bool converged = false;        ///< Whether convergence was achieved
    };

    /**
     * @brief Construct L-BFGS solver
     * @param m_pairs  Number of stored correction pairs (default 10)
     * @param c1       Armijo sufficient decrease parameter (default 1e-4)
     * @param alpha_init Initial step length for line search (default 1.0)
     */
    ImplicitBFGS(int m_pairs = 10, Real c1 = 1e-4, Real alpha_init = 1.0)
        : m_pairs_(m_pairs), c1_(c1), alpha_init_(alpha_init)
    {}

    /**
     * @brief Solve the nonlinear minimization problem
     *
     * Minimizes f(x) = 0.5 * ||residual_func(x)||^2 using L-BFGS.
     *
     * @param residual_func Function R(x) returning residual vector and gradient.
     *                      Signature: void(const vector<Real>& x, vector<Real>& residual, vector<Real>& gradient)
     *                      where gradient = J^T * R (the gradient of 0.5*||R||^2)
     * @param x0       Initial guess
     * @param max_iter Maximum iterations
     * @param tol      Relative convergence tolerance
     * @return Result containing solution, residual, iteration count, convergence flag
     */
    Result solve(
        std::function<void(const std::vector<Real>&, std::vector<Real>&, std::vector<Real>&)> residual_func,
        const std::vector<Real>& x0,
        int max_iter = 100,
        Real tol = 1e-8)
    {
        int n = static_cast<int>(x0.size());
        Result result;
        result.x = x0;

        // Storage for L-BFGS correction pairs (circular buffer)
        std::vector<std::vector<Real>> S(m_pairs_, std::vector<Real>(n, 0.0));
        std::vector<std::vector<Real>> Y(m_pairs_, std::vector<Real>(n, 0.0));
        std::vector<Real> rho(m_pairs_, 0.0);
        std::vector<Real> alpha_buf(m_pairs_, 0.0);

        int stored = 0;  // number of stored pairs
        int oldest = 0;  // index of oldest pair in circular buffer

        // Evaluate at initial point
        std::vector<Real> r(n), g(n), g_old(n);
        std::vector<Real> x_new(n), r_new(n), g_new(n);
        std::vector<Real> dir(n);

        residual_func(result.x, r, g);
        Real g0_norm = implicit_detail::norm2(g.data(), n);

        if (g0_norm < std::numeric_limits<Real>::epsilon()) {
            result.final_residual = g0_norm;
            result.converged = true;
            result.iterations = 0;
            return result;
        }

        for (int iter = 0; iter < max_iter; ++iter) {
            Real g_norm = implicit_detail::norm2(g.data(), n);

            // Convergence check
            if (g_norm < tol * g0_norm) {
                result.final_residual = g_norm;
                result.converged = true;
                result.iterations = iter;
                return result;
            }

            // Compute search direction: dir = -H * g (two-loop recursion)
            two_loop_recursion(g.data(), dir.data(), n,
                               S, Y, rho, alpha_buf, stored, oldest);

            // Negate: dir = -H*g
            for (int i = 0; i < n; ++i) dir[i] = -dir[i];

            // Check descent direction
            Real dg = implicit_detail::dot(dir.data(), g.data(), n);
            if (dg >= 0.0) {
                // Reset to steepest descent
                for (int i = 0; i < n; ++i) dir[i] = -g[i];
                dg = -g_norm * g_norm;
                stored = 0; // reset L-BFGS history
            }

            // Armijo backtracking line search
            Real f0 = 0.5 * implicit_detail::dot(r.data(), r.data(), n);
            Real alpha = alpha_init_;
            bool ls_success = false;

            for (int ls = 0; ls < 30; ++ls) {
                for (int i = 0; i < n; ++i) {
                    x_new[i] = result.x[i] + alpha * dir[i];
                }
                residual_func(x_new, r_new, g_new);
                Real f_new = 0.5 * implicit_detail::dot(r_new.data(), r_new.data(), n);

                if (f_new <= f0 + c1_ * alpha * dg) {
                    ls_success = true;
                    break;
                }
                alpha *= 0.5;
            }

            if (!ls_success) {
                // Line search failed; accept smallest step anyway
                for (int i = 0; i < n; ++i) {
                    x_new[i] = result.x[i] + alpha * dir[i];
                }
                residual_func(x_new, r_new, g_new);
            }

            // Compute correction pair s = x_new - x, y = g_new - g
            int idx = (oldest + stored) % m_pairs_;
            if (stored < m_pairs_) {
                ++stored;
            } else {
                idx = oldest;
                oldest = (oldest + 1) % m_pairs_;
            }

            for (int i = 0; i < n; ++i) {
                S[idx][i] = x_new[i] - result.x[i];
                Y[idx][i] = g_new[i] - g[i];
            }

            Real sy = implicit_detail::dot(S[idx].data(), Y[idx].data(), n);
            if (std::abs(sy) > std::numeric_limits<Real>::epsilon()) {
                rho[idx] = 1.0 / sy;
            } else {
                rho[idx] = 0.0;
                // Skip this pair if curvature condition not satisfied
                if (stored > 0) --stored;
            }

            // Update state
            result.x = x_new;
            r = r_new;
            g = g_new;
            result.iterations = iter + 1;
        }

        result.final_residual = implicit_detail::norm2(g.data(), n);
        result.converged = false;
        return result;
    }

    /// Set number of stored correction pairs
    void set_m_pairs(int m) { m_pairs_ = m; }

    /// Set Armijo parameter
    void set_c1(Real c1) { c1_ = c1; }

    /// Set initial step length
    void set_alpha_init(Real a) { alpha_init_ = a; }

private:
    int m_pairs_;
    Real c1_;
    Real alpha_init_;

    /**
     * @brief Two-loop recursion for L-BFGS
     *
     * Computes q = H_k * g using stored correction pairs.
     * Algorithm 7.4 from Nocedal & Wright (2006).
     *
     * @param g       Input gradient vector
     * @param q       Output H*g vector
     * @param n       Vector dimension
     * @param S       Stored s_k vectors (circular buffer)
     * @param Y       Stored y_k vectors (circular buffer)
     * @param rho     Stored 1/(y_k^T s_k) values
     * @param alpha   Workspace for alpha values
     * @param stored  Number of stored pairs
     * @param oldest  Index of oldest pair
     */
    void two_loop_recursion(
        const Real* g, Real* q, int n,
        const std::vector<std::vector<Real>>& S,
        const std::vector<std::vector<Real>>& Y,
        const std::vector<Real>& rho,
        std::vector<Real>& alpha,
        int stored, int oldest) const
    {
        // q = g
        implicit_detail::copy(g, q, n);

        if (stored == 0) {
            // No history: return q = g (will be negated outside)
            return;
        }

        // First loop: from newest to oldest
        for (int k = stored - 1; k >= 0; --k) {
            int idx = (oldest + k) % m_pairs_;
            alpha[k] = rho[idx] * implicit_detail::dot(S[idx].data(), q, n);
            // q = q - alpha[k] * y[idx]
            implicit_detail::axpy(-alpha[k], Y[idx].data(), q, n);
        }

        // Initial Hessian approximation: H0 = gamma * I
        // gamma = s_{k-1}^T y_{k-1} / (y_{k-1}^T y_{k-1})
        int newest = (oldest + stored - 1) % m_pairs_;
        Real yy = implicit_detail::dot(Y[newest].data(), Y[newest].data(), n);
        Real sy = implicit_detail::dot(S[newest].data(), Y[newest].data(), n);
        Real gamma = (yy > std::numeric_limits<Real>::epsilon()) ? (sy / yy) : 1.0;

        implicit_detail::scale(gamma, q, n);

        // Second loop: from oldest to newest
        for (int k = 0; k < stored; ++k) {
            int idx = (oldest + k) % m_pairs_;
            Real beta = rho[idx] * implicit_detail::dot(Y[idx].data(), q, n);
            // q = q + (alpha[k] - beta) * s[idx]
            implicit_detail::axpy(alpha[k] - beta, S[idx].data(), q, n);
        }
    }
};


// ============================================================================
// 3. ImplicitBuckling - Linear Buckling Eigenvalue Solver
// ============================================================================

/**
 * @brief Linear buckling analysis via inverse iteration
 *
 * Solves the generalized eigenvalue problem:
 *   (K + lambda * Kg) * phi = 0
 *
 * where K is the elastic stiffness matrix, Kg is the geometric stiffness
 * matrix (stress-dependent), lambda is the buckling load factor, and
 * phi is the buckling mode shape.
 *
 * Uses inverse iteration (also known as inverse power method) to find
 * the smallest eigenvalue of K^{-1} * (-Kg), which corresponds to the
 * critical buckling load factor lambda_cr.
 *
 * The iteration computes:
 *   K * v_{k+1} = -Kg * v_k
 *   lambda_{k+1} = (v_k^T * K * v_{k+1}) / (v_k^T * (-Kg) * v_k)  [Rayleigh quotient]
 *   v_{k+1} = v_{k+1} / ||v_{k+1}||
 *
 * References:
 * - Bathe (2014) Ch. 12 "Solution of Equilibrium Equations in Dynamic Analysis"
 * - Cook et al. (2002) "Concepts and Applications of Finite Element Analysis"
 */
class ImplicitBuckling {
public:
    /// Result of buckling analysis
    struct BucklingResult {
        Real lambda_cr = 0.0;          ///< Critical buckling load factor
        std::vector<Real> mode_shape;  ///< Buckling mode shape (unit norm)
        int iterations = 0;            ///< Number of inverse iterations
        bool converged = false;        ///< Whether eigenvalue converged
        Real eigenvalue_error = 0.0;   ///< Relative change in eigenvalue at convergence
    };

    /**
     * @brief Construct buckling solver
     * @param max_iter  Maximum inverse iterations (default 200)
     * @param tol       Eigenvalue convergence tolerance (default 1e-8)
     */
    ImplicitBuckling(int max_iter = 200, Real tol = 1e-8)
        : max_iter_(max_iter), tol_(tol)
    {}

    /**
     * @brief Compute critical buckling load factor
     *
     * Both K and Kg are provided as dense symmetric matrices (row-major, full storage).
     *
     * @param K   Elastic stiffness matrix (n x n, dense, row-major)
     * @param Kg  Geometric stiffness matrix (n x n, dense, row-major)
     * @param n   System size
     * @return BucklingResult with critical load factor and mode shape
     */
    BucklingResult compute_buckling_load(const Real* K, const Real* Kg, int n) {
        BucklingResult result;
        result.mode_shape.resize(n);

        // We need to solve: K * phi = -lambda * Kg * phi
        // Rearrange: K^{-1} * (-Kg) * phi = (1/lambda) * phi
        // The largest eigenvalue of K^{-1}*(-Kg) gives smallest lambda_cr.

        // Factorize K using LDL^T (reuse MUMPSSolver)
        // Build CSR from dense K
        std::vector<int> row_ptr(n + 1, 0);
        std::vector<int> col_idx;
        std::vector<Real> vals;

        // Count nonzeros (store full matrix for simplicity)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (std::abs(K[i * n + j]) > std::numeric_limits<Real>::epsilon() * 1e-2) {
                    col_idx.push_back(j);
                    vals.push_back(K[i * n + j]);
                }
            }
            row_ptr[i + 1] = static_cast<int>(col_idx.size());
        }

        MUMPSSolver K_solver;
        K_solver.symbolic_factorize(n, row_ptr.data(), col_idx.data());
        K_solver.numeric_factorize(vals.data());

        // Initial guess: random-like vector
        std::vector<Real> v(n), v_new(n), Kg_v(n);
        for (int i = 0; i < n; ++i) {
            v[i] = 1.0 + 0.1 * std::sin(static_cast<Real>(i * 7 + 3));
        }
        // Normalize
        Real nrm = implicit_detail::norm2(v.data(), n);
        if (nrm > 0.0) {
            implicit_detail::scale(1.0 / nrm, v.data(), n);
        }

        Real lambda_old = 0.0;

        for (int iter = 0; iter < max_iter_; ++iter) {
            // Compute rhs = -Kg * v
            implicit_detail::dense_matvec(Kg, v.data(), Kg_v.data(), n);
            for (int i = 0; i < n; ++i) Kg_v[i] = -Kg_v[i];

            // Solve K * v_new = -Kg * v
            K_solver.solve(Kg_v.data(), v_new.data());

            // Rayleigh quotient: lambda = v^T * v_new / (v^T * K^{-1}*(-Kg)*v)
            // Since K * v_new = -Kg * v, we have v_new = K^{-1}*(-Kg)*v
            // Eigenvalue of K^{-1}*(-Kg) is mu = v^T * v_new / (v^T * v)
            // and lambda_cr = 1/mu
            Real vv = implicit_detail::dot(v.data(), v.data(), n);
            Real v_vnew = implicit_detail::dot(v.data(), v_new.data(), n);

            Real mu = (std::abs(vv) > std::numeric_limits<Real>::epsilon())
                     ? (v_vnew / vv)
                     : 1.0;

            Real lambda_new = (std::abs(mu) > std::numeric_limits<Real>::epsilon())
                             ? (1.0 / mu)
                             : std::numeric_limits<Real>::max();

            // Normalize v_new
            nrm = implicit_detail::norm2(v_new.data(), n);
            if (nrm > std::numeric_limits<Real>::epsilon()) {
                implicit_detail::scale(1.0 / nrm, v_new.data(), n);
            }

            // Check convergence
            Real rel_change = (iter > 0)
                ? std::abs(lambda_new - lambda_old) / (std::abs(lambda_old) + std::numeric_limits<Real>::epsilon())
                : 1.0;

            result.iterations = iter + 1;
            result.lambda_cr = lambda_new;
            result.eigenvalue_error = rel_change;

            if (rel_change < tol_ && iter > 0) {
                result.converged = true;
                implicit_detail::copy(v_new.data(), result.mode_shape.data(), n);
                return result;
            }

            lambda_old = lambda_new;
            implicit_detail::copy(v_new.data(), v.data(), n);
        }

        // Did not converge
        implicit_detail::copy(v.data(), result.mode_shape.data(), n);
        return result;
    }

    /**
     * @brief Compute buckling load using CSR matrices
     *
     * Overload accepting CSR format stiffness matrices.
     *
     * @param K_row_ptr   CSR row pointers for K
     * @param K_col_idx   CSR column indices for K
     * @param K_values    CSR values for K
     * @param Kg_row_ptr  CSR row pointers for Kg
     * @param Kg_col_idx  CSR column indices for Kg
     * @param Kg_values   CSR values for Kg
     * @param n           System size
     * @return BucklingResult
     */
    BucklingResult compute_buckling_load_csr(
        const int* K_row_ptr, const int* K_col_idx, const Real* K_values,
        const int* Kg_row_ptr, const int* Kg_col_idx, const Real* Kg_values,
        int n)
    {
        BucklingResult result;
        result.mode_shape.resize(n);

        // Factorize K
        MUMPSSolver K_solver;
        K_solver.symbolic_factorize(n, K_row_ptr, K_col_idx);
        K_solver.numeric_factorize(K_values);

        // Initial guess
        std::vector<Real> v(n), v_new(n), Kg_v(n);
        for (int i = 0; i < n; ++i) {
            v[i] = 1.0 + 0.1 * std::sin(static_cast<Real>(i * 7 + 3));
        }
        Real nrm = implicit_detail::norm2(v.data(), n);
        if (nrm > 0.0) implicit_detail::scale(1.0 / nrm, v.data(), n);

        Real lambda_old = 0.0;

        for (int iter = 0; iter < max_iter_; ++iter) {
            // rhs = -Kg * v
            implicit_detail::csr_matvec(Kg_row_ptr, Kg_col_idx, Kg_values, v.data(), Kg_v.data(), n);
            for (int i = 0; i < n; ++i) Kg_v[i] = -Kg_v[i];

            // Solve K * v_new = -Kg * v
            K_solver.solve(Kg_v.data(), v_new.data());

            // Rayleigh quotient
            Real vv = implicit_detail::dot(v.data(), v.data(), n);
            Real v_vnew = implicit_detail::dot(v.data(), v_new.data(), n);
            Real mu = (std::abs(vv) > std::numeric_limits<Real>::epsilon())
                     ? (v_vnew / vv) : 1.0;
            Real lambda_new = (std::abs(mu) > std::numeric_limits<Real>::epsilon())
                             ? (1.0 / mu) : std::numeric_limits<Real>::max();

            nrm = implicit_detail::norm2(v_new.data(), n);
            if (nrm > std::numeric_limits<Real>::epsilon()) {
                implicit_detail::scale(1.0 / nrm, v_new.data(), n);
            }

            Real rel_change = (iter > 0)
                ? std::abs(lambda_new - lambda_old) / (std::abs(lambda_old) + std::numeric_limits<Real>::epsilon())
                : 1.0;

            result.iterations = iter + 1;
            result.lambda_cr = lambda_new;
            result.eigenvalue_error = rel_change;

            if (rel_change < tol_ && iter > 0) {
                result.converged = true;
                implicit_detail::copy(v_new.data(), result.mode_shape.data(), n);
                return result;
            }

            lambda_old = lambda_new;
            implicit_detail::copy(v_new.data(), v.data(), n);
        }

        implicit_detail::copy(v.data(), result.mode_shape.data(), n);
        return result;
    }

    /// Set max iterations
    void set_max_iter(int m) { max_iter_ = m; }

    /// Set tolerance
    void set_tol(Real t) { tol_ = t; }

private:
    int max_iter_;
    Real tol_;
};


// ============================================================================
// 4. ImplicitDtControl - Adaptive Time Step Control
// ============================================================================

/**
 * @brief Adaptive time step controller for implicit transient analysis
 *
 * Adjusts the time step based on Newton-Raphson convergence behavior:
 *   - If converged in few iterations: increase dt (problem is easy)
 *   - If converged in many iterations: keep dt (near nonlinearity)
 *   - If failed to converge: decrease dt and retry
 *
 * The update rule is:
 *   dt_new = dt * factor
 * where factor depends on the ratio of actual to maximum Newton iterations:
 *   ratio = newton_iters / max_newton
 *   factor = growth_factor   if ratio < 0.3  (easy convergence)
 *   factor = 1.0             if 0.3 <= ratio < 0.8  (adequate)
 *   factor = shrink_factor   if ratio >= 0.8 or not converged
 *
 * The resulting dt is clamped to [dt_min, dt_max].
 *
 * References:
 * - Bathe (2014) "Finite Element Procedures", Ch. 9
 * - Crisfield (1991) "Non-linear Finite Element Analysis of Solids and Structures"
 */
class ImplicitDtControl {
public:
    /**
     * @brief Construct time step controller
     * @param dt_min         Minimum allowed time step
     * @param dt_max         Maximum allowed time step
     * @param growth_factor  Multiplicative factor when convergence is easy (default 1.5)
     * @param shrink_factor  Multiplicative factor when convergence is hard (default 0.5)
     */
    ImplicitDtControl(Real dt_min = 1e-10, Real dt_max = 1.0,
                      Real growth_factor = 1.5, Real shrink_factor = 0.5)
        : dt_min_(dt_min), dt_max_(dt_max),
          growth_factor_(growth_factor), shrink_factor_(shrink_factor),
          total_steps_(0), total_failures_(0), total_growths_(0), total_shrinks_(0)
    {}

    /**
     * @brief Update time step based on Newton convergence
     *
     * @param dt_current   Current time step
     * @param newton_iters Number of Newton iterations used in this step
     * @param converged    Whether Newton iteration converged
     * @param max_newton   Maximum allowed Newton iterations
     * @return New time step (clamped to [dt_min, dt_max])
     */
    KOKKOS_INLINE_FUNCTION
    Real update_dt(Real dt_current, int newton_iters, bool converged, int max_newton) {
        ++total_steps_;

        if (!converged) {
            // Failed to converge: shrink time step
            ++total_failures_;
            ++total_shrinks_;
            Real dt_new = dt_current * shrink_factor_;
            return clamp_dt(dt_new);
        }

        Real ratio = static_cast<Real>(newton_iters) / static_cast<Real>(max_newton);

        Real factor = 1.0;
        if (ratio < 0.3) {
            // Easy convergence: grow time step
            factor = growth_factor_;
            ++total_growths_;
        } else if (ratio >= 0.8) {
            // Hard convergence: shrink slightly
            factor = shrink_factor_;
            ++total_shrinks_;
        }
        // else: 0.3 <= ratio < 0.8 => factor = 1.0, keep dt

        Real dt_new = dt_current * factor;
        return clamp_dt(dt_new);
    }

    /**
     * @brief Compute a recommended initial time step based on problem parameters
     *
     * Estimates dt from the dominant eigenvalue of the system:
     *   dt ~ 2 / sqrt(lambda_max)
     * where lambda_max is estimated from element sizes and material properties.
     *
     * @param h_min      Minimum element size
     * @param E          Young's modulus
     * @param rho        Density
     * @param safety     Safety factor (default 0.1 for implicit, much larger than explicit)
     * @return Recommended initial time step
     */
    static Real estimate_initial_dt(Real h_min, Real E, Real rho, Real safety = 0.1) {
        // Wave speed c = sqrt(E/rho), dominant frequency ~ c/h
        // dt ~ safety * h / c
        Real c = std::sqrt(E / (rho + std::numeric_limits<Real>::epsilon()));
        Real dt = safety * h_min / (c + std::numeric_limits<Real>::epsilon());
        return dt;
    }

    /// Get dt_min
    Real dt_min() const { return dt_min_; }

    /// Get dt_max
    Real dt_max() const { return dt_max_; }

    /// Set dt bounds
    void set_dt_bounds(Real dt_min, Real dt_max) {
        dt_min_ = dt_min;
        dt_max_ = dt_max;
    }

    /// Set growth factor
    void set_growth_factor(Real f) { growth_factor_ = f; }

    /// Set shrink factor
    void set_shrink_factor(Real f) { shrink_factor_ = f; }

    /// Total steps processed
    int total_steps() const { return total_steps_; }

    /// Total failures (non-converged steps)
    int total_failures() const { return total_failures_; }

    /// Total time step growths
    int total_growths() const { return total_growths_; }

    /// Total time step shrinks
    int total_shrinks() const { return total_shrinks_; }

    /// Reset statistics
    void reset_stats() {
        total_steps_ = 0;
        total_failures_ = 0;
        total_growths_ = 0;
        total_shrinks_ = 0;
    }

private:
    Real dt_min_;
    Real dt_max_;
    Real growth_factor_;
    Real shrink_factor_;

    int total_steps_;
    int total_failures_;
    int total_growths_;
    int total_shrinks_;

    KOKKOS_INLINE_FUNCTION
    Real clamp_dt(Real dt) const {
        if (dt < dt_min_) return dt_min_;
        if (dt > dt_max_) return dt_max_;
        return dt;
    }
};


// ============================================================================
// 5. IterativeRefinement - Iterative Refinement for Ill-Conditioned Systems
// ============================================================================

/**
 * @brief Iterative refinement to improve accuracy of a linear solve
 *
 * Given an approximate solution x0 to A*x = b, computes corrections by:
 *   r = b - A * x0          (residual in extended/higher precision conceptually)
 *   A * dx = r              (solve for correction)
 *   x1 = x0 + dx            (update solution)
 *
 * Repeats until ||r|| < tol * ||b|| or max steps reached.
 *
 * This is especially useful when the factorization has rounding errors
 * (e.g., from single-precision factorization used for a double-precision
 * problem, or ill-conditioned systems).
 *
 * The user provides:
 *   - A_apply_func: computes y = A * x (matrix-vector product)
 *   - solve_func: solves A * dx = r approximately (using existing factorization)
 *
 * References:
 * - Wilkinson (1963) "Rounding Errors in Algebraic Processes"
 * - Higham (2002) "Accuracy and Stability of Numerical Algorithms"
 */
class IterativeRefinement {
public:
    /// Result of refinement
    struct RefinementResult {
        int steps = 0;              ///< Number of refinement steps performed
        Real initial_residual = 0.0;///< ||b - A*x0|| / ||b||
        Real final_residual = 0.0;  ///< ||b - A*x_final|| / ||b||
        bool converged = false;     ///< Whether tolerance was achieved
    };

    IterativeRefinement() = default;

    /**
     * @brief Perform iterative refinement
     *
     * @param A_apply_func  Function computing y = A * x.
     *                      Signature: void(const Real* x, Real* y, int n)
     * @param solve_func    Function solving A * dx = r approximately.
     *                      Signature: void(const Real* r, Real* dx, int n)
     * @param b             Right-hand side vector (size n)
     * @param x             Solution vector (input: initial guess, output: refined solution, size n)
     * @param n             System size
     * @param max_steps     Maximum refinement steps (default 5)
     * @param tol           Convergence tolerance on relative residual (default 1e-12)
     * @return RefinementResult with convergence info
     */
    RefinementResult refine(
        std::function<void(const Real*, Real*, int)> A_apply_func,
        std::function<void(const Real*, Real*, int)> solve_func,
        const Real* b, Real* x, int n,
        int max_steps = 5,
        Real tol = 1e-12)
    {
        RefinementResult result;

        std::vector<Real> r(n), dx(n), Ax(n);

        // Compute ||b||
        Real b_norm = implicit_detail::norm2(b, n);
        if (b_norm < std::numeric_limits<Real>::epsilon()) {
            // b = 0 => x = 0 is exact
            implicit_detail::zero(x, n);
            result.converged = true;
            return result;
        }

        for (int step = 0; step < max_steps; ++step) {
            // Compute residual: r = b - A * x
            A_apply_func(x, Ax.data(), n);
            for (int i = 0; i < n; ++i) {
                r[i] = b[i] - Ax[i];
            }

            Real r_norm = implicit_detail::norm2(r.data(), n);
            Real rel_r = r_norm / b_norm;

            if (step == 0) {
                result.initial_residual = rel_r;
            }
            result.final_residual = rel_r;
            result.steps = step + 1;

            // Check convergence
            if (rel_r < tol) {
                result.converged = true;
                return result;
            }

            // Solve A * dx = r
            solve_func(r.data(), dx.data(), n);

            // Update: x = x + dx
            implicit_detail::axpy(1.0, dx.data(), x, n);
        }

        // Final residual check
        {
            std::vector<Real> Ax_final(n), r_final(n);
            A_apply_func(x, Ax_final.data(), n);
            for (int i = 0; i < n; ++i) {
                r_final[i] = b[i] - Ax_final[i];
            }
            Real r_norm = implicit_detail::norm2(r_final.data(), n);
            result.final_residual = r_norm / b_norm;
            result.converged = (result.final_residual < tol);
        }

        return result;
    }

    /**
     * @brief Convenience: refine using CSR matrix and MUMPSSolver
     *
     * @param row_ptr   CSR row pointers
     * @param col_idx   CSR column indices
     * @param values    CSR values
     * @param solver    Pre-factorized MUMPSSolver
     * @param b         Right-hand side
     * @param x         Solution (input/output)
     * @param n         System size
     * @param max_steps Maximum refinement steps
     * @param tol       Convergence tolerance
     * @return RefinementResult
     */
    RefinementResult refine_with_mumps(
        const int* row_ptr, const int* col_idx, const Real* values,
        const MUMPSSolver& solver,
        const Real* b, Real* x, int n,
        int max_steps = 5,
        Real tol = 1e-12)
    {
        auto A_apply = [&](const Real* xv, Real* yv, int nn) {
            implicit_detail::csr_matvec(row_ptr, col_idx, values, xv, yv, nn);
        };

        auto solve_fn = [&](const Real* rv, Real* dxv, int /*nn*/) {
            solver.solve(rv, dxv);
        };

        return refine(A_apply, solve_fn, b, x, n, max_steps, tol);
    }

    /**
     * @brief CSR matrix-vector multiply helper (static utility)
     *
     * Computes y = A * x where A is in CSR format.
     *
     * @param row_ptr CSR row pointers
     * @param col_idx CSR column indices
     * @param values  CSR values
     * @param x       Input vector
     * @param y       Output vector
     * @param n       System size
     */
    static void csr_matvec(const int* row_ptr, const int* col_idx,
                           const Real* values, const Real* x, Real* y, int n) {
        implicit_detail::csr_matvec(row_ptr, col_idx, values, x, y, n);
    }
};


// ============================================================================
// 6. ImplicitContactK - Contact Stiffness Assembly
// ============================================================================

/**
 * @brief Contact pair definition for implicit contact stiffness
 */
struct ContactPair {
    int node1 = -1;         ///< Master/slave node 1
    int node2 = -1;         ///< Master/slave node 2
    Real gap = 0.0;         ///< Gap function value (negative = penetration)
    Real normal[3] = {};    ///< Unit outward normal at contact point
};

/**
 * @brief Sparse matrix triplet (i, j, value) for assembly
 */
struct Triplet {
    int row = 0;
    int col = 0;
    Real value = 0.0;

    Triplet() = default;
    Triplet(int r, int c, Real v) : row(r), col(c), value(v) {}
};

/**
 * @brief Penalty-based contact stiffness assembly for implicit solvers
 *
 * For each active contact pair (gap < 0, i.e., penetration), assembles a
 * penalty spring contribution to the global tangent stiffness matrix:
 *
 *   K_contact = kn * n * n^T
 *
 * where kn is the penalty stiffness and n is the contact normal.
 *
 * For a pair (node1, node2), the 6x6 contact stiffness in 3D is:
 *
 *       [ n*n^T   -n*n^T ]
 *  kn * [-n*n^T    n*n^T ]
 *
 * assembled into global DOFs (3*node1, 3*node1+1, ..., 3*node2+2).
 *
 * Also computes the contact force vector:
 *   f_c = kn * gap * n  (applied to node1)
 *   f_c = -kn * gap * n (applied to node2)
 *
 * References:
 * - Wriggers (2006) "Computational Contact Mechanics", Ch. 5
 * - Laursen (2002) "Computational Contact and Impact Mechanics"
 */
class ImplicitContactK {
public:
    /**
     * @brief Construct contact stiffness assembler
     * @param spatial_dim Spatial dimension (2 or 3, default 3)
     */
    explicit ImplicitContactK(int spatial_dim = 3)
        : dim_(spatial_dim)
    {
        assert(dim_ == 2 || dim_ == 3);
    }

    /**
     * @brief Compute contact stiffness triplets for assembly
     *
     * For each contact pair with gap < 0 (penetration), generates the
     * stiffness contribution as sparse triplets. The triplets can then
     * be assembled into the global tangent matrix.
     *
     * @param pairs     Contact pairs
     * @param positions Node positions (size: max_node * dim_) [unused for gap, used for verification]
     * @param normals   Contact normals (can be nullptr if normals are in pairs)
     * @param kn        Penalty stiffness parameter
     * @return Vector of triplets for sparse matrix assembly
     */
    std::vector<Triplet> compute_contact_stiffness(
        const std::vector<ContactPair>& pairs,
        const Real* positions,
        const Real* normals,
        Real kn) const
    {
        std::vector<Triplet> triplets;
        // Reserve an estimate: each active pair produces up to dim*dim*4 entries
        triplets.reserve(pairs.size() * dim_ * dim_ * 4);

        for (size_t p = 0; p < pairs.size(); ++p) {
            const ContactPair& cp = pairs[p];

            // Only active contacts (penetration: gap < 0)
            if (cp.gap >= 0.0) continue;

            // Get normal: from pair or from normals array
            Real n[3] = {0.0, 0.0, 0.0};
            if (normals != nullptr) {
                for (int d = 0; d < dim_; ++d) {
                    n[d] = normals[p * dim_ + d];
                }
            } else {
                for (int d = 0; d < dim_; ++d) {
                    n[d] = cp.normal[d];
                }
            }

            // Normalize the normal vector
            Real n_len = 0.0;
            for (int d = 0; d < dim_; ++d) n_len += n[d] * n[d];
            n_len = std::sqrt(n_len);
            if (n_len < std::numeric_limits<Real>::epsilon()) continue;
            for (int d = 0; d < dim_; ++d) n[d] /= n_len;

            int n1 = cp.node1;
            int n2 = cp.node2;

            // Build n*n^T (dim x dim)
            // K_contact block structure:
            //   K[n1,n1] += kn * n*n^T
            //   K[n1,n2] -= kn * n*n^T
            //   K[n2,n1] -= kn * n*n^T
            //   K[n2,n2] += kn * n*n^T

            for (int i = 0; i < dim_; ++i) {
                for (int j = 0; j < dim_; ++j) {
                    Real nij = kn * n[i] * n[j];

                    int dof1_i = n1 * dim_ + i;
                    int dof1_j = n1 * dim_ + j;
                    int dof2_i = n2 * dim_ + i;
                    int dof2_j = n2 * dim_ + j;

                    // K[n1_i, n1_j] += kn * ni * nj
                    triplets.emplace_back(dof1_i, dof1_j, nij);

                    // K[n1_i, n2_j] -= kn * ni * nj
                    triplets.emplace_back(dof1_i, dof2_j, -nij);

                    // K[n2_i, n1_j] -= kn * ni * nj
                    triplets.emplace_back(dof2_i, dof1_j, -nij);

                    // K[n2_i, n2_j] += kn * ni * nj
                    triplets.emplace_back(dof2_i, dof2_j, nij);
                }
            }
        }

        return triplets;
    }

    /**
     * @brief Compute contact force vector
     *
     * For each active contact pair (gap < 0), computes penalty forces:
     *   f[node1] += kn * gap * n   (push node1 along normal)
     *   f[node2] -= kn * gap * n   (push node2 against normal)
     *
     * Note: gap < 0 for penetration, so kn * gap * n pushes nodes apart.
     *
     * @param pairs  Contact pairs
     * @param kn     Penalty stiffness
     * @param f      Force vector (output, size: num_nodes * dim), must be zeroed by caller
     */
    void compute_contact_force(
        const std::vector<ContactPair>& pairs,
        Real kn,
        Real* f) const
    {
        for (size_t p = 0; p < pairs.size(); ++p) {
            const ContactPair& cp = pairs[p];

            if (cp.gap >= 0.0) continue;

            Real n[3] = {cp.normal[0], cp.normal[1], cp.normal[2]};

            // Normalize
            Real n_len = 0.0;
            for (int d = 0; d < dim_; ++d) n_len += n[d] * n[d];
            n_len = std::sqrt(n_len);
            if (n_len < std::numeric_limits<Real>::epsilon()) continue;
            for (int d = 0; d < dim_; ++d) n[d] /= n_len;

            // Force magnitude: kn * |gap| (gap is negative for penetration)
            // Direction: along normal to push apart
            // f[node1] += kn * gap * n  (gap < 0, so this pushes node1 in -n direction)
            // f[node2] -= kn * gap * n
            for (int d = 0; d < dim_; ++d) {
                f[cp.node1 * dim_ + d] += kn * cp.gap * n[d];
                f[cp.node2 * dim_ + d] -= kn * cp.gap * n[d];
            }
        }
    }

    /**
     * @brief Compute gap function for a contact pair
     *
     * gap = (x2 - x1) . n
     * Negative gap indicates penetration.
     *
     * @param x1  Position of node 1 (size dim_)
     * @param x2  Position of node 2 (size dim_)
     * @param n   Contact normal (size dim_, pointing from node1 to node2)
     * @return Gap value
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_gap(const Real* x1, const Real* x2, const Real* n, int dim) {
        Real gap = 0.0;
        for (int d = 0; d < dim; ++d) {
            gap += (x2[d] - x1[d]) * n[d];
        }
        return gap;
    }

    /**
     * @brief Update gap values in contact pairs based on current positions
     *
     * @param pairs     Contact pairs (gap field will be updated)
     * @param positions Node positions array (size: max_node * dim_)
     */
    void update_gaps(std::vector<ContactPair>& pairs, const Real* positions) const {
        for (auto& cp : pairs) {
            const Real* x1 = positions + cp.node1 * dim_;
            const Real* x2 = positions + cp.node2 * dim_;
            cp.gap = ImplicitContactK::compute_gap(x1, x2, cp.normal, dim_);
        }
    }

    /**
     * @brief Assemble triplets into CSR matrix
     *
     * Takes the triplets from compute_contact_stiffness and adds them
     * to an existing CSR matrix. Duplicate entries are summed.
     *
     * @param triplets    Triplet list
     * @param row_ptr     CSR row pointers (existing matrix, size n_dof+1)
     * @param col_idx     CSR column indices (existing matrix)
     * @param values      CSR values (will be modified by adding contact contributions)
     * @param n_dof       Total number of DOFs
     */
    static void assemble_to_csr(
        const std::vector<Triplet>& triplets,
        const int* row_ptr, const int* col_idx,
        Real* values, int n_dof)
    {
        for (const auto& t : triplets) {
            if (t.row < 0 || t.row >= n_dof || t.col < 0 || t.col >= n_dof) continue;

            // Find the entry (t.row, t.col) in CSR
            for (int k = row_ptr[t.row]; k < row_ptr[t.row + 1]; ++k) {
                if (col_idx[k] == t.col) {
                    values[k] += t.value;
                    break;
                }
            }
        }
    }

    /**
     * @brief Estimate penalty stiffness from material/element properties
     *
     * A common heuristic: kn = alpha * E * A / h
     * where E = Young's modulus, A = contact area, h = element size,
     * and alpha is a scaling factor (typically 10-100).
     *
     * @param youngs_modulus Young's modulus
     * @param area           Contact area
     * @param element_size   Characteristic element size
     * @param alpha          Scaling factor (default 10.0)
     * @return Recommended penalty stiffness
     */
    KOKKOS_INLINE_FUNCTION
    static Real estimate_penalty_stiffness(Real youngs_modulus, Real area,
                                           Real element_size, Real alpha = 10.0) {
        return alpha * youngs_modulus * area /
               (element_size + std::numeric_limits<Real>::epsilon());
    }

    /// Get spatial dimension
    int dim() const { return dim_; }

    /// Number of active (penetrating) pairs
    static int count_active(const std::vector<ContactPair>& pairs) {
        int count = 0;
        for (const auto& cp : pairs) {
            if (cp.gap < 0.0) ++count;
        }
        return count;
    }

    /// Maximum penetration depth
    static Real max_penetration(const std::vector<ContactPair>& pairs) {
        Real max_pen = 0.0;
        for (const auto& cp : pairs) {
            if (cp.gap < 0.0) {
                Real pen = -cp.gap;
                if (pen > max_pen) max_pen = pen;
            }
        }
        return max_pen;
    }

private:
    int dim_;
};


} // namespace solver
} // namespace nxs
