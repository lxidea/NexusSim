#pragma once

/**
 * @file solver_wave30.hpp
 * @brief Wave 30: Advanced iterative solvers and preconditioners
 *
 * Components:
 * 1. LineSearch            - Armijo backtracking with Wolfe conditions
 * 2. LBFGSSolver          - Limited-Memory BFGS quasi-Newton optimizer (m=10)
 * 3. GMRESSolver           - Restarted GMRES(30) for non-symmetric linear systems
 * 4. Preconditioners       - Jacobi (diagonal) and ILUT (incomplete LU with threshold)
 *
 * References:
 * - Nocedal & Wright (2006) "Numerical Optimization", 2nd edition
 * - Saad & Schultz (1986) "GMRES: A Generalized Minimal Residual Algorithm"
 * - Saad (1994) "ILUT: A Dual Threshold Incomplete LU Factorization"
 * - Liu (1989) "Efficient Implementation of Incomplete Factorizations"
 */

#include <nexussim/core/types.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>
#include <cstring>
#include <cassert>

namespace nxs {
namespace solver {

using Real = nxs::Real;

// ============================================================================
// Utility: dot product, norm, axpy for dense vectors
// ============================================================================

namespace detail {

inline Real dot(const Real* a, const Real* b, int n) {
    Real s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline Real norm2(const Real* v, int n) {
    return std::sqrt(dot(v, v, n));
}

/// y = y + alpha * x
inline void axpy(Real alpha, const Real* x, Real* y, int n) {
    for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}

/// y = x
inline void copy(const Real* x, Real* y, int n) {
    std::memcpy(y, x, static_cast<size_t>(n) * sizeof(Real));
}

/// y = alpha * x
inline void scale_copy(Real alpha, const Real* x, Real* y, int n) {
    for (int i = 0; i < n; ++i) y[i] = alpha * x[i];
}

/// y *= alpha
inline void scale(Real alpha, Real* y, int n) {
    for (int i = 0; i < n; ++i) y[i] *= alpha;
}

/// zero vector
inline void zero(Real* y, int n) {
    std::memset(y, 0, static_cast<size_t>(n) * sizeof(Real));
}

} // namespace detail


// ============================================================================
// 1. LineSearch - Armijo Backtracking with Wolfe Conditions
// ============================================================================

/**
 * @brief Line search along a descent direction using Armijo/Wolfe conditions
 *
 * Given a current point x and descent direction d, finds step length alpha
 * such that:
 *   - Armijo (sufficient decrease): f(x + alpha*d) <= f(x) + c1*alpha*grad'*d
 *   - Wolfe (curvature):  |grad_new' * d| <= c2 * |grad' * d|
 *
 * Backtracking strategy: start with alpha=1, multiply by rho until Armijo holds.
 */
class LineSearch {
public:
    /**
     * @brief Construct a line search with given parameters
     * @param c1 Sufficient decrease parameter (Armijo), typically 1e-4
     * @param c2 Curvature condition parameter (Wolfe), typically 0.9
     * @param rho Backtracking reduction factor, typically 0.5
     * @param max_iter Maximum backtracking iterations
     */
    LineSearch(Real c1 = 1e-4, Real c2 = 0.9, Real rho = 0.5, int max_iter = 20)
        : c1_(c1), c2_(c2), rho_(rho), max_iter_(max_iter),
          iterations_(0), armijo_satisfied_(false), wolfe_satisfied_(false)
    {}

    /**
     * @brief Perform Armijo backtracking line search
     *
     * Starting from alpha_init, repeatedly reduce alpha by factor rho until
     * the Armijo sufficient decrease condition is satisfied or max iterations
     * are reached.
     *
     * @param f_func  Evaluates f(x + alpha * d) for given alpha
     * @param grad_dot_d  Directional derivative at current point: nabla f(x) . d
     * @param f0  Function value at current point: f(x)
     * @param alpha_init  Initial step length (default 1.0)
     * @return Step length alpha satisfying Armijo, or last tried alpha
     */
    Real backtrack(std::function<Real(Real)> f_func,
                   Real grad_dot_d, Real f0,
                   Real alpha_init = 1.0)
    {
        iterations_ = 0;
        armijo_satisfied_ = false;
        wolfe_satisfied_ = false;

        // If gradient dot direction is non-negative, not a descent direction.
        // Return 0 to indicate no useful step.
        if (grad_dot_d >= 0.0) {
            return 0.0;
        }

        Real alpha = alpha_init;

        for (int k = 0; k < max_iter_; ++k) {
            iterations_ = k + 1;
            Real f_new = f_func(alpha);

            if (armijo_check(f_new, f0, alpha, grad_dot_d)) {
                armijo_satisfied_ = true;
                return alpha;
            }

            alpha *= rho_;
        }

        // Return the last alpha tried even if Armijo not satisfied
        return alpha;
    }

    /**
     * @brief Armijo sufficient decrease check
     *
     * Returns true if f_new <= f0 + c1 * alpha * grad_dot_d
     * Since grad_dot_d < 0 for a descent direction, the RHS < f0.
     */
    bool armijo_check(Real f_new, Real f0, Real alpha, Real grad_dot_d) const {
        return f_new <= f0 + c1_ * alpha * grad_dot_d;
    }

    /**
     * @brief Wolfe curvature condition check
     *
     * Returns true if |grad_new_dot_d| <= c2 * |grad_dot_d|
     * This ensures the slope has been sufficiently reduced.
     */
    bool wolfe_check(Real grad_new_dot_d, Real grad_dot_d) const {
        return std::abs(grad_new_dot_d) <= c2_ * std::abs(grad_dot_d);
    }

    /// Number of backtracking iterations in last call
    int iterations() const { return iterations_; }

    /// Whether Armijo was satisfied in last backtrack call
    bool armijo_satisfied() const { return armijo_satisfied_; }

    /// Whether Wolfe was satisfied in last backtrack call
    bool wolfe_satisfied() const { return wolfe_satisfied_; }

    // Parameter accessors
    Real c1() const { return c1_; }
    Real c2() const { return c2_; }
    Real rho() const { return rho_; }
    int max_iter() const { return max_iter_; }

    void set_c1(Real c1) { c1_ = c1; }
    void set_c2(Real c2) { c2_ = c2; }
    void set_rho(Real rho) { rho_ = rho; }
    void set_max_iter(int m) { max_iter_ = m; }

private:
    Real c1_;         ///< Sufficient decrease parameter
    Real c2_;         ///< Curvature condition parameter
    Real rho_;        ///< Backtracking reduction factor
    int max_iter_;    ///< Maximum backtracking iterations
    int iterations_;  ///< Iterations used in last call
    bool armijo_satisfied_;
    bool wolfe_satisfied_;
};


// ============================================================================
// 2. LBFGSSolver - Limited-Memory BFGS Quasi-Newton Optimizer
// ============================================================================

/**
 * @brief Limited-Memory BFGS (L-BFGS) optimizer
 *
 * Approximates the inverse Hessian using the most recent m correction pairs
 * (s_k, y_k). The search direction is computed via the two-loop recursion
 * of Nocedal (1980). Line search is performed using Armijo backtracking.
 *
 * Memory cost: O(m*n) where m is the history size and n is the problem dimension.
 *
 * Algorithm:
 *   1. Compute gradient g_k = nabla f(x_k)
 *   2. Compute search direction d_k = -H_k * g_k via two-loop recursion
 *   3. Line search: alpha_k = argmin f(x_k + alpha * d_k)
 *   4. Update: x_{k+1} = x_k + alpha_k * d_k
 *   5. Store correction pair: s_k = x_{k+1} - x_k, y_k = g_{k+1} - g_k
 *   6. If ||g_{k+1}|| < tol, stop
 */
class LBFGSSolver {
public:
    /**
     * @brief Construct L-BFGS solver
     * @param m Number of correction pairs to store (default 10)
     * @param max_iter Maximum iterations (default 100)
     * @param tol Gradient norm tolerance for convergence (default 1e-8)
     */
    LBFGSSolver(int m = 10, int max_iter = 100, Real tol = 1e-8)
        : m_(m), max_iter_(max_iter), tol_(tol),
          converged_(false), iterations_(0), n_(0), num_stored_(0), oldest_(0)
    {}

    /**
     * @brief Solve the unconstrained minimization problem
     *
     * Minimizes f(x) starting from x0 using L-BFGS with Armijo line search.
     *
     * @param residual_func  Evaluates f(x): takes const Real*, returns Real
     * @param gradient_func  Computes nabla f(x): takes (const Real* x, Real* grad)
     * @param x0  Initial guess (n values). Modified in place with the solution.
     * @param n   Problem dimension
     * @return Solution vector (same as modified x0)
     */
    std::vector<Real> solve(
        std::function<Real(const Real*)> residual_func,
        std::function<void(const Real*, Real*)> gradient_func,
        const Real* x0, int n)
    {
        n_ = n;
        converged_ = false;
        iterations_ = 0;
        num_stored_ = 0;
        oldest_ = 0;

        // Allocate history storage
        s_history_.assign(m_, std::vector<Real>(n, 0.0));
        y_history_.assign(m_, std::vector<Real>(n, 0.0));
        rho_history_.assign(m_, 0.0);

        // Current point and gradient
        std::vector<Real> x(x0, x0 + n);
        std::vector<Real> grad(n), grad_new(n);
        std::vector<Real> direction(n);
        std::vector<Real> s(n), y(n);

        gradient_func(x.data(), grad.data());

        // Check if already at minimum
        Real grad_norm = detail::norm2(grad.data(), n);
        if (grad_norm < tol_) {
            converged_ = true;
            return x;
        }

        LineSearch ls;

        for (int k = 0; k < max_iter_; ++k) {
            iterations_ = k + 1;

            // Compute search direction: d = -H_k * g
            two_loop_recursion(grad.data(), direction.data(), n);

            // Negate: direction = -H*g
            for (int i = 0; i < n; ++i) direction[i] = -direction[i];

            // Directional derivative
            Real grad_dot_d = detail::dot(grad.data(), direction.data(), n);

            // If not a descent direction, reset to steepest descent
            if (grad_dot_d >= 0.0) {
                for (int i = 0; i < n; ++i) direction[i] = -grad[i];
                grad_dot_d = -detail::dot(grad.data(), grad.data(), n);
            }

            Real f0 = residual_func(x.data());

            // Line search
            std::vector<Real> x_trial(n);
            auto f_func = [&](Real alpha) -> Real {
                for (int i = 0; i < n; ++i) x_trial[i] = x[i] + alpha * direction[i];
                return residual_func(x_trial.data());
            };

            Real alpha = ls.backtrack(f_func, grad_dot_d, f0, 1.0);

            if (alpha == 0.0) {
                // Line search failed, try small step
                alpha = 1e-6;
            }

            // Update x
            for (int i = 0; i < n; ++i) {
                s[i] = alpha * direction[i];
                x[i] += s[i];
            }

            // New gradient
            gradient_func(x.data(), grad_new.data());

            // y = grad_new - grad
            for (int i = 0; i < n; ++i) {
                y[i] = grad_new[i] - grad[i];
            }

            // Update history
            update_history(s.data(), y.data(), n);

            // Check convergence
            grad_norm = detail::norm2(grad_new.data(), n);
            if (grad_norm < tol_) {
                converged_ = true;
                detail::copy(grad_new.data(), grad.data(), n);
                break;
            }

            detail::copy(grad_new.data(), grad.data(), n);
        }

        return x;
    }

    /**
     * @brief Two-loop recursion to compute H_k * grad
     *
     * Computes the product of the L-BFGS approximate inverse Hessian with
     * the gradient vector, using stored correction pairs.
     *
     * Result is stored in `result`. Note: the caller should negate this
     * to get the search direction d = -H*g.
     *
     * @param grad Input gradient vector
     * @param result Output: H_k * grad
     * @param n Problem dimension
     */
    void two_loop_recursion(const Real* grad, Real* result, int n) {
        std::vector<Real> q(grad, grad + n);
        std::vector<Real> alpha_vals(num_stored_);

        // First loop: from newest to oldest
        for (int j = num_stored_ - 1; j >= 0; --j) {
            int idx = history_index(j);
            Real rho = rho_history_[idx];
            Real a = rho * detail::dot(s_history_[idx].data(), q.data(), n);
            alpha_vals[j] = a;
            detail::axpy(-a, y_history_[idx].data(), q.data(), n);
        }

        // Initial Hessian approximation: H0 = gamma * I
        Real gamma = 1.0;
        if (num_stored_ > 0) {
            int newest = history_index(num_stored_ - 1);
            Real sy = detail::dot(s_history_[newest].data(), y_history_[newest].data(), n);
            Real yy = detail::dot(y_history_[newest].data(), y_history_[newest].data(), n);
            if (yy > 0.0) {
                gamma = sy / yy;
            }
        }

        // r = gamma * q
        detail::scale_copy(gamma, q.data(), result, n);

        // Second loop: from oldest to newest
        for (int j = 0; j < num_stored_; ++j) {
            int idx = history_index(j);
            Real rho = rho_history_[idx];
            Real beta = rho * detail::dot(y_history_[idx].data(), result, n);
            detail::axpy(alpha_vals[j] - beta, s_history_[idx].data(), result, n);
        }
    }

    /**
     * @brief Store a new correction pair (s, y) in circular buffer
     *
     * @param s  Step vector: x_{k+1} - x_k
     * @param y  Gradient difference: grad_{k+1} - grad_k
     * @param n  Problem dimension
     */
    void update_history(const Real* s, const Real* y, int n) {
        Real sy = detail::dot(s, y, n);
        if (sy <= 1e-30) return; // Skip if curvature condition not met

        int slot;
        if (num_stored_ < m_) {
            slot = num_stored_;
            num_stored_++;
        } else {
            slot = oldest_;
            oldest_ = (oldest_ + 1) % m_;
        }

        detail::copy(s, s_history_[slot].data(), n);
        detail::copy(y, y_history_[slot].data(), n);
        rho_history_[slot] = 1.0 / sy;
    }

    /// Whether the solver converged
    bool converged() const { return converged_; }

    /// Number of iterations performed
    int iterations() const { return iterations_; }

    /// History size parameter
    int history_size() const { return m_; }

    /// Number of stored correction pairs
    int num_stored() const { return num_stored_; }

    /// Tolerance
    Real tolerance() const { return tol_; }

    void set_tolerance(Real tol) { tol_ = tol; }
    void set_max_iter(int m) { max_iter_ = m; }

private:
    /// Map logical index (0=oldest, num_stored-1=newest) to buffer slot
    int history_index(int j) const {
        if (num_stored_ < m_) {
            return j;
        }
        return (oldest_ + j) % m_;
    }

    int m_;           ///< Max correction pairs
    int max_iter_;    ///< Max solver iterations
    Real tol_;        ///< Gradient norm tolerance
    bool converged_;
    int iterations_;
    int n_;           ///< Problem dimension

    // Circular buffer for correction pairs
    std::vector<std::vector<Real>> s_history_;
    std::vector<std::vector<Real>> y_history_;
    std::vector<Real> rho_history_;
    int num_stored_;
    int oldest_;
};


// ============================================================================
// 3. GMRESSolver - Restarted GMRES for Non-Symmetric Linear Systems
// ============================================================================

/**
 * @brief Restarted GMRES(m) solver for non-symmetric linear systems Ax = b
 *
 * Uses the Arnoldi process to build an orthonormal Krylov basis, then
 * solves the projected least-squares problem using Givens rotations.
 * Restarts after m iterations to limit memory growth.
 *
 * Optionally supports left preconditioning: solve M^{-1}Ax = M^{-1}b.
 *
 * Algorithm:
 *   1. r0 = b - A*x0, beta = ||r0||, v1 = r0/beta
 *   2. Arnoldi: for j=1..m, compute w = A*v_j, orthogonalize, store h_{i,j}
 *   3. Apply Givens rotations to reduce H to upper triangular
 *   4. Solve triangular system, form solution x = x0 + V*y
 *   5. If ||r|| > tol, restart
 */
class GMRESSolver {
public:
    /**
     * @brief Construct GMRES solver
     * @param restart Restart dimension (default 30)
     * @param max_restarts Maximum number of restarts (default 10)
     * @param tol Convergence tolerance on relative residual (default 1e-8)
     */
    GMRESSolver(int restart = 30, int max_restarts = 10, Real tol = 1e-8)
        : restart_(restart), max_restarts_(max_restarts), tol_(tol),
          converged_(false), iterations_(0), residual_norm_(0.0),
          has_preconditioner_(false)
    {}

    /**
     * @brief Set a left preconditioner M^{-1}
     *
     * The preconditioner function applies M^{-1} to a vector:
     *   precond(x, y) computes y = M^{-1} * x
     *
     * @param precond_func Function that applies M^{-1}
     */
    void set_preconditioner(std::function<void(const Real*, Real*)> precond_func) {
        precond_func_ = precond_func;
        has_preconditioner_ = true;
    }

    /**
     * @brief Solve Ax = b using restarted GMRES
     *
     * @param matvec_func  Computes A*x: takes (const Real* x, Real* Ax)
     * @param rhs  Right-hand side vector b (size n)
     * @param x0   Initial guess, overwritten with solution (size n)
     * @param n    System dimension
     */
    void solve(std::function<void(const Real*, Real*)> matvec_func,
               const Real* rhs, Real* x0, int n)
    {
        converged_ = false;
        iterations_ = 0;

        int m = std::min(restart_, n);

        // Working vectors
        std::vector<Real> r(n), w(n), temp(n);

        // Arnoldi basis V: (m+1) vectors of size n
        std::vector<std::vector<Real>> V(m + 1, std::vector<Real>(n, 0.0));

        // Upper Hessenberg matrix H: (m+1) x m
        std::vector<std::vector<Real>> H(m + 1, std::vector<Real>(m, 0.0));

        // Givens rotation parameters
        std::vector<Real> cs(m, 0.0), sn(m, 0.0);

        // Right-hand side for least squares
        std::vector<Real> g(m + 1, 0.0);

        // Compute initial residual norm for relative tolerance
        matvec_func(x0, r.data());
        for (int i = 0; i < n; ++i) r[i] = rhs[i] - r[i];

        if (has_preconditioner_) {
            detail::copy(r.data(), temp.data(), n);
            precond_func_(temp.data(), r.data());
        }

        Real beta = detail::norm2(r.data(), n);
        Real rhs_norm = detail::norm2(rhs, n);
        Real tol_abs = tol_ * std::max(rhs_norm, static_cast<Real>(1e-30));

        residual_norm_ = beta;

        if (beta <= tol_abs) {
            converged_ = true;
            return;
        }

        for (int restart = 0; restart <= max_restarts_; ++restart) {
            // r = b - A*x
            matvec_func(x0, r.data());
            for (int i = 0; i < n; ++i) r[i] = rhs[i] - r[i];

            if (has_preconditioner_) {
                detail::copy(r.data(), temp.data(), n);
                precond_func_(temp.data(), r.data());
            }

            beta = detail::norm2(r.data(), n);
            residual_norm_ = beta;

            if (beta <= tol_abs) {
                converged_ = true;
                return;
            }

            // v1 = r / beta
            detail::scale_copy(1.0 / beta, r.data(), V[0].data(), n);

            // Initialize RHS of least squares
            for (int i = 0; i <= m; ++i) g[i] = 0.0;
            g[0] = beta;

            // Reset H
            for (int i = 0; i <= m; ++i)
                for (int j = 0; j < m; ++j)
                    H[i][j] = 0.0;

            int j_conv = m; // will be set if converged early

            for (int j = 0; j < m; ++j) {
                iterations_++;

                // w = A * v_j (or M^{-1} * A * v_j)
                matvec_func(V[j].data(), w.data());
                if (has_preconditioner_) {
                    detail::copy(w.data(), temp.data(), n);
                    precond_func_(temp.data(), w.data());
                }

                // Modified Gram-Schmidt orthogonalization
                for (int i = 0; i <= j; ++i) {
                    H[i][j] = detail::dot(w.data(), V[i].data(), n);
                    detail::axpy(-H[i][j], V[i].data(), w.data(), n);
                }

                H[j + 1][j] = detail::norm2(w.data(), n);

                if (std::abs(H[j + 1][j]) < 1e-30) {
                    // Lucky breakdown: solution is in current subspace
                    j_conv = j + 1;

                    // Apply previous Givens rotations to column j
                    for (int i = 0; i < j; ++i) {
                        apply_givens(cs[i], sn[i], H[i][j], H[i + 1][j]);
                    }

                    // Compute Givens rotation for this column
                    compute_givens(H[j][j], H[j + 1][j], cs[j], sn[j]);
                    apply_givens(cs[j], sn[j], H[j][j], H[j + 1][j]);
                    apply_givens(cs[j], sn[j], g[j], g[j + 1]);

                    converged_ = true;
                    break;
                }

                detail::scale_copy(1.0 / H[j + 1][j], w.data(), V[j + 1].data(), n);

                // Apply previous Givens rotations to new column of H
                for (int i = 0; i < j; ++i) {
                    apply_givens(cs[i], sn[i], H[i][j], H[i + 1][j]);
                }

                // Compute new Givens rotation
                compute_givens(H[j][j], H[j + 1][j], cs[j], sn[j]);
                apply_givens(cs[j], sn[j], H[j][j], H[j + 1][j]);
                apply_givens(cs[j], sn[j], g[j], g[j + 1]);

                residual_norm_ = std::abs(g[j + 1]);

                if (residual_norm_ <= tol_abs) {
                    j_conv = j + 1;
                    converged_ = true;
                    break;
                }
            }

            // Solve upper triangular system H*y = g
            int k = converged_ ? j_conv : m;
            std::vector<Real> y(k);
            for (int i = k - 1; i >= 0; --i) {
                y[i] = g[i];
                for (int j2 = i + 1; j2 < k; ++j2) {
                    y[i] -= H[i][j2] * y[j2];
                }
                if (std::abs(H[i][i]) > 1e-30) {
                    y[i] /= H[i][i];
                }
            }

            // Update solution: x = x0 + V * y
            for (int i = 0; i < k; ++i) {
                detail::axpy(y[i], V[i].data(), x0, n);
            }

            if (converged_) return;
        }
    }

    /// Residual norm after solve
    Real residual_norm() const { return residual_norm_; }

    /// Total iterations across all restarts
    int iterations() const { return iterations_; }

    /// Whether solver converged
    bool converged() const { return converged_; }

    /// Restart dimension
    int restart() const { return restart_; }

    /// Tolerance
    Real tolerance() const { return tol_; }

    void set_tolerance(Real tol) { tol_ = tol; }
    void set_max_restarts(int m) { max_restarts_ = m; }

private:
    /// Compute Givens rotation to zero out b in [a; b]
    static void compute_givens(Real a, Real b, Real& cs, Real& sn) {
        if (std::abs(b) < 1e-30) {
            cs = 1.0;
            sn = 0.0;
        } else if (std::abs(b) > std::abs(a)) {
            Real t = a / b;
            sn = 1.0 / std::sqrt(1.0 + t * t);
            cs = t * sn;
        } else {
            Real t = b / a;
            cs = 1.0 / std::sqrt(1.0 + t * t);
            sn = t * cs;
        }
    }

    /// Apply Givens rotation to (h1, h2)
    static void apply_givens(Real cs, Real sn, Real& h1, Real& h2) {
        Real t1 = cs * h1 + sn * h2;
        Real t2 = -sn * h1 + cs * h2;
        h1 = t1;
        h2 = t2;
    }

    int restart_;
    int max_restarts_;
    Real tol_;
    bool converged_;
    int iterations_;
    Real residual_norm_;
    bool has_preconditioner_;
    std::function<void(const Real*, Real*)> precond_func_;
};


// ============================================================================
// 4a. SparseMatrix - Compressed Sparse Row format
// ============================================================================

/**
 * @brief Compressed Sparse Row (CSR) sparse matrix
 *
 * Stores matrix entries in three arrays:
 *   values[]:      non-zero values, row by row
 *   col_indices[]: column index for each value
 *   row_ptr[]:     index into values/col_indices for start of each row
 *                  row_ptr[n] = total number of non-zeros
 */
struct SparseMatrix {
    std::vector<Real> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;  ///< Size = n+1
    int n;                     ///< Matrix dimension (n x n)

    SparseMatrix() : n(0) {}

    SparseMatrix(int dim) : n(dim) {
        row_ptr.assign(dim + 1, 0);
    }

    /// Number of non-zeros
    int nnz() const { return static_cast<int>(values.size()); }

    /// Matrix-vector product: y = A * x
    void matvec(const Real* x, Real* y) const {
        for (int i = 0; i < n; ++i) {
            Real sum = 0.0;
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                sum += values[j] * x[col_indices[j]];
            }
            y[i] = sum;
        }
    }

    /// Get diagonal entry for row i. Returns 0 if not present.
    Real diagonal(int i) const {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            if (col_indices[j] == i) return values[j];
        }
        return 0.0;
    }

    /**
     * @brief Build CSR from dense matrix (row-major)
     * @param dense  Dense matrix in row-major order (n x n)
     * @param dim    Matrix dimension
     * @param drop_tol  Entries with |value| < drop_tol are dropped (0 = keep all)
     */
    static SparseMatrix from_dense(const Real* dense, int dim, Real drop_tol = 0.0) {
        SparseMatrix A(dim);
        A.row_ptr[0] = 0;
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                Real val = dense[i * dim + j];
                if (std::abs(val) > drop_tol || i == j) {
                    A.values.push_back(val);
                    A.col_indices.push_back(j);
                }
            }
            A.row_ptr[i + 1] = static_cast<int>(A.values.size());
        }
        return A;
    }

    /**
     * @brief Build a diagonal CSR matrix
     * @param diag  Diagonal values (size dim)
     * @param dim   Matrix dimension
     */
    static SparseMatrix diagonal_matrix(const Real* diag, int dim) {
        SparseMatrix A(dim);
        A.values.resize(dim);
        A.col_indices.resize(dim);
        A.row_ptr.resize(dim + 1);
        for (int i = 0; i < dim; ++i) {
            A.values[i] = diag[i];
            A.col_indices[i] = i;
            A.row_ptr[i] = i;
        }
        A.row_ptr[dim] = dim;
        return A;
    }

    /**
     * @brief Build a tridiagonal CSR matrix (-1, d, -1)
     * @param d     Diagonal value
     * @param dim   Matrix dimension
     */
    static SparseMatrix tridiagonal(Real d, int dim) {
        SparseMatrix A(dim);
        A.row_ptr[0] = 0;
        for (int i = 0; i < dim; ++i) {
            if (i > 0) {
                A.values.push_back(-1.0);
                A.col_indices.push_back(i - 1);
            }
            A.values.push_back(d);
            A.col_indices.push_back(i);
            if (i < dim - 1) {
                A.values.push_back(-1.0);
                A.col_indices.push_back(i + 1);
            }
            A.row_ptr[i + 1] = static_cast<int>(A.values.size());
        }
        return A;
    }
};


// ============================================================================
// 4b. JacobiPreconditioner - Diagonal (Jacobi) Preconditioner
// ============================================================================

/**
 * @brief Jacobi (diagonal) preconditioner
 *
 * M = diag(A), so M^{-1} * x = x_i / A_{ii}.
 * Simple but effective for diagonally dominant systems.
 */
class JacobiPreconditioner {
public:
    /**
     * @brief Construct from diagonal values
     * @param diag_values  Diagonal entries A_{ii} (size n)
     * @param n  System dimension
     */
    JacobiPreconditioner(const Real* diag_values, int n) : n_(n) {
        inv_diag_.resize(n);
        for (int i = 0; i < n; ++i) {
            if (std::abs(diag_values[i]) > 1e-30) {
                inv_diag_[i] = 1.0 / diag_values[i];
            } else {
                inv_diag_[i] = 1.0; // fallback for zero diagonal
            }
        }
    }

    /**
     * @brief Construct from sparse matrix (extracts diagonal)
     * @param A  Sparse matrix in CSR format
     */
    explicit JacobiPreconditioner(const SparseMatrix& A) : n_(A.n) {
        inv_diag_.resize(n_);
        for (int i = 0; i < n_; ++i) {
            Real d = A.diagonal(i);
            if (std::abs(d) > 1e-30) {
                inv_diag_[i] = 1.0 / d;
            } else {
                inv_diag_[i] = 1.0;
            }
        }
    }

    /**
     * @brief Apply preconditioner: y = M^{-1} * x
     * @param x  Input vector (size n)
     * @param y  Output vector (size n)
     * @param n  Vector size (ignored, uses stored n_)
     */
    void apply(const Real* x, Real* y, int /*n*/ = 0) const {
        for (int i = 0; i < n_; ++i) {
            y[i] = inv_diag_[i] * x[i];
        }
    }

    /// System dimension
    int size() const { return n_; }

    /// Access inverse diagonal
    const std::vector<Real>& inv_diagonal() const { return inv_diag_; }

private:
    int n_;
    std::vector<Real> inv_diag_;
};


// ============================================================================
// 4c. ILUTPreconditioner - Incomplete LU with Threshold
// ============================================================================

/**
 * @brief Incomplete LU factorization with dual threshold (ILUT)
 *
 * Computes approximate L and U factors of A such that A ~= L*U, where:
 *   - Entries with magnitude below tau * ||row|| are dropped
 *   - At most lfil entries are kept per row in L and U
 *
 * This provides a sparser but less accurate factorization that serves as
 * a good preconditioner for iterative methods (CG, GMRES).
 *
 * Storage: L is unit lower triangular (diagonal=1 not stored),
 *          U is upper triangular.
 * Both stored in CSR format.
 */
class ILUTPreconditioner {
public:
    /**
     * @brief Construct ILUT preconditioner
     * @param A    Sparse matrix in CSR format
     * @param tau  Drop tolerance (entries < tau * ||row|| are dropped)
     * @param lfil Maximum fill per row in L and U
     */
    ILUTPreconditioner(const SparseMatrix& A, Real tau = 1e-3, int lfil = 10)
        : A_(A), tau_(tau), lfil_(lfil), n_(A.n), factorized_(false),
          L_(A.n), U_(A.n)
    {}

    /**
     * @brief Perform the ILUT factorization
     *
     * For each row i of A:
     *   1. Copy row i into a dense work vector w
     *   2. For k = 0..i-1 where w[k] != 0:
     *      w[k] = w[k] / U[k][k]
     *      For j in row k of U (j > k): w[j] -= w[k] * U[k][j]
     *   3. Drop small entries: |w[j]| < tau * ||w||
     *   4. Keep at most lfil entries in L (j<i) and lfil entries in U (j>=i)
     *
     * @return true if factorization succeeded (no zero pivots)
     */
    bool factorize() {
        factorized_ = false;

        // Initialize L and U
        L_ = SparseMatrix(n_);
        U_ = SparseMatrix(n_);
        L_.row_ptr[0] = 0;
        U_.row_ptr[0] = 0;

        // Dense work row and index tracking
        std::vector<Real> w(n_, 0.0);
        std::vector<int> jw(n_, -1); // maps column to position in work set
        std::vector<int> work_cols;  // columns with non-zero entries

        for (int i = 0; i < n_; ++i) {
            // Clear work arrays
            work_cols.clear();
            for (int j = 0; j < n_; ++j) {
                w[j] = 0.0;
                jw[j] = -1;
            }

            // Copy row i of A into w
            for (int p = A_.row_ptr[i]; p < A_.row_ptr[i + 1]; ++p) {
                int col = A_.col_indices[p];
                w[col] = A_.values[p];
                jw[col] = static_cast<int>(work_cols.size());
                work_cols.push_back(col);
            }

            // Elimination: for each column k < i with w[k] != 0
            // Process columns in order
            std::vector<int> lower_cols;
            for (int c : work_cols) {
                if (c < i) lower_cols.push_back(c);
            }
            std::sort(lower_cols.begin(), lower_cols.end());

            for (int k : lower_cols) {
                if (std::abs(w[k]) < 1e-30) continue;

                // Get U[k][k] diagonal
                Real u_kk = 0.0;
                for (int p = U_.row_ptr[k]; p < U_.row_ptr[k + 1]; ++p) {
                    if (U_.col_indices[p] == k) {
                        u_kk = U_.values[p];
                        break;
                    }
                }
                if (std::abs(u_kk) < 1e-30) continue;

                Real factor = w[k] / u_kk;
                w[k] = factor;

                // w[j] -= factor * U[k][j] for j > k
                for (int p = U_.row_ptr[k]; p < U_.row_ptr[k + 1]; ++p) {
                    int j = U_.col_indices[p];
                    if (j <= k) continue;
                    w[j] -= factor * U_.values[p];
                    if (jw[j] < 0) {
                        jw[j] = static_cast<int>(work_cols.size());
                        work_cols.push_back(j);
                    }
                }
            }

            // Compute row norm for dropping
            Real row_norm = 0.0;
            for (int c : work_cols) {
                row_norm += w[c] * w[c];
            }
            row_norm = std::sqrt(row_norm);
            Real drop_tol = tau_ * row_norm;

            // Separate L and U entries, applying threshold
            struct Entry {
                int col;
                Real val;
                bool operator<(const Entry& o) const {
                    return std::abs(val) > std::abs(o.val); // sort by magnitude desc
                }
            };

            std::vector<Entry> l_entries, u_entries;

            for (int c : work_cols) {
                Real val = w[c];
                if (c < i) {
                    if (std::abs(val) >= drop_tol) {
                        l_entries.push_back({c, val});
                    }
                } else if (c == i) {
                    // Always keep diagonal
                    u_entries.push_back({c, val});
                } else {
                    if (std::abs(val) >= drop_tol) {
                        u_entries.push_back({c, val});
                    }
                }
            }

            // Sort by magnitude and keep at most lfil entries
            std::sort(l_entries.begin(), l_entries.end());
            std::sort(u_entries.begin(), u_entries.end());

            if (static_cast<int>(l_entries.size()) > lfil_) {
                l_entries.resize(lfil_);
            }
            // For U, keep diagonal + at most lfil off-diagonal
            if (static_cast<int>(u_entries.size()) > lfil_ + 1) {
                // Keep diagonal (first after sort might not be it), so be careful
                // Re-sort by column for proper storage
                u_entries.resize(lfil_ + 1);
            }

            // Sort by column index for CSR storage
            std::sort(l_entries.begin(), l_entries.end(),
                      [](const Entry& a, const Entry& b) { return a.col < b.col; });
            std::sort(u_entries.begin(), u_entries.end(),
                      [](const Entry& a, const Entry& b) { return a.col < b.col; });

            // Ensure diagonal is present in U
            bool has_diag = false;
            for (auto& e : u_entries) {
                if (e.col == i) { has_diag = true; break; }
            }
            if (!has_diag) {
                // Zero pivot - factorization may fail
                u_entries.push_back({i, 1e-10}); // small regularization
                std::sort(u_entries.begin(), u_entries.end(),
                          [](const Entry& a, const Entry& b) { return a.col < b.col; });
            }

            // Store L row
            for (auto& e : l_entries) {
                L_.values.push_back(e.val);
                L_.col_indices.push_back(e.col);
            }
            L_.row_ptr[i + 1] = static_cast<int>(L_.values.size());

            // Store U row
            for (auto& e : u_entries) {
                U_.values.push_back(e.val);
                U_.col_indices.push_back(e.col);
            }
            U_.row_ptr[i + 1] = static_cast<int>(U_.values.size());
        }

        factorized_ = true;
        return true;
    }

    /**
     * @brief Apply preconditioner: y = (LU)^{-1} * x
     *
     * Solves L*z = x (forward substitution), then U*y = z (backward substitution).
     *
     * @param x  Input vector (size n)
     * @param y  Output vector (size n)
     */
    void apply(const Real* x, Real* y) const {
        if (!factorized_) {
            // If not factorized, just copy (identity preconditioner)
            detail::copy(x, y, n_);
            return;
        }

        std::vector<Real> z(n_);

        // Forward substitution: L * z = x (L is unit lower triangular)
        forward_substitution(x, z.data());

        // Backward substitution: U * y = z
        backward_substitution(z.data(), y);
    }

    /**
     * @brief Forward substitution: solve L * z = x
     *
     * L is unit lower triangular (implicit diagonal of 1).
     * z[i] = x[i] - sum_{j<i} L[i][j] * z[j]
     */
    void forward_substitution(const Real* x, Real* z) const {
        for (int i = 0; i < n_; ++i) {
            Real sum = x[i];
            for (int p = L_.row_ptr[i]; p < L_.row_ptr[i + 1]; ++p) {
                sum -= L_.values[p] * z[L_.col_indices[p]];
            }
            z[i] = sum;
        }
    }

    /**
     * @brief Backward substitution: solve U * y = z
     *
     * U is upper triangular.
     * y[i] = (z[i] - sum_{j>i} U[i][j] * y[j]) / U[i][i]
     */
    void backward_substitution(const Real* z, Real* y) const {
        for (int i = n_ - 1; i >= 0; --i) {
            Real sum = z[i];
            Real diag = 1.0;
            for (int p = U_.row_ptr[i]; p < U_.row_ptr[i + 1]; ++p) {
                int col = U_.col_indices[p];
                if (col == i) {
                    diag = U_.values[p];
                } else if (col > i) {
                    sum -= U_.values[p] * y[col];
                }
            }
            if (std::abs(diag) > 1e-30) {
                y[i] = sum / diag;
            } else {
                y[i] = sum;
            }
        }
    }

    /// Whether factorization has been performed
    bool factorized() const { return factorized_; }

    /// Access L factor
    const SparseMatrix& L() const { return L_; }

    /// Access U factor
    const SparseMatrix& U() const { return U_; }

    /// Drop tolerance
    Real tau() const { return tau_; }

    /// Fill limit
    int lfil() const { return lfil_; }

    /// Matrix dimension
    int size() const { return n_; }

private:
    SparseMatrix A_;
    Real tau_;
    int lfil_;
    int n_;
    bool factorized_;
    SparseMatrix L_;
    SparseMatrix U_;
};


} // namespace solver
} // namespace nxs
