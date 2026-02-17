#pragma once

/**
 * @file implicit_solver.hpp
 * @brief Implicit time integration and static solver
 *
 * Implements:
 * - Sparse matrix storage (CSR format)
 * - Newton-Raphson nonlinear solver
 * - Newmark-β implicit time integration
 * - Linear solvers (Direct LU, Conjugate Gradient)
 * - Static structural analysis
 *
 * For systems: M*a + C*v + K*u = F_ext
 * Or static:   K*u = F_ext
 */

#include <nexussim/core/core.hpp>
#include <Kokkos_Core.hpp>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <iostream>
#include <functional>
#include <algorithm>

namespace nxs {
namespace solver {

// ============================================================================
// Sparse Matrix (CSR Format)
// ============================================================================

/**
 * @brief Compressed Sparse Row (CSR) matrix storage
 *
 * Standard CSR format:
 * - values[]: Non-zero values (size = nnz)
 * - col_indices[]: Column index for each value (size = nnz)
 * - row_ptr[]: Start of each row in values (size = nrows + 1)
 */
class SparseMatrix {
public:
    SparseMatrix() = default;

    /**
     * @brief Create from COO (coordinate) format
     */
    void from_coo(size_t nrows, size_t ncols,
                  const std::vector<size_t>& rows,
                  const std::vector<size_t>& cols,
                  const std::vector<Real>& vals) {
        nrows_ = nrows;
        ncols_ = ncols;

        // Count entries per row
        std::vector<size_t> row_count(nrows, 0);
        for (size_t i = 0; i < rows.size(); ++i) {
            row_count[rows[i]]++;
        }

        // Build row pointers
        row_ptr_.resize(nrows + 1);
        row_ptr_[0] = 0;
        for (size_t i = 0; i < nrows; ++i) {
            row_ptr_[i + 1] = row_ptr_[i] + row_count[i];
        }

        // Allocate
        size_t nnz = row_ptr_[nrows];
        col_indices_.resize(nnz);
        values_.resize(nnz);

        // Fill values (reset row_count for insertion)
        std::fill(row_count.begin(), row_count.end(), 0);
        for (size_t i = 0; i < rows.size(); ++i) {
            size_t row = rows[i];
            size_t pos = row_ptr_[row] + row_count[row];
            col_indices_[pos] = cols[i];
            values_[pos] = vals[i];
            row_count[row]++;
        }

        // Sort within each row by column index
        for (size_t row = 0; row < nrows; ++row) {
            size_t start = row_ptr_[row];
            size_t end = row_ptr_[row + 1];

            // Simple bubble sort (rows are typically small)
            for (size_t i = start; i < end; ++i) {
                for (size_t j = i + 1; j < end; ++j) {
                    if (col_indices_[j] < col_indices_[i]) {
                        std::swap(col_indices_[i], col_indices_[j]);
                        std::swap(values_[i], values_[j]);
                    }
                }
            }
        }
    }

    /**
     * @brief Create empty matrix with sparsity pattern
     */
    void create_pattern(size_t nrows, size_t ncols,
                        const std::vector<std::vector<size_t>>& pattern) {
        nrows_ = nrows;
        ncols_ = ncols;

        row_ptr_.resize(nrows + 1);
        row_ptr_[0] = 0;

        for (size_t i = 0; i < nrows; ++i) {
            row_ptr_[i + 1] = row_ptr_[i] + pattern[i].size();
        }

        size_t nnz = row_ptr_[nrows];
        col_indices_.resize(nnz);
        values_.resize(nnz, 0.0);

        for (size_t row = 0; row < nrows; ++row) {
            size_t pos = row_ptr_[row];
            for (size_t col : pattern[row]) {
                col_indices_[pos++] = col;
            }
        }
    }

    /**
     * @brief Set all values to zero (keep pattern)
     */
    void zero() {
        std::fill(values_.begin(), values_.end(), 0.0);
    }

    /**
     * @brief Add value at (row, col) - assumes entry exists
     */
    void add(size_t row, size_t col, Real val) {
        for (size_t i = row_ptr_[row]; i < row_ptr_[row + 1]; ++i) {
            if (col_indices_[i] == col) {
                values_[i] += val;
                return;
            }
        }
        // Entry not found - should not happen if pattern is correct
        NXS_LOG_WARN("SparseMatrix::add - entry ({}, {}) not in pattern", row, col);
    }

    /**
     * @brief Set value at (row, col)
     */
    void set(size_t row, size_t col, Real val) {
        for (size_t i = row_ptr_[row]; i < row_ptr_[row + 1]; ++i) {
            if (col_indices_[i] == col) {
                values_[i] = val;
                return;
            }
        }
    }

    /**
     * @brief Get value at (row, col)
     */
    Real get(size_t row, size_t col) const {
        for (size_t i = row_ptr_[row]; i < row_ptr_[row + 1]; ++i) {
            if (col_indices_[i] == col) {
                return values_[i];
            }
        }
        return 0.0;
    }

    /**
     * @brief Matrix-vector product: y = A * x
     */
    void multiply(const std::vector<Real>& x, std::vector<Real>& y) const {
        y.resize(nrows_);
        for (size_t row = 0; row < nrows_; ++row) {
            Real sum = 0.0;
            for (size_t i = row_ptr_[row]; i < row_ptr_[row + 1]; ++i) {
                sum += values_[i] * x[col_indices_[i]];
            }
            y[row] = sum;
        }
    }

    /**
     * @brief Compute y = A*x + beta*y
     */
    void multiply_add(const std::vector<Real>& x, std::vector<Real>& y, Real beta = 0.0) const {
        for (size_t row = 0; row < nrows_; ++row) {
            Real sum = beta * y[row];
            for (size_t i = row_ptr_[row]; i < row_ptr_[row + 1]; ++i) {
                sum += values_[i] * x[col_indices_[i]];
            }
            y[row] = sum;
        }
    }

    /**
     * @brief Get diagonal values
     */
    void get_diagonal(std::vector<Real>& diag) const {
        diag.resize(std::min(nrows_, ncols_));
        for (size_t row = 0; row < diag.size(); ++row) {
            diag[row] = get(row, row);
        }
    }

    /**
     * @brief Add local element stiffness to global matrix
     */
    void add_element_matrix(const std::vector<Index>& dof_map,
                            const std::vector<Real>& ke) {
        size_t n = dof_map.size();
        for (size_t i = 0; i < n; ++i) {
            Index row = dof_map[i];
            if (row == static_cast<Index>(-1)) continue;  // Constrained DOF

            for (size_t j = 0; j < n; ++j) {
                Index col = dof_map[j];
                if (col == static_cast<Index>(-1)) continue;

                add(row, col, ke[i * n + j]);
            }
        }
    }

    size_t rows() const { return nrows_; }
    size_t cols() const { return ncols_; }
    size_t nnz() const { return values_.size(); }

    const std::vector<Real>& values() const { return values_; }
    const std::vector<size_t>& col_indices() const { return col_indices_; }
    const std::vector<size_t>& row_ptr() const { return row_ptr_; }

private:
    size_t nrows_ = 0;
    size_t ncols_ = 0;
    std::vector<Real> values_;
    std::vector<size_t> col_indices_;
    std::vector<size_t> row_ptr_;
};

// ============================================================================
// Linear Solver Interface
// ============================================================================

enum class LinearSolverType {
    DirectLU,           ///< Direct LU factorization (dense, small problems)
    ConjugateGradient,  ///< CG (symmetric positive definite)
    GMRES,              ///< GMRES (general non-symmetric)
    Jacobi              ///< Jacobi iteration (simple, for preconditioning)
};

struct LinearSolverResult {
    bool converged = false;
    int iterations = 0;
    Real residual = 0.0;
    Real relative_residual = 0.0;
    std::string diagnostic;
};

/**
 * @brief Base class for linear solvers
 */
class LinearSolver {
public:
    virtual ~LinearSolver() = default;

    /**
     * @brief Solve A*x = b
     */
    virtual LinearSolverResult solve(const SparseMatrix& A,
                                     const std::vector<Real>& b,
                                     std::vector<Real>& x) = 0;

    void set_tolerance(Real tol) { tolerance_ = tol; }
    void set_max_iterations(int max_iter) { max_iterations_ = max_iter; }

protected:
    Real tolerance_ = 1e-8;
    int max_iterations_ = 1000;
};

/**
 * @brief Conjugate Gradient solver (for SPD matrices)
 */
class CGSolver : public LinearSolver {
public:
    CGSolver() = default;

    /**
     * @brief Enable Jacobi preconditioning
     */
    void set_preconditioner(bool use_jacobi) { use_jacobi_ = use_jacobi; }

    LinearSolverResult solve(const SparseMatrix& A,
                             const std::vector<Real>& b,
                             std::vector<Real>& x) override {
        size_t n = A.rows();
        x.resize(n, 0.0);

        // Get preconditioner (diagonal)
        std::vector<Real> M_inv;
        if (use_jacobi_) {
            A.get_diagonal(M_inv);
            for (auto& m : M_inv) {
                m = (std::abs(m) > 1e-14) ? 1.0 / m : 1.0;
            }
        }

        // r = b - A*x
        std::vector<Real> r(n), z(n), p(n), Ap(n);
        A.multiply(x, r);
        for (size_t i = 0; i < n; ++i) {
            r[i] = b[i] - r[i];
        }

        // Initial residual norm
        Real b_norm = 0.0;
        for (size_t i = 0; i < n; ++i) {
            b_norm += b[i] * b[i];
        }
        b_norm = std::sqrt(b_norm);

        // Guard: NaN/Inf in RHS
        if (std::isnan(b_norm) || std::isinf(b_norm)) {
            LinearSolverResult result;
            result.converged = false;
            result.diagnostic = "NaN or Inf detected in RHS vector";
            return result;
        }

        if (b_norm < 1e-14) b_norm = 1.0;

        // z = M^{-1} * r (or z = r if no preconditioner)
        if (use_jacobi_) {
            for (size_t i = 0; i < n; ++i) z[i] = M_inv[i] * r[i];
        } else {
            z = r;
        }

        p = z;

        Real rz = 0.0;
        for (size_t i = 0; i < n; ++i) rz += r[i] * z[i];

        LinearSolverResult result;

        // Check initial residual for early exit
        Real initial_r_norm = 0.0;
        for (size_t i = 0; i < n; ++i) initial_r_norm += r[i] * r[i];
        initial_r_norm = std::sqrt(initial_r_norm);

        // Early exit: initial guess already satisfies the system
        if (initial_r_norm < tolerance_ * b_norm) {
            result.converged = true;
            result.residual = initial_r_norm;
            result.relative_residual = initial_r_norm / b_norm;
            return result;
        }

        for (int iter = 0; iter < max_iterations_; ++iter) {
            // Ap = A * p
            A.multiply(p, Ap);

            // alpha = (r'*z) / (p'*Ap)
            Real pAp = 0.0;
            for (size_t i = 0; i < n; ++i) pAp += p[i] * Ap[i];

            if (std::abs(pAp) < 1e-30) {
                result.converged = false;
                result.iterations = iter;
                result.diagnostic = "pAp near zero — possible singular or indefinite matrix";
                return result;
            }

            Real alpha = rz / pAp;

            // x = x + alpha * p
            // r = r - alpha * Ap
            for (size_t i = 0; i < n; ++i) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            // Check convergence
            Real r_norm = 0.0;
            for (size_t i = 0; i < n; ++i) r_norm += r[i] * r[i];
            r_norm = std::sqrt(r_norm);

            // Guard: NaN in residual
            if (std::isnan(r_norm) || std::isinf(r_norm)) {
                result.converged = false;
                result.iterations = iter + 1;
                result.diagnostic = "NaN or Inf in residual at iteration " + std::to_string(iter + 1);
                return result;
            }

            result.residual = r_norm;
            result.relative_residual = r_norm / b_norm;
            result.iterations = iter + 1;

            if (result.relative_residual < tolerance_) {
                result.converged = true;
                return result;
            }

            // z = M^{-1} * r
            if (use_jacobi_) {
                for (size_t i = 0; i < n; ++i) z[i] = M_inv[i] * r[i];
            } else {
                z = r;
            }

            // beta = (r_new' * z_new) / (r_old' * z_old)
            Real rz_new = 0.0;
            for (size_t i = 0; i < n; ++i) rz_new += r[i] * z[i];

            Real beta = rz_new / rz;
            rz = rz_new;

            // p = z + beta * p
            for (size_t i = 0; i < n; ++i) {
                p[i] = z[i] + beta * p[i];
            }
        }

        return result;
    }

private:
    bool use_jacobi_ = true;
};

/**
 * @brief Simple direct solver using LU decomposition (dense)
 * For small problems or debugging
 */
class DirectSolver : public LinearSolver {
public:
    LinearSolverResult solve(const SparseMatrix& A,
                             const std::vector<Real>& b,
                             std::vector<Real>& x) override {
        size_t n = A.rows();
        x = b;

        // Convert to dense (only for small matrices!)
        if (n > 5000) {
            NXS_LOG_WARN("DirectSolver: matrix size {} too large, use iterative solver", n);
        }

        std::vector<Real> dense(n * n, 0.0);
        const auto& row_ptr = A.row_ptr();
        const auto& col_idx = A.col_indices();
        const auto& vals = A.values();

        for (size_t row = 0; row < n; ++row) {
            for (size_t i = row_ptr[row]; i < row_ptr[row + 1]; ++i) {
                dense[row * n + col_idx[i]] = vals[i];
            }
        }

        // LU decomposition with partial pivoting
        std::vector<size_t> pivot(n);
        for (size_t i = 0; i < n; ++i) pivot[i] = i;

        for (size_t k = 0; k < n; ++k) {
            // Find pivot
            size_t max_row = k;
            Real max_val = std::abs(dense[k * n + k]);
            for (size_t i = k + 1; i < n; ++i) {
                if (std::abs(dense[i * n + k]) > max_val) {
                    max_val = std::abs(dense[i * n + k]);
                    max_row = i;
                }
            }

            if (max_val < 1e-14) {
                LinearSolverResult result;
                result.converged = false;
                return result;
            }

            // Swap rows
            if (max_row != k) {
                std::swap(pivot[k], pivot[max_row]);
                for (size_t j = 0; j < n; ++j) {
                    std::swap(dense[k * n + j], dense[max_row * n + j]);
                }
                std::swap(x[k], x[max_row]);
            }

            // Eliminate
            for (size_t i = k + 1; i < n; ++i) {
                Real factor = dense[i * n + k] / dense[k * n + k];
                dense[i * n + k] = factor;  // Store L
                for (size_t j = k + 1; j < n; ++j) {
                    dense[i * n + j] -= factor * dense[k * n + j];
                }
                x[i] -= factor * x[k];
            }
        }

        // Back substitution
        for (int i = n - 1; i >= 0; --i) {
            for (size_t j = i + 1; j < n; ++j) {
                x[i] -= dense[i * n + j] * x[j];
            }
            x[i] /= dense[i * n + i];
        }

        // Guard: scan solution for NaN/Inf
        LinearSolverResult result;
        result.iterations = 1;
        for (size_t i = 0; i < n; ++i) {
            if (std::isnan(x[i]) || std::isinf(x[i])) {
                result.converged = false;
                result.diagnostic = "NaN or Inf in solution after back-substitution";
                return result;
            }
        }
        result.converged = true;
        return result;
    }
};

// ============================================================================
// Newton-Raphson Nonlinear Solver
// ============================================================================

struct NewtonRaphsonResult {
    bool converged = false;
    int iterations = 0;
    Real residual_norm = 0.0;
    Real relative_residual = 0.0;
    std::vector<Real> residual_history;
};

/**
 * @brief Newton-Raphson solver for nonlinear problems
 *
 * Solves: R(u) = F_int(u) - F_ext = 0
 *
 * Iteration: u_{n+1} = u_n - K_t^{-1} * R(u_n)
 * where K_t = dR/du is the tangent stiffness
 */
class NewtonRaphsonSolver {
public:
    using ResidualFunction = std::function<void(const std::vector<Real>& u,
                                                 std::vector<Real>& residual)>;

    using TangentFunction = std::function<void(const std::vector<Real>& u,
                                                SparseMatrix& K_tangent)>;

    NewtonRaphsonSolver() : linear_solver_(std::make_unique<CGSolver>()) {}

    /**
     * @brief Set callback for residual computation
     */
    void set_residual_function(ResidualFunction func) {
        compute_residual_ = std::move(func);
    }

    /**
     * @brief Set callback for tangent stiffness computation
     */
    void set_tangent_function(TangentFunction func) {
        compute_tangent_ = std::move(func);
    }

    /**
     * @brief Set linear solver type
     */
    void set_linear_solver(LinearSolverType type) {
        switch (type) {
            case LinearSolverType::ConjugateGradient:
                linear_solver_ = std::make_unique<CGSolver>();
                break;
            case LinearSolverType::DirectLU:
                linear_solver_ = std::make_unique<DirectSolver>();
                break;
            default:
                linear_solver_ = std::make_unique<CGSolver>();
        }
    }

    /**
     * @brief Solve the nonlinear system
     * @param u Initial guess and final solution
     */
    NewtonRaphsonResult solve(std::vector<Real>& u) {
        NewtonRaphsonResult result;
        size_t n = u.size();

        std::vector<Real> residual(n);
        std::vector<Real> delta_u(n);

        // Compute initial residual
        compute_residual_(u, residual);

        Real r0_norm = vector_norm(residual);
        result.residual_history.push_back(r0_norm);

        if (r0_norm < abs_tolerance_) {
            result.converged = true;
            result.residual_norm = r0_norm;
            return result;
        }

        for (int iter = 0; iter < max_iterations_; ++iter) {
            // Compute tangent stiffness
            compute_tangent_(u, K_tangent_);

            // Solve: K_t * delta_u = -R
            std::vector<Real> neg_r(n);
            for (size_t i = 0; i < n; ++i) neg_r[i] = -residual[i];

            auto lin_result = linear_solver_->solve(K_tangent_, neg_r, delta_u);

            if (!lin_result.converged) {
                NXS_LOG_WARN("Newton-Raphson: linear solver failed at iteration {}", iter);
                result.converged = false;
                result.iterations = iter + 1;
                return result;
            }

            // Line search (optional)
            Real alpha = 1.0;
            if (use_line_search_) {
                alpha = line_search(u, delta_u, residual);
            }

            // Update: u = u + alpha * delta_u
            for (size_t i = 0; i < n; ++i) {
                u[i] += alpha * delta_u[i];
            }

            // Compute new residual
            compute_residual_(u, residual);

            Real r_norm = vector_norm(residual);
            result.residual_history.push_back(r_norm);
            result.residual_norm = r_norm;
            result.relative_residual = r_norm / r0_norm;
            result.iterations = iter + 1;

            if (verbose_) {
                std::cout << "  Newton iter " << iter + 1
                          << ": |R| = " << r_norm
                          << ", |R|/|R0| = " << result.relative_residual << "\n";
            }

            // Check convergence
            if (r_norm < abs_tolerance_ || result.relative_residual < rel_tolerance_) {
                result.converged = true;
                return result;
            }
        }

        return result;
    }

    // Configuration
    void set_tolerance(Real abs_tol, Real rel_tol) {
        abs_tolerance_ = abs_tol;
        rel_tolerance_ = rel_tol;
    }
    void set_max_iterations(int max_iter) { max_iterations_ = max_iter; }
    void set_line_search(bool enable) { use_line_search_ = enable; }
    void set_verbose(bool verbose) { verbose_ = verbose; }

private:
    Real vector_norm(const std::vector<Real>& v) {
        Real sum = 0.0;
        for (Real x : v) sum += x * x;
        return std::sqrt(sum);
    }

    Real line_search(const std::vector<Real>& u,
                     const std::vector<Real>& delta_u,
                     const std::vector<Real>& residual) {
        // Backtracking line search
        Real alpha = 1.0;
        Real r0_norm = vector_norm(residual);

        std::vector<Real> u_trial(u.size());
        std::vector<Real> r_trial(u.size());

        for (int i = 0; i < 10; ++i) {
            for (size_t j = 0; j < u.size(); ++j) {
                u_trial[j] = u[j] + alpha * delta_u[j];
            }

            compute_residual_(u_trial, r_trial);
            Real r_norm = vector_norm(r_trial);

            if (r_norm < (1.0 - 0.1 * alpha) * r0_norm) {
                return alpha;
            }

            alpha *= 0.5;
        }

        return alpha;
    }

    ResidualFunction compute_residual_;
    TangentFunction compute_tangent_;
    SparseMatrix K_tangent_;

    std::unique_ptr<LinearSolver> linear_solver_;

    Real abs_tolerance_ = 1e-8;
    Real rel_tolerance_ = 1e-6;
    int max_iterations_ = 20;
    bool use_line_search_ = true;
    bool verbose_ = false;
};

// ============================================================================
// Implicit Time Integrator (Newmark-β)
// ============================================================================

/**
 * @brief Newmark-β implicit time integration
 *
 * Solves: M*a + C*v + K*u = F_ext
 *
 * Newmark formulas:
 *   u_{n+1} = u_n + dt*v_n + dt²*((0.5-β)*a_n + β*a_{n+1})
 *   v_{n+1} = v_n + dt*((1-γ)*a_n + γ*a_{n+1})
 *
 * Standard choices:
 *   β=0.25, γ=0.5: Average acceleration (unconditionally stable)
 *   β=1/6, γ=0.5:  Linear acceleration
 */
class NewmarkIntegrator {
public:
    NewmarkIntegrator(Real beta = 0.25, Real gamma = 0.5)
        : beta_(beta), gamma_(gamma) {}

    /**
     * @brief Initialize integrator
     */
    void initialize(size_t ndof) {
        ndof_ = ndof;
        u_.resize(ndof, 0.0);
        v_.resize(ndof, 0.0);
        a_.resize(ndof, 0.0);
        u_pred_.resize(ndof);
        v_pred_.resize(ndof);
    }

    /**
     * @brief Set mass matrix (diagonal for now)
     */
    void set_mass(const std::vector<Real>& M_diag) {
        M_diag_ = M_diag;
    }

    /**
     * @brief Set damping (Rayleigh: C = alpha*M + beta*K)
     */
    void set_damping(Real alpha_M, Real beta_K) {
        damping_alpha_ = alpha_M;
        damping_beta_ = beta_K;
    }

    /**
     * @brief Perform one implicit time step
     */
    void step(Real dt,
              const SparseMatrix& K,
              const std::vector<Real>& F_ext,
              NewtonRaphsonSolver& newton_solver) {
        dt_ = dt;

        // Predictor step
        for (size_t i = 0; i < ndof_; ++i) {
            u_pred_[i] = u_[i] + dt * v_[i] + dt * dt * (0.5 - beta_) * a_[i];
            v_pred_[i] = v_[i] + dt * (1.0 - gamma_) * a_[i];
        }

        // Set up Newton-Raphson for effective system
        // Effective stiffness: K_eff = K + γ/(β*dt)*C + 1/(β*dt²)*M
        // Effective force: F_eff = F_ext - K*u_pred - C*v_pred - M*a_pred

        newton_solver.set_residual_function(
            [this, &K, &F_ext](const std::vector<Real>& delta_u, std::vector<Real>& R) {
                compute_residual(delta_u, K, F_ext, R);
            });

        newton_solver.set_tangent_function(
            [this, &K](const std::vector<Real>& delta_u, SparseMatrix& K_eff) {
                compute_effective_stiffness(K, K_eff);
            });

        // Solve for displacement increment
        std::vector<Real> delta_u(ndof_, 0.0);
        auto result = newton_solver.solve(delta_u);

        if (!result.converged) {
            NXS_LOG_WARN("Newmark: Newton-Raphson did not converge");
        }

        // Corrector step
        for (size_t i = 0; i < ndof_; ++i) {
            u_[i] = u_pred_[i] + beta_ * dt * dt * delta_u[i] / (beta_ * dt * dt);
            // Actually delta_u IS the displacement correction, so:
            u_[i] = u_pred_[i] + delta_u[i];

            // Update acceleration and velocity
            a_[i] = (u_[i] - u_pred_[i]) / (beta_ * dt * dt);
            v_[i] = v_pred_[i] + gamma_ * dt * a_[i];
        }

        time_ += dt;
    }

    // Accessors
    const std::vector<Real>& displacement() const { return u_; }
    const std::vector<Real>& velocity() const { return v_; }
    const std::vector<Real>& acceleration() const { return a_; }
    Real time() const { return time_; }

    std::vector<Real>& displacement() { return u_; }
    std::vector<Real>& velocity() { return v_; }
    std::vector<Real>& acceleration() { return a_; }

private:
    void compute_residual(const std::vector<Real>& delta_u,
                          const SparseMatrix& K,
                          const std::vector<Real>& F_ext,
                          std::vector<Real>& R) {
        // R = K*(u_pred + delta_u) + C*(v_pred + gamma/(beta*dt)*delta_u)
        //   + M*(a_pred + 1/(beta*dt²)*delta_u) - F_ext

        std::vector<Real> u_new(ndof_), v_new(ndof_), a_new(ndof_);

        Real c1 = 1.0 / (beta_ * dt_ * dt_);
        Real c2 = gamma_ / (beta_ * dt_);

        for (size_t i = 0; i < ndof_; ++i) {
            u_new[i] = u_pred_[i] + delta_u[i];
            a_new[i] = c1 * delta_u[i];
            v_new[i] = v_pred_[i] + c2 * delta_u[i];
        }

        // R = K*u + C*v + M*a - F_ext
        K.multiply(u_new, R);

        for (size_t i = 0; i < ndof_; ++i) {
            // Add damping (Rayleigh: C = alpha*M + beta*K)
            Real C_term = (damping_alpha_ * M_diag_[i] + damping_beta_ * K.get(i, i)) * v_new[i];

            // Add inertia
            Real M_term = M_diag_[i] * a_new[i];

            R[i] += C_term + M_term - F_ext[i];
        }
    }

    void compute_effective_stiffness(const SparseMatrix& K, SparseMatrix& K_eff) {
        // K_eff = K + gamma/(beta*dt)*C + 1/(beta*dt²)*M
        K_eff = K;  // Copy pattern and values

        Real c1 = 1.0 / (beta_ * dt_ * dt_);
        Real c2 = gamma_ / (beta_ * dt_);

        // Add diagonal terms from M and C
        for (size_t i = 0; i < ndof_; ++i) {
            Real K_ii = K.get(i, i);
            Real M_ii = M_diag_[i];
            Real C_ii = damping_alpha_ * M_ii + damping_beta_ * K_ii;

            K_eff.set(i, i, K_ii + c2 * C_ii + c1 * M_ii);
        }
    }

    Real beta_, gamma_;
    Real dt_ = 0.0;
    Real time_ = 0.0;
    size_t ndof_ = 0;

    std::vector<Real> u_, v_, a_;
    std::vector<Real> u_pred_, v_pred_;
    std::vector<Real> M_diag_;

    Real damping_alpha_ = 0.0;
    Real damping_beta_ = 0.0;
};

// ============================================================================
// Static Solver
// ============================================================================

struct StaticSolverResult {
    bool converged = false;
    int load_steps = 0;
    int newton_iterations = 0;
    Real final_residual = 0.0;
};

/**
 * @brief Static structural analysis solver
 *
 * Solves: K(u) * u = F_ext
 * With optional load stepping for nonlinear problems
 */
class StaticSolver {
public:
    using StiffnessFunction = std::function<void(const std::vector<Real>& u,
                                                  SparseMatrix& K)>;

    using InternalForceFunction = std::function<void(const std::vector<Real>& u,
                                                      std::vector<Real>& F_int)>;

    StaticSolver() {
        newton_solver_.set_tolerance(1e-8, 1e-6);
        newton_solver_.set_max_iterations(20);
    }

    /**
     * @brief Set stiffness matrix computation callback
     */
    void set_stiffness_function(StiffnessFunction func) {
        compute_stiffness_ = std::move(func);
    }

    /**
     * @brief Set internal force computation callback
     */
    void set_internal_force_function(InternalForceFunction func) {
        compute_internal_force_ = std::move(func);
    }

    /**
     * @brief Solve static equilibrium with load stepping
     * @param F_ext Total external load
     * @param u Solution (displacement)
     * @param num_load_steps Number of load increments
     */
    StaticSolverResult solve(const std::vector<Real>& F_ext,
                             std::vector<Real>& u,
                             int num_load_steps = 1) {
        StaticSolverResult result;
        size_t ndof = F_ext.size();
        u.resize(ndof, 0.0);

        std::vector<Real> F_step(ndof);
        std::vector<Real> F_applied(ndof, 0.0);

        for (int step = 1; step <= num_load_steps; ++step) {
            Real load_factor = static_cast<Real>(step) / num_load_steps;

            // Incremental load
            for (size_t i = 0; i < ndof; ++i) {
                F_step[i] = load_factor * F_ext[i];
            }

            // Set up Newton-Raphson
            newton_solver_.set_residual_function(
                [this, &F_step](const std::vector<Real>& u_curr, std::vector<Real>& R) {
                    compute_internal_force_(u_curr, R);
                    for (size_t i = 0; i < R.size(); ++i) {
                        R[i] -= F_step[i];  // R = F_int - F_ext
                    }
                });

            newton_solver_.set_tangent_function(
                [this](const std::vector<Real>& u_curr, SparseMatrix& K) {
                    compute_stiffness_(u_curr, K);
                });

            // Solve
            auto nr_result = newton_solver_.solve(u);

            result.newton_iterations += nr_result.iterations;
            result.load_steps = step;

            if (!nr_result.converged) {
                NXS_LOG_WARN("Static solver: Newton-Raphson failed at load step {}", step);
                result.converged = false;
                result.final_residual = nr_result.residual_norm;
                return result;
            }

            if (verbose_) {
                std::cout << "Load step " << step << "/" << num_load_steps
                          << " converged in " << nr_result.iterations << " iterations\n";
            }
        }

        result.converged = true;
        return result;
    }

    /**
     * @brief Solve linear static problem (single step)
     */
    StaticSolverResult solve_linear(const SparseMatrix& K,
                                    const std::vector<Real>& F_ext,
                                    std::vector<Real>& u) {
        StaticSolverResult result;

        CGSolver cg;
        cg.set_tolerance(1e-10);
        cg.set_max_iterations(2000);

        auto lin_result = cg.solve(K, F_ext, u);

        result.converged = lin_result.converged;
        result.newton_iterations = lin_result.iterations;
        result.final_residual = lin_result.residual;

        return result;
    }

    void set_verbose(bool verbose) {
        verbose_ = verbose;
        newton_solver_.set_verbose(verbose);
    }

    NewtonRaphsonSolver& newton_solver() { return newton_solver_; }

private:
    StiffnessFunction compute_stiffness_;
    InternalForceFunction compute_internal_force_;
    NewtonRaphsonSolver newton_solver_;
    bool verbose_ = false;
};

} // namespace solver
} // namespace nxs
