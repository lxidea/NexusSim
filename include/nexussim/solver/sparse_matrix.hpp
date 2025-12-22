#pragma once

/**
 * @file sparse_matrix.hpp
 * @brief Sparse matrix storage and linear solvers for implicit FEM
 *
 * This module provides:
 * - CSR (Compressed Sparse Row) matrix storage
 * - Preconditioned Conjugate Gradient (PCG) solver
 * - Jacobi and SSOR preconditioners
 * - Matrix assembly utilities
 *
 * Designed for GPU compatibility with Kokkos.
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>

namespace nxs {
namespace solver {

// ============================================================================
// Sparse Matrix in CSR Format
// ============================================================================

/**
 * @brief Compressed Sparse Row (CSR) matrix storage
 *
 * CSR format stores:
 * - values: Non-zero values, row by row
 * - col_indices: Column index for each value
 * - row_ptr: Index into values/col_indices for start of each row
 *
 * For row i: values[row_ptr[i]:row_ptr[i+1]] are the non-zeros
 *            col_indices[row_ptr[i]:row_ptr[i+1]] are their column indices
 */
class SparseMatrixCSR {
public:
    SparseMatrixCSR() : num_rows_(0), num_cols_(0), nnz_(0) {}

    SparseMatrixCSR(Index num_rows, Index num_cols)
        : num_rows_(num_rows)
        , num_cols_(num_cols)
        , nnz_(0)
        , row_ptr_(num_rows + 1, 0)
    {}

    // ========================================================================
    // Matrix Properties
    // ========================================================================

    Index num_rows() const { return num_rows_; }
    Index num_cols() const { return num_cols_; }
    Index nnz() const { return nnz_; }

    const std::vector<Real>& values() const { return values_; }
    const std::vector<Index>& col_indices() const { return col_indices_; }
    const std::vector<Index>& row_ptr() const { return row_ptr_; }

    std::vector<Real>& values() { return values_; }
    std::vector<Index>& col_indices() { return col_indices_; }
    std::vector<Index>& row_ptr() { return row_ptr_; }

    // ========================================================================
    // Matrix Construction
    // ========================================================================

    /**
     * @brief Build matrix from COO (coordinate) format triplets
     * @param rows Row indices
     * @param cols Column indices
     * @param vals Values
     *
     * Duplicate entries are summed (standard FEM assembly behavior).
     */
    void build_from_coo(const std::vector<Index>& rows,
                        const std::vector<Index>& cols,
                        const std::vector<Real>& vals) {
        if (rows.size() != cols.size() || rows.size() != vals.size()) {
            throw std::runtime_error("COO arrays must have same size");
        }

        // Use map to accumulate duplicates
        std::map<std::pair<Index, Index>, Real> entries;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            entries[{rows[i], cols[i]}] += vals[i];
        }

        // Count entries per row
        std::vector<Index> row_counts(num_rows_, 0);
        for (const auto& entry : entries) {
            row_counts[entry.first.first]++;
        }

        // Build row_ptr
        row_ptr_.resize(num_rows_ + 1);
        row_ptr_[0] = 0;
        for (Index i = 0; i < num_rows_; ++i) {
            row_ptr_[i + 1] = row_ptr_[i] + row_counts[i];
        }

        // Allocate storage
        nnz_ = row_ptr_[num_rows_];
        values_.resize(nnz_);
        col_indices_.resize(nnz_);

        // Fill values and col_indices
        std::vector<Index> current_pos(num_rows_, 0);
        for (const auto& entry : entries) {
            Index row = entry.first.first;
            Index col = entry.first.second;
            Real val = entry.second;

            Index pos = row_ptr_[row] + current_pos[row];
            values_[pos] = val;
            col_indices_[pos] = col;
            current_pos[row]++;
        }
    }

    /**
     * @brief Set matrix structure from sparsity pattern (values initialized to zero)
     */
    void set_structure(const std::vector<Index>& row_ptr,
                       const std::vector<Index>& col_indices) {
        row_ptr_ = row_ptr;
        col_indices_ = col_indices;
        nnz_ = col_indices.size();
        values_.assign(nnz_, 0.0);
    }

    /**
     * @brief Zero all values (keep structure)
     */
    void zero() {
        std::fill(values_.begin(), values_.end(), 0.0);
    }

    /**
     * @brief Add value to matrix entry (assumes entry exists in structure)
     */
    void add(Index row, Index col, Real val) {
        for (Index k = row_ptr_[row]; k < row_ptr_[row + 1]; ++k) {
            if (col_indices_[k] == col) {
                values_[k] += val;
                return;
            }
        }
        // Entry not found - this shouldn't happen if structure is correct
    }

    /**
     * @brief Get matrix entry (returns 0 if not in structure)
     */
    Real get(Index row, Index col) const {
        for (Index k = row_ptr_[row]; k < row_ptr_[row + 1]; ++k) {
            if (col_indices_[k] == col) {
                return values_[k];
            }
        }
        return 0.0;
    }

    /**
     * @brief Get diagonal element
     */
    Real diagonal(Index i) const {
        return get(i, i);
    }

    /**
     * @brief Extract diagonal as vector
     */
    void get_diagonal(std::vector<Real>& diag) const {
        diag.resize(num_rows_);
        for (Index i = 0; i < num_rows_; ++i) {
            diag[i] = diagonal(i);
        }
    }

    // ========================================================================
    // Matrix-Vector Operations
    // ========================================================================

    /**
     * @brief Matrix-vector product: y = A * x
     */
    void multiply(const std::vector<Real>& x, std::vector<Real>& y) const {
        y.assign(num_rows_, 0.0);
        for (Index i = 0; i < num_rows_; ++i) {
            Real sum = 0.0;
            for (Index k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
                sum += values_[k] * x[col_indices_[k]];
            }
            y[i] = sum;
        }
    }

    /**
     * @brief Matrix-vector product: y = A * x (Kokkos views)
     */
    template<typename ViewX, typename ViewY>
    void multiply(const ViewX& x, ViewY& y) const {
        Kokkos::parallel_for("SpMV", num_rows_, KOKKOS_LAMBDA(const Index i) {
            Real sum = 0.0;
            for (Index k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
                sum += values_[k] * x(col_indices_[k]);
            }
            y(i) = sum;
        });
    }

    // ========================================================================
    // Utilities
    // ========================================================================

    /**
     * @brief Print matrix in dense format (for debugging small matrices)
     */
    void print_dense(std::ostream& os = std::cout) const {
        for (Index i = 0; i < num_rows_; ++i) {
            for (Index j = 0; j < num_cols_; ++j) {
                os << get(i, j) << " ";
            }
            os << "\n";
        }
    }

    /**
     * @brief Print sparsity pattern statistics
     */
    void print_stats(std::ostream& os = std::cout) const {
        os << "Sparse Matrix (CSR):\n"
           << "  Rows: " << num_rows_ << "\n"
           << "  Cols: " << num_cols_ << "\n"
           << "  NNZ:  " << nnz_ << "\n"
           << "  Density: " << (100.0 * nnz_ / (num_rows_ * num_cols_)) << "%\n";
    }

private:
    Index num_rows_;
    Index num_cols_;
    Index nnz_;
    std::vector<Real> values_;
    std::vector<Index> col_indices_;
    std::vector<Index> row_ptr_;
};

// ============================================================================
// Preconditioners
// ============================================================================

/**
 * @brief Preconditioner interface
 */
class Preconditioner {
public:
    virtual ~Preconditioner() = default;

    /**
     * @brief Apply preconditioner: z = M^{-1} * r
     */
    virtual void apply(const std::vector<Real>& r, std::vector<Real>& z) const = 0;

    /**
     * @brief Setup preconditioner from matrix
     */
    virtual void setup(const SparseMatrixCSR& A) = 0;
};

/**
 * @brief Jacobi (diagonal) preconditioner: M = diag(A)
 */
class JacobiPreconditioner : public Preconditioner {
public:
    void setup(const SparseMatrixCSR& A) override {
        A.get_diagonal(diag_inv_);
        for (auto& d : diag_inv_) {
            d = (std::abs(d) > 1.0e-30) ? 1.0 / d : 1.0;
        }
    }

    void apply(const std::vector<Real>& r, std::vector<Real>& z) const override {
        z.resize(r.size());
        for (std::size_t i = 0; i < r.size(); ++i) {
            z[i] = diag_inv_[i] * r[i];
        }
    }

private:
    std::vector<Real> diag_inv_;
};

/**
 * @brief SSOR (Symmetric Successive Over-Relaxation) preconditioner
 *
 * More effective than Jacobi for ill-conditioned systems.
 * M = (D + ωL) D^{-1} (D + ωU) where L, U are strict lower/upper parts
 */
class SSORPreconditioner : public Preconditioner {
public:
    SSORPreconditioner(Real omega = 1.0) : omega_(omega) {}

    void setup(const SparseMatrixCSR& A) override {
        A_ = &A;
        A.get_diagonal(diag_);
    }

    void apply(const std::vector<Real>& r, std::vector<Real>& z) const override {
        const Index n = r.size();
        z.assign(n, 0.0);
        std::vector<Real> y(n, 0.0);

        // Forward sweep: (D + ωL) y = r
        for (Index i = 0; i < n; ++i) {
            Real sum = r[i];
            for (Index k = A_->row_ptr()[i]; k < A_->row_ptr()[i + 1]; ++k) {
                Index j = A_->col_indices()[k];
                if (j < i) {
                    sum -= omega_ * A_->values()[k] * y[j];
                }
            }
            y[i] = sum / diag_[i];
        }

        // Scale: z = D * y
        for (Index i = 0; i < n; ++i) {
            z[i] = diag_[i] * y[i];
        }

        // Backward sweep: (D + ωU) z = D * y
        for (Index i = n - 1; i >= 0; --i) {
            Real sum = z[i];
            for (Index k = A_->row_ptr()[i]; k < A_->row_ptr()[i + 1]; ++k) {
                Index j = A_->col_indices()[k];
                if (j > i) {
                    sum -= omega_ * A_->values()[k] * z[j];
                }
            }
            z[i] = sum / diag_[i];
            if (i == 0) break;  // Prevent underflow for unsigned Index
        }
    }

private:
    Real omega_;
    const SparseMatrixCSR* A_;
    std::vector<Real> diag_;
};

// ============================================================================
// Conjugate Gradient Solver
// ============================================================================

/**
 * @brief Solver result information
 */
struct SolverResult {
    bool converged;
    int iterations;
    Real residual_norm;
    Real initial_residual;
    std::string message;
};

/**
 * @brief Preconditioned Conjugate Gradient (PCG) solver
 *
 * Solves A * x = b for symmetric positive definite A.
 *
 * Algorithm:
 * 1. r = b - A*x
 * 2. z = M^{-1} * r
 * 3. p = z
 * 4. For k = 1, 2, ...
 *    a. α = (r·z) / (p·A*p)
 *    b. x = x + α*p
 *    c. r_new = r - α*A*p
 *    d. Check convergence
 *    e. z_new = M^{-1} * r_new
 *    f. β = (r_new·z_new) / (r·z)
 *    g. p = z_new + β*p
 *    h. r = r_new, z = z_new
 */
class ConjugateGradientSolver {
public:
    struct Config {
        int max_iterations;
        Real tolerance;
        bool verbose;

        Config()
            : max_iterations(1000)
            , tolerance(1.0e-10)
            , verbose(false)
        {}
    };

    ConjugateGradientSolver(const Config& config = Config())
        : config_(config)
    {}

    /**
     * @brief Solve A * x = b
     * @param A Sparse matrix (symmetric positive definite)
     * @param b Right-hand side vector
     * @param x Solution vector (initial guess on input, solution on output)
     * @param precond Preconditioner (optional)
     * @return Solver result
     */
    SolverResult solve(const SparseMatrixCSR& A,
                       const std::vector<Real>& b,
                       std::vector<Real>& x,
                       const Preconditioner* precond = nullptr) {
        const Index n = A.num_rows();
        SolverResult result;
        result.converged = false;
        result.iterations = 0;

        // Default preconditioner if not provided
        JacobiPreconditioner default_precond;
        if (!precond) {
            default_precond.setup(A);
            precond = &default_precond;
        }

        // Initialize solution if empty
        if (x.size() != n) {
            x.assign(n, 0.0);
        }

        // Allocate work vectors
        std::vector<Real> r(n), z(n), p(n), Ap(n);

        // r = b - A*x
        A.multiply(x, Ap);
        for (Index i = 0; i < n; ++i) {
            r[i] = b[i] - Ap[i];
        }

        // Initial residual
        result.initial_residual = norm(r);
        if (result.initial_residual < config_.tolerance) {
            result.converged = true;
            result.residual_norm = result.initial_residual;
            result.message = "Already converged";
            return result;
        }

        // z = M^{-1} * r
        precond->apply(r, z);

        // p = z
        p = z;

        // r·z
        Real rz = dot(r, z);

        for (int iter = 0; iter < config_.max_iterations; ++iter) {
            // Ap = A * p
            A.multiply(p, Ap);

            // α = (r·z) / (p·Ap)
            Real pAp = dot(p, Ap);
            if (std::abs(pAp) < 1.0e-30) {
                result.message = "Breakdown: p·Ap = 0";
                result.iterations = iter;
                result.residual_norm = norm(r);
                return result;
            }
            Real alpha = rz / pAp;

            // x = x + α*p
            // r_new = r - α*Ap
            for (Index i = 0; i < n; ++i) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            // Check convergence
            Real res_norm = norm(r);
            Real rel_res = res_norm / result.initial_residual;

            if (config_.verbose && iter % 10 == 0) {
                std::cout << "  CG iter " << iter << ": |r| = " << res_norm
                          << ", rel = " << rel_res << "\n";
            }

            if (res_norm < config_.tolerance ||
                rel_res < config_.tolerance) {
                result.converged = true;
                result.iterations = iter + 1;
                result.residual_norm = res_norm;
                result.message = "Converged";
                return result;
            }

            // z_new = M^{-1} * r_new
            precond->apply(r, z);

            // β = (r_new·z_new) / (r·z)
            Real rz_new = dot(r, z);
            Real beta = rz_new / rz;
            rz = rz_new;

            // p = z + β*p
            for (Index i = 0; i < n; ++i) {
                p[i] = z[i] + beta * p[i];
            }
        }

        result.iterations = config_.max_iterations;
        result.residual_norm = norm(r);
        result.message = "Max iterations reached";
        return result;
    }

private:
    Config config_;

    static Real dot(const std::vector<Real>& a, const std::vector<Real>& b) {
        Real sum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    static Real norm(const std::vector<Real>& v) {
        return std::sqrt(dot(v, v));
    }
};

// ============================================================================
// Stiffness Matrix Assembler
// ============================================================================

/**
 * @brief Helper class for assembling global stiffness matrix from element contributions
 */
class StiffnessAssembler {
public:
    /**
     * @brief Initialize assembler with DOF count
     */
    StiffnessAssembler(Index num_dof)
        : num_dof_(num_dof)
    {}

    /**
     * @brief Add element stiffness matrix to global assembly
     * @param elem_dofs Global DOF indices for element nodes
     * @param K_elem Element stiffness matrix (dense, row-major)
     * @param num_elem_dof Number of DOFs per element
     */
    void add_element(const std::vector<Index>& elem_dofs,
                     const Real* K_elem,
                     Index num_elem_dof) {
        for (Index i = 0; i < num_elem_dof; ++i) {
            Index row = elem_dofs[i];
            for (Index j = 0; j < num_elem_dof; ++j) {
                Index col = elem_dofs[j];
                Real val = K_elem[i * num_elem_dof + j];
                if (std::abs(val) > 1.0e-30) {
                    coo_rows_.push_back(row);
                    coo_cols_.push_back(col);
                    coo_vals_.push_back(val);
                }
            }
        }
    }

    /**
     * @brief Build CSR matrix from accumulated element contributions
     */
    SparseMatrixCSR build() {
        SparseMatrixCSR K(num_dof_, num_dof_);
        K.build_from_coo(coo_rows_, coo_cols_, coo_vals_);
        return K;
    }

    /**
     * @brief Clear accumulated entries
     */
    void clear() {
        coo_rows_.clear();
        coo_cols_.clear();
        coo_vals_.clear();
    }

private:
    Index num_dof_;
    std::vector<Index> coo_rows_;
    std::vector<Index> coo_cols_;
    std::vector<Real> coo_vals_;
};

// ============================================================================
// Boundary Condition Application
// ============================================================================

/**
 * @brief Apply Dirichlet boundary conditions to linear system
 *
 * Method: Modify matrix rows to enforce u_i = g_i
 * - Set row i to identity: A_ij = δ_ij
 * - Set RHS: b_i = g_i
 * - Modify other rows for symmetry (optional)
 */
class BoundaryConditionApplicator {
public:
    /**
     * @brief Apply Dirichlet BC by zeroing row and column (maintains symmetry)
     */
    static void apply_dirichlet(SparseMatrixCSR& A,
                                std::vector<Real>& b,
                                const std::vector<Index>& bc_dofs,
                                const std::vector<Real>& bc_values,
                                bool maintain_symmetry = true) {
        // Store diagonal values to preserve conditioning
        std::vector<Real> diag_vals(bc_dofs.size());
        for (std::size_t k = 0; k < bc_dofs.size(); ++k) {
            diag_vals[k] = A.diagonal(bc_dofs[k]);
            if (std::abs(diag_vals[k]) < 1.0e-30) {
                diag_vals[k] = 1.0;  // Default diagonal
            }
        }

        if (maintain_symmetry) {
            // Modify columns for symmetry
            for (std::size_t k = 0; k < bc_dofs.size(); ++k) {
                Index bc_dof = bc_dofs[k];
                Real bc_val = bc_values[k];

                // For each row, modify RHS and zero column entry
                for (Index i = 0; i < A.num_rows(); ++i) {
                    for (Index j = A.row_ptr()[i]; j < A.row_ptr()[i + 1]; ++j) {
                        if (A.col_indices()[j] == bc_dof && i != bc_dof) {
                            b[i] -= A.values()[j] * bc_val;
                            A.values()[j] = 0.0;
                        }
                    }
                }
            }
        }

        // Zero BC rows and set diagonal
        for (std::size_t k = 0; k < bc_dofs.size(); ++k) {
            Index bc_dof = bc_dofs[k];
            Real bc_val = bc_values[k];

            // Zero row
            for (Index j = A.row_ptr()[bc_dof]; j < A.row_ptr()[bc_dof + 1]; ++j) {
                if (A.col_indices()[j] == bc_dof) {
                    A.values()[j] = diag_vals[k];
                } else {
                    A.values()[j] = 0.0;
                }
            }

            // Set RHS
            b[bc_dof] = diag_vals[k] * bc_val;
        }
    }
};

} // namespace solver
} // namespace nxs
