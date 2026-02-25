#pragma once

/**
 * @file petsc_solver.hpp
 * @brief PETSc linear solver wrapper (optional backend)
 *
 * Entirely behind NEXUSSIM_HAVE_PETSC compile guard.
 * Provides scalable sparse solvers for very large problems.
 */

#ifdef NEXUSSIM_HAVE_PETSC

#include <nexussim/solver/implicit_solver.hpp>
#include <petsc.h>
#include <vector>
#include <string>
#include <stdexcept>

namespace nxs {
namespace solver {

/**
 * @brief RAII guard for PETSc initialization/finalization
 */
class PETScContext {
public:
    PETScContext(int* argc = nullptr, char*** argv = nullptr) {
        PetscInitialize(argc, argv, nullptr, nullptr);
    }
    ~PETScContext() {
        PetscFinalize();
    }
    PETScContext(const PETScContext&) = delete;
    PETScContext& operator=(const PETScContext&) = delete;
};

enum class PETScKSPType { CG, GMRES, BCGS, PREONLY };
enum class PETScPCType { None, Jacobi, ILU, ICC, AMG, LU, Cholesky };

/**
 * @brief PETSc-based linear solver
 */
class PETScLinearSolver : public LinearSolver {
public:
    static_assert(sizeof(Real) == sizeof(PetscScalar),
                  "NexusSim Real and PetscScalar must have same size");

    PETScLinearSolver() = default;

    void set_ksp_type(PETScKSPType type) { ksp_type_ = type; }
    void set_pc_type(PETScPCType type) { pc_type_ = type; }
    void set_from_options(bool enable) { from_options_ = enable; }

    LinearSolverResult solve(const SparseMatrix& A,
                             const std::vector<Real>& b,
                             std::vector<Real>& x) override {
        LinearSolverResult result;
        PetscInt n = static_cast<PetscInt>(A.rows());

        // Store CSR data (PETSc doesn't copy with MatCreateSeqAIJWithArrays)
        const auto& row_ptr_orig = A.row_ptr();
        const auto& col_idx_orig = A.col_indices();
        const auto& vals_orig = A.values();

        // Convert to PetscInt arrays
        std::vector<PetscInt> row_ptr(row_ptr_orig.begin(), row_ptr_orig.end());
        std::vector<PetscInt> col_idx(col_idx_orig.begin(), col_idx_orig.end());
        std::vector<PetscScalar> vals(vals_orig.begin(), vals_orig.end());

        // Create PETSc matrix
        Mat Amat;
        MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, n, n,
                                   row_ptr.data(), col_idx.data(), vals.data(), &Amat);

        // Create vectors
        Vec bvec, xvec;
        VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n, b.data(), &bvec);
        x.resize(n, 0.0);
        VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n, x.data(), &xvec);

        // Create KSP solver
        KSP ksp;
        KSPCreate(PETSC_COMM_SELF, &ksp);
        KSPSetOperators(ksp, Amat, Amat);

        // Set solver type
        switch (ksp_type_) {
            case PETScKSPType::CG:      KSPSetType(ksp, KSPCG); break;
            case PETScKSPType::GMRES:   KSPSetType(ksp, KSPGMRES); break;
            case PETScKSPType::BCGS:    KSPSetType(ksp, KSPBCGS); break;
            case PETScKSPType::PREONLY: KSPSetType(ksp, KSPPREONLY); break;
        }

        // Set preconditioner
        PC pc;
        KSPGetPC(ksp, &pc);
        switch (pc_type_) {
            case PETScPCType::None:      PCSetType(pc, PCNONE); break;
            case PETScPCType::Jacobi:    PCSetType(pc, PCJACOBI); break;
            case PETScPCType::ILU:       PCSetType(pc, PCILU); break;
            case PETScPCType::ICC:       PCSetType(pc, PCICC); break;
            case PETScPCType::AMG:       PCSetType(pc, PCGAMG); break;
            case PETScPCType::LU:        PCSetType(pc, PCLU); break;
            case PETScPCType::Cholesky:  PCSetType(pc, PCCHOLESKY); break;
        }

        KSPSetTolerances(ksp, tolerance_, PETSC_DEFAULT, PETSC_DEFAULT, max_iterations_);

        if (from_options_) {
            KSPSetFromOptions(ksp);
        }

        // Solve
        KSPSolve(ksp, bvec, xvec);

        // Check convergence
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp, &reason);
        result.converged = (reason > 0);

        PetscInt its;
        KSPGetIterationNumber(ksp, &its);
        result.iterations = static_cast<int>(its);

        PetscReal rnorm;
        KSPGetResidualNorm(ksp, &rnorm);
        result.residual = static_cast<Real>(rnorm);

        // Cleanup
        KSPDestroy(&ksp);
        VecDestroy(&bvec);
        VecDestroy(&xvec);
        MatDestroy(&Amat);

        return result;
    }

private:
    PETScKSPType ksp_type_ = PETScKSPType::CG;
    PETScPCType pc_type_ = PETScPCType::Jacobi;
    bool from_options_ = false;
};

} // namespace solver
} // namespace nxs

#endif // NEXUSSIM_HAVE_PETSC
