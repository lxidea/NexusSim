#pragma once

/**
 * @file arc_length_solver.hpp
 * @brief Crisfield's cylindrical arc-length method for nonlinear equilibrium path tracing
 *
 * Traces load-displacement paths past limit points (snap-through buckling).
 * Uses bordering technique for corrector with quadratic constraint equation.
 */

#include <nexussim/solver/implicit_solver.hpp>
#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include <algorithm>

namespace nxs {
namespace solver {

struct PathPoint {
    Real load_factor = 0.0;
    std::vector<Real> displacement;
    int iterations = 0;
};

struct ArcLengthResult {
    bool converged = false;
    int total_steps = 0;
    int total_iterations = 0;
    Real final_load_factor = 0.0;
    std::vector<PathPoint> path;
};

/**
 * @brief Crisfield's cylindrical arc-length solver
 *
 * Solves: F_int(u) = lambda * F_ref
 * where lambda is the load factor and F_ref is the reference load pattern.
 *
 * Algorithm per step:
 * 1. Predictor: tangent prediction along arc
 * 2. Corrector: bordering technique with cylindrical constraint
 * 3. Adapt: adjust arc-length based on convergence rate
 */
class ArcLengthSolver {
public:
    using InternalForceFunction = std::function<void(const std::vector<Real>& u,
                                                      std::vector<Real>& F_int)>;
    using TangentFunction = std::function<void(const std::vector<Real>& u,
                                                SparseMatrix& K_t)>;

    ArcLengthSolver() : linear_solver_(std::make_unique<DirectSolver>()) {}

    void set_internal_force_function(InternalForceFunction func) {
        compute_internal_force_ = std::move(func);
    }

    void set_tangent_function(TangentFunction func) {
        compute_tangent_ = std::move(func);
    }

    void set_reference_load(const std::vector<Real>& F_ref) {
        F_ref_ = F_ref;
    }

    void set_arc_length(Real delta_l) { delta_l_ = delta_l; }
    void set_psi(Real psi) { psi_ = psi; }
    void set_desired_iterations(int n) { n_desired_ = n; }
    void set_arc_length_bounds(Real min_dl, Real max_dl) {
        min_dl_ = min_dl;
        max_dl_ = max_dl;
    }
    void set_max_steps(int n) { max_steps_ = n; }
    void set_max_corrections(int n) { max_corrections_ = n; }
    void set_tolerance(Real tol) { tolerance_ = tol; }
    void set_verbose(bool v) { verbose_ = v; }

    void set_linear_solver(LinearSolverType type) {
        switch (type) {
            case LinearSolverType::ConjugateGradient: {
                auto cg = std::make_unique<CGSolver>();
                cg->set_preconditioner(true);
                linear_solver_ = std::move(cg);
                break;
            }
            case LinearSolverType::DirectLU:
                linear_solver_ = std::make_unique<DirectSolver>();
                break;
            default:
                linear_solver_ = std::make_unique<DirectSolver>();
        }
    }

    /**
     * @brief Main entry point: trace equilibrium path
     * @param u Initial displacement (modified in place)
     * @param lambda Initial load factor (modified in place)
     * @param lambda_max Maximum load factor to reach
     */
    ArcLengthResult solve(std::vector<Real>& u, Real& lambda, Real lambda_max) {
        ArcLengthResult result;
        size_t ndof = u.size();

        if (F_ref_.size() != ndof) {
            if (verbose_) std::cout << "Arc-length: F_ref size mismatch\n";
            return result;
        }

        // F_ref norm for constraint scaling
        Real F_ref_norm = vector_norm(F_ref_);
        if (F_ref_norm < 1e-30) {
            if (verbose_) std::cout << "Arc-length: zero reference load\n";
            return result;
        }

        // Store initial state as first path point
        {
            PathPoint pp;
            pp.load_factor = lambda;
            pp.displacement = u;
            pp.iterations = 0;
            result.path.push_back(pp);
        }

        // Previous increment direction (for sign detection)
        std::vector<Real> delta_u_prev(ndof, 0.0);
        Real delta_lambda_prev = 1.0;  // Positive initial direction

        Real current_dl = delta_l_;

        for (int step = 0; step < max_steps_; ++step) {
            // ======== PREDICTOR ========
            // Solve K_t * du_t = F_ref
            compute_tangent_(u, K_tangent_);

            std::vector<Real> du_t(ndof);
            auto lin_result = linear_solver_->solve(K_tangent_, F_ref_, du_t);
            if (!lin_result.converged) {
                if (verbose_) std::cout << "Arc-length step " << step << ": predictor solve failed\n";
                break;
            }

            // Compute predictor load increment
            Real du_t_norm_sq = dot(du_t, du_t);
            Real denom = std::sqrt(du_t_norm_sq + psi_ * psi_ * F_ref_norm * F_ref_norm);
            if (denom < 1e-30) break;

            Real delta_lambda = current_dl / denom;

            // Sign: follow previous direction
            if (step > 0) {
                Real sign_check = dot(du_t, delta_u_prev) + psi_ * psi_ * F_ref_norm * F_ref_norm * delta_lambda_prev;
                if (sign_check < 0.0) delta_lambda = -delta_lambda;
            }

            // Predicted increments
            std::vector<Real> Delta_u(ndof);
            for (size_t i = 0; i < ndof; ++i) {
                Delta_u[i] = delta_lambda * du_t[i];
            }
            Real Delta_lambda = delta_lambda;

            // Apply predictor
            for (size_t i = 0; i < ndof; ++i) {
                u[i] += Delta_u[i];
            }
            lambda += Delta_lambda;

            // ======== CORRECTOR (bordering technique) ========
            int corrections = 0;
            bool step_converged = false;

            for (int iter = 0; iter < max_corrections_; ++iter) {
                // Compute residual: R = F_int(u) - lambda * F_ref
                std::vector<Real> R(ndof);
                compute_internal_force_(u, R);
                for (size_t i = 0; i < ndof; ++i) {
                    R[i] -= lambda * F_ref_[i];
                }

                Real R_norm = vector_norm(R);

                // Check convergence
                Real ref_force = std::abs(lambda) * F_ref_norm;
                if (ref_force < 1e-14) ref_force = F_ref_norm;
                Real rel_residual = R_norm / ref_force;

                if (rel_residual < tolerance_ || R_norm < 1e-14) {
                    step_converged = true;
                    corrections = iter;
                    break;
                }

                // Solve K_t * du_R = -R and K_t * du_F = F_ref
                compute_tangent_(u, K_tangent_);

                std::vector<Real> neg_R(ndof);
                for (size_t i = 0; i < ndof; ++i) neg_R[i] = -R[i];

                std::vector<Real> du_R(ndof), du_F(ndof);
                auto res1 = linear_solver_->solve(K_tangent_, neg_R, du_R);
                auto res2 = linear_solver_->solve(K_tangent_, F_ref_, du_F);

                if (!res1.converged || !res2.converged) {
                    if (verbose_) std::cout << "  Corrector solve failed at iter " << iter << "\n";
                    break;
                }

                // Solve quadratic constraint for d_lambda
                // ||Delta_u + du_R + d_lambda * du_F||^2 + psi^2 * (Delta_lambda + d_lambda)^2 * ||F_ref||^2 = dl^2
                // Expanding: a * d_lambda^2 + b * d_lambda + c = 0
                Real a = dot(du_F, du_F) + psi_ * psi_ * F_ref_norm * F_ref_norm;

                std::vector<Real> Delta_u_plus_duR(ndof);
                for (size_t i = 0; i < ndof; ++i) {
                    Delta_u_plus_duR[i] = Delta_u[i] + du_R[i];
                }

                Real b = 2.0 * dot(Delta_u_plus_duR, du_F) + 2.0 * psi_ * psi_ * Delta_lambda * F_ref_norm * F_ref_norm;
                Real c = dot(Delta_u_plus_duR, Delta_u_plus_duR) + psi_ * psi_ * Delta_lambda * Delta_lambda * F_ref_norm * F_ref_norm - current_dl * current_dl;

                Real discriminant = b * b - 4.0 * a * c;
                Real d_lambda;

                if (discriminant < 0.0) {
                    // Use linear approximation (normal plane correction)
                    d_lambda = -c / b;
                } else {
                    Real sqrt_disc = std::sqrt(discriminant);
                    Real root1 = (-b + sqrt_disc) / (2.0 * a);
                    Real root2 = (-b - sqrt_disc) / (2.0 * a);

                    // Choose root that follows the path (smaller angle with current direction)
                    std::vector<Real> trial1(ndof), trial2(ndof);
                    for (size_t i = 0; i < ndof; ++i) {
                        trial1[i] = Delta_u[i] + du_R[i] + root1 * du_F[i];
                        trial2[i] = Delta_u[i] + du_R[i] + root2 * du_F[i];
                    }
                    Real cos1 = dot(trial1, Delta_u);
                    Real cos2 = dot(trial2, Delta_u);

                    d_lambda = (cos1 > cos2) ? root1 : root2;
                }

                // Update
                for (size_t i = 0; i < ndof; ++i) {
                    Real du_i = du_R[i] + d_lambda * du_F[i];
                    u[i] += du_i;
                    Delta_u[i] += du_i;
                }
                Delta_lambda += d_lambda;
                lambda += d_lambda;

                corrections = iter + 1;
            }

            if (!step_converged) {
                if (verbose_) std::cout << "Arc-length step " << step << ": corrector did not converge\n";
                // Revert and stop
                for (size_t i = 0; i < ndof; ++i) {
                    u[i] -= Delta_u[i];
                }
                lambda -= Delta_lambda;
                break;
            }

            // Record path point
            PathPoint pp;
            pp.load_factor = lambda;
            pp.displacement = u;
            pp.iterations = corrections;
            result.path.push_back(pp);
            result.total_steps = step + 1;
            result.total_iterations += corrections;

            if (verbose_) {
                std::cout << "Arc-length step " << step + 1
                          << ": lambda = " << lambda
                          << ", |u| = " << vector_norm(u)
                          << ", iters = " << corrections
                          << ", dl = " << current_dl << "\n";
            }

            // Store direction for next step
            delta_u_prev = Delta_u;
            delta_lambda_prev = Delta_lambda;

            // ======== ADAPTIVE STEP SIZE ========
            if (corrections > 0 && n_desired_ > 0) {
                Real ratio = std::sqrt(static_cast<Real>(n_desired_) / corrections);
                ratio = std::max(0.25, std::min(ratio, 4.0));  // Clamp ratio
                current_dl *= ratio;
            }
            current_dl = std::max(min_dl_, std::min(current_dl, max_dl_));

            // Check termination
            if (std::abs(lambda) >= std::abs(lambda_max)) {
                result.converged = true;
                break;
            }
        }

        result.final_load_factor = lambda;
        if (result.total_steps > 0 && result.path.size() > 1) {
            result.converged = true;
        }

        return result;
    }

private:
    static Real dot(const std::vector<Real>& a, const std::vector<Real>& b) {
        Real sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
        return sum;
    }

    static Real vector_norm(const std::vector<Real>& v) {
        return std::sqrt(dot(v, v));
    }

    InternalForceFunction compute_internal_force_;
    TangentFunction compute_tangent_;
    std::vector<Real> F_ref_;
    SparseMatrix K_tangent_;
    std::unique_ptr<LinearSolver> linear_solver_;

    Real delta_l_ = 1.0;
    Real psi_ = 0.0;           // 0 = cylindrical
    int n_desired_ = 5;
    Real min_dl_ = 1e-6;
    Real max_dl_ = 10.0;
    int max_steps_ = 100;
    int max_corrections_ = 20;
    Real tolerance_ = 1e-6;
    bool verbose_ = false;
};

} // namespace solver
} // namespace nxs
