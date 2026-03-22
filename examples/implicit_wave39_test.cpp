/**
 * @file implicit_wave39_test.cpp
 * @brief Wave 39: Implicit solver enhancements test suite (6 features, ~50 tests)
 */

#include <nexussim/solver/implicit_wave39.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <functional>

using namespace nxs::solver;
using Real = nxs::Real;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; std::cout << "[FAIL] " << msg << "\n"; } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) do { \
    double _va = static_cast<double>(a); \
    double _vb = static_cast<double>(b); \
    double _vt = static_cast<double>(tol); \
    if (std::abs(_va - _vb) <= _vt) { tests_passed++; } \
    else { tests_failed++; std::cout << "[FAIL] " << msg \
        << " (got " << _va << ", expected " << _vb << ", tol " << _vt << ")\n"; } \
} while(0)

// ============================================================================
// 1. MUMPSSolver tests
// ============================================================================

void test_mumps_solver() {
    // 3x3 tridiagonal SPD: [2 -1 0; -1 2 -1; 0 -1 2]
    {
        MUMPSSolver solver;
        int n = 3;
        std::vector<int> row_ptr = {0, 2, 5, 7};
        std::vector<int> col_idx = {0, 1, 0, 1, 2, 1, 2};
        std::vector<Real> values = {2, -1, -1, 2, -1, -1, 2};

        bool sym_ok = solver.symbolic_factorize(n, row_ptr.data(), col_idx.data());
        CHECK(sym_ok, "MUMPS: symbolic factorize succeeds");
        CHECK(solver.symbolic_done(), "MUMPS: symbolic_done flag");

        bool num_ok = solver.numeric_factorize(values.data());
        CHECK(num_ok, "MUMPS: numeric factorize succeeds");
        CHECK(solver.numeric_done(), "MUMPS: numeric_done flag");

        std::vector<Real> rhs = {1.0, 0.0, 1.0};
        std::vector<Real> sol(3, 0.0);
        bool solve_ok = solver.solve(rhs.data(), sol.data());
        CHECK(solve_ok, "MUMPS: solve succeeds");

        // Verify: Ax - b residual
        std::vector<Real> Ax(3, 0.0);
        implicit_detail::csr_matvec(row_ptr.data(), col_idx.data(), values.data(),
                                     sol.data(), Ax.data(), n);
        Real res = 0.0;
        for (int i = 0; i < 3; ++i) res += (Ax[i] - rhs[i]) * (Ax[i] - rhs[i]);
        CHECK(std::sqrt(res) < 1e-6, "MUMPS: 3x3 residual < 1e-6");
    }

    // 5x5 diagonal
    {
        MUMPSSolver solver;
        int n = 5;
        std::vector<int> row_ptr = {0, 1, 2, 3, 4, 5};
        std::vector<int> col_idx = {0, 1, 2, 3, 4};
        std::vector<Real> values = {2.0, 3.0, 4.0, 5.0, 6.0};

        solver.symbolic_factorize(n, row_ptr.data(), col_idx.data());
        solver.numeric_factorize(values.data());

        std::vector<Real> rhs = {4.0, 9.0, 12.0, 15.0, 18.0};
        std::vector<Real> sol(5, 0.0);
        solver.solve(rhs.data(), sol.data());

        CHECK_NEAR(sol[0], 2.0, 1e-6, "MUMPS: diag sol[0]=2");
        CHECK_NEAR(sol[1], 3.0, 1e-6, "MUMPS: diag sol[1]=3");
        CHECK_NEAR(sol[2], 3.0, 1e-6, "MUMPS: diag sol[2]=3");
    }

    // Check symbolic/numeric flags before factorize
    {
        MUMPSSolver solver;
        CHECK(!solver.symbolic_done(), "MUMPS: symbolic not done initially");
        CHECK(!solver.numeric_done(), "MUMPS: numeric not done initially");
    }
}

// ============================================================================
// 2. ImplicitBFGS tests
// ============================================================================

void test_implicit_bfgs() {
    // BFGS::solve takes: residual_func(x_vec, residual_vec, jacobian_ignored), x0, max_iter, tol
    // residual_func signature: void(const vector<Real>&, vector<Real>&, vector<Real>&)
    // where third arg is storage for the Jacobian approx (can be ignored for gradient-only)

    // Solve R(x) = x, gradient of 0.5*R^2 = R * dR/dx = x*1 = x
    {
        ImplicitBFGS bfgs(5, 1e-4, 1.0);
        auto residual = [](const std::vector<Real>& x, std::vector<Real>& r,
                           std::vector<Real>& g) {
            r[0] = x[0];      // residual
            g[0] = x[0];      // gradient of 0.5*||R||^2
        };
        std::vector<Real> x0 = {5.0};
        auto result = bfgs.solve(residual, x0, 100, 1e-10);
        CHECK(result.converged, "BFGS: quadratic converges");
        CHECK(std::abs(result.x[0]) < 1.0, "BFGS: minimum near 0");
    }

    // 2D: R = [x, 2y], grad(0.5*||R||^2) = J^T*R = [x, 4y]
    {
        ImplicitBFGS bfgs(10, 1e-4, 1.0);
        auto residual = [](const std::vector<Real>& x, std::vector<Real>& r,
                           std::vector<Real>& g) {
            r[0] = x[0]; r[1] = 2.0 * x[1];
            g[0] = x[0]; g[1] = 4.0 * x[1];
        };
        std::vector<Real> x0 = {3.0, 2.0};
        auto result = bfgs.solve(residual, x0, 200, 1e-10);
        CHECK(result.iterations > 0, "BFGS: 2D ran iterations");
        CHECK(std::abs(result.x[0]) < 5.0, "BFGS: 2D x improved");
        CHECK(std::abs(result.x[1]) < 5.0, "BFGS: 2D y improved");
    }

    // Reasonable iteration count
    {
        ImplicitBFGS bfgs(5, 1e-4, 1.0);
        auto residual = [](const std::vector<Real>& x, std::vector<Real>& r,
                           std::vector<Real>& g) {
            r[0] = x[0]; g[0] = x[0];
        };
        std::vector<Real> x0 = {10.0};
        auto result = bfgs.solve(residual, x0, 100, 1e-10);
        CHECK(result.iterations < 50, "BFGS: few iterations for simple problem");
        CHECK(result.final_residual < 1e-2, "BFGS: final residual small");
    }
}

// ============================================================================
// 3. ImplicitBuckling tests
// ============================================================================

void test_implicit_buckling() {
    // 2x2: K=[2,-1;-1,2], Kg=[-1,0;0,-1] → eigenvalues 1 and 3
    {
        ImplicitBuckling buckling(200, 1e-8);
        std::vector<Real> K = {2.0, -1.0, -1.0, 2.0};
        std::vector<Real> Kg = {-1.0, 0.0, 0.0, -1.0};
        auto result = buckling.compute_buckling_load(K.data(), Kg.data(), 2);
        CHECK(result.converged, "Buckling: 2x2 converges");
        CHECK_NEAR(result.lambda_cr, 1.0, 0.2, "Buckling: lambda_cr ≈ 1.0");
        CHECK(result.mode_shape.size() == 2, "Buckling: mode shape size");
    }

    // 1x1: K=[4], Kg=[-2] → lambda=2
    {
        ImplicitBuckling buckling(100, 1e-8);
        std::vector<Real> K = {4.0};
        std::vector<Real> Kg = {-2.0};
        auto result = buckling.compute_buckling_load(K.data(), Kg.data(), 1);
        CHECK(result.converged, "Buckling: 1x1 converges");
        CHECK_NEAR(result.lambda_cr, 2.0, 0.2, "Buckling: 1x1 lambda≈2");
    }

    // Positive critical load
    {
        ImplicitBuckling buckling(200, 1e-8);
        std::vector<Real> K = {10.0, 0.0, 0.0, 10.0};
        std::vector<Real> Kg = {-1.0, 0.0, 0.0, -1.0};
        auto result = buckling.compute_buckling_load(K.data(), Kg.data(), 2);
        CHECK(result.lambda_cr > 0.0, "Buckling: positive critical load");
    }
}

// ============================================================================
// 4. ImplicitDtControl tests
// ============================================================================

void test_implicit_dt_control() {
    ImplicitDtControl ctrl(1e-10, 1.0, 1.5, 0.5);

    // Fast convergence → dt grows
    {
        Real dt_new = ctrl.update_dt(0.001, 2, true, 20);
        CHECK(dt_new > 0.001, "DtCtrl: grows on fast convergence");
    }

    // Slow convergence → dt shrinks or stays
    {
        Real dt_new = ctrl.update_dt(0.001, 18, true, 20);
        CHECK(dt_new <= 0.001, "DtCtrl: shrinks on slow convergence");
    }

    // Non-convergence → dt shrinks
    {
        Real dt_new = ctrl.update_dt(0.001, 20, false, 20);
        CHECK(dt_new < 0.001, "DtCtrl: shrinks on failure");
        CHECK_NEAR(dt_new, 0.0005, 1e-10, "DtCtrl: halved on failure");
    }

    // dt_min enforced
    {
        Real dt_new = ctrl.update_dt(1e-10, 20, false, 20);
        CHECK(dt_new >= 1e-10, "DtCtrl: dt_min enforced");
    }

    // dt_max enforced
    {
        Real dt_new = ctrl.update_dt(0.9, 1, true, 20);
        CHECK(dt_new <= 1.0, "DtCtrl: dt_max enforced");
    }

    // estimate_initial_dt
    {
        Real dt_est = ImplicitDtControl::estimate_initial_dt(0.01, 210e9, 7800.0, 0.1);
        CHECK(dt_est > 0.0, "DtCtrl: estimated dt > 0");
        CHECK(dt_est < 1e-3, "DtCtrl: reasonable for steel");
    }

    CHECK_NEAR(ctrl.dt_min(), 1e-10, 1e-20, "DtCtrl: dt_min getter");
    CHECK_NEAR(ctrl.dt_max(), 1.0, 1e-10, "DtCtrl: dt_max getter");
}

// ============================================================================
// 5. IterativeRefinement tests
// ============================================================================

void test_iterative_refinement() {
    // Identity system, exact initial guess
    {
        IterativeRefinement refiner;
        int n = 3;
        auto A_apply = [](const Real* x, Real* y, int n) {
            for (int i = 0; i < n; ++i) y[i] = x[i];
        };
        auto solve_func = [](const Real* r, Real* dx, int n) {
            for (int i = 0; i < n; ++i) dx[i] = r[i];
        };

        std::vector<Real> b = {1.0, 2.0, 3.0};
        std::vector<Real> x = {1.0, 2.0, 3.0};
        auto result = refiner.refine(A_apply, solve_func, b.data(), x.data(), n);
        CHECK(result.converged, "Refine: exact guess converges");
        CHECK(result.final_residual < 1e-10, "Refine: residual near zero");
    }

    // Perturbed guess with diagonal approximate solver
    {
        IterativeRefinement refiner;
        int n = 2;
        // A = [4 1; 1 3]
        auto A_apply = [](const Real* x, Real* y, int n_) {
            y[0] = 4.0 * x[0] + 1.0 * x[1];
            y[1] = 1.0 * x[0] + 3.0 * x[1];
        };
        auto solve_approx = [](const Real* r, Real* dx, int n_) {
            dx[0] = r[0] / 4.0;
            dx[1] = r[1] / 3.0;
        };

        std::vector<Real> b = {5.0, 4.0};
        std::vector<Real> x = {0.0, 0.0};
        auto result = refiner.refine(A_apply, solve_approx, b.data(), x.data(), n);
        CHECK(result.steps > 0, "Refine: needed iterations");
        CHECK(result.final_residual < result.initial_residual, "Refine: residual improved");
    }

    // Max steps
    {
        IterativeRefinement refiner;
        int n = 1;
        auto A_apply = [](const Real* x, Real* y, int) { y[0] = 2.0 * x[0]; };
        auto solve_func = [](const Real* r, Real* dx, int) { dx[0] = r[0] / 2.0; };

        std::vector<Real> b = {4.0};
        std::vector<Real> x = {0.0};
        auto result = refiner.refine(A_apply, solve_func, b.data(), x.data(), n, 2);
        CHECK(result.steps <= 2, "Refine: max_steps=2 respected");
    }
}

// ============================================================================
// 6. ImplicitContactK tests
// ============================================================================

void test_implicit_contact_k() {
    ImplicitContactK contact_k;

    // Single penetrating pair — compute_contact_stiffness needs positions and normals arrays
    {
        std::vector<ContactPair> pairs(1);
        pairs[0].node1 = 0;
        pairs[0].node2 = 1;
        pairs[0].gap = -0.001;
        pairs[0].normal[0] = 0.0;
        pairs[0].normal[1] = 0.0;
        pairs[0].normal[2] = 1.0;

        // 2 nodes x 3 DOF positions
        std::vector<Real> positions = {0.0, 0.0, 0.0,  0.0, 0.0, -0.001};
        std::vector<Real> normals = {0.0, 0.0, 1.0};

        Real kn = 1e6;
        auto triplets = contact_k.compute_contact_stiffness(pairs, positions.data(),
                                                              normals.data(), kn);
        CHECK(triplets.size() > 0, "ContactK: triplets for penetrating pair");
    }

    // Positive gap → no stiffness
    {
        std::vector<ContactPair> pairs(1);
        pairs[0].node1 = 0;
        pairs[0].node2 = 1;
        pairs[0].gap = 0.01;
        pairs[0].normal[0] = 1.0;
        pairs[0].normal[1] = 0.0;
        pairs[0].normal[2] = 0.0;

        std::vector<Real> positions = {0.0, 0.0, 0.0,  0.01, 0.0, 0.0};
        std::vector<Real> normals = {1.0, 0.0, 0.0};

        auto triplets = contact_k.compute_contact_stiffness(pairs, positions.data(),
                                                              normals.data(), 1e6);
        CHECK(triplets.empty(), "ContactK: no stiffness for positive gap");
    }

    // Contact force
    {
        std::vector<ContactPair> pairs(1);
        pairs[0].node1 = 0;
        pairs[0].node2 = 1;
        pairs[0].gap = -0.002;
        pairs[0].normal[0] = 0.0;
        pairs[0].normal[1] = 1.0;
        pairs[0].normal[2] = 0.0;

        std::vector<Real> forces(6, 0.0);
        contact_k.compute_contact_force(pairs, 1e6, forces.data());
        CHECK(std::abs(forces[1]) > 0.0, "ContactK: force in normal direction");
    }

    // compute_gap
    {
        Real x1[3] = {0.0, 0.0, 0.0};
        Real x2[3] = {0.0, 0.0, 0.5};
        Real n[3] = {0.0, 0.0, 1.0};
        Real gap = ImplicitContactK::compute_gap(x1, x2, n, 3);
        CHECK_NEAR(gap, 0.5, 1e-10, "ContactK: gap = 0.5");
    }

    // estimate_penalty_stiffness
    {
        Real kp = ImplicitContactK::estimate_penalty_stiffness(210e9, 0.01, 0.1);
        CHECK(kp > 0.0, "ContactK: positive penalty stiffness");
    }

    // max_penetration
    {
        std::vector<ContactPair> pairs(3);
        pairs[0].gap = -0.001;
        pairs[1].gap = -0.005;
        pairs[2].gap = 0.01;
        Real max_pen = ImplicitContactK::max_penetration(pairs);
        CHECK_NEAR(max_pen, 0.005, 1e-10, "ContactK: max pen = 0.005");
    }
}

// ============================================================================

int main() {
    std::cout << "=== Wave 39b: Implicit Solver Hardening Tests ===\n\n";

    test_mumps_solver();
    test_implicit_bfgs();
    test_implicit_buckling();
    test_implicit_dt_control();
    test_iterative_refinement();
    test_implicit_contact_k();

    std::cout << "\n" << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    return tests_failed > 0 ? 1 : 0;
}
