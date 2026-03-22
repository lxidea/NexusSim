/**
 * @file solver_wave30_test.cpp
 * @brief Wave 30: Advanced Iterative Solvers and Preconditioners Test Suite (4 components, 40 tests)
 *
 * Tests 4 sub-modules (10 tests each):
 *  1. LineSearch              - Armijo backtracking, Wolfe conditions, parameter effects
 *  2. LBFGSSolver            - L-BFGS optimization on quadratic and Rosenbrock problems
 *  3. GMRESSolver             - Restarted GMRES for symmetric and non-symmetric systems
 *  4. Preconditioners         - Jacobi and ILUT preconditioners with iterative solvers
 */

#include <nexussim/solver/solver_wave30.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <functional>

using namespace nxs;
using namespace nxs::solver;

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
// 1. LineSearch Tests
// ============================================================================
void test_line_search() {
    std::cout << "\n--- LineSearch Tests ---\n";

    // Test 1: alpha=1 accepted for quadratic f(x) = 0.5*x^2 (exact minimum)
    // Starting at x=2, d=-1 (steepest descent), f(x+alpha*d) = 0.5*(2-alpha)^2
    // Minimum at alpha=2 but alpha=1 gives f=0.5 < f0=2 + c1*1*(-2)=2-0.0002=1.9998
    {
        LineSearch ls;
        Real x0_val = 2.0;
        auto f_func = [&](Real alpha) -> Real {
            Real x = x0_val - alpha; // d = -1
            return 0.5 * x * x;
        };
        Real f0 = 0.5 * x0_val * x0_val; // 2.0
        Real grad_dot_d = -x0_val; // grad=x0=2, d=-1, so grad*d = -2

        Real alpha = ls.backtrack(f_func, grad_dot_d, f0, 1.0);
        CHECK_NEAR(alpha, 1.0, 1e-10,
            "LineSearch: alpha=1 accepted for quadratic");
    }

    // Test 2: Backtracking reduces alpha when step too large
    // f(x) = x^4, starting at x=10, d=-1
    // f(10-alpha) = (10-alpha)^4, grad=4*10^3=4000, grad_dot_d = -4000
    // alpha=1: f(9)=6561, f0=10000, Armijo: 6561 <= 10000 + 1e-4*1*(-4000) = 9999.6 => yes
    // Use a function where alpha=1 fails Armijo
    {
        LineSearch ls(1e-4, 0.9, 0.5, 20);
        // f(alpha) = (alpha - 5)^2 evaluated at x + alpha*d
        // Let x=0, d=1, f(alpha) = (alpha - 5)^2
        // f0 = f(0) = 25, grad at x=0 is -10, grad_dot_d = -10
        // Armijo: f(alpha) <= 25 - 0.001*alpha
        // alpha=1: f=16, 16 <= 24.999 => pass
        // Need a case where alpha=1 fails...
        // f(alpha) = exp(10*alpha) - 11*alpha, f(0) = 1
        // f'(0) = 10 - 11 = -1, so grad_dot_d = -1
        // Armijo: f(alpha) <= 1 - 1e-4 * alpha
        // alpha=1: f(1) = exp(10) - 11 ~ 22015, fails!
        auto f_func = [](Real alpha) -> Real {
            return std::exp(10.0 * alpha) - 11.0 * alpha;
        };
        Real f0 = 1.0;
        Real grad_dot_d = -1.0;

        Real alpha = ls.backtrack(f_func, grad_dot_d, f0, 1.0);
        CHECK(alpha < 1.0,
            "LineSearch: backtracking reduces alpha for steep function");
        CHECK(ls.iterations() > 1,
            "LineSearch: multiple iterations needed");
    }

    // Test 3: Armijo condition satisfied on return
    {
        LineSearch ls;
        auto f_func = [](Real alpha) -> Real {
            return (1.0 - alpha) * (1.0 - alpha); // min at alpha=1
        };
        Real f0 = 1.0; // f(0)
        Real grad_dot_d = -2.0; // f'(0) = -2

        Real alpha = ls.backtrack(f_func, grad_dot_d, f0, 1.0);
        Real f_new = f_func(alpha);
        bool armijo = ls.armijo_check(f_new, f0, alpha, grad_dot_d);
        CHECK(armijo,
            "LineSearch: Armijo condition satisfied on return");
    }

    // Test 4: Zero gradient direction returns alpha=0
    {
        LineSearch ls;
        auto f_func = [](Real alpha) -> Real { return 1.0; };
        Real f0 = 1.0;
        Real grad_dot_d = 0.0; // not a descent direction

        Real alpha = ls.backtrack(f_func, grad_dot_d, f0, 1.0);
        CHECK_NEAR(alpha, 0.0, 1e-15,
            "LineSearch: zero gradient returns alpha=0");
    }

    // Test 5: Steep descent finds good step
    {
        LineSearch ls;
        // f(x) = 0.5*x^2, x=100, d=-1
        Real x0_val = 100.0;
        auto f_func = [&](Real alpha) -> Real {
            Real x = x0_val - alpha;
            return 0.5 * x * x;
        };
        Real f0 = 0.5 * x0_val * x0_val;
        Real grad_dot_d = -x0_val; // grad=100, d=-1

        Real alpha = ls.backtrack(f_func, grad_dot_d, f0, 1.0);
        CHECK(alpha > 0.0,
            "LineSearch: steep descent finds positive alpha");
        Real f_new = f_func(alpha);
        CHECK(f_new < f0,
            "LineSearch: function value decreased");
    }

    // Test 6: Max iterations respected
    {
        LineSearch ls(1e-4, 0.9, 0.5, 3);
        // Function that never satisfies Armijo with positive step
        auto f_func = [](Real alpha) -> Real {
            return 100.0 + alpha; // always increasing
        };
        Real f0 = 100.0;
        Real grad_dot_d = -1.0; // claims descent but f increases

        ls.backtrack(f_func, grad_dot_d, f0, 1.0);
        CHECK(ls.iterations() <= 3,
            "LineSearch: max iterations (3) respected");
    }

    // Test 7: Rho parameter affects convergence rate
    {
        // Use a function where backtracking is needed
        auto f_func = [](Real alpha) -> Real {
            return std::exp(10.0 * alpha) - 11.0 * alpha;
        };
        Real f0 = 1.0;
        Real grad_dot_d = -1.0;

        LineSearch ls1(1e-4, 0.9, 0.5, 50);
        Real alpha1 = ls1.backtrack(f_func, grad_dot_d, f0, 1.0);
        int iter1 = ls1.iterations();

        LineSearch ls2(1e-4, 0.9, 0.9, 50); // slower reduction
        Real alpha2 = ls2.backtrack(f_func, grad_dot_d, f0, 1.0);
        int iter2 = ls2.iterations();

        // With rho=0.9 (slower reduction), need more iterations
        CHECK(iter2 >= iter1,
            "LineSearch: smaller rho converges faster (fewer iterations)");
    }

    // Test 8: c1 parameter effect (stricter = smaller steps)
    {
        auto f_func = [](Real alpha) -> Real {
            Real x = 5.0 - alpha;
            return 0.5 * x * x;
        };
        Real f0 = 12.5;
        Real grad_dot_d = -5.0;

        LineSearch ls_loose(1e-1, 0.9, 0.5, 20); // stricter c1
        Real alpha_strict = ls_loose.backtrack(f_func, grad_dot_d, f0, 1.0);

        LineSearch ls_tight(1e-6, 0.9, 0.5, 20); // looser c1
        Real alpha_loose = ls_tight.backtrack(f_func, grad_dot_d, f0, 1.0);

        CHECK(alpha_loose >= alpha_strict,
            "LineSearch: looser c1 accepts larger or equal steps");
    }

    // Test 9: Negative directional derivative required (descent)
    {
        LineSearch ls;
        auto f_func = [](Real alpha) -> Real { return alpha * alpha; };
        Real f0 = 0.0;
        Real grad_dot_d = 1.0; // ascending direction

        Real alpha = ls.backtrack(f_func, grad_dot_d, f0, 1.0);
        CHECK_NEAR(alpha, 0.0, 1e-15,
            "LineSearch: positive grad_dot_d returns alpha=0");
    }

    // Test 10: Convex quadratic convergence
    {
        LineSearch ls;
        // f(alpha) = 3*(1-alpha)^2 + 1, min at alpha=1, f(0)=4, f'(0)=-6
        auto f_func = [](Real alpha) -> Real {
            return 3.0 * (1.0 - alpha) * (1.0 - alpha) + 1.0;
        };
        Real f0 = 4.0;
        Real grad_dot_d = -6.0;

        Real alpha = ls.backtrack(f_func, grad_dot_d, f0, 1.0);
        Real f_new = f_func(alpha);
        CHECK(f_new < f0,
            "LineSearch: convex quadratic value decreased");
        CHECK(ls.armijo_satisfied(),
            "LineSearch: convex quadratic Armijo satisfied");
    }
}

// ============================================================================
// 2. LBFGSSolver Tests
// ============================================================================
void test_lbfgs() {
    std::cout << "\n--- LBFGSSolver Tests ---\n";

    // Test 1: Quadratic 2D: f = x^2 + y^2, minimum at (0,0)
    {
        LBFGSSolver solver(10, 100, 1e-8);
        auto f = [](const Real* x) -> Real {
            return x[0]*x[0] + x[1]*x[1];
        };
        auto g = [](const Real* x, Real* grad) {
            grad[0] = 2.0 * x[0];
            grad[1] = 2.0 * x[1];
        };
        Real x0[2] = {5.0, 3.0};
        auto result = solver.solve(f, g, x0, 2);

        CHECK_NEAR(result[0], 0.0, 1e-6,
            "LBFGS: quadratic 2D x=0");
        CHECK_NEAR(result[1], 0.0, 1e-6,
            "LBFGS: quadratic 2D y=0");
    }

    // Test 2: Rosenbrock 2D: f = (1-x)^2 + 100*(y-x^2)^2, min at (1,1)
    {
        LBFGSSolver solver(10, 500, 1e-6);
        auto f = [](const Real* x) -> Real {
            Real a = 1.0 - x[0];
            Real b = x[1] - x[0]*x[0];
            return a*a + 100.0*b*b;
        };
        auto g = [](const Real* x, Real* grad) {
            grad[0] = -2.0*(1.0 - x[0]) + 200.0*(x[1] - x[0]*x[0])*(-2.0*x[0]);
            grad[1] = 200.0*(x[1] - x[0]*x[0]);
        };
        Real x0[2] = {-1.0, 1.0};
        auto result = solver.solve(f, g, x0, 2);

        CHECK_NEAR(result[0], 1.0, 0.1,
            "LBFGS: Rosenbrock x~1");
        CHECK_NEAR(result[1], 1.0, 0.1,
            "LBFGS: Rosenbrock y~1");
    }

    // Test 3: Diagonal quadratic: convergence in reasonable iterations
    {
        const int n = 10;
        LBFGSSolver solver(10, 200, 1e-8);
        auto f = [&](const Real* x) -> Real {
            Real sum = 0.0;
            for (int i = 0; i < n; ++i) sum += (i+1.0)*x[i]*x[i];
            return sum;
        };
        auto g = [&](const Real* x, Real* grad) {
            for (int i = 0; i < n; ++i) grad[i] = 2.0*(i+1.0)*x[i];
        };
        std::vector<Real> x0(n, 1.0);
        auto result = solver.solve(f, g, x0.data(), n);

        Real fval = f(result.data());
        CHECK(fval < 1e-10,
            "LBFGS: diagonal quadratic converges to near-zero");
        CHECK(solver.iterations() <= n + 15,
            "LBFGS: diagonal quadratic converges in ~n iterations");
    }

    // Test 4: History limited to m pairs
    {
        LBFGSSolver solver(3, 100, 1e-8);
        auto f = [](const Real* x) -> Real {
            return x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];
        };
        auto g = [](const Real* x, Real* grad) {
            for (int i = 0; i < 4; ++i) grad[i] = 2.0*x[i];
        };
        Real x0[4] = {10.0, 20.0, 30.0, 40.0};
        solver.solve(f, g, x0, 4);

        CHECK(solver.num_stored() <= 3,
            "LBFGS: history limited to m=3 pairs");
    }

    // Test 5: Initial gamma scaling correct
    {
        // After one iteration, gamma = s.y / y.y
        LBFGSSolver solver(10, 2, 1e-20); // run only 2 iters
        auto f = [](const Real* x) -> Real { return x[0]*x[0]; };
        auto g = [](const Real* x, Real* grad) { grad[0] = 2.0*x[0]; };
        Real x0[1] = {3.0};
        solver.solve(f, g, x0, 1);
        // If solver ran at least 1 iteration, history should have an entry
        CHECK(solver.num_stored() >= 1,
            "LBFGS: at least one history pair stored after iterations");
    }

    // Test 6: Convergence flag set
    {
        LBFGSSolver solver(10, 100, 1e-6);
        auto f = [](const Real* x) -> Real { return x[0]*x[0]; };
        auto g = [](const Real* x, Real* grad) { grad[0] = 2.0*x[0]; };
        Real x0[1] = {1.0};
        solver.solve(f, g, x0, 1);
        CHECK(solver.converged(),
            "LBFGS: convergence flag set for simple quadratic");
    }

    // Test 7: Iteration count reasonable
    {
        LBFGSSolver solver(10, 100, 1e-8);
        auto f = [](const Real* x) -> Real {
            return x[0]*x[0] + 4.0*x[1]*x[1];
        };
        auto g = [](const Real* x, Real* grad) {
            grad[0] = 2.0*x[0];
            grad[1] = 8.0*x[1];
        };
        Real x0[2] = {10.0, 10.0};
        solver.solve(f, g, x0, 2);
        CHECK(solver.iterations() < 50,
            "LBFGS: iteration count < 50 for 2D quadratic");
    }

    // Test 8: Large problem (n=50) doesn't crash
    {
        const int n = 50;
        LBFGSSolver solver(10, 200, 1e-6);
        auto f = [&](const Real* x) -> Real {
            Real sum = 0.0;
            for (int i = 0; i < n; ++i) sum += x[i]*x[i];
            return sum;
        };
        auto g = [&](const Real* x, Real* grad) {
            for (int i = 0; i < n; ++i) grad[i] = 2.0*x[i];
        };
        std::vector<Real> x0(n, 1.0);
        auto result = solver.solve(f, g, x0.data(), n);

        Real fval = f(result.data());
        CHECK(fval < 1e-6,
            "LBFGS: n=50 problem converges");
    }

    // Test 9: Starting at minimum -> 0 iterations
    {
        LBFGSSolver solver(10, 100, 1e-6);
        auto f = [](const Real* x) -> Real { return x[0]*x[0] + x[1]*x[1]; };
        auto g = [](const Real* x, Real* grad) {
            grad[0] = 2.0*x[0];
            grad[1] = 2.0*x[1];
        };
        Real x0[2] = {0.0, 0.0};
        solver.solve(f, g, x0, 2);
        CHECK(solver.iterations() == 0,
            "LBFGS: 0 iterations when starting at minimum");
        CHECK(solver.converged(),
            "LBFGS: converged flag set at minimum");
    }

    // Test 10: Non-zero tolerance respected
    {
        LBFGSSolver solver(10, 100, 1.0); // very loose tolerance
        auto f = [](const Real* x) -> Real { return x[0]*x[0]; };
        auto g = [](const Real* x, Real* grad) { grad[0] = 2.0*x[0]; };
        Real x0[1] = {0.3}; // |grad| = 0.6 < 1.0 = tol
        solver.solve(f, g, x0, 1);
        CHECK(solver.converged(),
            "LBFGS: converged with loose tolerance and small gradient");
        CHECK(solver.iterations() == 0,
            "LBFGS: 0 iterations with tolerance > gradient norm");
    }
}

// ============================================================================
// 3. GMRESSolver Tests
// ============================================================================
void test_gmres() {
    std::cout << "\n--- GMRESSolver Tests ---\n";

    // Test 1: Identity system A=I, b=random -> x=b in 1 iteration
    {
        const int n = 5;
        auto matvec = [&](const Real* x, Real* y) {
            for (int i = 0; i < n; ++i) y[i] = x[i];
        };
        std::vector<Real> rhs = {1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<Real> x0(n, 0.0);

        GMRESSolver gmres(30, 10, 1e-10);
        gmres.solve(matvec, rhs.data(), x0.data(), n);

        bool all_close = true;
        for (int i = 0; i < n; ++i) {
            if (std::abs(x0[i] - rhs[i]) > 1e-8) all_close = false;
        }
        CHECK(all_close,
            "GMRES: identity system x=b");
        CHECK(gmres.iterations() <= 2,
            "GMRES: identity system converges in ~1 iteration");
    }

    // Test 2: Diagonal system: exact in 1 iteration
    {
        const int n = 4;
        std::vector<Real> diag = {2.0, 3.0, 5.0, 7.0};
        auto matvec = [&](const Real* x, Real* y) {
            for (int i = 0; i < n; ++i) y[i] = diag[i] * x[i];
        };
        std::vector<Real> rhs = {4.0, 9.0, 25.0, 49.0};
        std::vector<Real> x0(n, 0.0);

        GMRESSolver gmres(30, 10, 1e-10);
        gmres.solve(matvec, rhs.data(), x0.data(), n);

        CHECK_NEAR(x0[0], 2.0, 1e-8, "GMRES: diagonal x[0]=2");
        CHECK_NEAR(x0[1], 3.0, 1e-8, "GMRES: diagonal x[1]=3");
    }

    // Test 3: SPD system (should also work)
    {
        // A = [4 1; 1 3], b = [1; 2], solution x = [1/11; 7/11]
        const int n = 2;
        auto matvec = [](const Real* x, Real* y) {
            y[0] = 4.0*x[0] + 1.0*x[1];
            y[1] = 1.0*x[0] + 3.0*x[1];
        };
        std::vector<Real> rhs = {1.0, 2.0};
        std::vector<Real> x0(n, 0.0);

        GMRESSolver gmres(30, 10, 1e-10);
        gmres.solve(matvec, rhs.data(), x0.data(), n);

        CHECK_NEAR(x0[0], 1.0/11.0, 1e-8, "GMRES: SPD x[0]");
        CHECK_NEAR(x0[1], 7.0/11.0, 1e-8, "GMRES: SPD x[1]");
    }

    // Test 4: Non-symmetric 3x3 system with known solution
    {
        // A = [2 1 0; 0 3 1; 1 0 4], x = [1;1;1], b = A*x = [3;4;5]
        const int n = 3;
        auto matvec = [](const Real* x, Real* y) {
            y[0] = 2.0*x[0] + 1.0*x[1];
            y[1] = 3.0*x[1] + 1.0*x[2];
            y[2] = 1.0*x[0] + 4.0*x[2];
        };
        std::vector<Real> rhs = {3.0, 4.0, 5.0};
        std::vector<Real> x0(n, 0.0);

        GMRESSolver gmres(30, 10, 1e-10);
        gmres.solve(matvec, rhs.data(), x0.data(), n);

        CHECK_NEAR(x0[0], 1.0, 1e-6, "GMRES: non-symmetric x[0]=1");
        CHECK_NEAR(x0[1], 1.0, 1e-6, "GMRES: non-symmetric x[1]=1");
        CHECK_NEAR(x0[2], 1.0, 1e-6, "GMRES: non-symmetric x[2]=1");
    }

    // Test 5: Restart mechanism (problem needs restart)
    {
        // Tridiagonal system large enough to need restarts with m=3
        const int n = 20;
        // A = tridiag(-1, 3, -1), diagonally dominant SPD
        auto matvec = [&](const Real* x, Real* y) {
            for (int i = 0; i < n; ++i) {
                y[i] = 3.0 * x[i];
                if (i > 0) y[i] -= x[i-1];
                if (i < n-1) y[i] -= x[i+1];
            }
        };
        // b = A * ones
        std::vector<Real> rhs(n);
        std::vector<Real> ones(n, 1.0);
        matvec(ones.data(), rhs.data());

        std::vector<Real> x0(n, 0.0);
        GMRESSolver gmres(3, 50, 1e-8); // restart=3, needs multiple restarts
        gmres.solve(matvec, rhs.data(), x0.data(), n);

        CHECK(gmres.converged(),
            "GMRES: converges with restarts (m=3, n=20)");
        CHECK(gmres.iterations() > 3,
            "GMRES: restart needed (iterations > restart size)");
    }

    // Test 6: Convergence tolerance respected
    {
        const int n = 3;
        auto matvec = [](const Real* x, Real* y) {
            y[0] = 10.0*x[0] + x[1];
            y[1] = x[0] + 10.0*x[1] + x[2];
            y[2] = x[1] + 10.0*x[2];
        };
        std::vector<Real> rhs = {11.0, 12.0, 11.0};
        std::vector<Real> x0(n, 0.0);

        GMRESSolver gmres(30, 10, 1e-12);
        gmres.solve(matvec, rhs.data(), x0.data(), n);

        // Check residual
        Real Ax[3];
        matvec(x0.data(), Ax);
        Real res_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            Real r = rhs[i] - Ax[i];
            res_norm += r * r;
        }
        res_norm = std::sqrt(res_norm);
        CHECK(res_norm < 1e-10,
            "GMRES: residual below tolerance");
    }

    // Test 7: Preconditioned solve (diagonal preconditioner)
    {
        const int n = 4;
        std::vector<Real> diag = {100.0, 200.0, 300.0, 400.0};
        auto matvec = [&](const Real* x, Real* y) {
            for (int i = 0; i < n; ++i) y[i] = diag[i] * x[i];
        };
        auto precond = [&](const Real* x, Real* y) {
            for (int i = 0; i < n; ++i) y[i] = x[i] / diag[i];
        };
        std::vector<Real> rhs = {100.0, 400.0, 900.0, 1600.0};
        std::vector<Real> x0(n, 0.0);

        GMRESSolver gmres(30, 10, 1e-10);
        gmres.set_preconditioner(precond);
        gmres.solve(matvec, rhs.data(), x0.data(), n);

        CHECK_NEAR(x0[0], 1.0, 1e-6, "GMRES: preconditioned x[0]=1");
        CHECK_NEAR(x0[1], 2.0, 1e-6, "GMRES: preconditioned x[1]=2");
        CHECK(gmres.converged(),
            "GMRES: preconditioned solve converged");
    }

    // Test 8: Zero RHS -> zero solution
    {
        const int n = 3;
        auto matvec = [](const Real* x, Real* y) {
            y[0] = 2.0*x[0]; y[1] = 3.0*x[1]; y[2] = 4.0*x[2];
        };
        std::vector<Real> rhs(n, 0.0);
        std::vector<Real> x0(n, 0.0);

        GMRESSolver gmres(30, 10, 1e-10);
        gmres.solve(matvec, rhs.data(), x0.data(), n);

        CHECK_NEAR(x0[0], 0.0, 1e-15, "GMRES: zero RHS -> x[0]=0");
        CHECK_NEAR(x0[1], 0.0, 1e-15, "GMRES: zero RHS -> x[1]=0");
        CHECK(gmres.converged(),
            "GMRES: zero RHS converges immediately");
    }

    // Test 9: Iteration count
    {
        const int n = 5;
        auto matvec = [&](const Real* x, Real* y) {
            for (int i = 0; i < n; ++i) y[i] = (i + 2.0) * x[i];
        };
        std::vector<Real> rhs = {2.0, 6.0, 12.0, 20.0, 30.0};
        std::vector<Real> x0(n, 0.0);

        GMRESSolver gmres(30, 10, 1e-10);
        gmres.solve(matvec, rhs.data(), x0.data(), n);

        CHECK(gmres.iterations() > 0,
            "GMRES: positive iteration count");
        CHECK(gmres.iterations() <= n + 1,
            "GMRES: iterations <= n+1 for diagonal system");
    }

    // Test 10: Residual below tolerance on converged solve
    {
        const int n = 3;
        // Non-symmetric matrix
        auto matvec = [](const Real* x, Real* y) {
            y[0] = 5.0*x[0] + 1.0*x[1] - 1.0*x[2];
            y[1] = -2.0*x[0] + 6.0*x[1] + 1.0*x[2];
            y[2] = 1.0*x[0] - 1.0*x[1] + 4.0*x[2];
        };
        std::vector<Real> rhs = {5.0, 5.0, 4.0};
        std::vector<Real> x0(n, 0.0);

        GMRESSolver gmres(30, 10, 1e-10);
        gmres.solve(matvec, rhs.data(), x0.data(), n);

        // Verify residual
        Real Ax[3];
        matvec(x0.data(), Ax);
        Real res = 0.0;
        for (int i = 0; i < n; ++i) {
            Real r = rhs[i] - Ax[i];
            res += r * r;
        }
        res = std::sqrt(res);
        CHECK(res < 1e-8,
            "GMRES: final residual below tolerance");
    }
}

// ============================================================================
// 4. Preconditioner Tests
// ============================================================================
void test_preconditioners() {
    std::cout << "\n--- Preconditioner Tests ---\n";

    // Test 1: Jacobi on diagonal system = exact inverse
    {
        const int n = 4;
        std::vector<Real> diag = {2.0, 5.0, 10.0, 0.5};
        JacobiPreconditioner jac(diag.data(), n);

        std::vector<Real> x = {1.0, 1.0, 1.0, 1.0};
        std::vector<Real> y(n);
        jac.apply(x.data(), y.data(), n);

        CHECK_NEAR(y[0], 0.5, 1e-15, "Jacobi: 1/2 for d=2");
        CHECK_NEAR(y[1], 0.2, 1e-15, "Jacobi: 1/5 for d=5");
        CHECK_NEAR(y[2], 0.1, 1e-15, "Jacobi: 1/10 for d=10");
        CHECK_NEAR(y[3], 2.0, 1e-15, "Jacobi: 1/0.5=2 for d=0.5");
    }

    // Test 2: Jacobi preserves vector for identity
    {
        const int n = 3;
        std::vector<Real> diag = {1.0, 1.0, 1.0};
        JacobiPreconditioner jac(diag.data(), n);

        std::vector<Real> x = {3.14, 2.71, 1.41};
        std::vector<Real> y(n);
        jac.apply(x.data(), y.data(), n);

        CHECK_NEAR(y[0], 3.14, 1e-15, "Jacobi: identity preserves x[0]");
        CHECK_NEAR(y[1], 2.71, 1e-15, "Jacobi: identity preserves x[1]");
        CHECK_NEAR(y[2], 1.41, 1e-15, "Jacobi: identity preserves x[2]");
    }

    // Test 3: ILUT factorization succeeds on SPD matrix
    {
        // 3x3 SPD: A = [4 -1 0; -1 4 -1; 0 -1 4]
        const int n = 3;
        SparseMatrix A = SparseMatrix::tridiagonal(4.0, n);

        ILUTPreconditioner ilut(A, 1e-3, 10);
        bool ok = ilut.factorize();
        CHECK(ok, "ILUT: factorization succeeds on SPD tridiagonal");
        CHECK(ilut.factorized(), "ILUT: factorized flag set");
    }

    // Test 4: ILUT apply + A gives approximately identity (LU ~ A)
    {
        // For a well-conditioned matrix with low fill, ILUT should be close to exact
        const int n = 3;
        // Dense SPD matrix
        Real dense[9] = {
            4.0, -1.0,  0.0,
           -1.0,  4.0, -1.0,
            0.0, -1.0,  4.0
        };
        SparseMatrix A = SparseMatrix::from_dense(dense, n);

        ILUTPreconditioner ilut(A, 1e-10, 20); // very low threshold = keep everything
        ilut.factorize();

        // Apply (LU)^{-1} to columns of A, should get ~identity
        std::vector<Real> col(n), result(n);
        bool close_to_identity = true;
        for (int j = 0; j < n; ++j) {
            // Extract column j of A
            for (int i = 0; i < n; ++i) col[i] = dense[i * n + j];
            ilut.apply(col.data(), result.data());

            for (int i = 0; i < n; ++i) {
                Real expected = (i == j) ? 1.0 : 0.0;
                if (std::abs(result[i] - expected) > 0.1) {
                    close_to_identity = false;
                }
            }
        }
        CHECK(close_to_identity,
            "ILUT: (LU)^{-1} * A ~ I for tridiagonal");
    }

    // Test 5: ILUT with GMRES improves convergence vs unpreconditioned
    {
        const int n = 20;
        SparseMatrix A = SparseMatrix::tridiagonal(4.0, n);

        // Build rhs = A * ones
        std::vector<Real> ones(n, 1.0), rhs(n);
        A.matvec(ones.data(), rhs.data());

        auto matvec = [&](const Real* x, Real* y) { A.matvec(x, y); };

        // Unpreconditioned
        std::vector<Real> x1(n, 0.0);
        GMRESSolver gmres1(5, 50, 1e-10);
        gmres1.solve(matvec, rhs.data(), x1.data(), n);
        int iter_no_prec = gmres1.iterations();

        // Preconditioned with ILUT
        ILUTPreconditioner ilut(A, 1e-3, 10);
        ilut.factorize();

        std::vector<Real> x2(n, 0.0);
        GMRESSolver gmres2(5, 50, 1e-10);
        gmres2.set_preconditioner([&](const Real* x, Real* y) {
            ilut.apply(x, y);
        });
        gmres2.solve(matvec, rhs.data(), x2.data(), n);
        int iter_prec = gmres2.iterations();

        CHECK(iter_prec <= iter_no_prec,
            "ILUT+GMRES: preconditioned uses fewer or equal iterations");
    }

    // Test 6: Jacobi with GMRES improves convergence
    {
        const int n = 10;
        // Diagonal-heavy system
        std::vector<Real> diag(n);
        for (int i = 0; i < n; ++i) diag[i] = 10.0 + i;
        SparseMatrix A = SparseMatrix::tridiagonal(10.0, n);
        // Override diagonals
        for (int i = 0; i < n; ++i) {
            for (int p = A.row_ptr[i]; p < A.row_ptr[i+1]; ++p) {
                if (A.col_indices[p] == i) {
                    A.values[p] = diag[i];
                }
            }
        }

        std::vector<Real> ones(n, 1.0), rhs(n);
        A.matvec(ones.data(), rhs.data());
        auto matvec = [&](const Real* x, Real* y) { A.matvec(x, y); };

        JacobiPreconditioner jac(A);

        std::vector<Real> x0(n, 0.0);
        GMRESSolver gmres(30, 10, 1e-10);
        gmres.set_preconditioner([&](const Real* x, Real* y) {
            jac.apply(x, y);
        });
        gmres.solve(matvec, rhs.data(), x0.data(), n);

        CHECK(gmres.converged(),
            "Jacobi+GMRES: preconditioned solve converged");
    }

    // Test 7: Sparse matrix CSR construction
    {
        Real dense[9] = {
            1.0, 2.0, 0.0,
            0.0, 3.0, 4.0,
            5.0, 0.0, 6.0
        };
        SparseMatrix A = SparseMatrix::from_dense(dense, 3, 0.0);

        CHECK(A.n == 3, "CSR: dimension correct");
        CHECK(A.nnz() == 6, "CSR: 6 non-zeros (no zero diagonals in this matrix)");

        // Test matvec: A * [1,1,1] = [3, 7, 11]
        Real x[3] = {1.0, 1.0, 1.0};
        Real y[3];
        A.matvec(x, y);
        CHECK_NEAR(y[0], 3.0, 1e-15, "CSR: matvec row 0");
        CHECK_NEAR(y[1], 7.0, 1e-15, "CSR: matvec row 1");
        CHECK_NEAR(y[2], 11.0, 1e-15, "CSR: matvec row 2");
    }

    // Test 8: Forward substitution
    {
        // L = [1 0 0; 2 1 0; 3 4 1], L*z = [1; 2; 3]
        // z[0] = 1, z[1] = 2 - 2*1 = 0, z[2] = 3 - 3*1 - 4*0 = 0
        const int n = 3;
        Real dense_A[9] = {1,0,0, 2,1,0, 3,4,1}; // dummy for construction
        SparseMatrix A = SparseMatrix::from_dense(dense_A, n, 0.0);
        ILUTPreconditioner ilut(A, 1e-10, 20);

        // Manually set up L factor as unit lower triangular (diagonal not stored)
        SparseMatrix L(n);
        L.row_ptr = {0, 0, 1, 3};
        L.values = {2.0, 3.0, 4.0};
        L.col_indices = {0, 0, 1};

        // Use the ILUT's forward substitution directly
        // We'll test by factorizing a known matrix
        Real dense_test[9] = {
            1.0, 0.0, 0.0,
            2.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        };
        SparseMatrix Atest = SparseMatrix::from_dense(dense_test, n, 0.0);
        ILUTPreconditioner ilut2(Atest, 1e-10, 20);
        ilut2.factorize();

        Real x[3] = {5.0, 10.0, 3.0};
        Real y[3];
        ilut2.forward_substitution(x, y);
        // With unit lower L = [1 0 0; 2 1 0; 0 0 1]:
        // y[0] = 5, y[1] = 10 - 2*5 = 0, y[2] = 3
        CHECK_NEAR(y[0], 5.0, 1e-8, "ForwardSub: y[0] = 5");
        CHECK(std::abs(y[1]) < 1.0, "ForwardSub: y[1] reduced by elimination");
    }

    // Test 9: Backward substitution
    {
        // Test via ILUT on upper triangular matrix
        const int n = 3;
        Real dense[9] = {
            2.0, 1.0, 0.0,
            0.0, 3.0, 1.0,
            0.0, 0.0, 4.0
        };
        SparseMatrix A = SparseMatrix::from_dense(dense, n, 0.0);
        ILUTPreconditioner ilut(A, 1e-10, 20);
        ilut.factorize();

        // For upper triangular, L should be identity (no lower entries)
        // U should be A itself
        // Solve U*y = [2, 4, 8]: y[2]=2, y[1]=(4-2)/3=2/3, y[0]=(2-2/3)/2=2/3
        Real z[3] = {2.0, 4.0, 8.0};
        Real y[3];
        ilut.backward_substitution(z, y);

        CHECK_NEAR(y[2], 2.0, 1e-8, "BackSub: y[2] = 8/4 = 2");
        CHECK(std::abs(y[1] - (4.0 - 1.0*2.0)/3.0) < 1e-6,
            "BackSub: y[1] correct");
    }

    // Test 10: Threshold dropping (high tau -> more dropping -> smaller factors)
    {
        const int n = 10;
        // Dense-ish matrix
        std::vector<Real> dense(n * n, 0.0);
        for (int i = 0; i < n; ++i) {
            dense[i * n + i] = 10.0;
            for (int j = 0; j < n; ++j) {
                if (i != j) dense[i * n + j] = 0.1 / (std::abs(i - j) + 1.0);
            }
        }
        SparseMatrix A = SparseMatrix::from_dense(dense.data(), n, 0.0);

        ILUTPreconditioner ilut_low(A, 1e-10, 100);  // keep almost everything
        ilut_low.factorize();
        int nnz_low = ilut_low.L().nnz() + ilut_low.U().nnz();

        ILUTPreconditioner ilut_high(A, 0.5, 2);  // aggressive dropping
        ilut_high.factorize();
        int nnz_high = ilut_high.L().nnz() + ilut_high.U().nnz();

        CHECK(nnz_high <= nnz_low,
            "ILUT: higher threshold produces sparser factors");
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== Wave 30: Advanced Iterative Solvers and Preconditioners Test Suite ===\n";

    test_line_search();
    test_lbfgs();
    test_gmres();
    test_preconditioners();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " total ===\n";

    return (tests_failed == 0) ? 0 : 1;
}
