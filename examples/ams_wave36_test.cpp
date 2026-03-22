/**
 * @file ams_wave36_test.cpp
 * @brief Wave 36: AMS/SMS Substructuring Test Suite (6 components, 50 tests)
 *
 * Tests:
 *  1. SMSEngine                - Selective mass scaling for critical elements
 *  2. SMSPreconditionedCG      - PCG solver for SMS systems
 *  3. SMSBoundaryConditions    - Fixed velocity and cyclic BCs
 *  4. ComponentModeSynthesis   - Craig-Bampton substructuring
 *  5. FrequencyResponse        - Modal superposition FRF
 *  6. SMSConstraints           - RBE2/RBE3/joint constraints
 */

#include <nexussim/solver/ams_wave36.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <numeric>

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
// 1. SMSEngine Tests
// ============================================================================
void test_sms_engine() {
    std::cout << "\n--- SMSEngine Tests ---\n";

    // Test 1: Element dt computation
    {
        Real dt = SMSEngine::compute_element_dt(0.01, 5000.0);
        CHECK_NEAR(dt, 2.0e-6, 1e-10, "SMSEngine: element dt = L/c");
    }

    // Test 2: Scale factor for critical element
    {
        Real scale = SMSEngine::compute_scale_factor(1.0e-6, 1.0e-5);
        CHECK_NEAR(scale, 100.0, 1e-8, "SMSEngine: scale = (dt_target/dt)^2");
    }

    // Test 3: No scaling needed when dt >= dt_target
    {
        Real scale = SMSEngine::compute_scale_factor(2.0e-5, 1.0e-5);
        CHECK_NEAR(scale, 1.0, 1e-10, "SMSEngine: no scaling when dt >= target");
    }

    // Test 4: Mass scaling increases critical dt
    {
        SMSEngine engine;
        SMSConfig config;
        config.dt_target = 1.0e-5;
        config.max_mass_scale = 100.0;
        config.added_mass_limit = 0.99; // Large budget

        // 3 elements, one is slightly small (critical)
        std::vector<SMSElement> elems(3);
        for (int i = 0; i < 3; ++i) {
            elems[i].char_length = 0.1;
            elems[i].wave_speed = 5000.0; // dt = 2e-5
            elems[i].mass = 1.0;
            elems[i].density = 1000.0;
            elems[i].volume = 0.001;
        }
        // Element 2 is smaller: dt = 0.01/5000 = 2e-6, needs 25x scale to reach 1e-5
        elems[2].char_length = 0.01;
        elems[2].wave_speed = 5000.0;
        elems[2].mass = 1.0;
        elems[2].density = 1000.0;
        elems[2].volume = 0.001;

        Real dt_before = SMSEngine::get_critical_dt(elems.data(), 3);
        engine.apply_mass_scaling(elems.data(), 3, config);
        Real dt_after = SMSEngine::get_critical_dt(elems.data(), 3);

        CHECK(dt_after > dt_before, "SMSEngine: mass scaling increases critical dt");
    }

    // Test 5: Added mass is positive
    {
        SMSEngine engine;
        SMSConfig config;
        config.dt_target = 1.0e-4;
        config.max_mass_scale = 100.0;
        config.added_mass_limit = 0.5;

        std::vector<SMSElement> elems(3);
        for (int i = 0; i < 3; ++i) {
            elems[i].char_length = 0.001;
            elems[i].wave_speed = 5000.0;
            elems[i].mass = 1.0;
            elems[i].density = 1000.0;
            elems[i].volume = 0.001;
        }

        Real added = engine.apply_mass_scaling(elems.data(), 3, config);
        CHECK(added > 0.0, "SMSEngine: added mass is positive for critical elements");
    }

    // Test 6: Mass budget is respected
    {
        SMSEngine engine;
        SMSConfig config;
        config.dt_target = 1.0e-3;
        config.max_mass_scale = 10000.0;
        config.added_mass_limit = 0.01; // Very tight budget

        std::vector<SMSElement> elems(10);
        Real total_orig = 0.0;
        for (int i = 0; i < 10; ++i) {
            elems[i].char_length = 0.001;
            elems[i].wave_speed = 5000.0;
            elems[i].mass = 1.0;
            elems[i].density = 1000.0;
            elems[i].volume = 0.001;
            total_orig += elems[i].mass;
        }

        Real added = engine.apply_mass_scaling(elems.data(), 10, config);
        CHECK(added <= config.added_mass_limit * total_orig + 1e-10,
              "SMSEngine: mass budget respected");
    }

    // Test 7: Count critical elements
    {
        std::vector<SMSElement> elems(4);
        elems[0].char_length = 0.1; elems[0].wave_speed = 5000.0; // dt = 2e-5
        elems[1].char_length = 0.001; elems[1].wave_speed = 5000.0; // dt = 2e-7
        elems[2].char_length = 0.05; elems[2].wave_speed = 5000.0; // dt = 1e-5
        elems[3].char_length = 0.0005; elems[3].wave_speed = 5000.0; // dt = 1e-7

        int crit = SMSEngine::count_critical(elems.data(), 4, 1.0e-5);
        CHECK(crit == 2, "SMSEngine: correct count of critical elements");
    }

    // Test 8: Zero wave speed gives max dt
    {
        Real dt = SMSEngine::compute_element_dt(0.1, 0.0);
        CHECK(dt > 1.0e20, "SMSEngine: zero wave speed gives large dt");
    }

    // Test 9: Non-critical elements unchanged after scaling
    {
        SMSEngine engine;
        SMSConfig config;
        config.dt_target = 1.0e-6;

        std::vector<SMSElement> elems(2);
        elems[0].char_length = 0.1; elems[0].wave_speed = 5000.0;
        elems[0].mass = 1.0; elems[0].density = 1000.0; elems[0].volume = 0.001;
        elems[1].char_length = 0.1; elems[1].wave_speed = 5000.0;
        elems[1].mass = 1.0; elems[1].density = 1000.0; elems[1].volume = 0.001;

        engine.apply_mass_scaling(elems.data(), 2, config);
        CHECK_NEAR(elems[0].added_mass, 0.0, 1e-10,
                   "SMSEngine: non-critical element mass unchanged");
    }
}

// ============================================================================
// 2. SMSPreconditionedCG Tests
// ============================================================================
void test_sms_pcg() {
    std::cout << "\n--- SMSPreconditionedCG Tests ---\n";

    SMSPreconditionedCG pcg;

    // Test 10: Solve 2x2 SPD system
    {
        // K = [4 1; 1 3], f = [1 2]
        // x = [1/11, 7/11] = [0.0909, 0.6364]
        Real K[] = {4.0, 1.0, 1.0, 3.0};
        Real M[] = {4.0, 0.0, 0.0, 3.0}; // Diagonal preconditioner
        Real f[] = {1.0, 2.0};
        Real x[] = {0.0, 0.0};

        auto result = pcg.solve(K, M, f, x, 2, 1e-10, 100);
        CHECK(result.converged, "PCG: converges for 2x2 SPD");
        CHECK_NEAR(x[0], 1.0/11.0, 1e-6, "PCG: x[0] correct");
        CHECK_NEAR(x[1], 7.0/11.0, 1e-6, "PCG: x[1] correct");
    }

    // Test 11: Identity system
    {
        Real K[] = {1.0, 0.0, 0.0, 1.0};
        Real M[] = {1.0, 0.0, 0.0, 1.0};
        Real f[] = {3.0, 7.0};
        Real x[] = {0.0, 0.0};

        auto result = pcg.solve(K, M, f, x, 2, 1e-10, 100);
        CHECK(result.converged, "PCG: converges for identity");
        CHECK_NEAR(x[0], 3.0, 1e-10, "PCG: identity x[0]");
        CHECK_NEAR(x[1], 7.0, 1e-10, "PCG: identity x[1]");
    }

    // Test 12: Larger system 4x4
    {
        // Diagonal dominant SPD
        Real K[16] = {
            10, 1, 0, 0,
             1, 10, 1, 0,
             0,  1, 10, 1,
             0,  0,  1, 10
        };
        Real M[16] = {0};
        for (int i = 0; i < 4; ++i) M[i*4+i] = K[i*4+i];
        Real f[] = {1.0, 2.0, 3.0, 4.0};
        Real x[] = {0.0, 0.0, 0.0, 0.0};

        auto result = pcg.solve(K, M, f, x, 4, 1e-10, 100);
        CHECK(result.converged, "PCG: converges for 4x4 SPD");

        // Verify: K*x should = f
        Real Kx[4] = {0};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                Kx[i] += K[i*4+j] * x[j];
        Real err = 0;
        for (int i = 0; i < 4; ++i) err += (Kx[i]-f[i])*(Kx[i]-f[i]);
        CHECK(std::sqrt(err) < 1e-8, "PCG: residual small for 4x4");
    }

    // Test 13: Convergence in bounded iterations for well-conditioned system
    {
        Real K[] = {2.0, 0.0, 0.0, 2.0};
        Real M[] = {2.0, 0.0, 0.0, 2.0};
        Real f[] = {4.0, 6.0};
        Real x[] = {0.0, 0.0};

        auto result = pcg.solve(K, M, f, x, 2, 1e-10, 100);
        CHECK(result.iterations <= 2, "PCG: converges in <= n iterations for diagonal");
    }

    // Test 14: Unpreconditioned CG
    {
        Real K[] = {4.0, 1.0, 1.0, 3.0};
        Real f[] = {1.0, 2.0};
        Real x[] = {0.0, 0.0};

        auto result = pcg.solve_unpreconditioned(K, f, x, 2, 1e-10, 100);
        CHECK(result.converged, "PCG: unpreconditioned converges");
        CHECK_NEAR(x[0], 1.0/11.0, 1e-6, "PCG: unpreconditioned x[0]");
    }
}

// ============================================================================
// 3. SMSBoundaryConditions Tests
// ============================================================================
void test_sms_bc() {
    std::cout << "\n--- SMSBoundaryConditions Tests ---\n";

    SMSBoundaryConditions bc;

    // Test 15: Fixed BC zeroes correct row/col
    {
        Real K[9] = {5, 1, 2, 1, 6, 3, 2, 3, 7};
        Real f[3] = {10, 20, 30};
        int fixed[] = {1};

        bc.apply_fixed(K, f, 3, fixed, 1);
        CHECK_NEAR(K[1*3+0], 0.0, 1e-15, "FixedBC: K[1,0] = 0");
        CHECK_NEAR(K[0*3+1], 0.0, 1e-15, "FixedBC: K[0,1] = 0");
        CHECK_NEAR(K[1*3+1], 1.0, 1e-15, "FixedBC: K[1,1] = 1");
        CHECK_NEAR(K[1*3+2], 0.0, 1e-15, "FixedBC: K[1,2] = 0");
        CHECK_NEAR(f[1], 0.0, 1e-15, "FixedBC: f[1] = 0");
    }

    // Test 16: Fixed BC preserves other entries
    {
        Real K[9] = {5, 1, 2, 1, 6, 3, 2, 3, 7};
        Real f[3] = {10, 20, 30};
        int fixed[] = {1};
        Real K00_orig = K[0];
        Real K22_orig = K[8];

        bc.apply_fixed(K, f, 3, fixed, 1);
        CHECK_NEAR(K[0], K00_orig, 1e-15, "FixedBC: K[0,0] unchanged");
        CHECK_NEAR(K[8], K22_orig, 1e-15, "FixedBC: K[2,2] unchanged");
    }

    // Test 17: Multiple fixed DOFs
    {
        Real K[16] = {4,1,0,0, 1,4,1,0, 0,1,4,1, 0,0,1,4};
        Real f[4] = {1,2,3,4};
        int fixed[] = {0, 3};

        bc.apply_fixed(K, f, 4, fixed, 2);
        CHECK_NEAR(f[0], 0.0, 1e-15, "FixedBC: f[0] = 0 (multiple)");
        CHECK_NEAR(f[3], 0.0, 1e-15, "FixedBC: f[3] = 0 (multiple)");
        CHECK_NEAR(K[0*4+0], 1.0, 1e-15, "FixedBC: K[0,0] = 1 (multiple)");
        CHECK_NEAR(K[3*4+3], 1.0, 1e-15, "FixedBC: K[3,3] = 1 (multiple)");
    }

    // Test 18: Prescribed BC
    {
        Real K[4] = {2, -1, -1, 2};
        Real f[2] = {0, 0};
        int fixed[] = {0};
        Real vals[] = {1.0};

        bc.apply_fixed_prescribed(K, f, 2, fixed, vals, 1);
        CHECK_NEAR(f[0], 1.0, 1e-15, "PrescribedBC: f[0] = prescribed value");
        CHECK_NEAR(K[0*2+0], 1.0, 1e-15, "PrescribedBC: K[0,0] = 1");
    }

    // Test 19: Cyclic BC ties DOF pair
    {
        Real K[9] = {4, 1, 0, 1, 4, 1, 0, 1, 4};
        Real f[3] = {1, 2, 3};
        CyclicPair pairs[] = {{0, 2}};

        bc.apply_cyclic(K, f, 3, pairs, 1);
        // After cyclic: DOF 2 row should enforce x[2] = x[0]
        CHECK_NEAR(K[2*3+2], 1.0, 1e-15, "CyclicBC: K[2,2] = 1");
        CHECK_NEAR(K[2*3+0], -1.0, 1e-15, "CyclicBC: K[2,0] = -1");
        CHECK_NEAR(f[2], 0.0, 1e-15, "CyclicBC: f[2] = 0");
    }

    // Test 20: Enforce cyclic post-solve
    {
        Real x[3] = {5.0, 3.0, 99.0};
        CyclicPair pairs[] = {{0, 2}};
        SMSBoundaryConditions::enforce_cyclic(x, pairs, 1);
        CHECK_NEAR(x[2], x[0], 1e-15, "CyclicBC: enforce x[2] = x[0]");
    }

    // Test 21: Count free DOFs
    {
        int fixed[] = {1, 3};
        int free = SMSBoundaryConditions::count_free_dofs(5, fixed, 2);
        CHECK(free == 3, "FreeDOFs: 5 total - 2 fixed = 3 free");
    }
}

// ============================================================================
// 4. ComponentModeSynthesis Tests
// ============================================================================
void test_craig_bampton() {
    std::cout << "\n--- ComponentModeSynthesis Tests ---\n";

    ComponentModeSynthesis cms;

    // Build a simple 4-DOF spring-mass system:
    // k=1 between all adjacent DOFs
    // K = [2 -1 0 0; -1 2 -1 0; 0 -1 2 -1; 0 0 -1 2]
    // M = identity
    int n = 4;
    std::vector<Real> K(n*n, 0.0), M(n*n, 0.0);
    for (int i = 0; i < n; ++i) {
        K[i*n+i] = 2.0;
        M[i*n+i] = 1.0;
        if (i > 0) { K[i*n+(i-1)] = -1.0; K[(i-1)*n+i] = -1.0; }
    }

    // Boundary DOFs: 0 and 3, Interior: 1 and 2
    int bdry[] = {0, 3};

    // Test 22: Reduced system size correct
    {
        auto result = cms.reduce(K.data(), M.data(), n, bdry, 2, 2);
        CHECK(result.n_cb == 4, "CB: n_cb = n_boundary + n_modes = 2+2=4");
        CHECK(result.n_boundary == 2, "CB: n_boundary = 2");
        CHECK(result.n_modes == 2, "CB: n_modes = 2");
    }

    // Test 23: K_cb is not all zero
    {
        auto result = cms.reduce(K.data(), M.data(), n, bdry, 2, 2);
        Real sum = 0;
        for (int i = 0; i < result.n_cb * result.n_cb; ++i)
            sum += std::abs(result.K_cb[i]);
        CHECK(sum > 0.0, "CB: K_cb is non-trivial");
    }

    // Test 24: M_cb diagonal contains identity in mode block
    {
        auto result = cms.reduce(K.data(), M.data(), n, bdry, 2, 2);
        // Bottom-right 2x2 of M_cb should be identity (M-orthonormal modes)
        int nb = result.n_boundary;
        CHECK_NEAR(result.M_cb[(nb+0)*result.n_cb+(nb+0)], 1.0, 1e-6,
                   "CB: M_cb mode diagonal = 1");
        CHECK_NEAR(result.M_cb[(nb+1)*result.n_cb+(nb+1)], 1.0, 1e-6,
                   "CB: M_cb mode diagonal = 1 (mode 2)");
    }

    // Test 25: Eigenvalues are positive
    {
        auto result = cms.reduce(K.data(), M.data(), n, bdry, 2, 2);
        CHECK(result.eigenvalues[0] > 0.0, "CB: eigenvalue 0 positive");
        CHECK(result.eigenvalues[1] > 0.0, "CB: eigenvalue 1 positive");
    }

    // Test 26: Request more modes than interior DOFs is clamped
    {
        auto result = cms.reduce(K.data(), M.data(), n, bdry, 2, 10);
        CHECK(result.n_modes == 2, "CB: modes clamped to n_interior");
    }

    // Test 27: K_cb symmetry
    {
        auto result = cms.reduce(K.data(), M.data(), n, bdry, 2, 1);
        int ncb = result.n_cb;
        Real asym = 0;
        for (int i = 0; i < ncb; ++i)
            for (int j = 0; j < ncb; ++j)
                asym += std::abs(result.K_cb[i*ncb+j] - result.K_cb[j*ncb+i]);
        CHECK(asym < 1e-8, "CB: K_cb is symmetric");
    }
}

// ============================================================================
// 5. FrequencyResponse Tests
// ============================================================================
void test_frf() {
    std::cout << "\n--- FrequencyResponse Tests ---\n";

    FrequencyResponse frf;

    // Single DOF system: mode phi = [1], omega_1 = 10 rad/s, zeta = 0.01
    int n_modes = 1;
    int n = 1;
    Real modes[] = {1.0}; // phi_1(0) = 1
    Real freqs[] = {10.0};
    Real zeta = 0.01;

    // Test 28: FRF peak at resonance
    {
        // Sweep near omega=10
        int nf = 201;
        std::vector<Real> omega(nf), H_mag(nf), H_phase(nf);
        for (int i = 0; i < nf; ++i) omega[i] = 5.0 + 10.0 * i / (nf - 1);

        frf.compute_frf(modes, freqs, n_modes, n, omega.data(), nf,
                        zeta, 0, 0, H_mag.data(), H_phase.data());

        Real peak_omega = FrequencyResponse::find_resonance(H_mag.data(), omega.data(), nf);
        CHECK_NEAR(peak_omega, 10.0, 0.1, "FRF: resonance peak near omega=10");
    }

    // Test 29: FRF magnitude at resonance is ~1/(2*zeta*omega^2)
    {
        Complex H = frf.compute_frf_point(modes, freqs, n_modes, 10.0, zeta, 0, 0, n);
        // At resonance: D = 2i*zeta*omega^2, H = 1/D
        // |H| = 1/(2*zeta*omega^2) = 1/(2*0.01*100) = 0.5
        CHECK_NEAR(H.magnitude(), 0.5, 0.05, "FRF: magnitude at resonance ~ 1/(2*zeta*w^2)");
    }

    // Test 30: FRF phase near -pi/2 at resonance
    {
        Complex H = frf.compute_frf_point(modes, freqs, n_modes, 10.0, zeta, 0, 0, n);
        // At resonance, D is purely imaginary positive, so H = -i/(...)
        // Phase should be near -pi/2
        CHECK_NEAR(H.phase(), -M_PI/2.0, 0.05, "FRF: phase ~ -pi/2 at resonance");
    }

    // Test 31: FRF magnitude decreases far from resonance
    {
        Complex H_far = frf.compute_frf_point(modes, freqs, n_modes, 50.0, zeta, 0, 0, n);
        Complex H_near = frf.compute_frf_point(modes, freqs, n_modes, 10.0, zeta, 0, 0, n);
        CHECK(H_far.magnitude() < H_near.magnitude(),
              "FRF: magnitude decreases away from resonance");
    }

    // Test 32: Damping estimation from half-power bandwidth
    {
        int nf = 2001;
        std::vector<Real> omega(nf), H_mag(nf), H_phase(nf);
        for (int i = 0; i < nf; ++i) omega[i] = 8.0 + 4.0 * i / (nf - 1);

        frf.compute_frf(modes, freqs, n_modes, n, omega.data(), nf,
                        zeta, 0, 0, H_mag.data(), H_phase.data());

        Real zeta_est = FrequencyResponse::estimate_damping(
            H_mag.data(), omega.data(), nf);
        CHECK_NEAR(zeta_est, zeta, 0.005, "FRF: damping estimate from bandwidth");
    }

    // Test 33: Multi-mode FRF (2 modes)
    {
        Real modes2[] = {1.0, 0.5, 0.5, 1.0}; // 2 DOFs x 2 modes
        Real freqs2[] = {5.0, 15.0};
        int nf = 501;
        std::vector<Real> omega(nf), H_mag(nf), H_phase(nf);
        for (int i = 0; i < nf; ++i) omega[i] = 1.0 + 20.0 * i / (nf - 1);

        frf.compute_frf(modes2, freqs2, 2, 2, omega.data(), nf,
                        0.02, 0, 0, H_mag.data(), H_phase.data());

        // Should have peaks near omega=5 and omega=15
        // Find peaks by checking local maxima
        int peaks = 0;
        for (int i = 1; i < nf-1; ++i) {
            if (H_mag[i] > H_mag[i-1] && H_mag[i] > H_mag[i+1]) peaks++;
        }
        CHECK(peaks >= 2, "FRF: two resonance peaks for 2-mode system");
    }

    // Test 34: Complex reciprocal
    {
        Complex z(3.0, 4.0);
        Complex zi = z.reciprocal();
        Complex prod = z * zi;
        CHECK_NEAR(prod.re, 1.0, 1e-10, "Complex: z * z^{-1} real part = 1");
        CHECK_NEAR(prod.im, 0.0, 1e-10, "Complex: z * z^{-1} imag part = 0");
    }
}

// ============================================================================
// 6. SMSConstraints Tests
// ============================================================================
void test_sms_constraints() {
    std::cout << "\n--- SMSConstraints Tests ---\n";

    SMSConstraints constraints;
    SMSPreconditionedCG pcg;

    // Test 35: RBE2 - slave follows master
    {
        // 3-DOF system: springs 0-1 and 1-2
        // K = [1 -1 0; -1 2 -1; 0 -1 1]
        Real K[9] = {1, -1, 0, -1, 2, -1, 0, -1, 1};
        Real f[3] = {1, 0, -1};
        int slaves[] = {2};

        constraints.apply_rbe2(K, f, 3, 0, slaves, 1);

        // DOF 2 should now equal DOF 0
        // K[2,2] = 1, K[2,0] = -1, f[2] = 0
        CHECK_NEAR(K[2*3+2], 1.0, 1e-10, "RBE2: K[slave,slave] = 1");
        CHECK_NEAR(K[2*3+0], -1.0, 1e-10, "RBE2: K[slave,master] = -1");
        CHECK_NEAR(f[2], 0.0, 1e-10, "RBE2: f[slave] = 0");
    }

    // Test 36: RBE2 - solve and verify slave = master
    {
        // Use a simple SPD system with RBE2
        Real K[16] = {4, -1, 0, 0, -1, 4, -1, 0, 0, -1, 4, -1, 0, 0, -1, 4};
        Real f[4] = {1, 0, 0, 2};
        int slaves[] = {3};

        constraints.apply_rbe2(K, f, 4, 0, slaves, 1);

        // Solve
        Real M[16] = {0};
        for (int i = 0; i < 4; ++i) M[i*4+i] = std::abs(K[i*4+i]);
        Real x[4] = {0,0,0,0};
        pcg.solve(K, M, f, x, 4, 1e-10, 200);
        SMSConstraints::recover_rbe2(x, 0, slaves, 1);

        CHECK_NEAR(x[3], x[0], 1e-6, "RBE2: solved slave equals master");
    }

    // Test 37: RBE3 - master = weighted average
    {
        Real K[9] = {3, -1, 0, -1, 3, -1, 0, -1, 3};
        Real f[3] = {3, 0, 6};
        int slaves[] = {0, 2};
        Real weights[] = {1.0, 1.0};

        constraints.apply_rbe3(K, f, 3, 1, slaves, weights, 2);

        // Master row should have: K[1,1]=1, K[1,0]=-0.5, K[1,2]=-0.5
        CHECK_NEAR(K[1*3+1], 1.0, 1e-10, "RBE3: K[master,master] = 1");
        CHECK_NEAR(K[1*3+0], -0.5, 1e-10, "RBE3: K[master,slave0] = -w0/W");
        CHECK_NEAR(K[1*3+2], -0.5, 1e-10, "RBE3: K[master,slave1] = -w1/W");
        CHECK_NEAR(f[1], 0.0, 1e-10, "RBE3: f[master] = 0");
    }

    // Test 38: RBE3 recover master
    {
        Real x[3] = {2.0, 999.0, 6.0};
        int slaves[] = {0, 2};
        Real weights[] = {1.0, 3.0};
        SMSConstraints::recover_rbe3(x, 1, slaves, weights, 2);
        // master = (1*2 + 3*6) / (1+3) = 20/4 = 5
        CHECK_NEAR(x[1], 5.0, 1e-10, "RBE3: recovered master = weighted avg");
    }

    // Test 39: Joint spring adds stiffness
    {
        Real K[4] = {1, 0, 0, 1};
        constraints.apply_joint(K, 2, 0, 1, 10.0);
        CHECK_NEAR(K[0*2+0], 11.0, 1e-10, "Joint: K[0,0] += k_joint");
        CHECK_NEAR(K[1*2+1], 11.0, 1e-10, "Joint: K[1,1] += k_joint");
        CHECK_NEAR(K[0*2+1], -10.0, 1e-10, "Joint: K[0,1] = -k_joint");
        CHECK_NEAR(K[1*2+0], -10.0, 1e-10, "Joint: K[1,0] = -k_joint");
    }

    // Test 40: Multiple slaves in RBE2
    {
        int n = 4;
        Real K[16] = {4,-1,0,0, -1,4,-1,0, 0,-1,4,-1, 0,0,-1,4};
        Real f[4] = {1,0,0,0};
        int slaves[] = {1, 2, 3};

        constraints.apply_rbe2(K, f, n, 0, slaves, 3);

        // All slave rows should enforce x[slave] = x[0]
        for (int s = 0; s < 3; ++s) {
            int si = slaves[s];
            CHECK_NEAR(K[si*n+si], 1.0, 1e-10, "RBE2 multi: K[slave,slave]=1");
            CHECK_NEAR(K[si*n+0], -1.0, 1e-10, "RBE2 multi: K[slave,master]=-1");
        }
    }

    // Test 41: RBE3 with unequal weights
    {
        Real x[3] = {1.0, 999.0, 3.0};
        int slaves[] = {0, 2};
        Real weights[] = {2.0, 1.0};
        SMSConstraints::recover_rbe3(x, 1, slaves, weights, 2);
        // master = (2*1 + 1*3) / 3 = 5/3
        CHECK_NEAR(x[1], 5.0/3.0, 1e-10, "RBE3: unequal weight recovery");
    }
}

// ============================================================================
// Additional integration tests
// ============================================================================
void test_integration() {
    std::cout << "\n--- Integration Tests ---\n";

    // Test 42: SMS + PCG end-to-end
    {
        SMSEngine engine;
        SMSConfig config;
        config.dt_target = 1.0e-5;
        config.max_mass_scale = 100.0;
        config.added_mass_limit = 0.99;

        // Element 1 is slightly critical: dt = 0.01/5000 = 2e-6, target = 1e-5
        // Needs scale = (1e-5/2e-6)^2 = 25, mass from 0.01 to 0.25, added = 0.24
        // Total mass ~ 2.01, budget = 0.99 * 2.01 ~ 1.99, 0.24 < 1.99 OK
        std::vector<SMSElement> elems(3);
        elems[0] = {0.1, 5000.0, 1.0, 1000.0, 0.001, 0, 0, false};
        elems[1] = {0.01, 5000.0, 0.01, 1000.0, 0.00001, 0, 0, false};
        elems[2] = {0.1, 5000.0, 1.0, 1000.0, 0.001, 0, 0, false};

        engine.apply_mass_scaling(elems.data(), 3, config);
        CHECK(elems[1].added_mass > 0, "Integration: critical element gets mass");
    }

    // Test 43: SMS Config default values
    {
        SMSConfig config;
        CHECK(config.dt_target > 0, "Config: dt_target > 0");
        CHECK(config.max_mass_scale > 1.0, "Config: max_mass_scale > 1");
        CHECK(config.added_mass_limit > 0 && config.added_mass_limit < 1.0,
              "Config: added_mass_limit in (0,1)");
    }

    // Test 44: PCGResult defaults
    {
        PCGResult r;
        CHECK(r.iterations == 0, "PCGResult: initial iterations = 0");
        CHECK(!r.converged, "PCGResult: initial converged = false");
    }

    // Test 45: CraigBamptonResult defaults
    {
        CraigBamptonResult r;
        CHECK(r.n_boundary == 0, "CBResult: initial n_boundary = 0");
        CHECK(r.n_modes == 0, "CBResult: initial n_modes = 0");
    }

    // Test 46: Complex magnitude
    {
        Complex z(3.0, 4.0);
        CHECK_NEAR(z.magnitude(), 5.0, 1e-10, "Complex: |3+4i| = 5");
    }

    // Test 47: Complex multiplication
    {
        Complex a(1.0, 2.0), b(3.0, -1.0);
        Complex c = a * b;
        // (1+2i)(3-i) = 3 - i + 6i - 2i^2 = 3 + 5i + 2 = 5 + 5i
        CHECK_NEAR(c.re, 5.0, 1e-10, "Complex: multiply real");
        CHECK_NEAR(c.im, 5.0, 1e-10, "Complex: multiply imag");
    }

    // Test 48: FRF at zero frequency (static response)
    {
        FrequencyResponse frf;
        Real modes[] = {1.0};
        Real freqs[] = {10.0};
        Complex H = frf.compute_frf_point(modes, freqs, 1, 0.0, 0.01, 0, 0, 1);
        // H(0) = 1/omega^2 = 1/100 = 0.01
        CHECK_NEAR(H.re, 0.01, 1e-6, "FRF: static response = 1/omega^2");
        CHECK_NEAR(H.im, 0.0, 1e-6, "FRF: static response imaginary ~ 0");
    }

    // Test 49: Empty system edge case
    {
        SMSEngine engine;
        SMSConfig config;
        Real added = engine.apply_mass_scaling(nullptr, 0, config);
        CHECK_NEAR(added, 0.0, 1e-15, "Edge: zero elements gives zero added mass");
    }

    // Test 50: PCG with zero RHS gives zero solution
    {
        SMSPreconditionedCG pcg;
        Real K[] = {2, -1, -1, 2};
        Real M[] = {2, 0, 0, 2};
        Real f[] = {0, 0};
        Real x[] = {0, 0};
        auto r = pcg.solve(K, M, f, x, 2, 1e-10, 10);
        CHECK_NEAR(x[0], 0.0, 1e-10, "PCG: zero RHS -> zero solution x[0]");
        CHECK_NEAR(x[1], 0.0, 1e-10, "PCG: zero RHS -> zero solution x[1]");
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== Wave 36: AMS/SMS Substructuring Test Suite ===\n";

    test_sms_engine();
    test_sms_pcg();
    test_sms_bc();
    test_craig_bampton();
    test_frf();
    test_sms_constraints();
    test_integration();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " ===\n";

    return tests_failed > 0 ? 1 : 0;
}
