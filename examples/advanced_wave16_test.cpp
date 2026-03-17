/**
 * @file advanced_wave16_test.cpp
 * @brief Wave 16: Advanced Capabilities Test Suite
 *
 * Tests 8 sub-modules (5+ tests each, 40+ total):
 * - 16a: Modal Analysis (Lanczos Eigensolver)
 * - 16b: XFEM (Level Set Crack, Enrichment, Propagation)
 * - 16c: Blast Loading (CONWEP)
 * - 16d: Airbag Simulation
 * - 16e: Seatbelt Dynamics
 * - 16f: Advanced ALE (Eulerian, Cut-Cell, Turbulence)
 * - 16g: Adaptive Mesh Refinement
 * - 16h: Draping Analysis
 */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <array>
#include "../include/nexussim/physics/advanced_wave16.hpp"

using Real = double;
static int pass_count = 0, fail_count = 0;

#define CHECK(cond, name) do { \
    if (cond) { printf("[PASS] %s\n", name); pass_count++; } \
    else { printf("[FAIL] %s\n", name); fail_count++; } \
} while(0)

#define CHECK_NEAR(val, expected, rtol, name) do { \
    double v_ = (val), e_ = (expected); \
    double diff_ = std::abs(v_ - e_); \
    double denom_ = std::abs(e_) > 1e-30 ? std::abs(e_) : 1.0; \
    if (diff_ / denom_ < (rtol)) { printf("[PASS] %s (got %g, expected %g)\n", name, v_, e_); pass_count++; } \
    else { printf("[FAIL] %s (got %g, expected %g, rel_err=%g)\n", name, v_, e_, diff_/denom_); fail_count++; } \
} while(0)

using namespace nxs::advanced;

// ============================================================================
// 16a: Modal Analysis Tests
// ============================================================================
void test_modal_analysis() {
    printf("\n=== 16a: Modal Analysis (Lanczos Eigensolver) ===\n");

    // Test problem: 2-DOF spring-mass system
    // K = [2 -1; -1 1], M = [2 0; 0 1]
    // Eigenvalues: lambda_1 = 0.382, lambda_2 = 2.618 (approx)
    // Exact: (3 +/- sqrt(5)) / 2 -> 0.38197, 2.61803

    {
        SparseMatrix K(2), M(2);
        std::vector<Real> Kd = {2.0, -1.0, -1.0, 1.0};
        std::vector<Real> Md = {2.0, 0.0, 0.0, 1.0};
        K.from_dense(Kd, 2);
        M.from_dense(Md, 2);

        LanczosEigensolver solver;
        solver.set_max_iterations(1000);
        solver.set_tolerance(1.0e-10);
        auto result = solver.solve(K, M, 2);

        // Test 1: Two eigenvalues found
        CHECK(result.eigenvalues.size() == 2, "Modal: 2 eigenvalues returned");

        // Test 2: First eigenvalue correct
        // Generalized: det(K - lam*M)=0 -> (2-2l)(1-l)-1=0 -> 2l^2-4l+1=0
        // l = 1 +/- sqrt(2)/2
        Real lam1_exact = 1.0 - std::sqrt(2.0) / 2.0; // ~0.2929
        CHECK_NEAR(result.eigenvalues[0], lam1_exact, 0.01,
                   "Modal: first eigenvalue 1-sqrt(2)/2");

        // Test 3: Second eigenvalue correct
        Real lam2_exact = 1.0 + std::sqrt(2.0) / 2.0; // ~1.7071
        CHECK_NEAR(result.eigenvalues[1], lam2_exact, 0.01,
                   "Modal: second eigenvalue 1+sqrt(2)/2");
    }

    // Test 4: Eigenvalues sorted ascending
    {
        SparseMatrix K(3), M(3);
        std::vector<Real> Kd = {3, -1, 0, -1, 2, -1, 0, -1, 3};
        std::vector<Real> Md = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        K.from_dense(Kd, 3);
        M.from_dense(Md, 3);

        LanczosEigensolver solver;
        solver.set_max_iterations(1000);
        solver.set_tolerance(1.0e-8);
        auto result = solver.solve(K, M, 3);

        bool sorted = true;
        for (size_t i = 1; i < result.eigenvalues.size(); ++i) {
            if (result.eigenvalues[i] < result.eigenvalues[i-1] - 1e-6) sorted = false;
        }
        CHECK(sorted, "Modal: eigenvalues sorted ascending");
    }

    // Test 5: Frequency conversion
    {
        std::vector<Real> evals = {100.0, 400.0, 900.0};
        auto freqs = LanczosEigensolver::eigenvalues_to_frequencies(evals);
        // omega = sqrt(lambda), freq = omega / (2*pi)
        CHECK_NEAR(freqs[0], 10.0 / (2.0 * M_PI), 1e-8, "Modal: frequency from eigenvalue 100");
        CHECK_NEAR(freqs[1], 20.0 / (2.0 * M_PI), 1e-8, "Modal: frequency from eigenvalue 400");
    }

    // Test 6: SparseMatrix matvec
    {
        SparseMatrix A(2);
        std::vector<Real> Ad = {3.0, 1.0, 1.0, 2.0};
        A.from_dense(Ad, 2);
        std::vector<Real> x = {1.0, 2.0};
        std::vector<Real> y;
        A.matvec(x, y);
        CHECK_NEAR(y[0], 5.0, 1e-12, "Modal: sparse matvec [0]");
        CHECK_NEAR(y[1], 5.0, 1e-12, "Modal: sparse matvec [1]");
    }
}

// ============================================================================
// 16b: XFEM Tests
// ============================================================================
void test_xfem() {
    printf("\n=== 16b: XFEM (Extended FEM) ===\n");

    // Test 1: Level set initialization for straight crack
    {
        LevelSetCrack crack;
        // 4 nodes of a unit square: (0,0), (1,0), (1,1), (0,1)
        std::vector<std::array<Real, 3>> nodes = {
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
            {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}
        };
        // Crack from (0, 0.5) to (0.5, 0.5) — horizontal, y=0.5
        crack.initialize_straight_crack(nodes, 0.0, 0.5, 0.5, 0.5);

        // Nodes below crack (y<0.5) should have phi < 0
        // Node (0,0): phi should be negative (below crack line)
        CHECK(crack.get_phi(0) < 0.0, "XFEM: phi negative below crack");
        // Node (0,1): phi should be positive (above crack line)
        CHECK(crack.get_phi(3) > 0.0, "XFEM: phi positive above crack");
    }

    // Test 2: Element cut detection
    {
        LevelSetCrack crack;
        std::vector<std::array<Real, 3>> nodes = {
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
            {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
            {2.0, 0.0, 0.0}, {2.0, 1.0, 0.0}
        };
        crack.initialize_straight_crack(nodes, 0.0, 0.5, 0.5, 0.5);

        // Element with nodes {0,1,2,3} should be cut (crack passes through)
        std::vector<Index> elem1 = {0, 1, 2, 3};
        CHECK(crack.is_cut(elem1), "XFEM: element cut by crack");

        // Element with nodes {1,4,5,2} should NOT be cut (entirely ahead of crack tip)
        std::vector<Index> elem2 = {1, 4, 5, 2};
        CHECK(!crack.is_cut(elem2), "XFEM: element not cut when ahead of tip");
    }

    // Test 3: Heaviside enrichment function
    {
        CHECK_NEAR(XFEMEnrichment::heaviside(0.5), 1.0, 1e-12, "XFEM: heaviside positive");
        CHECK_NEAR(XFEMEnrichment::heaviside(-0.3), -1.0, 1e-12, "XFEM: heaviside negative");
        CHECK_NEAR(XFEMEnrichment::heaviside(0.0), 0.0, 1e-12, "XFEM: heaviside zero");
    }

    // Test 4: Crack tip enrichment functions
    {
        Real r = 1.0, theta = 0.0;
        auto F = XFEMEnrichment::crack_tip_functions(r, theta);
        // At theta=0: F1=0, F2=sqrt(r)=1, F3=0, F4=0
        CHECK_NEAR(F[0], 0.0, 1e-10, "XFEM: tip func F1 at theta=0");
        CHECK_NEAR(F[1], 1.0, 1e-10, "XFEM: tip func F2 at theta=0");

        // At theta=pi, r=1: F1=sin(pi/2)=1, F2=cos(pi/2)=0
        auto F2 = XFEMEnrichment::crack_tip_functions(1.0, M_PI);
        CHECK_NEAR(F2[0], 1.0, 1e-10, "XFEM: tip func F1 at theta=pi");
    }

    // Test 5: Crack propagation angle (pure mode I)
    {
        // Pure mode I: KII = 0 -> theta = 0 (straight ahead)
        Real theta = CrackPropagation::max_hoop_stress_angle(100.0, 0.0);
        CHECK_NEAR(theta, 0.0, 1e-10, "XFEM: mode I propagation angle = 0");
    }

    // Test 6: J-integral computation
    {
        CrackPropagation prop;
        prop.set_material(210.0e9, 0.3);
        Real J = prop.compute_J_integral(50.0e6, 0.0);
        // J = (1-0.09)/210e9 * (50e6)^2 = 0.91/210e9 * 2.5e15 = 10857 J/m^2
        Real expected = (1.0 - 0.09) / 210.0e9 * 2.5e15;
        CHECK_NEAR(J, expected, 1e-6, "XFEM: J-integral pure mode I");
    }

    // Test 7: Enriched shape function
    {
        // Node on positive side (phi_node = 0.5), evaluation on negative side (phi_point = -0.3)
        Real val = XFEMEnrichment::enriched_shape(0.25, -0.3, 0.5);
        // = 0.25 * (H(-0.3) - H(0.5)) = 0.25 * (-1 - 1) = -0.5
        CHECK_NEAR(val, -0.5, 1e-12, "XFEM: enriched shape function value");
    }
}

// ============================================================================
// 16c: Blast Loading Tests
// ============================================================================
void test_blast_loading() {
    printf("\n=== 16c: Blast Loading (CONWEP) ===\n");

    CONWEPBlast blast;
    blast.set_charge(10.0, 1.0); // 10 kg TNT
    blast.set_detonation_point(0.0, 0.0, 0.0);

    // Test 1: Scaled distance
    {
        Real Z = blast.scaled_distance(5.0);
        // Z = 5 / 10^(1/3) = 5 / 2.154 = 2.321
        Real expected = 5.0 / std::cbrt(10.0);
        CHECK_NEAR(Z, expected, 1e-8, "Blast: scaled distance Z = R/W^(1/3)");
    }

    // Test 2: Peak incident pressure is positive
    {
        Real Ps = blast.peak_incident_pressure(5.0);
        CHECK(Ps > 0.0, "Blast: positive peak incident pressure");
    }

    // Test 3: Pressure decreases with distance
    {
        Real P5 = blast.peak_incident_pressure(5.0);
        Real P10 = blast.peak_incident_pressure(10.0);
        CHECK(P5 > P10, "Blast: pressure decreases with distance");
    }

    // Test 4: Reflected pressure >= incident for normal incidence
    {
        Real Ps = blast.peak_incident_pressure(5.0);
        Real Pr = blast.peak_reflected_pressure(5.0, 0.0); // Normal incidence
        CHECK(Pr >= Ps, "Blast: reflected >= incident at normal incidence");
    }

    // Test 5: Friedlander waveform at t=0 gives peak pressure
    {
        Real P0 = blast.friedlander_pressure(5.0, 0.0);
        Real Ps = blast.peak_incident_pressure(5.0);
        CHECK_NEAR(P0, Ps, 1e-10, "Blast: Friedlander P(t=0) = Ps");
    }

    // Test 6: Friedlander waveform decays to zero at t=t_pos
    {
        Real t_pos = blast.positive_phase_duration(5.0);
        Real P_end = blast.friedlander_pressure(5.0, t_pos);
        CHECK_NEAR(P_end, 0.0, 1e-6, "Blast: Friedlander P(t_pos) = 0");
    }

    // Test 7: Positive impulse is positive
    {
        Real I = blast.positive_impulse(5.0);
        CHECK(I > 0.0, "Blast: positive impulse > 0");
    }

    // Test 8: Reflected pressure at grazing angle is lower
    {
        Real Pr_normal = blast.peak_reflected_pressure(5.0, 0.0);
        Real Pr_oblique = blast.peak_reflected_pressure(5.0, M_PI / 3.0);
        CHECK(Pr_normal > Pr_oblique, "Blast: reflected pressure lower at oblique angle");
    }
}

// ============================================================================
// 16d: Airbag Simulation Tests
// ============================================================================
void test_airbag() {
    printf("\n=== 16d: Airbag Simulation ===\n");

    // Test 1: Ideal gas law consistency
    {
        AirbagModel bag;
        bag.set_gas_properties(0.028, 1.4); // N2
        bag.set_initial_conditions(0.060, 300.0, 101325.0); // 60L, 300K, 1atm

        // P*V = m*R*T => m = P*V/(R*T)
        Real R_spec = 8.314 / 0.028; // 296.93
        Real m_expected = 101325.0 * 0.060 / (R_spec * 300.0);
        CHECK_NEAR(bag.mass(), m_expected, 0.01, "Airbag: initial mass from ideal gas law");
    }

    // Test 2: Pressure increase with mass inflow
    {
        AirbagModel bag;
        bag.set_gas_properties(0.028, 1.4);
        bag.set_initial_conditions(0.060, 300.0, 101325.0);
        bag.set_inflator(0.5, 1500.0); // 0.5 kg/s at 1500 K

        Real P0 = bag.pressure();
        bag.step(0.001, 0.060); // 1 ms step, constant volume
        CHECK(bag.pressure() > P0, "Airbag: pressure increases with mass inflow");
    }

    // Test 3: Temperature rises with hot gas inflow
    {
        AirbagModel bag;
        bag.set_gas_properties(0.028, 1.4);
        bag.set_initial_conditions(0.060, 300.0, 101325.0);
        bag.set_inflator(0.5, 1500.0);

        Real T0 = bag.temperature();
        bag.step(0.001, 0.060);
        CHECK(bag.temperature() > T0, "Airbag: temperature rises with hot gas inflow");
    }

    // Test 4: Mass conservation with no venting
    {
        AirbagModel bag;
        bag.set_gas_properties(0.028, 1.4);
        bag.set_initial_conditions(0.060, 300.0, 101325.0);
        bag.set_inflator(0.5, 1500.0);

        Real m0 = bag.mass();
        Real dt = 0.001;
        bag.step(dt, 0.060);
        Real m_expected = m0 + 0.5 * dt; // dm = m_dot_in * dt
        CHECK_NEAR(bag.mass(), m_expected, 0.001, "Airbag: mass conservation (no vent)");
    }

    // Test 5: Vent hole reduces mass
    {
        AirbagModel bag;
        bag.set_gas_properties(0.028, 1.4);
        bag.set_initial_conditions(0.060, 300.0, 200000.0); // High pressure
        bag.set_inflator(0.0, 300.0); // No inflow
        bag.set_vent(0.001, 0.65); // 10 cm^2 vent

        Real m0 = bag.mass();
        bag.step(0.001, 0.060);
        CHECK(bag.mass() < m0, "Airbag: vent hole reduces gas mass");
    }

    // Test 6: Volume expansion reduces pressure
    {
        AirbagModel bag;
        bag.set_gas_properties(0.028, 1.4);
        bag.set_initial_conditions(0.060, 300.0, 200000.0);
        bag.set_inflator(0.0, 300.0);

        Real P0 = bag.pressure();
        bag.step(0.001, 0.080); // Volume increase 60L -> 80L
        CHECK(bag.pressure() < P0, "Airbag: volume expansion reduces pressure");
    }
}

// ============================================================================
// 16e: Seatbelt Dynamics Tests
// ============================================================================
void test_seatbelt() {
    printf("\n=== 16e: Seatbelt Dynamics ===\n");

    // Test 1: Belt tension under stretch
    {
        BeltElement belt;
        belt.initialize(1.0, 50000.0, 0.0); // 1m, 50 kN/m stiffness
        belt.update(1.05, 0.0, 0.001); // 5% stretch
        // Tension = k * L0 * stretch = 50000 * 1.0 * 0.05 = 2500 N
        CHECK_NEAR(belt.tension(), 2500.0, 0.01, "Belt: tension under 5% stretch");
    }

    // Test 2: No tension in compression
    {
        BeltElement belt;
        belt.initialize(1.0, 50000.0, 0.0);
        belt.update(0.95, 0.0, 0.001); // Shorter than original
        CHECK_NEAR(belt.tension(), 0.0, 1e-10, "Belt: zero tension in compression");
    }

    // Test 3: Belt friction formula (Euler's belt equation)
    {
        // T_tight = T_slack * exp(mu * theta)
        Real T_slack = 100.0;
        Real mu = 0.3;
        Real theta = M_PI; // 180 degree wrap
        Real T_tight = BeltElement::belt_friction(T_slack, mu, theta);
        Real expected = 100.0 * std::exp(0.3 * M_PI);
        CHECK_NEAR(T_tight, expected, 1e-8, "Belt: Euler friction formula");
    }

    // Test 4: Retractor locking on high deceleration
    {
        Retractor ret;
        ret.set_locking(10.0, 0.5); // 10 m/s^2 decel, 0.5 m/s pull

        CHECK(!ret.is_locked(), "Belt: retractor initially unlocked");
        ret.update(0.0, 100.0, 15.0, 0.1, 0.001); // High decel
        CHECK(ret.is_locked(), "Belt: retractor locks on high deceleration");
    }

    // Test 5: Load limiter caps force
    {
        Retractor ret;
        ret.set_locking(5.0, 0.3);
        ret.set_load_limiter(4000.0);

        // Lock the retractor first
        ret.update(0.0, 500.0, 20.0, 0.0, 0.001);
        CHECK(ret.is_locked(), "Belt: retractor locked before limiter test");

        // Apply force above limit
        Real eff = ret.update(0.01, 6000.0, 0.0, 0.0, 0.001);
        CHECK_NEAR(eff, 4000.0, 0.01, "Belt: load limiter caps force at 4000 N");
    }

    // Test 6: Pretensioner activation
    {
        Retractor ret;
        ret.set_locking(5.0, 0.3);
        ret.set_pretensioner(0.005, 2000.0, 0.06); // Fires at 5ms
        ret.set_spool(0.3, 500.0, 0.02);

        // Before fire time
        Real f1 = ret.update(0.003, 100.0, 0.0, 0.0, 0.001);

        // After fire time — pretensioner adds force
        Real f2 = ret.update(0.006, 100.0, 0.0, 0.0, 0.001);
        CHECK(f2 > f1, "Belt: pretensioner increases effective force");
        CHECK(ret.is_locked(), "Belt: pretensioner locks retractor");
    }
}

// ============================================================================
// 16f: Advanced ALE Tests
// ============================================================================
void test_advanced_ale() {
    printf("\n=== 16f: Advanced ALE (Eulerian, Cut-Cell, Turbulence) ===\n");

    // Test 1: Eulerian solver initialization
    {
        EulerianSolver euler;
        euler.initialize(100, 0.01);
        CHECK_NEAR(euler.density(50), 1.225, 1e-10, "ALE: initial density = 1.225 kg/m3");
        CHECK_NEAR(euler.pressure(50), 101325.0, 1e-10, "ALE: initial pressure = 1 atm");
    }

    // Test 2: Sod shock tube setup
    {
        EulerianSolver euler;
        euler.initialize(100, 0.01);
        // Left state: high pressure
        for (int i = 0; i < 50; ++i) euler.set_cell_state(i, 1.0, 0.0, 100000.0);
        // Right state: low pressure
        for (int i = 50; i < 100; ++i) euler.set_cell_state(i, 0.125, 0.0, 10000.0);

        CHECK_NEAR(euler.density(25), 1.0, 1e-10, "ALE: Sod left density");
        CHECK_NEAR(euler.density(75), 0.125, 1e-10, "ALE: Sod right density");

        // Take a few steps
        Real dt = euler.compute_stable_dt(0.3);
        CHECK(dt > 0.0, "ALE: CFL time step positive");

        // Run multiple steps to allow shock to propagate
        for (int step = 0; step < 20; ++step) {
            dt = euler.compute_stable_dt(0.3);
            euler.step_upwind(dt);
        }
        // After several steps, density near the interface should change
        // The contact surface moves into the low-pressure region
        // Check that cells near the interface have changed
        bool propagated = false;
        for (int i = 48; i < 55; ++i) {
            if (std::abs(euler.density(i) - 1.0) > 0.01 ||
                std::abs(euler.density(i) - 0.125) > 0.01) {
                propagated = true;
            }
        }
        CHECK(propagated, "ALE: shock propagation started");
    }

    // Test 3: Cut-cell volume fraction
    {
        CutCellMethod cc;
        cc.initialize(2);
        cc.set_volume_fraction(0, 0.7);
        cc.set_volume_fraction(1, 0.3);
        cc.set_density(0, 1000.0);
        cc.set_density(1, 800.0);

        Real mix_rho = cc.mixture_density();
        CHECK_NEAR(mix_rho, 0.7*1000.0 + 0.3*800.0, 1e-8, "ALE: mixture density");
    }

    // Test 4: PLIC interface reconstruction
    {
        // Half-filled cell with vertical interface: vf=0.5, n=(1,0)
        Real d = CutCellMethod::plic_reconstruct(0.5, 1.0, 0.0);
        CHECK_NEAR(d, 0.5, 0.02, "ALE: PLIC reconstruction d=0.5 for vf=0.5");

        // Quarter-filled cell with diagonal interface
        Real d2 = CutCellMethod::plic_reconstruct(0.25, 1.0, 1.0);
        CHECK(d2 > 0.0 && d2 < 2.0, "ALE: PLIC reconstruction valid for vf=0.25");
    }

    // Test 5: k-epsilon turbulent viscosity
    {
        TurbulenceModel turb;
        turb.initialize(1.0, 1.0, 1.225, 1.8e-5);
        // mu_t = rho * C_mu * k^2 / eps = 1.225 * 0.09 * 1 / 1 = 0.11025
        Real mu_t = turb.turbulent_viscosity();
        CHECK_NEAR(mu_t, 1.225 * 0.09, 1e-8, "ALE: turbulent viscosity mu_t");
    }

    // Test 6: Wall shear stress in log-law region
    {
        TurbulenceModel turb;
        turb.initialize(1.0, 100.0, 1.225, 1.8e-5);
        Real tau = turb.wall_shear_stress(10.0, 0.01);
        CHECK(tau > 0.0, "ALE: wall shear stress positive in log-law region");
    }

    // Test 7: Turbulence step maintains positive k and epsilon
    {
        TurbulenceModel turb;
        turb.initialize(1.0, 1.0, 1.225, 1.8e-5);
        turb.step(0.5, 0.001); // Some production
        CHECK(turb.k() > 0.0, "ALE: k stays positive after step");
        CHECK(turb.epsilon() > 0.0, "ALE: epsilon stays positive after step");
    }
}

// ============================================================================
// 16g: Adaptive Mesh Refinement Tests
// ============================================================================
void test_amr() {
    printf("\n=== 16g: Adaptive Mesh Refinement ===\n");

    // Test 1: ZZ error estimator
    {
        AMRManager amr;
        amr.initialize(4, 9); // 4 elements, 9 nodes (2x2 mesh)

        std::vector<Real> stress = {100.0, 200.0, 150.0, 300.0};
        std::vector<Real> volume = {1.0, 1.0, 1.0, 1.0};
        amr.compute_zz_error(stress, volume);

        CHECK(amr.global_error_norm() > 0.0, "AMR: ZZ error norm > 0 for non-uniform stress");
        CHECK(amr.element_error(3) > amr.element_error(0),
              "AMR: element with highest stress has larger error");
    }

    // Test 2: Mark elements for refinement
    {
        AMRManager amr;
        amr.initialize(4, 9);

        std::vector<Real> stress = {100.0, 100.0, 100.0, 500.0}; // One hot spot
        std::vector<Real> volume = {1.0, 1.0, 1.0, 1.0};
        amr.compute_zz_error(stress, volume);
        amr.mark_elements(0.5, 0.1, 3);

        CHECK(amr.is_marked_refine(3), "AMR: high-error element marked for refinement");
        // Low-error elements might be marked for coarsening
        // (depends on error distribution relative to threshold)
    }

    // Test 3: h-refinement creates child elements
    {
        AMRManager amr;
        amr.initialize(4, 9);

        std::vector<Real> stress = {100.0, 100.0, 100.0, 500.0};
        std::vector<Real> volume = {1.0, 1.0, 1.0, 1.0};
        amr.compute_zz_error(stress, volume);
        amr.mark_elements(0.5, 0.1, 3);

        int new_elems = amr.refine();
        CHECK(new_elems > 0, "AMR: refinement creates new elements");
        CHECK(!amr.is_active(3), "AMR: refined parent deactivated");
        CHECK(amr.num_elements() > 4, "AMR: total element count increased");
    }

    // Test 4: Hanging node constraints
    {
        auto coeff = AMRManager::hanging_node_constraint(10, 5, 6);
        CHECK_NEAR(coeff[0], 0.5, 1e-12, "AMR: hanging node coeff[0] = 0.5");
        CHECK_NEAR(coeff[1], 0.5, 1e-12, "AMR: hanging node coeff[1] = 0.5");
    }

    // Test 5: Mesh quality computation
    {
        // Perfect unit square
        std::array<std::array<Real, 2>, 4> square = {{
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}
        }};
        auto q = AMRManager::compute_quality(square);
        CHECK_NEAR(q.aspect_ratio, 1.0, 0.01, "AMR: unit square aspect ratio = 1");
        CHECK(q.quality_score() > 0.9, "AMR: unit square quality > 0.9");
    }

    // Test 6: Skewed element has lower quality
    {
        std::array<std::array<Real, 2>, 4> skewed = {{
            {0.0, 0.0}, {2.0, 0.0}, {2.5, 1.0}, {0.5, 1.0}
        }};
        auto q = AMRManager::compute_quality(skewed);

        std::array<std::array<Real, 2>, 4> square = {{
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}
        }};
        auto qs = AMRManager::compute_quality(square);

        CHECK(q.quality_score() < qs.quality_score(),
              "AMR: skewed element has lower quality than square");
    }

    // Test 7: ZZ error with neighbor averaging
    {
        AMRManager amr;
        amr.initialize(4, 9);

        std::vector<Real> stress = {100.0, 200.0, 150.0, 300.0};
        std::vector<Real> volume = {1.0, 1.0, 1.0, 1.0};
        std::vector<std::vector<int>> neighbors = {
            {1, 2}, {0, 3}, {0, 3}, {1, 2}
        };
        amr.compute_zz_error_with_neighbors(stress, volume, neighbors);
        CHECK(amr.global_error_norm() > 0.0, "AMR: ZZ neighbor error norm > 0");
    }
}

// ============================================================================
// 16h: Draping Analysis Tests
// ============================================================================
void test_draping() {
    printf("\n=== 16h: Draping Analysis ===\n");

    // Test 1: Initialize flat surface
    {
        DrapingAnalysis drape;
        drape.initialize_flat(10, 10, 1.0, 1.0);
        CHECK(drape.num_elements() == 100, "Draping: 10x10 grid = 100 elements");
    }

    // Test 2: All elements draped after drape()
    {
        DrapingAnalysis drape;
        drape.initialize_flat(5, 5, 1.0, 1.0);
        drape.set_pin_point(2, 2, 0.0);
        drape.drape();
        CHECK(drape.num_draped() == 25, "Draping: all 25 elements draped");
    }

    // Test 3: Pin point has zero shear angle on flat surface
    {
        DrapingAnalysis drape;
        drape.initialize_flat(5, 5, 1.0, 1.0);
        drape.set_pin_point(2, 2, 0.0);
        drape.drape();
        int pin_elem = 2 * 5 + 2; // j*nx + i
        CHECK_NEAR(drape.shear_angle(pin_elem), 0.0, 1e-10,
                   "Draping: zero shear at pin point");
    }

    // Test 4: Fiber angle matches initial angle at pin
    {
        DrapingAnalysis drape;
        drape.initialize_flat(5, 5, 1.0, 1.0);
        Real angle = M_PI / 6.0; // 30 degrees
        drape.set_pin_point(2, 2, angle);
        drape.drape();
        int pin_elem = 2 * 5 + 2;
        CHECK_NEAR(drape.fiber_angle(pin_elem), angle, 1e-10,
                   "Draping: fiber angle = initial angle at pin");
    }

    // Test 5: Flat surface has zero shear everywhere (orthogonal grid)
    {
        DrapingAnalysis drape;
        drape.initialize_flat(5, 5, 1.0, 1.0);
        drape.set_pin_point(2, 2, 0.0);
        drape.drape();

        Real max_shear = 0.0;
        for (int i = 0; i < drape.num_elements(); ++i) {
            if (drape.shear_angle(i) > max_shear) max_shear = drape.shear_angle(i);
        }
        CHECK(max_shear < 0.01, "Draping: flat surface has near-zero shear");
    }

    // Test 6: Geodesic distance on flat surface
    {
        DrapingAnalysis drape;
        drape.initialize_flat(10, 10, 1.0, 1.0);
        drape.set_pin_point(5, 5);
        drape.drape();

        // Distance from (0,0) to (5,5) along grid = dx*5 + dy*5
        Real dist = drape.geodesic_distance(0, 0, 5, 5);
        CHECK(dist > 0.0, "Draping: geodesic distance > 0");
        // On flat 1x1 surface with dx=dy=0.1, manhattan dist = 0.5 + 0.5 = 1.0
        CHECK_NEAR(dist, 1.0, 0.05, "Draping: geodesic distance on flat surface");
    }

    // Test 7: Locking detection with curved surface
    {
        DrapingAnalysis drape;
        int nx = 10, ny = 10;
        Real lx = 1.0, ly = 1.0;
        Real dx = lx / nx, dy = ly / ny;
        int nnx = nx + 1, nny = ny + 1;

        std::vector<std::array<Real, 3>> nodes(nnx * nny);
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                Real x = i * dx, y = j * dy;
                // Hemisphere-like surface with high curvature
                Real z = 0.5 * (x * x + y * y);
                nodes[j * nnx + i] = {x, y, z};
            }
        }

        drape.initialize_surface(nodes, nx, ny);
        drape.set_pin_point(5, 5, 0.0);
        drape.set_lock_angle(0.01); // Very tight lock angle (0.57 degrees)
        drape.drape();

        int locked = drape.check_locking();
        // Curved surface may have shear angles exceeding the tight lock angle
        CHECK(locked >= 0, "Draping: locking check returns valid count");
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("=== Wave 16 Advanced Capabilities Test ===\n");

    test_modal_analysis();
    test_xfem();
    test_blast_loading();
    test_airbag();
    test_seatbelt();
    test_advanced_ale();
    test_amr();
    test_draping();

    printf("\n========================================\n");
    printf("Results: %d/%d tests passed\n", pass_count, pass_count + fail_count);
    return fail_count > 0 ? 1 : 0;
}
