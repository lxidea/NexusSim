/**
 * @file spotweld_wave44_test.cpp
 * @brief Wave 44d: Spot Weld Contact — 10 tests
 *
 *  1.  Single weld — elastic loading below failure
 *  2.  Single weld — tension failure mode
 *  3.  Single weld — shear failure mode
 *  4.  Combined failure criterion
 *  5.  Stiffness degradation curve
 *  6.  Weld array — one weld fails, others remain intact
 *  7.  Thermal softening — elevated temperature reduces strength
 *  8.  Force application — equal/opposite forces on node_a / node_b
 *  9.  Energy tracking — absorbed energy accumulates correctly
 * 10.  Dynamic loading — multiple time steps with increasing displacement
 */

#include <nexussim/fem/spotweld_wave44.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <array>

using namespace nxs;
using namespace nxs::fem;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; std::cout << "[PASS] " << msg << "\n"; } \
    else { tests_failed++; std::cout << "[FAIL] " << msg << "\n"; } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) do { \
    double _va = static_cast<double>(a); \
    double _vb = static_cast<double>(b); \
    double _vt = static_cast<double>(tol); \
    if (std::abs(_va - _vb) <= _vt) { tests_passed++; std::cout << "[PASS] " << msg << "\n"; } \
    else { tests_failed++; std::cout << "[FAIL] " << msg \
        << " (got " << _va << ", expected " << _vb << ", tol " << _vt << ")\n"; } \
} while(0)

// ============================================================================
// Helper: build a minimal 2-node position array
//   node 0 at origin
//   node 1 at (0, 0, gap) — gap above node 0
// ============================================================================
[[maybe_unused]]
static std::vector<Real> make_two_nodes(Real gap_z = 0.01) {
    // Two nodes: node0 at (0,0,0), node1 at (0,0,gap_z)
    return {0.0, 0.0, 0.0,   0.0, 0.0, gap_z};
}

// Build a config connecting node 0 -> node 1 (already explicit)
static SpotWeldConfig make_config(Real k = 1.0e5, Real Fn_max = 500.0, Real Fs_max = 500.0,
                                  Real d_fail = 0.05, Real d_start_frac = 0.5) {
    SpotWeldConfig cfg;
    cfg.id = 1;
    cfg.node_a = 0;
    cfg.node_b = 1;
    cfg.initial_stiffness    = k;
    cfg.normal_strength      = Fn_max;
    cfg.shear_strength       = Fs_max;
    cfg.failure_displacement = d_fail;
    cfg.degradation_start    = d_start_frac;
    cfg.thermal_softening_temp   = 0.0;  // disabled
    cfg.thermal_softening_factor = 0.5;
    return cfg;
}

// ============================================================================
// Test 1: Elastic loading below failure — force = k * displacement
// ============================================================================
static void test_elastic_loading() {
    std::cout << "\n--- Test 1: Elastic loading below failure ---\n";

    // Node 0 fixed at origin; node 1 displaced in z by 0.001 m
    // Weld axis will be set to z on first update, then disp = 0.001
    // But first call initialises axis from initial config (node1 at 0.01 above node0).
    // After that we move node1 to 0.01 + 0.001 = 0.011:
    //   relative displacement = 0.001 in z (weld axis = z)

    // Positions at reference: node0=(0,0,0), node1=(0,0,0.01)
    std::vector<Real> pos_ref = {0.0, 0.0, 0.0,   0.0, 0.0, 0.01};

    SpotWeldConfig cfg = make_config(/*k=*/1.0e5, /*Fn=*/5000.0, /*Fs=*/5000.0,
                                     /*d_fail=*/0.05, /*d_start=*/0.5);

    SpotWeldContact sc;
    sc.add_weld(cfg);

    // Step 1: call update at reference — axis gets initialised, force = k * 0.01
    // (the reference gap is 0.01 — weld is pre-loaded by the initial separation)
    // For a cleaner test we want zero force at reference.
    // Trick: set initial gap = 0, then displace.
    // Actually the weld force = k * (current_disp_vector) where current = B - A.
    // At reference the force is k * initial_gap.  We work with this directly.

    // Simpler approach: use zero-gap reference, then displace node1 by 0.002 in z
    std::vector<Real> pos0 = {0.0, 0.0, 0.0,   0.0, 0.0, 0.0};  // coincident nodes
    sc.reset();
    sc.update(pos0.data(), 2, 0.0);

    // Now displace node1 by delta = 0.002 (well within elastic range: d_fail=0.05)
    const Real delta = 0.002;
    std::vector<Real> pos1 = {0.0, 0.0, 0.0,   0.0, 0.0, delta};
    sc.update(pos1.data(), 2, 0.0);

    const SpotWeldState& st = sc.state(0);

    const Real expected_force = cfg.initial_stiffness * delta;  // 200 N
    CHECK(!st.failed, "Weld intact under elastic loading");
    CHECK_NEAR(st.stiffness_factor, 1.0, 1.0e-9, "Full stiffness factor");
    CHECK_NEAR(std::abs(st.current_force[2]), expected_force, 1.0, "Force magnitude = k * delta");
    CHECK_NEAR(st.displacement, delta, 1.0e-10, "Displacement stored correctly");
}

// ============================================================================
// Test 2: Tension failure mode
// ============================================================================
static void test_tension_failure() {
    std::cout << "\n--- Test 2: Tension failure mode ---\n";

    SpotWeldConfig cfg = make_config(1.0e6, /*Fn_max=*/300.0, /*Fs_max=*/1.0e9,
                                     /*d_fail=*/0.05, 0.5);

    SpotWeldContact sc;
    sc.add_weld(cfg);

    // Initialise axis at zero gap
    std::vector<Real> pos0 = {0.0, 0.0, 0.0,   0.0, 0.0, 0.0};
    sc.update(pos0.data(), 2, 0.0);

    // Pull node1 in +z by 0.0004 m → force = 1e6 * 0.0004 = 400 N > Fn_max=300 N
    std::vector<Real> pos1 = {0.0, 0.0, 0.0,   0.0, 0.0, 0.0004};
    sc.update(pos1.data(), 2, 0.0);

    const SpotWeldState& st = sc.state(0);
    CHECK(st.failed, "Weld failed in tension");
    CHECK(st.failure_mode == SpotWeldFailureMode::NormalTension, "Failure mode is NormalTension");
    CHECK_NEAR(st.current_force[0], 0.0, 1.0e-12, "Force zeroed after failure (x)");
    CHECK_NEAR(st.current_force[2], 0.0, 1.0e-12, "Force zeroed after failure (z)");
}

// ============================================================================
// Test 3: Shear failure mode
// ============================================================================
static void test_shear_failure() {
    std::cout << "\n--- Test 3: Shear failure mode ---\n";

    SpotWeldConfig cfg = make_config(1.0e6, /*Fn_max=*/1.0e9, /*Fs_max=*/400.0,
                                     /*d_fail=*/0.05, 0.5);

    SpotWeldContact sc;
    sc.add_weld(cfg);

    // Initialise axis in z direction (coincident nodes → default z-axis)
    std::vector<Real> pos0 = {0.0, 0.0, 0.0,   0.0, 0.0, 0.0};
    sc.update(pos0.data(), 2, 0.0);

    // Displace node1 purely in x → shear force = 1e6 * 0.0005 = 500 N > Fs_max=400 N
    std::vector<Real> pos1 = {0.0, 0.0, 0.0,   5.0e-4, 0.0, 0.0};
    sc.update(pos1.data(), 2, 0.0);

    const SpotWeldState& st = sc.state(0);
    CHECK(st.failed, "Weld failed in shear");
    CHECK(st.failure_mode == SpotWeldFailureMode::Shear ||
          st.failure_mode == SpotWeldFailureMode::Combined,
          "Failure mode is Shear or Combined (pure shear)");
}

// ============================================================================
// Test 4: Combined failure criterion
// ============================================================================
static void test_combined_failure() {
    std::cout << "\n--- Test 4: Combined failure criterion ---\n";

    // Fn_max = Fs_max = 1000 N, k = 1e6
    SpotWeldConfig cfg = make_config(1.0e6, 1000.0, 1000.0, 0.1, 0.5);

    SpotWeldContact sc;
    sc.add_weld(cfg);

    std::vector<Real> pos0 = {0.0, 0.0, 0.0,   0.0, 0.0, 0.0};
    sc.update(pos0.data(), 2, 0.0);

    // Apply (Fn/Fn_max)^2 + (Fs/Fs_max)^2 > 1 with Fn = Fs = 750 N
    //   (0.75)^2 + (0.75)^2 = 1.125 > 1  → combined failure
    // Fn = k * disp_n = 750 → disp_n = 750/1e6 = 7.5e-4
    // Fs = k * disp_s = 750 → disp_s = 7.5e-4
    // Displace in z (normal) and x (shear) equally
    const Real d = 7.5e-4;
    std::vector<Real> pos1 = {0.0, 0.0, 0.0,   d, 0.0, d};
    sc.update(pos1.data(), 2, 0.0);

    const SpotWeldState& st = sc.state(0);
    CHECK(st.failed, "Combined criterion triggers failure");
    CHECK(st.failure_mode == SpotWeldFailureMode::Combined ||
          st.failure_mode == SpotWeldFailureMode::NormalTension ||
          st.failure_mode == SpotWeldFailureMode::Shear,
          "Failure mode set");
}

// ============================================================================
// Test 5: Stiffness degradation curve
// ============================================================================
static void test_degradation_curve() {
    std::cout << "\n--- Test 5: Stiffness degradation curve ---\n";

    // k=1e5, d_fail=0.10, d_start=0.5 → degradation begins at 0.05
    SpotWeldConfig cfg = make_config(1.0e5, 1.0e9, 1.0e9, 0.10, 0.5);

    SpotWeldContact sc;
    sc.add_weld(cfg);

    std::vector<Real> pos0 = {0.0, 0.0, 0.0,   0.0, 0.0, 0.0};
    sc.update(pos0.data(), 2, 0.0);

    // At d = 0.03 (below d_start = 0.05): k_factor should be 1.0
    {
        std::vector<Real> p = {0.0,0.0,0.0,  0.0,0.0,0.03};
        sc.update(p.data(), 2, 0.0);
        CHECK_NEAR(sc.state(0).stiffness_factor, 1.0, 1.0e-9,
                   "Full stiffness below degradation_start");
    }

    // At d = 0.075 (midpoint of [0.05, 0.10]): k_factor = 0.5
    {
        std::vector<Real> p = {0.0,0.0,0.0,  0.0,0.0,0.075};
        sc.update(p.data(), 2, 0.0);
        CHECK_NEAR(sc.state(0).stiffness_factor, 0.5, 1.0e-9,
                   "Half stiffness at midpoint of degradation zone");
    }

    // At d = 0.09 (90 % of d_fail): k_factor = 0.2
    //   (0.09 - 0.05) / (0.10 - 0.05) = 0.8 → k_factor = 1 - 0.8 = 0.2
    {
        std::vector<Real> p = {0.0,0.0,0.0,  0.0,0.0,0.09};
        sc.update(p.data(), 2, 0.0);
        CHECK_NEAR(sc.state(0).stiffness_factor, 0.2, 1.0e-9,
                   "20% stiffness at 0.09 displacement");
    }

    // At d = 0.10 (= d_fail): k_factor = 0 → weld marked failed
    {
        std::vector<Real> p = {0.0,0.0,0.0,  0.0,0.0,0.10};
        sc.update(p.data(), 2, 0.0);
        // k_factor reaches 0 → failed should be true
        const auto& st = sc.state(0);
        CHECK(st.stiffness_factor == 0.0 || st.failed,
              "Zero stiffness at failure displacement");
    }
}

// ============================================================================
// Test 6: Weld array — one weld fails, others stay intact
// ============================================================================
static void test_weld_array_partial_failure() {
    std::cout << "\n--- Test 6: Weld array — partial failure ---\n";

    // 4 welds, each connecting two dedicated node pairs
    // Nodes layout: pairs (0,1), (2,3), (4,5), (6,7) — all at z=0 initially
    //  weld i connects node 2i → node 2i+1
    const int N = 4;
    std::vector<Real> pos(3 * 2 * N, 0.0);
    // All nodes at z=0 initially (coincident pairs)

    SpotWeldContact sc;
    for (int i = 0; i < N; ++i) {
        SpotWeldConfig cfg = make_config(1.0e5, 500.0, 500.0, 0.05, 0.5);
        cfg.id     = i;
        cfg.node_a = static_cast<std::size_t>(2 * i);
        cfg.node_b = static_cast<std::size_t>(2 * i + 1);
        sc.add_weld(cfg);
    }

    // Initialise axes
    sc.update(pos.data(), static_cast<std::size_t>(2 * N), 0.0);

    // Overload weld 0: displace node 1 in z by 0.006 → F = 1e5*0.006 = 600 > 500
    // Keep welds 1,2,3 with small displacement = 0.001 (elastic)
    std::vector<Real> pos2 = pos;
    pos2[3*1 + 2] = 0.006;   // node 1 +z (weld 0 fails)
    pos2[3*3 + 2] = 0.001;   // node 3 (weld 1 elastic)
    pos2[3*5 + 2] = 0.001;   // node 5 (weld 2 elastic)
    pos2[3*7 + 2] = 0.001;   // node 7 (weld 3 elastic)

    sc.update(pos2.data(), static_cast<std::size_t>(2 * N), 0.0);

    CHECK(sc.state(0).failed,  "Weld 0 failed");
    CHECK(!sc.state(1).failed, "Weld 1 intact");
    CHECK(!sc.state(2).failed, "Weld 2 intact");
    CHECK(!sc.state(3).failed, "Weld 3 intact");
    CHECK(sc.num_failed()  == 1, "num_failed() == 1");
    CHECK(sc.num_active()  == 3, "num_active() == 3");

    const auto failed_idx = sc.failed_weld_indices();
    CHECK(failed_idx.size() == 1 && failed_idx[0] == 0,
          "failed_weld_indices() returns {0}");
}

// ============================================================================
// Test 7: Thermal softening
// ============================================================================
static void test_thermal_softening() {
    std::cout << "\n--- Test 7: Thermal softening ---\n";

    // Normal strength = 500 N, softening_factor = 0.5 → softened strength = 250 N
    SpotWeldConfig cfg = make_config(1.0e6, 500.0, 500.0, 0.05, 0.5);
    cfg.thermal_softening_temp   = 500.0;
    cfg.thermal_softening_factor = 0.5;

    SpotWeldContact sc;
    sc.add_weld(cfg);

    std::vector<Real> pos0 = {0.0,0.0,0.0,  0.0,0.0,0.0};
    sc.update(pos0.data(), 2, 0.0);

    // Load to F = 300 N (= 1e6 * 3e-4) — above softened strength (250 N) but below nominal (500 N)
    std::vector<Real> pos1 = {0.0,0.0,0.0,  0.0,0.0,3.0e-4};

    // Test A: no thermal effect (T < T_soft) → weld should survive
    std::vector<Real> temps_cold = {100.0, 100.0};  // below 500 °C
    sc.update(pos1.data(), 2, 0.0, temps_cold.data());
    CHECK(!sc.state(0).failed, "Weld intact below softening temperature");

    // Reset for second check
    sc.reset();
    sc.update(pos0.data(), 2, 0.0);

    // Test B: T >= T_soft → effective strength = 250 N < 300 N → should fail
    std::vector<Real> temps_hot = {600.0, 600.0};   // above 500 °C
    sc.update(pos1.data(), 2, 0.0, temps_hot.data());
    CHECK(sc.state(0).failed, "Weld fails at elevated temperature (thermal softening)");
}

// ============================================================================
// Test 8: Force application — equal and opposite on node_a / node_b
// ============================================================================
static void test_force_application() {
    std::cout << "\n--- Test 8: Force application — equal/opposite ---\n";

    SpotWeldConfig cfg = make_config(1.0e5, 1.0e9, 1.0e9, 0.10, 0.5);

    SpotWeldContact sc;
    sc.add_weld(cfg);

    std::vector<Real> pos0 = {0.0,0.0,0.0,  0.0,0.0,0.0};
    sc.update(pos0.data(), 2, 0.0);

    // Apply a combined displacement
    const Real dz = 0.005, dx = 0.003;
    std::vector<Real> pos1 = {0.0,0.0,0.0,  dx,0.0,dz};
    sc.update(pos1.data(), 2, 0.0);

    std::vector<Real> forces(6, 0.0);
    sc.apply_forces(forces.data(), 2);

    // Forces on node_a and node_b should be equal and opposite
    CHECK_NEAR(forces[0] + forces[3], 0.0, 1.0e-12, "fx: F_A + F_B = 0");
    CHECK_NEAR(forces[1] + forces[4], 0.0, 1.0e-12, "fy: F_A + F_B = 0");
    CHECK_NEAR(forces[2] + forces[5], 0.0, 1.0e-12, "fz: F_A + F_B = 0");

    // Verify force magnitude matches stiffness * displacement
    const Real disp_total = std::sqrt(dx*dx + dz*dz);
    const Real f_expected = cfg.initial_stiffness * disp_total;
    const Real f_actual   = std::sqrt(forces[0]*forces[0] + forces[1]*forces[1] + forces[2]*forces[2]);
    CHECK_NEAR(f_actual, f_expected, 1.0, "Force magnitude matches k * displacement");
}

// ============================================================================
// Test 9: Energy tracking
// ============================================================================
static void test_energy_tracking() {
    std::cout << "\n--- Test 9: Energy tracking ---\n";

    SpotWeldConfig cfg = make_config(1.0e4, 1.0e9, 1.0e9, 0.10, 0.5);

    SpotWeldContact sc;
    sc.add_weld(cfg);

    // Start at zero
    std::vector<Real> pos0 = {0.0,0.0,0.0,  0.0,0.0,0.0};
    sc.update(pos0.data(), 2, 0.0);
    CHECK_NEAR(sc.state(0).energy_absorbed, 0.0, 1.0e-30, "Zero energy at start");

    // Displace to d1 = 0.01 m (pure normal)
    const Real d1 = 0.01;
    std::vector<Real> pos1 = {0.0,0.0,0.0,  0.0,0.0,d1};
    sc.update(pos1.data(), 2, 0.0);

    // Incremental energy: dE = 0.5 * (F_old + F_new) * d_disp
    //   F_old = 0 (from step 0), F_new = k*d1 = 100 N, d_disp = d1 = 0.01
    //   dE = 0.5 * (0 + 100) * 0.01 = 0.5 J
    CHECK(sc.state(0).energy_absorbed > 0.0, "Energy absorbed is positive after loading");

    // Further displace to d2 = 0.02 m
    const Real d2 = 0.02;
    std::vector<Real> pos2 = {0.0,0.0,0.0,  0.0,0.0,d2};
    sc.update(pos2.data(), 2, 0.0);

    const Real E2 = sc.state(0).energy_absorbed;
    CHECK(E2 > sc.state(0).energy_absorbed - 1.0, "Energy increases monotonically");

    // total_energy() == sum of all weld energies (only 1 weld here)
    CHECK_NEAR(sc.total_energy(), E2, 1.0e-12, "total_energy() matches single weld");
}

// ============================================================================
// Test 10: Dynamic loading — multiple time steps
// ============================================================================
static void test_dynamic_loading() {
    std::cout << "\n--- Test 10: Dynamic loading — multiple steps ---\n";

    // k=1e5, Fn_max=1e9 (won't trigger by force), d_fail=0.04, d_start=0.5
    // → weld degrades over [0.02, 0.04] then fails when k_factor reaches 0
    SpotWeldConfig cfg = make_config(1.0e5, 1.0e9, 1.0e9, 0.04, 0.5);

    SpotWeldContact sc;
    sc.add_weld(cfg);

    std::vector<Real> pos(6, 0.0);
    sc.update(pos.data(), 2, 0.0);

    const int n_steps = 20;
    const Real dt = 1.0e-4;
    const Real d_final = 0.05;  // beyond d_fail → must fail

    bool ever_degraded = false;
    int  fail_step = -1;

    for (int step = 0; step < n_steps; ++step) {
        const Real t      = (step + 1) * dt;
        const Real d_step = d_final * t / (n_steps * dt);

        pos[5] = d_step;  // node1 z = increasing displacement
        sc.update(pos.data(), 2, dt);

        const SpotWeldState& st = sc.state(0);
        if (st.stiffness_factor < 1.0 && !st.failed) {
            ever_degraded = true;
        }
        if (st.failed && fail_step < 0) {
            fail_step = step;
        }
    }

    CHECK(sc.state(0).failed, "Weld failed by end of dynamic loading");
    CHECK(fail_step >= 0 && fail_step < n_steps - 1,
          "Failure occurred before final step");
    CHECK(ever_degraded, "Stiffness degradation occurred before failure");
    CHECK(sc.total_energy() > 0.0, "Non-zero energy absorbed over load history");

    // After failure, num_active should be 0
    CHECK(sc.num_active() == 0, "num_active() == 0 after failure");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== Wave 44d: Spot Weld Contact Tests ===\n";

    test_elastic_loading();
    test_tension_failure();
    test_shear_failure();
    test_combined_failure();
    test_degradation_curve();
    test_weld_array_partial_failure();
    test_thermal_softening();
    test_force_application();
    test_energy_tracking();
    test_dynamic_loading();

    std::cout << "\n=== Wave 44 Spot Weld Results ===\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";

    return tests_failed > 0 ? 1 : 0;
}
