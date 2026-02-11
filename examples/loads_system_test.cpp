/**
 * @file loads_system_test.cpp
 * @brief Comprehensive test for Wave 4: Loads, load curves, initial conditions
 */

#include <nexussim/fem/load_curve.hpp>
#include <nexussim/fem/loads.hpp>
#include <nexussim/fem/initial_conditions.hpp>
#include <iostream>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { \
        std::cout << "[PASS] " << msg << "\n"; \
        tests_passed++; \
    } else { \
        std::cout << "[FAIL] " << msg << "\n"; \
        tests_failed++; \
    } \
} while(0)

static bool near(Real a, Real b, Real tol = 1.0e-6) {
    return std::fabs(a - b) < tol;
}

// ==========================================================================
// Test 1: Load Curve Interpolation
// ==========================================================================
void test_load_curve() {
    std::cout << "\n=== Test 1: Load Curve Interpolation ===\n";

    LoadCurve lc;
    lc.id = 1;
    lc.add_point(0.0, 0.0);
    lc.add_point(0.5, 1.0);
    lc.add_point(1.0, 1.0);
    lc.add_point(2.0, 0.0);

    CHECK(near(lc.evaluate(0.0), 0.0), "t=0: value = 0");
    CHECK(near(lc.evaluate(0.25), 0.5), "t=0.25: midpoint interpolation = 0.5");
    CHECK(near(lc.evaluate(0.5), 1.0), "t=0.5: value = 1.0");
    CHECK(near(lc.evaluate(0.75), 1.0), "t=0.75: plateau = 1.0");
    CHECK(near(lc.evaluate(1.5), 0.5), "t=1.5: decreasing = 0.5");
    CHECK(near(lc.evaluate(2.0), 0.0), "t=2.0: value = 0");

    // Default extrapolation (constant)
    CHECK(near(lc.evaluate(3.0), 0.0), "t=3.0: constant extrap = 0");
    CHECK(near(lc.evaluate(-1.0), 0.0), "t=-1.0: constant extrap = 0");
}

// ==========================================================================
// Test 2: Load Curve Extrapolation Modes
// ==========================================================================
void test_extrap_modes() {
    std::cout << "\n=== Test 2: Extrapolation Modes ===\n";

    // Zero extrapolation
    LoadCurve lc_zero;
    lc_zero.add_point(0.0, 5.0);
    lc_zero.add_point(1.0, 10.0);
    lc_zero.extrap_low = ExtrapolationMode::Zero;
    lc_zero.extrap_high = ExtrapolationMode::Zero;

    CHECK(near(lc_zero.evaluate(-1.0), 0.0), "Zero extrap below: 0");
    CHECK(near(lc_zero.evaluate(2.0), 0.0), "Zero extrap above: 0");

    // Linear extrapolation
    LoadCurve lc_lin;
    lc_lin.add_point(0.0, 0.0);
    lc_lin.add_point(1.0, 2.0);
    lc_lin.extrap_high = ExtrapolationMode::Linear;

    CHECK(near(lc_lin.evaluate(2.0), 4.0), "Linear extrap above: 4.0");
}

// ==========================================================================
// Test 3: Load Curve Manager
// ==========================================================================
void test_curve_manager() {
    std::cout << "\n=== Test 3: Load Curve Manager ===\n";

    LoadCurveManager mgr;
    auto& c1 = mgr.add_curve(1);
    c1.add_point(0.0, 0.0);
    c1.add_point(1.0, 100.0);

    auto& c2 = mgr.add_curve(2);
    c2.add_point(0.0, 1.0);
    c2.add_point(1.0, 0.0);

    CHECK(mgr.num_curves() == 2, "Two curves added");
    CHECK(near(mgr.evaluate(1, 0.5), 50.0), "Curve 1 at t=0.5: 50");
    CHECK(near(mgr.evaluate(2, 0.5), 0.5), "Curve 2 at t=0.5: 0.5");
    CHECK(near(mgr.evaluate(99, 0.5), 1.0), "Missing curve: default 1.0");
}

// ==========================================================================
// Test 4: Nodal Force with Load Curve
// ==========================================================================
void test_nodal_force() {
    std::cout << "\n=== Test 4: Nodal Force with Curve ===\n";

    LoadCurveManager curves;
    auto& c = curves.add_curve(1);
    c.add_point(0.0, 0.0);
    c.add_point(0.01, 1.0);
    c.add_point(0.02, 0.5);

    LoadManager loads;
    loads.set_curve_manager(&curves);

    auto& f = loads.add_load(LoadType::NodalForce, 1);
    f.magnitude = 1000.0;  // 1000 N
    f.direction[0] = 0.0; f.direction[1] = 0.0; f.direction[2] = -1.0;
    f.dof = 2;  // z-direction
    f.node_set = {0, 1};
    f.load_curve_id = 1;

    const int num_nodes = 4;
    Real positions[12] = {};
    Real velocities[12] = {};
    Real forces[12] = {};
    Real masses[4] = {1,1,1,1};

    // At t=0: curve = 0 → no force
    loads.apply_loads(0.0, num_nodes, positions, velocities, forces, masses, 0.001);
    CHECK(near(forces[2], 0.0), "t=0: no force (curve=0)");

    // At t=0.01: curve = 1 → full force
    std::fill(forces, forces+12, 0.0);
    loads.apply_loads(0.01, num_nodes, positions, velocities, forces, masses, 0.001);
    CHECK(near(forces[2], -1000.0), "t=0.01: node 0 Fz = -1000N");
    CHECK(near(forces[5], -1000.0), "t=0.01: node 1 Fz = -1000N");
    CHECK(near(forces[8], 0.0), "t=0.01: node 2 Fz = 0 (not in set)");

    // At t=0.02: curve = 0.5
    std::fill(forces, forces+12, 0.0);
    loads.apply_loads(0.02, num_nodes, positions, velocities, forces, masses, 0.001);
    CHECK(near(forces[2], -500.0), "t=0.02: node 0 Fz = -500N (half)");
}

// ==========================================================================
// Test 5: Gravity Load
// ==========================================================================
void test_gravity() {
    std::cout << "\n=== Test 5: Gravity Load ===\n";

    LoadManager loads;

    auto& g = loads.add_load(LoadType::Gravity, 1);
    g.magnitude = 9.81;
    g.direction[0] = 0.0; g.direction[1] = 0.0; g.direction[2] = -1.0;
    // Empty node_set → all nodes

    const int num_nodes = 3;
    Real positions[9] = {};
    Real velocities[9] = {};
    Real forces[9] = {};
    Real masses[3] = {2.0, 3.0, 1.0};

    loads.apply_loads(0.0, num_nodes, positions, velocities, forces, masses, 0.001);

    CHECK(near(forces[2], -2.0*9.81), "Node 0: Fz = -m*g = -19.62");
    CHECK(near(forces[5], -3.0*9.81), "Node 1: Fz = -29.43");
    CHECK(near(forces[8], -1.0*9.81), "Node 2: Fz = -9.81");
    CHECK(near(forces[0], 0.0), "No force in x");
}

// ==========================================================================
// Test 6: Imposed Velocity
// ==========================================================================
void test_imposed_velocity() {
    std::cout << "\n=== Test 6: Imposed Velocity ===\n";

    LoadCurveManager curves;
    auto& c = curves.add_curve(1);
    c.add_point(0.0, 0.0);
    c.add_point(0.01, 1.0);  // Ramp up over 10ms

    LoadManager loads;
    loads.set_curve_manager(&curves);

    auto& iv = loads.add_load(LoadType::ImposedVelocity, 1);
    iv.magnitude = 10.0;  // 10 m/s target
    iv.dof = 0;           // x-direction
    iv.node_set = {0, 1};
    iv.load_curve_id = 1;

    const int num_nodes = 4;
    Real positions[12] = {};
    Real velocities[12] = {0,0,0, 0,0,0, 5,0,0, 0,0,0};
    Real forces[12] = {};
    Real masses[4] = {1,1,1,1};

    // At t=0.005: curve = 0.5, target = 5 m/s
    loads.apply_loads(0.005, num_nodes, positions, velocities, forces, masses, 0.001);
    CHECK(near(velocities[0], 5.0), "t=5ms: node 0 vx = 5 m/s");
    CHECK(near(velocities[3], 5.0), "t=5ms: node 1 vx = 5 m/s");
    CHECK(near(velocities[6], 5.0), "Node 2 unchanged (not in set)");

    // At t=0.01: curve = 1.0, target = 10 m/s
    loads.apply_loads(0.01, num_nodes, positions, velocities, forces, masses, 0.001);
    CHECK(near(velocities[0], 10.0), "t=10ms: node 0 vx = 10 m/s");
}

// ==========================================================================
// Test 7: Initial Velocity
// ==========================================================================
void test_initial_velocity() {
    std::cout << "\n=== Test 7: Initial Velocity ===\n";

    LoadManager loads;

    auto& iv = loads.add_load(LoadType::InitialVelocity, 1);
    iv.magnitude = 50.0;  // 50 m/s
    iv.direction[0] = 1.0; iv.direction[1] = 0.0; iv.direction[2] = 0.0;
    iv.node_set = {0, 1, 2};

    const int num_nodes = 4;
    Real positions[12] = {};
    Real velocities[12] = {};
    Real forces[12] = {};
    Real masses[4] = {1,1,1,1};

    loads.apply_loads(0.0, num_nodes, positions, velocities, forces, masses, 0.001);

    CHECK(near(velocities[0], 50.0), "Node 0: vx = 50 m/s");
    CHECK(near(velocities[3], 50.0), "Node 1: vx = 50 m/s");
    CHECK(near(velocities[6], 50.0), "Node 2: vx = 50 m/s");
    CHECK(near(velocities[9], 0.0), "Node 3: vx = 0 (not in set)");

    // Apply again: should NOT re-apply (already applied)
    std::fill(velocities, velocities+12, 0.0);
    loads.apply_loads(0.001, num_nodes, positions, velocities, forces, masses, 0.001);
    CHECK(near(velocities[0], 0.0), "Second apply: not re-applied");
}

// ==========================================================================
// Test 8: Initial Condition Manager
// ==========================================================================
void test_initial_conditions() {
    std::cout << "\n=== Test 8: Initial Conditions ===\n";

    InitialConditionManager ic_mgr;

    auto& vel_ic = ic_mgr.add_condition(InitialConditionType::Velocity, 1);
    vel_ic.value[0] = 100.0;
    vel_ic.value[1] = 0.0;
    vel_ic.value[2] = -5.0;
    vel_ic.node_set = {0, 1};

    auto& temp_ic = ic_mgr.add_condition(InitialConditionType::Temperature, 2);
    temp_ic.scalar_value = 300.0;  // 300 K

    CHECK(ic_mgr.num_conditions() == 2, "Two initial conditions");

    const int num_nodes = 4;
    Real velocities[12] = {};
    Real temperatures[4] = {0, 0, 0, 0};

    ic_mgr.apply(num_nodes, velocities, nullptr, temperatures);

    CHECK(near(velocities[0], 100.0), "Node 0: vx = 100");
    CHECK(near(velocities[2], -5.0), "Node 0: vz = -5");
    CHECK(near(velocities[6], 0.0), "Node 2: unaffected");
    CHECK(near(temperatures[0], 300.0), "All nodes: T = 300K");
    CHECK(near(temperatures[3], 300.0), "Node 3: T = 300K");

    // Apply again: should not re-apply
    std::fill(velocities, velocities+12, 0.0);
    ic_mgr.apply(num_nodes, velocities);
    CHECK(near(velocities[0], 0.0), "Not re-applied");
}

// ==========================================================================
// Test 9: Crash Pulse Load Curve
// ==========================================================================
void test_crash_pulse() {
    std::cout << "\n=== Test 9: Crash Pulse ===\n";

    // Typical crash deceleration pulse
    LoadCurve pulse;
    pulse.id = 1;
    pulse.add_point(0.000, 0.0);     // Start
    pulse.add_point(0.005, -20.0);   // Initial spike (20g)
    pulse.add_point(0.020, -15.0);   // Sustained decel
    pulse.add_point(0.050, -10.0);   // Decreasing
    pulse.add_point(0.080, 0.0);     // End of crash

    CHECK(near(pulse.evaluate(0.005), -20.0), "Peak at 5ms");
    CHECK(pulse.evaluate(0.010) < 0.0, "Negative during crash");
    CHECK(near(pulse.evaluate(0.080), 0.0), "Zero at end");
    CHECK(pulse.evaluate(0.005) < pulse.evaluate(0.050), "Peak > sustained");
}

// ==========================================================================
// Test 10: Multiple Loads Combined
// ==========================================================================
void test_combined_loads() {
    std::cout << "\n=== Test 10: Combined Loads ===\n";

    LoadManager loads;

    // Gravity
    auto& g = loads.add_load(LoadType::Gravity, 1);
    g.magnitude = 9.81;
    g.direction[2] = -1.0;

    // Upward force on node 0
    auto& f = loads.add_load(LoadType::NodalForce, 2);
    f.magnitude = 100.0;
    f.direction[2] = 1.0;
    f.dof = 2;
    f.node_set = {0};

    const int num_nodes = 2;
    Real positions[6] = {};
    Real velocities[6] = {};
    Real forces[6] = {};
    Real masses[2] = {5.0, 5.0};

    loads.apply_loads(0.0, num_nodes, positions, velocities, forces, masses, 0.001);

    // Node 0: gravity (-5*9.81 = -49.05) + upward (100) = 50.95
    Real expected_0 = -5.0*9.81 + 100.0;
    CHECK(near(forces[2], expected_0, 0.1), "Node 0: combined gravity + force");

    // Node 1: gravity only (-5*9.81 = -49.05)
    CHECK(near(forces[5], -5.0*9.81, 0.01), "Node 1: gravity only");
}

// ==========================================================================
// Main
// ==========================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "NexusSim Wave 4: Loads System Test\n";
    std::cout << "========================================\n";

    test_load_curve();
    test_extrap_modes();
    test_curve_manager();
    test_nodal_force();
    test_gravity();
    test_imposed_velocity();
    test_initial_velocity();
    test_initial_conditions();
    test_crash_pulse();
    test_combined_loads();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    std::cout << "========================================\n";

    return (tests_failed > 0) ? 1 : 0;
}
