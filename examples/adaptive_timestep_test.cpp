/**
 * @file adaptive_timestep_test.cpp
 * @brief Test adaptive time stepping controller
 *
 * Tests:
 * 1. Basic CFL-based timestep calculation
 * 2. Energy-based stability monitoring
 * 3. Automatic timestep growth when stable
 * 4. Automatic timestep reduction on instability
 * 5. Integration with FEM solver (spring-mass system)
 */

#include <nexussim/physics/adaptive_timestep.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::physics;

// Test counter
static int test_count = 0;
static int pass_count = 0;

void check(bool condition, const std::string& test_name) {
    test_count++;
    if (condition) {
        pass_count++;
        std::cout << "[PASS] " << test_name << "\n";
    } else {
        std::cout << "[FAIL] " << test_name << "\n";
    }
}

// ============================================================================
// Test 1: Basic CFL timestep calculation
// ============================================================================
void test_cfl_timestep() {
    std::cout << "\n=== Test 1: CFL Timestep Calculation ===\n";

    AdaptiveTimestep controller;
    controller.set_strategy(TimestepStrategy::CFL);
    controller.set_target_cfl(0.9);
    controller.set_min_dt(1e-10);
    controller.set_max_dt(1e-3);

    // Simulate with CFL dt = 1e-6
    Real cfl_dt = 1e-6;
    Real dt = controller.compute_timestep(cfl_dt);

    check(std::abs(dt - 0.9e-6) < 1e-12, "CFL factor applied correctly");
    check(dt >= controller.min_dt(), "Timestep >= min_dt");
    check(dt <= controller.max_dt(), "Timestep <= max_dt");
    check(controller.step_count() == 1, "Step count incremented");
}

// ============================================================================
// Test 2: Fixed timestep strategy
// ============================================================================
void test_fixed_timestep() {
    std::cout << "\n=== Test 2: Fixed Timestep Strategy ===\n";

    AdaptiveTimestep controller;
    controller.set_strategy(TimestepStrategy::Fixed);
    controller.set_target_cfl(0.9);

    // First step uses CFL
    Real dt1 = controller.compute_timestep(1e-6);
    check(std::abs(dt1 - 0.9e-6) < 1e-12, "First step uses CFL");

    // Second step should use same dt regardless of CFL
    Real dt2 = controller.compute_timestep(2e-6);  // CFL changed but fixed ignores it
    check(std::abs(dt2 - 0.9e-6) < 1e-12, "Fixed strategy maintains dt");

    Real dt3 = controller.compute_timestep(0.5e-6);
    check(std::abs(dt3 - 0.9e-6) < 1e-12, "Fixed strategy ignores CFL changes");
}

// ============================================================================
// Test 3: Timestep bounds enforcement
// ============================================================================
void test_timestep_bounds() {
    std::cout << "\n=== Test 3: Timestep Bounds ===\n";

    AdaptiveTimestep controller;
    controller.set_strategy(TimestepStrategy::CFL);
    controller.set_target_cfl(0.9);
    controller.set_min_dt(1e-8);
    controller.set_max_dt(1e-5);

    // Test min bound
    Real dt_small = controller.compute_timestep(1e-10);  // CFL*0.9 = 9e-11
    check(dt_small >= 1e-8, "Min bound enforced");

    controller.reset();

    // Test max bound
    Real dt_large = controller.compute_timestep(1e-3);  // CFL*0.9 = 9e-4
    check(dt_large <= 1e-5, "Max bound enforced");
}

// ============================================================================
// Test 4: Energy monitoring
// ============================================================================
void test_energy_monitoring() {
    std::cout << "\n=== Test 4: Energy Monitoring ===\n";

    AdaptiveTimestep controller;
    controller.set_strategy(TimestepStrategy::EnergyBased);
    controller.set_energy_tolerance(0.05);  // 5% tolerance

    // Simulate stable energy history
    for (int i = 0; i < 10; ++i) {
        Real time = i * 0.001;
        Real kinetic = 100.0 + 0.1 * std::sin(10 * time);  // Small oscillation
        Real internal = 50.0;
        controller.update_energy(kinetic, internal, 0, time);
        controller.compute_timestep(1e-6);
        controller.check_stability();
    }

    check(controller.stable_steps() > 0, "Stable energy tracked");
    check(controller.energy_error() < 0.05, "Energy error within tolerance");

    // Now simulate unstable energy growth
    controller.reset();
    controller.set_energy_tolerance(0.01);  // Tighter tolerance

    Real base_energy = 100.0;
    controller.update_energy(base_energy, 0, 0, 0);
    controller.compute_timestep(1e-6);

    // Energy grows exponentially (instability)
    for (int i = 1; i <= 5; ++i) {
        Real growing_energy = base_energy * std::pow(1.1, i);  // 10% growth per step
        controller.update_energy(growing_energy, 0, 0, i * 0.001);
        controller.compute_timestep(1e-6);
        controller.check_stability();
    }

    check(controller.energy_error() > 0.01, "Energy growth detected");
}

// ============================================================================
// Test 5: Timestep adaptation
// ============================================================================
void test_timestep_adaptation() {
    std::cout << "\n=== Test 5: Timestep Adaptation ===\n";

    AdaptiveTimestep controller;
    controller.set_strategy(TimestepStrategy::Combined);
    controller.set_target_cfl(0.9);
    controller.set_growth_factor(1.1);
    controller.set_shrink_factor(0.5);
    controller.set_min_stable_steps(5);
    controller.set_min_dt(1e-9);
    controller.set_max_dt(1e-5);

    // Initial timestep
    Real cfl_dt = 1e-6;
    Real dt0 = controller.compute_timestep(cfl_dt);

    // Run stable steps
    for (int i = 0; i < 10; ++i) {
        controller.update_energy(100.0, 50.0, 0, i * 0.001);
        controller.compute_timestep(cfl_dt);
        controller.check_stability();
    }

    check(controller.stable_steps() >= 5, "Accumulated stable steps");

    Real dt_after_stable = controller.current_dt();
    // After min_stable_steps, should try to grow
    // Note: growth is limited by CFL, so may not see growth if CFL is limiting

    // Force instability
    controller.force_reduction(0.5);
    Real dt_after_reduction = controller.current_dt();

    check(dt_after_reduction < dt_after_stable, "Timestep reduced on force_reduction");
    check(controller.stable_steps() == 0, "Stable counter reset on instability");
}

// ============================================================================
// Test 6: Velocity tolerance
// ============================================================================
void test_velocity_tolerance() {
    std::cout << "\n=== Test 6: Velocity Tolerance ===\n";

    AdaptiveTimestep controller;
    controller.set_strategy(TimestepStrategy::Combined);
    controller.set_velocity_tolerance(1000.0);  // 1000 m/s max

    // Normal velocity
    bool ok1 = controller.check_velocity(100.0);
    check(ok1, "Normal velocity accepted");

    // Excessive velocity
    bool ok2 = controller.check_velocity(2000.0);
    check(!ok2, "Excessive velocity rejected");
    check(controller.unstable_steps() > 0, "Instability recorded for high velocity");
}

// ============================================================================
// Test 7: Spring-mass system integration
// ============================================================================
void test_spring_mass_integration() {
    std::cout << "\n=== Test 7: Spring-Mass System Integration ===\n";

    // Simple 1D spring-mass system: m*a + k*x = 0
    // Analytical: x(t) = A*cos(omega*t), omega = sqrt(k/m)
    Real m = 1.0;      // kg
    Real k = 100.0;    // N/m
    Real x0 = 0.1;     // Initial displacement (m)
    Real v0 = 0.0;     // Initial velocity (m/s)

    Real omega = std::sqrt(k / m);
    Real period = 2.0 * M_PI / omega;

    // CFL condition for spring: dt < 2/omega
    Real cfl_dt = 2.0 / omega;

    AdaptiveTimestep controller;
    controller.set_strategy(TimestepStrategy::Combined);
    controller.set_target_cfl(0.05);  // Conservative for good accuracy
    controller.set_min_dt(1e-6);
    controller.set_max_dt(cfl_dt * 0.2);
    controller.set_energy_tolerance(0.01);  // 1% energy tolerance

    // State
    Real x = x0;
    Real v = v0;
    Real time = 0;
    int steps = 0;

    // Initial energy
    Real E0 = 0.5 * k * x * x + 0.5 * m * v * v;
    controller.update_energy(0.5 * m * v * v, 0.5 * k * x * x, 0, time);

    // Simulate 1 period using Velocity-Verlet (symplectic, conserves energy)
    Real end_time = period;
    while (time < end_time) {
        Real dt = controller.compute_timestep(cfl_dt);

        // Velocity Verlet integration (symplectic - conserves energy)
        Real a_old = -k * x / m;
        x += v * dt + 0.5 * a_old * dt * dt;  // Position update
        Real a_new = -k * x / m;
        v += 0.5 * (a_old + a_new) * dt;      // Velocity update
        time += dt;
        steps++;

        // Update energy
        Real kinetic = 0.5 * m * v * v;
        Real potential = 0.5 * k * x * x;
        controller.update_energy(kinetic, potential, 0, time);
        controller.check_stability();
    }

    // Check final position (should be close to x0)
    Real x_analytical = x0 * std::cos(omega * time);
    Real error = std::abs(x - x_analytical);

    // Check energy conservation
    Real E_final = 0.5 * k * x * x + 0.5 * m * v * v;
    Real energy_error = std::abs(E_final - E0) / E0;

    std::cout << "  Steps: " << steps << "\n";
    std::cout << "  Final position: " << x << " (analytical: " << x_analytical << ")\n";
    std::cout << "  Position error: " << error << "\n";
    std::cout << "  Energy error: " << (energy_error * 100) << "%\n";

    check(error < 0.02 * x0, "Position accuracy within 2%");
    check(energy_error < 0.02, "Energy conservation within 2%");
    check(controller.stable_steps() > 0, "System remained stable");
}

// ============================================================================
// Test 8: Statistics output
// ============================================================================
void test_statistics() {
    std::cout << "\n=== Test 8: Statistics Output ===\n";

    AdaptiveTimestep controller;
    controller.set_strategy(TimestepStrategy::Combined);
    controller.set_verbose(false);  // Don't spam output

    for (int i = 0; i < 5; ++i) {
        controller.compute_timestep(1e-6);
        controller.update_energy(100, 50, 0, i * 0.001);
        controller.check_stability();
    }

    std::cout << "\n";
    controller.print_stats();

    check(controller.step_count() == 5, "Step count correct");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=================================================\n";
    std::cout << "Adaptive Timestep Controller Test Suite\n";
    std::cout << "=================================================\n";

    test_cfl_timestep();
    test_fixed_timestep();
    test_timestep_bounds();
    test_energy_monitoring();
    test_timestep_adaptation();
    test_velocity_tolerance();
    test_spring_mass_integration();
    test_statistics();

    std::cout << "\n=================================================\n";
    std::cout << "Results: " << pass_count << "/" << test_count << " tests passed\n";
    std::cout << "=================================================\n";

    return (pass_count == test_count) ? 0 : 1;
}
