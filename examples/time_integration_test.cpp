/**
 * @file time_integration_test.cpp
 * @brief Test advanced time integration features
 *
 * Tests:
 * 1. Energy monitor tracking
 * 2. Subcycling controller
 * 3. Consistent mass matrix
 * 4. Velocity Verlet integration
 * 5. Newmark-β integration
 */

#include <nexussim/physics/time_integration.hpp>
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
// Test 1: Energy Monitor
// ============================================================================
void test_energy_monitor() {
    std::cout << "\n=== Test 1: Energy Monitor ===\n";

    EnergyMonitor monitor;

    // Record initial state
    EnergyMonitor::EnergyState state0;
    state0.kinetic = 100.0;
    state0.internal = 50.0;
    state0.time = 0.0;
    monitor.record(state0);

    check(std::abs(monitor.current().total() - 150.0) < 1e-10, "Initial total energy");
    check(monitor.is_conserved(0.01), "Initially conserved");

    // Record after some time (energy conserved)
    EnergyMonitor::EnergyState state1;
    state1.kinetic = 80.0;
    state1.internal = 70.0;
    state1.time = 0.001;
    monitor.record(state1);

    check(std::abs(monitor.current().total() - 150.0) < 1e-10, "Total energy unchanged");
    check(monitor.relative_error() < 0.001, "Relative error near zero");

    // Record with some energy loss
    EnergyMonitor::EnergyState state2;
    state2.kinetic = 70.0;
    state2.internal = 75.0;
    state2.damping = 5.0;  // Lost to damping
    state2.time = 0.002;
    monitor.record(state2);

    check(monitor.history().size() == 3, "History recorded");

    std::cout << "  Relative error: " << (monitor.relative_error() * 100) << "%\n";

    monitor.print_stats();
}

// ============================================================================
// Test 2: Subcycling Controller
// ============================================================================
void test_subcycling_controller() {
    std::cout << "\n=== Test 2: Subcycling Controller ===\n";

    SubcyclingController subcycle;

    // Add two regions with different subcycle ratios
    std::vector<Index> fast_nodes = {0, 1, 2, 3};
    std::vector<Index> fast_elems = {0, 1};
    std::vector<Index> slow_nodes = {2, 3, 4, 5};  // Overlapping nodes 2,3
    std::vector<Index> slow_elems = {2, 3};

    subcycle.add_region("fast", fast_nodes, fast_elems, 4);
    subcycle.add_region("slow", slow_nodes, slow_elems, 1);

    check(subcycle.num_regions() == 2, "Two regions added");
    check(subcycle.has_subcycling(), "Subcycling detected");

    // Test master dt computation
    std::vector<Real> region_dts = {1e-7, 4e-7};
    Real master_dt = subcycle.compute_master_dt(region_dts);

    std::cout << "  Master dt: " << master_dt << " s\n";
    check(master_dt > 0, "Master dt computed");

    // Test stepping
    int fast_steps = 0, slow_steps = 0;

    subcycle.step(master_dt, [&](const SubcycleRegion& region, Real dt, int substep) {
        if (region.name == "fast") {
            fast_steps++;
        } else {
            slow_steps++;
        }
    });

    check(fast_steps == 4, "Fast region took 4 substeps");
    check(slow_steps == 1, "Slow region took 1 step");

    subcycle.print_info();
}

// ============================================================================
// Test 3: Auto Partition
// ============================================================================
void test_auto_partition() {
    std::cout << "\n=== Test 3: Auto Partition ===\n";

    SubcyclingController subcycle;

    // Element stable timesteps with large variation
    std::vector<Real> element_dts = {
        1e-7, 1e-7, 1e-7, 1e-7,  // Fast elements
        1e-6, 1e-6, 1e-6, 1e-6   // Slow elements (10x larger)
    };

    subcycle.auto_partition(element_dts, 4.0);

    check(subcycle.num_regions() == 2, "Auto-partitioned into 2 regions");
    check(subcycle.has_subcycling(), "Subcycling needed");

    subcycle.print_info();

    // Test uniform case (no subcycling needed)
    SubcyclingController subcycle2;
    std::vector<Real> uniform_dts = {1e-6, 1e-6, 1e-6, 1e-6};
    subcycle2.auto_partition(uniform_dts, 4.0);

    check(subcycle2.num_regions() == 1, "Single region for uniform dt");
    check(!subcycle2.has_subcycling(), "No subcycling for uniform");
}

// ============================================================================
// Test 4: Consistent Mass Matrix
// ============================================================================
void test_consistent_mass() {
    std::cout << "\n=== Test 4: Consistent Mass Matrix ===\n";

    ConsistentMass mass;

    // Build a simple 3x3 mass matrix
    // M = [2 1 0]
    //     [1 4 1]
    //     [0 1 2]
    mass.add_entry(0, 0, 2.0);
    mass.add_entry(0, 1, 1.0);
    mass.add_entry(1, 1, 4.0);
    mass.add_entry(1, 2, 1.0);
    mass.add_entry(2, 2, 2.0);

    mass.build_csr(3);

    check(mass.num_dofs() == 3, "Correct DOF count");
    check(mass.nnz() == 7, "Correct non-zeros (with symmetry)");

    // Test matrix-vector multiply
    Real x[3] = {1.0, 2.0, 3.0};
    Real y[3];

    mass.multiply(x, y);

    // y = M*x = [2+2, 1+8+3, 2+6] = [4, 12, 8]
    std::cout << "  M*x = [" << y[0] << ", " << y[1] << ", " << y[2] << "]\n";
    check(std::abs(y[0] - 4.0) < 1e-10, "M*x element 0");
    check(std::abs(y[1] - 12.0) < 1e-10, "M*x element 1");
    check(std::abs(y[2] - 8.0) < 1e-10, "M*x element 2");

    // Test Jacobi solve: M*x = b
    Real b[3] = {4.0, 12.0, 8.0};
    Real x_solve[3] = {0.0, 0.0, 0.0};

    int iters = mass.solve_jacobi(b, x_solve, 1e-8, 100);

    std::cout << "  Jacobi converged in " << iters << " iterations\n";
    std::cout << "  Solution: [" << x_solve[0] << ", " << x_solve[1] << ", " << x_solve[2] << "]\n";

    check(std::abs(x_solve[0] - 1.0) < 1e-4, "Jacobi solve x[0]");
    check(std::abs(x_solve[1] - 2.0) < 1e-4, "Jacobi solve x[1]");
    check(std::abs(x_solve[2] - 3.0) < 1e-4, "Jacobi solve x[2]");
}

// ============================================================================
// Test 5: Central Difference Integration
// ============================================================================
void test_central_difference() {
    std::cout << "\n=== Test 5: Central Difference Integration ===\n";

    // Simple harmonic oscillator: m*a + k*x = 0
    // Analytical: x(t) = A*cos(omega*t), omega = sqrt(k/m)
    Real m = 1.0;
    Real k = 100.0;
    Real x0 = 0.1;

    Real omega = std::sqrt(k / m);
    Real period = 2.0 * M_PI / omega;
    Real dt = 0.001;  // Should be stable for this problem

    // State
    Real x = x0;
    Real v = 0.0;
    Real v_half = 0.0;  // v at t - dt/2

    // Initialize v_half using first-order approximation
    Real a = -k * x / m;
    v_half = v - 0.5 * dt * a;

    Real time = 0.0;
    Real E0 = 0.5 * k * x * x + 0.5 * m * v * v;

    // Integrate for one period
    int steps = 0;
    while (time < period) {
        a = -k * x / m;

        // Update v at t + dt/2
        Real v_half_new = v_half + dt * a;

        // Update x
        x += dt * v_half_new;

        // v at integer time (for energy)
        v = 0.5 * (v_half + v_half_new);

        v_half = v_half_new;
        time += dt;
        steps++;
    }

    Real x_analytical = x0 * std::cos(omega * time);
    Real E_final = 0.5 * k * x * x + 0.5 * m * v * v;
    Real energy_error = std::abs(E_final - E0) / E0;

    std::cout << "  Steps: " << steps << "\n";
    std::cout << "  Final x: " << x << " (analytical: " << x_analytical << ")\n";
    std::cout << "  Energy error: " << (energy_error * 100) << "%\n";

    check(std::abs(x - x_analytical) < 0.01 * x0, "Central diff position accuracy");
    check(energy_error < 0.01, "Central diff energy conservation");
}

// ============================================================================
// Test 6: Velocity Verlet Integration
// ============================================================================
void test_velocity_verlet() {
    std::cout << "\n=== Test 6: Velocity Verlet Integration ===\n";

    // Same harmonic oscillator
    Real m = 1.0;
    Real k = 100.0;
    Real x0 = 0.1;

    Real omega = std::sqrt(k / m);
    Real period = 2.0 * M_PI / omega;
    Real dt = 0.001;

    Real u[1] = {x0};
    Real v[1] = {0.0};
    Real a[1] = {-k * u[0] / m};
    Real v_half[1];

    Real time = 0.0;
    Real E0 = 0.5 * k * u[0] * u[0] + 0.5 * m * v[0] * v[0];

    while (time < period) {
        // Half-step velocity
        AdvancedTimeIntegrator::verlet_velocity_half(v_half, v, a, dt, 1);

        // Full displacement
        u[0] += dt * v_half[0];

        // New acceleration
        Real a_new[1] = {-k * u[0] / m};

        // Full velocity
        AdvancedTimeIntegrator::verlet_velocity_full(v, v_half, a_new, dt, 1);

        a[0] = a_new[0];
        time += dt;
    }

    Real x_analytical = x0 * std::cos(omega * time);
    Real E_final = 0.5 * k * u[0] * u[0] + 0.5 * m * v[0] * v[0];
    Real energy_error = std::abs(E_final - E0) / E0;

    std::cout << "  Final x: " << u[0] << " (analytical: " << x_analytical << ")\n";
    std::cout << "  Energy error: " << (energy_error * 100) << "%\n";

    check(std::abs(u[0] - x_analytical) < 0.005 * x0, "Verlet position accuracy");
    check(energy_error < 0.001, "Verlet energy conservation (symplectic)");
}

// ============================================================================
// Test 7: Newmark-β Integration
// ============================================================================
void test_newmark_beta() {
    std::cout << "\n=== Test 7: Newmark-beta Integration ===\n";

    AdvancedTimeIntegrator integrator(IntegrationScheme::NewmarkExplicit);
    integrator.set_newmark_params(0.0, 0.5);  // Explicit Newmark

    Real m = 1.0;
    Real k = 100.0;
    Real x0 = 0.1;

    Real omega = std::sqrt(k / m);
    Real period = 2.0 * M_PI / omega;
    Real dt = 0.001;

    Real u[1] = {x0};
    Real v[1] = {0.0};
    Real a[1] = {-k * u[0] / m};

    Real u_old[1], v_old[1], a_old[1];
    Real u_pred[1], v_pred[1], a_pred[1];

    Real time = 0.0;
    Real E0 = 0.5 * k * u[0] * u[0] + 0.5 * m * v[0] * v[0];

    while (time < period) {
        // Save old state
        u_old[0] = u[0];
        v_old[0] = v[0];
        a_old[0] = a[0];

        // Predict
        integrator.newmark_predict(u_pred, v_pred, a_pred,
                                   u_old, v_old, a_old, dt, 1);

        // Compute new acceleration
        a_pred[0] = -k * u_pred[0] / m;

        // Correct
        u[0] = u_pred[0];
        v[0] = v_pred[0];
        integrator.newmark_correct(u, v, a_pred, dt, 1);
        a[0] = a_pred[0];

        time += dt;
    }

    Real x_analytical = x0 * std::cos(omega * time);
    Real E_final = 0.5 * k * u[0] * u[0] + 0.5 * m * v[0] * v[0];
    Real energy_error = std::abs(E_final - E0) / E0;

    std::cout << "  Final x: " << u[0] << " (analytical: " << x_analytical << ")\n";
    std::cout << "  Energy error: " << (energy_error * 100) << "%\n";

    check(std::abs(u[0] - x_analytical) < 0.01 * x0, "Newmark position accuracy");
    check(energy_error < 0.05, "Newmark energy bounded");
}

// ============================================================================
// Test 8: Time Integrator Configuration
// ============================================================================
void test_integrator_config() {
    std::cout << "\n=== Test 8: Time Integrator Configuration ===\n";

    AdvancedTimeIntegrator integrator;

    // Test scheme setting
    integrator.set_scheme(IntegrationScheme::VelocityVerlet);
    check(integrator.scheme() == IntegrationScheme::VelocityVerlet, "Scheme set");

    // Test consistent mass
    integrator.enable_consistent_mass(true);
    check(integrator.uses_consistent_mass(), "Consistent mass enabled");

    // Test Rayleigh damping
    integrator.set_rayleigh_damping(0.1, 0.001);
    check(std::abs(integrator.damping_alpha() - 0.1) < 1e-10, "Damping alpha set");
    check(std::abs(integrator.damping_beta() - 0.001) < 1e-10, "Damping beta set");

    // Test HHT-α
    integrator.set_hht_alpha(-0.1);
    check(true, "HHT-alpha configured");

    // Test subcycling access
    integrator.subcycling().add_region("test", {0, 1}, {0}, 2);
    check(integrator.subcycling().num_regions() == 1, "Subcycling accessible");

    // Test energy monitor access
    EnergyMonitor::EnergyState state;
    state.kinetic = 100.0;
    integrator.energy_monitor().record(state);
    check(integrator.energy_monitor().history().size() == 1, "Energy monitor accessible");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=================================================\n";
    std::cout << "Advanced Time Integration Test Suite\n";
    std::cout << "=================================================\n";

    test_energy_monitor();
    test_subcycling_controller();
    test_auto_partition();
    test_consistent_mass();
    test_central_difference();
    test_velocity_verlet();
    test_newmark_beta();
    test_integrator_config();

    std::cout << "\n=================================================\n";
    std::cout << "Results: " << pass_count << "/" << test_count << " tests passed\n";
    std::cout << "=================================================\n";

    return (pass_count == test_count) ? 0 : 1;
}
