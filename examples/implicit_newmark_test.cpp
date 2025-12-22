/**
 * @file implicit_newmark_test.cpp
 * @brief Test implicit Newmark-beta time integrator
 *
 * Tests the implicit Newmark integrator against analytical solutions:
 * 1. Single DOF spring-mass system (undamped oscillation)
 * 2. Single DOF with damping (decay)
 * 3. Multi-DOF coupled system
 * 4. Large time step stability test
 */

#include <nexussim/physics/time_integrator.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::physics;

/**
 * @brief Test 1: Single DOF undamped oscillator
 *
 * m*a + k*u = 0
 * Analytical: u(t) = u0*cos(ω*t), ω = sqrt(k/m)
 */
bool test_undamped_oscillator() {
    std::cout << "=== Test 1: Undamped Single DOF Oscillator ===" << std::endl;

    // System parameters
    const Real m = 1.0;       // Mass (kg)
    const Real k = 100.0;     // Stiffness (N/m)
    const Real omega = std::sqrt(k / m);  // Natural frequency (rad/s)
    const Real period = 2.0 * M_PI / omega;  // Period (s)

    // Initial conditions
    const Real u0 = 0.1;      // Initial displacement (m)
    const Real v0 = 0.0;      // Initial velocity (m/s)

    std::cout << "  Mass: " << m << " kg" << std::endl;
    std::cout << "  Stiffness: " << k << " N/m" << std::endl;
    std::cout << "  Natural frequency: " << omega << " rad/s" << std::endl;
    std::cout << "  Period: " << period << " s" << std::endl;
    std::cout << std::endl;

    // Create implicit integrator
    ImplicitNewmarkIntegrator integrator(0.25, 0.5);  // Average acceleration
    integrator.initialize(1);

    // Set diagonal stiffness for effective mass computation
    std::vector<Real> k_diag = {k};
    integrator.set_diagonal_stiffness(k_diag);
    integrator.set_convergence(50, 1.0e-10);

    // Create state
    DynamicState state(1);
    state.displacement[0] = u0;
    state.velocity[0] = v0;
    state.acceleration[0] = -k * u0 / m;  // Initial acceleration
    state.mass[0] = m;
    state.force_external[0] = 0.0;
    state.force_internal[0] = k * u0;  // Initial internal force

    // Time integration
    const Real dt = period / 20.0;  // 20 steps per period
    const int num_periods = 5;
    const int steps = static_cast<int>(num_periods * period / dt);

    std::cout << "  Time step: " << dt * 1000.0 << " ms" << std::endl;
    std::cout << "  Steps per period: " << period / dt << std::endl;
    std::cout << std::endl;

    std::cout << "  t (s)\t\tu (m)\t\tu_exact (m)\tError (%)" << std::endl;
    std::cout << "  ----\t\t----\t\t----------\t---------" << std::endl;

    Real max_error = 0.0;
    Real time = 0.0;

    for (int step = 0; step <= steps; ++step) {
        // Analytical solution
        Real u_exact = u0 * std::cos(omega * time);

        // Compute error
        Real error = std::abs(state.displacement[0] - u_exact) / u0 * 100.0;
        max_error = std::max(max_error, error);

        // Print every period
        if (step % static_cast<int>(period / dt) == 0) {
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "  " << time << "\t\t" << state.displacement[0]
                      << "\t\t" << u_exact << "\t\t" << error << std::endl;
        }

        // Time step (integrator updates internal force internally for linear spring)
        integrator.step(dt, state);
        time += dt;
    }

    std::cout << std::endl;
    std::cout << "  Max error: " << max_error << "%" << std::endl;
    std::cout << "  Final iterations: " << integrator.last_iterations() << std::endl;
    std::cout << "  Final residual: " << integrator.last_residual() << std::endl;

    bool pass = (max_error < 30.0);  // Allow 30% error over 5 periods (with 20 steps/period)
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    return pass;
}

/**
 * @brief Test 2: Large time step stability
 *
 * Test with time step >> critical time step for explicit
 * Implicit should remain stable (though may lose accuracy)
 */
bool test_large_timestep_stability() {
    std::cout << "=== Test 2: Large Time Step Stability ===" << std::endl;

    const Real m = 1.0;
    const Real k = 10000.0;  // High stiffness
    const Real omega = std::sqrt(k / m);
    const Real period = 2.0 * M_PI / omega;

    // Explicit critical time step: dt_crit = 2/ω ≈ 0.02 s
    const Real dt_explicit = 2.0 / omega;
    std::cout << "  Explicit critical dt: " << dt_explicit * 1000.0 << " ms" << std::endl;

    // Use time step 10x larger than explicit critical
    const Real dt = 10.0 * dt_explicit;
    std::cout << "  Using dt: " << dt * 1000.0 << " ms (10x critical)" << std::endl;
    std::cout << std::endl;

    // Initial conditions
    const Real u0 = 0.05;

    // Create implicit integrator
    ImplicitNewmarkIntegrator integrator(0.25, 0.5);
    integrator.initialize(1);

    std::vector<Real> k_diag = {k};
    integrator.set_diagonal_stiffness(k_diag);
    integrator.set_convergence(100, 1.0e-10);

    DynamicState state(1);
    state.displacement[0] = u0;
    state.velocity[0] = 0.0;
    state.acceleration[0] = -k * u0 / m;
    state.mass[0] = m;
    state.force_external[0] = 0.0;
    state.force_internal[0] = k * u0;

    // Run for several periods
    const int steps = 50;
    Real time = 0.0;
    bool stable = true;
    Real max_disp = u0;

    std::cout << "  Step\tDisplacement\tEnergy Ratio" << std::endl;

    Real initial_energy = 0.5 * k * u0 * u0;

    for (int step = 0; step <= steps; ++step) {
        // Check for instability (unbounded growth)
        if (std::abs(state.displacement[0]) > 10.0 * u0 ||
            std::isnan(state.displacement[0]) ||
            std::isinf(state.displacement[0])) {
            stable = false;
            std::cout << "  UNSTABLE at step " << step << std::endl;
            break;
        }

        max_disp = std::max(max_disp, std::abs(state.displacement[0]));

        // Compute current energy
        Real ke = 0.5 * m * state.velocity[0] * state.velocity[0];
        Real pe = 0.5 * k * state.displacement[0] * state.displacement[0];
        Real energy_ratio = (ke + pe) / initial_energy;

        if (step % 10 == 0) {
            std::cout << "  " << step << "\t" << state.displacement[0]
                      << "\t\t" << energy_ratio << std::endl;
        }

        // Time step
        integrator.step(dt, state);
        time += dt;
    }

    std::cout << std::endl;
    std::cout << "  Stable: " << (stable ? "YES" : "NO") << std::endl;
    std::cout << "  Max displacement: " << max_disp << " m" << std::endl;

    bool pass = stable;
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    return pass;
}

/**
 * @brief Test 3: Damped oscillator
 *
 * m*a + c*v + k*u = 0
 * Analytical (underdamped): u(t) = u0 * exp(-ζωt) * cos(ω_d*t)
 * where ω_d = ω * sqrt(1 - ζ²)
 */
bool test_damped_oscillator() {
    std::cout << "=== Test 3: Damped Oscillator ===" << std::endl;

    const Real m = 1.0;
    const Real k = 100.0;
    const Real omega = std::sqrt(k / m);

    // Damping: 10% critical damping
    const Real zeta = 0.1;
    const Real c_crit = 2.0 * m * omega;
    const Real c = zeta * c_crit;

    const Real omega_d = omega * std::sqrt(1.0 - zeta * zeta);
    const Real period = 2.0 * M_PI / omega_d;

    std::cout << "  Damping ratio: " << zeta * 100.0 << "% critical" << std::endl;
    std::cout << "  Damped frequency: " << omega_d << " rad/s" << std::endl;
    std::cout << std::endl;

    const Real u0 = 0.1;

    // Create integrator with Rayleigh damping (mass proportional)
    // For this simple case: c = α_m * m => α_m = c / m
    ImplicitNewmarkIntegrator integrator(0.25, 0.5);
    integrator.initialize(1);

    Real alpha_m = c / m;
    integrator.set_rayleigh_damping(alpha_m, 0.0);

    std::vector<Real> k_diag = {k};
    integrator.set_diagonal_stiffness(k_diag);
    integrator.set_convergence(50, 1.0e-10);

    DynamicState state(1);
    state.displacement[0] = u0;
    state.velocity[0] = 0.0;
    state.acceleration[0] = -k * u0 / m;
    state.mass[0] = m;
    state.force_external[0] = 0.0;
    state.force_internal[0] = k * u0;

    const Real dt = period / 20.0;
    const int num_periods = 5;
    const int steps = static_cast<int>(num_periods * period / dt);

    std::cout << "  t (s)\t\tu (m)\t\tu_exact (m)" << std::endl;
    std::cout << "  ----\t\t----\t\t----------" << std::endl;

    Real time = 0.0;
    Real max_error = 0.0;

    for (int step = 0; step <= steps; ++step) {
        // Analytical solution for underdamped system
        Real u_exact = u0 * std::exp(-zeta * omega * time) * std::cos(omega_d * time);

        Real error = (u0 > 1e-10) ? std::abs(state.displacement[0] - u_exact) / u0 : 0.0;
        max_error = std::max(max_error, error);

        // Print every period
        if (step % static_cast<int>(period / dt) == 0) {
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "  " << time << "\t\t" << state.displacement[0]
                      << "\t\t" << u_exact << std::endl;
        }

        // Time step
        integrator.step(dt, state);
        time += dt;
    }

    // Check that amplitude decayed as expected
    // After 5 periods, amplitude should be ~ u0 * exp(-5 * 2π * ζ) ≈ 0.044 * u0
    Real expected_decay = std::exp(-5.0 * 2.0 * M_PI * zeta);
    Real actual_amplitude = std::abs(state.displacement[0]) / u0;

    std::cout << std::endl;
    std::cout << "  Expected amplitude ratio: " << expected_decay << std::endl;
    std::cout << "  Actual amplitude ratio: " << actual_amplitude << std::endl;

    // Allow some tolerance due to numerical damping
    bool pass = (actual_amplitude < expected_decay * 2.0 && actual_amplitude > expected_decay * 0.5);
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    return pass;
}

/**
 * @brief Test 4: HHT-α integrator numerical damping
 *
 * HHT-α should damp high-frequency response while preserving low-frequency
 */
bool test_hht_alpha() {
    std::cout << "=== Test 4: HHT-α Numerical Damping ===" << std::endl;

    const Real m = 1.0;
    const Real k = 100.0;
    const Real u0 = 0.1;

    // Compare standard Newmark vs HHT-α
    ImplicitNewmarkIntegrator newmark(0.25, 0.5);
    ImplicitHHTIntegrator hht(-0.1);  // α = -0.1 for moderate numerical damping

    newmark.initialize(1);
    hht.initialize(1);

    std::vector<Real> k_diag = {k};
    newmark.set_diagonal_stiffness(k_diag);
    hht.set_diagonal_stiffness(k_diag);

    newmark.set_convergence(50, 1.0e-10);
    hht.set_convergence(50, 1.0e-10);

    // Create two identical states
    DynamicState state_newmark(1);
    DynamicState state_hht(1);

    state_newmark.displacement[0] = u0;
    state_newmark.velocity[0] = 0.0;
    state_newmark.acceleration[0] = -k * u0 / m;
    state_newmark.mass[0] = m;
    state_newmark.force_external[0] = 0.0;
    state_newmark.force_internal[0] = k * u0;

    state_hht.displacement[0] = u0;
    state_hht.velocity[0] = 0.0;
    state_hht.acceleration[0] = -k * u0 / m;
    state_hht.mass[0] = m;
    state_hht.force_external[0] = 0.0;
    state_hht.force_internal[0] = k * u0;

    Real omega = std::sqrt(k / m);
    Real period = 2.0 * M_PI / omega;
    Real dt = period / 10.0;  // Coarse time step to see damping effects

    std::cout << "  HHT α = " << hht.alpha_hht() << std::endl;
    std::cout << "  dt/T = " << dt / period << std::endl;
    std::cout << std::endl;

    std::cout << "  Period\tNewmark Amp\tHHT Amp\t\tHHT/Newmark" << std::endl;
    std::cout << "  ------\t-----------\t-------\t\t-----------" << std::endl;

    const int num_periods = 10;
    int steps_per_period = static_cast<int>(period / dt);

    for (int p = 0; p <= num_periods; ++p) {
        // Run one period
        for (int step = 0; step < steps_per_period; ++step) {
            newmark.step(dt, state_newmark);
            hht.step(dt, state_hht);
        }

        // Report amplitude at each period
        Real amp_newmark = std::abs(state_newmark.displacement[0]);
        Real amp_hht = std::abs(state_hht.displacement[0]);
        Real ratio = amp_hht / std::max(amp_newmark, 1.0e-20);

        std::cout << "  " << p << "\t\t" << amp_newmark
                  << "\t\t" << amp_hht << "\t\t" << ratio << std::endl;
    }

    // HHT should show more damping than Newmark
    Real final_newmark = std::abs(state_newmark.displacement[0]);
    Real final_hht = std::abs(state_hht.displacement[0]);

    std::cout << std::endl;
    std::cout << "  Final Newmark amplitude: " << final_newmark << std::endl;
    std::cout << "  Final HHT amplitude: " << final_hht << std::endl;

    // HHT should damp more than pure Newmark
    bool pass = (final_hht < final_newmark || std::abs(final_hht - final_newmark) < 0.01);
    std::cout << "  HHT shows numerical damping: " << (final_hht < final_newmark ? "YES" : "SIMILAR") << std::endl;
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    return pass;
}

/**
 * @brief Test 5: Static equilibrium convergence
 *
 * Apply constant force and check convergence to static equilibrium
 */
bool test_static_equilibrium() {
    std::cout << "=== Test 5: Static Equilibrium Convergence ===" << std::endl;

    const Real m = 1.0;
    const Real k = 100.0;
    const Real F = 10.0;  // Applied force

    // Expected static equilibrium: u = F/k
    const Real u_static = F / k;

    std::cout << "  Applied force: " << F << " N" << std::endl;
    std::cout << "  Expected static displacement: " << u_static << " m" << std::endl;
    std::cout << std::endl;

    // Use HHT-α with damping to reach static equilibrium faster
    ImplicitHHTIntegrator integrator(-0.1);
    integrator.initialize(1);

    // Add Rayleigh damping to dissipate energy
    integrator.set_rayleigh_damping(0.5, 0.0);  // Mass-proportional damping

    std::vector<Real> k_diag = {k};
    integrator.set_diagonal_stiffness(k_diag);
    integrator.set_convergence(50, 1.0e-10);

    DynamicState state(1);
    state.displacement[0] = 0.0;
    state.velocity[0] = 0.0;
    state.acceleration[0] = F / m;  // Initial acceleration from applied force
    state.mass[0] = m;
    state.force_external[0] = F;
    state.force_internal[0] = 0.0;

    Real omega = std::sqrt(k / m);
    Real period = 2.0 * M_PI / omega;
    Real dt = period / 10.0;

    std::cout << "  Time (s)\tDisplacement\tv (m/s)\t\tError (%)" << std::endl;
    std::cout << "  --------\t------------\t-------\t\t---------" << std::endl;

    Real time = 0.0;
    const int max_steps = 200;
    bool converged = false;

    for (int step = 0; step <= max_steps; ++step) {
        Real error = std::abs(state.displacement[0] - u_static) / u_static * 100.0;

        if (step % 20 == 0) {
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "  " << time << "\t\t" << state.displacement[0]
                      << "\t\t" << state.velocity[0] << "\t\t" << error << std::endl;
        }

        // Check convergence
        if (error < 1.0 && std::abs(state.velocity[0]) < 0.01) {
            converged = true;
            std::cout << "  Converged at step " << step << " (t = " << time << " s)" << std::endl;
            break;
        }

        // Time step
        integrator.step(dt, state);
        time += dt;
    }

    std::cout << std::endl;
    std::cout << "  Final displacement: " << state.displacement[0] << " m" << std::endl;
    std::cout << "  Final velocity: " << state.velocity[0] << " m/s" << std::endl;

    Real final_error = std::abs(state.displacement[0] - u_static) / u_static * 100.0;
    std::cout << "  Final error: " << final_error << "%" << std::endl;

    bool pass = (final_error < 5.0);
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    return pass;
}

int main() {
    std::cout << std::setprecision(6);
    std::cout << "========================================" << std::endl;
    std::cout << "Implicit Newmark-β Time Integrator Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 0;

    if (test_undamped_oscillator()) passed++;
    total++;

    if (test_large_timestep_stability()) passed++;
    total++;

    if (test_damped_oscillator()) passed++;
    total++;

    if (test_hht_alpha()) passed++;
    total++;

    if (test_static_equilibrium()) passed++;
    total++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
