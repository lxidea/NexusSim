/**
 * @file ale_wave27_test.cpp
 * @brief Wave 27: Advanced ALE Extensions Test Suite
 *
 * Tests 5 sub-modules (10 tests each, 50 total):
 *  1. KEpsilonTurbulence   — k-eps model, viscosity, production, decay, wall functions
 *  2. EulerianBimatTracker — VOF, PLIC, advection, mixing, interface detection
 *  3. PorousMediaFlow      — Darcy velocity, effective stress, pressure diffusion
 *  4. ALEErosionHandler    — Mass/momentum/energy conservation, void tracking
 *  5. ALEAdaptiveRemesh    — Error indicators, refinement marking, solution transfer
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <numeric>
#include <nexussim/fem/ale_wave27.hpp>

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

using namespace nxs::fem;
using Real = nxs::Real;

// ============================================================================
// Test Group 1: KEpsilonTurbulence (10 tests)
// ============================================================================

void test_kepsilon() {
    std::cout << "=== KEpsilonTurbulence Tests ===\n";
    KEpsilonModel ke;

    // Test 1: Turbulent viscosity formula
    // mu_t = C_mu * rho * k^2 / eps = 0.09 * 1.2 * 4.0^2 / 0.5
    {
        Real rho = 1.2, k = 4.0, eps = 0.5;
        Real mu_t = ke.compute_turbulent_viscosity(rho, k, eps);
        Real expected = 0.09 * 1.2 * 16.0 / 0.5; // = 3.456
        CHECK_NEAR(mu_t, expected, 1.0e-10, "k-eps: turbulent viscosity formula");
    }

    // Test 2: Production term
    // P_k = mu_t * |S|^2 = 3.456 * 100.0 = 345.6
    {
        Real mu_t = 3.456, S_mag = 10.0;
        Real P_k = ke.compute_production(mu_t, S_mag);
        CHECK_NEAR(P_k, 345.6, 1.0e-10, "k-eps: production term");
    }

    // Test 3: Decay of k/eps with no production (analytical solution)
    {
        Real k0 = 1.0, eps0 = 0.1;
        Real t = 2.0;
        Real k_analytical, eps_analytical;
        ke.decay_analytical(k0, eps0, t, k_analytical, eps_analytical);

        // Numerical: many small steps
        Real k_num = k0, eps_num = eps0;
        int nsteps = 100000;
        Real dt = t / nsteps;
        for (int s = 0; s < nsteps; ++s) {
            Real k_new, eps_new;
            ke.advance(k_num, eps_num, 1.0, 0.0, 0.0, 0.0, dt, k_new, eps_new);
            k_num = k_new;
            eps_num = eps_new;
        }
        CHECK_NEAR(k_num, k_analytical, 1.0e-3, "k-eps: decay k matches analytical");
        CHECK_NEAR(eps_num, eps_analytical, 1.0e-4, "k-eps: decay eps matches analytical");
    }

    // Test 4: Wall function — log-law region (y+ = 100)
    {
        Real yp = 100.0;
        Real up = ke.wall_function(yp);
        Real expected = (1.0/0.41) * std::log(100.0) + 5.2;
        CHECK_NEAR(up, expected, 1.0e-10, "k-eps: wall function log-law");
    }

    // Test 5: Wall function — viscous sublayer (y+ = 5)
    {
        Real yp = 5.0;
        Real up = ke.wall_function(yp);
        CHECK_NEAR(up, 5.0, 1.0e-14, "k-eps: wall function viscous sublayer");
    }

    // Test 6: Positive k/eps enforcement
    {
        Real k_new, eps_new;
        ke.advance(1.0e-15, 1.0e-15, 1.0, 0.0, 0.0, 0.0, 100.0, k_new, eps_new);
        CHECK(k_new > 0.0, "k-eps: k remains positive after large dt");
        CHECK(eps_new > 0.0, "k-eps: eps remains positive after large dt");
    }

    // Test 7: Equilibrium condition (P_k = rho * eps)
    {
        Real rho = 1.0, eps = 0.5, S_mag = 10.0;
        Real k_eq = ke.equilibrium_k(eps, S_mag);
        Real mu_t = ke.compute_turbulent_viscosity(rho, k_eq, eps);
        Real P_k = ke.compute_production(mu_t, S_mag);
        Real rho_eps = rho * eps;
        CHECK_NEAR(P_k, rho_eps, 1.0e-8, "k-eps: equilibrium P_k = rho*eps");
    }

    // Test 8: Standard constants
    {
        CHECK_NEAR(KEpsilonModel::C_mu, 0.09, 1.0e-14, "k-eps: C_mu = 0.09");
        CHECK_NEAR(KEpsilonModel::C_e1, 1.44, 1.0e-14, "k-eps: C_e1 = 1.44");
        CHECK_NEAR(KEpsilonModel::C_e2, 1.92, 1.0e-14, "k-eps: C_e2 = 1.92");
        CHECK_NEAR(KEpsilonModel::sigma_k, 1.0, 1.0e-14, "k-eps: sigma_k = 1.0");
        CHECK_NEAR(KEpsilonModel::sigma_eps, 1.3, 1.0e-14, "k-eps: sigma_eps = 1.3");
    }

    // Test 9: k and eps remain bounded and positive after long integration
    {
        Real rho = 1.0, k = 0.1, eps = 0.01;
        Real S_mag = 5.0;
        Real dt = 1.0e-4;
        int nsteps = 10000;

        for (int s = 0; s < nsteps; ++s) {
            Real mu_t = ke.compute_turbulent_viscosity(rho, k, eps);
            Real P_k = ke.compute_production(mu_t, S_mag);
            Real k_new, eps_new;
            ke.advance(k, eps, rho, P_k, 0.0, mu_t, dt, k_new, eps_new);
            k = k_new;
            eps = eps_new;
        }
        // k and eps should remain positive and finite
        CHECK(k > 0.0 && k < 1.0e10, "k-eps: k remains bounded and positive");
        CHECK(eps > 0.0 && eps < 1.0e10, "k-eps: eps remains bounded and positive");
    }

    // Test 10: Coupling with ALE state — turbulent viscosity modifies effective viscosity
    {
        Real rho = 1.225, k = 3.0, eps = 0.3;
        Real mu_mol = 1.8e-5;
        Real mu_t = ke.compute_turbulent_viscosity(rho, k, eps);
        Real mu_eff = mu_mol + mu_t;
        // mu_t = 0.09 * 1.225 * 9.0 / 0.3 = 3.3075
        CHECK_NEAR(mu_t, 0.09 * 1.225 * 9.0 / 0.3, 1.0e-10,
                   "k-eps: coupling mu_t correct");
        CHECK(mu_eff > mu_mol, "k-eps: effective viscosity > molecular");
        CHECK(mu_t / mu_eff > 0.99, "k-eps: turbulent viscosity dominates");
    }
}

// ============================================================================
// Test Group 2: EulerianBimatTracker (10 tests)
// ============================================================================

void test_bimat_tracker() {
    std::cout << "=== EulerianBimatTracker Tests ===\n";
    BimatTracker tracker;

    // Test 1: Pure cells stay pure under advection
    {
        Real alpha_pure1 = tracker.advect_volume_fraction(1.0, 0.0, 0.0, 1.0, 0.01);
        Real alpha_pure0 = tracker.advect_volume_fraction(0.0, 0.0, 0.0, 1.0, 0.01);
        CHECK_NEAR(alpha_pure1, 1.0, 1.0e-14, "bimat: pure cell alpha=1 stays 1");
        CHECK_NEAR(alpha_pure0, 0.0, 1.0e-14, "bimat: pure cell alpha=0 stays 0");
    }

    // Test 2: Interface detection
    {
        CHECK(tracker.is_interface_cell(0.5), "bimat: alpha=0.5 is interface");
        CHECK(!tracker.is_interface_cell(0.0), "bimat: alpha=0 is not interface");
        CHECK(!tracker.is_interface_cell(1.0), "bimat: alpha=1 is not interface");
        CHECK(!tracker.is_interface_cell(0.005), "bimat: alpha=0.005 below tol");
        CHECK(tracker.is_interface_cell(0.5, 0.01), "bimat: alpha=0.5 with tol=0.01");
    }

    // Test 3: Normal computation — gradient in x-direction
    {
        // Neighbors: [x-, x+, y-, y+, z-, z+]
        // alpha decreases in x: left=1.0, right=0.0
        Real neighbors[6] = {1.0, 0.0, 0.5, 0.5, 0.5, 0.5};
        Real normal[3];
        tracker.compute_interface_normal(neighbors, normal);
        // grad_alpha = (0 - 1) = -1 in x, 0 in y,z
        // normal = -grad/|grad| = +1 in x
        CHECK_NEAR(normal[0], 1.0, 1.0e-10, "bimat: normal x-component for x-gradient");
        CHECK_NEAR(normal[1], 0.0, 1.0e-10, "bimat: normal y-component for x-gradient");
        CHECK_NEAR(normal[2], 0.0, 1.0e-10, "bimat: normal z-component for x-gradient");
    }

    // Test 4: Advection conservation — total alpha*volume conserved
    {
        int n = 20;
        std::vector<Real> alpha(n, 0.0);
        // Initial step function: left half = 1, right half = 0
        for (int i = 0; i < n/2; ++i) alpha[i] = 1.0;
        Real dx = 0.1, velocity = 1.0, dt = 0.01;
        Real sum_before = 0.0;
        for (int i = 0; i < n; ++i) sum_before += alpha[i] * dx;

        std::vector<Real> alpha_new;
        tracker.advect_1d(alpha, velocity, dx, dt, n, alpha_new);

        Real sum_after = 0.0;
        for (int i = 0; i < n; ++i) sum_after += alpha_new[i] * dx;
        CHECK_NEAR(sum_after, sum_before, 0.02, "bimat: advection conserves total fraction");
    }

    // Test 5: PLIC volume match (axis-aligned normal for best accuracy)
    {
        Real normal[3] = {1.0, 0.0, 0.0};
        Real alpha = 0.3;
        Real cell_size = 1.0;
        Real d = tracker.compute_plic_offset(normal, alpha, cell_size);
        Real vol = tracker.plic_truncated_volume(normal, d, cell_size);
        CHECK_NEAR(vol, alpha * cell_size * cell_size * cell_size, 0.05,
                   "bimat: PLIC volume matches target");
    }

    // Test 6: Mixing rule
    {
        Real alpha = 0.6, rho1 = 1000.0, rho2 = 1.2;
        Real mixed = tracker.mix_properties(alpha, rho1, rho2);
        Real expected = 0.6 * 1000.0 + 0.4 * 1.2; // = 600.48
        CHECK_NEAR(mixed, expected, 1.0e-10, "bimat: mixing rule rho");
    }

    // Test 7: Uniform advection — block moves intact
    {
        int n = 40;
        std::vector<Real> alpha(n, 0.0);
        for (int i = 10; i < 15; ++i) alpha[i] = 1.0;
        Real dx = 0.1, velocity = 1.0, dt = 0.005;

        std::vector<Real> alpha_new;
        // Advect several steps
        for (int step = 0; step < 10; ++step) {
            tracker.advect_1d(alpha, velocity, dx, dt, n, alpha_new);
            alpha = alpha_new;
        }
        // The block should have moved right
        Real center_before = 12.0; // original center index
        Real sum_i = 0.0, sum_a = 0.0;
        for (int i = 0; i < n; ++i) {
            sum_i += i * alpha[i];
            sum_a += alpha[i];
        }
        Real center_after = (sum_a > 0.0) ? sum_i / sum_a : 0.0;
        CHECK(center_after > center_before, "bimat: block advects to the right");
    }

    // Test 8: Interface sharpness — step function stays relatively sharp
    {
        int n = 50;
        std::vector<Real> alpha(n, 0.0);
        for (int i = 0; i < 25; ++i) alpha[i] = 1.0;

        int interface_cells_before = 0;
        for (int i = 0; i < n; ++i) {
            if (tracker.is_interface_cell(alpha[i], 0.01)) interface_cells_before++;
        }

        std::vector<Real> alpha_new;
        Real dx = 0.1, velocity = 0.5, dt = 0.01;
        for (int step = 0; step < 5; ++step) {
            tracker.advect_1d(alpha, velocity, dx, dt, n, alpha_new);
            alpha = alpha_new;
        }

        int interface_cells_after = 0;
        for (int i = 0; i < n; ++i) {
            if (tracker.is_interface_cell(alpha[i], 0.01)) interface_cells_after++;
        }
        // Interface should not smear to more than a few cells
        CHECK(interface_cells_after <= 5, "bimat: interface stays relatively sharp");
    }

    // Test 9: Bounds enforcement [0, 1]
    {
        // Large outgoing flux should clamp to 0
        Real alpha = tracker.advect_volume_fraction(0.1, 0.0, 100.0, 1.0, 0.01);
        CHECK(alpha >= 0.0, "bimat: alpha clamped >= 0");
        // Large incoming flux should clamp to 1
        Real alpha2 = tracker.advect_volume_fraction(0.9, 100.0, 0.0, 1.0, 0.01);
        CHECK(alpha2 <= 1.0, "bimat: alpha clamped <= 1");
    }

    // Test 10: Gradient accuracy — diagonal interface
    {
        // Alpha decreasing equally in x and y
        Real neighbors[6] = {0.8, 0.2, 0.8, 0.2, 0.5, 0.5};
        Real normal[3];
        tracker.compute_interface_normal(neighbors, normal);
        // grad_x = 0.2 - 0.8 = -0.6, grad_y = 0.2 - 0.8 = -0.6, grad_z = 0
        // normal = -grad/|grad| = (0.6, 0.6, 0)/|..| = (1/sqrt(2), 1/sqrt(2), 0)
        Real expected_n = 1.0 / std::sqrt(2.0);
        CHECK_NEAR(normal[0], expected_n, 1.0e-10, "bimat: diagonal normal x");
        CHECK_NEAR(normal[1], expected_n, 1.0e-10, "bimat: diagonal normal y");
        CHECK_NEAR(normal[2], 0.0, 1.0e-10, "bimat: diagonal normal z");
    }
}

// ============================================================================
// Test Group 3: PorousMediaFlow (10 tests)
// ============================================================================

void test_porous_media() {
    std::cout << "=== PorousMediaFlow Tests ===\n";
    PorousMediaSolver solver;

    // Test 1: Darcy velocity magnitude
    {
        Real kappa = 1.0e-12; // permeability [m^2]
        Real mu = 1.0e-3;     // viscosity [Pa.s]
        Real grad_p[3] = {1000.0, 0.0, 0.0}; // 1 kPa/m
        Real rho_f = 1000.0;
        Real g[3] = {0.0, 0.0, 0.0};
        Real v[3];
        solver.darcy_velocity(kappa, mu, grad_p, rho_f, g, v);
        // v = -(1e-12/1e-3)*1000 = -1e-6 m/s
        CHECK_NEAR(v[0], -1.0e-6, 1.0e-12, "porous: Darcy velocity x");
        CHECK_NEAR(v[1], 0.0, 1.0e-14, "porous: Darcy velocity y = 0");
        CHECK_NEAR(v[2], 0.0, 1.0e-14, "porous: Darcy velocity z = 0");
    }

    // Test 2: No flow at zero pressure gradient (no gravity)
    {
        Real grad_p[3] = {0.0, 0.0, 0.0};
        Real g[3] = {0.0, 0.0, 0.0};
        Real v[3];
        solver.darcy_velocity(1.0e-10, 1.0e-3, grad_p, 1000.0, g, v);
        CHECK_NEAR(v[0], 0.0, 1.0e-20, "porous: no flow zero gradient x");
        CHECK_NEAR(v[1], 0.0, 1.0e-20, "porous: no flow zero gradient y");
        CHECK_NEAR(v[2], 0.0, 1.0e-20, "porous: no flow zero gradient z");
    }

    // Test 3: Effective stress — Terzaghi (biot_coeff = 1)
    {
        Real sigma[6] = {-100.0, -200.0, -300.0, 10.0, 20.0, 30.0};
        Real p = 50.0;
        Real eff[6];
        solver.effective_stress(sigma, p, 1.0, eff);
        // sigma' = sigma + 1.0 * p * delta
        CHECK_NEAR(eff[0], -50.0, 1.0e-10, "porous: Terzaghi eff stress s11");
        CHECK_NEAR(eff[1], -150.0, 1.0e-10, "porous: Terzaghi eff stress s22");
        CHECK_NEAR(eff[2], -250.0, 1.0e-10, "porous: Terzaghi eff stress s33");
        CHECK_NEAR(eff[3], 10.0, 1.0e-10, "porous: Terzaghi shear unchanged s12");
    }

    // Test 4: Biot coefficient effect
    {
        Real sigma[6] = {-100.0, -100.0, -100.0, 0.0, 0.0, 0.0};
        Real p = 50.0;
        Real eff_07[6], eff_10[6];
        solver.effective_stress(sigma, p, 0.7, eff_07);
        solver.effective_stress(sigma, p, 1.0, eff_10);
        // alpha_B = 0.7: eff = -100 + 35 = -65
        // alpha_B = 1.0: eff = -100 + 50 = -50
        CHECK_NEAR(eff_07[0], -65.0, 1.0e-10, "porous: Biot 0.7 effect");
        CHECK_NEAR(eff_10[0], -50.0, 1.0e-10, "porous: Biot 1.0 effect");
        CHECK(eff_10[0] > eff_07[0], "porous: higher Biot gives higher eff stress");
    }

    // Test 5: Pressure diffusion — analytical 1D
    // Initial: p = sin(pi*x/L), decays as exp(-D*pi^2/L^2 * t)
    {
        int n = 100;
        Real L = 1.0, dx = L / n;
        Real kappa = 1.0e-6, mu = 1.0e-3, S = 1.0;
        Real D = kappa / (mu * S); // diffusivity

        std::vector<Real> p(n);
        for (int i = 0; i < n; ++i) {
            Real x = (i + 0.5) * dx;
            p[i] = std::sin(M_PI * x / L);
        }

        Real dt = 0.1 * dx * dx * S * mu / kappa; // stable
        Real total_time = 0.0;
        int nsteps = 500;
        std::vector<Real> p_new;
        for (int s = 0; s < nsteps; ++s) {
            solver.pressure_diffusion_step(p, kappa, mu, S, dx, dt, n, p_new);
            p = p_new;
            total_time += dt;
        }

        // Analytical: p(x,t) = sin(pi*x/L) * exp(-D*(pi/L)^2 * t)
        Real decay = std::exp(-D * (M_PI/L) * (M_PI/L) * total_time);
        Real x_mid = 0.5 * L;
        int i_mid = n / 2;
        Real analytical = std::sin(M_PI * x_mid / L) * decay;
        CHECK_NEAR(p[i_mid], analytical, 0.05, "porous: pressure diffusion analytical");
    }

    // Test 6: Conservation — zero flux BC should conserve total pressure integral
    {
        int n = 50;
        Real dx = 0.1;
        std::vector<Real> p(n, 100.0); // uniform
        Real sum_before = 0.0;
        for (int i = 0; i < n; ++i) sum_before += p[i];

        std::vector<Real> p_new;
        solver.pressure_diffusion_step(p, 1.0e-6, 1.0e-3, 1.0, dx, 0.001, n, p_new);
        Real sum_after = 0.0;
        for (int i = 0; i < n; ++i) sum_after += p_new[i];
        CHECK_NEAR(sum_after, sum_before, 1.0e-8, "porous: uniform pressure conserved");
    }

    // Test 7: Steady-state pressure — linear profile with fixed endpoints
    // For constant kappa, steady state d2p/dx2 = 0 => linear
    {
        int n = 20;
        Real dx = 0.05;
        std::vector<Real> p(n);
        // Start with parabolic: should relax toward linear
        for (int i = 0; i < n; ++i) {
            Real x = (i + 0.5) * dx;
            p[i] = x * (1.0 - x); // parabolic
        }
        // Fix boundaries manually after each step (Dirichlet)
        Real p_left = 0.0, p_right = 0.0;
        std::vector<Real> p_new;
        // Use small storage coefficient for faster diffusion convergence
        Real storage = 0.001;
        for (int s = 0; s < 50000; ++s) {
            solver.pressure_diffusion_step(p, 1.0e-6, 1.0e-3, storage, dx, 0.001, n, p_new);
            p_new[0] = p_left;
            p_new[n-1] = p_right;
            p = p_new;
        }
        // Steady state with p=0 at both ends => p ≈ 0 everywhere
        Real max_p = 0.0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(p[i]) > max_p) max_p = std::abs(p[i]);
        }
        CHECK(max_p < 0.01, "porous: steady state converges to zero (fixed BC)");
    }

    // Test 8: Permeability scaling — doubling kappa doubles velocity
    {
        Real grad_p[3] = {1000.0, 0.0, 0.0};
        Real g[3] = {0.0, 0.0, 0.0};
        Real v1[3], v2[3];
        solver.darcy_velocity(1.0e-12, 1.0e-3, grad_p, 0.0, g, v1);
        solver.darcy_velocity(2.0e-12, 1.0e-3, grad_p, 0.0, g, v2);
        CHECK_NEAR(v2[0], 2.0 * v1[0], 1.0e-15, "porous: doubling kappa doubles velocity");
    }

    // Test 9: Gravity-driven flow
    {
        Real grad_p[3] = {0.0, 0.0, 0.0};
        Real g[3] = {0.0, 0.0, -9.81};
        Real rho_f = 1000.0;
        Real kappa = 1.0e-10, mu = 1.0e-3;
        Real v[3];
        solver.darcy_velocity(kappa, mu, grad_p, rho_f, g, v);
        // v_z = -(kappa/mu)*(0 - rho_f*(-9.81)) = -(kappa/mu)*(9810) = -9.81e-4
        Real expected_vz = -(kappa / mu) * (0.0 - rho_f * (-9.81));
        CHECK_NEAR(v[2], expected_vz, 1.0e-12, "porous: gravity-driven flow z");
        CHECK(v[2] < 0.0, "porous: gravity drives flow downward");
    }

    // Test 10: Storage coefficient — larger S means slower diffusion
    {
        int n = 30;
        Real dx = 0.1;
        std::vector<Real> p(n, 0.0);
        p[n/2] = 1.0; // point source

        std::vector<Real> p_new_S1, p_new_S10;
        solver.pressure_diffusion_step(p, 1.0e-6, 1.0e-3, 1.0, dx, 0.001, n, p_new_S1);
        solver.pressure_diffusion_step(p, 1.0e-6, 1.0e-3, 10.0, dx, 0.001, n, p_new_S10);

        // Peak pressure after one step: larger S => slower diffusion => higher peak
        CHECK(p_new_S10[n/2] > p_new_S1[n/2],
              "porous: larger storage coefficient slows diffusion");
    }
}

// ============================================================================
// Test Group 4: ALEErosionHandler (10 tests)
// ============================================================================

void test_erosion() {
    std::cout << "=== ALEErosionHandler Tests ===\n";

    // Test 1: Mass conservation after erosion
    {
        ALEErosionHandler handler;
        Real mass = 10.0;
        Real momentum[3] = {1.0, 2.0, 3.0};
        Real energy = 50.0;
        std::vector<Real> areas = {1.0, 2.0, 3.0};
        int n_neighbors = 3;

        std::vector<Real> om, opx, opy, opz, oe;
        handler.erode_element(0, mass, momentum, energy, areas, n_neighbors,
                              om, opx, opy, opz, oe);

        Real total_m = 0.0;
        for (int i = 0; i < n_neighbors; ++i) total_m += om[i];
        CHECK_NEAR(total_m, mass, 1.0e-12, "erosion: mass conserved");
    }

    // Test 2: Momentum conservation
    {
        ALEErosionHandler handler;
        Real mass = 5.0;
        Real momentum[3] = {10.0, -20.0, 30.0};
        Real energy = 100.0;
        std::vector<Real> areas = {1.0, 1.0, 1.0, 1.0};

        std::vector<Real> om, opx, opy, opz, oe;
        handler.erode_element(1, mass, momentum, energy, areas, 4,
                              om, opx, opy, opz, oe);

        Real total_px = 0.0, total_py = 0.0, total_pz = 0.0;
        for (int i = 0; i < 4; ++i) {
            total_px += opx[i];
            total_py += opy[i];
            total_pz += opz[i];
        }
        CHECK_NEAR(total_px, momentum[0], 1.0e-12, "erosion: momentum-x conserved");
        CHECK_NEAR(total_py, momentum[1], 1.0e-12, "erosion: momentum-y conserved");
        CHECK_NEAR(total_pz, momentum[2], 1.0e-12, "erosion: momentum-z conserved");
    }

    // Test 3: Energy conservation
    {
        ALEErosionHandler handler;
        Real mass = 1.0, momentum[3] = {0.0, 0.0, 0.0}, energy = 77.0;
        std::vector<Real> areas = {2.0, 3.0};

        std::vector<Real> om, opx, opy, opz, oe;
        handler.erode_element(2, mass, momentum, energy, areas, 2,
                              om, opx, opy, opz, oe);

        Real total_e = oe[0] + oe[1];
        CHECK_NEAR(total_e, energy, 1.0e-12, "erosion: energy conserved");
    }

    // Test 4: Void marking
    {
        ALEErosionHandler handler;
        handler.mark_eroded(5);
        handler.mark_eroded(10);
        CHECK(handler.is_eroded(5), "erosion: element 5 marked as eroded");
        CHECK(handler.is_eroded(10), "erosion: element 10 marked as eroded");
        CHECK(!handler.is_eroded(3), "erosion: element 3 not eroded");
    }

    // Test 5: Multiple erosions
    {
        ALEErosionHandler handler;
        for (int id = 0; id < 5; ++id) {
            Real mass = 1.0, momentum[3] = {0.0, 0.0, 0.0}, energy = 1.0;
            std::vector<Real> areas = {1.0};
            std::vector<Real> om, opx, opy, opz, oe;
            handler.erode_element(id, mass, momentum, energy, areas, 1,
                                  om, opx, opy, opz, oe);
        }
        CHECK(handler.num_eroded() == 5, "erosion: 5 elements eroded");
        for (int id = 0; id < 5; ++id) {
            CHECK(handler.is_eroded(id), "erosion: all 5 elements marked");
        }
    }

    // Test 6: Neighbor weighting by area
    {
        ALEErosionHandler handler;
        Real mass = 12.0, momentum[3] = {0.0, 0.0, 0.0}, energy = 0.0;
        // Areas: 1, 2, 3 => weights 1/6, 2/6, 3/6
        std::vector<Real> areas = {1.0, 2.0, 3.0};

        std::vector<Real> om, opx, opy, opz, oe;
        handler.erode_element(0, mass, momentum, energy, areas, 3,
                              om, opx, opy, opz, oe);

        CHECK_NEAR(om[0], 2.0, 1.0e-12, "erosion: weight 1/6 of 12 = 2");
        CHECK_NEAR(om[1], 4.0, 1.0e-12, "erosion: weight 2/6 of 12 = 4");
        CHECK_NEAR(om[2], 6.0, 1.0e-12, "erosion: weight 3/6 of 12 = 6");
    }

    // Test 7: No redistribution to already-eroded neighbors
    // (handler doesn't filter — but we verify eroded state is tracked)
    {
        ALEErosionHandler handler;
        handler.mark_eroded(100);
        CHECK(handler.is_eroded(100), "erosion: pre-eroded element tracked");
        // User should check is_eroded before including in neighbor list
        CHECK(!handler.is_eroded(101), "erosion: non-eroded neighbor available");
    }

    // Test 8: Total mass tracking
    {
        Real error = ALEErosionHandler::total_mass_check(100.0, 100.0);
        CHECK_NEAR(error, 0.0, 1.0e-14, "erosion: zero mass error for perfect conservation");

        Real error2 = ALEErosionHandler::total_mass_check(100.0, 99.0);
        CHECK_NEAR(error2, 0.01, 1.0e-14, "erosion: 1% mass error detected");
    }

    // Test 9: Chain erosion — erode neighbors sequentially
    {
        ALEErosionHandler handler;
        // Element 0 erodes, distributes to element 1
        Real m0 = 10.0, p0[3] = {5.0, 0.0, 0.0}, e0 = 20.0;
        std::vector<Real> areas0 = {1.0}; // only neighbor is elem 1
        std::vector<Real> om, opx, opy, opz, oe;
        handler.erode_element(0, m0, p0, e0, areas0, 1, om, opx, opy, opz, oe);

        // Element 1 now has extra mass, then also erodes
        Real m1 = 8.0 + om[0]; // original + received
        Real p1[3] = {3.0 + opx[0], opy[0], opz[0]};
        Real e1 = 15.0 + oe[0];
        std::vector<Real> areas1 = {2.0}; // distributes to elem 2
        std::vector<Real> om2, opx2, opy2, opz2, oe2;
        handler.erode_element(1, m1, p1, e1, areas1, 1, om2, opx2, opy2, opz2, oe2);

        CHECK_NEAR(om2[0], m1, 1.0e-12, "erosion: chain erosion mass passes through");
        CHECK(handler.is_eroded(0) && handler.is_eroded(1),
              "erosion: both chain elements marked");
    }

    // Test 10: Empty neighbor list handling
    {
        ALEErosionHandler handler;
        Real mass = 5.0, momentum[3] = {1.0, 2.0, 3.0}, energy = 10.0;
        std::vector<Real> areas;
        std::vector<Real> om, opx, opy, opz, oe;
        handler.erode_element(99, mass, momentum, energy, areas, 0,
                              om, opx, opy, opz, oe);
        CHECK(handler.is_eroded(99), "erosion: element eroded even with no neighbors");
        CHECK(om.empty(), "erosion: empty output for no neighbors");
    }
}

// ============================================================================
// Test Group 5: ALEAdaptiveRemesh (10 tests)
// ============================================================================

void test_adaptive_remesh() {
    std::cout << "=== ALEAdaptiveRemesh Tests ===\n";
    AdaptiveRemeshIndicator remesh;

    // Test 1: Uniform field → zero error indicator
    {
        int n = 10;
        std::vector<Real> values(n, 5.0);
        std::vector<Real> gradients(n, 0.0);
        std::vector<Real> sizes(n, 0.1);
        std::vector<Real> indicators;
        remesh.compute_error_indicator(values, gradients, sizes, n, indicators);

        Real max_eta = *std::max_element(indicators.begin(), indicators.end());
        CHECK_NEAR(max_eta, 0.0, 1.0e-14, "remesh: uniform field zero error");
    }

    // Test 2: Linear gradient → small error (constant gradient → zero second derivative)
    {
        int n = 20;
        std::vector<Real> values(n), gradients(n), sizes(n, 0.1);
        for (int i = 0; i < n; ++i) {
            values[i] = 2.0 * i * 0.1; // linear
            gradients[i] = 2.0;         // constant gradient
        }
        std::vector<Real> indicators;
        remesh.compute_error_indicator(values, gradients, sizes, n, indicators);

        // Interior elements should have zero indicator (constant gradient)
        Real max_interior = 0.0;
        for (int i = 1; i < n-1; ++i) {
            if (indicators[i] > max_interior) max_interior = indicators[i];
        }
        CHECK_NEAR(max_interior, 0.0, 1.0e-14, "remesh: linear gradient zero error interior");
    }

    // Test 3: High gradient region → flagged for refinement
    {
        int n = 20;
        std::vector<Real> values(n), gradients(n), sizes(n, 0.1);
        for (int i = 0; i < n; ++i) {
            values[i] = 0.0;
            gradients[i] = 0.0;
        }
        // Spike in gradient at center
        gradients[9]  = 0.0;
        gradients[10] = 100.0;
        gradients[11] = 0.0;

        std::vector<Real> indicators;
        remesh.compute_error_indicator(values, gradients, sizes, n, indicators);

        std::vector<int> flags;
        remesh.mark_refinement(indicators, n, 0.5, 0.1, flags);

        // Elements near the spike should be flagged for refinement
        bool any_refine = false;
        for (int i = 8; i <= 12; ++i) {
            if (flags[i] == 1) any_refine = true;
        }
        CHECK(any_refine, "remesh: high gradient region flagged for refinement");
    }

    // Test 4: Smooth field → peak not coarsened
    {
        int n = 20;
        std::vector<Real> values(n), gradients(n), sizes(n, 0.1);
        for (int i = 0; i < n; ++i) {
            Real x = (i + 0.5) * 0.1;
            values[i] = std::sin(M_PI * x / 2.0);
            gradients[i] = (M_PI / 2.0) * std::cos(M_PI * x / 2.0);
        }

        std::vector<Real> indicators;
        remesh.compute_error_indicator(values, gradients, sizes, n, indicators);

        std::vector<int> flags;
        remesh.mark_refinement(indicators, n, 0.5, 0.1, flags);

        // The peak of the sine is at x=1 (last elements) — gradient near zero there,
        // but second derivative is nonzero. Key: the maximum error region should not
        // be flagged for coarsening.
        int max_idx = 0;
        Real max_val = indicators[0];
        for (int i = 1; i < n; ++i) {
            if (indicators[i] > max_val) { max_val = indicators[i]; max_idx = i; }
        }
        CHECK(flags[max_idx] != -1, "remesh: peak error not flagged for coarsening");
    }

    // Test 5: Transfer preserves constant field
    {
        int n_old = 10, n_new = 15;
        std::vector<Real> old_vals(n_old, 42.0);
        std::vector<Real> old_coords(n_old), new_coords(n_new);
        for (int i = 0; i < n_old; ++i) old_coords[i] = i * 0.1;
        for (int i = 0; i < n_new; ++i) new_coords[i] = i * 0.06;

        std::vector<Real> new_vals;
        remesh.transfer_field(old_vals, old_coords, new_coords, n_old, n_new, new_vals);

        bool all_42 = true;
        for (int i = 0; i < n_new; ++i) {
            if (std::abs(new_vals[i] - 42.0) > 1.0e-10) all_42 = false;
        }
        CHECK(all_42, "remesh: transfer preserves constant field");
    }

    // Test 6: Transfer preserves linear field
    {
        int n_old = 10, n_new = 20;
        std::vector<Real> old_vals(n_old), old_coords(n_old), new_coords(n_new);
        for (int i = 0; i < n_old; ++i) {
            old_coords[i] = i * 0.1;
            old_vals[i] = 3.0 * old_coords[i] + 1.0; // f(x) = 3x + 1
        }
        for (int i = 0; i < n_new; ++i) {
            new_coords[i] = i * 0.045; // different spacing
        }

        std::vector<Real> new_vals;
        remesh.transfer_field(old_vals, old_coords, new_coords, n_old, n_new, new_vals);

        bool linear_ok = true;
        for (int i = 0; i < n_new; ++i) {
            Real expected = 3.0 * new_coords[i] + 1.0;
            if (std::abs(new_vals[i] - expected) > 1.0e-8) {
                linear_ok = false;
            }
        }
        CHECK(linear_ok, "remesh: transfer preserves linear field");
    }

    // Test 7: Conservation of integral (area under curve)
    {
        int n_old = 20, n_new = 20;
        Real dx_old = 0.05, dx_new = 0.05;
        std::vector<Real> old_vals(n_old), old_coords(n_old), new_coords(n_new);
        std::vector<Real> old_sizes(n_old, dx_old), new_sizes(n_new, dx_new);

        for (int i = 0; i < n_old; ++i) {
            old_coords[i] = (i + 0.5) * dx_old;
            old_vals[i] = std::sin(M_PI * old_coords[i]);
        }
        for (int i = 0; i < n_new; ++i) {
            new_coords[i] = (i + 0.5) * dx_new;
        }

        std::vector<Real> new_vals;
        remesh.transfer_field(old_vals, old_coords, new_coords, n_old, n_new, new_vals);

        Real integral_old = AdaptiveRemeshIndicator::compute_integral(old_vals, old_sizes, n_old);
        Real integral_new = AdaptiveRemeshIndicator::compute_integral(new_vals, new_sizes, n_new);
        CHECK_NEAR(integral_new, integral_old, 0.01, "remesh: integral conserved");
    }

    // Test 8: Refinement fraction — about half flagged for refine with theta=0.5
    {
        int n = 100;
        std::vector<Real> indicators(n);
        // Linearly increasing indicators
        for (int i = 0; i < n; ++i) indicators[i] = static_cast<Real>(i);

        std::vector<int> flags;
        remesh.mark_refinement(indicators, n, 0.5, 0.1, flags);

        int refine_count = 0;
        for (int i = 0; i < n; ++i) {
            if (flags[i] == 1) refine_count++;
        }
        // With linear indicators 0..99, max=99, threshold=49.5
        // Elements 50..99 should be flagged (50 elements)
        CHECK_NEAR(refine_count, 50, 1, "remesh: ~50% flagged for refinement");
    }

    // Test 9: Coarsening fraction
    {
        int n = 100;
        std::vector<Real> indicators(n);
        for (int i = 0; i < n; ++i) indicators[i] = static_cast<Real>(i);

        std::vector<int> flags;
        remesh.mark_refinement(indicators, n, 0.5, 0.1, flags);

        int coarsen_count = 0;
        for (int i = 0; i < n; ++i) {
            if (flags[i] == -1) coarsen_count++;
        }
        // threshold = 0.1 * 99 = 9.9 => elements 0..9 flagged (10 elements)
        CHECK_NEAR(coarsen_count, 10, 1, "remesh: ~10% flagged for coarsening");
    }

    // Test 10: Combined refine + coarsen — flags are mutually exclusive
    {
        int n = 50;
        std::vector<Real> indicators(n);
        for (int i = 0; i < n; ++i) {
            indicators[i] = (i < 10) ? 0.01 : ((i > 40) ? 10.0 : 1.0);
        }

        std::vector<int> flags;
        remesh.mark_refinement(indicators, n, 0.5, 0.1, flags);

        bool no_overlap = true;
        for (int i = 0; i < n; ++i) {
            if (flags[i] != -1 && flags[i] != 0 && flags[i] != 1) {
                no_overlap = false;
            }
        }
        CHECK(no_overlap, "remesh: flags are valid (-1, 0, or 1)");

        // Check that low-error region is coarsened
        bool low_coarsened = false;
        for (int i = 0; i < 10; ++i) {
            if (flags[i] == -1) low_coarsened = true;
        }
        CHECK(low_coarsened, "remesh: low-error region has coarsened elements");

        // Check that high-error region is refined
        bool high_refined = false;
        for (int i = 41; i < 50; ++i) {
            if (flags[i] == 1) high_refined = true;
        }
        CHECK(high_refined, "remesh: high-error region has refined elements");
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "Wave 27: Advanced ALE Extensions Test Suite\n";
    std::cout << "============================================\n\n";

    test_kepsilon();
    test_bimat_tracker();
    test_porous_media();
    test_erosion();
    test_adaptive_remesh();

    std::cout << "\n============================================\n";
    std::cout << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " total\n";

    return (tests_failed > 0) ? 1 : 0;
}
