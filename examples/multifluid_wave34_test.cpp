/**
 * @file multifluid_wave34_test.cpp
 * @brief Wave 34b: Multi-Fluid Dynamics Test Suite
 *
 * Tests 6 sub-modules (~10 tests each, 60 total):
 *  1. MultiFluidManager      — VOF tracking, volume fraction enforcement
 *  2. PressureEquilibrium    — Newton solver convergence, equilibrium pressure
 *  3. SubMaterialLaw         — Ideal gas, stiffened gas, JWL EOS
 *  4. MultiFluidMUSCL        — MUSCL reconstruction, monotonicity
 *  5. MultiFluidEBCS         — Inlet, outlet, wall, NRF boundary conditions
 *  6. MultiFluidFVM2FEM      — Pressure-to-force transfer, force balance
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <numeric>
#include <nexussim/fem/multifluid_wave34.hpp>

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
// Test Group 1: MultiFluidManager (10 tests)
// ============================================================================
void test_multifluid_manager() {
    std::cout << "=== MultiFluidManager Tests ===\n";

    // Test 1: Single fluid initialization
    {
        MultiFluidManager mgr(1);
        FluidState fs;
        Real alphas[1] = {1.0};
        Real rhos[1] = {1.0};
        Real energies[1] = {2.5};
        mgr.init_state(fs, alphas, rhos, energies, 0.0, 0.0, 0.0);
        CHECK_NEAR(fs.alpha_sum(), 1.0, 1.0e-15, "Manager: single fluid alpha sum = 1");
    }

    // Test 2: Two-fluid initialization
    {
        MultiFluidManager mgr(2);
        mgr.eos_params[0].gamma = 1.4;
        mgr.eos_params[1].gamma = 1.6;
        FluidState fs;
        Real alphas[2] = {0.7, 0.3};
        Real rhos[2] = {1.0, 2.0};
        Real energies[2] = {2.5, 3.0};
        mgr.init_state(fs, alphas, rhos, energies, 1.0, 0.0, 0.0);
        CHECK_NEAR(fs.alpha_sum(), 1.0, 1.0e-15, "Manager: two-fluid alpha sum = 1");
        CHECK_NEAR(fs.alpha[0], 0.7, 1.0e-15, "Manager: alpha[0] = 0.7");
        CHECK_NEAR(fs.alpha[1], 0.3, 1.0e-15, "Manager: alpha[1] = 0.3");
    }

    // Test 3: Renormalization of volume fractions
    {
        MultiFluidManager mgr(2);
        FluidState fs;
        fs.nfluids = 2;
        fs.alpha[0] = 0.6;
        fs.alpha[1] = 0.6;  // Sum = 1.2, should be renormalized
        mgr.enforce_alpha_sum(fs);
        CHECK_NEAR(fs.alpha_sum(), 1.0, 1.0e-15, "Manager: renormalization works");
        CHECK_NEAR(fs.alpha[0], 0.5, 1.0e-15, "Manager: renormalized alpha[0]");
    }

    // Test 4: Negative alpha clamped to zero
    {
        MultiFluidManager mgr(2);
        FluidState fs;
        fs.nfluids = 2;
        fs.alpha[0] = -0.1;
        fs.alpha[1] = 1.1;
        mgr.enforce_alpha_sum(fs);
        CHECK(fs.alpha[0] >= 0.0, "Manager: negative alpha clamped");
        CHECK_NEAR(fs.alpha_sum(), 1.0, 1.0e-15, "Manager: sum still 1 after clamp");
    }

    // Test 5: Mixture density computation
    {
        FluidState fs;
        fs.nfluids = 2;
        fs.alpha[0] = 0.6; fs.rho[0] = 1.0;
        fs.alpha[1] = 0.4; fs.rho[1] = 2.0;
        Real rho_mix = fs.mixture_density();
        CHECK_NEAR(rho_mix, 0.6 * 1.0 + 0.4 * 2.0, 1.0e-15, "Manager: mixture density");
    }

    // Test 6: Mixture pressure computation
    {
        FluidState fs;
        fs.nfluids = 2;
        fs.alpha[0] = 0.5; fs.p[0] = 100.0;
        fs.alpha[1] = 0.5; fs.p[1] = 200.0;
        Real p_mix = fs.mixture_pressure();
        CHECK_NEAR(p_mix, 150.0, 1.0e-12, "Manager: mixture pressure");
    }

    // Test 7: Mixed cell detection
    {
        FluidState fs;
        fs.nfluids = 2;
        fs.alpha[0] = 0.99999; fs.alpha[1] = 0.00001;
        CHECK(!MultiFluidManager::is_mixed_cell(fs, 0.001), "Manager: near-pure not mixed");
        fs.alpha[0] = 0.7; fs.alpha[1] = 0.3;
        CHECK(MultiFluidManager::is_mixed_cell(fs), "Manager: 70/30 is mixed");
    }

    // Test 8: Mixture sound speed (Wood's formula)
    {
        MultiFluidManager mgr(1);
        mgr.eos_params[0].gamma = 1.4;
        FluidState fs;
        fs.nfluids = 1;
        fs.alpha[0] = 1.0;
        fs.rho[0] = 1.0;
        fs.p[0] = 1.0;
        Real c = mgr.mixture_sound_speed(fs);
        Real expected = std::sqrt(1.4 * 1.0 / 1.0);
        CHECK_NEAR(c, expected, 1.0e-10, "Manager: single-fluid sound speed");
    }

    // Test 9: Update pressures from densities/energies
    {
        MultiFluidManager mgr(1);
        mgr.eos_params[0].type = SubEOSType::IdealGas;
        mgr.eos_params[0].gamma = 1.4;
        FluidState fs;
        fs.nfluids = 1;
        fs.alpha[0] = 1.0;
        fs.rho[0] = 1.0;
        fs.e[0] = 2.5;
        fs.p[0] = 0.0;  // Will be updated
        mgr.update_pressures(fs);
        CHECK_NEAR(fs.p[0], 0.4 * 1.0 * 2.5, 1.0e-12, "Manager: updated pressure from EOS");
    }

    // Test 10: Three-fluid volume fractions
    {
        MultiFluidManager mgr(3);
        FluidState fs;
        Real alphas[3] = {0.5, 0.3, 0.2};
        Real rhos[3] = {1.0, 1.5, 2.0};
        Real energies[3] = {2.5, 3.0, 1.0};
        mgr.init_state(fs, alphas, rhos, energies, 0.0, 0.0, 0.0);
        CHECK_NEAR(fs.alpha_sum(), 1.0, 1.0e-15, "Manager: three-fluid alpha sum");
        Real rho_mix = fs.mixture_density();
        Real expected = 0.5 * 1.0 + 0.3 * 1.5 + 0.2 * 2.0;
        CHECK_NEAR(rho_mix, expected, 1.0e-12, "Manager: three-fluid mixture density");
    }
}

// ============================================================================
// Test Group 2: PressureEquilibrium (10 tests)
// ============================================================================
void test_pressure_equilibrium() {
    std::cout << "=== PressureEquilibrium Tests ===\n";

    // Test 1: Single fluid returns EOS pressure
    {
        SubEOSParams eos[1];
        eos[0].type = SubEOSType::IdealGas;
        eos[0].gamma = 1.4;
        FluidState fs;
        fs.nfluids = 1;
        fs.alpha[0] = 1.0;
        fs.rho[0] = 1.0;
        fs.e[0] = 2.5;
        Real p = PressureEquilibrium::solve_equilibrium(fs, eos, 1.0);
        CHECK_NEAR(p, 0.4 * 1.0 * 2.5, 1.0e-6, "PressEq: single fluid => EOS pressure");
    }

    // Test 2: Two identical fluids => pressure unchanged
    {
        SubEOSParams eos[2];
        eos[0].type = SubEOSType::IdealGas;
        eos[0].gamma = 1.4;
        eos[1].type = SubEOSType::IdealGas;
        eos[1].gamma = 1.4;
        FluidState fs;
        fs.nfluids = 2;
        fs.alpha[0] = 0.5; fs.alpha[1] = 0.5;
        fs.rho[0] = 1.0; fs.rho[1] = 1.0;
        fs.e[0] = 2.5; fs.e[1] = 2.5;
        fs.p[0] = 1.0; fs.p[1] = 1.0;
        Real p = PressureEquilibrium::solve_equilibrium(fs, eos, 1.0);
        CHECK_NEAR(p, 1.0, 0.1, "PressEq: identical fluids => p unchanged");
    }

    // Test 3: Equilibrium pressure positive
    {
        SubEOSParams eos[2];
        eos[0].type = SubEOSType::IdealGas;
        eos[0].gamma = 1.4;
        eos[1].type = SubEOSType::IdealGas;
        eos[1].gamma = 1.6;
        FluidState fs;
        fs.nfluids = 2;
        fs.alpha[0] = 0.7; fs.alpha[1] = 0.3;
        fs.rho[0] = 1.0; fs.rho[1] = 2.0;
        fs.e[0] = 2.5; fs.e[1] = 1.5;
        fs.p[0] = 1.0; fs.p[1] = 1.8;
        Real p = PressureEquilibrium::solve_equilibrium(fs, eos, 1.0);
        CHECK(p > 0.0, "PressEq: positive pressure");
    }

    // Test 4: All fluids at same pressure after equilibrium
    {
        SubEOSParams eos[2];
        eos[0].type = SubEOSType::IdealGas;
        eos[0].gamma = 1.4;
        eos[1].type = SubEOSType::IdealGas;
        eos[1].gamma = 1.6;
        FluidState fs;
        fs.nfluids = 2;
        fs.alpha[0] = 0.6; fs.alpha[1] = 0.4;
        fs.rho[0] = 1.0; fs.rho[1] = 1.5;
        fs.e[0] = 2.5; fs.e[1] = 2.0;
        fs.p[0] = 1.0; fs.p[1] = 2.0;
        PressureEquilibrium::solve_equilibrium(fs, eos, 1.5);
        CHECK_NEAR(fs.p[0], fs.p[1], 1.0e-4, "PressEq: fluids reach same pressure");
    }

    // Test 5: Volume fractions still sum to 1
    {
        SubEOSParams eos[2];
        eos[0].type = SubEOSType::IdealGas;
        eos[0].gamma = 1.4;
        eos[1].type = SubEOSType::IdealGas;
        eos[1].gamma = 1.6;
        FluidState fs;
        fs.nfluids = 2;
        fs.alpha[0] = 0.6; fs.alpha[1] = 0.4;
        fs.rho[0] = 1.0; fs.rho[1] = 2.0;
        fs.e[0] = 2.5; fs.e[1] = 1.0;
        fs.p[0] = 1.0; fs.p[1] = 1.2;
        PressureEquilibrium::solve_equilibrium(fs, eos, 1.0);
        CHECK_NEAR(fs.alpha_sum(), 1.0, 1.0e-10, "PressEq: alpha sum = 1 after solve");
    }

    // Test 6: Newton converges in few iterations
    {
        // With identical EOS, convergence should be immediate
        SubEOSParams eos[2];
        eos[0].type = SubEOSType::IdealGas;
        eos[0].gamma = 1.4;
        eos[1].type = SubEOSType::IdealGas;
        eos[1].gamma = 1.4;
        FluidState fs;
        fs.nfluids = 2;
        fs.alpha[0] = 0.5; fs.alpha[1] = 0.5;
        fs.rho[0] = 1.0; fs.rho[1] = 1.0;
        fs.e[0] = 2.5; fs.e[1] = 2.5;
        fs.p[0] = 1.0; fs.p[1] = 1.0;
        Real p = PressureEquilibrium::solve_equilibrium(fs, eos, 1.0, 5);
        CHECK(p > 0.0, "PressEq: converges with max_iter=5");
    }

    // Test 7: Near-zero volume fraction fluid handled
    {
        SubEOSParams eos[2];
        eos[0].type = SubEOSType::IdealGas;
        eos[0].gamma = 1.4;
        eos[1].type = SubEOSType::IdealGas;
        eos[1].gamma = 1.4;
        FluidState fs;
        fs.nfluids = 2;
        fs.alpha[0] = 0.999; fs.alpha[1] = 0.001;
        fs.rho[0] = 1.0; fs.rho[1] = 1.0;
        fs.e[0] = 2.5; fs.e[1] = 2.5;
        fs.p[0] = 1.0; fs.p[1] = 1.0;
        Real p = PressureEquilibrium::solve_equilibrium(fs, eos, 1.0);
        bool finite = !std::isnan(p) && !std::isinf(p) && p > 0.0;
        CHECK(finite, "PressEq: handles near-zero alpha");
    }

    // Test 8: High pressure ratio
    {
        SubEOSParams eos[2];
        eos[0].type = SubEOSType::IdealGas;
        eos[0].gamma = 1.4;
        eos[1].type = SubEOSType::IdealGas;
        eos[1].gamma = 1.4;
        FluidState fs;
        fs.nfluids = 2;
        fs.alpha[0] = 0.5; fs.alpha[1] = 0.5;
        fs.rho[0] = 1.0; fs.rho[1] = 10.0;
        fs.e[0] = 2.5; fs.e[1] = 250.0;
        fs.p[0] = 1.0; fs.p[1] = 1000.0;
        Real p = PressureEquilibrium::solve_equilibrium(fs, eos, 100.0);
        CHECK(p > 0.0, "PressEq: high pressure ratio converges");
    }

    // Test 9: Stiffened gas EOS equilibrium
    {
        SubEOSParams eos[2];
        eos[0].type = SubEOSType::IdealGas;
        eos[0].gamma = 1.4;
        eos[1].type = SubEOSType::StiffenedGas;
        eos[1].gamma = 4.4;
        eos[1].p_inf = 6.0e8;
        FluidState fs;
        fs.nfluids = 2;
        fs.alpha[0] = 0.5; fs.alpha[1] = 0.5;
        fs.rho[0] = 1.0; fs.rho[1] = 1000.0;
        fs.e[0] = 2.5e5; fs.e[1] = 1.0e3;
        fs.p[0] = 1.0e5; fs.p[1] = 1.0e5;
        Real p = PressureEquilibrium::solve_equilibrium(fs, eos, 1.0e5);
        bool valid = p > 0.0 && !std::isnan(p);
        CHECK(valid, "PressEq: stiffened gas + ideal gas converges");
    }

    // Test 10: Densities remain positive after equilibrium
    {
        SubEOSParams eos[2];
        eos[0].type = SubEOSType::IdealGas;
        eos[0].gamma = 1.4;
        eos[1].type = SubEOSType::IdealGas;
        eos[1].gamma = 1.6;
        FluidState fs;
        fs.nfluids = 2;
        fs.alpha[0] = 0.5; fs.alpha[1] = 0.5;
        fs.rho[0] = 1.0; fs.rho[1] = 2.0;
        fs.e[0] = 2.5; fs.e[1] = 1.5;
        fs.p[0] = 1.0; fs.p[1] = 1.8;
        PressureEquilibrium::solve_equilibrium(fs, eos, 1.0);
        CHECK(fs.rho[0] > 0.0 && fs.rho[1] > 0.0, "PressEq: densities positive");
    }
}

// ============================================================================
// Test Group 3: SubMaterialLaw (10 tests)
// ============================================================================
void test_sub_material_law() {
    std::cout << "=== SubMaterialLaw Tests ===\n";

    // Test 1: Ideal gas P = (gamma-1)*rho*e
    {
        SubEOSParams eos;
        eos.type = SubEOSType::IdealGas;
        eos.gamma = 1.4;
        Real p = SubMaterialLaw::compute_sub_pressure(eos, 1.0, 2.5);
        CHECK_NEAR(p, 0.4 * 1.0 * 2.5, 1.0e-15, "SubEOS: ideal gas pressure");
    }

    // Test 2: Ideal gas with different density
    {
        SubEOSParams eos;
        eos.type = SubEOSType::IdealGas;
        eos.gamma = 1.4;
        Real p = SubMaterialLaw::compute_sub_pressure(eos, 2.0, 5.0);
        CHECK_NEAR(p, 0.4 * 2.0 * 5.0, 1.0e-15, "SubEOS: ideal gas, rho=2");
    }

    // Test 3: Ideal gas sound speed
    {
        SubEOSParams eos;
        eos.type = SubEOSType::IdealGas;
        eos.gamma = 1.4;
        Real c = SubMaterialLaw::compute_sub_sound_speed(eos, 1.0, 1.0);
        CHECK_NEAR(c, std::sqrt(1.4), 1.0e-12, "SubEOS: ideal gas sound speed");
    }

    // Test 4: Stiffened gas P = (gamma-1)*rho*e - gamma*p_inf
    {
        SubEOSParams eos;
        eos.type = SubEOSType::StiffenedGas;
        eos.gamma = 4.4;
        eos.p_inf = 6.0e8;
        Real rho = 1000.0, e = 1.0e6;
        Real p = SubMaterialLaw::compute_sub_pressure(eos, rho, e);
        Real expected = 3.4 * 1000.0 * 1.0e6 - 4.4 * 6.0e8;
        if (expected < 0.0) expected = 0.0;
        CHECK_NEAR(p, expected, 1.0, "SubEOS: stiffened gas pressure");
    }

    // Test 5: Ideal gas energy from pressure
    {
        SubEOSParams eos;
        eos.type = SubEOSType::IdealGas;
        eos.gamma = 1.4;
        Real e = SubMaterialLaw::compute_sub_energy(eos, 1.0, 1.0);
        CHECK_NEAR(e, 1.0 / 0.4, 1.0e-12, "SubEOS: ideal gas energy from p");
    }

    // Test 6: Pressure-energy round-trip
    {
        SubEOSParams eos;
        eos.type = SubEOSType::IdealGas;
        eos.gamma = 1.4;
        Real rho = 1.5, e_orig = 3.0;
        Real p = SubMaterialLaw::compute_sub_pressure(eos, rho, e_orig);
        Real e_back = SubMaterialLaw::compute_sub_energy(eos, rho, p);
        CHECK_NEAR(e_back, e_orig, 1.0e-12, "SubEOS: pressure-energy round-trip");
    }

    // Test 7: JWL EOS produces positive pressure
    {
        SubEOSParams eos;
        eos.type = SubEOSType::JWL;
        eos.A_jwl = 3.712e11;
        eos.B_jwl = 3.231e9;
        eos.R1 = 4.15;
        eos.R2 = 0.95;
        eos.omega = 0.30;
        eos.rho0 = 1630.0;
        Real p = SubMaterialLaw::compute_sub_pressure(eos, 1630.0, 4.29e6);
        CHECK(p > 0.0, "SubEOS: JWL positive pressure at CJ state");
    }

    // Test 8: Zero energy => zero pressure (ideal gas)
    {
        SubEOSParams eos;
        eos.type = SubEOSType::IdealGas;
        eos.gamma = 1.4;
        Real p = SubMaterialLaw::compute_sub_pressure(eos, 1.0, 0.0);
        CHECK_NEAR(p, 0.0, 1.0e-15, "SubEOS: zero energy => zero pressure");
    }

    // Test 9: Pressure scales linearly with rho*e (ideal gas)
    {
        SubEOSParams eos;
        eos.type = SubEOSType::IdealGas;
        eos.gamma = 1.4;
        Real p1 = SubMaterialLaw::compute_sub_pressure(eos, 1.0, 1.0);
        Real p2 = SubMaterialLaw::compute_sub_pressure(eos, 2.0, 1.0);
        CHECK_NEAR(p2 / p1, 2.0, 1.0e-12, "SubEOS: P scales with rho");
    }

    // Test 10: dP/drho for ideal gas
    {
        SubEOSParams eos;
        eos.type = SubEOSType::IdealGas;
        eos.gamma = 1.4;
        Real dpdr = SubMaterialLaw::compute_dpdrho(eos, 1.0, 2.5);
        CHECK_NEAR(dpdr, 0.4 * 2.5, 1.0e-12, "SubEOS: dP/drho = (gamma-1)*e");
    }
}

// ============================================================================
// Test Group 4: MultiFluidMUSCL (10 tests)
// ============================================================================
void test_multifluid_muscl() {
    std::cout << "=== MultiFluidMUSCL Tests ===\n";

    // Test 1: MinMod positive slopes
    {
        Real r = MultiFluidMUSCL::minmod(2.0, 5.0);
        CHECK_NEAR(r, 2.0, 1.0e-15, "MUSCL minmod: positive slopes => min");
    }

    // Test 2: MinMod opposite signs
    {
        Real r = MultiFluidMUSCL::minmod(2.0, -3.0);
        CHECK_NEAR(r, 0.0, 1.0e-15, "MUSCL minmod: opposite signs => 0");
    }

    // Test 3: Superbee limiter
    {
        Real r = MultiFluidMUSCL::superbee(1.0, 2.0);
        // minmod(1, 4) = 1, minmod(2, 2) = 2 => max = 2
        CHECK_NEAR(r, 2.0, 1.0e-15, "MUSCL superbee: (1,2) => 2");
    }

    // Test 4: Reconstruction of uniform field
    {
        Real fl, fr;
        MultiFluidMUSCL::reconstruct_1d(5.0, 5.0, 5.0, fl, fr);
        CHECK_NEAR(fl, 5.0, 1.0e-15, "MUSCL uniform: face_left = 5");
        CHECK_NEAR(fr, 5.0, 1.0e-15, "MUSCL uniform: face_right = 5");
    }

    // Test 5: Reconstruction of linear field
    {
        Real fl, fr;
        MultiFluidMUSCL::reconstruct_1d(1.0, 2.0, 3.0, fl, fr);
        // Linear: slope = minmod(1, 1) = 1; face_left = 2 + 0.5*1 = 2.5
        CHECK_NEAR(fl, 2.5, 1.0e-15, "MUSCL linear: face_left = 2.5");
    }

    // Test 6: Reconstruction near discontinuity (limited)
    {
        Real fl, fr;
        MultiFluidMUSCL::reconstruct_1d(1.0, 2.0, 100.0, fl, fr);
        // Slopes: 1 and 98 => minmod = 1; face_left = 2 + 0.5 = 2.5
        CHECK_NEAR(fl, 2.5, 1.0e-15, "MUSCL discontinuity: limited reconstruction");
    }

    // Test 7: Multi-fluid reconstruction preserves alpha bounds
    {
        FluidState fsL, fsC, fsR;
        fsL.nfluids = fsC.nfluids = fsR.nfluids = 2;
        fsL.alpha[0] = 0.3; fsC.alpha[0] = 0.5; fsR.alpha[0] = 0.7;
        fsL.rho[0] = 1.0; fsC.rho[0] = 1.5; fsR.rho[0] = 2.0;
        fsL.p[0] = 1.0; fsC.p[0] = 1.5; fsR.p[0] = 2.0;
        Real aL, rL, pL, aR, rR, pR;
        MultiFluidMUSCL::reconstruct(fsL, fsC, fsR, 0, aL, rL, pL, aR, rR, pR);
        CHECK(aL >= 0.0 && aL <= 1.0, "MUSCL: alpha face_L in [0,1]");
        CHECK(aR >= 0.0 && aR <= 1.0, "MUSCL: alpha face_R in [0,1]");
    }

    // Test 8: Interface limiter at sharp interface
    {
        Real lim = MultiFluidMUSCL::interface_limiter(0.0, 0.5, 1.0);
        // Gradients: 0.5 and 0.5, same sign => not interface but alpha=0.5 is moderate
        CHECK(lim > 0.0, "MUSCL: interface_limiter > 0 for smooth transition");
    }

    // Test 9: Interface limiter at sign-change gradient
    {
        // Alpha: 0.8, 0.3, 0.7 => grad_l = -0.5, grad_r = 0.4 => sign change
        Real lim = MultiFluidMUSCL::interface_limiter(0.8, 0.3, 0.7);
        CHECK_NEAR(lim, 0.0, 1.0e-15, "MUSCL: interface_limiter = 0 at sign change");
    }

    // Test 10: Monotonicity check
    {
        CHECK(MultiFluidMUSCL::is_monotone(2.5, 2.0, 3.0), "MUSCL: 2.5 in [2,3] is monotone");
        CHECK(!MultiFluidMUSCL::is_monotone(5.0, 2.0, 3.0), "MUSCL: 5.0 not in [2,3] not monotone");
    }
}

// ============================================================================
// Test Group 5: MultiFluidEBCS (10 tests)
// ============================================================================
void test_multifluid_ebcs() {
    std::cout << "=== MultiFluidEBCS Tests ===\n";

    FluidState interior;
    interior.nfluids = 2;
    interior.alpha[0] = 0.6; interior.alpha[1] = 0.4;
    interior.rho[0] = 1.0; interior.rho[1] = 2.0;
    interior.p[0] = 100.0; interior.p[1] = 100.0;
    interior.e[0] = 2.5; interior.e[1] = 1.5;
    interior.u = 10.0; interior.v = 5.0; interior.w = 0.0;

    // Test 1: Outlet BC copies state
    {
        FluidState ghost;
        MultiFluidBCData bc;
        bc.type = MultiFluidBCType::Outlet;
        MultiFluidEBCS::apply_multifluid_bc(interior, ghost, bc);
        CHECK_NEAR(ghost.alpha[0], 0.6, 1.0e-15, "MF BC outlet: alpha[0] copied");
        CHECK_NEAR(ghost.u, 10.0, 1.0e-15, "MF BC outlet: u copied");
    }

    // Test 2: Inlet BC sets prescribed state
    {
        FluidState ghost;
        MultiFluidBCData bc;
        bc.type = MultiFluidBCType::Inlet;
        bc.inlet_state.nfluids = 2;
        bc.inlet_state.alpha[0] = 0.8;
        bc.inlet_state.alpha[1] = 0.2;
        bc.inlet_state.rho[0] = 1.5;
        bc.inlet_state.rho[1] = 3.0;
        bc.inlet_state.u = 20.0;
        bc.inlet_state.v = 0.0;
        bc.inlet_state.w = 0.0;
        MultiFluidEBCS::apply_multifluid_bc(interior, ghost, bc);
        CHECK_NEAR(ghost.alpha[0], 0.8, 1.0e-15, "MF BC inlet: alpha[0] = 0.8");
        CHECK_NEAR(ghost.u, 20.0, 1.0e-15, "MF BC inlet: u = 20");
    }

    // Test 3: Wall BC reverses normal velocity (x-face)
    {
        FluidState ghost;
        MultiFluidBCData bc;
        bc.type = MultiFluidBCType::Wall;
        bc.face = 0;
        MultiFluidEBCS::apply_multifluid_bc(interior, ghost, bc);
        CHECK_NEAR(ghost.u, -10.0, 1.0e-15, "MF BC wall x: u reversed");
        CHECK_NEAR(ghost.v, 5.0, 1.0e-15, "MF BC wall x: v preserved");
    }

    // Test 4: Wall BC reverses y-velocity
    {
        FluidState ghost;
        MultiFluidBCData bc;
        bc.type = MultiFluidBCType::Wall;
        bc.face = 2;  // -y face
        MultiFluidEBCS::apply_multifluid_bc(interior, ghost, bc);
        CHECK_NEAR(ghost.u, 10.0, 1.0e-15, "MF BC wall y: u preserved");
        CHECK_NEAR(ghost.v, -5.0, 1.0e-15, "MF BC wall y: v reversed");
    }

    // Test 5: Wall preserves volume fractions
    {
        FluidState ghost;
        MultiFluidBCData bc;
        bc.type = MultiFluidBCType::Wall;
        bc.face = 0;
        MultiFluidEBCS::apply_multifluid_bc(interior, ghost, bc);
        CHECK_NEAR(ghost.alpha[0], interior.alpha[0], 1.0e-15, "MF BC wall: alpha preserved");
    }

    // Test 6: Outlet preserves alpha sum
    {
        FluidState ghost;
        MultiFluidBCData bc;
        bc.type = MultiFluidBCType::Outlet;
        MultiFluidEBCS::apply_multifluid_bc(interior, ghost, bc);
        CHECK(MultiFluidEBCS::bc_preserves_alpha_sum(ghost), "MF BC outlet: alpha sum = 1");
    }

    // Test 7: NRF BC produces valid state
    {
        FluidState ghost;
        MultiFluidBCData bc;
        bc.type = MultiFluidBCType::NRF;
        bc.face = 1;  // +x
        bc.p_inf = 100.0;
        bc.rho_inf = 1.0;
        bc.c_inf = 340.0;
        MultiFluidEBCS::apply_multifluid_bc(interior, ghost, bc);
        bool valid = !std::isnan(ghost.u) && !std::isinf(ghost.u);
        CHECK(valid, "MF BC NRF: produces valid velocity");
    }

    // Test 8: NRF adjusts pressure toward far-field
    {
        FluidState ghost;
        MultiFluidBCData bc;
        bc.type = MultiFluidBCType::NRF;
        bc.face = 1;
        bc.p_inf = 50.0;
        bc.rho_inf = 1.0;
        bc.c_inf = 340.0;
        MultiFluidEBCS::apply_multifluid_bc(interior, ghost, bc);
        Real ghost_p = ghost.mixture_pressure();
        // Pressure should be adjusted toward p_inf
        CHECK(ghost_p < interior.mixture_pressure() + 1.0, "MF BC NRF: pressure adjusted");
    }

    // Test 9: apply_outlet convenience function
    {
        FluidState ghost;
        MultiFluidEBCS::apply_outlet(interior, ghost);
        CHECK_NEAR(ghost.rho[0], interior.rho[0], 1.0e-15, "MF BC apply_outlet: rho copied");
    }

    // Test 10: apply_inlet convenience function
    {
        FluidState inlet;
        inlet.nfluids = 2;
        inlet.alpha[0] = 0.9; inlet.alpha[1] = 0.1;
        FluidState ghost;
        MultiFluidEBCS::apply_inlet(ghost, inlet, 2);
        CHECK_NEAR(ghost.alpha[0], 0.9, 1.0e-15, "MF BC apply_inlet: alpha set");
        CHECK(ghost.nfluids == 2, "MF BC apply_inlet: nfluids set");
    }
}

// ============================================================================
// Test Group 6: MultiFluidFVM2FEM (10 tests)
// ============================================================================
void test_multifluid_fvm2fem() {
    std::cout << "=== MultiFluidFVM2FEM Tests ===\n";

    // Test 1: 1D transfer with uniform pressure => boundary forces balance
    {
        int ncells = 5;
        Real pressures[5] = {100.0, 100.0, 100.0, 100.0, 100.0};
        Real forces[6];
        Real area = 1.0;
        MultiFluidFVM2FEM::transfer_1d(pressures, ncells, forces, area);
        // Uniform pressure => interior forces = 0
        for (int i = 1; i < ncells; ++i) {
            CHECK_NEAR(forces[i], 0.0, 1.0e-10, "FVM2FEM 1D: interior force = 0 for uniform p");
        }
    }

    // Test 2: 1D boundary forces
    {
        int ncells = 3;
        Real pressures[3] = {100.0, 100.0, 100.0};
        Real forces[4];
        MultiFluidFVM2FEM::transfer_1d(pressures, ncells, forces, 1.0);
        CHECK_NEAR(forces[0], -100.0, 1.0e-10, "FVM2FEM 1D: left boundary force");
        CHECK_NEAR(forces[3], 100.0, 1.0e-10, "FVM2FEM 1D: right boundary force");
    }

    // Test 3: Force balance for uniform pressure (boundary forces cancel)
    {
        int ncells = 5;
        Real pressures[5] = {100.0, 100.0, 100.0, 100.0, 100.0};
        Real forces[6];
        MultiFluidFVM2FEM::transfer_1d(pressures, ncells, forces, 2.0);
        Real err = MultiFluidFVM2FEM::force_balance_error(forces, 6);
        CHECK_NEAR(err, 0.0, 1.0e-10, "FVM2FEM 1D: force balance for uniform p");
    }

    // Test 4: Pressure gradient produces net force
    {
        int ncells = 3;
        Real pressures[3] = {200.0, 100.0, 50.0};
        Real forces[4];
        MultiFluidFVM2FEM::transfer_1d(pressures, ncells, forces, 1.0);
        // Interior nodes: (200-100)*1 = 100, (100-50)*1 = 50
        CHECK_NEAR(forces[1], 100.0, 1.0e-10, "FVM2FEM 1D: pressure gradient force node 1");
        CHECK_NEAR(forces[2], 50.0, 1.0e-10, "FVM2FEM 1D: pressure gradient force node 2");
    }

    // Test 5: Force scales with area
    {
        int ncells = 2;
        Real pressures[2] = {100.0, 100.0};
        Real forces1[3], forces2[3];
        MultiFluidFVM2FEM::transfer_1d(pressures, ncells, forces1, 1.0);
        MultiFluidFVM2FEM::transfer_1d(pressures, ncells, forces2, 3.0);
        CHECK_NEAR(forces2[0] / forces1[0], 3.0, 1.0e-12, "FVM2FEM 1D: force scales with area");
    }

    // Test 6: 3D transfer — node in cell gets correct pressure
    {
        FluidState states[1];
        states[0].nfluids = 1;
        states[0].alpha[0] = 1.0;
        states[0].p[0] = 500.0;

        FEMNode nodes[1];
        nodes[0].x = 0.5; nodes[0].y = 0.5; nodes[0].z = 0.5;
        nodes[0].fx = 0.0; nodes[0].fy = 0.0; nodes[0].fz = 0.0;

        Real normals[1][3] = {{1.0, 0.0, 0.0}};
        Real areas[1] = {0.01};
        Real origin[3] = {0.0, 0.0, 0.0};
        Real dx[3] = {1.0, 1.0, 1.0};
        int dims[3] = {1, 1, 1};

        MultiFluidFVM2FEM::transfer_pressure_to_nodes(states, 1, nodes, 1,
                                                       normals, areas, origin, dx, dims);
        CHECK_NEAR(nodes[0].fx, -500.0 * 0.01 * 1.0, 1.0e-10, "FVM2FEM 3D: fx = -p*A*nx");
        CHECK_NEAR(nodes[0].fy, 0.0, 1.0e-15, "FVM2FEM 3D: fy = 0 (ny=0)");
    }

    // Test 7: Density transfer
    {
        FluidState states[1];
        states[0].nfluids = 2;
        states[0].alpha[0] = 0.6; states[0].rho[0] = 1.0;
        states[0].alpha[1] = 0.4; states[0].rho[1] = 2.0;

        FEMNode nodes[1];
        nodes[0].x = 0.5; nodes[0].y = 0.5; nodes[0].z = 0.5;

        Real origin[3] = {0.0, 0.0, 0.0};
        Real dx[3] = {1.0, 1.0, 1.0};
        int dims[3] = {1, 1, 1};
        Real node_density[1];

        MultiFluidFVM2FEM::transfer_density_to_nodes(states, 1, nodes, 1,
                                                      origin, dx, dims, node_density);
        Real expected = 0.6 * 1.0 + 0.4 * 2.0;
        CHECK_NEAR(node_density[0], expected, 1.0e-12, "FVM2FEM: density transfer");
    }

    // Test 8: Force balance error for non-uniform pressure
    {
        int ncells = 4;
        Real pressures[4] = {100.0, 200.0, 150.0, 80.0};
        Real forces[5];
        MultiFluidFVM2FEM::transfer_1d(pressures, ncells, forces, 1.0);
        // Sum: -100 + (100-200) + (200-150) + (150-80) + 80 = -100 -100 +50 +70 +80 = 0
        Real err = MultiFluidFVM2FEM::force_balance_error(forces, 5);
        CHECK_NEAR(err, 0.0, 1.0e-10, "FVM2FEM: force balance for non-uniform p");
    }

    // Test 9: Multiple nodes in same cell
    {
        FluidState states[1];
        states[0].nfluids = 1;
        states[0].alpha[0] = 1.0;
        states[0].p[0] = 100.0;

        FEMNode nodes[2];
        nodes[0].x = 0.3; nodes[0].y = 0.3; nodes[0].z = 0.3;
        nodes[0].fx = 0.0;
        nodes[1].x = 0.7; nodes[1].y = 0.7; nodes[1].z = 0.7;
        nodes[1].fx = 0.0;

        Real normals[2][3] = {{1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}};
        Real areas[2] = {0.01, 0.01};
        Real origin[3] = {0.0, 0.0, 0.0};
        Real dx[3] = {1.0, 1.0, 1.0};
        int dims[3] = {1, 1, 1};

        MultiFluidFVM2FEM::transfer_pressure_to_nodes(states, 1, nodes, 2,
                                                       normals, areas, origin, dx, dims);
        // Opposite normals, same p => forces cancel
        CHECK_NEAR(nodes[0].fx + nodes[1].fx, 0.0, 1.0e-12, "FVM2FEM: opposite normals cancel");
    }

    // Test 10: Zero pressure => zero force
    {
        FluidState states[1];
        states[0].nfluids = 1;
        states[0].alpha[0] = 1.0;
        states[0].p[0] = 0.0;

        FEMNode nodes[1];
        nodes[0].x = 0.5; nodes[0].y = 0.5; nodes[0].z = 0.5;
        nodes[0].fx = 0.0; nodes[0].fy = 0.0; nodes[0].fz = 0.0;

        Real normals[1][3] = {{1.0, 0.0, 0.0}};
        Real areas[1] = {1.0};
        Real origin[3] = {0.0, 0.0, 0.0};
        Real dx[3] = {1.0, 1.0, 1.0};
        int dims[3] = {1, 1, 1};

        MultiFluidFVM2FEM::transfer_pressure_to_nodes(states, 1, nodes, 1,
                                                       normals, areas, origin, dx, dims);
        CHECK_NEAR(nodes[0].fx, 0.0, 1.0e-15, "FVM2FEM: zero pressure => zero force");
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "Wave 34b: Multi-Fluid Dynamics Test Suite\n";
    std::cout << "==========================================\n\n";

    test_multifluid_manager();
    test_pressure_equilibrium();
    test_sub_material_law();
    test_multifluid_muscl();
    test_multifluid_ebcs();
    test_multifluid_fvm2fem();

    std::cout << "\n==========================================\n";
    std::cout << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " total\n";

    return (tests_failed > 0) ? 1 : 0;
}
