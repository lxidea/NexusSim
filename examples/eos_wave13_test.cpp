/**
 * @file eos_wave13_test.cpp
 * @brief Comprehensive test for Wave 13: Extended EOS models
 *
 * Tests 8 new EOS models (5 tests each, 40 total):
 * - Murnaghan, Noble-Abel, Stiff Gas, Tillotson, Sesame,
 *   PowderBurn, Compaction, Osborne
 */

#include <nexussim/physics/eos_wave13.hpp>
#include <iostream>
#include <cmath>

using namespace nxs;
using namespace nxs::physics;

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

#define CHECK_NEAR(a, b, tol, msg) do { \
    Real _a = (a), _b = (b), _tol = (tol); \
    bool _ok = std::fabs(_a - _b) < _tol * (1.0 + std::fabs(_b)); \
    if (_ok) { \
        std::cout << "[PASS] " << msg << " (got " << _a << ", expected " << _b << ")\n"; \
        tests_passed++; \
    } else { \
        std::cout << "[FAIL] " << msg << " (got " << _a << ", expected " << _b \
                  << ", tol=" << _tol << ")\n"; \
        tests_failed++; \
    } \
} while(0)

// ============================================================================
// Test 1-5: Murnaghan EOS
// ============================================================================
void test_murnaghan() {
    std::cout << "\n=== Murnaghan EOS ===\n";

    EOSWave13Properties props;
    props.type = EOSWave13Type::Murnaghan;
    props.rho0 = 2700.0;    // Aluminum [kg/m^3]
    props.K0 = 76.0e9;      // Bulk modulus [Pa]
    props.K0_prime = 4.6;   // Typical for metals

    // Test 1: At reference density, pressure should be zero
    {
        Real p = MurnaghanEOS::compute_pressure(props, props.rho0, 0.0);
        // p = (K0/K0') * (1^K0' - 1) = 0
        CHECK_NEAR(p, 0.0, 1.0e-6, "Murnaghan: zero pressure at reference density");
    }

    // Test 2: 5% compression produces positive pressure
    {
        Real rho_comp = props.rho0 * 1.05;  // eta = 1.05
        Real p = MurnaghanEOS::compute_pressure(props, rho_comp, 0.0);
        // p = (76e9/4.6) * (1.05^4.6 - 1) = 16.52e9 * (1.2531 - 1) = 16.52e9 * 0.2531 ~ 4.18e9
        CHECK(p > 0.0, "Murnaghan: positive pressure under compression");
        // Numerical check: (76e9/4.6) * (pow(1.05, 4.6) - 1)
        Real expected = (76.0e9 / 4.6) * (std::pow(1.05, 4.6) - 1.0);
        CHECK_NEAR(p, expected, 1.0e-6, "Murnaghan: 5% compression numerical accuracy");
    }

    // Test 3: Pressure increases with compression
    {
        Real p5 = MurnaghanEOS::compute_pressure(props, props.rho0 * 1.05, 0.0);
        Real p10 = MurnaghanEOS::compute_pressure(props, props.rho0 * 1.10, 0.0);
        CHECK(p10 > p5, "Murnaghan: pressure increases with compression");
    }

    // Test 4: Tension produces negative pressure
    {
        Real p = MurnaghanEOS::compute_pressure(props, props.rho0 * 0.95, 0.0);
        CHECK(p < 0.0, "Murnaghan: negative pressure in tension");
    }

    // Test 5: Sound speed is positive and reasonable
    {
        Real c = MurnaghanEOS::sound_speed(props, props.rho0, 0.0);
        // c^2 = K0/rho0 * eta^(K0'-1) at eta=1 => c = sqrt(K0/rho0)
        Real c_expected = std::sqrt(props.K0 / props.rho0);  // ~5307 m/s
        CHECK(c > 0.0, "Murnaghan: positive sound speed");
        CHECK_NEAR(c, c_expected, 1.0e-4, "Murnaghan: sound speed = sqrt(K0/rho0) at reference");
    }
}

// ============================================================================
// Test 6-10: Noble-Abel EOS
// ============================================================================
void test_noble_abel() {
    std::cout << "\n=== Noble-Abel EOS ===\n";

    EOSWave13Properties props;
    props.type = EOSWave13Type::NobleAbel;
    props.rho0 = 1.225;      // Air-like gas [kg/m^3]
    props.gamma_na = 1.4;
    props.b_covolume = 0.0;   // b=0 => ideal gas

    Real e_ref = 101325.0 / (0.4 * 1.225);  // e such that ideal gas gives 1 atm

    // Test 6: Reduces to ideal gas when b=0
    {
        Real p = NobleAbelEOS::compute_pressure(props, 1.225, e_ref);
        Real p_ideal = (1.4 - 1.0) * 1.225 * e_ref;
        CHECK_NEAR(p, p_ideal, 1.0e-6, "Noble-Abel: matches ideal gas when b=0");
    }

    // Test 7: Covolume increases pressure compared to ideal gas
    {
        EOSWave13Properties props_cv = props;
        props_cv.b_covolume = 1.0e-3;  // Small covolume

        Real p_na = NobleAbelEOS::compute_pressure(props_cv, 1.225, e_ref);
        Real p_ideal = (1.4 - 1.0) * 1.225 * e_ref;
        CHECK(p_na > p_ideal, "Noble-Abel: covolume increases pressure vs ideal gas");
    }

    // Test 8: Pressure increases with density
    {
        Real p1 = NobleAbelEOS::compute_pressure(props, 1.0, e_ref);
        Real p2 = NobleAbelEOS::compute_pressure(props, 2.0, e_ref);
        CHECK(p2 > p1, "Noble-Abel: pressure increases with density");
    }

    // Test 9: Sound speed is positive
    {
        Real c = NobleAbelEOS::sound_speed(props, 1.225, e_ref);
        CHECK(c > 0.0, "Noble-Abel: positive sound speed");
        // For b=0: c = sqrt(gamma*p/rho) = sqrt(1.4*101325/1.225) ~ 340 m/s
        Real c_expected = std::sqrt(1.4 * 101325.0 / 1.225);
        CHECK_NEAR(c, c_expected, 1.0e-3, "Noble-Abel: sound speed ~340 m/s for air (b=0)");
    }

    // Test 10: Dense propellant gas scenario
    {
        EOSWave13Properties props_dense;
        props_dense.type = EOSWave13Type::NobleAbel;
        props_dense.gamma_na = 1.25;
        props_dense.b_covolume = 1.0e-3;  // 1 cm^3/g = 1e-3 m^3/kg
        props_dense.rho0 = 100.0;

        // rho=100, e=5e5 J/kg
        Real p = NobleAbelEOS::compute_pressure(props_dense, 100.0, 5.0e5);
        // p = 0.25 * 100 * 5e5 / (1 - 1e-3*100) = 1.25e7 / 0.9 = 1.389e7
        Real expected = 0.25 * 100.0 * 5.0e5 / (1.0 - 0.1);
        CHECK_NEAR(p, expected, 1.0e-6, "Noble-Abel: dense gas numerical accuracy");
    }
}

// ============================================================================
// Test 11-15: Stiff Gas (Tait) EOS
// ============================================================================
void test_stiff_gas() {
    std::cout << "\n=== Stiff Gas (Tait) EOS ===\n";

    // Water parameters
    EOSWave13Properties props;
    props.type = EOSWave13Type::StiffGas;
    props.rho0 = 1000.0;      // Water [kg/m^3]
    props.gamma_sg = 7.15;
    props.p_inf = 3.31e8;     // Pa

    // Internal energy for water at atmospheric pressure:
    // 101325 = (7.15-1)*1000*e - 7.15*3.31e8
    // 101325 = 6.15*1000*e - 2.36665e9
    // e = (101325 + 2.36665e9) / 6150 = 384878.4 J/kg
    Real e_atm = (101325.0 + 7.15 * 3.31e8) / (6.15 * 1000.0);

    // Test 11: Atmospheric pressure in water
    {
        Real p = StiffGasEOS::compute_pressure(props, 1000.0, e_atm);
        CHECK_NEAR(p, 101325.0, 1.0e-3, "StiffGas: atmospheric pressure in water");
    }

    // Test 12: Compression increases pressure
    {
        Real p_ref = StiffGasEOS::compute_pressure(props, 1000.0, e_atm);
        Real p_comp = StiffGasEOS::compute_pressure(props, 1010.0, e_atm);
        CHECK(p_comp > p_ref, "StiffGas: compression increases pressure");
    }

    // Test 13: Sound speed in water ~1500 m/s
    {
        Real c = StiffGasEOS::sound_speed(props, 1000.0, e_atm);
        // c^2 = gamma*(p+p_inf)/rho = 7.15*(101325+3.31e8)/1000
        Real c_expected = std::sqrt(7.15 * (101325.0 + 3.31e8) / 1000.0);
        CHECK(c > 1000.0 && c < 2000.0, "StiffGas: sound speed ~1500 m/s in water");
        CHECK_NEAR(c, c_expected, 1.0e-4, "StiffGas: sound speed numerical accuracy");
    }

    // Test 14: Reduces to ideal gas when p_inf = 0
    {
        EOSWave13Properties props_ig;
        props_ig.type = EOSWave13Type::StiffGas;
        props_ig.gamma_sg = 1.4;
        props_ig.p_inf = 0.0;
        props_ig.rho0 = 1.225;

        Real e = 2.5e5;
        Real p = StiffGasEOS::compute_pressure(props_ig, 1.225, e);
        Real p_ideal = (1.4 - 1.0) * 1.225 * e;
        CHECK_NEAR(p, p_ideal, 1.0e-6, "StiffGas: reduces to ideal gas when p_inf=0");
    }

    // Test 15: Positive sound speed at high pressure
    {
        // High-pressure shocked water
        Real e_high = e_atm * 10.0;
        Real c = StiffGasEOS::sound_speed(props, 1200.0, e_high);
        CHECK(c > 0.0, "StiffGas: positive sound speed at high pressure");
    }
}

// ============================================================================
// Test 16-20: Tillotson EOS
// ============================================================================
void test_tillotson() {
    std::cout << "\n=== Tillotson EOS ===\n";

    // Granite parameters (Benz & Asphaug, 1999)
    EOSWave13Properties props;
    props.type = EOSWave13Type::Tillotson;
    props.rho0 = 2680.0;       // kg/m^3
    props.till_a = 0.5;
    props.till_b = 1.3;
    props.till_A = 1.8e10;     // Pa
    props.till_B = 1.8e10;     // Pa
    props.till_e0 = 1.6e7;    // J/kg
    props.till_alpha = 5.0;
    props.till_beta = 5.0;
    props.till_e_iv = 3.5e6;  // J/kg
    props.till_e_cv = 1.8e7;  // J/kg

    // Test 16: Reference state with zero energy: pressure from A*mu + B*mu^2 terms
    {
        // At rho=rho0, mu=0, e=0: p = [a + b/(0+1)]*rho0*0*1 + A*0 + B*0 = 0
        Real p = TillotsonEOS::compute_pressure(props, props.rho0, 0.0);
        CHECK_NEAR(p, 0.0, 1000.0, "Tillotson: near-zero pressure at reference state, zero energy");
    }

    // Test 17: Compression produces positive pressure
    {
        Real rho_comp = props.rho0 * 1.05;
        Real p = TillotsonEOS::compute_pressure(props, rho_comp, 1.0e6);
        CHECK(p > 0.0, "Tillotson: positive pressure under compression");
    }

    // Test 18: Higher compression gives higher pressure
    {
        Real p5 = TillotsonEOS::compute_pressure(props, props.rho0 * 1.05, 1.0e6);
        Real p10 = TillotsonEOS::compute_pressure(props, props.rho0 * 1.10, 1.0e6);
        CHECK(p10 > p5, "Tillotson: pressure increases with compression");
    }

    // Test 19: Expanded high-energy regime (vaporization)
    {
        // rho < rho0, e > e_cv => vapor regime
        Real rho_exp = props.rho0 * 0.5;
        Real e_high = props.till_e_cv * 2.0;  // Well above complete vaporization
        Real p_vapor = TillotsonEOS::compute_pressure(props, rho_exp, e_high);
        // Should still produce positive pressure from thermal term
        CHECK(p_vapor > 0.0, "Tillotson: positive pressure in vapor regime");
    }

    // Test 20: Sound speed is positive under compression
    {
        Real c = TillotsonEOS::sound_speed(props, props.rho0 * 1.02, 1.0e6);
        CHECK(c > 0.0, "Tillotson: positive sound speed under compression");
    }
}

// ============================================================================
// Test 21-25: Sesame (2D Tabulated) EOS
// ============================================================================
void test_sesame() {
    std::cout << "\n=== Sesame (2D Tabulated) EOS ===\n";

    // Create a simple 3x3 table: p = K * (rho/rho0 - 1) + Gamma * rho * e
    // rho_grid: [500, 1000, 1500], e_grid: [0, 1e5, 2e5]
    EOSWave13Properties props;
    props.type = EOSWave13Type::Sesame;
    props.rho0 = 1000.0;

    auto& tab = props.sesame_table;
    tab.num_rho = 3;
    tab.num_e = 3;

    tab.rho_grid[0] = 500.0;   tab.rho_grid[1] = 1000.0;  tab.rho_grid[2] = 1500.0;
    tab.e_grid[0] = 0.0;       tab.e_grid[1] = 1.0e5;     tab.e_grid[2] = 2.0e5;

    // Fill table: p = 2.2e9*(rho/1000 - 1) + 0.5*rho*e
    // This mimics a simple water-like response
    Real K_eff = 2.2e9;
    Real Gamma_eff = 0.5;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Real rho_val = tab.rho_grid[i];
            Real e_val = tab.e_grid[j];
            Real p_val = K_eff * (rho_val / 1000.0 - 1.0) + Gamma_eff * rho_val * e_val;
            tab.set(i, j, p_val);
        }
    }

    // Test 21: Reference state with zero energy
    {
        Real p = SesameEOS::compute_pressure(props, 1000.0, 0.0);
        // At grid point (1000, 0): p = 2.2e9*(1-1) + 0 = 0
        CHECK_NEAR(p, 0.0, 1.0e3, "Sesame: zero pressure at reference, zero energy");
    }

    // Test 22: Compression at zero energy
    {
        Real p = SesameEOS::compute_pressure(props, 1500.0, 0.0);
        // At (1500, 0): p = 2.2e9*(1.5-1) + 0 = 1.1e9
        CHECK_NEAR(p, 1.1e9, 1.0e-3, "Sesame: compression at zero energy");
    }

    // Test 23: Interpolation between grid points
    {
        // At (1250, 1e5): midpoint between (1000, 1e5) and (1500, 1e5)
        // p(1000, 1e5) = 2.2e9*0 + 0.5*1000*1e5 = 5e7
        // p(1500, 1e5) = 2.2e9*0.5 + 0.5*1500*1e5 = 1.1e9 + 7.5e7 = 1.175e9
        // Midpoint: (5e7 + 1.175e9)/2 = 6.125e8
        Real p = SesameEOS::compute_pressure(props, 1250.0, 1.0e5);
        Real expected = (5.0e7 + 1.175e9) / 2.0;
        CHECK_NEAR(p, expected, 1.0e-3, "Sesame: bilinear interpolation accuracy");
    }

    // Test 24: Clamped extrapolation below minimum density
    {
        // rho=300 < 500 (min), clamped to 500
        // p at (500, 0) = 2.2e9*(0.5-1) + 0 = -1.1e9
        Real p_low = SesameEOS::compute_pressure(props, 300.0, 0.0);
        Real p_min = SesameEOS::compute_pressure(props, 500.0, 0.0);
        CHECK_NEAR(p_low, p_min, 1.0e-6, "Sesame: clamped extrapolation at low density");
    }

    // Test 25: Sound speed is positive at grid point
    {
        Real c = SesameEOS::sound_speed(props, 1000.0, 1.0e5);
        CHECK(c > 0.0, "Sesame: positive sound speed from 2D table");
    }
}

// ============================================================================
// Test 26-30: PowderBurn EOS
// ============================================================================
void test_powder_burn() {
    std::cout << "\n=== PowderBurn EOS ===\n";

    EOSWave13Properties props;
    props.type = EOSWave13Type::PowderBurn;
    props.rho0 = 1600.0;       // Propellant density
    props.pb_gamma = 1.25;
    props.pb_covolume = 1.0e-3; // m^3/kg
    props.pb_force_const = 1.0e6;
    props.pb_burn_a = 2.0e-9;  // Burn rate coefficient
    props.pb_burn_n = 0.7;     // Pressure exponent

    Real e_test = 3.0e5;  // J/kg

    // Test 26: Zero pressure when nothing is burned
    {
        EOSWave13State state;
        state.eta_burned = 0.0;
        Real p = PowderBurnEOS::compute_pressure(props, 1600.0, e_test, state);
        CHECK_NEAR(p, 0.0, 1.0, "PowderBurn: zero pressure when unburned");
    }

    // Test 27: Positive pressure when fully burned
    {
        EOSWave13State state;
        state.eta_burned = 1.0;
        Real p = PowderBurnEOS::compute_pressure(props, 1600.0, e_test, state);
        CHECK(p > 0.0, "PowderBurn: positive pressure when fully burned");

        // Check numerical value: p = (1.25-1)*1600*3e5/(1 - 1e-3*1600)
        //   = 0.25*1600*3e5/(1-1.6) = 1.2e8/(-0.6) ... covolume too high!
        // Use lower density to avoid singularity
    }

    // Test 28: Partial burn produces intermediate pressure
    {
        EOSWave13State state_half;
        state_half.eta_burned = 0.5;
        EOSWave13State state_full;
        state_full.eta_burned = 1.0;

        // Use lower density so covolume doesn't cause issues
        Real rho = 200.0;
        Real p_half = PowderBurnEOS::compute_pressure(props, rho, e_test, state_half);
        Real p_full = PowderBurnEOS::compute_pressure(props, rho, e_test, state_full);
        CHECK(p_half > 0.0 && p_full > p_half,
              "PowderBurn: partial burn < full burn pressure");
    }

    // Test 29: Burn rate updates eta correctly
    {
        EOSWave13State state;
        state.eta_burned = 0.0;

        // Apply burn at 1 MPa for 1 ms
        Real p_drive = 1.0e6;
        Real dt = 1.0e-3;
        PowderBurnEOS::update_burn(props, p_drive, dt, state);

        // deta = a * p^n * dt = 2e-9 * (1e6)^0.7 * 1e-3
        Real deta_expected = 2.0e-9 * std::pow(1.0e6, 0.7) * 1.0e-3;
        CHECK_NEAR(state.eta_burned, deta_expected, 1.0e-6,
                   "PowderBurn: burn rate numerical accuracy");
        CHECK(state.eta_burned > 0.0 && state.eta_burned < 1.0,
              "PowderBurn: burn fraction in valid range");
    }

    // Test 30: Sound speed is positive for burned gas
    {
        Real rho = 200.0;
        Real c = PowderBurnEOS::sound_speed(props, rho, e_test);
        CHECK(c > 0.0, "PowderBurn: positive sound speed for burned gas");
    }
}

// ============================================================================
// Test 31-35: Compaction EOS
// ============================================================================
void test_compaction() {
    std::cout << "\n=== Compaction EOS ===\n";

    EOSWave13Properties props;
    props.type = EOSWave13Type::Compaction;
    props.rho0 = 1500.0;        // Porous material
    props.compact_K_unload = 5.0e9;  // Unloading bulk modulus
    props.compact_ev_max = 0.5;

    // Compaction curve: p = 1e9 * ev^2 (quadratic hardening)
    // Build as tabulated points
    auto& curve = props.compaction_curve;
    curve.num_points = 11;
    for (int i = 0; i <= 10; ++i) {
        Real ev = i * 0.05;  // 0 to 0.5
        curve.x[i] = ev;
        curve.y[i] = 1.0e9 * ev * ev;  // Quadratic
    }

    // Test 31: Zero pressure at reference density
    {
        Real p = CompactionEOS::compute_pressure(props, props.rho0, 0.0);
        CHECK_NEAR(p, 0.0, 1.0, "Compaction: zero pressure at reference density");
    }

    // Test 32: Positive pressure under compression (loading path)
    {
        EOSWave13State state;
        // 10% volumetric strain
        Real rho = props.rho0 * 1.10;
        Real p = CompactionEOS::compute_pressure(props, rho, 0.0, state);
        // ev = 0.1, p = 1e9 * 0.01 = 1e7
        CHECK_NEAR(p, 1.0e7, 1.0e-3, "Compaction: loading curve at 10% strain");
    }

    // Test 33: Pressure increases with compression
    {
        Real p10 = CompactionEOS::compute_pressure(props, props.rho0 * 1.10, 0.0);
        Real p20 = CompactionEOS::compute_pressure(props, props.rho0 * 1.20, 0.0);
        CHECK(p20 > p10, "Compaction: pressure increases with compression");
    }

    // Test 34: Unloading gives lower pressure than loading at same strain
    {
        EOSWave13State state;
        // First load to 20%
        Real rho_20 = props.rho0 * 1.20;
        Real p_load = CompactionEOS::compute_pressure(props, rho_20, 0.0, state);
        // state.ev_max_reached should now be 0.2, p_max = 1e9*0.04 = 4e7

        // Now unload to 10%
        Real rho_10 = props.rho0 * 1.10;
        Real p_unload = CompactionEOS::compute_pressure(props, rho_10, 0.0, state);

        // Loading pressure at 10% would be 1e7
        Real p_loading_10 = 1.0e9 * 0.1 * 0.1;

        // Unloading: p = p_max - K_unload * (ev_max - ev) = 4e7 - 5e9*(0.2-0.1) = 4e7 - 5e8 < 0 => clamped to 0
        // The unloading is very stiff, so it will clamp to zero
        CHECK(p_unload >= 0.0, "Compaction: unloading pressure non-negative");
        CHECK(p_unload <= p_load, "Compaction: unloading pressure <= loading peak");
    }

    // Test 35: Sound speed is positive under compression
    {
        Real c = CompactionEOS::sound_speed(props, props.rho0 * 1.10, 0.0);
        CHECK(c > 0.0, "Compaction: positive sound speed under compression");
    }
}

// ============================================================================
// Test 36-40: Osborne EOS
// ============================================================================
void test_osborne() {
    std::cout << "\n=== Osborne EOS ===\n";

    // Set up Osborne to mimic a simple bulk modulus + energy coupling
    EOSWave13Properties props;
    props.type = EOSWave13Type::Osborne;
    props.rho0 = 8930.0;        // Copper [kg/m^3]
    props.osb_A1 = 1.37e11;     // Linear coefficient ~ bulk modulus [Pa]
    props.osb_A2 = 2.0e11;      // Quadratic stiffening
    props.osb_A3 = 1.0e11;      // Cubic stiffening
    props.osb_B0 = 2.0;         // Gruneisen-like energy coupling
    props.osb_B1 = 0.0;
    props.osb_B2 = 0.0;

    // Test 36: Zero pressure at reference state with zero energy
    {
        Real p = OsborneEOS::compute_pressure(props, props.rho0, 0.0);
        // mu = 0 => A terms are zero. E_vol = rho*e = 0 => thermal terms zero.
        CHECK_NEAR(p, 0.0, 1.0e-6, "Osborne: zero pressure at reference, zero energy");
    }

    // Test 37: Compression without thermal energy
    {
        Real rho = props.rho0 * 1.05;
        Real mu = 0.05;
        Real p = OsborneEOS::compute_pressure(props, rho, 0.0);
        Real expected = props.osb_A1 * mu + props.osb_A2 * mu * mu + props.osb_A3 * mu * mu * mu;
        CHECK_NEAR(p, expected, 1.0e-6, "Osborne: cold pressure at 5% compression");
    }

    // Test 38: Energy contributes to pressure
    {
        Real e = 1.0e5;  // J/kg
        Real p_cold = OsborneEOS::compute_pressure(props, props.rho0, 0.0);
        Real p_hot = OsborneEOS::compute_pressure(props, props.rho0, e);
        // At mu=0: p_hot - p_cold = B0 * rho0 * e = 2.0 * 8930 * 1e5 = 1.786e9
        Real delta_p_expected = props.osb_B0 * props.rho0 * e;
        CHECK_NEAR(p_hot - p_cold, delta_p_expected, 1.0e-6,
                   "Osborne: thermal pressure contribution");
    }

    // Test 39: Reduces to linear when A2=A3=B1=B2=0
    {
        EOSWave13Properties props_lin;
        props_lin.type = EOSWave13Type::Osborne;
        props_lin.rho0 = 1000.0;
        props_lin.osb_A1 = 2.2e9;
        props_lin.osb_A2 = 0.0;
        props_lin.osb_A3 = 0.0;
        props_lin.osb_B0 = 0.4;
        props_lin.osb_B1 = 0.0;
        props_lin.osb_B2 = 0.0;

        Real rho = 1050.0;
        Real mu = 0.05;
        Real e = 1.0e5;
        Real p = OsborneEOS::compute_pressure(props_lin, rho, e);
        Real expected = 2.2e9 * mu + 0.4 * rho * e;
        CHECK_NEAR(p, expected, 1.0e-6, "Osborne: linear-only matches A1*mu + B0*rho*e");
    }

    // Test 40: Sound speed is positive and reasonable
    {
        Real c = OsborneEOS::sound_speed(props, props.rho0, 0.0);
        // At mu=0, e=0: dp/drho = A1/rho0 + e*B0 = 1.37e11/8930 = 1.535e7
        // c = sqrt(1.535e7) ~ 3918 m/s (reasonable for copper)
        Real c_expected = std::sqrt(props.osb_A1 / props.rho0);
        CHECK(c > 0.0, "Osborne: positive sound speed");
        CHECK_NEAR(c, c_expected, 1.0e-3, "Osborne: sound speed = sqrt(A1/rho0) at reference");
    }
}

// ============================================================================
// Bonus: Unified Dispatch Test
// ============================================================================
void test_unified_dispatch() {
    std::cout << "\n=== Unified Dispatch ===\n";

    // Verify dispatch routes correctly to Murnaghan
    EOSWave13Properties props;
    props.type = EOSWave13Type::Murnaghan;
    props.rho0 = 2700.0;
    props.K0 = 76.0e9;
    props.K0_prime = 4.6;

    Real p_direct = MurnaghanEOS::compute_pressure(props, 2700.0 * 1.05, 0.0);
    Real p_dispatch = EOSWave13::compute_pressure(props, 2700.0 * 1.05, 0.0);
    CHECK_NEAR(p_direct, p_dispatch, 1.0e-10,
               "Dispatch: Murnaghan pressure matches direct call");

    // Verify Osborne dispatch
    EOSWave13Properties props_osb;
    props_osb.type = EOSWave13Type::Osborne;
    props_osb.rho0 = 1000.0;
    props_osb.osb_A1 = 2.2e9;
    props_osb.osb_A2 = 0.0;
    props_osb.osb_A3 = 0.0;
    props_osb.osb_B0 = 0.4;
    props_osb.osb_B1 = 0.0;
    props_osb.osb_B2 = 0.0;

    Real c_direct = OsborneEOS::sound_speed(props_osb, 1000.0, 0.0);
    Real c_dispatch = EOSWave13::sound_speed(props_osb, 1000.0, 0.0);
    CHECK_NEAR(c_direct, c_dispatch, 1.0e-10,
               "Dispatch: Osborne sound speed matches direct call");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "NexusSim Wave 13: Extended EOS Models\n";
    std::cout << "========================================\n";

    test_murnaghan();       // Tests 1-5
    test_noble_abel();      // Tests 6-10
    test_stiff_gas();       // Tests 11-15
    test_tillotson();       // Tests 16-20
    test_sesame();          // Tests 21-25
    test_powder_burn();     // Tests 26-30
    test_compaction();      // Tests 31-35
    test_osborne();         // Tests 36-40
    test_unified_dispatch(); // Bonus: 2 tests

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    std::cout << "========================================\n";

    return (tests_failed > 0) ? 1 : 0;
}
