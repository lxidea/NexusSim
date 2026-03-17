/**
 * @file failure_wave11_test.cpp
 * @brief Comprehensive test for 12 failure/damage models in failure_wave11.hpp
 *
 * Tests: JohnsonCookFailure, CockcroftLathamFailure, LemaitreCDMFailure,
 *        PuckFailure, FLDFailure, WilkinsFailure, TulerButcherFailure,
 *        MaxStressFailure, MaxStrainFailure, EnergyFailure,
 *        WierzbickiFailure, FabricFailure
 */

#include <nexussim/physics/failure/failure_wave11.hpp>
#include <iostream>
#include <cmath>

using namespace nxs;
using namespace nxs::physics;
using namespace nxs::physics::failure;

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
// Helper: reset a MaterialState to zeros (stress/strain cleared)
// ============================================================================
static MaterialState make_clean_state() {
    MaterialState ms;
    for (int i = 0; i < 6; ++i) {
        ms.stress[i] = 0.0;
        ms.strain[i] = 0.0;
    }
    ms.plastic_strain = 0.0;
    ms.effective_strain_rate = 0.0;
    ms.temperature = 293.15;
    return ms;
}

// ============================================================================
// 1. Johnson-Cook Failure
// ============================================================================
static void test_johnson_cook_failure() {
    std::cout << "\n=== Johnson-Cook Failure ===\n";

    FailureModelParameters base;
    JohnsonCookFailureParams jc;
    jc.d1 = 0.05;
    jc.d2 = 3.44;
    jc.d3 = -2.12;
    jc.d4 = 0.002;
    jc.d5 = 0.61;
    jc.eps_dot_ref = 1.0;
    jc.T_melt = 1800.0;
    jc.T_room = 293.15;

    JohnsonCookFailure model(base, jc);

    // (a) Default construction -- no crash
    CHECK(true, "JC: construction OK");

    // (b) Below threshold: small plastic strain increment => no failure
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = 500.0e6;  // Uniaxial tension
        ms.plastic_strain = 0.01;
        ms.temperature = 293.15;

        FailureState fs;
        // Set history[2] (previous plastic strain) to simulate first step
        fs.history[2] = 0.0;

        model.compute_damage(ms, fs, 1.0e-6, 0.001);

        CHECK(!fs.failed, "JC: small plastic strain => no failure");
        CHECK(fs.damage < 1.0, "JC: damage < 1.0 below threshold");
        CHECK(fs.history[0] > 0.0, "JC: damage accumulation started");
    }

    // (c) Above threshold: large cumulative damage => failure
    {
        FailureState fs;
        fs.history[0] = 0.99;  // Damage near 1.0
        fs.history[2] = 0.0;   // Previous plastic strain

        MaterialState ms = make_clean_state();
        ms.stress[0] = 500.0e6;  // Moderate triaxiality ~1/3
        // For uniaxial tension eta ~ 1/3, eps_f = [0.05 + 3.44*exp(-2.12*1/3)] * 1 * 1
        // ~ [0.05 + 3.44*exp(-0.707)] ~ [0.05 + 1.695] ~ 1.745
        // Need delta_eps_p / eps_f >= 0.01 to push damage over 1.0
        ms.plastic_strain = 0.10;  // Large plastic strain increment

        model.compute_damage(ms, fs, 1.0e-6, 0.001);

        CHECK(fs.failed, "JC: large cumulative damage => failure");
        CHECK(fs.failure_mode == 1, "JC: failure mode == 1");
    }

    // (d) History variable updates: triaxiality tracking
    {
        FailureState fs;
        MaterialState ms = make_clean_state();
        ms.stress[0] = 300.0e6;
        ms.stress[1] = 300.0e6;
        ms.stress[2] = 300.0e6;  // Hydrostatic => eta high
        ms.plastic_strain = 0.01;

        model.compute_damage(ms, fs, 1.0e-6, 0.001);

        // Hydrostatic stress: sigma_eq ~ 0, eta computed via guard
        CHECK(fs.history[2] > 0.0, "JC: history[2] stores previous plastic strain");
    }
}

// ============================================================================
// 2. Cockcroft-Latham Failure
// ============================================================================
static void test_cockcroft_latham_failure() {
    std::cout << "\n=== Cockcroft-Latham Failure ===\n";

    FailureModelParameters base;
    CockcroftLathamParams cl;
    cl.W_crit = 100.0e6;  // Critical plastic work

    CockcroftLathamFailure model(base, cl);

    CHECK(true, "CL: construction OK");

    // (b) Below threshold: small tensile stress and small plastic strain increment
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = 200.0e6;  // Tensile sigma_xx
        ms.plastic_strain = 0.01;

        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 0.001);

        // W = max(sigma1, 0) * delta_eps_p = 200e6 * 0.01 = 2e6
        CHECK(!fs.failed, "CL: below W_crit => no failure");
        CHECK_NEAR(fs.history[0], 200.0e6 * 0.01, 1.0, "CL: accumulated work correct");
    }

    // (c) Above threshold
    {
        FailureState fs;
        fs.history[0] = 99.0e6;  // Near critical
        fs.history[1] = 0.0;     // Previous plastic strain

        MaterialState ms = make_clean_state();
        ms.stress[0] = 500.0e6;
        ms.plastic_strain = 0.01;  // Increment: 500e6 * 0.01 = 5e6 => total 104e6 > 100e6

        model.compute_damage(ms, fs, 1.0e-6, 0.001);

        CHECK(fs.failed, "CL: above W_crit => failure");
        CHECK(fs.failure_mode == 1, "CL: failure mode == 1");
    }

    // (d) Compressive stress should not contribute (sigma1_pos = 0)
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = -500.0e6;  // All compressive
        ms.stress[1] = -300.0e6;
        ms.stress[2] = -200.0e6;
        ms.plastic_strain = 0.1;

        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 0.001);

        CHECK_NEAR(fs.history[0], 0.0, 1.0e-10, "CL: compressive => zero work accumulated");
    }
}

// ============================================================================
// 3. Lemaitre CDM Failure
// ============================================================================
static void test_lemaitre_cdm_failure() {
    std::cout << "\n=== Lemaitre CDM Failure ===\n";

    FailureModelParameters base;
    LemaitreCDMParams lem;
    lem.S = 1.0e6;
    lem.s_exp = 1.0;
    lem.E = 2.1e11;
    lem.D_crit = 0.99;
    lem.eps_D = 0.0;

    LemaitreCDMFailure model(base, lem);

    CHECK(true, "Lemaitre: construction OK");

    // (b) Below threshold: small plastic strain
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = 300.0e6;
        ms.plastic_strain = 0.001;

        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 0.001);

        CHECK(!fs.failed, "Lemaitre: small strain => no failure");
        CHECK(fs.damage < 1.0, "Lemaitre: damage < 1");
        CHECK(fs.history[0] >= 0.0, "Lemaitre: damage non-negative");
    }

    // (c) Above threshold: push damage past D_crit
    {
        FailureState fs;
        fs.history[0] = 0.99;  // Already at D_crit
        fs.history[1] = 0.0;

        MaterialState ms = make_clean_state();
        ms.stress[0] = 500.0e6;
        ms.plastic_strain = 0.1;

        model.compute_damage(ms, fs, 1.0e-6, 0.001);

        CHECK(fs.failed, "Lemaitre: D >= D_crit => failure");
        CHECK(fs.failure_mode == 1, "Lemaitre: failure mode == 1");
        CHECK_NEAR(fs.damage, 1.0, 1.0e-10, "Lemaitre: damage = 1 when D >= D_crit");
    }

    // (d) Stress degradation
    {
        FailureState fs;
        fs.damage = 0.5;
        Real sigma[6] = {100.0, 200.0, 300.0, 50.0, 60.0, 70.0};
        model.degrade_stress(sigma, fs);
        CHECK_NEAR(sigma[0], 50.0, 1.0e-8, "Lemaitre: degrade_stress factor (1-D)");
        CHECK_NEAR(sigma[3], 25.0, 1.0e-8, "Lemaitre: degrade_stress shear");
    }

    // (e) Damage threshold strain: no damage below eps_D
    {
        LemaitreCDMParams lem2;
        lem2.S = 1.0e6;
        lem2.s_exp = 1.0;
        lem2.E = 2.1e11;
        lem2.D_crit = 0.99;
        lem2.eps_D = 0.1;  // Only damage after 10% plastic strain

        LemaitreCDMFailure model2(base, lem2);

        MaterialState ms = make_clean_state();
        ms.stress[0] = 500.0e6;
        ms.plastic_strain = 0.05;  // Below eps_D

        FailureState fs;
        model2.compute_damage(ms, fs, 1.0e-6, 0.001);

        CHECK_NEAR(fs.history[0], 0.0, 1.0e-10, "Lemaitre: below eps_D => no damage");
    }
}

// ============================================================================
// 4. Puck Failure
// ============================================================================
static void test_puck_failure() {
    std::cout << "\n=== Puck Failure ===\n";

    FailureModelParameters base;
    PuckFailureParams pk;
    pk.R_para_t = 1.5e9;
    pk.R_para_c = 1.2e9;
    pk.R_perp_t = 50.0e6;
    pk.R_perp_c = 200.0e6;
    pk.R_perp_para = 75.0e6;
    pk.eps_1t = 0.02;

    PuckFailure model(base, pk);

    CHECK(true, "Puck: construction OK");

    // (b) Below threshold: moderate fiber strain
    {
        MaterialState ms = make_clean_state();
        ms.strain[0] = 0.01;  // Below eps_1t = 0.02
        ms.stress[0] = 500.0e6;
        ms.stress[1] = 20.0e6;  // Below R_perp_t = 50 MPa

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(!fs.failed, "Puck: below all limits => no failure");
        CHECK(fs.history[0] < 1.0, "Puck: FF tension index < 1");
    }

    // (c) Fiber tension failure: strain exceeds eps_1t
    {
        MaterialState ms = make_clean_state();
        ms.strain[0] = 0.025;  // Above eps_1t = 0.02
        ms.stress[0] = 2.0e9;  // Tensile

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "Puck: fiber tension failure");
        CHECK(fs.failure_mode == 1, "Puck: failure mode = 1 (FF tension)");
        CHECK(fs.history[4] >= 1.0, "Puck: FF tension damage flag set");
    }

    // (d) IFF failure: high transverse stress
    {
        MaterialState ms = make_clean_state();
        ms.strain[0] = 0.001;   // Small fiber strain (no FF)
        ms.stress[0] = 100.0e6; // Low fiber stress
        ms.stress[1] = 60.0e6;  // Above R_perp_t = 50 MPa => (60/50)^2 = 1.44 > 1

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "Puck: IFF failure from transverse stress");
        CHECK(fs.failure_mode == 3, "Puck: failure mode = 3 (IFF)");
        CHECK(fs.history[6] >= 1.0, "Puck: IFF damage flag set");
    }

    // (e) Fiber compression failure
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = -1.3e9;  // Exceeds R_para_c = 1.2 GPa
        ms.strain[0] = -0.001;  // Negative => no FF tension

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "Puck: fiber compression failure");
        CHECK(fs.failure_mode == 2, "Puck: failure mode = 2 (FF compression)");
        CHECK(fs.history[5] >= 1.0, "Puck: FF compression damage flag set");
    }
}

// ============================================================================
// 5. FLD Failure
// ============================================================================
static void test_fld_failure() {
    std::cout << "\n=== FLD Failure ===\n";

    FailureModelParameters base;
    FLDFailureParams fld;
    // Build a simple V-shaped FLD curve: minor strain [-0.3, 0, 0.3] => major limit [0.6, 0.3, 0.6]
    fld.fld_curve.add_point(-0.3, 0.6);
    fld.fld_curve.add_point(0.0, 0.3);
    fld.fld_curve.add_point(0.3, 0.6);

    FLDFailure model(base, fld);

    CHECK(true, "FLD: construction OK");

    // (b) Below FLD curve: safe forming state
    {
        MaterialState ms = make_clean_state();
        ms.strain[0] = 0.15;  // eps_xx
        ms.strain[1] = 0.05;  // eps_yy
        ms.strain[3] = 0.0;   // gamma_xy
        // Major = 0.15, minor = 0.05 => FLD limit at minor=0.05 ~ 0.35 => safe

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(!fs.failed, "FLD: below curve => no failure");
        CHECK(fs.damage < 1.0, "FLD: damage < 1 when safe");
    }

    // (c) Above FLD curve: failure
    {
        MaterialState ms = make_clean_state();
        ms.strain[0] = 0.50;  // eps_xx (high)
        ms.strain[1] = 0.0;   // eps_yy
        ms.strain[3] = 0.0;   // gamma_xy
        // Major = 0.50, minor = 0.0 => FLD limit at minor=0 is 0.3 => ratio=0.5/0.3 > 1

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "FLD: above curve => failure");
        CHECK(fs.failure_mode == 1, "FLD: failure mode == 1");
    }

    // (d) History stores principal strains
    {
        MaterialState ms = make_clean_state();
        ms.strain[0] = 0.20;
        ms.strain[1] = 0.10;
        ms.strain[3] = 0.0;

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK_NEAR(fs.history[1], 0.20, 1.0e-10, "FLD: history[1] = major strain");
        CHECK_NEAR(fs.history[2], 0.10, 1.0e-10, "FLD: history[2] = minor strain");
    }
}

// ============================================================================
// 6. Wilkins Failure
// ============================================================================
static void test_wilkins_failure() {
    std::cout << "\n=== Wilkins Failure ===\n";

    FailureModelParameters base;
    WilkinsFailureParams wk;
    wk.a_exp = 1.0;
    wk.b_exp = 1.0;
    wk.f_crit = 100.0e6;

    WilkinsFailure model(base, wk);

    CHECK(true, "Wilkins: construction OK");

    // (b) Below threshold: small plastic strain
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = 400.0e6;  // Tensile principal
        ms.stress[1] = 200.0e6;
        ms.stress[2] = 100.0e6;
        ms.plastic_strain = 0.001;

        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 0.001);

        CHECK(!fs.failed, "Wilkins: small strain => no failure");
        CHECK(fs.history[0] > 0.0, "Wilkins: integral accumulating");
    }

    // (c) Above threshold
    {
        FailureState fs;
        fs.history[0] = 99.0e6;   // Near f_crit
        fs.history[1] = 0.0;      // Previous plastic strain

        MaterialState ms = make_clean_state();
        ms.stress[0] = 500.0e6;
        ms.stress[1] = 200.0e6;
        ms.stress[2] = 100.0e6;
        ms.plastic_strain = 0.01;
        // sigma1=500e6, A = min(200/500, 100/500) = 0.2
        // Integrand = 500e6^1 * (2-0.2)^1 * 0.01 = 500e6 * 1.8 * 0.01 = 9e6
        // Total = 99e6 + 9e6 = 108e6 > 100e6

        model.compute_damage(ms, fs, 1.0e-6, 0.001);

        CHECK(fs.failed, "Wilkins: above f_crit => failure");
        CHECK(fs.failure_mode == 1, "Wilkins: failure mode == 1");
    }

    // (d) Compressive principal stress: sigma1_pos = 0 => no accumulation
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = -100.0e6;
        ms.stress[1] = -200.0e6;
        ms.stress[2] = -300.0e6;
        ms.plastic_strain = 0.1;

        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 0.001);

        CHECK_NEAR(fs.history[0], 0.0, 1.0e-10, "Wilkins: compressive => no accumulation");
    }
}

// ============================================================================
// 7. Tuler-Butcher Failure
// ============================================================================
static void test_tuler_butcher_failure() {
    std::cout << "\n=== Tuler-Butcher Failure ===\n";

    FailureModelParameters base;
    TulerButcherParams tb;
    tb.sigma_spall = 1.0e9;
    tb.lambda = 2.0;
    tb.K_crit = 1.0e3;

    TulerButcherFailure model(base, tb);

    CHECK(true, "TB: construction OK");

    // (b) Below spall threshold: stress below sigma_spall
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = 0.5e9;  // Below 1.0 GPa spall threshold

        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 0.001);

        CHECK(!fs.failed, "TB: below spall stress => no failure");
        CHECK_NEAR(fs.history[0], 0.0, 1.0e-10, "TB: no accumulation below threshold");
    }

    // (c) Above spall threshold
    {
        FailureState fs;
        fs.history[0] = 999.0;  // Near K_crit = 1000

        MaterialState ms = make_clean_state();
        ms.stress[0] = 2.0e9;  // 1 GPa overstress
        // Increment = (1e9)^2 * dt = 1e18 * 1e-6 = 1e12 >> remaining
        Real dt = 1.0e-6;

        model.compute_damage(ms, fs, dt, 0.001);

        CHECK(fs.failed, "TB: above K_crit => spall failure");
        CHECK(fs.failure_mode == 1, "TB: failure mode == 1");
    }

    // (d) Time-step dependence: impulse integral scales with dt
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = 1.5e9;  // 0.5 GPa overstress
        Real dt = 1.0e-7;

        FailureState fs;
        model.compute_damage(ms, fs, dt, 0.001);

        // Expected: (0.5e9)^2 * 1e-7 = 2.5e10
        Real expected = std::pow(0.5e9, 2.0) * dt;
        CHECK_NEAR(fs.history[0], expected, expected * 1.0e-6, "TB: impulse scales with dt");
    }
}

// ============================================================================
// 8. Maximum Stress Failure
// ============================================================================
static void test_max_stress_failure() {
    std::cout << "\n=== MaxStress Failure ===\n";

    FailureModelParameters base;
    MaxStressFailureParams ms_params;
    ms_params.sigma_limit[0] = 500.0e6;  // sigma_xx limit
    ms_params.sigma_limit[1] = 400.0e6;  // sigma_yy limit
    ms_params.sigma_limit[2] = 300.0e6;  // sigma_zz limit
    ms_params.sigma_limit[3] = 200.0e6;  // tau_xy limit
    ms_params.sigma_limit[4] = 150.0e6;  // tau_yz limit
    ms_params.sigma_limit[5] = 150.0e6;  // tau_xz limit

    MaxStressFailure model(base, ms_params);

    CHECK(true, "MaxStress: construction OK");

    // (b) Below all limits
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = 200.0e6;
        ms.stress[1] = 100.0e6;
        ms.stress[2] = 50.0e6;

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(!fs.failed, "MaxStress: below limits => no failure");
        // Ratio for component 0: 200/500 = 0.4
        CHECK_NEAR(fs.history[0], 0.4, 1.0e-6, "MaxStress: history[0] = stress ratio");
    }

    // (c) Exceeding sigma_yy limit
    {
        MaterialState ms = make_clean_state();
        ms.stress[1] = 450.0e6;  // Exceeds sigma_yy limit of 400 MPa

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "MaxStress: sigma_yy > limit => failure");
        CHECK(fs.failure_mode == 2, "MaxStress: failure mode = 2 (yy component, 1-indexed)");
    }

    // (d) Negative stress also checked (absolute value)
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = -600.0e6;  // |sigma_xx| > 500 MPa limit

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "MaxStress: |negative stress| > limit => failure");
        CHECK(fs.failure_mode == 1, "MaxStress: failure mode = 1 (xx component)");
    }
}

// ============================================================================
// 9. Maximum Strain Failure
// ============================================================================
static void test_max_strain_failure() {
    std::cout << "\n=== MaxStrain Failure ===\n";

    FailureModelParameters base;
    MaxStrainFailureParams ms_params;
    ms_params.eps_limit[0] = 0.10;  // eps_xx
    ms_params.eps_limit[1] = 0.08;  // eps_yy
    ms_params.eps_limit[2] = 0.05;  // eps_zz
    ms_params.eps_limit[3] = 0.15;  // gamma_xy
    ms_params.eps_limit[4] = 0.12;  // gamma_yz
    ms_params.eps_limit[5] = 0.12;  // gamma_xz

    MaxStrainFailure model(base, ms_params);

    CHECK(true, "MaxStrain: construction OK");

    // (b) Below all limits
    {
        MaterialState ms = make_clean_state();
        ms.strain[0] = 0.03;
        ms.strain[1] = 0.02;
        ms.strain[2] = 0.01;

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(!fs.failed, "MaxStrain: below limits => no failure");
        CHECK_NEAR(fs.history[0], 0.03 / 0.10, 1.0e-6, "MaxStrain: history[0] = strain ratio");
    }

    // (c) Exceeding eps_zz limit
    {
        MaterialState ms = make_clean_state();
        ms.strain[2] = 0.06;  // Exceeds eps_zz limit of 0.05

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "MaxStrain: eps_zz > limit => failure");
        CHECK(fs.failure_mode == 3, "MaxStrain: failure mode = 3 (zz component, 1-indexed)");
    }

    // (d) Negative strain also checked (absolute value)
    {
        MaterialState ms = make_clean_state();
        ms.strain[1] = -0.09;  // |eps_yy| > 0.08 limit

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "MaxStrain: |negative strain| > limit => failure");
        CHECK(fs.failure_mode == 2, "MaxStrain: failure mode = 2 (yy component)");
    }
}

// ============================================================================
// 10. Energy Failure
// ============================================================================
static void test_energy_failure() {
    std::cout << "\n=== Energy Failure ===\n";

    FailureModelParameters base;
    EnergyFailureParams en;
    en.W_crit = 1.0e6;  // 1 MJ/m^3

    EnergyFailure model(base, en);

    CHECK(true, "Energy: construction OK");

    // (b) Below threshold
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = 100.0e6;
        ms.strain[0] = 0.001;
        // W = 0.5 * 100e6 * 0.001 = 5e4 < 1e6

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(!fs.failed, "Energy: below W_crit => no failure");
        CHECK_NEAR(fs.history[0], 0.5 * 100.0e6 * 0.001, 1.0, "Energy: W = 0.5*sigma*epsilon");
    }

    // (c) Above threshold
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = 500.0e6;
        ms.strain[0] = 0.01;
        // W = 0.5 * 500e6 * 0.01 = 2.5e6 > 1e6

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "Energy: above W_crit => failure");
        CHECK(fs.failure_mode == 1, "Energy: failure mode == 1");
    }

    // (d) History tracks maximum W
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = 300.0e6;
        ms.strain[0] = 0.001;
        // W = 0.5 * 300e6 * 0.001 = 1.5e5

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        Real W_first = fs.history[0];

        // Second call with lower energy: max should remain
        ms.stress[0] = 50.0e6;
        ms.strain[0] = 0.0001;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK_NEAR(fs.history[0], W_first, 1.0e-6, "Energy: history[0] tracks max W");
    }

    // (e) Shear contribution
    {
        MaterialState ms = make_clean_state();
        ms.stress[3] = 300.0e6;  // tau_xy
        ms.strain[3] = 0.02;     // gamma_xy
        // W = 0.5 * 300e6 * 0.02 = 3e6 > 1e6

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "Energy: shear energy above W_crit => failure");
    }
}

// ============================================================================
// 11. Wierzbicki (Modified Mohr-Coulomb) Failure
// ============================================================================
static void test_wierzbicki_failure() {
    std::cout << "\n=== Wierzbicki Failure ===\n";

    FailureModelParameters base;
    WierzbickiFailureParams wb;
    wb.C1 = 0.5;
    wb.C2 = 600.0e6;
    wb.C_theta_s = 1.0;
    wb.C_theta_ax = 1.0;
    wb.sigma_bar_0 = 500.0e6;
    wb.c_friction = 0.1;

    WierzbickiFailure model(base, wb);

    CHECK(true, "Wierzbicki: construction OK");

    // (b) Below threshold: small plastic strain
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = 500.0e6;  // Uniaxial tension
        ms.plastic_strain = 0.001;

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.001);

        CHECK(!fs.failed, "Wierzbicki: small strain => no failure");
        CHECK(fs.history[0] > 0.0, "Wierzbicki: damage accumulating");
        CHECK(fs.damage < 1.0, "Wierzbicki: damage < 1");
    }

    // (c) Above threshold: large cumulative damage
    {
        FailureState fs;
        fs.history[0] = 0.999;  // Near 1.0
        fs.history[3] = 0.0;    // Previous plastic strain

        MaterialState ms = make_clean_state();
        ms.stress[0] = 500.0e6;
        ms.plastic_strain = 10.0;  // Massive increment to ensure failure

        model.compute_damage(ms, fs, 0.0, 0.001);

        CHECK(fs.failed, "Wierzbicki: large damage => failure");
        CHECK(fs.failure_mode == 1, "Wierzbicki: failure mode == 1");
    }

    // (d) History stores triaxiality and Lode parameter
    {
        MaterialState ms = make_clean_state();
        ms.stress[0] = 400.0e6;
        ms.stress[1] = 200.0e6;
        ms.stress[2] = 100.0e6;
        ms.plastic_strain = 0.01;

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.001);

        // Triaxiality eta = sigma_m / sigma_eq (should be positive for tension)
        CHECK(fs.history[1] > 0.0, "Wierzbicki: positive triaxiality in tension");
        // Lode parameter stored in history[2]
        CHECK(std::abs(fs.history[2]) <= 1.0 + 1.0e-6,
              "Wierzbicki: Lode parameter in [-1, 1]");
    }
}

// ============================================================================
// 12. Fabric Failure
// ============================================================================
static void test_fabric_failure() {
    std::cout << "\n=== Fabric Failure ===\n";

    FailureModelParameters base;
    FabricFailureParams fab;
    fab.eps_1f = 0.05;   // Warp failure strain
    fab.eps_2f = 0.05;   // Weft failure strain
    fab.gamma_12f = 0.10; // Shear failure strain

    FabricFailure model(base, fab);

    CHECK(true, "Fabric: construction OK");

    // (b) Below threshold: small strains
    {
        MaterialState ms = make_clean_state();
        ms.strain[0] = 0.02;  // Warp
        ms.strain[1] = 0.01;  // Weft
        ms.strain[3] = 0.02;  // Shear
        // FI = (0.02/0.05)^2 + (0.01/0.05)^2 + (0.02/0.10)^2
        //    = 0.16 + 0.04 + 0.04 = 0.24 < 1

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(!fs.failed, "Fabric: below FI=1 => no failure");
        CHECK_NEAR(fs.history[0], 0.24, 1.0e-6, "Fabric: FI = 0.24");
    }

    // (c) Warp-dominant failure
    {
        MaterialState ms = make_clean_state();
        ms.strain[0] = 0.06;  // Warp exceeds 0.05
        ms.strain[1] = 0.0;
        ms.strain[3] = 0.0;
        // FI = (0.06/0.05)^2 = 1.44 > 1

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "Fabric: warp failure");
        CHECK(fs.failure_mode == 1, "Fabric: failure mode = 1 (warp)");
    }

    // (d) Weft-dominant failure
    {
        MaterialState ms = make_clean_state();
        ms.strain[0] = 0.0;
        ms.strain[1] = 0.06;  // Weft exceeds 0.05
        ms.strain[3] = 0.0;

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "Fabric: weft failure");
        CHECK(fs.failure_mode == 2, "Fabric: failure mode = 2 (weft)");
    }

    // (e) Shear-dominant failure
    {
        MaterialState ms = make_clean_state();
        ms.strain[0] = 0.0;
        ms.strain[1] = 0.0;
        ms.strain[3] = 0.12;  // Shear exceeds 0.10

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "Fabric: shear failure");
        CHECK(fs.failure_mode == 3, "Fabric: failure mode = 3 (shear)");
    }

    // (f) Stress degradation: warp failure degrades sigma[0] and sigma[3]
    {
        FailureState fs;
        fs.failed = true;
        fs.failure_mode = 1;

        Real sigma[6] = {100.0, 200.0, 300.0, 50.0, 60.0, 70.0};
        model.degrade_stress(sigma, fs);

        CHECK_NEAR(sigma[0], 0.0, 1.0e-10, "Fabric: warp failure => sigma[0]=0");
        CHECK_NEAR(sigma[3], 0.0, 1.0e-10, "Fabric: warp failure => sigma[3]=0");
        CHECK_NEAR(sigma[1], 200.0, 1.0e-10, "Fabric: warp failure => sigma[1] preserved");
    }

    // (g) History stores component ratios
    {
        MaterialState ms = make_clean_state();
        ms.strain[0] = 0.025;  // r1 = 0.025/0.05 = 0.5
        ms.strain[1] = 0.030;  // r2 = 0.030/0.05 = 0.6
        ms.strain[3] = 0.040;  // r12 = 0.040/0.10 = 0.4

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK_NEAR(fs.history[1], 0.5, 1.0e-6, "Fabric: history[1] = warp ratio");
        CHECK_NEAR(fs.history[2], 0.6, 1.0e-6, "Fabric: history[2] = weft ratio");
        CHECK_NEAR(fs.history[3], 0.4, 1.0e-6, "Fabric: history[3] = shear ratio");
    }
}

// ============================================================================
// Bonus: FailureModelTypeExt to_string coverage
// ============================================================================
static void test_type_ext_to_string() {
    std::cout << "\n=== FailureModelTypeExt to_string ===\n";

    CHECK(std::string(to_string(FailureModelTypeExt::JohnsonCook)) == "JohnsonCook",
          "to_string JohnsonCook");
    CHECK(std::string(to_string(FailureModelTypeExt::CockcroftLatham)) == "CockcroftLatham",
          "to_string CockcroftLatham");
    CHECK(std::string(to_string(FailureModelTypeExt::Fabric)) == "Fabric",
          "to_string Fabric");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== Wave 11 Failure Models Test ===\n";

    test_johnson_cook_failure();
    test_cockcroft_latham_failure();
    test_lemaitre_cdm_failure();
    test_puck_failure();
    test_fld_failure();
    test_wilkins_failure();
    test_tuler_butcher_failure();
    test_max_stress_failure();
    test_max_strain_failure();
    test_energy_failure();
    test_wierzbicki_failure();
    test_fabric_failure();
    test_type_ext_to_string();

    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";
    std::cout << "Total:  " << (tests_passed + tests_failed) << "\n";

    return (tests_failed > 0) ? 1 : 0;
}
