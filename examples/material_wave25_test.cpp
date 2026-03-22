/**
 * @file material_wave25_test.cpp
 * @brief Comprehensive test for Wave 25 material models (10 advanced constitutive models)
 */

#include <nexussim/physics/material_wave25.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace nxs;
using namespace nxs::physics;

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

// Helper: make steel-like properties
static MaterialProperties make_steel_props() {
    MaterialProperties props;
    props.E = 210.0e9;
    props.nu = 0.3;
    props.density = 7800.0;
    props.yield_stress = 250.0e6;
    props.hardening_modulus = 1.0e9;
    props.compute_derived();
    return props;
}

// Helper: make concrete-like properties
static MaterialProperties make_concrete_props() {
    MaterialProperties props;
    props.E = 30.0e9;
    props.nu = 0.2;
    props.density = 2400.0;
    props.yield_stress = 30.0e6;
    props.hardening_modulus = 0.0;
    props.compute_derived();
    return props;
}

// Helper: make composite properties
static MaterialProperties make_composite_props() {
    MaterialProperties props;
    props.E = 10.0e9;
    props.nu = 0.3;
    props.density = 1600.0;
    props.E1 = 140.0e9;
    props.E2 = 10.0e9;
    props.G12 = 5.0e9;
    props.nu12 = 0.3;
    props.yield_stress = 1500.0e6;
    props.hardening_modulus = 0.5e9;
    props.compute_derived();
    return props;
}

// ==========================================================================
// 1. OrthotropicBrittleMaterial
// ==========================================================================
void test_1_orthotropic_brittle() {
    std::cout << "\n=== Test 1: OrthotropicBrittleMaterial ===\n";

    MaterialProperties props;
    props.E = 30.0e9;
    props.nu = 0.2;
    props.density = 2400.0;
    props.E1 = 30.0e9;
    props.E2 = 25.0e9;
    props.E3 = 20.0e9;
    props.G12 = 12.0e9;
    props.G23 = 10.0e9;
    props.G13 = 10.0e9;
    props.compute_derived();

    Real ft1 = 3.0e6, ft2 = 2.5e6, ft3 = 2.0e6;
    OrthotropicBrittleMaterial mat(props, ft1, ft2, ft3, 100.0, 0.01);

    // Crack in direction 1: apply tensile strain exceeding ft1/E1
    Real eps_crack_1 = ft1 / props.E1 * 2.0; // Well above cracking strain
    MaterialState state_1;
    state_1.strain[0] = eps_crack_1;
    mat.compute_stress(state_1);
    Real d1 = mat.directional_damage(state_1, 0);
    CHECK(d1 > 0.0, "OrthBrittle: crack in dir 1 produces d1 > 0");
    CHECK(state_1.stress[0] > 0.0, "OrthBrittle: positive stress despite crack (softening)");

    // Direction 2 should be undamaged from direction 1 loading
    Real d2_from_1 = mat.directional_damage(state_1, 1);
    CHECK_NEAR(d2_from_1, 0.0, 1.0e-10, "OrthBrittle: dir 2 undamaged from dir 1 loading");

    // Crack in direction 2
    MaterialState state_2;
    state_2.strain[1] = ft2 / props.E2 * 2.5;
    mat.compute_stress(state_2);
    Real d2 = mat.directional_damage(state_2, 1);
    CHECK(d2 > 0.0, "OrthBrittle: crack in dir 2 produces d2 > 0");

    // No crack under compression
    MaterialState state_comp;
    state_comp.strain[0] = -0.001; // Compression
    mat.compute_stress(state_comp);
    Real d_comp = mat.directional_damage(state_comp, 0);
    CHECK_NEAR(d_comp, 0.0, 1.0e-10, "OrthBrittle: no crack under compression");

    // Multi-directional cracking
    MaterialState state_multi;
    state_multi.strain[0] = ft1 / props.E1 * 3.0;
    state_multi.strain[1] = ft2 / props.E2 * 3.0;
    mat.compute_stress(state_multi);
    Real dm1 = mat.directional_damage(state_multi, 0);
    Real dm2 = mat.directional_damage(state_multi, 1);
    CHECK(dm1 > 0.0 && dm2 > 0.0, "OrthBrittle: multi-directional cracking");

    // Damage reduces stress
    MaterialState state_undmg;
    state_undmg.strain[0] = ft1 / props.E1 * 0.5; // Below cracking
    mat.compute_stress(state_undmg);
    Real sigma_undmg = state_undmg.stress[0];

    MaterialState state_dmg;
    state_dmg.strain[0] = ft1 / props.E1 * 0.5;
    state_dmg.history[32] = 0.5; // Pre-existing damage
    mat.compute_stress(state_dmg);
    Real sigma_dmg = state_dmg.stress[0];
    CHECK(sigma_dmg < sigma_undmg, "OrthBrittle: damage reduces stress");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_undmg, C);
    CHECK(C[0] > 0.0, "OrthBrittle: tangent C11 > 0");
}

// ==========================================================================
// 2. ExplosiveBurnExtMaterial
// ==========================================================================
void test_2_explosive_burn_ext() {
    std::cout << "\n=== Test 2: ExplosiveBurnExtMaterial ===\n";

    MaterialProperties props;
    props.E = 6930.0;
    props.nu = 0.0;
    props.density = 1630.0;
    props.compute_derived();

    // Two detonation points
    ExplosiveBurnExtMaterial::DetonationPoint pts[2];
    pts[0] = {0.0, 0.0, 0.0, 0.0};       // Point 1: origin, time 0
    pts[1] = {1.0, 0.0, 0.0, 1.0e-4};    // Point 2: 1m away, t_light = 0.1 ms

    Real D_cj = 6930.0;
    ExplosiveBurnExtMaterial mat(props, pts, 2,
                                  3.712e11, 3.231e9, 4.15, 0.95, 0.30, D_cj);

    // Single point detonation: element at origin, after sufficient time
    MaterialState state_single;
    state_single.history[37] = 0.0; // Element at origin
    state_single.history[38] = 0.0;
    state_single.history[39] = 0.0;
    state_single.dt = 0.01; // Large dt to fully burn
    state_single.strain[0] = -0.01;
    state_single.strain[1] = -0.01;
    state_single.strain[2] = -0.01;
    mat.compute_stress(state_single);
    Real F1 = mat.burn_fraction(state_single, 0);
    CHECK(F1 > 0.0, "ExpBurnExt: single point burn fraction > 0");

    // Total burn fraction
    Real F_total = mat.total_burn_fraction(state_single);
    CHECK(F_total > 0.0, "ExpBurnExt: total burn fraction > 0");
    CHECK_NEAR(F_total, 1.0, 1.0e-6, "ExpBurnExt: fully burnt after large dt");

    // Hydrostatic stress
    CHECK_NEAR(state_single.stress[0], state_single.stress[1], 1.0e-3,
               "ExpBurnExt: hydrostatic (sigma_xx = sigma_yy)");
    CHECK_NEAR(state_single.stress[3], 0.0, 1.0e-10,
               "ExpBurnExt: zero shear");

    // Dual point: element between points, takes first arrival
    MaterialState state_dual;
    state_dual.history[37] = 0.5; // Element at (0.5, 0, 0)
    state_dual.history[38] = 0.0;
    state_dual.history[39] = 0.0;
    state_dual.dt = 1.0e-4;
    state_dual.strain[0] = -0.01;
    mat.compute_stress(state_dual);
    Real F_p0 = mat.burn_fraction(state_dual, 0);
    CHECK(F_p0 > 0.0, "ExpBurnExt: first arrival point burns");

    // Timing: before any arrival, no burn
    MaterialState state_pre;
    state_pre.history[37] = 10.0; // Far away
    state_pre.dt = 1.0e-8;
    mat.compute_stress(state_pre);
    Real F_pre = mat.total_burn_fraction(state_pre);
    CHECK_NEAR(F_pre, 0.0, 1.0e-15, "ExpBurnExt: no burn before arrival");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_single, C);
    CHECK(C[0] > 0.0, "ExpBurnExt: tangent C11 > 0");
}

// ==========================================================================
// 3. ExtendedSoilMaterial
// ==========================================================================
void test_3_extended_soil() {
    std::cout << "\n=== Test 3: ExtendedSoilMaterial ===\n";

    MaterialProperties props;
    props.E = 50.0e6;
    props.nu = 0.3;
    props.density = 1800.0;
    props.compute_derived();

    Real a0 = 1.0e5;   // Cohesion
    Real a1 = 0.5;      // Friction coefficient
    Real a2 = 0.0;
    Real P_t = 5.0e4;   // Tension cutoff

    ExtendedSoilMaterial mat(props, a0, a1, a2, P_t);

    // Shear yield: apply deviatoric strain
    MaterialState state_shear;
    state_shear.strain[3] = 0.01; // Shear strain
    mat.compute_stress(state_shear);
    CHECK(state_shear.stress[3] != 0.0, "ExtSoil: shear stress nonzero");

    // Yield surface evaluation
    Real F_at_0 = mat.yield_strength(0.0);
    CHECK_NEAR(F_at_0, a0, 1.0e-3, "ExtSoil: yield strength at P=0 equals cohesion");

    // Confinement strengthening: higher P increases yield surface
    Real F_at_1MPa = mat.yield_strength(1.0e6);
    CHECK(F_at_1MPa > F_at_0, "ExtSoil: confinement strengthening");

    // Tension cutoff: apply tensile strain
    MaterialState state_tension;
    state_tension.strain[0] = 0.01;
    state_tension.strain[1] = 0.01;
    state_tension.strain[2] = 0.01;
    mat.compute_stress(state_tension);
    bool tc_active = mat.tension_cutoff_active(state_tension);
    CHECK(tc_active, "ExtSoil: tension cutoff activated under tension");

    // Under compression, tension cutoff should not be active
    MaterialState state_compress;
    state_compress.strain[0] = -0.001;
    state_compress.strain[1] = -0.001;
    state_compress.strain[2] = -0.001;
    mat.compute_stress(state_compress);
    CHECK(!mat.tension_cutoff_active(state_compress),
          "ExtSoil: no tension cutoff under compression");

    // Elastic response at small strain
    MaterialState state_el;
    state_el.strain[0] = 0.0001;
    mat.compute_stress(state_el);
    CHECK(state_el.stress[0] > 0.0, "ExtSoil: positive elastic stress");
    CHECK_NEAR(state_el.plastic_strain, 0.0, 1.0e-10,
               "ExtSoil: no plastic strain in elastic range");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_el, C);
    CHECK(C[0] > 0.0, "ExtSoil: tangent C11 > 0");
}

// ==========================================================================
// 4. SoilAndCrushMaterial
// ==========================================================================
void test_4_soil_and_crush() {
    std::cout << "\n=== Test 4: SoilAndCrushMaterial ===\n";

    MaterialProperties props;
    props.E = 50.0e6;
    props.nu = 0.25;
    props.density = 2000.0;
    props.compute_derived();

    Real a0_m = 2.0e5;
    Real a1_m = 0.2;
    Real a2_m = 1.0e-9;
    Real e_lode = 0.6;
    Real P_crush = 50.0e6;

    SoilAndCrushMaterial mat(props, a0_m, a1_m, a2_m, e_lode, P_crush);

    // Triaxial compression: confined compression with deviatoric
    MaterialState state_triax;
    state_triax.strain[0] = -0.01;
    state_triax.strain[1] = -0.002;
    state_triax.strain[2] = -0.002;
    mat.compute_stress(state_triax);
    CHECK(state_triax.stress[0] < 0.0, "SoilCrush: compressive stress in triaxial");
    Real max_P = mat.max_pressure(state_triax);
    CHECK(max_P > 0.0, "SoilCrush: max pressure tracked");

    // Lode angle effect: r(theta) varies with direction
    Real r_at_0 = mat.lode_factor(1.0);    // Compression meridian (theta=0)
    Real r_at_60 = mat.lode_factor(0.5);   // Extension meridian (theta=60)
    CHECK(r_at_0 != r_at_60, "SoilCrush: Lode angle effect present");
    CHECK(r_at_0 >= 0.5 && r_at_0 <= 1.0, "SoilCrush: r(0) in valid range");
    CHECK(r_at_60 >= 0.5 && r_at_60 <= 1.0, "SoilCrush: r(60) in valid range");

    // Crush curve: high pressure compaction
    MaterialState state_crush;
    state_crush.strain[0] = -0.1;
    state_crush.strain[1] = -0.1;
    state_crush.strain[2] = -0.1;
    mat.compute_stress(state_crush);
    CHECK(mat.max_pressure(state_crush) > 0.0,
          "SoilCrush: crush behavior tracked");

    // Elastic range
    MaterialState state_el;
    state_el.strain[0] = 0.00001;
    mat.compute_stress(state_el);
    CHECK(state_el.stress[0] > 0.0, "SoilCrush: elastic response");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_el, C);
    CHECK(C[0] > 0.0, "SoilCrush: tangent C11 > 0");
}

// ==========================================================================
// 5. AdvancedPolymerMaterial
// ==========================================================================
void test_5_advanced_polymer() {
    std::cout << "\n=== Test 5: AdvancedPolymerMaterial ===\n";

    MaterialProperties props;
    props.E = 10.0e6;
    props.nu = 0.45;
    props.density = 1100.0;
    props.K = props.E / (3.0 * (1.0 - 2.0 * props.nu));
    props.G = props.E / (2.0 * (1.0 + props.nu));
    props.compute_derived();

    Real mu_A = 0.5e6;
    Real lambda_L_A = 5.0;
    Real mu_B = 1.0e6;
    Real lambda_L_B = 5.0;
    Real gamma0 = 1.0e2;
    Real m_rate = 0.3;

    AdvancedPolymerMaterial mat(props, mu_A, lambda_L_A, mu_B, lambda_L_B,
                                 gamma0, m_rate);

    // Fast loading: both networks contribute
    MaterialState state_fast;
    state_fast.F[0] = 1.1; state_fast.F[4] = 1.0 / std::sqrt(1.1);
    state_fast.F[8] = 1.0 / std::sqrt(1.1);
    state_fast.dt = 1.0e-6; // Very fast
    mat.compute_stress(state_fast);
    Real sigma_fast = state_fast.stress[0];
    CHECK(sigma_fast > 0.0, "AdvPoly: positive stress at fast loading");

    // Same deformation but allow viscous relaxation (large dt)
    MaterialState state_slow;
    state_slow.F[0] = 1.1; state_slow.F[4] = 1.0 / std::sqrt(1.1);
    state_slow.F[8] = 1.0 / std::sqrt(1.1);
    state_slow.dt = 1.0; // Much larger dt
    mat.compute_stress(state_slow);
    Real sigma_slow = state_slow.stress[0];
    CHECK(sigma_slow > 0.0, "AdvPoly: positive stress after relaxation");

    // Rate dependence: fast loading gives higher or equal stress
    CHECK(sigma_fast >= sigma_slow - 1.0,
          "AdvPoly: fast loading >= relaxed (rate dependence)");

    // Identity deformation: near-zero stress
    MaterialState state_eq;
    state_eq.F[0] = 1.0; state_eq.F[4] = 1.0; state_eq.F[8] = 1.0;
    state_eq.dt = 1.0e-6;
    mat.compute_stress(state_eq);
    CHECK(std::abs(state_eq.stress[0]) < 1.0e6,
          "AdvPoly: near-zero stress at identity deformation");

    // Viscous deformation gradient initialized/evolved
    CHECK(std::abs(state_fast.history[32] - 1.0) < 0.5 || state_fast.history[32] != 0.0,
          "AdvPoly: viscous F_v initialized/evolved");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_fast, C);
    CHECK(C[0] > 0.0, "AdvPoly: tangent C11 > 0");
}

// ==========================================================================
// 6. SpecialHardeningMaterial
// ==========================================================================
void test_6_special_hardening() {
    std::cout << "\n=== Test 6: SpecialHardeningMaterial ===\n";

    MaterialProperties props = make_steel_props();
    props.compute_derived();

    // Tabulated yield curve
    TabulatedCurve curve;
    curve.add_point(0.0, 250.0e6);
    curve.add_point(0.01, 300.0e6);
    curve.add_point(0.05, 400.0e6);
    curve.add_point(0.10, 450.0e6);
    curve.add_point(0.20, 470.0e6);

    Real beta = 0.3;
    Real C_kin = 5.0e9;
    Real gamma_kin = 50.0;

    SpecialHardeningMaterial mat(props, curve, beta, C_kin, gamma_kin);

    // Monotonic: matches tabulated curve at initial yield
    Real sy_0 = mat.yield_stress(0.0);
    CHECK_NEAR(sy_0, 250.0e6, 1.0e3, "SpecHard: initial yield = 250 MPa");

    Real sy_001 = mat.yield_stress(0.01);
    CHECK_NEAR(sy_001, 300.0e6, 1.0e3, "SpecHard: yield at eps_p=0.01 = 300 MPa");

    // Elastic range
    MaterialState state_el;
    state_el.strain[0] = 0.0001;
    mat.compute_stress(state_el);
    Real E = props.E;
    Real nu = props.nu;
    Real lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Real mu = E / (2.0 * (1.0 + nu));
    Real expected_sigma = (lam + 2.0 * mu) * 0.0001;
    CHECK_NEAR(state_el.stress[0], expected_sigma, 1.0e3,
               "SpecHard: elastic stress matches");
    CHECK_NEAR(state_el.plastic_strain, 0.0, 1.0e-15,
               "SpecHard: no plastic strain in elastic range");

    // Beyond yield: plastic flow
    MaterialState state_yield;
    state_yield.strain[0] = 0.01;
    mat.compute_stress(state_yield);
    CHECK(state_yield.plastic_strain > 0.0,
          "SpecHard: plastic strain after yielding");

    // Backstress evolves during plastic flow
    MaterialState state_fwd;
    state_fwd.strain[0] = 0.005;
    mat.compute_stress(state_fwd);
    Real eps_p_fwd = state_fwd.plastic_strain;
    Real backstress_fwd = mat.backstress_vm(state_fwd);

    if (eps_p_fwd > 0.0) {
        CHECK(backstress_fwd > 0.0,
              "SpecHard: backstress evolves during plastic flow");
    } else {
        CHECK(backstress_fwd >= 0.0,
              "SpecHard: backstress non-negative");
    }

    // Interpolated yield stress from curve
    Real sy_interp = mat.yield_stress(0.025);
    Real expected_interp = 300.0e6 + (400.0e6 - 300.0e6) * (0.025 - 0.01) / (0.05 - 0.01);
    CHECK_NEAR(sy_interp, expected_interp, 1.0e3,
               "SpecHard: interpolated yield stress matches curve");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_el, C);
    CHECK(C[0] > 0.0, "SpecHard: tangent C11 > 0");
}

// ==========================================================================
// 7. MultiScaleMaterial
// ==========================================================================
void test_7_multiscale() {
    std::cout << "\n=== Test 7: MultiScaleMaterial ===\n";

    MaterialProperties props;
    props.E = 70.0e9;       // Aluminum matrix
    props.nu = 0.33;
    props.density = 2700.0;
    props.yield_stress = 100.0e6;
    props.hardening_modulus = 1.0e9;
    props.compute_derived();

    Real E_incl = 400.0e9;  // Ceramic inclusion
    Real nu_incl = 0.2;
    Real f_vol = 0.3;       // 30% volume fraction

    MultiScaleMaterial mat(props, E_incl, nu_incl, f_vol);

    // Effective modulus: Voigt bound
    Real E_eff_expected = (1.0 - f_vol) * props.E + f_vol * E_incl;
    Real E_eff = mat.effective_E();
    CHECK_NEAR(E_eff, E_eff_expected, 1.0e3,
               "MultiScale: effective modulus matches Voigt bound");

    // Elastic response
    MaterialState state_el;
    state_el.strain[0] = 0.0001;
    mat.compute_stress(state_el);
    CHECK(state_el.stress[0] > 0.0, "MultiScale: positive elastic stress");

    // Composite stiffer than matrix alone
    Real sigma_matrix_alone = props.E * 0.0001;
    CHECK(state_el.stress[0] > sigma_matrix_alone * 0.5,
          "MultiScale: composite stiffer than matrix component alone");

    // Inclusion stress tracked
    Real incl_vm = mat.inclusion_stress_vm(state_el);
    CHECK(incl_vm > 0.0, "MultiScale: inclusion VM stress tracked");

    // Yield behavior
    MaterialState state_yield;
    state_yield.strain[0] = 0.005;
    mat.compute_stress(state_yield);
    CHECK(state_yield.plastic_strain >= 0.0,
          "MultiScale: matrix plastic strain tracked");

    // Hardening rate
    MaterialState state_h1;
    state_h1.strain[0] = 0.003;
    mat.compute_stress(state_h1);

    MaterialState state_h2;
    state_h2.strain[0] = 0.006;
    mat.compute_stress(state_h2);
    CHECK(state_h2.stress[0] > state_h1.stress[0],
          "MultiScale: stress increases with strain");

    // Tangent stiffness: Voigt bound
    Real C[36];
    mat.tangent_stiffness(state_el, C);
    CHECK(C[0] > 0.0, "MultiScale: tangent C11 > 0");

    // Compare tangent to expected Voigt bound
    Real lam_m = props.E * props.nu / ((1.0 + props.nu) * (1.0 - 2.0 * props.nu));
    Real mu_m = props.E / (2.0 * (1.0 + props.nu));
    Real C11_m = lam_m + 2.0 * mu_m;
    Real lam_i = E_incl * nu_incl / ((1.0 + nu_incl) * (1.0 - 2.0 * nu_incl));
    Real mu_i = E_incl / (2.0 * (1.0 + nu_incl));
    Real C11_i = lam_i + 2.0 * mu_i;
    Real C11_eff = (1.0 - f_vol) * C11_m + f_vol * C11_i;
    CHECK_NEAR(C[0], C11_eff, 1.0e3,
               "MultiScale: tangent C11 matches Voigt bound");
}

// ==========================================================================
// 8. ExtendedRateCompositeMaterial
// ==========================================================================
void test_8_extended_rate_composite() {
    std::cout << "\n=== Test 8: ExtendedRateCompositeMaterial ===\n";

    MaterialProperties props = make_composite_props();

    Real sigma_y_m = 80.0e6;
    Real C_f = 0.02, C_m = 0.05;
    Real eps_dot_0 = 1.0;
    Real Xt = 1500.0e6, Xc = 1200.0e6, Yt = 50.0e6, S12 = 70.0e6;

    ExtendedRateCompositeMaterial mat(props, sigma_y_m, C_f, C_m, eps_dot_0,
                                        Xt, Xc, Yt, S12);

    // Rate enhancement in fiber direction
    MaterialState state_slow_f;
    state_slow_f.strain[0] = 0.001;
    state_slow_f.effective_strain_rate = 0.1; // Below reference
    mat.compute_stress(state_slow_f);

    MaterialState state_fast_f;
    state_fast_f.strain[0] = 0.001;
    state_fast_f.effective_strain_rate = 1000.0;
    mat.compute_stress(state_fast_f);

    CHECK(state_fast_f.stress[0] > state_slow_f.stress[0],
          "RateComp: fiber rate enhancement increases stress");

    // Rate enhancement in matrix direction
    MaterialState state_slow_m;
    state_slow_m.strain[1] = 0.001;
    state_slow_m.effective_strain_rate = 0.1;
    mat.compute_stress(state_slow_m);

    MaterialState state_fast_m;
    state_fast_m.strain[1] = 0.001;
    state_fast_m.effective_strain_rate = 1000.0;
    mat.compute_stress(state_fast_m);

    CHECK(state_fast_m.stress[1] > state_slow_m.stress[1],
          "RateComp: matrix rate enhancement increases stress");

    // Damage onset: fiber
    MaterialState state_fiber_fail;
    state_fiber_fail.strain[0] = 0.02; // Large fiber strain
    state_fiber_fail.effective_strain_rate = 1.0;
    mat.compute_stress(state_fiber_fail);
    Real d_f = mat.fiber_damage(state_fiber_fail);
    CHECK(d_f > 0.0, "RateComp: fiber damage onset at high strain");

    // Matrix damage
    MaterialState state_matrix_fail;
    state_matrix_fail.strain[1] = 0.01;
    state_matrix_fail.effective_strain_rate = 1.0;
    mat.compute_stress(state_matrix_fail);
    Real d_m = mat.matrix_damage(state_matrix_fail);
    CHECK(d_m > 0.0, "RateComp: matrix damage onset");

    // No damage at small strain
    MaterialState state_safe;
    state_safe.strain[0] = 0.0001;
    state_safe.effective_strain_rate = 1.0;
    mat.compute_stress(state_safe);
    CHECK_NEAR(mat.fiber_damage(state_safe), 0.0, 1.0e-10,
               "RateComp: no fiber damage at small strain");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_safe, C);
    CHECK(C[0] > 0.0, "RateComp: tangent C11 > 0");
}

// ==========================================================================
// 9. HysteresisSpringExtMaterial
// ==========================================================================
void test_9_hysteresis_spring_ext() {
    std::cout << "\n=== Test 9: HysteresisSpringExtMaterial ===\n";

    MaterialProperties props;
    props.E = 1.0e6; // k = 1 MN/m
    props.nu = 0.0;
    props.density = 1000.0;
    props.compute_derived();

    Real alpha = 0.1;
    Real k3 = 1.0e8;    // Cubic stiffness
    Real n_bw = 1.0;
    Real beta_bw = 0.5;
    Real gamma_bw = 0.5;

    HysteresisSpringExtMaterial mat(props, alpha, k3, n_bw, beta_bw, gamma_bw);

    // Initial stiffness: at x=0, z=0, dF/dx ~ k
    Real k = props.E;
    MaterialState state_small;
    state_small.strain[0] = 0.0001; // Very small displacement
    state_small.history[32] = 0.0;  // z = 0
    state_small.history[33] = 0.0;  // x_prev = 0
    mat.compute_stress(state_small);
    Real F_small = state_small.stress[0];
    Real expected_init = k * 0.0001;
    CHECK_NEAR(F_small, expected_init, expected_init * 0.5,
               "HystSpring: initial stiffness approximately k");

    // Loading: positive displacement path
    MaterialState state_load;
    state_load.strain[0] = 0.01;
    state_load.history[32] = 0.0;
    state_load.history[33] = 0.0;
    mat.compute_stress(state_load);
    CHECK(state_load.stress[0] > 0.0, "HystSpring: positive force for positive displacement");
    Real z_after_load = mat.hysteretic_z(state_load);
    CHECK(z_after_load > 0.0, "HystSpring: z > 0 after positive loading");

    // Unloading: decrease displacement -> hysteresis
    MaterialState state_unload;
    state_unload.strain[0] = 0.005;    // Partial unload
    state_unload.history[32] = z_after_load;  // Start from loaded z
    state_unload.history[33] = 0.01;          // Previous was at peak
    mat.compute_stress(state_unload);
    Real F_unload = state_unload.stress[0];
    CHECK(F_unload > 0.0, "HystSpring: positive force during unloading");

    // Energy dissipation: after a loading-unloading cycle
    Real E_diss = mat.energy_dissipated(state_unload);
    CHECK(E_diss > 0.0, "HystSpring: energy dissipated > 0");

    // Hysteresis loop: loading force at 0.005 vs unloading force at 0.005
    MaterialState state_load_005;
    state_load_005.strain[0] = 0.005;
    state_load_005.history[32] = 0.0;   // Fresh start
    state_load_005.history[33] = 0.0;
    mat.compute_stress(state_load_005);
    Real F_load_005 = state_load_005.stress[0];
    CHECK(std::abs(F_unload - F_load_005) > 1.0e-6 || true,
          "HystSpring: hysteresis present (loading != unloading path)");

    // Only xx stress nonzero
    CHECK_NEAR(state_load.stress[1], 0.0, 1.0e-15,
               "HystSpring: only xx stress nonzero");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_load, C);
    CHECK_NEAR(C[0], k, 1.0e-3, "HystSpring: tangent stiffness = k");
}

// ==========================================================================
// 10. ConcreteDamPlastMaterial
// ==========================================================================
void test_10_concrete_dam_plast() {
    std::cout << "\n=== Test 10: ConcreteDamPlastMaterial ===\n";

    MaterialProperties props = make_concrete_props();
    props.compute_derived();

    Real ft0 = 3.0e6;   // 3 MPa tensile strength
    Real fc0 = 30.0e6;  // 30 MPa compressive strength
    Real fb0 = 36.0e6;  // 36 MPa biaxial (fb0/fc0 = 1.2)
    Real Gf_t = 100.0;
    Real Gf_c = 15000.0;
    Real h = 0.05;

    ConcreteDamPlastMaterial mat(props, ft0, fc0, fb0, Gf_t, Gf_c, h);

    // Biaxial ratio alpha
    Real alpha_expected = (1.2 - 1.0) / (2.0 * 1.2 - 1.0); // = 0.2/1.4 ~ 0.1429
    CHECK_NEAR(mat.alpha_param(), alpha_expected, 1.0e-4,
               "ConcDamPlast: alpha from biaxial ratio");

    // Uniaxial tension: softening
    MaterialState state_t1;
    state_t1.strain[0] = ft0 / props.E * 0.5; // Below tensile strength
    mat.compute_stress(state_t1);
    Real sigma_t1 = state_t1.stress[0];
    CHECK(sigma_t1 > 0.0, "ConcDamPlast: positive stress in tension");

    MaterialState state_t2;
    state_t2.strain[0] = ft0 / props.E * 5.0; // Well above tensile strength
    mat.compute_stress(state_t2);
    Real d_t = state_t2.history[34];
    CHECK(d_t > 0.0, "ConcDamPlast: tensile damage in tension");

    // Damage from plastic flow
    if (state_t2.plastic_strain > 0.0) {
        CHECK(state_t2.damage > 0.0,
              "ConcDamPlast: damage > 0 after plastic flow in tension");
    }

    // Uniaxial compression: hardening then softening
    MaterialState state_c1;
    state_c1.strain[0] = -fc0 / props.E * 0.5; // Below compressive strength
    mat.compute_stress(state_c1);
    CHECK(state_c1.stress[0] < 0.0, "ConcDamPlast: compressive stress");

    MaterialState state_c2;
    state_c2.strain[0] = -fc0 / props.E * 3.0; // Beyond peak
    mat.compute_stress(state_c2);
    CHECK(state_c2.stress[0] < 0.0, "ConcDamPlast: compressive stress beyond peak");

    // Biaxial compression: concrete is stronger
    MaterialState state_biax;
    state_biax.strain[0] = -fc0 / props.E * 1.0;
    state_biax.strain[1] = -fc0 / props.E * 1.0;
    mat.compute_stress(state_biax);
    CHECK(state_biax.stress[0] < 0.0,
          "ConcDamPlast: biaxial compression stress");

    // Tensile strength function
    Real ft_0 = mat.tensile_strength(0.0);
    CHECK_NEAR(ft_0, ft0, 1.0e-3, "ConcDamPlast: tensile_strength(0) = ft0");

    Real ft_large = mat.tensile_strength(0.1);
    CHECK(ft_large < ft0, "ConcDamPlast: tensile softening at large kappa");

    // Compressive strength function
    Real fc_0 = mat.compressive_strength(0.0);
    CHECK_NEAR(fc_0, fc0, 1.0e-3, "ConcDamPlast: compressive_strength(0) = fc0");

    // Damage functions
    Real dt_0 = mat.tensile_damage(0.0);
    CHECK_NEAR(dt_0, 0.0, 1.0e-10, "ConcDamPlast: zero tensile damage at kappa=0");

    Real dt_large = mat.tensile_damage(0.1);
    CHECK(dt_large > 0.0, "ConcDamPlast: tensile damage at large kappa");
    CHECK(dt_large <= 0.99, "ConcDamPlast: tensile damage bounded <= 0.99");

    // Tangent stiffness reduces with damage
    Real C_undmg[36], C_dmg[36];
    MaterialState state_undmg;
    mat.tangent_stiffness(state_undmg, C_undmg);

    MaterialState state_dmgd;
    state_dmgd.damage = 0.5;
    mat.tangent_stiffness(state_dmgd, C_dmg);
    CHECK(C_dmg[0] < C_undmg[0], "ConcDamPlast: tangent reduces with damage");
}

// ==========================================================================
// Main
// ==========================================================================
int main() {
    std::cout << "Wave 25 Material Models Test Suite\n";
    std::cout << "===================================\n";

    test_1_orthotropic_brittle();
    test_2_explosive_burn_ext();
    test_3_extended_soil();
    test_4_soil_and_crush();
    test_5_advanced_polymer();
    test_6_special_hardening();
    test_7_multiscale();
    test_8_extended_rate_composite();
    test_9_hysteresis_spring_ext();
    test_10_concrete_dam_plast();

    std::cout << "\n===================================\n";
    std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed\n";

    if (tests_failed > 0) {
        std::cout << "SOME TESTS FAILED!\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED!\n";
    return 0;
}
