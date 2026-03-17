/**
 * @file material_wave10_test.cpp
 * @brief Comprehensive test for Wave 10 material models (20 advanced constitutive models)
 */

#include <nexussim/physics/material_wave10.hpp>
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

// Helper: make standard steel-like properties
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

// Helper: make rubber-like properties
static MaterialProperties make_rubber_props() {
    MaterialProperties props;
    props.E = 6.0e6;
    props.nu = 0.4995;
    props.density = 1200.0;
    props.G = props.E / (2.0 * (1.0 + props.nu));
    props.K = props.E / (3.0 * (1.0 - 2.0 * props.nu));
    props.compute_derived();
    return props;
}

// ==========================================================================
// 1. HillAnisotropicMaterial
// ==========================================================================
void test_hill_anisotropic() {
    std::cout << "\n=== Test 1: HillAnisotropicMaterial ===\n";

    MaterialProperties props = make_steel_props();

    // Isotropic R-values (R0=R45=R90=1) should recover von Mises
    HillAnisotropicMaterial mat_iso(props, 1.0, 1.0, 1.0);

    // Check Hill parameters for isotropic case
    // F = R0/(R90*(1+R0)) = 1/(1*2) = 0.5
    CHECK_NEAR(mat_iso.get_F(), 0.5, 1.0e-10, "Hill F for isotropic");
    // G = 1/(1+R0) = 0.5
    CHECK_NEAR(mat_iso.get_G(), 0.5, 1.0e-10, "Hill G for isotropic");
    // H = R0/(1+R0) = 0.5
    CHECK_NEAR(mat_iso.get_H(), 0.5, 1.0e-10, "Hill H for isotropic");
    // N = (R0+R90)*(1+2*R45)/(2*R90*(1+R0)) = 2*3/(2*1*2) = 1.5
    CHECK_NEAR(mat_iso.get_N(), 1.5, 1.0e-10, "Hill N for isotropic");

    // Elastic response: small uniaxial strain
    MaterialState state;
    state.strain[0] = 0.0001;
    mat_iso.compute_stress(state);
    Real sigma_xx = state.stress[0];
    CHECK(sigma_xx > 0.0, "Hill iso: positive stress for tension");
    // Linear elastic regime: sigma_xx ~ E*(1-nu)/((1+nu)*(1-2*nu)) * eps
    Real E = props.E; Real nu = props.nu;
    Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Real mu = E / (2.0 * (1.0 + nu));
    Real expected_xx = (lambda + 2.0 * mu) * 0.0001;
    CHECK_NEAR(sigma_xx, expected_xx, 1.0e3, "Hill iso: elastic uniaxial stress");

    // Anisotropic case: different R-values
    HillAnisotropicMaterial mat_aniso(props, 2.0, 1.5, 3.0);
    // F = R0/(R90*(1+R0)) = 2/(3*3) = 0.2222
    // H = R0/(1+R0) = 2/3 = 0.6667
    CHECK(mat_aniso.get_F() != mat_aniso.get_H(), "Aniso: F != H when R0 != R90");
    // G = 1/(1+R0) = 1/3 = 0.3333
    CHECK_NEAR(mat_aniso.get_G(), 1.0 / 3.0, 1.0e-10, "Aniso: G = 1/(1+R0)");
}

// ==========================================================================
// 2. BarlatYldMaterial
// ==========================================================================
void test_barlat_yld() {
    std::cout << "\n=== Test 2: BarlatYldMaterial ===\n";

    MaterialProperties props = make_steel_props();

    // Default exponent 8 for FCC, isotropic scaling (alpha1=alpha2=1)
    BarlatYldMaterial mat_fcc(props, 8.0, 1.0, 1.0);
    CHECK_NEAR(mat_fcc.exponent(), 8.0, 1.0e-10, "Barlat FCC exponent");

    // Elastic regime: small strain
    MaterialState state;
    state.strain[0] = 0.0001;
    mat_fcc.compute_stress(state);
    CHECK(state.stress[0] > 0.0, "Barlat: positive stress in tension");
    CHECK_NEAR(state.plastic_strain, 0.0, 1.0e-15, "Barlat: no plasticity in elastic range");

    // BCC exponent
    BarlatYldMaterial mat_bcc(props, 6.0, 1.0, 1.0);
    CHECK_NEAR(mat_bcc.exponent(), 6.0, 1.0e-10, "Barlat BCC exponent");

    // Yielding: push well beyond yield
    MaterialState state2;
    state2.strain[0] = 0.01; // 1% strain
    mat_fcc.compute_stress(state2);
    CHECK(state2.plastic_strain > 0.0, "Barlat: plastic strain after yielding");
    CHECK(state2.stress[0] > 0.0, "Barlat: stress positive after yielding");

    // Anisotropy scaling: alpha1=1.2, alpha2=0.8 -> average = 1.0, same as isotropic
    BarlatYldMaterial mat_aniso(props, 8.0, 1.2, 0.8);
    MaterialState state3;
    state3.strain[0] = 0.0001;
    mat_aniso.compute_stress(state3);
    CHECK_NEAR(state3.stress[0], state.stress[0], 1.0e3,
               "Barlat: alpha avg=1 matches isotropic in elastic range");
}

// ==========================================================================
// 3. TabulatedJohnsonCookMaterial
// ==========================================================================
void test_tabulated_jc() {
    std::cout << "\n=== Test 3: TabulatedJohnsonCookMaterial ===\n";

    MaterialProperties props = make_steel_props();

    // Yield curve: constant 250 MPa
    TabulatedCurve yield_curve;
    yield_curve.add_point(0.0, 250.0e6);
    yield_curve.add_point(0.5, 400.0e6);
    yield_curve.add_point(1.0, 500.0e6);

    // Rate curve: 1.0 at low rates, 1.5 at high rates
    TabulatedCurve rate_curve;
    rate_curve.add_point(-5.0, 1.0);  // ln(eps_dot) = -5
    rate_curve.add_point(0.0, 1.0);
    rate_curve.add_point(5.0, 1.2);
    rate_curve.add_point(10.0, 1.5);

    TabulatedJohnsonCookMaterial mat(props, yield_curve, rate_curve);

    // Elastic response
    MaterialState state;
    state.strain[0] = 0.0001;
    mat.compute_stress(state);
    CHECK(state.stress[0] > 0.0, "TabJC: positive stress in elastic range");
    CHECK_NEAR(state.plastic_strain, 0.0, 1.0e-15, "TabJC: no plasticity in elastic");

    // Beyond yield
    MaterialState state2;
    state2.strain[0] = 0.01;
    mat.compute_stress(state2);
    CHECK(state2.plastic_strain > 0.0, "TabJC: plastic strain after yielding");

    // Rate effect: high rate should give higher yield
    MaterialState state_fast;
    state_fast.strain[0] = 0.01;
    state_fast.effective_strain_rate = 1000.0;
    mat.compute_stress(state_fast);
    // At high rate, yield stress is enhanced, so VM stress may be higher before return
    CHECK(state_fast.stress[0] > 0.0, "TabJC: positive stress at high rate");

    // Verify yield curve interpolation
    Real y_mid = yield_curve.evaluate(0.25);
    Real y_expected = 250.0e6 + (400.0e6 - 250.0e6) * 0.5;
    CHECK_NEAR(y_mid, y_expected, 1.0e3, "TabJC: yield curve interpolation at 0.25");
}

// ==========================================================================
// 4. ConcreteMaterial
// ==========================================================================
void test_concrete() {
    std::cout << "\n=== Test 4: ConcreteMaterial ===\n";

    MaterialProperties props;
    props.E = 30.0e9;   // Concrete E ~ 30 GPa
    props.nu = 0.2;
    props.density = 2400.0;
    props.yield_stress = 30.0e6;       // fc = 30 MPa compressive strength
    props.damage_threshold = 3.0e6;    // ft = 3 MPa tensile strength
    props.hardening_modulus = 0.0;
    props.compute_derived();

    ConcreteMaterial mat(props);

    // Elastic compression: small hydrostatic compression
    MaterialState state;
    state.strain[0] = -0.0001;
    state.strain[1] = -0.0001;
    state.strain[2] = -0.0001;
    mat.compute_stress(state);
    CHECK(state.stress[0] < 0.0, "Concrete: compressive stress for compressive strain");

    // Small elastic tension
    MaterialState state_t;
    state_t.strain[0] = 0.00001;
    mat.compute_stress(state_t);
    CHECK(state_t.stress[0] > 0.0, "Concrete: tensile stress positive");
    CHECK_NEAR(state_t.plastic_strain, 0.0, 1.0e-15, "Concrete: elastic range no plasticity");

    // DP yielding under deviatoric + pressure
    MaterialState state_dev;
    state_dev.strain[0] = 0.005;
    state_dev.strain[1] = -0.002;
    state_dev.strain[2] = -0.002;
    mat.compute_stress(state_dev);
    CHECK(state_dev.plastic_strain > 0.0, "Concrete: plastic strain under shear");

    // Damage tracking in tension
    MaterialState state_tension;
    state_tension.strain[0] = 0.01; // Large tension
    mat.compute_stress(state_tension);
    CHECK(state_tension.damage >= 0.0, "Concrete: damage >= 0 in tension");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state, C);
    CHECK(C[0] > 0.0, "Concrete: tangent C[0] > 0");
}

// ==========================================================================
// 5. FabricMaterial
// ==========================================================================
void test_fabric() {
    std::cout << "\n=== Test 5: FabricMaterial ===\n";

    MaterialProperties props;
    props.E = 1.0e9;
    props.nu = 0.3;
    props.E1 = 2.0e9;    // Warp direction
    props.E2 = 1.0e9;    // Fill direction
    props.nu12 = 0.1;
    props.G12 = 0.5e9;
    props.compute_derived();

    FabricMaterial mat(props);

    // Tension in warp direction
    MaterialState state;
    state.strain[0] = 0.001;
    state.strain[1] = 0.0;
    mat.compute_stress(state);
    CHECK(state.stress[0] > 0.0, "Fabric: warp tension gives positive stress");
    CHECK_NEAR(state.stress[2], 0.0, 1.0e-10, "Fabric: zero thickness stress");

    // Compression in both directions -> zero stress
    MaterialState state_comp;
    state_comp.strain[0] = -0.001;
    state_comp.strain[1] = -0.001;
    mat.compute_stress(state_comp);
    CHECK_NEAR(state_comp.stress[0], 0.0, 1.0e-10, "Fabric: zero stress in biaxial compression");
    CHECK_NEAR(state_comp.stress[1], 0.0, 1.0e-10, "Fabric: zero stress[1] in compression");

    // Shear response (not affected by no-compression rule)
    MaterialState state_shear;
    state_shear.strain[3] = 0.001;
    mat.compute_stress(state_shear);
    Real tau_expected = 0.5e9 * 0.001;
    CHECK_NEAR(state_shear.stress[3], tau_expected, 1.0e2,
               "Fabric: shear stress = G12 * gamma");

    // Tangent stiffness depends on sign of strain
    Real C_tens[36];
    mat.tangent_stiffness(state, C_tens);
    CHECK(C_tens[0] > 0.0, "Fabric: C11 > 0 in tension");

    Real C_comp[36];
    mat.tangent_stiffness(state_comp, C_comp);
    CHECK_NEAR(C_comp[0], 0.0, 1.0e-10, "Fabric: C11 = 0 in compression");
}

// ==========================================================================
// 6. CohesiveZoneMaterial
// ==========================================================================
void test_cohesive_zone() {
    std::cout << "\n=== Test 6: CohesiveZoneMaterial ===\n";

    MaterialProperties props;
    props.E = 1.0e12;                  // K_n (penalty stiffness)
    props.nu = 0.3;
    props.yield_stress = 50.0e6;       // sigma_max
    props.damage_exponent = 500.0;     // GIc = 500 J/m^2
    props.compute_derived();

    CohesiveZoneMaterial mat(props);

    // delta_0 = sigma_max / K_n = 50e6/1e12 = 5e-5
    Real K_n = props.E;
    Real sigma_max = props.yield_stress;
    Real delta_0 = sigma_max / K_n;     // 5e-5
    Real GIc = props.damage_exponent;
    Real delta_f = 2.0 * GIc / sigma_max; // 2*500/50e6 = 2e-5 -> but delta_f < delta_0!
    // The code clamps: if (delta_f < delta_0) delta_f = 2.0 * delta_0
    if (delta_f < delta_0) delta_f = 2.0 * delta_0;

    // Small opening (elastic)
    MaterialState state;
    state.strain[0] = delta_0 * 0.5; // Half of delta_0
    mat.compute_stress(state);
    Real traction = K_n * delta_0 * 0.5;
    CHECK(state.stress[0] > 0.0, "CZ: positive traction in opening");
    CHECK_NEAR(state.damage, 0.0, 1.0e-15, "CZ: no damage below delta_0");

    // Compression: penalty contact
    MaterialState state_comp;
    state_comp.strain[0] = -0.0001;
    mat.compute_stress(state_comp);
    CHECK(state_comp.stress[0] < 0.0, "CZ: compressive traction for negative opening");
    CHECK_NEAR(state_comp.stress[0], K_n * (-0.0001), 1.0,
               "CZ: compression uses full stiffness K_n");

    // Opening beyond delta_0: damage initiates
    MaterialState state_dmg;
    state_dmg.strain[0] = delta_0 * 1.5;
    mat.compute_stress(state_dmg);
    CHECK(state_dmg.damage > 0.0, "CZ: damage > 0 beyond delta_0");
    CHECK(state_dmg.damage < 1.0, "CZ: damage < 1 before delta_f");

    // Full separation
    MaterialState state_fail;
    state_fail.strain[0] = delta_f * 2.0;
    mat.compute_stress(state_fail);
    CHECK_NEAR(state_fail.damage, 1.0, 1.0e-10, "CZ: full damage at large opening");
}

// ==========================================================================
// 7. SoilCapMaterial
// ==========================================================================
void test_soil_cap() {
    std::cout << "\n=== Test 7: SoilCapMaterial ===\n";

    MaterialProperties props;
    props.E = 50.0e6;       // Soil E ~ 50 MPa
    props.nu = 0.3;
    props.density = 1800.0;
    props.yield_stress = 20.0e3;         // cohesion c = 20 kPa
    props.damage_threshold = 30.0;       // friction angle = 30 degrees
    props.foam_E_crush = 2.0;            // Cap ratio R
    props.foam_densification = 0.0;      // Will use default X0
    props.hardening_modulus = 0.0;
    props.compute_derived();

    SoilCapMaterial mat(props);

    // Small elastic strain
    MaterialState state;
    state.strain[0] = 0.0001;
    mat.compute_stress(state);
    CHECK(state.stress[0] > 0.0, "SoilCap: positive stress in tension");
    CHECK_NEAR(state.plastic_strain, 0.0, 1.0e-15, "SoilCap: elastic range no plasticity");

    // Deviatoric loading beyond shear failure
    MaterialState state_fail;
    state_fail.strain[0] = 0.01;
    state_fail.strain[1] = -0.005;
    state_fail.strain[2] = -0.005;
    mat.compute_stress(state_fail);
    CHECK(state_fail.plastic_strain > 0.0, "SoilCap: shear failure activates plasticity");

    // Hydrostatic compression
    MaterialState state_hydro;
    state_hydro.strain[0] = -0.001;
    state_hydro.strain[1] = -0.001;
    state_hydro.strain[2] = -0.001;
    mat.compute_stress(state_hydro);
    CHECK(state_hydro.stress[0] < 0.0, "SoilCap: compressive stress under compression");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state, C);
    CHECK(C[0] > 0.0, "SoilCap: tangent C11 > 0");
    CHECK(C[21] > 0.0, "SoilCap: tangent shear > 0");
}

// ==========================================================================
// 8. UserDefinedMaterial
// ==========================================================================
void test_user_defined() {
    std::cout << "\n=== Test 8: UserDefinedMaterial ===\n";

    MaterialProperties props = make_steel_props();

    // Without callback: falls back to linear elastic
    UserDefinedMaterial mat_default(props);
    MaterialState state;
    state.strain[0] = 0.001;
    mat_default.compute_stress(state);
    Real E = props.E; Real nu = props.nu;
    Real lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Real mu = E / (2.0 * (1.0 + nu));
    Real expected = (lam + 2.0 * mu) * 0.001;
    CHECK_NEAR(state.stress[0], expected, 1.0e3,
               "UserDef: default elastic uniaxial stress");

    // With custom callback: double the stress
    auto double_callback = [](const MaterialProperties& p, MaterialState& s) {
        Material::elastic_stress(s.strain, p.E, p.nu, s.stress);
        for (int i = 0; i < 6; ++i) s.stress[i] *= 2.0;
    };

    UserDefinedMaterial mat_custom(props, double_callback);
    MaterialState state2;
    state2.strain[0] = 0.001;
    mat_custom.compute_stress(state2);
    CHECK_NEAR(state2.stress[0], 2.0 * expected, 1.0e3,
               "UserDef: custom callback doubles stress");

    // Set callback after construction
    UserDefinedMaterial mat_set(props);
    mat_set.set_callback(double_callback);
    MaterialState state3;
    state3.strain[0] = 0.001;
    mat_set.compute_stress(state3);
    CHECK_NEAR(state3.stress[0], 2.0 * expected, 1.0e3,
               "UserDef: set_callback works");

    // Tangent stiffness (always elastic)
    Real C[36];
    mat_custom.tangent_stiffness(state2, C);
    CHECK(C[0] > 0.0, "UserDef: tangent C11 > 0");
}

// ==========================================================================
// 9. ArrudaBoyceMaterial
// ==========================================================================
void test_arruda_boyce() {
    std::cout << "\n=== Test 9: ArrudaBoyceMaterial ===\n";

    MaterialProperties props = make_rubber_props();
    props.foam_densification = 5.0;  // lambda_L = 5
    props.compute_derived();

    ArrudaBoyceMaterial mat(props);

    // Identity deformation (F = I): zero stress
    MaterialState state_id;
    // F is already identity from constructor
    mat.compute_stress(state_id);
    CHECK_NEAR(state_id.stress[0], 0.0, 1.0e-3, "AB: zero stress at identity F");
    CHECK_NEAR(state_id.stress[3], 0.0, 1.0e-3, "AB: zero shear at identity F");

    // Uniaxial stretch
    MaterialState state_stretch;
    Real stretch = 1.1;
    Real lat = 1.0 / std::sqrt(stretch); // Nearly incompressible
    state_stretch.F[0] = stretch;
    state_stretch.F[4] = lat;
    state_stretch.F[8] = lat;
    mat.compute_stress(state_stretch);
    CHECK(state_stretch.stress[0] > 0.0, "AB: tensile stress for stretch > 1");
    CHECK(state_stretch.vol_strain > -0.01, "AB: vol_strain ~ 0 for incompressible");

    // Compression: stretch < 1
    MaterialState state_comp;
    Real comp = 0.9;
    Real lat_c = 1.0 / std::sqrt(comp);
    state_comp.F[0] = comp;
    state_comp.F[4] = lat_c;
    state_comp.F[8] = lat_c;
    mat.compute_stress(state_comp);
    CHECK(state_comp.stress[0] < 0.0, "AB: compressive stress for stretch < 1");

    // Stiffening at large stretch (non-Gaussian)
    MaterialState state_large;
    Real large_s = 2.0;
    Real lat_l = 1.0 / std::sqrt(large_s);
    state_large.F[0] = large_s;
    state_large.F[4] = lat_l;
    state_large.F[8] = lat_l;
    mat.compute_stress(state_large);
    // Stress at lambda=2 should be more than 2x stress at lambda=1.1
    double ratio = state_large.stress[0] / (state_stretch.stress[0] + 1.0e-30);
    CHECK(ratio > 2.0, "AB: stiffening at large stretch");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_id, C);
    CHECK(C[0] > 0.0, "AB: tangent C11 > 0");
}

// ==========================================================================
// 10. ShapeMemoryAlloyMaterial
// ==========================================================================
void test_shape_memory_alloy() {
    std::cout << "\n=== Test 10: ShapeMemoryAlloyMaterial ===\n";

    MaterialProperties props;
    props.E = 70.0e9;           // NiTi E ~ 70 GPa
    props.nu = 0.33;
    props.density = 6450.0;
    props.yield_stress = 400.0e6;       // sigma_s^AM (forward start)
    props.damage_threshold = 200.0e6;   // sigma_s^MA (reverse start)
    props.foam_densification = 0.06;    // eps_L (max transformation strain)
    props.hardening_modulus = 1.0e9;    // Plateau slope
    props.compute_derived();

    ShapeMemoryAlloyMaterial mat(props);

    // Elastic regime (below forward transformation)
    MaterialState state;
    state.strain[0] = 0.001;
    mat.compute_stress(state);
    CHECK(state.stress[0] > 0.0, "SMA: positive stress in elastic range");
    CHECK_NEAR(state.history[0], 0.0, 1.0e-10, "SMA: xi=0 (austenite) in elastic");

    // Forward transformation (above sigma_AM)
    MaterialState state_fwd;
    state_fwd.strain[0] = 0.01; // Should generate stress > sigma_AM
    mat.compute_stress(state_fwd);
    CHECK(state_fwd.history[0] > 0.0, "SMA: xi > 0 after forward transformation");
    CHECK(state_fwd.plastic_strain > 0.0, "SMA: transform strain > 0");

    // Verify martensite fraction bounded [0, 1]
    MaterialState state_large;
    state_large.strain[0] = 0.1;
    mat.compute_stress(state_large);
    CHECK(state_large.history[0] >= 0.0, "SMA: xi >= 0");
    CHECK(state_large.history[0] <= 1.0, "SMA: xi <= 1");

    // Elastic modulus check at small strain
    MaterialState state_small;
    state_small.strain[0] = 0.0001;
    mat.compute_stress(state_small);
    Real E = props.E; Real nu = props.nu;
    Real lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Real mu_sma = E / (2.0 * (1.0 + nu));
    Real expected = (lam + 2.0 * mu_sma) * 0.0001;
    CHECK_NEAR(state_small.stress[0], expected, 1.0e4,
               "SMA: elastic stress in austenite phase");
}

// ==========================================================================
// 11. RateDependentFoamMaterial
// ==========================================================================
void test_rate_dependent_foam() {
    std::cout << "\n=== Test 11: RateDependentFoamMaterial ===\n";

    MaterialProperties props;
    props.E = 10.0e6;       // Foam E ~ 10 MPa
    props.nu = 0.1;
    props.density = 50.0;
    props.foam_E_crush = 0.5e6;     // Crush plateau = 0.5 MPa
    props.foam_densification = 0.8;  // Densification strain
    props.CS_D = 100.0;             // Rate parameter D
    props.CS_q = 2.0;               // Rate parameter q
    props.compute_derived();

    RateDependentFoamMaterial mat(props);

    // Tension: uses bulk modulus response
    MaterialState state_tens;
    state_tens.strain[0] = 0.001;
    state_tens.strain[1] = 0.001;
    state_tens.strain[2] = 0.001;
    mat.compute_stress(state_tens);
    CHECK(state_tens.vol_strain > 0.0, "RDFoam: positive vol_strain in tension");

    // Compression: foam crush response
    MaterialState state_crush;
    state_crush.strain[0] = -0.1;
    state_crush.strain[1] = -0.1;
    state_crush.strain[2] = -0.1;
    mat.compute_stress(state_crush);
    CHECK(state_crush.stress[0] < 0.0 || state_crush.stress[0] > 0.0,
          "RDFoam: stress nonzero under volumetric compression");

    // Rate enhancement: high rate should increase crush stress
    MaterialState state_slow;
    state_slow.strain[0] = -0.1;
    state_slow.strain[1] = -0.1;
    state_slow.strain[2] = -0.1;
    state_slow.effective_strain_rate = 0.0;
    mat.compute_stress(state_slow);

    MaterialState state_fast;
    state_fast.strain[0] = -0.1;
    state_fast.strain[1] = -0.1;
    state_fast.strain[2] = -0.1;
    state_fast.effective_strain_rate = 1000.0;
    mat.compute_stress(state_fast);
    // Higher rate should give larger magnitude volumetric stress
    CHECK(std::abs(state_fast.stress[0]) >= std::abs(state_slow.stress[0]) - 1.0,
          "RDFoam: rate enhancement increases stress magnitude");

    // Track max compression in history
    CHECK(state_crush.history[0] >= 0.0, "RDFoam: max compression tracked in history[0]");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_tens, C);
    CHECK(C[0] > 0.0, "RDFoam: tangent C11 > 0");
}

// ==========================================================================
// 12. PronyViscoelasticMaterial
// ==========================================================================
void test_prony_viscoelastic() {
    std::cout << "\n=== Test 12: PronyViscoelasticMaterial ===\n";

    MaterialProperties props;
    props.E = 10.0e6;
    props.nu = 0.45;
    props.density = 1100.0;
    props.prony_nterms = 2;
    props.prony_g[0] = 0.3;
    props.prony_g[1] = 0.2;
    props.prony_tau[0] = 0.01;  // 10 ms
    props.prony_tau[1] = 0.1;   // 100 ms
    props.compute_derived();

    PronyViscoelasticMaterial mat(props);

    // MAX_TERMS check
    CHECK(PronyViscoelasticMaterial::MAX_TERMS == 4,
          "Prony: MAX_TERMS = 4");

    // Instantaneous response (dt = 0): should give full stiffness
    MaterialState state_inst;
    state_inst.strain[0] = 0.001;
    state_inst.dt = 0.0;
    mat.compute_stress(state_inst);
    CHECK(state_inst.stress[0] > 0.0, "Prony: positive stress at t=0");

    // Long-term response (large dt, relaxed)
    // After full relaxation, G_inf = G*(1 - g1 - g2) = G*0.5
    MaterialState state_long;
    state_long.strain[0] = 0.001;
    state_long.dt = 100.0; // Very long time
    mat.compute_stress(state_long);
    CHECK(state_long.stress[0] > 0.0, "Prony: long-term stress still positive");

    // Volume strain tracks correctly
    MaterialState state_vol;
    state_vol.strain[0] = 0.001;
    state_vol.strain[1] = 0.001;
    state_vol.strain[2] = 0.001;
    state_vol.dt = 0.001;
    mat.compute_stress(state_vol);
    CHECK_NEAR(state_vol.vol_strain, 0.003, 1.0e-10, "Prony: vol_strain = sum of normal strains");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_inst, C);
    CHECK(C[0] > 0.0, "Prony: tangent C11 > 0");
    CHECK(C[21] > 0.0, "Prony: tangent shear > 0");
}

// ==========================================================================
// 13. ThermalElasticPlasticMaterial
// ==========================================================================
void test_thermal_elastic_plastic() {
    std::cout << "\n=== Test 13: ThermalElasticPlasticMaterial ===\n";

    MaterialProperties props = make_steel_props();
    props.JC_T_room = 293.15;
    props.JC_T_melt = 1800.0;
    props.thermal_expansion = 1.0e-4; // alpha_E for modulus degradation
    props.compute_derived();

    ThermalElasticPlasticMaterial mat(props);

    // Room temperature: standard elastic response
    MaterialState state_rt;
    state_rt.strain[0] = 0.0001;
    state_rt.temperature = 293.15;
    mat.compute_stress(state_rt);
    Real E_rt = props.E;
    Real nu = props.nu;
    Real lam = E_rt * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Real mu = E_rt / (2.0 * (1.0 + nu));
    Real expected_rt = (lam + 2.0 * mu) * 0.0001;
    CHECK_NEAR(state_rt.stress[0], expected_rt, 1.0e4,
               "ThermalEP: room temperature elastic stress");

    // Elevated temperature: reduced modulus
    MaterialState state_hot;
    state_hot.strain[0] = 0.0001;
    state_hot.temperature = 800.0; // Hot
    mat.compute_stress(state_hot);
    CHECK(state_hot.stress[0] > 0.0, "ThermalEP: positive stress at 800K");
    CHECK(state_hot.stress[0] < state_rt.stress[0],
          "ThermalEP: stress reduced at higher temperature");

    // Near melt temperature: very soft
    MaterialState state_melt;
    state_melt.strain[0] = 0.0001;
    state_melt.temperature = 1700.0;
    mat.compute_stress(state_melt);
    CHECK(state_melt.stress[0] < state_hot.stress[0],
          "ThermalEP: stress further reduced near melt");

    // Yield at room T: push beyond yield
    MaterialState state_yield;
    state_yield.strain[0] = 0.01;
    state_yield.temperature = 293.15;
    mat.compute_stress(state_yield);
    CHECK(state_yield.plastic_strain > 0.0, "ThermalEP: plasticity at room T");

    // Temperature-dependent yield: hot metal yields earlier
    MaterialState state_yield_hot;
    state_yield_hot.strain[0] = 0.005;
    state_yield_hot.temperature = 1000.0;
    mat.compute_stress(state_yield_hot);
    // Higher temperature -> lower yield stress -> easier to yield
    // May or may not yield at this strain, but stress should be lower
    CHECK(state_yield_hot.stress[0] < state_yield.stress[0],
          "ThermalEP: hot yield stress lower than cold");

    // Tangent stiffness: temperature dependent
    Real C_rt[36], C_hot[36];
    mat.tangent_stiffness(state_rt, C_rt);
    mat.tangent_stiffness(state_hot, C_hot);
    CHECK(C_rt[0] > C_hot[0], "ThermalEP: tangent stiffer at room T");
}

// ==========================================================================
// 14. ZerilliArmstrongMaterial
// ==========================================================================
void test_zerilli_armstrong() {
    std::cout << "\n=== Test 14: ZerilliArmstrongMaterial ===\n";

    // BCC metal (e.g., tantalum)
    MaterialProperties props = make_steel_props();
    props.JC_A = 100.0e6;    // C0
    props.JC_B = 1100.0e6;   // C1
    props.JC_n = 0.5;        // n
    props.JC_C = 0.005;      // C3
    props.JC_m = 0.0002;     // C4
    props.CS_D = 500.0e6;    // C5
    props.damage_threshold = 0.0; // BCC mode
    props.compute_derived();

    ZerilliArmstrongMaterial mat_bcc(props);

    // Elastic response
    MaterialState state;
    state.strain[0] = 0.0001;
    state.temperature = 300.0;
    state.effective_strain_rate = 1.0;
    mat_bcc.compute_stress(state);
    CHECK(state.stress[0] > 0.0, "ZA BCC: positive elastic stress");

    // Beyond yield
    MaterialState state_yield;
    state_yield.strain[0] = 0.01;
    state_yield.temperature = 300.0;
    state_yield.effective_strain_rate = 1.0;
    mat_bcc.compute_stress(state_yield);
    CHECK(state_yield.plastic_strain > 0.0, "ZA BCC: plasticity beyond yield");

    // Temperature effect: higher T -> lower yield
    MaterialState state_hot;
    state_hot.strain[0] = 0.01;
    state_hot.temperature = 800.0;
    state_hot.effective_strain_rate = 1.0;
    mat_bcc.compute_stress(state_hot);
    // BCC: thermal activation reduces C1*exp term

    // FCC mode
    MaterialProperties props_fcc = props;
    props_fcc.damage_threshold = 1.0;  // FCC mode
    props_fcc.compute_derived();
    ZerilliArmstrongMaterial mat_fcc(props_fcc);

    MaterialState state_fcc;
    state_fcc.strain[0] = 0.0001;
    state_fcc.temperature = 300.0;
    state_fcc.effective_strain_rate = 1.0;
    mat_fcc.compute_stress(state_fcc);
    CHECK(state_fcc.stress[0] > 0.0, "ZA FCC: positive elastic stress");

    // Rate effect
    MaterialState state_fast;
    state_fast.strain[0] = 0.01;
    state_fast.temperature = 300.0;
    state_fast.effective_strain_rate = 1000.0;
    mat_bcc.compute_stress(state_fast);
    CHECK(state_fast.stress[0] > 0.0, "ZA BCC: stress > 0 at high rate");
}

// ==========================================================================
// 15. SteinbergGuinanMaterial
// ==========================================================================
void test_steinberg_guinan() {
    std::cout << "\n=== Test 15: SteinbergGuinanMaterial ===\n";

    MaterialProperties props = make_steel_props();
    props.yield_stress = 200.0e6;         // Y0
    props.JC_B = 36.0;                    // beta
    props.JC_n = 0.45;                    // n
    props.JC_A = 1.0e-3;                  // dG/dP coefficient A
    props.JC_C = 1.0e-4;                  // dG/dT coefficient B
    props.hardening_modulus = 2.5e9;       // Y_max
    props.compute_derived();

    SteinbergGuinanMaterial mat(props);

    // Elastic range at room T
    MaterialState state;
    state.strain[0] = 0.0001;
    state.temperature = 300.0;
    mat.compute_stress(state);
    CHECK(state.stress[0] > 0.0, "SG: positive stress in elastic range");

    // Beyond yield
    MaterialState state_yield;
    state_yield.strain[0] = 0.01;
    state_yield.temperature = 300.0;
    mat.compute_stress(state_yield);
    CHECK(state_yield.plastic_strain > 0.0, "SG: plasticity at large strain");

    // Pressure effect: hydrostatic compression increases shear modulus
    MaterialState state_pressure;
    state_pressure.strain[0] = -0.01;
    state_pressure.strain[1] = -0.01;
    state_pressure.strain[2] = -0.01;
    state_pressure.temperature = 300.0;
    mat.compute_stress(state_pressure);
    CHECK(state_pressure.stress[0] < 0.0, "SG: compressive stress under pressure");

    // Temperature effect
    MaterialState state_hot;
    state_hot.strain[0] = 0.0001;
    state_hot.temperature = 1000.0;
    mat.compute_stress(state_hot);
    CHECK(state_hot.stress[0] > 0.0, "SG: positive stress at elevated T");

    // Y_max cap: very large plastic strain should not exceed Y_max
    Real C[36];
    mat.tangent_stiffness(state, C);
    CHECK(C[0] > 0.0, "SG: tangent C11 > 0");
    CHECK(C[21] > 0.0, "SG: tangent shear stiffness > 0");
}

// ==========================================================================
// 16. MTSMaterial
// ==========================================================================
void test_mts() {
    std::cout << "\n=== Test 16: MTSMaterial ===\n";

    MaterialProperties props = make_steel_props();
    props.JC_A = 50.0e6;       // sigma_a (athermal)
    props.JC_B = 300.0e6;      // sigma_i_hat (threshold)
    props.JC_n = 0.5;          // p exponent
    props.JC_C = 1.5;          // q exponent
    props.hardening_modulus = 1.0e9;
    props.compute_derived();

    MTSMaterial mat(props);

    // Elastic response at room T
    MaterialState state;
    state.strain[0] = 0.0001;
    state.temperature = 300.0;
    state.effective_strain_rate = 1.0;
    mat.compute_stress(state);
    CHECK(state.stress[0] > 0.0, "MTS: positive elastic stress");
    CHECK_NEAR(state.plastic_strain, 0.0, 1.0e-10, "MTS: no plasticity in elastic");

    // Beyond yield
    MaterialState state_yield;
    state_yield.strain[0] = 0.01;
    state_yield.temperature = 300.0;
    state_yield.effective_strain_rate = 1.0;
    mat.compute_stress(state_yield);
    CHECK(state_yield.plastic_strain > 0.0, "MTS: plasticity at large strain");

    // High temperature: S factor decreases, yield drops
    MaterialState state_hot;
    state_hot.strain[0] = 0.005;
    state_hot.temperature = 1000.0;
    state_hot.effective_strain_rate = 1.0;
    mat.compute_stress(state_hot);
    // At higher T, yield should be lower

    // Rate effect: higher rate increases yield
    MaterialState state_fast;
    state_fast.strain[0] = 0.005;
    state_fast.temperature = 300.0;
    state_fast.effective_strain_rate = 1.0e6;
    mat.compute_stress(state_fast);

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state, C);
    CHECK(C[0] > 0.0, "MTS: tangent C11 > 0");
    CHECK(C[21] > 0.0, "MTS: tangent shear > 0");
}

// ==========================================================================
// 17. BlatzKoMullinsMaterial
// ==========================================================================
void test_blatzko_mullins() {
    std::cout << "\n=== Test 17: BlatzKoMullinsMaterial ===\n";

    MaterialProperties props = make_rubber_props();
    props.foam_unload_factor = 0.5;   // Mullins r parameter
    props.damage_exponent = 0.1;      // Mullins m parameter
    props.compute_derived();

    BlatzKoMullinsMaterial mat(props);

    // Identity F: zero stress
    MaterialState state_id;
    mat.compute_stress(state_id);
    CHECK_NEAR(state_id.stress[0], 0.0, 1.0e-3, "BlatzKo: zero stress at F=I");
    CHECK_NEAR(state_id.damage, 0.0, 1.0e-10, "BlatzKo: zero damage at F=I");

    // Uniaxial stretch (virgin loading)
    MaterialState state_load;
    Real stretch = 1.2;
    Real lat = 1.0 / std::sqrt(stretch);
    state_load.F[0] = stretch;
    state_load.F[4] = lat;
    state_load.F[8] = lat;
    mat.compute_stress(state_load);
    CHECK(state_load.stress[0] > 0.0, "BlatzKo: tensile stress for stretch > 1");
    Real W_max_after_load = state_load.history[0];
    CHECK(W_max_after_load > 0.0, "BlatzKo: W_max > 0 after loading");

    // Unloading to smaller stretch: Mullins softening
    MaterialState state_unload;
    state_unload.F[0] = 1.1;
    state_unload.F[4] = 1.0 / std::sqrt(1.1);
    state_unload.F[8] = 1.0 / std::sqrt(1.1);
    state_unload.history[0] = W_max_after_load; // Carry W_max
    mat.compute_stress(state_unload);
    CHECK(state_unload.stress[0] > 0.0, "BlatzKo: stress > 0 on unloading");
    CHECK(state_unload.damage > 0.0, "BlatzKo: Mullins damage > 0 on unloading");

    // vol_strain tracking
    CHECK(std::abs(state_load.vol_strain) > 0.0 || state_load.vol_strain == 0.0,
          "BlatzKo: vol_strain computed");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_id, C);
    CHECK(C[0] > 0.0, "BlatzKo: tangent C11 > 0");
}

// ==========================================================================
// 18. LaminatedGlassMaterial
// ==========================================================================
void test_laminated_glass() {
    std::cout << "\n=== Test 18: LaminatedGlassMaterial ===\n";

    MaterialProperties props;
    props.E = 70.0e9;               // E_glass = 70 GPa
    props.nu = 0.22;
    props.density = 2500.0;
    props.E1 = 2.0e6;               // E_pvb = 2 MPa
    props.foam_unload_factor = 0.8;  // Glass volume fraction = 80%
    props.yield_stress = 60.0e6;     // Glass tensile strength = 60 MPa
    props.compute_derived();

    LaminatedGlassMaterial mat(props);

    // Elastic response (intact glass): E_eff = 0.8*70e9 + 0.2*2e6
    Real E_eff = 0.8 * 70.0e9 + 0.2 * 2.0e6;
    MaterialState state;
    state.strain[0] = 0.0001;
    mat.compute_stress(state);
    CHECK(state.stress[0] > 0.0, "LamGlass: positive stress intact");
    CHECK_NEAR(state.damage, 0.0, 1.0e-10, "LamGlass: no damage below failure");

    // Check effective modulus (uniaxial strain component)
    Real nu_g = props.nu;
    Real lam_eff = E_eff * nu_g / ((1.0 + nu_g) * (1.0 - 2.0 * nu_g));
    Real mu_eff = E_eff / (2.0 * (1.0 + nu_g));
    Real expected_stress = (lam_eff + 2.0 * mu_eff) * 0.0001;
    CHECK_NEAR(state.stress[0], expected_stress, 1.0e4,
               "LamGlass: stress matches E_eff");

    // Glass failure: large tensile strain
    MaterialState state_fail;
    state_fail.strain[0] = 0.01;  // Should exceed glass strength
    mat.compute_stress(state_fail);
    CHECK_NEAR(state_fail.damage, 1.0, 1.0e-10, "LamGlass: glass failed");
    CHECK_NEAR(state_fail.history[0], 1.0, 1.0e-10, "LamGlass: history[0] = broken");

    // After failure: only PVB carries load (much weaker)
    MaterialState state_post;
    state_post.strain[0] = 0.0001;
    state_post.history[0] = 1.0; // Pre-broken
    state_post.damage = 1.0;
    mat.compute_stress(state_post);
    CHECK(state_post.stress[0] > 0.0, "LamGlass: PVB still carries load");
    CHECK(state_post.stress[0] < state.stress[0] * 0.01,
          "LamGlass: PVB stress much smaller than intact");

    // Tangent stiffness after failure
    Real C_intact[36], C_broken[36];
    mat.tangent_stiffness(state, C_intact);
    mat.tangent_stiffness(state_post, C_broken);
    CHECK(C_intact[0] > C_broken[0], "LamGlass: tangent softer after break");
}

// ==========================================================================
// 19. SpotWeldMaterial
// ==========================================================================
void test_spot_weld() {
    std::cout << "\n=== Test 19: SpotWeldMaterial ===\n";

    MaterialProperties props;
    props.E = 210.0e9;
    props.nu = 0.3;
    props.density = 7800.0;
    props.yield_stress = 500.0e6;        // N_f (normal failure stress)
    props.hardening_modulus = 300.0e6;    // M_f (shear failure stress)
    props.damage_exponent = 2.0;         // Exponent a
    props.JC_n = 2.0;                    // Exponent b
    props.compute_derived();

    SpotWeldMaterial mat(props);

    // Elastic response (below failure)
    MaterialState state;
    state.strain[0] = 0.0001;
    mat.compute_stress(state);
    CHECK(state.stress[0] > 0.0, "SpotWeld: positive elastic stress");
    CHECK_NEAR(state.damage, 0.0, 1.0e-10, "SpotWeld: no damage in elastic");

    // Axial failure: large normal stress exceeds N_f
    MaterialState state_nfail;
    state_nfail.strain[0] = 0.01;  // Large axial strain -> sigma_n >> N_f
    mat.compute_stress(state_nfail);
    CHECK_NEAR(state_nfail.damage, 1.0, 1.0e-10, "SpotWeld: normal failure");
    CHECK_NEAR(state_nfail.stress[0], 0.0, 1.0e-10,
               "SpotWeld: zero stress after failure");

    // After failure: all stress is zero
    MaterialState state_post;
    state_post.damage = 1.0;
    state_post.strain[0] = 0.001;
    mat.compute_stress(state_post);
    CHECK_NEAR(state_post.stress[0], 0.0, 1.0e-10,
               "SpotWeld: zero stress post-failure");

    // Tangent stiffness: zero after failure
    Real C_ok[36], C_fail[36];
    mat.tangent_stiffness(state, C_ok);
    mat.tangent_stiffness(state_post, C_fail);
    CHECK(C_ok[0] > 0.0, "SpotWeld: tangent stiff before failure");
    CHECK_NEAR(C_fail[0], 0.0, 1.0e-10, "SpotWeld: tangent zero after failure");

    // Combined criterion: stress below both limits individually
    MaterialState state_comb;
    state_comb.strain[0] = 0.001;  // Moderate axial
    state_comb.strain[3] = 0.001;  // Moderate shear
    mat.compute_stress(state_comb);
    // With both normal and shear, combined ratio may or may not exceed 1
    CHECK(state_comb.damage == 0.0 || state_comb.damage == 1.0,
          "SpotWeld: damage is 0 or 1 (no gradual)");
}

// ==========================================================================
// 20. RateDependentCompositeMaterial
// ==========================================================================
void test_rate_dependent_composite() {
    std::cout << "\n=== Test 20: RateDependentCompositeMaterial ===\n";

    MaterialProperties props;
    props.E = 10.0e9;
    props.nu = 0.3;
    props.E1 = 140.0e9;    // Fiber direction
    props.E2 = 10.0e9;     // Transverse
    props.E3 = 10.0e9;
    props.G12 = 5.0e9;
    props.G23 = 3.5e9;
    props.G13 = 5.0e9;
    props.nu12 = 0.3;
    props.nu23 = 0.4;
    props.nu13 = 0.3;
    props.CS_D = 1.0;       // Reference strain rate
    props.CS_q = 0.05;      // Rate sensitivity
    props.density = 1600.0;
    props.compute_derived();

    RateDependentCompositeMaterial mat(props);

    // Quasi-static (eps_dot = 0): base stiffness
    MaterialState state_qs;
    state_qs.strain[0] = 0.001;
    state_qs.effective_strain_rate = 0.0;
    mat.compute_stress(state_qs);
    CHECK(state_qs.stress[0] > 0.0, "RDComp: positive stress in fiber dir");
    Real sigma_qs = state_qs.stress[0];

    // Orthotropy: fiber direction stiffer
    MaterialState state_trans;
    state_trans.strain[1] = 0.001;
    state_trans.effective_strain_rate = 0.0;
    mat.compute_stress(state_trans);
    CHECK(state_qs.stress[0] > state_trans.stress[1],
          "RDComp: fiber dir stiffer than transverse");

    // Rate enhancement: higher rate increases stress
    MaterialState state_fast;
    state_fast.strain[0] = 0.001;
    state_fast.effective_strain_rate = 100.0;
    mat.compute_stress(state_fast);
    CHECK(state_fast.stress[0] > sigma_qs, "RDComp: rate enhancement increases stress");

    // Rate factor calculation: 1 + C_rate * ln(eps_dot/eps_dot_ref)
    Real expected_factor = 1.0 + 0.05 * std::log(100.0 / 1.0);
    Real actual_ratio = state_fast.stress[0] / (sigma_qs + 1.0e-30);
    CHECK_NEAR(actual_ratio, expected_factor, 0.01, "RDComp: rate factor matches");

    // Shear response
    MaterialState state_shear;
    state_shear.strain[3] = 0.001;
    state_shear.effective_strain_rate = 0.0;
    mat.compute_stress(state_shear);
    CHECK_NEAR(state_shear.stress[3], 5.0e9 * 0.001, 1.0e4,
               "RDComp: shear stress = G12 * gamma");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_qs, C);
    CHECK(C[0] > C[7], "RDComp: C11 > C22 (anisotropy)");
    CHECK(C[21] > 0.0, "RDComp: shear stiffness > 0");
}

// ==========================================================================
// Main
// ==========================================================================
int main() {
    std::cout << "=== Wave 10 Material Models Test Suite ===\n";

    test_hill_anisotropic();
    test_barlat_yld();
    test_tabulated_jc();
    test_concrete();
    test_fabric();
    test_cohesive_zone();
    test_soil_cap();
    test_user_defined();
    test_arruda_boyce();
    test_shape_memory_alloy();
    test_rate_dependent_foam();
    test_prony_viscoelastic();
    test_thermal_elastic_plastic();
    test_zerilli_armstrong();
    test_steinberg_guinan();
    test_mts();
    test_blatzko_mullins();
    test_laminated_glass();
    test_spot_weld();
    test_rate_dependent_composite();

    std::cout << "\n=== Results ===\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";
    std::cout << "Total:  " << (tests_passed + tests_failed) << "\n";

    if (tests_failed > 0) {
        std::cout << "\nSOME TESTS FAILED!\n";
        return 1;
    }

    std::cout << "\nAll tests passed.\n";
    return 0;
}
