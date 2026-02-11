/**
 * @file material_models_test.cpp
 * @brief Comprehensive test for Wave 1 expanded material models
 */

#include <nexussim/physics/material_models.hpp>
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

// Helper: apply uniaxial tension and return axial stress
static Real uniaxial_test(Material& mat, Real strain_val) {
    MaterialState state;
    state.strain[0] = strain_val;
    state.strain[1] = -0.3 * strain_val; // Approximate lateral
    state.strain[2] = -0.3 * strain_val;
    mat.compute_stress(state);
    return state.stress[0];
}

// Helper: apply uniaxial tension via deformation gradient
static Real uniaxial_F_test(Material& mat, Real stretch) {
    MaterialState state;
    Real nu = 0.3;
    Real lat = 1.0 / std::sqrt(stretch); // Incompressible lateral
    state.F[0] = stretch;
    state.F[4] = lat;
    state.F[8] = lat;
    mat.compute_stress(state);
    return state.stress[0];
}

// ==========================================================================
// Test 1: Orthotropic Elastic
// ==========================================================================
void test_orthotropic() {
    std::cout << "\n=== Test 1: Orthotropic Elastic ===\n";

    MaterialProperties props;
    props.E1 = 140.0e9;  // Carbon fiber direction
    props.E2 = 10.0e9;
    props.E3 = 10.0e9;
    props.G12 = 5.0e9;
    props.G23 = 3.5e9;
    props.G13 = 5.0e9;
    props.nu12 = 0.3;
    props.nu23 = 0.4;
    props.nu13 = 0.3;

    OrthotropicMaterial mat(props);

    // Uniaxial strain in fiber direction
    MaterialState state;
    state.strain[0] = 0.001;  // 0.1% strain in dir 1
    mat.compute_stress(state);

    CHECK(state.stress[0] > 0.0, "Fiber-direction stress is positive");
    CHECK(state.stress[0] > state.stress[1], "Fiber direction stiffer than transverse");

    // Pure shear
    MaterialState shear_state;
    shear_state.strain[3] = 0.001;
    mat.compute_stress(shear_state);
    Real tau_expected = 5.0e9 * 0.001;
    CHECK(std::fabs(shear_state.stress[3] - tau_expected) < 1.0e3,
          "Shear stress matches G12");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state, C);
    CHECK(C[0] > C[7], "C11 > C22 (fiber direction stiffer)");
    CHECK(C[21] > 0.0, "Shear stiffness positive");
}

// ==========================================================================
// Test 2: Mooney-Rivlin Hyperelastic
// ==========================================================================
void test_mooney_rivlin() {
    std::cout << "\n=== Test 2: Mooney-Rivlin Hyperelastic ===\n";

    MaterialProperties props;
    props.density = 1200.0;
    props.E = 6.0e6;
    props.nu = 0.4995;
    props.C10 = 0.8e6;
    props.C01 = 0.2e6;
    props.compute_derived();

    MooneyRivlinMaterial mat(props);

    // No deformation → zero stress
    MaterialState state0;
    mat.compute_stress(state0);
    Real max_s0 = 0.0;
    for (int i = 0; i < 6; ++i) max_s0 = std::max(max_s0, std::fabs(state0.stress[i]));
    CHECK(max_s0 < 1.0, "Zero deformation gives near-zero stress");

    // Uniaxial stretch 10%
    MaterialState state;
    state.F[0] = 1.1;
    Real lat = 1.0 / std::sqrt(1.1);
    state.F[4] = lat;
    state.F[8] = lat;
    mat.compute_stress(state);
    CHECK(state.stress[0] > 0.0, "Tension gives positive stress");

    // Compression
    MaterialState state_c;
    state_c.F[0] = 0.9;
    Real lat_c = 1.0 / std::sqrt(0.9);
    state_c.F[4] = lat_c;
    state_c.F[8] = lat_c;
    mat.compute_stress(state_c);
    CHECK(state_c.stress[0] < 0.0, "Compression gives negative stress");
}

// ==========================================================================
// Test 3: Ogden Hyperelastic
// ==========================================================================
void test_ogden() {
    std::cout << "\n=== Test 3: Ogden Hyperelastic ===\n";

    MaterialProperties props;
    props.density = 1100.0;
    props.E = 4.0e6;
    props.nu = 0.4995;
    props.ogden_mu[0] = 1.5e6;
    props.ogden_alpha[0] = 2.0;
    props.ogden_nterms = 1;
    props.compute_derived();

    OgdenMaterial mat(props);

    // Identity → zero stress
    MaterialState state0;
    mat.compute_stress(state0);
    Real max_s = 0.0;
    for (int i = 0; i < 6; ++i) max_s = std::max(max_s, std::fabs(state0.stress[i]));
    CHECK(max_s < 100.0, "Identity deformation gives near-zero stress");

    // Stretch
    MaterialState state;
    state.F[0] = 1.2;
    state.F[4] = 1.0 / std::sqrt(1.2);
    state.F[8] = 1.0 / std::sqrt(1.2);
    mat.compute_stress(state);
    CHECK(state.stress[0] > 0.0, "Uniaxial stretch gives positive stress");
}

// ==========================================================================
// Test 4: Piecewise-Linear Plasticity
// ==========================================================================
void test_piecewise_linear() {
    std::cout << "\n=== Test 4: Piecewise-Linear Plasticity ===\n";

    MaterialProperties props;
    props.density = 7850.0;
    props.E = 200.0e9;
    props.nu = 0.3;
    props.yield_stress = 250.0e6;
    props.compute_derived();

    // Define yield curve: bilinear hardening then plateau
    TabulatedCurve curve;
    curve.add_point(0.0, 250.0e6);
    curve.add_point(0.05, 400.0e6);
    curve.add_point(0.2, 500.0e6);
    curve.add_point(0.5, 500.0e6);  // Plateau

    PiecewiseLinearMaterial mat(props, curve);

    // Elastic range
    Real sigma_elastic = uniaxial_test(mat, 0.0005);
    CHECK(sigma_elastic > 0.0, "Elastic stress is positive");

    // Plastic range
    MaterialState state;
    state.strain[0] = 0.01; // 1% strain → should yield
    mat.compute_stress(state);
    Real vm = Material::von_mises_stress(state.stress);
    CHECK(vm >= 240.0e6, "Stress exceeds initial yield");
    CHECK(state.plastic_strain > 0.0, "Plastic strain accumulated");

    // Curve interpolation test
    CHECK(std::fabs(curve.evaluate(0.025) - 325.0e6) < 1.0e6,
          "Curve interpolation at eps_p=0.025");
    CHECK(std::fabs(curve.evaluate(0.3) - 500.0e6) < 1.0e6,
          "Curve interpolation in plateau");
}

// ==========================================================================
// Test 5: Tabulated Material
// ==========================================================================
void test_tabulated() {
    std::cout << "\n=== Test 5: Tabulated Stress-Strain ===\n";

    MaterialProperties props;
    props.E = 100.0e9;
    props.nu = 0.3;
    props.compute_derived();

    TabulatedCurve curve;
    curve.add_point(0.0, 0.0);
    curve.add_point(0.001, 100.0e6);
    curve.add_point(0.01, 300.0e6);
    curve.add_point(0.1, 500.0e6);

    TabulatedMaterial mat(props, curve);

    MaterialState state;
    state.strain[0] = 0.005;
    mat.compute_stress(state);
    CHECK(state.stress[0] != 0.0, "Tabulated material produces non-zero stress");
}

// ==========================================================================
// Test 6: Crushable Foam
// ==========================================================================
void test_crushable_foam() {
    std::cout << "\n=== Test 6: Crushable Foam ===\n";

    auto props = MaterialLibrary::get(MaterialName::Foam_EPS);
    props.compute_derived();

    CrushableFoamMaterial mat(props);

    // Compression test (uniaxial, ev = -0.3)
    MaterialState state;
    state.strain[0] = -0.3; // 30% volumetric compression (uniaxial)
    mat.compute_stress(state);
    CHECK(state.stress[0] < 0.0, "Compressive stress is negative");

    // Tension test
    MaterialState state_t;
    state_t.strain[0] = 0.01;
    mat.compute_stress(state_t);
    CHECK(state_t.stress[0] > 0.0, "Tensile stress is positive");

    // Unloading: compress then release (uniaxial)
    MaterialState state_u;
    // First compress to 50% volumetric strain
    state_u.strain[0] = -0.5;
    mat.compute_stress(state_u);
    Real stress_at_50 = state_u.stress[0];

    // Now partially unload to 30%
    state_u.strain[0] = -0.3;
    mat.compute_stress(state_u);
    CHECK(state_u.stress[0] > stress_at_50, "Unloading gives higher (less negative) stress");
}

// ==========================================================================
// Test 7: Honeycomb
// ==========================================================================
void test_honeycomb() {
    std::cout << "\n=== Test 7: Honeycomb ===\n";

    auto props = MaterialLibrary::get(MaterialName::Honeycomb_Al_3003);
    props.compute_derived();

    HoneycombMaterial mat(props);

    // Crush in T-direction (direction 3)
    MaterialState state;
    state.strain[2] = -0.3; // 30% compression in dir 3
    mat.compute_stress(state);
    CHECK(state.stress[2] < 0.0, "T-direction crush gives compressive stress");

    // Orthotropic behavior: in-plane should be weaker
    MaterialState state_ip;
    state_ip.strain[0] = -0.3;
    mat.compute_stress(state_ip);
    CHECK(std::fabs(state_ip.stress[0]) <= std::fabs(state.stress[2]),
          "In-plane crush weaker than T-direction");
}

// ==========================================================================
// Test 8: Viscoelastic
// ==========================================================================
void test_viscoelastic() {
    std::cout << "\n=== Test 8: Viscoelastic (Maxwell) ===\n";

    MaterialProperties props;
    props.density = 1200.0;
    props.E = 10.0e6;
    props.nu = 0.4;
    props.prony_g[0] = 0.3;
    props.prony_tau[0] = 0.01;  // 10 ms
    props.prony_g[1] = 0.2;
    props.prony_tau[1] = 0.1;   // 100 ms
    props.prony_nterms = 2;
    props.compute_derived();

    ViscoelasticMaterial mat(props);

    // Step 1: Apply strain suddenly (small dt, strain goes from 0 to 0.01)
    MaterialState state;
    state.strain[0] = 0.01;
    state.dt = 1.0e-6;  // Very small timestep = sudden application
    mat.compute_stress(state);
    Real stress_initial = state.stress[0];
    CHECK(stress_initial > 0.0, "Instantaneous stress is positive");

    // Steps 2-N: Hold strain constant, let Prony terms relax
    // With tau = 10ms and 100ms, after 1000 steps of 10ms (10s), both terms relax
    for (int i = 0; i < 1000; ++i) {
        state.dt = 0.01;  // 10ms per step
        mat.compute_stress(state);  // Same strain, history carries over
    }
    Real stress_relaxed = state.stress[0];
    CHECK(stress_relaxed > 0.0, "Long-term stress is positive");
    CHECK(stress_relaxed < stress_initial, "Stress relaxes over time (Prony decay)");
}

// ==========================================================================
// Test 9: Cowper-Symonds Rate Dependent
// ==========================================================================
void test_cowper_symonds() {
    std::cout << "\n=== Test 9: Cowper-Symonds Rate-Dependent ===\n";

    MaterialProperties props;
    props.density = 7850.0;
    props.E = 200.0e9;
    props.nu = 0.3;
    props.yield_stress = 250.0e6;
    props.hardening_modulus = 1.0e9;
    props.CS_D = 40.4;    // Mild steel
    props.CS_q = 5.0;
    props.compute_derived();

    CowperSymondsMaterial mat(props);

    // Quasi-static (low rate)
    MaterialState state_qs;
    state_qs.strain[0] = 0.01;
    state_qs.effective_strain_rate = 0.001;
    mat.compute_stress(state_qs);
    Real sigma_qs = Material::von_mises_stress(state_qs.stress);

    // High rate
    MaterialState state_hr;
    state_hr.strain[0] = 0.01;
    state_hr.effective_strain_rate = 1000.0;
    mat.compute_stress(state_hr);
    Real sigma_hr = Material::von_mises_stress(state_hr.stress);

    CHECK(sigma_hr > sigma_qs, "Higher strain rate gives higher stress");
    Real ratio = sigma_hr / (sigma_qs + 1.0);
    CHECK(ratio > 1.1, "Significant rate enhancement at 1000/s");
}

// ==========================================================================
// Test 10: Elastic-Plastic with Failure
// ==========================================================================
void test_elastic_plastic_fail() {
    std::cout << "\n=== Test 10: Elastic-Plastic with Failure ===\n";

    MaterialProperties props;
    props.density = 7850.0;
    props.E = 200.0e9;
    props.nu = 0.3;
    props.yield_stress = 300.0e6;
    props.hardening_modulus = 1.0e9;
    props.failure_plastic_strain = 0.1;  // Fail at 10% plastic strain
    props.compute_derived();

    ElasticPlasticFailMaterial mat(props);

    // Before failure
    MaterialState state1;
    state1.strain[0] = 0.005;
    mat.compute_stress(state1);
    CHECK(state1.damage < 1.0, "No failure at small strain");
    CHECK(state1.stress[0] > 0.0, "Stress is positive before failure");

    // At failure: large strain to exceed plastic strain limit
    MaterialState state2;
    state2.strain[0] = 0.5;
    mat.compute_stress(state2);
    // May or may not fail in one step depending on strain increment
    // Apply incrementally
    MaterialState state_inc;
    for (int i = 0; i < 100; ++i) {
        state_inc.strain[0] = 0.005 * (i + 1);
        mat.compute_stress(state_inc);
        if (state_inc.damage >= 1.0) break;
    }
    CHECK(state_inc.damage >= 1.0, "Material fails at critical plastic strain");
    CHECK(std::fabs(state_inc.stress[0]) < 1.0, "Stress is zero after failure");
}

// ==========================================================================
// Test 11: Rigid Material
// ==========================================================================
void test_rigid() {
    std::cout << "\n=== Test 11: Rigid Material ===\n";

    MaterialProperties props;
    props.density = 7850.0;
    props.E = 200.0e9;
    props.nu = 0.3;
    props.compute_derived();

    RigidMaterial mat(props);

    MaterialState state;
    state.strain[0] = 0.001;
    mat.compute_stress(state);

    // Should be 1000x stiffer than elastic
    ElasticMaterial emat(props);
    MaterialState estate;
    estate.strain[0] = 0.001;
    emat.compute_stress(estate);

    Real ratio = state.stress[0] / (estate.stress[0] + 1.0e-10);
    CHECK(ratio > 500.0, "Rigid material much stiffer than elastic");
}

// ==========================================================================
// Test 12: Null Material
// ==========================================================================
void test_null() {
    std::cout << "\n=== Test 12: Null Material ===\n";

    MaterialProperties props;
    props.density = 1000.0;
    props.E = 2.2e9;
    props.nu = 0.4;
    props.compute_derived();

    NullMaterial mat(props);

    // Apply deviatoric strain → should have zero deviatoric stress
    MaterialState state;
    state.strain[0] = 0.01;
    state.strain[1] = -0.005;
    state.strain[2] = -0.005;
    mat.compute_stress(state);

    // All normal stresses should be equal (hydrostatic only)
    CHECK(std::fabs(state.stress[0] - state.stress[1]) < 1.0,
          "No deviatoric stress difference (null material)");
    CHECK(std::fabs(state.stress[3]) < 1.0, "Zero shear stress");
    CHECK(std::fabs(state.stress[4]) < 1.0, "Zero shear stress (yz)");
    CHECK(std::fabs(state.stress[5]) < 1.0, "Zero shear stress (xz)");
}

// ==========================================================================
// Test 13: Blatz-Ko Foam
// ==========================================================================
void test_foam() {
    std::cout << "\n=== Test 13: Blatz-Ko Foam ===\n";

    MaterialProperties props;
    props.density = 100.0;
    props.E = 1.0e6;
    props.nu = 0.25;
    props.compute_derived();

    FoamMaterial mat(props);

    // Identity → zero stress
    MaterialState state0;
    mat.compute_stress(state0);
    Real max_s = 0.0;
    for (int i = 0; i < 6; ++i) max_s = std::max(max_s, std::fabs(state0.stress[i]));
    CHECK(max_s < 100.0, "Identity gives near-zero stress");

    // Compression
    MaterialState state_c;
    state_c.F[0] = 0.8;
    state_c.F[4] = 1.0;
    state_c.F[8] = 1.0;
    mat.compute_stress(state_c);
    CHECK(state_c.stress[0] < 0.0, "Compression gives negative axial stress");
}

// ==========================================================================
// Test 14: TabulatedCurve Functionality
// ==========================================================================
void test_tabulated_curve() {
    std::cout << "\n=== Test 14: TabulatedCurve ===\n";

    TabulatedCurve curve;
    curve.add_point(0.0, 0.0);
    curve.add_point(1.0, 100.0);
    curve.add_point(2.0, 300.0);
    curve.add_point(5.0, 300.0);

    CHECK(std::fabs(curve.evaluate(0.0) - 0.0) < 1.0e-10, "Exact first point");
    CHECK(std::fabs(curve.evaluate(0.5) - 50.0) < 1.0e-10, "Midpoint interpolation");
    CHECK(std::fabs(curve.evaluate(1.0) - 100.0) < 1.0e-10, "Exact second point");
    CHECK(std::fabs(curve.evaluate(1.5) - 200.0) < 1.0e-10, "Interior interpolation");
    CHECK(std::fabs(curve.evaluate(3.0) - 300.0) < 1.0e-10, "Plateau region");
    CHECK(std::fabs(curve.evaluate(-1.0) - 0.0) < 1.0e-10, "Below range extrapolation");
    CHECK(std::fabs(curve.evaluate(10.0) - 300.0) < 1.0e-10, "Above range extrapolation");

    // Empty curve
    TabulatedCurve empty;
    CHECK(std::fabs(empty.evaluate(1.0)) < 1.0e-10, "Empty curve returns 0");

    // Single point
    TabulatedCurve single;
    single.add_point(1.0, 42.0);
    CHECK(std::fabs(single.evaluate(5.0) - 42.0) < 1.0e-10, "Single point returns value");
}

// ==========================================================================
// Test 15: Material Factory and Type Strings
// ==========================================================================
void test_factory_and_types() {
    std::cout << "\n=== Test 15: Material Factory & Type Strings ===\n";

    // Factory creates base types
    MaterialProperties props;
    props.E = 200.0e9;
    props.nu = 0.3;
    props.yield_stress = 250.0e6;
    props.JC_A = 250.0e6;
    props.compute_derived();

    auto elastic = MaterialFactory::create(physics::MaterialType::Elastic, props);
    CHECK(elastic != nullptr, "Factory creates Elastic");
    CHECK(elastic->type() == physics::MaterialType::Elastic, "Type is Elastic");

    auto plastic = MaterialFactory::create(physics::MaterialType::Plastic, props);
    CHECK(plastic != nullptr, "Factory creates Plastic");

    auto jc = MaterialFactory::create(physics::MaterialType::JohnsonCook, props);
    CHECK(jc != nullptr, "Factory creates JohnsonCook");

    auto neo = MaterialFactory::create(physics::MaterialType::Hyperelastic, props);
    CHECK(neo != nullptr, "Factory creates Hyperelastic");

    // Type strings
    CHECK(MaterialFactory::to_string(physics::MaterialType::Orthotropic) == "Orthotropic",
          "Orthotropic type string");
    CHECK(MaterialFactory::to_string(physics::MaterialType::CrushableFoam) == "CrushableFoam",
          "CrushableFoam type string");
    CHECK(MaterialFactory::to_string(physics::MaterialType::Rigid) == "Rigid",
          "Rigid type string");
}

// ==========================================================================
// Test 16: Material Library Presets
// ==========================================================================
void test_library_presets() {
    std::cout << "\n=== Test 16: Material Library Presets ===\n";

    auto eps = MaterialLibrary::get(MaterialName::Foam_EPS);
    CHECK(eps.density > 0.0 && eps.density < 100.0, "EPS foam density reasonable");
    CHECK(eps.foam_E_crush > 0.0, "EPS has crush stress");

    auto al_hc = MaterialLibrary::get(MaterialName::Honeycomb_Al_3003);
    CHECK(al_hc.E3 > al_hc.E1, "Honeycomb T-direction stiffer than in-plane");

    auto neo_rubber = MaterialLibrary::get(MaterialName::Rubber_Neoprene);
    CHECK(neo_rubber.C10 > 0.0, "Neoprene has Mooney-Rivlin C10");
    CHECK(neo_rubber.nu > 0.49, "Rubber is nearly incompressible");

    CHECK(MaterialLibrary::to_string(MaterialName::Foam_EPS) == "EPS Foam",
          "EPS foam name string");
    CHECK(MaterialLibrary::to_string(MaterialName::Honeycomb_Nomex) == "Nomex Honeycomb",
          "Nomex honeycomb name string");
}

// ==========================================================================
// Main
// ==========================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "NexusSim Wave 1: Material Models Test\n";
    std::cout << "========================================\n";

    test_orthotropic();
    test_mooney_rivlin();
    test_ogden();
    test_piecewise_linear();
    test_tabulated();
    test_crushable_foam();
    test_honeycomb();
    test_viscoelastic();
    test_cowper_symonds();
    test_elastic_plastic_fail();
    test_rigid();
    test_null();
    test_foam();
    test_tabulated_curve();
    test_factory_and_types();
    test_library_presets();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    std::cout << "========================================\n";

    return tests_failed > 0 ? 1 : 0;
}
