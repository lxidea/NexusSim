/**
 * @file material_wave32_test.cpp
 * @brief Comprehensive test for Wave 32 material models (20 constitutive models)
 */

#include <nexussim/physics/material_wave32.hpp>
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

// Helper: make foam properties
static MaterialProperties make_foam_props() {
    MaterialProperties props;
    props.E = 50.0e6;
    props.nu = 0.1;
    props.density = 80.0;
    props.yield_stress = 2.0e6;
    props.hardening_modulus = 1.0e6;
    props.compute_derived();
    return props;
}

// Helper: make aluminum properties
static MaterialProperties make_aluminum_props() {
    MaterialProperties props;
    props.E = 70.0e9;
    props.nu = 0.33;
    props.density = 2700.0;
    props.yield_stress = 324.0e6;
    props.hardening_modulus = 500.0e6;
    props.specific_heat = 910.0;
    props.compute_derived();
    return props;
}

// ==========================================================================
// 1. OrthotropicHillMaterial
// ==========================================================================
void test_1_orthotropic_hill() {
    std::cout << "\n=== Test 1: OrthotropicHillMaterial ===\n";

    auto props = make_steel_props();
    OrthotropicHillMaterial mat(props, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 250.0e6, 1.0e9);

    // Elastic range: small strain
    MaterialState state_e;
    state_e.strain[0] = 0.0001;
    mat.compute_stress(state_e);
    CHECK(state_e.stress[0] > 0.0, "OrthoHill: positive stress in tension");
    CHECK_NEAR(state_e.history[32], 0.0, 1.0e-10, "OrthoHill: no plasticity in elastic range");

    // Yield: large uniaxial strain
    MaterialState state_y;
    state_y.strain[0] = 0.01;
    mat.compute_stress(state_y);
    CHECK(state_y.history[32] > 0.0, "OrthoHill: plastic strain after yield");

    // Hardening: yield stress increases
    Real sy = mat.current_yield(state_y);
    CHECK(sy > 250.0e6, "OrthoHill: hardened yield > initial yield");

    // Hill equivalent stress
    Real sigma_hill = mat.hill_equivalent(state_y);
    CHECK(sigma_hill > 0.0, "OrthoHill: Hill equivalent > 0");

    // Symmetry: equal F,G,H should give isotropic-like response
    MaterialState state_s1, state_s2;
    state_s1.strain[0] = 0.005;
    state_s2.strain[1] = 0.005;
    mat.compute_stress(state_s1);
    mat.compute_stress(state_s2);
    CHECK_NEAR(state_s1.stress[0], state_s2.stress[1], state_s1.stress[0] * 0.01,
               "OrthoHill: isotropic when F=G=H");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_e, C);
    CHECK(C[0] > 0.0, "OrthoHill: tangent C11 > 0");
    CHECK(C[21] > 0.0, "OrthoHill: tangent C44 > 0");

    // Anisotropic: different F,G,H
    OrthotropicHillMaterial mat2(props, 0.3, 0.7, 0.5, 1.5, 1.5, 1.5, 250.0e6, 1.0e9);
    MaterialState state_a1, state_a2;
    state_a1.strain[0] = 0.01;
    state_a2.strain[1] = 0.01;
    mat2.compute_stress(state_a1);
    mat2.compute_stress(state_a2);
    CHECK(std::abs(state_a1.stress[0] - state_a2.stress[1]) > 1.0e3,
          "OrthoHill: anisotropic response with different F,G,H");
}

// ==========================================================================
// 2. VegterYieldMaterial
// ==========================================================================
void test_2_vegter_yield() {
    std::cout << "\n=== Test 2: VegterYieldMaterial ===\n";

    auto props = make_steel_props();

    // Create yield points
    VegterYieldMaterial::YieldPoint pts[8];
    Real pi = 3.14159265358979323846;
    for (int i = 0; i < 8; ++i) {
        pts[i].theta = i * pi / 4.0;
        pts[i].sigma = 250.0e6 * (1.0 + 0.1 * std::sin(2.0 * pts[i].theta));
    }

    VegterYieldMaterial mat(props, pts, 8, 1.0e9);
    CHECK(mat.num_points() == 8, "Vegter: 8 yield points stored");

    // Elastic range
    MaterialState state_e;
    state_e.strain[0] = 0.0001;
    mat.compute_stress(state_e);
    CHECK(state_e.stress[0] > 0.0, "Vegter: elastic stress positive");
    CHECK_NEAR(state_e.history[32], 0.0, 1.0e-10, "Vegter: no plastic strain in elastic");

    // Yield
    MaterialState state_y;
    state_y.strain[0] = 0.01;
    mat.compute_stress(state_y);
    CHECK(state_y.history[32] > 0.0, "Vegter: plastic strain after yield");

    // Interpolation
    Real sy_0 = mat.interpolate_yield(0.0);
    Real sy_pi4 = mat.interpolate_yield(pi / 4.0);
    CHECK(sy_0 > 0.0, "Vegter: interpolated yield at 0 > 0");
    CHECK(sy_pi4 > 0.0, "Vegter: interpolated yield at pi/4 > 0");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_e, C);
    CHECK(C[0] > 0.0, "Vegter: tangent C11 > 0");

    // Default (circular) yield surface
    VegterYieldMaterial mat_default(props);
    MaterialState state_d;
    state_d.strain[0] = 0.001;
    mat_default.compute_stress(state_d);
    CHECK(state_d.stress[0] > 0.0, "Vegter: default model produces stress");
}

// ==========================================================================
// 3. MarlowHyperelasticMaterial
// ==========================================================================
void test_3_marlow_hyperelastic() {
    std::cout << "\n=== Test 3: MarlowHyperelasticMaterial ===\n";

    MaterialProperties props;
    props.E = 10.0e6;
    props.nu = 0.45;
    props.density = 1100.0;
    props.compute_derived();

    // Rubber-like test data
    MarlowHyperelasticMaterial::StressStrainPoint data[6];
    data[0] = {0.0, 0.0};
    data[1] = {0.1, 0.8e6};
    data[2] = {0.3, 1.5e6};
    data[3] = {0.5, 2.5e6};
    data[4] = {1.0, 5.0e6};
    data[5] = {2.0, 15.0e6};

    MarlowHyperelasticMaterial mat(props, data, 6);
    CHECK(mat.num_data_points() == 6, "Marlow: 6 data points stored");

    // Stress interpolation test
    Real s_at_0_2 = mat.interpolate_stress(0.2);
    CHECK(s_at_0_2 > 0.8e6 && s_at_0_2 < 1.5e6, "Marlow: interpolated stress in range");

    // Compute full 3D stress
    MaterialState state_1;
    state_1.strain[0] = 0.1;
    mat.compute_stress(state_1);
    CHECK(state_1.stress[0] > 0.0, "Marlow: tensile stress positive");

    // Track max stretch
    CHECK(state_1.history[32] >= 1.1, "Marlow: max stretch tracked");

    // Strain energy
    CHECK(state_1.history[33] > 0.0, "Marlow: strain energy positive");

    // Neo-Hookean fallback (no data)
    MarlowHyperelasticMaterial mat_nh(props);
    MaterialState state_nh;
    state_nh.strain[0] = 0.05;
    mat_nh.compute_stress(state_nh);
    CHECK(state_nh.stress[0] > 0.0, "Marlow: Neo-Hookean fallback works");

    // Tangent
    Real C[36];
    mat.tangent_stiffness(state_1, C);
    CHECK(C[0] > 0.0, "Marlow: tangent C11 > 0");

    // Large strain data-driven
    Real s_at_1_5 = mat.interpolate_stress(1.5);
    CHECK(s_at_1_5 > 5.0e6 && s_at_1_5 < 15.0e6, "Marlow: interpolation at large strain");
}

// ==========================================================================
// 4. DeshpandeFleckMaterial
// ==========================================================================
void test_4_deshpande_fleck() {
    std::cout << "\n=== Test 4: DeshpandeFleckMaterial ===\n";

    auto props = make_foam_props();
    DeshpandeFleckMaterial mat(props, 2.0e6, 1.5, 0.7);

    CHECK_NEAR(mat.plateau_stress(), 2.0e6, 1.0, "DeshFleck: plateau stress stored");
    CHECK_NEAR(mat.alpha(), 1.5, 1.0e-10, "DeshFleck: alpha stored");

    // Elastic range: small strain
    MaterialState state_e;
    state_e.strain[0] = -0.0001;
    state_e.strain[1] = -0.0001;
    state_e.strain[2] = -0.0001;
    mat.compute_stress(state_e);
    CHECK_NEAR(state_e.history[32], 0.0, 1.0e-10, "DeshFleck: elastic in small strain");

    // Hydrostatic compression: triggers yield (alpha > 0)
    MaterialState state_h;
    state_h.strain[0] = -0.1;
    state_h.strain[1] = -0.1;
    state_h.strain[2] = -0.1;
    mat.compute_stress(state_h);
    CHECK(state_h.history[32] > 0.0, "DeshFleck: plastic under hydrostatic compression");

    // DF equivalent stress
    Real df_eq = mat.df_equivalent(state_h.stress);
    CHECK(df_eq > 0.0, "DeshFleck: DF equivalent > 0");

    // Shear also yields
    MaterialState state_s;
    state_s.strain[3] = 0.1;
    mat.compute_stress(state_s);
    CHECK(state_s.history[32] > 0.0, "DeshFleck: plastic under shear");

    // Tangent
    Real C[36];
    mat.tangent_stiffness(state_e, C);
    CHECK(C[0] > 0.0, "DeshFleck: tangent C11 > 0");

    // Densification at high volumetric strain
    MaterialState state_d;
    state_d.strain[0] = -0.25;
    state_d.strain[1] = -0.25;
    state_d.strain[2] = -0.25;
    mat.compute_stress(state_d);
    Real p_dense = -(state_d.stress[0] + state_d.stress[1] + state_d.stress[2]) / 3.0;
    Real p_mod = -(state_h.stress[0] + state_h.stress[1] + state_h.stress[2]) / 3.0;
    CHECK(p_dense > p_mod, "DeshFleck: densification increases pressure");
}

// ==========================================================================
// 5. ModifiedLaDevezeMaterial
// ==========================================================================
void test_5_modified_ladeveze() {
    std::cout << "\n=== Test 5: ModifiedLaDevezeMaterial ===\n";

    auto props = make_composite_props();
    ModifiedLaDevezeMaterial mat(props, 140.0e9, 10.0e9, 5.0e9, 50.0e6, 80.0e6, 0.99);

    // Small elastic shear
    MaterialState state_e;
    state_e.strain[3] = 0.0001;
    mat.compute_stress(state_e);
    CHECK(state_e.stress[3] > 0.0, "LaDeveze: shear stress positive");
    CHECK_NEAR(mat.shear_damage(state_e), 0.0, 1.0e-6, "LaDeveze: no shear damage at small strain");

    // Shear damage: large shear strain
    MaterialState state_s;
    state_s.strain[3] = 0.05;
    mat.compute_stress(state_s);
    Real d12 = mat.shear_damage(state_s);
    CHECK(d12 > 0.0, "LaDeveze: shear damage at large shear strain");

    // Transverse tensile damage
    MaterialState state_t;
    state_t.strain[1] = 0.02;
    mat.compute_stress(state_t);
    Real d22 = mat.transverse_damage(state_t);
    CHECK(d22 > 0.0, "LaDeveze: transverse damage at large transverse strain");

    // Fiber failure
    Real eps_fiber = 50.0e6 / 140.0e9 * 10.0 * 1.5; // Well above fiber failure
    MaterialState state_f;
    state_f.strain[0] = eps_fiber;
    mat.compute_stress(state_f);
    CHECK(mat.fiber_failed(state_f), "LaDeveze: fiber failure at large fiber strain");

    // No fiber damage at small strain
    MaterialState state_nf;
    state_nf.strain[0] = 0.001;
    mat.compute_stress(state_nf);
    CHECK(!mat.fiber_failed(state_nf), "LaDeveze: no fiber failure at small strain");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_e, C);
    CHECK(C[0] > 0.0, "LaDeveze: tangent C11 > 0");
    CHECK(C[21] > 0.0, "LaDeveze: tangent C44 (shear) > 0");

    // Damage reduces stress
    MaterialState state_d1, state_d2;
    state_d1.strain[3] = 0.01;
    mat.compute_stress(state_d1);
    state_d2.strain[3] = 0.01;
    state_d2.history[32] = 0.5; // pre-existing shear damage
    mat.compute_stress(state_d2);
    CHECK(std::abs(state_d2.stress[3]) < std::abs(state_d1.stress[3]),
          "LaDeveze: damage reduces shear stress");
}

// ==========================================================================
// 6. CDPM2ConcreteMaterial
// ==========================================================================
void test_6_cdpm2_concrete() {
    std::cout << "\n=== Test 6: CDPM2ConcreteMaterial ===\n";

    auto props = make_concrete_props();
    CDPM2ConcreteMaterial mat(props, 30.0e6, 3.0e6, 100.0, 15000.0, 0.002, 0.5);

    CHECK_NEAR(mat.fc(), 30.0e6, 1.0, "CDPM2: fc stored");
    CHECK_NEAR(mat.ft(), 3.0e6, 1.0, "CDPM2: ft stored");

    // Elastic compression
    MaterialState state_ec;
    state_ec.strain[0] = -0.0001;
    state_ec.strain[1] = -0.0001;
    state_ec.strain[2] = -0.0001;
    mat.compute_stress(state_ec);
    CHECK(state_ec.stress[0] < 0.0, "CDPM2: compressive stress negative");
    CHECK_NEAR(mat.compressive_damage(state_ec), 0.0, 1.0e-10,
               "CDPM2: no compressive damage in elastic");

    // Tensile damage
    MaterialState state_t;
    state_t.strain[0] = 0.001;
    mat.compute_stress(state_t);
    Real d_t = mat.tensile_damage(state_t);
    CHECK(d_t > 0.0, "CDPM2: tensile damage at large tensile strain");

    // Shear-dominated with light compression: triggers DP yield
    // sqJ2 + alpha*I1 > k_dp requires large deviatoric with moderate I1
    MaterialState state_lc;
    state_lc.strain[0] = -0.002;
    state_lc.strain[3] = 0.005;
    mat.compute_stress(state_lc);
    CHECK(state_lc.history[33] > 0.0, "CDPM2: compressive plastic strain");

    // Damage reduces stiffness
    Real C_ud[36], C_d[36];
    MaterialState state_und;
    mat.tangent_stiffness(state_und, C_ud);
    MaterialState state_dg;
    state_dg.history[34] = 0.5; // tensile damage
    state_dg.history[35] = 0.3; // compressive damage
    mat.tangent_stiffness(state_dg, C_d);
    CHECK(C_d[0] < C_ud[0], "CDPM2: damage reduces tangent stiffness");

    // Overall damage
    mat.compute_stress(state_t);
    CHECK(state_t.damage > 0.0, "CDPM2: overall damage > 0 after tensile loading");

    // Plastic strain storage (with deviatoric loading, plastic strain accumulates in some component)
    bool has_eps_p = false;
    for (int i = 0; i < 6; ++i) {
        if (state_lc.history[36 + i] != 0.0) has_eps_p = true;
    }
    CHECK(has_eps_p, "CDPM2: plastic strain stored in history[36..41]");
}

// ==========================================================================
// 7. JHConcreteMaterial
// ==========================================================================
void test_7_jh_concrete() {
    std::cout << "\n=== Test 7: JHConcreteMaterial ===\n";

    auto props = make_concrete_props();
    JHConcreteMaterial mat(props, 48.0e6, 4.0e6, 0.79, 1.60, 0.007, 0.61, 7.0);

    CHECK_NEAR(mat.fc(), 48.0e6, 1.0, "JHConcrete: fc stored");
    CHECK_NEAR(mat.ft(), 4.0e6, 1.0, "JHConcrete: ft stored");

    // Small elastic load
    MaterialState state_e;
    state_e.strain[0] = -0.0001;
    mat.compute_stress(state_e);
    CHECK_NEAR(mat.jh_damage(state_e), 0.0, 1.0e-10, "JHConcrete: no damage in elastic");

    // Large compression with deviatoric: should damage
    MaterialState state_c;
    state_c.strain[0] = -0.01;
    state_c.strain[1] = -0.003;
    state_c.strain[2] = -0.003;
    state_c.strain[3] = 0.005;
    mat.compute_stress(state_c);
    Real D = mat.jh_damage(state_c);
    CHECK(D > 0.0, "JHConcrete: damage under large compression with shear");

    // Pressure state
    Real P = mat.pressure_state(state_c);
    CHECK(P > 0.0, "JHConcrete: positive pressure in compression");

    // Rate effect: higher rate should increase strength
    MaterialState state_r1, state_r2;
    state_r1.strain[3] = 0.01;
    state_r1.effective_strain_rate = 1.0;
    mat.compute_stress(state_r1);

    state_r2.strain[3] = 0.01;
    state_r2.effective_strain_rate = 1000.0;
    mat.compute_stress(state_r2);
    CHECK(std::abs(state_r2.stress[3]) >= std::abs(state_r1.stress[3]) * 0.99,
          "JHConcrete: rate enhancement");

    // Tangent with damage
    Real C[36];
    MaterialState state_d;
    state_d.history[32] = 0.5;
    mat.tangent_stiffness(state_d, C);
    CHECK(C[0] > 0.0, "JHConcrete: damaged tangent > 0");
    CHECK(C[0] < 30.0e9, "JHConcrete: damaged tangent < undamaged");

    // Damage bounded [0, 1]
    MaterialState state_ex;
    state_ex.strain[0] = -0.1;
    state_ex.strain[1] = -0.1;
    state_ex.strain[2] = -0.1;
    mat.compute_stress(state_ex);
    CHECK(mat.jh_damage(state_ex) <= 1.0, "JHConcrete: damage bounded <= 1");
}

// ==========================================================================
// 8. EnhancedCompositeMaterial
// ==========================================================================
void test_8_enhanced_composite() {
    std::cout << "\n=== Test 8: EnhancedCompositeMaterial ===\n";

    auto props = make_composite_props();
    EnhancedCompositeMaterial mat(props, 230.0e9, 3.5e9, 0.6, 0.4,
                                   3500.0e6, 1500.0e6, 80.0e6, 200.0e6);

    // Fiber direction stiffness
    MaterialState state_f;
    state_f.strain[0] = 0.001;
    mat.compute_stress(state_f);
    CHECK(state_f.stress[0] > 100.0e6, "EnhComp: fiber direction stiff");
    CHECK_NEAR(mat.fiber_damage(state_f), 0.0, 1.0e-10, "EnhComp: no fiber damage at small strain");

    // Matrix direction (transverse)
    MaterialState state_m;
    state_m.strain[1] = 0.001;
    mat.compute_stress(state_m);
    CHECK(state_m.stress[1] > 0.0, "EnhComp: transverse stress positive");
    CHECK(state_m.stress[1] < state_f.stress[0], "EnhComp: transverse < fiber");

    // Fiber failure
    Real eps_fiber_fail = 3500.0e6 / (230.0e9 * 0.6) + 0.001;
    MaterialState state_ff;
    state_ff.strain[0] = eps_fiber_fail;
    mat.compute_stress(state_ff);
    CHECK(mat.fiber_damage(state_ff) > 0.9, "EnhComp: fiber failure at critical strain");

    // Matrix failure
    Real eps_mat_fail = 80.0e6 / (3.5e9 * 0.4) + 0.01;
    MaterialState state_mf;
    state_mf.strain[1] = eps_mat_fail;
    mat.compute_stress(state_mf);
    CHECK(mat.matrix_damage(state_mf) > 0.9, "EnhComp: matrix failure at critical strain");

    // Tangent stiffness: fiber direction should be dominant
    Real C[36];
    MaterialState state_0;
    mat.tangent_stiffness(state_0, C);
    CHECK(C[0] > C[7], "EnhComp: C11 > C22 (fiber > matrix)");
    CHECK(C[0] > 100.0e9, "EnhComp: fiber stiffness > 100 GPa");

    // No damage in compression fiber direction at moderate strain
    MaterialState state_comp;
    state_comp.strain[0] = -0.001;
    mat.compute_stress(state_comp);
    CHECK(mat.fiber_damage(state_comp) < 0.01, "EnhComp: no fiber damage in small compression");

    // Max strain tracked
    mat.compute_stress(state_f);
    CHECK(state_f.history[34] >= 0.001, "EnhComp: max fiber strain tracked");
}

// ==========================================================================
// 9. GranularMaterial
// ==========================================================================
void test_9_granular() {
    std::cout << "\n=== Test 9: GranularMaterial ===\n";

    MaterialProperties props;
    props.E = 50.0e6;
    props.nu = 0.3;
    props.density = 1800.0;
    props.yield_stress = 200.0e3;
    props.hardening_modulus = 1.0e6;
    props.compute_derived();

    GranularMaterial mat(props, 30.0, 15.0, 0.1, 0.001);

    CHECK(mat.friction_alpha() > 0.0, "Granular: friction alpha > 0");

    // Elastic range
    MaterialState state_e;
    state_e.strain[0] = -0.00001;
    state_e.strain[1] = -0.00001;
    state_e.strain[2] = -0.00001;
    mat.compute_stress(state_e);
    CHECK_NEAR(state_e.history[32], 0.0, 1.0e-10, "Granular: elastic in small strain");

    // Confined shear: yield
    MaterialState state_s;
    state_s.strain[0] = -0.001; // Confinement
    state_s.strain[1] = -0.001;
    state_s.strain[2] = -0.001;
    state_s.strain[3] = 0.01;   // Shear
    mat.compute_stress(state_s);
    CHECK(state_s.history[32] > 0.0, "Granular: plastic strain under shear");

    // Rolling resistance dissipation
    CHECK(mat.rolling_dissipation(state_s) >= 0.0, "Granular: rolling dissipation >= 0");

    // Tension cutoff: limit tensile stress
    MaterialState state_t;
    state_t.strain[0] = 0.01;
    state_t.strain[1] = 0.01;
    state_t.strain[2] = 0.01;
    mat.compute_stress(state_t);
    Real p_t = (state_t.stress[0] + state_t.stress[1] + state_t.stress[2]) / 3.0;
    CHECK(p_t < 1.0e6, "Granular: tension cutoff limits tensile pressure");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_e, C);
    CHECK(C[0] > 0.0, "Granular: tangent C11 > 0");

    // Hardening increases strength
    MaterialState state_h1, state_h2;
    state_h1.strain[3] = 0.02;
    state_h1.strain[0] = -0.005;
    state_h1.strain[1] = -0.005;
    state_h1.strain[2] = -0.005;
    mat.compute_stress(state_h1);
    state_h2.strain[3] = 0.02;
    state_h2.strain[0] = -0.005;
    state_h2.strain[1] = -0.005;
    state_h2.strain[2] = -0.005;
    state_h2.history[32] = 0.01; // Pre-existing plastic strain
    mat.compute_stress(state_h2);
    CHECK(std::abs(state_h2.stress[3]) >= std::abs(state_h1.stress[3]) * 0.99,
          "Granular: hardening at least maintains stress");
}

// ==========================================================================
// 10. ViscousFoamMaterial
// ==========================================================================
void test_10_viscous_foam() {
    std::cout << "\n=== Test 10: ViscousFoamMaterial ===\n";

    auto props = make_foam_props();
    ViscousFoamMaterial mat(props, 1.0e6, 0.8, 0.05, 0.1);

    CHECK_NEAR(mat.plateau_modulus(), 1.0e6, 1.0, "ViscFoam: plateau modulus stored");

    // Small compression
    MaterialState state_c;
    state_c.strain[0] = -0.01;
    state_c.strain[1] = -0.01;
    state_c.strain[2] = -0.01;
    mat.compute_stress(state_c);
    CHECK(state_c.stress[0] < 0.0 || state_c.stress[1] < 0.0,
          "ViscFoam: compressive stress");

    // Rate effect: higher rate = higher stress
    MaterialState state_r1, state_r2;
    state_r1.strain[0] = -0.05;
    state_r1.strain[1] = -0.05;
    state_r1.strain[2] = -0.05;
    state_r1.effective_strain_rate = 1.0;
    mat.compute_stress(state_r1);

    state_r2.strain[0] = -0.05;
    state_r2.strain[1] = -0.05;
    state_r2.strain[2] = -0.05;
    state_r2.effective_strain_rate = 100.0;
    mat.compute_stress(state_r2);
    Real p1 = -(state_r1.stress[0] + state_r1.stress[1] + state_r1.stress[2]) / 3.0;
    Real p2 = -(state_r2.stress[0] + state_r2.stress[1] + state_r2.stress[2]) / 3.0;
    CHECK(p2 > p1, "ViscFoam: rate enhancement increases pressure");

    // Unloading detection
    MaterialState state_u;
    state_u.strain[0] = -0.05;
    state_u.strain[1] = -0.05;
    state_u.strain[2] = -0.05;
    mat.compute_stress(state_u); // Load
    Real max_s = mat.max_strain(state_u);
    CHECK(max_s > 0.0, "ViscFoam: max strain tracked");

    // Simulate unloading: use smaller strain with existing history
    state_u.strain[0] = -0.02;
    state_u.strain[1] = -0.02;
    state_u.strain[2] = -0.02;
    mat.compute_stress(state_u);
    CHECK(mat.is_unloading(state_u), "ViscFoam: unloading detected");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_c, C);
    CHECK(C[0] > 0.0, "ViscFoam: tangent C11 > 0");

    // Unloading tangent is softer
    Real C_u[36];
    MaterialState state_ul;
    state_ul.history[33] = 1.0; // unloading flag
    mat.tangent_stiffness(state_ul, C_u);
    CHECK(C_u[0] < C[0], "ViscFoam: unloading tangent softer");
}

// ==========================================================================
// 11. FabricNLMaterial
// ==========================================================================
void test_11_fabric_nl() {
    std::cout << "\n=== Test 11: FabricNLMaterial ===\n";

    MaterialProperties props;
    props.E = 1.0e9;
    props.nu = 0.3;
    props.density = 800.0;
    props.compute_derived();

    FabricNLMaterial mat(props, 1.0e9, 0.5e9, 50.0e9, 0.5);

    CHECK_NEAR(mat.lock_angle(), 0.5, 1.0e-10, "FabricNL: lock angle stored");

    // Tension in warp direction
    MaterialState state_w;
    state_w.strain[0] = 0.01;
    mat.compute_stress(state_w);
    CHECK(state_w.stress[0] > 0.0, "FabricNL: warp tension positive");

    // No stress in compression (fabric)
    MaterialState state_c;
    state_c.strain[0] = -0.01;
    mat.compute_stress(state_c);
    CHECK_NEAR(state_c.stress[0], 0.0, 1.0e-5, "FabricNL: no compression resistance in warp");

    // Shear below locking: low stiffness
    MaterialState state_s1;
    state_s1.strain[3] = 0.1;
    mat.compute_stress(state_s1);
    Real tau_low = state_s1.stress[3];

    // Shear above locking: high stiffness
    MaterialState state_s2;
    state_s2.strain[3] = 0.8;
    mat.compute_stress(state_s2);
    Real tau_high = state_s2.stress[3];
    CHECK(tau_high / (0.8 + 1.0e-30) > tau_low / (0.1 + 1.0e-30),
          "FabricNL: locking increases shear stiffness");

    // Locking detection
    CHECK(mat.is_locked(state_s2), "FabricNL: fabric locked at large shear");
    CHECK(!mat.is_locked(state_s1), "FabricNL: fabric not locked at small shear");

    // Max shear angle tracked
    CHECK(mat.max_shear_angle(state_s2) >= 0.8, "FabricNL: max shear angle tracked");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_w, C);
    CHECK(C[0] > 0.0, "FabricNL: tangent C11 > 0");

    // Weft direction
    MaterialState state_weft;
    state_weft.strain[1] = 0.01;
    mat.compute_stress(state_weft);
    CHECK(state_weft.stress[1] > 0.0, "FabricNL: weft tension positive");
    CHECK(state_weft.stress[1] < state_w.stress[0], "FabricNL: weft < warp (lower modulus)");
}

// ==========================================================================
// 12. ARUPAdhesiveMaterial
// ==========================================================================
void test_12_arup_adhesive() {
    std::cout << "\n=== Test 12: ARUPAdhesiveMaterial ===\n";

    MaterialProperties props;
    props.E = 3.0e9;
    props.nu = 0.35;
    props.density = 1200.0;
    props.compute_derived();

    // Use consistent params: GIc = 0.5*sigma_max*delta_nf => delta_nf = 2*GIc/sigma_max
    // For delta_n0 = 1e-4, sigma_max = 30 MPa, GIc needs to be >> 0.5*sigma*delta_n0 = 1500
    Real delta_n0 = 1.0e-4, delta_s0 = 2.0e-4;
    Real sig_n = 30.0e6, tau_s = 40.0e6;
    Real GIc_val = 15000.0, GIIc_val = 30000.0;  // Large enough that D < 1 in linear region
    ARUPAdhesiveMaterial mat(props, GIc_val, GIIc_val, sig_n, tau_s, delta_n0, delta_s0);

    CHECK_NEAR(mat.GIc(), GIc_val, 1.0e-10, "ARUPAdh: GIc stored");
    CHECK_NEAR(mat.GIIc(), GIIc_val, 1.0e-10, "ARUPAdh: GIIc stored");

    // Normal opening: in linear rise region (delta < delta_n0)
    MaterialState state_n;
    state_n.strain[2] = delta_n0 * 0.5;
    mat.compute_stress(state_n);
    CHECK(state_n.stress[2] > 0.0, "ARUPAdh: normal traction positive");

    // Normal at peak
    MaterialState state_peak;
    state_peak.strain[2] = delta_n0;
    mat.compute_stress(state_peak);
    CHECK_NEAR(state_peak.stress[2], sig_n, sig_n * 0.2,
               "ARUPAdh: near peak normal traction at delta_n0");

    // Normal softening (well past peak and failure opening)
    Real delta_nf = 2.0 * GIc_val / (sig_n);
    MaterialState state_soft;
    state_soft.strain[2] = delta_nf * 0.8;
    mat.compute_stress(state_soft);
    CHECK(state_soft.stress[2] < state_peak.stress[2],
          "ARUPAdh: softening after peak");

    // Shear traction
    MaterialState state_s;
    state_s.strain[3] = delta_s0 * 0.5;
    mat.compute_stress(state_s);
    CHECK(state_s.stress[3] > 0.0, "ARUPAdh: shear traction positive");

    // Compression: penalty
    MaterialState state_comp;
    state_comp.strain[2] = -delta_n0;
    mat.compute_stress(state_comp);
    CHECK(state_comp.stress[2] < 0.0, "ARUPAdh: compression penalty");

    // Mixed mode damage: use openings well beyond failure
    MaterialState state_mm;
    state_mm.strain[2] = delta_nf * 0.9;
    state_mm.strain[3] = delta_s0 * 2.0;
    mat.compute_stress(state_mm);
    CHECK(mat.adhesive_damage(state_mm) > 0.0, "ARUPAdh: mixed-mode damage");

    // Mode I and II energies
    CHECK(mat.mode_I_energy(state_mm) >= 0.0, "ARUPAdh: mode I energy >= 0");
    CHECK(mat.mode_II_energy(state_mm) >= 0.0, "ARUPAdh: mode II energy >= 0");

    // Tangent stiffness (undamaged state)
    Real C[36];
    MaterialState state_und;
    mat.tangent_stiffness(state_und, C);
    CHECK(C[14] > 0.0, "ARUPAdh: normal stiffness > 0");
}

// ==========================================================================
// 13. FoamDuboisMaterial
// ==========================================================================
void test_13_foam_dubois() {
    std::cout << "\n=== Test 13: FoamDuboisMaterial ===\n";

    auto props = make_foam_props();
    FoamDuboisMaterial mat(props, 2.0e6, 0.7, 0.1, 0.05, 0.3);

    CHECK_NEAR(mat.sigma_p(), 2.0e6, 1.0, "FoamDubois: sigma_p stored");

    // Compression
    MaterialState state_c;
    state_c.strain[0] = -0.05;
    state_c.strain[1] = -0.05;
    state_c.strain[2] = -0.05;
    mat.compute_stress(state_c);
    Real p = -(state_c.stress[0] + state_c.stress[1] + state_c.stress[2]) / 3.0;
    CHECK(p > 0.0, "FoamDubois: positive pressure in compression");

    // Plateau stress enhanced by rate
    MaterialState state_r;
    state_r.strain[0] = -0.05;
    state_r.strain[1] = -0.05;
    state_r.strain[2] = -0.05;
    state_r.effective_strain_rate = 100.0;
    mat.compute_stress(state_r);
    Real plateau_r = mat.current_plateau(state_r);
    CHECK(plateau_r > 2.0e6, "FoamDubois: rate-enhanced plateau > base");

    // Densification at high strain
    MaterialState state_d;
    state_d.strain[0] = -0.2;
    state_d.strain[1] = -0.2;
    state_d.strain[2] = -0.2;
    mat.compute_stress(state_d);
    Real p_d = -(state_d.stress[0] + state_d.stress[1] + state_d.stress[2]) / 3.0;
    CHECK(p_d > p, "FoamDubois: densification increases pressure");

    // Energy absorption
    CHECK(mat.energy_absorbed(state_c) >= 0.0, "FoamDubois: energy absorbed >= 0");

    // Tangent
    Real C[36];
    mat.tangent_stiffness(state_c, C);
    CHECK(C[0] > 0.0, "FoamDubois: tangent C11 > 0");

    // Pressure coupling
    MaterialState state_p1, state_p2;
    state_p1.strain[0] = -0.1;
    state_p1.strain[1] = -0.1;
    state_p1.strain[2] = -0.1;
    state_p1.effective_strain_rate = 0.0;
    mat.compute_stress(state_p1);
    Real plateau1 = mat.current_plateau(state_p1);
    CHECK(plateau1 > 2.0e6 * 0.9, "FoamDubois: pressure coupling factor");
}

// ==========================================================================
// 14. HenselSpittelMaterial
// ==========================================================================
void test_14_hensel_spittel() {
    std::cout << "\n=== Test 14: HenselSpittelMaterial ===\n";

    MaterialProperties props;
    props.E = 200.0e9;
    props.nu = 0.3;
    props.density = 7800.0;
    props.compute_derived();

    // Use m3 negative (standard for hot forming: exp(m3/eps) -> softening at low eps)
    // A must be large enough (~ 1e8) to produce physically meaningful flow stress
    HenselSpittelMaterial mat(props, 5.0e8, -0.001, 0.15, -0.01, -0.001, 0.05);

    // Flow stress function
    Real sy_300 = mat.flow_stress(0.01, 1.0, 300.0);
    CHECK(sy_300 > 0.0, "HenselSpittel: flow stress at 300K > 0");

    // Temperature softening
    Real sy_1000 = mat.flow_stress(0.01, 1.0, 1000.0);
    CHECK(sy_1000 < sy_300, "HenselSpittel: thermal softening at high T");

    // Strain hardening: with negative m3, exp(m3/eps) decreases at small eps
    // so eps^m2 dominates => higher eps gives higher stress
    Real sy_low = mat.flow_stress(0.01, 1.0, 600.0);
    Real sy_high = mat.flow_stress(0.5, 1.0, 600.0);
    CHECK(sy_high > sy_low, "HenselSpittel: strain hardening");

    // Elastic range
    MaterialState state_e;
    state_e.strain[0] = 0.0001;
    state_e.temperature = 800.0;
    mat.compute_stress(state_e);
    CHECK(state_e.stress[0] > 0.0, "HenselSpittel: stress positive in tension");

    // Plastic deformation at hot forming temperature
    MaterialState state_p;
    state_p.strain[0] = 0.01;
    state_p.temperature = 1000.0;  // High T => low yield => yields sooner
    mat.compute_stress(state_p);
    CHECK(state_p.history[32] > 0.0, "HenselSpittel: plastic strain at high strain/temp");

    // Current flow stored
    CHECK(mat.current_flow(state_p) > 0.0, "HenselSpittel: current flow stress stored");

    // Tangent stiffness
    Real C[36];
    mat.tangent_stiffness(state_e, C);
    CHECK(C[0] > 0.0, "HenselSpittel: tangent C11 > 0");

    // Rate sensitivity
    Real sy_low_rate = mat.flow_stress(0.01, 0.01, 600.0);
    Real sy_high_rate = mat.flow_stress(0.01, 100.0, 600.0);
    CHECK(sy_high_rate > sy_low_rate, "HenselSpittel: rate sensitivity");
}

// ==========================================================================
// 15. PaperLightMaterial
// ==========================================================================
void test_15_paper_light() {
    std::cout << "\n=== Test 15: PaperLightMaterial ===\n";

    MaterialProperties props;
    props.E = 5.0e9;
    props.nu = 0.3;
    props.density = 700.0;
    props.compute_derived();

    PaperLightMaterial mat(props, 5.0e9, 2.5e9, 0.5e9, 50.0e6, 25.0e6);

    // MD tension
    MaterialState state_md;
    state_md.strain[0] = 0.005;
    mat.compute_stress(state_md);
    CHECK(state_md.stress[0] > 0.0, "Paper: MD tension positive");

    // CD tension: weaker
    MaterialState state_cd;
    state_cd.strain[1] = 0.005;
    mat.compute_stress(state_cd);
    CHECK(state_cd.stress[1] > 0.0, "Paper: CD tension positive");
    CHECK(state_cd.stress[1] < state_md.stress[0], "Paper: CD weaker than MD");

    // Yield cap: stress limited
    MaterialState state_yc;
    state_yc.strain[0] = 0.1;
    mat.compute_stress(state_yc);
    CHECK(state_yc.stress[0] <= 50.0e6, "Paper: MD stress capped at yield");

    // Compression: softer
    MaterialState state_comp;
    state_comp.strain[0] = -0.005;
    mat.compute_stress(state_comp);
    CHECK(state_comp.stress[0] < 0.0, "Paper: compression stress negative");

    // Damage accumulation
    MaterialState state_big;
    state_big.strain[0] = 0.05;
    state_big.strain[1] = 0.05;
    mat.compute_stress(state_big);
    CHECK(mat.paper_damage(state_big) > 0.0, "Paper: damage at large strain");

    // ZD direction: very soft
    MaterialState state_z;
    state_z.strain[2] = 0.005;
    mat.compute_stress(state_z);
    CHECK(state_z.stress[2] > 0.0, "Paper: ZD stress positive");
    CHECK(state_z.stress[2] < state_cd.stress[1], "Paper: ZD softer than CD");

    // Tangent
    Real C[36];
    MaterialState state_0;
    mat.tangent_stiffness(state_0, C);
    CHECK(C[0] > C[7], "Paper: MD stiffer than CD");
    CHECK(C[7] > C[14], "Paper: CD stiffer than ZD");
}

// ==========================================================================
// 16. JWLBMaterial
// ==========================================================================
void test_16_jwlb() {
    std::cout << "\n=== Test 16: JWLBMaterial ===\n";

    MaterialProperties props;
    props.E = 1.0e10;
    props.nu = 0.0;
    props.density = 1630.0;
    props.compute_derived();

    JWLBMaterial mat(props, 3.712e11, 3.231e9, 4.15, 0.95, 0.30,
                      7.0e9, 2.0e9, 1.0e-4, 1.3);

    // Compression: JWL pressure
    MaterialState state_c;
    state_c.strain[0] = -0.01;
    state_c.strain[1] = -0.01;
    state_c.strain[2] = -0.01;
    state_c.dt = 1.0e-5;
    mat.compute_stress(state_c);
    CHECK(state_c.stress[0] < 0.0, "JWLB: compressive stress (negative = pressure)");

    // Hydrostatic: all normal stresses equal
    CHECK_NEAR(state_c.stress[0], state_c.stress[1], std::abs(state_c.stress[0]) * 0.01,
               "JWLB: hydrostatic sxx = syy");
    CHECK_NEAR(state_c.stress[3], 0.0, 1.0e-5, "JWLB: zero shear");

    // Afterburn energy increases with time
    Real Q1 = mat.afterburn_energy(state_c);
    CHECK(Q1 > 0.0, "JWLB: afterburn energy > 0 after dt");

    MaterialState state_c2 = state_c;
    state_c2.dt = 0.01;
    mat.compute_stress(state_c2);
    Real Q2 = mat.afterburn_energy(state_c2);
    CHECK(Q2 > Q1, "JWLB: afterburn increases with time");

    // Tangent
    Real C[36];
    mat.tangent_stiffness(state_c, C);
    CHECK(C[0] > 0.0, "JWLB: tangent bulk stiffness > 0");

    // Near zero volume: stable
    MaterialState state_0;
    state_0.strain[0] = 0.0;
    state_0.dt = 0.0;
    mat.compute_stress(state_0);
    CHECK(std::isfinite(state_0.stress[0]), "JWLB: stable at zero strain");
}

// ==========================================================================
// 17. PPPolymerMaterial
// ==========================================================================
void test_17_pp_polymer() {
    std::cout << "\n=== Test 17: PPPolymerMaterial ===\n";

    MaterialProperties props;
    props.E = 1.5e9;
    props.nu = 0.4;
    props.density = 900.0;
    props.hardening_modulus = 100.0e6;
    props.compute_derived();

    PPPolymerMaterial mat(props, 35.0e6, 20.0e6, 1.0e-8, 0.3);

    // Temperature-dependent yield
    Real sy_20 = mat.yield_at_temp(293.15);
    Real sy_80 = mat.yield_at_temp(353.15);
    CHECK(sy_20 > sy_80, "PPPoly: softer at higher temperature");
    CHECK_NEAR(sy_20, 35.0e6, 1.0e3, "PPPoly: yield at 20C");

    // Elastic range
    MaterialState state_e;
    state_e.strain[0] = 0.001;
    mat.compute_stress(state_e);
    CHECK(state_e.stress[0] > 0.0, "PPPoly: elastic stress positive");

    // Plastic at large strain
    MaterialState state_p;
    state_p.strain[0] = 0.1;
    mat.compute_stress(state_p);
    CHECK(state_p.history[32] > 0.0, "PPPoly: plastic strain at large deformation");

    // Creep strain evolves
    MaterialState state_cr;
    state_cr.strain[0] = 0.001;
    state_cr.dt = 0.01;
    state_cr.history[33] = 1.0e-6; // Small existing creep
    mat.compute_stress(state_cr);
    CHECK(mat.creep_strain(state_cr) > 1.0e-6, "PPPoly: creep strain evolves");

    // Tangent
    Real C[36];
    mat.tangent_stiffness(state_e, C);
    CHECK(C[0] > 0.0, "PPPoly: tangent C11 > 0");

    // High temperature: lower yield
    MaterialState state_ht;
    state_ht.strain[0] = 0.05;
    state_ht.temperature = 353.15; // 80C
    mat.compute_stress(state_ht);
    CHECK(state_ht.history[32] > 0.0, "PPPoly: yields at high temp");
}

// ==========================================================================
// 18. DruckerPrager3Material
// ==========================================================================
void test_18_drucker_prager3() {
    std::cout << "\n=== Test 18: DruckerPrager3Material ===\n";

    MaterialProperties props;
    props.E = 30.0e9;
    props.nu = 0.2;
    props.density = 2000.0;
    props.hardening_modulus = 1.0e6;
    props.compute_derived();

    DruckerPrager3Material mat(props, 1.0e6, 30.0, 0.5e6);

    CHECK_NEAR(mat.tension_cutoff(), 0.5e6, 1.0, "DP3: tension cutoff stored");

    // Elastic range
    MaterialState state_e;
    state_e.strain[0] = -0.00001;
    mat.compute_stress(state_e);
    CHECK_NEAR(state_e.history[32], 0.0, 1.0e-10, "DP3: elastic at small strain");

    // Confined shear yield
    MaterialState state_s;
    state_s.strain[0] = -0.001;
    state_s.strain[1] = -0.001;
    state_s.strain[2] = -0.001;
    state_s.strain[3] = 0.01;
    mat.compute_stress(state_s);
    CHECK(state_s.history[32] > 0.0, "DP3: plastic under confined shear");

    // Tension cutoff: p limited
    MaterialState state_t;
    state_t.strain[0] = 0.01;
    state_t.strain[1] = 0.01;
    state_t.strain[2] = 0.01;
    mat.compute_stress(state_t);
    Real p = (state_t.stress[0] + state_t.stress[1] + state_t.stress[2]) / 3.0;
    CHECK(p <= 0.5e6 + 1.0e3, "DP3: tension cutoff enforced");

    // Tangent
    Real C[36];
    mat.tangent_stiffness(state_e, C);
    CHECK(C[0] > 0.0, "DP3: tangent C11 > 0");
    CHECK(C[21] > 0.0, "DP3: tangent C44 > 0");

    // Pure compression: no cutoff
    MaterialState state_c;
    state_c.strain[0] = -0.005;
    state_c.strain[1] = -0.005;
    state_c.strain[2] = -0.005;
    mat.compute_stress(state_c);
    Real p_c = (state_c.stress[0] + state_c.stress[1] + state_c.stress[2]) / 3.0;
    CHECK(p_c < 0.0, "DP3: compression not cutoff");
}

// ==========================================================================
// 19. JCookAluminumMaterial
// ==========================================================================
void test_19_jcook_aluminum() {
    std::cout << "\n=== Test 19: JCookAluminumMaterial ===\n";

    auto props = make_aluminum_props();
    JCookAluminumMaterial mat(props, 324.0e6, 114.0e6, 0.42, 0.002, 1.34, 925.0, 293.15);

    // JC yield function
    Real sy_rt = mat.jc_yield(0.01, 1.0, 293.15);
    CHECK(sy_rt > 324.0e6, "JCAlum: yield > A at eps_p > 0");

    // Thermal softening
    Real sy_hot = mat.jc_yield(0.01, 1.0, 700.0);
    CHECK(sy_hot < sy_rt, "JCAlum: thermal softening at 700K");

    // Rate sensitivity
    Real sy_fast = mat.jc_yield(0.01, 1000.0, 293.15);
    CHECK(sy_fast >= sy_rt, "JCAlum: rate hardening");

    // Elastic
    MaterialState state_e;
    state_e.strain[0] = 0.001;
    mat.compute_stress(state_e);
    CHECK(state_e.stress[0] > 0.0, "JCAlum: elastic stress positive");

    // Plastic with adiabatic heating
    MaterialState state_p;
    state_p.strain[0] = 0.02;
    state_p.effective_strain_rate = 100.0;
    mat.compute_stress(state_p);
    CHECK(state_p.history[32] > 0.0, "JCAlum: plastic strain");
    Real dT = mat.temperature_rise(state_p);
    CHECK(dT > 0.0, "JCAlum: adiabatic temperature rise");

    // Tangent
    Real C[36];
    mat.tangent_stiffness(state_e, C);
    CHECK(C[0] > 0.0, "JCAlum: tangent C11 > 0");

    // Full melt: very low yield
    Real sy_melt = mat.jc_yield(0.01, 1.0, 920.0);
    CHECK(sy_melt < sy_rt * 0.1, "JCAlum: near-melt very soft");
}

// ==========================================================================
// 20. SpringGeneralizedMaterial
// ==========================================================================
void test_20_spring_generalized() {
    std::cout << "\n=== Test 20: SpringGeneralizedMaterial ===\n";

    MaterialProperties props;
    props.E = 1.0e9;
    props.nu = 0.3;
    props.density = 1000.0;
    props.compute_derived();

    Real K_trans[3] = {1.0e6, 2.0e6, 3.0e6};
    Real K_rot[3] = {1.0e3, 2.0e3, 3.0e3};

    SpringGeneralizedMaterial mat(props, K_trans, K_rot);

    CHECK_NEAR(mat.trans_stiffness(0), 1.0e6, 1.0, "Spring: K_trans[0] stored");
    CHECK_NEAR(mat.rot_stiffness(2), 3.0e3, 1.0, "Spring: K_rot[2] stored");

    // Linear spring: force = K * displacement
    MaterialState state_1;
    state_1.strain[0] = 0.5;
    mat.compute_stress(state_1);
    CHECK_NEAR(state_1.stress[0], 0.5e6, 0.5e6 * 0.01,
               "Spring: F = K * d for translational");

    // Rotational DOF
    MaterialState state_r;
    state_r.strain[3] = 0.1;
    mat.compute_stress(state_r);
    CHECK_NEAR(state_r.stress[3], 0.1 * 1.0e3, 0.1 * 1.0e3 * 0.01,
               "Spring: M = K_rot * theta");

    // Different stiffness per direction
    MaterialState state_multi;
    state_multi.strain[0] = 1.0;
    state_multi.strain[1] = 1.0;
    state_multi.strain[2] = 1.0;
    mat.compute_stress(state_multi);
    CHECK(state_multi.stress[1] > state_multi.stress[0], "Spring: K2 > K1");
    CHECK(state_multi.stress[2] > state_multi.stress[1], "Spring: K3 > K2");

    // Tangent stiffness: diagonal
    Real C[36];
    MaterialState state_0;
    mat.tangent_stiffness(state_0, C);
    CHECK(C[0] > 0.0, "Spring: tangent K1 > 0");
    CHECK_NEAR(C[1], 0.0, 1.0e-5, "Spring: off-diagonal zero");

    // Max displacement tracked
    CHECK(state_1.history[32] >= 0.5, "Spring: max displacement tracked");

    // Custom curve
    TabulatedCurve curve;
    curve.add_point(-1.0, -0.5);
    curve.add_point(0.0, 0.0);
    curve.add_point(1.0, 0.8);
    curve.add_point(2.0, 1.0);
    mat.set_curve(0, curve);

    MaterialState state_curve;
    state_curve.strain[0] = 1.5;
    mat.compute_stress(state_curve);
    CHECK(state_curve.stress[0] > 0.0, "Spring: custom curve produces positive force");
}

// ==========================================================================
// Main
// ==========================================================================
int main() {
    std::cout << "Wave 32 Material Models Test Suite\n";
    std::cout << "===================================\n";

    test_1_orthotropic_hill();
    test_2_vegter_yield();
    test_3_marlow_hyperelastic();
    test_4_deshpande_fleck();
    test_5_modified_ladeveze();
    test_6_cdpm2_concrete();
    test_7_jh_concrete();
    test_8_enhanced_composite();
    test_9_granular();
    test_10_viscous_foam();
    test_11_fabric_nl();
    test_12_arup_adhesive();
    test_13_foam_dubois();
    test_14_hensel_spittel();
    test_15_paper_light();
    test_16_jwlb();
    test_17_pp_polymer();
    test_18_drucker_prager3();
    test_19_jcook_aluminum();
    test_20_spring_generalized();

    std::cout << "\n===================================\n";
    std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed\n";

    if (tests_failed > 0) {
        std::cout << "SOME TESTS FAILED!\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED!\n";
    return 0;
}
