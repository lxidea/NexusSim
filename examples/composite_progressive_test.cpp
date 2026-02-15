/**
 * @file composite_progressive_test.cpp
 * @brief Tests for Wave 7 composite enhancements: thermal, interlaminar, progressive failure
 */

#include <nexussim/physics/composite_layup.hpp>
#include <nexussim/physics/composite_utils.hpp>
#include <nexussim/physics/composite_thermal.hpp>
#include <nexussim/physics/composite_interlaminar.hpp>
#include <nexussim/physics/composite_progressive_failure.hpp>
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

static bool near(Real a, Real b, Real tol = 1.0e-6) {
    return std::fabs(a - b) < tol * (1.0 + std::fabs(b));
}

static bool near_rel(Real a, Real b, Real rel_tol = 0.01) {
    if (std::fabs(b) < 1.0e-30) return std::fabs(a) < 1.0e-20;
    return std::fabs(a - b) / std::fabs(b) < rel_tol;
}

// Material constants
static constexpr Real E1 = 138.0e9;
static constexpr Real E2 = 8.96e9;
static constexpr Real G12 = 7.1e9;
static constexpr Real nu12 = 0.30;
static constexpr Real ply_t = 0.000125;

// Thermal
static constexpr Real alpha1 = -0.3e-6;
static constexpr Real alpha2 = 28.1e-6;
static constexpr Real T_cure = 450.0;
static constexpr Real T_service = 295.0;

// Strengths
static constexpr Real Xt = 1500.0e6;
static constexpr Real Xc = 1200.0e6;
static constexpr Real Yt = 50.0e6;
static constexpr Real Yc = 200.0e6;
static constexpr Real S12s = 70.0e6;
static constexpr Real S23s = 40.0e6;

// ==========================================================================
// Test 1: CTE transform
// ==========================================================================
void test_cte_transform() {
    std::cout << "\n=== Test 1: CTE Transform ===\n";

    Real ab[3];

    // 0 degree: alpha_xx = alpha1, alpha_yy = alpha2
    composite_detail::transform_cte(0.0, alpha1, alpha2, ab);
    CHECK(near(ab[0], alpha1, 1e-10), "0 deg: alpha_xx = alpha1");
    CHECK(near(ab[1], alpha2, 1e-10), "0 deg: alpha_yy = alpha2");

    // 90 degree: alpha_xx = alpha2, alpha_yy = alpha1
    composite_detail::transform_cte(90.0, alpha1, alpha2, ab);
    CHECK(near(ab[0], alpha2, 1e-10), "90 deg: alpha_xx = alpha2");
    CHECK(near(ab[1], alpha1, 1e-10), "90 deg: alpha_yy = alpha1");

    // 45 degree: alpha_xx ~ alpha_yy (average)
    composite_detail::transform_cte(45.0, alpha1, alpha2, ab);
    CHECK(near_rel(ab[0], ab[1], 0.01), "45 deg: alpha_xx ~ alpha_yy");
    Real avg = (alpha1 + alpha2) / 2.0;
    CHECK(near_rel(ab[0], avg, 0.01), "45 deg: alpha_xx ~ average(alpha1, alpha2)");
}

// ==========================================================================
// Test 2: Thermal resultants symmetric laminate
// ==========================================================================
void test_thermal_resultants_symmetric() {
    std::cout << "\n=== Test 2: Thermal Resultants Symmetric ===\n";

    auto lam = layup_presets::cross_ply(E1, E2, G12, nu12, ply_t, 1);
    lam.compute_abd();

    CompositeThermalAnalysis therm;
    therm.set_all_ply_cte(lam.num_plies(), alpha1, alpha2);
    therm.set_temperatures(T_cure, T_service);

    Real N_T[3], M_T[3];
    therm.compute_thermal_resultants(lam, N_T, M_T);

    // Symmetric laminate: M_T should be ~0
    CHECK(std::fabs(M_T[0]) < std::fabs(N_T[0]) * 1e-8, "Symmetric: M_T_xx ~ 0");
    CHECK(std::fabs(M_T[1]) < std::fabs(N_T[0]) * 1e-8, "Symmetric: M_T_yy ~ 0");

    // N_T should be nonzero (CTE mismatch generates thermal forces)
    CHECK(std::fabs(N_T[0]) > 0.0, "Symmetric: N_T_xx != 0");
    CHECK(std::fabs(N_T[1]) > 0.0, "Symmetric: N_T_yy != 0");
}

// ==========================================================================
// Test 3: Thermal resultants unsymmetric laminate
// ==========================================================================
void test_thermal_resultants_unsymmetric() {
    std::cout << "\n=== Test 3: Thermal Resultants Unsymmetric ===\n";

    // [0/90] unsymmetric
    CompositeLaminate lam;
    PlyDefinition p0(E1, E2, G12, nu12, ply_t, 0.0);
    PlyDefinition p90(E1, E2, G12, nu12, ply_t, 90.0);
    lam.add_ply(p0);
    lam.add_ply(p90);
    lam.compute_abd();

    CompositeThermalAnalysis therm;
    therm.set_all_ply_cte(lam.num_plies(), alpha1, alpha2);
    therm.set_temperatures(T_cure, T_service);

    Real N_T[3], M_T[3];
    therm.compute_thermal_resultants(lam, N_T, M_T);

    // Unsymmetric: M_T should be nonzero (CTE bending coupling)
    bool has_moment = std::fabs(M_T[0]) > 1e-6 || std::fabs(M_T[1]) > 1e-6;
    CHECK(has_moment, "Unsymmetric [0/90]: M_T != 0 (bending coupling)");
    CHECK(std::fabs(N_T[0]) > 0.0, "Unsymmetric: N_T_xx != 0");
    CHECK(std::fabs(N_T[1]) > 0.0, "Unsymmetric: N_T_yy != 0");
}

// ==========================================================================
// Test 4: Thermal deformation symmetric (QI)
// ==========================================================================
void test_thermal_deformation_symmetric() {
    std::cout << "\n=== Test 4: Thermal Deformation Symmetric ===\n";

    auto lam = layup_presets::quasi_isotropic(E1, E2, G12, nu12, ply_t);
    lam.compute_abd();

    CompositeThermalAnalysis therm;
    therm.set_all_ply_cte(lam.num_plies(), alpha1, alpha2);
    therm.set_temperatures(T_cure, T_service);

    Real eps0_T[3], kappa_T[3];
    therm.compute_thermal_deformation(lam, eps0_T, kappa_T);

    // QI symmetric: kappa_T ~ 0
    CHECK(std::fabs(kappa_T[0]) < 1e-3, "QI: kappa_xx_T ~ 0");
    CHECK(std::fabs(kappa_T[1]) < 1e-3, "QI: kappa_yy_T ~ 0");

    // QI: eps_xx ~ eps_yy (quasi-isotropic in-plane)
    CHECK(near_rel(eps0_T[0], eps0_T[1], 0.05), "QI: eps_xx_T ~ eps_yy_T");

    // Thermal strain should be nonzero
    CHECK(std::fabs(eps0_T[0]) > 1e-10, "QI: eps_T nonzero");
}

// ==========================================================================
// Test 5: Thermal ply stresses [0/90]s cool-down
// ==========================================================================
void test_thermal_ply_stresses() {
    std::cout << "\n=== Test 5: Thermal Ply Stresses ===\n";

    auto lam = layup_presets::cross_ply(E1, E2, G12, nu12, ply_t, 1);
    lam.compute_abd();

    CompositeThermalAnalysis therm;
    therm.set_all_ply_cte(lam.num_plies(), alpha1, alpha2);
    therm.set_temperatures(T_cure, T_service);

    PlyState states[8];
    therm.compute_thermal_ply_stresses(lam, states);

    // [0/90]s cool-down (dT < 0): 0-deg plies have low CTE in fiber dir,
    // laminate contracts less than 0-deg wants -> 0-deg ply gets compressive sigma11
    // and 90-deg plies get tensile sigma11 (they want to shrink more transversely)
    // Actually: dT = 295-450 = -155K (cool-down)
    // 0-deg: alpha1 is negative -> wants to expand on cool-down, constrained -> compression
    // 90-deg: alpha2 is large positive -> wants to shrink on cool-down, constrained -> tension in 22
    // Let's just check the sign pattern is consistent and stresses are nonzero

    // Ply 0 is 0-deg, ply 1 is 90-deg
    CHECK(std::fabs(states[0].stress_local[0]) > 1e3, "0-deg ply: sigma_11 nonzero");
    CHECK(std::fabs(states[1].stress_local[0]) > 1e3, "90-deg ply: sigma_11 nonzero");

    // 0-deg and 90-deg plies should have opposite sign thermal stresses
    // (self-equilibrating within symmetric laminate)
    CHECK(states[0].stress_global[0] * states[1].stress_global[0] < 0.0,
          "0-deg and 90-deg opposite sign sigma_xx");
    CHECK(states[0].stress_global[1] * states[1].stress_global[1] < 0.0,
          "0-deg and 90-deg opposite sign sigma_yy");
}

// ==========================================================================
// Test 6: Thermal zero delta-T
// ==========================================================================
void test_thermal_zero_dt() {
    std::cout << "\n=== Test 6: Thermal Zero dT ===\n";

    auto lam = layup_presets::cross_ply(E1, E2, G12, nu12, ply_t, 1);
    lam.compute_abd();

    CompositeThermalAnalysis therm;
    therm.set_all_ply_cte(lam.num_plies(), alpha1, alpha2);
    therm.set_temperatures(300.0, 300.0); // dT = 0

    Real N_T[3], M_T[3];
    therm.compute_thermal_resultants(lam, N_T, M_T);

    CHECK(std::fabs(N_T[0]) < 1e-10, "dT=0: N_T_xx = 0");
    CHECK(std::fabs(N_T[1]) < 1e-10, "dT=0: N_T_yy = 0");
    CHECK(std::fabs(M_T[0]) < 1e-10, "dT=0: M_T_xx = 0");
}

// ==========================================================================
// Test 7: Interlaminar shear UD [0]4
// ==========================================================================
void test_interlaminar_shear_ud() {
    std::cout << "\n=== Test 7: Interlaminar Shear UD ===\n";

    auto lam = layup_presets::unidirectional(E1, E2, G12, nu12, ply_t, 4);
    lam.compute_abd();

    CompositeInterlaminarAnalysis ila;
    Real V[2] = {1000.0, 0.0}; // 1 kN shear force

    InterlaminarStress stresses[8];
    int n = ila.compute_interlaminar_shear(lam, V, stresses);

    CHECK(n == 3, "[0]4: 3 interfaces");

    // For UD [0]4 under Vx: parabolic-like profile, max at midplane
    // Interface at midplane (index 1 for 4 plies: between ply 1 and 2)
    CHECK(std::fabs(stresses[1].tau_xz) >= std::fabs(stresses[0].tau_xz),
          "UD: midplane tau_xz >= outer interface");
    CHECK(std::fabs(stresses[1].tau_xz) >= std::fabs(stresses[2].tau_xz),
          "UD: midplane tau_xz >= outer interface (other side)");
    CHECK(std::fabs(stresses[1].tau_xz) > 0.0, "UD: midplane tau_xz > 0");
}

// ==========================================================================
// Test 8: Interlaminar shear cross-ply [0/90]s
// ==========================================================================
void test_interlaminar_shear_cross_ply() {
    std::cout << "\n=== Test 8: Interlaminar Shear Cross-Ply ===\n";

    auto lam = layup_presets::cross_ply(E1, E2, G12, nu12, ply_t, 1);
    lam.compute_abd();

    CompositeInterlaminarAnalysis ila;
    Real V[2] = {1000.0, 0.0};

    InterlaminarStress stresses[8];
    int n = ila.compute_interlaminar_shear(lam, V, stresses);

    CHECK(n == 3, "[0/90]s: 3 interfaces");

    // All interfaces should have nonzero tau_xz
    CHECK(std::fabs(stresses[0].tau_xz) > 0.0, "Interface 0: tau_xz > 0");
    CHECK(std::fabs(stresses[1].tau_xz) > 0.0, "Interface 1 (midplane): tau_xz > 0");

    // Peak should be near midplane
    Real max_tau = 0.0;
    int max_idx = -1;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(stresses[i].tau_xz) > max_tau) {
            max_tau = std::fabs(stresses[i].tau_xz);
            max_idx = i;
        }
    }
    CHECK(max_idx == 1, "Cross-ply: peak near midplane");
    CHECK(max_tau > 0.0, "Cross-ply: max tau > 0");
}

// ==========================================================================
// Test 9: Interlaminar shear symmetric property
// ==========================================================================
void test_interlaminar_shear_symmetric() {
    std::cout << "\n=== Test 9: Interlaminar Shear Symmetric ===\n";

    auto lam = layup_presets::quasi_isotropic(E1, E2, G12, nu12, ply_t);
    lam.compute_abd();

    CompositeInterlaminarAnalysis ila;
    Real V[2] = {1000.0, 0.0};

    InterlaminarStress stresses[32];
    int n = ila.compute_interlaminar_shear(lam, V, stresses);

    CHECK(n == 7, "QI [0/+45/-45/90]s: 7 interfaces");

    // Max should be near the midplane
    Real max_tau = ila.max_interlaminar_shear(lam, V);
    CHECK(max_tau > 0.0, "QI: max interlaminar shear > 0");

    // Midplane interface should have high tau
    int mid_idx = n / 2;
    CHECK(std::fabs(stresses[mid_idx].tau_xz) > 0.5 * max_tau,
          "QI: midplane region has significant tau");
}

// ==========================================================================
// Test 10: Interlaminar shear zero V
// ==========================================================================
void test_interlaminar_shear_zero() {
    std::cout << "\n=== Test 10: Interlaminar Shear Zero V ===\n";

    auto lam = layup_presets::cross_ply(E1, E2, G12, nu12, ply_t, 1);
    lam.compute_abd();

    CompositeInterlaminarAnalysis ila;
    Real V[2] = {0.0, 0.0};

    InterlaminarStress stresses[8];
    int n = ila.compute_interlaminar_shear(lam, V, stresses);

    for (int i = 0; i < n; ++i) {
        CHECK(std::fabs(stresses[i].tau_xz) < 1e-10, "V=0: tau_xz = 0");
    }
}

// ==========================================================================
// Test 11: FPF uniaxial tension [0/90]s
// ==========================================================================
void test_fpf_uniaxial_tension() {
    std::cout << "\n=== Test 11: FPF Uniaxial Tension ===\n";

    auto lam = layup_presets::cross_ply(E1, E2, G12, nu12, ply_t, 1);
    lam.compute_abd();

    CompositeProgressiveFailure cpf;
    cpf.set_failure_params(Xt, Xc, Yt, Yc, S12s, S23s);

    // Unit load in Nxx direction
    Real N_app[3] = {1.0, 0.0, 0.0};
    Real M_app[3] = {0.0, 0.0, 0.0};

    auto fpf = cpf.first_ply_failure(lam, N_app, M_app);

    CHECK(fpf.load_multiplier > 0.0, "FPF tension: lambda > 0");
    CHECK(fpf.critical_ply >= 0, "FPF tension: critical ply identified");

    // 90-deg plies should fail first (matrix tension in transverse direction)
    // plies 1,2 are 90-deg in [0/90]s
    CHECK(fpf.failure_mode == 3, "FPF tension: matrix tension mode (mode 3)");
    CHECK(fpf.critical_ply == 1 || fpf.critical_ply == 2,
          "FPF tension: 90-deg ply fails first");

    // Load multiplier should be finite and reasonable
    CHECK(fpf.load_multiplier < 1e8, "FPF tension: lambda is finite");
}

// ==========================================================================
// Test 12: FPF uniaxial compression [0/90]s
// ==========================================================================
void test_fpf_uniaxial_compression() {
    std::cout << "\n=== Test 12: FPF Uniaxial Compression ===\n";

    auto lam = layup_presets::cross_ply(E1, E2, G12, nu12, ply_t, 1);
    lam.compute_abd();

    CompositeProgressiveFailure cpf;
    cpf.set_failure_params(Xt, Xc, Yt, Yc, S12s, S23s);

    Real N_tens[3] = {1.0, 0.0, 0.0};
    Real N_comp[3] = {-1.0, 0.0, 0.0};
    Real M_app[3] = {0.0, 0.0, 0.0};

    auto fpf_t = cpf.first_ply_failure(lam, N_tens, M_app);
    auto fpf_c = cpf.first_ply_failure(lam, N_comp, M_app);
    CHECK(fpf_c.load_multiplier > 0.0, "FPF compression: lambda > 0");
    // Under uniaxial compression, 0-deg plies carry fiber-direction compression
    // and hit fiber compression (mode 2) before 90-deg plies hit matrix compression
    CHECK(fpf_c.failure_mode == 2, "FPF compression: fiber compression mode");

    // Compression FPF should be higher than tension FPF (fiber vs matrix)
    CHECK(fpf_c.load_multiplier > fpf_t.load_multiplier,
          "FPF: lambda_comp > lambda_tens (fiber comp stronger than matrix tens)");
    CHECK(fpf_c.load_multiplier < 1e8, "FPF compression: lambda finite");
}

// ==========================================================================
// Test 13: FPF biaxial
// ==========================================================================
void test_fpf_biaxial() {
    std::cout << "\n=== Test 13: FPF Biaxial ===\n";

    auto lam = layup_presets::quasi_isotropic(E1, E2, G12, nu12, ply_t);
    lam.compute_abd();

    CompositeProgressiveFailure cpf;
    cpf.set_failure_params(Xt, Xc, Yt, Yc, S12s, S23s);

    Real N_uni[3] = {1.0, 0.0, 0.0};
    Real N_bi[3] = {1.0, 1.0, 0.0};
    Real M_app[3] = {0.0, 0.0, 0.0};

    auto fpf_uni = cpf.first_ply_failure(lam, N_uni, M_app);
    auto fpf_bi = cpf.first_ply_failure(lam, N_bi, M_app);

    CHECK(fpf_bi.load_multiplier > 0.0, "FPF biaxial: lambda > 0");
    CHECK(fpf_bi.critical_ply >= 0, "FPF biaxial: critical ply found");
    // Biaxial should differ from uniaxial
    CHECK(std::fabs(fpf_bi.load_multiplier - fpf_uni.load_multiplier) > 1.0,
          "FPF biaxial: different from uniaxial");
}

// ==========================================================================
// Test 14: FPF pure shear
// ==========================================================================
void test_fpf_pure_shear() {
    std::cout << "\n=== Test 14: FPF Pure Shear ===\n";

    auto lam = layup_presets::quasi_isotropic(E1, E2, G12, nu12, ply_t);
    lam.compute_abd();

    CompositeProgressiveFailure cpf;
    cpf.set_failure_params(Xt, Xc, Yt, Yc, S12s, S23s);

    Real N_app[3] = {0.0, 0.0, 1.0};
    Real M_app[3] = {0.0, 0.0, 0.0};

    auto fpf = cpf.first_ply_failure(lam, N_app, M_app);

    CHECK(fpf.load_multiplier > 0.0, "FPF shear: lambda > 0");
    CHECK(fpf.critical_ply >= 0, "FPF shear: critical ply found");
    CHECK(fpf.load_multiplier < 1e8, "FPF shear: lambda finite");
}

// ==========================================================================
// Test 15: FPF Tsai-Wu vs Hashin
// ==========================================================================
void test_fpf_tsai_wu_vs_hashin() {
    std::cout << "\n=== Test 15: FPF Tsai-Wu vs Hashin ===\n";

    auto lam = layup_presets::cross_ply(E1, E2, G12, nu12, ply_t, 1);
    lam.compute_abd();

    CompositeProgressiveFailure cpf;
    cpf.set_failure_params(Xt, Xc, Yt, Yc, S12s, S23s);
    cpf.set_F12_star(-0.5);

    Real N_app[3] = {1.0, 0.0, 0.0};
    Real M_app[3] = {0.0, 0.0, 0.0};

    auto fpf_h = cpf.first_ply_failure_criterion(lam, N_app, M_app, FailureCriterion::Hashin);
    auto fpf_tw = cpf.first_ply_failure_criterion(lam, N_app, M_app, FailureCriterion::TsaiWu);

    CHECK(fpf_h.load_multiplier > 0.0, "Hashin FPF: lambda > 0");
    CHECK(fpf_tw.load_multiplier > 0.0, "Tsai-Wu FPF: lambda > 0");

    // Tsai-Wu is generally more conservative due to interaction terms
    CHECK(fpf_tw.load_multiplier <= fpf_h.load_multiplier * 1.1,
          "Tsai-Wu ~ Hashin or more conservative");
    CHECK(fpf_tw.load_multiplier > fpf_h.load_multiplier * 0.3,
          "Tsai-Wu and Hashin within reasonable range");
}

// ==========================================================================
// Test 16: Progressive cross-ply tension
// ==========================================================================
void test_progressive_cross_ply_tension() {
    std::cout << "\n=== Test 16: Progressive Cross-Ply Tension ===\n";

    auto lam = layup_presets::cross_ply(E1, E2, G12, nu12, ply_t, 1);
    lam.compute_abd();

    CompositeProgressiveFailure cpf;
    cpf.set_failure_params(Xt, Xc, Yt, Yc, S12s, S23s);
    cpf.set_degradation_model(DegradationModel::Ply_Discount);
    cpf.set_residual_stiffness(0.0);

    Real N_app[3] = {1.0, 0.0, 0.0};
    Real M_app[3] = {0.0, 0.0, 0.0};

    ProgressiveFailureResult results[10];
    int events = cpf.progressive_analysis(lam, N_app, M_app, results, 10);

    CHECK(events >= 2, "Progressive: at least 2 failure events");

    // First event: 90-deg plies fail (matrix tension)
    CHECK(results[0].ply_status[1].failed || results[0].ply_status[2].failed,
          "Progressive: 90-deg plies fail first");

    // Second/later event: 0-deg plies fail (fiber)
    bool zero_deg_failed = false;
    for (int e = 1; e < events; ++e) {
        if (results[e].ply_status[0].failed || results[e].ply_status[3].failed)
            zero_deg_failed = true;
    }
    CHECK(zero_deg_failed, "Progressive: 0-deg plies eventually fail");

    // Fiber failure at much higher load than matrix failure
    if (events >= 2) {
        CHECK(results[events - 1].load_multiplier > results[0].load_multiplier * 2.0,
              "Progressive: fiber failure >> matrix failure load");
    }

    // Check failure modes
    CHECK(results[0].ply_status[1].failure_mode == 3 || results[0].ply_status[2].failure_mode == 3,
          "Progressive: first failure is matrix tension (mode 3)");
}

// ==========================================================================
// Test 17: Progressive degraded ABD
// ==========================================================================
void test_progressive_degraded_abd() {
    std::cout << "\n=== Test 17: Progressive Degraded ABD ===\n";

    auto lam = layup_presets::cross_ply(E1, E2, G12, nu12, ply_t, 1);
    lam.compute_abd();

    Real A11_orig = lam.A()[0];
    Real A22_orig = lam.A()[4];

    CompositeProgressiveFailure cpf;
    cpf.set_failure_params(Xt, Xc, Yt, Yc, S12s, S23s);
    cpf.set_degradation_model(DegradationModel::Ply_Discount);
    cpf.set_residual_stiffness(0.0);

    Real N_app[3] = {1.0, 0.0, 0.0};
    Real M_app[3] = {0.0, 0.0, 0.0};

    ProgressiveFailureResult results[10];
    int events = cpf.progressive_analysis(lam, N_app, M_app, results, 10);

    if (events >= 1) {
        // After 90-deg failure: A22 should drop significantly (90-deg plies contribute most to A22)
        Real A22_after = results[0].degraded_A[4];
        CHECK(A22_after < A22_orig, "After 90-deg failure: A22 drops");
        CHECK(A22_after < 0.7 * A22_orig, "After 90-deg failure: A22 drops significantly");

        // A11 should also drop but less (90-deg contributes little to A11)
        Real A11_after = results[0].degraded_A[0];
        CHECK(A11_after < A11_orig, "After 90-deg failure: A11 also drops");
        CHECK(A11_after > 0.3 * A11_orig, "After 90-deg failure: A11 doesn't drop as much");
    }
}

// ==========================================================================
// Test 18: Progressive selective vs discount
// ==========================================================================
void test_progressive_selective_vs_discount() {
    std::cout << "\n=== Test 18: Selective vs Discount ===\n";

    auto lam = layup_presets::cross_ply(E1, E2, G12, nu12, ply_t, 1);
    lam.compute_abd();

    // Apply moderate strain causing matrix failure but not fiber failure
    // Use evaluate() for single-step check
    Real eps0[3] = {0.01, 0.0, 0.0}; // Large enough to fail matrix in 90-deg plies
    Real kappa[3] = {0.0, 0.0, 0.0};

    CompositeProgressiveFailure cpf_disc;
    cpf_disc.set_failure_params(Xt, Xc, Yt, Yc, S12s, S23s);
    cpf_disc.set_degradation_model(DegradationModel::Ply_Discount);
    cpf_disc.set_residual_stiffness(0.0);

    CompositeProgressiveFailure cpf_sel;
    cpf_sel.set_failure_params(Xt, Xc, Yt, Yc, S12s, S23s);
    cpf_sel.set_degradation_model(DegradationModel::Selective_Discount);
    cpf_sel.set_residual_stiffness(0.0);

    auto res_disc = cpf_disc.evaluate(lam, eps0, kappa);
    auto res_sel = cpf_sel.evaluate(lam, eps0, kappa);

    // Both should detect failures
    CHECK(res_disc.num_failed_plies > 0, "Discount: plies failed");
    CHECK(res_sel.num_failed_plies > 0, "Selective: plies failed");

    // Selective should retain more stiffness than full discount
    CHECK(res_sel.degraded_A[0] >= res_disc.degraded_A[0],
          "Selective retains >= discount A11");
    CHECK(res_sel.degraded_A[4] >= res_disc.degraded_A[4],
          "Selective retains >= discount A22");
}

// ==========================================================================
// Test 19: Strength envelope QI
// ==========================================================================
void test_strength_envelope_qi() {
    std::cout << "\n=== Test 19: Strength Envelope QI ===\n";

    auto lam = layup_presets::quasi_isotropic(E1, E2, G12, nu12, ply_t);
    lam.compute_abd();

    CompositeProgressiveFailure cpf;
    cpf.set_failure_params(Xt, Xc, Yt, Yc, S12s, S23s);

    StrengthPoint points[36];
    int n = cpf.strength_envelope(lam, 1.0, points, 36);

    CHECK(n == 36, "36 envelope points");

    // QI: envelope should be roughly circular (Nxx_max ~ Nyy_max)
    Real max_Nxx = 0.0, max_Nyy = 0.0;
    Real min_Nxx = 0.0, min_Nyy = 0.0;
    for (int i = 0; i < n; ++i) {
        if (points[i].Nxx > max_Nxx) max_Nxx = points[i].Nxx;
        if (points[i].Nxx < min_Nxx) min_Nxx = points[i].Nxx;
        if (points[i].Nyy > max_Nyy) max_Nyy = points[i].Nyy;
        if (points[i].Nyy < min_Nyy) min_Nyy = points[i].Nyy;
    }

    CHECK(max_Nxx > 0.0, "Envelope: positive Nxx_max");
    CHECK(min_Nxx < 0.0, "Envelope: negative Nxx_min (compression)");

    // QI: tension and compression strengths should differ
    CHECK(max_Nxx > -min_Nxx * 0.3, "Envelope: tension/compression asymmetry");

    // QI: roughly symmetric in x and y
    CHECK(near_rel(max_Nxx, max_Nyy, 0.15), "QI envelope: Nxx_max ~ Nyy_max");
}

// ==========================================================================
// Test 20: Strength envelope UD
// ==========================================================================
void test_strength_envelope_ud() {
    std::cout << "\n=== Test 20: Strength Envelope UD ===\n";

    auto lam = layup_presets::unidirectional(E1, E2, G12, nu12, ply_t, 4);
    lam.compute_abd();

    CompositeProgressiveFailure cpf;
    cpf.set_failure_params(Xt, Xc, Yt, Yc, S12s, S23s);

    StrengthPoint points[36];
    int n = cpf.strength_envelope(lam, 1.0, points, 36);

    Real max_Nxx = 0.0, max_Nyy = 0.0;
    for (int i = 0; i < n; ++i) {
        if (points[i].Nxx > max_Nxx) max_Nxx = points[i].Nxx;
        if (points[i].Nyy > max_Nyy) max_Nyy = points[i].Nyy;
    }

    // UD [0]: much stronger in fiber direction than transverse
    CHECK(max_Nxx > 5.0 * max_Nyy, "UD: Nxx_max >> Nyy_max (fiber vs matrix)");
    CHECK(max_Nxx > 0.0, "UD: Nxx_max > 0");
    CHECK(max_Nyy > 0.0, "UD: Nyy_max > 0");
}

// ==========================================================================
// Main
// ==========================================================================
int main() {
    std::cout << "================================================\n";
    std::cout << "NexusSim Wave 7: Composite Progressive Test\n";
    std::cout << "================================================\n";

    test_cte_transform();
    test_thermal_resultants_symmetric();
    test_thermal_resultants_unsymmetric();
    test_thermal_deformation_symmetric();
    test_thermal_ply_stresses();
    test_thermal_zero_dt();
    test_interlaminar_shear_ud();
    test_interlaminar_shear_cross_ply();
    test_interlaminar_shear_symmetric();
    test_interlaminar_shear_zero();
    test_fpf_uniaxial_tension();
    test_fpf_uniaxial_compression();
    test_fpf_biaxial();
    test_fpf_pure_shear();
    test_fpf_tsai_wu_vs_hashin();
    test_progressive_cross_ply_tension();
    test_progressive_degraded_abd();
    test_progressive_selective_vs_discount();
    test_strength_envelope_qi();
    test_strength_envelope_ud();

    std::cout << "\n================================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    std::cout << "================================================\n";

    return (tests_failed > 0) ? 1 : 0;
}
