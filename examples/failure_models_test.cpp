/**
 * @file failure_models_test.cpp
 * @brief Comprehensive test for Wave 2 failure/damage models
 */

#include <nexussim/physics/failure/failure_models.hpp>
#include <nexussim/physics/element_erosion.hpp>
#include <iostream>
#include <cmath>

using namespace nxs;
using namespace nxs::physics;
using namespace nxs::physics::failure;

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

// ==========================================================================
// Test 1: Hashin - Fiber Tension
// ==========================================================================
void test_hashin_fiber_tension() {
    std::cout << "\n=== Test 1: Hashin - Fiber Tension ===\n";

    FailureModelParameters params;
    params.type = FailureModelType::Hashin;
    params.Xt = 1500.0e6;  // T300/epoxy
    params.Xc = 1200.0e6;
    params.Yt = 50.0e6;
    params.Yc = 200.0e6;
    params.S12 = 70.0e6;
    params.S23 = 40.0e6;

    HashinFailure model(params);

    // Below fiber tension strength
    MaterialState mstate;
    mstate.stress[0] = 1000.0e6;  // σ11 = 1000 MPa (below Xt=1500)
    mstate.stress[3] = 0.0;

    FailureState fstate;
    model.compute_damage(mstate, fstate, 0.0, 0.0);
    CHECK(!fstate.failed, "Below Xt: no failure");
    CHECK(fstate.history[0] < 1.0, "Fiber tension index < 1");

    // Above fiber tension strength
    mstate.stress[0] = 1600.0e6;  // σ11 > Xt
    FailureState fstate2;
    model.compute_damage(mstate, fstate2, 0.0, 0.0);
    CHECK(fstate2.failed, "Above Xt: fiber tension failure");
    CHECK(fstate2.failure_mode == 1, "Failure mode = 1 (fiber tension)");
    CHECK(fstate2.history[0] >= 1.0, "Fiber tension index >= 1");
}

// ==========================================================================
// Test 2: Hashin - Matrix Failure Modes
// ==========================================================================
void test_hashin_matrix() {
    std::cout << "\n=== Test 2: Hashin - Matrix Failure ===\n";

    FailureModelParameters params;
    params.Xt = 1500.0e6;
    params.Xc = 1200.0e6;
    params.Yt = 50.0e6;
    params.Yc = 200.0e6;
    params.S12 = 70.0e6;
    params.S23 = 40.0e6;

    HashinFailure model(params);

    // Matrix tension failure (σ22 > 0)
    MaterialState mstate;
    mstate.stress[1] = 60.0e6;   // σ22 > Yt=50 MPa
    mstate.stress[3] = 10.0e6;   // Some shear

    FailureState fstate;
    model.compute_damage(mstate, fstate, 0.0, 0.0);
    CHECK(fstate.failed, "Matrix tension failure");
    CHECK(fstate.failure_mode == 3, "Failure mode = 3 (matrix tension)");

    // Matrix compression failure (σ22 < 0)
    MaterialState mstate_c;
    mstate_c.stress[1] = -150.0e6;
    mstate_c.stress[3] = 50.0e6;

    FailureState fstate_c;
    model.compute_damage(mstate_c, fstate_c, 0.0, 0.0);
    CHECK(fstate_c.history[3] > 0.0, "Matrix compression index computed");

    // Stress degradation after matrix failure
    Real sigma[6] = {100.0e6, 50.0e6, 0.0, 30.0e6, 10.0e6, 0.0};
    FailureState fstate_deg;
    fstate_deg.history[6] = 1.0;  // Matrix tension mode damaged
    model.degrade_stress(sigma, fstate_deg);
    CHECK(sigma[1] == 0.0, "σ22 zeroed after matrix failure");
    CHECK(sigma[0] == 100.0e6, "σ11 preserved (fiber intact)");
}

// ==========================================================================
// Test 3: Tsai-Wu Polynomial Criterion
// ==========================================================================
void test_tsai_wu() {
    std::cout << "\n=== Test 3: Tsai-Wu Polynomial ===\n";

    FailureModelParameters params;
    params.Xt = 1500.0e6;
    params.Xc = 1200.0e6;
    params.Yt = 50.0e6;
    params.Yc = 200.0e6;
    params.S12 = 70.0e6;
    params.F12_star = -0.5;

    TsaiWuFailure model(params);

    // Pure fiber tension below limit
    MaterialState mstate;
    mstate.stress[0] = 500.0e6;  // Well below Xt

    FailureState fstate;
    model.compute_damage(mstate, fstate, 0.0, 0.0);
    CHECK(!fstate.failed, "Below failure surface: safe");
    CHECK(fstate.history[0] < 1.0, "Failure index < 1");
    CHECK(fstate.damage < 1.0, "Damage < 1");

    // Combined loading causing failure
    MaterialState mstate_fail;
    mstate_fail.stress[0] = 200.0e6;   // Moderate fiber stress
    mstate_fail.stress[1] = 45.0e6;    // Near matrix tension limit
    mstate_fail.stress[3] = 40.0e6;    // Significant shear

    FailureState fstate_fail;
    model.compute_damage(mstate_fail, fstate_fail, 0.0, 0.0);
    CHECK(fstate_fail.history[0] > 0.0, "Failure index computed for combined loading");

    // Pure transverse tension failure
    MaterialState mstate_yt;
    mstate_yt.stress[1] = 55.0e6;  // > Yt = 50 MPa

    FailureState fstate_yt;
    model.compute_damage(mstate_yt, fstate_yt, 0.0, 0.0);
    CHECK(fstate_yt.failed, "Transverse tension exceeding Yt causes failure");
}

// ==========================================================================
// Test 4: Chang-Chang Laminate Failure
// ==========================================================================
void test_chang_chang() {
    std::cout << "\n=== Test 4: Chang-Chang ===\n";

    FailureModelParameters params;
    params.Xt = 1500.0e6;
    params.Xc = 1200.0e6;
    params.Yt = 50.0e6;
    params.Yc = 200.0e6;
    params.S12 = 70.0e6;

    ChangChangFailure model(params);

    // Fiber compression failure
    MaterialState mstate;
    mstate.stress[0] = -1300.0e6;  // > Xc = 1200 MPa

    FailureState fstate;
    model.compute_damage(mstate, fstate, 0.0, 0.0);
    CHECK(fstate.failed, "Fiber compression failure");
    CHECK(fstate.failure_mode == 2, "Mode 2 (fiber compression)");

    // Safe state
    MaterialState mstate_safe;
    mstate_safe.stress[0] = 500.0e6;
    mstate_safe.stress[1] = 20.0e6;
    mstate_safe.stress[3] = 30.0e6;

    FailureState fstate_safe;
    model.compute_damage(mstate_safe, fstate_safe, 0.0, 0.0);
    CHECK(!fstate_safe.failed, "Safe loading: no failure");

    // Stress degradation
    Real sigma[6] = {100.0e6, 50.0e6, 0.0, 30.0e6, 10.0e6, 5.0e6};
    FailureState fstate_deg;
    fstate_deg.history[4] = 1.0;  // Fiber tension damaged
    model.degrade_stress(sigma, fstate_deg);
    CHECK(sigma[0] == 0.0, "σ11 zeroed after fiber failure");
    CHECK(sigma[3] == 0.0, "τ12 zeroed after fiber failure");
    CHECK(sigma[1] == 50.0e6, "σ22 preserved (matrix intact)");
}

// ==========================================================================
// Test 5: GTN Ductile Damage
// ==========================================================================
void test_gtn() {
    std::cout << "\n=== Test 5: GTN Ductile Damage ===\n";

    FailureModelParameters params;
    params.f0 = 0.001;     // Initial void fraction
    params.fN = 0.04;      // Nucleation fraction
    params.sN = 0.1;       // Nucleation spread
    params.epsN = 0.3;     // Nucleation strain
    params.fc = 0.15;      // Coalescence onset
    params.fF = 0.25;      // Final failure
    params.q1 = 1.5;
    params.q2 = 1.0;
    params.q3 = 2.25;

    GTNFailure model(params);

    // Initial state - no damage
    MaterialState mstate;
    mstate.stress[0] = 300.0e6;
    mstate.plastic_strain = 0.0;

    FailureState fstate;
    model.compute_damage(mstate, fstate, 1.0e-6, 1.0e-3);
    CHECK(fstate.history[0] > 0.0, "Void fraction initialized");
    CHECK(!fstate.failed, "No failure at start");

    // Progressive damage with increasing plastic strain
    FailureState fstate_prog;
    Real damage_prev = 0.0;
    bool damage_grows = true;
    for (int step = 0; step < 100; ++step) {
        MaterialState ms;
        ms.stress[0] = 400.0e6;
        ms.stress[1] = 100.0e6;
        ms.stress[2] = 100.0e6;
        ms.plastic_strain = 0.005 * (step + 1);
        ms.vol_strain = 0.001 * (step + 1);

        model.compute_damage(ms, fstate_prog, 1.0e-5, 1.0e-3);
        if (fstate_prog.history[0] < damage_prev && step > 5) {
            damage_grows = false;
        }
        damage_prev = fstate_prog.history[0];
    }
    CHECK(damage_grows, "Void fraction grows monotonically");
    CHECK(fstate_prog.history[0] > params.f0, "Void fraction exceeds initial value");
    CHECK(fstate_prog.history[3] > 0.0, "Nucleated void fraction > 0");

    // Coalescence check
    FailureState fstate_coal;
    fstate_coal.history[0] = 0.20;  // f > fc (0.15)
    MaterialState ms_coal;
    ms_coal.plastic_strain = 0.5;
    ms_coal.vol_strain = 0.1;
    model.compute_damage(ms_coal, fstate_coal, 1.0e-5, 1.0e-3);
    CHECK(fstate_coal.history[1] > fstate_coal.history[0],
          "Effective f* > f after coalescence");
}

// ==========================================================================
// Test 6: GISSMO Mesh-Regularized Damage
// ==========================================================================
void test_gissmo() {
    std::cout << "\n=== Test 6: GISSMO Damage ===\n";

    FailureModelParameters params;
    params.n_exp = 2.0;
    params.dcrit = 0.5;
    params.fadexp = 2.0;
    params.lc_ref = 2.0e-3;  // 2mm reference

    // Setup a simple failure envelope
    params.failure_envelope.num_points = 5;
    params.failure_envelope.x[0] = -0.33; params.failure_envelope.y[0] = 1.0;
    params.failure_envelope.x[1] = 0.0;   params.failure_envelope.y[1] = 0.6;
    params.failure_envelope.x[2] = 0.33;  params.failure_envelope.y[2] = 0.4;
    params.failure_envelope.x[3] = 0.67;  params.failure_envelope.y[3] = 0.25;
    params.failure_envelope.x[4] = 1.0;   params.failure_envelope.y[4] = 0.15;

    GISSMOFailure model(params);

    // No damage before plastic straining
    MaterialState mstate;
    mstate.stress[0] = 100.0e6;
    mstate.stress[1] = 100.0e6;
    mstate.stress[2] = 100.0e6;
    mstate.plastic_strain = 0.0;

    FailureState fstate;
    model.compute_damage(mstate, fstate, 1.0e-6, 2.0e-3);
    CHECK(fstate.damage == 0.0 || fstate.history[0] < 0.01,
          "Minimal damage without plastic strain");

    // Progressive damage
    FailureState fstate_prog;
    Real prev_D = 0.0;
    bool D_grows = true;
    for (int step = 1; step <= 50; ++step) {
        MaterialState ms;
        // Uniaxial tension: η ≈ 1/3
        ms.stress[0] = 400.0e6;
        ms.stress[1] = 0.0;
        ms.stress[2] = 0.0;
        ms.plastic_strain = 0.01 * step;

        model.compute_damage(ms, fstate_prog, 1.0e-5, 2.0e-3);
        if (fstate_prog.history[0] < prev_D && step > 2) {
            D_grows = false;
        }
        prev_D = fstate_prog.history[0];
    }
    CHECK(D_grows, "GISSMO damage grows monotonically");
    CHECK(fstate_prog.history[0] > 0.1, "Significant damage accumulated");

    // Mesh regularization: finer mesh should give different result
    FailureState fstate_fine;
    for (int step = 1; step <= 50; ++step) {
        MaterialState ms;
        ms.stress[0] = 400.0e6;
        ms.plastic_strain = 0.01 * step;
        model.compute_damage(ms, fstate_fine, 1.0e-5, 1.0e-3);  // 1mm (finer)
    }
    FailureState fstate_coarse;
    for (int step = 1; step <= 50; ++step) {
        MaterialState ms;
        ms.stress[0] = 400.0e6;
        ms.plastic_strain = 0.01 * step;
        model.compute_damage(ms, fstate_coarse, 1.0e-5, 4.0e-3);  // 4mm (coarser)
    }
    CHECK(fstate_fine.history[0] != fstate_coarse.history[0],
          "Mesh regularization gives different damage for different element sizes");
}

// ==========================================================================
// Test 7: Tabulated Failure Envelope
// ==========================================================================
void test_tabulated_envelope() {
    std::cout << "\n=== Test 7: Tabulated Failure Envelope ===\n";

    FailureModelParameters params;

    // Triaxiality-dependent failure strain
    params.failure_envelope.num_points = 4;
    params.failure_envelope.x[0] = -0.33;  // Compression
    params.failure_envelope.y[0] = 2.0;    // Very ductile
    params.failure_envelope.x[1] = 0.0;    // Shear
    params.failure_envelope.y[1] = 0.8;
    params.failure_envelope.x[2] = 0.33;   // Uniaxial tension
    params.failure_envelope.y[2] = 0.4;
    params.failure_envelope.x[3] = 0.67;   // Biaxial tension
    params.failure_envelope.y[3] = 0.2;

    TabulatedFailure model(params);

    // Loading under uniaxial tension (η ≈ 0.33, eps_f ≈ 0.4)
    FailureState fstate;
    for (int step = 1; step <= 20; ++step) {
        MaterialState ms;
        ms.stress[0] = 400.0e6;
        ms.plastic_strain = 0.02 * step;  // 2% per step
        model.compute_damage(ms, fstate, 1.0e-5, 1.0e-3);
    }
    CHECK(fstate.damage > 0.0, "Damage accumulated under tension");
    CHECK(fstate.history[2] > 0.0, "Triaxiality tracked");

    // Loading under compression (η < 0, higher failure strain)
    FailureState fstate_comp;
    for (int step = 1; step <= 20; ++step) {
        MaterialState ms;
        ms.stress[0] = -400.0e6;
        ms.stress[1] = -200.0e6;
        ms.stress[2] = -200.0e6;
        ms.plastic_strain = 0.02 * step;
        model.compute_damage(ms, fstate_comp, 1.0e-5, 1.0e-3);
    }
    CHECK(fstate_comp.damage < fstate.damage,
          "Less damage under compression (higher eps_f)");
}

// ==========================================================================
// Test 8: Erosion System Integration
// ==========================================================================
void test_erosion_integration() {
    std::cout << "\n=== Test 8: Erosion System Integration ===\n";

    // Verify new FailureCriterion enum values exist
    FailureCriterion fc_hashin = FailureCriterion::Hashin;
    FailureCriterion fc_tsai = FailureCriterion::TsaiWu;
    FailureCriterion fc_chang = FailureCriterion::ChangChang;
    FailureCriterion fc_gtn = FailureCriterion::GTN;
    FailureCriterion fc_gissmo = FailureCriterion::GISSMO;
    FailureCriterion fc_tab = FailureCriterion::TabulatedEnvelope;

    CHECK(fc_hashin != FailureCriterion::None, "Hashin criterion exists");
    CHECK(fc_tsai != FailureCriterion::None, "TsaiWu criterion exists");
    CHECK(fc_chang != FailureCriterion::None, "ChangChang criterion exists");
    CHECK(fc_gtn != FailureCriterion::None, "GTN criterion exists");
    CHECK(fc_gissmo != FailureCriterion::None, "GISSMO criterion exists");
    CHECK(fc_tab != FailureCriterion::None, "TabulatedEnvelope criterion exists");

    // Type string verification
    CHECK(std::string(FailureModel::to_string(FailureModelType::Hashin)) == "Hashin",
          "Hashin type string");
    CHECK(std::string(FailureModel::to_string(FailureModelType::GTN)) == "GTN",
          "GTN type string");
    CHECK(std::string(FailureModel::to_string(FailureModelType::GISSMO)) == "GISSMO",
          "GISSMO type string");
}

// ==========================================================================
// Test 9: Hashin Fiber Compression
// ==========================================================================
void test_hashin_fiber_compression() {
    std::cout << "\n=== Test 9: Hashin - Fiber Compression ===\n";

    FailureModelParameters params;
    params.Xt = 1500.0e6;
    params.Xc = 1200.0e6;
    params.Yt = 50.0e6;
    params.Yc = 200.0e6;
    params.S12 = 70.0e6;
    params.S23 = 40.0e6;

    HashinFailure model(params);

    // Below compression strength
    MaterialState mstate;
    mstate.stress[0] = -800.0e6;

    FailureState fstate;
    model.compute_damage(mstate, fstate, 0.0, 0.0);
    CHECK(!fstate.failed, "Below Xc: no failure");

    // Above compression strength
    mstate.stress[0] = -1300.0e6;  // |σ11| > Xc
    FailureState fstate2;
    model.compute_damage(mstate, fstate2, 0.0, 0.0);
    CHECK(fstate2.failed, "Above Xc: fiber compression failure");
    CHECK(fstate2.failure_mode == 2, "Failure mode = 2 (fiber compression)");
}

// ==========================================================================
// Test 10: GTN Void Nucleation
// ==========================================================================
void test_gtn_nucleation() {
    std::cout << "\n=== Test 10: GTN Void Nucleation ===\n";

    FailureModelParameters params;
    params.f0 = 0.001;
    params.fN = 0.04;
    params.sN = 0.1;
    params.epsN = 0.3;
    params.fc = 0.15;
    params.fF = 0.25;
    params.q1 = 1.5;

    GTNFailure model(params);

    // Strain near nucleation peak (εp ≈ εN = 0.3)
    FailureState fstate;
    for (int step = 1; step <= 40; ++step) {
        MaterialState ms;
        ms.stress[0] = 300.0e6;
        ms.plastic_strain = 0.01 * step;  // 1% per step, reaches εN at step 30
        model.compute_damage(ms, fstate, 1.0e-5, 1.0e-3);
    }

    CHECK(fstate.history[3] > 0.0, "Nucleated voids present");
    CHECK(fstate.history[0] > params.f0, "Total void fraction exceeds initial");

    // Far from nucleation strain: less nucleation
    FailureState fstate_far;
    MaterialState ms_far;
    ms_far.stress[0] = 300.0e6;
    ms_far.plastic_strain = 0.001;  // Small plastic strain, far from εN
    model.compute_damage(ms_far, fstate_far, 1.0e-5, 1.0e-3);
    CHECK(fstate_far.history[3] < fstate.history[3],
          "Less nucleation far from εN");
}

// ==========================================================================
// Test 11: GISSMO Stress Fading
// ==========================================================================
void test_gissmo_fading() {
    std::cout << "\n=== Test 11: GISSMO Stress Fading ===\n";

    FailureModelParameters params;
    params.n_exp = 2.0;
    params.dcrit = 0.3;
    params.fadexp = 2.0;
    params.lc_ref = 2.0e-3;

    GISSMOFailure model(params);

    // Manually set damage below dcrit
    FailureState fstate_low;
    fstate_low.history[0] = 0.2;  // D < dcrit
    MaterialState ms;
    ms.stress[0] = 400.0e6;
    ms.plastic_strain = 0.0;
    model.compute_damage(ms, fstate_low, 1.0e-5, 2.0e-3);
    CHECK(fstate_low.damage == 0.0, "No stress fading below dcrit");

    // Damage above dcrit: stress fading should kick in
    FailureState fstate_high;
    fstate_high.history[0] = 0.8;  // D > dcrit
    model.compute_damage(ms, fstate_high, 1.0e-5, 2.0e-3);
    CHECK(fstate_high.damage > 0.0, "Stress fading active above dcrit");
    CHECK(fstate_high.damage < 1.0, "Partial fading (not fully failed)");
}

// ==========================================================================
// Main
// ==========================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "NexusSim Wave 2: Failure Models Test\n";
    std::cout << "========================================\n";

    test_hashin_fiber_tension();
    test_hashin_matrix();
    test_tsai_wu();
    test_chang_chang();
    test_gtn();
    test_gissmo();
    test_tabulated_envelope();
    test_erosion_integration();
    test_hashin_fiber_compression();
    test_gtn_nucleation();
    test_gissmo_fading();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    std::cout << "========================================\n";

    return (tests_failed > 0) ? 1 : 0;
}
