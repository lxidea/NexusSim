/**
 * @file failure_wave33_test.cpp
 * @brief Wave 33: Failure Models Test Suite (14 models, ~120 tests)
 *
 * Models tested:
 *   1.  EMCFailure              - Extended Mohr-Coulomb ductile fracture
 *   2.  SahraeiFailure          - Battery cell short-circuit criterion
 *   3.  SyazwanFailure          - Strain-rate dependent ductile failure
 *   4.  VisualFailure           - Visual damage indicator (no deletion)
 *   5.  NXTFailure              - Next-gen composite (fiber/matrix damage)
 *   6.  Gene1Failure            - Polynomial stress-space failure
 *   7.  IniEvoFailure           - Initial damage + linear evolution
 *   8.  AlterFailure            - Alternating-load fatigue (Miner's rule)
 *   9.  BiquadFailure           - Biquadratic failure surface
 *   10. OrthBiquadFailure       - Orthotropic biquadratic failure
 *   11. OrthEnergFailure        - Orthotropic energy-based failure
 *   12. OrthStrainFailure       - Orthotropic max-strain failure
 *   13. TensStrainFailure       - Tensile strain criterion
 *   14. SnConnectFailure        - Spot-weld connection failure
 */

#include <nexussim/physics/failure/failure_wave33.hpp>
#include <iostream>
#include <cmath>
#include <vector>

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
// Helper: create a zeroed MaterialState
// ============================================================================
static MaterialState make_zero_state() {
    MaterialState ms;
    for (int i = 0; i < 6; ++i) {
        ms.stress[i] = 0.0;
        ms.strain[i] = 0.0;
        ms.strain_rate[i] = 0.0;
    }
    ms.plastic_strain = 0.0;
    ms.effective_strain_rate = 0.0;
    ms.temperature = 293.15;
    ms.damage = 0.0;
    ms.vol_strain = 0.0;
    ms.dt = 1.0e-6;
    return ms;
}

// ============================================================================
// 1. EMCFailure
// ============================================================================
static void test_1_emc_failure() {
    std::cout << "\n--- 1. EMCFailure ---\n";

    FailureModelParameters base;
    EMCFailureParams params;
    params.c0 = 100.0e6;
    params.phi = 0.5236; // 30 deg
    params.A_mc = 0.8;
    params.B_mc = 0.6;
    params.n_mc = 0.1;

    EMCFailure model(base, params);

    // (1) Construction
    CHECK(std::string(model.name()) == "EMCFailure", "EMC: model name");

    // (2) Zero damage at zero strain
    {
        MaterialState ms = make_zero_state();
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "EMC: zero damage at zero strain");
        CHECK(!fs.failed, "EMC: not failed at zero strain");
    }

    // (3) Small plastic strain accumulates some damage
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 200.0e6; // uniaxial tension
        ms.plastic_strain = 0.05;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage > 0.0, "EMC: nonzero damage under plastic strain");
        CHECK(fs.damage <= 1.0, "EMC: damage bounded by 1.0");
    }

    // (4) Larger plastic strain -> more damage
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 300.0e6;
        ms.plastic_strain = 0.1;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        Real d1 = fs.damage;

        ms.plastic_strain = 0.3;
        fs.history[4] = 0.1; // previous plastic strain
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage > d1, "EMC: more plastic strain -> more damage");
    }

    // (5) History tracking: triaxiality stored
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 100.0e6;
        ms.stress[1] = 100.0e6;
        ms.stress[2] = 100.0e6;
        ms.plastic_strain = 0.01;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        // High triaxiality for hydrostatic tension
        CHECK(fs.history[2] > 0.5, "EMC: triaxiality stored in history[2]");
    }

    // (6) Fracture strain stored in history
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 200.0e6;
        ms.plastic_strain = 0.05;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.history[1] > 0.0, "EMC: fracture strain eps_f stored in history[1]");
    }

    // (7) Damage monotonically increases (irreversible)
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 200.0e6;
        ms.plastic_strain = 0.2;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        Real d_high = fs.damage;

        // Reduce stress but keep same plastic strain
        ms.stress[0] = 50.0e6;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage >= d_high, "EMC: damage irreversible");
    }

    // (8) Eventual failure at large plastic strain
    {
        FailureState fs;
        for (int i = 0; i < 50; ++i) {
            MaterialState ms = make_zero_state();
            ms.stress[0] = 400.0e6;
            ms.plastic_strain = 0.05 * (i + 1);
            fs.history[4] = 0.05 * i; // previous plastic strain
            model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        }
        CHECK(fs.damage >= 0.5, "EMC: significant damage after many increments");
    }
}

// ============================================================================
// 2. SahraeiFailure
// ============================================================================
static void test_2_sahraei_failure() {
    std::cout << "\n--- 2. SahraeiFailure ---\n";

    FailureModelParameters base;
    SahraeiFailureParams params;
    params.eps_crit = 0.15;
    params.pressure_crit = 50.0e6;
    params.combined_weight = 0.5;

    SahraeiFailure model(base, params);

    // (1) Construction
    CHECK(std::string(model.name()) == "SahraeiFailure", "Sahraei: model name");

    // (2) Zero damage at zero state
    {
        MaterialState ms = make_zero_state();
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "Sahraei: zero damage at zero state");
        CHECK(!fs.failed, "Sahraei: not failed at zero state");
    }

    // (3) Strain-only failure
    {
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 0.30; // 2x eps_crit -> strain_term = 2.0
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        // alpha*2.0 + (1-alpha)*0 = 0.5*2.0 = 1.0
        CHECK(fs.damage >= 1.0, "Sahraei: strain-only failure at 2x eps_crit");
        CHECK(fs.failed, "Sahraei: failed flag set");
    }

    // (4) Pressure-only failure
    {
        MaterialState ms = make_zero_state();
        // Compressive stress -> positive pressure
        ms.stress[0] = -100.0e6;
        ms.stress[1] = -100.0e6;
        ms.stress[2] = -100.0e6;
        // P = -(-100-100-100)/3 = 100 MPa, P/P_crit = 2.0
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        // alpha*0 + (1-alpha)*2.0 = 0.5*2.0 = 1.0
        CHECK(fs.damage >= 1.0, "Sahraei: pressure-only failure");
    }

    // (5) Combined sub-critical: both below threshold
    {
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 0.05; // 0.05/0.15 = 0.333
        ms.stress[0] = -10.0e6;
        ms.stress[1] = -10.0e6;
        ms.stress[2] = -10.0e6; // P = 10 MPa, P/50 = 0.2
        // F = 0.5*0.333 + 0.5*0.2 = 0.267
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(!fs.failed, "Sahraei: sub-critical combined loading");
        CHECK(fs.damage < 1.0, "Sahraei: damage < 1 in sub-critical");
        CHECK(fs.damage > 0.0, "Sahraei: some damage in sub-critical");
    }

    // (6) Tensile pressure does not contribute
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 100.0e6; // tension -> negative pressure
        ms.stress[1] = 100.0e6;
        ms.stress[2] = 100.0e6;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.history[2], 0.0, 1.0e-10, "Sahraei: tensile pressure ignored");
    }

    // (7) Damage bounds [0,1]
    {
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 1.0;
        ms.stress[0] = -200.0e6;
        ms.stress[1] = -200.0e6;
        ms.stress[2] = -200.0e6;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage <= 1.0, "Sahraei: damage capped at 1.0");
        CHECK(fs.damage >= 0.0, "Sahraei: damage >= 0");
    }

    // (8) Irreversible damage
    {
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 0.20;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        Real d1 = fs.damage;

        ms.plastic_strain = 0.01; // reduce
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage >= d1, "Sahraei: damage irreversible");
    }
}

// ============================================================================
// 3. SyazwanFailure
// ============================================================================
static void test_3_syazwan_failure() {
    std::cout << "\n--- 3. SyazwanFailure ---\n";

    FailureModelParameters base;
    SyazwanFailureParams params;
    params.eps_f0 = 0.5;
    params.C_rate = 0.04;
    params.n_rate = 1.0;
    params.eps_dot_ref = 1.0;

    SyazwanFailure model(base, params);

    // (1) Construction
    CHECK(std::string(model.name()) == "SyazwanFailure", "Syazwan: model name");

    // (2) Zero damage at zero strain
    {
        MaterialState ms = make_zero_state();
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "Syazwan: zero damage at zero strain");
    }

    // (3) Quasi-static: failure strain = eps_f0
    {
        MaterialState ms = make_zero_state();
        ms.effective_strain_rate = 1.0; // == ref rate
        ms.plastic_strain = 0.1;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.history[1], 0.5, 0.01, "Syazwan: eps_f = eps_f0 at ref rate");
        CHECK(fs.damage > 0.0, "Syazwan: some damage at 0.1 strain");
    }

    // (4) Higher rate -> higher failure strain
    {
        MaterialState ms1 = make_zero_state();
        ms1.effective_strain_rate = 1.0;
        ms1.plastic_strain = 0.1;
        FailureState fs1;
        model.compute_damage(ms1, fs1, 1.0e-6, 1.0e-3);

        MaterialState ms2 = make_zero_state();
        ms2.effective_strain_rate = 100.0;
        ms2.plastic_strain = 0.1;
        FailureState fs2;
        model.compute_damage(ms2, fs2, 1.0e-6, 1.0e-3);

        CHECK(fs2.history[1] > fs1.history[1], "Syazwan: higher rate -> higher eps_f");
    }

    // (5) More plastic strain -> more damage
    {
        FailureState fs;
        MaterialState ms = make_zero_state();
        ms.effective_strain_rate = 1.0;
        ms.plastic_strain = 0.1;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        Real d1 = fs.damage;

        ms.plastic_strain = 0.3;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage > d1, "Syazwan: more strain -> more damage");
    }

    // (6) Failure at eps_p = eps_f0
    {
        FailureState fs;
        MaterialState ms = make_zero_state();
        ms.effective_strain_rate = 1.0;
        ms.plastic_strain = 0.5; // == eps_f0
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 1.0, 0.01, "Syazwan: failure at eps_p = eps_f0");
    }

    // (7) Damage bounded [0,1]
    {
        FailureState fs;
        MaterialState ms = make_zero_state();
        ms.effective_strain_rate = 1.0;
        ms.plastic_strain = 2.0;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage <= 1.0, "Syazwan: damage <= 1.0");
        CHECK(fs.damage >= 0.0, "Syazwan: damage >= 0.0");
    }

    // (8) Previous strain tracking
    {
        FailureState fs;
        MaterialState ms = make_zero_state();
        ms.effective_strain_rate = 1.0;
        ms.plastic_strain = 0.1;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.history[2], 0.1, 1.0e-10, "Syazwan: previous strain tracked");
    }
}

// ============================================================================
// 4. VisualFailure
// ============================================================================
static void test_4_visual_failure() {
    std::cout << "\n--- 4. VisualFailure ---\n";

    FailureModelParameters base;
    VisualFailureParams params;
    params.d_threshold_low = 0.25;
    params.d_threshold_med = 0.5;
    params.d_threshold_high = 0.75;
    params.eps_ref = 0.1;

    VisualFailure model(base, params);

    // (1) Construction
    CHECK(std::string(model.name()) == "VisualFailure", "Visual: model name");

    // (2) Zero damage at zero strain
    {
        MaterialState ms = make_zero_state();
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "Visual: zero damage at zero strain");
        CHECK(!fs.failed, "Visual: never fails");
    }

    // (3) Low damage level (green)
    {
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 0.02; // 0.02/0.1 = 0.2 < 0.25
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.history[0], 0.0, 1.0e-10, "Visual: level 0 (green)");
        CHECK(!fs.failed, "Visual: not failed at low strain");
    }

    // (4) Medium damage level (yellow)
    {
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 0.03; // 0.03/0.1 = 0.3 -> level 1
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.history[0], 1.0, 1.0e-10, "Visual: level 1 (yellow)");
    }

    // (5) High damage level (orange)
    {
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 0.06; // 0.06/0.1 = 0.6 -> level 2
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.history[0], 2.0, 1.0e-10, "Visual: level 2 (orange)");
    }

    // (6) Critical damage level (red)
    {
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 0.08; // 0.08/0.1 = 0.8 -> level 3
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.history[0], 3.0, 1.0e-10, "Visual: level 3 (red)");
    }

    // (7) Never triggers element deletion even at high strain
    {
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 1.0; // 10x reference
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(!fs.failed, "Visual: never triggers deletion");
        CHECK(fs.damage <= 1.0, "Visual: damage capped at 1.0");
    }

    // (8) Damage tracks plastic strain ratio
    {
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 0.05;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.5, 1.0e-10, "Visual: damage = eps_p/eps_ref");
        CHECK_NEAR(fs.history[1], 0.5, 1.0e-10, "Visual: raw damage in history[1]");
    }

    // (9) No stress degradation
    {
        Real sigma[6] = {100.0e6, 50.0e6, 0.0, 25.0e6, 0.0, 0.0};
        FailureState fs;
        fs.damage = 0.8;
        model.degrade_stress(sigma, fs);
        CHECK_NEAR(sigma[0], 100.0e6, 1.0, "Visual: no stress degradation");
    }
}

// ============================================================================
// 5. NXTFailure
// ============================================================================
static void test_5_nxt_failure() {
    std::cout << "\n--- 5. NXTFailure ---\n";

    FailureModelParameters base;
    NXTFailureParams params;
    params.Xt = 1500.0e6;
    params.Xc = 1200.0e6;
    params.Yt = 50.0e6;
    params.Yc = 200.0e6;
    params.S12 = 75.0e6;
    params.m_fiber = 2.0;
    params.m_matrix = 1.5;

    NXTFailure model(base, params);

    // (1) Construction
    CHECK(std::string(model.name()) == "NXTFailure", "NXT: model name");

    // (2) Zero damage at zero stress
    {
        MaterialState ms = make_zero_state();
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "NXT: zero damage at zero stress");
    }

    // (3) Fiber tension damage
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 2000.0e6; // > Xt = 1500 MPa
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(model.fiber_tension_damage(fs) > 0.0, "NXT: fiber tension damage > 0");
    }

    // (4) Fiber compression damage
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = -1500.0e6; // > Xc = 1200 MPa in compression
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(model.fiber_compression_damage(fs) > 0.0, "NXT: fiber compression damage > 0");
    }

    // (5) Matrix tension damage
    {
        MaterialState ms = make_zero_state();
        ms.stress[1] = 80.0e6; // > Yt = 50 MPa
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(model.matrix_tension_damage(fs) > 0.0, "NXT: matrix tension damage > 0");
    }

    // (6) Matrix compression damage
    {
        MaterialState ms = make_zero_state();
        ms.stress[1] = -300.0e6; // > Yc = 200 MPa in compression
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(model.matrix_compression_damage(fs) > 0.0, "NXT: matrix compression damage > 0");
    }

    // (7) Independent modes: fiber damage does not affect matrix
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 2000.0e6; // fiber tension only
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(model.matrix_tension_damage(fs), 0.0, 1.0e-10,
                   "NXT: fiber load doesn't damage matrix tension");
        CHECK_NEAR(model.matrix_compression_damage(fs), 0.0, 1.0e-10,
                   "NXT: fiber load doesn't damage matrix compression");
    }

    // (8) Damage bounded
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 5000.0e6;
        ms.stress[1] = 500.0e6;
        ms.stress[3] = 500.0e6;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage <= 1.0, "NXT: total damage <= 1.0");
        CHECK(fs.damage >= 0.0, "NXT: total damage >= 0.0");
    }
}

// ============================================================================
// 6. Gene1Failure
// ============================================================================
static void test_6_gene1_failure() {
    std::cout << "\n--- 6. Gene1Failure ---\n";

    FailureModelParameters base;
    Gene1FailureParams params;
    // Set so that 1 GPa in any direction fails
    params.a[0] = 1.0e-18; // s1^2
    params.a[1] = 1.0e-18; // s2^2
    params.a[2] = 0.0;     // cross term
    params.a[3] = 3.0e-18; // s12^2
    params.a[4] = 0.0;
    params.a[5] = 0.0;

    Gene1Failure model(base, params);

    // (1) Construction
    CHECK(std::string(model.name()) == "Gene1Failure", "Gene1: model name");

    // (2) Zero damage at zero stress
    {
        MaterialState ms = make_zero_state();
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "Gene1: zero damage at zero stress");
    }

    // (3) Sub-critical: F < 1
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 500.0e6; // F = 1e-18 * (500e6)^2 = 0.25
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(!fs.failed, "Gene1: sub-critical s1 = 500 MPa");
        CHECK(fs.damage < 1.0, "Gene1: damage < 1 sub-critical");
    }

    // (4) Critical: F = 1
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 1000.0e6; // F = 1e-18 * (1e9)^2 = 1.0
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.failed, "Gene1: failure at F = 1");
        CHECK_NEAR(fs.damage, 1.0, 1.0e-10, "Gene1: damage = 1 at failure");
    }

    // (5) Shear failure
    {
        MaterialState ms = make_zero_state();
        ms.stress[3] = 600.0e6; // F = 3e-18 * (6e8)^2 = 1.08
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.failed, "Gene1: shear failure");
    }

    // (6) Failure index tracking
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 700.0e6; // F = 0.49
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        Real fi = model.failure_index(fs);
        CHECK_NEAR(fi, 0.49, 0.01, "Gene1: failure index = 0.49");
    }

    // (7) Combined loading
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 600.0e6; // 0.36
        ms.stress[1] = 600.0e6; // 0.36
        ms.stress[3] = 200.0e6; // 3*0.04 = 0.12
        // Total = 0.84 < 1
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(!fs.failed, "Gene1: combined sub-critical");
        CHECK(model.failure_index(fs) > 0.5, "Gene1: nonzero failure index in combined loading");
    }

    // (8) Damage bounded
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 5000.0e6;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage <= 1.0, "Gene1: damage capped at 1.0");
    }
}

// ============================================================================
// 7. IniEvoFailure
// ============================================================================
static void test_7_inievo_failure() {
    std::cout << "\n--- 7. IniEvoFailure ---\n";

    FailureModelParameters base;
    IniEvoFailureParams params;
    params.d0 = 0.05;
    params.eps_i = 0.01;
    params.eps_f = 0.2;

    IniEvoFailure model(base, params);

    // (1) Construction
    CHECK(std::string(model.name()) == "IniEvoFailure", "IniEvo: model name");

    // (2) Initial damage at zero strain
    {
        MaterialState ms = make_zero_state();
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.05, 1.0e-10, "IniEvo: initial damage d0 at zero strain");
    }

    // (3) Still d0 below initiation
    {
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 0.005; // < eps_i = 0.01
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.05, 1.0e-10, "IniEvo: d0 below initiation strain");
        CHECK(!fs.failed, "IniEvo: not failed below initiation");
    }

    // (4) Linear evolution between eps_i and eps_f
    {
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 0.105; // midpoint between 0.01 and 0.2
        // D = 0.05 + 0.95 * (0.105 - 0.01) / (0.2 - 0.01) = 0.05 + 0.95 * 0.5 = 0.525
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.525, 0.01, "IniEvo: linear evolution at midpoint");
    }

    // (5) Failure at eps_f
    {
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 0.2;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 1.0, 1.0e-10, "IniEvo: full damage at eps_f");
        CHECK(fs.failed, "IniEvo: failed at eps_f");
    }

    // (6) Beyond eps_f still = 1
    {
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 0.5;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 1.0, 1.0e-10, "IniEvo: damage = 1 beyond eps_f");
    }

    // (7) Damage never below d0
    {
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 0.0;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage >= 0.05, "IniEvo: damage never below d0");
    }

    // (8) Irreversible: damage doesn't decrease when strain decreases
    {
        FailureState fs;
        MaterialState ms = make_zero_state();
        ms.plastic_strain = 0.15;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        Real d_high = fs.damage;

        ms.plastic_strain = 0.05; // lower strain
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage >= d_high, "IniEvo: irreversible damage");
    }

    // (9) Damage monotonically increases with strain
    {
        Real d_prev = 0.0;
        for (int i = 0; i <= 10; ++i) {
            MaterialState ms = make_zero_state();
            ms.plastic_strain = 0.02 * i;
            FailureState fs;
            model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
            CHECK(fs.damage >= d_prev, "IniEvo: monotonically increasing");
            d_prev = fs.damage;
        }
    }
}

// ============================================================================
// 8. AlterFailure
// ============================================================================
static void test_8_alter_failure() {
    std::cout << "\n--- 8. AlterFailure ---\n";

    FailureModelParameters base;
    AlterFailureParams params;
    params.S_f = 200.0e6;
    params.b_basquin = -0.12;
    params.c_coffin = -0.6;
    params.sigma_f_prime = 900.0e6;
    params.eps_f_prime = 0.5;

    AlterFailure model(base, params);

    // (1) Construction
    CHECK(std::string(model.name()) == "AlterFailure", "Alter: model name");

    // (2) Zero damage at zero stress
    {
        MaterialState ms = make_zero_state();
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "Alter: zero damage at zero stress");
    }

    // (3) No damage without cycling (monotonic)
    {
        FailureState fs;
        for (int i = 0; i < 10; ++i) {
            MaterialState ms = make_zero_state();
            ms.stress[0] = 300.0e6 + i * 10.0e6; // monotonically increasing
            model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        }
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "Alter: no damage without cycling");
    }

    // (4) Damage accumulates with cycling
    {
        FailureState fs;
        // Simulate alternating tension-compression
        for (int i = 0; i < 20; ++i) {
            MaterialState ms = make_zero_state();
            Real sign = (i % 2 == 0) ? 1.0 : -1.0;
            ms.stress[0] = sign * 500.0e6; // well above S_f
            model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        }
        CHECK(fs.damage > 0.0, "Alter: damage accumulates with cycling");
    }

    // (5) Below endurance limit: no damage
    {
        FailureState fs;
        for (int i = 0; i < 20; ++i) {
            MaterialState ms = make_zero_state();
            Real sign = (i % 2 == 0) ? 1.0 : -1.0;
            ms.stress[0] = sign * 100.0e6; // below S_f = 200 MPa (amplitude = 100)
            model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        }
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "Alter: no damage below endurance limit");
    }

    // (6) Cycle counting
    {
        FailureState fs;
        for (int i = 0; i < 10; ++i) {
            MaterialState ms = make_zero_state();
            Real sign = (i % 2 == 0) ? 1.0 : -1.0;
            ms.stress[0] = sign * 500.0e6;
            model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        }
        CHECK(model.cycle_count(fs) > 0.0, "Alter: cycle count > 0");
    }

    // (7) Damage bounded
    {
        FailureState fs;
        for (int i = 0; i < 1000; ++i) {
            MaterialState ms = make_zero_state();
            Real sign = (i % 2 == 0) ? 1.0 : -1.0;
            ms.stress[0] = sign * 800.0e6;
            model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        }
        CHECK(fs.damage <= 1.0, "Alter: damage capped at 1.0");
        CHECK(fs.damage >= 0.0, "Alter: damage >= 0.0");
    }

    // (8) Higher amplitude -> faster damage
    {
        auto run_cycles = [&](Real amplitude, int n_cycles) -> Real {
            FailureState fs;
            for (int i = 0; i < n_cycles * 2; ++i) {
                MaterialState ms = make_zero_state();
                Real sign = (i % 2 == 0) ? 1.0 : -1.0;
                ms.stress[0] = sign * amplitude;
                model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
            }
            return fs.damage;
        };
        Real d_low = run_cycles(300.0e6, 10);
        Real d_high = run_cycles(700.0e6, 10);
        CHECK(d_high > d_low, "Alter: higher amplitude -> faster damage");
    }
}

// ============================================================================
// 9. BiquadFailure
// ============================================================================
static void test_9_biquad_failure() {
    std::cout << "\n--- 9. BiquadFailure ---\n";

    FailureModelParameters base;
    BiquadFailureParams params;
    params.F1 = 1.0e-18; // (1 GPa)^2 -> F = 1
    params.F2 = 1.0e-18;
    params.F3 = 0.0;
    params.F4 = 3.0e-18;
    params.F5 = 0.0;
    params.F6 = 0.0;

    BiquadFailure model(base, params);

    // (1) Construction
    CHECK(std::string(model.name()) == "BiquadFailure", "Biquad: model name");

    // (2) Zero damage at zero stress
    {
        MaterialState ms = make_zero_state();
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "Biquad: zero damage at zero stress");
    }

    // (3) Sub-critical
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 500.0e6; // F = 0.25
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(!fs.failed, "Biquad: sub-critical");
        CHECK_NEAR(model.failure_index(fs), 0.25, 0.01, "Biquad: FI = 0.25");
    }

    // (4) Failure at critical
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 1000.0e6; // F = 1.0
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.failed, "Biquad: failure at F = 1");
    }

    // (5) Shear failure
    {
        MaterialState ms = make_zero_state();
        ms.stress[3] = 600.0e6; // F = 3 * 0.36 = 1.08
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.failed, "Biquad: shear failure");
    }

    // (6) Combined loading
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 700.0e6; // 0.49
        ms.stress[1] = 700.0e6; // 0.49 -> total = 0.98
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(!fs.failed, "Biquad: combined just below 1.0");
    }

    // (7) Damage bounded
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 5000.0e6;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage <= 1.0, "Biquad: damage <= 1.0");
        CHECK(fs.damage >= 0.0, "Biquad: damage >= 0.0");
    }

    // (8) Irreversible failure index
    {
        FailureState fs;
        MaterialState ms = make_zero_state();
        ms.stress[0] = 800.0e6;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        Real fi_high = model.failure_index(fs);

        ms.stress[0] = 100.0e6;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(model.failure_index(fs) >= fi_high, "Biquad: FI irreversible");
    }
}

// ============================================================================
// 10. OrthBiquadFailure
// ============================================================================
static void test_10_orthbiquad_failure() {
    std::cout << "\n--- 10. OrthBiquadFailure ---\n";

    FailureModelParameters base;
    OrthBiquadFailureParams params;
    params.Xt = 1500.0e6;
    params.Xc = 1200.0e6;
    params.Yt = 50.0e6;
    params.Yc = 200.0e6;
    params.S12 = 75.0e6;
    params.S23 = 50.0e6;

    OrthBiquadFailure model(base, params);

    // (1) Construction
    CHECK(std::string(model.name()) == "OrthBiquadFailure", "OrthBiquad: model name");

    // (2) Zero damage at zero stress
    {
        MaterialState ms = make_zero_state();
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "OrthBiquad: zero damage at zero stress");
    }

    // (3) Fiber tension failure
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 1600.0e6; // > Xt = 1500 MPa -> (1600/1500)^2 = 1.14
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.failed, "OrthBiquad: fiber tension failure");
    }

    // (4) Matrix tension failure
    {
        MaterialState ms = make_zero_state();
        ms.stress[1] = 55.0e6; // > Yt = 50 MPa -> (55/50)^2 = 1.21
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.failed, "OrthBiquad: matrix tension failure");
    }

    // (5) Compression uses different strengths
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = -1100.0e6; // |s1|/Xc = 1100/1200 = 0.917 < 1
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(!fs.failed, "OrthBiquad: compression below Xc");
    }

    // (6) Shear contribution
    {
        MaterialState ms = make_zero_state();
        ms.stress[3] = 80.0e6; // (80/75)^2 = 1.14
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.failed, "OrthBiquad: shear failure");
    }

    // (7) Interaction term reduces threshold
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 1000.0e6; // (1000/1500)^2 = 0.444
        ms.stress[1] = 30.0e6;   // (30/50)^2 = 0.36, cross = -(1000*30)/(1500*50) = -0.4
        // F = 0.444 + 0.36 + 0 + 0 - 0.4 = 0.404
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(!fs.failed, "OrthBiquad: interaction term effect");
    }

    // (8) Damage bounded
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 5000.0e6;
        ms.stress[1] = 500.0e6;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage <= 1.0, "OrthBiquad: damage <= 1.0");
        CHECK(fs.damage >= 0.0, "OrthBiquad: damage >= 0.0");
    }
}

// ============================================================================
// 11. OrthEnergFailure
// ============================================================================
static void test_11_orthenerg_failure() {
    std::cout << "\n--- 11. OrthEnergFailure ---\n";

    FailureModelParameters base;
    OrthEnergFailureParams params;
    params.G1c = 1.0e6;   // 1 MJ/m^3
    params.G2c = 0.5e6;
    params.G12c = 0.8e6;

    OrthEnergFailure model(base, params);

    // (1) Construction
    CHECK(std::string(model.name()) == "OrthEnergFailure", "OrthEnerg: model name");

    // (2) Zero damage at zero stress/strain
    {
        MaterialState ms = make_zero_state();
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "OrthEnerg: zero damage at zero state");
    }

    // (3) Direction 1 energy accumulation
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 100.0e6;
        ms.strain[0] = 0.01;
        // W_1 = 0.5 * 100e6 * 0.01 = 500,000 J/m^3, D1 = 0.5e6/1.0e6 = 0.5
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(model.direction_damage(fs, 0), 0.5, 0.01, "OrthEnerg: D1 = 0.5");
    }

    // (4) Direction 2 energy accumulation
    {
        MaterialState ms = make_zero_state();
        ms.stress[1] = 100.0e6;
        ms.strain[1] = 0.01;
        // W_2 = 500,000, D2 = 0.5e6/0.5e6 = 1.0
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(model.direction_damage(fs, 1), 1.0, 0.01, "OrthEnerg: D2 = 1.0 (weaker dir)");
        CHECK(fs.failed, "OrthEnerg: failure in weaker direction");
    }

    // (5) Shear energy
    {
        MaterialState ms = make_zero_state();
        ms.stress[3] = 80.0e6;
        ms.strain[3] = 0.01;
        // W_12 = 0.5 * 80e6 * 0.01 = 400,000, D12 = 0.4e6/0.8e6 = 0.5
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(model.direction_damage(fs, 2), 0.5, 0.01, "OrthEnerg: shear D12 = 0.5");
    }

    // (6) Total damage = max of directional
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 50.0e6;  ms.strain[0] = 0.005; // W1 = 125000, D1 = 0.125
        ms.stress[1] = 50.0e6;  ms.strain[1] = 0.005; // W2 = 125000, D2 = 0.25
        ms.stress[3] = 50.0e6;  ms.strain[3] = 0.005; // W12 = 125000, D12 = 0.156
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.25, 0.01, "OrthEnerg: total D = max(D_i)");
    }

    // (7) Damage bounded
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 1000.0e6; ms.strain[0] = 0.1;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage <= 1.0, "OrthEnerg: damage <= 1.0");
    }

    // (8) Irreversible energy tracking
    {
        FailureState fs;
        MaterialState ms = make_zero_state();
        ms.stress[0] = 200.0e6; ms.strain[0] = 0.02;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        Real d1 = fs.damage;

        ms.stress[0] = 10.0e6; ms.strain[0] = 0.001;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage >= d1, "OrthEnerg: damage irreversible");
    }
}

// ============================================================================
// 12. OrthStrainFailure
// ============================================================================
static void test_12_orthstrain_failure() {
    std::cout << "\n--- 12. OrthStrainFailure ---\n";

    FailureModelParameters base;
    OrthStrainFailureParams params;
    params.eps_1t = 0.02;
    params.eps_1c = 0.015;
    params.eps_2t = 0.005;
    params.eps_2c = 0.01;
    params.gamma_12_max = 0.025;

    OrthStrainFailure model(base, params);

    // (1) Construction
    CHECK(std::string(model.name()) == "OrthStrainFailure", "OrthStrain: model name");

    // (2) Zero damage at zero strain
    {
        MaterialState ms = make_zero_state();
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "OrthStrain: zero damage at zero strain");
    }

    // (3) Tension dir 1: below limit
    {
        MaterialState ms = make_zero_state();
        ms.strain[0] = 0.01; // 0.01/0.02 = 0.5
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(!fs.failed, "OrthStrain: dir 1 tension below limit");
        CHECK_NEAR(fs.history[0], 0.5, 0.01, "OrthStrain: f1 = 0.5");
    }

    // (4) Tension dir 1: failure
    {
        MaterialState ms = make_zero_state();
        ms.strain[0] = 0.025; // 0.025/0.02 = 1.25
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.failed, "OrthStrain: dir 1 tension failure");
        CHECK(fs.failure_mode == 1, "OrthStrain: failure mode = 1");
    }

    // (5) Compression dir 1: uses eps_1c
    {
        MaterialState ms = make_zero_state();
        ms.strain[0] = -0.01; // |0.01|/0.015 = 0.667
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(!fs.failed, "OrthStrain: dir 1 compression below eps_1c");
    }

    // (6) Tension dir 2: most restrictive
    {
        MaterialState ms = make_zero_state();
        ms.strain[1] = 0.006; // 0.006/0.005 = 1.2
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.failed, "OrthStrain: dir 2 tension failure (weakest direction)");
        CHECK(fs.failure_mode == 2, "OrthStrain: failure mode = 2");
    }

    // (7) Shear failure
    {
        MaterialState ms = make_zero_state();
        ms.strain[3] = 0.03; // 0.03/0.025 = 1.2
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.failed, "OrthStrain: shear failure");
        CHECK(fs.failure_mode == 3, "OrthStrain: failure mode = 3 (shear)");
    }

    // (8) Damage bounded
    {
        MaterialState ms = make_zero_state();
        ms.strain[0] = 0.1;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage <= 1.0, "OrthStrain: damage <= 1.0");
        CHECK(fs.damage >= 0.0, "OrthStrain: damage >= 0.0");
    }

    // (9) Irreversible
    {
        FailureState fs;
        MaterialState ms = make_zero_state();
        ms.strain[0] = 0.015;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        Real d1 = fs.damage;

        ms.strain[0] = 0.001;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage >= d1, "OrthStrain: damage irreversible");
    }
}

// ============================================================================
// 13. TensStrainFailure
// ============================================================================
static void test_13_tensstrain_failure() {
    std::cout << "\n--- 13. TensStrainFailure ---\n";

    FailureModelParameters base;
    TensStrainFailureParams params;
    params.eps_crit = 0.01;
    params.damage_evolution_exponent = 2.0;

    TensStrainFailure model(base, params);

    // (1) Construction
    CHECK(std::string(model.name()) == "TensStrainFailure", "TensStrain: model name");

    // (2) Zero damage below critical strain
    {
        MaterialState ms = make_zero_state();
        ms.strain[0] = 0.005; // below eps_crit
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "TensStrain: no damage below eps_crit");
    }

    // (3) Damage onset at eps_crit
    {
        MaterialState ms = make_zero_state();
        ms.strain[0] = 0.015; // excess = (0.015 - 0.01)/0.01 = 0.5, D = 0.5^2 = 0.25
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.25, 0.01, "TensStrain: D = 0.25 at 1.5x eps_crit");
    }

    // (4) Failure at 2x eps_crit
    {
        MaterialState ms = make_zero_state();
        ms.strain[0] = 0.02; // excess = 1.0, D = 1.0^2 = 1.0
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 1.0, 0.01, "TensStrain: failure at 2x eps_crit");
        CHECK(fs.failed, "TensStrain: failed flag set");
    }

    // (5) Compressive strain does not trigger
    {
        MaterialState ms = make_zero_state();
        ms.strain[0] = -0.05; // large compression
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "TensStrain: compression doesn't trigger");
    }

    // (6) Uses max principal strain (including shear)
    {
        MaterialState ms = make_zero_state();
        ms.strain[0] = 0.008;
        ms.strain[1] = -0.008;
        ms.strain[3] = 0.02; // large shear -> increases principal strain
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage > 0.0, "TensStrain: shear contributes to principal strain");
    }

    // (7) Damage bounded
    {
        MaterialState ms = make_zero_state();
        ms.strain[0] = 0.1;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage <= 1.0, "TensStrain: damage capped at 1.0");
    }

    // (8) Irreversible
    {
        FailureState fs;
        MaterialState ms = make_zero_state();
        ms.strain[0] = 0.018;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        Real d1 = fs.damage;

        ms.strain[0] = 0.005;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage >= d1, "TensStrain: damage irreversible");
    }

    // (9) Power law exponent effect
    {
        TensStrainFailureParams params_low;
        params_low.eps_crit = 0.01;
        params_low.damage_evolution_exponent = 1.0; // linear
        TensStrainFailure model_low(base, params_low);

        MaterialState ms = make_zero_state();
        ms.strain[0] = 0.015;

        FailureState fs_quad, fs_lin;
        model.compute_damage(ms, fs_quad, 1.0e-6, 1.0e-3);
        model_low.compute_damage(ms, fs_lin, 1.0e-6, 1.0e-3);

        CHECK(fs_lin.damage > fs_quad.damage,
              "TensStrain: linear exponent gives more damage than quadratic");
    }
}

// ============================================================================
// 14. SnConnectFailure
// ============================================================================
static void test_14_snconnect_failure() {
    std::cout << "\n--- 14. SnConnectFailure ---\n";

    FailureModelParameters base;
    SnConnectFailureParams params;
    params.F_normal_max = 5000.0;
    params.F_shear_max = 8000.0;
    params.n_norm = 2.0;
    params.n_shear = 2.0;
    params.area = 2.0e-5; // 20 mm^2

    SnConnectFailure model(base, params);

    // (1) Construction
    CHECK(std::string(model.name()) == "SnConnectFailure", "SnConnect: model name");

    // (2) Zero damage at zero stress
    {
        MaterialState ms = make_zero_state();
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10, "SnConnect: zero damage at zero stress");
    }

    // (3) Normal force failure
    {
        MaterialState ms = make_zero_state();
        // F_n = sigma_n * area, need F_n > 5000 N
        // sigma_n = 5000 / 2e-5 = 250 MPa
        ms.stress[2] = 300.0e6; // F_n = 6000 N > 5000
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.failed, "SnConnect: normal force failure");
        CHECK(fs.failure_mode == 1, "SnConnect: normal failure mode");
    }

    // (4) Shear force failure
    {
        MaterialState ms = make_zero_state();
        // F_s = sqrt(tau_xz^2 + tau_yz^2) * area, need F_s > 8000
        // tau = 8000 / 2e-5 = 400 MPa
        ms.stress[5] = 500.0e6; // F_s = 10000 > 8000
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.failed, "SnConnect: shear force failure");
        CHECK(fs.failure_mode == 2, "SnConnect: shear failure mode");
    }

    // (5) Sub-critical loading
    {
        MaterialState ms = make_zero_state();
        ms.stress[2] = 100.0e6;  // F_n = 2000, r_n = 0.4, r_n^2 = 0.16
        ms.stress[5] = 100.0e6;  // F_s = 2000, r_s = 0.25, r_s^2 = 0.0625
        // FI = 0.16 + 0.0625 = 0.2225
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(!fs.failed, "SnConnect: sub-critical combined");
        CHECK(fs.damage > 0.0, "SnConnect: nonzero damage under load");
        CHECK(fs.damage < 1.0, "SnConnect: damage < 1 sub-critical");
    }

    // (6) Combined normal + shear failure
    {
        MaterialState ms = make_zero_state();
        ms.stress[2] = 200.0e6;  // F_n = 4000, r_n = 0.8, r_n^2 = 0.64
        ms.stress[5] = 300.0e6;  // F_s = 6000, r_s = 0.75, r_s^2 = 0.5625
        // FI = 0.64 + 0.5625 = 1.2025
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.failed, "SnConnect: combined N+S failure");
    }

    // (7) Damage bounded
    {
        MaterialState ms = make_zero_state();
        ms.stress[2] = 1000.0e6;
        ms.stress[5] = 1000.0e6;
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage <= 1.0, "SnConnect: damage <= 1.0");
    }

    // (8) Irreversible
    {
        FailureState fs;
        MaterialState ms = make_zero_state();
        ms.stress[2] = 200.0e6;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        Real d1 = fs.damage;

        ms.stress[2] = 10.0e6;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK(fs.damage >= d1, "SnConnect: damage irreversible");
    }

    // (9) Force ratios stored in history
    {
        MaterialState ms = make_zero_state();
        ms.stress[2] = 125.0e6; // F_n = 2500, r_n = 0.5
        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 1.0e-3);
        CHECK_NEAR(fs.history[1], 0.5, 0.01, "SnConnect: normal force ratio in history[1]");
    }
}

// ==========================================================================
// Main
// ==========================================================================
int main() {
    std::cout << "Wave 33 Failure Models Test Suite\n";
    std::cout << "===================================\n";

    test_1_emc_failure();
    test_2_sahraei_failure();
    test_3_syazwan_failure();
    test_4_visual_failure();
    test_5_nxt_failure();
    test_6_gene1_failure();
    test_7_inievo_failure();
    test_8_alter_failure();
    test_9_biquad_failure();
    test_10_orthbiquad_failure();
    test_11_orthenerg_failure();
    test_12_orthstrain_failure();
    test_13_tensstrain_failure();
    test_14_snconnect_failure();

    std::cout << "\n===================================\n";
    std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed\n";

    if (tests_failed > 0) {
        std::cout << "SOME TESTS FAILED!\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED!\n";
    return 0;
}
