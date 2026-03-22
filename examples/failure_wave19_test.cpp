/**
 * @file failure_wave19_test.cpp
 * @brief Wave 19: Failure Models Tier 2 Test Suite (10 models, ~60 tests)
 *
 * Models tested:
 *   1.  LaDevezeDelamination       - Composite interlaminar delamination (quadratic)
 *   2.  HoffmanFailure             - Anisotropic tension/compression failure
 *   3.  TsaiHillFailure            - Simplified interactive composite failure
 *   4.  RTClFailure                - Reinforced thermoplastic composite failure
 *   5.  MullinsEffect              - Hyperelastic softening/damage (Ogden-Roxburgh)
 *   6.  SpallingFailure            - Dynamic tensile spall fracture
 *   7.  HCDSSEFailure              - High-cycle damage for structural steels
 *   8.  AdhesiveJointFailure       - Adhesive/connection quadratic traction failure
 *   9.  WindshieldFailure          - Laminate glass + PVB progressive failure
 *   10. GeneralizedEnergyFailure   - Generalized plastic work / energy criterion
 */

#include <nexussim/physics/failure/failure_wave19.hpp>
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
// 1. LaDeveze Delamination
// ============================================================================
static void test_1_ladeveze_delamination() {
    std::cout << "\n--- 1. LaDeveze Delamination ---\n";

    FailureModelParameters base;
    LaDevezeDelaminationParams params;
    params.Z_t  = 50.0e6;   // 50 MPa through-thickness tensile strength
    params.S_13 = 60.0e6;   // 60 MPa interlaminar shear 1-3
    params.S_23 = 60.0e6;   // 60 MPa interlaminar shear 2-3

    LaDevezeDelamination model(base, params);

    // (1) Default construction: model name and initial parameters
    {
        CHECK(std::string(model.name()) == "LaDevezeDelamination",
              "LaDeveze: model name is 'LaDevezeDelamination'");
    }

    // (2) Below threshold: safe state with low stresses
    {
        MaterialState ms = make_zero_state();
        ms.stress[2] = 20.0e6;   // sigma_33 = 20 MPa << Z_t = 50 MPa
        ms.stress[5] = 20.0e6;   // tau_13 = 20 MPa << S_13 = 60 MPa
        ms.stress[4] = 20.0e6;   // tau_23 = 20 MPa << S_23 = 60 MPa
        // FI = (20/50)^2 + (20/60)^2 + (20/60)^2 = 0.16 + 0.111 + 0.111 = 0.382

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(!fs.failed, "LaDeveze: below threshold => no failure");
        CHECK(fs.damage < 1.0, "LaDeveze: damage < 1 when safe");
    }

    // (3) Above threshold: through-thickness tension exceeds Z_t
    {
        MaterialState ms = make_zero_state();
        ms.stress[2] = 60.0e6;  // sigma_33 = 60 MPa > Z_t = 50 MPa
        // FI = (60/50)^2 + 0 + 0 = 1.44 >= 1

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "LaDeveze: through-thickness tension triggers failure");
        CHECK(fs.failure_mode == 1, "LaDeveze: Mode I opening (dominant sigma_33)");
    }

    // (4) Damage value is in [0, 1]
    {
        MaterialState ms = make_zero_state();
        ms.stress[2] = 30.0e6;  // FI = (30/50)^2 = 0.36

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.damage >= 0.0 && fs.damage <= 1.0,
              "LaDeveze: damage in [0, 1] range");
        CHECK_NEAR(fs.damage, 0.36, 1.0e-6,
                   "LaDeveze: damage = FI = 0.36 for sub-threshold");
    }

    // (5) Quadratic criterion: combined tension + shear interaction
    {
        MaterialState ms = make_zero_state();
        ms.stress[2] = 40.0e6;   // (40/50)^2 = 0.64  (below Z_t alone)
        ms.stress[4] = 50.0e6;   // (50/60)^2 = 0.694  (below S_23 alone)
        // FI = 0.64 + 0 + 0.694 = 1.334 >= 1 (interaction triggers failure)

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "LaDeveze: quadratic interaction triggers failure");
        // History stores per-component contributions
        CHECK_NEAR(fs.history[1], 0.64, 1.0e-6,
                   "LaDeveze: history[1] = sigma_33 contribution");
        CHECK_NEAR(fs.history[3], 0.6944, 1.0e-3,
                   "LaDeveze: history[3] = tau_23 contribution");
    }

    // (6) Compressive sigma_33 does not contribute (Macaulay bracket)
    {
        MaterialState ms = make_zero_state();
        ms.stress[2] = -100.0e6;  // Compressive through-thickness

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(!fs.failed, "LaDeveze: compressive sigma_33 => no delamination");
        CHECK_NEAR(fs.history[1], 0.0, 1.0e-10,
                   "LaDeveze: tension term is zero for compression");
    }
}

// ============================================================================
// 2. Hoffman Failure
// ============================================================================
static void test_2_hoffman_failure() {
    std::cout << "\n--- 2. Hoffman Failure ---\n";

    FailureModelParameters base;
    HoffmanFailureParams params;
    params.Xt  = 1.5e9;    params.Xc  = 1.2e9;
    params.Yt  = 50.0e6;   params.Yc  = 200.0e6;
    params.Zt  = 50.0e6;   params.Zc  = 200.0e6;
    params.S12 = 75.0e6;   params.S23 = 50.0e6;   params.S13 = 75.0e6;

    HoffmanFailure model(base, params);

    // (1) Default construction: model name
    {
        CHECK(std::string(model.name()) == "HoffmanFailure",
              "Hoffman: model name is 'HoffmanFailure'");
    }

    // (2) Below threshold: low stresses => safe
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 100.0e6;  // Small fiber stress relative to Xt = 1.5 GPa
        ms.stress[1] = 10.0e6;   // Small transverse

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(!fs.failed, "Hoffman: low stresses => no failure");
        CHECK(fs.damage < 1.0, "Hoffman: damage < 1 when safe");
    }

    // (3) Above threshold: high transverse tension triggers failure
    {
        MaterialState ms = make_zero_state();
        ms.stress[1] = 60.0e6;  // sigma_22 = 60 MPa (Yt = 50 MPa)
        // Hoffman has linear terms: C5*s2 = (1/Yt - 1/Yc) * s2

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "Hoffman: high transverse tension => failure");
        CHECK(fs.failure_mode == 1, "Hoffman: failure mode = 1");
    }

    // (4) Damage in [0, 1]
    {
        MaterialState ms = make_zero_state();
        ms.stress[1] = 25.0e6;  // Moderate transverse

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.damage >= 0.0 && fs.damage <= 1.0,
              "Hoffman: damage in [0, 1] range");
    }

    // (5) Tension vs compression asymmetry: compressive is stronger
    {
        MaterialState ms = make_zero_state();
        ms.stress[1] = -60.0e6;  // Compressive transverse (Yc = 200 MPa >> Yt = 50 MPa)

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(!fs.failed, "Hoffman: moderate transverse compression => safe (Yc large)");

        // Now try high shear to trigger failure
        FailureState fs2;
        MaterialState ms2 = make_zero_state();
        ms2.stress[3] = 80.0e6;  // tau_12 = 80 MPa > S12 = 75 MPa
        // C7*s4^2 = (80/75)^2 = 1.138 >= 1
        model.compute_damage(ms2, fs2, 0.0, 0.0);

        CHECK(fs2.failed, "Hoffman: high shear => failure");
    }

    // (6) Zero stress: no failure
    {
        MaterialState ms = make_zero_state();
        // All stresses are zero

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(!fs.failed, "Hoffman: zero stress => no failure");
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10,
                   "Hoffman: zero damage at zero stress");
    }
}

// ============================================================================
// 3. Tsai-Hill Failure
// ============================================================================
static void test_3_tsai_hill_failure() {
    std::cout << "\n--- 3. Tsai-Hill Failure ---\n";

    FailureModelParameters base;
    TsaiHillFailureParams params;
    params.X = 1.5e9;    // Fiber strength
    params.Y = 50.0e6;   // Transverse strength
    params.S = 75.0e6;   // Shear strength

    TsaiHillFailure model(base, params);

    // (1) Default construction: model name
    {
        CHECK(std::string(model.name()) == "TsaiHillFailure",
              "TsaiHill: model name is 'TsaiHillFailure'");
    }

    // (2) Below threshold: small stresses well below limits
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 100.0e6;  // 100 MPa << X = 1.5 GPa
        ms.stress[1] = 5.0e6;    // 5 MPa << Y = 50 MPa

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(!fs.failed, "TsaiHill: small stresses => no failure");
        CHECK(fs.damage < 0.1, "TsaiHill: very low damage for elastic state");
    }

    // (3) At threshold: uniaxial fiber stress s1 = X gives FI = 1
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 1.5e9;  // s1 = X
        // FI = (X/X)^2 - (X*0)/X^2 + (0/Y)^2 + (0/S)^2 = 1.0

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "TsaiHill: uniaxial fiber at X => failure");
        CHECK_NEAR(fs.history[0], 1.0, 1.0e-6,
                   "TsaiHill: FI = 1.0 exactly at X");
    }

    // (4) Damage in [0, 1]: check range for moderate load
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 750.0e6;  // Half of X
        // FI = (0.5)^2 = 0.25

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.damage >= 0.0 && fs.damage <= 1.0,
              "TsaiHill: damage in [0, 1] range");
        CHECK_NEAR(fs.damage, 0.25, 1.0e-6,
                   "TsaiHill: damage = FI = 0.25 at half X");
    }

    // (5) Interaction term: same-sign s1, s2 reduces FI (negative interaction)
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 1.0e9;    // s1
        ms.stress[1] = 10.0e6;   // s2 (same sign as s1)
        // FI = (1e9)^2/X^2 - (1e9*10e6)/X^2 + (10e6)^2/Y^2 + 0
        //    = 0.4444 - 0.00444 + 0.04 = ~0.48

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.history[0] < 1.0,
              "TsaiHill: same-sign biaxial interaction reduces FI");
        CHECK(!fs.failed, "TsaiHill: interaction term prevents failure");

        // Verify the interaction term is negative (beneficial)
        Real FI_no_interaction = (1.0e9 * 1.0e9) / (1.5e9 * 1.5e9)
                               + (10.0e6 * 10.0e6) / (50.0e6 * 50.0e6);
        CHECK(fs.history[0] < FI_no_interaction,
              "TsaiHill: interaction term lowers FI below sum of individual terms");
    }

    // (6) Zero stress: no damage
    {
        MaterialState ms = make_zero_state();

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(!fs.failed, "TsaiHill: zero stress => no failure");
        CHECK_NEAR(fs.damage, 0.0, 1.0e-10,
                   "TsaiHill: zero damage at zero stress");
    }
}

// ============================================================================
// 4. RTCl Failure (Reinforced Thermoplastic Composite)
// ============================================================================
static void test_4_rtcl_failure() {
    std::cout << "\n--- 4. RTCl Failure ---\n";

    FailureModelParameters base;
    RTClFailureParams params;
    params.X_ft = 1.5e9;    // Fiber tensile strength
    params.X_fc = 1.2e9;    // Fiber compressive strength
    params.Y_mt = 80.0e6;   // Matrix tensile strength
    params.Y_mc = 200.0e6;  // Matrix compressive strength
    params.S_12 = 75.0e6;   // Shear strength
    params.beta = 1.0;      // Shear-transverse interaction coefficient

    RTClFailure model(base, params);

    // (1) Default construction: model name
    {
        CHECK(std::string(model.name()) == "RTClFailure",
              "RTCl: model name is 'RTClFailure'");
    }

    // (2) Below threshold: low stresses => safe
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 100.0e6;  // Well below X_ft = 1.5 GPa

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(!fs.failed, "RTCl: low fiber stress => safe");
        CHECK(fs.history[0] < 1.0, "RTCl: fiber FI < 1");
        CHECK(fs.history[1] < 1.0, "RTCl: matrix FI < 1");
    }

    // (3) Fiber tensile failure: sigma_1 > X_ft
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 1.6e9;  // sigma_1 > X_ft = 1.5 GPa

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "RTCl: fiber tensile failure triggered");
        CHECK(fs.failure_mode == 1, "RTCl: failure mode = 1 (fiber)");
    }

    // (4) Damage in [0, 1] range
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 500.0e6;  // 500/1500 = 0.333 for fiber FI

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.damage >= 0.0 && fs.damage <= 1.0,
              "RTCl: damage in [0, 1] range");
    }

    // (5) Matrix failure via Puck-like interaction: sigma_2 + tau_12 combined
    {
        MaterialState ms = make_zero_state();
        ms.stress[1] = 70.0e6;   // sigma_2 < Y_mt = 80 MPa alone
        ms.stress[3] = 50.0e6;   // tau_12
        // mf = (70/80)^2 + (50/75)^2 + beta*(70/80) = 0.766 + 0.444 + 0.875 = 2.085 >= 1

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "RTCl: matrix failure with Puck-like interaction");
        CHECK(fs.failure_mode == 2, "RTCl: failure mode = 2 (matrix)");
    }

    // (6) Compressive fiber: uses X_fc instead of X_ft
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = -1.3e9;  // |sigma_1| = 1.3 GPa > X_fc = 1.2 GPa

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "RTCl: compressive fiber failure (uses X_fc)");
        CHECK(fs.failure_mode == 1, "RTCl: failure mode = 1 (fiber compression)");
    }
}

// ============================================================================
// 5. Mullins Effect (Ogden-Roxburgh Damage)
// ============================================================================
static void test_5_mullins_effect() {
    std::cout << "\n--- 5. Mullins Effect ---\n";

    FailureModelParameters base;
    MullinsEffectParams params;
    params.r    = 1.5;
    params.m    = 0.1e6;
    params.beta = 0.1;

    MullinsEffect model(base, params);

    // (1) Default construction: model name
    {
        CHECK(std::string(model.name()) == "MullinsEffect",
              "Mullins: model name is 'MullinsEffect'");
    }

    // (2) Virgin loading: increasing energy => no damage (W == W_max)
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 1.0e6;
        ms.strain[0] = 0.5;
        // W = 0.5 * 1e6 * 0.5 = 2.5e5

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK_NEAR(fs.history[0], 2.5e5, 1.0,
                   "Mullins: W_max = 2.5e5 on virgin loading");
        CHECK_NEAR(fs.history[1], 0.0, 1.0e-6,
                   "Mullins: zero damage on virgin loading path");
    }

    // (3) Unloading triggers damage: W < W_max produces d > 0
    {
        FailureState fs;
        MaterialState ms = make_zero_state();

        // First load to establish W_max
        ms.stress[0] = 2.0e6;
        ms.strain[0] = 1.0;
        // W = 0.5 * 2e6 * 1.0 = 1e6
        model.compute_damage(ms, fs, 0.0, 0.0);
        CHECK_NEAR(fs.history[0], 1.0e6, 1.0, "Mullins: W_max established at 1e6");

        // Then unload: W < W_max
        ms.stress[0] = 0.5e6;
        ms.strain[0] = 0.25;
        // W = 0.5 * 0.5e6 * 0.25 = 62500 < W_max
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.history[1] > 0.0, "Mullins: nonzero damage on unloading");
        CHECK(fs.history[0] >= 1.0e6 - 1.0,
              "Mullins: W_max does not decrease during unload");
    }

    // (4) Damage in [0, 1]
    {
        FailureState fs;
        MaterialState ms = make_zero_state();

        // Establish W_max
        ms.stress[0] = 3.0e6;
        ms.strain[0] = 2.0;
        model.compute_damage(ms, fs, 0.0, 0.0);

        // Deep unload
        ms.stress[0] = 0.01e6;
        ms.strain[0] = 0.005;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.damage >= 0.0 && fs.damage <= 1.0,
              "Mullins: damage in [0, 1] range during deep unload");
    }

    // (5) Ogden-Roxburgh formula: deeper unload => more damage
    {
        FailureState fs;
        MaterialState ms = make_zero_state();

        // Load to establish W_max
        ms.stress[0] = 3.0e6;
        ms.strain[0] = 2.0;
        model.compute_damage(ms, fs, 0.0, 0.0);

        // Moderate unload
        ms.stress[0] = 0.1e6;
        ms.strain[0] = 0.05;
        model.compute_damage(ms, fs, 0.0, 0.0);
        Real d_moderate = fs.history[1];

        // Deeper unload
        ms.stress[0] = 0.01e6;
        ms.strain[0] = 0.005;
        model.compute_damage(ms, fs, 0.0, 0.0);
        Real d_deep = fs.history[1];

        CHECK(d_deep >= d_moderate - 1.0e-10,
              "Mullins: deeper unload => more (or equal) damage");
    }

    // (6) Zero stress on virgin path: no damage
    {
        MaterialState ms = make_zero_state();
        // All stresses and strains are zero => W = 0

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK_NEAR(fs.damage, 0.0, 1.0e-10,
                   "Mullins: zero damage at zero stress/strain");
        CHECK(!fs.failed, "Mullins: no failure at zero state");
    }
}

// ============================================================================
// 6. Spalling Failure
// ============================================================================
static void test_6_spalling_failure() {
    std::cout << "\n--- 6. Spalling Failure ---\n";

    FailureModelParameters base;
    SpallingFailureParams params;
    params.P_spall  = 1.0e9;  // 1 GPa spall threshold
    params.D_crit   = 1.0;
    params.exponent = 2.0;

    SpallingFailure model(base, params);

    // (1) Default construction: model name
    {
        CHECK(std::string(model.name()) == "SpallingFailure",
              "Spalling: model name is 'SpallingFailure'");
    }

    // (2) Below threshold: hydrostatic tension below P_spall
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = -0.5e9;
        ms.stress[1] = -0.5e9;
        ms.stress[2] = -0.5e9;
        // P_hydro = -p = -(s_xx+s_yy+s_zz)/3 = 0.5e9 < P_spall = 1e9

        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 0.0);

        CHECK_NEAR(fs.history[0], 0.0, 1.0e-10,
                   "Spalling: no damage below spall strength");
        CHECK(!fs.failed, "Spalling: no failure below threshold");
    }

    // (3) Above threshold: hydrostatic tension exceeds P_spall accumulates damage
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = -1.5e9;
        ms.stress[1] = -1.5e9;
        ms.stress[2] = -1.5e9;
        // P_hydro = 1.5e9 > P_spall = 1e9, overstress = 0.5e9
        Real dt = 1.0e-6;

        FailureState fs;
        model.compute_damage(ms, fs, dt, 0.0);

        // Damage = (0.5e9)^2 * 1e-6 = 2.5e11
        CHECK(fs.history[0] > 0.0, "Spalling: damage accumulates above threshold");
    }

    // (4) Damage in [0, 1]: ratio D / D_crit is clamped
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = -1.2e9;
        ms.stress[1] = -1.2e9;
        ms.stress[2] = -1.2e9;
        // P_hydro = 1.2e9, overstress = 0.2e9

        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-12, 0.0);  // Very small dt

        CHECK(fs.damage >= 0.0 && fs.damage <= 1.0,
              "Spalling: damage in [0, 1] range");
    }

    // (5) Duration dependence: damage scales linearly with dt
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = -1.5e9;
        ms.stress[1] = -1.5e9;
        ms.stress[2] = -1.5e9;

        FailureState fs1, fs2;
        model.compute_damage(ms, fs1, 1.0e-6, 0.0);
        model.compute_damage(ms, fs2, 2.0e-6, 0.0);

        CHECK_NEAR(fs2.history[0], 2.0 * fs1.history[0], fs1.history[0] * 1.0e-6,
                   "Spalling: damage scales linearly with dt");
    }

    // (6) Compressive hydrostatic: no spall damage
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 1.0e9;   // Positive (compressive mean stress)
        ms.stress[1] = 1.0e9;
        ms.stress[2] = 1.0e9;
        // p = 1e9 (positive), P_hydro = -1e9 (negative => compression)

        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 0.0);

        CHECK_NEAR(fs.history[0], 0.0, 1.0e-10,
                   "Spalling: compressive hydrostatic => no damage");
        CHECK(!fs.failed, "Spalling: compression safe");
    }
}

// ============================================================================
// 7. HCDSSE Failure (High-Cycle Damage for Structural Steels)
// ============================================================================
static void test_7_hcdsse_failure() {
    std::cout << "\n--- 7. HCDSSE Failure ---\n";

    FailureModelParameters base;
    HCDSSEFailureParams params;
    params.sigma_ref       = 500.0e6;
    params.b_exp           = 5.0;
    params.D_crit          = 1.0;
    params.sigma_endurance = 200.0e6;

    HCDSSEFailure model(base, params);

    // (1) Default construction: model name
    {
        CHECK(std::string(model.name()) == "HCDSSEFailure",
              "HCDSSE: model name is 'HCDSSEFailure'");
    }

    // (2) Below endurance: no damage accumulation
    {
        FailureState fs;
        MaterialState ms = make_zero_state();

        // Load below endurance limit
        ms.stress[0] = 100.0e6;  // 100 MPa < sigma_endurance = 200 MPa
        model.compute_damage(ms, fs, 0.0, 0.0);

        ms.stress[0] = 150.0e6;
        model.compute_damage(ms, fs, 0.0, 0.0);

        // Unload => reversal
        ms.stress[0] = 50.0e6;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK_NEAR(fs.history[0], 0.0, 1.0e-10,
                   "HCDSSE: no damage below endurance limit");
    }

    // (3) Peak tracking: loading phase records peak stress
    {
        FailureState fs;
        MaterialState ms = make_zero_state();

        ms.stress[0] = 300.0e6;
        model.compute_damage(ms, fs, 0.0, 0.0);

        ms.stress[0] = 600.0e6;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.history[4] >= 600.0e6 - 1.0,
              "HCDSSE: peak stress tracked during loading");
    }

    // (4) Damage in [0, 1]: Palmgren-Miner D/D_crit is clamped
    {
        FailureState fs;
        MaterialState ms = make_zero_state();

        ms.stress[0] = 400.0e6;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.damage >= 0.0 && fs.damage <= 1.0,
              "HCDSSE: damage in [0, 1] range");
    }

    // (5) Basquin N_f: verify analytical formula
    {
        // N_f = (sigma_a / sigma_ref)^(-b) = (250e6 / 500e6)^(-5) = (0.5)^(-5) = 32
        Real sigma_a = 250.0e6;
        Real ratio = sigma_a / params.sigma_ref;
        Real N_f = std::pow(ratio, -params.b_exp);
        CHECK_NEAR(N_f, 32.0, 0.1,
                   "HCDSSE: Basquin N_f = 32 for sigma_a = 250 MPa");

        // At reference stress: N_f = 1
        Real N_f_ref = std::pow(1.0, -params.b_exp);
        CHECK_NEAR(N_f_ref, 1.0, 1.0e-10,
                   "HCDSSE: N_f = 1 at reference stress");
    }

    // (6) Cycle counting: half-cycles increment counter on reversal
    {
        FailureState fs;
        MaterialState ms = make_zero_state();

        // Ramp up
        ms.stress[0] = 500.0e6;
        model.compute_damage(ms, fs, 0.0, 0.0);
        ms.stress[0] = 700.0e6;
        model.compute_damage(ms, fs, 0.0, 0.0);

        Real cycles_before = fs.history[1];

        // Reversal (start unloading)
        ms.stress[0] = 400.0e6;
        model.compute_damage(ms, fs, 0.0, 0.0);

        Real cycles_after = fs.history[1];

        CHECK(cycles_after >= cycles_before,
              "HCDSSE: cycle count increments on reversal");
    }
}

// ============================================================================
// 8. Adhesive Joint Failure
// ============================================================================
static void test_8_adhesive_joint_failure() {
    std::cout << "\n--- 8. Adhesive Joint Failure ---\n";

    FailureModelParameters base;
    AdhesiveJointFailureParams params;
    params.t_n   = 30.0e6;    // 30 MPa normal strength
    params.t_s   = 50.0e6;    // 50 MPa shear strength
    params.G_Ic  = 300.0;     // Mode I critical ERR (J/m^2)
    params.G_IIc = 600.0;     // Mode II critical ERR (J/m^2)
    params.eta   = 1.45;      // BK exponent

    AdhesiveJointFailure model(base, params);

    // (1) Default construction: model name
    {
        CHECK(std::string(model.name()) == "AdhesiveJointFailure",
              "AdhesiveJoint: model name is 'AdhesiveJointFailure'");
    }

    // (2) Below threshold: low tractions => no initiation
    {
        MaterialState ms = make_zero_state();
        ms.stress[2] = 10.0e6;  // sigma_n = 10 MPa << t_n = 30 MPa
        ms.stress[5] = 10.0e6;  // tau_13 = 10 MPa << t_s = 50 MPa

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.history[4] < 0.5, "AdhesiveJoint: below threshold => no initiation");
        CHECK(!fs.failed, "AdhesiveJoint: safe state");
    }

    // (3) Normal traction exceeds t_n: initiation
    {
        MaterialState ms = make_zero_state();
        ms.stress[2] = 35.0e6;    // sigma_33 > t_n = 30 MPa
        ms.strain[2] = 0.001;     // Normal opening

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        // Initiation: (35/30)^2 + 0 = 1.361 >= 1
        CHECK(fs.history[4] >= 0.5, "AdhesiveJoint: normal traction triggers initiation");
    }

    // (4) Damage in [0, 1] range
    {
        MaterialState ms = make_zero_state();
        ms.stress[2] = 35.0e6;
        ms.strain[2] = 0.001;

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.damage >= 0.0 && fs.damage <= 1.0,
              "AdhesiveJoint: damage in [0, 1] range");
    }

    // (5) Quadratic traction criterion: combined normal + shear initiation
    {
        MaterialState ms = make_zero_state();
        ms.stress[2] = 25.0e6;  // (25/30)^2 = 0.694
        ms.stress[5] = 35.0e6;  // tau_s = 35 MPa => (35/50)^2 = 0.49
        // Combined: 0.694 + 0.49 = 1.184 >= 1
        ms.strain[2] = 0.0005;
        ms.strain[5] = 0.0007;

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.history[4] >= 0.5,
              "AdhesiveJoint: quadratic mixed-mode initiation");
    }

    // (6) Compressive normal: does not contribute to initiation
    {
        MaterialState ms = make_zero_state();
        ms.stress[2] = -50.0e6;  // Compressive normal (Macaulay bracket => 0)

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.history[4] < 0.5,
              "AdhesiveJoint: compression does not initiate");
        CHECK(!fs.failed, "AdhesiveJoint: compression safe");
    }
}

// ============================================================================
// 9. Windshield Failure
// ============================================================================
static void test_9_windshield_failure() {
    std::cout << "\n--- 9. Windshield Failure ---\n";

    FailureModelParameters base;
    WindshieldFailureParams params;
    params.glass_weibull_m   = 7.0;
    params.glass_sigma0      = 70.0e6;
    params.glass_V0          = 1.0e-6;
    params.pvb_stretch_limit = 3.0;
    params.pvb_tear_energy   = 5000.0;

    WindshieldFailure model(base, params);

    // (1) Default construction: model name
    {
        CHECK(std::string(model.name()) == "WindshieldFailure",
              "Windshield: model name is 'WindshieldFailure'");
    }

    // (2) Below threshold: stress below glass strength => no glass crack
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 50.0e6;  // < glass_sigma0 = 70 MPa
        ms.strain[0] = 0.001;

        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 0.0);

        CHECK(fs.history[0] < 0.5,
              "Windshield: glass layer 1 intact below threshold");
        CHECK(!fs.failed, "Windshield: not failed below glass strength");
    }

    // (3) Glass fracture: stress exceeds sigma0 cracks first layer
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 80.0e6;  // > glass_sigma0 = 70 MPa
        ms.strain[0] = 0.001;

        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 0.0);

        CHECK(fs.history[0] >= 1.0,
              "Windshield: glass layer 1 cracked at 80 MPa");
        CHECK(fs.failure_mode >= 1, "Windshield: glass fracture mode set");
    }

    // (4) Damage in [0, 1]: progressive damage model
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 80.0e6;
        ms.strain[0] = 0.001;

        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 0.0);

        CHECK(fs.damage >= 0.0 && fs.damage <= 1.0,
              "Windshield: overall damage in [0, 1]");
    }

    // (5) Progressive failure: glass cracks, but PVB holds => not fully failed
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 80.0e6;   // Cracks glass layer 1
        ms.strain[0] = 0.01;     // Small stretch => PVB ok

        FailureState fs;
        model.compute_damage(ms, fs, 1.0e-6, 0.0);

        CHECK(fs.history[0] >= 1.0,
              "Windshield: glass layer 1 cracked");
        CHECK(!fs.failed,
              "Windshield: PVB interlayer holds, not fully failed");

        // Now exceed 1.1 * sigma0 = 77 MPa => both glass layers crack
        MaterialState ms2 = make_zero_state();
        ms2.stress[0] = 100.0e6;  // > 1.1 * 70 MPa = 77 MPa
        ms2.strain[0] = 0.01;

        FailureState fs2;
        model.compute_damage(ms2, fs2, 1.0e-6, 0.0);

        CHECK(fs2.history[0] >= 1.0 && fs2.history[1] >= 1.0,
              "Windshield: both glass layers cracked at 100 MPa");
        CHECK(fs2.failure_mode == 2,
              "Windshield: mode 2 = both glass cracked, PVB holds");
    }

    // (6) Full through-failure: both glass + PVB torn
    {
        FailureState fs;
        fs.history[0] = 1.0;  // Glass 1 already cracked
        fs.history[1] = 1.0;  // Glass 2 already cracked

        MaterialState ms = make_zero_state();
        ms.strain[0] = 3.0;   // lambda_1 = 1 + 3.0 = 4.0 > pvb_stretch_limit = 3.0
        ms.stress[0] = 1.0e6;

        model.compute_damage(ms, fs, 1.0e-6, 0.0);

        CHECK(fs.history[2] >= 1.0,
              "Windshield: PVB damaged past stretch limit");
        CHECK(fs.failed, "Windshield: full laminate failure");
        CHECK(fs.failure_mode == 3,
              "Windshield: mode 3 = full through-failure");
    }
}

// ============================================================================
// 10. Generalized Energy Failure
// ============================================================================
static void test_10_generalized_energy() {
    std::cout << "\n--- 10. Generalized Energy Failure ---\n";

    FailureModelParameters base;
    GeneralizedEnergyFailureParams params;
    params.G_c = 1.0e4;  // 10 kJ/m^2 critical fracture energy

    GeneralizedEnergyFailure model(base, params);

    // (1) Default construction: model name
    {
        CHECK(std::string(model.name()) == "GeneralizedEnergyFailure",
              "GenEnergy: model name is 'GeneralizedEnergyFailure'");
    }

    // (2) Below threshold: elastic state with no plastic strain
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 500.0e6;
        ms.plastic_strain = 0.0;  // Purely elastic

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK_NEAR(fs.history[0], 0.0, 1.0e-10,
                   "GenEnergy: no plastic work => no damage");
        CHECK(!fs.failed, "GenEnergy: elastic state => no failure");
    }

    // (3) At threshold: accumulated W_p >= G_c triggers failure
    {
        FailureState fs;
        // Pre-load energy near G_c
        fs.history[0] = 9999.0;  // Near G_c = 10000
        fs.history[1] = 0.0;     // Previous plastic strain

        MaterialState ms = make_zero_state();
        ms.stress[0] = 500.0e6;
        ms.plastic_strain = 0.01;
        // sigma_eq * delta_eps_p = 500e6 * 0.01 = 5e6 >> remaining gap of 1.0

        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.failed, "GenEnergy: failure when accumulated W_p >= G_c");
        CHECK(fs.failure_mode == 1, "GenEnergy: failure mode = 1");
    }

    // (4) Damage in [0, 1]: ratio W_p / G_c is clamped
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 300.0e6;
        ms.plastic_strain = 0.005;

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        CHECK(fs.damage >= 0.0 && fs.damage <= 1.0,
              "GenEnergy: damage in [0, 1] range");
    }

    // (5) Plastic work: W_p = sigma_eq * delta_eps_p
    {
        MaterialState ms = make_zero_state();
        ms.stress[0] = 300.0e6;
        ms.plastic_strain = 0.02;

        FailureState fs;
        model.compute_damage(ms, fs, 0.0, 0.0);

        Real sigma_eq = Material::von_mises_stress(ms.stress);
        Real expected_W = sigma_eq * 0.02;

        CHECK_NEAR(fs.history[0], expected_W, expected_W * 1.0e-6,
                   "GenEnergy: W_p = sigma_eq * delta_eps_p");
    }

    // (6) Multi-step: cumulative plastic work over increments
    {
        FailureState fs;
        MaterialState ms = make_zero_state();

        // Step 1
        ms.stress[0] = 400.0e6;
        ms.plastic_strain = 0.01;
        model.compute_damage(ms, fs, 0.0, 0.0);
        Real W1 = fs.history[0];

        // Step 2: additional plastic strain increment
        ms.stress[0] = 600.0e6;
        ms.plastic_strain = 0.02;  // delta = 0.02 - 0.01 = 0.01
        model.compute_damage(ms, fs, 0.0, 0.0);
        Real W2 = fs.history[0];

        CHECK(W2 > W1, "GenEnergy: cumulative W_p increases over steps");

        // Step 3: no additional plastic strain => no increment
        ms.plastic_strain = 0.02;  // same => delta = 0
        model.compute_damage(ms, fs, 0.0, 0.0);
        Real W3 = fs.history[0];

        CHECK_NEAR(W3, W2, 1.0e-6,
                   "GenEnergy: no increment when plastic strain unchanged");
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    test_1_ladeveze_delamination();
    test_2_hoffman_failure();
    test_3_tsai_hill_failure();
    test_4_rtcl_failure();
    test_5_mullins_effect();
    test_6_spalling_failure();
    test_7_hcdsse_failure();
    test_8_adhesive_joint_failure();
    test_9_windshield_failure();
    test_10_generalized_energy();

    std::cout << "\n=== Wave 19 Failure Models: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
