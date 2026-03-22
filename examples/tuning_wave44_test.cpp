// Wave 44 Tuning Constants Test
// Tests: HourglassCoefficients defaults, effective_viscosity/stiffness,
//        DrillingPenaltyCalibration (well/poorly conditioned),
//        FrictionModelVariants (Coulomb, StaticDynamic), TuningParameterSet presets.

#include <nexussim/fem/tuning_wave44.hpp>

#include <cmath>
#include <iostream>

// ============================================================================
// Test harness
// ============================================================================

static int tests_failed = 0;
static int tests_passed = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cerr << "  FAIL: " << #cond << "  (" << __FILE__ << ":"    \
                      << __LINE__ << ")\n";                                  \
            ++tests_failed;                                                  \
        } else {                                                             \
            ++tests_passed;                                                  \
        }                                                                    \
    } while (false)

#define CHECK_NEAR(a, b, tol)                                                \
    do {                                                                     \
        double _a = static_cast<double>(a);                                  \
        double _b = static_cast<double>(b);                                  \
        double _t = static_cast<double>(tol);                               \
        if (std::abs(_a - _b) > _t) {                                       \
            std::cerr << "  FAIL: |" << #a << " - " << #b << "| = "        \
                      << std::abs(_a - _b) << " > " << _t                   \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";     \
            ++tests_failed;                                                  \
        } else {                                                             \
            ++tests_passed;                                                  \
        }                                                                    \
    } while (false)

// ============================================================================
// Test 1: IHQ1 defaults
// ============================================================================

static void test_ihq1_defaults() {
    std::cout << "Test 1: IHQ1 defaults\n";
    using namespace nxs::fem;

    HourglassCoefficients c = HourglassCoefficients::defaults(HourglassMode::IHQ1);

    CHECK_NEAR(c.qh, 0.1, 1e-12);
    CHECK_NEAR(c.qm, 0.0, 1e-12);
    CHECK_NEAR(c.bulk_viscosity_q1, 0.06, 1e-12);
    CHECK_NEAR(c.bulk_viscosity_q2, 1.2,  1e-12);
}

// ============================================================================
// Test 2: IHQ4 defaults
// ============================================================================

static void test_ihq4_defaults() {
    std::cout << "Test 2: IHQ4 defaults\n";
    using namespace nxs::fem;

    HourglassCoefficients c = HourglassCoefficients::defaults(HourglassMode::IHQ4);

    CHECK_NEAR(c.qh, 0.0,  1e-12);
    CHECK_NEAR(c.qm, 0.05, 1e-12);
    CHECK_NEAR(c.bulk_viscosity_q1, 0.06, 1e-12);
    CHECK_NEAR(c.bulk_viscosity_q2, 1.2,  1e-12);
}

// ============================================================================
// Test 3: IHQ8 defaults — both hourglass coefficients are zero
// ============================================================================

static void test_ihq8_defaults() {
    std::cout << "Test 3: IHQ8 defaults (no hourglass)\n";
    using namespace nxs::fem;

    HourglassCoefficients c = HourglassCoefficients::defaults(HourglassMode::IHQ8);

    CHECK_NEAR(c.qh, 0.0, 1e-12);
    CHECK_NEAR(c.qm, 0.0, 1e-12);
    // Bulk viscosity still present even without hourglass modes
    CHECK_NEAR(c.bulk_viscosity_q1, 0.06, 1e-12);
    CHECK_NEAR(c.bulk_viscosity_q2, 1.2,  1e-12);
}

// ============================================================================
// Test 4: Effective viscosity and stiffness computation
// ============================================================================

static void test_effective_viscosity_stiffness() {
    std::cout << "Test 4: Effective viscosity / stiffness computation\n";
    using namespace nxs::fem;

    // IHQ1: qh=0.1, qm=0.0
    HourglassCoefficients c1 = HourglassCoefficients::defaults(HourglassMode::IHQ1);

    // rho=7800, c=5000, le=0.01
    // expected: 0.1 * 7800 * 5000 * 0.01 = 39000.0
    nxs::Real visc = c1.effective_viscosity(7800.0, 5000.0, 0.01);
    CHECK_NEAR(visc, 39000.0, 1e-9);

    // IHQ2: qh=0.0, qm=0.1 => effective_stiffness = 0.1 * E / le
    HourglassCoefficients c2 = HourglassCoefficients::defaults(HourglassMode::IHQ2);
    // E=210e9, le=0.01 => 0.1 * 210e9 / 0.01 = 2.1e12
    nxs::Real stiff = c2.effective_stiffness(210.0e9, 0.01);
    CHECK_NEAR(stiff, 2.1e12, 1.0); // tolerance 1 Pa (relative to 2.1e12)

    // IHQ4: qm=0.05 => 0.05 * 210e9 / 0.01 = 1.05e12
    HourglassCoefficients c4 = HourglassCoefficients::defaults(HourglassMode::IHQ4);
    nxs::Real stiff4 = c4.effective_stiffness(210.0e9, 0.01);
    CHECK_NEAR(stiff4, 1.05e12, 1.0);

    // IHQ8: both zero => viscosity and stiffness should both be 0
    HourglassCoefficients c8 = HourglassCoefficients::defaults(HourglassMode::IHQ8);
    CHECK_NEAR(c8.effective_viscosity(7800.0, 5000.0, 0.01), 0.0, 1e-30);
    CHECK_NEAR(c8.effective_stiffness(210.0e9, 0.01), 0.0, 1e-30);
}

// ============================================================================
// Test 5: Drilling calibration — well-conditioned mesh
// ============================================================================

static void test_drilling_well_conditioned() {
    std::cout << "Test 5: Drilling calibration — well-conditioned mesh\n";
    using namespace nxs::fem;

    DrillingPenaltyCalibration calib;

    // Diagonal with nearly uniform entries: condition number ~2
    const std::size_t n = 6;
    nxs::Real diag[n] = {1.0e6, 1.1e6, 0.9e6, 1.05e6, 0.95e6, 1.02e6};

    nxs::Real alpha = calib.calibrate_from_diagonal(diag, n, 1.0e8);

    // Well-conditioned => alpha should stay at base value 1e-3
    CHECK_NEAR(alpha, 1.0e-3, 1.0e-6);
    CHECK_NEAR(calib.alpha(), 1.0e-3, 1.0e-6);
}

// ============================================================================
// Test 6: Drilling calibration — poorly-conditioned mesh
// ============================================================================

static void test_drilling_poorly_conditioned() {
    std::cout << "Test 6: Drilling calibration — poorly-conditioned mesh\n";
    using namespace nxs::fem;

    DrillingPenaltyCalibration calib;

    // Large condition number: max/min = 1e10 >> 1e8 threshold
    const std::size_t n = 4;
    nxs::Real diag[n] = {1.0e10, 1.0e10, 1.0e10, 1.0};  // cond = 1e10

    nxs::Real alpha = calib.calibrate_from_diagonal(diag, n, 1.0e8);

    // Poorly conditioned: alpha = min_diag / max_diag * base_alpha
    //   = 1.0 / 1e10 * 1e-3 = 1e-13
    nxs::Real expected = (1.0 / 1.0e10) * 1.0e-3;
    CHECK_NEAR(alpha, expected, expected * 1.0e-9);
    CHECK(alpha < 1.0e-3);  // must be reduced relative to base
}

// ============================================================================
// Test 7: Friction Coulomb — constant mu regardless of velocity
// ============================================================================

static void test_friction_coulomb() {
    std::cout << "Test 7: Friction Coulomb\n";
    using namespace nxs::fem;

    FrictionModelVariants fr(FrictionType::Coulomb, 0.25);

    CHECK_NEAR(fr.mu_static(), 0.25, 1e-12);
    CHECK(fr.type() == FrictionType::Coulomb);

    // mu must be constant regardless of velocity, pressure, or temperature
    CHECK_NEAR(fr.compute_friction(0.0,  1.0e6), 0.25, 1e-12);
    CHECK_NEAR(fr.compute_friction(0.1,  1.0e6), 0.25, 1e-12);
    CHECK_NEAR(fr.compute_friction(10.0, 5.0e6), 0.25, 1e-12);
    CHECK_NEAR(fr.compute_friction(1e3,  1.0,    500.0), 0.25, 1e-12);
}

// ============================================================================
// Test 8: Friction StaticDynamic — transitions with velocity
// ============================================================================

static void test_friction_static_dynamic() {
    std::cout << "Test 8: Friction StaticDynamic\n";
    using namespace nxs::fem;

    FrictionModelVariants fr(FrictionType::StaticDynamic, 0.40);
    fr.set_dynamic(0.20, 2.0);  // decay_exponent = 2.0

    CHECK_NEAR(fr.mu_static(),  0.40, 1e-12);
    CHECK_NEAR(fr.mu_dynamic(), 0.20, 1e-12);
    CHECK(fr.type() == FrictionType::StaticDynamic);

    // At v=0: mu = mu_dyn + (mu_s - mu_dyn) * exp(0) = mu_s = 0.40
    nxs::Real mu_v0 = fr.compute_friction(0.0, 1.0e6);
    CHECK_NEAR(mu_v0, 0.40, 1e-12);

    // At high velocity: exp(-2*large) -> 0, so mu -> mu_dynamic = 0.20
    nxs::Real mu_high = fr.compute_friction(50.0, 1.0e6);
    CHECK_NEAR(mu_high, 0.20, 1e-6);

    // At intermediate v=1: mu = 0.20 + (0.40-0.20)*exp(-2*1) = 0.20 + 0.20*exp(-2)
    nxs::Real expected_v1 = 0.20 + (0.40 - 0.20) * std::exp(-2.0 * 1.0);
    nxs::Real mu_v1 = fr.compute_friction(1.0, 1.0e6);
    CHECK_NEAR(mu_v1, expected_v1, 1e-12);

    // mu at intermediate v must lie strictly between dynamic and static
    CHECK(mu_v1 > fr.mu_dynamic());
    CHECK(mu_v1 < fr.mu_static());
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Wave 44 Tuning Constants Tests ===\n\n";

    test_ihq1_defaults();
    test_ihq4_defaults();
    test_ihq8_defaults();
    test_effective_viscosity_stiffness();
    test_drilling_well_conditioned();
    test_drilling_poorly_conditioned();
    test_friction_coulomb();
    test_friction_static_dynamic();

    std::cout << "\n=== Wave 44 Tuning Results ===\n";
    std::cout << "  Passed: " << tests_passed << "\n";
    std::cout << "  Failed: " << tests_failed << "\n";

    return tests_failed > 0 ? 1 : 0;
}
