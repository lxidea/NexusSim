/**
 * @file sensor_wave44_test.cpp
 * @brief Wave 44: Sensor Aggregation + Expressions — 10 tests
 */

#include <nexussim/fem/sensor_wave44.hpp>

#include <cmath>
#include <iostream>
#include <string>

// ============================================================================
// Test harness
// ============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "  FAIL: " << #cond                                    \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";         \
            ++tests_failed;                                                      \
        } else {                                                                 \
            ++tests_passed;                                                      \
        }                                                                        \
    } while (false)

#define CHECK_NEAR(a, b, tol)                                                    \
    do {                                                                         \
        double _a = static_cast<double>(a);                                      \
        double _b = static_cast<double>(b);                                      \
        double _d = std::abs(_a - _b);                                           \
        if (_d > static_cast<double>(tol)) {                                     \
            std::cerr << "  FAIL: |" << #a << " - " << #b << "| = " << _d      \
                      << " > " << (tol)                                          \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";         \
            ++tests_failed;                                                      \
        } else {                                                                 \
            ++tests_passed;                                                      \
        }                                                                        \
    } while (false)

using namespace nxs::fem;
using nxs::Real;

// ============================================================================
// Helper: build a SensorManager with velocity gauges at known nodes
// ============================================================================

static nxs::fem::SensorManager make_velocity_manager(int id,
                                                      double x_vel,
                                                      double y_vel,
                                                      double z_vel) {
    nxs::fem::SensorManager mgr;

    SensorConfig cfg;
    cfg.type      = SensorType::VelocityGauge;
    cfg.id        = id;
    cfg.name      = "vel_" + std::to_string(id);
    cfg.node_id   = 0;       // first node
    cfg.direction = SensorDirection::X;
    mgr.add_sensor(cfg);

    // positions and forces unused; velocities provided
    double vel[3] = { x_vel, y_vel, z_vel };
    mgr.measure_all(0.0, 1e-4, 1,
                    nullptr, vel, nullptr, nullptr, 0, nullptr);
    return mgr;
}

// ============================================================================
// Test 1: Min aggregation
// ============================================================================

static void test_min_aggregation() {
    std::cout << "Test 1: Min aggregation\n";

    SensorAggregator agg;
    agg.configure(1.0, AggregationType::Min);

    agg.push(0.1, 5.0);
    agg.push(0.2, 2.0);
    agg.push(0.3, 8.0);
    agg.push(0.4, -1.0);
    agg.push(0.5, 3.0);

    CHECK_NEAR(agg.result(), -1.0, 1e-9);
    CHECK(agg.num_samples() == 5);
}

// ============================================================================
// Test 2: Max aggregation
// ============================================================================

static void test_max_aggregation() {
    std::cout << "Test 2: Max aggregation\n";

    SensorAggregator agg;
    agg.configure(1.0, AggregationType::Max);

    agg.push(0.1, 3.0);
    agg.push(0.2, 7.5);
    agg.push(0.3, 2.0);
    agg.push(0.4, 6.0);

    CHECK_NEAR(agg.result(), 7.5, 1e-9);
    CHECK(agg.num_samples() == 4);
}

// ============================================================================
// Test 3: RMS aggregation
// ============================================================================

static void test_rms_aggregation() {
    std::cout << "Test 3: RMS aggregation\n";

    SensorAggregator agg;
    agg.configure(1.0, AggregationType::RMS);

    // Values: 3, 4, 0  → sum of squares = 9+16+0 = 25, mean = 25/3, rms = sqrt(25/3)
    agg.push(0.1, 3.0);
    agg.push(0.2, 4.0);
    agg.push(0.3, 0.0);

    double expected_rms = std::sqrt(25.0 / 3.0);
    CHECK_NEAR(agg.result(), expected_rms, 1e-9);
    CHECK(agg.num_samples() == 3);
}

// ============================================================================
// Test 4: Mean aggregation
// ============================================================================

static void test_mean_aggregation() {
    std::cout << "Test 4: Mean aggregation\n";

    SensorAggregator agg;
    agg.configure(1.0, AggregationType::Mean);

    agg.push(0.0, 10.0);
    agg.push(0.1, 20.0);
    agg.push(0.2, 30.0);
    agg.push(0.3, 40.0);

    CHECK_NEAR(agg.result(), 25.0, 1e-9);
    CHECK(agg.num_samples() == 4);
    CHECK(agg.type() == AggregationType::Mean);
}

// ============================================================================
// Test 5: Envelope aggregation (max absolute value)
// ============================================================================

static void test_envelope_aggregation() {
    std::cout << "Test 5: Envelope aggregation\n";

    SensorAggregator agg;
    agg.configure(1.0, AggregationType::Envelope);

    agg.push(0.1,  3.0);
    agg.push(0.2, -9.0);   // largest absolute value
    agg.push(0.3,  4.0);
    agg.push(0.4, -2.0);

    CHECK_NEAR(agg.result(), 9.0, 1e-9);
    CHECK(agg.num_samples() == 4);
}

// ============================================================================
// Test 6: Time window trimming
// ============================================================================

static void test_time_window_trimming() {
    std::cout << "Test 6: Time window trimming\n";

    SensorAggregator agg;
    agg.configure(0.2, AggregationType::Mean);  // 200 ms window

    // Push samples spread over time
    agg.push(0.0, 100.0);   // will be trimmed once t > 0.2
    agg.push(0.1, 200.0);   // will be trimmed once t > 0.3
    agg.push(0.2, 300.0);

    // Before any trimming: mean = (100+200+300)/3 = 200
    CHECK_NEAR(agg.result(), 200.0, 1e-9);
    CHECK(agg.num_samples() == 3);

    // Push at t=0.4: samples with time < 0.2 are trimmed (t=0.0 trimmed)
    agg.push(0.4, 400.0);
    // Remaining: t=0.2 (300), t=0.4 (400) — t=0.1 is also trimmed (0.1 < 0.4-0.2 = 0.2)
    CHECK(agg.num_samples() == 2);
    CHECK_NEAR(agg.result(), 350.0, 1e-9);   // (300+400)/2

    // Push at t=1.0: only t=1.0 (500) survives
    agg.push(1.0, 500.0);
    CHECK(agg.num_samples() == 1);
    CHECK_NEAR(agg.result(), 500.0, 1e-9);

    // Window size accessor
    CHECK_NEAR(agg.window_size(), 0.2, 1e-9);
}

// ============================================================================
// Test 7: Expression parsing
// ============================================================================

static void test_expression_parsing() {
    std::cout << "Test 7: Expression parsing\n";

    // "S1 C100.0 >"  — postfix for "sensor_1 > 100.0"
    auto expr = SensorExpression::parse("S1 C100.0 >");

    CHECK(expr.num_tokens() == 3);
    CHECK(expr.is_valid());

    const auto& toks = expr.tokens();
    CHECK(toks[0].type      == ExprTokenType::Sensor);
    CHECK(toks[0].sensor_id == 1);
    CHECK(toks[1].type      == ExprTokenType::Constant);
    CHECK_NEAR(toks[1].constant_value, 100.0, 1e-9);
    CHECK(toks[2].type      == ExprTokenType::GreaterThan);

    // Validate is_valid() on an invalid expression
    auto bad = SensorExpression::parse("S1 >");   // only one operand for binary op
    CHECK(!bad.is_valid());

    // Single sensor token is valid
    auto single = SensorExpression::parse("S5");
    CHECK(single.is_valid());
    CHECK(single.num_tokens() == 1);
}

// ============================================================================
// Test 8: Expression evaluation
// ============================================================================

static void test_expression_evaluation() {
    std::cout << "Test 8: Expression evaluation\n";

    // Sensor 1: X-velocity = 150.0  → result of "S1 C100.0 >" should be 1.0
    auto mgr1 = make_velocity_manager(1, 150.0, 0.0, 0.0);

    auto expr_gt = SensorExpression::parse("S1 C100.0 >");
    CHECK_NEAR(expr_gt.evaluate(mgr1), 1.0, 1e-9);

    // Sensor 1: X-velocity = 50.0  → "S1 C100.0 >" = 0.0
    auto mgr2 = make_velocity_manager(1, 50.0, 0.0, 0.0);
    CHECK_NEAR(expr_gt.evaluate(mgr2), 0.0, 1e-9);

    // Arithmetic: "S1 C10.0 *"  with velocity 7.0 → 70.0
    auto mgr3 = make_velocity_manager(1, 7.0, 0.0, 0.0);
    auto expr_mul = SensorExpression::parse("S1 C10.0 *");
    CHECK_NEAR(expr_mul.evaluate(mgr3), 70.0, 1e-9);

    // Unary negate: "S1 ~" with velocity 5.0 → -5.0
    auto mgr4 = make_velocity_manager(1, 5.0, 0.0, 0.0);
    auto expr_neg = SensorExpression::parse("S1 ~");
    CHECK_NEAR(expr_neg.evaluate(mgr4), -5.0, 1e-9);
}

// ============================================================================
// Test 9: Sensor-driven TH trigger — callback fires when threshold crossed
// ============================================================================

static void test_expression_trigger() {
    std::cout << "Test 9: Sensor expression trigger\n";

    // Expression: "S1 C100.0 >"  → 1.0 when sensor 1 > 100.0
    auto expr = SensorExpression::parse("S1 C100.0 >");
    SensorExpressionTrigger trigger(std::move(expr), 0.5, true);

    int fired_count = 0;
    trigger.set_callback([&fired_count](){ ++fired_count; });

    // Sensor value below threshold — should NOT fire
    auto mgr_low = make_velocity_manager(1, 50.0, 0.0, 0.0);
    trigger.evaluate_and_trigger(mgr_low);
    CHECK(fired_count == 0);

    // check() returns false
    CHECK(!trigger.check(mgr_low));

    // Sensor value above threshold — should fire
    auto mgr_high = make_velocity_manager(1, 200.0, 0.0, 0.0);
    trigger.evaluate_and_trigger(mgr_high);
    CHECK(fired_count == 1);

    // check() returns true
    CHECK(trigger.check(mgr_high));

    // Fire again to confirm callback re-fires each call
    trigger.evaluate_and_trigger(mgr_high);
    CHECK(fired_count == 2);

    // Accessors
    CHECK_NEAR(trigger.threshold(), 0.5, 1e-9);
    CHECK(trigger.trigger_above());
}

// ============================================================================
// Test 10: Multi-sensor expression "S1 S2 + C200.0 >"
// ============================================================================

static void test_multi_sensor_expression() {
    std::cout << "Test 10: Multi-sensor expression\n";

    // Build a manager with two sensors (both velocity gauges, X direction)
    nxs::fem::SensorManager mgr;

    SensorConfig cfg1;
    cfg1.type      = SensorType::VelocityGauge;
    cfg1.id        = 1;
    cfg1.name      = "vel_1";
    cfg1.node_id   = 0;
    cfg1.direction = SensorDirection::X;
    mgr.add_sensor(cfg1);

    SensorConfig cfg2;
    cfg2.type      = SensorType::VelocityGauge;
    cfg2.id        = 2;
    cfg2.name      = "vel_2";
    cfg2.node_id   = 1;
    cfg2.direction = SensorDirection::X;
    mgr.add_sensor(cfg2);

    // Two nodes: vel_node0 = 120, vel_node1 = 100
    double vel[6] = { 120.0, 0.0, 0.0,   // node 0
                      100.0, 0.0, 0.0 };  // node 1
    mgr.measure_all(0.0, 1e-4, 2,
                    nullptr, vel, nullptr, nullptr, 0, nullptr);

    // Expression: "S1 S2 + C200.0 >"
    // S1=120, S2=100 → 220 > 200 → 1.0
    auto expr = SensorExpression::parse("S1 S2 + C200.0 >");
    CHECK(expr.num_tokens() == 5);
    CHECK(expr.is_valid());
    CHECK_NEAR(expr.evaluate(mgr), 1.0, 1e-9);

    // Change velocities so sum < 200: node0=80, node1=70 → 150 > 200 → 0.0
    double vel2[6] = { 80.0, 0.0, 0.0,
                       70.0, 0.0, 0.0 };
    mgr.measure_all(0.1, 1e-4, 2,
                    nullptr, vel2, nullptr, nullptr, 0, nullptr);
    CHECK_NEAR(expr.evaluate(mgr), 0.0, 1e-9);

    // Also verify MultiSensorAggregator collects both sensors
    MultiSensorAggregator msa;
    msa.configure_all({1, 2}, 1.0, AggregationType::Max);

    // First measurement: 80 and 70 (already stored in mgr from above)
    msa.push_all(mgr, 0.1);

    // Second measurement: 90 and 110
    double vel3[6] = { 90.0, 0.0, 0.0,
                       110.0, 0.0, 0.0 };
    mgr.measure_all(0.2, 1e-4, 2,
                    nullptr, vel3, nullptr, nullptr, 0, nullptr);
    msa.push_all(mgr, 0.2);

    // Max over window: S1 max = max(80,90) = 90, S2 max = max(70,110) = 110
    CHECK_NEAR(msa.get_result(1), 90.0, 1e-9);
    CHECK_NEAR(msa.get_result(2), 110.0, 1e-9);
    CHECK(msa.num_aggregators() == 2);

    // Unknown sensor returns 0
    CHECK_NEAR(msa.get_result(999), 0.0, 1e-9);
}

// ============================================================================
// main
// ============================================================================

int main() {
    test_min_aggregation();
    test_max_aggregation();
    test_rms_aggregation();
    test_mean_aggregation();
    test_envelope_aggregation();
    test_time_window_trimming();
    test_expression_parsing();
    test_expression_evaluation();
    test_expression_trigger();
    test_multi_sensor_expression();

    std::cout << "\n=== Wave 44 Sensor Expression Results ===\n";
    std::cout << "  Passed: " << tests_passed << "\n";
    std::cout << "  Failed: " << tests_failed << "\n";

    return tests_failed > 0 ? 1 : 0;
}
