/**
 * @file sensor_ale_test.cpp
 * @brief Wave 8 test: Sensors, Controls, and ALE mesh management
 *
 * Tests:
 *  1. Accelerometer sensor basic
 *  2. Velocity gauge sensor
 *  3. Distance sensor
 *  4. Strain gauge sensor
 *  5. CFC filter (Butterworth low-pass)
 *  6. Sensor sampling interval
 *  7. Sensor threshold detection
 *  8. Sensor manager
 *  9. Control rule: terminate on threshold
 * 10. Control rule: activate/deactivate loads
 * 11. Control one-shot vs repeating
 * 12. Control custom callback
 * 13. Mesh adjacency construction
 * 14. Laplacian mesh smoothing
 * 15. Weighted mesh smoothing
 * 16. Boundary node preservation
 * 17. Donor cell advection
 * 18. Van Leer advection
 * 19. Full ALE step (smooth + advect)
 * 20. Element quality metric
 * 21. Force sensor
 * 22. Sensor direction components
 */

#include <nexussim/fem/sensor.hpp>
#include <nexussim/fem/controls.hpp>
#include <nexussim/physics/ale_solver.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace nxs;
using Real = nxs::Real;
using Index = nxs::Index;

static int tests_passed = 0;
static int tests_failed = 0;

static bool near(Real a, Real b, Real tol = 1.0e-6) {
    return std::fabs(a - b) < tol;
}

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; std::cout << "FAIL: " << msg << "\n"; } \
} while(0)

// ============================================================================
// Test 1: Accelerometer sensor basic
// ============================================================================
void test_accelerometer_basic() {
    std::cout << "Test 1: Accelerometer sensor basic\n";

    fem::SensorConfig cfg;
    cfg.type = fem::SensorType::Accelerometer;
    cfg.id = 1;
    cfg.name = "Accel_1";
    cfg.node_id = 0;
    cfg.direction = fem::SensorDirection::X;

    fem::Sensor sensor(cfg);
    CHECK(sensor.id() == 1, "sensor ID");
    CHECK(sensor.name() == "Accel_1", "sensor name");
    CHECK(sensor.num_readings() == 0, "no readings initially");
    CHECK(near(sensor.current_value(), 0.0), "current value zero initially");

    // Provide accelerations for 4 nodes
    std::size_t num_nodes = 4;
    std::vector<Real> accel = {
        10.0, 20.0, 30.0,   // node 0: ax=10, ay=20, az=30
         1.0,  2.0,  3.0,   // node 1
         0.0,  0.0,  0.0,   // node 2
         5.0,  5.0,  5.0    // node 3
    };

    sensor.measure(0.0, 1.0e-5, num_nodes,
                   nullptr, nullptr, accel.data(), nullptr);

    CHECK(sensor.num_readings() == 1, "one reading after measure");
    CHECK(near(sensor.current_value(), 10.0), "accel X = 10");

    // Take another reading at different time
    accel[0] = 15.0;
    sensor.measure(1.0e-5, 1.0e-5, num_nodes,
                   nullptr, nullptr, accel.data(), nullptr);
    CHECK(sensor.num_readings() == 2, "two readings");
    // Without CFC filter, raw = filtered = 15.0
    CHECK(near(sensor.current_value(), 15.0), "accel X = 15 at t2");
}

// ============================================================================
// Test 2: Velocity gauge sensor
// ============================================================================
void test_velocity_gauge() {
    std::cout << "Test 2: Velocity gauge sensor\n";

    fem::SensorConfig cfg;
    cfg.type = fem::SensorType::VelocityGauge;
    cfg.id = 2;
    cfg.name = "Vel_1";
    cfg.node_id = 1;
    cfg.direction = fem::SensorDirection::Magnitude;

    fem::Sensor sensor(cfg);

    std::size_t num_nodes = 3;
    std::vector<Real> vel = {
        0.0, 0.0, 0.0,     // node 0
        3.0, 4.0, 0.0,     // node 1: |v| = 5
        1.0, 1.0, 1.0      // node 2
    };

    sensor.measure(0.0, 1.0e-5, num_nodes,
                   nullptr, vel.data(), nullptr, nullptr);

    CHECK(sensor.num_readings() == 1, "one reading");
    CHECK(near(sensor.current_value(), 5.0), "velocity magnitude = 5");
}

// ============================================================================
// Test 3: Distance sensor
// ============================================================================
void test_distance_sensor() {
    std::cout << "Test 3: Distance sensor\n";

    fem::SensorConfig cfg;
    cfg.type = fem::SensorType::DistanceSensor;
    cfg.id = 3;
    cfg.name = "Dist_1";
    cfg.node_id = 0;
    cfg.node_id2 = 2;
    cfg.direction = fem::SensorDirection::Magnitude;

    fem::Sensor sensor(cfg);

    std::size_t num_nodes = 3;
    std::vector<Real> pos = {
        0.0, 0.0, 0.0,     // node 0
        5.0, 0.0, 0.0,     // node 1
        3.0, 4.0, 0.0      // node 2: distance to node 0 = 5
    };

    sensor.measure(0.0, 1.0e-5, num_nodes,
                   pos.data(), nullptr, nullptr, nullptr);

    CHECK(sensor.num_readings() == 1, "one reading");
    CHECK(near(sensor.current_value(), 5.0), "distance 0-2 = 5");

    // Move node 2 closer
    pos[6] = 1.0; pos[7] = 0.0; pos[8] = 0.0;
    sensor.measure(1.0e-5, 1.0e-5, num_nodes,
                   pos.data(), nullptr, nullptr, nullptr);
    CHECK(near(sensor.current_value(), 1.0), "distance reduced to 1");
}

// ============================================================================
// Test 4: Strain gauge sensor
// ============================================================================
void test_strain_gauge() {
    std::cout << "Test 4: Strain gauge sensor\n";

    fem::SensorConfig cfg;
    cfg.type = fem::SensorType::StrainGauge;
    cfg.id = 4;
    cfg.name = "Strain_1";
    cfg.element_id = 0;
    cfg.direction = fem::SensorDirection::X;  // eps_xx

    fem::Sensor sensor(cfg);

    std::size_t num_elems = 2;
    // 6 components per element: eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz
    std::vector<Real> strains = {
        0.01, 0.005, -0.002, 0.001, 0.0, 0.0,   // element 0
        0.02, 0.01,   0.003, 0.002, 0.0, 0.0     // element 1
    };

    sensor.measure(0.0, 1.0e-5, 0,
                   nullptr, nullptr, nullptr, nullptr,
                   num_elems, strains.data());

    CHECK(sensor.num_readings() == 1, "one reading");
    CHECK(near(sensor.current_value(), 0.01), "strain XX = 0.01");

    // Test effective strain (von Mises equivalent) on element 1
    fem::SensorConfig cfg2;
    cfg2.type = fem::SensorType::StrainGauge;
    cfg2.id = 5;
    cfg2.element_id = 1;
    cfg2.direction = fem::SensorDirection::Magnitude;

    fem::Sensor sensor2(cfg2);
    sensor2.measure(0.0, 1.0e-5, 0,
                    nullptr, nullptr, nullptr, nullptr,
                    num_elems, strains.data());

    CHECK(sensor2.num_readings() == 1, "one reading for effective strain");
    CHECK(sensor2.current_value() > 0.0, "effective strain > 0");
}

// ============================================================================
// Test 5: CFC filter (Butterworth low-pass)
// ============================================================================
void test_cfc_filter() {
    std::cout << "Test 5: CFC filter\n";

    fem::CFCFilter filter;
    CHECK(!filter.enabled, "filter disabled by default");

    // Setup CFC60 filter
    Real dt = 1.0e-4;  // 10 kHz sampling
    filter.setup(60, dt);
    CHECK(filter.enabled, "filter enabled after setup");
    CHECK(near(filter.cutoff_freq, 60.0), "cutoff = 60 Hz");

    // Feed a step function - filter should smooth it
    Real prev = 0.0;
    bool has_intermediate = false;
    for (int i = 0; i < 100; ++i) {
        Real input = 1.0;  // Step function
        Real output = filter.apply(input);
        if (i > 0 && output > 0.01 && output < 0.99) {
            has_intermediate = true;
        }
        prev = output;
    }
    CHECK(has_intermediate, "CFC filter produces transient (not instant step)");
    // After many samples, should approach 1.0
    CHECK(prev > 0.5, "filter output approaches steady state");

    // Reset and verify
    filter.reset();
    Real after_reset = filter.apply(0.0);
    CHECK(near(after_reset, 0.0), "filter reset clears state");
}

// ============================================================================
// Test 6: Sensor sampling interval
// ============================================================================
void test_sampling_interval() {
    std::cout << "Test 6: Sensor sampling interval\n";

    fem::SensorConfig cfg;
    cfg.type = fem::SensorType::Accelerometer;
    cfg.id = 10;
    cfg.node_id = 0;
    cfg.direction = fem::SensorDirection::X;
    cfg.sample_interval = 1.0e-3;  // 1 ms interval

    fem::Sensor sensor(cfg);

    std::size_t num_nodes = 2;
    std::vector<Real> accel = {5.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Measure at times closer than interval - should skip
    sensor.measure(0.0, 1.0e-5, num_nodes, nullptr, nullptr, accel.data(), nullptr);
    sensor.measure(0.5e-3, 1.0e-5, num_nodes, nullptr, nullptr, accel.data(), nullptr);
    CHECK(sensor.num_readings() == 1, "skipped sub-interval reading");

    // Measure at interval boundary
    sensor.measure(1.0e-3, 1.0e-5, num_nodes, nullptr, nullptr, accel.data(), nullptr);
    CHECK(sensor.num_readings() == 2, "recorded at interval boundary");

    sensor.measure(1.5e-3, 1.0e-5, num_nodes, nullptr, nullptr, accel.data(), nullptr);
    CHECK(sensor.num_readings() == 2, "skipped sub-interval again");

    sensor.measure(2.0e-3, 1.0e-5, num_nodes, nullptr, nullptr, accel.data(), nullptr);
    CHECK(sensor.num_readings() == 3, "recorded at 2ms");
}

// ============================================================================
// Test 7: Sensor threshold detection
// ============================================================================
void test_threshold_detection() {
    std::cout << "Test 7: Sensor threshold detection\n";

    // Threshold above
    fem::SensorConfig cfg;
    cfg.type = fem::SensorType::Accelerometer;
    cfg.id = 20;
    cfg.node_id = 0;
    cfg.direction = fem::SensorDirection::X;
    cfg.threshold_value = 100.0;
    cfg.threshold_above = true;

    fem::Sensor sensor(cfg);

    std::size_t num_nodes = 2;
    std::vector<Real> accel = {50.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    sensor.measure(0.0, 1.0e-5, num_nodes, nullptr, nullptr, accel.data(), nullptr);
    CHECK(!sensor.threshold_triggered(), "below threshold");

    accel[0] = 150.0;
    sensor.measure(1.0e-5, 1.0e-5, num_nodes, nullptr, nullptr, accel.data(), nullptr);
    CHECK(sensor.threshold_triggered(), "above threshold triggers");

    // Clear and test threshold below
    sensor.clear_readings();
    CHECK(!sensor.threshold_triggered(), "cleared threshold");

    // Below-threshold sensor
    fem::SensorConfig cfg2;
    cfg2.type = fem::SensorType::VelocityGauge;
    cfg2.id = 21;
    cfg2.node_id = 0;
    cfg2.direction = fem::SensorDirection::X;
    cfg2.threshold_value = 5.0;
    cfg2.threshold_above = false;

    fem::Sensor sensor2(cfg2);
    std::vector<Real> vel = {10.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    sensor2.measure(0.0, 1.0e-5, num_nodes, nullptr, vel.data(), nullptr, nullptr);
    CHECK(!sensor2.threshold_triggered(), "above threshold (below mode) - no trigger");

    vel[0] = 3.0;
    sensor2.measure(1.0e-5, 1.0e-5, num_nodes, nullptr, vel.data(), nullptr, nullptr);
    CHECK(sensor2.threshold_triggered(), "below threshold triggers in below mode");
}

// ============================================================================
// Test 8: Sensor manager
// ============================================================================
void test_sensor_manager() {
    std::cout << "Test 8: Sensor manager\n";

    fem::SensorManager mgr;
    CHECK(mgr.num_sensors() == 0, "empty initially");

    // Add accelerometer
    fem::SensorConfig cfg1;
    cfg1.type = fem::SensorType::Accelerometer;
    cfg1.id = 100;
    cfg1.node_id = 0;
    cfg1.direction = fem::SensorDirection::X;
    cfg1.threshold_value = 50.0;
    cfg1.threshold_above = true;
    mgr.add_sensor(cfg1);

    // Add velocity gauge
    fem::SensorConfig cfg2;
    cfg2.type = fem::SensorType::VelocityGauge;
    cfg2.id = 101;
    cfg2.node_id = 1;
    cfg2.direction = fem::SensorDirection::Magnitude;
    mgr.add_sensor(cfg2);

    CHECK(mgr.num_sensors() == 2, "two sensors");

    // Find by ID
    CHECK(mgr.find(100) != nullptr, "find sensor 100");
    CHECK(mgr.find(101) != nullptr, "find sensor 101");
    CHECK(mgr.find(999) == nullptr, "sensor 999 not found");

    // Measure all
    std::size_t num_nodes = 3;
    std::vector<Real> vel = {0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<Real> accel = {60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    mgr.measure_all(0.0, 1.0e-5, num_nodes, nullptr, vel.data(), accel.data());

    CHECK(mgr.find(100)->num_readings() == 1, "sensor 100 has reading");
    CHECK(mgr.find(101)->num_readings() == 1, "sensor 101 has reading");
    CHECK(near(mgr.find(100)->current_value(), 60.0), "accel = 60");
    CHECK(near(mgr.find(101)->current_value(), 5.0), "vel mag = 5");

    // Check threshold
    CHECK(mgr.any_threshold_triggered(), "one sensor triggered");
    auto triggered = mgr.triggered_sensor_ids();
    CHECK(triggered.size() == 1, "one triggered sensor");
    CHECK(triggered[0] == 100, "sensor 100 triggered");
}

// ============================================================================
// Test 9: Control rule: terminate on threshold
// ============================================================================
void test_control_terminate() {
    std::cout << "Test 9: Control terminate\n";

    // Setup sensor that will trigger
    fem::SensorManager sensors;
    fem::SensorConfig scfg;
    scfg.type = fem::SensorType::Accelerometer;
    scfg.id = 1;
    scfg.node_id = 0;
    scfg.direction = fem::SensorDirection::X;
    scfg.threshold_value = 100.0;
    scfg.threshold_above = true;
    sensors.add_sensor(scfg);

    // Setup control rule
    fem::ControlManager controls;
    auto& rule = controls.add_rule(1);
    rule.name = "Terminate on high accel";
    rule.sensor_id = 1;
    rule.trigger_on_exceed = true;
    rule.action = fem::ControlActionType::TerminateSimulation;
    rule.one_shot = true;

    CHECK(controls.num_rules() == 1, "one rule");
    CHECK(!controls.should_terminate(), "not terminated initially");

    // Sensor below threshold
    std::size_t nn = 2;
    std::vector<Real> accel = {50.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    sensors.measure_all(0.0, 1.0e-5, nn, nullptr, nullptr, accel.data());
    auto actions = controls.evaluate(sensors);
    CHECK(actions.empty(), "no actions below threshold");
    CHECK(!controls.should_terminate(), "no terminate below threshold");

    // Sensor exceeds threshold
    accel[0] = 200.0;
    sensors.measure_all(1.0e-5, 1.0e-5, nn, nullptr, nullptr, accel.data());
    actions = controls.evaluate(sensors);
    CHECK(actions.size() == 1, "one action triggered");
    CHECK(actions[0].action == fem::ControlActionType::TerminateSimulation, "action is terminate");
    CHECK(controls.should_terminate(), "simulation should terminate");
}

// ============================================================================
// Test 10: Control rule: activate/deactivate loads
// ============================================================================
void test_control_loads() {
    std::cout << "Test 10: Control activate/deactivate loads\n";

    fem::SensorManager sensors;
    fem::SensorConfig scfg;
    scfg.type = fem::SensorType::Accelerometer;
    scfg.id = 1;
    scfg.node_id = 0;
    scfg.direction = fem::SensorDirection::X;
    scfg.threshold_value = 50.0;
    scfg.threshold_above = true;
    sensors.add_sensor(scfg);

    fem::ControlManager controls;

    // Rule 1: activate load 10
    auto& r1 = controls.add_rule(1);
    r1.sensor_id = 1;
    r1.trigger_on_exceed = true;
    r1.action = fem::ControlActionType::ActivateLoad;
    r1.target_id = 10;

    // Rule 2: deactivate load 20
    auto& r2 = controls.add_rule(2);
    r2.sensor_id = 1;
    r2.trigger_on_exceed = true;
    r2.action = fem::ControlActionType::DeactivateLoad;
    r2.target_id = 20;

    // Trigger both
    std::size_t nn = 2;
    std::vector<Real> accel = {100.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    sensors.measure_all(0.0, 1.0e-5, nn, nullptr, nullptr, accel.data());
    auto actions = controls.evaluate(sensors);
    CHECK(actions.size() == 2, "two actions triggered");

    auto active = controls.active_loads();
    CHECK(active.size() == 1 && active[0] == 10, "load 10 activated");

    auto deactivated = controls.deactivated_loads();
    CHECK(deactivated.size() == 1 && deactivated[0] == 20, "load 20 deactivated");
}

// ============================================================================
// Test 11: Control one-shot vs repeating
// ============================================================================
void test_control_oneshot() {
    std::cout << "Test 11: One-shot vs repeating\n";

    fem::SensorManager sensors;
    fem::SensorConfig scfg;
    scfg.type = fem::SensorType::Accelerometer;
    scfg.id = 1;
    scfg.node_id = 0;
    scfg.direction = fem::SensorDirection::X;
    scfg.threshold_value = 50.0;
    scfg.threshold_above = true;
    sensors.add_sensor(scfg);

    fem::ControlManager controls;
    auto& rule = controls.add_rule(1);
    rule.sensor_id = 1;
    rule.trigger_on_exceed = true;
    rule.action = fem::ControlActionType::WriteCheckpoint;
    rule.one_shot = true;

    std::size_t nn = 2;
    std::vector<Real> accel = {100.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    sensors.measure_all(0.0, 1.0e-5, nn, nullptr, nullptr, accel.data());

    // First evaluation
    auto actions1 = controls.evaluate(sensors);
    CHECK(actions1.size() == 1, "triggered first time");

    // Second evaluation of same state - one_shot should prevent re-trigger
    auto actions2 = controls.evaluate(sensors);
    CHECK(actions2.empty(), "one-shot prevents re-trigger");

    // Reset and re-trigger
    controls.reset_all();
    auto actions3 = controls.evaluate(sensors);
    CHECK(actions3.size() == 1, "triggers again after reset");
}

// ============================================================================
// Test 12: Control custom callback
// ============================================================================
void test_control_callback() {
    std::cout << "Test 12: Custom callback\n";

    fem::SensorManager sensors;
    fem::SensorConfig scfg;
    scfg.type = fem::SensorType::Accelerometer;
    scfg.id = 1;
    scfg.node_id = 0;
    scfg.direction = fem::SensorDirection::X;
    scfg.threshold_value = 10.0;
    scfg.threshold_above = true;
    sensors.add_sensor(scfg);

    int callback_count = 0;
    fem::ControlManager controls;
    auto& rule = controls.add_rule(1);
    rule.sensor_id = 1;
    rule.trigger_on_exceed = true;
    rule.action = fem::ControlActionType::CustomCallback;
    rule.callback = [&callback_count]() { callback_count++; };
    rule.one_shot = true;

    std::size_t nn = 2;
    std::vector<Real> accel = {50.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    sensors.measure_all(0.0, 1.0e-5, nn, nullptr, nullptr, accel.data());

    controls.evaluate(sensors);
    CHECK(callback_count == 1, "callback invoked once");

    // One-shot: should not invoke again
    controls.evaluate(sensors);
    CHECK(callback_count == 1, "callback not invoked again (one-shot)");
}

// ============================================================================
// Test 13: Mesh adjacency construction
// ============================================================================
void test_mesh_adjacency() {
    std::cout << "Test 13: Mesh adjacency\n";

    // 2 quad4 elements sharing edge (nodes 1-2)
    //  0---1---3
    //  |   |   |
    //  4---2---5
    //
    // Element 0: 0,1,2,4
    // Element 1: 1,3,5,2

    std::vector<Index> conn = {0, 1, 2, 4,   1, 3, 5, 2};
    physics::MeshAdjacency adj;
    adj.build(6, 2, conn.data(), 4);

    CHECK(adj.num_nodes() == 6, "6 nodes");

    // Node 0: connected to 1, 2, 4 (from element 0)
    CHECK(adj.num_neighbors(0) == 3, "node 0 has 3 neighbors");
    CHECK(adj.neighbors(0).count(1) == 1, "0 connected to 1");
    CHECK(adj.neighbors(0).count(2) == 1, "0 connected to 2");
    CHECK(adj.neighbors(0).count(4) == 1, "0 connected to 4");

    // Node 1: connected to 0,2,4 (elem 0) + 3,5,2 (elem 1) = 0,2,3,4,5
    CHECK(adj.num_neighbors(1) == 5, "node 1 has 5 neighbors");

    // Node 2: connected to 0,1,4 (elem 0) + 1,3,5 (elem 1) = 0,1,3,4,5
    CHECK(adj.num_neighbors(2) == 5, "node 2 has 5 neighbors");
}

// ============================================================================
// Test 14: Laplacian mesh smoothing
// ============================================================================
void test_laplacian_smoothing() {
    std::cout << "Test 14: Laplacian smoothing\n";

    // 4 nodes forming a quad, center node displaced
    //  0-----1
    //  |  2  |
    //  3-----4  (but node 2 is off-center)
    //
    // 2 triangle elements: 0,1,2 and 0,2,3 and 1,4,2 and 2,4,3
    // Simplify: just 2 triangles sharing node 2
    // Element 0: 0, 1, 2
    // Element 1: 0, 2, 3
    // Element 2: 1, 4, 2
    // Element 3: 2, 4, 3

    std::vector<Real> coords = {
        0.0, 1.0, 0.0,   // node 0
        1.0, 1.0, 0.0,   // node 1
        0.6, 0.6, 0.0,   // node 2 (slightly off center, should be 0.5, 0.5)
        0.0, 0.0, 0.0,   // node 3
        1.0, 0.0, 0.0    // node 4
    };

    std::vector<Index> conn = {0,1,2,  0,2,3,  1,4,2,  2,4,3};
    std::set<Index> boundary = {0, 1, 3, 4};

    physics::ALESolver ale;
    physics::ALEConfig cfg;
    cfg.smoothing = physics::SmoothingMethod::Laplacian;
    cfg.smoothing_weight = 1.0;  // Full Laplacian
    cfg.smoothing_iterations = 5;
    cfg.boundary_fixed = true;
    ale.set_config(cfg);

    ale.initialize(5, 4, conn.data(), 3, boundary);

    Real max_disp = ale.smooth(coords.data());
    CHECK(max_disp > 0.0, "smoothing displaced nodes");

    // Node 2 should move toward center of its neighbors (0,1,3,4)
    // Average of neighbors: ((0+1+0+1)/4, (1+1+0+0)/4, 0) = (0.5, 0.5, 0)
    CHECK(near(coords[6], 0.5, 0.01), "node 2 x ~ 0.5 after smoothing");
    CHECK(near(coords[7], 0.5, 0.01), "node 2 y ~ 0.5 after smoothing");

    // Boundary nodes should not move
    CHECK(near(coords[0], 0.0) && near(coords[1], 1.0), "node 0 fixed");
    CHECK(near(coords[3], 1.0) && near(coords[4], 1.0), "node 1 fixed");
    CHECK(near(coords[9], 0.0) && near(coords[10], 0.0), "node 3 fixed");
}

// ============================================================================
// Test 15: Weighted mesh smoothing
// ============================================================================
void test_weighted_smoothing() {
    std::cout << "Test 15: Weighted smoothing\n";

    // Same setup as test 14 but with weighted smoothing
    std::vector<Real> coords = {
        0.0, 1.0, 0.0,
        1.0, 1.0, 0.0,
        0.6, 0.6, 0.0,   // off-center
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0
    };

    std::vector<Index> conn = {0,1,2,  0,2,3,  1,4,2,  2,4,3};
    std::set<Index> boundary = {0, 1, 3, 4};

    physics::ALESolver ale;
    physics::ALEConfig cfg;
    cfg.smoothing = physics::SmoothingMethod::Weighted;
    cfg.smoothing_weight = 1.0;
    cfg.smoothing_iterations = 10;
    cfg.boundary_fixed = true;
    ale.set_config(cfg);

    ale.initialize(5, 4, conn.data(), 3, boundary);

    Real max_disp = ale.smooth(coords.data());
    CHECK(max_disp > 0.0, "weighted smoothing displaced nodes");

    // Node 2 should move toward center (weighted average may differ slightly)
    CHECK(near(coords[6], 0.5, 0.05), "node 2 x near center");
    CHECK(near(coords[7], 0.5, 0.05), "node 2 y near center");

    // Boundary nodes fixed
    CHECK(near(coords[0], 0.0), "boundary x preserved");
    CHECK(near(coords[9], 0.0), "boundary node 3 preserved");
}

// ============================================================================
// Test 16: Boundary node preservation
// ============================================================================
void test_boundary_preservation() {
    std::cout << "Test 16: Boundary node preservation\n";

    // 3 nodes in a line, middle one off
    // 0---1---2 (but node 1 is displaced)
    std::vector<Real> coords = {
        0.0, 0.0, 0.0,   // node 0
        0.6, 0.1, 0.0,   // node 1 (should be ~0.5, 0, 0)
        1.0, 0.0, 0.0    // node 2
    };

    std::vector<Index> conn = {0, 1, 2};  // One triangle-like element

    // Case 1: All boundary - no smoothing
    {
        std::set<Index> boundary = {0, 1, 2};
        physics::ALESolver ale;
        physics::ALEConfig cfg;
        cfg.smoothing_weight = 1.0;
        cfg.smoothing_iterations = 5;
        cfg.boundary_fixed = true;
        ale.set_config(cfg);
        ale.initialize(3, 1, conn.data(), 3, boundary);

        std::vector<Real> coords_copy = coords;
        Real disp = ale.smooth(coords_copy.data());
        CHECK(near(disp, 0.0), "no smoothing when all nodes are boundary");
    }

    // Case 2: boundary_fixed = false - all nodes smooth
    {
        std::set<Index> boundary = {0, 2};
        physics::ALESolver ale;
        physics::ALEConfig cfg;
        cfg.smoothing_weight = 1.0;
        cfg.smoothing_iterations = 5;
        cfg.boundary_fixed = false;
        ale.set_config(cfg);
        ale.initialize(3, 1, conn.data(), 3, boundary);

        std::vector<Real> coords_copy = coords;
        ale.smooth(coords_copy.data());
        // All nodes should move (including boundary) since boundary_fixed=false
        // Node 0 neighbors are 1,2 -> avg = ((0.6+1.0)/2, (0.1+0.0)/2, 0) = (0.8, 0.05, 0)
        // So node 0 moves (it's no longer pinned)
        CHECK(!near(coords_copy[0], 0.0, 1e-10), "node 0 moved when boundary_fixed=false");
    }
}

// ============================================================================
// Test 17: Donor cell advection
// ============================================================================
void test_donor_cell_advection() {
    std::cout << "Test 17: Donor cell advection\n";

    // 3 nodes in a line
    std::vector<Real> old_coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        2.0, 0.0, 0.0
    };

    // Move node 1 slightly to the right
    std::vector<Real> new_coords = {
        0.0, 0.0, 0.0,
        1.1, 0.0, 0.0,   // moved +0.1
        2.0, 0.0, 0.0
    };

    // Scalar field: step function
    std::vector<Real> field = {10.0, 5.0, 1.0};

    std::vector<Index> conn = {0, 1, 2};

    physics::ALESolver ale;
    physics::ALEConfig cfg;
    cfg.advection = physics::AdvectionMethod::DonorCell;
    cfg.smoothing_weight = 0.0;  // No smoothing in this test
    ale.set_config(cfg);
    ale.initialize(3, 1, conn.data(), 3);

    Real dt = 1.0;
    ale.advect_scalar(old_coords.data(), new_coords.data(), field.data(), dt);

    // Node 0 didn't move - field[0] should stay the same
    CHECK(near(field[0], 10.0), "unmoved node field unchanged");

    // Node 1 moved right, so upstream is node 0 (left)
    // Should pick up some of field[0]=10 into field[1]=5
    CHECK(field[1] >= 5.0, "advected field at moved node >= original");
}

// ============================================================================
// Test 18: Van Leer advection
// ============================================================================
void test_van_leer_advection() {
    std::cout << "Test 18: Van Leer advection\n";

    // 3 nodes in a line
    std::vector<Real> old_coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        2.0, 0.0, 0.0
    };

    std::vector<Real> new_coords = {
        0.0, 0.0, 0.0,
        1.1, 0.0, 0.0,
        2.0, 0.0, 0.0
    };

    std::vector<Real> field = {10.0, 5.0, 1.0};
    std::vector<Real> field_dc = field;  // Copy for donor cell comparison

    std::vector<Index> conn = {0, 1, 2};

    physics::ALESolver ale_vl;
    physics::ALEConfig cfg_vl;
    cfg_vl.advection = physics::AdvectionMethod::VanLeer;
    ale_vl.set_config(cfg_vl);
    ale_vl.initialize(3, 1, conn.data(), 3);

    physics::ALESolver ale_dc;
    physics::ALEConfig cfg_dc;
    cfg_dc.advection = physics::AdvectionMethod::DonorCell;
    ale_dc.set_config(cfg_dc);
    ale_dc.initialize(3, 1, conn.data(), 3);

    Real dt = 1.0;
    ale_vl.advect_scalar(old_coords.data(), new_coords.data(), field.data(), dt);
    ale_dc.advect_scalar(old_coords.data(), new_coords.data(), field_dc.data(), dt);

    CHECK(std::isfinite(field[1]), "Van Leer result is finite");
    // Van Leer should be less diffusive (different from donor cell for non-uniform field)
    // Both should be >= 5 since upstream has value 10
    CHECK(field[1] >= 5.0 - 0.01, "Van Leer preserves lower bound");
}

// ============================================================================
// Test 19: Full ALE step (smooth + advect)
// ============================================================================
void test_ale_full_step() {
    std::cout << "Test 19: Full ALE step\n";

    // 5 nodes: 4 fixed boundary, 1 interior
    std::vector<Real> coords = {
        0.0, 1.0, 0.0,
        1.0, 1.0, 0.0,
        0.6, 0.6, 0.0,   // interior (off-center)
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0
    };

    std::vector<Real> velocities = {
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,   // node 2 has velocity
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    };

    std::vector<Real> density = {1.0, 1.0, 2.0, 1.0, 1.0};  // scalar field
    std::vector<Real*> scalars = {density.data()};

    std::vector<Index> conn = {0,1,2,  0,2,3,  1,4,2,  2,4,3};
    std::set<Index> boundary = {0, 1, 3, 4};

    physics::ALESolver ale;
    physics::ALEConfig cfg;
    cfg.smoothing = physics::SmoothingMethod::Laplacian;
    cfg.smoothing_weight = 0.5;
    cfg.smoothing_iterations = 3;
    cfg.boundary_fixed = true;
    cfg.advection = physics::AdvectionMethod::DonorCell;
    ale.set_config(cfg);
    ale.initialize(5, 4, conn.data(), 3, boundary);

    Real dt = 1.0e-3;
    Real disp = ale.ale_step(coords.data(), velocities.data(), scalars, dt);

    CHECK(disp > 0.0, "ALE step produced smoothing");
    CHECK(ale.ale_step_count() == 1, "one ALE step completed");
    CHECK(ale.total_smoothing_displacement() > 0.0, "total smoothing > 0");

    // Boundary nodes should still be fixed
    CHECK(near(coords[0], 0.0) && near(coords[1], 1.0), "boundary preserved after ALE");
    CHECK(near(coords[9], 0.0) && near(coords[10], 0.0), "boundary preserved after ALE (2)");

    // Velocities and density should be finite
    for (int i = 0; i < 5; ++i) {
        CHECK(std::isfinite(velocities[3*i]) && std::isfinite(velocities[3*i+1]),
              "velocity finite after advection");
        CHECK(std::isfinite(density[i]), "density finite after advection");
    }
}

// ============================================================================
// Test 20: Element quality metric
// ============================================================================
void test_element_quality() {
    std::cout << "Test 20: Element quality\n";

    // Perfect unit square quad
    std::vector<Real> coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0
    };
    Index elem_perfect[] = {0, 1, 2, 3};
    Real q1 = physics::ALESolver::element_quality(coords.data(), elem_perfect, 4);
    // For square: min edge = 1.0, max edge = sqrt(2) (diagonal)
    // Ratio = 1/sqrt(2) ~ 0.707
    CHECK(q1 > 0.5 && q1 < 1.0, "square quality good (0.5-1.0)");
    CHECK(near(q1, 1.0 / std::sqrt(2.0), 0.01), "square quality = 1/sqrt(2)");

    // Degenerate element (all same point)
    std::vector<Real> degen_coords = {
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    };
    Index elem_degen[] = {0, 1, 2, 3};
    Real q2 = physics::ALESolver::element_quality(degen_coords.data(), elem_degen, 4);
    CHECK(near(q2, 0.0), "degenerate element quality = 0");

    // Stretched element (high aspect ratio)
    std::vector<Real> stretched = {
        0.0, 0.0, 0.0,
        10.0, 0.0, 0.0,
        10.0, 0.1, 0.0,
        0.0,  0.1, 0.0
    };
    Index elem_stretch[] = {0, 1, 2, 3};
    Real q3 = physics::ALESolver::element_quality(stretched.data(), elem_stretch, 4);
    CHECK(q3 < q1, "stretched element has lower quality");
    CHECK(q3 > 0.0 && q3 < 0.5, "stretched quality in low range");

    // Average quality
    // Use perfect square elements
    std::vector<Real> mesh_coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        2.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        1.0, 1.0, 0.0,
        2.0, 1.0, 0.0
    };
    std::vector<Index> mesh_conn = {0,1,4,3,  1,2,5,4};
    physics::ALESolver ale;
    Real avg = ale.average_quality(mesh_coords.data(), mesh_conn.data(), 2, 4);
    CHECK(near(avg, 1.0/std::sqrt(2.0), 0.01), "average quality of uniform mesh");
}

// ============================================================================
// Test 21: Force sensor
// ============================================================================
void test_force_sensor() {
    std::cout << "Test 21: Force sensor\n";

    fem::SensorConfig cfg;
    cfg.type = fem::SensorType::ForceSensor;
    cfg.id = 30;
    cfg.name = "Force_1";
    cfg.node_id = 0;
    cfg.direction = fem::SensorDirection::Y;

    fem::Sensor sensor(cfg);

    std::size_t nn = 2;
    std::vector<Real> forces = {100.0, -250.0, 0.0,  0.0, 0.0, 0.0};

    sensor.measure(0.0, 1.0e-5, nn,
                   nullptr, nullptr, nullptr, forces.data());

    CHECK(sensor.num_readings() == 1, "one reading");
    CHECK(near(sensor.current_value(), -250.0), "force Y = -250");
}

// ============================================================================
// Test 22: Sensor direction components
// ============================================================================
void test_sensor_directions() {
    std::cout << "Test 22: Sensor directions\n";

    std::size_t nn = 2;
    std::vector<Real> accel = {3.0, 4.0, 5.0, 0.0, 0.0, 0.0};

    // Test each direction
    auto make_sensor = [](fem::SensorDirection dir) {
        fem::SensorConfig cfg;
        cfg.type = fem::SensorType::Accelerometer;
        cfg.id = 0;
        cfg.node_id = 0;
        cfg.direction = dir;
        return fem::Sensor(cfg);
    };

    fem::Sensor sx = make_sensor(fem::SensorDirection::X);
    sx.measure(0.0, 1e-5, nn, nullptr, nullptr, accel.data(), nullptr);
    CHECK(near(sx.current_value(), 3.0), "direction X = 3");

    fem::Sensor sy = make_sensor(fem::SensorDirection::Y);
    sy.measure(0.0, 1e-5, nn, nullptr, nullptr, accel.data(), nullptr);
    CHECK(near(sy.current_value(), 4.0), "direction Y = 4");

    fem::Sensor sz = make_sensor(fem::SensorDirection::Z);
    sz.measure(0.0, 1e-5, nn, nullptr, nullptr, accel.data(), nullptr);
    CHECK(near(sz.current_value(), 5.0), "direction Z = 5");

    fem::Sensor smag = make_sensor(fem::SensorDirection::Magnitude);
    smag.measure(0.0, 1e-5, nn, nullptr, nullptr, accel.data(), nullptr);
    Real expected_mag = std::sqrt(3.0*3.0 + 4.0*4.0 + 5.0*5.0);
    CHECK(near(smag.current_value(), expected_mag, 1e-10), "direction Magnitude");

    fem::Sensor sxy = make_sensor(fem::SensorDirection::ResultantXY);
    sxy.measure(0.0, 1e-5, nn, nullptr, nullptr, accel.data(), nullptr);
    Real expected_xy = std::sqrt(3.0*3.0 + 4.0*4.0);
    CHECK(near(sxy.current_value(), expected_xy, 1e-10), "direction ResultantXY = 5");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "Wave 8: Sensors, Controls, and ALE Test\n";
    std::cout << "========================================\n\n";

    // Sensor tests
    test_accelerometer_basic();
    test_velocity_gauge();
    test_distance_sensor();
    test_strain_gauge();
    test_cfc_filter();
    test_sampling_interval();
    test_threshold_detection();
    test_sensor_manager();

    // Control tests
    test_control_terminate();
    test_control_loads();
    test_control_oneshot();
    test_control_callback();

    // ALE tests
    test_mesh_adjacency();
    test_laplacian_smoothing();
    test_weighted_smoothing();
    test_boundary_preservation();
    test_donor_cell_advection();
    test_van_leer_advection();
    test_ale_full_step();
    test_element_quality();

    // Additional tests
    test_force_sensor();
    test_sensor_directions();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed\n";
    std::cout << "========================================\n";

    return tests_failed > 0 ? 1 : 0;
}
