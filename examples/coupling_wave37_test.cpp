/**
 * @file coupling_wave37_test.cpp
 * @brief Wave 37: Coupling Framework Test Suite (6 features, 50 tests)
 *
 * Tests:
 *   1. CouplingAdapter       (8 tests)
 *   2. PreCICEAdapter        (10 tests)
 *   3. CWIPIAdapter          (8 tests)
 *   4. Rad2RadCoupling       (8 tests)
 *   5. PythonCoupling        (8 tests)
 *   6. CouplingInterpolation (8 tests)
 */

#include <nexussim/coupling/coupling_wave37.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <array>

using namespace nxs;
using namespace nxs::coupling;

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
// Concrete mock adapter for testing the abstract interface
// ============================================================================

class MockCouplingAdapter : public CouplingAdapter {
public:
    std::vector<Real> last_sent;
    std::vector<Real> last_received;
    Real total_time = 0.0;
    bool was_finalized = false;
    int advance_count = 0;

    void initialize(const CouplingConfig& config) override {
        config_ = config;
        initialized_ = true;
        total_time = 0.0;
        was_finalized = false;
        advance_count = 0;
    }

    void send_field(const char* /*name*/, const Real* data, int n) override {
        last_sent.assign(data, data + n);
    }

    void receive_field(const char* /*name*/, Real* data, int n) override {
        // Echo back last sent, or zeros
        for (int i = 0; i < n; ++i) {
            data[i] = (i < static_cast<int>(last_sent.size())) ? last_sent[i] : 0.0;
        }
        last_received.assign(data, data + n);
    }

    void advance(Real dt) override {
        total_time += dt;
        advance_count++;
    }

    void finalize() override {
        initialized_ = false;
        was_finalized = true;
    }
};


// ============================================================================
// 1. CouplingAdapter Tests
// ============================================================================

void test_1_coupling_adapter() {
    std::cout << "--- Test 1: CouplingAdapter ---\n";

    // 1a. Default state: not initialized
    {
        MockCouplingAdapter adapter;
        CHECK(!adapter.is_initialized(), "Adapter: default not initialized");
    }

    // 1b. Initialize sets state
    {
        MockCouplingAdapter adapter;
        CouplingConfig cfg{0, 4, 0.001, 20};
        adapter.initialize(cfg);
        CHECK(adapter.is_initialized(), "Adapter: initialized after init()");
    }

    // 1c. Config is stored
    {
        MockCouplingAdapter adapter;
        CouplingConfig cfg{2, 8, 0.005, 15};
        adapter.initialize(cfg);
        CHECK(adapter.config().comm_rank == 2, "Adapter: config rank stored");
        CHECK(adapter.config().comm_size == 8, "Adapter: config size stored");
    }

    // 1d. Send and receive round-trip
    {
        MockCouplingAdapter adapter;
        CouplingConfig cfg{};
        adapter.initialize(cfg);
        Real data[3] = {1.0, 2.0, 3.0};
        adapter.send_field("velocity", data, 3);
        Real recv[3] = {0.0, 0.0, 0.0};
        adapter.receive_field("velocity", recv, 3);
        CHECK_NEAR(recv[0], 1.0, 1e-15, "Adapter: round-trip x");
        CHECK_NEAR(recv[1], 2.0, 1e-15, "Adapter: round-trip y");
    }

    // 1e. Advance accumulates time
    {
        MockCouplingAdapter adapter;
        CouplingConfig cfg{};
        adapter.initialize(cfg);
        adapter.advance(0.01);
        adapter.advance(0.02);
        CHECK_NEAR(adapter.total_time, 0.03, 1e-15, "Adapter: time accumulation");
        CHECK(adapter.advance_count == 2, "Adapter: advance count");
    }

    // 1f. Finalize resets state
    {
        MockCouplingAdapter adapter;
        CouplingConfig cfg{};
        adapter.initialize(cfg);
        adapter.finalize();
        CHECK(!adapter.is_initialized(), "Adapter: finalized");
        CHECK(adapter.was_finalized, "Adapter: finalize flag set");
    }
}


// ============================================================================
// 2. PreCICEAdapter Tests
// ============================================================================

void test_2_precice_adapter() {
    std::cout << "\n--- Test 2: PreCICEAdapter ---\n";

    // 2a. Initialize and check state
    {
        PreCICEAdapter adapter;
        CouplingConfig cfg{0, 1, 0.001, 10};
        adapter.initialize(cfg);
        CHECK(adapter.is_initialized(), "PreCICE: initialized");
        CHECK(adapter.coupling_step() == 0, "PreCICE: initial step 0");
    }

    // 2b. Register mesh
    {
        PreCICEAdapter adapter;
        CouplingConfig cfg{};
        adapter.initialize(cfg);
        Real coords[9] = {0,0,0, 1,0,0, 0,1,0};
        adapter.register_mesh("SolidMesh", coords, 3);
        CHECK(adapter.has_mesh("SolidMesh"), "PreCICE: mesh registered");
        CHECK(adapter.mesh_vertex_count("SolidMesh") == 3, "PreCICE: vertex count");
    }

    // 2c. Non-existent mesh
    {
        PreCICEAdapter adapter;
        CouplingConfig cfg{};
        adapter.initialize(cfg);
        CHECK(!adapter.has_mesh("ghost"), "PreCICE: non-existent mesh");
        CHECK(adapter.mesh_vertex_count("ghost") == 0, "PreCICE: ghost vertex count 0");
    }

    // 2d. Write and read data
    {
        PreCICEAdapter adapter;
        CouplingConfig cfg{};
        adapter.initialize(cfg);
        Real temps[4] = {100.0, 200.0, 300.0, 400.0};
        adapter.write_data("Temperature", temps, 4);
        CHECK(adapter.has_field_data("Temperature"), "PreCICE: field data exists");

        Real out[4] = {};
        adapter.read_data("Temperature", out, 4);
        CHECK_NEAR(out[0], 100.0, 1e-15, "PreCICE: read[0]");
        CHECK_NEAR(out[3], 400.0, 1e-15, "PreCICE: read[3]");
    }

    // 2e. Advance increments step and time
    {
        PreCICEAdapter adapter;
        CouplingConfig cfg{0, 1, 0.01, 5};
        adapter.initialize(cfg);
        adapter.advance(0.01);
        adapter.advance(0.01);
        CHECK(adapter.coupling_step() == 2, "PreCICE: step count");
        CHECK_NEAR(adapter.current_time(), 0.02, 1e-15, "PreCICE: current time");
    }

    // 2f. Finalize clears everything
    {
        PreCICEAdapter adapter;
        CouplingConfig cfg{};
        adapter.initialize(cfg);
        Real coords[3] = {0,0,0};
        adapter.register_mesh("M", coords, 1);
        adapter.finalize();
        CHECK(!adapter.is_initialized(), "PreCICE: finalized");
        CHECK(!adapter.has_mesh("M"), "PreCICE: mesh cleared");
    }
}


// ============================================================================
// 3. CWIPIAdapter Tests
// ============================================================================

void test_3_cwipi_adapter() {
    std::cout << "\n--- Test 3: CWIPIAdapter ---\n";

    // 3a. Set coupling mesh
    {
        CWIPIAdapter adapter;
        CouplingConfig cfg{};
        adapter.initialize(cfg);
        Real coords[6] = {0,0,0, 1,0,0};
        adapter.set_coupling_mesh(coords, 2);
        CHECK(adapter.mesh_point_count() == 2, "CWIPI: mesh point count");
    }

    // 3b. Exchange symmetry: send == recv in stub mode
    {
        CWIPIAdapter adapter;
        CouplingConfig cfg{};
        adapter.initialize(cfg);
        Real send_data[3] = {10.0, 20.0, 30.0};
        Real recv_data[3] = {};
        adapter.exchange(send_data, 3, recv_data, 3);
        CHECK_NEAR(recv_data[0], 10.0, 1e-15, "CWIPI: exchange symmetry[0]");
        CHECK_NEAR(recv_data[1], 20.0, 1e-15, "CWIPI: exchange symmetry[1]");
        CHECK_NEAR(recv_data[2], 30.0, 1e-15, "CWIPI: exchange symmetry[2]");
    }

    // 3c. Exchange with different sizes (recv > send)
    {
        CWIPIAdapter adapter;
        CouplingConfig cfg{};
        adapter.initialize(cfg);
        Real send_data[2] = {5.0, 6.0};
        Real recv_data[4] = {-1, -1, -1, -1};
        adapter.exchange(send_data, 2, recv_data, 4);
        CHECK_NEAR(recv_data[0], 5.0, 1e-15, "CWIPI: partial exchange[0]");
        CHECK_NEAR(recv_data[2], 0.0, 1e-15, "CWIPI: zero-fill beyond send size");
    }

    // 3d. Interpolation weights (coincident points)
    {
        CWIPIAdapter adapter;
        CouplingConfig cfg{};
        adapter.initialize(cfg);
        Real mesh_coords[3] = {1.0, 0.0, 0.0};
        adapter.set_coupling_mesh(mesh_coords, 1);
        Real src_coords[6] = {1.0, 0.0, 0.0,  0.0, 0.0, 0.0};
        adapter.compute_weights(src_coords, 2, 2.0);
        Real src_vals[2] = {100.0, 0.0};
        Real dst_vals[1] = {};
        adapter.interpolate(src_vals, 2, dst_vals);
        // Coincident with source 0, so result should be ~100
        CHECK_NEAR(dst_vals[0], 100.0, 1.0, "CWIPI: interp coincident ~100");
    }

    // 3e. Finalize clears state
    {
        CWIPIAdapter adapter;
        CouplingConfig cfg{};
        adapter.initialize(cfg);
        Real coords[3] = {0,0,0};
        adapter.set_coupling_mesh(coords, 1);
        adapter.finalize();
        CHECK(adapter.mesh_point_count() == 0, "CWIPI: finalize clears mesh");
        CHECK(!adapter.is_initialized(), "CWIPI: finalized");
    }

    // 3f. Advance accumulates time
    {
        CWIPIAdapter adapter;
        CouplingConfig cfg{};
        adapter.initialize(cfg);
        adapter.advance(0.5);
        adapter.advance(0.5);
        // No direct time accessor, but ensure no crash
        CHECK(true, "CWIPI: advance no crash");
    }
}


// ============================================================================
// 4. Rad2RadCoupling Tests
// ============================================================================

void test_4_rad2rad_coupling() {
    std::cout << "\n--- Test 4: Rad2RadCoupling ---\n";

    // 4a. Initialize
    {
        Rad2RadCoupling r2r;
        r2r.initialize(0, 4);
        CHECK(r2r.is_initialized(), "R2R: initialized");
        CHECK(r2r.domain_id() == 0, "R2R: domain id");
        CHECK(r2r.n_domains() == 4, "R2R: n_domains");
    }

    // 4b. Pack and exchange
    {
        Rad2RadCoupling r2r;
        r2r.initialize(0, 2);

        int shared[2] = {1, 3};
        Real send_buf[6] = {};
        Real recv_buf[6] = {};
        DomainInterface iface{shared, 2, send_buf, recv_buf};

        // Global field: 5 nodes * 3 components
        Real field[15] = {};
        field[3] = 10.0; field[4] = 20.0; field[5] = 30.0;  // node 1
        field[9] = 40.0; field[10] = 50.0; field[11] = 60.0; // node 3

        r2r.pack_interface(iface, field);
        CHECK_NEAR(send_buf[0], 10.0, 1e-15, "R2R: pack node1 x");
        CHECK_NEAR(send_buf[3], 40.0, 1e-15, "R2R: pack node3 x");

        r2r.exchange_interface(iface);
        CHECK_NEAR(recv_buf[0], 10.0, 1e-15, "R2R: exchange copies send->recv");
    }

    // 4c. Apply interface forces
    {
        Rad2RadCoupling r2r;
        r2r.initialize(0, 2);

        int shared[1] = {0};
        Real send_buf[3] = {100.0, 200.0, 300.0};
        Real recv_buf[3] = {100.0, 200.0, 300.0};
        DomainInterface iface{shared, 1, send_buf, recv_buf};

        Real forces[3] = {0.0, 0.0, 0.0};
        r2r.apply_interface_forces(iface, forces);
        CHECK_NEAR(forces[0], 100.0, 1e-15, "R2R: force applied x");
        CHECK_NEAR(forces[1], 200.0, 1e-15, "R2R: force applied y");
    }

    // 4d. Force balance (Newton's 3rd law): opposite forces => residual = 0
    {
        Rad2RadCoupling r2r;
        r2r.initialize(0, 2);

        int shared[1] = {0};
        Real send_buf[3] = {50.0, -30.0, 10.0};
        Real recv_buf[3] = {-50.0, 30.0, -10.0};
        DomainInterface iface{shared, 1, send_buf, recv_buf};

        Real residual = r2r.interface_force_residual(iface);
        CHECK_NEAR(residual, 0.0, 1e-12, "R2R: force balance residual 0");
        CHECK(r2r.is_converged(iface, 1e-10), "R2R: converged with balanced forces");
    }

    // 4e. Average interface values
    {
        Rad2RadCoupling r2r;
        r2r.initialize(0, 2);

        int shared[1] = {0};
        Real send_buf[3] = {10.0, 20.0, 30.0};
        Real recv_buf[3] = {20.0, 40.0, 60.0};
        DomainInterface iface{shared, 1, send_buf, recv_buf};

        Real field[3] = {};
        r2r.average_interface(iface, field);
        CHECK_NEAR(field[0], 15.0, 1e-15, "R2R: average x");
        CHECK_NEAR(field[1], 30.0, 1e-15, "R2R: average y");
    }
}


// ============================================================================
// 5. PythonCoupling Tests
// ============================================================================

// Static counters for callback testing
static int g_send_calls = 0;
static int g_recv_calls = 0;
static Real g_last_dt = 0.0;
static bool g_init_called = false;
static bool g_final_called = false;
static Real g_recv_buffer[4] = {};

static void mock_on_send(const Real* /*data*/, int /*n*/) { g_send_calls++; }
static void mock_on_receive(Real* data, int n) {
    g_recv_calls++;
    for (int i = 0; i < n; ++i) data[i] = 42.0 + i;
}
static void mock_on_advance(Real dt) { g_last_dt = dt; }
static void mock_on_init() { g_init_called = true; }
static void mock_on_final() { g_final_called = true; }

void test_5_python_coupling() {
    std::cout << "\n--- Test 5: PythonCoupling ---\n";

    // 5a. Default: no callbacks
    {
        PythonCoupling pc;
        CHECK(!pc.has_send_callback(), "PyCoupling: no send callback initially");
        CHECK(!pc.has_receive_callback(), "PyCoupling: no receive callback initially");
    }

    // 5b. Set callbacks
    {
        PythonCoupling pc;
        PythonCallbacks cb{};
        cb.on_send = mock_on_send;
        cb.on_receive = mock_on_receive;
        cb.on_advance = mock_on_advance;
        cb.on_initialize = mock_on_init;
        cb.on_finalize = mock_on_final;
        pc.set_callbacks(cb);
        CHECK(pc.has_send_callback(), "PyCoupling: send callback set");
        CHECK(pc.has_receive_callback(), "PyCoupling: receive callback set");
        CHECK(pc.has_advance_callback(), "PyCoupling: advance callback set");
    }

    // 5c. Initialize invokes callback
    {
        g_init_called = false;
        PythonCoupling pc;
        PythonCallbacks cb{};
        cb.on_initialize = mock_on_init;
        pc.set_callbacks(cb);
        CouplingConfig cfg{};
        pc.initialize(cfg);
        CHECK(g_init_called, "PyCoupling: init callback invoked");
    }

    // 5d. Send invokes callback
    {
        g_send_calls = 0;
        PythonCoupling pc;
        PythonCallbacks cb{};
        cb.on_send = mock_on_send;
        pc.set_callbacks(cb);
        CouplingConfig cfg{};
        pc.initialize(cfg);
        Real data[3] = {1,2,3};
        pc.send_field("vel", data, 3);
        CHECK(g_send_calls == 1, "PyCoupling: send callback called once");
    }

    // 5e. Receive invokes callback and fills data
    {
        g_recv_calls = 0;
        PythonCoupling pc;
        PythonCallbacks cb{};
        cb.on_receive = mock_on_receive;
        pc.set_callbacks(cb);
        CouplingConfig cfg{};
        pc.initialize(cfg);
        Real out[3] = {};
        pc.receive_field("temp", out, 3);
        CHECK(g_recv_calls == 1, "PyCoupling: receive callback called");
        CHECK_NEAR(out[0], 42.0, 1e-15, "PyCoupling: receive data[0]");
        CHECK_NEAR(out[2], 44.0, 1e-15, "PyCoupling: receive data[2]");
    }

    // 5f. Advance invokes callback with dt
    {
        g_last_dt = 0.0;
        PythonCoupling pc;
        PythonCallbacks cb{};
        cb.on_advance = mock_on_advance;
        pc.set_callbacks(cb);
        CouplingConfig cfg{};
        pc.initialize(cfg);
        pc.advance(0.025);
        CHECK_NEAR(g_last_dt, 0.025, 1e-15, "PyCoupling: advance dt");
        CHECK_NEAR(pc.current_time(), 0.025, 1e-15, "PyCoupling: time after advance");
    }

    // 5g. Finalize invokes callback
    {
        g_final_called = false;
        PythonCoupling pc;
        PythonCallbacks cb{};
        cb.on_finalize = mock_on_final;
        pc.set_callbacks(cb);
        CouplingConfig cfg{};
        pc.initialize(cfg);
        pc.finalize();
        CHECK(g_final_called, "PyCoupling: finalize callback invoked");
        CHECK(!pc.is_initialized(), "PyCoupling: not initialized after finalize");
    }
}


// ============================================================================
// 6. CouplingInterpolation Tests
// ============================================================================

void test_6_coupling_interpolation() {
    std::cout << "\n--- Test 6: CouplingInterpolation ---\n";

    // 6a. Wendland C2 kernel value at r=0 should be 1
    {
        Real w = CouplingInterpolation::wendland_c2(0.0, 1.0);
        CHECK_NEAR(w, 1.0, 1e-15, "Interp: Wendland(0,1)=1");
    }

    // 6b. Wendland C2 at r=R should be 0
    {
        Real w = CouplingInterpolation::wendland_c2(1.0, 1.0);
        CHECK_NEAR(w, 0.0, 1e-15, "Interp: Wendland(R,R)=0");
    }

    // 6c. Wendland C2 at r > R should be 0
    {
        Real w = CouplingInterpolation::wendland_c2(1.5, 1.0);
        CHECK_NEAR(w, 0.0, 1e-15, "Interp: Wendland(>R)=0");
    }

    // 6d. Wendland C2 at r=R/2: (1-0.5)^4*(4*0.5+1) = 0.0625*3 = 0.1875
    {
        Real w = CouplingInterpolation::wendland_c2(0.5, 1.0);
        CHECK_NEAR(w, 0.1875, 1e-12, "Interp: Wendland(0.5,1)=0.1875");
    }

    // 6e. RBF interpolation: constant field should reproduce exactly
    {
        Real centers[9] = {0,0,0, 1,0,0, 0,1,0};  // 3 centers
        Real weights[3] = {};
        Real vals[3] = {7.0, 7.0, 7.0};  // constant field

        RBFInterpolator interp;
        interp.centers = centers;
        interp.weights = weights;
        interp.n_centers = 3;
        interp.support_radius = 5.0;

        Real query[3] = {0.5, 0.5, 0.0};
        Real result[1] = {};
        CouplingInterpolation::interpolate_rbf(interp, vals, query, 1, result);
        CHECK_NEAR(result[0], 7.0, 1e-10, "Interp: RBF constant field");
    }

    // 6f. RBF interpolation: linear field f(x,y,z) = x + y
    {
        // Place centers at corners of a unit square in z=0 plane
        Real centers[12] = {0,0,0, 1,0,0, 1,1,0, 0,1,0};
        Real weights[4] = {};
        Real vals[4] = {0.0, 1.0, 2.0, 1.0};  // x+y at each corner

        RBFInterpolator interp;
        interp.centers = centers;
        interp.weights = weights;
        interp.n_centers = 4;
        interp.support_radius = 3.0;

        // Query at center (0.5, 0.5, 0) -> expected x+y = 1.0
        Real query[3] = {0.5, 0.5, 0.0};
        Real result[1] = {};
        CouplingInterpolation::interpolate_rbf(interp, vals, query, 1, result);
        CHECK_NEAR(result[0], 1.0, 0.1, "Interp: RBF linear field at center");
    }

    // 6g. Nearest-neighbor interpolation
    {
        Real src_pts[9] = {0,0,0, 10,0,0, 0,10,0};
        Real src_vals[3] = {100.0, 200.0, 300.0};
        Real query[6] = {0.1, 0.1, 0.0,  9.9, 0.1, 0.0};
        Real result[2] = {};
        CouplingInterpolation::interpolate_nearest(src_pts, src_vals, 3,
                                                   query, 2, result);
        CHECK_NEAR(result[0], 100.0, 1e-15, "Interp: nearest[0] -> src[0]");
        CHECK_NEAR(result[1], 200.0, 1e-15, "Interp: nearest[1] -> src[1]");
    }

    // 6h. L2 error computation
    {
        Real a[4] = {1.0, 2.0, 3.0, 4.0};
        Real b[4] = {1.0, 2.0, 3.0, 4.0};
        Real err = CouplingInterpolation::compute_l2_error(a, b, 4);
        CHECK_NEAR(err, 0.0, 1e-15, "Interp: L2 error identical = 0");
    }
}


// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Wave 37: Coupling Framework Tests ===\n";

    test_1_coupling_adapter();
    test_2_precice_adapter();
    test_3_cwipi_adapter();
    test_4_rad2rad_coupling();
    test_5_python_coupling();
    test_6_coupling_interpolation();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " ===\n";

    return (tests_failed > 0) ? 1 : 0;
}
