/**
 * @file restart_output_test.cpp
 * @brief Comprehensive test for Wave 6: Restart/Checkpoint, Time History, Animation Output
 */

#include <nexussim/io/checkpoint.hpp>
#include <nexussim/io/time_history.hpp>
#include <nexussim/io/animation_writer.hpp>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <filesystem>

using namespace nxs;
using namespace nxs::io;

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

static bool near(Real a, Real b, Real tol = 1.0e-10) {
    return std::fabs(a - b) < tol * (1.0 + std::fabs(b));
}

// Helper: Create a simple test mesh (4 nodes, 1 hex8 element)
static Mesh create_test_mesh() {
    Mesh mesh(4);
    mesh.set_node_coordinates(0, {0.0, 0.0, 0.0});
    mesh.set_node_coordinates(1, {1.0, 0.0, 0.0});
    mesh.set_node_coordinates(2, {1.0, 1.0, 0.0});
    mesh.set_node_coordinates(3, {0.0, 1.0, 0.0});

    auto bid = mesh.add_element_block("quads", ElementType::Shell4, 1, 4);
    auto nodes = mesh.element_block(bid).element_nodes(0);
    nodes[0] = 0; nodes[1] = 1; nodes[2] = 2; nodes[3] = 3;

    return mesh;
}

// ==========================================================================
// Test 1: Checkpoint Header Validity
// ==========================================================================
void test_checkpoint_header() {
    std::cout << "\n=== Test 1: Checkpoint Header ===\n";

    CheckpointHeader hdr;
    CHECK(hdr.is_valid(), "Default header is valid");
    CHECK(hdr.version == 1, "Version = 1");
    CHECK(hdr.precision == sizeof(Real), "Precision matches Real");
    CHECK(hdr.endian_check == 0x01020304, "Endian check marker correct");

    // Corrupt header
    CheckpointHeader bad;
    bad.magic[0] = 'X';
    CHECK(!bad.is_valid(), "Corrupted magic → invalid");

    CheckpointHeader bad2;
    bad2.endian_check = 0x04030201;
    CHECK(!bad2.is_valid(), "Wrong endianness → invalid");
}

// ==========================================================================
// Test 2: Checkpoint Write/Read Round-Trip
// ==========================================================================
void test_checkpoint_roundtrip() {
    std::cout << "\n=== Test 2: Checkpoint Round-Trip ===\n";

    std::string ckpt_file = "/tmp/nxs_test_checkpoint.nxs";

    // Create mesh and state
    Mesh mesh = create_test_mesh();
    State state(mesh);

    // Set non-trivial state
    state.set_time(0.00123);
    state.set_step(456);

    auto& disp = state.displacement();
    disp.at(0, 0) = 0.001;  // Node 0 x-displacement
    disp.at(1, 1) = -0.002; // Node 1 y-displacement
    disp.at(2, 2) = 0.003;  // Node 2 z-displacement

    auto& vel = state.velocity();
    vel.at(0, 0) = 10.0;
    vel.at(3, 2) = -5.0;

    auto& mass_field = state.mass();
    mass_field[0] = 1.0;
    mass_field[1] = 2.0;
    mass_field[2] = 3.0;
    mass_field[3] = 4.0;

    // Create material states
    std::vector<physics::MaterialState> mat_states(2);
    mat_states[0].stress[0] = 1.0e6;   // σxx
    mat_states[0].plastic_strain = 0.01;
    mat_states[0].temperature = 400.0;
    mat_states[1].strain[3] = 0.002;    // γxy
    mat_states[1].damage = 0.5;
    mat_states[1].history[10] = 42.0;

    // Write checkpoint
    CheckpointWriter writer(ckpt_file);
    bool write_ok = writer.write(mesh, state, mat_states);
    CHECK(write_ok, "Checkpoint write succeeded");
    CHECK(writer.bytes_written() > 64, "File has data beyond header");

    // Read checkpoint header
    CheckpointReader reader(ckpt_file);
    bool hdr_ok = reader.read_header();
    CHECK(hdr_ok, "Header read succeeded");
    CHECK(reader.num_nodes() == 4, "Header: 4 nodes");
    CHECK(reader.num_elements() == 1, "Header: 1 element");
    CHECK(near(reader.time(), 0.00123), "Header: time = 0.00123");
    CHECK(reader.step() == 456, "Header: step = 456");

    // Read full state into new objects
    Mesh mesh2(4);
    // Initialize coordinates so the field exists
    mesh2.set_node_coordinates(0, {0,0,0});
    mesh2.set_node_coordinates(1, {0,0,0});
    mesh2.set_node_coordinates(2, {0,0,0});
    mesh2.set_node_coordinates(3, {0,0,0});
    State state2(mesh2);
    std::vector<physics::MaterialState> mat_states2;

    CheckpointReader reader2(ckpt_file);
    bool read_ok = reader2.read(mesh2, state2, &mat_states2);
    CHECK(read_ok, "Full read succeeded");

    // Verify mesh coordinates
    auto c0 = mesh2.get_node_coordinates(0);
    auto c1 = mesh2.get_node_coordinates(1);
    CHECK(near(c0[0], 0.0) && near(c0[1], 0.0) && near(c0[2], 0.0), "Node 0 coords preserved");
    CHECK(near(c1[0], 1.0) && near(c1[1], 0.0) && near(c1[2], 0.0), "Node 1 coords preserved");

    // Verify state fields
    CHECK(near(state2.time(), 0.00123), "Time preserved");
    CHECK(state2.step() == 456, "Step preserved");

    CHECK(near(state2.displacement().at(0, 0), 0.001), "Disp node 0 x preserved");
    CHECK(near(state2.displacement().at(1, 1), -0.002), "Disp node 1 y preserved");
    CHECK(near(state2.velocity().at(0, 0), 10.0), "Vel node 0 x preserved");
    CHECK(near(state2.mass()[1], 2.0), "Mass node 1 preserved");

    // Verify material states
    CHECK(mat_states2.size() == 2, "2 material states read");
    CHECK(near(mat_states2[0].stress[0], 1.0e6), "Mat state 0: σxx preserved");
    CHECK(near(mat_states2[0].plastic_strain, 0.01), "Mat state 0: eps_p preserved");
    CHECK(near(mat_states2[0].temperature, 400.0), "Mat state 0: temperature preserved");
    CHECK(near(mat_states2[1].strain[3], 0.002), "Mat state 1: γxy preserved");
    CHECK(near(mat_states2[1].damage, 0.5), "Mat state 1: damage preserved");
    CHECK(near(mat_states2[1].history[10], 42.0), "Mat state 1: history[10] preserved");

    std::remove(ckpt_file.c_str());
}

// ==========================================================================
// Test 3: Fields-Only Checkpoint
// ==========================================================================
void test_fields_only_checkpoint() {
    std::cout << "\n=== Test 3: Fields-Only Checkpoint ===\n";

    std::string ckpt_file = "/tmp/nxs_test_fields.nxs";

    Mesh mesh = create_test_mesh();
    State state(mesh);
    state.set_time(0.5);
    state.set_step(100);

    state.displacement().at(0, 0) = 1.234;
    state.velocity().at(1, 2) = 5.678;

    // Write only displacement and velocity
    CheckpointWriter writer(ckpt_file);
    bool ok = writer.write_fields_only(state, {"displacement", "velocity"});
    CHECK(ok, "Fields-only write succeeded");

    // Read back
    Mesh mesh2 = create_test_mesh();
    State state2(mesh2);
    CheckpointReader reader(ckpt_file);
    bool rok = reader.read_fields(state2, {"displacement"});
    CHECK(rok, "Selective field read succeeded");
    CHECK(near(state2.displacement().at(0, 0), 1.234), "Displacement preserved");
    CHECK(near(state2.time(), 0.5), "Time from header preserved");

    std::remove(ckpt_file.c_str());
}

// ==========================================================================
// Test 4: Checkpoint Manager (Periodic + Cleanup)
// ==========================================================================
void test_checkpoint_manager() {
    std::cout << "\n=== Test 4: Checkpoint Manager ===\n";

    std::string base = "/tmp/nxs_ckpt_mgr";

    Mesh mesh = create_test_mesh();
    State state(mesh);

    CheckpointManager mgr;
    mgr.configure(base, 10, 2);  // Every 10 steps, keep last 2

    CHECK(!mgr.should_write(5), "Step 5: not time to write");
    CHECK(mgr.should_write(10), "Step 10: time to write");
    CHECK(mgr.should_write(20), "Step 20: time to write");

    // Simulate steps
    for (int step = 0; step <= 30; step += 10) {
        state.set_time(step * 1.0e-3);
        state.set_step(step);
        mgr.maybe_write(step, mesh, state);
    }

    CHECK(mgr.total_checkpoints() == 4, "4 checkpoints written (0, 10, 20, 30)");
    CHECK(mgr.checkpoint_files().size() == 2, "Only 2 files kept (cleanup)");
    CHECK(mgr.total_bytes() > 0, "Total bytes > 0");

    // Latest should be step 30
    auto latest = mgr.find_latest();
    CHECK(!latest.empty(), "Latest checkpoint found");

    // Read latest
    Mesh mesh2(4);
    mesh2.set_node_coordinates(0, {0,0,0});
    mesh2.set_node_coordinates(1, {0,0,0});
    mesh2.set_node_coordinates(2, {0,0,0});
    mesh2.set_node_coordinates(3, {0,0,0});
    State state2(mesh2);
    bool rok = mgr.read_latest(mesh2, state2);
    CHECK(rok, "Read latest checkpoint");
    CHECK(state2.step() == 30, "Latest step = 30");

    // Cleanup
    for (const auto& f : mgr.checkpoint_files()) {
        std::remove(f.c_str());
    }
}

// ==========================================================================
// Test 5: Time History - Nodal Probes
// ==========================================================================
void test_nodal_probes() {
    std::cout << "\n=== Test 5: Nodal Probes ===\n";

    TimeHistoryRecorder rec;
    rec.add_nodal_probe("tip_x", {0}, NodalQuantity::DisplacementX);
    rec.add_nodal_probe("tip_vel", {0, 1}, NodalQuantity::VelocityMag);

    CHECK(rec.num_nodal_probes() == 2, "Two nodal probes");
    CHECK(rec.num_probes() == 2, "Two total probes");

    // Simulate 5 time steps
    const int n = 4;
    for (int step = 0; step < 5; ++step) {
        Real t = step * 0.001;
        Real disp[12] = {};
        Real vel[12] = {};

        disp[0] = t * 10.0;        // Node 0 x-displacement grows linearly
        vel[0] = 10.0;             // Node 0 vx = 10
        vel[3] = 5.0;             // Node 1 vx = 5

        rec.record(t, n, disp, vel);
    }

    CHECK(rec.num_records() == 5, "5 records");

    auto* data_x = rec.get_probe_data("tip_x");
    CHECK(data_x != nullptr, "tip_x data found");
    CHECK(data_x->size() == 5, "tip_x has 5 entries");
    CHECK(near((*data_x)[0][0], 0.0), "t=0: disp_x = 0");
    CHECK(near((*data_x)[4][0], 0.04), "t=4ms: disp_x = 0.04");

    auto* data_vel = rec.get_probe_data("tip_vel");
    CHECK(data_vel != nullptr, "tip_vel data found");
    CHECK(near((*data_vel)[0][0], 10.0), "Node 0 vel mag = 10");
    CHECK(near((*data_vel)[0][1], 5.0), "Node 1 vel mag = 5");
}

// ==========================================================================
// Test 6: Time History - Element Probes
// ==========================================================================
void test_element_probes() {
    std::cout << "\n=== Test 6: Element Probes ===\n";

    TimeHistoryRecorder rec;
    rec.add_element_probe("vm_stress", {0}, ElementQuantity::VonMises);
    rec.add_element_probe("pressure", {0}, ElementQuantity::Pressure);
    rec.add_element_probe("eps_p", {0, 1}, ElementQuantity::EffectivePlasticStrain);

    CHECK(rec.num_element_probes() == 3, "Three element probes");

    // Uniaxial stress: σxx = 200 MPa, others zero
    Real stress[12] = {200.0e6, 0, 0, 0, 0, 0,   // Element 0
                       100.0e6, 0, 0, 0, 0, 0};   // Element 1
    Real eps_p[2] = {0.05, 0.02};

    rec.record(0.001, 0, nullptr, nullptr, nullptr, nullptr,
               2, stress, nullptr, eps_p);

    auto* vm = rec.get_probe_data("vm_stress");
    CHECK(vm != nullptr, "VonMises data found");
    CHECK(near((*vm)[0][0], 200.0e6, 1.0), "σ_vm = σxx for uniaxial");

    auto* pr = rec.get_probe_data("pressure");
    CHECK(pr != nullptr, "Pressure data found");
    CHECK(near((*pr)[0][0], -200.0e6/3.0, 100.0), "p = -σxx/3 for uniaxial");

    auto* ep = rec.get_probe_data("eps_p");
    CHECK(ep != nullptr, "Plastic strain data found");
    CHECK(near((*ep)[0][0], 0.05), "Element 0: eps_p = 0.05");
    CHECK(near((*ep)[0][1], 0.02), "Element 1: eps_p = 0.02");
}

// ==========================================================================
// Test 7: Time History - CSV Output
// ==========================================================================
void test_csv_output() {
    std::cout << "\n=== Test 7: CSV Output ===\n";

    TimeHistoryRecorder rec;
    rec.add_nodal_probe("disp", {0}, NodalQuantity::DisplacementX);

    Real disp[3] = {0.001, 0, 0};
    rec.record(0.0, 1, disp);
    disp[0] = 0.002;
    rec.record(0.001, 1, disp);

    std::string csv_base = "/tmp/nxs_test_history";
    bool ok = rec.write_csv(csv_base);
    CHECK(ok, "CSV write succeeded");

    // Verify file exists and has content
    std::string csv_file = csv_base + "_disp.csv";
    std::ifstream f(csv_file);
    CHECK(f.is_open(), "CSV file created");

    std::string line;
    std::getline(f, line);
    CHECK(line.find("time") != std::string::npos, "CSV header contains 'time'");

    std::getline(f, line);
    CHECK(!line.empty(), "CSV has data row");

    f.close();
    std::remove(csv_file.c_str());
}

// ==========================================================================
// Test 8: Time History - Recording Interval
// ==========================================================================
void test_recording_interval() {
    std::cout << "\n=== Test 8: Recording Interval ===\n";

    TimeHistoryRecorder rec;
    rec.set_interval(3);  // Record every 3rd call
    rec.add_nodal_probe("test", {0}, NodalQuantity::DisplacementX);

    Real disp[3] = {};
    for (int i = 0; i < 10; ++i) {
        disp[0] = i * 0.1;
        rec.record(i * 0.001, 1, disp);
    }

    // Calls 1, 4, 7, 10 should be recorded (1-indexed: first call + every 3rd)
    CHECK(rec.num_records() == 4, "4 records from 10 calls with interval=3");
}

// ==========================================================================
// Test 9: Energy Tracker
// ==========================================================================
void test_energy_tracker() {
    std::cout << "\n=== Test 9: Energy Tracker ===\n";

    EnergyTracker tracker;

    // Initial state: KE=100, IE=0
    tracker.record(0.0, 100.0, 0.0);
    CHECK(near(tracker.latest().total, 100.0), "Initial total = 100");
    CHECK(near(tracker.latest().balance_error, 0.0), "Initial error = 0");

    // Energy conversion: KE→IE (should conserve)
    tracker.record(0.001, 80.0, 20.0);
    CHECK(near(tracker.latest().total, 100.0), "Conserved: total still 100");
    CHECK(near(tracker.latest().balance_error, 0.0), "No error");

    // Small dissipation via contact
    tracker.record(0.002, 60.0, 35.0, 0.0, 3.0);
    Real total = 60.0 + 35.0 + 3.0;
    CHECK(near(tracker.latest().total, total), "Total includes contact energy");

    // Energy balance error
    Real err = (total - 100.0) / 100.0;
    CHECK(near(tracker.latest().balance_error, err, 1e-6), "Balance error computed");

    CHECK(tracker.num_records() == 3, "3 energy records");
    CHECK(tracker.max_balance_error() < 0.05, "Max error < 5%");

    // CSV output
    std::string csv = "/tmp/nxs_energy.csv";
    bool ok = tracker.write_csv(csv);
    CHECK(ok, "Energy CSV written");
    std::remove(csv.c_str());
}

// ==========================================================================
// Test 10: Animation Writer - ASCII
// ==========================================================================
void test_animation_ascii() {
    std::cout << "\n=== Test 10: Animation Writer ASCII ===\n";

    std::string base = "/tmp/nxs_anim_ascii";

    AnimationWriter anim(base);
    anim.set_format(VTKFormat::ASCII);
    anim.add_nodal_field("displacement", 3);
    anim.add_cell_field("stress", 1);

    // Simple 4-node quad mesh
    Real coords[12] = {0,0,0, 1,0,0, 1,1,0, 0,1,0};
    Index conn[4] = {0, 1, 2, 3};
    Real disp[12] = {0,0,0, 0.01,0,0, 0.01,0.01,0, 0,0.01,0};
    Real stress[1] = {1.0e6};

    std::map<std::string, const Real*> fields;
    fields["displacement"] = disp;
    fields["stress"] = stress;

    bool ok = anim.write_frame(0, 0.0, 4, coords, 1, conn, 4, 9, fields);
    CHECK(ok, "ASCII frame 0 written");

    // Move nodes slightly for frame 1
    for (int i = 0; i < 12; ++i) disp[i] *= 2.0;
    stress[0] = 2.0e6;
    ok = anim.write_frame(1, 0.001, 4, coords, 1, conn, 4, 9, fields);
    CHECK(ok, "ASCII frame 1 written");

    CHECK(anim.frame_count() == 2, "2 frames written");

    // Write PVD
    ok = anim.finalize();
    CHECK(ok, "PVD file written");

    // Verify PVD exists
    std::string pvd = base + ".pvd";
    std::ifstream f(pvd);
    CHECK(f.is_open(), "PVD file exists");
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    CHECK(content.find("Collection") != std::string::npos, "PVD contains Collection tag");
    CHECK(content.find("timestep") != std::string::npos, "PVD contains timestep entries");
    f.close();

    // Verify VTK frame file
    std::string vtk0 = base + "_000000.vtk";
    std::ifstream fv(vtk0);
    CHECK(fv.is_open(), "VTK frame file exists");
    std::string vtk_content((std::istreambuf_iterator<char>(fv)),
                              std::istreambuf_iterator<char>());
    CHECK(vtk_content.find("UNSTRUCTURED_GRID") != std::string::npos, "VTK is unstructured grid");
    CHECK(vtk_content.find("POINTS 4") != std::string::npos, "VTK has 4 points");
    CHECK(vtk_content.find("displacement") != std::string::npos, "VTK has displacement field");
    CHECK(vtk_content.find("stress") != std::string::npos, "VTK has stress field");
    fv.close();

    // Cleanup
    std::remove(pvd.c_str());
    std::remove(vtk0.c_str());
    std::remove((base + "_000001.vtk").c_str());
}

// ==========================================================================
// Test 11: Animation Writer - Binary
// ==========================================================================
void test_animation_binary() {
    std::cout << "\n=== Test 11: Animation Writer Binary ===\n";

    std::string base = "/tmp/nxs_anim_binary";

    AnimationWriter anim(base);
    anim.set_format(VTKFormat::Binary);
    anim.add_nodal_field("displacement", 3);

    Real coords[12] = {0,0,0, 1,0,0, 1,1,0, 0,1,0};
    Index conn[4] = {0, 1, 2, 3};
    Real disp[12] = {0.01, 0, 0, 0.02, 0, 0, 0.02, 0.01, 0, 0.01, 0.01, 0};

    std::map<std::string, const Real*> fields;
    fields["displacement"] = disp;

    bool ok = anim.write_frame(0, 0.0, 4, coords, 1, conn, 4, 9, fields);
    CHECK(ok, "Binary frame written");

    // Binary VTK file should exist and be valid
    std::string vtk = base + "_000000.vtk";
    std::ifstream f(vtk, std::ios::binary);
    CHECK(f.is_open(), "Binary VTK file exists");

    // Read first few lines to verify header
    std::string line;
    std::getline(f, line);
    CHECK(line.find("vtk DataFile") != std::string::npos, "VTK header present");
    std::getline(f, line);  // description
    std::getline(f, line);  // format
    CHECK(line.find("BINARY") != std::string::npos, "Format is BINARY");
    f.close();

    // Binary should be smaller than or equal to ASCII for small datasets
    // (overhead from header means it may be slightly larger for tiny meshes)
    CHECK(anim.frame_count() == 1, "1 frame written");

    anim.finalize();

    // Cleanup
    std::remove(vtk.c_str());
    std::remove((base + ".pvd").c_str());
}

// ==========================================================================
// Test 12: Animation Output Interval
// ==========================================================================
void test_animation_interval() {
    std::cout << "\n=== Test 12: Animation Output Interval ===\n";

    std::string base = "/tmp/nxs_anim_interval";

    AnimationWriter anim(base);
    anim.set_format(VTKFormat::ASCII);
    anim.set_output_interval(5);
    anim.add_nodal_field("displacement", 3);

    Real coords[12] = {0,0,0, 1,0,0, 1,1,0, 0,1,0};
    Index conn[4] = {0, 1, 2, 3};
    Real disp[12] = {};
    std::map<std::string, const Real*> fields = {{"displacement", disp}};

    for (int step = 0; step <= 20; ++step) {
        anim.write_frame(step, step * 0.001, 4, coords, 1, conn, 4, 9, fields);
    }

    // Steps 0, 5, 10, 15, 20 → 5 frames
    CHECK(anim.frame_count() == 5, "5 frames with interval=5 over 21 steps");

    anim.finalize();

    // Cleanup
    std::remove((base + ".pvd").c_str());
    for (int i = 0; i < 5; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s_%06d.vtk", base.c_str(), i);
        std::remove(buf);
    }
}

// ==========================================================================
// Test 13: MaterialStateData Serialization
// ==========================================================================
void test_material_state_serialization() {
    std::cout << "\n=== Test 13: MaterialState Serialization ===\n";

    physics::MaterialState ms;
    ms.stress[0] = 1.0e8;
    ms.stress[1] = -5.0e7;
    ms.strain[3] = 0.001;
    ms.plastic_strain = 0.02;
    ms.temperature = 500.0;
    ms.damage = 0.3;
    ms.history[0] = 1.0;
    ms.history[15] = 2.0;
    ms.history[31] = 3.0;
    ms.effective_strain_rate = 100.0;
    ms.dt = 1.0e-6;

    MaterialStateData msd;
    msd.from_material_state(ms);

    CHECK(near(msd.stress[0], 1.0e8), "Stress transferred");
    CHECK(near(msd.history[31], 3.0), "History[31] transferred");

    physics::MaterialState ms2;
    msd.to_material_state(ms2);

    CHECK(near(ms2.stress[0], 1.0e8), "Stress round-trip");
    CHECK(near(ms2.stress[1], -5.0e7), "Stress[1] round-trip");
    CHECK(near(ms2.strain[3], 0.001), "Strain round-trip");
    CHECK(near(ms2.plastic_strain, 0.02), "Plastic strain round-trip");
    CHECK(near(ms2.temperature, 500.0), "Temperature round-trip");
    CHECK(near(ms2.damage, 0.3), "Damage round-trip");
    CHECK(near(ms2.history[0], 1.0), "History[0] round-trip");
    CHECK(near(ms2.history[15], 2.0), "History[15] round-trip");
    CHECK(near(ms2.history[31], 3.0), "History[31] round-trip");
    CHECK(near(ms2.effective_strain_rate, 100.0), "Strain rate round-trip");
    CHECK(near(ms2.dt, 1.0e-6), "dt round-trip");
}

// ==========================================================================
// Test 14: Quantity String Conversion
// ==========================================================================
void test_quantity_strings() {
    std::cout << "\n=== Test 14: Quantity Strings ===\n";

    CHECK(std::string(to_string(NodalQuantity::DisplacementX)) == "disp_x",
          "NodalQuantity::DisplacementX → 'disp_x'");
    CHECK(std::string(to_string(NodalQuantity::VelocityMag)) == "vel_mag",
          "NodalQuantity::VelocityMag → 'vel_mag'");
    CHECK(std::string(to_string(ElementQuantity::VonMises)) == "von_mises",
          "ElementQuantity::VonMises → 'von_mises'");
    CHECK(std::string(to_string(ElementQuantity::EffectivePlasticStrain)) == "eff_plastic_strain",
          "ElementQuantity::EffectivePlasticStrain → 'eff_plastic_strain'");
}

// ==========================================================================
// Main
// ==========================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "NexusSim Wave 6: Restart & Output Test\n";
    std::cout << "========================================\n";

    test_checkpoint_header();
    test_checkpoint_roundtrip();
    test_fields_only_checkpoint();
    test_checkpoint_manager();
    test_nodal_probes();
    test_element_probes();
    test_csv_output();
    test_recording_interval();
    test_energy_tracker();
    test_animation_ascii();
    test_animation_binary();
    test_animation_interval();
    test_material_state_serialization();
    test_quantity_strings();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    std::cout << "========================================\n";

    return (tests_failed > 0) ? 1 : 0;
}
