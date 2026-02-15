/**
 * @file enhanced_output_test.cpp
 * @brief Comprehensive test for Wave 6 Enhanced: Extended Checkpoint, Part Energy,
 *        Interface Forces, Cross-Section Forces, Result Database
 */

#include <nexussim/io/extended_checkpoint.hpp>
#include <nexussim/io/part_energy.hpp>
#include <nexussim/io/interface_force_output.hpp>
#include <nexussim/io/cross_section_force.hpp>
#include <nexussim/io/result_database.hpp>
#include <iostream>
#include <cmath>
#include <cstdio>

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

// Helper: Create a simple test mesh (4 nodes, 1 quad element)
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
// Test 1: FailureState Serialization
// ==========================================================================
void test_failure_state_serialization() {
    std::cout << "\n=== Test 1: FailureState Serialization ===\n";

    physics::failure::FailureState fs;
    fs.damage = 0.75;
    fs.failed = true;
    fs.failure_mode = 3;
    fs.history[0] = 1.5;
    fs.history[15] = 99.9;

    FailureStateData fsd;
    fsd.from_failure_state(fs);

    physics::failure::FailureState fs2;
    fsd.to_failure_state(fs2);

    CHECK(near(fs2.damage, 0.75), "Damage round-trip");
    CHECK(fs2.failed == true, "Failed flag round-trip");
    CHECK(fs2.failure_mode == 3, "Failure mode round-trip");
    CHECK(near(fs2.history[0], 1.5), "History[0] round-trip");
    CHECK(near(fs2.history[15], 99.9), "History[15] round-trip");
}

// ==========================================================================
// Test 2: RigidBody State Serialization
// ==========================================================================
void test_rigid_body_state_serialization() {
    std::cout << "\n=== Test 2: RigidBody State Serialization ===\n";

    fem::RigidBody rb(42, "test_body");
    rb.properties().com[0] = 1.0;
    rb.properties().com[1] = 2.0;
    rb.properties().com[2] = 3.0;
    rb.velocity()[0] = 10.0;
    rb.velocity()[1] = 20.0;
    rb.velocity()[2] = 30.0;
    rb.angular_velocity()[0] = 0.1;
    rb.angular_velocity()[1] = 0.2;
    rb.angular_velocity()[2] = 0.3;

    RigidBodyStateData rbd;
    rbd.from_rigid_body(rb);

    CHECK(rbd.id == 42, "ID captured");
    CHECK(near(rbd.com[0], 1.0), "COM[0] captured");
    CHECK(near(rbd.velocity[1], 20.0), "Velocity[1] captured");
    CHECK(near(rbd.angular_velocity[2], 0.3), "AngVel[2] captured");

    // Quaternion should be identity (no rotation applied)
    CHECK(near(rbd.orientation[0], 1.0), "Quaternion w = 1 (identity)");
    CHECK(near(rbd.orientation[1], 0.0), "Quaternion x = 0 (identity)");
}

// ==========================================================================
// Test 3: ContactState Serialization
// ==========================================================================
void test_contact_state_serialization() {
    std::cout << "\n=== Test 3: ContactState Serialization ===\n";

    fem::ContactPair cp;
    cp.slave_node = 10;
    cp.master_face = 20;
    cp.penetration_depth = 0.005;
    cp.normal[0] = 0.0; cp.normal[1] = 0.0; cp.normal[2] = 1.0;

    ContactStateData csd;
    csd.from_contact_pair(cp, true, false);
    csd.tangent_slip[0] = 0.001;

    CHECK(near(csd.gap, 0.005), "Gap captured");
    CHECK(near(csd.normal[2], 1.0), "Normal[2] captured");
    CHECK(near(csd.tangent_slip[0], 0.001), "Tangent slip captured");
    CHECK(csd.active == 1, "Active flag");
    CHECK(csd.sticking == 0, "Not sticking");
}

// ==========================================================================
// Test 4: TiedPair Serialization
// ==========================================================================
void test_tied_pair_serialization() {
    std::cout << "\n=== Test 4: TiedPair Serialization ===\n";

    fem::TiedPair tp;
    tp.slave_node = 5;
    tp.master_nodes[0] = 10; tp.master_nodes[1] = 11;
    tp.master_nodes[2] = 12; tp.master_nodes[3] = 13;
    tp.phi[0] = 0.25; tp.phi[1] = 0.25;
    tp.phi[2] = 0.25; tp.phi[3] = 0.25;
    tp.num_master_nodes = 4;
    tp.active = true;
    tp.accumulated_force = 500.0;
    tp.gap_initial[0] = 0.01;

    TiedPairData tpd;
    tpd.from_tied_pair(tp);

    fem::TiedPair tp2;
    tpd.to_tied_pair(tp2);

    CHECK(near(tp2.phi[0], 0.25), "Phi[0] round-trip");
    CHECK(tp2.active == true, "Active round-trip");
    CHECK(near(tp2.accumulated_force, 500.0), "Accumulated force round-trip");
    CHECK(near(tp2.gap_initial[0], 0.01), "Gap initial round-trip");
    CHECK(tp2.num_master_nodes == 4, "Num master nodes round-trip");
}

// ==========================================================================
// Test 5: Extended Checkpoint Write/Read
// ==========================================================================
void test_extended_checkpoint() {
    std::cout << "\n=== Test 5: Extended Checkpoint Write/Read ===\n";

    std::string ckpt_file = "/tmp/nxs_ext_checkpoint.nxs";

    // Setup mesh and state
    Mesh mesh = create_test_mesh();
    State state(mesh);
    state.set_time(0.005);
    state.set_step(100);
    state.displacement().at(0, 0) = 0.01;

    // Material states
    std::vector<physics::MaterialState> mat_states(1);
    mat_states[0].stress[0] = 1.0e6;

    // Failure states
    std::vector<physics::failure::FailureState> fail_states(2);
    fail_states[0].damage = 0.3;
    fail_states[0].failed = false;
    fail_states[1].damage = 1.0;
    fail_states[1].failed = true;
    fail_states[1].failure_mode = 2;

    // Rigid bodies
    fem::RigidBody rb1(1, "rb1");
    rb1.velocity()[0] = 5.0;
    rb1.properties().com[0] = 1.0;
    std::vector<fem::RigidBody*> rbs = {&rb1};

    // Contact pairs
    std::vector<fem::ContactPair> contacts(1);
    contacts[0].slave_node = 3;
    contacts[0].master_face = 7;
    contacts[0].penetration_depth = 0.002;
    contacts[0].normal[2] = 1.0;

    // Tied pairs
    std::vector<fem::TiedPair> tied(1);
    tied[0].slave_node = 1;
    tied[0].master_nodes[0] = 10;
    tied[0].phi[0] = 1.0;
    tied[0].num_master_nodes = 1;
    tied[0].active = true;

    // Write
    ExtendedCheckpointWriter writer(ckpt_file);
    bool wok = writer.write(mesh, state, mat_states, fail_states, rbs, contacts, tied);
    CHECK(wok, "Extended checkpoint write succeeded");
    CHECK(writer.bytes_written() > 64, "File has data beyond header");

    // Read back
    Mesh mesh2(4);
    mesh2.set_node_coordinates(0, {0,0,0});
    mesh2.set_node_coordinates(1, {0,0,0});
    mesh2.set_node_coordinates(2, {0,0,0});
    mesh2.set_node_coordinates(3, {0,0,0});
    State state2(mesh2);
    std::vector<physics::MaterialState> mat_states2;
    std::vector<physics::failure::FailureState> fail_states2;
    std::vector<RigidBodyStateData> rb_states2;
    std::vector<ContactStateData> contact_states2;
    std::vector<fem::TiedPair> tied2;

    ExtendedCheckpointReader reader(ckpt_file);
    bool rok = reader.read(mesh2, state2, &mat_states2, &fail_states2,
                            &rb_states2, &contact_states2, &tied2);
    CHECK(rok, "Extended checkpoint read succeeded");
    CHECK(near(state2.time(), 0.005), "Time preserved");
    CHECK(state2.step() == 100, "Step preserved");
    CHECK(near(state2.displacement().at(0, 0), 0.01), "Displacement preserved");

    // Verify material states
    CHECK(mat_states2.size() == 1, "1 material state read");
    CHECK(near(mat_states2[0].stress[0], 1.0e6), "Material stress preserved");

    // Verify failure states
    CHECK(fail_states2.size() == 2, "2 failure states read");
    CHECK(near(fail_states2[0].damage, 0.3), "Failure state 0 damage preserved");
    CHECK(fail_states2[1].failed == true, "Failure state 1 failed flag preserved");

    // Verify rigid body states
    CHECK(rb_states2.size() == 1, "1 rigid body state read");
    CHECK(near(rb_states2[0].velocity[0], 5.0), "RB velocity preserved");

    std::remove(ckpt_file.c_str());
}

// ==========================================================================
// Test 6: Backward Compatibility (V1 checkpoint readable by ExtendedReader)
// ==========================================================================
void test_backward_compatibility() {
    std::cout << "\n=== Test 6: Backward Compatibility ===\n";

    std::string ckpt_file = "/tmp/nxs_v1_compat.nxs";

    // Write a standard v1 checkpoint (no extended sections)
    Mesh mesh = create_test_mesh();
    State state(mesh);
    state.set_time(0.01);
    state.set_step(50);
    state.displacement().at(1, 0) = 0.5;

    CheckpointWriter writer(ckpt_file);
    bool wok = writer.write(mesh, state);
    CHECK(wok, "V1 checkpoint write succeeded");

    // Read with extended reader
    Mesh mesh2(4);
    mesh2.set_node_coordinates(0, {0,0,0});
    mesh2.set_node_coordinates(1, {0,0,0});
    mesh2.set_node_coordinates(2, {0,0,0});
    mesh2.set_node_coordinates(3, {0,0,0});
    State state2(mesh2);

    ExtendedCheckpointReader reader(ckpt_file);
    bool rok = reader.read(mesh2, state2);
    CHECK(rok, "V1 file readable by ExtendedReader");
    CHECK(near(state2.time(), 0.01), "Time preserved from V1 file");
    CHECK(!reader.has_section(ExtendedSectionType::FailureState),
          "has_section(Failure) = false for V1");
    CHECK(!reader.has_section(ExtendedSectionType::RigidBody),
          "has_section(RigidBody) = false for V1");

    std::remove(ckpt_file.c_str());
}

// ==========================================================================
// Test 7: Per-Part Energy Tracking
// ==========================================================================
void test_part_energy() {
    std::cout << "\n=== Test 7: Per-Part Energy Tracking ===\n";

    PartEnergyTracker tracker;

    // Two parts: part 1 has elements {0, 1}, part 2 has element {2}
    tracker.register_part(1, "bumper", {0, 1});
    tracker.register_part(2, "frame", {2});

    CHECK(tracker.num_parts() == 2, "2 parts registered");

    // Time step 1
    Real ke1[3] = {100.0, 200.0, 50.0};  // per-element kinetic energy
    Real ie1[3] = {10.0, 20.0, 5.0};     // per-element internal energy
    tracker.record(0.001, 3, ke1, ie1);

    // Time step 2
    Real ke2[3] = {80.0, 160.0, 60.0};
    Real ie2[3] = {30.0, 60.0, 10.0};
    tracker.record(0.002, 3, ke2, ie2);

    // Time step 3
    Real ke3[3] = {50.0, 100.0, 70.0};
    Real ie3[3] = {60.0, 100.0, 15.0};
    tracker.record(0.003, 3, ke3, ie3);

    CHECK(tracker.num_records() == 3, "3 time records");

    auto* p1 = tracker.get_part_records(1);
    CHECK(p1 != nullptr, "Part 1 records found");
    CHECK(p1->size() == 3, "Part 1 has 3 records");

    // Part 1 at t=0.001: KE = 100+200 = 300, IE = 10+20 = 30
    CHECK(near((*p1)[0].kinetic_energy, 300.0), "Part 1 KE at t1 = 300");
    CHECK(near((*p1)[0].internal_energy, 30.0), "Part 1 IE at t1 = 30");
    CHECK(near((*p1)[0].total_energy, 330.0), "Part 1 total at t1 = 330");

    // Part 2 at t=0.001: KE = 50, IE = 5
    auto* p2 = tracker.get_part_records(2);
    CHECK(near((*p2)[0].kinetic_energy, 50.0), "Part 2 KE at t1 = 50");
    CHECK(near((*p2)[0].internal_energy, 5.0), "Part 2 IE at t1 = 5");

    // CSV output
    std::string csv = "/tmp/nxs_part_energy.csv";
    bool ok = tracker.write_csv(csv);
    CHECK(ok, "Part energy CSV written");
    std::remove(csv.c_str());
}

// ==========================================================================
// Test 8: Interface Force Recording (Contact)
// ==========================================================================
void test_interface_force_contact() {
    std::cout << "\n=== Test 8: Interface Force Recording (Contact) ===\n";

    InterfaceForceTracker tracker;
    tracker.register_contact_interface(1, "bumper_hood");

    // Create mock contact pairs
    std::vector<fem::ContactPair> pairs(3);

    // Pair 0: penetrating along z
    pairs[0].penetration_depth = 0.01;
    pairs[0].normal[0] = 0.0; pairs[0].normal[1] = 0.0; pairs[0].normal[2] = 1.0;

    // Pair 1: penetrating along z
    pairs[1].penetration_depth = 0.02;
    pairs[1].normal[0] = 0.0; pairs[1].normal[1] = 0.0; pairs[1].normal[2] = 1.0;

    // Pair 2: not penetrating
    pairs[2].penetration_depth = 0.0;
    pairs[2].normal[0] = 0.0; pairs[2].normal[1] = 0.0; pairs[2].normal[2] = 1.0;

    Real k = 1.0e8; // penalty stiffness
    tracker.record_contact_forces(0.001, 1, pairs, k);

    auto* records = tracker.get_records(1);
    CHECK(records != nullptr, "Contact records found");
    CHECK(records->size() == 1, "1 contact record");

    // Expected: F_z = k * (0.01 + 0.02) = 1e8 * 0.03 = 3e6
    CHECK(near((*records)[0].normal_force[2], 3.0e6, 1.0), "Fz = 3e6");
    CHECK(near((*records)[0].normal_force[0], 0.0), "Fx = 0");
    CHECK(near((*records)[0].normal_force[1], 0.0), "Fy = 0");
    CHECK(near((*records)[0].total_force_magnitude, 3.0e6, 1.0), "Force mag = 3e6");
    CHECK((*records)[0].num_active_pairs == 2, "2 active pairs");

    // CSV output
    std::string csv = "/tmp/nxs_interface_force.csv";
    bool ok = tracker.write_csv(csv);
    CHECK(ok, "Interface force CSV written");
    std::remove(csv.c_str());
}

// ==========================================================================
// Test 9: Rigid Wall Force Recording
// ==========================================================================
void test_wall_force_recording() {
    std::cout << "\n=== Test 9: Rigid Wall Force Recording ===\n";

    InterfaceForceTracker tracker;
    tracker.register_wall_interface(10, "ground_plane");

    fem::RigidWallContact::WallStats stats;
    stats.active_contacts = 5;
    stats.max_penetration = 0.003;
    stats.total_normal_force = 1.5e6;

    Real wall_normal[3] = {0.0, 0.0, 1.0};
    tracker.record_wall_forces(0.002, 10, stats, wall_normal);

    auto* records = tracker.get_records(10);
    CHECK(records != nullptr, "Wall records found");
    CHECK(near((*records)[0].total_force_magnitude, 1.5e6, 1.0), "Wall force mag = 1.5e6");
    CHECK(near((*records)[0].normal_force[2], 1.5e6, 1.0), "Wall Fz = 1.5e6");
    CHECK((*records)[0].num_active_pairs == 5, "5 active wall contacts");

    // Verify CSV generation
    std::string csv = "/tmp/nxs_wall_force.csv";
    CHECK(tracker.write_csv(csv), "Wall force CSV written");
    std::remove(csv.c_str());
}

// ==========================================================================
// Test 10: Cross-Section Force
// ==========================================================================
void test_cross_section_force() {
    std::cout << "\n=== Test 10: Cross-Section Force ===\n";

    CrossSectionForceTracker tracker;

    // Define cross-section: cut plane perpendicular to x-axis at x=0.5
    Real origin[3] = {0.5, 0.0, 0.0};
    Real normal[3] = {1.0, 0.0, 0.0};
    tracker.add_section(1, "mid_section", origin, normal, {0, 1});

    // Uniaxial stress: σ_xx = 100 MPa, all others zero
    // 2 elements, each with same stress
    Real stresses[12] = {
        100.0e6, 0, 0, 0, 0, 0,   // Element 0: σxx=100MPa
        100.0e6, 0, 0, 0, 0, 0    // Element 1: σxx=100MPa
    };
    Real areas[2] = {0.01, 0.01};  // 0.01 m² each
    Real centroids[6] = {
        0.25, 0.25, 0.0,   // Element 0 centroid
        0.75, 0.25, 0.0    // Element 1 centroid
    };

    tracker.record(0.001, stresses, areas, centroids);

    auto* records = tracker.get_records(1);
    CHECK(records != nullptr, "Cross-section records found");
    CHECK(records->size() == 1, "1 cross-section record");

    // F_x = σ_xx * n_x * (A0 + A1) = 100e6 * 1.0 * 0.02 = 2e6 N
    CHECK(near((*records)[0].force[0], 2.0e6, 1.0), "Fx = 2 MN (traction σ·n)");
    CHECK(near((*records)[0].force[1], 0.0, 100.0), "Fy ≈ 0");
    CHECK(near((*records)[0].force[2], 0.0, 100.0), "Fz ≈ 0");
    CHECK(near((*records)[0].force_magnitude, 2.0e6, 1.0), "Force magnitude = 2 MN");

    // Moments: M = r × F
    // Element 0: r = (0.25-0.5, 0.25-0, 0-0) = (-0.25, 0.25, 0)
    //   F0 = (100e6*0.01, 0, 0) = (1e6, 0, 0)
    //   M0 = r × F = (0.25*0 - 0*0, 0*1e6 - (-0.25)*0, (-0.25)*0 - 0.25*1e6)
    //      = (0, 0, -0.25e6)
    // Element 1: r = (0.75-0.5, 0.25-0, 0-0) = (0.25, 0.25, 0)
    //   F1 = (1e6, 0, 0)
    //   M1 = (0, 0, 0.25*0 - 0.25*1e6) = (0, 0, -0.25e6)
    // Total moment_z = -0.5e6
    CHECK(near((*records)[0].moment[2], -0.5e6, 100.0), "Mz from cross product");
    CHECK(near((*records)[0].moment[0], 0.0, 100.0), "Mx ≈ 0");
    CHECK(near((*records)[0].moment[1], 0.0, 100.0), "My ≈ 0");
}

// ==========================================================================
// Test 11: Result Database Write/Read
// ==========================================================================
void test_result_database() {
    std::cout << "\n=== Test 11: Result Database Write/Read ===\n";

    std::string db_file = "/tmp/nxs_result.nxr";

    // Write 5 frames
    {
        ResultDatabaseWriter writer(db_file);
        writer.add_nodal_field("displacement", 3);
        writer.add_cell_field("stress", 6);

        bool ok = writer.open(4, 1);
        CHECK(ok, "DB writer opened");

        for (int frame = 0; frame < 5; ++frame) {
            Real t = frame * 0.001;

            // Nodal displacement: 4 nodes × 3 components
            std::vector<Real> disp(12, 0.0);
            disp[0] = t * 10.0;  // Node 0 x grows with time

            // Cell stress: 1 element × 6 components
            std::vector<Real> stress(6, 0.0);
            stress[0] = (frame + 1) * 1.0e6;  // σ_xx grows with frame

            std::vector<const Real*> field_data = {disp.data(), stress.data()};
            bool wok = writer.write_frame(t, frame, field_data);
            CHECK(wok, std::string("Frame ") + std::to_string(frame) + " written");
        }

        writer.finalize();
        CHECK(writer.is_finalized(), "DB finalized");
        CHECK(writer.num_frames() == 5, "5 frames written");
    }

    // Read back
    {
        ResultDatabaseReader reader(db_file);
        bool ok = reader.open();
        CHECK(ok, "DB reader opened");
        CHECK(reader.num_frames() == 5, "Reader sees 5 frames");
        CHECK(reader.num_nodes() == 4, "Reader sees 4 nodes");
        CHECK(reader.num_cells() == 1, "Reader sees 1 cell");

        // Read frame 3 (random access)
        double time;
        int64_t step;
        std::vector<std::vector<Real>> field_data;
        bool rok = reader.read_frame(3, time, step, field_data);
        CHECK(rok, "Frame 3 read succeeded");
        CHECK(near(time, 0.003), "Frame 3 time = 0.003");
        CHECK(step == 3, "Frame 3 step = 3");

        // Check displacement field (field 0)
        CHECK(field_data.size() == 2, "2 fields read");
        CHECK(near(field_data[0][0], 0.003 * 10.0), "Frame 3 disp[0] = 0.03");

        // Check stress field (field 1)
        CHECK(near(field_data[1][0], 4.0e6), "Frame 3 stress_xx = 4e6");

        // Read specific field from frame 0
        std::vector<Real> stress_data;
        bool fok = reader.read_field(0, "stress", stress_data);
        CHECK(fok, "Read specific field from frame 0");
        CHECK(near(stress_data[0], 1.0e6), "Frame 0 stress_xx = 1e6");

        // Verify TOC
        CHECK(reader.toc().size() == 5, "TOC has 5 entries");
        CHECK(near(reader.frame_time(0), 0.0), "TOC frame 0 time = 0");
        CHECK(near(reader.frame_time(4), 0.004), "TOC frame 4 time = 0.004");

        reader.close();
    }

    std::remove(db_file.c_str());
}

// ==========================================================================
// Test 12: Result Database Efficiency
// ==========================================================================
void test_result_database_efficiency() {
    std::cout << "\n=== Test 12: Result Database Efficiency ===\n";

    std::string db_file = "/tmp/nxs_result_eff.nxr";

    {
        ResultDatabaseWriter writer(db_file);
        writer.add_nodal_field("velocity", 3);

        bool ok = writer.open(10, 0);
        CHECK(ok, "Efficiency DB opened");

        for (int i = 0; i < 3; ++i) {
            std::vector<Real> vel(30, static_cast<Real>(i));
            std::vector<const Real*> data = {vel.data()};
            writer.write_frame(i * 0.1, i, data);
        }

        writer.finalize();
        CHECK(writer.num_frames() == 3, "3 frames in efficiency test");
    }

    // Verify file exists and has content
    std::ifstream f(db_file, std::ios::binary | std::ios::ate);
    CHECK(f.is_open(), "DB file exists");
    auto file_size = f.tellg();
    CHECK(file_size > 0, "DB file has content");
    f.close();

    // Re-open and verify
    ResultDatabaseReader reader(db_file);
    bool ok = reader.open();
    CHECK(ok, "Reader re-opened");
    CHECK(reader.num_frames() == 3, "Num frames correct on re-read");
    reader.close();

    std::remove(db_file.c_str());
}

// ==========================================================================
// Main
// ==========================================================================
int main() {
    std::cout << "================================================\n";
    std::cout << "NexusSim Wave 6 Enhanced: Output & Checkpoint Test\n";
    std::cout << "================================================\n";

    test_failure_state_serialization();
    test_rigid_body_state_serialization();
    test_contact_state_serialization();
    test_tied_pair_serialization();
    test_extended_checkpoint();
    test_backward_compatibility();
    test_part_energy();
    test_interface_force_contact();
    test_wall_force_recording();
    test_cross_section_force();
    test_result_database();
    test_result_database_efficiency();

    std::cout << "\n================================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    std::cout << "================================================\n";

    return (tests_failed > 0) ? 1 : 0;
}
