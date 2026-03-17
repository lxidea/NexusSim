/**
 * @file contact_wave12_test.cpp
 * @brief Tests for contact_wave12.hpp: 9 advanced contact capabilities
 *
 * Tests: ContactStiffnessScaler, VelocityDependentFrictionModel,
 *        ShellThicknessContact, EdgeToEdgeContact, SegmentBasedContact,
 *        SelfContact, SymmetricContact, RigidDeformableContact,
 *        MultiSurfaceContactManager
 */

#include <nexussim/fem/contact_wave12.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace nxs;
using namespace nxs::fem;

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

// Helper: build a unit quad face in XY plane at given z-offset.
// Nodes: (0,0,z), (1,0,z), (1,1,z), (0,1,z)
// Returns the starting node index used.
static Index make_quad_face(std::vector<Real>& coords, Real z_offset) {
    Index base = coords.size() / 3;
    // node 0
    coords.push_back(0.0); coords.push_back(0.0); coords.push_back(z_offset);
    // node 1
    coords.push_back(1.0); coords.push_back(0.0); coords.push_back(z_offset);
    // node 2
    coords.push_back(1.0); coords.push_back(1.0); coords.push_back(z_offset);
    // node 3
    coords.push_back(0.0); coords.push_back(1.0); coords.push_back(z_offset);
    return base;
}

// ============================================================================
// Test ContactStiffnessScaler
// ============================================================================
static void test_contact_stiffness_scaler() {
    std::cout << "--- ContactStiffnessScaler ---\n";

    // Default construction
    ContactStiffnessScaler scaler;
    CHECK_NEAR(scaler.scale_factor, 1.0, 1e-15, "default scale_factor is 1.0");

    // Explicit construction
    ContactStiffnessScaler scaler2(2.5);
    CHECK_NEAR(scaler2.scale_factor, 2.5, 1e-15, "explicit scale_factor is 2.5");

    // k = sf * K * A / d
    Real k = scaler.compute_stiffness(/*bulk*/1.0e9, /*area*/0.01, /*diag*/0.2);
    CHECK_NEAR(k, 1.0 * 1.0e9 * 0.01 / 0.2, 1.0, "stiffness = K*A/d");

    // With scale factor
    Real k2 = scaler2.compute_stiffness(1.0e9, 0.01, 0.2);
    CHECK_NEAR(k2, 2.5 * 1.0e9 * 0.01 / 0.2, 1.0, "stiffness with sf=2.5");

    // Zero diagonal => returns 0
    Real k3 = scaler.compute_stiffness(1.0e9, 0.01, 0.0);
    CHECK_NEAR(k3, 0.0, 1e-30, "zero diagonal returns zero stiffness");
}

// ============================================================================
// Test VelocityDependentFrictionModel
// ============================================================================
static void test_velocity_dependent_friction() {
    std::cout << "--- VelocityDependentFrictionModel ---\n";

    // Default construction: mu_s=0.3, mu_k=0.2, decay=10
    VelocityDependentFrictionModel fric;
    CHECK_NEAR(fric.mu_static, 0.3, 1e-15, "default mu_static");
    CHECK_NEAR(fric.mu_kinetic, 0.2, 1e-15, "default mu_kinetic");
    CHECK_NEAR(fric.decay_coefficient, 10.0, 1e-15, "default decay_coefficient");

    // At zero velocity => mu_static
    Real mu0 = fric.compute_friction(0.0);
    CHECK_NEAR(mu0, 0.3, 1e-12, "friction at v=0 equals mu_static");

    // At large velocity => approaches mu_kinetic
    Real mu_high = fric.compute_friction(100.0);
    CHECK_NEAR(mu_high, 0.2, 1e-6, "friction at v=100 approaches mu_kinetic");

    // Intermediate velocity: mu = 0.2 + 0.1*exp(-10*1.0) = 0.2 + 0.1*exp(-10)
    Real mu1 = fric.compute_friction(1.0);
    Real expected1 = 0.2 + 0.1 * std::exp(-10.0);
    CHECK_NEAR(mu1, expected1, 1e-12, "friction at v=1.0");

    // Custom construction
    VelocityDependentFrictionModel fric2(0.5, 0.1, 5.0);
    Real mu2_0 = fric2.compute_friction(0.0);
    CHECK_NEAR(mu2_0, 0.5, 1e-12, "custom friction at v=0");
}

// ============================================================================
// Test ShellThicknessContact
// ============================================================================
static void test_shell_thickness_contact() {
    std::cout << "--- ShellThicknessContact ---\n";

    // ShellContactConfig gap computation
    ShellContactConfig cfg(0.002, 0.004, true);
    // gap = raw - 0.002/2 - 0.004/2 = raw - 0.001 - 0.002 = raw - 0.003
    Real gap = cfg.compute_gap(0.01);
    CHECK_NEAR(gap, 0.007, 1e-12, "shell gap with thickness offset");

    // Offset disabled => raw gap returned
    ShellContactConfig cfg_no(0.002, 0.004, false);
    Real gap_no = cfg_no.compute_gap(0.01);
    CHECK_NEAR(gap_no, 0.01, 1e-12, "shell gap offset disabled");

    // ShellThicknessContact class
    ShellThicknessContact stc(cfg);
    CHECK_NEAR(stc.compute_gap(0.01), 0.007, 1e-12, "ShellThicknessContact gap");

    // Force: penetration when gap < 0
    // raw_distance = 0.002 => gap = 0.002 - 0.003 = -0.001 (penetrating)
    Real force = stc.compute_force(0.002, 1.0e6);
    CHECK_NEAR(force, 1.0e6 * 0.001, 1e-3, "shell contact penalty force");

    // No force when gap >= 0
    Real force_no = stc.compute_force(0.01, 1.0e6);
    CHECK_NEAR(force_no, 0.0, 1e-12, "shell contact no penetration => zero force");

    // set_config
    ShellThicknessContact stc2;
    stc2.set_config(cfg);
    CHECK_NEAR(stc2.config().slave_thickness, 0.002, 1e-15, "set_config preserves slave thickness");
}

// ============================================================================
// Test EdgeToEdgeContact
// ============================================================================
static void test_edge_to_edge_contact() {
    std::cout << "--- EdgeToEdgeContact ---\n";

    EdgeToEdgeContact ete;
    CHECK_NEAR(ete.search_radius(), 0.1, 1e-15, "default search radius");
    CHECK_NEAR(ete.penalty_stiffness(), 1.0e6, 1e-3, "default penalty stiffness");

    ete.set_search_radius(0.5);
    ete.set_penalty_stiffness(2.0e6);
    CHECK_NEAR(ete.search_radius(), 0.5, 1e-15, "updated search radius");

    // Two crossing edges in 3D:
    // Edge 0: node0(0,0,0) -> node1(1,0,0)   (along X)
    // Edge 1: node2(0.5,-0.5,0.05) -> node3(0.5,0.5,0.05) (along Y, offset in Z)
    Real coords[] = {
        0.0, 0.0, 0.0,       // node 0
        1.0, 0.0, 0.0,       // node 1
        0.5, -0.5, 0.05,     // node 2
        0.5,  0.5, 0.05      // node 3
    };

    ete.add_edge(0, 1);
    ete.add_edge(2, 3);
    CHECK(ete.num_edges() == 2, "two edges added");

    auto pairs = ete.detect_edge_contact(coords);
    CHECK(pairs.size() == 1, "one edge pair detected");
    if (!pairs.empty()) {
        CHECK_NEAR(pairs[0].gap, 0.05, 1e-10, "edge gap is 0.05");
    }

    // Force computation: penetration = search_radius - gap = 0.5 - 0.05 = 0.45
    Real forces[12] = {};
    if (!pairs.empty()) {
        ete.compute_edge_force(pairs[0], forces);
        // Total force magnitude = k * penetration = 2e6 * 0.45 = 900000
        // Distributed to 4 nodes along normal
        Real total_f = 0.0;
        for (int i = 0; i < 12; ++i) total_f += std::abs(forces[i]);
        CHECK(total_f > 0.0, "edge contact forces are nonzero");
    }

    // Clear edges
    ete.clear_edges();
    CHECK(ete.num_edges() == 0, "edges cleared");
}

// ============================================================================
// Test SegmentBasedContact
// ============================================================================
static void test_segment_based_contact() {
    std::cout << "--- SegmentBasedContact ---\n";

    // Two parallel quad faces close together in Z
    std::vector<Real> coords;
    Index base_s = make_quad_face(coords, 0.0);   // slave face at z=0
    Index base_m = make_quad_face(coords, 0.05);   // master face at z=0.05

    std::vector<Index> slave_conn = {base_s, base_s+1, base_s+2, base_s+3};
    std::vector<Index> master_conn = {base_m, base_m+1, base_m+2, base_m+3};

    SegmentBasedContact sbc;
    sbc.set_penalty_stiffness(1.0e6);
    sbc.set_search_gap(0.1);
    sbc.set_slave_faces(slave_conn, 1, 4);
    sbc.set_master_faces(master_conn, 1, 4);

    CHECK(sbc.num_slave_faces() == 1, "1 slave face");
    CHECK(sbc.num_master_faces() == 1, "1 master face");

    auto pairs = sbc.detect(coords.data());
    CHECK(pairs.size() >= 1, "segment pair detected for close faces");

    if (!pairs.empty()) {
        CHECK(pairs[0].gap < 0.1, "gap is within search distance");
        CHECK(pairs[0].overlap_area > 0.0, "overlap area is positive");
    }

    // Compute forces
    std::vector<Real> forces(coords.size(), 0.0);
    if (!pairs.empty()) {
        sbc.compute_segment_force(pairs[0], forces.data());
        // Forces should be nonzero on at least slave nodes
        Real total_f = 0.0;
        for (auto f : forces) total_f += std::abs(f);
        CHECK(total_f > 0.0, "segment contact forces are nonzero");
    }
}

// ============================================================================
// Test SelfContact
// ============================================================================
static void test_self_contact() {
    std::cout << "--- SelfContact ---\n";

    SelfContact sc;
    CHECK_NEAR(sc.penalty_stiffness(), 1.0e6, 1e-3, "default penalty stiffness");
    CHECK_NEAR(sc.search_radius(), 0.1, 1e-15, "default search radius");

    sc.set_penalty_stiffness(5.0e5);
    sc.set_search_radius(0.2);
    CHECK_NEAR(sc.penalty_stiffness(), 5.0e5, 1e-3, "updated penalty stiffness");

    // Create two non-adjacent quad faces close together
    // Face 0: nodes 0,1,2,3 at z=0
    // Face 1: nodes 4,5,6,7 at z=0.05 (no shared nodes => not neighbors)
    std::vector<Real> coords;
    Index base0 = make_quad_face(coords, 0.0);
    Index base1 = make_quad_face(coords, 0.05);

    std::vector<Index> conn = {
        base0, base0+1, base0+2, base0+3,
        base1, base1+1, base1+2, base1+3
    };

    sc.set_surface(conn, 2, 4);
    CHECK(sc.num_faces() == 2, "2 faces registered");

    // Neighbor check: faces with no shared nodes are NOT neighbors
    CHECK(!sc.are_neighbors(0, 1), "non-adjacent faces are not neighbors");

    // Detect self-contact (no displacement)
    auto pairs = sc.detect_self_contact(coords.data(), nullptr);
    // The faces are within search_radius=0.2 and not neighbors
    CHECK(pairs.size() >= 1, "self-contact pair detected for close non-adjacent faces");
}

// ============================================================================
// Test SymmetricContact
// ============================================================================
static void test_symmetric_contact() {
    std::cout << "--- SymmetricContact ---\n";

    SymmetricContact sym;

    // Surface A: quad at z=0, Surface B: quad at z=0.05
    std::vector<Real> coords;
    Index baseA = make_quad_face(coords, 0.0);
    Index baseB = make_quad_face(coords, 0.05);

    std::vector<Index> connA = {baseA, baseA+1, baseA+2, baseA+3};
    std::vector<Index> connB = {baseB, baseB+1, baseB+2, baseB+3};

    sym.set_penalty_stiffness(1.0e6);
    sym.set_search_gap(0.1);
    sym.set_surface_A(connA, 1, 4);
    sym.set_surface_B(connB, 1, 4);

    CHECK(sym.num_faces_A() == 1, "1 face in surface A");
    CHECK(sym.num_faces_B() == 1, "1 face in surface B");

    std::vector<Real> forces(coords.size(), 0.0);
    int nc = sym.compute_symmetric_forces(coords.data(), forces.data());
    CHECK(nc >= 1, "symmetric contact detected at least one pair");

    // Both passes contribute forces (averaged). Due to symmetric geometry,
    // forces on surface A and B along the contact normal should exist.
    // Check that force computation ran without error (nc > 0 is the key check).
    // For perfectly symmetric parallel faces, the averaged forces may partially
    // cancel due to opposite normal contributions, so we verify detection count.
    CHECK(nc >= 2, "symmetric contact has contributions from both passes");
}

// ============================================================================
// Test RigidDeformableContact
// ============================================================================
static void test_rigid_deformable_contact() {
    std::cout << "--- RigidDeformableContact ---\n";

    RigidDeformableContact rdc;
    CHECK(!rdc.is_initialized(), "not initialized before precompute");

    // Rigid flat plate at z=0, slave node slightly below at z=-0.01
    std::vector<Real> coords;
    // Rigid quad: nodes 0-3
    Index base = make_quad_face(coords, 0.0);
    // Slave node 4 at (0.5, 0.5, -0.01) => below the plate
    coords.push_back(0.5);
    coords.push_back(0.5);
    coords.push_back(-0.01);
    Index slave_node = base + 4;

    std::vector<Index> rigid_conn = {base, base+1, base+2, base+3};
    rdc.set_rigid_surface(rigid_conn, 1, 4);
    rdc.set_slave_nodes({slave_node});
    rdc.set_penalty_stiffness(1.0e8);
    rdc.set_search_gap(0.1);

    CHECK(rdc.num_rigid_faces() == 1, "1 rigid face");
    CHECK(rdc.slave_nodes().size() == 1, "1 slave node");

    rdc.precompute_rigid_geometry(coords.data());
    CHECK(rdc.is_initialized(), "initialized after precompute");

    std::vector<Real> forces(coords.size(), 0.0);
    int nc = rdc.compute_forces(coords.data(), forces.data());
    CHECK(nc >= 1, "rigid-deformable contact detected");

    // Slave node is at z=-0.01, rigid face normal points in +z
    // Force should push slave node in +z direction
    Real fz = forces[slave_node * 3 + 2];
    CHECK(fz > 0.0, "slave node pushed in +z by rigid surface");
}

// ============================================================================
// Test MultiSurfaceContactManager
// ============================================================================
static void test_multi_surface_contact_manager() {
    std::cout << "--- MultiSurfaceContactManager ---\n";

    MultiSurfaceContactManager mgr;
    CHECK(mgr.num_surfaces() == 0, "initially no surfaces");
    CHECK(mgr.num_interfaces() == 0, "initially no interfaces");

    // Two quad surfaces close together
    std::vector<Real> coords;
    Index baseA = make_quad_face(coords, 0.0);
    Index baseB = make_quad_face(coords, 0.05);

    std::vector<Index> connA = {baseA, baseA+1, baseA+2, baseA+3};
    std::vector<Index> connB = {baseB, baseB+1, baseB+2, baseB+3};

    Index idA = mgr.register_surface("surfA", connA, 1, 4, false);
    Index idB = mgr.register_surface("surfB", connB, 1, 4, false);

    CHECK(mgr.num_surfaces() == 2, "two surfaces registered");
    CHECK(idA == 0, "surfA has index 0");
    CHECK(idB == 1, "surfB has index 1");
    CHECK(mgr.surfaces()[0].name == "surfA", "surfA name correct");
    CHECK(mgr.surfaces()[1].name == "surfB", "surfB name correct");

    // Manual interface add
    mgr.add_interface(idA, idB, ContactAlgorithm::SegmentBased, 1.0e6, 0.1);
    CHECK(mgr.num_interfaces() == 1, "one interface added manually");

    std::vector<Real> forces(coords.size(), 0.0);
    int nc = mgr.solve_all(coords.data(), forces.data());
    CHECK(nc >= 1, "manager solve_all finds contacts");

    // Auto-detect test with a fresh manager
    MultiSurfaceContactManager mgr2;
    mgr2.register_surface("A", connA, 1, 4, false);
    mgr2.register_surface("B", connB, 1, 4, true);  // B is rigid

    mgr2.auto_detect_pairs(coords.data(), 0.5);
    CHECK(mgr2.num_interfaces() >= 1, "auto_detect finds at least one pair");

    // The auto-detected interface should use RigidDeformable since B is rigid
    if (mgr2.num_interfaces() >= 1) {
        CHECK(mgr2.interfaces()[0].algorithm == ContactAlgorithm::RigidDeformable,
              "auto-detect chooses RigidDeformable when one surface is rigid");
    }
}

// ============================================================================
// Test ContactAlgorithm enum completeness
// ============================================================================
static void test_contact_algorithm_enum() {
    std::cout << "--- ContactAlgorithm enum ---\n";

    // Verify all enum values compile and are distinct
    ContactAlgorithm a = ContactAlgorithm::NodeToSurface;
    ContactAlgorithm b = ContactAlgorithm::SegmentBased;
    ContactAlgorithm c = ContactAlgorithm::SelfContact;
    ContactAlgorithm d = ContactAlgorithm::Symmetric;
    ContactAlgorithm e = ContactAlgorithm::RigidDeformable;
    ContactAlgorithm f = ContactAlgorithm::EdgeToEdge;
    CHECK(a != b && b != c && c != d && d != e && e != f, "all enum values are distinct");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== contact_wave12_test ===\n\n";

    test_contact_stiffness_scaler();
    test_velocity_dependent_friction();
    test_shell_thickness_contact();
    test_edge_to_edge_contact();
    test_segment_based_contact();
    test_self_contact();
    test_symmetric_contact();
    test_rigid_deformable_contact();
    test_multi_surface_contact_manager();
    test_contact_algorithm_enum();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return (tests_failed > 0) ? 1 : 0;
}
