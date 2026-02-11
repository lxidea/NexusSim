/**
 * @file rigid_body_test.cpp
 * @brief Comprehensive test for Wave 3: Rigid bodies, constraints, rigid walls
 */

#include <nexussim/fem/rigid_body.hpp>
#include <nexussim/fem/constraints.hpp>
#include <nexussim/fem/rigid_wall.hpp>
#include <iostream>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;

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

static bool near(Real a, Real b, Real tol = 1.0e-6) {
    return std::fabs(a - b) < tol;
}

// ==========================================================================
// Test 1: Rigid Body Free Fall
// ==========================================================================
void test_rigid_body_free_fall() {
    std::cout << "\n=== Test 1: Rigid Body Free Fall ===\n";

    // 4-node rigid body, mass = 4 kg, at height z = 10
    const int num_nodes = 4;
    Real positions[12] = {
        0.0, 0.0, 10.0,  // node 0
        1.0, 0.0, 10.0,  // node 1
        0.0, 1.0, 10.0,  // node 2
        1.0, 1.0, 10.0   // node 3
    };
    Real masses[4] = {1.0, 1.0, 1.0, 1.0};  // 1 kg each

    RigidBody rb(1, "FallingBlock");
    rb.set_slave_nodes({0, 1, 2, 3});
    rb.compute_properties(positions, masses);

    CHECK(near(rb.properties().mass, 4.0), "Total mass = 4 kg");
    CHECK(near(rb.properties().com[0], 0.5), "COM x = 0.5");
    CHECK(near(rb.properties().com[1], 0.5), "COM y = 0.5");
    CHECK(near(rb.properties().com[2], 10.0), "COM z = 10.0");

    // Apply gravity for 100 steps
    Real dt = 0.001;
    Real g = -9.81;
    for (int step = 0; step < 100; ++step) {
        // Apply gravity force
        rb.force()[2] += rb.properties().mass * g;
        rb.update(dt);
    }

    // After 0.1s: v_z = g*t = -0.981, z = 10 + 0.5*g*t² ≈ 9.95
    CHECK(rb.velocity()[2] < 0.0, "Falling: negative z-velocity");
    CHECK(near(rb.velocity()[2], g * 0.1, 0.01), "v_z ≈ -0.981 m/s");
    CHECK(rb.properties().com[2] < 10.0, "COM moved down");
}

// ==========================================================================
// Test 2: Rigid Body Rotation
// ==========================================================================
void test_rigid_body_rotation() {
    std::cout << "\n=== Test 2: Rigid Body Rotation ===\n";

    const int num_nodes = 4;
    Real positions[12] = {
        -1.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.0,-1.0, 0.0,
         0.0, 1.0, 0.0
    };
    Real masses[4] = {1.0, 1.0, 1.0, 1.0};

    RigidBody rb(2, "Spinner");
    rb.set_slave_nodes({0, 1, 2, 3});
    rb.compute_properties(positions, masses);

    CHECK(near(rb.properties().com[0], 0.0), "COM at origin x");
    CHECK(near(rb.properties().com[1], 0.0), "COM at origin y");
    CHECK(rb.properties().inertia[2] > 0.0, "Izz > 0 (moment of inertia)");

    // Apply torque about z-axis
    rb.torque()[2] = 10.0;  // Nm
    rb.update(0.01);

    CHECK(rb.angular_velocity()[2] > 0.0, "Spinning about z-axis");

    // Scatter velocities: nodes at (±1,0,0) should have vy from rotation
    Real velocities[12] = {};
    rb.scatter_to_nodes(positions, velocities);

    // v = v_com + ω × r.  ω = (0,0,ω_z>0), r = (-1,0,0) → ω×r = (0, ω_z*(-1), 0)
    CHECK(velocities[3*0+1] < 0.0, "Node 0: negative vy from CCW rotation at (-1,0,0)");
    // Node 1 at (1,0,0): ω×r = (0, ω_z*1, 0)
    CHECK(velocities[3*1+1] > 0.0, "Node 1: positive vy from CCW rotation at (1,0,0)");
}

// ==========================================================================
// Test 3: Rigid Body Manager
// ==========================================================================
void test_rigid_body_manager() {
    std::cout << "\n=== Test 3: Rigid Body Manager ===\n";

    RigidBodyManager mgr;
    mgr.add_rigid_body(1, "Body1");
    mgr.add_rigid_body(2, "Body2");

    CHECK(mgr.num_bodies() == 2, "Two bodies added");
    CHECK(mgr.find_by_id(1) != nullptr, "Find body by ID");
    CHECK(mgr.find_by_id(3) == nullptr, "Missing ID returns null");

    // Use indices instead of references (vector may reallocate)
    mgr.body(0).add_slave_node(0);
    mgr.body(0).add_slave_node(1);
    mgr.body(1).add_slave_node(2);
    mgr.body(1).add_slave_node(3);

    Real positions[12] = {0,0,0, 1,0,0, 2,0,0, 3,0,0};
    Real masses[4] = {1,1,1,1};

    mgr.initialize(positions, masses);
    CHECK(near(mgr.body(0).properties().mass, 2.0), "Body1 mass = 2 kg");
    CHECK(near(mgr.body(1).properties().mass, 2.0), "Body2 mass = 2 kg");
}

// ==========================================================================
// Test 4: Quaternion Operations
// ==========================================================================
void test_quaternion() {
    std::cout << "\n=== Test 4: Quaternion Operations ===\n";

    Quaternion q;
    CHECK(near(q.w, 1.0), "Identity quaternion w=1");
    CHECK(near(q.norm(), 1.0), "Identity norm = 1");

    // Rotation matrix from identity
    Real R[9];
    q.to_rotation_matrix(R);
    CHECK(near(R[0], 1.0) && near(R[4], 1.0) && near(R[8], 1.0),
          "Identity rotation matrix diagonal");

    // 90-degree rotation about z-axis
    Real angle = M_PI / 2.0;
    Quaternion qz(std::cos(angle/2), 0, 0, std::sin(angle/2));
    Real v_in[3] = {1.0, 0.0, 0.0};
    Real v_out[3];
    qz.rotate_vector(v_in, v_out);
    CHECK(near(v_out[0], 0.0, 1e-10) && near(v_out[1], 1.0, 1e-10),
          "90° z-rotation: (1,0,0) → (0,1,0)");

    // Quaternion multiplication (composition)
    Quaternion q2 = qz * qz;  // 180° about z
    Real v_out2[3];
    q2.rotate_vector(v_in, v_out2);
    CHECK(near(v_out2[0], -1.0, 1e-10), "180° z-rotation: (1,0,0) → (-1,0,0)");
}

// ==========================================================================
// Test 5: RBE2 Constraint
// ==========================================================================
void test_rbe2_constraint() {
    std::cout << "\n=== Test 5: RBE2 Kinematic Coupling ===\n";

    ConstraintManager mgr;

    auto& c = mgr.add_constraint(ConstraintType::RBE2, 1);
    c.master_node = 0;
    c.slave_nodes = {1, 2, 3};

    // Master moving at (1,0,0), slaves initially stationary
    Real positions[12] = {0,0,0, 1,0,0, 0,1,0, 1,1,0};
    Real velocities[12] = {1.0, 2.0, 3.0,  0,0,0,  0,0,0,  0,0,0};
    Real accelerations[12] = {};

    mgr.apply_constraints(positions, velocities, accelerations, 0.001);

    // Slaves should match master velocity
    CHECK(near(velocities[3], 1.0), "Slave 1: vx = master vx");
    CHECK(near(velocities[4], 2.0), "Slave 1: vy = master vy");
    CHECK(near(velocities[5], 3.0), "Slave 1: vz = master vz");
    CHECK(near(velocities[6], 1.0), "Slave 2: vx = master vx");
    CHECK(near(velocities[9], 1.0), "Slave 3: vx = master vx");
}

// ==========================================================================
// Test 6: RBE3 Weighted Average
// ==========================================================================
void test_rbe3_constraint() {
    std::cout << "\n=== Test 6: RBE3 Interpolation ===\n";

    ConstraintManager mgr;

    auto& c = mgr.add_constraint(ConstraintType::RBE3, 2);
    c.master_node = 0;  // Dependent
    c.slave_nodes = {1, 2};  // Independent
    c.weights = {1.0, 1.0};  // Equal weights

    Real positions[9] = {0,0,0, 1,0,0, -1,0,0};
    Real velocities[9] = {0,0,0, 2.0,0,0, 4.0,0,0};
    Real accelerations[9] = {};

    mgr.apply_constraints(positions, velocities, accelerations, 0.001);

    // Master velocity should be average: (2 + 4) / 2 = 3
    CHECK(near(velocities[0], 3.0), "RBE3 master vx = weighted average (3.0)");
}

// ==========================================================================
// Test 7: Planar Rigid Wall
// ==========================================================================
void test_planar_wall() {
    std::cout << "\n=== Test 7: Planar Rigid Wall ===\n";

    RigidWallContact contact;
    contact.set_penalty_stiffness(1.0e8);

    // Wall at z=0, normal pointing up
    auto& w = contact.add_wall(RigidWallType::Planar, 1);
    w.origin[0] = 0; w.origin[1] = 0; w.origin[2] = 0;
    w.normal[0] = 0; w.normal[1] = 0; w.normal[2] = 1;

    // Node above wall: no contact
    Real pos_above[3] = {0.0, 0.0, 1.0};
    Real vel[3] = {0.0, 0.0, -1.0};
    Real mass[1] = {1.0};
    Real forces[3] = {0.0, 0.0, 0.0};

    contact.compute_forces(1, pos_above, vel, mass, forces, 0.001);
    CHECK(near(forces[2], 0.0), "No contact force above wall");

    // Node below wall: penetrating
    Real pos_below[3] = {0.0, 0.0, -0.01};  // 1cm penetration
    Real forces2[3] = {0.0, 0.0, 0.0};

    contact.compute_forces(1, pos_below, vel, mass, forces2, 0.001);
    CHECK(forces2[2] > 0.0, "Upward contact force for penetrating node");

    auto stats = contact.get_stats();
    CHECK(stats.active_contacts == 1, "One active contact");
    CHECK(stats.max_penetration > 0.0, "Penetration detected");
}

// ==========================================================================
// Test 8: Wall with Friction
// ==========================================================================
void test_wall_friction() {
    std::cout << "\n=== Test 8: Wall Friction ===\n";

    RigidWallContact contact;
    contact.set_penalty_stiffness(1.0e8);

    auto& w = contact.add_wall(RigidWallType::Planar, 1);
    w.origin[2] = 0.0;
    w.normal[2] = 1.0;
    w.friction = 0.3;

    // Node penetrating with tangential velocity
    Real pos[3] = {0.0, 0.0, -0.005};
    Real vel[3] = {10.0, 0.0, -1.0};  // Sliding in x
    Real mass[1] = {1.0};
    Real forces[3] = {0.0, 0.0, 0.0};

    contact.compute_forces(1, pos, vel, mass, forces, 0.001);

    CHECK(forces[2] > 0.0, "Normal force pushes up");
    CHECK(forces[0] < 0.0, "Friction opposes sliding in x");
    CHECK(near(forces[1], 0.0, 1.0), "No friction in y (no sliding)");
}

// ==========================================================================
// Test 9: Moving Wall
// ==========================================================================
void test_moving_wall() {
    std::cout << "\n=== Test 9: Moving Wall ===\n";

    RigidWallContact contact;
    contact.set_penalty_stiffness(1.0e8);

    auto& w = contact.add_wall(RigidWallType::Moving, 1);
    w.origin[2] = 0.0;
    w.normal[2] = 1.0;
    w.velocity[2] = 10.0;  // Wall moving up at 10 m/s

    Real initial_z = w.origin[2];
    contact.update_walls(0.01);
    CHECK(w.origin[2] > initial_z, "Moving wall advances");
    CHECK(near(w.origin[2], 0.1), "Wall at z = 0.1 after 0.01s");
}

// ==========================================================================
// Test 10: Spherical Rigid Wall
// ==========================================================================
void test_spherical_wall() {
    std::cout << "\n=== Test 10: Spherical Rigid Wall ===\n";

    RigidWallContact contact;
    contact.set_penalty_stiffness(1.0e8);

    // Sphere centered at origin, radius 1
    auto& w = contact.add_wall(RigidWallType::Spherical, 1);
    w.origin[0] = 0; w.origin[1] = 0; w.origin[2] = 0;
    w.radius = 1.0;

    // Node outside sphere: no contact
    Real pos_out[3] = {2.0, 0.0, 0.0};
    Real vel[3] = {0, 0, 0};
    Real mass[1] = {1.0};
    Real forces_out[3] = {0, 0, 0};
    contact.compute_forces(1, pos_out, vel, mass, forces_out, 0.001);
    CHECK(near(forces_out[0], 0.0), "No force outside sphere");

    // Node inside sphere: pushed outward
    Real pos_in[3] = {0.5, 0.0, 0.0};
    Real forces_in[3] = {0, 0, 0};
    contact.compute_forces(1, pos_in, vel, mass, forces_in, 0.001);
    CHECK(forces_in[0] > 0.0, "Pushed outward in +x");
    CHECK(near(forces_in[1], 0.0), "No force in y");
    CHECK(near(forces_in[2], 0.0), "No force in z");
}

// ==========================================================================
// Test 11: Constraint Manager with Multiple Types
// ==========================================================================
void test_constraint_manager() {
    std::cout << "\n=== Test 11: Constraint Manager ===\n";

    ConstraintManager mgr;
    mgr.add_constraint(ConstraintType::RBE2, 1);
    mgr.add_constraint(ConstraintType::RBE3, 2);
    mgr.add_constraint(ConstraintType::Joint_Spherical, 3);

    CHECK(mgr.num_constraints() == 3, "Three constraints added");
    CHECK(mgr.constraint(0).type == ConstraintType::RBE2, "First is RBE2");
    CHECK(mgr.constraint(1).type == ConstraintType::RBE3, "Second is RBE3");
    CHECK(mgr.constraint(2).type == ConstraintType::Joint_Spherical, "Third is Spherical");
}

// ==========================================================================
// Test 12: Rigid Body Force Gathering
// ==========================================================================
void test_force_gathering() {
    std::cout << "\n=== Test 12: Force Gathering ===\n";

    Real positions[12] = {0,0,0, 2,0,0, 0,2,0, 2,2,0};
    Real masses[4] = {1, 1, 1, 1};

    RigidBody rb(1, "Test");
    rb.set_slave_nodes({0, 1, 2, 3});
    rb.compute_properties(positions, masses);

    // Apply forces to nodes
    Real forces[12] = {
        0,0,10,   // Node 0: 10N up
        0,0,10,   // Node 1: 10N up
        0,0,10,   // Node 2: 10N up
        0,0,10    // Node 3: 10N up
    };

    rb.gather_forces(positions, forces);

    CHECK(near(rb.force()[2], 40.0), "Total Fz = 40N");
    CHECK(near(rb.force()[0], 0.0), "No net Fx");

    // Asymmetric force should create torque
    Real forces2[12] = {
        0,0,0,     // Node 0: no force
        10,0,0,    // Node 1: 10N in x (at (2,0,0), COM at (1,1,0))
        0,0,0,     // Node 2: no force
        0,0,0      // Node 3: no force
    };

    RigidBody rb2(2, "Test2");
    rb2.set_slave_nodes({0, 1, 2, 3});
    rb2.compute_properties(positions, masses);
    rb2.gather_forces(positions, forces2);

    CHECK(near(rb2.force()[0], 10.0), "Fx = 10N");
    // Torque about z: rx * fy - ry * fx = (1)*0 - (-1)*10 = 10
    CHECK(rb2.torque()[2] != 0.0, "Non-zero torque from off-center force");
}

// ==========================================================================
// Main
// ==========================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "NexusSim Wave 3: Rigid Bodies Test\n";
    std::cout << "========================================\n";

    test_rigid_body_free_fall();
    test_rigid_body_rotation();
    test_rigid_body_manager();
    test_quaternion();
    test_rbe2_constraint();
    test_rbe3_constraint();
    test_planar_wall();
    test_wall_friction();
    test_moving_wall();
    test_spherical_wall();
    test_constraint_manager();
    test_force_gathering();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    std::cout << "========================================\n";

    return (tests_failed > 0) ? 1 : 0;
}
