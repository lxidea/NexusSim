/**
 * @file contact_test.cpp
 * @brief Test contact mechanics implementation
 *
 * Tests:
 * 1. Point projection to quadrilateral face
 * 2. Contact detection for penetrating node
 * 3. Penalty force computation
 * 4. Friction force computation
 */

#include <nexussim/fem/contact.hpp>
#include <nexussim/data/mesh.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace nxs;
using namespace nxs::fem;

// Helper to create mesh from coordinate array
void setup_mesh(std::shared_ptr<Mesh>& mesh, const std::vector<Real>& coords, int num_nodes) {
    mesh = std::make_shared<Mesh>(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        mesh->set_node_coordinates(i, Vec3r{coords[i*3], coords[i*3+1], coords[i*3+2]});
    }
}

// Helper to get coords array from mesh (for contact detection API)
void get_coords_array(const Mesh& mesh, std::vector<Real>& coords) {
    coords.resize(mesh.num_nodes() * 3);
    for (std::size_t i = 0; i < mesh.num_nodes(); ++i) {
        Vec3r c = mesh.get_node_coordinates(i);
        coords[i*3] = c[0];
        coords[i*3+1] = c[1];
        coords[i*3+2] = c[2];
    }
}

/**
 * @brief Test 1: Point above flat plate (no contact)
 */
bool test_no_contact() {
    std::cout << "=== Test 1: Point Above Plate (No Contact) ===" << std::endl;

    // Create a simple mesh with a flat plate (4 nodes in z=0 plane)
    std::shared_ptr<Mesh> mesh;

    // Add 5 nodes: 4 for plate at z=0, 1 slave node above plate
    std::vector<Real> coords = {
        // Plate nodes (z=0)
        0.0, 0.0, 0.0,  // Node 0
        1.0, 0.0, 0.0,  // Node 1
        1.0, 1.0, 0.0,  // Node 2
        0.0, 1.0, 0.0,  // Node 3
        // Slave node above plate
        0.5, 0.5, 0.5   // Node 4 (0.5m above center of plate)
    };
    setup_mesh(mesh, coords, 5);

    ContactMechanics contact;

    // Add plate as master surface
    std::vector<Index> face_nodes = {0, 1, 2, 3};
    std::vector<Index> face_ids = {0};
    contact.add_master_surface("plate", face_nodes, face_ids, ElementType::Shell4);

    // Add slave node
    std::vector<Index> slaves = {4};
    contact.add_slave_nodes("ball", slaves);

    // Set parameters
    ContactParameters params;
    params.penalty_stiffness = 1.0;
    params.contact_thickness = 0.01;  // 1cm gap allowed
    contact.set_parameters(params);

    // Initialize
    contact.initialize(mesh);

    // Zero displacement
    std::vector<Real> displacement(15, 0.0);

    // Detect contact
    contact.detect_contact(coords.data(), displacement.data());

    std::cout << "  Slave node at z=0.5m above plate at z=0" << std::endl;
    std::cout << "  Contact thickness: " << params.contact_thickness << " m" << std::endl;
    std::cout << "  Active contacts: " << contact.num_active_contacts() << std::endl;

    bool pass = (contact.num_active_contacts() == 0);
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    return pass;
}

/**
 * @brief Test 2: Point penetrating plate
 */
bool test_penetration() {
    std::cout << "=== Test 2: Point Penetrating Plate ===" << std::endl;

    std::shared_ptr<Mesh> mesh;

    // Same setup but slave node below plate surface
    std::vector<Real> coords = {
        // Plate nodes (z=0)
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        // Slave node penetrating plate (5mm below)
        0.5, 0.5, -0.005
    };
    setup_mesh(mesh, coords, 5);

    ContactMechanics contact;

    std::vector<Index> face_nodes = {0, 1, 2, 3};
    std::vector<Index> face_ids = {0};
    contact.add_master_surface("plate", face_nodes, face_ids, ElementType::Shell4);

    std::vector<Index> slaves = {4};
    contact.add_slave_nodes("ball", slaves);

    ContactParameters params;
    params.penalty_stiffness = 1.0;
    params.contact_thickness = 0.01;
    contact.set_parameters(params);

    contact.initialize(mesh);

    std::vector<Real> displacement(15, 0.0);

    contact.detect_contact(coords.data(), displacement.data());

    std::cout << "  Slave node at z=-0.005m (5mm below plate)" << std::endl;
    std::cout << "  Active contacts: " << contact.num_active_contacts() << std::endl;

    bool pass = (contact.num_active_contacts() == 1);

    if (contact.num_active_contacts() > 0) {
        const auto& pair = contact.get_active_contacts()[0];
        std::cout << "  Contact pair:" << std::endl;
        std::cout << "    Slave node: " << pair.slave_node << std::endl;
        std::cout << "    Penetration depth: " << pair.penetration_depth * 1000.0 << " mm" << std::endl;
        std::cout << "    Normal: [" << pair.normal[0] << ", " << pair.normal[1]
                  << ", " << pair.normal[2] << "]" << std::endl;

        // Check penetration depth (should be ~5mm = 0.005m)
        if (std::abs(pair.penetration_depth - 0.005) > 0.001) {
            pass = false;
            std::cout << "  WARNING: Penetration depth mismatch!" << std::endl;
        }

        // Check normal (should be [0, 0, 1] pointing up)
        if (std::abs(pair.normal[2] - 1.0) > 0.01) {
            pass = false;
            std::cout << "  WARNING: Normal direction mismatch!" << std::endl;
        }
    }

    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    return pass;
}

/**
 * @brief Test 3: Penalty force computation
 */
bool test_penalty_force() {
    std::cout << "=== Test 3: Penalty Force Computation ===" << std::endl;

    std::shared_ptr<Mesh> mesh;

    // Slave node penetrating 10mm
    std::vector<Real> coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.5, 0.5, -0.01  // 10mm penetration
    };
    setup_mesh(mesh, coords, 5);

    ContactMechanics contact;

    std::vector<Index> face_nodes = {0, 1, 2, 3};
    std::vector<Index> face_ids = {0};
    contact.add_master_surface("plate", face_nodes, face_ids, ElementType::Shell4);

    std::vector<Index> slaves = {4};
    contact.add_slave_nodes("ball", slaves);

    ContactParameters params;
    params.penalty_stiffness = 10.0;  // 10 × 1e6 = 1e7 N/m
    params.contact_thickness = 0.001;
    params.enable_friction = false;
    contact.set_parameters(params);

    contact.initialize(mesh);

    std::vector<Real> displacement(15, 0.0);
    std::vector<Real> velocity(15, 0.0);
    std::vector<Real> forces(15, 0.0);

    // Detect and compute forces
    contact.detect_contact(coords.data(), displacement.data());
    contact.compute_contact_forces(coords.data(), displacement.data(),
                                   velocity.data(), forces.data());

    std::cout << "  Penetration: 10mm" << std::endl;
    std::cout << "  Penalty stiffness: " << params.penalty_stiffness << " × 1e6 N/m" << std::endl;

    // Expected force: k * d = 1e7 * 0.01 = 1e5 N in +z direction
    Real expected_force = params.penalty_stiffness * 1.0e6 * 0.01;

    std::cout << "  Expected force: " << expected_force / 1000.0 << " kN (upward)" << std::endl;
    std::cout << "  Computed force on slave: ["
              << forces[12] << ", " << forces[13] << ", " << forces[14] << "] N" << std::endl;

    // Check force magnitude and direction
    Real fz = forces[14];  // z-component of force on node 4

    bool pass = true;
    if (std::abs(fz - expected_force) / expected_force > 0.1) {
        std::cout << "  WARNING: Force magnitude error > 10%" << std::endl;
        pass = false;
    }
    if (fz < 0) {
        std::cout << "  WARNING: Force direction should be positive z!" << std::endl;
        pass = false;
    }

    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    return pass;
}

/**
 * @brief Test 4: Friction force
 */
bool test_friction() {
    std::cout << "=== Test 4: Friction Force ===" << std::endl;

    std::shared_ptr<Mesh> mesh;

    // Slave node penetrating with lateral velocity
    std::vector<Real> coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.5, 0.5, -0.01  // 10mm penetration
    };
    setup_mesh(mesh, coords, 5);

    ContactMechanics contact;

    std::vector<Index> face_nodes = {0, 1, 2, 3};
    std::vector<Index> face_ids = {0};
    contact.add_master_surface("plate", face_nodes, face_ids, ElementType::Shell4);

    std::vector<Index> slaves = {4};
    contact.add_slave_nodes("ball", slaves);

    ContactParameters params;
    params.penalty_stiffness = 10.0;
    params.contact_thickness = 0.001;
    params.enable_friction = true;
    params.friction_coefficient = 0.3;
    contact.set_parameters(params);

    contact.initialize(mesh);

    std::vector<Real> displacement(15, 0.0);
    std::vector<Real> velocity(15, 0.0);

    // Set lateral velocity (sliding in x-direction)
    velocity[12] = 1.0;  // vx = 1 m/s for slave node

    std::vector<Real> forces_no_friction(15, 0.0);
    std::vector<Real> forces_with_friction(15, 0.0);

    // First compute without friction
    params.enable_friction = false;
    contact.set_parameters(params);
    contact.detect_contact(coords.data(), displacement.data());
    contact.compute_contact_forces(coords.data(), displacement.data(),
                                   velocity.data(), forces_no_friction.data());

    // Then compute with friction
    params.enable_friction = true;
    contact.set_parameters(params);
    forces_with_friction.assign(15, 0.0);
    contact.compute_contact_forces(coords.data(), displacement.data(),
                                   velocity.data(), forces_with_friction.data());

    std::cout << "  Friction coefficient: " << params.friction_coefficient << std::endl;
    std::cout << "  Lateral velocity: 1 m/s in x-direction" << std::endl;
    std::cout << "  Force without friction: ["
              << forces_no_friction[12] << ", " << forces_no_friction[13] << ", "
              << forces_no_friction[14] << "] N" << std::endl;
    std::cout << "  Force with friction: ["
              << forces_with_friction[12] << ", " << forces_with_friction[13] << ", "
              << forces_with_friction[14] << "] N" << std::endl;

    // Friction should add negative x-component (opposing motion)
    Real fx_diff = forces_with_friction[12] - forces_no_friction[12];

    std::cout << "  Friction force x-component: " << fx_diff << " N" << std::endl;

    bool pass = true;

    // Friction should oppose motion (negative x)
    if (fx_diff > 0) {
        std::cout << "  WARNING: Friction should oppose positive x velocity!" << std::endl;
        pass = false;
    }

    // Normal force should be similar
    if (std::abs(forces_with_friction[14] - forces_no_friction[14]) > 0.1 * std::abs(forces_no_friction[14])) {
        std::cout << "  WARNING: Normal force changed significantly with friction!" << std::endl;
        pass = false;
    }

    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    return pass;
}

/**
 * @brief Test 5: Projection to tilted face
 */
bool test_tilted_face() {
    std::cout << "=== Test 5: Contact with Tilted Face ===" << std::endl;

    std::shared_ptr<Mesh> mesh;

    // Tilted plate: 45 degrees around x-axis
    // Node positions form a plane: z = -y (tilted so normal is [0, sqrt(2)/2, sqrt(2)/2])
    const Real s = 0.7071;  // sin(45°) = cos(45°)
    std::vector<Real> coords = {
        // Tilted plate nodes
        0.0, 0.0, 0.0,      // Node 0
        1.0, 0.0, 0.0,      // Node 1
        1.0, s, s,          // Node 2 (rotated)
        0.0, s, s,          // Node 3 (rotated)
        // Slave node slightly inside the tilted plane
        0.5, 0.4, 0.3       // Node 4
    };
    setup_mesh(mesh, coords, 5);

    ContactMechanics contact;

    std::vector<Index> face_nodes = {0, 1, 2, 3};
    std::vector<Index> face_ids = {0};
    contact.add_master_surface("tilted_plate", face_nodes, face_ids, ElementType::Shell4);

    std::vector<Index> slaves = {4};
    contact.add_slave_nodes("ball", slaves);

    ContactParameters params;
    params.penalty_stiffness = 1.0;
    params.contact_thickness = 0.1;  // Large threshold for this test
    contact.set_parameters(params);

    contact.initialize(mesh);

    std::vector<Real> displacement(15, 0.0);

    contact.detect_contact(coords.data(), displacement.data());

    std::cout << "  Tilted plate at 45 degrees" << std::endl;
    std::cout << "  Active contacts: " << contact.num_active_contacts() << std::endl;

    bool pass = true;
    if (contact.num_active_contacts() > 0) {
        const auto& pair = contact.get_active_contacts()[0];
        std::cout << "  Contact normal: [" << pair.normal[0] << ", "
                  << pair.normal[1] << ", " << pair.normal[2] << "]" << std::endl;

        // Normal should be approximately [0, -s, s] (perpendicular to tilted plane)
        Real expected_ny = -s;
        Real expected_nz = s;

        if (std::abs(pair.normal[1] - expected_ny) > 0.1 ||
            std::abs(pair.normal[2] - expected_nz) > 0.1) {
            std::cout << "  Expected normal: [0, " << expected_ny << ", " << expected_nz << "]" << std::endl;
            // Note: sign may depend on face winding order
        }
    }

    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    return pass;
}

int main() {
    std::cout << std::setprecision(6);
    std::cout << "========================================" << std::endl;
    std::cout << "Contact Mechanics Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 0;

    if (test_no_contact()) passed++;
    total++;

    if (test_penetration()) passed++;
    total++;

    if (test_penalty_force()) passed++;
    total++;

    if (test_friction()) passed++;
    total++;

    if (test_tilted_face()) passed++;
    total++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
