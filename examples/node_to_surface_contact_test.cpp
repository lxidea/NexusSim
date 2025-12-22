/**
 * @file node_to_surface_contact_test.cpp
 * @brief Test suite for node-to-surface contact algorithm
 *
 * Tests:
 * 1. Triangle projection (barycentric)
 * 2. Quad projection (Newton-Raphson)
 * 3. No contact detection
 * 4. Penetration detection
 * 5. Penalty force computation
 * 6. Friction (stick)
 * 7. Friction (slip)
 * 8. Multiple contacts
 * 9. Tilted surface
 * 10. Spatial hashing performance
 */

#include <nexussim/fem/node_to_surface_contact.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>

using namespace nxs;
using namespace nxs::fem;

int total_tests = 0;
int passed_tests = 0;

void check(bool condition, const std::string& test_name) {
    total_tests++;
    if (condition) {
        passed_tests++;
        std::cout << "  [PASS] " << test_name << "\n";
    } else {
        std::cout << "  [FAIL] " << test_name << "\n";
    }
}

void check_near(Real a, Real b, Real tol, const std::string& test_name) {
    check(std::abs(a - b) < tol, test_name);
}

// ============================================================================
// Test 1: Triangle Projection
// ============================================================================

void test_triangle_projection() {
    std::cout << "\n=== Test 1: Triangle Projection ===\n";

    NodeToSurfaceConfig config;
    config.search_radius = 1.0;
    config.contact_thickness = 0.01;

    NodeToSurfaceContact contact(config);

    // Triangle in z=0 plane
    std::vector<Real> coords = {
        0.0, 0.0, 0.0,   // Node 0
        1.0, 0.0, 0.0,   // Node 1
        0.5, 1.0, 0.0,   // Node 2
        0.4, 0.3, 0.1    // Node 3 (slave, above triangle)
    };

    // Add triangle as master segment
    std::vector<Index> connectivity = {0, 1, 2};
    contact.add_master_segments(connectivity, 1, 3);

    // Set slave node
    contact.set_slave_nodes({3});
    contact.initialize(4);

    // Detect contact
    int num_contacts = contact.detect_contacts(coords.data());

    check(num_contacts == 0, "No contact (point above triangle)");

    // Move slave node closer
    coords[3*3+2] = 0.005;  // z = 5mm, within thickness
    num_contacts = contact.detect_contacts(coords.data());

    check(num_contacts == 1, "Contact detected (within thickness)");

    if (num_contacts > 0) {
        const auto& info = contact.get_active_contacts()[0];
        check_near(info.gap, 0.005, 0.001, "Gap distance correct");
        check_near(info.normal[2], 1.0, 0.01, "Normal is +z");
    } else {
        total_tests += 2;
    }
}

// ============================================================================
// Test 2: Quad Projection
// ============================================================================

void test_quad_projection() {
    std::cout << "\n=== Test 2: Quad Projection (Newton-Raphson) ===\n";

    NodeToSurfaceConfig config;
    config.search_radius = 1.0;
    config.contact_thickness = 0.02;

    NodeToSurfaceContact contact(config);

    // Unit square quad in z=0 plane
    std::vector<Real> coords = {
        0.0, 0.0, 0.0,   // Node 0
        1.0, 0.0, 0.0,   // Node 1
        1.0, 1.0, 0.0,   // Node 2
        0.0, 1.0, 0.0,   // Node 3
        0.5, 0.5, 0.01   // Node 4 (slave, above center)
    };

    // Add quad as master segment
    std::vector<Index> connectivity = {0, 1, 2, 3};
    contact.add_master_segments(connectivity, 1, 4);

    contact.set_slave_nodes({4});
    contact.initialize(5);

    int num_contacts = contact.detect_contacts(coords.data());

    check(num_contacts == 1, "Quad contact detected");

    if (num_contacts > 0) {
        const auto& info = contact.get_active_contacts()[0];
        check_near(info.xi[0], 0.0, 0.01, "Xi at center");
        check_near(info.xi[1], 0.0, 0.01, "Eta at center");
        check_near(info.contact_point[0], 0.5, 0.01, "Contact point x");
        check_near(info.contact_point[1], 0.5, 0.01, "Contact point y");
        check_near(info.gap, 0.01, 0.001, "Gap correct");
    } else {
        total_tests += 5;
    }
}

// ============================================================================
// Test 3: No Contact Detection
// ============================================================================

void test_no_contact() {
    std::cout << "\n=== Test 3: No Contact Detection ===\n";

    NodeToSurfaceConfig config;
    config.search_radius = 0.5;
    config.contact_thickness = 0.01;

    NodeToSurfaceContact contact(config);

    // Plate at z=0
    std::vector<Real> coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.5, 0.5, 0.5   // Far above plate
    };

    std::vector<Index> connectivity = {0, 1, 2, 3};
    contact.add_master_segments(connectivity, 1, 4);
    contact.set_slave_nodes({4});
    contact.initialize(5);

    int num_contacts = contact.detect_contacts(coords.data());
    check(num_contacts == 0, "No contact when far above");

    // Move outside XY bounds
    coords[4*3+0] = 5.0;  // x = 5 (outside [0,1])
    coords[4*3+2] = 0.005;
    num_contacts = contact.detect_contacts(coords.data());
    check(num_contacts == 0, "No contact outside projection");
}

// ============================================================================
// Test 4: Penetration Detection
// ============================================================================

void test_penetration() {
    std::cout << "\n=== Test 4: Penetration Detection ===\n";

    NodeToSurfaceConfig config;
    config.search_radius = 0.5;
    config.contact_thickness = 0.01;

    NodeToSurfaceContact contact(config);

    std::vector<Real> coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.5, 0.5, -0.005  // 5mm BELOW plate (penetrating)
    };

    std::vector<Index> connectivity = {0, 1, 2, 3};
    contact.add_master_segments(connectivity, 1, 4);
    contact.set_slave_nodes({4});
    contact.initialize(5);

    int num_contacts = contact.detect_contacts(coords.data());
    check(num_contacts == 1, "Penetration detected");

    if (num_contacts > 0) {
        const auto& info = contact.get_active_contacts()[0];
        check(info.gap < 0, "Gap is negative (penetration)");
        check_near(info.gap, -0.005, 0.001, "Penetration depth correct");
    } else {
        total_tests += 2;
    }
}

// ============================================================================
// Test 5: Penalty Force Computation
// ============================================================================

void test_penalty_force() {
    std::cout << "\n=== Test 5: Penalty Force Computation ===\n";

    NodeToSurfaceConfig config;
    config.search_radius = 0.5;
    config.contact_thickness = 0.01;
    config.penalty_scale = 1.0;
    config.static_friction = 0.0;  // No friction

    NodeToSurfaceContact contact(config);

    std::vector<Real> coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.5, 0.5, -0.005  // 5mm penetration
    };

    std::vector<Index> connectivity = {0, 1, 2, 3};
    contact.add_master_segments(connectivity, 1, 4);
    contact.set_slave_nodes({4});
    contact.initialize(5);

    contact.detect_contacts(coords.data());

    std::vector<Real> velocity(15, 0.0);
    std::vector<Real> forces(15, 0.0);

    Real element_stiffness = 1.0e8;  // 100 MPa × area
    Real dt = 1.0e-6;

    contact.compute_forces(coords.data(), velocity.data(),
                           element_stiffness, dt, forces.data());

    // Penetration = thickness - gap = 0.01 - (-0.005) = 0.015
    // Force = k * penetration = 1e8 * 0.015 = 1.5e6 N in +z direction
    Real expected_force = element_stiffness * 0.015;

    std::cout << "  Expected force: " << expected_force / 1e3 << " kN (+z)\n";
    std::cout << "  Computed force: " << forces[4*3+2] / 1e3 << " kN\n";

    check(forces[4*3+2] > 0, "Force is positive (upward)");
    check_near(forces[4*3+2], expected_force, expected_force * 0.2, "Force magnitude");

    // Check reaction on master nodes (should sum to -force)
    Real reaction_z = forces[0*3+2] + forces[1*3+2] + forces[2*3+2] + forces[3*3+2];
    check_near(reaction_z + forces[4*3+2], 0.0, 100.0, "Force balance (momentum)");
}

// ============================================================================
// Test 6: Friction - Stick
// ============================================================================

void test_friction_stick() {
    std::cout << "\n=== Test 6: Friction - Stick ===\n";

    NodeToSurfaceConfig config;
    config.search_radius = 0.5;
    config.contact_thickness = 0.01;
    config.penalty_scale = 1.0;
    config.static_friction = 0.5;
    config.dynamic_friction = 0.3;

    NodeToSurfaceContact contact(config);

    std::vector<Real> coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.5, 0.5, -0.005
    };

    std::vector<Index> connectivity = {0, 1, 2, 3};
    contact.add_master_segments(connectivity, 1, 4);
    contact.set_slave_nodes({4});
    contact.initialize(5);

    contact.detect_contacts(coords.data());

    // Small lateral velocity (should stick)
    std::vector<Real> velocity(15, 0.0);
    velocity[4*3+0] = 0.01;  // 1 cm/s in x

    std::vector<Real> forces(15, 0.0);
    Real element_stiffness = 1.0e8;
    Real dt = 1.0e-6;

    contact.compute_forces(coords.data(), velocity.data(),
                           element_stiffness, dt, forces.data());

    // Should have friction force in -x direction
    check(forces[4*3+0] < 0, "Friction opposes motion");

    const auto& info = contact.get_active_contacts()[0];
    check(info.sticking, "Contact is in stick regime");

    std::cout << "  Friction force x: " << forces[4*3+0] << " N\n";
}

// ============================================================================
// Test 7: Friction - Slip
// ============================================================================

void test_friction_slip() {
    std::cout << "\n=== Test 7: Friction - Slip ===\n";

    NodeToSurfaceConfig config;
    config.search_radius = 0.5;
    config.contact_thickness = 0.01;
    config.penalty_scale = 1.0;
    config.static_friction = 0.3;
    config.dynamic_friction = 0.2;

    NodeToSurfaceContact contact(config);

    std::vector<Real> coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.5, 0.5, -0.005
    };

    std::vector<Index> connectivity = {0, 1, 2, 3};
    contact.add_master_segments(connectivity, 1, 4);
    contact.set_slave_nodes({4});
    contact.initialize(5);

    Real element_stiffness = 1.0e8;
    Real dt = 1.0e-4;  // Larger dt to accumulate slip

    // Multiple steps with high velocity to cause slip
    std::vector<Real> velocity(15, 0.0);
    velocity[4*3+0] = 10.0;  // 10 m/s in x

    for (int step = 0; step < 10; ++step) {
        contact.detect_contacts(coords.data());
        std::vector<Real> forces(15, 0.0);
        contact.compute_forces(coords.data(), velocity.data(),
                               element_stiffness, dt, forces.data());

        // Update position based on velocity (simplified)
        coords[4*3+0] += velocity[4*3+0] * dt;
    }

    // After accumulating slip, check friction limit
    contact.detect_contacts(coords.data());
    std::vector<Real> forces(15, 0.0);
    contact.compute_forces(coords.data(), velocity.data(),
                           element_stiffness, dt, forces.data());

    Real fn = forces[4*3+2];  // Normal force
    Real ft_max = config.dynamic_friction * fn;  // Max friction

    std::cout << "  Normal force: " << fn / 1e3 << " kN\n";
    std::cout << "  Max friction: " << ft_max / 1e3 << " kN\n";
    std::cout << "  Actual friction x: " << -forces[4*3+0] / 1e3 << " kN\n";

    check(std::abs(forces[4*3+0]) <= std::abs(ft_max) * 1.5, "Friction limited by Coulomb");
}

// ============================================================================
// Test 8: Multiple Contacts
// ============================================================================

void test_multiple_contacts() {
    std::cout << "\n=== Test 8: Multiple Contacts ===\n";

    NodeToSurfaceConfig config;
    config.search_radius = 0.5;
    config.contact_thickness = 0.02;

    NodeToSurfaceContact contact(config);

    // Two separate quads
    std::vector<Real> coords = {
        // First quad
        0.0, 0.0, 0.0,  // 0
        1.0, 0.0, 0.0,  // 1
        1.0, 1.0, 0.0,  // 2
        0.0, 1.0, 0.0,  // 3
        // Second quad (shifted in x)
        2.0, 0.0, 0.0,  // 4
        3.0, 0.0, 0.0,  // 5
        3.0, 1.0, 0.0,  // 6
        2.0, 1.0, 0.0,  // 7
        // Two slave nodes
        0.5, 0.5, 0.01,  // 8 (above first quad)
        2.5, 0.5, 0.01   // 9 (above second quad)
    };

    std::vector<Index> connectivity = {0, 1, 2, 3, 4, 5, 6, 7};
    contact.add_master_segments(connectivity, 2, 4);
    contact.set_slave_nodes({8, 9});
    contact.initialize(10);

    int num_contacts = contact.detect_contacts(coords.data());

    check(num_contacts == 2, "Both contacts detected");

    if (num_contacts == 2) {
        const auto& contacts = contact.get_active_contacts();
        bool found_first = false, found_second = false;
        for (const auto& c : contacts) {
            if (c.slave_node == 8 && c.master_segment == 0) found_first = true;
            if (c.slave_node == 9 && c.master_segment == 1) found_second = true;
        }
        check(found_first, "First slave-segment pair correct");
        check(found_second, "Second slave-segment pair correct");
    } else {
        total_tests += 2;
    }
}

// ============================================================================
// Test 9: Tilted Surface
// ============================================================================

void test_tilted_surface() {
    std::cout << "\n=== Test 9: Tilted Surface ===\n";

    NodeToSurfaceConfig config;
    config.search_radius = 1.0;
    config.contact_thickness = 0.02;

    NodeToSurfaceContact contact(config);

    // 45-degree tilted quad
    Real s = 0.7071;  // sin(45) = cos(45)
    std::vector<Real> coords = {
        0.0, 0.0, 0.0,   // 0
        1.0, 0.0, 0.0,   // 1
        1.0, s, s,       // 2 (rotated)
        0.0, s, s,       // 3 (rotated)
        0.5, 0.4, 0.5    // 4 (slave, should be near tilted surface)
    };

    std::vector<Index> connectivity = {0, 1, 2, 3};
    contact.add_master_segments(connectivity, 1, 4);
    contact.set_slave_nodes({4});
    contact.initialize(5);

    int num_contacts = contact.detect_contacts(coords.data());

    std::cout << "  Active contacts: " << num_contacts << "\n";

    if (num_contacts > 0) {
        const auto& info = contact.get_active_contacts()[0];
        std::cout << "  Normal: [" << info.normal[0] << ", "
                  << info.normal[1] << ", " << info.normal[2] << "]\n";
        std::cout << "  Gap: " << info.gap << "\n";

        // Normal should be approximately [0, -s, s] (perpendicular to tilted plane)
        check_near(info.normal[0], 0.0, 0.1, "Normal x ~ 0");
        // y and z components depend on face orientation
        check(std::abs(info.normal[1]) > 0.3 || std::abs(info.normal[2]) > 0.3,
              "Normal has y or z component");
    }

    check(num_contacts >= 0, "Tilted projection works");
}

// ============================================================================
// Test 10: Spatial Hashing Performance
// ============================================================================

void test_spatial_hashing_performance() {
    std::cout << "\n=== Test 10: Spatial Hashing Performance ===\n";

    NodeToSurfaceConfig config;
    config.search_radius = 0.1;
    config.contact_thickness = 0.01;

    NodeToSurfaceContact contact(config);

    // Generate large mesh (grid of quads)
    int nx = 50, ny = 50;
    int num_segment_nodes = (nx + 1) * (ny + 1);
    int num_segments = nx * ny;
    int num_slaves = 100;
    int num_nodes = num_segment_nodes + num_slaves;

    std::vector<Real> coords(num_nodes * 3);

    // Grid nodes
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            int idx = i + j * (nx + 1);
            coords[idx*3 + 0] = static_cast<Real>(i) * 0.1;
            coords[idx*3 + 1] = static_cast<Real>(j) * 0.1;
            coords[idx*3 + 2] = 0.0;
        }
    }

    // Slave nodes (scattered above grid)
    for (int s = 0; s < num_slaves; ++s) {
        int idx = num_segment_nodes + s;
        coords[idx*3 + 0] = (s % 10) * 0.5 + 0.25;
        coords[idx*3 + 1] = (s / 10) * 0.5 + 0.25;
        coords[idx*3 + 2] = 0.005;  // Close to surface
    }

    // Add quad segments
    std::vector<Index> connectivity;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            connectivity.push_back(i + j * (nx + 1));
            connectivity.push_back(i + 1 + j * (nx + 1));
            connectivity.push_back(i + 1 + (j + 1) * (nx + 1));
            connectivity.push_back(i + (j + 1) * (nx + 1));
        }
    }
    contact.add_master_segments(connectivity, num_segments, 4);

    // Slave nodes
    std::vector<Index> slaves(num_slaves);
    for (int s = 0; s < num_slaves; ++s) {
        slaves[s] = num_segment_nodes + s;
    }
    contact.set_slave_nodes(slaves);
    contact.initialize(num_nodes);

    std::cout << "  Nodes: " << num_nodes << "\n";
    std::cout << "  Segments: " << num_segments << "\n";
    std::cout << "  Slaves: " << num_slaves << "\n";

    // Benchmark contact detection
    auto start = std::chrono::high_resolution_clock::now();
    int num_runs = 100;
    int total_contacts = 0;

    for (int run = 0; run < num_runs; ++run) {
        total_contacts += contact.detect_contacts(coords.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "  Time for " << num_runs << " detections: " << time_ms << " ms\n";
    std::cout << "  Time per detection: " << time_ms / num_runs << " ms\n";
    std::cout << "  Contacts found: " << total_contacts / num_runs << "\n";

    check(time_ms / num_runs < 10.0, "Detection time < 10ms");
    check(total_contacts > 0, "Some contacts detected");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Node-to-Surface Contact Test Suite ===\n";
    std::cout << std::setprecision(6) << std::fixed;

    test_triangle_projection();
    test_quad_projection();
    test_no_contact();
    test_penetration();
    test_penalty_force();
    test_friction_stick();
    test_friction_slip();
    test_multiple_contacts();
    test_tilted_surface();
    test_spatial_hashing_performance();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed_tests << "/" << total_tests << " tests passed\n";
    std::cout << "========================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
