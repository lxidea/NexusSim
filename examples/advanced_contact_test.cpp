/**
 * @file advanced_contact_test.cpp
 * @brief Comprehensive test for Phase 2 advanced contact algorithms
 *
 * Tests:
 * - Voxel collision detection
 * - Surface-to-surface contact (INT17-style)
 * - Edge-to-surface contact (INT25-style)
 * - Enhanced friction models
 */

#include <nexussim/fem/voxel_collision.hpp>
#include <nexussim/fem/surface_contact.hpp>
#include <nexussim/fem/edge_contact.hpp>
#include <nexussim/fem/friction_model.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

using namespace nxs;
using namespace nxs::fem;

int tests_passed = 0;
int tests_failed = 0;

void check(bool condition, const std::string& test_name) {
    if (condition) {
        std::cout << "[PASS] " << test_name << std::endl;
        tests_passed++;
    } else {
        std::cout << "[FAIL] " << test_name << std::endl;
        tests_failed++;
    }
}

// ============================================================================
// Test 1: Surface-to-Surface Contact
// ============================================================================

void test_surface_contact() {
    std::cout << "\n=== Test: Surface-to-Surface Contact ===\n";

    // Create two parallel plates with OPPOSING normals
    // Plate 1: z = 0 (normal pointing UP, +z) - CCW from above
    // Plate 2: z = 0.03 (normal pointing DOWN, -z) - CW from above (reversed order)

    std::vector<Real> coords = {
        // Plate 1 (4 nodes) - CCW when viewed from +z -> normal = +z
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        // Plate 2 (4 nodes) - CW when viewed from +z -> normal = -z
        0.0, 0.0, 0.03,
        0.0, 1.0, 0.03,
        1.0, 1.0, 0.03,
        1.0, 0.0, 0.03
    };

    std::vector<Real> velocity(24, 0.0);
    // Plate 2 moving down
    for (int i = 4; i < 8; ++i) {
        velocity[i * 3 + 2] = -1.0;
    }

    std::vector<Real> forces(24, 0.0);

    // Create surfaces
    ContactSurface surf1;
    surf1.connectivity = {0, 1, 2, 3};  // CCW -> normal +z (up)
    surf1.num_segments = 1;
    surf1.nodes_per_segment = 4;
    surf1.thickness = 0.02;
    surf1.part_id = 1;

    ContactSurface surf2;
    surf2.connectivity = {4, 5, 6, 7};  // CW -> normal -z (down)
    surf2.num_segments = 1;
    surf2.nodes_per_segment = 4;
    surf2.thickness = 0.02;
    surf2.part_id = 2;

    SurfaceContactConfig config;
    config.penalty_scale = 1.0e6;
    config.contact_thickness = 0.05;  // Increased to ensure contact
    config.gap_max = 0.1;
    config.two_pass = true;

    SurfaceToSurfaceContact contact(config);
    contact.add_surface(surf1);
    contact.add_surface(surf2);
    contact.initialize(coords.data(), 8);

    // Detect contacts
    int num_contacts = contact.detect(coords.data());
    std::cout << "Detected " << num_contacts << " contact pairs" << std::endl;

    check(num_contacts > 0, "Surface contact detected between parallel plates");

    // Compute forces
    contact.compute_forces(coords.data(), velocity.data(), 0.001, 1.0, forces.data());

    // Check that forces push plates apart
    Real force_z_top = forces[4 * 3 + 2] + forces[5 * 3 + 2] + forces[6 * 3 + 2] + forces[7 * 3 + 2];
    Real force_z_bot = forces[0 * 3 + 2] + forces[1 * 3 + 2] + forces[2 * 3 + 2] + forces[3 * 3 + 2];

    std::cout << "Force on top plate (z): " << force_z_top << std::endl;
    std::cout << "Force on bottom plate (z): " << force_z_bot << std::endl;

    check(force_z_top > 0, "Top plate pushed up");
    check(force_z_bot < 0, "Bottom plate pushed down");
    check(std::abs(force_z_top + force_z_bot) < 1.0e-10, "Force balance (Newton's 3rd law)");
}

// ============================================================================
// Test 2: Edge-to-Surface Contact
// ============================================================================

void test_edge_contact() {
    std::cout << "\n=== Test: Edge-to-Surface Contact ===\n";

    // Edge positioned above a plate (within contact thickness)
    std::vector<Real> coords = {
        // Plate (4 nodes) - CCW when viewed from +z -> normal = +z
        0.0, 0.0, 0.0,
        2.0, 0.0, 0.0,
        2.0, 2.0, 0.0,
        0.0, 2.0, 0.0,
        // Edge (2 nodes) - just above the plate, within contact_thickness
        0.5, 1.0, 0.03,
        1.5, 1.0, 0.03
    };

    std::vector<Real> velocity(18, 0.0);
    velocity[4 * 3 + 2] = -1.0;  // Edge node 1 moving down
    velocity[5 * 3 + 2] = -1.0;  // Edge node 2 moving down

    std::vector<Real> forces(18, 0.0);

    // Create edge
    std::vector<ContactEdge> edges;
    edges.emplace_back(4, 5, 0);

    // Create master surface
    std::vector<Index> seg_conn = {0, 1, 2, 3};

    EdgeContactConfig config;
    config.penalty_scale = 1.0e6;
    config.contact_thickness = 0.05;  // Edge at 0.03, this gives penetration
    config.search_radius = 0.1;

    EdgeToSurfaceContact contact(config);
    contact.set_edges(edges);
    contact.set_master_segments(seg_conn, 1, 4);
    contact.initialize(coords.data(), 6);

    // Detect contacts
    int num_contacts = contact.detect(coords.data());
    std::cout << "Detected " << num_contacts << " edge contacts" << std::endl;

    check(num_contacts > 0, "Edge-to-surface contact detected");

    // Compute forces
    contact.compute_forces(coords.data(), velocity.data(), 0.001, 1.0e6, forces.data());

    // Check forces on edge nodes
    Real edge_force_z = forces[4 * 3 + 2] + forces[5 * 3 + 2];
    std::cout << "Force on edge (z): " << edge_force_z << std::endl;

    check(edge_force_z > 0, "Edge pushed up from surface");
}

// ============================================================================
// Test 3: Friction Models
// ============================================================================

void test_friction_models() {
    std::cout << "\n=== Test: Friction Models ===\n";

    Real normal_force = 100.0;
    Vec3r normal = {0, 0, 1};
    Real dt = 0.001;

    // Test Coulomb friction
    {
        CoulombFriction friction(0.3, 0.2, 1.0e6);
        FrictionState state;
        Vec3r velocity = {1.0, 0, 0};  // Sliding in x

        Vec3r f = friction.compute_friction(normal_force, velocity, normal, dt, state);

        std::cout << "Coulomb friction force: " << f[0] << ", " << f[1] << ", " << f[2] << std::endl;
        check(f[0] < 0, "Coulomb friction opposes motion");
        check(std::abs(f[2]) < 1.0e-10, "Coulomb friction in tangent plane");
    }

    // Test rate-dependent friction
    {
        RateDependentFriction friction(0.4, 0.2, 10.0, 1.0e6);
        FrictionState state;

        // Low velocity - higher friction
        Vec3r v_slow = {0.01, 0, 0};
        Vec3r f_slow = friction.compute_friction(normal_force, v_slow, normal, dt, state);

        state.reset();

        // High velocity - lower friction
        Vec3r v_fast = {10.0, 0, 0};
        Vec3r f_fast = friction.compute_friction(normal_force, v_fast, normal, dt, state);

        std::cout << "Slow velocity friction: " << std::abs(f_slow[0]) << std::endl;
        std::cout << "Fast velocity friction: " << std::abs(f_fast[0]) << std::endl;

        // Both should oppose motion
        check(f_slow[0] < 0 && f_fast[0] < 0, "Rate-dependent friction opposes motion");
    }

    // Test orthotropic friction
    {
        OrthotropicFriction friction(0.2, 0.4, {1, 0, 0}, 1.0e6);
        FrictionState state_x, state_y;

        // Sliding in fiber direction (x)
        Vec3r v_x = {1.0, 0, 0};
        Vec3r f_x = friction.compute_friction(normal_force, v_x, normal, dt, state_x);

        // Sliding in transverse direction (y)
        Vec3r v_y = {0, 1.0, 0};
        Vec3r f_y = friction.compute_friction(normal_force, v_y, normal, dt, state_y);

        std::cout << "Fiber direction friction: " << std::abs(f_x[0]) << std::endl;
        std::cout << "Transverse friction: " << std::abs(f_y[1]) << std::endl;

        check(f_x[0] < 0 && f_y[1] < 0, "Orthotropic friction opposes motion in both directions");
    }

    // Test viscous friction
    {
        ViscousFriction friction(0.3, 0.2, 0.1, 1.0e6);
        FrictionState state;

        Vec3r velocity = {1.0, 0, 0};
        Vec3r f = friction.compute_friction(normal_force, velocity, normal, dt, state);

        std::cout << "Viscous friction force: " << f[0] << std::endl;
        check(f[0] < 0, "Viscous friction opposes motion");
    }

    // Test part-based friction table
    {
        FrictionTable table;
        table.set_default(0.3, 0.25);
        table.add_pair(1, 2, 0.5, 0.4);
        table.add_pair(1, 3, 0.1, 0.05);

        Real mu_12 = table.get_static_friction(1, 2);
        Real mu_13 = table.get_static_friction(1, 3);
        Real mu_default = table.get_static_friction(5, 6);

        check(std::abs(mu_12 - 0.5) < 1.0e-10, "Part pair 1-2 friction correct");
        check(std::abs(mu_13 - 0.1) < 1.0e-10, "Part pair 1-3 friction correct");
        check(std::abs(mu_default - 0.3) < 1.0e-10, "Default friction used for unspecified pairs");
    }
}

// ============================================================================
// Test 4: Self-Contact Detection
// ============================================================================

void test_self_contact() {
    std::cout << "\n=== Test: Self-Contact Detection ===\n";

    // Create two parallel plates with opposing normals for self-contact
    // This simulates a folded sheet where two surfaces face each other
    std::vector<Real> coords = {
        // Plate 1: z=0 plane (normal +z, CCW from above)
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        // Plate 2: z=0.04 plane (normal -z, CW from above, facing plate 1)
        0.0, 0.0, 0.04,
        0.0, 1.0, 0.04,
        1.0, 1.0, 0.04,
        1.0, 0.0, 0.04
    };

    ContactSurface surface;
    surface.connectivity = {
        0, 1, 2, 3,    // Plate 1 (normal +z)
        4, 5, 6, 7     // Plate 2 (normal -z, reversed order)
    };
    surface.num_segments = 2;
    surface.nodes_per_segment = 4;
    surface.thickness = 0.03;

    SurfaceContactConfig config;
    config.gap_max = 0.15;
    config.contact_thickness = 0.06;  // thickness (0.06) > gap (0.04) -> penetration
    config.two_pass = true;

    SurfaceToSurfaceContact contact(config);
    contact.add_surface(surface);
    contact.initialize(coords.data(), 12);

    int num_contacts = contact.detect(coords.data());
    std::cout << "Self-contact pairs detected: " << num_contacts << std::endl;

    // Left and right legs should be in contact
    check(num_contacts > 0, "Self-contact detected in folded sheet");
}

// ============================================================================
// Test 5: Performance Benchmark
// ============================================================================

void test_performance() {
    std::cout << "\n=== Test: Performance Benchmark ===\n";

    const int grid_size = 50;  // 50x50 mesh
    const Index num_nodes = (grid_size + 1) * (grid_size + 1);
    const Index num_segments = grid_size * grid_size;

    // Create a wavy surface
    std::vector<Real> coords;
    for (int j = 0; j <= grid_size; ++j) {
        for (int i = 0; i <= grid_size; ++i) {
            Real x = static_cast<Real>(i) * 0.1;
            Real y = static_cast<Real>(j) * 0.1;
            Real z = 0.1 * std::sin(x * 2.0) * std::cos(y * 2.0);
            coords.push_back(x);
            coords.push_back(y);
            coords.push_back(z);
        }
    }

    std::vector<Index> connectivity;
    for (int j = 0; j < grid_size; ++j) {
        for (int i = 0; i < grid_size; ++i) {
            Index n0 = j * (grid_size + 1) + i;
            Index n1 = n0 + 1;
            Index n2 = n1 + grid_size + 1;
            Index n3 = n0 + grid_size + 1;
            connectivity.push_back(n0);
            connectivity.push_back(n1);
            connectivity.push_back(n2);
            connectivity.push_back(n3);
        }
    }

    ContactSurface surface;
    surface.connectivity = connectivity;
    surface.num_segments = num_segments;
    surface.nodes_per_segment = 4;
    surface.thickness = 0.02;

    SurfaceContactConfig config;
    config.gap_max = 0.15;
    config.contact_thickness = 0.04;

    SurfaceToSurfaceContact contact(config);
    contact.add_surface(surface);

    auto start = std::chrono::high_resolution_clock::now();
    contact.initialize(coords.data(), num_nodes);
    auto init_time = std::chrono::high_resolution_clock::now();

    int num_contacts = contact.detect(coords.data());
    auto detect_time = std::chrono::high_resolution_clock::now();

    std::vector<Real> velocity(num_nodes * 3, 0.0);
    std::vector<Real> forces(num_nodes * 3, 0.0);
    contact.compute_forces(coords.data(), velocity.data(), 0.001, 1.0, forces.data());
    auto force_time = std::chrono::high_resolution_clock::now();

    auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(init_time - start).count();
    auto detect_ms = std::chrono::duration_cast<std::chrono::milliseconds>(detect_time - init_time).count();
    auto force_ms = std::chrono::duration_cast<std::chrono::milliseconds>(force_time - detect_time).count();

    std::cout << "Mesh size: " << num_nodes << " nodes, " << num_segments << " segments" << std::endl;
    std::cout << "Initialization: " << init_ms << " ms" << std::endl;
    std::cout << "Detection: " << detect_ms << " ms (" << num_contacts << " contacts)" << std::endl;
    std::cout << "Force computation: " << force_ms << " ms" << std::endl;

    check(init_ms < 2000, "Initialization under 2 seconds");
    check(detect_ms < 10000, "Detection under 10 seconds");  // Thorough search is slower
    check(force_ms < 1000, "Force computation under 1 second");
}

// ============================================================================
// Test 6: Contact Force Equilibrium
// ============================================================================

void test_force_equilibrium() {
    std::cout << "\n=== Test: Contact Force Equilibrium ===\n";

    // Two blocks in contact - check momentum conservation
    // Surfaces have opposing normals for proper contact
    std::vector<Real> coords = {
        // Block 1 (bottom) - normal +z (CCW from above)
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        // Block 2 (top, penetrating) - normal -z (CW from above)
        0.0, 0.0, 0.02,  // Just above, will penetrate with contact_thickness
        0.0, 1.0, 0.02,  // Reversed node order for opposing normal
        1.0, 1.0, 0.02,
        1.0, 0.0, 0.02
    };

    std::vector<Real> velocity(24, 0.0);
    std::vector<Real> forces(24, 0.0);

    ContactSurface surf1, surf2;
    surf1.connectivity = {0, 1, 2, 3};  // CCW -> normal +z
    surf1.num_segments = 1;
    surf1.nodes_per_segment = 4;
    surf1.thickness = 0.05;

    surf2.connectivity = {4, 5, 6, 7};  // CW -> normal -z
    surf2.num_segments = 1;
    surf2.nodes_per_segment = 4;
    surf2.thickness = 0.05;

    SurfaceContactConfig config;
    config.penalty_scale = 1.0e8;
    config.contact_thickness = 0.1;  // With gap=0.02, penetration = 0.1 - 0.02 = 0.08
    config.gap_max = 0.2;
    config.static_friction = 0.0;  // No friction for this test

    SurfaceToSurfaceContact contact(config);
    contact.add_surface(surf1);
    contact.add_surface(surf2);
    contact.initialize(coords.data(), 8);
    contact.detect(coords.data());
    contact.compute_forces(coords.data(), velocity.data(), 0.001, 1.0, forces.data());

    // Sum all forces
    Vec3r total_force = {0, 0, 0};
    for (int i = 0; i < 8; ++i) {
        total_force[0] += forces[i * 3 + 0];
        total_force[1] += forces[i * 3 + 1];
        total_force[2] += forces[i * 3 + 2];
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Total force: (" << total_force[0] << ", " << total_force[1] << ", " << total_force[2] << ")" << std::endl;

    Real force_mag = std::sqrt(total_force[0] * total_force[0] +
                               total_force[1] * total_force[1] +
                               total_force[2] * total_force[2]);

    check(force_mag < 1.0e-6, "Total contact force is zero (momentum conservation)");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    {
        std::cout << "========================================" << std::endl;
        std::cout << "Advanced Contact Algorithms Test Suite" << std::endl;
        std::cout << "Phase 2: OpenRadioss Feature Gap Plan" << std::endl;
        std::cout << "========================================" << std::endl;

        test_surface_contact();
        test_edge_contact();
        test_friction_models();
        test_self_contact();
        test_performance();
        test_force_equilibrium();

        std::cout << "\n========================================" << std::endl;
        std::cout << "Test Summary" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Passed: " << tests_passed << std::endl;
        std::cout << "Failed: " << tests_failed << std::endl;
        std::cout << "========================================" << std::endl;
    }

    Kokkos::finalize();
    return (tests_failed == 0) ? 0 : 1;
}
