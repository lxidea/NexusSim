/**
 * @file element_library_test.cpp
 * @brief Comprehensive test for all FEM elements in library
 *
 * Tests instantiation and basic functionality of:
 * - Hex8 (8-node hexahedron)
 * - Tet4 (4-node tetrahedron)
 * - Shell4 (4-node shell)
 * - Beam2 (2-node beam)
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/discretization/hex8.hpp>
#include <nexussim/discretization/tet4.hpp>
#include <nexussim/discretization/shell4.hpp>
#include <nexussim/discretization/beam2.hpp>
#include <cmath>
#include <iostream>

using namespace nxs;
using namespace nxs::fem;

void test_hex8() {
    NXS_LOG_INFO("=== Testing Hex8 Element ===");

    Hex8Element hex;
    auto props = hex.properties();

    NXS_LOG_INFO("  Nodes: {}, DOFs: {}", props.num_nodes, props.num_dof_per_node * props.num_nodes);

    // Unit cube coordinates
    Real coords[24] = {
        0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0,
        0, 0, 1,  1, 0, 1,  1, 1, 1,  0, 1, 1
    };

    // Test shape functions at center
    Real xi[3] = {0, 0, 0};
    Real N[8];
    hex.shape_functions(xi, N);
    Real sum = 0;
    for (int i = 0; i < 8; ++i) sum += N[i];
    NXS_LOG_INFO("  Shape functions sum: {:.6f} (should be 1.0)", sum);

    // Test volume
    Real vol = hex.volume(coords);
    NXS_LOG_INFO("  Volume: {:.6f} (should be 1.0)", vol);

    // Test mass matrix
    Real M[24*24] = {0};
    hex.mass_matrix(coords, 7850.0, M);
    Real total_mass = 0;
    for (int i = 0; i < 24; ++i) total_mass += M[i*24 + i];
    NXS_LOG_INFO("  Total mass: {:.2f} kg (should be 7850.0)", total_mass);

    NXS_LOG_INFO("  Hex8: PASS\n");
}

void test_tet4() {
    NXS_LOG_INFO("=== Testing Tet4 Element ===");

    Tet4Element tet;
    auto props = tet.properties();

    NXS_LOG_INFO("  Nodes: {}, DOFs: {}", props.num_nodes, props.num_dof_per_node * props.num_nodes);

    // Unit tetrahedron coordinates
    Real coords[12] = {
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };

    // Test shape functions at center
    Real xi[3] = {0.25, 0.25, 0.25};
    Real N[4];
    tet.shape_functions(xi, N);
    Real sum = 0;
    for (int i = 0; i < 4; ++i) sum += N[i];
    NXS_LOG_INFO("  Shape functions sum: {:.6f} (should be 1.0)", sum);

    // Test volume
    Real vol = tet.volume(coords);
    Real expected_vol = 1.0 / 6.0;
    NXS_LOG_INFO("  Volume: {:.6f} (should be {:.6f})", vol, expected_vol);

    // Test mass matrix
    Real M[12*12] = {0};
    tet.mass_matrix(coords, 7850.0, M);
    Real total_mass = 0;
    for (int i = 0; i < 12; ++i) total_mass += M[i*12 + i];
    Real expected_mass = 7850.0 * expected_vol;
    NXS_LOG_INFO("  Total mass: {:.2f} kg (should be {:.2f})", total_mass, expected_mass);

    NXS_LOG_INFO("  Tet4: PASS\n");
}

void test_shell4() {
    NXS_LOG_INFO("=== Testing Shell4 Element ===");

    Shell4Element shell;
    auto props = shell.properties();

    NXS_LOG_INFO("  Nodes: {}, DOFs per node: {}, Total DOFs: {}",
                 props.num_nodes, props.num_dof_per_node,
                 props.num_dof_per_node * props.num_nodes);

    // Set thickness
    shell.set_thickness(0.01);  // 10 mm
    NXS_LOG_INFO("  Thickness: {:.3f} m", shell.thickness());

    // Unit square coordinates (in x-y plane)
    Real coords[12] = {
        0, 0, 0,
        1, 0, 0,
        1, 1, 0,
        0, 1, 0
    };

    // Test shape functions at center
    Real xi[3] = {0, 0, 0};
    Real N[4];
    shell.shape_functions(xi, N);
    Real sum = 0;
    for (int i = 0; i < 4; ++i) sum += N[i];
    NXS_LOG_INFO("  Shape functions sum: {:.6f} (should be 1.0)", sum);

    // Test area (volume returns area for shells)
    Real area = shell.volume(coords);
    NXS_LOG_INFO("  Area: {:.6f} (should be 1.0)", area);

    // Test mass matrix (24x24 for 4 nodes × 6 DOFs)
    Real M[24*24] = {0};
    shell.mass_matrix(coords, 7850.0, M);
    Real total_mass = 0;
    for (int i = 0; i < 24; ++i) total_mass += M[i*24 + i];
    Real expected_mass = 7850.0 * area * shell.thickness();
    NXS_LOG_INFO("  Total mass: {:.2f} kg (should be {:.2f})", total_mass, expected_mass);

    NXS_LOG_INFO("  Shell4: PASS\n");
}

void test_beam2() {
    NXS_LOG_INFO("=== Testing Beam2 Element ===");

    Beam2Element beam;
    auto props = beam.properties();

    NXS_LOG_INFO("  Nodes: {}, DOFs per node: {}, Total DOFs: {}",
                 props.num_nodes, props.num_dof_per_node,
                 props.num_dof_per_node * props.num_nodes);

    // Set circular cross-section (radius = 0.05 m)
    beam.set_circular_section(0.05);
    NXS_LOG_INFO("  Cross-section: circular, r = 0.05 m");
    NXS_LOG_INFO("  Area: {:.6e} m²", beam.area());
    NXS_LOG_INFO("  Iy: {:.6e} m⁴", beam.moment_y());
    NXS_LOG_INFO("  Iz: {:.6e} m⁴", beam.moment_z());
    NXS_LOG_INFO("  J: {:.6e} m⁴", beam.torsion_constant());

    // Beam along x-axis, length = 2.0 m
    Real coords[6] = {
        0, 0, 0,
        2, 0, 0
    };

    // Test shape functions at center
    Real xi[3] = {0, 0, 0};
    Real N[2];
    beam.shape_functions(xi, N);
    Real sum = 0;
    for (int i = 0; i < 2; ++i) sum += N[i];
    NXS_LOG_INFO("  Shape functions sum: {:.6f} (should be 1.0)", sum);

    // Test length
    Real len = beam.length(coords);
    NXS_LOG_INFO("  Length: {:.6f} m (should be 2.0)", len);

    // Test volume (length × area)
    Real vol = beam.volume(coords);
    Real expected_vol = len * beam.area();
    NXS_LOG_INFO("  Volume: {:.6e} m³ (should be {:.6e})", vol, expected_vol);

    // Test mass matrix (12x12 for 2 nodes × 6 DOFs)
    Real M[12*12] = {0};
    beam.mass_matrix(coords, 7850.0, M);
    Real total_mass = 0;
    for (int i = 0; i < 12; ++i) total_mass += M[i*12 + i];
    Real expected_mass = 7850.0 * expected_vol;
    NXS_LOG_INFO("  Total mass: {:.2f} kg (should be {:.2f})", total_mass, expected_mass);

    NXS_LOG_INFO("  Beam2: PASS\n");
}

int main() {
    // Initialize NexusSim
    nxs::InitOptions options;
    options.log_level = nxs::Logger::Level::Info;
    nxs::Context context(options);

    NXS_LOG_INFO("=================================================");
    NXS_LOG_INFO("NexusSim Element Library Test");
    NXS_LOG_INFO("=================================================\n");

    try {
        test_hex8();
        test_tet4();
        test_shell4();
        test_beam2();

        NXS_LOG_INFO("=================================================");
        NXS_LOG_INFO("All Element Tests PASSED!");
        NXS_LOG_INFO("=================================================");

    } catch (const std::exception& e) {
        NXS_LOG_ERROR("Test failed with exception: {}", e.what());
        return 1;
    }

    return 0;
}
