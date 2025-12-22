/**
 * @file hex8_element_test.cpp
 * @brief Test Hex8 element functionality
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/discretization/hex8.hpp>
#include <iostream>
#include <iomanip>

int main() {
    // Initialize NexusSim
    nxs::InitOptions options;
    options.log_level = nxs::Logger::Level::Info;
    nxs::Context context(options);

    NXS_LOG_INFO("=================================================");
    NXS_LOG_INFO("Hex8 Element Test");
    NXS_LOG_INFO("=================================================");

    // Create Hex8 element
    nxs::fem::Hex8Element hex8;

    // Print element properties
    auto props = hex8.properties();
    NXS_LOG_INFO("Element properties:");
    NXS_LOG_INFO("  Type: Hex8");
    NXS_LOG_INFO("  Nodes: {}", props.num_nodes);
    NXS_LOG_INFO("  DOFs per node: {}", props.num_dof_per_node);
    NXS_LOG_INFO("  Gauss points: {}", props.num_gauss_points);
    NXS_LOG_INFO("  Spatial dim: {}", props.spatial_dim);

    // Define a unit cube element coordinates
    // 8 nodes x 3 coordinates
    nxs::Real coords[24] = {
        0.0, 0.0, 0.0,  // Node 0
        1.0, 0.0, 0.0,  // Node 1
        1.0, 1.0, 0.0,  // Node 2
        0.0, 1.0, 0.0,  // Node 3
        0.0, 0.0, 1.0,  // Node 4
        1.0, 0.0, 1.0,  // Node 5
        1.0, 1.0, 1.0,  // Node 6
        0.0, 1.0, 1.0   // Node 7
    };

    // ========================================================================
    // Test 1: Shape Functions at Element Center
    // ========================================================================
    NXS_LOG_INFO("\n--- Test 1: Shape Functions at Center ---");
    nxs::Real xi_center[3] = {0.0, 0.0, 0.0};
    nxs::Real N[8];

    hex8.shape_functions(xi_center, N);

    NXS_LOG_INFO("Shape functions at center (0,0,0):");
    for (int i = 0; i < 8; ++i) {
        std::cout << "  N[" << i << "] = " << std::fixed << std::setprecision(6) << N[i];
        if (i % 4 == 3) std::cout << std::endl;
    }

    // All shape functions should be 1/8 = 0.125 at center
    nxs::Real sum_N = 0.0;
    for (int i = 0; i < 8; ++i) {
        sum_N += N[i];
    }
    NXS_LOG_INFO("Sum of shape functions: {:.6f} (should be 1.0)", sum_N);

    // ========================================================================
    // Test 2: Jacobian at Element Center
    // ========================================================================
    NXS_LOG_INFO("\n--- Test 2: Jacobian Matrix ---");
    nxs::Real J[9];
    nxs::Real det_J = hex8.jacobian(xi_center, coords, J);

    NXS_LOG_INFO("Jacobian matrix at center:");
    NXS_LOG_INFO("  [{:.4f}  {:.4f}  {:.4f}]", J[0], J[1], J[2]);
    NXS_LOG_INFO("  [{:.4f}  {:.4f}  {:.4f}]", J[3], J[4], J[5]);
    NXS_LOG_INFO("  [{:.4f}  {:.4f}  {:.4f}]", J[6], J[7], J[8]);
    NXS_LOG_INFO("Jacobian determinant: {:.6f}", det_J);
    NXS_LOG_INFO("(For unit cube, det(J) should be 0.125)");

    // ========================================================================
    // Test 3: Element Volume
    // ========================================================================
    NXS_LOG_INFO("\n--- Test 3: Element Volume ---");
    nxs::Real volume = hex8.volume(coords);
    NXS_LOG_INFO("Element volume: {:.6f} (should be 1.0 for unit cube)", volume);

    // ========================================================================
    // Test 4: Characteristic Length
    // ========================================================================
    NXS_LOG_INFO("\n--- Test 4: Characteristic Length ---");
    nxs::Real char_length = hex8.characteristic_length(coords);
    NXS_LOG_INFO("Characteristic length: {:.6f} (should be 1.0 for unit cube)", char_length);

    // ========================================================================
    // Test 5: Lumped Mass Matrix
    // ========================================================================
    NXS_LOG_INFO("\n--- Test 5: Lumped Mass Matrix ---");
    nxs::Real M_lumped[24];
    nxs::Real density = 7850.0;  // Steel density kg/m^3

    hex8.lumped_mass_matrix(coords, density, M_lumped);

    NXS_LOG_INFO("Lumped mass (first 4 DOFs):");
    for (int i = 0; i < 4; ++i) {
        NXS_LOG_INFO("  M[{}] = {:.6f} kg", i, M_lumped[i]);
    }

    nxs::Real total_mass_lumped = 0.0;
    for (int i = 0; i < 24; ++i) {
        total_mass_lumped += M_lumped[i];
    }
    NXS_LOG_INFO("Total lumped mass: {:.2f} kg", total_mass_lumped);
    NXS_LOG_INFO("Expected: {:.2f} kg", density * volume);

    // ========================================================================
    // Test 6: B-Matrix (Strain-Displacement)
    // ========================================================================
    NXS_LOG_INFO("\n--- Test 6: B-Matrix ---");
    nxs::Real B[6 * 24];  // 6 x 24
    hex8.strain_displacement_matrix(xi_center, coords, B);

    NXS_LOG_INFO("B-matrix computed successfully (6x24)");
    NXS_LOG_INFO("Sample values (first column for node 0, DOF x):");
    for (int row = 0; row < 6; ++row) {
        NXS_LOG_INFO("  B[{}][0] = {:.6f}", row, B[row * 24 + 0]);
    }

    // ========================================================================
    // Test 7: Constitutive Matrix
    // ========================================================================
    NXS_LOG_INFO("\n--- Test 7: Constitutive Matrix ---");
    nxs::Real C[36];  // 6 x 6
    nxs::Real E = 200.0e9;  // Young's modulus (Pa)
    nxs::Real nu = 0.3;     // Poisson's ratio

    hex8.constitutive_matrix(E, nu, C);

    NXS_LOG_INFO("Constitutive matrix (first row):");
    NXS_LOG_INFO("  [{:.3e}  {:.3e}  {:.3e}  {:.3e}  {:.3e}  {:.3e}]",
                C[0], C[1], C[2], C[3], C[4], C[5]);

    // ========================================================================
    // Test 8: Stiffness Matrix (first few entries)
    // ========================================================================
    NXS_LOG_INFO("\n--- Test 8: Stiffness Matrix ---");
    nxs::Real K[24 * 24];  // 24 x 24
    hex8.stiffness_matrix(coords, E, nu, K);

    NXS_LOG_INFO("Stiffness matrix computed successfully (24x24)");
    NXS_LOG_INFO("K[0][0] = {:.6e} N/m", K[0]);
    NXS_LOG_INFO("K[1][1] = {:.6e} N/m", K[1*24 + 1]);
    NXS_LOG_INFO("K[0][1] = {:.6e} N/m", K[1]);

    // Check symmetry
    bool is_symmetric = true;
    for (int i = 0; i < 24 && is_symmetric; ++i) {
        for (int j = i+1; j < 24; ++j) {
            if (std::abs(K[i*24 + j] - K[j*24 + i]) > 1.0e-6) {
                is_symmetric = false;
                break;
            }
        }
    }
    NXS_LOG_INFO("Stiffness matrix symmetry: {}", is_symmetric ? "PASS" : "FAIL");

    // ========================================================================
    // Test 9: Point Location Test
    // ========================================================================
    NXS_LOG_INFO("\n--- Test 9: Point Location Test ---");

    // Test point at element center
    nxs::Real point_center[3] = {0.5, 0.5, 0.5};
    nxs::Real xi_found[3];
    bool inside = hex8.contains_point(coords, point_center, xi_found);

    NXS_LOG_INFO("Point (0.5, 0.5, 0.5):");
    NXS_LOG_INFO("  Inside element: {}", inside ? "YES" : "NO");
    if (inside) {
        NXS_LOG_INFO("  Natural coords: ({:.6f}, {:.6f}, {:.6f})",
                    xi_found[0], xi_found[1], xi_found[2]);
        NXS_LOG_INFO("  (Should be close to (0, 0, 0))");
    }

    // Test point outside element
    nxs::Real point_outside[3] = {2.0, 2.0, 2.0};
    inside = hex8.contains_point(coords, point_outside, xi_found);
    NXS_LOG_INFO("Point (2.0, 2.0, 2.0): Inside = {}", inside ? "YES" : "NO");

    // ========================================================================
    // Test 10: Internal Force (with dummy stress)
    // ========================================================================
    NXS_LOG_INFO("\n--- Test 10: Internal Force ---");
    nxs::Real disp[24] = {0};  // Zero displacements for this test
    nxs::Real stress[6] = {1.0e6, 1.0e6, 1.0e6, 0.0, 0.0, 0.0};  // 1 MPa hydrostatic stress
    nxs::Real fint[24];

    hex8.internal_force(coords, disp, stress, fint);

    NXS_LOG_INFO("Internal force (first 6 components):");
    for (int i = 0; i < 6; ++i) {
        NXS_LOG_INFO("  f_int[{}] = {:.6e} N", i, fint[i]);
    }

    // ========================================================================
    // Summary
    // ========================================================================
    NXS_LOG_INFO("\n=================================================");
    NXS_LOG_INFO("Hex8 Element Test Complete!");
    NXS_LOG_INFO("All tests executed successfully");
    NXS_LOG_INFO("=================================================");

    return 0;
}
