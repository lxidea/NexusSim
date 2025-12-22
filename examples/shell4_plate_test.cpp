/**
 * @file shell4_plate_test.cpp
 * @brief Validation test for Shell4 element: flat plate bending
 *
 * Tests a flat plate under uniform load using Shell4 elements.
 * Verifies shape functions, Jacobian, and basic element properties.
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/discretization/shell4.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;

int main() {
    std::cout << "=================================================\n";
    std::cout << "Shell4 Flat Plate Test\n";
    std::cout << "=================================================\n\n";

    Shell4Element elem;

    // Test 1: Shape Function Partition of Unity
    std::cout << "--- Test 1: Shape Functions ---\n";
    Real xi_center[3] = {0.0, 0.0, 0.0};  // Center of shell element
    Real N[4];
    elem.shape_functions(xi_center, N);

    Real sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        sum += N[i];
        std::cout << "N[" << i << "] = " << N[i] << "\n";
    }
    std::cout << "Sum of shape functions at center: " << sum << "\n";
    std::cout << "Expected: 1.0\n";
    std::cout << "Status: " << (std::abs(sum - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";

    // Test 2: Shape Functions at Corners
    std::cout << "--- Test 2: Shape Functions at Corners ---\n";
    Real xi_corners[4][3] = {
        {-1.0, -1.0, 0.0},  // Node 0
        { 1.0, -1.0, 0.0},  // Node 1
        { 1.0,  1.0, 0.0},  // Node 2
        {-1.0,  1.0, 0.0}   // Node 3
    };

    bool corner_test_pass = true;
    for (int corner = 0; corner < 4; ++corner) {
        elem.shape_functions(xi_corners[corner], N);
        for (int i = 0; i < 4; ++i) {
            Real expected = (i == corner) ? 1.0 : 0.0;
            if (std::abs(N[i] - expected) > 1e-10) {
                std::cout << "FAIL: At corner " << corner << ", N[" << i << "] = " << N[i]
                          << " (expected " << expected << ")\n";
                corner_test_pass = false;
            }
        }
    }
    std::cout << "Corner test: " << (corner_test_pass ? "PASS" : "FAIL") << "\n\n";

    // Test 3: Area Calculation
    std::cout << "--- Test 3: Area Calculation ---\n";
    // Unit square shell: 1m x 1m
    Real coords[4 * 3] = {
        0.0, 0.0, 0.0,  // Node 0
        1.0, 0.0, 0.0,  // Node 1
        1.0, 1.0, 0.0,  // Node 2
        0.0, 1.0, 0.0   // Node 3
    };

    Real area = elem.volume(coords);  // volume() returns area for shells
    Real expected_area = 1.0;  // 1m x 1m
    std::cout << "Computed area: " << area << " m²\n";
    std::cout << "Expected area: " << expected_area << " m²\n";
    std::cout << "Error: " << std::abs(area - expected_area) / expected_area * 100.0 << "%\n";
    std::cout << "Status: " << (std::abs(area - expected_area) / expected_area < 0.01 ? "PASS" : "FAIL") << "\n\n";

    // Test 4: Jacobian at Center
    std::cout << "--- Test 4: Jacobian at Center ---\n";
    Real J[9];
    Real det_J = elem.jacobian(xi_center, coords, J);
    std::cout << "Jacobian determinant: " << det_J << "\n";
    std::cout << "Expected: 0.25 (for 1x1 element mapped from [-1,1]²)\n";
    std::cout << "Status: " << (std::abs(det_J - 0.25) < 1e-6 ? "PASS" : "FAIL") << "\n\n";

    // Test 5: Mass Matrix
    std::cout << "--- Test 5: Mass Matrix ---\n";
    Real thickness = 0.01;  // 10mm thick shell
    elem.set_thickness(thickness);
    Real density = 1000.0;  // kg/m³ (volumetric density)
    constexpr int NUM_DOF = 24;  // 4 nodes * 6 DOF/node (3 trans + 3 rot)
    Real M[NUM_DOF * NUM_DOF];  // 24x24 mass matrix
    elem.mass_matrix(coords, density, M);

    Real total_mass = 0.0;
    for (int i = 0; i < NUM_DOF; ++i) {
        for (int j = 0; j < NUM_DOF; ++j) {
            total_mass += M[i * NUM_DOF + j];
        }
    }

    Real expected_mass = density * area * thickness;
    std::cout << "Total mass: " << total_mass << " kg\n";
    std::cout << "Expected mass: " << expected_mass << " kg\n";
    std::cout << "Error: " << std::abs(total_mass - expected_mass) / expected_mass * 100.0 << "%\n";
    std::cout << "Status: " << (std::abs(total_mass - expected_mass) / expected_mass < 0.01 ? "PASS" : "FAIL") << "\n\n";

    // Test 6: Check for Zero Rows
    std::cout << "--- Test 6: Mass Matrix Zero Rows ---\n";
    int zero_rows = 0;
    for (int i = 0; i < NUM_DOF; ++i) {
        Real row_sum = 0.0;
        for (int j = 0; j < NUM_DOF; ++j) {
            row_sum += std::abs(M[i * NUM_DOF + j]);
        }
        if (row_sum < 1e-10) {
            std::cout << "WARNING: Row " << i << " is zero!\n";
            zero_rows++;
        }
    }
    std::cout << "Zero rows: " << zero_rows << "\n";
    std::cout << "Status: " << (zero_rows == 0 ? "PASS" : "FAIL") << "\n\n";

    // Test 7: Characteristic Length
    std::cout << "--- Test 7: Characteristic Length ---\n";
    Real char_length = elem.characteristic_length(coords);
    std::cout << "Characteristic length: " << char_length << " m\n";
    std::cout << "Expected: ~1.0 m (edge length)\n";
    std::cout << "Status: " << (std::abs(char_length - 1.0) < 0.1 ? "PASS" : "FAIL") << "\n\n";

    std::cout << "=================================================\n";
    std::cout << "Shell4 Validation Test Complete\n";
    std::cout << "=================================================\n";

    return 0;
}
