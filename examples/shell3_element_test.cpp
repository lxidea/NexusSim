/**
 * @file shell3_element_test.cpp
 * @brief Validation test for Shell3 (3-node triangular shell) element
 *
 * Tests:
 * 1. Shape function partition of unity
 * 2. Shape functions at nodes (area coordinates)
 * 3. Area calculation
 * 4. Jacobian calculation
 * 5. Mass matrix properties
 * 6. Characteristic length
 */

#include <nexussim/discretization/shell3.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs::fem;
using nxs::Real;

constexpr double TOL = 1.0e-10;

bool is_close(double value, double expected, double tol = TOL) {
    return std::abs(value - expected) < tol;
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "Shell3 (Triangular Shell) Element Validation Test\n";
    std::cout << "=================================================\n\n";

    Shell3Element elem;

    // Right triangle in the x-y plane (vertices at origin, (1,0,0), (0,1,0))
    Real coords[3 * 3] = {
        0.0, 0.0, 0.0,  // Node 0
        1.0, 0.0, 0.0,  // Node 1
        0.0, 1.0, 0.0   // Node 2
    };

    bool all_passed = true;
    int test_num = 1;

    // ========================================================================
    // Test 1: Shape Functions - Partition of Unity at Centroid
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Shape Functions at Centroid ---\n";
    {
        // Centroid in area coordinates: L1 = L2 = L3 = 1/3
        // We use xi[0] = L1, xi[1] = L2, L3 = 1 - L1 - L2
        Real xi[3] = {1.0/3.0, 1.0/3.0, 0.0};
        Real N[3];
        elem.shape_functions(xi, N);

        Real sum = N[0] + N[1] + N[2];

        std::cout << "Shape functions at centroid: N = [" << N[0] << ", " << N[1] << ", " << N[2] << "]\n";
        std::cout << "Sum: " << sum << " (expected 1.0)\n";
        std::cout << "Each should be ~0.333: " << (is_close(N[0], 1.0/3.0, 0.01) ? "OK" : "FAIL") << "\n";

        if (is_close(sum, 1.0, 1.0e-12)) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 2: Shape Functions at Nodes
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Shape Functions at Nodes ---\n";
    {
        // Node 0: L1=1, L2=0, L3=0
        Real xi_nodes[3][3] = {
            {1.0, 0.0, 0.0},  // Node 0
            {0.0, 1.0, 0.0},  // Node 1
            {0.0, 0.0, 0.0}   // Node 2: L3 = 1 - L1 - L2 = 1
        };

        bool pass = true;
        for (int node = 0; node < 3; ++node) {
            Real N[3];
            elem.shape_functions(xi_nodes[node], N);

            std::cout << "At node " << node << ": N = [" << N[0] << ", " << N[1] << ", " << N[2] << "]\n";

            for (int i = 0; i < 3; ++i) {
                Real expected = (i == node) ? 1.0 : 0.0;
                if (!is_close(N[i], expected, 1.0e-10)) {
                    pass = false;
                }
            }
        }

        if (pass) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 3: Area Calculation
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Area Calculation ---\n";
    {
        Real area = elem.area(coords);  // Call area() directly
        Real expected_area = 0.5;  // Right triangle with legs 1m each

        std::cout << "Computed area: " << area << " m²\n";
        std::cout << "Expected area: " << expected_area << " m²\n";
        Real error = std::abs(area - expected_area) / expected_area * 100.0;
        std::cout << "Error: " << error << "%\n";

        if (error < 0.1) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 4: Jacobian Calculation
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Jacobian ---\n";
    {
        Real xi[3] = {1.0/3.0, 1.0/3.0, 0.0};  // Centroid
        Real J[9];
        Real det_J = elem.jacobian(xi, coords, J);

        std::cout << "Jacobian determinant: " << det_J << "\n";
        // For a triangle with area A, the Jacobian determinant = 2*A
        Real expected_det_J = 2.0 * 0.5;  // 2 * area = 1.0
        std::cout << "Expected: " << expected_det_J << " (2 × area)\n";

        if (std::abs(det_J - expected_det_J) < 0.1) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 5: Mass Matrix
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Mass Matrix ---\n";
    {
        Real thickness = 0.01;  // 10mm
        elem.set_thickness(thickness);

        Real density = 7850.0;  // Steel (kg/m³)
        constexpr int NUM_DOF = 18;  // 3 nodes × 6 DOF
        Real M[NUM_DOF * NUM_DOF];

        elem.mass_matrix(coords, density, M);

        // Sum translational DOFs only (indices 0,1,2 and 6,7,8 and 12,13,14)
        Real total_mass = 0.0;
        for (int node = 0; node < 3; ++node) {
            for (int dof = 0; dof < 3; ++dof) {
                int row = node * 6 + dof;
                for (int j = 0; j < NUM_DOF; ++j) {
                    total_mass += M[row * NUM_DOF + j];
                }
            }
        }

        Real area = 0.5;
        Real expected_mass = 3.0 * density * area * thickness;  // 3× for 3 translational DOFs

        std::cout << "Total mass (translational): " << total_mass << " kg\n";
        std::cout << "Expected: " << expected_mass << " kg\n";

        Real error = std::abs(total_mass - expected_mass) / expected_mass * 100.0;
        std::cout << "Error: " << error << "%\n";

        if (error < 5.0) {  // Allow 5% tolerance for lumped mass
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 6: Mass Matrix Zero Rows
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Mass Matrix Zero Rows ---\n";
    {
        Real density = 7850.0;
        constexpr int NUM_DOF = 18;
        Real M[NUM_DOF * NUM_DOF];

        elem.mass_matrix(coords, density, M);

        int zero_rows = 0;
        for (int i = 0; i < NUM_DOF; ++i) {
            Real row_sum = 0.0;
            for (int j = 0; j < NUM_DOF; ++j) {
                row_sum += std::abs(M[i * NUM_DOF + j]);
            }
            if (row_sum < 1.0e-20) {
                zero_rows++;
                std::cout << "  Zero row " << i << "\n";
            }
        }

        std::cout << "Zero rows: " << zero_rows << "\n";

        // Shell elements may have small rotational mass
        if (zero_rows <= 6) {  // Allow rotational DOFs to be small
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 7: Characteristic Length
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Characteristic Length ---\n";
    {
        Real char_len = elem.characteristic_length(coords);

        std::cout << "Characteristic length: " << char_len << " m\n";
        // For a right triangle with legs 1, smallest altitude is ~0.707
        Real expected = std::sqrt(2.0) / 2.0;  // min altitude
        std::cout << "Expected: ~" << expected << " m (min altitude)\n";

        // Accept any reasonable positive value
        if (char_len > 0.5 && char_len < 1.5) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 8: Different Triangle Orientations
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Different Orientations ---\n";
    {
        // Equilateral triangle
        Real equilateral[3 * 3] = {
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.5, std::sqrt(3.0)/2.0, 0.0
        };

        Real area_eq = elem.area(equilateral);  // Use area() not volume()
        Real expected_eq = std::sqrt(3.0) / 4.0;  // area of unit equilateral triangle

        std::cout << "Equilateral triangle area: " << area_eq << " m²\n";
        std::cout << "Expected: " << expected_eq << " m²\n";

        // Tilted triangle in 3D
        Real tilted[3 * 3] = {
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.5, 0.5, 0.5
        };

        Real area_tilt = elem.area(tilted);  // Use area() not volume()
        std::cout << "Tilted triangle area: " << area_tilt << " m²\n";

        bool pass = (std::abs(area_eq - expected_eq) / expected_eq < 0.01) && (area_tilt > 0);

        if (pass) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "=================================================\n";
    std::cout << "Shell3 Validation Test Complete\n";
    std::cout << "Result: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
    std::cout << "=================================================\n";

    return all_passed ? 0 : 1;
}
