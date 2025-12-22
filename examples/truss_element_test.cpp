/**
 * @file truss_element_test.cpp
 * @brief Validation test for Truss (2-node axial bar) element
 *
 * Tests:
 * 1. Shape function partition of unity
 * 2. Shape functions at nodes
 * 3. Length calculation
 * 4. Axial strain computation
 * 5. Mass matrix properties
 * 6. Stiffness matrix properties
 * 7. Different orientations in 3D
 */

#include <nexussim/discretization/truss.hpp>
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
    std::cout << "Truss (2-Node Axial Bar) Element Validation Test\n";
    std::cout << "=================================================\n\n";

    TrussElement elem;

    // Simple truss along x-axis from (0,0,0) to (1,0,0)
    Real coords[2 * 3] = {
        0.0, 0.0, 0.0,  // Node 0
        1.0, 0.0, 0.0   // Node 1
    };

    bool all_passed = true;
    int test_num = 1;

    // ========================================================================
    // Test 1: Shape Functions - Partition of Unity
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Shape Functions at Center ---\n";
    {
        Real xi[3] = {0.0, 0.0, 0.0};  // Center
        Real N[2];
        elem.shape_functions(xi, N);

        Real sum = N[0] + N[1];

        std::cout << "Shape functions at center: N = [" << N[0] << ", " << N[1] << "]\n";
        std::cout << "Sum: " << sum << " (expected 1.0)\n";

        if (is_close(sum, 1.0, 1.0e-12) && is_close(N[0], 0.5) && is_close(N[1], 0.5)) {
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
        Real xi_node0[3] = {-1.0, 0.0, 0.0};
        Real xi_node1[3] = { 1.0, 0.0, 0.0};
        Real N[2];

        elem.shape_functions(xi_node0, N);
        std::cout << "At node 0 (xi=-1): N = [" << N[0] << ", " << N[1] << "]\n";
        bool pass = is_close(N[0], 1.0) && is_close(N[1], 0.0);

        elem.shape_functions(xi_node1, N);
        std::cout << "At node 1 (xi=+1): N = [" << N[0] << ", " << N[1] << "]\n";
        pass = pass && is_close(N[0], 0.0) && is_close(N[1], 1.0);

        if (pass) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 3: Length Calculation
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Length Calculation ---\n";
    {
        Real length = elem.length(coords);

        std::cout << "Computed length: " << length << " m\n";
        std::cout << "Expected: 1.0 m\n";

        if (is_close(length, 1.0, 1.0e-12)) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 4: Axial Strain Computation
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Axial Strain ---\n";
    {
        // Displacements: node 1 moves +0.01 in x direction (1% extension)
        Real disp[6] = {
            0.0, 0.0, 0.0,   // Node 0: no displacement
            0.01, 0.0, 0.0   // Node 1: +0.01 in x
        };

        Real strain = elem.axial_strain(coords, disp);
        Real expected_strain = 0.01;  // 1% strain

        std::cout << "Axial displacement: " << disp[3] << " m\n";
        std::cout << "Computed strain: " << strain << "\n";
        std::cout << "Expected strain: " << expected_strain << " (ε = δL/L = 0.01/1.0)\n";

        if (is_close(strain, expected_strain, 1.0e-10)) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 5: Cross-Section Area
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Cross-Section Area ---\n";
    {
        Real area = 0.001;  // 10 cm² = 0.001 m²
        elem.set_area(area);

        // Volume should be A × L
        Real volume = elem.volume(coords);
        Real expected = area * 1.0;

        std::cout << "Set area: " << area << " m²\n";
        std::cout << "Computed volume: " << volume << " m³\n";
        std::cout << "Expected: " << expected << " m³ (A × L)\n";

        if (is_close(volume, expected, 1.0e-12)) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 6: Mass Matrix
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Mass Matrix ---\n";
    {
        Real area = 0.001;  // m²
        Real density = 7850.0;  // Steel kg/m³
        elem.set_area(area);

        constexpr int NUM_DOF = 6;  // 2 nodes × 3 DOF
        Real M[NUM_DOF * NUM_DOF];
        elem.mass_matrix(coords, density, M);

        // Total mass should equal ρ × A × L
        Real total_mass = 0.0;
        for (int i = 0; i < NUM_DOF; ++i) {
            for (int j = 0; j < NUM_DOF; ++j) {
                total_mass += M[i * NUM_DOF + j];
            }
        }

        // Expected: 3 × ρAL (factor of 3 for 3 translational DOFs summed)
        Real expected_mass = 3.0 * density * area * 1.0;

        std::cout << "Total mass (all DOFs): " << total_mass << " kg\n";
        std::cout << "Expected: " << expected_mass << " kg\n";

        Real error = std::abs(total_mass - expected_mass) / expected_mass * 100.0;
        std::cout << "Error: " << error << "%\n";

        if (error < 1.0) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 7: Mass Matrix - Lumped (Diagonal)
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Lumped Mass ---\n";
    {
        Real area = 0.001;
        Real density = 7850.0;
        elem.set_area(area);

        constexpr int NUM_DOF = 6;
        Real M[NUM_DOF * NUM_DOF];
        elem.mass_matrix(coords, density, M);

        // For lumped mass, diagonal entries should be ρAL/2 for each DOF
        Real expected_diag = density * area * 1.0 / 2.0;

        std::cout << "Diagonal entries:\n";
        bool diag_ok = true;
        for (int i = 0; i < NUM_DOF; ++i) {
            std::cout << "  M[" << i << "," << i << "] = " << M[i * NUM_DOF + i] << "\n";
            if (!is_close(M[i * NUM_DOF + i], expected_diag, expected_diag * 0.01)) {
                diag_ok = false;
            }
        }

        std::cout << "Expected diagonal: " << expected_diag << " kg\n";

        if (diag_ok) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 8: Stiffness Matrix
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Stiffness Matrix ---\n";
    {
        Real area = 0.001;  // m²
        Real E = 2.1e11;    // Steel Young's modulus (Pa)
        Real nu = 0.3;      // Poisson's ratio (not used for truss)
        elem.set_area(area);

        constexpr int NUM_DOF = 6;
        Real K[NUM_DOF * NUM_DOF];
        elem.stiffness_matrix(coords, E, nu, K);

        // For truss along x-axis, K should have non-zero entries only in x-DOFs
        // K_local = (EA/L) * [1, -1; -1, 1]

        Real k_axial = E * area / 1.0;  // EA/L

        std::cout << "Expected axial stiffness EA/L: " << k_axial << " N/m\n";
        std::cout << "K[0,0] (x-stiffness node 0): " << K[0] << "\n";
        std::cout << "K[0,3] (coupling node 0 to 1): " << K[0 * NUM_DOF + 3] << "\n";
        std::cout << "K[3,3] (x-stiffness node 1): " << K[3 * NUM_DOF + 3] << "\n";

        // Check key entries
        bool pass = true;
        pass = pass && is_close(K[0], k_axial, k_axial * 1e-10);           // K[0,0]
        pass = pass && is_close(K[3 * NUM_DOF + 3], k_axial, k_axial * 1e-10); // K[3,3]
        pass = pass && is_close(K[0 * NUM_DOF + 3], -k_axial, k_axial * 1e-10); // K[0,3]
        pass = pass && is_close(K[3 * NUM_DOF + 0], -k_axial, k_axial * 1e-10); // K[3,0]

        // Check symmetry
        for (int i = 0; i < NUM_DOF; ++i) {
            for (int j = 0; j < NUM_DOF; ++j) {
                if (!is_close(K[i * NUM_DOF + j], K[j * NUM_DOF + i], 1.0e-10)) {
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
    // Test 9: 3D Orientation (Diagonal truss)
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": 3D Diagonal Truss ---\n";
    {
        // Truss from (0,0,0) to (1,1,1) - length = sqrt(3)
        Real coords_3d[6] = {
            0.0, 0.0, 0.0,
            1.0, 1.0, 1.0
        };

        Real length = elem.length(coords_3d);
        Real expected_length = std::sqrt(3.0);

        std::cout << "3D truss from (0,0,0) to (1,1,1)\n";
        std::cout << "Computed length: " << length << " m\n";
        std::cout << "Expected: " << expected_length << " m\n";

        // Test axial strain: equal extension in all directions
        Real disp_3d[6] = {
            0.0, 0.0, 0.0,
            0.01, 0.01, 0.01  // Extension along the bar direction
        };

        Real strain = elem.axial_strain(coords_3d, disp_3d);
        // Projection of displacement onto bar direction
        Real delta = (0.01 + 0.01 + 0.01) / std::sqrt(3.0);
        Real expected_strain = delta / expected_length;

        std::cout << "Computed strain: " << strain << "\n";
        std::cout << "Expected strain: " << expected_strain << "\n";

        bool pass = is_close(length, expected_length, 1e-10) &&
                    is_close(strain, expected_strain, 1e-8);

        if (pass) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 10: Characteristic Length
    // ========================================================================
    std::cout << "--- Test " << test_num++ << ": Characteristic Length ---\n";
    {
        Real char_len = elem.characteristic_length(coords);

        std::cout << "Characteristic length: " << char_len << " m\n";
        std::cout << "Expected: 1.0 m (bar length)\n";

        if (is_close(char_len, 1.0, 1e-12)) {
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
    std::cout << "Truss Element Validation Test Complete\n";
    std::cout << "Result: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
    std::cout << "=================================================\n";

    return all_passed ? 0 : 1;
}
