/**
 * @file wedge6_element_test.cpp
 * @brief Validation test for Wedge6 (6-node prism) element
 *
 * Tests:
 * 1. Shape function partition of unity
 * 2. Shape functions at nodes
 * 3. Volume calculation
 * 4. Mass matrix properties
 * 5. Jacobian computation
 */

#include <nexussim/discretization/wedge6.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs::fem;

// Test tolerance
constexpr double TOL = 1.0e-10;

// Helper function to check if value is close to expected
bool is_close(double value, double expected, double tol = TOL) {
    return std::abs(value - expected) < tol;
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "Wedge6 Element Validation Test\n";
    std::cout << "=================================================\n\n";

    Wedge6Element elem;

    // Define a unit wedge element:
    // Bottom triangle: (0,0,0), (1,0,0), (0,1,0)
    // Top triangle: (0,0,1), (1,0,1), (0,1,1)
    double coords[6 * 3] = {
        // Bottom triangle (ζ = -1 maps to z = 0)
        1.0, 0.0, 0.0,  // Node 0
        0.0, 1.0, 0.0,  // Node 1
        0.0, 0.0, 0.0,  // Node 2
        // Top triangle (ζ = +1 maps to z = 1)
        1.0, 0.0, 1.0,  // Node 3
        0.0, 1.0, 1.0,  // Node 4
        0.0, 0.0, 1.0   // Node 5
    };

    bool all_passed = true;

    // ========================================================================
    // Test 1: Shape Functions - Partition of Unity
    // ========================================================================
    std::cout << "--- Test 1: Shape Functions ---\n";
    {
        // Test at center: ξ=1/3, η=1/3, ζ=0
        double xi[3] = {1.0/3.0, 1.0/3.0, 0.0};
        double N[6];
        elem.shape_functions(xi, N);

        double sum = 0.0;
        for (int i = 0; i < 6; ++i) {
            sum += N[i];
        }

        std::cout << "Sum of shape functions at center: " << sum << "\n";
        std::cout << "Expected: 1.0\n";

        if (is_close(sum, 1.0, 1.0e-12)) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 2: Shape Functions at Corner Nodes
    // ========================================================================
    std::cout << "--- Test 2: Shape Functions at Corner Nodes ---\n";
    {
        // Test all 6 corner nodes
        // Note: Natural coords (ξ,η,ζ) map to shape functions with offset
        double natural_coords[6][3] = {
            {1.0, 0.0, -1.0},  // Node 0: expect N[1]=1
            {0.0, 1.0, -1.0},  // Node 1: expect N[2]=1
            {0.0, 0.0, -1.0},  // Node 2: expect N[0]=1
            {1.0, 0.0, +1.0},  // Node 3: expect N[4]=1
            {0.0, 1.0, +1.0},  // Node 4: expect N[5]=1
            {0.0, 0.0, +1.0}   // Node 5: expect N[3]=1
        };
        int expected_idx[6] = {1, 2, 0, 4, 5, 3};

        bool pass = true;
        for (int node = 0; node < 6; ++node) {
            double N[6];
            elem.shape_functions(natural_coords[node], N);

            int expected = expected_idx[node];
            if (!is_close(N[expected], 1.0, 1.0e-12)) {
                pass = false;
                std::cout << "Node " << node << ": N[" << expected << "]=" << N[expected]
                          << " (expected 1.0)\n";
            }
            for (int i = 0; i < 6; ++i) {
                if (i != expected && !is_close(N[i], 0.0, 1.0e-12)) {
                    pass = false;
                    std::cout << "Node " << node << ": N[" << i << "]=" << N[i]
                              << " (expected 0.0)\n";
                }
            }
        }

        if (pass) {
            std::cout << "All corner nodes: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 3: Volume Calculation
    // ========================================================================
    std::cout << "--- Test 3: Volume Calculation ---\n";
    {
        double volume = elem.volume(coords);

        // Expected volume: 1/2 base area × height = 0.5 * (1/2 * 1 * 1) * 1 = 0.25
        // The triangle has vertices (0,0), (1,0), (0,1) -> area = 0.5
        // Height = 1
        // Volume = 0.5 * 1 = 0.5
        double expected_volume = 0.5;
        double error = std::abs(volume - expected_volume) / expected_volume * 100.0;

        std::cout << "Computed volume: " << volume << " m³\n";
        std::cout << "Expected volume: " << expected_volume << " m³\n";
        std::cout << "Error: " << error << "%\n";

        if (error < 0.01) {  // Less than 0.01% error
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 4: Mass Matrix
    // ========================================================================
    std::cout << "--- Test 4: Mass Matrix ---\n";
    {
        double density = 1000.0;  // kg/m³
        double M[18 * 18];  // 6 nodes × 3 DOFs = 18 DOFs

        elem.mass_matrix(coords, density, M);

        // Compute total mass by summing all matrix entries
        // Note: For a mass matrix with NUM_DIMS DOFs per node,
        // sum of all entries = NUM_DIMS × (density × volume)
        double total_mass = 0.0;
        for (int i = 0; i < 18 * 18; ++i) {
            total_mass += M[i];
        }

        // Expected: NUM_DIMS × density × volume = 3 × 1000 × 0.5 = 1500 kg
        double expected_mass = 3.0 * density * 0.5;
        double error = std::abs(total_mass - expected_mass) / expected_mass * 100.0;

        std::cout << "Total mass: " << total_mass << " kg\n";
        std::cout << "Expected mass: " << expected_mass << " kg\n";
        std::cout << "Error: " << error << "%\n";

        if (error < 0.01) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 5: Mass Matrix Zero Rows
    // ========================================================================
    std::cout << "--- Test 5: Mass Matrix Zero Rows ---\n";
    {
        double density = 1000.0;
        double M[18 * 18];

        elem.mass_matrix(coords, density, M);

        // Check for zero or negative row sums (lumped mass)
        int zero_rows = 0;
        for (int i = 0; i < 18; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < 18; ++j) {
                row_sum += M[i * 18 + j];
            }
            if (row_sum <= 1.0e-10) {
                zero_rows++;
            }
        }

        std::cout << "Zero rows: " << zero_rows << "\n";

        if (zero_rows == 0) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL (has zero/negative mass DOFs)\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 6: Jacobian at Center
    // ========================================================================
    std::cout << "--- Test 6: Jacobian at Center ---\n";
    {
        double xi[3] = {1.0/3.0, 1.0/3.0, 0.0};
        double J[9];
        double det_J = elem.jacobian(xi, coords, J);

        std::cout << "Jacobian determinant: " << det_J << "\n";
        std::cout << "Expected: ~0.25 (1/4 for this geometry)\n";

        if (det_J > 0.0 && det_J < 1.0) {  // Reasonable range
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 7: Characteristic Length
    // ========================================================================
    std::cout << "--- Test 7: Characteristic Length ---\n";
    {
        double char_len = elem.characteristic_length(coords);

        std::cout << "Characteristic length: " << char_len << " m\n";
        std::cout << "Expected: ~1.0 m (edge length)\n";

        if (char_len > 0.5 && char_len < 2.0) {  // Reasonable range
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
    std::cout << "Wedge6 Validation Test Complete\n";
    std::cout << "=================================================\n";

    return all_passed ? 0 : 1;
}
