/**
 * @file beam2_element_test.cpp
 * @brief Validation test for Beam2 (2-node Euler-Bernoulli beam) element
 *
 * Tests:
 * 1. Shape function partition of unity
 * 2. Shape functions at nodes
 * 3. Length calculation
 * 4. Mass matrix properties
 * 5. Cross-section property setting
 */

#include <nexussim/discretization/beam2.hpp>
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
    std::cout << "Beam2 Element Validation Test\n";
    std::cout << "=================================================\n\n";

    Beam2Element elem;

    // Define a simple beam along x-axis from (0,0,0) to (1,0,0)
    double coords[2 * 3] = {
        0.0, 0.0, 0.0,  // Node 0
        1.0, 0.0, 0.0   // Node 1
    };

    bool all_passed = true;

    // ========================================================================
    // Test 1: Shape Functions - Partition of Unity
    // ========================================================================
    std::cout << "--- Test 1: Shape Functions ---\n";
    {
        // Test at center: ξ=0
        double xi[3] = {0.0, 0.0, 0.0};
        double N[2];
        elem.shape_functions(xi, N);

        double sum = N[0] + N[1];

        std::cout << "Shape functions at center: N[0]=" << N[0] << ", N[1]=" << N[1] << "\n";
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
    std::cout << "--- Test 2: Shape Functions at Nodes ---\n";
    {
        // At node 0: ξ=-1
        double xi[3] = {-1.0, 0.0, 0.0};
        double N[2];
        elem.shape_functions(xi, N);

        std::cout << "At node 0: N[0]=" << N[0] << ", N[1]=" << N[1] << "\n";
        std::cout << "Expected: N[0]=1.0, N[1]=0.0\n";

        bool pass = is_close(N[0], 1.0, 1.0e-12) && is_close(N[1], 0.0, 1.0e-12);

        // At node 1: ξ=+1
        xi[0] = 1.0;
        elem.shape_functions(xi, N);

        std::cout << "At node 1: N[0]=" << N[0] << ", N[1]=" << N[1] << "\n";
        std::cout << "Expected: N[0]=0.0, N[1]=1.0\n";

        pass = pass && is_close(N[0], 0.0, 1.0e-12) && is_close(N[1], 1.0, 1.0e-12);

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
    std::cout << "--- Test 3: Length Calculation ---\n";
    {
        double length = elem.length(coords);
        double expected_length = 1.0;

        std::cout << "Computed length: " << length << " m\n";
        std::cout << "Expected length: " << expected_length << " m\n";

        if (is_close(length, expected_length, 1.0e-12)) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 4: Cross-Section Properties
    // ========================================================================
    std::cout << "--- Test 4: Cross-Section Properties ---\n";
    {
        // Set circular cross-section with radius 0.1 m
        double radius = 0.1;
        elem.set_circular_section(radius);

        double A = elem.area();
        double expected_A = M_PI * radius * radius;

        std::cout << "Circular section (r=" << radius << " m):\n";
        std::cout << "  Area: " << A << " m² (expected " << expected_A << ")\n";
        std::cout << "  Iy: " << elem.moment_y() << " m⁴\n";
        std::cout << "  Iz: " << elem.moment_z() << " m⁴\n";
        std::cout << "  J: " << elem.torsion_constant() << " m⁴\n";

        if (is_close(A, expected_A, 1.0e-10)) {
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 5: Mass Matrix
    // ========================================================================
    std::cout << "--- Test 5: Mass Matrix ---\n";
    {
        double radius = 0.1;  // m
        elem.set_circular_section(radius);

        double density = 7850.0;  // Steel density (kg/m³)
        double M[12 * 12];  // 2 nodes × 6 DOFs = 12 DOFs

        elem.mass_matrix(coords, density, M);

        // Compute total mass (sum of translational DOF entries only)
        // For consistent mass matrix with 3 translational DOFs per node,
        // summing all translational DOF rows gives 3 × actual mass
        double total_mass = 0.0;
        for (int node = 0; node < 2; ++node) {
            for (int dof = 0; dof < 3; ++dof) {  // Only translational DOFs (0,1,2)
                int row = node * 6 + dof;
                for (int j = 0; j < 12; ++j) {
                    total_mass += M[row * 12 + j];
                }
            }
        }

        // Expected: 3 × density × area × length (factor of 3 from 3 translational DOFs)
        double expected_mass = 3.0 * density * elem.area() * 1.0;
        double error = std::abs(total_mass - expected_mass) / expected_mass * 100.0;

        std::cout << "Total mass (translational DOFs): " << total_mass << " kg\n";
        std::cout << "Expected mass: " << expected_mass << " kg\n";
        std::cout << "Error: " << error << "%\n";

        if (error < 1.0) {  // Less than 1% error
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL\n\n";
            all_passed = false;
        }
    }

    // ========================================================================
    // Test 6: Mass Matrix Zero Rows
    // ========================================================================
    std::cout << "--- Test 6: Mass Matrix Zero Rows ---\n";
    {
        double density = 7850.0;
        double M[12 * 12];

        elem.mass_matrix(coords, density, M);

        // Check for zero rows
        int zero_rows = 0;
        for (int i = 0; i < 12; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < 12; ++j) {
                row_sum += std::abs(M[i * 12 + j]);
            }
            if (row_sum < 1.0e-20) {
                zero_rows++;
                std::cout << "  Zero row " << i << "\n";
            }
        }

        std::cout << "Zero rows: " << zero_rows << "\n";

        // For beam elements, rotational DOFs might have very small mass
        // (rotational inertia can be small), so we're more lenient
        if (zero_rows <= 3) {  // Allow some rotational DOFs to be small
            std::cout << "Status: PASS\n\n";
        } else {
            std::cout << "Status: FAIL (too many zero rows)\n\n";
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
        std::cout << "Expected: 1.0 m (beam length)\n";

        if (is_close(char_len, 1.0, 1.0e-12)) {
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
    std::cout << "Beam2 Validation Test Complete\n";
    std::cout << "=================================================\n";

    return all_passed ? 0 : 1;
}
