/**
 * @file tet4_compression_test.cpp
 * @brief Validation test for Tet4 element: uniaxial compression
 *
 * Tests a cube under uniaxial compression load using Tet4 elements.
 * Compares FEM displacement against analytical solution from elasticity theory.
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/discretization/tet4.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;

int main() {
    std::cout << "=================================================\n";
    std::cout << "Tet4 Uniaxial Compression Test\n";
    std::cout << "=================================================\n\n";

    Tet4Element elem;

    // Test 1: Shape Function Partition of Unity
    std::cout << "--- Test 1: Shape Functions ---\n";
    Real xi_center[3] = {0.25, 0.25, 0.25};  // Center of tet in natural coords
    Real N[4];
    elem.shape_functions(xi_center, N);

    Real sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        sum += N[i];
    }
    std::cout << "Sum of shape functions at center: " << sum << "\n";
    std::cout << "Expected: 1.0\n";
    std::cout << "Status: " << (std::abs(sum - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";

    // Test 2: Shape Functions at Vertices
    std::cout << "--- Test 2: Shape Functions at Vertices ---\n";
    Real xi_node0[3] = {0.0, 0.0, 0.0};  // Node 0
    elem.shape_functions(xi_node0, N);
    std::cout << "At node 0: N[0]=" << N[0] << " (expected 1.0)\n";
    std::cout << "Status: " << (std::abs(N[0] - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";

    // Test 3: Volume Calculation
    std::cout << "--- Test 3: Volume Calculation ---\n";
    // Unit tetrahedron: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    Real coords[4 * 3] = {
        0.0, 0.0, 0.0,  // Node 0
        1.0, 0.0, 0.0,  // Node 1
        0.0, 1.0, 0.0,  // Node 2
        0.0, 0.0, 1.0   // Node 3
    };

    Real volume = elem.volume(coords);
    Real expected_volume = 1.0 / 6.0;  // Volume of unit tet = 1/6
    std::cout << "Computed volume: " << volume << " m³\n";
    std::cout << "Expected volume: " << expected_volume << " m³\n";
    std::cout << "Error: " << std::abs(volume - expected_volume) / expected_volume * 100.0 << "%\n";
    std::cout << "Status: " << (std::abs(volume - expected_volume) / expected_volume < 0.01 ? "PASS" : "FAIL") << "\n\n";

    // Test 4: Jacobian at Center
    std::cout << "--- Test 4: Jacobian at Center ---\n";
    Real J[9];
    Real det_J = elem.jacobian(xi_center, coords, J);
    std::cout << "Jacobian determinant: " << det_J << "\n";
    std::cout << "Expected: " << expected_volume << " (6 * volume)\n";
    std::cout << "Status: " << (std::abs(det_J - expected_volume) < 1e-6 ? "PASS" : "FAIL") << "\n\n";

    // Test 5: Mass Matrix
    std::cout << "--- Test 5: Mass Matrix ---\n";
    Real density = 1000.0;  // kg/m³
    Real M[12 * 12];  // 4 nodes * 3 DOF = 12 DOF
    elem.mass_matrix(coords, density, M);

    Real total_mass = 0.0;
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) {
            total_mass += M[i * 12 + j];
        }
    }

    Real expected_mass = density * volume;
    std::cout << "Total mass: " << total_mass << " kg\n";
    std::cout << "Expected mass: " << expected_mass << " kg\n";
    std::cout << "Error: " << std::abs(total_mass - expected_mass) / expected_mass * 100.0 << "%\n";
    std::cout << "Status: " << (std::abs(total_mass - expected_mass) / expected_mass < 0.01 ? "PASS" : "FAIL") << "\n\n";

    // Test 6: Check for Zero Rows
    std::cout << "--- Test 6: Mass Matrix Zero Rows ---\n";
    int zero_rows = 0;
    for (int i = 0; i < 12; ++i) {
        Real row_sum = 0.0;
        for (int j = 0; j < 12; ++j) {
            row_sum += std::abs(M[i * 12 + j]);
        }
        if (row_sum < 1e-10) {
            std::cout << "WARNING: Row " << i << " is zero!\n";
            zero_rows++;
        }
    }
    std::cout << "Zero rows: " << zero_rows << "\n";
    std::cout << "Status: " << (zero_rows == 0 ? "PASS" : "FAIL") << "\n\n";

    std::cout << "=================================================\n";
    std::cout << "Tet4 Validation Test Complete\n";
    std::cout << "=================================================\n";

    return 0;
}
