/**
 * @file tet10_element_test.cpp
 * @brief Validation test for Tet10 element
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/discretization/tet10.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;

int main() {
    std::cout << "=================================================\n";
    std::cout << "Tet10 Element Validation Test\n";
    std::cout << "=================================================\n\n";

    Tet10Element elem;

    // Test 1: Shape Function Partition of Unity
    std::cout << "--- Test 1: Shape Functions ---\n";
    Real xi_center[3] = {0.25, 0.25, 0.25};  // Center of tet
    Real N[10];
    elem.shape_functions(xi_center, N);

    Real sum = 0.0;
    for (int i = 0; i < 10; ++i) {
        sum += N[i];
    }
    std::cout << "Sum of shape functions at center: " << sum << "\n";
    std::cout << "Expected: 1.0\n";
    std::cout << "Status: " << (std::abs(sum - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";

    // Test 2: Shape Functions at Vertices
    std::cout << "--- Test 2: Shape Functions at Corner Node ---\n";
    // In Tet10: xi=(L1,L2,L3), L4=1-L1-L2-L3
    // Node 0 is at L1=1: xi=(1,0,0)
    Real xi_node0[3] = {1.0, 0.0, 0.0};  // Corner node 0 at L1=1
    elem.shape_functions(xi_node0, N);
    std::cout << "At corner node 0: N[0]=" << N[0] << " (expected 1.0)\n";
    std::cout << "Status: " << (std::abs(N[0] - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";

    // Test 3: Volume Calculation
    std::cout << "--- Test 3: Volume Calculation ---\n";
    // Unit tetrahedron with mid-edge nodes
    // Tet10 node ordering (barycentric):
    // Node 0: L1=1 -> (1,0,0) in physical space
    // Node 1: L2=1 -> (0,1,0) in physical space
    // Node 2: L3=1 -> (0,0,1) in physical space
    // Node 3: L4=1 -> (0,0,0) in physical space
    Real coords[10 * 3] = {
        // Corner nodes (0-3)
        1.0, 0.0, 0.0,  // 0: at L1=1
        0.0, 1.0, 0.0,  // 1: at L2=1
        0.0, 0.0, 1.0,  // 2: at L3=1
        0.0, 0.0, 0.0,  // 3: at L4=1
        // Mid-edge nodes (4-9)
        0.5, 0.5, 0.0,  // 4: edge 0-1
        0.0, 0.5, 0.5,  // 5: edge 1-2
        0.5, 0.0, 0.5,  // 6: edge 2-0
        0.5, 0.0, 0.0,  // 7: edge 0-3
        0.0, 0.5, 0.0,  // 8: edge 1-3
        0.0, 0.0, 0.5   // 9: edge 2-3
    };

    Real volume = elem.volume(coords);
    Real expected_volume = 1.0 / 6.0;  // Unit tet volume
    std::cout << "Computed volume: " << volume << " m³\n";
    std::cout << "Expected volume: " << expected_volume << " m³\n";
    std::cout << "Error: " << std::abs(volume - expected_volume) / expected_volume * 100.0 << "%\n";
    std::cout << "Status: " << (std::abs(volume - expected_volume) / expected_volume < 0.01 ? "PASS" : "FAIL") << "\n\n";

    // Test 4: Mass Matrix
    std::cout << "--- Test 4: Mass Matrix ---\n";
    Real density = 1000.0;  // kg/m³
    Real M[30 * 30];  // 10 nodes * 3 DOF = 30 DOF
    elem.mass_matrix(coords, density, M);

    Real total_mass = 0.0;
    for (int i = 0; i < 30; ++i) {
        for (int j = 0; j < 30; ++j) {
            total_mass += M[i * 30 + j];
        }
    }

    Real expected_mass = density * volume;
    std::cout << "Total mass: " << total_mass << " kg\n";
    std::cout << "Expected mass: " << expected_mass << " kg\n";
    std::cout << "Error: " << std::abs(total_mass - expected_mass) / expected_mass * 100.0 << "%\n";
    std::cout << "Status: " << (std::abs(total_mass - expected_mass) / expected_mass < 0.01 ? "PASS" : "FAIL") << "\n\n";

    // Test 5: Check for Zero Rows
    std::cout << "--- Test 5: Mass Matrix Zero Rows ---\n";
    int zero_rows = 0;
    for (int i = 0; i < 30; ++i) {
        Real row_sum = 0.0;
        for (int j = 0; j < 30; ++j) {
            row_sum += std::abs(M[i * 30 + j]);
        }
        if (row_sum < 1e-10) {
            std::cout << "WARNING: Row " << i << " is zero!\n";
            zero_rows++;
        }
    }
    std::cout << "Zero rows: " << zero_rows << "\n";
    std::cout << "Status: " << (zero_rows == 0 ? "PASS" : "FAIL") << "\n\n";

    // Test 6: Jacobian
    std::cout << "--- Test 6: Jacobian at Center ---\n";
    Real J[9];
    Real det_J = elem.jacobian(xi_center, coords, J);
    std::cout << "Jacobian determinant: " << det_J << "\n";
    std::cout << "Expected: ~" << 6.0 * volume << " (6 * volume)\n";
    std::cout << "Status: " << (std::abs(det_J - 6.0 * volume) < 1e-6 ? "PASS" : "FAIL") << "\n\n";

    std::cout << "=================================================\n";
    std::cout << "Tet10 Validation Test Complete\n";
    std::cout << "=================================================\n";

    return 0;
}
