/**
 * @file hex20_single_element_test.cpp
 * @brief Single element test for Hex20 to verify mass matrix computation
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/discretization/hex20.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;

int main() {
    std::cout << "=================================================\n";
    std::cout << "Hex20 Single Element Test\n";
    std::cout << "=================================================\n\n";

    Hex20Element elem;

    // Create a simple unit cube with 20 nodes
    // Corner nodes at ±1 in each direction
    // Mid-edge nodes at 0 along edges
    Real coords[20 * 3] = {
        // Corner nodes (0-7)
        0.0, 0.0, 0.0,  // 0: origin
        1.0, 0.0, 0.0,  // 1
        1.0, 1.0, 0.0,  // 2
        0.0, 1.0, 0.0,  // 3
        0.0, 0.0, 1.0,  // 4
        1.0, 0.0, 1.0,  // 5
        1.0, 1.0, 1.0,  // 6
        0.0, 1.0, 1.0,  // 7
        // Bottom face mid-edge nodes (8-11)
        0.5, 0.0, 0.0,  // 8: edge 0-1
        1.0, 0.5, 0.0,  // 9: edge 1-2
        0.5, 1.0, 0.0,  // 10: edge 2-3
        0.0, 0.5, 0.0,  // 11: edge 3-0
        // Vertical mid-edge nodes (12-15)
        0.0, 0.0, 0.5,  // 12: edge 0-4
        1.0, 0.0, 0.5,  // 13: edge 1-5
        1.0, 1.0, 0.5,  // 14: edge 2-6
        0.0, 1.0, 0.5,  // 15: edge 3-7
        // Top face mid-edge nodes (16-19)
        0.5, 0.0, 1.0,  // 16: edge 4-5
        1.0, 0.5, 1.0,  // 17: edge 5-6
        0.5, 1.0, 1.0,  // 18: edge 6-7
        0.0, 0.5, 1.0   // 19: edge 7-4
    };

    std::cout << "--- Test 1: Shape Functions at Center ---\n";
    Real xi_center[3] = {0.0, 0.0, 0.0};
    Real N[20];
    elem.shape_functions(xi_center, N);

    Real sum = 0.0;
    for (int i = 0; i < 20; ++i) {
        sum += N[i];
    }
    std::cout << "Sum of shape functions at center: " << sum << "\n";
    std::cout << "Expected: 1.0\n";
    std::cout << "Status: " << (std::abs(sum - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";

    std::cout << "--- Test 2: Shape Functions at Corner ---\n";
    Real xi_corner[3] = {-1.0, -1.0, -1.0};  // Node 0
    elem.shape_functions(xi_corner, N);

    std::cout << "Shape functions at ξ=(-1,-1,-1) (should be 1 at node 0, 0 elsewhere):\n";
    for (int i = 0; i < 20; ++i) {
        if (std::abs(N[i]) > 1e-10) {
            std::cout << "  N[" << i << "] = " << N[i] << "\n";
        }
    }
    std::cout << "Expected: N[0] = 1.0\n";
    std::cout << "Status: " << (std::abs(N[0] - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";

    std::cout << "--- Test 3: Jacobian at Center ---\n";
    Real J[9];
    Real det_J = elem.jacobian(xi_center, coords, J);

    std::cout << "Jacobian determinant at center: " << det_J << "\n";
    std::cout << "Expected: 0.125 (volume of unit cube in natural coords)\n";
    std::cout << "Status: " << (std::abs(det_J - 0.125) < 1e-6 ? "PASS" : "FAIL") << "\n\n";

    std::cout << "Jacobian matrix:\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [ ";
        for (int j = 0; j < 3; ++j) {
            std::cout << std::setw(10) << J[i * 3 + j] << " ";
        }
        std::cout << "]\n";
    }
    std::cout << "\n";

    std::cout << "--- Test 4: Mass Matrix Computation ---\n";
    Real density = 1000.0;  // kg/m³
    Real M[60 * 60];
    elem.mass_matrix(coords, density, M);

    // Check total mass
    // NOTE: For consistent mass matrix in 3D, sum(M) = 3 × ρ × V
    // This is because each of the 3 DOFs (x,y,z) contributes independently
    Real total_mass_sum = 0.0;
    for (int i = 0; i < 60; ++i) {
        for (int j = 0; j < 60; ++j) {
            total_mass_sum += M[i * 60 + j];
        }
    }

    Real volume = 1.0 * 1.0 * 1.0;  // Unit cube
    Real expected_mass = density * volume;
    Real expected_mass_sum = 3.0 * expected_mass;  // 3 DOFs per node

    std::cout << "Total sum of mass matrix: " << total_mass_sum << " kg\n";
    std::cout << "Expected sum (3 × ρ × V): " << expected_mass_sum << " kg\n";
    std::cout << "Error: " << std::abs(total_mass_sum - expected_mass_sum) / expected_mass_sum * 100.0 << "%\n";
    std::cout << "Status: " << (std::abs(total_mass_sum - expected_mass_sum) / expected_mass_sum < 0.01 ? "PASS" : "FAIL") << "\n\n";

    // Check for zero rows (indicates problem)
    std::cout << "Checking for zero rows in mass matrix...\n";
    int zero_rows = 0;
    for (int i = 0; i < 60; ++i) {
        Real row_sum = 0.0;
        for (int j = 0; j < 60; ++j) {
            row_sum += std::abs(M[i * 60 + j]);
        }
        if (row_sum < 1e-10) {
            std::cout << "  WARNING: Row " << i << " is zero!\n";
            zero_rows++;
        }
    }

    if (zero_rows == 0) {
        std::cout << "  No zero rows found - GOOD!\n";
    } else {
        std::cout << "  Found " << zero_rows << " zero rows - PROBLEM!\n";
    }
    std::cout << "\n";

    // Lump the mass matrix
    std::cout << "--- Test 5: Lumped Mass ---\n";
    Real M_lumped[60] = {0};
    for (int i = 0; i < 60; ++i) {
        for (int j = 0; j < 60; ++j) {
            M_lumped[i] += M[i * 60 + j];
        }
    }

    Real lumped_total = 0.0;
    int zero_lumped = 0;
    for (int i = 0; i < 60; ++i) {
        lumped_total += M_lumped[i];
        if (std::abs(M_lumped[i]) < 1e-10) {
            std::cout << "  WARNING: Lumped mass DOF " << i << " is zero!\n";
            zero_lumped++;
        }
    }

    std::cout << "Total lumped mass: " << lumped_total << " kg\n";
    std::cout << "Zero lumped mass DOFs: " << zero_lumped << "\n";
    std::cout << "Status: " << (zero_lumped == 0 ? "PASS" : "FAIL") << "\n\n";

    std::cout << "--- Test 6: Volume Calculation ---\n";
    Real computed_volume = elem.volume(coords);
    std::cout << "Computed volume: " << computed_volume << " m³\n";
    std::cout << "Expected volume: " << volume << " m³\n";
    std::cout << "Error: " << std::abs(computed_volume - volume) / volume * 100.0 << "%\n";
    std::cout << "Status: " << (std::abs(computed_volume - volume) / volume < 0.01 ? "PASS" : "FAIL") << "\n\n";

    std::cout << "=================================================\n";
    std::cout << "Hex20 Single Element Test Complete\n";
    std::cout << "=================================================\n";

    return 0;
}
