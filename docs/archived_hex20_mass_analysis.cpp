/**
 * @file hex20_mass_analysis.cpp
 * @brief Detailed analysis of Hex20 mass matrix
 */

#include <nexussim/discretization/hex20.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace nxs::fem;

int main() {
    std::cout << "=================================================\n";
    std::cout << "Hex20 Mass Matrix Detailed Analysis\n";
    std::cout << "=================================================\n\n";

    Hex20Element elem;

    // Unit cube with proper Hex20 node ordering
    // Based on the coordinates from hex20_bending_test.cpp
    Real coords[20 * 3] = {
        // Corner nodes (0-7)
        0.0, 0.0, 0.0,  // 0
        1.0, 0.0, 0.0,  // 1
        1.0, 1.0, 0.0,  // 2
        0.0, 1.0, 0.0,  // 3
        0.0, 0.0, 1.0,  // 4
        1.0, 0.0, 1.0,  // 5
        1.0, 1.0, 1.0,  // 6
        0.0, 1.0, 1.0,  // 7
        // Bottom face mid-edge nodes (8-11)
        0.5, 0.0, 0.0,  // 8
        1.0, 0.5, 0.0,  // 9
        0.5, 1.0, 0.0,  // 10
        0.0, 0.5, 0.0,  // 11
        // Vertical mid-edge nodes (12-15)
        0.0, 0.0, 0.5,  // 12
        1.0, 0.0, 0.5,  // 13
        1.0, 1.0, 0.5,  // 14
        0.0, 1.0, 0.5,  // 15
        // Top face mid-edge nodes (16-19)
        0.5, 0.0, 1.0,  // 16
        1.0, 0.5, 1.0,  // 17
        0.5, 1.0, 1.0,  // 18
        0.0, 0.5, 1.0   // 19
    };

    const Real density = 1000.0;

    // Compute consistent mass matrix
    const int dof_per_elem = 60;  // 20 nodes * 3 DOFs
    std::vector<Real> M(dof_per_elem * dof_per_elem, 0.0);

    elem.mass_matrix(coords, density, M.data());

    std::cout << std::setprecision(6) << std::fixed;

    // Compute row sums (lumped mass)
    std::cout << "Row-sum lumped masses (by node):\n";
    std::cout << "Node | DOF_X      | DOF_Y      | DOF_Z      | Sum        | Type\n";
    std::cout << "-----+------------+------------+------------+------------+---------\n";

    std::vector<Real> lumped_mass(dof_per_elem, 0.0);
    for (int i = 0; i < dof_per_elem; ++i) {
        for (int j = 0; j < dof_per_elem; ++j) {
            lumped_mass[i] += M[i * dof_per_elem + j];
        }
    }

    Real total_mass = 0.0;
    int negative_count = 0;
    int zero_count = 0;

    for (int node = 0; node < 20; ++node) {
        Real mx = lumped_mass[node * 3 + 0];
        Real my = lumped_mass[node * 3 + 1];
        Real mz = lumped_mass[node * 3 + 2];
        Real sum = mx + my + mz;

        const char* type = "";
        if (node < 8) {
            type = "Corner";
        } else {
            type = "Mid-edge";
        }

        std::cout << std::setw(4) << node << " | ";
        std::cout << std::setw(10) << mx << " | ";
        std::cout << std::setw(10) << my << " | ";
        std::cout << std::setw(10) << mz << " | ";
        std::cout << std::setw(10) << sum << " | ";
        std::cout << type;

        if (mx < -1.0e-10 || my < -1.0e-10 || mz < -1.0e-10) {
            std::cout << " [NEGATIVE]";
            negative_count++;
        } else if (mx < 1.0e-10 || my < 1.0e-10 || mz < 1.0e-10) {
            std::cout << " [ZERO]";
            zero_count++;
        }
        std::cout << "\n";

        total_mass += sum;
    }

    std::cout << "\n=================================================\n";
    std::cout << "Summary:\n";
    std::cout << "  Total mass: " << total_mass << " kg\n";
    std::cout << "  Expected: " << density * 1.0 << " kg\n";
    std::cout << "  Nodes with negative lumped mass: " << negative_count << "\n";
    std::cout << "  Nodes with zero lumped mass: " << zero_count << "\n";
    std::cout << "=================================================\n";

    // Check partition of unity
    std::cout << "\nChecking partition of unity at element center:\n";
    Real xi[3] = {0.0, 0.0, 0.0};
    Real N[20];
    elem.shape_functions(xi, N);

    Real sum_N = 0.0;
    for (int i = 0; i < 20; ++i) {
        sum_N += N[i];
    }
    std::cout << "  Sum of shape functions at (0,0,0): " << sum_N << " (expected 1.0)\n";

    // Print individual shape function values
    std::cout << "\nShape function values at element center:\n";
    for (int i = 0; i < 20; ++i) {
        std::cout << "  N[" << i << "] = " << N[i];
        if (i < 8) {
            std::cout << " (corner)";
        } else {
            std::cout << " (mid-edge)";
        }
        std::cout << "\n";
    }

    return 0;
}
