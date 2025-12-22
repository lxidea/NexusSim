/**
 * @file hex20_mass_matrix_check.cpp
 * @brief Check if Hex20 consistent mass matrix has negative diagonal entries
 */

#include <nexussim/discretization/hex20.hpp>
#include <iostream>
#include <iomanip>
#include <algorithm>

using namespace nxs;
using namespace nxs::fem;

int main() {
    std::cout << std::setprecision(10);

    // Create a unit cube Hex20 element
    const Real L = 1.0;
    Real coords[60]; // 20 nodes × 3 coords

    // Corner nodes (0-7)
    coords[0*3+0] = 0.0; coords[0*3+1] = 0.0; coords[0*3+2] = 0.0;
    coords[1*3+0] = L;   coords[1*3+1] = 0.0; coords[1*3+2] = 0.0;
    coords[2*3+0] = L;   coords[2*3+1] = L;   coords[2*3+2] = 0.0;
    coords[3*3+0] = 0.0; coords[3*3+1] = L;   coords[3*3+2] = 0.0;
    coords[4*3+0] = 0.0; coords[4*3+1] = 0.0; coords[4*3+2] = L;
    coords[5*3+0] = L;   coords[5*3+1] = 0.0; coords[5*3+2] = L;
    coords[6*3+0] = L;   coords[6*3+1] = L;   coords[6*3+2] = L;
    coords[7*3+0] = 0.0; coords[7*3+1] = L;   coords[7*3+2] = L;

    // Mid-edge nodes (8-19) - at midpoints
    coords[8*3+0] = 0.5*L; coords[8*3+1] = 0.0;   coords[8*3+2] = 0.0;
    coords[9*3+0] = L;     coords[9*3+1] = 0.5*L; coords[9*3+2] = 0.0;
    coords[10*3+0] = 0.5*L; coords[10*3+1] = L;   coords[10*3+2] = 0.0;
    coords[11*3+0] = 0.0;   coords[11*3+1] = 0.5*L; coords[11*3+2] = 0.0;
    coords[12*3+0] = 0.0;   coords[12*3+1] = 0.0;   coords[12*3+2] = 0.5*L;
    coords[13*3+0] = L;     coords[13*3+1] = 0.0;   coords[13*3+2] = 0.5*L;
    coords[14*3+0] = L;     coords[14*3+1] = L;     coords[14*3+2] = 0.5*L;
    coords[15*3+0] = 0.0;   coords[15*3+1] = L;     coords[15*3+2] = 0.5*L;
    coords[16*3+0] = 0.5*L; coords[16*3+1] = 0.0;   coords[16*3+2] = L;
    coords[17*3+0] = L;     coords[17*3+1] = 0.5*L; coords[17*3+2] = L;
    coords[18*3+0] = 0.5*L; coords[18*3+1] = L;     coords[18*3+2] = L;
    coords[19*3+0] = 0.0;   coords[19*3+1] = 0.5*L; coords[19*3+2] = L;

    // Material properties
    const Real density = 7850.0;  // kg/m³ (steel)

    // Create element
    Hex20Element elem;

    // Compute consistent mass matrix
    Real M[60*60];
    elem.mass_matrix(coords, density, M);

    // Check diagonal entries
    std::cout << "===== Hex20 Consistent Mass Matrix Diagonal Check =====" << std::endl;
    std::cout << "Element: Unit cube (1m × 1m × 1m)" << std::endl;
    std::cout << "Density: " << density << " kg/m³" << std::endl << std::endl;

    int num_negative = 0;
    int num_zero = 0;
    Real min_diag = 1e308;
    Real max_diag = -1e308;
    int min_node = -1, max_node = -1;

    std::cout << "Nodal mass diagonal entries (M[i,i] for DOF i):" << std::endl;
    std::cout << "Node  Mx          My          Mz          Min" << std::endl;
    std::cout << "----  ----------  ----------  ----------  ----------" << std::endl;

    for (int node = 0; node < 20; ++node) {
        Real mx = M[(node*3+0)*60 + (node*3+0)];
        Real my = M[(node*3+1)*60 + (node*3+1)];
        Real mz = M[(node*3+2)*60 + (node*3+2)];
        Real min_mass = std::min({mx, my, mz});

        std::cout << std::setw(4) << node << "  ";
        std::cout << std::setw(10) << mx << "  ";
        std::cout << std::setw(10) << my << "  ";
        std::cout << std::setw(10) << mz << "  ";
        std::cout << std::setw(10) << min_mass;

        if (min_mass < 0) {
            std::cout << "  ← NEGATIVE!";
            num_negative++;
        } else if (min_mass == 0) {
            std::cout << "  ← ZERO!";
            num_zero++;
        }
        std::cout << std::endl;

        if (min_mass < min_diag) {
            min_diag = min_mass;
            min_node = node;
        }
        if (min_mass > max_diag) {
            max_diag = min_mass;
            max_node = node;
        }
    }

    std::cout << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  Nodes with negative mass: " << num_negative << " / 20" << std::endl;
    std::cout << "  Nodes with zero mass:     " << num_zero << " / 20" << std::endl;
    std::cout << "  Min diagonal entry:       " << min_diag << " (node " << min_node << ")" << std::endl;
    std::cout << "  Max diagonal entry:       " << max_diag << " (node " << max_node << ")" << std::endl;

    // Compute total mass
    Real total_mass = 0.0;
    for (int i = 0; i < 60; ++i) {
        total_mass += M[i*60 + i];
    }
    Real expected_mass = density * L * L * L;  // ρ × volume
    std::cout << "  Total mass (diagonal sum): " << total_mass << " kg" << std::endl;
    std::cout << "  Expected mass (ρ×V):       " << expected_mass << " kg" << std::endl;
    std::cout << "  Ratio:                     " << total_mass / expected_mass << std::endl;

    std::cout << std::endl;
    if (num_negative > 0) {
        std::cout << "✗ PROBLEM: Consistent mass matrix has NEGATIVE diagonal entries!" << std::endl;
        std::cout << "  This causes numerical instability in explicit time integration." << std::endl;
        std::cout << "  Solution: Use HRZ lumped mass instead of consistent mass." << std::endl;
        return 1;
    } else {
        std::cout << "✓ OK: All diagonal entries are positive." << std::endl;
        return 0;
    }
}
