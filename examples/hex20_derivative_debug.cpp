/**
 * @file hex20_derivative_debug.cpp
 * @brief Debug Hex20 shape function derivatives by comparing with Hex8
 */

#include <nexussim/discretization/hex8.hpp>
#include <nexussim/discretization/hex20.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs::fem;

int main() {
    std::cout << std::setprecision(12) << std::fixed;

    // Create simple cube element (1x1x1)
    Real coords_hex8[8*3] = {
        0.0, 0.0, 0.0,  // 0
        1.0, 0.0, 0.0,  // 1
        1.0, 1.0, 0.0,  // 2
        0.0, 1.0, 0.0,  // 3
        0.0, 0.0, 1.0,  // 4
        1.0, 0.0, 1.0,  // 5
        1.0, 1.0, 1.0,  // 6
        0.0, 1.0, 1.0   // 7
    };

    // Hex20 has same corner nodes plus mid-edge nodes
    Real coords_hex20[20*3];
    for (int i = 0; i < 8; ++i) {
        coords_hex20[i*3 + 0] = coords_hex8[i*3 + 0];
        coords_hex20[i*3 + 1] = coords_hex8[i*3 + 1];
        coords_hex20[i*3 + 2] = coords_hex8[i*3 + 2];
    }

    // Mid-edge nodes (straight edges for cube)
    coords_hex20[8*3 + 0]  = 0.5; coords_hex20[8*3 + 1]  = 0.0; coords_hex20[8*3 + 2]  = 0.0;  // Edge 0-1
    coords_hex20[9*3 + 0]  = 1.0; coords_hex20[9*3 + 1]  = 0.5; coords_hex20[9*3 + 2]  = 0.0;  // Edge 1-2
    coords_hex20[10*3 + 0] = 0.5; coords_hex20[10*3 + 1] = 1.0; coords_hex20[10*3 + 2] = 0.0;  // Edge 2-3
    coords_hex20[11*3 + 0] = 0.0; coords_hex20[11*3 + 1] = 0.5; coords_hex20[11*3 + 2] = 0.0;  // Edge 3-0
    coords_hex20[12*3 + 0] = 0.0; coords_hex20[12*3 + 1] = 0.0; coords_hex20[12*3 + 2] = 0.5;  // Edge 0-4
    coords_hex20[13*3 + 0] = 1.0; coords_hex20[13*3 + 1] = 0.0; coords_hex20[13*3 + 2] = 0.5;  // Edge 1-5
    coords_hex20[14*3 + 0] = 1.0; coords_hex20[14*3 + 1] = 1.0; coords_hex20[14*3 + 2] = 0.5;  // Edge 2-6
    coords_hex20[15*3 + 0] = 0.0; coords_hex20[15*3 + 1] = 1.0; coords_hex20[15*3 + 2] = 0.5;  // Edge 3-7
    coords_hex20[16*3 + 0] = 0.5; coords_hex20[16*3 + 1] = 0.0; coords_hex20[16*3 + 2] = 1.0;  // Edge 4-5
    coords_hex20[17*3 + 0] = 1.0; coords_hex20[17*3 + 1] = 0.5; coords_hex20[17*3 + 2] = 1.0;  // Edge 5-6
    coords_hex20[18*3 + 0] = 0.5; coords_hex20[18*3 + 1] = 1.0; coords_hex20[18*3 + 2] = 1.0;  // Edge 6-7
    coords_hex20[19*3 + 0] = 0.0; coords_hex20[19*3 + 1] = 0.5; coords_hex20[19*3 + 2] = 1.0;  // Edge 7-4

    Hex8Element hex8;
    Hex20Element hex20;

    // Test at element center
    Real xi[3] = {0.0, 0.0, 0.0};

    std::cout << "=== Shape Function Derivatives at Element Center ===" << std::endl;
    std::cout << "Testing with unit cube element" << std::endl << std::endl;

    // Hex8 derivatives
    Real dNdx_hex8[8*3];
    hex8.shape_derivatives_global(xi, coords_hex8, dNdx_hex8);

    std::cout << "Hex8 derivatives (corner nodes 0-7):" << std::endl;
    std::cout << "Node  |    dN/dx    |    dN/dy    |    dN/dz    |" << std::endl;
    std::cout << "------|-------------|-------------|-------------|" << std::endl;
    for (int i = 0; i < 8; ++i) {
        std::cout << "  " << i << "   | "
                  << std::setw(11) << dNdx_hex8[i*3 + 0] << " | "
                  << std::setw(11) << dNdx_hex8[i*3 + 1] << " | "
                  << std::setw(11) << dNdx_hex8[i*3 + 2] << " |" << std::endl;
    }
    std::cout << std::endl;

    // Hex20 derivatives
    Real dNdx_hex20[20*3];
    hex20.shape_derivatives_global(xi, coords_hex20, dNdx_hex20);

    std::cout << "Hex20 derivatives:" << std::endl;
    std::cout << "Node  |    dN/dx    |    dN/dy    |    dN/dz    | Type" << std::endl;
    std::cout << "------|-------------|-------------|-------------|-------" << std::endl;
    for (int i = 0; i < 20; ++i) {
        const char* type = (i < 8) ? "Corner" : "MidEdge";
        std::cout << " " << std::setw(2) << i << "   | "
                  << std::setw(11) << dNdx_hex20[i*3 + 0] << " | "
                  << std::setw(11) << dNdx_hex20[i*3 + 1] << " | "
                  << std::setw(11) << dNdx_hex20[i*3 + 2] << " | " << type << std::endl;
    }
    std::cout << std::endl;

    // Compare corner nodes - they should be similar but not identical
    std::cout << "=== Comparison of Corner Nodes ===" << std::endl;
    std::cout << "At element center, Hex20 corner node derivatives should be similar to Hex8" << std::endl;
    std::cout << "Node | Hex8 dN/dx | Hex20 dN/dx | Difference |" << std::endl;
    std::cout << "-----|------------|-------------|------------|" << std::endl;
    for (int i = 0; i < 8; ++i) {
        Real diff_x = dNdx_hex20[i*3 + 0] - dNdx_hex8[i*3 + 0];
        Real diff_y = dNdx_hex20[i*3 + 1] - dNdx_hex8[i*3 + 1];
        Real diff_z = dNdx_hex20[i*3 + 2] - dNdx_hex8[i*3 + 2];
        Real diff_norm = std::sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

        std::cout << "  " << i << "  | "
                  << std::setw(10) << dNdx_hex8[i*3 + 0] << " | "
                  << std::setw(11) << dNdx_hex20[i*3 + 0] << " | "
                  << std::setw(10) << diff_norm << " |" << std::endl;
    }
    std::cout << std::endl;

    // Check Jacobian
    Real J_hex8[9], J_hex20[9];
    Real det_hex8 = hex8.jacobian(xi, coords_hex8, J_hex8);
    Real det_hex20 = hex20.jacobian(xi, coords_hex20, J_hex20);

    std::cout << "=== Jacobian Determinants ===" << std::endl;
    std::cout << "Hex8:  " << det_hex8 << " (should be ~1.0 for unit cube)" << std::endl;
    std::cout << "Hex20: " << det_hex20 << " (should be ~1.0 for unit cube)" << std::endl;
    std::cout << std::endl;

    // Test with small displacement
    std::cout << "=== Force Direction Test ===" << std::endl;
    std::cout << "Apply small positive z-displacement to node 1 (corner)" << std::endl;

    Real disp_hex8[24] = {0};  // All zeros
    Real disp_hex20[60] = {0};  // All zeros

    // Small upward displacement at node 1
    disp_hex8[1*3 + 2] = 0.001;   // uz at node 1
    disp_hex20[1*3 + 2] = 0.001;  // uz at node 1

    // Compute strain
    Real B_hex8[6*24], B_hex20[6*60];
    hex8.strain_displacement_matrix(xi, coords_hex8, B_hex8);
    hex20.strain_displacement_matrix(xi, coords_hex20, B_hex20);

    Real strain_hex8[6] = {0};
    Real strain_hex20[6] = {0};

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 24; ++j) {
            strain_hex8[i] += B_hex8[i*24 + j] * disp_hex8[j];
        }
        for (int j = 0; j < 60; ++j) {
            strain_hex20[i] += B_hex20[i*60 + j] * disp_hex20[j];
        }
    }

    std::cout << "Strains from +z displacement at node 1:" << std::endl;
    std::cout << "Hex8:  εzz = " << strain_hex8[2] << std::endl;
    std::cout << "Hex20: εzz = " << strain_hex20[2] << std::endl;
    std::cout << "(Both should be positive for tensile strain)" << std::endl;
    std::cout << std::endl;

    // Compute stress (just εzz component for simplicity)
    Real E = 1.0e6;  // Arbitrary
    Real nu = 0.3;
    Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Real mu = E / (2.0 * (1.0 + nu));

    Real stress_zz_hex8 = (lambda + 2.0*mu) * strain_hex8[2] + lambda * (strain_hex8[0] + strain_hex8[1]);
    Real stress_zz_hex20 = (lambda + 2.0*mu) * strain_hex20[2] + lambda * (strain_hex20[0] + strain_hex20[1]);

    std::cout << "Stress σzz:" << std::endl;
    std::cout << "Hex8:  " << stress_zz_hex8 << std::endl;
    std::cout << "Hex20: " << stress_zz_hex20 << std::endl;
    std::cout << "(Both should be positive for tensile stress)" << std::endl;
    std::cout << std::endl;

    // The internal force at node 1 in z-direction should OPPOSE the displacement
    // i.e., if uz > 0 (upward), then fz should be < 0 (downward)
    std::cout << "Expected behavior:" << std::endl;
    std::cout << "- Displacement: +z" << std::endl;
    std::cout << "- Strain: positive (tension)" << std::endl;
    std::cout << "- Stress: positive (tension)" << std::endl;
    std::cout << "- Internal force at node 1: -z (resisting the displacement)" << std::endl;

    return 0;
}
