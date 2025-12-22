/**
 * @file hex20_derivative_check.cpp
 * @brief Verify Hex20 shape function derivatives using numerical differentiation
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
    std::cout << "Hex20 Derivative Verification\n";
    std::cout << "=================================================\n\n";

    Hex20Element elem;

    // Test at center of element
    Real xi[3] = {0.0, 0.0, 0.0};
    const Real h = 1e-6;  // Finite difference step

    // Compute analytical derivatives
    Real dN_analytical[20 * 3];
    elem.shape_derivatives(xi, dN_analytical);

    // Compute numerical derivatives
    Real dN_numerical[20 * 3];
    Real N_plus[20], N_minus[20];

    std::cout << std::setprecision(10);
    std::cout << "Checking derivatives at center (ξ=0, η=0, ζ=0):\n\n";

    int errors = 0;
    Real max_error = 0.0;

    // Check ∂N/∂ξ
    for (int i = 0; i < 20; ++i) {
        Real xi_plus[3] = {xi[0] + h, xi[1], xi[2]};
        Real xi_minus[3] = {xi[0] - h, xi[1], xi[2]};
        elem.shape_functions(xi_plus, N_plus);
        elem.shape_functions(xi_minus, N_minus);
        dN_numerical[i*3 + 0] = (N_plus[i] - N_minus[i]) / (2.0 * h);

        Real error = std::abs(dN_analytical[i*3 + 0] - dN_numerical[i*3 + 0]);
        if (error > max_error) max_error = error;
        if (error > 1e-5) {
            std::cout << "Node " << i << " ∂N/∂ξ: analytical=" << dN_analytical[i*3 + 0]
                      << ", numerical=" << dN_numerical[i*3 + 0] << ", error=" << error << "\n";
            errors++;
        }
    }

    // Check ∂N/∂η
    for (int i = 0; i < 20; ++i) {
        Real xi_plus[3] = {xi[0], xi[1] + h, xi[2]};
        Real xi_minus[3] = {xi[0], xi[1] - h, xi[2]};
        elem.shape_functions(xi_plus, N_plus);
        elem.shape_functions(xi_minus, N_minus);
        dN_numerical[i*3 + 1] = (N_plus[i] - N_minus[i]) / (2.0 * h);

        Real error = std::abs(dN_analytical[i*3 + 1] - dN_numerical[i*3 + 1]);
        if (error > max_error) max_error = error;
        if (error > 1e-5) {
            std::cout << "Node " << i << " ∂N/∂η: analytical=" << dN_analytical[i*3 + 1]
                      << ", numerical=" << dN_numerical[i*3 + 1] << ", error=" << error << "\n";
            errors++;
        }
    }

    // Check ∂N/∂ζ
    for (int i = 0; i < 20; ++i) {
        Real xi_plus[3] = {xi[0], xi[1], xi[2] + h};
        Real xi_minus[3] = {xi[0], xi[1], xi[2] - h};
        elem.shape_functions(xi_plus, N_plus);
        elem.shape_functions(xi_minus, N_minus);
        dN_numerical[i*3 + 2] = (N_plus[i] - N_minus[i]) / (2.0 * h);

        Real error = std::abs(dN_analytical[i*3 + 2] - dN_numerical[i*3 + 2]);
        if (error > max_error) max_error = error;
        if (error > 1e-5) {
            std::cout << "Node " << i << " ∂N/∂ζ: analytical=" << dN_analytical[i*3 + 2]
                      << ", numerical=" << dN_numerical[i*3 + 2] << ", error=" << error << "\n";
            errors++;
        }
    }

    std::cout << "\nMax derivative error: " << max_error << "\n";
    std::cout << "Total derivative errors (>1e-5): " << errors << "\n";
    std::cout << "Status: " << (errors == 0 ? "PASS" : "FAIL") << "\n\n";

    // Now check Jacobian computation for unit cube
    std::cout << "--- Jacobian Check for Unit Cube ---\n";
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

    Real J[9];
    Real det_J = elem.jacobian(xi, coords, J);

    std::cout << "Jacobian at center:\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [";
        for (int j = 0; j < 3; ++j) {
            std::cout << std::setw(12) << J[i*3 + j];
        }
        std::cout << " ]\n";
    }
    std::cout << "\nDeterminant: " << det_J << "\n";
    std::cout << "Expected: 0.125 (for unit cube mapped from [-1,1]³)\n";
    std::cout << "Expected Jacobian: diag(0.5, 0.5, 0.5) for unit cube\n\n";

    // Manual Jacobian computation to verify
    std::cout << "--- Manual Jacobian Verification ---\n";
    Real J_manual[9] = {0};
    elem.shape_derivatives(xi, dN_analytical);

    for (int i = 0; i < 20; ++i) {
        const Real x = coords[i*3 + 0];
        const Real y = coords[i*3 + 1];
        const Real z = coords[i*3 + 2];

        const Real dNdxi   = dN_analytical[i*3 + 0];
        const Real dNdeta  = dN_analytical[i*3 + 1];
        const Real dNdzeta = dN_analytical[i*3 + 2];

        J_manual[0] += dNdxi * x;    // ∂x/∂ξ
        J_manual[3] += dNdxi * y;    // ∂y/∂ξ
        J_manual[6] += dNdxi * z;    // ∂z/∂ξ

        J_manual[1] += dNdeta * x;   // ∂x/∂η
        J_manual[4] += dNdeta * y;   // ∂y/∂η
        J_manual[7] += dNdeta * z;   // ∂z/∂η

        J_manual[2] += dNdzeta * x;  // ∂x/∂ζ
        J_manual[5] += dNdzeta * y;  // ∂y/∂ζ
        J_manual[8] += dNdzeta * z;  // ∂z/∂ζ
    }

    std::cout << "Manual Jacobian:\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [";
        for (int j = 0; j < 3; ++j) {
            std::cout << std::setw(12) << J_manual[i*3 + j];
        }
        std::cout << " ]\n";
    }

    Real det_J_manual = J_manual[0] * (J_manual[4] * J_manual[8] - J_manual[5] * J_manual[7])
                      - J_manual[1] * (J_manual[3] * J_manual[8] - J_manual[5] * J_manual[6])
                      + J_manual[2] * (J_manual[3] * J_manual[7] - J_manual[4] * J_manual[6]);
    std::cout << "\nManual determinant: " << det_J_manual << "\n";

    std::cout << "\n=================================================\n";
    std::cout << "Test Complete\n";
    std::cout << "=================================================\n";

    return (errors == 0 && std::abs(det_J - 0.125) < 1e-5) ? 0 : 1;
}
