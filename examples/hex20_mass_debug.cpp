/**
 * @file hex20_mass_debug.cpp
 * @brief Debug Hex20 mass matrix integration
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
    std::cout << "Hex20 Mass Matrix Debug\n";
    std::cout << "=================================================\n\n";

    Hex20Element elem;

    // Unit cube
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

    // Get Gauss points and weights
    Real gp[27*3], gw[27];
    elem.gauss_quadrature(gp, gw);

    std::cout << std::setprecision(12);

    // Check weight sum
    Real weight_sum = 0.0;
    for (int ig = 0; ig < 27; ++ig) {
        weight_sum += gw[ig];
    }
    std::cout << "Sum of Gauss weights: " << weight_sum << " (expected 8.0 for [-1,1]^3)\n\n";

    // Integrate constant function (volume calculation)
    Real vol = 0.0;
    for (int ig = 0; ig < 27; ++ig) {
        Real xi[3] = {gp[ig*3 + 0], gp[ig*3 + 1], gp[ig*3 + 2]};
        Real J[9];
        Real det_J = elem.jacobian(xi, coords, J);
        vol += gw[ig] * det_J;
    }
    std::cout << "Computed volume: " << vol << " m³ (expected 1.0 m³)\n\n";

    // Integrate sum of shape functions (should give volume * 1.0 = volume)
    Real shape_integral = 0.0;
    for (int ig = 0; ig < 27; ++ig) {
        Real xi[3] = {gp[ig*3 + 0], gp[ig*3 + 1], gp[ig*3 + 2]};
        Real N[20];
        elem.shape_functions(xi, N);

        Real N_sum = 0.0;
        for (int i = 0; i < 20; ++i) {
            N_sum += N[i];
        }

        Real J[9];
        Real det_J = elem.jacobian(xi, coords, J);
        shape_integral += gw[ig] * N_sum * det_J;
    }
    std::cout << "Integral of sum(N_i): " << shape_integral << " (expected " << vol << ")\n\n";

    // Integrate N_i * N_j for all pairs (should sum to density * volume)
    Real mass_integral = 0.0;
    const Real density = 1000.0;

    for (int ig = 0; ig < 27; ++ig) {
        Real xi[3] = {gp[ig*3 + 0], gp[ig*3 + 1], gp[ig*3 + 2]};
        Real N[20];
        elem.shape_functions(xi, N);

        Real J[9];
        Real det_J = elem.jacobian(xi, coords, J);

        for (int i = 0; i < 20; ++i) {
            for (int j = 0; j < 20; ++j) {
                mass_integral += density * gw[ig] * N[i] * N[j] * det_J;
            }
        }
    }

    std::cout << "Total mass (sum of all M_ij entries): " << mass_integral << " kg\n";
    std::cout << "Expected mass: " << density * vol << " kg\n";
    std::cout << "Ratio: " << mass_integral / (density * vol) << "\n\n";

    // Now check individual Gauss point contributions
    std::cout << "--- Gauss Point Contributions ---\n";
    for (int ig = 0; ig < 27; ++ig) {
        Real xi[3] = {gp[ig*3 + 0], gp[ig*3 + 1], gp[ig*3 + 2]};
        Real J[9];
        Real det_J = elem.jacobian(xi, coords, J);

        Real N[20];
        elem.shape_functions(xi, N);
        Real N_sum = 0.0;
        for (int i = 0; i < 20; ++i) {
            N_sum += N[i];
        }

        if (ig < 5 || ig >= 25) {  // Show first few and last few
            std::cout << "GP " << ig << ": ξ=(" << xi[0] << ", " << xi[1] << ", " << xi[2] << ")";
            std::cout << " w=" << gw[ig] << " det(J)=" << det_J << " sum(N)=" << N_sum << "\n";
        } else if (ig == 5) {
            std::cout << "...\n";
        }
    }

    std::cout << "\n=================================================\n";
    std::cout << "Debug Complete\n";
    std::cout << "=================================================\n";

    return 0;
}
