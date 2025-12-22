/**
 * @file hex20_derivative_verify.cpp
 * @brief Verify Hex20 shape function derivatives against analytical formulas
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/discretization/hex20.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;

// Analytical shape functions for Hex20 (from Hughes FEM textbook)
// Corner nodes (0-7): N_i = (1/8)(1 + ξ₀)(1 + η₀)(1 + ζ₀)(ξ₀ + η₀ + ζ₀ - 2)
// where ξ₀ = ξᵢξ, η₀ = ηᵢη, ζ₀ = ζᵢζ
// Mid-edge nodes: standard serendipity formulas

// Node natural coordinates
const double node_coords[20][3] = {
    // Corner nodes
    {-1, -1, -1},  // 0
    { 1, -1, -1},  // 1
    { 1,  1, -1},  // 2
    {-1,  1, -1},  // 3
    {-1, -1,  1},  // 4
    { 1, -1,  1},  // 5
    { 1,  1,  1},  // 6
    {-1,  1,  1},  // 7
    // Mid-edge nodes (bottom face z=-1)
    { 0, -1, -1},  // 8  (edge 0-1)
    { 1,  0, -1},  // 9  (edge 1-2)
    { 0,  1, -1},  // 10 (edge 2-3)
    {-1,  0, -1},  // 11 (edge 3-0)
    // Mid-edge nodes (vertical edges)
    {-1, -1,  0},  // 12 (edge 0-4)
    { 1, -1,  0},  // 13 (edge 1-5)
    { 1,  1,  0},  // 14 (edge 2-6)
    {-1,  1,  0},  // 15 (edge 3-7)
    // Mid-edge nodes (top face z=+1)
    { 0, -1,  1},  // 16 (edge 4-5)
    { 1,  0,  1},  // 17 (edge 5-6)
    { 0,  1,  1},  // 18 (edge 6-7)
    {-1,  0,  1},  // 19 (edge 7-4)
};

// Analytical shape functions
void analytical_shape_functions(double xi, double eta, double zeta, double* N) {
    // Corner nodes (0-7)
    for (int i = 0; i < 8; ++i) {
        double xi_i = node_coords[i][0];
        double eta_i = node_coords[i][1];
        double zeta_i = node_coords[i][2];

        double xi0 = 1.0 + xi_i * xi;
        double eta0 = 1.0 + eta_i * eta;
        double zeta0 = 1.0 + zeta_i * zeta;

        N[i] = 0.125 * xi0 * eta0 * zeta0 * (xi_i*xi + eta_i*eta + zeta_i*zeta - 2.0);
    }

    // Mid-edge nodes
    double xi2 = xi * xi;
    double eta2 = eta * eta;
    double zeta2 = zeta * zeta;

    // Bottom face (z=-1)
    N[8]  = 0.25 * (1.0 - xi2) * (1.0 - eta) * (1.0 - zeta);
    N[9]  = 0.25 * (1.0 + xi) * (1.0 - eta2) * (1.0 - zeta);
    N[10] = 0.25 * (1.0 - xi2) * (1.0 + eta) * (1.0 - zeta);
    N[11] = 0.25 * (1.0 - xi) * (1.0 - eta2) * (1.0 - zeta);

    // Vertical edges
    N[12] = 0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 - zeta2);
    N[13] = 0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - zeta2);
    N[14] = 0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - zeta2);
    N[15] = 0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 - zeta2);

    // Top face (z=+1)
    N[16] = 0.25 * (1.0 - xi2) * (1.0 - eta) * (1.0 + zeta);
    N[17] = 0.25 * (1.0 + xi) * (1.0 - eta2) * (1.0 + zeta);
    N[18] = 0.25 * (1.0 - xi2) * (1.0 + eta) * (1.0 + zeta);
    N[19] = 0.25 * (1.0 - xi) * (1.0 - eta2) * (1.0 + zeta);
}

// Analytical derivatives (correct formulas from Hughes)
void analytical_shape_derivatives(double xi, double eta, double zeta, double* dN) {
    // Corner nodes (0-7) - using product rule on N = (1/8)(1+ξ₀)(1+η₀)(1+ζ₀)(ξ₀+η₀+ζ₀-2)
    for (int i = 0; i < 8; ++i) {
        double xi_i = node_coords[i][0];
        double eta_i = node_coords[i][1];
        double zeta_i = node_coords[i][2];

        double a = 1.0 + xi_i * xi;      // (1 + ξᵢξ)
        double b = 1.0 + eta_i * eta;    // (1 + ηᵢη)
        double c = 1.0 + zeta_i * zeta;  // (1 + ζᵢζ)
        double s = xi_i*xi + eta_i*eta + zeta_i*zeta - 2.0;  // (ξᵢξ + ηᵢη + ζᵢζ - 2)

        // dN/dξ = (1/8) * [ξᵢ*b*c*s + a*b*c*ξᵢ] = (1/8) * ξᵢ * b * c * (s + a)
        //       = (1/8) * ξᵢ * b * c * (ξᵢξ + ηᵢη + ζᵢζ - 2 + 1 + ξᵢξ)
        //       = (1/8) * ξᵢ * b * c * (2ξᵢξ + ηᵢη + ζᵢζ - 1)
        dN[i*3 + 0] = 0.125 * xi_i * b * c * (2.0*xi_i*xi + eta_i*eta + zeta_i*zeta - 1.0);
        dN[i*3 + 1] = 0.125 * eta_i * a * c * (xi_i*xi + 2.0*eta_i*eta + zeta_i*zeta - 1.0);
        dN[i*3 + 2] = 0.125 * zeta_i * a * b * (xi_i*xi + eta_i*eta + 2.0*zeta_i*zeta - 1.0);
    }

    // Mid-edge nodes
    double xi2 = xi * xi;
    double eta2 = eta * eta;
    double zeta2 = zeta * zeta;

    // Node 8: (0,-1,-1) on edge 0-1
    dN[8*3 + 0] = -0.5 * xi * (1.0 - eta) * (1.0 - zeta);
    dN[8*3 + 1] = -0.25 * (1.0 - xi2) * (1.0 - zeta);
    dN[8*3 + 2] = -0.25 * (1.0 - xi2) * (1.0 - eta);

    // Node 9: (1,0,-1) on edge 1-2
    dN[9*3 + 0] = 0.25 * (1.0 - eta2) * (1.0 - zeta);
    dN[9*3 + 1] = -0.5 * eta * (1.0 + xi) * (1.0 - zeta);
    dN[9*3 + 2] = -0.25 * (1.0 + xi) * (1.0 - eta2);

    // Node 10: (0,1,-1) on edge 2-3
    dN[10*3 + 0] = -0.5 * xi * (1.0 + eta) * (1.0 - zeta);
    dN[10*3 + 1] = 0.25 * (1.0 - xi2) * (1.0 - zeta);
    dN[10*3 + 2] = -0.25 * (1.0 - xi2) * (1.0 + eta);

    // Node 11: (-1,0,-1) on edge 3-0
    dN[11*3 + 0] = -0.25 * (1.0 - eta2) * (1.0 - zeta);
    dN[11*3 + 1] = -0.5 * eta * (1.0 - xi) * (1.0 - zeta);
    dN[11*3 + 2] = -0.25 * (1.0 - xi) * (1.0 - eta2);

    // Node 12: (-1,-1,0) on edge 0-4
    dN[12*3 + 0] = -0.25 * (1.0 - eta) * (1.0 - zeta2);
    dN[12*3 + 1] = -0.25 * (1.0 - xi) * (1.0 - zeta2);
    dN[12*3 + 2] = -0.5 * zeta * (1.0 - xi) * (1.0 - eta);

    // Node 13: (1,-1,0) on edge 1-5
    dN[13*3 + 0] = 0.25 * (1.0 - eta) * (1.0 - zeta2);
    dN[13*3 + 1] = -0.25 * (1.0 + xi) * (1.0 - zeta2);
    dN[13*3 + 2] = -0.5 * zeta * (1.0 + xi) * (1.0 - eta);

    // Node 14: (1,1,0) on edge 2-6
    dN[14*3 + 0] = 0.25 * (1.0 + eta) * (1.0 - zeta2);
    dN[14*3 + 1] = 0.25 * (1.0 + xi) * (1.0 - zeta2);
    dN[14*3 + 2] = -0.5 * zeta * (1.0 + xi) * (1.0 + eta);

    // Node 15: (-1,1,0) on edge 3-7
    dN[15*3 + 0] = -0.25 * (1.0 + eta) * (1.0 - zeta2);
    dN[15*3 + 1] = 0.25 * (1.0 - xi) * (1.0 - zeta2);
    dN[15*3 + 2] = -0.5 * zeta * (1.0 - xi) * (1.0 + eta);

    // Node 16: (0,-1,1) on edge 4-5
    dN[16*3 + 0] = -0.5 * xi * (1.0 - eta) * (1.0 + zeta);
    dN[16*3 + 1] = -0.25 * (1.0 - xi2) * (1.0 + zeta);
    dN[16*3 + 2] = 0.25 * (1.0 - xi2) * (1.0 - eta);

    // Node 17: (1,0,1) on edge 5-6
    dN[17*3 + 0] = 0.25 * (1.0 - eta2) * (1.0 + zeta);
    dN[17*3 + 1] = -0.5 * eta * (1.0 + xi) * (1.0 + zeta);
    dN[17*3 + 2] = 0.25 * (1.0 + xi) * (1.0 - eta2);

    // Node 18: (0,1,1) on edge 6-7
    dN[18*3 + 0] = -0.5 * xi * (1.0 + eta) * (1.0 + zeta);
    dN[18*3 + 1] = 0.25 * (1.0 - xi2) * (1.0 + zeta);
    dN[18*3 + 2] = 0.25 * (1.0 - xi2) * (1.0 + eta);

    // Node 19: (-1,0,1) on edge 7-4
    dN[19*3 + 0] = -0.25 * (1.0 - eta2) * (1.0 + zeta);
    dN[19*3 + 1] = -0.5 * eta * (1.0 - xi) * (1.0 + zeta);
    dN[19*3 + 2] = 0.25 * (1.0 - xi) * (1.0 - eta2);
}

int main() {
    nxs::initialize();

    std::cout << "=================================================\n";
    std::cout << "Hex20 Shape Function Derivative Verification\n";
    std::cout << "=================================================\n\n";

    Hex20Element elem;

    // Test points
    const int num_tests = 5;
    double test_pts[num_tests][3] = {
        {0.0, 0.0, 0.0},     // Center
        {0.5, 0.5, 0.5},     // Arbitrary
        {-0.5, 0.3, -0.2},   // Arbitrary
        {0.7745966692, 0.0, 0.0},  // Gauss point
        {-1.0, -1.0, -1.0},  // Corner (node 0)
    };

    double max_error_N = 0.0;
    double max_error_dN = 0.0;
    int max_error_node = -1;
    int max_error_deriv = -1;

    for (int t = 0; t < num_tests; ++t) {
        double xi = test_pts[t][0];
        double eta = test_pts[t][1];
        double zeta = test_pts[t][2];

        std::cout << "Test point: (" << xi << ", " << eta << ", " << zeta << ")\n";
        std::cout << "-------------------------------------------\n";

        // Get implementation values
        Real xi_arr[3] = {xi, eta, zeta};
        Real N_impl[20], dN_impl[60];
        elem.shape_functions(xi_arr, N_impl);
        elem.shape_derivatives(xi_arr, dN_impl);

        // Get analytical values
        double N_anal[20], dN_anal[60];
        analytical_shape_functions(xi, eta, zeta, N_anal);
        analytical_shape_derivatives(xi, eta, zeta, dN_anal);

        // Check partition of unity
        double sum_N = 0.0;
        for (int i = 0; i < 20; ++i) sum_N += N_impl[i];
        std::cout << "Sum of N (should be 1.0): " << std::setprecision(10) << sum_N << "\n";

        // Check derivative sum (should be 0)
        double sum_dNdxi = 0.0, sum_dNdeta = 0.0, sum_dNdzeta = 0.0;
        for (int i = 0; i < 20; ++i) {
            sum_dNdxi += dN_impl[i*3 + 0];
            sum_dNdeta += dN_impl[i*3 + 1];
            sum_dNdzeta += dN_impl[i*3 + 2];
        }
        std::cout << "Sum of dN/dxi (should be 0): " << sum_dNdxi << "\n";
        std::cout << "Sum of dN/deta (should be 0): " << sum_dNdeta << "\n";
        std::cout << "Sum of dN/dzeta (should be 0): " << sum_dNdzeta << "\n\n";

        // Compare values
        std::cout << "Node | N_impl     | N_anal     | diff       | dNdxi_impl | dNdxi_anal | diff\n";
        std::cout << "-----------------------------------------------------------------------------\n";

        for (int i = 0; i < 20; ++i) {
            double diff_N = std::abs(N_impl[i] - N_anal[i]);
            double diff_dN0 = std::abs(dN_impl[i*3+0] - dN_anal[i*3+0]);
            double diff_dN1 = std::abs(dN_impl[i*3+1] - dN_anal[i*3+1]);
            double diff_dN2 = std::abs(dN_impl[i*3+2] - dN_anal[i*3+2]);

            if (diff_N > max_error_N) max_error_N = diff_N;
            if (diff_dN0 > max_error_dN) { max_error_dN = diff_dN0; max_error_node = i; max_error_deriv = 0; }
            if (diff_dN1 > max_error_dN) { max_error_dN = diff_dN1; max_error_node = i; max_error_deriv = 1; }
            if (diff_dN2 > max_error_dN) { max_error_dN = diff_dN2; max_error_node = i; max_error_deriv = 2; }

            // Only print if there's an error
            if (diff_N > 1e-10 || diff_dN0 > 1e-10 || diff_dN1 > 1e-10 || diff_dN2 > 1e-10) {
                std::cout << std::setw(4) << i << " | ";
                std::cout << std::setw(10) << std::setprecision(6) << N_impl[i] << " | ";
                std::cout << std::setw(10) << N_anal[i] << " | ";
                std::cout << std::setw(10) << diff_N << " | ";
                std::cout << std::setw(10) << dN_impl[i*3+0] << " | ";
                std::cout << std::setw(10) << dN_anal[i*3+0] << " | ";
                std::cout << std::setw(10) << diff_dN0 << "\n";

                if (diff_dN1 > 1e-10) {
                    std::cout << "     |            |            |            | ";
                    std::cout << std::setw(10) << dN_impl[i*3+1] << " | ";
                    std::cout << std::setw(10) << dN_anal[i*3+1] << " | ";
                    std::cout << std::setw(10) << diff_dN1 << " (deta)\n";
                }
                if (diff_dN2 > 1e-10) {
                    std::cout << "     |            |            |            | ";
                    std::cout << std::setw(10) << dN_impl[i*3+2] << " | ";
                    std::cout << std::setw(10) << dN_anal[i*3+2] << " | ";
                    std::cout << std::setw(10) << diff_dN2 << " (dzeta)\n";
                }
            }
        }
        std::cout << "\n";
    }

    std::cout << "=================================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "=================================================\n";
    std::cout << "Max shape function error: " << max_error_N << "\n";
    std::cout << "Max derivative error: " << max_error_dN;
    if (max_error_node >= 0) {
        std::cout << " (node " << max_error_node << ", deriv " << max_error_deriv << ")";
    }
    std::cout << "\n";

    if (max_error_N < 1e-10 && max_error_dN < 1e-10) {
        std::cout << "\n*** ALL TESTS PASSED ***\n";
        return 0;
    } else {
        std::cout << "\n*** ERRORS DETECTED ***\n";
        return 1;
    }
}
