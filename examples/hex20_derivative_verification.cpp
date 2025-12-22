/**
 * @file hex20_derivative_verification.cpp
 * @brief Systematic verification of Hex20 shape function derivatives
 *
 * This tool computes numerical derivatives and compares them to analytical
 * derivatives to find errors in the implementation.
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/discretization/hex20.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;

// Compute numerical derivative using central difference
void numerical_derivatives(Hex20Element& elem, const Real xi[3], Real* dN_numerical) {
    const Real h = 1e-7;  // Small step for finite difference

    Real N_plus[20], N_minus[20];
    Real xi_perturbed[3];

    // For each coordinate direction (ξ, η, ζ)
    for (int dir = 0; dir < 3; ++dir) {
        // Perturb coordinate +h
        xi_perturbed[0] = xi[0];
        xi_perturbed[1] = xi[1];
        xi_perturbed[2] = xi[2];
        xi_perturbed[dir] += h;
        elem.shape_functions(xi_perturbed, N_plus);

        // Perturb coordinate -h
        xi_perturbed[dir] = xi[dir] - h;
        elem.shape_functions(xi_perturbed, N_minus);

        // Central difference: (f(x+h) - f(x-h)) / (2h)
        for (int node = 0; node < 20; ++node) {
            dN_numerical[node*3 + dir] = (N_plus[node] - N_minus[node]) / (2.0 * h);
        }
    }
}

void verify_node_derivatives(int node_id, const char* node_name,
                             const Real xi[3], Hex20Element& elem) {

    std::cout << "\n=== Node " << node_id << ": " << node_name << " ===\n";
    std::cout << "Natural coordinates: (ξ=" << xi[0] << ", η=" << xi[1] << ", ζ=" << xi[2] << ")\n\n";

    // Compute analytical derivatives
    Real dN_analytical[60];  // 20 nodes * 3 derivatives
    elem.shape_derivatives(xi, dN_analytical);

    // Compute numerical derivatives
    Real dN_numerical[60];
    numerical_derivatives(elem, xi, dN_numerical);

    // Compare for this specific node
    std::cout << "              Analytical    Numerical     Error       Status\n";
    std::cout << "              ----------    ---------     -----       ------\n";

    const char* coord_names[3] = {"∂N/∂ξ", "∂N/∂η", "∂N/∂ζ"};

    bool all_ok = true;
    for (int dir = 0; dir < 3; ++dir) {
        Real analytical = dN_analytical[node_id*3 + dir];
        Real numerical = dN_numerical[node_id*3 + dir];
        Real error = std::abs(analytical - numerical);

        std::cout << std::setw(10) << coord_names[dir] << ": ";
        std::cout << std::setw(12) << std::fixed << std::setprecision(8) << analytical << "  ";
        std::cout << std::setw(12) << numerical << "  ";
        std::cout << std::setw(10) << std::scientific << std::setprecision(2) << error << "  ";

        if (error < 1e-6) {
            std::cout << "[OK]\n";
        } else {
            std::cout << "[FAIL]\n";
            all_ok = false;
        }
    }

    if (all_ok) {
        std::cout << "\n✓ Node " << node_id << " derivatives are CORRECT\n";
    } else {
        std::cout << "\n✗ Node " << node_id << " has INCORRECT derivatives\n";
    }
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "Hex20 Shape Function Derivative Verification\n";
    std::cout << "=================================================\n";

    Hex20Element elem;

    // Test all 8 corner nodes at their natural coordinate positions
    struct CornerNode {
        int id;
        const char* name;
        Real xi[3];
    };

    CornerNode corners[8] = {
        {0, "(-1,-1,-1)", {-1.0, -1.0, -1.0}},
        {1, "(+1,-1,-1)", {+1.0, -1.0, -1.0}},
        {2, "(+1,+1,-1)", {+1.0, +1.0, -1.0}},
        {3, "(-1,+1,-1)", {-1.0, +1.0, -1.0}},
        {4, "(-1,-1,+1)", {-1.0, -1.0, +1.0}},
        {5, "(+1,-1,+1)", {+1.0, -1.0, +1.0}},
        {6, "(+1,+1,+1)", {+1.0, +1.0, +1.0}},
        {7, "(-1,+1,+1)", {-1.0, +1.0, +1.0}}
    };

    std::cout << "\nTesting CORNER nodes (0-7):\n";
    std::cout << "========================================\n";

    int num_failed = 0;
    for (int i = 0; i < 8; ++i) {
        verify_node_derivatives(corners[i].id, corners[i].name, corners[i].xi, elem);

        // Quick check if failed
        Real dN[60];
        elem.shape_derivatives(corners[i].xi, dN);
        Real dN_num[60];
        numerical_derivatives(elem, corners[i].xi, dN_num);

        bool failed = false;
        for (int d = 0; d < 3; ++d) {
            if (std::abs(dN[corners[i].id*3 + d] - dN_num[corners[i].id*3 + d]) > 1e-6) {
                failed = true;
                break;
            }
        }
        if (failed) num_failed++;
    }

    // Test a few mid-edge nodes
    std::cout << "\n\nTesting MID-EDGE nodes (8-19):\n";
    std::cout << "========================================\n";

    struct EdgeNode {
        int id;
        const char* name;
        Real xi[3];
    };

    EdgeNode edges[4] = {
        {8,  "Edge 0-1", { 0.0, -1.0, -1.0}},
        {12, "Edge 0-4", {-1.0, -1.0,  0.0}},
        {16, "Edge 4-5", { 0.0, -1.0, +1.0}},
        {17, "Edge 5-6", {+1.0,  0.0, +1.0}}
    };

    for (int i = 0; i < 4; ++i) {
        verify_node_derivatives(edges[i].id, edges[i].name, edges[i].xi, elem);

        // Quick check if failed
        Real dN[60];
        elem.shape_derivatives(edges[i].xi, dN);
        Real dN_num[60];
        numerical_derivatives(elem, edges[i].xi, dN_num);

        bool failed = false;
        for (int d = 0; d < 3; ++d) {
            if (std::abs(dN[edges[i].id*3 + d] - dN_num[edges[i].id*3 + d]) > 1e-6) {
                failed = true;
                break;
            }
        }
        if (failed) num_failed++;
    }

    std::cout << "\n=================================================\n";
    std::cout << "Summary\n";
    std::cout << "=================================================\n";
    std::cout << "Failed nodes: " << num_failed << " / 12 tested\n";

    if (num_failed == 0) {
        std::cout << "\n✓ All tested derivatives are CORRECT!\n";
    } else {
        std::cout << "\n✗ Some derivatives need fixing\n";
    }

    std::cout << "=================================================\n";

    return (num_failed == 0) ? 0 : 1;
}
