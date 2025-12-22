/**
 * @file hex20_multistep_debug.cpp
 * @brief Multi-step dynamics test to find where oscillation starts
 *
 * Manually performs 5 time steps and tracks:
 * - Displacement at each step
 * - Internal force at each step
 * - Acceleration at each step
 * - Sign changes and amplification factors
 */

#include <nexussim/discretization/hex20.hpp>
#include <nexussim/discretization/hex8.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;

template<typename ElementType>
void test_multistep(const char* element_name, int n_nodes) {
    std::cout << "\n========================================" << std::endl;
    std::cout << element_name << " Multi-Step Dynamics Test" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Unit cube
    const Real L = 1.0;
    const Real W = 0.2;
    const Real H = 0.2;
    Real coords[60];

    // Corner nodes (same for Hex8 and Hex20)
    coords[0*3+0] = 0.0; coords[0*3+1] = 0.0; coords[0*3+2] = 0.0;
    coords[1*3+0] = L;   coords[1*3+1] = 0.0; coords[1*3+2] = 0.0;
    coords[2*3+0] = L;   coords[2*3+1] = W;   coords[2*3+2] = 0.0;
    coords[3*3+0] = 0.0; coords[3*3+1] = W;   coords[3*3+2] = 0.0;
    coords[4*3+0] = 0.0; coords[4*3+1] = 0.0; coords[4*3+2] = H;
    coords[5*3+0] = L;   coords[5*3+1] = 0.0; coords[5*3+2] = H;
    coords[6*3+0] = L;   coords[6*3+1] = W;   coords[6*3+2] = H;
    coords[7*3+0] = 0.0; coords[7*3+1] = W;   coords[7*3+2] = H;

    if (n_nodes == 20) {
        // Mid-edge nodes for Hex20
        coords[8*3+0] = L/2; coords[8*3+1] = 0.0; coords[8*3+2] = 0.0;
        coords[9*3+0] = L;   coords[9*3+1] = W/2; coords[9*3+2] = 0.0;
        coords[10*3+0] = L/2; coords[10*3+1] = W;  coords[10*3+2] = 0.0;
        coords[11*3+0] = 0.0; coords[11*3+1] = W/2; coords[11*3+2] = 0.0;
        coords[12*3+0] = 0.0; coords[12*3+1] = 0.0; coords[12*3+2] = H/2;
        coords[13*3+0] = L;   coords[13*3+1] = 0.0; coords[13*3+2] = H/2;
        coords[14*3+0] = L;   coords[14*3+1] = W;   coords[14*3+2] = H/2;
        coords[15*3+0] = 0.0; coords[15*3+1] = W;   coords[15*3+2] = H/2;
        coords[16*3+0] = L/2; coords[16*3+1] = 0.0; coords[16*3+2] = H;
        coords[17*3+0] = L;   coords[17*3+1] = W/2; coords[17*3+2] = H;
        coords[18*3+0] = L/2; coords[18*3+1] = W;   coords[18*3+2] = H;
        coords[19*3+0] = 0.0; coords[19*3+1] = W/2; coords[19*3+2] = H;
    }

    const Real E = 210e9;
    const Real nu = 0.3;
    const Real density = 7850.0;

    ElementType elem;

    // Compute lumped mass
    std::vector<Real> M_lumped(n_nodes * 3, 0.0);
    if constexpr (std::is_same_v<ElementType, Hex20Element>) {
        std::vector<Real> nodal_mass(n_nodes, 0.0);
        elem.lumped_mass_hrz(coords, density, nodal_mass.data());
        for (int i = 0; i < n_nodes; ++i) {
            M_lumped[i*3 + 0] = nodal_mass[i];
            M_lumped[i*3 + 1] = nodal_mass[i];
            M_lumped[i*3 + 2] = nodal_mass[i];
        }
    } else {
        Real M_mat[24];
        elem.lumped_mass_matrix(coords, density, M_mat);
        for (int i = 0; i < n_nodes * 3; ++i) {
            M_lumped[i] = M_mat[i];
        }
    }

    // Compute stiffness matrix once
    Real K[60*60];
    elem.stiffness_matrix(coords, E, nu, K);

    // Initialize state
    std::vector<Real> u(n_nodes * 3, 0.0);
    std::vector<Real> v(n_nodes * 3, 0.0);
    std::vector<Real> a(n_nodes * 3, 0.0);
    std::vector<Real> f_int(n_nodes * 3, 0.0);
    std::vector<Real> f_ext(n_nodes * 3, 0.0);

    // Apply external force at node 1 (top right corner)
    f_ext[1*3 + 2] = -125.0;  // -z force (downward)

    // Fix left face nodes (x=0) - for Hex8: nodes 0,3,4,7
    std::vector<int> fixed_nodes;
    if (n_nodes == 8) {
        fixed_nodes = {0, 3, 4, 7};
    } else {
        fixed_nodes = {0, 3, 4, 7, 11, 12, 15, 19};
    }

    const Real dt = 1e-6;  // 1 microsecond
    const int num_steps = 5;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Element size: " << L << " × " << W << " × " << H << " m" << std::endl;
    std::cout << "  Time step: " << dt << " s" << std::endl;
    std::cout << "  External force: " << f_ext[1*3+2] << " N at node 1 (-z)" << std::endl;
    std::cout << "  Fixed nodes: " << fixed_nodes.size() << " nodes at x=0" << std::endl;
    std::cout << "  Mass at node 1: " << M_lumped[1*3+2] << " kg" << std::endl;
    std::cout << std::endl;

    std::cout << "Step | uz[1]          | fz_int[1]      | az[1]          | Sign Check" << std::endl;
    std::cout << "-----+----------------+----------------+----------------+------------------" << std::endl;

    Real prev_fz = 0.0;
    int sign_flips = 0;

    for (int step = 0; step <= num_steps; ++step) {
        // Compute internal force: f_int = K * u
        std::fill(f_int.begin(), f_int.end(), 0.0);
        for (int i = 0; i < n_nodes * 3; ++i) {
            for (int j = 0; j < n_nodes * 3; ++j) {
                f_int[i] += K[i * n_nodes * 3 + j] * u[j];
            }
        }

        // Compute acceleration: a = (f_ext - f_int) / M
        for (int i = 0; i < n_nodes * 3; ++i) {
            if (M_lumped[i] > 1e-15) {
                a[i] = (f_ext[i] - f_int[i]) / M_lumped[i];
            }
        }

        // Apply BCs to acceleration (fixed DOFs)
        for (int node : fixed_nodes) {
            a[node*3 + 0] = 0.0;
            a[node*3 + 1] = 0.0;
            a[node*3 + 2] = 0.0;
        }

        // Check for sign flip
        std::string sign_check = "";
        if (step > 0) {
            Real curr_fz = f_int[1*3 + 2];
            if (prev_fz * curr_fz < 0) {
                sign_flips++;
                Real amplification = std::abs(curr_fz / prev_fz);
                sign_check = "SIGN FLIP! (×" + std::to_string((int)amplification) + ")";
            }
            prev_fz = curr_fz;
        } else {
            prev_fz = f_int[1*3 + 2];
        }

        // Print current state
        std::cout << std::setw(4) << step << " | ";
        std::cout << std::setw(14) << std::scientific << std::setprecision(6) << u[1*3 + 2] << " | ";
        std::cout << std::setw(14) << std::scientific << std::setprecision(6) << f_int[1*3 + 2] << " | ";
        std::cout << std::setw(14) << std::scientific << std::setprecision(6) << a[1*3 + 2] << " | ";
        std::cout << sign_check << std::endl;

        // Time integration (explicit central difference)
        // v^{n+1} = v^n + a^n * dt
        for (int i = 0; i < n_nodes * 3; ++i) {
            v[i] += a[i] * dt;
        }

        // u^{n+1} = u^n + v^{n+1} * dt
        for (int i = 0; i < n_nodes * 3; ++i) {
            u[i] += v[i] * dt;
        }

        // Apply BCs to displacement and velocity (fixed DOFs)
        for (int node : fixed_nodes) {
            u[node*3 + 0] = 0.0;
            u[node*3 + 1] = 0.0;
            u[node*3 + 2] = 0.0;
            v[node*3 + 0] = 0.0;
            v[node*3 + 1] = 0.0;
            v[node*3 + 2] = 0.0;
        }
    }

    std::cout << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  Total sign flips: " << sign_flips << " / " << num_steps << " steps" << std::endl;

    if (sign_flips > 0) {
        std::cout << "  ✗ UNSTABLE: Forces oscillate in sign (numerical instability)" << std::endl;
    } else {
        std::cout << "  ✓ STABLE: No sign flips detected" << std::endl;
    }
}

int main() {
    std::cout << std::setprecision(10);

    std::cout << "====================================================" << std::endl;
    std::cout << "Multi-Step Dynamics Comparison: Hex8 vs Hex20" << std::endl;
    std::cout << "====================================================" << std::endl;
    std::cout << "Test: Cantilevered beam with fixed left face," << std::endl;
    std::cout << "      downward force at top-right corner." << std::endl;
    std::cout << "Goal: Identify when force sign begins to oscillate." << std::endl;

    test_multistep<Hex8Element>("Hex8", 8);
    test_multistep<Hex20Element>("Hex20", 20);

    return 0;
}
