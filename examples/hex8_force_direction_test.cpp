/**
 * @file hex8_force_direction_test.cpp
 * @brief Test Hex8 forces for comparison with Hex20
 */

#include <nexussim/discretization/hex8.hpp>
#include <iostream>
#include <iomanip>

using namespace nxs;
using namespace nxs::fem;

int main() {
    std::cout << std::setprecision(10);

    // Create a unit cube Hex8 element
    const Real L = 1.0;
    Real coords[24]; // 8 nodes × 3 coords

    // Corner nodes (0-7)
    coords[0*3+0] = 0.0; coords[0*3+1] = 0.0; coords[0*3+2] = 0.0;
    coords[1*3+0] = L;   coords[1*3+1] = 0.0; coords[1*3+2] = 0.0;
    coords[2*3+0] = L;   coords[2*3+1] = L;   coords[2*3+2] = 0.0;
    coords[3*3+0] = 0.0; coords[3*3+1] = L;   coords[3*3+2] = 0.0;
    coords[4*3+0] = 0.0; coords[4*3+1] = 0.0; coords[4*3+2] = L;
    coords[5*3+0] = L;   coords[5*3+1] = 0.0; coords[5*3+2] = L;
    coords[6*3+0] = L;   coords[6*3+1] = L;   coords[6*3+2] = L;
    coords[7*3+0] = 0.0; coords[7*3+1] = L;   coords[7*3+2] = L;

    // Material properties (steel)
    const Real E = 210e9;   // Young's modulus (Pa)
    const Real nu = 0.3;    // Poisson's ratio

    // Create element
    Hex8Element elem;

    // Check Jacobian determinant at element center
    Real xi_center[3] = {0.0, 0.0, 0.0};
    Real J[9];
    Real det_J = elem.jacobian(xi_center, coords, J);
    std::cout << "DEBUG: Jacobian determinant at center = " << det_J << std::endl;

    // Test: Apply uniform +z displacement to ALL top nodes (4,5,6,7)
    Real displacement[24] = {0.0};
    const Real disp_magnitude = 0.001 * L;
    displacement[4*3 + 2] = disp_magnitude;
    displacement[5*3 + 2] = disp_magnitude;
    displacement[6*3 + 2] = disp_magnitude;
    displacement[7*3 + 2] = disp_magnitude;

    // Debug: Check B-matrix at element center for node 6
    Real B_debug[6 * 24];
    elem.strain_displacement_matrix(xi_center, coords, B_debug);
    std::cout << "DEBUG: B[2,20] (dN6/dz for εzz) = " << B_debug[2*24 + 20] << std::endl;

    // Compute element stiffness matrix
    Real K[24*24];
    elem.stiffness_matrix(coords, E, nu, K);

    // Compute internal force: f_int = K * u
    Real f_int[24] = {0.0};
    for (int i = 0; i < 24; ++i) {
        for (int j = 0; j < 24; ++j) {
            f_int[i] += K[i*24 + j] * displacement[j];
        }
    }

    // Check force at node 6
    const Real fz_node6 = f_int[6*3 + 2];
    const Real K_diag = K[(6*3+2)*24 + (6*3+2)];

    // Check total force on top face
    Real total_fz_top = 0.0;
    for (int node : {4, 5, 6, 7}) {
        total_fz_top += f_int[node*3 + 2];
    }

    std::cout << "===== Hex8 Force Direction Test (Uniform Displacement) =====" << std::endl;
    std::cout << "Applied displacement: uz[all top nodes] = +" << disp_magnitude << " m" << std::endl;
    std::cout << "Internal force:       fz[node 6]       = " << fz_node6 << " N" << std::endl;
    std::cout << "Total z-force on top face              = " << total_fz_top << " N" << std::endl;
    std::cout << "Stiffness diagonal:   K[20,20]         = " << K_diag << " N/m" << std::endl;
    std::cout << std::endl;

    // Check if internal force is correct (positive for positive displacement)
    // In FEM dynamics: M*a = f_ext - f_int
    // For a stretched element:
    //   - Positive displacement (tension) => positive stress
    //   - Positive stress => positive internal force (f_int = B^T * sigma > 0)
    //   - If f_ext = 0, then a = -f_int / m < 0 => nodes decelerate/return (restoring)
    Real avg_fz_top = total_fz_top / 4.0;
    bool correct_sign = (avg_fz_top > 0.0);  // f_int should be POSITIVE for POSITIVE displacement

    if (correct_sign) {
        std::cout << "✓ PASS: Internal force has correct sign for FEM dynamics" << std::endl;
        std::cout << "  Displacement: +z (tension)" << std::endl;
        std::cout << "  Internal Force: +z" << std::endl;
        std::cout << "  In dynamics: M*a = f_ext - f_int => a < 0 (restoring)" << std::endl;
        return 0;
    } else {
        std::cout << "✗ FAIL: Internal force has WRONG sign!" << std::endl;
        std::cout << "  Displacement: +z (tension)" << std::endl;
        std::cout << "  Internal Force: -z (wrong - would cause instability)" << std::endl;
        return 1;
    }
}
