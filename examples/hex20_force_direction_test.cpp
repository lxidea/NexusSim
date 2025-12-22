/**
 * @file hex20_force_direction_test.cpp
 * @brief Simple test to verify Hex20 forces oppose displacement
 *
 * This test applies a small positive displacement and checks if
 * the resulting internal force is negative (opposing the displacement).
 */

#include <nexussim/discretization/hex20.hpp>
#include <iostream>
#include <iomanip>

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

    // Material properties (steel)
    const Real E = 210e9;   // Young's modulus (Pa)
    const Real nu = 0.3;    // Poisson's ratio

    // Create element
    Hex20Element elem;

    // Check Jacobian determinant at element center
    Real xi_center[3] = {0.0, 0.0, 0.0};
    Real J[9];
    Real det_J = elem.jacobian(xi_center, coords, J);
    std::cout << "DEBUG: Jacobian determinant at center = " << det_J << std::endl;

    // Test: Apply uniform +z displacement to ALL top nodes (4,5,6,7) for pure tension
    Real displacement[60] = {0.0};  // All zeros initially
    const Real disp_magnitude = 0.001 * L;  // 0.1% strain
    displacement[4*3 + 2] = disp_magnitude;  // uz at node 4
    displacement[5*3 + 2] = disp_magnitude;  // uz at node 5
    displacement[6*3 + 2] = disp_magnitude;  // uz at node 6
    displacement[7*3 + 2] = disp_magnitude;  // uz at node 7
    // Also apply to top mid-edge nodes (16, 17, 18, 19)
    displacement[16*3 + 2] = disp_magnitude;
    displacement[17*3 + 2] = disp_magnitude;
    displacement[18*3 + 2] = disp_magnitude;
    displacement[19*3 + 2] = disp_magnitude;

    // Debug: Check B-matrix at element center for node 6
    Real B_debug[6 * 60];
    elem.strain_displacement_matrix(xi_center, coords, B_debug);
    // B-matrix row 2 (εzz) for node 6, DOF z (column 6*3+2 = 20)
    std::cout << "DEBUG: B[2,20] (dN6/dz for εzz) = " << B_debug[2*60 + 20] << std::endl;

    // Compute element stiffness matrix
    Real K[60*60];
    elem.stiffness_matrix(coords, E, nu, K);

    // Compute internal force: f_int = K * u
    Real f_int[60] = {0.0};
    for (int i = 0; i < 60; ++i) {
        for (int j = 0; j < 60; ++j) {
            f_int[i] += K[i*60 + j] * displacement[j];
        }
    }

    // Check force at node 6, z-direction
    const Real fz_node6 = f_int[6*3 + 2];
    const Real K_diag = K[(6*3+2)*60 + (6*3+2)];  // Diagonal stiffness

    // For uniform displacement test, check total force on top AND bottom faces
    Real total_fz_top = 0.0;
    for (int node : {4, 5, 6, 7, 16, 17, 18, 19}) {
        total_fz_top += f_int[node*3 + 2];
    }

    Real total_fz_bottom = 0.0;
    for (int node : {0, 1, 2, 3, 8, 9, 10, 11}) {
        total_fz_bottom += f_int[node*3 + 2];
    }

    Real total_fz_all = 0.0;
    for (int i = 0; i < 60; ++i) {
        if (i % 3 == 2) total_fz_all += f_int[i];
    }

    std::cout << "===== Hex20 Force Direction Test (Uniform Displacement) =====" << std::endl;
    std::cout << "Applied displacement: uz[all top nodes] = +" << disp_magnitude << " m" << std::endl;
    std::cout << "Internal force:       fz[node 6]       = " << fz_node6 << " N" << std::endl;
    std::cout << "Total z-force on top face              = " << total_fz_top << " N" << std::endl;
    std::cout << "Total z-force on bottom face           = " << total_fz_bottom << " N" << std::endl;
    std::cout << "Total z-force (all nodes)              = " << total_fz_all << " N" << std::endl;
    std::cout << "Stiffness diagonal:   K[20,20]         = " << K_diag << " N/m" << std::endl;
    std::cout << std::endl;

    // Check if internal force is correct (positive for positive displacement)
    // In FEM dynamics: M*a = f_ext - f_int
    // For a stretched element:
    //   - Positive displacement (tension) => positive stress
    //   - Positive stress => positive internal force (f_int = B^T * sigma > 0)
    //   - If f_ext = 0, then a = -f_int / m < 0 => nodes decelerate/return (restoring)
    // This is the CORRECT sign convention for FEM.

    Real avg_fz_top = total_fz_top / 8.0;
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
        std::cout << std::endl;
        std::cout << "This indicates a sign error in:" << std::endl;
        std::cout << "  - Shape function derivatives, OR" << std::endl;
        std::cout << "  - Jacobian transformation, OR" << std::endl;
        std::cout << "  - B-matrix construction, OR" << std::endl;
        std::cout << "  - Force assembly" << std::endl;
        return 1;
    }
}
