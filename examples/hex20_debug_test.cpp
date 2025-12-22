/**
 * @file hex20_debug_test.cpp
 * @brief Debug test to trace through Hex20 internal force computation
 */

#include <nexussim/discretization/hex20.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;

int main() {
    std::cout << std::setprecision(12);

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
    const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const Real mu = E / (2.0 * (1.0 + nu));

    // Create element
    Hex20Element elem;

    std::cout << "=== Hex20 Debug Test ===" << std::endl;
    std::cout << "E = " << E << " Pa" << std::endl;
    std::cout << "nu = " << nu << std::endl;
    std::cout << "lambda = " << lambda << std::endl;
    std::cout << "mu = " << mu << std::endl << std::endl;

    // Apply uniform z-displacement to top face (nodes 4-7, 16-19)
    Real displacement[60] = {0.0};
    const Real disp_magnitude = 0.001 * L;  // 0.1% strain

    // Top corner nodes
    displacement[4*3 + 2] = disp_magnitude;
    displacement[5*3 + 2] = disp_magnitude;
    displacement[6*3 + 2] = disp_magnitude;
    displacement[7*3 + 2] = disp_magnitude;
    // Top mid-edge nodes
    displacement[16*3 + 2] = disp_magnitude;
    displacement[17*3 + 2] = disp_magnitude;
    displacement[18*3 + 2] = disp_magnitude;
    displacement[19*3 + 2] = disp_magnitude;

    std::cout << "Applied displacement: uz = " << disp_magnitude << " m on top nodes" << std::endl;
    std::cout << std::endl;

    // Test at element center (0,0,0)
    Real xi[3] = {0.0, 0.0, 0.0};

    // Get B-matrix
    Real B[6 * 60];
    elem.strain_displacement_matrix(xi, coords, B);

    // Compute strain: ε = B * u
    Real strain[6] = {0.0};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 60; ++j) {
            strain[i] += B[i * 60 + j] * displacement[j];
        }
    }

    std::cout << "Strain at center (ξ=0,0,0):" << std::endl;
    std::cout << "  εxx = " << strain[0] << std::endl;
    std::cout << "  εyy = " << strain[1] << std::endl;
    std::cout << "  εzz = " << strain[2] << std::endl;
    std::cout << "  γxy = " << strain[3] << std::endl;
    std::cout << "  γyz = " << strain[4] << std::endl;
    std::cout << "  γxz = " << strain[5] << std::endl;
    std::cout << std::endl;

    // Compute stress: σ = C * ε
    Real stress[6];
    stress[0] = (lambda + 2.0 * mu) * strain[0] + lambda * (strain[1] + strain[2]);
    stress[1] = (lambda + 2.0 * mu) * strain[1] + lambda * (strain[0] + strain[2]);
    stress[2] = (lambda + 2.0 * mu) * strain[2] + lambda * (strain[0] + strain[1]);
    stress[3] = mu * strain[3];
    stress[4] = mu * strain[4];
    stress[5] = mu * strain[5];

    std::cout << "Stress at center:" << std::endl;
    std::cout << "  σxx = " << stress[0] << " Pa" << std::endl;
    std::cout << "  σyy = " << stress[1] << " Pa" << std::endl;
    std::cout << "  σzz = " << stress[2] << " Pa" << std::endl;
    std::cout << "  τxy = " << stress[3] << " Pa" << std::endl;
    std::cout << "  τyz = " << stress[4] << " Pa" << std::endl;
    std::cout << "  τxz = " << stress[5] << " Pa" << std::endl;
    std::cout << std::endl;

    // Get Jacobian determinant
    Real J[9];
    Real det_J = elem.jacobian(xi, coords, J);
    std::cout << "Jacobian determinant at center: " << det_J << std::endl;
    std::cout << std::endl;

    // Compute internal force using the CPU fallback method (same as fem_solver.cpp)
    // Get Gauss points
    Real gp[27*3], gw[27];
    elem.gauss_quadrature(gp, gw);

    Real f_int[60] = {0.0};

    for (int gp_idx = 0; gp_idx < 27; ++gp_idx) {
        Real xi_gp[3] = {gp[gp_idx*3+0], gp[gp_idx*3+1], gp[gp_idx*3+2]};
        Real weight = gw[gp_idx];

        // B-matrix at Gauss point
        Real B_gp[6 * 60];
        elem.strain_displacement_matrix(xi_gp, coords, B_gp);

        // Jacobian at Gauss point
        Real J_gp[9];
        Real det_J_gp = elem.jacobian(xi_gp, coords, J_gp);

        // Strain at Gauss point
        Real strain_gp[6] = {0.0};
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 60; ++j) {
                strain_gp[i] += B_gp[i * 60 + j] * displacement[j];
            }
        }

        // Stress at Gauss point
        Real stress_gp[6];
        stress_gp[0] = (lambda + 2.0 * mu) * strain_gp[0] + lambda * (strain_gp[1] + strain_gp[2]);
        stress_gp[1] = (lambda + 2.0 * mu) * strain_gp[1] + lambda * (strain_gp[0] + strain_gp[2]);
        stress_gp[2] = (lambda + 2.0 * mu) * strain_gp[2] + lambda * (strain_gp[0] + strain_gp[1]);
        stress_gp[3] = mu * strain_gp[3];
        stress_gp[4] = mu * strain_gp[4];
        stress_gp[5] = mu * strain_gp[5];

        // Accumulate internal force: f += B^T * σ * detJ * weight
        Real factor = det_J_gp * weight;
        for (int j = 0; j < 60; ++j) {
            for (int i = 0; i < 6; ++i) {
                f_int[j] += B_gp[i * 60 + j] * stress_gp[i] * factor;
            }
        }
    }

    // Print forces on top nodes (z-direction)
    std::cout << "Internal forces (z-direction) at top nodes:" << std::endl;
    Real total_fz_top = 0.0;
    for (int node : {4, 5, 6, 7, 16, 17, 18, 19}) {
        Real fz = f_int[node*3 + 2];
        total_fz_top += fz;
        std::cout << "  f_int_z[node " << node << "] = " << fz << " N" << std::endl;
    }
    std::cout << "  TOTAL on top face = " << total_fz_top << " N" << std::endl;
    std::cout << std::endl;

    // Print forces on bottom nodes (z-direction)
    std::cout << "Internal forces (z-direction) at bottom nodes:" << std::endl;
    Real total_fz_bottom = 0.0;
    for (int node : {0, 1, 2, 3, 8, 9, 10, 11}) {
        Real fz = f_int[node*3 + 2];
        total_fz_bottom += fz;
        std::cout << "  f_int_z[node " << node << "] = " << fz << " N" << std::endl;
    }
    std::cout << "  TOTAL on bottom face = " << total_fz_bottom << " N" << std::endl;
    std::cout << std::endl;

    // Total z-force (should be ~0 for equilibrium)
    Real total_fz = 0.0;
    for (int i = 0; i < 60; i += 3) {
        total_fz += f_int[i + 2];
    }
    std::cout << "Total z-force (all nodes) = " << total_fz << " N (should be ~0)" << std::endl;
    std::cout << std::endl;

    // Analytical check: For uniform axial strain, total force should be σ * A
    // Area = 1 m² for unit cube
    // εzz = disp_magnitude / L = 0.001
    // σzz = (λ + 2μ) * εzz + λ * (εxx + εyy) for constrained case
    // For free-lateral case, εxx = εyy = -ν * εzz, so:
    // σzz = E * εzz = 210e9 * 0.001 = 210e6 Pa
    // Force = σzz * A = 210e6 * 1 = 210e6 N

    std::cout << "=== Analysis ===" << std::endl;
    std::cout << "Applied strain (approximate): εzz = " << disp_magnitude / L << std::endl;

    // For uniform tension with Poisson effect (free lateral expansion)
    // The stress is NOT E * ε because we're constraining lateral motion
    // In this test, bottom face is fixed (uz=0) but we only move top nodes
    // There's no lateral constraint, so this should approximate uniaxial tension

    // Expected: F = E * A * ε = 210e9 * 1.0 * 0.001 = 210e6 N
    // But our boundary conditions create more complex stress state

    Real expected_force = E * 1.0 * (disp_magnitude / L);  // E * A * ε
    std::cout << "Expected total force (E*A*ε): " << expected_force << " N" << std::endl;
    std::cout << "Computed total on top: " << total_fz_top << " N" << std::endl;
    std::cout << "Ratio: " << total_fz_top / expected_force << std::endl;
    std::cout << std::endl;

    // Check sign convention
    std::cout << "=== Sign Convention Check ===" << std::endl;
    std::cout << "Positive displacement (top moves +z)" << std::endl;
    std::cout << "Element stretched (tensile stress)" << std::endl;
    std::cout << "Internal force sign on top: " << (total_fz_top > 0 ? "POSITIVE" : "NEGATIVE") << std::endl;
    std::cout << std::endl;
    std::cout << "In FEM dynamics: M*a = f_ext - f_int" << std::endl;
    std::cout << "If f_int > 0 at top (where f_ext = 0 after loading):" << std::endl;
    std::cout << "  => a < 0 => nodes decelerate/return (CORRECT restoring behavior)" << std::endl;
    std::cout << "If f_int < 0 at top:" << std::endl;
    std::cout << "  => a > 0 => nodes accelerate further (WRONG - unstable!)" << std::endl;
    std::cout << std::endl;

    if (total_fz_top > 0) {
        std::cout << "✓ SIGN IS CORRECT: Internal force opposes motion" << std::endl;
        return 0;
    } else {
        std::cout << "✗ SIGN IS WRONG: Internal force assists motion (unstable!)" << std::endl;
        return 1;
    }
}
