/**
 * @file hex20_force_compare.cpp
 * @brief Compare Hex20 internal force from stiffness matrix vs direct calculation
 */

#include <nexussim/discretization/hex20.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;

int main() {
    std::cout << std::setprecision(6);
    std::cout << "===== Hex20 Force Calculation Comparison =====" << std::endl;

    // Create a NON-CUBIC Hex20 element (same as dynamic debug test)
    const Real L = 1.0;
    const Real W = 0.2;  // Width
    const Real H = 0.2;  // Height
    std::cout << "Element geometry: L=" << L << ", W=" << W << ", H=" << H << std::endl;

    Real coords[60]; // 20 nodes × 3 coords

    // Corner nodes (0-7)
    coords[0*3+0] = 0.0; coords[0*3+1] = 0.0; coords[0*3+2] = 0.0;
    coords[1*3+0] = L;   coords[1*3+1] = 0.0; coords[1*3+2] = 0.0;
    coords[2*3+0] = L;   coords[2*3+1] = W;   coords[2*3+2] = 0.0;
    coords[3*3+0] = 0.0; coords[3*3+1] = W;   coords[3*3+2] = 0.0;
    coords[4*3+0] = 0.0; coords[4*3+1] = 0.0; coords[4*3+2] = H;
    coords[5*3+0] = L;   coords[5*3+1] = 0.0; coords[5*3+2] = H;
    coords[6*3+0] = L;   coords[6*3+1] = W;   coords[6*3+2] = H;
    coords[7*3+0] = 0.0; coords[7*3+1] = W;   coords[7*3+2] = H;

    // Mid-edge nodes (8-19)
    coords[8*3+0] = 0.5*L; coords[8*3+1] = 0.0;   coords[8*3+2] = 0.0;
    coords[9*3+0] = L;     coords[9*3+1] = 0.5*W; coords[9*3+2] = 0.0;
    coords[10*3+0] = 0.5*L; coords[10*3+1] = W;   coords[10*3+2] = 0.0;
    coords[11*3+0] = 0.0;   coords[11*3+1] = 0.5*W; coords[11*3+2] = 0.0;
    coords[12*3+0] = 0.0;   coords[12*3+1] = 0.0;   coords[12*3+2] = 0.5*H;
    coords[13*3+0] = L;     coords[13*3+1] = 0.0;   coords[13*3+2] = 0.5*H;
    coords[14*3+0] = L;     coords[14*3+1] = W;     coords[14*3+2] = 0.5*H;
    coords[15*3+0] = 0.0;   coords[15*3+1] = W;     coords[15*3+2] = 0.5*H;
    coords[16*3+0] = 0.5*L; coords[16*3+1] = 0.0;   coords[16*3+2] = H;
    coords[17*3+0] = L;     coords[17*3+1] = 0.5*W; coords[17*3+2] = H;
    coords[18*3+0] = 0.5*L; coords[18*3+1] = W;     coords[18*3+2] = H;
    coords[19*3+0] = 0.0;   coords[19*3+1] = 0.5*W; coords[19*3+2] = H;

    // Material properties
    const Real E = 210e9;
    const Real nu = 0.3;
    const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const Real mu = E / (2.0 * (1.0 + nu));

    Hex20Element elem;

    // Apply displacement - test with non-uniform like dynamic simulation
    const bool use_uniform = false;  // Set true for uniform test

    Real displacement[60] = {0.0};
    const Real disp_val = 1e-9;  // Very small

    if (use_uniform) {
        // Top face nodes get same displacement
        for (int node : {4, 5, 6, 7, 16, 17, 18, 19}) {
            displacement[node*3 + 2] = disp_val;
        }
        std::cout << "Using UNIFORM displacement" << std::endl;
    } else {
        // Non-uniform: loaded nodes (right face) get different displacements
        // based on mass ratio (corner:mid-edge = 1:4 mass, so 4:1 acceleration)
        for (int node : {1, 2, 5, 6}) {  // Corner nodes on right face
            displacement[node*3 + 2] = -disp_val * 4.0;
        }
        for (int node : {9, 13, 14, 17}) {  // Mid-edge nodes on right face
            displacement[node*3 + 2] = -disp_val * 1.0;
        }
        // Unloaded mid-edge nodes (not on right face) stay at zero
        std::cout << "Using NON-UNIFORM displacement (mimics HRZ mass ratio)" << std::endl;
    }

    // Method 1: f_int = K * u (stiffness matrix approach)
    std::cout << "\n--- Method 1: Stiffness Matrix (f = K*u) ---" << std::endl;
    Real K[60*60];
    elem.stiffness_matrix(coords, E, nu, K);

    Real f_stiff[60] = {0.0};
    for (int i = 0; i < 60; ++i) {
        for (int j = 0; j < 60; ++j) {
            f_stiff[i] += K[i*60 + j] * displacement[j];
        }
    }

    // Method 2: Direct internal force calculation (mimics FEM solver CPU fallback)
    std::cout << "\n--- Method 2: Direct B^T*sigma Integration ---" << std::endl;

    Real f_direct[60] = {0.0};

    // Use 27-point Gauss quadrature
    const Real a = std::sqrt(0.6);
    const Real gp[3] = {-a, 0.0, a};
    const Real gw[3] = {5.0/9.0, 8.0/9.0, 5.0/9.0};

    for (int gi = 0; gi < 3; ++gi) {
        for (int gj = 0; gj < 3; ++gj) {
            for (int gk = 0; gk < 3; ++gk) {
                Real xi[3] = {gp[gi], gp[gj], gp[gk]};
                Real weight = gw[gi] * gw[gj] * gw[gk];

                // Compute B-matrix
                Real B[6 * 60];
                elem.strain_displacement_matrix(xi, coords, B);

                // Compute Jacobian determinant
                Real J[9];
                Real detJ = elem.jacobian(xi, coords, J);

                // Compute strain: ε = B * u
                Real strain[6] = {0.0};
                for (int i = 0; i < 6; ++i) {
                    for (int j = 0; j < 60; ++j) {
                        strain[i] += B[i * 60 + j] * displacement[j];
                    }
                }

                // Compute stress: σ = C * ε (isotropic elastic)
                Real stress[6];
                stress[0] = (lambda + 2.0*mu) * strain[0] + lambda * (strain[1] + strain[2]);
                stress[1] = (lambda + 2.0*mu) * strain[1] + lambda * (strain[0] + strain[2]);
                stress[2] = (lambda + 2.0*mu) * strain[2] + lambda * (strain[0] + strain[1]);
                stress[3] = mu * strain[3];
                stress[4] = mu * strain[4];
                stress[5] = mu * strain[5];

                // Compute internal force: f += B^T * σ * detJ * weight
                Real factor = detJ * weight;
                for (int i = 0; i < 60; ++i) {
                    for (int j = 0; j < 6; ++j) {
                        f_direct[i] += B[j * 60 + i] * stress[j] * factor;
                    }
                }
            }
        }
    }

    // Compare results
    std::cout << "\n--- Comparison (Node z-forces) ---" << std::endl;
    std::cout << std::setw(6) << "Node"
              << std::setw(16) << "f_stiff"
              << std::setw(16) << "f_direct"
              << std::setw(16) << "Difference"
              << std::endl;

    Real max_diff = 0.0;
    Real max_f = 0.0;
    for (int node = 0; node < 20; ++node) {
        int dof = node * 3 + 2;  // z-direction
        Real diff = f_stiff[dof] - f_direct[dof];
        max_diff = std::max(max_diff, std::abs(diff));
        max_f = std::max(max_f, std::max(std::abs(f_stiff[dof]), std::abs(f_direct[dof])));

        std::cout << std::setw(6) << node
                  << std::setw(16) << f_stiff[dof]
                  << std::setw(16) << f_direct[dof]
                  << std::setw(16) << diff
                  << std::endl;
    }

    std::cout << "\n--- Summary ---" << std::endl;
    std::cout << "Max absolute difference: " << max_diff << " N" << std::endl;
    std::cout << "Max force magnitude: " << max_f << " N" << std::endl;
    std::cout << "Relative difference: " << (max_f > 0 ? max_diff/max_f * 100.0 : 0.0) << "%" << std::endl;

    // Total z-force on top and bottom faces
    Real total_z_top_stiff = 0.0, total_z_top_direct = 0.0;
    Real total_z_bot_stiff = 0.0, total_z_bot_direct = 0.0;

    for (int node : {4, 5, 6, 7, 16, 17, 18, 19}) {
        total_z_top_stiff += f_stiff[node*3 + 2];
        total_z_top_direct += f_direct[node*3 + 2];
    }
    for (int node : {0, 1, 2, 3, 8, 9, 10, 11}) {
        total_z_bot_stiff += f_stiff[node*3 + 2];
        total_z_bot_direct += f_direct[node*3 + 2];
    }

    std::cout << "\nTotal z-force on top face:" << std::endl;
    std::cout << "  Stiffness: " << total_z_top_stiff << " N" << std::endl;
    std::cout << "  Direct:    " << total_z_top_direct << " N" << std::endl;

    std::cout << "\nTotal z-force on bottom face:" << std::endl;
    std::cout << "  Stiffness: " << total_z_bot_stiff << " N" << std::endl;
    std::cout << "  Direct:    " << total_z_bot_direct << " N" << std::endl;

    // Check for discrepancy
    if (max_diff / max_f > 0.01) {
        std::cout << "\n✗ SIGNIFICANT DISCREPANCY between stiffness and direct methods!" << std::endl;
        return 1;
    } else {
        std::cout << "\n✓ Methods agree (difference < 1%)" << std::endl;
        return 0;
    }
}
