/**
 * @file hex20_stiffness_check.cpp
 * @brief Check Hex20 stiffness matrix properties
 */

#include <nexussim/discretization/hex20.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace nxs;
using namespace nxs::fem;

int main() {
    std::cout << std::setprecision(6);
    std::cout << "===== Hex20 Stiffness Matrix Check =====" << std::endl;

    // Create element with same geometry as dynamic test
    // Note: Using exact same geometry as hex20_dynamic_debug.cpp
    const Real L = 1.0, W = 0.2, H = 0.2;
    std::cout << "Element geometry: L=" << L << ", W=" << W << ", H=" << H << std::endl;
    std::vector<Real> coords(60);

    // Corner nodes
    coords[0*3+0] = 0.0; coords[0*3+1] = 0.0; coords[0*3+2] = 0.0;
    coords[1*3+0] = L;   coords[1*3+1] = 0.0; coords[1*3+2] = 0.0;
    coords[2*3+0] = L;   coords[2*3+1] = W;   coords[2*3+2] = 0.0;
    coords[3*3+0] = 0.0; coords[3*3+1] = W;   coords[3*3+2] = 0.0;
    coords[4*3+0] = 0.0; coords[4*3+1] = 0.0; coords[4*3+2] = H;
    coords[5*3+0] = L;   coords[5*3+1] = 0.0; coords[5*3+2] = H;
    coords[6*3+0] = L;   coords[6*3+1] = W;   coords[6*3+2] = H;
    coords[7*3+0] = 0.0; coords[7*3+1] = W;   coords[7*3+2] = H;

    // Mid-edge nodes
    coords[8*3+0] = 0.5*L; coords[8*3+1] = 0.0;   coords[8*3+2] = 0.0;
    coords[9*3+0] = L;     coords[9*3+1] = 0.5*W; coords[9*3+2] = 0.0;
    coords[10*3+0] = 0.5*L; coords[10*3+1] = W;   coords[10*3+2] = 0.0;
    coords[11*3+0] = 0.0;   coords[11*3+1] = 0.5*W; coords[11*3+2] = 0.0;
    coords[12*3+0] = 0.0;   coords[12*3+1] = 0.0; coords[12*3+2] = 0.5*H;
    coords[13*3+0] = L;     coords[13*3+1] = 0.0; coords[13*3+2] = 0.5*H;
    coords[14*3+0] = L;     coords[14*3+1] = W;   coords[14*3+2] = 0.5*H;
    coords[15*3+0] = 0.0;   coords[15*3+1] = W;   coords[15*3+2] = 0.5*H;
    coords[16*3+0] = 0.5*L; coords[16*3+1] = 0.0; coords[16*3+2] = H;
    coords[17*3+0] = L;     coords[17*3+1] = 0.5*W; coords[17*3+2] = H;
    coords[18*3+0] = 0.5*L; coords[18*3+1] = W;   coords[18*3+2] = H;
    coords[19*3+0] = 0.0;   coords[19*3+1] = 0.5*W; coords[19*3+2] = H;

    const Real E = 210e9, nu = 0.3;
    Hex20Element elem;

    // Compute stiffness matrix
    std::vector<Real> K(60*60);
    elem.stiffness_matrix(coords.data(), E, nu, K.data());

    // Check symmetry
    std::cout << "\n--- Symmetry Check ---" << std::endl;
    Real max_asym = 0;
    int max_i = 0, max_j = 0;
    for (int i = 0; i < 60; ++i) {
        for (int j = i+1; j < 60; ++j) {
            Real asym = std::abs(K[i*60+j] - K[j*60+i]);
            Real avg = 0.5 * (std::abs(K[i*60+j]) + std::abs(K[j*60+i]));
            if (avg > 1e-10 && asym / avg > max_asym) {
                max_asym = asym / avg;
                max_i = i; max_j = j;
            }
        }
    }
    std::cout << "Max relative asymmetry: " << max_asym << " at (" << max_i << "," << max_j << ")" << std::endl;
    std::cout << "K[" << max_i << "][" << max_j << "] = " << K[max_i*60+max_j] << std::endl;
    std::cout << "K[" << max_j << "][" << max_i << "] = " << K[max_j*60+max_i] << std::endl;

    // Check diagonal entries (should be positive)
    std::cout << "\n--- Diagonal Check ---" << std::endl;
    int neg_diag = 0;
    for (int i = 0; i < 60; ++i) {
        if (K[i*60+i] < 0) {
            neg_diag++;
            std::cout << "NEGATIVE diagonal: K[" << i << "][" << i << "] = " << K[i*60+i] << std::endl;
        }
    }
    std::cout << "Number of negative diagonals: " << neg_diag << std::endl;

    // Check strain energy for various displacement modes
    std::cout << "\n--- Strain Energy Check ---" << std::endl;

    // Test: uniform z-displacement on top face (tension)
    std::vector<Real> u_tension(60, 0.0);
    for (int node : {4, 5, 6, 7, 16, 17, 18, 19}) {
        u_tension[node*3+2] = 0.001;
    }

    Real energy_tension = 0;
    for (int i = 0; i < 60; ++i) {
        for (int j = 0; j < 60; ++j) {
            energy_tension += 0.5 * u_tension[i] * K[i*60+j] * u_tension[j];
        }
    }
    std::cout << "Tension test: strain energy = " << energy_tension << " J (should be > 0)" << std::endl;

    // Test: bending-like displacement (tip down)
    std::vector<Real> u_bending(60, 0.0);
    for (int node = 0; node < 20; ++node) {
        Real x = coords[node*3+0];
        u_bending[node*3+2] = -0.001 * x / L;  // Linear z-displacement with x
    }

    Real energy_bending = 0;
    for (int i = 0; i < 60; ++i) {
        for (int j = 0; j < 60; ++j) {
            energy_bending += 0.5 * u_bending[i] * K[i*60+j] * u_bending[j];
        }
    }
    std::cout << "Bending test: strain energy = " << energy_bending << " J (should be > 0)" << std::endl;

    // Test: f = K*u check
    std::cout << "\n--- Force Direction Check (K*u) ---" << std::endl;
    std::vector<Real> f_tension(60, 0.0);
    for (int i = 0; i < 60; ++i) {
        for (int j = 0; j < 60; ++j) {
            f_tension[i] += K[i*60+j] * u_tension[j];
        }
    }

    // For tension, internal force at top nodes should be positive (same as displacement)
    Real f_top_z = 0;
    for (int node : {4, 5, 6, 7, 16, 17, 18, 19}) {
        f_top_z += f_tension[node*3+2];
    }
    std::cout << "Top face total fz from K*u = " << f_top_z << " (should be > 0 for positive uz)" << std::endl;

    // Now test direct B^T*sigma integration
    std::cout << "\n--- Direct Integration Check (B^T*sigma) ---" << std::endl;

    const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const Real mu = E / (2.0 * (1.0 + nu));

    std::vector<Real> f_direct(60, 0.0);
    std::vector<Real> gp_coords(27*3), gp_weights(27);
    elem.gauss_quadrature(gp_coords.data(), gp_weights.data());

    for (int gp = 0; gp < 27; ++gp) {
        Real xi[3] = {gp_coords[gp*3], gp_coords[gp*3+1], gp_coords[gp*3+2]};
        Real w = gp_weights[gp];

        std::vector<Real> B(6*60);
        elem.strain_displacement_matrix(xi, coords.data(), B.data());

        Real J[9];
        Real detJ = elem.jacobian(xi, coords.data(), J);

        // Compute strain
        std::vector<Real> strain(6, 0.0);
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 60; ++j) {
                strain[i] += B[i*60+j] * u_tension[j];
            }
        }

        // Compute stress
        std::vector<Real> stress(6);
        stress[0] = (lambda+2*mu)*strain[0] + lambda*(strain[1]+strain[2]);
        stress[1] = (lambda+2*mu)*strain[1] + lambda*(strain[0]+strain[2]);
        stress[2] = (lambda+2*mu)*strain[2] + lambda*(strain[0]+strain[1]);
        stress[3] = mu*strain[3];
        stress[4] = mu*strain[4];
        stress[5] = mu*strain[5];

        // Compute force
        Real factor = detJ * w;
        for (int i = 0; i < 60; ++i) {
            for (int j = 0; j < 6; ++j) {
                f_direct[i] += B[j*60+i] * stress[j] * factor;
            }
        }
    }

    Real f_top_z_direct = 0;
    for (int node : {4, 5, 6, 7, 16, 17, 18, 19}) {
        f_top_z_direct += f_direct[node*3+2];
    }
    std::cout << "Top face total fz from B^T*sigma = " << f_top_z_direct << std::endl;

    // Compare
    Real max_diff = 0;
    for (int i = 0; i < 60; ++i) {
        max_diff = std::max(max_diff, std::abs(f_tension[i] - f_direct[i]));
    }
    std::cout << "Max |K*u - B^T*sigma| = " << max_diff << std::endl;

    // Check work
    Real work_Ku = 0, work_direct = 0;
    for (int i = 0; i < 60; ++i) {
        work_Ku += f_tension[i] * u_tension[i];
        work_direct += f_direct[i] * u_tension[i];
    }
    std::cout << "\nWork f.u from K*u = " << work_Ku << std::endl;
    std::cout << "Work f.u from B^T*sigma = " << work_direct << std::endl;
    std::cout << "2 * Strain energy (u^T K u) = " << 2*energy_tension << std::endl;

    if (work_Ku > 0 && work_direct > 0) {
        std::cout << "\n✓ PASS: Work is positive for both methods" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ FAIL: Work should be positive!" << std::endl;
        return 1;
    }
}
