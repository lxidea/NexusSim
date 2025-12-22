/**
 * @file hex20_dynamic_debug.cpp
 * @brief Debug Hex20 dynamic simulation step-by-step
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
    std::cout << "===== Hex20 Dynamic Simulation Debug =====" << std::endl;

    // Create a unit cube Hex20 element (same as single_element_bending test)
    const Real L = 1.0;
    const Real W = 0.2;
    const Real H = 0.2;

    std::vector<Real> coords(60);

    // Corner nodes (0-7)
    coords[0*3+0] = 0.0; coords[0*3+1] = 0.0;   coords[0*3+2] = 0.0;
    coords[1*3+0] = L;   coords[1*3+1] = 0.0;   coords[1*3+2] = 0.0;
    coords[2*3+0] = L;   coords[2*3+1] = W;     coords[2*3+2] = 0.0;
    coords[3*3+0] = 0.0; coords[3*3+1] = W;     coords[3*3+2] = 0.0;
    coords[4*3+0] = 0.0; coords[4*3+1] = 0.0;   coords[4*3+2] = H;
    coords[5*3+0] = L;   coords[5*3+1] = 0.0;   coords[5*3+2] = H;
    coords[6*3+0] = L;   coords[6*3+1] = W;     coords[6*3+2] = H;
    coords[7*3+0] = 0.0; coords[7*3+1] = W;     coords[7*3+2] = H;

    // Mid-edge nodes (8-19)
    coords[8*3+0] = 0.5*L; coords[8*3+1] = 0.0;     coords[8*3+2] = 0.0;
    coords[9*3+0] = L;     coords[9*3+1] = 0.5*W;   coords[9*3+2] = 0.0;
    coords[10*3+0] = 0.5*L; coords[10*3+1] = W;     coords[10*3+2] = 0.0;
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
    const Real density = 7850.0;
    const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const Real mu = E / (2.0 * (1.0 + nu));

    Hex20Element elem;

    // Compute lumped mass
    // Option 1: HRZ method (gives different masses to corner vs mid-edge nodes)
    // Option 2: Uniform mass (equal mass to all nodes)
    const bool use_hrz = false;  // Test with uniform mass first

    std::vector<Real> nodal_mass(20);
    if (use_hrz) {
        elem.lumped_mass_hrz(coords.data(), density, nodal_mass.data());
    } else {
        // Uniform mass distribution
        Real element_mass = density * L * W * H;
        Real mass_per_node = element_mass / 20.0;
        for (int i = 0; i < 20; ++i) {
            nodal_mass[i] = mass_per_node;
        }
    }

    // Total mass
    Real total_mass = 0.0;
    for (int i = 0; i < 20; ++i) {
        total_mass += nodal_mass[i];
    }
    std::cout << "Total mass: " << total_mass << " kg" << std::endl;
    std::cout << "Expected mass: " << density * L * W * H << " kg" << std::endl;
    std::cout << "Nodal masses: ";
    for (int i = 0; i < 20; ++i) {
        std::cout << i << ":" << nodal_mass[i] << " ";
    }
    std::cout << std::endl;

    // DOF-level mass (each node has 3 DOFs with same mass)
    std::vector<Real> mass(60);
    for (int n = 0; n < 20; ++n) {
        for (int d = 0; d < 3; ++d) {
            mass[n*3 + d] = nodal_mass[n];
        }
    }

    // Fixed nodes (left face: x=0)
    std::vector<int> fixed_nodes = {0, 3, 4, 7, 11, 12, 15, 19};

    // Loaded nodes (right face: x=L)
    std::vector<int> loaded_nodes = {1, 2, 5, 6, 9, 13, 14, 17};
    const Real total_force = -1000.0;
    const Real force_per_node = total_force / loaded_nodes.size();

    // Initialize state vectors
    std::vector<Real> displacement(60, 0.0);
    std::vector<Real> velocity(60, 0.0);
    std::vector<Real> acceleration(60, 0.0);
    std::vector<Real> f_int(60, 0.0);
    std::vector<Real> f_ext(60, 0.0);

    // Apply external force
    for (int node : loaded_nodes) {
        f_ext[node*3 + 2] = force_per_node;
    }

    // Characteristic length and wave speed for time step
    const Real wave_speed = std::sqrt(E / density);
    const Real char_length = elem.characteristic_length(coords.data());
    const Real dt = 0.5 * char_length / wave_speed;

    std::cout << "Characteristic length: " << char_length << " m" << std::endl;
    std::cout << "Wave speed: " << wave_speed << " m/s" << std::endl;
    std::cout << "Time step: " << dt << " s" << std::endl;

    // Damping - use critical damping for faster convergence
    const Real damping = 0.5;  // 50% velocity reduction per step

    std::cout << "\n--- Simulation Steps ---\n" << std::endl;

    // Run a few steps
    for (int step = 0; step < 10; ++step) {
        // DEBUG: Print displacement BEFORE computing forces
        if (step < 3) {
            Real max_disp = 0;
            for (int i = 0; i < 60; ++i) {
                max_disp = std::max(max_disp, std::abs(displacement[i]));
            }
            std::cout << "BEFORE Step " << step << ": max|u| = " << max_disp << std::endl;
        }

        // 1. Compute internal forces (CPU fallback style)
        std::fill(f_int.begin(), f_int.end(), 0.0);

        // Also compute f_stiff = K*u for comparison
        std::vector<Real> K(60*60);
        elem.stiffness_matrix(coords.data(), E, nu, K.data());
        std::vector<Real> f_stiff(60, 0.0);
        for (int i = 0; i < 60; ++i) {
            for (int j = 0; j < 60; ++j) {
                f_stiff[i] += K[i*60+j] * displacement[j];
            }
        }

        // Get Gauss points
        std::vector<Real> gp_coords(27 * 3);
        std::vector<Real> gp_weights(27);
        elem.gauss_quadrature(gp_coords.data(), gp_weights.data());

        // Loop over Gauss points
        for (int gp = 0; gp < 27; ++gp) {
            Real xi[3] = {gp_coords[gp*3+0], gp_coords[gp*3+1], gp_coords[gp*3+2]};
            Real weight = gp_weights[gp];

            // B-matrix
            std::vector<Real> B(6 * 60);
            elem.strain_displacement_matrix(xi, coords.data(), B.data());

            // Jacobian
            Real J[9];
            Real detJ = elem.jacobian(xi, coords.data(), J);

            // Strain: ε = B * u
            std::vector<Real> strain(6, 0.0);
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 60; ++j) {
                    strain[i] += B[i*60 + j] * displacement[j];
                }
            }

            // Stress: σ = C * ε
            std::vector<Real> stress(6);
            stress[0] = (lambda + 2.0*mu) * strain[0] + lambda * (strain[1] + strain[2]);
            stress[1] = (lambda + 2.0*mu) * strain[1] + lambda * (strain[0] + strain[2]);
            stress[2] = (lambda + 2.0*mu) * strain[2] + lambda * (strain[0] + strain[1]);
            stress[3] = mu * strain[3];
            stress[4] = mu * strain[4];
            stress[5] = mu * strain[5];

            // Force: f += B^T * σ * detJ * weight
            Real factor = detJ * weight;
            for (int i = 0; i < 60; ++i) {
                for (int j = 0; j < 6; ++j) {
                    f_int[i] += B[j*60 + i] * stress[j] * factor;
                }
            }
        }

        // 2. Time integration
        for (int i = 0; i < 60; ++i) {
            // Check if this DOF is fixed
            int node = i / 3;
            bool is_fixed = false;
            for (int fn : fixed_nodes) {
                if (fn == node) {
                    is_fixed = true;
                    break;
                }
            }

            if (is_fixed) {
                displacement[i] = 0.0;
                velocity[i] = 0.0;
                acceleration[i] = 0.0;
            } else {
                // a = (f_ext - f_int) / m
                Real net_force = f_ext[i] - f_int[i];
                acceleration[i] = net_force / mass[i];

                // v += a*dt - damping*v
                velocity[i] += acceleration[i] * dt;
                velocity[i] *= (1.0 - damping);

                // u += v*dt
                displacement[i] += velocity[i] * dt;
            }
        }

        // Print diagnostics for node 1 (corner at right end)
        int node = 1;
        std::cout << "Step " << step << ":" << std::endl;

        // Print all z-displacements on first few steps
        if (step < 3) {
            std::cout << "  All uz: ";
            for (int n = 0; n < 20; ++n) {
                std::cout << displacement[n*3+2] << " ";
            }
            std::cout << std::endl;
            std::cout << "  All f_int_z: ";
            for (int n = 0; n < 20; ++n) {
                std::cout << f_int[n*3+2] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "  uz[1] = " << displacement[node*3+2] << std::endl;
        std::cout << "  vz[1] = " << velocity[node*3+2] << std::endl;
        std::cout << "  az[1] = " << acceleration[node*3+2] << std::endl;
        std::cout << "  f_int_z[1] = " << f_int[node*3+2] << std::endl;
        std::cout << "  f_ext_z[1] = " << f_ext[node*3+2] << std::endl;
        std::cout << "  net_force_z[1] = " << (f_ext[node*3+2] - f_int[node*3+2]) << std::endl;
        std::cout << "  mass[1] = " << mass[node*3+2] << std::endl;

        // Print total internal z-force on loaded and fixed faces
        Real f_int_z_loaded = 0, f_int_z_fixed = 0;
        for (int n : loaded_nodes) f_int_z_loaded += f_int[n*3+2];
        for (int n : fixed_nodes) f_int_z_fixed += f_int[n*3+2];
        std::cout << "  Total f_int_z on loaded face = " << f_int_z_loaded << std::endl;
        std::cout << "  Total f_int_z on fixed face = " << f_int_z_fixed << std::endl;

        // Check sign of sum(f_int_z * uz) - should be positive for stable behavior
        Real work = 0;
        for (int i = 0; i < 60; ++i) {
            work += f_int[i] * displacement[i];  // Full work, not just z
        }
        std::cout << "  Full work (f_int . u) = " << work << " (should be >= 0)" << std::endl;

        // Also check u^T K u using stiffness matrix (reuse K from earlier)
        Real uKu = 0;
        for (int i = 0; i < 60; ++i) {
            for (int j = 0; j < 60; ++j) {
                uKu += displacement[i] * K[i*60+j] * displacement[j];
            }
        }
        std::cout << "  Strain energy (u^T K u / 2) = " << 0.5 * uKu << " (should be >= 0)" << std::endl;

        // Compare f_int with K*u (reuse f_stiff from earlier)
        Real max_diff = 0;
        int max_idx = 0;
        for (int i = 0; i < 60; ++i) {
            Real diff = std::abs(f_int[i] - f_stiff[i]);
            if (diff > max_diff) {
                max_diff = diff;
                max_idx = i;
            }
        }
        std::cout << "  Max |f_int - K*u| = " << max_diff << " at DOF " << max_idx << std::endl;
        if (max_diff > 1e-10) {
            std::cout << "    f_int[" << max_idx << "] = " << f_int[max_idx] << std::endl;
            std::cout << "    (K*u)[" << max_idx << "] = " << f_stiff[max_idx] << std::endl;
            // Check work from K*u method
            Real work_ku = 0;
            for (int i = 0; i < 60; ++i) {
                work_ku += f_stiff[i] * displacement[i];
            }
            std::cout << "    Work from K*u = " << work_ku << std::endl;
        }
        std::cout << std::endl;

        // Check for divergence
        if (std::isnan(displacement[node*3+2]) || std::abs(displacement[node*3+2]) > 1.0) {
            std::cout << "DIVERGENCE DETECTED at step " << step << std::endl;
            return 1;
        }
    }

    std::cout << "Completed 10 steps without divergence" << std::endl;
    return 0;
}
