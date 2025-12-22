/**
 * @file hex20_single_step_debug.cpp
 * @brief Debug single time step: manually compute one explicit dynamics step
 *
 * This test manually performs one time step to see what values are computed:
 * 1. Compute lumped mass matrix M
 * 2. Apply initial displacement u₀
 * 3. Compute internal force f_int = K*u
 * 4. Compute acceleration a = M⁻¹ * (f_ext - f_int)
 * 5. Update velocity v₁ = v₀ + a*dt
 * 6. Update displacement u₁ = u₀ + v₁*dt
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
void test_single_step(const char* element_name, int n_nodes) {
    std::cout << "\n========================================" << std::endl;
    std::cout << element_name << " Single Time Step Test" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Unit cube
    const Real L = 1.0;
    Real coords[60];  // Max 20 nodes × 3

    // Corner nodes (same for Hex8 and Hex20)
    coords[0*3+0] = 0.0; coords[0*3+1] = 0.0; coords[0*3+2] = 0.0;
    coords[1*3+0] = L;   coords[1*3+1] = 0.0; coords[1*3+2] = 0.0;
    coords[2*3+0] = L;   coords[2*3+1] = L;   coords[2*3+2] = 0.0;
    coords[3*3+0] = 0.0; coords[3*3+1] = L;   coords[3*3+2] = 0.0;
    coords[4*3+0] = 0.0; coords[4*3+1] = 0.0; coords[4*3+2] = L;
    coords[5*3+0] = L;   coords[5*3+1] = 0.0; coords[5*3+2] = L;
    coords[6*3+0] = L;   coords[6*3+1] = L;   coords[6*3+2] = L;
    coords[7*3+0] = 0.0; coords[7*3+1] = L;   coords[7*3+2] = L;

    if (n_nodes == 20) {
        // Mid-edge nodes for Hex20
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
    }

    const Real E = 210e9;
    const Real nu = 0.3;
    const Real density = 7850.0;

    ElementType elem;

    // 1. Compute lumped mass
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
        // Hex8 uses lumped_mass_matrix which returns 8x3 matrix
        Real M_mat[24];  // 8 nodes × 3 DOF
        elem.lumped_mass_matrix(coords, density, M_mat);
        for (int i = 0; i < n_nodes * 3; ++i) {
            M_lumped[i] = M_mat[i];
        }
    }

    std::cout << "Lumped mass diagonal (first 4 nodes, z-component):" << std::endl;
    for (int i = 0; i < std::min(4, n_nodes); ++i) {
        std::cout << "  M[" << i << "] = " << M_lumped[i*3 + 2] << std::endl;
    }
    std::cout << std::endl;

    // 2. Compute stiffness
    Real K[60*60];
    elem.stiffness_matrix(coords, E, nu, K);

    // 3. Apply initial displacement (small displacement at top nodes)
    std::vector<Real> u0(n_nodes * 3, 0.0);
    const Real u_init = 1e-6;  // 1 micron

    // Apply to top corner node 6 only
    u0[6*3 + 2] = u_init;

    std::cout << "Initial displacement: u0[node 6, z] = " << u_init << " m" << std::endl;

    // 4. Compute internal force f_int = K * u0
    std::vector<Real> f_int(n_nodes * 3, 0.0);
    for (int i = 0; i < n_nodes * 3; ++i) {
        for (int j = 0; j < n_nodes * 3; ++j) {
            f_int[i] += K[i * n_nodes * 3 + j] * u0[j];
        }
    }

    std::cout << "Internal force at node 6, z: f_int = " << f_int[6*3 + 2] << " N" << std::endl;

    // 5. External force (zero for this test)
    std::vector<Real> f_ext(n_nodes * 3, 0.0);

    // 6. Compute acceleration a = M⁻¹ * (f_ext - f_int)
    std::vector<Real> a(n_nodes * 3, 0.0);
    for (int i = 0; i < n_nodes * 3; ++i) {
        if (M_lumped[i] > 1e-15) {
            a[i] = (f_ext[i] - f_int[i]) / M_lumped[i];
        }
    }

    std::cout << "Acceleration at node 6, z: a = " << a[6*3 + 2] << " m/s²" << std::endl;

    // 7. Compute velocity v1 = v0 + a*dt (assume v0 = 0)
    const Real dt = 1e-6;  // 1 microsecond
    std::vector<Real> v1(n_nodes * 3, 0.0);
    for (int i = 0; i < n_nodes * 3; ++i) {
        v1[i] = a[i] * dt;
    }

    std::cout << "Velocity after dt=" << dt << " s: v1[node 6, z] = " << v1[6*3 + 2] << " m/s" << std::endl;

    // 8. Compute displacement u1 = u0 + v1*dt
    std::vector<Real> u1(n_nodes * 3, 0.0);
    for (int i = 0; i < n_nodes * 3; ++i) {
        u1[i] = u0[i] + v1[i] * dt;
    }

    std::cout << "Displacement after dt: u1[node 6, z] = " << u1[6*3 + 2] << " m" << std::endl;

    // Check direction
    std::cout << "\n--- Analysis ---" << std::endl;
    std::cout << "Initial displacement: uz = " << u0[6*3 + 2] << " (+z)" << std::endl;
    std::cout << "Internal force:       fz = " << f_int[6*3 + 2];
    if (f_int[6*3 + 2] > 0) {
        std::cout << " (+z, SAME direction as displacement!)" << std::endl;
    } else {
        std::cout << " (-z, opposite to displacement)" << std::endl;
    }

    std::cout << "Acceleration:         az = " << a[6*3 + 2];
    if (a[6*3 + 2] < 0) {
        std::cout << " (-z, restoring force)" << std::endl;
    } else {
        std::cout << " (+z, NOT restoring!)" << std::endl;
    }

    std::cout << "Change in displacement: Δu = " << (u1[6*3 + 2] - u0[6*3 + 2]);
    if ((u1[6*3 + 2] - u0[6*3 + 2]) < 0) {
        std::cout << " (reducing displacement - STABLE)" << std::endl;
    } else {
        std::cout << " (increasing displacement - UNSTABLE!)" << std::endl;
    }

    // Stability check
    bool is_stable = (a[6*3 + 2] < 0);  // Acceleration should oppose displacement
    std::cout << "\n";
    if (is_stable) {
        std::cout << "✓ STABLE: Acceleration opposes displacement (restoring force)" << std::endl;
    } else {
        std::cout << "✗ UNSTABLE: Acceleration AMPLIFIES displacement!" << std::endl;
        std::cout << "  This will cause exponential growth in dynamics!" << std::endl;
    }
}

int main() {
    std::cout << std::setprecision(10);

    std::cout << "====================================================" << std::endl;
    std::cout << "Single Time Step Comparison: Hex8 vs Hex20" << std::endl;
    std::cout << "====================================================" << std::endl;
    std::cout << "Test: Apply small +z displacement at node 6," << std::endl;
    std::cout << "      then compute one explicit dynamics time step." << std::endl;
    std::cout << "Expected: Acceleration should OPPOSE displacement (negative)" << std::endl;

    test_single_step<Hex8Element>("Hex8", 8);
    test_single_step<Hex20Element>("Hex20", 20);

    return 0;
}
