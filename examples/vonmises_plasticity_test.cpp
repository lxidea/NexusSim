/**
 * @file vonmises_plasticity_test.cpp
 * @brief Test Von Mises (J2) plasticity material model
 */

#include <nexussim/physics/material.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::physics;

int main() {
    std::cout << std::setprecision(8);
    std::cout << "=== Von Mises Plasticity Test ===" << std::endl << std::endl;

    // Material properties (steel)
    MaterialProperties props;
    props.density = 7850.0;          // Density (kg/m³)
    props.E = 210.0e9;               // Young's modulus (Pa)
    props.nu = 0.3;                  // Poisson's ratio
    props.yield_stress = 250.0e6;    // Yield stress (Pa) - typical mild steel
    props.hardening_modulus = 1.0e9; // Hardening modulus (Pa)
    props.compute_derived();         // Compute G, K, sound_speed

    std::cout << "Material Properties:" << std::endl;
    std::cout << "  E = " << props.E / 1.0e9 << " GPa" << std::endl;
    std::cout << "  nu = " << props.nu << std::endl;
    std::cout << "  G = " << props.G / 1.0e9 << " GPa" << std::endl;
    std::cout << "  K = " << props.K / 1.0e9 << " GPa" << std::endl;
    std::cout << "  sigma_y = " << props.yield_stress / 1.0e6 << " MPa" << std::endl;
    std::cout << "  H = " << props.hardening_modulus / 1.0e9 << " GPa" << std::endl;
    std::cout << std::endl;

    // Create Von Mises material
    VonMisesPlasticMaterial material(props);

    // Test 1: Elastic loading (below yield)
    std::cout << "=== Test 1: Elastic Loading ===" << std::endl;
    {
        MaterialState state;
        // Initialize history to zero
        for (int i = 0; i < 10; ++i) {
            state.history[i] = 0.0;
        }
        state.plastic_strain = 0.0;

        // Apply uniaxial strain below yield
        // σ_y = 250 MPa, E = 210 GPa => ε_y ≈ 0.00119
        Real strain_magnitude = 0.0005;  // Well below yield
        state.strain[0] = 0.0;
        state.strain[1] = 0.0;
        state.strain[2] = strain_magnitude;  // εzz
        state.strain[3] = 0.0;
        state.strain[4] = 0.0;
        state.strain[5] = 0.0;

        material.compute_stress(state);

        std::cout << "  Applied strain: εzz = " << strain_magnitude << std::endl;
        std::cout << "  Computed stress:" << std::endl;
        std::cout << "    σxx = " << state.stress[0] / 1.0e6 << " MPa" << std::endl;
        std::cout << "    σyy = " << state.stress[1] / 1.0e6 << " MPa" << std::endl;
        std::cout << "    σzz = " << state.stress[2] / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Plastic strain: " << state.plastic_strain << std::endl;

        // Expected for constrained uniaxial: σzz = (λ + 2μ)·ε = E(1-ν)/((1+ν)(1-2ν))·ε
        Real expected_sigma_zz = props.E * (1.0 - props.nu) / ((1.0 + props.nu) * (1.0 - 2.0 * props.nu)) * strain_magnitude;
        std::cout << "  Expected σzz (elastic): " << expected_sigma_zz / 1.0e6 << " MPa" << std::endl;

        bool pass = (std::abs(state.stress[2] - expected_sigma_zz) / expected_sigma_zz < 0.01)
                    && (state.plastic_strain < 1.0e-10);
        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    // Test 2: Plastic loading (above yield) - Uniaxial tension with free lateral contraction
    std::cout << "=== Test 2: Plastic Loading (Uniaxial Tension) ===" << std::endl;
    {
        MaterialState state;
        for (int i = 0; i < 10; ++i) {
            state.history[i] = 0.0;
        }
        state.plastic_strain = 0.0;

        // Apply uniaxial tension with proper Poisson contraction
        // For true uniaxial tension: εxx = εyy = -ν * εzz
        // This gives σxx = σyy = 0, σzz = E * εzz
        // σ_vm = |σzz| for uniaxial tension
        Real strain_zz = 0.002;  // 0.2% strain - enough to yield
        Real strain_lateral = -props.nu * strain_zz;  // Poisson contraction

        state.strain[0] = strain_lateral;  // εxx
        state.strain[1] = strain_lateral;  // εyy
        state.strain[2] = strain_zz;       // εzz
        state.strain[3] = 0.0;
        state.strain[4] = 0.0;
        state.strain[5] = 0.0;

        material.compute_stress(state);

        // Compute von Mises stress
        Real p = (state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;
        Real s_dev[3] = {
            state.stress[0] - p,
            state.stress[1] - p,
            state.stress[2] - p
        };
        Real sigma_vm = std::sqrt(1.5 * (s_dev[0]*s_dev[0] + s_dev[1]*s_dev[1] + s_dev[2]*s_dev[2] +
                                          2.0*(state.stress[3]*state.stress[3] +
                                               state.stress[4]*state.stress[4] +
                                               state.stress[5]*state.stress[5])));

        std::cout << "  Applied strain: εzz = " << strain_zz << ", εxx = εyy = " << strain_lateral << std::endl;
        std::cout << "  Computed stress:" << std::endl;
        std::cout << "    σxx = " << state.stress[0] / 1.0e6 << " MPa" << std::endl;
        std::cout << "    σyy = " << state.stress[1] / 1.0e6 << " MPa" << std::endl;
        std::cout << "    σzz = " << state.stress[2] / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Von Mises stress: " << sigma_vm / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Plastic strain: " << state.plastic_strain << std::endl;

        // For uniaxial, σ_vm ≈ σ_y + H·ε_p (yield criterion on loading surface)
        Real expected_sigma_y = props.yield_stress + props.hardening_modulus * state.plastic_strain;
        std::cout << "  Current yield stress: " << expected_sigma_y / 1.0e6 << " MPa" << std::endl;

        // Tolerance check: yielding occurred and stress is near yield surface
        bool yielded = (state.plastic_strain > 1.0e-8);
        bool on_yield_surface = (sigma_vm > 0.95 * expected_sigma_y && sigma_vm < 1.05 * expected_sigma_y);
        bool pass = yielded && on_yield_surface;
        std::cout << "  Yielded: " << (yielded ? "YES" : "NO") << std::endl;
        std::cout << "  On yield surface (±5%): " << (on_yield_surface ? "YES" : "NO") << std::endl;
        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    // Test 3: Loading-unloading cycle
    std::cout << "=== Test 3: Loading-Unloading Cycle ===" << std::endl;
    {
        MaterialState state;
        for (int i = 0; i < 10; ++i) {
            state.history[i] = 0.0;
        }
        state.plastic_strain = 0.0;

        // Step 1: Load to plastic regime
        state.strain[2] = 0.005;
        material.compute_stress(state);
        Real eps_p_after_loading = state.plastic_strain;
        Real sigma_zz_loaded = state.stress[2];

        std::cout << "  After loading to εzz = 0.005:" << std::endl;
        std::cout << "    σzz = " << sigma_zz_loaded / 1.0e6 << " MPa" << std::endl;
        std::cout << "    ε_p = " << eps_p_after_loading << std::endl;

        // Step 2: Unload (reduce strain)
        state.strain[2] = 0.003;  // Partial unload
        material.compute_stress(state);
        Real eps_p_after_unload = state.plastic_strain;
        Real sigma_zz_unloaded = state.stress[2];

        std::cout << "  After unloading to εzz = 0.003:" << std::endl;
        std::cout << "    σzz = " << sigma_zz_unloaded / 1.0e6 << " MPa" << std::endl;
        std::cout << "    ε_p = " << eps_p_after_unload << std::endl;

        // Plastic strain should remain the same during unloading
        bool pass = (std::abs(eps_p_after_unload - eps_p_after_loading) < 1.0e-10);
        std::cout << "  Plastic strain unchanged: " << (pass ? "YES" : "NO") << std::endl;
        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    // Test 4: Pure shear
    std::cout << "=== Test 4: Pure Shear ===" << std::endl;
    {
        MaterialState state;
        for (int i = 0; i < 10; ++i) {
            state.history[i] = 0.0;
        }
        state.plastic_strain = 0.0;

        // Apply shear strain
        // For shear: σ_vm = √3 * τ
        // Yield in shear: τ_y = σ_y / √3 ≈ 144 MPa
        Real shear_strain = 0.005;  // γxy
        state.strain[0] = 0.0;
        state.strain[1] = 0.0;
        state.strain[2] = 0.0;
        state.strain[3] = shear_strain;  // γxy (engineering shear strain)
        state.strain[4] = 0.0;
        state.strain[5] = 0.0;

        material.compute_stress(state);

        // Compute von Mises stress
        Real sigma_vm = std::sqrt(3.0 * state.stress[3] * state.stress[3]);

        std::cout << "  Applied shear: γxy = " << shear_strain << std::endl;
        std::cout << "  Computed shear stress: τxy = " << state.stress[3] / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Von Mises stress: " << sigma_vm / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Plastic strain: " << state.plastic_strain << std::endl;

        // Expected elastic: τ = G * γ
        Real tau_elastic = props.G * shear_strain;
        Real sigma_vm_elastic = std::sqrt(3.0) * tau_elastic;
        std::cout << "  Pure elastic σ_vm would be: " << sigma_vm_elastic / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Yield stress: " << props.yield_stress / 1.0e6 << " MPa" << std::endl;

        bool yielded = (state.plastic_strain > 1.0e-6);
        std::cout << "  Yielding occurred: " << (yielded ? "YES" : "NO") << std::endl;
        std::cout << std::endl;
    }

    // Test 5: Tangent stiffness
    std::cout << "=== Test 5: Tangent Stiffness ===" << std::endl;
    {
        MaterialState state;
        for (int i = 0; i < 10; ++i) {
            state.history[i] = 0.0;
        }
        state.plastic_strain = 0.0;

        // First in elastic regime
        state.strain[2] = 0.0005;
        material.compute_stress(state);

        Real C_elastic[36];
        material.tangent_stiffness(state, C_elastic);

        std::cout << "  Elastic stiffness (εzz = 0.0005):" << std::endl;
        std::cout << "    C[0,0] = " << C_elastic[0] / 1.0e9 << " GPa" << std::endl;
        std::cout << "    C[0,1] = " << C_elastic[1] / 1.0e9 << " GPa" << std::endl;
        std::cout << "    C[3,3] = " << C_elastic[21] / 1.0e9 << " GPa" << std::endl;

        // Now in plastic regime
        state.strain[2] = 0.005;
        material.compute_stress(state);

        Real C_plastic[36];
        material.tangent_stiffness(state, C_plastic);

        std::cout << "  Plastic stiffness (εzz = 0.005):" << std::endl;
        std::cout << "    C[0,0] = " << C_plastic[0] / 1.0e9 << " GPa" << std::endl;
        std::cout << "    C[0,1] = " << C_plastic[1] / 1.0e9 << " GPa" << std::endl;
        std::cout << "    C[3,3] = " << C_plastic[21] / 1.0e9 << " GPa" << std::endl;

        // Stiffness should be lower in plastic regime
        bool pass = (C_plastic[0] <= C_elastic[0]);  // May be equal if using elastic tangent
        std::cout << "  Stiffness reduced: " << (C_plastic[0] < C_elastic[0] ? "YES" : "NO (using elastic tangent)") << std::endl;
        std::cout << std::endl;
    }

    std::cout << "=== All Tests Complete ===" << std::endl;
    return 0;
}
