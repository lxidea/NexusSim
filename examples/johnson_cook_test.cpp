/**
 * @file johnson_cook_test.cpp
 * @brief Test Johnson-Cook plasticity material model
 *
 * Tests strain hardening, strain rate effects, and thermal softening.
 */

#include <nexussim/physics/material.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::physics;

int main() {
    std::cout << std::setprecision(8);
    std::cout << "=== Johnson-Cook Plasticity Test ===" << std::endl << std::endl;

    // Material properties for OFHC Copper (typical JC parameters)
    // Reference: Johnson & Cook, 1983
    MaterialProperties props;
    props.density = 8960.0;          // Density (kg/m³)
    props.E = 124.0e9;               // Young's modulus (Pa)
    props.nu = 0.34;                 // Poisson's ratio

    // Johnson-Cook parameters for OFHC Copper
    props.JC_A = 90.0e6;             // A = 90 MPa
    props.JC_B = 292.0e6;            // B = 292 MPa
    props.JC_n = 0.31;               // n = 0.31
    props.JC_C = 0.025;              // C = 0.025
    props.JC_m = 1.09;               // m = 1.09
    props.JC_eps_dot_ref = 1.0;      // Reference strain rate (1/s)
    props.JC_T_melt = 1356.0;        // Melting temperature (K)
    props.JC_T_room = 293.0;         // Room temperature (K)

    props.compute_derived();

    std::cout << "Material Properties (OFHC Copper):" << std::endl;
    std::cout << "  E = " << props.E / 1.0e9 << " GPa" << std::endl;
    std::cout << "  nu = " << props.nu << std::endl;
    std::cout << "  G = " << props.G / 1.0e9 << " GPa" << std::endl;
    std::cout << std::endl;
    std::cout << "Johnson-Cook Parameters:" << std::endl;
    std::cout << "  A = " << props.JC_A / 1.0e6 << " MPa" << std::endl;
    std::cout << "  B = " << props.JC_B / 1.0e6 << " MPa" << std::endl;
    std::cout << "  n = " << props.JC_n << std::endl;
    std::cout << "  C = " << props.JC_C << std::endl;
    std::cout << "  m = " << props.JC_m << std::endl;
    std::cout << "  T_room = " << props.JC_T_room << " K" << std::endl;
    std::cout << "  T_melt = " << props.JC_T_melt << " K" << std::endl;
    std::cout << std::endl;

    // Create Johnson-Cook material
    JohnsonCookMaterial material(props);

    // Test 1: Quasi-static loading at room temperature
    std::cout << "=== Test 1: Quasi-Static Loading (Room Temperature) ===" << std::endl;
    {
        MaterialState state;
        state.temperature = props.JC_T_room;
        state.effective_strain_rate = 1.0;  // Reference strain rate

        // Apply strain
        Real strain_zz = 0.002;
        Real strain_lateral = -props.nu * strain_zz;
        state.strain[0] = strain_lateral;
        state.strain[1] = strain_lateral;
        state.strain[2] = strain_zz;

        material.compute_stress(state);

        // Compute von Mises stress
        Real p = (state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;
        Real s[3] = {state.stress[0] - p, state.stress[1] - p, state.stress[2] - p};
        Real sigma_vm = std::sqrt(1.5 * (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]));

        // Expected yield stress at this plastic strain
        Real eps_p = state.plastic_strain;
        Real expected_sigma_y = props.JC_A;
        if (eps_p > 1.0e-10) {
            expected_sigma_y += props.JC_B * std::pow(eps_p, props.JC_n);
        }

        std::cout << "  Applied strain: εzz = " << strain_zz << std::endl;
        std::cout << "  Von Mises stress: " << sigma_vm / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Plastic strain: " << eps_p << std::endl;
        std::cout << "  Expected σ_y (JC): " << expected_sigma_y / 1.0e6 << " MPa" << std::endl;

        bool yielded = (eps_p > 1.0e-8);
        bool pass = yielded && (std::abs(sigma_vm - expected_sigma_y) / expected_sigma_y < 0.05);
        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    // Test 2: Strain rate effect
    std::cout << "=== Test 2: Strain Rate Effect ===" << std::endl;
    {
        // Low strain rate
        MaterialState state_low;
        state_low.temperature = props.JC_T_room;
        state_low.effective_strain_rate = 1.0;  // 1/s

        Real strain_zz = 0.01;
        Real strain_lateral = -props.nu * strain_zz;
        state_low.strain[0] = strain_lateral;
        state_low.strain[1] = strain_lateral;
        state_low.strain[2] = strain_zz;

        material.compute_stress(state_low);

        Real p_low = (state_low.stress[0] + state_low.stress[1] + state_low.stress[2]) / 3.0;
        Real s_low[3] = {state_low.stress[0] - p_low, state_low.stress[1] - p_low, state_low.stress[2] - p_low};
        Real sigma_vm_low = std::sqrt(1.5 * (s_low[0]*s_low[0] + s_low[1]*s_low[1] + s_low[2]*s_low[2]));

        // High strain rate
        MaterialState state_high;
        state_high.temperature = props.JC_T_room;
        state_high.effective_strain_rate = 1000.0;  // 1000/s

        state_high.strain[0] = strain_lateral;
        state_high.strain[1] = strain_lateral;
        state_high.strain[2] = strain_zz;

        material.compute_stress(state_high);

        Real p_high = (state_high.stress[0] + state_high.stress[1] + state_high.stress[2]) / 3.0;
        Real s_high[3] = {state_high.stress[0] - p_high, state_high.stress[1] - p_high, state_high.stress[2] - p_high};
        Real sigma_vm_high = std::sqrt(1.5 * (s_high[0]*s_high[0] + s_high[1]*s_high[1] + s_high[2]*s_high[2]));

        // Expected rate factor: (1 + C * ln(1000)) = 1 + 0.025 * 6.908 ≈ 1.173
        Real expected_rate_factor = 1.0 + props.JC_C * std::log(1000.0);

        std::cout << "  Low strain rate (1/s):" << std::endl;
        std::cout << "    σ_vm = " << sigma_vm_low / 1.0e6 << " MPa" << std::endl;
        std::cout << "  High strain rate (1000/s):" << std::endl;
        std::cout << "    σ_vm = " << sigma_vm_high / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Ratio (high/low): " << sigma_vm_high / sigma_vm_low << std::endl;
        std::cout << "  Expected ratio: " << expected_rate_factor << std::endl;

        // Note: The actual ratio won't exactly match because plastic strains differ
        bool pass = (sigma_vm_high > sigma_vm_low);
        std::cout << "  High rate gives higher stress: " << (pass ? "YES" : "NO") << std::endl;
        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    // Test 3: Thermal softening effect
    std::cout << "=== Test 3: Thermal Softening ===" << std::endl;
    {
        Real strain_zz = 0.01;
        Real strain_lateral = -props.nu * strain_zz;

        // Room temperature
        MaterialState state_cold;
        state_cold.temperature = props.JC_T_room;
        state_cold.effective_strain_rate = 1.0;
        state_cold.strain[0] = strain_lateral;
        state_cold.strain[1] = strain_lateral;
        state_cold.strain[2] = strain_zz;
        material.compute_stress(state_cold);

        Real p_cold = (state_cold.stress[0] + state_cold.stress[1] + state_cold.stress[2]) / 3.0;
        Real s_cold[3] = {state_cold.stress[0] - p_cold, state_cold.stress[1] - p_cold, state_cold.stress[2] - p_cold};
        Real sigma_vm_cold = std::sqrt(1.5 * (s_cold[0]*s_cold[0] + s_cold[1]*s_cold[1] + s_cold[2]*s_cold[2]));

        // Elevated temperature (half way to melt)
        MaterialState state_hot;
        Real T_hot = (props.JC_T_room + props.JC_T_melt) / 2.0;
        state_hot.temperature = T_hot;
        state_hot.effective_strain_rate = 1.0;
        state_hot.strain[0] = strain_lateral;
        state_hot.strain[1] = strain_lateral;
        state_hot.strain[2] = strain_zz;
        material.compute_stress(state_hot);

        Real p_hot = (state_hot.stress[0] + state_hot.stress[1] + state_hot.stress[2]) / 3.0;
        Real s_hot[3] = {state_hot.stress[0] - p_hot, state_hot.stress[1] - p_hot, state_hot.stress[2] - p_hot};
        Real sigma_vm_hot = std::sqrt(1.5 * (s_hot[0]*s_hot[0] + s_hot[1]*s_hot[1] + s_hot[2]*s_hot[2]));

        // Expected thermal softening: (1 - T*^m) where T* = 0.5
        Real T_star = 0.5;
        Real expected_thermal_factor = 1.0 - std::pow(T_star, props.JC_m);

        std::cout << "  Room temperature (" << props.JC_T_room << " K):" << std::endl;
        std::cout << "    σ_vm = " << sigma_vm_cold / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Elevated temperature (" << T_hot << " K):" << std::endl;
        std::cout << "    σ_vm = " << sigma_vm_hot / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Ratio (hot/cold): " << sigma_vm_hot / sigma_vm_cold << std::endl;
        std::cout << "  Expected thermal factor: " << expected_thermal_factor << std::endl;

        bool pass = (sigma_vm_hot < sigma_vm_cold);
        std::cout << "  Higher temperature gives lower stress: " << (pass ? "YES" : "NO") << std::endl;
        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    // Test 4: Strain hardening curve
    std::cout << "=== Test 4: Strain Hardening Curve ===" << std::endl;
    {
        std::cout << "  ε_p\t\tσ_y (MPa)\tExpected (MPa)" << std::endl;
        std::cout << "  ----\t\t--------\t-------------" << std::endl;

        MaterialState state;
        state.temperature = props.JC_T_room;
        state.effective_strain_rate = 1.0;

        Real plastic_strains[] = {0.0, 0.01, 0.05, 0.10, 0.20, 0.50};
        bool all_pass = true;

        for (Real target_eps_p : plastic_strains) {
            // Reset state
            for (int i = 0; i < 10; ++i) state.history[i] = 0.0;
            state.plastic_strain = 0.0;

            // Apply enough strain to reach target plastic strain
            // Use iterative approach
            Real strain_zz = 0.001 + target_eps_p * 1.5;  // Approximate
            Real strain_lateral = -props.nu * strain_zz;
            state.strain[0] = strain_lateral;
            state.strain[1] = strain_lateral;
            state.strain[2] = strain_zz;

            material.compute_stress(state);

            Real eps_p = state.plastic_strain;
            Real p = (state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;
            Real s[3] = {state.stress[0] - p, state.stress[1] - p, state.stress[2] - p};
            Real sigma_vm = std::sqrt(1.5 * (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]));

            // Expected from JC model
            Real expected = props.JC_A;
            if (eps_p > 1.0e-10) {
                expected += props.JC_B * std::pow(eps_p, props.JC_n);
            }

            std::cout << "  " << eps_p << "\t\t" << sigma_vm / 1.0e6 << "\t\t" << expected / 1.0e6 << std::endl;

            if (eps_p > 1.0e-6 && std::abs(sigma_vm - expected) / expected > 0.1) {
                all_pass = false;
            }
        }

        std::cout << std::endl;
        std::cout << "  Result: " << (all_pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    // Test 5: Near-melt behavior
    std::cout << "=== Test 5: Near-Melt Behavior ===" << std::endl;
    {
        Real strain_zz = 0.01;
        Real strain_lateral = -props.nu * strain_zz;

        // Just below melt
        MaterialState state_near_melt;
        state_near_melt.temperature = props.JC_T_melt - 10.0;
        state_near_melt.effective_strain_rate = 1.0;
        state_near_melt.strain[0] = strain_lateral;
        state_near_melt.strain[1] = strain_lateral;
        state_near_melt.strain[2] = strain_zz;
        material.compute_stress(state_near_melt);

        Real p1 = (state_near_melt.stress[0] + state_near_melt.stress[1] + state_near_melt.stress[2]) / 3.0;
        Real s1[3] = {state_near_melt.stress[0] - p1, state_near_melt.stress[1] - p1, state_near_melt.stress[2] - p1};
        Real sigma_vm_near = std::sqrt(1.5 * (s1[0]*s1[0] + s1[1]*s1[1] + s1[2]*s1[2]));

        // At melt
        MaterialState state_melt;
        state_melt.temperature = props.JC_T_melt;
        state_melt.effective_strain_rate = 1.0;
        state_melt.strain[0] = strain_lateral;
        state_melt.strain[1] = strain_lateral;
        state_melt.strain[2] = strain_zz;
        material.compute_stress(state_melt);

        Real p2 = (state_melt.stress[0] + state_melt.stress[1] + state_melt.stress[2]) / 3.0;
        Real s2[3] = {state_melt.stress[0] - p2, state_melt.stress[1] - p2, state_melt.stress[2] - p2};
        Real sigma_vm_melt = std::sqrt(1.5 * (s2[0]*s2[0] + s2[1]*s2[1] + s2[2]*s2[2]));

        std::cout << "  Near melt (" << props.JC_T_melt - 10.0 << " K): σ_vm = " << sigma_vm_near / 1.0e6 << " MPa" << std::endl;
        std::cout << "  At melt (" << props.JC_T_melt << " K): σ_vm = " << sigma_vm_melt / 1.0e6 << " MPa" << std::endl;

        // At melt, strength should be very low or zero
        bool pass = (sigma_vm_near > sigma_vm_melt) && (sigma_vm_near > 0.1e6);
        std::cout << "  Material weakens significantly at melt: " << (pass ? "YES" : "NO") << std::endl;
        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    std::cout << "=== All Johnson-Cook Tests Complete ===" << std::endl;
    return 0;
}
