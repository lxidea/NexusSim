/**
 * @file neohookean_test.cpp
 * @brief Test Neo-Hookean hyperelastic material model
 *
 * Tests uniaxial tension, biaxial tension, shear, and compression.
 */

#include <nexussim/physics/material.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::physics;

int main() {
    std::cout << std::setprecision(8);
    std::cout << "=== Neo-Hookean Hyperelastic Test ===" << std::endl << std::endl;

    // Material properties for rubber-like material
    MaterialProperties props;
    props.density = 1100.0;          // Density (kg/m³)
    props.E = 1.0e6;                 // Young's modulus (Pa) - typical rubber
    props.nu = 0.49;                 // Poisson's ratio (nearly incompressible)
    props.compute_derived();

    std::cout << "Material Properties (Rubber-like):" << std::endl;
    std::cout << "  E = " << props.E / 1.0e6 << " MPa" << std::endl;
    std::cout << "  nu = " << props.nu << std::endl;
    std::cout << "  G (μ) = " << props.G / 1.0e6 << " MPa" << std::endl;
    std::cout << "  K (κ) = " << props.K / 1.0e6 << " MPa" << std::endl;
    std::cout << std::endl;

    // Create Neo-Hookean material
    NeoHookeanMaterial material(props);

    // Test 1: Small strain uniaxial tension (should match linear elastic for small strain)
    std::cout << "=== Test 1: Small Strain Uniaxial Tension ===" << std::endl;
    {
        MaterialState state;

        // Apply small uniaxial stretch with volumetric change (unconstrained lateral)
        Real lambda_z = 1.01;  // 1% extension
        Real lambda_lat = 1.0; // No lateral constraint

        // Set deformation gradient
        state.F[0] = lambda_lat;  // F_xx
        state.F[4] = lambda_lat;  // F_yy
        state.F[8] = lambda_z;    // F_zz

        material.compute_stress(state);

        std::cout << "  Applied stretch: λz = " << lambda_z << " (1% extension)" << std::endl;
        std::cout << "  Computed σzz = " << state.stress[2] / 1.0e6 << " MPa" << std::endl;
        std::cout << "  σxx = " << state.stress[0] / 1.0e6 << " MPa" << std::endl;
        std::cout << "  σyy = " << state.stress[1] / 1.0e6 << " MPa" << std::endl;

        // For near-incompressible: stress should be positive in z and approximately symmetric
        bool pass = (state.stress[2] > 0 && std::abs(state.stress[0] - state.stress[1]) < 1.0e-6);
        std::cout << "  Positive tension and symmetric: " << (pass ? "YES" : "NO") << std::endl;
        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    // Test 2: Large strain uniaxial tension
    std::cout << "=== Test 2: Large Strain Uniaxial Tension ===" << std::endl;
    {
        MaterialState state;

        // Apply 50% extension - volumetric (J > 1)
        Real lambda_z = 1.5;  // 50% extension
        Real lambda_lat = 1.0;  // No lateral compression (volumetric change)

        state.F[0] = lambda_lat;
        state.F[4] = lambda_lat;
        state.F[8] = lambda_z;

        material.compute_stress(state);

        // For volumetric stretch, check that σzz > σxx and stress increases with stretch
        std::cout << "  Applied stretch: λz = " << lambda_z << " (50% extension, J = " << lambda_z << ")" << std::endl;
        std::cout << "  Computed σzz = " << state.stress[2] / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Computed σxx = " << state.stress[0] / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Computed σyy = " << state.stress[1] / 1.0e6 << " MPa" << std::endl;

        // Basic check: tensile stress in z-direction, positive stress
        bool pass = (state.stress[2] > 0 && state.stress[2] > state.stress[0]);
        std::cout << "  σzz > 0 and σzz > σxx: " << (pass ? "YES" : "NO") << std::endl;
        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    // Test 3: Pure shear
    std::cout << "=== Test 3: Pure Shear ===" << std::endl;
    {
        MaterialState state;

        // Simple shear: F = [[1, γ, 0], [0, 1, 0], [0, 0, 1]]
        Real gamma = 0.1;  // Shear strain

        state.F[0] = 1.0;
        state.F[1] = gamma;  // F_xy = γ
        state.F[4] = 1.0;
        state.F[8] = 1.0;
        // Other terms are already 0

        material.compute_stress(state);

        // For Neo-Hookean simple shear: τxy = μ * γ
        Real tau_analytical = props.G * gamma;

        std::cout << "  Applied shear: γ = " << gamma << std::endl;
        std::cout << "  Computed τxy = " << state.stress[3] / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Analytical: " << tau_analytical / 1.0e6 << " MPa" << std::endl;

        bool pass = (std::abs(state.stress[3] - tau_analytical) / tau_analytical < 0.05);
        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    // Test 4: Hydrostatic compression
    std::cout << "=== Test 4: Hydrostatic Compression ===" << std::endl;
    {
        MaterialState state;

        // Uniform compression: λ = 0.95 in all directions
        Real lambda = 0.95;

        state.F[0] = lambda;
        state.F[4] = lambda;
        state.F[8] = lambda;

        material.compute_stress(state);

        // Volume change
        Real J = lambda * lambda * lambda;

        // For Neo-Hookean: p = κ * (J - 1)
        Real p_analytical = props.K * (J - 1.0);

        // Hydrostatic stress: σii should all equal p (approximately for this case)
        Real p_computed = (state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;

        std::cout << "  Applied compression: λ = " << lambda << " in all directions" << std::endl;
        std::cout << "  Volume ratio J = " << J << std::endl;
        std::cout << "  Computed pressure = " << p_computed / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Analytical pressure: " << p_analytical / 1.0e6 << " MPa" << std::endl;

        bool pass = (std::abs(p_computed - p_analytical) / std::abs(p_analytical) < 0.1);
        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    // Test 5: Stress-strain curve for uniaxial tension (monotonically increasing)
    std::cout << "=== Test 5: Uniaxial Stress-Strain Curve ===" << std::endl;
    {
        std::cout << "  λ\t\tJ\t\tσzz (MPa)" << std::endl;
        std::cout << "  ---\t\t---\t\t---------" << std::endl;

        Real stretches[] = {1.0, 1.1, 1.2, 1.5, 2.0, 2.5};
        bool all_pass = true;
        Real prev_stress = -1.0e10;

        for (Real lambda_z : stretches) {
            MaterialState state;
            // Volumetric stretch (no lateral constraint)
            state.F[0] = 1.0;
            state.F[4] = 1.0;
            state.F[8] = lambda_z;

            material.compute_stress(state);

            std::cout << "  " << lambda_z << "\t\t" << lambda_z
                      << "\t\t" << state.stress[2] / 1.0e6 << std::endl;

            // Check monotonically increasing stress with stretch
            if (lambda_z > 1.0 && state.stress[2] <= prev_stress) {
                all_pass = false;
            }
            prev_stress = state.stress[2];
        }

        std::cout << std::endl;
        std::cout << "  Stress monotonically increasing: " << (all_pass ? "YES" : "NO") << std::endl;
        std::cout << "  Result: " << (all_pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    // Test 6: Biaxial tension
    std::cout << "=== Test 6: Biaxial Tension ===" << std::endl;
    {
        MaterialState state;

        // Biaxial stretch with volumetric change
        Real lambda_xy = 1.2;
        Real lambda_z = 1.0;  // No z constraint

        state.F[0] = lambda_xy;
        state.F[4] = lambda_xy;
        state.F[8] = lambda_z;

        material.compute_stress(state);

        std::cout << "  Applied stretch: λx = λy = " << lambda_xy << ", λz = " << lambda_z << std::endl;
        std::cout << "  J = " << lambda_xy * lambda_xy * lambda_z << std::endl;
        std::cout << "  Computed σxx = " << state.stress[0] / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Computed σyy = " << state.stress[1] / 1.0e6 << " MPa" << std::endl;
        std::cout << "  Computed σzz = " << state.stress[2] / 1.0e6 << " MPa" << std::endl;

        // Check: σxx = σyy (symmetric) and positive in tension
        bool pass = (std::abs(state.stress[0] - state.stress[1]) < 1.0e-6 && state.stress[0] > 0);
        std::cout << "  Symmetric biaxial stress (σxx = σyy): " << (pass ? "YES" : "NO") << std::endl;
        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    // Test 7: Identity deformation (no stress)
    std::cout << "=== Test 7: Identity Deformation (Zero Stress) ===" << std::endl;
    {
        MaterialState state;

        // Identity F
        state.F[0] = 1.0;
        state.F[4] = 1.0;
        state.F[8] = 1.0;

        material.compute_stress(state);

        Real max_stress = 0.0;
        for (int i = 0; i < 6; ++i) {
            max_stress = std::max(max_stress, std::abs(state.stress[i]));
        }

        std::cout << "  F = Identity" << std::endl;
        std::cout << "  Max stress component: " << max_stress << " Pa" << std::endl;

        bool pass = (max_stress < 1.0e-6);  // Should be essentially zero
        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    std::cout << "=== All Neo-Hookean Tests Complete ===" << std::endl;
    return 0;
}
