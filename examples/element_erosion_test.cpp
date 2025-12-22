/**
 * @file element_erosion_test.cpp
 * @brief Test suite for element erosion and failure models
 *
 * Tests:
 * 1. Principal stress computation
 * 2. Stress triaxiality calculation
 * 3. Maximum principal stress failure
 * 4. Plastic strain failure
 * 5. Johnson-Cook damage accumulation
 * 6. Cockcroft-Latham criterion
 * 7. Element erosion manager
 * 8. Mass redistribution
 */

#include <nexussim/physics/element_erosion.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::physics;

constexpr Real TOL = 1.0e-6;

int tests_passed = 0;
int tests_total = 0;

void check(bool condition, const std::string& test_name) {
    tests_total++;
    if (condition) {
        tests_passed++;
        std::cout << "  [PASS] " << test_name << "\n";
    } else {
        std::cout << "  [FAIL] " << test_name << "\n";
    }
}

// Test 1: Principal stress computation
void test_principal_stresses() {
    std::cout << "\n=== Test 1: Principal Stress Computation ===\n";

    // Hydrostatic stress: σ = p*I
    Real sigma_hydro[6] = {100e6, 100e6, 100e6, 0, 0, 0};
    Real principal_hydro[3];
    compute_principal_stresses(sigma_hydro, principal_hydro);

    check(std::abs(principal_hydro[0] - 100e6) < TOL * 100e6,
          "Hydrostatic σ1 = p");
    check(std::abs(principal_hydro[1] - 100e6) < TOL * 100e6,
          "Hydrostatic σ2 = p");
    check(std::abs(principal_hydro[2] - 100e6) < TOL * 100e6,
          "Hydrostatic σ3 = p");

    // Uniaxial tension
    Real sigma_uniax[6] = {200e6, 0, 0, 0, 0, 0};
    Real principal_uniax[3];
    compute_principal_stresses(sigma_uniax, principal_uniax);

    check(std::abs(principal_uniax[0] - 200e6) < TOL * 200e6,
          "Uniaxial σ1 = σ_xx");
    check(std::abs(principal_uniax[1]) < TOL,
          "Uniaxial σ2 = 0");
    check(std::abs(principal_uniax[2]) < TOL,
          "Uniaxial σ3 = 0");

    // Pure shear: σ_xy = τ
    Real tau = 50e6;
    Real sigma_shear[6] = {0, 0, 0, tau, 0, 0};
    Real principal_shear[3];
    compute_principal_stresses(sigma_shear, principal_shear);

    // Principal stresses for pure shear: ±τ, 0
    check(std::abs(principal_shear[0] - tau) < TOL * tau,
          "Pure shear σ1 = τ");
    check(std::abs(principal_shear[1]) < TOL * tau,
          "Pure shear σ2 = 0");
    check(std::abs(principal_shear[2] + tau) < TOL * tau,
          "Pure shear σ3 = -τ");

    std::cout << "  Principal (shear): " << principal_shear[0]/1e6 << ", "
              << principal_shear[1]/1e6 << ", " << principal_shear[2]/1e6 << " MPa\n";
}

// Test 2: Stress triaxiality
void test_triaxiality() {
    std::cout << "\n=== Test 2: Stress Triaxiality ===\n";

    // Uniaxial tension: η = 1/3
    Real sigma_uniax[6] = {200e6, 0, 0, 0, 0, 0};
    Real eta_uniax = compute_triaxiality(sigma_uniax);
    check(std::abs(eta_uniax - 1.0/3.0) < TOL,
          "Uniaxial tension η = 1/3");
    std::cout << "  η (uniaxial) = " << eta_uniax << "\n";

    // Pure shear: η = 0
    Real sigma_shear[6] = {0, 0, 0, 50e6, 0, 0};
    Real eta_shear = compute_triaxiality(sigma_shear);
    check(std::abs(eta_shear) < TOL,
          "Pure shear η = 0");
    std::cout << "  η (pure shear) = " << eta_shear << "\n";

    // Equibiaxial tension: η = 2/3
    Real sigma_biax[6] = {100e6, 100e6, 0, 0, 0, 0};
    Real eta_biax = compute_triaxiality(sigma_biax);
    check(std::abs(eta_biax - 2.0/3.0) < TOL,
          "Equibiaxial tension η = 2/3");
    std::cout << "  η (equibiaxial) = " << eta_biax << "\n";

    // Uniaxial compression: η = -1/3
    Real sigma_comp[6] = {-200e6, 0, 0, 0, 0, 0};
    Real eta_comp = compute_triaxiality(sigma_comp);
    check(std::abs(eta_comp + 1.0/3.0) < TOL,
          "Uniaxial compression η = -1/3");
    std::cout << "  η (compression) = " << eta_comp << "\n";
}

// Test 3: Maximum principal stress failure
void test_max_principal_stress_failure() {
    std::cout << "\n=== Test 3: Max Principal Stress Failure ===\n";

    FailureParameters params;
    params.criterion = FailureCriterion::MaxPrincipalStress;
    params.max_principal_stress = 300e6;  // 300 MPa tensile limit
    params.min_principal_stress = -500e6; // 500 MPa compressive limit

    // Below limit - should not fail
    Real sigma_safe[6] = {200e6, 50e6, 0, 0, 0, 0};
    check(!check_max_principal_stress(sigma_safe, params),
          "Below tensile limit - no failure");

    // Above tensile limit - should fail
    Real sigma_fail_tension[6] = {350e6, 0, 0, 0, 0, 0};
    check(check_max_principal_stress(sigma_fail_tension, params),
          "Above tensile limit - failure");

    // Above compressive limit - should fail
    Real sigma_fail_comp[6] = {-600e6, 0, 0, 0, 0, 0};
    check(check_max_principal_stress(sigma_fail_comp, params),
          "Above compressive limit - failure");

    // Mixed state at limits - should not fail
    Real sigma_mixed[6] = {250e6, -400e6, 0, 0, 0, 0};
    check(!check_max_principal_stress(sigma_mixed, params),
          "Mixed state within limits - no failure");
}

// Test 4: Plastic strain failure
void test_plastic_strain_failure() {
    std::cout << "\n=== Test 4: Plastic Strain Failure ===\n";

    FailureParameters params;
    params.max_plastic_strain = 0.5;  // 50% failure strain

    check(!check_plastic_strain(0.3, params),
          "Below limit (30%) - no failure");

    check(check_plastic_strain(0.6, params),
          "Above limit (60%) - failure");

    check(!check_plastic_strain(0.5 - 1e-10, params),
          "At limit (50% - ε) - no failure");

    check(check_plastic_strain(0.5 + 1e-10, params),
          "Just above limit (50% + ε) - failure");
}

// Test 5: Johnson-Cook damage accumulation
void test_jc_damage() {
    std::cout << "\n=== Test 5: Johnson-Cook Damage ===\n";

    FailureParameters params;
    params.JC_D1 = 0.05;
    params.JC_D2 = 3.44;
    params.JC_D3 = -2.12;
    params.JC_D4 = 0.002;
    params.JC_D5 = 0.61;

    // Test failure strain at different triaxialities
    // Uniaxial tension: η = 1/3
    Real eps_f_uniax = johnson_cook_failure_strain(1.0/3.0, 1.0, 0.0, params);
    std::cout << "  ε_f (uniaxial, η=1/3) = " << eps_f_uniax << "\n";
    check(eps_f_uniax > 0.0 && eps_f_uniax < 2.0,
          "Failure strain reasonable for uniaxial tension");

    // Pure shear: η = 0 (higher ductility)
    Real eps_f_shear = johnson_cook_failure_strain(0.0, 1.0, 0.0, params);
    std::cout << "  ε_f (pure shear, η=0) = " << eps_f_shear << "\n";
    check(eps_f_shear > eps_f_uniax,
          "Higher ductility in pure shear");

    // Compression: η = -1/3 (even higher ductility)
    Real eps_f_comp = johnson_cook_failure_strain(-1.0/3.0, 1.0, 0.0, params);
    std::cout << "  ε_f (compression, η=-1/3) = " << eps_f_comp << "\n";
    check(eps_f_comp > eps_f_shear,
          "Higher ductility in compression");

    // Damage accumulation
    Real sigma[6] = {300e6, 0, 0, 0, 0, 0};  // Uniaxial
    Real damage = 0.0;
    Real delta_eps = 0.01;

    for (int i = 0; i < 50; ++i) {
        damage = update_jc_damage(damage, delta_eps, sigma, 1.0, 0.0, params);
    }
    std::cout << "  Accumulated damage (50 steps): " << damage << "\n";
    check(damage > 0.0 && damage < 2.0,
          "Damage accumulates reasonably");
}

// Test 6: Cockcroft-Latham criterion
void test_cockcroft_latham() {
    std::cout << "\n=== Test 6: Cockcroft-Latham Criterion ===\n";

    // Tensile stress state
    Real sigma_tension[6] = {200e6, 0, 0, 0, 0, 0};
    Real W = 0.0;
    Real delta_eps = 0.01;

    W = update_cockcroft_latham(W, sigma_tension, delta_eps);
    Real expected_W = 200e6 * 0.01;
    check(std::abs(W - expected_W) < TOL * expected_W,
          "CL integral for tension");
    std::cout << "  W (tension) = " << W << " J/m³\n";

    // Compressive stress state - no contribution
    Real sigma_comp[6] = {-200e6, 0, 0, 0, 0, 0};
    Real W_comp = update_cockcroft_latham(0.0, sigma_comp, delta_eps);
    check(W_comp < TOL,
          "CL integral zero for compression");
    std::cout << "  W (compression) = " << W_comp << " J/m³\n";

    // Mixed state - only tensile σ1 contributes
    Real sigma_mixed[6] = {100e6, -50e6, -50e6, 0, 0, 0};
    Real W_mixed = update_cockcroft_latham(0.0, sigma_mixed, delta_eps);
    Real expected_mixed = 100e6 * 0.01;  // Only σ1 = 100 MPa contributes
    check(std::abs(W_mixed - expected_mixed) < TOL * expected_mixed,
          "CL integral for mixed state");
    std::cout << "  W (mixed) = " << W_mixed << " J/m³\n";
}

// Test 7: Element erosion manager
void test_erosion_manager() {
    std::cout << "\n=== Test 7: Element Erosion Manager ===\n";

    const std::size_t num_elements = 100;
    ElementErosionManager manager(num_elements);

    FailureParameters params;
    params.criterion = FailureCriterion::EffectivePlasticStrain;
    params.max_plastic_strain = 0.5;
    params.delete_on_failure = true;
    manager.set_failure_parameters(params);

    // Check initial state
    check(manager.eroded_count() == 0,
          "Initial eroded count is 0");

    for (std::size_t i = 0; i < num_elements; ++i) {
        check(manager.element_active(i),
              "All elements initially active");
        if (i > 0) break;  // Just check first few
    }

    // Create material state for testing
    MaterialState state_safe;
    state_safe.plastic_strain = 0.3;

    MaterialState state_fail;
    state_fail.plastic_strain = 0.6;

    // Check safe element
    bool safe_failed = manager.check_failure(10, state_safe);
    check(!safe_failed, "Safe element does not fail");
    check(manager.element_active(10), "Safe element still active");

    // Check failing element
    bool fail_failed = manager.check_failure(20, state_fail);
    check(fail_failed, "Failing element fails");
    check(!manager.element_active(20), "Failed element is not active");

    check(manager.eroded_count() == 1,
          "Eroded count incremented");

    // Fail more elements
    for (std::size_t i = 0; i < 10; ++i) {
        manager.check_failure(30 + i, state_fail);
    }
    check(manager.eroded_count() == 11,
          "Multiple elements eroded");

    // Print stats
    manager.print_summary();
}

// Test 8: Mass redistribution
void test_mass_redistribution() {
    std::cout << "\n=== Test 8: Mass Redistribution ===\n";

    const std::size_t num_elements = 10;
    ElementErosionManager manager(num_elements);

    FailureParameters params;
    params.criterion = FailureCriterion::EffectivePlasticStrain;
    params.max_plastic_strain = 0.5;
    params.redistribute_mass = true;
    manager.set_failure_parameters(params);

    // Element with 8 nodes
    Index node_indices[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    Real node_masses[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    Real element_mass = 2.0;  // Total element mass

    // Erode element and redistribute
    MaterialState state_fail;
    state_fail.plastic_strain = 0.6;
    manager.check_failure(0, state_fail);  // Erode element 0

    manager.redistribute_mass(0, element_mass, node_indices, 8, node_masses);

    // Check mass was redistributed
    Real expected_mass = 1.0 + element_mass / 8.0;
    bool masses_correct = true;
    for (int i = 0; i < 8; ++i) {
        if (std::abs(node_masses[i] - expected_mass) > TOL) {
            masses_correct = false;
        }
    }
    check(masses_correct, "Mass redistributed equally to nodes");

    // Check total mass conserved
    Real total_mass = 0.0;
    for (int i = 0; i < 8; ++i) {
        total_mass += node_masses[i];
    }
    check(std::abs(total_mass - 8.0 - element_mass) < TOL,
          "Total mass conserved");

    std::cout << "  Original node mass: 1.0 kg\n";
    std::cout << "  Element mass: " << element_mass << " kg\n";
    std::cout << "  New node mass: " << node_masses[0] << " kg\n";
}

// Test 9: Combined failure criteria
void test_combined_failure() {
    std::cout << "\n=== Test 9: Combined Failure Criteria ===\n";

    const std::size_t num_elements = 10;
    ElementErosionManager manager(num_elements);

    FailureParameters params;
    params.criterion = FailureCriterion::Combined;
    params.max_plastic_strain = 0.5;
    params.max_principal_stress = 300e6;
    params.max_vonmises_stress = 400e6;
    manager.set_failure_parameters(params);

    // Test 1: Fails by plastic strain
    MaterialState state1;
    state1.plastic_strain = 0.6;
    for (int i = 0; i < 6; ++i) state1.stress[i] = 0;
    state1.stress[0] = 100e6;
    check(manager.check_failure(0, state1),
          "Fails by plastic strain criterion");

    // Test 2: Fails by principal stress
    MaterialState state2;
    state2.plastic_strain = 0.1;
    for (int i = 0; i < 6; ++i) state2.stress[i] = 0;
    state2.stress[0] = 350e6;  // Above 300 MPa limit
    check(manager.check_failure(1, state2),
          "Fails by principal stress criterion");

    // Test 3: Fails by von Mises stress
    MaterialState state3;
    state3.plastic_strain = 0.1;
    // Biaxial stress state with high von Mises
    state3.stress[0] = 300e6;
    state3.stress[1] = -200e6;
    state3.stress[2] = 0;
    state3.stress[3] = 0;
    state3.stress[4] = 0;
    state3.stress[5] = 0;
    Real vm = Material::von_mises_stress(state3.stress);
    std::cout << "  Von Mises stress: " << vm/1e6 << " MPa\n";
    if (vm > params.max_vonmises_stress) {
        check(manager.check_failure(2, state3),
              "Fails by von Mises stress criterion");
    } else {
        std::cout << "  [INFO] VM stress below limit, skipping check\n";
    }

    // Test 4: All criteria satisfied - no failure
    MaterialState state4;
    state4.plastic_strain = 0.1;
    state4.stress[0] = 100e6;
    state4.stress[1] = 50e6;
    for (int i = 2; i < 6; ++i) state4.stress[i] = 0;
    check(!manager.check_failure(5, state4),
          "All criteria satisfied - no failure");
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Element Erosion and Failure Tests\n";
    std::cout << "========================================\n";

    Kokkos::initialize();
    {
        test_principal_stresses();
        test_triaxiality();
        test_max_principal_stress_failure();
        test_plastic_strain_failure();
        test_jc_damage();
        test_cockcroft_latham();
        test_erosion_manager();
        test_mass_redistribution();
        test_combined_failure();
    }
    Kokkos::finalize();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << tests_total << " tests passed\n";
    std::cout << "========================================\n";

    return (tests_passed == tests_total) ? 0 : 1;
}
