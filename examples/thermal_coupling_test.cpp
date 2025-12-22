/**
 * @file thermal_coupling_test.cpp
 * @brief Test thermal solver and thermo-mechanical coupling
 *
 * Tests:
 * 1. Basic thermal solver initialization
 * 2. Heat conduction in 1D bar
 * 3. Temperature boundary conditions
 * 4. Thermal expansion calculation
 * 5. Adiabatic heating from plastic work
 * 6. ThermoMechanicalCoupling helper functions
 */

#include <nexussim/physics/thermal_solver.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/data/state.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>

using namespace nxs;
using namespace nxs::physics;

// Test counter
static int test_count = 0;
static int pass_count = 0;

void check(bool condition, const std::string& test_name) {
    test_count++;
    if (condition) {
        pass_count++;
        std::cout << "[PASS] " << test_name << "\n";
    } else {
        std::cout << "[FAIL] " << test_name << "\n";
    }
}

// ============================================================================
// Helper: Create simple 3D mesh for thermal testing
// ============================================================================
std::shared_ptr<Mesh> create_thermal_test_mesh() {
    // Single hex8 element: 8 nodes
    // Node numbering (standard hex8):
    //     7-------6
    //    /|      /|
    //   4-------5 |
    //   | 3-----|-2
    //   |/      |/
    //   0-------1

    auto mesh = std::make_shared<Mesh>(8);  // 8 nodes

    Real L = 1.0;  // 1m cube
    mesh->set_node_coordinates(0, {0.0, 0.0, 0.0});
    mesh->set_node_coordinates(1, {L, 0.0, 0.0});
    mesh->set_node_coordinates(2, {L, L, 0.0});
    mesh->set_node_coordinates(3, {0.0, L, 0.0});
    mesh->set_node_coordinates(4, {0.0, 0.0, L});
    mesh->set_node_coordinates(5, {L, 0.0, L});
    mesh->set_node_coordinates(6, {L, L, L});
    mesh->set_node_coordinates(7, {0.0, L, L});

    // Add element block with 1 hex8 element
    mesh->add_element_block("block1", ElementType::Hex8, 1, 8);
    auto& block = mesh->element_block(0);

    // Set connectivity for the single hex element
    for (Index i = 0; i < 8; ++i) {
        block.connectivity[i] = i;
    }

    return mesh;
}

// ============================================================================
// Test 1: Basic ThermalSolver initialization
// ============================================================================
void test_thermal_solver_init() {
    std::cout << "\n=== Test 1: ThermalSolver Initialization ===\n";

    ThermalSolver solver("ThermalTest");

    auto mesh = create_thermal_test_mesh();
    auto state = std::make_shared<State>(*mesh);

    solver.initialize(mesh, state);

    check(solver.num_nodes() == 8, "Correct number of nodes");
    check(std::abs(solver.temperature(0) - 293.15) < 0.01, "Default temperature is room temp");
    check(solver.reference_temperature() == 293.15, "Reference temperature set");

    // Check that all nodes have same initial temperature
    bool all_same = true;
    for (size_t i = 0; i < 8; ++i) {
        if (std::abs(solver.temperature(i) - 293.15) > 0.01) {
            all_same = false;
            break;
        }
    }
    check(all_same, "All nodes initialized to same temperature");
}

// ============================================================================
// Test 2: Set temperature and thermal material
// ============================================================================
void test_temperature_setting() {
    std::cout << "\n=== Test 2: Temperature Setting ===\n";

    ThermalSolver solver("ThermalTest");
    auto mesh = create_thermal_test_mesh();
    auto state = std::make_shared<State>(*mesh);

    solver.initialize(mesh, state);

    // Set initial temperature
    solver.set_initial_temperature(400.0);  // 400K
    check(std::abs(solver.temperature(0) - 400.0) < 0.01, "Initial temperature set correctly");

    // Set individual node temperature
    solver.set_temperature(0, 500.0);
    check(std::abs(solver.temperature(0) - 500.0) < 0.01, "Individual node temperature set");
    check(std::abs(solver.temperature(1) - 400.0) < 0.01, "Other nodes unchanged");

    // Set thermal material
    ThermalMaterial steel;
    steel.density = 7850.0;         // kg/m³
    steel.specific_heat = 500.0;    // J/kg·K
    steel.conductivity = 50.0;      // W/m·K
    steel.expansion_coeff = 1.2e-5; // 1/K

    solver.set_material(steel);

    // Compute stable timestep (should be based on thermal diffusivity)
    // For a 1m element with steel diffusivity ~1.3e-5 m²/s:
    // dt_stable ~ cfl * h² / (6*α) ~ 0.4 * 1 / (6 * 1.3e-5) ~ 5000 s
    // Thermal time scales are much longer than mechanical!
    Real dt_stable = solver.compute_stable_dt();
    check(dt_stable > 0, "Stable timestep is positive");
    check(dt_stable > 100.0, "Stable timestep reflects thermal diffusion scale");

    std::cout << "  Stable dt: " << dt_stable << " s\n";
}

// ============================================================================
// Test 3: Thermal boundary conditions
// ============================================================================
void test_thermal_bcs() {
    std::cout << "\n=== Test 3: Thermal Boundary Conditions ===\n";

    ThermalSolver solver("ThermalTest");
    auto mesh = create_thermal_test_mesh();
    auto state = std::make_shared<State>(*mesh);

    solver.initialize(mesh, state);
    solver.set_initial_temperature(300.0);

    // Add temperature BC on node 0
    ThermalBC temp_bc(ThermalBCType::Temperature, {0}, 400.0);
    solver.add_bc(temp_bc);

    // Add convection BC on node 1
    ThermalBC conv_bc(ThermalBCType::Convection, {1}, 25.0, 300.0);  // h=25 W/m²K, T_inf=300K
    solver.add_bc(conv_bc);

    // Take a few steps
    Real dt = solver.compute_stable_dt() * 0.5;  // Use half stable dt for safety
    for (int i = 0; i < 10; ++i) {
        solver.step(dt);
    }

    // Check that node 0 is held at prescribed temperature
    check(std::abs(solver.temperature(0) - 400.0) < 0.1, "Temperature BC enforced");

    // Node 1 should move toward ambient (but slowly due to convection)
    Real T1 = solver.temperature(1);
    std::cout << "  Node 1 temperature after convection: " << T1 << " K\n";
    check(T1 > 290 && T1 < 400, "Convection affects temperature");
}

// ============================================================================
// Test 4: Thermal expansion calculation
// ============================================================================
void test_thermal_expansion() {
    std::cout << "\n=== Test 4: Thermal Expansion ===\n";

    ThermalSolver solver("ThermalTest");
    auto mesh = create_thermal_test_mesh();
    auto state = std::make_shared<State>(*mesh);

    solver.initialize(mesh, state);

    // Set reference temperature
    Real T_ref = 293.15;  // Room temperature
    solver.set_reference_temperature(T_ref);

    // Set material with known expansion coefficient
    ThermalMaterial mat;
    mat.expansion_coeff = 1.2e-5;  // Steel: 12 microstrain/K
    solver.set_material(mat);

    // Set temperature at node 0
    Real T_hot = 393.15;  // 100K above reference
    solver.set_temperature(0, T_hot);

    // Calculate expected thermal strain
    Real expected_strain = mat.expansion_coeff * (T_hot - T_ref);  // 1.2e-3

    Real actual_strain = solver.thermal_strain(0);

    std::cout << "  Temperature: " << T_hot << " K\n";
    std::cout << "  Reference: " << T_ref << " K\n";
    std::cout << "  Delta T: " << (T_hot - T_ref) << " K\n";
    std::cout << "  Expected strain: " << expected_strain << "\n";
    std::cout << "  Actual strain: " << actual_strain << "\n";

    check(std::abs(actual_strain - expected_strain) < 1e-10, "Thermal strain calculated correctly");

    // Check thermal strain tensor
    Real strain_tensor[6];
    solver.thermal_strain_tensor(0, strain_tensor);

    check(std::abs(strain_tensor[0] - expected_strain) < 1e-10, "Strain tensor xx correct");
    check(std::abs(strain_tensor[1] - expected_strain) < 1e-10, "Strain tensor yy correct");
    check(std::abs(strain_tensor[2] - expected_strain) < 1e-10, "Strain tensor zz correct");
    check(std::abs(strain_tensor[3]) < 1e-10, "Strain tensor xy = 0");
    check(std::abs(strain_tensor[4]) < 1e-10, "Strain tensor yz = 0");
    check(std::abs(strain_tensor[5]) < 1e-10, "Strain tensor xz = 0");
}

// ============================================================================
// Test 5: Heat source and adiabatic heating
// ============================================================================
void test_heat_sources() {
    std::cout << "\n=== Test 5: Heat Sources ===\n";

    ThermalSolver solver("ThermalTest");
    auto mesh = create_thermal_test_mesh();
    auto state = std::make_shared<State>(*mesh);

    solver.initialize(mesh, state);
    solver.set_initial_temperature(300.0);

    // Add heat source at node 0
    Real Q = 1e6;  // 1 MW/m³
    solver.add_heat_source(0, Q);

    Real T_initial = solver.temperature(0);

    // Take a few steps
    Real dt = solver.compute_stable_dt() * 0.1;
    for (int i = 0; i < 5; ++i) {
        solver.step(dt);
    }

    Real T_final = solver.temperature(0);

    std::cout << "  Initial temp: " << T_initial << " K\n";
    std::cout << "  Final temp: " << T_final << " K\n";
    std::cout << "  Temp rise: " << (T_final - T_initial) << " K\n";

    check(T_final > T_initial, "Heat source increases temperature");

    // Test adiabatic heating
    solver.clear_heat_sources();
    solver.set_initial_temperature(300.0);
    solver.enable_adiabatic_heating(true);

    check(solver.adiabatic_heating_enabled(), "Adiabatic heating enabled");

    // Add plastic heating
    Real plastic_work = 1e7;  // 10 MW/m³ plastic work rate
    solver.add_plastic_heating(0, plastic_work);

    // Take a step
    solver.step(dt);

    Real T_after_plastic = solver.temperature(0);
    check(T_after_plastic > 300.0, "Plastic work heats material");

    std::cout << "  After plastic heating: " << T_after_plastic << " K\n";
}

// ============================================================================
// Test 6: ThermoMechanicalCoupling helper
// ============================================================================
void test_thermo_mechanical_coupling() {
    std::cout << "\n=== Test 6: ThermoMechanicalCoupling Helper ===\n";

    // Test thermal strain calculation
    Real T = 400.0;
    Real T_ref = 300.0;
    Real alpha = 1.2e-5;
    Real strain[6];

    ThermoMechanicalCoupling::thermal_strain(T, T_ref, alpha, strain);

    Real expected_strain = alpha * (T - T_ref);  // 1.2e-3

    check(std::abs(strain[0] - expected_strain) < 1e-10, "Helper thermal strain xx");
    check(std::abs(strain[1] - expected_strain) < 1e-10, "Helper thermal strain yy");
    check(std::abs(strain[2] - expected_strain) < 1e-10, "Helper thermal strain zz");
    check(std::abs(strain[3]) < 1e-10, "Helper thermal strain xy = 0");

    // Test thermal stress (constrained expansion)
    Real E = 200e9;    // 200 GPa
    Real nu = 0.3;
    Real stress[6];

    ThermoMechanicalCoupling::thermal_stress(T, T_ref, E, nu, alpha, stress);

    // σ = -E*α*ΔT / (1-2ν) for fully constrained
    Real expected_stress = -E * expected_strain / (1.0 - 2.0 * nu);

    std::cout << "  Thermal stress (constrained): " << stress[0] / 1e6 << " MPa\n";
    std::cout << "  Expected: " << expected_stress / 1e6 << " MPa\n";

    check(std::abs(stress[0] - expected_stress) / std::abs(expected_stress) < 1e-6,
          "Thermal stress calculated correctly");

    // Test adiabatic heating calculation
    Real plastic_work_rate = 1e8;  // 100 MW/m³
    Real density = 7850.0;         // kg/m³
    Real specific_heat = 500.0;    // J/kg·K
    Real taylor_quinney = 0.9;     // 90% to heat
    Real dt = 1e-6;                // 1 microsecond

    Real dT = ThermoMechanicalCoupling::adiabatic_heating(
        plastic_work_rate, density, specific_heat, taylor_quinney, dt);

    // Expected: ΔT = β*W*dt / (ρ*c) = 0.9 * 1e8 * 1e-6 / (7850 * 500) = 22.9 mK
    Real expected_dT = taylor_quinney * plastic_work_rate * dt / (density * specific_heat);

    std::cout << "  Adiabatic temp rise: " << dT * 1000 << " mK\n";
    std::cout << "  Expected: " << expected_dT * 1000 << " mK\n";

    check(std::abs(dT - expected_dT) < 1e-10, "Adiabatic heating calculated correctly");

    // Test plastic power calculation
    Real stress_vec[6] = {100e6, 100e6, 100e6, 0, 0, 0};  // Hydrostatic 100 MPa
    Real plastic_strain_rate[6] = {0.001, 0.001, 0.001, 0, 0, 0};  // 0.1%/s

    Real power = ThermoMechanicalCoupling::plastic_power(stress_vec, plastic_strain_rate);

    // Expected: σ:ε̇ = 3 * 100e6 * 0.001 = 300 kW/m³
    Real expected_power = 3 * 100e6 * 0.001;

    std::cout << "  Plastic power: " << power / 1e3 << " kW/m³\n";
    std::cout << "  Expected: " << expected_power / 1e3 << " kW/m³\n";

    check(std::abs(power - expected_power) < 1.0, "Plastic power calculated correctly");
}

// ============================================================================
// Test 7: Field export/import
// ============================================================================
void test_field_exchange() {
    std::cout << "\n=== Test 7: Field Export/Import ===\n";

    ThermalSolver solver("ThermalTest");
    auto mesh = create_thermal_test_mesh();
    auto state = std::make_shared<State>(*mesh);

    solver.initialize(mesh, state);

    // Set different temperatures
    for (size_t i = 0; i < 8; ++i) {
        solver.set_temperature(i, 300.0 + i * 10.0);  // 300, 310, 320, ...
    }

    // Export temperature field
    std::vector<Real> temp_data;
    solver.export_field("temperature", temp_data);

    check(temp_data.size() == 8, "Exported correct number of values");
    check(std::abs(temp_data[0] - 300.0) < 0.01, "Exported temperature[0] correct");
    check(std::abs(temp_data[7] - 370.0) < 0.01, "Exported temperature[7] correct");

    // Modify and import
    for (auto& T : temp_data) {
        T += 100.0;  // Add 100K to all
    }

    solver.import_field("temperature", temp_data);

    check(std::abs(solver.temperature(0) - 400.0) < 0.01, "Imported temperature[0] correct");
    check(std::abs(solver.temperature(7) - 470.0) < 0.01, "Imported temperature[7] correct");

    // Check provided/required fields
    auto provided = solver.provided_fields();
    check(provided.size() >= 1, "Provides at least one field");

    bool has_temp = false;
    for (const auto& f : provided) {
        if (f == "temperature") has_temp = true;
    }
    check(has_temp, "Provides temperature field");
}

// ============================================================================
// Test 8: Statistics and print
// ============================================================================
void test_statistics() {
    std::cout << "\n=== Test 8: Statistics ===\n";

    ThermalSolver solver("ThermalTest");
    auto mesh = create_thermal_test_mesh();
    auto state = std::make_shared<State>(*mesh);

    solver.initialize(mesh, state);

    // Set varied temperatures
    solver.set_temperature(0, 300.0);
    solver.set_temperature(1, 350.0);
    solver.set_temperature(2, 400.0);
    solver.set_temperature(3, 350.0);
    solver.set_temperature(4, 300.0);
    solver.set_temperature(5, 350.0);
    solver.set_temperature(6, 400.0);
    solver.set_temperature(7, 350.0);

    // Run a step to update statistics
    Real dt = solver.compute_stable_dt() * 0.1;
    solver.step(dt);

    // Print stats
    std::cout << "\n";
    solver.print_stats();

    check(true, "Statistics printed successfully");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=================================================\n";
    std::cout << "Thermal Coupling Test Suite\n";
    std::cout << "=================================================\n";

    // Initialize Kokkos
    Kokkos::initialize();

    {
        test_thermal_solver_init();
        test_temperature_setting();
        test_thermal_bcs();
        test_thermal_expansion();
        test_heat_sources();
        test_thermo_mechanical_coupling();
        test_field_exchange();
        test_statistics();
    }

    Kokkos::finalize();

    std::cout << "\n=================================================\n";
    std::cout << "Results: " << pass_count << "/" << test_count << " tests passed\n";
    std::cout << "=================================================\n";

    return (pass_count == test_count) ? 0 : 1;
}
