/**
 * @file thermal_wave14_test.cpp
 * @brief Comprehensive test for Wave 14: Thermal solver capabilities
 *
 * Tests 8 classes (~40 assertions total):
 *  1. HeatConductionSolver  - initialization, conduction, heat sources
 *  2. ConvectionBC           - Newton cooling, apply_direct
 *  3. RadiationBC            - Stefan-Boltzmann, linearized h_rad
 *  4. FixedTemperatureBC     - Dirichlet overwrite
 *  5. HeatFluxBC             - Neumann flux
 *  6. AdiabaticHeating       - Taylor-Quinney plastic work to heat
 *  7. ThermalTimeStep        - stability limit, diffusivity, subcycles
 *  8. CoupledThermoMechanical - staggered coupling with BCs
 */

#include <nexussim/physics/thermal_wave14.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>

using namespace nxs;
using namespace nxs::physics;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { \
        std::cout << "[PASS] " << msg << "\n"; \
        tests_passed++; \
    } else { \
        std::cout << "[FAIL] " << msg << "\n"; \
        tests_failed++; \
    } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) do { \
    double _va = static_cast<double>(a); \
    double _vb = static_cast<double>(b); \
    double _vt = static_cast<double>(tol); \
    if (std::abs(_va - _vb) <= _vt) { \
        std::cout << "[PASS] " << msg << " (got " << _va << ", expected " << _vb << ")\n"; \
        tests_passed++; \
    } else { \
        std::cout << "[FAIL] " << msg \
            << " (got " << _va << ", expected " << _vb << ", tol " << _vt << ")\n"; \
        tests_failed++; \
    } \
} while(0)

// ============================================================================
// Test 1: HeatConductionSolver
// ============================================================================
void test_heat_conduction_solver() {
    std::cout << "\n=== HeatConductionSolver ===\n";

    // --- Test 1: Initialization defaults ---
    HeatConductionSolver solver;
    CHECK(!solver.initialized(), "Solver not initialized before init()");

    solver.initialize(5, 300.0, 7850.0, 500.0);
    CHECK(solver.initialized(), "Solver initialized after init()");
    CHECK(solver.num_nodes() == 5, "Solver has 5 nodes");
    CHECK_NEAR(solver.get_temperature(0), 300.0, 1e-12,
               "Initial temperature = 300 K");

    // --- Test 2: Set/get conductivity, density, specific heat ---
    solver.set_conductivity(45.0);
    CHECK_NEAR(solver.conductivity(), 45.0, 1e-12,
               "Conductivity set to 45 W/m-K");
    solver.set_density(8000.0);
    CHECK_NEAR(solver.density(), 8000.0, 1e-12, "Density set to 8000 kg/m^3");
    solver.set_specific_heat(450.0);
    CHECK_NEAR(solver.specific_heat(), 450.0, 1e-12,
               "Specific heat set to 450 J/kg-K");

    // --- Test 3: Heat source on isolated node (no connections) ---
    // With Q (W/m^3) on node 2 and no connectivity, after one step:
    //   dT = dt * Q / (rho * Cp)
    solver.initialize(3, 300.0, 8000.0, 450.0);
    solver.set_conductivity(45.0);
    Real Q = 1.0e6;  // 1 MW/m^3
    solver.add_heat_source(1, Q);
    Real dt = 0.01;
    solver.step(dt);
    Real expected_dT = dt * Q / (8000.0 * 450.0);
    CHECK_NEAR(solver.get_temperature(1), 300.0 + expected_dT, 1e-10,
               "Heat source raises temperature correctly");
    CHECK_NEAR(solver.get_temperature(0), 300.0, 1e-12,
               "Unheated node stays at initial temperature");

    // --- Test 4: Conduction between two connected nodes ---
    // Node 0 at 400 K, node 1 at 300 K, connected with area_over_dist = 0.01 m
    // flux = k * (A/d) * (T1-T0) into node 0 => negative (cooling node 0)
    solver.initialize(2, 300.0, 8000.0, 450.0);
    solver.set_conductivity(45.0);
    solver.set_temperature(0, 400.0);
    solver.set_temperature(1, 300.0);
    solver.set_nodal_volume(0, 1.0);
    solver.set_nodal_volume(1, 1.0);
    solver.add_connection(0, 1, 0.01);

    solver.step(0.001);
    // Node 0 should cool, node 1 should heat (energy conservation)
    Real T0 = solver.get_temperature(0);
    Real T1 = solver.get_temperature(1);
    CHECK(T0 < 400.0, "Hot node cools after conduction step");
    CHECK(T1 > 300.0, "Cold node heats after conduction step");
    // Energy conservation: rho*Cp*(T0+T1) should be same before and after
    Real energy_before = 8000.0 * 450.0 * (400.0 + 300.0);
    Real energy_after  = 8000.0 * 450.0 * (T0 + T1);
    CHECK_NEAR(energy_after, energy_before, 1e-6,
               "Energy conserved in conduction step");
}

// ============================================================================
// Test 2: ConvectionBC
// ============================================================================
void test_convection_bc() {
    std::cout << "\n=== ConvectionBC ===\n";

    // Node at 400 K, ambient 300 K, h=25 W/m^2-K, A=0.01 m^2
    // q_conv = -h * A * (T - T_amb) = -25 * 0.01 * 100 = -25 W
    ConvectionCondition cond;
    cond.boundary_nodes = {0};
    cond.h_coefficient = 25.0;
    cond.T_ambient = 300.0;

    ConvectionBC conv_bc(cond);
    conv_bc.set_uniform_area(0.01);

    // --- Test: apply to heat rates ---
    std::vector<Real> temps = {400.0};
    std::vector<Real> heat_rates = {0.0};
    conv_bc.apply(temps, heat_rates, 1.0, 0.001);
    CHECK_NEAR(heat_rates[0], -25.0, 1e-10,
               "Convection heat rate = -h*A*(T-T_amb) = -25 W");

    // --- Test: apply_direct lowers temperature ---
    temps = {400.0};
    Real rho = 8000.0, Cp = 450.0, V = 0.001;  // 1 cm^3
    Real dt = 0.01;
    conv_bc.apply_direct(temps, dt, rho, Cp, V);
    // dT = -25*0.01*100*0.01 / (8000*450*0.001) = -0.25/3.6 = -0.06944...
    Real expected = 400.0 + (-25.0 * 0.01 * 100.0 * dt) / (rho * Cp * V);
    CHECK_NEAR(temps[0], expected, 1e-10,
               "Convection apply_direct cools hot node correctly");

    // --- Test: at ambient, no heat transfer ---
    temps = {300.0};
    heat_rates = {0.0};
    conv_bc.apply(temps, heat_rates, 1.0, 0.001);
    CHECK_NEAR(heat_rates[0], 0.0, 1e-12,
               "No convection when T == T_ambient");
}

// ============================================================================
// Test 3: RadiationBC
// ============================================================================
void test_radiation_bc() {
    std::cout << "\n=== RadiationBC ===\n";

    // --- Test: Linearized h_rad ---
    // h_rad = 4 * sigma * eps * T^3
    // At 300 K, eps=0.8: h_rad = 4 * 5.6704e-8 * 0.8 * 300^3
    Real h_expected = 4.0 * STEFAN_BOLTZMANN * 0.8 * 300.0 * 300.0 * 300.0;
    Real h_calc = RadiationBC::linearized_h_rad(0.8, 300.0);
    CHECK_NEAR(h_calc, h_expected, 1e-12,
               "Linearized h_rad at 300 K, eps=0.8");

    // --- Test: Linearized h_rad with view factor ---
    Real h_vf = RadiationBC::linearized_h_rad(0.8, 0.5, 300.0);
    CHECK_NEAR(h_vf, h_expected * 0.5, 1e-12,
               "Linearized h_rad with view_factor=0.5");

    // --- Test: Radiation heat loss from hot surface ---
    RadiationCondition rcond;
    rcond.boundary_nodes = {0};
    rcond.emissivity = 1.0;
    rcond.T_environment = 0.0;  // Cold space
    rcond.view_factor = 1.0;

    RadiationBC rad_bc(rcond);
    rad_bc.set_uniform_area(1.0);

    std::vector<Real> temps = {1000.0};  // 1000 K
    std::vector<Real> heat_rates = {0.0};
    rad_bc.apply(temps, heat_rates);
    // q = -sigma * 1.0 * 1.0 * 1.0 * (1000^4 - 0)
    Real q_expected = -STEFAN_BOLTZMANN * 1.0 * 1.0 * 1.0 * 1e12;
    CHECK_NEAR(heat_rates[0], q_expected, std::abs(q_expected) * 1e-10,
               "Radiation flux from 1000 K surface to 0 K environment");

    // --- Test: No radiation when T == T_env ---
    rcond.T_environment = 500.0;
    RadiationBC rad_bc2(RadiationCondition{{0}, 0.9, 500.0, 1.0});
    rad_bc2.set_uniform_area(1.0);
    temps = {500.0};
    heat_rates = {0.0};
    rad_bc2.apply(temps, heat_rates);
    CHECK_NEAR(heat_rates[0], 0.0, 1e-10,
               "No radiation when T == T_environment");
}

// ============================================================================
// Test 4: FixedTemperatureBC
// ============================================================================
void test_fixed_temperature_bc() {
    std::cout << "\n=== FixedTemperatureBC ===\n";

    FixedTempCondition cond;
    cond.nodes = {0, 2};
    cond.temperature = 500.0;

    FixedTemperatureBC bc(cond);

    std::vector<Real> temps = {300.0, 300.0, 300.0, 300.0};
    bc.apply(temps);

    CHECK_NEAR(temps[0], 500.0, 1e-12, "Fixed BC overwrites node 0 to 500 K");
    CHECK_NEAR(temps[1], 300.0, 1e-12, "Non-fixed node 1 unchanged at 300 K");
    CHECK_NEAR(temps[2], 500.0, 1e-12, "Fixed BC overwrites node 2 to 500 K");

    // --- Test: add_condition and clear ---
    FixedTempCondition cond2;
    cond2.nodes = {3};
    cond2.temperature = 1000.0;
    bc.add_condition(cond2);

    temps = {0.0, 0.0, 0.0, 0.0};
    bc.apply(temps);
    CHECK_NEAR(temps[3], 1000.0, 1e-12,
               "Added condition sets node 3 to 1000 K");

    bc.clear();
    CHECK(bc.conditions().empty(), "Clear removes all conditions");
}

// ============================================================================
// Test 5: HeatFluxBC
// ============================================================================
void test_heat_flux_bc() {
    std::cout << "\n=== HeatFluxBC ===\n";

    // Flux of 1000 W/m^2 on node 0 with area 0.01 m^2
    HeatFluxCondition cond;
    cond.boundary_nodes = {0};
    cond.flux = 1000.0;

    HeatFluxBC bc(cond);
    bc.set_uniform_area(0.01);

    // --- Test: apply to heat rates ---
    std::vector<Real> heat_rates = {0.0, 0.0};
    bc.apply(heat_rates);
    CHECK_NEAR(heat_rates[0], 10.0, 1e-12,
               "Heat flux: Q = flux * A = 1000 * 0.01 = 10 W");
    CHECK_NEAR(heat_rates[1], 0.0, 1e-12,
               "Non-boundary node heat rate unchanged");

    // --- Test: apply_direct temperature increment ---
    std::vector<Real> temps = {300.0, 300.0};
    Real rho = 8000.0, Cp = 450.0, V = 0.001;
    Real dt = 0.01;
    bc.apply_direct(temps, dt, rho, Cp, V);
    // dT = flux * A * dt / (rho * Cp * V) = 1000 * 0.01 * 0.01 / (8000*450*0.001)
    Real dT_expected = 1000.0 * 0.01 * dt / (rho * Cp * V);
    CHECK_NEAR(temps[0], 300.0 + dT_expected, 1e-10,
               "Heat flux apply_direct raises temperature correctly");
}

// ============================================================================
// Test 6: AdiabaticHeating
// ============================================================================
void test_adiabatic_heating() {
    std::cout << "\n=== AdiabaticHeating ===\n";

    // --- Test: Basic Taylor-Quinney heating ---
    // dT = eta * sigma * d_eps_p / (rho * Cp)
    Real sigma = 500.0e6;    // 500 MPa
    Real deps_p = 0.001;     // 0.1% plastic strain increment
    Real rho = 7850.0;
    Real Cp = 500.0;
    Real eta = 0.9;
    Real dT = AdiabaticHeating::compute_heating(sigma, deps_p, rho, Cp, eta);
    Real dT_expected = 0.9 * 500.0e6 * 0.001 / (7850.0 * 500.0);
    CHECK_NEAR(dT, dT_expected, 1e-10,
               "Taylor-Quinney dT for 500 MPa, 0.1% strain");

    // --- Test: Zero density guard ---
    Real dT_zero = AdiabaticHeating::compute_heating(sigma, deps_p, 0.0, Cp);
    CHECK_NEAR(dT_zero, 0.0, 1e-15,
               "Zero density returns zero heating");

    // --- Test: compute_heating_from_power ---
    Real power = 1.0e9;  // 1 GW/m^3 plastic dissipation rate
    Real dt = 1.0e-6;
    Real dT_power = AdiabaticHeating::compute_heating_from_power(
        power, dt, rho, Cp, eta);
    Real dT_power_expected = 0.9 * 1.0e9 * 1.0e-6 / (7850.0 * 500.0);
    CHECK_NEAR(dT_power, dT_power_expected, 1e-10,
               "Heating from plastic power rate");

    // --- Test: apply_to_state ---
    MaterialState state;
    state.stress[0] = 400.0e6;  // uniaxial stress => von Mises = 400 MPa
    state.stress[1] = 0.0;
    state.stress[2] = 0.0;
    state.stress[3] = 0.0;
    state.stress[4] = 0.0;
    state.stress[5] = 0.0;
    state.plastic_strain = 0.005;
    state.temperature = 300.0;

    Real prev_eps_p = 0.004;  // increment = 0.001
    AdiabaticHeating::apply_to_state(state, prev_eps_p, rho, Cp, eta);
    Real vm = 400.0e6;  // von Mises for uniaxial
    Real expected_T = 300.0 + eta * vm * 0.001 / (rho * Cp);
    CHECK_NEAR(state.temperature, expected_T, 1e-6,
               "apply_to_state updates temperature from plastic work");

    // --- Test: DEFAULT_ETA constant ---
    CHECK_NEAR(AdiabaticHeating::DEFAULT_ETA, 0.9, 1e-15,
               "Default Taylor-Quinney coefficient = 0.9");
}

// ============================================================================
// Test 7: ThermalTimeStep
// ============================================================================
void test_thermal_timestep() {
    std::cout << "\n=== ThermalTimeStep ===\n";

    // Steel: k=50, rho=7850, Cp=500, h=0.01 m
    Real k = 50.0, rho = 7850.0, Cp = 500.0, h = 0.01;

    // --- Test: stable dt = h^2 * rho * Cp / (2*k) ---
    Real dt = ThermalTimeStep::compute_stable_dt(h, k, rho, Cp);
    Real dt_expected = h * h * rho * Cp / (2.0 * k);
    CHECK_NEAR(dt, dt_expected, 1e-12,
               "Stable thermal dt for steel, h=10mm");

    // --- Test: thermal diffusivity ---
    Real alpha = ThermalTimeStep::diffusivity(k, rho, Cp);
    Real alpha_expected = k / (rho * Cp);
    CHECK_NEAR(alpha, alpha_expected, 1e-15,
               "Thermal diffusivity alpha = k/(rho*Cp)");

    // --- Test: zero conductivity returns max dt ---
    Real dt_zero = ThermalTimeStep::compute_stable_dt(h, 0.0, rho, Cp);
    CHECK(dt_zero > 1e30, "Zero conductivity returns very large dt");

    // --- Test: thermal_to_mechanical_ratio ---
    Real c_sound = 5000.0;  // m/s for steel
    Real ratio = ThermalTimeStep::thermal_to_mechanical_ratio(
        h, k, rho, Cp, c_sound);
    Real dt_mech = h / c_sound;
    Real expected_ratio = dt_expected / dt_mech;
    CHECK_NEAR(ratio, expected_ratio, 1e-10,
               "Thermal-to-mechanical time step ratio");
    CHECK(ratio > 1.0,
          "Thermal dt > mechanical dt (typical for metals)");

    // --- Test: recommended subcycles ---
    Real mech_dt = 1.0e-3;  // 1 ms mechanical step
    int nsub = ThermalTimeStep::recommended_subcycles(mech_dt, h, k, rho, Cp);
    int nsub_expected = static_cast<int>(std::ceil(mech_dt / dt_expected));
    CHECK(nsub == nsub_expected,
          "Recommended subcycles matches ceiling(dt_mech/dt_thermal)");
    CHECK(nsub >= 1, "Subcycles at least 1");
}

// ============================================================================
// Test 8: CoupledThermoMechanical
// ============================================================================
void test_coupled_thermo_mechanical() {
    std::cout << "\n=== CoupledThermoMechanical ===\n";

    // Setup thermal solver: 3 nodes, 300 K
    HeatConductionSolver thermal;
    thermal.initialize(3, 300.0, 8000.0, 450.0);
    thermal.set_conductivity(45.0);

    // Setup fixed temp BC on node 0
    FixedTempCondition fix_cond;
    fix_cond.nodes = {0};
    fix_cond.temperature = 500.0;
    FixedTemperatureBC fix_bc(fix_cond);

    // Setup convection BC on node 2
    ConvectionCondition conv_cond;
    conv_cond.boundary_nodes = {2};
    conv_cond.h_coefficient = 25.0;
    conv_cond.T_ambient = 300.0;
    ConvectionBC conv_bc(conv_cond);
    conv_bc.set_uniform_area(0.01);

    // Configure coupling
    CouplingConfig cfg;
    cfg.thermal_subcycles = 5;
    cfg.taylor_quinney = 0.9;
    cfg.staggered = true;
    cfg.reference_temperature = 293.15;

    CoupledThermoMechanical coupled;
    coupled.configure(cfg);
    coupled.set_thermal_solver(&thermal);
    coupled.set_fixed_temp_bc(&fix_bc);
    coupled.set_convection_bc(&conv_bc);

    // --- Test: initial state ---
    CHECK(coupled.step_count() == 0, "Step count initially 0");
    CHECK_NEAR(coupled.total_time(), 0.0, 1e-15, "Total time initially 0");

    // --- Test: configuration read back ---
    CHECK(coupled.config().thermal_subcycles == 5,
          "Config thermal_subcycles = 5");
    CHECK_NEAR(coupled.config().taylor_quinney, 0.9, 1e-15,
               "Config Taylor-Quinney = 0.9");

    // --- Test: step with fixed BC and convection ---
    bool mech_called = false;
    auto mech_step = [&mech_called](Real /*dt*/) { mech_called = true; };

    Real dt = 0.001;
    coupled.step(dt, mech_step);

    CHECK(mech_called, "Mechanical step function was invoked");
    CHECK(coupled.step_count() == 1, "Step count after one step");
    CHECK_NEAR(coupled.total_time(), dt, 1e-15, "Total time after one step");

    // Fixed BC node must be 500 K
    CHECK_NEAR(thermal.get_temperature(0), 500.0, 1e-12,
               "Fixed BC node 0 remains at 500 K after coupled step");

    // --- Test: adiabatic heating ---
    Real T_before = thermal.get_temperature(1);
    coupled.add_adiabatic_heat(1, 500.0e6, 0.001, 8000.0, 450.0);
    coupled.step(dt, nullptr);  // null mechanical step is valid

    Real T_after = thermal.get_temperature(1);
    CHECK(T_after > T_before,
          "Adiabatic heating raises node 1 temperature");
    CHECK(coupled.step_count() == 2, "Step count after two steps");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "NexusSim Wave 14: Thermal Solver Tests\n";
    std::cout << "========================================\n";

    test_heat_conduction_solver();      // 8 assertions
    test_convection_bc();               // 3 assertions
    test_radiation_bc();                // 4 assertions
    test_fixed_temperature_bc();        // 5 assertions
    test_heat_flux_bc();                // 3 assertions
    test_adiabatic_heating();           // 5 assertions
    test_thermal_timestep();            // 6 assertions
    test_coupled_thermo_mechanical();   // 9 assertions

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    std::cout << "========================================\n";

    return (tests_failed > 0) ? 1 : 0;
}
