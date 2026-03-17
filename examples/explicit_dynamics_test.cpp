/**
 * @file explicit_dynamics_test.cpp
 * @brief Wave 9: Comprehensive tests for explicit dynamics enhancements
 *
 * Tests:
 * 1. Bulk viscosity (12 tests)
 * 2. Hourglass control (11 tests)
 * 3. Energy monitor (16 tests)
 * 4. Element erosion integration (12 tests)
 * 5. Explicit dynamics configuration (10 tests)
 * 6. Integration tests (10 tests)
 * Total: ~71 tests
 */

#include <nexussim/fem/explicit_dynamics.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/physics/element_erosion.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <string>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; std::cout << "[FAIL] " << msg << "\n"; } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) do { \
    double _va = static_cast<double>(a); \
    double _vb = static_cast<double>(b); \
    double _vt = static_cast<double>(tol); \
    if (std::abs(_va - _vb) <= _vt) { tests_passed++; } \
    else { tests_failed++; std::cout << "[FAIL] " << msg \
        << " (got " << _va << ", expected " << _vb << ", tol " << _vt << ")\n"; } \
} while(0)

using namespace nxs;
using namespace nxs::fem;
using namespace nxs::physics;

// ============================================================================
// 1. Bulk Viscosity Tests
// ============================================================================

void test_bulk_viscosity_default() {
    BulkViscosity bv;
    CHECK_NEAR(bv.C_linear, 0.06, 1e-10, "BV default C_linear");
    CHECK_NEAR(bv.C_quadratic, 1.2, 1e-10, "BV default C_quadratic");
}

void test_bulk_viscosity_custom() {
    BulkViscosity bv(0.1, 1.5);
    CHECK_NEAR(bv.C_linear, 0.1, 1e-10, "BV custom C_linear");
    CHECK_NEAR(bv.C_quadratic, 1.5, 1e-10, "BV custom C_quadratic");
}

void test_bulk_viscosity_zero_in_tension() {
    BulkViscosity bv;
    Real q = bv.compute(0.1, 7800.0, 5000.0, 0.01);
    CHECK_NEAR(q, 0.0, 1e-20, "BV zero in tension");
}

void test_bulk_viscosity_zero_at_zero() {
    BulkViscosity bv;
    Real q = bv.compute(0.0, 7800.0, 5000.0, 0.01);
    CHECK_NEAR(q, 0.0, 1e-20, "BV zero at zero rate");
}

void test_bulk_viscosity_positive_compression() {
    BulkViscosity bv;
    Real q = bv.compute(-1000.0, 7800.0, 5000.0, 0.01);
    CHECK(q > 0.0, "BV positive in compression");
}

void test_bulk_viscosity_linear_only() {
    BulkViscosity bv(0.06, 0.0);
    Real rho = 7800.0, c = 5000.0, L = 0.01, rate = -1000.0;
    Real q = bv.compute(rate, rho, c, L);
    Real expected = 0.06 * rho * c * L * 1000.0;
    CHECK_NEAR(q, expected, 1e-6, "BV linear component");
}

void test_bulk_viscosity_quadratic_only() {
    BulkViscosity bv(0.0, 1.2);
    Real rho = 7800.0, L = 0.01, rate = -1000.0;
    Real q = bv.compute(rate, rho, 5000.0, L);
    Real expected = 1.2 * rho * L * L * 1000.0 * 1000.0;
    CHECK_NEAR(q, expected, 1e-6, "BV quadratic component");
}

void test_bulk_viscosity_combined() {
    BulkViscosity bv(0.06, 1.2);
    Real rho = 7800.0, c = 5000.0, L = 0.01, rate = -500.0;
    Real q = bv.compute(rate, rho, c, L);
    Real q_l = 0.06 * rho * c * L * 500.0;
    Real q_q = 1.2 * rho * L * L * 500.0 * 500.0;
    CHECK_NEAR(q, q_l + q_q, 1e-6, "BV combined = linear + quadratic");
}

void test_bulk_viscosity_energy_rate() {
    Real q = 1000.0, ev_dot = -500.0, V = 1e-6;
    Real e_rate = BulkViscosity::energy_rate(q, ev_dot, V);
    CHECK_NEAR(e_rate, 0.5, 1e-10, "BV energy rate = -q*ev_dot*V");
    CHECK(e_rate > 0.0, "BV energy rate positive");
}

void test_bulk_viscosity_add_to_stress() {
    Real stress[6] = {100.0, 200.0, 300.0, 50.0, 60.0, 70.0};
    BulkViscosity::add_to_stress(50.0, stress);
    CHECK_NEAR(stress[0], 50.0, 1e-10, "BV stress xx -= q");
    CHECK_NEAR(stress[1], 150.0, 1e-10, "BV stress yy -= q");
    CHECK_NEAR(stress[2], 250.0, 1e-10, "BV stress zz -= q");
    CHECK_NEAR(stress[3], 50.0, 1e-10, "BV shear xy unchanged");
}

void test_bulk_viscosity_super_linear() {
    BulkViscosity bv;
    Real rho = 7800.0, c = 5000.0, L = 0.01;
    Real q1 = bv.compute(-100.0, rho, c, L);
    Real q2 = bv.compute(-200.0, rho, c, L);
    CHECK(q2 > 2.0 * q1, "BV scales super-linearly with rate");
}

// ============================================================================
// 2. Hourglass Control Tests
// ============================================================================

void test_hourglass_default() {
    HourglassControl hg;
    CHECK(hg.type == HourglassType::FlanaganBelytschko, "HG default type");
    CHECK_NEAR(hg.viscous_coefficient, 0.1, 1e-10, "HG default viscous");
    CHECK_NEAR(hg.stiffness_coefficient, 0.05, 1e-10, "HG default stiffness");
    CHECK_NEAR(hg.total_energy, 0.0, 1e-20, "HG initial energy");
}

void test_hourglass_custom() {
    HourglassControl hg(HourglassType::PerturbationStiffness, 0.2, 0.1);
    CHECK(hg.type == HourglassType::PerturbationStiffness, "HG custom type");
}

void test_hourglass_fb_stiffness() {
    HourglassControl hg(HourglassType::FlanaganBelytschko, 0.1, 0.05);
    Real K = 160e9, G = 80e9;
    Real k_hg = hg.compute_stiffness(K, G);
    CHECK_NEAR(k_hg, 0.1 * G, 1e-6, "HG FB stiffness = 0.1*G");
}

void test_hourglass_perturbation_stiffness() {
    HourglassControl hg(HourglassType::PerturbationStiffness, 0.1, 0.05);
    Real K = 160e9, G = 80e9;
    Real expected = 0.05 * (K + 4.0 / 3.0 * G);
    CHECK_NEAR(hg.compute_stiffness(K, G), expected, 1e-6, "HG perturbation stiffness");
}

void test_hourglass_combined_stiffness() {
    HourglassControl hg(HourglassType::Combined, 0.1, 0.05);
    Real K = 160e9, G = 80e9;
    Real expected = 0.1 * G + 0.05 * K;
    CHECK_NEAR(hg.compute_stiffness(K, G), expected, 1e-6, "HG combined stiffness");
}

void test_hourglass_none_stiffness() {
    HourglassControl hg(HourglassType::None);
    CHECK_NEAR(hg.compute_stiffness(160e9, 80e9), 0.0, 1e-20, "HG none stiffness = 0");
}

void test_hourglass_energy_computation() {
    Real hg_force[6] = {10.0, 20.0, 30.0, -10.0, -20.0, -30.0};
    Real velocity[6] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    Real dt = 1e-6;
    Real energy = HourglassControl::compute_energy(hg_force, velocity, 6, dt);
    CHECK_NEAR(energy, 0.0, 1e-20, "HG energy for antisymmetric forces = 0");
}

void test_hourglass_accumulation() {
    HourglassControl hg;
    hg.accumulate_energy(0.1);
    hg.accumulate_energy(0.2);
    hg.accumulate_energy(0.3);
    CHECK_NEAR(hg.total_energy, 0.6, 1e-10, "HG energy accumulation");
}

void test_hourglass_reset() {
    HourglassControl hg;
    hg.accumulate_energy(1.0);
    hg.reset_energy();
    CHECK_NEAR(hg.total_energy, 0.0, 1e-20, "HG energy reset");
}

void test_hourglass_excessive() {
    HourglassControl hg;
    hg.accumulate_energy(15.0);
    CHECK(hg.is_excessive(100.0, 0.1), "HG 15% > 10% threshold");
    CHECK(!hg.is_excessive(200.0, 0.1), "HG 7.5% < 10% threshold");
    CHECK(!hg.is_excessive(0.0, 0.1), "HG with zero IE returns false");
}

// ============================================================================
// 3. Energy Monitor Tests
// ============================================================================

void test_energy_monitor_default() {
    EnergyMonitor em;
    CHECK_NEAR(em.tolerance(), 0.05, 1e-10, "EM default tolerance");
    CHECK(!em.is_initialized(), "EM not initialized initially");
    CHECK(em.num_records() == 0, "EM no records initially");
}

void test_energy_monitor_tolerance() {
    EnergyMonitor em;
    em.set_tolerance(0.02);
    CHECK_NEAR(em.tolerance(), 0.02, 1e-10, "EM custom tolerance");
}

void test_energy_components_default() {
    EnergyMonitor::EnergyComponents ec;
    CHECK_NEAR(ec.kinetic, 0.0, 1e-20, "EC default kinetic");
    CHECK_NEAR(ec.internal, 0.0, 1e-20, "EC default internal");
    CHECK_NEAR(ec.total(), 0.0, 1e-20, "EC default total");
}

void test_energy_components_total() {
    EnergyMonitor::EnergyComponents ec;
    ec.kinetic = 100.0;
    ec.internal = 200.0;
    ec.hourglass = 10.0;
    CHECK_NEAR(ec.total(), 310.0, 1e-10, "EC total = KE + IE + HG");
}

void test_energy_components_expected() {
    EnergyMonitor::EnergyComponents ec;
    ec.external_work = 500.0;
    ec.damping = 50.0;
    ec.contact = 30.0;
    ec.bulk_viscosity = 20.0;
    ec.eroded = 10.0;
    CHECK_NEAR(ec.expected(), 390.0, 1e-10, "EC expected = W - damp - contact - BV - eroded");
}

void test_energy_components_balance() {
    EnergyMonitor::EnergyComponents ec;
    ec.kinetic = 200.0;
    ec.internal = 100.0;
    ec.external_work = 300.0;
    CHECK_NEAR(ec.balance_error(), 0.0, 1e-10, "EC zero balance error when balanced");
}

void test_energy_components_hg_ratio() {
    EnergyMonitor::EnergyComponents ec;
    ec.internal = 100.0;
    ec.hourglass = 5.0;
    CHECK_NEAR(ec.hourglass_ratio(), 0.05, 1e-10, "EC HG ratio = 5%");
}

void test_energy_monitor_initialize() {
    EnergyMonitor em;
    EnergyMonitor::EnergyComponents initial;
    initial.kinetic = 100.0;
    initial.internal = 50.0;
    em.initialize(initial);
    CHECK(em.is_initialized(), "EM initialized");
    CHECK(em.num_records() == 1, "EM 1 record after init");
    CHECK_NEAR(em.initial_total(), 150.0, 1e-10, "EM initial total");
}

void test_energy_monitor_record() {
    EnergyMonitor em;
    EnergyMonitor::EnergyComponents ec;
    ec.kinetic = 100.0;
    em.initialize(ec);

    ec.kinetic = 80.0;
    ec.internal = 15.0;
    auto flags = em.record(ec);
    CHECK(em.num_records() == 2, "EM 2 records");
    CHECK(!flags.energy_explosion, "EM no explosion");
}

void test_energy_monitor_explosion() {
    EnergyMonitor em;
    EnergyMonitor::EnergyComponents initial;
    initial.kinetic = 100.0;
    em.initialize(initial);

    EnergyMonitor::EnergyComponents exploded;
    exploded.kinetic = 5000.0;
    auto flags = em.record(exploded);
    CHECK(flags.energy_explosion, "EM detects energy explosion");
}

void test_energy_monitor_balance_violation() {
    EnergyMonitor em;
    em.set_tolerance(0.01);
    EnergyMonitor::EnergyComponents initial;
    initial.kinetic = 100.0;
    em.initialize(initial);

    EnergyMonitor::EnergyComponents bad;
    bad.kinetic = 120.0;
    auto flags = em.record(bad);
    CHECK(flags.energy_balance_violated, "EM detects balance violation");
}

void test_energy_monitor_hg_warning() {
    EnergyMonitor em;
    EnergyMonitor::EnergyComponents ec;
    ec.kinetic = 100.0;
    ec.internal = 100.0;
    ec.hourglass = 15.0;
    em.initialize(ec);
    auto flags = em.record(ec);
    CHECK(flags.hourglass_excessive, "EM detects excessive HG");
}

void test_energy_kinetic_computation() {
    Real velocity[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Real mass[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    Real ke = EnergyMonitor::compute_kinetic_energy(velocity, mass, 6);
    CHECK_NEAR(ke, 45.5, 1e-10, "EM kinetic = 0.5 * sum(m*v^2)");
}

void test_energy_internal_computation() {
    Real stress[6] = {100.0, 200.0, 300.0, 50.0, 60.0, 70.0};
    Real strain[6] = {0.001, 0.002, 0.003, 0.0005, 0.0006, 0.0007};
    Real volume = 1e-6;
    Real ie = EnergyMonitor::compute_element_internal_energy(stress, strain, volume);
    Real expected = 0.5 * (0.1 + 0.4 + 0.9 + 0.025 + 0.036 + 0.049) * volume;
    CHECK_NEAR(ie, expected, 1e-15, "EM internal energy");
}

void test_energy_external_work() {
    Real f_ext[3] = {100.0, 200.0, 300.0};
    Real du[3] = {0.001, 0.002, 0.003};
    Real work = EnergyMonitor::compute_external_work_increment(f_ext, du, 3);
    CHECK_NEAR(work, 1.4, 1e-10, "EM external work = f·du");
}

void test_energy_history() {
    EnergyMonitor em;
    EnergyMonitor::EnergyComponents ec;
    ec.kinetic = 100.0;
    em.initialize(ec);
    ec.kinetic = 90.0;
    em.record(ec);
    ec.kinetic = 80.0;
    em.record(ec);
    CHECK(em.history().size() == 3, "EM history size 3");
    CHECK_NEAR(em.latest().kinetic, 80.0, 1e-10, "EM latest KE");
}

// ============================================================================
// 4. Element Erosion Integration Tests (using physics::ElementErosionManager)
// ============================================================================

void test_erosion_construction() {
    physics::ElementErosionManager eem(4);
    CHECK(eem.element_active(0), "EEM element 0 active");
    CHECK(eem.element_active(3), "EEM element 3 active");
    CHECK(eem.eroded_count() == 0, "EEM none eroded");
}

void test_erosion_erode_element() {
    physics::ElementErosionManager eem(4);
    eem.erode_element(1, 2.5);
    CHECK(!eem.element_active(1), "EEM element 1 eroded");
    CHECK(eem.element_active(0), "EEM element 0 still active");
    CHECK(eem.eroded_count() == 1, "EEM 1 eroded");
    CHECK_NEAR(eem.total_eroded_mass(), 2.5, 1e-10, "EEM eroded mass");
}

void test_erosion_double_erode() {
    physics::ElementErosionManager eem(4);
    eem.erode_element(1, 2.5);
    eem.erode_element(1, 2.5);  // Double call
    CHECK(eem.eroded_count() == 1, "EEM no double-erode");
}

void test_erosion_state() {
    physics::ElementErosionManager eem(4);
    CHECK(eem.element_state(0) == physics::ElementState::Active, "EEM state active");
    eem.erode_element(0, 1.0);
    CHECK(eem.element_state(0) == physics::ElementState::Eroded, "EEM state eroded");
}

void test_erosion_check_plastic_strain() {
    physics::ElementErosionManager eem(4);
    physics::FailureParameters params;
    params.criterion = physics::FailureCriterion::EffectivePlasticStrain;
    params.max_plastic_strain = 0.5;
    eem.set_failure_parameters(params);

    physics::MaterialState state;
    state.plastic_strain = 0.6;  // Exceeds threshold
    bool failed = eem.check_failure(0, state);
    CHECK(failed, "EEM plastic strain failure detected");
    CHECK(!eem.element_active(0), "EEM element eroded after failure");
}

void test_erosion_check_no_failure() {
    physics::ElementErosionManager eem(4);
    physics::FailureParameters params;
    params.criterion = physics::FailureCriterion::EffectivePlasticStrain;
    params.max_plastic_strain = 0.5;
    eem.set_failure_parameters(params);

    physics::MaterialState state;
    state.plastic_strain = 0.3;  // Below threshold
    bool failed = eem.check_failure(0, state);
    CHECK(!failed, "EEM no failure below threshold");
    CHECK(eem.element_active(0), "EEM element still active");
}

void test_erosion_vonmises() {
    physics::ElementErosionManager eem(3);
    physics::FailureParameters params;
    params.criterion = physics::FailureCriterion::VonMisesStress;
    params.max_vonmises_stress = 300e6;
    eem.set_failure_parameters(params);

    physics::MaterialState state;
    state.stress[0] = 400e6;  // σxx > threshold → VM ≈ 400 MPa > 300 MPa
    bool failed = eem.check_failure(0, state);
    CHECK(failed, "EEM von Mises failure detected");
}

void test_erosion_stats() {
    physics::ElementErosionManager eem(10);
    eem.erode_element(0, 1.0);
    eem.erode_element(3, 1.5);
    eem.erode_element(7, 2.0);
    auto stats = eem.get_stats();
    CHECK(stats.total_elements == 10, "EEM stats total");
    CHECK(stats.eroded_elements == 3, "EEM stats eroded");
    CHECK(stats.active_elements == 7, "EEM stats active");
    CHECK_NEAR(stats.total_eroded_mass, 4.5, 1e-10, "EEM stats mass");
}

void test_erosion_combined_criteria() {
    physics::ElementErosionManager eem(3);
    physics::FailureParameters params;
    params.criterion = physics::FailureCriterion::Combined;
    params.max_plastic_strain = 0.5;
    params.max_principal_stress = 500e6;
    params.max_vonmises_stress = 300e6;
    eem.set_failure_parameters(params);

    // Element fails by plastic strain only
    physics::MaterialState state;
    state.plastic_strain = 0.6;
    state.stress[0] = 100e6;  // Low stress
    bool failed = eem.check_failure(0, state);
    CHECK(failed, "EEM combined: plastic strain triggers failure");
}

void test_erosion_already_eroded() {
    physics::ElementErosionManager eem(3);
    eem.erode_element(0, 1.0);

    physics::FailureParameters params;
    params.criterion = physics::FailureCriterion::EffectivePlasticStrain;
    params.max_plastic_strain = 0.5;
    eem.set_failure_parameters(params);

    physics::MaterialState state;
    state.plastic_strain = 0.6;
    bool failed = eem.check_failure(0, state);
    // Already eroded, check_failure returns true but no double-count
    CHECK(failed, "EEM already-eroded returns true");
    CHECK(eem.eroded_count() == 1, "EEM eroded count doesn't increase");
}

void test_eroded_energy_computation() {
    // Test the helper function from explicit_dynamics.hpp
    physics::ElementErosionManager eem(3);
    eem.erode_element(1, 1.0);

    Real stresses[18] = {}; // 3 elements × 6
    Real strains[18] = {};
    Real volumes[3] = {1e-6, 1e-6, 1e-6};
    stresses[6] = 100e6;  // Element 1 σxx
    strains[6] = 0.001;   // Element 1 εxx

    Real e_eroded = compute_eroded_energy(eem, stresses, strains, volumes, 3);
    Real expected = 0.5 * 100e6 * 0.001 * 1e-6;
    CHECK_NEAR(e_eroded, expected, 1e-10, "Eroded energy computation");
}

// ============================================================================
// 5. Explicit Dynamics Configuration Tests
// ============================================================================

void test_config_default() {
    ExplicitDynamicsConfig cfg;
    CHECK(cfg.bulk_viscosity_enabled, "CFG BV enabled by default");
    CHECK(cfg.hourglass_enabled, "CFG HG enabled by default");
    CHECK(cfg.energy_monitoring_enabled, "CFG energy monitoring enabled");
    CHECK(!cfg.erosion_enabled, "CFG erosion disabled by default");
    CHECK_NEAR(cfg.cfl_factor, 0.9, 1e-10, "CFG default CFL");
    CHECK_NEAR(cfg.energy_tolerance, 0.05, 1e-10, "CFG default energy tol");
}

void test_config_crash_preset() {
    auto cfg = ExplicitDynamicsConfig::crash_preset();
    CHECK(cfg.bulk_viscosity_enabled, "CFG crash BV enabled");
    CHECK(cfg.hourglass_enabled, "CFG crash HG enabled");
    CHECK(cfg.erosion_enabled, "CFG crash erosion enabled");
    CHECK_NEAR(cfg.erosion_mass_limit, 0.1, 1e-10, "CFG crash erosion limit");
    CHECK(cfg.hourglass.type == HourglassType::FlanaganBelytschko, "CFG crash HG type");
}

void test_config_blast_preset() {
    auto cfg = ExplicitDynamicsConfig::blast_preset();
    CHECK_NEAR(cfg.bulk_viscosity.C_linear, 0.1, 1e-10, "CFG blast BV C_l");
    CHECK_NEAR(cfg.bulk_viscosity.C_quadratic, 1.5, 1e-10, "CFG blast BV C_q");
    CHECK(cfg.hourglass.type == HourglassType::PerturbationStiffness, "CFG blast HG type");
    CHECK_NEAR(cfg.cfl_factor, 0.7, 1e-10, "CFG blast CFL");
}

void test_config_impact_preset() {
    auto cfg = ExplicitDynamicsConfig::impact_preset();
    CHECK(!cfg.erosion_enabled, "CFG impact erosion disabled");
    CHECK_NEAR(cfg.hourglass.viscous_coefficient, 0.05, 1e-10, "CFG impact HG viscous");
}

void test_config_rayleigh() {
    ExplicitDynamicsConfig cfg;
    cfg.rayleigh_alpha = 10.0;
    cfg.rayleigh_beta = 0.0001;
    CHECK_NEAR(cfg.rayleigh_alpha, 10.0, 1e-10, "CFG Rayleigh alpha");
    CHECK_NEAR(cfg.rayleigh_beta, 0.0001, 1e-10, "CFG Rayleigh beta");
}

void test_config_dt_limits() {
    ExplicitDynamicsConfig cfg;
    CHECK_NEAR(cfg.dt_min, 1.0e-12, 1e-20, "CFG dt_min");
    CHECK_NEAR(cfg.dt_max, 1.0e-3, 1e-10, "CFG dt_max");
    CHECK(!cfg.adaptive_dt, "CFG adaptive dt off");
}

// ============================================================================
// 6. Integration Tests
// ============================================================================

void test_vol_strain_rate() {
    Real strain_rate[6] = {-100.0, -50.0, -30.0, 10.0, 20.0, 30.0};
    Real ev_dot = volumetric_strain_rate(strain_rate);
    CHECK_NEAR(ev_dot, -180.0, 1e-10, "Vol strain rate = trace");
}

void test_vol_strain_rate_from_vel() {
    Real dNdx[6] = {-1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    Real velocity[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    Real ev_dot = volumetric_strain_rate_from_velocity(dNdx, velocity, 2);
    CHECK_NEAR(ev_dot, 1.0, 1e-10, "Vol strain rate from velocity");
}

void test_stable_dt_no_compression() {
    BulkViscosity bv;
    Real dt = stable_dt_with_viscosity(0.01, 5000.0, bv, 0.0, 0.9);
    Real expected = 0.9 * 0.01 / (5000.0 * 1.732050808);
    CHECK_NEAR(dt, expected, 1e-12, "Stable dt without compression");
}

void test_stable_dt_with_compression() {
    BulkViscosity bv(0.06, 1.2);
    Real L = 0.01, c = 5000.0, ev_dot = -1000.0;
    Real dt = stable_dt_with_viscosity(L, c, bv, ev_dot, 0.9);
    Real c_eff = c + 1.2 * L * 1000.0;
    Real expected = 0.9 * L / (c_eff * 1.732050808);
    CHECK_NEAR(dt, expected, 1e-12, "Stable dt with compression");
}

void test_energy_cycle() {
    EnergyMonitor em;
    EnergyMonitor::EnergyComponents e0;
    e0.kinetic = 100.0;
    em.initialize(e0);

    EnergyMonitor::EnergyComponents e1;
    e1.kinetic = 50.0;
    e1.internal = 50.0;
    auto f1 = em.record(e1);
    CHECK(!f1.energy_explosion, "Cycle: no explosion step 1");

    EnergyMonitor::EnergyComponents e2;
    e2.kinetic = 10.0;
    e2.internal = 90.0;
    auto f2 = em.record(e2);
    CHECK(!f2.energy_explosion, "Cycle: no explosion step 2");
    CHECK(em.num_records() == 3, "Cycle: 3 records");
}

void test_bv_with_energy_monitor() {
    BulkViscosity bv(0.06, 1.2);
    Real ev_dot = -500.0, rho = 7800.0, c = 5000.0, L = 0.01, V = 1e-6;
    Real q = bv.compute(ev_dot, rho, c, L);
    Real e_rate = BulkViscosity::energy_rate(q, ev_dot, V);
    Real dt = 1e-6;
    Real de_bv = e_rate * dt;

    EnergyMonitor em;
    EnergyMonitor::EnergyComponents e0;
    e0.kinetic = 100.0;
    em.initialize(e0);

    EnergyMonitor::EnergyComponents e1;
    e1.kinetic = 100.0 - de_bv;
    e1.bulk_viscosity = de_bv;
    auto flags = em.record(e1);
    CHECK(!flags.energy_explosion, "BV+EM: no explosion");
}

void test_hg_with_energy_monitor() {
    HourglassControl hg;
    hg.accumulate_energy(5.0);

    EnergyMonitor em;
    EnergyMonitor::EnergyComponents ec;
    ec.kinetic = 100.0;
    ec.internal = 50.0;
    ec.hourglass = hg.total_energy;
    em.initialize(ec);
    auto flags = em.record(ec);
    CHECK_NEAR(em.latest().hourglass_ratio(), 0.1, 1e-10, "HG+EM: ratio = 10%");
}

void test_full_system_config() {
    // Test that all components work together with crash preset
    auto cfg = ExplicitDynamicsConfig::crash_preset();

    // Bulk viscosity computation
    Real q = cfg.bulk_viscosity.compute(-500.0, 7800.0, 5000.0, 0.01);
    CHECK(q > 0.0, "System: BV positive");

    // Hourglass stiffness
    Real k_hg = cfg.hourglass.compute_stiffness(160e9, 80e9);
    CHECK(k_hg > 0.0, "System: HG stiffness positive");

    // Energy monitor
    EnergyMonitor em;
    em.set_tolerance(cfg.energy_tolerance);
    EnergyMonitor::EnergyComponents e0;
    e0.kinetic = 1000.0;
    em.initialize(e0);
    CHECK(em.is_initialized(), "System: EM initialized");

    // Erosion manager
    physics::ElementErosionManager eem(100);
    CHECK(eem.element_active(50), "System: EEM active");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Wave 9: Explicit Dynamics Test Suite ===\n\n";

    // 1. Bulk Viscosity (12 tests)
    std::cout << "--- Bulk Viscosity Tests ---\n";
    test_bulk_viscosity_default();
    test_bulk_viscosity_custom();
    test_bulk_viscosity_zero_in_tension();
    test_bulk_viscosity_zero_at_zero();
    test_bulk_viscosity_positive_compression();
    test_bulk_viscosity_linear_only();
    test_bulk_viscosity_quadratic_only();
    test_bulk_viscosity_combined();
    test_bulk_viscosity_energy_rate();
    test_bulk_viscosity_add_to_stress();
    test_bulk_viscosity_super_linear();

    // 2. Hourglass Control (11 tests)
    std::cout << "\n--- Hourglass Control Tests ---\n";
    test_hourglass_default();
    test_hourglass_custom();
    test_hourglass_fb_stiffness();
    test_hourglass_perturbation_stiffness();
    test_hourglass_combined_stiffness();
    test_hourglass_none_stiffness();
    test_hourglass_energy_computation();
    test_hourglass_accumulation();
    test_hourglass_reset();
    test_hourglass_excessive();

    // 3. Energy Monitor (16 tests)
    std::cout << "\n--- Energy Monitor Tests ---\n";
    test_energy_monitor_default();
    test_energy_monitor_tolerance();
    test_energy_components_default();
    test_energy_components_total();
    test_energy_components_expected();
    test_energy_components_balance();
    test_energy_components_hg_ratio();
    test_energy_monitor_initialize();
    test_energy_monitor_record();
    test_energy_monitor_explosion();
    test_energy_monitor_balance_violation();
    test_energy_monitor_hg_warning();
    test_energy_kinetic_computation();
    test_energy_internal_computation();
    test_energy_external_work();
    test_energy_history();

    // 4. Element Erosion (12 tests)
    std::cout << "\n--- Element Erosion Tests ---\n";
    test_erosion_construction();
    test_erosion_erode_element();
    test_erosion_double_erode();
    test_erosion_state();
    test_erosion_check_plastic_strain();
    test_erosion_check_no_failure();
    test_erosion_vonmises();
    test_erosion_stats();
    test_erosion_combined_criteria();
    test_erosion_already_eroded();
    test_eroded_energy_computation();

    // 5. Configuration (10 tests)
    std::cout << "\n--- Configuration Tests ---\n";
    test_config_default();
    test_config_crash_preset();
    test_config_blast_preset();
    test_config_impact_preset();
    test_config_rayleigh();
    test_config_dt_limits();

    // 6. Integration Tests (10 tests)
    std::cout << "\n--- Integration Tests ---\n";
    test_vol_strain_rate();
    test_vol_strain_rate_from_vel();
    test_stable_dt_no_compression();
    test_stable_dt_with_compression();
    test_energy_cycle();
    test_bv_with_energy_monitor();
    test_hg_with_energy_monitor();
    test_full_system_config();

    // Summary
    std::cout << "\n=== Results ===\n";
    std::cout << "Passed: " << tests_passed << "/" << (tests_passed + tests_failed) << "\n";
    if (tests_failed > 0) {
        std::cout << "FAILED: " << tests_failed << " tests\n";
    } else {
        std::cout << "All tests PASSED!\n";
    }

    return (tests_failed > 0) ? 1 : 0;
}
