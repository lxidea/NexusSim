/**
 * @file realistic_crash_test.cpp
 * @brief Big realistic numerical integration tests combining multiple subsystems
 *
 * These tests exercise the full NexusSim physics pipeline in crash-simulation-like
 * scenarios with realistic material parameters, multi-step time integration,
 * and conservation law verification.
 *
 * Tests:
 *  1. Frontal crash: Steel block hitting rigid wall at 15 m/s (54 km/h)
 *  2. Foam absorber: Crushable foam between rigid plates with load-curve compression
 *  3. Spot-welded assembly: Two steel plates with tied contact + progressive failure
 *  4. Drop test: Object in free-fall under gravity bouncing off rigid ground
 *  5. Multi-material crash column: Steel tube + foam fill hitting moving wall
 *  6. Blast loading with JWL EOS: Detonation products expanding in a tube
 *  7. Composite plate impact: Layered composite with ply stress analysis
 *  8. Sensor-monitored crash: Full sensor/control system monitoring a wall impact
 *  9. Rigid body spinning gyroscope: Angular momentum conservation
 * 10. Moving wall crush: Prescribed velocity wall compressing material
 * 11. ALE mesh quality recovery on distorted mesh
 * 12. Constrained assembly (RBE2 + Loads)
 * 13. Material stress-strain curves (multiple materials)
 * 14. Element erosion under combined loading
 */

#include <nexussim/fem/rigid_body.hpp>
#include <nexussim/fem/rigid_wall.hpp>
#include <nexussim/fem/tied_contact.hpp>
#include <nexussim/fem/loads.hpp>
#include <nexussim/fem/load_curve.hpp>
#include <nexussim/fem/constraints.hpp>
#include <nexussim/fem/sensor.hpp>
#include <nexussim/fem/controls.hpp>
#include <nexussim/fem/contact.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/physics/material_models.hpp>
#include <nexussim/physics/eos.hpp>
#include <nexussim/physics/composite_layup.hpp>
#include <nexussim/physics/ale_solver.hpp>
#include <nexussim/physics/element_erosion.hpp>
#include <nexussim/io/time_history.hpp>

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>

using namespace nxs;
using namespace nxs::fem;
using namespace nxs::physics;

static int tests_passed = 0;
static int tests_failed = 0;

static bool near(Real a, Real b, Real tol = 1.0e-6) {
    return std::fabs(a - b) < tol * (1.0 + std::fabs(a) + std::fabs(b));
}

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; std::cout << "  FAIL: " << msg << "\n"; } \
} while(0)

// ============================================================================
// Helper: Compute kinetic energy from velocities and masses
// ============================================================================
static Real compute_KE(const std::vector<Real>& vel, const std::vector<Real>& mass,
                       std::size_t num_nodes) {
    Real KE = 0.0;
    for (std::size_t i = 0; i < num_nodes; ++i) {
        Real vx = vel[3*i+0], vy = vel[3*i+1], vz = vel[3*i+2];
        KE += 0.5 * mass[i] * (vx*vx + vy*vy + vz*vz);
    }
    return KE;
}

// ============================================================================
// Helper: Compute momentum
// ============================================================================
static void compute_momentum(const std::vector<Real>& vel, const std::vector<Real>& mass,
                              std::size_t num_nodes, Real p[3]) {
    p[0] = p[1] = p[2] = 0.0;
    for (std::size_t i = 0; i < num_nodes; ++i) {
        p[0] += mass[i] * vel[3*i+0];
        p[1] += mass[i] * vel[3*i+1];
        p[2] += mass[i] * vel[3*i+2];
    }
}

// ============================================================================
// Test 1: Frontal crash — Steel block hitting rigid wall at 15 m/s
// ============================================================================
void test_frontal_crash() {
    std::cout << "\n=== Test 1: Frontal Crash (15 m/s into rigid wall) ===\n";

    const std::size_t num_nodes = 20;
    const Real node_mass = 0.5;

    std::vector<Real> positions(3 * num_nodes);
    std::vector<Real> velocities(3 * num_nodes, 0.0);
    std::vector<Real> forces(3 * num_nodes, 0.0);
    std::vector<Real> masses(num_nodes, node_mass);

    int idx = 0;
    for (int ix = 0; ix < 5; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 0; iz < 2; ++iz) {
                positions[3*idx+0] = 0.05 + ix * 0.02;
                positions[3*idx+1] = iy * 0.02;
                positions[3*idx+2] = iz * 0.02;
                velocities[3*idx+0] = -15.0;
                idx++;
            }

    Real total_mass = node_mass * num_nodes;
    Real initial_KE = compute_KE(velocities, masses, num_nodes);

    std::cout << "  Total mass: " << total_mass << " kg\n";
    std::cout << "  Initial KE: " << initial_KE << " J\n";

    RigidWallContact walls;
    walls.set_penalty_stiffness(1.0e9);
    auto& wall = walls.add_wall(RigidWallType::Planar, 1);
    wall.name = "barrier";
    wall.normal[0] = 1.0; wall.normal[1] = 0.0; wall.normal[2] = 0.0;
    wall.friction = 0.3;

    const Real dt = 1.0e-6;
    const int num_steps = 10000;
    Real time = 0.0;
    Real max_wall_force = 0.0;
    Real max_penetration = 0.0;
    int contact_start_step = -1;

    for (int step = 0; step < num_steps; ++step) {
        std::fill(forces.begin(), forces.end(), 0.0);

        walls.compute_forces(num_nodes, positions.data(), velocities.data(),
                             masses.data(), forces.data(), dt);

        auto stats = walls.get_stats();
        if (stats.total_normal_force > max_wall_force)
            max_wall_force = stats.total_normal_force;
        if (stats.max_penetration > max_penetration)
            max_penetration = stats.max_penetration;
        if (stats.active_contacts > 0 && contact_start_step < 0)
            contact_start_step = step;

        for (std::size_t i = 0; i < num_nodes; ++i) {
            Real inv_m = 1.0 / masses[i];
            for (int d = 0; d < 3; ++d) {
                velocities[3*i+d] += forces[3*i+d] * inv_m * dt;
                positions[3*i+d] += velocities[3*i+d] * dt;
            }
        }
        time += dt;
    }

    Real final_KE = compute_KE(velocities, masses, num_nodes);
    Real p_final[3];
    compute_momentum(velocities, masses, num_nodes, p_final);

    std::cout << "  Simulation time: " << time * 1000.0 << " ms\n";
    std::cout << "  Contact started at step: " << contact_start_step << "\n";
    std::cout << "  Max wall force: " << max_wall_force / 1000.0 << " kN\n";
    std::cout << "  Max penetration: " << max_penetration * 1000.0 << " mm\n";
    std::cout << "  Final KE: " << final_KE << " J\n";

    CHECK(contact_start_step > 0 && contact_start_step < 5000,
          "Contact detected within first half");
    CHECK(max_wall_force > 1000.0, "Significant wall force (>1 kN)");
    CHECK(max_penetration < 0.05, "Penetration controlled (<50mm)");

    bool no_breakthrough = true;
    for (std::size_t i = 0; i < num_nodes; ++i)
        if (positions[3*i+0] < -0.01) no_breakthrough = false;
    CHECK(no_breakthrough, "No node broke through the wall");

    Real avg_vx = p_final[0] / total_mass;
    CHECK(avg_vx > -15.0, "Block decelerating");
    CHECK(final_KE <= initial_KE * 1.1, "Energy conservation");
}

// ============================================================================
// Test 2: Foam absorber compression with load curve
// ============================================================================
void test_foam_absorber() {
    std::cout << "\n=== Test 2: Foam Absorber Compression ===\n";

    MaterialProperties foam_props;
    foam_props.density = 30.0;
    foam_props.E = 5.0e6;
    foam_props.nu = 0.0;
    foam_props.yield_stress = 0.2e6;
    foam_props.foam_E_crush = 0.3e6;
    foam_props.foam_densification = 0.7;

    CrushableFoamMaterial foam(foam_props);

    MaterialState state;

    LoadCurveManager curves;
    LoadCurve& lc = curves.add_curve(1);
    lc.add_point(0.0, 0.0);
    lc.add_point(0.005, 0.5);
    lc.add_point(0.010, 0.8);
    lc.add_point(0.015, 0.8);

    const int num_steps = 150;
    const Real total_time = 0.015;
    const Real dt_step = total_time / num_steps;

    std::vector<Real> stress_history;
    Real max_stress = 0.0;
    bool densification_reached = false;
    Real densification_stress = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        Real t = step * dt_step;
        Real eps_total = curves.evaluate(1, t);
        Real eps_prev = (step > 0) ? curves.evaluate(1, (step-1)*dt_step) : 0.0;
        Real deps = eps_total - eps_prev;

        state.strain[0] -= deps;
        foam.compute_stress(state);

        Real sigma_xx = std::fabs(state.stress[0]);
        stress_history.push_back(sigma_xx);
        if (sigma_xx > max_stress) max_stress = sigma_xx;

        if (!densification_reached && eps_total > 0.6) {
            densification_stress = sigma_xx;
            densification_reached = true;
        }
    }

    std::cout << "  Foam: 30 kg/m³, yield 200 kPa\n";
    std::cout << "  Max stress: " << max_stress / 1.0e6 << " MPa\n";
    std::cout << "  Densification stress: " << densification_stress / 1.0e6 << " MPa\n";

    CHECK(max_stress > foam_props.yield_stress * 0.5, "Foam reaches significant stress");
    CHECK(densification_reached, "Densification region reached");
    CHECK((int)stress_history.size() == num_steps, "All steps computed");

    int rising = 0;
    for (std::size_t i = 1; i < stress_history.size(); ++i)
        if (stress_history[i] >= stress_history[i-1] - 1.0) rising++;
    CHECK(rising > num_steps / 2, "Stress generally rises with compression");
}

// ============================================================================
// Test 3: Spot-welded assembly with progressive failure
// ============================================================================
void test_spot_weld_failure() {
    std::cout << "\n=== Test 3: Spot-Welded Assembly Failure ===\n";

    const std::size_t num_nodes = 20;
    std::vector<Real> positions(3 * num_nodes, 0.0);
    std::vector<Real> velocities(3 * num_nodes, 0.0);
    std::vector<Real> forces(3 * num_nodes, 0.0);

    for (int i = 0; i < 10; ++i) {
        positions[3*i+0] = (i % 5) * 0.02;
        positions[3*i+1] = 0.0;
    }
    for (int i = 10; i < 20; ++i) {
        positions[3*i+0] = ((i-10) % 5) * 0.02;
        positions[3*i+1] = 0.001;
    }

    TiedContact tc;
    TiedContactConfig cfg;
    cfg.type = TiedContactType::TiedWithFailure;
    cfg.penalty_stiffness = 1.0e9;
    cfg.failure_force = 5000.0;
    tc.set_config(cfg);

    for (int i = 0; i < 5; ++i) {
        auto& pair = tc.add_pair();
        pair.slave_node = i;
        pair.master_nodes[0] = i + 10;
        pair.num_master_nodes = 1;
        pair.phi[0] = 1.0;
        // Set initial gap so penalty is zero at start
        pair.gap_initial[0] = positions[3*i+0] - positions[3*(i+10)+0];
        pair.gap_initial[1] = positions[3*i+1] - positions[3*(i+10)+1];
        pair.gap_initial[2] = positions[3*i+2] - positions[3*(i+10)+2];
    }

    CHECK(tc.num_pairs() == 5, "5 spot welds created");
    CHECK(tc.num_active_pairs() == 5, "All active initially");

    const int num_steps = 2000;
    const Real dt = 1.0e-5;
    int first_failure_step = -1;
    int all_failed_step = -1;

    for (int step = 0; step < num_steps; ++step) {
        std::fill(forces.begin(), forces.end(), 0.0);

        Real pull_force = 200.0 * step;
        for (int i = 10; i < 20; ++i)
            forces[3*i+1] = pull_force;

        tc.apply_tied_constraints(positions.data(), velocities.data(),
                                  forces.data(), dt);

        auto stats = tc.get_stats();
        if (stats.failed_pairs > 0 && first_failure_step < 0) {
            first_failure_step = step;
            std::cout << "  First weld failure at step " << step << "\n";
        }
        if (stats.active_pairs == 0 && all_failed_step < 0) {
            all_failed_step = step;
            std::cout << "  All welds failed at step " << step << "\n";
        }

        for (int i = 10; i < 20; ++i) {
            velocities[3*i+1] += forces[3*i+1] / 1.0 * dt;
            positions[3*i+1] += velocities[3*i+1] * dt;
        }

        if (all_failed_step > 0) break;
    }

    CHECK(first_failure_step > 0, "Spot welds survived initial loading");
    CHECK(first_failure_step < num_steps, "Eventually some welds failed");
    CHECK(tc.get_stats().failed_pairs > 0, "At least one weld failed");
}

// ============================================================================
// Test 4: Drop test — gravity + rigid ground bounce
// ============================================================================
void test_drop_test() {
    std::cout << "\n=== Test 4: Drop Test (1m height, gravity) ===\n";

    const std::size_t num_nodes = 8;
    const Real node_mass = 0.125;

    std::vector<Real> positions(3 * num_nodes);
    std::vector<Real> velocities(3 * num_nodes, 0.0);
    std::vector<Real> forces(3 * num_nodes, 0.0);
    std::vector<Real> masses(num_nodes, node_mass);

    int idx = 0;
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 0; iz < 2; ++iz) {
                positions[3*idx+0] = ix * 0.01;
                positions[3*idx+1] = iy * 0.01;
                positions[3*idx+2] = 1.0 + iz * 0.01;
                idx++;
            }

    RigidWallContact walls;
    walls.set_penalty_stiffness(1.0e8);
    auto& ground = walls.add_wall(RigidWallType::Planar, 1);
    ground.normal[0] = 0.0; ground.normal[1] = 0.0; ground.normal[2] = 1.0;

    LoadCurveManager curves;
    LoadManager loads;
    loads.set_curve_manager(&curves);
    auto& grav = loads.add_load(LoadType::Gravity, 1);
    grav.magnitude = 9.81;
    grav.direction[0] = 0.0; grav.direction[1] = 0.0; grav.direction[2] = -1.0;

    const Real dt = 1.0e-5;
    const int max_steps = 100000;
    Real time = 0.0;
    bool hit_ground = false;
    bool bounced = false;
    Real vz_at_impact = 0.0;
    Real min_z = 1.0;

    for (int step = 0; step < max_steps; ++step) {
        std::fill(forces.begin(), forces.end(), 0.0);

        loads.apply_loads(time, num_nodes, positions.data(), velocities.data(),
                          forces.data(), masses.data(), dt);
        walls.compute_forces(num_nodes, positions.data(), velocities.data(),
                             masses.data(), forces.data(), dt);

        auto stats = walls.get_stats();
        if (stats.active_contacts > 0 && !hit_ground) {
            hit_ground = true;
            vz_at_impact = velocities[2];
            std::cout << "  Impact at t=" << time * 1000.0 << " ms, Vz="
                      << vz_at_impact << " m/s\n";
        }
        if (hit_ground && stats.active_contacts == 0) bounced = true;

        for (std::size_t i = 0; i < num_nodes; ++i) {
            for (int d = 0; d < 3; ++d) {
                velocities[3*i+d] += forces[3*i+d] / masses[i] * dt;
                positions[3*i+d] += velocities[3*i+d] * dt;
            }
        }
        for (std::size_t i = 0; i < num_nodes; ++i)
            if (positions[3*i+2] < min_z) min_z = positions[3*i+2];

        time += dt;
        if (bounced && time > 0.5) break;
    }

    Real expected_v = std::sqrt(2.0 * 9.81 * 1.0);
    std::cout << "  Expected impact velocity: " << expected_v << " m/s\n";
    std::cout << "  Actual impact Vz: " << vz_at_impact << " m/s\n";
    std::cout << "  Min z reached: " << min_z * 1000.0 << " mm\n";

    CHECK(hit_ground, "Object hit the ground");
    CHECK(std::fabs(vz_at_impact) > 3.0, "Impact velocity near expected (> 3 m/s)");
    CHECK(min_z > -0.01, "No significant ground penetration");
}

// ============================================================================
// Test 5: Multi-material crash column (steel tube + foam fill)
// ============================================================================
void test_multi_material_column() {
    std::cout << "\n=== Test 5: Multi-Material Crash Column ===\n";

    const std::size_t num_foam_nodes = 10;
    const std::size_t num_steel_nodes = 4;
    const std::size_t num_nodes = num_foam_nodes + num_steel_nodes;
    const Real foam_node_mass = 0.003;
    const Real steel_node_mass = 0.25;

    std::vector<Real> positions(3 * num_nodes, 0.0);
    std::vector<Real> velocities(3 * num_nodes, 0.0);
    std::vector<Real> forces(3 * num_nodes, 0.0);
    std::vector<Real> masses(num_nodes);

    for (std::size_t i = 0; i < num_foam_nodes; ++i) {
        positions[3*i+0] = 0.05 + i * 0.01;
        masses[i] = foam_node_mass;
        velocities[3*i+0] = -10.0;
    }
    for (std::size_t i = 0; i < num_steel_nodes; ++i) {
        std::size_t n = num_foam_nodes + i;
        positions[3*n+0] = 0.15 + (i % 2) * 0.01;
        positions[3*n+1] = (i / 2) * 0.01;
        masses[n] = steel_node_mass;
        velocities[3*n+0] = -10.0;
    }

    RigidBody steel_tube(1, "SteelTube");
    std::vector<Index> steel_ids;
    for (std::size_t i = num_foam_nodes; i < num_nodes; ++i) steel_ids.push_back(i);
    steel_tube.set_slave_nodes(steel_ids);
    steel_tube.compute_properties(positions.data(), masses.data());
    steel_tube.velocity()[0] = -10.0;

    RigidWallContact walls;
    walls.set_penalty_stiffness(1.0e8);
    auto& wall = walls.add_wall(RigidWallType::Planar, 1);
    wall.normal[0] = 1.0;

    Real total_mass = 0.0;
    for (auto m : masses) total_mass += m;
    Real initial_KE = 0.5 * total_mass * 100.0;
    std::cout << "  Total mass: " << total_mass * 1000.0 << " g, Initial KE: " << initial_KE << " J\n";

    const Real dt = 1.0e-6;
    const int num_steps = 20000;
    Real max_wall_force = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        std::fill(forces.begin(), forces.end(), 0.0);
        walls.compute_forces(num_nodes, positions.data(), velocities.data(),
                             masses.data(), forces.data(), dt);
        auto stats = walls.get_stats();
        if (stats.total_normal_force > max_wall_force) max_wall_force = stats.total_normal_force;

        for (std::size_t i = 0; i < num_foam_nodes; ++i)
            for (int d = 0; d < 3; ++d) {
                velocities[3*i+d] += forces[3*i+d] / masses[i] * dt;
                positions[3*i+d] += velocities[3*i+d] * dt;
            }

        steel_tube.gather_forces(positions.data(), forces.data());
        steel_tube.update(dt);
        steel_tube.scatter_to_nodes(positions.data(), velocities.data());
        for (auto n : steel_ids)
            for (int d = 0; d < 3; ++d)
                positions[3*n+d] += velocities[3*n+d] * dt;
    }

    Real final_KE = compute_KE(velocities, masses, num_nodes);
    std::cout << "  Max wall force: " << max_wall_force / 1000.0 << " kN\n";
    std::cout << "  Final KE: " << final_KE << " J, Steel Vx: " << steel_tube.velocity()[0] << " m/s\n";

    CHECK(max_wall_force > 100.0, "Significant wall force during foam crush");
    // Allow small energy gain from penalty contact (< 5%)
    CHECK(final_KE < initial_KE * 1.05, "Energy roughly conserved during crash");
    CHECK(steel_tube.velocity()[0] > -10.0, "Steel tube decelerated");
}

// ============================================================================
// Test 6: Blast loading with JWL EOS
// ============================================================================
void test_blast_loading() {
    std::cout << "\n=== Test 6: Blast Loading (JWL EOS) ===\n";

    EOSProperties jwl;
    jwl.type = EOSType::JWL;
    jwl.rho0 = 1630.0;
    jwl.A_jwl = 3.712e11;
    jwl.B_jwl = 3.231e9;
    jwl.R1 = 4.15;
    jwl.R2 = 0.95;
    jwl.omega = 0.30;

    Real e0 = 7.0e6;
    const int ncells = 10;
    std::vector<Real> density(ncells, jwl.rho0);
    std::vector<Real> energy(ncells, 0.0);
    std::vector<Real> pressure(ncells, 0.0);
    std::vector<Real> velocity_cell(ncells + 1, 0.0);

    for (int i = 0; i < 3; ++i) energy[i] = e0;

    Real cell_length = 0.01;
    const Real dt_blast = 1.0e-7;
    const int blast_steps = 1000;
    Real max_pressure = 0.0;
    Real max_velocity = 0.0;

    for (int step = 0; step < blast_steps; ++step) {
        for (int i = 0; i < ncells; ++i) {
            pressure[i] = EquationOfState::compute_pressure(jwl, density[i], energy[i]);
            if (pressure[i] > max_pressure) max_pressure = pressure[i];
        }
        for (int i = 1; i < ncells; ++i) {
            Real dp = pressure[i] - pressure[i-1];
            Real rho_avg = 0.5 * (density[i] + density[i-1]);
            velocity_cell[i] += -dt_blast * dp / (rho_avg * cell_length);
            if (std::fabs(velocity_cell[i]) > max_velocity)
                max_velocity = std::fabs(velocity_cell[i]);
        }
        for (int i = 0; i < ncells; ++i) {
            Real dv = velocity_cell[i+1] - velocity_cell[i];
            density[i] -= density[i] * dv / cell_length * dt_blast;
            if (density[i] < 1.0) density[i] = 1.0;
        }
        for (int i = 0; i < ncells; ++i) {
            Real dv = velocity_cell[i+1] - velocity_cell[i];
            if (density[i] > 1.0) {
                energy[i] -= pressure[i] / density[i] * dv / cell_length * dt_blast;
                if (energy[i] < 0.0) energy[i] = 0.0;
            }
        }
    }

    EOSProperties cu;
    cu.type = EOSType::Gruneisen;
    cu.rho0 = 8930.0; cu.C0 = 3940.0; cu.S1 = 1.489;
    cu.gamma0 = 2.02; cu.a_coeff = 0.47;
    Real p_cu = EquationOfState::compute_pressure(cu, 8930.0 * 1.10, 0.0);
    Real c_cu = EquationOfState::sound_speed(cu, 8930.0, 0.0);

    std::cout << "  JWL max pressure: " << max_pressure / 1.0e9 << " GPa\n";
    std::cout << "  JWL max velocity: " << max_velocity << " m/s\n";
    std::cout << "  Copper 10% compress: " << p_cu / 1.0e9 << " GPa\n";
    std::cout << "  Copper sound speed: " << c_cu << " m/s\n";

    CHECK(max_pressure > 1.0e9, "Blast pressure > 1 GPa");
    CHECK(max_velocity > 100.0, "Blast wave velocity > 100 m/s");
    CHECK(p_cu > 1.0e9, "Copper 10% compress > 1 GPa");
    CHECK(near(c_cu, 3940.0, 10.0), "Copper sound speed ~ 3940 m/s");
    // Blast front cells should have expanded (right side of charge)
    bool any_expanded = false;
    for (int i = 0; i < ncells; ++i)
        if (density[i] < jwl.rho0 * 0.99) any_expanded = true;
    CHECK(any_expanded, "Some cells expanded from blast");
}

// ============================================================================
// Test 7: Composite plate impact with ply stress analysis
// ============================================================================
void test_composite_impact() {
    std::cout << "\n=== Test 7: Composite Plate Impact ===\n";

    auto layup = layup_presets::quasi_isotropic(
        200.0e9, 10.0e9, 5.0e9, 0.3, 0.125e-3);
    layup.compute_abd();

    auto ep = layup.effective_properties();
    std::cout << "  Layup: [0/45/90/-45]s, 8 plies\n";
    std::cout << "  Effective Ex: " << ep.Ex / 1.0e9 << " GPa\n";
    std::cout << "  Effective Ey: " << ep.Ey / 1.0e9 << " GPa\n";

    CHECK(layup.is_symmetric(), "QI layup is symmetric");
    CHECK(layup.num_plies() == 8, "8 plies");

    // Composite strength values for Hashin-style manual check
    Real Xt = 2000.0e6, Yt = 50.0e6, S12 = 100.0e6;

    Real global_strain[3] = {0.0, 0.0, 0.0};
    Real global_curvature[3] = {0.0, 0.0, 0.0};

    const int strain_steps = 200;
    Real max_strain = 0.02;
    int first_failure_step = -1;
    int num_failed_plies = 0;

    std::vector<PlyState> ply_states(layup.num_plies());

    std::vector<bool> ply_failed(layup.num_plies(), false);

    for (int step = 0; step < strain_steps; ++step) {
        Real eps_x = max_strain * step / strain_steps;
        global_strain[0] = eps_x;

        layup.compute_ply_stresses(global_strain, global_curvature, ply_states.data());

        for (int p = 0; p < (int)layup.num_plies(); ++p) {
            if (ply_failed[p]) continue;
            const auto& ps = ply_states[p];

            bool failed = false;
            // Fiber tension failure: (σ1/Xt)² + (τ12/S12)² >= 1
            if (ps.stress_local[0] > 0.0) {
                Real f = (ps.stress_local[0] / Xt) * (ps.stress_local[0] / Xt) +
                         (ps.stress_local[2] / S12) * (ps.stress_local[2] / S12);
                if (f >= 1.0) failed = true;
            }
            // Matrix tension failure
            if (ps.stress_local[1] > 0.0) {
                Real f = (ps.stress_local[1] / Yt) * (ps.stress_local[1] / Yt) +
                         (ps.stress_local[2] / S12) * (ps.stress_local[2] / S12);
                if (f >= 1.0) failed = true;
            }
            if (failed) {
                ply_failed[p] = true;
                if (first_failure_step < 0) first_failure_step = step;
            }
        }
        num_failed_plies = 0;
        for (bool f : ply_failed) if (f) num_failed_plies++;

        if (num_failed_plies > 0 && step == first_failure_step)
            std::cout << "  First ply failure at " << eps_x * 100.0 << "% strain\n";
    }

    std::cout << "  Failed plies at 2%: " << num_failed_plies << "/" << layup.num_plies() << "\n";

    CHECK(std::fabs(ep.Ex - ep.Ey) / ep.Ex < 0.15, "QI laminate roughly isotropic");
    CHECK(ep.Ex > 50.0e9, "Effective modulus > 50 GPa");
    CHECK(first_failure_step > 0, "Survived initial loading");
    CHECK(num_failed_plies > 0, "Some plies failed at 2% strain");
}

// ============================================================================
// Test 8: Sensor-monitored crash with control actions
// ============================================================================
void test_sensor_monitored_crash() {
    std::cout << "\n=== Test 8: Sensor-Monitored Crash ===\n";

    SensorManager sensors;

    SensorConfig accel_cfg;
    accel_cfg.type = SensorType::Accelerometer;
    accel_cfg.id = 1; accel_cfg.name = "Front_Accel";
    accel_cfg.node_id = 0;
    accel_cfg.direction = SensorDirection::X;
    accel_cfg.threshold_value = 500.0;
    accel_cfg.threshold_above = true;
    sensors.add_sensor(accel_cfg);

    SensorConfig vel_cfg;
    vel_cfg.type = SensorType::VelocityGauge;
    vel_cfg.id = 2; vel_cfg.name = "Front_Vel";
    vel_cfg.node_id = 0;
    vel_cfg.direction = SensorDirection::X;
    vel_cfg.threshold_value = 1.0;
    vel_cfg.threshold_above = false;
    sensors.add_sensor(vel_cfg);

    SensorConfig dist_cfg;
    dist_cfg.type = SensorType::DistanceSensor;
    dist_cfg.id = 3; dist_cfg.name = "Crush_Dist";
    dist_cfg.node_id = 0; dist_cfg.node_id2 = 3;
    sensors.add_sensor(dist_cfg);

    ControlManager controls;
    auto& terminate_rule = controls.add_rule(1);
    terminate_rule.name = "TerminateOnDecel";
    terminate_rule.sensor_id = 1;
    terminate_rule.trigger_on_exceed = true;
    terminate_rule.action = ControlActionType::TerminateSimulation;

    bool callback_fired = false;
    auto& cb_rule = controls.add_rule(2);
    cb_rule.sensor_id = 2;
    cb_rule.trigger_on_exceed = false;
    cb_rule.action = ControlActionType::CustomCallback;
    cb_rule.callback = [&callback_fired]() { callback_fired = true; };

    const std::size_t num_nodes = 4;
    std::vector<Real> positions = {0.05,0,0, 0.06,0,0, 0.08,0,0, 0.10,0,0};
    std::vector<Real> velocities = {-10,0,0, -10,0,0, -10,0,0, -10,0,0};
    std::vector<Real> accels(12, 0.0);
    std::vector<Real> forces(12, 0.0);
    std::vector<Real> masses(4, 1.0);

    RigidWallContact walls;
    walls.set_penalty_stiffness(1.0e8);
    walls.add_wall(RigidWallType::Planar).normal[0] = 1.0;

    const Real dt = 1.0e-5;
    const int max_steps = 50000;
    bool terminated = false;
    int terminate_step = -1;

    for (int step = 0; step < max_steps; ++step) {
        std::fill(forces.begin(), forces.end(), 0.0);
        walls.compute_forces(num_nodes, positions.data(), velocities.data(),
                             masses.data(), forces.data(), dt);

        for (std::size_t i = 0; i < num_nodes; ++i)
            for (int d = 0; d < 3; ++d)
                accels[3*i+d] = forces[3*i+d] / masses[i];

        sensors.measure_all(step * dt, dt, num_nodes,
                            positions.data(), velocities.data(),
                            accels.data(), forces.data());
        controls.evaluate(sensors);

        if (controls.should_terminate()) {
            terminated = true;
            terminate_step = step;
        }

        for (std::size_t i = 0; i < num_nodes; ++i)
            for (int d = 0; d < 3; ++d) {
                velocities[3*i+d] += accels[3*i+d] * dt;
                positions[3*i+d] += velocities[3*i+d] * dt;
            }

        if (terminated) break;
    }

    std::cout << "  Terminated: " << (terminated ? "YES" : "NO")
              << " at step " << terminate_step << "\n";
    std::cout << "  Accel readings: " << sensors.find(1)->num_readings() << "\n";
    std::cout << "  Callback fired: " << (callback_fired ? "YES" : "NO") << "\n";

    CHECK(sensors.find(1)->num_readings() > 10, "Accelerometer recorded data");
    CHECK(sensors.find(1)->threshold_triggered(), "High-g threshold triggered");
    CHECK(terminated, "Simulation terminated by sensor control");
    CHECK(terminate_step > 0, "Termination at reasonable step");
}

// ============================================================================
// Test 9: Spinning gyroscope (angular momentum conservation)
// ============================================================================
void test_gyroscope() {
    std::cout << "\n=== Test 9: Spinning Gyroscope ===\n";

    const std::size_t num_nodes = 4;
    std::vector<Real> positions = {0.1,0,0, -0.1,0,0, 0,0.1,0, 0,-0.1,0};
    std::vector<Real> masses = {1.0, 1.0, 1.0, 1.0};
    std::vector<Real> velocities(12, 0.0);

    RigidBody gyro(1, "Gyroscope");
    gyro.set_slave_nodes({0, 1, 2, 3});
    gyro.compute_properties(positions.data(), masses.data());

    Real omega_z = 100.0;
    gyro.angular_velocity()[2] = omega_z;
    Real Izz = gyro.properties().inertia[2];
    Real L0 = Izz * omega_z;
    Real KE0 = 0.5 * Izz * omega_z * omega_z;

    const Real dt = 1.0e-4;
    const int num_steps = 10000;

    for (int step = 0; step < num_steps; ++step) {
        gyro.update(dt);
        gyro.scatter_to_nodes(positions.data(), velocities.data());
        for (std::size_t i = 0; i < num_nodes; ++i)
            for (int d = 0; d < 3; ++d)
                positions[3*i+d] += velocities[3*i+d] * dt;
    }

    Real omega_final = gyro.angular_velocity()[2];
    Real L_final = Izz * omega_final;
    Real KE_final = 0.5 * Izz * omega_final * omega_final;

    std::cout << "  omega: " << omega_z << " -> " << omega_final << " rad/s\n";
    std::cout << "  L: " << L0 << " -> " << L_final << " kg*m²/s\n";
    std::cout << "  Quaternion norm: " << gyro.orientation().norm() << "\n";

    CHECK(near(omega_final, omega_z, omega_z * 0.01), "Angular velocity conserved (1%)");
    CHECK(near(L_final, L0, L0 * 0.01), "Angular momentum conserved (1%)");
    CHECK(near(gyro.orientation().norm(), 1.0, 1e-6), "Quaternion unit norm");
    CHECK(near(KE_final, KE0, KE0 * 0.01), "Rotational KE conserved (1%)");
}

// ============================================================================
// Test 10: Moving wall compression with time history
// ============================================================================
void test_moving_wall_compression() {
    std::cout << "\n=== Test 10: Moving Wall Compression ===\n";

    const std::size_t num_nodes = 8;
    std::vector<Real> positions(3 * num_nodes, 0.0);
    std::vector<Real> velocities(3 * num_nodes, 0.0);
    std::vector<Real> forces(3 * num_nodes, 0.0);
    std::vector<Real> masses(num_nodes, 0.5);

    for (std::size_t i = 0; i < num_nodes; ++i)
        positions[3*i+0] = 0.01 + i * 0.01;

    RigidWallContact walls;
    walls.set_penalty_stiffness(1.0e8);
    auto& fixed_wall = walls.add_wall(RigidWallType::Planar, 1);
    fixed_wall.normal[0] = 1.0;

    auto& moving_wall = walls.add_wall(RigidWallType::Moving, 2);
    moving_wall.origin[0] = 0.10;
    moving_wall.normal[0] = -1.0;
    moving_wall.velocity[0] = -2.0;

    io::TimeHistoryRecorder history;
    history.add_nodal_probe("front_node", {Index(0)}, io::NodalQuantity::VelocityX);
    history.add_nodal_probe("rear_node", {Index(num_nodes-1)}, io::NodalQuantity::VelocityX);

    const Real dt = 1.0e-5;
    const int num_steps = 20000;
    Real time = 0.0;
    Real max_force = 0.0;
    Real initial_length = positions[3*(num_nodes-1)] - positions[0];

    for (int step = 0; step < num_steps; ++step) {
        std::fill(forces.begin(), forces.end(), 0.0);
        walls.update_walls(dt);
        walls.compute_forces(num_nodes, positions.data(), velocities.data(),
                             masses.data(), forces.data(), dt);

        Real total_f = 0.0;
        for (std::size_t i = 0; i < num_nodes; ++i)
            total_f += std::fabs(forces[3*i+0]);
        if (total_f > max_force) max_force = total_f;

        history.record(time, num_nodes, nullptr, velocities.data(), nullptr, forces.data());

        for (std::size_t i = 0; i < num_nodes; ++i) {
            velocities[3*i+0] += forces[3*i+0] / masses[i] * dt;
            positions[3*i+0] += velocities[3*i+0] * dt;
        }
        time += dt;
    }

    Real final_length = positions[3*(num_nodes-1)] - positions[0];
    Real compression = 1.0 - final_length / initial_length;

    std::cout << "  Compression: " << compression * 100.0 << "%\n";
    std::cout << "  Max force: " << max_force / 1000.0 << " kN\n";
    std::cout << "  Time history readings: " << history.num_records() << "\n";

    CHECK(max_force > 1000.0, "Significant compression force");
    CHECK(compression > 0.1, "Material compressed > 10%");
    CHECK(history.num_records() > 1000, "Time history captured");
}

// ============================================================================
// Test 11: ALE mesh quality recovery
// ============================================================================
void test_ale_mesh_quality() {
    std::cout << "\n=== Test 11: ALE Mesh Quality Recovery ===\n";

    const std::size_t num_nodes = 9;
    std::vector<Real> coords = {
        0.0,0.0,0.0, 0.5,0.0,0.0, 1.0,0.0,0.0,
        0.0,0.5,0.0, 0.8,0.8,0.0, 1.0,0.5,0.0,  // node 4 distorted
        0.0,1.0,0.0, 0.5,1.0,0.0, 1.0,1.0,0.0
    };
    std::vector<Index> conn = {0,1,4,3, 1,2,5,4, 3,4,7,6, 4,5,8,7};
    std::set<Index> boundary = {0,1,2,3,5,6,7,8};

    ALESolver ale;
    ALEConfig cfg;
    cfg.smoothing = SmoothingMethod::Laplacian;
    cfg.smoothing_weight = 0.8;
    cfg.smoothing_iterations = 20;
    cfg.boundary_fixed = true;
    ale.set_config(cfg);
    ale.initialize(num_nodes, 4, conn.data(), 4, boundary);

    Real q_initial = ale.average_quality(coords.data(), conn.data(), 4, 4);
    ale.smooth(coords.data());
    Real q_final = ale.average_quality(coords.data(), conn.data(), 4, 4);

    std::cout << "  Quality: " << q_initial << " -> " << q_final << "\n";
    std::cout << "  Node 4: (" << coords[12] << ", " << coords[13] << ")\n";

    CHECK(q_final >= q_initial, "Quality improved");
    CHECK(near(coords[12], 0.5, 0.05), "Node 4 x ~ 0.5");
    CHECK(near(coords[13], 0.5, 0.05), "Node 4 y ~ 0.5");
}

// ============================================================================
// Test 12: Constrained assembly (RBE2 + Loads)
// ============================================================================
void test_constrained_assembly() {
    std::cout << "\n=== Test 12: Constrained Assembly ===\n";

    const std::size_t num_nodes = 5;
    std::vector<Real> positions = {0,0,0, 0.1,0,0, -0.1,0,0, 0,0.1,0, 0,-0.1,0};
    std::vector<Real> velocities(15, 0.0);
    std::vector<Real> accels(15, 0.0);
    std::vector<Real> forces(15, 0.0);
    std::vector<Real> masses(5, 1.0);

    ConstraintManager constraints;
    auto& rbe2 = constraints.add_constraint(ConstraintType::RBE2, 1);
    rbe2.master_node = 0;
    rbe2.slave_nodes = {1, 2, 3, 4};
    for (int i = 0; i < 6; ++i) rbe2.tied_dofs[i] = true;

    LoadCurveManager curves;
    LoadCurve& lc = curves.add_curve(1);
    lc.add_point(0.0, 0.0); lc.add_point(0.01, 1.0); lc.add_point(0.1, 1.0);

    LoadManager loads;
    loads.set_curve_manager(&curves);
    auto& force_load = loads.add_load(LoadType::NodalForce, 1);
    force_load.magnitude = 100.0;
    force_load.direction[0] = 0.0; force_load.direction[1] = 0.0; force_load.direction[2] = 1.0;
    force_load.node_set = {0, 1, 2, 3, 4};
    force_load.dof = -1;

    const Real dt = 1.0e-4;
    const int num_steps = 1000;

    for (int step = 0; step < num_steps; ++step) {
        std::fill(forces.begin(), forces.end(), 0.0);
        loads.apply_loads(step * dt, num_nodes, positions.data(), velocities.data(),
                          forces.data(), masses.data(), dt);
        for (std::size_t i = 0; i < num_nodes; ++i)
            for (int d = 0; d < 3; ++d) {
                accels[3*i+d] = forces[3*i+d] / masses[i];
                velocities[3*i+d] += accels[3*i+d] * dt;
                positions[3*i+d] += velocities[3*i+d] * dt;
            }
        constraints.apply_constraints(positions.data(), velocities.data(), accels.data(), dt);
    }

    Real master_vz = velocities[2];
    std::cout << "  Master Vz: " << master_vz << " m/s\n";
    std::cout << "  Slave Vz: " << velocities[5] << ", " << velocities[8] << ", "
              << velocities[11] << ", " << velocities[14] << "\n";

    Real tol = std::fabs(master_vz) * 0.05 + 0.01;
    CHECK(near(velocities[5], master_vz, tol), "Slave 1 Vz ~ master");
    CHECK(near(velocities[8], master_vz, tol), "Slave 2 Vz ~ master");
    CHECK(near(velocities[11], master_vz, tol), "Slave 3 Vz ~ master");
    CHECK(near(velocities[14], master_vz, tol), "Slave 4 Vz ~ master");
    CHECK(positions[2] > 0.0, "Assembly moved in +z");
}

// ============================================================================
// Test 13: Material stress-strain curves (multiple materials)
// ============================================================================
void test_material_stress_strain() {
    std::cout << "\n=== Test 13: Material Stress-Strain Curves ===\n";

    // Steel: elastic-plastic (J2)
    {
        MaterialProperties steel;
        steel.density = 7850.0;
        steel.E = 210.0e9;
        steel.nu = 0.3;
        steel.yield_stress = 350.0e6;
        steel.hardening_modulus = 1.0e9;
        steel.compute_derived();  // Compute G, K from E, nu

        VonMisesPlasticMaterial mat(steel);
        MaterialState state;

        Real max_stress = 0.0;
        const int steps = 100;
        Real deps = 0.01 / steps;
        Real nu_steel = steel.nu;

        for (int i = 0; i < steps; ++i) {
            Real eps = deps * (i + 1);
            state.strain[0] = eps;
            state.strain[1] = -nu_steel * eps;  // Poisson contraction
            state.strain[2] = -nu_steel * eps;
            mat.compute_stress(state);
            if (std::fabs(state.stress[0]) > max_stress) max_stress = std::fabs(state.stress[0]);
        }

        std::cout << "  Steel max stress: " << max_stress / 1.0e6 << " MPa\n";
        CHECK(max_stress > 300.0e6, "Steel: stress > 300 MPa");
        CHECK(std::isfinite(max_stress), "Steel: finite stress");
    }

    // Aluminum: Cowper-Symonds
    {
        MaterialProperties al;
        al.density = 2700.0;
        al.E = 70.0e9;
        al.nu = 0.33;
        al.yield_stress = 275.0e6;
        al.hardening_modulus = 500.0e6;
        al.CS_D = 6500.0;
        al.CS_q = 4.0;
        al.compute_derived();  // Compute G, K from E, nu

        CowperSymondsMaterial mat(al);
        MaterialState state;
        state.effective_strain_rate = 100.0;  // High strain rate

        Real max_stress = 0.0;
        const int steps = 100;
        Real deps = 0.01 / steps;
        Real nu_al = al.nu;

        for (int i = 0; i < steps; ++i) {
            Real eps = deps * (i + 1);
            state.strain[0] = eps;
            state.strain[1] = -nu_al * eps;  // Poisson contraction
            state.strain[2] = -nu_al * eps;
            mat.compute_stress(state);
            if (std::fabs(state.stress[0]) > max_stress) max_stress = std::fabs(state.stress[0]);
        }

        std::cout << "  Aluminum CS max stress: " << max_stress / 1.0e6 << " MPa\n";
        CHECK(max_stress > 250.0e6, "Al: stress > 250 MPa");
        CHECK(std::isfinite(max_stress), "Al: finite stress");
    }

    // Rubber: Mooney-Rivlin
    {
        MaterialProperties rubber;
        rubber.density = 1100.0;
        rubber.E = 3.0e6;
        rubber.nu = 0.499;
        rubber.C10 = 0.5e6;
        rubber.C01 = 0.1e6;

        MooneyRivlinMaterial mat(rubber);
        MaterialState state;

        Real max_stress = 0.0;
        const int steps = 50;
        Real deps = 0.5 / steps;

        for (int i = 0; i < steps; ++i) {
            state.strain[0] += deps;
            state.F[0] = 1.0 + state.strain[0];  // Update deformation gradient
            mat.compute_stress(state);
            if (std::fabs(state.stress[0]) > max_stress) max_stress = std::fabs(state.stress[0]);
        }

        std::cout << "  Rubber MR max stress: " << max_stress / 1.0e6 << " MPa\n";
        CHECK(max_stress > 0.0, "Rubber: non-zero stress");
        CHECK(std::isfinite(max_stress), "Rubber: finite stress");
    }
}

// ============================================================================
// Test 14: Element erosion under combined loading
// ============================================================================
void test_erosion_combined() {
    std::cout << "\n=== Test 14: Element Erosion ===\n";

    const std::size_t num_elements = 10;
    ElementErosionManager erosion(num_elements);

    FailureParameters params;
    params.criterion = FailureCriterion::EffectivePlasticStrain;
    params.max_plastic_strain = 0.3;
    params.delete_on_failure = true;
    params.redistribute_mass = true;
    erosion.set_failure_parameters(params);

    for (std::size_t e = 0; e < num_elements; ++e) {
        MaterialState state;
        Real eps_p = 0.05 * (e + 1);
        state.plastic_strain = eps_p;  // Erosion checks this field
        state.history[0] = eps_p;
        state.stress[0] = 500.0e6;
        state.stress[1] = 200.0e6;
        state.stress[2] = 100.0e6;
        erosion.check_failure(e, state);
    }

    auto stats = erosion.get_stats();
    std::cout << "  Total: " << stats.total_elements << "\n";
    std::cout << "  Active: " << stats.active_elements << "\n";
    std::cout << "  Eroded: " << erosion.eroded_count() << "\n";

    CHECK(erosion.eroded_count() > 0, "Some elements eroded");
    CHECK(stats.active_elements + (int)erosion.eroded_count() <= (int)num_elements,
          "Active + eroded <= total");
    // Element 0 (5% strain) should survive
    CHECK(erosion.element_active(0), "Element 0 (5%) survived");
    CHECK(erosion.element_active(1), "Element 1 (10%) survived");
    // Last element (50% strain) should be eroded
    CHECK(erosion.element_state(num_elements-1) == ElementState::Eroded,
          "Last element (50%) eroded");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << std::setprecision(6);
    std::cout << "================================================================\n";
    std::cout << "NexusSim Realistic Integration Tests\n";
    std::cout << "================================================================\n";

    test_frontal_crash();
    test_foam_absorber();
    test_spot_weld_failure();
    test_drop_test();
    test_multi_material_column();
    test_blast_loading();
    test_composite_impact();
    test_sensor_monitored_crash();
    test_gyroscope();
    test_moving_wall_compression();
    test_ale_mesh_quality();
    test_constrained_assembly();
    test_material_stress_strain();
    test_erosion_combined();

    std::cout << "\n================================================================\n";
    std::cout << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " total\n";
    std::cout << "================================================================\n";

    return tests_failed > 0 ? 1 : 0;
}
