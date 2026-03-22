/**
 * @file sph_wave19_test.cpp
 * @brief Wave 19: SPH Enrichment Test Suite (7 features, ~40 tests)
 *
 * Tests the 7 SPH enrichment classes from sph_wave19.hpp:
 *   1. TensileInstabilityCorrection
 *   2. MultiPhaseSPH
 *   3. SPHBoundaryTreatment
 *   4. SPHContactHandler
 *   5. VerletNeighborList
 *   6. SPHThermalCoupling
 *   7. SPHMUSCLReconstruction
 */

#include <nexussim/sph/sph_wave19.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace nxs;
using namespace nxs::sph;

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

// ============================================================================
// 1. TensileInstabilityCorrection
// ============================================================================

void test_1_tensile_instability() {
    std::cout << "--- Test 1: TensileInstabilityCorrection ---\n";

    // 1a. Construction with default parameters
    {
        TensileInstabilityCorrection tic;
        CHECK_NEAR(tic.epsilon(), 0.3, 1e-15, "TIC: default epsilon = 0.3");
        CHECK_NEAR(tic.exponent(), 4.0, 1e-15, "TIC: default exponent = 4.0");
        CHECK_NEAR(tic.delta(), 0.01, 1e-15, "TIC: default delta = 0.01");
        CHECK(std::string(tic.name()) == "TensileInstabilityCorrection",
              "TIC: name is TensileInstabilityCorrection");
    }

    // 1b. Construction with custom parameters and setters
    {
        TensileInstabilityCorrection tic(0.5, 2.0, 0.005);
        CHECK_NEAR(tic.epsilon(), 0.5, 1e-15, "TIC: custom epsilon = 0.5");
        CHECK_NEAR(tic.exponent(), 2.0, 1e-15, "TIC: custom exponent = 2.0");
        CHECK_NEAR(tic.delta(), 0.005, 1e-15, "TIC: custom delta = 0.005");
    }

    // 1c. Artificial stress for tension: R = epsilon * sigma for positive normal stresses
    {
        TensileInstabilityCorrection tic(0.3, 4.0, 0.01);
        // Voigt notation: [s_xx, s_yy, s_zz, s_xy, s_yz, s_xz]
        Real stress[6] = {100.0, 200.0, 300.0, 10.0, 20.0, 30.0};
        Real R[6] = {};
        tic.compute_artificial_stress(stress, R);
        CHECK_NEAR(R[0], 0.3 * 100.0, 1e-10, "TIC: R_xx = eps * sigma_xx for tension");
        CHECK_NEAR(R[1], 0.3 * 200.0, 1e-10, "TIC: R_yy = eps * sigma_yy for tension");
        CHECK_NEAR(R[2], 0.3 * 300.0, 1e-10, "TIC: R_zz = eps * sigma_zz for tension");
    }

    // 1d. No correction for compression: R = 0 for negative normal stresses
    {
        TensileInstabilityCorrection tic(0.3, 4.0, 0.01);
        Real stress[6] = {-100.0, -200.0, -300.0, 10.0, 20.0, 30.0};
        Real R[6] = {};
        tic.compute_artificial_stress(stress, R);
        CHECK_NEAR(R[0], 0.0, 1e-15, "TIC: R_xx = 0 for compressive sigma_xx");
        CHECK_NEAR(R[1], 0.0, 1e-15, "TIC: R_yy = 0 for compressive sigma_yy");
        CHECK_NEAR(R[2], 0.0, 1e-15, "TIC: R_zz = 0 for compressive sigma_zz");
    }

    // 1e. Mixed tension/compression: only tensile components get artificial stress
    {
        TensileInstabilityCorrection tic(0.3, 4.0, 0.01);
        Real stress[6] = {100.0, -200.0, 50.0, 0.0, 0.0, 0.0};
        Real R[6] = {};
        tic.compute_artificial_stress(stress, R);
        CHECK_NEAR(R[0], 0.3 * 100.0, 1e-10, "TIC: mixed R_xx nonzero for tension");
        CHECK_NEAR(R[1], 0.0, 1e-15, "TIC: mixed R_yy zero for compression");
        CHECK_NEAR(R[2], 0.3 * 50.0, 1e-10, "TIC: mixed R_zz nonzero for tension");
    }

    // 1f. Kernel ratio factor: (W(r)/W(delta))^n
    {
        TensileInstabilityCorrection tic(0.3, 4.0, 0.01);
        SPHKernel kernel(KernelType::CubicSpline, 3);
        Real h = 0.02;

        // When r = delta, ratio = 1, factor = 1^4 = 1
        Real f_at_delta = tic.kernel_ratio_factor(0.01, h, kernel);
        CHECK_NEAR(f_at_delta, 1.0, 1e-10, "TIC: kernel ratio factor = 1 when r = delta");

        // When r < delta, W(r) > W(delta), so factor > 1 (repulsive enhancement)
        Real f_close = tic.kernel_ratio_factor(0.005, h, kernel);
        CHECK(f_close > 1.0, "TIC: kernel ratio factor > 1 when r < delta");

        // When r > delta, W(r) < W(delta), so factor < 1 (correction decays)
        Real f_far = tic.kernel_ratio_factor(0.015, h, kernel);
        CHECK(f_far < 1.0, "TIC: kernel ratio factor < 1 when r > delta");
    }

    // 1g. Pair acceleration computation
    {
        TensileInstabilityCorrection tic(0.3, 4.0, 0.01);
        Real R_i[6] = {30.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        Real R_j[6] = {30.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        Real gWx = 100.0, gWy = 0.0, gWz = 0.0;
        Real mass_j = 0.001;
        Real rho_i = 1000.0, rho_j = 1000.0;
        Real f = 1.0;
        Real ax, ay, az;
        tic.compute_pair_acceleration(R_i, R_j, gWx, gWy, gWz,
                                      mass_j, rho_i, rho_j, f, ax, ay, az);
        CHECK(std::abs(ax) > 1e-20, "TIC: pair acceleration nonzero in x for tensile correction");
        CHECK_NEAR(ay, 0.0, 1e-20, "TIC: pair acceleration zero in y (no grad_W_y)");
        CHECK_NEAR(az, 0.0, 1e-20, "TIC: pair acceleration zero in z (no grad_W_z)");
    }
}

// ============================================================================
// 2. MultiPhaseSPH
// ============================================================================

void test_2_multi_phase() {
    std::cout << "--- Test 2: MultiPhaseSPH ---\n";

    // 2a. Construction and default properties
    {
        MultiPhaseSPH mp;
        CHECK(mp.num_phases() == 2, "MultiPhase: default num_phases = 2");
        CHECK_NEAR(mp.interface_width(), 1.5, 1e-15, "MultiPhase: default interface_width = 1.5");
        CHECK(std::string(mp.name()) == "MultiPhaseSPH", "MultiPhase: name is MultiPhaseSPH");
    }

    // 2b. Construction with custom phase count
    {
        MultiPhaseSPH mp(3, 2.0);
        CHECK(mp.num_phases() == 3, "MultiPhase: custom num_phases = 3");
        CHECK_NEAR(mp.interface_width(), 2.0, 1e-15, "MultiPhase: custom interface_width = 2.0");
    }

    // 2c. Same-phase pressure uses standard SPH formulation: p_i/rho_i^2 + p_j/rho_j^2
    {
        MultiPhaseSPH mp;
        Real p_i = 1000.0, p_j = 2000.0;
        Real rho_i = 1000.0, rho_j = 1000.0;
        Real result = mp.smoothed_pressure_term(p_i, p_j, rho_i, rho_j, 0, 0);
        Real expected = p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j);
        CHECK_NEAR(result, expected, 1e-15, "MultiPhase: same-phase uses standard SPH pressure");
    }

    // 2d. Cross-phase pressure uses Hu-Adams weighted average
    {
        MultiPhaseSPH mp;
        Real p_i = 1000.0, p_j = 1000.0;
        Real rho_i = 1000.0, rho_j = 1.2;  // Water/air density ratio
        Real result = mp.smoothed_pressure_term(p_i, p_j, rho_i, rho_j, 0, 1);
        Real rho_avg = 0.5 * (rho_i + rho_j);
        Real p_avg = (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j);
        Real expected = 2.0 * p_avg / (rho_avg * rho_avg);
        CHECK_NEAR(result, expected, 1e-15, "MultiPhase: cross-phase uses Hu-Adams formulation");
    }

    // 2e. Interface detection via color function thresholds
    {
        MultiPhaseSPH mp;
        CHECK(!mp.is_interface(0.0), "MultiPhase: color=0.0 is NOT interface (pure phase 0)");
        CHECK(!mp.is_interface(1.0), "MultiPhase: color=1.0 is NOT interface (pure phase 1)");
        CHECK(mp.is_interface(0.5), "MultiPhase: color=0.5 IS interface");
        CHECK(!mp.is_interface(0.005), "MultiPhase: color=0.005 NOT interface (below 0.01)");
        CHECK(!mp.is_interface(0.995), "MultiPhase: color=0.995 NOT interface (above 0.99)");
    }

    // 2f. Effective viscosity: harmonic mean at interface
    {
        MultiPhaseSPH mp;
        mp.set_phase_properties(0, 1000.0, 0.001);
        mp.set_phase_properties(1, 1.2, 1.8e-5);

        Real mu_same = mp.effective_viscosity(0, 0);
        CHECK_NEAR(mu_same, 0.001, 1e-15, "MultiPhase: same-phase viscosity = phase viscosity");

        Real mu_cross = mp.effective_viscosity(0, 1);
        Real mu_0 = 0.001, mu_1 = 1.8e-5;
        Real expected_mu = 2.0 * mu_0 * mu_1 / (mu_0 + mu_1);
        CHECK_NEAR(mu_cross, expected_mu, 1e-15, "MultiPhase: cross-phase = harmonic mean");
    }
}

// ============================================================================
// 3. SPHBoundaryTreatment
// ============================================================================

void test_3_boundary_treatment() {
    std::cout << "--- Test 3: SPHBoundaryTreatment ---\n";

    // 3a. Construction and wall addition
    {
        SPHBoundaryTreatment bt;
        CHECK(bt.num_walls() == 0, "Boundary: no walls initially");
        CHECK(std::string(bt.name()) == "SPHBoundaryTreatment",
              "Boundary: name is SPHBoundaryTreatment");

        bt.add_planar_wall(0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           SPHBoundaryTreatment::BoundaryType::Repulsive);
        CHECK(bt.num_walls() == 1, "Boundary: one wall added");
    }

    // 3b. Ghost particle generation: mirror position across wall
    {
        SPHBoundaryTreatment bt(1480.0, 0.01);
        // Floor at z=0, normal (0,0,1)
        bt.add_planar_wall(0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           SPHBoundaryTreatment::BoundaryType::Mirror);
        Real mx, my, mz;
        bt.mirror_position(1.0, 2.0, 0.005, 0, mx, my, mz);
        CHECK_NEAR(mx, 1.0, 1e-12, "Boundary: mirror x unchanged");
        CHECK_NEAR(my, 2.0, 1e-12, "Boundary: mirror y unchanged");
        CHECK_NEAR(mz, -0.005, 1e-12, "Boundary: mirror z reflected below floor");
    }

    // 3c. Repulsive force pushes particle away from wall
    {
        SPHBoundaryTreatment bt(1480.0, 0.01);
        bt.add_planar_wall(0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           SPHBoundaryTreatment::BoundaryType::Repulsive);
        Real fx, fy, fz;
        // Particle at z=0.005, within r0=0.01
        bt.compute_repulsive_force(0.0, 0.0, 0.005, fx, fy, fz);
        CHECK(fz > 0.0, "Boundary: repulsive force pushes away from wall (+z)");
        CHECK_NEAR(fx, 0.0, 1e-15, "Boundary: no tangential force in x");
        CHECK_NEAR(fy, 0.0, 1e-15, "Boundary: no tangential force in y");
    }

    // 3d. No force beyond cutoff r0
    {
        SPHBoundaryTreatment bt(1480.0, 0.01);
        bt.add_planar_wall(0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           SPHBoundaryTreatment::BoundaryType::Repulsive);
        Real fx, fy, fz;
        bt.compute_repulsive_force(0.0, 0.0, 0.1, fx, fy, fz);
        CHECK_NEAR(fx, 0.0, 1e-15, "Boundary: no force in x when far from wall");
        CHECK_NEAR(fy, 0.0, 1e-15, "Boundary: no force in y when far from wall");
        CHECK_NEAR(fz, 0.0, 1e-15, "Boundary: no force in z when far from wall");
    }

    // 3e. Force increases as particle approaches wall (LJ behavior)
    {
        SPHBoundaryTreatment bt(1480.0, 0.01);
        bt.add_planar_wall(0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           SPHBoundaryTreatment::BoundaryType::Repulsive);
        Real fx1, fy1, fz1, fx2, fy2, fz2;
        bt.compute_repulsive_force(0.0, 0.0, 0.003, fx1, fy1, fz1);
        bt.compute_repulsive_force(0.0, 0.0, 0.008, fx2, fy2, fz2);
        CHECK(fz1 > fz2, "Boundary: force larger when closer to wall");
    }

    // 3f. Near-boundary detection
    {
        SPHBoundaryTreatment bt(1480.0, 0.01);
        bt.add_planar_wall(0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           SPHBoundaryTreatment::BoundaryType::Repulsive);
        CHECK(bt.near_boundary(0.0, 0.0, 0.005, 0.02),
              "Boundary: particle at z=0.005 is near wall (influence=0.02)");
        CHECK(!bt.near_boundary(0.0, 0.0, 0.5, 0.02),
              "Boundary: particle at z=0.5 is NOT near wall (influence=0.02)");
    }
}

// ============================================================================
// 4. SPHContactHandler
// ============================================================================

void test_4_contact_handler() {
    std::cout << "--- Test 4: SPHContactHandler ---\n";

    // 4a. Construction and accessors
    {
        SPHContactHandler ch;
        CHECK_NEAR(ch.penalty_stiffness(), 1.0e9, 1e-5, "Contact: default stiffness = 1e9");
        CHECK_NEAR(ch.friction_coefficient(), 0.3, 1e-15, "Contact: default friction = 0.3");
        CHECK_NEAR(ch.contact_distance(), 0.01, 1e-15, "Contact: default distance = 0.01");
        CHECK(std::string(ch.name()) == "SPHContactHandler",
              "Contact: name is SPHContactHandler");
    }

    // 4b. Contact detection: particles within contact distance
    {
        SPHContactHandler ch(1.0e9, 0.0, 0.01);  // No friction for simplicity
        Real rx = 0.008, ry = 0.0, rz = 0.0;
        Real r_mag = 0.008;  // < h_contact = 0.01
        Real vx = 0.0, vy = 0.0, vz = 0.0;
        Real fx, fy, fz;
        bool active = ch.compute_contact_force(rx, ry, rz, r_mag,
                                               vx, vy, vz, fx, fy, fz);
        CHECK(active, "Contact: active when r < h_contact");
    }

    // 4c. Penalty force magnitude: F = k_c * (h_contact - r)
    {
        SPHContactHandler ch(1.0e9, 0.0, 0.01);
        Real rx = 0.008, ry = 0.0, rz = 0.0;
        Real r_mag = 0.008;
        Real vx = 0.0, vy = 0.0, vz = 0.0;
        Real fx, fy, fz;
        ch.compute_contact_force(rx, ry, rz, r_mag, vx, vy, vz, fx, fy, fz);
        // d_pen = 0.01 - 0.008 = 0.002, Fn = 1e9 * 0.002 = 2e6
        Real expected_Fn = 1.0e9 * 0.002;
        // Force is along the x-direction (unit normal = (1,0,0))
        CHECK_NEAR(fx, expected_Fn, 1.0, "Contact: penalty force Fx = k * d_pen");
        CHECK_NEAR(fy, 0.0, 1e-10, "Contact: no force in y for x-aligned pair");
        CHECK_NEAR(fz, 0.0, 1e-10, "Contact: no force in z for x-aligned pair");
    }

    // 4d. No contact when particles are separated (r > h_contact)
    {
        SPHContactHandler ch(1.0e9, 0.0, 0.01);
        Real rx = 0.02, ry = 0.0, rz = 0.0;
        Real r_mag = 0.02;
        Real vx = 0.0, vy = 0.0, vz = 0.0;
        Real fx, fy, fz;
        bool active = ch.compute_contact_force(rx, ry, rz, r_mag,
                                               vx, vy, vz, fx, fy, fz);
        CHECK(!active, "Contact: not active when r > h_contact");
        CHECK_NEAR(fx, 0.0, 1e-15, "Contact: zero force when no contact");
    }

    // 4e. Friction force opposes tangential sliding velocity
    {
        SPHContactHandler ch(1.0e9, 0.5, 0.01);
        // Particles overlap in x, sliding in y
        Real rx = 0.008, ry = 0.0, rz = 0.0;
        Real r_mag = 0.008;
        Real vx = 0.0, vy = 10.0, vz = 0.0;  // Tangential sliding
        Real fx, fy, fz;
        bool active = ch.compute_contact_force(rx, ry, rz, r_mag,
                                               vx, vy, vz, fx, fy, fz);
        CHECK(active, "Contact: active with friction");
        CHECK(fy < 0.0, "Contact: friction force opposes sliding in y");
    }

    // 4f. Setters update parameters
    {
        SPHContactHandler ch;
        ch.set_penalty_stiffness(2.0e10);
        ch.set_friction(0.7);
        ch.set_contact_distance(0.005);
        CHECK_NEAR(ch.penalty_stiffness(), 2.0e10, 1e-5, "Contact: set stiffness = 2e10");
        CHECK_NEAR(ch.friction_coefficient(), 0.7, 1e-15, "Contact: set friction = 0.7");
        CHECK_NEAR(ch.contact_distance(), 0.005, 1e-15, "Contact: set distance = 0.005");
    }
}

// ============================================================================
// 5. VerletNeighborList
// ============================================================================

void test_5_verlet_neighbor_list() {
    std::cout << "--- Test 5: VerletNeighborList ---\n";

    // 5a. Construction with explicit skin distance
    {
        VerletNeighborList vnl(0.04, 0.008);
        CHECK_NEAR(vnl.support_radius(), 0.04, 1e-15, "Verlet: support_radius = 0.04");
        CHECK_NEAR(vnl.skin_distance(), 0.008, 1e-15, "Verlet: skin_distance = 0.008");
        CHECK_NEAR(vnl.total_radius(), 0.048, 1e-15, "Verlet: total = support + skin");
        CHECK(vnl.build_count() == 0, "Verlet: initial build_count = 0");
        CHECK(std::string(vnl.name()) == "VerletNeighborList",
              "Verlet: name is VerletNeighborList");
    }

    // 5b. Default skin distance = 10% of support radius
    {
        VerletNeighborList vnl(0.1);
        CHECK_NEAR(vnl.skin_distance(), 0.01, 1e-15, "Verlet: default skin = 0.1 * support");
        CHECK_NEAR(vnl.total_radius(), 0.11, 1e-15, "Verlet: total = 0.1 + 0.01");
    }

    // 5c. Build neighbor list with simple particle configuration
    {
        Real px[3] = {0.0, 0.01, 0.02};
        Real py[3] = {0.0, 0.0, 0.0};
        Real pz[3] = {0.0, 0.0, 0.0};
        Real support = 0.025;
        Real skin = 0.005;

        VerletNeighborList vnl(support, skin);
        SpatialHashGrid grid(support + skin);
        vnl.build(px, py, pz, 3, grid);

        CHECK(vnl.build_count() == 1, "Verlet: build count incremented to 1");
        CHECK(vnl.verlet_pair_count() > 0, "Verlet: found neighbor pairs after build");
    }

    // 5d. No rebuild needed when particles haven't moved
    {
        Real px[3] = {0.0, 0.01, 0.02};
        Real py[3] = {0.0, 0.0, 0.0};
        Real pz[3] = {0.0, 0.0, 0.0};
        Real support = 0.025;
        Real skin = 0.01;

        VerletNeighborList vnl(support, skin);
        SpatialHashGrid grid(support + skin);
        vnl.build(px, py, pz, 3, grid);

        bool rebuild = vnl.needs_rebuild(px, py, pz, 3);
        CHECK(!rebuild, "Verlet: no rebuild when particles stationary");
    }

    // 5e. Rebuild triggered when max displacement exceeds skin/2
    {
        Real px[3] = {0.0, 0.01, 0.02};
        Real py[3] = {0.0, 0.0, 0.0};
        Real pz[3] = {0.0, 0.0, 0.0};
        Real support = 0.025;
        Real skin = 0.004;  // half_skin = 0.002

        VerletNeighborList vnl(support, skin);
        SpatialHashGrid grid(support + skin);
        vnl.build(px, py, pz, 3, grid);

        // Move particle 0 by 0.003, exceeding half_skin = 0.002
        Real px2[3] = {0.003, 0.01, 0.02};
        bool rebuild = vnl.needs_rebuild(px2, py, pz, 3);
        CHECK(rebuild, "Verlet: rebuild triggered when displacement > skin/2");
    }

    // 5f. Setters update radius correctly
    {
        VerletNeighborList vnl(0.04, 0.008);
        vnl.set_support_radius(0.05);
        CHECK_NEAR(vnl.support_radius(), 0.05, 1e-15, "Verlet: updated support to 0.05");
        CHECK_NEAR(vnl.total_radius(), 0.05 + 0.008, 1e-15,
                   "Verlet: total updated with new support");

        vnl.set_skin_distance(0.01);
        CHECK_NEAR(vnl.skin_distance(), 0.01, 1e-15, "Verlet: updated skin to 0.01");
        CHECK_NEAR(vnl.total_radius(), 0.05 + 0.01, 1e-15,
                   "Verlet: total updated with new skin");
    }
}

// ============================================================================
// 6. SPHThermalCoupling
// ============================================================================

void test_6_thermal_coupling() {
    std::cout << "--- Test 6: SPHThermalCoupling ---\n";

    // 6a. Construction and default properties
    {
        SPHThermalCoupling tc;
        CHECK_NEAR(tc.conductivity(), 50.0, 1e-15, "Thermal: default conductivity = 50 W/m/K");
        CHECK_NEAR(tc.specific_heat(), 500.0, 1e-15, "Thermal: default cp = 500 J/kg/K");
        CHECK(std::string(tc.name()) == "SPHThermalCoupling",
              "Thermal: name is SPHThermalCoupling");
    }

    // 6b. Custom construction and setters
    {
        SPHThermalCoupling tc(200.0, 900.0);
        CHECK_NEAR(tc.conductivity(), 200.0, 1e-15, "Thermal: custom conductivity = 200");
        CHECK_NEAR(tc.specific_heat(), 900.0, 1e-15, "Thermal: custom cp = 900");

        tc.set_conductivity(237.0);
        tc.set_specific_heat(897.0);
        CHECK_NEAR(tc.conductivity(), 237.0, 1e-15, "Thermal: set conductivity to 237 (Al)");
        CHECK_NEAR(tc.specific_heat(), 897.0, 1e-15, "Thermal: set cp to 897 (Al)");
    }

    // 6c. Pair heat flux: heat flows from hot to cold
    {
        Real T_i = 300.0, T_j = 500.0;
        Real mass_i = 0.001, mass_j = 0.001;
        Real rho_i = 1000.0, rho_j = 1000.0;
        Real k_i = 50.0, k_j = 50.0;
        Real cp_i = 500.0, cp_j = 500.0;
        Real r = 0.01;
        Real grad_W_mag = 1000.0;

        Real dTdt_i, dTdt_j;
        SPHThermalCoupling::compute_pair_heat_flux(
            T_i, T_j, mass_i, mass_j, rho_i, rho_j,
            k_i, k_j, cp_i, cp_j, r, grad_W_mag,
            dTdt_i, dTdt_j);
        CHECK(dTdt_i > 0.0, "Thermal: cold particle heats up (dTdt_i > 0)");
        CHECK(dTdt_j < 0.0, "Thermal: hot particle cools down (dTdt_j < 0)");
    }

    // 6d. Zero flux at uniform temperature
    {
        Real T_i = 400.0, T_j = 400.0;
        Real mass_i = 0.001, mass_j = 0.001;
        Real rho_i = 1000.0, rho_j = 1000.0;
        Real k_i = 50.0, k_j = 50.0;
        Real cp_i = 500.0, cp_j = 500.0;
        Real r = 0.01;
        Real grad_W_mag = 1000.0;

        Real dTdt_i, dTdt_j;
        SPHThermalCoupling::compute_pair_heat_flux(
            T_i, T_j, mass_i, mass_j, rho_i, rho_j,
            k_i, k_j, cp_i, cp_j, r, grad_W_mag,
            dTdt_i, dTdt_j);
        CHECK_NEAR(dTdt_i, 0.0, 1e-15, "Thermal: zero flux at uniform T (particle i)");
        CHECK_NEAR(dTdt_j, 0.0, 1e-15, "Thermal: zero flux at uniform T (particle j)");
    }

    // 6e. Interface conductivity uses harmonic mean
    {
        Real T_i = 300.0, T_j = 500.0;
        Real mass_i = 0.001, mass_j = 0.001;
        Real rho_i = 1000.0, rho_j = 1000.0;
        Real k_i = 100.0, k_j = 50.0;  // Different conductivities
        Real cp_i = 500.0, cp_j = 500.0;
        Real r = 0.01;
        Real grad_W_mag = 1000.0;

        Real dTdt_i, dTdt_j;
        SPHThermalCoupling::compute_pair_heat_flux(
            T_i, T_j, mass_i, mass_j, rho_i, rho_j,
            k_i, k_j, cp_i, cp_j, r, grad_W_mag,
            dTdt_i, dTdt_j);

        // k_ij = 2*100*50 / (100+50) = 66.6667
        Real k_ij = 2.0 * k_i * k_j / (k_i + k_j);
        Real vol_j = mass_j / rho_j;
        Real laplacian = 2.0 * grad_W_mag / r;
        Real dT = T_j - T_i;
        Real expected_dTdt_i = (k_ij / (rho_i * cp_i)) * vol_j * dT * laplacian;
        CHECK_NEAR(dTdt_i, expected_dTdt_i, 1e-10,
                   "Thermal: dTdt_i matches harmonic-mean conductivity formula");
    }

    // 6f. Energy conservation in symmetric pair exchange
    {
        Real T_i = 400.0, T_j = 300.0;
        Real mass_i = 0.001, mass_j = 0.001;
        Real rho_i = 1000.0, rho_j = 1000.0;
        Real k_i = 50.0, k_j = 50.0;
        Real cp_i = 500.0, cp_j = 500.0;
        Real r = 0.01;
        Real grad_W_mag = 1000.0;

        Real dTdt_i, dTdt_j;
        SPHThermalCoupling::compute_pair_heat_flux(
            T_i, T_j, mass_i, mass_j, rho_i, rho_j,
            k_i, k_j, cp_i, cp_j, r, grad_W_mag,
            dTdt_i, dTdt_j);

        // Energy change: dE_i = m_i * cp_i * dTdt_i, dE_j = m_j * cp_j * dTdt_j
        // Should sum to zero (energy conservation)
        Real dE_i = mass_i * cp_i * dTdt_i;
        Real dE_j = mass_j * cp_j * dTdt_j;
        CHECK_NEAR(dE_i + dE_j, 0.0, std::abs(dE_i) * 1e-6,
                   "Thermal: energy conserved in symmetric pair exchange");
    }
}

// ============================================================================
// 7. SPHMUSCLReconstruction
// ============================================================================

void test_7_muscl_reconstruction() {
    std::cout << "--- Test 7: SPHMUSCLReconstruction ---\n";

    // 7a. Construction with default limiter
    {
        SPHMUSCLReconstruction muscl;
        CHECK(muscl.limiter_type() == SPHMUSCLReconstruction::LimiterType::Minmod,
              "MUSCL: default limiter = Minmod");
        CHECK(std::string(muscl.name()) == "SPHMUSCLReconstruction",
              "MUSCL: name is SPHMUSCLReconstruction");
    }

    // 7b. Minmod limiter properties
    {
        CHECK_NEAR(SPHMUSCLReconstruction::minmod(0.5), 0.5, 1e-15,
                   "MUSCL: minmod(0.5) = 0.5 (linear region)");
        CHECK_NEAR(SPHMUSCLReconstruction::minmod(2.0), 1.0, 1e-15,
                   "MUSCL: minmod(2.0) = 1.0 (clamped)");
        CHECK_NEAR(SPHMUSCLReconstruction::minmod(-1.0), 0.0, 1e-15,
                   "MUSCL: minmod(-1) = 0 (sign reversal kills slope)");
        CHECK_NEAR(SPHMUSCLReconstruction::minmod(1.0), 1.0, 1e-15,
                   "MUSCL: minmod(1) = 1 (full second-order)");
    }

    // 7c. Van Leer limiter properties
    {
        CHECK_NEAR(SPHMUSCLReconstruction::van_leer(1.0), 1.0, 1e-15,
                   "MUSCL: van_leer(1) = 1");
        CHECK_NEAR(SPHMUSCLReconstruction::van_leer(-1.0), 0.0, 1e-15,
                   "MUSCL: van_leer(-1) = 0");
        CHECK_NEAR(SPHMUSCLReconstruction::van_leer(2.0), 4.0 / 3.0, 1e-15,
                   "MUSCL: van_leer(2) = 4/3");
    }

    // 7d. Superbee limiter properties
    {
        // r=0.5: max(min(1,1), min(0.5,2)) = max(1, 0.5) = 1
        CHECK_NEAR(SPHMUSCLReconstruction::superbee(0.5), 1.0, 1e-15,
                   "MUSCL: superbee(0.5) = 1.0");
        CHECK_NEAR(SPHMUSCLReconstruction::superbee(1.0), 1.0, 1e-15,
                   "MUSCL: superbee(1.0) = 1.0");
        CHECK_NEAR(SPHMUSCLReconstruction::superbee(-1.0), 0.0, 1e-15,
                   "MUSCL: superbee(-1) = 0 (opposite sign)");
    }

    // 7e. Gradient computation for linear field
    {
        SPHMUSCLReconstruction muscl;
        SPHKernel kernel(KernelType::CubicSpline, 3);
        Real h = 0.02;

        // Linear field phi = x with 3 particles
        Real phi[3] = {0.0, 0.01, 0.02};
        Real mass[3] = {1.0, 1.0, 1.0};
        Real rho_arr[3] = {1000.0, 1000.0, 1000.0};
        Real px[3] = {0.0, 0.01, 0.02};
        Real py[3] = {0.0, 0.0, 0.0};
        Real pz[3] = {0.0, 0.0, 0.0};

        Index nbrs[2] = {0, 2};
        Real gx, gy, gz;
        muscl.compute_gradient(1, phi, mass, rho_arr, px, py, pz,
                               nbrs, 2, h, kernel, gx, gy, gz);

        CHECK(std::abs(gx) > 0.0, "MUSCL: gradient x nonzero for linear field phi=x");
        CHECK_NEAR(gy, 0.0, 1e-6, "MUSCL: gradient y zero for 1D field");
        CHECK_NEAR(gz, 0.0, 1e-6, "MUSCL: gradient z zero for 1D field");
    }

    // 7f. Uniform field: reconstruction preserves constant values
    {
        SPHMUSCLReconstruction muscl(SPHMUSCLReconstruction::LimiterType::Minmod);
        Real phi_i = 10.0, phi_j = 10.0;
        Real grad_i[3] = {0.0, 0.0, 0.0};
        Real grad_j[3] = {0.0, 0.0, 0.0};
        Real rx = 0.01, ry = 0.0, rz = 0.0;
        Real phi_L, phi_R;
        muscl.reconstruct(phi_i, phi_j, grad_i, grad_j, rx, ry, rz, phi_L, phi_R);
        CHECK_NEAR(phi_L, 10.0, 1e-12, "MUSCL: constant field phi_L = phi_i");
        CHECK_NEAR(phi_R, 10.0, 1e-12, "MUSCL: constant field phi_R = phi_j");
    }

    // 7g. Density reconstruction enforces positivity
    {
        SPHMUSCLReconstruction muscl(SPHMUSCLReconstruction::LimiterType::Minmod);
        Real rho_i = 0.001, rho_j = 0.001;
        Real grad_rho_i[3] = {-1000.0, 0.0, 0.0};
        Real grad_rho_j[3] = {-1000.0, 0.0, 0.0};
        Real rx = 0.01, ry = 0.0, rz = 0.0;
        Real rho_L, rho_R;
        muscl.reconstruct_density(rho_i, rho_j, grad_rho_i, grad_rho_j,
                                  rx, ry, rz, rho_L, rho_R);
        CHECK(rho_L >= 1.0e-10, "MUSCL: density rho_L enforced >= 1e-10");
        CHECK(rho_R >= 1.0e-10, "MUSCL: density rho_R enforced >= 1e-10");
    }

    // 7h. Limiter setter
    {
        SPHMUSCLReconstruction muscl;
        muscl.set_limiter(SPHMUSCLReconstruction::LimiterType::VanLeer);
        CHECK(muscl.limiter_type() == SPHMUSCLReconstruction::LimiterType::VanLeer,
              "MUSCL: limiter changed to VanLeer");
        muscl.set_limiter(SPHMUSCLReconstruction::LimiterType::Superbee);
        CHECK(muscl.limiter_type() == SPHMUSCLReconstruction::LimiterType::Superbee,
              "MUSCL: limiter changed to Superbee");
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Wave 19 SPH Enrichment Test Suite ===\n\n";

    test_1_tensile_instability();
    test_2_multi_phase();
    test_3_boundary_treatment();
    test_4_contact_handler();
    test_5_verlet_neighbor_list();
    test_6_thermal_coupling();
    test_7_muscl_reconstruction();

    std::cout << "\n=== Wave 19 SPH Enrichment: " << tests_passed
              << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
