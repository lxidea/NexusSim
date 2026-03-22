/**
 * @file contact_wave28_test.cpp
 * @brief Wave 28: Advanced Contact Algorithms Test Suite (5 features, ~45 tests)
 *
 * Tests 5 sub-modules (9 tests each):
 *  1. CurvedSegmentContact     — Quadratic interpolation, projection, gap, force
 *  2. ThermalContactResistance — Gap conductance, radiation, total flux
 *  3. WearModelContact         — Archard wear, geometry update, volume
 *  4. RollingResistanceContact — Rolling/spin moments, effective radius
 *  5. IntersectionAwareContact — Overlap detection, ramp, force clamping, energy
 */

#include <nexussim/fem/contact_wave28.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <array>

using namespace nxs;
using namespace nxs::fem;

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
// 1. CurvedSegmentContact Tests
// ============================================================================
void test_curved_segment_contact() {
    std::cout << "\n=== 28a: CurvedSegmentContact ===\n";

    CurvedSegmentContact contact;

    // Straight segment along x-axis: nodes at (-1,0,0), (0,0,0), (1,0,0)
    Real straight_nodes[3][3] = {
        {-1.0, 0.0, 0.0},
        { 0.0, 0.0, 0.0},
        { 1.0, 0.0, 0.0}
    };

    // Curved segment: midpoint displaced in y to form a parabola
    Real curved_nodes[3][3] = {
        {-1.0, 0.0, 0.0},
        { 0.0, 0.5, 0.0},
        { 1.0, 0.0, 0.0}
    };

    // Test 1: Projection on midpoint of straight segment
    {
        Real slave[3] = {0.0, 1.0, 0.0};
        Real xi, gap;
        Real normal[3];
        contact.project_point(slave, straight_nodes, xi, gap, normal);
        CHECK_NEAR(xi, 0.0, 1e-6, "CurvedSeg: projection on midpoint gives xi=0");
    }

    // Test 2: Projection on endpoint of straight segment
    {
        Real slave[3] = {1.0, 1.0, 0.0};
        Real xi, gap;
        Real normal[3];
        contact.project_point(slave, straight_nodes, xi, gap, normal);
        CHECK_NEAR(xi, 1.0, 1e-6, "CurvedSeg: projection on endpoint gives xi=1");
    }

    // Test 3: Curved vs flat gap accuracy
    {
        // Slave point directly above midpoint
        Real slave[3] = {0.0, 1.0, 0.0};
        Real xi_c, gap_c, normal_c[3];
        Real xi_s, gap_s, normal_s[3];
        contact.project_point(slave, curved_nodes, xi_c, gap_c, normal_c);
        contact.project_point(slave, straight_nodes, xi_s, gap_s, normal_s);
        // Curved segment midpoint is at y=0.5, so absolute gap should be smaller
        CHECK(std::abs(gap_c) < std::abs(gap_s),
              "CurvedSeg: curved gap smaller than flat gap for raised midpoint");
    }

    // Test 4: Newton convergence (few iterations for simple case)
    {
        Real slave[3] = {0.3, 0.5, 0.0};
        Real xi, gap;
        Real normal[3];
        int iters = contact.project_point(slave, straight_nodes, xi, gap, normal);
        CHECK(iters <= 10, "CurvedSeg: Newton converges in <= 10 iterations");
    }

    // Test 5: Normal direction (should be perpendicular to straight segment)
    {
        Real slave[3] = {0.0, 1.0, 0.0};
        Real xi, gap;
        Real normal[3];
        contact.project_point(slave, straight_nodes, xi, gap, normal);
        // For a segment along x, normal should be in y direction
        Real ny = std::abs(normal[1]);
        CHECK(ny > 0.9, "CurvedSeg: normal is approximately in y-direction for x-segment");
    }

    // Test 6: Force magnitude for penetration
    {
        Real gap = -0.01;
        Real stiffness = 1.0e6;
        Real force = contact.contact_force(gap, stiffness);
        CHECK_NEAR(force, 1.0e4, 1e-6, "CurvedSeg: force = stiffness * |gap|");
    }

    // Test 7: No penetration force for positive gap
    {
        Real gap = 0.05;
        Real stiffness = 1.0e6;
        Real force = contact.contact_force(gap, stiffness);
        CHECK_NEAR(force, 0.0, 1e-12, "CurvedSeg: zero force for positive gap");
    }

    // Test 8: Quadratic shape function check (partition of unity)
    {
        Real N[3];
        CurvedSegmentContact::shape_functions(0.3, N);
        Real sum = N[0] + N[1] + N[2];
        CHECK_NEAR(sum, 1.0, 1e-12, "CurvedSeg: shape functions sum to 1.0");
    }

    // Test 9: Tangent vector for straight segment is constant
    {
        Real tang1[3], tang2[3];
        contact.tangent_at(-0.5, straight_nodes, tang1);
        contact.tangent_at(0.5, straight_nodes, tang2);
        // For straight segment, tangent direction should be the same
        Real dot = tang1[0]*tang2[0] + tang1[1]*tang2[1] + tang1[2]*tang2[2];
        Real mag1 = std::sqrt(tang1[0]*tang1[0] + tang1[1]*tang1[1] + tang1[2]*tang1[2]);
        Real mag2 = std::sqrt(tang2[0]*tang2[0] + tang2[1]*tang2[1] + tang2[2]*tang2[2]);
        Real cos_angle = dot / (mag1 * mag2);
        CHECK_NEAR(cos_angle, 1.0, 1e-10, "CurvedSeg: tangent constant for straight segment");
    }
}

// ============================================================================
// 2. ThermalContactResistance Tests
// ============================================================================
void test_thermal_contact_resistance() {
    std::cout << "\n=== 28b: ThermalContactResistance ===\n";

    // h_c0=1000, p_ref=1e6, n=0.5, k_gas=0.025, eps1=0.8, eps2=0.8
    ThermalContactResistance tcr(1000.0, 1.0e6, 0.5, 0.025, 0.8, 0.8);

    // Test 1: Conductance at zero pressure (only gas term)
    {
        Real h = tcr.gap_conductance(0.0, 0.001);
        // h = 0 (pressure term) + 1.0 * 0.025 / 0.001 = 25.0
        CHECK_NEAR(h, 25.0, 1e-6, "ThermalCR: zero pressure gives gas-only conductance");
    }

    // Test 2: Conductance at reference pressure
    {
        Real h = tcr.gap_conductance(1.0e6, 0.001);
        // h = 1000 * (1e6/1e6)^0.5 + 0.025/0.001 = 1000 + 25 = 1025
        CHECK_NEAR(h, 1025.0, 1e-6, "ThermalCR: conductance at p_ref");
    }

    // Test 3: Pressure exponent effect
    {
        Real h1 = tcr.gap_conductance(0.25e6, 1.0e10);  // tiny gas term
        Real h2 = tcr.gap_conductance(1.0e6, 1.0e10);
        // h1 pressure part = 1000*(0.25)^0.5 = 500
        // h2 pressure part = 1000*(1.0)^0.5 = 1000
        // Ratio should be 0.5
        Real ratio = h1 / h2;
        CHECK_NEAR(ratio, 0.5, 0.01, "ThermalCR: pressure exponent effect (sqrt)");
    }

    // Test 4: Gas conductance for large gap
    {
        Real h_small = tcr.gap_conductance(0.0, 0.001);
        Real h_large = tcr.gap_conductance(0.0, 0.01);
        // Gas only: h = k_gas/d, so h_small/h_large = 10
        Real ratio = h_small / h_large;
        CHECK_NEAR(ratio, 10.0, 1e-6, "ThermalCR: gas conductance inversely proportional to gap");
    }

    // Test 5: Radiation at equal temperatures = 0
    {
        Real q = tcr.radiation_flux(500.0, 500.0);
        CHECK_NEAR(q, 0.0, 1e-10, "ThermalCR: zero radiation for equal temperatures");
    }

    // Test 6: Radiation direction (T1 > T2 gives positive flux)
    {
        Real q = tcr.radiation_flux(600.0, 300.0);
        CHECK(q > 0.0, "ThermalCR: positive radiation from hot to cold");
        Real q_rev = tcr.radiation_flux(300.0, 600.0);
        CHECK(q_rev < 0.0, "ThermalCR: negative radiation from cold to hot");
    }

    // Test 7: Total flux combines conduction and radiation
    {
        Real T1 = 500.0, T2 = 300.0;
        Real p = 1.0e6, d = 0.001;
        Real q_total = tcr.total_heat_flux(T1, T2, p, d);
        Real h_c = tcr.gap_conductance(p, d);
        Real q_cond = h_c * (T1 - T2);
        Real q_rad = tcr.radiation_flux(T1, T2);
        CHECK_NEAR(q_total, q_cond + q_rad, 1e-6, "ThermalCR: total = conduction + radiation");
    }

    // Test 8: High pressure limit
    {
        Real h_high = tcr.gap_conductance(100.0e6, 1.0e10);
        Real h_ref = tcr.gap_conductance(1.0e6, 1.0e10);
        // h_high ~ 1000 * (100)^0.5 = 10000, h_ref ~ 1000
        Real ratio = h_high / h_ref;
        CHECK_NEAR(ratio, 10.0, 0.1, "ThermalCR: high pressure conductance scales as sqrt(p)");
    }

    // Test 9: Linearized radiation conductance
    {
        Real T_avg = 500.0;
        Real h_rad = tcr.linearized_radiation_conductance(500.0, 500.0);
        // eps_eff = 1/(1/0.8 + 1/0.8 - 1) = 1/(2.5-1) = 1/1.5 = 2/3
        Real eps_eff = 2.0/3.0;
        Real expected = 4.0 * eps_eff * 5.67e-8 * T_avg * T_avg * T_avg;
        CHECK_NEAR(h_rad, expected, 1e-3, "ThermalCR: linearized radiation conductance");
    }
}

// ============================================================================
// 3. WearModelContact Tests
// ============================================================================
void test_wear_model_contact() {
    std::cout << "\n=== 28c: WearModelContact ===\n";

    // K=1e-4, H=1e9
    WearModelContact wear(1.0e-4, 1.0e9);

    // Test 1: Zero sliding = zero wear
    {
        Real rate = wear.wear_depth_rate(1.0e6, 0.0);
        CHECK_NEAR(rate, 0.0, 1e-20, "Wear: zero sliding velocity gives zero wear");
    }

    // Test 2: Wear rate proportional to pressure
    {
        Real r1 = wear.wear_depth_rate(1.0e6, 1.0);
        Real r2 = wear.wear_depth_rate(2.0e6, 1.0);
        CHECK_NEAR(r2 / r1, 2.0, 1e-10, "Wear: rate proportional to pressure");
    }

    // Test 3: Wear rate proportional to velocity
    {
        Real r1 = wear.wear_depth_rate(1.0e6, 1.0);
        Real r2 = wear.wear_depth_rate(1.0e6, 3.0);
        CHECK_NEAR(r2 / r1, 3.0, 1e-10, "Wear: rate proportional to sliding velocity");
    }

    // Test 4: Inverse proportional to hardness
    {
        WearModelContact wear_soft(1.0e-4, 0.5e9);
        Real r_hard = wear.wear_depth_rate(1.0e6, 1.0);
        Real r_soft = wear_soft.wear_depth_rate(1.0e6, 1.0);
        CHECK_NEAR(r_soft / r_hard, 2.0, 1e-10, "Wear: rate inversely proportional to hardness");
    }

    // Test 5: Geometry shift direction
    {
        Real pos[3] = {1.0, 2.0, 3.0};
        Real normal[3] = {0.0, 1.0, 0.0};
        Real updated[3];
        wear.geometry_update(pos, normal, 0.01, updated);
        // Node should shift inward (opposite to normal)
        CHECK_NEAR(updated[0], 1.0, 1e-12, "Wear: x unchanged for y-normal");
        CHECK_NEAR(updated[1], 1.99, 1e-12, "Wear: y shifted inward by wear depth");
        CHECK_NEAR(updated[2], 3.0, 1e-12, "Wear: z unchanged for y-normal");
    }

    // Test 6: Accumulated wear
    {
        Real depth = 0.0;
        Real rate = wear.wear_depth_rate(1.0e6, 1.0);
        Real dt = 0.01;
        depth = wear.accumulate_wear(depth, rate, dt);
        // rate = 1e-4 * 1e6 * 1.0 / 1e9 = 1e-7
        // depth = 0 + 1e-7 * 0.01 = 1e-9
        CHECK_NEAR(depth, 1.0e-9, 1e-15, "Wear: accumulated depth after one step");
    }

    // Test 7: Total volume calculation
    {
        Real depths[3] = {0.001, 0.002, 0.003};
        Real areas[3] = {0.01, 0.02, 0.03};
        Real vol = wear.total_volume_worn(depths, areas, 3);
        // V = 0.001*0.01 + 0.002*0.02 + 0.003*0.03 = 1e-5 + 4e-5 + 9e-5 = 1.4e-4
        CHECK_NEAR(vol, 1.4e-4, 1e-10, "Wear: total volume = sum(depth*area)");
    }

    // Test 8: Wear coefficient effect
    {
        WearModelContact wear_high(2.0e-4, 1.0e9);
        Real r1 = wear.wear_depth_rate(1.0e6, 1.0);
        Real r2 = wear_high.wear_depth_rate(1.0e6, 1.0);
        CHECK_NEAR(r2 / r1, 2.0, 1e-10, "Wear: doubling K doubles wear rate");
    }

    // Test 9: Multiple time steps accumulation
    {
        Real depth = 0.0;
        Real rate = wear.wear_depth_rate(1.0e6, 1.0);  // 1e-7
        Real dt = 0.01;
        for (int i = 0; i < 100; ++i) {
            depth = wear.accumulate_wear(depth, rate, dt);
        }
        // depth = 100 * 1e-7 * 0.01 = 1e-7
        CHECK_NEAR(depth, 1.0e-7, 1e-13, "Wear: accumulated depth after 100 steps");
    }
}

// ============================================================================
// 4. RollingResistanceContact Tests
// ============================================================================
void test_rolling_resistance_contact() {
    std::cout << "\n=== 28d: RollingResistanceContact ===\n";

    RollingResistanceContact rrc(0.01, 0.005);

    // Test 1: Effective radius formula
    {
        Real R_eff = rrc.effective_radius(0.1, 0.1);
        // R_eff = 0.1*0.1 / 0.2 = 0.05
        CHECK_NEAR(R_eff, 0.05, 1e-12, "Rolling: R_eff = R/2 for equal radii");
    }

    // Test 2: Rolling moment magnitude
    {
        Real omega[3] = {0.0, 0.0, 10.0};
        Real M_roll[3];
        rrc.rolling_moment(100.0, 0.1, 0.1, omega, M_roll);
        Real mag = std::sqrt(M_roll[0]*M_roll[0] + M_roll[1]*M_roll[1] + M_roll[2]*M_roll[2]);
        // |M| = mu_roll * F_n * R_eff = 0.01 * 100 * 0.05 = 0.05
        CHECK_NEAR(mag, 0.05, 1e-10, "Rolling: moment magnitude = mu*Fn*Reff");
    }

    // Test 3: Spin moment direction along normal
    {
        Real normal[3] = {0.0, 0.0, 1.0};
        Real M_spin[3];
        rrc.spin_moment(100.0, 0.1, 0.1, 10.0, normal, M_spin);
        // Should be along z-axis (normal), opposing spin
        CHECK(M_spin[2] < 0.0, "Rolling: spin moment opposes positive spin");
        CHECK_NEAR(std::abs(M_spin[0]), 0.0, 1e-12, "Rolling: spin moment x=0 for z-normal");
        CHECK_NEAR(std::abs(M_spin[1]), 0.0, 1e-12, "Rolling: spin moment y=0 for z-normal");
    }

    // Test 4: Zero omega = zero moment
    {
        Real omega[3] = {0.0, 0.0, 0.0};
        Real M_roll[3];
        rrc.rolling_moment(100.0, 0.1, 0.1, omega, M_roll);
        Real mag = std::sqrt(M_roll[0]*M_roll[0] + M_roll[1]*M_roll[1] + M_roll[2]*M_roll[2]);
        CHECK_NEAR(mag, 0.0, 1e-20, "Rolling: zero omega gives zero moment");
    }

    // Test 5: Direction opposes motion
    {
        Real omega[3] = {5.0, 0.0, 0.0};  // Rolling about x-axis
        Real M_roll[3];
        rrc.rolling_moment(100.0, 0.1, 0.1, omega, M_roll);
        // Moment should oppose omega: M_roll[0] < 0
        CHECK(M_roll[0] < 0.0, "Rolling: moment opposes angular velocity direction");
    }

    // Test 6: mu_roll scaling
    {
        RollingResistanceContact rrc2(0.02, 0.005);
        Real omega[3] = {0.0, 0.0, 10.0};
        Real M1[3], M2[3];
        rrc.rolling_moment(100.0, 0.1, 0.1, omega, M1);
        rrc2.rolling_moment(100.0, 0.1, 0.1, omega, M2);
        Real mag1 = std::sqrt(M1[0]*M1[0] + M1[1]*M1[1] + M1[2]*M1[2]);
        Real mag2 = std::sqrt(M2[0]*M2[0] + M2[1]*M2[1] + M2[2]*M2[2]);
        CHECK_NEAR(mag2 / mag1, 2.0, 1e-10, "Rolling: doubling mu_roll doubles moment");
    }

    // Test 7: Infinite R2 -> R_eff = R1
    {
        Real R_eff = rrc.effective_radius(0.1, 1.0e10);
        CHECK_NEAR(R_eff, 0.1, 1e-6, "Rolling: large R2 gives R_eff ~ R1");
    }

    // Test 8: Combined rolling + spin
    {
        Real omega[3] = {5.0, 0.0, 3.0};  // Rolling in x, spin in z
        Real normal[3] = {0.0, 0.0, 1.0};
        Real M_total[3];
        rrc.total_resistance_moment(100.0, 0.1, 0.1, omega, normal, M_total);
        Real mag = std::sqrt(M_total[0]*M_total[0] + M_total[1]*M_total[1] + M_total[2]*M_total[2]);
        CHECK(mag > 0.0, "Rolling: combined moment is nonzero for mixed omega");
    }

    // Test 9: Moment symmetry (swapping R1, R2 gives same magnitude)
    {
        Real omega[3] = {0.0, 0.0, 10.0};
        Real M1[3], M2[3];
        rrc.rolling_moment(100.0, 0.1, 0.2, omega, M1);
        rrc.rolling_moment(100.0, 0.2, 0.1, omega, M2);
        Real mag1 = std::sqrt(M1[0]*M1[0] + M1[1]*M1[1] + M1[2]*M1[2]);
        Real mag2 = std::sqrt(M2[0]*M2[0] + M2[1]*M2[1] + M2[2]*M2[2]);
        CHECK_NEAR(mag1, mag2, 1e-12, "Rolling: moment symmetric under R1<->R2 swap");
    }
}

// ============================================================================
// 5. IntersectionAwareContact Tests
// ============================================================================
void test_intersection_aware_contact() {
    std::cout << "\n=== 28e: IntersectionAwareContact ===\n";

    IntersectionAwareContact iac(100, 1.0e6);

    // Test 1: Overlap detection (negative gap)
    {
        Real a[3] = {0.0, 0.0, 0.0};
        Real b[3] = {0.0, 0.0, 0.01};
        Real n[3] = {0.0, 0.0, 1.0};
        Real gap = -0.005;  // Penetrating
        Real overlap = iac.detect_initial_overlap(a, b, n, gap);
        CHECK_NEAR(overlap, 0.005, 1e-12, "Intersection: overlap = |gap| for negative gap");
    }

    // Test 2: Ramp at step 0 = 0
    {
        Real disp = iac.correction_displacement(0.01, 0);
        CHECK_NEAR(disp, 0.0, 1e-15, "Intersection: zero correction at step 0");
    }

    // Test 3: Ramp at final step = full correction
    {
        Real disp = iac.correction_displacement(0.01, 100);
        CHECK_NEAR(disp, 0.01, 1e-12, "Intersection: full correction at final ramp step");
    }

    // Test 4: Force clamping
    {
        Real f = iac.correction_force(0.01, 1.0e9, 1.0e6);
        // stiffness*overlap = 1e9*0.01 = 1e7 > max_force = 1e6
        CHECK_NEAR(f, 1.0e6, 1e-6, "Intersection: force clamped to max");
    }

    // Test 5: Energy tracking
    {
        Real e = iac.energy_injected(1000.0, 0.001);
        CHECK_NEAR(e, 1.0, 1e-10, "Intersection: energy = force * displacement");
    }

    // Test 6: No correction for positive gap
    {
        Real a[3] = {0.0, 0.0, 0.0};
        Real b[3] = {0.0, 0.0, 0.1};
        Real n[3] = {0.0, 0.0, 1.0};
        Real gap = 0.05;  // Separated
        Real overlap = iac.detect_initial_overlap(a, b, n, gap);
        CHECK_NEAR(overlap, 0.0, 1e-15, "Intersection: no overlap for positive gap");
    }

    // Test 7: Ramp fraction at midpoint
    {
        Real frac = iac.ramp_fraction(50);
        CHECK_NEAR(frac, 0.5, 1e-12, "Intersection: ramp fraction = 0.5 at midpoint");
    }

    // Test 8: Max force not exceeded
    {
        // Even with huge stiffness, force should not exceed max
        Real f = iac.correction_force(1.0, 1.0e15, 1.0e6);
        CHECK(f <= 1.0e6 + 1e-6, "Intersection: force does not exceed max_correction_force");
    }

    // Test 9: Ramp complete flag
    {
        CHECK(!iac.is_ramp_complete(50), "Intersection: ramp not complete at step 50");
        CHECK(iac.is_ramp_complete(100), "Intersection: ramp complete at step 100");
        CHECK(iac.is_ramp_complete(200), "Intersection: ramp complete at step 200");
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== Wave 28 Advanced Contact Algorithms Test ===\n";

    test_curved_segment_contact();
    test_thermal_contact_resistance();
    test_wear_model_contact();
    test_rolling_resistance_contact();
    test_intersection_aware_contact();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed) << " tests passed\n";
    return tests_failed > 0 ? 1 : 0;
}
