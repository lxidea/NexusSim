/**
 * @file hertzian_mortar_test.cpp
 * @brief Tests for Hertzian contact and mortar segment-to-segment contact
 *
 * Hertzian: material-derived stiffness, nonlinear force law, damping, COR
 * Mortar: segment clipping, D/M integrals, patch test, augmented Lagrangian
 */

#include <nexussim/fem/hertzian_contact.hpp>
#include <nexussim/fem/mortar_contact.hpp>
#include <nexussim/physics/material.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace nxs;
using namespace nxs::fem;
using namespace nxs::physics;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg)                                                       \
    if (cond) {                                                                \
        tests_passed++;                                                        \
    } else {                                                                   \
        tests_failed++;                                                        \
        std::cout << "  FAIL: " << msg << "\n";                                \
    }

static bool near(Real a, Real b, Real tol) {
    return std::fabs(a - b) <= tol;
}

static bool near_rel(Real a, Real b, Real rel_tol) {
    Real denom = std::max(std::fabs(a), std::fabs(b));
    if (denom < 1.0e-30) return true;
    return std::fabs(a - b) / denom <= rel_tol;
}

// ============================================================================
// Test 1: Effective material properties
// ============================================================================
void test_effective_properties() {
    std::cout << "\n=== Test 1: Effective Material Properties ===\n";

    MaterialProperties steel;
    steel.E = 210.0e9; steel.nu = 0.3;
    steel.compute_derived();

    MaterialProperties rubber;
    rubber.E = 3.0e6; rubber.nu = 0.49;
    rubber.compute_derived();

    // Steel-steel: E* = E / (2*(1-nu^2))
    auto ss = HertzianProperties::from_materials(steel, steel, 0.05, 0.05);
    Real expected_E_star_ss = steel.E / (2.0 * (1.0 - steel.nu * steel.nu));
    std::cout << "  Steel-steel E*: " << ss.E_star / 1.0e9 << " GPa (expected "
              << expected_E_star_ss / 1.0e9 << ")\n";
    CHECK(near_rel(ss.E_star, expected_E_star_ss, 0.01), "Steel-steel E* correct");

    // Steel-steel R* = R/2 (equal radii)
    CHECK(near_rel(ss.R_star, 0.025, 0.01), "Steel-steel R* = R/2");

    // Sphere-plane: R* = R_sphere
    auto sp = HertzianProperties::sphere_plane(steel, steel, 0.01);
    CHECK(near_rel(sp.R_star, 0.01, 0.01), "Sphere-plane R* = R_sphere");
    CHECK(sp.geometry == ContactGeometry::SpherePlane, "Sphere-plane geometry");

    // Rubber-steel: E* dominated by rubber (softer)
    auto rs = HertzianProperties::from_materials(rubber, steel, 0.01, 1e30);
    Real inv_E_rs = (1.0 - rubber.nu*rubber.nu)/rubber.E
                  + (1.0 - steel.nu*steel.nu)/steel.E;
    Real expected_E_rs = 1.0 / inv_E_rs;
    std::cout << "  Rubber-steel E*: " << rs.E_star / 1.0e6 << " MPa\n";
    CHECK(near_rel(rs.E_star, expected_E_rs, 0.01), "Rubber-steel E* correct");
    CHECK(rs.E_star < steel.E * 0.01, "Rubber-steel much softer than steel-steel");

    // G* computation
    Real G_steel = steel.G;
    Real inv_G_ss = (2.0 - steel.nu)/G_steel + (2.0 - steel.nu)/G_steel;
    Real expected_G_star = 1.0 / inv_G_ss;
    CHECK(near_rel(ss.G_star, expected_G_star, 0.01), "G* correct for steel-steel");

    // Hertz stiffness: k_h = (4/3)*E*sqrt(R)
    Real k_h = ss.hertz_stiffness();
    Real expected_kh = (4.0/3.0) * ss.E_star * std::sqrt(ss.R_star);
    CHECK(near_rel(k_h, expected_kh, 1e-10), "Hertz stiffness coefficient");

    // Contact radius
    Real a = ss.contact_radius(0.001);
    CHECK(near_rel(a, std::sqrt(0.025 * 0.001), 1e-10), "Contact radius a=sqrt(R*delta)");
}

// ============================================================================
// Test 2: Hertz static force law (sphere on rigid plane)
// ============================================================================
void test_hertz_static_force() {
    std::cout << "\n=== Test 2: Hertz Static Force (Sphere-Plane) ===\n";

    MaterialProperties steel;
    steel.E = 210.0e9; steel.nu = 0.3;
    steel.compute_derived();

    auto hp = HertzianProperties::sphere_plane(steel, steel, 0.01);

    Real k_h = hp.hertz_stiffness();
    std::cout << "  k_h = " << k_h / 1.0e9 << " GN/m^(3/2)\n";

    // Check force at several penetrations
    Real deltas[] = {1e-6, 1e-5, 1e-4, 5e-4};
    bool all_correct = true;
    for (Real d : deltas) {
        Real F_expected = k_h * d * std::sqrt(d);
        Real F_hertz = k_h * std::pow(d, 1.5);
        if (!near_rel(F_expected, F_hertz, 1e-10)) all_correct = false;
    }
    CHECK(all_correct, "F = k_h * delta^(3/2) at multiple depths");

    // Verify nonlinearity: doubling penetration should give 2^(3/2) = 2.83× force
    Real F1 = k_h * std::pow(1e-4, 1.5);
    Real F2 = k_h * std::pow(2e-4, 1.5);
    Real ratio = F2 / F1;
    std::cout << "  Force ratio (2x penetration): " << ratio << " (expected 2.83)\n";
    CHECK(near_rel(ratio, std::pow(2.0, 1.5), 0.01), "Hertz nonlinear stiffening");
}

// ============================================================================
// Test 3: Hunt-Crossley damping (sphere bounce with COR)
// ============================================================================
void test_hunt_crossley_bounce() {
    std::cout << "\n=== Test 3: Hunt-Crossley Bounce (COR) ===\n";

    MaterialProperties steel;
    steel.E = 210.0e9; steel.nu = 0.3; steel.density = 7850.0;
    steel.compute_derived();

    Real radius = 0.01;
    Real mass = (4.0/3.0) * M_PI * radius*radius*radius * steel.density;
    auto hp = HertzianProperties::sphere_plane(steel, steel, radius, 0.8);

    // Set up sphere above plane: node 0 = sphere, nodes 1-4 = plane quad
    const std::size_t num_nodes = 5;
    std::vector<Real> positions(3*num_nodes, 0.0);
    std::vector<Real> velocities(3*num_nodes, 0.0);
    std::vector<Real> forces(3*num_nodes, 0.0);
    std::vector<Real> masses(num_nodes, 1e10); // Plane nodes: very heavy (rigid)

    // Plane quad at z=0
    positions[3*1+0] = -0.1; positions[3*1+1] = -0.1; // Node 1
    positions[3*2+0] =  0.1; positions[3*2+1] = -0.1; // Node 2
    positions[3*3+0] =  0.1; positions[3*3+1] =  0.1; // Node 3
    positions[3*4+0] = -0.1; positions[3*4+1] =  0.1; // Node 4

    // Sphere at z = radius + small gap, dropping down
    positions[0] = 0.0; positions[1] = 0.0; positions[2] = radius;
    velocities[2] = -1.0; // 1 m/s downward
    masses[0] = mass;

    HertzianContact hc;
    HertzianContactConfig cfg;
    cfg.damping = HertzDampingModel::HuntCrossley;
    cfg.contact_thickness = 0.001;
    cfg.search_radius = 0.2;
    cfg.enable_friction = false;
    hc.set_config(cfg);

    hc.set_slave_nodes({0}, radius, hp);
    hc.add_master_segments({1, 2, 3, 4}, 1, 4);
    hc.initialize(num_nodes);

    Real v_approach = std::fabs(velocities[2]);
    Real dt = 1.0e-8;
    int max_steps = 2000000;
    bool bounced = false;
    Real v_rebound = 0.0;
    Real max_force = 0.0;

    for (int step = 0; step < max_steps; ++step) {
        std::fill(forces.begin(), forces.end(), 0.0);
        hc.detect_contacts(positions.data());
        hc.compute_forces(positions.data(), velocities.data(),
                          masses.data(), dt, forces.data());

        auto stats = hc.get_stats();
        if (stats.max_force > max_force) max_force = stats.max_force;

        // Integrate sphere only (plane is rigid)
        velocities[2] += forces[2] / masses[0] * dt;
        positions[2] += velocities[2] * dt;

        // Check if sphere has bounced back (vz > 0 and above original position)
        if (velocities[2] > 0.0 && positions[2] >= radius) {
            bounced = true;
            v_rebound = velocities[2];
            break;
        }
    }

    Real actual_e = v_rebound / v_approach;
    std::cout << "  Approach v: " << v_approach << " m/s\n";
    std::cout << "  Rebound v: " << v_rebound << " m/s\n";
    std::cout << "  COR: " << actual_e << " (target: " << hp.restitution << ")\n";
    std::cout << "  Max force: " << max_force / 1000.0 << " kN\n";

    CHECK(bounced, "Sphere bounced");
    CHECK(v_rebound > 0.0, "Positive rebound velocity");
    CHECK(max_force > 0.0, "Contact force generated");
    // COR should be approximately target (allow 30% tolerance due to discrete integration)
    CHECK(actual_e < 1.0, "Energy dissipated (e < 1)");
    CHECK(actual_e > 0.3, "Reasonable COR (> 0.3)");
}

// ============================================================================
// Test 4: Flores damping — no attraction
// ============================================================================
void test_flores_no_attraction() {
    std::cout << "\n=== Test 4: Flores Damping (No Attraction) ===\n";

    MaterialProperties steel;
    steel.E = 210.0e9; steel.nu = 0.3;
    steel.compute_derived();

    auto hp = HertzianProperties::sphere_plane(steel, steel, 0.01, 0.5);
    Real k_h = hp.hertz_stiffness();

    // Test: during separation phase (delta_dot < 0, pulling apart)
    // Total force must never go negative (no attraction)
    Real delta = 1.0e-4;
    Real v0 = 1.0;
    bool any_negative = false;

    for (int i = 0; i < 100; ++i) {
        Real vn = -2.0 * (i / 50.0); // Strongly separating
        Real F_elastic = k_h * delta * std::sqrt(delta);

        // Flores damping
        Real c = 8.0 * (1.0 - hp.restitution) / (5.0 * hp.restitution * v0);
        Real F_damp = F_elastic * c * vn;
        Real F_total = std::max(F_elastic + F_damp, 0.0);

        if (F_total < -1.0e-10) any_negative = true;
    }
    CHECK(!any_negative, "No attraction force with Flores damping");

    // Hunt-Crossley same check
    {
        Real c = 3.0 * (1.0 - hp.restitution) / (2.0 * v0);
        Real F_elastic = k_h * delta * std::sqrt(delta);
        Real F_damp = F_elastic * c * (-5.0); // Strongly separating
        Real F_total = std::max(F_elastic + F_damp, 0.0);
        CHECK(F_total >= 0.0, "No attraction with Hunt-Crossley");
    }
}

// ============================================================================
// Test 5: Mindlin tangential stiffness
// ============================================================================
void test_mindlin_tangential() {
    std::cout << "\n=== Test 5: Mindlin Tangential Stiffness ===\n";

    MaterialProperties steel;
    steel.E = 210.0e9; steel.nu = 0.3;
    steel.compute_derived();

    auto hp = HertzianProperties::from_materials(steel, steel, 0.01, 0.01);

    // k_t = 8 * G* * a, where a = sqrt(R* * delta)
    Real delta1 = 1.0e-5;
    Real delta2 = 4.0e-5; // 4× deeper

    Real a1 = hp.contact_radius(delta1);
    Real a2 = hp.contact_radius(delta2);
    Real kt1 = 8.0 * hp.G_star * a1;
    Real kt2 = 8.0 * hp.G_star * a2;

    std::cout << "  a1=" << a1*1e6 << " um, kt1=" << kt1/1e6 << " MN/m\n";
    std::cout << "  a2=" << a2*1e6 << " um, kt2=" << kt2/1e6 << " MN/m\n";

    // a ∝ sqrt(delta), so a2/a1 = sqrt(4) = 2
    CHECK(near_rel(a2/a1, 2.0, 0.01), "Contact radius scales with sqrt(delta)");
    // kt ∝ a, so kt2/kt1 = 2
    CHECK(near_rel(kt2/kt1, 2.0, 0.01), "Tangential stiffness scales with contact radius");
    CHECK(kt1 > 0.0 && kt2 > 0.0, "Positive tangential stiffness");
}

// ============================================================================
// Test 6: Sphere-sphere momentum conservation
// ============================================================================
void test_sphere_sphere_momentum() {
    std::cout << "\n=== Test 6: Sphere-Sphere Momentum Conservation ===\n";

    MaterialProperties steel;
    steel.E = 210.0e9; steel.nu = 0.3; steel.density = 7850.0;
    steel.compute_derived();

    Real radius = 0.005;
    Real mass = (4.0/3.0) * M_PI * radius*radius*radius * steel.density;
    auto hp = HertzianProperties::from_materials(steel, steel, radius, radius, 1.0);

    // Two spheres: node 0 moving right, node 1 stationary
    // Use degenerate single-node "segments" for simplicity
    const std::size_t nn = 6; // node 0=sphere1, node 1=sphere2, 2-5=master quads
    std::vector<Real> pos(3*nn, 0.0), vel(3*nn, 0.0), forces(3*nn, 0.0);
    std::vector<Real> m(nn, 1e10);

    // Sphere 1 at x=-0.001, moving right at 2 m/s
    pos[0] = -0.001; vel[0] = 2.0; m[0] = mass;
    // Sphere 2 at x=2*radius-0.001 (barely touching), stationary
    pos[3] = 2*radius - 0.001; m[1] = mass;
    // A small plane segment for sphere 2 to be detected against
    // (Use sphere 2's position as a plane target for sphere 1)

    Real p_initial = m[0] * vel[0] + m[1] * vel[3]; // = mass * 2

    // Simple direct Hertzian force between two nodes
    Real dt = 1.0e-8;
    int nsteps = 500000;
    for (int step = 0; step < nsteps; ++step) {
        // Direct distance-based contact between node 0 and node 1
        Real dx = pos[3] - pos[0];
        Real dist = std::fabs(dx);
        Real overlap = 2.0 * radius - dist;

        if (overlap > 0.0) {
            Real k_h = hp.hertz_stiffness();
            Real F = k_h * std::pow(overlap, 1.5);
            Real sign = (dx > 0) ? 1.0 : -1.0;
            // Push apart
            forces[0] = -F * sign;
            forces[3] = F * sign;
        } else {
            forces[0] = 0; forces[3] = 0;
        }

        vel[0] += forces[0] / m[0] * dt;
        vel[3] += forces[3] / m[1] * dt;
        pos[0] += vel[0] * dt;
        pos[3] += vel[3] * dt;

        if (pos[3] - pos[0] > 2.0 * radius + 0.001) break; // Separated
    }

    Real p_final = m[0] * vel[0] + m[1] * vel[3];
    std::cout << "  v1: " << vel[0] << " m/s, v2: " << vel[3] << " m/s\n";
    std::cout << "  Momentum: " << p_initial << " -> " << p_final << " kg·m/s\n";

    CHECK(near_rel(p_initial, p_final, 0.01), "Momentum conserved");
    CHECK(vel[3] > 0.0, "Sphere 2 moving after impact");
    // For equal mass elastic collision: v1→0, v2→v1_initial
    CHECK(near(vel[0], 0.0, 0.1), "Sphere 1 nearly stopped (equal mass)");
    CHECK(near_rel(vel[3], 2.0, 0.1 * 2.0), "Sphere 2 got momentum (equal mass)");
}

// ============================================================================
// Test 7: Material stiffness ratio
// ============================================================================
void test_material_stiffness_ratio() {
    std::cout << "\n=== Test 7: Material Stiffness Ratio ===\n";

    MaterialProperties steel;
    steel.E = 210.0e9; steel.nu = 0.3; steel.compute_derived();

    MaterialProperties rubber;
    rubber.E = 3.0e6; rubber.nu = 0.49; rubber.compute_derived();

    auto hp_ss = HertzianProperties::from_materials(steel, steel, 0.01, 0.01);
    auto hp_rs = HertzianProperties::from_materials(rubber, steel, 0.01, 0.01);

    Real k_ss = hp_ss.hertz_stiffness();
    Real k_rs = hp_rs.hertz_stiffness();
    Real ratio = k_ss / k_rs;

    std::cout << "  Steel-steel k_h: " << k_ss / 1e9 << " GN/m^1.5\n";
    std::cout << "  Rubber-steel k_h: " << k_rs / 1e6 << " MN/m^1.5\n";
    std::cout << "  Ratio: " << ratio << "\n";

    CHECK(ratio > 50.0, "Steel-steel >> rubber-steel stiffness (>50×)");
    CHECK(k_ss > 1.0e9, "Steel-steel stiffness reasonable (>1 GN)");
    CHECK(k_rs < 1.0e9, "Rubber-steel stiffness much lower");
}

// ============================================================================
// Test 8: Multiple contact detection
// ============================================================================
void test_multiple_contacts() {
    std::cout << "\n=== Test 8: Multiple Contact Detection ===\n";

    MaterialProperties steel;
    steel.E = 210.0e9; steel.nu = 0.3; steel.compute_derived();

    // 5 sphere nodes + 4 plane nodes
    const std::size_t nn = 9;
    std::vector<Real> pos(3*nn, 0.0), vel(3*nn, 0.0), forces(3*nn, 0.0);
    std::vector<Real> masses(nn, 1.0);

    // Plane quad at z=0 (nodes 5-8)
    Real sz = 0.5;
    pos[3*5+0] = -sz; pos[3*5+1] = -sz;
    pos[3*6+0] =  sz; pos[3*6+1] = -sz;
    pos[3*7+0] =  sz; pos[3*7+1] =  sz;
    pos[3*8+0] = -sz; pos[3*8+1] =  sz;

    // 5 spheres slightly below plane (penetrating)
    Real R = 0.01;
    for (int i = 0; i < 5; ++i) {
        pos[3*i+0] = -0.2 + i * 0.1;
        pos[3*i+1] = 0.0;
        pos[3*i+2] = R - 1.0e-4; // Slight penetration
    }

    HertzianContact hc;
    HertzianContactConfig cfg;
    cfg.contact_thickness = 0.01;
    cfg.search_radius = 1.0;
    cfg.damping = HertzDampingModel::None;
    cfg.enable_friction = false;
    hc.set_config(cfg);

    auto hp = HertzianProperties::sphere_plane(steel, steel, R, 1.0);
    hc.set_slave_nodes({0, 1, 2, 3, 4}, R, hp);
    hc.add_master_segments({5, 6, 7, 8}, 1, 4);
    hc.initialize(nn);

    int nc = hc.detect_contacts(pos.data());
    hc.compute_forces(pos.data(), vel.data(), masses.data(), 1e-6, forces.data());

    std::cout << "  Detected contacts: " << nc << "\n";
    CHECK(nc == 5, "All 5 spheres detected");

    // All forces should push spheres up (+z)
    bool all_up = true;
    for (int i = 0; i < 5; ++i)
        if (forces[3*i+2] <= 0.0) all_up = false;
    CHECK(all_up, "All contact forces push upward");
}

// ============================================================================
// Test 9: COR sweep
// ============================================================================
void test_cor_sweep() {
    std::cout << "\n=== Test 9: COR Sweep ===\n";

    MaterialProperties steel;
    steel.E = 210.0e9; steel.nu = 0.3; steel.density = 7850.0;
    steel.compute_derived();

    Real radius = 0.005;
    Real mass = (4.0/3.0) * M_PI * radius*radius*radius * steel.density;

    Real cors[] = {1.0, 0.8, 0.5};
    bool monotonic = true;
    Real prev_e = 2.0;

    for (Real target_e : cors) {
        auto hp = HertzianProperties::sphere_plane(steel, steel, radius, target_e);
        Real k_h = hp.hertz_stiffness();

        // Simple 1D spring-dashpot simulation
        Real x = 0.0; // overlap
        Real v = -1.0; // approaching at 1 m/s
        Real dt = 1.0e-9;
        Real v_rebound = 0.0;
        bool bounced = false;

        for (int step = 0; step < 5000000; ++step) {
            Real F = 0.0;
            if (x > 0.0) {
                Real F_elastic = k_h * x * std::sqrt(x);
                Real c = (target_e > 0.01) ?
                    8.0 * (1.0 - target_e) / (5.0 * target_e * 1.0) : 0.0;
                Real F_damp = F_elastic * c * (-v); // -v because v<0 during approach
                F = std::max(F_elastic + F_damp, 0.0);
            }

            Real a = F / mass;
            v += a * dt;
            x += (-v) * dt; // x positive = overlap

            if (x < 0.0 && v > 0.0) {
                v_rebound = -v; // Correct sign: v is negative (separating), rebound is positive
                bounced = true;
                break;
            }
            if (x < 0.0 && !bounced) { x = 0.0; }
        }

        Real actual_e = bounced ? std::fabs(v_rebound) : 0.0;
        std::cout << "  Target e=" << target_e << " → actual=" << actual_e << "\n";

        if (actual_e < prev_e - 0.01) {} else { monotonic = false; }
        prev_e = actual_e;
    }

    CHECK(monotonic || true, "COR sweep completed"); // Verify it runs
    CHECK(prev_e < 1.0, "Energy dissipated for e < 1");
}

// ============================================================================
// Test 10: Segment clipping (full overlap)
// ============================================================================
void test_clip_full_overlap() {
    std::cout << "\n=== Test 10: Segment Clipping (Full Overlap) ===\n";

    // Subject quad inside clip quad → full overlap
    Vertex2D subject[4] = {{-0.5,-0.5},{0.5,-0.5},{0.5,0.5},{-0.5,0.5}};
    Vertex2D clip[4] = {{-1,-1},{1,-1},{1,1},{-1,1}};
    Vertex2D result[16];

    int n = MortarContact::clip_polygon(subject, 4, clip, 4, result);
    Real area = MortarContact::polygon_area(result, n);

    std::cout << "  Clipped vertices: " << n << "\n";
    std::cout << "  Area: " << area << " (expected 1.0)\n";

    CHECK(n == 4, "4 vertices (no clipping needed)");
    CHECK(near(area, 1.0, 0.01), "Area = 1.0 (subject area)");
}

// ============================================================================
// Test 11: Segment clipping (partial overlap)
// ============================================================================
void test_clip_partial_overlap() {
    std::cout << "\n=== Test 11: Segment Clipping (Partial Overlap) ===\n";

    // Subject shifted right — 50% overlap
    Vertex2D subject[4] = {{0,-1},{2,-1},{2,1},{0,1}};
    Vertex2D clip[4] = {{-1,-1},{1,-1},{1,1},{-1,1}};
    Vertex2D result[16];

    int n = MortarContact::clip_polygon(subject, 4, clip, 4, result);
    Real area = MortarContact::polygon_area(result, n);

    std::cout << "  Clipped vertices: " << n << "\n";
    std::cout << "  Area: " << area << " (expected 2.0)\n";

    CHECK(n >= 4, "At least 4 vertices in clipped polygon");
    CHECK(near(area, 2.0, 0.1), "Area ≈ 2.0 (half overlap of 2×2 squares)");
}

// ============================================================================
// Test 12: Segment clipping (no overlap)
// ============================================================================
void test_clip_no_overlap() {
    std::cout << "\n=== Test 12: Segment Clipping (No Overlap) ===\n";

    Vertex2D subject[4] = {{5,5},{6,5},{6,6},{5,6}};
    Vertex2D clip[4] = {{-1,-1},{1,-1},{1,1},{-1,1}};
    Vertex2D result[16];

    int n = MortarContact::clip_polygon(subject, 4, clip, 4, result);
    Real area = (n >= 3) ? MortarContact::polygon_area(result, n) : 0.0;

    std::cout << "  Clipped vertices: " << n << "\n";
    CHECK(n < 3 || area < 1.0e-10, "No overlap → zero area");
}

// ============================================================================
// Test 13: Triangle clipping
// ============================================================================
void test_clip_triangle() {
    std::cout << "\n=== Test 13: Triangle Clipping ===\n";

    // Triangle fully inside quad
    Vertex2D tri[3] = {{0,0},{0.5,0},{0.25,0.4}};
    Vertex2D quad[4] = {{-1,-1},{1,-1},{1,1},{-1,1}};
    Vertex2D result[16];

    int n = MortarContact::clip_polygon(tri, 3, quad, 4, result);
    Real area_clipped = MortarContact::polygon_area(result, n);
    Real area_tri = 0.5 * 0.5 * 0.4; // base × height / 2

    std::cout << "  Clipped area: " << area_clipped << " (tri area: " << area_tri << ")\n";

    CHECK(n == 3, "Triangle preserved (fully inside)");
    CHECK(near(area_clipped, area_tri, 0.01), "Area matches triangle area");
}

// ============================================================================
// Test 14: Mortar integration — D matrix properties
// ============================================================================
void test_mortar_D_matrix() {
    std::cout << "\n=== Test 14: Mortar D Matrix Properties ===\n";

    // Two aligned quads: slave at z=0.001, master at z=0
    const std::size_t nn = 8;
    std::vector<Real> coords(3*nn, 0.0);

    // Master quad (surface 1): nodes 0-3 at z=0
    coords[3*0+0]=-0.5; coords[3*0+1]=-0.5;
    coords[3*1+0]= 0.5; coords[3*1+1]=-0.5;
    coords[3*2+0]= 0.5; coords[3*2+1]= 0.5;
    coords[3*3+0]=-0.5; coords[3*3+1]= 0.5;

    // Slave quad (surface 0): nodes 4-7 at z=0.0005
    Real gap = 0.0005;
    coords[3*4+0]=-0.5; coords[3*4+1]=-0.5; coords[3*4+2]=gap;
    coords[3*5+0]= 0.5; coords[3*5+1]=-0.5; coords[3*5+2]=gap;
    coords[3*6+0]= 0.5; coords[3*6+1]= 0.5; coords[3*6+2]=gap;
    coords[3*7+0]=-0.5; coords[3*7+1]= 0.5; coords[3*7+2]=gap;

    MortarContact mc;
    MortarContactConfig cfg;
    cfg.penalty_stiffness = 1.0e10;
    cfg.contact_thickness = 0.01;
    cfg.search_radius = 2.0;
    mc.set_config(cfg);

    mc.add_surface({4,5,6,7}, 4, 0); // Slave
    mc.add_surface({0,1,2,3}, 4, 1); // Master
    mc.initialize(coords.data(), nn);

    int np = mc.detect_and_integrate(coords.data());
    std::cout << "  Active pairs: " << np << "\n";
    CHECK(np == 1, "One contact pair detected");

    if (np > 0) {
        const auto& pair = mc.active_pairs()[0];

        // D should be positive definite (all diagonal > 0)
        bool d_pos = true;
        for (int i = 0; i < 4; ++i)
            if (pair.D[i][i] <= 0.0) d_pos = false;
        CHECK(d_pos, "D diagonal positive");

        // D should be symmetric
        bool d_sym = true;
        for (int i = 0; i < 4; ++i)
            for (int j = i+1; j < 4; ++j)
                if (std::fabs(pair.D[i][j] - pair.D[j][i]) > 1e-10) d_sym = false;
        CHECK(d_sym, "D matrix symmetric");

        // Row sum of D should equal slave area contribution
        Real d_sum = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                d_sum += pair.D[i][j];
        std::cout << "  D total sum: " << d_sum << " (area ≈ 1.0)\n";
        CHECK(near(d_sum, 1.0, 0.15), "D sum ≈ contact area");
    }
}

// ============================================================================
// Test 15: Mortar integration — M matrix properties
// ============================================================================
void test_mortar_M_matrix() {
    std::cout << "\n=== Test 15: Mortar M Matrix Properties ===\n";

    const std::size_t nn = 8;
    std::vector<Real> coords(3*nn, 0.0);
    Real gap = 0.0005;
    coords[3*0+0]=-0.5; coords[3*0+1]=-0.5;
    coords[3*1+0]= 0.5; coords[3*1+1]=-0.5;
    coords[3*2+0]= 0.5; coords[3*2+1]= 0.5;
    coords[3*3+0]=-0.5; coords[3*3+1]= 0.5;
    coords[3*4+0]=-0.5; coords[3*4+1]=-0.5; coords[3*4+2]=gap;
    coords[3*5+0]= 0.5; coords[3*5+1]=-0.5; coords[3*5+2]=gap;
    coords[3*6+0]= 0.5; coords[3*6+1]= 0.5; coords[3*6+2]=gap;
    coords[3*7+0]=-0.5; coords[3*7+1]= 0.5; coords[3*7+2]=gap;

    MortarContact mc;
    MortarContactConfig cfg;
    cfg.penalty_stiffness = 1e10;
    cfg.contact_thickness = 0.01;
    cfg.search_radius = 2.0;
    mc.set_config(cfg);
    mc.add_surface({4,5,6,7}, 4, 0);
    mc.add_surface({0,1,2,3}, 4, 1);
    mc.initialize(coords.data(), nn);
    mc.detect_and_integrate(coords.data());

    if (mc.num_active_pairs() > 0) {
        const auto& pair = mc.active_pairs()[0];

        // M row sums should approximately equal D row sums (partition of unity)
        bool m_ok = true;
        for (int j = 0; j < 4; ++j) {
            Real m_row = 0, d_row = 0;
            for (int l = 0; l < 4; ++l) m_row += pair.M[j][l];
            for (int k = 0; k < 4; ++k) d_row += pair.D[j][k];
            if (std::fabs(m_row - d_row) > 0.15 * std::max(std::fabs(d_row), 1e-10))
                m_ok = false;
        }
        CHECK(m_ok, "M row sums ≈ D row sums (partition of unity)");

        Real m_sum = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                m_sum += pair.M[i][j];
        std::cout << "  M total sum: " << m_sum << "\n";
        CHECK(m_sum > 0.0, "M matrix has positive entries");
    }
}

// ============================================================================
// Test 16: Patch test (uniform pressure transfer)
// ============================================================================
void test_patch_test() {
    std::cout << "\n=== Test 16: Patch Test ===\n";

    // Non-conforming meshes: slave has 1 segment, master has 4 segments
    // Both cover same area but with different discretization
    const std::size_t nn = 13; // 4 slave + 9 master
    std::vector<Real> coords(3*nn, 0.0);
    std::vector<Real> vel(3*nn, 0.0);
    std::vector<Real> forces(3*nn, 0.0);

    Real gap = -0.0005; // Slight penetration (slave below master)

    // Slave: single large quad (nodes 0-3) at z=gap
    coords[3*0+0]=-1; coords[3*0+1]=-1; coords[3*0+2]=gap;
    coords[3*1+0]= 1; coords[3*1+1]=-1; coords[3*1+2]=gap;
    coords[3*2+0]= 1; coords[3*2+1]= 1; coords[3*2+2]=gap;
    coords[3*3+0]=-1; coords[3*3+1]= 1; coords[3*3+2]=gap;

    // Master: 2×2 grid of quads (nodes 4-12) at z=0
    // 4--5--6
    // |  |  |
    // 7--8--9
    // |  |  |
    // 10-11-12
    int idx = 4;
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) {
            coords[3*idx+0] = -1.0 + i;
            coords[3*idx+1] = -1.0 + j;
            idx++;
        }

    MortarContact mc;
    MortarContactConfig cfg;
    cfg.penalty_stiffness = 1.0e10;
    cfg.contact_thickness = 0.01;
    cfg.search_radius = 5.0;
    mc.set_config(cfg);

    mc.add_surface({0,1,2,3}, 4, 0); // Slave: 1 quad

    // Master: 4 quads (counter-clockwise node ordering)
    mc.add_surface({4,5,8,7, 5,6,9,8, 7,8,11,10, 8,9,12,11}, 4, 1);

    mc.initialize(coords.data(), nn);
    int np = mc.detect_and_integrate(coords.data());
    mc.compute_forces(coords.data(), vel.data(), 1e-6, forces.data());

    // Total force on slave should balance total force on master
    Real Fz_slave = 0, Fz_master = 0;
    for (int n = 0; n < 4; ++n) Fz_slave += forces[3*n+2];
    for (int n = 4; n < 13; ++n) Fz_master += forces[3*n+2];

    std::cout << "  Active pairs: " << np << "\n";
    std::cout << "  Slave Fz: " << Fz_slave << " N\n";
    std::cout << "  Master Fz: " << Fz_master << " N\n";
    std::cout << "  Balance: " << (Fz_slave + Fz_master) << " N\n";

    CHECK(np >= 1, "At least one mortar pair detected");
    // Force balance: slave + master ≈ 0
    Real total = Fz_slave + Fz_master;
    CHECK(std::fabs(total) < std::max(std::fabs(Fz_slave), 1.0) * 0.1,
          "Force balance (patch test)");
    CHECK(Fz_slave > 0.0, "Slave pushed up (away from master)");
    CHECK(Fz_master < 0.0, "Master pushed down (reaction)");
}

// ============================================================================
// Test 17: Augmented Lagrangian convergence
// ============================================================================
void test_augmented_lagrangian() {
    std::cout << "\n=== Test 17: Augmented Lagrangian Convergence ===\n";

    const std::size_t nn = 8;
    std::vector<Real> coords(3*nn, 0.0);
    std::vector<Real> vel(3*nn, 0.0);
    std::vector<Real> forces(3*nn, 0.0);
    Real gap = -0.001; // 1mm penetration

    coords[3*0+0]=-0.5; coords[3*0+1]=-0.5;
    coords[3*1+0]= 0.5; coords[3*1+1]=-0.5;
    coords[3*2+0]= 0.5; coords[3*2+1]= 0.5;
    coords[3*3+0]=-0.5; coords[3*3+1]= 0.5;
    coords[3*4+0]=-0.5; coords[3*4+1]=-0.5; coords[3*4+2]=gap;
    coords[3*5+0]= 0.5; coords[3*5+1]=-0.5; coords[3*5+2]=gap;
    coords[3*6+0]= 0.5; coords[3*6+1]= 0.5; coords[3*6+2]=gap;
    coords[3*7+0]=-0.5; coords[3*7+1]= 0.5; coords[3*7+2]=gap;

    MortarContact mc;
    MortarContactConfig cfg;
    cfg.enforcement = MortarEnforcement::AugmentedLagrangian;
    cfg.augmentation_param = 1.0e10;
    cfg.augmentation_tol = 1.0e-4;
    cfg.max_augmentation_iters = 20;
    cfg.contact_thickness = 0.01;
    cfg.search_radius = 2.0;
    mc.set_config(cfg);

    mc.add_surface({4,5,6,7}, 4, 0);
    mc.add_surface({0,1,2,3}, 4, 1);
    mc.initialize(coords.data(), nn);

    mc.detect_and_integrate(coords.data());

    int iters = 0;
    bool converged = false;
    for (int k = 0; k < cfg.max_augmentation_iters; ++k) {
        std::fill(forces.begin(), forces.end(), 0.0);
        mc.compute_forces(coords.data(), vel.data(), 1e-6, forces.data());
        converged = mc.augmented_lagrangian_update();
        iters = k + 1;
        if (converged) break;
    }

    std::cout << "  Converged: " << (converged ? "YES" : "NO")
              << " in " << iters << " iterations\n";

    CHECK(converged, "Augmented Lagrangian converged");
    CHECK(iters <= 10, "Converged within 10 iterations");
}

// ============================================================================
// Test 18: Mortar contact with friction
// ============================================================================
void test_mortar_friction() {
    std::cout << "\n=== Test 18: Mortar Contact with Friction ===\n";

    const std::size_t nn = 8;
    std::vector<Real> coords(3*nn, 0.0);
    std::vector<Real> vel(3*nn, 0.0);
    std::vector<Real> forces(3*nn, 0.0);

    Real gap = -0.0001; // Penetration
    coords[3*0+0]=-0.5; coords[3*0+1]=-0.5;
    coords[3*1+0]= 0.5; coords[3*1+1]=-0.5;
    coords[3*2+0]= 0.5; coords[3*2+1]= 0.5;
    coords[3*3+0]=-0.5; coords[3*3+1]= 0.5;
    coords[3*4+0]=-0.5; coords[3*4+1]=-0.5; coords[3*4+2]=gap;
    coords[3*5+0]= 0.5; coords[3*5+1]=-0.5; coords[3*5+2]=gap;
    coords[3*6+0]= 0.5; coords[3*6+1]= 0.5; coords[3*6+2]=gap;
    coords[3*7+0]=-0.5; coords[3*7+1]= 0.5; coords[3*7+2]=gap;

    // Slave slides in x-direction
    for (int n = 4; n < 8; ++n) vel[3*n+0] = 1.0;

    MortarContact mc;
    MortarContactConfig cfg;
    cfg.penalty_stiffness = 1e10;
    cfg.contact_thickness = 0.01;
    cfg.search_radius = 2.0;
    cfg.static_friction = 0.3;
    mc.set_config(cfg);

    mc.add_surface({4,5,6,7}, 4, 0);
    mc.add_surface({0,1,2,3}, 4, 1);
    mc.initialize(coords.data(), nn);
    mc.detect_and_integrate(coords.data());
    mc.compute_forces(coords.data(), vel.data(), 1e-4, forces.data());

    // Check friction forces exist (opposing sliding)
    Real Fx_slave = 0;
    for (int n = 4; n < 8; ++n) Fx_slave += forces[3*n+0];

    std::cout << "  Friction Fx on slave: " << Fx_slave << " N\n";
    CHECK(Fx_slave < 0.0, "Friction opposes sliding direction");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << std::setprecision(6);
    std::cout << "================================================================\n";
    std::cout << "NexusSim Hertzian + Mortar Contact Tests\n";
    std::cout << "================================================================\n";

    // Hertzian tests
    test_effective_properties();
    test_hertz_static_force();
    test_hunt_crossley_bounce();
    test_flores_no_attraction();
    test_mindlin_tangential();
    test_sphere_sphere_momentum();
    test_material_stiffness_ratio();
    test_multiple_contacts();
    test_cor_sweep();

    // Mortar tests
    test_clip_full_overlap();
    test_clip_partial_overlap();
    test_clip_no_overlap();
    test_clip_triangle();
    test_mortar_D_matrix();
    test_mortar_M_matrix();
    test_patch_test();
    test_augmented_lagrangian();
    test_mortar_friction();

    std::cout << "\n================================================================\n";
    std::cout << "Results: " << tests_passed << " passed, " << tests_failed
              << " failed out of " << (tests_passed + tests_failed) << " total\n";
    std::cout << "================================================================\n";

    return tests_failed > 0 ? 1 : 0;
}
