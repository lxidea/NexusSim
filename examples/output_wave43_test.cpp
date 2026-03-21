/**
 * @file output_wave43_test.cpp
 * @brief Tests for Wave 43 per-entity output extractors.
 *
 * Tests cover:
 *  - NodeResultExtractor: displacement, velocity, acceleration, reaction forces
 *  - ShellResultExtractor: von Mises, thickness reduction, fiber stress
 *  - SolidResultExtractor: von Mises, principal stresses, pressure, plastic strain
 *  - SPHResultExtractor: density, smoothing length, pressure
 *  - BeamResultExtractor: axial force, bending moment
 *  - RigidBodyExtractor: CoM position, velocity, angular velocity
 *  - InterfaceForceExtractor: contact force, contact gap
 *  - CrackResultExtractor: crack length, SIF, crack angle
 *  - SectionForceExtractor: section force, section moment
 *  - OutputDispatcher: registration, dispatch, has_extractor, empty dispatch
 *  - Edge cases: zero/empty data, mismatched sizes
 */

#include <nexussim/io/output_wave43.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>

using namespace nxs::io;

// ============================================================================
// Test infrastructure
// ============================================================================

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
// Helpers
// ============================================================================

static double von_mises_ref(double sxx, double syy, double szz,
                             double sxy, double syz, double sxz) {
    return std::sqrt(0.5 * (
        (sxx - syy) * (sxx - syy) +
        (syy - szz) * (syy - szz) +
        (szz - sxx) * (szz - sxx) +
        6.0 * (sxy * sxy + syz * syz + sxz * sxz)
    ));
}

// ============================================================================
// Test: ResultField basic structure
// ============================================================================

static void test_result_field_structure() {
    ResultField rf;
    rf.name = "TestField";
    rf.type = ResultFieldType::Displacement;
    rf.num_components = 3;
    rf.data = {1.0, 2.0, 3.0,  4.0, 5.0, 6.0};

    CHECK(rf.name == "TestField", "ResultField name");
    CHECK(rf.type == ResultFieldType::Displacement, "ResultField type");
    CHECK(rf.num_components == 3, "ResultField num_components");
    CHECK(rf.num_entities() == 2, "ResultField num_entities");
    CHECK_NEAR(rf.scalar_at(0), 1.0, 1e-12, "ResultField scalar_at(0)");
    CHECK_NEAR(rf.component_at(0, 0), 1.0, 1e-12, "component_at(0,0)");
    CHECK_NEAR(rf.component_at(0, 1), 2.0, 1e-12, "component_at(0,1)");
    CHECK_NEAR(rf.component_at(0, 2), 3.0, 1e-12, "component_at(0,2)");
    CHECK_NEAR(rf.component_at(1, 0), 4.0, 1e-12, "component_at(1,0)");
    CHECK_NEAR(rf.component_at(1, 2), 6.0, 1e-12, "component_at(1,2)");
}

// ============================================================================
// Test: NodeResultExtractor
// ============================================================================

static void test_node_displacement() {
    // 3 nodes, known displacement = {1,2,3, 4,5,6, 7,8,9}
    std::vector<Real> pos     = {2.0, 3.0, 4.0,  5.0, 6.0, 7.0,  8.0, 9.0, 10.0};
    std::vector<Real> ref_pos = {1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0,  1.0};

    ResultField rf = NodeResultExtractor::extract_displacement(pos, ref_pos, 3);

    CHECK(rf.type == ResultFieldType::Displacement, "Node disp type");
    CHECK(rf.num_components == 3, "Node disp num_components");
    CHECK(rf.num_entities() == 3, "Node disp num_entities");
    CHECK_NEAR(rf.component_at(0, 0), 1.0, 1e-12, "Node disp[0] x");
    CHECK_NEAR(rf.component_at(0, 1), 2.0, 1e-12, "Node disp[0] y");
    CHECK_NEAR(rf.component_at(0, 2), 3.0, 1e-12, "Node disp[0] z");
    CHECK_NEAR(rf.component_at(2, 0), 7.0, 1e-12, "Node disp[2] x");
    CHECK_NEAR(rf.component_at(2, 2), 9.0, 1e-12, "Node disp[2] z");
}

static void test_node_displacement_zero_ref() {
    // Zero reference → displacement equals position
    std::vector<Real> pos = {3.0, 0.0, 0.0};
    std::vector<Real> ref;  // empty → treated as zero

    ResultField rf = NodeResultExtractor::extract_displacement(pos, ref, 1);
    CHECK_NEAR(rf.component_at(0, 0), 3.0, 1e-12, "Node disp zero-ref x");
    CHECK_NEAR(rf.component_at(0, 1), 0.0, 1e-12, "Node disp zero-ref y");
    CHECK_NEAR(rf.component_at(0, 2), 0.0, 1e-12, "Node disp zero-ref z");
}

static void test_node_velocity() {
    std::vector<Real> vel = {1.5, 2.5, 3.5,  -1.0, 0.0, 1.0};
    ResultField rf = NodeResultExtractor::extract_velocity(vel, 2);

    CHECK(rf.type == ResultFieldType::Velocity, "Node vel type");
    CHECK(rf.num_components == 3, "Node vel components");
    CHECK_NEAR(rf.component_at(0, 0), 1.5, 1e-12, "Node vel[0] vx");
    CHECK_NEAR(rf.component_at(1, 1), 0.0, 1e-12, "Node vel[1] vy");
    CHECK_NEAR(rf.component_at(1, 2), 1.0, 1e-12, "Node vel[1] vz");
}

static void test_node_acceleration() {
    std::vector<Real> acc = {9.81, 0.0, 0.0};
    ResultField rf = NodeResultExtractor::extract_acceleration(acc, 1);

    CHECK(rf.type == ResultFieldType::Acceleration, "Node acc type");
    CHECK_NEAR(rf.component_at(0, 0), 9.81, 1e-10, "Node acc[0] ax");
    CHECK_NEAR(rf.component_at(0, 1), 0.0,  1e-12, "Node acc[0] ay");
}

static void test_node_reaction_forces() {
    // 4 nodes; nodes 1 and 3 are constrained
    std::vector<Real> forces = {
        10.0, 20.0, 30.0,   // node 0 (not constrained, should be zero in output)
        -5.0, -6.0, -7.0,   // node 1 (constrained)
         0.0,  0.0,  0.0,   // node 2
         3.0,  4.0,  5.0    // node 3 (constrained)
    };
    std::vector<int> constrained = {1, 3};

    ResultField rf = NodeResultExtractor::extract_reaction_forces(forces, constrained, 4);
    CHECK(rf.type == ResultFieldType::ReactionForce, "Reaction force type");
    // Node 0 should be zero (not constrained)
    CHECK_NEAR(rf.component_at(0, 0), 0.0, 1e-12, "Reaction unconstrained node 0");
    // Node 1 constrained
    CHECK_NEAR(rf.component_at(1, 0), -5.0, 1e-12, "Reaction node 1 Fx");
    CHECK_NEAR(rf.component_at(1, 1), -6.0, 1e-12, "Reaction node 1 Fy");
    CHECK_NEAR(rf.component_at(1, 2), -7.0, 1e-12, "Reaction node 1 Fz");
    // Node 3 constrained
    CHECK_NEAR(rf.component_at(3, 0),  3.0, 1e-12, "Reaction node 3 Fx");
    CHECK_NEAR(rf.component_at(3, 2),  5.0, 1e-12, "Reaction node 3 Fz");
}

// ============================================================================
// Test: ShellResultExtractor
// ============================================================================

static void test_shell_von_mises() {
    // Pure uniaxial: sxx = 100, others = 0 → VM = 100
    ShellState st;
    st.stress = {100.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    std::vector<ShellState> states = {st};
    ResultField rf = ShellResultExtractor::extract_stress(states, 1);

    CHECK(rf.type == ResultFieldType::VonMisesStress, "Shell VM type");
    CHECK_NEAR(rf.scalar_at(0), 100.0, 1e-8, "Shell pure uniaxial VM");
}

static void test_shell_von_mises_biaxial() {
    // Equal biaxial: sxx = syy = 100, szz = 0 → VM = 100
    ShellState st;
    st.stress = {100.0, 100.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<ShellState> states = {st};
    ResultField rf = ShellResultExtractor::extract_stress(states, 1);
    double expected = von_mises_ref(100.0, 100.0, 0.0, 0.0, 0.0, 0.0);
    CHECK_NEAR(rf.scalar_at(0), expected, 1e-8, "Shell biaxial VM");
}

static void test_shell_thickness_reduction() {
    std::vector<Real> t0  = {2.0, 4.0};
    std::vector<Real> cur = {1.8, 3.6};

    ResultField rf = ShellResultExtractor::extract_thickness(t0, cur, 2);
    CHECK(rf.type == ResultFieldType::ThicknessReduction, "Thickness type");
    CHECK_NEAR(rf.scalar_at(0), 0.10, 1e-12, "Thickness reduction elem 0");
    CHECK_NEAR(rf.scalar_at(1), 0.10, 1e-12, "Thickness reduction elem 1");
}

static void test_shell_thickness_zero_t0() {
    // t0 = 0 should not produce NaN/inf — result is 0
    std::vector<Real> t0  = {0.0};
    std::vector<Real> cur = {0.5};
    ResultField rf = ShellResultExtractor::extract_thickness(t0, cur, 1);
    CHECK_NEAR(rf.scalar_at(0), 0.0, 1e-12, "Thickness zero t0 guard");
}

static void test_shell_fiber_stress() {
    ShellState st;
    st.stress     = {100.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    st.stress_top = {120.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    st.stress_bot = { 80.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    std::vector<ShellState> states = {st};

    ResultField mid = ShellResultExtractor::extract_fiber_stress(states, 1, ShellSurface::Mid);
    ResultField top = ShellResultExtractor::extract_fiber_stress(states, 1, ShellSurface::Top);
    ResultField bot = ShellResultExtractor::extract_fiber_stress(states, 1, ShellSurface::Bottom);

    CHECK_NEAR(mid.scalar_at(0), 100.0, 1e-8, "Fiber stress mid");
    CHECK_NEAR(top.scalar_at(0), 120.0, 1e-8, "Fiber stress top");
    CHECK_NEAR(bot.scalar_at(0),  80.0, 1e-8, "Fiber stress bot");
    CHECK(top.name == "FiberStress_Top", "Fiber top name");
    CHECK(bot.name == "FiberStress_Bot", "Fiber bot name");
}

static void test_shell_plastic_strain() {
    ShellState st;
    st.plastic_strain = 0.05;
    std::vector<ShellState> states = {st};
    ResultField rf = ShellResultExtractor::extract_plastic_strain(states, 1);
    CHECK(rf.type == ResultFieldType::PlasticStrain, "Shell ps type");
    CHECK_NEAR(rf.scalar_at(0), 0.05, 1e-12, "Shell plastic strain value");
}

// ============================================================================
// Test: SolidResultExtractor
// ============================================================================

static void test_solid_von_mises() {
    SolidState st;
    // Pure shear: sxy = 50, others = 0 → VM = sqrt(3)*50 ≈ 86.60
    st.stress = {0.0, 0.0, 0.0, 50.0, 0.0, 0.0};

    std::vector<SolidState> states = {st};
    ResultField rf = SolidResultExtractor::extract_stress(states, 1);

    double expected = von_mises_ref(0.0, 0.0, 0.0, 50.0, 0.0, 0.0);
    CHECK(rf.type == ResultFieldType::VonMisesStress, "Solid VM type");
    CHECK_NEAR(rf.scalar_at(0), expected, 1e-8, "Solid pure shear VM");
}

static void test_solid_pressure() {
    SolidState st;
    // sxx = syy = szz = -100 (compression) → p = -(-300)/3 = 100
    st.stress = {-100.0, -100.0, -100.0, 0.0, 0.0, 0.0};

    std::vector<SolidState> states = {st};
    ResultField rf = SolidResultExtractor::extract_pressure(states, 1);

    CHECK(rf.type == ResultFieldType::Pressure, "Solid pressure type");
    CHECK_NEAR(rf.scalar_at(0), 100.0, 1e-10, "Solid hydrostatic pressure");
}

static void test_solid_pressure_zero() {
    SolidState st;
    st.stress = {};  // all zeros
    std::vector<SolidState> states = {st};
    ResultField rf = SolidResultExtractor::extract_pressure(states, 1);
    CHECK_NEAR(rf.scalar_at(0), 0.0, 1e-12, "Solid zero pressure");
}

static void test_solid_principal_stress_uniaxial() {
    // Uniaxial: sxx = 200, all others = 0
    // Principal stresses should be: 200, 0, 0
    SolidState st;
    st.stress = {200.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    std::vector<SolidState> states = {st};
    ResultField rf = SolidResultExtractor::extract_principal_stress(states, 1);

    CHECK(rf.type == ResultFieldType::PrincipalStress, "Principal stress type");
    CHECK(rf.num_components == 3, "Principal stress components");
    // Sort descending: sigma_1 = 200, sigma_2 = sigma_3 ≈ 0
    CHECK_NEAR(rf.component_at(0, 0), 200.0, 1e-6, "Principal sigma_1 (uniaxial)");
    CHECK_NEAR(rf.component_at(0, 1),   0.0, 1e-6, "Principal sigma_2 (uniaxial)");
    CHECK_NEAR(rf.component_at(0, 2),   0.0, 1e-6, "Principal sigma_3 (uniaxial)");
}

static void test_solid_principal_stress_hydrostatic() {
    // Hydrostatic: sxx = syy = szz = 50
    // All three principal stresses = 50
    SolidState st;
    st.stress = {50.0, 50.0, 50.0, 0.0, 0.0, 0.0};

    std::vector<SolidState> states = {st};
    ResultField rf = SolidResultExtractor::extract_principal_stress(states, 1);

    CHECK_NEAR(rf.component_at(0, 0), 50.0, 1e-6, "Hydrostatic sigma_1");
    CHECK_NEAR(rf.component_at(0, 1), 50.0, 1e-6, "Hydrostatic sigma_2");
    CHECK_NEAR(rf.component_at(0, 2), 50.0, 1e-6, "Hydrostatic sigma_3");
}

static void test_solid_principal_stress_sum() {
    // Trace must be preserved: sigma_1 + sigma_2 + sigma_3 = sxx + syy + szz
    SolidState st;
    st.stress = {100.0, 50.0, -30.0, 20.0, 10.0, 5.0};
    double trace = 100.0 + 50.0 + (-30.0);

    std::vector<SolidState> states = {st};
    ResultField rf = SolidResultExtractor::extract_principal_stress(states, 1);
    double sum = rf.component_at(0, 0) + rf.component_at(0, 1) + rf.component_at(0, 2);
    CHECK_NEAR(sum, trace, 1e-6, "Principal stress trace invariant");
}

static void test_solid_principal_descending_order() {
    // sigma_1 >= sigma_2 >= sigma_3
    SolidState st;
    st.stress = {-50.0, 80.0, 20.0, 15.0, 5.0, -10.0};
    std::vector<SolidState> states = {st};
    ResultField rf = SolidResultExtractor::extract_principal_stress(states, 1);
    CHECK(rf.component_at(0, 0) >= rf.component_at(0, 1) - 1e-8,
          "Principal sigma_1 >= sigma_2");
    CHECK(rf.component_at(0, 1) >= rf.component_at(0, 2) - 1e-8,
          "Principal sigma_2 >= sigma_3");
}

static void test_solid_plastic_strain_and_damage() {
    SolidState st;
    st.plastic_strain = 0.12;
    st.damage = 0.35;
    std::vector<SolidState> states = {st};

    ResultField ps = SolidResultExtractor::extract_plastic_strain(states, 1);
    ResultField dm = SolidResultExtractor::extract_damage(states, 1);

    CHECK_NEAR(ps.scalar_at(0), 0.12, 1e-12, "Solid plastic strain");
    CHECK_NEAR(dm.scalar_at(0), 0.35, 1e-12, "Solid damage");
}

// ============================================================================
// Test: SPHResultExtractor
// ============================================================================

static void test_sph_density() {
    std::vector<Real> rho = {1000.0, 1050.0, 998.5};
    ResultField rf = SPHResultExtractor::extract_density(rho, 3);

    CHECK(rf.type == ResultFieldType::SPHDensity, "SPH density type");
    CHECK(rf.num_entities() == 3, "SPH density entities");
    CHECK_NEAR(rf.scalar_at(0), 1000.0, 1e-10, "SPH rho[0]");
    CHECK_NEAR(rf.scalar_at(2),  998.5, 1e-10, "SPH rho[2]");
}

static void test_sph_smoothing_length() {
    std::vector<Real> h = {0.01, 0.012, 0.009};
    ResultField rf = SPHResultExtractor::extract_smoothing_length(h, 3);

    CHECK(rf.type == ResultFieldType::SPHSmoothingLength, "SPH h type");
    CHECK_NEAR(rf.scalar_at(1), 0.012, 1e-12, "SPH h[1]");
}

static void test_sph_pressure() {
    std::vector<Real> p = {1.5e5, 2.0e5};
    ResultField rf = SPHResultExtractor::extract_pressure(p, 2);

    CHECK(rf.type == ResultFieldType::SPHPressure, "SPH pressure type");
    CHECK_NEAR(rf.scalar_at(0), 1.5e5, 1.0, "SPH p[0]");
    CHECK_NEAR(rf.scalar_at(1), 2.0e5, 1.0, "SPH p[1]");
}

// ============================================================================
// Test: BeamResultExtractor
// ============================================================================

static void test_beam_axial_force() {
    BeamState bs;
    bs.axial_force = 500.0;
    bs.bending_my = 200.0;
    bs.bending_mz = 100.0;

    std::vector<BeamState> states = {bs};
    ResultField rf = BeamResultExtractor::extract_axial_force(states, 1);

    CHECK(rf.type == ResultFieldType::AxialForce, "Beam axial type");
    CHECK_NEAR(rf.scalar_at(0), 500.0, 1e-10, "Beam axial force value");
}

static void test_beam_bending_moment() {
    BeamState bs;
    bs.bending_my = 300.0;
    bs.bending_mz = 400.0;

    std::vector<BeamState> states = {bs};
    ResultField rf = BeamResultExtractor::extract_bending_moment(states, 1);

    // Resultant = sqrt(300^2 + 400^2) = 500
    CHECK(rf.type == ResultFieldType::BendingMoment, "Beam bending type");
    CHECK_NEAR(rf.scalar_at(0), 500.0, 1e-8, "Beam resultant bending moment");
}

static void test_beam_zero_state() {
    BeamState bs;
    std::vector<BeamState> states = {bs};

    ResultField ra = BeamResultExtractor::extract_axial_force(states, 1);
    ResultField rm = BeamResultExtractor::extract_bending_moment(states, 1);

    CHECK_NEAR(ra.scalar_at(0), 0.0, 1e-12, "Beam zero axial");
    CHECK_NEAR(rm.scalar_at(0), 0.0, 1e-12, "Beam zero moment");
}

// ============================================================================
// Test: RigidBodyExtractor
// ============================================================================

static void test_rigid_body_position() {
    RigidBodyData rb;
    rb.cx = 1.0; rb.cy = 2.0; rb.cz = 3.0;
    std::vector<RigidBodyData> rbs = {rb};

    ResultField rf = RigidBodyExtractor::extract_com_position(rbs, 1);
    CHECK(rf.type == ResultFieldType::COMPosition, "RB position type");
    CHECK(rf.num_components == 3, "RB position components");
    CHECK_NEAR(rf.component_at(0, 0), 1.0, 1e-12, "RB CoM x");
    CHECK_NEAR(rf.component_at(0, 1), 2.0, 1e-12, "RB CoM y");
    CHECK_NEAR(rf.component_at(0, 2), 3.0, 1e-12, "RB CoM z");
}

static void test_rigid_body_velocity() {
    RigidBodyData rb;
    rb.vx = 5.0; rb.vy = -3.0; rb.vz = 0.0;
    std::vector<RigidBodyData> rbs = {rb};

    ResultField rf = RigidBodyExtractor::extract_com_velocity(rbs, 1);
    CHECK(rf.type == ResultFieldType::COMVelocity, "RB velocity type");
    CHECK_NEAR(rf.component_at(0, 0),  5.0, 1e-12, "RB vx");
    CHECK_NEAR(rf.component_at(0, 1), -3.0, 1e-12, "RB vy");
}

static void test_rigid_body_angular_velocity() {
    RigidBodyData rb;
    rb.wx = 0.1; rb.wy = 0.2; rb.wz = 0.3;
    std::vector<RigidBodyData> rbs = {rb};

    ResultField rf = RigidBodyExtractor::extract_angular_velocity(rbs, 1);
    CHECK(rf.type == ResultFieldType::AngularVelocity, "RB omega type");
    CHECK_NEAR(rf.component_at(0, 0), 0.1, 1e-12, "RB wx");
    CHECK_NEAR(rf.component_at(0, 2), 0.3, 1e-12, "RB wz");
}

// ============================================================================
// Test: InterfaceForceExtractor
// ============================================================================

static void test_interface_contact_force() {
    InterfaceData id;
    id.force_x = 100.0; id.force_y = -50.0; id.force_z = 0.0;
    id.gap = 0.001;

    std::vector<InterfaceData> idata = {id};
    ResultField rf = InterfaceForceExtractor::extract_contact_force(idata, 1);

    CHECK(rf.type == ResultFieldType::ContactForce, "Interface force type");
    CHECK(rf.num_components == 3, "Interface force components");
    CHECK_NEAR(rf.component_at(0, 0),  100.0, 1e-12, "Interface Fx");
    CHECK_NEAR(rf.component_at(0, 1),  -50.0, 1e-12, "Interface Fy");
    CHECK_NEAR(rf.component_at(0, 2),    0.0, 1e-12, "Interface Fz");
}

static void test_interface_contact_gap() {
    InterfaceData id;
    id.gap = 0.005;
    std::vector<InterfaceData> idata = {id};

    ResultField rf = InterfaceForceExtractor::extract_contact_gap(idata, 1);
    CHECK(rf.type == ResultFieldType::ContactGap, "Interface gap type");
    CHECK_NEAR(rf.scalar_at(0), 0.005, 1e-12, "Interface gap value");
}

static void test_interface_zero_gap() {
    InterfaceData id;
    id.gap = 0.0;
    std::vector<InterfaceData> idata = {id};
    ResultField rf = InterfaceForceExtractor::extract_contact_gap(idata, 1);
    CHECK_NEAR(rf.scalar_at(0), 0.0, 1e-12, "Interface zero gap");
}

// ============================================================================
// Test: CrackResultExtractor
// ============================================================================

static void test_crack_length() {
    CrackData cd;
    cd.length = 3.14;
    std::vector<CrackData> cracks = {cd};

    ResultField rf = CrackResultExtractor::extract_crack_length(cracks, 1);
    CHECK(rf.type == ResultFieldType::CrackLength, "Crack length type");
    CHECK_NEAR(rf.scalar_at(0), 3.14, 1e-12, "Crack length value");
}

static void test_crack_sif() {
    CrackData cd;
    cd.sif_I = 25.0e6; cd.sif_II = 5.0e6; cd.sif_III = 1.0e6;
    std::vector<CrackData> cracks = {cd};

    ResultField rf = CrackResultExtractor::extract_sif(cracks, 1);
    CHECK(rf.type == ResultFieldType::StressIntensityFactor, "SIF type");
    CHECK(rf.num_components == 3, "SIF components");
    CHECK_NEAR(rf.component_at(0, 0), 25.0e6, 1.0, "SIF K_I");
    CHECK_NEAR(rf.component_at(0, 1),  5.0e6, 1.0, "SIF K_II");
    CHECK_NEAR(rf.component_at(0, 2),  1.0e6, 1.0, "SIF K_III");
}

static void test_crack_angle() {
    CrackData cd;
    cd.angle_deg = 45.0;
    std::vector<CrackData> cracks = {cd};

    ResultField rf = CrackResultExtractor::extract_crack_angle(cracks, 1);
    CHECK(rf.type == ResultFieldType::CrackAngle, "Crack angle type");
    CHECK_NEAR(rf.scalar_at(0), 45.0, 1e-12, "Crack angle value");
}

// ============================================================================
// Test: SectionForceExtractor
// ============================================================================

static void test_section_force() {
    SectionData sd;
    sd.fx = 1000.0; sd.fy = -200.0; sd.fz = 50.0;
    std::vector<SectionData> sdata = {sd};

    ResultField rf = SectionForceExtractor::extract_section_force(sdata, 1);
    CHECK(rf.type == ResultFieldType::SectionForce, "Section force type");
    CHECK(rf.num_components == 3, "Section force components");
    CHECK_NEAR(rf.component_at(0, 0), 1000.0, 1e-10, "Section Fx");
    CHECK_NEAR(rf.component_at(0, 1), -200.0, 1e-10, "Section Fy");
    CHECK_NEAR(rf.component_at(0, 2),   50.0, 1e-10, "Section Fz");
}

static void test_section_moment() {
    SectionData sd;
    sd.mx = 500.0; sd.my = -300.0; sd.mz = 100.0;
    std::vector<SectionData> sdata = {sd};

    ResultField rf = SectionForceExtractor::extract_section_moment(sdata, 1);
    CHECK(rf.type == ResultFieldType::SectionMoment, "Section moment type");
    CHECK_NEAR(rf.component_at(0, 0),  500.0, 1e-10, "Section Mx");
    CHECK_NEAR(rf.component_at(0, 1), -300.0, 1e-10, "Section My");
    CHECK_NEAR(rf.component_at(0, 2),  100.0, 1e-10, "Section Mz");
}

// ============================================================================
// Test: OutputDispatcher
// ============================================================================

static void test_dispatcher_register_and_extract() {
    OutputDispatcher disp;

    // Register a simple solid von Mises extractor
    disp.register_extractor(EntityType::Solid, ResultFieldType::VonMisesStress,
        [](const void* data, int n) -> ResultField {
            auto* states = static_cast<const SolidState*>(data);
            std::vector<SolidState> sv(states, states + n);
            return SolidResultExtractor::extract_stress(sv, n);
        });

    CHECK(disp.has_extractor(EntityType::Solid, ResultFieldType::VonMisesStress),
          "Dispatcher: has_extractor registered");
    CHECK(!disp.has_extractor(EntityType::Node, ResultFieldType::Displacement),
          "Dispatcher: has_extractor unregistered");
    CHECK(disp.num_registered() == 1, "Dispatcher: num_registered");

    SolidState st;
    st.stress = {100.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    ResultField rf = disp.extract(EntityType::Solid,
                                  ResultFieldType::VonMisesStress,
                                  &st, 1);
    CHECK(rf.type == ResultFieldType::VonMisesStress, "Dispatcher extract type");
    CHECK_NEAR(rf.scalar_at(0), 100.0, 1e-8, "Dispatcher extract value");
}

static void test_dispatcher_empty_route() {
    OutputDispatcher disp;
    SolidState st;
    // No extractor registered → should return empty field without crashing
    ResultField rf = disp.extract(EntityType::Solid, ResultFieldType::Pressure,
                                  &st, 1);
    CHECK(rf.name == "Unknown", "Dispatcher empty route name");
    CHECK(rf.data.empty(), "Dispatcher empty route data empty");
}

static void test_dispatcher_multiple_registrations() {
    OutputDispatcher disp;

    disp.register_extractor(EntityType::Node, ResultFieldType::Displacement,
        [](const void*, int) -> ResultField {
            ResultField rf;
            rf.name = "Displacement";
            rf.type = ResultFieldType::Displacement;
            rf.num_components = 3;
            rf.data = {1.0, 2.0, 3.0};
            return rf;
        });

    disp.register_extractor(EntityType::SPH, ResultFieldType::SPHDensity,
        [](const void* data, int n) -> ResultField {
            auto* rho = static_cast<const Real*>(data);
            std::vector<Real> rv(rho, rho + n);
            return SPHResultExtractor::extract_density(rv, n);
        });

    CHECK(disp.num_registered() == 2, "Dispatcher: 2 registrations");
    CHECK(disp.has_extractor(EntityType::Node, ResultFieldType::Displacement),
          "Dispatcher: node disp registered");
    CHECK(disp.has_extractor(EntityType::SPH, ResultFieldType::SPHDensity),
          "Dispatcher: sph density registered");
    CHECK(!disp.has_extractor(EntityType::Shell, ResultFieldType::VonMisesStress),
          "Dispatcher: shell VM not registered");
}

static void test_dispatcher_unregister() {
    OutputDispatcher disp;
    disp.register_extractor(EntityType::Beam, ResultFieldType::AxialForce,
        [](const void*, int) -> ResultField { return {}; });
    CHECK(disp.has_extractor(EntityType::Beam, ResultFieldType::AxialForce),
          "Before unregister");
    disp.unregister(EntityType::Beam, ResultFieldType::AxialForce);
    CHECK(!disp.has_extractor(EntityType::Beam, ResultFieldType::AxialForce),
          "After unregister");
    CHECK(disp.num_registered() == 0, "After unregister count");
}

static void test_dispatcher_clear() {
    OutputDispatcher disp;
    disp.register_extractor(EntityType::Node, ResultFieldType::Velocity,
        [](const void*, int) -> ResultField { return {}; });
    disp.register_extractor(EntityType::Solid, ResultFieldType::Pressure,
        [](const void*, int) -> ResultField { return {}; });
    disp.clear();
    CHECK(disp.num_registered() == 0, "Dispatcher clear");
}

// ============================================================================
// Test: edge cases — empty/zero data
// ============================================================================

static void test_empty_node_data() {
    std::vector<Real> empty;
    ResultField rf = NodeResultExtractor::extract_velocity(empty, 0);
    CHECK(rf.num_entities() == 0, "Empty node velocity entities");
    CHECK(rf.data.empty(), "Empty node velocity data");
}

static void test_empty_solid_states() {
    std::vector<SolidState> states;
    ResultField rf = SolidResultExtractor::extract_stress(states, 0);
    CHECK(rf.num_entities() == 0, "Empty solid stress entities");
}

static void test_solid_multiple_elements() {
    SolidState s1, s2;
    s1.stress = {100.0, 0.0, 0.0, 0.0, 0.0, 0.0};   // VM = 100
    s2.stress = {0.0, 0.0, 0.0, 50.0, 0.0, 0.0};     // VM = sqrt(3)*50

    std::vector<SolidState> states = {s1, s2};
    ResultField rf = SolidResultExtractor::extract_stress(states, 2);

    CHECK(rf.num_entities() == 2, "Multi-solid entities");
    CHECK_NEAR(rf.scalar_at(0), 100.0, 1e-8, "Multi-solid VM[0]");
    double expected1 = von_mises_ref(0.0, 0.0, 0.0, 50.0, 0.0, 0.0);
    CHECK_NEAR(rf.scalar_at(1), expected1, 1e-8, "Multi-solid VM[1]");
}

static void test_multiple_rigid_bodies() {
    RigidBodyData rb1, rb2;
    rb1.cx = 1.0; rb1.cy = 0.0; rb1.cz = 0.0;
    rb2.cx = 5.0; rb2.cy = 5.0; rb2.cz = 5.0;
    std::vector<RigidBodyData> rbs = {rb1, rb2};

    ResultField rf = RigidBodyExtractor::extract_com_position(rbs, 2);
    CHECK(rf.num_entities() == 2, "Multi-body entities");
    CHECK_NEAR(rf.component_at(0, 0), 1.0, 1e-12, "Body 0 cx");
    CHECK_NEAR(rf.component_at(1, 0), 5.0, 1e-12, "Body 1 cx");
    CHECK_NEAR(rf.component_at(1, 2), 5.0, 1e-12, "Body 1 cz");
}

static void test_section_multiple() {
    SectionData sd1, sd2;
    sd1.fx = 100.0; sd1.fy = 0.0; sd1.fz = 0.0;
    sd2.fx = 0.0;   sd2.fy = 200.0; sd2.fz = 0.0;
    std::vector<SectionData> sdata = {sd1, sd2};

    ResultField rf = SectionForceExtractor::extract_section_force(sdata, 2);
    CHECK(rf.num_entities() == 2, "Section multi entities");
    CHECK_NEAR(rf.component_at(0, 0), 100.0, 1e-10, "Section[0] Fx");
    CHECK_NEAR(rf.component_at(1, 1), 200.0, 1e-10, "Section[1] Fy");
}

// ============================================================================
// main
// ============================================================================

int main() {
    std::cout << "=== Wave 43: Per-entity output extractors ===\n\n";

    // ResultField
    test_result_field_structure();

    // Node extractors
    test_node_displacement();
    test_node_displacement_zero_ref();
    test_node_velocity();
    test_node_acceleration();
    test_node_reaction_forces();

    // Shell extractors
    test_shell_von_mises();
    test_shell_von_mises_biaxial();
    test_shell_thickness_reduction();
    test_shell_thickness_zero_t0();
    test_shell_fiber_stress();
    test_shell_plastic_strain();

    // Solid extractors
    test_solid_von_mises();
    test_solid_pressure();
    test_solid_pressure_zero();
    test_solid_principal_stress_uniaxial();
    test_solid_principal_stress_hydrostatic();
    test_solid_principal_stress_sum();
    test_solid_principal_descending_order();
    test_solid_plastic_strain_and_damage();

    // SPH
    test_sph_density();
    test_sph_smoothing_length();
    test_sph_pressure();

    // Beam
    test_beam_axial_force();
    test_beam_bending_moment();
    test_beam_zero_state();

    // Rigid body
    test_rigid_body_position();
    test_rigid_body_velocity();
    test_rigid_body_angular_velocity();

    // Interface
    test_interface_contact_force();
    test_interface_contact_gap();
    test_interface_zero_gap();

    // Crack
    test_crack_length();
    test_crack_sif();
    test_crack_angle();

    // Section
    test_section_force();
    test_section_moment();

    // Dispatcher
    test_dispatcher_register_and_extract();
    test_dispatcher_empty_route();
    test_dispatcher_multiple_registrations();
    test_dispatcher_unregister();
    test_dispatcher_clear();

    // Edge cases
    test_empty_node_data();
    test_empty_solid_states();
    test_solid_multiple_elements();
    test_multiple_rigid_bodies();
    test_section_multiple();

    std::cout << "\n=== Results: "
              << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return tests_failed > 0 ? 1 : 0;
}
