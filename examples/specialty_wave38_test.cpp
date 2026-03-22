/**
 * @file specialty_wave38_test.cpp
 * @brief Wave 38: Specialty Elements Test Suite (4 elements, 25 tests)
 *
 * Tests:
 *   6. HermiteBeam18      (8 tests)
 *   7. RivetElement        (6 tests)
 *   8. WeldElement         (5 tests)
 *   9. GeneralSpringBeam   (6 tests)
 */

#include <nexussim/discretization/specialty_wave38.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>

using namespace nxs;
using namespace nxs::discretization;

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
// 6. HermiteBeam18 Tests
// ============================================================================

void test_6_hermite_beam() {
    std::cout << "--- Test 6: HermiteBeam18 ---\n";

    HermiteBeam18 beam;
    Real node1[3] = {0.0, 0.0, 0.0};
    Real node2[3] = {1.0, 0.0, 0.0};  // L = 1.0 m

    BeamSection section;
    section.A = 0.01;          // 100 cm^2
    section.Iy = 8.333e-6;     // I for 0.1 x 0.1 cross-section
    section.Iz = 8.333e-6;
    section.J = 1.406e-5;
    section.Cw = 0.0;
    section.E = 2.1e11;
    section.G = 8.08e10;
    section.rho = 7800.0;

    // 6a. Stiffness matrix symmetry
    {
        Real K[144];
        beam.compute_stiffness(node1, node2, section, K);

        Real max_asym = 0.0;
        for (int i = 0; i < 12; ++i) {
            for (int j = i + 1; j < 12; ++j) {
                Real diff = std::fabs(K[i * 12 + j] - K[j * 12 + i]);
                Real scale = std::fmax(std::fabs(K[i * 12 + j]), 1.0);
                if (diff / scale > max_asym) max_asym = diff / scale;
            }
        }
        CHECK(max_asym < 1.0e-10, "Beam: stiffness matrix is symmetric");
    }

    // 6b. Axial stiffness K(0,0) = EA/L
    {
        Real K[144];
        beam.compute_stiffness(node1, node2, section, K);
        Real EA_L = section.E * section.A / 1.0;
        CHECK_NEAR(K[0], EA_L, EA_L * 1.0e-10, "Beam: K(0,0) = EA/L");
    }

    // 6c. Diagonal entries are positive
    {
        Real K[144];
        beam.compute_stiffness(node1, node2, section, K);
        bool all_pos = true;
        for (int i = 0; i < 12; ++i) {
            if (K[i * 12 + i] <= 0.0) all_pos = false;
        }
        CHECK(all_pos, "Beam: all diagonal stiffness entries positive");
    }

    // 6d. Zero displacement produces zero force
    {
        Real displ[12] = {};
        Real forces[12];
        beam.compute_internal_force(node1, node2, section, displ, forces);

        Real max_f = 0.0;
        for (int i = 0; i < 12; ++i) {
            if (std::fabs(forces[i]) > max_f) max_f = std::fabs(forces[i]);
        }
        CHECK(max_f < 1.0e-10, "Beam: zero displacement -> zero force");
    }

    // 6e. Cantilever tip deflection
    {
        Real P = 1000.0;  // 1 kN tip load in y
        Real L = 1.0;
        Real E = section.E;
        Real I = section.Iz;
        Real delta_analytical = HermiteBeam18::cantilever_deflection(P, L, E, I);
        Real delta_expected = P * L * L * L / (3.0 * E * I);
        CHECK_NEAR(delta_analytical, delta_expected, 1.0e-15,
                   "Beam: cantilever deflection P*L^3/(3EI)");
    }

    // 6f. Bending stiffness K(1,1) = 12*EIz/L^3
    {
        Real K[144];
        beam.compute_stiffness(node1, node2, section, K);
        Real L = 1.0;
        Real expected = 12.0 * section.E * section.Iz / (L * L * L);
        CHECK_NEAR(K[1 * 12 + 1], expected, expected * 1.0e-10,
                   "Beam: K(1,1) = 12*E*Iz/L^3");
    }

    // 6g. Hermite shape functions at endpoints
    {
        Real N[4];
        Real L = 1.0;
        HermiteBeam18::hermite_shape_functions(0.0, L, N);
        CHECK_NEAR(N[0], 1.0, 1.0e-15, "Beam: N1(0) = 1");
        CHECK_NEAR(N[2], 0.0, 1.0e-15, "Beam: N3(0) = 0");

        HermiteBeam18::hermite_shape_functions(1.0, L, N);
        CHECK_NEAR(N[0], 0.0, 1.0e-15, "Beam: N1(1) = 0");
        CHECK_NEAR(N[2], 1.0, 1.0e-15, "Beam: N3(1) = 1");
    }

    // 6h. Lumped mass total equals rho*A*L
    {
        Real M_diag[12];
        beam.compute_mass_lumped(node1, node2, section, M_diag);
        Real total_trans_mass = M_diag[0] + M_diag[1] + M_diag[2]
                              + M_diag[6] + M_diag[7] + M_diag[8];
        Real expected = 3.0 * section.rho * section.A * 1.0;  // 3 DOF * 2 nodes * m/2
        CHECK_NEAR(total_trans_mass, expected, expected * 1.0e-10,
                   "Beam: total translational mass = 3*rho*A*L");
    }
}

// ============================================================================
// 7. RivetElement Tests
// ============================================================================

void test_7_rivet_element() {
    std::cout << "--- Test 7: RivetElement ---\n";

    RivetElement rivet;
    RivetProps props;
    props.K_axial = 1.0e6;
    props.K_shear = 2.0e6;
    props.F_axial_max = 5000.0;
    props.F_shear_max = 8000.0;
    props.failure_disp = 0.01;

    // 7a. Linear elastic response
    {
        Real F_a, F_s;
        bool failed;
        rivet.compute_rivet_force(0.001, 0.0, props, F_a, F_s, failed);
        CHECK_NEAR(F_a, 1000.0, 0.01, "Rivet: axial force = K*delta");
        CHECK(!failed, "Rivet: not failed in elastic range");
    }

    // 7b. Shear force linearity
    {
        Real F_a, F_s;
        bool failed;
        rivet.compute_rivet_force(0.0, 0.002, props, F_a, F_s, failed);
        CHECK_NEAR(F_s, 4000.0, 0.01, "Rivet: shear force = K_s*delta_s");
        CHECK_NEAR(F_a, 0.0, 1.0e-10, "Rivet: zero axial when delta_n=0");
    }

    // 7c. Pure axial failure
    {
        Real F_a, F_s;
        bool failed;
        // F_a = K*d = 1e6 * 0.005 = 5000 = F_axial_max -> failure_index = 1.0
        rivet.compute_rivet_force(0.005, 0.0, props, F_a, F_s, failed);
        CHECK(failed, "Rivet: pure axial failure at F_a = F_max");
    }

    // 7d. Mixed-mode failure
    {
        Real F_a, F_s;
        bool failed;
        // F_a = 1e6 * 0.003 = 3000, ratio = 3000/5000 = 0.6
        // F_s = 2e6 * 0.0032 = 6400, ratio = 6400/8000 = 0.8
        // failure_index = 0.36 + 0.64 = 1.0
        rivet.compute_rivet_force(0.003, 0.0032, props, F_a, F_s, failed);
        CHECK(failed, "Rivet: mixed-mode failure (0.6^2 + 0.8^2 = 1.0)");
    }

    // 7e. Failure index computation
    {
        Real fi = RivetElement::failure_index(3000.0, 6400.0, props);
        // (3000/5000)^2 + (6400/8000)^2 = 0.36 + 0.64 = 1.0
        CHECK_NEAR(fi, 1.0, 1.0e-10, "Rivet: failure index = 1.0 at boundary");
    }

    // 7f. Zero deformation produces zero force
    {
        Real F_a, F_s;
        bool failed;
        rivet.compute_rivet_force(0.0, 0.0, props, F_a, F_s, failed);
        CHECK_NEAR(F_a, 0.0, 1.0e-15, "Rivet: zero deformation -> zero axial");
        CHECK_NEAR(F_s, 0.0, 1.0e-15, "Rivet: zero deformation -> zero shear");
        CHECK(!failed, "Rivet: not failed at zero deformation");
    }
}

// ============================================================================
// 8. WeldElement Tests
// ============================================================================

void test_8_weld_element() {
    std::cout << "--- Test 8: WeldElement ---\n";

    WeldElement weld;

    // 8a. Elastic response (small displacement)
    {
        WeldProps props;
        props.diameter = 0.006;
        props.E_weld = 2.1e11;
        props.sigma_y_weld = 600.0e6;
        props.E_haz = 1.8e11;
        props.sigma_y_haz = 400.0e6;
        props.haz_width = 0.003;
        props.sheet_thick = 0.001;
        props.damage = 0.0;

        Real force;
        Real K = weld.compute_stiffness(props);
        weld.compute_weld_force(1.0e-6, props, force);
        CHECK_NEAR(force, K * 1.0e-6, std::fabs(K * 1.0e-6 * 0.01),
                   "Weld: elastic response F = K*delta");
    }

    // 8b. HAZ softening factor
    {
        WeldProps props;
        props.sigma_y_weld = 600.0e6;
        props.sigma_y_haz = 400.0e6;
        Real sf = WeldElement::haz_softening_factor(props);
        CHECK_NEAR(sf, 400.0 / 600.0, 1.0e-10, "Weld: HAZ softening = sigma_haz/sigma_weld");
    }

    // 8c. Damage reduces force
    {
        WeldProps props1, props2;
        props1.damage = 0.0;
        props2.damage = 0.5;

        Real force1, force2;
        weld.compute_weld_force(1.0e-6, props1, force1);
        weld.compute_weld_force(1.0e-6, props2, force2);
        CHECK(std::fabs(force2) < std::fabs(force1),
              "Weld: damage reduces force");
    }

    // 8d. Zero displacement produces zero force
    {
        WeldProps props;
        props.damage = 0.0;
        Real force;
        weld.compute_weld_force(0.0, props, force);
        CHECK_NEAR(force, 0.0, 1.0e-10, "Weld: zero displacement -> zero force");
    }

    // 8e. Full damage produces zero force
    {
        WeldProps props;
        props.damage = 1.0;
        props.max_damage = 1.0;
        Real force;
        weld.compute_weld_force(0.001, props, force);
        CHECK_NEAR(force, 0.0, 1.0e-10, "Weld: full damage -> zero force");
    }
}

// ============================================================================
// 9. GeneralSpringBeam Tests
// ============================================================================

void test_9_spring_beam() {
    std::cout << "--- Test 9: GeneralSpringBeam ---\n";

    GeneralSpringBeam sb;

    // 9a. Linear spring: F = K * d
    {
        SpringBeamProps props;
        props.K[0] = 1.0e6; props.K[1] = 2.0e6; props.K[2] = 3.0e6;
        props.K[3] = 100.0; props.K[4] = 200.0; props.K[5] = 300.0;

        Real displ[3] = {0.001, 0.002, 0.003};
        Real rot[3] = {0.01, 0.02, 0.03};
        Real force[3], moment[3];

        sb.compute_spring_force(displ, rot, props, force, moment);

        CHECK_NEAR(force[0], 1000.0, 0.01, "SpringBeam: F_x = K_x * d_x");
        CHECK_NEAR(force[1], 4000.0, 0.01, "SpringBeam: F_y = K_y * d_y");
        CHECK_NEAR(moment[2], 9.0, 0.01, "SpringBeam: M_z = K_rz * theta_z");
    }

    // 9b. Independent axes: x-displacement only affects F_x
    {
        SpringBeamProps props;
        props.K[0] = 1.0e6; props.K[1] = 2.0e6; props.K[2] = 3.0e6;
        props.K[3] = 100.0; props.K[4] = 200.0; props.K[5] = 300.0;

        Real displ[3] = {0.001, 0.0, 0.0};
        Real rot[3] = {0.0, 0.0, 0.0};
        Real force[3], moment[3];

        sb.compute_spring_force(displ, rot, props, force, moment);
        CHECK_NEAR(force[0], 1000.0, 0.01, "SpringBeam: x-disp -> F_x");
        CHECK_NEAR(force[1], 0.0, 1.0e-10, "SpringBeam: x-disp -> F_y = 0");
        CHECK_NEAR(force[2], 0.0, 1.0e-10, "SpringBeam: x-disp -> F_z = 0");
        CHECK_NEAR(moment[0], 0.0, 1.0e-10, "SpringBeam: x-disp -> M_x = 0");
    }

    // 9c. Tabulated curve response
    {
        SpringBeamProps props;
        props.use_curve[0] = true;
        props.n_pts[0] = 3;
        // Bilinear: elastic then constant
        props.curve_x[0][0] = 0.0;   props.curve_y[0][0] = 0.0;
        props.curve_x[0][1] = 0.001; props.curve_y[0][1] = 1000.0;
        props.curve_x[0][2] = 0.01;  props.curve_y[0][2] = 1000.0;

        Real displ[3] = {0.0005, 0.0, 0.0};
        Real rot[3] = {0.0, 0.0, 0.0};
        Real force[3], moment[3];

        sb.compute_spring_force(displ, rot, props, force, moment);
        CHECK_NEAR(force[0], 500.0, 1.0, "SpringBeam: tabulated curve interpolation");
    }

    // 9d. Zero deformation -> zero force
    {
        SpringBeamProps props;
        props.K[0] = 1.0e6; props.K[1] = 1.0e6; props.K[2] = 1.0e6;
        props.K[3] = 100.0; props.K[4] = 100.0; props.K[5] = 100.0;

        Real displ[3] = {0.0, 0.0, 0.0};
        Real rot[3] = {0.0, 0.0, 0.0};
        Real force[3], moment[3];

        sb.compute_spring_force(displ, rot, props, force, moment);

        Real max_f = 0.0;
        for (int i = 0; i < 3; ++i) {
            if (std::fabs(force[i]) > max_f) max_f = std::fabs(force[i]);
            if (std::fabs(moment[i]) > max_f) max_f = std::fabs(moment[i]);
        }
        CHECK(max_f < 1.0e-10, "SpringBeam: zero deformation -> zero force/moment");
    }

    // 9e. Tangent stiffness for linear spring equals K
    {
        SpringBeamProps props;
        props.K[0] = 1.5e6;

        Real Kt = sb.tangent_stiffness(0, 0.001, props);
        CHECK_NEAR(Kt, 1.5e6, 1.0, "SpringBeam: tangent stiffness = K for linear");
    }

    // 9f. Energy for linear spring: E = 0.5*K*d^2
    {
        SpringBeamProps props;
        props.K[0] = 1.0e6;
        Real d = 0.001;
        Real energy = sb.compute_energy(0, d, props);
        Real expected = 0.5 * 1.0e6 * d * d;
        CHECK_NEAR(energy, expected, expected * 1.0e-6,
                   "SpringBeam: energy = 0.5*K*d^2");
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Wave 38: Specialty Elements Test Suite ===\n\n";

    test_6_hermite_beam();
    test_7_rivet_element();
    test_8_weld_element();
    test_9_spring_beam();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " ===\n";

    return tests_failed > 0 ? 1 : 0;
}
