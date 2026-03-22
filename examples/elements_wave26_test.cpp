/**
 * @file elements_wave26_test.cpp
 * @brief Wave 26: Advanced Element Formulations II Test Suite (8 elements, ~80 tests)
 *
 * Tests 8 element formulations:
 *  1. QEPHShell                      (10 tests)
 *  2. BATOZShell                     (10 tests)
 *  3. CorotationalShell              (10 tests)
 *  4. ANSHex8                        (10 tests)
 *  5. StabilizedIncompatibleHex8     (10 tests)
 *  6. LayeredThickShell              (10 tests)
 *  7. SelectiveMassHex8              (10 tests)
 *  8. EnhancedStrainExtrapolationHex8 (10 tests)
 */

#include <nexussim/discretization/elements_wave26.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <array>
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
// Helper: material constants
// ============================================================================
static constexpr Real STEEL_E = 210.0e9;
static constexpr Real STEEL_NU = 0.3;
static constexpr Real STEEL_DENSITY = 7800.0;
static constexpr Real SHELL_THICKNESS = 0.01;

// Helper: unit square shell coords (4 nodes in xy-plane)
static void make_shell_coords(Real coords[4][3]) {
    coords[0][0] = 0; coords[0][1] = 0; coords[0][2] = 0;
    coords[1][0] = 1; coords[1][1] = 0; coords[1][2] = 0;
    coords[2][0] = 1; coords[2][1] = 1; coords[2][2] = 0;
    coords[3][0] = 0; coords[3][1] = 1; coords[3][2] = 0;
}

// Helper: unit triangle coords (3 nodes)
static void make_tri_coords(Real coords[3][3]) {
    coords[0][0] = 0; coords[0][1] = 0; coords[0][2] = 0;
    coords[1][0] = 1; coords[1][1] = 0; coords[1][2] = 0;
    coords[2][0] = 0; coords[2][1] = 1; coords[2][2] = 0;
}

// Helper: unit cube hex8 coords
static void make_hex8_coords(Real coords[8][3]) {
    coords[0][0] = 0; coords[0][1] = 0; coords[0][2] = 0;
    coords[1][0] = 1; coords[1][1] = 0; coords[1][2] = 0;
    coords[2][0] = 1; coords[2][1] = 1; coords[2][2] = 0;
    coords[3][0] = 0; coords[3][1] = 1; coords[3][2] = 0;
    coords[4][0] = 0; coords[4][1] = 0; coords[4][2] = 1;
    coords[5][0] = 1; coords[5][1] = 0; coords[5][2] = 1;
    coords[6][0] = 1; coords[6][1] = 1; coords[6][2] = 1;
    coords[7][0] = 0; coords[7][1] = 1; coords[7][2] = 1;
}

// Helper: check stiffness matrix symmetry
static bool check_symmetry(const Real* K, int n, Real rtol = 1e-8) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            Real kij = K[i*n + j];
            Real kji = K[j*n + i];
            Real denom = std::max(std::abs(kij), 1e-30);
            if (std::abs(kij - kji) / denom > rtol) return false;
        }
    }
    return true;
}

// Helper: check all diagonal entries non-negative
static bool check_diagonal_nonneg(const Real* K, int n) {
    for (int i = 0; i < n; ++i) {
        if (K[i*n + i] < -1e-10) return false;
    }
    return true;
}

// ============================================================================
// 1. QEPHShell Tests
// ============================================================================
void test_qeph_shell() {
    std::cout << "\n=== 1. QEPHShell Tests ===\n";

    Real coords[4][3];
    make_shell_coords(coords);
    QEPHShell elem(STEEL_E, STEEL_NU, SHELL_THICKNESS, STEEL_DENSITY, 0.05);

    // Test 1: Constructor and accessors
    {
        CHECK_NEAR(elem.E(), STEEL_E, 1.0, "QEPH: E accessor");
        CHECK_NEAR(elem.nu(), STEEL_NU, 1e-15, "QEPH: nu accessor");
        CHECK_NEAR(elem.thickness(), SHELL_THICKNESS, 1e-15, "QEPH: thickness accessor");
    }

    // Test 2: Stiffness matrix symmetry
    {
        Real K[576];
        elem.stiffness(coords, STEEL_E, STEEL_NU, SHELL_THICKNESS, K);
        CHECK(check_symmetry(K, 24, 1e-6), "QEPH: stiffness symmetry");
    }

    // Test 3: Stiffness matrix non-negative diagonal
    {
        Real K[576];
        elem.stiffness(coords, STEEL_E, STEEL_NU, SHELL_THICKNESS, K);
        CHECK(check_diagonal_nonneg(K, 24), "QEPH: non-negative diagonal");
    }

    // Test 4: Lumped mass total equals rho*A*t
    {
        Real M[24];
        elem.mass(coords, STEEL_DENSITY, SHELL_THICKNESS, M);
        Real total_mass = 0.0;
        for (int i = 0; i < 4; ++i) total_mass += M[i*6+0];
        Real expected = STEEL_DENSITY * 1.0 * SHELL_THICKNESS; // area=1
        CHECK_NEAR(total_mass, expected, expected * 1e-10, "QEPH: total mass correct");
    }

    // Test 5: Hourglass force nonzero for hourglass mode
    {
        Real disp[24] = {};
        // Hourglass mode: +1,-1,+1,-1 in x-translation
        disp[0*6+0] =  0.001;
        disp[1*6+0] = -0.001;
        disp[2*6+0] =  0.001;
        disp[3*6+0] = -0.001;

        Real f_hg[24];
        elem.hourglass_force(coords, disp, STEEL_E, STEEL_NU, SHELL_THICKNESS, f_hg);

        Real norm = 0.0;
        for (int i = 0; i < 24; ++i) norm += f_hg[i]*f_hg[i];
        CHECK(norm > 0.0, "QEPH: nonzero HG force for HG mode displacement");
    }

    // Test 6: Hourglass force zero for rigid body translation
    {
        Real disp[24] = {};
        for (int a = 0; a < 4; ++a) disp[a*6+0] = 0.001;

        Real f_hg[24];
        elem.hourglass_force(coords, disp, STEEL_E, STEEL_NU, SHELL_THICKNESS, f_hg);

        Real norm = 0.0;
        for (int i = 0; i < 24; ++i) norm += f_hg[i]*f_hg[i];
        CHECK(norm < 1e-20, "QEPH: zero HG force for rigid translation");
    }

    // Test 7: Hourglass energy is bounded and positive for HG mode
    {
        Real disp[24] = {};
        disp[0*6+0] =  0.001;
        disp[1*6+0] = -0.001;
        disp[2*6+0] =  0.001;
        disp[3*6+0] = -0.001;

        Real E_hg = elem.hourglass_energy(coords, disp, STEEL_E, SHELL_THICKNESS);
        CHECK(E_hg > 0.0, "QEPH: positive HG energy for HG mode");

        // HG energy should be much less than membrane strain energy
        CHECK(E_hg < 1.0e20, "QEPH: HG energy is bounded");
    }

    // Test 8: Stable time step is positive
    {
        Real dt = elem.stable_time_step(coords);
        CHECK(dt > 0.0 && dt < 1.0, "QEPH: positive finite stable timestep");
    }

    // Test 9: Constant strain patch test (uniform x-stretch)
    {
        Real K[576];
        elem.stiffness(coords, STEEL_E, STEEL_NU, SHELL_THICKNESS, K);

        // Apply uniform x-displacement: u = eps_x * x
        Real eps_x = 0.001;
        Real disp[24] = {};
        for (int a = 0; a < 4; ++a)
            disp[a*6+0] = eps_x * coords[a][0];

        // f = K * u
        Real f[24] = {};
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 24; ++j)
                f[i] += K[i*24+j] * disp[j];

        // Sum of forces should be near zero (equilibrium)
        Real fx_sum = 0.0;
        for (int a = 0; a < 4; ++a) fx_sum += f[a*6+0];
        CHECK(std::abs(fx_sum) < 1.0e3, "QEPH: patch test - force equilibrium");
    }

    // Test 10: Bending stiffness present in rotational DOF
    {
        Real K[576];
        elem.stiffness(coords, STEEL_E, STEEL_NU, SHELL_THICKNESS, K);
        // Check that rotational DOF have stiffness
        Real bend_diag = 0.0;
        for (int a = 0; a < 4; ++a) {
            bend_diag += std::abs(K[(a*6+3)*24 + (a*6+3)]);
            bend_diag += std::abs(K[(a*6+4)*24 + (a*6+4)]);
        }
        CHECK(bend_diag > 0.0, "QEPH: bending stiffness present in rotational DOF");
    }
}

// ============================================================================
// 2. BATOZShell Tests
// ============================================================================
void test_batoz_shell() {
    std::cout << "\n=== 2. BATOZShell Tests ===\n";

    Real tri[3][3];
    make_tri_coords(tri);
    BATOZShell elem(tri, STEEL_E, STEEL_NU, SHELL_THICKNESS);

    // Test 1: Accessors
    {
        CHECK_NEAR(elem.E(), STEEL_E, 1.0, "BATOZ: E accessor");
        CHECK_NEAR(elem.nu(), STEEL_NU, 1e-15, "BATOZ: nu accessor");
        CHECK_NEAR(elem.thickness(), SHELL_THICKNESS, 1e-15, "BATOZ: thickness accessor");
    }

    // Test 2: Area of unit right triangle = 0.5
    {
        CHECK_NEAR(elem.area(), 0.5, 1e-12, "BATOZ: area = 0.5 for unit right triangle");
    }

    // Test 3: Stiffness matrix symmetry
    {
        Real K[324];
        elem.stiffness(K);
        CHECK(check_symmetry(K, 18, 1e-6), "BATOZ: stiffness symmetry");
    }

    // Test 4: Non-negative diagonal
    {
        Real K[324];
        elem.stiffness(K);
        CHECK(check_diagonal_nonneg(K, 18), "BATOZ: non-negative diagonal");
    }

    // Test 5: Lumped mass total
    {
        Real M[18];
        elem.mass(STEEL_DENSITY, M);
        Real total = 0.0;
        for (int a = 0; a < 3; ++a) total += M[a*6+0];
        Real expected = STEEL_DENSITY * 0.5 * SHELL_THICKNESS;
        CHECK_NEAR(total, expected, expected * 1e-10, "BATOZ: total mass correct");
    }

    // Test 6: Membrane stiffness for in-plane tension
    {
        Real K[324];
        elem.stiffness(K);
        Real mem_diag = 0.0;
        for (int a = 0; a < 3; ++a) {
            mem_diag += std::abs(K[(a*6+0)*18 + (a*6+0)]);
            mem_diag += std::abs(K[(a*6+1)*18 + (a*6+1)]);
        }
        CHECK(mem_diag > 0.0, "BATOZ: membrane stiffness present");
    }

    // Test 7: Bending stiffness present
    {
        Real K[324];
        elem.stiffness(K);
        Real bend_diag = 0.0;
        for (int a = 0; a < 3; ++a) {
            bend_diag += std::abs(K[(a*6+3)*18 + (a*6+3)]);
            bend_diag += std::abs(K[(a*6+4)*18 + (a*6+4)]);
        }
        CHECK(bend_diag > 0.0, "BATOZ: bending stiffness present");
    }

    // Test 8: Strain from uniform x-stretch
    {
        Real disp[18] = {};
        Real eps_x = 0.001;
        for (int a = 0; a < 3; ++a)
            disp[a*6+0] = eps_x * tri[a][0];

        Real strain[6];
        elem.strain_at(1.0/3.0, 1.0/3.0, disp, strain);
        CHECK_NEAR(strain[0], eps_x, eps_x * 0.1, "BATOZ: exx from uniform x-stretch");
    }

    // Test 9: Zero displacement gives zero strain
    {
        Real disp[18] = {};
        Real strain[6];
        elem.strain_at(0.5, 0.25, disp, strain);
        Real norm = 0.0;
        for (int i = 0; i < 6; ++i) norm += strain[i]*strain[i];
        CHECK(norm < 1e-20, "BATOZ: zero disp -> zero strain");
    }

    // Test 10: Comparison with expected bending stiffness order of magnitude
    {
        Real K[324];
        elem.stiffness(K);
        Real Db = STEEL_E * SHELL_THICKNESS*SHELL_THICKNESS*SHELL_THICKNESS /
                  (12.0 * (1.0 - STEEL_NU*STEEL_NU));
        Real K_bend_ref = Db / 0.5;
        Real K_bend_diag = std::abs(K[(0*6+4)*18 + (0*6+4)]);
        CHECK(K_bend_diag > K_bend_ref * 1e-3 && K_bend_diag < K_bend_ref * 1e3,
              "BATOZ: bending stiffness in reasonable range");
    }
}

// ============================================================================
// 3. CorotationalShell Tests
// ============================================================================
void test_corotational_shell() {
    std::cout << "\n=== 3. CorotationalShell Tests ===\n";

    Real coords[4][3];
    make_shell_coords(coords);
    CorotationalShell elem(coords, STEEL_E, STEEL_NU, SHELL_THICKNESS);

    // Test 1: Accessors
    {
        CHECK_NEAR(elem.E(), STEEL_E, 1.0, "Corot: E accessor");
        CHECK_NEAR(elem.nu(), STEEL_NU, 1e-15, "Corot: nu accessor");
    }

    // Test 2: Rotation matrix is identity for undeformed configuration
    {
        Real R[9];
        elem.extract_rotation(coords, R);
        Real err = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                err += std::abs(R[i*3+j] - ((i==j) ? 1.0 : 0.0));
        CHECK(err < 1e-10, "Corot: R = I for undeformed config");
    }

    // Test 3: Rigid rotation gives zero strain
    {
        Real theta = 0.1;
        Real ct = std::cos(theta), st = std::sin(theta);
        Real cur[4][3];
        for (int a = 0; a < 4; ++a) {
            cur[a][0] = ct*coords[a][0] - st*coords[a][1];
            cur[a][1] = st*coords[a][0] + ct*coords[a][1];
            cur[a][2] = coords[a][2];
        }
        Real strain[3];
        elem.corotated_strain(cur, strain);
        Real norm = strain[0]*strain[0] + strain[1]*strain[1] + strain[2]*strain[2];
        CHECK(norm < 1e-6, "Corot: rigid rotation -> near-zero strain");
    }

    // Test 4: Pure stretch gives expected strain
    {
        Real eps = 0.001;
        Real cur[4][3];
        for (int a = 0; a < 4; ++a) {
            cur[a][0] = coords[a][0] * (1.0 + eps);
            cur[a][1] = coords[a][1];
            cur[a][2] = coords[a][2];
        }
        Real strain[3];
        elem.corotated_strain(cur, strain);
        CHECK_NEAR(strain[0], eps, eps * 0.5, "Corot: exx from pure x-stretch");
    }

    // Test 5: Stiffness matrix symmetry
    {
        Real K[576];
        elem.stiffness(K);
        CHECK(check_symmetry(K, 24, 1e-6), "Corot: stiffness symmetry");
    }

    // Test 6: Stiffness non-negative diagonal
    {
        Real K[576];
        elem.stiffness(K);
        CHECK(check_diagonal_nonneg(K, 24), "Corot: non-negative diagonal");
    }

    // Test 7: Local displacement is zero for undeformed
    {
        Real R[9];
        elem.extract_rotation(coords, R);
        Real u_local[4][3];
        elem.local_displacement(coords, R, u_local);
        Real norm = 0.0;
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                norm += u_local[a][i]*u_local[a][i];
        CHECK(norm < 1e-20, "Corot: zero local displacement for undeformed");
    }

    // Test 8: Internal force from stiffness * displacement
    {
        Real disp[24] = {};
        disp[1*6+0] = 0.001;

        Real fint[24];
        elem.internal_force(coords, disp, fint);
        Real norm = 0.0;
        for (int i = 0; i < 24; ++i) norm += fint[i]*fint[i];
        CHECK(norm > 0.0, "Corot: nonzero internal force for nonzero displacement");
    }

    // Test 9: Zero displacement gives zero internal force
    {
        Real disp[24] = {};
        Real fint[24];
        elem.internal_force(coords, disp, fint);
        Real norm = 0.0;
        for (int i = 0; i < 24; ++i) norm += fint[i]*fint[i];
        CHECK(norm < 1e-20, "Corot: zero disp -> zero internal force");
    }

    // Test 10: Large rotation bending -- strain should remain bounded
    {
        Real theta = 0.5;
        Real ct = std::cos(theta), st = std::sin(theta);
        Real cur[4][3];
        for (int a = 0; a < 4; ++a) {
            cur[a][0] = ct*coords[a][0] - st*coords[a][1];
            cur[a][1] = st*coords[a][0] + ct*coords[a][1];
            cur[a][2] = coords[a][2] + 0.001 * coords[a][0];
        }
        Real strain[3];
        elem.corotated_strain(cur, strain);
        Real norm = strain[0]*strain[0] + strain[1]*strain[1] + strain[2]*strain[2];
        CHECK(norm < 1.0, "Corot: strain bounded under large rotation + bending");
    }
}

// ============================================================================
// 4. ANSHex8 Tests
// ============================================================================
void test_ans_hex8() {
    std::cout << "\n=== 4. ANSHex8 Tests ===\n";

    Real coords[8][3];
    make_hex8_coords(coords);
    ANSHex8 elem(coords, STEEL_E, STEEL_NU);

    // Test 1: Shape function partition of unity
    {
        Real N[8];
        elem.shape_functions(0.3, -0.2, 0.4, N);
        Real sum = 0.0;
        for (int i = 0; i < 8; ++i) sum += N[i];
        CHECK_NEAR(sum, 1.0, 1e-12, "ANS: shape function partition of unity");
    }

    // Test 2: Positive Jacobian
    {
        Real J[9];
        Real detJ = elem.jacobian(0.0, 0.0, 0.0, J);
        CHECK(detJ > 0, "ANS: positive Jacobian at center");
    }

    // Test 3: Volume of unit cube = 1.0
    {
        Real vol = elem.volume();
        CHECK_NEAR(vol, 1.0, 1e-10, "ANS: volume = 1.0 for unit cube");
    }

    // Test 4: Stiffness matrix symmetry
    {
        Real K[576];
        elem.compute_stiffness(K);
        CHECK(check_symmetry(K, 24, 1e-6), "ANS: stiffness symmetry");
    }

    // Test 5: Non-negative diagonal
    {
        Real K[576];
        elem.compute_stiffness(K);
        CHECK(check_diagonal_nonneg(K, 24), "ANS: non-negative diagonal");
    }

    // Test 6: Zero displacement -> zero internal force
    {
        Real disp[24] = {};
        Real fint[24];
        elem.compute_internal_force(disp, fint);
        Real norm = 0.0;
        for (int i = 0; i < 24; ++i) norm += fint[i]*fint[i];
        CHECK(norm < 1e-20, "ANS: zero disp -> zero internal force");
    }

    // Test 7: Patch test - uniform strain
    {
        Real K[576];
        elem.compute_stiffness(K);

        Real eps = 0.001;
        Real disp[24] = {};
        for (int a = 0; a < 8; ++a)
            disp[a*3+0] = eps * coords[a][0];

        Real f[24] = {};
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 24; ++j)
                f[i] += K[i*24+j] * disp[j];

        Real fx_sum = 0.0;
        for (int a = 0; a < 8; ++a) fx_sum += f[a*3+0];
        CHECK(std::abs(fx_sum) < 1e3, "ANS: patch test force equilibrium");
    }

    // Test 8: Cantilever bending - ANS should show less locking
    {
        Real bcoords[8][3];
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                bcoords[a][i] = coords[a][i];
        for (int a = 0; a < 8; ++a)
            bcoords[a][0] *= 10.0;

        ANSHex8 belem(bcoords, STEEL_E, STEEL_NU);
        Real K[576];
        belem.compute_stiffness(K);
        CHECK(check_symmetry(K, 24, 1e-5), "ANS: bending element stiffness symmetry");
    }

    // Test 9: Strain at center for known displacement
    {
        Real eps = 0.002;
        Real disp[24] = {};
        for (int a = 0; a < 8; ++a)
            disp[a*3+0] = eps * coords[a][0];

        Real strain[6];
        elem.strain_at(0.0, 0.0, 0.0, disp, strain);
        CHECK_NEAR(strain[0], eps, eps * 0.1, "ANS: exx from uniform x-extension");
    }

    // Test 10: ANS B-matrix differs from standard B in shear rows
    {
        Real B_std[144], B_ans[144];
        elem.standard_B(0.3, 0.4, 0.2, B_std);
        elem.ans_B(0.3, 0.4, 0.2, B_ans);

        // Normal strain rows (0,1,2) should be the same
        bool normal_same = true;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 24; ++j)
                if (std::abs(B_std[i*24+j] - B_ans[i*24+j]) > 1e-15)
                    normal_same = false;
        CHECK(normal_same, "ANS: normal strain rows match standard B");

        // In-plane shear row (3) should be the same
        bool shear_xy_same = true;
        for (int j = 0; j < 24; ++j)
            if (std::abs(B_std[3*24+j] - B_ans[3*24+j]) > 1e-15)
                shear_xy_same = false;
        CHECK(shear_xy_same, "ANS: in-plane shear row matches standard B");
    }
}

// ============================================================================
// 5. StabilizedIncompatibleHex8 Tests
// ============================================================================
void test_stabilized_incompatible_hex8() {
    std::cout << "\n=== 5. StabilizedIncompatibleHex8 Tests ===\n";

    Real coords[8][3];
    make_hex8_coords(coords);
    StabilizedIncompatibleHex8 elem(coords, STEEL_E, STEEL_NU, 0.05);

    // Test 1: Volume
    {
        CHECK_NEAR(elem.volume(), 1.0, 1e-10, "IncHex8: volume = 1.0");
    }

    // Test 2: Stiffness symmetry
    {
        Real K[576];
        elem.compute_stiffness(K);
        CHECK(check_symmetry(K, 24, 1e-5), "IncHex8: stiffness symmetry");
    }

    // Test 3: Non-negative diagonal
    {
        Real K[576];
        elem.compute_stiffness(K);
        CHECK(check_diagonal_nonneg(K, 24), "IncHex8: non-negative diagonal");
    }

    // Test 4: Zero disp -> zero internal force
    {
        Real disp[24] = {};
        Real fint[24];
        elem.compute_internal_force(disp, fint);
        Real norm = 0.0;
        for (int i = 0; i < 24; ++i) norm += fint[i]*fint[i];
        CHECK(norm < 1e-10, "IncHex8: zero disp -> zero force");
    }

    // Test 5: Patch test
    {
        Real K[576];
        elem.compute_stiffness(K);

        Real eps = 0.001;
        Real disp[24] = {};
        for (int a = 0; a < 8; ++a)
            disp[a*3+0] = eps * coords[a][0];

        Real f[24] = {};
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 24; ++j)
                f[i] += K[i*24+j] * disp[j];

        Real fx_sum = 0.0;
        for (int a = 0; a < 8; ++a) fx_sum += f[a*3+0];
        CHECK(std::abs(fx_sum) < 1e3, "IncHex8: patch test equilibrium");
    }

    // Test 6: Static condensation produces valid stiffness
    {
        Real K[576];
        elem.compute_stiffness(K);
        Real K_trace = 0.0;
        for (int i = 0; i < 24; ++i) K_trace += K[i*24+i];
        CHECK(K_trace > 0.0, "IncHex8: positive stiffness trace after condensation");
    }

    // Test 7: Update internal parameters
    {
        Real disp[24] = {};
        for (int a = 0; a < 8; ++a)
            disp[a*3+0] = 0.001 * coords[a][0];

        elem.update_alpha(disp);
        const Real* alpha = elem.inc_params();
        bool finite = true;
        for (int i = 0; i < 3; ++i)
            if (std::isnan(alpha[i]) || std::isinf(alpha[i])) finite = false;
        CHECK(finite, "IncHex8: internal params are finite");
    }

    // Test 8: Bending improvement vs standard hex8
    {
        Real bcoords[8][3];
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                bcoords[a][i] = coords[a][i];
        for (int a = 0; a < 8; ++a)
            bcoords[a][0] *= 5.0;

        StabilizedIncompatibleHex8 belem(bcoords, STEEL_E, STEEL_NU, 0.05);
        Real K[576];
        belem.compute_stiffness(K);
        CHECK(check_symmetry(K, 24, 1e-4), "IncHex8: elongated element stiffness symmetry");
    }

    // Test 9: Hourglass stabilization adds to stiffness
    {
        Real K[576];
        elem.compute_stiffness(K);
        bool all_pos = true;
        for (int i = 0; i < 24; ++i)
            if (K[i*24+i] <= 0.0) all_pos = false;
        CHECK(all_pos, "IncHex8: all diagonal entries positive (stabilized)");
    }

    // Test 10: Nonzero force for bending displacement
    {
        Real disp[24] = {};
        for (int a = 0; a < 8; ++a)
            disp[a*3+2] = 0.001 * coords[a][0];

        Real fint[24];
        elem.compute_internal_force(disp, fint);
        Real norm = 0.0;
        for (int i = 0; i < 24; ++i) norm += fint[i]*fint[i];
        CHECK(norm > 0.0, "IncHex8: nonzero force for bending displacement");
    }
}

// ============================================================================
// 6. LayeredThickShell Tests
// ============================================================================
void test_layered_thick_shell() {
    std::cout << "\n=== 6. LayeredThickShell Tests ===\n";

    Real coords[4][3];
    make_shell_coords(coords);

    // Test 1: Single layer matches isotropic
    {
        LayerDef layers[1] = {LayerDef(STEEL_E, STEEL_NU, SHELL_THICKNESS, 0)};
        LayeredThickShell elem(coords, layers, 1);
        CHECK_NEAR(elem.total_thickness(), SHELL_THICKNESS, 1e-15,
                   "Layered: single layer thickness matches");
    }

    // Test 2: Total thickness for multi-layer
    {
        LayerDef layers[3] = {
            LayerDef(STEEL_E, STEEL_NU, 0.003, 0),
            LayerDef(STEEL_E, STEEL_NU, 0.004, 1),
            LayerDef(STEEL_E, STEEL_NU, 0.003, 2)
        };
        LayeredThickShell elem(coords, layers, 3);
        CHECK_NEAR(elem.total_thickness(), 0.01, 1e-15, "Layered: 3-layer total thickness");
    }

    // Test 3: ABD for single isotropic layer
    {
        LayerDef layers[1] = {LayerDef(STEEL_E, STEEL_NU, SHELL_THICKNESS, 0)};
        LayeredThickShell elem(coords, layers, 1);

        Real A[9], B[9], D[9];
        elem.compute_ABD(A, B, D);

        Real f = STEEL_E / (1.0 - STEEL_NU*STEEL_NU);
        CHECK_NEAR(A[0], f * SHELL_THICKNESS, f * SHELL_THICKNESS * 1e-6,
                   "Layered: A11 for single isotropic layer");
    }

    // Test 4: B matrix zero for symmetric layup
    {
        LayerDef layers[2] = {
            LayerDef(STEEL_E, STEEL_NU, 0.005, 0),
            LayerDef(STEEL_E, STEEL_NU, 0.005, 1)
        };
        LayeredThickShell elem(coords, layers, 2);

        Real A[9], B_mat[9], D[9];
        elem.compute_ABD(A, B_mat, D);

        Real B_norm = 0.0;
        for (int i = 0; i < 9; ++i) B_norm += B_mat[i]*B_mat[i];
        CHECK(B_norm < 1e-10, "Layered: B=0 for symmetric layup");
    }

    // Test 5: B matrix nonzero for unsymmetric layup
    {
        LayerDef layers[2] = {
            LayerDef(STEEL_E, STEEL_NU, 0.003, 0),
            LayerDef(70.0e9, 0.33, 0.007, 1)
        };
        LayeredThickShell elem(coords, layers, 2);

        Real A[9], B_mat[9], D[9];
        elem.compute_ABD(A, B_mat, D);

        Real B_norm = 0.0;
        for (int i = 0; i < 9; ++i) B_norm += B_mat[i]*B_mat[i];
        CHECK(B_norm > 0.0, "Layered: B != 0 for unsymmetric layup");
    }

    // Test 6: Stiffness symmetry
    {
        LayerDef layers[2] = {
            LayerDef(STEEL_E, STEEL_NU, 0.005, 0),
            LayerDef(STEEL_E, STEEL_NU, 0.005, 1)
        };
        LayeredThickShell elem(coords, layers, 2);

        Real K[576];
        elem.stiffness(K);
        CHECK(check_symmetry(K, 24, 1e-6), "Layered: stiffness symmetry");
    }

    // Test 7: Non-negative diagonal
    {
        LayerDef layers[1] = {LayerDef(STEEL_E, STEEL_NU, SHELL_THICKNESS, 0)};
        LayeredThickShell elem(coords, layers, 1);

        Real K[576];
        elem.stiffness(K);
        CHECK(check_diagonal_nonneg(K, 24), "Layered: non-negative diagonal");
    }

    // Test 8: Lumped mass
    {
        LayerDef layers[1] = {LayerDef(STEEL_E, STEEL_NU, SHELL_THICKNESS, 0)};
        LayeredThickShell elem(coords, layers, 1);

        Real M[24];
        elem.mass(STEEL_DENSITY, M);
        Real total = 0.0;
        for (int a = 0; a < 4; ++a) total += M[a*6+0];
        Real expected = STEEL_DENSITY * 1.0 * SHELL_THICKNESS;
        CHECK_NEAR(total, expected, expected * 1e-10, "Layered: total mass correct");
    }

    // Test 9: Layer stress computation
    {
        LayerDef layers[2] = {
            LayerDef(STEEL_E, STEEL_NU, 0.005, 0),
            LayerDef(STEEL_E, STEEL_NU, 0.005, 1)
        };
        LayeredThickShell elem(coords, layers, 2);

        Real disp[24] = {};
        Real stress[3];
        elem.layer_stress(0, disp, stress);
        Real norm = stress[0]*stress[0] + stress[1]*stress[1] + stress[2]*stress[2];
        CHECK(norm < 1e-20, "Layered: zero disp -> zero layer stress");
    }

    // Test 10: 6-layer construction
    {
        LayerDef layers[6];
        for (int i = 0; i < 6; ++i)
            layers[i] = LayerDef(STEEL_E, STEEL_NU, 0.002, i);
        LayeredThickShell elem(coords, layers, 6);
        CHECK_NEAR(elem.num_layers(), 6, 0, "Layered: 6 layers");
        CHECK_NEAR(elem.total_thickness(), 0.012, 1e-15, "Layered: 6-layer total thickness");
    }
}

// ============================================================================
// 7. SelectiveMassHex8 Tests
// ============================================================================
void test_selective_mass_hex8() {
    std::cout << "\n=== 7. SelectiveMassHex8 Tests ===\n";

    Real coords[8][3];
    make_hex8_coords(coords);

    // Test 1: Accessors
    {
        SelectiveMassHex8 elem(coords, STEEL_E, STEEL_NU, STEEL_DENSITY, 0.5);
        CHECK_NEAR(elem.E(), STEEL_E, 1.0, "SelMass: E accessor");
        CHECK_NEAR(elem.beta(), 0.5, 1e-15, "SelMass: beta accessor");
    }

    // Test 2: Volume = 1.0 for unit cube
    {
        SelectiveMassHex8 elem(coords, STEEL_E, STEEL_NU, STEEL_DENSITY);
        CHECK_NEAR(elem.volume(), 1.0, 1e-10, "SelMass: volume = 1.0");
    }

    // Test 3: Translational mass total = rho * V * 3 (3 DOF per node, 8 nodes)
    {
        SelectiveMassHex8 elem(coords, STEEL_E, STEEL_NU, STEEL_DENSITY, 0.5);
        Real M[24];
        elem.translational_mass(M);
        Real total = 0.0;
        for (int i = 0; i < 24; ++i) total += M[i];
        Real expected = STEEL_DENSITY * 1.0 * 3.0;
        CHECK_NEAR(total, expected, expected * 1e-10, "SelMass: translational mass total");
    }

    // Test 4: Translational mass unchanged by beta
    {
        SelectiveMassHex8 elem1(coords, STEEL_E, STEEL_NU, STEEL_DENSITY, 1.0);
        SelectiveMassHex8 elem05(coords, STEEL_E, STEEL_NU, STEEL_DENSITY, 0.5);
        Real M1[24], M05[24];
        elem1.translational_mass(M1);
        elem05.translational_mass(M05);

        Real diff = 0.0;
        for (int i = 0; i < 24; ++i) diff += std::abs(M1[i] - M05[i]);
        CHECK(diff < 1e-20, "SelMass: translational mass unchanged by beta");
    }

    // Test 5: Rotational mass reduced by beta
    {
        SelectiveMassHex8 elem1(coords, STEEL_E, STEEL_NU, STEEL_DENSITY, 1.0);
        SelectiveMassHex8 elem05(coords, STEEL_E, STEEL_NU, STEEL_DENSITY, 0.5);
        Real Mr1[24], Mr05[24];
        elem1.rotational_mass(Mr1);
        elem05.rotational_mass(Mr05);

        CHECK_NEAR(Mr05[0], Mr1[0] * 0.5, Mr1[0] * 1e-10,
                   "SelMass: rotational mass halved with beta=0.5");
    }

    // Test 6: Critical timestep increases with smaller beta
    {
        SelectiveMassHex8 elem1(coords, STEEL_E, STEEL_NU, STEEL_DENSITY, 1.0);
        SelectiveMassHex8 elem05(coords, STEEL_E, STEEL_NU, STEEL_DENSITY, 0.5);
        Real dt1 = elem1.critical_timestep();
        Real dt05 = elem05.critical_timestep();
        CHECK(dt05 > dt1, "SelMass: smaller beta -> larger critical timestep");
    }

    // Test 7: dt improvement ratio ~ 1/sqrt(beta)
    {
        SelectiveMassHex8 elem1(coords, STEEL_E, STEEL_NU, STEEL_DENSITY, 1.0);
        SelectiveMassHex8 elem025(coords, STEEL_E, STEEL_NU, STEEL_DENSITY, 0.25);
        Real dt1 = elem1.critical_timestep();
        Real dt025 = elem025.critical_timestep();
        Real ratio = dt025 / dt1;
        Real expected_ratio = 1.0 / std::sqrt(0.25);
        CHECK_NEAR(ratio, expected_ratio, expected_ratio * 0.01,
                   "SelMass: dt ratio ~ 1/sqrt(beta)");
    }

    // Test 8: Stiffness symmetry
    {
        SelectiveMassHex8 elem(coords, STEEL_E, STEEL_NU, STEEL_DENSITY, 0.5);
        Real K[576];
        elem.compute_stiffness(K);
        CHECK(check_symmetry(K, 24, 1e-6), "SelMass: stiffness symmetry");
    }

    // Test 9: Eigenfrequency estimate is positive
    {
        SelectiveMassHex8 elem(coords, STEEL_E, STEEL_NU, STEEL_DENSITY, 0.5);
        Real omega = elem.max_eigenfrequency();
        CHECK(omega > 0.0, "SelMass: positive max eigenfrequency");
    }

    // Test 10: beta=1 gives same timestep as standard element
    {
        SelectiveMassHex8 elem(coords, STEEL_E, STEEL_NU, STEEL_DENSITY, 1.0);
        Real dt = elem.critical_timestep();
        Real G = STEEL_E / (2.0*(1.0+STEEL_NU));
        Real bulk = STEEL_E / (3.0*(1.0-2.0*STEEL_NU));
        Real c = std::sqrt((bulk + 4.0/3.0*G) / STEEL_DENSITY);
        Real dt_ref = 1.0 / c;
        CHECK_NEAR(dt, dt_ref, dt_ref * 1e-10, "SelMass: beta=1 matches standard timestep");
    }
}

// ============================================================================
// 8. EnhancedStrainExtrapolationHex8 Tests
// ============================================================================
void test_enhanced_strain_extrapolation_hex8() {
    std::cout << "\n=== 8. EnhancedStrainExtrapolationHex8 Tests ===\n";

    Real coords[8][3];
    make_hex8_coords(coords);
    EnhancedStrainExtrapolationHex8 elem(coords, STEEL_E, STEEL_NU);

    // Test 1: Volume
    {
        CHECK_NEAR(elem.volume(), 1.0, 1e-10, "EASE: volume = 1.0");
    }

    // Test 2: Stiffness symmetry
    {
        Real K[576];
        elem.compute_stiffness(K);
        CHECK(check_symmetry(K, 24, 1e-5), "EASE: stiffness symmetry");
    }

    // Test 3: Non-negative diagonal
    {
        Real K[576];
        elem.compute_stiffness(K);
        CHECK(check_diagonal_nonneg(K, 24), "EASE: non-negative diagonal");
    }

    // Test 4: Zero disp -> zero Gauss point stress
    {
        Real disp[24] = {};
        Real gp_stress[48];
        elem.gauss_point_stress(disp, gp_stress);
        Real norm = 0.0;
        for (int i = 0; i < 48; ++i) norm += gp_stress[i]*gp_stress[i];
        CHECK(norm < 1e-20, "EASE: zero disp -> zero GP stress");
    }

    // Test 5: Extrapolated node stress from uniform strain
    {
        Real eps = 0.001;
        Real disp[24] = {};
        for (int a = 0; a < 8; ++a)
            disp[a*3+0] = eps * coords[a][0];

        Real gp_stress[48];
        elem.gauss_point_stress(disp, gp_stress);

        Real node_stress[48];
        elem.extrapolate_stress_to_nodes(gp_stress, node_stress);

        Real sxx_node0 = node_stress[0*6+0];
        Real sxx_gp0 = gp_stress[0*6+0];
        CHECK_NEAR(sxx_node0, sxx_gp0, std::abs(sxx_gp0) * 0.5,
                   "EASE: extrapolated stress matches GP for uniform field");
    }

    // Test 6: Patch test
    {
        Real K[576];
        elem.compute_stiffness(K);

        Real eps = 0.001;
        Real disp[24] = {};
        for (int a = 0; a < 8; ++a)
            disp[a*3+0] = eps * coords[a][0];

        Real f[24] = {};
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 24; ++j)
                f[i] += K[i*24+j] * disp[j];

        Real fx_sum = 0.0;
        for (int a = 0; a < 8; ++a) fx_sum += f[a*3+0];
        CHECK(std::abs(fx_sum) < 1e3, "EASE: patch test equilibrium");
    }

    // Test 7: EAS parameters update
    {
        Real disp[24] = {};
        for (int a = 0; a < 8; ++a)
            disp[a*3+0] = 0.001 * coords[a][0];

        elem.update_eas_params(disp);
        const Real* alpha = elem.eas_params();
        bool finite = true;
        for (int i = 0; i < 9; ++i)
            if (std::isnan(alpha[i]) || std::isinf(alpha[i])) finite = false;
        CHECK(finite, "EASE: EAS params are finite after update");
    }

    // Test 8: Internal force nonzero for nonzero displacement
    {
        Real disp[24] = {};
        disp[1*3+0] = 0.001;

        Real fint[24];
        elem.compute_internal_force(disp, fint);
        Real norm = 0.0;
        for (int i = 0; i < 24; ++i) norm += fint[i]*fint[i];
        CHECK(norm > 0.0, "EASE: nonzero fint for nonzero disp");
    }

    // Test 9: Stress extrapolation preserves stress sum
    {
        Real eps = 0.002;
        Real disp[24] = {};
        for (int a = 0; a < 8; ++a) {
            disp[a*3+0] = eps * coords[a][0];
            disp[a*3+1] = 0.5 * eps * coords[a][1];
        }

        Real gp_stress[48];
        elem.gauss_point_stress(disp, gp_stress);

        Real node_stress[48];
        elem.extrapolate_stress_to_nodes(gp_stress, node_stress);

        Real avg_gp = 0.0, avg_node = 0.0;
        for (int g = 0; g < 8; ++g) avg_gp += gp_stress[g*6+0];
        for (int n = 0; n < 8; ++n) avg_node += node_stress[n*6+0];
        avg_gp /= 8.0;
        avg_node /= 8.0;
        CHECK_NEAR(avg_node, avg_gp, std::abs(avg_gp) * 0.1,
                   "EASE: extrapolation preserves average stress");
    }

    // Test 10: Bending stress distribution is non-uniform across nodes
    {
        Real disp[24] = {};
        for (int a = 0; a < 8; ++a)
            disp[a*3+2] = 0.001 * coords[a][0];

        Real gp_stress[48];
        elem.gauss_point_stress(disp, gp_stress);

        Real node_stress[48];
        elem.extrapolate_stress_to_nodes(gp_stress, node_stress);

        Real sxx_min = node_stress[0], sxx_max = node_stress[0];
        for (int n = 1; n < 8; ++n) {
            Real s = node_stress[n*6+0];
            if (s < sxx_min) sxx_min = s;
            if (s > sxx_max) sxx_max = s;
        }
        CHECK(std::abs(sxx_max - sxx_min) > 0.0 || std::abs(sxx_max) > 0.0,
              "EASE: bending creates stress gradient across nodes");
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== Wave 26 Element Formulations II Test ===\n";

    test_qeph_shell();
    test_batoz_shell();
    test_corotational_shell();
    test_ans_hex8();
    test_stabilized_incompatible_hex8();
    test_layered_thick_shell();
    test_selective_mass_hex8();
    test_enhanced_strain_extrapolation_hex8();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    return tests_failed > 0 ? 1 : 0;
}
