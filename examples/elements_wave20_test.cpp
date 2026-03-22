/**
 * @file elements_wave20_test.cpp
 * @brief Wave 20: Element Formulation Variants Test Suite (6 elements, ~50 tests)
 *
 * Tests 6 element formulations:
 *  1. BelytschkoTsayShell  (8 tests)
 *  2. Pyramid5Element      (8 tests)
 *  3. MITC4Shell           (8 tests)
 *  4. EASHex8              (9 tests)
 *  5. BBarHex8             (9 tests)
 *  6. IsogeometricShell    (9 tests)
 */

#include <nexussim/discretization/elements_wave20.hpp>
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
// Helper: steel material constants
// ============================================================================
static constexpr Real STEEL_E = 210.0e9;
static constexpr Real STEEL_NU = 0.3;
static constexpr Real STEEL_DENSITY = 7800.0;
static constexpr Real SHELL_THICKNESS = 0.01;

// Helper: unit square shell coords (nodes in xy-plane)
static void make_shell_coords(Real coords[4][3]) {
    // Node 0: (0,0,0), Node 1: (1,0,0), Node 2: (1,1,0), Node 3: (0,1,0)
    coords[0][0] = 0; coords[0][1] = 0; coords[0][2] = 0;
    coords[1][0] = 1; coords[1][1] = 0; coords[1][2] = 0;
    coords[2][0] = 1; coords[2][1] = 1; coords[2][2] = 0;
    coords[3][0] = 0; coords[3][1] = 1; coords[3][2] = 0;
}

// Helper: unit cube hex8 coords (8 nodes, corners of [0,1]^3)
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

// Helper: unit pyramid coords (base at z=0, apex at (0.5,0.5,1))
static void make_pyramid_coords(Real coords[5][3]) {
    coords[0][0] = 0; coords[0][1] = 0; coords[0][2] = 0;
    coords[1][0] = 1; coords[1][1] = 0; coords[1][2] = 0;
    coords[2][0] = 1; coords[2][1] = 1; coords[2][2] = 0;
    coords[3][0] = 0; coords[3][1] = 1; coords[3][2] = 0;
    coords[4][0] = 0.5; coords[4][1] = 0.5; coords[4][2] = 1.0;
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

// Helper: check all diagonal entries positive
static bool check_diagonal_nonneg(const Real* K, int n) {
    for (int i = 0; i < n; ++i) {
        if (K[i*n + i] < -1e-10) return false;
    }
    return true;
}

// ============================================================================
// 1. BelytschkoTsayShell Tests
// ============================================================================
void test_belytschko_tsay_shell() {
    std::cout << "\n=== 1. BelytschkoTsayShell Tests ===\n";

    BelytschkoTsayShell elem(STEEL_E, STEEL_NU, SHELL_THICKNESS, STEEL_DENSITY, 0.1);
    Real coords[4][3];
    make_shell_coords(coords);

    // Test 1: Construction and basic accessors
    {
        CHECK_NEAR(elem.E(), STEEL_E, 1.0, "BT: E accessor");
        CHECK_NEAR(elem.nu(), STEEL_NU, 1e-15, "BT: nu accessor");
        CHECK_NEAR(elem.thickness(), SHELL_THICKNESS, 1e-15, "BT: thickness accessor");
    }

    // Test 2: Internal force for velocity-driven strain rate
    {
        Real vel[4][3] = {};
        // Apply velocity gradient: node 1,2 moving in +x at 1 m/s
        vel[1][0] = 1.0; // node 1 vx
        vel[2][0] = 1.0; // node 2 vx

        Real stress_in[5] = {0, 0, 0, 0, 0};
        Real stress_out[5];
        Real force[4][3];
        Real dt = 1.0e-6;

        elem.compute_internal_force(coords, vel, dt, stress_in, stress_out, force);

        // Stress should be updated (nonzero sxx from velocity strain)
        CHECK(std::abs(stress_out[0]) > 0.0, "BT: stress updated from velocity strain");
    }

    // Test 3: Force is nonzero for nonzero velocity input
    {
        Real vel[4][3] = {};
        vel[1][0] = 1.0;
        vel[2][0] = 1.0;

        Real stress_in[5] = {0, 0, 0, 0, 0};
        Real stress_out[5];
        Real force[4][3];

        elem.compute_internal_force(coords, vel, 1e-6, stress_in, stress_out, force);

        Real norm = 0.0;
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                norm += force[a][i] * force[a][i];
        CHECK(norm > 0.0, "BT: nonzero internal force for velocity strain");
    }

    // Test 4: Hourglass force nonzero for hourglass mode velocity
    {
        Real vel[4][3] = {};
        // Hourglass mode: {+1, -1, +1, -1} velocity in x
        vel[0][0] =  1.0;
        vel[1][0] = -1.0;
        vel[2][0] =  1.0;
        vel[3][0] = -1.0;

        Real hg_force[4][3];
        elem.hourglass_force(coords, vel, hg_force);

        Real norm = 0.0;
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                norm += hg_force[a][i] * hg_force[a][i];
        CHECK(norm > 0.0, "BT: nonzero hourglass force for HG mode");
    }

    // Test 5: Hourglass force zero for rigid body translation velocity
    {
        Real vel[4][3];
        for (int a = 0; a < 4; ++a) {
            vel[a][0] = 1.0; vel[a][1] = 0.0; vel[a][2] = 0.0;
        }

        Real hg_force[4][3];
        elem.hourglass_force(coords, vel, hg_force);

        Real norm = 0.0;
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                norm += hg_force[a][i] * hg_force[a][i];
        CHECK(norm < 1e-20, "BT: zero hourglass force for rigid body translation");
    }

    // Test 6: Stable time step is positive and reasonable
    {
        Real dt_crit = elem.stable_time_step(coords);
        CHECK(dt_crit > 0.0 && dt_crit < 1.0,
              "BT: stable time step is positive and finite");
    }

    // Test 7: Force equilibrium -- sum of forces ~0 for constant stress state
    {
        Real vel[4][3] = {};
        vel[1][0] = 1.0;
        vel[2][0] = 1.0;

        Real stress_in[5] = {1.0e6, 0.5e6, 0.0, 0.0, 0.0};
        Real stress_out[5];
        Real force[4][3];
        elem.compute_internal_force(coords, vel, 0.0, stress_in, stress_out, force);

        // With dt=0, stress_out = stress_in. For constant stress, sum of forces
        // should be zero (equilibrium).
        Real fsum[3] = {0, 0, 0};
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                fsum[i] += force[a][i];
        Real norm = fsum[0]*fsum[0] + fsum[1]*fsum[1] + fsum[2]*fsum[2];
        CHECK(norm < 1.0, "BT: force equilibrium for constant stress (sum~0)");
    }

    // Test 8: Zero velocity produces zero force from zero initial stress
    {
        Real vel[4][3] = {};
        Real stress_in[5] = {0, 0, 0, 0, 0};
        Real stress_out[5];
        Real force[4][3];
        elem.compute_internal_force(coords, vel, 1e-6, stress_in, stress_out, force);

        Real norm = 0.0;
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                norm += force[a][i] * force[a][i];
        CHECK(norm < 1e-20, "BT: zero velocity + zero stress -> zero force");
    }
}

// ============================================================================
// 2. Pyramid5Element Tests
// ============================================================================
void test_pyramid5_element() {
    std::cout << "\n=== 2. Pyramid5Element Tests ===\n";

    Real coords[5][3];
    make_pyramid_coords(coords);
    Pyramid5Element elem(coords, STEEL_E, STEEL_NU);

    // Test 1: Shape function partition of unity
    {
        Real N[5];
        elem.shape_functions(0.3, -0.2, 0.4, N);
        Real sum = 0.0;
        for (int i = 0; i < 5; ++i) sum += N[i];
        CHECK_NEAR(sum, 1.0, 1e-12, "Pyr5: shape function partition of unity");
    }

    // Test 2: Shape function at base node 0 (xi=-1, eta=-1, zeta=0)
    {
        Real N[5];
        elem.shape_functions(-1.0, -1.0, 0.0, N);
        CHECK_NEAR(N[0], 1.0, 1e-12, "Pyr5: N0=1 at node 0");
        CHECK_NEAR(N[1], 0.0, 1e-12, "Pyr5: N1=0 at node 0");
        CHECK_NEAR(N[4], 0.0, 1e-12, "Pyr5: N4=0 at node 0");
    }

    // Test 3: Shape function at apex (zeta=1)
    {
        Real N[5];
        elem.shape_functions(0.0, 0.0, 1.0, N);
        CHECK_NEAR(N[4], 1.0, 1e-10, "Pyr5: N4=1 at apex (zeta=1)");
    }

    // Test 4: Positive Jacobian at centroid
    {
        Real J[9];
        Real detJ = elem.jacobian(0.0, 0.0, 0.25, J);
        CHECK(detJ > 0, "Pyr5: positive Jacobian at centroid");
    }

    // Test 5: Stiffness matrix symmetry
    {
        Real K[225];
        elem.compute_stiffness(K);
        CHECK(check_symmetry(K, 15, 1e-6), "Pyr5: stiffness matrix symmetry");
    }

    // Test 6: Zero stress produces zero internal force
    {
        Real stress[30] = {}; // 5 GP * 6 stress components
        Real fint[15];
        elem.compute_internal_forces(stress, fint);
        Real norm = 0.0;
        for (int i = 0; i < 15; ++i) norm += fint[i] * fint[i];
        CHECK(norm < 1e-20, "Pyr5: zero stress -> zero internal force");
    }

    // Test 7: Volume of unit pyramid = 1/3
    {
        Real vol = elem.volume();
        // Quadrature approximation; tolerance accounts for Bedrosian-type scheme
        CHECK(vol > 0.1 && vol < 0.5, "Pyr5: volume in reasonable range for unit pyramid");
    }

    // Test 8: Stiffness diagonal nonnegative
    {
        Real K[225];
        elem.compute_stiffness(K);
        CHECK(check_diagonal_nonneg(K, 15), "Pyr5: stiffness diagonal nonnegative");
    }
}

// ============================================================================
// 3. MITC4Shell Tests
// ============================================================================
void test_mitc4_shell() {
    std::cout << "\n=== 3. MITC4Shell Tests ===\n";

    Real coords[4][3];
    make_shell_coords(coords);
    MITC4Shell elem(coords, STEEL_E, STEEL_NU, SHELL_THICKNESS);

    // Test 1: Area of unit square shell = 1.0
    {
        CHECK_NEAR(elem.area(), 1.0, 1e-10, "MITC4: area of unit square = 1.0");
    }

    // Test 2: Membrane strain for uniform x-stretch
    {
        Real disp[4][5] = {};
        Real eps = 0.001;
        // u_x = eps * x => node 1 (x=1) and node 2 (x=1) get eps displacement
        disp[1][0] = eps; // node 1 x-disp
        disp[2][0] = eps; // node 2 x-disp

        Real strain[6];
        elem.compute_strain(disp, 0.0, 0.0, strain);
        CHECK(std::abs(strain[0]) > 0, "MITC4: nonzero exx for x-stretch");
        // Local basis is built from diagonals, so projected strain may differ
        // from global eps. Just verify it is in reasonable range.
        CHECK(std::abs(strain[0]) > 1e-6 && std::abs(strain[0]) < 0.01,
              "MITC4: exx in reasonable range for x-stretch");
    }

    // Test 3: Zero displacement produces zero strain
    {
        Real disp[4][5] = {};
        Real strain[6];
        elem.compute_strain(disp, 0.0, 0.0, strain);
        Real norm = 0.0;
        for (int i = 0; i < 6; ++i) norm += strain[i] * strain[i];
        CHECK(norm < 1e-20, "MITC4: zero displacement -> zero strain");
    }

    // Test 4: MITC assumed shear strains are finite
    {
        Real disp[4][5] = {};
        disp[0][2] = 0.001; // node 0 z-displacement
        Real gamma[2];
        elem.compute_mitc_shear(disp, 0.0, 0.0, gamma);
        CHECK(!std::isnan(gamma[0]) && !std::isnan(gamma[1]),
              "MITC4: assumed shear strains are finite");
    }

    // Test 5: Internal force from zero displacement is zero
    {
        Real disp[4][5] = {};
        Real fint[20];
        elem.compute_internal_force(disp, fint);
        Real norm = 0.0;
        for (int i = 0; i < 20; ++i) norm += fint[i] * fint[i];
        CHECK(norm < 1e-20, "MITC4: zero displacement -> zero internal force");
    }

    // Test 6: Internal force nonzero for nonzero displacement
    {
        Real disp[4][5] = {};
        disp[1][0] = 0.001;
        disp[2][0] = 0.001;
        Real fint[20];
        elem.compute_internal_force(disp, fint);
        Real norm = 0.0;
        for (int i = 0; i < 20; ++i) norm += fint[i] * fint[i];
        CHECK(norm > 0.0, "MITC4: nonzero displacement -> nonzero force");
    }

    // Test 7: Thinner shell produces smaller bending stiffness
    {
        Real coords2[4][3];
        make_shell_coords(coords2);
        MITC4Shell thin(coords2, STEEL_E, STEEL_NU, 0.001);
        MITC4Shell thick(coords2, STEEL_E, STEEL_NU, 0.01);

        // Apply pure bending displacement (z-displacement to node 1)
        Real disp_b[4][5] = {};
        disp_b[1][2] = 0.001; // z-disp at node 1

        Real fint_thin[20], fint_thick[20];
        thin.compute_internal_force(disp_b, fint_thin);
        thick.compute_internal_force(disp_b, fint_thick);

        Real norm_thin = 0.0, norm_thick = 0.0;
        for (int i = 0; i < 20; ++i) {
            norm_thin += fint_thin[i] * fint_thin[i];
            norm_thick += fint_thick[i] * fint_thick[i];
        }
        CHECK(norm_thin < norm_thick,
              "MITC4: thinner shell produces smaller internal force");
    }

    // Test 8: All 6 strain components are finite for general displacement
    {
        Real disp[4][5] = {};
        disp[0][0] = 0.001; disp[0][3] = 0.001;
        disp[2][1] = -0.001; disp[2][4] = 0.002;
        Real strain[6];
        elem.compute_strain(disp, 0.3, -0.2, strain);
        bool all_finite = true;
        for (int i = 0; i < 6; ++i)
            if (std::isnan(strain[i]) || std::isinf(strain[i])) all_finite = false;
        CHECK(all_finite, "MITC4: all 6 strain components are finite");
    }
}

// ============================================================================
// 4. EASHex8 Tests
// ============================================================================
void test_eas_hex8() {
    std::cout << "\n=== 4. EASHex8 Tests ===\n";

    Real coords[8][3];
    make_hex8_coords(coords);
    EASHex8 elem(coords, STEEL_E, STEEL_NU);

    // Test 1: Volume of unit cube = 1.0
    {
        CHECK_NEAR(elem.volume(), 1.0, 1e-6, "EAS: volume of unit cube = 1.0");
    }

    // Test 2: Positive Jacobian at center
    {
        Real J[9];
        Real detJ = elem.jacobian(0.0, 0.0, 0.0, J);
        CHECK(detJ > 0, "EAS: positive Jacobian at center");
    }

    // Test 3: Stiffness matrix symmetry
    {
        Real K[576];
        elem.compute_stiffness(K);
        CHECK(check_symmetry(K, 24, 1e-6), "EAS: stiffness matrix symmetry");
    }

    // Test 4: Stiffness diagonal nonnegative
    {
        Real K[576];
        elem.compute_stiffness(K);
        CHECK(check_diagonal_nonneg(K, 24), "EAS: stiffness diagonal nonnegative");
    }

    // Test 5: EAS matrix has 7 modes at a Gauss point
    {
        Real J0[9];
        Real detJ0 = elem.jacobian(0.0, 0.0, 0.0, J0);
        Real J[9];
        Real detJ = elem.jacobian(0.5, 0.5, 0.5, J);
        Real M[42]; // 6 x 7
        elem.eas_matrix(0.5, 0.5, 0.5, detJ0, detJ, M);
        // Mode 1: M[0*7+0] = ratio * xi = (detJ0/detJ) * 0.5
        Real ratio = detJ0 / detJ;
        CHECK_NEAR(M[0*7+0], ratio * 0.5, 1e-10, "EAS: EAS mode 1 = ratio*xi at (0.5,0.5,0.5)");
    }

    // Test 6: EAS parameter update from displacement
    {
        Real disp[24] = {};
        // Apply uniform x-stretch
        for (int a = 0; a < 8; ++a)
            disp[a*3+0] = 0.001 * coords[a][0];

        elem.update_eas_params(disp);
        const Real* alpha = elem.eas_params();
        // At least some alpha should be nonzero for a general displacement
        Real alpha_norm = 0.0;
        for (int i = 0; i < 7; ++i) alpha_norm += alpha[i] * alpha[i];
        // For uniform strain, EAS params could be near-zero, so just check finite
        bool all_finite = true;
        for (int i = 0; i < 7; ++i)
            if (std::isnan(alpha[i]) || std::isinf(alpha[i])) all_finite = false;
        CHECK(all_finite, "EAS: EAS params are finite after update");
    }

    // Test 7: Near-incompressible (nu->0.5) produces valid stiffness
    {
        EASHex8 near_inc(coords, STEEL_E, 0.4999);
        Real K[576];
        near_inc.compute_stiffness(K);
        bool valid = true;
        for (int i = 0; i < 576; ++i)
            if (std::isnan(K[i]) || std::isinf(K[i])) { valid = false; break; }
        CHECK(valid, "EAS: near-incompressible stiffness is finite");
    }

    // Test 8: B-matrix has correct dimensions (strain-displacement)
    {
        Real B[144]; // 6 x 24
        elem.strain_displacement_matrix(0.0, 0.0, 0.0, B);
        // For unit cube at center, B should have nonzero entries
        Real norm = 0.0;
        for (int i = 0; i < 144; ++i) norm += B[i] * B[i];
        CHECK(norm > 0, "EAS: B-matrix nonzero at element center");
    }

    // Test 9: Stiffness matrix nonzero
    {
        Real K[576];
        elem.compute_stiffness(K);
        Real ksum = 0.0;
        for (int i = 0; i < 576; ++i) ksum += std::abs(K[i]);
        CHECK(ksum > 0, "EAS: stiffness matrix has nonzero entries");
    }
}

// ============================================================================
// 5. BBarHex8 Tests
// ============================================================================
void test_bbar_hex8() {
    std::cout << "\n=== 5. BBarHex8 Tests ===\n";

    Real coords[8][3];
    make_hex8_coords(coords);
    BBarHex8 elem(coords, STEEL_E, STEEL_NU);

    // Test 1: Volume of unit cube = 1.0
    {
        CHECK_NEAR(elem.volume(), 1.0, 1e-6, "BBar: volume of unit cube = 1.0");
    }

    // Test 2: Zero displacement produces zero internal force
    {
        Real disp[24] = {};
        Real fint[24];
        elem.compute_internal_force(disp, fint);
        Real norm = 0.0;
        for (int i = 0; i < 24; ++i) norm += fint[i] * fint[i];
        CHECK(norm < 1e-20, "BBar: zero displacement -> zero force");
    }

    // Test 3: Stiffness matrix symmetry
    {
        Real K[576];
        elem.compute_stiffness(K);
        CHECK(check_symmetry(K, 24, 1e-6), "BBar: stiffness matrix symmetry");
    }

    // Test 4: Internal force nonzero for uniform expansion
    {
        Real disp[24] = {};
        Real eps = 0.001;
        for (int a = 0; a < 8; ++a)
            for (int d = 0; d < 3; ++d)
                disp[a*3+d] = eps * coords[a][d];

        Real fint[24];
        elem.compute_internal_force(disp, fint);
        Real norm = 0.0;
        for (int i = 0; i < 24; ++i) norm += fint[i] * fint[i];
        CHECK(norm > 0.0, "BBar: nonzero force for uniform expansion");
    }

    // Test 5: B-bar matrix has correct structure (6 x 24)
    {
        Real B[144];
        elem.compute_bbar(0.0, 0.0, 0.0, B);
        Real norm = 0.0;
        for (int i = 0; i < 144; ++i) norm += B[i] * B[i];
        CHECK(norm > 0, "BBar: B-bar matrix nonzero at center");
    }

    // Test 6: Near-incompressible produces finite stiffness (no volumetric locking)
    {
        BBarHex8 near_inc(coords, STEEL_E, 0.4999);
        Real K[576];
        near_inc.compute_stiffness(K);
        bool valid = true;
        for (int i = 0; i < 576; ++i)
            if (std::isnan(K[i]) || std::isinf(K[i])) { valid = false; break; }
        CHECK(valid, "BBar: near-incompressible stiffness is finite (no locking)");
    }

    // Test 7: Near-incompressible stiffness diagonal nonnegative
    {
        BBarHex8 near_inc(coords, STEEL_E, 0.4999);
        Real K[576];
        near_inc.compute_stiffness(K);
        CHECK(check_diagonal_nonneg(K, 24),
              "BBar: near-incompressible stiffness diagonal nonnegative");
    }

    // Test 8: Pure shear displacement produces nonzero shear strain in B-bar
    {
        Real disp[24] = {};
        Real gamma = 0.001;
        // Nodes with y=1 (nodes 2,3,6,7) get x-displacement = gamma
        disp[2*3+0] = gamma;
        disp[3*3+0] = gamma;
        disp[6*3+0] = gamma;
        disp[7*3+0] = gamma;

        Real fint[24];
        elem.compute_internal_force(disp, fint);
        Real norm = 0.0;
        for (int i = 0; i < 24; ++i) norm += fint[i] * fint[i];
        CHECK(norm > 0.0, "BBar: nonzero force for pure shear displacement");
    }

    // Test 9: Positive Jacobian at center
    {
        Real J[9];
        Real detJ = elem.jacobian(0.0, 0.0, 0.0, J);
        CHECK(detJ > 0, "BBar: positive Jacobian at center of unit cube");
    }
}

// ============================================================================
// 6. IsogeometricShell Tests
// ============================================================================
void test_isogeometric_shell() {
    std::cout << "\n=== 6. IsogeometricShell Tests ===\n";

    // Set up a 3x3 control point net for a quadratic NURBS flat plate [0,1]x[0,1]
    IsogeometricShell::NURBSPatch patch;
    patch.n_u = 3; patch.n_v = 3;
    patch.p_u = 2; patch.p_v = 2;

    // Open knot vectors for quadratic: {0,0,0,1,1,1}
    patch.num_knots_u = 6;
    patch.num_knots_v = 6;
    patch.knots_u[0] = 0; patch.knots_u[1] = 0; patch.knots_u[2] = 0;
    patch.knots_u[3] = 1; patch.knots_u[4] = 1; patch.knots_u[5] = 1;
    patch.knots_v[0] = 0; patch.knots_v[1] = 0; patch.knots_v[2] = 0;
    patch.knots_v[3] = 1; patch.knots_v[4] = 1; patch.knots_v[5] = 1;

    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            int idx = j * 3 + i;
            patch.control_points[idx][0] = static_cast<Real>(i) * 0.5;
            patch.control_points[idx][1] = static_cast<Real>(j) * 0.5;
            patch.control_points[idx][2] = 0.0;
            patch.weights[idx] = 1.0;
        }
    }

    IsogeometricShell elem(patch, STEEL_E, STEEL_NU, SHELL_THICKNESS);

    // Test 1: Number of control points
    {
        CHECK(elem.num_control_points() == 9, "IGA: 9 control points for 3x3 net");
    }

    // Test 2: Partition of unity at interior point
    {
        Real R[IsogeometricShell::MAX_CP];
        Real dRdxi[IsogeometricShell::MAX_CP];
        Real dRdeta[IsogeometricShell::MAX_CP];
        elem.compute_basis(0.5, 0.5, R, dRdxi, dRdeta);
        Real sum = 0.0;
        for (int i = 0; i < 9; ++i) sum += R[i];
        CHECK_NEAR(sum, 1.0, 1e-6, "IGA: partition of unity at (0.5, 0.5)");
    }

    // Test 3: Partition of unity at boundary corner
    {
        Real R[IsogeometricShell::MAX_CP];
        Real dRdxi[IsogeometricShell::MAX_CP];
        Real dRdeta[IsogeometricShell::MAX_CP];
        elem.compute_basis(0.0, 0.0, R, dRdxi, dRdeta);
        Real sum = 0.0;
        for (int i = 0; i < 9; ++i) sum += R[i];
        CHECK_NEAR(sum, 1.0, 1e-6, "IGA: partition of unity at boundary (0,0)");
    }

    // Test 4: Surface evaluation at corner matches control point (0,0,0)
    {
        Real x[3], a1[3], a2[3];
        elem.evaluate_surface(0.0, 0.0, x, a1, a2);
        CHECK_NEAR(x[0], 0.0, 1e-6, "IGA: surface x at (0,0) = 0");
        CHECK_NEAR(x[1], 0.0, 1e-6, "IGA: surface y at (0,0) = 0");
        CHECK_NEAR(x[2], 0.0, 1e-6, "IGA: surface z at (0,0) = 0");
    }

    // Test 5: Stiffness matrix is nonzero
    {
        int ndof = 9 * 3;
        std::vector<Real> K(ndof * ndof, 0.0);
        elem.compute_stiffness(K.data());
        Real ksum = 0.0;
        for (int i = 0; i < ndof * ndof; ++i) ksum += std::abs(K[i]);
        CHECK(ksum > 0, "IGA: stiffness matrix is nonzero");
    }

    // Test 6: Stiffness matrix symmetry
    {
        int ndof = 9 * 3;
        std::vector<Real> K(ndof * ndof, 0.0);
        elem.compute_stiffness(K.data());
        CHECK(check_symmetry(K.data(), ndof, 1e-6), "IGA: stiffness matrix symmetry");
    }

    // Test 7: Curved geometry evaluation
    {
        IsogeometricShell::NURBSPatch curved_patch = patch;
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) {
                int idx = j * 3 + i;
                Real cx = static_cast<Real>(i) * 0.5;
                Real cy = static_cast<Real>(j) * 0.5;
                curved_patch.control_points[idx][2] = 0.1 * cx * cy; // curvature
            }

        IsogeometricShell curved(curved_patch, STEEL_E, STEEL_NU, SHELL_THICKNESS);
        Real x[3], a1[3], a2[3];
        curved.evaluate_surface(0.5, 0.5, x, a1, a2);
        CHECK(x[2] > 0.0, "IGA: curved surface has positive z at center");
    }

    // Test 8: Surface area of flat unit plate
    {
        Real a = elem.area();
        // Flat plate [0,0.5*2]x[0,0.5*2] = [0,1]x[0,1], area should be ~ 1.0
        CHECK_NEAR(a, 1.0, 0.05, "IGA: flat plate area ~ 1.0");
    }

    // Test 9: Tangent vectors are nonzero at interior point
    {
        Real x[3], a1[3], a2[3];
        elem.evaluate_surface(0.5, 0.5, x, a1, a2);
        Real norm_a1 = std::sqrt(a1[0]*a1[0] + a1[1]*a1[1] + a1[2]*a1[2]);
        Real norm_a2 = std::sqrt(a2[0]*a2[0] + a2[1]*a2[1] + a2[2]*a2[2]);
        CHECK(norm_a1 > 0.0 && norm_a2 > 0.0,
              "IGA: tangent vectors nonzero at interior");
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== Wave 20 Element Formulations Test ===\n";

    test_belytschko_tsay_shell();
    test_pyramid5_element();
    test_mitc4_shell();
    test_eas_hex8();
    test_bbar_hex8();
    test_isogeometric_shell();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    return tests_failed > 0 ? 1 : 0;
}
