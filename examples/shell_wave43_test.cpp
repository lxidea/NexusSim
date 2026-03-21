/**
 * @file shell_wave43_test.cpp
 * @brief Wave 43: Warped shell element corrections — ~50 tests
 *
 * Covers:
 *  1. WarpDetector              (12 tests)
 *  2. WarpedShellCorrector      (12 tests)
 *  3. DrillingDOFStabilization  ( 8 tests)
 *  4. HourglassControl          (10 tests)
 *  5. ShellThicknessUpdate      ( 8 tests)
 */

#include <nexussim/discretization/shell_wave43.hpp>
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>

using namespace nxs;
using namespace nxs::fem;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; std::cout << "[FAIL] " << (msg) << "\n"; } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) do { \
    double _va = static_cast<double>(a); \
    double _vb = static_cast<double>(b); \
    double _vt = static_cast<double>(tol); \
    if (std::abs(_va - _vb) <= _vt) { tests_passed++; } \
    else { tests_failed++; std::cout << "[FAIL] " << (msg) \
        << " (got " << _va << ", expected " << _vb << ", tol " << _vt << ")\n"; } \
} while(0)

// ============================================================================
// Geometry helpers
// ============================================================================

/// Flat unit square in the XY plane: z = 0 for all nodes.
static void make_flat_quad(Real coords[4][3]) {
    coords[0][0] = 0; coords[0][1] = 0; coords[0][2] = 0;
    coords[1][0] = 1; coords[1][1] = 0; coords[1][2] = 0;
    coords[2][0] = 1; coords[2][1] = 1; coords[2][2] = 0;
    coords[3][0] = 0; coords[3][1] = 1; coords[3][2] = 0;
}

/// Warped quad: node 2 lifted by dz in the Z direction.
static void make_warped_quad(Real coords[4][3], Real dz) {
    make_flat_quad(coords);
    coords[2][2] = dz;
}

/// Rectangular quad with aspect ratio ar (width = ar, height = 1).
static void make_rect_quad(Real coords[4][3], Real ar) {
    coords[0][0] = 0;  coords[0][1] = 0; coords[0][2] = 0;
    coords[1][0] = ar; coords[1][1] = 0; coords[1][2] = 0;
    coords[2][0] = ar; coords[2][1] = 1; coords[2][2] = 0;
    coords[3][0] = 0;  coords[3][1] = 1; coords[3][2] = 0;
}

/// Skewed (rhombus) quad: node 1 and 2 shifted in X by skew.
static void make_skew_quad(Real coords[4][3], Real skew) {
    coords[0][0] = 0;    coords[0][1] = 0; coords[0][2] = 0;
    coords[1][0] = 1;    coords[1][1] = 0; coords[1][2] = 0;
    coords[2][0] = 1+skew; coords[2][1] = 1; coords[2][2] = 0;
    coords[3][0] = skew;   coords[3][1] = 1; coords[3][2] = 0;
}

// ============================================================================
// 1. WarpDetector tests
// ============================================================================

static void test_warp_detector() {
    std::cout << "\n--- WarpDetector ---\n";

    // --- Flat quad: warp angle should be exactly 0 ---
    {
        Real c[4][3];
        make_flat_quad(c);
        Real w = WarpDetector::compute_warp_angle(c);
        CHECK_NEAR(w, 0.0, 1.0e-10, "Flat quad warp angle == 0");
        CHECK(!WarpDetector::is_severely_warped(w), "Flat quad not severely warped");
    }

    // --- Warped quad: node 2 raised by 0.5 (45° warp expected > 10°) ---
    {
        Real c[4][3];
        make_warped_quad(c, 0.5);
        Real w = WarpDetector::compute_warp_angle(c);
        CHECK(w > 0.0, "Warped quad warp angle > 0");
        CHECK(WarpDetector::is_severely_warped(w), "Warped quad is severely warped");
        // For a 1×1 quad with node 2 at z=0.5, the two triangle normals differ noticeably
        CHECK(w > 0.1, "Warped quad warp angle > 0.1 rad");
    }

    // --- Flat quad: aspect ratio == 1 ---
    {
        Real c[4][3];
        make_flat_quad(c);
        Real ar = WarpDetector::compute_aspect_ratio(c);
        CHECK_NEAR(ar, 1.0, 1.0e-10, "Unit square aspect ratio == 1");
    }

    // --- Rectangular quad: aspect ratio matches construction ---
    {
        Real c[4][3];
        make_rect_quad(c, 4.0);
        Real ar = WarpDetector::compute_aspect_ratio(c);
        CHECK_NEAR(ar, 4.0, 1.0e-10, "4:1 rectangle aspect ratio == 4");
    }

    // --- Flat square: skew angle == 0 (diagonals are perpendicular) ---
    {
        Real c[4][3];
        make_flat_quad(c);
        Real sk = WarpDetector::compute_skew_angle(c);
        CHECK_NEAR(sk, 0.0, 1.0e-10, "Unit square skew angle == 0");
    }

    // --- Parallelogram: skew angle > 0 ---
    {
        Real c[4][3];
        make_skew_quad(c, 0.5);
        Real sk = WarpDetector::compute_skew_angle(c);
        CHECK(sk > 0.0, "Skewed quad skew angle > 0");
    }

    // --- is_severely_warped threshold ---
    {
        // 9° = 0.1571 rad < threshold 0.1745 rad → NOT severely warped
        CHECK(!WarpDetector::is_severely_warped(0.1571), "9° not severely warped");
        // 11° = 0.1920 rad > threshold → severely warped
        CHECK(WarpDetector::is_severely_warped(0.1920), "11° is severely warped");
        // Custom threshold
        CHECK(WarpDetector::is_severely_warped(0.05, 0.04), "Custom threshold test");
        CHECK(!WarpDetector::is_severely_warped(0.03, 0.04), "Custom threshold not warped");
    }

    // --- Warp angle is non-negative ---
    {
        Real c[4][3];
        make_warped_quad(c, -0.3);  // negative z — angle still non-negative
        Real w = WarpDetector::compute_warp_angle(c);
        CHECK(w >= 0.0, "Warp angle is non-negative for downward warp");
    }

    // --- Warp angle in [0, π] ---
    {
        Real c[4][3];
        make_warped_quad(c, 2.0);  // large warp
        Real w = WarpDetector::compute_warp_angle(c);
        CHECK(w >= 0.0 && w <= M_PI, "Warp angle in [0, pi]");
    }
}

// ============================================================================
// 2. WarpedShellCorrector tests
// ============================================================================

static void test_warped_shell_corrector() {
    std::cout << "\n--- WarpedShellCorrector ---\n";

    // --- Local frame orthogonality on flat quad ---
    {
        Real c[4][3];
        make_flat_quad(c);
        Real e1[3], e2[3], e3[3];
        WarpedShellCorrector::compute_local_frame(c, e1, e2, e3);

        auto dot3 = [](const Real* a, const Real* b) {
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
        };
        CHECK_NEAR(dot3(e1, e2), 0.0, 1.0e-10, "e1 ⊥ e2 on flat quad");
        CHECK_NEAR(dot3(e1, e3), 0.0, 1.0e-10, "e1 ⊥ e3 on flat quad");
        CHECK_NEAR(dot3(e2, e3), 0.0, 1.0e-10, "e2 ⊥ e3 on flat quad");
    }

    // --- Local frame unit vectors on flat quad ---
    {
        Real c[4][3];
        make_flat_quad(c);
        Real e1[3], e2[3], e3[3];
        WarpedShellCorrector::compute_local_frame(c, e1, e2, e3);

        auto norm3 = [](const Real* v) {
            return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        };
        CHECK_NEAR(norm3(e1), 1.0, 1.0e-10, "|e1| == 1 on flat quad");
        CHECK_NEAR(norm3(e2), 1.0, 1.0e-10, "|e2| == 1 on flat quad");
        CHECK_NEAR(norm3(e3), 1.0, 1.0e-10, "|e3| == 1 on flat quad");
    }

    // --- Local frame orthogonality on warped quad ---
    {
        Real c[4][3];
        make_warped_quad(c, 0.5);
        Real e1[3], e2[3], e3[3];
        WarpedShellCorrector::compute_local_frame(c, e1, e2, e3);

        auto dot3 = [](const Real* a, const Real* b) {
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
        };
        CHECK_NEAR(dot3(e1, e2), 0.0, 1.0e-10, "e1 ⊥ e2 on warped quad");
        CHECK_NEAR(dot3(e1, e3), 0.0, 1.0e-10, "e1 ⊥ e3 on warped quad");
        CHECK_NEAR(dot3(e2, e3), 0.0, 1.0e-10, "e2 ⊥ e3 on warped quad");
    }

    // --- Project to local: node 0 maps to (0, 0) ---
    {
        Real c[4][3];
        make_flat_quad(c);
        Real e1[3] = {1,0,0}, e2[3] = {0,1,0}, e3[3] = {0,0,1};
        Real lc[4][2];
        WarpedShellCorrector::project_to_local(c, lc, e1, e2, e3);
        CHECK_NEAR(lc[0][0], 0.0, 1.0e-10, "Node 0 local x == 0");
        CHECK_NEAR(lc[0][1], 0.0, 1.0e-10, "Node 0 local y == 0");
    }

    // --- Project to local: node 1 maps to (1, 0) in xy-plane ---
    {
        Real c[4][3];
        make_flat_quad(c);
        Real e1[3] = {1,0,0}, e2[3] = {0,1,0}, e3[3] = {0,0,1};
        Real lc[4][2];
        WarpedShellCorrector::project_to_local(c, lc, e1, e2, e3);
        CHECK_NEAR(lc[1][0], 1.0, 1.0e-10, "Node 1 local x == 1");
        CHECK_NEAR(lc[1][1], 0.0, 1.0e-10, "Node 1 local y == 0");
    }

    // --- Warp correction: flat element → K unchanged ---
    {
        Real K_flat[24*24] = {};
        // Fill diagonal with 1.0
        for (int i = 0; i < 24; ++i) K_flat[i*24 + i] = 1.0;

        Real K_corr[24*24];
        WarpedShellCorrector::compute_warp_correction_matrix(0.0, K_flat, K_corr);

        // Off-diagonal membrane-bending coupling should remain zero
        bool all_diag_equal = true;
        for (int i = 0; i < 24; ++i)
            if (std::abs(K_corr[i*24 + i] - 1.0) > 1.0e-14) all_diag_equal = false;
        CHECK(all_diag_equal, "Flat element: diagonal unchanged after zero-warp correction");
    }

    // --- Warp correction: warped element → coupling terms added ---
    {
        Real K_flat[24*24] = {};
        for (int i = 0; i < 24; ++i) K_flat[i*24 + i] = 1.0e6;

        Real K_corr[24*24];
        Real warp_angle = 0.3;  // ~17°
        WarpedShellCorrector::compute_warp_correction_matrix(warp_angle, K_flat, K_corr);

        // Node 0: mem DOF 0 (row) coupled to bending DOF 2 (col)
        Real coupling = K_corr[0*24 + 2];
        CHECK(coupling > 0.0, "Warped element: membrane-bending coupling added (>0)");
    }

    // --- Warp correction: symmetry preserved ---
    {
        Real K_flat[24*24] = {};
        for (int i = 0; i < 24; ++i) K_flat[i*24 + i] = 2.0e5;

        Real K_corr[24*24];
        WarpedShellCorrector::compute_warp_correction_matrix(0.25, K_flat, K_corr);

        bool symmetric = true;
        for (int i = 0; i < 24 && symmetric; ++i)
            for (int j = 0; j < 24 && symmetric; ++j)
                if (std::abs(K_corr[i*24+j] - K_corr[j*24+i]) > 1.0e-10)
                    symmetric = false;
        CHECK(symmetric, "Warp-corrected K is symmetric");
    }

    // --- Warp correction: diagonal remains dominant (coupling << diagonal) ---
    {
        Real K_flat[24*24] = {};
        for (int i = 0; i < 24; ++i) K_flat[i*24 + i] = 1.0e8;

        Real K_corr[24*24];
        WarpedShellCorrector::compute_warp_correction_matrix(0.1, K_flat, K_corr);

        bool diag_dominates = true;
        for (int i = 0; i < 24 && diag_dominates; ++i) {
            Real sum_off = 0.0;
            for (int j = 0; j < 24; ++j)
                if (j != i) sum_off += std::abs(K_corr[i*24+j]);
            if (sum_off > std::abs(K_corr[i*24+i]))
                diag_dominates = false;
        }
        CHECK(diag_dominates, "Warp correction: diagonal dominance maintained");
    }
}

// ============================================================================
// 3. DrillingDOFStabilization tests
// ============================================================================

static void test_drilling_dof() {
    std::cout << "\n--- DrillingDOFStabilization ---\n";

    const Real E = 210.0e9;
    const Real t = 0.01;
    const Real A = 1.0;  // 1 m² element area

    // --- Drilling stiffness is positive ---
    {
        Real kd = DrillingDOFStabilization::compute_drilling_stiffness(E, t, A);
        CHECK(kd > 0.0, "Drilling stiffness is positive");
    }

    // --- Drilling stiffness formula: k = alpha * E * t * A ---
    {
        Real kd = DrillingDOFStabilization::compute_drilling_stiffness(E, t, A);
        Real expected = DrillingDOFStabilization::ALPHA_DRILL * E * t * A;
        CHECK_NEAR(kd, expected, 1.0, "Drilling stiffness matches formula");
    }

    // --- Drilling stiffness scales linearly with E ---
    {
        Real kd1 = DrillingDOFStabilization::compute_drilling_stiffness(E, t, A);
        Real kd2 = DrillingDOFStabilization::compute_drilling_stiffness(2*E, t, A);
        CHECK_NEAR(kd2 / kd1, 2.0, 1.0e-10, "Drilling stiffness scales with E");
    }

    // --- Drilling stiffness scales linearly with area ---
    {
        Real kd1 = DrillingDOFStabilization::compute_drilling_stiffness(E, t, 1.0);
        Real kd2 = DrillingDOFStabilization::compute_drilling_stiffness(E, t, 3.0);
        CHECK_NEAR(kd2 / kd1, 3.0, 1.0e-10, "Drilling stiffness scales with area");
    }

    // --- add_drilling_to_stiffness: only θz DOFs (DOF 5 per node) are modified ---
    {
        Real K[24*24] = {};
        Real kd = 1000.0;
        DrillingDOFStabilization::add_drilling_to_stiffness(K, kd, 4);

        // DOF 5 (θz of node 0), DOF 11 (node 1), DOF 17 (node 2), DOF 23 (node 3)
        CHECK_NEAR(K[5*24 + 5],   kd, 1.0e-10, "θz of node 0 has k_drill");
        CHECK_NEAR(K[11*24 + 11], kd, 1.0e-10, "θz of node 1 has k_drill");
        CHECK_NEAR(K[17*24 + 17], kd, 1.0e-10, "θz of node 2 has k_drill");
        CHECK_NEAR(K[23*24 + 23], kd, 1.0e-10, "θz of node 3 has k_drill");
    }

    // --- Non-θz diagonal entries remain zero ---
    {
        Real K[24*24] = {};
        DrillingDOFStabilization::add_drilling_to_stiffness(K, 500.0, 4);
        // DOF 0 (ux node 0) should remain 0
        CHECK_NEAR(K[0*24 + 0], 0.0, 1.0e-10, "ux diagonal not modified by drilling");
        CHECK_NEAR(K[2*24 + 2], 0.0, 1.0e-10, "uz diagonal not modified by drilling");
    }

    // --- add_drilling accumulates (call twice) ---
    {
        Real K[24*24] = {};
        DrillingDOFStabilization::add_drilling_to_stiffness(K, 300.0, 4);
        DrillingDOFStabilization::add_drilling_to_stiffness(K, 200.0, 4);
        CHECK_NEAR(K[5*24 + 5], 500.0, 1.0e-10, "Drilling stiffness accumulates correctly");
    }
}

// ============================================================================
// 4. HourglassControl tests
// ============================================================================

static void test_hourglass_control() {
    std::cout << "\n--- HourglassControl ---\n";

    Real coords[4][3];
    make_flat_quad(coords);

    HourglassParams params;
    params.coefficient    = 0.05;
    params.viscous_coeff  = 0.05;
    params.stiffness_coeff= 0.1;
    params.E              = 210.0e9;
    params.thickness      = 0.01;

    // --- Zero velocity → zero hourglass forces (all modes) ---
    {
        Real vel[4][6] = {};
        Real F[24];

        HourglassControl::compute_hourglass_forces(
            HourglassType::FlanaganBelytschko, coords, vel, params, F);
        bool all_zero = true;
        for (int i = 0; i < 24; ++i) if (std::abs(F[i]) > 1.0e-20) all_zero = false;
        CHECK(all_zero, "F-B: zero velocity → zero forces");

        HourglassControl::compute_hourglass_forces(
            HourglassType::Viscous, coords, vel, params, F);
        all_zero = true;
        for (int i = 0; i < 24; ++i) if (std::abs(F[i]) > 1.0e-20) all_zero = false;
        CHECK(all_zero, "Viscous: zero velocity → zero forces");
    }

    // --- Rigid body motion (uniform velocity) → zero hourglass forces ---
    // For uniform vx=1, Γ·v = 1-1+1-1 = 0 → no hourglass participation
    {
        Real vel[4][6] = {};
        for (int n = 0; n < 4; ++n) vel[n][0] = 1.0;  // uniform vx
        Real F[24];

        HourglassControl::compute_hourglass_forces(
            HourglassType::FlanaganBelytschko, coords, vel, params, F);
        bool all_zero = true;
        for (int i = 0; i < 6; i += 1) if (std::abs(F[i]) > 1.0e-20) all_zero = false;
        CHECK(all_zero, "F-B: rigid body motion → no hourglass in x");
    }

    // --- Hourglass mode velocity activates F-B forces ---
    // Hourglass velocity pattern: v_hg = [1, -1, 1, -1] (matches Γ)
    {
        Real vel[4][6] = {};
        const Real gamma[4] = {1.0, -1.0, 1.0, -1.0};
        for (int n = 0; n < 4; ++n) vel[n][0] = gamma[n];  // hourglass vx pattern
        Real F[24];

        HourglassControl::compute_hourglass_forces(
            HourglassType::FlanaganBelytschko, coords, vel, params, F);
        // q = 1+1+1+1 = 4, F_hg = coeff * q * gamma_n → forces non-zero and with pattern
        CHECK(std::abs(F[0]) > 1.0e-10, "F-B: hourglass pattern activates force node 0");
        // Node 0 and node 1 forces should have opposite signs (gamma[0]=+1, gamma[1]=-1)
        CHECK(F[0] * F[6] < 0.0, "F-B: hourglass forces alternate signs (nodes 0 and 1)");
    }

    // --- Physical mode: forces are non-zero for hourglass velocity ---
    {
        Real vel[4][6] = {};
        const Real gamma[4] = {1.0, -1.0, 1.0, -1.0};
        for (int n = 0; n < 4; ++n) vel[n][0] = gamma[n];
        Real F[24];

        HourglassControl::compute_hourglass_forces(
            HourglassType::Physical, coords, vel, params, F);
        CHECK(std::abs(F[0]) > 1.0e-10, "Physical: hourglass pattern activates force");
    }

    // --- Viscous mode: forces are non-zero for hourglass velocity ---
    {
        Real vel[4][6] = {};
        const Real gamma[4] = {1.0, -1.0, 1.0, -1.0};
        for (int n = 0; n < 4; ++n) vel[n][0] = gamma[n];
        Real F[24];

        HourglassControl::compute_hourglass_forces(
            HourglassType::Viscous, coords, vel, params, F);
        CHECK(std::abs(F[0]) > 1.0e-10, "Viscous: hourglass pattern activates force");
    }

    // --- Stiffness mode: forces are non-zero for hourglass displacement ---
    {
        Real vel[4][6] = {};
        const Real gamma[4] = {1.0, -1.0, 1.0, -1.0};
        for (int n = 0; n < 4; ++n) vel[n][0] = gamma[n];
        Real F[24];

        HourglassControl::compute_hourglass_forces(
            HourglassType::Stiffness, coords, vel, params, F);
        CHECK(std::abs(F[0]) > 1.0e-10, "Stiffness: hourglass pattern activates force");
    }

    // --- F-B forces scale linearly with coefficient ---
    {
        const Real gamma[4] = {1.0, -1.0, 1.0, -1.0};
        Real vel[4][6] = {};
        for (int n = 0; n < 4; ++n) vel[n][1] = gamma[n];

        HourglassParams p1 = params; p1.coefficient = 0.02;
        HourglassParams p2 = params; p2.coefficient = 0.04;
        Real F1[24], F2[24];
        HourglassControl::compute_hourglass_forces(
            HourglassType::FlanaganBelytschko, coords, vel, p1, F1);
        HourglassControl::compute_hourglass_forces(
            HourglassType::FlanaganBelytschko, coords, vel, p2, F2);

        // ratio of forces should be 2.0
        if (std::abs(F1[1]) > 1.0e-20)
            CHECK_NEAR(F2[1] / F1[1], 2.0, 1.0e-10,
                       "F-B forces scale linearly with coefficient");
        else
            CHECK(false, "F-B forces non-zero for y hourglass");
    }
}

// ============================================================================
// 5. ShellThicknessUpdate tests
// ============================================================================

static void test_thickness_update() {
    std::cout << "\n--- ShellThicknessUpdate ---\n";

    // --- Thickness unchanged when F_33 = 1 ---
    {
        Real t0 = 0.005;
        Real t = ShellThicknessUpdate::update_thickness(t0, 1.0);
        CHECK_NEAR(t, t0, 1.0e-14, "update_thickness: F33=1 → t unchanged");
    }

    // --- Thickness compressed: F_33 < 1 ---
    {
        Real t0 = 0.01;
        Real t = ShellThicknessUpdate::update_thickness(t0, 0.8);
        CHECK_NEAR(t, 0.008, 1.0e-14, "update_thickness: F33=0.8 → t reduced");
    }

    // --- Thickness stretched: F_33 > 1 ---
    {
        Real t0 = 0.01;
        Real t = ShellThicknessUpdate::update_thickness(t0, 1.5);
        CHECK_NEAR(t, 0.015, 1.0e-14, "update_thickness: F33=1.5 → t increased");
    }

    // --- compute_thickness_stretch: incompressible, equal areas → F33 = 1 ---
    {
        Real F33 = ShellThicknessUpdate::compute_thickness_stretch(1.0, 1.0, true);
        CHECK_NEAR(F33, 1.0, 1.0e-14,
                   "compute_thickness_stretch: equal areas → F33 = 1");
    }

    // --- compute_thickness_stretch: area doubles → F33 = 0.5 ---
    {
        Real F33 = ShellThicknessUpdate::compute_thickness_stretch(1.0, 2.0, true);
        CHECK_NEAR(F33, 0.5, 1.0e-14,
                   "compute_thickness_stretch: area doubles → F33 = 0.5");
    }

    // --- compute_thickness_stretch: area halves → F33 = 2.0 ---
    {
        Real F33 = ShellThicknessUpdate::compute_thickness_stretch(1.0, 0.5, true);
        CHECK_NEAR(F33, 2.0, 1.0e-14,
                   "compute_thickness_stretch: area halves → F33 = 2");
    }

    // --- compute_thickness_stretch: non-incompressible → F33 = 1 ---
    {
        Real F33 = ShellThicknessUpdate::compute_thickness_stretch(1.0, 2.0, false);
        CHECK_NEAR(F33, 1.0, 1.0e-14,
                   "compute_thickness_stretch: non-incompressible → F33 = 1");
    }

    // --- Round-trip: update then inverse area ratio → original thickness ---
    {
        Real t0 = 0.008;
        Real A0 = 4.0, A_cur = 5.0;
        Real F33 = ShellThicknessUpdate::compute_thickness_stretch(A0, A_cur, true);
        Real t_new = ShellThicknessUpdate::update_thickness(t0, F33);
        // t_new * A_cur should equal t0 * A0 (volume conservation)
        CHECK_NEAR(t_new * A_cur, t0 * A0, 1.0e-12,
                   "Round-trip: t * A conserved under incompressible stretch");
    }
}

// ============================================================================
// main
// ============================================================================

int main() {
    std::cout << "=== Wave 43: Warped Shell Element Corrections ===\n";

    test_warp_detector();
    test_warped_shell_corrector();
    test_drilling_dof();
    test_hourglass_control();
    test_thickness_update();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return (tests_failed > 0) ? 1 : 0;
}
