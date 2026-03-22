/**
 * @file euler_wave34_test.cpp
 * @brief Wave 34a: Pure Eulerian Solver Test Suite
 *
 * Tests 6 sub-modules (~12 tests each, 70 total):
 *  1. EulerianFlux         — HLLC/Roe flux conservation, symmetry, shock capture
 *  2. EulerianGradient     — MinMod limiter, gradient reconstruction
 *  3. EulerianTimeStepping — CFL dt computation, stability
 *  4. EulerianBCs          — Reflecting, transmitting, inflow, outflow
 *  5. Euler2DSolver        — 2D Godunov step, Sod problem, mass conservation
 *  6. Euler3DSolver        — 3D Godunov step, symmetry, energy conservation
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <numeric>
#include <nexussim/fem/euler_wave34.hpp>

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

using namespace nxs::fem;
using Real = nxs::Real;

// ============================================================================
// Helper: create EulerCell from primitives
// ============================================================================
static EulerCell make_cell(Real rho, Real u, Real v, Real w, Real p, Real gamma = 1.4) {
    EulerCell c;
    c.rho = rho; c.u = u; c.v = v; c.w = w; c.p = p; c.gamma = gamma;
    c.dx = 1.0; c.dy = 1.0; c.dz = 1.0;
    c.compute_total_energy();
    return c;
}

// ============================================================================
// Test Group 1: EulerianFlux (12 tests)
// ============================================================================
void test_eulerian_flux() {
    std::cout << "=== EulerianFlux Tests ===\n";

    // Test 1: HLLC flux for identical states => physical flux
    {
        EulerCell c = make_cell(1.0, 1.0, 0.0, 0.0, 1.0);
        Real UL[5], UR[5], flux[5];
        c.to_conservative(UL);
        c.to_conservative(UR);
        EulerianFlux::hllc_flux(UL, UR, c.p, c.p, c.gamma, c.gamma, 0, flux);
        // Physical flux in x: [rho*u, rho*u^2+p, rho*u*v, rho*u*w, (E+p)*u]
        Real vel[3] = {1.0, 0.0, 0.0};
        Real f_exact[5];
        EulerianFlux::euler_physical_flux(UL, c.p, vel, 0, f_exact);
        Real max_err = 0.0;
        for (int i = 0; i < 5; ++i) {
            Real err = std::abs(flux[i] - f_exact[i]);
            if (err > max_err) max_err = err;
        }
        CHECK(max_err < 1.0e-12, "HLLC: identical states => physical flux");
    }

    // Test 2: HLLC flux conservation (flux_L + flux_R = 0 for symmetric states)
    {
        Real UL[5] = {1.0, 1.0, 0.0, 0.0, 2.5};  // rho=1, u=1, p=1, gamma=1.4
        Real UR[5] = {1.0, -1.0, 0.0, 0.0, 2.5}; // mirror
        Real fluxLR[5], fluxRL[5];
        EulerianFlux::hllc_flux(UL, UR, 1.0, 1.0, 1.4, 1.4, 0, fluxLR);
        EulerianFlux::hllc_flux(UR, UL, 1.0, 1.0, 1.4, 1.4, 0, fluxRL);
        // Mass flux should cancel
        CHECK_NEAR(fluxLR[0] + fluxRL[0], 0.0, 1.0e-10, "HLLC: symmetric mass flux cancels");
    }

    // Test 3: HLLC flux for equal states (all components match physical flux)
    {
        EulerCell c = make_cell(1.225, 0.0, 0.0, 0.0, 101325.0);
        Real U[5];
        c.to_conservative(U);
        Real flux[5];
        EulerianFlux::hllc_flux(U, U, c.p, c.p, c.gamma, c.gamma, 0, flux);
        // Zero velocity => mass flux = 0, momentum flux = p, energy flux = 0
        CHECK_NEAR(flux[0], 0.0, 1.0e-6, "HLLC: zero-vel mass flux = 0");
        CHECK_NEAR(flux[1], 101325.0, 1.0, "HLLC: zero-vel momentum flux = p");
        CHECK_NEAR(flux[4], 0.0, 1.0e-6, "HLLC: zero-vel energy flux = 0");
    }

    // Test 4: Roe flux for identical states
    {
        EulerCell c = make_cell(1.0, 2.0, 0.0, 0.0, 1.0);
        Real U[5];
        c.to_conservative(U);
        Real flux[5];
        EulerianFlux::roe_flux(U, U, c.p, c.p, c.gamma, 0, flux);
        Real vel[3] = {2.0, 0.0, 0.0};
        Real f_exact[5];
        EulerianFlux::euler_physical_flux(U, c.p, vel, 0, f_exact);
        Real max_err = 0.0;
        for (int i = 0; i < 5; ++i) {
            Real err = std::abs(flux[i] - f_exact[i]);
            if (err > max_err) max_err = err;
        }
        CHECK(max_err < 1.0e-10, "Roe: identical states => physical flux");
    }

    // Test 5: Roe flux symmetry — F(UL,UR) mass + F(UR,UL) mass for symmetric states
    {
        Real UL[5] = {1.0, 0.5, 0.0, 0.0, 2.625};
        Real UR[5] = {1.0, -0.5, 0.0, 0.0, 2.625};
        Real fluxLR[5], fluxRL[5];
        EulerianFlux::roe_flux(UL, UR, 1.0, 1.0, 1.4, 0, fluxLR);
        EulerianFlux::roe_flux(UR, UL, 1.0, 1.0, 1.4, 0, fluxRL);
        CHECK_NEAR(fluxLR[0] + fluxRL[0], 0.0, 1.0e-10, "Roe: symmetric mass flux cancels");
    }

    // Test 6: HLLC handles vacuum (zero density)
    {
        Real UL[5] = {1.0, 0.0, 0.0, 0.0, 2.5};
        Real UR[5] = {1.0e-35, 0.0, 0.0, 0.0, 1.0e-35};
        Real flux[5];
        EulerianFlux::hllc_flux(UL, UR, 1.0, 0.0, 1.4, 1.4, 0, flux);
        // Should not crash
        bool finite = true;
        for (int i = 0; i < 5; ++i) {
            if (std::isnan(flux[i]) || std::isinf(flux[i])) finite = false;
        }
        CHECK(finite, "HLLC: handles near-vacuum without NaN");
    }

    // Test 7: Physical flux in y-direction
    {
        EulerCell c = make_cell(1.0, 0.0, 3.0, 0.0, 1.0);
        Real U[5];
        c.to_conservative(U);
        Real vel[3] = {0.0, 3.0, 0.0};
        Real flux[5];
        EulerianFlux::euler_physical_flux(U, c.p, vel, 1, flux);
        CHECK_NEAR(flux[0], 3.0, 1.0e-12, "Physical flux y: mass = rho*v");
        CHECK_NEAR(flux[2], 9.0 + 1.0, 1.0e-12, "Physical flux y: mom_y = rho*v^2 + p");
    }

    // Test 8: HLLC in y-direction
    {
        EulerCell cL = make_cell(1.0, 0.0, 1.0, 0.0, 1.0);
        EulerCell cR = make_cell(1.0, 0.0, 1.0, 0.0, 1.0);
        Real UL[5], UR[5], flux[5];
        cL.to_conservative(UL);
        cR.to_conservative(UR);
        EulerianFlux::hllc_flux(UL, UR, cL.p, cR.p, cL.gamma, cR.gamma, 1, flux);
        CHECK_NEAR(flux[0], 1.0, 1.0e-10, "HLLC y-dir: mass flux = rho*v");
    }

    // Test 9: Sod shock tube left/right states produce reasonable HLLC flux
    {
        // Left: rho=1, u=0, p=1 => E = p/(gamma-1) = 2.5
        // Right: rho=0.125, u=0, p=0.1 => E = 0.25
        Real UL[5] = {1.0, 0.0, 0.0, 0.0, 2.5};
        Real UR[5] = {0.125, 0.0, 0.0, 0.0, 0.25};
        Real flux[5];
        EulerianFlux::hllc_flux(UL, UR, 1.0, 0.1, 1.4, 1.4, 0, flux);
        // Pressure imbalance drives flow right => positive mass flux
        CHECK(flux[0] > 0.0, "Sod HLLC: positive mass flux (left-to-right)");
        CHECK(flux[1] > 0.0, "Sod HLLC: positive momentum flux");
    }

    // Test 10: HLLC flux with different gammas
    {
        Real UL[5] = {1.0, 0.0, 0.0, 0.0, 2.5};   // gamma = 1.4
        Real UR[5] = {1.0, 0.0, 0.0, 0.0, 1.667};  // gamma = 1.6
        Real flux[5];
        EulerianFlux::hllc_flux(UL, UR, 1.0, 1.0, 1.4, 1.6, 0, flux);
        bool finite = true;
        for (int i = 0; i < 5; ++i) {
            if (std::isnan(flux[i]) || std::isinf(flux[i])) finite = false;
        }
        CHECK(finite, "HLLC: different gammas produce finite flux");
    }

    // Test 11: HLLC flux in z-direction
    {
        EulerCell c = make_cell(2.0, 0.0, 0.0, 5.0, 10.0);
        Real U[5];
        c.to_conservative(U);
        Real flux[5];
        EulerianFlux::hllc_flux(U, U, c.p, c.p, c.gamma, c.gamma, 2, flux);
        CHECK_NEAR(flux[0], 2.0 * 5.0, 1.0e-10, "HLLC z-dir: mass flux = rho*w");
    }

    // Test 12: Roe flux for supersonic flow (all eigenvalues same sign)
    {
        // u = 500 m/s, c = sqrt(1.4*101325/1.225) ~ 340 m/s => Mach ~ 1.47
        EulerCell c = make_cell(1.225, 500.0, 0.0, 0.0, 101325.0);
        Real U[5];
        c.to_conservative(U);
        Real flux[5];
        EulerianFlux::roe_flux(U, U, c.p, c.p, c.gamma, 0, flux);
        CHECK(flux[0] > 0.0, "Roe: supersonic => positive mass flux");
    }
}

// ============================================================================
// Test Group 2: EulerianGradient (12 tests)
// ============================================================================
void test_eulerian_gradient() {
    std::cout << "=== EulerianGradient Tests ===\n";

    // Test 1: MinMod with same-sign slopes
    {
        Real r = EulerianGradient::minmod(2.0, 5.0);
        CHECK_NEAR(r, 2.0, 1.0e-15, "MinMod: same sign => min magnitude");
    }

    // Test 2: MinMod with opposite-sign slopes
    {
        Real r = EulerianGradient::minmod(2.0, -3.0);
        CHECK_NEAR(r, 0.0, 1.0e-15, "MinMod: opposite sign => 0");
    }

    // Test 3: MinMod with zero
    {
        Real r = EulerianGradient::minmod(0.0, 5.0);
        CHECK_NEAR(r, 0.0, 1.0e-15, "MinMod: zero slope => 0");
    }

    // Test 4: 1D gradient of linear function
    {
        Real grad = EulerianGradient::compute_1d_gradient(2.0, 1.0, 3.0, 1.0);
        CHECK_NEAR(grad, 1.0, 1.0e-15, "Gradient 1D: linear function => exact");
    }

    // Test 5: 1D gradient with discontinuity (limiter activates)
    {
        // Left=1, center=2, right=10 => slopes 1 and 8 => minmod picks 1
        Real grad = EulerianGradient::compute_1d_gradient(2.0, 1.0, 10.0, 1.0);
        CHECK_NEAR(grad, 1.0, 1.0e-15, "Gradient 1D: discontinuity => limited slope");
    }

    // Test 6: 1D gradient with sign change
    {
        // Left=3, center=1, right=5 => slopes -2 and 4 => minmod = 0 (extremum)
        Real grad = EulerianGradient::compute_1d_gradient(1.0, 3.0, 5.0, 1.0);
        CHECK_NEAR(grad, 0.0, 1.0e-15, "Gradient 1D: local extremum => 0");
    }

    // Test 7: 3D gradient computation
    {
        // Linear field: q(x,y,z) = 2x + 3y + z
        // Cell at (1,1,1), neighbors at (0,1,1), (2,1,1), etc.
        // q values: center=6, xm=4, xp=8, ym=3, yp=9, zm=5, zp=7
        std::vector<Real> vals = {4.0, 8.0, 3.0, 9.0, 5.0, 7.0, 6.0};
        // Cell 6 is center, neighbors are 0,1,2,3,4,5
        int neighbors[6] = {0, 1, 2, 3, 4, 5};
        Real gradient[3];
        EulerianGradient::compute_gradient(vals.data(), 6, neighbors, 1.0, 1.0, 1.0, 7, gradient);
        CHECK_NEAR(gradient[0], 2.0, 1.0e-12, "Gradient 3D: dq/dx = 2");
        CHECK_NEAR(gradient[1], 3.0, 1.0e-12, "Gradient 3D: dq/dy = 3");
        CHECK_NEAR(gradient[2], 1.0, 1.0e-12, "Gradient 3D: dq/dz = 1");
    }

    // Test 8: Face reconstruction
    {
        Real vl, vr;
        EulerianGradient::reconstruct_face(5.0, 2.0, 1.0, vl, vr);
        CHECK_NEAR(vl, 4.0, 1.0e-15, "Face reconstruct: left = center - 0.5*grad*dx");
        CHECK_NEAR(vr, 6.0, 1.0e-15, "Face reconstruct: right = center + 0.5*grad*dx");
    }

    // Test 9: Gradient with boundary neighbors (= -1, use center value)
    {
        std::vector<Real> vals = {5.0};
        int neighbors[6] = {-1, -1, -1, -1, -1, -1};
        Real gradient[3];
        EulerianGradient::compute_gradient(vals.data(), 0, neighbors, 1.0, 1.0, 1.0, 1, gradient);
        CHECK_NEAR(gradient[0], 0.0, 1.0e-15, "Gradient boundary: all boundary => 0 gradient");
        CHECK_NEAR(gradient[1], 0.0, 1.0e-15, "Gradient boundary: y = 0");
        CHECK_NEAR(gradient[2], 0.0, 1.0e-15, "Gradient boundary: z = 0");
    }

    // Test 10: MinMod negative slopes
    {
        Real r = EulerianGradient::minmod(-3.0, -7.0);
        CHECK_NEAR(r, -3.0, 1.0e-15, "MinMod: both negative => -3");
    }
}

// ============================================================================
// Test Group 3: EulerianTimeStepping (12 tests)
// ============================================================================
void test_eulerian_timestepping() {
    std::cout << "=== EulerianTimeStepping Tests ===\n";

    // Test 1: CFL dt for uniform grid at rest
    {
        EulerCell cells[4];
        for (int i = 0; i < 4; ++i) {
            cells[i] = make_cell(1.0, 0.0, 0.0, 0.0, 1.0);
            cells[i].dx = 0.1; cells[i].dy = 0.1; cells[i].dz = 0.1;
        }
        Real c = std::sqrt(1.4 * 1.0 / 1.0);  // ~1.183
        Real dt = EulerianTimeStepping::compute_dt(cells, 4, 0.5);
        Real expected = 0.5 * 0.1 / c;
        CHECK_NEAR(dt, expected, 1.0e-10, "CFL dt: uniform at rest");
    }

    // Test 2: CFL dt decreases with velocity
    {
        EulerCell cells1[1], cells2[1];
        cells1[0] = make_cell(1.0, 0.0, 0.0, 0.0, 1.0);
        cells1[0].dx = 0.1; cells1[0].dy = 0.1; cells1[0].dz = 0.1;
        cells2[0] = make_cell(1.0, 10.0, 0.0, 0.0, 1.0);
        cells2[0].dx = 0.1; cells2[0].dy = 0.1; cells2[0].dz = 0.1;
        Real dt1 = EulerianTimeStepping::compute_dt(cells1, 1, 0.5);
        Real dt2 = EulerianTimeStepping::compute_dt(cells2, 1, 0.5);
        CHECK(dt2 < dt1, "CFL dt: velocity reduces dt");
    }

    // Test 3: CFL dt scales with CFL number
    {
        EulerCell cells[1];
        cells[0] = make_cell(1.0, 1.0, 0.0, 0.0, 1.0);
        cells[0].dx = 0.1; cells[0].dy = 0.1; cells[0].dz = 0.1;
        Real dt_05 = EulerianTimeStepping::compute_dt(cells, 1, 0.5);
        Real dt_09 = EulerianTimeStepping::compute_dt(cells, 1, 0.9);
        CHECK_NEAR(dt_09 / dt_05, 0.9 / 0.5, 1.0e-12, "CFL dt: scales linearly with CFL");
    }

    // Test 4: CFL dt for 2D solver
    {
        EulerCell cells[1];
        cells[0] = make_cell(1.225, 0.0, 0.0, 0.0, 101325.0);
        cells[0].dx = 0.01; cells[0].dy = 0.01;
        Real dt = EulerianTimeStepping::compute_dt_2d(cells, 1, 0.5);
        Real c = std::sqrt(1.4 * 101325.0 / 1.225);
        Real expected = 0.5 * 0.01 / c;
        CHECK_NEAR(dt, expected, 1.0e-10, "CFL dt 2D: atmospheric air");
    }

    // Test 5: dt decreases with smaller cells
    {
        EulerCell cells1[1], cells2[1];
        cells1[0] = make_cell(1.0, 0.0, 0.0, 0.0, 1.0);
        cells1[0].dx = 1.0; cells1[0].dy = 1.0; cells1[0].dz = 1.0;
        cells2[0] = make_cell(1.0, 0.0, 0.0, 0.0, 1.0);
        cells2[0].dx = 0.1; cells2[0].dy = 0.1; cells2[0].dz = 0.1;
        Real dt1 = EulerianTimeStepping::compute_dt(cells1, 1, 0.5);
        Real dt2 = EulerianTimeStepping::compute_dt(cells2, 1, 0.5);
        CHECK_NEAR(dt1 / dt2, 10.0, 1.0e-10, "CFL dt: scales with dx");
    }

    // Test 6: dt picks minimum across cells
    {
        EulerCell cells[3];
        cells[0] = make_cell(1.0, 0.0, 0.0, 0.0, 1.0);
        cells[0].dx = 1.0; cells[0].dy = 1.0; cells[0].dz = 1.0;
        cells[1] = make_cell(1.0, 100.0, 0.0, 0.0, 1.0);
        cells[1].dx = 1.0; cells[1].dy = 1.0; cells[1].dz = 1.0;
        cells[2] = make_cell(1.0, 0.0, 0.0, 0.0, 1.0);
        cells[2].dx = 1.0; cells[2].dy = 1.0; cells[2].dz = 1.0;
        Real dt = EulerianTimeStepping::compute_dt(cells, 3, 0.5);
        // Cell 1 has highest speed => determines dt
        Real c = std::sqrt(1.4);
        Real dt_cell1 = 0.5 * 1.0 / (100.0 + c);
        CHECK_NEAR(dt, dt_cell1, 1.0e-10, "CFL dt: picks minimum cell");
    }

    // Test 7: dt positive for valid input
    {
        EulerCell cells[2];
        cells[0] = make_cell(1.0, 0.0, 0.0, 0.0, 1.0);
        cells[0].dx = 0.1; cells[0].dy = 0.1; cells[0].dz = 0.1;
        cells[1] = make_cell(2.0, 5.0, 3.0, 1.0, 10.0);
        cells[1].dx = 0.1; cells[1].dy = 0.1; cells[1].dz = 0.1;
        Real dt = EulerianTimeStepping::compute_dt(cells, 2, 0.8);
        CHECK(dt > 0.0, "CFL dt: always positive");
    }

    // Test 8: dt handles high Mach number
    {
        EulerCell cells[1];
        cells[0] = make_cell(1.0, 1000.0, 0.0, 0.0, 1.0);
        cells[0].dx = 0.01; cells[0].dy = 0.01; cells[0].dz = 0.01;
        Real dt = EulerianTimeStepping::compute_dt(cells, 1, 0.5);
        Real c = std::sqrt(1.4);
        Real expected = 0.5 * 0.01 / (1000.0 + c);
        CHECK_NEAR(dt, expected, 1.0e-10, "CFL dt: high Mach");
    }

    // Test 9: y-direction dominates for anisotropic grid
    {
        EulerCell cells[1];
        cells[0] = make_cell(1.0, 0.0, 0.0, 0.0, 1.0);
        cells[0].dx = 1.0; cells[0].dy = 0.001; cells[0].dz = 1.0;
        Real dt = EulerianTimeStepping::compute_dt(cells, 1, 0.5);
        Real c = std::sqrt(1.4);
        Real dt_y = 0.5 * 0.001 / c;
        CHECK_NEAR(dt, dt_y, 1.0e-10, "CFL dt: anisotropic grid, dy dominates");
    }

    // Test 10: Sound speed increases with pressure
    {
        EulerCell c1 = make_cell(1.0, 0.0, 0.0, 0.0, 1.0);
        EulerCell c2 = make_cell(1.0, 0.0, 0.0, 0.0, 100.0);
        CHECK(c2.sound_speed() > c1.sound_speed(), "Sound speed: increases with pressure");
    }

    // Test 11: Sound speed formula
    {
        EulerCell c = make_cell(1.225, 0.0, 0.0, 0.0, 101325.0);
        Real cs = c.sound_speed();
        Real expected = std::sqrt(1.4 * 101325.0 / 1.225);
        CHECK_NEAR(cs, expected, 1.0e-6, "Sound speed: atmospheric air ~ 340 m/s");
    }

    // Test 12: CFL < 1 ensures stability (Courant condition)
    {
        EulerCell cells[1];
        cells[0] = make_cell(1.0, 0.0, 0.0, 0.0, 1.0);
        cells[0].dx = 1.0; cells[0].dy = 1.0; cells[0].dz = 1.0;
        Real c = std::sqrt(1.4);
        Real dt = EulerianTimeStepping::compute_dt(cells, 1, 0.9);
        Real courant = dt * c / 1.0;
        CHECK(courant <= 0.9 + 1.0e-12, "CFL: Courant number <= CFL");
    }
}

// ============================================================================
// Test Group 4: EulerianBCs (12 tests)
// ============================================================================
void test_eulerian_bcs() {
    std::cout << "=== EulerianBCs Tests ===\n";

    EulerCell interior = make_cell(1.0, 2.0, 3.0, 4.0, 5.0);
    EulerBCData bc_data;

    // Test 1: Transmitting BC copies state
    {
        EulerCell ghost;
        EulerianBCs::apply_bc(interior, ghost, EulerBCType::Transmitting, bc_data, 0);
        CHECK_NEAR(ghost.rho, interior.rho, 1.0e-15, "BC transmit: rho copied");
        CHECK_NEAR(ghost.u, interior.u, 1.0e-15, "BC transmit: u copied");
    }

    // Test 2: Reflecting BC reverses normal velocity (x-face)
    {
        EulerCell ghost;
        EulerianBCs::apply_bc(interior, ghost, EulerBCType::Reflecting, bc_data, 0);
        CHECK_NEAR(ghost.u, -2.0, 1.0e-15, "BC reflect x: u reversed");
        CHECK_NEAR(ghost.v, 3.0, 1.0e-15, "BC reflect x: v preserved");
        CHECK_NEAR(ghost.w, 4.0, 1.0e-15, "BC reflect x: w preserved");
    }

    // Test 3: Reflecting BC reverses normal velocity (y-face)
    {
        EulerCell ghost;
        EulerianBCs::apply_bc(interior, ghost, EulerBCType::Reflecting, bc_data, 2);
        CHECK_NEAR(ghost.u, 2.0, 1.0e-15, "BC reflect y: u preserved");
        CHECK_NEAR(ghost.v, -3.0, 1.0e-15, "BC reflect y: v reversed");
        CHECK_NEAR(ghost.w, 4.0, 1.0e-15, "BC reflect y: w preserved");
    }

    // Test 4: Reflecting BC reverses normal velocity (z-face)
    {
        EulerCell ghost;
        EulerianBCs::apply_bc(interior, ghost, EulerBCType::Reflecting, bc_data, 4);
        CHECK_NEAR(ghost.u, 2.0, 1.0e-15, "BC reflect z: u preserved");
        CHECK_NEAR(ghost.v, 3.0, 1.0e-15, "BC reflect z: v preserved");
        CHECK_NEAR(ghost.w, -4.0, 1.0e-15, "BC reflect z: w reversed");
    }

    // Test 5: Reflecting BC preserves density
    {
        EulerCell ghost;
        EulerianBCs::apply_bc(interior, ghost, EulerBCType::Reflecting, bc_data, 0);
        CHECK_NEAR(ghost.rho, interior.rho, 1.0e-15, "BC reflect: rho preserved");
    }

    // Test 6: Inflow BC sets prescribed state
    {
        bc_data.rho_bc = 2.0;
        bc_data.u_bc = 10.0;
        bc_data.v_bc = 0.0;
        bc_data.w_bc = 0.0;
        bc_data.p_bc = 200.0;
        bc_data.gamma_bc = 1.4;
        EulerCell ghost;
        EulerianBCs::apply_bc(interior, ghost, EulerBCType::Inflow, bc_data, 0);
        CHECK_NEAR(ghost.rho, 2.0, 1.0e-15, "BC inflow: rho = prescribed");
        CHECK_NEAR(ghost.u, 10.0, 1.0e-15, "BC inflow: u = prescribed");
        CHECK_NEAR(ghost.p, 200.0, 1.0e-10, "BC inflow: p = prescribed");
    }

    // Test 7: Outflow BC copies state
    {
        EulerCell ghost;
        EulerianBCs::apply_bc(interior, ghost, EulerBCType::Outflow, bc_data, 1);
        CHECK_NEAR(ghost.rho, interior.rho, 1.0e-15, "BC outflow: rho copied");
        CHECK_NEAR(ghost.u, interior.u, 1.0e-15, "BC outflow: u copied");
    }

    // Test 8: create_ghost helper works
    {
        EulerCell ghost = EulerianBCs::create_ghost(interior, EulerBCType::Reflecting, bc_data, 0);
        CHECK_NEAR(ghost.u, -interior.u, 1.0e-15, "create_ghost: reflect x");
    }

    // Test 9: Reflecting preserves pressure
    {
        EulerCell ghost;
        EulerianBCs::apply_bc(interior, ghost, EulerBCType::Reflecting, bc_data, 0);
        // p computed from total energy; should be close to original
        CHECK_NEAR(ghost.p, interior.p, 1.0e-10, "BC reflect: pressure preserved");
    }

    // Test 10: Reflecting BC on +x face still reverses u
    {
        EulerCell ghost;
        EulerianBCs::apply_bc(interior, ghost, EulerBCType::Reflecting, bc_data, 1);
        CHECK_NEAR(ghost.u, -interior.u, 1.0e-15, "BC reflect +x: u reversed");
    }

    // Test 11: Reflecting BC preserves kinetic energy
    {
        EulerCell ghost;
        EulerianBCs::apply_bc(interior, ghost, EulerBCType::Reflecting, bc_data, 0);
        Real ke_int = 0.5 * (interior.u * interior.u + interior.v * interior.v + interior.w * interior.w);
        Real ke_ghost = 0.5 * (ghost.u * ghost.u + ghost.v * ghost.v + ghost.w * ghost.w);
        CHECK_NEAR(ke_ghost, ke_int, 1.0e-10, "BC reflect: kinetic energy preserved");
    }

    // Test 12: Inflow total energy consistent
    {
        bc_data.rho_bc = 1.5;
        bc_data.u_bc = 5.0;
        bc_data.v_bc = 0.0;
        bc_data.w_bc = 0.0;
        bc_data.p_bc = 100.0;
        bc_data.gamma_bc = 1.4;
        EulerCell ghost;
        EulerianBCs::apply_bc(interior, ghost, EulerBCType::Inflow, bc_data, 0);
        Real E_expected = 100.0 / 0.4 + 0.5 * 1.5 * 25.0;
        CHECK_NEAR(ghost.E, E_expected, 1.0e-10, "BC inflow: E consistent with p, rho, u");
    }
}

// ============================================================================
// Test Group 5: Euler2DSolver (11 tests)
// ============================================================================
void test_euler2d_solver() {
    std::cout << "=== Euler2DSolver Tests ===\n";

    // Test 1: Grid initialization
    {
        Euler2DSolver solver(10, 5, 0.1, 0.1, 1.4, 0.5);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        CHECK(static_cast<int>(cells.size()) == 50, "2D grid: correct size");
    }

    // Test 2: Neighbor connectivity
    {
        Euler2DSolver solver(3, 3, 1.0, 1.0, 1.4, 0.5);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        // Center cell (1,1) = index 4
        CHECK(cells[4].neighbors[0] == 3, "2D grid: center left neighbor");
        CHECK(cells[4].neighbors[1] == 5, "2D grid: center right neighbor");
        CHECK(cells[4].neighbors[2] == 1, "2D grid: center bottom neighbor");
        CHECK(cells[4].neighbors[3] == 7, "2D grid: center top neighbor");
    }

    // Test 3: Corner cell boundary neighbors
    {
        Euler2DSolver solver(3, 3, 1.0, 1.0, 1.4, 0.5);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        // Cell (0,0) = index 0
        CHECK(cells[0].neighbors[0] == -1, "2D grid: corner -x boundary");
        CHECK(cells[0].neighbors[2] == -1, "2D grid: corner -y boundary");
    }

    // Test 4: Uniform state => no change after step
    {
        Euler2DSolver solver(5, 5, 0.1, 0.1, 1.4, 0.5);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        for (auto& c : cells) {
            c.rho = 1.0; c.u = 0.0; c.v = 0.0; c.w = 0.0;
            c.p = 1.0; c.compute_total_energy();
        }
        Real dt = solver.compute_dt(cells.data(), cells.size());
        std::vector<EulerCell> cells_copy = cells;
        solver.solve_step(cells.data(), cells.size(), dt * 0.5);
        Real max_drho = 0.0;
        for (int i = 0; i < static_cast<int>(cells.size()); ++i) {
            Real dr = std::abs(cells[i].rho - cells_copy[i].rho);
            if (dr > max_drho) max_drho = dr;
        }
        CHECK(max_drho < 1.0e-10, "2D solver: uniform state unchanged");
    }

    // Test 5: Mass conservation for Sod-like problem
    {
        Euler2DSolver solver(20, 1, 0.05, 0.05, 1.4, 0.5);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        for (int i = 0; i < 20; ++i) {
            if (i < 10) {
                cells[i].rho = 1.0; cells[i].p = 1.0;
            } else {
                cells[i].rho = 0.125; cells[i].p = 0.1;
            }
            cells[i].u = 0.0; cells[i].v = 0.0; cells[i].w = 0.0;
            cells[i].compute_total_energy();
        }
        // Total mass before
        Real mass_before = 0.0;
        for (int i = 0; i < 20; ++i) mass_before += cells[i].rho * 0.05 * 0.05;

        // Take some steps
        for (int step = 0; step < 10; ++step) {
            Real dt = solver.compute_dt(cells.data(), cells.size());
            solver.solve_step(cells.data(), cells.size(), dt);
        }

        Real mass_after = 0.0;
        for (int i = 0; i < 20; ++i) mass_after += cells[i].rho * 0.05 * 0.05;
        CHECK_NEAR(mass_after, mass_before, 1.0e-10, "2D Sod: mass conserved");
    }

    // Test 6: Sod shock tube produces correct qualitative structure
    {
        Euler2DSolver solver(100, 1, 0.01, 0.01, 1.4, 0.4);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        for (int i = 0; i < 100; ++i) {
            if (i < 50) {
                cells[i].rho = 1.0; cells[i].p = 1.0;
            } else {
                cells[i].rho = 0.125; cells[i].p = 0.1;
            }
            cells[i].u = 0.0; cells[i].v = 0.0; cells[i].w = 0.0;
            cells[i].compute_total_energy();
        }
        for (int step = 0; step < 50; ++step) {
            Real dt = solver.compute_dt(cells.data(), cells.size());
            solver.solve_step(cells.data(), cells.size(), dt);
        }
        // Shock should have moved right: density at cell 70 should be higher than initial 0.125
        CHECK(cells[70].rho > 0.125, "2D Sod: shock propagates right");
        // Rarefaction moves left: density at cell 30 should be lower than initial 1.0
        CHECK(cells[30].rho < 1.0, "2D Sod: rarefaction propagates left");
    }

    // Test 7: CFL dt positive
    {
        Euler2DSolver solver(5, 5, 0.1, 0.1, 1.4, 0.5);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        for (auto& c : cells) {
            c.rho = 1.0; c.u = 1.0; c.v = 1.0; c.w = 0.0;
            c.p = 1.0; c.compute_total_energy();
        }
        Real dt = solver.compute_dt(cells.data(), cells.size());
        CHECK(dt > 0.0, "2D solver: dt > 0");
    }

    // Test 8: Pressure positivity after step
    {
        Euler2DSolver solver(10, 10, 0.1, 0.1, 1.4, 0.3);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        for (auto& c : cells) {
            c.rho = 1.0; c.u = 0.0; c.v = 0.0; c.w = 0.0;
            c.p = 1.0; c.compute_total_energy();
        }
        // Perturb center
        cells[55].p = 10.0;
        cells[55].compute_total_energy();
        Real dt = solver.compute_dt(cells.data(), cells.size());
        solver.solve_step(cells.data(), cells.size(), dt);
        bool all_positive = true;
        for (auto& c : cells) {
            if (c.p < 0.0 || c.rho < 0.0) all_positive = false;
        }
        CHECK(all_positive, "2D solver: pressure and density remain positive");
    }

    // Test 9: Symmetric initial condition maintains symmetry
    {
        Euler2DSolver solver(5, 5, 0.1, 0.1, 1.4, 0.3);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        for (auto& c : cells) {
            c.rho = 1.0; c.u = 0.0; c.v = 0.0; c.w = 0.0;
            c.p = 1.0; c.compute_total_energy();
        }
        // Central pressure pulse
        cells[12].p = 5.0; cells[12].compute_total_energy();
        Real dt = solver.compute_dt(cells.data(), cells.size());
        solver.solve_step(cells.data(), cells.size(), dt);
        // Cells (1,2)=11 and (3,2)=13 should have same rho (x-symmetry)
        CHECK_NEAR(cells[11].rho, cells[13].rho, 1.0e-12, "2D solver: x-symmetry preserved");
        // Cells (2,1)=7 and (2,3)=17 should have same rho (y-symmetry)
        CHECK_NEAR(cells[7].rho, cells[17].rho, 1.0e-12, "2D solver: y-symmetry preserved");
    }
}

// ============================================================================
// Test Group 6: Euler3DSolver (11 tests)
// ============================================================================
void test_euler3d_solver() {
    std::cout << "=== Euler3DSolver Tests ===\n";

    // Test 1: Grid initialization
    {
        Euler3DSolver solver(3, 4, 5, 0.1, 0.1, 0.1, 1.4, 0.5);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        CHECK(static_cast<int>(cells.size()) == 60, "3D grid: correct size (3*4*5=60)");
    }

    // Test 2: 3D neighbor connectivity
    {
        Euler3DSolver solver(3, 3, 3, 1.0, 1.0, 1.0, 1.4, 0.5);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        // Center cell (1,1,1) = index 1 + 1*3 + 1*9 = 13
        CHECK(cells[13].neighbors[0] == 12, "3D grid: -x neighbor");
        CHECK(cells[13].neighbors[1] == 14, "3D grid: +x neighbor");
        CHECK(cells[13].neighbors[2] == 10, "3D grid: -y neighbor");
        CHECK(cells[13].neighbors[3] == 16, "3D grid: +y neighbor");
        CHECK(cells[13].neighbors[4] == 4,  "3D grid: -z neighbor");
        CHECK(cells[13].neighbors[5] == 22, "3D grid: +z neighbor");
    }

    // Test 3: Uniform state unchanged
    {
        Euler3DSolver solver(4, 4, 4, 0.1, 0.1, 0.1, 1.4, 0.5);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        for (auto& c : cells) {
            c.rho = 1.0; c.u = 0.0; c.v = 0.0; c.w = 0.0;
            c.p = 1.0; c.compute_total_energy();
        }
        Real dt = solver.compute_dt(cells.data(), cells.size());
        std::vector<EulerCell> orig = cells;
        solver.solve_step(cells.data(), cells.size(), dt * 0.5);
        Real max_drho = 0.0;
        for (size_t i = 0; i < cells.size(); ++i) {
            Real dr = std::abs(cells[i].rho - orig[i].rho);
            if (dr > max_drho) max_drho = dr;
        }
        CHECK(max_drho < 1.0e-10, "3D solver: uniform state unchanged");
    }

    // Test 4: Mass conservation
    {
        Euler3DSolver solver(10, 1, 1, 0.1, 0.1, 0.1, 1.4, 0.4);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        for (int i = 0; i < 10; ++i) {
            cells[i].rho = (i < 5) ? 1.0 : 0.125;
            cells[i].p = (i < 5) ? 1.0 : 0.1;
            cells[i].u = 0.0; cells[i].v = 0.0; cells[i].w = 0.0;
            cells[i].compute_total_energy();
        }
        Real vol = 0.1 * 0.1 * 0.1;
        Real mass0 = 0.0;
        for (int i = 0; i < 10; ++i) mass0 += cells[i].rho * vol;
        for (int step = 0; step < 5; ++step) {
            Real dt = solver.compute_dt(cells.data(), cells.size());
            solver.solve_step(cells.data(), cells.size(), dt);
        }
        Real mass1 = 0.0;
        for (int i = 0; i < 10; ++i) mass1 += cells[i].rho * vol;
        CHECK_NEAR(mass1, mass0, 1.0e-10, "3D solver: mass conserved");
    }

    // Test 5: CFL dt positive
    {
        Euler3DSolver solver(3, 3, 3, 0.1, 0.1, 0.1, 1.4, 0.5);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        for (auto& c : cells) {
            c.rho = 1.0; c.u = 1.0; c.v = 2.0; c.w = 3.0;
            c.p = 1.0; c.compute_total_energy();
        }
        Real dt = solver.compute_dt(cells.data(), cells.size());
        CHECK(dt > 0.0, "3D solver: dt > 0");
    }

    // Test 6: Energy conservation
    {
        Euler3DSolver solver(5, 1, 1, 0.2, 0.2, 0.2, 1.4, 0.3);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        for (int i = 0; i < 5; ++i) {
            cells[i].rho = 1.0; cells[i].u = 0.0; cells[i].v = 0.0; cells[i].w = 0.0;
            cells[i].p = (i == 2) ? 10.0 : 1.0;
            cells[i].compute_total_energy();
        }
        Real vol = 0.2 * 0.2 * 0.2;
        Real E0 = 0.0;
        for (int i = 0; i < 5; ++i) E0 += cells[i].E * vol;
        Real dt = solver.compute_dt(cells.data(), cells.size());
        solver.solve_step(cells.data(), cells.size(), dt);
        Real E1 = 0.0;
        for (int i = 0; i < 5; ++i) E1 += cells[i].E * vol;
        CHECK_NEAR(E1, E0, 1.0e-10, "3D solver: energy conserved");
    }

    // Test 7: Density positivity
    {
        Euler3DSolver solver(4, 4, 4, 0.1, 0.1, 0.1, 1.4, 0.3);
        std::vector<EulerCell> cells;
        solver.init_grid(cells);
        for (auto& c : cells) {
            c.rho = 1.0; c.u = 0.0; c.v = 0.0; c.w = 0.0;
            c.p = 1.0; c.compute_total_energy();
        }
        cells[0].p = 100.0; cells[0].compute_total_energy();
        Real dt = solver.compute_dt(cells.data(), cells.size());
        solver.solve_step(cells.data(), cells.size(), dt);
        bool all_pos = true;
        for (auto& c : cells) if (c.rho <= 0.0) all_pos = false;
        CHECK(all_pos, "3D solver: density positive");
    }

    // Test 8: Conservative/primitive round-trip
    {
        EulerCell c = make_cell(2.0, 3.0, -1.0, 0.5, 10.0);
        Real U[5];
        c.to_conservative(U);
        EulerCell c2;
        c2.gamma = 1.4;
        c2.from_conservative(U);
        CHECK_NEAR(c2.rho, c.rho, 1.0e-12, "Round-trip: rho");
        CHECK_NEAR(c2.u, c.u, 1.0e-12, "Round-trip: u");
        CHECK_NEAR(c2.v, c.v, 1.0e-12, "Round-trip: v");
        CHECK_NEAR(c2.w, c.w, 1.0e-12, "Round-trip: w");
        CHECK_NEAR(c2.p, c.p, 1.0e-10, "Round-trip: p");
    }

    // Test 9: Total energy formula
    {
        EulerCell c = make_cell(2.0, 3.0, 0.0, 0.0, 10.0);
        // E = p/(gamma-1) + 0.5*rho*u^2 = 10/0.4 + 0.5*2*9 = 25 + 9 = 34
        CHECK_NEAR(c.E, 34.0, 1.0e-12, "Total energy: E = p/(g-1) + 0.5*rho*u^2");
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "Wave 34a: Pure Eulerian Solver Test Suite\n";
    std::cout << "==========================================\n\n";

    test_eulerian_flux();
    test_eulerian_gradient();
    test_eulerian_timestepping();
    test_eulerian_bcs();
    test_euler2d_solver();
    test_euler3d_solver();

    std::cout << "\n==========================================\n";
    std::cout << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " total\n";

    return (tests_failed > 0) ? 1 : 0;
}
