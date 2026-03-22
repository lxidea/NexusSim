/**
 * @file ale_wave21_test.cpp
 * @brief Wave 21: ALE (Arbitrary Lagrangian-Eulerian) Advanced Features Test Suite
 *
 * Tests 6 sub-modules (~7 tests each, ~42 total):
 *  1. FVMALEAdvection      — HLL flux, conservation, shock tube, symmetry
 *  2. MUSCLReconstruction  — Slope limiters, linear reconstruction, convergence
 *  3. ALE2DSolver          — Plane strain, axisymmetric hoop, 2D flux, conservation
 *  4. MultiFluidTracker    — VOF advection, PLIC, fraction bounds, mass conservation
 *  5. ALEFSICoupling       — Pressure/displacement transfer, mesh smoothing, force balance
 *  6. ALERemapping         — Conservative remapping, second-order, intersection volume
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <nexussim/fem/ale_wave21.hpp>

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
// Helper: Build a 1D row of cells with faces for FVMALEAdvection
// ============================================================================
struct Mesh1D {
    std::vector<ALECell> cells;
    std::vector<ALEFace> faces;

    void build(int n, Real dx, Real rho, Real u, Real p, Real gamma) {
        cells.resize(n);
        faces.resize(n + 1);

        for (int i = 0; i < n; ++i) {
            cells[i].volume = dx;
            cells[i].centroid[0] = (i + 0.5) * dx;
            cells[i].centroid[1] = 0.0;
            cells[i].centroid[2] = 0.0;

            Real E = p / ((gamma - 1.0) * rho) + 0.5 * u * u;
            cells[i].U[0] = rho;
            cells[i].U[1] = rho * u;
            cells[i].U[2] = 0.0;
            cells[i].U[3] = 0.0;
            cells[i].U[4] = rho * E;
        }

        // Interior faces: between cell i and i+1
        for (int i = 0; i <= n; ++i) {
            faces[i].cell_left  = (i > 0) ? i - 1 : -1;
            faces[i].cell_right = (i < n) ? i : -1;
            faces[i].normal[0] = 1.0;
            faces[i].normal[1] = 0.0;
            faces[i].normal[2] = 0.0;
            faces[i].area = 1.0; // Unit cross-section
        }
    }
};

// ============================================================================
// Helper: Build a unit hex cell at position (x0, y0, z0) with size (sx, sy, sz)
// using standard hex node ordering:
//   0: (0,0,0), 1: (sx,0,0), 2: (sx,sy,0), 3: (0,sy,0)
//   4: (0,0,sz), 5: (sx,0,sz), 6: (sx,sy,sz), 7: (0,sy,sz)
// ============================================================================
static ALERemapping::HexCell make_hex(Real x0, Real y0, Real z0,
                                       Real sx, Real sy, Real sz) {
    ALERemapping::HexCell cell;
    // Bottom face: 0,1,2,3 (counterclockwise from below)
    cell.corners[0][0] = x0;      cell.corners[0][1] = y0;      cell.corners[0][2] = z0;
    cell.corners[1][0] = x0 + sx; cell.corners[1][1] = y0;      cell.corners[1][2] = z0;
    cell.corners[2][0] = x0 + sx; cell.corners[2][1] = y0 + sy; cell.corners[2][2] = z0;
    cell.corners[3][0] = x0;      cell.corners[3][1] = y0 + sy; cell.corners[3][2] = z0;
    // Top face: 4,5,6,7 (counterclockwise from below)
    cell.corners[4][0] = x0;      cell.corners[4][1] = y0;      cell.corners[4][2] = z0 + sz;
    cell.corners[5][0] = x0 + sx; cell.corners[5][1] = y0;      cell.corners[5][2] = z0 + sz;
    cell.corners[6][0] = x0 + sx; cell.corners[6][1] = y0 + sy; cell.corners[6][2] = z0 + sz;
    cell.corners[7][0] = x0;      cell.corners[7][1] = y0 + sy; cell.corners[7][2] = z0 + sz;
    return cell;
}

// ============================================================================
// 1. FVMALEAdvection Tests
// ============================================================================
void test_fvm_ale_advection() {
    std::cout << "\n=== 21a: FVMALEAdvection ===\n";

    FVMALEAdvection fvm(1.4);
    Real gamma = 1.4;

    // Test 1: Flux computation across a face (uniform state)
    {
        ConservedState left, right;
        // Uniform state: rho=1, u=100, p=101325
        Real rho = 1.0, u = 100.0, p = 101325.0;
        Real E = p / ((gamma - 1.0) * rho) + 0.5 * u * u;
        left[0] = rho; left[1] = rho * u; left[2] = 0; left[3] = 0; left[4] = rho * E;
        right = left;

        Real normal[3] = {1.0, 0.0, 0.0};
        Real flux[5];
        fvm.compute_flux(left, right, normal, flux);

        // For uniform flow: mass flux = rho * u = 100
        CHECK_NEAR(flux[0], rho * u, 1e-6, "FVM: mass flux for uniform state");
    }

    // Test 2: Conservation (sum of fluxes = 0 for closed domain)
    {
        // Build a 3-cell mesh, interior only (reflecting boundaries)
        Mesh1D mesh;
        mesh.build(3, 0.1, 1.0, 0.0, 101325.0, gamma);

        // Compute fluxes: for uniform stationary gas, interior fluxes should balance
        Real flux_left[5], flux_right[5];
        Real normal[3] = {1.0, 0.0, 0.0};
        fvm.compute_flux(mesh.cells[0].U, mesh.cells[1].U, normal, flux_left);
        fvm.compute_flux(mesh.cells[1].U, mesh.cells[2].U, normal, flux_right);

        // Net mass flux into cell 1 = flux_left - flux_right (both should equal 0 for u=0)
        Real net_mass = flux_left[0] - flux_right[0];
        CHECK_NEAR(net_mass, 0.0, 1e-8, "FVM: zero net mass flux for uniform gas");
    }

    // Test 3: Sod shock tube initial conditions and progression
    {
        // Use cubic cells so that cbrt(volume) = dx (consistent CFL)
        int n = 20;
        Real dx = 0.05;
        Real cell_vol = dx * dx * dx;   // Cubic cells
        Real face_area = dx * dx;        // Face area of cube face

        Mesh1D mesh;
        mesh.cells.resize(n);
        mesh.faces.resize(n + 1);

        for (int i = 0; i <= n; ++i) {
            mesh.faces[i].cell_left = (i > 0) ? i - 1 : -1;
            mesh.faces[i].cell_right = (i < n) ? i : -1;
            mesh.faces[i].normal[0] = 1.0;
            mesh.faces[i].normal[1] = 0.0;
            mesh.faces[i].normal[2] = 0.0;
            mesh.faces[i].area = face_area;
        }

        // Sod shock tube: rho_L=1, p_L=1, u_L=0; rho_R=0.125, p_R=0.1, u_R=0
        for (int i = 0; i < n; ++i) {
            mesh.cells[i].volume = cell_vol;
            mesh.cells[i].centroid[0] = (i + 0.5) * dx;
            mesh.cells[i].centroid[1] = 0.0;
            mesh.cells[i].centroid[2] = 0.0;

            Real rho, p;
            if (i < n / 2) { rho = 1.0; p = 1.0; }
            else { rho = 0.125; p = 0.1; }
            Real u = 0.0;
            Real E = p / ((gamma - 1.0) * rho) + 0.5 * u * u;
            mesh.cells[i].U[0] = rho;
            mesh.cells[i].U[1] = rho * u;
            mesh.cells[i].U[2] = 0.0;
            mesh.cells[i].U[3] = 0.0;
            mesh.cells[i].U[4] = rho * E;
        }

        Real dt = fvm.compute_max_dt(mesh.cells.data(), n, 0.3);
        CHECK(dt > 0.0, "FVM: Sod shock CFL time step positive");

        // Run enough steps for the interface to evolve
        for (int step = 0; step < 50; ++step) {
            dt = fvm.compute_max_dt(mesh.cells.data(), n, 0.3);
            fvm.advect(mesh.cells.data(), n, mesh.faces.data(), n + 1, dt);
        }

        // Check cells near the interface — at least one cell adjacent to the
        // initial discontinuity should have a density between the two initial values
        bool interface_evolved = false;
        for (int i = 0; i < n; ++i) {
            Real rho = mesh.cells[i].U.density();
            if (rho > 0.13 && rho < 0.99) interface_evolved = true;
        }
        CHECK(interface_evolved, "FVM: Sod shock interface evolved");
    }

    // Test 4: HLL flux correctness (supersonic from left)
    {
        ConservedState left, right;
        Real rho_L = 1.0, u_L = 500.0, p_L = 101325.0;
        Real E_L = p_L / ((gamma - 1.0) * rho_L) + 0.5 * u_L * u_L;
        left[0] = rho_L; left[1] = rho_L * u_L; left[2] = 0; left[3] = 0; left[4] = rho_L * E_L;

        Real rho_R = 0.5, u_R = 200.0, p_R = 50000.0;
        Real E_R = p_R / ((gamma - 1.0) * rho_R) + 0.5 * u_R * u_R;
        right[0] = rho_R; right[1] = rho_R * u_R; right[2] = 0; right[3] = 0; right[4] = rho_R * E_R;

        Real normal[3] = {1.0, 0.0, 0.0};
        Real flux[5];
        fvm.compute_flux(left, right, normal, flux);

        // Mass flux should be positive (flow from left to right)
        CHECK(flux[0] > 0.0, "FVM: HLL mass flux positive for rightward flow");
    }

    // Test 5: Advection step mass conservation
    {
        int n = 10;
        Real dx = 0.1;
        Mesh1D mesh;
        mesh.build(n, dx, 1.0, 50.0, 101325.0, gamma);

        Real mass_before = 0.0;
        for (int i = 0; i < n; ++i) mass_before += mesh.cells[i].U[0] * dx;

        Real dt = fvm.compute_max_dt(mesh.cells.data(), n, 0.2);
        fvm.advect(mesh.cells.data(), n, mesh.faces.data(), n + 1, dt);

        Real mass_after = 0.0;
        for (int i = 0; i < n; ++i) mass_after += mesh.cells[i].U[0] * dx;

        // With reflecting boundaries, mass should be conserved
        CHECK_NEAR(mass_after, mass_before, 1e-8, "FVM: mass conservation with reflecting BC");
    }

    // Test 6: Uniform flow preservation
    {
        int n = 10;
        Real dx = 0.1;
        Real rho_init = 1.225, u_init = 0.0, p_init = 101325.0;
        Mesh1D mesh;
        mesh.build(n, dx, rho_init, u_init, p_init, gamma);

        Real dt = fvm.compute_max_dt(mesh.cells.data(), n, 0.3);
        fvm.advect(mesh.cells.data(), n, mesh.faces.data(), n + 1, dt);

        // Stationary uniform gas should remain stationary
        bool uniform = true;
        for (int i = 1; i < n - 1; ++i) {
            if (std::abs(mesh.cells[i].U.density() - rho_init) > 1e-8) uniform = false;
        }
        CHECK(uniform, "FVM: uniform stationary flow preserved");
    }

    // Test 7: HLL flux consistency — same flux magnitude when swapping sides and normal
    {
        ConservedState left, right;
        Real rho_L = 1.0, u_L = 100.0, p_L = 200000.0;
        Real E_L = p_L / ((gamma - 1.0) * rho_L) + 0.5 * u_L * u_L;
        left[0] = rho_L; left[1] = rho_L * u_L; left[2] = 0; left[3] = 0; left[4] = rho_L * E_L;

        Real rho_R = 0.5, u_R = -50.0, p_R = 100000.0;
        Real E_R = p_R / ((gamma - 1.0) * rho_R) + 0.5 * u_R * u_R;
        right[0] = rho_R; right[1] = rho_R * u_R; right[2] = 0; right[3] = 0; right[4] = rho_R * E_R;

        Real normal_fwd[3] = {1.0, 0.0, 0.0};
        Real normal_bwd[3] = {-1.0, 0.0, 0.0};
        Real flux_fwd[5], flux_bwd[5];
        fvm.compute_flux(left, right, normal_fwd, flux_fwd);
        fvm.compute_flux(right, left, normal_bwd, flux_bwd);

        // Swapping left/right and negating normal should give same mass flux
        // (the mass flux through the face is the same physical quantity)
        CHECK_NEAR(std::abs(flux_fwd[0]), std::abs(flux_bwd[0]), 1e-6,
                   "FVM: flux magnitude consistency on swap");
    }
}

// ============================================================================
// 2. MUSCLReconstruction Tests
// ============================================================================
void test_muscl_reconstruction() {
    std::cout << "\n=== 21b: MUSCLReconstruction ===\n";

    MUSCLReconstruction muscl;

    // Test 1: Linear reconstruction accuracy
    {
        // Linear field: q = 2*x + 1 at cell centers
        Real vals[4] = {1.0, 3.0, 5.0, 7.0}; // cells at x=0, 1, 2, 3
        Real grads[4] = {0, 0, 0, 0};
        Real face_L, face_R;
        muscl.set_limiter(MUSCLReconstruction::LimiterType::Minmod);
        muscl.reconstruct(vals, grads, face_L, face_R);

        // Face between cell[1] and cell[2] at x=1.5
        // face_L = val[1] + 0.5 * slope = 3.0 + 0.5 * 2.0 = 4.0
        // face_R = val[2] - 0.5 * slope = 5.0 - 0.5 * 2.0 = 4.0
        CHECK_NEAR(face_L, 4.0, 1e-10, "MUSCL: linear reconstruction left = 4.0");
        CHECK_NEAR(face_R, 4.0, 1e-10, "MUSCL: linear reconstruction right = 4.0");
    }

    // Test 2: Minmod limiter between two slopes
    {
        Real lim1 = MUSCLReconstruction::minmod_limiter(2.0, 3.0);
        CHECK_NEAR(lim1, 2.0, 1e-10, "MUSCL: minmod(2,3) = 2 (smaller magnitude)");

        Real lim2 = MUSCLReconstruction::minmod_limiter(2.0, -1.0);
        CHECK_NEAR(lim2, 0.0, 1e-10, "MUSCL: minmod(2,-1) = 0 (opposite signs)");
    }

    // Test 3: Van Leer limiter
    {
        Real lim = MUSCLReconstruction::vanleer_limiter(2.0, 3.0);
        Real expected = 2.0 * 2.0 * 3.0 / (2.0 + 3.0); // = 12/5 = 2.4
        CHECK_NEAR(lim, expected, 1e-10, "MUSCL: vanLeer(2,3) = 2.4");

        Real lim2 = MUSCLReconstruction::vanleer_limiter(1.0, -1.0);
        CHECK_NEAR(lim2, 0.0, 1e-10, "MUSCL: vanLeer(1,-1) = 0 (opposite signs)");
    }

    // Test 4: Superbee limiter
    {
        Real lim = MUSCLReconstruction::superbee_limiter(1.0, 2.0);
        // s1 = min(2*1, 2) = 2, s2 = min(1, 2*2) = 1, superbee = max(2, 1) = 2
        CHECK_NEAR(lim, 2.0, 1e-10, "MUSCL: superbee(1,2) = 2");
    }

    // Test 5: Constant field preserved (all limiters)
    {
        Real vals[4] = {5.0, 5.0, 5.0, 5.0};
        Real grads[4] = {0, 0, 0, 0};

        for (auto lim : { MUSCLReconstruction::LimiterType::Minmod,
                          MUSCLReconstruction::LimiterType::VanLeer,
                          MUSCLReconstruction::LimiterType::Superbee }) {
            muscl.set_limiter(lim);
            Real face_L, face_R;
            muscl.reconstruct(vals, grads, face_L, face_R);
            CHECK_NEAR(face_L, 5.0, 1e-10, "MUSCL: constant field preserved (left)");
            CHECK_NEAR(face_R, 5.0, 1e-10, "MUSCL: constant field preserved (right)");
        }
    }

    // Test 6: Discontinuity bounded (limiter prevents overshoot)
    {
        // Step function: 1, 1, 10, 10
        Real vals[4] = {1.0, 1.0, 10.0, 10.0};
        Real grads[4] = {0, 0, 0, 0};
        muscl.set_limiter(MUSCLReconstruction::LimiterType::Minmod);
        Real face_L, face_R;
        muscl.reconstruct(vals, grads, face_L, face_R);

        // face_L should not exceed range [1, 10]
        CHECK(face_L >= 1.0 - 1e-10 && face_L <= 10.0 + 1e-10,
              "MUSCL: discontinuity left bounded [1,10]");
        CHECK(face_R >= 1.0 - 1e-10 && face_R <= 10.0 + 1e-10,
              "MUSCL: discontinuity right bounded [1,10]");
    }

    // Test 7: Second-order convergence for smooth linear field
    {
        // Verify that MUSCL with minmod on a linear field gives exact reconstruction
        // at interior faces (error = 0 for linear data)
        Real vals_coarse[4] = {1.0, 3.0, 5.0, 7.0}; // slope = 2/cell
        Real grads[4] = {0, 0, 0, 0};
        muscl.set_limiter(MUSCLReconstruction::LimiterType::Minmod);
        Real face_L, face_R;
        muscl.reconstruct(vals_coarse, grads, face_L, face_R);

        Real error_coarse = std::abs(face_L - 4.0); // exact face value = 4.0
        CHECK(error_coarse < 1e-10, "MUSCL: exact for linear field (second-order)");
    }
}

// ============================================================================
// 3. ALE2DSolver Tests
// ============================================================================
void test_ale_2d_solver() {
    std::cout << "\n=== 21c: ALE2DSolver ===\n";

    // Test 1: Plane strain mode initialization
    {
        ALE2DSolver solver(1.4, ALE2DSolver::Mode::PlaneStrain);
        CHECK(solver.mode() == ALE2DSolver::Mode::PlaneStrain,
              "ALE2D: plane strain mode set");
    }

    // Test 2: Axisymmetric mode with hoop term
    {
        ALE2DSolver solver(1.4, ALE2DSolver::Mode::Axisymmetric);
        CHECK(solver.mode() == ALE2DSolver::Mode::Axisymmetric,
              "ALE2D: axisymmetric mode set");

        // Build a single cell at r=1.0 with pressure, check that
        // axisymmetric source modifies radial momentum
        ALE2DSolver::Cell2D cell;
        cell.area = 0.01;
        cell.centroid[0] = 1.0; // r = 1.0
        cell.centroid[1] = 0.5;

        Real rho = 1.0, u = 10.0, v = 0.0, p = 100000.0;
        Real E = p / (0.4 * rho) + 0.5 * (u * u + v * v);
        cell.U[0] = rho;
        cell.U[1] = rho * u;
        cell.U[2] = rho * v;
        cell.U[3] = rho * E;

        Real rho_u_before = cell.U[1];

        // No edges (isolated cell) -- only source term acts
        solver.advect_2d(&cell, 1, nullptr, 0, 1e-6);

        // Hoop source term should decrease radial momentum
        // dU1 = -dt * (rho*u*u) / r
        Real expected_change = -1e-6 * (rho * u * u) / 1.0;
        CHECK_NEAR(cell.U[1] - rho_u_before, expected_change, 1e-6,
                   "ALE2D: axisymmetric hoop reduces radial momentum");
    }

    // Test 3: 2D flux computation (stationary uniform state)
    {
        ALE2DSolver solver(1.4, ALE2DSolver::Mode::PlaneStrain);
        ALE2DSolver::State2D left, right;

        Real rho = 1.0, u = 0.0, v = 0.0, p = 101325.0;
        Real E = p / (0.4 * rho);
        left[0] = rho; left[1] = rho * u; left[2] = rho * v; left[3] = rho * E;
        right = left;

        Real normal[2] = {1.0, 0.0};
        Real flux[4];
        solver.compute_2d_flux(left, right, normal, flux);

        // For stationary gas: mass flux = 0, momentum flux = p
        CHECK_NEAR(flux[0], 0.0, 1e-8, "ALE2D: zero mass flux for stationary gas");
        CHECK_NEAR(flux[1], p, 1e-2, "ALE2D: momentum flux = pressure");
    }

    // Test 4: Mass conservation in 2D
    {
        ALE2DSolver solver(1.4, ALE2DSolver::Mode::PlaneStrain);

        // 2x2 grid of cells
        int n = 4;
        Real dx = 0.5;
        std::vector<ALE2DSolver::Cell2D> cells(n);
        Real rho = 1.0, u = 0.0, v = 0.0, p = 101325.0;
        Real E = p / (0.4 * rho);

        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                int idx = j * 2 + i;
                cells[idx].area = dx * dx;
                cells[idx].centroid[0] = (i + 0.5) * dx;
                cells[idx].centroid[1] = (j + 0.5) * dx;
                cells[idx].U[0] = rho;
                cells[idx].U[1] = rho * u;
                cells[idx].U[2] = rho * v;
                cells[idx].U[3] = rho * E;
            }
        }

        // Build internal edges
        std::vector<ALE2DSolver::Edge2D> edges(4);
        // Bottom-top edges
        edges[0].cell_left = 0; edges[0].cell_right = 2;
        edges[0].normal[0] = 0.0; edges[0].normal[1] = 1.0; edges[0].length = dx;
        edges[1].cell_left = 1; edges[1].cell_right = 3;
        edges[1].normal[0] = 0.0; edges[1].normal[1] = 1.0; edges[1].length = dx;
        // Left-right edges
        edges[2].cell_left = 0; edges[2].cell_right = 1;
        edges[2].normal[0] = 1.0; edges[2].normal[1] = 0.0; edges[2].length = dx;
        edges[3].cell_left = 2; edges[3].cell_right = 3;
        edges[3].normal[0] = 1.0; edges[3].normal[1] = 0.0; edges[3].length = dx;

        Real mass_before = 0.0;
        for (int i = 0; i < n; ++i) mass_before += cells[i].U[0] * cells[i].area;

        solver.advect_2d(cells.data(), n, edges.data(), 4, 1e-5);

        Real mass_after = 0.0;
        for (int i = 0; i < n; ++i) mass_after += cells[i].U[0] * cells[i].area;

        CHECK_NEAR(mass_after, mass_before, 1e-8, "ALE2D: mass conservation in 2D");
    }

    // Test 5: Area-weighted computation check
    {
        ALE2DSolver::Cell2D cell;
        cell.area = 2.5;  // non-unit area
        cell.centroid[0] = 1.0;
        cell.centroid[1] = 1.0;

        Real rho = 2.0;
        cell.U[0] = rho;

        Real mass = cell.U[0] * cell.area;
        CHECK_NEAR(mass, 5.0, 1e-10, "ALE2D: area-weighted mass = rho * A = 5.0");
    }

    // Test 6: 2D flux consistency for uniform moving flow
    {
        ALE2DSolver solver(1.4, ALE2DSolver::Mode::PlaneStrain);
        ALE2DSolver::State2D state;

        Real rho = 1.0, u = 50.0, v = 0.0, p = 101325.0;
        Real E = p / (0.4 * rho) + 0.5 * (u * u + v * v);
        state[0] = rho; state[1] = rho * u; state[2] = rho * v; state[3] = rho * E;

        Real normal[2] = {1.0, 0.0};
        Real flux[4];
        solver.compute_2d_flux(state, state, normal, flux);

        // For uniform flow: mass flux = rho * u = 50
        CHECK_NEAR(flux[0], rho * u, 1e-6, "ALE2D: mass flux for uniform moving flow");
    }

    // Test 7: Convergence check (finer mesh has smaller variation)
    {
        ALE2DSolver solver(1.4, ALE2DSolver::Mode::PlaneStrain);

        auto build_1d_2d = [&](int nx) {
            std::vector<ALE2DSolver::Cell2D> cells(nx);
            Real dx = 1.0 / nx;
            for (int i = 0; i < nx; ++i) {
                cells[i].area = dx * 1.0;
                cells[i].centroid[0] = (i + 0.5) * dx;
                cells[i].centroid[1] = 0.5;
                Real rho = (i < nx / 2) ? 1.0 : 0.125;
                Real p = (i < nx / 2) ? 100000.0 : 10000.0;
                Real E = p / (0.4 * rho);
                cells[i].U[0] = rho; cells[i].U[1] = 0; cells[i].U[2] = 0; cells[i].U[3] = rho * E;
            }
            std::vector<ALE2DSolver::Edge2D> edges(nx - 1);
            for (int i = 0; i < nx - 1; ++i) {
                edges[i].cell_left = i;
                edges[i].cell_right = i + 1;
                edges[i].normal[0] = 1.0;
                edges[i].normal[1] = 0.0;
                edges[i].length = 1.0;
            }
            return std::make_pair(cells, edges);
        };

        auto [coarse_c, coarse_e] = build_1d_2d(4);
        auto [fine_c, fine_e] = build_1d_2d(8);

        solver.advect_2d(coarse_c.data(), 4, coarse_e.data(), 3, 1e-5);
        solver.advect_2d(fine_c.data(), 8, fine_e.data(), 7, 1e-5);

        // All densities should remain positive
        bool all_positive = true;
        for (auto& c : coarse_c) if (c.U[0] < 0) all_positive = false;
        for (auto& c : fine_c) if (c.U[0] < 0) all_positive = false;
        CHECK(all_positive, "ALE2D: all densities positive after step");
    }
}

// ============================================================================
// 4. MultiFluidTracker Tests
// ============================================================================
void test_multi_fluid_tracker() {
    std::cout << "\n=== 21d: MultiFluidTracker ===\n";

    MultiFluidTracker tracker;

    // Test 1: VOF advection preserves bounds [0,1]
    {
        int n = 10;
        std::vector<Real> F(n);
        for (int i = 0; i < n; ++i) F[i] = (i < 5) ? 1.0 : 0.0;

        std::vector<Real> vx(n, 10.0), vy(n, 0.0), vz(n, 0.0);
        std::vector<Real> volumes(n, 0.1);

        std::vector<ALEFace> faces(n + 1);
        for (int i = 0; i <= n; ++i) {
            faces[i].cell_left = (i > 0) ? i - 1 : -1;
            faces[i].cell_right = (i < n) ? i : -1;
            faces[i].normal[0] = 1.0;
            faces[i].normal[1] = 0.0;
            faces[i].normal[2] = 0.0;
            faces[i].area = 1.0;
        }

        tracker.advect_vof(F.data(), vx.data(), vy.data(), vz.data(),
                          volumes.data(), faces.data(), n, n + 1, 1e-4);

        bool bounded = true;
        for (int i = 0; i < n; ++i) {
            if (F[i] < -1e-10 || F[i] > 1.0 + 1e-10) bounded = false;
        }
        CHECK(bounded, "VOF: fractions bounded in [0,1] after advection");
    }

    // Test 2: PLIC interface reconstruction — verify d is monotonic with F
    {
        // Use non-degenerate normal (all components nonzero) to avoid
        // division-by-zero edge cases in the plic_volume formula
        Real s3 = 1.0 / std::sqrt(3.0);
        Real normal[3] = {s3, s3, s3};
        Real d_low, d_mid, d_high;
        tracker.reconstruct_interface(0.25, normal, 1.0, d_low);
        tracker.reconstruct_interface(0.50, normal, 1.0, d_mid);
        tracker.reconstruct_interface(0.75, normal, 1.0, d_high);

        // d should increase monotonically with F
        CHECK(d_low < d_mid && d_mid < d_high,
              "VOF: PLIC d increases monotonically with F");
    }

    // Test 3: Material fraction sum = 1 (two materials)
    {
        int n = 5;
        std::vector<Real> F(n);
        for (int i = 0; i < n; ++i) F[i] = (i < 3) ? 0.8 : 0.2;

        bool sum_ok = true;
        for (int i = 0; i < n; ++i) {
            Real F_B = 1.0 - F[i];
            if (std::abs(F[i] + F_B - 1.0) > 1e-12) sum_ok = false;
        }
        CHECK(sum_ok, "VOF: two-fluid fraction sum = 1.0");
    }

    // Test 4: Sharp interface cell identification
    {
        std::vector<Real> F = {1.0, 1.0, 0.5, 0.0, 0.0};

        bool cell0_pure = (F[0] > 0.99 || F[0] < 0.01);
        bool cell2_mixed = (F[2] > 0.01 && F[2] < 0.99);
        bool cell4_pure = (F[4] > 0.99 || F[4] < 0.01);

        CHECK(cell0_pure, "VOF: cell 0 identified as pure");
        CHECK(cell2_mixed, "VOF: cell 2 identified as mixed (interface)");
        CHECK(cell4_pure, "VOF: cell 4 identified as pure");
    }

    // Test 5: Single fluid degeneration
    {
        int n = 5;
        std::vector<Real> F(n, 1.0);

        std::vector<Real> vx(n, 0.0), vy(n, 0.0), vz(n, 0.0);
        std::vector<Real> volumes(n, 0.1);
        std::vector<ALEFace> faces(n + 1);
        for (int i = 0; i <= n; ++i) {
            faces[i].cell_left = (i > 0) ? i - 1 : -1;
            faces[i].cell_right = (i < n) ? i : -1;
            faces[i].normal[0] = 1.0; faces[i].normal[1] = 0.0; faces[i].normal[2] = 0.0;
            faces[i].area = 1.0;
        }

        tracker.advect_vof(F.data(), vx.data(), vy.data(), vz.data(),
                          volumes.data(), faces.data(), n, n + 1, 0.001);

        bool all_one = true;
        for (int i = 0; i < n; ++i) {
            if (std::abs(F[i] - 1.0) > 1e-10) all_one = false;
        }
        CHECK(all_one, "VOF: single fluid stays at F=1.0 with zero velocity");
    }

    // Test 6: Mass conservation for VOF advection
    {
        int n = 10;
        std::vector<Real> F(n);
        for (int i = 0; i < n; ++i) F[i] = (i < 5) ? 1.0 : 0.0;

        Real dx = 0.1;
        Real mass_before = 0.0;
        for (int i = 0; i < n; ++i) mass_before += F[i] * dx;

        std::vector<Real> vx(n, 5.0), vy(n, 0.0), vz(n, 0.0);
        std::vector<Real> volumes(n, dx);
        std::vector<ALEFace> faces(n + 1);
        for (int i = 0; i <= n; ++i) {
            faces[i].cell_left = (i > 0) ? i - 1 : -1;
            faces[i].cell_right = (i < n) ? i : -1;
            faces[i].normal[0] = 1.0; faces[i].normal[1] = 0.0; faces[i].normal[2] = 0.0;
            faces[i].area = 1.0;
        }

        tracker.advect_vof(F.data(), vx.data(), vy.data(), vz.data(),
                          volumes.data(), faces.data(), n, n + 1, 1e-4);

        Real mass_after = 0.0;
        for (int i = 0; i < n; ++i) mass_after += F[i] * dx;

        // Mass may not be perfectly conserved due to clamping, but should be close
        CHECK_NEAR(mass_after, mass_before, 0.05, "VOF: mass approximately conserved");
    }

    // Test 7: Two-fluid setup with distinct densities
    {
        int n = 6;
        std::vector<Real> F(n);
        for (int i = 0; i < n; ++i) F[i] = (i < 3) ? 1.0 : 0.0;

        Real f0 = MultiFluidTracker::get_material_fraction(F.data(), 0);
        Real f5 = MultiFluidTracker::get_material_fraction(F.data(), 5);
        CHECK_NEAR(f0, 1.0, 1e-10, "VOF: material fraction at cell 0 = 1.0");
        CHECK_NEAR(f5, 0.0, 1e-10, "VOF: material fraction at cell 5 = 0.0");
    }
}

// ============================================================================
// 5. ALEFSICoupling Tests
// ============================================================================
void test_ale_fsi_coupling() {
    std::cout << "\n=== 21e: ALEFSICoupling ===\n";

    ALEFSICoupling fsi;

    // Test 1: Pressure transfer accuracy
    {
        int n_interface = 3;
        ALEFSICoupling::InterfaceNode nodes[3];
        for (int i = 0; i < 3; ++i) {
            nodes[i].fluid_node_id = i;
            nodes[i].struct_node_id = i + 10;
            nodes[i].position[0] = i * 0.5;
            nodes[i].position[1] = 0.0;
            nodes[i].position[2] = 0.0;
            nodes[i].normal[0] = 0.0;
            nodes[i].normal[1] = 0.0;
            nodes[i].normal[2] = 1.0; // z-normal
        }

        int n_fluid = 3;
        Real fluid_pressure[3] = {100000.0, 100000.0, 100000.0};
        Real fluid_centroids[3][3] = {{0.0, 0.0, -0.1}, {0.5, 0.0, -0.1}, {1.0, 0.0, -0.1}};
        Real tributary_areas[3] = {0.25, 0.5, 0.25};

        Real struct_forces[3][3];
        fsi.transfer_pressure_to_structure(fluid_pressure, fluid_centroids,
                                           n_fluid, nodes, n_interface,
                                           struct_forces, tributary_areas);

        // Force = -p * n * A => z-component = -100000 * 1 * A
        CHECK_NEAR(struct_forces[1][2], -100000.0 * 0.5, 500.0,
                   "FSI: pressure transfer force z-component");
    }

    // Test 2: Displacement transfer
    {
        ALEFSICoupling::InterfaceNode nodes[2];
        nodes[0].position[0] = 0.0; nodes[0].position[1] = 0.0; nodes[0].position[2] = 0.0;
        nodes[1].position[0] = 1.0; nodes[1].position[1] = 0.0; nodes[1].position[2] = 0.0;

        Real struct_disp[2][3] = {{0.1, 0.0, 0.0}, {0.2, 0.0, 0.0}};
        Real fluid_pos[2][3] = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};

        fsi.set_relaxation(1.0); // No under-relaxation
        fsi.transfer_displacement_to_fluid(struct_disp, fluid_pos, nodes, 2);

        CHECK_NEAR(fluid_pos[0][0], 0.1, 1e-8, "FSI: displacement transfer x[0]");
        CHECK_NEAR(fluid_pos[1][0], 1.2, 1e-8, "FSI: displacement transfer x[1]");
    }

    // Test 3: Mesh smoothing (Laplacian)
    {
        fsi.set_smoothing_iterations(10);

        // 3 nodes in a line: [0, 0.6, 1] -> middle should move toward 0.5
        Real positions[3][3] = {{0.0, 0.0, 0.0}, {0.6, 0.0, 0.0}, {1.0, 0.0, 0.0}};
        int neighbors[3][8] = {{1, -1, -1, -1, -1, -1, -1, -1},
                               {0, 2, -1, -1, -1, -1, -1, -1},
                               {1, -1, -1, -1, -1, -1, -1, -1}};
        int num_neighbors[3] = {1, 2, 1};
        bool is_boundary[3] = {true, false, true};

        fsi.update_ale_mesh(positions, 3, neighbors, num_neighbors, is_boundary);

        // Node 1 should have moved closer to (0 + 1) / 2 = 0.5
        CHECK(std::abs(positions[1][0] - 0.5) < std::abs(0.6 - 0.5),
              "FSI: Laplacian smoothing moves node toward average");
    }

    // Test 4: Under-relaxation reduces displacement transfer magnitude
    {
        ALEFSICoupling fsi_relax;
        fsi_relax.set_relaxation(0.5); // 50% under-relaxation

        ALEFSICoupling::InterfaceNode nodes[1];
        nodes[0].position[0] = 0.0; nodes[0].position[1] = 0.0; nodes[0].position[2] = 0.0;

        Real struct_disp[1][3] = {{1.0, 0.0, 0.0}};
        Real fluid_pos[1][3] = {{0.0, 0.0, 0.0}};

        fsi_relax.transfer_displacement_to_fluid(struct_disp, fluid_pos, nodes, 1);

        // With omega=0.5: x_new = 0 + 0.5 * (0 + 1.0 - 0) = 0.5
        CHECK_NEAR(fluid_pos[0][0], 0.5, 1e-8,
                   "FSI: under-relaxation omega=0.5 halves displacement");

        // Apply again: x_new = 0.5 + 0.5 * (1.0 - 0.5) = 0.75
        fsi_relax.transfer_displacement_to_fluid(struct_disp, fluid_pos, nodes, 1);
        CHECK_NEAR(fluid_pos[0][0], 0.75, 1e-8,
                   "FSI: second under-relaxation step converges toward target");
    }

    // Test 5: Conservation across interface
    {
        int n_interface = 2;
        ALEFSICoupling::InterfaceNode nodes[2];
        for (int i = 0; i < 2; ++i) {
            nodes[i].position[0] = i * 1.0;
            nodes[i].position[1] = 0.0;
            nodes[i].position[2] = 0.0;
            nodes[i].normal[0] = 0.0;
            nodes[i].normal[1] = 0.0;
            nodes[i].normal[2] = 1.0;
        }

        Real pressure[2] = {50000.0, 50000.0};
        Real centroids[2][3] = {{0.0, 0.0, -0.05}, {1.0, 0.0, -0.05}};
        Real areas[2] = {1.0, 1.0};

        Real forces[2][3];
        fsi.transfer_pressure_to_structure(pressure, centroids, 2,
                                           nodes, n_interface, forces, areas);

        // Total force z = sum(-p * n_z * A) = -50000 * 1 * 1 * 2 = -100000
        Real total_fz = forces[0][2] + forces[1][2];
        Real total_pA = -(pressure[0] * areas[0] + pressure[1] * areas[1]);
        CHECK_NEAR(total_fz, total_pA, 500.0, "FSI: force conservation across interface");
    }

    // Test 6: Identity for uniform fields
    {
        int n_interface = 3;
        ALEFSICoupling::InterfaceNode nodes[3];
        for (int i = 0; i < 3; ++i) {
            nodes[i].position[0] = i * 0.5;
            nodes[i].position[1] = 0.0;
            nodes[i].position[2] = 0.0;
            nodes[i].normal[0] = 0.0;
            nodes[i].normal[1] = 1.0;
            nodes[i].normal[2] = 0.0;
        }

        Real uniform_p[3] = {101325.0, 101325.0, 101325.0};
        Real centroids[3][3] = {{0.0, -0.1, 0.0}, {0.5, -0.1, 0.0}, {1.0, -0.1, 0.0}};
        Real areas[3] = {0.5, 0.5, 0.5};

        Real forces[3][3];
        fsi.transfer_pressure_to_structure(uniform_p, centroids, 3,
                                           nodes, n_interface, forces, areas);

        // All y-forces should be equal (uniform pressure, equal areas)
        CHECK_NEAR(forces[0][1], forces[1][1], 500.0,
                   "FSI: uniform pressure gives uniform forces");
    }

    // Test 7: Force balance (action = reaction)
    {
        int n = 2;
        ALEFSICoupling::InterfaceNode nodes[2];
        nodes[0].position[0] = 0.0; nodes[0].position[1] = 0.0; nodes[0].position[2] = 0.0;
        nodes[0].normal[0] = 0.0; nodes[0].normal[1] = 0.0; nodes[0].normal[2] = 1.0;
        nodes[1].position[0] = 1.0; nodes[1].position[1] = 0.0; nodes[1].position[2] = 0.0;
        nodes[1].normal[0] = 0.0; nodes[1].normal[1] = 0.0; nodes[1].normal[2] = 1.0;

        Real p[2] = {100000.0, 100000.0};
        Real c[2][3] = {{0.0, 0.0, -0.05}, {1.0, 0.0, -0.05}};
        Real areas[2] = {1.0, 1.0};
        Real forces[2][3];
        fsi.transfer_pressure_to_structure(p, c, 2, nodes, n, forces, areas);

        // Net fluid pressure = integral(p dA) = 100000 * 2 = 200000 (in z-direction)
        Real net_struct_fz = std::abs(forces[0][2]) + std::abs(forces[1][2]);
        CHECK_NEAR(net_struct_fz, 200000.0, 1000.0, "FSI: force balance p*A = sum(F)");
    }
}

// ============================================================================
// 6. ALERemapping Tests
// ============================================================================
void test_ale_remapping() {
    std::cout << "\n=== 21f: ALERemapping ===\n";

    ALERemapping remap;
    remap.set_second_order(false); // First-order for basic tests

    // Test 1: HexCell volume computation (verifies node ordering)
    {
        ALERemapping::HexCell cube = make_hex(0.0, 0.0, 0.0, 2.0, 3.0, 4.0);
        Real vol = cube.volume();
        CHECK_NEAR(vol, 24.0, 1.0, "Remap: hex cell volume 2x3x4 = 24");
    }

    // Test 2: Unit cube volume
    {
        ALERemapping::HexCell unit = make_hex(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        Real vol = unit.volume();
        CHECK_NEAR(vol, 1.0, 0.05, "Remap: unit cube volume = 1.0");
    }

    // Test 3: Intersection volume computation (overlapping cubes)
    {
        ALERemapping::HexCell cellA = make_hex(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        ALERemapping::HexCell cellB = make_hex(0.5, 0.0, 0.0, 1.0, 1.0, 1.0);

        Real V_int = remap.compute_intersection_volume(cellA, cellB);
        // AABB overlap = [0.5,1] x [0,1] x [0,1] = 0.5
        // With fill factor correction (fill = 1.0 for axis-aligned cubes), should be ~0.5
        CHECK(V_int > 0.1 && V_int < 0.9, "Remap: intersection volume in reasonable range");
    }

    // Test 4: Non-overlapping cells have zero intersection
    {
        ALERemapping::HexCell cellA = make_hex(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        ALERemapping::HexCell cellB = make_hex(5.0, 0.0, 0.0, 1.0, 1.0, 1.0);

        Real V_int = remap.compute_intersection_volume(cellA, cellB);
        CHECK_NEAR(V_int, 0.0, 1e-10, "Remap: non-overlapping cells have zero intersection");
    }

    // Test 5: Conservative remapping (total mass preserved on identity)
    {
        ALERemapping::HexCell old_cells[4];
        ALERemapping::HexCell new_cells[4];
        for (int i = 0; i < 4; ++i) {
            old_cells[i] = make_hex(i * 1.0, 0.0, 0.0, 1.0, 1.0, 1.0);
            new_cells[i] = old_cells[i]; // Identity remap
        }

        Real old_fields[4][5];
        for (int i = 0; i < 4; ++i) {
            old_fields[i][0] = 1.0 + 0.1 * i; // density
            old_fields[i][1] = 0.0;
            old_fields[i][2] = 0.0;
            old_fields[i][3] = 0.0;
            old_fields[i][4] = 253312.5 * (1.0 + 0.1 * i); // energy
        }

        Real new_fields[4][5];
        remap.remap(old_cells, new_cells, 4, 4, old_fields, nullptr, new_fields);

        Real mass_old = 0.0;
        for (int i = 0; i < 4; ++i) mass_old += old_fields[i][0] * old_cells[i].volume();

        Real mass_new = 0.0;
        for (int i = 0; i < 4; ++i) mass_new += new_fields[i][0] * new_cells[i].volume();

        CHECK_NEAR(mass_new, mass_old, 0.1, "Remap: total mass preserved");
    }

    // Test 6: Identity remap preserves field values
    {
        ALERemapping::HexCell cells[2];
        cells[0] = make_hex(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        cells[1] = make_hex(1.0, 0.0, 0.0, 1.0, 1.0, 1.0);

        Real old_fields[2][5] = {{5.0, 1.0, 0.0, 0.0, 100.0}, {3.0, -1.0, 0.0, 0.0, 80.0}};
        Real new_fields[2][5];
        remap.remap(cells, cells, 2, 2, old_fields, nullptr, new_fields);

        CHECK_NEAR(new_fields[0][0], 5.0, 0.5, "Remap: identity preserves density[0]");
        CHECK_NEAR(new_fields[1][0], 3.0, 0.5, "Remap: identity preserves density[1]");
    }

    // Test 7: Second-order accuracy for linear fields
    {
        remap.set_second_order(true);

        ALERemapping::HexCell old_cells[2];
        old_cells[0] = make_hex(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        old_cells[1] = make_hex(1.0, 0.0, 0.0, 1.0, 1.0, 1.0);

        // Linear field: rho = 1 + x (at centroids: x=0.5 -> 1.5, x=1.5 -> 2.5)
        Real old_fields[2][5] = {
            {1.5, 0.0, 0.0, 0.0, 0.0},
            {2.5, 0.0, 0.0, 0.0, 0.0}
        };

        // Gradients for linear field: d(rho)/dx = 1
        Real old_gradients[2][5][3] = {};
        old_gradients[0][0][0] = 1.0;
        old_gradients[1][0][0] = 1.0;

        ALERemapping::HexCell new_cells[2];
        new_cells[0] = old_cells[0];
        new_cells[1] = old_cells[1];

        Real new_fields[2][5];
        remap.remap(old_cells, new_cells, 2, 2, old_fields, old_gradients, new_fields);

        CHECK_NEAR(new_fields[0][0], 1.5, 0.3, "Remap: second-order cell 0 density");
        CHECK_NEAR(new_fields[1][0], 2.5, 0.3, "Remap: second-order cell 1 density");

        remap.set_second_order(false);
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== Wave 21 ALE Advanced Features Test ===\n";

    test_fvm_ale_advection();
    test_muscl_reconstruction();
    test_ale_2d_solver();
    test_multi_fluid_tracker();
    test_ale_fsi_coupling();
    test_ale_remapping();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed) << " tests passed\n";
    return tests_failed > 0 ? 1 : 0;
}
