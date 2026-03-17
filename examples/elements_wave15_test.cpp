/**
 * @file elements_wave15_test.cpp
 * @brief Comprehensive test for Wave 15: Element Formulation Expansion
 *
 * Tests 7 element types with 50+ tests total:
 *  1. ThickShell8          - 8-node thick shell
 *  2. ThickShell6          - 6-node thick shell (wedge)
 *  3. DKTShell             - Discrete Kirchhoff Triangle
 *  4. DKQShell             - Discrete Kirchhoff Quadrilateral
 *  5. PlaneElement         - 2D plane-stress / plane-strain
 *  6. AxisymmetricElement  - Axisymmetric solid of revolution
 *  7. ConnectorElement     - Spot weld / rivet / fastener
 */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "../include/nexussim/physics/elements_wave15.hpp"

using Real = double;
using namespace nxs::elements;

static int pass_count = 0, fail_count = 0;

#define CHECK(cond, name) do { \
    if (cond) { printf("[PASS] %s\n", name); pass_count++; } \
    else { printf("[FAIL] %s\n", name); fail_count++; } \
} while(0)

#define CHECK_NEAR(val, expected, rtol, name) do { \
    double v_ = (val), e_ = (expected); \
    double diff_ = std::abs(v_ - e_); \
    double denom_ = std::abs(e_) > 1e-30 ? std::abs(e_) : 1.0; \
    if (diff_ / denom_ < (rtol)) { printf("[PASS] %s (got %g, expected %g)\n", name, v_, e_); pass_count++; } \
    else { printf("[FAIL] %s (got %g, expected %g, rdiff %g)\n", name, v_, e_, diff_/denom_); fail_count++; } \
} while(0)

// ============================================================================
// Helper: unit cube coords for ThickShell8 (1x1x0.1 plate)
// ============================================================================
static void make_plate_8node(Real* coords, Real Lx, Real Ly, Real t) {
    // Bottom face z=0, top face z=t
    // Node order: 0(-,-,-) 1(+,-,-) 2(+,+,-) 3(-,+,-) 4(-,-,+) 5(+,-,+) 6(+,+,+) 7(-,+,+)
    Real x[8] = {0,Lx,Lx,0, 0,Lx,Lx,0};
    Real y[8] = {0,0,Ly,Ly, 0,0,Ly,Ly};
    Real z[8] = {0,0,0,0, t,t,t,t};
    for (int i = 0; i < 8; ++i) {
        coords[i*3+0] = x[i];
        coords[i*3+1] = y[i];
        coords[i*3+2] = z[i];
    }
}

// ============================================================================
// 1. ThickShell8 Tests
// ============================================================================
void test_thick_shell_8() {
    printf("\n--- ThickShell8 Tests ---\n");

    Real coords[24];
    make_plate_8node(coords, 1.0, 1.0, 0.1);
    Real E = 210000.0, nu = 0.3;
    ThickShell8 elem(coords, E, nu);

    // Test 1: Node count
    CHECK(elem.num_nodes() == 8, "TS8: num_nodes == 8");

    // Test 2: DOF per node
    CHECK(elem.dof_per_node() == 3, "TS8: dof_per_node == 3");

    // Test 3: Integration points
    CHECK(elem.num_integration_points() == 8, "TS8: num_integration_points == 8");

    // Test 4: Shape function partition of unity
    Real N[8];
    elem.shape_functions(0.3, -0.2, 0.5, N);
    Real sum = 0; for (int i = 0; i < 8; ++i) sum += N[i];
    CHECK_NEAR(sum, 1.0, 1e-12, "TS8: shape function partition of unity");

    // Test 5: Shape function at corner node 0 = (-1,-1,-1)
    elem.shape_functions(-1.0, -1.0, -1.0, N);
    CHECK_NEAR(N[0], 1.0, 1e-12, "TS8: N0 at corner (-1,-1,-1)");
    CHECK_NEAR(N[1], 0.0, 1e-12, "TS8: N1 at corner (-1,-1,-1)");

    // Test 6: Volume = Lx * Ly * t = 1.0*1.0*0.1 = 0.1
    Real vol = elem.volume();
    CHECK_NEAR(vol, 0.1, 1e-10, "TS8: volume of 1x1x0.1 plate");

    // Test 7: Stiffness matrix symmetry
    Real K[24*24];
    elem.stiffness_matrix(K);
    bool sym = true;
    for (int i = 0; i < 24 && sym; ++i)
        for (int j = i+1; j < 24 && sym; ++j)
            if (std::abs(K[i*24+j] - K[j*24+i]) > 1e-6 * (std::abs(K[i*24+j]) + 1e-30))
                sym = false;
    CHECK(sym, "TS8: stiffness matrix symmetry");

    // Test 8: Stiffness diagonal positive
    bool diag_pos = true;
    for (int i = 0; i < 24; ++i)
        if (K[i*24+i] <= 0) diag_pos = false;
    CHECK(diag_pos, "TS8: stiffness diagonal positive");

    // Test 9: Rigid body mode (uniform translation) -> zero strain
    Real B[6*24];
    elem.strain_displacement_matrix(0.0, 0.0, 0.0, B);
    Real u_rigid[24];
    for (int i = 0; i < 8; ++i) { u_rigid[i*3+0]=1.0; u_rigid[i*3+1]=0.0; u_rigid[i*3+2]=0.0; }
    Real strain[6];
    detail::zero(strain, 6);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 24; ++j)
            strain[i] += B[i*24+j] * u_rigid[j];
    Real strain_norm = 0; for (int i = 0; i < 6; ++i) strain_norm += strain[i]*strain[i];
    CHECK(strain_norm < 1e-20, "TS8: rigid body translation -> zero strain");

    // Test 10: Jacobian positive at center
    Real J[9];
    Real detJ = elem.jacobian(0, 0, 0, J);
    CHECK(detJ > 0, "TS8: positive Jacobian at center");
}

// ============================================================================
// 2. ThickShell6 Tests
// ============================================================================
void test_thick_shell_6() {
    printf("\n--- ThickShell6 Tests ---\n");

    // Wedge: bottom triangle at z=0, top at z=0.1
    // Triangle: (0,0), (1,0), (0,1)
    Real coords[18] = {
        0,0,0,  1,0,0,  0,1,0,     // bottom
        0,0,0.1, 1,0,0.1, 0,1,0.1  // top
    };
    Real E = 210000.0, nu = 0.3;
    ThickShell6 elem(coords, E, nu);

    // Test 1: Node count
    CHECK(elem.num_nodes() == 6, "TS6: num_nodes == 6");

    // Test 2: DOF per node
    CHECK(elem.dof_per_node() == 3, "TS6: dof_per_node == 3");

    // Test 3: Integration points
    CHECK(elem.num_integration_points() == 6, "TS6: num_integration_points == 6");

    // Test 4: Shape function partition of unity
    Real N[6];
    elem.shape_functions(0.2, 0.3, 0.0, N);
    Real sum = 0; for (int i = 0; i < 6; ++i) sum += N[i];
    CHECK_NEAR(sum, 1.0, 1e-12, "TS6: shape function partition of unity");

    // Test 5: Shape function at node 0 (xi=1, eta=0, zeta=-1)
    elem.shape_functions(1.0, 0.0, -1.0, N);
    CHECK_NEAR(N[0], 1.0, 1e-12, "TS6: N0 at node 0");

    // Test 6: Volume = 0.5*1*1*0.1 = 0.05
    Real vol = elem.volume();
    CHECK_NEAR(vol, 0.05, 1e-8, "TS6: volume of triangular wedge");

    // Test 7: Stiffness matrix symmetry
    Real K[18*18];
    elem.stiffness_matrix(K);
    bool sym = true;
    for (int i = 0; i < 18 && sym; ++i)
        for (int j = i+1; j < 18 && sym; ++j)
            if (std::abs(K[i*18+j] - K[j*18+i]) > 1e-6*(std::abs(K[i*18+j])+1e-30))
                sym = false;
    CHECK(sym, "TS6: stiffness matrix symmetry");

    // Test 8: Diagonal positive
    bool diag_pos = true;
    for (int i = 0; i < 18; ++i)
        if (K[i*18+i] <= 0) diag_pos = false;
    CHECK(diag_pos, "TS6: stiffness diagonal positive");

    // Test 9: Jacobian at centroid positive
    Real J[9];
    Real detJ = elem.jacobian(1.0/3.0, 1.0/3.0, 0.0, J);
    CHECK(detJ > 0, "TS6: positive Jacobian at centroid");

    // Test 10: Shape function at node 4 (xi=0, eta=1, zeta=+1)
    elem.shape_functions(0.0, 1.0, 1.0, N);
    CHECK_NEAR(N[4], 1.0, 1e-12, "TS6: N4 at node 4");
}

// ============================================================================
// 3. DKTShell Tests
// ============================================================================
void test_dkt_shell() {
    printf("\n--- DKTShell Tests ---\n");

    // Right triangle in x-y plane
    Real coords[6] = {0,0, 1,0, 0,1};
    Real E = 210000, nu = 0.3, t = 0.01;
    DKTShell elem(coords, E, nu, t);

    // Test 1: Node count
    CHECK(elem.num_nodes() == 3, "DKT: num_nodes == 3");

    // Test 2: DOF per node
    CHECK(elem.dof_per_node() == 3, "DKT: dof_per_node == 3");

    // Test 3: Integration points
    CHECK(elem.num_integration_points() == 3, "DKT: num_integration_points == 3");

    // Test 4: Thickness
    CHECK_NEAR(elem.thickness(), 0.01, 1e-12, "DKT: thickness");

    // Test 5: Triangle area
    CHECK_NEAR(elem.area(), 0.5, 1e-12, "DKT: area of unit right triangle");

    // Test 6: Shape function partition of unity
    Real N[3];
    elem.shape_functions(0.25, 0.25, N);
    Real sum = N[0]+N[1]+N[2];
    CHECK_NEAR(sum, 1.0, 1e-12, "DKT: shape function partition of unity");

    // Test 7: Stiffness matrix symmetry
    Real K[81];
    elem.stiffness_matrix(K);
    bool sym = true;
    for (int i = 0; i < 9 && sym; ++i)
        for (int j = i+1; j < 9 && sym; ++j)
            if (std::abs(K[i*9+j] - K[j*9+i]) > 1e-4*(std::abs(K[i*9+j])+1e-30))
                sym = false;
    CHECK(sym, "DKT: stiffness matrix symmetry");

    // Test 8: Stiffness not all zero
    Real Ksum = 0;
    for (int i = 0; i < 81; ++i) Ksum += std::abs(K[i]);
    CHECK(Ksum > 0, "DKT: stiffness matrix not all zero");

    // Test 9: Bending B-matrix has correct dimensions (3x9)
    Real Bb[27];
    elem.bending_B_matrix(1.0/3.0, 1.0/3.0, Bb);
    bool has_nonzero = false;
    for (int i = 0; i < 27; ++i) if (std::abs(Bb[i]) > 1e-15) has_nonzero = true;
    CHECK(has_nonzero, "DKT: B-matrix has non-zero entries");

    // Test 10: Internal forces from uniform moment
    Real moments[9] = {1,0,0, 1,0,0, 1,0,0}; // Mxx at 3 GP
    Real fint[9];
    elem.compute_internal_forces(moments, fint);
    Real fint_norm = 0;
    for (int i = 0; i < 9; ++i) fint_norm += fint[i]*fint[i];
    CHECK(fint_norm > 0, "DKT: internal forces non-zero for uniform Mxx");
}

// ============================================================================
// 4. DKQShell Tests
// ============================================================================
void test_dkq_shell() {
    printf("\n--- DKQShell Tests ---\n");

    // Unit square
    Real coords[8] = {0,0, 1,0, 1,1, 0,1};
    Real E = 210000, nu = 0.3, t = 0.01;
    DKQShell elem(coords, E, nu, t);

    // Test 1: Node count
    CHECK(elem.num_nodes() == 4, "DKQ: num_nodes == 4");

    // Test 2: DOF per node
    CHECK(elem.dof_per_node() == 3, "DKQ: dof_per_node == 3");

    // Test 3: Integration points
    CHECK(elem.num_integration_points() == 4, "DKQ: num_integration_points == 4");

    // Test 4: Area
    CHECK_NEAR(elem.area(), 1.0, 1e-12, "DKQ: area of unit square");

    // Test 5: Shape function partition of unity
    Real N[4];
    elem.shape_functions(0.3, -0.2, N);
    Real sum = 0; for (int i = 0; i < 4; ++i) sum += N[i];
    CHECK_NEAR(sum, 1.0, 1e-12, "DKQ: shape function partition of unity");

    // Test 6: Shape function at corner node 0 = (-1,-1)
    elem.shape_functions(-1.0, -1.0, N);
    CHECK_NEAR(N[0], 1.0, 1e-12, "DKQ: N0 at corner (-1,-1)");

    // Test 7: Stiffness matrix symmetry
    Real K[144];
    elem.stiffness_matrix(K);
    bool sym = true;
    for (int i = 0; i < 12 && sym; ++i)
        for (int j = i+1; j < 12 && sym; ++j)
            if (std::abs(K[i*12+j] - K[j*12+i]) > 1e-4*(std::abs(K[i*12+j])+1e-30))
                sym = false;
    CHECK(sym, "DKQ: stiffness matrix symmetry");

    // Test 8: Stiffness not zero
    Real Ksum = 0;
    for (int i = 0; i < 144; ++i) Ksum += std::abs(K[i]);
    CHECK(Ksum > 0, "DKQ: stiffness matrix not all zero");

    // Test 9: Bending B-matrix at center
    Real Bb[36];
    elem.bending_B_matrix(0.0, 0.0, Bb);
    bool has_nonzero = false;
    for (int i = 0; i < 36; ++i) if (std::abs(Bb[i]) > 1e-15) has_nonzero = true;
    CHECK(has_nonzero, "DKQ: B-matrix has non-zero entries at center");

    // Test 10: Internal forces
    Real moments[12] = {1,0,0, 1,0,0, 1,0,0, 1,0,0};
    Real fint[12];
    elem.compute_internal_forces(moments, fint);
    Real fint_norm = 0;
    for (int i = 0; i < 12; ++i) fint_norm += fint[i]*fint[i];
    CHECK(fint_norm > 0, "DKQ: internal forces non-zero for uniform Mxx");
}

// ============================================================================
// 5. PlaneElement Tests
// ============================================================================
void test_plane_element() {
    printf("\n--- PlaneElement Tests ---\n");

    // --- Quad4 plane stress ---
    Real q_coords[8] = {0,0, 1,0, 1,1, 0,1};
    Real E = 200000, nu = 0.3, thickness = 0.5;
    PlaneElement q4(q_coords, 4, E, nu, thickness, PlaneMode::PlaneStress);

    // Test 1: Node count
    CHECK(q4.num_nodes() == 4, "PE_Q4: num_nodes == 4");

    // Test 2: Mode
    CHECK(q4.mode() == PlaneMode::PlaneStress, "PE_Q4: mode == PlaneStress");

    // Test 3: Topology
    CHECK(q4.topology() == PlaneTopology::Quad4, "PE_Q4: topology == Quad4");

    // Test 4: Shape function PU
    Real N[4];
    q4.shape_functions(0.3, -0.2, N);
    Real sum = 0; for (int i = 0; i < 4; ++i) sum += N[i];
    CHECK_NEAR(sum, 1.0, 1e-12, "PE_Q4: shape function PU");

    // Test 5: Stiffness symmetry
    Real K[64];
    q4.stiffness_matrix(K);
    bool sym = true;
    for (int i = 0; i < 8 && sym; ++i)
        for (int j = i+1; j < 8 && sym; ++j)
            if (std::abs(K[i*8+j] - K[j*8+i]) > 1e-6*(std::abs(K[i*8+j])+1e-30))
                sym = false;
    CHECK(sym, "PE_Q4: stiffness symmetry");

    // Test 6: Rigid body -> zero internal force
    Real stress_zero[12] = {};
    Real fint[8];
    q4.compute_internal_forces(stress_zero, fint);
    Real fn = 0; for (int i = 0; i < 8; ++i) fn += fint[i]*fint[i];
    CHECK(fn < 1e-20, "PE_Q4: zero stress -> zero internal force");

    // Test 7: Area
    CHECK_NEAR(q4.area(), 1.0, 1e-12, "PE_Q4: area of unit square");

    // --- Tri3 plane strain ---
    Real t_coords[6] = {0,0, 2,0, 0,2};
    PlaneElement t3(t_coords, 3, E, nu, thickness, PlaneMode::PlaneStrain);

    // Test 8: Node count
    CHECK(t3.num_nodes() == 3, "PE_T3: num_nodes == 3");

    // Test 9: Mode
    CHECK(t3.mode() == PlaneMode::PlaneStrain, "PE_T3: mode == PlaneStrain");

    // Test 10: Shape function PU for Tri3
    Real Nt[3];
    t3.shape_functions(0.3, 0.2, Nt);
    sum = 0; for (int i = 0; i < 3; ++i) sum += Nt[i];
    CHECK_NEAR(sum, 1.0, 1e-12, "PE_T3: shape function PU");

    // Test 11: Area = 0.5*2*2 = 2
    CHECK_NEAR(t3.area(), 2.0, 1e-12, "PE_T3: area of triangle");

    // Test 12: Constitutive matrix differs between stress/strain
    Real C_stress[9], C_strain[9];
    q4.constitutive_matrix(C_stress);
    t3.constitutive_matrix(C_strain);
    CHECK(std::abs(C_stress[0] - C_strain[0]) > 1.0, "PE: plane stress != plane strain C matrix");

    // Test 13: Tri3 stiffness diagonal positive
    Real Kt[36];
    t3.stiffness_matrix(Kt);
    bool dp = true;
    for (int i = 0; i < 6; ++i) if (Kt[i*6+i] <= 0) dp = false;
    CHECK(dp, "PE_T3: stiffness diagonal positive");
}

// ============================================================================
// 6. AxisymmetricElement Tests
// ============================================================================
void test_axisymmetric_element() {
    printf("\n--- AxisymmetricElement Tests ---\n");

    // Quad4 in r-z plane: annular ring r in [1,2], z in [0,1]
    Real coords[8] = {1,0, 2,0, 2,1, 1,1};
    Real E = 200000, nu = 0.3;
    AxisymmetricElement elem(coords, 4, E, nu);

    // Test 1: Node count
    CHECK(elem.num_nodes() == 4, "AXI_Q4: num_nodes == 4");

    // Test 2: DOF per node
    CHECK(elem.dof_per_node() == 2, "AXI_Q4: dof_per_node == 2");

    // Test 3: Integration points
    CHECK(elem.num_integration_points() == 4, "AXI_Q4: num_integration_points == 4");

    // Test 4: Shape function PU
    Real N[4];
    elem.shape_functions(0.3, -0.2, N);
    Real sum = 0; for (int i = 0; i < 4; ++i) sum += N[i];
    CHECK_NEAR(sum, 1.0, 1e-12, "AXI_Q4: shape function PU");

    // Test 5: Radius at center = 1.5
    Real r_center = elem.radius_at(0.0, 0.0);
    CHECK_NEAR(r_center, 1.5, 1e-12, "AXI_Q4: radius at center = 1.5");

    // Test 6: Stiffness symmetry
    Real K[64];
    elem.stiffness_matrix(K);
    bool sym = true;
    for (int i = 0; i < 8 && sym; ++i)
        for (int j = i+1; j < 8 && sym; ++j)
            if (std::abs(K[i*8+j] - K[j*8+i]) > 1e-4*(std::abs(K[i*8+j])+1e-30))
                sym = false;
    CHECK(sym, "AXI_Q4: stiffness symmetry");

    // Test 7: Diagonal positive
    bool dp = true;
    for (int i = 0; i < 8; ++i) if (K[i*8+i] <= 0) dp = false;
    CHECK(dp, "AXI_Q4: stiffness diagonal positive");

    // Test 8: B-matrix has hoop strain row
    Real B[4*8];
    elem.strain_displacement_matrix(0.0, 0.0, B);
    // Row 2 is hoop strain (ett = N_a/r * u_r) - should have non-zero entries
    bool hoop = false;
    for (int j = 0; j < 8; ++j) if (std::abs(B[2*8+j]) > 1e-15) hoop = true;
    CHECK(hoop, "AXI_Q4: B-matrix has hoop strain entries");

    // Test 9: Constitutive matrix (4x4)
    Real C[16];
    elem.constitutive_matrix(C);
    CHECK(C[0] > 0 && C[3*4+3] > 0, "AXI_Q4: constitutive matrix diagonal positive");

    // Test 10: Tri3 axisymmetric
    Real tri_coords[6] = {1,0, 2,0, 1.5,1};
    AxisymmetricElement tri(tri_coords, 3, E, nu);
    CHECK(tri.num_nodes() == 3, "AXI_T3: num_nodes == 3");
    CHECK(tri.num_integration_points() == 1, "AXI_T3: num_integration_points == 1");
}

// ============================================================================
// 7. ConnectorElement Tests
// ============================================================================
void test_connector_element() {
    printf("\n--- ConnectorElement Tests ---\n");

    // Two nodes along x-axis, separated by 0.1
    Real coords[6] = {0,0,0, 0.1,0,0};
    Real Kt[3] = {1e6, 1e6, 1e6};  // translational stiffness
    Real Kr[3] = {1e4, 1e4, 1e4};  // rotational stiffness
    ConnectorElement elem(coords, Kt, Kr);

    // Test 1: Node count
    CHECK(elem.num_nodes() == 2, "CONN: num_nodes == 2");

    // Test 2: DOF per node
    CHECK(elem.dof_per_node() == 6, "CONN: dof_per_node == 6");

    // Test 3: Length
    CHECK_NEAR(elem.length(), 0.1, 1e-12, "CONN: length == 0.1");

    // Test 4: Not failed initially
    CHECK(!elem.failed(), "CONN: not failed initially");

    // Test 5: Shape function partition of unity
    Real N[2];
    elem.shape_functions(0.3, N);
    CHECK_NEAR(N[0]+N[1], 1.0, 1e-12, "CONN: shape function PU");

    // Test 6: Shape function at node 0
    elem.shape_functions(-1.0, N);
    CHECK_NEAR(N[0], 1.0, 1e-12, "CONN: N0 at xi=-1");

    // Test 7: Stiffness matrix symmetry
    Real K[144];
    elem.stiffness_matrix(K);
    bool sym = true;
    for (int i = 0; i < 12 && sym; ++i)
        for (int j = i+1; j < 12 && sym; ++j)
            if (std::abs(K[i*12+j] - K[j*12+i]) > 1e-4*(std::abs(K[i*12+j])+1e-30))
                sym = false;
    CHECK(sym, "CONN: stiffness matrix symmetry");

    // Test 8: Zero displacement -> zero force
    Real u_zero[12] = {};
    Real fint[12];
    elem.compute_internal_forces(u_zero, fint);
    Real fn = 0; for (int i = 0; i < 12; ++i) fn += fint[i]*fint[i];
    CHECK(fn < 1e-20, "CONN: zero displacement -> zero force");

    // Test 9: Axial displacement produces force
    Real u_axial[12] = {0,0,0,0,0,0, 0.001,0,0,0,0,0};
    elem.compute_internal_forces(u_axial, fint);
    Real axial_force_sq = fint[0]*fint[0] + fint[6]*fint[6];
    CHECK(axial_force_sq > 0, "CONN: axial displacement -> non-zero force");

    // Test 10: Failure criterion
    ConnectorFailureCriteria fc;
    fc.normal_force_limit = 100.0;
    fc.shear_force_limit = 50.0;
    fc.moment_limit = 10.0;
    elem.set_failure_criteria(fc);

    // Small displacement -> no failure
    Real u_small[12] = {0,0,0,0,0,0, 1e-8,0,0,0,0,0};
    Real fi = elem.check_failure(u_small);
    CHECK(fi < 1.0, "CONN: small displacement -> no failure");
    CHECK(!elem.failed(), "CONN: still not failed after small load");

    // Large displacement -> failure
    elem.reset_failure();
    Real u_large[12] = {0,0,0,0,0,0, 1.0,0,0,0,0,0};
    fi = elem.check_failure(u_large);
    CHECK(fi >= 1.0, "CONN: large displacement -> failure triggered");
    CHECK(elem.failed(), "CONN: failed state set");

    // Test 11: Failed element has zero stiffness
    Real K_fail[144];
    elem.stiffness_matrix(K_fail);
    Real Kf_sum = 0;
    for (int i = 0; i < 144; ++i) Kf_sum += std::abs(K_fail[i]);
    CHECK(Kf_sum < 1e-20, "CONN: failed element zero stiffness");

    // Test 12: Reset failure
    elem.reset_failure();
    CHECK(!elem.failed(), "CONN: failure reset works");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("=== Wave 15 Element Formulations Test ===\n");

    test_thick_shell_8();
    test_thick_shell_6();
    test_dkt_shell();
    test_dkq_shell();
    test_plane_element();
    test_axisymmetric_element();
    test_connector_element();

    printf("\nResults: %d/%d tests passed\n", pass_count, pass_count + fail_count);
    return fail_count > 0 ? 1 : 0;
}
