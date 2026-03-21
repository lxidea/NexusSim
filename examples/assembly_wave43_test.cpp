/**
 * @file assembly_wave43_test.cpp
 * @brief Wave 43: Per-element implicit assemblers + element buffer system (~50 tests)
 *
 * Test groups:
 *  A. Stiffness symmetry  (Hex8, Hex20, Tet4, Tet10, Shell4, Shell3, Beam2, Spring)
 *  B. Positive diagonal   (no constrained BC needed for diagonal check)
 *  C. Analytical Beam2 axial stiffness  (EA/L known exactly)
 *  D. Spring stiffness    (axis-aligned and diagonal)
 *  E. AssemblyDispatcher  (type identity, caching, nullptr for unknown)
 *  F. ElementBuffer       (allocation, accessor, state layout)
 *  G. ElementBufferManager  (allocate, num_elements, copy round-trip, clear)
 *  H. IntegrationPointState  (default values, clear)
 */

#include <nexussim/solver/assembly_wave43.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace nxs;
using namespace nxs::solver;
using nxs::physics::MaterialState;

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool is_symmetric(const std::vector<Real>& K, int n, double tol = 1.0e-8) {
    // Compute global scale (max |K_ij|) for relative tolerance
    double scale = 1.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            scale = std::max(scale, std::abs(K[i*n+j]));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (std::abs(K[i*n+j] - K[j*n+i]) > tol * scale)
                return false;
    return true;
}

static bool all_diagonal_nonneg(const std::vector<Real>& K, int n) {
    // Use relative tolerance: allow entries very slightly negative due to floating point
    double scale = 0.0;
    for (int i = 0; i < n; ++i) scale = std::max(scale, std::abs(K[i*n+i]));
    double tol = scale * 1.0e-8 + 1.0e-20;
    for (int i = 0; i < n; ++i)
        if (K[i*n+i] < -tol)
            return false;
    return true;
}

// Unit cube node coordinates for Hex8
static std::vector<Real> unit_cube_hex8() {
    return {
        0,0,0,  1,0,0,  1,1,0,  0,1,0,
        0,0,1,  1,0,1,  1,1,1,  0,1,1
    };
}

// 2x2x2 cube node coordinates for Hex20 (8 corners + 12 mid-edge)
static std::vector<Real> unit_cube_hex20() {
    // 8 corners (same as Hex8)
    std::vector<Real> c = {
        0,0,0,  1,0,0,  1,1,0,  0,1,0,
        0,0,1,  1,0,1,  1,1,1,  0,1,1,
        // 12 mid-edge nodes (matching Hex20 ordering in assembler)
        0.5,0,0,  1,0.5,0,  0.5,1,0,  0,0.5,0,  // edges on z=0 face
        0.5,0,1,  1,0.5,1,  0.5,1,1,  0,0.5,1,  // edges on z=1 face
        0,0,0.5,  1,0,0.5,  1,1,0.5,  0,1,0.5   // vertical edges
    };
    return c;
}

// Unit tet node coordinates
static std::vector<Real> unit_tet4() {
    return {0,0,0, 1,0,0, 0,1,0, 0,0,1};
}

// Tet10: 4 corners + 6 mid-edge nodes
static std::vector<Real> unit_tet10() {
    return {
        0,0,0,  1,0,0,  0,1,0,  0,0,1,      // corners
        0.5,0,0,  0.5,0.5,0,  0,0.5,0,       // mid-edges on face z=0
        0,0,0.5,  0.5,0,0.5,  0,0.5,0.5      // remaining mid-edges
    };
}

// Unit square shell4 (flat, z=0)
static std::vector<Real> unit_square_shell4() {
    return {0,0,0, 1,0,0, 1,1,0, 0,1,0};
}

// Unit triangle shell3 (flat, z=0)
static std::vector<Real> unit_triangle_shell3() {
    return {0,0,0, 1,0,0, 0,1,0};
}

// Beam along x from (0,0,0) to (L,0,0)
static std::vector<Real> beam_coords(double L) {
    return {0,0,0, L,0,0};
}

// Spring along x
static std::vector<Real> spring_x_coords() {
    return {0,0,0, 1,0,0};
}

// Spring along diagonal (1,1,0)/sqrt(2)
static std::vector<Real> spring_diag_coords() {
    double s = 1.0/std::sqrt(2.0);
    return {0,0,0, s,s,0};
}

// ===========================================================================
// A. Stiffness symmetry tests
// ===========================================================================

static void test_symmetry_hex8() {
    Hex8Assembler asm8;
    auto coords = unit_cube_hex8();
    std::vector<Real> K(24*24, 0.0);
    asm8.assemble_element_stiffness(0, coords.data(), 210e9, 0.3, K.data());
    CHECK(is_symmetric(K, 24, 1e-6), "Hex8 stiffness symmetry");
}

static void test_symmetry_hex20() {
    Hex20Assembler asm20;
    auto coords = unit_cube_hex20();
    std::vector<Real> K(60*60, 0.0);
    asm20.assemble_element_stiffness(0, coords.data(), 210e9, 0.3, K.data());
    CHECK(is_symmetric(K, 60, 1e-6), "Hex20 stiffness symmetry");
}

static void test_symmetry_tet4() {
    Tet4Assembler asm4;
    auto coords = unit_tet4();
    std::vector<Real> K(12*12, 0.0);
    asm4.assemble_element_stiffness(0, coords.data(), 210e9, 0.3, K.data());
    CHECK(is_symmetric(K, 12, 1e-6), "Tet4 stiffness symmetry");
}

static void test_symmetry_tet10() {
    Tet10Assembler asm10;
    auto coords = unit_tet10();
    std::vector<Real> K(30*30, 0.0);
    asm10.assemble_element_stiffness(0, coords.data(), 210e9, 0.3, K.data());
    CHECK(is_symmetric(K, 30, 1e-6), "Tet10 stiffness symmetry");
}

static void test_symmetry_shell4() {
    Shell4Assembler asmS4;
    auto coords = unit_square_shell4();
    std::vector<Real> K(24*24, 0.0);
    asmS4.assemble_element_stiffness(0, coords.data(), 210e9, 0.3, K.data());
    CHECK(is_symmetric(K, 24, 1e-5), "Shell4 stiffness symmetry");
}

static void test_symmetry_shell3() {
    Shell3Assembler asmS3;
    auto coords = unit_triangle_shell3();
    std::vector<Real> K(18*18, 0.0);
    asmS3.assemble_element_stiffness(0, coords.data(), 210e9, 0.3, K.data());
    CHECK(is_symmetric(K, 18, 1e-5), "Shell3 stiffness symmetry");
}

static void test_symmetry_beam2() {
    Beam2Assembler asmB;
    auto coords = beam_coords(1.0);
    std::vector<Real> K(12*12, 0.0);
    asmB.assemble_element_stiffness(0, coords.data(), 210e9, 0.3, K.data());
    CHECK(is_symmetric(K, 12, 1e-8), "Beam2 stiffness symmetry");
}

static void test_symmetry_spring() {
    SpringAssembler asmSp;
    auto coords = spring_x_coords();
    std::vector<Real> K(6*6, 0.0);
    asmSp.assemble_element_stiffness(0, coords.data(), 1000.0, 0.0, K.data());
    CHECK(is_symmetric(K, 6, 1e-12), "Spring stiffness symmetry");
}

// ===========================================================================
// B. Positive diagonal tests
// ===========================================================================

static void test_diag_nonneg_hex8() {
    Hex8Assembler asm8;
    auto coords = unit_cube_hex8();
    std::vector<Real> K(24*24, 0.0);
    asm8.assemble_element_stiffness(0, coords.data(), 210e9, 0.3, K.data());
    CHECK(all_diagonal_nonneg(K, 24), "Hex8 non-negative diagonal");
}

static void test_diag_nonneg_tet4() {
    Tet4Assembler asm4;
    auto coords = unit_tet4();
    std::vector<Real> K(12*12, 0.0);
    asm4.assemble_element_stiffness(0, coords.data(), 210e9, 0.3, K.data());
    CHECK(all_diagonal_nonneg(K, 12), "Tet4 non-negative diagonal");
}

static void test_diag_nonneg_tet10() {
    Tet10Assembler asm10;
    auto coords = unit_tet10();
    std::vector<Real> K(30*30, 0.0);
    asm10.assemble_element_stiffness(0, coords.data(), 210e9, 0.3, K.data());
    CHECK(all_diagonal_nonneg(K, 30), "Tet10 non-negative diagonal");
}

static void test_diag_nonneg_shell4() {
    Shell4Assembler asmS4;
    auto coords = unit_square_shell4();
    std::vector<Real> K(24*24, 0.0);
    asmS4.assemble_element_stiffness(0, coords.data(), 210e9, 0.3, K.data());
    CHECK(all_diagonal_nonneg(K, 24), "Shell4 non-negative diagonal");
}

static void test_diag_nonneg_beam2() {
    Beam2Assembler asmB;
    auto coords = beam_coords(1.0);
    std::vector<Real> K(12*12, 0.0);
    asmB.assemble_element_stiffness(0, coords.data(), 210e9, 0.3, K.data());
    CHECK(all_diagonal_nonneg(K, 12), "Beam2 non-negative diagonal");
}

// ===========================================================================
// C. Analytical Beam2 tests
// ===========================================================================

static void test_beam2_axial() {
    Beam2Assembler asmB;
    double E = 210.0e9, L = 2.0;
    double A = 1.0;
    asmB.A = A;
    auto coords = beam_coords(L);
    std::vector<Real> K(12*12, 0.0);
    asmB.assemble_element_stiffness(0, coords.data(), E, 0.3, K.data());
    double EA_L = E * A / L;
    // K[0,0] = EA/L
    CHECK_NEAR(K[0*12+0], EA_L, 1.0, "Beam2 axial K[0,0] = EA/L");
    // K[0,6] = -EA/L
    CHECK_NEAR(K[0*12+6], -EA_L, 1.0, "Beam2 axial K[0,6] = -EA/L");
    // K[6,6] = EA/L
    CHECK_NEAR(K[6*12+6], EA_L, 1.0, "Beam2 axial K[6,6] = EA/L");
}

static void test_beam2_bending_z() {
    Beam2Assembler asmB;
    double E = 210.0e9, L = 3.0;
    double Iz = 1.0/12.0;
    asmB.Iz = Iz;
    auto coords = beam_coords(L);
    std::vector<Real> K(12*12, 0.0);
    asmB.assemble_element_stiffness(0, coords.data(), E, 0.3, K.data());
    double EI = E * Iz;
    double k12  = 12.0 * EI / (L*L*L);
    double k4EI = 4.0 * EI / L;
    // v DOF = index 1; K[1,1] = 12EI/L^3
    CHECK_NEAR(K[1*12+1], k12, 1e3, "Beam2 bending K[v1,v1] = 12EI/L^3");
    // theta_z DOF = index 5; K[5,5] = 4EI/L
    CHECK_NEAR(K[5*12+5], k4EI, 1e3, "Beam2 bending K[tz1,tz1] = 4EI/L");
}

static void test_beam2_torsion() {
    Beam2Assembler asmB;
    double E = 210.0e9, nu = 0.3;
    double G = E / (2.0*(1.0+nu));
    double J = 1.0/6.0, L = 1.0;
    asmB.J = J;
    auto coords = beam_coords(L);
    std::vector<Real> K(12*12, 0.0);
    asmB.assemble_element_stiffness(0, coords.data(), E, nu, K.data());
    double GJ_L = G * J / L;
    // theta_x DOF = index 3; K[3,3] = GJ/L
    CHECK_NEAR(K[3*12+3], GJ_L, 1.0, "Beam2 torsion K[tx1,tx1] = GJ/L");
    CHECK_NEAR(K[3*12+9], -GJ_L, 1.0, "Beam2 torsion K[tx1,tx2] = -GJ/L");
}

// ===========================================================================
// D. Spring stiffness tests
// ===========================================================================

static void test_spring_x_axis() {
    SpringAssembler asmSp;
    double k = 1000.0;
    auto coords = spring_x_coords();
    std::vector<Real> K(6*6, 0.0);
    asmSp.assemble_element_stiffness(0, coords.data(), k, 0.0, K.data());
    // Along x: K[0,0]=k, K[0,3]=-k, K[3,0]=-k, K[3,3]=k; y,z = 0
    CHECK_NEAR(K[0*6+0],  k, 1e-6, "Spring x-axis K[0,0] = k");
    CHECK_NEAR(K[0*6+3], -k, 1e-6, "Spring x-axis K[0,3] = -k");
    CHECK_NEAR(K[3*6+3],  k, 1e-6, "Spring x-axis K[3,3] = k");
    CHECK_NEAR(K[1*6+1],  0.0, 1e-6, "Spring x-axis K[y,y] = 0");
    CHECK_NEAR(K[2*6+2],  0.0, 1e-6, "Spring x-axis K[z,z] = 0");
}

static void test_spring_diagonal() {
    SpringAssembler asmSp;
    double k = 500.0;
    auto coords = spring_diag_coords(); // along (1,1,0)/sqrt(2)
    std::vector<Real> K(6*6, 0.0);
    asmSp.assemble_element_stiffness(0, coords.data(), k, 0.0, K.data());
    // Direction cosines l = m = 1/sqrt(2), n = 0
    double l2 = 0.5;
    CHECK_NEAR(K[0*6+0], k*l2, 1e-6, "Spring diagonal K[x,x] = k/2");
    CHECK_NEAR(K[0*6+1], k*l2, 1e-6, "Spring diagonal K[x,y] = k/2");
    CHECK_NEAR(K[1*6+1], k*l2, 1e-6, "Spring diagonal K[y,y] = k/2");
    CHECK_NEAR(K[2*6+2],  0.0, 1e-6, "Spring diagonal K[z,z] = 0");
}

static void test_spring_symmetry_diagonal() {
    SpringAssembler asmSp;
    auto coords = spring_diag_coords();
    std::vector<Real> K(6*6, 0.0);
    asmSp.assemble_element_stiffness(0, coords.data(), 800.0, 0.0, K.data());
    CHECK(is_symmetric(K, 6, 1e-12), "Spring diagonal stiffness symmetry");
}

// ===========================================================================
// E. AssemblyDispatcher tests
// ===========================================================================

static void test_dispatcher_returns_correct_type() {
    AssemblyDispatcher disp;

    auto* h8 = disp.get_assembler(nxs::ElementType::Hex8);
    CHECK(h8 != nullptr, "Dispatcher: Hex8 not null");
    CHECK(h8->element_type() == nxs::ElementType::Hex8, "Dispatcher: Hex8 type match");
    CHECK(h8->ndof() == 24, "Dispatcher: Hex8 ndof = 24");

    auto* h20 = disp.get_assembler(nxs::ElementType::Hex20);
    CHECK(h20 != nullptr, "Dispatcher: Hex20 not null");
    CHECK(h20->ndof() == 60, "Dispatcher: Hex20 ndof = 60");

    auto* t4 = disp.get_assembler(nxs::ElementType::Tet4);
    CHECK(t4 != nullptr, "Dispatcher: Tet4 not null");
    CHECK(t4->ndof() == 12, "Dispatcher: Tet4 ndof = 12");

    auto* t10 = disp.get_assembler(nxs::ElementType::Tet10);
    CHECK(t10 != nullptr, "Dispatcher: Tet10 not null");
    CHECK(t10->ndof() == 30, "Dispatcher: Tet10 ndof = 30");

    auto* s4 = disp.get_assembler(nxs::ElementType::Shell4);
    CHECK(s4 != nullptr, "Dispatcher: Shell4 not null");
    CHECK(s4->ndof() == 24, "Dispatcher: Shell4 ndof = 24");

    auto* s3 = disp.get_assembler(nxs::ElementType::Shell3);
    CHECK(s3 != nullptr, "Dispatcher: Shell3 not null");
    CHECK(s3->ndof() == 18, "Dispatcher: Shell3 ndof = 18");

    auto* b2 = disp.get_assembler(nxs::ElementType::Beam2);
    CHECK(b2 != nullptr, "Dispatcher: Beam2 not null");
    CHECK(b2->ndof() == 12, "Dispatcher: Beam2 ndof = 12");

    auto* sp = disp.get_assembler(nxs::ElementType::Spring);
    CHECK(sp != nullptr, "Dispatcher: Spring not null");
    CHECK(sp->ndof() == 6, "Dispatcher: Spring ndof = 6");
}

static void test_dispatcher_caching() {
    AssemblyDispatcher disp;
    auto* first  = disp.get_assembler(nxs::ElementType::Tet4);
    auto* second = disp.get_assembler(nxs::ElementType::Tet4);
    CHECK(first == second, "Dispatcher: same pointer on second call (cached)");
}

static void test_dispatcher_unknown_returns_nullptr() {
    AssemblyDispatcher disp;
    // Wedge6 not implemented — should return nullptr
    auto* w = disp.get_assembler(nxs::ElementType::Wedge6);
    CHECK(w == nullptr, "Dispatcher: Wedge6 returns nullptr");
}

// ===========================================================================
// F. ElementBuffer tests
// ===========================================================================

static void test_element_buffer_default_construct() {
    ElementBuffer buf;
    CHECK(buf.num_integration_points == 0, "ElementBuffer default num_ip = 0");
    CHECK(buf.num_layers == 1, "ElementBuffer default num_layers = 1");
    CHECK(buf.ip_states.empty(), "ElementBuffer default ip_states empty");
}

static void test_element_buffer_allocate() {
    ElementBuffer buf(8, nxs::ElementType::Hex8, 1);
    CHECK(buf.num_integration_points == 8, "ElementBuffer hex8 num_ip = 8");
    CHECK(buf.num_layers == 1, "ElementBuffer hex8 num_layers = 1");
    CHECK(buf.total_states() == 8, "ElementBuffer hex8 total_states = 8");
}

static void test_element_buffer_accessor() {
    ElementBuffer buf(4, nxs::ElementType::Shell4, 3);
    CHECK(buf.total_states() == 12, "Shell4 3-layer total = 12 states");

    // Write to a specific IP/layer
    buf.state(2, 1).stress[0] = 1234.5;
    CHECK_NEAR(buf.state(2, 1).stress[0], 1234.5, 1e-9, "ElementBuffer accessor write/read");
}

static void test_element_buffer_clear() {
    ElementBuffer buf(2, nxs::ElementType::Tet4, 1);
    buf.state(0).plastic_strain = 0.05;
    buf.state(1).temperature = 500.0;
    buf.clear();
    CHECK_NEAR(buf.state(0).plastic_strain, 0.0, 1e-12, "ElementBuffer clear plastic_strain");
    CHECK_NEAR(buf.state(1).temperature, 293.15, 1e-6, "ElementBuffer clear temperature reset");
}

static void test_element_buffer_composite_layers() {
    ElementBuffer buf(2, nxs::ElementType::Shell4, 5);
    CHECK(buf.num_layers == 5, "Shell4 5-layer num_layers");
    CHECK(buf.total_states() == 10, "Shell4 5-layer total_states = 10");
    buf.state(1, 4).damage = 0.3;
    CHECK_NEAR(buf.state(1, 4).damage, 0.3, 1e-12, "Composite layer damage access");
}

// ===========================================================================
// G. ElementBufferManager tests
// ===========================================================================

static void test_manager_allocate_hex8() {
    ElementBufferManager mgr;
    mgr.allocate(10, nxs::ElementType::Hex8);
    CHECK(mgr.num_elements() == 10, "Manager: allocate 10 Hex8 elements");
    CHECK(mgr.element_type() == nxs::ElementType::Hex8, "Manager: element_type == Hex8");
    CHECK(mgr.get_buffer(0).num_integration_points == 8, "Manager: Hex8 default 8 IPs");
    CHECK(mgr.get_buffer(9).num_integration_points == 8, "Manager: last element 8 IPs");
}

static void test_manager_allocate_custom_ip() {
    ElementBufferManager mgr;
    mgr.allocate(5, nxs::ElementType::Tet4, 4, 1); // 4 IPs override
    CHECK(mgr.get_buffer(0).num_integration_points == 4, "Manager: custom num_ip override");
}

static void test_manager_allocate_tet4_default_ip() {
    ElementBufferManager mgr;
    mgr.allocate(3, nxs::ElementType::Tet4);
    CHECK(mgr.get_buffer(0).num_integration_points == 1, "Manager: Tet4 default 1 IP");
}

static void test_manager_copy_round_trip() {
    ElementBufferManager mgr;
    mgr.allocate(2, nxs::ElementType::Tet4, 1, 1);

    // Set state in buffer
    ElementBuffer& buf = mgr.get_buffer(0);
    IntegrationPointState& s = buf.state(0);
    s.stress[0] = 1.0e6;
    s.stress[3] = 5.0e5;
    s.strain[1] = 0.001;
    s.history[7] = 42.0;
    s.plastic_strain = 0.02;
    s.damage = 0.1;
    s.temperature = 350.0;

    // Copy to MaterialState
    MaterialState ms;
    mgr.copy_to_material_state(0, 0, ms);

    CHECK_NEAR(ms.stress[0], 1.0e6, 1.0, "copy_to_material_state stress[0]");
    CHECK_NEAR(ms.stress[3], 5.0e5, 1.0, "copy_to_material_state stress[3]");
    CHECK_NEAR(ms.strain[1], 0.001, 1e-9, "copy_to_material_state strain[1]");
    CHECK_NEAR(ms.history[7], 42.0, 1e-9, "copy_to_material_state history[7]");
    CHECK_NEAR(ms.plastic_strain, 0.02, 1e-9, "copy_to_material_state plastic_strain");
    CHECK_NEAR(ms.damage, 0.1, 1e-9, "copy_to_material_state damage");
    CHECK_NEAR(ms.temperature, 350.0, 1e-9, "copy_to_material_state temperature");

    // Modify MaterialState and copy back
    ms.stress[0] = 2.0e6;
    ms.plastic_strain = 0.05;
    ms.history[10] = 99.0;
    mgr.copy_from_material_state(0, 0, ms);

    CHECK_NEAR(mgr.get_buffer(0).state(0).stress[0], 2.0e6, 1.0, "copy_from_material_state stress[0]");
    CHECK_NEAR(mgr.get_buffer(0).state(0).plastic_strain, 0.05, 1e-9, "copy_from_material_state plastic_strain");
    CHECK_NEAR(mgr.get_buffer(0).state(0).history[10], 99.0, 1e-9, "copy_from_material_state history[10]");
}

static void test_manager_clear() {
    ElementBufferManager mgr;
    mgr.allocate(3, nxs::ElementType::Hex8, 8, 1);
    mgr.get_buffer(1).state(3).stress[2] = 999.0;
    mgr.get_buffer(2).state(7).damage = 0.9;
    mgr.clear();
    CHECK_NEAR(mgr.get_buffer(1).state(3).stress[2], 0.0, 1e-12, "Manager clear stress");
    CHECK_NEAR(mgr.get_buffer(2).state(7).damage, 0.0, 1e-12, "Manager clear damage");
}

static void test_manager_reallocate() {
    ElementBufferManager mgr;
    mgr.allocate(5, nxs::ElementType::Hex8);
    CHECK(mgr.num_elements() == 5, "Manager: initial 5 elements");
    mgr.allocate(12, nxs::ElementType::Tet4);
    CHECK(mgr.num_elements() == 12, "Manager: reallocated 12 elements");
    CHECK(mgr.element_type() == nxs::ElementType::Tet4, "Manager: element_type after reallocate");
}

// ===========================================================================
// H. IntegrationPointState default value tests
// ===========================================================================

static void test_ip_state_defaults() {
    IntegrationPointState s;
    for (int i = 0; i < 6; ++i) {
        CHECK_NEAR(s.stress[i], 0.0, 1e-15, "IPState default stress = 0");
        CHECK_NEAR(s.strain[i], 0.0, 1e-15, "IPState default strain = 0");
    }
    for (int i = 0; i < 64; ++i)
        CHECK_NEAR(s.history[i], 0.0, 1e-15, "IPState default history = 0");
    CHECK_NEAR(s.plastic_strain, 0.0, 1e-15, "IPState default plastic_strain = 0");
    CHECK_NEAR(s.damage, 0.0, 1e-15, "IPState default damage = 0");
    CHECK_NEAR(s.temperature, 293.15, 1e-6, "IPState default temperature = 293.15");
}

static void test_ip_state_clear() {
    IntegrationPointState s;
    s.stress[0] = 1e9;
    s.strain[5] = 0.1;
    s.history[63] = -1.0;
    s.plastic_strain = 5.0;
    s.damage = 0.7;
    s.temperature = 600.0;
    s.clear();
    CHECK_NEAR(s.stress[0], 0.0, 1e-12, "IPState clear stress");
    CHECK_NEAR(s.strain[5], 0.0, 1e-12, "IPState clear strain");
    CHECK_NEAR(s.history[63], 0.0, 1e-12, "IPState clear history");
    CHECK_NEAR(s.plastic_strain, 0.0, 1e-12, "IPState clear plastic_strain");
    CHECK_NEAR(s.damage, 0.0, 1e-12, "IPState clear damage");
    CHECK_NEAR(s.temperature, 293.15, 1e-6, "IPState clear temperature reset");
}

// ===========================================================================
// Extra: Tet4 known volume test
// ===========================================================================

static void test_tet4_stiffness_scale() {
    // Unit tet: same result under uniform scaling
    Tet4Assembler asm4;
    std::vector<Real> K1(12*12, 0.0);
    auto c1 = unit_tet4();
    asm4.assemble_element_stiffness(0, c1.data(), 1.0, 0.0, K1.data());

    // Scaled tet (coords * 2) — volume * 8, but B scales 1/L so B^T*D*B scales 1/L^2.
    // K = V * B^T*D*B: scales as L^3 * L^{-2} = L. So K(2x) = 2 * K(1x).
    std::vector<Real> c2(12);
    for (int i = 0; i < 12; ++i) c2[i] = c1[i] * 2.0;
    std::vector<Real> K2(12*12, 0.0);
    asm4.assemble_element_stiffness(0, c2.data(), 1.0, 0.0, K2.data());

    CHECK_NEAR(K2[0*12+0], 2.0*K1[0*12+0], std::abs(K1[0*12+0])*1e-6,
               "Tet4 stiffness scales linearly with element size");
}

// ===========================================================================
// main
// ===========================================================================

int main() {
    std::cout << "=== Wave 43: Assembly Assemblers + Element Buffer Tests ===\n\n";

    // A. Symmetry
    std::cout << "--- A. Stiffness Symmetry ---\n";
    test_symmetry_hex8();
    test_symmetry_hex20();
    test_symmetry_tet4();
    test_symmetry_tet10();
    test_symmetry_shell4();
    test_symmetry_shell3();
    test_symmetry_beam2();
    test_symmetry_spring();

    // B. Non-negative diagonal
    std::cout << "--- B. Non-negative Diagonal ---\n";
    test_diag_nonneg_hex8();
    test_diag_nonneg_tet4();
    test_diag_nonneg_tet10();
    test_diag_nonneg_shell4();
    test_diag_nonneg_beam2();

    // C. Analytical Beam2
    std::cout << "--- C. Analytical Beam2 ---\n";
    test_beam2_axial();
    test_beam2_bending_z();
    test_beam2_torsion();

    // D. Spring
    std::cout << "--- D. Spring Stiffness ---\n";
    test_spring_x_axis();
    test_spring_diagonal();
    test_spring_symmetry_diagonal();

    // E. Dispatcher
    std::cout << "--- E. AssemblyDispatcher ---\n";
    test_dispatcher_returns_correct_type();
    test_dispatcher_caching();
    test_dispatcher_unknown_returns_nullptr();

    // F. ElementBuffer
    std::cout << "--- F. ElementBuffer ---\n";
    test_element_buffer_default_construct();
    test_element_buffer_allocate();
    test_element_buffer_accessor();
    test_element_buffer_clear();
    test_element_buffer_composite_layers();

    // G. ElementBufferManager
    std::cout << "--- G. ElementBufferManager ---\n";
    test_manager_allocate_hex8();
    test_manager_allocate_custom_ip();
    test_manager_allocate_tet4_default_ip();
    test_manager_copy_round_trip();
    test_manager_clear();
    test_manager_reallocate();

    // H. IntegrationPointState
    std::cout << "--- H. IntegrationPointState ---\n";
    test_ip_state_defaults();
    test_ip_state_clear();

    // Extra
    std::cout << "--- Extra: Tet4 scaling ---\n";
    test_tet4_stiffness_scale();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return (tests_failed > 0) ? 1 : 0;
}
