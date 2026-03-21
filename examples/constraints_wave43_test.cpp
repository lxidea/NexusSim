/**
 * @file constraints_wave43_test.cpp
 * @brief Tests for Wave 43: FXBODY superelement, RLINK rigid link, GroupSet algebra
 *
 * ~45 tests covering:
 *   - SuperelementData construction and matrix layout
 *   - FXBODYConstraint stiffness/mass assembly, force recovery, internal recovery
 *   - RLINKConstraint Type0/1/2/10 velocity enforcement and constraint forces
 *   - GroupSet add/remove/contains, union/intersect/difference/complement
 *   - GroupSetManager named sets and expression evaluation
 */

#include <nexussim/fem/constraints_wave43.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>

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
// FXBODY Tests
// ============================================================================

static void test_superelement_construction() {
    using namespace nxs::fem;

    SuperelementData se(1, "SE_A", 2, 3);

    CHECK(se.id == 1, "SuperelementData id");
    CHECK(se.name == "SE_A", "SuperelementData name");
    CHECK(se.num_interface_dofs == 2, "num_interface_dofs");
    CHECK(se.num_internal_modes == 3, "num_internal_modes");
    CHECK(se.K_reduced.size() == 4u, "K_reduced size = N^2 = 4");
    CHECK(se.M_reduced.size() == 4u, "M_reduced size = N^2 = 4");
    CHECK(se.recovery_matrix.size() == 6u, "recovery_matrix size = N_int * N_intf = 6");
    CHECK(se.interface_dof_ids.size() == 2u, "interface_dof_ids size");
}

static void test_fxbody_stiffness_assembly() {
    using namespace nxs::fem;

    // 2-DOF superelement with K_red = [[10, -10], [-10, 10]]
    SuperelementData se(1, "SE_K", 2, 0);
    se.interface_dof_ids = {0, 1};  // global DOFs 0 and 1
    se.K_reduced = {10.0, -10.0,
                    -10.0, 10.0};

    FXBODYConstraint fxb(std::move(se));

    // Assemble into 2-DOF system
    const nxs::Index N_global = 2;
    std::vector<nxs::Real> K_diag(N_global, 0.0);
    std::vector<nxs::Real> K_offdiag(N_global * N_global, 0.0);

    fxb.assemble_stiffness(K_diag.data(), K_offdiag.data(), N_global);

    CHECK_NEAR(K_diag[0], 10.0, 1e-12, "K_diag[0] = K_red(0,0)");
    CHECK_NEAR(K_diag[1], 10.0, 1e-12, "K_diag[1] = K_red(1,1)");
    CHECK_NEAR(K_offdiag[0*2+1], -10.0, 1e-12, "K_offdiag[0,1] = K_red(0,1)");
    CHECK_NEAR(K_offdiag[1*2+0], -10.0, 1e-12, "K_offdiag[1,0] = K_red(1,0)");
}

static void test_fxbody_stiffness_accumulation() {
    using namespace nxs::fem;

    // Two superelements sharing DOF 1; check that entries accumulate
    SuperelementData se1(1, "SE1", 2, 0);
    se1.interface_dof_ids = {0, 1};
    se1.K_reduced = {5.0, -5.0, -5.0, 5.0};

    SuperelementData se2(2, "SE2", 2, 0);
    se2.interface_dof_ids = {1, 2};
    se2.K_reduced = {3.0, -3.0, -3.0, 3.0};

    FXBODYConstraint fxb1(std::move(se1));
    FXBODYConstraint fxb2(std::move(se2));

    const nxs::Index NG = 3;
    std::vector<nxs::Real> K_diag(NG, 0.0);
    std::vector<nxs::Real> K_od(NG * NG, 0.0);

    fxb1.assemble_stiffness(K_diag.data(), K_od.data(), NG);
    fxb2.assemble_stiffness(K_diag.data(), K_od.data(), NG);

    CHECK_NEAR(K_diag[0], 5.0, 1e-12, "Accumulated K_diag[0]");
    CHECK_NEAR(K_diag[1], 8.0, 1e-12, "Accumulated K_diag[1] = 5+3");
    CHECK_NEAR(K_diag[2], 3.0, 1e-12, "Accumulated K_diag[2]");
}

static void test_fxbody_mass_assembly() {
    using namespace nxs::fem;

    SuperelementData se(2, "SE_M", 2, 0);
    se.interface_dof_ids = {0, 1};
    se.M_reduced = {2.0, 0.5, 0.5, 2.0};

    FXBODYConstraint fxb(std::move(se));

    const nxs::Index NG = 2;
    std::vector<nxs::Real> M_diag(NG, 0.0);
    std::vector<nxs::Real> M_od(NG * NG, 0.0);

    fxb.assemble_mass(M_diag.data(), M_od.data(), NG);

    CHECK_NEAR(M_diag[0], 2.0, 1e-12, "M_diag[0]");
    CHECK_NEAR(M_diag[1], 2.0, 1e-12, "M_diag[1]");
    CHECK_NEAR(M_od[0*2+1], 0.5, 1e-12, "M_offdiag[0,1]");
    CHECK_NEAR(M_od[1*2+0], 0.5, 1e-12, "M_offdiag[1,0]");
}

static void test_fxbody_interface_forces() {
    using namespace nxs::fem;

    // K_red = [[10, -10], [-10, 10]], u = {1, 0}  -> f = {10, -10}
    SuperelementData se(3, "SE_F", 2, 0);
    se.K_reduced = {10.0, -10.0, -10.0, 10.0};

    FXBODYConstraint fxb(std::move(se));

    std::vector<nxs::Real> u = {1.0, 0.0};
    auto f = fxb.compute_interface_forces(u);

    CHECK_NEAR(f[0], 10.0, 1e-12, "interface force[0]");
    CHECK_NEAR(f[1], -10.0, 1e-12, "interface force[1]");

    // u = {1, 1} -> f = {0, 0} (rigid body mode)
    u = {1.0, 1.0};
    f = fxb.compute_interface_forces(u);
    CHECK_NEAR(f[0], 0.0, 1e-12, "interface force[0] rigid mode");
    CHECK_NEAR(f[1], 0.0, 1e-12, "interface force[1] rigid mode");
}

static void test_fxbody_internal_recovery() {
    using namespace nxs::fem;

    // 2-interface, 3-internal modes
    // recovery_matrix = [[1, 0], [0, 1], [1, 1]]
    // u_intf = {2, 3}
    // q_internal = {2, 3, 5}
    SuperelementData se(4, "SE_R", 2, 3);
    se.recovery_matrix = {1.0, 0.0,
                           0.0, 1.0,
                           1.0, 1.0};

    FXBODYConstraint fxb(std::move(se));

    std::vector<nxs::Real> u_intf = {2.0, 3.0};
    auto q = fxb.recover_internal(u_intf);

    CHECK(q.size() == 3u, "recovered internal size == num_internal_modes");
    CHECK_NEAR(q[0], 2.0, 1e-12, "q[0] = 1*2 + 0*3");
    CHECK_NEAR(q[1], 3.0, 1e-12, "q[1] = 0*2 + 1*3");
    CHECK_NEAR(q[2], 5.0, 1e-12, "q[2] = 1*2 + 1*3");
}

static void test_fxbody_diagonal_only_assembly() {
    using namespace nxs::fem;

    // Pass nullptr for off-diagonal — should not crash
    SuperelementData se(5, "SE_diag", 2, 0);
    se.interface_dof_ids = {0, 1};
    se.K_reduced = {4.0, -2.0, -2.0, 4.0};

    FXBODYConstraint fxb(std::move(se));

    std::vector<nxs::Real> K_diag(2, 0.0);
    fxb.assemble_stiffness(K_diag.data(), nullptr, 2);

    CHECK_NEAR(K_diag[0], 4.0, 1e-12, "diagonal-only K_diag[0]");
    CHECK_NEAR(K_diag[1], 4.0, 1e-12, "diagonal-only K_diag[1]");
}

// ============================================================================
// RLINK Tests
// ============================================================================

static void test_rlink_type0_translation() {
    using namespace nxs::fem;

    // Master node 0 at origin, slave node 1 at (1,0,0)
    // Master translational velocity = (2, 0, 0), no rotation
    const int N = 2;
    std::vector<nxs::Real> vel(6 * N, 0.0);
    std::vector<nxs::Real> pos(3 * N, 0.0);

    pos[3*0+0] = 0.0; pos[3*0+1] = 0.0; pos[3*0+2] = 0.0;
    pos[3*1+0] = 1.0; pos[3*1+1] = 0.0; pos[3*1+2] = 0.0;

    vel[6*0+0] = 2.0;  // master vx
    vel[6*1+0] = 0.0;  // slave vx before constraint

    RLINKConstraint rl(0, {1}, RLINKType::Type0);
    rl.apply_velocity_constraint(vel.data(), pos.data(), N);

    CHECK_NEAR(vel[6*1+0], 2.0, 1e-12, "Type0: slave vx matches master (no rotation)");
    CHECK_NEAR(vel[6*1+1], 0.0, 1e-12, "Type0: slave vy = 0");
    CHECK_NEAR(vel[6*1+2], 0.0, 1e-12, "Type0: slave vz = 0");
}

static void test_rlink_type0_with_rotation() {
    using namespace nxs::fem;

    // Master at origin, omega_z = 1 rad/s, slave at (0,1,0)
    // v_slave = v_master + omega x r = (0,0,0) + (0,0,1) x (0,1,0) = (-1,0,0)
    const int N = 2;
    std::vector<nxs::Real> vel(6 * N, 0.0);
    std::vector<nxs::Real> pos(3 * N, 0.0);

    pos[3*1+1] = 1.0;   // slave at y=1
    vel[6*0+5] = 1.0;   // master omega_z = 1

    RLINKConstraint rl(0, {1}, RLINKType::Type0);
    rl.apply_velocity_constraint(vel.data(), pos.data(), N);

    // omega x r = (0,0,1) x (0,1,0) = (0*0-1*1, 1*0-0*0, 0*1-0*0) = (-1, 0, 0)
    CHECK_NEAR(vel[6*1+0], -1.0, 1e-12, "Type0 omega x r: slave vx = -1");
    CHECK_NEAR(vel[6*1+1],  0.0, 1e-12, "Type0 omega x r: slave vy = 0");
    CHECK_NEAR(vel[6*1+2],  0.0, 1e-12, "Type0 omega x r: slave vz = 0");
    // Rotational DOFs coupled
    CHECK_NEAR(vel[6*1+5], 1.0, 1e-12, "Type0: slave omega_z = master omega_z");
}

static void test_rlink_type1_translation_only() {
    using namespace nxs::fem;

    // Type1 should copy only translational velocity
    const int N = 2;
    std::vector<nxs::Real> vel(6 * N, 0.0);
    std::vector<nxs::Real> pos(3 * N, 0.0);

    vel[6*0+0] = 5.0;   // master vx
    vel[6*0+3] = 3.0;   // master omega_x (should NOT propagate for Type1)
    vel[6*1+3] = 7.0;   // slave omega_x (should remain unchanged)

    RLINKConstraint rl(0, {1}, RLINKType::Type1);
    rl.apply_velocity_constraint(vel.data(), pos.data(), N);

    CHECK_NEAR(vel[6*1+0], 5.0, 1e-12, "Type1: slave vx set to master");
    CHECK_NEAR(vel[6*1+3], 7.0, 1e-12, "Type1: slave omega_x unchanged");
}

static void test_rlink_type10_spherical() {
    using namespace nxs::fem;

    // Type10 same as Type1 — shared translation, free rotation
    const int N = 2;
    std::vector<nxs::Real> vel(6 * N, 0.0);
    std::vector<nxs::Real> pos(3 * N, 0.0);

    vel[6*0+1] = 4.0;   // master vy
    vel[6*0+4] = 2.0;   // master omega_y
    vel[6*1+4] = 9.0;   // slave omega_y (should stay free)

    RLINKConstraint rl(0, {1}, RLINKType::Type10);
    rl.apply_velocity_constraint(vel.data(), pos.data(), N);

    CHECK_NEAR(vel[6*1+1], 4.0, 1e-12, "Type10: slave vy = master vy");
    CHECK_NEAR(vel[6*1+4], 9.0, 1e-12, "Type10: slave omega_y free");
}

static void test_rlink_type2_selected_dofs() {
    using namespace nxs::fem;

    // Type2: release DOF 1 (uy), constrain rest
    const int N = 2;
    std::vector<nxs::Real> vel(6 * N, 0.0);
    std::vector<nxs::Real> pos(3 * N, 0.0);

    vel[6*0+0] = 1.0;  // master vx
    vel[6*0+1] = 2.0;  // master vy
    vel[6*0+2] = 3.0;  // master vz
    vel[6*1+1] = 99.0; // slave vy (should stay free because DOF 1 released)

    RLINKConstraint rl(0, {1}, RLINKType::Type2);
    rl.released_dofs[1] = true;  // release uy

    rl.apply_velocity_constraint(vel.data(), pos.data(), N);

    CHECK_NEAR(vel[6*1+0], 1.0,  1e-12, "Type2: slave vx constrained");
    CHECK_NEAR(vel[6*1+1], 99.0, 1e-12, "Type2: slave vy released (unchanged)");
    CHECK_NEAR(vel[6*1+2], 3.0,  1e-12, "Type2: slave vz constrained");
}

static void test_rlink_multiple_slaves() {
    using namespace nxs::fem;

    // Master node 0, slaves 1 and 2, Type1
    const int N = 3;
    std::vector<nxs::Real> vel(6 * N, 0.0);
    std::vector<nxs::Real> pos(3 * N, 0.0);

    vel[6*0+0] = 7.0;

    RLINKConstraint rl(0, {1, 2}, RLINKType::Type1);
    rl.apply_velocity_constraint(vel.data(), pos.data(), N);

    CHECK_NEAR(vel[6*1+0], 7.0, 1e-12, "Multi-slave: slave1 vx");
    CHECK_NEAR(vel[6*2+0], 7.0, 1e-12, "Multi-slave: slave2 vx");
}

static void test_rlink_constraint_forces() {
    using namespace nxs::fem;

    // Master node 0 acc=(10,0,0,...), slave node 1 acc=(0,0,0,...), mass=2
    // reaction = mass * (a_master - a_slave) = 2 * (10,0,0,...) = (20,0,0,...)
    const int N = 2;
    std::vector<nxs::Real> acc(6 * N, 0.0);
    std::vector<nxs::Real> mass(N, 0.0);

    acc[6*0+0] = 10.0;
    mass[1] = 2.0;

    RLINKConstraint rl(0, {1}, RLINKType::Type1);
    auto forces = rl.compute_constraint_forces(acc.data(), mass.data());

    CHECK(forces.size() == 6u, "constraint forces size = 6 per slave");
    CHECK_NEAR(forces[0], 20.0, 1e-12, "reaction force[0] = 2*10");
    CHECK_NEAR(forces[1], 0.0,  1e-12, "reaction force[1] = 0");
}

// ============================================================================
// GroupSet Tests
// ============================================================================

static void test_groupset_add_contains() {
    using namespace nxs::fem;

    GroupSet gs(SetType::NodeSet);
    gs.add(5);
    gs.add(3);
    gs.add(7);
    gs.add(3);  // duplicate

    CHECK(gs.size() == 3u, "add: size=3 (duplicate ignored)");
    CHECK(gs.contains(3), "contains 3");
    CHECK(gs.contains(5), "contains 5");
    CHECK(gs.contains(7), "contains 7");
    CHECK(!gs.contains(1), "does not contain 1");
}

static void test_groupset_remove() {
    using namespace nxs::fem;

    GroupSet gs(SetType::NodeSet, {1, 2, 3, 4, 5});
    gs.remove(3);
    gs.remove(99);  // no-op

    CHECK(gs.size() == 4u, "remove: size=4");
    CHECK(!gs.contains(3), "3 removed");
    CHECK(gs.contains(5), "5 still present");
}

static void test_groupset_sorted_invariant() {
    using namespace nxs::fem;

    GroupSet gs(SetType::NodeSet);
    gs.add(10);
    gs.add(2);
    gs.add(7);

    const auto& ids = gs.ids();
    CHECK(ids[0] == 2, "sorted: ids[0]=2");
    CHECK(ids[1] == 7, "sorted: ids[1]=7");
    CHECK(ids[2] == 10, "sorted: ids[2]=10");
}

static void test_groupset_union() {
    using namespace nxs::fem;

    GroupSet a(SetType::NodeSet, {1, 3, 5});
    GroupSet b(SetType::NodeSet, {3, 4, 5, 6});

    auto u = a.union_with(b);
    CHECK(u.size() == 5u, "union size = 5");
    CHECK(u.contains(1), "union has 1");
    CHECK(u.contains(3), "union has 3");
    CHECK(u.contains(4), "union has 4");
    CHECK(u.contains(6), "union has 6");
}

static void test_groupset_intersection() {
    using namespace nxs::fem;

    GroupSet a(SetType::NodeSet, {1, 2, 3, 4});
    GroupSet b(SetType::NodeSet, {2, 4, 6});

    auto inter = a.intersect(b);
    CHECK(inter.size() == 2u, "intersection size = 2");
    CHECK(inter.contains(2), "intersection has 2");
    CHECK(inter.contains(4), "intersection has 4");
    CHECK(!inter.contains(1), "intersection lacks 1");
    CHECK(!inter.contains(6), "intersection lacks 6");
}

static void test_groupset_difference() {
    using namespace nxs::fem;

    GroupSet a(SetType::NodeSet, {1, 2, 3, 4, 5});
    GroupSet b(SetType::NodeSet, {2, 4});

    auto diff = a.difference(b);
    CHECK(diff.size() == 3u, "difference size = 3");
    CHECK(diff.contains(1), "diff has 1");
    CHECK(diff.contains(3), "diff has 3");
    CHECK(diff.contains(5), "diff has 5");
    CHECK(!diff.contains(2), "diff lacks 2");
    CHECK(!diff.contains(4), "diff lacks 4");
}

static void test_groupset_complement() {
    using namespace nxs::fem;

    GroupSet universe(SetType::NodeSet, {1, 2, 3, 4, 5});
    GroupSet subset(SetType::NodeSet, {2, 4});

    auto comp = subset.complement(universe);
    CHECK(comp.size() == 3u, "complement size = 3");
    CHECK(comp.contains(1), "complement has 1");
    CHECK(comp.contains(3), "complement has 3");
    CHECK(comp.contains(5), "complement has 5");
    CHECK(!comp.contains(2), "complement lacks 2");
}

static void test_groupset_empty_ops() {
    using namespace nxs::fem;

    GroupSet a(SetType::NodeSet, {1, 2, 3});
    GroupSet empty(SetType::NodeSet);

    auto u = a.union_with(empty);
    CHECK(u.size() == 3u, "union with empty = original");

    auto inter = a.intersect(empty);
    CHECK(inter.size() == 0u, "intersect with empty = empty");

    auto diff = a.difference(empty);
    CHECK(diff.size() == 3u, "difference with empty = original");
}

// ============================================================================
// GroupSetManager Tests
// ============================================================================

static void test_manager_create_get() {
    using namespace nxs::fem;

    GroupSetManager mgr;
    auto& s = mgr.create_set("NODES_ALL", SetType::NodeSet);
    s.add(1); s.add(2); s.add(3);

    CHECK(mgr.has_set("NODES_ALL"), "manager has NODES_ALL");
    CHECK(mgr.get_set("NODES_ALL").size() == 3u, "manager set size = 3");
    CHECK(mgr.num_sets() == 1u, "manager has 1 set");
}

static void test_manager_expression_union() {
    using namespace nxs::fem;

    GroupSetManager mgr;
    auto& a = mgr.create_set("A", SetType::NodeSet);
    a.add(1); a.add(2);
    auto& b = mgr.create_set("B", SetType::NodeSet);
    b.add(3); b.add(4);

    auto result = mgr.evaluate_expression("A + B");
    CHECK(result.size() == 4u, "expr A+B: size=4");
    CHECK(result.contains(1), "expr A+B: has 1");
    CHECK(result.contains(4), "expr A+B: has 4");
}

static void test_manager_expression_difference() {
    using namespace nxs::fem;

    GroupSetManager mgr;
    auto& a = mgr.create_set("ALL", SetType::NodeSet);
    for (int i = 1; i <= 5; ++i) a.add(i);
    auto& b = mgr.create_set("EXCL", SetType::NodeSet);
    b.add(2); b.add(4);

    auto result = mgr.evaluate_expression("ALL - EXCL");
    CHECK(result.size() == 3u, "expr ALL-EXCL: size=3");
    CHECK(!result.contains(2), "expr ALL-EXCL: lacks 2");
    CHECK(!result.contains(4), "expr ALL-EXCL: lacks 4");
    CHECK(result.contains(1), "expr ALL-EXCL: has 1");
}

static void test_manager_expression_chain() {
    using namespace nxs::fem;

    GroupSetManager mgr;
    auto& a = mgr.create_set("S1", SetType::NodeSet);
    a.add(1); a.add(2); a.add(3);
    auto& b = mgr.create_set("S2", SetType::NodeSet);
    b.add(4); b.add(5);
    auto& c = mgr.create_set("S3", SetType::NodeSet);
    c.add(3); c.add(5);

    // S1 + S2 - S3 = {1,2,3,4,5} - {3,5} = {1,2,4}
    auto result = mgr.evaluate_expression("S1 + S2 - S3");
    CHECK(result.size() == 3u, "chain S1+S2-S3: size=3");
    CHECK(result.contains(1), "chain: has 1");
    CHECK(result.contains(2), "chain: has 2");
    CHECK(result.contains(4), "chain: has 4");
    CHECK(!result.contains(3), "chain: lacks 3");
    CHECK(!result.contains(5), "chain: lacks 5");
}

static void test_manager_expression_whitespace() {
    using namespace nxs::fem;

    GroupSetManager mgr;
    auto& a = mgr.create_set("X", SetType::NodeSet);
    a.add(10); a.add(20);
    auto& b = mgr.create_set("Y", SetType::NodeSet);
    b.add(20); b.add(30);

    // Extra spaces around operators
    auto result = mgr.evaluate_expression("  X  +  Y  ");
    CHECK(result.size() == 3u, "whitespace expr: size=3");
}

static void test_manager_missing_set_throws() {
    using namespace nxs::fem;

    GroupSetManager mgr;
    bool threw = false;
    try {
        mgr.get_set("NONEXISTENT");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw, "get_set throws for unknown name");
}

static void test_manager_expression_missing_throws() {
    using namespace nxs::fem;

    GroupSetManager mgr;
    mgr.create_set("A", SetType::NodeSet);

    bool threw = false;
    try {
        mgr.evaluate_expression("A + MISSING");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw, "evaluate_expression throws for missing set");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Wave 43: FXBODY + RLINK + GroupSet Tests ===\n\n";

    // FXBODY
    std::cout << "-- FXBODY Superelement --\n";
    test_superelement_construction();
    test_fxbody_stiffness_assembly();
    test_fxbody_stiffness_accumulation();
    test_fxbody_mass_assembly();
    test_fxbody_interface_forces();
    test_fxbody_internal_recovery();
    test_fxbody_diagonal_only_assembly();

    // RLINK
    std::cout << "-- RLINK Velocity Constraints --\n";
    test_rlink_type0_translation();
    test_rlink_type0_with_rotation();
    test_rlink_type1_translation_only();
    test_rlink_type10_spherical();
    test_rlink_type2_selected_dofs();
    test_rlink_multiple_slaves();
    test_rlink_constraint_forces();

    // GroupSet
    std::cout << "-- GroupSet Algebra --\n";
    test_groupset_add_contains();
    test_groupset_remove();
    test_groupset_sorted_invariant();
    test_groupset_union();
    test_groupset_intersection();
    test_groupset_difference();
    test_groupset_complement();
    test_groupset_empty_ops();

    // GroupSetManager
    std::cout << "-- GroupSetManager --\n";
    test_manager_create_get();
    test_manager_expression_union();
    test_manager_expression_difference();
    test_manager_expression_chain();
    test_manager_expression_whitespace();
    test_manager_missing_set_throws();
    test_manager_expression_missing_throws();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return tests_failed > 0 ? 1 : 0;
}
