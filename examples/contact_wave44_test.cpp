/**
 * @file contact_wave44_test.cpp
 * @brief Wave 44a: Contact Gap Evolution — 15 tests
 *
 * Tests:
 *  1.  IGAP=0 (Constant)          — gap unchanged after thickness update
 *  2.  IGAP=1 (Variable)          — gap scales with thickness ratio
 *  3.  IGAP=2 (VariableScaled)    — Variable + scale factor applied
 *  4.  Thinning shell scenario     — 50% thickness, gap follows proportionally
 *  5.  Gap clamping (min/max)      — set_gap_limits clamps updates
 *  6.  Deleted element handling    — set_deleted → gap = infinity
 *  7.  max_gap / min_gap queries   — correct statistics over all nodes
 *  8.  Reset to uniform thickness  — reset() restores initial state
 *  9.  GapAwarePairFilter pair filter — filter_pair returns correct bool
 * 10.  GapAwarePairFilter search distance — adjusted_search_distance = max+margin
 * 11.  BilateralGapHandler symmetric gap — sum of both node gaps
 * 12.  Zero-thickness guard         — no division by zero
 * 13.  Large model (1000 nodes)     — performance and correctness
 * 14.  Multiple update cycles       — gap evolves correctly over several steps
 * 15.  Mixed deleted/active nodes   — statistics ignore deleted
 */

#include <nexussim/fem/contact_wave44.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <limits>

using namespace nxs;
using namespace nxs::fem;

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
// Test 1: IGAP=0 Constant — gap never changes
// ============================================================================
void test_constant_gap() {
    std::cout << "\n=== Test 1: IGAP=0 Constant gap ===\n";

    const std::size_t N  = 4;
    const Real        t0 = 2.0;
    ContactGapEvolution cge(N, t0, GapMode::Constant);

    // Initial gaps should equal the initial thickness
    for (std::size_t i = 0; i < N; ++i) {
        CHECK_NEAR(cge.get_effective_gap(i), t0, 1e-12,
                   "Constant: initial gap == t0 for node " + std::to_string(i));
    }

    // Update with halved thickness — gaps must NOT change
    std::vector<Real> pos(N * 3, 0.0);
    std::vector<Real> thk = {1.0, 0.5, 1.5, 0.8};
    cge.update_gaps(pos.data(), thk.data(), N);

    for (std::size_t i = 0; i < N; ++i) {
        CHECK_NEAR(cge.get_effective_gap(i), t0, 1e-12,
                   "Constant: gap unchanged after update, node " + std::to_string(i));
    }
}

// ============================================================================
// Test 2: IGAP=1 Variable — gap scales with thickness ratio
// ============================================================================
void test_variable_gap() {
    std::cout << "\n=== Test 2: IGAP=1 Variable gap ===\n";

    const std::size_t N  = 3;
    const Real        t0 = 4.0;
    ContactGapEvolution cge(N, t0, GapMode::Variable);

    std::vector<Real> pos(N * 3, 0.0);
    // Each node has a different current thickness
    std::vector<Real> thk = {4.0, 2.0, 8.0}; // ratios: 1.0, 0.5, 2.0

    cge.update_gaps(pos.data(), thk.data(), N);

    // gap_s[i] = gap_s0[i] * (thk[i] / thknod0[i]) = t0 * ratio
    CHECK_NEAR(cge.get_effective_gap(0), 4.0, 1e-10, "Variable: ratio=1.0, gap=4.0");
    CHECK_NEAR(cge.get_effective_gap(1), 2.0, 1e-10, "Variable: ratio=0.5, gap=2.0");
    CHECK_NEAR(cge.get_effective_gap(2), 8.0, 1e-10, "Variable: ratio=2.0, gap=8.0");
}

// ============================================================================
// Test 3: IGAP=2 VariableScaled — Variable * scale factor
// ============================================================================
void test_variable_scaled_gap() {
    std::cout << "\n=== Test 3: IGAP=2 VariableScaled gap ===\n";

    const std::size_t N     = 2;
    const Real        t0    = 4.0;
    const Real        scale = 0.5;
    ContactGapEvolution cge(N, t0, GapMode::VariableScaled);
    cge.apply_gap_scale(scale);

    std::vector<Real> pos(N * 3, 0.0);
    std::vector<Real> thk = {4.0, 2.0}; // ratios: 1.0, 0.5

    cge.update_gaps(pos.data(), thk.data(), N);

    // gap_s[i] = gap_s0[i] * ratio * scale
    CHECK_NEAR(cge.get_effective_gap(0), 4.0 * 1.0 * scale, 1e-10,
               "VarScaled: node0 gap = 4*1*0.5 = 2.0");
    CHECK_NEAR(cge.get_effective_gap(1), 4.0 * 0.5 * scale, 1e-10,
               "VarScaled: node1 gap = 4*0.5*0.5 = 1.0");
}

// ============================================================================
// Test 4: Thinning shell scenario — 50% thickness → 50% gap
// ============================================================================
void test_thinning_shell() {
    std::cout << "\n=== Test 4: Thinning shell scenario ===\n";

    const std::size_t N  = 5;
    const Real        t0 = 10.0;
    ContactGapEvolution cge(N, t0, GapMode::Variable);

    std::vector<Real> pos(N * 3, 0.0);
    std::vector<Real> thk(N, 5.0); // 50% of initial

    cge.update_gaps(pos.data(), thk.data(), N);

    for (std::size_t i = 0; i < N; ++i) {
        CHECK_NEAR(cge.get_effective_gap(i), 5.0, 1e-10,
                   "Thinning: 50% thickness → 50% gap, node " + std::to_string(i));
    }

    // Now check that Constant mode is unchanged under same thinning
    ContactGapEvolution cge_const(N, t0, GapMode::Constant);
    cge_const.update_gaps(pos.data(), thk.data(), N);
    for (std::size_t i = 0; i < N; ++i) {
        CHECK_NEAR(cge_const.get_effective_gap(i), t0, 1e-10,
                   "Thinning: Constant mode stays at t0, node " + std::to_string(i));
    }
}

// ============================================================================
// Test 5: Gap clamping (min/max limits)
// ============================================================================
void test_gap_clamping() {
    std::cout << "\n=== Test 5: Gap clamping ===\n";

    const std::size_t N  = 4;
    const Real        t0 = 4.0;
    ContactGapEvolution cge(N, t0, GapMode::Variable);

    // Clamp to [2.0, 6.0]
    cge.set_gap_limits(2.0, 6.0);

    std::vector<Real> pos(N * 3, 0.0);
    // Ratios: 0.25 → 1.0 (clamp to 2.0), 1.0 → 4.0 (ok), 2.0 → 8.0 (clamp to 6.0), 0.1 → 0.4 (clamp to 2.0)
    std::vector<Real> thk = {1.0, 4.0, 8.0, 0.4};

    cge.update_gaps(pos.data(), thk.data(), N);

    CHECK_NEAR(cge.get_effective_gap(0), 2.0, 1e-10, "Clamp: below min → 2.0");
    CHECK_NEAR(cge.get_effective_gap(1), 4.0, 1e-10, "Clamp: in range → 4.0");
    CHECK_NEAR(cge.get_effective_gap(2), 6.0, 1e-10, "Clamp: above max → 6.0");
    CHECK_NEAR(cge.get_effective_gap(3), 2.0, 1e-10, "Clamp: far below min → 2.0");
}

// ============================================================================
// Test 6: Deleted element handling
// ============================================================================
void test_deleted_element() {
    std::cout << "\n=== Test 6: Deleted element handling ===\n";

    const std::size_t N  = 3;
    const Real        t0 = 5.0;
    ContactGapEvolution cge(N, t0, GapMode::Constant);

    CHECK(!cge.is_deleted(0), "Not deleted initially: node 0");
    CHECK(!cge.is_deleted(1), "Not deleted initially: node 1");

    cge.set_deleted(1);

    CHECK(!cge.is_deleted(0), "Node 0 still active after deleting node 1");
    CHECK(cge.is_deleted(1),  "Node 1 is deleted");
    CHECK(!cge.is_deleted(2), "Node 2 still active");

    // Deleted node returns infinity
    const Real inf_gap = cge.get_effective_gap(1);
    CHECK(std::isinf(inf_gap), "Deleted node gap is +infinity");

    // Active nodes unaffected
    CHECK_NEAR(cge.get_effective_gap(0), t0, 1e-12, "Active node 0 gap = t0");
    CHECK_NEAR(cge.get_effective_gap(2), t0, 1e-12, "Active node 2 gap = t0");

    // Out-of-range index also returns infinity safely
    const Real oob_gap = cge.get_effective_gap(999);
    CHECK(std::isinf(oob_gap), "Out-of-range node returns infinity");
}

// ============================================================================
// Test 7: max_gap / min_gap queries
// ============================================================================
void test_max_min_gap() {
    std::cout << "\n=== Test 7: max_gap / min_gap ===\n";

    const std::size_t N  = 4;
    const Real        t0 = 4.0;
    ContactGapEvolution cge(N, t0, GapMode::Variable);

    std::vector<Real> pos(N * 3, 0.0);
    std::vector<Real> thk = {2.0, 4.0, 8.0, 1.0}; // gaps: 2, 4, 8, 1

    cge.update_gaps(pos.data(), thk.data(), N);

    CHECK_NEAR(cge.max_gap(), 8.0, 1e-10, "max_gap = 8.0");
    CHECK_NEAR(cge.min_gap(), 1.0, 1e-10, "min_gap = 1.0");

    // Delete the node with max gap — max should update
    cge.set_deleted(2); // gap was 8.0
    CHECK_NEAR(cge.max_gap(), 4.0, 1e-10, "max_gap after deleting node2 = 4.0");
    CHECK_NEAR(cge.min_gap(), 1.0, 1e-10, "min_gap unaffected by deleting node2 = 1.0");
}

// ============================================================================
// Test 8: Reset to uniform thickness
// ============================================================================
void test_reset() {
    std::cout << "\n=== Test 8: Reset to uniform thickness ===\n";

    const std::size_t N  = 4;
    const Real        t0 = 4.0;
    ContactGapEvolution cge(N, t0, GapMode::Variable);

    // Thin some nodes, delete one
    std::vector<Real> pos(N * 3, 0.0);
    std::vector<Real> thk = {1.0, 2.0, 3.0, 4.0};
    cge.update_gaps(pos.data(), thk.data(), N);
    cge.set_deleted(3);

    // Reset to new uniform thickness
    const Real t_new = 7.0;
    cge.reset(t_new);

    // All nodes including former deleted node should be back
    CHECK(!cge.is_deleted(3), "Node 3 un-deleted after reset");
    for (std::size_t i = 0; i < N; ++i) {
        CHECK_NEAR(cge.get_effective_gap(i), t_new, 1e-10,
                   "Reset: gap = t_new for node " + std::to_string(i));
    }
    CHECK_NEAR(cge.max_gap(), t_new, 1e-10, "Reset: max_gap = t_new");
    CHECK_NEAR(cge.min_gap(), t_new, 1e-10, "Reset: min_gap = t_new");
}

// ============================================================================
// Test 9: GapAwarePairFilter — filter_pair correctness
// ============================================================================
void test_pair_filter() {
    std::cout << "\n=== Test 9: GapAwarePairFilter::filter_pair ===\n";

    const std::size_t N  = 3;
    const Real        t0 = 5.0;
    ContactGapEvolution cge(N, t0, GapMode::Constant);
    // Gaps are all 5.0

    GapAwarePairFilter filter(cge, 0.0 /*margin*/);

    // penetration = 6.0 > gap(5.0) - tol(0.0) = 5.0  → keep
    CHECK(filter.filter_pair(0, 6.0, 0.0), "Filter: penetration 6 > gap 5 → keep");

    // penetration = 4.0 < gap(5.0) - tol(0.0) = 5.0  → reject
    CHECK(!filter.filter_pair(0, 4.0, 0.0), "Filter: penetration 4 < gap 5 → reject");

    // penetration = 5.0 == gap(5.0) - tol(0.0) = 5.0  → reject (not strictly greater)
    CHECK(!filter.filter_pair(0, 5.0, 0.0), "Filter: penetration == gap → reject");

    // With tolerance = 2.0: effective threshold = 5.0 - 2.0 = 3.0
    // penetration = 4.0 > 3.0 → keep
    CHECK(filter.filter_pair(1, 4.0, 2.0), "Filter: penetration 4 > (gap 5 - tol 2) = 3 → keep");

    // Deleted node always rejected
    cge.set_deleted(2);
    CHECK(!filter.filter_pair(2, 100.0, 0.0), "Filter: deleted node always rejected");
}

// ============================================================================
// Test 10: GapAwarePairFilter — adjusted_search_distance
// ============================================================================
void test_adjusted_search_distance() {
    std::cout << "\n=== Test 10: GapAwarePairFilter::adjusted_search_distance ===\n";

    const std::size_t N      = 4;
    const Real        t0     = 4.0;
    const Real        margin = 1.5;
    ContactGapEvolution cge(N, t0, GapMode::Variable);

    std::vector<Real> pos(N * 3, 0.0);
    std::vector<Real> thk = {2.0, 4.0, 8.0, 1.0}; // gaps: 2, 4, 8, 1 after update
    cge.update_gaps(pos.data(), thk.data(), N);

    GapAwarePairFilter filter(cge, margin);

    // max_gap() = 8.0, margin = 1.5 → adjusted = 9.5
    CHECK_NEAR(filter.adjusted_search_distance(), 9.5, 1e-10,
               "adjusted_search_distance = max_gap(8) + margin(1.5) = 9.5");

    // Delete the max-gap node → max becomes 4.0, adjusted = 5.5
    cge.set_deleted(2);
    CHECK_NEAR(filter.adjusted_search_distance(), 5.5, 1e-10,
               "adjusted_search_distance after delete = max_gap(4) + margin(1.5) = 5.5");
}

// ============================================================================
// Test 11: BilateralGapHandler — symmetric gap computation
// ============================================================================
void test_bilateral_gap() {
    std::cout << "\n=== Test 11: BilateralGapHandler symmetric gap ===\n";

    const std::size_t N_a = 3;
    const std::size_t N_b = 3;
    ContactGapEvolution gap_a(N_a, 3.0, GapMode::Constant);
    ContactGapEvolution gap_b(N_b, 2.0, GapMode::Constant);

    BilateralGapHandler bgh;

    // Combined gap = 3.0 + 2.0 = 5.0
    CHECK_NEAR(bgh.compute_bilateral_gap(0, 0, gap_a, gap_b), 5.0, 1e-10,
               "Bilateral: gap_a(3)+gap_b(2) = 5.0");

    // in_contact: distance < combined gap
    CHECK(bgh.in_contact(4.9, 0, 0, gap_a, gap_b), "Bilateral: distance 4.9 < 5.0 → in contact");
    CHECK(!bgh.in_contact(5.0, 0, 0, gap_a, gap_b), "Bilateral: distance 5.0 == 5.0 → not in contact");
    CHECK(!bgh.in_contact(5.1, 0, 0, gap_a, gap_b), "Bilateral: distance 5.1 > 5.0 → not in contact");

    // Deleted node on surface A → infinity
    gap_a.set_deleted(1);
    const Real g_del = bgh.compute_bilateral_gap(1, 0, gap_a, gap_b);
    CHECK(std::isinf(g_del), "Bilateral: deleted node A → infinity");
    CHECK(!bgh.in_contact(0.0, 1, 0, gap_a, gap_b),
          "Bilateral: deleted node A → not in contact");

    // Deleted node on surface B → infinity
    gap_b.set_deleted(2);
    const Real g_del_b = bgh.compute_bilateral_gap(0, 2, gap_a, gap_b);
    CHECK(std::isinf(g_del_b), "Bilateral: deleted node B → infinity");
}

// ============================================================================
// Test 12: Zero-thickness guard — no division by zero
// ============================================================================
void test_zero_thickness_guard() {
    std::cout << "\n=== Test 12: Zero-thickness guard ===\n";

    const std::size_t N  = 2;
    const Real        t0 = 0.0; // degenerate: zero initial thickness
    // Should not throw or produce NaN/Inf in the gap arrays
    ContactGapEvolution cge(N, t0, GapMode::Variable);

    std::vector<Real> pos(N * 3, 0.0);
    std::vector<Real> thk_zero = {0.0, 0.0};

    // This must not crash or divide by zero
    cge.update_gaps(pos.data(), thk_zero.data(), N);

    for (std::size_t i = 0; i < N; ++i) {
        const Real g = cge.get_effective_gap(i);
        CHECK(!std::isnan(g), "Zero-thickness: gap is not NaN, node " + std::to_string(i));
        CHECK(!std::isinf(g) || g > 0, "Zero-thickness: gap is finite or +inf (not -inf), node " + std::to_string(i));
    }

    // Same with VariableScaled
    ContactGapEvolution cge2(N, t0, GapMode::VariableScaled);
    cge2.apply_gap_scale(0.5);
    cge2.update_gaps(pos.data(), thk_zero.data(), N);

    for (std::size_t i = 0; i < N; ++i) {
        const Real g = cge2.get_effective_gap(i);
        CHECK(!std::isnan(g), "Zero-thickness scaled: gap is not NaN, node " + std::to_string(i));
    }
}

// ============================================================================
// Test 13: Large model (1000 nodes) — performance and correctness
// ============================================================================
void test_large_model() {
    std::cout << "\n=== Test 13: Large model (1000 nodes) ===\n";

    const std::size_t N  = 1000;
    const Real        t0 = 10.0;
    ContactGapEvolution cge(N, t0, GapMode::Variable);

    CHECK(cge.num_nodes() == N, "Large model: num_nodes == 1000");

    std::vector<Real> pos(N * 3, 0.0);
    std::vector<Real> thk(N);
    // Linearly varying thickness: 1.0 to 20.0
    for (std::size_t i = 0; i < N; ++i)
        thk[i] = Real(1.0) + Real(19.0) * Real(i) / Real(N - 1);

    cge.update_gaps(pos.data(), thk.data(), N);

    // Check first, middle, and last node
    // gap[0] = t0 * (1.0 / t0) = 1.0
    CHECK_NEAR(cge.get_effective_gap(0),   1.0,  1e-8, "Large: node 0 gap = 1.0");
    // gap[N-1] = t0 * (20.0 / t0) = 20.0
    CHECK_NEAR(cge.get_effective_gap(N-1), 20.0, 1e-8, "Large: node 999 gap = 20.0");

    CHECK_NEAR(cge.max_gap(), 20.0, 1e-8, "Large: max_gap = 20.0");
    CHECK_NEAR(cge.min_gap(), 1.0,  1e-8, "Large: min_gap = 1.0");

    // All gaps should be finite and positive
    bool all_valid = true;
    for (std::size_t i = 0; i < N; ++i) {
        if (std::isnan(cge.get_effective_gap(i)) || cge.get_effective_gap(i) <= 0)
            all_valid = false;
    }
    CHECK(all_valid, "Large: all 1000 gaps are finite and positive");
}

// ============================================================================
// Test 14: Multiple update cycles — gap evolves correctly over several steps
// ============================================================================
void test_multiple_update_cycles() {
    std::cout << "\n=== Test 14: Multiple update cycles ===\n";

    const std::size_t N  = 3;
    const Real        t0 = 8.0;
    ContactGapEvolution cge(N, t0, GapMode::Variable);

    std::vector<Real> pos(N * 3, 0.0);

    // Step 1: thickness = 8.0 (unchanged)
    {
        std::vector<Real> thk = {8.0, 8.0, 8.0};
        cge.update_gaps(pos.data(), thk.data(), N);
        CHECK_NEAR(cge.get_effective_gap(0), 8.0, 1e-10, "Cycle1: gap=8.0");
    }

    // Step 2: thickness = 4.0 (50% thinning)
    {
        std::vector<Real> thk = {4.0, 4.0, 4.0};
        cge.update_gaps(pos.data(), thk.data(), N);
        CHECK_NEAR(cge.get_effective_gap(0), 4.0, 1e-10, "Cycle2: gap=4.0 after 50% thin");
    }

    // Step 3: thickness = 2.0 (25% of original)
    {
        std::vector<Real> thk = {2.0, 2.0, 2.0};
        cge.update_gaps(pos.data(), thk.data(), N);
        CHECK_NEAR(cge.get_effective_gap(0), 2.0, 1e-10, "Cycle3: gap=2.0 after 75% thin");
    }

    // Step 4: thickness rebounds to 6.0 (thickening)
    {
        std::vector<Real> thk = {6.0, 6.0, 6.0};
        cge.update_gaps(pos.data(), thk.data(), N);
        // gap = t0 * (6 / 8) = 6.0
        CHECK_NEAR(cge.get_effective_gap(0), 6.0, 1e-10, "Cycle4: gap=6.0 after rebound");
    }

    // Constant mode should remain unchanged throughout all cycles
    ContactGapEvolution cge_c(N, t0, GapMode::Constant);
    {
        std::vector<Real> thk = {1.0, 1.0, 1.0};
        cge_c.update_gaps(pos.data(), thk.data(), N);
        cge_c.update_gaps(pos.data(), thk.data(), N);
        cge_c.update_gaps(pos.data(), thk.data(), N);
    }
    CHECK_NEAR(cge_c.get_effective_gap(0), t0, 1e-10,
               "Cycle: Constant mode unchanged after 3 updates");
}

// ============================================================================
// Test 15: Mixed deleted/active nodes — statistics ignore deleted
// ============================================================================
void test_mixed_deleted_active() {
    std::cout << "\n=== Test 15: Mixed deleted/active nodes ===\n";

    const std::size_t N  = 6;
    const Real        t0 = 5.0;
    ContactGapEvolution cge(N, t0, GapMode::Variable);

    std::vector<Real> pos(N * 3, 0.0);
    // Gaps after update: 1, 3, 5, 7, 9, 11
    std::vector<Real> thk = {1.0, 3.0, 5.0, 7.0, 9.0, 11.0};
    cge.update_gaps(pos.data(), thk.data(), N);

    // Delete nodes with gap=9 (node 4) and gap=1 (node 0)
    cge.set_deleted(0); // gap was 1.0
    cge.set_deleted(4); // gap was 9.0

    // max over active nodes: max(3, 5, 7, 11) = 11
    CHECK_NEAR(cge.max_gap(), 11.0, 1e-10, "Mixed: max_gap ignores deleted → 11.0");

    // min over active nodes: min(3, 5, 7, 11) = 3
    CHECK_NEAR(cge.min_gap(), 3.0, 1e-10, "Mixed: min_gap ignores deleted → 3.0");

    // is_deleted correctly distinguishes nodes
    CHECK(cge.is_deleted(0),  "Mixed: node 0 is deleted");
    CHECK(!cge.is_deleted(1), "Mixed: node 1 is active");
    CHECK(!cge.is_deleted(2), "Mixed: node 2 is active");
    CHECK(!cge.is_deleted(3), "Mixed: node 3 is active");
    CHECK(cge.is_deleted(4),  "Mixed: node 4 is deleted");
    CHECK(!cge.is_deleted(5), "Mixed: node 5 is active");

    // Active gaps are finite
    CHECK(!std::isinf(cge.get_effective_gap(1)), "Mixed: active node 1 gap is finite");
    CHECK(!std::isinf(cge.get_effective_gap(5)), "Mixed: active node 5 gap is finite");

    // Deleted gaps are infinity
    CHECK(std::isinf(cge.get_effective_gap(0)), "Mixed: deleted node 0 gap is +inf");
    CHECK(std::isinf(cge.get_effective_gap(4)), "Mixed: deleted node 4 gap is +inf");

    // GapAwarePairFilter rejects deleted nodes
    GapAwarePairFilter filter(cge, 0.0);
    CHECK(!filter.filter_pair(0, 1e6, 0.0), "Mixed filter: deleted node 0 rejected");
    CHECK(!filter.filter_pair(4, 1e6, 0.0), "Mixed filter: deleted node 4 rejected");
    CHECK(filter.filter_pair(5, 20.0, 0.0), "Mixed filter: active node 5, penetration 20 > gap 11 → keep");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== Wave 44 Contact Gap Evolution Tests ===\n";

    test_constant_gap();
    test_variable_gap();
    test_variable_scaled_gap();
    test_thinning_shell();
    test_gap_clamping();
    test_deleted_element();
    test_max_min_gap();
    test_reset();
    test_pair_filter();
    test_adjusted_search_distance();
    test_bilateral_gap();
    test_zero_thickness_guard();
    test_large_model();
    test_multiple_update_cycles();
    test_mixed_deleted_active();

    std::cout << "\n=== Wave 44 Contact Gap Results ===\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";
    return tests_failed > 0 ? 1 : 0;
}
