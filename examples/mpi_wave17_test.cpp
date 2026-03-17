/**
 * @file mpi_wave17_test.cpp
 * @brief Comprehensive test for Wave 17: MPI Completion
 *
 * Tests 6 classes (30+ assertions) in serial mode:
 *  1. DistributedAssembler   - CSR build, local assembly, ghost exchange
 *  2. GhostExchanger         - Pattern setup, field synchronization
 *  3. DomainDecomposer       - RCB, greedy, quality metrics
 *  4. ParallelContactDetector - AABB overlap, contact detection
 *  5. LoadBalancer            - Imbalance, migration plans
 *  6. ScalabilityBenchmark   - Timing, speedup, reporting
 */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>
#include "../include/nexussim/parallel/mpi_wave17.hpp"

using Real = double;
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
    else { printf("[FAIL] %s (got %g, expected %g)\n", name, v_, e_); fail_count++; } \
} while(0)

using namespace nxs::parallel;

// ============================================================================
// Test 1: CSR Matrix and Distributed Assembler
// ============================================================================
void test_csr_matrix() {
    printf("--- CSR Matrix ---\n");

    // Build a 3x3 matrix from triplets:
    // [ 2  1  0 ]
    // [ 1  3  1 ]
    // [ 0  1  2 ]
    std::vector<Triplet> triplets = {
        {0, 0, 1.0}, {0, 0, 1.0},  // row 0, col 0: 1+1=2
        {0, 1, 1.0},               // row 0, col 1: 1
        {1, 0, 1.0},               // row 1, col 0: 1
        {1, 1, 3.0},               // row 1, col 1: 3
        {1, 2, 1.0},               // row 1, col 2: 1
        {2, 1, 1.0},               // row 2, col 1: 1
        {2, 2, 2.0}                // row 2, col 2: 2
    };

    CSRMatrix csr = build_csr_from_triplets(triplets, 3, 3);

    CHECK(csr.n_rows == 3, "CSR n_rows = 3");
    CHECK(csr.n_cols == 3, "CSR n_cols = 3");
    CHECK(csr.row_ptr.size() == 4, "CSR row_ptr size = 4");

    // Check entries via get_entry
    CHECK_NEAR(csr.get_entry(0, 0), 2.0, 1e-10, "CSR(0,0) = 2.0 (merged duplicates)");
    CHECK_NEAR(csr.get_entry(0, 1), 1.0, 1e-10, "CSR(0,1) = 1.0");
    CHECK_NEAR(csr.get_entry(1, 1), 3.0, 1e-10, "CSR(1,1) = 3.0");
    CHECK_NEAR(csr.get_entry(2, 2), 2.0, 1e-10, "CSR(2,2) = 2.0");
    CHECK_NEAR(csr.get_entry(0, 2), 0.0, 1e-10, "CSR(0,2) = 0.0 (zero entry)");
}

void test_distributed_assembler() {
    printf("\n--- Distributed Assembler ---\n");

    DistributedAssembler assembler;

    // Single rank owning 4 rows (global rows 0..3), total 4 rows
    // Ghost nodes: global ID 4 maps back to local (if range extended)
    std::vector<Index> ghost_ids = {0, 1};  // Ghost IDs map to local rows
    assembler.initialize(4, 0, 4, ghost_ids, 0, 1);

    CHECK(assembler.n_local_rows() == 4, "Assembler local rows = 4");
    CHECK(assembler.global_row_start() == 0, "Assembler global row start = 0");
    CHECK(assembler.n_global_rows() == 4, "Assembler global rows = 4");
    CHECK(assembler.rank() == 0, "Assembler rank = 0");
    CHECK(assembler.n_ranks() == 1, "Assembler n_ranks = 1");

    // Add stiffness entries for a simple tridiagonal matrix
    for (Index i = 0; i < 4; ++i) {
        assembler.add_stiffness(i, i, 2.0);
        if (i > 0) assembler.add_stiffness(i, i - 1, -1.0);
        if (i < 3) assembler.add_stiffness(i, i + 1, -1.0);
    }

    // Add forces
    assembler.add_force(0, 1.0);
    assembler.add_force(3, -1.0);

    // Add ghost contributions
    assembler.add_ghost_contribution(0, 0.5);  // ghost[0] -> global 0
    assembler.add_ghost_contribution(1, 0.3);  // ghost[1] -> global 1

    // Assemble
    assembler.assemble_local();
    CHECK(assembler.is_assembled(), "Assembler is assembled");

    const CSRMatrix& K = assembler.local_matrix();
    CHECK(K.n_rows == 4, "Local K has 4 rows");
    CHECK_NEAR(K.get_entry(0, 0), 2.0, 1e-10, "K(0,0) = 2.0");
    CHECK_NEAR(K.get_entry(1, 1), 2.0, 1e-10, "K(1,1) = 2.0");

    // Check force
    CHECK_NEAR(assembler.local_force()[0], 1.0, 1e-10, "f[0] = 1.0 before ghost exchange");

    // Exchange ghosts (serial: accumulates local ghost contributions)
    assembler.exchange_ghost_contributions();
    CHECK_NEAR(assembler.local_force()[0], 1.5, 1e-10, "f[0] = 1.5 after ghost exchange");
    CHECK_NEAR(assembler.local_force()[1], 0.3, 1e-10, "f[1] = 0.3 after ghost exchange");

    // Global row mapping
    CHECK(assembler.get_global_row(0) == 0, "Global row 0 maps correctly");
    CHECK(assembler.get_global_row(3) == 3, "Global row 3 maps correctly");
}

// ============================================================================
// Test 2: Ghost Exchanger
// ============================================================================
void test_ghost_exchanger() {
    printf("\n--- Ghost Exchanger ---\n");

    GhostExchanger exchanger;

    // Serial setup: 5 local nodes, 2 ghost nodes at local indices 5,6
    std::vector<int> ghost_owner = {0, 0};  // all owned by rank 0
    std::vector<Index> ghost_local = {5, 6};
    std::vector<std::pair<Index, int>> shared;  // no sharing in serial

    exchanger.setup_communication_pattern(5, ghost_owner, ghost_local, shared, 0, 1);

    CHECK(exchanger.is_pattern_ready(), "Ghost pattern is ready");
    CHECK(exchanger.n_ghost_nodes() == 2, "Ghost exchanger has 2 ghost nodes");
    CHECK(exchanger.n_neighbors() == 0, "No neighbors in serial mode");
    CHECK(exchanger.my_rank() == 0, "Ghost exchanger rank = 0");

    // Create field data: 7 nodes, 1 DOF each
    std::vector<Real> field = {1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0};

    // In serial mode, update_ghost_values is a no-op
    exchanger.update_ghost_values(field, 1);
    CHECK(!exchanger.is_exchange_in_progress(), "Exchange completed");

    // Field should be unchanged in serial (no real comm)
    CHECK_NEAR(field[0], 1.0, 1e-10, "Field[0] unchanged after serial exchange");
    CHECK_NEAR(field[4], 5.0, 1e-10, "Field[4] unchanged after serial exchange");

    // Test with multi-DOF
    std::vector<Real> field3 = {
        1.0, 0.0, 0.0,  // node 0
        0.0, 2.0, 0.0,  // node 1
        0.0, 0.0, 3.0,  // node 2
        1.0, 1.0, 1.0,  // node 3
        2.0, 2.0, 2.0,  // node 4
        0.0, 0.0, 0.0,  // ghost 0
        0.0, 0.0, 0.0   // ghost 1
    };
    exchanger.update_ghost_values(field3, 3);
    CHECK(!exchanger.is_exchange_in_progress(), "Multi-DOF exchange completed");
}

// ============================================================================
// Test 3: Domain Decomposition
// ============================================================================
void test_domain_decomposer() {
    printf("\n--- Domain Decomposition ---\n");

    DomainDecomposer decomposer;

    // Create 12 elements in a 4x3 grid (x=[0..3], y=[0..2], z=0)
    const Index n_elem = 12;
    std::vector<Real> coords(n_elem * 3);
    for (Index i = 0; i < 4; ++i) {
        for (Index j = 0; j < 3; ++j) {
            Index idx = i * 3 + j;
            coords[idx * 3 + 0] = static_cast<Real>(i);  // x
            coords[idx * 3 + 1] = static_cast<Real>(j);  // y
            coords[idx * 3 + 2] = 0.0;                   // z
        }
    }

    // Test RCB with 2 partitions
    auto parts2 = decomposer.partition_rcb(2, coords, n_elem);
    CHECK(parts2.size() == n_elem, "RCB 2-part returns correct size");

    // Check that we have exactly 2 partitions
    std::set<int> unique_parts2(parts2.begin(), parts2.end());
    CHECK(unique_parts2.size() == 2, "RCB produces exactly 2 partitions");

    // Count elements per partition
    int count0 = 0, count1 = 0;
    for (int p : parts2) {
        if (p == 0) count0++;
        else count1++;
    }
    CHECK(count0 == 6 && count1 == 6, "RCB 2-part splits 12 elements into 6+6");

    // Test RCB with 4 partitions
    auto parts4 = decomposer.partition_rcb(4, coords, n_elem);
    std::set<int> unique_parts4(parts4.begin(), parts4.end());
    CHECK(unique_parts4.size() == 4, "RCB produces exactly 4 partitions");

    // Each partition should have 3 elements (12/4)
    std::vector<int> part_counts(4, 0);
    for (int p : parts4) part_counts[p]++;
    CHECK(part_counts[0] == 3 && part_counts[1] == 3 &&
          part_counts[2] == 3 && part_counts[3] == 3,
          "RCB 4-part splits 12 elements into 3+3+3+3");

    // Test with weights
    std::vector<Real> weights(n_elem, 1.0);
    weights[0] = 5.0;  // First element is 5x heavier
    auto parts_w = decomposer.partition_rcb(2, coords, n_elem, weights);
    CHECK(parts_w.size() == n_elem, "Weighted RCB returns correct size");

    // Test quality metrics
    // Build simple adjacency for the 4x3 grid
    std::vector<std::vector<Index>> adj(n_elem);
    for (Index i = 0; i < 4; ++i) {
        for (Index j = 0; j < 3; ++j) {
            Index idx = i * 3 + j;
            if (i > 0) adj[idx].push_back((i - 1) * 3 + j);
            if (i < 3) adj[idx].push_back((i + 1) * 3 + j);
            if (j > 0) adj[idx].push_back(i * 3 + (j - 1));
            if (j < 2) adj[idx].push_back(i * 3 + (j + 1));
        }
    }

    auto quality = decomposer.compute_quality(parts2, 2, adj);
    CHECK(quality.n_parts == 2, "Quality n_parts = 2");
    CHECK_NEAR(quality.load_imbalance_ratio, 1.0, 0.01, "Balanced 2-part imbalance = 1.0");
    CHECK(quality.edge_cut_count > 0, "Edge cuts exist for 2-part grid");

    // Test greedy partitioning
    auto parts_greedy = decomposer.partition_greedy(3, adj, n_elem);
    CHECK(parts_greedy.size() == n_elem, "Greedy partition returns correct size");
    std::set<int> unique_greedy(parts_greedy.begin(), parts_greedy.end());
    CHECK(unique_greedy.size() == 3, "Greedy produces exactly 3 partitions");

    // Test unified partition interface
    auto parts_unified = decomposer.partition(2, coords, n_elem, {},
                                              DomainDecomposer::Method::RCB);
    CHECK(parts_unified.size() == n_elem, "Unified partition returns correct size");

    // Test single partition
    auto parts1 = decomposer.partition_rcb(1, coords, n_elem);
    bool all_zero = true;
    for (int p : parts1) if (p != 0) all_zero = false;
    CHECK(all_zero, "1-part partition assigns all to partition 0");
}

// ============================================================================
// Test 4: Parallel Contact Detection
// ============================================================================
void test_parallel_contact() {
    printf("\n--- Parallel Contact Detection ---\n");

    // Test AABB
    AABB box;
    box.expand(0.0, 0.0, 0.0);
    box.expand(1.0, 1.0, 1.0);
    CHECK(box.is_valid(), "AABB is valid after expansion");
    CHECK_NEAR(box.volume(), 1.0, 1e-10, "AABB volume = 1.0");

    Real cx, cy, cz;
    box.center(cx, cy, cz);
    CHECK_NEAR(cx, 0.5, 1e-10, "AABB center x = 0.5");
    CHECK_NEAR(cy, 0.5, 1e-10, "AABB center y = 0.5");

    // Test AABB intersection
    AABB box2;
    box2.expand(0.5, 0.5, 0.5);
    box2.expand(1.5, 1.5, 1.5);
    CHECK(box.intersects(box2), "Overlapping AABBs intersect");

    AABB box3;
    box3.expand(2.0, 2.0, 2.0);
    box3.expand(3.0, 3.0, 3.0);
    CHECK(!box.intersects(box3), "Non-overlapping AABBs do not intersect");

    // Test inflate
    AABB box4 = box3;
    box4.inflate(1.5);
    CHECK(box.intersects(box4), "Inflated AABB now intersects");

    // Test contact detection
    ParallelContactDetector detector;

    // Surface A: two faces at z=0
    std::vector<AABB> surf_a = {
        AABB(), AABB()
    };
    surf_a[0].expand(0.0, 0.0, -0.1);
    surf_a[0].expand(1.0, 1.0, 0.1);
    surf_a[1].expand(1.0, 0.0, -0.1);
    surf_a[1].expand(2.0, 1.0, 0.1);

    // Surface B: two faces at z=0.15 (close to A)
    std::vector<AABB> surf_b = {
        AABB(), AABB()
    };
    surf_b[0].expand(0.0, 0.0, 0.05);
    surf_b[0].expand(1.0, 1.0, 0.25);
    surf_b[1].expand(3.0, 3.0, 3.0);  // far away
    surf_b[1].expand(4.0, 4.0, 4.0);

    std::vector<Index> a_ids = {10, 11};
    std::vector<Index> b_ids = {20, 21};

    detector.initialize(surf_a, surf_b, a_ids, b_ids, 0, 1);
    detector.set_search_tolerance(0.1);

    auto contacts = detector.detect_contacts();
    // surf_a[0] should overlap with surf_b[0] (close in z), not surf_b[1] (far)
    // surf_a[1] might overlap with surf_b[0] depending on tolerance
    CHECK(contacts.size() >= 1, "At least 1 contact pair detected");

    // Check that the detected pair includes the close surfaces
    bool found_close_pair = false;
    for (const auto& cp : contacts) {
        if (cp.surface_a_id == 10 && cp.surface_b_id == 20) {
            found_close_pair = true;
            CHECK(cp.rank_a == 0, "Contact pair rank_a = 0");
            CHECK(cp.rank_b == 0, "Contact pair rank_b = 0");
        }
    }
    CHECK(found_close_pair, "Close surface pair (10,20) detected");

    // Check that far pair is not detected
    bool found_far_pair = false;
    for (const auto& cp : contacts) {
        if (cp.surface_a_id == 10 && cp.surface_b_id == 21) {
            found_far_pair = true;
        }
    }
    CHECK(!found_far_pair, "Far surface pair (10,21) not detected");
}

// ============================================================================
// Test 5: Load Balancer
// ============================================================================
void test_load_balancer() {
    printf("\n--- Load Balancer ---\n");

    LoadBalancer balancer;
    balancer.initialize(4, 1.15, 0);  // 4 ranks, 15% threshold

    CHECK(balancer.n_ranks() == 4, "Load balancer has 4 ranks");
    CHECK_NEAR(balancer.threshold(), 1.15, 1e-10, "Threshold = 1.15");

    // Register elements with imbalanced weights
    // Rank 0: elements 0-4, weight 10 each = 50
    // Rank 1: elements 5-7, weight 10 each = 30
    // Rank 2: elements 8-9, weight 10 each = 20
    // Rank 3: elements 10-14, weight 2 each = 10
    // Total: 110, avg: 27.5, max: 50, ratio: 1.818
    std::vector<Index> eids;
    std::vector<Real> weights;
    std::vector<int> ranks;

    for (int i = 0; i < 5; ++i) { eids.push_back(i); weights.push_back(10.0); ranks.push_back(0); }
    for (int i = 5; i < 8; ++i) { eids.push_back(i); weights.push_back(10.0); ranks.push_back(1); }
    for (int i = 8; i < 10; ++i) { eids.push_back(i); weights.push_back(10.0); ranks.push_back(2); }
    for (int i = 10; i < 15; ++i) { eids.push_back(i); weights.push_back(2.0); ranks.push_back(3); }

    balancer.register_elements(eids, weights, ranks);

    Real imbalance = balancer.compute_imbalance();
    CHECK_NEAR(imbalance, 50.0 / 27.5, 0.01, "Imbalance = 50/27.5 = 1.818");
    CHECK(imbalance > 1.15, "Imbalance exceeds threshold");

    // Generate migration plan
    auto plan = balancer.generate_migration_plan();
    CHECK(plan.should_migrate, "Migration plan says migration needed");
    CHECK(plan.total_migrations() > 0, "Migration plan has entries");
    CHECK(plan.imbalance_after < plan.imbalance_before,
          "Migration plan reduces imbalance");

    // Verify migration entries move elements from overloaded rank
    bool moves_from_rank0 = false;
    for (const auto& e : plan.entries) {
        if (e.from_rank == 0) moves_from_rank0 = true;
    }
    CHECK(moves_from_rank0, "Migration moves elements from most overloaded rank");

    // Execute migration
    balancer.execute_migration(plan);
    Real new_imbalance = balancer.compute_imbalance();
    CHECK(new_imbalance < imbalance, "Imbalance reduced after migration");

    // Test balanced case: all ranks equal
    LoadBalancer balanced;
    balanced.initialize(2, 1.1, 0);
    std::vector<Index> b_eids = {0, 1, 2, 3};
    std::vector<Real> b_weights = {10.0, 10.0, 10.0, 10.0};
    std::vector<int> b_ranks = {0, 0, 1, 1};
    balanced.register_elements(b_eids, b_weights, b_ranks);

    CHECK_NEAR(balanced.compute_imbalance(), 1.0, 0.01, "Balanced case: imbalance = 1.0");
    auto balanced_plan = balanced.generate_migration_plan();
    CHECK(!balanced_plan.should_migrate, "No migration needed when balanced");
}

// ============================================================================
// Test 6: Scalability Benchmark
// ============================================================================
void test_scalability_benchmark() {
    printf("\n--- Scalability Benchmark ---\n");

    ScalabilityBenchmark bench;

    // Register phases
    bench.register_phase("assembly");
    bench.register_phase("solve");
    bench.register_phase("comm_exchange");

    // Run a simulated benchmark
    auto result = bench.run_benchmark(1000, 1, [&](Index n) {
        bench.start_phase("assembly");
        // Simulate work
        volatile double sum = 0.0;
        for (Index i = 0; i < n; ++i) sum += static_cast<double>(i);
        bench.stop_phase("assembly");

        bench.start_phase("solve");
        for (Index i = 0; i < n * 2; ++i) sum += std::sqrt(static_cast<double>(i));
        bench.stop_phase("solve");

        bench.start_phase("comm_exchange");
        for (Index i = 0; i < n / 10; ++i) sum += static_cast<double>(i);
        bench.stop_phase("comm_exchange");
    });

    CHECK(result.problem_size == 1000, "Benchmark problem size = 1000");
    CHECK(result.n_ranks == 1, "Benchmark n_ranks = 1");
    CHECK(result.total_time > 0.0, "Benchmark total time > 0");
    CHECK(result.computation_time > 0.0, "Benchmark computation time > 0");
    CHECK(result.phase_times.count("assembly") > 0, "Assembly phase recorded");
    CHECK(result.phase_times.count("solve") > 0, "Solve phase recorded");

    // Compute scaling metrics
    bench.compute_scaling_metrics();
    CHECK_NEAR(result.speedup, 1.0, 0.01, "Single-rank speedup = 1.0");
    // After compute_scaling_metrics, check stored results
    const auto& results = bench.results();
    CHECK(results.size() == 1, "One benchmark result stored");
    CHECK_NEAR(results[0].efficiency, 1.0, 0.01, "Single-rank efficiency = 1.0");

    // Add simulated multi-rank results
    BenchmarkResult r2;
    r2.problem_size = 1000;
    r2.n_ranks = 2;
    r2.total_time = result.total_time * 0.55;  // 55% of single-rank time
    r2.computation_time = r2.total_time * 0.9;
    r2.communication_time = r2.total_time * 0.1;
    bench.add_result(r2);

    BenchmarkResult r4;
    r4.problem_size = 1000;
    r4.n_ranks = 4;
    r4.total_time = result.total_time * 0.3;  // 30% of single-rank time
    r4.computation_time = r4.total_time * 0.8;
    r4.communication_time = r4.total_time * 0.2;
    bench.add_result(r4);

    bench.compute_scaling_metrics();

    const auto& all_results = bench.results();
    CHECK(all_results.size() == 3, "Three benchmark results stored");

    // 2-rank speedup should be ~1.82
    CHECK(all_results[1].speedup > 1.5, "2-rank speedup > 1.5");
    // 4-rank speedup should be ~3.33
    CHECK(all_results[2].speedup > 2.5, "4-rank speedup > 2.5");

    // Communication ratio for 4-rank case
    CHECK_NEAR(all_results[2].comm_to_comp_ratio, 0.25, 0.01,
               "4-rank comm/comp ratio = 0.25");

    // Generate report
    std::string report = bench.report();
    CHECK(report.size() > 100, "Report is non-trivial length");
    CHECK(report.find("Scalability") != std::string::npos, "Report contains header");
    CHECK(report.find("Speedup") != std::string::npos, "Report contains Speedup column");

    printf("--- Benchmark Report ---\n%s", report.c_str());
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("=== Wave 17 MPI Completion Test ===\n\n");

    test_csr_matrix();
    test_distributed_assembler();
    test_ghost_exchanger();
    test_domain_decomposer();
    test_parallel_contact();
    test_load_balancer();
    test_scalability_benchmark();

    printf("\nResults: %d/%d tests passed\n", pass_count, pass_count + fail_count);
    return fail_count > 0 ? 1 : 0;
}
