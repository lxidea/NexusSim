/**
 * @file mesh_partition_test.cpp
 * @brief Test suite for mesh partitioning
 *
 * Tests:
 * 1. RCB partitioning (2 parts)
 * 2. RCB partitioning (4 parts)
 * 3. RCB partitioning (8 parts)
 * 4. Load balance quality
 * 5. Ghost layer generation
 * 6. Communication patterns
 * 7. Global-to-local mapping
 * 8. Large mesh partitioning
 */

#include <nexussim/discretization/mesh_partition.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>

using namespace nxs;
using namespace nxs::discretization;

int total_tests = 0;
int passed_tests = 0;

void check(bool condition, const std::string& test_name) {
    total_tests++;
    if (condition) {
        passed_tests++;
        std::cout << "  [PASS] " << test_name << "\n";
    } else {
        std::cout << "  [FAIL] " << test_name << "\n";
    }
}

void check_near(Real a, Real b, Real tol, const std::string& test_name) {
    check(std::abs(a - b) < tol, test_name);
}

/**
 * @brief Create structured hex mesh for testing
 */
void create_hex_mesh(int nx, int ny, int nz,
                     std::vector<Real>& coords,
                     std::vector<Index>& connectivity) {
    // Nodes
    int num_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    coords.resize(num_nodes * 3);

    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                int node = i + (nx + 1) * (j + (ny + 1) * k);
                coords[node * 3 + 0] = static_cast<Real>(i);
                coords[node * 3 + 1] = static_cast<Real>(j);
                coords[node * 3 + 2] = static_cast<Real>(k);
            }
        }
    }

    // Elements (hex8)
    int num_elements = nx * ny * nz;
    connectivity.resize(num_elements * 8);

    auto node_idx = [nx, ny](int i, int j, int k) {
        return i + (nx + 1) * (j + (ny + 1) * k);
    };

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int elem = i + nx * (j + ny * k);
                connectivity[elem * 8 + 0] = node_idx(i, j, k);
                connectivity[elem * 8 + 1] = node_idx(i + 1, j, k);
                connectivity[elem * 8 + 2] = node_idx(i + 1, j + 1, k);
                connectivity[elem * 8 + 3] = node_idx(i, j + 1, k);
                connectivity[elem * 8 + 4] = node_idx(i, j, k + 1);
                connectivity[elem * 8 + 5] = node_idx(i + 1, j, k + 1);
                connectivity[elem * 8 + 6] = node_idx(i + 1, j + 1, k + 1);
                connectivity[elem * 8 + 7] = node_idx(i, j + 1, k + 1);
            }
        }
    }
}

// ============================================================================
// Test 1: RCB Partitioning (2 parts)
// ============================================================================

void test_rcb_2_parts() {
    std::cout << "\n=== Test 1: RCB Partitioning (2 parts) ===\n";

    std::vector<Real> coords;
    std::vector<Index> connectivity;
    create_hex_mesh(4, 4, 4, coords, connectivity);

    int num_nodes = 5 * 5 * 5;  // 125
    int num_elements = 4 * 4 * 4;  // 64
    int nodes_per_elem = 8;

    RCBPartitioner partitioner;
    auto partition = partitioner.partition_elements(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, nodes_per_elem, 2);

    // Count elements per partition
    int count0 = 0, count1 = 0;
    for (int e = 0; e < num_elements; ++e) {
        if (partition[e] == 0) count0++;
        else if (partition[e] == 1) count1++;
    }

    std::cout << "  Elements: " << num_elements << "\n";
    std::cout << "  Part 0: " << count0 << " elements\n";
    std::cout << "  Part 1: " << count1 << " elements\n";

    check(count0 + count1 == num_elements, "All elements assigned");
    check(count0 == 32 && count1 == 32, "Equal split (32/32)");
}

// ============================================================================
// Test 2: RCB Partitioning (4 parts)
// ============================================================================

void test_rcb_4_parts() {
    std::cout << "\n=== Test 2: RCB Partitioning (4 parts) ===\n";

    std::vector<Real> coords;
    std::vector<Index> connectivity;
    create_hex_mesh(4, 4, 4, coords, connectivity);

    int num_nodes = 5 * 5 * 5;
    int num_elements = 4 * 4 * 4;
    int nodes_per_elem = 8;

    RCBPartitioner partitioner;
    auto partition = partitioner.partition_elements(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, nodes_per_elem, 4);

    // Count elements per partition
    std::vector<int> counts(4, 0);
    for (int e = 0; e < num_elements; ++e) {
        counts[partition[e]]++;
    }

    std::cout << "  Part counts: ";
    for (int p = 0; p < 4; ++p) {
        std::cout << counts[p] << " ";
    }
    std::cout << "\n";

    int total = 0;
    for (int c : counts) total += c;
    check(total == num_elements, "All elements assigned");

    // Check balance (each part should have ~16 elements)
    bool balanced = true;
    for (int c : counts) {
        if (c < 14 || c > 18) balanced = false;  // Allow ±2 imbalance
    }
    check(balanced, "Balanced within tolerance");
}

// ============================================================================
// Test 3: RCB Partitioning (8 parts)
// ============================================================================

void test_rcb_8_parts() {
    std::cout << "\n=== Test 3: RCB Partitioning (8 parts) ===\n";

    std::vector<Real> coords;
    std::vector<Index> connectivity;
    create_hex_mesh(4, 4, 4, coords, connectivity);

    int num_nodes = 5 * 5 * 5;
    int num_elements = 4 * 4 * 4;
    int nodes_per_elem = 8;

    RCBPartitioner partitioner;
    auto partition = partitioner.partition_elements(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, nodes_per_elem, 8);

    std::vector<int> counts(8, 0);
    for (int e = 0; e < num_elements; ++e) {
        counts[partition[e]]++;
    }

    std::cout << "  Part counts: ";
    for (int p = 0; p < 8; ++p) {
        std::cout << counts[p] << " ";
    }
    std::cout << "\n";

    // Each part should have 8 elements (64/8)
    bool perfect_split = true;
    for (int c : counts) {
        if (c != 8) perfect_split = false;
    }
    check(perfect_split, "Perfect split (8 elements each)");
}

// ============================================================================
// Test 4: Load Balance Quality
// ============================================================================

void test_load_balance() {
    std::cout << "\n=== Test 4: Load Balance Quality ===\n";

    std::vector<Real> coords;
    std::vector<Index> connectivity;
    create_hex_mesh(10, 10, 10, coords, connectivity);

    int num_nodes = 11 * 11 * 11;
    int num_elements = 10 * 10 * 10;  // 1000
    int nodes_per_elem = 8;
    int num_parts = 8;

    RCBPartitioner partitioner;
    auto partition = partitioner.partition_elements(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, nodes_per_elem, num_parts);

    auto quality = partitioner.analyze_quality(
        partition, connectivity.data(),
        num_elements, num_nodes, nodes_per_elem, num_parts);

    quality.print(std::cout);

    check(quality.element_imbalance < 1.2, "Element imbalance < 1.2x");
    check(quality.communication_ratio < 0.5, "Interface nodes < 50%");
}

// ============================================================================
// Test 5: Ghost Layer Generation
// ============================================================================

void test_ghost_layer() {
    std::cout << "\n=== Test 5: Ghost Layer Generation ===\n";

    std::vector<Real> coords;
    std::vector<Index> connectivity;
    create_hex_mesh(4, 4, 2, coords, connectivity);

    int num_nodes = 5 * 5 * 3;  // 75
    int num_elements = 4 * 4 * 2;  // 32
    int nodes_per_elem = 8;
    int num_parts = 2;

    RCBPartitioner partitioner;

    // Create partition for rank 0
    auto part0 = partitioner.create_partition(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, nodes_per_elem, num_parts, 0);

    // Create partition for rank 1
    auto part1 = partitioner.create_partition(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, nodes_per_elem, num_parts, 1);

    std::cout << "  Rank 0:\n";
    std::cout << "    Local elements: " << part0.num_local_elements() << "\n";
    std::cout << "    Local nodes: " << part0.num_local_nodes() << "\n";
    std::cout << "    Ghost nodes: " << part0.num_ghost_nodes() << "\n";

    std::cout << "  Rank 1:\n";
    std::cout << "    Local elements: " << part1.num_local_elements() << "\n";
    std::cout << "    Local nodes: " << part1.num_local_nodes() << "\n";
    std::cout << "    Ghost nodes: " << part1.num_ghost_nodes() << "\n";

    check(part0.num_local_elements() + part1.num_local_elements() == static_cast<Index>(num_elements),
          "Total elements preserved");

    check(part0.num_ghost_nodes() > 0, "Rank 0 has ghost nodes");
    check(part1.num_ghost_nodes() > 0, "Rank 1 has ghost nodes");

    // Ghosts on one rank should be owned by the other
    for (Index ghost : part0.ghost_nodes) {
        check(part0.node_owner[ghost] == 1, "Rank 0 ghost owned by rank 1");
        break;  // Just check first one
    }
}

// ============================================================================
// Test 6: Communication Patterns
// ============================================================================

void test_comm_patterns() {
    std::cout << "\n=== Test 6: Communication Patterns ===\n";

    std::vector<Real> coords;
    std::vector<Index> connectivity;
    create_hex_mesh(4, 4, 4, coords, connectivity);

    int num_nodes = 5 * 5 * 5;
    int num_elements = 4 * 4 * 4;
    int nodes_per_elem = 8;
    int num_parts = 4;

    RCBPartitioner partitioner;

    auto part0 = partitioner.create_partition(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, nodes_per_elem, num_parts, 0);

    std::cout << "  Rank 0 communication patterns:\n";
    for (const auto& pattern : part0.comm_patterns) {
        std::cout << "    -> Rank " << pattern.neighbor_rank
                  << ": send " << pattern.send_nodes.size()
                  << " nodes, recv " << pattern.recv_nodes.size() << " nodes\n";
    }

    check(part0.comm_patterns.size() > 0, "Has neighbor communications");

    // Verify send/recv consistency
    bool consistent = true;
    for (const auto& pattern : part0.comm_patterns) {
        // For each node we send, it should be local to us
        for (Index local : pattern.send_nodes) {
            if (local >= part0.local_nodes.size()) {
                consistent = false;
                break;
            }
        }
        // For each node we receive, it should be a ghost
        for (Index local : pattern.recv_nodes) {
            if (local < part0.local_nodes.size()) {
                consistent = false;
                break;
            }
        }
    }
    check(consistent, "Send/recv node indices consistent");
}

// ============================================================================
// Test 7: Global-to-Local Mapping
// ============================================================================

void test_global_local_mapping() {
    std::cout << "\n=== Test 7: Global-to-Local Mapping ===\n";

    std::vector<Real> coords;
    std::vector<Index> connectivity;
    create_hex_mesh(4, 4, 4, coords, connectivity);

    int num_nodes = 5 * 5 * 5;
    int num_elements = 4 * 4 * 4;
    int nodes_per_elem = 8;
    int num_parts = 2;

    RCBPartitioner partitioner;

    auto part = partitioner.create_partition(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, nodes_per_elem, num_parts, 0);

    // Test element mapping
    bool elem_mapping_ok = true;
    for (Index local = 0; local < part.num_local_elements(); ++local) {
        Index global = part.local_to_global_elem[local];
        if (part.global_to_local_elem[global] != local) {
            elem_mapping_ok = false;
            break;
        }
    }
    check(elem_mapping_ok, "Element global<->local bijection");

    // Test node mapping
    bool node_mapping_ok = true;
    Index total_local_nodes = part.num_local_nodes() + part.num_ghost_nodes();
    for (Index local = 0; local < total_local_nodes; ++local) {
        Index global = part.local_to_global_node[local];
        if (part.global_to_local_node[global] != local) {
            node_mapping_ok = false;
            break;
        }
    }
    check(node_mapping_ok, "Node global<->local bijection");

    // Verify element connectivity can be converted to local
    bool connectivity_ok = true;
    for (Index local_elem = 0; local_elem < part.num_local_elements(); ++local_elem) {
        Index global_elem = part.local_to_global_elem[local_elem];
        for (int n = 0; n < nodes_per_elem; ++n) {
            Index global_node = connectivity[global_elem * nodes_per_elem + n];
            if (part.global_to_local_node.count(global_node) == 0) {
                connectivity_ok = false;
                break;
            }
        }
    }
    check(connectivity_ok, "All element nodes in local map");
}

// ============================================================================
// Test 8: Large Mesh Partitioning Performance
// ============================================================================

void test_large_mesh() {
    std::cout << "\n=== Test 8: Large Mesh Partitioning ===\n";

    std::vector<Real> coords;
    std::vector<Index> connectivity;
    create_hex_mesh(20, 20, 20, coords, connectivity);

    int num_nodes = 21 * 21 * 21;  // 9261
    int num_elements = 20 * 20 * 20;  // 8000
    int nodes_per_elem = 8;
    int num_parts = 16;

    std::cout << "  Nodes: " << num_nodes << "\n";
    std::cout << "  Elements: " << num_elements << "\n";
    std::cout << "  Partitions: " << num_parts << "\n";

    RCBPartitioner partitioner;

    auto start = std::chrono::high_resolution_clock::now();

    auto partition = partitioner.partition_elements(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, nodes_per_elem, num_parts);

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "  Partition time: " << time_ms << " ms\n";

    auto quality = partitioner.analyze_quality(
        partition, connectivity.data(),
        num_elements, num_nodes, nodes_per_elem, num_parts);

    std::cout << "  Element imbalance: " << quality.element_imbalance << "x\n";
    std::cout << "  Interface ratio: " << (100.0 * quality.communication_ratio) << "%\n";

    check(time_ms < 1000.0, "Partition time < 1s");
    check(quality.element_imbalance < 1.1, "Element imbalance < 1.1x");

    // Create full partition for one rank
    start = std::chrono::high_resolution_clock::now();

    auto full_part = partitioner.create_partition(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, nodes_per_elem, num_parts, 0);

    end = std::chrono::high_resolution_clock::now();
    time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "  Full partition create time: " << time_ms << " ms\n";
    std::cout << "  Rank 0 local elements: " << full_part.num_local_elements() << "\n";
    std::cout << "  Rank 0 local nodes: " << full_part.num_local_nodes() << "\n";
    std::cout << "  Rank 0 ghost nodes: " << full_part.num_ghost_nodes() << "\n";

    check(full_part.num_local_elements() > 0, "Has local elements");
    check(full_part.num_local_nodes() > 0, "Has local nodes");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Mesh Partition Test Suite ===\n";
    std::cout << std::setprecision(4) << std::fixed;

    test_rcb_2_parts();
    test_rcb_4_parts();
    test_rcb_8_parts();
    test_load_balance();
    test_ghost_layer();
    test_comm_patterns();
    test_global_local_mapping();
    test_large_mesh();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed_tests << "/" << total_tests << " tests passed\n";
    std::cout << "========================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
