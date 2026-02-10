/**
 * @file mpi_partition_test.cpp
 * @brief Test MPI mesh partitioning and ghost exchange
 *
 * Tests:
 * 1. RCB partitioning algorithm
 * 2. Ghost node detection
 * 3. Communication pattern generation
 * 4. Ghost exchange (when MPI enabled)
 */

#include <nexussim/core/core.hpp>
#include <nexussim/core/mpi.hpp>
#include <nexussim/discretization/mesh_partition.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace nxs;
using namespace nxs::discretization;

// Test result tracking
static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { \
        std::cout << "[PASS] " << msg << "\n"; \
        tests_passed++; \
    } else { \
        std::cout << "[FAIL] " << msg << "\n"; \
        tests_failed++; \
    } \
} while(0)

/**
 * @brief Create a simple 3D mesh for testing
 * @param nx, ny, nz Number of elements in each direction
 * @param coords Output: node coordinates (3 * num_nodes)
 * @param connectivity Output: element connectivity (8 * num_elements for Hex8)
 */
void create_test_mesh(int nx, int ny, int nz,
                      std::vector<Real>& coords,
                      std::vector<Index>& connectivity)
{
    int num_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    int num_elements = nx * ny * nz;

    coords.resize(num_nodes * 3);
    connectivity.resize(num_elements * 8);

    // Create nodes
    Index node_id = 0;
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                coords[node_id * 3 + 0] = static_cast<Real>(i);
                coords[node_id * 3 + 1] = static_cast<Real>(j);
                coords[node_id * 3 + 2] = static_cast<Real>(k);
                node_id++;
            }
        }
    }

    // Create Hex8 elements
    Index elem_id = 0;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                Index n0 = k * (ny + 1) * (nx + 1) + j * (nx + 1) + i;
                Index n1 = n0 + 1;
                Index n2 = n0 + (nx + 1) + 1;
                Index n3 = n0 + (nx + 1);
                Index n4 = n0 + (ny + 1) * (nx + 1);
                Index n5 = n4 + 1;
                Index n6 = n4 + (nx + 1) + 1;
                Index n7 = n4 + (nx + 1);

                connectivity[elem_id * 8 + 0] = n0;
                connectivity[elem_id * 8 + 1] = n1;
                connectivity[elem_id * 8 + 2] = n2;
                connectivity[elem_id * 8 + 3] = n3;
                connectivity[elem_id * 8 + 4] = n4;
                connectivity[elem_id * 8 + 5] = n5;
                connectivity[elem_id * 8 + 6] = n6;
                connectivity[elem_id * 8 + 7] = n7;
                elem_id++;
            }
        }
    }
}

// ============================================================================
// Test 1: RCB Partitioning
// ============================================================================

bool test_rcb_partitioning() {
    std::cout << "\n=== Test 1: RCB Partitioning ===\n";

    const int nx = 4, ny = 2, nz = 2;
    std::vector<Real> coords;
    std::vector<Index> connectivity;
    create_test_mesh(nx, ny, nz, coords, connectivity);

    Index num_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    Index num_elements = nx * ny * nz;

    std::cout << "Mesh: " << num_elements << " elements, " << num_nodes << " nodes\n";

    RCBPartitioner partitioner;

    // Test 2-way partition
    auto partition_2 = partitioner.partition_elements(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, 8, 2);

    CHECK(partition_2.size() == num_elements, "Partition vector has correct size");

    // Count elements per partition
    int count_0 = 0, count_1 = 0;
    for (int p : partition_2) {
        if (p == 0) count_0++;
        else if (p == 1) count_1++;
    }

    std::cout << "2-way partition: " << count_0 << " + " << count_1 << " elements\n";

    CHECK(count_0 + count_1 == static_cast<int>(num_elements), "All elements assigned");
    CHECK(count_0 > 0 && count_1 > 0, "Both partitions have elements");

    // Test 4-way partition
    auto partition_4 = partitioner.partition_elements(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, 8, 4);

    std::vector<int> counts(4, 0);
    for (int p : partition_4) {
        if (p >= 0 && p < 4) counts[p]++;
    }

    std::cout << "4-way partition: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << counts[i] << " ";
    }
    std::cout << "elements\n";

    int total = 0;
    bool all_have_elements = true;
    for (int c : counts) {
        total += c;
        if (c == 0) all_have_elements = false;
    }

    CHECK(total == static_cast<int>(num_elements), "All elements assigned in 4-way");
    CHECK(all_have_elements, "All 4 partitions have elements");

    return true;
}

// ============================================================================
// Test 2: Partition Quality
// ============================================================================

bool test_partition_quality() {
    std::cout << "\n=== Test 2: Partition Quality ===\n";

    const int nx = 8, ny = 4, nz = 4;
    std::vector<Real> coords;
    std::vector<Index> connectivity;
    create_test_mesh(nx, ny, nz, coords, connectivity);

    Index num_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    Index num_elements = nx * ny * nz;

    std::cout << "Mesh: " << num_elements << " elements, " << num_nodes << " nodes\n";

    RCBPartitioner partitioner;

    auto partition = partitioner.partition_elements(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, 8, 4);

    auto quality = partitioner.analyze_quality(
        partition, connectivity.data(),
        num_elements, num_nodes, 8, 4);

    quality.print();

    CHECK(quality.element_imbalance < 1.5, "Element imbalance < 1.5x");
    CHECK(quality.communication_ratio < 0.5, "Interface ratio < 50%");

    return true;
}

// ============================================================================
// Test 3: Full Partition with Ghost Nodes
// ============================================================================

bool test_full_partition() {
    std::cout << "\n=== Test 3: Full Partition with Ghost Nodes ===\n";

    const int nx = 4, ny = 2, nz = 2;
    std::vector<Real> coords;
    std::vector<Index> connectivity;
    create_test_mesh(nx, ny, nz, coords, connectivity);

    Index num_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    Index num_elements = nx * ny * nz;

    RCBPartitioner partitioner;

    // Create partition for rank 0
    auto part0 = partitioner.create_partition(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, 8, 2, 0);

    std::cout << "Rank 0 partition:\n";
    std::cout << "  Local elements: " << part0.num_local_elements() << "\n";
    std::cout << "  Local nodes: " << part0.num_local_nodes() << "\n";
    std::cout << "  Ghost nodes: " << part0.num_ghost_nodes() << "\n";
    std::cout << "  Comm patterns: " << part0.comm_patterns.size() << "\n";

    CHECK(part0.num_local_elements() > 0, "Rank 0 has local elements");
    CHECK(part0.num_local_nodes() > 0, "Rank 0 has local nodes");

    // Create partition for rank 1
    auto part1 = partitioner.create_partition(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, 8, 2, 1);

    std::cout << "Rank 1 partition:\n";
    std::cout << "  Local elements: " << part1.num_local_elements() << "\n";
    std::cout << "  Local nodes: " << part1.num_local_nodes() << "\n";
    std::cout << "  Ghost nodes: " << part1.num_ghost_nodes() << "\n";

    CHECK(part1.num_local_elements() > 0, "Rank 1 has local elements");
    CHECK(part1.num_local_nodes() > 0, "Rank 1 has local nodes");

    // Verify total elements
    CHECK(part0.num_local_elements() + part1.num_local_elements() == num_elements,
          "Total elements preserved");

    // Verify ghost nodes exist at interface
    bool has_interface = (part0.num_ghost_nodes() > 0 || part1.num_ghost_nodes() > 0);
    CHECK(has_interface, "Interface has ghost nodes");

    return true;
}

// ============================================================================
// Test 4: Global-to-Local Mapping
// ============================================================================

bool test_mapping() {
    std::cout << "\n=== Test 4: Global-to-Local Mapping ===\n";

    const int nx = 4, ny = 2, nz = 2;
    std::vector<Real> coords;
    std::vector<Index> connectivity;
    create_test_mesh(nx, ny, nz, coords, connectivity);

    Index num_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    Index num_elements = nx * ny * nz;

    RCBPartitioner partitioner;
    auto part = partitioner.create_partition(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, 8, 2, 0);

    // Test element mapping
    bool elem_mapping_valid = true;
    for (Index local = 0; local < part.local_to_global_elem.size(); ++local) {
        Index global = part.local_to_global_elem[local];
        if (part.global_to_local_elem.at(global) != local) {
            elem_mapping_valid = false;
            break;
        }
    }
    CHECK(elem_mapping_valid, "Element global-to-local mapping is consistent");

    // Test node mapping
    bool node_mapping_valid = true;
    for (Index local = 0; local < part.local_to_global_node.size(); ++local) {
        Index global = part.local_to_global_node[local];
        if (part.global_to_local_node.at(global) != local) {
            node_mapping_valid = false;
            break;
        }
    }
    CHECK(node_mapping_valid, "Node global-to-local mapping is consistent");

    return true;
}

// ============================================================================
// Test 5: Communication Pattern
// ============================================================================

bool test_comm_pattern() {
    std::cout << "\n=== Test 5: Communication Pattern ===\n";

    const int nx = 4, ny = 2, nz = 2;
    std::vector<Real> coords;
    std::vector<Index> connectivity;
    create_test_mesh(nx, ny, nz, coords, connectivity);

    Index num_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    Index num_elements = nx * ny * nz;

    RCBPartitioner partitioner;
    auto part0 = partitioner.create_partition(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, 8, 2, 0);

    std::cout << "Rank 0 communication patterns:\n";
    for (const auto& pattern : part0.comm_patterns) {
        std::cout << "  -> Rank " << pattern.neighbor_rank
                  << ": send " << pattern.send_nodes.size()
                  << ", recv " << pattern.recv_nodes.size() << " nodes\n";
    }

    // For a 2-way partition, rank 0 should communicate with rank 1
    bool found_neighbor = false;
    for (const auto& pattern : part0.comm_patterns) {
        if (pattern.neighbor_rank == 1) {
            found_neighbor = true;
            break;
        }
    }

    CHECK(part0.comm_patterns.size() > 0 || part0.ghost_nodes.empty(),
          "Comm patterns exist if there are ghost nodes");

    if (part0.ghost_nodes.size() > 0) {
        CHECK(found_neighbor, "Found communication with neighbor rank");
    }

    return true;
}

// ============================================================================
// Test 6: MPI Manager (Serial Mode)
// ============================================================================

bool test_mpi_manager() {
    std::cout << "\n=== Test 6: MPI Manager ===\n";

    auto& mpi = MPIManager::instance();

    // In serial mode (no MPI), these should still work
    std::cout << "MPI initialized: " << mpi.is_initialized() << "\n";
    std::cout << "Rank: " << mpi.rank() << "\n";
    std::cout << "Size: " << mpi.size() << "\n";
    std::cout << "Is root: " << mpi.is_root() << "\n";

    CHECK(mpi.rank() == 0, "Rank is 0 (serial mode)");
    CHECK(mpi.size() == 1, "Size is 1 (serial mode)");
    CHECK(mpi.is_root() == true, "Is root in serial mode");

    // Test serial allreduce
    Real send_val = 42.0;
    Real recv_val = 0.0;
    allreduce_sum(&send_val, &recv_val, 1);

    CHECK(recv_val == 42.0, "Serial allreduce_sum works");

    return true;
}

// ============================================================================
// Test 7: Ghost Exchange (Serial Stub)
// ============================================================================

bool test_ghost_exchange() {
    std::cout << "\n=== Test 7: Ghost Exchange (Serial) ===\n";

    const int nx = 4, ny = 2, nz = 2;
    std::vector<Real> coords;
    std::vector<Index> connectivity;
    create_test_mesh(nx, ny, nz, coords, connectivity);

    Index num_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    Index num_elements = nx * ny * nz;

    RCBPartitioner partitioner;
    auto part = partitioner.create_partition(
        coords.data(), connectivity.data(),
        num_nodes, num_elements, 8, 1, 0);  // Single partition

    GhostExchange exchange;
    exchange.initialize(part);

    // Create test field
    std::vector<Real> field(part.local_to_global_node.size() * 3, 0.0);
    for (std::size_t i = 0; i < field.size(); ++i) {
        field[i] = static_cast<Real>(i);
    }

    // Exchange should be no-op in serial mode
    exchange.exchange_vector(field, 3);

    // Verify field unchanged
    bool unchanged = true;
    for (std::size_t i = 0; i < field.size(); ++i) {
        if (std::abs(field[i] - static_cast<Real>(i)) > 1e-10) {
            unchanged = false;
            break;
        }
    }

    CHECK(unchanged, "Serial ghost exchange is no-op");

    return true;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "========================================\n";
    std::cout << "NexusSim MPI Partition Test Suite\n";
    std::cout << "========================================\n";

    // Initialize MPI if available
    auto& mpi = MPIManager::instance();
    mpi.initialize(&argc, &argv);

    test_rcb_partitioning();
    test_partition_quality();
    test_full_partition();
    test_mapping();
    test_comm_pattern();
    test_mpi_manager();
    test_ghost_exchange();

    mpi.finalize();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    std::cout << "========================================\n";

    return tests_failed > 0 ? 1 : 0;
}
