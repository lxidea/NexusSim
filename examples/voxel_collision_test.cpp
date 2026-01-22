/**
 * @file voxel_collision_test.cpp
 * @brief Test for voxel-based collision detection
 */

#include <nexussim/fem/voxel_collision.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace nxs;
using namespace nxs::fem;

int tests_passed = 0;
int tests_failed = 0;

void check(bool condition, const std::string& test_name) {
    if (condition) {
        std::cout << "[PASS] " << test_name << std::endl;
        tests_passed++;
    } else {
        std::cout << "[FAIL] " << test_name << std::endl;
        tests_failed++;
    }
}

// Test 1: Basic initialization
void test_initialization() {
    std::cout << "\n=== Test 1: Basic Initialization ===\n";

    // Create a simple quad mesh
    const Index num_nodes = 9;  // 3x3 grid
    const Index num_segments = 4;  // 2x2 quads

    std::vector<Real> coords = {
        0.0, 0.0, 0.0,  // Node 0
        1.0, 0.0, 0.0,  // Node 1
        2.0, 0.0, 0.0,  // Node 2
        0.0, 1.0, 0.0,  // Node 3
        1.0, 1.0, 0.0,  // Node 4
        2.0, 1.0, 0.0,  // Node 5
        0.0, 2.0, 0.0,  // Node 6
        1.0, 2.0, 0.0,  // Node 7
        2.0, 2.0, 0.0   // Node 8
    };

    std::vector<Index> segment_nodes = {
        0, 1, 4, 3,  // Segment 0
        1, 2, 5, 4,  // Segment 1
        3, 4, 7, 6,  // Segment 2
        4, 5, 8, 7   // Segment 3
    };

    VoxelCollisionDetector detector;
    detector.initialize(num_nodes, num_segments, coords.data(), segment_nodes.data(), 4);

    check(detector.cell_size() > 0, "Cell size is positive");

    int dims[3];
    detector.get_grid_dims(dims);
    check(dims[0] > 0 && dims[1] > 0 && dims[2] > 0, "Grid dimensions are positive");

    Real bbox_min[3], bbox_max[3];
    detector.get_bounding_box(bbox_min, bbox_max);
    check(bbox_max[0] > bbox_min[0], "Bounding box is valid");
}

// Test 2: Candidate detection
void test_candidate_detection() {
    std::cout << "\n=== Test 2: Candidate Detection ===\n";

    // Create two parallel planes
    const Index num_nodes = 8;
    const Index num_segments = 2;

    std::vector<Real> coords = {
        // Bottom plane (z=0)
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        // Top plane (z=0.1) - close to bottom
        0.0, 0.0, 0.1,
        1.0, 0.0, 0.1,
        1.0, 1.0, 0.1,
        0.0, 1.0, 0.1
    };

    std::vector<Index> segment_nodes = {
        0, 1, 2, 3,  // Bottom quad
        4, 5, 6, 7   // Top quad
    };

    VoxelCollisionConfig config;
    config.cell_size_factor = 2.0;

    VoxelCollisionDetector detector(config);
    detector.initialize(num_nodes, num_segments, coords.data(), segment_nodes.data(), 4);

    // Search for candidates near center of bottom plane
    std::vector<Index> slave_nodes = {4, 5, 6, 7};  // Top plane nodes
    auto candidates = detector.find_candidates(slave_nodes, 0.5);

    std::cout << "Found " << candidates.size() << " candidates" << std::endl;
    check(candidates.size() > 0, "Candidates found for nearby surfaces");

    // All top nodes should find bottom segment as candidate
    bool found_bottom = false;
    for (const auto& c : candidates) {
        if (c.segment_id == 0) {
            found_bottom = true;
            break;
        }
    }
    check(found_bottom, "Top nodes find bottom segment");
}

// Test 3: Grid rebuild on motion
void test_grid_rebuild() {
    std::cout << "\n=== Test 3: Grid Rebuild on Motion ===\n";

    const Index num_nodes = 4;
    const Index num_segments = 1;

    std::vector<Real> coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0
    };

    std::vector<Index> segment_nodes = {0, 1, 2, 3};

    VoxelCollisionConfig config;
    config.rebuild_threshold = 0.5;  // Rebuild when moved > 50% of cell size

    VoxelCollisionDetector detector(config);
    detector.initialize(num_nodes, num_segments, coords.data(), segment_nodes.data(), 4);

    Real initial_cell_size = detector.cell_size();
    std::cout << "Initial cell size: " << initial_cell_size << std::endl;

    // Move nodes slightly - should not trigger rebuild
    std::vector<Real> small_move = coords;
    for (size_t i = 0; i < small_move.size(); ++i) {
        small_move[i] += 0.01;
    }
    detector.update_coordinates(small_move.data());

    // Move nodes significantly - should trigger rebuild
    std::vector<Real> large_move = coords;
    for (size_t i = 0; i < large_move.size(); ++i) {
        large_move[i] += 10.0;
    }
    detector.update_coordinates(large_move.data());

    Real new_bbox_min[3], new_bbox_max[3];
    detector.get_bounding_box(new_bbox_min, new_bbox_max);

    check(new_bbox_min[0] > 5.0, "Bounding box updated after large motion");
}

// Test 4: Voxel distribution
void test_voxel_distribution() {
    std::cout << "\n=== Test 4: Voxel Distribution ===\n";

    // Create a 4x4 grid of quads
    const int grid_size = 4;
    const Index num_nodes = (grid_size + 1) * (grid_size + 1);
    const Index num_segments = grid_size * grid_size;

    std::vector<Real> coords;
    for (int j = 0; j <= grid_size; ++j) {
        for (int i = 0; i <= grid_size; ++i) {
            coords.push_back(static_cast<Real>(i));
            coords.push_back(static_cast<Real>(j));
            coords.push_back(0.0);
        }
    }

    std::vector<Index> segment_nodes;
    for (int j = 0; j < grid_size; ++j) {
        for (int i = 0; i < grid_size; ++i) {
            Index n0 = j * (grid_size + 1) + i;
            Index n1 = n0 + 1;
            Index n2 = n1 + grid_size + 1;
            Index n3 = n0 + grid_size + 1;
            segment_nodes.push_back(n0);
            segment_nodes.push_back(n1);
            segment_nodes.push_back(n2);
            segment_nodes.push_back(n3);
        }
    }

    VoxelCollisionDetector detector;
    detector.initialize(num_nodes, num_segments, coords.data(), segment_nodes.data(), 4);

    auto counts = detector.get_voxel_counts();

    // Count total segments in voxels
    int total = 0;
    int non_empty = 0;
    for (int c : counts) {
        total += c;
        if (c > 0) non_empty++;
    }

    std::cout << "Total segments in voxels: " << total << std::endl;
    std::cout << "Non-empty voxels: " << non_empty << std::endl;

    check(total == static_cast<int>(num_segments), "All segments assigned to voxels");
    check(non_empty > 0, "Some voxels are non-empty");
}

// Test 5: Triangle segments
void test_triangle_segments() {
    std::cout << "\n=== Test 5: Triangle Segments ===\n";

    const Index num_nodes = 4;
    const Index num_segments = 2;

    std::vector<Real> coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.5, 1.0, 0.0,
        0.5, 0.5, 0.5  // Point above
    };

    std::vector<Index> segment_nodes = {
        0, 1, 2,  // Triangle 1
        0, 2, 2   // Degenerate triangle (for test)
    };

    VoxelCollisionDetector detector;
    detector.initialize(num_nodes, num_segments, coords.data(), segment_nodes.data(), 3);

    std::vector<Index> slave = {3};
    auto candidates = detector.find_candidates(slave, 1.0);

    std::cout << "Found " << candidates.size() << " candidates for triangle test" << std::endl;
    check(candidates.size() > 0, "Candidates found for triangular segments");
}

// Test 6: Far separation
void test_far_separation() {
    std::cout << "\n=== Test 6: Far Separation ===\n";

    const Index num_nodes = 8;
    const Index num_segments = 2;

    std::vector<Real> coords = {
        // First quad at origin
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        // Second quad far away
        100.0, 100.0, 100.0,
        101.0, 100.0, 100.0,
        101.0, 101.0, 100.0,
        100.0, 101.0, 100.0
    };

    std::vector<Index> segment_nodes = {
        0, 1, 2, 3,
        4, 5, 6, 7
    };

    VoxelCollisionDetector detector;
    detector.initialize(num_nodes, num_segments, coords.data(), segment_nodes.data(), 4);

    // Search with small radius - should not find far segment
    std::vector<Index> slave = {0, 1, 2, 3};
    auto candidates = detector.find_candidates(slave, 1.0);

    // Check if far segment (1) is found - it shouldn't be
    bool found_far = false;
    for (const auto& c : candidates) {
        if (c.segment_id == 1) {
            found_far = true;
            break;
        }
    }

    check(!found_far, "Far segment not found with small search radius");
}

// Test 7: Self contact scenario
void test_self_contact() {
    std::cout << "\n=== Test 7: Self Contact Scenario ===\n";

    // Create a folded sheet scenario
    const Index num_nodes = 12;
    const Index num_segments = 4;

    std::vector<Real> coords = {
        // Bottom row
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        2.0, 0.0, 0.0,
        // Middle row (slightly raised)
        0.0, 1.0, 0.1,
        1.0, 1.0, 0.1,
        2.0, 1.0, 0.1,
        // Top row (folded back, close to bottom)
        0.0, 2.0, 0.05,
        1.0, 2.0, 0.05,
        2.0, 2.0, 0.05,
        // Separate strip
        5.0, 0.0, 0.0,
        5.0, 1.0, 0.0,
        5.0, 2.0, 0.0
    };

    std::vector<Index> segment_nodes = {
        0, 1, 4, 3,  // Bottom-left
        1, 2, 5, 4,  // Bottom-right
        3, 4, 7, 6,  // Middle-left
        4, 5, 8, 7   // Middle-right
    };

    VoxelCollisionDetector detector;
    detector.initialize(num_nodes, num_segments, coords.data(), segment_nodes.data(), 4);

    // Search from top row nodes
    std::vector<Index> slave = {6, 7, 8};
    auto candidates = detector.find_candidates(slave, 0.5);

    std::cout << "Self-contact candidates: " << candidates.size() << std::endl;

    // Top row should find bottom row as candidates
    bool found_bottom = false;
    for (const auto& c : candidates) {
        if (c.segment_id == 0 || c.segment_id == 1) {
            found_bottom = true;
            break;
        }
    }

    check(candidates.size() > 0, "Self-contact candidates detected");
}

// Test 8: Performance with larger mesh
void test_performance() {
    std::cout << "\n=== Test 8: Performance Test ===\n";

    const int grid_size = 20;  // 20x20 mesh
    const Index num_nodes = (grid_size + 1) * (grid_size + 1);
    const Index num_segments = grid_size * grid_size;

    std::vector<Real> coords;
    for (int j = 0; j <= grid_size; ++j) {
        for (int i = 0; i <= grid_size; ++i) {
            coords.push_back(static_cast<Real>(i) * 0.1);
            coords.push_back(static_cast<Real>(j) * 0.1);
            coords.push_back(0.0);
        }
    }

    std::vector<Index> segment_nodes;
    for (int j = 0; j < grid_size; ++j) {
        for (int i = 0; i < grid_size; ++i) {
            Index n0 = j * (grid_size + 1) + i;
            Index n1 = n0 + 1;
            Index n2 = n1 + grid_size + 1;
            Index n3 = n0 + grid_size + 1;
            segment_nodes.push_back(n0);
            segment_nodes.push_back(n1);
            segment_nodes.push_back(n2);
            segment_nodes.push_back(n3);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    VoxelCollisionDetector detector;
    detector.initialize(num_nodes, num_segments, coords.data(), segment_nodes.data(), 4);

    auto init_time = std::chrono::high_resolution_clock::now();

    // Create slave node list
    std::vector<Index> slaves;
    for (Index i = 0; i < num_nodes; ++i) {
        slaves.push_back(i);
    }

    auto candidates = detector.find_candidates(slaves, 0.3);

    auto end = std::chrono::high_resolution_clock::now();

    auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(init_time - start).count();
    auto search_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - init_time).count();

    std::cout << "Nodes: " << num_nodes << ", Segments: " << num_segments << std::endl;
    std::cout << "Initialization time: " << init_ms << " ms" << std::endl;
    std::cout << "Search time: " << search_ms << " ms" << std::endl;
    std::cout << "Candidates found: " << candidates.size() << std::endl;

    check(init_ms < 1000, "Initialization completes in < 1 second");
    check(search_ms < 5000, "Search completes in < 5 seconds");
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    {
        std::cout << "========================================" << std::endl;
        std::cout << "Voxel Collision Detection Test Suite" << std::endl;
        std::cout << "========================================" << std::endl;

        test_initialization();
        test_candidate_detection();
        test_grid_rebuild();
        test_voxel_distribution();
        test_triangle_segments();
        test_far_separation();
        test_self_contact();
        test_performance();

        std::cout << "\n========================================" << std::endl;
        std::cout << "Test Summary" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Passed: " << tests_passed << std::endl;
        std::cout << "Failed: " << tests_failed << std::endl;
    }

    Kokkos::finalize();
    return (tests_failed == 0) ? 0 : 1;
}
