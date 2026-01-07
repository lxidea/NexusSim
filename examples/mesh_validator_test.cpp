/**
 * @file mesh_validator_test.cpp
 * @brief Test for mesh validation utilities
 */

#include <nexussim/core/core.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/io/mesh_validator.hpp>
#include <iostream>
#include <cmath>

using namespace nxs;
using namespace nxs::io;

// Create a valid hex8 mesh (unit cube)
std::shared_ptr<Mesh> create_valid_hex_mesh() {
    auto mesh = std::make_shared<Mesh>(8);

    // Unit cube nodes
    mesh->set_node_coordinates(0, {0.0, 0.0, 0.0});
    mesh->set_node_coordinates(1, {1.0, 0.0, 0.0});
    mesh->set_node_coordinates(2, {1.0, 1.0, 0.0});
    mesh->set_node_coordinates(3, {0.0, 1.0, 0.0});
    mesh->set_node_coordinates(4, {0.0, 0.0, 1.0});
    mesh->set_node_coordinates(5, {1.0, 0.0, 1.0});
    mesh->set_node_coordinates(6, {1.0, 1.0, 1.0});
    mesh->set_node_coordinates(7, {0.0, 1.0, 1.0});

    // One hex element
    mesh->add_element_block("block1", ElementType::Hex8, 1, 8);
    auto& block = mesh->element_block(0);
    auto nodes = block.element_nodes(0);
    nodes[0] = 0; nodes[1] = 1; nodes[2] = 2; nodes[3] = 3;
    nodes[4] = 4; nodes[5] = 5; nodes[6] = 6; nodes[7] = 7;

    return mesh;
}

// Create a mesh with an orphan node
std::shared_ptr<Mesh> create_mesh_with_orphan() {
    auto mesh = std::make_shared<Mesh>(9);  // 9 nodes, but only 8 used

    // Unit cube nodes
    mesh->set_node_coordinates(0, {0.0, 0.0, 0.0});
    mesh->set_node_coordinates(1, {1.0, 0.0, 0.0});
    mesh->set_node_coordinates(2, {1.0, 1.0, 0.0});
    mesh->set_node_coordinates(3, {0.0, 1.0, 0.0});
    mesh->set_node_coordinates(4, {0.0, 0.0, 1.0});
    mesh->set_node_coordinates(5, {1.0, 0.0, 1.0});
    mesh->set_node_coordinates(6, {1.0, 1.0, 1.0});
    mesh->set_node_coordinates(7, {0.0, 1.0, 1.0});
    mesh->set_node_coordinates(8, {2.0, 2.0, 2.0});  // Orphan node

    // One hex element (doesn't use node 8)
    mesh->add_element_block("block1", ElementType::Hex8, 1, 8);
    auto& block = mesh->element_block(0);
    auto nodes = block.element_nodes(0);
    nodes[0] = 0; nodes[1] = 1; nodes[2] = 2; nodes[3] = 3;
    nodes[4] = 4; nodes[5] = 5; nodes[6] = 6; nodes[7] = 7;

    return mesh;
}

// Create a mesh with duplicate nodes
std::shared_ptr<Mesh> create_mesh_with_duplicates() {
    auto mesh = std::make_shared<Mesh>(9);

    // Unit cube nodes with a duplicate
    mesh->set_node_coordinates(0, {0.0, 0.0, 0.0});
    mesh->set_node_coordinates(1, {1.0, 0.0, 0.0});
    mesh->set_node_coordinates(2, {1.0, 1.0, 0.0});
    mesh->set_node_coordinates(3, {0.0, 1.0, 0.0});
    mesh->set_node_coordinates(4, {0.0, 0.0, 1.0});
    mesh->set_node_coordinates(5, {1.0, 0.0, 1.0});
    mesh->set_node_coordinates(6, {1.0, 1.0, 1.0});
    mesh->set_node_coordinates(7, {0.0, 1.0, 1.0});
    mesh->set_node_coordinates(8, {0.0, 0.0, 0.0});  // Duplicate of node 0

    // Two hex elements using the duplicate
    mesh->add_element_block("block1", ElementType::Hex8, 1, 8);
    auto& block = mesh->element_block(0);
    auto nodes = block.element_nodes(0);
    nodes[0] = 0; nodes[1] = 1; nodes[2] = 2; nodes[3] = 3;
    nodes[4] = 4; nodes[5] = 5; nodes[6] = 6; nodes[7] = 7;

    return mesh;
}

// Create a mesh with inverted element (negative Jacobian)
std::shared_ptr<Mesh> create_mesh_with_inverted_element() {
    auto mesh = std::make_shared<Mesh>(8);

    // Inverted cube (node order reversed)
    mesh->set_node_coordinates(0, {0.0, 0.0, 0.0});
    mesh->set_node_coordinates(1, {1.0, 0.0, 0.0});
    mesh->set_node_coordinates(2, {1.0, 1.0, 0.0});
    mesh->set_node_coordinates(3, {0.0, 1.0, 0.0});
    mesh->set_node_coordinates(4, {0.0, 0.0, 1.0});
    mesh->set_node_coordinates(5, {1.0, 0.0, 1.0});
    mesh->set_node_coordinates(6, {1.0, 1.0, 1.0});
    mesh->set_node_coordinates(7, {0.0, 1.0, 1.0});

    // Inverted hex element (wrong node order)
    mesh->add_element_block("block1", ElementType::Hex8, 1, 8);
    auto& block = mesh->element_block(0);
    auto nodes = block.element_nodes(0);
    // Swap top and bottom to create negative Jacobian
    nodes[0] = 4; nodes[1] = 5; nodes[2] = 6; nodes[3] = 7;
    nodes[4] = 0; nodes[5] = 1; nodes[6] = 2; nodes[7] = 3;

    return mesh;
}

// Create a mesh with poor quality element (high aspect ratio)
std::shared_ptr<Mesh> create_mesh_with_poor_quality() {
    auto mesh = std::make_shared<Mesh>(8);

    // Long thin hex (aspect ratio ~100)
    mesh->set_node_coordinates(0, {0.0, 0.0, 0.0});
    mesh->set_node_coordinates(1, {100.0, 0.0, 0.0});
    mesh->set_node_coordinates(2, {100.0, 1.0, 0.0});
    mesh->set_node_coordinates(3, {0.0, 1.0, 0.0});
    mesh->set_node_coordinates(4, {0.0, 0.0, 1.0});
    mesh->set_node_coordinates(5, {100.0, 0.0, 1.0});
    mesh->set_node_coordinates(6, {100.0, 1.0, 1.0});
    mesh->set_node_coordinates(7, {0.0, 1.0, 1.0});

    mesh->add_element_block("block1", ElementType::Hex8, 1, 8);
    auto& block = mesh->element_block(0);
    auto nodes = block.element_nodes(0);
    nodes[0] = 0; nodes[1] = 1; nodes[2] = 2; nodes[3] = 3;
    nodes[4] = 4; nodes[5] = 5; nodes[6] = 6; nodes[7] = 7;

    return mesh;
}

// Create a valid tet4 mesh
std::shared_ptr<Mesh> create_valid_tet_mesh() {
    auto mesh = std::make_shared<Mesh>(4);

    // Regular tetrahedron
    mesh->set_node_coordinates(0, {0.0, 0.0, 0.0});
    mesh->set_node_coordinates(1, {1.0, 0.0, 0.0});
    mesh->set_node_coordinates(2, {0.5, 0.866, 0.0});
    mesh->set_node_coordinates(3, {0.5, 0.289, 0.816});

    mesh->add_element_block("block1", ElementType::Tet4, 1, 4);
    auto& block = mesh->element_block(0);
    auto nodes = block.element_nodes(0);
    nodes[0] = 0; nodes[1] = 1; nodes[2] = 2; nodes[3] = 3;

    return mesh;
}

// Create a valid shell4 mesh
std::shared_ptr<Mesh> create_valid_shell_mesh() {
    auto mesh = std::make_shared<Mesh>(6);

    // Two quad shells
    mesh->set_node_coordinates(0, {0.0, 0.0, 0.0});
    mesh->set_node_coordinates(1, {1.0, 0.0, 0.0});
    mesh->set_node_coordinates(2, {2.0, 0.0, 0.0});
    mesh->set_node_coordinates(3, {0.0, 1.0, 0.0});
    mesh->set_node_coordinates(4, {1.0, 1.0, 0.0});
    mesh->set_node_coordinates(5, {2.0, 1.0, 0.0});

    mesh->add_element_block("block1", ElementType::Shell4, 2, 4);
    auto& block = mesh->element_block(0);

    auto nodes0 = block.element_nodes(0);
    nodes0[0] = 0; nodes0[1] = 1; nodes0[2] = 4; nodes0[3] = 3;

    auto nodes1 = block.element_nodes(1);
    nodes1[0] = 1; nodes1[1] = 2; nodes1[2] = 5; nodes1[3] = 4;

    return mesh;
}

int main() {
    nxs::initialize();

    int passed = 0;
    int failed = 0;

    std::cout << "============================================================\n";
    std::cout << "Mesh Validator Test Suite\n";
    std::cout << "============================================================\n\n";

    MeshValidator validator;

    // Test 1: Valid hex mesh
    {
        std::cout << "Test 1: Valid Hex8 Mesh\n";
        std::cout << "-----------------------\n";

        auto mesh = create_valid_hex_mesh();
        auto summary = validator.validate(*mesh);

        if (summary.is_valid) {
            std::cout << "  [PASS] Mesh is valid\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Mesh should be valid\n";
            failed++;
        }

        if (summary.orphan_nodes == 0) {
            std::cout << "  [PASS] No orphan nodes\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Expected 0 orphan nodes, got " << summary.orphan_nodes << "\n";
            failed++;
        }

        if (summary.negative_jacobian_elements == 0) {
            std::cout << "  [PASS] All Jacobians positive\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Expected 0 negative Jacobians\n";
            failed++;
        }

        summary.print(std::cout);
        std::cout << "\n";
    }

    // Test 2: Mesh with orphan node
    {
        std::cout << "Test 2: Mesh with Orphan Node\n";
        std::cout << "-----------------------------\n";

        auto mesh = create_mesh_with_orphan();
        auto summary = validator.validate(*mesh);

        if (summary.orphan_nodes == 1) {
            std::cout << "  [PASS] Found 1 orphan node\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Expected 1 orphan node, got " << summary.orphan_nodes << "\n";
            failed++;
        }

        // Orphan nodes are warnings, not errors
        if (summary.is_valid) {
            std::cout << "  [PASS] Mesh still valid (orphan is warning)\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Mesh should be valid\n";
            failed++;
        }

        summary.print(std::cout);
        std::cout << "\n";
    }

    // Test 3: Mesh with duplicate nodes
    {
        std::cout << "Test 3: Mesh with Duplicate Nodes\n";
        std::cout << "----------------------------------\n";

        auto mesh = create_mesh_with_duplicates();
        auto summary = validator.validate(*mesh);

        if (summary.duplicate_nodes > 0) {
            std::cout << "  [PASS] Found duplicate nodes\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Should find duplicate nodes\n";
            failed++;
        }

        summary.print(std::cout);
        std::cout << "\n";
    }

    // Test 4: Mesh with inverted element
    {
        std::cout << "Test 4: Mesh with Inverted Element (Negative Jacobian)\n";
        std::cout << "------------------------------------------------------\n";

        auto mesh = create_mesh_with_inverted_element();
        auto summary = validator.validate(*mesh);

        if (summary.negative_jacobian_elements == 1) {
            std::cout << "  [PASS] Found 1 negative Jacobian element\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Expected 1 negative Jacobian, got " << summary.negative_jacobian_elements << "\n";
            failed++;
        }

        if (!summary.is_valid) {
            std::cout << "  [PASS] Mesh correctly marked invalid\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Mesh should be invalid\n";
            failed++;
        }

        summary.print(std::cout);
        std::cout << "\n";
    }

    // Test 5: Mesh with poor quality element
    {
        std::cout << "Test 5: Mesh with Poor Quality Element\n";
        std::cout << "---------------------------------------\n";

        ValidationOptions opts;
        opts.max_aspect_ratio = 50.0;  // Set threshold below actual aspect ratio
        MeshValidator strict_validator(opts);

        auto mesh = create_mesh_with_poor_quality();
        auto summary = strict_validator.validate(*mesh);

        if (summary.poor_quality_elements > 0) {
            std::cout << "  [PASS] Found poor quality elements\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Should find poor quality elements\n";
            failed++;
        }

        summary.print(std::cout);
        std::cout << "\n";
    }

    // Test 6: Valid tet mesh
    {
        std::cout << "Test 6: Valid Tet4 Mesh\n";
        std::cout << "-----------------------\n";

        auto mesh = create_valid_tet_mesh();
        auto summary = validator.validate(*mesh);

        if (summary.is_valid) {
            std::cout << "  [PASS] Tet mesh is valid\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Tet mesh should be valid\n";
            failed++;
        }

        if (summary.negative_jacobian_elements == 0) {
            std::cout << "  [PASS] Tet4 Jacobian positive\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Tet4 should have positive Jacobian\n";
            failed++;
        }

        summary.print(std::cout);
        std::cout << "\n";
    }

    // Test 7: Valid shell mesh
    {
        std::cout << "Test 7: Valid Shell4 Mesh\n";
        std::cout << "-------------------------\n";

        auto mesh = create_valid_shell_mesh();
        auto summary = validator.validate(*mesh);

        if (summary.is_valid) {
            std::cout << "  [PASS] Shell mesh is valid\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Shell mesh should be valid\n";
            failed++;
        }

        if (summary.non_manifold_edges == 0) {
            std::cout << "  [PASS] No non-manifold edges\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Should have no non-manifold edges\n";
            failed++;
        }

        summary.print(std::cout);
        std::cout << "\n";
    }

    // Test 8: Element quality metrics
    {
        std::cout << "Test 8: Element Quality Metrics\n";
        std::cout << "--------------------------------\n";

        auto mesh = create_valid_hex_mesh();
        auto qualities = validator.compute_all_quality(*mesh);

        if (qualities.size() == 1) {
            std::cout << "  [PASS] Got quality for 1 element\n";
            passed++;

            auto& q = qualities[0];
            std::cout << "  Element 0 quality:\n";
            std::cout << "    Jacobian: " << q.jacobian << "\n";
            std::cout << "    Aspect ratio: " << q.aspect_ratio << "\n";
            std::cout << "    Skewness: " << q.skewness << "\n";

            // Unit cube should have reasonable aspect ratio (simplified calculation gives sqrt(2))
            if (q.aspect_ratio < 2.0) {
                std::cout << "  [PASS] Aspect ratio < 2 for unit cube (got " << q.aspect_ratio << ")\n";
                passed++;
            } else {
                std::cout << "  [FAIL] Expected aspect ratio < 2, got " << q.aspect_ratio << "\n";
                failed++;
            }

            // Jacobian should be positive (volume ~1)
            if (q.jacobian > 0) {
                std::cout << "  [PASS] Jacobian positive\n";
                passed++;
            } else {
                std::cout << "  [FAIL] Jacobian should be positive\n";
                failed++;
            }
        } else {
            std::cout << "  [FAIL] Expected 1 element quality\n";
            failed++;
        }

        std::cout << "\n";
    }

    // Summary
    std::cout << "============================================================\n";
    std::cout << "Test Summary: " << passed << " passed, " << failed << " failed\n";
    std::cout << "============================================================\n";

    nxs::finalize();
    return failed > 0 ? 1 : 0;
}
