/**
 * @file radioss_reader_test.cpp
 * @brief Test for OpenRadioss input deck reader
 */

#include <nexussim/core/core.hpp>
#include <nexussim/io/radioss_reader.hpp>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace nxs;
using namespace nxs::io;

// Create a simple test Radioss file
void create_test_file(const std::string& filename) {
    std::ofstream file(filename);
    file << R"(#RADIOSS STARTER
/BEGIN
Simple Cube Test Model
                    2021
mm kg ms
/NODE
       1       0.0       0.0       0.0
       2       1.0       0.0       0.0
       3       1.0       1.0       0.0
       4       0.0       1.0       0.0
       5       0.0       0.0       1.0
       6       1.0       0.0       1.0
       7       1.0       1.0       1.0
       8       0.0       1.0       1.0
#
# Material - elastic steel
/MAT/ELAST/1
Steel
    7800.0   2.1e+11      0.30
#
# Part definition
/PART/1
Cube Part
       1       1
#
# Hex8 element
/BRICK/1
       1       1       2       3       4       5       6       7       8
#
# Node group for boundary conditions
/GRNOD/bottom
       1       2       3       4
#
# Boundary condition - fix bottom in Z
/BCS/1
       4       0       1
#
/END
)";
    file.close();
}

// Create mixed element test file
void create_mixed_test_file(const std::string& filename) {
    std::ofstream file(filename);
    file << R"(#RADIOSS STARTER
/BEGIN
Mixed Element Test Model
                    2021
mm kg ms
/NODE
       1       0.0       0.0       0.0
       2       1.0       0.0       0.0
       3       2.0       0.0       0.0
       4       0.0       1.0       0.0
       5       1.0       1.0       0.0
       6       2.0       1.0       0.0
       7       0.0       0.0       1.0
       8       1.0       0.0       1.0
#
# Material
/MAT/ELAST/1
Steel
    7800.0   2.1e+11      0.30
#
# Parts
/PART/10
Shell Part
       1       1
/PART/20
Beam Part
       2       1
#
# Shell elements
/SHELL/10
      10       1       2       5       4
      11       2       3       6       5
#
# Beam element (id n1 n2 n3_orient)
/BEAM/20
      20       7       8       1
#
# Node groups
/GRNOD/left
       1       4       7
/GRNOD/right
       3       6
#
/END
)";
    file.close();
}

int main() {
    nxs::initialize();

    int passed = 0;
    int failed = 0;

    std::cout << "============================================================\n";
    std::cout << "Radioss Reader Test Suite\n";
    std::cout << "============================================================\n\n";

    // Test 1: Basic hex element file
    {
        std::cout << "Test 1: Basic Hex8 Element File\n";
        std::cout << "--------------------------------\n";

        std::string test_file = "/tmp/test_radioss_hex.rad";
        create_test_file(test_file);

        RadiossReader reader;
        bool success = reader.read(test_file);

        if (success) {
            std::cout << "  [PASS] File read successfully\n";
            passed++;
        } else {
            std::cout << "  [FAIL] File read failed\n";
            failed++;
        }

        // Check nodes
        if (reader.nodes().size() == 8) {
            std::cout << "  [PASS] Node count = 8\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Node count = " << reader.nodes().size() << " (expected 8)\n";
            failed++;
        }

        // Check elements
        if (reader.elements().size() == 1) {
            std::cout << "  [PASS] Element count = 1\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Element count = " << reader.elements().size() << " (expected 1)\n";
            failed++;
        }

        // Check element type
        if (!reader.elements().empty() && reader.elements()[0].type == ElementType::Hex8) {
            std::cout << "  [PASS] Element type = Hex8\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Wrong element type\n";
            failed++;
        }

        // Check parts
        if (reader.parts().size() == 1) {
            std::cout << "  [PASS] Part count = 1\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Part count = " << reader.parts().size() << "\n";
            failed++;
        }

        // Check boundary conditions
        if (reader.boundary_conditions().size() == 1) {
            std::cout << "  [PASS] Boundary condition count = 1\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Boundary condition count = " << reader.boundary_conditions().size() << "\n";
            failed++;
        }

        // Create mesh and verify
        auto mesh = reader.create_mesh();
        if (mesh) {
            std::cout << "  [PASS] Mesh created successfully\n";
            passed++;

            if (mesh->num_nodes() == 8) {
                std::cout << "  [PASS] Mesh node count = 8\n";
                passed++;
            } else {
                std::cout << "  [FAIL] Mesh node count = " << mesh->num_nodes() << "\n";
                failed++;
            }

            if (mesh->num_elements() == 1) {
                std::cout << "  [PASS] Mesh element count = 1\n";
                passed++;
            } else {
                std::cout << "  [FAIL] Mesh element count = " << mesh->num_elements() << "\n";
                failed++;
            }
        } else {
            std::cout << "  [FAIL] Mesh creation failed\n";
            failed++;
        }

        reader.print_summary(std::cout);
        std::cout << "\n";
    }

    // Test 2: Mixed element file
    {
        std::cout << "Test 2: Mixed Element File (Shell + Beam)\n";
        std::cout << "-----------------------------------------\n";

        std::string test_file = "/tmp/test_radioss_mixed.rad";
        create_mixed_test_file(test_file);

        RadiossReader reader;
        bool success = reader.read(test_file);

        if (success) {
            std::cout << "  [PASS] File read successfully\n";
            passed++;
        } else {
            std::cout << "  [FAIL] File read failed\n";
            failed++;
        }

        // Check nodes
        if (reader.nodes().size() == 8) {
            std::cout << "  [PASS] Node count = 8\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Node count = " << reader.nodes().size() << "\n";
            failed++;
        }

        // Check elements (2 shells + 1 beam = 3)
        if (reader.elements().size() == 3) {
            std::cout << "  [PASS] Element count = 3\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Element count = " << reader.elements().size() << "\n";
            failed++;
        }

        // Check shell elements
        int shell_count = 0;
        int beam_count = 0;
        for (const auto& elem : reader.elements()) {
            if (elem.type == ElementType::Shell4) shell_count++;
            if (elem.type == ElementType::Beam2) beam_count++;
        }

        if (shell_count == 2) {
            std::cout << "  [PASS] Shell element count = 2\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Shell element count = " << shell_count << "\n";
            failed++;
        }

        if (beam_count == 1) {
            std::cout << "  [PASS] Beam element count = 1\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Beam element count = " << beam_count << "\n";
            failed++;
        }

        // Check parts
        if (reader.parts().size() == 2) {
            std::cout << "  [PASS] Part count = 2\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Part count = " << reader.parts().size() << "\n";
            failed++;
        }

        // Create mesh
        auto mesh = reader.create_mesh();
        if (mesh) {
            std::cout << "  [PASS] Mesh created successfully\n";
            passed++;

            // Should have 2 element blocks (shell and beam)
            if (mesh->num_element_blocks() == 2) {
                std::cout << "  [PASS] Element blocks = 2\n";
                passed++;
            } else {
                std::cout << "  [FAIL] Element blocks = " << mesh->num_element_blocks() << "\n";
                failed++;
            }

            // Check node sets
            auto left_set = mesh->get_node_set("left");
            if (left_set && left_set->size() == 3) {
                std::cout << "  [PASS] Node set 'left' has 3 nodes\n";
                passed++;
            } else {
                std::cout << "  [FAIL] Node set 'left' not found or wrong size\n";
                failed++;
            }
        } else {
            std::cout << "  [FAIL] Mesh creation failed\n";
            failed++;
        }

        reader.print_summary(std::cout);
        std::cout << "\n";
    }

    // Summary
    std::cout << "============================================================\n";
    std::cout << "Test Summary: " << passed << " passed, " << failed << " failed\n";
    std::cout << "============================================================\n";

    nxs::finalize();
    return failed > 0 ? 1 : 0;
}
