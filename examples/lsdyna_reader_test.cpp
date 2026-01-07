/**
 * @file lsdyna_reader_test.cpp
 * @brief Test for LS-DYNA keyword format reader
 */

#include <nexussim/core/core.hpp>
#include <nexussim/io/lsdyna_reader.hpp>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace nxs;
using namespace nxs::io;

// Create a simple test LS-DYNA keyword file
void create_test_file(const std::string& filename) {
    std::ofstream file(filename);
    file << R"(*KEYWORD
*TITLE
Simple Cube Test Model
$
$ Nodes for a single hex element
*NODE
       1       0.0       0.0       0.0
       2       1.0       0.0       0.0
       3       1.0       1.0       0.0
       4       0.0       1.0       0.0
       5       0.0       0.0       1.0
       6       1.0       0.0       1.0
       7       1.0       1.0       1.0
       8       0.0       1.0       1.0
$
$ Material definition - elastic steel
*MAT_ELASTIC
       1    7800.0   2.1e+11      0.30
$
$ Part definition
*PART
Steel Cube
       1       1       1
$
$ Element definition - single hex8
*ELEMENT_SOLID
       1       1       1       2       3       4       5       6       7       8
$
$ Node set for boundary conditions
*SET_NODE_LIST
       1
       1       2       3       4
$
$ Fix bottom face in Z direction
*BOUNDARY_SPC_SET
       1       0       0       0       1       0       0       0
$
*END
)";
    file.close();
}

// Create a more complex test file with shell and beam elements
void create_mixed_test_file(const std::string& filename) {
    std::ofstream file(filename);
    file << R"(*KEYWORD
*TITLE
Mixed Element Test Model
$
$ Nodes
*NODE
       1       0.0       0.0       0.0
       2       1.0       0.0       0.0
       3       2.0       0.0       0.0
       4       0.0       1.0       0.0
       5       1.0       1.0       0.0
       6       2.0       1.0       0.0
       7       0.0       0.0       1.0
       8       1.0       0.0       1.0
$
$ Elastic material
*MAT_ELASTIC
       1    7800.0   2.1e+11      0.30
$
$ Johnson-Cook material
*MAT_JOHNSON_COOK
       2    7800.0   8.0e+10   2.1e+11      0.30
  7.92e+8  5.10e+8      0.26     0.014      1.03    1793.0   293.0      1.0
$
$ Parts
*PART
Shell Part
      10       1       1
*PART
Beam Part
      20       2       1
$
$ Shell elements
*ELEMENT_SHELL
      10      10       1       2       5       4
      11      10       2       3       6       5
$
$ Beam elements
*ELEMENT_BEAM
      20      20       7       8       1
$
$ Node sets
*SET_NODE_LIST
       1
       1       4       7
*SET_NODE_LIST
       2
       3       6
$
$ Boundary conditions
*BOUNDARY_SPC_SET
       1       0       1       1       1       0       0       0
$
$ Load on right edge
*LOAD_NODE_SET
       2       1       1     1000.0
$
$ Load curve
*DEFINE_CURVE
       1
       0.0       0.0
       0.001     1.0
       1.0       1.0
$
*END
)";
    file.close();
}

int main() {
    nxs::initialize();

    int passed = 0;
    int failed = 0;

    std::cout << "============================================================\n";
    std::cout << "LS-DYNA Reader Test Suite\n";
    std::cout << "============================================================\n\n";

    // Test 1: Basic hex element file
    {
        std::cout << "Test 1: Basic Hex8 Element File\n";
        std::cout << "--------------------------------\n";

        std::string test_file = "/tmp/test_hex.k";
        create_test_file(test_file);

        LSDynaReader reader;
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
            std::cout << "  [PASS] Element type = HEX8\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Wrong element type\n";
            failed++;
        }

        // Check material
        if (reader.materials().size() == 1) {
            std::cout << "  [PASS] Material count = 1\n";
            passed++;

            auto& mat = reader.materials().begin()->second;
            if (std::abs(mat.e - 2.1e11) < 1e6) {
                std::cout << "  [PASS] Young's modulus = " << mat.e << "\n";
                passed++;
            } else {
                std::cout << "  [FAIL] Wrong Young's modulus: " << mat.e << "\n";
                failed++;
            }
        } else {
            std::cout << "  [FAIL] Material count = " << reader.materials().size() << "\n";
            failed++;
        }

        // Check node sets
        if (reader.node_sets().size() == 1) {
            std::cout << "  [PASS] Node set count = 1\n";
            passed++;

            auto nodes = reader.get_node_set(1);
            if (nodes.size() == 4) {
                std::cout << "  [PASS] Node set 1 has 4 nodes\n";
                passed++;
            } else {
                std::cout << "  [FAIL] Node set 1 has " << nodes.size() << " nodes\n";
                failed++;
            }
        } else {
            std::cout << "  [FAIL] Node set count = " << reader.node_sets().size() << "\n";
            failed++;
        }

        // Check SPCs
        if (reader.spcs().size() == 1) {
            std::cout << "  [PASS] SPC count = 1\n";
            passed++;

            const auto& spc = reader.spcs()[0];
            if (spc.dofz == 1) {
                std::cout << "  [PASS] SPC constrains Z direction\n";
                passed++;
            } else {
                std::cout << "  [FAIL] SPC wrong DOF constraint\n";
                failed++;
            }
        } else {
            std::cout << "  [FAIL] SPC count = " << reader.spcs().size() << "\n";
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

            // Check coordinates
            auto coord = mesh->get_node_coordinates(1);  // Node 2 at (1,0,0)
            if (std::abs(coord[0] - 1.0) < 1e-10) {
                std::cout << "  [PASS] Node coordinates correct\n";
                passed++;
            } else {
                std::cout << "  [FAIL] Node coordinates wrong\n";
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

        std::string test_file = "/tmp/test_mixed.k";
        create_mixed_test_file(test_file);

        LSDynaReader reader;
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

        // Check materials (elastic + J-C)
        if (reader.materials().size() == 2) {
            std::cout << "  [PASS] Material count = 2\n";
            passed++;

            // Check Johnson-Cook parameters
            auto it = reader.materials().find(2);
            if (it != reader.materials().end()) {
                const auto& jc = it->second;
                if (jc.type == "JOHNSON_COOK" && std::abs(jc.a_jc - 7.92e8) < 1e5) {
                    std::cout << "  [PASS] Johnson-Cook A = " << jc.a_jc << "\n";
                    passed++;
                } else {
                    std::cout << "  [FAIL] Johnson-Cook parameters wrong\n";
                    failed++;
                }
            }
        } else {
            std::cout << "  [FAIL] Material count = " << reader.materials().size() << "\n";
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

        // Check node sets
        if (reader.node_sets().size() == 2) {
            std::cout << "  [PASS] Node set count = 2\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Node set count = " << reader.node_sets().size() << "\n";
            failed++;
        }

        // Check loads
        if (reader.loads().size() == 1) {
            std::cout << "  [PASS] Load count = 1\n";
            passed++;

            const auto& load = reader.loads()[0];
            if (load.dof == 1 && std::abs(load.sf - 1000.0) < 1e-6) {
                std::cout << "  [PASS] Load parameters correct\n";
                passed++;
            } else {
                std::cout << "  [FAIL] Load parameters wrong\n";
                failed++;
            }
        } else {
            std::cout << "  [FAIL] Load count = " << reader.loads().size() << "\n";
            failed++;
        }

        // Check load curves
        if (reader.load_curves().size() == 1) {
            std::cout << "  [PASS] Load curve count = 1\n";
            passed++;

            auto it = reader.load_curves().find(1);
            if (it != reader.load_curves().end() && it->second.points.size() == 3) {
                std::cout << "  [PASS] Load curve has 3 points\n";
                passed++;
            } else {
                std::cout << "  [FAIL] Load curve wrong\n";
                failed++;
            }
        } else {
            std::cout << "  [FAIL] Load curve count = " << reader.load_curves().size() << "\n";
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
        } else {
            std::cout << "  [FAIL] Mesh creation failed\n";
            failed++;
        }

        reader.print_summary(std::cout);
        std::cout << "\n";
    }

    // Test 3: Tetrahedral elements
    {
        std::cout << "Test 3: Tetrahedral Elements\n";
        std::cout << "----------------------------\n";

        // Create tet4 test file
        std::ofstream file("/tmp/test_tet.k");
        file << R"(*KEYWORD
*NODE
       1       0.0       0.0       0.0
       2       1.0       0.0       0.0
       3       0.5       0.866     0.0
       4       0.5       0.289     0.816
*MAT_ELASTIC
       1    7800.0   2.1e+11      0.30
*PART
Tet Part
       1       1       1
*ELEMENT_SOLID
       1       1       1       2       3       4
*END
)";
        file.close();

        LSDynaReader reader;
        bool success = reader.read("/tmp/test_tet.k");

        if (success && reader.elements().size() == 1) {
            const auto& elem = reader.elements()[0];
            if (elem.type == ElementType::Tet4 && elem.nodes.size() == 4) {
                std::cout << "  [PASS] TET4 element parsed correctly\n";
                passed++;
            } else {
                std::cout << "  [FAIL] TET4 element wrong type or nodes\n";
                failed++;
            }
        } else {
            std::cout << "  [FAIL] Failed to parse TET4 file\n";
            failed++;
        }
        std::cout << "\n";
    }

    // Test 4: Comma-separated format
    {
        std::cout << "Test 4: Comma-Separated Format\n";
        std::cout << "------------------------------\n";

        std::ofstream file("/tmp/test_comma.k");
        file << R"(*KEYWORD
*NODE
1,0.0,0.0,0.0
2,1.0,0.0,0.0
3,0.5,0.866,0.0
4,0.5,0.289,0.816
*MAT_ELASTIC
1,7800.0,2.1e+11,0.30
*PART
Test
1,1,1
*ELEMENT_SOLID
1,1,1,2,3,4
*END
)";
        file.close();

        LSDynaReader reader;
        bool success = reader.read("/tmp/test_comma.k");

        if (success && reader.nodes().size() == 4 && reader.elements().size() == 1) {
            std::cout << "  [PASS] Comma-separated format parsed\n";
            passed++;
        } else {
            std::cout << "  [FAIL] Comma-separated format failed\n";
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
