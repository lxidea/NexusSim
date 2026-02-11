/**
 * @file lsdyna_reader_test.cpp
 * @brief Test LS-DYNA keyword file reader
 */

#include <nexussim/core/core.hpp>
#include <nexussim/io/lsdyna_reader.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace nxs;
using namespace nxs::io;

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

const char* SAMPLE_LSDYNA_FILE = R"(
*KEYWORD
*TITLE
Simple Cube Test Model
*NODE
       1       0.0       0.0       0.0
       2       1.0       0.0       0.0
       3       1.0       1.0       0.0
       4       0.0       1.0       0.0
       5       0.0       0.0       1.0
       6       1.0       0.0       1.0
       7       1.0       1.0       1.0
       8       0.0       1.0       1.0
*ELEMENT_SOLID
       1         1         1         2         3         4         5         6         7         8
*MAT_ELASTIC
         1    7850.0    2.0E11       0.3
*PART
Cube Part
         1         1         1
*SET_NODE_LIST
         1
         1         2         3         4
*BOUNDARY_SPC_SET
         1         0         1         1         1         0         0         0
*DEFINE_CURVE
         1
               0.0               0.0
             0.001          1000.0
*END
)";

bool test_simple_model() {
    std::cout << "\n=== Test 1: Parse Simple Model ===\n";

    std::string filename = "/tmp/test_simple.k";
    {
        std::ofstream file(filename);
        file << SAMPLE_LSDYNA_FILE;
    }

    LSDynaReader reader;
    bool success = reader.read(filename);

    CHECK(success, "File read successfully");
    CHECK(reader.nodes().size() == 8, "8 nodes parsed");
    CHECK(reader.elements().size() == 1, "1 element parsed");
    CHECK(reader.materials().size() == 1, "1 material parsed");
    CHECK(reader.parts().size() == 1, "1 part parsed");
    CHECK(reader.node_sets().size() == 1, "1 node set parsed");
    CHECK(reader.spcs().size() == 1, "1 SPC parsed");
    CHECK(reader.load_curves().size() == 1, "1 load curve parsed");
    CHECK(reader.title() == "Simple Cube Test Model", "Title parsed correctly");

    const auto& elem = reader.elements()[0];
    CHECK(elem.type == ElementType::Hex8, "Element type is Hex8");
    CHECK(elem.nodes.size() == 8, "Element has 8 nodes");

    const auto& mat = reader.materials().at(1);
    CHECK(mat.type == "ELASTIC", "Material type is ELASTIC");
    CHECK(std::abs(mat.ro - 7850.0) < 1e-6, "Material density correct");
    CHECK(std::abs(mat.e - 2.0e11) < 1e6, "Material E correct");

    reader.print_summary();
    return true;
}

bool test_mesh_creation() {
    std::cout << "\n=== Test 2: Mesh Creation ===\n";

    std::string filename = "/tmp/test_mesh.k";
    {
        std::ofstream file(filename);
        file << SAMPLE_LSDYNA_FILE;
    }

    LSDynaReader reader;
    reader.read(filename);

    auto mesh = reader.create_mesh();

    CHECK(mesh != nullptr, "Mesh created successfully");
    CHECK(mesh->num_nodes() == 8, "Mesh has 8 nodes");
    CHECK(mesh->num_elements() == 1, "Mesh has 1 element");

    auto coords = mesh->get_node_coordinates(0);
    CHECK(std::abs(coords[0] - 0.0) < 1e-10, "Node 0 x-coord correct");

    coords = mesh->get_node_coordinates(7);
    CHECK(std::abs(coords[2] - 1.0) < 1e-10, "Node 7 z-coord correct");

    return true;
}

bool test_material_extraction() {
    std::cout << "\n=== Test 3: Material Extraction ===\n";

    std::string filename = "/tmp/test_mat.k";
    {
        std::ofstream file(filename);
        file << SAMPLE_LSDYNA_FILE;
    }

    LSDynaReader reader;
    reader.read(filename);

    auto props = reader.get_material(1);
    CHECK(std::abs(props.density - 7850.0) < 1e-6, "Part 1 density correct");
    CHECK(std::abs(props.E - 2.0e11) < 1e6, "Part 1 E correct");
    CHECK(std::abs(props.nu - 0.3) < 1e-6, "Part 1 nu correct");

    return true;
}

bool test_node_set() {
    std::cout << "\n=== Test 4: Node Set ===\n";

    std::string filename = "/tmp/test_ns.k";
    {
        std::ofstream file(filename);
        file << SAMPLE_LSDYNA_FILE;
    }

    LSDynaReader reader;
    reader.read(filename);

    auto ns = reader.get_node_set(1);
    CHECK(ns.size() == 4, "Node set 1 has 4 nodes");

    return true;
}

bool test_error_handling() {
    std::cout << "\n=== Test 5: Error Handling ===\n";

    LSDynaReader reader;
    bool result = reader.read("/tmp/nonexistent_file_xyz.k");
    CHECK(!result, "Non-existent file returns false");

    std::string filename = "/tmp/test_empty.k";
    {
        std::ofstream file(filename);
        file << "*KEYWORD\n*END\n";
    }

    result = reader.read(filename);
    CHECK(result, "Empty model reads without error");
    CHECK(reader.nodes().empty(), "Empty model has no nodes");

    return true;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "NexusSim LS-DYNA Reader Test Suite\n";
    std::cout << "========================================\n";

    test_simple_model();
    test_mesh_creation();
    test_material_extraction();
    test_node_set();
    test_error_handling();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    std::cout << "========================================\n";

    return tests_failed > 0 ? 1 : 0;
}
