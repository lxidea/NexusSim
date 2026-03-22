/**
 * @file reader_wave22_test.cpp
 * @brief Wave 22: Multi-format reader and model validation tests
 *
 * Tests 4 components (~7 tests each = 30 tests total):
 * - 22a: RadiossStarterReader (Radioss .d00 format)
 * - 22b: LSDYNAFullReader (LS-DYNA keyword format)
 * - 22c: AbaqusINPReader (Abaqus .inp format)
 * - 22d: ModelValidator (cross-format validation)
 */

#include <nexussim/io/reader_wave22.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

using namespace nxs;
using namespace nxs::io;

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

// Helper: write a unit cube Radioss model to file
// Radioss format: /NODE block with whitespace-separated fields,
// /BRICK/part_id with element_id + node_ids,
// /MAT/LAW/id with title line then density E nu,
// /BCS with grp_id Tx Ty Tz Rx Ry Rz (7 fields)
static void write_radioss_unit_cube(const std::string& path) {
    std::ofstream f(path);
    f << "/BEGIN\n";
    f << "unit_cube_test\n";
    f << "/NODE\n";
    f << "1 0.0 0.0 0.0\n";
    f << "2 1.0 0.0 0.0\n";
    f << "3 1.0 1.0 0.0\n";
    f << "4 0.0 1.0 0.0\n";
    f << "5 0.0 0.0 1.0\n";
    f << "6 1.0 0.0 1.0\n";
    f << "7 1.0 1.0 1.0\n";
    f << "8 0.0 1.0 1.0\n";
    f << "/BRICK/1\n";
    f << "1 1 2 3 4 5 6 7 8\n";
    f << "/MAT/ELASTIC/1\n";
    f << "Steel\n";
    f << "7800.0 210.0e9 0.3\n";
    f << "/BCS\n";
    f << "1 1 1 1 0 0 0\n";
    f << "2 1 1 1 0 0 0\n";
    f << "3 1 1 1 0 0 0\n";
    f << "4 1 1 1 0 0 0\n";
    f.close();
}

// Helper: write LS-DYNA keyword file with comma-separated free format
static void write_lsdyna_model(const std::string& path) {
    std::ofstream f(path);
    f << "*KEYWORD\n";
    f << "*TITLE\n";
    f << "LS-DYNA test model\n";
    f << "$\n";
    f << "$ Node definitions\n";
    f << "$\n";
    f << "*NODE\n";
    f << "       1,     0.000,     0.000,     0.000\n";
    f << "       2,     1.000,     0.000,     0.000\n";
    f << "       3,     1.000,     1.000,     0.000\n";
    f << "       4,     0.000,     1.000,     0.000\n";
    f << "       5,     0.000,     0.000,     1.000\n";
    f << "       6,     1.000,     0.000,     1.000\n";
    f << "       7,     1.000,     1.000,     1.000\n";
    f << "       8,     0.000,     1.000,     1.000\n";
    f << "*ELEMENT_SOLID\n";
    f << "       1,       1,       1,       2,       3,       4,       5,       6\n";
    f << "       7,       8\n";  // continuation line for remaining 2 nodes
    f << "*MAT_ELASTIC\n";
    f << "       1,  7800.0,  2.1e11,     0.3\n";
    f << "*BOUNDARY_SPC\n";
    f << "       1,       0,       1,       1,       1,       0,       0,       0\n";
    f << "       2,       0,       1,       1,       1,       0,       0,       0\n";
    f << "*END\n";
    f.close();
}

// Helper: write Abaqus .inp file with comma-separated data
static void write_abaqus_model(const std::string& path) {
    std::ofstream f(path);
    f << "** Abaqus test model\n";
    f << "**\n";
    f << "*NODE\n";
    f << " 1, 0.0, 0.0, 0.0\n";
    f << " 2, 1.0, 0.0, 0.0\n";
    f << " 3, 1.0, 1.0, 0.0\n";
    f << " 4, 0.0, 1.0, 0.0\n";
    f << " 5, 0.0, 0.0, 1.0\n";
    f << " 6, 1.0, 0.0, 1.0\n";
    f << " 7, 1.0, 1.0, 1.0\n";
    f << " 8, 0.0, 1.0, 1.0\n";
    f << "*ELEMENT, TYPE=C3D8\n";
    f << " 1, 1, 2, 3, 4, 5, 6, 7, 8\n";
    f << "*MATERIAL, NAME=Steel\n";
    f << "*ELASTIC\n";
    f << " 210.0e9, 0.3\n";
    f << "*BOUNDARY\n";
    f << " 1, 1, 3\n";
    f << " 4, 1, 3\n";
    f << "*STEP\n";
    f << "*STATIC\n";
    f << " 0.1, 1.0\n";
    f << "*END STEP\n";
    f.close();
}

// ============================================================================
// 22a: RadiossStarterReader Tests
// ============================================================================
void test_radioss_starter_reader() {
    std::cout << "\n=== 22a: RadiossStarterReader ===\n";

    std::string model_path = "/tmp/radioss_wave22_test.d00";
    write_radioss_unit_cube(model_path);

    RadiossStarterReader reader;

    // Test 1: Parse node block - 8 nodes for unit cube
    {
        auto model = reader.read(model_path);
        CHECK(model.nodes.size() == 8, "Radioss: 8 nodes parsed from unit cube");
    }

    // Test 2: Parse element block - 1 brick element
    {
        auto model = reader.read(model_path);
        CHECK(model.elements.size() == 1, "Radioss: 1 brick element parsed");
        if (!model.elements.empty()) {
            CHECK(model.elements[0].type == 5, "Radioss: element type = 5 (hex8)");
        }
    }

    // Test 3: Parse material block - properties stored in map
    {
        auto model = reader.read(model_path);
        CHECK(model.materials.size() == 1, "Radioss: 1 material parsed");
        if (!model.materials.empty()) {
            auto& mat = model.materials[0];
            CHECK_NEAR(mat.properties["E"], 210.0e9, 1.0e6,
                       "Radioss: material E = 210 GPa");
            CHECK_NEAR(mat.properties["density"], 7800.0, 1.0,
                       "Radioss: material density = 7800");
        }
    }

    // Test 4: Parse BC block - 4 BCs with dof_mask
    {
        auto model = reader.read(model_path);
        CHECK(model.bcs.size() == 4, "Radioss: 4 boundary conditions parsed");
        if (!model.bcs.empty()) {
            // dof_mask for Tx=1,Ty=1,Tz=1,Rx=0,Ry=0,Rz=0 -> bits 0,1,2 set -> mask = 7
            CHECK(model.bcs[0].dof_mask == 7,
                  "Radioss: BC dof_mask = 7 (xyz fixed)");
        }
    }

    // Test 5: Node coordinates correct
    {
        auto model = reader.read(model_path);
        if (model.nodes.size() >= 2) {
            CHECK_NEAR(model.nodes[0].x, 0.0, 1e-10, "Radioss: node 1 x = 0.0");
            CHECK_NEAR(model.nodes[1].x, 1.0, 1e-10, "Radioss: node 2 x = 1.0");
        }
    }

    // Test 6: Handle missing file - throws exception
    {
        bool threw = false;
        try {
            reader.read("/tmp/nonexistent_model_wave22.d00");
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw, "Radioss: exception thrown for missing file");
    }

    // Test 7: Material type and ID parsed
    {
        auto model = reader.read(model_path);
        if (!model.materials.empty()) {
            CHECK(model.materials[0].id == 1, "Radioss: material ID = 1");
            CHECK(model.materials[0].type == "ELASTIC",
                  "Radioss: material type = ELASTIC");
        }
    }
}

// ============================================================================
// 22b: LSDYNAFullReader Tests
// ============================================================================
void test_lsdyna_full_reader() {
    std::cout << "\n=== 22b: LSDYNAFullReader ===\n";

    std::string model_path = "/tmp/lsdyna_wave22_test.k";
    write_lsdyna_model(model_path);

    LSDYNAFullReader reader;

    // Test 1: Parse *NODE card - 8 nodes
    {
        auto model = reader.read(model_path);
        CHECK(model.nodes.size() == 8, "LSDYNA: 8 nodes parsed");
    }

    // Test 2: Parse *ELEMENT_SOLID with continuation line
    {
        auto model = reader.read(model_path);
        CHECK(model.elements.size() == 1, "LSDYNA: 1 solid element parsed");
        if (!model.elements.empty()) {
            CHECK(model.elements[0].type == 5, "LSDYNA: element type = 5 (hex8)");
            CHECK(model.elements[0].num_nodes == 8,
                  "LSDYNA: solid element has 8 nodes (with continuation)");
        }
    }

    // Test 3: Parse *MAT_ELASTIC - density, E, nu in properties map
    {
        auto model = reader.read(model_path);
        CHECK(model.materials.size() == 1, "LSDYNA: 1 material parsed");
        if (!model.materials.empty()) {
            auto& mat = model.materials[0];
            CHECK_NEAR(mat.properties["density"], 7800.0, 1.0,
                       "LSDYNA: material density = 7800");
            CHECK_NEAR(mat.properties["nu"], 0.3, 0.01,
                       "LSDYNA: material nu = 0.3");
        }
    }

    // Test 4: Parse *BOUNDARY_SPC - 2 entries
    {
        auto model = reader.read(model_path);
        CHECK(model.bcs.size() == 2, "LSDYNA: 2 SPC entries parsed");
        if (!model.bcs.empty()) {
            // Fields: NSID=1, CID=0, DOFX=1, DOFY=1, DOFZ=1
            // dof_mask: bit0 (x from field[2]) + bit1 (y from field[3]) + bit2 (z from field[4])
            CHECK((model.bcs[0].dof_mask & 7) == 7,
                  "LSDYNA: SPC fixes xyz DOFs (mask & 7 == 7)");
        }
    }

    // Test 5: Free format (comma-separated) parsing
    {
        auto model = reader.read(model_path);
        if (model.nodes.size() >= 2) {
            CHECK_NEAR(model.nodes[1].x, 1.0, 1e-10,
                       "LSDYNA: free format node 2 x = 1.0");
        }
    }

    // Test 6: $ comment lines skipped
    {
        auto model = reader.read(model_path);
        // Model has $ comment lines that should be skipped without error
        CHECK(model.nodes.size() == 8, "LSDYNA: comments skipped, 8 nodes still parsed");
    }

    // Test 7: Missing file throws exception
    {
        bool threw = false;
        try {
            reader.read("/tmp/lsdyna_nonexistent_wave22.k");
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw, "LSDYNA: exception thrown for missing file");
    }
}

// ============================================================================
// 22c: AbaqusINPReader Tests
// ============================================================================
void test_abaqus_inp_reader() {
    std::cout << "\n=== 22c: AbaqusINPReader ===\n";

    std::string model_path = "/tmp/abaqus_wave22_test.inp";
    write_abaqus_model(model_path);

    AbaqusINPReader reader;

    // Test 1: Parse *NODE - 8 nodes with correct coordinates
    {
        auto model = reader.read(model_path);
        CHECK(model.nodes.size() == 8, "Abaqus: 8 nodes parsed");
        if (model.nodes.size() >= 3) {
            CHECK_NEAR(model.nodes[2].x, 1.0, 1e-10, "Abaqus: node 3 x = 1.0");
            CHECK_NEAR(model.nodes[2].y, 1.0, 1e-10, "Abaqus: node 3 y = 1.0");
        }
    }

    // Test 2: Parse *ELEMENT TYPE=C3D8
    {
        auto model = reader.read(model_path);
        CHECK(model.elements.size() == 1, "Abaqus: 1 element parsed");
        if (!model.elements.empty()) {
            CHECK(model.elements[0].type == 5, "Abaqus: element type = 5 (C3D8 -> hex8)");
            CHECK(model.elements[0].num_nodes == 8,
                  "Abaqus: C3D8 element has 8 nodes");
        }
    }

    // Test 3: Parse *MATERIAL + *ELASTIC
    {
        auto model = reader.read(model_path);
        CHECK(model.materials.size() == 1, "Abaqus: 1 material parsed");
        if (!model.materials.empty()) {
            auto& mat = model.materials[0];
            CHECK_NEAR(mat.properties["E"], 210.0e9, 1.0e6,
                       "Abaqus: material E = 210 GPa");
            CHECK_NEAR(mat.properties["nu"], 0.3, 0.01,
                       "Abaqus: material nu = 0.3");
            CHECK(mat.type == "ELASTIC", "Abaqus: material type = ELASTIC");
        }
    }

    // Test 4: Parse *BOUNDARY
    {
        auto model = reader.read(model_path);
        CHECK(model.bcs.size() == 2, "Abaqus: 2 boundary conditions parsed");
        if (!model.bcs.empty()) {
            // " 1, 1, 3" -> node 1, DOFs 1-3 fixed -> mask = 0b111 = 7
            CHECK(model.bcs[0].dof_mask == 7,
                  "Abaqus: BC DOFs 1-3 fixed (mask=7) for node 1");
        }
    }

    // Test 5: Comment handling (** lines skipped)
    {
        auto model = reader.read(model_path);
        // Model has ** comment lines that should be skipped
        CHECK(model.nodes.size() == 8,
              "Abaqus: ** comments handled, 8 nodes parsed");
    }

    // Test 6: Data line comma-separated format
    {
        auto model = reader.read(model_path);
        CHECK(model.nodes.size() > 0 && model.elements.size() > 0 &&
              model.materials.size() > 0,
              "Abaqus: complete model read successfully");
    }

    // Test 7: Missing file throws exception
    {
        bool threw = false;
        try {
            reader.read("/tmp/abaqus_nonexistent_wave22.inp");
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw, "Abaqus: exception thrown for missing file");
    }
}

// ============================================================================
// 22d: ModelValidator Tests
// ============================================================================

// Helper: create a valid unit cube ModelData for testing
static ModelData make_valid_cube_model() {
    ModelData model;
    // 8 nodes of unit cube
    model.nodes = {
        {1, 0,0,0}, {2, 1,0,0}, {3, 1,1,0}, {4, 0,1,0},
        {5, 0,0,1}, {6, 1,0,1}, {7, 1,1,1}, {8, 0,1,1}
    };
    // 1 hex8 element
    ElementData elem;
    elem.id = 1; elem.part_id = 1; elem.type = 5; elem.num_nodes = 8;
    std::fill(std::begin(elem.nodes), std::end(elem.nodes), 0);
    elem.nodes[0]=1; elem.nodes[1]=2; elem.nodes[2]=3; elem.nodes[3]=4;
    elem.nodes[4]=5; elem.nodes[5]=6; elem.nodes[6]=7; elem.nodes[7]=8;
    model.elements.push_back(elem);
    // Material with id=1 matching part_id
    MaterialData mat;
    mat.id = 1; mat.type = "ELASTIC";
    mat.properties["E"] = 210e9; mat.properties["nu"] = 0.3;
    mat.properties["density"] = 7800.0;
    model.materials.push_back(mat);
    return model;
}

void test_model_validator() {
    std::cout << "\n=== 22d: ModelValidator ===\n";

    ModelValidator validator;

    // Test 1: Detect orphan nodes
    {
        auto model = make_valid_cube_model();
        // Add an orphan node (not referenced by any element)
        model.nodes.push_back({99, 5.0, 5.0, 5.0});

        auto msgs = validator.validate(model);
        bool found_orphan = false;
        for (auto& m : msgs) {
            if (m.code == 1001) { // orphan node code
                found_orphan = true;
                CHECK(m.level == 1, "Validator: orphan node is warning (level 1)");
            }
        }
        CHECK(found_orphan, "Validator: orphan node detected (code 1001)");
    }

    // Test 2: Detect negative Jacobian (inverted hex element)
    {
        ModelData model;
        model.nodes = {
            {1, 0,0,0}, {2, 1,0,0}, {3, 1,1,0}, {4, 0,1,0},
            {5, 0,0,1}, {6, 1,0,1}, {7, 1,1,1}, {8, 0,1,1}
        };
        // Inverted hex: swap bottom and top to get negative Jacobian
        ElementData elem;
        elem.id = 1; elem.part_id = 1; elem.type = 5; elem.num_nodes = 8;
        std::fill(std::begin(elem.nodes), std::end(elem.nodes), 0);
        elem.nodes[0]=5; elem.nodes[1]=6; elem.nodes[2]=7; elem.nodes[3]=8;
        elem.nodes[4]=1; elem.nodes[5]=2; elem.nodes[6]=3; elem.nodes[7]=4;
        model.elements.push_back(elem);
        MaterialData mat; mat.id = 1; mat.type = "ELASTIC";
        model.materials.push_back(mat);

        auto msgs = validator.validate(model);
        bool found_neg_jac = false;
        for (auto& m : msgs) {
            if (m.code == 2001) { // negative Jacobian code
                found_neg_jac = true;
                CHECK(m.level == 2, "Validator: negative Jacobian is error (level 2)");
            }
        }
        CHECK(found_neg_jac, "Validator: negative Jacobian detected (code 2001)");
    }

    // Test 3: Detect missing material assignment
    {
        ModelData model;
        model.nodes = {
            {1, 0,0,0}, {2, 1,0,0}, {3, 1,1,0}, {4, 0,1,0},
            {5, 0,0,1}, {6, 1,0,1}, {7, 1,1,1}, {8, 0,1,1}
        };
        ElementData elem;
        elem.id = 1; elem.part_id = 77; elem.type = 5; elem.num_nodes = 8;
        std::fill(std::begin(elem.nodes), std::end(elem.nodes), 0);
        elem.nodes[0]=1; elem.nodes[1]=2; elem.nodes[2]=3; elem.nodes[3]=4;
        elem.nodes[4]=5; elem.nodes[5]=6; elem.nodes[6]=7; elem.nodes[7]=8;
        model.elements.push_back(elem);
        // Material id=1 but element part_id=77 -> no match
        MaterialData mat; mat.id = 1; mat.type = "ELASTIC";
        model.materials.push_back(mat);

        auto msgs = validator.validate(model);
        bool found_missing = false;
        for (auto& m : msgs) {
            if (m.code == 1002) { // missing material code
                found_missing = true;
            }
        }
        CHECK(found_missing, "Validator: missing material assignment detected (code 1002)");
    }

    // Test 4: Detect over-constrained node (> 6 DOF constraints)
    {
        auto model = make_valid_cube_model();
        // Apply 4 BCs to node 1 each with dof_mask=7 (3 DOFs) -> 12 total > 6
        for (int i = 0; i < 4; i++) {
            BCData bc;
            bc.node_id = 1;
            bc.dof_mask = 7; // bits 0,1,2 = xyz
            bc.value = 0.0;
            model.bcs.push_back(bc);
        }

        auto msgs = validator.validate(model);
        bool found_over = false;
        for (auto& m : msgs) {
            if (m.code == 1003) { // over-constrained code
                found_over = true;
            }
        }
        CHECK(found_over, "Validator: over-constrained node detected (code 1003)");
    }

    // Test 5: Clean model passes validation (no errors)
    {
        auto model = make_valid_cube_model();
        auto msgs = validator.validate(model);
        // First message is summary; check summary level
        CHECK(!msgs.empty(), "Validator: at least summary message returned");
        if (!msgs.empty()) {
            // Summary level should be 0 (info) for clean model
            CHECK(msgs[0].level == 0,
                  "Validator: clean unit cube passes (summary level 0)");
        }
    }

    // Test 6: Multiple issues detected simultaneously
    {
        auto model = make_valid_cube_model();
        // Add orphan node
        model.nodes.push_back({99, 10.0, 10.0, 10.0});
        // Change element part_id so material doesn't match
        model.elements[0].part_id = 77;

        auto msgs = validator.validate(model);
        int issue_count = 0;
        for (auto& m : msgs) {
            if (m.code > 0) issue_count++; // non-summary messages
        }
        CHECK(issue_count >= 2,
              "Validator: multiple issues detected (orphan + missing mat)");
    }

    // Test 7: Severity levels - orphan=warning(1), neg_jac=error(2)
    {
        // Test that orphan nodes produce level=1 (warning)
        auto model = make_valid_cube_model();
        model.nodes.push_back({99, 5.0, 5.0, 5.0});

        auto msgs = validator.validate(model);
        bool has_warning = false;
        for (auto& m : msgs) {
            if (m.level == 1 && m.code > 0) has_warning = true;
        }
        CHECK(has_warning, "Validator: orphan node reported as warning (level 1)");

        // Summary should be level=1 (worst issue is warning)
        if (!msgs.empty()) {
            CHECK(msgs[0].level == 1,
                  "Validator: summary level = 1 when worst issue is warning");
        }
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "===================================================\n";
    std::cout << "Wave 22: Multi-Format Reader & Validator Test Suite\n";
    std::cout << "===================================================\n";

    test_radioss_starter_reader();
    test_lsdyna_full_reader();
    test_abaqus_inp_reader();
    test_model_validator();

    std::cout << "\n===================================\n";
    std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed\n";

    if (tests_failed > 0) {
        std::cout << "SOME TESTS FAILED!\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED!\n";
    return 0;
}
