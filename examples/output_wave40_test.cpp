/**
 * @file output_wave40_test.cpp
 * @brief Tests for Wave 40: Extended Output Format Writers
 *
 * Tests all 6 output classes:
 *   1. RadiossAnimWriter
 *   2. StatusFileWriter
 *   3. DynainWriter
 *   4. ReactionForcesTH
 *   5. QAPrintWriter
 *   6. ReportGenerator
 */

#include <nexussim/io/output_wave40.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>

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

// ============================================================================
// Helper: create sample node/element data
// ============================================================================

static std::vector<AnimNodeData> make_nodes(int count) {
    std::vector<AnimNodeData> nodes(count);
    for (int i = 0; i < count; ++i) {
        nodes[i].id = i + 1;
        nodes[i].x = static_cast<Real>(i) * 1.0;
        nodes[i].y = static_cast<Real>(i) * 2.0;
        nodes[i].z = static_cast<Real>(i) * 3.0;
        nodes[i].vx = 0.1 * i;
        nodes[i].vy = 0.2 * i;
        nodes[i].vz = 0.3 * i;
        nodes[i].ax = 0.01 * i;
        nodes[i].ay = 0.02 * i;
        nodes[i].az = 0.03 * i;
    }
    return nodes;
}

static std::vector<AnimElementData> make_elements(int count, bool shell = false) {
    std::vector<AnimElementData> elems(count);
    for (int i = 0; i < count; ++i) {
        elems[i].id = i + 1;
        elems[i].part_id = 1;
        elems[i].type = shell ? 1 : 2;
        elems[i].stress = {100.0 + i, 200.0 + i, 300.0 + i, 10.0, 20.0, 30.0};
        elems[i].strain = {0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003};
        elems[i].plastic_strain = 0.01 * (i + 1);
        if (shell) {
            elems[i].connectivity = {1, 2, 3, 4};  // 4 nodes => shell
        } else {
            elems[i].connectivity = {1, 2, 3, 4, 5, 6, 7, 8};  // 8 nodes => solid
        }
    }
    return elems;
}

// ============================================================================
// 1. RadiossAnimWriter tests
// ============================================================================

void test_radioss_anim_writer() {
    std::cout << "--- RadiossAnimWriter tests ---\n";

    // Test 1: Constants
    CHECK(RadiossAnimWriter::MAGIC == 0x414E494D, "MAGIC constant");
    CHECK(RadiossAnimWriter::VERSION == 1, "VERSION constant");
    CHECK(RadiossAnimWriter::TITLE_LEN == 80, "TITLE_LEN constant");

    // Test 2: Default construction
    {
        RadiossAnimWriter writer;
        CHECK(writer.frame_count() == 0, "Default frame_count is 0");
        CHECK(writer.filename().empty(), "Default filename is empty");
    }

    // Test 3: Open file
    {
        RadiossAnimWriter writer;
        bool ok = writer.open("/tmp/nexussim_test_anim.anim");
        CHECK(ok, "Open anim file succeeds");
        CHECK(writer.filename() == "/tmp/nexussim_test_anim.anim", "Filename stored");
        writer.close();
    }

    // Test 4: Write single frame
    {
        RadiossAnimWriter writer;
        writer.open("/tmp/nexussim_test_anim2.anim");
        auto nodes = make_nodes(4);
        auto elems = make_elements(2);
        bool ok = writer.write_frame(0.0, nodes, elems);
        CHECK(ok, "write_frame returns true");
        CHECK(writer.frame_count() == 1, "frame_count after 1 write");
        writer.close();
    }

    // Test 5: Write multiple frames
    {
        RadiossAnimWriter writer;
        writer.open("/tmp/nexussim_test_anim3.anim");
        auto nodes = make_nodes(3);
        auto elems = make_elements(1);
        writer.write_frame(0.0, nodes, elems);
        writer.write_frame(0.001, nodes, elems);
        writer.write_frame(0.002, nodes, elems);
        CHECK(writer.frame_count() == 3, "frame_count after 3 writes");
        writer.close();
    }

    // Test 6: File exists and is non-empty after close
    {
        RadiossAnimWriter writer;
        writer.open("/tmp/nexussim_test_anim4.anim");
        auto nodes = make_nodes(2);
        auto elems = make_elements(1);
        writer.write_frame(0.0, nodes, elems);
        writer.close();

        std::ifstream f("/tmp/nexussim_test_anim4.anim", std::ios::binary | std::ios::ate);
        CHECK(f.is_open(), "Output anim file exists");
        auto sz = f.tellg();
        CHECK(sz > 0, "Output anim file is non-empty");
        f.close();
    }

    // Test 7: Write with empty nodes/elements
    {
        RadiossAnimWriter writer;
        writer.open("/tmp/nexussim_test_anim5.anim");
        std::vector<AnimNodeData> empty_nodes;
        std::vector<AnimElementData> empty_elems;
        bool ok = writer.write_frame(0.0, empty_nodes, empty_elems);
        CHECK(ok, "write_frame with empty data succeeds");
        writer.close();
    }

    // Test 8: Write fails on closed writer
    {
        RadiossAnimWriter writer;
        auto nodes = make_nodes(1);
        auto elems = make_elements(1);
        bool ok = writer.write_frame(0.0, nodes, elems);
        CHECK(!ok, "write_frame fails when file not opened");
    }

    // Test 9: Open invalid path
    {
        RadiossAnimWriter writer;
        bool ok = writer.open("/nonexistent_dir/test.anim");
        CHECK(!ok, "Open invalid path returns false");
    }
}

// ============================================================================
// 2. StatusFileWriter tests
// ============================================================================

void test_status_file_writer() {
    std::cout << "--- StatusFileWriter tests ---\n";

    // Test 10: Default construction
    {
        StatusFileWriter writer;
        CHECK(writer.line_count() == 0, "Default line_count is 0");
        CHECK(writer.filename().empty(), "Default filename is empty");
    }

    // Test 11: Open and write
    {
        StatusFileWriter writer;
        bool ok = writer.open("/tmp/nexussim_test_status.sta");
        CHECK(ok, "Open status file succeeds");

        EnergyData ed;
        ed.kinetic = 1000.0;
        ed.internal = 500.0;
        ed.contact = 10.0;
        ed.hourglass = 5.0;

        ok = writer.write_status(1, 0.001, 1e-6, ed, 0.0);
        CHECK(ok, "write_status returns true");
        CHECK(writer.line_count() == 1, "line_count after 1 write");
        writer.close();
    }

    // Test 12: Multiple status lines
    {
        StatusFileWriter writer;
        writer.open("/tmp/nexussim_test_status2.sta");

        EnergyData ed;
        ed.kinetic = 1000.0;
        ed.internal = 500.0;
        for (int i = 0; i < 5; ++i) {
            writer.write_status(i + 1, 0.001 * (i + 1), 1e-6, ed);
        }
        CHECK(writer.line_count() == 5, "line_count after 5 writes");
        writer.close();
    }

    // Test 13: EnergyData total
    {
        EnergyData ed;
        ed.kinetic = 100.0;
        ed.internal = 200.0;
        ed.contact = 30.0;
        ed.hourglass = 5.0;
        ed.external_work = 50.0;  // external_work not in total()
        CHECK_NEAR(ed.total(), 335.0, 1e-10, "EnergyData::total() = kinetic+internal+contact+hourglass");
    }

    // Test 14: Status file content check
    {
        StatusFileWriter writer;
        writer.open("/tmp/nexussim_test_status3.sta");
        EnergyData ed;
        ed.kinetic = 123.0;
        ed.internal = 456.0;
        writer.write_status(10, 0.01, 1e-5, ed, 0.001);
        writer.close();

        std::ifstream f("/tmp/nexussim_test_status3.sta");
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        CHECK(content.find("CYCLE") != std::string::npos, "Status file has header CYCLE");
        CHECK(content.find("TIME") != std::string::npos, "Status file has header TIME");
        f.close();
    }

    // Test 15: Write fails when not opened
    {
        StatusFileWriter writer;
        EnergyData ed;
        bool ok = writer.write_status(1, 0.0, 0.0, ed);
        CHECK(!ok, "write_status fails when not opened");
    }

    // Test 16: Filename stored
    {
        StatusFileWriter writer;
        writer.open("/tmp/nexussim_test_status4.sta");
        CHECK(writer.filename() == "/tmp/nexussim_test_status4.sta", "Filename stored correctly");
        writer.close();
    }
}

// ============================================================================
// 3. DynainWriter tests
// ============================================================================

void test_dynain_writer() {
    std::cout << "--- DynainWriter tests ---\n";

    // Test 17: Default construction
    {
        DynainWriter writer;
        CHECK(writer.nodes_written() == 0, "Default nodes_written is 0");
        CHECK(writer.elements_written() == 0, "Default elements_written is 0");
    }

    // Test 18: Write solid elements
    {
        DynainWriter writer;
        auto nodes = make_nodes(8);
        auto elems = make_elements(2, false);  // solid
        bool ok = writer.write("/tmp/nexussim_test_dynain.key", nodes, elems);
        CHECK(ok, "Write dynain file succeeds");
        CHECK(writer.nodes_written() == 8, "nodes_written after write");
        CHECK(writer.elements_written() == 2, "elements_written after write");
    }

    // Test 19: File contains keyword markers
    {
        DynainWriter writer;
        auto nodes = make_nodes(4);
        auto elems = make_elements(1, false);
        writer.write("/tmp/nexussim_test_dynain2.key", nodes, elems);

        std::ifstream f("/tmp/nexussim_test_dynain2.key");
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        CHECK(content.find("*KEYWORD") != std::string::npos, "Dynain has *KEYWORD");
        CHECK(content.find("*NODE") != std::string::npos, "Dynain has *NODE");
        CHECK(content.find("*ELEMENT_SOLID") != std::string::npos, "Dynain has *ELEMENT_SOLID");
        CHECK(content.find("*INITIAL_STRESS_SOLID") != std::string::npos, "Dynain has *INITIAL_STRESS_SOLID");
        CHECK(content.find("*END") != std::string::npos, "Dynain has *END");
        f.close();
    }

    // Test 20: Write shell elements
    {
        DynainWriter writer;
        auto nodes = make_nodes(4);
        auto elems = make_elements(2, true);  // shell (4-node connectivity)
        writer.write("/tmp/nexussim_test_dynain3.key", nodes, elems);

        std::ifstream f("/tmp/nexussim_test_dynain3.key");
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        CHECK(content.find("*ELEMENT_SHELL") != std::string::npos, "Dynain has *ELEMENT_SHELL for shell elements");
        f.close();
    }

    // Test 21: Write with no elements
    {
        DynainWriter writer;
        auto nodes = make_nodes(3);
        std::vector<AnimElementData> empty_elems;
        bool ok = writer.write("/tmp/nexussim_test_dynain4.key", nodes, empty_elems);
        CHECK(ok, "Write with no elements succeeds");
        CHECK(writer.nodes_written() == 3, "nodes_written with no elements");
        CHECK(writer.elements_written() == 0, "elements_written is 0");
    }

    // Test 22: Write to invalid path
    {
        DynainWriter writer;
        auto nodes = make_nodes(1);
        auto elems = make_elements(1);
        bool ok = writer.write("/nonexistent_dir/test.key", nodes, elems);
        CHECK(!ok, "Write to invalid path returns false");
    }

    // Test 23: Comment line present
    {
        DynainWriter writer;
        auto nodes = make_nodes(2);
        auto elems = make_elements(1, false);
        writer.write("/tmp/nexussim_test_dynain5.key", nodes, elems);

        std::ifstream f("/tmp/nexussim_test_dynain5.key");
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        CHECK(content.find("$ NexusSim") != std::string::npos, "Dynain has comment line");
        f.close();
    }
}

// ============================================================================
// 4. ReactionForcesTH tests
// ============================================================================

void test_reaction_forces_th() {
    std::cout << "--- ReactionForcesTH tests ---\n";

    // Test 24: Default construction
    {
        ReactionForcesTH th;
        CHECK(th.num_records() == 0, "Default num_records is 0");
        CHECK(th.tracked_nodes().empty(), "Default tracked_nodes is empty");
        CHECK(th.total_entries() == 0, "Default total_entries is 0");
    }

    // Test 25: Add nodes
    {
        ReactionForcesTH th;
        th.add_node(10);
        th.add_node(20);
        CHECK(static_cast<int>(th.tracked_nodes().size()) == 2, "tracked_nodes size after add");
        CHECK(th.tracked_nodes()[0] == 10, "tracked_nodes[0]");
        CHECK(th.tracked_nodes()[1] == 20, "tracked_nodes[1]");
    }

    // Test 26: Record reaction forces
    {
        ReactionForcesTH th;
        th.add_node(1);
        th.add_node(2);

        std::vector<ReactionForce> forces;
        ReactionForce f1; f1.node_id = 1; f1.fx = 100.0; f1.fy = 200.0; f1.fz = 300.0;
        ReactionForce f2; f2.node_id = 2; f2.fx = 50.0; f2.fy = 60.0; f2.fz = 70.0;
        forces.push_back(f1);
        forces.push_back(f2);

        th.record(0.001, forces);
        CHECK(th.num_records() == 1, "num_records after 1 record");
        CHECK(th.total_entries() == 2, "total_entries after recording 2 nodes");
    }

    // Test 27: Only tracked nodes are stored
    {
        ReactionForcesTH th;
        th.add_node(1);  // Only track node 1

        std::vector<ReactionForce> forces;
        ReactionForce f1; f1.node_id = 1; f1.fx = 100.0;
        ReactionForce f2; f2.node_id = 99; f2.fx = 999.0;  // Not tracked
        forces.push_back(f1);
        forces.push_back(f2);

        th.record(0.0, forces);
        CHECK(th.total_entries() == 1, "Only tracked nodes stored");
    }

    // Test 28: Resultant force
    {
        ReactionForcesTH th;
        th.add_node(1);
        th.add_node(2);

        std::vector<ReactionForce> forces;
        ReactionForce f1; f1.node_id = 1; f1.fx = 100.0; f1.fy = 200.0; f1.fz = 300.0;
        f1.mx = 10.0; f1.my = 20.0; f1.mz = 30.0;
        ReactionForce f2; f2.node_id = 2; f2.fx = 50.0; f2.fy = 60.0; f2.fz = 70.0;
        f2.mx = 5.0; f2.my = 6.0; f2.mz = 7.0;
        forces.push_back(f1);
        forces.push_back(f2);

        th.record(0.001, forces);
        auto res = th.resultant(0);
        CHECK_NEAR(res.fx, 150.0, 1e-10, "Resultant fx");
        CHECK_NEAR(res.fy, 260.0, 1e-10, "Resultant fy");
        CHECK_NEAR(res.fz, 370.0, 1e-10, "Resultant fz");
        CHECK_NEAR(res.mx, 15.0, 1e-10, "Resultant mx");
        CHECK_NEAR(res.my, 26.0, 1e-10, "Resultant my");
        CHECK_NEAR(res.mz, 37.0, 1e-10, "Resultant mz");
    }

    // Test 29: Resultant out of range
    {
        ReactionForcesTH th;
        auto res = th.resultant(-1);
        CHECK_NEAR(res.fx, 0.0, 1e-10, "Resultant out of range returns zero fx");
        auto res2 = th.resultant(100);
        CHECK_NEAR(res2.fy, 0.0, 1e-10, "Resultant out of range returns zero fy");
    }

    // Test 30: Multiple records
    {
        ReactionForcesTH th;
        th.add_node(1);

        for (int i = 0; i < 5; ++i) {
            std::vector<ReactionForce> forces;
            ReactionForce f; f.node_id = 1; f.fx = 10.0 * (i + 1);
            forces.push_back(f);
            th.record(0.001 * (i + 1), forces);
        }
        CHECK(th.num_records() == 5, "num_records after 5 records");
        CHECK(th.total_entries() == 5, "total_entries after 5 records");
    }

    // Test 31: Write to file
    {
        ReactionForcesTH th;
        th.add_node(1);

        std::vector<ReactionForce> forces;
        ReactionForce f; f.node_id = 1; f.fx = 100.0; f.fy = 200.0; f.fz = 300.0;
        forces.push_back(f);
        th.record(0.001, forces);

        bool ok = th.write("/tmp/nexussim_test_reaction.th");
        CHECK(ok, "Write reaction TH file succeeds");

        std::ifstream file("/tmp/nexussim_test_reaction.th");
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        CHECK(content.find("TIME") != std::string::npos, "TH file has TIME header");
        CHECK(content.find("NODE_ID") != std::string::npos, "TH file has NODE_ID header");
        CHECK(content.find("FX") != std::string::npos, "TH file has FX header");
        file.close();
    }

    // Test 32: Write to invalid path
    {
        ReactionForcesTH th;
        bool ok = th.write("/nonexistent_dir/test.th");
        CHECK(!ok, "Write to invalid path returns false");
    }
}

// ============================================================================
// 5. QAPrintWriter tests
// ============================================================================

void test_qa_print_writer() {
    std::cout << "--- QAPrintWriter tests ---\n";

    // Test 33: Default state
    {
        QAPrintWriter qa;
        CHECK(!qa.is_analyzed(), "Default is_analyzed is false");
        CHECK(qa.warning_count() == 0, "Default warning_count is 0");
        CHECK(qa.error_count() == 0, "Default error_count is 0");
    }

    // Test 34: Analyze clean mesh
    {
        MeshInfo info;
        info.node_count = 100;
        info.elem_count = 50;
        info.material_count = 3;
        info.part_count = 2;
        info.shell_count = 30;
        info.solid_count = 20;
        info.beam_count = 0;
        info.min_aspect_ratio = 1.0;
        info.max_aspect_ratio = 3.0;
        info.min_jacobian = 0.5;
        info.max_jacobian = 1.0;
        info.max_warping = 5.0;
        info.unassigned_material_count = 0;

        QAPrintWriter qa;
        qa.analyze(info);
        CHECK(qa.is_analyzed(), "is_analyzed after analyze");
        CHECK(qa.warning_count() == 0, "No warnings for clean mesh");
        CHECK(qa.error_count() == 0, "No errors for clean mesh");
    }

    // Test 35: mesh_info accessor
    {
        MeshInfo info;
        info.node_count = 42;
        info.elem_count = 17;

        QAPrintWriter qa;
        qa.analyze(info);
        CHECK(qa.mesh_info().node_count == 42, "mesh_info().node_count");
        CHECK(qa.mesh_info().elem_count == 17, "mesh_info().elem_count");
    }

    // Test 36: Negative Jacobian triggers error
    {
        MeshInfo info;
        info.min_jacobian = -0.5;

        QAPrintWriter qa;
        qa.analyze(info);
        CHECK(qa.error_count() >= 1, "Negative Jacobian triggers error");
    }

    // Test 37: High aspect ratio triggers warning
    {
        MeshInfo info;
        info.max_aspect_ratio = 15.0;

        QAPrintWriter qa;
        qa.analyze(info);
        CHECK(qa.warning_count() >= 1, "High aspect ratio triggers warning");
    }

    // Test 38: Excessive warping triggers warning
    {
        MeshInfo info;
        info.max_warping = 20.0;

        QAPrintWriter qa;
        qa.analyze(info);
        CHECK(qa.warning_count() >= 1, "Excessive warping triggers warning");
    }

    // Test 39: Unassigned material triggers error
    {
        MeshInfo info;
        info.unassigned_material_count = 5;

        QAPrintWriter qa;
        qa.analyze(info);
        CHECK(qa.error_count() >= 1, "Unassigned material triggers error");
    }

    // Test 40: Multiple issues
    {
        MeshInfo info;
        info.min_jacobian = -0.1;          // error
        info.max_aspect_ratio = 20.0;      // warning
        info.max_warping = 30.0;           // warning
        info.unassigned_material_count = 3; // error

        QAPrintWriter qa;
        qa.analyze(info);
        CHECK(qa.warning_count() == 2, "Two warnings for aspect+warping");
        CHECK(qa.error_count() == 2, "Two errors for jacobian+unassigned");
    }

    // Test 41: MeshInfo warnings/errors are merged
    {
        MeshInfo info;
        info.warnings.push_back("User warning 1");
        info.errors.push_back("User error 1");

        QAPrintWriter qa;
        qa.analyze(info);
        CHECK(qa.warning_count() >= 1, "MeshInfo warnings merged");
        CHECK(qa.error_count() >= 1, "MeshInfo errors merged");
    }

    // Test 42: Write QA report
    {
        MeshInfo info;
        info.node_count = 1000;
        info.elem_count = 500;
        info.shell_count = 300;
        info.solid_count = 200;
        info.material_count = 5;
        info.part_count = 3;

        QAPrintWriter qa;
        qa.analyze(info);
        bool ok = qa.write("/tmp/nexussim_test_qa.txt");
        CHECK(ok, "Write QA report succeeds");

        std::ifstream f("/tmp/nexussim_test_qa.txt");
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        CHECK(content.find("Quality Assurance Report") != std::string::npos, "QA file has title");
        CHECK(content.find("MODEL SUMMARY") != std::string::npos, "QA file has MODEL SUMMARY");
        CHECK(content.find("MESH QUALITY") != std::string::npos, "QA file has MESH QUALITY");
        CHECK(content.find("End of QA Report") != std::string::npos, "QA file has end marker");
        f.close();
    }

    // Test 43: Write to invalid path
    {
        QAPrintWriter qa;
        MeshInfo info;
        qa.analyze(info);
        bool ok = qa.write("/nonexistent_dir/qa.txt");
        CHECK(!ok, "Write to invalid path returns false");
    }
}

// ============================================================================
// 6. ReportGenerator tests
// ============================================================================

void test_report_generator() {
    std::cout << "--- ReportGenerator tests ---\n";

    // Test 44: Default state
    {
        ReportGenerator rg;
        CHECK(rg.num_sections() == 0, "Default num_sections is 0");
    }

    // Test 45: Add section
    {
        ReportGenerator rg;
        rg.add_section("TEST", "test content");
        CHECK(rg.num_sections() == 1, "num_sections after add_section");
    }

    // Test 46: Add multiple sections
    {
        ReportGenerator rg;
        rg.add_section("A", "a");
        rg.add_section("B", "b");
        rg.add_section("C", "c");
        CHECK(rg.num_sections() == 3, "num_sections after 3 add_section");
    }

    // Test 47: Clear
    {
        ReportGenerator rg;
        rg.add_section("A", "a");
        rg.add_section("B", "b");
        rg.clear();
        CHECK(rg.num_sections() == 0, "num_sections after clear");
    }

    // Test 48: add_run_info
    {
        ReportGenerator rg;
        rg.add_run_info("Explicit", 0.01, 1e-6, 8, "AMD Ryzen");
        CHECK(rg.num_sections() == 1, "add_run_info adds 1 section");
    }

    // Test 49: add_energy_summary
    {
        ReportGenerator rg;
        EnergyData initial;
        initial.kinetic = 1000.0;
        initial.internal = 0.0;
        EnergyData final_e;
        final_e.kinetic = 500.0;
        final_e.internal = 490.0;
        rg.add_energy_summary(initial, final_e);
        CHECK(rg.num_sections() == 1, "add_energy_summary adds 1 section");
    }

    // Test 50: add_timing
    {
        ReportGenerator rg;
        rg.add_timing(120.5, 450.3, 10000);
        CHECK(rg.num_sections() == 1, "add_timing adds 1 section");
    }

    // Test 51: add_contact_summary
    {
        ReportGenerator rg;
        rg.add_contact_summary(5, 100000, 250);
        CHECK(rg.num_sections() == 1, "add_contact_summary adds 1 section");
    }

    // Test 52: Generate report
    {
        ReportGenerator rg;
        rg.add_run_info("Explicit", 0.01, 1e-6, 8, "AMD Ryzen");

        EnergyData initial; initial.kinetic = 1000.0;
        EnergyData final_e; final_e.kinetic = 500.0; final_e.internal = 500.0;
        rg.add_energy_summary(initial, final_e);

        rg.add_timing(120.5, 450.3, 10000);
        rg.add_contact_summary(3, 50000, 100);
        rg.add_section("NOTES", "Simulation completed successfully.");

        bool ok = rg.generate("/tmp/nexussim_test_report.txt");
        CHECK(ok, "Generate report succeeds");
        CHECK(rg.num_sections() == 5, "num_sections in full report");
    }

    // Test 53: Report file content verification
    {
        std::ifstream f("/tmp/nexussim_test_report.txt");
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        CHECK(content.find("Simulation Summary Report") != std::string::npos, "Report has title");
        CHECK(content.find("RUN PARAMETERS") != std::string::npos, "Report has RUN PARAMETERS");
        CHECK(content.find("ENERGY BALANCE") != std::string::npos, "Report has ENERGY BALANCE");
        CHECK(content.find("TIMING") != std::string::npos, "Report has TIMING");
        CHECK(content.find("CONTACT SUMMARY") != std::string::npos, "Report has CONTACT SUMMARY");
        CHECK(content.find("NOTES") != std::string::npos, "Report has NOTES");
        CHECK(content.find("End of Report") != std::string::npos, "Report has end marker");
        f.close();
    }

    // Test 54: Generate to invalid path
    {
        ReportGenerator rg;
        rg.add_section("X", "y");
        bool ok = rg.generate("/nonexistent_dir/report.txt");
        CHECK(!ok, "Generate to invalid path returns false");
    }

    // Test 55: add_timing with zero cycles
    {
        ReportGenerator rg;
        rg.add_timing(10.0, 20.0, 0);
        CHECK(rg.num_sections() == 1, "add_timing with 0 cycles adds section");

        bool ok = rg.generate("/tmp/nexussim_test_report_zero.txt");
        CHECK(ok, "Generate with 0 cycles succeeds");
    }
}

// ============================================================================
// Data structure tests
// ============================================================================

void test_data_structures() {
    std::cout << "--- Data structure tests ---\n";

    // Test 56: AnimNodeData defaults
    {
        AnimNodeData n;
        CHECK(n.id == 0, "AnimNodeData default id");
        CHECK_NEAR(n.x, 0.0, 1e-15, "AnimNodeData default x");
        CHECK_NEAR(n.vx, 0.0, 1e-15, "AnimNodeData default vx");
        CHECK_NEAR(n.ax, 0.0, 1e-15, "AnimNodeData default ax");
    }

    // Test 57: AnimElementData defaults
    {
        AnimElementData e;
        CHECK(e.id == 0, "AnimElementData default id");
        CHECK(e.part_id == 0, "AnimElementData default part_id");
        CHECK(e.type == 0, "AnimElementData default type");
        CHECK_NEAR(e.plastic_strain, 0.0, 1e-15, "AnimElementData default plastic_strain");
        CHECK(e.connectivity.empty(), "AnimElementData default connectivity empty");
    }

    // Test 58: EnergyData defaults and total
    {
        EnergyData ed;
        CHECK_NEAR(ed.kinetic, 0.0, 1e-15, "EnergyData default kinetic");
        CHECK_NEAR(ed.total(), 0.0, 1e-15, "EnergyData default total");
    }

    // Test 59: ReactionForce defaults
    {
        ReactionForce rf;
        CHECK(rf.node_id == 0, "ReactionForce default node_id");
        CHECK_NEAR(rf.fx, 0.0, 1e-15, "ReactionForce default fx");
        CHECK_NEAR(rf.mx, 0.0, 1e-15, "ReactionForce default mx");
    }

    // Test 60: MeshInfo defaults
    {
        MeshInfo mi;
        CHECK(mi.node_count == 0, "MeshInfo default node_count");
        CHECK(mi.elem_count == 0, "MeshInfo default elem_count");
        CHECK_NEAR(mi.min_aspect_ratio, 1.0, 1e-15, "MeshInfo default min_aspect_ratio");
        CHECK_NEAR(mi.max_aspect_ratio, 1.0, 1e-15, "MeshInfo default max_aspect_ratio");
        CHECK_NEAR(mi.min_jacobian, 1.0, 1e-15, "MeshInfo default min_jacobian");
        CHECK_NEAR(mi.max_warping, 0.0, 1e-15, "MeshInfo default max_warping");
        CHECK(mi.warnings.empty(), "MeshInfo default warnings empty");
        CHECK(mi.errors.empty(), "MeshInfo default errors empty");
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Wave 40: Extended Output Format Writers Tests ===\n\n";

    test_radioss_anim_writer();
    test_status_file_writer();
    test_dynain_writer();
    test_reaction_forces_th();
    test_qa_print_writer();
    test_report_generator();
    test_data_structures();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " tests ===\n";

    return tests_failed > 0 ? 1 : 0;
}
