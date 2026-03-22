/**
 * @file th_wave44_test.cpp
 * @brief Tests for Wave 44: RADIOSS .TH Binary Time History Writer
 *
 * 12 tests covering:
 *   1.  Write/read roundtrip — single node group
 *   2.  Write/read roundtrip — single shell element group
 *   3.  Multi-group write/read — node + shell + solid
 *   4.  Per-entity-type records — each THEntityType
 *   5.  Large model — 100 variables, 1000 time steps
 *   6.  Append mode — write initial records, then append more
 *   7.  Format validation — Fortran record markers are correct
 *   8.  Empty group handling — group with no records
 *   9.  RecorderToTH conversion — populate recorder, convert, verify
 *  10.  Variable info preservation — names and units survive roundtrip
 *  11.  Time ordering — records are time-ordered after write
 *  12.  File creation — .T01 extension, file exists, non-empty
 */

#include <nexussim/io/th_wave44.hpp>
#include <nexussim/io/time_history.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace nxs::io;

// ============================================================================
// Test harness
// ============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++tests_passed; } \
    else { ++tests_failed; \
           std::cout << "[FAIL] " << (msg) << "\n"; } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) do { \
    double _va = static_cast<double>(a); \
    double _vb = static_cast<double>(b); \
    double _vt = static_cast<double>(tol); \
    if (std::abs(_va - _vb) <= _vt) { ++tests_passed; } \
    else { ++tests_failed; \
           std::cout << "[FAIL] " << (msg) \
                     << " (got " << _va << ", expected " << _vb \
                     << ", tol " << _vt << ")\n"; } \
} while(0)

// ============================================================================
// Test 1: Write/read roundtrip — single node group
// ============================================================================

static void test_node_roundtrip() {
    std::cout << "--- Test 1: node group roundtrip ---\n";

    const std::string base = "/tmp/nxs_th44_test1";
    const std::string path = base + "_nodegroup.T01";

    // Build writer
    RadiossTHWriter writer;
    writer.set_title("Test1");
    writer.set_run_id(1);

    THGroup& g = writer.add_node_group("nodegroup", {10, 20}, {"DX", "VX"});
    // node 10: DX, VX — node 20: DX, VX  => 4 variables total

    // Write 3 time steps
    for (int step = 0; step < 3; ++step) {
        double t = step * 0.01;
        std::vector<Real> vals = {
            1.0 + step,  // node10 DX
            0.1 + step,  // node10 VX
            2.0 + step,  // node20 DX
            0.2 + step   // node20 VX
        };
        g.add_record(static_cast<Real>(t), vals);
    }

    bool ok = writer.write(base);
    CHECK(ok, "write returns true");

    // Read back
    THReader reader;
    bool rok = reader.read(path);
    CHECK(rok, "read returns true");
    CHECK(reader.num_groups() == 1, "one group after read");

    const auto& groups = reader.groups();
    CHECK(!groups.empty(), "groups not empty");

    if (!groups.empty()) {
        const THGroup& rg = groups[0];
        CHECK(rg.num_variables() == 4, "4 variables read back");
        CHECK(rg.num_records()   == 3, "3 records read back");

        if (rg.num_records() == 3) {
            CHECK_NEAR(rg.records[0].time, 0.00, 1e-9, "t[0]==0.00");
            CHECK_NEAR(rg.records[1].time, 0.01, 1e-9, "t[1]==0.01");
            CHECK_NEAR(rg.records[2].time, 0.02, 1e-9, "t[2]==0.02");

            if (rg.records[0].values.size() == 4) {
                CHECK_NEAR(rg.records[0].values[0], 1.0, 1e-9, "step0 node10 DX");
                CHECK_NEAR(rg.records[0].values[1], 0.1, 1e-9, "step0 node10 VX");
                CHECK_NEAR(rg.records[0].values[2], 2.0, 1e-9, "step0 node20 DX");
                CHECK_NEAR(rg.records[0].values[3], 0.2, 1e-9, "step0 node20 VX");
            }
            if (rg.records[2].values.size() == 4) {
                CHECK_NEAR(rg.records[2].values[0], 3.0, 1e-9, "step2 node10 DX");
                CHECK_NEAR(rg.records[2].values[2], 4.0, 1e-9, "step2 node20 DX");
            }
        }
    }
}

// ============================================================================
// Test 2: Write/read roundtrip — single shell element group
// ============================================================================

static void test_shell_roundtrip() {
    std::cout << "--- Test 2: shell element group roundtrip ---\n";

    const std::string base = "/tmp/nxs_th44_test2";
    const std::string path = base + "_shellgrp.T01";

    RadiossTHWriter writer;
    writer.set_title("ShellTest");
    writer.set_run_id(2);

    THGroup& g = writer.add_element_group(
        "shellgrp", THEntityType::Shell, {100, 200}, {"SXX", "SYY"});
    // elem 100: SXX, SYY — elem 200: SXX, SYY => 4 variables

    std::vector<Real> vals0 = {100.0, 200.0, 300.0, 400.0};
    std::vector<Real> vals1 = {101.0, 201.0, 301.0, 401.0};
    g.add_record(0.0f, vals0);
    g.add_record(0.01f, vals1);

    bool ok = writer.write(base);
    CHECK(ok, "shell write ok");

    THReader reader;
    bool rok = reader.read(path);
    CHECK(rok, "shell read ok");

    const auto& groups = reader.groups();
    CHECK(!groups.empty(), "shell groups not empty");

    if (!groups.empty()) {
        const THGroup& rg = groups[0];
        CHECK(rg.num_variables() == 4, "shell: 4 variables");
        CHECK(rg.num_records()   == 2, "shell: 2 records");

        if (rg.num_records() >= 2 && rg.records[0].values.size() == 4) {
            CHECK_NEAR(rg.records[0].values[0], 100.0, 1e-6, "shell sxx elem100 t=0");
            CHECK_NEAR(rg.records[1].values[1], 201.0, 1e-6, "shell syy elem100 t=1");
            CHECK_NEAR(rg.records[0].values[2], 300.0, 1e-6, "shell sxx elem200 t=0");
        }
    }
}

// ============================================================================
// Test 3: Multi-group write/read — node + shell + solid
// ============================================================================

static void test_multi_group() {
    std::cout << "--- Test 3: multi-group (node + shell + solid) ---\n";

    const std::string base = "/tmp/nxs_th44_test3";

    RadiossTHWriter writer;
    writer.set_title("MultiGroup");
    writer.set_run_id(3);

    // Node group
    THGroup& ng = writer.add_node_group("nodes", {1, 2}, {"DX"});
    ng.add_record(0.0, {1.0, 2.0});
    ng.add_record(0.01, {1.1, 2.1});

    // Shell group
    THGroup& sg = writer.add_element_group(
        "shells", THEntityType::Shell, {10}, {"SXX"});
    sg.add_record(0.0,  {500.0});
    sg.add_record(0.01, {510.0});

    // Solid group
    THGroup& solid_g = writer.add_element_group(
        "solids", THEntityType::Solid, {20}, {"SXX", "SYY", "SZZ"});
    solid_g.add_record(0.0,  {100.0, 200.0, 300.0});
    solid_g.add_record(0.01, {110.0, 210.0, 310.0});

    bool ok = writer.write(base);
    CHECK(ok, "multi-group write ok");
    CHECK(writer.num_groups() == 3, "writer has 3 groups");

    // Read back all three files
    {
        THReader r;
        bool rok = r.read(base + "_nodes.T01");
        CHECK(rok, "nodes group read ok");
        if (rok && !r.groups().empty()) {
            CHECK(r.groups()[0].num_variables() == 2, "nodes: 2 vars");
            CHECK(r.groups()[0].num_records()   == 2, "nodes: 2 records");
        }
    }
    {
        THReader r;
        bool rok = r.read(base + "_shells.T01");
        CHECK(rok, "shells group read ok");
        if (rok && !r.groups().empty()) {
            CHECK(r.groups()[0].num_variables() == 1, "shells: 1 var");
            CHECK_NEAR(r.groups()[0].records[1].values[0], 510.0, 1e-6,
                       "shells: sxx at t=1");
        }
    }
    {
        THReader r;
        bool rok = r.read(base + "_solids.T01");
        CHECK(rok, "solids group read ok");
        if (rok && !r.groups().empty()) {
            CHECK(r.groups()[0].num_variables() == 3, "solids: 3 vars");
            CHECK_NEAR(r.groups()[0].records[0].values[2], 300.0, 1e-6,
                       "solids: szz at t=0");
        }
    }
}

// ============================================================================
// Test 4: Per-entity-type records — each THEntityType
// ============================================================================

static void test_entity_types() {
    std::cout << "--- Test 4: per-entity-type ---\n";

    const std::string base = "/tmp/nxs_th44_test4";

    struct TypeCase {
        THEntityType type;
        std::string  name;
    };

    std::vector<TypeCase> cases = {
        { THEntityType::Node,      "type_node"  },
        { THEntityType::Shell,     "type_shell" },
        { THEntityType::Solid,     "type_solid" },
        { THEntityType::Beam,      "type_beam"  },
        { THEntityType::Interface, "type_inter" },
        { THEntityType::RigidBody, "type_rbody" },
        { THEntityType::Sensor,    "type_sens"  }
    };

    for (const auto& c : cases) {
        RadiossTHWriter writer;
        writer.set_title("TypeTest");
        writer.set_run_id(4);

        THGroup& g = writer.add_group(c.name, c.type);
        THVariableInfo vi("VAL", "unit", 1, c.type);
        g.variables.push_back(vi);
        g.add_record(0.0, {42.0});
        g.add_record(0.1, {43.0});

        bool ok = writer.write(base);
        CHECK(ok, "entity type write: " + c.name);

        std::string path = base + "_" + c.name + ".T01";
        THReader reader;
        bool rok = reader.read(path);
        CHECK(rok, "entity type read: " + c.name);

        if (rok && !reader.groups().empty()) {
            const THGroup& rg = reader.groups()[0];
            CHECK(rg.num_records() == 2,
                  "entity type records: " + c.name);
            if (rg.num_records() == 2) {
                CHECK_NEAR(rg.records[0].values[0], 42.0, 1e-9,
                           "entity val[0]: " + c.name);
                CHECK_NEAR(rg.records[1].values[0], 43.0, 1e-9,
                           "entity val[1]: " + c.name);
            }
            if (!rg.variables.empty()) {
                CHECK(rg.variables[0].entity_type == c.type,
                      "entity type preserved: " + c.name);
            }
        }
    }
}

// ============================================================================
// Test 5: Large model — 100 variables, 1000 time steps
// ============================================================================

static void test_large_model() {
    std::cout << "--- Test 5: large model (100 vars, 1000 steps) ---\n";

    const std::string base = "/tmp/nxs_th44_test5";
    const std::string path = base + "_biggrp.T01";

    const int NUM_VARS  = 100;
    const int NUM_STEPS = 1000;

    RadiossTHWriter writer;
    writer.set_title("LargeModel");
    writer.set_run_id(5);

    THGroup& g = writer.add_group("biggrp", THEntityType::Solid);
    for (int v = 0; v < NUM_VARS; ++v) {
        THVariableInfo vi("V" + std::to_string(v), "MPa",
                          v + 1, THEntityType::Solid);
        g.variables.push_back(vi);
    }

    for (int step = 0; step < NUM_STEPS; ++step) {
        Real t = static_cast<Real>(step) * 1e-4;
        std::vector<Real> vals(static_cast<std::size_t>(NUM_VARS));
        for (int v = 0; v < NUM_VARS; ++v) {
            vals[static_cast<std::size_t>(v)] =
                static_cast<Real>(step * NUM_VARS + v);
        }
        g.add_record(t, vals);
    }

    bool ok = writer.write(base);
    CHECK(ok, "large model write ok");

    THReader reader;
    bool rok = reader.read(path);
    CHECK(rok, "large model read ok");

    if (rok && !reader.groups().empty()) {
        const THGroup& rg = reader.groups()[0];
        CHECK(rg.num_variables() == static_cast<std::size_t>(NUM_VARS),
              "large: 100 variables");
        CHECK(rg.num_records()   == static_cast<std::size_t>(NUM_STEPS),
              "large: 1000 records");

        // Spot-check last record
        if (rg.num_records() == static_cast<std::size_t>(NUM_STEPS)) {
            int last = NUM_STEPS - 1;
            CHECK_NEAR(rg.records[static_cast<std::size_t>(last)].time,
                       static_cast<Real>(last) * 1e-4, 1e-8,
                       "large: last record time");
            CHECK_NEAR(
                rg.records[static_cast<std::size_t>(last)].values[0],
                static_cast<Real>(last * NUM_VARS + 0), 1e-3,
                "large: last record val[0]");
            CHECK_NEAR(
                rg.records[static_cast<std::size_t>(last)].values[99],
                static_cast<Real>(last * NUM_VARS + 99), 1e-3,
                "large: last record val[99]");
        }
    }
}

// ============================================================================
// Test 6: Append mode
// ============================================================================

static void test_append_mode() {
    std::cout << "--- Test 6: append mode ---\n";

    const std::string base = "/tmp/nxs_th44_test6";
    const std::string path = base + "_appgrp.T01";

    // --- Initial write: steps 0, 1, 2 ---
    {
        RadiossTHWriter writer;
        writer.set_title("AppendTest");
        writer.set_run_id(6);

        THGroup& g = writer.add_group("appgrp", THEntityType::Node);
        THVariableInfo vi("DX", "mm", 1, THEntityType::Node);
        g.variables.push_back(vi);

        g.add_record(0.00, {10.0});
        g.add_record(0.01, {11.0});
        g.add_record(0.02, {12.0});

        bool ok = writer.write(base);
        CHECK(ok, "append: initial write ok");
    }

    // --- Append: steps 3, 4 (time > 0.02) ---
    {
        RadiossTHWriter writer;
        writer.set_title("AppendTest");
        writer.set_run_id(6);

        THGroup& g = writer.add_group("appgrp", THEntityType::Node);
        THVariableInfo vi("DX", "mm", 1, THEntityType::Node);
        g.variables.push_back(vi);

        // All records — only those with time > start_time=0.02 will be appended
        g.add_record(0.00, {10.0}); // skipped
        g.add_record(0.01, {11.0}); // skipped
        g.add_record(0.02, {12.0}); // skipped (not strictly >)
        g.add_record(0.03, {13.0}); // appended
        g.add_record(0.04, {14.0}); // appended

        bool ok = writer.append(base, 0.02);
        CHECK(ok, "append: append mode ok");
    }

    // --- Read file: expect header + 3 original + 2 appended = 5 records ---
    {
        THReader reader;
        bool rok = reader.read(path);
        CHECK(rok, "append: read ok");

        if (rok && !reader.groups().empty()) {
            const THGroup& rg = reader.groups()[0];
            // Header was from original write; records = 3 + 2 = 5
            CHECK(rg.num_records() == 5, "append: 5 records total");

            if (rg.num_records() >= 5) {
                CHECK_NEAR(rg.records[0].values[0], 10.0, 1e-9, "append: rec0");
                CHECK_NEAR(rg.records[2].values[0], 12.0, 1e-9, "append: rec2");
                CHECK_NEAR(rg.records[3].values[0], 13.0, 1e-9, "append: rec3");
                CHECK_NEAR(rg.records[4].values[0], 14.0, 1e-9, "append: rec4");
            }
        }
    }
}

// ============================================================================
// Test 7: Format validation — Fortran record markers
// ============================================================================

static void test_fortran_markers() {
    std::cout << "--- Test 7: Fortran record markers ---\n";

    const std::string base = "/tmp/nxs_th44_test7";
    const std::string path = base + "_fmtgrp.T01";

    RadiossTHWriter writer;
    writer.set_title("FmtTest");
    writer.set_run_id(7);

    THGroup& g = writer.add_group("fmtgrp", THEntityType::Node);
    THVariableInfo vi("DX", "mm", 5, THEntityType::Node);
    g.variables.push_back(vi);
    g.add_record(0.0, {3.14});

    bool ok = writer.write(base);
    CHECK(ok, "format: write ok");

    // Manually parse the raw bytes of the file
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    CHECK(f.is_open(), "format: file opens for manual parse");

    if (f.is_open()) {
        auto file_size = f.tellg();
        CHECK(file_size > 0, "format: file non-empty");

        f.seekg(0, std::ios::beg);

        // --- Record 1: header ---
        int32_t n1_lead = 0;
        f.read(reinterpret_cast<char*>(&n1_lead), 4);
        CHECK(n1_lead > 0, "format: header leading marker > 0");

        // Skip payload
        f.seekg(n1_lead, std::ios::cur);

        int32_t n1_trail = 0;
        f.read(reinterpret_cast<char*>(&n1_trail), 4);
        CHECK(n1_lead == n1_trail, "format: header leading==trailing marker");

        // --- Record 2: data ---
        int32_t n2_lead = 0;
        f.read(reinterpret_cast<char*>(&n2_lead), 4);
        CHECK(n2_lead > 0, "format: data leading marker > 0");

        // Data record: (1 + 1) * 8 = 16 bytes
        CHECK(n2_lead == 16, "format: data record size == 16 bytes");

        // Read time
        double t_val = 0.0;
        f.read(reinterpret_cast<char*>(&t_val), 8);
        CHECK_NEAR(t_val, 0.0, 1e-12, "format: time value == 0.0");

        // Read val
        double d_val = 0.0;
        f.read(reinterpret_cast<char*>(&d_val), 8);
        CHECK_NEAR(d_val, 3.14, 1e-9, "format: data value == 3.14");

        int32_t n2_trail = 0;
        f.read(reinterpret_cast<char*>(&n2_trail), 4);
        CHECK(n2_lead == n2_trail, "format: data leading==trailing marker");
    }
}

// ============================================================================
// Test 8: Empty group handling
// ============================================================================

static void test_empty_group() {
    std::cout << "--- Test 8: empty group ---\n";

    const std::string base = "/tmp/nxs_th44_test8";
    const std::string path = base + "_emptygrp.T01";

    RadiossTHWriter writer;
    writer.set_title("EmptyTest");
    writer.set_run_id(8);

    THGroup& g = writer.add_group("emptygrp", THEntityType::Shell);
    THVariableInfo vi("SXX", "MPa", 1, THEntityType::Shell);
    g.variables.push_back(vi);
    // No records added

    CHECK(g.num_records() == 0, "empty group: 0 records before write");

    bool ok = writer.write(base);
    CHECK(ok, "empty group: write ok");

    // File should exist and contain only the header record
    std::ifstream chk(path, std::ios::binary | std::ios::ate);
    CHECK(chk.is_open(), "empty group: file exists");
    auto sz = chk.tellg();
    CHECK(sz > 0, "empty group: file non-empty (has header)");
    chk.close();

    THReader reader;
    bool rok = reader.read(path);
    CHECK(rok, "empty group: read ok");

    if (rok && !reader.groups().empty()) {
        const THGroup& rg = reader.groups()[0];
        CHECK(rg.num_records()   == 0, "empty group: 0 records read back");
        CHECK(rg.num_variables() == 1, "empty group: 1 variable in header");
    }
}

// ============================================================================
// Test 9: RecorderToTH conversion
// ============================================================================

static void test_recorder_to_th() {
    std::cout << "--- Test 9: RecorderToTH conversion ---\n";

    // Build a TimeHistoryRecorder
    TimeHistoryRecorder recorder;

    // Add probes with names matching the synthetic convention in recorder_to_th
    recorder.add_nodal_probe("nodal_0", {0, 1}, NodalQuantity::DisplacementX);
    recorder.add_element_probe("element_0", {0}, ElementQuantity::VonMises);

    // Simulate 5 time steps
    const int N_NODES = 2;
    const int N_ELEMS = 1;
    std::vector<Real> disp(3 * N_NODES, 0.0);
    std::vector<Real> stress(6 * N_ELEMS, 0.0);
    std::vector<Real> strain(6 * N_ELEMS, 0.0);
    std::vector<Real> eps_p(static_cast<std::size_t>(N_ELEMS), 0.0);

    for (int step = 0; step < 5; ++step) {
        Real t = static_cast<Real>(step) * 0.01;
        disp[0] = static_cast<Real>(step) * 0.1;  // node 0 DX
        disp[3] = static_cast<Real>(step) * 0.2;  // node 1 DX
        stress[0] = static_cast<Real>(step) * 100.0;  // elem 0 SXX
        recorder.record(t,
                        static_cast<std::size_t>(N_NODES),
                        disp.data(), nullptr, nullptr, nullptr,
                        static_cast<std::size_t>(N_ELEMS),
                        stress.data(), strain.data(), eps_p.data());
    }

    CHECK(recorder.num_records() == 5, "recorder: 5 records");
    CHECK(recorder.num_nodal_probes()   == 1, "recorder: 1 nodal probe");
    CHECK(recorder.num_element_probes() == 1, "recorder: 1 element probe");

    // Convert
    RadiossTHWriter writer;
    writer.set_title("RecorderConversion");
    writer.set_run_id(9);
    recorder_to_th(recorder, writer);

    CHECK(writer.num_groups() == 2, "recorder_to_th: 2 groups created");

    // Write and read back
    const std::string base = "/tmp/nxs_th44_test9";
    bool ok = writer.write(base);
    CHECK(ok, "recorder_to_th: write ok");

    // Check the nodal group
    {
        THReader reader;
        bool rok = reader.read(base + "_nodal_0.T01");
        CHECK(rok, "recorder_to_th: nodal_0 read ok");
        if (rok && !reader.groups().empty()) {
            CHECK(reader.groups()[0].num_records() == 5,
                  "recorder_to_th: nodal group 5 records");
        }
    }

    // Check the element group
    {
        THReader reader;
        bool rok = reader.read(base + "_element_0.T01");
        CHECK(rok, "recorder_to_th: element_0 read ok");
        if (rok && !reader.groups().empty()) {
            CHECK(reader.groups()[0].num_records() == 5,
                  "recorder_to_th: element group 5 records");
        }
    }
}

// ============================================================================
// Test 10: Variable info preservation
// ============================================================================

static void test_variable_info_preservation() {
    std::cout << "--- Test 10: variable info preservation ---\n";

    const std::string base = "/tmp/nxs_th44_test10";
    const std::string path = base + "_varinfo.T01";

    RadiossTHWriter writer;
    writer.set_title("VarInfoTest");
    writer.set_run_id(10);

    THGroup& g = writer.add_group("varinfo", THEntityType::Shell);
    // Add 3 variables with distinct names, units, entity IDs, types
    g.variables.emplace_back("SXX",  "MPa",    101, THEntityType::Shell);
    g.variables.emplace_back("VX",   "mm/ms",  202, THEntityType::Node);
    g.variables.emplace_back("PLAS", "no_dim", 303, THEntityType::Solid);

    g.add_record(0.0, {1.1, 2.2, 3.3});

    bool ok = writer.write(base);
    CHECK(ok, "varinfo: write ok");

    THReader reader;
    bool rok = reader.read(path);
    CHECK(rok, "varinfo: read ok");

    if (rok && !reader.groups().empty()) {
        const THGroup& rg = reader.groups()[0];
        CHECK(rg.num_variables() == 3, "varinfo: 3 variables read");

        if (rg.num_variables() == 3) {
            CHECK(rg.variables[0].name == "SXX",    "varinfo: name[0]");
            CHECK(rg.variables[1].name == "VX",     "varinfo: name[1]");
            CHECK(rg.variables[2].name == "PLAS",   "varinfo: name[2]");

            CHECK(rg.variables[0].unit == "MPa",    "varinfo: unit[0]");
            CHECK(rg.variables[1].unit == "mm/ms",  "varinfo: unit[1]");
            CHECK(rg.variables[2].unit == "no_dim", "varinfo: unit[2]");

            CHECK(rg.variables[0].entity_id == 101, "varinfo: entity_id[0]");
            CHECK(rg.variables[1].entity_id == 202, "varinfo: entity_id[1]");
            CHECK(rg.variables[2].entity_id == 303, "varinfo: entity_id[2]");

            CHECK(rg.variables[0].entity_type == THEntityType::Shell,
                  "varinfo: entity_type[0] Shell");
            CHECK(rg.variables[1].entity_type == THEntityType::Node,
                  "varinfo: entity_type[1] Node");
            CHECK(rg.variables[2].entity_type == THEntityType::Solid,
                  "varinfo: entity_type[2] Solid");
        }

        if (rg.num_records() == 1 && rg.records[0].values.size() == 3) {
            CHECK_NEAR(rg.records[0].values[0], 1.1, 1e-9, "varinfo: val[0]");
            CHECK_NEAR(rg.records[0].values[1], 2.2, 1e-9, "varinfo: val[1]");
            CHECK_NEAR(rg.records[0].values[2], 3.3, 1e-9, "varinfo: val[2]");
        }
    }

    // Also verify title and run_id are preserved
    if (rok) {
        const THFileHeader& hdr = reader.header();
        CHECK(hdr.run_id == 10, "varinfo: run_id preserved");
        // Title is trimmed of trailing spaces; check prefix
        CHECK(hdr.title.substr(0, 11) == "VarInfoTest", "varinfo: title preserved");
    }
}

// ============================================================================
// Test 11: Time ordering
// ============================================================================

static void test_time_ordering() {
    std::cout << "--- Test 11: time ordering ---\n";

    const std::string base = "/tmp/nxs_th44_test11";
    const std::string path = base + "_timeord.T01";

    RadiossTHWriter writer;
    writer.set_title("TimeOrd");
    writer.set_run_id(11);

    THGroup& g = writer.add_group("timeord", THEntityType::Node);
    THVariableInfo vi("DX", "mm", 1, THEntityType::Node);
    g.variables.push_back(vi);

    // Add records in ascending time order
    const int N = 20;
    for (int i = 0; i < N; ++i) {
        Real t = static_cast<Real>(i) * 0.001;
        g.add_record(t, {static_cast<Real>(i) * 0.5});
    }

    bool ok = writer.write(base);
    CHECK(ok, "time_ord: write ok");

    THReader reader;
    bool rok = reader.read(path);
    CHECK(rok, "time_ord: read ok");

    if (rok && !reader.groups().empty()) {
        const THGroup& rg = reader.groups()[0];
        CHECK(rg.num_records() == static_cast<std::size_t>(N),
              "time_ord: all records present");

        // Verify monotone increasing
        bool ordered = true;
        for (std::size_t i = 1; i < rg.num_records(); ++i) {
            if (rg.records[i].time < rg.records[i-1].time) {
                ordered = false;
                break;
            }
        }
        CHECK(ordered, "time_ord: records are time-ordered");

        // Spot-check a few
        if (rg.num_records() == static_cast<std::size_t>(N)) {
            CHECK_NEAR(rg.records[0].values[0],  0.0,  1e-9, "time_ord: val[0]");
            CHECK_NEAR(rg.records[5].values[0],  2.5,  1e-9, "time_ord: val[5]");
            CHECK_NEAR(rg.records[19].values[0], 9.5,  1e-9, "time_ord: val[19]");
        }
    }
}

// ============================================================================
// Test 12: File creation — .T01 extension, file exists, non-empty
// ============================================================================

static void test_file_creation() {
    std::cout << "--- Test 12: file creation ---\n";

    const std::string base = "/tmp/nxs_th44_test12";

    RadiossTHWriter writer;
    writer.set_title("FileCreation");
    writer.set_run_id(12);

    THGroup& g = writer.add_group("fcgrp", THEntityType::Sensor);
    THVariableInfo vi("SIG", "V", 1, THEntityType::Sensor);
    g.variables.push_back(vi);
    g.add_record(0.0,  {1.0});
    g.add_record(0.01, {2.0});

    bool ok = writer.write(base);
    CHECK(ok, "file_creation: write returns true");

    const std::string expected_path = base + "_fcgrp.T01";

    // Check extension
    bool has_t01 = expected_path.size() >= 4 &&
                   expected_path.substr(expected_path.size() - 4) == ".T01";
    CHECK(has_t01, "file_creation: .T01 extension");

    // File exists
    std::ifstream f(expected_path, std::ios::binary | std::ios::ate);
    CHECK(f.is_open(), "file_creation: file exists");

    // File non-empty
    if (f.is_open()) {
        auto sz = f.tellg();
        CHECK(sz > 0, "file_creation: file non-empty");

        // Sanity: file size should be at least:
        //   header fortran wrapper (4+4) + header payload + data record wrapper + data
        // Header payload >= TITLE_LEN(80) + 4 + 4 + 1*(8+8+4+4) = 112
        // Data record    = 4 + 16 + 4 = 24
        // Total min ~= 8 + 112 + 8 + 24 + 8 + 24 = ~184 bytes  (two records)
        CHECK(sz >= 100, "file_creation: file large enough");

        f.close();
    }

    // Verify path stored in writer matches
    CHECK(writer.num_groups() == 1, "file_creation: writer has 1 group");
    CHECK(writer.groups()[0].name == "fcgrp", "file_creation: group name correct");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Wave 44 TH Writer Tests ===\n\n";

    test_node_roundtrip();
    test_shell_roundtrip();
    test_multi_group();
    test_entity_types();
    test_large_model();
    test_append_mode();
    test_fortran_markers();
    test_empty_group();
    test_recorder_to_th();
    test_variable_info_preservation();
    test_time_ordering();
    test_file_creation();

    std::cout << "\n=== Wave 44 TH Writer Results ===\n";
    std::cout << "  Passed: " << tests_passed << "\n";
    std::cout << "  Failed: " << tests_failed << "\n";

    if (tests_failed == 0) {
        std::cout << "  ALL TESTS PASSED\n";
    } else {
        std::cout << "  SOME TESTS FAILED\n";
    }

    return tests_failed > 0 ? 1 : 0;
}
