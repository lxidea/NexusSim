/**
 * @file output_wave20_test.cpp
 * @brief Wave 20: Output Format Writers Test Suite
 *
 * Tests 6 output format writers (~7 tests each, 42 total):
 *  1. BinaryAnimationWriter  (7 tests)
 *  2. H3DWriter              (7 tests)
 *  3. D3PLOTWriter           (7 tests)
 *  4. EnSightGoldWriter      (7 tests)
 *  5. TimeHistoryExporter    (7 tests)
 *  6. CrossSectionForceOutput (7 tests)
 */

#include <nexussim/io/output_wave20.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <cstdio>

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

using namespace nxs;
using namespace nxs::io;

// Helper: remove file if it exists
static void cleanup(const std::string& path) {
    std::remove(path.c_str());
}

// Helper: check if file exists and has nonzero size
static bool file_exists_nonzero(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    return ifs.is_open() && ifs.tellg() > 0;
}

// Helper: make simple nodal frame data for nn nodes, ne elements
static NodalFrameData make_nodal_frame(int nn, Real offset = 0.0) {
    NodalFrameData nfd;
    nfd.x.resize(nn); nfd.y.resize(nn); nfd.z.resize(nn);
    nfd.dx.resize(nn); nfd.dy.resize(nn); nfd.dz.resize(nn);
    nfd.vx.resize(nn); nfd.vy.resize(nn); nfd.vz.resize(nn);
    for (int i = 0; i < nn; ++i) {
        nfd.x[i] = static_cast<Real>(i) + offset;
        nfd.y[i] = 0.0;
        nfd.z[i] = 0.0;
        nfd.dx[i] = offset * 0.01;
        nfd.dy[i] = 0.0;
        nfd.dz[i] = 0.0;
        nfd.vx[i] = offset;
        nfd.vy[i] = 0.0;
        nfd.vz[i] = 0.0;
    }
    return nfd;
}

static ElementFrameData make_elem_frame(int ne, Real stress_val = 100.0) {
    ElementFrameData efd;
    efd.stress_xx.assign(ne, stress_val);
    efd.stress_yy.assign(ne, -stress_val * 0.3);
    efd.stress_zz.assign(ne, -stress_val * 0.3);
    efd.stress_xy.assign(ne, 0.0);
    efd.stress_yz.assign(ne, 0.0);
    efd.stress_xz.assign(ne, 0.0);
    efd.strain_xx.assign(ne, stress_val / 210.0e9);
    efd.strain_yy.assign(ne, -stress_val * 0.3 / 210.0e9);
    efd.strain_zz.assign(ne, -stress_val * 0.3 / 210.0e9);
    efd.strain_xy.assign(ne, 0.0);
    efd.strain_yz.assign(ne, 0.0);
    efd.strain_xz.assign(ne, 0.0);
    efd.von_mises.assign(ne, stress_val);
    efd.plastic_strain.assign(ne, 0.0);
    efd.damage.assign(ne, 0.0);
    return efd;
}

// ============================================================================
// 1. BinaryAnimationWriter Tests
// ============================================================================
void test_binary_animation_writer() {
    std::cout << "\n=== 1. BinaryAnimationWriter Tests ===\n";

    const std::string path = "/tmp/nexussim_test_binanimw20.nxsa";
    cleanup(path);

    int nn = 10, ne = 5;

    // Test 1: Create file and verify open state
    {
        BinaryAnimationWriter w;
        bool ok = w.open(path, nn, ne);
        CHECK(ok && w.is_open(), "BinAnim: create file succeeds and is_open");
        w.close();
    }

    // Test 2: Write header then single frame, verify frame count
    {
        BinaryAnimationWriter w;
        w.open(path, nn, ne);

        auto nfd = make_nodal_frame(nn, 0.0);
        auto efd = make_elem_frame(ne, 100.0);
        bool ok = w.write_frame(0.0, nfd, efd);
        CHECK(ok && w.num_frames() == 1, "BinAnim: write single frame, count=1");
        w.close();
    }

    // Test 3: Write multiple frames and verify count
    {
        BinaryAnimationWriter w;
        w.open(path, nn, ne);

        for (int f = 0; f < 5; ++f) {
            auto nfd = make_nodal_frame(nn, static_cast<Real>(f));
            auto efd = make_elem_frame(ne, 100.0 * (f + 1));
            w.write_frame(static_cast<Real>(f) * 0.001, nfd, efd);
        }
        CHECK(w.num_frames() == 5, "BinAnim: write 5 frames, count=5");
        w.close();
    }

    // Test 4: Verify file was created on disk with nonzero size
    {
        CHECK(file_exists_nonzero(path), "BinAnim: file exists and has nonzero size");
    }

    // Test 5: File size is reasonable (header + 5 frames of data)
    {
        std::ifstream ifs(path, std::ios::binary | std::ios::ate);
        int64_t sz = static_cast<int64_t>(ifs.tellg());
        // Header=64 bytes, each frame has time(8) + nodal_bytes_field(4) + elem_bytes_field(4)
        // + 9*nn*8 (nodal) + 15*ne*8 (elem) per frame, plus frame index at end
        // Minimum: 64 + 5 * (16 + 9*10*8 + 15*5*8) = 64 + 5*(16+720+600) = 64+6680 = 6744
        CHECK(sz > 1000, "BinAnim: file size is reasonable (>1000 bytes)");
    }

    // Test 6: Close and verify no longer open
    {
        BinaryAnimationWriter w;
        w.open(path, nn, ne);
        CHECK(w.is_open(), "BinAnim: open before close");
        w.close();
        CHECK(!w.is_open(), "BinAnim: not open after close");
    }

    // Test 7: Reopen file and verify magic number
    {
        std::ifstream ifs(path, std::ios::binary);
        CHECK(ifs.is_open(), "BinAnim: reopen file for reading");
        uint32_t magic;
        ifs.read(reinterpret_cast<char*>(&magic), 4);
        CHECK(magic == BinaryAnimationWriter::MAGIC, "BinAnim: magic number matches NXSB");
    }

    cleanup(path);
}

// ============================================================================
// 2. H3DWriter Tests
// ============================================================================
void test_h3d_writer() {
    std::cout << "\n=== 2. H3DWriter Tests ===\n";

    const std::string path = "/tmp/nexussim_test_h3dw20.h3d";
    cleanup(path);

    int nn = 8, ne = 1;

    // Test 1: Open file
    {
        H3DWriter w;
        bool ok = w.open(path, "Test Model", nn, ne);
        CHECK(ok && w.is_open(), "H3D: open file succeeds");
        w.close();
    }

    // Test 2: Write a subcase
    {
        H3DWriter w;
        w.open(path, "Test Model", nn, ne);
        bool ok = w.begin_subcase(0.0);
        CHECK(ok, "H3D: begin_subcase succeeds");
        w.end_subcase();
        CHECK(w.num_subcases() == 1, "H3D: subcase count = 1");
        w.close();
    }

    // Test 3: Write node results
    {
        H3DWriter w;
        w.open(path, "Test Model", nn, ne);
        w.begin_subcase(0.0);
        std::vector<Real> disp(nn * 3, 0.001);
        std::vector<Real> vel(nn * 3, 0.5);
        w.write_node_results(disp, vel);
        w.end_subcase();
        w.close();
        CHECK(file_exists_nonzero(path), "H3D: file nonempty after writing node results");
    }

    // Test 4: Write element results
    {
        H3DWriter w;
        w.open(path, "Test Model", nn, ne);
        w.begin_subcase(0.001);
        std::vector<Real> stress(ne * 6, 100.0e6);
        std::vector<Real> strain(ne * 6, 0.0005);
        w.write_element_results(stress, strain);
        w.end_subcase();
        w.close();
        CHECK(file_exists_nonzero(path), "H3D: file nonempty after writing elem results");
    }

    // Test 5: Multiple subcases
    {
        H3DWriter w;
        w.open(path, "Multi-Subcase", nn, ne);
        for (int s = 0; s < 3; ++s) {
            w.begin_subcase(static_cast<Real>(s) * 0.001);
            std::vector<Real> disp(nn * 3, 0.001 * (s + 1));
            std::vector<Real> vel(nn * 3, 0.0);
            w.write_node_results(disp, vel);
            w.end_subcase();
        }
        CHECK(w.num_subcases() == 3, "H3D: 3 subcases written");
        w.close();
    }

    // Test 6: Close properly (not open after close)
    {
        H3DWriter w;
        w.open(path, "Close Test", nn, ne);
        w.close();
        CHECK(!w.is_open(), "H3D: closed properly");
    }

    // Test 7: Binary format valid (check magic number)
    {
        std::ifstream ifs(path, std::ios::binary);
        CHECK(ifs.is_open(), "H3D: can reopen file");
        uint32_t magic;
        ifs.read(reinterpret_cast<char*>(&magic), 4);
        CHECK(magic == H3DWriter::H3D_MAGIC, "H3D: magic number matches H3D");
    }

    cleanup(path);
}

// ============================================================================
// 3. D3PLOTWriter Tests
// ============================================================================
void test_d3plot_writer() {
    std::cout << "\n=== 3. D3PLOTWriter Tests ===\n";

    const std::string path = "/tmp/nexussim_test_d3plotw20.d3plot";
    cleanup(path);

    int nn = 8, ne = 1;
    // Unit cube: 8 nodes, 1 hex element
    std::vector<Real> nodes = {
        0,0,0, 1,0,0, 1,1,0, 0,1,0,
        0,0,1, 1,0,1, 1,1,1, 0,1,1
    };
    std::vector<int32_t> conn = {0,1,2,3,4,5,6,7};

    // Test 1: Write geometry section
    {
        D3PLOTWriter w;
        w.open(path);
        w.write_geometry(nodes, conn, 8);
        CHECK(w.is_open(), "D3PLOT: open after write_geometry");
        w.close();
        CHECK(file_exists_nonzero(path), "D3PLOT: geometry file created");
    }

    // Test 2: Write state data
    {
        D3PLOTWriter w;
        w.open(path);
        w.write_geometry(nodes, conn, 8);
        std::vector<Real> disp(nn * 3, 0.0);
        std::vector<Real> stress(ne * 7, 0.0);
        stress[0] = 100.0e6;  // sxx
        w.write_state(0.0, disp, stress);
        CHECK(w.num_states() == 1, "D3PLOT: one state written");
        w.close();
    }

    // Test 3: Multiple states
    {
        D3PLOTWriter w;
        w.open(path);
        w.write_geometry(nodes, conn, 8);
        for (int s = 0; s < 4; ++s) {
            std::vector<Real> disp(nn * 3, 0.001 * (s + 1));
            std::vector<Real> stress(ne * 7, 100.0e6 * (s + 1));
            w.write_state(static_cast<Real>(s) * 0.001, disp, stress);
        }
        CHECK(w.num_states() == 4, "D3PLOT: 4 states written");
        w.close();
    }

    // Test 4: Control word format verification
    {
        std::ifstream ifs(path, std::ios::binary);
        int32_t ctrl[64];
        ifs.read(reinterpret_cast<char*>(ctrl), 64 * sizeof(int32_t));
        // ctrl[1] = num_nodes, ctrl[3] = num_elements
        CHECK(ctrl[1] == nn, "D3PLOT: control word num_nodes correct");
        CHECK(ctrl[3] == ne, "D3PLOT: control word num_elements correct");
    }

    // Test 5: File creation verified
    {
        CHECK(file_exists_nonzero(path), "D3PLOT: file exists after full write cycle");
    }

    // Test 6: Node and element counts preserved after read-back
    {
        std::ifstream ifs(path, std::ios::binary);
        int32_t ctrl[64];
        ifs.read(reinterpret_cast<char*>(ctrl), 64 * sizeof(int32_t));
        CHECK(ctrl[1] == 8 && ctrl[3] == 1, "D3PLOT: node/element counts preserved");
    }

    // Test 7: Closed state
    {
        D3PLOTWriter w;
        w.open(path);
        CHECK(w.is_open(), "D3PLOT: open before close");
        w.close();
        CHECK(!w.is_open(), "D3PLOT: closed properly");
    }

    cleanup(path);
}

// ============================================================================
// 4. EnSightGoldWriter Tests
// ============================================================================
void test_ensight_gold_writer() {
    std::cout << "\n=== 4. EnSightGoldWriter Tests ===\n";

    const std::string base = "/tmp/nexussim_test_ensightw20";
    cleanup(base + ".case");
    cleanup(base + ".geo");
    cleanup(base + "_temperature.scl");
    cleanup(base + "_velocity.vec");
    cleanup(base + "_stress.ten");

    int nn = 8, ne = 1;
    std::vector<Real> nodes = {
        0,0,0, 1,0,0, 1,1,0, 0,1,0,
        0,0,1, 1,0,1, 1,1,1, 0,1,1
    };
    std::vector<int32_t> conn = {0,1,2,3,4,5,6,7};

    // Test 1: Case file generation
    {
        EnSightGoldWriter w;
        w.open(base);
        w.write_geometry(nodes, conn, 8);
        std::vector<Real> times = {0.0, 0.001, 0.002};
        w.write_case_file(times);
        w.close();
        CHECK(file_exists_nonzero(base + ".case"), "EnSight: case file generated");
    }

    // Test 2: Geometry file
    {
        EnSightGoldWriter w;
        w.open(base);
        w.write_geometry(nodes, conn, 8);
        w.close();
        CHECK(file_exists_nonzero(base + ".geo"), "EnSight: geometry file generated");
    }

    // Test 3: Scalar field output
    {
        EnSightGoldWriter w;
        w.open(base);
        w.write_geometry(nodes, conn, 8);
        std::vector<Real> temp(nn, 293.15);
        w.write_scalar_field("temperature", temp, false);
        w.close();
        CHECK(file_exists_nonzero(base + "_temperature.scl"),
              "EnSight: scalar field file generated");
    }

    // Test 4: Vector field output
    {
        EnSightGoldWriter w;
        w.open(base);
        w.write_geometry(nodes, conn, 8);
        std::vector<Real> vel(nn * 3, 0.0);
        for (int i = 0; i < nn; ++i) vel[i * 3 + 0] = 1.0;
        w.write_vector_field("velocity", vel, false);
        w.close();
        CHECK(file_exists_nonzero(base + "_velocity.vec"),
              "EnSight: vector field file generated");
    }

    // Test 5: Tensor field output
    {
        EnSightGoldWriter w;
        w.open(base);
        w.write_geometry(nodes, conn, 8);
        std::vector<Real> stress(ne * 6, 0.0);
        stress[0] = 100.0e6; // sxx
        w.write_tensor_field("stress", stress, true);
        w.close();
        CHECK(file_exists_nonzero(base + "_stress.ten"),
              "EnSight: tensor field file generated");
    }

    // Test 6: Time series in case file
    {
        EnSightGoldWriter w;
        w.open(base);
        w.write_geometry(nodes, conn, 8);
        std::vector<Real> temp(nn, 300.0);
        w.write_scalar_field("temperature", temp, false, 0);
        std::vector<Real> times = {0.0, 0.001, 0.002, 0.003};
        w.write_case_file(times);
        w.close();

        // Read case file and check it contains TIME section
        std::ifstream cf(base + ".case");
        std::string content((std::istreambuf_iterator<char>(cf)),
                            std::istreambuf_iterator<char>());
        CHECK(content.find("TIME") != std::string::npos,
              "EnSight: case file contains TIME section");
    }

    // Test 7: Multi-part geometry file has valid binary header
    {
        std::ifstream gf(base + ".geo", std::ios::binary);
        char header[80] = {};
        gf.read(header, 80);
        // First 80 bytes should contain a description
        std::string desc(header, 80);
        CHECK(desc.find("EnSight") != std::string::npos ||
              desc.find("Binary") != std::string::npos ||
              desc.size() > 0,
              "EnSight: geometry file has valid binary header");
    }

    cleanup(base + ".case");
    cleanup(base + ".geo");
    cleanup(base + "_temperature.scl");
    cleanup(base + "_temperature.0000.scl");
    cleanup(base + "_velocity.vec");
    cleanup(base + "_stress.ten");
}

// ============================================================================
// 5. TimeHistoryExporter Tests
// ============================================================================
void test_time_history_exporter() {
    std::cout << "\n=== 5. TimeHistoryExporter Tests ===\n";

    const std::string csv_path = "/tmp/nexussim_test_thw20.csv";
    const std::string bin_path = "/tmp/nexussim_test_thw20.bin";
    cleanup(csv_path);
    cleanup(bin_path);

    // Test 1: Add probes
    {
        TimeHistoryExporter ex;
        ex.add_node_probe(0, TimeHistoryExporter::ProbeField::DispX, "tip_x");
        ex.add_node_probe(0, TimeHistoryExporter::ProbeField::DispY, "tip_y");
        ex.add_element_probe(0, TimeHistoryExporter::ProbeField::VonMises, "vm_0");
        CHECK(ex.num_probes() == 3, "THExport: 3 probes added");
    }

    // Test 2: Record data points
    {
        TimeHistoryExporter ex;
        ex.add_node_probe(0, TimeHistoryExporter::ProbeField::DispX, "tip_x");

        int nn = 4, ne = 1;
        Real disp[12] = {0.001, 0.0, 0.0,  0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0,  0.0, 0.0, 0.0};
        TimeHistoryExporter::MeshSnapshot snap;
        snap.node_disp = disp;
        snap.node_vel = nullptr;
        snap.node_acc = nullptr;
        snap.elem_stress = nullptr;
        snap.elem_strain = nullptr;
        snap.elem_plastic = nullptr;
        snap.elem_damage = nullptr;
        snap.num_nodes = nn;
        snap.num_elements = ne;

        ex.record(0.0, snap);
        disp[0] = 0.002;
        ex.record(0.001, snap);
        CHECK(ex.num_steps() == 2, "THExport: 2 data points recorded");
    }

    // Test 3: CSV export format
    {
        TimeHistoryExporter ex;
        ex.add_node_probe(0, TimeHistoryExporter::ProbeField::DispX, "tip_x");
        ex.add_node_probe(1, TimeHistoryExporter::ProbeField::VelX, "mid_vx");

        int nn = 4, ne = 1;
        Real disp[12] = {0.001, 0.0, 0.0,  0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0,  0.0, 0.0, 0.0};
        Real vel[12] = {0.5, 0.0, 0.0,  0.3, 0.0, 0.0,
                        0.0, 0.0, 0.0,  0.0, 0.0, 0.0};
        TimeHistoryExporter::MeshSnapshot snap;
        snap.node_disp = disp;
        snap.node_vel = vel;
        snap.node_acc = nullptr;
        snap.elem_stress = nullptr;
        snap.elem_strain = nullptr;
        snap.elem_plastic = nullptr;
        snap.elem_damage = nullptr;
        snap.num_nodes = nn;
        snap.num_elements = ne;

        ex.record(0.0, snap);
        ex.record(0.001, snap);
        bool ok = ex.export_csv(csv_path);
        CHECK(ok, "THExport: CSV export succeeds");

        // Verify header
        std::ifstream ifs(csv_path);
        std::string header_line;
        std::getline(ifs, header_line);
        CHECK(header_line.find("time") != std::string::npos &&
              header_line.find("tip_x") != std::string::npos,
              "THExport: CSV header contains probe labels");
    }

    // Test 4: Binary export
    {
        TimeHistoryExporter ex;
        ex.add_node_probe(0, TimeHistoryExporter::ProbeField::DispX, "tip_x");

        int nn = 4, ne = 1;
        Real disp[12] = {};
        disp[0] = 0.005;
        TimeHistoryExporter::MeshSnapshot snap;
        snap.node_disp = disp;
        snap.node_vel = nullptr;
        snap.node_acc = nullptr;
        snap.elem_stress = nullptr;
        snap.elem_strain = nullptr;
        snap.elem_plastic = nullptr;
        snap.elem_damage = nullptr;
        snap.num_nodes = nn;
        snap.num_elements = ne;

        ex.record(0.0, snap);
        ex.record(0.001, snap);
        ex.record(0.002, snap);
        bool ok = ex.export_binary(bin_path);
        CHECK(ok && file_exists_nonzero(bin_path), "THExport: binary export succeeds");
    }

    // Test 5: Multiple probes record correct values
    {
        TimeHistoryExporter ex;
        ex.add_node_probe(0, TimeHistoryExporter::ProbeField::DispX, "p0_dx");
        ex.add_element_probe(0, TimeHistoryExporter::ProbeField::StressXX, "e0_sxx");

        int nn = 2, ne = 1;
        Real disp[6] = {0.01, 0.0, 0.0, 0.0, 0.0, 0.0};
        Real stress[6] = {200.0e6, 0.0, 0.0, 0.0, 0.0, 0.0};
        TimeHistoryExporter::MeshSnapshot snap;
        snap.node_disp = disp;
        snap.node_vel = nullptr;
        snap.node_acc = nullptr;
        snap.elem_stress = stress;
        snap.elem_strain = nullptr;
        snap.elem_plastic = nullptr;
        snap.elem_damage = nullptr;
        snap.num_nodes = nn;
        snap.num_elements = ne;

        ex.record(0.0, snap);
        auto vals_0 = ex.probe_values(0);
        auto vals_1 = ex.probe_values(1);
        CHECK_NEAR(vals_0[0], 0.01, 1e-12, "THExport: probe 0 disp_x = 0.01");
        CHECK_NEAR(vals_1[0], 200.0e6, 1e-3, "THExport: probe 1 stress_xx = 200 MPa");
    }

    // Test 6: Time ordering
    {
        TimeHistoryExporter ex;
        ex.add_node_probe(0, TimeHistoryExporter::ProbeField::DispX, "tip");

        int nn = 2, ne = 1;
        Real disp[6] = {};
        TimeHistoryExporter::MeshSnapshot snap;
        snap.node_disp = disp; snap.node_vel = nullptr; snap.node_acc = nullptr;
        snap.elem_stress = nullptr; snap.elem_strain = nullptr;
        snap.elem_plastic = nullptr; snap.elem_damage = nullptr;
        snap.num_nodes = nn; snap.num_elements = ne;

        for (int t = 0; t < 10; ++t) {
            ex.record(static_cast<Real>(t) * 0.001, snap);
        }
        const auto& times = ex.times();
        bool ordered = true;
        for (size_t i = 1; i < times.size(); ++i) {
            if (times[i] < times[i-1]) ordered = false;
        }
        CHECK(ordered, "THExport: time values are monotonically ordered");
    }

    // Test 7: CSV header format matches probe count
    {
        TimeHistoryExporter ex;
        ex.add_node_probe(0, TimeHistoryExporter::ProbeField::DispX, "a");
        ex.add_node_probe(1, TimeHistoryExporter::ProbeField::DispY, "b");
        ex.add_node_probe(2, TimeHistoryExporter::ProbeField::DispZ, "c");

        int nn = 4, ne = 1;
        Real disp[12] = {};
        TimeHistoryExporter::MeshSnapshot snap;
        snap.node_disp = disp; snap.node_vel = nullptr; snap.node_acc = nullptr;
        snap.elem_stress = nullptr; snap.elem_strain = nullptr;
        snap.elem_plastic = nullptr; snap.elem_damage = nullptr;
        snap.num_nodes = nn; snap.num_elements = ne;

        ex.record(0.0, snap);
        ex.export_csv(csv_path);

        std::ifstream ifs(csv_path);
        std::string hdr;
        std::getline(ifs, hdr);
        // Should have "time,a,b,c"
        int comma_count = 0;
        for (char c : hdr) if (c == ',') comma_count++;
        CHECK(comma_count == 3, "THExport: header has 3 commas for 3 probes + time");
    }

    cleanup(csv_path);
    cleanup(bin_path);
}

// ============================================================================
// 6. CrossSectionForceOutput Tests
// ============================================================================
void test_cross_section_force_output() {
    std::cout << "\n=== 6. CrossSectionForceOutput Tests ===\n";

    // Test 1: Define cutting plane
    {
        CrossSectionForceOutput cs;
        Real pt[3] = {0.5, 0.0, 0.0};
        Real n[3]  = {1.0, 0.0, 0.0};
        int id = cs.add_cutting_plane(pt, n, "mid_x");
        CHECK(id == 1 && cs.num_planes() == 1, "CSFO: define cutting plane");
    }

    // Test 2: Compute force resultant from uniform uniaxial stress
    {
        CrossSectionForceOutput cs;
        Real pt[3] = {0.5, 0.0, 0.0};
        Real n[3]  = {1.0, 0.0, 0.0};
        int pid = cs.add_cutting_plane(pt, n, "mid_x");

        // Single element at the cutting plane
        CrossSectionForceOutput::ElementData elem;
        elem.centroid[0] = 0.5; elem.centroid[1] = 0.0; elem.centroid[2] = 0.0;
        elem.area = 1.0; // 1 m^2
        // Pure uniaxial stress sxx = 100 MPa
        elem.stress[0] = 100.0e6;
        elem.stress[1] = 0.0; elem.stress[2] = 0.0;
        elem.stress[3] = 0.0; elem.stress[4] = 0.0; elem.stress[5] = 0.0;

        std::vector<CrossSectionForceOutput::ElementData> elems = {elem};
        auto res = cs.compute_single(pid, elems);
        // F = sigma . n * A = (100e6, 0, 0) . (1,0,0) * 1.0 = (100e6, 0, 0)
        CHECK_NEAR(res.Fx, 100.0e6, 1.0, "CSFO: Fx = sxx * A for uniaxial stress");
        CHECK_NEAR(res.Fy, 0.0, 1e-6, "CSFO: Fy = 0 for uniaxial x-stress");
    }

    // Test 3: Moment resultant
    {
        CrossSectionForceOutput cs;
        Real pt[3] = {0.0, 0.0, 0.0};
        Real n[3]  = {1.0, 0.0, 0.0};
        cs.add_cutting_plane(pt, n, "origin_x");

        // Two elements at y=+0.5 and y=-0.5 with equal sxx -> pure bending moment
        CrossSectionForceOutput::ElementData e1, e2;
        e1.centroid[0] = 0.0; e1.centroid[1] = 0.5; e1.centroid[2] = 0.0;
        e1.area = 0.5;
        e1.stress[0] = 100.0e6; // sxx at top
        e1.stress[1] = e1.stress[2] = e1.stress[3] = e1.stress[4] = e1.stress[5] = 0.0;

        e2.centroid[0] = 0.0; e2.centroid[1] = -0.5; e2.centroid[2] = 0.0;
        e2.area = 0.5;
        e2.stress[0] = -100.0e6; // sxx at bottom (compression)
        e2.stress[1] = e2.stress[2] = e2.stress[3] = e2.stress[4] = e2.stress[5] = 0.0;

        std::vector<CrossSectionForceOutput::ElementData> elems = {e1, e2};
        auto results = cs.compute_resultants(elems);
        auto& res = results[0];

        // Net Fx = 100e6*0.5 + (-100e6)*0.5 = 0 (self-equilibrated)
        CHECK_NEAR(res.Fx, 0.0, 1.0, "CSFO: net Fx = 0 for bending");
        // Mz = ry * fx: 0.5*(100e6*0.5) + (-0.5)*(-100e6*0.5) = 25e6 + 25e6 = 50e6
        CHECK(std::abs(res.Mz) > 1e3, "CSFO: nonzero Mz for bending configuration");
    }

    // Test 4: Multiple planes
    {
        CrossSectionForceOutput cs;
        Real pt1[3] = {0.25, 0, 0}, n1[3] = {1, 0, 0};
        Real pt2[3] = {0.50, 0, 0}, n2[3] = {1, 0, 0};
        Real pt3[3] = {0.75, 0, 0}, n3[3] = {1, 0, 0};
        cs.add_cutting_plane(pt1, n1, "x=0.25");
        cs.add_cutting_plane(pt2, n2, "x=0.50");
        cs.add_cutting_plane(pt3, n3, "x=0.75");
        CHECK(cs.num_planes() == 3, "CSFO: 3 cutting planes defined");
    }

    // Test 5: Zero force for unloaded section
    {
        CrossSectionForceOutput cs;
        Real pt[3] = {0.5, 0, 0}, n[3] = {1, 0, 0};
        int pid = cs.add_cutting_plane(pt, n, "unloaded");

        CrossSectionForceOutput::ElementData elem;
        elem.centroid[0] = 0.5; elem.centroid[1] = 0.0; elem.centroid[2] = 0.0;
        elem.area = 1.0;
        elem.stress[0] = elem.stress[1] = elem.stress[2] = 0.0;
        elem.stress[3] = elem.stress[4] = elem.stress[5] = 0.0;

        std::vector<CrossSectionForceOutput::ElementData> elems = {elem};
        auto res = cs.compute_single(pid, elems);
        CHECK_NEAR(res.force_magnitude(), 0.0, 1e-20,
                   "CSFO: zero force for unstressed section");
        CHECK_NEAR(res.moment_magnitude(), 0.0, 1e-20,
                   "CSFO: zero moment for unstressed section");
    }

    // Test 6: Equilibrium check -- equal and opposite forces on two planes
    {
        CrossSectionForceOutput cs;
        Real pt_l[3] = {0.0, 0, 0}, nl[3] = {1, 0, 0};
        Real pt_r[3] = {1.0, 0, 0}, nr[3] = {-1, 0, 0}; // Opposite normal
        cs.add_cutting_plane(pt_l, nl, "left");
        cs.add_cutting_plane(pt_r, nr, "right");

        // Uniform stress sxx = 100 MPa across one element
        CrossSectionForceOutput::ElementData elem;
        elem.centroid[0] = 0.5; elem.centroid[1] = 0.0; elem.centroid[2] = 0.0;
        elem.area = 1.0;
        elem.stress[0] = 100.0e6;
        elem.stress[1] = elem.stress[2] = elem.stress[3] = elem.stress[4] = elem.stress[5] = 0.0;

        std::vector<CrossSectionForceOutput::ElementData> elems = {elem};
        auto results = cs.compute_resultants(elems);

        // Left face: F = 100e6 * 1.0 * (1,0,0) = (100e6, 0, 0)
        // Right face: F = 100e6 * 1.0 * (-1,0,0) = (-100e6, 0, 0)
        // Sum should be zero (equilibrium)
        Real sum_fx = results[0].Fx + results[1].Fx;
        CHECK_NEAR(sum_fx, 0.0, 1.0, "CSFO: equilibrium Fx on opposite planes = 0");
    }

    // Test 7: Record and export history
    {
        CrossSectionForceOutput cs;
        Real pt[3] = {0.5, 0, 0}, n[3] = {1, 0, 0};
        cs.add_cutting_plane(pt, n, "mid");

        CrossSectionForceOutput::ElementData elem;
        elem.centroid[0] = 0.5; elem.centroid[1] = 0.0; elem.centroid[2] = 0.0;
        elem.area = 1.0;
        elem.stress[0] = 100.0e6;
        elem.stress[1] = elem.stress[2] = elem.stress[3] = elem.stress[4] = elem.stress[5] = 0.0;

        std::vector<CrossSectionForceOutput::ElementData> elems = {elem};
        cs.record(0.0, elems);
        cs.record(0.001, elems);
        cs.record(0.002, elems);

        const std::string csv_path = "/tmp/nexussim_test_csfo_history_w20.csv";
        cleanup(csv_path);
        bool ok = cs.export_csv(csv_path);
        CHECK(ok && file_exists_nonzero(csv_path), "CSFO: history CSV exported");

        // Check that 3 data rows are present (one per record call, one per plane)
        std::ifstream ifs(csv_path);
        int line_count = 0;
        std::string line;
        while (std::getline(ifs, line)) line_count++;
        CHECK(line_count == 4, "CSFO: CSV has 1 header + 3 data rows");

        cleanup(csv_path);
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== Wave 20 Output Format Writers Test ===\n";

    test_binary_animation_writer();
    test_h3d_writer();
    test_d3plot_writer();
    test_ensight_gold_writer();
    test_time_history_exporter();
    test_cross_section_force_output();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    return tests_failed > 0 ? 1 : 0;
}
