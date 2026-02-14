/**
 * @file lsdyna_reader_ext_test.cpp
 * @brief Tests for LS-DYNA reader extension (sections, materials, contacts, EOS, etc.)
 */

#include <nexussim/core/core.hpp>
#include <nexussim/io/lsdyna_reader.hpp>
#include <nexussim/physics/section.hpp>
#include <nexussim/physics/eos.hpp>
#include <iostream>
#include <fstream>
#include <cmath>

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

#define NEAR(a, b, tol) (std::abs((a) - (b)) < (tol))

// ============================================================================
// Helper: write string to temp file and read
// ============================================================================
static LSDynaReader read_string(const std::string& content) {
    std::string filename = "/tmp/lsdyna_ext_test.k";
    {
        std::ofstream file(filename);
        file << content;
    }
    LSDynaReader reader;
    reader.read(filename);
    return reader;
}

// ============================================================================
// Test 1: Section Parsing
// ============================================================================
void test_sections() {
    std::cout << "\n=== Test 1: Section Parsing ===\n";

    auto reader = read_string(R"(
*KEYWORD
*SECTION_SHELL
         1         2       0.8         5
       1.5       1.5       1.5       1.5
*SECTION_SOLID
         2         1
*SECTION_BEAM
         3         1       1.0         0         1
      0.05      0.10
*PART
Shell Part
         1         1         1
*MAT_ELASTIC
         1    7850.0    2.1E11       0.3
*PART
Beam Part
         2         3         1
*END
)");

    CHECK(reader.sections().size() == 3, "3 sections parsed");

    // Shell section
    const auto& shell = reader.sections().at(1);
    CHECK(shell.type == "SHELL", "Section 1 is SHELL");
    CHECK(shell.elform == 2, "Shell ELFORM = 2");
    CHECK(NEAR(shell.shrf, 0.8, 1e-6), "Shell SHRF = 0.8");
    CHECK(shell.nip == 5, "Shell NIP = 5");
    CHECK(NEAR(shell.t1, 1.5, 1e-6), "Shell T1 = 1.5");

    // Solid section
    const auto& solid = reader.sections().at(2);
    CHECK(solid.type == "SOLID", "Section 2 is SOLID");
    CHECK(solid.elform == 1, "Solid ELFORM = 1");

    // Beam section
    const auto& beam = reader.sections().at(3);
    CHECK(beam.type == "BEAM", "Section 3 is BEAM");
    CHECK(beam.cst == 1, "Beam CST = 1");
    CHECK(NEAR(beam.ts1, 0.05, 1e-6), "Beam TS1 = 0.05");
    CHECK(NEAR(beam.ts2, 0.10, 1e-6), "Beam TS2 = 0.10");

    // get_section() mapping
    auto sec_props = reader.get_section(1);
    CHECK(sec_props.type == physics::SectionType::ShellUniform, "get_section: shell -> ShellUniform");
    CHECK(NEAR(sec_props.thickness, 1.5, 1e-6), "get_section: thickness = 1.5");
    CHECK(sec_props.num_ip_thickness == 5, "get_section: NIP = 5");

    auto beam_props = reader.get_section(2);
    CHECK(beam_props.type == physics::SectionType::BeamRectangular, "get_section: beam CST=1 -> BeamRectangular");
}

// ============================================================================
// Test 2: Material Models
// ============================================================================
void test_materials() {
    std::cout << "\n=== Test 2: Material Models ===\n";

    auto reader = read_string(R"(
*KEYWORD
*MAT_RIGID
         1    7800.0    2.1E11       0.3
       1.0       4.0       7.0
*MAT_NULL
         2    1000.0       0.0     0.001
*MAT_PLASTIC_KINEMATIC
         3    2700.0    7.0E10       0.3  2.75E8   1.0E9       0.0     40.4       5.0    0.20
*MAT_MOONEY-RIVLIN_RUBBER
         4    1100.0      0.49   0.293E6   0.177E6
*MAT_OGDEN_RUBBER
         5    1050.0      0.49
   6.3E5  1.2E3  -1.0E4   1.3  5.0  -2.0
*MAT_VISCOELASTIC
         6    1200.0    2.0E9    4.0E6    3.0E6   100.0
*MAT_ORTHOTROPIC_ELASTIC
         7    1600.0    1.4E11  1.0E10  1.0E10     0.28     0.02     0.02
   5.0E9  3.0E9  5.0E9
*MAT_CRUSHABLE_FOAM
         8     100.0    5.0E6      0.05         0      1.0E5      0.1
*MAT_HONEYCOMB
         9     200.0    1.0E7      0.05  5.0E5     0.6     0.01    1.0E8
*MAT_COWPER_SYMONDS
        10    7850.0    2.1E11       0.3  2.5E8  1.0E9       0.0     40.4       5.0
*MAT_FOAM
        11      50.0    2.0E6         0     1000.0
*MAT_ELASTIC_PLASTIC_HYDRO
        12    8900.0    4.6E10    9.0E7    1.0E9    1.3E11
*PART
Part 1
         1         1         1
*PART
Part 2
         2         1         2
*PART
Part 3
         3         1         3
*PART
Part 4
         4         1         4
*PART
Part 5
         5         1         5
*PART
Part 6
         6         1         6
*PART
Part 7
         7         1         7
*PART
Part 8
         8         1         8
*PART
Part 9
         9         1         9
*PART
Part 10
        10         1        10
*PART
Part 11
        11         1        11
*PART
Part 12
        12         1        12
*SECTION_SHELL
         1         2       1.0         2
       1.0       1.0       1.0       1.0
*END
)");

    CHECK(reader.materials().size() == 12, "12 materials parsed");

    // MAT_RIGID
    const auto& rigid = reader.materials().at(1);
    CHECK(rigid.type == "RIGID", "MAT 1 type = RIGID");
    CHECK(NEAR(rigid.ro, 7800.0, 1e-3), "Rigid density = 7800");
    CHECK(NEAR(rigid.cmo, 1.0, 1e-6), "Rigid CMO = 1.0");
    CHECK(NEAR(rigid.con1, 4.0, 1e-6), "Rigid CON1 = 4.0");

    // MAT_NULL
    const auto& null_m = reader.materials().at(2);
    CHECK(null_m.type == "NULL", "MAT 2 type = NULL");
    CHECK(NEAR(null_m.mu, 0.001, 1e-6), "Null MU = 0.001");

    // MAT_PLASTIC_KINEMATIC
    const auto& pk = reader.materials().at(3);
    CHECK(pk.type == "PLASTIC_KINEMATIC", "MAT 3 type = PLASTIC_KINEMATIC");
    CHECK(NEAR(pk.sigy, 2.75E8, 1e3), "PK yield = 2.75E8");
    CHECK(NEAR(pk.cs_d, 40.4, 1e-3), "PK CS_D = 40.4");
    CHECK(NEAR(pk.cs_q, 5.0, 1e-6), "PK CS_q = 5.0");

    // MAT_MOONEY-RIVLIN_RUBBER
    const auto& mr = reader.materials().at(4);
    CHECK(mr.type == "MOONEY_RIVLIN", "MAT 4 type = MOONEY_RIVLIN");
    CHECK(NEAR(mr.c10_mr, 0.293E6, 1.0), "MR C10 = 0.293E6");
    CHECK(NEAR(mr.c01_mr, 0.177E6, 1.0), "MR C01 = 0.177E6");

    // MAT_OGDEN_RUBBER
    const auto& ogden = reader.materials().at(5);
    CHECK(ogden.type == "OGDEN", "MAT 5 type = OGDEN");
    CHECK(NEAR(ogden.mu1, 6.3E5, 1.0), "Ogden MU1 = 6.3E5");
    CHECK(NEAR(ogden.alpha1, 1.3, 1e-6), "Ogden ALPHA1 = 1.3");
    CHECK(NEAR(ogden.mu3, -1.0E4, 1.0), "Ogden MU3 = -1.0E4");
    CHECK(NEAR(ogden.alpha3, -2.0, 1e-6), "Ogden ALPHA3 = -2.0");

    // MAT_VISCOELASTIC
    const auto& ve = reader.materials().at(6);
    CHECK(ve.type == "VISCOELASTIC", "MAT 6 type = VISCOELASTIC");
    CHECK(NEAR(ve.bulk, 2.0E9, 1e3), "VE BULK = 2.0E9");
    CHECK(NEAR(ve.g0, 4.0E6, 1.0), "VE G0 = 4.0E6");
    CHECK(NEAR(ve.gi, 3.0E6, 1.0), "VE GI = 3.0E6");
    CHECK(NEAR(ve.beta_ve, 100.0, 1e-3), "VE BETA = 100.0");

    // MAT_ORTHOTROPIC_ELASTIC
    const auto& ortho = reader.materials().at(7);
    CHECK(ortho.type == "ORTHOTROPIC", "MAT 7 type = ORTHOTROPIC");
    CHECK(NEAR(ortho.ea, 1.4E11, 1e5), "Ortho EA = 1.4E11");
    CHECK(NEAR(ortho.gab, 5.0E9, 1e3), "Ortho GAB = 5.0E9");

    // MAT_CRUSHABLE_FOAM
    const auto& cf = reader.materials().at(8);
    CHECK(cf.type == "CRUSHABLE_FOAM", "MAT 8 type = CRUSHABLE_FOAM");
    CHECK(NEAR(cf.tension_cutoff, 1.0E5, 1.0), "CF tension cutoff = 1.0E5");
    CHECK(NEAR(cf.damp_foam, 0.1, 1e-6), "CF DAMP = 0.1");

    // MAT_HONEYCOMB
    const auto& hc = reader.materials().at(9);
    CHECK(hc.type == "HONEYCOMB", "MAT 9 type = HONEYCOMB");
    CHECK(NEAR(hc.sigy, 5.0E5, 1.0), "HC SIGY = 5.0E5");
    CHECK(NEAR(hc.vf, 0.6, 1e-6), "HC VF = 0.6");

    // MAT_COWPER_SYMONDS
    const auto& cs = reader.materials().at(10);
    CHECK(cs.type == "COWPER_SYMONDS", "MAT 10 type = COWPER_SYMONDS");
    CHECK(NEAR(cs.cs_d, 40.4, 1e-3), "CS D = 40.4");
    CHECK(NEAR(cs.cs_q, 5.0, 1e-6), "CS q = 5.0");

    // MAT_FOAM
    const auto& foam = reader.materials().at(11);
    CHECK(foam.type == "FOAM", "MAT 11 type = FOAM");
    CHECK(NEAR(foam.e, 2.0E6, 1.0), "Foam E = 2.0E6");

    // MAT_ELASTIC_PLASTIC_HYDRO
    const auto& eph = reader.materials().at(12);
    CHECK(eph.type == "ELASTIC_PLASTIC_HYDRO", "MAT 12 type = ELASTIC_PLASTIC_HYDRO");
    CHECK(NEAR(eph.g_shear, 4.6E10, 1e4), "EPH G = 4.6E10");
    CHECK(NEAR(eph.sigy, 9.0E7, 1e2), "EPH SIGY = 9.0E7");

    // get_material() mapping for Mooney-Rivlin
    auto mr_props = reader.get_material(4);
    CHECK(NEAR(mr_props.C10, 0.293E6, 1.0), "get_material MR: C10 correct");
    CHECK(NEAR(mr_props.C01, 0.177E6, 1.0), "get_material MR: C01 correct");

    // get_material() mapping for Ogden
    auto og_props = reader.get_material(5);
    CHECK(og_props.ogden_nterms == 3, "get_material Ogden: nterms = 3");
    CHECK(NEAR(og_props.ogden_mu[0], 6.3E5, 1.0), "get_material Ogden: mu[0] correct");

    // get_material() mapping for Orthotropic
    auto ort_props = reader.get_material(7);
    CHECK(NEAR(ort_props.E1, 1.4E11, 1e5), "get_material Ortho: E1 correct");
    CHECK(NEAR(ort_props.G12, 5.0E9, 1e3), "get_material Ortho: G12 correct");

    // get_material() mapping for Viscoelastic
    auto ve_props = reader.get_material(6);
    CHECK(NEAR(ve_props.K, 2.0E9, 1e3), "get_material VE: K (bulk) correct");
    CHECK(ve_props.prony_nterms == 1, "get_material VE: prony_nterms = 1");

    // get_material() mapping for Cowper-Symonds
    auto cs_props = reader.get_material(10);
    CHECK(NEAR(cs_props.CS_D, 40.4, 1e-3), "get_material CS: D correct");
    CHECK(NEAR(cs_props.CS_q, 5.0, 1e-6), "get_material CS: q correct");

    // get_material() mapping for Elastic-Plastic-Hydro
    auto eph_props = reader.get_material(12);
    CHECK(NEAR(eph_props.G, 4.6E10, 1e4), "get_material EPH: G correct");
    CHECK(NEAR(eph_props.yield_stress, 9.0E7, 1e2), "get_material EPH: yield correct");
}

// ============================================================================
// Test 3: Contact Cards
// ============================================================================
void test_contacts() {
    std::cout << "\n=== Test 3: Contact Cards ===\n";

    auto reader = read_string(R"(
*KEYWORD
*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE
         1         2         0         0
      0.30      0.20     0.001       0.0
       1.5       2.0
*CONTACT_AUTOMATIC_NODES_TO_SURFACE
         3         4         1         2
      0.10
*CONTACT_AUTOMATIC_SINGLE_SURFACE
         5
      0.25
*CONTACT_TIED_SURFACE_TO_SURFACE
         6         7
*END
)");

    CHECK(reader.contacts().size() == 4, "4 contacts parsed");

    // S2S
    const auto& s2s = reader.contacts()[0];
    CHECK(s2s.type == "SURFACE_TO_SURFACE", "Contact 1 type = S2S");
    CHECK(s2s.ssid == 1, "S2S SSID = 1");
    CHECK(s2s.msid == 2, "S2S MSID = 2");
    CHECK(NEAR(s2s.fs, 0.30, 1e-6), "S2S FS = 0.30");
    CHECK(NEAR(s2s.fd, 0.20, 1e-6), "S2S FD = 0.20");
    CHECK(NEAR(s2s.sfs, 1.5, 1e-6), "S2S SFS = 1.5");
    CHECK(NEAR(s2s.sfm, 2.0, 1e-6), "S2S SFM = 2.0");

    // N2S
    const auto& n2s = reader.contacts()[1];
    CHECK(n2s.type == "NODES_TO_SURFACE", "Contact 2 type = N2S");
    CHECK(n2s.sstyp == 1, "N2S SSTYP = 1");
    CHECK(n2s.mstyp == 2, "N2S MSTYP = 2");

    // Single surface
    const auto& ss = reader.contacts()[2];
    CHECK(ss.type == "SINGLE_SURFACE", "Contact 3 type = SINGLE_SURFACE");
    CHECK(ss.ssid == 5, "SS SSID = 5");

    // Tied
    const auto& tied = reader.contacts()[3];
    CHECK(tied.type == "TIED", "Contact 4 type = TIED");
    CHECK(tied.ssid == 6, "Tied SSID = 6");
}

// ============================================================================
// Test 4: EOS Cards
// ============================================================================
void test_eos() {
    std::cout << "\n=== Test 4: EOS Cards ===\n";

    auto reader = read_string(R"(
*KEYWORD
*EOS_GRUNEISEN
         1      5240     1.338     0.000     0.000      1.97     0.480       0.0
       1.0
*EOS_JWL
         2   3.712E11   3.231E9       4.15      0.95      0.30   7.0E9       1.0
*EOS_LINEAR_POLYNOMIAL
         3       0.0    2.0E9       0.0       0.0     0.4       0.0       0.0
    1.0E6       1.0
*END
)");

    CHECK(reader.eos_map().size() == 3, "3 EOS parsed");

    // Gruneisen
    const auto& gru = reader.eos_map().at(1);
    CHECK(gru.type == "GRUNEISEN", "EOS 1 type = GRUNEISEN");
    CHECK(NEAR(gru.C0, 5240.0, 1e-3), "Gruneisen C = 5240");
    CHECK(NEAR(gru.S1, 1.338, 1e-6), "Gruneisen S1 = 1.338");
    CHECK(NEAR(gru.gamma0, 1.97, 1e-6), "Gruneisen gamma0 = 1.97");
    CHECK(NEAR(gru.V0, 1.0, 1e-6), "Gruneisen V0 = 1.0");

    // JWL
    const auto& jwl = reader.eos_map().at(2);
    CHECK(jwl.type == "JWL", "EOS 2 type = JWL");
    CHECK(NEAR(jwl.A_jwl, 3.712E11, 1e5), "JWL A = 3.712E11");
    CHECK(NEAR(jwl.R1, 4.15, 1e-6), "JWL R1 = 4.15");
    CHECK(NEAR(jwl.omega, 0.30, 1e-6), "JWL omega = 0.30");

    // Linear polynomial
    const auto& poly = reader.eos_map().at(3);
    CHECK(poly.type == "LINEAR_POLYNOMIAL", "EOS 3 type = LINEAR_POLYNOMIAL");
    CHECK(NEAR(poly.cpoly[1], 2.0E9, 1e3), "Poly C1 = 2.0E9");
    CHECK(NEAR(poly.cpoly[4], 0.4, 1e-6), "Poly C4 = 0.4");

    // get_eos() mapping
    auto gru_props = reader.get_eos(1);
    CHECK(gru_props.type == physics::EOSType::Gruneisen, "get_eos: Gruneisen type");
    CHECK(NEAR(gru_props.C0, 5240.0, 1e-3), "get_eos: C0 correct");
    CHECK(NEAR(gru_props.gamma0, 1.97, 1e-6), "get_eos: gamma0 correct");

    auto jwl_props = reader.get_eos(2);
    CHECK(jwl_props.type == physics::EOSType::JWL, "get_eos: JWL type");

    auto poly_props = reader.get_eos(3);
    CHECK(poly_props.type == physics::EOSType::LinearPolynomial, "get_eos: Polynomial type");
}

// ============================================================================
// Test 5: Sets
// ============================================================================
void test_sets() {
    std::cout << "\n=== Test 5: Sets ===\n";

    auto reader = read_string(R"(
*KEYWORD
*SET_PART_LIST
         1
         1         2         3
*SET_SHELL_LIST
         2
        10        20        30        40
*SET_SOLID_LIST
         3
       100       200
*END
)");

    CHECK(reader.part_sets().size() == 1, "1 part set parsed");
    CHECK(reader.element_sets().size() == 2, "2 element sets parsed");

    const auto& ps = reader.part_sets().at(1);
    CHECK(ps.parts.size() == 3, "Part set has 3 parts");
    CHECK(ps.parts[0] == 1, "Part set[0] = 1");
    CHECK(ps.parts[2] == 3, "Part set[2] = 3");

    const auto& es_shell = reader.element_sets().at(2);
    CHECK(es_shell.elements.size() == 4, "Shell set has 4 elements");
    CHECK(es_shell.elements[1] == 20, "Shell set[1] = 20");

    const auto& es_solid = reader.element_sets().at(3);
    CHECK(es_solid.elements.size() == 2, "Solid set has 2 elements");
    CHECK(es_solid.elements[0] == 100, "Solid set[0] = 100");
}

// ============================================================================
// Test 6: Initial Conditions
// ============================================================================
void test_initial_conditions() {
    std::cout << "\n=== Test 6: Initial Conditions ===\n";

    auto reader = read_string(R"(
*KEYWORD
*INITIAL_VELOCITY
         1    10.0    20.0    30.0
*INITIAL_VELOCITY_GENERATION
         2   -5.0     0.0    15.0     1.0     2.0     3.0
*END
)");

    CHECK(reader.initial_velocities().size() == 2, "2 initial velocities parsed");

    const auto& iv1 = reader.initial_velocities()[0];
    CHECK(iv1.nsid == 1, "IV1 NSID = 1");
    CHECK(NEAR(iv1.vx, 10.0, 1e-6), "IV1 VX = 10.0");
    CHECK(NEAR(iv1.vy, 20.0, 1e-6), "IV1 VY = 20.0");
    CHECK(NEAR(iv1.vz, 30.0, 1e-6), "IV1 VZ = 30.0");

    const auto& iv2 = reader.initial_velocities()[1];
    CHECK(iv2.nsid == 2, "IV2 NSID = 2");
    CHECK(NEAR(iv2.vx, -5.0, 1e-6), "IV2 VX = -5.0");
    CHECK(NEAR(iv2.vrx, 1.0, 1e-6), "IV2 VRX = 1.0");
}

// ============================================================================
// Test 7: Control
// ============================================================================
void test_control() {
    std::cout << "\n=== Test 7: Control ===\n";

    auto reader = read_string(R"(
*KEYWORD
*CONTROL_TERMINATION
    0.0025
*CONTROL_TIMESTEP
    1.0E-7       0.67         0         0  -1.0E-8
*END
)");

    CHECK(NEAR(reader.control().termination_time, 0.0025, 1e-8), "Termination time = 0.0025");
    CHECK(NEAR(reader.control().dt_init, 1.0E-7, 1e-12), "DT_INIT = 1.0E-7");
    CHECK(NEAR(reader.control().dt_scale, 0.67, 1e-6), "TSSFAC = 0.67");
    CHECK(reader.control().dt_ms_flag == 1, "DT2MS flag set");
    CHECK(NEAR(reader.control().dt_min, -1.0E-8, 1e-14), "DT2MS = -1.0E-8");

    // Default values when not set
    LSDynaReader reader2;
    CHECK(NEAR(reader2.control().termination_time, 1.0, 1e-6), "Default termination = 1.0");
    CHECK(NEAR(reader2.control().dt_scale, 0.9, 1e-6), "Default TSSFAC = 0.9");
    CHECK(reader2.control().dt_ms_flag == 0, "Default DT2MS flag = 0");
}

// ============================================================================
// Test 8: Loads and Output
// ============================================================================
void test_loads_output() {
    std::cout << "\n=== Test 8: Loads and Output ===\n";

    auto reader = read_string(R"(
*KEYWORD
*LOAD_BODY_Z
         1   -9810.0
*LOAD_BODY_X
         2     500.0
*LOAD_BODY_Y
         0    -100.0
*BOUNDARY_PRESCRIBED_MOTION_SET
         3         2         1         5       2.5
*DATABASE_BINARY_D3PLOT
    0.0001
*END
)");

    CHECK(reader.body_loads().size() == 3, "3 body loads parsed");

    const auto& blz = reader.body_loads()[0];
    CHECK(blz.dof == 3, "Body load Z: dof = 3");
    CHECK(blz.lcid == 1, "Body load Z: lcid = 1");
    CHECK(NEAR(blz.sf, -9810.0, 1e-3), "Body load Z: sf = -9810");

    const auto& blx = reader.body_loads()[1];
    CHECK(blx.dof == 1, "Body load X: dof = 1");

    const auto& bly = reader.body_loads()[2];
    CHECK(bly.dof == 2, "Body load Y: dof = 2");

    CHECK(reader.prescribed_motions().size() == 1, "1 prescribed motion parsed");
    const auto& pm = reader.prescribed_motions()[0];
    CHECK(pm.nsid == 3, "PM NSID = 3");
    CHECK(pm.dof == 2, "PM DOF = 2");
    CHECK(pm.vad == 1, "PM VAD = 1 (velocity)");
    CHECK(pm.lcid == 5, "PM LCID = 5");
    CHECK(NEAR(pm.sf, 2.5, 1e-6), "PM SF = 2.5");

    CHECK(NEAR(reader.d3plot_dt(), 0.0001, 1e-8), "D3PLOT DT = 0.0001");
}

// ============================================================================
// Test 9: Full Model Integration
// ============================================================================
void test_full_model() {
    std::cout << "\n=== Test 9: Full Model Integration ===\n";

    auto reader = read_string(R"(
*KEYWORD
*TITLE
Full Integration Test Model
*NODE
       1       0.0       0.0       0.0
       2       1.0       0.0       0.0
       3       1.0       1.0       0.0
       4       0.0       1.0       0.0
       5       0.0       0.0       1.0
       6       1.0       0.0       1.0
       7       1.0       1.0       1.0
       8       0.0       1.0       1.0
       9       2.0       0.0       0.0
      10       2.0       1.0       0.0
*ELEMENT_SOLID
       1         1         1         2         3         4         5         6         7         8
*ELEMENT_SHELL
       2         2         2         3         7         6
*SECTION_SOLID
         1         1
*SECTION_SHELL
         2        16       1.0         3
       2.0       2.0       2.0       2.0
*MAT_ELASTIC
         1    7850.0    2.1E11       0.3
*MAT_RIGID
         2    7800.0    2.0E11       0.3
*PART
Solid Part
         1         1         1
*PART
Shell Part
         2         2         2
*SET_NODE_LIST
         1
         1         2         3         4
*SET_PART_LIST
        10
         1         2
*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE
         1         2
      0.15
*EOS_GRUNEISEN
         1      5240     1.338     0.000     0.000      1.97     0.480
*INITIAL_VELOCITY
         0     0.0     0.0   -100.0
*CONTROL_TERMINATION
    0.005
*CONTROL_TIMESTEP
    0.0         0.9
*LOAD_BODY_Z
         0   -9810.0
*DEFINE_CURVE
         1
               0.0               1.0
             0.005               1.0
*BOUNDARY_SPC_SET
         1         0         1         1         1
*DATABASE_BINARY_D3PLOT
    0.001
*END
)");

    CHECK(reader.title() == "Full Integration Test Model", "Title parsed");
    CHECK(reader.nodes().size() == 10, "10 nodes");
    CHECK(reader.elements().size() == 2, "2 elements");
    CHECK(reader.materials().size() == 2, "2 materials");
    CHECK(reader.parts().size() == 2, "2 parts");
    CHECK(reader.sections().size() == 2, "2 sections");
    CHECK(reader.contacts().size() == 1, "1 contact");
    CHECK(reader.eos_map().size() == 1, "1 EOS");
    CHECK(reader.initial_velocities().size() == 1, "1 initial velocity");
    CHECK(reader.body_loads().size() == 1, "1 body load");
    CHECK(reader.load_curves().size() == 1, "1 load curve");
    CHECK(reader.spcs().size() == 1, "1 SPC");

    // Cross-references: part→section→properties
    auto sec = reader.get_section(2);
    CHECK(NEAR(sec.thickness, 2.0, 1e-6), "Part 2 section thickness = 2.0");

    // Cross-references: part→material→properties
    auto mat = reader.get_material(1);
    CHECK(NEAR(mat.E, 2.1E11, 1e5), "Part 1 E = 2.1E11");
    CHECK(NEAR(mat.density, 7850.0, 1e-3), "Part 1 density = 7850");

    // Control
    CHECK(NEAR(reader.control().termination_time, 0.005, 1e-8), "Termination = 0.005");
    CHECK(NEAR(reader.control().dt_scale, 0.9, 1e-6), "TSSFAC = 0.9");

    // Output
    CHECK(NEAR(reader.d3plot_dt(), 0.001, 1e-6), "D3PLOT dt = 0.001");

    // Summary should not crash
    reader.print_summary();
}

// ============================================================================
// Test 10: Keyword Variants
// ============================================================================
void test_keyword_variants() {
    std::cout << "\n=== Test 10: Keyword Variants ===\n";

    // _TITLE suffix should be stripped
    auto reader = read_string(R"(
*KEYWORD
*MAT_RIGID_TITLE
Rigid steel
         1    7800.0    2.1E11       0.3
*MAT_ELASTIC_TITLE
Aluminum
         2    2700.0    7.0E10       0.33
*SECTION_SHELL_TITLE
Shell section 1
         1         2       1.0         2
       1.0
*END
)");

    CHECK(reader.materials().size() == 2, "2 materials from _TITLE variants");
    CHECK(reader.materials().at(1).type == "RIGID", "MAT_RIGID_TITLE → RIGID");
    CHECK(reader.materials().at(2).type == "ELASTIC", "MAT_ELASTIC_TITLE → ELASTIC");
    CHECK(reader.sections().size() == 1, "1 section from _TITLE variant");
    CHECK(reader.sections().at(1).type == "SHELL", "SECTION_SHELL_TITLE → SHELL");

    // Comma-separated format
    auto reader2 = read_string(R"(
*KEYWORD
*MAT_ELASTIC
1,7850.0,2.0E11,0.3
*SECTION_SOLID
1,1
*END
)");

    CHECK(reader2.materials().size() == 1, "Comma-separated material parsed");
    CHECK(NEAR(reader2.materials().at(1).e, 2.0E11, 1e5), "Comma-separated E correct");
    CHECK(reader2.sections().size() == 1, "Comma-separated section parsed");

    // Mixed keyword ordering (EOS before MAT)
    auto reader3 = read_string(R"(
*KEYWORD
*EOS_JWL
         1   3.0E11   2.0E9       4.0      1.0      0.35   5.0E9
*MAT_ELASTIC
         1    7850.0    2.0E11       0.3
*SECTION_SHELL
         1         2
       1.0
*PART
Test
         1         1         1
*CONTACT_TIED_SURFACE_TO_SURFACE
         1         2
*INITIAL_VELOCITY
         1     0.0     0.0   -50.0
*END
)");

    CHECK(reader3.eos_map().size() == 1, "Mixed order: EOS parsed");
    CHECK(reader3.materials().size() == 1, "Mixed order: MAT parsed");
    CHECK(reader3.sections().size() == 1, "Mixed order: SECTION parsed");
    CHECK(reader3.contacts().size() == 1, "Mixed order: CONTACT parsed");
    CHECK(reader3.initial_velocities().size() == 1, "Mixed order: INITIAL_VELOCITY parsed");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "NexusSim LS-DYNA Reader Extension Tests\n";
    std::cout << "========================================\n";

    test_sections();
    test_materials();
    test_contacts();
    test_eos();
    test_sets();
    test_initial_conditions();
    test_control();
    test_loads_output();
    test_full_model();
    test_keyword_variants();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    std::cout << "========================================\n";

    return tests_failed > 0 ? 1 : 0;
}
