/**
 * @file starter_wave40_test.cpp
 * @brief Tests for Wave 40: Starter/Reader Extensions and Error System
 *
 * Covers: RadiossStarterFull, PropertyReader, SectionForceOutput,
 *         ModelValidatorExt, ErrorMessageSystem
 */

#include <nexussim/io/starter_wave40.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <sstream>
#include <fstream>

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

using namespace nxs::io;

// ============================================================================
// Helper: write a temporary file and return its path
// ============================================================================
static std::string write_temp_file(const std::string& name, const std::string& content) {
    std::string path = "/tmp/nxs_wave40_" + name;
    std::ofstream f(path);
    f << content;
    f.close();
    return path;
}

// ============================================================================
// 1. RadiossStarterFull tests
// ============================================================================
void test_starter_nonexistent_file() {
    RadiossStarterFull starter;
    auto model = starter.parse("/tmp/nxs_wave40_does_not_exist.d00");
    CHECK(model.parse_errors == 1, "starter: nonexistent file sets parse_errors=1");
    CHECK(model.nodes.empty(), "starter: nonexistent file yields empty nodes");
}

void test_starter_empty_file() {
    auto path = write_temp_file("empty.d00", "");
    RadiossStarterFull starter;
    auto model = starter.parse(path);
    CHECK(model.parse_errors == 0, "starter: empty file no parse errors");
    CHECK(model.node_count() == 0, "starter: empty file zero nodes");
    CHECK(model.element_count() == 0, "starter: empty file zero elements");
}

void test_starter_begin_title() {
    std::string content =
        "/BEGIN\n"
        "My Test Model\n";
    auto path = write_temp_file("title.d00", content);
    RadiossStarterFull starter;
    auto model = starter.parse(path);
    CHECK(model.title == "My Test Model", "starter: /BEGIN parses title");
}

void test_starter_node_parsing() {
    // Fixed-field format: 10-char wide fields after trim: ID, X, Y, Z
    // Fields must be aligned from position 0 after leading whitespace is stripped.
    std::string content =
        "/NODE\n"
        "1         0.0       0.0       0.0       \n"
        "2         1.0       0.0       0.0       \n"
        "3         1.0       1.0       0.0       \n";
    auto path = write_temp_file("nodes.d00", content);
    RadiossStarterFull starter;
    auto model = starter.parse(path);
    CHECK(model.node_count() == 3, "starter: parsed 3 nodes");
    CHECK(model.nodes[0].id == 1, "starter: node 1 id");
    CHECK_NEAR(model.nodes[1].x, 1.0, 1e-10, "starter: node 2 x");
    CHECK_NEAR(model.nodes[2].y, 1.0, 1e-10, "starter: node 3 y");
}

void test_starter_shell_parsing() {
    // SHELL: ID, PART_ID, N1, N2, N3, N4 (6 fields minimum, 10-char each after trim)
    std::string content =
        "/SHELL\n"
        "1         1         1         2         3         4         \n";
    auto path = write_temp_file("shell.d00", content);
    RadiossStarterFull starter;
    auto model = starter.parse(path);
    CHECK(model.element_count() == 1, "starter: parsed 1 shell element");
    CHECK(model.elements[0].type == "SHELL", "starter: element type is SHELL");
    CHECK(model.elements[0].part_id == 1, "starter: shell part_id");
    CHECK(static_cast<int>(model.elements[0].node_ids.size()) == 4,
          "starter: shell has 4 node ids");
}

void test_starter_brick_parsing() {
    // BRICK: ID, PART_ID, N1..N8 (10 fields minimum, 10-char each after trim)
    std::string content =
        "/BRICK\n"
        "1         1         1         2         3         4         5         6         7         8         \n";
    auto path = write_temp_file("brick.d00", content);
    RadiossStarterFull starter;
    auto model = starter.parse(path);
    CHECK(model.element_count() == 1, "starter: parsed 1 brick element");
    CHECK(model.elements[0].type == "BRICK", "starter: element type is BRICK");
    CHECK(static_cast<int>(model.elements[0].node_ids.size()) == 8,
          "starter: brick has 8 node ids");
}

void test_starter_bcs_parsing() {
    // BCS: node_id(10), trarot(10), group_name(10) -- 10-char fields after trim
    std::string content =
        "/BCS\n"
        "1         111000    fixed     \n"
        "2         111111    clamped   \n";
    auto path = write_temp_file("bcs.d00", content);
    RadiossStarterFull starter;
    auto model = starter.parse(path);
    CHECK(static_cast<int>(model.bcs.size()) == 2, "starter: parsed 2 BCs");
    CHECK(model.bcs[0].node_id == 1, "starter: BC node_id");
    CHECK(model.bcs[0].trarot == 111000, "starter: BC trarot");
    CHECK(model.bcs[1].group_name == "clamped", "starter: BC group_name");
}

void test_starter_custom_parser() {
    RadiossStarterFull starter;
    bool custom_called = false;
    starter.register_parser("/CUSTOM", [&custom_called](std::ifstream&, StarterModel& m) {
        custom_called = true;
        m.title = "custom_parsed";
    });
    std::string content = "/CUSTOM\nsome data\n";
    auto path = write_temp_file("custom.d00", content);
    auto model = starter.parse(path);
    CHECK(custom_called, "starter: custom parser was called");
    CHECK(model.title == "custom_parsed", "starter: custom parser modified model");
}

void test_starter_comment_lines() {
    // Comments before keywords are skipped by the main loop.
    std::string content =
        "# This is a comment at the top\n"
        "/NODE\n"
        "1         5.0       6.0       7.0       \n"
        "2         8.0       9.0       10.0      \n";
    auto path = write_temp_file("comments.d00", content);
    RadiossStarterFull starter;
    auto model = starter.parse(path);
    CHECK(model.node_count() == 2, "starter: comments skipped, 2 nodes parsed");
    CHECK_NEAR(model.nodes[0].x, 5.0, 1e-10, "starter: node 1 x after comment");
    CHECK_NEAR(model.nodes[1].x, 8.0, 1e-10, "starter: node 2 x");
}

// ============================================================================
// 2. PropertyReader tests
// ============================================================================
void test_property_reader_shell() {
    PropertyReader reader;
    // 10-char fields after trim: "1" + pad to 10, "SHELL" + pad to 10
    std::vector<std::string> lines = {
        "1         SHELL     ",
        "2.5       5         "
    };
    auto prop = reader.read_property(lines);
    CHECK(prop.id == 1, "prop_reader: shell id");
    CHECK(prop.type == "SHELL", "prop_reader: shell type");
    CHECK_NEAR(prop.thickness(), 2.5, 1e-10, "prop_reader: shell thickness");
    CHECK(prop.num_integration_points() == 5, "prop_reader: shell nip");
}

void test_property_reader_beam() {
    PropertyReader reader;
    std::vector<std::string> lines = {
        "2         BEAM      ",
        "0.01      1e-4      2e-4      "
    };
    auto prop = reader.read_property(lines);
    CHECK(prop.id == 2, "prop_reader: beam id");
    CHECK(prop.type == "BEAM", "prop_reader: beam type");
    CHECK_NEAR(prop.cross_section_area(), 0.01, 1e-10, "prop_reader: beam area");
    CHECK_NEAR(prop.iyy(), 1e-4, 1e-10, "prop_reader: beam Iyy");
    CHECK_NEAR(prop.izz(), 2e-4, 1e-10, "prop_reader: beam Izz");
}

void test_property_reader_solid() {
    PropertyReader reader;
    std::vector<std::string> lines = {
        "3         SOLID     ",
        "2         "
    };
    auto prop = reader.read_property(lines);
    CHECK(prop.id == 3, "prop_reader: solid id");
    CHECK(prop.type == "SOLID", "prop_reader: solid type");
    CHECK(prop.integration_rule() == 2, "prop_reader: solid integration_rule");
}

void test_property_reader_line_shell() {
    PropertyReader reader;
    auto prop = reader.read_property_line("10 shell 1.5 3");
    CHECK(prop.id == 10, "prop_line: shell id");
    CHECK(prop.type == "SHELL", "prop_line: type uppercased");
    CHECK_NEAR(prop.thickness(), 1.5, 1e-10, "prop_line: shell thickness");
    CHECK(prop.num_integration_points() == 3, "prop_line: shell nip");
}

void test_property_reader_line_beam() {
    PropertyReader reader;
    auto prop = reader.read_property_line("20 beam 0.05 3e-5 4e-5");
    CHECK(prop.id == 20, "prop_line: beam id");
    CHECK(prop.type == "BEAM", "prop_line: beam type");
    CHECK_NEAR(prop.cross_section_area(), 0.05, 1e-10, "prop_line: beam area");
    CHECK_NEAR(prop.iyy(), 3e-5, 1e-10, "prop_line: beam Iyy");
    CHECK_NEAR(prop.izz(), 4e-5, 1e-10, "prop_line: beam Izz");
}

void test_property_reader_empty() {
    PropertyReader reader;
    std::vector<std::string> lines;
    auto prop = reader.read_property(lines);
    CHECK(prop.id == 0, "prop_reader: empty lines gives default id");
    CHECK(prop.thickness() == 0.0, "prop_reader: empty lines gives zero thickness");
}

// ============================================================================
// 3. SectionForceOutput tests
// ============================================================================
void test_section_force_undefined() {
    SectionForceOutput sec;
    CHECK(!sec.is_defined(), "section: initially undefined");
    std::vector<ElementStressData> elems;
    auto result = sec.compute(elems);
    CHECK(result.contributing_elements == 0, "section: undefined yields 0 contributors");
}

void test_section_force_define() {
    SectionForceOutput sec;
    sec.define_section({1.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
    CHECK(sec.is_defined(), "section: defined after define_section");
    CHECK_NEAR(sec.section_point()[0], 1.0, 1e-10, "section: point x");
    CHECK_NEAR(sec.section_normal()[0], 1.0, 1e-10, "section: normal x");
    CHECK_NEAR(sec.section_normal()[1], 0.0, 1e-10, "section: normal y");
}

void test_section_force_normal_normalization() {
    SectionForceOutput sec;
    sec.define_section({0, 0, 0}, {3.0, 4.0, 0.0});
    CHECK_NEAR(sec.section_normal()[0], 0.6, 1e-10, "section: normalized nx");
    CHECK_NEAR(sec.section_normal()[1], 0.8, 1e-10, "section: normalized ny");
}

void test_section_force_compute_uniaxial() {
    // Cutting plane at x=0, normal in x-direction
    // Element at centroid (0,0,0) with uniaxial stress sxx=100, area=2
    SectionForceOutput sec;
    sec.define_section({0, 0, 0}, {1, 0, 0});
    sec.set_tolerance(0.5);

    ElementStressData e;
    e.id = 1;
    e.centroid = {0.0, 0.0, 0.0};
    e.stress = {100.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // sxx=100
    e.area = 2.0;

    auto result = sec.compute({e});
    CHECK(result.contributing_elements == 1, "section: 1 contributor in uniaxial");
    CHECK_NEAR(result.fx, 200.0, 1e-10, "section: fx = sxx * area");
    CHECK_NEAR(result.fy, 0.0, 1e-10, "section: fy = 0 in uniaxial");
    CHECK_NEAR(result.fz, 0.0, 1e-10, "section: fz = 0 in uniaxial");
}

void test_section_force_out_of_tolerance() {
    SectionForceOutput sec;
    sec.define_section({0, 0, 0}, {1, 0, 0});
    sec.set_tolerance(0.05);

    ElementStressData e;
    e.id = 1;
    e.centroid = {1.0, 0.0, 0.0};  // Far from plane
    e.stress = {100.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    e.area = 1.0;

    auto result = sec.compute({e});
    CHECK(result.contributing_elements == 0, "section: element outside tolerance excluded");
    CHECK_NEAR(result.fx, 0.0, 1e-10, "section: no force when no contributors");
}

void test_section_force_magnitude() {
    SectionForces sf;
    sf.fx = 3.0; sf.fy = 4.0; sf.fz = 0.0;
    sf.mx = 0.0; sf.my = 0.0; sf.mz = 5.0;
    CHECK_NEAR(sf.force_magnitude(), 5.0, 1e-10, "section: force magnitude 3-4-5");
    CHECK_NEAR(sf.moment_magnitude(), 5.0, 1e-10, "section: moment magnitude");
}

void test_section_force_moment() {
    // Element offset from section point should produce moment
    SectionForceOutput sec;
    sec.define_section({0, 0, 0}, {1, 0, 0});
    sec.set_tolerance(0.5);

    ElementStressData e;
    e.id = 1;
    e.centroid = {0.0, 1.0, 0.0};  // offset in y
    e.stress = {100.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // sxx=100
    e.area = 1.0;
    // Force = (100, 0, 0), r = (0, 1, 0)
    // M = r x F = (1*0 - 0*0, 0*100 - 0*0, 0*0 - 1*100) = (0, 0, -100)

    auto result = sec.compute({e});
    CHECK_NEAR(result.mz, -100.0, 1e-10, "section: moment mz from offset");
}

// ============================================================================
// 4. ModelValidatorExt tests
// ============================================================================
void test_validator_empty_model() {
    ModelValidatorExt validator;
    StarterModel model;
    auto report = validator.validate(model);
    CHECK(report.is_valid(), "validator: empty model is valid");
    CHECK(report.error_count() == 0, "validator: empty model no errors");
}

void test_validator_valid_model() {
    ModelValidatorExt validator;
    StarterModel model;
    model.nodes.push_back({1, 0, 0, 0});
    model.nodes.push_back({2, 1, 0, 0});
    model.nodes.push_back({3, 1, 1, 0});
    model.nodes.push_back({4, 0, 1, 0});

    StarterElement elem;
    elem.id = 1; elem.part_id = 1; elem.type = "SHELL";
    elem.node_ids = {1, 2, 3, 4};
    model.elements.push_back(elem);

    auto report = validator.validate(model);
    CHECK(report.is_valid(), "validator: valid model passes");
    CHECK(report.info_count() > 0, "validator: info messages present");
}

void test_validator_dangling_node() {
    ModelValidatorExt validator;
    StarterModel model;
    model.nodes.push_back({1, 0, 0, 0});

    StarterElement elem;
    elem.id = 1; elem.type = "SHELL";
    elem.node_ids = {1, 2, 3, 4};  // nodes 2,3,4 don't exist
    model.elements.push_back(elem);

    auto report = validator.validate(model);
    CHECK(!report.is_valid(), "validator: dangling nodes cause errors");
    CHECK(report.error_count() >= 1, "validator: at least 1 error for dangling");
}

void test_validator_duplicate_bc() {
    ModelValidatorExt validator;
    StarterModel model;
    model.nodes.push_back({1, 0, 0, 0});
    model.bcs.push_back({1, 111000, "g1"});
    model.bcs.push_back({1, 111111, "g2"});  // duplicate BC on node 1

    auto report = validator.validate(model);
    CHECK(report.warning_count() >= 1, "validator: duplicate BC triggers warning");
}

void test_validator_negative_friction() {
    ModelValidatorExt validator;
    StarterModel model;
    StarterInterface iface;
    iface.id = 1; iface.type = 7; iface.friction = -0.1;
    model.interfaces.push_back(iface);

    auto report = validator.validate(model);
    CHECK(report.warning_count() >= 1, "validator: negative friction triggers warning");
}

void test_validator_summary_string() {
    ValidationReport report;
    report.errors.push_back({"cat", "err1"});
    report.warnings.push_back({"cat", "warn1"});
    report.info.push_back({"cat", "info1"});
    std::string s = report.summary();
    CHECK(s.find("1 errors") != std::string::npos, "validator: summary has error count");
    CHECK(s.find("1 warnings") != std::string::npos, "validator: summary has warning count");
}

// ============================================================================
// 5. ErrorMessageSystem tests
// ============================================================================
void test_errsys_empty() {
    ErrorMessageSystem ems;
    CHECK(ems.count() == 0, "errsys: initially empty");
    CHECK(!ems.has_errors(), "errsys: no errors initially");
    CHECK(!ems.has_warnings(), "errsys: no warnings initially");
}

void test_errsys_add_error() {
    ErrorMessageSystem ems;
    ems.error(1001, "Something failed", "Fix it");
    CHECK(ems.count() == 1, "errsys: 1 message after error");
    CHECK(ems.has_errors(), "errsys: has_errors after adding error");
    auto s = ems.get_summary();
    CHECK(s.error_count == 1, "errsys: summary error_count");
    CHECK(s.total() == 1, "errsys: summary total");
}

void test_errsys_add_error_with_context() {
    ErrorMessageSystem ems;
    ems.error(2001, "material", "Invalid density", "Set density > 0");
    CHECK(ems.messages()[0].context == "material", "errsys: error context");
    CHECK(ems.messages()[0].suggestion == "Set density > 0", "errsys: error suggestion");
}

void test_errsys_add_warning() {
    ErrorMessageSystem ems;
    ems.warning(3001, "Low quality element");
    CHECK(ems.has_warnings(), "errsys: has_warnings");
    CHECK(!ems.has_errors(), "errsys: no errors after warning only");
}

void test_errsys_add_info_debug() {
    ErrorMessageSystem ems;
    ems.info(4001, "Model loaded");
    ems.debug(5001, "Entering solver loop");
    auto s = ems.get_summary();
    CHECK(s.info_count == 1, "errsys: info_count");
    CHECK(s.debug_count == 1, "errsys: debug_count");
    CHECK(s.total() == 2, "errsys: total info+debug");
}

void test_errsys_filter_by_severity() {
    ErrorMessageSystem ems;
    ems.error(1, "err1");
    ems.warning(2, "warn1");
    ems.info(3, "info1");
    ems.error(4, "err2");

    auto errors = ems.messages_by_severity(ErrorMessageSystem::Severity::Error);
    CHECK(static_cast<int>(errors.size()) == 2, "errsys: filter returns 2 errors");
    auto warnings = ems.messages_by_severity(ErrorMessageSystem::Severity::Warning);
    CHECK(static_cast<int>(warnings.size()) == 1, "errsys: filter returns 1 warning");
}

void test_errsys_clear() {
    ErrorMessageSystem ems;
    ems.error(1, "err");
    ems.warning(2, "warn");
    ems.clear();
    CHECK(ems.count() == 0, "errsys: clear empties messages");
    CHECK(!ems.has_errors(), "errsys: no errors after clear");
}

void test_errsys_write_log() {
    ErrorMessageSystem ems;
    ems.error(1001, "solver", "Diverged", "Reduce timestep");
    ems.warning(2001, "Large displacement detected");
    ems.info(3001, "Step completed");

    std::string path = "/tmp/nxs_wave40_test.log";
    bool ok = ems.write_log(path);
    CHECK(ok, "errsys: write_log succeeds");

    // Read back and verify content
    std::ifstream f(path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    CHECK(content.find("ERROR") != std::string::npos, "errsys: log contains ERROR");
    CHECK(content.find("Diverged") != std::string::npos, "errsys: log contains message text");
    CHECK(content.find("Suggestion") != std::string::npos, "errsys: log contains suggestion");
}

void test_errsys_warning_with_context() {
    ErrorMessageSystem ems;
    ems.warning(100, "element", "Jacobian negative");
    CHECK(ems.messages()[0].context == "element", "errsys: warning context");
    CHECK(ems.messages()[0].text == "Jacobian negative", "errsys: warning text");
}

// ============================================================================
// Struct/data tests
// ============================================================================
void test_starter_node_defaults() {
    StarterNode n;
    CHECK(n.id == 0, "struct: StarterNode default id");
    CHECK_NEAR(n.x, 0.0, 1e-15, "struct: StarterNode default x");
}

void test_property_card_defaults() {
    PropertyCard p;
    CHECK(p.thickness() == 0.0, "struct: PropertyCard default thickness");
    CHECK(p.num_integration_points() == 1, "struct: PropertyCard default nip");
    CHECK(p.cross_section_area() == 0.0, "struct: PropertyCard default area");
    CHECK(p.integration_rule() == 0, "struct: PropertyCard default integration_rule");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== Wave 40: Starter/Reader Extensions and Error System ===\n\n";

    // RadiossStarterFull
    test_starter_nonexistent_file();
    test_starter_empty_file();
    test_starter_begin_title();
    test_starter_node_parsing();
    test_starter_shell_parsing();
    test_starter_brick_parsing();
    test_starter_bcs_parsing();
    test_starter_custom_parser();
    test_starter_comment_lines();

    // PropertyReader
    test_property_reader_shell();
    test_property_reader_beam();
    test_property_reader_solid();
    test_property_reader_line_shell();
    test_property_reader_line_beam();
    test_property_reader_empty();

    // SectionForceOutput
    test_section_force_undefined();
    test_section_force_define();
    test_section_force_normal_normalization();
    test_section_force_compute_uniaxial();
    test_section_force_out_of_tolerance();
    test_section_force_magnitude();
    test_section_force_moment();

    // ModelValidatorExt
    test_validator_empty_model();
    test_validator_valid_model();
    test_validator_dangling_node();
    test_validator_duplicate_bc();
    test_validator_negative_friction();
    test_validator_summary_string();

    // ErrorMessageSystem
    test_errsys_empty();
    test_errsys_add_error();
    test_errsys_add_error_with_context();
    test_errsys_add_warning();
    test_errsys_add_info_debug();
    test_errsys_filter_by_severity();
    test_errsys_clear();
    test_errsys_write_log();
    test_errsys_warning_with_context();

    // Struct defaults
    test_starter_node_defaults();
    test_property_card_defaults();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " ===\n";

    return tests_failed > 0 ? 1 : 0;
}
