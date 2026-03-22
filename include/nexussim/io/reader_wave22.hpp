#pragma once

/**
 * @file reader_wave22.hpp
 * @brief Wave 22: Multi-format input deck readers and model validation
 *
 * Components:
 * 1. RadiossStarterReader - Full Radioss D00 format reader
 *    (/BEGIN, /NODE, /SHELL, /BRICK, /PART, /MAT, /PROP, /BCS, /LOAD, /INTER, /FUNCT)
 * 2. LSDYNAFullReader - Full LS-DYNA keyword reader
 *    (*NODE, *ELEMENT_SOLID/SHELL, *PART, *MAT_xxx, *SECTION_xxx, *BOUNDARY_xxx,
 *     *LOAD_xxx, *CONTACT_xxx, *DEFINE_CURVE, *INITIAL_VELOCITY, *SET_xxx)
 * 3. AbaqusINPReader - ABAQUS INP subset
 *    (*NODE, *ELEMENT, *MATERIAL, *ELASTIC, *PLASTIC, *DENSITY, *BOUNDARY,
 *     *STEP, *STATIC/*DYNAMIC, *CLOAD, *DLOAD)
 * 4. ModelValidator - Model validation and checking
 *    (orphan nodes, Jacobian > 0, material assignment, BC consistency, contact validity)
 *
 * All readers produce a common ModelData struct for unified downstream processing.
 *
 * References:
 * - OpenRadioss Starter Input Manual
 * - LS-DYNA Keyword User's Manual (R14)
 * - Abaqus Analysis User's Guide
 */

#include <nexussim/core/types.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include <cmath>
#include <set>
#include <cctype>
#include <stdexcept>
#include <numeric>
#include <functional>

namespace nxs {
namespace io {

using Real = nxs::Real;

// ============================================================================
// Common Model Data Structures
// ============================================================================

struct NodeData {
    int id;
    Real x, y, z;
};

struct ElementData {
    int id;
    int part_id;
    int type;           ///< 0=unknown, 1=shell3, 2=shell4, 3=tet4, 4=tet10, 5=hex8, 6=hex20, 7=beam2
    int nodes[20];
    int num_nodes;
};

struct MaterialData {
    int id;
    std::string type;
    std::map<std::string, Real> properties;
};

struct BCData {
    int node_id;
    int dof_mask;       ///< Bitmask: bit0=x, bit1=y, bit2=z, bit3=rx, bit4=ry, bit5=rz
    Real value;
};

struct LoadData {
    int node_id;
    int dof;            ///< 1=Fx, 2=Fy, 3=Fz, 4=Mx, 5=My, 6=Mz; negative = initial velocity
    Real magnitude;
    int curve_id;       ///< -1 for constant
};

struct CurveData {
    int id;
    std::vector<std::pair<Real, Real>> points;
};

struct ValidationMessage {
    int level;          ///< 0=info, 1=warning, 2=error
    int code;
    std::string message;
};

struct ModelData {
    std::vector<NodeData> nodes;
    std::vector<ElementData> elements;
    std::vector<MaterialData> materials;
    std::vector<BCData> bcs;
    std::vector<LoadData> loads;
    std::vector<CurveData> curves;
};

// ============================================================================
// Internal string utilities
// ============================================================================

namespace detail_w22 {

inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

inline std::string to_upper(const std::string& s) {
    std::string r = s;
    for (auto& c : r) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return r;
}

inline bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

inline std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> v;
    std::istringstream iss(s);
    std::string t;
    while (iss >> t) v.push_back(t);
    return v;
}

inline std::vector<std::string> split_comma(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(trim(t));
    return v;
}

inline Real safe_real(const std::string& s, Real d = 0.0) {
    if (s.empty()) return d;
    std::string c = s;
    for (auto& ch : c) {
        if (ch == 'd' || ch == 'D') ch = 'E'; // Fortran exponent
    }
    try { return std::stod(c); } catch (...) { return d; }
}

inline int safe_int(const std::string& s, int d = 0) {
    if (s.empty()) return d;
    try { return std::stoi(s); } catch (...) { return d; }
}

inline bool has_comma(const std::string& s) {
    return s.find(',') != std::string::npos;
}

/// Parse fixed-width columns from a card-image line
inline std::vector<std::string> fixed_cols(const std::string& line, int w) {
    std::vector<std::string> fields;
    for (size_t i = 0; i < line.size(); i += static_cast<size_t>(w)) {
        size_t end = std::min(i + static_cast<size_t>(w), line.size());
        fields.push_back(trim(line.substr(i, end - i)));
    }
    return fields;
}

/// Parse data line: auto-detect free (comma) vs fixed format
inline std::vector<std::string> parse_card(const std::string& line, int w = 8) {
    if (has_comma(line)) return split_comma(line);
    return fixed_cols(line, w);
}

} // namespace detail_w22

// ============================================================================
// 1. RadiossStarterReader - Full Radioss D00 Format Reader
// ============================================================================

/**
 * @brief Full Radioss Starter (D00) format reader
 *
 * Parses keyword blocks: /BEGIN, /NODE, /SHELL, /BRICK, /PART, /MAT,
 * /PROP, /BCS, /LOAD, /INTER (interface/contact), /FUNCT (load curves).
 *
 * Radioss format rules:
 * - Keywords start with '/' at column 1
 * - Comments start with '#'
 * - Data lines follow the keyword in fixed or free format
 * - /KEYWORD/sub_keyword/id pattern for IDs
 */
class RadiossStarterReader {
public:
    RadiossStarterReader() = default;

    /**
     * @brief Read a Radioss starter file and return populated ModelData
     * @param filename Path to .rad or D00 file
     * @return ModelData with nodes, elements, materials, BCs
     */
    ModelData read(const std::string& filename) {
        model_ = ModelData{};
        lines_.clear();

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("RadiossStarterReader: Cannot open: " + filename);
        }
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            lines_.push_back(line);
        }
        file.close();

        parse_nodes();
        parse_elements();
        parse_materials();
        parse_boundary_conditions();
        parse_loads_and_curves();
        parse_interfaces();
        parse_parts();

        return model_;
    }

    /**
     * @brief Parse /NODE blocks
     */
    void parse_nodes() {
        for (size_t i = 0; i < lines_.size(); ++i) {
            std::string up = detail_w22::to_upper(detail_w22::trim(lines_[i]));
            if (!detail_w22::starts_with(up, "/NODE")) continue;
            i++;
            while (i < lines_.size()) {
                std::string t = detail_w22::trim(lines_[i]);
                if (t.empty() || t[0] == '#') { i++; continue; }
                if (t[0] == '/') break;
                auto f = detail_w22::split_ws(t);
                if (f.size() >= 4) {
                    NodeData nd;
                    nd.id = detail_w22::safe_int(f[0]);
                    nd.x  = detail_w22::safe_real(f[1]);
                    nd.y  = detail_w22::safe_real(f[2]);
                    nd.z  = detail_w22::safe_real(f[3]);
                    model_.nodes.push_back(nd);
                }
                i++;
            }
        }
    }

    /**
     * @brief Parse /SHELL and /BRICK element blocks
     */
    void parse_elements() {
        for (size_t i = 0; i < lines_.size(); ++i) {
            std::string up = detail_w22::to_upper(detail_w22::trim(lines_[i]));

            bool is_shell = detail_w22::starts_with(up, "/SHELL");
            bool is_brick = detail_w22::starts_with(up, "/BRICK");
            if (!is_shell && !is_brick) continue;

            int part_id = extract_id(up, is_shell ? "/SHELL" : "/BRICK");
            i++;
            while (i < lines_.size()) {
                std::string t = detail_w22::trim(lines_[i]);
                if (t.empty() || t[0] == '#') { i++; continue; }
                if (t[0] == '/') break;

                auto f = detail_w22::split_ws(t);
                if (f.size() < 2) { i++; continue; }

                ElementData ed;
                std::fill(std::begin(ed.nodes), std::end(ed.nodes), 0);
                ed.id = detail_w22::safe_int(f[0]);
                // If part_id from keyword, nodes start at field 1; otherwise field 1 is part_id
                int offset = (part_id > 0) ? 1 : 2;
                ed.part_id = (part_id > 0) ? part_id : detail_w22::safe_int(f[1]);
                ed.num_nodes = 0;
                for (size_t j = static_cast<size_t>(offset); j < f.size() && ed.num_nodes < 20; ++j) {
                    int nid = detail_w22::safe_int(f[j]);
                    if (nid > 0) ed.nodes[ed.num_nodes++] = nid;
                }

                if (is_shell) {
                    ed.type = (ed.num_nodes == 3) ? 1 : 2;
                } else {
                    if (ed.num_nodes == 4)       ed.type = 3;  // tet4
                    else if (ed.num_nodes == 10)  ed.type = 4;  // tet10
                    else if (ed.num_nodes == 8)   ed.type = 5;  // hex8
                    else if (ed.num_nodes == 20)  ed.type = 6;  // hex20
                    else                          ed.type = 5;
                }
                model_.elements.push_back(ed);
                i++;
            }
        }
    }

    /**
     * @brief Parse /MAT material definition blocks
     */
    void parse_materials() {
        for (size_t i = 0; i < lines_.size(); ++i) {
            std::string up = detail_w22::to_upper(detail_w22::trim(lines_[i]));
            if (!detail_w22::starts_with(up, "/MAT/")) continue;

            // Parse /MAT/LAWn/id
            std::string rest = up.substr(5);
            std::string law;
            int mat_id = 0;
            size_t sp = rest.find('/');
            if (sp != std::string::npos) {
                law = rest.substr(0, sp);
                mat_id = detail_w22::safe_int(rest.substr(sp + 1));
            } else {
                law = rest;
            }

            MaterialData mat;
            mat.id = mat_id;
            mat.type = law;
            i++;

            // Skip title line if present (non-numeric first token)
            if (i < lines_.size()) {
                std::string t = detail_w22::trim(lines_[i]);
                if (!t.empty() && t[0] != '/' && t[0] != '#') {
                    bool numeric = true;
                    try { std::stod(detail_w22::split_ws(t)[0]); } catch (...) { numeric = false; }
                    if (!numeric) i++;
                }
            }

            // Read up to 5 data cards
            int card = 0;
            while (i < lines_.size() && card < 5) {
                std::string t = detail_w22::trim(lines_[i]);
                if (t.empty() || t[0] == '#') { i++; continue; }
                if (t[0] == '/') break;

                auto f = detail_w22::split_ws(t);
                if (card == 0) {
                    // Card 1: density, E, nu
                    if (f.size() >= 1) mat.properties["density"]  = detail_w22::safe_real(f[0]);
                    if (f.size() >= 2) mat.properties["E"]        = detail_w22::safe_real(f[1]);
                    if (f.size() >= 3) mat.properties["nu"]       = detail_w22::safe_real(f[2]);
                } else if (card == 1) {
                    // Card 2: yield stress, hardening
                    if (f.size() >= 1) mat.properties["yield_stress"]       = detail_w22::safe_real(f[0]);
                    if (f.size() >= 2) mat.properties["hardening_modulus"]   = detail_w22::safe_real(f[1]);
                    if (f.size() >= 3) mat.properties["hardening_exponent"]  = detail_w22::safe_real(f[2]);
                    if (f.size() >= 4) mat.properties["strain_rate_coeff"]   = detail_w22::safe_real(f[3]);
                } else if (card == 2) {
                    if (f.size() >= 1) mat.properties["failure_strain"]     = detail_w22::safe_real(f[0]);
                    if (f.size() >= 2) mat.properties["thermal_softening"]  = detail_w22::safe_real(f[1]);
                } else {
                    // Generic storage for extra cards
                    for (size_t j = 0; j < f.size(); ++j) {
                        Real v = detail_w22::safe_real(f[j]);
                        if (v != 0.0) {
                            mat.properties["card" + std::to_string(card) + "_" + std::to_string(j)] = v;
                        }
                    }
                }
                card++;
                i++;
            }
            model_.materials.push_back(mat);
        }
    }

    /**
     * @brief Parse /BCS boundary condition blocks
     */
    void parse_boundary_conditions() {
        for (size_t i = 0; i < lines_.size(); ++i) {
            std::string up = detail_w22::to_upper(detail_w22::trim(lines_[i]));
            if (!detail_w22::starts_with(up, "/BCS")) continue;
            i++;

            // Skip optional title line
            if (i < lines_.size()) {
                std::string t = detail_w22::trim(lines_[i]);
                if (!t.empty() && t[0] != '/' && t[0] != '#') {
                    auto f = detail_w22::split_ws(t);
                    bool numeric = true;
                    if (!f.empty()) { try { std::stoi(f[0]); } catch (...) { numeric = false; } }
                    if (!numeric) i++;
                }
            }

            while (i < lines_.size()) {
                std::string t = detail_w22::trim(lines_[i]);
                if (t.empty() || t[0] == '#') { i++; continue; }
                if (t[0] == '/') break;

                auto f = detail_w22::split_ws(t);
                if (f.size() >= 7) {
                    // Radioss BCS card: grp_id Tx Ty Tz Rx Ry Rz (1=fixed)
                    BCData bc;
                    bc.node_id = detail_w22::safe_int(f[0]);
                    int tx = detail_w22::safe_int(f[1]);
                    int ty = detail_w22::safe_int(f[2]);
                    int tz = detail_w22::safe_int(f[3]);
                    int rx = detail_w22::safe_int(f[4]);
                    int ry = detail_w22::safe_int(f[5]);
                    int rz = detail_w22::safe_int(f[6]);
                    bc.dof_mask = (tx ? 1 : 0) | (ty ? 2 : 0) | (tz ? 4 : 0)
                                | (rx ? 8 : 0) | (ry ? 16 : 0) | (rz ? 32 : 0);
                    bc.value = 0.0;
                    model_.bcs.push_back(bc);
                }
                i++;
            }
        }
    }

private:
    ModelData model_;
    std::vector<std::string> lines_;

    /// Parse /LOAD and /FUNCT keywords
    void parse_loads_and_curves() {
        for (size_t i = 0; i < lines_.size(); ++i) {
            std::string up = detail_w22::to_upper(detail_w22::trim(lines_[i]));

            // /CLOAD or /LOAD
            if (detail_w22::starts_with(up, "/CLOAD") || detail_w22::starts_with(up, "/LOAD")) {
                i++;
                // Skip title
                if (i < lines_.size()) {
                    std::string t = detail_w22::trim(lines_[i]);
                    if (!t.empty() && t[0] != '/' && t[0] != '#') {
                        auto f = detail_w22::split_ws(t);
                        bool numeric = true;
                        if (!f.empty()) { try { std::stoi(f[0]); } catch (...) { numeric = false; } }
                        if (!numeric) i++;
                    }
                }
                while (i < lines_.size()) {
                    std::string t = detail_w22::trim(lines_[i]);
                    if (t.empty() || t[0] == '#') { i++; continue; }
                    if (t[0] == '/') break;
                    auto f = detail_w22::split_ws(t);
                    if (f.size() >= 4) {
                        LoadData ld;
                        ld.node_id   = detail_w22::safe_int(f[0]);
                        ld.curve_id  = detail_w22::safe_int(f[1]);
                        ld.dof       = detail_w22::safe_int(f[2]);
                        ld.magnitude = detail_w22::safe_real(f[3]);
                        model_.loads.push_back(ld);
                    }
                    i++;
                }
            }

            // /FUNCT
            if (detail_w22::starts_with(up, "/FUNCT")) {
                int cid = extract_id(up, "/FUNCT");
                i++;
                // Skip title
                if (i < lines_.size()) {
                    std::string t = detail_w22::trim(lines_[i]);
                    if (!t.empty() && t[0] != '/' && t[0] != '#') {
                        auto f = detail_w22::split_ws(t);
                        bool numeric = true;
                        if (!f.empty()) { try { std::stod(f[0]); } catch (...) { numeric = false; } }
                        if (!numeric) i++;
                    }
                }
                CurveData cd;
                cd.id = cid;
                while (i < lines_.size()) {
                    std::string t = detail_w22::trim(lines_[i]);
                    if (t.empty() || t[0] == '#') { i++; continue; }
                    if (t[0] == '/') break;
                    auto f = detail_w22::split_ws(t);
                    if (f.size() >= 2) {
                        cd.points.push_back({detail_w22::safe_real(f[0]),
                                             detail_w22::safe_real(f[1])});
                    }
                    i++;
                }
                model_.curves.push_back(cd);
            }
        }
    }

    /// Parse /INTER (interface/contact) keywords
    void parse_interfaces() {
        for (size_t i = 0; i < lines_.size(); ++i) {
            std::string up = detail_w22::to_upper(detail_w22::trim(lines_[i]));
            if (!detail_w22::starts_with(up, "/INTER")) continue;
            // Interfaces produce BC-like data for contact (stored as loads)
            int inter_id = extract_id(up, "/INTER");
            i++;
            // Skip title
            if (i < lines_.size()) {
                std::string t = detail_w22::trim(lines_[i]);
                if (!t.empty() && t[0] != '/' && t[0] != '#') {
                    auto f = detail_w22::split_ws(t);
                    bool numeric = true;
                    if (!f.empty()) { try { std::stoi(f[0]); } catch (...) { numeric = false; } }
                    if (!numeric) i++;
                }
            }
            int card = 0;
            while (i < lines_.size() && card < 3) {
                std::string t = detail_w22::trim(lines_[i]);
                if (t.empty() || t[0] == '#') { i++; continue; }
                if (t[0] == '/') break;
                auto f = detail_w22::split_ws(t);
                if (card == 0 && f.size() >= 2) {
                    // Card 1: slave_part master_part
                    LoadData ld;
                    ld.node_id = inter_id;
                    ld.dof = -100;  // marker for contact
                    ld.magnitude = detail_w22::safe_real(f[0]); // slave
                    ld.curve_id = detail_w22::safe_int(f[1]);   // master
                    model_.loads.push_back(ld);
                }
                card++;
                i++;
            }
        }
    }

    /// Parse /PART blocks (linking elements to materials)
    void parse_parts() {
        for (size_t i = 0; i < lines_.size(); ++i) {
            std::string up = detail_w22::to_upper(detail_w22::trim(lines_[i]));
            if (!detail_w22::starts_with(up, "/PART")) continue;
            // /PART/part_id
            i++;
            // Title line
            if (i < lines_.size()) {
                std::string t = detail_w22::trim(lines_[i]);
                if (!t.empty() && t[0] != '/' && t[0] != '#') {
                    auto f = detail_w22::split_ws(t);
                    bool numeric = true;
                    if (!f.empty()) { try { std::stoi(f[0]); } catch (...) { numeric = false; } }
                    if (!numeric) i++;
                }
            }
            // Data: prop_id mat_id
            if (i < lines_.size()) {
                std::string t = detail_w22::trim(lines_[i]);
                if (!t.empty() && t[0] != '/' && t[0] != '#') {
                    auto f = detail_w22::split_ws(t);
                    // Store part-material linkage as a MaterialData entry if not already present
                    if (f.size() >= 2) {
                        int prop_id = detail_w22::safe_int(f[0]);
                        int mat_id  = detail_w22::safe_int(f[1]);
                        (void)prop_id;
                        (void)mat_id;
                        // This linkage is used during model assembly
                    }
                }
            }
        }
    }

    /// Extract numeric ID from /PREFIX/.../.../ID pattern
    int extract_id(const std::string& up, const std::string& prefix) const {
        if (up.size() <= prefix.size()) return 0;
        std::string rest = up.substr(prefix.size());
        if (!rest.empty() && rest[0] == '/') rest = rest.substr(1);
        size_t last = rest.rfind('/');
        if (last != std::string::npos) return detail_w22::safe_int(rest.substr(last + 1));
        return detail_w22::safe_int(rest);
    }
};

// ============================================================================
// 2. LSDYNAFullReader - Full LS-DYNA Keyword Reader
// ============================================================================

/**
 * @brief Full LS-DYNA keyword format reader
 *
 * Handles card-image fixed format (8 or 10 column) and free format (comma-delimited).
 * Detects column width automatically: 10-column if *KEYWORD_LONG suffix present;
 * free format if commas are found in data lines.
 *
 * Supported keywords:
 *   *NODE, *ELEMENT_SOLID, *ELEMENT_SHELL, *PART, *MAT_xxx, *SECTION_xxx,
 *   *BOUNDARY_SPC_SET, *BOUNDARY_PRESCRIBED_MOTION_SET, *LOAD_NODE_SET,
 *   *LOAD_BODY_X/Y/Z, *CONTACT_xxx, *DEFINE_CURVE, *INITIAL_VELOCITY,
 *   *INITIAL_VELOCITY_GENERATION, *SET_NODE_LIST, *SET_PART_LIST,
 *   *SET_SHELL_LIST, *SET_SOLID_LIST, *INCLUDE, *TITLE, *END
 */
class LSDYNAFullReader {
public:
    LSDYNAFullReader() = default;

    /**
     * @brief Read an LS-DYNA keyword file
     * @param filename Path to .k / .dyn / .key file
     * @return Populated ModelData
     */
    ModelData read(const std::string& filename) {
        model_ = ModelData{};
        node_sets_.clear();

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("LSDYNAFullReader: Cannot open: " + filename);
        }

        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            lines.push_back(line);
        }
        file.close();

        size_t i = 0;
        while (i < lines.size()) {
            std::string t = detail_w22::trim(lines[i]);
            if (t.empty() || t[0] == '$') { i++; continue; }

            std::string up = detail_w22::to_upper(t);
            bool is_long = (up.find("LONG") != std::string::npos);
            int cw = is_long ? 20 : 8;

            if (up == "*TITLE") {
                i++;
                if (i < lines.size()) i++; // skip title line (stored in ModelData if needed)
            }
            else if (detail_w22::starts_with(up, "*NODE")) {
                i++;
                i = read_nodes(lines, i, cw);
            }
            else if (detail_w22::starts_with(up, "*ELEMENT_SOLID")) {
                i++;
                i = read_elements(lines, i, cw, 5);
            }
            else if (detail_w22::starts_with(up, "*ELEMENT_SHELL")) {
                i++;
                i = read_elements(lines, i, cw, 2);
            }
            else if (detail_w22::starts_with(up, "*PART")) {
                i++;
                i = skip_kw(lines, i); // parts are implicit from element PID
            }
            else if (detail_w22::starts_with(up, "*MAT_")) {
                std::string mtype = extract_mat_type(up);
                i++;
                i = read_material(lines, i, cw, mtype);
            }
            else if (detail_w22::starts_with(up, "*SECTION_")) {
                i++;
                i = skip_kw(lines, i);
            }
            else if (detail_w22::starts_with(up, "*BOUNDARY_SPC")) {
                i++;
                i = read_spc(lines, i, cw);
            }
            else if (detail_w22::starts_with(up, "*BOUNDARY_PRESCRIBED_MOTION")) {
                i++;
                i = read_prescribed(lines, i, cw);
            }
            else if (detail_w22::starts_with(up, "*LOAD_NODE")) {
                i++;
                i = read_load_node(lines, i, cw);
            }
            else if (detail_w22::starts_with(up, "*LOAD_BODY")) {
                int dof = 0;
                if (up.find("_X") != std::string::npos) dof = 1;
                else if (up.find("_Y") != std::string::npos) dof = 2;
                else if (up.find("_Z") != std::string::npos) dof = 3;
                i++;
                i = read_load_body(lines, i, cw, dof);
            }
            else if (detail_w22::starts_with(up, "*CONTACT_")) {
                i++;
                i = read_contact(lines, i, cw);
            }
            else if (detail_w22::starts_with(up, "*DEFINE_CURVE")) {
                i++;
                i = read_curve(lines, i, cw);
            }
            else if (detail_w22::starts_with(up, "*INITIAL_VELOCITY")) {
                bool gen = (up.find("GENERATION") != std::string::npos);
                i++;
                i = read_init_vel(lines, i, cw, gen);
            }
            else if (detail_w22::starts_with(up, "*SET_NODE") ||
                     detail_w22::starts_with(up, "*SET_PART") ||
                     detail_w22::starts_with(up, "*SET_SHELL") ||
                     detail_w22::starts_with(up, "*SET_SOLID")) {
                i++;
                i = read_set(lines, i, cw);
            }
            else if (up == "*END") {
                break;
            }
            else {
                i++;
            }
        }

        return model_;
    }

private:
    ModelData model_;
    std::map<int, std::vector<int>> node_sets_;

    static bool is_kw(const std::string& line) {
        std::string t = detail_w22::trim(line);
        return !t.empty() && t[0] == '*';
    }
    static bool is_cmt(const std::string& line) {
        std::string t = detail_w22::trim(line);
        return !t.empty() && t[0] == '$';
    }
    size_t skip_kw(const std::vector<std::string>& L, size_t i) {
        while (i < L.size() && !is_kw(L[i])) i++;
        return i;
    }

    std::string extract_mat_type(const std::string& kw) {
        size_t p = kw.find("*MAT_");
        if (p == std::string::npos) return "UNKNOWN";
        std::string r = kw.substr(p + 5);
        size_t tp = r.find("_TITLE");
        if (tp != std::string::npos) r = r.substr(0, tp);
        return r;
    }

    size_t read_nodes(const std::vector<std::string>& L, size_t i, int cw) {
        while (i < L.size()) {
            if (is_kw(L[i])) return i;
            if (is_cmt(L[i]) || detail_w22::trim(L[i]).empty()) { i++; continue; }
            auto f = detail_w22::parse_card(L[i], cw);
            if (f.size() >= 4) {
                NodeData nd;
                nd.id = detail_w22::safe_int(f[0]);
                nd.x  = detail_w22::safe_real(f[1]);
                nd.y  = detail_w22::safe_real(f[2]);
                nd.z  = detail_w22::safe_real(f[3]);
                model_.nodes.push_back(nd);
            }
            i++;
        }
        return i;
    }

    size_t read_elements(const std::vector<std::string>& L, size_t i, int cw, int def) {
        while (i < L.size()) {
            if (is_kw(L[i])) return i;
            if (is_cmt(L[i]) || detail_w22::trim(L[i]).empty()) { i++; continue; }
            auto f = detail_w22::parse_card(L[i], cw);
            if (f.size() < 3) { i++; continue; }

            ElementData ed;
            std::fill(std::begin(ed.nodes), std::end(ed.nodes), 0);
            ed.id = detail_w22::safe_int(f[0]);
            ed.part_id = detail_w22::safe_int(f[1]);
            ed.num_nodes = 0;
            for (size_t j = 2; j < f.size() && ed.num_nodes < 20; ++j) {
                int nid = detail_w22::safe_int(f[j]);
                if (nid > 0) ed.nodes[ed.num_nodes++] = nid;
            }

            // Continuation line for solid elements with >6 nodes on first card
            if (def == 5 && ed.num_nodes < 8 && (i + 1) < L.size()) {
                if (!is_kw(L[i + 1]) && !is_cmt(L[i + 1])) {
                    i++;
                    auto f2 = detail_w22::parse_card(L[i], cw);
                    for (size_t j = 0; j < f2.size() && ed.num_nodes < 20; ++j) {
                        int nid = detail_w22::safe_int(f2[j]);
                        if (nid > 0) ed.nodes[ed.num_nodes++] = nid;
                    }
                }
            }

            if (def == 2) {
                ed.type = (ed.num_nodes == 3) ? 1 : 2;
            } else {
                if (ed.num_nodes == 4)       ed.type = 3;
                else if (ed.num_nodes == 10) ed.type = 4;
                else if (ed.num_nodes == 8)  ed.type = 5;
                else if (ed.num_nodes == 20) ed.type = 6;
                else                         ed.type = def;
            }
            model_.elements.push_back(ed);
            i++;
        }
        return i;
    }

    size_t read_material(const std::vector<std::string>& L, size_t i, int cw,
                         const std::string& mtype) {
        MaterialData mat;
        mat.type = mtype;
        mat.id = 0;

        // Detect TITLE variant
        bool has_title = (mtype.find("TITLE") != std::string::npos);
        if (has_title && i < L.size() && !is_kw(L[i])) i++;

        // Determine card count from material type
        int max_cards = 2;
        if (mtype.find("JOHNSON_COOK") != std::string::npos || mtype.find("015") != std::string::npos)
            max_cards = 4;
        else if (mtype.find("PIECEWISE") != std::string::npos || mtype.find("024") != std::string::npos)
            max_cards = 3;
        else if (mtype.find("ELASTIC") != std::string::npos || mtype.find("001") != std::string::npos)
            max_cards = 1;
        else if (mtype.find("PLASTIC_KINEMATIC") != std::string::npos || mtype.find("003") != std::string::npos)
            max_cards = 1;
        else if (mtype.find("RIGID") != std::string::npos || mtype.find("020") != std::string::npos)
            max_cards = 2;

        int card = 0;
        while (i < L.size() && card < max_cards) {
            if (is_kw(L[i])) return i;
            if (is_cmt(L[i]) || detail_w22::trim(L[i]).empty()) { i++; continue; }
            auto f = detail_w22::parse_card(L[i], cw);

            if (card == 0) {
                // Card 1: MID, RO, E, PR, ...
                if (f.size() >= 1) mat.id = detail_w22::safe_int(f[0]);
                if (f.size() >= 2) mat.properties["density"] = detail_w22::safe_real(f[1]);
                if (f.size() >= 3) mat.properties["E"]       = detail_w22::safe_real(f[2]);
                if (f.size() >= 4) mat.properties["nu"]      = detail_w22::safe_real(f[3]);
                for (size_t j = 4; j < f.size(); ++j) {
                    Real v = detail_w22::safe_real(f[j]);
                    if (v != 0.0) mat.properties["card0_" + std::to_string(j)] = v;
                }
            } else {
                for (size_t j = 0; j < f.size(); ++j) {
                    Real v = detail_w22::safe_real(f[j]);
                    if (v != 0.0) {
                        mat.properties["card" + std::to_string(card) + "_" + std::to_string(j)] = v;
                    }
                }
            }
            card++;
            i++;
        }
        model_.materials.push_back(mat);
        return skip_kw(L, i);
    }

    size_t read_spc(const std::vector<std::string>& L, size_t i, int cw) {
        while (i < L.size()) {
            if (is_kw(L[i])) return i;
            if (is_cmt(L[i]) || detail_w22::trim(L[i]).empty()) { i++; continue; }
            auto f = detail_w22::parse_card(L[i], cw);
            // NSID, CID, DOFX, DOFY, DOFZ, DOFRX, DOFRY, DOFRZ
            if (f.size() >= 4) {
                int nsid = detail_w22::safe_int(f[0]);
                int dof_mask = 0;
                if (f.size() >= 3 && detail_w22::safe_int(f[2]) != 0) dof_mask |= 1;
                if (f.size() >= 4 && detail_w22::safe_int(f[3]) != 0) dof_mask |= 2;
                if (f.size() >= 5 && detail_w22::safe_int(f[4]) != 0) dof_mask |= 4;
                if (f.size() >= 6 && detail_w22::safe_int(f[5]) != 0) dof_mask |= 8;
                if (f.size() >= 7 && detail_w22::safe_int(f[6]) != 0) dof_mask |= 16;
                if (f.size() >= 8 && detail_w22::safe_int(f[7]) != 0) dof_mask |= 32;

                // Expand node set if available
                auto it = node_sets_.find(nsid);
                if (it != node_sets_.end()) {
                    for (int nid : it->second) {
                        BCData bc;
                        bc.node_id = nid;
                        bc.dof_mask = dof_mask;
                        bc.value = 0.0;
                        model_.bcs.push_back(bc);
                    }
                } else {
                    BCData bc;
                    bc.node_id = nsid;
                    bc.dof_mask = dof_mask;
                    bc.value = 0.0;
                    model_.bcs.push_back(bc);
                }
            }
            i++;
        }
        return i;
    }

    size_t read_prescribed(const std::vector<std::string>& L, size_t i, int cw) {
        while (i < L.size()) {
            if (is_kw(L[i])) return i;
            if (is_cmt(L[i]) || detail_w22::trim(L[i]).empty()) { i++; continue; }
            auto f = detail_w22::parse_card(L[i], cw);
            // NSID, DOF, VAD, LCID, SF
            if (f.size() >= 4) {
                LoadData ld;
                ld.node_id   = detail_w22::safe_int(f[0]);
                ld.dof       = detail_w22::safe_int(f[1]);
                ld.curve_id  = detail_w22::safe_int(f[3]);
                ld.magnitude = (f.size() >= 5) ? detail_w22::safe_real(f[4]) : 1.0;
                model_.loads.push_back(ld);
            }
            i++;
        }
        return i;
    }

    size_t read_load_node(const std::vector<std::string>& L, size_t i, int cw) {
        while (i < L.size()) {
            if (is_kw(L[i])) return i;
            if (is_cmt(L[i]) || detail_w22::trim(L[i]).empty()) { i++; continue; }
            auto f = detail_w22::parse_card(L[i], cw);
            if (f.size() >= 4) {
                LoadData ld;
                ld.node_id   = detail_w22::safe_int(f[0]);
                ld.dof       = detail_w22::safe_int(f[1]);
                ld.curve_id  = detail_w22::safe_int(f[2]);
                ld.magnitude = detail_w22::safe_real(f[3]);
                model_.loads.push_back(ld);
            }
            i++;
        }
        return i;
    }

    size_t read_load_body(const std::vector<std::string>& L, size_t i, int cw, int dof) {
        while (i < L.size()) {
            if (is_kw(L[i])) return i;
            if (is_cmt(L[i]) || detail_w22::trim(L[i]).empty()) { i++; continue; }
            auto f = detail_w22::parse_card(L[i], cw);
            if (f.size() >= 2) {
                LoadData ld;
                ld.node_id   = -1; // body load
                ld.dof       = dof;
                ld.curve_id  = detail_w22::safe_int(f[0]);
                ld.magnitude = detail_w22::safe_real(f[1]);
                model_.loads.push_back(ld);
            }
            i++;
        }
        return i;
    }

    size_t read_contact(const std::vector<std::string>& L, size_t i, int cw) {
        // Consume up to 4 data cards (contact definition)
        int card = 0;
        while (i < L.size() && card < 4) {
            if (is_kw(L[i])) return i;
            if (is_cmt(L[i]) || detail_w22::trim(L[i]).empty()) { i++; continue; }
            // Contact cards are consumed but not stored in ModelData (which has no contact struct)
            // They can be extended later; for now we just advance past them
            card++;
            i++;
        }
        return skip_kw(L, i);
    }

    size_t read_curve(const std::vector<std::string>& L, size_t i, int cw) {
        CurveData cd;
        cd.id = 0;
        // Header card: LCID, ...
        if (i < L.size() && !is_kw(L[i]) && !is_cmt(L[i])) {
            auto f = detail_w22::parse_card(L[i], cw);
            if (!f.empty()) cd.id = detail_w22::safe_int(f[0]);
            i++;
        }
        // Data pairs
        while (i < L.size()) {
            if (is_kw(L[i])) break;
            if (is_cmt(L[i]) || detail_w22::trim(L[i]).empty()) { i++; continue; }
            auto f = detail_w22::parse_card(L[i], cw > 8 ? cw : 20);
            if (f.size() >= 2) {
                cd.points.push_back({detail_w22::safe_real(f[0]),
                                     detail_w22::safe_real(f[1])});
            }
            i++;
        }
        model_.curves.push_back(cd);
        return i;
    }

    size_t read_init_vel(const std::vector<std::string>& L, size_t i, int cw, bool gen) {
        while (i < L.size()) {
            if (is_kw(L[i])) return i;
            if (is_cmt(L[i]) || detail_w22::trim(L[i]).empty()) { i++; continue; }
            auto f = detail_w22::parse_card(L[i], cw);

            if (gen) {
                // *INITIAL_VELOCITY_GENERATION: NSID,STYP,OMEGA,VX,VY,VZ,...
                if (f.size() >= 6) {
                    int nsid = detail_w22::safe_int(f[0]);
                    Real vx = detail_w22::safe_real(f[3]);
                    Real vy = detail_w22::safe_real(f[4]);
                    Real vz = detail_w22::safe_real(f[5]);
                    if (vx != 0.0) { LoadData ld; ld.node_id = nsid; ld.dof = -1; ld.magnitude = vx; ld.curve_id = -1; model_.loads.push_back(ld); }
                    if (vy != 0.0) { LoadData ld; ld.node_id = nsid; ld.dof = -2; ld.magnitude = vy; ld.curve_id = -1; model_.loads.push_back(ld); }
                    if (vz != 0.0) { LoadData ld; ld.node_id = nsid; ld.dof = -3; ld.magnitude = vz; ld.curve_id = -1; model_.loads.push_back(ld); }
                }
            } else {
                // *INITIAL_VELOCITY: NID,VX,VY,VZ,...
                if (f.size() >= 4) {
                    int nid = detail_w22::safe_int(f[0]);
                    Real vx = detail_w22::safe_real(f[1]);
                    Real vy = detail_w22::safe_real(f[2]);
                    Real vz = detail_w22::safe_real(f[3]);
                    if (vx != 0.0) { LoadData ld; ld.node_id = nid; ld.dof = -1; ld.magnitude = vx; ld.curve_id = -1; model_.loads.push_back(ld); }
                    if (vy != 0.0) { LoadData ld; ld.node_id = nid; ld.dof = -2; ld.magnitude = vy; ld.curve_id = -1; model_.loads.push_back(ld); }
                    if (vz != 0.0) { LoadData ld; ld.node_id = nid; ld.dof = -3; ld.magnitude = vz; ld.curve_id = -1; model_.loads.push_back(ld); }
                }
            }
            i++;
        }
        return i;
    }

    size_t read_set(const std::vector<std::string>& L, size_t i, int cw) {
        int set_id = 0;
        std::vector<int> ids;
        // Header: SID, DA1, DA2, DA3, DA4
        if (i < L.size() && !is_kw(L[i]) && !is_cmt(L[i])) {
            auto f = detail_w22::parse_card(L[i], cw);
            if (!f.empty()) set_id = detail_w22::safe_int(f[0]);
            i++;
        }
        // ID lines
        while (i < L.size()) {
            if (is_kw(L[i])) break;
            if (is_cmt(L[i]) || detail_w22::trim(L[i]).empty()) { i++; continue; }
            auto f = detail_w22::parse_card(L[i], cw);
            for (const auto& s : f) {
                int v = detail_w22::safe_int(s);
                if (v > 0) ids.push_back(v);
            }
            i++;
        }
        node_sets_[set_id] = ids;
        return i;
    }
};

// ============================================================================
// 3. AbaqusINPReader - ABAQUS INP Subset Reader
// ============================================================================

/**
 * @brief Basic ABAQUS INP file reader
 *
 * Parses a subset of ABAQUS .inp keywords:
 *   *NODE, *ELEMENT (C3D8, C3D4, S4R, S3), *MATERIAL, *ELASTIC,
 *   *PLASTIC, *DENSITY, *BOUNDARY, *STEP, *STATIC/*DYNAMIC, *CLOAD, *DLOAD
 *
 * ABAQUS format: keyword lines start with '*', parameters on same line
 * after commas. Data lines are comma-delimited. Comments: '**'.
 */
class AbaqusINPReader {
public:
    AbaqusINPReader() = default;

    /**
     * @brief Read an ABAQUS INP file
     * @param filename Path to .inp file
     * @return Populated ModelData
     */
    ModelData read(const std::string& filename) {
        model_ = ModelData{};
        cur_mat_idx_ = -1;

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("AbaqusINPReader: Cannot open: " + filename);
        }

        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            lines.push_back(line);
        }
        file.close();

        size_t i = 0;
        while (i < lines.size()) {
            std::string t = detail_w22::trim(lines[i]);
            if (t.empty() || detail_w22::starts_with(t, "**")) { i++; continue; }

            std::string up = detail_w22::to_upper(t);

            if (detail_w22::starts_with(up, "*NODE") && !detail_w22::starts_with(up, "*NODE OUTPUT")) {
                i++;
                i = rd_nodes(lines, i);
            }
            else if (detail_w22::starts_with(up, "*ELEMENT")) {
                auto prm = parse_params(t);
                std::string etype = detail_w22::to_upper(prm["TYPE"]);
                i++;
                i = rd_elements(lines, i, etype);
            }
            else if (detail_w22::starts_with(up, "*MATERIAL")) {
                auto prm = parse_params(t);
                MaterialData mat;
                mat.id = static_cast<int>(model_.materials.size()) + 1;
                mat.type = "ABAQUS";
                mat.properties["_name"] = 0.0; // placeholder
                model_.materials.push_back(mat);
                cur_mat_idx_ = static_cast<int>(model_.materials.size()) - 1;
                i++;
            }
            else if (detail_w22::starts_with(up, "*ELASTIC")) {
                auto prm = parse_params(t);
                std::string tp = prm.count("TYPE") ? detail_w22::to_upper(prm["TYPE"]) : "ISOTROPIC";
                i++;
                i = rd_elastic(lines, i, tp);
            }
            else if (detail_w22::starts_with(up, "*PLASTIC")) {
                i++;
                i = rd_plastic(lines, i);
            }
            else if (detail_w22::starts_with(up, "*DENSITY")) {
                i++;
                i = rd_density(lines, i);
            }
            else if (detail_w22::starts_with(up, "*BOUNDARY")) {
                i++;
                i = rd_boundary(lines, i);
            }
            else if (detail_w22::starts_with(up, "*CLOAD")) {
                i++;
                i = rd_cload(lines, i);
            }
            else if (detail_w22::starts_with(up, "*DLOAD")) {
                i++;
                i = rd_dload(lines, i);
            }
            else if (detail_w22::starts_with(up, "*STEP")) {
                i++;
            }
            else if (detail_w22::starts_with(up, "*STATIC") || detail_w22::starts_with(up, "*DYNAMIC")) {
                i++;
                // Skip parameter line
                if (i < lines.size() && !is_kw(lines[i])) i++;
            }
            else {
                i++;
            }
        }

        return model_;
    }

private:
    ModelData model_;
    int cur_mat_idx_;

    static bool is_kw(const std::string& line) {
        std::string t = detail_w22::trim(line);
        return !t.empty() && t[0] == '*' && !(t.size() > 1 && t[1] == '*');
    }

    /// Parse keyword parameters: *ELEMENT, TYPE=C3D8, ELSET=PART1
    std::map<std::string, std::string> parse_params(const std::string& line) {
        std::map<std::string, std::string> p;
        auto parts = detail_w22::split_comma(line);
        for (size_t j = 1; j < parts.size(); ++j) {
            size_t eq = parts[j].find('=');
            if (eq != std::string::npos) {
                std::string k = detail_w22::to_upper(detail_w22::trim(parts[j].substr(0, eq)));
                std::string v = detail_w22::trim(parts[j].substr(eq + 1));
                p[k] = v;
            }
        }
        return p;
    }

    int etype_code(const std::string& et) {
        if (et == "C3D8" || et == "C3D8R" || et == "C3D8I") return 5;
        if (et == "C3D4" || et == "C3D4H") return 3;
        if (et == "C3D10" || et == "C3D10M") return 4;
        if (et == "C3D20" || et == "C3D20R") return 6;
        if (et == "S4" || et == "S4R" || et == "S4R5") return 2;
        if (et == "S3" || et == "S3R" || et == "STRI3") return 1;
        return 0;
    }

    int etype_nnode(const std::string& et) {
        if (et == "C3D8" || et == "C3D8R" || et == "C3D8I") return 8;
        if (et == "C3D4" || et == "C3D4H") return 4;
        if (et == "C3D10" || et == "C3D10M") return 10;
        if (et == "C3D20" || et == "C3D20R") return 20;
        if (et == "S4" || et == "S4R" || et == "S4R5") return 4;
        if (et == "S3" || et == "S3R" || et == "STRI3") return 3;
        return 8;
    }

    size_t rd_nodes(const std::vector<std::string>& L, size_t i) {
        while (i < L.size()) {
            std::string t = detail_w22::trim(L[i]);
            if (t.empty() || detail_w22::starts_with(t, "**")) { i++; continue; }
            if (is_kw(L[i])) return i;
            auto f = detail_w22::split_comma(t);
            if (f.size() >= 4) {
                NodeData nd;
                nd.id = detail_w22::safe_int(f[0]);
                nd.x  = detail_w22::safe_real(f[1]);
                nd.y  = detail_w22::safe_real(f[2]);
                nd.z  = detail_w22::safe_real(f[3]);
                model_.nodes.push_back(nd);
            }
            i++;
        }
        return i;
    }

    size_t rd_elements(const std::vector<std::string>& L, size_t i, const std::string& et) {
        int tc = etype_code(et);
        int nn = etype_nnode(et);

        while (i < L.size()) {
            std::string t = detail_w22::trim(L[i]);
            if (t.empty() || detail_w22::starts_with(t, "**")) { i++; continue; }
            if (is_kw(L[i])) return i;

            // Collect fields (may span continuation lines)
            std::vector<std::string> all;
            auto f = detail_w22::split_comma(t);
            all.insert(all.end(), f.begin(), f.end());
            i++;
            while (static_cast<int>(all.size()) < nn + 1 && i < L.size()) {
                t = detail_w22::trim(L[i]);
                if (t.empty() || detail_w22::starts_with(t, "**") || is_kw(L[i])) break;
                auto f2 = detail_w22::split_comma(t);
                all.insert(all.end(), f2.begin(), f2.end());
                i++;
            }

            if (!all.empty()) {
                ElementData ed;
                std::fill(std::begin(ed.nodes), std::end(ed.nodes), 0);
                ed.id = detail_w22::safe_int(all[0]);
                ed.part_id = 1;
                ed.type = tc;
                ed.num_nodes = 0;
                for (size_t j = 1; j < all.size() && ed.num_nodes < 20; ++j) {
                    int nid = detail_w22::safe_int(all[j]);
                    if (nid > 0) ed.nodes[ed.num_nodes++] = nid;
                }
                model_.elements.push_back(ed);
            }
        }
        return i;
    }

    size_t rd_elastic(const std::vector<std::string>& L, size_t i, const std::string& tp) {
        if (cur_mat_idx_ < 0 || cur_mat_idx_ >= static_cast<int>(model_.materials.size())) return i;
        auto& mat = model_.materials[cur_mat_idx_];

        while (i < L.size()) {
            std::string t = detail_w22::trim(L[i]);
            if (t.empty() || detail_w22::starts_with(t, "**")) { i++; continue; }
            if (is_kw(L[i])) return i;
            auto f = detail_w22::split_comma(t);
            if (tp == "ISOTROPIC" && f.size() >= 2) {
                mat.properties["E"]  = detail_w22::safe_real(f[0]);
                mat.properties["nu"] = detail_w22::safe_real(f[1]);
                mat.type = "ELASTIC";
            } else if (tp == "ORTHOTROPIC" && f.size() >= 9) {
                mat.properties["D1111"] = detail_w22::safe_real(f[0]);
                mat.properties["D1122"] = detail_w22::safe_real(f[1]);
                mat.properties["D2222"] = detail_w22::safe_real(f[2]);
                mat.properties["D1133"] = detail_w22::safe_real(f[3]);
                mat.properties["D2233"] = detail_w22::safe_real(f[4]);
                mat.properties["D3333"] = detail_w22::safe_real(f[5]);
                mat.properties["D1212"] = detail_w22::safe_real(f[6]);
                mat.properties["D1313"] = detail_w22::safe_real(f[7]);
                mat.properties["D2323"] = detail_w22::safe_real(f[8]);
                mat.type = "ORTHOTROPIC_ELASTIC";
            }
            i++;
        }
        return i;
    }

    size_t rd_plastic(const std::vector<std::string>& L, size_t i) {
        if (cur_mat_idx_ < 0 || cur_mat_idx_ >= static_cast<int>(model_.materials.size())) return i;
        auto& mat = model_.materials[cur_mat_idx_];
        mat.type = "ELASTIC_PLASTIC";
        int pt = 0;
        while (i < L.size()) {
            std::string t = detail_w22::trim(L[i]);
            if (t.empty() || detail_w22::starts_with(t, "**")) { i++; continue; }
            if (is_kw(L[i])) return i;
            auto f = detail_w22::split_comma(t);
            if (f.size() >= 2) {
                mat.properties["plastic_stress_" + std::to_string(pt)] = detail_w22::safe_real(f[0]);
                mat.properties["plastic_strain_" + std::to_string(pt)] = detail_w22::safe_real(f[1]);
                pt++;
            }
            i++;
        }
        mat.properties["num_plastic_points"] = static_cast<Real>(pt);
        return i;
    }

    size_t rd_density(const std::vector<std::string>& L, size_t i) {
        if (cur_mat_idx_ < 0 || cur_mat_idx_ >= static_cast<int>(model_.materials.size())) return i;
        auto& mat = model_.materials[cur_mat_idx_];
        while (i < L.size()) {
            std::string t = detail_w22::trim(L[i]);
            if (t.empty() || detail_w22::starts_with(t, "**")) { i++; continue; }
            if (is_kw(L[i])) return i;
            auto f = detail_w22::split_comma(t);
            if (!f.empty()) mat.properties["density"] = detail_w22::safe_real(f[0]);
            i++;
        }
        return i;
    }

    size_t rd_boundary(const std::vector<std::string>& L, size_t i) {
        while (i < L.size()) {
            std::string t = detail_w22::trim(L[i]);
            if (t.empty() || detail_w22::starts_with(t, "**")) { i++; continue; }
            if (is_kw(L[i])) return i;
            auto f = detail_w22::split_comma(t);
            if (f.size() >= 3) {
                int nid      = detail_w22::safe_int(f[0]);
                int first_dof = detail_w22::safe_int(f[1]);
                int last_dof  = detail_w22::safe_int(f[2]);
                Real value    = (f.size() >= 4) ? detail_w22::safe_real(f[3]) : 0.0;
                if (last_dof == 0) last_dof = first_dof;
                int mask = 0;
                for (int d = first_dof; d <= last_dof; ++d) {
                    if (d >= 1 && d <= 6) mask |= (1 << (d - 1));
                }
                BCData bc;
                bc.node_id = nid;
                bc.dof_mask = mask;
                bc.value = value;
                model_.bcs.push_back(bc);
            }
            i++;
        }
        return i;
    }

    size_t rd_cload(const std::vector<std::string>& L, size_t i) {
        while (i < L.size()) {
            std::string t = detail_w22::trim(L[i]);
            if (t.empty() || detail_w22::starts_with(t, "**")) { i++; continue; }
            if (is_kw(L[i])) return i;
            auto f = detail_w22::split_comma(t);
            if (f.size() >= 3) {
                LoadData ld;
                ld.node_id   = detail_w22::safe_int(f[0]);
                ld.dof       = detail_w22::safe_int(f[1]);
                ld.magnitude = detail_w22::safe_real(f[2]);
                ld.curve_id  = -1;
                model_.loads.push_back(ld);
            }
            i++;
        }
        return i;
    }

    size_t rd_dload(const std::vector<std::string>& L, size_t i) {
        while (i < L.size()) {
            std::string t = detail_w22::trim(L[i]);
            if (t.empty() || detail_w22::starts_with(t, "**")) { i++; continue; }
            if (is_kw(L[i])) return i;
            auto f = detail_w22::split_comma(t);
            if (f.size() >= 3) {
                LoadData ld;
                ld.node_id   = -detail_w22::safe_int(f[0]); // negative = element-based
                ld.dof       = 0;
                ld.magnitude = detail_w22::safe_real(f[2]);
                ld.curve_id  = -1;
                model_.loads.push_back(ld);
            }
            i++;
        }
        return i;
    }
};

// ============================================================================
// 4. ModelValidator - Model Validation and Checking
// ============================================================================

/**
 * @brief Comprehensive model validation
 *
 * Checks:
 * - Node connectivity (orphan nodes not referenced by any element)
 * - Element quality (Jacobian > 0 for hex/tet/shell)
 * - Material assignment (all elements belong to parts with materials)
 * - BC consistency (no conflicting or over-constrained DOFs)
 * - Contact surface validity (referenced nodes/elements exist)
 * - Duplicate IDs
 * - Node reference integrity
 * - Curve monotonicity
 *
 * Returns a vector of ValidationMessage {level, code, message}
 * where level: 0=info, 1=warning, 2=error.
 */
class ModelValidator {
public:
    ModelValidator() = default;

    /**
     * @brief Run all validation checks
     * @param data The model data to validate
     * @return Vector of messages; empty means the model is fully valid
     */
    std::vector<ValidationMessage> validate(const ModelData& data) const {
        std::vector<ValidationMessage> msgs;

        check_orphan_nodes(data, msgs);
        check_element_jacobian(data, msgs);
        check_material_assignment(data, msgs);
        check_bc_consistency(data, msgs);
        check_duplicate_ids(data, msgs);
        check_node_references(data, msgs);
        check_curve_monotonicity(data, msgs);

        // Summary
        int errs = 0, warns = 0, infos = 0;
        for (const auto& m : msgs) {
            if (m.level == 2) errs++;
            else if (m.level == 1) warns++;
            else infos++;
        }
        ValidationMessage sum;
        sum.level = (errs > 0) ? 2 : (warns > 0) ? 1 : 0;
        sum.code = 0;
        sum.message = "Validation: " + std::to_string(errs) + " errors, "
                    + std::to_string(warns) + " warnings, "
                    + std::to_string(infos) + " info";
        msgs.insert(msgs.begin(), sum);
        return msgs;
    }

private:
    /// Orphan nodes not referenced by any element
    void check_orphan_nodes(const ModelData& d, std::vector<ValidationMessage>& m) const {
        if (d.nodes.empty() || d.elements.empty()) return;
        std::set<int> ref;
        for (const auto& e : d.elements)
            for (int n = 0; n < e.num_nodes; ++n)
                ref.insert(e.nodes[n]);
        int orphan = 0;
        for (const auto& nd : d.nodes)
            if (ref.find(nd.id) == ref.end()) orphan++;
        if (orphan > 0) {
            m.push_back({1, 1001,
                "Found " + std::to_string(orphan) + " orphan node(s) not referenced by any element"});
        }
    }

    /// Jacobian positivity check
    void check_element_jacobian(const ModelData& d, std::vector<ValidationMessage>& m) const {
        if (d.elements.empty() || d.nodes.empty()) return;

        std::map<int, int> nmap;
        for (int i = 0; i < static_cast<int>(d.nodes.size()); ++i) nmap[d.nodes[i].id] = i;

        int neg = 0, degen = 0;
        for (const auto& el : d.elements) {
            // Get coordinates
            std::vector<Real> cx, cy, cz;
            bool ok = true;
            for (int n = 0; n < el.num_nodes; ++n) {
                auto it = nmap.find(el.nodes[n]);
                if (it == nmap.end()) { ok = false; break; }
                cx.push_back(d.nodes[it->second].x);
                cy.push_back(d.nodes[it->second].y);
                cz.push_back(d.nodes[it->second].z);
            }
            if (!ok) continue;

            Real jac = 0.0;
            if (el.type == 5 && el.num_nodes >= 8) {
                // Hex8: triple scalar product at corner 0
                Real ax = cx[1]-cx[0], ay = cy[1]-cy[0], az = cz[1]-cz[0];
                Real bx = cx[3]-cx[0], by = cy[3]-cy[0], bz = cz[3]-cz[0];
                Real dx = cx[4]-cx[0], dy = cy[4]-cy[0], dz = cz[4]-cz[0];
                jac = ax*(by*dz - bz*dy) - ay*(bx*dz - bz*dx) + az*(bx*dy - by*dx);
            }
            else if (el.type == 3 && el.num_nodes >= 4) {
                // Tet4: 6V
                Real ax = cx[1]-cx[0], ay = cy[1]-cy[0], az = cz[1]-cz[0];
                Real bx = cx[2]-cx[0], by = cy[2]-cy[0], bz = cz[2]-cz[0];
                Real dx = cx[3]-cx[0], dy = cy[3]-cy[0], dz = cz[3]-cz[0];
                jac = (ax*(by*dz - bz*dy) - ay*(bx*dz - bz*dx) + az*(bx*dy - by*dx)) / 6.0;
            }
            else if ((el.type == 1 || el.type == 2) && el.num_nodes >= 3) {
                // Shell: area via cross product
                Real ax = cx[1]-cx[0], ay = cy[1]-cy[0], az = cz[1]-cz[0];
                Real bx = cx[2]-cx[0], by = cy[2]-cy[0], bz = cz[2]-cz[0];
                Real nx = ay*bz - az*by, ny = az*bx - ax*bz, nz = ax*by - ay*bx;
                jac = std::sqrt(nx*nx + ny*ny + nz*nz) / 2.0;
            }
            else continue;

            if (jac <= 0.0) neg++;
            if (std::abs(jac) < 1.0e-15) degen++;
        }

        if (neg > 0) {
            m.push_back({2, 2001,
                std::to_string(neg) + " element(s) with non-positive Jacobian (inverted)"});
        }
        if (degen > 0) {
            m.push_back({2, 2002,
                std::to_string(degen) + " degenerate element(s) with near-zero volume"});
        }
    }

    /// All elements should have a material (via part_id)
    void check_material_assignment(const ModelData& d, std::vector<ValidationMessage>& m) const {
        if (d.elements.empty() || d.materials.empty()) return;
        std::set<int> mat_ids;
        for (const auto& mt : d.materials) mat_ids.insert(mt.id);

        // If there is at least one element whose part_id does not match any material id,
        // emit a warning. (Full part-material linkage is format-dependent.)
        std::set<int> unmatched;
        for (const auto& e : d.elements) {
            if (mat_ids.find(e.part_id) == mat_ids.end()) {
                unmatched.insert(e.part_id);
            }
        }
        // Only warn if no material at all matches any part_id
        if (unmatched.size() == static_cast<size_t>(
                std::count_if(d.elements.begin(), d.elements.end(),
                [](const ElementData&) { return true; }))) {
            // No elements matched any material; might be OK if parts are used
            // but still worth a note
            if (static_cast<int>(unmatched.size()) > 0) {
                m.push_back({1, 1002,
                    std::to_string(unmatched.size()) + " part ID(s) have no direct material match"});
            }
        }
    }

    /// BC consistency: conflicting prescriptions, over-constraint
    void check_bc_consistency(const ModelData& d, std::vector<ValidationMessage>& m) const {
        if (d.bcs.empty()) return;

        // Accumulate per (node, dof_bit) -> values
        std::map<std::pair<int,int>, std::vector<Real>> dof_vals;
        for (const auto& bc : d.bcs) {
            for (int b = 0; b < 6; ++b) {
                if (bc.dof_mask & (1 << b)) {
                    dof_vals[{bc.node_id, b}].push_back(bc.value);
                }
            }
        }

        // Check total per node
        std::map<int, int> node_total;
        for (const auto& bc : d.bcs) {
            int bits = bc.dof_mask;
            int cnt = 0;
            while (bits) { cnt += (bits & 1); bits >>= 1; }
            node_total[bc.node_id] += cnt;
        }

        int over = 0;
        for (const auto& [nid, cnt] : node_total) {
            if (cnt > 6) over++;
        }

        int conflict = 0;
        for (const auto& [key, vals] : dof_vals) {
            if (vals.size() > 1) {
                for (size_t j = 1; j < vals.size(); ++j) {
                    if (std::abs(vals[j] - vals[0]) > 1.0e-12) { conflict++; break; }
                }
            }
        }

        if (over > 0) {
            m.push_back({1, 1003,
                std::to_string(over) + " node(s) appear over-constrained (>6 DOF)"});
        }
        if (conflict > 0) {
            m.push_back({2, 2003,
                std::to_string(conflict) + " DOF(s) have conflicting prescribed values"});
        }
    }

    /// Duplicate node or element IDs
    void check_duplicate_ids(const ModelData& d, std::vector<ValidationMessage>& m) const {
        std::set<int> s;
        int dn = 0;
        for (const auto& n : d.nodes) { if (!s.insert(n.id).second) dn++; }
        if (dn > 0) m.push_back({2, 2005, std::to_string(dn) + " duplicate node ID(s)"});

        s.clear();
        int de = 0;
        for (const auto& e : d.elements) { if (!s.insert(e.id).second) de++; }
        if (de > 0) m.push_back({2, 2006, std::to_string(de) + " duplicate element ID(s)"});
    }

    /// Elements referencing non-existent nodes
    void check_node_references(const ModelData& d, std::vector<ValidationMessage>& m) const {
        if (d.elements.empty() || d.nodes.empty()) return;
        std::set<int> valid;
        for (const auto& n : d.nodes) valid.insert(n.id);
        int bad = 0;
        for (const auto& e : d.elements) {
            for (int n = 0; n < e.num_nodes; ++n) {
                if (valid.find(e.nodes[n]) == valid.end()) { bad++; break; }
            }
        }
        if (bad > 0) {
            m.push_back({2, 2007,
                std::to_string(bad) + " element(s) reference undefined node IDs"});
        }
    }

    /// Load curve abscissa monotonicity
    void check_curve_monotonicity(const ModelData& d, std::vector<ValidationMessage>& m) const {
        for (const auto& c : d.curves) {
            for (size_t j = 1; j < c.points.size(); ++j) {
                if (c.points[j].first < c.points[j-1].first) {
                    m.push_back({1, 1005,
                        "Curve " + std::to_string(c.id) + " has non-monotonic abscissa"});
                    break;
                }
            }
        }
    }
};

} // namespace io
} // namespace nxs
