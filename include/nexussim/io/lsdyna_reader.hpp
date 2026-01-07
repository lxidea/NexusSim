#pragma once

/**
 * @file lsdyna_reader.hpp
 * @brief LS-DYNA keyword format reader
 *
 * Parses LS-DYNA keyword (.k, .dyn, .key) format files and creates NexusSim mesh objects.
 *
 * Supported keywords:
 * - *NODE - Node definitions
 * - *ELEMENT_SOLID - Solid elements (Hex8, Tet4, Tet10, Wedge6)
 * - *ELEMENT_SHELL - Shell elements (Shell3, Shell4)
 * - *ELEMENT_BEAM - Beam elements
 * - *PART - Part definitions
 * - *MAT_ELASTIC - Elastic material
 * - *MAT_PIECEWISE_LINEAR_PLASTICITY - Plasticity with curve
 * - *MAT_JOHNSON_COOK - Johnson-Cook plasticity
 * - *SET_NODE_LIST - Node sets
 * - *SET_PART_LIST - Part sets
 * - *BOUNDARY_SPC_SET - Single point constraints
 * - *LOAD_NODE_SET - Nodal loads
 *
 * File format reference: LS-DYNA Keyword User's Manual
 */

#include <nexussim/core/core.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/physics/material.hpp>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <memory>
#include <functional>
#include <algorithm>
#include <cctype>

namespace nxs {
namespace io {

// Forward declarations
struct LSDynaNode;
struct LSDynaElement;
struct LSDynaMaterial;
struct LSDynaPart;
struct LSDynaNodeSet;
struct LSDynaSPC;
struct LSDynaLoad;

/**
 * @brief LS-DYNA node data
 */
struct LSDynaNode {
    Index nid;      // Node ID
    Real x, y, z;   // Coordinates
    Int tc;         // Translational constraint code
    Int rc;         // Rotational constraint code
};

/**
 * @brief LS-DYNA element data
 */
struct LSDynaElement {
    Index eid;              // Element ID
    Index pid;              // Part ID
    ElementType type;       // NexusSim element type
    std::vector<Index> nodes;  // Node IDs
};

/**
 * @brief LS-DYNA material data
 */
struct LSDynaMaterial {
    Index mid;              // Material ID
    std::string type;       // Material type string (e.g., "ELASTIC")
    std::string title;      // Optional title

    // Common properties
    Real ro;                // Density
    Real e;                 // Young's modulus
    Real pr;                // Poisson's ratio

    // Johnson-Cook parameters
    Real a_jc;              // Yield stress A
    Real b_jc;              // Hardening B
    Real n_jc;              // Hardening exponent n
    Real c_jc;              // Strain rate coefficient C
    Real m_jc;              // Thermal softening m
    Real tm;                // Melting temperature
    Real tr;                // Room temperature
    Real epso;              // Reference strain rate

    // Piecewise linear plasticity
    Real sigy;              // Initial yield stress
    Real etan;              // Tangent modulus
    Real fail;              // Failure strain
    Index lcss;             // Load curve ID for stress-strain

    LSDynaMaterial()
        : mid(0), ro(0), e(0), pr(0)
        , a_jc(0), b_jc(0), n_jc(0), c_jc(0), m_jc(0)
        , tm(0), tr(0), epso(1.0)
        , sigy(0), etan(0), fail(0), lcss(0)
    {}
};

/**
 * @brief LS-DYNA part data
 */
struct LSDynaPart {
    Index pid;          // Part ID
    Index secid;        // Section ID
    Index mid;          // Material ID
    std::string title;  // Part title
};

/**
 * @brief LS-DYNA node set data
 */
struct LSDynaNodeSet {
    Index sid;                  // Set ID
    std::string title;          // Set title
    std::vector<Index> nodes;   // Node IDs in set
};

/**
 * @brief LS-DYNA single point constraint
 */
struct LSDynaSPC {
    Index nsid;         // Node set ID
    Int cid;            // Coordinate system ID
    Int dofx, dofy, dofz;       // DOF constraints (0=free, 1=fixed)
    Int dofrx, dofry, dofrz;    // Rotational DOF constraints

    LSDynaSPC()
        : nsid(0), cid(0)
        , dofx(0), dofy(0), dofz(0)
        , dofrx(0), dofry(0), dofrz(0)
    {}
};

/**
 * @brief LS-DYNA nodal load
 */
struct LSDynaLoad {
    Index nsid;         // Node set ID
    Int dof;            // Degree of freedom (1=x, 2=y, 3=z)
    Index lcid;         // Load curve ID
    Real sf;            // Scale factor

    LSDynaLoad() : nsid(0), dof(0), lcid(0), sf(1.0) {}
};

/**
 * @brief LS-DYNA load curve
 */
struct LSDynaLoadCurve {
    Index lcid;                         // Load curve ID
    std::vector<std::pair<Real, Real>> points;  // (time, value) pairs
};

/**
 * @brief LS-DYNA keyword format reader
 *
 * Usage:
 * ```cpp
 * LSDynaReader reader;
 * reader.read("model.k");
 *
 * auto mesh = reader.create_mesh();
 * auto materials = reader.materials();
 * auto bcs = reader.boundary_conditions();
 * ```
 */
class LSDynaReader {
public:
    LSDynaReader() = default;

    /**
     * @brief Read LS-DYNA keyword file
     * @param filename Path to .k, .dyn, or .key file
     * @return true on success
     */
    bool read(const std::string& filename);

    /**
     * @brief Create NexusSim mesh from parsed data
     * @return Shared pointer to mesh
     */
    std::shared_ptr<Mesh> create_mesh() const;

    // Accessors for parsed data
    const std::vector<LSDynaNode>& nodes() const { return nodes_; }
    const std::vector<LSDynaElement>& elements() const { return elements_; }
    const std::map<Index, LSDynaMaterial>& materials() const { return materials_; }
    const std::map<Index, LSDynaPart>& parts() const { return parts_; }
    const std::map<Index, LSDynaNodeSet>& node_sets() const { return node_sets_; }
    const std::vector<LSDynaSPC>& spcs() const { return spcs_; }
    const std::vector<LSDynaLoad>& loads() const { return loads_; }
    const std::map<Index, LSDynaLoadCurve>& load_curves() const { return load_curves_; }

    /**
     * @brief Get node set by ID
     */
    std::vector<Index> get_node_set(Index sid) const;

    /**
     * @brief Get elements by part ID
     */
    std::vector<Index> get_elements_by_part(Index pid) const;

    /**
     * @brief Get material properties for a part
     */
    physics::MaterialProperties get_material(Index pid) const;

    /**
     * @brief Get model title
     */
    const std::string& title() const { return title_; }

    /**
     * @brief Print summary of parsed model
     */
    void print_summary(std::ostream& os = std::cout) const;

private:
    // Keyword handler type
    using KeywordHandler = std::function<void(std::istream&)>;

    // Initialize keyword handlers
    void init_keyword_handlers();

    // Process keyword with data lines
    void process_keyword(const std::string& keyword, const std::vector<std::string>& data_lines);

    // Parsing functions
    void parse_node(std::istream& is);
    void parse_element_solid(std::istream& is);
    void parse_element_shell(std::istream& is);
    void parse_element_beam(std::istream& is);
    void parse_part(std::istream& is);
    void parse_mat_elastic(std::istream& is);
    void parse_mat_plastic(std::istream& is);
    void parse_mat_johnson_cook(std::istream& is);
    void parse_set_node_list(std::istream& is);
    void parse_boundary_spc_set(std::istream& is);
    void parse_load_node_set(std::istream& is);
    void parse_define_curve(std::istream& is);
    void parse_title(std::istream& is);

    // Helper functions
    std::string trim(const std::string& s) const;
    std::string to_upper(const std::string& s) const;
    std::vector<std::string> split_fields(const std::string& line, int field_width = 10) const;
    bool is_keyword(const std::string& line) const;
    std::string extract_keyword(const std::string& line) const;
    bool is_comment(const std::string& line) const;
    Real parse_real(const std::string& s, Real default_val = 0.0) const;
    Index parse_index(const std::string& s, Index default_val = 0) const;
    Int parse_int(const std::string& s, Int default_val = 0) const;
    ElementType determine_solid_type(int num_nodes) const;
    ElementType determine_shell_type(int num_nodes) const;

    // Parsed data
    std::string title_;

    std::vector<LSDynaNode> nodes_;
    std::vector<LSDynaElement> elements_;
    std::map<Index, LSDynaMaterial> materials_;
    std::map<Index, LSDynaPart> parts_;
    std::map<Index, LSDynaNodeSet> node_sets_;
    std::vector<LSDynaSPC> spcs_;
    std::vector<LSDynaLoad> loads_;
    std::map<Index, LSDynaLoadCurve> load_curves_;

    // Node ID to index mapping
    std::map<Index, Index> node_id_to_index_;

    // Keyword dispatch table
    std::unordered_map<std::string, KeywordHandler> keyword_handlers_;
};

// ============================================================================
// Implementation
// ============================================================================

inline void LSDynaReader::init_keyword_handlers() {
    keyword_handlers_["*NODE"] = [this](std::istream& is) { parse_node(is); };
    keyword_handlers_["*ELEMENT_SOLID"] = [this](std::istream& is) { parse_element_solid(is); };
    keyword_handlers_["*ELEMENT_SHELL"] = [this](std::istream& is) { parse_element_shell(is); };
    keyword_handlers_["*ELEMENT_BEAM"] = [this](std::istream& is) { parse_element_beam(is); };
    keyword_handlers_["*PART"] = [this](std::istream& is) { parse_part(is); };
    keyword_handlers_["*MAT_ELASTIC"] = [this](std::istream& is) { parse_mat_elastic(is); };
    keyword_handlers_["*MAT_PIECEWISE_LINEAR_PLASTICITY"] = [this](std::istream& is) { parse_mat_plastic(is); };
    keyword_handlers_["*MAT_JOHNSON_COOK"] = [this](std::istream& is) { parse_mat_johnson_cook(is); };
    keyword_handlers_["*SET_NODE_LIST"] = [this](std::istream& is) { parse_set_node_list(is); };
    keyword_handlers_["*BOUNDARY_SPC_SET"] = [this](std::istream& is) { parse_boundary_spc_set(is); };
    keyword_handlers_["*LOAD_NODE_SET"] = [this](std::istream& is) { parse_load_node_set(is); };
    keyword_handlers_["*DEFINE_CURVE"] = [this](std::istream& is) { parse_define_curve(is); };
    keyword_handlers_["*TITLE"] = [this](std::istream& is) { parse_title(is); };
}

inline bool LSDynaReader::read(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        NXS_LOG_ERROR("Failed to open file: {}", filename);
        return false;
    }

    NXS_LOG_INFO("Reading LS-DYNA keyword file: {}", filename);

    // Initialize keyword handlers
    init_keyword_handlers();

    // Clear previous data
    nodes_.clear();
    elements_.clear();
    materials_.clear();
    parts_.clear();
    node_sets_.clear();
    spcs_.clear();
    loads_.clear();
    load_curves_.clear();
    node_id_to_index_.clear();

    // Read all lines into a vector for easier handling
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    // Process lines
    size_t i = 0;
    while (i < lines.size()) {
        std::string trimmed = trim(lines[i]);

        // Skip empty lines and comments
        if (trimmed.empty() || is_comment(trimmed)) {
            ++i;
            continue;
        }

        // Check for keyword
        if (is_keyword(trimmed)) {
            std::string keyword = extract_keyword(trimmed);
            std::string upper_keyword = to_upper(keyword);

            if (upper_keyword == "*END") {
                break;
            }

            ++i;  // Move past keyword line

            // Collect data lines until next keyword
            std::vector<std::string> data_lines;
            while (i < lines.size()) {
                std::string data_trimmed = trim(lines[i]);
                if (data_trimmed.empty() || is_comment(data_trimmed)) {
                    ++i;
                    continue;
                }
                if (is_keyword(data_trimmed)) {
                    break;  // Don't increment - let outer loop handle
                }
                data_lines.push_back(lines[i]);
                ++i;
            }

            // Process the keyword with collected data
            process_keyword(upper_keyword, data_lines);
        } else {
            ++i;
        }
    }

    // Build node ID to index mapping
    for (size_t i = 0; i < nodes_.size(); ++i) {
        node_id_to_index_[nodes_[i].nid] = i;
    }

    NXS_LOG_INFO("Parsed {} nodes, {} elements, {} materials, {} parts",
                 nodes_.size(), elements_.size(), materials_.size(), parts_.size());

    return true;
}

inline void LSDynaReader::process_keyword(const std::string& keyword,
                                          const std::vector<std::string>& data_lines) {
    if (keyword == "*NODE") {
        for (const auto& line : data_lines) {
            auto fields = split_fields(line, 8);
            if (fields.size() < 4) continue;

            LSDynaNode node;
            node.nid = parse_index(fields[0]);
            node.x = parse_real(fields[1]);
            node.y = parse_real(fields[2]);
            node.z = parse_real(fields[3]);
            node.tc = fields.size() > 4 ? parse_int(fields[4]) : 0;
            node.rc = fields.size() > 5 ? parse_int(fields[5]) : 0;
            nodes_.push_back(node);
        }
    }
    else if (keyword == "*ELEMENT_SOLID") {
        for (const auto& line : data_lines) {
            auto fields = split_fields(line, 8);
            if (fields.size() < 3) continue;

            LSDynaElement elem;
            elem.eid = parse_index(fields[0]);
            elem.pid = parse_index(fields[1]);

            for (size_t i = 2; i < fields.size(); ++i) {
                Index nid = parse_index(fields[i]);
                if (nid > 0) {
                    elem.nodes.push_back(nid);
                }
            }
            elem.type = determine_solid_type(static_cast<int>(elem.nodes.size()));
            elements_.push_back(elem);
        }
    }
    else if (keyword == "*ELEMENT_SHELL") {
        for (const auto& line : data_lines) {
            auto fields = split_fields(line, 8);
            if (fields.size() < 4) continue;

            LSDynaElement elem;
            elem.eid = parse_index(fields[0]);
            elem.pid = parse_index(fields[1]);

            for (size_t i = 2; i < fields.size() && i < 6; ++i) {
                Index nid = parse_index(fields[i]);
                if (nid > 0) {
                    elem.nodes.push_back(nid);
                }
            }
            elem.type = determine_shell_type(static_cast<int>(elem.nodes.size()));
            elements_.push_back(elem);
        }
    }
    else if (keyword == "*ELEMENT_BEAM") {
        for (const auto& line : data_lines) {
            auto fields = split_fields(line, 8);
            if (fields.size() < 5) continue;

            LSDynaElement elem;
            elem.eid = parse_index(fields[0]);
            elem.pid = parse_index(fields[1]);
            elem.nodes.push_back(parse_index(fields[2]));
            elem.nodes.push_back(parse_index(fields[3]));
            elem.type = ElementType::Beam2;
            elements_.push_back(elem);
        }
    }
    else if (keyword == "*PART") {
        // First line is title, second line is data
        std::string title;
        if (data_lines.size() >= 1) {
            title = trim(data_lines[0]);
        }
        if (data_lines.size() >= 2) {
            auto fields = split_fields(data_lines[1], 10);
            if (fields.size() >= 3) {
                LSDynaPart part;
                part.pid = parse_index(fields[0]);
                part.secid = parse_index(fields[1]);
                part.mid = parse_index(fields[2]);
                part.title = title;
                parts_[part.pid] = part;
            }
        }
    }
    else if (keyword == "*MAT_ELASTIC") {
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 4) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "ELASTIC";
                mat.ro = parse_real(fields[1]);
                mat.e = parse_real(fields[2]);
                mat.pr = parse_real(fields[3]);
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_PIECEWISE_LINEAR_PLASTICITY") {
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 5) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields[0]);
                mat.type = "PIECEWISE_LINEAR_PLASTICITY";
                mat.ro = parse_real(fields[1]);
                mat.e = parse_real(fields[2]);
                mat.pr = parse_real(fields[3]);
                mat.sigy = parse_real(fields[4]);
                mat.etan = fields.size() > 5 ? parse_real(fields[5]) : 0.0;
                mat.fail = fields.size() > 6 ? parse_real(fields[6]) : 0.0;
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*MAT_JOHNSON_COOK") {
        if (data_lines.size() >= 2) {
            auto fields1 = split_fields(data_lines[0], 10);
            auto fields2 = split_fields(data_lines[1], 10);

            if (fields1.size() >= 5) {
                LSDynaMaterial mat;
                mat.mid = parse_index(fields1[0]);
                mat.type = "JOHNSON_COOK";
                mat.ro = parse_real(fields1[1]);
                Real g = parse_real(fields1[2]);
                mat.e = parse_real(fields1[3]);
                mat.pr = parse_real(fields1[4]);
                if (mat.e == 0.0 && g > 0.0 && mat.pr > 0.0) {
                    mat.e = 2.0 * g * (1.0 + mat.pr);
                }

                if (fields2.size() >= 5) {
                    mat.a_jc = parse_real(fields2[0]);
                    mat.b_jc = parse_real(fields2[1]);
                    mat.n_jc = parse_real(fields2[2]);
                    mat.c_jc = parse_real(fields2[3]);
                    mat.m_jc = parse_real(fields2[4]);
                    mat.tm = fields2.size() > 5 ? parse_real(fields2[5]) : 0.0;
                    mat.tr = fields2.size() > 6 ? parse_real(fields2[6]) : 298.0;
                    mat.epso = fields2.size() > 7 ? parse_real(fields2[7]) : 1.0;
                }
                materials_[mat.mid] = mat;
            }
        }
    }
    else if (keyword == "*SET_NODE_LIST") {
        LSDynaNodeSet node_set;
        node_set.sid = 0;

        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 1) {
                node_set.sid = parse_index(fields[0]);
            }

            // Rest are node IDs
            for (size_t li = 1; li < data_lines.size(); ++li) {
                auto node_fields = split_fields(data_lines[li], 10);
                for (const auto& f : node_fields) {
                    Index nid = parse_index(f);
                    if (nid > 0) {
                        node_set.nodes.push_back(nid);
                    }
                }
            }

            if (node_set.sid > 0) {
                node_sets_[node_set.sid] = node_set;
            }
        }
    }
    else if (keyword == "*BOUNDARY_SPC_SET") {
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 5) {
                LSDynaSPC spc;
                spc.nsid = parse_index(fields[0]);
                spc.cid = parse_int(fields[1]);
                spc.dofx = parse_int(fields[2]);
                spc.dofy = parse_int(fields[3]);
                spc.dofz = parse_int(fields[4]);
                spc.dofrx = fields.size() > 5 ? parse_int(fields[5]) : 0;
                spc.dofry = fields.size() > 6 ? parse_int(fields[6]) : 0;
                spc.dofrz = fields.size() > 7 ? parse_int(fields[7]) : 0;
                spcs_.push_back(spc);
            }
        }
    }
    else if (keyword == "*LOAD_NODE_SET") {
        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 3) {
                LSDynaLoad load;
                load.nsid = parse_index(fields[0]);
                load.dof = parse_int(fields[1]);
                load.lcid = parse_index(fields[2]);
                load.sf = fields.size() > 3 ? parse_real(fields[3]) : 1.0;
                loads_.push_back(load);
            }
        }
    }
    else if (keyword == "*DEFINE_CURVE") {
        LSDynaLoadCurve curve;
        curve.lcid = 0;

        if (!data_lines.empty()) {
            auto fields = split_fields(data_lines[0], 10);
            if (fields.size() >= 1) {
                curve.lcid = parse_index(fields[0]);
            }

            // Rest are (abscissa, ordinate) pairs
            for (size_t li = 1; li < data_lines.size(); ++li) {
                auto curve_fields = split_fields(data_lines[li], 20);
                if (curve_fields.size() >= 2) {
                    Real a = parse_real(curve_fields[0]);
                    Real o = parse_real(curve_fields[1]);
                    curve.points.emplace_back(a, o);
                }
            }

            if (curve.lcid > 0) {
                load_curves_[curve.lcid] = curve;
            }
        }
    }
    else if (keyword == "*TITLE") {
        if (!data_lines.empty()) {
            title_ = trim(data_lines[0]);
        }
    }
    // Other keywords are silently ignored
}

inline void LSDynaReader::parse_node(std::istream& is) {
    std::string line;
    while (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || is_comment(trimmed)) continue;
        if (is_keyword(trimmed)) {
            // Put line back by seeking (not possible with getline)
            // Instead, we need to handle this differently
            // For now, we'll break and let the main loop handle it
            // But we need to store the line somewhere
            break;
        }

        auto fields = split_fields(trimmed, 8);  // LS-DYNA uses 8-character fields typically
        if (fields.size() < 4) continue;

        LSDynaNode node;
        node.nid = parse_index(fields[0]);
        node.x = parse_real(fields[1]);
        node.y = parse_real(fields[2]);
        node.z = parse_real(fields[3]);
        node.tc = fields.size() > 4 ? parse_int(fields[4]) : 0;
        node.rc = fields.size() > 5 ? parse_int(fields[5]) : 0;

        nodes_.push_back(node);
    }
}

inline void LSDynaReader::parse_element_solid(std::istream& is) {
    std::string line;
    while (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || is_comment(trimmed)) continue;
        if (is_keyword(trimmed)) break;

        auto fields = split_fields(trimmed, 8);
        if (fields.size() < 3) continue;

        LSDynaElement elem;
        elem.eid = parse_index(fields[0]);
        elem.pid = parse_index(fields[1]);

        // Collect node IDs (eid, pid, n1, n2, n3, n4, n5, n6, n7, n8, ...)
        for (size_t i = 2; i < fields.size(); ++i) {
            Index nid = parse_index(fields[i]);
            if (nid > 0) {
                elem.nodes.push_back(nid);
            }
        }

        // Determine element type from number of nodes
        elem.type = determine_solid_type(static_cast<int>(elem.nodes.size()));
        elements_.push_back(elem);
    }
}

inline void LSDynaReader::parse_element_shell(std::istream& is) {
    std::string line;
    while (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || is_comment(trimmed)) continue;
        if (is_keyword(trimmed)) break;

        auto fields = split_fields(trimmed, 8);
        if (fields.size() < 4) continue;

        LSDynaElement elem;
        elem.eid = parse_index(fields[0]);
        elem.pid = parse_index(fields[1]);

        // Shell nodes: n1, n2, n3, n4
        for (size_t i = 2; i < fields.size() && i < 6; ++i) {
            Index nid = parse_index(fields[i]);
            if (nid > 0) {
                elem.nodes.push_back(nid);
            }
        }

        elem.type = determine_shell_type(static_cast<int>(elem.nodes.size()));
        elements_.push_back(elem);
    }
}

inline void LSDynaReader::parse_element_beam(std::istream& is) {
    std::string line;
    while (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || is_comment(trimmed)) continue;
        if (is_keyword(trimmed)) break;

        auto fields = split_fields(trimmed, 8);
        if (fields.size() < 5) continue;

        LSDynaElement elem;
        elem.eid = parse_index(fields[0]);
        elem.pid = parse_index(fields[1]);
        elem.nodes.push_back(parse_index(fields[2]));  // n1
        elem.nodes.push_back(parse_index(fields[3]));  // n2
        // n3 is orientation node, skip for now

        elem.type = ElementType::Beam2;
        elements_.push_back(elem);
    }
}

inline void LSDynaReader::parse_part(std::istream& is) {
    std::string line;
    std::string title;

    // First line is title (optional)
    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (!is_keyword(trimmed) && !trimmed.empty()) {
            title = trimmed;
        }
    }

    // Second line: pid, secid, mid, ...
    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 3) {
            LSDynaPart part;
            part.pid = parse_index(fields[0]);
            part.secid = parse_index(fields[1]);
            part.mid = parse_index(fields[2]);
            part.title = title;
            parts_[part.pid] = part;
        }
    }
}

inline void LSDynaReader::parse_mat_elastic(std::istream& is) {
    std::string line;

    // First line: mid, ro, e, pr, ...
    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 4) {
            LSDynaMaterial mat;
            mat.mid = parse_index(fields[0]);
            mat.type = "ELASTIC";
            mat.ro = parse_real(fields[1]);
            mat.e = parse_real(fields[2]);
            mat.pr = parse_real(fields[3]);
            materials_[mat.mid] = mat;
        }
    }
}

inline void LSDynaReader::parse_mat_plastic(std::istream& is) {
    std::string line;

    // Card 1: mid, ro, e, pr, sigy, etan, fail, tdel
    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 5) {
            LSDynaMaterial mat;
            mat.mid = parse_index(fields[0]);
            mat.type = "PIECEWISE_LINEAR_PLASTICITY";
            mat.ro = parse_real(fields[1]);
            mat.e = parse_real(fields[2]);
            mat.pr = parse_real(fields[3]);
            mat.sigy = parse_real(fields[4]);
            mat.etan = fields.size() > 5 ? parse_real(fields[5]) : 0.0;
            mat.fail = fields.size() > 6 ? parse_real(fields[6]) : 0.0;
            materials_[mat.mid] = mat;
        }
    }

    // Card 2: c, p, lcss, lcsr, vp
    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;
        // Parse if needed
    }
}

inline void LSDynaReader::parse_mat_johnson_cook(std::istream& is) {
    std::string line;

    // Card 1: mid, ro, g, e, pr, ...
    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 5) {
            LSDynaMaterial mat;
            mat.mid = parse_index(fields[0]);
            mat.type = "JOHNSON_COOK";
            mat.ro = parse_real(fields[1]);
            // G is shear modulus, E is computed from G and pr
            Real g = parse_real(fields[2]);
            mat.e = parse_real(fields[3]);
            mat.pr = parse_real(fields[4]);
            if (mat.e == 0.0 && g > 0.0 && mat.pr > 0.0) {
                mat.e = 2.0 * g * (1.0 + mat.pr);
            }
            materials_[mat.mid] = mat;
        }
    }

    // Card 2: a, b, n, c, m, tm, tr, epso
    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 5 && materials_.count(parse_index(split_fields(trimmed, 10)[0])) == 0) {
            // This is for the last material added
            if (!materials_.empty()) {
                auto& mat = materials_.rbegin()->second;
                mat.a_jc = parse_real(fields[0]);
                mat.b_jc = parse_real(fields[1]);
                mat.n_jc = parse_real(fields[2]);
                mat.c_jc = parse_real(fields[3]);
                mat.m_jc = parse_real(fields[4]);
                mat.tm = fields.size() > 5 ? parse_real(fields[5]) : 0.0;
                mat.tr = fields.size() > 6 ? parse_real(fields[6]) : 298.0;
                mat.epso = fields.size() > 7 ? parse_real(fields[7]) : 1.0;
            }
        }
    }
}

inline void LSDynaReader::parse_set_node_list(std::istream& is) {
    std::string line;
    LSDynaNodeSet node_set;

    // First line: sid, da1, da2, da3, da4, solver
    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 1) {
            node_set.sid = parse_index(fields[0]);
        }
    }

    // Following lines: node IDs (8 per line)
    while (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || is_comment(trimmed)) continue;
        if (is_keyword(trimmed)) break;

        auto fields = split_fields(trimmed, 10);
        for (const auto& f : fields) {
            Index nid = parse_index(f);
            if (nid > 0) {
                node_set.nodes.push_back(nid);
            }
        }
    }

    if (node_set.sid > 0) {
        node_sets_[node_set.sid] = node_set;
    }
}

inline void LSDynaReader::parse_boundary_spc_set(std::istream& is) {
    std::string line;

    // Card: nsid, cid, dofx, dofy, dofz, dofrx, dofry, dofrz
    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 5) {
            LSDynaSPC spc;
            spc.nsid = parse_index(fields[0]);
            spc.cid = parse_int(fields[1]);
            spc.dofx = parse_int(fields[2]);
            spc.dofy = parse_int(fields[3]);
            spc.dofz = parse_int(fields[4]);
            spc.dofrx = fields.size() > 5 ? parse_int(fields[5]) : 0;
            spc.dofry = fields.size() > 6 ? parse_int(fields[6]) : 0;
            spc.dofrz = fields.size() > 7 ? parse_int(fields[7]) : 0;
            spcs_.push_back(spc);
        }
    }
}

inline void LSDynaReader::parse_load_node_set(std::istream& is) {
    std::string line;

    // Card: nsid, dof, lcid, sf, cid, m1, m2, m3
    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 3) {
            LSDynaLoad load;
            load.nsid = parse_index(fields[0]);
            load.dof = parse_int(fields[1]);
            load.lcid = parse_index(fields[2]);
            load.sf = fields.size() > 3 ? parse_real(fields[3]) : 1.0;
            loads_.push_back(load);
        }
    }
}

inline void LSDynaReader::parse_define_curve(std::istream& is) {
    std::string line;
    LSDynaLoadCurve curve;

    // Card 1: lcid, sidr, sfa, sfo, offa, offo, dattyp
    if (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (is_keyword(trimmed)) return;

        auto fields = split_fields(trimmed, 10);
        if (fields.size() >= 1) {
            curve.lcid = parse_index(fields[0]);
        }
    }

    // Following lines: a1, o1
    while (std::getline(is, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || is_comment(trimmed)) continue;
        if (is_keyword(trimmed)) break;

        auto fields = split_fields(trimmed, 20);  // Wider fields for curve data
        if (fields.size() >= 2) {
            Real a = parse_real(fields[0]);
            Real o = parse_real(fields[1]);
            curve.points.emplace_back(a, o);
        }
    }

    if (curve.lcid > 0) {
        load_curves_[curve.lcid] = curve;
    }
}

inline void LSDynaReader::parse_title(std::istream& is) {
    std::string line;
    if (std::getline(is, line)) {
        title_ = trim(line);
    }
}

inline std::shared_ptr<Mesh> LSDynaReader::create_mesh() const {
    if (nodes_.empty()) {
        NXS_LOG_ERROR("No nodes parsed");
        return nullptr;
    }

    auto mesh = std::make_shared<Mesh>(nodes_.size());

    // Set node coordinates
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& node = nodes_[i];
        mesh->set_node_coordinates(i, Vec3r{node.x, node.y, node.z});
    }

    // Group elements by type and part
    std::map<std::pair<ElementType, Index>, std::vector<const LSDynaElement*>> element_groups;
    for (const auto& elem : elements_) {
        element_groups[{elem.type, elem.pid}].push_back(&elem);
    }

    // Create element blocks
    for (const auto& [key, elems] : element_groups) {
        auto [type, pid] = key;
        size_t nodes_per_elem = elems[0]->nodes.size();

        std::string block_name = "part_" + std::to_string(pid);
        Index block_id = mesh->add_element_block(block_name, type, elems.size(), nodes_per_elem);
        auto& block = mesh->element_block(block_id);

        // Set connectivity
        for (size_t ei = 0; ei < elems.size(); ++ei) {
            auto elem_nodes = block.element_nodes(ei);
            const auto& elem = *elems[ei];

            for (size_t ni = 0; ni < elem.nodes.size(); ++ni) {
                Index nid = elem.nodes[ni];
                auto it = node_id_to_index_.find(nid);
                if (it != node_id_to_index_.end()) {
                    elem_nodes[ni] = it->second;
                } else {
                    NXS_LOG_WARN("Node ID {} not found for element {}", nid, elem.eid);
                    elem_nodes[ni] = 0;
                }
            }

            // Set material and part IDs
            block.material_ids.component(0)[ei] = pid;  // Use part ID as material reference
            block.part_ids.component(0)[ei] = pid;
        }
    }

    // Add node sets
    for (const auto& [sid, ns] : node_sets_) {
        std::vector<Index> indices;
        for (Index nid : ns.nodes) {
            auto it = node_id_to_index_.find(nid);
            if (it != node_id_to_index_.end()) {
                indices.push_back(it->second);
            }
        }
        mesh->add_node_set("set_" + std::to_string(sid), indices);
    }

    return mesh;
}

inline std::vector<Index> LSDynaReader::get_node_set(Index sid) const {
    auto it = node_sets_.find(sid);
    if (it != node_sets_.end()) {
        std::vector<Index> indices;
        for (Index nid : it->second.nodes) {
            auto nit = node_id_to_index_.find(nid);
            if (nit != node_id_to_index_.end()) {
                indices.push_back(nit->second);
            }
        }
        return indices;
    }
    return {};
}

inline std::vector<Index> LSDynaReader::get_elements_by_part(Index pid) const {
    std::vector<Index> result;
    for (size_t i = 0; i < elements_.size(); ++i) {
        if (elements_[i].pid == pid) {
            result.push_back(i);
        }
    }
    return result;
}

inline physics::MaterialProperties LSDynaReader::get_material(Index pid) const {
    physics::MaterialProperties props;

    auto pit = parts_.find(pid);
    if (pit == parts_.end()) {
        return props;
    }

    Index mid = pit->second.mid;
    auto mit = materials_.find(mid);
    if (mit == materials_.end()) {
        return props;
    }

    const auto& mat = mit->second;
    props.density = mat.ro;
    props.E = mat.e;
    props.nu = mat.pr;
    props.yield_stress = mat.sigy > 0 ? mat.sigy : mat.a_jc;

    return props;
}

inline void LSDynaReader::print_summary(std::ostream& os) const {
    os << "LS-DYNA Model Summary\n";
    os << "=====================\n";
    if (!title_.empty()) {
        os << "Title: " << title_ << "\n";
    }
    os << "Nodes: " << nodes_.size() << "\n";
    os << "Elements: " << elements_.size() << "\n";
    os << "Materials: " << materials_.size() << "\n";
    os << "Parts: " << parts_.size() << "\n";
    os << "Node sets: " << node_sets_.size() << "\n";
    os << "SPCs: " << spcs_.size() << "\n";
    os << "Loads: " << loads_.size() << "\n";
    os << "Load curves: " << load_curves_.size() << "\n";

    // Element type breakdown
    std::map<ElementType, size_t> type_counts;
    for (const auto& elem : elements_) {
        type_counts[elem.type]++;
    }
    os << "\nElement Types:\n";
    for (const auto& [type, count] : type_counts) {
        os << "  " << to_string(type) << ": " << count << "\n";
    }
}

// Helper function implementations
inline std::string LSDynaReader::trim(const std::string& s) const {
    auto start = s.find_first_not_of(" \t\r\n");
    auto end = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

inline std::string LSDynaReader::to_upper(const std::string& s) const {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

inline std::vector<std::string> LSDynaReader::split_fields(const std::string& line, int field_width) const {
    std::vector<std::string> fields;

    // LS-DYNA uses fixed-width fields (typically 8 or 10 characters)
    // But we also need to handle comma-separated format
    if (line.find(',') != std::string::npos) {
        // Comma-separated
        std::istringstream iss(line);
        std::string field;
        while (std::getline(iss, field, ',')) {
            fields.push_back(trim(field));
        }
    } else {
        // Fixed-width or space-separated
        // Try space-separated first
        std::istringstream iss(line);
        std::string field;
        while (iss >> field) {
            fields.push_back(field);
        }
    }

    return fields;
}

inline bool LSDynaReader::is_keyword(const std::string& line) const {
    return !line.empty() && line[0] == '*';
}

inline std::string LSDynaReader::extract_keyword(const std::string& line) const {
    // Extract keyword up to underscore variants
    // e.g., "*ELEMENT_SOLID" -> "*ELEMENT_SOLID"
    // but "*MAT_ELASTIC_TITLE" -> "*MAT_ELASTIC"
    std::string kw = line;

    // Remove any trailing content after space
    auto space = kw.find(' ');
    if (space != std::string::npos) {
        kw = kw.substr(0, space);
    }

    // Check for _TITLE suffix and remove it
    if (kw.length() > 6 && kw.substr(kw.length() - 6) == "_TITLE") {
        kw = kw.substr(0, kw.length() - 6);
    }

    return kw;
}

inline bool LSDynaReader::is_comment(const std::string& line) const {
    return !line.empty() && line[0] == '$';
}

inline Real LSDynaReader::parse_real(const std::string& s, Real default_val) const {
    std::string trimmed = trim(s);
    if (trimmed.empty()) return default_val;
    try {
        return std::stod(trimmed);
    } catch (...) {
        return default_val;
    }
}

inline Index LSDynaReader::parse_index(const std::string& s, Index default_val) const {
    std::string trimmed = trim(s);
    if (trimmed.empty()) return default_val;
    try {
        return static_cast<Index>(std::stoul(trimmed));
    } catch (...) {
        return default_val;
    }
}

inline Int LSDynaReader::parse_int(const std::string& s, Int default_val) const {
    std::string trimmed = trim(s);
    if (trimmed.empty()) return default_val;
    try {
        return std::stoi(trimmed);
    } catch (...) {
        return default_val;
    }
}

inline ElementType LSDynaReader::determine_solid_type(int num_nodes) const {
    switch (num_nodes) {
        case 4:  return ElementType::Tet4;
        case 6:  return ElementType::Wedge6;
        case 8:  return ElementType::Hex8;
        case 10: return ElementType::Tet10;
        case 20: return ElementType::Hex20;
        default: return ElementType::Hex8;  // Default
    }
}

inline ElementType LSDynaReader::determine_shell_type(int num_nodes) const {
    switch (num_nodes) {
        case 3: return ElementType::Shell3;
        case 4: return ElementType::Shell4;
        default: return ElementType::Shell4;
    }
}

} // namespace io
} // namespace nxs
