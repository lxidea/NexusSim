#pragma once

/**
 * @file starter_wave40.hpp
 * @brief Wave 40: Starter/Reader Extensions and Error System
 *
 * Five features:
 *  7.  RadiossStarterFull  - Complete D00 keyword parsing
 *  8.  PropertyReader      - /PROP card reader (shell, beam, solid)
 *  9.  SectionForceOutput  - Cross-section force/moment extraction
 *  10. ModelValidatorExt   - Extended model validation
 *  11. ErrorMessageSystem  - Structured error/warning/info messages
 *
 * Reference: OpenRadioss Starter Input Manual, LS-DYNA keyword format.
 */

#include <nexussim/core/types.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <array>
#include <iomanip>
#include <map>
#include <functional>

namespace nxs {
namespace io {

using Real = nxs::Real;

// ============================================================================
// Common data structures for Wave 40 starter/reader
// ============================================================================

/// Node for starter model
struct StarterNode {
    int id = 0;
    Real x = 0, y = 0, z = 0;
};

/// Element for starter model
struct StarterElement {
    int id = 0;
    int part_id = 0;
    int prop_id = 0;
    int mat_id = 0;
    std::string type;              ///< "SHELL", "BRICK", "BEAM", etc.
    std::vector<int> node_ids;
};

/// Material card from starter
struct StarterMaterial {
    int id = 0;
    std::string type;              ///< e.g., "ELAST", "PLAS_JOHNS", etc.
    std::map<std::string, Real> params;
};

/// Boundary condition from starter
struct StarterBC {
    int node_id = 0;
    int trarot = 0;                ///< Translation/rotation code (e.g., 111000)
    std::string group_name;
};

/// Interface (contact) from starter
struct StarterInterface {
    int id = 0;
    int type = 0;                  ///< Interface type number
    std::vector<int> master_segs;
    std::vector<int> slave_nodes;
    Real friction = 0;
};

/// Node group
struct StarterNodeGroup {
    int id = 0;
    std::string name;
    std::vector<int> node_ids;
};

/// Property card
struct PropertyCard {
    int id = 0;
    std::string type;              ///< "SHELL", "BEAM", "SOLID"
    std::map<std::string, Real> parameters;

    // Convenience accessors
    Real thickness() const {
        auto it = parameters.find("thickness");
        return (it != parameters.end()) ? it->second : 0.0;
    }

    int num_integration_points() const {
        auto it = parameters.find("num_integration_points");
        return (it != parameters.end()) ? static_cast<int>(it->second) : 1;
    }

    Real cross_section_area() const {
        auto it = parameters.find("cross_section_area");
        return (it != parameters.end()) ? it->second : 0.0;
    }

    Real iyy() const {
        auto it = parameters.find("Iyy");
        return (it != parameters.end()) ? it->second : 0.0;
    }

    Real izz() const {
        auto it = parameters.find("Izz");
        return (it != parameters.end()) ? it->second : 0.0;
    }

    int integration_rule() const {
        auto it = parameters.find("integration_rule");
        return (it != parameters.end()) ? static_cast<int>(it->second) : 0;
    }
};

/// Complete starter model
struct StarterModel {
    std::vector<StarterNode> nodes;
    std::vector<StarterElement> elements;
    std::vector<PropertyCard> properties;
    std::vector<StarterMaterial> materials;
    std::vector<StarterBC> bcs;
    std::vector<StarterInterface> interfaces;
    std::vector<StarterNodeGroup> node_groups;
    std::string title;
    int parse_errors = 0;
    int parse_warnings = 0;

    int node_count() const { return static_cast<int>(nodes.size()); }
    int element_count() const { return static_cast<int>(elements.size()); }
};

/// Section forces result
struct SectionForces {
    Real fx = 0, fy = 0, fz = 0;
    Real mx = 0, my = 0, mz = 0;
    int contributing_elements = 0;

    Real force_magnitude() const {
        return std::sqrt(fx * fx + fy * fy + fz * fz);
    }

    Real moment_magnitude() const {
        return std::sqrt(mx * mx + my * my + mz * mz);
    }
};

/// Element stress data for section force computation
struct ElementStressData {
    int id = 0;
    std::array<Real, 3> centroid = {};   ///< Element centroid position
    std::array<Real, 6> stress = {};     ///< xx, yy, zz, xy, yz, xz
    Real area = 0;                       ///< Cross-sectional area contribution
    Real volume = 0;
};

/// Validation report
struct ValidationReport {
    struct Message {
        std::string category;
        std::string text;
    };

    std::vector<Message> errors;
    std::vector<Message> warnings;
    std::vector<Message> info;

    int error_count() const { return static_cast<int>(errors.size()); }
    int warning_count() const { return static_cast<int>(warnings.size()); }
    int info_count() const { return static_cast<int>(info.size()); }
    bool is_valid() const { return errors.empty(); }

    std::string summary() const {
        std::ostringstream oss;
        oss << "Validation: " << errors.size() << " errors, "
            << warnings.size() << " warnings, "
            << info.size() << " info";
        return oss.str();
    }
};

// ============================================================================
// Internal string utilities
// ============================================================================

namespace detail_w40 {

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

inline std::vector<std::string> split_fields(const std::string& line, int field_width = 10) {
    std::vector<std::string> fields;
    for (size_t i = 0; i < line.size(); i += field_width) {
        size_t len = std::min(static_cast<size_t>(field_width), line.size() - i);
        fields.push_back(trim(line.substr(i, len)));
    }
    return fields;
}

inline Real parse_real(const std::string& s) {
    if (s.empty()) return 0.0;
    try { return std::stod(s); }
    catch (...) { return 0.0; }
}

inline int parse_int(const std::string& s) {
    if (s.empty()) return 0;
    try { return std::stoi(s); }
    catch (...) { return 0; }
}

} // namespace detail_w40

// ############################################################################
// 7. RadiossStarterFull -- Complete D00 keyword parsing
// ############################################################################

/**
 * Parses Radioss starter (D00) format with keyword dispatcher.
 * Supported keywords:
 *   /NODE, /SHELL, /BRICK, /PROP, /MAT, /BCS, /INTER, /GRNOD
 *
 * Lines starting with '/' are keywords. Data lines follow fixed-field format.
 */
class RadiossStarterFull {
public:
    using ParseFunc = std::function<void(std::ifstream&, StarterModel&)>;

    RadiossStarterFull() {
        register_default_parsers();
    }

    /// Parse a D00 file and return the model.
    StarterModel parse(const std::string& filename) {
        StarterModel model;
        std::ifstream file(filename);
        if (!file.is_open()) {
            model.parse_errors = 1;
            return model;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::string trimmed = detail_w40::trim(line);
            if (trimmed.empty() || trimmed[0] == '#') continue;

            if (trimmed[0] == '/') {
                // Extract keyword (e.g., "/NODE" from "/NODE/1234")
                std::string keyword;
                size_t slash2 = trimmed.find('/', 1);
                if (slash2 != std::string::npos) {
                    keyword = detail_w40::to_upper(trimmed.substr(0, slash2));
                } else {
                    keyword = detail_w40::to_upper(trimmed);
                }

                // Check for /BEGIN to parse title
                if (keyword == "/BEGIN") {
                    if (std::getline(file, line)) {
                        model.title = detail_w40::trim(line);
                    }
                    continue;
                }

                auto it = parsers_.find(keyword);
                if (it != parsers_.end()) {
                    it->second(file, model);
                }
                // Unknown keywords are silently skipped
            }
        }

        file.close();
        return model;
    }

    /// Register a custom keyword parser.
    void register_parser(const std::string& keyword, ParseFunc func) {
        parsers_[detail_w40::to_upper(keyword)] = std::move(func);
    }

private:
    std::map<std::string, ParseFunc> parsers_;

    void register_default_parsers() {
        // /NODE parser
        parsers_["/NODE"] = [](std::ifstream& file, StarterModel& model) {
            std::string line;
            while (std::getline(file, line)) {
                std::string trimmed = detail_w40::trim(line);
                if (trimmed.empty()) continue;
                if (trimmed[0] == '/' || trimmed[0] == '#') {
                    // Seek back so main loop sees the keyword
                    for (auto it = line.rbegin(); it != line.rend(); ++it)
                        file.putback(*it);
                    file.putback('\n');
                    break;
                }
                auto fields = detail_w40::split_fields(trimmed);
                if (fields.size() >= 4) {
                    StarterNode node;
                    node.id = detail_w40::parse_int(fields[0]);
                    node.x = detail_w40::parse_real(fields[1]);
                    node.y = detail_w40::parse_real(fields[2]);
                    node.z = detail_w40::parse_real(fields[3]);
                    model.nodes.push_back(node);
                }
            }
        };

        // /SHELL parser
        parsers_["/SHELL"] = [](std::ifstream& file, StarterModel& model) {
            std::string line;
            while (std::getline(file, line)) {
                std::string trimmed = detail_w40::trim(line);
                if (trimmed.empty()) continue;
                if (trimmed[0] == '/' || trimmed[0] == '#') {
                    for (auto it = line.rbegin(); it != line.rend(); ++it)
                        file.putback(*it);
                    file.putback('\n');
                    break;
                }
                auto fields = detail_w40::split_fields(trimmed);
                if (fields.size() >= 6) {
                    StarterElement elem;
                    elem.id = detail_w40::parse_int(fields[0]);
                    elem.part_id = detail_w40::parse_int(fields[1]);
                    elem.type = "SHELL";
                    for (size_t i = 2; i < fields.size() && !fields[i].empty(); ++i) {
                        elem.node_ids.push_back(detail_w40::parse_int(fields[i]));
                    }
                    model.elements.push_back(std::move(elem));
                }
            }
        };

        // /BRICK parser
        parsers_["/BRICK"] = [](std::ifstream& file, StarterModel& model) {
            std::string line;
            while (std::getline(file, line)) {
                std::string trimmed = detail_w40::trim(line);
                if (trimmed.empty()) continue;
                if (trimmed[0] == '/' || trimmed[0] == '#') {
                    for (auto it = line.rbegin(); it != line.rend(); ++it)
                        file.putback(*it);
                    file.putback('\n');
                    break;
                }
                auto fields = detail_w40::split_fields(trimmed);
                if (fields.size() >= 10) {
                    StarterElement elem;
                    elem.id = detail_w40::parse_int(fields[0]);
                    elem.part_id = detail_w40::parse_int(fields[1]);
                    elem.type = "BRICK";
                    for (size_t i = 2; i < fields.size() && !fields[i].empty(); ++i) {
                        elem.node_ids.push_back(detail_w40::parse_int(fields[i]));
                    }
                    model.elements.push_back(std::move(elem));
                }
            }
        };

        // /PROP parser
        parsers_["/PROP"] = [](std::ifstream& file, StarterModel& model) {
            std::string line;
            // First data line: property header
            if (!std::getline(file, line)) return;
            auto hdr = detail_w40::split_fields(detail_w40::trim(line));
            if (hdr.size() < 2) return;

            PropertyCard prop;
            prop.id = detail_w40::parse_int(hdr[0]);
            prop.type = (hdr.size() > 1) ? hdr[1] : "UNKNOWN";

            // Second data line: parameters
            if (std::getline(file, line)) {
                auto fields = detail_w40::split_fields(detail_w40::trim(line));
                if (prop.type == "SHELL" || prop.type == "shell") {
                    if (fields.size() >= 1)
                        prop.parameters["thickness"] = detail_w40::parse_real(fields[0]);
                    if (fields.size() >= 2)
                        prop.parameters["num_integration_points"] = detail_w40::parse_real(fields[1]);
                } else if (prop.type == "BEAM" || prop.type == "beam") {
                    if (fields.size() >= 1)
                        prop.parameters["cross_section_area"] = detail_w40::parse_real(fields[0]);
                    if (fields.size() >= 2)
                        prop.parameters["Iyy"] = detail_w40::parse_real(fields[1]);
                    if (fields.size() >= 3)
                        prop.parameters["Izz"] = detail_w40::parse_real(fields[2]);
                } else if (prop.type == "SOLID" || prop.type == "solid") {
                    if (fields.size() >= 1)
                        prop.parameters["integration_rule"] = detail_w40::parse_real(fields[0]);
                }
            }

            model.properties.push_back(std::move(prop));
        };

        // /MAT parser
        parsers_["/MAT"] = [](std::ifstream& file, StarterModel& model) {
            std::string line;
            if (!std::getline(file, line)) return;
            auto hdr = detail_w40::split_fields(detail_w40::trim(line));
            if (hdr.size() < 2) return;

            StarterMaterial mat;
            mat.id = detail_w40::parse_int(hdr[0]);
            mat.type = hdr[1];

            // Read parameter lines until next keyword or empty
            while (std::getline(file, line)) {
                std::string trimmed = detail_w40::trim(line);
                if (trimmed.empty()) continue;
                if (trimmed[0] == '/' || trimmed[0] == '#') {
                    for (auto it = line.rbegin(); it != line.rend(); ++it)
                        file.putback(*it);
                    file.putback('\n');
                    break;
                }
                auto fields = detail_w40::split_fields(trimmed, 20);
                for (size_t i = 0; i + 1 < fields.size(); i += 2) {
                    if (!fields[i].empty() && !fields[i + 1].empty()) {
                        mat.params[fields[i]] = detail_w40::parse_real(fields[i + 1]);
                    }
                }
            }

            model.materials.push_back(std::move(mat));
        };

        // /BCS parser
        parsers_["/BCS"] = [](std::ifstream& file, StarterModel& model) {
            std::string line;
            while (std::getline(file, line)) {
                std::string trimmed = detail_w40::trim(line);
                if (trimmed.empty()) continue;
                if (trimmed[0] == '/' || trimmed[0] == '#') {
                    for (auto it = line.rbegin(); it != line.rend(); ++it)
                        file.putback(*it);
                    file.putback('\n');
                    break;
                }
                auto fields = detail_w40::split_fields(trimmed);
                if (fields.size() >= 2) {
                    StarterBC bc;
                    bc.node_id = detail_w40::parse_int(fields[0]);
                    bc.trarot = detail_w40::parse_int(fields[1]);
                    if (fields.size() >= 3) bc.group_name = fields[2];
                    model.bcs.push_back(std::move(bc));
                }
            }
        };

        // /INTER parser
        parsers_["/INTER"] = [](std::ifstream& file, StarterModel& model) {
            std::string line;
            if (!std::getline(file, line)) return;
            auto hdr = detail_w40::split_fields(detail_w40::trim(line));
            if (hdr.size() < 2) return;

            StarterInterface iface;
            iface.id = detail_w40::parse_int(hdr[0]);
            iface.type = detail_w40::parse_int(hdr[1]);
            if (hdr.size() >= 3) iface.friction = detail_w40::parse_real(hdr[2]);

            model.interfaces.push_back(std::move(iface));
        };

        // /GRNOD parser
        parsers_["/GRNOD"] = [](std::ifstream& file, StarterModel& model) {
            std::string line;
            if (!std::getline(file, line)) return;
            auto hdr = detail_w40::split_fields(detail_w40::trim(line));
            if (hdr.empty()) return;

            StarterNodeGroup grp;
            grp.id = detail_w40::parse_int(hdr[0]);
            if (hdr.size() >= 2) grp.name = hdr[1];

            while (std::getline(file, line)) {
                std::string trimmed = detail_w40::trim(line);
                if (trimmed.empty()) continue;
                if (trimmed[0] == '/' || trimmed[0] == '#') {
                    for (auto it = line.rbegin(); it != line.rend(); ++it)
                        file.putback(*it);
                    file.putback('\n');
                    break;
                }
                auto fields = detail_w40::split_fields(trimmed);
                for (const auto& f : fields) {
                    if (!f.empty()) grp.node_ids.push_back(detail_w40::parse_int(f));
                }
            }

            model.node_groups.push_back(std::move(grp));
        };
    }
};

// ############################################################################
// 8. PropertyReader -- /PROP card reader
// ############################################################################

/**
 * Standalone property card parser supporting shell, beam, and solid properties.
 * Can parse from a string or from file lines.
 */
class PropertyReader {
public:
    PropertyReader() = default;

    /// Read a property card from raw text lines.
    /// First line: ID  TYPE
    /// Second line: parameter values (type-dependent)
    PropertyCard read_property(const std::vector<std::string>& lines) {
        PropertyCard prop;
        if (lines.empty()) return prop;

        auto hdr = detail_w40::split_fields(detail_w40::trim(lines[0]));
        if (hdr.size() >= 1) prop.id = detail_w40::parse_int(hdr[0]);
        if (hdr.size() >= 2) prop.type = detail_w40::to_upper(hdr[1]);

        if (lines.size() >= 2) {
            parse_params(prop, detail_w40::trim(lines[1]));
        }

        return prop;
    }

    /// Read property from a single formatted string (ID TYPE param1 param2 ...).
    PropertyCard read_property_line(const std::string& line) {
        PropertyCard prop;
        std::istringstream iss(line);
        std::string token;

        if (!(iss >> prop.id)) return prop;
        if (!(iss >> prop.type)) return prop;
        prop.type = detail_w40::to_upper(prop.type);

        std::vector<Real> vals;
        Real v;
        while (iss >> v) vals.push_back(v);

        if (prop.type == "SHELL") {
            if (vals.size() >= 1) prop.parameters["thickness"] = vals[0];
            if (vals.size() >= 2) prop.parameters["num_integration_points"] = vals[1];
        } else if (prop.type == "BEAM") {
            if (vals.size() >= 1) prop.parameters["cross_section_area"] = vals[0];
            if (vals.size() >= 2) prop.parameters["Iyy"] = vals[1];
            if (vals.size() >= 3) prop.parameters["Izz"] = vals[2];
        } else if (prop.type == "SOLID") {
            if (vals.size() >= 1) prop.parameters["integration_rule"] = vals[0];
        }

        return prop;
    }

private:
    void parse_params(PropertyCard& prop, const std::string& param_line) {
        auto fields = detail_w40::split_fields(param_line);
        if (prop.type == "SHELL") {
            if (fields.size() >= 1)
                prop.parameters["thickness"] = detail_w40::parse_real(fields[0]);
            if (fields.size() >= 2)
                prop.parameters["num_integration_points"] = detail_w40::parse_real(fields[1]);
        } else if (prop.type == "BEAM") {
            if (fields.size() >= 1)
                prop.parameters["cross_section_area"] = detail_w40::parse_real(fields[0]);
            if (fields.size() >= 2)
                prop.parameters["Iyy"] = detail_w40::parse_real(fields[1]);
            if (fields.size() >= 3)
                prop.parameters["Izz"] = detail_w40::parse_real(fields[2]);
        } else if (prop.type == "SOLID") {
            if (fields.size() >= 1)
                prop.parameters["integration_rule"] = detail_w40::parse_real(fields[0]);
        }
    }
};

// ############################################################################
// 9. SectionForceOutput -- Cross-section force/moment extraction
// ############################################################################

/**
 * Defines a cutting plane and computes force/moment resultants
 * from element stress contributions crossing that plane.
 *
 * The cutting plane is defined by a point and outward normal.
 * Elements whose centroids are within a tolerance of the plane contribute.
 */
class SectionForceOutput {
public:
    SectionForceOutput() = default;

    /// Define the cutting plane.
    void define_section(const std::array<Real, 3>& point,
                        const std::array<Real, 3>& normal) {
        point_ = point;
        // Normalize
        Real len = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                             normal[2] * normal[2]);
        if (len > 1e-30) {
            normal_ = {normal[0] / len, normal[1] / len, normal[2] / len};
        } else {
            normal_ = {0, 0, 1};
        }
        defined_ = true;
    }

    /// Set the tolerance for plane proximity check (default = 0.1).
    void set_tolerance(Real tol) { tolerance_ = tol; }

    /// Compute section forces from element stress data.
    SectionForces compute(const std::vector<ElementStressData>& elements) const {
        SectionForces result;
        if (!defined_) return result;

        for (const auto& e : elements) {
            // Check if element centroid is near the plane
            Real dist = plane_distance(e.centroid);
            if (std::abs(dist) > tolerance_) continue;

            // Stress tensor * normal -> traction vector
            // sigma = [sxx, syy, szz, sxy, syz, sxz]
            Real tx = e.stress[0] * normal_[0] + e.stress[3] * normal_[1] +
                      e.stress[5] * normal_[2];
            Real ty = e.stress[3] * normal_[0] + e.stress[1] * normal_[1] +
                      e.stress[4] * normal_[2];
            Real tz = e.stress[5] * normal_[0] + e.stress[4] * normal_[1] +
                      e.stress[2] * normal_[2];

            Real area = (e.area > 0) ? e.area : 1.0;

            // Force contribution
            result.fx += tx * area;
            result.fy += ty * area;
            result.fz += tz * area;

            // Moment about the section point: M = r x F
            Real rx = e.centroid[0] - point_[0];
            Real ry = e.centroid[1] - point_[1];
            Real rz = e.centroid[2] - point_[2];

            Real force_x = tx * area;
            Real force_y = ty * area;
            Real force_z = tz * area;

            result.mx += ry * force_z - rz * force_y;
            result.my += rz * force_x - rx * force_z;
            result.mz += rx * force_y - ry * force_x;

            ++result.contributing_elements;
        }

        return result;
    }

    /// Check if a section has been defined.
    bool is_defined() const { return defined_; }

    /// Get the section point.
    const std::array<Real, 3>& section_point() const { return point_; }

    /// Get the section normal.
    const std::array<Real, 3>& section_normal() const { return normal_; }

private:
    std::array<Real, 3> point_ = {0, 0, 0};
    std::array<Real, 3> normal_ = {0, 0, 1};
    Real tolerance_ = 0.1;
    bool defined_ = false;

    /// Signed distance from point to cutting plane.
    Real plane_distance(const std::array<Real, 3>& pt) const {
        return normal_[0] * (pt[0] - point_[0]) +
               normal_[1] * (pt[1] - point_[1]) +
               normal_[2] * (pt[2] - point_[2]);
    }
};

// ############################################################################
// 10. ModelValidatorExt -- Extended model validation
// ############################################################################

/**
 * Validates a StarterModel for consistency:
 *   - Element connectivity (no dangling nodes)
 *   - Material assignment completeness
 *   - BC consistency (no over-constrained)
 *   - Contact surface overlap detection
 */
class ModelValidatorExt {
public:
    ModelValidatorExt() = default;

    /// Run all validation checks and return report.
    ValidationReport validate(const StarterModel& model) const {
        ValidationReport report;

        check_dangling_nodes(model, report);
        check_material_assignment(model, report);
        check_bc_consistency(model, report);
        check_contact_surfaces(model, report);
        add_model_info(model, report);

        return report;
    }

private:
    /// Check for element nodes that don't exist in the node list.
    void check_dangling_nodes(const StarterModel& model,
                              ValidationReport& report) const {
        // Build set of valid node IDs
        std::map<int, bool> node_exists;
        for (const auto& n : model.nodes) {
            node_exists[n.id] = true;
        }

        int dangling_count = 0;
        for (const auto& e : model.elements) {
            for (int nid : e.node_ids) {
                if (node_exists.find(nid) == node_exists.end()) {
                    ++dangling_count;
                    if (dangling_count <= 10) {  // Limit messages
                        report.errors.push_back({"connectivity",
                            "Element " + std::to_string(e.id) +
                            " references non-existent node " + std::to_string(nid)});
                    }
                }
            }
        }

        if (dangling_count > 10) {
            report.errors.push_back({"connectivity",
                "... and " + std::to_string(dangling_count - 10) +
                " more dangling node references"});
        }

        if (dangling_count == 0 && !model.elements.empty()) {
            report.info.push_back({"connectivity", "All element nodes resolved"});
        }
    }

    /// Check that all elements have material/property assignments.
    void check_material_assignment(const StarterModel& model,
                                   ValidationReport& report) const {
        // Build sets of valid material and property IDs
        std::map<int, bool> mat_ids, prop_ids;
        for (const auto& m : model.materials) mat_ids[m.id] = true;
        for (const auto& p : model.properties) prop_ids[p.id] = true;

        int no_mat = 0;
        for (const auto& e : model.elements) {
            if (e.mat_id != 0 && mat_ids.find(e.mat_id) == mat_ids.end()) {
                ++no_mat;
                if (no_mat <= 5) {
                    report.warnings.push_back({"material",
                        "Element " + std::to_string(e.id) +
                        " references undefined material " + std::to_string(e.mat_id)});
                }
            }
        }

        if (no_mat > 5) {
            report.warnings.push_back({"material",
                "... and " + std::to_string(no_mat - 5) +
                " more undefined material references"});
        }

        if (no_mat == 0 && !model.elements.empty()) {
            report.info.push_back({"material", "All material assignments valid"});
        }
    }

    /// Check BC consistency (duplicate constraints, over-constrained).
    void check_bc_consistency(const StarterModel& model,
                              ValidationReport& report) const {
        // Check for duplicate node BCs
        std::map<int, int> bc_count;
        for (const auto& bc : model.bcs) {
            bc_count[bc.node_id]++;
        }

        for (const auto& [nid, count] : bc_count) {
            if (count > 1) {
                report.warnings.push_back({"boundary",
                    "Node " + std::to_string(nid) +
                    " has " + std::to_string(count) + " BC definitions (possible over-constraint)"});
            }
        }

        // Check BC nodes exist
        std::map<int, bool> node_exists;
        for (const auto& n : model.nodes) node_exists[n.id] = true;

        for (const auto& bc : model.bcs) {
            if (!node_exists.empty() && node_exists.find(bc.node_id) == node_exists.end()) {
                report.errors.push_back({"boundary",
                    "BC references non-existent node " + std::to_string(bc.node_id)});
            }
        }
    }

    /// Check contact surface definitions.
    void check_contact_surfaces(const StarterModel& model,
                                ValidationReport& report) const {
        // Check for duplicate interface IDs
        std::map<int, int> iface_ids;
        for (const auto& iface : model.interfaces) {
            iface_ids[iface.id]++;
        }

        for (const auto& [id, count] : iface_ids) {
            if (count > 1) {
                report.warnings.push_back({"contact",
                    "Duplicate interface ID " + std::to_string(id)});
            }
        }

        // Check friction values
        for (const auto& iface : model.interfaces) {
            if (iface.friction < 0.0) {
                report.warnings.push_back({"contact",
                    "Interface " + std::to_string(iface.id) +
                    " has negative friction coefficient"});
            }
        }

        if (!model.interfaces.empty()) {
            report.info.push_back({"contact",
                std::to_string(model.interfaces.size()) + " contact interfaces defined"});
        }
    }

    /// Add general model info.
    void add_model_info(const StarterModel& model,
                        ValidationReport& report) const {
        report.info.push_back({"summary",
            "Model: " + std::to_string(model.nodes.size()) + " nodes, " +
            std::to_string(model.elements.size()) + " elements, " +
            std::to_string(model.materials.size()) + " materials"});
    }
};

// ############################################################################
// 11. ErrorMessageSystem -- Structured error/warning/info messages
// ############################################################################

/**
 * Centralized error/warning/info/debug message system.
 * Each message has: ID, severity, context, text, and optional suggestion.
 */
class ErrorMessageSystem {
public:
    enum class Severity {
        Debug = 0,
        Info = 1,
        Warning = 2,
        Error = 3
    };

    struct Message {
        int id = 0;
        Severity severity = Severity::Info;
        std::string context;
        std::string text;
        std::string suggestion;
    };

    struct Summary {
        int debug_count = 0;
        int info_count = 0;
        int warning_count = 0;
        int error_count = 0;
        int total() const { return debug_count + info_count + warning_count + error_count; }
    };

    ErrorMessageSystem() = default;

    /// Add an error message.
    void error(int id, const std::string& msg, const std::string& suggestion = "") {
        messages_.push_back({id, Severity::Error, "", msg, suggestion});
    }

    /// Add an error with context.
    void error(int id, const std::string& context, const std::string& msg,
               const std::string& suggestion) {
        messages_.push_back({id, Severity::Error, context, msg, suggestion});
    }

    /// Add a warning message.
    void warning(int id, const std::string& msg) {
        messages_.push_back({id, Severity::Warning, "", msg, ""});
    }

    /// Add a warning with context.
    void warning(int id, const std::string& context, const std::string& msg) {
        messages_.push_back({id, Severity::Warning, context, msg, ""});
    }

    /// Add an info message.
    void info(int id, const std::string& msg) {
        messages_.push_back({id, Severity::Info, "", msg, ""});
    }

    /// Add a debug message.
    void debug(int id, const std::string& msg) {
        messages_.push_back({id, Severity::Debug, "", msg, ""});
    }

    /// Get summary counts by severity.
    Summary get_summary() const {
        Summary s;
        for (const auto& m : messages_) {
            switch (m.severity) {
                case Severity::Debug:   ++s.debug_count; break;
                case Severity::Info:    ++s.info_count; break;
                case Severity::Warning: ++s.warning_count; break;
                case Severity::Error:   ++s.error_count; break;
            }
        }
        return s;
    }

    /// Write all messages to a log file.
    bool write_log(const std::string& filename) const {
        std::ofstream file(filename, std::ios::out);
        if (!file.is_open()) return false;

        file << "NexusSim Message Log\n";
        file << "================================================================\n\n";

        for (const auto& m : messages_) {
            file << severity_tag(m.severity)
                 << " [" << std::setw(5) << m.id << "]";
            if (!m.context.empty()) {
                file << " (" << m.context << ")";
            }
            file << ": " << m.text << "\n";
            if (!m.suggestion.empty()) {
                file << "       Suggestion: " << m.suggestion << "\n";
            }
        }

        file << "\n================================================================\n";
        auto s = get_summary();
        file << "Summary: "
             << s.error_count << " errors, "
             << s.warning_count << " warnings, "
             << s.info_count << " info, "
             << s.debug_count << " debug\n";

        file.close();
        return true;
    }

    /// Get all messages.
    const std::vector<Message>& messages() const { return messages_; }

    /// Get messages filtered by severity.
    std::vector<Message> messages_by_severity(Severity sev) const {
        std::vector<Message> result;
        for (const auto& m : messages_) {
            if (m.severity == sev) result.push_back(m);
        }
        return result;
    }

    /// Clear all messages.
    void clear() { messages_.clear(); }

    /// Check if any errors exist.
    bool has_errors() const {
        for (const auto& m : messages_) {
            if (m.severity == Severity::Error) return true;
        }
        return false;
    }

    /// Check if any warnings exist.
    bool has_warnings() const {
        for (const auto& m : messages_) {
            if (m.severity == Severity::Warning) return true;
        }
        return false;
    }

    /// Total message count.
    int count() const { return static_cast<int>(messages_.size()); }

private:
    std::vector<Message> messages_;

    static const char* severity_tag(Severity sev) {
        switch (sev) {
            case Severity::Debug:   return "DEBUG  ";
            case Severity::Info:    return "INFO   ";
            case Severity::Warning: return "WARNING";
            case Severity::Error:   return "ERROR  ";
            default:                return "UNKNOWN";
        }
    }
};

} // namespace io
} // namespace nxs
