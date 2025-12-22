/**
 * @file radioss_reader.cpp
 * @brief OpenRadioss input deck reader implementation
 */

#include <nexussim/io/radioss_reader.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <iomanip>

namespace nxs {
namespace io {

// Helper function to convert ElementType to string
static std::string element_type_to_string(ElementType type) {
    switch (type) {
        case ElementType::Hex8: return "Hex8";
        case ElementType::Hex20: return "Hex20";
        case ElementType::Tet4: return "Tet4";
        case ElementType::Tet10: return "Tet10";
        case ElementType::Wedge6: return "Wedge6";
        case ElementType::Shell4: return "Shell4";
        case ElementType::Shell3: return "Shell3";
        case ElementType::Beam2: return "Beam2";
        case ElementType::Truss: return "Truss";
        case ElementType::Spring: return "Spring";
        case ElementType::Damper: return "Damper";
        case ElementType::SpringDamper: return "SpringDamper";
        default: return "Unknown";
    }
}

bool RadiossReader::read(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "RadiossReader: Cannot open file: " << filename << std::endl;
        return false;
    }

    std::cout << "RadiossReader: Reading " << filename << std::endl;

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        // Parse keywords starting with /
        if (line[0] == '/') {
            parse_line(line);

            // Handle multi-line blocks
            if (starts_with(line, "/BEGIN")) {
                parse_begin(file);
            } else if (starts_with(line, "/TITLE")) {
                parse_title(file);
            } else if (starts_with(line, "/NODE")) {
                parse_node(file);
            } else if (starts_with(line, "/SHELL/")) {
                Index part_id = parse_part_id(line);
                parse_shell(file, part_id);
            } else if (starts_with(line, "/SH3N/")) {
                Index part_id = parse_part_id(line);
                parse_sh3n(file, part_id);
            } else if (starts_with(line, "/BRICK/")) {
                Index part_id = parse_part_id(line);
                parse_brick(file, part_id);
            } else if (starts_with(line, "/BEAM/")) {
                Index part_id = parse_part_id(line);
                parse_beam(file, part_id);
            } else if (starts_with(line, "/SPRING/")) {
                Index part_id = parse_part_id(line);
                parse_spring(file, part_id);
            } else if (starts_with(line, "/MAT/LAW")) {
                // Parse material law and ID: /MAT/LAW1/123
                // split returns ["MAT", "LAW1", "123"]
                auto parts = split(line);
                if (parts.size() >= 3) {
                    std::string law = parts[1];  // "LAW1"
                    Index mat_id = std::stoul(parts[2]);  // "123"
                    parse_mat(file, law, mat_id);
                }
            } else if (starts_with(line, "/PART/")) {
                Index part_id = parse_part_id(line);
                parse_part(file, part_id);
            } else if (starts_with(line, "/BCS")) {
                parse_bcs(file);
            } else if (starts_with(line, "/IMPVEL")) {
                parse_impvel(file);
            } else if (starts_with(line, "/CLOAD")) {
                parse_cload(file);
            }
        }
    }

    // Build node ID to index mapping
    for (size_t i = 0; i < nodes_.size(); ++i) {
        node_id_to_index_[nodes_[i].id] = i;
    }

    std::cout << "RadiossReader: Parsed " << nodes_.size() << " nodes, "
              << elements_.size() << " elements, "
              << materials_.size() << " materials" << std::endl;

    return true;
}

void RadiossReader::parse_line(const std::string& line) {
    // Currently just for logging/debugging
    // NXS_LOG_DEBUG("RadiossReader: Parsing {}", line);
    (void)line;  // Suppress unused warning
}

void RadiossReader::parse_begin(std::istream& is) {
    std::string line;

    // Line 1: Model name
    if (std::getline(is, line)) {
        title_ = trim(line);
    }

    // Line 2: Version info (skip)
    std::getline(is, line);

    // Line 3: Units (mass, length, time)
    if (std::getline(is, line)) {
        units_ = trim(line);
    }
}

void RadiossReader::parse_title(std::istream& is) {
    std::string line;
    if (std::getline(is, line)) {
        title_ = trim(line);
    }
}

void RadiossReader::parse_node(std::istream& is) {
    std::string line;
    while (std::getline(is, line)) {
        line = trim(line);

        // Stop at next keyword or empty line
        if (line.empty() || line[0] == '/') {
            break;
        }

        // Skip comments
        if (line[0] == '#') continue;

        // Parse: node_id x y z
        std::istringstream iss(line);
        RadiossNode node;
        if (iss >> node.id >> node.x >> node.y >> node.z) {
            nodes_.push_back(node);
        }
    }
}

void RadiossReader::parse_shell(std::istream& is, Index part_id) {
    std::string line;
    while (std::getline(is, line)) {
        line = trim(line);

        if (line.empty() || line[0] == '/') break;
        if (line[0] == '#') continue;

        // Parse: elem_id n1 n2 n3 n4 [optional fields]
        std::istringstream iss(line);
        RadiossElement elem;
        elem.part_id = part_id;
        elem.type = ElementType::Shell4;

        Index n1, n2, n3, n4;
        if (iss >> elem.id >> n1 >> n2 >> n3 >> n4) {
            elem.nodes = {n1, n2, n3, n4};
            elements_.push_back(elem);
        }
    }
}

void RadiossReader::parse_sh3n(std::istream& is, Index part_id) {
    std::string line;
    while (std::getline(is, line)) {
        line = trim(line);

        if (line.empty() || line[0] == '/') break;
        if (line[0] == '#') continue;

        // Parse: elem_id n1 n2 n3
        std::istringstream iss(line);
        RadiossElement elem;
        elem.part_id = part_id;
        elem.type = ElementType::Shell3;

        Index n1, n2, n3;
        if (iss >> elem.id >> n1 >> n2 >> n3) {
            elem.nodes = {n1, n2, n3};
            elements_.push_back(elem);
        }
    }
}

void RadiossReader::parse_brick(std::istream& is, Index part_id) {
    std::string line;
    while (std::getline(is, line)) {
        line = trim(line);

        if (line.empty() || line[0] == '/') break;
        if (line[0] == '#') continue;

        // Parse: elem_id n1 n2 n3 n4 n5 n6 n7 n8
        std::istringstream iss(line);
        RadiossElement elem;
        elem.part_id = part_id;
        elem.type = ElementType::Hex8;

        Index n1, n2, n3, n4, n5, n6, n7, n8;
        if (iss >> elem.id >> n1 >> n2 >> n3 >> n4 >> n5 >> n6 >> n7 >> n8) {
            elem.nodes = {n1, n2, n3, n4, n5, n6, n7, n8};
            elements_.push_back(elem);
        }
    }
}

void RadiossReader::parse_beam(std::istream& is, Index part_id) {
    std::string line;
    while (std::getline(is, line)) {
        line = trim(line);

        if (line.empty() || line[0] == '/') break;
        if (line[0] == '#') continue;

        // Parse: elem_id n1 n2 n3 (n3 is orientation node)
        std::istringstream iss(line);
        RadiossElement elem;
        elem.part_id = part_id;
        elem.type = ElementType::Beam2;

        Index n1, n2, n3;
        if (iss >> elem.id >> n1 >> n2 >> n3) {
            elem.nodes = {n1, n2};  // Only store structural nodes
            elements_.push_back(elem);
        }
    }
}

void RadiossReader::parse_spring(std::istream& is, Index part_id) {
    std::string line;
    while (std::getline(is, line)) {
        line = trim(line);

        if (line.empty() || line[0] == '/') break;
        if (line[0] == '#') continue;

        // Parse: elem_id n1 n2
        std::istringstream iss(line);
        RadiossElement elem;
        elem.part_id = part_id;
        elem.type = ElementType::Spring;  // Use Spring for spring elements

        Index n1, n2;
        if (iss >> elem.id >> n1 >> n2) {
            elem.nodes = {n1, n2};
            elements_.push_back(elem);
        }
    }
}

void RadiossReader::parse_mat(std::istream& is, const std::string& law, Index mat_id) {
    RadiossMaterial mat;
    mat.id = mat_id;
    mat.law = law;

    std::string line;

    // Line 1: Material name
    if (std::getline(is, line)) {
        mat.name = trim(line);
    }

    // Line 2: Density
    if (std::getline(is, line)) {
        line = trim(line);
        if (!line.empty() && line[0] != '#') {
            std::istringstream iss(line);
            iss >> mat.density;
        }
    }

    // For LAW1 (elastic): E, nu
    if (law == "LAW1") {
        if (std::getline(is, line)) {
            line = trim(line);
            if (!line.empty() && line[0] != '#') {
                std::istringstream iss(line);
                iss >> mat.E >> mat.nu;
            }
        }
    }
    // For LAW2 (Johnson-Cook): more complex
    else if (law == "LAW2") {
        // Simplified: just read E, nu from next line
        if (std::getline(is, line)) {
            line = trim(line);
            if (!line.empty() && line[0] != '#') {
                std::istringstream iss(line);
                iss >> mat.E >> mat.nu;
            }
        }
    }

    materials_[mat_id] = mat;
}

void RadiossReader::parse_part(std::istream& is, Index part_id) {
    RadiossPart part;
    part.id = part_id;

    std::string line;

    // Line 1: Part name
    if (std::getline(is, line)) {
        part.name = trim(line);
    }

    // Line 2: Property ID, Material ID
    if (std::getline(is, line)) {
        line = trim(line);
        if (!line.empty() && line[0] != '#') {
            std::istringstream iss(line);
            iss >> part.property_id >> part.material_id;
        }
    }

    parts_[part_id] = part;
}

void RadiossReader::parse_bcs(std::istream& is) {
    RadiossBoundaryCondition bc;
    bc.fix_x = bc.fix_y = bc.fix_z = false;
    bc.fix_rx = bc.fix_ry = bc.fix_rz = false;

    std::string line;

    // Skip until we find constraint data
    while (std::getline(is, line)) {
        line = trim(line);

        if (line.empty() || line[0] == '/') break;
        if (line[0] == '#') continue;

        // Parse node IDs and DOF constraints
        // Format varies - simplified parsing
        std::istringstream iss(line);
        Index node_id;
        while (iss >> node_id) {
            bc.node_ids.push_back(node_id);
        }
    }

    if (!bc.node_ids.empty()) {
        bc.name = "BCS_" + std::to_string(bcs_.size());
        bcs_.push_back(bc);
    }
}

void RadiossReader::parse_impvel(std::istream& is) {
    // Imposed velocity - treat as BC
    RadiossBoundaryCondition bc;
    bc.name = "IMPVEL";
    bc.fix_x = bc.fix_y = bc.fix_z = true;

    std::string line;
    while (std::getline(is, line)) {
        line = trim(line);

        if (line.empty() || line[0] == '/') break;
        if (line[0] == '#') continue;

        std::istringstream iss(line);
        Index node_id;
        while (iss >> node_id) {
            bc.node_ids.push_back(node_id);
        }
    }

    if (!bc.node_ids.empty()) {
        bcs_.push_back(bc);
    }
}

void RadiossReader::parse_cload(std::istream& is) {
    RadiossLoad load;
    load.fx = load.fy = load.fz = 0.0;
    load.function_id = 0;

    std::string line;
    while (std::getline(is, line)) {
        line = trim(line);

        if (line.empty() || line[0] == '/') break;
        if (line[0] == '#') continue;

        // Simplified: just collect node IDs
        std::istringstream iss(line);
        Index node_id;
        while (iss >> node_id) {
            load.node_ids.push_back(node_id);
        }
    }

    if (!load.node_ids.empty()) {
        load.name = "CLOAD_" + std::to_string(loads_.size());
        loads_.push_back(load);
    }
}

std::shared_ptr<Mesh> RadiossReader::create_mesh() const {
    if (nodes_.empty()) {
        std::cerr << "RadiossReader: No nodes to create mesh" << std::endl;
        return nullptr;
    }

    auto mesh = std::make_shared<Mesh>(nodes_.size());

    // Set node coordinates
    for (size_t i = 0; i < nodes_.size(); ++i) {
        mesh->set_node_coordinates(i, {nodes_[i].x, nodes_[i].y, nodes_[i].z});
    }

    // Group elements by part and type
    std::map<std::pair<Index, ElementType>, std::vector<const RadiossElement*>> elem_groups;
    for (const auto& elem : elements_) {
        elem_groups[{elem.part_id, elem.type}].push_back(&elem);
    }

    // Create element blocks
    for (const auto& [key, elems] : elem_groups) {
        auto [part_id, elem_type] = key;

        std::string block_name = "part_" + std::to_string(part_id);

        // Get nodes per element
        size_t nodes_per_elem = 0;
        switch (elem_type) {
            case ElementType::Hex8:   nodes_per_elem = 8; break;
            case ElementType::Tet4:   nodes_per_elem = 4; break;
            case ElementType::Shell4: nodes_per_elem = 4; break;
            case ElementType::Shell3: nodes_per_elem = 3; break;
            case ElementType::Beam2:  nodes_per_elem = 2; break;
            case ElementType::Truss: nodes_per_elem = 2; break;
            case ElementType::Spring: nodes_per_elem = 2; break;
            default: continue;
        }

        mesh->add_element_block(block_name, elem_type, elems.size(), nodes_per_elem);
        auto& block = mesh->element_block(mesh->num_element_blocks() - 1);

        for (size_t i = 0; i < elems.size(); ++i) {
            auto elem_nodes = block.element_nodes(i);
            for (size_t j = 0; j < elems[i]->nodes.size(); ++j) {
                // Map Radioss node ID to mesh index
                auto it = node_id_to_index_.find(elems[i]->nodes[j]);
                if (it != node_id_to_index_.end()) {
                    elem_nodes[j] = it->second;
                }
            }
        }
    }

    return mesh;
}

std::vector<Index> RadiossReader::get_node_set(const std::string& name) const {
    for (const auto& bc : bcs_) {
        if (bc.name == name) {
            return bc.node_ids;
        }
    }
    return {};
}

std::vector<Index> RadiossReader::get_element_set(Index part_id) const {
    std::vector<Index> result;
    for (size_t i = 0; i < elements_.size(); ++i) {
        if (elements_[i].part_id == part_id) {
            result.push_back(i);
        }
    }
    return result;
}

physics::MaterialProperties RadiossReader::get_material(Index part_id) const {
    physics::MaterialProperties props;

    auto part_it = parts_.find(part_id);
    if (part_it == parts_.end()) {
        return props;
    }

    auto mat_it = materials_.find(part_it->second.material_id);
    if (mat_it == materials_.end()) {
        return props;
    }

    const auto& mat = mat_it->second;
    props.density = mat.density;
    props.E = mat.E;
    props.nu = mat.nu;
    props.G = mat.E / (2.0 * (1.0 + mat.nu));
    props.K = mat.E / (3.0 * (1.0 - 2.0 * mat.nu));

    return props;
}

void RadiossReader::print_summary(std::ostream& os) const {
    os << "=================================================\n";
    os << "Radioss Model Summary\n";
    os << "=================================================\n";
    os << "Title: " << title_ << "\n";
    os << "Units: " << units_ << "\n";
    os << "\n";
    os << "Nodes: " << nodes_.size() << "\n";
    os << "Elements: " << elements_.size() << "\n";

    // Count by type
    std::map<ElementType, int> type_counts;
    for (const auto& elem : elements_) {
        type_counts[elem.type]++;
    }

    os << "\nElement types:\n";
    for (const auto& [type, count] : type_counts) {
        os << "  " << element_type_to_string(type) << ": " << count << "\n";
    }

    os << "\nMaterials: " << materials_.size() << "\n";
    for (const auto& [id, mat] : materials_) {
        os << "  [" << id << "] " << mat.name << " (" << mat.law << ")\n";
        os << "       density=" << mat.density << ", E=" << mat.E << ", nu=" << mat.nu << "\n";
    }

    os << "\nParts: " << parts_.size() << "\n";
    for (const auto& [id, part] : parts_) {
        os << "  [" << id << "] " << part.name << "\n";
    }

    os << "\nBoundary conditions: " << bcs_.size() << "\n";
    os << "Loads: " << loads_.size() << "\n";
    os << "=================================================\n";
}

// Helper functions

std::string RadiossReader::trim(const std::string& s) const {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::vector<std::string> RadiossReader::split(const std::string& s) const {
    std::vector<std::string> parts;
    std::istringstream iss(s);
    std::string part;
    while (std::getline(iss, part, '/')) {
        part = trim(part);
        if (!part.empty()) {
            parts.push_back(part);
        }
    }
    return parts;
}

bool RadiossReader::starts_with(const std::string& s, const std::string& prefix) const {
    return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

Index RadiossReader::parse_part_id(const std::string& keyword) const {
    // Extract part ID from /KEYWORD/123
    size_t last_slash = keyword.rfind('/');
    if (last_slash != std::string::npos && last_slash + 1 < keyword.size()) {
        try {
            return std::stoul(keyword.substr(last_slash + 1));
        } catch (...) {
            return 0;
        }
    }
    return 0;
}

} // namespace io
} // namespace nxs
