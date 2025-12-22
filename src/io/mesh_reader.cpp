/**
 * @file mesh_reader.cpp
 * @brief Mesh reader implementation
 */

#include <nexussim/io/mesh_reader.hpp>
#include <nexussim/core/logger.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace nxs {
namespace io {

// ============================================================================
// Simple ASCII Mesh Reader
// ============================================================================

std::shared_ptr<Mesh> SimpleMeshReader::read(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw FileIOError("Failed to open mesh file: " + filename);
    }

    NXS_LOG_INFO("Reading mesh file: {}", filename);

    node_sets_.clear();
    element_sets_.clear();

    std::string line;
    std::size_t num_nodes = 0;
    std::vector<Vec3r> node_coords;

    // Storage for element blocks
    struct ElementBlock {
        std::string name;
        ElementType type;
        std::size_t num_elements;
        std::vector<Index> connectivity;
    };
    std::vector<ElementBlock> element_blocks;

    // Parse file
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;

        // Convert to uppercase for case-insensitive comparison
        std::transform(keyword.begin(), keyword.end(), keyword.begin(),
                      [](unsigned char c) { return std::toupper(c); });

        if (keyword == "NODES") {
            // Read number of nodes
            iss >> num_nodes;
            node_coords.reserve(num_nodes);

            NXS_LOG_INFO("Reading {} nodes", num_nodes);

            // Read node coordinates
            for (std::size_t i = 0; i < num_nodes; ++i) {
                if (!std::getline(file, line)) {
                    throw FileIOError("Unexpected end of file while reading nodes");
                }

                std::istringstream node_iss(line);
                Index node_id;
                Real x, y, z;
                node_iss >> node_id >> x >> y >> z;

                if (node_id != i) {
                    NXS_LOG_WARN("Non-sequential node ID: expected {}, got {}", i, node_id);
                }

                node_coords.push_back({x, y, z});
            }

        } else if (keyword == "ELEMENTS") {
            // Read element block
            ElementBlock block;
            std::string type_str;
            iss >> block.name >> type_str >> block.num_elements;

            block.type = parse_element_type(type_str);
            const std::size_t nnodes = nodes_per_element(block.type);

            NXS_LOG_INFO("Reading element block '{}': {} {} elements",
                        block.name, block.num_elements, type_str);

            block.connectivity.reserve(block.num_elements * nnodes);

            // Read element connectivity
            for (std::size_t e = 0; e < block.num_elements; ++e) {
                if (!std::getline(file, line)) {
                    throw FileIOError("Unexpected end of file while reading elements");
                }

                std::istringstream elem_iss(line);
                Index elem_id;
                elem_iss >> elem_id;

                if (elem_id != e) {
                    NXS_LOG_WARN("Non-sequential element ID: expected {}, got {}", e, elem_id);
                }

                // Read node IDs
                for (std::size_t n = 0; n < nnodes; ++n) {
                    Index node_id;
                    elem_iss >> node_id;
                    block.connectivity.push_back(node_id);
                }
            }

            element_blocks.push_back(std::move(block));

        } else if (keyword == "NODESETS" || keyword == "NODESET") {
            // Read node set
            std::string set_name;
            std::size_t num_set_nodes;
            iss >> set_name >> num_set_nodes;

            NXS_LOG_INFO("Reading node set '{}': {} nodes", set_name, num_set_nodes);

            std::vector<Index> node_ids;
            node_ids.reserve(num_set_nodes);

            // Read node IDs (may span multiple lines)
            std::size_t nodes_read = 0;
            while (nodes_read < num_set_nodes && std::getline(file, line)) {
                std::istringstream set_iss(line);
                Index node_id;
                while (set_iss >> node_id && nodes_read < num_set_nodes) {
                    node_ids.push_back(node_id);
                    ++nodes_read;
                }
            }

            node_sets_[set_name] = std::move(node_ids);

        } else if (keyword == "ELEMENTSETS" || keyword == "ELEMENTSET") {
            // Read element set
            std::string set_name;
            std::size_t num_set_elems;
            iss >> set_name >> num_set_elems;

            NXS_LOG_INFO("Reading element set '{}': {} elements", set_name, num_set_elems);

            std::vector<Index> elem_ids;
            elem_ids.reserve(num_set_elems);

            // Read element IDs (may span multiple lines)
            std::size_t elems_read = 0;
            while (elems_read < num_set_elems && std::getline(file, line)) {
                std::istringstream set_iss(line);
                Index elem_id;
                while (set_iss >> elem_id && elems_read < num_set_elems) {
                    elem_ids.push_back(elem_id);
                    ++elems_read;
                }
            }

            element_sets_[set_name] = std::move(elem_ids);

        } else {
            NXS_LOG_WARN("Unknown keyword in mesh file: {}", keyword);
        }
    }

    file.close();

    // Create mesh
    auto mesh = std::make_shared<Mesh>(num_nodes);

    // Set node coordinates
    for (std::size_t i = 0; i < num_nodes; ++i) {
        mesh->set_node_coordinates(i, node_coords[i]);
    }

    // Add element blocks
    for (const auto& block : element_blocks) {
        const std::size_t nnodes = nodes_per_element(block.type);
        mesh->add_element_block(block.name, block.type,
                               block.num_elements, nnodes);

        auto& mesh_block = mesh->element_block(mesh->num_element_blocks() - 1);

        // Copy connectivity data
        for (std::size_t i = 0; i < block.connectivity.size(); ++i) {
            mesh_block.connectivity[i] = block.connectivity[i];
        }
    }

    NXS_LOG_INFO("Mesh loaded successfully:");
    NXS_LOG_INFO("  Nodes: {}", num_nodes);
    NXS_LOG_INFO("  Element blocks: {}", element_blocks.size());
    NXS_LOG_INFO("  Node sets: {}", node_sets_.size());
    NXS_LOG_INFO("  Element sets: {}", element_sets_.size());

    return mesh;
}

ElementType SimpleMeshReader::parse_element_type(const std::string& type_str) {
    std::string upper = type_str;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                  [](unsigned char c) { return std::toupper(c); });

    if (upper == "HEX8") return ElementType::Hex8;
    if (upper == "TET4") return ElementType::Tet4;
    if (upper == "SHELL4" || upper == "QUAD4") return ElementType::Shell4;
    if (upper == "SHELL3" || upper == "TRI3") return ElementType::Shell3;
    if (upper == "BEAM2" || upper == "LINE2") return ElementType::Beam2;

    throw InvalidArgumentError("Unknown element type: " + type_str);
}

std::size_t SimpleMeshReader::nodes_per_element(ElementType type) {
    switch (type) {
        case ElementType::Hex8: return 8;
        case ElementType::Hex20: return 20;
        case ElementType::Hex27: return 27;
        case ElementType::Tet4: return 4;
        case ElementType::Tet10: return 10;
        case ElementType::Shell4: return 4;
        case ElementType::Shell3: return 3;
        case ElementType::Beam2: return 2;
        case ElementType::Beam3: return 3;
        default:
            throw InvalidArgumentError("Cannot determine nodes per element for this type");
    }
}

// ============================================================================
// Gmsh Reader (Placeholder)
// ============================================================================

std::shared_ptr<Mesh> GmshReader::read(const std::string& filename) {
    throw NotImplementedError("Gmsh reader not yet implemented");
}

} // namespace io
} // namespace nxs
