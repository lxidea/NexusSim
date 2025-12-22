#pragma once

/**
 * @file mesh_reader.hpp
 * @brief Mesh readers for various file formats
 *
 * Supported formats:
 * - Simple ASCII format (.mesh)
 * - Gmsh format (.msh) - future
 * - Exodus format (.exo) - future
 */

#include <nexussim/core/core.hpp>
#include <nexussim/data/mesh.hpp>
#include <string>
#include <vector>
#include <map>

namespace nxs {
namespace io {

/**
 * @brief Simple ASCII mesh reader
 *
 * File format:
 * ```
 * # Comments start with #
 * NODES <num_nodes>
 * <node_id> <x> <y> <z>
 * ...
 *
 * ELEMENTS <block_name> <element_type> <num_elements>
 * <elem_id> <node1> <node2> ... <nodeN>
 * ...
 *
 * NODESETS <set_name> <num_nodes>
 * <node_id1> <node_id2> ...
 * ```
 *
 * Element types: HEX8, TET4, SHELL4, SHELL3, BEAM2
 */
class SimpleMeshReader {
public:
    SimpleMeshReader() = default;

    /**
     * @brief Read mesh from file
     * @param filename Path to mesh file
     * @return Shared pointer to mesh
     */
    std::shared_ptr<Mesh> read(const std::string& filename);

    /**
     * @brief Get node sets defined in file
     * @return Map of node set name to node IDs
     */
    const std::map<std::string, std::vector<Index>>& node_sets() const {
        return node_sets_;
    }

    /**
     * @brief Get element sets defined in file
     * @return Map of element set name to element IDs
     */
    const std::map<std::string, std::vector<Index>>& element_sets() const {
        return element_sets_;
    }

private:
    /**
     * @brief Parse element type string
     */
    ElementType parse_element_type(const std::string& type_str);

    /**
     * @brief Get number of nodes per element
     */
    std::size_t nodes_per_element(ElementType type);

    std::map<std::string, std::vector<Index>> node_sets_;
    std::map<std::string, std::vector<Index>> element_sets_;
};

/**
 * @brief Gmsh mesh reader (future implementation)
 */
class GmshReader {
public:
    GmshReader() = default;

    /**
     * @brief Read Gmsh format (.msh)
     */
    std::shared_ptr<Mesh> read(const std::string& filename);

private:
    // TODO: Implement Gmsh format parsing
};

} // namespace io
} // namespace nxs
