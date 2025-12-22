#pragma once

/**
 * @file radioss_reader.hpp
 * @brief OpenRadioss input deck reader
 *
 * Parses OpenRadioss Starter (.rad) format files and creates NexusSim mesh objects.
 *
 * Supported keywords:
 * - /BEGIN - Model start
 * - /NODE - Node definitions
 * - /SHELL/part_id - 4-node shell elements
 * - /BRICK/part_id - 8-node solid (hex) elements
 * - /SH3N/part_id - 3-node shell elements
 * - /BEAM/part_id - Beam elements
 * - /SPRING/part_id - Spring elements
 * - /MAT/LAW* - Material definitions
 * - /PROP/TYPE* - Property definitions
 * - /PART/part_id - Part definitions
 * - /BCS - Boundary conditions
 * - /IMPVEL, /IMPDISP - Imposed velocity/displacement
 * - /CLOAD - Concentrated loads
 *
 * File format reference: OpenRadioss Documentation
 */

#include <nexussim/core/core.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/physics/material.hpp>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <memory>
#include <functional>

namespace nxs {
namespace io {

// Forward declarations
struct RadiossNode;
struct RadiossElement;
struct RadiossMaterial;
struct RadiossPart;
struct RadiossBoundaryCondition;
struct RadiossLoad;

/**
 * @brief Radioss node data
 */
struct RadiossNode {
    Index id;
    Real x, y, z;
};

/**
 * @brief Radioss element data
 */
struct RadiossElement {
    Index id;
    Index part_id;
    ElementType type;
    std::vector<Index> nodes;
};

/**
 * @brief Radioss material data
 */
struct RadiossMaterial {
    Index id;
    std::string name;
    std::string law;   // e.g., "LAW1" (elastic), "LAW2" (Johnson-Cook)
    Real density;
    Real E;            // Young's modulus
    Real nu;           // Poisson's ratio

    // Johnson-Cook parameters (LAW2)
    Real A, B, n_jc;   // Yield stress parameters
    Real C;            // Strain rate sensitivity
    Real m;            // Temperature softening

    // Plastic failure strain
    Real eps_max;
};

/**
 * @brief Radioss part data (links elements to materials/properties)
 */
struct RadiossPart {
    Index id;
    std::string name;
    Index property_id;
    Index material_id;
};

/**
 * @brief Radioss boundary condition
 */
struct RadiossBoundaryCondition {
    std::string name;
    std::vector<Index> node_ids;
    bool fix_x, fix_y, fix_z;
    bool fix_rx, fix_ry, fix_rz;
};

/**
 * @brief Radioss concentrated load
 */
struct RadiossLoad {
    std::string name;
    std::vector<Index> node_ids;
    Real fx, fy, fz;
    Index function_id;  // Time function reference
};

/**
 * @brief OpenRadioss input deck reader
 *
 * Usage:
 * ```cpp
 * RadiossReader reader;
 * reader.read("model_0000.rad");
 *
 * auto mesh = reader.create_mesh();
 * auto materials = reader.materials();
 * auto bcs = reader.boundary_conditions();
 * ```
 */
class RadiossReader {
public:
    RadiossReader() = default;

    /**
     * @brief Read Radioss input deck
     * @param filename Path to .rad file
     * @return true on success
     */
    bool read(const std::string& filename);

    /**
     * @brief Create NexusSim mesh from parsed data
     * @return Shared pointer to mesh
     */
    std::shared_ptr<Mesh> create_mesh() const;

    // Accessors for parsed data
    const std::vector<RadiossNode>& nodes() const { return nodes_; }
    const std::vector<RadiossElement>& elements() const { return elements_; }
    const std::map<Index, RadiossMaterial>& materials() const { return materials_; }
    const std::map<Index, RadiossPart>& parts() const { return parts_; }
    const std::vector<RadiossBoundaryCondition>& boundary_conditions() const { return bcs_; }
    const std::vector<RadiossLoad>& loads() const { return loads_; }

    /**
     * @brief Get node set by name
     */
    std::vector<Index> get_node_set(const std::string& name) const;

    /**
     * @brief Get element set by part ID
     */
    std::vector<Index> get_element_set(Index part_id) const;

    /**
     * @brief Get material properties for a part
     */
    physics::MaterialProperties get_material(Index part_id) const;

    /**
     * @brief Get model title
     */
    const std::string& title() const { return title_; }

    /**
     * @brief Get unit system
     */
    const std::string& units() const { return units_; }

    /**
     * @brief Print summary of parsed model
     */
    void print_summary(std::ostream& os = std::cout) const;

private:
    // Parsing functions
    void parse_line(const std::string& line);
    void parse_begin(std::istream& is);
    void parse_title(std::istream& is);
    void parse_node(std::istream& is);
    void parse_shell(std::istream& is, Index part_id);
    void parse_sh3n(std::istream& is, Index part_id);
    void parse_brick(std::istream& is, Index part_id);
    void parse_beam(std::istream& is, Index part_id);
    void parse_spring(std::istream& is, Index part_id);
    void parse_mat(std::istream& is, const std::string& law, Index mat_id);
    void parse_part(std::istream& is, Index part_id);
    void parse_bcs(std::istream& is);
    void parse_impvel(std::istream& is);
    void parse_cload(std::istream& is);

    // Helper functions
    std::string trim(const std::string& s) const;
    std::vector<std::string> split(const std::string& s) const;
    bool starts_with(const std::string& s, const std::string& prefix) const;
    Index parse_part_id(const std::string& keyword) const;

    // Parsed data
    std::string title_;
    std::string units_;

    std::vector<RadiossNode> nodes_;
    std::vector<RadiossElement> elements_;
    std::map<Index, RadiossMaterial> materials_;
    std::map<Index, RadiossPart> parts_;
    std::vector<RadiossBoundaryCondition> bcs_;
    std::vector<RadiossLoad> loads_;

    // Node ID to index mapping
    std::map<Index, Index> node_id_to_index_;
};

} // namespace io
} // namespace nxs
