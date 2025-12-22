#pragma once

#include <nexussim/core/core.hpp>
#include <nexussim/data/field.hpp>
#include <map>
#include <set>
#include <optional>

namespace nxs {

// ============================================================================
// Mesh - Core Geometric and Topological Data Structure
// ============================================================================

class Mesh {
public:
    // ========================================================================
    // Node Data
    // ========================================================================

    struct NodeData {
        Field<Real> coordinates;      // 3 components: x, y, z
        Field<Int> global_ids;        // Global node IDs
        std::set<Index> boundary_nodes;  // Nodes on boundary

        explicit NodeData(std::size_t num_nodes)
            : coordinates("coordinates", FieldType::Vector, FieldLocation::Node, num_nodes, 3)
            , global_ids("global_ids", FieldType::Scalar, FieldLocation::Node, num_nodes, 1)
        {}
    };

    // ========================================================================
    // Element Data
    // ========================================================================

    struct ElementBlock {
        std::string name;
        ElementType type;
        std::size_t num_nodes_per_elem;
        std::size_t num_elems_;                // Number of elements in this block
        Field<Index> connectivity;    // Flat array: [elem0_node0, elem0_node1, ..., elem1_node0, ...]
        Field<Int> global_ids;        // Global element IDs
        Field<Index> material_ids;    // Material ID for each element
        Field<Index> part_ids;        // Part ID for each element

        ElementBlock(std::string block_name,
                     ElementType elem_type,
                     std::size_t num_elems,
                     std::size_t nodes_per_elem)
            : name(std::move(block_name))
            , type(elem_type)
            , num_nodes_per_elem(nodes_per_elem)
            , num_elems_(num_elems)
            // Connectivity as flat array: total size = num_elems * nodes_per_elem, 1 component (Scalar)
            , connectivity("connectivity", FieldType::Scalar, FieldLocation::Element,
                          num_elems * nodes_per_elem, 1)
            , global_ids("global_ids", FieldType::Scalar, FieldLocation::Element, num_elems, 1)
            , material_ids("material_ids", FieldType::Scalar, FieldLocation::Element, num_elems, 1)
            , part_ids("part_ids", FieldType::Scalar, FieldLocation::Element, num_elems, 1)
        {}

        std::size_t num_elements() const {
            return num_elems_;
        }

        // Get connectivity for a specific element
        std::span<Index> element_nodes(std::size_t elem_id) {
            NXS_CHECK_RANGE(elem_id, num_elements());
            return connectivity.component(0).subspan(elem_id * num_nodes_per_elem, num_nodes_per_elem);
        }

        std::span<const Index> element_nodes(std::size_t elem_id) const {
            NXS_CHECK_RANGE(elem_id, num_elements());
            return connectivity.component(0).subspan(elem_id * num_nodes_per_elem, num_nodes_per_elem);
        }
    };

    // ========================================================================
    // Constructors
    // ========================================================================

    Mesh() : nodes_(0) {}

    explicit Mesh(std::size_t num_nodes)
        : nodes_(num_nodes)
    {
        NXS_LOG_DEBUG("Created mesh with {} nodes", num_nodes);
    }

    // ========================================================================
    // Node Operations
    // ========================================================================

    std::size_t num_nodes() const { return nodes_.coordinates.num_entities(); }

    void set_node_coordinates(std::size_t node_id, const Vec3r& coords) {
        NXS_CHECK_RANGE(node_id, num_nodes());
        nodes_.coordinates.set_vec(node_id, coords);
    }

    Vec3r get_node_coordinates(std::size_t node_id) const {
        NXS_CHECK_RANGE(node_id, num_nodes());
        return nodes_.coordinates.get_vec<3>(node_id);
    }

    Field<Real>& coordinates() { return nodes_.coordinates; }
    const Field<Real>& coordinates() const { return nodes_.coordinates; }

    // ========================================================================
    // Element Block Operations
    // ========================================================================

    std::size_t num_element_blocks() const { return element_blocks_.size(); }

    std::size_t num_elements() const {
        std::size_t total = 0;
        for (const auto& block : element_blocks_) {
            total += block.num_elements();
        }
        return total;
    }

    // Add a new element block
    Index add_element_block(const std::string& name,
                           ElementType type,
                           std::size_t num_elems,
                           std::size_t nodes_per_elem) {
        element_blocks_.emplace_back(name, type, num_elems, nodes_per_elem);
        Index block_id = element_blocks_.size() - 1;

        NXS_LOG_DEBUG("Added element block '{}' with {} elements (type: {})",
                     name, num_elems, to_string(type));

        return block_id;
    }

    // Get element block by index
    ElementBlock& element_block(Index block_id) {
        NXS_CHECK_RANGE(block_id, element_blocks_.size());
        return element_blocks_[block_id];
    }

    const ElementBlock& element_block(Index block_id) const {
        NXS_CHECK_RANGE(block_id, element_blocks_.size());
        return element_blocks_[block_id];
    }

    // Find element block by name
    std::optional<Index> find_element_block(const std::string& name) const {
        for (std::size_t i = 0; i < element_blocks_.size(); ++i) {
            if (element_blocks_[i].name == name) {
                return i;
            }
        }
        return std::nullopt;
    }

    // Access all element blocks
    std::vector<ElementBlock>& element_blocks() { return element_blocks_; }
    const std::vector<ElementBlock>& element_blocks() const { return element_blocks_; }

    // ========================================================================
    // Node Sets and Side Sets
    // ========================================================================

    void add_node_set(const std::string& name, const std::vector<Index>& node_ids) {
        node_sets_[name] = node_ids;
        NXS_LOG_DEBUG("Added node set '{}' with {} nodes", name, node_ids.size());
    }

    std::optional<std::vector<Index>> get_node_set(const std::string& name) const {
        auto it = node_sets_.find(name);
        if (it != node_sets_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    const std::map<std::string, std::vector<Index>>& node_sets() const {
        return node_sets_;
    }

    void add_side_set(const std::string& name,
                      const std::vector<std::pair<Index, Index>>& elem_side_pairs) {
        side_sets_[name] = elem_side_pairs;
        NXS_LOG_DEBUG("Added side set '{}' with {} sides", name, elem_side_pairs.size());
    }

    std::optional<std::vector<std::pair<Index, Index>>>
    get_side_set(const std::string& name) const {
        auto it = side_sets_.find(name);
        if (it != side_sets_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    // ========================================================================
    // Bounding Box and Geometry Info
    // ========================================================================

    struct BoundingBox {
        Vec3r min;
        Vec3r max;

        Real volume() const {
            return (max[0] - min[0]) * (max[1] - min[1]) * (max[2] - min[2]);
        }

        Vec3r center() const {
            return {
                (min[0] + max[0]) / 2.0,
                (min[1] + max[1]) / 2.0,
                (min[2] + max[2]) / 2.0
            };
        }
    };

    BoundingBox compute_bounding_box() const {
        if (num_nodes() == 0) {
            return BoundingBox{{0, 0, 0}, {0, 0, 0}};
        }

        auto x = nodes_.coordinates.component(0);
        auto y = nodes_.coordinates.component(1);
        auto z = nodes_.coordinates.component(2);

        BoundingBox bbox;
        bbox.min[0] = *std::min_element(x.begin(), x.end());
        bbox.min[1] = *std::min_element(y.begin(), y.end());
        bbox.min[2] = *std::min_element(z.begin(), z.end());

        bbox.max[0] = *std::max_element(x.begin(), x.end());
        bbox.max[1] = *std::max_element(y.begin(), y.end());
        bbox.max[2] = *std::max_element(z.begin(), z.end());

        return bbox;
    }

    // ========================================================================
    // I/O Operations (to be implemented)
    // ========================================================================

    static Mesh from_file(const std::string& filename) {
        NXS_NOT_IMPLEMENTED_MSG("Mesh::from_file");
    }

    void write_to_file(const std::string& filename) const {
        NXS_NOT_IMPLEMENTED_MSG("Mesh::write_to_file");
    }

    // ========================================================================
    // Statistics and Info
    // ========================================================================

    void print_info() const {
        NXS_LOG_INFO("Mesh Information:");
        NXS_LOG_INFO("  Nodes: {}", num_nodes());
        NXS_LOG_INFO("  Element blocks: {}", num_element_blocks());
        NXS_LOG_INFO("  Total elements: {}", num_elements());

        for (std::size_t i = 0; i < element_blocks_.size(); ++i) {
            const auto& block = element_blocks_[i];
            NXS_LOG_INFO("    Block {}: '{}' ({} {}-node {} elements)",
                        i, block.name, block.num_elements(),
                        block.num_nodes_per_elem, to_string(block.type));
        }

        NXS_LOG_INFO("  Node sets: {}", node_sets_.size());
        NXS_LOG_INFO("  Side sets: {}", side_sets_.size());

        auto bbox = compute_bounding_box();
        NXS_LOG_INFO("  Bounding box:");
        NXS_LOG_INFO("    Min: ({:.3f}, {:.3f}, {:.3f})", bbox.min[0], bbox.min[1], bbox.min[2]);
        NXS_LOG_INFO("    Max: ({:.3f}, {:.3f}, {:.3f})", bbox.max[0], bbox.max[1], bbox.max[2]);
        NXS_LOG_INFO("    Center: ({:.3f}, {:.3f}, {:.3f})",
                    bbox.center()[0], bbox.center()[1], bbox.center()[2]);
    }

private:
    NodeData nodes_;
    std::vector<ElementBlock> element_blocks_;
    std::map<std::string, std::vector<Index>> node_sets_;
    std::map<std::string, std::vector<std::pair<Index, Index>>> side_sets_;
};

} // namespace nxs
