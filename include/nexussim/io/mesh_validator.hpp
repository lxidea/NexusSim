#pragma once

/**
 * @file mesh_validator.hpp
 * @brief Mesh validation and quality checking utilities
 *
 * Performs various checks on finite element meshes:
 * - Node connectivity (orphan node detection)
 * - Element Jacobian sign (positive volume check)
 * - Duplicate node detection
 * - Element quality metrics (aspect ratio, skewness)
 * - Manifold surface check (for contact surfaces)
 */

#include <nexussim/core/core.hpp>
#include <nexussim/data/mesh.hpp>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <span>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace nxs {
namespace io {

/**
 * @brief Result of a validation check
 */
struct ValidationResult {
    bool passed;
    std::string message;
    std::vector<Index> affected_indices;  // Node or element indices
};

/**
 * @brief Element quality metrics
 */
struct ElementQuality {
    Index element_id;
    Real jacobian;           // Jacobian determinant (volume proxy)
    Real aspect_ratio;       // Max edge / min edge
    Real skewness;           // Deviation from ideal shape [0-1]
    Real min_angle;          // Minimum interior angle (degrees)
    Real max_angle;          // Maximum interior angle (degrees)
};

/**
 * @brief Mesh validation options
 */
struct ValidationOptions {
    Real duplicate_tolerance = 1e-10;   // Distance tolerance for duplicate nodes
    Real min_jacobian = 0.0;            // Minimum allowed Jacobian
    Real max_aspect_ratio = 100.0;      // Maximum allowed aspect ratio
    Real max_skewness = 0.95;           // Maximum allowed skewness [0-1]
    bool check_manifold = true;         // Check for non-manifold edges
    bool verbose = false;               // Print detailed output
};

/**
 * @brief Mesh validation summary
 */
struct ValidationSummary {
    bool is_valid;
    size_t total_nodes;
    size_t total_elements;

    size_t orphan_nodes;
    size_t duplicate_nodes;
    size_t negative_jacobian_elements;
    size_t poor_quality_elements;
    size_t non_manifold_edges;

    std::vector<ValidationResult> results;

    void print(std::ostream& os = std::cout) const {
        os << "=================================================\n";
        os << "Mesh Validation Summary\n";
        os << "=================================================\n";
        os << "Nodes: " << total_nodes << "\n";
        os << "Elements: " << total_elements << "\n";
        os << "\n";
        os << "Validation Results:\n";
        os << "  Orphan nodes: " << orphan_nodes;
        os << (orphan_nodes > 0 ? " [WARNING]" : " [OK]") << "\n";
        os << "  Duplicate nodes: " << duplicate_nodes;
        os << (duplicate_nodes > 0 ? " [WARNING]" : " [OK]") << "\n";
        os << "  Negative Jacobian: " << negative_jacobian_elements;
        os << (negative_jacobian_elements > 0 ? " [ERROR]" : " [OK]") << "\n";
        os << "  Poor quality: " << poor_quality_elements;
        os << (poor_quality_elements > 0 ? " [WARNING]" : " [OK]") << "\n";
        os << "  Non-manifold: " << non_manifold_edges;
        os << (non_manifold_edges > 0 ? " [WARNING]" : " [OK]") << "\n";
        os << "\n";
        os << "Overall: " << (is_valid ? "VALID" : "INVALID") << "\n";
        os << "=================================================\n";
    }
};

/**
 * @brief Mesh validator class
 *
 * Usage:
 * ```cpp
 * MeshValidator validator;
 * ValidationSummary summary = validator.validate(mesh);
 * summary.print();
 *
 * if (!summary.is_valid) {
 *     // Handle invalid mesh
 * }
 * ```
 */
class MeshValidator {
public:
    MeshValidator() = default;
    explicit MeshValidator(const ValidationOptions& opts) : options_(opts) {}

    /**
     * @brief Run all validation checks
     */
    ValidationSummary validate(const Mesh& mesh) const {
        ValidationSummary summary;
        summary.total_nodes = mesh.num_nodes();
        summary.total_elements = mesh.num_elements();

        // Run individual checks
        auto orphan_result = check_orphan_nodes(mesh);
        auto duplicate_result = check_duplicate_nodes(mesh);
        auto jacobian_result = check_jacobian(mesh);
        auto quality_result = check_element_quality(mesh);

        summary.orphan_nodes = orphan_result.affected_indices.size();
        summary.duplicate_nodes = duplicate_result.affected_indices.size();
        summary.negative_jacobian_elements = jacobian_result.affected_indices.size();
        summary.poor_quality_elements = quality_result.affected_indices.size();

        summary.results.push_back(orphan_result);
        summary.results.push_back(duplicate_result);
        summary.results.push_back(jacobian_result);
        summary.results.push_back(quality_result);

        if (options_.check_manifold) {
            auto manifold_result = check_manifold(mesh);
            summary.non_manifold_edges = manifold_result.affected_indices.size();
            summary.results.push_back(manifold_result);
        } else {
            summary.non_manifold_edges = 0;
        }

        // Determine overall validity (negative Jacobian is critical)
        summary.is_valid = (summary.negative_jacobian_elements == 0);

        return summary;
    }

    /**
     * @brief Check for orphan nodes (nodes not connected to any element)
     */
    ValidationResult check_orphan_nodes(const Mesh& mesh) const {
        ValidationResult result;
        result.message = "Orphan node check";

        std::set<Index> connected_nodes;

        // Collect all nodes referenced by elements
        for (size_t b = 0; b < mesh.num_element_blocks(); ++b) {
            const auto& block = mesh.element_block(b);
            for (size_t e = 0; e < block.num_elements(); ++e) {
                auto nodes = block.element_nodes(e);
                for (size_t n = 0; n < nodes.size(); ++n) {
                    connected_nodes.insert(nodes[n]);
                }
            }
        }

        // Find orphan nodes
        for (Index i = 0; i < mesh.num_nodes(); ++i) {
            if (connected_nodes.find(i) == connected_nodes.end()) {
                result.affected_indices.push_back(i);
            }
        }

        result.passed = result.affected_indices.empty();
        if (!result.passed) {
            result.message += ": Found " + std::to_string(result.affected_indices.size()) + " orphan nodes";
        }

        return result;
    }

    /**
     * @brief Check for duplicate nodes (nodes at same location)
     */
    ValidationResult check_duplicate_nodes(const Mesh& mesh) const {
        ValidationResult result;
        result.message = "Duplicate node check";

        const Real tol = options_.duplicate_tolerance;
        const Real tol_sq = tol * tol;

        // Simple O(n^2) check - could be optimized with spatial hashing
        for (Index i = 0; i < mesh.num_nodes(); ++i) {
            Vec3r coord_i = mesh.get_node_coordinates(i);
            for (Index j = i + 1; j < mesh.num_nodes(); ++j) {
                Vec3r coord_j = mesh.get_node_coordinates(j);
                Real dx = coord_i[0] - coord_j[0];
                Real dy = coord_i[1] - coord_j[1];
                Real dz = coord_i[2] - coord_j[2];
                Real dist_sq = dx*dx + dy*dy + dz*dz;

                if (dist_sq < tol_sq) {
                    result.affected_indices.push_back(i);
                    result.affected_indices.push_back(j);
                }
            }
        }

        result.passed = result.affected_indices.empty();
        if (!result.passed) {
            result.message += ": Found " + std::to_string(result.affected_indices.size() / 2) + " duplicate node pairs";
        }

        return result;
    }

    /**
     * @brief Check element Jacobian (positive volume)
     */
    ValidationResult check_jacobian(const Mesh& mesh) const {
        ValidationResult result;
        result.message = "Jacobian check";

        Index elem_idx = 0;

        for (size_t b = 0; b < mesh.num_element_blocks(); ++b) {
            const auto& block = mesh.element_block(b);
            ElementType type = block.type;

            for (size_t e = 0; e < block.num_elements(); ++e) {
                auto nodes = block.element_nodes(e);
                Real jac = compute_jacobian(mesh, nodes, type);

                if (jac <= options_.min_jacobian) {
                    result.affected_indices.push_back(elem_idx);
                }
                elem_idx++;
            }
        }

        result.passed = result.affected_indices.empty();
        if (!result.passed) {
            result.message += ": Found " + std::to_string(result.affected_indices.size()) +
                             " elements with non-positive Jacobian";
        }

        return result;
    }

    /**
     * @brief Check element quality metrics
     */
    ValidationResult check_element_quality(const Mesh& mesh) const {
        ValidationResult result;
        result.message = "Element quality check";

        Index elem_idx = 0;

        for (size_t b = 0; b < mesh.num_element_blocks(); ++b) {
            const auto& block = mesh.element_block(b);
            ElementType type = block.type;

            for (size_t e = 0; e < block.num_elements(); ++e) {
                auto nodes = block.element_nodes(e);
                ElementQuality quality = compute_quality(mesh, nodes, type, elem_idx);

                bool poor = false;
                if (quality.aspect_ratio > options_.max_aspect_ratio) poor = true;
                if (quality.skewness > options_.max_skewness) poor = true;

                if (poor) {
                    result.affected_indices.push_back(elem_idx);
                }
                elem_idx++;
            }
        }

        result.passed = result.affected_indices.empty();
        if (!result.passed) {
            result.message += ": Found " + std::to_string(result.affected_indices.size()) +
                             " poor quality elements";
        }

        return result;
    }

    /**
     * @brief Check for non-manifold edges (edge shared by more than 2 faces)
     */
    ValidationResult check_manifold(const Mesh& mesh) const {
        ValidationResult result;
        result.message = "Manifold check";

        // Build edge-to-face map for shell elements
        std::map<std::pair<Index, Index>, int> edge_count;

        for (size_t b = 0; b < mesh.num_element_blocks(); ++b) {
            const auto& block = mesh.element_block(b);
            ElementType type = block.type;

            // Only check shell/2D elements
            if (type != ElementType::Shell3 && type != ElementType::Shell4) {
                continue;
            }

            for (size_t e = 0; e < block.num_elements(); ++e) {
                auto nodes = block.element_nodes(e);
                size_t nn = nodes.size();

                for (size_t i = 0; i < nn; ++i) {
                    Index n1 = nodes[i];
                    Index n2 = nodes[(i + 1) % nn];

                    // Canonical edge representation (smaller index first)
                    auto edge = std::make_pair(std::min(n1, n2), std::max(n1, n2));
                    edge_count[edge]++;
                }
            }
        }

        // Find non-manifold edges (count > 2)
        for (const auto& [edge, count] : edge_count) {
            if (count > 2) {
                result.affected_indices.push_back(edge.first);
                result.affected_indices.push_back(edge.second);
            }
        }

        result.passed = result.affected_indices.empty();
        if (!result.passed) {
            result.message += ": Found " + std::to_string(result.affected_indices.size() / 2) +
                             " non-manifold edges";
        }

        return result;
    }

    /**
     * @brief Compute quality metrics for all elements
     */
    std::vector<ElementQuality> compute_all_quality(const Mesh& mesh) const {
        std::vector<ElementQuality> qualities;
        Index elem_idx = 0;

        for (size_t b = 0; b < mesh.num_element_blocks(); ++b) {
            const auto& block = mesh.element_block(b);
            ElementType type = block.type;

            for (size_t e = 0; e < block.num_elements(); ++e) {
                auto nodes = block.element_nodes(e);
                qualities.push_back(compute_quality(mesh, nodes, type, elem_idx));
                elem_idx++;
            }
        }

        return qualities;
    }

    // Options accessor
    ValidationOptions& options() { return options_; }
    const ValidationOptions& options() const { return options_; }

private:
    ValidationOptions options_;

    /**
     * @brief Compute Jacobian for an element (simplified)
     */
    Real compute_jacobian(const Mesh& mesh,
                         std::span<const Index> nodes,
                         ElementType type) const {
        // Simplified Jacobian computation at element center
        size_t nn = nodes.size();

        if (type == ElementType::Hex8 && nn >= 8) {
            // Hex8: Use triple scalar product approximation
            Real x[8], y[8], z[8];
            for (int i = 0; i < 8; ++i) {
                Vec3r coord = mesh.get_node_coordinates(nodes[i]);
                x[i] = coord[0];
                y[i] = coord[1];
                z[i] = coord[2];
            }

            // Compute vectors along edges from node 0
            Real ax = x[1] - x[0], ay = y[1] - y[0], az = z[1] - z[0];
            Real bx = x[3] - x[0], by = y[3] - y[0], bz = z[3] - z[0];
            Real cx = x[4] - x[0], cy = y[4] - y[0], cz = z[4] - z[0];

            // Triple scalar product (a . (b x c))
            return ax*(by*cz - bz*cy) - ay*(bx*cz - bz*cx) + az*(bx*cy - by*cx);
        }
        else if (type == ElementType::Tet4 && nn >= 4) {
            // Tet4: Volume = (1/6) * |a . (b x c)|
            Real x[4], y[4], z[4];
            for (int i = 0; i < 4; ++i) {
                Vec3r coord = mesh.get_node_coordinates(nodes[i]);
                x[i] = coord[0];
                y[i] = coord[1];
                z[i] = coord[2];
            }

            Real ax = x[1] - x[0], ay = y[1] - y[0], az = z[1] - z[0];
            Real bx = x[2] - x[0], by = y[2] - y[0], bz = z[2] - z[0];
            Real cx = x[3] - x[0], cy = y[3] - y[0], cz = z[3] - z[0];

            return (ax*(by*cz - bz*cy) - ay*(bx*cz - bz*cx) + az*(bx*cy - by*cx)) / 6.0;
        }
        else if ((type == ElementType::Shell4 || type == ElementType::Shell3) && nn >= 3) {
            // Shell: Compute area using cross product
            Real x[4], y[4], z[4];
            for (size_t i = 0; i < nn && i < 4; ++i) {
                Vec3r coord = mesh.get_node_coordinates(nodes[i]);
                x[i] = coord[0];
                y[i] = coord[1];
                z[i] = coord[2];
            }

            Real ax = x[1] - x[0], ay = y[1] - y[0], az = z[1] - z[0];
            Real bx = x[2] - x[0], by = y[2] - y[0], bz = z[2] - z[0];

            // Cross product
            Real cx = ay*bz - az*by;
            Real cy = az*bx - ax*bz;
            Real cz = ax*by - ay*bx;

            return std::sqrt(cx*cx + cy*cy + cz*cz) / 2.0;
        }

        // Default: return positive value (assume valid)
        return 1.0;
    }

    /**
     * @brief Compute quality metrics for an element
     */
    ElementQuality compute_quality(const Mesh& mesh,
                                  std::span<const Index> nodes,
                                  ElementType type,
                                  Index elem_id) const {
        ElementQuality quality;
        quality.element_id = elem_id;

        size_t nn = nodes.size();
        if (nn < 2) {
            quality.jacobian = 1.0;
            quality.aspect_ratio = 1.0;
            quality.skewness = 0.0;
            quality.min_angle = 90.0;
            quality.max_angle = 90.0;
            return quality;
        }

        // Compute Jacobian
        quality.jacobian = compute_jacobian(mesh, nodes, type);

        // Compute edge lengths
        std::vector<Real> edge_lengths;
        for (size_t i = 0; i < nn; ++i) {
            Vec3r coord1 = mesh.get_node_coordinates(nodes[i]);
            Vec3r coord2 = mesh.get_node_coordinates(nodes[(i + 1) % nn]);

            Real dx = coord2[0] - coord1[0];
            Real dy = coord2[1] - coord1[1];
            Real dz = coord2[2] - coord1[2];
            edge_lengths.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
        }

        // Aspect ratio
        Real min_edge = *std::min_element(edge_lengths.begin(), edge_lengths.end());
        Real max_edge = *std::max_element(edge_lengths.begin(), edge_lengths.end());
        quality.aspect_ratio = (min_edge > 1e-15) ? max_edge / min_edge : 1e10;

        // Skewness (simplified - based on aspect ratio)
        // 0 = perfect, 1 = degenerate
        quality.skewness = 1.0 - 1.0 / quality.aspect_ratio;
        if (quality.skewness < 0) quality.skewness = 0;
        if (quality.skewness > 1) quality.skewness = 1;

        // Angles (simplified for quads/triangles)
        quality.min_angle = 60.0;  // Ideal triangle
        quality.max_angle = 120.0;

        if (type == ElementType::Shell4 && nn >= 4) {
            // Compute actual angles for quads
            Real angles[4];
            for (int i = 0; i < 4; ++i) {
                Vec3r c0 = mesh.get_node_coordinates(nodes[(i + 3) % 4]);
                Vec3r c1 = mesh.get_node_coordinates(nodes[i]);
                Vec3r c2 = mesh.get_node_coordinates(nodes[(i + 1) % 4]);

                Real v1x = c0[0] - c1[0];
                Real v1y = c0[1] - c1[1];
                Real v1z = c0[2] - c1[2];

                Real v2x = c2[0] - c1[0];
                Real v2y = c2[1] - c1[1];
                Real v2z = c2[2] - c1[2];

                Real dot = v1x*v2x + v1y*v2y + v1z*v2z;
                Real len1 = std::sqrt(v1x*v1x + v1y*v1y + v1z*v1z);
                Real len2 = std::sqrt(v2x*v2x + v2y*v2y + v2z*v2z);

                if (len1 > 1e-15 && len2 > 1e-15) {
                    Real cos_angle = dot / (len1 * len2);
                    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
                    angles[i] = std::acos(cos_angle) * 180.0 / M_PI;
                } else {
                    angles[i] = 90.0;
                }
            }

            quality.min_angle = *std::min_element(angles, angles + 4);
            quality.max_angle = *std::max_element(angles, angles + 4);
        }

        return quality;
    }
};

} // namespace io
} // namespace nxs
