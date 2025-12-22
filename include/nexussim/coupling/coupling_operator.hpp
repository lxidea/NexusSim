#pragma once

#include <nexussim/core/types.hpp>
#include <nexussim/core/exception.hpp>
#include <nexussim/core/logger.hpp>
#include <nexussim/core/gpu.hpp>
#include <nexussim/data/field.hpp>
#include <nexussim/data/mesh.hpp>
#include <memory>
#include <string>
#include <vector>

namespace nxs {
namespace coupling {

// Type alias for convenience
using FieldRef = Field<Real>;

// ============================================================================
// Coupling Method Types
// ============================================================================

enum class CouplingMethod {
    Direct,            ///< Direct copy (same discretization)
    Interpolation,     ///< Spatial interpolation
    Projection,        ///< L2 projection
    NearestNeighbor,   ///< Nearest neighbor mapping
    RadialBasis        ///< Radial basis function interpolation
};

// ============================================================================
// Coupling Interface
// ============================================================================

/**
 * @brief Interface region between two physics modules
 */
struct CouplingInterface {
    std::string name;
    std::string source_module;
    std::string target_module;
    std::shared_ptr<Mesh> source_mesh;
    std::shared_ptr<Mesh> target_mesh;
    CouplingMethod method;

    // Interface nodes/elements
    std::vector<std::size_t> source_nodes;
    std::vector<std::size_t> target_nodes;

    // Interpolation weights (for non-direct methods)
    std::vector<std::vector<std::size_t>> neighbor_indices;
    std::vector<std::vector<Real>> neighbor_weights;
};

// ============================================================================
// Coupling Operator Base Class
// ============================================================================

/**
 * @brief Base class for coupling operators
 *
 * Coupling operators transfer field data between physics modules,
 * handling different discretizations and spatial locations.
 */
class CouplingOperator {
public:
    CouplingOperator(const std::string& name, CouplingMethod method)
        : name_(name), method_(method) {}

    virtual ~CouplingOperator() = default;

    /**
     * @brief Initialize the coupling operator
     * @param interface Coupling interface definition
     */
    virtual void initialize(const CouplingInterface& interface) = 0;

    /**
     * @brief Transfer field data from source to target
     * @param source_field Source field
     * @param target_field Target field
     */
    virtual void transfer(const FieldRef& source_field,
                         FieldRef& target_field) = 0;

    /**
     * @brief Get coupling method
     */
    CouplingMethod method() const { return method_; }

    /**
     * @brief Get operator name
     */
    const std::string& name() const { return name_; }

protected:
    std::string name_;
    CouplingMethod method_;
};

// ============================================================================
// Direct Coupling Operator
// ============================================================================

/**
 * @brief Direct copy coupling (same discretization)
 */
class DirectCouplingOperator : public CouplingOperator {
public:
    DirectCouplingOperator()
        : CouplingOperator("Direct", CouplingMethod::Direct) {}

    void initialize(const CouplingInterface& interface) override {
        interface_ = interface;

        // Verify source and target have same size
        if (interface.source_nodes.size() != interface.target_nodes.size()) {
            throw InvalidArgumentError(
                "Direct coupling requires same number of nodes in source and target"
            );
        }

        NXS_LOG_INFO("Initialized direct coupling: {} nodes",
                    interface.source_nodes.size());
    }

    void transfer(const FieldRef& source_field, FieldRef& target_field) override {
        const std::size_t n = interface_.source_nodes.size();

        // Direct copy
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t src_idx = interface_.source_nodes[i];
            const std::size_t tgt_idx = interface_.target_nodes[i];

            for (int comp = 0; comp < source_field.num_components(); ++comp) {
                target_field.at(tgt_idx, comp) = source_field.at(src_idx, comp);
            }
        }
    }

private:
    CouplingInterface interface_;
};

// ============================================================================
// Interpolation Coupling Operator
// ============================================================================

/**
 * @brief Interpolation-based coupling operator
 *
 * Uses precomputed interpolation weights to transfer data between
 * different discretizations.
 */
class InterpolationCouplingOperator : public CouplingOperator {
public:
    InterpolationCouplingOperator()
        : CouplingOperator("Interpolation", CouplingMethod::Interpolation) {}

    void initialize(const CouplingInterface& interface) override {
        interface_ = interface;

        // Verify interpolation data is provided
        if (interface.neighbor_indices.empty() ||
            interface.neighbor_weights.empty()) {
            throw InvalidArgumentError(
                "Interpolation coupling requires neighbor indices and weights"
            );
        }

        NXS_LOG_INFO("Initialized interpolation coupling: {} target nodes, avg {} neighbors",
                    interface.target_nodes.size(),
                    interface.neighbor_indices[0].size());
    }

    void transfer(const FieldRef& source_field, FieldRef& target_field) override {
        const std::size_t n_target = interface_.target_nodes.size();
        const int n_comp = source_field.num_components();

        // Interpolate for each target node
        for (std::size_t i = 0; i < n_target; ++i) {
            const std::size_t tgt_idx = interface_.target_nodes[i];
            const auto& neighbors = interface_.neighbor_indices[i];
            const auto& weights = interface_.neighbor_weights[i];

            // Weighted average
            for (int comp = 0; comp < n_comp; ++comp) {
                Real value = 0.0;
                for (std::size_t j = 0; j < neighbors.size(); ++j) {
                    value += weights[j] * source_field.at(neighbors[j], comp);
                }
                target_field.at(tgt_idx, comp) = value;
            }
        }
    }

private:
    CouplingInterface interface_;
};

// ============================================================================
// GPU-Compatible Interpolation Kernel
// ============================================================================

/**
 * @brief GPU kernel for interpolation coupling
 */
struct InterpolationKernel {
    // Source and target fields
    View2D<Real> source_data;
    View2D<Real> target_data;

    // Target node indices
    View1D<std::size_t> target_nodes;

    // Interpolation data
    View2D<std::size_t> neighbor_indices;
    View2D<Real> neighbor_weights;

    int num_neighbors;
    int num_components;

    InterpolationKernel(View2D<Real> src, View2D<Real> tgt,
                       View1D<std::size_t> tgt_nodes,
                       View2D<std::size_t> nbr_idx,
                       View2D<Real> nbr_wgt,
                       int n_nbr, int n_comp)
        : source_data(src), target_data(tgt)
        , target_nodes(tgt_nodes)
        , neighbor_indices(nbr_idx), neighbor_weights(nbr_wgt)
        , num_neighbors(n_nbr), num_components(n_comp)
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        const std::size_t tgt_idx = target_nodes(i);

        // Interpolate each component
        for (int comp = 0; comp < num_components; ++comp) {
            Real value = 0.0;

            // Weighted sum over neighbors
            for (int j = 0; j < num_neighbors; ++j) {
                const std::size_t src_idx = neighbor_indices(i, j);
                const Real weight = neighbor_weights(i, j);
                value += weight * source_data(src_idx, comp);
            }

            target_data(tgt_idx, comp) = value;
        }
    }
};

// ============================================================================
// Nearest Neighbor Coupling Operator
// ============================================================================

/**
 * @brief Nearest neighbor coupling operator
 *
 * Maps each target point to its nearest source point.
 * Simple but can be inaccurate for large distance differences.
 */
class NearestNeighborCouplingOperator : public CouplingOperator {
public:
    NearestNeighborCouplingOperator()
        : CouplingOperator("NearestNeighbor", CouplingMethod::NearestNeighbor) {}

    void initialize(const CouplingInterface& interface) override {
        interface_ = interface;

        // Build nearest neighbor mapping
        build_nearest_neighbor_map();

        NXS_LOG_INFO("Initialized nearest neighbor coupling: {} target nodes",
                    interface_.target_nodes.size());
    }

    void transfer(const FieldRef& source_field, FieldRef& target_field) override {
        const std::size_t n_target = interface_.target_nodes.size();
        const int n_comp = source_field.num_components();

        for (std::size_t i = 0; i < n_target; ++i) {
            const std::size_t tgt_idx = interface_.target_nodes[i];
            const std::size_t src_idx = nearest_source_[i];

            for (int comp = 0; comp < n_comp; ++comp) {
                target_field.at(tgt_idx, comp) = source_field.at(src_idx, comp);
            }
        }
    }

private:
    void build_nearest_neighbor_map() {
        const auto& src_nodes = interface_.source_nodes;
        const auto& tgt_nodes = interface_.target_nodes;
        const auto& src_mesh = interface_.source_mesh;
        const auto& tgt_mesh = interface_.target_mesh;

        nearest_source_.resize(tgt_nodes.size());

        // For each target node, find nearest source node
        for (std::size_t i = 0; i < tgt_nodes.size(); ++i) {
            const auto tgt_coord = tgt_mesh->get_node_coordinates(tgt_nodes[i]);

            Real min_dist = std::numeric_limits<Real>::max();
            std::size_t nearest_idx = 0;

            for (std::size_t j = 0; j < src_nodes.size(); ++j) {
                const auto src_coord = src_mesh->get_node_coordinates(src_nodes[j]);

                // Compute distance
                Real dist = 0.0;
                for (int d = 0; d < 3; ++d) {
                    const Real dx = tgt_coord[d] - src_coord[d];
                    dist += dx * dx;
                }

                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_idx = src_nodes[j];
                }
            }

            nearest_source_[i] = nearest_idx;
        }
    }

    CouplingInterface interface_;
    std::vector<std::size_t> nearest_source_;
};

// ============================================================================
// Coupling Operator Factory
// ============================================================================

class CouplingOperatorFactory {
public:
    static std::unique_ptr<CouplingOperator> create(CouplingMethod method) {
        switch (method) {
            case CouplingMethod::Direct:
                return std::make_unique<DirectCouplingOperator>();
            case CouplingMethod::Interpolation:
                return std::make_unique<InterpolationCouplingOperator>();
            case CouplingMethod::NearestNeighbor:
                return std::make_unique<NearestNeighborCouplingOperator>();
            default:
                throw NotImplementedError("Coupling method not implemented");
        }
    }

    static std::string to_string(CouplingMethod method) {
        switch (method) {
            case CouplingMethod::Direct: return "Direct";
            case CouplingMethod::Interpolation: return "Interpolation";
            case CouplingMethod::Projection: return "Projection";
            case CouplingMethod::NearestNeighbor: return "NearestNeighbor";
            case CouplingMethod::RadialBasis: return "RadialBasis";
            default: return "Unknown";
        }
    }
};

} // namespace coupling
} // namespace nxs
