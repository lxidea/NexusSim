/**
 * @file contact.hpp
 * @brief Contact mechanics for explicit FEM simulations
 *
 * Implements penalty-based contact detection and force computation for
 * crash/impact simulations in explicit dynamics.
 */

#pragma once

#include <nexussim/core/types.hpp>
#include <nexussim/data/mesh.hpp>
#include <vector>
#include <memory>
#include <Kokkos_Core.hpp>

namespace nxs {
namespace fem {

/**
 * @brief Contact pair representing a node penetrating a surface
 */
struct ContactPair {
    Index slave_node;        ///< Node that is penetrating
    Index master_face;       ///< Face element being penetrated
    Real penetration_depth;  ///< Distance of penetration (positive = penetrating)
    Real normal[3];          ///< Contact normal (pointing away from master surface)
    Real xi[2];              ///< Parametric coordinates on master face
    Real contact_point[3];   ///< Physical location of contact point on master surface

    ContactPair() : slave_node(-1), master_face(-1), penetration_depth(0.0) {
        normal[0] = normal[1] = normal[2] = 0.0;
        xi[0] = xi[1] = 0.0;
        contact_point[0] = contact_point[1] = contact_point[2] = 0.0;
    }
};

/**
 * @brief Contact surface definition
 */
struct ContactSurface {
    std::string name;
    std::vector<Index> face_nodes;  ///< Connectivity for all faces (4 nodes per face for Shell4)
    std::vector<Index> face_ids;    ///< Element IDs for each face
    ElementType face_type;          ///< Type of face element (typically Shell4)

    ContactSurface(const std::string& n, ElementType type)
        : name(n), face_type(type) {}
};

/**
 * @brief Contact parameters
 */
struct ContactParameters {
    Real penalty_stiffness;    ///< Penalty stiffness factor (multiplied by element stiffness)
    Real friction_coefficient; ///< Coulomb friction coefficient (0 = frictionless)
    Real contact_thickness;    ///< Contact detection thickness (positive = gap allowed)
    bool enable_friction;      ///< Enable/disable friction

    ContactParameters()
        : penalty_stiffness(1.0)
        , friction_coefficient(0.0)
        , contact_thickness(0.0)
        , enable_friction(false) {}
};

/**
 * @brief Contact mechanics manager for explicit FEM
 *
 * Implements node-to-surface penalty contact with optional friction.
 * Uses bucket sort spatial hashing for efficient contact detection.
 */
class ContactMechanics {
public:
    ContactMechanics();
    ~ContactMechanics();

    /**
     * @brief Add a master contact surface (can be penetrated by slave nodes)
     */
    void add_master_surface(const std::string& name,
                           const std::vector<Index>& face_nodes,
                           const std::vector<Index>& face_ids,
                           ElementType face_type);

    /**
     * @brief Add slave nodes (can penetrate master surfaces)
     */
    void add_slave_nodes(const std::string& name, const std::vector<Index>& nodes);

    /**
     * @brief Set contact parameters
     */
    void set_parameters(const ContactParameters& params);

    /**
     * @brief Initialize contact mechanics with mesh
     */
    void initialize(std::shared_ptr<Mesh> mesh);

    /**
     * @brief Detect contact pairs at current configuration
     * @param coords Current nodal coordinates
     * @param displacement Current nodal displacements
     */
    void detect_contact(const Real* coords, const Real* displacement);

    /**
     * @brief Compute contact forces for detected contact pairs
     * @param coords Current nodal coordinates
     * @param displacement Current nodal displacements
     * @param velocity Current nodal velocities
     * @param contact_forces Output: contact forces (added to existing forces)
     */
    void compute_contact_forces(const Real* coords,
                               const Real* displacement,
                               const Real* velocity,
                               Real* contact_forces);

    /**
     * @brief Get number of active contact pairs
     */
    std::size_t num_active_contacts() const { return active_contacts_.size(); }

    /**
     * @brief Get active contact pairs (for visualization/debugging)
     */
    const std::vector<ContactPair>& get_active_contacts() const { return active_contacts_; }

private:
    // Contact surfaces
    std::vector<ContactSurface> master_surfaces_;
    std::vector<std::string> slave_node_groups_;
    std::vector<std::vector<Index>> slave_nodes_;

    // Contact parameters
    ContactParameters params_;

    // Current active contacts
    std::vector<ContactPair> active_contacts_;

    // Mesh reference
    std::shared_ptr<Mesh> mesh_;

    // Spatial hashing for efficient contact detection
    struct SpatialHash {
        Real cell_size;
        std::vector<std::vector<Index>> buckets;  // Node/face indices in each bucket
        int grid_dims[3];
        Real bbox_min[3];
        Real bbox_max[3];
    };
    SpatialHash spatial_hash_;

    // Helper methods
    void build_spatial_hash(const Real* coords, const Real* displacement);
    void detect_node_to_surface(Index slave_node,
                               const ContactSurface& master,
                               const Real* coords,
                               const Real* displacement);
    bool project_point_to_face(const Real point[3],
                              const Real face_coords[12],  // 4 nodes × 3 coords
                              Real xi[2],
                              Real projected[3],
                              Real normal[3],
                              Real& distance);
    void compute_penalty_force(const ContactPair& pair,
                             const Real* coords,
                             const Real* displacement,
                             const Real* velocity,
                             Real* contact_forces);
};

} // namespace fem
} // namespace nxs
