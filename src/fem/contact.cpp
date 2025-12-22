/**
 * @file contact.cpp
 * @brief Contact mechanics implementation
 */

#include <nexussim/fem/contact.hpp>
#include <nexussim/core/logger.hpp>
#include <cmath>
#include <algorithm>
#include <limits>

namespace nxs {
namespace fem {

ContactMechanics::ContactMechanics() {
    NXS_LOG_DEBUG("ContactMechanics constructor");
}

ContactMechanics::~ContactMechanics() {
    NXS_LOG_DEBUG("ContactMechanics destructor");
}

void ContactMechanics::add_master_surface(const std::string& name,
                                         const std::vector<Index>& face_nodes,
                                         const std::vector<Index>& face_ids,
                                         ElementType face_type) {
    ContactSurface surface(name, face_type);
    surface.face_nodes = face_nodes;
    surface.face_ids = face_ids;
    master_surfaces_.push_back(surface);

    const std::size_t num_faces = face_ids.size();
    NXS_LOG_INFO("Added master contact surface '{}': {} faces of type {}",
                 name, num_faces, static_cast<int>(face_type));
}

void ContactMechanics::add_slave_nodes(const std::string& name,
                                      const std::vector<Index>& nodes) {
    slave_node_groups_.push_back(name);
    slave_nodes_.push_back(nodes);

    NXS_LOG_INFO("Added slave node group '{}': {} nodes", name, nodes.size());
}

void ContactMechanics::set_parameters(const ContactParameters& params) {
    params_ = params;

    NXS_LOG_INFO("Contact parameters set:");
    NXS_LOG_INFO("  Penalty stiffness factor: {}", params_.penalty_stiffness);
    NXS_LOG_INFO("  Friction coefficient: {}", params_.friction_coefficient);
    NXS_LOG_INFO("  Contact thickness: {} m", params_.contact_thickness);
    NXS_LOG_INFO("  Friction enabled: {}", params_.enable_friction);
}

void ContactMechanics::initialize(std::shared_ptr<Mesh> mesh) {
    mesh_ = mesh;

    NXS_LOG_INFO("Initializing contact mechanics");
    NXS_LOG_INFO("  Master surfaces: {}", master_surfaces_.size());
    NXS_LOG_INFO("  Slave node groups: {}", slave_node_groups_.size());

    // Count total slave nodes
    std::size_t total_slave_nodes = 0;
    for (const auto& group : slave_nodes_) {
        total_slave_nodes += group.size();
    }
    NXS_LOG_INFO("  Total slave nodes: {}", total_slave_nodes);

    // Count total master faces
    std::size_t total_master_faces = 0;
    for (const auto& surface : master_surfaces_) {
        total_master_faces += surface.face_ids.size();
    }
    NXS_LOG_INFO("  Total master faces: {}", total_master_faces);
}

void ContactMechanics::build_spatial_hash(const Real* coords, const Real* displacement) {
    // Simple spatial hashing: divide domain into cubic cells
    // Cell size should be ~2× the expected contact thickness for efficiency

    // Find bounding box of all nodes
    const auto& mesh_coords = mesh_->coordinates();
    const std::size_t num_nodes = mesh_->num_nodes();

    if (num_nodes == 0) return;

    // Initialize bbox with first node
    Real current_pos[3];
    for (int d = 0; d < 3; ++d) {
        current_pos[d] = mesh_coords.at(0, d) + displacement[d];
        spatial_hash_.bbox_min[d] = current_pos[d];
        spatial_hash_.bbox_max[d] = current_pos[d];
    }

    // Expand bbox to include all nodes
    for (std::size_t i = 1; i < num_nodes; ++i) {
        for (int d = 0; d < 3; ++d) {
            current_pos[d] = mesh_coords.at(i, d) + displacement[i * 3 + d];
            spatial_hash_.bbox_min[d] = std::min(spatial_hash_.bbox_min[d], current_pos[d]);
            spatial_hash_.bbox_max[d] = std::max(spatial_hash_.bbox_max[d], current_pos[d]);
        }
    }

    // Add padding for contact thickness
    const Real padding = std::max(params_.contact_thickness * 2.0, 0.1);
    for (int d = 0; d < 3; ++d) {
        spatial_hash_.bbox_min[d] -= padding;
        spatial_hash_.bbox_max[d] += padding;
    }

    // Determine cell size and grid dimensions
    Real domain_size[3];
    for (int d = 0; d < 3; ++d) {
        domain_size[d] = spatial_hash_.bbox_max[d] - spatial_hash_.bbox_min[d];
    }

    // Target: ~10-20 cells per dimension for typical meshes
    const int target_cells = 20;
    const Real max_domain = std::max({domain_size[0], domain_size[1], domain_size[2]});
    spatial_hash_.cell_size = max_domain / target_cells;

    // Calculate grid dimensions
    for (int d = 0; d < 3; ++d) {
        spatial_hash_.grid_dims[d] = static_cast<int>(std::ceil(domain_size[d] / spatial_hash_.cell_size)) + 1;
    }

    const std::size_t total_buckets = spatial_hash_.grid_dims[0] *
                                     spatial_hash_.grid_dims[1] *
                                     spatial_hash_.grid_dims[2];

    spatial_hash_.buckets.clear();
    spatial_hash_.buckets.resize(total_buckets);

    NXS_LOG_DEBUG("Spatial hash grid: {}×{}×{} = {} buckets, cell size = {} m",
                  spatial_hash_.grid_dims[0], spatial_hash_.grid_dims[1],
                  spatial_hash_.grid_dims[2], total_buckets, spatial_hash_.cell_size);
}

void ContactMechanics::detect_contact(const Real* coords, const Real* displacement) {
    // Clear previous contacts
    active_contacts_.clear();

    // Build spatial hash for efficient detection
    build_spatial_hash(coords, displacement);

    // For each slave node group
    for (std::size_t g = 0; g < slave_nodes_.size(); ++g) {
        const auto& slave_group = slave_nodes_[g];

        // Check each slave node against all master surfaces
        for (const Index slave_node : slave_group) {
            for (const auto& master_surface : master_surfaces_) {
                detect_node_to_surface(slave_node, master_surface, coords, displacement);
            }
        }
    }

    if (active_contacts_.size() > 0) {
        NXS_LOG_DEBUG("Detected {} active contact pairs", active_contacts_.size());
    }
}

void ContactMechanics::detect_node_to_surface(Index slave_node,
                                             const ContactSurface& master,
                                             const Real* coords,
                                             const Real* displacement) {
    // Get current slave node position
    const auto& mesh_coords = mesh_->coordinates();
    Real slave_pos[3];
    for (int d = 0; d < 3; ++d) {
        slave_pos[d] = mesh_coords.at(slave_node, d) + displacement[slave_node * 3 + d];
    }

    // For now, use brute-force search (TODO: spatial hashing optimization)
    // Check against all master faces
    const std::size_t num_faces = master.face_ids.size();
    const int nodes_per_face = 4;  // Assuming Shell4 faces

    Real min_distance = std::numeric_limits<Real>::max();
    ContactPair closest_pair;
    bool found_contact = false;

    for (std::size_t f = 0; f < num_faces; ++f) {
        // Get face node coordinates
        Real face_coords[12];  // 4 nodes × 3 coords
        for (int n = 0; n < nodes_per_face; ++n) {
            const Index node_id = master.face_nodes[f * nodes_per_face + n];
            for (int d = 0; d < 3; ++d) {
                face_coords[n * 3 + d] = mesh_coords.at(node_id, d) + displacement[node_id * 3 + d];
            }
        }

        // Project slave node onto face
        Real xi[2], projected[3], normal[3];
        Real distance;

        if (project_point_to_face(slave_pos, face_coords, xi, projected, normal, distance)) {
            // Check if this is penetration (distance < 0) or within contact threshold
            const Real penetration = -distance;  // Positive = penetrating

            if (penetration > -params_.contact_thickness && std::abs(distance) < std::abs(min_distance)) {
                min_distance = distance;
                closest_pair.slave_node = slave_node;
                closest_pair.master_face = master.face_ids[f];
                closest_pair.penetration_depth = penetration;
                closest_pair.xi[0] = xi[0];
                closest_pair.xi[1] = xi[1];

                for (int d = 0; d < 3; ++d) {
                    closest_pair.normal[d] = normal[d];
                    closest_pair.contact_point[d] = projected[d];
                }

                found_contact = true;
            }
        }
    }

    // Add closest contact if within threshold
    if (found_contact && closest_pair.penetration_depth > -params_.contact_thickness) {
        active_contacts_.push_back(closest_pair);
    }
}

bool ContactMechanics::project_point_to_face(const Real point[3],
                                            const Real face_coords[12],
                                            Real xi[2],
                                            Real projected[3],
                                            Real normal[3],
                                            Real& distance) {
    // Simple projection for quadrilateral face using Newton-Raphson
    // Face parameterization: X(xi, eta) with xi, eta ∈ [-1, 1]

    // Initial guess: center of face
    xi[0] = 0.0;
    xi[1] = 0.0;

    const int max_iter = 10;
    const Real tol = 1.0e-6;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Bilinear shape functions for quad4
        const Real N[4] = {
            0.25 * (1.0 - xi[0]) * (1.0 - xi[1]),
            0.25 * (1.0 + xi[0]) * (1.0 - xi[1]),
            0.25 * (1.0 + xi[0]) * (1.0 + xi[1]),
            0.25 * (1.0 - xi[0]) * (1.0 + xi[1])
        };

        // Shape function derivatives
        const Real dN_dxi[4] = {
            -0.25 * (1.0 - xi[1]),
             0.25 * (1.0 - xi[1]),
             0.25 * (1.0 + xi[1]),
            -0.25 * (1.0 + xi[1])
        };

        const Real dN_deta[4] = {
            -0.25 * (1.0 - xi[0]),
            -0.25 * (1.0 + xi[0]),
             0.25 * (1.0 + xi[0]),
             0.25 * (1.0 - xi[0])
        };

        // Compute current position and tangent vectors
        Real X[3] = {0, 0, 0};
        Real dX_dxi[3] = {0, 0, 0};
        Real dX_deta[3] = {0, 0, 0};

        for (int n = 0; n < 4; ++n) {
            for (int d = 0; d < 3; ++d) {
                X[d] += N[n] * face_coords[n * 3 + d];
                dX_dxi[d] += dN_dxi[n] * face_coords[n * 3 + d];
                dX_deta[d] += dN_deta[n] * face_coords[n * 3 + d];
            }
        }

        // Residual: R = X(xi, eta) - point
        Real R[3];
        for (int d = 0; d < 3; ++d) {
            R[d] = X[d] - point[d];
        }

        // Check convergence
        const Real res_norm = std::sqrt(R[0]*R[0] + R[1]*R[1] + R[2]*R[2]);
        if (res_norm < tol) break;

        // Jacobian: J = [dX/dxi, dX/deta]^T
        // We need: dR/dxi = dX/dxi, dR/deta = dX/deta
        // Solve: J^T * dxi = -R
        const Real J11 = dX_dxi[0]*dX_dxi[0] + dX_dxi[1]*dX_dxi[1] + dX_dxi[2]*dX_dxi[2];
        const Real J12 = dX_dxi[0]*dX_deta[0] + dX_dxi[1]*dX_deta[1] + dX_dxi[2]*dX_deta[2];
        const Real J22 = dX_deta[0]*dX_deta[0] + dX_deta[1]*dX_deta[1] + dX_deta[2]*dX_deta[2];

        const Real rhs1 = -(dX_dxi[0]*R[0] + dX_dxi[1]*R[1] + dX_dxi[2]*R[2]);
        const Real rhs2 = -(dX_deta[0]*R[0] + dX_deta[1]*R[1] + dX_deta[2]*R[2]);

        const Real det = J11 * J22 - J12 * J12;
        if (std::abs(det) < 1.0e-12) return false;  // Singular

        const Real dxi = (J22 * rhs1 - J12 * rhs2) / det;
        const Real deta = (-J12 * rhs1 + J11 * rhs2) / det;

        xi[0] += dxi;
        xi[1] += deta;
    }

    // Check if projection is within face bounds (with small tolerance)
    const Real param_tol = 1.1;  // Allow 10% outside for robustness
    if (std::abs(xi[0]) > param_tol || std::abs(xi[1]) > param_tol) {
        return false;  // Projection outside face
    }

    // Compute final projected point and normal
    const Real N[4] = {
        0.25 * (1.0 - xi[0]) * (1.0 - xi[1]),
        0.25 * (1.0 + xi[0]) * (1.0 - xi[1]),
        0.25 * (1.0 + xi[0]) * (1.0 + xi[1]),
        0.25 * (1.0 - xi[0]) * (1.0 + xi[1])
    };

    for (int d = 0; d < 3; ++d) {
        projected[d] = 0.0;
        for (int n = 0; n < 4; ++n) {
            projected[d] += N[n] * face_coords[n * 3 + d];
        }
    }

    // Compute tangent vectors and normal
    Real dX_dxi[3] = {0, 0, 0};
    Real dX_deta[3] = {0, 0, 0};

    const Real dN_dxi[4] = {
        -0.25 * (1.0 - xi[1]),
         0.25 * (1.0 - xi[1]),
         0.25 * (1.0 + xi[1]),
        -0.25 * (1.0 + xi[1])
    };

    const Real dN_deta[4] = {
        -0.25 * (1.0 - xi[0]),
        -0.25 * (1.0 + xi[0]),
         0.25 * (1.0 + xi[0]),
         0.25 * (1.0 - xi[0])
    };

    for (int n = 0; n < 4; ++n) {
        for (int d = 0; d < 3; ++d) {
            dX_dxi[d] += dN_dxi[n] * face_coords[n * 3 + d];
            dX_deta[d] += dN_deta[n] * face_coords[n * 3 + d];
        }
    }

    // Normal = dX/dxi × dX/deta
    normal[0] = dX_dxi[1] * dX_deta[2] - dX_dxi[2] * dX_deta[1];
    normal[1] = dX_dxi[2] * dX_deta[0] - dX_dxi[0] * dX_deta[2];
    normal[2] = dX_dxi[0] * dX_deta[1] - dX_dxi[1] * dX_deta[0];

    const Real normal_mag = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
    if (normal_mag < 1.0e-12) return false;  // Degenerate face

    for (int d = 0; d < 3; ++d) {
        normal[d] /= normal_mag;
    }

    // Signed distance (positive = point above surface)
    Real diff[3];
    for (int d = 0; d < 3; ++d) {
        diff[d] = point[d] - projected[d];
    }
    distance = diff[0] * normal[0] + diff[1] * normal[1] + diff[2] * normal[2];

    return true;
}

void ContactMechanics::compute_contact_forces(const Real* coords,
                                             const Real* displacement,
                                             const Real* velocity,
                                             Real* contact_forces) {
    // For each active contact pair, compute penalty forces
    for (const auto& pair : active_contacts_) {
        compute_penalty_force(pair, coords, displacement, velocity, contact_forces);
    }
}

void ContactMechanics::compute_penalty_force(const ContactPair& pair,
                                            const Real* coords,
                                            const Real* displacement,
                                            const Real* velocity,
                                            Real* contact_forces) {
    // Penalty method: F = k_penalty * penetration * normal
    // where k_penalty is proportional to element stiffness

    if (pair.penetration_depth <= 0.0) return;  // No penetration

    // Estimate penalty stiffness
    // Scale by penalty factor parameter
    const Real penalty_stiffness = params_.penalty_stiffness * 1.0e6;  // N/m

    // Normal force magnitude (penalty)
    const Real fn_mag = penalty_stiffness * pair.penetration_depth;

    // Normal force vector (pointing away from master surface into slave)
    Real fn[3];
    for (int d = 0; d < 3; ++d) {
        fn[d] = fn_mag * pair.normal[d];
    }

    // Apply normal force to slave node (push it out of master surface)
    for (int d = 0; d < 3; ++d) {
        contact_forces[pair.slave_node * 3 + d] += fn[d];
    }

    // Distribute reaction force to master face nodes using shape functions
    // The contact point parametric coordinates are stored in pair.xi
    const Real xi = pair.xi[0];
    const Real eta = pair.xi[1];

    // Bilinear shape functions for master face nodes
    const Real N[4] = {
        0.25 * (1.0 - xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 + eta),
        0.25 * (1.0 - xi) * (1.0 + eta)
    };

    // Get master face connectivity from stored data
    // Note: pair.master_face is the face index, we need to find the surface
    // For now, assume we know which surface this face belongs to
    // In a full implementation, this would be tracked in the ContactPair

    // Apply reaction force (opposite direction) distributed to master nodes
    // For a simple implementation, apply equal fraction to each master node
    // This is a simplification - proper implementation would use shape function weighting

    // Compute friction force if enabled
    Real ft[3] = {0.0, 0.0, 0.0};
    if (params_.enable_friction && params_.friction_coefficient > 0.0) {
        // Get relative velocity at contact point
        Real v_slave[3];
        for (int d = 0; d < 3; ++d) {
            v_slave[d] = velocity[pair.slave_node * 3 + d];
        }

        // Tangential velocity (relative to master surface)
        // v_t = v - (v · n) * n
        Real v_dot_n = v_slave[0] * pair.normal[0] +
                       v_slave[1] * pair.normal[1] +
                       v_slave[2] * pair.normal[2];

        Real v_tan[3];
        for (int d = 0; d < 3; ++d) {
            v_tan[d] = v_slave[d] - v_dot_n * pair.normal[d];
        }

        // Tangential velocity magnitude
        Real v_tan_mag = std::sqrt(v_tan[0]*v_tan[0] + v_tan[1]*v_tan[1] + v_tan[2]*v_tan[2]);

        if (v_tan_mag > 1.0e-10) {
            // Coulomb friction: |F_t| ≤ μ * |F_n|
            Real max_friction = params_.friction_coefficient * fn_mag;

            // Regularized friction: use penalty in tangent direction
            Real friction_stiffness = 0.1 * penalty_stiffness;  // Lower stiffness for friction
            Real ft_mag = friction_stiffness * v_tan_mag * 0.001;  // Scale by small dt estimate

            // Cap at Coulomb limit
            ft_mag = std::min(ft_mag, max_friction);

            // Friction force opposes tangential motion
            Real v_tan_unit[3];
            for (int d = 0; d < 3; ++d) {
                v_tan_unit[d] = v_tan[d] / v_tan_mag;
                ft[d] = -ft_mag * v_tan_unit[d];
            }
        }
    }

    // Apply friction force to slave node
    for (int d = 0; d < 3; ++d) {
        contact_forces[pair.slave_node * 3 + d] += ft[d];
    }
}

} // namespace fem
} // namespace nxs
