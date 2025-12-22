#pragma once

/**
 * @file node_to_surface_contact.hpp
 * @brief Advanced node-to-surface contact algorithm
 *
 * Features:
 * - Bilinear quad and triangular face projection
 * - Newton-Raphson iterative projection
 * - Bucket search spatial hashing for O(n) detection
 * - Mortar-style force distribution
 * - Self-contact capability
 * - Friction with stick/slip transition
 *
 * Based on established contact algorithms from LS-DYNA/RADIOSS methodology.
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <iostream>

namespace nxs {
namespace fem {

// ============================================================================
// Contact Pair Data Structures
// ============================================================================

/**
 * @brief Detailed contact information for a node-surface pair
 */
struct ContactInfo {
    Index slave_node;            // Penetrating node
    Index master_segment;        // Surface segment index
    Real gap;                    // Gap distance (negative = penetration)
    Vec3r normal;                // Contact normal (outward from master)
    Vec3r contact_point;         // Closest point on master surface
    Real xi[2];                  // Parametric coords on master segment
    Real phi[4];                 // Shape functions at contact point
    bool active;                 // Is contact currently active?

    // Friction state
    Vec3r tangent_slip;          // Accumulated tangential slip
    bool sticking;               // Currently in stick regime?

    ContactInfo()
        : slave_node(-1)
        , master_segment(-1)
        , gap(0.0)
        , normal{0, 0, 0}
        , contact_point{0, 0, 0}
        , active(false)
        , tangent_slip{0, 0, 0}
        , sticking(true)
    {
        xi[0] = xi[1] = 0.0;
        phi[0] = phi[1] = phi[2] = phi[3] = 0.25;
    }
};

/**
 * @brief Surface segment (triangle or quad)
 */
struct SurfaceSegment {
    Index nodes[4];              // Node indices (4th = -1 for triangle)
    Index element_id;            // Parent element ID
    int num_nodes;               // 3 for triangle, 4 for quad

    SurfaceSegment() : element_id(-1), num_nodes(4) {
        nodes[0] = nodes[1] = nodes[2] = nodes[3] = -1;
    }
};

// ============================================================================
// Contact Configuration
// ============================================================================

struct NodeToSurfaceConfig {
    // Penalty parameters
    Real penalty_scale;          // Scale factor for penalty stiffness
    Real contact_thickness;      // Shell thickness for contact

    // Friction
    Real static_friction;        // Static friction coefficient
    Real dynamic_friction;       // Dynamic friction coefficient (≤ static)
    Real friction_decay;         // Decay rate for stick-slip transition

    // Detection
    Real search_radius;          // Maximum search distance
    Real bucket_size_factor;     // Bucket size = factor × search_radius

    // Algorithm options
    bool enable_self_contact;    // Allow self-contact detection
    bool two_pass_contact;       // Master-slave symmetry
    int max_newton_iters;        // Max iterations for projection
    Real newton_tolerance;       // Convergence tolerance

    // Numerical
    Real gap_offset;             // Small offset to prevent exact contact
    Real min_gap_velocity;       // Minimum velocity for friction direction

    NodeToSurfaceConfig()
        : penalty_scale(1.0)
        , contact_thickness(0.001)
        , static_friction(0.3)
        , dynamic_friction(0.2)
        , friction_decay(10.0)
        , search_radius(0.1)
        , bucket_size_factor(2.0)
        , enable_self_contact(false)
        , two_pass_contact(false)
        , max_newton_iters(10)
        , newton_tolerance(1.0e-6)
        , gap_offset(1.0e-8)
        , min_gap_velocity(1.0e-10)
    {}
};

// ============================================================================
// Spatial Hashing (Bucket Search)
// ============================================================================

class SpatialHashGrid {
public:
    SpatialHashGrid() : cell_size_(0.1), initialized_(false) {}

    void initialize(Real cell_size, const Real* coords, Index num_nodes) {
        cell_size_ = cell_size;

        // Compute bounding box
        bbox_min_ = {1e30, 1e30, 1e30};
        bbox_max_ = {-1e30, -1e30, -1e30};

        for (Index i = 0; i < num_nodes; ++i) {
            for (int d = 0; d < 3; ++d) {
                bbox_min_[d] = std::min(bbox_min_[d], coords[i*3 + d]);
                bbox_max_[d] = std::max(bbox_max_[d], coords[i*3 + d]);
            }
        }

        // Add margin
        Real margin = cell_size * 2.0;
        for (int d = 0; d < 3; ++d) {
            bbox_min_[d] -= margin;
            bbox_max_[d] += margin;
        }

        // Compute grid dimensions
        for (int d = 0; d < 3; ++d) {
            grid_dims_[d] = static_cast<int>((bbox_max_[d] - bbox_min_[d]) / cell_size_) + 1;
            grid_dims_[d] = std::max(grid_dims_[d], 1);
        }

        // Allocate buckets
        int total_cells = grid_dims_[0] * grid_dims_[1] * grid_dims_[2];
        buckets_.clear();
        buckets_.resize(total_cells);

        initialized_ = true;
    }

    void clear() {
        for (auto& bucket : buckets_) {
            bucket.clear();
        }
    }

    void insert(Index id, const Vec3r& pos) {
        int cell = get_cell_index(pos);
        if (cell >= 0 && cell < static_cast<int>(buckets_.size())) {
            buckets_[cell].push_back(id);
        }
    }

    void insert_segment(Index seg_id, const Real* node_coords, const Index* seg_nodes, int num_nodes) {
        // Compute segment AABB
        Vec3r seg_min = {1e30, 1e30, 1e30};
        Vec3r seg_max = {-1e30, -1e30, -1e30};

        for (int n = 0; n < num_nodes; ++n) {
            Index node = seg_nodes[n];
            for (int d = 0; d < 3; ++d) {
                Real c = node_coords[node*3 + d];
                seg_min[d] = std::min(seg_min[d], c);
                seg_max[d] = std::max(seg_max[d], c);
            }
        }

        // Insert into all overlapping cells
        int i_min = static_cast<int>((seg_min[0] - bbox_min_[0]) / cell_size_);
        int j_min = static_cast<int>((seg_min[1] - bbox_min_[1]) / cell_size_);
        int k_min = static_cast<int>((seg_min[2] - bbox_min_[2]) / cell_size_);
        int i_max = static_cast<int>((seg_max[0] - bbox_min_[0]) / cell_size_);
        int j_max = static_cast<int>((seg_max[1] - bbox_min_[1]) / cell_size_);
        int k_max = static_cast<int>((seg_max[2] - bbox_min_[2]) / cell_size_);

        i_min = std::max(0, std::min(i_min, grid_dims_[0] - 1));
        j_min = std::max(0, std::min(j_min, grid_dims_[1] - 1));
        k_min = std::max(0, std::min(k_min, grid_dims_[2] - 1));
        i_max = std::max(0, std::min(i_max, grid_dims_[0] - 1));
        j_max = std::max(0, std::min(j_max, grid_dims_[1] - 1));
        k_max = std::max(0, std::min(k_max, grid_dims_[2] - 1));

        for (int i = i_min; i <= i_max; ++i) {
            for (int j = j_min; j <= j_max; ++j) {
                for (int k = k_min; k <= k_max; ++k) {
                    int cell = i + grid_dims_[0] * (j + grid_dims_[1] * k);
                    buckets_[cell].push_back(seg_id);
                }
            }
        }
    }

    const std::vector<Index>& get_candidates(const Vec3r& pos) const {
        static std::vector<Index> empty;
        int cell = get_cell_index(pos);
        if (cell >= 0 && cell < static_cast<int>(buckets_.size())) {
            return buckets_[cell];
        }
        return empty;
    }

    std::vector<Index> get_nearby_candidates(const Vec3r& pos, Real radius) const {
        std::vector<Index> result;

        int i_min = static_cast<int>((pos[0] - radius - bbox_min_[0]) / cell_size_);
        int j_min = static_cast<int>((pos[1] - radius - bbox_min_[1]) / cell_size_);
        int k_min = static_cast<int>((pos[2] - radius - bbox_min_[2]) / cell_size_);
        int i_max = static_cast<int>((pos[0] + radius - bbox_min_[0]) / cell_size_);
        int j_max = static_cast<int>((pos[1] + radius - bbox_min_[1]) / cell_size_);
        int k_max = static_cast<int>((pos[2] + radius - bbox_min_[2]) / cell_size_);

        i_min = std::max(0, std::min(i_min, grid_dims_[0] - 1));
        j_min = std::max(0, std::min(j_min, grid_dims_[1] - 1));
        k_min = std::max(0, std::min(k_min, grid_dims_[2] - 1));
        i_max = std::max(0, std::min(i_max, grid_dims_[0] - 1));
        j_max = std::max(0, std::min(j_max, grid_dims_[1] - 1));
        k_max = std::max(0, std::min(k_max, grid_dims_[2] - 1));

        for (int i = i_min; i <= i_max; ++i) {
            for (int j = j_min; j <= j_max; ++j) {
                for (int k = k_min; k <= k_max; ++k) {
                    int cell = i + grid_dims_[0] * (j + grid_dims_[1] * k);
                    const auto& bucket = buckets_[cell];
                    result.insert(result.end(), bucket.begin(), bucket.end());
                }
            }
        }

        // Remove duplicates
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());

        return result;
    }

private:
    int get_cell_index(const Vec3r& pos) const {
        if (!initialized_) return -1;

        int i = static_cast<int>((pos[0] - bbox_min_[0]) / cell_size_);
        int j = static_cast<int>((pos[1] - bbox_min_[1]) / cell_size_);
        int k = static_cast<int>((pos[2] - bbox_min_[2]) / cell_size_);

        if (i < 0 || i >= grid_dims_[0] ||
            j < 0 || j >= grid_dims_[1] ||
            k < 0 || k >= grid_dims_[2]) {
            return -1;
        }

        return i + grid_dims_[0] * (j + grid_dims_[1] * k);
    }

    Real cell_size_;
    Vec3r bbox_min_, bbox_max_;
    int grid_dims_[3];
    std::vector<std::vector<Index>> buckets_;
    bool initialized_;
};

// ============================================================================
// Node-to-Surface Contact Algorithm
// ============================================================================

class NodeToSurfaceContact {
public:
    NodeToSurfaceContact(const NodeToSurfaceConfig& config = NodeToSurfaceConfig())
        : config_(config)
        , num_nodes_(0)
    {}

    // ========================================================================
    // Setup
    // ========================================================================

    /**
     * @brief Set slave nodes (nodes that can penetrate master surfaces)
     */
    void set_slave_nodes(const std::vector<Index>& slave_nodes) {
        slave_nodes_ = slave_nodes;
    }

    /**
     * @brief Add master surface segments (surfaces that can be penetrated)
     * @param connectivity Node indices for each segment (3 or 4 per segment)
     * @param num_segments Number of segments
     * @param nodes_per_segment 3 for triangles, 4 for quads
     */
    void add_master_segments(const std::vector<Index>& connectivity,
                             int num_segments,
                             int nodes_per_segment) {
        for (int s = 0; s < num_segments; ++s) {
            SurfaceSegment seg;
            seg.num_nodes = nodes_per_segment;
            seg.element_id = static_cast<Index>(master_segments_.size());
            for (int n = 0; n < nodes_per_segment; ++n) {
                seg.nodes[n] = connectivity[s * nodes_per_segment + n];
            }
            master_segments_.push_back(seg);
        }
    }

    /**
     * @brief Initialize with mesh size
     */
    void initialize(Index num_nodes) {
        num_nodes_ = num_nodes;
        contact_history_.clear();
        contact_history_.resize(slave_nodes_.size());
    }

    // ========================================================================
    // Contact Detection
    // ========================================================================

    /**
     * @brief Detect contacts at current configuration
     * @param coords Nodal coordinates (3 × num_nodes)
     * @return Number of active contacts
     */
    int detect_contacts(const Real* coords) {
        active_contacts_.clear();

        if (slave_nodes_.empty() || master_segments_.empty()) {
            return 0;
        }

        // Build spatial hash for master segments
        Real bucket_size = config_.search_radius * config_.bucket_size_factor;
        segment_hash_.initialize(bucket_size, coords, num_nodes_);
        segment_hash_.clear();

        for (std::size_t s = 0; s < master_segments_.size(); ++s) {
            const auto& seg = master_segments_[s];
            segment_hash_.insert_segment(static_cast<Index>(s), coords,
                                         seg.nodes, seg.num_nodes);
        }

        // Check each slave node against nearby segments
        for (std::size_t i = 0; i < slave_nodes_.size(); ++i) {
            Index slave = slave_nodes_[i];
            Vec3r slave_pos = {coords[slave*3], coords[slave*3+1], coords[slave*3+2]};

            auto candidates = segment_hash_.get_nearby_candidates(slave_pos, config_.search_radius);

            // Find closest penetrating segment
            Real min_gap = config_.search_radius;
            ContactInfo best_contact;
            best_contact.slave_node = slave;

            for (Index seg_id : candidates) {
                if (config_.enable_self_contact) {
                    // Skip if slave node is part of this segment
                    const auto& seg = master_segments_[seg_id];
                    bool skip = false;
                    for (int n = 0; n < seg.num_nodes; ++n) {
                        if (seg.nodes[n] == slave) {
                            skip = true;
                            break;
                        }
                    }
                    if (skip) continue;
                }

                ContactInfo info;
                if (project_to_segment(slave_pos, seg_id, coords, info)) {
                    if (info.gap < min_gap) {
                        min_gap = info.gap;
                        best_contact = info;
                        best_contact.slave_node = slave;
                        best_contact.master_segment = seg_id;
                        best_contact.active = (info.gap < config_.contact_thickness);
                    }
                }
            }

            if (best_contact.active) {
                // Restore friction history
                if (contact_history_[i].active) {
                    best_contact.tangent_slip = contact_history_[i].tangent_slip;
                    best_contact.sticking = contact_history_[i].sticking;
                }

                active_contacts_.push_back(best_contact);
                contact_history_[i] = best_contact;
            } else {
                contact_history_[i].active = false;
            }
        }

        return static_cast<int>(active_contacts_.size());
    }

    // ========================================================================
    // Force Computation
    // ========================================================================

    /**
     * @brief Compute contact forces for active contacts
     * @param coords Nodal coordinates
     * @param velocity Nodal velocities
     * @param element_stiffness Representative element stiffness (for penalty)
     * @param dt Time step (for friction regularization)
     * @param forces Output forces (added to existing)
     */
    void compute_forces(const Real* coords,
                        const Real* velocity,
                        Real element_stiffness,
                        Real dt,
                        Real* forces) {
        Real penalty_stiffness = config_.penalty_scale * element_stiffness;

        for (auto& contact : active_contacts_) {
            if (!contact.active) continue;

            // Get slave node position and velocity
            Index slave = contact.slave_node;
            Vec3r slave_vel = {velocity[slave*3], velocity[slave*3+1], velocity[slave*3+2]};

            // Compute master surface velocity at contact point
            const auto& seg = master_segments_[contact.master_segment];
            Vec3r master_vel = {0, 0, 0};
            for (int n = 0; n < seg.num_nodes; ++n) {
                Index node = seg.nodes[n];
                for (int d = 0; d < 3; ++d) {
                    master_vel[d] += contact.phi[n] * velocity[node*3 + d];
                }
            }

            // Relative velocity
            Vec3r rel_vel;
            for (int d = 0; d < 3; ++d) {
                rel_vel[d] = slave_vel[d] - master_vel[d];
            }

            // Normal and tangential relative velocity
            Real vn = 0.0;
            for (int d = 0; d < 3; ++d) {
                vn += rel_vel[d] * contact.normal[d];
            }

            Vec3r v_tang;
            for (int d = 0; d < 3; ++d) {
                v_tang[d] = rel_vel[d] - vn * contact.normal[d];
            }

            // ================================================================
            // Normal force (penalty)
            // ================================================================
            Real penetration = config_.contact_thickness - contact.gap;
            if (penetration <= 0.0) continue;

            Real fn = penalty_stiffness * penetration;

            // Add damping for stability
            Real damping_ratio = 0.1;
            Real damping = 2.0 * damping_ratio * std::sqrt(penalty_stiffness);
            if (vn < 0) {  // Approaching
                fn -= damping * vn;
            }
            fn = std::max(fn, 0.0);  // No tensile contact

            Vec3r normal_force;
            for (int d = 0; d < 3; ++d) {
                normal_force[d] = fn * contact.normal[d];
            }

            // ================================================================
            // Friction force
            // ================================================================
            Vec3r friction_force = {0, 0, 0};

            if (config_.static_friction > 0.0 && fn > 0.0) {
                // Update tangential slip
                for (int d = 0; d < 3; ++d) {
                    contact.tangent_slip[d] += v_tang[d] * dt;
                }

                // Stick friction (penalty on slip)
                Real slip_mag = std::sqrt(contact.tangent_slip[0] * contact.tangent_slip[0] +
                                          contact.tangent_slip[1] * contact.tangent_slip[1] +
                                          contact.tangent_slip[2] * contact.tangent_slip[2]);

                Real friction_coeff = contact.sticking ?
                    config_.static_friction : config_.dynamic_friction;
                Real max_friction = friction_coeff * fn;

                // Trial friction force (stick)
                Real stick_stiffness = penalty_stiffness * 0.5;  // Tangential stiffness
                Real trial_friction = stick_stiffness * slip_mag;

                if (trial_friction <= max_friction && slip_mag > config_.min_gap_velocity) {
                    // Sticking
                    contact.sticking = true;
                    Real scale = trial_friction / slip_mag;
                    for (int d = 0; d < 3; ++d) {
                        friction_force[d] = -scale * contact.tangent_slip[d];
                    }
                } else if (slip_mag > config_.min_gap_velocity) {
                    // Sliding
                    contact.sticking = false;
                    Real scale = max_friction / slip_mag;
                    for (int d = 0; d < 3; ++d) {
                        friction_force[d] = -scale * contact.tangent_slip[d];
                        // Reset slip for next step
                        contact.tangent_slip[d] = friction_force[d] / stick_stiffness;
                    }
                }
            }

            // ================================================================
            // Apply forces
            // ================================================================

            // Force on slave node
            for (int d = 0; d < 3; ++d) {
                forces[slave*3 + d] += normal_force[d] + friction_force[d];
            }

            // Reaction forces on master segment nodes (distributed by shape functions)
            for (int n = 0; n < seg.num_nodes; ++n) {
                Index node = seg.nodes[n];
                for (int d = 0; d < 3; ++d) {
                    forces[node*3 + d] -= contact.phi[n] * (normal_force[d] + friction_force[d]);
                }
            }
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    const std::vector<ContactInfo>& get_active_contacts() const {
        return active_contacts_;
    }

    int num_active_contacts() const {
        return static_cast<int>(active_contacts_.size());
    }

    const NodeToSurfaceConfig& config() const { return config_; }
    NodeToSurfaceConfig& config() { return config_; }

private:
    // ========================================================================
    // Projection Algorithm
    // ========================================================================

    /**
     * @brief Project point to surface segment using Newton-Raphson
     */
    bool project_to_segment(const Vec3r& point,
                            Index seg_id,
                            const Real* coords,
                            ContactInfo& info) {
        const auto& seg = master_segments_[seg_id];

        // Get segment node coordinates
        Vec3r x[4];
        for (int n = 0; n < seg.num_nodes; ++n) {
            Index node = seg.nodes[n];
            x[n] = {coords[node*3], coords[node*3+1], coords[node*3+2]};
        }

        if (seg.num_nodes == 3) {
            return project_to_triangle(point, x, info);
        } else {
            return project_to_quad(point, x, info);
        }
    }

    /**
     * @brief Project point to triangle
     */
    bool project_to_triangle(const Vec3r& point, const Vec3r x[3], ContactInfo& info) {
        // Edge vectors
        Vec3r e1 = {x[1][0] - x[0][0], x[1][1] - x[0][1], x[1][2] - x[0][2]};
        Vec3r e2 = {x[2][0] - x[0][0], x[2][1] - x[0][1], x[2][2] - x[0][2]};

        // Normal
        Vec3r n = {
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0]
        };
        Real n_mag = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        if (n_mag < 1.0e-20) return false;

        for (int d = 0; d < 3; ++d) n[d] /= n_mag;

        // Point to plane distance
        Vec3r p0 = {point[0] - x[0][0], point[1] - x[0][1], point[2] - x[0][2]};
        Real dist = p0[0]*n[0] + p0[1]*n[1] + p0[2]*n[2];

        // Projected point on plane
        Vec3r proj;
        for (int d = 0; d < 3; ++d) {
            proj[d] = point[d] - dist * n[d];
        }

        // Barycentric coordinates
        Real area2 = n_mag;
        Vec3r v0 = {proj[0] - x[0][0], proj[1] - x[0][1], proj[2] - x[0][2]};
        Vec3r v1 = {x[1][0] - x[0][0], x[1][1] - x[0][1], x[1][2] - x[0][2]};
        Vec3r v2 = {x[2][0] - x[0][0], x[2][1] - x[0][1], x[2][2] - x[0][2]};

        Real d00 = v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2];
        Real d01 = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
        Real d11 = v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2];
        Real d20 = v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2];
        Real d21 = v0[0]*v2[0] + v0[1]*v2[1] + v0[2]*v2[2];

        Real denom = d00*d11 - d01*d01;
        if (std::abs(denom) < 1.0e-20) return false;

        Real v = (d11*d20 - d01*d21) / denom;
        Real w = (d00*d21 - d01*d20) / denom;
        Real u = 1.0 - v - w;

        // Check if inside triangle (with tolerance)
        Real tol = -0.05;  // Allow slight outside for better contact capture
        if (u < tol || v < tol || w < tol || u > 1.0 - tol || v > 1.0 - tol || w > 1.0 - tol) {
            return false;
        }

        // Clamp to triangle
        u = std::max(0.0, std::min(1.0, u));
        v = std::max(0.0, std::min(1.0, v));
        w = std::max(0.0, std::min(1.0, w));
        Real sum = u + v + w;
        u /= sum; v /= sum; w /= sum;

        // Fill contact info
        info.xi[0] = v;
        info.xi[1] = w;
        info.phi[0] = u;
        info.phi[1] = v;
        info.phi[2] = w;
        info.phi[3] = 0.0;

        for (int d = 0; d < 3; ++d) {
            info.contact_point[d] = u*x[0][d] + v*x[1][d] + w*x[2][d];
            info.normal[d] = n[d];
        }
        info.gap = dist;

        return true;
    }

    /**
     * @brief Project point to bilinear quad using Newton-Raphson
     */
    bool project_to_quad(const Vec3r& point, const Vec3r x[4], ContactInfo& info) {
        // Newton-Raphson to find parametric coordinates (xi, eta)
        Real xi = 0.0, eta = 0.0;  // Initial guess at center

        for (int iter = 0; iter < config_.max_newton_iters; ++iter) {
            // Shape functions
            Real N[4] = {
                0.25 * (1-xi) * (1-eta),
                0.25 * (1+xi) * (1-eta),
                0.25 * (1+xi) * (1+eta),
                0.25 * (1-xi) * (1+eta)
            };

            // Shape function derivatives
            Real dN_dxi[4] = {
                -0.25 * (1-eta),
                 0.25 * (1-eta),
                 0.25 * (1+eta),
                -0.25 * (1+eta)
            };
            Real dN_deta[4] = {
                -0.25 * (1-xi),
                -0.25 * (1+xi),
                 0.25 * (1+xi),
                 0.25 * (1-xi)
            };

            // Position and derivatives on surface
            Vec3r r = {0, 0, 0};
            Vec3r dr_dxi = {0, 0, 0};
            Vec3r dr_deta = {0, 0, 0};

            for (int n = 0; n < 4; ++n) {
                for (int d = 0; d < 3; ++d) {
                    r[d] += N[n] * x[n][d];
                    dr_dxi[d] += dN_dxi[n] * x[n][d];
                    dr_deta[d] += dN_deta[n] * x[n][d];
                }
            }

            // Normal vector (cross product of tangents)
            Vec3r normal = {
                dr_dxi[1]*dr_deta[2] - dr_dxi[2]*dr_deta[1],
                dr_dxi[2]*dr_deta[0] - dr_dxi[0]*dr_deta[2],
                dr_dxi[0]*dr_deta[1] - dr_dxi[1]*dr_deta[0]
            };
            Real n_mag = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
            if (n_mag < 1.0e-20) return false;
            for (int d = 0; d < 3; ++d) normal[d] /= n_mag;

            // Residual: point - r should be parallel to normal
            Vec3r res = {point[0] - r[0], point[1] - r[1], point[2] - r[2]};

            // Project residual onto tangent plane
            Real res_xi = res[0]*dr_dxi[0] + res[1]*dr_dxi[1] + res[2]*dr_dxi[2];
            Real res_eta = res[0]*dr_deta[0] + res[1]*dr_deta[1] + res[2]*dr_deta[2];

            // Jacobian for tangent projection
            Real J11 = dr_dxi[0]*dr_dxi[0] + dr_dxi[1]*dr_dxi[1] + dr_dxi[2]*dr_dxi[2];
            Real J12 = dr_dxi[0]*dr_deta[0] + dr_dxi[1]*dr_deta[1] + dr_dxi[2]*dr_deta[2];
            Real J22 = dr_deta[0]*dr_deta[0] + dr_deta[1]*dr_deta[1] + dr_deta[2]*dr_deta[2];

            Real det = J11*J22 - J12*J12;
            if (std::abs(det) < 1.0e-20) return false;

            // Newton update
            Real dxi = (J22*res_xi - J12*res_eta) / det;
            Real deta = (-J12*res_xi + J11*res_eta) / det;

            xi += dxi;
            eta += deta;

            // Check convergence
            if (std::abs(dxi) < config_.newton_tolerance &&
                std::abs(deta) < config_.newton_tolerance) {
                break;
            }
        }

        // Check if inside quad (with tolerance)
        Real tol = 1.05;  // Allow slight outside
        if (std::abs(xi) > tol || std::abs(eta) > tol) {
            return false;
        }

        // Clamp to quad
        xi = std::max(-1.0, std::min(1.0, xi));
        eta = std::max(-1.0, std::min(1.0, eta));

        // Final shape functions
        info.phi[0] = 0.25 * (1-xi) * (1-eta);
        info.phi[1] = 0.25 * (1+xi) * (1-eta);
        info.phi[2] = 0.25 * (1+xi) * (1+eta);
        info.phi[3] = 0.25 * (1-xi) * (1+eta);

        info.xi[0] = xi;
        info.xi[1] = eta;

        // Contact point
        for (int d = 0; d < 3; ++d) {
            info.contact_point[d] = 0.0;
            for (int n = 0; n < 4; ++n) {
                info.contact_point[d] += info.phi[n] * x[n][d];
            }
        }

        // Normal at contact point
        Real dN_dxi[4] = {
            -0.25 * (1-eta),
             0.25 * (1-eta),
             0.25 * (1+eta),
            -0.25 * (1+eta)
        };
        Real dN_deta[4] = {
            -0.25 * (1-xi),
            -0.25 * (1+xi),
             0.25 * (1+xi),
             0.25 * (1-xi)
        };

        Vec3r dr_dxi = {0, 0, 0};
        Vec3r dr_deta = {0, 0, 0};
        for (int n = 0; n < 4; ++n) {
            for (int d = 0; d < 3; ++d) {
                dr_dxi[d] += dN_dxi[n] * x[n][d];
                dr_deta[d] += dN_deta[n] * x[n][d];
            }
        }

        Vec3r n = {
            dr_dxi[1]*dr_deta[2] - dr_dxi[2]*dr_deta[1],
            dr_dxi[2]*dr_deta[0] - dr_dxi[0]*dr_deta[2],
            dr_dxi[0]*dr_deta[1] - dr_dxi[1]*dr_deta[0]
        };
        Real n_mag = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        if (n_mag < 1.0e-20) return false;

        for (int d = 0; d < 3; ++d) {
            info.normal[d] = n[d] / n_mag;
        }

        // Gap distance
        Vec3r diff = {
            point[0] - info.contact_point[0],
            point[1] - info.contact_point[1],
            point[2] - info.contact_point[2]
        };
        info.gap = diff[0]*info.normal[0] + diff[1]*info.normal[1] + diff[2]*info.normal[2];

        return true;
    }

    // ========================================================================
    // Data Members
    // ========================================================================

    NodeToSurfaceConfig config_;
    Index num_nodes_;

    std::vector<Index> slave_nodes_;
    std::vector<SurfaceSegment> master_segments_;

    std::vector<ContactInfo> active_contacts_;
    std::vector<ContactInfo> contact_history_;  // For friction tracking

    SpatialHashGrid segment_hash_;
};

} // namespace fem
} // namespace nxs
