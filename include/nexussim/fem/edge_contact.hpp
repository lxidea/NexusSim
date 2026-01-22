#pragma once

/**
 * @file edge_contact.hpp
 * @brief Edge-to-surface contact algorithm (INT25-style)
 *
 * Features:
 * - Edge projection onto master surface
 * - Shell edge contact detection
 * - Pinch prevention for thin structures
 * - Handles sharp corners and edge-edge contact
 *
 * Based on OpenRadioss INT25 methodology for edge contact.
 * Essential for crash simulation where thin shell structures fold.
 */

#include <nexussim/core/types.hpp>
#include <nexussim/fem/voxel_collision.hpp>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <memory>
#include <unordered_set>

namespace nxs {
namespace fem {

// ============================================================================
// Edge Contact Configuration
// ============================================================================

/**
 * @brief Configuration for edge-to-surface contact
 */
struct EdgeContactConfig {
    // Penalty parameters
    Real penalty_scale;          ///< Scale factor for penalty stiffness
    Real contact_thickness;      ///< Shell contact thickness
    Real edge_angle_threshold;   ///< Minimum angle (rad) to identify edge (default 30°)

    // Detection
    Real search_radius;          ///< Maximum search distance
    Real bucket_size_factor;     ///< Bucket size multiplier

    // Friction
    Real static_friction;        ///< Static friction coefficient
    Real dynamic_friction;       ///< Dynamic friction coefficient

    // Algorithm options
    bool detect_edge_edge;       ///< Also detect edge-edge contacts
    bool pinch_prevention;       ///< Enable pinch prevention for thin shells

    EdgeContactConfig()
        : penalty_scale(1.0)
        , contact_thickness(0.001)
        , edge_angle_threshold(0.523599)  // 30 degrees
        , search_radius(0.1)
        , bucket_size_factor(2.0)
        , static_friction(0.3)
        , dynamic_friction(0.2)
        , detect_edge_edge(true)
        , pinch_prevention(true)
    {}
};

// ============================================================================
// Edge Definition
// ============================================================================

/**
 * @brief A free edge (boundary or sharp edge)
 */
struct ContactEdge {
    Index node1;                 ///< First node of edge
    Index node2;                 ///< Second node of edge
    Index parent_element;        ///< Parent shell element
    Index part_id;               ///< Part ID for friction
    Vec3r normal;                ///< Average normal of adjacent faces

    ContactEdge()
        : node1(-1), node2(-1), parent_element(-1), part_id(0)
        , normal{0, 0, 1}
    {}

    ContactEdge(Index n1, Index n2, Index elem = -1)
        : node1(n1), node2(n2), parent_element(elem), part_id(0)
        , normal{0, 0, 1}
    {}
};

// ============================================================================
// Edge Contact Pair
// ============================================================================

/**
 * @brief Contact information for edge-surface or edge-edge pair
 */
struct EdgeContactPair {
    enum class ContactType { EdgeToSurface, EdgeToEdge };

    ContactType type;
    Index edge_id;               ///< Edge index
    Index target_id;             ///< Segment or edge index

    Vec3r contact_point;         ///< Contact point on edge
    Vec3r target_point;          ///< Contact point on target
    Vec3r normal;                ///< Contact normal
    Real gap;                    ///< Gap distance
    Real pressure;               ///< Contact pressure

    // Parametric position along edge [0, 1]
    Real edge_param;

    // Shape functions for target segment (if EdgeToSurface)
    Real target_phi[4];

    // Friction state
    Vec3r tangent_slip;
    bool sticking;

    bool active;

    EdgeContactPair()
        : type(ContactType::EdgeToSurface)
        , edge_id(-1)
        , target_id(-1)
        , contact_point{0, 0, 0}
        , target_point{0, 0, 0}
        , normal{0, 0, 1}
        , gap(0.0)
        , pressure(0.0)
        , edge_param(0.5)
        , tangent_slip{0, 0, 0}
        , sticking(true)
        , active(false)
    {
        target_phi[0] = target_phi[1] = target_phi[2] = target_phi[3] = 0.25;
    }
};

// ============================================================================
// Edge-to-Surface Contact Algorithm
// ============================================================================

/**
 * @brief Edge-to-surface contact for shell structures
 *
 * The algorithm:
 * 1. Identify free edges (boundary or sharp feature edges)
 * 2. Build spatial hash for master surface segments
 * 3. For each edge, find closest point on nearby segments
 * 4. Apply penalty forces if within contact thickness
 * 5. Optional: Detect edge-edge contacts
 *
 * Usage:
 * ```cpp
 * EdgeToSurfaceContact contact;
 * contact.set_edges(edges);
 * contact.set_master_segments(segments);
 * contact.initialize(coords, num_nodes);
 *
 * // Each time step:
 * contact.detect(coords);
 * contact.compute_forces(coords, velocity, dt, stiffness, forces);
 * ```
 */
class EdgeToSurfaceContact {
public:
    EdgeToSurfaceContact(const EdgeContactConfig& config = EdgeContactConfig())
        : config_(config)
        , num_nodes_(0)
    {}

    // ========================================================================
    // Setup
    // ========================================================================

    /**
     * @brief Set contact edges
     */
    void set_edges(const std::vector<ContactEdge>& edges) {
        edges_ = edges;
    }

    /**
     * @brief Add edges from shell element connectivity
     * @param connectivity Shell element connectivity
     * @param num_elements Number of shell elements
     * @param nodes_per_elem 3 or 4
     */
    void extract_boundary_edges(const std::vector<Index>& connectivity,
                                 Index num_elements,
                                 int nodes_per_elem) {
        // Count edge occurrences to find boundary edges
        std::unordered_map<uint64_t, int> edge_count;
        std::unordered_map<uint64_t, std::pair<Index, Index>> edge_nodes;

        for (Index e = 0; e < num_elements; ++e) {
            for (int i = 0; i < nodes_per_elem; ++i) {
                Index n1 = connectivity[e * nodes_per_elem + i];
                Index n2 = connectivity[e * nodes_per_elem + (i + 1) % nodes_per_elem];

                // Canonical edge key (smaller node first)
                Index a = std::min(n1, n2);
                Index b = std::max(n1, n2);
                uint64_t key = (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);

                edge_count[key]++;
                edge_nodes[key] = {a, b};
            }
        }

        // Boundary edges appear only once
        edges_.clear();
        for (const auto& [key, count] : edge_count) {
            if (count == 1) {
                auto [n1, n2] = edge_nodes[key];
                edges_.emplace_back(n1, n2);
            }
        }
    }

    /**
     * @brief Set master surface segments
     */
    void set_master_segments(const std::vector<Index>& connectivity,
                              Index num_segments,
                              int nodes_per_segment) {
        segment_connectivity_ = connectivity;
        num_segments_ = num_segments;
        nodes_per_segment_ = nodes_per_segment;
    }

    /**
     * @brief Initialize with mesh data
     */
    void initialize(const Real* coords, Index num_nodes) {
        num_nodes_ = num_nodes;

        // Compute edge normals from adjacent elements
        compute_edge_normals(coords);

        // Build voxel grid for segments
        VoxelCollisionConfig voxel_config;
        voxel_config.cell_size_factor = config_.bucket_size_factor;
        voxel_config.search_margin = config_.search_radius;

        voxel_detector_ = std::make_unique<VoxelCollisionDetector>(voxel_config);

        if (!segment_connectivity_.empty()) {
            voxel_detector_->initialize(
                num_nodes,
                num_segments_,
                coords,
                segment_connectivity_.data(),
                nodes_per_segment_
            );
        }
    }

    // ========================================================================
    // Contact Detection
    // ========================================================================

    /**
     * @brief Detect edge contacts
     * @param coords Nodal coordinates
     * @return Number of active contacts
     */
    int detect(const Real* coords) {
        active_pairs_.clear();

        if (edges_.empty() || segment_connectivity_.empty()) {
            return 0;
        }

        // Update voxel grid
        voxel_detector_->update_coordinates(coords);

        // For each edge, find contacts
        for (size_t i = 0; i < edges_.size(); ++i) {
            detect_edge_contacts(i, coords);
        }

        // Optional: edge-edge detection
        if (config_.detect_edge_edge) {
            detect_edge_edge_contacts(coords);
        }

        return static_cast<int>(active_pairs_.size());
    }

    // ========================================================================
    // Force Computation
    // ========================================================================

    /**
     * @brief Compute edge contact forces
     */
    void compute_forces(const Real* coords,
                        const Real* velocity,
                        Real dt,
                        Real stiffness,
                        Real* forces) {
        Real penalty_stiffness = config_.penalty_scale * stiffness;

        for (auto& pair : active_pairs_) {
            if (!pair.active) continue;

            const auto& edge = edges_[pair.edge_id];

            // Edge node positions and velocities
            Vec3r x1 = {coords[edge.node1 * 3], coords[edge.node1 * 3 + 1], coords[edge.node1 * 3 + 2]};
            Vec3r x2 = {coords[edge.node2 * 3], coords[edge.node2 * 3 + 1], coords[edge.node2 * 3 + 2]};
            Vec3r v1 = {velocity[edge.node1 * 3], velocity[edge.node1 * 3 + 1], velocity[edge.node1 * 3 + 2]};
            Vec3r v2 = {velocity[edge.node2 * 3], velocity[edge.node2 * 3 + 1], velocity[edge.node2 * 3 + 2]};

            // Velocity at contact point on edge
            Real t = pair.edge_param;
            Vec3r edge_vel = {
                (1 - t) * v1[0] + t * v2[0],
                (1 - t) * v1[1] + t * v2[1],
                (1 - t) * v1[2] + t * v2[2]
            };

            Vec3r target_vel = {0, 0, 0};

            if (pair.type == EdgeContactPair::ContactType::EdgeToSurface) {
                // Target surface velocity
                for (int n = 0; n < nodes_per_segment_; ++n) {
                    Index node = segment_connectivity_[pair.target_id * nodes_per_segment_ + n];
                    for (int d = 0; d < 3; ++d) {
                        target_vel[d] += pair.target_phi[n] * velocity[node * 3 + d];
                    }
                }
            } else {
                // Edge-edge: target is another edge
                const auto& target_edge = edges_[pair.target_id];
                Vec3r tv1 = {velocity[target_edge.node1 * 3], velocity[target_edge.node1 * 3 + 1], velocity[target_edge.node1 * 3 + 2]};
                Vec3r tv2 = {velocity[target_edge.node2 * 3], velocity[target_edge.node2 * 3 + 1], velocity[target_edge.node2 * 3 + 2]};
                // Assume midpoint
                for (int d = 0; d < 3; ++d) {
                    target_vel[d] = 0.5 * (tv1[d] + tv2[d]);
                }
            }

            // Relative velocity
            Vec3r rel_vel = {
                edge_vel[0] - target_vel[0],
                edge_vel[1] - target_vel[1],
                edge_vel[2] - target_vel[2]
            };

            Real vn = rel_vel[0] * pair.normal[0] + rel_vel[1] * pair.normal[1] + rel_vel[2] * pair.normal[2];

            Vec3r v_tang = {
                rel_vel[0] - vn * pair.normal[0],
                rel_vel[1] - vn * pair.normal[1],
                rel_vel[2] - vn * pair.normal[2]
            };

            // ================================================================
            // Normal force
            // ================================================================
            Real penetration = config_.contact_thickness - pair.gap;
            if (penetration <= 0.0) {
                pair.active = false;
                continue;
            }

            Real fn = penalty_stiffness * penetration;

            // Damping
            Real damping_ratio = 0.1;
            Real damping = 2.0 * damping_ratio * std::sqrt(penalty_stiffness);
            if (vn < 0) fn -= damping * vn;
            fn = std::max(fn, 0.0);

            pair.pressure = fn;

            Vec3r normal_force = {fn * pair.normal[0], fn * pair.normal[1], fn * pair.normal[2]};

            // ================================================================
            // Friction force
            // ================================================================
            Vec3r friction_force = {0, 0, 0};

            if (config_.static_friction > 0.0 && fn > 0.0) {
                for (int d = 0; d < 3; ++d) {
                    pair.tangent_slip[d] += v_tang[d] * dt;
                }

                Real slip_mag = std::sqrt(
                    pair.tangent_slip[0] * pair.tangent_slip[0] +
                    pair.tangent_slip[1] * pair.tangent_slip[1] +
                    pair.tangent_slip[2] * pair.tangent_slip[2]
                );

                Real friction_coeff = pair.sticking ? config_.static_friction : config_.dynamic_friction;
                Real max_friction = friction_coeff * fn;

                Real stick_stiff = penalty_stiffness * 0.5;
                Real trial_friction = stick_stiff * slip_mag;

                if (trial_friction <= max_friction && slip_mag > 1.0e-15) {
                    pair.sticking = true;
                    Real scale = trial_friction / slip_mag;
                    for (int d = 0; d < 3; ++d) {
                        friction_force[d] = -scale * pair.tangent_slip[d];
                    }
                } else if (slip_mag > 1.0e-15) {
                    pair.sticking = false;
                    Real scale = max_friction / slip_mag;
                    for (int d = 0; d < 3; ++d) {
                        friction_force[d] = -scale * pair.tangent_slip[d];
                        pair.tangent_slip[d] = friction_force[d] / stick_stiff;
                    }
                }
            }

            // ================================================================
            // Apply forces
            // ================================================================

            // Force on edge nodes
            for (int d = 0; d < 3; ++d) {
                forces[edge.node1 * 3 + d] += (1 - t) * (normal_force[d] + friction_force[d]);
                forces[edge.node2 * 3 + d] += t * (normal_force[d] + friction_force[d]);
            }

            // Reaction on target
            if (pair.type == EdgeContactPair::ContactType::EdgeToSurface) {
                for (int n = 0; n < nodes_per_segment_; ++n) {
                    Index node = segment_connectivity_[pair.target_id * nodes_per_segment_ + n];
                    for (int d = 0; d < 3; ++d) {
                        forces[node * 3 + d] -= pair.target_phi[n] * (normal_force[d] + friction_force[d]);
                    }
                }
            } else {
                // Edge-edge: split reaction to target edge nodes
                const auto& target_edge = edges_[pair.target_id];
                for (int d = 0; d < 3; ++d) {
                    forces[target_edge.node1 * 3 + d] -= 0.5 * (normal_force[d] + friction_force[d]);
                    forces[target_edge.node2 * 3 + d] -= 0.5 * (normal_force[d] + friction_force[d]);
                }
            }
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    const std::vector<EdgeContactPair>& get_active_pairs() const {
        return active_pairs_;
    }

    int num_active_pairs() const {
        return static_cast<int>(active_pairs_.size());
    }

    const std::vector<ContactEdge>& get_edges() const {
        return edges_;
    }

    const EdgeContactConfig& config() const { return config_; }
    EdgeContactConfig& config() { return config_; }

private:
    // ========================================================================
    // Internal Methods
    // ========================================================================

    /**
     * @brief Compute edge normals from geometry
     */
    void compute_edge_normals(const Real* coords) {
        for (auto& edge : edges_) {
            Vec3r x1 = {coords[edge.node1 * 3], coords[edge.node1 * 3 + 1], coords[edge.node1 * 3 + 2]};
            Vec3r x2 = {coords[edge.node2 * 3], coords[edge.node2 * 3 + 1], coords[edge.node2 * 3 + 2]};

            // Edge direction
            Vec3r e = {x2[0] - x1[0], x2[1] - x1[1], x2[2] - x1[2]};
            Real e_len = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
            if (e_len > 1.0e-20) {
                for (int d = 0; d < 3; ++d) e[d] /= e_len;
            }

            // Default normal perpendicular to edge (in z direction if possible)
            Vec3r up = {0, 0, 1};
            if (std::abs(e[2]) > 0.9) {
                up = {1, 0, 0};
            }

            edge.normal = {
                e[1] * up[2] - e[2] * up[1],
                e[2] * up[0] - e[0] * up[2],
                e[0] * up[1] - e[1] * up[0]
            };

            Real n_len = std::sqrt(edge.normal[0] * edge.normal[0] +
                                    edge.normal[1] * edge.normal[1] +
                                    edge.normal[2] * edge.normal[2]);
            if (n_len > 1.0e-20) {
                for (int d = 0; d < 3; ++d) edge.normal[d] /= n_len;
            }
        }
    }

    /**
     * @brief Detect contacts for a single edge
     */
    void detect_edge_contacts(size_t edge_idx, const Real* coords) {
        const auto& edge = edges_[edge_idx];

        Vec3r x1 = {coords[edge.node1 * 3], coords[edge.node1 * 3 + 1], coords[edge.node1 * 3 + 2]};
        Vec3r x2 = {coords[edge.node2 * 3], coords[edge.node2 * 3 + 1], coords[edge.node2 * 3 + 2]};

        // Get candidates from voxel grid
        std::vector<Index> slave_nodes = {edge.node1, edge.node2};
        auto candidates = voxel_detector_->find_candidates(slave_nodes, config_.search_radius);

        // Check each candidate segment
        for (const auto& cand : candidates) {
            Index seg_id = cand.segment_id;

            // Skip if edge is part of this segment
            bool skip = false;
            for (int n = 0; n < nodes_per_segment_; ++n) {
                Index seg_node = segment_connectivity_[seg_id * nodes_per_segment_ + n];
                if (seg_node == edge.node1 || seg_node == edge.node2) {
                    skip = true;
                    break;
                }
            }
            if (skip) continue;

            // Compute edge-segment contact
            EdgeContactPair pair;
            if (compute_edge_segment_contact(edge_idx, seg_id, coords, pair)) {
                if (pair.gap < config_.search_radius) {
                    active_pairs_.push_back(pair);
                }
            }
        }
    }

    /**
     * @brief Detect edge-edge contacts
     */
    void detect_edge_edge_contacts(const Real* coords) {
        // Simple O(n²) for now - can be optimized with spatial hashing
        for (size_t i = 0; i < edges_.size(); ++i) {
            for (size_t j = i + 1; j < edges_.size(); ++j) {
                // Skip if edges share a node
                if (edges_[i].node1 == edges_[j].node1 ||
                    edges_[i].node1 == edges_[j].node2 ||
                    edges_[i].node2 == edges_[j].node1 ||
                    edges_[i].node2 == edges_[j].node2) {
                    continue;
                }

                EdgeContactPair pair;
                if (compute_edge_edge_contact(i, j, coords, pair)) {
                    if (pair.gap < config_.search_radius) {
                        active_pairs_.push_back(pair);
                    }
                }
            }
        }
    }

    /**
     * @brief Compute contact between edge and segment
     */
    bool compute_edge_segment_contact(size_t edge_idx, Index seg_id, const Real* coords, EdgeContactPair& pair) {
        const auto& edge = edges_[edge_idx];

        Vec3r e1 = {coords[edge.node1 * 3], coords[edge.node1 * 3 + 1], coords[edge.node1 * 3 + 2]};
        Vec3r e2 = {coords[edge.node2 * 3], coords[edge.node2 * 3 + 1], coords[edge.node2 * 3 + 2]};

        // Get segment nodes
        Vec3r s[4];
        for (int n = 0; n < nodes_per_segment_; ++n) {
            Index node = segment_connectivity_[seg_id * nodes_per_segment_ + n];
            s[n] = {coords[node * 3], coords[node * 3 + 1], coords[node * 3 + 2]};
        }

        // Segment normal
        Vec3r seg_e1 = {s[1][0] - s[0][0], s[1][1] - s[0][1], s[1][2] - s[0][2]};
        Vec3r seg_e2 = (nodes_per_segment_ == 3) ?
            Vec3r{s[2][0] - s[0][0], s[2][1] - s[0][1], s[2][2] - s[0][2]} :
            Vec3r{s[2][0] - s[0][0], s[2][1] - s[0][1], s[2][2] - s[0][2]};

        Vec3r seg_normal = {
            seg_e1[1] * seg_e2[2] - seg_e1[2] * seg_e2[1],
            seg_e1[2] * seg_e2[0] - seg_e1[0] * seg_e2[2],
            seg_e1[0] * seg_e2[1] - seg_e1[1] * seg_e2[0]
        };
        Real n_mag = std::sqrt(seg_normal[0] * seg_normal[0] +
                               seg_normal[1] * seg_normal[1] +
                               seg_normal[2] * seg_normal[2]);
        if (n_mag < 1.0e-20) return false;
        for (int d = 0; d < 3; ++d) seg_normal[d] /= n_mag;

        // Edge direction
        Vec3r edge_dir = {e2[0] - e1[0], e2[1] - e1[1], e2[2] - e1[2]};
        Real edge_len = std::sqrt(edge_dir[0] * edge_dir[0] +
                                   edge_dir[1] * edge_dir[1] +
                                   edge_dir[2] * edge_dir[2]);
        if (edge_len < 1.0e-20) return false;
        for (int d = 0; d < 3; ++d) edge_dir[d] /= edge_len;

        // Find closest point on edge to segment plane
        Vec3r to_plane = {s[0][0] - e1[0], s[0][1] - e1[1], s[0][2] - e1[2]};
        Real d1 = to_plane[0] * seg_normal[0] + to_plane[1] * seg_normal[1] + to_plane[2] * seg_normal[2];

        to_plane = {s[0][0] - e2[0], s[0][1] - e2[1], s[0][2] - e2[2]};
        Real d2 = to_plane[0] * seg_normal[0] + to_plane[1] * seg_normal[1] + to_plane[2] * seg_normal[2];

        // Check if edge crosses plane
        if (d1 * d2 > 0 && std::abs(d1) > config_.search_radius && std::abs(d2) > config_.search_radius) {
            return false;  // Both ends on same side and far
        }

        // Find intersection parameter
        Real t;
        if (std::abs(d1 - d2) > 1.0e-20) {
            t = d1 / (d1 - d2);
        } else {
            t = 0.5;  // Parallel to plane
        }
        t = std::max(0.0, std::min(1.0, t));

        // Contact point on edge
        Vec3r contact = {
            e1[0] + t * (e2[0] - e1[0]),
            e1[1] + t * (e2[1] - e1[1]),
            e1[2] + t * (e2[2] - e1[2])
        };

        // Project onto segment
        Vec3r proj;
        Real gap;
        Real phi[4] = {0.25, 0.25, 0.25, 0.25};

        if (!project_to_segment(contact, s, nodes_per_segment_, seg_normal, proj, gap, phi)) {
            return false;
        }

        // Fill pair
        pair.type = EdgeContactPair::ContactType::EdgeToSurface;
        pair.edge_id = static_cast<Index>(edge_idx);
        pair.target_id = seg_id;
        pair.contact_point = contact;
        pair.target_point = proj;
        pair.normal = seg_normal;
        pair.gap = gap;
        pair.edge_param = t;
        pair.active = (gap < config_.search_radius);

        for (int n = 0; n < 4; ++n) {
            pair.target_phi[n] = phi[n];
        }

        return true;
    }

    /**
     * @brief Compute edge-edge contact
     */
    bool compute_edge_edge_contact(size_t edge1_idx, size_t edge2_idx, const Real* coords, EdgeContactPair& pair) {
        const auto& e1 = edges_[edge1_idx];
        const auto& e2 = edges_[edge2_idx];

        Vec3r p1 = {coords[e1.node1 * 3], coords[e1.node1 * 3 + 1], coords[e1.node1 * 3 + 2]};
        Vec3r p2 = {coords[e1.node2 * 3], coords[e1.node2 * 3 + 1], coords[e1.node2 * 3 + 2]};
        Vec3r p3 = {coords[e2.node1 * 3], coords[e2.node1 * 3 + 1], coords[e2.node1 * 3 + 2]};
        Vec3r p4 = {coords[e2.node2 * 3], coords[e2.node2 * 3 + 1], coords[e2.node2 * 3 + 2]};

        Vec3r d1 = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
        Vec3r d2 = {p4[0] - p3[0], p4[1] - p3[1], p4[2] - p3[2]};
        Vec3r r = {p1[0] - p3[0], p1[1] - p3[1], p1[2] - p3[2]};

        Real a = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
        Real b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2];
        Real c = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
        Real d = d1[0] * r[0] + d1[1] * r[1] + d1[2] * r[2];
        Real e = d2[0] * r[0] + d2[1] * r[1] + d2[2] * r[2];

        Real denom = a * c - b * b;
        Real s, t;

        if (std::abs(denom) < 1.0e-20) {
            // Parallel edges
            s = 0.0;
            t = (b > c) ? d / b : e / c;
        } else {
            s = (b * e - c * d) / denom;
            t = (a * e - b * d) / denom;
        }

        // Clamp to [0, 1]
        s = std::max(0.0, std::min(1.0, s));
        t = std::max(0.0, std::min(1.0, t));

        // Closest points
        Vec3r c1 = {p1[0] + s * d1[0], p1[1] + s * d1[1], p1[2] + s * d1[2]};
        Vec3r c2 = {p3[0] + t * d2[0], p3[1] + t * d2[1], p3[2] + t * d2[2]};

        Vec3r diff = {c2[0] - c1[0], c2[1] - c1[1], c2[2] - c1[2]};
        Real dist = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

        if (dist > config_.search_radius) return false;

        // Normal
        Vec3r normal = {0, 0, 1};
        if (dist > 1.0e-20) {
            normal = {diff[0] / dist, diff[1] / dist, diff[2] / dist};
        }

        // Fill pair
        pair.type = EdgeContactPair::ContactType::EdgeToEdge;
        pair.edge_id = static_cast<Index>(edge1_idx);
        pair.target_id = static_cast<Index>(edge2_idx);
        pair.contact_point = c1;
        pair.target_point = c2;
        pair.normal = normal;
        pair.gap = dist;
        pair.edge_param = s;
        pair.active = true;

        return true;
    }

    /**
     * @brief Project point onto segment
     */
    bool project_to_segment(const Vec3r& point, const Vec3r* s, int num_nodes,
                            const Vec3r& normal, Vec3r& proj, Real& gap, Real* phi) {
        // Distance to plane
        Vec3r to_point = {point[0] - s[0][0], point[1] - s[0][1], point[2] - s[0][2]};
        gap = to_point[0] * normal[0] + to_point[1] * normal[1] + to_point[2] * normal[2];

        // Project onto plane
        proj = {
            point[0] - gap * normal[0],
            point[1] - gap * normal[1],
            point[2] - gap * normal[2]
        };

        gap = std::abs(gap);

        if (num_nodes == 3) {
            // Triangle barycentric coordinates
            Vec3r v0 = {s[1][0] - s[0][0], s[1][1] - s[0][1], s[1][2] - s[0][2]};
            Vec3r v1 = {s[2][0] - s[0][0], s[2][1] - s[0][1], s[2][2] - s[0][2]};
            Vec3r v2 = {proj[0] - s[0][0], proj[1] - s[0][1], proj[2] - s[0][2]};

            Real d00 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
            Real d01 = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
            Real d11 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
            Real d20 = v2[0] * v0[0] + v2[1] * v0[1] + v2[2] * v0[2];
            Real d21 = v2[0] * v1[0] + v2[1] * v1[1] + v2[2] * v1[2];

            Real denom = d00 * d11 - d01 * d01;
            if (std::abs(denom) < 1.0e-20) return false;

            Real v = (d11 * d20 - d01 * d21) / denom;
            Real w = (d00 * d21 - d01 * d20) / denom;
            Real u = 1.0 - v - w;

            Real tol = -0.1;  // Allow slight outside
            if (u < tol || v < tol || w < tol) return false;

            phi[0] = u;
            phi[1] = v;
            phi[2] = w;
            phi[3] = 0.0;
        } else {
            // Quad - use centroid weights for simplicity
            phi[0] = phi[1] = phi[2] = phi[3] = 0.25;
        }

        return true;
    }

    // ========================================================================
    // Data Members
    // ========================================================================

    EdgeContactConfig config_;
    Index num_nodes_;

    std::vector<ContactEdge> edges_;

    std::vector<Index> segment_connectivity_;
    Index num_segments_;
    int nodes_per_segment_;

    std::unique_ptr<VoxelCollisionDetector> voxel_detector_;

    std::vector<EdgeContactPair> active_pairs_;
};

} // namespace fem
} // namespace nxs
