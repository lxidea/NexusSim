#pragma once

/**
 * @file contact_wave12.hpp
 * @brief Contact expansion: 9 advanced contact capabilities for NexusSim
 *
 * Implements:
 * 1. SelfContact - Single-surface self-contact with bucket-sort spatial hash
 * 2. EdgeToEdgeContact - Edge-pair crossing/proximity detection
 * 3. SegmentBasedContact - Face-to-face overlap with distributed penalty
 * 4. SymmetricContact - Bidirectional master-slave penalty (no surface bias)
 * 5. RigidDeformableContact - Rigid master surface with simplified projection
 * 6. ShellThicknessContact - Shell-thickness-aware gap computation
 * 7. MultiSurfaceContactManager - Multi-interface dispatch and auto-detection
 * 8. VelocityDependentFriction - mu(v) = mu_d + (mu_s - mu_d)*exp(-decay*|v_t|)
 * 9. ContactStiffnessScaling - Segment-based penalty: k = alpha*K*A/d
 */

#include <nexussim/core/types.hpp>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <string>
#include <limits>
#include <cassert>

namespace nxs {
namespace fem {

// ============================================================================
// Forward declarations and utility helpers
// ============================================================================

namespace detail {

/// Dot product of two 3-vectors stored as Real[3]
inline Real dot3(const Real a[3], const Real b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// Cross product c = a x b
inline void cross3(const Real a[3], const Real b[3], Real c[3]) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

/// Euclidean norm of a 3-vector
inline Real norm3(const Real v[3]) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/// Normalize in-place; returns original length
inline Real normalize3(Real v[3]) {
    Real len = norm3(v);
    if (len > 1.0e-30) {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
    return len;
}

/// Subtraction: c = a - b
inline void sub3(const Real a[3], const Real b[3], Real c[3]) {
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
}

/// Addition: c = a + b
inline void add3(const Real a[3], const Real b[3], Real c[3]) {
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
    c[2] = a[2] + b[2];
}

/// Scale: c = alpha * a
inline void scale3(Real alpha, const Real a[3], Real c[3]) {
    c[0] = alpha * a[0];
    c[1] = alpha * a[1];
    c[2] = alpha * a[2];
}

/// Midpoint of two 3-vectors
inline void midpoint3(const Real a[3], const Real b[3], Real m[3]) {
    m[0] = 0.5 * (a[0] + b[0]);
    m[1] = 0.5 * (a[1] + b[1]);
    m[2] = 0.5 * (a[2] + b[2]);
}

/// Distance between two points
inline Real distance3(const Real a[3], const Real b[3]) {
    Real d[3];
    sub3(a, b, d);
    return norm3(d);
}

/// Compute quad face normal from 4 corner coordinates (each Real[3])
/// Returns area * 2
inline Real quad_normal(const Real c0[3], const Real c1[3],
                        const Real c2[3], const Real c3[3],
                        Real normal[3]) {
    // Diagonals: d1 = c2 - c0, d2 = c3 - c1
    Real d1[3], d2[3];
    sub3(c2, c0, d1);
    sub3(c3, c1, d2);
    cross3(d1, d2, normal);
    return normalize3(normal);
}

/// Compute quad face area (approximate as half of cross-product of diagonals)
inline Real quad_area(const Real c0[3], const Real c1[3],
                      const Real c2[3], const Real c3[3]) {
    Real d1[3], d2[3], cr[3];
    sub3(c2, c0, d1);
    sub3(c3, c1, d2);
    cross3(d1, d2, cr);
    return 0.5 * norm3(cr);
}

/// Compute quad face diagonal length (max of both diagonals)
inline Real quad_diagonal(const Real c0[3], const Real c1[3],
                          const Real c2[3], const Real c3[3]) {
    return std::max(distance3(c0, c2), distance3(c1, c3));
}

/// Centroid of a quad face
inline void quad_centroid(const Real c0[3], const Real c1[3],
                          const Real c2[3], const Real c3[3],
                          Real cen[3]) {
    cen[0] = 0.25 * (c0[0] + c1[0] + c2[0] + c3[0]);
    cen[1] = 0.25 * (c0[1] + c1[1] + c2[1] + c3[1]);
    cen[2] = 0.25 * (c0[2] + c1[2] + c2[2] + c3[2]);
}

/// Spatial hash key for bucket sort
inline int64_t spatial_hash_key(int ix, int iy, int iz, int dim_y, int dim_z) {
    return static_cast<int64_t>(ix) * dim_y * dim_z +
           static_cast<int64_t>(iy) * dim_z +
           static_cast<int64_t>(iz);
}

} // namespace detail

// ============================================================================
// 9. ContactStiffnessScaling
// ============================================================================

/**
 * @brief Segment-based penalty stiffness: k = scale_factor * bulk_modulus * area / diagonal
 *
 * Auto-scales per-segment to ensure stable and consistent contact response
 * regardless of segment size variation.
 */
struct ContactStiffnessScaler {
    Real scale_factor;  ///< User-defined multiplier (default 1.0)

    ContactStiffnessScaler() : scale_factor(1.0) {}
    explicit ContactStiffnessScaler(Real sf) : scale_factor(sf) {}

    /**
     * @brief Compute penalty stiffness for a segment
     * @param bulk_modulus Material bulk modulus (Pa)
     * @param area Segment area (m^2)
     * @param diagonal Segment diagonal length (m)
     * @return Penalty stiffness (N/m)
     */
    Real compute_stiffness(Real bulk_modulus, Real area, Real diagonal) const {
        if (diagonal < 1.0e-30) return 0.0;
        return scale_factor * bulk_modulus * area / diagonal;
    }
};

// ============================================================================
// 8. VelocityDependentFriction
// ============================================================================

/**
 * @brief Velocity-dependent friction model
 *
 * mu(v) = mu_kinetic + (mu_static - mu_kinetic) * exp(-decay_coefficient * |v_tangential|)
 *
 * At zero velocity: mu = mu_static
 * At high velocity: mu -> mu_kinetic
 */
struct VelocityDependentFrictionModel {
    Real mu_static;           ///< Static friction coefficient
    Real mu_kinetic;          ///< Kinetic (dynamic) friction coefficient
    Real decay_coefficient;   ///< Exponential decay rate (1/velocity)

    VelocityDependentFrictionModel()
        : mu_static(0.3)
        , mu_kinetic(0.2)
        , decay_coefficient(10.0)
    {}

    VelocityDependentFrictionModel(Real ms, Real mk, Real dc)
        : mu_static(ms), mu_kinetic(mk), decay_coefficient(dc)
    {}

    /**
     * @brief Compute effective friction coefficient at given tangential velocity
     * @param v_tangential Magnitude of tangential velocity
     * @return Effective friction coefficient
     */
    Real compute_friction(Real v_tangential) const {
        Real v_abs = std::abs(v_tangential);
        return mu_kinetic + (mu_static - mu_kinetic) * std::exp(-decay_coefficient * v_abs);
    }
};

// ============================================================================
// 6. ShellThicknessContact
// ============================================================================

/**
 * @brief Configuration for shell-thickness-aware contact
 *
 * Gap = distance - slave_thickness/2 - master_thickness/2
 */
struct ShellContactConfig {
    Real slave_thickness;    ///< Slave surface shell thickness
    Real master_thickness;   ///< Master surface shell thickness
    bool offset_enabled;     ///< Enable thickness offset in gap computation

    ShellContactConfig()
        : slave_thickness(0.001)
        , master_thickness(0.001)
        , offset_enabled(true)
    {}

    ShellContactConfig(Real st, Real mt, bool enabled = true)
        : slave_thickness(st), master_thickness(mt), offset_enabled(enabled)
    {}

    /**
     * @brief Compute effective gap accounting for shell thickness
     * @param raw_distance Raw distance between surfaces
     * @return Effective gap (negative = penetration)
     */
    Real compute_gap(Real raw_distance) const {
        if (!offset_enabled) return raw_distance;
        return raw_distance - slave_thickness * 0.5 - master_thickness * 0.5;
    }
};

/**
 * @brief Shell-thickness-aware contact detection and force computation
 */
class ShellThicknessContact {
public:
    ShellThicknessContact() = default;
    explicit ShellThicknessContact(const ShellContactConfig& config)
        : config_(config) {}

    void set_config(const ShellContactConfig& config) { config_ = config; }
    const ShellContactConfig& config() const { return config_; }

    /**
     * @brief Compute gap with shell thickness offset
     */
    Real compute_gap(Real raw_distance) const {
        return config_.compute_gap(raw_distance);
    }

    /**
     * @brief Compute penalty force for shell contact
     * @param raw_distance Raw distance between mid-surfaces
     * @param stiffness Penalty stiffness
     * @return Penalty force magnitude (positive = repulsive)
     */
    Real compute_force(Real raw_distance, Real stiffness) const {
        Real gap = config_.compute_gap(raw_distance);
        if (gap >= 0.0) return 0.0;
        return -stiffness * gap;  // gap is negative, so force is positive
    }

private:
    ShellContactConfig config_;
};

// ============================================================================
// 2. EdgeToEdgeContact
// ============================================================================

/**
 * @brief Edge pair result from edge-to-edge contact detection
 */
struct EdgePair {
    Index edge1_n1;   ///< First node of edge 1
    Index edge1_n2;   ///< Second node of edge 1
    Index edge2_n1;   ///< First node of edge 2
    Index edge2_n2;   ///< Second node of edge 2
    Real gap;         ///< Minimum distance between edges
    Real normal[3];   ///< Contact normal (from edge1 to edge2)

    /// Parametric position on edge 1 [0,1]
    Real s;
    /// Parametric position on edge 2 [0,1]
    Real t;

    EdgePair()
        : edge1_n1(-1), edge1_n2(-1)
        , edge2_n1(-1), edge2_n2(-1)
        , gap(0.0), s(0.5), t(0.5)
    {
        normal[0] = normal[1] = 0.0;
        normal[2] = 1.0;
    }
};

/**
 * @brief Edge-to-edge contact detection
 *
 * Checks pairs of edges for proximity/crossing. For each edge pair, computes
 * the closest points on two line segments and the gap distance.
 */
class EdgeToEdgeContact {
public:
    EdgeToEdgeContact() : search_radius_(0.1), penalty_stiffness_(1.0e6) {}

    void set_search_radius(Real r) { search_radius_ = r; }
    void set_penalty_stiffness(Real k) { penalty_stiffness_ = k; }
    Real search_radius() const { return search_radius_; }
    Real penalty_stiffness() const { return penalty_stiffness_; }

    /**
     * @brief Add an edge defined by two node indices
     */
    void add_edge(Index n1, Index n2) {
        edges_.push_back({n1, n2});
    }

    /**
     * @brief Clear all edges
     */
    void clear_edges() { edges_.clear(); }

    /**
     * @brief Number of registered edges
     */
    std::size_t num_edges() const { return edges_.size(); }

    /**
     * @brief Detect edge-to-edge contacts
     * @param coords Nodal coordinates (3*num_nodes)
     * @return Vector of detected edge pairs
     */
    std::vector<EdgePair> detect_edge_contact(const Real* coords) const {
        std::vector<EdgePair> pairs;

        for (std::size_t i = 0; i < edges_.size(); ++i) {
            for (std::size_t j = i + 1; j < edges_.size(); ++j) {
                // Skip edges sharing a node
                if (edges_[i].first == edges_[j].first ||
                    edges_[i].first == edges_[j].second ||
                    edges_[i].second == edges_[j].first ||
                    edges_[i].second == edges_[j].second) {
                    continue;
                }

                EdgePair ep;
                if (compute_closest_points(edges_[i], edges_[j], coords, ep)) {
                    if (ep.gap <= search_radius_) {
                        pairs.push_back(ep);
                    }
                }
            }
        }

        return pairs;
    }

    /**
     * @brief Compute contact forces for detected edge pairs
     * @param pair Edge pair with contact information
     * @param forces Output force array (accumulated)
     */
    void compute_edge_force(const EdgePair& pair, Real* forces) const {
        if (pair.gap >= search_radius_ || pair.gap < 0.0) return;

        Real penetration = search_radius_ - pair.gap;
        if (penetration <= 0.0) return;

        Real fn = penalty_stiffness_ * penetration;

        // Distribute force to edge nodes
        Real f1 = fn * (1.0 - pair.s);
        Real f2 = fn * pair.s;
        Real f3 = fn * (1.0 - pair.t);
        Real f4 = fn * pair.t;

        for (int d = 0; d < 3; ++d) {
            forces[pair.edge1_n1 * 3 + d] += f1 * pair.normal[d];
            forces[pair.edge1_n2 * 3 + d] += f2 * pair.normal[d];
            forces[pair.edge2_n1 * 3 + d] -= f3 * pair.normal[d];
            forces[pair.edge2_n2 * 3 + d] -= f4 * pair.normal[d];
        }
    }

private:
    using Edge = std::pair<Index, Index>;
    std::vector<Edge> edges_;
    Real search_radius_;
    Real penalty_stiffness_;

    /**
     * @brief Compute closest points between two line segments
     *
     * Segment 1: P(s) = p1 + s*(p2-p1), s in [0,1]
     * Segment 2: Q(t) = q1 + t*(q2-q1), t in [0,1]
     */
    bool compute_closest_points(const Edge& e1, const Edge& e2,
                                const Real* coords, EdgePair& pair) const {
        const Real* p1 = &coords[e1.first * 3];
        const Real* p2 = &coords[e1.second * 3];
        const Real* q1 = &coords[e2.first * 3];
        const Real* q2 = &coords[e2.second * 3];

        Real d1[3], d2[3], r[3];
        detail::sub3(p2, p1, d1);  // d1 = p2 - p1
        detail::sub3(q2, q1, d2);  // d2 = q2 - q1
        detail::sub3(p1, q1, r);   // r  = p1 - q1

        Real a = detail::dot3(d1, d1);  // |d1|^2
        Real b = detail::dot3(d1, d2);
        Real c = detail::dot3(d2, d2);  // |d2|^2
        Real d = detail::dot3(d1, r);
        Real e = detail::dot3(d2, r);

        Real denom = a * c - b * b;

        Real s, t;
        if (std::abs(denom) < 1.0e-20) {
            // Nearly parallel segments
            s = 0.0;
            t = (b > c) ? (d / b) : (e / c);
        } else {
            s = (b * e - c * d) / denom;
            t = (a * e - b * d) / denom;
        }

        // Clamp to [0, 1]
        s = std::max(0.0, std::min(1.0, s));
        t = std::max(0.0, std::min(1.0, t));

        // Recompute closest points
        Real cp1[3], cp2[3];
        for (int i = 0; i < 3; ++i) {
            cp1[i] = p1[i] + s * d1[i];
            cp2[i] = q1[i] + t * d2[i];
        }

        Real diff[3];
        detail::sub3(cp2, cp1, diff);
        Real dist = detail::norm3(diff);

        // Fill result
        pair.edge1_n1 = e1.first;
        pair.edge1_n2 = e1.second;
        pair.edge2_n1 = e2.first;
        pair.edge2_n2 = e2.second;
        pair.gap = dist;
        pair.s = s;
        pair.t = t;

        if (dist > 1.0e-20) {
            pair.normal[0] = diff[0] / dist;
            pair.normal[1] = diff[1] / dist;
            pair.normal[2] = diff[2] / dist;
        } else {
            pair.normal[0] = 0.0;
            pair.normal[1] = 0.0;
            pair.normal[2] = 1.0;
        }

        return true;
    }
};

// ============================================================================
// 3. SegmentBasedContact
// ============================================================================

/**
 * @brief Result of segment-based (face-to-face) contact detection
 */
struct SegmentPair {
    Index slave_face;    ///< Slave face index
    Index master_face;   ///< Master face index
    Real overlap_area;   ///< Estimated overlap area
    Real gap;            ///< Average gap between face pair
    Real normal[3];      ///< Contact normal (master -> slave direction)

    SegmentPair()
        : slave_face(-1), master_face(-1)
        , overlap_area(0.0), gap(0.0)
    {
        normal[0] = normal[1] = 0.0;
        normal[2] = 1.0;
    }
};

/**
 * @brief Segment-based (face-to-face) contact with bucket sort
 *
 * For each slave face, finds overlapping master faces using spatial hashing.
 * Computes overlap area and applies distributed penalty force.
 */
class SegmentBasedContact {
public:
    SegmentBasedContact()
        : penalty_stiffness_(1.0e6)
        , search_gap_(0.1)
        , num_slave_faces_(0)
        , num_master_faces_(0)
        , nodes_per_face_(4)
    {}

    void set_penalty_stiffness(Real k) { penalty_stiffness_ = k; }
    void set_search_gap(Real g) { search_gap_ = g; }

    /**
     * @brief Set slave surface faces
     * @param connectivity Node indices for all faces (nodes_per_face per face)
     * @param num_faces Number of faces
     * @param npf Nodes per face (3 or 4)
     */
    void set_slave_faces(const std::vector<Index>& connectivity,
                         Index num_faces, int npf = 4) {
        slave_conn_ = connectivity;
        num_slave_faces_ = num_faces;
        nodes_per_face_ = npf;
    }

    /**
     * @brief Set master surface faces
     */
    void set_master_faces(const std::vector<Index>& connectivity,
                          Index num_faces, int npf = 4) {
        master_conn_ = connectivity;
        num_master_faces_ = num_faces;
        nodes_per_face_ = npf;
    }

    /**
     * @brief Detect segment-based contacts using bucket sort
     * @param coords Nodal coordinates
     * @return Vector of detected segment pairs
     */
    std::vector<SegmentPair> detect(const Real* coords) const {
        std::vector<SegmentPair> pairs;
        if (num_slave_faces_ == 0 || num_master_faces_ == 0) return pairs;

        // Build spatial hash for master faces (bucket sort)
        Real bbox_min[3] = { 1e30,  1e30,  1e30};
        Real bbox_max[3] = {-1e30, -1e30, -1e30};

        // Compute centroids and bounding box for master faces
        std::vector<Real> master_centroids(num_master_faces_ * 3);
        for (Index f = 0; f < num_master_faces_; ++f) {
            Real cen[3] = {0, 0, 0};
            for (int n = 0; n < nodes_per_face_; ++n) {
                Index node = master_conn_[f * nodes_per_face_ + n];
                for (int d = 0; d < 3; ++d) {
                    cen[d] += coords[node * 3 + d];
                }
            }
            for (int d = 0; d < 3; ++d) {
                cen[d] /= nodes_per_face_;
                master_centroids[f * 3 + d] = cen[d];
                bbox_min[d] = std::min(bbox_min[d], cen[d]);
                bbox_max[d] = std::max(bbox_max[d], cen[d]);
            }
        }

        // Cell size based on search gap
        Real cell_size = search_gap_ * 3.0;
        if (cell_size < 1.0e-10) cell_size = 1.0;

        int dims[3];
        for (int d = 0; d < 3; ++d) {
            dims[d] = std::max(1, static_cast<int>((bbox_max[d] - bbox_min[d]) / cell_size) + 2);
        }

        // Insert master faces into buckets
        std::unordered_map<int64_t, std::vector<Index>> buckets;
        for (Index f = 0; f < num_master_faces_; ++f) {
            int ix = static_cast<int>((master_centroids[f * 3 + 0] - bbox_min[0]) / cell_size);
            int iy = static_cast<int>((master_centroids[f * 3 + 1] - bbox_min[1]) / cell_size);
            int iz = static_cast<int>((master_centroids[f * 3 + 2] - bbox_min[2]) / cell_size);
            int64_t key = detail::spatial_hash_key(ix, iy, iz, dims[1], dims[2]);
            buckets[key].push_back(f);
        }

        // For each slave face, search nearby buckets
        for (Index sf = 0; sf < num_slave_faces_; ++sf) {
            Real slave_cen[3] = {0, 0, 0};
            Real slave_corners[4][3];
            for (int n = 0; n < nodes_per_face_; ++n) {
                Index node = slave_conn_[sf * nodes_per_face_ + n];
                for (int d = 0; d < 3; ++d) {
                    slave_corners[n][d] = coords[node * 3 + d];
                    slave_cen[d] += coords[node * 3 + d];
                }
            }
            for (int d = 0; d < 3; ++d) slave_cen[d] /= nodes_per_face_;

            // Slave face normal
            Real slave_normal[3];
            detail::quad_normal(slave_corners[0], slave_corners[1],
                                slave_corners[2], slave_corners[3],
                                slave_normal);

            int ix = static_cast<int>((slave_cen[0] - bbox_min[0]) / cell_size);
            int iy = static_cast<int>((slave_cen[1] - bbox_min[1]) / cell_size);
            int iz = static_cast<int>((slave_cen[2] - bbox_min[2]) / cell_size);

            // Check 3x3x3 neighborhood
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    for (int dk = -1; dk <= 1; ++dk) {
                        int64_t key = detail::spatial_hash_key(
                            ix + di, iy + dj, iz + dk, dims[1], dims[2]);
                        auto it = buckets.find(key);
                        if (it == buckets.end()) continue;

                        for (Index mf : it->second) {
                            SegmentPair sp;
                            if (check_segment_pair(sf, mf, coords,
                                                   slave_cen, slave_normal, sp)) {
                                pairs.push_back(sp);
                            }
                        }
                    }
                }
            }
        }

        return pairs;
    }

    /**
     * @brief Compute distributed penalty forces for a segment pair
     */
    void compute_segment_force(const SegmentPair& pair, Real* forces) const {
        if (pair.gap >= search_gap_ || pair.overlap_area <= 0.0) return;

        Real penetration = search_gap_ - pair.gap;
        if (penetration <= 0.0) return;

        // Distributed pressure = stiffness * penetration (force per area)
        // Total force = pressure * overlap_area
        Real total_force = penalty_stiffness_ * penetration * pair.overlap_area;

        // Distribute equally to slave face nodes
        Real per_node = total_force / nodes_per_face_;
        for (int n = 0; n < nodes_per_face_; ++n) {
            Index node = slave_conn_[pair.slave_face * nodes_per_face_ + n];
            for (int d = 0; d < 3; ++d) {
                forces[node * 3 + d] += per_node * pair.normal[d];
            }
        }
        // Reaction on master face
        for (int n = 0; n < nodes_per_face_; ++n) {
            Index node = master_conn_[pair.master_face * nodes_per_face_ + n];
            for (int d = 0; d < 3; ++d) {
                forces[node * 3 + d] -= per_node * pair.normal[d];
            }
        }
    }

    Index num_slave_faces() const { return num_slave_faces_; }
    Index num_master_faces() const { return num_master_faces_; }

private:
    Real penalty_stiffness_;
    Real search_gap_;
    std::vector<Index> slave_conn_;
    std::vector<Index> master_conn_;
    Index num_slave_faces_;
    Index num_master_faces_;
    int nodes_per_face_;

    /**
     * @brief Check if slave face sf and master face mf are in contact
     */
    bool check_segment_pair(Index sf, Index mf, const Real* coords,
                            const Real slave_cen[3],
                            const Real slave_normal[3],
                            SegmentPair& pair) const {
        // Master face centroid and normal
        Real master_corners[4][3];
        Real master_cen[3] = {0, 0, 0};
        for (int n = 0; n < nodes_per_face_; ++n) {
            Index node = master_conn_[mf * nodes_per_face_ + n];
            for (int d = 0; d < 3; ++d) {
                master_corners[n][d] = coords[node * 3 + d];
                master_cen[d] += coords[node * 3 + d];
            }
        }
        for (int d = 0; d < 3; ++d) master_cen[d] /= nodes_per_face_;

        Real master_normal[3];
        detail::quad_normal(master_corners[0], master_corners[1],
                            master_corners[2], master_corners[3],
                            master_normal);

        // Check that normals are roughly opposing (faces facing each other)
        Real ndot = detail::dot3(slave_normal, master_normal);
        // For faces facing each other, normals should be anti-parallel (dot < 0)
        // But accept a wide range for flexibility
        // (No strict filtering here to keep things general)

        // Compute gap as distance between centroids projected onto normal
        Real diff[3];
        detail::sub3(slave_cen, master_cen, diff);
        Real gap = std::abs(detail::dot3(diff, master_normal));

        if (gap > search_gap_) return false;

        // Estimate overlap area as min of both face areas (simplified)
        Real slave_area = detail::quad_area(
            &coords[slave_conn_[sf * nodes_per_face_ + 0] * 3],
            &coords[slave_conn_[sf * nodes_per_face_ + 1] * 3],
            &coords[slave_conn_[sf * nodes_per_face_ + 2] * 3],
            &coords[slave_conn_[sf * nodes_per_face_ + 3] * 3]);

        Real master_area = detail::quad_area(
            master_corners[0], master_corners[1],
            master_corners[2], master_corners[3]);

        // Lateral distance between centroids (perpendicular to normal)
        Real proj = detail::dot3(diff, master_normal);
        Real lateral[3];
        for (int d = 0; d < 3; ++d) {
            lateral[d] = diff[d] - proj * master_normal[d];
        }
        Real lateral_dist = detail::norm3(lateral);

        // Rough overlap check: if centroids are too far apart laterally, no overlap
        Real char_size = std::sqrt(std::min(slave_area, master_area));
        if (lateral_dist > char_size * 2.0) return false;

        // Overlap area estimation (simple linear decay)
        Real overlap_fraction = std::max(0.0, 1.0 - lateral_dist / (char_size + 1.0e-30));
        Real overlap = overlap_fraction * std::min(slave_area, master_area);

        pair.slave_face = sf;
        pair.master_face = mf;
        pair.overlap_area = overlap;
        pair.gap = gap;
        // Normal from master to slave
        for (int d = 0; d < 3; ++d) pair.normal[d] = master_normal[d];

        return true;
    }
};

// ============================================================================
// 1. SelfContact
// ============================================================================

/**
 * @brief Self-contact: single surface contacts itself
 *
 * Uses bucket sort spatial hash for broad phase.
 * Each face can be both master and slave.
 * Face-neighbor pairs (faces sharing a node) are excluded.
 */
class SelfContact {
public:
    SelfContact()
        : penalty_stiffness_(1.0e6)
        , search_radius_(0.1)
        , nodes_per_face_(4)
        , num_faces_(0)
    {}

    void set_penalty_stiffness(Real k) { penalty_stiffness_ = k; }
    void set_search_radius(Real r) { search_radius_ = r; }
    Real penalty_stiffness() const { return penalty_stiffness_; }
    Real search_radius() const { return search_radius_; }

    /**
     * @brief Set the surface connectivity
     * @param connectivity Node indices for all faces
     * @param num_faces Number of faces
     * @param npf Nodes per face (3 or 4)
     */
    void set_surface(const std::vector<Index>& connectivity,
                     Index num_faces, int npf = 4) {
        conn_ = connectivity;
        num_faces_ = num_faces;
        nodes_per_face_ = npf;

        // Build face-neighbor set (faces sharing at least one node)
        build_neighbor_set();
    }

    /**
     * @brief Detect self-contact pairs
     * @param coords Nodal coordinates (undeformed)
     * @param displacement Nodal displacements (can be nullptr)
     * @return Vector of contact pairs (using SegmentPair for face-face)
     */
    std::vector<SegmentPair> detect_self_contact(const Real* coords,
                                                  const Real* displacement) const {
        std::vector<SegmentPair> pairs;
        if (num_faces_ < 2) return pairs;

        // Get current coordinates
        std::vector<Real> cur_coords;
        const Real* effective_coords = coords;
        if (displacement) {
            // Find max node index
            Index max_node = 0;
            for (auto ni : conn_) max_node = std::max(max_node, ni);
            cur_coords.resize((max_node + 1) * 3);
            for (Index i = 0; i <= max_node; ++i) {
                for (int d = 0; d < 3; ++d) {
                    cur_coords[i * 3 + d] = coords[i * 3 + d] + displacement[i * 3 + d];
                }
            }
            effective_coords = cur_coords.data();
        }

        // Bucket sort spatial hash
        Real bbox_min[3] = { 1e30,  1e30,  1e30};
        Real bbox_max[3] = {-1e30, -1e30, -1e30};

        std::vector<Real> centroids(num_faces_ * 3);
        for (Index f = 0; f < num_faces_; ++f) {
            Real cen[3] = {0, 0, 0};
            for (int n = 0; n < nodes_per_face_; ++n) {
                Index node = conn_[f * nodes_per_face_ + n];
                for (int d = 0; d < 3; ++d) {
                    cen[d] += effective_coords[node * 3 + d];
                }
            }
            for (int d = 0; d < 3; ++d) {
                cen[d] /= nodes_per_face_;
                centroids[f * 3 + d] = cen[d];
                bbox_min[d] = std::min(bbox_min[d], cen[d]);
                bbox_max[d] = std::max(bbox_max[d], cen[d]);
            }
        }

        Real cell_size = search_radius_ * 2.0;
        if (cell_size < 1.0e-10) cell_size = 1.0;

        int dims[3];
        for (int d = 0; d < 3; ++d) {
            dims[d] = std::max(1, static_cast<int>((bbox_max[d] - bbox_min[d]) / cell_size) + 2);
        }

        std::unordered_map<int64_t, std::vector<Index>> buckets;
        for (Index f = 0; f < num_faces_; ++f) {
            int ix = static_cast<int>((centroids[f * 3 + 0] - bbox_min[0]) / cell_size);
            int iy = static_cast<int>((centroids[f * 3 + 1] - bbox_min[1]) / cell_size);
            int iz = static_cast<int>((centroids[f * 3 + 2] - bbox_min[2]) / cell_size);
            int64_t key = detail::spatial_hash_key(ix, iy, iz, dims[1], dims[2]);
            buckets[key].push_back(f);
        }

        // Track processed pairs to avoid duplicates
        std::unordered_set<uint64_t> processed;

        for (Index f = 0; f < num_faces_; ++f) {
            int ix = static_cast<int>((centroids[f * 3 + 0] - bbox_min[0]) / cell_size);
            int iy = static_cast<int>((centroids[f * 3 + 1] - bbox_min[1]) / cell_size);
            int iz = static_cast<int>((centroids[f * 3 + 2] - bbox_min[2]) / cell_size);

            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    for (int dk = -1; dk <= 1; ++dk) {
                        int64_t key = detail::spatial_hash_key(
                            ix + di, iy + dj, iz + dk, dims[1], dims[2]);
                        auto it = buckets.find(key);
                        if (it == buckets.end()) continue;

                        for (Index g : it->second) {
                            if (g <= f) continue;  // Only check f < g

                            // Skip neighbor faces (sharing a node)
                            if (are_neighbors(f, g)) continue;

                            // Create unique pair key
                            uint64_t pair_key = (static_cast<uint64_t>(f) << 32) |
                                                static_cast<uint64_t>(g);
                            if (processed.count(pair_key)) continue;
                            processed.insert(pair_key);

                            // Compute gap
                            Real diff[3];
                            detail::sub3(&centroids[f * 3], &centroids[g * 3], diff);

                            // Face normal of f
                            Real fn[3];
                            detail::quad_normal(
                                &effective_coords[conn_[f * nodes_per_face_ + 0] * 3],
                                &effective_coords[conn_[f * nodes_per_face_ + 1] * 3],
                                &effective_coords[conn_[f * nodes_per_face_ + 2] * 3],
                                &effective_coords[conn_[f * nodes_per_face_ + 3] * 3],
                                fn);

                            Real gap = std::abs(detail::dot3(diff, fn));

                            if (gap <= search_radius_) {
                                SegmentPair sp;
                                sp.slave_face = f;
                                sp.master_face = g;
                                sp.gap = gap;
                                sp.overlap_area = detail::quad_area(
                                    &effective_coords[conn_[f * nodes_per_face_ + 0] * 3],
                                    &effective_coords[conn_[f * nodes_per_face_ + 1] * 3],
                                    &effective_coords[conn_[f * nodes_per_face_ + 2] * 3],
                                    &effective_coords[conn_[f * nodes_per_face_ + 3] * 3]);
                                for (int d = 0; d < 3; ++d) sp.normal[d] = fn[d];
                                pairs.push_back(sp);
                            }
                        }
                    }
                }
            }
        }

        return pairs;
    }

    /**
     * @brief Check if two faces are neighbors (share a node)
     */
    bool are_neighbors(Index f1, Index f2) const {
        uint64_t key = (f1 < f2) ?
            ((static_cast<uint64_t>(f1) << 32) | static_cast<uint64_t>(f2)) :
            ((static_cast<uint64_t>(f2) << 32) | static_cast<uint64_t>(f1));
        return neighbor_set_.count(key) > 0;
    }

    Index num_faces() const { return num_faces_; }

private:
    Real penalty_stiffness_;
    Real search_radius_;
    int nodes_per_face_;
    Index num_faces_;
    std::vector<Index> conn_;
    std::unordered_set<uint64_t> neighbor_set_;

    void build_neighbor_set() {
        neighbor_set_.clear();
        // Build node->face adjacency
        std::unordered_map<Index, std::vector<Index>> node_to_faces;
        for (Index f = 0; f < num_faces_; ++f) {
            for (int n = 0; n < nodes_per_face_; ++n) {
                node_to_faces[conn_[f * nodes_per_face_ + n]].push_back(f);
            }
        }

        // Two faces are neighbors if they share at least one node
        for (const auto& [node, faces] : node_to_faces) {
            for (std::size_t i = 0; i < faces.size(); ++i) {
                for (std::size_t j = i + 1; j < faces.size(); ++j) {
                    Index a = std::min(faces[i], faces[j]);
                    Index b = std::max(faces[i], faces[j]);
                    uint64_t key = (static_cast<uint64_t>(a) << 32) |
                                    static_cast<uint64_t>(b);
                    neighbor_set_.insert(key);
                }
            }
        }
    }
};

// ============================================================================
// 4. SymmetricContact
// ============================================================================

/**
 * @brief Symmetric (two-pass) contact: both surfaces are simultaneously
 *        master and slave. Runs penalty from both directions and averages.
 *
 * Prevents surface bias that occurs in single-pass node-to-surface contact.
 */
class SymmetricContact {
public:
    SymmetricContact()
        : penalty_stiffness_(1.0e6)
        , search_gap_(0.1)
        , nodes_per_face_(4)
        , num_faces_A_(0)
        , num_faces_B_(0)
    {}

    void set_penalty_stiffness(Real k) { penalty_stiffness_ = k; }
    void set_search_gap(Real g) { search_gap_ = g; }

    /**
     * @brief Set surface A
     */
    void set_surface_A(const std::vector<Index>& conn, Index num_faces, int npf = 4) {
        conn_A_ = conn;
        num_faces_A_ = num_faces;
        nodes_per_face_ = npf;
    }

    /**
     * @brief Set surface B
     */
    void set_surface_B(const std::vector<Index>& conn, Index num_faces, int npf = 4) {
        conn_B_ = conn;
        num_faces_B_ = num_faces;
        nodes_per_face_ = npf;
    }

    /**
     * @brief Detect and compute symmetric contact forces
     *
     * Pass 1: A as slave, B as master
     * Pass 2: B as slave, A as master
     * Average the forces from both passes.
     *
     * @param coords Nodal coordinates
     * @param forces Output force array (accumulated)
     * @return Number of contact pairs detected
     */
    int compute_symmetric_forces(const Real* coords, Real* forces) const {
        // Pass 1: A->B
        SegmentBasedContact pass1;
        pass1.set_penalty_stiffness(penalty_stiffness_);
        pass1.set_search_gap(search_gap_);
        pass1.set_slave_faces(conn_A_, num_faces_A_, nodes_per_face_);
        pass1.set_master_faces(conn_B_, num_faces_B_, nodes_per_face_);
        auto pairs1 = pass1.detect(coords);

        // Find max node for force array sizing
        Index max_node = 0;
        for (auto ni : conn_A_) max_node = std::max(max_node, ni);
        for (auto ni : conn_B_) max_node = std::max(max_node, ni);
        std::size_t force_size = (max_node + 1) * 3;

        std::vector<Real> forces1(force_size, 0.0);
        for (const auto& p : pairs1) {
            pass1.compute_segment_force(p, forces1.data());
        }

        // Pass 2: B->A
        SegmentBasedContact pass2;
        pass2.set_penalty_stiffness(penalty_stiffness_);
        pass2.set_search_gap(search_gap_);
        pass2.set_slave_faces(conn_B_, num_faces_B_, nodes_per_face_);
        pass2.set_master_faces(conn_A_, num_faces_A_, nodes_per_face_);
        auto pairs2 = pass2.detect(coords);

        std::vector<Real> forces2(force_size, 0.0);
        for (const auto& p : pairs2) {
            pass2.compute_segment_force(p, forces2.data());
        }

        // Average forces from both passes
        for (std::size_t i = 0; i < force_size; ++i) {
            forces[i] += 0.5 * (forces1[i] + forces2[i]);
        }

        return static_cast<int>(pairs1.size() + pairs2.size());
    }

    int num_faces_A() const { return static_cast<int>(num_faces_A_); }
    int num_faces_B() const { return static_cast<int>(num_faces_B_); }

private:
    Real penalty_stiffness_;
    Real search_gap_;
    int nodes_per_face_;
    std::vector<Index> conn_A_;
    std::vector<Index> conn_B_;
    Index num_faces_A_;
    Index num_faces_B_;
};

// ============================================================================
// 5. RigidDeformableContact
// ============================================================================

/**
 * @brief Contact between a rigid body surface (master) and deformable surface (slave)
 *
 * Simplified contact: the rigid master surface normals are fixed (not updated),
 * only slave nodes move. Higher penalty stiffness by default.
 */
class RigidDeformableContact {
public:
    RigidDeformableContact()
        : penalty_stiffness_(1.0e8)  // Higher stiffness for rigid
        , search_gap_(0.1)
        , nodes_per_face_(4)
        , num_rigid_faces_(0)
    {}

    void set_penalty_stiffness(Real k) { penalty_stiffness_ = k; }
    void set_search_gap(Real g) { search_gap_ = g; }

    /**
     * @brief Set rigid master surface (positions are fixed)
     */
    void set_rigid_surface(const std::vector<Index>& connectivity,
                           Index num_faces, int npf = 4) {
        rigid_conn_ = connectivity;
        num_rigid_faces_ = num_faces;
        nodes_per_face_ = npf;
    }

    /**
     * @brief Set deformable slave nodes
     */
    void set_slave_nodes(const std::vector<Index>& nodes) {
        slave_nodes_ = nodes;
    }

    /**
     * @brief Precompute rigid surface normals and centroids (call once at setup)
     * @param coords Reference coordinates of the rigid body
     */
    void precompute_rigid_geometry(const Real* coords) {
        rigid_normals_.resize(num_rigid_faces_ * 3);
        rigid_centroids_.resize(num_rigid_faces_ * 3);

        for (Index f = 0; f < num_rigid_faces_; ++f) {
            Real corners[4][3];
            Real cen[3] = {0, 0, 0};
            for (int n = 0; n < nodes_per_face_; ++n) {
                Index node = rigid_conn_[f * nodes_per_face_ + n];
                for (int d = 0; d < 3; ++d) {
                    corners[n][d] = coords[node * 3 + d];
                    cen[d] += coords[node * 3 + d];
                }
            }
            for (int d = 0; d < 3; ++d) {
                cen[d] /= nodes_per_face_;
                rigid_centroids_[f * 3 + d] = cen[d];
            }

            Real normal[3];
            detail::quad_normal(corners[0], corners[1], corners[2], corners[3], normal);
            for (int d = 0; d < 3; ++d) {
                rigid_normals_[f * 3 + d] = normal[d];
            }
        }
    }

    /**
     * @brief Detect contact and apply forces to slave nodes only
     * @param coords Current coordinates
     * @param forces Force array (only slave node forces are modified)
     * @return Number of active contact pairs
     */
    int compute_forces(const Real* coords, Real* forces) const {
        int num_contacts = 0;

        for (Index sn : slave_nodes_) {
            const Real* slave_pos = &coords[sn * 3];

            for (Index f = 0; f < num_rigid_faces_; ++f) {
                const Real* normal = &rigid_normals_[f * 3];
                const Real* centroid = &rigid_centroids_[f * 3];

                // Signed distance from slave node to rigid face plane
                Real diff[3];
                detail::sub3(slave_pos, centroid, diff);
                Real signed_dist = detail::dot3(diff, normal);

                if (signed_dist < search_gap_ && signed_dist > -search_gap_) {
                    // Check if within face extent (simplified bounding sphere)
                    Real lateral[3];
                    for (int d = 0; d < 3; ++d) {
                        lateral[d] = diff[d] - signed_dist * normal[d];
                    }
                    Real lateral_dist = detail::norm3(lateral);

                    // Compute characteristic face size from first face diagonal
                    Real c0[3], c2[3];
                    for (int d = 0; d < 3; ++d) {
                        c0[d] = coords[rigid_conn_[f * nodes_per_face_ + 0] * 3 + d];
                        c2[d] = coords[rigid_conn_[f * nodes_per_face_ + 2] * 3 + d];
                    }
                    Real face_size = detail::distance3(c0, c2);

                    if (lateral_dist > face_size) continue;

                    // Penetration check
                    if (signed_dist < 0.0) {
                        // Penetrating: push slave node out along rigid normal
                        Real fn = penalty_stiffness_ * (-signed_dist);
                        for (int d = 0; d < 3; ++d) {
                            forces[sn * 3 + d] += fn * normal[d];
                        }
                        // No reaction force on rigid surface (it's rigid)
                        num_contacts++;
                    } else if (signed_dist < search_gap_) {
                        // Within gap: soft repulsion
                        Real penetration = search_gap_ - signed_dist;
                        Real fn = penalty_stiffness_ * penetration * 0.1;
                        for (int d = 0; d < 3; ++d) {
                            forces[sn * 3 + d] += fn * normal[d];
                        }
                        num_contacts++;
                    }
                }
            }
        }

        return num_contacts;
    }

    /**
     * @brief Check if rigid surface normals are precomputed
     */
    bool is_initialized() const { return !rigid_normals_.empty(); }

    Index num_rigid_faces() const { return num_rigid_faces_; }
    const std::vector<Index>& slave_nodes() const { return slave_nodes_; }

private:
    Real penalty_stiffness_;
    Real search_gap_;
    int nodes_per_face_;
    Index num_rigid_faces_;
    std::vector<Index> rigid_conn_;
    std::vector<Index> slave_nodes_;
    std::vector<Real> rigid_normals_;
    std::vector<Real> rigid_centroids_;
};

// ============================================================================
// 7. MultiSurfaceContactManager
// ============================================================================

/**
 * @brief Contact algorithm type for dispatching
 */
enum class ContactAlgorithm {
    NodeToSurface,     ///< Standard node-to-surface
    SegmentBased,      ///< Face-to-face segment contact
    SelfContact,       ///< Single-surface self-contact
    Symmetric,         ///< Bidirectional symmetric
    RigidDeformable,   ///< Rigid master + deformable slave
    EdgeToEdge         ///< Edge-edge crossing
};

/**
 * @brief A registered contact surface in the manager
 */
struct ManagedSurface {
    std::string name;
    std::vector<Index> connectivity;
    Index num_faces;
    int nodes_per_face;
    bool is_rigid;
    Real characteristic_size;  ///< Average face diagonal

    ManagedSurface()
        : num_faces(0), nodes_per_face(4), is_rigid(false), characteristic_size(0.0)
    {}
};

/**
 * @brief A contact interface (pair of surfaces)
 */
struct ContactInterface {
    Index surface_A;
    Index surface_B;
    ContactAlgorithm algorithm;
    Real penalty_stiffness;
    Real search_gap;

    ContactInterface()
        : surface_A(-1), surface_B(-1)
        , algorithm(ContactAlgorithm::SegmentBased)
        , penalty_stiffness(1.0e6)
        , search_gap(0.1)
    {}
};

/**
 * @brief Multi-surface contact manager
 *
 * Manages multiple contact surfaces and interfaces.
 * Can auto-detect surface pairs based on proximity.
 * Dispatches to appropriate contact algorithm.
 */
class MultiSurfaceContactManager {
public:
    MultiSurfaceContactManager() = default;

    /**
     * @brief Register a contact surface
     * @param name Surface name
     * @param connectivity Face connectivity
     * @param num_faces Number of faces
     * @param npf Nodes per face
     * @param is_rigid Whether surface is rigid
     * @return Surface index
     */
    Index register_surface(const std::string& name,
                           const std::vector<Index>& connectivity,
                           Index num_faces,
                           int npf = 4,
                           bool is_rigid = false) {
        ManagedSurface surf;
        surf.name = name;
        surf.connectivity = connectivity;
        surf.num_faces = num_faces;
        surf.nodes_per_face = npf;
        surf.is_rigid = is_rigid;
        surf.characteristic_size = 0.0;

        surfaces_.push_back(surf);
        return surfaces_.size() - 1;
    }

    /**
     * @brief Auto-detect contact pairs based on proximity
     * @param coords Nodal coordinates
     * @param proximity_threshold Maximum distance for contact pair detection
     */
    void auto_detect_pairs(const Real* coords, Real proximity_threshold = 0.5) {
        interfaces_.clear();

        // Compute centroids and characteristic sizes for each surface
        for (auto& surf : surfaces_) {
            if (surf.num_faces == 0) continue;

            Real total_diag = 0.0;
            for (Index f = 0; f < surf.num_faces; ++f) {
                Real c0[3], c2[3];
                for (int d = 0; d < 3; ++d) {
                    c0[d] = coords[surf.connectivity[f * surf.nodes_per_face + 0] * 3 + d];
                    c2[d] = coords[surf.connectivity[f * surf.nodes_per_face + 2] * 3 + d];
                }
                total_diag += detail::distance3(c0, c2);
            }
            surf.characteristic_size = total_diag / surf.num_faces;
        }

        // Check all surface pairs for proximity
        for (std::size_t i = 0; i < surfaces_.size(); ++i) {
            // Self-contact check
            // (For simplicity, don't auto-detect self-contact; user must request it)

            for (std::size_t j = i + 1; j < surfaces_.size(); ++j) {
                if (surfaces_in_proximity(i, j, coords, proximity_threshold)) {
                    ContactInterface iface;
                    iface.surface_A = static_cast<Index>(i);
                    iface.surface_B = static_cast<Index>(j);
                    iface.search_gap = proximity_threshold;

                    // Choose algorithm based on surface properties
                    if (surfaces_[i].is_rigid || surfaces_[j].is_rigid) {
                        iface.algorithm = ContactAlgorithm::RigidDeformable;
                        iface.penalty_stiffness = 1.0e8;
                    } else {
                        iface.algorithm = ContactAlgorithm::SegmentBased;
                        iface.penalty_stiffness = 1.0e6;
                    }

                    interfaces_.push_back(iface);
                }
            }
        }
    }

    /**
     * @brief Manually add a contact interface
     */
    void add_interface(Index surf_A, Index surf_B, ContactAlgorithm algo,
                       Real stiffness = 1.0e6, Real gap = 0.1) {
        ContactInterface iface;
        iface.surface_A = surf_A;
        iface.surface_B = surf_B;
        iface.algorithm = algo;
        iface.penalty_stiffness = stiffness;
        iface.search_gap = gap;
        interfaces_.push_back(iface);
    }

    /**
     * @brief Solve all contact interfaces
     * @param coords Nodal coordinates
     * @param forces Output force array (accumulated)
     * @return Total number of contact pairs detected
     */
    int solve_all(const Real* coords, Real* forces) const {
        int total_contacts = 0;

        for (const auto& iface : interfaces_) {
            const auto& surfA = surfaces_[iface.surface_A];
            const auto& surfB = surfaces_[iface.surface_B];

            switch (iface.algorithm) {
                case ContactAlgorithm::SegmentBased: {
                    SegmentBasedContact contact;
                    contact.set_penalty_stiffness(iface.penalty_stiffness);
                    contact.set_search_gap(iface.search_gap);
                    contact.set_slave_faces(surfA.connectivity, surfA.num_faces, surfA.nodes_per_face);
                    contact.set_master_faces(surfB.connectivity, surfB.num_faces, surfB.nodes_per_face);
                    auto pairs = contact.detect(coords);
                    for (const auto& p : pairs) {
                        contact.compute_segment_force(p, forces);
                    }
                    total_contacts += static_cast<int>(pairs.size());
                    break;
                }

                case ContactAlgorithm::Symmetric: {
                    SymmetricContact sym;
                    sym.set_penalty_stiffness(iface.penalty_stiffness);
                    sym.set_search_gap(iface.search_gap);
                    sym.set_surface_A(surfA.connectivity, surfA.num_faces, surfA.nodes_per_face);
                    sym.set_surface_B(surfB.connectivity, surfB.num_faces, surfB.nodes_per_face);
                    total_contacts += sym.compute_symmetric_forces(coords, forces);
                    break;
                }

                case ContactAlgorithm::RigidDeformable: {
                    RigidDeformableContact rdc;
                    rdc.set_penalty_stiffness(iface.penalty_stiffness);
                    rdc.set_search_gap(iface.search_gap);

                    // Determine which surface is rigid
                    const auto& rigid_surf = surfA.is_rigid ? surfA : surfB;
                    const auto& deform_surf = surfA.is_rigid ? surfB : surfA;

                    rdc.set_rigid_surface(rigid_surf.connectivity, rigid_surf.num_faces,
                                          rigid_surf.nodes_per_face);
                    rdc.precompute_rigid_geometry(coords);

                    // Collect slave nodes from deformable surface
                    std::unordered_set<Index> slave_set;
                    for (auto ni : deform_surf.connectivity) slave_set.insert(ni);
                    std::vector<Index> slave_nodes(slave_set.begin(), slave_set.end());
                    rdc.set_slave_nodes(slave_nodes);

                    total_contacts += rdc.compute_forces(coords, forces);
                    break;
                }

                default:
                    // Other algorithms can be added as needed
                    break;
            }
        }

        return total_contacts;
    }

    // Accessors
    std::size_t num_surfaces() const { return surfaces_.size(); }
    std::size_t num_interfaces() const { return interfaces_.size(); }
    const std::vector<ManagedSurface>& surfaces() const { return surfaces_; }
    const std::vector<ContactInterface>& interfaces() const { return interfaces_; }

private:
    std::vector<ManagedSurface> surfaces_;
    std::vector<ContactInterface> interfaces_;

    /**
     * @brief Check if two surfaces are within proximity threshold
     */
    bool surfaces_in_proximity(std::size_t i, std::size_t j,
                               const Real* coords, Real threshold) const {
        const auto& surfA = surfaces_[i];
        const auto& surfB = surfaces_[j];

        // Compare bounding boxes
        Real minA[3] = { 1e30,  1e30,  1e30};
        Real maxA[3] = {-1e30, -1e30, -1e30};
        Real minB[3] = { 1e30,  1e30,  1e30};
        Real maxB[3] = {-1e30, -1e30, -1e30};

        for (Index f = 0; f < surfA.num_faces; ++f) {
            for (int n = 0; n < surfA.nodes_per_face; ++n) {
                Index node = surfA.connectivity[f * surfA.nodes_per_face + n];
                for (int d = 0; d < 3; ++d) {
                    minA[d] = std::min(minA[d], coords[node * 3 + d]);
                    maxA[d] = std::max(maxA[d], coords[node * 3 + d]);
                }
            }
        }

        for (Index f = 0; f < surfB.num_faces; ++f) {
            for (int n = 0; n < surfB.nodes_per_face; ++n) {
                Index node = surfB.connectivity[f * surfB.nodes_per_face + n];
                for (int d = 0; d < 3; ++d) {
                    minB[d] = std::min(minB[d], coords[node * 3 + d]);
                    maxB[d] = std::max(maxB[d], coords[node * 3 + d]);
                }
            }
        }

        // Check AABB overlap with threshold margin
        for (int d = 0; d < 3; ++d) {
            if (minA[d] - threshold > maxB[d] || minB[d] - threshold > maxA[d]) {
                return false;
            }
        }

        return true;
    }
};

} // namespace fem
} // namespace nxs
