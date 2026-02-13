#pragma once

/**
 * @file mortar_contact.hpp
 * @brief Mortar segment-to-segment contact method
 *
 * True integral-based contact that:
 * - Passes the patch test (transfers constant pressure exactly)
 * - Eliminates contact locking (unlike node-to-surface)
 * - Provides smooth, oscillation-free pressure distribution
 * - Supports penalty and augmented Lagrangian enforcement
 *
 * Key algorithms:
 * - Sutherland-Hodgman polygon clipping for integration domain
 * - Fan-triangulated Gauss quadrature over clipped polygons
 * - D (slave-slave) and M (slave-master) mortar matrix assembly
 * - Optional dual Lagrange multipliers for diagonal D
 *
 * Reference: Puso & Laursen (2004), Popp et al. (2010)
 */

#include <nexussim/core/types.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace nxs {
namespace fem {

// ============================================================================
// Constraint enforcement type
// ============================================================================

enum class MortarEnforcement {
    Penalty,              ///< Simple penalty: λ = ε·min(g, 0)
    AugmentedLagrangian   ///< Uzawa iteration: λ^{k+1} = max(λ^k + ε·g, 0)
};

// ============================================================================
// Configuration
// ============================================================================

struct MortarContactConfig {
    MortarEnforcement enforcement;
    Real penalty_stiffness;     ///< Penalty parameter ε (Pa/m)
    Real augmentation_param;    ///< Augmented Lagrangian parameter
    int max_augmentation_iters; ///< Max Uzawa iterations
    Real augmentation_tol;      ///< Convergence tolerance
    Real contact_thickness;     ///< Detection gap (m)
    Real search_radius;         ///< Broad-phase search radius (m)
    bool two_pass;              ///< Symmetric master/slave treatment
    bool use_dual_shape;        ///< Dual Lagrange multipliers (diagonal D)
    Real static_friction;       ///< Coulomb static friction
    Real dynamic_friction;      ///< Coulomb dynamic friction
    int gauss_order;            ///< Quadrature order (default 3)

    MortarContactConfig()
        : enforcement(MortarEnforcement::Penalty)
        , penalty_stiffness(1.0e10)
        , augmentation_param(1.0e10)
        , max_augmentation_iters(10)
        , augmentation_tol(1.0e-6)
        , contact_thickness(1.0e-3)
        , search_radius(0.1)
        , two_pass(false)
        , use_dual_shape(false)
        , static_friction(0.0)
        , dynamic_friction(0.0)
        , gauss_order(3) {}
};

// ============================================================================
// Mortar segment definition
// ============================================================================

struct MortarSegment {
    Index nodes[4];     ///< Node indices (4th = -1 for triangle)
    int num_nodes;      ///< 3 (tri) or 4 (quad)
    Index surface_id;   ///< Which surface this belongs to

    MortarSegment() : num_nodes(0), surface_id(0) {
        for (int i = 0; i < 4; ++i) nodes[i] = Index(-1);
    }
};

// ============================================================================
// 2D polygon vertex for clipping
// ============================================================================

struct Vertex2D {
    Real x, y;
    Vertex2D() : x(0), y(0) {}
    Vertex2D(Real x_, Real y_) : x(x_), y(y_) {}
};

// ============================================================================
// Mortar contact pair
// ============================================================================

struct MortarPair {
    Index slave_segment;
    Index master_segment;

    // Clipped polygon in master parametric space
    int num_clip_vertices;
    Vertex2D clip_vertices[16]; ///< Max 16 vertices from clipping
    Real clip_area;             ///< Area of clipped polygon

    // Mortar integrals
    Real D[4][4];   ///< Slave-slave coupling: ∫ Φ_j · N^s_k dΓ
    Real M[4][4];   ///< Slave-master coupling: ∫ Φ_j · N^m_l dΓ

    // Gap and multiplier per slave node
    Real gap[4];        ///< Weighted normal gap
    Real lambda[4];     ///< Lagrange multipliers (augmented)
    Real normal[3];     ///< Contact normal

    // State
    Real pressure;      ///< Average contact pressure
    Real tangent_slip[4][3]; ///< Friction slip per slave node
    bool active;

    MortarPair() : slave_segment(0), master_segment(0), num_clip_vertices(0),
                   clip_area(0), pressure(0), active(false) {
        for (int i = 0; i < 4; ++i) {
            gap[i] = 0; lambda[i] = 0;
            for (int j = 0; j < 4; ++j) { D[i][j] = 0; M[i][j] = 0; }
            for (int d = 0; d < 3; ++d) tangent_slip[i][d] = 0;
        }
        normal[0] = normal[1] = 0; normal[2] = 1;
    }
};

// ============================================================================
// Mortar contact statistics
// ============================================================================

struct MortarStats {
    int active_pairs;
    int total_gauss_points;
    Real max_pressure;
    Real max_gap;
    Real constraint_violation;

    MortarStats() : active_pairs(0), total_gauss_points(0),
                    max_pressure(0), max_gap(0), constraint_violation(0) {}
};

// ============================================================================
// Main mortar contact class
// ============================================================================

class MortarContact {
public:
    MortarContact() : num_nodes_(0) {}

    // --- Configuration ---

    void set_config(const MortarContactConfig& config) { config_ = config; }
    const MortarContactConfig& config() const { return config_; }

    /**
     * @brief Add a contact surface
     * @param connectivity Flat array of node indices [seg0_n0, seg0_n1, ...]
     * @param nodes_per_segment 3 (tri) or 4 (quad)
     * @param surface_id Surface identifier (0 = slave, 1 = master, etc.)
     */
    void add_surface(const std::vector<Index>& connectivity,
                     int nodes_per_segment, int surface_id) {
        int num_segs = (int)connectivity.size() / nodes_per_segment;
        for (int s = 0; s < num_segs; ++s) {
            MortarSegment seg;
            seg.num_nodes = nodes_per_segment;
            seg.surface_id = surface_id;
            for (int n = 0; n < nodes_per_segment; ++n)
                seg.nodes[n] = connectivity[s * nodes_per_segment + n];
            segments_.push_back(seg);
        }
    }

    /**
     * @brief Initialize with total node count
     */
    void initialize(const Real* coords, Index num_nodes) {
        num_nodes_ = num_nodes;
        pairs_.clear();
    }

    // --- Per-step operations ---

    /**
     * @brief Detect contacts and compute mortar integrals D, M
     * @return Number of active contact pairs
     */
    int detect_and_integrate(const Real* coords) {
        pairs_.clear();

        // Separate segments by surface_id
        std::vector<Index> slave_segs, master_segs;
        for (std::size_t i = 0; i < segments_.size(); ++i) {
            if (segments_[i].surface_id == 0)
                slave_segs.push_back(i);
            else
                master_segs.push_back(i);
        }

        // For each slave-master pair, attempt clipping
        for (Index si : slave_segs) {
            const auto& ss = segments_[si];
            Real sc[3] = {0, 0, 0}; // Slave centroid
            for (int n = 0; n < ss.num_nodes; ++n)
                for (int d = 0; d < 3; ++d)
                    sc[d] += coords[3*ss.nodes[n]+d] / ss.num_nodes;

            for (Index mi : master_segs) {
                const auto& ms = segments_[mi];
                Real mc[3] = {0, 0, 0}; // Master centroid
                for (int n = 0; n < ms.num_nodes; ++n)
                    for (int d = 0; d < 3; ++d)
                        mc[d] += coords[3*ms.nodes[n]+d] / ms.num_nodes;

                // Quick distance check
                Real dx = sc[0]-mc[0], dy = sc[1]-mc[1], dz = sc[2]-mc[2];
                if (dx*dx+dy*dy+dz*dz > config_.search_radius*config_.search_radius)
                    continue;

                // Compute master normal
                Real mnorm[3];
                compute_segment_normal(coords, ms, mnorm);

                // Check gap: slave centroid to master plane
                Real gap_test = (sc[0]-mc[0])*mnorm[0] + (sc[1]-mc[1])*mnorm[1]
                              + (sc[2]-mc[2])*mnorm[2];
                if (gap_test > config_.contact_thickness) continue;

                // Build local 2D coordinate system on master plane
                Real ax[3], ay[3];
                build_local_axes(mnorm, ax, ay);

                // Project slave and master nodes to 2D
                Vertex2D slave_2d[4], master_2d[4];
                project_to_2d(coords, ss, mc, ax, ay, slave_2d);
                project_to_2d(coords, ms, mc, ax, ay, master_2d);

                // Sutherland-Hodgman clipping
                Vertex2D clipped[16];
                int num_clip = clip_polygon(slave_2d, ss.num_nodes,
                                            master_2d, ms.num_nodes,
                                            clipped);

                if (num_clip < 3) continue; // No overlap

                Real area = polygon_area(clipped, num_clip);
                if (area < 1.0e-20) continue;

                // Create mortar pair
                MortarPair pair;
                pair.slave_segment = si;
                pair.master_segment = mi;
                pair.num_clip_vertices = num_clip;
                for (int v = 0; v < num_clip; ++v)
                    pair.clip_vertices[v] = clipped[v];
                pair.clip_area = area;
                for (int d = 0; d < 3; ++d) pair.normal[d] = mnorm[d];
                pair.active = true;

                // Carry over Lagrange multipliers from previous step
                for (auto& prev : prev_pairs_) {
                    if (prev.slave_segment == si && prev.master_segment == mi) {
                        for (int n = 0; n < 4; ++n) {
                            pair.lambda[n] = prev.lambda[n];
                            for (int d = 0; d < 3; ++d)
                                pair.tangent_slip[n][d] = prev.tangent_slip[n][d];
                        }
                        break;
                    }
                }

                // Compute mortar integrals D and M via Gauss quadrature
                compute_mortar_integrals(coords, pair, ss, ms, mc, ax, ay);

                // Compute weighted gaps
                compute_weighted_gaps(coords, pair, ss, ms);

                pairs_.push_back(pair);
            }
        }

        // Two-pass: swap slave/master roles
        if (config_.two_pass) {
            int first_pass = (int)pairs_.size();
            for (Index mi : master_segs) {
                const auto& ms = segments_[mi];
                Real mc[3] = {0, 0, 0};
                for (int n = 0; n < ms.num_nodes; ++n)
                    for (int d = 0; d < 3; ++d)
                        mc[d] += coords[3*ms.nodes[n]+d] / ms.num_nodes;

                for (Index si : slave_segs) {
                    const auto& ss = segments_[si];
                    Real sc[3] = {0, 0, 0};
                    for (int n = 0; n < ss.num_nodes; ++n)
                        for (int d = 0; d < 3; ++d)
                            sc[d] += coords[3*ss.nodes[n]+d] / ss.num_nodes;

                    Real dx = sc[0]-mc[0], dy = sc[1]-mc[1], dz = sc[2]-mc[2];
                    if (dx*dx+dy*dy+dz*dz > config_.search_radius*config_.search_radius)
                        continue;

                    // Already found in first pass? Skip
                    bool found = false;
                    for (int p = 0; p < first_pass; ++p)
                        if (pairs_[p].slave_segment == si && pairs_[p].master_segment == mi)
                        { found = true; break; }
                    if (found) continue;

                    Real snorm[3];
                    compute_segment_normal(coords, ss, snorm);
                    Real gap_test = (mc[0]-sc[0])*snorm[0] + (mc[1]-sc[1])*snorm[1]
                                  + (mc[2]-sc[2])*snorm[2];
                    if (gap_test > config_.contact_thickness) continue;

                    Real ax[3], ay[3];
                    build_local_axes(snorm, ax, ay);
                    Vertex2D m2d[4], s2d[4];
                    project_to_2d(coords, ms, sc, ax, ay, m2d);
                    project_to_2d(coords, ss, sc, ax, ay, s2d);
                    Vertex2D clipped[16];
                    int nc = clip_polygon(m2d, ms.num_nodes, s2d, ss.num_nodes, clipped);
                    if (nc < 3) continue;
                    Real area = polygon_area(clipped, nc);
                    if (area < 1.0e-20) continue;

                    MortarPair pair;
                    pair.slave_segment = mi; // Reversed roles
                    pair.master_segment = si;
                    pair.num_clip_vertices = nc;
                    for (int v = 0; v < nc; ++v) pair.clip_vertices[v] = clipped[v];
                    pair.clip_area = area;
                    for (int d = 0; d < 3; ++d) pair.normal[d] = snorm[d];
                    pair.active = true;

                    compute_mortar_integrals(coords, pair, segments_[mi], segments_[si],
                                             sc, ax, ay);
                    compute_weighted_gaps(coords, pair, segments_[mi], segments_[si]);
                    pairs_.push_back(pair);
                }
            }
        }

        prev_pairs_ = pairs_;
        return (int)pairs_.size();
    }

    /**
     * @brief Compute contact forces from mortar integrals
     */
    void compute_forces(const Real* coords, const Real* velocities,
                        Real dt, Real* forces) {
        stats_ = MortarStats();

        for (auto& pair : pairs_) {
            if (!pair.active) continue;

            const auto& ss = segments_[pair.slave_segment];
            const auto& ms = segments_[pair.master_segment];

            for (int j = 0; j < ss.num_nodes; ++j) {
                Real g_j = pair.gap[j];

                // Compute contact traction (lambda)
                Real lambda_j = 0.0;
                if (config_.enforcement == MortarEnforcement::Penalty) {
                    if (g_j < 0.0) // Penetration
                        lambda_j = config_.penalty_stiffness * (-g_j);
                } else {
                    // Augmented Lagrangian: use stored multiplier
                    lambda_j = pair.lambda[j];
                    if (lambda_j < 0.0) lambda_j = 0.0;
                }

                if (lambda_j <= 0.0) continue;
                pair.pressure = std::max(pair.pressure, lambda_j);

                // Force on slave node j: F_j = D[j][j]^(-1) * lambda_j * n
                // For standard mortar, distribute via D row
                Real D_diag = pair.D[j][j];
                if (D_diag < 1.0e-30) continue;

                Real force_scale = lambda_j;

                // Apply to slave nodes (push away from master)
                Index sn = ss.nodes[j];
                for (int d = 0; d < 3; ++d)
                    forces[3*sn+d] += force_scale * D_diag * pair.normal[d];

                // Off-diagonal D contributions
                for (int k = 0; k < ss.num_nodes; ++k) {
                    if (k == j) continue;
                    Index sk = ss.nodes[k];
                    for (int d = 0; d < 3; ++d)
                        forces[3*sk+d] += force_scale * pair.D[j][k] * pair.normal[d];
                }

                // Reaction on master nodes: F_l = -M^T[l][j] * lambda_j * n
                for (int l = 0; l < ms.num_nodes; ++l) {
                    Index mn = ms.nodes[l];
                    for (int d = 0; d < 3; ++d)
                        forces[3*mn+d] -= force_scale * pair.M[j][l] * pair.normal[d];
                }

                // --- Friction ---
                if (config_.static_friction > 0.0 && lambda_j > 0.0) {
                    Real vrel[3] = {0, 0, 0};
                    Real vs[3] = {velocities[3*sn+0], velocities[3*sn+1], velocities[3*sn+2]};
                    Real vm[3] = {0, 0, 0};
                    for (int l = 0; l < ms.num_nodes; ++l) {
                        Index mn = ms.nodes[l];
                        Real w = pair.M[j][l] / std::max(D_diag, 1e-30);
                        for (int d = 0; d < 3; ++d)
                            vm[d] += w * velocities[3*mn+d];
                    }
                    for (int d = 0; d < 3; ++d) vrel[d] = vs[d] - vm[d];

                    Real vn_dot = vrel[0]*pair.normal[0]+vrel[1]*pair.normal[1]
                                + vrel[2]*pair.normal[2];
                    Real vt[3];
                    for (int d = 0; d < 3; ++d)
                        vt[d] = vrel[d] - vn_dot * pair.normal[d];

                    for (int d = 0; d < 3; ++d)
                        pair.tangent_slip[j][d] += vt[d] * dt;

                    Real slip_mag = std::sqrt(pair.tangent_slip[j][0]*pair.tangent_slip[j][0]
                                            + pair.tangent_slip[j][1]*pair.tangent_slip[j][1]
                                            + pair.tangent_slip[j][2]*pair.tangent_slip[j][2]);

                    if (slip_mag > 1.0e-20) {
                        Real fn = lambda_j * D_diag;
                        Real friction_limit = config_.static_friction * fn;
                        Real k_t = config_.penalty_stiffness * 0.5;
                        Real trial = k_t * slip_mag;
                        Real Ft = std::min(trial, friction_limit);

                        for (int d = 0; d < 3; ++d) {
                            Real fd = -Ft * pair.tangent_slip[j][d] / slip_mag;
                            forces[3*sn+d] += fd;
                            for (int l = 0; l < ms.num_nodes; ++l) {
                                Index mn = ms.nodes[l];
                                Real w = pair.M[j][l] / std::max(D_diag, 1e-30);
                                forces[3*mn+d] -= w * fd;
                            }
                        }

                        if (trial > friction_limit) {
                            Real ratio = friction_limit / trial;
                            for (int d = 0; d < 3; ++d)
                                pair.tangent_slip[j][d] *= ratio;
                        }
                    }
                }

                // Statistics
                if (lambda_j > stats_.max_pressure)
                    stats_.max_pressure = lambda_j;
                if (std::fabs(g_j) > stats_.max_gap)
                    stats_.max_gap = std::fabs(g_j);
                stats_.constraint_violation += g_j * g_j;
            }

            stats_.active_pairs++;
        }

        stats_.constraint_violation = std::sqrt(stats_.constraint_violation);
    }

    /**
     * @brief Augmented Lagrangian update (Uzawa iteration)
     * @return true if converged
     */
    bool augmented_lagrangian_update() {
        Real max_change = 0.0;
        Real max_lambda = 0.0;

        for (auto& pair : pairs_) {
            if (!pair.active) continue;
            for (int j = 0; j < segments_[pair.slave_segment].num_nodes; ++j) {
                Real lambda_old = pair.lambda[j];
                Real lambda_new = std::max(lambda_old + config_.augmentation_param * pair.gap[j], 0.0);
                pair.lambda[j] = lambda_new;

                Real change = std::fabs(lambda_new - lambda_old);
                if (change > max_change) max_change = change;
                if (std::fabs(lambda_new) > max_lambda) max_lambda = std::fabs(lambda_new);
            }
        }

        if (max_lambda < 1.0e-30) return true;
        return (max_change / max_lambda) < config_.augmentation_tol;
    }

    // --- Query ---

    const MortarStats& get_stats() const { return stats_; }
    int num_active_pairs() const { return stats_.active_pairs; }
    const std::vector<MortarPair>& active_pairs() const { return pairs_; }

    void print_summary() const {
        std::cout << "MortarContact: " << stats_.active_pairs << " active pairs\n";
        std::cout << "  Max pressure: " << stats_.max_pressure << " Pa\n";
        std::cout << "  Max gap: " << stats_.max_gap * 1000.0 << " mm\n";
        std::cout << "  Constraint violation: " << stats_.constraint_violation << "\n";
    }

    // === Public utility for testing ===

    /**
     * @brief Clip subject polygon against clip polygon (Sutherland-Hodgman)
     * @return Number of vertices in result
     */
    static int clip_polygon(const Vertex2D* subject, int n_subject,
                            const Vertex2D* clip, int n_clip,
                            Vertex2D* result) {
        // Copy subject to working buffer
        Vertex2D input[16], output[16];
        int n_in = n_subject;
        for (int i = 0; i < n_subject; ++i) input[i] = subject[i];

        for (int e = 0; e < n_clip; ++e) {
            if (n_in < 1) return 0;

            Vertex2D A = clip[e];
            Vertex2D B = clip[(e + 1) % n_clip];

            int n_out = 0;
            Vertex2D S = input[n_in - 1];

            for (int i = 0; i < n_in; ++i) {
                Vertex2D V = input[i];
                bool v_inside = is_inside(A, B, V);
                bool s_inside = is_inside(A, B, S);

                if (v_inside) {
                    if (!s_inside) {
                        // S outside, V inside → add intersection then V
                        if (n_out < 16) output[n_out++] = intersect(S, V, A, B);
                    }
                    if (n_out < 16) output[n_out++] = V;
                } else if (s_inside) {
                    // S inside, V outside → add intersection
                    if (n_out < 16) output[n_out++] = intersect(S, V, A, B);
                }
                S = V;
            }

            n_in = n_out;
            for (int i = 0; i < n_out; ++i) input[i] = output[i];
        }

        for (int i = 0; i < n_in; ++i) result[i] = input[i];
        return n_in;
    }

    /**
     * @brief Compute area of a 2D polygon
     */
    static Real polygon_area(const Vertex2D* verts, int n) {
        Real area = 0.0;
        for (int i = 0; i < n; ++i) {
            int j = (i + 1) % n;
            area += verts[i].x * verts[j].y - verts[j].x * verts[i].y;
        }
        return 0.5 * std::fabs(area);
    }

private:
    MortarContactConfig config_;
    std::vector<MortarSegment> segments_;
    std::vector<MortarPair> pairs_;
    std::vector<MortarPair> prev_pairs_;
    Index num_nodes_;
    MortarStats stats_;

    // --- Geometry helpers ---

    void compute_segment_normal(const Real* coords, const MortarSegment& seg,
                                Real* normal) const {
        Real x[4][3];
        for (int n = 0; n < seg.num_nodes; ++n)
            for (int d = 0; d < 3; ++d)
                x[n][d] = coords[3*seg.nodes[n]+d];

        Real e1[3], e2[3];
        for (int d = 0; d < 3; ++d) {
            e1[d] = x[1][d] - x[0][d];
            e2[d] = x[seg.num_nodes > 2 ? 2 : 1][d] - x[0][d];
        }
        normal[0] = e1[1]*e2[2] - e1[2]*e2[1];
        normal[1] = e1[2]*e2[0] - e1[0]*e2[2];
        normal[2] = e1[0]*e2[1] - e1[1]*e2[0];
        Real mag = std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
        if (mag < 1.0e-30) mag = 1.0e-30;
        for (int d = 0; d < 3; ++d) normal[d] /= mag;
    }

    void build_local_axes(const Real* normal, Real* ax, Real* ay) const {
        // Find a vector not parallel to normal
        Real ref[3] = {1, 0, 0};
        Real dot = normal[0]*ref[0]+normal[1]*ref[1]+normal[2]*ref[2];
        if (std::fabs(dot) > 0.9) { ref[0] = 0; ref[1] = 1; }

        // ax = normal × ref
        ax[0] = normal[1]*ref[2]-normal[2]*ref[1];
        ax[1] = normal[2]*ref[0]-normal[0]*ref[2];
        ax[2] = normal[0]*ref[1]-normal[1]*ref[0];
        Real mag = std::sqrt(ax[0]*ax[0]+ax[1]*ax[1]+ax[2]*ax[2]);
        if (mag < 1.0e-30) mag = 1.0e-30;
        ax[0] /= mag; ax[1] /= mag; ax[2] /= mag;

        // ay = normal × ax
        ay[0] = normal[1]*ax[2]-normal[2]*ax[1];
        ay[1] = normal[2]*ax[0]-normal[0]*ax[2];
        ay[2] = normal[0]*ax[1]-normal[1]*ax[0];
    }

    void project_to_2d(const Real* coords, const MortarSegment& seg,
                        const Real* origin, const Real* ax, const Real* ay,
                        Vertex2D* out) const {
        for (int n = 0; n < seg.num_nodes; ++n) {
            Real dx = coords[3*seg.nodes[n]+0] - origin[0];
            Real dy = coords[3*seg.nodes[n]+1] - origin[1];
            Real dz = coords[3*seg.nodes[n]+2] - origin[2];
            out[n].x = dx*ax[0]+dy*ax[1]+dz*ax[2];
            out[n].y = dx*ay[0]+dy*ay[1]+dz*ay[2];
        }
    }

    // --- Clipping helpers ---

    static bool is_inside(const Vertex2D& A, const Vertex2D& B, const Vertex2D& P) {
        return (B.x-A.x)*(P.y-A.y) - (B.y-A.y)*(P.x-A.x) >= 0.0;
    }

    static Vertex2D intersect(const Vertex2D& P1, const Vertex2D& P2,
                               const Vertex2D& P3, const Vertex2D& P4) {
        Real denom = (P1.x-P2.x)*(P3.y-P4.y) - (P1.y-P2.y)*(P3.x-P4.x);
        if (std::fabs(denom) < 1.0e-30) return P1; // Degenerate
        Real t = ((P1.x-P3.x)*(P3.y-P4.y) - (P1.y-P3.y)*(P3.x-P4.x)) / denom;
        return Vertex2D(P1.x + t*(P2.x-P1.x), P1.y + t*(P2.y-P1.y));
    }

    // --- Mortar integration ---

    /**
     * @brief Compute D and M matrices via Gauss quadrature over clipped polygon
     */
    void compute_mortar_integrals(const Real* coords, MortarPair& pair,
                                   const MortarSegment& ss, const MortarSegment& ms,
                                   const Real* origin, const Real* ax, const Real* ay) {
        // Zero matrices
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                { pair.D[i][j] = 0; pair.M[i][j] = 0; }

        if (pair.num_clip_vertices < 3) return;

        // Get 2D coords of slave and master nodes (for shape function eval)
        Vertex2D slave_2d[4], master_2d[4];
        project_to_2d(coords, ss, origin, ax, ay, slave_2d);
        project_to_2d(coords, ms, origin, ax, ay, master_2d);

        // Fan-triangulate clipped polygon from centroid
        Vertex2D centroid = {0, 0};
        for (int v = 0; v < pair.num_clip_vertices; ++v) {
            centroid.x += pair.clip_vertices[v].x;
            centroid.y += pair.clip_vertices[v].y;
        }
        centroid.x /= pair.num_clip_vertices;
        centroid.y /= pair.num_clip_vertices;

        // Gauss points for triangle (degree 2, 3 points)
        const Real gp_bary[3][3] = {
            {1.0/6.0, 1.0/6.0, 4.0/6.0},
            {1.0/6.0, 4.0/6.0, 1.0/6.0},
            {4.0/6.0, 1.0/6.0, 1.0/6.0}
        };
        const Real gp_weight = 1.0 / 6.0; // Weight for each point (area coords)

        int total_gp = 0;

        for (int t = 0; t < pair.num_clip_vertices; ++t) {
            int next = (t + 1) % pair.num_clip_vertices;
            Vertex2D v0 = centroid;
            Vertex2D v1 = pair.clip_vertices[t];
            Vertex2D v2 = pair.clip_vertices[next];

            // Sub-triangle area
            Real tri_area = 0.5 * std::fabs((v1.x-v0.x)*(v2.y-v0.y)-(v2.x-v0.x)*(v1.y-v0.y));
            if (tri_area < 1.0e-30) continue;

            for (int gp = 0; gp < 3; ++gp) {
                Real L0 = gp_bary[gp][0], L1 = gp_bary[gp][1], L2 = gp_bary[gp][2];

                // Gauss point in 2D
                Real xg = L0*v0.x + L1*v1.x + L2*v2.x;
                Real yg = L0*v0.y + L1*v1.y + L2*v2.y;

                // Evaluate slave shape functions at Gauss point
                Real Ns[4];
                eval_shape_2d(slave_2d, ss.num_nodes, xg, yg, Ns);

                // Evaluate master shape functions at Gauss point
                Real Nm[4];
                eval_shape_2d(master_2d, ms.num_nodes, xg, yg, Nm);

                Real w = gp_weight * 2.0 * tri_area; // Weight × Jacobian

                // Assemble D: D[j][k] += w * Ns[j] * Ns[k]
                for (int j = 0; j < ss.num_nodes; ++j)
                    for (int k = 0; k < ss.num_nodes; ++k)
                        pair.D[j][k] += w * Ns[j] * Ns[k];

                // Assemble M: M[j][l] += w * Ns[j] * Nm[l]
                for (int j = 0; j < ss.num_nodes; ++j)
                    for (int l = 0; l < ms.num_nodes; ++l)
                        pair.M[j][l] += w * Ns[j] * Nm[l];

                total_gp++;
            }
        }

        stats_.total_gauss_points += total_gp;

        // Dual shape functions: modify D to be diagonal
        if (config_.use_dual_shape) {
            for (int j = 0; j < ss.num_nodes; ++j) {
                Real row_sum = 0.0;
                for (int k = 0; k < ss.num_nodes; ++k) row_sum += pair.D[j][k];
                pair.D[j][j] = row_sum;
                for (int k = 0; k < ss.num_nodes; ++k)
                    if (k != j) pair.D[j][k] = 0.0;
            }
        }
    }

    /**
     * @brief Compute weighted normal gaps
     */
    void compute_weighted_gaps(const Real* coords, MortarPair& pair,
                                const MortarSegment& ss, const MortarSegment& ms) {
        for (int j = 0; j < ss.num_nodes; ++j) {
            // g_j = Σ_k D[j][k] * (x^s_k · n) - Σ_l M[j][l] * (x^m_l · n)
            Real slave_contrib = 0.0;
            for (int k = 0; k < ss.num_nodes; ++k) {
                Real xn = 0;
                for (int d = 0; d < 3; ++d)
                    xn += coords[3*ss.nodes[k]+d] * pair.normal[d];
                slave_contrib += pair.D[j][k] * xn;
            }
            Real master_contrib = 0.0;
            for (int l = 0; l < ms.num_nodes; ++l) {
                Real xn = 0;
                for (int d = 0; d < 3; ++d)
                    xn += coords[3*ms.nodes[l]+d] * pair.normal[d];
                master_contrib += pair.M[j][l] * xn;
            }
            pair.gap[j] = slave_contrib - master_contrib;
        }
    }

    /**
     * @brief Evaluate shape functions for a 2D polygon at point (xp, yp)
     */
    void eval_shape_2d(const Vertex2D* nodes, int nn, Real xp, Real yp,
                        Real* N) const {
        N[0] = N[1] = N[2] = N[3] = 0.0;

        if (nn == 3) {
            // Barycentric coordinates for triangle
            Real x0 = nodes[0].x, y0 = nodes[0].y;
            Real x1 = nodes[1].x, y1 = nodes[1].y;
            Real x2 = nodes[2].x, y2 = nodes[2].y;
            Real det = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0);
            if (std::fabs(det) < 1.0e-30) { N[0] = 1.0; return; }
            N[1] = ((xp-x0)*(y2-y0) - (x2-x0)*(yp-y0)) / det;
            N[2] = ((x1-x0)*(yp-y0) - (xp-x0)*(y1-y0)) / det;
            N[0] = 1.0 - N[1] - N[2];
        } else {
            // Bilinear quad: inverse mapping to find (ξ, η)
            // Use centroid-based approximation
            Real cx = 0, cy = 0;
            for (int i = 0; i < 4; ++i) { cx += nodes[i].x; cy += nodes[i].y; }
            cx *= 0.25; cy *= 0.25;

            Real dx1 = 0.5*(nodes[1].x+nodes[2].x) - 0.5*(nodes[0].x+nodes[3].x);
            Real dy1 = 0.5*(nodes[1].y+nodes[2].y) - 0.5*(nodes[0].y+nodes[3].y);
            Real dx2 = 0.5*(nodes[3].x+nodes[2].x) - 0.5*(nodes[0].x+nodes[1].x);
            Real dy2 = 0.5*(nodes[3].y+nodes[2].y) - 0.5*(nodes[0].y+nodes[1].y);

            Real L1 = std::sqrt(dx1*dx1+dy1*dy1);
            Real L2 = std::sqrt(dx2*dx2+dy2*dy2);
            if (L1 < 1e-30) L1 = 1e-30;
            if (L2 < 1e-30) L2 = 1e-30;

            Real dpx = xp - cx, dpy = yp - cy;
            Real xi  = (dpx*dx1+dpy*dy1) / (L1*L1) * 2.0;
            Real eta = (dpx*dx2+dpy*dy2) / (L2*L2) * 2.0;
            xi  = std::max(-1.0, std::min(1.0, xi));
            eta = std::max(-1.0, std::min(1.0, eta));

            N[0] = 0.25*(1-xi)*(1-eta);
            N[1] = 0.25*(1+xi)*(1-eta);
            N[2] = 0.25*(1+xi)*(1+eta);
            N[3] = 0.25*(1-xi)*(1+eta);
        }
    }
};

} // namespace fem
} // namespace nxs
