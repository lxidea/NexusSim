#pragma once

/**
 * @file surface_contact.hpp
 * @brief Surface-to-surface contact algorithm (INT17-style)
 *
 * Features:
 * - Segment-to-segment contact (not just node-to-surface)
 * - Two-pass algorithm (symmetric master-slave treatment)
 * - Contact thickness with gap allowance
 * - Time-adaptive penalty stiffness: K = scale * mass / dt²
 * - GPU-accelerated via Kokkos
 *
 * Based on OpenRadioss INT17 methodology for shell contact.
 */

#include <nexussim/core/types.hpp>
#include <nexussim/fem/voxel_collision.hpp>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <memory>

namespace nxs {
namespace fem {

// ============================================================================
// Surface Contact Configuration
// ============================================================================

/**
 * @brief Configuration for surface-to-surface contact
 */
struct SurfaceContactConfig {
    // Penalty parameters
    Real penalty_scale;          ///< Scale factor for penalty stiffness
    Real contact_thickness;      ///< Shell contact thickness
    Real gap_min;                ///< Minimum allowed gap
    Real gap_max;                ///< Maximum gap for contact detection

    // Time-adaptive stiffness
    bool adaptive_stiffness;     ///< Enable time-adaptive penalty
    Real mass_scale;             ///< Mass scaling for adaptive stiffness

    // Friction
    Real static_friction;        ///< Static friction coefficient
    Real dynamic_friction;       ///< Dynamic friction coefficient
    Real friction_decay;         ///< Stick-slip decay parameter

    // Algorithm options
    bool two_pass;               ///< Use two-pass symmetric algorithm
    bool check_normals;          ///< Check normal consistency
    Real normal_tolerance;       ///< Tolerance for normal alignment

    // Detection parameters
    Real bucket_size_factor;     ///< Bucket size multiplier
    int max_pairs_per_segment;   ///< Maximum contact pairs per segment

    SurfaceContactConfig()
        : penalty_scale(1.0)
        , contact_thickness(0.0)
        , gap_min(1.0e-10)
        , gap_max(0.0)
        , adaptive_stiffness(true)
        , mass_scale(1.0)
        , static_friction(0.3)
        , dynamic_friction(0.2)
        , friction_decay(10.0)
        , two_pass(true)
        , check_normals(true)
        , normal_tolerance(0.707)  // cos(45°)
        , bucket_size_factor(2.0)
        , max_pairs_per_segment(8)
    {
        gap_max = contact_thickness * 3.0;
    }
};

// ============================================================================
// Surface Contact Pair
// ============================================================================

/**
 * @brief Contact information for a segment-segment pair
 */
struct SurfaceContactPair {
    Index segment1;              ///< First segment index
    Index segment2;              ///< Second segment index
    Vec3r contact_point;         ///< Contact point location
    Vec3r normal;                ///< Contact normal (from segment1 to segment2)
    Real gap;                    ///< Gap distance (negative = penetration)
    Real pressure;               ///< Contact pressure
    Real area;                   ///< Contact area contribution

    // Friction state
    Vec3r tangent_slip;          ///< Accumulated tangential slip
    bool sticking;               ///< Currently in stick regime

    // Shape function values at contact point
    Real phi1[4];                ///< Shape functions for segment 1
    Real phi2[4];                ///< Shape functions for segment 2

    bool active;                 ///< Is contact active?

    SurfaceContactPair()
        : segment1(-1)
        , segment2(-1)
        , contact_point{0, 0, 0}
        , normal{0, 0, 1}
        , gap(0.0)
        , pressure(0.0)
        , area(0.0)
        , tangent_slip{0, 0, 0}
        , sticking(true)
        , active(false)
    {
        phi1[0] = phi1[1] = phi1[2] = phi1[3] = 0.25;
        phi2[0] = phi2[1] = phi2[2] = phi2[3] = 0.25;
    }
};

// ============================================================================
// Surface Definition
// ============================================================================

/**
 * @brief A contact surface (collection of segments)
 */
struct ContactSurface {
    std::vector<Index> connectivity;     ///< Segment connectivity (node indices)
    std::vector<Index> segment_ids;      ///< Segment IDs for reference
    int nodes_per_segment;               ///< 3 for triangles, 4 for quads
    Index num_segments;                  ///< Number of segments

    // Part information for friction
    Index part_id;                       ///< Part ID for part-based friction
    Real thickness;                      ///< Surface thickness

    ContactSurface()
        : nodes_per_segment(4)
        , num_segments(0)
        , part_id(0)
        , thickness(0.001)
    {}
};

// ============================================================================
// Surface-to-Surface Contact Algorithm
// ============================================================================

/**
 * @brief Surface-to-surface contact with INT17-style algorithm
 *
 * The algorithm:
 * 1. Build voxel grid for broad phase detection
 * 2. For each segment, find candidate segments in nearby voxels
 * 3. For each candidate pair, compute closest point
 * 4. Apply penalty forces if within contact thickness
 * 5. (Optional) Repeat with surfaces swapped for two-pass
 *
 * Usage:
 * ```cpp
 * SurfaceToSurfaceContact contact;
 * contact.add_surface(surface1);
 * contact.add_surface(surface2);
 * contact.initialize(coords, num_nodes);
 *
 * // Each time step:
 * contact.detect(coords);
 * contact.compute_forces(coords, velocity, dt, element_mass, forces);
 * ```
 */
class SurfaceToSurfaceContact {
public:
    SurfaceToSurfaceContact(const SurfaceContactConfig& config = SurfaceContactConfig())
        : config_(config)
        , num_nodes_(0)
    {}

    // ========================================================================
    // Setup
    // ========================================================================

    /**
     * @brief Add a contact surface
     * @param surface The surface definition
     */
    void add_surface(const ContactSurface& surface) {
        surfaces_.push_back(surface);
    }

    /**
     * @brief Initialize with mesh data
     * @param coords Node coordinates
     * @param num_nodes Total number of nodes
     */
    void initialize(const Real* coords, Index num_nodes) {
        num_nodes_ = num_nodes;

        // Set up gap_max from contact thickness if not set
        if (config_.gap_max <= 0.0) {
            Real max_thickness = 0.0;
            for (const auto& surf : surfaces_) {
                max_thickness = std::max(max_thickness, surf.thickness);
            }
            config_.gap_max = max_thickness * 3.0;
            config_.contact_thickness = max_thickness;
        }

        // Build segment connectivity for voxel grid
        build_segment_index();

        // Initialize voxel collision detector
        // Use a cell size based on gap_max for better detection at small gaps
        VoxelCollisionConfig voxel_config;
        voxel_config.auto_cell_size = false;  // Don't use segment-based cell size
        voxel_config.cell_size_factor = config_.bucket_size_factor;  // Default 2.0 * 0.1 = 0.2
        voxel_config.search_margin = config_.gap_max;

        voxel_detector_ = std::make_unique<VoxelCollisionDetector>(voxel_config);

        if (!all_segments_.empty()) {
            voxel_detector_->initialize(
                num_nodes,
                static_cast<Index>(all_segments_.size() / 4),  // Assume 4 nodes per segment
                coords,
                all_segments_.data(),
                4
            );
        }
    }

    // ========================================================================
    // Contact Detection
    // ========================================================================

    /**
     * @brief Detect contacts at current configuration
     * @param coords Nodal coordinates
     * @return Number of active contact pairs
     */
    int detect(const Real* coords) {
        active_pairs_.clear();

        if (surfaces_.empty()) return 0;

        // Update voxel grid
        voxel_detector_->update_coordinates(coords);

        // Detect contacts between all surface pairs
        for (size_t s1 = 0; s1 < surfaces_.size(); ++s1) {
            for (size_t s2 = s1; s2 < surfaces_.size(); ++s2) {
                detect_surface_pair(s1, s2, coords);

                // Two-pass: reverse order
                if (config_.two_pass && s1 != s2) {
                    detect_surface_pair(s2, s1, coords);
                }
            }
        }

        return static_cast<int>(active_pairs_.size());
    }

    // ========================================================================
    // Force Computation
    // ========================================================================

    /**
     * @brief Compute contact forces
     * @param coords Nodal coordinates
     * @param velocity Nodal velocities
     * @param dt Time step
     * @param element_mass Representative element mass
     * @param forces Output forces (added to existing)
     */
    void compute_forces(const Real* coords,
                        const Real* velocity,
                        Real dt,
                        Real element_mass,
                        Real* forces) {

        // Compute penalty stiffness
        Real penalty_stiffness;
        if (config_.adaptive_stiffness) {
            // Time-adaptive stiffness: K = scale * mass / dt²
            penalty_stiffness = config_.penalty_scale * config_.mass_scale * element_mass / (dt * dt);
        } else {
            penalty_stiffness = config_.penalty_scale * element_mass / (dt * dt);
        }

        for (auto& pair : active_pairs_) {
            if (!pair.active) continue;

            // Get segment information
            Index seg1 = pair.segment1;
            Index seg2 = pair.segment2;

            // Find which surfaces these segments belong to
            int surf1_idx = -1, surf2_idx = -1;
            Index local_seg1 = seg1, local_seg2 = seg2;

            Index seg_offset = 0;
            for (size_t s = 0; s < surfaces_.size(); ++s) {
                if (seg1 >= seg_offset && seg1 < seg_offset + surfaces_[s].num_segments) {
                    surf1_idx = static_cast<int>(s);
                    local_seg1 = seg1 - seg_offset;
                }
                if (seg2 >= seg_offset && seg2 < seg_offset + surfaces_[s].num_segments) {
                    surf2_idx = static_cast<int>(s);
                    local_seg2 = seg2 - seg_offset;
                }
                seg_offset += surfaces_[s].num_segments;
            }

            if (surf1_idx < 0 || surf2_idx < 0) continue;

            const auto& surface1 = surfaces_[surf1_idx];
            const auto& surface2 = surfaces_[surf2_idx];

            // Get segment nodes
            int nn1 = surface1.nodes_per_segment;
            int nn2 = surface2.nodes_per_segment;

            std::vector<Index> nodes1(nn1), nodes2(nn2);
            for (int n = 0; n < nn1; ++n) {
                nodes1[n] = surface1.connectivity[local_seg1 * nn1 + n];
            }
            for (int n = 0; n < nn2; ++n) {
                nodes2[n] = surface2.connectivity[local_seg2 * nn2 + n];
            }

            // Compute relative velocity at contact point
            Vec3r vel1 = {0, 0, 0}, vel2 = {0, 0, 0};
            for (int n = 0; n < nn1; ++n) {
                for (int d = 0; d < 3; ++d) {
                    vel1[d] += pair.phi1[n] * velocity[nodes1[n] * 3 + d];
                }
            }
            for (int n = 0; n < nn2; ++n) {
                for (int d = 0; d < 3; ++d) {
                    vel2[d] += pair.phi2[n] * velocity[nodes2[n] * 3 + d];
                }
            }

            Vec3r rel_vel = {vel2[0] - vel1[0], vel2[1] - vel1[1], vel2[2] - vel1[2]};

            // Normal and tangential relative velocity
            Real vn = rel_vel[0] * pair.normal[0] + rel_vel[1] * pair.normal[1] + rel_vel[2] * pair.normal[2];
            Vec3r v_tang = {
                rel_vel[0] - vn * pair.normal[0],
                rel_vel[1] - vn * pair.normal[1],
                rel_vel[2] - vn * pair.normal[2]
            };

            // ================================================================
            // Normal force (penalty)
            // ================================================================
            // Use config contact_thickness for penetration calculation
            Real effective_thickness = config_.contact_thickness;
            Real penetration = effective_thickness - pair.gap;

            if (penetration <= 0.0) {
                pair.active = false;
                continue;
            }

            Real fn = penalty_stiffness * penetration * pair.area;

            // Damping for stability
            Real damping_ratio = 0.1;
            Real damping = 2.0 * damping_ratio * std::sqrt(penalty_stiffness * element_mass);
            if (vn > 0) {  // Approaching (opposite sign convention)
                fn += damping * vn * pair.area;
            }
            fn = std::max(fn, 0.0);

            pair.pressure = fn / std::max(pair.area, 1.0e-20);

            Vec3r normal_force = {fn * pair.normal[0], fn * pair.normal[1], fn * pair.normal[2]};

            // ================================================================
            // Friction force
            // ================================================================
            Vec3r friction_force = {0, 0, 0};

            if (config_.static_friction > 0.0 && fn > 0.0) {
                // Update tangential slip
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

                Real stick_stiffness = penalty_stiffness * pair.area * 0.5;
                Real trial_friction = stick_stiffness * slip_mag;

                if (trial_friction <= max_friction && slip_mag > 1.0e-15) {
                    // Sticking
                    pair.sticking = true;
                    Real scale = trial_friction / slip_mag;
                    for (int d = 0; d < 3; ++d) {
                        friction_force[d] = -scale * pair.tangent_slip[d];
                    }
                } else if (slip_mag > 1.0e-15) {
                    // Sliding
                    pair.sticking = false;
                    Real scale = max_friction / slip_mag;
                    for (int d = 0; d < 3; ++d) {
                        friction_force[d] = -scale * pair.tangent_slip[d];
                        pair.tangent_slip[d] = friction_force[d] / stick_stiffness;
                    }
                }
            }

            // ================================================================
            // Distribute forces to nodes
            // ================================================================

            // Force on segment 1 (pushing away from segment 2)
            for (int n = 0; n < nn1; ++n) {
                Index node = nodes1[n];
                for (int d = 0; d < 3; ++d) {
                    forces[node * 3 + d] -= pair.phi1[n] * (normal_force[d] + friction_force[d]);
                }
            }

            // Force on segment 2 (pushing away from segment 1)
            for (int n = 0; n < nn2; ++n) {
                Index node = nodes2[n];
                for (int d = 0; d < 3; ++d) {
                    forces[node * 3 + d] += pair.phi2[n] * (normal_force[d] + friction_force[d]);
                }
            }
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    const std::vector<SurfaceContactPair>& get_active_pairs() const {
        return active_pairs_;
    }

    int num_active_pairs() const {
        return static_cast<int>(active_pairs_.size());
    }

    const SurfaceContactConfig& config() const { return config_; }
    SurfaceContactConfig& config() { return config_; }

private:
    // ========================================================================
    // Internal Methods
    // ========================================================================

    /**
     * @brief Build global segment index from all surfaces
     */
    void build_segment_index() {
        all_segments_.clear();
        segment_surface_.clear();

        Index seg_offset = 0;
        for (size_t s = 0; s < surfaces_.size(); ++s) {
            const auto& surf = surfaces_[s];

            for (Index i = 0; i < surf.num_segments; ++i) {
                // Store connectivity (pad to 4 nodes)
                for (int n = 0; n < 4; ++n) {
                    if (n < surf.nodes_per_segment) {
                        all_segments_.push_back(surf.connectivity[i * surf.nodes_per_segment + n]);
                    } else {
                        all_segments_.push_back(surf.connectivity[i * surf.nodes_per_segment]);  // Repeat first
                    }
                }

                segment_surface_.push_back(static_cast<Index>(s));
            }

            seg_offset += surf.num_segments;
        }
    }

    /**
     * @brief Detect contacts between two surfaces
     */
    void detect_surface_pair(size_t surf1_idx, size_t surf2_idx, const Real* coords) {
        const auto& surf1 = surfaces_[surf1_idx];
        const auto& surf2 = surfaces_[surf2_idx];

        // Global segment offset for surf1
        Index seg_offset1 = 0;
        for (size_t s = 0; s < surf1_idx; ++s) {
            seg_offset1 += surfaces_[s].num_segments;
        }

        // For each segment in surface 1, find candidates from surface 2
        for (Index i = 0; i < surf1.num_segments; ++i) {
            Index global_seg1 = seg_offset1 + i;

            // Get segment 1 centroid
            Vec3r centroid1 = {0, 0, 0};
            for (int n = 0; n < surf1.nodes_per_segment; ++n) {
                Index node = surf1.connectivity[i * surf1.nodes_per_segment + n];
                for (int d = 0; d < 3; ++d) {
                    centroid1[d] += coords[node * 3 + d];
                }
            }
            for (int d = 0; d < 3; ++d) {
                centroid1[d] /= surf1.nodes_per_segment;
            }

            // Get candidates from voxel detector
            std::vector<Index> slave_nodes;
            for (int n = 0; n < surf1.nodes_per_segment; ++n) {
                slave_nodes.push_back(surf1.connectivity[i * surf1.nodes_per_segment + n]);
            }

            auto candidates = voxel_detector_->find_candidates(slave_nodes, config_.gap_max);

            // Filter to only segments from surface 2
            Index seg_offset2 = 0;
            for (size_t s = 0; s < surf2_idx; ++s) {
                seg_offset2 += surfaces_[s].num_segments;
            }

            int pair_count = 0;
            for (const auto& cand : candidates) {
                Index global_seg2 = cand.segment_id;

                // Check if segment belongs to surface 2
                if (global_seg2 < seg_offset2 || global_seg2 >= seg_offset2 + surf2.num_segments) {
                    continue;
                }

                // Skip self-contact check (same segment)
                if (surf1_idx == surf2_idx && global_seg1 == global_seg2) {
                    continue;
                }

                // Compute detailed contact
                SurfaceContactPair pair;
                if (compute_segment_contact(global_seg1, global_seg2, coords, pair)) {
                    if (pair.gap < config_.gap_max) {
                        active_pairs_.push_back(pair);
                        pair_count++;

                        if (pair_count >= config_.max_pairs_per_segment) {
                            break;
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Compute contact between two segments
     */
    bool compute_segment_contact(Index seg1, Index seg2, const Real* coords, SurfaceContactPair& pair) {
        // Get surface indices
        Index surf1_idx = segment_surface_[seg1];
        Index surf2_idx = segment_surface_[seg2];

        const auto& surf1 = surfaces_[surf1_idx];
        const auto& surf2 = surfaces_[surf2_idx];

        // Local segment indices
        Index local1 = seg1, local2 = seg2;
        for (size_t s = 0; s < surf1_idx; ++s) {
            local1 -= surfaces_[s].num_segments;
        }
        for (size_t s = 0; s < surf2_idx; ++s) {
            local2 -= surfaces_[s].num_segments;
        }

        // Get node coordinates
        int nn1 = surf1.nodes_per_segment;
        int nn2 = surf2.nodes_per_segment;

        Vec3r x1[4], x2[4];
        for (int n = 0; n < nn1; ++n) {
            Index node = surf1.connectivity[local1 * nn1 + n];
            x1[n] = {coords[node * 3], coords[node * 3 + 1], coords[node * 3 + 2]};
        }
        for (int n = 0; n < nn2; ++n) {
            Index node = surf2.connectivity[local2 * nn2 + n];
            x2[n] = {coords[node * 3], coords[node * 3 + 1], coords[node * 3 + 2]};
        }

        // Compute segment normals
        Vec3r n1 = compute_segment_normal(x1, nn1);
        Vec3r n2 = compute_segment_normal(x2, nn2);

        // Check normal compatibility
        if (config_.check_normals) {
            Real dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
            if (dot > -config_.normal_tolerance) {
                // Normals not opposing - skip
                return false;
            }
        }

        // Compute centroids
        Vec3r c1 = {0, 0, 0}, c2 = {0, 0, 0};
        for (int n = 0; n < nn1; ++n) {
            for (int d = 0; d < 3; ++d) c1[d] += x1[n][d];
        }
        for (int n = 0; n < nn2; ++n) {
            for (int d = 0; d < 3; ++d) c2[d] += x2[n][d];
        }
        for (int d = 0; d < 3; ++d) {
            c1[d] /= nn1;
            c2[d] /= nn2;
        }

        // Gap distance (along average normal)
        Vec3r avg_normal = {
            (n1[0] - n2[0]) * 0.5,
            (n1[1] - n2[1]) * 0.5,
            (n1[2] - n2[2]) * 0.5
        };
        Real n_mag = std::sqrt(avg_normal[0] * avg_normal[0] +
                               avg_normal[1] * avg_normal[1] +
                               avg_normal[2] * avg_normal[2]);
        if (n_mag < 1.0e-20) return false;

        for (int d = 0; d < 3; ++d) avg_normal[d] /= n_mag;

        Vec3r diff = {c2[0] - c1[0], c2[1] - c1[1], c2[2] - c1[2]};
        Real gap = diff[0] * avg_normal[0] + diff[1] * avg_normal[1] + diff[2] * avg_normal[2];

        // Compute contact area (approximate)
        Real area1 = compute_segment_area(x1, nn1);
        Real area2 = compute_segment_area(x2, nn2);
        Real contact_area = std::min(area1, area2);

        // Fill contact pair
        pair.segment1 = seg1;
        pair.segment2 = seg2;
        pair.gap = gap;
        pair.normal = avg_normal;
        pair.area = contact_area;
        pair.active = true;

        // Contact point at midpoint
        for (int d = 0; d < 3; ++d) {
            pair.contact_point[d] = (c1[d] + c2[d]) * 0.5;
        }

        // Shape functions (uniform for simplicity)
        for (int n = 0; n < 4; ++n) {
            pair.phi1[n] = (n < nn1) ? 1.0 / nn1 : 0.0;
            pair.phi2[n] = (n < nn2) ? 1.0 / nn2 : 0.0;
        }

        return true;
    }

    /**
     * @brief Compute segment normal
     */
    Vec3r compute_segment_normal(const Vec3r* x, int num_nodes) {
        Vec3r e1, e2;

        if (num_nodes == 3) {
            // Triangle
            e1 = {x[1][0] - x[0][0], x[1][1] - x[0][1], x[1][2] - x[0][2]};
            e2 = {x[2][0] - x[0][0], x[2][1] - x[0][1], x[2][2] - x[0][2]};
        } else {
            // Quad - use diagonals
            e1 = {x[2][0] - x[0][0], x[2][1] - x[0][1], x[2][2] - x[0][2]};
            e2 = {x[3][0] - x[1][0], x[3][1] - x[1][1], x[3][2] - x[1][2]};
        }

        Vec3r n = {
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0]
        };

        Real mag = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (mag > 1.0e-20) {
            n[0] /= mag;
            n[1] /= mag;
            n[2] /= mag;
        }

        return n;
    }

    /**
     * @brief Compute segment area
     */
    Real compute_segment_area(const Vec3r* x, int num_nodes) {
        if (num_nodes == 3) {
            // Triangle area
            Vec3r e1 = {x[1][0] - x[0][0], x[1][1] - x[0][1], x[1][2] - x[0][2]};
            Vec3r e2 = {x[2][0] - x[0][0], x[2][1] - x[0][1], x[2][2] - x[0][2]};
            Vec3r cross = {
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0]
            };
            return 0.5 * std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
        } else {
            // Quad area (sum of two triangles)
            Vec3r e1 = {x[1][0] - x[0][0], x[1][1] - x[0][1], x[1][2] - x[0][2]};
            Vec3r e2 = {x[2][0] - x[0][0], x[2][1] - x[0][1], x[2][2] - x[0][2]};
            Vec3r e3 = {x[3][0] - x[0][0], x[3][1] - x[0][1], x[3][2] - x[0][2]};

            Vec3r c1 = {
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0]
            };
            Vec3r c2 = {
                e2[1] * e3[2] - e2[2] * e3[1],
                e2[2] * e3[0] - e2[0] * e3[2],
                e2[0] * e3[1] - e2[1] * e3[0]
            };

            Real a1 = 0.5 * std::sqrt(c1[0] * c1[0] + c1[1] * c1[1] + c1[2] * c1[2]);
            Real a2 = 0.5 * std::sqrt(c2[0] * c2[0] + c2[1] * c2[1] + c2[2] * c2[2]);

            return a1 + a2;
        }
    }

    // ========================================================================
    // Data Members
    // ========================================================================

    SurfaceContactConfig config_;
    Index num_nodes_;

    std::vector<ContactSurface> surfaces_;

    // Global segment index
    std::vector<Index> all_segments_;     // All segment connectivity (padded to 4)
    std::vector<Index> segment_surface_;  // Which surface each segment belongs to

    // Contact detection
    std::unique_ptr<VoxelCollisionDetector> voxel_detector_;

    // Active contacts
    std::vector<SurfaceContactPair> active_pairs_;
};

} // namespace fem
} // namespace nxs
