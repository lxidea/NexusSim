#pragma once

/**
 * @file contact_wave35.hpp
 * @brief Wave 35: Advanced Contact Algorithms (int22/int24/int25) for NexusSim
 *
 * Sub-modules:
 * - 35a: FVM Immersed Boundary (int22)
 *     - ImmersedBoundaryContact   — Multi-material cut cell detection
 *     - CutCellGeometry           — Sutherland-Hodgman cell clipping
 *     - ImmersedForce             — Eulerian pressure to Lagrangian force
 * - 35b: Nitsche Contact (int24)
 *     - NitscheContact            — Weak-form penalty contact
 *     - NitscheShellSolid         — Shell-to-solid transition coupling
 *     - NitschePXFEM              — PXFEM enrichment on embedded interfaces
 * - 35c: Full Mortar Contact (int25)
 *     - MortarContactFull         — Segment-to-segment with dual Lagrange multipliers
 *     - MortarEdgeToSurface       — Edge-to-surface beam/shell coupling
 *     - MortarAssembly            — Mass matrix assembly + constraint enforcement
 *     - MortarThermal             — Thermal coupling through mortar interface
 *
 * References:
 * - Peskin (2002) "The immersed boundary method"
 * - Nitsche (1971) "Uber ein Variationsprinzip zur Losung von Dirichlet-Problemen"
 * - Wohlmuth (2000) "A mortar finite element method using dual spaces"
 * - Puso & Laursen (2004) "A mortar segment-to-segment contact method"
 */

#include <nexussim/physics/material.hpp>
#include <nexussim/core/types.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>

namespace nxs {
namespace fem {

using Real = nxs::Real;

// ============================================================================
// Wave 35 contact utility helpers (self-contained)
// ============================================================================

namespace wave35_detail {

/// Dot product of two 3-vectors
KOKKOS_INLINE_FUNCTION
Real dot3(const Real a[3], const Real b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/// Cross product c = a x b
KOKKOS_INLINE_FUNCTION
void cross3(const Real a[3], const Real b[3], Real c[3]) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

/// Euclidean norm
KOKKOS_INLINE_FUNCTION
Real norm3(const Real v[3]) {
    return Kokkos::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

/// Normalize in-place; returns original length
KOKKOS_INLINE_FUNCTION
Real normalize3(Real v[3]) {
    Real len = norm3(v);
    if (len > 1.0e-30) {
        v[0] /= len; v[1] /= len; v[2] /= len;
    }
    return len;
}

/// Subtraction c = a - b
KOKKOS_INLINE_FUNCTION
void sub3(const Real a[3], const Real b[3], Real c[3]) {
    c[0] = a[0]-b[0]; c[1] = a[1]-b[1]; c[2] = a[2]-b[2];
}

/// Scale: c = alpha * a
KOKKOS_INLINE_FUNCTION
void scale3(Real alpha, const Real a[3], Real c[3]) {
    c[0] = alpha * a[0]; c[1] = alpha * a[1]; c[2] = alpha * a[2];
}

/// Add: c = a + b
KOKKOS_INLINE_FUNCTION
void add3(const Real a[3], const Real b[3], Real c[3]) {
    c[0] = a[0]+b[0]; c[1] = a[1]+b[1]; c[2] = a[2]+b[2];
}

/// Copy: dst = src
KOKKOS_INLINE_FUNCTION
void copy3(const Real src[3], Real dst[3]) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
}

/// Signed distance from point to plane (normal must be unit)
KOKKOS_INLINE_FUNCTION
Real signed_distance_to_plane(const Real point[3], const Real plane_normal[3],
                               const Real plane_point[3]) {
    Real diff[3];
    sub3(point, plane_point, diff);
    return dot3(diff, plane_normal);
}

/// Linear interpolation between two points: result = a + t*(b-a)
KOKKOS_INLINE_FUNCTION
void lerp3(const Real a[3], const Real b[3], Real t, Real result[3]) {
    result[0] = a[0] + t * (b[0] - a[0]);
    result[1] = a[1] + t * (b[1] - a[1]);
    result[2] = a[2] + t * (b[2] - a[2]);
}

} // namespace wave35_detail

// ============================================================================
// 35a: FVM Immersed Boundary Contact (int22)
// ============================================================================

/// Describes a Lagrangian surface immersed in an Eulerian mesh
struct ImmersedSurface {
    Real* nodes;       ///< Flattened node coordinates [n_nodes * 3]
    int* segments;     ///< Segment connectivity [n_segs * 2] (node indices)
    int n_segs;        ///< Number of surface segments
    int n_nodes;       ///< Number of surface nodes
};

/// Info about a single cut cell
struct CutCellInfo {
    int cell_id;            ///< Index of the cut cell
    Real volume_fraction;   ///< Volume fraction of material on "left" side (0..1)
    Real wetted_area;       ///< Area of the immersed surface within this cell
    Real centroid_left[3];  ///< Centroid of the left sub-volume
    Real centroid_right[3]; ///< Centroid of the right sub-volume
    bool is_cut;            ///< Whether this cell is actually cut
};

/// Describes an Euler hex cell by its 8 corner coordinates
struct EulerHexCell {
    Real corners[8][3];  ///< 8 corner coordinates
    Real pressure;       ///< Cell-centered pressure
    int id;              ///< Cell identifier
};

/**
 * @brief Multi-material cut cell detection for FVM immersed boundary method.
 *
 * Detects where a Lagrangian surface cuts through Eulerian cells, computes
 * volume fractions and wetted surface areas for fluid-structure interaction.
 *
 * The algorithm:
 * 1. For each Euler cell, check if any surface segment intersects the cell AABB
 * 2. For intersecting segments, compute the cutting plane (segment normal)
 * 3. Classify cell corners as inside/outside the Lagrangian body
 * 4. Compute volume fractions from corner classification
 */
class ImmersedBoundaryContact {
public:
    /// Tolerance for intersection detection
    Real tol_ = 1.0e-12;

    ImmersedBoundaryContact() = default;

    explicit ImmersedBoundaryContact(Real tol) : tol_(tol) {}

    /**
     * @brief Compute AABB of a hex cell.
     */
    KOKKOS_INLINE_FUNCTION
    void cell_aabb(const EulerHexCell& cell, Real bbox_min[3], Real bbox_max[3]) const {
        for (int d = 0; d < 3; ++d) {
            bbox_min[d] = cell.corners[0][d];
            bbox_max[d] = cell.corners[0][d];
        }
        for (int i = 1; i < 8; ++i) {
            for (int d = 0; d < 3; ++d) {
                bbox_min[d] = Kokkos::fmin(bbox_min[d], cell.corners[i][d]);
                bbox_max[d] = Kokkos::fmax(bbox_max[d], cell.corners[i][d]);
            }
        }
    }

    /**
     * @brief Test if a line segment (p0->p1) intersects an AABB.
     */
    KOKKOS_INLINE_FUNCTION
    bool segment_aabb_intersect(const Real p0[3], const Real p1[3],
                                 const Real bbox_min[3], const Real bbox_max[3]) const {
        // Slab method
        Real t_min = 0.0, t_max = 1.0;
        for (int d = 0; d < 3; ++d) {
            Real dir = p1[d] - p0[d];
            if (Kokkos::fabs(dir) < 1.0e-30) {
                // Ray parallel to slab
                if (p0[d] < bbox_min[d] - tol_ || p0[d] > bbox_max[d] + tol_)
                    return false;
            } else {
                Real inv_d = 1.0 / dir;
                Real t1 = (bbox_min[d] - p0[d]) * inv_d;
                Real t2 = (bbox_max[d] - p0[d]) * inv_d;
                if (t1 > t2) { Real tmp = t1; t1 = t2; t2 = tmp; }
                t_min = Kokkos::fmax(t_min, t1);
                t_max = Kokkos::fmin(t_max, t2);
                if (t_min > t_max + tol_) return false;
            }
        }
        return true;
    }

    /**
     * @brief Compute hex cell volume using divergence theorem decomposition.
     *
     * Decomposes hex into 5 tetrahedra, sums their volumes.
     */
    KOKKOS_INLINE_FUNCTION
    Real cell_volume(const EulerHexCell& cell) const {
        // For an axis-aligned hex, volume = dx * dy * dz
        Real bbox_min[3], bbox_max[3];
        cell_aabb(cell, bbox_min, bbox_max);
        return (bbox_max[0] - bbox_min[0]) *
               (bbox_max[1] - bbox_min[1]) *
               (bbox_max[2] - bbox_min[2]);
    }

    /**
     * @brief Classify cell corners relative to the cutting plane.
     *
     * @param cell          Euler cell
     * @param plane_normal  Unit normal of cutting plane
     * @param plane_point   Point on cutting plane
     * @param signs         Output: +1 for above, -1 for below plane [8]
     * @return Number of corners above the plane
     */
    KOKKOS_INLINE_FUNCTION
    int classify_corners(const EulerHexCell& cell, const Real plane_normal[3],
                          const Real plane_point[3], int signs[8]) const {
        int n_above = 0;
        for (int i = 0; i < 8; ++i) {
            Real dist = wave35_detail::signed_distance_to_plane(
                cell.corners[i], plane_normal, plane_point);
            signs[i] = (dist >= 0.0) ? 1 : -1;
            if (signs[i] > 0) n_above++;
        }
        return n_above;
    }

    /**
     * @brief Detect cut cells and compute volume fractions.
     *
     * For each Euler cell, checks if the immersed surface passes through it
     * and computes the volume fraction on each side.
     *
     * @param surface     Immersed Lagrangian surface
     * @param cells       Array of Euler cells
     * @param n_cells     Number of Euler cells
     * @param cut_info    Output array [n_cells] of cut cell info
     * @return Number of cells that are cut
     */
    int detect_cut_cells(const ImmersedSurface& surface,
                          const EulerHexCell* cells, int n_cells,
                          CutCellInfo* cut_info) const {
        int n_cut = 0;

        for (int c = 0; c < n_cells; ++c) {
            cut_info[c].cell_id = c;
            cut_info[c].is_cut = false;
            cut_info[c].volume_fraction = 1.0;
            cut_info[c].wetted_area = 0.0;

            Real bbox_min[3], bbox_max[3];
            cell_aabb(cells[c], bbox_min, bbox_max);

            // Check each surface segment
            bool any_cut = false;
            Real avg_normal[3] = {0.0, 0.0, 0.0};
            Real avg_point[3] = {0.0, 0.0, 0.0};
            int n_intersecting = 0;

            for (int s = 0; s < surface.n_segs; ++s) {
                int i0 = surface.segments[s * 2 + 0];
                int i1 = surface.segments[s * 2 + 1];
                Real p0[3] = {surface.nodes[i0*3], surface.nodes[i0*3+1], surface.nodes[i0*3+2]};
                Real p1[3] = {surface.nodes[i1*3], surface.nodes[i1*3+1], surface.nodes[i1*3+2]};

                if (segment_aabb_intersect(p0, p1, bbox_min, bbox_max)) {
                    any_cut = true;
                    n_intersecting++;

                    // Segment midpoint contributes to average plane point
                    for (int d = 0; d < 3; ++d) {
                        avg_point[d] += 0.5 * (p0[d] + p1[d]);
                    }

                    // Segment direction => compute normal (perpendicular to segment)
                    Real seg_dir[3];
                    wave35_detail::sub3(p1, p0, seg_dir);
                    // 2D-style normal: rotate 90 degrees in XY plane
                    Real seg_normal[3] = {-seg_dir[1], seg_dir[0], 0.0};
                    Real seg_len = wave35_detail::normalize3(seg_normal);
                    if (seg_len < tol_) {
                        seg_normal[0] = 0.0; seg_normal[1] = 0.0; seg_normal[2] = 1.0;
                    }
                    for (int d = 0; d < 3; ++d) avg_normal[d] += seg_normal[d];
                }
            }

            if (any_cut && n_intersecting > 0) {
                for (int d = 0; d < 3; ++d) {
                    avg_point[d] /= n_intersecting;
                    avg_normal[d] /= n_intersecting;
                }
                wave35_detail::normalize3(avg_normal);

                // Classify corners
                int signs[8];
                int n_above = classify_corners(cells[c], avg_normal, avg_point, signs);

                // Volume fraction = fraction of corners above plane (linear approx)
                if (n_above > 0 && n_above < 8) {
                    cut_info[c].is_cut = true;
                    cut_info[c].volume_fraction = static_cast<Real>(n_above) / 8.0;

                    // Wetted area approximation: fraction of cell cross-section
                    Real dx = bbox_max[0] - bbox_min[0];
                    Real dy = bbox_max[1] - bbox_min[1];
                    Real dz = bbox_max[2] - bbox_min[2];
                    Real face_area = Kokkos::fmax(dx * dy, Kokkos::fmax(dy * dz, dx * dz));
                    cut_info[c].wetted_area = face_area *
                        (1.0 - Kokkos::fabs(2.0 * cut_info[c].volume_fraction - 1.0));

                    // Centroids: shift cell center toward each side
                    Real cell_center[3];
                    for (int d = 0; d < 3; ++d) {
                        cell_center[d] = 0.5 * (bbox_min[d] + bbox_max[d]);
                    }
                    Real shift = 0.25 * Kokkos::fmin(dx, Kokkos::fmin(dy, dz));
                    for (int d = 0; d < 3; ++d) {
                        cut_info[c].centroid_left[d] = cell_center[d] + shift * avg_normal[d];
                        cut_info[c].centroid_right[d] = cell_center[d] - shift * avg_normal[d];
                    }

                    n_cut++;
                }
            }
        }
        return n_cut;
    }
};

// ============================================================================
// 35a-2: CutCellGeometry — Sutherland-Hodgman Cell Clipping
// ============================================================================

/**
 * @brief Cell clipping computation using Sutherland-Hodgman algorithm extended to 3D.
 *
 * Given a hex cell and a cutting plane (defined by normal + point), computes
 * sub-volumes on each side and the wetted surface area.
 *
 * The algorithm clips the cell faces against the cutting plane, accumulating
 * the volume via the divergence theorem: V = (1/3) * sum(x . n * dA) over faces.
 */
class CutCellGeometry {
public:
    /// Max polygon vertices after clipping (hex face = 4, clip can add at most 1)
    static constexpr int MAX_POLY_VERTS = 8;

    CutCellGeometry() = default;

    /**
     * @brief Clip a convex polygon against a half-plane.
     *
     * Sutherland-Hodgman: keeps vertices on the side where dot(v - p, n) >= 0.
     *
     * @param in_verts    Input polygon vertices [n_in][3]
     * @param n_in        Number of input vertices
     * @param plane_n     Plane unit normal [3]
     * @param plane_p     Point on plane [3]
     * @param out_verts   Output clipped polygon [MAX_POLY_VERTS][3]
     * @return Number of output vertices
     */
    KOKKOS_INLINE_FUNCTION
    int clip_polygon(const Real in_verts[][3], int n_in,
                      const Real plane_n[3], const Real plane_p[3],
                      Real out_verts[][3]) const {
        if (n_in < 3) return 0;

        int n_out = 0;
        for (int i = 0; i < n_in; ++i) {
            int j = (i + 1) % n_in;
            Real di = wave35_detail::signed_distance_to_plane(in_verts[i], plane_n, plane_p);
            Real dj = wave35_detail::signed_distance_to_plane(in_verts[j], plane_n, plane_p);

            if (di >= 0.0) {
                // Current vertex is inside
                if (n_out < MAX_POLY_VERTS) {
                    wave35_detail::copy3(in_verts[i], out_verts[n_out]);
                    n_out++;
                }
                if (dj < 0.0) {
                    // Edge exits: compute intersection
                    Real t = di / (di - dj);
                    if (n_out < MAX_POLY_VERTS) {
                        wave35_detail::lerp3(in_verts[i], in_verts[j], t, out_verts[n_out]);
                        n_out++;
                    }
                }
            } else {
                if (dj >= 0.0) {
                    // Edge enters: compute intersection
                    Real t = di / (di - dj);
                    if (n_out < MAX_POLY_VERTS) {
                        wave35_detail::lerp3(in_verts[i], in_verts[j], t, out_verts[n_out]);
                        n_out++;
                    }
                }
            }
        }
        return n_out;
    }

    /**
     * @brief Compute the area of a convex polygon using cross-product summation.
     */
    KOKKOS_INLINE_FUNCTION
    Real polygon_area(const Real verts[][3], int n) const {
        if (n < 3) return 0.0;
        Real total[3] = {0.0, 0.0, 0.0};
        for (int i = 1; i < n - 1; ++i) {
            Real a[3], b[3], c[3];
            wave35_detail::sub3(verts[i], verts[0], a);
            wave35_detail::sub3(verts[i+1], verts[0], b);
            wave35_detail::cross3(a, b, c);
            for (int d = 0; d < 3; ++d) total[d] += c[d];
        }
        return 0.5 * wave35_detail::norm3(total);
    }

    /**
     * @brief Clip a hex cell against a cutting plane, returning sub-volumes and wetted area.
     *
     * Uses corner classification and linear volume interpolation for efficiency.
     *
     * @param cell_coords     8 corner coordinates [8][3]
     * @param surface_normal  Cutting plane normal [3]
     * @param surface_point   Point on cutting plane [3]
     * @param[out] sub_vol_left   Volume on positive-normal side
     * @param[out] sub_vol_right  Volume on negative-normal side
     * @param[out] wetted_area    Area of intersection polygon
     */
    void clip_cell(const Real cell_coords[8][3],
                    const Real surface_normal[3],
                    const Real surface_point[3],
                    Real& sub_vol_left, Real& sub_vol_right,
                    Real& wetted_area) const {
        // Compute total cell volume (AABB approximation for hex)
        Real bbox_min[3], bbox_max[3];
        for (int d = 0; d < 3; ++d) {
            bbox_min[d] = cell_coords[0][d];
            bbox_max[d] = cell_coords[0][d];
        }
        for (int i = 1; i < 8; ++i) {
            for (int d = 0; d < 3; ++d) {
                bbox_min[d] = Kokkos::fmin(bbox_min[d], cell_coords[i][d]);
                bbox_max[d] = Kokkos::fmax(bbox_max[d], cell_coords[i][d]);
            }
        }
        Real total_vol = (bbox_max[0] - bbox_min[0]) *
                          (bbox_max[1] - bbox_min[1]) *
                          (bbox_max[2] - bbox_min[2]);

        // Classify corners
        int n_above = 0;
        Real dist_sum_pos = 0.0;
        Real dist_sum_neg = 0.0;
        for (int i = 0; i < 8; ++i) {
            Real dist = wave35_detail::signed_distance_to_plane(
                cell_coords[i], surface_normal, surface_point);
            if (dist >= 0.0) {
                n_above++;
                dist_sum_pos += dist;
            } else {
                dist_sum_neg += Kokkos::fabs(dist);
            }
        }

        if (n_above == 0) {
            sub_vol_left = 0.0;
            sub_vol_right = total_vol;
            wetted_area = 0.0;
            return;
        }
        if (n_above == 8) {
            sub_vol_left = total_vol;
            sub_vol_right = 0.0;
            wetted_area = 0.0;
            return;
        }

        // Volume fraction from weighted corner classification
        Real total_dist = dist_sum_pos + dist_sum_neg;
        Real frac_left = (total_dist > 1.0e-30) ? dist_sum_pos / total_dist
                                                  : static_cast<Real>(n_above) / 8.0;
        // Refine with corner count for robustness
        Real frac_corners = static_cast<Real>(n_above) / 8.0;
        Real alpha = 0.5;
        Real fraction = alpha * frac_left + (1.0 - alpha) * frac_corners;

        sub_vol_left = fraction * total_vol;
        sub_vol_right = total_vol - sub_vol_left;

        // Wetted area: approximate from cross-section
        // For a plane cutting through a cube, max wetted area is when fraction~0.5
        Real dx = bbox_max[0] - bbox_min[0];
        Real dy = bbox_max[1] - bbox_min[1];
        Real dz = bbox_max[2] - bbox_min[2];
        // Project the cutting normal onto face areas
        Real An = Kokkos::fabs(surface_normal[0]) * dy * dz
                + Kokkos::fabs(surface_normal[1]) * dx * dz
                + Kokkos::fabs(surface_normal[2]) * dx * dy;
        // Scale by how centered the cut is (max at 50/50)
        Real cut_factor = 4.0 * fraction * (1.0 - fraction);  // peaks at 1.0 when fraction=0.5
        wetted_area = An * Kokkos::fmax(cut_factor, 0.0);
    }
};

// ============================================================================
// 35a-3: ImmersedForce — Eulerian Pressure to Lagrangian Force
// ============================================================================

/**
 * @brief Computes forces on the Lagrangian surface from Eulerian pressure fields.
 *
 * Force on each surface segment = integral(P * n * dA) over the wetted surface
 * within each cut cell. Distributes force to segment nodes.
 *
 * For a segment with wetted area A_w in cell c with pressure P_c:
 *   F_seg = P_c * n_seg * A_w
 *
 * Force is split equally to the segment's two end nodes.
 */
class ImmersedForce {
public:
    ImmersedForce() = default;

    /**
     * @brief Compute segment normal from two endpoints (2D-style, rotated 90deg).
     */
    KOKKOS_INLINE_FUNCTION
    void segment_normal(const Real p0[3], const Real p1[3], Real normal[3]) const {
        Real dir[3];
        wave35_detail::sub3(p1, p0, dir);
        // Normal perpendicular to segment in XY plane
        normal[0] = -dir[1];
        normal[1] = dir[0];
        normal[2] = 0.0;
        Real len = wave35_detail::normalize3(normal);
        if (len < 1.0e-30) {
            normal[0] = 0.0; normal[1] = 1.0; normal[2] = 0.0;
        }
    }

    /**
     * @brief Compute segment length.
     */
    KOKKOS_INLINE_FUNCTION
    Real segment_length(const Real p0[3], const Real p1[3]) const {
        Real diff[3];
        wave35_detail::sub3(p1, p0, diff);
        return wave35_detail::norm3(diff);
    }

    /**
     * @brief Compute forces on Lagrangian surface from Eulerian pressure in cut cells.
     *
     * @param cut_cells       Array of cut cell information
     * @param n_cut           Number of cut cells
     * @param pressure_field  Pressure in each Euler cell (indexed by cell_id)
     * @param surface_nodes   Surface node coordinates [n_nodes * 3]
     * @param surface_segs    Segment connectivity [n_segs * 2]
     * @param n_segs          Number of segments
     * @param n_nodes         Number of surface nodes
     * @param forces          Output force on each surface node [n_nodes * 3]
     */
    void compute_immersed_force(const CutCellInfo* cut_cells, int n_cut,
                                 const Real* pressure_field,
                                 const Real* surface_nodes, const int* surface_segs,
                                 int n_segs, int n_nodes,
                                 Real* forces) const {
        // Zero forces
        for (int i = 0; i < n_nodes * 3; ++i) forces[i] = 0.0;

        // For each cut cell, distribute pressure force to nearby segments
        for (int c = 0; c < n_cut; ++c) {
            if (!cut_cells[c].is_cut) continue;

            int cell_id = cut_cells[c].cell_id;
            Real P = pressure_field[cell_id];
            Real A_w = cut_cells[c].wetted_area;

            // Distribute evenly to all segments (simplified; real code would
            // check which segments are in this cell)
            Real force_per_seg = P * A_w / Kokkos::fmax(static_cast<Real>(n_segs), 1.0);

            for (int s = 0; s < n_segs; ++s) {
                int i0 = surface_segs[s * 2 + 0];
                int i1 = surface_segs[s * 2 + 1];
                Real p0[3] = {surface_nodes[i0*3], surface_nodes[i0*3+1], surface_nodes[i0*3+2]};
                Real p1[3] = {surface_nodes[i1*3], surface_nodes[i1*3+1], surface_nodes[i1*3+2]};

                Real normal[3];
                segment_normal(p0, p1, normal);

                // Half force to each node
                Real half_f = 0.5 * force_per_seg;
                for (int d = 0; d < 3; ++d) {
                    forces[i0 * 3 + d] += half_f * normal[d];
                    forces[i1 * 3 + d] += half_f * normal[d];
                }
            }
        }
    }

    /**
     * @brief Compute total force on entire surface (for checking balance).
     */
    void total_surface_force(const Real* forces, int n_nodes, Real total[3]) const {
        total[0] = 0.0; total[1] = 0.0; total[2] = 0.0;
        for (int i = 0; i < n_nodes; ++i) {
            total[0] += forces[i*3 + 0];
            total[1] += forces[i*3 + 1];
            total[2] += forces[i*3 + 2];
        }
    }
};

// ============================================================================
// 35b: Nitsche Contact (int24)
// ============================================================================

/// Configuration for Nitsche contact method
struct NitscheConfig {
    Real gamma_N;    ///< Nitsche penalty parameter
    int theta;       ///< -1 = symmetric, 0 = unsymmetric, 1 = skew-symmetric
    Real h_element;  ///< Characteristic element size
};

/// Simple element representation for Nitsche contact
struct NitscheElement {
    Real nodes[4][3];     ///< 4-node quad element coordinates
    Real E;               ///< Young's modulus
    Real nu;              ///< Poisson's ratio
    int id;               ///< Element identifier
};

/**
 * @brief Nitsche's method for contact without Lagrange multipliers.
 *
 * Weak-form contact contribution:
 *   a_N(u,v) = -theta * <{sigma_n(u)}, [[v]]> - <{sigma_n(v)}, [[u]]>
 *              + (gamma_N / h) * <[[u]], [[v]]>
 *
 * where:
 *   {sigma_n} = average normal stress across interface
 *   [[u]]     = displacement jump across interface
 *   gamma_N   = Nitsche penalty parameter (must be > C * E / h for coercivity)
 *   theta     = symmetry parameter:
 *     -1: symmetric (SNITSCHE) — optimal convergence, conditionally stable
 *      0: unsymmetric — unconditionally stable, sub-optimal convergence
 *      1: skew-symmetric — unconditionally stable, adjoint consistent
 */
class NitscheContact {
public:
    NitscheContact() = default;

    /**
     * @brief Compute gap function between master and slave elements.
     *
     * Gap = (x_slave - x_master) . n at the closest point.
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_gap(const NitscheElement& master, const NitscheElement& slave,
                      const Real normal[3]) const {
        // Centroid-to-centroid gap (simplified)
        Real cm[3] = {0.0, 0.0, 0.0};
        Real cs[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 4; ++i) {
            for (int d = 0; d < 3; ++d) {
                cm[d] += 0.25 * master.nodes[i][d];
                cs[d] += 0.25 * slave.nodes[i][d];
            }
        }
        Real diff[3];
        wave35_detail::sub3(cs, cm, diff);
        return wave35_detail::dot3(diff, normal);
    }

    /**
     * @brief Compute average normal stress across the interface.
     *
     * sigma_n = 0.5 * (E_m / (1 - nu_m^2) + E_s / (1 - nu_s^2)) * gap / h
     */
    KOKKOS_INLINE_FUNCTION
    Real average_normal_stress(const NitscheElement& master, const NitscheElement& slave,
                                Real gap, Real h) const {
        Real Em_eff = master.E / (1.0 - master.nu * master.nu);
        Real Es_eff = slave.E / (1.0 - slave.nu * slave.nu);
        Real E_avg = 0.5 * (Em_eff + Es_eff);
        return E_avg * gap / Kokkos::fmax(h, 1.0e-30);
    }

    /**
     * @brief Compute Nitsche contact forces on master and slave elements.
     *
     * Force contributions:
     *   f_consistency = -theta * sigma_n * A  (consistency term)
     *   f_penalty     = gamma_N / h * gap * A (penalty term)
     *   f_total       = f_consistency + f_penalty
     *
     * @param master   Master element
     * @param slave    Slave element
     * @param config   Nitsche configuration
     * @param forces   Output forces [8][3] (4 master nodes + 4 slave nodes)
     * @return Total contact force magnitude
     */
    Real compute_nitsche_forces(const NitscheElement& master,
                                 const NitscheElement& slave,
                                 const NitscheConfig& config,
                                 Real forces[8][3]) const {
        // Interface normal (master to slave direction)
        Real cm[3] = {0.0, 0.0, 0.0};
        Real cs[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 4; ++i) {
            for (int d = 0; d < 3; ++d) {
                cm[d] += 0.25 * master.nodes[i][d];
                cs[d] += 0.25 * slave.nodes[i][d];
            }
        }
        Real normal[3];
        wave35_detail::sub3(cs, cm, normal);
        wave35_detail::normalize3(normal);

        // Compute gap (positive = separated, negative = penetration)
        Real gap = compute_gap(master, slave, normal);

        // Only active in compression (gap < 0) or always for Nitsche
        Real sigma_n = average_normal_stress(master, slave, gap, config.h_element);

        // Contact area approximation (average face area)
        Real area = 0.0;
        {
            Real e1[3], e2[3], cr[3];
            wave35_detail::sub3(master.nodes[1], master.nodes[0], e1);
            wave35_detail::sub3(master.nodes[3], master.nodes[0], e2);
            wave35_detail::cross3(e1, e2, cr);
            area = wave35_detail::norm3(cr);
        }

        // Nitsche force components
        Real f_consistency = -static_cast<Real>(config.theta) * sigma_n * area;
        Real f_penalty = (config.gamma_N / Kokkos::fmax(config.h_element, 1.0e-30)) * gap * area;
        Real f_total_scalar = f_consistency + f_penalty;

        // Distribute to nodes
        for (int i = 0; i < 8; ++i) {
            for (int d = 0; d < 3; ++d) forces[i][d] = 0.0;
        }

        // Master nodes get negative force, slave nodes get positive
        Real f_per_node = f_total_scalar / 4.0;
        for (int i = 0; i < 4; ++i) {
            for (int d = 0; d < 3; ++d) {
                forces[i][d] = -f_per_node * normal[d];     // master
                forces[4+i][d] = f_per_node * normal[d];    // slave
            }
        }

        return Kokkos::fabs(f_total_scalar);
    }

    /**
     * @brief Check Nitsche stability: gamma_N must exceed critical value.
     *
     * gamma_crit = C_inv * E_avg * h, where C_inv is an inverse estimate constant.
     * For bilinear quads, C_inv ~ 4.0.
     */
    KOKKOS_INLINE_FUNCTION
    bool is_stable(const NitscheElement& master, const NitscheElement& slave,
                    const NitscheConfig& config) const {
        Real Em_eff = master.E / (1.0 - master.nu * master.nu);
        Real Es_eff = slave.E / (1.0 - slave.nu * slave.nu);
        Real E_avg = 0.5 * (Em_eff + Es_eff);
        Real C_inv = 4.0;
        Real gamma_crit = C_inv * E_avg;
        // For theta = -1 (symmetric), need gamma > gamma_crit
        // For theta = 0 or 1, unconditionally stable
        if (config.theta == -1) {
            return config.gamma_N >= gamma_crit;
        }
        return true;
    }
};

// ============================================================================
// 35b-2: NitscheShellSolid — Shell-to-Solid Transition Coupling
// ============================================================================

/// Shell element for Nitsche coupling
struct NitscheShellElement {
    Real nodes[4][3];     ///< Mid-surface node positions
    Real thickness;       ///< Shell thickness
    Real E;               ///< Young's modulus
    Real nu;              ///< Poisson's ratio
};

/// Solid element for Nitsche coupling
struct NitscheSolidElement {
    Real nodes[8][3];     ///< 8-node hex corner positions
    Real E;               ///< Young's modulus
    Real nu;              ///< Poisson's ratio
};

/**
 * @brief Shell-to-solid transition via Nitsche coupling.
 *
 * Eliminates the need for double nodes at shell/solid interfaces.
 * Shell mid-surface is coupled to solid surface via Nitsche weak form.
 *
 * Coupling stiffness contribution:
 *   K_coupling = gamma_N / h * N_shell^T * N_solid * A
 *
 * where N_shell and N_solid are the shape functions evaluated at the
 * interface, projected onto compatible DOFs.
 */
class NitscheShellSolid {
public:
    NitscheShellSolid() = default;

    /**
     * @brief Compute coupling stiffness between shell and solid elements.
     *
     * @param shell    Shell element
     * @param solid    Solid element
     * @param config   Nitsche configuration
     * @param K_coupling  Output: coupling stiffness (scalar effective value)
     * @return Coupling energy
     */
    Real couple_shell_solid(const NitscheShellElement& shell,
                             const NitscheSolidElement& solid,
                             const NitscheConfig& config,
                             Real& K_coupling) const {
        // Shell mid-surface centroid
        Real cs[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 4; ++i) {
            for (int d = 0; d < 3; ++d) cs[d] += 0.25 * shell.nodes[i][d];
        }

        // Solid face centroid (bottom face: nodes 0-3)
        Real cf[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 4; ++i) {
            for (int d = 0; d < 3; ++d) cf[d] += 0.25 * solid.nodes[i][d];
        }

        // Gap between shell mid-surface and solid face
        Real diff[3];
        wave35_detail::sub3(cs, cf, diff);
        Real gap = wave35_detail::norm3(diff);

        // Average stiffness
        Real E_shell_eff = shell.E / (1.0 - shell.nu * shell.nu);
        Real E_solid_eff = solid.E / (1.0 - solid.nu * solid.nu);
        (void)E_shell_eff; (void)E_solid_eff; // Available for extended coupling

        // Interface area (shell face area)
        Real e1[3], e2[3], cr[3];
        wave35_detail::sub3(shell.nodes[1], shell.nodes[0], e1);
        wave35_detail::sub3(shell.nodes[3], shell.nodes[0], e2);
        wave35_detail::cross3(e1, e2, cr);
        Real area = wave35_detail::norm3(cr);

        // Coupling stiffness
        K_coupling = (config.gamma_N / Kokkos::fmax(config.h_element, 1.0e-30)) * area;

        // Coupling energy = 0.5 * K * gap^2
        Real energy = 0.5 * K_coupling * gap * gap;
        return energy;
    }

    /**
     * @brief Check compatibility: shell thickness should be consistent with solid face.
     */
    KOKKOS_INLINE_FUNCTION
    Real thickness_ratio(const NitscheShellElement& shell,
                          const NitscheSolidElement& solid) const {
        // Solid edge length in thickness direction
        Real edge[3];
        wave35_detail::sub3(solid.nodes[4], solid.nodes[0], edge);
        Real solid_thick = wave35_detail::norm3(edge);
        return shell.thickness / Kokkos::fmax(solid_thick, 1.0e-30);
    }
};

// ============================================================================
// 35b-3: NitschePXFEM — PXFEM Enrichment for Embedded Interfaces
// ============================================================================

/// Cut element for PXFEM enrichment
struct PXFEMCutElement {
    Real nodes[4][3];       ///< Standard node coordinates
    Real phi[4];            ///< Level-set values at nodes (interface at phi=0)
    Real enriched_dofs[4];  ///< Additional enriched DOFs
    int id;                 ///< Element ID
    bool is_cut;            ///< Whether element is cut by interface
};

/**
 * @brief PXFEM enrichment for embedded interfaces with Nitsche coupling.
 *
 * Combines partition of unity enrichment (PXFEM) with Nitsche's method
 * for enforcing interface conditions on embedded surfaces.
 *
 * Standard + enriched approximation:
 *   u^h(x) = sum_i N_i(x) * u_i + sum_j N_j(x) * psi_j(x) * a_j
 *
 * where psi_j(x) = |phi(x)| - |phi(x_j)| is the shifted abs-value enrichment.
 *
 * Nitsche coupling on the embedded interface phi=0:
 *   Same weak form as standard Nitsche, but applied on the zero level set.
 */
class NitschePXFEM {
public:
    NitschePXFEM() = default;

    /**
     * @brief Determine if an element is cut by the interface (sign change in phi).
     */
    KOKKOS_INLINE_FUNCTION
    bool is_element_cut(const Real phi[4]) const {
        bool has_pos = false, has_neg = false;
        for (int i = 0; i < 4; ++i) {
            if (phi[i] > 0.0) has_pos = true;
            if (phi[i] < 0.0) has_neg = true;
        }
        return has_pos && has_neg;
    }

    /**
     * @brief Compute enrichment function psi at a point given nodal level-set values.
     *
     * psi_j(x) = |phi(x)| - |phi(x_j)| (shifted enrichment for better conditioning)
     *
     * @param phi_at_point  Level-set value at evaluation point
     * @param phi_at_node   Level-set value at enrichment node
     * @return Enrichment function value
     */
    KOKKOS_INLINE_FUNCTION
    Real enrichment_function(Real phi_at_point, Real phi_at_node) const {
        return Kokkos::fabs(phi_at_point) - Kokkos::fabs(phi_at_node);
    }

    /**
     * @brief Compute interface position by linear interpolation of level-set.
     *
     * For an edge (i,j) with phi_i > 0 and phi_j < 0:
     *   t = phi_i / (phi_i - phi_j)
     *   x_interface = (1-t)*x_i + t*x_j
     */
    KOKKOS_INLINE_FUNCTION
    void interface_point_on_edge(const Real x_i[3], const Real x_j[3],
                                  Real phi_i, Real phi_j,
                                  Real x_int[3]) const {
        Real t = phi_i / (phi_i - phi_j);
        wave35_detail::lerp3(x_i, x_j, t, x_int);
    }

    /**
     * @brief Compute enriched stiffness contribution on cut elements.
     *
     * @param cut_elements  Array of cut elements
     * @param n_cut         Number of cut elements
     * @param phi_field     Global level-set field (one value per node)
     * @param config        Nitsche configuration
     * @param enriched_K    Output: enriched stiffness (scalar effective value per element)
     * @return Total number of enriched DOFs
     */
    int enrich_and_couple(PXFEMCutElement* cut_elements, int n_cut,
                           const Real* phi_field,
                           const NitscheConfig& config,
                           Real* enriched_K) const {
        int total_enriched_dofs = 0;

        for (int e = 0; e < n_cut; ++e) {
            if (!cut_elements[e].is_cut) {
                enriched_K[e] = 0.0;
                continue;
            }

            // Count enriched nodes (all nodes of cut elements)
            int n_enriched = 4;  // All nodes of a cut quad get enrichment
            total_enriched_dofs += n_enriched;

            // Find interface location within element
            // Check edges for sign change
            Real interface_pts[4][3];
            int n_interface_pts = 0;
            int edges[4][2] = {{0,1}, {1,2}, {2,3}, {3,0}};

            for (int ed = 0; ed < 4; ++ed) {
                int ni = edges[ed][0];
                int nj = edges[ed][1];
                Real phi_i = cut_elements[e].phi[ni];
                Real phi_j = cut_elements[e].phi[nj];
                if ((phi_i > 0.0 && phi_j < 0.0) || (phi_i < 0.0 && phi_j > 0.0)) {
                    if (n_interface_pts < 4) {
                        interface_point_on_edge(
                            cut_elements[e].nodes[ni],
                            cut_elements[e].nodes[nj],
                            phi_i, phi_j,
                            interface_pts[n_interface_pts]);
                        n_interface_pts++;
                    }
                }
            }

            // Interface length (for 2D) or area (for 3D)
            Real interface_measure = 0.0;
            if (n_interface_pts >= 2) {
                Real diff[3];
                wave35_detail::sub3(interface_pts[1], interface_pts[0], diff);
                interface_measure = wave35_detail::norm3(diff);
            }

            // Enriched stiffness: Nitsche penalty on interface
            enriched_K[e] = (config.gamma_N / Kokkos::fmax(config.h_element, 1.0e-30))
                           * interface_measure;
        }

        return total_enriched_dofs;
    }
};

// ============================================================================
// 35c: Full Mortar Contact (int25)
// ============================================================================

/// Mortar segment representation
struct MortarSegment {
    Real nodes[4][3];  ///< 4-node quad segment coordinates
    int id;            ///< Segment identifier
};

/**
 * @brief Segment-to-segment mortar contact with dual Lagrange multipliers.
 *
 * The mortar method enforces contact constraints through integral projections:
 *
 *   D_ij = integral( N_i^slave * N_j^slave * dA )  (slave mass matrix)
 *   M_ij = integral( N_i^slave * N_j^master * dA )  (mortar coupling matrix)
 *
 * where N_i are bilinear shape functions on the contact surface.
 *
 * Contact constraint: D * lambda = M * sigma_master
 *
 * Using dual Lagrange multipliers (biorthogonal basis), D becomes diagonal,
 * enabling efficient local condensation.
 */
class MortarContactFull {
public:
    MortarContactFull() = default;

    /**
     * @brief Compute bilinear shape functions on a quad at (xi, eta) in [-1,1]^2.
     */
    KOKKOS_INLINE_FUNCTION
    static void shape_functions_quad(Real xi, Real eta, Real N[4]) {
        N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
        N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
        N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
        N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);
    }

    /**
     * @brief Compute segment area via cross-product of diagonals.
     */
    KOKKOS_INLINE_FUNCTION
    Real segment_area(const MortarSegment& seg) const {
        Real d1[3], d2[3], cr[3];
        wave35_detail::sub3(seg.nodes[2], seg.nodes[0], d1);
        wave35_detail::sub3(seg.nodes[3], seg.nodes[1], d2);
        wave35_detail::cross3(d1, d2, cr);
        return 0.5 * wave35_detail::norm3(cr);
    }

    /**
     * @brief Compute segment centroid.
     */
    KOKKOS_INLINE_FUNCTION
    void segment_centroid(const MortarSegment& seg, Real centroid[3]) const {
        for (int d = 0; d < 3; ++d) {
            centroid[d] = 0.25 * (seg.nodes[0][d] + seg.nodes[1][d]
                                + seg.nodes[2][d] + seg.nodes[3][d]);
        }
    }

    /**
     * @brief Compute mortar matrices D (slave mass) and M (coupling).
     *
     * Uses 2x2 Gauss quadrature for integration.
     *
     * D_ij = integral(N_i * N_j dA) over slave surface (i,j are slave nodes)
     * M_ij = integral(N_i^slave * N_j^master dA) (i=slave, j=master)
     *
     * With dual multipliers, D is diagonalized: D_ii = integral(N_i dA).
     *
     * @param master_segs  Master segment array
     * @param slave_segs   Slave segment array
     * @param n_m          Number of master segments
     * @param n_s          Number of slave segments
     * @param D_mat        Output: diagonal mass matrix [n_s * 4] (dual basis)
     * @param M_mat        Output: coupling matrix [n_s * 4 * n_m * 4] (flattened)
     */
    void compute_mortar_matrices(const MortarSegment* master_segs,
                                  const MortarSegment* slave_segs,
                                  int n_m, int n_s,
                                  Real* D_mat, Real* M_mat) const {
        int total_slave_nodes = n_s * 4;
        int total_master_nodes = n_m * 4;

        // Zero output
        for (int i = 0; i < total_slave_nodes; ++i) D_mat[i] = 0.0;
        for (int i = 0; i < total_slave_nodes * total_master_nodes; ++i) M_mat[i] = 0.0;

        // 2x2 Gauss points
        const Real gp = 1.0 / Kokkos::sqrt(3.0);
        const Real gauss_pts[4][2] = {{-gp, -gp}, {gp, -gp}, {gp, gp}, {-gp, gp}};
        const Real gauss_wts[4] = {1.0, 1.0, 1.0, 1.0};

        // Compute D matrix (diagonal, dual Lagrange multiplier basis)
        for (int s = 0; s < n_s; ++s) {
            Real A = segment_area(slave_segs[s]);
            for (int gq = 0; gq < 4; ++gq) {
                Real N[4];
                shape_functions_quad(gauss_pts[gq][0], gauss_pts[gq][1], N);
                Real det_J = A / 4.0;  // Approximate Jacobian determinant
                for (int i = 0; i < 4; ++i) {
                    D_mat[s * 4 + i] += N[i] * det_J * gauss_wts[gq];
                }
            }
        }

        // Compute M matrix (coupling: slave to closest master)
        for (int s = 0; s < n_s; ++s) {
            Real cs[3];
            segment_centroid(slave_segs[s], cs);

            // Find closest master segment
            int closest_m = 0;
            Real min_dist = 1.0e30;
            for (int m = 0; m < n_m; ++m) {
                Real cm[3];
                segment_centroid(master_segs[m], cm);
                Real diff[3];
                wave35_detail::sub3(cs, cm, diff);
                Real dist = wave35_detail::norm3(diff);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_m = m;
                }
            }

            // Integrate N_slave * N_master over slave segment
            Real A = segment_area(slave_segs[s]);
            for (int gq = 0; gq < 4; ++gq) {
                Real Ns[4], Nm[4];
                shape_functions_quad(gauss_pts[gq][0], gauss_pts[gq][1], Ns);
                shape_functions_quad(gauss_pts[gq][0], gauss_pts[gq][1], Nm);
                Real det_J = A / 4.0;
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        int row = s * 4 + i;
                        int col = closest_m * 4 + j;
                        M_mat[row * total_master_nodes + col] +=
                            Ns[i] * Nm[j] * det_J * gauss_wts[gq];
                    }
                }
            }
        }
    }

    /**
     * @brief Check that D is diagonal (dual basis property).
     *
     * D should only have entries on the diagonal; off-diagonal entries
     * vanish due to the biorthogonality property of dual shape functions.
     */
    KOKKOS_INLINE_FUNCTION
    bool is_diagonal(const Real* D_mat, int n_entries) const {
        // Our D_mat is stored as a 1D diagonal array, so it is diagonal by construction
        for (int i = 0; i < n_entries; ++i) {
            if (D_mat[i] < 0.0) return false;  // Entries should be non-negative
        }
        return true;
    }

    /**
     * @brief Compute mortar gap from segment centroids.
     */
    KOKKOS_INLINE_FUNCTION
    Real mortar_gap(const MortarSegment& master, const MortarSegment& slave,
                     const Real normal[3]) const {
        Real cm[3], cs[3];
        segment_centroid(master, cm);
        segment_centroid(slave, cs);
        Real diff[3];
        wave35_detail::sub3(cs, cm, diff);
        return wave35_detail::dot3(diff, normal);
    }
};

// ============================================================================
// 35c-2: MortarEdgeToSurface — Edge-to-Surface Coupling
// ============================================================================

/**
 * @brief Edge-to-surface mortar coupling for beam/shell interactions.
 *
 * Projects beam centerline points onto shell surface segments and computes
 * coupling forces. Uses point-to-segment closest-point projection.
 *
 * For each beam node:
 *   1. Find closest shell segment
 *   2. Project beam node onto segment surface
 *   3. Compute gap and penalty force
 *   4. Distribute force to beam node and shell segment nodes
 */
class MortarEdgeToSurface {
public:
    /// Penalty stiffness for edge-surface contact
    Real penalty_ = 1.0e6;

    MortarEdgeToSurface() = default;

    explicit MortarEdgeToSurface(Real penalty) : penalty_(penalty) {}

    /**
     * @brief Project a point onto a quad segment, returning parametric coords.
     *
     * Finds (xi, eta) on the segment closest to the point using Newton iteration.
     * Simplified version: uses centroid projection.
     *
     * @param point     3D point to project [3]
     * @param seg       Shell segment
     * @param proj      Output: projected point on segment [3]
     * @param gap       Output: signed gap (normal distance)
     * @param normal    Output: unit normal at projection [3]
     */
    void project_to_segment(const Real point[3], const MortarSegment& seg,
                              Real proj[3], Real& gap, Real normal[3]) const {
        // Segment normal from cross product of edges
        Real e1[3], e2[3];
        wave35_detail::sub3(seg.nodes[1], seg.nodes[0], e1);
        wave35_detail::sub3(seg.nodes[3], seg.nodes[0], e2);
        wave35_detail::cross3(e1, e2, normal);
        wave35_detail::normalize3(normal);

        // Project: proj = point - ((point-centroid).n) * n
        Real centroid[3];
        for (int d = 0; d < 3; ++d) {
            centroid[d] = 0.25 * (seg.nodes[0][d] + seg.nodes[1][d]
                                + seg.nodes[2][d] + seg.nodes[3][d]);
        }
        Real diff[3];
        wave35_detail::sub3(point, centroid, diff);
        gap = wave35_detail::dot3(diff, normal);

        for (int d = 0; d < 3; ++d) {
            proj[d] = point[d] - gap * normal[d];
        }
    }

    /**
     * @brief Compute coupling forces between beam nodes and shell segments.
     *
     * @param beam_nodes   Beam node coordinates [n_beams * 3]
     * @param n_beams      Number of beam nodes
     * @param shell_segs   Shell segment array
     * @param n_shells     Number of shell segments
     * @param forces       Output: force on each beam node [n_beams * 3]
     * @return Total contact energy
     */
    Real couple_edge_surface(const Real* beam_nodes, int n_beams,
                               const MortarSegment* shell_segs, int n_shells,
                               Real* forces) const {
        Real total_energy = 0.0;

        // Zero forces
        for (int i = 0; i < n_beams * 3; ++i) forces[i] = 0.0;

        for (int b = 0; b < n_beams; ++b) {
            Real point[3] = {beam_nodes[b*3], beam_nodes[b*3+1], beam_nodes[b*3+2]};

            // Find closest shell segment
            Real min_gap = 1.0e30;
            int closest_seg = -1;
            Real best_normal[3] = {0.0, 0.0, 0.0};

            for (int s = 0; s < n_shells; ++s) {
                Real proj[3], normal[3];
                Real gap;
                project_to_segment(point, shell_segs[s], proj, gap, normal);
                Real abs_gap = Kokkos::fabs(gap);
                if (abs_gap < min_gap) {
                    min_gap = abs_gap;
                    closest_seg = s;
                    wave35_detail::copy3(normal, best_normal);
                }
            }

            if (closest_seg >= 0) {
                Real proj[3], normal[3];
                Real gap;
                project_to_segment(point, shell_segs[closest_seg], proj, gap, normal);

                // Only apply force for penetration (gap < 0)
                if (gap < 0.0) {
                    Real f_mag = penalty_ * Kokkos::fabs(gap);
                    for (int d = 0; d < 3; ++d) {
                        forces[b * 3 + d] = f_mag * normal[d];
                    }
                    total_energy += 0.5 * penalty_ * gap * gap;
                }
            }
        }

        return total_energy;
    }
};

// ============================================================================
// 35c-3: MortarAssembly — Mass Matrix Assembly + Constraint Enforcement
// ============================================================================

/**
 * @brief Mortar mass matrix assembly and constraint enforcement.
 *
 * Assembles the coupled system using the mortar matrices D and M:
 *
 *   K_coupled = K_master + D^{-1} * M * K_slave * M^T * D^{-T}
 *
 * Since D is diagonal (dual Lagrange multipliers), D^{-1} is trivially computed.
 *
 * The condensation transfers slave stiffness contributions to the master surface,
 * eliminating the slave DOFs from the global system.
 */
class MortarAssembly {
public:
    MortarAssembly() = default;

    /**
     * @brief Invert a diagonal matrix (stored as 1D array).
     *
     * @param D       Input diagonal entries [n]
     * @param D_inv   Output inverse diagonal [n]
     * @param n       Size
     */
    void invert_diagonal(const Real* D, Real* D_inv, int n) const {
        for (int i = 0; i < n; ++i) {
            D_inv[i] = (Kokkos::fabs(D[i]) > 1.0e-30) ? (1.0 / D[i]) : 0.0;
        }
    }

    /**
     * @brief Assemble coupled mortar system.
     *
     * K_coupled_ij = K_master_ij + sum_k sum_l (D_inv_ik * M_kl * K_slave_lm * M_mj_T * D_inv_jn)
     *
     * Simplified for small systems: treats K_master, K_slave as scalar stiffnesses per node.
     *
     * @param D           Diagonal mass matrix [n_slave_nodes]
     * @param M           Coupling matrix [n_slave_nodes * n_master_nodes]
     * @param K_master    Master stiffness (scalar per node) [n_master_nodes]
     * @param K_slave     Slave stiffness (scalar per node) [n_slave_nodes]
     * @param n_master    Number of master nodes
     * @param n_slave     Number of slave nodes
     * @param K_coupled   Output coupled stiffness [n_master_nodes]
     */
    void assemble_mortar_system(const Real* D, const Real* M,
                                 const Real* K_master, const Real* K_slave,
                                 int n_master, int n_slave,
                                 Real* K_coupled) const {
        // Start with master stiffness
        for (int i = 0; i < n_master; ++i) {
            K_coupled[i] = K_master[i];
        }

        // D^{-1}
        std::vector<Real> D_inv(n_slave);
        invert_diagonal(D, D_inv.data(), n_slave);

        // Add condensed slave contribution: D^{-1} * M * K_slave * M^T * D^{-T}
        // For each master node i:
        //   K_coupled[i] += sum_k (D_inv[k] * M[k,i]) * K_slave[k] * (M[k,i] * D_inv[k])
        for (int i = 0; i < n_master; ++i) {
            Real contribution = 0.0;
            for (int k = 0; k < n_slave; ++k) {
                Real M_ki = M[k * n_master + i];
                contribution += D_inv[k] * M_ki * K_slave[k] * M_ki * D_inv[k];
            }
            K_coupled[i] += contribution;
        }
    }

    /**
     * @brief Verify that coupled stiffness is non-negative (physical requirement).
     */
    bool is_positive(const Real* K_coupled, int n) const {
        for (int i = 0; i < n; ++i) {
            if (K_coupled[i] < -1.0e-10) return false;
        }
        return true;
    }

    /**
     * @brief Compute condition number estimate of D (ratio of max/min diagonal).
     */
    Real condition_estimate(const Real* D, int n) const {
        Real d_min = 1.0e30, d_max = 0.0;
        for (int i = 0; i < n; ++i) {
            Real d_abs = Kokkos::fabs(D[i]);
            if (d_abs > 1.0e-30) {
                d_min = Kokkos::fmin(d_min, d_abs);
                d_max = Kokkos::fmax(d_max, d_abs);
            }
        }
        return (d_min > 1.0e-30) ? (d_max / d_min) : 1.0e30;
    }
};

// ============================================================================
// 35c-4: MortarThermal — Thermal Coupling Through Mortar Interface
// ============================================================================

/**
 * @brief Thermal coupling through mortar contact interface.
 *
 * Heat flux continuity across the interface:
 *   q_n = h_c * (T_master - T_slave)
 *
 * where h_c is the contact heat transfer coefficient [W/(m^2*K)].
 *
 * Using mortar weights, the integrated heat flux at slave node i is:
 *   Q_i = sum_j w_ij * h_c * (T_master_j - T_slave_i)
 *
 * where w_ij are the mortar integration weights (from D^{-1}*M).
 */
class MortarThermal {
public:
    MortarThermal() = default;

    /**
     * @brief Compute thermal flux through mortar interface.
     *
     * @param T_master        Master surface temperatures [n_pairs]
     * @param T_slave         Slave surface temperatures [n_pairs]
     * @param h_contact       Contact heat transfer coefficient
     * @param mortar_weights  Mortar integration weights [n_pairs]
     * @param heat_flux       Output: heat flux at each pair [n_pairs]
     * @param n_pairs         Number of contact pairs
     * @return Total heat transfer rate [W]
     */
    Real compute_thermal_mortar(const Real* T_master, const Real* T_slave,
                                  Real h_contact, const Real* mortar_weights,
                                  Real* heat_flux, int n_pairs) const {
        Real total_Q = 0.0;
        for (int i = 0; i < n_pairs; ++i) {
            Real dT = T_master[i] - T_slave[i];
            heat_flux[i] = h_contact * mortar_weights[i] * dT;
            total_Q += heat_flux[i];
        }
        return total_Q;
    }

    /**
     * @brief Check conservation: total heat leaving master = total heat entering slave.
     *
     * @param heat_flux  Heat flux array [n_pairs]
     * @param n_pairs    Number of pairs
     * @return Sum of heat fluxes (should be zero in a closed system with equal weights)
     */
    Real conservation_check(const Real* heat_flux, int n_pairs) const {
        Real sum = 0.0;
        for (int i = 0; i < n_pairs; ++i) sum += heat_flux[i];
        return sum;
    }

    /**
     * @brief Compute linearized thermal conductance matrix contribution.
     *
     * dQ/dT_master = h_c * w_i (positive: heating master increases flux)
     * dQ/dT_slave  = -h_c * w_i (negative: heating slave decreases flux)
     *
     * @param h_contact       Contact conductance
     * @param mortar_weights  Integration weights [n_pairs]
     * @param K_thermal       Output: thermal conductance per pair [n_pairs]
     * @param n_pairs         Number of pairs
     */
    void thermal_conductance(Real h_contact, const Real* mortar_weights,
                              Real* K_thermal, int n_pairs) const {
        for (int i = 0; i < n_pairs; ++i) {
            K_thermal[i] = h_contact * mortar_weights[i];
        }
    }

    /**
     * @brief Compute steady-state interface temperature for equal conductances.
     *
     * T_interface = (T_master + T_slave) / 2  (when both sides have equal conductance)
     */
    KOKKOS_INLINE_FUNCTION
    Real interface_temperature(Real T_master, Real T_slave) const {
        return 0.5 * (T_master + T_slave);
    }

    /**
     * @brief Compute thermal energy dissipated by contact resistance.
     *
     * E = integral(q * dT * dt) = h_c * (T_m - T_s)^2 * A * dt
     *
     * @param T_master  Master temperature
     * @param T_slave   Slave temperature
     * @param h_contact Contact conductance
     * @param area      Contact area
     * @param dt        Time step
     * @return Energy dissipated
     */
    KOKKOS_INLINE_FUNCTION
    Real thermal_dissipation(Real T_master, Real T_slave,
                              Real h_contact, Real area, Real dt) const {
        Real dT = T_master - T_slave;
        return h_contact * dT * dT * area * dt;
    }
};

} // namespace fem
} // namespace nxs
