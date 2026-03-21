#pragma once

/**
 * @file shell_wave43.hpp
 * @brief Wave 43: Warped shell element corrections for real-world meshes.
 *
 * Components:
 *  1. WarpDetector              - Warpage metrics for quadrilateral elements
 *  2. WarpedShellCorrector      - Stiffness correction for warped quads
 *  3. DrillingDOFStabilization  - Drilling rotation (θz) stiffness
 *  4. HourglassControl          - Multiple hourglass stabilization modes
 *  5. ShellThicknessUpdate      - Thickness update for large deformation
 *
 * References:
 *  - Belytschko, Liu, Moran (2000) Nonlinear Finite Elements
 *  - Felippa (2003) A study of optimal membrane triangles with drilling freedoms
 *  - Flanagan, Belytschko (1981) uniform strain hexahedron and quadrilateral
 */

#include <cmath>
#include <cstring>
#include <array>
#include <algorithm>

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

namespace nxs {

using Real = double;

namespace fem {

// ============================================================================
// Internal math helpers — GPU-safe, no STL containers
// ============================================================================

namespace wave43_shell_detail {

KOKKOS_INLINE_FUNCTION
void zero_n(Real* a, int n) {
    for (int i = 0; i < n; ++i) a[i] = 0.0;
}

KOKKOS_INLINE_FUNCTION
Real dot3(const Real* a, const Real* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

KOKKOS_INLINE_FUNCTION
void cross3(const Real* a, const Real* b, Real* c) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

KOKKOS_INLINE_FUNCTION
Real norm3(const Real* v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

KOKKOS_INLINE_FUNCTION
void normalize3(Real* v) {
    Real n = norm3(v);
    if (n > 1.0e-30) { v[0] /= n; v[1] /= n; v[2] /= n; }
}

/// Edge vector from node i to node j of the quad (coords[4][3])
KOKKOS_INLINE_FUNCTION
void edge(const Real coords[4][3], int i, int j, Real* e) {
    e[0] = coords[j][0] - coords[i][0];
    e[1] = coords[j][1] - coords[i][1];
    e[2] = coords[j][2] - coords[i][2];
}

} // namespace wave43_shell_detail

// ============================================================================
// 1. WarpDetector
// ============================================================================

/**
 * @brief Compute warpage and quality metrics for a 4-node quad element.
 *
 * All functions accept coords[4][3] — four nodes, each with (x, y, z).
 */
struct WarpDetector {

    /**
     * @brief Angle (radians) between the normals of the two triangles formed
     *        by splitting the quad along diagonal 0-2.
     *
     *   Triangle A: nodes 0, 1, 2  (normal n_A)
     *   Triangle B: nodes 0, 2, 3  (normal n_B)
     *   warp_angle = acos( clamp(n_A · n_B, -1, 1) )
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_warp_angle(const Real coords[4][3]) {
        using namespace wave43_shell_detail;

        // Triangle A: 0-1-2
        Real a1[3], a2[3], nA[3];
        edge(coords, 0, 1, a1);
        edge(coords, 0, 2, a2);
        cross3(a1, a2, nA);
        normalize3(nA);

        // Triangle B: 0-2-3
        Real b1[3], b2[3], nB[3];
        edge(coords, 0, 2, b1);
        edge(coords, 0, 3, b2);
        cross3(b1, b2, nB);
        normalize3(nB);

        Real cosA = dot3(nA, nB);
        // Clamp to [-1, 1] for numerical safety
        if (cosA >  1.0) cosA =  1.0;
        if (cosA < -1.0) cosA = -1.0;
        return std::acos(cosA);
    }

    /**
     * @brief Max edge length divided by min edge length.
     *
     *  Edges: 0-1, 1-2, 2-3, 3-0
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_aspect_ratio(const Real coords[4][3]) {
        using namespace wave43_shell_detail;

        const int pairs[4][2] = {{0,1},{1,2},{2,3},{3,0}};
        Real emin = 1.0e30, emax = 0.0;
        for (int k = 0; k < 4; ++k) {
            Real ev[3];
            edge(coords, pairs[k][0], pairs[k][1], ev);
            Real len = norm3(ev);
            if (len < emin) emin = len;
            if (len > emax) emax = len;
        }
        if (emin < 1.0e-30) return 1.0e30;
        return emax / emin;
    }

    /**
     * @brief Deviation from 90° of element diagonals (radians).
     *
     *  Diagonals: 0-2 and 1-3.
     *  skew = |π/2 - acos(|d1 · d2| / (|d1| |d2|))|
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_skew_angle(const Real coords[4][3]) {
        using namespace wave43_shell_detail;

        Real d1[3], d2[3];
        edge(coords, 0, 2, d1);
        edge(coords, 1, 3, d2);

        Real n1 = norm3(d1), n2 = norm3(d2);
        if (n1 < 1.0e-30 || n2 < 1.0e-30) return 0.0;

        Real cosA = dot3(d1, d2) / (n1 * n2);
        if (cosA >  1.0) cosA =  1.0;
        if (cosA < -1.0) cosA = -1.0;
        // Angle between diagonals
        Real angle = std::acos(cosA);
        // Skew = deviation from π/2
        return std::abs(angle - 0.5 * M_PI);
    }

    /**
     * @brief Return true when warp_angle exceeds threshold.
     *
     * @param warp_angle  Warp angle in radians (from compute_warp_angle).
     * @param threshold   Default 0.1745 rad ≈ 10°.
     */
    KOKKOS_INLINE_FUNCTION
    static bool is_severely_warped(Real warp_angle, Real threshold = 0.1745) {
        return warp_angle > threshold;
    }
};

// ============================================================================
// 2. WarpedShellCorrector
// ============================================================================

/**
 * @brief Correct a flat-shell stiffness matrix for element warping.
 *
 * Strategy:
 *   - Compute a best-fit local frame (e1, e2, e3) for the warped quad.
 *   - Project nodes to the local 2D plane.
 *   - Add membrane-bending coupling proportional to warp_angle to K_flat,
 *     producing K_corrected.
 *
 * DOF ordering: [ux, uy, uz, θx, θy, θz] per node, 4 nodes → 24 DOFs.
 * Membrane DOFs: ux(0), uy(1) rows/cols per node.
 * Bending DOFs:  uz(2), θx(3), θy(4) rows/cols per node.
 */
struct WarpedShellCorrector {

    /**
     * @brief Compute a best-fit local coordinate frame for a warped quad.
     *
     *  e3 = average of the two triangle normals (normalised).
     *  e1 = projection of (node1 - node0) onto the plane perpendicular to e3.
     *  e2 = e3 × e1.
     *
     * @param coords  Node coordinates coords[4][3]
     * @param e1      Output: in-plane x-axis (3 components)
     * @param e2      Output: in-plane y-axis (3 components)
     * @param e3      Output: shell normal   (3 components)
     */
    KOKKOS_INLINE_FUNCTION
    static void compute_local_frame(const Real coords[4][3],
                                    Real e1[3], Real e2[3], Real e3[3]) {
        using namespace wave43_shell_detail;

        // Normal of triangle A: 0-1-2
        Real a1[3], a2[3], nA[3];
        edge(coords, 0, 1, a1);
        edge(coords, 0, 2, a2);
        cross3(a1, a2, nA);

        // Normal of triangle B: 0-2-3
        Real b1[3], b2[3], nB[3];
        edge(coords, 0, 2, b1);
        edge(coords, 0, 3, b2);
        cross3(b1, b2, nB);

        // Average normal
        e3[0] = nA[0] + nB[0];
        e3[1] = nA[1] + nB[1];
        e3[2] = nA[2] + nB[2];
        normalize3(e3);

        // e1 = (node1 - node0) projected onto plane of e3
        Real v01[3];
        edge(coords, 0, 1, v01);
        Real proj = dot3(v01, e3);
        e1[0] = v01[0] - proj * e3[0];
        e1[1] = v01[1] - proj * e3[1];
        e1[2] = v01[2] - proj * e3[2];
        normalize3(e1);

        // e2 = e3 x e1
        cross3(e3, e1, e2);
        normalize3(e2);
    }

    /**
     * @brief Project 3D node coordinates onto the local 2D frame.
     *
     * @param coords        Node coordinates coords[4][3] (global 3D)
     * @param local_coords  Output: local_coords[4][2] in (e1, e2) plane
     * @param e1            Local x-axis
     * @param e2            Local y-axis
     * @param e3            Shell normal (used to get origin offset only)
     */
    KOKKOS_INLINE_FUNCTION
    static void project_to_local(const Real coords[4][3],
                                 Real local_coords[4][2],
                                 const Real e1[3], const Real e2[3],
                                 const Real /*e3*/[3]) {
        using namespace wave43_shell_detail;
        // Use node 0 as origin
        const Real* origin = coords[0];
        for (int n = 0; n < 4; ++n) {
            Real dv[3] = { coords[n][0] - origin[0],
                           coords[n][1] - origin[1],
                           coords[n][2] - origin[2] };
            local_coords[n][0] = dot3(dv, e1);
            local_coords[n][1] = dot3(dv, e2);
        }
    }

    /**
     * @brief Apply warp correction to a 24x24 flat-shell stiffness matrix.
     *
     * Method: Add membrane-bending coupling terms proportional to sin(warp_angle)
     * at off-diagonal blocks (DOFs 0-1 coupled to DOFs 2-4 for each node pair).
     * The coupling magnitude is scaled by K_flat diagonal average to keep the
     * correction relative.
     *
     * @param warp_angle   Warp angle in radians
     * @param K_flat       Input: flat-shell 24x24 stiffness (row-major)
     * @param K_corrected  Output: corrected 24x24 stiffness (row-major)
     */
    KOKKOS_INLINE_FUNCTION
    static void compute_warp_correction_matrix(Real warp_angle,
                                               const Real K_flat[24*24],
                                               Real K_corrected[24*24]) {
        // Copy flat stiffness
        for (int i = 0; i < 24*24; ++i) K_corrected[i] = K_flat[i];

        Real coupling = std::sin(warp_angle);
        if (coupling < 1.0e-12) return;  // Flat element — no correction needed

        // Estimate average diagonal magnitude for relative scaling
        Real diag_avg = 0.0;
        for (int i = 0; i < 24; ++i) diag_avg += K_flat[i*24 + i];
        diag_avg /= 24.0;
        if (std::abs(diag_avg) < 1.0e-30) return;

        Real scale = coupling * std::abs(diag_avg);

        // For each node, add coupling between membrane DOFs (0,1) and bending DOFs (2,3,4)
        // DOF layout per node n: 6*n + [0=ux, 1=uy, 2=uz, 3=θx, 4=θy, 5=θz]
        for (int n = 0; n < 4; ++n) {
            int mem[2] = { 6*n + 0, 6*n + 1 };    // membrane
            int ben[3] = { 6*n + 2, 6*n + 3, 6*n + 4 };  // bending

            for (int mi = 0; mi < 2; ++mi) {
                for (int bi = 0; bi < 3; ++bi) {
                    int row = mem[mi], col = ben[bi];
                    // Symmetric coupling
                    K_corrected[row*24 + col] += scale * 0.5;
                    K_corrected[col*24 + row] += scale * 0.5;
                }
            }
        }
    }
};

// ============================================================================
// 3. DrillingDOFStabilization
// ============================================================================

/**
 * @brief Add drilling rotation (θz) stiffness to a shell stiffness matrix.
 *
 * Without drilling stiffness, a shell formulation has a singular stiffness
 * matrix for θz (rotation about the shell normal). A small artificial
 * stiffness stabilises the system without affecting physical results.
 *
 * k_drill = alpha * E * thickness * area,   alpha ~ 1e-3.
 * DOF ordering: [ux, uy, uz, θx, θy, θz] per node → θz is DOF index 5.
 */
struct DrillingDOFStabilization {

    /// Recommended alpha coefficient for drilling stiffness
    static constexpr Real ALPHA_DRILL = 1.0e-3;

    /**
     * @brief Compute the drilling stiffness coefficient.
     *
     * @param E          Young's modulus
     * @param thickness  Shell thickness
     * @param area       Element area
     * @return           k_drill = alpha * E * thickness * area
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_drilling_stiffness(Real E, Real thickness, Real area) {
        return ALPHA_DRILL * E * thickness * area;
    }

    /**
     * @brief Add k_drill to the diagonal θz entries of the stiffness matrix.
     *
     * θz is DOF 5 within each node's 6-DOF block.
     * For node n, the global DOF index is 6*n + 5.
     *
     * @param K         Stiffness matrix (num_nodes*6 × num_nodes*6, row-major)
     * @param k_drill   Drilling stiffness value
     * @param num_nodes Number of nodes (default 4 → 24×24 K)
     */
    KOKKOS_INLINE_FUNCTION
    static void add_drilling_to_stiffness(Real* K, Real k_drill, int num_nodes = 4) {
        int ndof = num_nodes * 6;
        for (int n = 0; n < num_nodes; ++n) {
            int dof_z = 6*n + 5;  // θz index
            K[dof_z * ndof + dof_z] += k_drill;
        }
    }
};

// ============================================================================
// 4. HourglassControl
// ============================================================================

/**
 * @brief Hourglass stabilization modes for under-integrated shell elements.
 */
enum class HourglassType {
    FlanaganBelytschko,  ///< F-B algorithm: F_hg = coeff * (Γ · v) * Γ
    Physical,            ///< Physical stiffness stabilization (material-based)
    Viscous,             ///< Velocity-based viscous damping of hourglass modes
    Stiffness            ///< Stiffness-based, adds to K rather than F
};

/**
 * @brief Parameters controlling hourglass stabilization.
 */
struct HourglassParams {
    Real coefficient = 0.03;   ///< F-B coefficient (qhg, typically 0.03–0.10)
    Real viscous_coeff = 0.05; ///< Viscous damping coefficient
    Real stiffness_coeff = 0.1; ///< Stiffness mode coefficient
    Real E = 210.0e9;          ///< Young's modulus (for physical mode)
    Real thickness = 0.01;     ///< Shell thickness (for physical mode)
};

/**
 * @brief Compute hourglass stabilization forces for a 4-node shell element.
 *
 * The element has 4 nodes × 6 DOFs = 24 DOFs.  Velocities and forces are
 * ordered [vx, vy, vz, ωx, ωy, ωz] per node.
 *
 * Hourglass mode vector for translation DOFs (Flanagan-Belytschko):
 *   Γ = [1, -1, 1, -1]  (node-ordered scalar selector for each translation)
 *
 * For each of the three translational directions (x, y, z):
 *   q_hg = Γ · v_dir   (hourglass participation)
 *   F_hg_dir = coeff * q_hg * Γ  (stabilisation force on each node)
 */
struct HourglassControl {

    /**
     * @brief Compute anti-hourglass nodal forces.
     *
     * @param type        Hourglass algorithm selection
     * @param coords      Node coordinates coords[4][3]
     * @param velocities  Nodal velocities vel[4][6] = [vx,vy,vz,ωx,ωy,ωz]
     * @param params      Algorithm parameters
     * @param forces      Output: nodal forces forces[24] (same DOF ordering)
     */
    KOKKOS_INLINE_FUNCTION
    static void compute_hourglass_forces(HourglassType type,
                                         const Real coords[4][3],
                                         const Real velocities[4][6],
                                         const HourglassParams& params,
                                         Real forces[24]) {
        using namespace wave43_shell_detail;
        zero_n(forces, 24);

        // Hourglass mode vector (translational, sign pattern for bilinear quad)
        const Real gamma[4] = { 1.0, -1.0, 1.0, -1.0 };

        // Estimate element area for scaling (cross of diagonals)
        Real d1[3], d2[3], cr[3];
        edge(coords, 0, 2, d1);
        edge(coords, 1, 3, d2);
        cross3(d1, d2, cr);
        Real area = 0.5 * norm3(cr);

        switch (type) {

            case HourglassType::FlanaganBelytschko: {
                // F-B: for each translational direction d,
                //   q  = sum_i( Γ_i * v_i_d )
                //   f_hg_id = coeff * q * Γ_i
                for (int d = 0; d < 3; ++d) {
                    Real q = 0.0;
                    for (int n = 0; n < 4; ++n) q += gamma[n] * velocities[n][d];
                    for (int n = 0; n < 4; ++n)
                        forces[6*n + d] += params.coefficient * q * gamma[n];
                }
                break;
            }

            case HourglassType::Physical: {
                // Physical stabilization: scale by E*t^3/area for bending mode
                Real k_physical = params.E * params.thickness * params.thickness
                                * params.thickness / (12.0 * area + 1.0e-30);
                for (int d = 0; d < 3; ++d) {
                    Real q = 0.0;
                    for (int n = 0; n < 4; ++n) q += gamma[n] * velocities[n][d];
                    for (int n = 0; n < 4; ++n)
                        forces[6*n + d] += params.coefficient * k_physical * q * gamma[n];
                }
                break;
            }

            case HourglassType::Viscous: {
                // Velocity-dependent viscous damping of hourglass modes
                for (int d = 0; d < 3; ++d) {
                    Real q = 0.0;
                    for (int n = 0; n < 4; ++n) q += gamma[n] * velocities[n][d];
                    // Viscous force proportional to q and element area
                    Real scale = params.viscous_coeff * area;
                    for (int n = 0; n < 4; ++n)
                        forces[6*n + d] += scale * q * gamma[n];
                }
                break;
            }

            case HourglassType::Stiffness: {
                // Stiffness mode: use a stiffness-like coefficient (no velocity)
                // In this version we treat velocities as displacements for the
                // stiffness-mode hourglass (static or linearised dynamic).
                for (int d = 0; d < 3; ++d) {
                    Real q = 0.0;
                    for (int n = 0; n < 4; ++n) q += gamma[n] * velocities[n][d];
                    Real scale = params.stiffness_coeff * area * params.E * params.thickness;
                    for (int n = 0; n < 4; ++n)
                        forces[6*n + d] += scale * q * gamma[n];
                }
                break;
            }
        }
    }
};

// ============================================================================
// 5. ShellThicknessUpdate
// ============================================================================

/**
 * @brief Update shell thickness for large-deformation analyses.
 *
 * In shells undergoing finite strains the through-thickness stretch F_33
 * changes the current thickness. Two methods are provided:
 *   1. Direct: t = t0 * F_33
 *   2. Incompressible volumetric: F_33 = A0 / A_current (area ratio)
 */
struct ShellThicknessUpdate {

    /**
     * @brief Update thickness from through-thickness stretch component.
     *
     * @param t0   Reference (initial) thickness
     * @param F_33 Through-thickness deformation gradient component
     * @return     Current thickness t = t0 * F_33
     */
    KOKKOS_INLINE_FUNCTION
    static Real update_thickness(Real t0, Real F_33) {
        return t0 * F_33;
    }

    /**
     * @brief Compute through-thickness stretch from area change (incompressible).
     *
     * For an incompressible material, J = 1 → F_33 = (A0 / A_current).
     *
     * @param area_0        Reference element area
     * @param area_current  Current element area
     * @param incompressible If true, use area ratio formula (default: true)
     * @return              F_33
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_thickness_stretch(Real area_0, Real area_current,
                                          bool incompressible = true) {
        if (!incompressible) return 1.0;
        if (area_current < 1.0e-30) return 1.0;
        return area_0 / area_current;
    }
};

} // namespace fem
} // namespace nxs
