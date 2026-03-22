#pragma once

/**
 * @file xfem_wave36.hpp
 * @brief Wave 36: XFEM Completion - 3D crack propagation, shell layers, enrichment
 *
 * Components:
 * 7.  XFEMCrackPropagation3D  - 3D crack front advancement via level sets
 * 8.  XFEMLayerAdvection      - Shell layer-by-layer crack tracking
 * 9.  XFEMEnrichment36        - Heaviside + tip enrichment with blending
 * 10. XFEMForceIntegration    - Sub-element integration for split elements
 * 11. XFEMCrackDirection      - Crack growth direction criteria (MTS, energy)
 * 12. XFEMVelocityUpdate      - Enriched DOF velocity for explicit dynamics
 *
 * References:
 * - Moes, Dolbow, Belytschko (1999) "A finite element method for crack growth"
 * - Stolarska et al. (2001) "Modelling crack growth by level sets in XFEM"
 * - Fries & Belytschko (2010) "The extended/generalized finite element method"
 * - Erdogan & Sih (1963) "On the crack extension in plates under plane loading"
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>

namespace nxs {
namespace fem {

using Real = nxs::Real;

// ============================================================================
// Utility functions for XFEM
// ============================================================================

namespace xfem_detail {

inline Real dot3(const Real* a, const Real* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline Real norm3(const Real* v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

inline void normalize3(Real* v) {
    Real n = norm3(v);
    if (n > 1.0e-30) { v[0] /= n; v[1] /= n; v[2] /= n; }
}

inline void cross3(const Real* a, const Real* b, Real* c) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

inline Real sign(Real x) {
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    return 0.0;
}

/// Linear interpolation of level set along an edge to find zero crossing
inline Real find_zero_crossing(Real phi_a, Real phi_b) {
    Real denom = phi_a - phi_b;
    if (std::abs(denom) < 1.0e-30) return 0.5;
    return phi_a / denom;  // parameter t in [0,1] where phi = 0
}

/// Compute area of triangle given 3 vertices in 2D
inline Real triangle_area_2d(Real x0, Real y0, Real x1, Real y1, Real x2, Real y2) {
    return 0.5 * std::abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
}

/// Compute area of triangle given 3 vertices in 3D
inline Real triangle_area_3d(const Real* p0, const Real* p1, const Real* p2) {
    Real u[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
    Real v[3] = {p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]};
    Real c[3];
    cross3(u, v, c);
    return 0.5 * norm3(c);
}

} // namespace xfem_detail


// ============================================================================
// 7. XFEMCrackPropagation3D - 3D crack front advancement
// ============================================================================

/**
 * @brief 3D node data for crack propagation.
 */
struct CrackNode3D {
    Real x, y, z;         ///< Position
    Real phi;             ///< Level set: signed distance to crack surface
    Real psi;             ///< Level set: signed distance to crack front
};

/**
 * @brief 3D crack front advancement via level set update.
 *
 * The crack is represented by two level set functions:
 * - phi: signed distance to the crack surface. phi=0 defines the crack plane.
 *   Nodes with phi < 0 are on one side, phi > 0 on the other.
 * - psi: signed distance to the crack front (tip in 3D). psi=0 is the front.
 *   The crack exists where phi=0 AND psi < 0.
 *
 * Crack propagation:
 * 1. Determine growth direction at each front node (from SIF computation)
 * 2. Extend the velocity field from the front to the volume
 * 3. Update level sets: phi_new = phi - V_n * dt, psi_new = psi - V_t * dt
 *
 * For simplicity, we use a direct update method:
 *   phi_new(x) = phi(x) if point is far from front
 *   phi_new(x) updated near the new crack surface
 *
 * Reference: Stolarska et al. (2001)
 */
class XFEMCrackPropagation3D {
public:
    XFEMCrackPropagation3D() = default;

    /**
     * @brief Advance the crack front by a given increment.
     *
     * Updates phi and psi level sets for all nodes based on crack
     * growth direction and increment da.
     *
     * Algorithm:
     * 1. Identify front nodes (where |psi| < band_width and phi changes sign nearby)
     * 2. For each node, compute signed distance to new crack surface
     * 3. Update phi: phi_new = min(phi_old, distance to new crack plane)
     * 4. Update psi: advance front by da along the growth direction
     *
     * @param nodes Array of 3D crack nodes (modified in place)
     * @param nnodes Number of nodes
     * @param direction Growth direction (unit vector, 3 components)
     * @param da Crack growth increment
     * @param front_point Crack front reference point (3 components)
     * @param front_tangent Crack front tangent direction (3 components)
     */
    void advance_crack(CrackNode3D* nodes, int nnodes,
                       const Real* direction, Real da,
                       const Real* front_point,
                       const Real* front_tangent) const {
        if (nnodes <= 0 || da <= 0.0) return;

        // Compute crack plane normal (direction of phi gradient)
        // The new crack surface extends in the growth direction from the front.
        // Normal to new crack surface = cross(direction, front_tangent)
        Real normal[3];
        xfem_detail::cross3(direction, front_tangent, normal);
        xfem_detail::normalize3(normal);

        // If normal is degenerate, use direction as the crack plane update
        Real nn = xfem_detail::norm3(normal);
        if (nn < 0.1) {
            normal[0] = direction[0];
            normal[1] = direction[1];
            normal[2] = direction[2];
        }

        // New front point = old front + da * direction
        Real new_front[3] = {
            front_point[0] + da * direction[0],
            front_point[1] + da * direction[1],
            front_point[2] + da * direction[2]
        };

        for (int i = 0; i < nnodes; ++i) {
            // Vector from node to old front
            Real dx_old = nodes[i].x - front_point[0];
            Real dy_old = nodes[i].y - front_point[1];
            Real dz_old = nodes[i].z - front_point[2];
            Real d_old[3] = {dx_old, dy_old, dz_old};

            // Vector from node to new front
            Real dx_new = nodes[i].x - new_front[0];
            Real dy_new = nodes[i].y - new_front[1];
            Real dz_new = nodes[i].z - new_front[2];
            Real d_new[3] = {dx_new, dy_new, dz_new};

            // Distance along growth direction from old front
            Real proj_growth = xfem_detail::dot3(d_old, direction);

            // Distance along front tangent
            Real proj_tangent = xfem_detail::dot3(d_old, front_tangent);

            // Signed distance to crack plane (phi)
            Real phi_new_surface = xfem_detail::dot3(d_old, normal);

            // Update phi: the crack surface extends from the old surface
            // to the new front. If the node is in the swept region,
            // phi should be updated.
            if (proj_growth >= 0.0 && proj_growth <= da) {
                // Node is in the swept region
                // phi = signed distance to the crack plane
                Real phi_candidate = phi_new_surface;
                // Take the minimum absolute value (closest to crack)
                if (std::abs(phi_candidate) < std::abs(nodes[i].phi)) {
                    nodes[i].phi = phi_candidate;
                }
            }

            // Update psi: signed distance to the new front
            // psi < 0 behind the front (cracked), psi > 0 ahead (intact)
            Real psi_new = xfem_detail::dot3(d_new, direction);
            // Only update if new psi is smaller (front has advanced)
            if (psi_new < nodes[i].psi) {
                nodes[i].psi = psi_new;
            }
        }
    }

    /**
     * @brief Identify nodes near the crack front.
     *
     * A node is near the front if |psi| < bandwidth and the element
     * containing it has a phi sign change.
     *
     * @param nodes Node array
     * @param nnodes Number of nodes
     * @param bandwidth Distance threshold for front proximity
     * @param front_nodes Output: indices of front-proximal nodes
     * @return Number of front nodes found
     */
    int find_front_nodes(const CrackNode3D* nodes, int nnodes,
                         Real bandwidth, int* front_nodes) const {
        int count = 0;
        for (int i = 0; i < nnodes; ++i) {
            if (std::abs(nodes[i].psi) < bandwidth) {
                front_nodes[count++] = i;
            }
        }
        return count;
    }

    /**
     * @brief Determine if an element is cut by the crack.
     *
     * An element is cut if phi changes sign across its nodes AND
     * at least one node has psi <= 0 (the crack exists there).
     *
     * @param phi_nodes Level set phi values at element nodes
     * @param psi_nodes Level set psi values at element nodes
     * @param nnodes_per_elem Number of nodes in the element
     * @return true if element is cut
     */
    static bool is_element_cut(const Real* phi_nodes, const Real* psi_nodes,
                                int nnodes_per_elem) {
        bool has_positive = false, has_negative = false;
        bool has_psi_behind = false;

        for (int i = 0; i < nnodes_per_elem; ++i) {
            if (phi_nodes[i] > 0.0) has_positive = true;
            if (phi_nodes[i] < 0.0) has_negative = true;
            if (psi_nodes[i] <= 0.0) has_psi_behind = true;
        }

        return has_positive && has_negative && has_psi_behind;
    }

    /**
     * @brief Check if an element contains the crack tip.
     *
     * The tip element has phi changing sign AND psi changing sign.
     */
    static bool is_tip_element(const Real* phi_nodes, const Real* psi_nodes,
                                int nnodes_per_elem) {
        bool phi_pos = false, phi_neg = false;
        bool psi_pos = false, psi_neg = false;

        for (int i = 0; i < nnodes_per_elem; ++i) {
            if (phi_nodes[i] > 0.0) phi_pos = true;
            if (phi_nodes[i] < 0.0) phi_neg = true;
            if (psi_nodes[i] > 0.0) psi_pos = true;
            if (psi_nodes[i] <= 0.0) psi_neg = true;
        }

        return phi_pos && phi_neg && psi_pos && psi_neg;
    }

    /**
     * @brief Compute the crack opening displacement at a point.
     *
     * COD = |u+ - u-| where u+ and u- are displacements on each side of the crack.
     * Approximated as 2 * |enriched displacement| at the crack surface.
     */
    static Real compute_cod(Real enriched_disp_x, Real enriched_disp_y,
                             Real enriched_disp_z) {
        return 2.0 * std::sqrt(enriched_disp_x * enriched_disp_x +
                                enriched_disp_y * enriched_disp_y +
                                enriched_disp_z * enriched_disp_z);
    }
};


// ============================================================================
// 8. XFEMLayerAdvection - Shell layer-by-layer crack tracking
// ============================================================================

/**
 * @brief Crack state through shell layers.
 */
struct ShellCrackState {
    Real phi_layer[8];  ///< Level set value per layer (up to 8 layers)
    int nlayers;        ///< Number of shell layers
    int cracked_layers; ///< Number of fully cracked layers
    Real total_thickness; ///< Total shell thickness
    Real crack_depth;   ///< Current crack penetration depth

    ShellCrackState() : nlayers(1), cracked_layers(0),
                        total_thickness(1.0), crack_depth(0.0) {
        for (int i = 0; i < 8; ++i) phi_layer[i] = 1.0;
    }
};

/**
 * @brief Shell layer-by-layer enrichment for through-thickness crack propagation.
 *
 * In shell structures, cracks can propagate through the thickness layer by layer.
 * Each layer has its own level set phi_layer[k] indicating whether that layer
 * is cracked (phi < 0) or intact (phi > 0).
 *
 * The advection algorithm:
 * 1. Crack initiates at one surface (layer 0 or nlayers-1)
 * 2. When a layer's damage criterion is met, its phi flips negative
 * 3. The crack front advances to the next layer
 * 4. Enrichment is applied per-layer: a partially cracked shell has
 *    different enrichment at each integration point through thickness
 *
 * Reference: Areias & Belytschko (2005) "Non-linear analysis of shells with
 *            arbitrary evolving cracks using XFEM"
 */
class XFEMLayerAdvection {
public:
    XFEMLayerAdvection() = default;

    /**
     * @brief Initialize shell crack state.
     *
     * @param state Output state (modified)
     * @param nlayers Number of through-thickness layers
     * @param thickness Total shell thickness
     */
    void initialize(ShellCrackState& state, int nlayers, Real thickness) const {
        state.nlayers = std::min(nlayers, 8);
        state.total_thickness = thickness;
        state.cracked_layers = 0;
        state.crack_depth = 0.0;
        for (int i = 0; i < 8; ++i) state.phi_layer[i] = 1.0; // All intact
    }

    /**
     * @brief Advect crack through shell layers.
     *
     * Given a growth direction (through-thickness: +1 = top to bottom,
     * -1 = bottom to top), advance the crack by one layer if the damage
     * criterion is met.
     *
     * @param state Shell crack state (modified)
     * @param direction +1 for top-down, -1 for bottom-up
     * @return true if a new layer was cracked
     */
    bool advect_layer(ShellCrackState& state, int direction) const {
        if (state.cracked_layers >= state.nlayers) return false;

        int next_layer;
        if (direction >= 0) {
            // Top-down: crack layer 0, 1, 2, ...
            next_layer = state.cracked_layers;
        } else {
            // Bottom-up: crack layer nlayers-1, nlayers-2, ...
            next_layer = state.nlayers - 1 - state.cracked_layers;
        }

        if (next_layer < 0 || next_layer >= state.nlayers) return false;

        // Crack this layer (set phi negative)
        state.phi_layer[next_layer] = -1.0;
        state.cracked_layers++;

        // Update crack depth
        Real layer_thickness = state.total_thickness / static_cast<Real>(state.nlayers);
        state.crack_depth = static_cast<Real>(state.cracked_layers) * layer_thickness;

        return true;
    }

    /**
     * @brief Check if a specific layer is cracked.
     */
    static bool is_layer_cracked(const ShellCrackState& state, int layer) {
        if (layer < 0 || layer >= state.nlayers) return false;
        return state.phi_layer[layer] < 0.0;
    }

    /**
     * @brief Check if the shell is fully cracked through thickness.
     */
    static bool is_fully_cracked(const ShellCrackState& state) {
        return state.cracked_layers >= state.nlayers;
    }

    /**
     * @brief Get remaining ligament fraction (uncracked thickness / total).
     */
    static Real ligament_fraction(const ShellCrackState& state) {
        if (state.nlayers <= 0) return 0.0;
        return 1.0 - static_cast<Real>(state.cracked_layers) /
                      static_cast<Real>(state.nlayers);
    }

    /**
     * @brief Compute effective stiffness reduction due to through-thickness cracking.
     *
     * For a shell with some cracked layers, the bending stiffness is reduced
     * more than the membrane stiffness due to the loss of material far from
     * the neutral axis.
     *
     * Membrane reduction: (1 - cracked_fraction)
     * Bending reduction:  (1 - cracked_fraction)^3 approximately for surface cracks
     *
     * @param state Shell crack state
     * @return {membrane_factor, bending_factor} in [0, 1]
     */
    static std::array<Real, 2> stiffness_reduction(const ShellCrackState& state) {
        Real f = ligament_fraction(state);
        Real membrane = f;
        Real bending = f * f * f;  // Cubic reduction for bending
        return {membrane, bending};
    }

    /**
     * @brief Get through-thickness coordinate of the crack front.
     *
     * Returns the z-coordinate (from -t/2 to +t/2) of the current crack tip.
     *
     * @param state Shell crack state
     * @param direction +1 for top-down (crack starts at z = +t/2),
     *                  -1 for bottom-up (crack starts at z = -t/2)
     */
    static Real crack_front_z(const ShellCrackState& state, int direction) {
        Real t = state.total_thickness;
        Real depth = state.crack_depth;
        if (direction >= 0) {
            return t / 2.0 - depth;  // Starting from top
        } else {
            return -t / 2.0 + depth; // Starting from bottom
        }
    }
};


// ============================================================================
// 9. XFEMEnrichment36 - Heaviside + Tip Enrichment with Blending
// ============================================================================

/**
 * @brief Enrichment type for a node.
 */
enum class EnrichmentType {
    None = 0,
    Heaviside,    ///< Node in fully cut element (split by crack)
    CrackTip,     ///< Node in element containing crack tip
    Blending      ///< Node in blending element (transition zone)
};

/**
 * @brief Enriched node data.
 */
struct EnrichedNode {
    int node_id;
    EnrichmentType type;
    int extra_dofs;    ///< Number of additional DOFs
    Real phi;          ///< Level set at this node
    Real psi;          ///< Front level set at this node
};

/**
 * @brief XFEM enrichment with Heaviside, crack-tip functions, and partition of unity blending.
 *
 * Heaviside enrichment H(phi):
 *   For elements fully cut by the crack, each node gets extra DOFs.
 *   The enriched approximation: u_h = sum N_I * u_I + sum N_I * H(phi) * a_I
 *   where a_I are the enriched DOF values.
 *
 * Tip enrichment:
 *   For elements containing the crack tip, 4 branch functions are used:
 *   F_alpha = {sqrt(r)*sin(t/2), sqrt(r)*cos(t/2),
 *              sqrt(r)*sin(t/2)*sin(t), sqrt(r)*cos(t/2)*sin(t)}
 *   These capture the r^{1/2} singularity at the tip.
 *
 * Blending:
 *   In elements adjacent to enriched elements, a blending/ramp function
 *   is used to transition smoothly: ramp_I = sum_{J in enriched} N_J
 *   The enriched shape function is: N_I * ramp_I * F(x)
 *
 * Reference: Fries & Belytschko (2010)
 */
class XFEMEnrichment36 {
public:
    XFEMEnrichment36() = default;

    /**
     * @brief Heaviside function.
     * H(phi) = +1 if phi > 0, -1 if phi < 0, 0 at phi = 0
     */
    static Real heaviside(Real phi) {
        if (phi > 0.0) return 1.0;
        if (phi < 0.0) return -1.0;
        return 0.0;
    }

    /**
     * @brief Modified Heaviside (shifted) for enrichment.
     * H_shifted(phi, phi_I) = H(phi) - H(phi_I)
     * This ensures the enrichment vanishes at the node location.
     */
    static Real heaviside_shifted(Real phi, Real phi_I) {
        return heaviside(phi) - heaviside(phi_I);
    }

    /**
     * @brief Crack-tip enrichment functions (2D).
     *
     * Given crack-tip polar coordinates (r, theta):
     *   F_1 = sqrt(r) * sin(theta/2)
     *   F_2 = sqrt(r) * cos(theta/2)
     *   F_3 = sqrt(r) * sin(theta/2) * sin(theta)
     *   F_4 = sqrt(r) * cos(theta/2) * sin(theta)
     *
     * These span the Williams asymptotic expansion for the crack-tip
     * displacement field and capture the r^{1/2} singularity.
     */
    static std::array<Real, 4> tip_functions(Real r, Real theta) {
        Real sqr = std::sqrt(std::max(r, 0.0));
        Real ht = theta * 0.5;
        Real sh = std::sin(ht);
        Real ch = std::cos(ht);
        Real st = std::sin(theta);
        return {sqr * sh, sqr * ch, sqr * sh * st, sqr * ch * st};
    }

    /**
     * @brief Derivatives of crack-tip functions w.r.t. r and theta.
     *
     * dF_1/dr = sin(theta/2) / (2*sqrt(r))
     * dF_1/dtheta = sqrt(r) * cos(theta/2) / 2
     * etc.
     *
     * @return Array of {dF1_dr, dF1_dt, dF2_dr, dF2_dt, dF3_dr, dF3_dt, dF4_dr, dF4_dt}
     */
    static std::array<Real, 8> tip_function_derivs(Real r, Real theta) {
        Real sqr = std::sqrt(std::max(r, 1.0e-30));
        Real inv_2sqr = 0.5 / sqr;
        Real ht = theta * 0.5;
        Real sh = std::sin(ht);
        Real ch = std::cos(ht);
        Real st = std::sin(theta);
        Real ct = std::cos(theta);

        return {
            sh * inv_2sqr,                     // dF1/dr
            sqr * ch * 0.5,                    // dF1/dtheta
            ch * inv_2sqr,                     // dF2/dr
            -sqr * sh * 0.5,                   // dF2/dtheta
            sh * st * inv_2sqr,                // dF3/dr
            sqr * (ch * st * 0.5 + sh * ct),  // dF3/dtheta
            ch * st * inv_2sqr,                // dF4/dr
            sqr * (-sh * st * 0.5 + ch * ct)  // dF4/dtheta
        };
    }

    /**
     * @brief Compute crack-tip polar coordinates from Cartesian.
     *
     * @param px, py Point position
     * @param tip_x, tip_y Crack tip position
     * @param crack_angle Crack direction (radians from x-axis)
     * @return {r, theta}
     */
    static std::array<Real, 2> to_polar(Real px, Real py,
                                         Real tip_x, Real tip_y,
                                         Real crack_angle) {
        Real dx = px - tip_x;
        Real dy = py - tip_y;
        Real ca = std::cos(crack_angle);
        Real sa = std::sin(crack_angle);
        Real lx = dx * ca + dy * sa;
        Real ly = -dx * sa + dy * ca;
        Real r = std::sqrt(lx * lx + ly * ly);
        Real theta = std::atan2(ly, lx);
        return {r, theta};
    }

    /**
     * @brief Determine enrichment type for a node based on element crack state.
     *
     * @param phi Level set value at node
     * @param psi Front level set at node
     * @param elem_is_cut Whether the node's element is cut
     * @param elem_is_tip Whether the node's element contains the tip
     * @param elem_neighbors_enriched Whether adjacent elements are enriched
     */
    static EnrichmentType classify_node(Real phi, Real psi,
                                         bool elem_is_cut,
                                         bool elem_is_tip,
                                         bool elem_neighbors_enriched) {
        if (elem_is_tip) return EnrichmentType::CrackTip;
        if (elem_is_cut) return EnrichmentType::Heaviside;
        if (elem_neighbors_enriched) return EnrichmentType::Blending;
        return EnrichmentType::None;
    }

    /**
     * @brief Enrich an element: identify enriched nodes and count extra DOFs.
     *
     * @param phi Level set values at element nodes
     * @param psi Front level set values
     * @param nnodes Number of nodes per element
     * @param ndim Spatial dimension (2 or 3)
     * @param enriched_nodes Output: enriched node info
     * @return Total number of extra DOFs added
     */
    int enrich_element(const Real* phi, const Real* psi, int nnodes,
                       int ndim, EnrichedNode* enriched_nodes) const {
        // Determine element type
        bool has_pos = false, has_neg = false;
        bool psi_pos = false, psi_neg = false;

        for (int i = 0; i < nnodes; ++i) {
            if (phi[i] > 0.0) has_pos = true;
            if (phi[i] < 0.0) has_neg = true;
            if (psi[i] > 0.0) psi_pos = true;
            if (psi[i] <= 0.0) psi_neg = true;
        }

        bool is_cut = has_pos && has_neg && psi_neg;
        bool is_tip = has_pos && has_neg && psi_pos && psi_neg;

        int total_extra = 0;
        for (int i = 0; i < nnodes; ++i) {
            enriched_nodes[i].node_id = i;
            enriched_nodes[i].phi = phi[i];
            enriched_nodes[i].psi = psi[i];

            if (is_tip) {
                enriched_nodes[i].type = EnrichmentType::CrackTip;
                enriched_nodes[i].extra_dofs = 4 * ndim;
            } else if (is_cut) {
                enriched_nodes[i].type = EnrichmentType::Heaviside;
                enriched_nodes[i].extra_dofs = ndim;
            } else {
                enriched_nodes[i].type = EnrichmentType::None;
                enriched_nodes[i].extra_dofs = 0;
            }
            total_extra += enriched_nodes[i].extra_dofs;
        }

        return total_extra;
    }

    /**
     * @brief Enriched shape function value for Heaviside enrichment.
     *
     * N_enriched_I(x) = N_I(x) * [H(phi(x)) - H(phi_I)]
     *
     * @param N_I Standard shape function at evaluation point
     * @param phi_point Level set at evaluation point
     * @param phi_node Level set at the enriched node
     */
    static Real enriched_shape_heaviside(Real N_I, Real phi_point, Real phi_node) {
        return N_I * heaviside_shifted(phi_point, phi_node);
    }

    /**
     * @brief Enriched shape function for tip enrichment.
     *
     * N_tip_I_alpha(x) = N_I(x) * [F_alpha(x) - F_alpha(x_I)]
     *
     * The shift F_alpha(x_I) ensures partition of unity and eliminates
     * blending issues.
     */
    static Real enriched_shape_tip(Real N_I, Real F_alpha_point,
                                    Real F_alpha_node) {
        return N_I * (F_alpha_point - F_alpha_node);
    }

    /**
     * @brief Number of extra DOFs per node for each enrichment type.
     */
    static int extra_dofs_per_node(EnrichmentType type, int ndim) {
        switch (type) {
            case EnrichmentType::Heaviside: return ndim;
            case EnrichmentType::CrackTip:  return 4 * ndim;
            case EnrichmentType::Blending:  return ndim;
            case EnrichmentType::None:      return 0;
        }
        return 0;
    }
};


// ============================================================================
// 10. XFEMForceIntegration - Sub-element integration for split elements
// ============================================================================

/**
 * @brief Gauss point for sub-element integration.
 */
struct SubGaussPoint {
    Real x, y;      ///< Position in parametric space
    Real weight;    ///< Quadrature weight (includes sub-element Jacobian)
    int subdomain;  ///< Which side of crack: +1 or -1
};

/**
 * @brief Sub-element integration for XFEM crack elements.
 *
 * Standard Gauss quadrature fails for elements cut by a crack because
 * the integrand is discontinuous. The element must be partitioned into
 * sub-triangles (2D) or sub-tetrahedra (3D) on each side of the crack,
 * and integration performed on each sub-element.
 *
 * Algorithm for 2D quad/tri:
 * 1. Find intersection points of crack (phi=0 line) with element edges
 * 2. Split element into sub-triangles on each side
 * 3. Map Gauss points from reference triangle to each sub-triangle
 * 4. Evaluate enriched shape functions and integrate
 *
 * Reference: Moes, Dolbow, Belytschko (1999)
 */
class XFEMForceIntegration {
public:
    XFEMForceIntegration() = default;

    /**
     * @brief Split a triangle element along phi=0 line.
     *
     * Given a triangle with vertices (x0,y0), (x1,y1), (x2,y2) and
     * level set values phi0, phi1, phi2, find the sub-triangulation.
     *
     * Returns sub-triangle vertices and their side (+/-).
     *
     * @param x Array of 3 vertex x-coordinates
     * @param y Array of 3 vertex y-coordinates
     * @param phi Array of 3 level set values
     * @param sub_x Output: sub-triangle vertex x-coords (max 4 triangles * 3 verts)
     * @param sub_y Output: sub-triangle vertex y-coords
     * @param sub_sides Output: side (+1/-1) for each sub-triangle
     * @return Number of sub-triangles
     */
    int split_triangle(const Real* x, const Real* y, const Real* phi,
                       Real* sub_x, Real* sub_y, int* sub_sides) const {
        // Count sign changes
        int pos_count = 0, neg_count = 0;
        for (int i = 0; i < 3; ++i) {
            if (phi[i] > 0.0) pos_count++;
            else if (phi[i] < 0.0) neg_count++;
            else pos_count++; // Treat zero as positive
        }

        if (pos_count == 3 || neg_count == 3) {
            // No split needed: one sub-triangle = the original
            for (int i = 0; i < 3; ++i) {
                sub_x[i] = x[i];
                sub_y[i] = y[i];
            }
            sub_sides[0] = (pos_count == 3) ? 1 : -1;
            return 1;
        }

        // Find the isolated vertex (the one on the minority side)
        int isolated = -1;
        for (int i = 0; i < 3; ++i) {
            bool this_pos = (phi[i] >= 0.0);
            bool next_pos = (phi[(i+1)%3] >= 0.0);
            bool prev_pos = (phi[(i+2)%3] >= 0.0);
            if (this_pos != next_pos && this_pos != prev_pos) {
                isolated = i;
                break;
            }
        }
        if (isolated < 0) isolated = 0; // Fallback

        int v0 = isolated;
        int v1 = (isolated + 1) % 3;
        int v2 = (isolated + 2) % 3;

        // Find intersection points on edges v0-v1 and v0-v2
        Real t01 = xfem_detail::find_zero_crossing(phi[v0], phi[v1]);
        Real t02 = xfem_detail::find_zero_crossing(phi[v0], phi[v2]);

        // Intersection points
        Real ix01 = x[v0] + t01 * (x[v1] - x[v0]);
        Real iy01 = y[v0] + t01 * (y[v1] - y[v0]);
        Real ix02 = x[v0] + t02 * (x[v2] - x[v0]);
        Real iy02 = y[v0] + t02 * (y[v2] - y[v0]);

        int side_isolated = (phi[v0] >= 0.0) ? 1 : -1;
        int side_other = -side_isolated;

        // Sub-triangle 1: v0, i01, i02 (isolated side)
        sub_x[0] = x[v0]; sub_y[0] = y[v0];
        sub_x[1] = ix01;  sub_y[1] = iy01;
        sub_x[2] = ix02;  sub_y[2] = iy02;
        sub_sides[0] = side_isolated;

        // Sub-triangle 2: i01, v1, i02 (other side)
        sub_x[3] = ix01;  sub_y[3] = iy01;
        sub_x[4] = x[v1]; sub_y[4] = y[v1];
        sub_x[5] = ix02;  sub_y[5] = iy02;
        sub_sides[1] = side_other;

        // Sub-triangle 3: v1, v2, i02 (other side)
        sub_x[6] = x[v1]; sub_y[6] = y[v1];
        sub_x[7] = x[v2]; sub_y[7] = y[v2];
        sub_x[8] = ix02;  sub_y[8] = iy02;
        sub_sides[2] = side_other;

        return 3;
    }

    /**
     * @brief Generate Gauss points for sub-triangles.
     *
     * Uses 3-point Gauss quadrature on each sub-triangle.
     *
     * @param sub_x Sub-triangle x-coordinates (nsub * 3)
     * @param sub_y Sub-triangle y-coordinates (nsub * 3)
     * @param sub_sides Side tags per sub-triangle (nsub)
     * @param nsub Number of sub-triangles
     * @param gauss_pts Output: Gauss points with weights
     * @return Number of Gauss points generated
     */
    int generate_gauss_points(const Real* sub_x, const Real* sub_y,
                               const int* sub_sides, int nsub,
                               SubGaussPoint* gauss_pts) const {
        // 3-point triangle quadrature
        // Points at (1/6, 1/6), (2/3, 1/6), (1/6, 2/3) with weight 1/6 each
        static const Real tri_xi[3]  = {1.0/6.0, 2.0/3.0, 1.0/6.0};
        static const Real tri_eta[3] = {1.0/6.0, 1.0/6.0, 2.0/3.0};
        static const Real tri_w[3]   = {1.0/6.0, 1.0/6.0, 1.0/6.0};

        int gp_count = 0;
        for (int t = 0; t < nsub; ++t) {
            int base = t * 3;
            Real x0 = sub_x[base], y0 = sub_y[base];
            Real x1 = sub_x[base+1], y1 = sub_y[base+1];
            Real x2 = sub_x[base+2], y2 = sub_y[base+2];

            // Sub-triangle area (2 * area for triangle Jacobian)
            Real area = xfem_detail::triangle_area_2d(x0, y0, x1, y1, x2, y2);

            for (int g = 0; g < 3; ++g) {
                // Map from reference triangle to physical sub-triangle
                Real L1 = 1.0 - tri_xi[g] - tri_eta[g];
                Real L2 = tri_xi[g];
                Real L3 = tri_eta[g];

                gauss_pts[gp_count].x = L1 * x0 + L2 * x1 + L3 * x2;
                gauss_pts[gp_count].y = L1 * y0 + L2 * y1 + L3 * y2;
                // Weight = reference weight * 2 * area (Jacobian of triangle mapping)
                gauss_pts[gp_count].weight = tri_w[g] * 2.0 * area;
                gauss_pts[gp_count].subdomain = sub_sides[t];
                gp_count++;
            }
        }
        return gp_count;
    }

    /**
     * @brief Integrate a function over a split element.
     *
     * Given function values at Gauss points, compute the integral:
     *   I = sum_g f(x_g) * w_g
     *
     * @param values Function values at Gauss points
     * @param gauss_pts Gauss point data
     * @param ngp Number of Gauss points
     * @return Integral value
     */
    static Real integrate(const Real* values, const SubGaussPoint* gauss_pts,
                           int ngp) {
        Real result = 0.0;
        for (int g = 0; g < ngp; ++g) {
            result += values[g] * gauss_pts[g].weight;
        }
        return result;
    }

    /**
     * @brief Integrate constant value over sub-domain (one side only).
     *
     * Returns integral of 1 over the specified side of the split element.
     * This gives the area of that sub-domain.
     *
     * @param gauss_pts Gauss points
     * @param ngp Number of Gauss points
     * @param side Which side to integrate (+1 or -1)
     */
    static Real subdomain_area(const SubGaussPoint* gauss_pts, int ngp,
                                int side) {
        Real area = 0.0;
        for (int g = 0; g < ngp; ++g) {
            if (gauss_pts[g].subdomain == side) {
                area += gauss_pts[g].weight;
            }
        }
        return area;
    }

    /**
     * @brief Build enriched stiffness contribution for a split element.
     *
     * For a 2D element with Heaviside enrichment, the enriched stiffness is:
     *   K_aa_IJ = integral B_I^T * D * B_J * [H(phi) - H(phi_I)] * [H(phi) - H(phi_J)] dV
     *
     * This simplifies on each sub-domain where H(phi) is constant.
     *
     * @param N Shape function values at Gauss points (ngp x nnodes)
     * @param B B-matrix values at Gauss points (ngp x nstress x ndof)
     * @param D Material tangent (nstress x nstress)
     * @param phi_nodes Level set at nodes (nnodes)
     * @param gauss_pts Gauss points
     * @param ngp Number of Gauss points
     * @param nnodes Nodes per element
     * @param ndim Spatial dimension
     * @param K_enriched Output: enriched stiffness (nnodes*ndim x nnodes*ndim)
     */
    void integrate_split_element(const Real* N, const Real* B, const Real* D,
                                  const Real* phi_nodes,
                                  const SubGaussPoint* gauss_pts, int ngp,
                                  int nnodes, int ndim, int nstress,
                                  Real* K_enriched) const {
        int ndof = nnodes * ndim;
        // Zero output
        std::memset(K_enriched, 0, ndof * ndof * sizeof(Real));

        // Temporary for B^T * D * B at each Gauss point
        std::vector<Real> DB(nstress * ndof, 0.0);
        std::vector<Real> BtDB(ndof * ndof, 0.0);

        for (int g = 0; g < ngp; ++g) {
            // Evaluate H(phi) at this Gauss point
            Real H_gp = (gauss_pts[g].subdomain > 0) ? 1.0 : -1.0;

            // DB = D * B_g
            const Real* Bg = B + g * nstress * ndof;
            for (int i = 0; i < nstress; ++i) {
                for (int j = 0; j < ndof; ++j) {
                    Real s = 0.0;
                    for (int k = 0; k < nstress; ++k) {
                        s += D[i * nstress + k] * Bg[k * ndof + j];
                    }
                    DB[i * ndof + j] = s;
                }
            }

            // K += w_g * B_g^T * D * B_g * (H-H_I)*(H-H_J)
            // For enriched part, the (H-H_I) factor modifies each node's contribution
            for (int I = 0; I < nnodes; ++I) {
                Real H_I = XFEMEnrichment36::heaviside(phi_nodes[I]);
                Real fI = H_gp - H_I;
                if (std::abs(fI) < 1.0e-15) continue;

                for (int J = 0; J < nnodes; ++J) {
                    Real H_J = XFEMEnrichment36::heaviside(phi_nodes[J]);
                    Real fJ = H_gp - H_J;
                    if (std::abs(fJ) < 1.0e-15) continue;

                    Real factor = gauss_pts[g].weight * fI * fJ;

                    // Add BtDB contribution for DOFs of nodes I and J
                    for (int di = 0; di < ndim; ++di) {
                        int row = I * ndim + di;
                        for (int dj = 0; dj < ndim; ++dj) {
                            int col = J * ndim + dj;
                            Real val = 0.0;
                            for (int s = 0; s < nstress; ++s) {
                                val += Bg[s * ndof + row] * DB[s * ndof + col];
                            }
                            K_enriched[row * ndof + col] += factor * val;
                        }
                    }
                }
            }
        }
    }
};


// ============================================================================
// 11. XFEMCrackDirection - Crack growth direction criteria
// ============================================================================

/**
 * @brief Crack direction result.
 */
struct CrackDirectionResult {
    Real theta_c;       ///< Propagation angle (radians, from crack plane)
    Real G;             ///< Energy release rate
    Real K_eq;          ///< Equivalent SIF
    bool should_grow;   ///< Whether K_eq exceeds K_Ic
};

/**
 * @brief Crack propagation direction criteria.
 *
 * Implements:
 * 1. Maximum Tangential Stress (MTS / max hoop stress) criterion:
 *    theta_c = 2 * atan( (K_I - sqrt(K_I^2 + 8*K_II^2)) / (4*K_II) )
 *    For pure mode I: theta_c = 0
 *    For pure mode II: theta_c ~ -70.5 degrees
 *
 * 2. Maximum Energy Release Rate:
 *    G = (K_I^2 + K_II^2) / E'  (plane strain: E' = E/(1-nu^2))
 *    G = (K_I^2 + K_II^2) / E   (plane stress)
 *    K_III contribution: G += (1+nu) * K_III^2 / E
 *
 * 3. Equivalent SIF for mixed mode:
 *    K_eq = cos(theta_c/2) * [K_I * cos^2(theta_c/2) - 3/2 * K_II * sin(theta_c)]
 *
 * Reference: Erdogan & Sih (1963), Hussain, Pu & Underwood (1974)
 */
class XFEMCrackDirection {
public:
    XFEMCrackDirection() = default;

    /**
     * @brief Set material properties.
     *
     * @param E Young's modulus
     * @param nu Poisson's ratio
     * @param K_Ic Fracture toughness (mode I critical SIF)
     * @param plane_strain True for plane strain, false for plane stress
     */
    void set_material(Real E, Real nu, Real K_Ic, bool plane_strain = true) {
        E_ = E;
        nu_ = nu;
        K_Ic_ = K_Ic;
        plane_strain_ = plane_strain;
        if (plane_strain) {
            E_prime_ = E / (1.0 - nu * nu);
        } else {
            E_prime_ = E;
        }
    }

    /**
     * @brief Compute crack propagation direction using MTS criterion.
     *
     * Maximum Tangential Stress criterion (Erdogan-Sih 1963):
     *   theta_c = 2 * atan( (K_I - sqrt(K_I^2 + 8*K_II^2)) / (4*K_II) )
     *
     * Special cases:
     *   Pure mode I (K_II = 0): theta_c = 0 (self-similar propagation)
     *   Pure mode II: theta_c = 2*atan((-sqrt(8))/4) ~ -70.5 deg
     *
     * @param K_I Mode I SIF
     * @param K_II Mode II SIF
     * @param K_III Mode III SIF (used for energy release only)
     * @return CrackDirectionResult with angle, G, K_eq, and growth flag
     */
    CrackDirectionResult compute_direction(Real K_I, Real K_II,
                                            Real K_III = 0.0) const {
        CrackDirectionResult result;

        // MTS criterion for theta_c
        if (std::abs(K_II) < 1.0e-15 * (std::abs(K_I) + 1.0e-30)) {
            // Pure mode I: straight ahead
            result.theta_c = 0.0;
        } else {
            Real disc = std::sqrt(K_I * K_I + 8.0 * K_II * K_II);
            Real numer = K_I - disc;
            Real denom = 4.0 * K_II;
            result.theta_c = 2.0 * std::atan2(numer, denom);
        }

        // Energy release rate
        result.G = (K_I * K_I + K_II * K_II) / E_prime_;
        if (std::abs(K_III) > 1.0e-30) {
            result.G += (1.0 + nu_) * K_III * K_III / E_;
        }

        // Equivalent SIF using MTS
        Real ht = result.theta_c * 0.5;
        Real cos_ht = std::cos(ht);
        Real sin_t = std::sin(result.theta_c);
        result.K_eq = cos_ht * (K_I * cos_ht * cos_ht - 1.5 * K_II * sin_t);

        // Growth check
        result.should_grow = (std::abs(result.K_eq) >= K_Ic_);

        return result;
    }

    /**
     * @brief Compute energy release rate from J-integral components.
     *
     * For a 2D crack:
     *   J = G = (K_I^2 + K_II^2) / E'
     *
     * Given J directly (from domain integral), extract SIFs using
     * interaction integral decomposition.
     *
     * @param J J-integral value
     * @return Equivalent K from J: K = sqrt(J * E')
     */
    Real K_from_J(Real J) const {
        if (J < 0.0) return 0.0;
        return std::sqrt(J * E_prime_);
    }

    /**
     * @brief Compute the hoop stress at angle theta for given SIFs.
     *
     * sigma_theta = (1 / sqrt(2*pi*r)) * cos(theta/2) *
     *               [K_I * cos^2(theta/2) - 3/2 * K_II * sin(theta)]
     *
     * The MTS criterion maximizes this over theta (at unit r).
     *
     * @param K_I Mode I SIF
     * @param K_II Mode II SIF
     * @param theta Angle from crack plane
     * @return Hoop stress factor (without 1/sqrt(2*pi*r) prefactor)
     */
    static Real hoop_stress(Real K_I, Real K_II, Real theta) {
        Real ht = theta * 0.5;
        Real cos_ht = std::cos(ht);
        Real sin_t = std::sin(theta);
        return cos_ht * (K_I * cos_ht * cos_ht - 1.5 * K_II * sin_t);
    }

    /**
     * @brief Compute mixed-mode ratio.
     *
     * K_II / K_I gives the mode mixity. Pure mode I = 0, pure mode II = inf.
     */
    static Real mode_mixity(Real K_I, Real K_II) {
        if (std::abs(K_I) < 1.0e-30) {
            return (std::abs(K_II) < 1.0e-30) ? 0.0 : std::numeric_limits<Real>::max();
        }
        return K_II / K_I;
    }

    /**
     * @brief Convert propagation angle from crack-local to global frame.
     *
     * @param theta_local Local propagation angle (from MTS)
     * @param crack_angle Current crack angle in global frame
     * @return New crack angle in global frame
     */
    static Real to_global_angle(Real theta_local, Real crack_angle) {
        return crack_angle + theta_local;
    }

    Real fracture_toughness() const { return K_Ic_; }
    Real E_prime() const { return E_prime_; }

private:
    Real E_ = 1.0;
    Real nu_ = 0.3;
    Real K_Ic_ = 1.0e30;
    Real E_prime_ = 1.0;
    bool plane_strain_ = true;
};


// ============================================================================
// 12. XFEMVelocityUpdate - Enriched DOF velocity for explicit dynamics
// ============================================================================

/**
 * @brief Enriched DOF velocity update for explicit XFEM dynamics.
 *
 * In explicit dynamics with XFEM, the enriched DOFs (Heaviside and tip)
 * must be integrated in time alongside the standard DOFs.
 *
 * For Heaviside-enriched nodes, the physical velocities on each side
 * of the crack are:
 *   v_plus  = v_standard + a_enriched   (phi > 0 side)
 *   v_minus = v_standard - a_enriched   (phi < 0 side)
 *
 * where a_enriched are the enriched DOF values (velocity-like).
 *
 * The enriched velocity is updated via:
 *   a_enriched_new = a_enriched_old + dt * (f_enriched / m_enriched)
 *
 * For elements that transition from intact to cracked, ghost node
 * velocities must be initialized from the pre-crack velocity field.
 *
 * Reference: Belytschko et al. (2001) "Arbitrary discontinuities in FE"
 */
class XFEMVelocityUpdate {
public:
    XFEMVelocityUpdate() = default;

    /**
     * @brief Update enriched DOF velocities (central difference).
     *
     * v_enriched += dt * f_enriched / m_enriched
     *
     * @param enriched_v Enriched velocity DOFs (ndofs), modified in place
     * @param enriched_f Enriched force DOFs (ndofs)
     * @param enriched_m Enriched mass DOFs (ndofs), diagonal mass
     * @param ndofs Number of enriched DOFs
     * @param dt Timestep
     */
    void update_velocity(Real* enriched_v, const Real* enriched_f,
                         const Real* enriched_m, int ndofs, Real dt) const {
        for (int i = 0; i < ndofs; ++i) {
            if (std::abs(enriched_m[i]) > 1.0e-30) {
                enriched_v[i] += dt * enriched_f[i] / enriched_m[i];
            }
        }
    }

    /**
     * @brief Compute physical velocities on each side of the crack.
     *
     * v_plus  = v_std + v_enr  (phi > 0 side)
     * v_minus = v_std - v_enr  (phi < 0 side)
     *
     * @param standard_v Standard velocity at node (ndim components)
     * @param enriched_v Enriched velocity at node (ndim components)
     * @param phi Level set value at node
     * @param ndim Spatial dimension
     * @param v_plus Output: velocity on positive side (ndim)
     * @param v_minus Output: velocity on negative side (ndim)
     */
    static void split_velocity(const Real* standard_v, const Real* enriched_v,
                                Real phi, int ndim,
                                Real* v_plus, Real* v_minus) {
        for (int d = 0; d < ndim; ++d) {
            v_plus[d]  = standard_v[d] + enriched_v[d];
            v_minus[d] = standard_v[d] - enriched_v[d];
        }
    }

    /**
     * @brief Initialize enriched velocity when a new crack cuts an element.
     *
     * When a previously intact element becomes cracked, the enriched DOFs
     * are initialized to capture the pre-existing velocity field:
     *   a_enriched = 0 (no initial discontinuity)
     *
     * This ensures continuity at the moment of cracking.
     *
     * @param enriched_v Enriched velocities (zeroed out)
     * @param ndofs Number of enriched DOFs
     */
    static void initialize_enriched(Real* enriched_v, int ndofs) {
        std::memset(enriched_v, 0, ndofs * sizeof(Real));
    }

    /**
     * @brief Compute kinetic energy of enriched DOFs.
     *
     * KE_enriched = 0.5 * sum(m_i * v_i^2)
     */
    static Real enriched_kinetic_energy(const Real* enriched_v,
                                         const Real* enriched_m, int ndofs) {
        Real ke = 0.0;
        for (int i = 0; i < ndofs; ++i) {
            ke += enriched_m[i] * enriched_v[i] * enriched_v[i];
        }
        return 0.5 * ke;
    }

    /**
     * @brief Update positions for enriched DOFs.
     *
     * x_enriched += dt * v_enriched
     *
     * These are the ghost node displacements.
     */
    static void update_position(Real* enriched_x, const Real* enriched_v,
                                 int ndofs, Real dt) {
        for (int i = 0; i < ndofs; ++i) {
            enriched_x[i] += dt * enriched_v[i];
        }
    }

    /**
     * @brief Compute crack opening velocity (jump in velocity across crack).
     *
     * COV = v_plus - v_minus = 2 * v_enriched
     *
     * @param enriched_v Enriched velocity components (ndim)
     * @param ndim Spatial dimension
     * @return Magnitude of crack opening velocity
     */
    static Real crack_opening_velocity(const Real* enriched_v, int ndim) {
        Real cov2 = 0.0;
        for (int d = 0; d < ndim; ++d) {
            Real dv = 2.0 * enriched_v[d];
            cov2 += dv * dv;
        }
        return std::sqrt(cov2);
    }

    /**
     * @brief Compute enriched nodal forces from internal stress.
     *
     * f_enriched_I = integral B_I^T * sigma * (H(phi) - H(phi_I)) dV
     *
     * This is evaluated using sub-element integration.
     *
     * @param B B-matrix values at Gauss points (ngp x nstress x ndof)
     * @param sigma Stress at Gauss points (ngp x nstress)
     * @param phi_nodes Level set at nodes (nnodes)
     * @param gauss_pts Sub-element Gauss points
     * @param ngp Number of Gauss points
     * @param nnodes Nodes per element
     * @param ndim Spatial dimension
     * @param nstress Number of stress components
     * @param f_enriched Output: enriched forces (nnodes * ndim)
     */
    void compute_enriched_forces(const Real* B, const Real* sigma,
                                  const Real* phi_nodes,
                                  const SubGaussPoint* gauss_pts, int ngp,
                                  int nnodes, int ndim, int nstress,
                                  Real* f_enriched) const {
        int ndof = nnodes * ndim;
        std::memset(f_enriched, 0, ndof * sizeof(Real));

        for (int g = 0; g < ngp; ++g) {
            Real H_gp = (gauss_pts[g].subdomain > 0) ? 1.0 : -1.0;
            const Real* Bg = B + g * nstress * ndof;
            const Real* sg = sigma + g * nstress;

            for (int I = 0; I < nnodes; ++I) {
                Real H_I = XFEMEnrichment36::heaviside(phi_nodes[I]);
                Real factor = (H_gp - H_I) * gauss_pts[g].weight;
                if (std::abs(factor) < 1.0e-15) continue;

                for (int d = 0; d < ndim; ++d) {
                    int dof = I * ndim + d;
                    Real val = 0.0;
                    for (int s = 0; s < nstress; ++s) {
                        val += Bg[s * ndof + dof] * sg[s];
                    }
                    f_enriched[dof] += factor * val;
                }
            }
        }
    }
};

} // namespace fem
} // namespace nxs
