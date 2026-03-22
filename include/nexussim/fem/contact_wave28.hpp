#pragma once

/**
 * @file contact_wave28.hpp
 * @brief Wave 28: Advanced Contact Algorithms for NexusSim
 *
 * Sub-modules:
 * - 28a: CurvedSegmentContact    — Quadratic interpolation on contact segments
 * - 28b: ThermalContactResistance — Pressure-dependent gap conductance + radiation
 * - 28c: WearModelContact         — Archard wear model with geometry update
 * - 28d: RollingResistanceContact — Moment-based rolling friction
 * - 28e: IntersectionAwareContact — Initial overlap detection and ramped resolution
 *
 * References:
 * - Wriggers (2006) "Computational Contact Mechanics"
 * - Archard (1953) "Contact and rubbing of flat surfaces"
 * - Johnson (1985) "Contact Mechanics"
 * - Laursen (2002) "Computational Contact and Impact Mechanics"
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
// Wave 28 contact utility helpers (self-contained)
// ============================================================================

namespace wave28_detail {

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
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
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

} // namespace wave28_detail

// ============================================================================
// 28a: CurvedSegmentContact — Quadratic Interpolation on Contact Segments
// ============================================================================

/**
 * @brief Contact algorithm using quadratic (3-node) segment interpolation.
 *
 * Standard penalty contact uses linear (flat) segments between two nodes.
 * Curved segment contact uses a 3-node quadratic parametric representation:
 *
 *   x(xi) = N1(xi)*x1 + N2(xi)*x2 + N3(xi)*x3
 *
 * where:
 *   N1(xi) = xi*(xi - 1)/2    (node 1 at xi = -1)
 *   N2(xi) = 1 - xi^2          (node 2 at xi = 0, midpoint)
 *   N3(xi) = xi*(xi + 1)/2    (node 3 at xi = +1)
 *
 * The closest point on the curve to a slave node is found via Newton iteration
 * on the distance-squared function. The contact gap is measured along the
 * curve normal at the closest point.
 *
 * Normal computation:
 *   tangent t = dx/dxi at closest point
 *   normal n = t x reference_axis, normalized
 *   (reference axis chosen as the axis most orthogonal to t)
 *
 * Gap function:
 *   g = (x_slave - x_master(xi_closest)) . n(xi_closest)
 *
 * Contact force:
 *   f = penalty_stiffness * |g|   (only when g < 0, i.e., penetration)
 */
class CurvedSegmentContact {
public:
    /// Maximum Newton iterations for closest-point projection
    int max_iterations_ = 20;

    /// Newton convergence tolerance
    Real newton_tol_ = 1.0e-12;

    /// Default reference direction for normal computation (z-axis)
    Real ref_axis_[3] = {0.0, 0.0, 1.0};

    CurvedSegmentContact() = default;

    /**
     * @brief Construct with custom parameters.
     * @param max_iter  Maximum Newton iterations
     * @param tol       Convergence tolerance
     * @param ref_axis  Reference axis for normal computation
     */
    CurvedSegmentContact(int max_iter, Real tol, const Real ref_axis[3])
        : max_iterations_(max_iter), newton_tol_(tol)
    {
        ref_axis_[0] = ref_axis[0];
        ref_axis_[1] = ref_axis[1];
        ref_axis_[2] = ref_axis[2];
    }

    // ---- Quadratic shape functions ----

    /**
     * @brief Evaluate quadratic shape functions at parameter xi in [-1, 1].
     * @param xi   Parametric coordinate
     * @param N    Output shape function values [3]
     */
    KOKKOS_INLINE_FUNCTION
    static void shape_functions(Real xi, Real N[3]) {
        N[0] = xi * (xi - 1.0) / 2.0;   // node 1 at xi = -1
        N[1] = 1.0 - xi * xi;             // node 2 at xi =  0 (midpoint)
        N[2] = xi * (xi + 1.0) / 2.0;    // node 3 at xi = +1
    }

    /**
     * @brief Evaluate shape function derivatives at parameter xi.
     * @param xi   Parametric coordinate
     * @param dN   Output shape function derivatives [3]
     */
    KOKKOS_INLINE_FUNCTION
    static void shape_function_derivs(Real xi, Real dN[3]) {
        dN[0] = xi - 0.5;      // dN1/dxi = (2*xi - 1)/2
        dN[1] = -2.0 * xi;     // dN2/dxi = -2*xi
        dN[2] = xi + 0.5;      // dN3/dxi = (2*xi + 1)/2
    }

    /**
     * @brief Second derivatives of shape functions (constant for quadratic).
     * @param d2N  Output second derivatives [3]
     */
    KOKKOS_INLINE_FUNCTION
    static void shape_function_second_derivs(Real d2N[3]) {
        d2N[0] = 1.0;    // d2N1/dxi2
        d2N[1] = -2.0;   // d2N2/dxi2
        d2N[2] = 1.0;    // d2N3/dxi2
    }

    // ---- Interpolation ----

    /**
     * @brief Quadratic interpolation: evaluate position on curved segment.
     * @param xi      Parametric coordinate in [-1, 1]
     * @param nodes   Segment node coordinates [3][3] (3 nodes, each xyz)
     * @param result  Interpolated point [3]
     */
    KOKKOS_INLINE_FUNCTION
    void quadratic_interpolate(Real xi, const Real nodes[3][3], Real result[3]) const {
        Real N[3];
        shape_functions(xi, N);
        for (int d = 0; d < 3; ++d) {
            result[d] = N[0]*nodes[0][d] + N[1]*nodes[1][d] + N[2]*nodes[2][d];
        }
    }

    /**
     * @brief Compute tangent vector dx/dxi at parametric coordinate xi.
     * @param xi      Parametric coordinate
     * @param nodes   Segment node coordinates [3][3]
     * @param tangent Output tangent vector [3]
     */
    KOKKOS_INLINE_FUNCTION
    void tangent_at(Real xi, const Real nodes[3][3], Real tangent[3]) const {
        Real dN[3];
        shape_function_derivs(xi, dN);
        for (int d = 0; d < 3; ++d) {
            tangent[d] = dN[0]*nodes[0][d] + dN[1]*nodes[1][d] + dN[2]*nodes[2][d];
        }
    }

    /**
     * @brief Compute second derivative d2x/dxi2 at parametric coordinate xi.
     * @param nodes   Segment node coordinates [3][3]
     * @param d2x     Output second derivative vector [3]
     */
    KOKKOS_INLINE_FUNCTION
    void second_deriv_at(const Real nodes[3][3], Real d2x[3]) const {
        Real d2N[3];
        shape_function_second_derivs(d2N);
        for (int d = 0; d < 3; ++d) {
            d2x[d] = d2N[0]*nodes[0][d] + d2N[1]*nodes[1][d] + d2N[2]*nodes[2][d];
        }
    }

    /**
     * @brief Compute outward normal at parametric coordinate xi.
     *
     * The normal is computed from the tangent vector and a reference axis:
     *   n = tangent x ref_axis, normalized.
     * If the tangent is nearly parallel to ref_axis, a fallback axis is chosen.
     *
     * @param xi      Parametric coordinate
     * @param nodes   Segment node coordinates [3][3]
     * @param normal  Output unit normal [3]
     */
    KOKKOS_INLINE_FUNCTION
    void normal_at(Real xi, const Real nodes[3][3], Real normal[3]) const {
        Real tang[3];
        tangent_at(xi, nodes, tang);
        wave28_detail::normalize3(tang);

        // Choose reference axis most orthogonal to tangent
        Real ref[3] = {ref_axis_[0], ref_axis_[1], ref_axis_[2]};
        Real dot_tr = std::abs(wave28_detail::dot3(tang, ref));
        if (dot_tr > 0.9) {
            // Tangent nearly parallel to ref; pick alternative
            Real alt1[3] = {1.0, 0.0, 0.0};
            Real alt2[3] = {0.0, 1.0, 0.0};
            if (std::abs(wave28_detail::dot3(tang, alt1)) < 0.9) {
                ref[0] = alt1[0]; ref[1] = alt1[1]; ref[2] = alt1[2];
            } else {
                ref[0] = alt2[0]; ref[1] = alt2[1]; ref[2] = alt2[2];
            }
        }

        wave28_detail::cross3(tang, ref, normal);
        wave28_detail::normalize3(normal);
    }

    /**
     * @brief Project a slave point onto the curved segment and compute gap.
     *
     * Uses Newton-Raphson iteration to find the closest point parameter xi*
     * that minimizes || slave_pt - x(xi) ||^2.
     *
     * The derivative of the distance-squared function is:
     *   f(xi)  = (slave - x(xi)) . (dx/dxi) = 0
     *   f'(xi) = -(dx/dxi).(dx/dxi) + (slave - x(xi)).(d2x/dxi2)
     *
     * @param slave_pt   Slave point coordinates [3]
     * @param seg_nodes  Segment node coordinates [3][3]
     * @param[out] xi_out    Closest parametric coordinate
     * @param[out] gap_out   Signed gap (negative = penetration)
     * @param[out] normal_out Contact normal at closest point [3]
     * @return Number of Newton iterations used
     */
    KOKKOS_INLINE_FUNCTION
    int project_point(const Real slave_pt[3], const Real seg_nodes[3][3],
                      Real& xi_out, Real& gap_out, Real normal_out[3]) const {
        // Initialize xi at the centroid parameter
        Real xi = 0.0;

        // Precompute second derivative (constant for quadratic)
        Real d2x[3];
        second_deriv_at(seg_nodes, d2x);

        int iter = 0;
        for (; iter < max_iterations_; ++iter) {
            // Current point on curve
            Real xc[3];
            quadratic_interpolate(xi, seg_nodes, xc);

            // Tangent at current xi
            Real tang[3];
            tangent_at(xi, seg_nodes, tang);

            // Difference: slave - x(xi)
            Real diff[3];
            wave28_detail::sub3(slave_pt, xc, diff);

            // f(xi) = diff . tang
            Real f_val = wave28_detail::dot3(diff, tang);

            // f'(xi) = -tang.tang + diff.d2x
            Real fp_val = -wave28_detail::dot3(tang, tang) + wave28_detail::dot3(diff, d2x);

            if (std::abs(fp_val) < 1.0e-30) break;

            Real dxi = -f_val / fp_val;
            xi += dxi;

            // Clamp to [-1, 1]
            if (xi < -1.0) xi = -1.0;
            if (xi >  1.0) xi =  1.0;

            if (std::abs(dxi) < newton_tol_) {
                ++iter;
                break;
            }
        }

        xi_out = xi;

        // Compute gap and normal at converged xi
        Real xc[3];
        quadratic_interpolate(xi, seg_nodes, xc);
        normal_at(xi, seg_nodes, normal_out);

        Real diff[3];
        wave28_detail::sub3(slave_pt, xc, diff);
        gap_out = wave28_detail::dot3(diff, normal_out);

        return iter;
    }

    /**
     * @brief Compute contact force magnitude from gap and penalty stiffness.
     *
     * Force is nonzero only for negative gap (penetration):
     *   F = penalty_stiffness * |gap|   if gap < 0
     *   F = 0                           if gap >= 0
     *
     * @param gap                Signed gap
     * @param penalty_stiffness  Penalty parameter (force/length)
     * @return Contact force magnitude (always >= 0)
     */
    KOKKOS_INLINE_FUNCTION
    Real contact_force(Real gap, Real penalty_stiffness) const {
        if (gap < 0.0) {
            return penalty_stiffness * (-gap);
        }
        return 0.0;
    }

    /**
     * @brief Compute the curvature of the segment at parametric coordinate xi.
     *
     * Curvature kappa = |dx/dxi x d2x/dxi2| / |dx/dxi|^3
     *
     * @param xi     Parametric coordinate
     * @param nodes  Segment nodes [3][3]
     * @return Curvature value
     */
    KOKKOS_INLINE_FUNCTION
    Real curvature_at(Real xi, const Real nodes[3][3]) const {
        Real tang[3], d2x[3];
        tangent_at(xi, nodes, tang);
        second_deriv_at(nodes, d2x);

        Real cross[3];
        wave28_detail::cross3(tang, d2x, cross);
        Real cross_mag = wave28_detail::norm3(cross);
        Real tang_mag = wave28_detail::norm3(tang);
        Real tang3 = tang_mag * tang_mag * tang_mag;

        if (tang3 < 1.0e-30) return 0.0;
        return cross_mag / tang3;
    }

    /**
     * @brief Compute the arc length of the curved segment by Gauss quadrature.
     *
     * Uses 3-point Gauss-Legendre quadrature on [-1, 1]:
     *   L = sum_i w_i * |dx/dxi(xi_i)|
     *
     * @param nodes  Segment nodes [3][3]
     * @return Approximate arc length
     */
    KOKKOS_INLINE_FUNCTION
    Real arc_length(const Real nodes[3][3]) const {
        // 3-point Gauss-Legendre points and weights on [-1, 1]
        const Real gp[3] = { -std::sqrt(3.0/5.0), 0.0, std::sqrt(3.0/5.0) };
        const Real gw[3] = { 5.0/9.0, 8.0/9.0, 5.0/9.0 };

        Real length = 0.0;
        for (int i = 0; i < 3; ++i) {
            Real tang[3];
            tangent_at(gp[i], nodes, tang);
            length += gw[i] * wave28_detail::norm3(tang);
        }
        return length;
    }
};

// ============================================================================
// 28b: ThermalContactResistance — Pressure-Dependent Gap Conductance
// ============================================================================

/**
 * @brief Thermal contact resistance model with pressure-dependent conductance
 *        and radiative heat transfer across the contact gap.
 *
 * Gap conductance model:
 *   h_c = h_c0 * (p / p_ref)^n + h_gas * (k_gas / d_gap)
 *
 *   - h_c0    : base contact conductance [W/(m^2 K)]
 *   - p       : contact pressure [Pa]
 *   - p_ref   : reference pressure [Pa]
 *   - n       : pressure exponent (typically 0.5 to 1.0)
 *   - h_gas   : dimensionless gas conductance multiplier
 *   - k_gas   : gas thermal conductivity [W/(m K)]
 *   - d_gap   : gap distance [m]
 *
 * Radiation across gap:
 *   q_rad = epsilon_eff * sigma_SB * (T1^4 - T2^4)
 *
 *   - epsilon_eff = 1 / (1/epsilon1 + 1/epsilon2 - 1)
 *   - sigma_SB    = 5.67e-8 W/(m^2 K^4)
 *
 * Total heat flux:
 *   q = h_c * (T1 - T2) + q_rad
 */
class ThermalContactResistance {
public:
    Real h_c0_;        ///< Base contact conductance [W/(m^2 K)]
    Real p_ref_;       ///< Reference pressure [Pa]
    Real exponent_;    ///< Pressure exponent
    Real k_gas_;       ///< Gas thermal conductivity [W/(m K)]
    Real h_gas_;       ///< Gas conductance multiplier (dimensionless)
    Real epsilon1_;    ///< Emissivity of surface 1
    Real epsilon2_;    ///< Emissivity of surface 2
    Real epsilon_eff_; ///< Effective emissivity

    /// Stefan-Boltzmann constant [W/(m^2 K^4)]
    static constexpr Real sigma_SB = 5.67e-8;

    /// Minimum gap distance to avoid division by zero [m]
    static constexpr Real min_gap = 1.0e-10;

    /**
     * @brief Default constructor.
     */
    ThermalContactResistance()
        : h_c0_(0.0), p_ref_(1.0), exponent_(1.0), k_gas_(0.0),
          h_gas_(1.0), epsilon1_(1.0), epsilon2_(1.0), epsilon_eff_(1.0) {}

    /**
     * @brief Construct with physical parameters.
     * @param h_c0     Base contact conductance [W/(m^2 K)]
     * @param p_ref    Reference pressure [Pa]
     * @param n        Pressure exponent
     * @param k_gas    Gas thermal conductivity [W/(m K)]
     * @param epsilon1 Emissivity of surface 1
     * @param epsilon2 Emissivity of surface 2
     */
    ThermalContactResistance(Real h_c0, Real p_ref, Real n, Real k_gas,
                             Real epsilon1, Real epsilon2)
        : h_c0_(h_c0), p_ref_(p_ref), exponent_(n), k_gas_(k_gas),
          h_gas_(1.0), epsilon1_(epsilon1), epsilon2_(epsilon2)
    {
        // Compute effective emissivity
        if (epsilon1 > 0.0 && epsilon2 > 0.0) {
            epsilon_eff_ = 1.0 / (1.0/epsilon1 + 1.0/epsilon2 - 1.0);
        } else {
            epsilon_eff_ = 0.0;
        }
    }

    /**
     * @brief Set gas conductance multiplier.
     * @param h_gas Dimensionless multiplier for gas conductance term
     */
    void set_gas_multiplier(Real h_gas) { h_gas_ = h_gas; }

    /**
     * @brief Compute gap conductance at given pressure and gap distance.
     *
     * h_c = h_c0 * (p / p_ref)^n + h_gas * k_gas / d_gap
     *
     * For zero or negative pressure, the pressure-dependent term is zero.
     * For very small gap distances, a minimum gap is enforced.
     *
     * @param pressure      Contact pressure [Pa]
     * @param gap_distance  Gap distance [m]
     * @return Gap conductance [W/(m^2 K)]
     */
    KOKKOS_INLINE_FUNCTION
    Real gap_conductance(Real pressure, Real gap_distance) const {
        Real h_pressure = 0.0;
        if (pressure > 0.0 && p_ref_ > 0.0) {
            h_pressure = h_c0_ * std::pow(pressure / p_ref_, exponent_);
        }

        Real h_gas_term = 0.0;
        Real d = (gap_distance > min_gap) ? gap_distance : min_gap;
        if (k_gas_ > 0.0) {
            h_gas_term = h_gas_ * k_gas_ / d;
        }

        return h_pressure + h_gas_term;
    }

    /**
     * @brief Compute radiative heat flux across the gap.
     *
     * q_rad = epsilon_eff * sigma_SB * (T1^4 - T2^4)
     *
     * The sign convention is positive heat flow from surface 1 to surface 2
     * when T1 > T2.
     *
     * @param T1 Temperature of surface 1 [K]
     * @param T2 Temperature of surface 2 [K]
     * @return Radiative heat flux [W/m^2]
     */
    KOKKOS_INLINE_FUNCTION
    Real radiation_flux(Real T1, Real T2) const {
        Real T1_4 = T1 * T1 * T1 * T1;
        Real T2_4 = T2 * T2 * T2 * T2;
        return epsilon_eff_ * sigma_SB * (T1_4 - T2_4);
    }

    /**
     * @brief Compute total heat flux combining conduction and radiation.
     *
     * q_total = h_c * (T1 - T2) + epsilon_eff * sigma_SB * (T1^4 - T2^4)
     *
     * @param T1            Temperature of surface 1 [K]
     * @param T2            Temperature of surface 2 [K]
     * @param pressure      Contact pressure [Pa]
     * @param gap_distance  Gap distance [m]
     * @return Total heat flux [W/m^2]
     */
    KOKKOS_INLINE_FUNCTION
    Real total_heat_flux(Real T1, Real T2, Real pressure, Real gap_distance) const {
        Real h_c = gap_conductance(pressure, gap_distance);
        Real q_cond = h_c * (T1 - T2);
        Real q_rad = radiation_flux(T1, T2);
        return q_cond + q_rad;
    }

    /**
     * @brief Compute linearized radiation conductance (for Newton iteration).
     *
     * h_rad = 4 * epsilon_eff * sigma_SB * T_avg^3
     *
     * where T_avg = (T1 + T2) / 2.
     *
     * @param T1 Temperature of surface 1 [K]
     * @param T2 Temperature of surface 2 [K]
     * @return Linearized radiation conductance [W/(m^2 K)]
     */
    KOKKOS_INLINE_FUNCTION
    Real linearized_radiation_conductance(Real T1, Real T2) const {
        Real T_avg = 0.5 * (T1 + T2);
        return 4.0 * epsilon_eff_ * sigma_SB * T_avg * T_avg * T_avg;
    }

    /**
     * @brief Compute total effective conductance (conduction + linearized radiation).
     *
     * @param T1            Temperature of surface 1 [K]
     * @param T2            Temperature of surface 2 [K]
     * @param pressure      Contact pressure [Pa]
     * @param gap_distance  Gap distance [m]
     * @return Total effective conductance [W/(m^2 K)]
     */
    KOKKOS_INLINE_FUNCTION
    Real total_conductance(Real T1, Real T2, Real pressure, Real gap_distance) const {
        Real h_c = gap_conductance(pressure, gap_distance);
        Real h_rad = linearized_radiation_conductance(T1, T2);
        return h_c + h_rad;
    }
};

// ============================================================================
// 28c: WearModelContact — Archard Wear with Geometry Update
// ============================================================================

/**
 * @brief Contact wear model based on Archard's equation with surface geometry
 *        update capability.
 *
 * Archard's wear law:
 *   V_wear = K * F_n * s / H
 *
 *   - K     : dimensionless wear coefficient
 *   - F_n   : normal contact force [N]
 *   - s     : sliding distance [m]
 *   - H     : material hardness [Pa]
 *
 * Wear depth rate (local form):
 *   dh/dt = K * p * v_slide / H
 *
 *   - p       : contact pressure [Pa]
 *   - v_slide : sliding velocity [m/s]
 *
 * Geometry update:
 *   Node position is shifted inward along the surface normal by the
 *   accumulated wear depth:
 *     x_new = x_old - wear_depth * normal
 */
class WearModelContact {
public:
    Real K_wear_;    ///< Dimensionless wear coefficient
    Real hardness_;  ///< Material hardness [Pa]

    /**
     * @brief Default constructor.
     */
    WearModelContact() : K_wear_(0.0), hardness_(1.0) {}

    /**
     * @brief Construct with wear parameters.
     * @param K_wear   Dimensionless wear coefficient
     * @param hardness Material hardness [Pa]
     */
    WearModelContact(Real K_wear, Real hardness)
        : K_wear_(K_wear), hardness_(hardness)
    {
        if (hardness_ < 1.0e-30) hardness_ = 1.0e-30;
    }

    /**
     * @brief Compute instantaneous wear depth rate.
     *
     * dh/dt = K * p * v_slide / H
     *
     * @param contact_pressure  Contact pressure [Pa]
     * @param sliding_velocity  Magnitude of sliding velocity [m/s]
     * @return Wear depth rate [m/s]
     */
    KOKKOS_INLINE_FUNCTION
    Real wear_depth_rate(Real contact_pressure, Real sliding_velocity) const {
        if (contact_pressure <= 0.0 || sliding_velocity <= 0.0) return 0.0;
        return K_wear_ * contact_pressure * sliding_velocity / hardness_;
    }

    /**
     * @brief Accumulate wear depth over a time step.
     *
     * h_new = h_old + dh/dt * dt
     *
     * @param current_depth  Current accumulated wear depth [m]
     * @param dh_dt          Wear depth rate [m/s]
     * @param dt             Time step [s]
     * @return Updated wear depth [m]
     */
    KOKKOS_INLINE_FUNCTION
    Real accumulate_wear(Real current_depth, Real dh_dt, Real dt) const {
        return current_depth + dh_dt * dt;
    }

    /**
     * @brief Update node position due to wear.
     *
     * The node is shifted inward along the surface normal:
     *   x_new = x_old - wear_depth * normal
     *
     * @param node_pos      Original node position [3]
     * @param normal        Outward surface normal [3] (unit vector)
     * @param wear_depth    Accumulated wear depth [m]
     * @param updated_pos   Updated node position [3]
     */
    KOKKOS_INLINE_FUNCTION
    void geometry_update(const Real node_pos[3], const Real normal[3],
                         Real wear_depth, Real updated_pos[3]) const {
        updated_pos[0] = node_pos[0] - wear_depth * normal[0];
        updated_pos[1] = node_pos[1] - wear_depth * normal[1];
        updated_pos[2] = node_pos[2] - wear_depth * normal[2];
    }

    /**
     * @brief Compute total volume worn across all contact patches.
     *
     * V_total = sum_i (wear_depth_i * area_i)
     *
     * @param wear_depths  Array of wear depths per contact [m]
     * @param areas        Array of contact areas per contact [m^2]
     * @param num_contacts Number of contacts
     * @return Total worn volume [m^3]
     */
    KOKKOS_INLINE_FUNCTION
    Real total_volume_worn(const Real wear_depths[], const Real areas[],
                           int num_contacts) const {
        Real vol = 0.0;
        for (int i = 0; i < num_contacts; ++i) {
            vol += wear_depths[i] * areas[i];
        }
        return vol;
    }

    /**
     * @brief Compute volumetric wear rate using Archard's law directly.
     *
     * dV/dt = K * F_n * v_slide / H
     *
     * @param normal_force     Normal contact force [N]
     * @param sliding_velocity Sliding velocity magnitude [m/s]
     * @return Volumetric wear rate [m^3/s]
     */
    KOKKOS_INLINE_FUNCTION
    Real volumetric_wear_rate(Real normal_force, Real sliding_velocity) const {
        if (normal_force <= 0.0 || sliding_velocity <= 0.0) return 0.0;
        return K_wear_ * normal_force * sliding_velocity / hardness_;
    }

    /**
     * @brief Compute wear depth at a single contact point over a time step.
     *
     * Convenience method combining wear_depth_rate and accumulate_wear.
     *
     * @param current_depth     Current accumulated depth [m]
     * @param contact_pressure  Contact pressure [Pa]
     * @param sliding_velocity  Sliding velocity magnitude [m/s]
     * @param dt                Time step [s]
     * @return Updated wear depth [m]
     */
    KOKKOS_INLINE_FUNCTION
    Real update_wear_depth(Real current_depth, Real contact_pressure,
                           Real sliding_velocity, Real dt) const {
        Real rate = wear_depth_rate(contact_pressure, sliding_velocity);
        return accumulate_wear(current_depth, rate, dt);
    }
};

// ============================================================================
// 28d: RollingResistanceContact — Moment-Based Rolling Friction
// ============================================================================

/**
 * @brief Rolling resistance contact model with rolling and spin friction moments.
 *
 * Rolling friction moment:
 *   M_roll = mu_roll * F_n * R_eff * (-omega_roll_hat)
 *
 *   - mu_roll  : rolling friction coefficient (dimensionless)
 *   - F_n      : normal contact force [N]
 *   - R_eff    : effective radius = R1*R2 / (R1 + R2) [m]
 *   - omega_roll_hat : unit vector in rolling angular velocity direction
 *
 * Spin resistance moment (about contact normal):
 *   M_spin = mu_spin * F_n * R_eff * (-omega_spin_hat) . n
 *
 *   - mu_spin  : spin friction coefficient (dimensionless)
 *   - n        : contact normal direction
 *
 * The total resistance moment is the sum of rolling and spin moments.
 * The moment direction opposes the angular velocity.
 */
class RollingResistanceContact {
public:
    Real mu_roll_;   ///< Rolling friction coefficient
    Real mu_spin_;   ///< Spin friction coefficient

    /**
     * @brief Default constructor.
     */
    RollingResistanceContact() : mu_roll_(0.0), mu_spin_(0.0) {}

    /**
     * @brief Construct with friction coefficients.
     * @param mu_roll  Rolling friction coefficient
     * @param mu_spin  Spin friction coefficient
     */
    RollingResistanceContact(Real mu_roll, Real mu_spin)
        : mu_roll_(mu_roll), mu_spin_(mu_spin) {}

    /**
     * @brief Compute effective radius from two body radii.
     *
     * R_eff = R1 * R2 / (R1 + R2)
     *
     * Special cases:
     *   - If R2 is very large (flat surface), R_eff -> R1
     *   - If both radii are equal, R_eff = R/2
     *
     * @param R1 Radius of body 1 [m]
     * @param R2 Radius of body 2 [m]
     * @return Effective radius [m]
     */
    KOKKOS_INLINE_FUNCTION
    Real effective_radius(Real R1, Real R2) const {
        Real sum = R1 + R2;
        if (sum < 1.0e-30) return 0.0;
        return R1 * R2 / sum;
    }

    /**
     * @brief Compute rolling friction moment vector.
     *
     * M_roll = mu_roll * F_n * R_eff * (-omega_roll / |omega_roll|)
     *
     * The moment opposes the rolling angular velocity. If angular velocity
     * is zero, the moment is zero.
     *
     * @param normal_force Normal contact force magnitude [N]
     * @param R1           Radius of body 1 [m]
     * @param R2           Radius of body 2 [m]
     * @param omega_roll   Rolling angular velocity vector [3] [rad/s]
     * @param M_roll       Output rolling moment vector [3] [N*m]
     */
    KOKKOS_INLINE_FUNCTION
    void rolling_moment(Real normal_force, Real R1, Real R2,
                        const Real omega_roll[3], Real M_roll[3]) const {
        Real R_eff = effective_radius(R1, R2);
        Real mag = wave28_detail::norm3(omega_roll);

        if (mag < 1.0e-30 || normal_force <= 0.0) {
            M_roll[0] = M_roll[1] = M_roll[2] = 0.0;
            return;
        }

        Real moment_mag = mu_roll_ * normal_force * R_eff;
        // Direction opposes angular velocity
        M_roll[0] = -moment_mag * omega_roll[0] / mag;
        M_roll[1] = -moment_mag * omega_roll[1] / mag;
        M_roll[2] = -moment_mag * omega_roll[2] / mag;
    }

    /**
     * @brief Compute spin friction moment vector (about contact normal).
     *
     * M_spin = mu_spin * F_n * R_eff * (-omega_spin_hat * n)
     *
     * The spin component is the projection of angular velocity onto the
     * contact normal. The moment opposes the spin.
     *
     * @param normal_force Normal contact force magnitude [N]
     * @param R1           Radius of body 1 [m]
     * @param R2           Radius of body 2 [m]
     * @param omega_spin   Spin angular velocity magnitude [rad/s]
     * @param normal       Contact normal direction [3] (unit vector)
     * @param M_spin       Output spin moment vector [3] [N*m]
     */
    KOKKOS_INLINE_FUNCTION
    void spin_moment(Real normal_force, Real R1, Real R2,
                     Real omega_spin, const Real normal[3],
                     Real M_spin[3]) const {
        Real R_eff = effective_radius(R1, R2);

        if (std::abs(omega_spin) < 1.0e-30 || normal_force <= 0.0) {
            M_spin[0] = M_spin[1] = M_spin[2] = 0.0;
            return;
        }

        Real moment_mag = mu_spin_ * normal_force * R_eff;
        // Direction opposes spin (along normal)
        Real sign = (omega_spin > 0.0) ? -1.0 : 1.0;
        M_spin[0] = sign * moment_mag * normal[0];
        M_spin[1] = sign * moment_mag * normal[1];
        M_spin[2] = sign * moment_mag * normal[2];
    }

    /**
     * @brief Compute total resistance moment combining rolling and spin.
     *
     * The full angular velocity omega is decomposed into:
     *   - omega_spin = (omega . n) * n   (component along normal)
     *   - omega_roll = omega - omega_spin (component tangential to contact)
     *
     * Total moment: M_total = M_roll + M_spin
     *
     * @param normal_force Normal contact force magnitude [N]
     * @param R1           Radius of body 1 [m]
     * @param R2           Radius of body 2 [m]
     * @param omega        Full angular velocity vector [3] [rad/s]
     * @param normal       Contact normal direction [3] (unit vector)
     * @param M_total      Output total resistance moment [3] [N*m]
     */
    KOKKOS_INLINE_FUNCTION
    void total_resistance_moment(Real normal_force, Real R1, Real R2,
                                 const Real omega[3], const Real normal[3],
                                 Real M_total[3]) const {
        // Decompose omega into spin (normal) and roll (tangential) components
        Real omega_n = wave28_detail::dot3(omega, normal);

        // omega_roll = omega - omega_n * normal
        Real omega_roll[3];
        omega_roll[0] = omega[0] - omega_n * normal[0];
        omega_roll[1] = omega[1] - omega_n * normal[1];
        omega_roll[2] = omega[2] - omega_n * normal[2];

        // Rolling moment
        Real M_roll[3];
        rolling_moment(normal_force, R1, R2, omega_roll, M_roll);

        // Spin moment
        Real M_spin[3];
        spin_moment(normal_force, R1, R2, omega_n, normal, M_spin);

        // Total
        M_total[0] = M_roll[0] + M_spin[0];
        M_total[1] = M_roll[1] + M_spin[1];
        M_total[2] = M_roll[2] + M_spin[2];
    }

    /**
     * @brief Compute the couple force from a moment at a given arm length.
     *
     * F_couple = M / arm_length
     *
     * @param moment_mag Moment magnitude [N*m]
     * @param arm_length Distance between couple force application points [m]
     * @return Couple force magnitude [N]
     */
    KOKKOS_INLINE_FUNCTION
    Real couple_force(Real moment_mag, Real arm_length) const {
        if (arm_length < 1.0e-30) return 0.0;
        return moment_mag / arm_length;
    }

    /**
     * @brief Compute effective rolling resistance torque magnitude.
     *
     * Convenience method returning scalar magnitude:
     *   |M| = mu_roll * F_n * R_eff
     *
     * @param normal_force Normal force [N]
     * @param R1           Radius of body 1 [m]
     * @param R2           Radius of body 2 [m]
     * @return Rolling moment magnitude [N*m]
     */
    KOKKOS_INLINE_FUNCTION
    Real rolling_moment_magnitude(Real normal_force, Real R1, Real R2) const {
        return mu_roll_ * normal_force * effective_radius(R1, R2);
    }
};

// ============================================================================
// 28e: IntersectionAwareContact — Initial Overlap Resolution
// ============================================================================

/**
 * @brief Contact algorithm that detects and gradually resolves initial mesh
 *        overlaps (penetrations at t=0).
 *
 * In many practical simulations, the initial mesh configuration has
 * pre-existing penetrations between contacting surfaces. Instantaneous
 * correction would inject large forces and destabilize the simulation.
 *
 * This algorithm:
 * 1. Detects initial overlaps (negative gaps) at t=0
 * 2. Gradually pushes nodes apart over N_ramp time steps
 * 3. Limits correction forces to prevent energy injection
 * 4. Tracks injected energy for conservation monitoring
 *
 * Ramp function:
 *   correction(step) = initial_overlap * min(step / N_ramp, 1.0)
 *
 * Force limiter:
 *   F_correction = min(stiffness * overlap, F_max)
 */
class IntersectionAwareContact {
public:
    int ramp_steps_;           ///< Number of steps for gradual resolution
    Real max_correction_force_; ///< Maximum correction force [N]

    /**
     * @brief Default constructor.
     */
    IntersectionAwareContact() : ramp_steps_(100), max_correction_force_(1.0e10) {}

    /**
     * @brief Construct with ramp parameters.
     * @param ramp_steps           Number of steps for gradual correction
     * @param max_correction_force Maximum correction force [N]
     */
    IntersectionAwareContact(int ramp_steps, Real max_correction_force)
        : ramp_steps_(ramp_steps), max_correction_force_(max_correction_force) {}

    /**
     * @brief Detect initial overlap between two nodes.
     *
     * Overlap is defined as the negative gap component: if gap < 0,
     * overlap = -gap (positive value representing penetration depth).
     *
     * @param node_a  Position of node A [3]
     * @param node_b  Position of node B [3]
     * @param normal  Contact normal from B to A [3] (unit vector)
     * @param gap     Signed gap (positive = separated, negative = penetration)
     * @return Overlap magnitude (positive if penetrating, 0 otherwise)
     */
    KOKKOS_INLINE_FUNCTION
    Real detect_initial_overlap(const Real node_a[3], const Real node_b[3],
                                const Real normal[3], Real gap) const {
        (void)node_a; (void)node_b; (void)normal;
        if (gap < 0.0) {
            return -gap;  // Return positive overlap depth
        }
        return 0.0;
    }

    /**
     * @brief Compute the correction displacement at a given time step.
     *
     * The correction ramps linearly from zero to the full initial overlap
     * over ramp_steps_ time steps:
     *
     *   correction = initial_overlap * min(step / N_ramp, 1.0)
     *
     * @param initial_overlap  Initial overlap magnitude (positive) [m]
     * @param current_step     Current time step number
     * @return Correction displacement to apply [m]
     */
    KOKKOS_INLINE_FUNCTION
    Real correction_displacement(Real initial_overlap, int current_step) const {
        if (initial_overlap <= 0.0) return 0.0;
        if (current_step <= 0) return 0.0;

        Real fraction = static_cast<Real>(current_step) / static_cast<Real>(ramp_steps_);
        if (fraction > 1.0) fraction = 1.0;

        return initial_overlap * fraction;
    }

    /**
     * @brief Compute the clamped correction force.
     *
     * F = min(stiffness * overlap, max_force)
     *
     * The force is clamped to prevent excessive energy injection.
     *
     * @param overlap    Current overlap (positive) [m]
     * @param stiffness  Contact stiffness [N/m]
     * @param max_force  Maximum allowable force [N]
     * @return Clamped correction force [N]
     */
    KOKKOS_INLINE_FUNCTION
    Real correction_force(Real overlap, Real stiffness, Real max_force) const {
        if (overlap <= 0.0) return 0.0;
        Real force = stiffness * overlap;
        if (force > max_force) force = max_force;
        return force;
    }

    /**
     * @brief Compute energy injected by the correction.
     *
     * E = F * d (work done by correction force over displacement)
     *
     * @param force        Correction force [N]
     * @param displacement Correction displacement [m]
     * @return Injected energy [J]
     */
    KOKKOS_INLINE_FUNCTION
    Real energy_injected(Real force, Real displacement) const {
        return force * displacement;
    }

    /**
     * @brief Check if the ramp is complete.
     *
     * @param current_step Current time step number
     * @return true if current_step >= ramp_steps_
     */
    KOKKOS_INLINE_FUNCTION
    bool is_ramp_complete(int current_step) const {
        return current_step >= ramp_steps_;
    }

    /**
     * @brief Compute the ramp fraction at a given step.
     *
     * @param current_step Current time step number
     * @return Fraction in [0, 1]
     */
    KOKKOS_INLINE_FUNCTION
    Real ramp_fraction(int current_step) const {
        if (current_step <= 0) return 0.0;
        Real frac = static_cast<Real>(current_step) / static_cast<Real>(ramp_steps_);
        if (frac > 1.0) frac = 1.0;
        return frac;
    }

    /**
     * @brief Apply correction to separate two overlapping nodes.
     *
     * Each node is moved by half the correction displacement along the normal
     * (node_a in +normal direction, node_b in -normal direction).
     *
     * @param node_a          Position of node A [3]
     * @param node_b          Position of node B [3]
     * @param normal          Contact normal from B to A [3] (unit vector)
     * @param correction_disp Total correction displacement [m]
     * @param updated_a       Updated position of A [3]
     * @param updated_b       Updated position of B [3]
     */
    KOKKOS_INLINE_FUNCTION
    void apply_correction(const Real node_a[3], const Real node_b[3],
                          const Real normal[3], Real correction_disp,
                          Real updated_a[3], Real updated_b[3]) const {
        Real half_disp = 0.5 * correction_disp;
        updated_a[0] = node_a[0] + half_disp * normal[0];
        updated_a[1] = node_a[1] + half_disp * normal[1];
        updated_a[2] = node_a[2] + half_disp * normal[2];

        updated_b[0] = node_b[0] - half_disp * normal[0];
        updated_b[1] = node_b[1] - half_disp * normal[1];
        updated_b[2] = node_b[2] - half_disp * normal[2];
    }

    /**
     * @brief Compute cumulative energy injected over multiple contacts.
     *
     * @param forces       Array of correction forces [N]
     * @param displacements Array of correction displacements [m]
     * @param num_contacts  Number of contacts
     * @return Total energy injected [J]
     */
    KOKKOS_INLINE_FUNCTION
    Real total_energy_injected(const Real forces[], const Real displacements[],
                               int num_contacts) const {
        Real total = 0.0;
        for (int i = 0; i < num_contacts; ++i) {
            total += energy_injected(forces[i], displacements[i]);
        }
        return total;
    }

    /**
     * @brief Reset ramp parameters.
     * @param ramp_steps New ramp step count
     * @param max_force  New maximum correction force
     */
    void reset(int ramp_steps, Real max_force) {
        ramp_steps_ = ramp_steps;
        max_correction_force_ = max_force;
    }
};

// ============================================================================
// Wave 28 Contact Data Structures
// ============================================================================

/**
 * @brief Configuration for a Wave 28 contact interface.
 */
struct Wave28ContactConfig {
    /// Contact algorithm type
    enum class Algorithm {
        CurvedSegment,
        ThermalResistance,
        Wear,
        RollingResistance,
        IntersectionAware
    };

    Algorithm algorithm = Algorithm::CurvedSegment;

    // Curved segment parameters
    int newton_max_iter = 20;
    Real newton_tol = 1.0e-12;

    // Thermal parameters
    Real thermal_h_c0 = 0.0;
    Real thermal_p_ref = 1.0e6;
    Real thermal_exponent = 0.7;
    Real thermal_k_gas = 0.025;
    Real thermal_epsilon1 = 0.8;
    Real thermal_epsilon2 = 0.8;

    // Wear parameters
    Real wear_K = 1.0e-4;
    Real wear_hardness = 1.0e9;

    // Rolling resistance parameters
    Real rolling_mu_roll = 0.01;
    Real rolling_mu_spin = 0.005;

    // Intersection-aware parameters
    int ramp_steps = 100;
    Real max_correction_force = 1.0e10;
};

/**
 * @brief State tracking for Wave 28 contact algorithms.
 */
struct Wave28ContactState {
    /// Wear state
    std::vector<Real> wear_depths;     ///< Accumulated wear depth per contact [m]
    std::vector<Real> contact_areas;   ///< Contact areas per contact [m^2]

    /// Intersection-aware state
    std::vector<Real> initial_overlaps; ///< Detected overlaps at t=0 [m]
    Real total_energy_injected = 0.0;   ///< Cumulative energy from overlap correction [J]
    int current_step = 0;               ///< Current time step

    /// Thermal state
    std::vector<Real> surface_temps_1;  ///< Temperature field on surface 1 [K]
    std::vector<Real> surface_temps_2;  ///< Temperature field on surface 2 [K]

    /**
     * @brief Initialize state arrays.
     * @param num_contacts Number of contact points
     */
    void initialize(int num_contacts) {
        wear_depths.assign(num_contacts, 0.0);
        contact_areas.assign(num_contacts, 0.0);
        initial_overlaps.assign(num_contacts, 0.0);
        surface_temps_1.assign(num_contacts, 300.0);
        surface_temps_2.assign(num_contacts, 300.0);
        total_energy_injected = 0.0;
        current_step = 0;
    }

    /**
     * @brief Advance to next time step.
     */
    void advance_step() { ++current_step; }
};

/**
 * @brief Factory function to create a CurvedSegmentContact from config.
 */
inline CurvedSegmentContact make_curved_contact(const Wave28ContactConfig& cfg) {
    Real ref[3] = {0.0, 0.0, 1.0};
    return CurvedSegmentContact(cfg.newton_max_iter, cfg.newton_tol, ref);
}

/**
 * @brief Factory function to create a ThermalContactResistance from config.
 */
inline ThermalContactResistance make_thermal_contact(const Wave28ContactConfig& cfg) {
    return ThermalContactResistance(cfg.thermal_h_c0, cfg.thermal_p_ref,
                                    cfg.thermal_exponent, cfg.thermal_k_gas,
                                    cfg.thermal_epsilon1, cfg.thermal_epsilon2);
}

/**
 * @brief Factory function to create a WearModelContact from config.
 */
inline WearModelContact make_wear_contact(const Wave28ContactConfig& cfg) {
    return WearModelContact(cfg.wear_K, cfg.wear_hardness);
}

/**
 * @brief Factory function to create a RollingResistanceContact from config.
 */
inline RollingResistanceContact make_rolling_contact(const Wave28ContactConfig& cfg) {
    return RollingResistanceContact(cfg.rolling_mu_roll, cfg.rolling_mu_spin);
}

/**
 * @brief Factory function to create an IntersectionAwareContact from config.
 */
inline IntersectionAwareContact make_intersection_contact(const Wave28ContactConfig& cfg) {
    return IntersectionAwareContact(cfg.ramp_steps, cfg.max_correction_force);
}

} // namespace fem
} // namespace nxs
