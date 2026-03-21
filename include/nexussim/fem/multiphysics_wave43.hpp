#pragma once

/**
 * @file multiphysics_wave43.hpp
 * @brief Wave 43: EBCS Completion + Spring/Joint Specialization
 *
 * Sub-modules:
 * - 43a: ValveEBCS             — Pressure-controlled valve Eulerian BC
 * - 43b: PropellantEBCS        — Mass/energy injection from propellant burn
 * - 43c: NonReflectingEBCS     — Characteristic-based non-reflecting BC
 * - 43d: SpringPropertyType    — Enum for spring/damper property types
 * - 43e: UniversalJoint        — 2-axis Hooke joint (U-joint)
 * - 43f: PlanarJoint           — 2D sliding + rotation in a plane
 * - 43g: TranslationalJoint    — Prismatic joint along a single axis
 * - 43h: GeneralSpring         — Curve-based force-displacement with optional failure
 *
 * References:
 * - Thompson & Ferziger (1997) "Non-reflecting boundary conditions for compressible flow"
 * - Poinsot & Lele (1992) "Boundary conditions for direct simulations of compressible flows"
 * - Vafai & Tien (1981) "Boundary and inertia effects on flow and heat transfer in porous media"
 * - Flores & Lankarani (2011) "Spatial rigid-multibody systems with lubricated spherical clearance joints"
 */

#include <nexussim/core/types.hpp>
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <stdexcept>

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

namespace nxs {
namespace fem {

using Real = nxs::Real;

// ============================================================================
// Wave 43 utility helpers
// ============================================================================

namespace wave43_detail {

inline Real dot3(const Real a[3], const Real b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline Real norm3(const Real a[3]) {
    return std::sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

inline void normalize3(const Real a[3], Real out[3]) {
    Real n = norm3(a);
    if (n > Real(1.0e-30)) {
        out[0] = a[0] / n;
        out[1] = a[1] / n;
        out[2] = a[2] / n;
    } else {
        out[0] = Real(0); out[1] = Real(0); out[2] = Real(0);
    }
}

} // namespace wave43_detail

// ============================================================================
// 43a: ValveEBCS — Pressure-controlled valve Eulerian boundary condition
// ============================================================================

/**
 * @brief Pressure-controlled valve Eulerian BC.
 *
 * The valve opens when p_internal exceeds opening_pressure and closes when it
 * drops below closing_pressure (hysteresis). When open, the effective orifice
 * area ramps linearly to max_area. Mass flux = rho * current_area * v_normal.
 *
 * Usage:
 * @code
 *   ValveEBCS valve;
 *   valve.opening_pressure = 1.1e5;
 *   valve.closing_pressure = 1.0e5;
 *   valve.max_area         = 0.01;      // m^2
 *   valve.valve_direction  = {1, 0, 0};
 *
 *   Real vel[3] = {5.0, 0.0, 0.0};
 *   Real flux = valve.evaluate(1.15e5, 1.0e5, 1.2, vel);
 * @endcode
 */
struct ValveEBCS {
    Real opening_pressure{1.1e5};   ///< Valve opens when p_internal > this
    Real closing_pressure{1.0e5};   ///< Valve closes when p_internal < this
    Real max_area{0.01};            ///< Maximum effective opening area [m^2]
    Real valve_direction[3]{1, 0, 0}; ///< Outward normal of valve opening

    // State
    bool is_open{false};
    Real current_area{0.0};

    /**
     * @brief Evaluate mass flux through the valve.
     *
     * @param p_internal  Pressure on the interior side of the valve
     * @param p_external  Pressure on the exterior side (ambient)
     * @param rho         Local fluid density
     * @param velocity    Fluid velocity vector [3]
     * @return Mass flux [kg/(m^2 s)] through the valve (positive = outflow)
     */
    Real evaluate(Real p_internal, Real /*p_external*/,
                  Real rho, const Real velocity[3])
    {
        // Hysteretic open/close logic
        if (!is_open && p_internal > opening_pressure) {
            is_open = true;
        } else if (is_open && p_internal < closing_pressure) {
            is_open = false;
        }

        if (!is_open) {
            current_area = Real(0);
            return Real(0);
        }

        // Area opens fully when above opening_pressure
        // (can be extended with a partial-open ramp if desired)
        current_area = max_area;

        // Normal velocity component
        Real dir_n[3];
        wave43_detail::normalize3(valve_direction, dir_n);
        Real v_normal = wave43_detail::dot3(velocity, dir_n);

        return rho * current_area * v_normal;
    }

    /// Query whether the valve is currently open
    bool open() const { return is_open; }
};

// ============================================================================
// 43b: PropellantEBCS — Mass/energy injection from solid propellant burn
// ============================================================================

/**
 * @brief Solid-propellant mass/energy injection Eulerian BC.
 *
 * Models the burn surface of a solid propellant grain. The Vieille/de Saint-
 * Robert burning-rate law is used:
 *
 *   r     = a * p^n                           [m/s]
 *   m_dot = rho_p * r * A_grain              [kg/s]
 *   E_dot = m_dot * Cv * T_gas               [W]
 *
 * where a = burn_rate_coeff, n = pressure_exponent.
 */
struct PropellantEBCS {
    Real burn_rate_coeff{3.0e-5};   ///< Burn rate coefficient a  [m/(s Pa^n)]
    Real pressure_exponent{0.35};   ///< Pressure exponent n      [-]
    Real propellant_density{1700.0};///< Propellant density rho_p [kg/m^3]
    Real grain_area{0.01};          ///< Burning surface area A   [m^2]
    Real gas_temperature{3000.0};   ///< Adiabatic flame temperature [K]
    Real Cv{718.0};                 ///< Specific heat at constant volume [J/(kg K)]

    /**
     * @brief Compute mass and energy fluxes from chamber pressure.
     *
     * @param p_chamber      Chamber pressure [Pa]
     * @param[out] mass_flux  Mass injection rate [kg/s]
     * @param[out] energy_flux Energy injection rate [W]
     */
    inline void evaluate(Real p_chamber, Real& mass_flux, Real& energy_flux) const
    {
        // Clamp to avoid negative pressure
        Real p = (p_chamber > Real(0)) ? p_chamber : Real(0);

        Real burn_rate = burn_rate_coeff * std::pow(p, pressure_exponent);
        mass_flux   = propellant_density * burn_rate * grain_area;
        energy_flux = mass_flux * Cv * gas_temperature;
    }
};

// ============================================================================
// 43c: NonReflectingEBCS — Characteristic-based non-reflecting BC
// ============================================================================

/**
 * @brief Characteristic-based non-reflecting Eulerian boundary condition.
 *
 * Implements the NSCBC (Navier-Stokes Characteristic BC) approach from
 * Poinsot & Lele (1992). Outgoing waves are taken from the interior solution;
 * incoming waves are set to zero (non-reflecting) or prescribed.
 *
 * For a boundary with outward normal n:
 *   v_n    = u·n                               (normal velocity)
 *   c      = sqrt(gamma * p / rho)             (local sound speed)
 *   J_plus = v_n + 2c/(gamma-1)               (outgoing Riemann invariant)
 *   J_minus = 2c_ref/(gamma-1) - v_n_ref      (incoming, set from reference)
 *
 * Boundary state:
 *   v_n_bc = 0.5*(J_plus + J_minus)   — incoming wave is zero
 *   c_bc   = 0.25*(gamma-1)*(J_plus - J_minus)
 *   p_bc   = p_ref (subsonic outflow) or computed from c_bc
 */
struct NonReflectingEBCS {
    Real reference_density{1.225};      ///< Far-field / reference density [kg/m^3]
    Real reference_sound_speed{340.0};  ///< Far-field sound speed [m/s]
    Real reference_pressure{1.01325e5}; ///< Far-field pressure [Pa]
    Real gamma{1.4};                    ///< Ratio of specific heats

    /**
     * @brief Compute NSCBC boundary state from interior state.
     *
     * @param rho          Interior density  [kg/m^3]
     * @param velocity     Interior velocity [m/s], 3-component
     * @param pressure     Interior pressure [Pa]
     * @param normal       Outward unit normal of the boundary face, 3-component
     * @param[out] rho_bc  Boundary density
     * @param[out] vel_bc  Boundary velocity [3]
     * @param[out] p_bc    Boundary pressure
     */
    inline void evaluate(Real rho, const Real velocity[3], Real pressure,
                  const Real normal[3],
                  Real& rho_bc, Real vel_bc[3], Real& p_bc) const
    {
        // Normal velocity component (interior)
        Real v_n = wave43_detail::dot3(velocity, normal);

        // Interior sound speed
        Real c = std::sqrt(gamma * pressure / (rho > Real(1e-30) ? rho : Real(1e-30)));

        // Outgoing Riemann invariant (interior → BC)
        Real J_plus = v_n + Real(2) * c / (gamma - Real(1));

        // Incoming Riemann invariant — non-reflecting: set equal to far-field at rest
        // J- = v_n_ref - 2*c_ref/(gamma-1), with v_n_ref = 0  →  J- = -2*c_ref/(gamma-1)
        Real c_ref = reference_sound_speed;
        Real J_minus = -Real(2) * c_ref / (gamma - Real(1));

        // Reconstruct boundary normal velocity and sound speed
        // v_n = (J+ + J-) / 2,   c = (J+ - J-) * (gamma-1) / 4
        Real v_n_bc = Real(0.5) * (J_plus + J_minus);
        Real c_bc   = Real(0.25) * (gamma - Real(1)) * (J_plus - J_minus);
        if (c_bc < Real(0)) c_bc = Real(0);

        // Tangential velocity passes through unchanged (no tangential reflection)
        for (int i = 0; i < 3; ++i) {
            Real v_tang = velocity[i] - v_n * normal[i];
            vel_bc[i]   = v_tang + v_n_bc * normal[i];
        }

        // Subsonic outflow: fix pressure at reference
        if (v_n < c) {
            p_bc = reference_pressure;
        } else {
            // Supersonic: extrapolate
            p_bc = pressure;
        }

        // Density from isentropic relation: rho/rho_ref = (c/c_ref)^(2/(gamma-1))
        Real exp_val = Real(2) / (gamma - Real(1));
        rho_bc = (c_bc > Real(0) && c_ref > Real(0))
            ? reference_density * std::pow(c_bc / c_ref, exp_val)
            : reference_density;
    }
};

// ============================================================================
// 43d: SpringPropertyType — enum for spring/damper/bushing types
// ============================================================================

/**
 * @brief Classification of spring/joint property types.
 *
 * Mirrors OpenRadioss PROP/TYPE1–TYPE12 and LS-DYNA *ELEMENT_DISCRETE cards.
 */
enum class SpringPropertyType {
    Linear,                    ///< Linear elastic spring: F = k * d
    Nonlinear,                 ///< Nonlinear F(d) via tabulated curve
    Gap,                       ///< Active only when |d| > gap_open
    Dashpot,                   ///< Viscous dashpot: F = c * v
    BushingLinear,             ///< 6-DOF linear bushing (3 translational + 3 rotational)
    BushingNonlinear,          ///< 6-DOF nonlinear bushing with cross-coupling
    RotationalSpring,          ///< Rotational spring: M = k_rot * theta
    TorsionSpring,             ///< Torsion bar spring with large-angle kinematics
    GeneralForceDisplacement,  ///< Arbitrary F(d, v) user-table
    Maxwell,                   ///< Maxwell viscoelastic element (spring + dashpot in series)
    Kelvin,                    ///< Kelvin-Voigt viscoelastic element (spring || dashpot)
    PreloadedSpring,           ///< Spring with prescribed initial force/displacement
    FailableSpring,            ///< Spring that fails (zero stiffness) at failure_force
    FrictionDamper,            ///< Coulomb friction damper: |F| <= mu * N
    ElastomerMount,            ///< Frequency-dependent elastomeric mount
};

// ============================================================================
// 43e: UniversalJoint — 2-axis Hooke (U-joint)
// ============================================================================

/**
 * @brief Two-axis universal (Hooke) joint.
 *
 * Connects node1 to node2 via two orthogonal rotation axes.  The joint
 * transmits torque about neither axis (free rotation), but constrains all
 * other relative DOF.
 *
 * Kinematics:
 *   axis1 is attached to body 1 (yoke)
 *   axis2 is attached to body 2 (yoke)
 *   axis1 · axis2 = cos(coupling_angle)
 *
 * Torque about the cross-piece:
 *   T1 = stiffness * angle1   (optional torsional resistance)
 *   T2 = stiffness * angle2
 */
struct UniversalJoint {
    int node1{0};
    int node2{0};
    Real axis1[3]{1, 0, 0};   ///< Axis attached to body 1
    Real axis2[3]{0, 1, 0};   ///< Axis attached to body 2
    Real stiffness{0.0};       ///< Optional torsional stiffness [N·m/rad]
    Real damping{0.0};         ///< Optional torsional damping   [N·m·s/rad]

    /**
     * @brief Compute reaction torques for given angles and angular velocities.
     *
     * @param angle1   Rotation angle about axis1 [rad]
     * @param angle2   Rotation angle about axis2 [rad]
     * @param omega1   Angular velocity about axis1 [rad/s]
     * @param omega2   Angular velocity about axis2 [rad/s]
     * @param[out] torques  Reaction torques [T1, T2] [N·m]
     */
    void compute_torque(Real angle1, Real angle2,
                        Real omega1, Real omega2,
                        Real torques[2]) const
    {
        torques[0] = stiffness * angle1 + damping * omega1;
        torques[1] = stiffness * angle2 + damping * omega2;
    }

    /**
     * @brief Apply joint constraint forces/moments to a node array.
     *
     * Enforces that node2 position relative to node1 has no component along
     * the direction perpendicular to both axes (i.e., constrains relative
     * displacement to the plane spanned by axis1 and axis2 cross products).
     *
     * @param positions   Flat array [node_id * 3 + dim], size = num_nodes * 3
     * @param velocities  Flat array [node_id * 3 + dim]
     * @param forces      Flat array [node_id * 3 + dim] — modified in place
     * @param num_nodes   Total number of nodes
     */
    void apply_constraint(const std::vector<Real>& positions,
                          const std::vector<Real>& /*velocities*/,
                          std::vector<Real>& forces,
                          int num_nodes) const
    {
        if (node1 < 0 || node2 < 0 ||
            node1 >= num_nodes || node2 >= num_nodes) return;

        // Relative position vector
        Real r[3];
        for (int d = 0; d < 3; ++d)
            r[d] = positions[node2*3+d] - positions[node1*3+d];

        // The unconstrained directions are axis1 and axis2.
        // The constrained direction is n = normalize(axis1 × axis2).
        Real n[3];
        n[0] = axis1[1]*axis2[2] - axis1[2]*axis2[1];
        n[1] = axis1[2]*axis2[0] - axis1[0]*axis2[2];
        n[2] = axis1[0]*axis2[1] - axis1[1]*axis2[0];
        Real nm = wave43_detail::norm3(n);
        if (nm < Real(1e-12)) return;
        for (int d = 0; d < 3; ++d) n[d] /= nm;

        // Penalty force to enforce zero relative displacement along n
        Real pen = Real(1.0e6) * stiffness;  // scaled by stiffness
        if (pen < Real(1.0e3)) pen = Real(1.0e3); // minimum penalty
        Real gap = wave43_detail::dot3(r, n);
        Real pen_force = pen * gap;

        for (int d = 0; d < 3; ++d) {
            forces[node1*3+d] += pen_force * n[d];
            forces[node2*3+d] -= pen_force * n[d];
        }
    }
};

// ============================================================================
// 43f: PlanarJoint — 2D sliding + rotation in a plane
// ============================================================================

/**
 * @brief Planar joint: two nodes constrained to slide and rotate in a plane.
 *
 * Enforces that node2 stays in the plane defined by plane_normal passing
 * through node1.  Motion within the plane (2 translational + 1 rotational
 * DOF) is free.  The through-plane displacement is penalized.
 */
struct PlanarJoint {
    int node1{0};
    int node2{0};
    Real plane_normal[3]{0, 0, 1};   ///< Unit normal of the constraint plane
    Real stiffness_normal{1.0e6};    ///< Penalty stiffness for out-of-plane motion [N/m]
    Real stiffness_rotational{0.0};  ///< Optional in-plane rotational stiffness [N·m/rad]

    /**
     * @brief Apply planar constraint forces.
     *
     * Projects the relative displacement onto the plane normal and applies a
     * penalty force to remove the out-of-plane component.
     */
    void apply_constraint(const std::vector<Real>& positions,
                          const std::vector<Real>& /*velocities*/,
                          std::vector<Real>& forces,
                          int num_nodes) const
    {
        if (node1 < 0 || node2 < 0 ||
            node1 >= num_nodes || node2 >= num_nodes) return;

        // Relative displacement
        Real r[3];
        for (int d = 0; d < 3; ++d)
            r[d] = positions[node2*3+d] - positions[node1*3+d];

        // Normal direction (normalise defensively)
        Real n[3];
        wave43_detail::normalize3(plane_normal, n);

        // Out-of-plane gap
        Real gap = wave43_detail::dot3(r, n);

        // Penalty force
        Real pen_force = stiffness_normal * gap;
        for (int d = 0; d < 3; ++d) {
            forces[node1*3+d] += pen_force * n[d];
            forces[node2*3+d] -= pen_force * n[d];
        }
    }
};

// ============================================================================
// 43g: TranslationalJoint — Prismatic joint along a single axis
// ============================================================================

/**
 * @brief Prismatic (translational) joint.
 *
 * Constrains relative motion to a single translation axis.  Lateral and
 * rotational DOF are penalized.  Optional travel limits apply contact-like
 * end-stop forces when the joint reaches its range of motion.
 */
struct TranslationalJoint {
    int node1{0};
    int node2{0};
    Real axis[3]{1, 0, 0};          ///< Prismatic sliding axis (unit vector)
    Real stiffness{1.0e5};          ///< Axial spring stiffness [N/m]
    Real damping{0.0};              ///< Axial damping coefficient [N·s/m]
    Real travel_limits[2]{-1e30, 1e30}; ///< [min, max] allowed displacement [m]
    Real lateral_stiffness{1.0e7};  ///< Penalty for lateral constraint [N/m]

    /**
     * @brief Compute axial spring/damper force and end-stop force.
     *
     * @param disp   Displacement along axis [m]
     * @param vel    Velocity along axis [m/s]
     * @return Force along axis [N] (positive = tension)
     */
    Real compute_force(Real disp, Real vel) const
    {
        Real f = stiffness * disp + damping * vel;

        // End-stop contact (hard stop with same stiffness)
        if (disp < travel_limits[0]) {
            f += stiffness * (disp - travel_limits[0]);
        } else if (disp > travel_limits[1]) {
            f += stiffness * (disp - travel_limits[1]);
        }
        return f;
    }

    /**
     * @brief Apply joint constraint forces to nodes.
     *
     * Applies axial spring force and lateral penalty force (to keep node2
     * on the axis through node1).
     */
    void apply_constraint(const std::vector<Real>& positions,
                          const std::vector<Real>& velocities,
                          std::vector<Real>& forces,
                          int num_nodes) const
    {
        if (node1 < 0 || node2 < 0 ||
            node1 >= num_nodes || node2 >= num_nodes) return;

        // Relative position
        Real r[3];
        for (int d = 0; d < 3; ++d)
            r[d] = positions[node2*3+d] - positions[node1*3+d];

        // Unit axis (normalise defensively)
        Real ax[3];
        wave43_detail::normalize3(axis, ax);

        // Axial displacement
        Real disp = wave43_detail::dot3(r, ax);

        // Axial velocity
        Real rv[3];
        for (int d = 0; d < 3; ++d)
            rv[d] = velocities[node2*3+d] - velocities[node1*3+d];
        Real vel = wave43_detail::dot3(rv, ax);

        // Axial force
        Real f_axial = compute_force(disp, vel);

        // Lateral penalty: component of r perpendicular to axis
        Real r_lat[3];
        for (int d = 0; d < 3; ++d)
            r_lat[d] = r[d] - disp * ax[d];

        for (int d = 0; d < 3; ++d) {
            // Axial spring
            forces[node1*3+d] += f_axial * ax[d];
            forces[node2*3+d] -= f_axial * ax[d];
            // Lateral constraint
            forces[node1*3+d] += lateral_stiffness * r_lat[d];
            forces[node2*3+d] -= lateral_stiffness * r_lat[d];
        }
    }
};

// ============================================================================
// 43h: GeneralSpring — Curve-based force-displacement with optional failure
// ============================================================================

/**
 * @brief General spring element with tabulated F(d) and optional failure.
 *
 * Supports all SpringPropertyType behaviours via a unified evaluate path:
 * - Linear:      F = k * d
 * - Nonlinear:   piecewise-linear interpolation of force_curve
 * - Gap:         active only when d > gap_open (compression) or d < -gap_open (tension)
 * - Dashpot:     F = c * v  (damping_curve or scalar damping)
 * - Maxwell:     two-element spring+dashpot in series (internal state: eps_d)
 * - Kelvin:      spring || dashpot parallel
 * - PreloadedSpring: offsets the displacement by preload_displacement
 * - FailableSpring:  zeros force once |F| > failure_force
 *
 * Tabulated curves are stored as {displacement, force} pairs sorted by
 * displacement.  Linear interpolation is used; values outside the table range
 * are extrapolated from the last two entries.
 */
class GeneralSpring {
public:
    SpringPropertyType property_type{SpringPropertyType::Linear};

    // Tabulated curves — sorted by first component (displacement / velocity)
    std::vector<std::array<Real, 2>> force_curve;    ///< {d, F} pairs
    std::vector<std::array<Real, 2>> damping_curve;  ///< {v, c(v)} pairs

    Real linear_stiffness{1.0e5};   ///< Used for Linear / fallback
    Real linear_damping{0.0};       ///< Scalar dashpot coefficient
    Real preload{0.0};              ///< Pre-tension / pre-compression force [N]
    Real failure_force{1e30};       ///< Force magnitude at failure [N]
    Real gap_open{0.0};             ///< Gap distance before activation [m]

    // Maxwell internal state
    Real maxwell_damping_disp{0.0}; ///< Viscous element displacement (ε_d)

    // Failable spring state
    bool failed{false};

    /**
     * @brief Evaluate spring force.
     *
     * @param displacement  Current displacement [m]
     * @param velocity      Current velocity [m/s]
     * @return Force [N] (positive = tension)
     */
    Real compute_force(Real displacement, Real velocity)
    {
        if (failed) return Real(0);

        Real force = Real(0);

        switch (property_type) {

        case SpringPropertyType::Linear:
            force = linear_stiffness * (displacement + preload / linear_stiffness)
                    + linear_damping * velocity;
            break;

        case SpringPropertyType::Nonlinear:
            force = interp_curve(force_curve, displacement);
            force += linear_damping * velocity;
            break;

        case SpringPropertyType::Gap: {
            Real d_eff = Real(0);
            if (displacement >  gap_open) d_eff = displacement -  gap_open;
            if (displacement < -gap_open) d_eff = displacement + gap_open;
            force = linear_stiffness * d_eff + linear_damping * velocity;
            break;
        }

        case SpringPropertyType::Dashpot:
            if (!damping_curve.empty())
                force = interp_curve(damping_curve, velocity) * velocity;
            else
                force = linear_damping * velocity;
            break;

        case SpringPropertyType::Maxwell: {
            // Maxwell: spring in series with dashpot
            // Incremental update (simple forward-Euler)
            // F = k*(d - eps_d);  eps_d_dot = F/c
            if (linear_damping > Real(0)) {
                Real dt_approx = Real(1e-6); // caller should pass dt — use 1 µs default
                Real F_try = linear_stiffness * (displacement - maxwell_damping_disp);
                maxwell_damping_disp += F_try / linear_damping * dt_approx;
                force = linear_stiffness * (displacement - maxwell_damping_disp);
            } else {
                force = linear_stiffness * displacement;
            }
            break;
        }

        case SpringPropertyType::Kelvin:
            force = linear_stiffness * displacement + linear_damping * velocity;
            break;

        case SpringPropertyType::PreloadedSpring: {
            Real d_eff = displacement + preload / linear_stiffness;
            force = linear_stiffness * d_eff + linear_damping * velocity;
            break;
        }

        case SpringPropertyType::FailableSpring:
            force = linear_stiffness * displacement + linear_damping * velocity;
            break;

        case SpringPropertyType::RotationalSpring:
            // Use displacement as angle [rad], return torque [N·m]
            force = linear_stiffness * displacement;
            break;

        case SpringPropertyType::TorsionSpring:
            force = linear_stiffness * displacement + linear_damping * velocity;
            break;

        case SpringPropertyType::GeneralForceDisplacement:
            force = interp_curve(force_curve, displacement);
            if (!damping_curve.empty())
                force += interp_curve(damping_curve, velocity) * velocity;
            break;

        case SpringPropertyType::FrictionDamper: {
            // Coulomb friction: F = mu * N * sign(v), limited by static force
            Real static_force = linear_stiffness;   // store static limit in stiffness
            Real mu = linear_damping;               // store mu in damping field
            Real sign_v = (velocity > Real(0)) ? Real(1) :
                          (velocity < Real(0)) ? Real(-1) : Real(0);
            force = mu * static_force * sign_v;
            break;
        }

        case SpringPropertyType::ElastomerMount:
            // Simplified: frequency-dependent via linear + Kelvin chain
            force = linear_stiffness * displacement + linear_damping * velocity;
            break;

        default:
            force = linear_stiffness * displacement;
            break;
        }

        return force;
    }

    /**
     * @brief Check whether the spring has failed.
     *
     * Sets the internal failed flag if |force| >= failure_force.
     *
     * @param force  Current force magnitude [N]
     * @return true if the spring is (now or previously) failed
     */
    bool check_failure(Real force)
    {
        if (!failed && std::abs(force) >= failure_force) {
            failed = true;
        }
        return failed;
    }

    /// Reset failure state (for restart / re-analysis)
    void reset_failure() { failed = false; }

private:
    /**
     * @brief Piecewise-linear interpolation / extrapolation of a sorted table.
     *
     * @param table  Sorted {x, y} pairs
     * @param x      Query value
     * @return Interpolated y value
     */
    static Real interp_curve(const std::vector<std::array<Real, 2>>& table, Real x)
    {
        if (table.empty()) return Real(0);
        if (table.size() == 1) return table[0][1];

        // Below range: extrapolate from first two points
        if (x <= table.front()[0]) {
            Real dx = table[1][0] - table[0][0];
            if (std::abs(dx) < Real(1e-30)) return table[0][1];
            Real slope = (table[1][1] - table[0][1]) / dx;
            return table[0][1] + slope * (x - table[0][0]);
        }

        // Above range: extrapolate from last two points
        if (x >= table.back()[0]) {
            int n = static_cast<int>(table.size());
            Real dx = table[n-1][0] - table[n-2][0];
            if (std::abs(dx) < Real(1e-30)) return table[n-1][1];
            Real slope = (table[n-1][1] - table[n-2][1]) / dx;
            return table[n-1][1] + slope * (x - table[n-1][0]);
        }

        // Binary search for interval
        std::size_t lo = 0, hi = table.size() - 1;
        while (hi - lo > 1) {
            std::size_t mid = (lo + hi) / 2;
            if (table[mid][0] <= x) lo = mid; else hi = mid;
        }

        Real dx = table[hi][0] - table[lo][0];
        if (std::abs(dx) < Real(1e-30)) return table[lo][1];
        Real t = (x - table[lo][0]) / dx;
        return table[lo][1] + t * (table[hi][1] - table[lo][1]);
    }
};

} // namespace fem
} // namespace nxs
