#pragma once

/**
 * @file loads_wave43.hpp
 * @brief Wave 43: 5 advanced load types missing from NexusSim base loads
 *
 * Load types:
 *   1. CentrifugalLoad       - Body force from rotation: F_i = rho * omega^2 * r_i
 *   2. CylindricalPressure   - Radial pressure on pipe/vessel surfaces
 *   3. FluidPressure         - Hydrostatic pressure: P = rho_fluid * g * depth
 *   4. LaserLoad             - Moving Gaussian heat source (welding/ablation)
 *   5. BoltPreload           - Assembly preload force along bolt axis
 *
 * References:
 * - OpenRadioss /engine/source/loads/
 * - Belytschko et al. "Nonlinear Finite Elements for Continua and Structures"
 */

#include <nexussim/core/types.hpp>
#include <nexussim/fem/load_curve.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <algorithm>

namespace nxs {
namespace fem {

// ============================================================================
// Wave 43 Load Type Enum
// ============================================================================

enum class LoadWave43Type {
    CentrifugalLoad,      ///< Centrifugal body force from rotating reference frame
    CylindricalPressure,  ///< Radial pressure on cylindrical surfaces
    FluidPressure,        ///< Hydrostatic depth-dependent pressure
    LaserLoad,            ///< Moving Gaussian heat/ablation source
    BoltPreload           ///< Axial preload force distributed across bolt nodes
};

// ============================================================================
// Internal math helpers
// ============================================================================

namespace loads43_detail {

/// Normalize a 3-vector in place. Returns original length.
inline Real normalize3(Real v[3]) {
    Real len = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (len > 1.0e-30) {
        v[0] /= len; v[1] /= len; v[2] /= len;
    }
    return len;
}

/// Dot product of two 3-vectors.
inline Real dot3(const Real a[3], const Real b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/// Cross product: c = a x b
inline void cross3(const Real a[3], const Real b[3], Real c[3]) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

} // namespace loads43_detail

// ============================================================================
// 1. CentrifugalLoad
// ============================================================================

/**
 * @brief Centrifugal body force for rotating structures.
 *
 * In the rotating reference frame, each material point at distance r from the
 * rotation axis experiences an outward body force:
 *   f_i = rho * omega^2 * r_i
 * where r_i is the component of (x - axis_point) perpendicular to axis_direction.
 *
 * apply() accumulates per-node forces given node masses and positions.
 */
struct CentrifugalLoad {
    Real omega;              ///< Angular velocity [rad/s]
    Real axis_point[3];      ///< A point on the rotation axis
    Real axis_direction[3];  ///< Unit vector along rotation axis (normalized on set)
    int  load_curve_id;      ///< Optional time-scaling curve (-1 = constant)
    bool active;

    CentrifugalLoad()
        : omega(0.0), load_curve_id(-1), active(true) {
        axis_point[0] = axis_point[1] = axis_point[2] = 0.0;
        axis_direction[0] = 0.0; axis_direction[1] = 0.0; axis_direction[2] = 1.0;
    }

    /// Normalize axis_direction (call after setting it).
    void normalize_axis() {
        loads43_detail::normalize3(axis_direction);
    }

    /**
     * @brief Compute radial distance vector from axis for a single node.
     *
     * Given node position x[3], returns the vector from the axis to x
     * (i.e., the component of (x - axis_point) perpendicular to axis_direction).
     */
    void radial_vector(const Real x[3], Real r_vec[3]) const {
        // d = x - axis_point
        Real d[3] = { x[0] - axis_point[0],
                      x[1] - axis_point[1],
                      x[2] - axis_point[2] };
        // projection onto axis
        Real proj = loads43_detail::dot3(d, axis_direction);
        // r_vec = d - proj * axis_direction
        r_vec[0] = d[0] - proj * axis_direction[0];
        r_vec[1] = d[1] - proj * axis_direction[1];
        r_vec[2] = d[2] - proj * axis_direction[2];
    }

    /**
     * @brief Apply centrifugal load to all nodes.
     *
     * @param curve_scale  Time-dependent scale factor (from load curve, already evaluated)
     * @param num_nodes    Number of nodes
     * @param positions    Node positions [3*num_nodes]
     * @param masses       Node masses [num_nodes]
     * @param forces       Force accumulator [3*num_nodes] (modified in place)
     */
    void apply(Real curve_scale,
               std::size_t num_nodes,
               const Real* positions,
               const Real* masses,
               Real* forces) const {
        if (!active) return;
        Real w2 = omega * omega * curve_scale;

        for (std::size_t i = 0; i < num_nodes; ++i) {
            const Real* xi = positions + 3*i;
            Real r_vec[3];
            radial_vector(xi, r_vec);
            // F = m * omega^2 * r_vec  (outward radial body force)
            Real mi = masses[i];
            forces[3*i + 0] += mi * w2 * r_vec[0];
            forces[3*i + 1] += mi * w2 * r_vec[1];
            forces[3*i + 2] += mi * w2 * r_vec[2];
        }
    }
};

// ============================================================================
// 2. CylindricalPressure
// ============================================================================

/**
 * @brief Radial pressure load on cylindrical pipe or vessel surfaces.
 *
 * The pressure acts in the radially outward direction from the cylinder axis.
 * For each node in the node_set, the outward unit radial vector is computed
 * from the node position and axis geometry, then scaled by pressure magnitude.
 *
 * F_node = pressure * area_per_node * r_hat
 *
 * where r_hat is the unit vector perpendicular to the axis pointing outward.
 * If area_per_node is not provided per node, a uniform value area_per_node is used.
 */
struct CylindricalPressure {
    Real axis_point[3];      ///< A point on the cylinder axis
    Real axis_direction[3];  ///< Unit vector along cylinder axis (normalized on set)
    Real pressure;           ///< Pressure magnitude [Pa]
    Real area_per_node;      ///< Tributary area per node [m^2] (uniform simplification)
    int  load_curve_id;      ///< Optional time-scaling curve (-1 = constant)
    bool active;
    std::vector<Index> node_set; ///< Nodes on the cylindrical surface

    CylindricalPressure()
        : pressure(0.0), area_per_node(1.0), load_curve_id(-1), active(true) {
        axis_point[0] = axis_point[1] = axis_point[2] = 0.0;
        axis_direction[0] = 0.0; axis_direction[1] = 0.0; axis_direction[2] = 1.0;
    }

    void normalize_axis() {
        loads43_detail::normalize3(axis_direction);
    }

    /**
     * @brief Compute the outward radial unit vector at node position x.
     *
     * Returns false if the node is on the axis (degenerate case, r_hat left zero).
     */
    bool radial_unit_vector(const Real x[3], Real r_hat[3]) const {
        Real d[3] = { x[0] - axis_point[0],
                      x[1] - axis_point[1],
                      x[2] - axis_point[2] };
        Real proj = loads43_detail::dot3(d, axis_direction);
        r_hat[0] = d[0] - proj * axis_direction[0];
        r_hat[1] = d[1] - proj * axis_direction[1];
        r_hat[2] = d[2] - proj * axis_direction[2];
        Real len = loads43_detail::normalize3(r_hat);
        return len > 1.0e-14;
    }

    /**
     * @brief Apply cylindrical pressure to the node set.
     *
     * @param curve_scale  Evaluated load curve scale factor
     * @param positions    Node positions [3*num_nodes_total]
     * @param forces       Force accumulator [3*num_nodes_total]
     */
    void apply(Real curve_scale,
               const Real* positions,
               Real* forces) const {
        if (!active) return;
        Real p_eff = pressure * curve_scale * area_per_node;

        for (Index n : node_set) {
            const Real* xn = positions + 3*n;
            Real r_hat[3];
            if (radial_unit_vector(xn, r_hat)) {
                forces[3*n + 0] += p_eff * r_hat[0];
                forces[3*n + 1] += p_eff * r_hat[1];
                forces[3*n + 2] += p_eff * r_hat[2];
            }
            // On-axis: no radial direction — skip (zero force, physically correct)
        }
    }
};

// ============================================================================
// 3. FluidPressure
// ============================================================================

/**
 * @brief Hydrostatic fluid pressure on submerged element faces.
 *
 * Pressure varies with depth below a free surface:
 *   P(x) = rho_fluid * g * max(0, depth(x))
 * where depth is measured in the direction opposite to free_surface_normal:
 *   depth(x) = (free_surface_height - x . free_surface_normal)
 *
 * The resulting nodal force is:
 *   F_node = P(x_node) * area_per_node * (-free_surface_normal)
 *            (pressure acts inward, i.e. opposite to outward normal of free surface)
 *
 * free_surface_normal should point UPWARD (out of the fluid).
 */
struct FluidPressure {
    Real fluid_density;           ///< Fluid mass density [kg/m^3]
    Real gravity;                 ///< Gravitational acceleration magnitude [m/s^2]
    Real free_surface_height;     ///< Signed height of free surface: x.n = free_surface_height
    Real free_surface_normal[3];  ///< Unit normal pointing out of fluid (upward)
    Real area_per_node;           ///< Tributary area per node [m^2]
    int  load_curve_id;           ///< Optional time-scaling curve (-1 = constant)
    bool active;
    std::vector<Index> node_set;  ///< Nodes on submerged faces

    FluidPressure()
        : fluid_density(1000.0), gravity(9.81)
        , free_surface_height(0.0), area_per_node(1.0)
        , load_curve_id(-1), active(true) {
        free_surface_normal[0] = 0.0;
        free_surface_normal[1] = 0.0;
        free_surface_normal[2] = 1.0;  // Default: Z is up
    }

    void normalize_normal() {
        loads43_detail::normalize3(free_surface_normal);
    }

    /**
     * @brief Compute depth of a node below the free surface.
     *
     * Positive depth means the node is submerged.
     */
    Real depth_at(const Real x[3]) const {
        Real h = loads43_detail::dot3(x, free_surface_normal);
        return free_surface_height - h;
    }

    /**
     * @brief Apply hydrostatic pressure to node set.
     *
     * @param curve_scale  Evaluated load curve scale factor
     * @param positions    Node positions [3*num_nodes_total]
     * @param forces       Force accumulator [3*num_nodes_total]
     */
    void apply(Real curve_scale,
               const Real* positions,
               Real* forces) const {
        if (!active) return;

        for (Index n : node_set) {
            const Real* xn = positions + 3*n;
            Real d = depth_at(xn);
            if (d <= 0.0) continue;  // Above free surface — no hydrostatic load

            // Pressure magnitude at this depth
            Real p = fluid_density * gravity * d * curve_scale;
            // Force = P * A * (-free_surface_normal) — pressure pushes inward
            Real f = p * area_per_node;
            forces[3*n + 0] -= f * free_surface_normal[0];
            forces[3*n + 1] -= f * free_surface_normal[1];
            forces[3*n + 2] -= f * free_surface_normal[2];
        }
    }
};

// ============================================================================
// 4. LaserLoad
// ============================================================================

/**
 * @brief Moving Gaussian heat/ablation source for laser welding or processing.
 *
 * The surface heat flux follows a Gaussian distribution:
 *   Q(r) = (P_laser / (2 * pi * sigma^2)) * exp(-r^2 / (2 * sigma^2))
 * where r is the distance from the beam center projected onto the surface.
 *
 * The beam center moves with constant velocity:
 *   x_beam(t) = position + direction * speed * t
 *
 * Each node receives a nodal heat flux contribution:
 *   Q_node = absorption_coeff * Q(r_node) * area_per_node
 *
 * For structural coupling the heat flux can be converted to a "thermal force"
 * or stored as an energy input. Here it is added to the forces array in the
 * DOF corresponding to the thermal DOF (index 3, if present) or as a scalar
 * heat flux stored separately. This implementation adds the heat flux to a
 * user-supplied heat_flux array (same layout as forces: one value per node).
 */
struct LaserLoad {
    Real power;             ///< Laser power [W]
    Real beam_radius;       ///< Gaussian beam radius sigma [m]
    Real position[3];       ///< Initial beam center position [m]
    Real direction[3];      ///< Beam travel direction (unit vector)
    Real speed;             ///< Beam travel speed [m/s]
    Real absorption_coeff;  ///< Material absorptivity [0..1]
    Real area_per_node;     ///< Tributary surface area per node [m^2]
    int  load_curve_id;     ///< Optional power scaling curve (-1 = constant)
    bool active;
    std::vector<Index> node_set; ///< Nodes on irradiated surface

    LaserLoad()
        : power(0.0), beam_radius(1.0e-3)
        , speed(0.0), absorption_coeff(1.0), area_per_node(1.0)
        , load_curve_id(-1), active(true) {
        position[0] = position[1] = position[2] = 0.0;
        direction[0] = 1.0; direction[1] = 0.0; direction[2] = 0.0;
    }

    void normalize_direction() {
        loads43_detail::normalize3(direction);
    }

    /**
     * @brief Compute current beam center position at time t.
     */
    void beam_center(Real t, Real center[3]) const {
        center[0] = position[0] + direction[0] * speed * t;
        center[1] = position[1] + direction[1] * speed * t;
        center[2] = position[2] + direction[2] * speed * t;
    }

    /**
     * @brief Gaussian intensity at distance r from beam center.
     *
     * Q(r) = P / (2 * pi * sigma^2) * exp(-r^2 / (2 * sigma^2))
     */
    Real gaussian_intensity(Real r) const {
        Real sigma2 = beam_radius * beam_radius;
        if (sigma2 < 1.0e-60) return 0.0;
        Real peak = power / (2.0 * M_PI * sigma2);
        return peak * std::exp(-0.5 * r * r / sigma2);
    }

    /**
     * @brief Apply laser heat flux to node set.
     *
     * Writes to heat_flux[n] (one scalar per node, units: W/m^2 * m^2 = W).
     * If heat_flux is nullptr, writes to forces[3*n+0] as a proxy (for testing).
     *
     * @param time         Current simulation time
     * @param curve_scale  Evaluated power scale factor
     * @param positions    Node positions [3*num_nodes_total]
     * @param heat_flux    Per-node heat flux accumulator [num_nodes_total] (may be nullptr)
     * @param forces       Force accumulator [3*num_nodes_total] (fallback if heat_flux==nullptr)
     */
    void apply(Real time, Real curve_scale,
               const Real* positions,
               Real* heat_flux,
               Real* forces) const {
        if (!active) return;

        Real center[3];
        beam_center(time, center);

        for (Index n : node_set) {
            const Real* xn = positions + 3*n;
            // 3D distance from beam center
            Real dx = xn[0] - center[0];
            Real dy = xn[1] - center[1];
            Real dz = xn[2] - center[2];
            Real r = std::sqrt(dx*dx + dy*dy + dz*dz);

            Real q = absorption_coeff * gaussian_intensity(r) * area_per_node * curve_scale;

            if (heat_flux) {
                heat_flux[n] += q;
            } else if (forces) {
                // Fallback: store in first DOF (for testing/coupling purposes)
                forces[3*n + 0] += q;
            }
        }
    }
};

// ============================================================================
// 5. BoltPreload
// ============================================================================

/**
 * @brief Bolt preload (assembly preload force) along the bolt axis.
 *
 * Models the pretension in a fastener. The preload force is distributed
 * uniformly among the bolt cross-section nodes. The application has two phases:
 *
 *  Phase A (apply): Force increments are applied incrementally until the target
 *                   preload is reached.
 *  Phase B (lock):  Once locked, the bolt length is frozen — the preload force
 *                   is replaced by a displacement constraint. This is modeled here
 *                   by zeroing the applied force (the structural stiffness maintains
 *                   the resulting strain).
 *
 * In explicit dynamics, the preload is ramped over a user-defined ramp_time.
 * After ramp_time the bolt is considered locked.
 *
 * Force per node = preload_force / num_bolt_nodes * bolt_axis (half +, half -)
 * The sign convention: the "head" half of nodes gets force in +bolt_axis direction,
 * and the "nut" half gets force in -bolt_axis direction, creating the clamping effect.
 * For simplicity here, if a single node set is provided the force is split evenly
 * with direction determined by bolt_axis sign (+/-).
 *
 * If head_nodes and nut_nodes are both non-empty, forces are applied in opposite
 * directions along bolt_axis to create the clamping couple.
 */
struct BoltPreload {
    Real preload_force;         ///< Target preload force magnitude [N]
    Real bolt_axis[3];          ///< Unit vector along bolt axis (head -> nut)
    Real ramp_time;             ///< Time over which preload is ramped [s]
    bool locked;                ///< True once ramp is complete and bolt is locked
    bool active;
    int  load_curve_id;         ///< Optional preload scaling curve (-1 = use ramp_time)

    std::vector<Index> head_nodes; ///< Nodes at bolt head face (receive +axis force)
    std::vector<Index> nut_nodes;  ///< Nodes at nut/shank face (receive -axis force)

    BoltPreload()
        : preload_force(0.0), ramp_time(1.0e-3)
        , locked(false), active(true), load_curve_id(-1) {
        bolt_axis[0] = 0.0; bolt_axis[1] = 0.0; bolt_axis[2] = 1.0;
    }

    void normalize_axis() {
        loads43_detail::normalize3(bolt_axis);
    }

    /**
     * @brief Ramp factor in [0, 1] for time-based ramping (no curve).
     */
    Real ramp_factor(Real time) const {
        if (ramp_time <= 0.0) return 1.0;
        Real f = time / ramp_time;
        return (f >= 1.0) ? 1.0 : f;
    }

    /**
     * @brief Check and update locked state.
     *
     * Call each step. Once locked, the preload is held via structural stiffness
     * and this load contributes zero incremental force.
     */
    void update_lock_state(Real time) {
        if (!locked && time >= ramp_time) {
            locked = true;
        }
    }

    /**
     * @brief Apply bolt preload forces to head_nodes and nut_nodes.
     *
     * @param time         Current simulation time
     * @param curve_scale  Evaluated load curve scale factor
     * @param forces       Force accumulator [3*num_nodes_total]
     */
    void apply(Real time, Real curve_scale, Real* forces) const {
        if (!active) return;
        if (locked) return;  // Locked: no incremental force contribution

        Real scale = ramp_factor(time) * curve_scale;
        Real f_total = preload_force * scale;

        // Distribute equally among head nodes (force in +bolt_axis direction)
        if (!head_nodes.empty()) {
            Real f_node = f_total / static_cast<Real>(head_nodes.size());
            for (Index n : head_nodes) {
                forces[3*n + 0] += f_node * bolt_axis[0];
                forces[3*n + 1] += f_node * bolt_axis[1];
                forces[3*n + 2] += f_node * bolt_axis[2];
            }
        }

        // Distribute equally among nut nodes (force in -bolt_axis direction)
        if (!nut_nodes.empty()) {
            Real f_node = f_total / static_cast<Real>(nut_nodes.size());
            for (Index n : nut_nodes) {
                forces[3*n + 0] -= f_node * bolt_axis[0];
                forces[3*n + 1] -= f_node * bolt_axis[1];
                forces[3*n + 2] -= f_node * bolt_axis[2];
            }
        }
    }
};

// ============================================================================
// LoadWave43 — unified struct wrapping all 5 load types
// ============================================================================

/**
 * @brief Polymorphic-like wrapper for Wave 43 loads.
 *
 * Holds a type tag and one active load payload. Use the specific member
 * (centrifugal, cyl_pressure, fluid_pressure, laser, bolt_preload) based on type.
 */
struct LoadWave43 {
    LoadWave43Type type;
    int id;
    std::string name;

    CentrifugalLoad    centrifugal;
    CylindricalPressure cyl_pressure;
    FluidPressure      fluid_pressure;
    LaserLoad          laser;
    BoltPreload        bolt_preload;

    LoadWave43() : type(LoadWave43Type::CentrifugalLoad), id(0) {}
};

// ============================================================================
// LoadWave43Manager
// ============================================================================

/**
 * @brief Manager for Wave 43 advanced load types.
 *
 * Mirrors the interface of LoadManager. Call apply_all() each time step.
 */
class LoadWave43Manager {
public:
    LoadWave43Manager() = default;

    void set_curve_manager(LoadCurveManager* mgr) { curves_ = mgr; }

    LoadWave43& add_load(LoadWave43Type type, int id = 0) {
        loads_.emplace_back();
        auto& l = loads_.back();
        l.type = type;
        l.id = id;
        return l;
    }

    std::size_t num_loads() const { return loads_.size(); }
    LoadWave43& load(std::size_t i) { return loads_[i]; }

    /**
     * @brief Apply all Wave 43 loads at the current time step.
     *
     * @param time         Current simulation time
     * @param num_nodes    Total number of nodes
     * @param positions    Node positions [3*num_nodes]
     * @param masses       Node masses [num_nodes]
     * @param forces       External force accumulator [3*num_nodes]
     * @param heat_flux    Per-node heat flux [num_nodes] — may be nullptr
     */
    void apply_all(Real time,
                   std::size_t num_nodes,
                   const Real* positions,
                   const Real* masses,
                   Real* forces,
                   Real* heat_flux = nullptr) {
        for (auto& l : loads_) {
            Real cs = get_curve_scale(l, time);

            switch (l.type) {
                case LoadWave43Type::CentrifugalLoad:
                    l.centrifugal.apply(cs, num_nodes, positions, masses, forces);
                    break;

                case LoadWave43Type::CylindricalPressure:
                    l.cyl_pressure.apply(cs, positions, forces);
                    break;

                case LoadWave43Type::FluidPressure:
                    l.fluid_pressure.apply(cs, positions, forces);
                    break;

                case LoadWave43Type::LaserLoad:
                    l.laser.apply(time, cs, positions, heat_flux, forces);
                    break;

                case LoadWave43Type::BoltPreload:
                    l.bolt_preload.update_lock_state(time);
                    l.bolt_preload.apply(time, cs, forces);
                    break;
            }
        }
    }

    void print_summary() const {
        std::cout << "LoadWave43Manager: " << loads_.size() << " loads\n";
        for (const auto& l : loads_) {
            const char* ts = "Unknown";
            switch (l.type) {
                case LoadWave43Type::CentrifugalLoad:     ts = "CentrifugalLoad";     break;
                case LoadWave43Type::CylindricalPressure: ts = "CylindricalPressure"; break;
                case LoadWave43Type::FluidPressure:       ts = "FluidPressure";       break;
                case LoadWave43Type::LaserLoad:           ts = "LaserLoad";           break;
                case LoadWave43Type::BoltPreload:         ts = "BoltPreload";         break;
            }
            std::cout << "  [" << l.id << "] " << ts << " name=" << l.name << "\n";
        }
    }

private:
    Real get_curve_scale(const LoadWave43& l, Real time) const {
        int cid = -1;
        switch (l.type) {
            case LoadWave43Type::CentrifugalLoad:     cid = l.centrifugal.load_curve_id;    break;
            case LoadWave43Type::CylindricalPressure: cid = l.cyl_pressure.load_curve_id;   break;
            case LoadWave43Type::FluidPressure:       cid = l.fluid_pressure.load_curve_id; break;
            case LoadWave43Type::LaserLoad:           cid = l.laser.load_curve_id;          break;
            case LoadWave43Type::BoltPreload:         cid = l.bolt_preload.load_curve_id;   break;
        }
        if (curves_ && cid >= 0) {
            return curves_->evaluate(cid, time);
        }
        return 1.0;
    }

    std::vector<LoadWave43> loads_;
    LoadCurveManager* curves_ = nullptr;
};

} // namespace fem
} // namespace nxs
