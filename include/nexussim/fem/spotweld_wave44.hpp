#pragma once

/**
 * @file spotweld_wave44.hpp
 * @brief Wave 44d: Spot Weld Contact
 *
 * Provides SpotWeldContact for penalty-based spot weld connections between
 * structural parts, with progressive damage, thermal softening, and energy
 * tracking.
 *
 * Namespace: nxs::fem
 */

#include <nexussim/core/types.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
#include <algorithm>
#include <numeric>

namespace nxs {
namespace fem {

// ============================================================================
// SpotWeldFailureMode
// ============================================================================

/// Identifies which failure criterion caused weld rupture.
enum class SpotWeldFailureMode {
    None,           ///< Weld is intact
    NormalTension,  ///< Pure tension failure (Fn > Fn_max)
    Shear,          ///< Pure shear failure  (Fs > Fs_max)
    Combined        ///< (Fn/Fn_max)^2 + (Fs/Fs_max)^2 > 1
};

// ============================================================================
// SpotWeldConfig
// ============================================================================

/// Configuration parameters for a single spot weld.
struct SpotWeldConfig {
    int         id;                      ///< Weld identifier
    std::size_t node_a;                  ///< Weld node on part A
    std::size_t node_b;                  ///< Attached node on part B (SIZE_MAX = unresolved)
    Real        initial_stiffness;       ///< Penalty stiffness [force/length]
    Real        normal_strength;         ///< Failure strength in tension [force]
    Real        shear_strength;          ///< Failure strength in shear [force]
    Real        failure_displacement;    ///< Displacement at complete failure [length]
    Real        degradation_start;       ///< Fraction of failure_displacement where degradation begins (0–1)
    Real        thermal_softening_temp;  ///< Temperature threshold for softening (0 = disabled)
    Real        thermal_softening_factor;///< Strength multiplier at softening temperature (0–1)

    SpotWeldConfig()
        : id(0)
        , node_a(0)
        , node_b(std::numeric_limits<std::size_t>::max())
        , initial_stiffness(1.0e6)
        , normal_strength(1.0e4)
        , shear_strength(1.0e4)
        , failure_displacement(1.0e-3)
        , degradation_start(0.5)
        , thermal_softening_temp(0.0)
        , thermal_softening_factor(0.5)
    {}
};

// ============================================================================
// SpotWeldState
// ============================================================================

/// Run-time state of a single spot weld.
struct SpotWeldState {
    Real current_force[3];       ///< Current force vector (fx, fy, fz) [force]
    Real normal_force;           ///< Scalar normal force component [force]
    Real shear_force;            ///< Scalar shear force component [force]
    Real displacement;           ///< Current relative displacement magnitude [length]
    Real damage;                 ///< Accumulated damage: 0 = intact, 1 = failed
    Real stiffness_factor;       ///< Current stiffness multiplier (1 → 0)
    bool failed;                 ///< True when weld has completely failed
    SpotWeldFailureMode failure_mode; ///< Which mode triggered failure
    Real energy_absorbed;        ///< Cumulative energy absorbed [energy]

    SpotWeldState()
        : current_force{0.0, 0.0, 0.0}
        , normal_force(0.0)
        , shear_force(0.0)
        , displacement(0.0)
        , damage(0.0)
        , stiffness_factor(1.0)
        , failed(false)
        , failure_mode(SpotWeldFailureMode::None)
        , energy_absorbed(0.0)
    {}

    void reset() {
        current_force[0] = current_force[1] = current_force[2] = 0.0;
        normal_force   = 0.0;
        shear_force    = 0.0;
        displacement   = 0.0;
        damage         = 0.0;
        stiffness_factor = 1.0;
        failed         = false;
        failure_mode   = SpotWeldFailureMode::None;
        energy_absorbed = 0.0;
    }
};

// ============================================================================
// SpotWeldContact
// ============================================================================

/**
 * @brief Host-only spot weld contact handler.
 *
 * Welds are registered via add_weld(). Partners are resolved from a flat
 * position array. update() recomputes forces and damage each time step.
 * apply_forces() scatters equal-and-opposite weld forces to the global
 * force vector.
 */
class SpotWeldContact {
public:
    SpotWeldContact() = default;

    // -----------------------------------------------------------------------
    // Registration
    // -----------------------------------------------------------------------

    /// Register a spot weld definition.
    void add_weld(const SpotWeldConfig& config) {
        configs_.push_back(config);
        states_.emplace_back();
        // Cache the initial axis (zero until find_partners resolves node_b)
        axes_.push_back({0.0, 0.0, 0.0});
        prev_forces_.push_back({0.0, 0.0, 0.0});
    }

    // -----------------------------------------------------------------------
    // Partner resolution
    // -----------------------------------------------------------------------

    /**
     * @brief For each weld with node_b == SIZE_MAX, find the nearest node
     *        within search_radius on the global node list and assign it.
     *
     * @param positions  Flat array of node positions [3*num_nodes]
     * @param num_nodes  Number of nodes
     * @param search_radius  Maximum distance to search for a partner
     */
    void find_partners(const Real* positions, std::size_t num_nodes,
                       Real search_radius) {
        const std::size_t unset = std::numeric_limits<std::size_t>::max();

        for (std::size_t w = 0; w < configs_.size(); ++w) {
            SpotWeldConfig& cfg = configs_[w];
            if (cfg.node_b != unset) continue;  // already assigned

            const Real ax = positions[3 * cfg.node_a + 0];
            const Real ay = positions[3 * cfg.node_a + 1];
            const Real az = positions[3 * cfg.node_a + 2];

            Real    best_dist2 = search_radius * search_radius;
            std::size_t best  = unset;

            for (std::size_t n = 0; n < num_nodes; ++n) {
                if (n == cfg.node_a) continue;
                const Real dx = positions[3 * n + 0] - ax;
                const Real dy = positions[3 * n + 1] - ay;
                const Real dz = positions[3 * n + 2] - az;
                const Real d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    best       = n;
                }
            }

            cfg.node_b = best;
        }
    }

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Compute weld forces and update damage for all welds.
     *
     * @param positions     Flat position array [3*num_nodes]
     * @param num_nodes     Number of nodes
     * @param dt            Time step size (used for energy integration)
     * @param temperatures  Optional nodal temperature array [num_nodes] (nullptr = disabled)
     */
    void update(const Real* positions, std::size_t num_nodes,
                Real /*dt*/, const Real* temperatures = nullptr) {
        const std::size_t unset = std::numeric_limits<std::size_t>::max();

        for (std::size_t w = 0; w < configs_.size(); ++w) {
            const SpotWeldConfig& cfg = configs_[w];
            SpotWeldState&        st  = states_[w];

            if (st.failed)                          continue;
            if (cfg.node_b == unset)                continue;
            if (cfg.node_a >= num_nodes)            continue;
            if (cfg.node_b >= num_nodes)            continue;

            // ---- relative displacement vector (B - A) --------------------
            const Real rx = positions[3 * cfg.node_b + 0] - positions[3 * cfg.node_a + 0];
            const Real ry = positions[3 * cfg.node_b + 1] - positions[3 * cfg.node_a + 1];
            const Real rz = positions[3 * cfg.node_b + 2] - positions[3 * cfg.node_a + 2];

            // ---- weld axis -----------------------------------------------
            // Initialise axis on first call (when displacement == 0 and axis
            // is still zero). We use the initial relative vector as the weld
            // axis if it is non-zero; otherwise fall back to a default.
            Real& ex = axes_[w][0];
            Real& ey = axes_[w][1];
            Real& ez = axes_[w][2];

            const Real axis_len2 = ex * ex + ey * ey + ez * ez;
            if (axis_len2 < 1.0e-30) {
                // First call — set axis from initial configuration
                const Real rl = std::sqrt(rx * rx + ry * ry + rz * rz);
                if (rl > 1.0e-15) {
                    ex = rx / rl;
                    ey = ry / rl;
                    ez = rz / rl;
                } else {
                    // Degenerate: coincident nodes, use z-axis
                    ex = 0.0; ey = 0.0; ez = 1.0;
                }
                // The initial configuration is the reference — no force yet
                // Store reference vector and continue to force computation
                // (force will be zero at this step since disp = 0)
            }

            // ---- decompose relative displacement -------------------------
            // Normal component (along weld axis)
            const Real disp_n  = rx * ex + ry * ey + rz * ez;
            // Shear component vector
            const Real sx = rx - disp_n * ex;
            const Real sy = ry - disp_n * ey;
            const Real sz = rz - disp_n * ez;
            const Real disp_s = std::sqrt(sx * sx + sy * sy + sz * sz);

            // Total displacement magnitude (used for degradation curve)
            const Real disp_total = std::sqrt(disp_n * disp_n + disp_s * disp_s);
            const Real old_disp   = st.displacement;
            st.displacement       = disp_total;

            // ---- thermal softening factor --------------------------------
            Real thermal_factor = 1.0;
            if (temperatures && cfg.thermal_softening_temp > 0.0) {
                // Average temperature of the two weld nodes
                const Real T = 0.5 * (temperatures[cfg.node_a] + temperatures[cfg.node_b]);
                if (T >= cfg.thermal_softening_temp) {
                    thermal_factor = cfg.thermal_softening_factor;
                }
            }

            // ---- stiffness degradation -----------------------------------
            const Real d_start = cfg.degradation_start * cfg.failure_displacement;
            const Real d_fail  = cfg.failure_displacement;

            Real k_factor = 1.0;
            if (disp_total >= d_fail) {
                k_factor = 0.0;
            } else if (disp_total >= d_start && d_fail > d_start) {
                // Linear degradation from 1 → 0 over [d_start, d_fail]
                k_factor = 1.0 - (disp_total - d_start) / (d_fail - d_start);
            }
            st.stiffness_factor = k_factor;
            st.damage           = 1.0 - k_factor;

            const Real k_eff = cfg.initial_stiffness * k_factor;

            // ---- compute forces ------------------------------------------
            // Save previous forces for energy integration
            const Real Fold0 = prev_forces_[w][0];
            const Real Fold1 = prev_forces_[w][1];
            const Real Fold2 = prev_forces_[w][2];

            // Normal force magnitude (tension positive → pulls nodes together)
            // Force on A is in the +axis direction (attracts towards B)
            Real Fn_scalar = k_eff * disp_n;   // can be negative (compression)
            Real Fn_abs    = std::abs(Fn_scalar);

            // Shear force magnitude
            Real Fs_scalar = k_eff * disp_s;
            Real Fs_abs    = Fs_scalar;          // disp_s >= 0

            // Force vector (acting on node A, pulling it toward B along axis)
            st.normal_force = Fn_abs;
            st.shear_force  = Fs_abs;

            // Full force vector
            Real fx = k_eff * rx;  // = Fn * ex + Fs * (sx/disp_s) if disp_s>0
            Real fy = k_eff * ry;
            Real fz = k_eff * rz;

            st.current_force[0] = fx;
            st.current_force[1] = fy;
            st.current_force[2] = fz;

            // ---- energy absorbed -----------------------------------------
            const Real d_disp = st.displacement - old_disp;
            const Real f_mag_old = std::sqrt(Fold0*Fold0 + Fold1*Fold1 + Fold2*Fold2);
            const Real f_mag_new = std::sqrt(fx*fx + fy*fy + fz*fz);
            st.energy_absorbed += 0.5 * (f_mag_old + f_mag_new) * std::abs(d_disp);

            prev_forces_[w][0] = fx;
            prev_forces_[w][1] = fy;
            prev_forces_[w][2] = fz;

            // ---- failure check -------------------------------------------
            const Real Fn_max = cfg.normal_strength * thermal_factor;
            const Real Fs_max = cfg.shear_strength  * thermal_factor;

            // Only check tension failure in pull direction
            const bool tension_fail = (disp_n > 0.0) && (Fn_abs > Fn_max);
            const bool shear_fail   = (Fs_abs > Fs_max);
            const Real combined_crit = (Fn_max > 0.0 ? (Fn_abs / Fn_max) * (Fn_abs / Fn_max) : 0.0)
                                     + (Fs_max > 0.0 ? (Fs_abs / Fs_max) * (Fs_abs / Fs_max) : 0.0);
            const bool combined_fail = (combined_crit > 1.0);

            if (k_factor == 0.0 || tension_fail || shear_fail || combined_fail) {
                st.failed = true;
                st.stiffness_factor = 0.0;
                st.damage           = 1.0;
                // Zero out force
                st.current_force[0] = 0.0;
                st.current_force[1] = 0.0;
                st.current_force[2] = 0.0;
                st.normal_force     = 0.0;
                st.shear_force      = 0.0;

                // Determine primary failure mode (individual modes take priority
                // over the combined criterion so that single-mode loading is
                // classified correctly even when it also satisfies combined).
                if (tension_fail && !shear_fail) {
                    st.failure_mode = SpotWeldFailureMode::NormalTension;
                } else if (shear_fail && !tension_fail) {
                    st.failure_mode = SpotWeldFailureMode::Shear;
                } else {
                    st.failure_mode = SpotWeldFailureMode::Combined;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Force scatter
    // -----------------------------------------------------------------------

    /**
     * @brief Add weld forces to the global force vector.
     *
     * Forces are equal and opposite: +F on node_a, -F on node_b.
     *
     * @param forces     Flat force array [3*num_nodes], accumulated in-place
     * @param num_nodes  Number of nodes
     */
    void apply_forces(Real* forces, std::size_t num_nodes) const {
        const std::size_t unset = std::numeric_limits<std::size_t>::max();

        for (std::size_t w = 0; w < configs_.size(); ++w) {
            const SpotWeldConfig& cfg = configs_[w];
            const SpotWeldState&  st  = states_[w];

            if (st.failed)               continue;
            if (cfg.node_b == unset)     continue;
            if (cfg.node_a >= num_nodes) continue;
            if (cfg.node_b >= num_nodes) continue;

            forces[3 * cfg.node_a + 0] += st.current_force[0];
            forces[3 * cfg.node_a + 1] += st.current_force[1];
            forces[3 * cfg.node_a + 2] += st.current_force[2];

            forces[3 * cfg.node_b + 0] -= st.current_force[0];
            forces[3 * cfg.node_b + 1] -= st.current_force[1];
            forces[3 * cfg.node_b + 2] -= st.current_force[2];
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    const SpotWeldState& state(std::size_t weld_index) const {
        return states_[weld_index];
    }

    std::size_t num_welds() const { return configs_.size(); }

    std::size_t num_failed() const {
        std::size_t n = 0;
        for (const auto& s : states_) n += s.failed ? 1u : 0u;
        return n;
    }

    std::size_t num_active() const {
        return num_welds() - num_failed();
    }

    Real total_energy() const {
        Real sum = 0.0;
        for (const auto& s : states_) sum += s.energy_absorbed;
        return sum;
    }

    std::vector<std::size_t> failed_weld_indices() const {
        std::vector<std::size_t> out;
        for (std::size_t i = 0; i < states_.size(); ++i) {
            if (states_[i].failed) out.push_back(i);
        }
        return out;
    }

    /// Reset all welds to undamaged state (configs are preserved).
    void reset() {
        for (auto& s : states_)      s.reset();
        for (auto& a : axes_)        a = {0.0, 0.0, 0.0};
        for (auto& pf : prev_forces_) pf = {0.0, 0.0, 0.0};
    }

private:
    std::vector<SpotWeldConfig>         configs_;
    std::vector<SpotWeldState>          states_;
    std::vector<std::array<Real, 3>>    axes_;         ///< Cached weld axes
    std::vector<std::array<Real, 3>>    prev_forces_;  ///< Forces from previous step (for energy)
};

// ============================================================================
// SpotWeldArray
// ============================================================================

/**
 * @brief Convenience wrapper that creates a regular grid of spot welds.
 *
 * The grid is defined by a base position (x0, y0, z0), spacing (dx, dy),
 * and counts (nx, ny).  Weld nodes are assumed to already exist in the
 * provided position array; partners are resolved via find_partners().
 */
class SpotWeldArray {
public:
    SpotWeldArray() = default;

    /**
     * @brief Create a rectangular grid of spot welds.
     *
     * @param x0, y0, z0       Origin of the grid
     * @param dx, dy           Spacing between welds in x and y
     * @param nx, ny           Number of welds in each direction
     * @param base_config      Template configuration (id, strengths, etc.)
     * @param positions        Flat position array [3*num_nodes]
     * @param num_nodes        Number of nodes
     * @param search_radius    Radius used to find node_b partners
     */
    void create_grid(Real x0, Real y0, Real z0,
                     Real dx, Real dy,
                     int nx, int ny,
                     const SpotWeldConfig& base_config,
                     const Real* positions, std::size_t num_nodes,
                     Real search_radius) {
        contact_.reset();

        int weld_id = base_config.id;
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                const Real wx = x0 + ix * dx;
                const Real wy = y0 + iy * dy;
                const Real wz = z0;

                // Find the nearest node to serve as node_a
                std::size_t best_a = std::numeric_limits<std::size_t>::max();
                Real best_d2 = search_radius * search_radius;
                for (std::size_t n = 0; n < num_nodes; ++n) {
                    const Real ddx = positions[3*n+0] - wx;
                    const Real ddy = positions[3*n+1] - wy;
                    const Real ddz = positions[3*n+2] - wz;
                    const Real d2  = ddx*ddx + ddy*ddy + ddz*ddz;
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_a  = n;
                    }
                }

                if (best_a == std::numeric_limits<std::size_t>::max()) continue;

                SpotWeldConfig cfg = base_config;
                cfg.id     = weld_id++;
                cfg.node_a = best_a;
                cfg.node_b = std::numeric_limits<std::size_t>::max();  // to be resolved
                contact_.add_weld(cfg);
            }
        }

        contact_.find_partners(positions, num_nodes, search_radius);
    }

    const SpotWeldContact& contact() const { return contact_; }
          SpotWeldContact& contact()       { return contact_; }

private:
    SpotWeldContact contact_;
};

} // namespace fem
} // namespace nxs
