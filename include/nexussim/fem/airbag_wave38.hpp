#pragma once

/**
 * @file airbag_wave38.hpp
 * @brief Wave 38: Production Airbag Simulation — 5 Features
 *
 * Features:
 *   1. FVBagSolver       - Finite-volume multi-chamber airbag solver
 *   2. AirbagInjection   - Inflator model with mass/temperature curves
 *   3. AirbagVenting     - Vent holes and fabric porosity
 *   4. AirbagFolding     - Folded bag initialization and unfolding
 *   5. AirbagThermal     - Gas thermodynamics and heat transfer
 *
 * References:
 * - Hallquist et al. (2007) "LS-DYNA Theory Manual - Airbag Models"
 * - Wang & Nefske (1988) "A new CAL3D airbag inflation model"
 * - Marklund & Nilsson (2002) "Optimization of airbag inflation parameters"
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <cstring>

namespace nxs {
namespace fem {

using Real = nxs::Real;

// ============================================================================
// Utility functions (Kokkos-compatible math)
// ============================================================================

namespace airbag_detail {

KOKKOS_INLINE_FUNCTION
Real k_sqrt(Real x) {
#ifdef __CUDA_ARCH__
    return Kokkos::sqrt(x);
#else
    return std::sqrt(x);
#endif
}

KOKKOS_INLINE_FUNCTION
Real k_exp(Real x) {
#ifdef __CUDA_ARCH__
    return Kokkos::exp(x);
#else
    return std::exp(x);
#endif
}

KOKKOS_INLINE_FUNCTION
Real k_fabs(Real x) {
#ifdef __CUDA_ARCH__
    return Kokkos::fabs(x);
#else
    return std::fabs(x);
#endif
}

KOKKOS_INLINE_FUNCTION
Real k_fmin(Real a, Real b) {
#ifdef __CUDA_ARCH__
    return Kokkos::fmin(a, b);
#else
    return std::fmin(a, b);
#endif
}

KOKKOS_INLINE_FUNCTION
Real k_fmax(Real a, Real b) {
#ifdef __CUDA_ARCH__
    return Kokkos::fmax(a, b);
#else
    return std::fmax(a, b);
#endif
}

/// Linear interpolation in a curve table
inline Real interpolate_curve(const Real* x_data, const Real* y_data, int n_pts, Real x) {
    if (n_pts <= 0) return 0.0;
    if (n_pts == 1) return y_data[0];
    if (x <= x_data[0]) return y_data[0];
    if (x >= x_data[n_pts - 1]) return y_data[n_pts - 1];
    for (int i = 0; i < n_pts - 1; ++i) {
        if (x >= x_data[i] && x <= x_data[i + 1]) {
            Real dx = x_data[i + 1] - x_data[i];
            if (dx < 1.0e-30) return y_data[i];
            Real t = (x - x_data[i]) / dx;
            return y_data[i] * (1.0 - t) + y_data[i + 1] * t;
        }
    }
    return y_data[n_pts - 1];
}

/// Euclidean distance between two 3D points
KOKKOS_INLINE_FUNCTION
Real dist3(const Real a[3], const Real b[3]) {
    Real dx = a[0] - b[0];
    Real dy = a[1] - b[1];
    Real dz = a[2] - b[2];
    return k_sqrt(dx * dx + dy * dy + dz * dz);
}

} // namespace airbag_detail

// ============================================================================
// 1. FVBagSolver — Finite-Volume Multi-Chamber Airbag Solver
// ============================================================================

/**
 * @brief Single airbag chamber state.
 *
 * Each chamber stores thermodynamic state variables for the contained gas.
 * The airbag may consist of multiple connected chambers with inter-chamber
 * flow through orifices.
 */
struct AirbagChamber {
    Real volume       = 1.0e-3;   ///< Current volume [m^3]
    Real pressure     = 101325.0; ///< Gas pressure [Pa]
    Real temperature  = 293.15;   ///< Gas temperature [K]
    Real mass         = 0.0;      ///< Gas mass [kg]
    int  n_vents      = 0;        ///< Number of vent holes in this chamber

    /// Volume rate of change (computed from fabric motion)
    Real dV_dt        = 0.0;

    /// Inter-chamber orifice areas [m^2], max 8 connections
    Real orifice_area[8] = {};
    /// Connected chamber indices (-1 = external)
    int  connected_to[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int  n_connections   = 0;
};

/**
 * @brief Airbag global configuration.
 */
struct AirbagConfig {
    int  n_chambers  = 1;         ///< Number of chambers
    Real gamma       = 1.4;       ///< Ratio of specific heats (Cp/Cv)
    Real R_gas       = 287.0;     ///< Specific gas constant [J/(kg*K)]
    Real T_ambient   = 293.15;    ///< Ambient temperature [K]
    Real P_ambient   = 101325.0;  ///< Ambient pressure [Pa]
};

/**
 * @brief Finite-volume airbag solver.
 *
 * Models airbag inflation using a control-volume approach. Each chamber
 * is treated as a uniform-state control volume. The ideal gas law
 * P*V = m*R*T governs the thermodynamic state.
 *
 * Inter-chamber flow is modeled as isentropic orifice flow:
 *   dm/dt = Cd * A * sqrt(2 * rho_up * |dP|) * sign(dP)
 *
 * The solver updates chamber states using forward Euler integration.
 */
class FVBagSolver {
public:
    FVBagSolver() = default;

    explicit FVBagSolver(const AirbagConfig& cfg)
        : config_(cfg) {}

    /**
     * @brief Update all chamber states by one time step.
     *
     * For each chamber:
     * 1. Compute density from mass/volume
     * 2. Process inter-chamber flows
     * 3. Update mass from flows
     * 4. Update temperature from energy balance
     * 5. Recompute pressure from ideal gas law
     *
     * @param chambers     Array of chamber states (modified in place)
     * @param n_chambers   Number of chambers
     * @param config       Global configuration
     * @param dt           Time step [s]
     */
    void solve_bag_step(AirbagChamber* chambers, int n_chambers,
                        const AirbagConfig& config, Real dt) {
        // Compute inter-chamber mass flows
        Real mass_flow[64] = {};  // max 64 connections total
        int flow_idx = 0;

        for (int i = 0; i < n_chambers; ++i) {
            for (int c = 0; c < chambers[i].n_connections; ++c) {
                int j = chambers[i].connected_to[c];
                if (j < 0 || j >= n_chambers) continue;
                if (j <= i) continue;  // avoid double-counting

                Real dP = chambers[i].pressure - chambers[j].pressure;
                Real rho_up = (dP > 0.0)
                    ? chambers[i].mass / airbag_detail::k_fmax(chambers[i].volume, 1.0e-20)
                    : chambers[j].mass / airbag_detail::k_fmax(chambers[j].volume, 1.0e-20);

                Real A_orifice = chambers[i].orifice_area[c];
                Real Cd = 0.6;  // discharge coefficient for orifice
                Real abs_dP = airbag_detail::k_fabs(dP);

                // Isentropic orifice flow: dm/dt = Cd * A * sqrt(2*rho*|dP|)
                Real dm_dt = Cd * A_orifice * airbag_detail::k_sqrt(
                    2.0 * airbag_detail::k_fmax(rho_up, 1.0e-10) * abs_dP);
                if (dP < 0.0) dm_dt = -dm_dt;

                mass_flow[flow_idx] = dm_dt * dt;
                flow_idx++;

                // Transfer mass between chambers
                chambers[i].mass -= dm_dt * dt;
                chambers[j].mass += dm_dt * dt;

                // Temperature mixing (energy-weighted)
                if (dm_dt > 0.0) {
                    // Flow from i to j: j gets gas at temperature of i
                    Real E_transfer = dm_dt * dt * config.R_gas * chambers[i].temperature
                                      / (config.gamma - 1.0);
                    Real E_j = chambers[j].mass * config.R_gas * chambers[j].temperature
                               / (config.gamma - 1.0);
                    if (chambers[j].mass > 1.0e-20) {
                        chambers[j].temperature = (E_j + E_transfer) * (config.gamma - 1.0)
                                                  / (chambers[j].mass * config.R_gas);
                    }
                } else if (dm_dt < 0.0) {
                    Real E_transfer = (-dm_dt * dt) * config.R_gas * chambers[j].temperature
                                      / (config.gamma - 1.0);
                    Real E_i = chambers[i].mass * config.R_gas * chambers[i].temperature
                               / (config.gamma - 1.0);
                    if (chambers[i].mass > 1.0e-20) {
                        chambers[i].temperature = (E_i + E_transfer) * (config.gamma - 1.0)
                                                  / (chambers[i].mass * config.R_gas);
                    }
                }
            }
        }

        // Update pressure from ideal gas law: P = m*R*T / V
        for (int i = 0; i < n_chambers; ++i) {
            chambers[i].mass = airbag_detail::k_fmax(chambers[i].mass, 0.0);
            Real V = airbag_detail::k_fmax(chambers[i].volume, 1.0e-20);
            chambers[i].pressure = chambers[i].mass * config.R_gas
                                   * chambers[i].temperature / V;
        }
    }

    /**
     * @brief Compute pressure from ideal gas law for a single chamber.
     * @param chamber  Chamber state
     * @param R_gas    Specific gas constant
     * @return Pressure [Pa]
     */
    static Real ideal_gas_pressure(const AirbagChamber& chamber, Real R_gas) {
        Real V = airbag_detail::k_fmax(chamber.volume, 1.0e-20);
        return chamber.mass * R_gas * chamber.temperature / V;
    }

    /**
     * @brief Compute total energy in a chamber (internal energy).
     * E = m * Cv * T = m * R * T / (gamma - 1)
     */
    static Real chamber_energy(const AirbagChamber& chamber, Real R_gas, Real gamma) {
        return chamber.mass * R_gas * chamber.temperature / (gamma - 1.0);
    }

    const AirbagConfig& config() const { return config_; }

private:
    AirbagConfig config_;
};

// ============================================================================
// 2. AirbagInjection — Inflator Model
// ============================================================================

/**
 * @brief Inflator data: time-dependent mass flow rate and temperature.
 *
 * The inflator is characterized by tabulated curves:
 * - mass_rate_curve[i]: mass flow rate [kg/s] at time time_pts[i]
 * - temp_curve[i]: gas temperature [K] at time time_pts[i]
 *
 * Between time points, values are linearly interpolated.
 */
struct InflatorData {
    Real mass_rate_curve[32] = {};  ///< Mass flow rate values [kg/s]
    Real temp_curve[32]      = {};  ///< Temperature values [K]
    Real time_pts[32]        = {};  ///< Time points [s]
    int  n_points            = 0;   ///< Number of curve points
    Real t_start             = 0.0; ///< Activation time [s]
    Real total_mass          = 0.0; ///< Total available propellant mass [kg]
    Real mass_injected       = 0.0; ///< Cumulative mass injected [kg]
};

/**
 * @brief Inflator model for airbag deployment.
 *
 * Computes mass flow rate and temperature at any time by interpolation
 * of the inflator curves. The injected mass and thermal energy are added
 * to the target chamber.
 */
class AirbagInjection {
public:
    AirbagInjection() = default;

    /**
     * @brief Compute injection rate and temperature at time t.
     *
     * @param inflator   Inflator data with curves
     * @param t          Current simulation time [s]
     * @param dm_dt      [out] Mass flow rate [kg/s]
     * @param T_inj      [out] Injection gas temperature [K]
     */
    void compute_injection(const InflatorData& inflator, Real t,
                           Real& dm_dt, Real& T_inj) const {
        Real t_rel = t - inflator.t_start;
        if (t_rel < 0.0 || inflator.n_points <= 0) {
            dm_dt = 0.0;
            T_inj = 293.15;
            return;
        }

        dm_dt = airbag_detail::interpolate_curve(
            inflator.time_pts, inflator.mass_rate_curve, inflator.n_points, t_rel);
        T_inj = airbag_detail::interpolate_curve(
            inflator.time_pts, inflator.temp_curve, inflator.n_points, t_rel);

        // Clamp to non-negative
        dm_dt = airbag_detail::k_fmax(dm_dt, 0.0);
        T_inj = airbag_detail::k_fmax(T_inj, 1.0);
    }

    /**
     * @brief Inject gas into a chamber for one time step.
     *
     * Adds mass dm = dm_dt * dt and updates the chamber temperature
     * via energy-weighted mixing:
     *   T_new = (m_old * T_old + dm * T_inj) / (m_old + dm)
     *
     * @param chamber    Chamber state (modified)
     * @param inflator   Inflator data (mass_injected updated)
     * @param R_gas      Specific gas constant
     * @param gamma      Ratio of specific heats
     * @param dt         Time step [s]
     */
    void inject(AirbagChamber& chamber, InflatorData& inflator,
                Real R_gas, Real gamma, Real dt) {
        Real dm_dt, T_inj;
        Real t_current = inflator.t_start;  // caller should set properly
        compute_injection(inflator, inflator.t_start + dt, dm_dt, T_inj);

        Real dm = dm_dt * dt;

        // Check propellant availability
        if (inflator.total_mass > 0.0) {
            Real remaining = inflator.total_mass - inflator.mass_injected;
            dm = airbag_detail::k_fmin(dm, airbag_detail::k_fmax(remaining, 0.0));
        }

        if (dm <= 0.0) return;

        // Energy-weighted temperature mixing
        Real Cv = R_gas / (gamma - 1.0);
        Real E_old = chamber.mass * Cv * chamber.temperature;
        Real E_inj = dm * Cv * T_inj;

        chamber.mass += dm;
        inflator.mass_injected += dm;

        if (chamber.mass > 1.0e-20) {
            chamber.temperature = (E_old + E_inj) / (chamber.mass * Cv);
        }

        // Update pressure
        Real V = airbag_detail::k_fmax(chamber.volume, 1.0e-20);
        chamber.pressure = chamber.mass * R_gas * chamber.temperature / V;
    }

    /**
     * @brief Inject a given amount of mass at a given temperature.
     *
     * Direct injection without curve interpolation.
     *
     * @param chamber    Chamber state (modified)
     * @param dm         Mass to inject [kg]
     * @param T_inj      Temperature of injected gas [K]
     * @param R_gas      Specific gas constant
     * @param gamma      Ratio of specific heats
     */
    void inject_direct(AirbagChamber& chamber, Real dm, Real T_inj,
                       Real R_gas, Real gamma) {
        if (dm <= 0.0) return;

        Real Cv = R_gas / (gamma - 1.0);
        Real E_old = chamber.mass * Cv * chamber.temperature;
        Real E_inj = dm * Cv * T_inj;

        chamber.mass += dm;
        if (chamber.mass > 1.0e-20) {
            chamber.temperature = (E_old + E_inj) / (chamber.mass * Cv);
        }

        Real V = airbag_detail::k_fmax(chamber.volume, 1.0e-20);
        chamber.pressure = chamber.mass * R_gas * chamber.temperature / V;
    }
};

// ============================================================================
// 3. AirbagVenting — Vent Holes and Fabric Porosity
// ============================================================================

/**
 * @brief Vent hole specification.
 */
struct VentData {
    Real area             = 0.0;      ///< Vent area [m^2]
    Real discharge_coeff  = 0.6;      ///< Discharge coefficient Cd
    Real opening_pressure = 0.0;      ///< Pressure threshold for vent opening [Pa]
    bool is_open          = false;    ///< Current open state
};

/**
 * @brief Airbag venting model: vent holes and fabric porosity.
 *
 * Mass loss through vent holes (Bernoulli):
 *   dm_vent = Cd * A_vent * sqrt(2 * rho * (P - P_ext)) * dt
 *   (only when P > P_ext and vent is open)
 *
 * Mass loss through fabric porosity:
 *   dm_porous = rho * A_fabric * v_leak * dt
 *   where v_leak = porosity_coeff * (P - P_ext) / (rho * thickness)
 *
 * The vent opens when chamber pressure exceeds the opening pressure threshold.
 */
class AirbagVenting {
public:
    AirbagVenting() = default;

    /**
     * @brief Compute total mass loss from venting and porosity for one step.
     *
     * @param chamber      Chamber state (modified: mass, pressure updated)
     * @param vents        Array of vent holes
     * @param n_vents      Number of vents
     * @param P_external   External (ambient) pressure [Pa]
     * @param A_fabric     Total fabric area [m^2]
     * @param porosity     Fabric porosity coefficient [m/s/Pa]
     * @param R_gas        Specific gas constant
     * @param dt           Time step [s]
     * @return Total mass lost [kg]
     */
    Real compute_venting(AirbagChamber& chamber, VentData* vents, int n_vents,
                         Real P_external, Real A_fabric, Real porosity,
                         Real R_gas, Real dt) {
        Real dP = chamber.pressure - P_external;
        if (dP <= 0.0) return 0.0;

        Real V = airbag_detail::k_fmax(chamber.volume, 1.0e-20);
        Real rho = chamber.mass / V;

        Real dm_total = 0.0;

        // Vent hole flow
        for (int i = 0; i < n_vents; ++i) {
            // Check vent opening
            if (!vents[i].is_open) {
                if (chamber.pressure >= vents[i].opening_pressure) {
                    vents[i].is_open = true;
                }
            }
            if (!vents[i].is_open) continue;

            Real Cd = vents[i].discharge_coeff;
            Real A = vents[i].area;

            // Bernoulli-based vent flow: dm = Cd * A * sqrt(2 * rho * dP)
            Real dm_vent = Cd * A * airbag_detail::k_sqrt(
                2.0 * airbag_detail::k_fmax(rho, 1.0e-10) * dP) * dt;
            dm_total += dm_vent;
        }

        // Fabric porosity leakage
        if (A_fabric > 0.0 && porosity > 0.0) {
            // v_leak = porosity * dP (simplified Darcy-type law)
            Real v_leak = porosity * dP;
            Real dm_porous = rho * A_fabric * v_leak * dt;
            dm_total += dm_porous;
        }

        // Clamp mass loss to available mass
        dm_total = airbag_detail::k_fmin(dm_total, chamber.mass * 0.99);

        // Update chamber state
        chamber.mass -= dm_total;
        chamber.mass = airbag_detail::k_fmax(chamber.mass, 0.0);

        // Update pressure (temperature unchanged for adiabatic venting)
        chamber.pressure = chamber.mass * R_gas * chamber.temperature / V;

        return dm_total;
    }

    /**
     * @brief Simple venting computation without porosity.
     */
    Real compute_vent_only(AirbagChamber& chamber, VentData* vents, int n_vents,
                           Real P_external, Real R_gas, Real dt) {
        return compute_venting(chamber, vents, n_vents, P_external, 0.0, 0.0, R_gas, dt);
    }
};

// ============================================================================
// 4. AirbagFolding — Folded Bag Initialization and Unfolding
// ============================================================================

/**
 * @brief Fold specification for a single fold operation.
 *
 * Each fold is defined by a fold axis (point + direction) and the set of
 * nodes that are reflected across that axis.
 */
struct FoldSpec {
    Real axis_point[3]  = {0.0, 0.0, 0.0};  ///< Point on fold axis
    Real axis_dir[3]    = {1.0, 0.0, 0.0};  ///< Fold axis direction
    Real fold_angle     = 180.0;             ///< Fold angle [degrees]
};

/**
 * @brief Folded bag state.
 */
struct FoldedBag {
    Real* node_coords    = nullptr;  ///< Current node coordinates [3*n_nodes]
    int   n_nodes        = 0;        ///< Number of mesh nodes
    FoldSpec fold_sequence[16] = {};  ///< Sequence of folds
    int   n_folds        = 0;        ///< Number of folds applied
    int   current_fold   = 0;        ///< Current fold being unfolded
    Real  unfold_progress = 0.0;     ///< Progress of current unfold [0,1]
    bool  fully_unfolded = false;    ///< Whether all folds are undone

    /// Flat (unfolded) reference coordinates [3*n_nodes]
    Real* flat_coords    = nullptr;
    /// Folded backup coordinates [3*n_nodes]
    Real* folded_coords  = nullptr;
};

/**
 * @brief Airbag folding model.
 *
 * Handles initialization of folded bag geometry and sequential unfolding
 * during deployment. Each fold is undone in reverse order. During unfolding,
 * nodes are interpolated between folded and unfolded positions.
 *
 * Self-contact detection: checks minimum distance between non-adjacent
 * nodes during unfolding to prevent fabric penetration.
 */
class AirbagFolding {
public:
    AirbagFolding() = default;

    /**
     * @brief Generate folded configuration from flat coordinates.
     *
     * Applies fold operations in sequence: fold_sequence[0], [1], ...
     * Each fold reflects nodes on one side of the fold plane.
     *
     * @param bag          Folded bag state (node_coords set to folded positions)
     * @param flat_coords  Flat (unfolded) reference coordinates [3*n_nodes]
     */
    void initialize_folded(FoldedBag& bag, const Real* flat_coords) {
        if (!bag.node_coords || !flat_coords || bag.n_nodes <= 0) return;

        // Copy flat coords to working buffer
        std::memcpy(bag.node_coords, flat_coords, sizeof(Real) * 3 * bag.n_nodes);

        // Store flat reference
        if (bag.flat_coords) {
            std::memcpy(bag.flat_coords, flat_coords, sizeof(Real) * 3 * bag.n_nodes);
        }

        // Apply folds in sequence
        for (int f = 0; f < bag.n_folds; ++f) {
            apply_fold(bag.node_coords, bag.n_nodes, bag.fold_sequence[f]);
        }

        // Store folded configuration
        if (bag.folded_coords) {
            std::memcpy(bag.folded_coords, bag.node_coords,
                        sizeof(Real) * 3 * bag.n_nodes);
        }

        bag.current_fold = bag.n_folds - 1;
        bag.unfold_progress = 0.0;
        bag.fully_unfolded = (bag.n_folds == 0);
    }

    /**
     * @brief Advance the unfolding by one time step.
     *
     * Unfolds in reverse order. For each fold, nodes are linearly
     * interpolated from folded to unfolded position.
     *
     * @param bag   Bag state (node_coords updated)
     * @param dt    Time step [s]
     * @return Minimum inter-node distance (for self-contact check)
     */
    Real unfold_step(FoldedBag& bag, Real dt) {
        if (bag.fully_unfolded || bag.n_folds == 0) return 1.0e30;
        if (!bag.node_coords || !bag.flat_coords) return 1.0e30;

        // Unfolding rate: complete each fold in ~10ms
        Real unfold_rate = 100.0;  // folds per second
        bag.unfold_progress += unfold_rate * dt;

        if (bag.unfold_progress >= 1.0) {
            // Current fold fully unfolded, move to next
            bag.unfold_progress = 0.0;
            bag.current_fold--;
            if (bag.current_fold < 0) {
                bag.fully_unfolded = true;
                // Set to flat coordinates
                std::memcpy(bag.node_coords, bag.flat_coords,
                            sizeof(Real) * 3 * bag.n_nodes);
                return compute_min_distance(bag);
            }
        }

        // Interpolate: for current fold level, compute intermediate position
        // Start from folded_coords, undo folds from n_folds-1 down to current_fold
        if (bag.folded_coords) {
            std::memcpy(bag.node_coords, bag.folded_coords,
                        sizeof(Real) * 3 * bag.n_nodes);
        }

        // Undo completed folds (fully unfolded)
        for (int f = bag.n_folds - 1; f > bag.current_fold; --f) {
            undo_fold(bag.node_coords, bag.n_nodes, bag.fold_sequence[f], 1.0);
        }

        // Partially undo current fold
        if (bag.current_fold >= 0) {
            undo_fold(bag.node_coords, bag.n_nodes,
                      bag.fold_sequence[bag.current_fold], bag.unfold_progress);
        }

        return compute_min_distance(bag);
    }

    /**
     * @brief Compute minimum distance between any two nodes.
     *
     * Used for self-contact detection during unfolding.
     * Uses O(n^2) brute force — adequate for moderate node counts.
     */
    Real compute_min_distance(const FoldedBag& bag) const {
        Real min_dist = 1.0e30;
        for (int i = 0; i < bag.n_nodes; ++i) {
            for (int j = i + 1; j < bag.n_nodes; ++j) {
                Real d = airbag_detail::dist3(&bag.node_coords[3 * i],
                                              &bag.node_coords[3 * j]);
                if (d < min_dist) min_dist = d;
            }
        }
        return min_dist;
    }

private:
    /**
     * @brief Apply a fold: reflect nodes on the positive side of the fold plane.
     *
     * The fold plane passes through axis_point with normal computed from
     * axis_dir cross up. Nodes with positive signed distance are reflected.
     */
    void apply_fold(Real* coords, int n_nodes, const FoldSpec& fold) {
        // Fold plane normal: perpendicular to axis_dir in the z-direction
        // For simplicity, use axis_dir as the plane normal
        Real nx = fold.axis_dir[0];
        Real ny = fold.axis_dir[1];
        Real nz = fold.axis_dir[2];
        Real len = airbag_detail::k_sqrt(nx * nx + ny * ny + nz * nz);
        if (len < 1.0e-30) return;
        nx /= len; ny /= len; nz /= len;

        Real px = fold.axis_point[0];
        Real py = fold.axis_point[1];
        Real pz = fold.axis_point[2];

        for (int i = 0; i < n_nodes; ++i) {
            Real dx = coords[3 * i]     - px;
            Real dy = coords[3 * i + 1] - py;
            Real dz = coords[3 * i + 2] - pz;

            Real dist = dx * nx + dy * ny + dz * nz;
            if (dist > 0.0) {
                // Reflect across the fold plane
                coords[3 * i]     -= 2.0 * dist * nx;
                coords[3 * i + 1] -= 2.0 * dist * ny;
                coords[3 * i + 2] -= 2.0 * dist * nz;
            }
        }
    }

    /**
     * @brief Undo a fold by interpolating reflected nodes back.
     *
     * @param coords   Node coordinates (modified)
     * @param n_nodes  Number of nodes
     * @param fold     Fold specification
     * @param alpha    Unfold fraction [0, 1]. 0 = folded, 1 = fully unfolded.
     */
    void undo_fold(Real* coords, int n_nodes, const FoldSpec& fold, Real alpha) {
        Real nx = fold.axis_dir[0];
        Real ny = fold.axis_dir[1];
        Real nz = fold.axis_dir[2];
        Real len = airbag_detail::k_sqrt(nx * nx + ny * ny + nz * nz);
        if (len < 1.0e-30) return;
        nx /= len; ny /= len; nz /= len;

        Real px = fold.axis_point[0];
        Real py = fold.axis_point[1];
        Real pz = fold.axis_point[2];

        // For nodes on the negative side (folded over), interpolate back
        for (int i = 0; i < n_nodes; ++i) {
            Real dx = coords[3 * i]     - px;
            Real dy = coords[3 * i + 1] - py;
            Real dz = coords[3 * i + 2] - pz;

            Real dist = dx * nx + dy * ny + dz * nz;
            if (dist < 0.0) {
                // This node was reflected; partially undo: move toward reflected position
                Real reflect_x = coords[3 * i]     - 2.0 * dist * nx;
                Real reflect_y = coords[3 * i + 1] - 2.0 * dist * ny;
                Real reflect_z = coords[3 * i + 2] - 2.0 * dist * nz;

                coords[3 * i]     = coords[3 * i]     * (1.0 - alpha) + reflect_x * alpha;
                coords[3 * i + 1] = coords[3 * i + 1] * (1.0 - alpha) + reflect_y * alpha;
                coords[3 * i + 2] = coords[3 * i + 2] * (1.0 - alpha) + reflect_z * alpha;
            }
        }
    }
};

// ============================================================================
// 5. AirbagThermal — Gas Thermodynamics and Heat Transfer
// ============================================================================

/**
 * @brief Gas mixture species data.
 */
struct GasSpecies {
    Real molar_mass  = 0.029;   ///< Molar mass [kg/mol] (default: air ~29 g/mol)
    Real Cv          = 718.0;   ///< Specific heat at constant volume [J/(kg*K)]
    Real moles       = 0.0;     ///< Number of moles in the mixture
};

/**
 * @brief Airbag thermal model.
 *
 * Handles gas thermodynamics including:
 * - Ideal gas mixture: P = sum(n_i) * R_universal * T / V
 * - Heat transfer: dT/dt = (Q_inj - Q_vent - h*A*(T - T_wall)) / (m * Cv)
 * - Convective heat loss to bag fabric wall
 * - Radiative heat transfer (linearized for small temperature differences)
 *
 * The universal gas constant R = 8.314 J/(mol*K).
 */
class AirbagThermal {
public:
    static constexpr Real R_universal = 8.314;  ///< Universal gas constant [J/(mol*K)]

    AirbagThermal() = default;

    /**
     * @brief Compute gas mixture pressure using ideal gas mixture law.
     *
     * P = sum(n_i) * R * T / V
     *
     * @param species    Array of gas species
     * @param n_species  Number of species
     * @param T          Temperature [K]
     * @param V          Volume [m^3]
     * @return Pressure [Pa]
     */
    Real mixture_pressure(const GasSpecies* species, int n_species,
                          Real T, Real V) const {
        Real total_moles = 0.0;
        for (int i = 0; i < n_species; ++i) {
            total_moles += species[i].moles;
        }
        Real V_safe = airbag_detail::k_fmax(V, 1.0e-20);
        return total_moles * R_universal * T / V_safe;
    }

    /**
     * @brief Compute mixture-averaged Cv.
     *
     * Cv_mix = sum(m_i * Cv_i) / sum(m_i)
     */
    Real mixture_Cv(const GasSpecies* species, int n_species) const {
        Real total_mass = 0.0;
        Real weighted_Cv = 0.0;
        for (int i = 0; i < n_species; ++i) {
            Real mi = species[i].moles * species[i].molar_mass;
            total_mass += mi;
            weighted_Cv += mi * species[i].Cv;
        }
        if (total_mass < 1.0e-30) return 718.0;  // default air Cv
        return weighted_Cv / total_mass;
    }

    /**
     * @brief Compute total mass of gas mixture.
     */
    Real mixture_mass(const GasSpecies* species, int n_species) const {
        Real total = 0.0;
        for (int i = 0; i < n_species; ++i) {
            total += species[i].moles * species[i].molar_mass;
        }
        return total;
    }

    /**
     * @brief Update chamber temperature for one time step.
     *
     * Energy balance:
     *   m * Cv * dT/dt = Q_inj - Q_vent - h_conv * A_wall * (T - T_wall)
     *
     * where:
     *   Q_inj  = dm_inj * Cv * T_inj  (energy from injection)
     *   Q_vent = dm_vent * Cv * T      (energy lost through venting)
     *   h_conv = convective heat transfer coefficient [W/(m^2*K)]
     *   A_wall = fabric surface area [m^2]
     *   T_wall = wall temperature [K]
     *
     * @param chamber    Chamber state (temperature updated)
     * @param h_conv     Convective heat transfer coefficient [W/(m^2*K)]
     * @param A_wall     Fabric wall area [m^2]
     * @param T_wall     Wall temperature [K]
     * @param Cv         Specific heat at constant volume [J/(kg*K)]
     * @param dt         Time step [s]
     */
    void compute_thermal(AirbagChamber& chamber, Real h_conv, Real A_wall,
                         Real T_wall, Real Cv, Real dt) {
        if (chamber.mass < 1.0e-20) return;

        // Convective heat loss rate
        Real Q_conv = h_conv * A_wall * (chamber.temperature - T_wall);

        // Temperature rate of change
        Real dT_dt = -Q_conv / (chamber.mass * Cv);

        // Forward Euler update
        chamber.temperature += dT_dt * dt;

        // Clamp to physical range
        chamber.temperature = airbag_detail::k_fmax(chamber.temperature, 1.0);
    }

    /**
     * @brief Compute convective + radiative heat transfer.
     *
     * Q_total = h_conv * A * (T - T_wall) + epsilon * sigma * A * (T^4 - T_wall^4)
     *
     * For computational efficiency, radiative term is linearized:
     *   Q_rad ~ epsilon * sigma * A * 4 * T_wall^3 * (T - T_wall)
     *
     * @param T         Gas temperature [K]
     * @param T_wall    Wall temperature [K]
     * @param h_conv    Convective coefficient [W/(m^2*K)]
     * @param A_wall    Surface area [m^2]
     * @param emissivity Surface emissivity (0 to 1)
     * @return Total heat transfer rate [W] (positive = heat loss from gas)
     */
    Real compute_heat_loss(Real T, Real T_wall, Real h_conv, Real A_wall,
                           Real emissivity) const {
        constexpr Real sigma_sb = 5.67e-8;  // Stefan-Boltzmann constant

        Real Q_conv = h_conv * A_wall * (T - T_wall);

        // Linearized radiation
        Real T_ref = airbag_detail::k_fmax(T_wall, 1.0);
        Real h_rad = emissivity * sigma_sb * 4.0 * T_ref * T_ref * T_ref;
        Real Q_rad = h_rad * A_wall * (T - T_wall);

        return Q_conv + Q_rad;
    }

    /**
     * @brief Compute equilibrium temperature (no injection/venting).
     *
     * At equilibrium: h * A * (T_eq - T_wall) = 0, so T_eq = T_wall
     * With radiation: equilibrium requires iterative solve, but for
     * linearized radiation, T_eq = T_wall.
     *
     * @param T_wall   Wall temperature [K]
     * @return Equilibrium gas temperature [K]
     */
    Real equilibrium_temperature(Real T_wall) const {
        return T_wall;
    }

    /**
     * @brief Compute time constant for thermal decay.
     *
     * tau = m * Cv / (h * A)
     *
     * Temperature decays as: T(t) = T_wall + (T0 - T_wall) * exp(-t/tau)
     *
     * @param mass     Gas mass [kg]
     * @param Cv       Specific heat [J/(kg*K)]
     * @param h_conv   Convective coefficient [W/(m^2*K)]
     * @param A_wall   Surface area [m^2]
     * @return Time constant [s]
     */
    Real thermal_time_constant(Real mass, Real Cv, Real h_conv, Real A_wall) const {
        Real denom = h_conv * A_wall;
        if (denom < 1.0e-30) return 1.0e30;
        return mass * Cv / denom;
    }
};

} // namespace fem
} // namespace nxs
