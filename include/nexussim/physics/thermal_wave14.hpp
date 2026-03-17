#pragma once

/**
 * @file thermal_wave14.hpp
 * @brief Wave 14: Comprehensive thermal solver capabilities for NexusSim
 *
 * Extends the basic thermal solver (thermal_solver.hpp) with:
 * - HeatConductionSolver: explicit forward Euler with lumped capacity matrix
 * - ConvectionBC: Newton's cooling law boundary condition
 * - RadiationBC: Stefan-Boltzmann radiation boundary condition
 * - FixedTemperatureBC: prescribed (Dirichlet) temperature BC
 * - HeatFluxBC: prescribed (Neumann) heat flux BC
 * - AdiabaticHeating: Taylor-Quinney plastic work to heat conversion
 * - ThermalTimeStep: stability limit computation
 * - CoupledThermoMechanical: staggered thermo-mechanical coupling manager
 *
 * Heat equation solved:
 *   dT/dt = (k / (rho * Cp)) * nabla^2 T + Q / (rho * Cp)
 */

#include <nexussim/physics/material.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <functional>
#include <limits>
#include <numeric>

namespace nxs {
namespace physics {

// ============================================================================
// Constants
// ============================================================================

/// Stefan-Boltzmann constant (W / m^2 / K^4)
inline constexpr Real STEFAN_BOLTZMANN = 5.670374419e-8;

// ============================================================================
// 1. HeatConductionSolver
// ============================================================================

/**
 * @brief Explicit forward-Euler heat conduction solver with lumped capacity matrix
 *
 * Solves the transient heat equation on a nodal temperature field:
 *   dT/dt = alpha * nabla^2 T + Q / (rho * Cp)
 *
 * where alpha = k / (rho * Cp) is the thermal diffusivity.
 *
 * The spatial Laplacian is approximated via a connectivity-based finite
 * difference stencil. For each node, the contribution from neighbor j is:
 *   (k * A_ij / d_ij) * (T_j - T_i)
 * summed into the lumped heat rate, then divided by (rho * Cp * V_i).
 *
 * Usage:
 * @code
 *   HeatConductionSolver solver;
 *   solver.initialize(num_nodes);
 *   solver.set_conductivity(50.0);  // W/m-K steel
 *   solver.add_heat_source(node_id, Q);
 *   solver.step(dt);
 *   Real T = solver.get_temperature(node_id);
 * @endcode
 */
class HeatConductionSolver {
public:
    HeatConductionSolver() = default;

    /**
     * @brief Initialize the solver with the given number of nodes
     * @param num_nodes Number of nodal degrees of freedom
     * @param initial_temperature Uniform initial temperature (K)
     * @param density Material density (kg/m^3)
     * @param specific_heat Specific heat capacity (J/kg-K)
     */
    void initialize(Index num_nodes,
                    Real initial_temperature = 293.15,
                    Real density = 7850.0,
                    Real specific_heat = 500.0) {
        num_nodes_ = num_nodes;
        temperature_.assign(num_nodes, initial_temperature);
        temperature_old_.assign(num_nodes, initial_temperature);
        heat_rate_.assign(num_nodes, 0.0);
        heat_source_.assign(num_nodes, 0.0);
        nodal_volume_.assign(num_nodes, 1.0);
        conductivity_ = 50.0;
        density_ = density;
        specific_heat_ = specific_heat;
        initialized_ = true;
    }

    /**
     * @brief Set uniform thermal conductivity
     * @param k Thermal conductivity (W/m-K)
     */
    void set_conductivity(Real k) { conductivity_ = k; }
    Real conductivity() const { return conductivity_; }

    /**
     * @brief Set material density
     * @param rho Density (kg/m^3)
     */
    void set_density(Real rho) { density_ = rho; }
    Real density() const { return density_; }

    /**
     * @brief Set specific heat capacity
     * @param Cp Specific heat (J/kg-K)
     */
    void set_specific_heat(Real Cp) { specific_heat_ = Cp; }
    Real specific_heat() const { return specific_heat_; }

    /**
     * @brief Set nodal volume for lumped capacity
     * @param node Node index
     * @param volume Nodal volume (m^3)
     */
    void set_nodal_volume(Index node, Real volume) {
        if (node < num_nodes_) nodal_volume_[node] = volume;
    }

    /**
     * @brief Add a neighbor connection for the FD Laplacian stencil
     * @param node_i Source node
     * @param node_j Neighbor node
     * @param area_over_dist Contact area divided by distance A_ij / d_ij (m)
     */
    void add_connection(Index node_i, Index node_j, Real area_over_dist) {
        connections_.push_back({node_i, node_j, area_over_dist});
    }

    /**
     * @brief Add volumetric heat source at a node
     * @param node Node index
     * @param Q Heat generation rate (W/m^3)
     */
    void add_heat_source(Index node, Real Q) {
        if (node < num_nodes_) heat_source_[node] += Q;
    }

    /**
     * @brief Clear all heat sources
     */
    void clear_heat_sources() {
        std::fill(heat_source_.begin(), heat_source_.end(), 0.0);
    }

    /**
     * @brief Perform one explicit forward-Euler time step
     * @param dt Time step size (s)
     *
     * Update formula (lumped capacity):
     *   T_i^{n+1} = T_i^n + dt / (rho * Cp) * [ sum_j k*A_ij/d_ij*(T_j - T_i)/V_i + Q_i ]
     */
    void step(Real dt) {
        // Save old temperatures
        temperature_old_ = temperature_;

        // Zero heat rate
        std::fill(heat_rate_.begin(), heat_rate_.end(), 0.0);

        // Accumulate conduction contributions from connectivity
        for (const auto& conn : connections_) {
            Real dT = temperature_old_[conn.node_j] - temperature_old_[conn.node_i];
            Real flux = conductivity_ * conn.area_over_dist * dT;
            heat_rate_[conn.node_i] += flux / nodal_volume_[conn.node_i];
            heat_rate_[conn.node_j] -= flux / nodal_volume_[conn.node_j];
        }

        // Add volumetric heat sources
        for (Index i = 0; i < num_nodes_; ++i) {
            heat_rate_[i] += heat_source_[i];
        }

        // Forward Euler update: T^{n+1} = T^n + dt * heat_rate / (rho * Cp)
        Real rho_Cp = density_ * specific_heat_;
        for (Index i = 0; i < num_nodes_; ++i) {
            temperature_[i] = temperature_old_[i] + dt * heat_rate_[i] / rho_Cp;
        }
    }

    /**
     * @brief Get temperature at a node
     */
    Real get_temperature(Index node) const {
        return (node < num_nodes_) ? temperature_[node] : 0.0;
    }

    /**
     * @brief Set temperature at a node
     */
    void set_temperature(Index node, Real T) {
        if (node < num_nodes_) temperature_[node] = T;
    }

    /**
     * @brief Get number of nodes
     */
    Index num_nodes() const { return num_nodes_; }

    /**
     * @brief Direct access to temperature array
     */
    const std::vector<Real>& temperatures() const { return temperature_; }
    std::vector<Real>& temperatures() { return temperature_; }

    /**
     * @brief Direct access to heat rate array
     */
    const std::vector<Real>& heat_rates() const { return heat_rate_; }

    /**
     * @brief Check if solver is initialized
     */
    bool initialized() const { return initialized_; }

private:
    /// Connection between two nodes for FD stencil
    struct Connection {
        Index node_i;
        Index node_j;
        Real area_over_dist;  ///< A_ij / d_ij (m)
    };

    Index num_nodes_ = 0;
    std::vector<Real> temperature_;
    std::vector<Real> temperature_old_;
    std::vector<Real> heat_rate_;
    std::vector<Real> heat_source_;
    std::vector<Real> nodal_volume_;
    std::vector<Connection> connections_;

    Real conductivity_ = 50.0;     // W/m-K (steel default)
    Real density_ = 7850.0;        // kg/m^3
    Real specific_heat_ = 500.0;   // J/kg-K
    bool initialized_ = false;
};

// ============================================================================
// 2. ConvectionBC
// ============================================================================

/**
 * @brief Convection boundary condition: q = h * (T - T_inf)
 *
 * Applies Newton's law of cooling on boundary faces.
 * The heat loss from node i is: Q_conv = -h * A * (T_i - T_ambient)
 * which is added to the nodal heat rate (negative = cooling).
 */
struct ConvectionCondition {
    std::vector<Index> boundary_nodes; ///< Nodes on convective boundary
    Real h_coefficient;                ///< Convective heat transfer coefficient (W/m^2-K)
    Real T_ambient;                    ///< Ambient fluid temperature (K)
};

class ConvectionBC {
public:
    ConvectionBC() = default;

    /**
     * @brief Construct with a single convection condition
     */
    explicit ConvectionBC(const ConvectionCondition& cond) {
        conditions_.push_back(cond);
    }

    /**
     * @brief Add a convection condition
     */
    void add_condition(const ConvectionCondition& cond) {
        conditions_.push_back(cond);
    }

    /**
     * @brief Set per-node boundary area (m^2)
     * @param areas Vector of areas indexed by node ID
     */
    void set_boundary_areas(const std::vector<Real>& areas) {
        boundary_areas_ = areas;
    }

    /**
     * @brief Set uniform boundary area for all nodes
     */
    void set_uniform_area(Real area) {
        uniform_area_ = area;
        use_uniform_area_ = true;
    }

    /**
     * @brief Apply convection BC to temperature array and heat rate array
     *
     * For each boundary node i:
     *   heat_rate[i] += -h * A * (T[i] - T_ambient)
     *
     * @param temperatures Current nodal temperature array
     * @param heat_rates Nodal heat rate array (modified in place)
     * @param rho_Cp Product rho * Cp for converting flux to dT/dt
     * @param dt Time step (for explicit integration)
     */
    void apply(std::vector<Real>& temperatures,
               std::vector<Real>& heat_rates,
               Real rho_Cp, Real dt) const {
        (void)dt;  // dt unused; heat_rate is accumulated
        for (const auto& cond : conditions_) {
            for (Index node : cond.boundary_nodes) {
                if (node >= temperatures.size()) continue;
                Real A = get_area(node);
                Real q_conv = -cond.h_coefficient * A * (temperatures[node] - cond.T_ambient);
                heat_rates[node] += q_conv;
            }
        }
    }

    /**
     * @brief Apply directly to solver temperatures (convenience for simple use)
     *
     * Computes dT = -h * A * (T - T_inf) * dt / (rho * Cp * V) and adds to T.
     *
     * @param temperatures Nodal temperature array (modified in place)
     * @param dt Time step
     * @param rho Density
     * @param Cp Specific heat
     * @param nodal_volume Volume per node
     */
    void apply_direct(std::vector<Real>& temperatures,
                      Real dt, Real rho, Real Cp, Real nodal_volume) const {
        Real rho_Cp_V = rho * Cp * nodal_volume;
        for (const auto& cond : conditions_) {
            for (Index node : cond.boundary_nodes) {
                if (node >= temperatures.size()) continue;
                Real A = get_area(node);
                Real dT = -cond.h_coefficient * A * (temperatures[node] - cond.T_ambient) * dt / rho_Cp_V;
                temperatures[node] += dT;
            }
        }
    }

    const std::vector<ConvectionCondition>& conditions() const { return conditions_; }

private:
    Real get_area(Index node) const {
        if (use_uniform_area_) return uniform_area_;
        if (node < boundary_areas_.size()) return boundary_areas_[node];
        return 1.0;  // Default unit area
    }

    std::vector<ConvectionCondition> conditions_;
    std::vector<Real> boundary_areas_;
    Real uniform_area_ = 1.0;
    bool use_uniform_area_ = false;
};

// ============================================================================
// 3. RadiationBC
// ============================================================================

/**
 * @brief Radiation boundary condition: q = sigma * epsilon * F * (T^4 - T_env^4)
 *
 * Implements Stefan-Boltzmann radiation heat transfer on boundary faces.
 * The net radiative heat loss from node i is:
 *   Q_rad = -sigma_SB * epsilon * F_view * A * (T_i^4 - T_env^4)
 *
 * For linearized form (used in stability analysis):
 *   h_rad ~ 4 * sigma * epsilon * T^3
 */
struct RadiationCondition {
    std::vector<Index> boundary_nodes; ///< Nodes on radiative boundary
    Real emissivity;                   ///< Surface emissivity (0-1)
    Real T_environment;                ///< Environment temperature (K)
    Real view_factor;                  ///< Geometric view factor (0-1)
};

class RadiationBC {
public:
    RadiationBC() = default;

    /**
     * @brief Construct with a single radiation condition
     */
    explicit RadiationBC(const RadiationCondition& cond) {
        conditions_.push_back(cond);
    }

    /**
     * @brief Add a radiation condition
     */
    void add_condition(const RadiationCondition& cond) {
        conditions_.push_back(cond);
    }

    /**
     * @brief Set per-node boundary area (m^2)
     */
    void set_boundary_areas(const std::vector<Real>& areas) {
        boundary_areas_ = areas;
    }

    /**
     * @brief Set uniform boundary area for all nodes
     */
    void set_uniform_area(Real area) {
        uniform_area_ = area;
        use_uniform_area_ = true;
    }

    /**
     * @brief Apply radiation BC to heat rate array
     *
     * For each boundary node i:
     *   heat_rate[i] += -sigma * eps * F * A * (T_i^4 - T_env^4)
     */
    void apply(std::vector<Real>& temperatures,
               std::vector<Real>& heat_rates) const {
        for (const auto& cond : conditions_) {
            Real T_env = cond.T_environment;
            Real T_env4 = T_env * T_env * T_env * T_env;
            for (Index node : cond.boundary_nodes) {
                if (node >= temperatures.size()) continue;
                Real T = temperatures[node];
                Real T4 = T * T * T * T;
                Real A = get_area(node);
                Real q_rad = -STEFAN_BOLTZMANN * cond.emissivity * cond.view_factor * A * (T4 - T_env4);
                heat_rates[node] += q_rad;
            }
        }
    }

    /**
     * @brief Apply directly to temperatures (convenience)
     */
    void apply_direct(std::vector<Real>& temperatures,
                      Real dt, Real rho, Real Cp, Real nodal_volume) const {
        Real rho_Cp_V = rho * Cp * nodal_volume;
        for (const auto& cond : conditions_) {
            Real T_env = cond.T_environment;
            Real T_env4 = T_env * T_env * T_env * T_env;
            for (Index node : cond.boundary_nodes) {
                if (node >= temperatures.size()) continue;
                Real T = temperatures[node];
                Real T4 = T * T * T * T;
                Real A = get_area(node);
                Real q_rad = -STEFAN_BOLTZMANN * cond.emissivity * cond.view_factor * A * (T4 - T_env4);
                Real dT = q_rad * dt / rho_Cp_V;
                temperatures[node] += dT;
            }
        }
    }

    /**
     * @brief Compute linearized radiation heat transfer coefficient
     *
     * h_rad = 4 * sigma_SB * epsilon * T^3
     *
     * @param emissivity Surface emissivity
     * @param T Temperature at which to linearize (K)
     * @return Linearized h_rad (W/m^2-K)
     */
    static Real linearized_h_rad(Real emissivity, Real T) {
        return 4.0 * STEFAN_BOLTZMANN * emissivity * T * T * T;
    }

    /**
     * @brief Compute linearized coefficient with view factor
     */
    static Real linearized_h_rad(Real emissivity, Real view_factor, Real T) {
        return 4.0 * STEFAN_BOLTZMANN * emissivity * view_factor * T * T * T;
    }

    const std::vector<RadiationCondition>& conditions() const { return conditions_; }

private:
    Real get_area(Index node) const {
        if (use_uniform_area_) return uniform_area_;
        if (node < boundary_areas_.size()) return boundary_areas_[node];
        return 1.0;
    }

    std::vector<RadiationCondition> conditions_;
    std::vector<Real> boundary_areas_;
    Real uniform_area_ = 1.0;
    bool use_uniform_area_ = false;
};

// ============================================================================
// 4. FixedTemperatureBC
// ============================================================================

/**
 * @brief Fixed (Dirichlet) temperature boundary condition
 *
 * Prescribes T at specified nodes. After each solver step,
 * apply() overwrites the nodal temperature to the prescribed value.
 */
struct FixedTempCondition {
    std::vector<Index> nodes;  ///< Nodes with fixed temperature
    Real temperature;          ///< Prescribed temperature (K)
};

class FixedTemperatureBC {
public:
    FixedTemperatureBC() = default;

    /**
     * @brief Construct with a single condition
     */
    explicit FixedTemperatureBC(const FixedTempCondition& cond) {
        conditions_.push_back(cond);
    }

    /**
     * @brief Add a fixed temperature condition
     */
    void add_condition(const FixedTempCondition& cond) {
        conditions_.push_back(cond);
    }

    /**
     * @brief Apply fixed temperature BC by overwriting nodal temperatures
     * @param temperatures Nodal temperature array (modified in place)
     */
    void apply(std::vector<Real>& temperatures) const {
        for (const auto& cond : conditions_) {
            for (Index node : cond.nodes) {
                if (node < temperatures.size()) {
                    temperatures[node] = cond.temperature;
                }
            }
        }
    }

    /**
     * @brief Clear all conditions
     */
    void clear() { conditions_.clear(); }

    const std::vector<FixedTempCondition>& conditions() const { return conditions_; }

private:
    std::vector<FixedTempCondition> conditions_;
};

// ============================================================================
// 5. HeatFluxBC
// ============================================================================

/**
 * @brief Prescribed heat flux (Neumann) boundary condition
 *
 * Applies a prescribed heat flux q (W/m^2) on boundary faces:
 *   Q_node = q * A
 * where A is the boundary area associated with the node.
 */
struct HeatFluxCondition {
    std::vector<Index> boundary_nodes; ///< Nodes on the flux boundary
    Real flux;                         ///< Prescribed heat flux (W/m^2), positive = into domain
};

class HeatFluxBC {
public:
    HeatFluxBC() = default;

    /**
     * @brief Construct with a single condition
     */
    explicit HeatFluxBC(const HeatFluxCondition& cond) {
        conditions_.push_back(cond);
    }

    /**
     * @brief Add a heat flux condition
     */
    void add_condition(const HeatFluxCondition& cond) {
        conditions_.push_back(cond);
    }

    /**
     * @brief Set per-node boundary area (m^2)
     */
    void set_boundary_areas(const std::vector<Real>& areas) {
        boundary_areas_ = areas;
    }

    /**
     * @brief Set uniform boundary area
     */
    void set_uniform_area(Real area) {
        uniform_area_ = area;
        use_uniform_area_ = true;
    }

    /**
     * @brief Apply heat flux BC to heat rate array
     *
     * For each boundary node:
     *   heat_rate[node] += flux * A
     */
    void apply(std::vector<Real>& heat_rates) const {
        for (const auto& cond : conditions_) {
            for (Index node : cond.boundary_nodes) {
                if (node >= heat_rates.size()) continue;
                Real A = get_area(node);
                heat_rates[node] += cond.flux * A;
            }
        }
    }

    /**
     * @brief Apply directly to temperatures (convenience)
     *
     * dT = flux * A * dt / (rho * Cp * V)
     */
    void apply_direct(std::vector<Real>& temperatures,
                      Real dt, Real rho, Real Cp, Real nodal_volume) const {
        Real rho_Cp_V = rho * Cp * nodal_volume;
        for (const auto& cond : conditions_) {
            for (Index node : cond.boundary_nodes) {
                if (node >= temperatures.size()) continue;
                Real A = get_area(node);
                Real dT = cond.flux * A * dt / rho_Cp_V;
                temperatures[node] += dT;
            }
        }
    }

    const std::vector<HeatFluxCondition>& conditions() const { return conditions_; }

private:
    Real get_area(Index node) const {
        if (use_uniform_area_) return uniform_area_;
        if (node < boundary_areas_.size()) return boundary_areas_[node];
        return 1.0;
    }

    std::vector<HeatFluxCondition> conditions_;
    std::vector<Real> boundary_areas_;
    Real uniform_area_ = 1.0;
    bool use_uniform_area_ = false;
};

// ============================================================================
// 6. AdiabaticHeating
// ============================================================================

/**
 * @brief Adiabatic heating from plastic work (Taylor-Quinney effect)
 *
 * In high-rate deformation, a fraction of plastic work converts to heat:
 *   Delta_T = eta * sigma_eq * Delta_eps_p / (rho * Cp)
 *
 * where eta is the Taylor-Quinney coefficient (typically 0.9 for metals).
 *
 * This is a utility class providing static methods for use in explicit
 * solver loops.
 */
class AdiabaticHeating {
public:
    /**
     * @brief Compute temperature rise from plastic dissipation
     *
     * @param stress Equivalent (von Mises) stress (Pa)
     * @param delta_eps_p Increment of equivalent plastic strain (dimensionless)
     * @param rho Density (kg/m^3)
     * @param Cp Specific heat capacity (J/kg-K)
     * @param eta Taylor-Quinney coefficient (default 0.9)
     * @return Temperature increment (K)
     */
    static Real compute_heating(Real stress, Real delta_eps_p,
                                Real rho, Real Cp, Real eta = 0.9) {
        if (rho <= 0.0 || Cp <= 0.0) return 0.0;
        return eta * stress * delta_eps_p / (rho * Cp);
    }

    /**
     * @brief Compute temperature rise from plastic work rate over a time step
     *
     * @param plastic_power Plastic power per unit volume sigma : eps_dot_p (W/m^3)
     * @param dt Time step (s)
     * @param rho Density (kg/m^3)
     * @param Cp Specific heat (J/kg-K)
     * @param eta Taylor-Quinney coefficient
     * @return Temperature increment (K)
     */
    static Real compute_heating_from_power(Real plastic_power, Real dt,
                                           Real rho, Real Cp, Real eta = 0.9) {
        if (rho <= 0.0 || Cp <= 0.0) return 0.0;
        return eta * plastic_power * dt / (rho * Cp);
    }

    /**
     * @brief Apply adiabatic heating to a MaterialState
     *
     * Updates the temperature field of the given state based on
     * the plastic strain increment stored in history[0] compared to
     * a previous value.
     *
     * @param state Material state (temperature modified in place)
     * @param prev_eps_p Previous accumulated plastic strain
     * @param rho Density
     * @param Cp Specific heat
     * @param eta Taylor-Quinney coefficient
     */
    static void apply_to_state(MaterialState& state, Real prev_eps_p,
                               Real rho, Real Cp, Real eta = 0.9) {
        Real sigma_eq = Material::von_mises_stress(state.stress);
        Real delta_eps_p = state.plastic_strain - prev_eps_p;
        if (delta_eps_p > 0.0) {
            Real dT = compute_heating(sigma_eq, delta_eps_p, rho, Cp, eta);
            state.temperature += dT;
        }
    }

    /// Default Taylor-Quinney coefficient for common metals
    static constexpr Real DEFAULT_ETA = 0.9;
};

// ============================================================================
// 7. ThermalTimeStep
// ============================================================================

/**
 * @brief Thermal time step stability computation
 *
 * For an explicit forward-Euler scheme, the stable time step is:
 *   dt_thermal = h^2 * rho * Cp / (2 * k)
 *
 * where h is the minimum element characteristic length.
 * In 3D with the full stencil, the factor 2 may be replaced by 2*ndim
 * for conservative estimates. We use the 1D factor 2 here.
 *
 * The thermal time step is typically much larger than the mechanical
 * time step, allowing subcycling.
 */
class ThermalTimeStep {
public:
    /**
     * @brief Compute the stable thermal time step
     *
     * @param min_element_size Minimum element characteristic length h (m)
     * @param conductivity Thermal conductivity k (W/m-K)
     * @param density Density rho (kg/m^3)
     * @param specific_heat Specific heat Cp (J/kg-K)
     * @return Stable time step (s)
     */
    static Real compute_stable_dt(Real min_element_size,
                                  Real conductivity,
                                  Real density,
                                  Real specific_heat) {
        if (conductivity <= 0.0) return std::numeric_limits<Real>::max();
        return min_element_size * min_element_size * density * specific_heat
               / (2.0 * conductivity);
    }

    /**
     * @brief Compute thermal diffusivity alpha = k / (rho * Cp)
     */
    static Real diffusivity(Real conductivity, Real density, Real specific_heat) {
        return conductivity / (density * specific_heat);
    }

    /**
     * @brief Compute the ratio of thermal to mechanical time step
     *
     * Mechanical: dt_mech ~ h / c  (c = sound speed)
     * Thermal:    dt_therm ~ h^2 * rho * Cp / (2 * k)
     *
     * Ratio = dt_therm / dt_mech = h * rho * Cp * c / (2 * k)
     *
     * @param min_element_size h (m)
     * @param conductivity k (W/m-K)
     * @param density rho (kg/m^3)
     * @param specific_heat Cp (J/kg-K)
     * @param sound_speed c (m/s)
     * @return dt_thermal / dt_mechanical
     */
    static Real thermal_to_mechanical_ratio(Real min_element_size,
                                            Real conductivity,
                                            Real density,
                                            Real specific_heat,
                                            Real sound_speed) {
        if (conductivity <= 0.0 || sound_speed <= 0.0)
            return std::numeric_limits<Real>::max();
        Real dt_therm = compute_stable_dt(min_element_size, conductivity,
                                          density, specific_heat);
        Real dt_mech = min_element_size / sound_speed;
        return dt_therm / dt_mech;
    }

    /**
     * @brief Compute recommended number of thermal subcycles per mechanical step
     *
     * N_sub = ceil(dt_mech_total / dt_thermal_stable)
     * Returns at least 1.
     */
    static int recommended_subcycles(Real mechanical_dt,
                                     Real min_element_size,
                                     Real conductivity,
                                     Real density,
                                     Real specific_heat) {
        Real dt_therm = compute_stable_dt(min_element_size, conductivity,
                                          density, specific_heat);
        if (dt_therm <= 0.0) return 1;
        int n = static_cast<int>(std::ceil(mechanical_dt / dt_therm));
        return std::max(n, 1);
    }
};

// ============================================================================
// 8. CoupledThermoMechanical
// ============================================================================

/**
 * @brief Configuration for coupled thermo-mechanical analysis
 */
struct CouplingConfig {
    int thermal_subcycles = 1;      ///< Number of thermal sub-steps per mechanical step
    Real taylor_quinney = 0.9;      ///< Taylor-Quinney coefficient
    bool staggered = true;          ///< Use staggered (isothermal split) coupling
    Real reference_temperature = 293.15; ///< Reference temperature for thermal expansion (K)
};

/**
 * @brief Manages staggered thermo-mechanical coupling
 *
 * Coupling loop per mechanical time step:
 * 1. Mechanical step (with current temperature field)
 * 2. Compute adiabatic heating from plastic work
 * 3. Thermal sub-stepping (with heat conduction + BCs)
 * 4. Update material properties from new temperature field
 *
 * Usage:
 * @code
 *   CoupledThermoMechanical coupled;
 *   CouplingConfig cfg;
 *   cfg.thermal_subcycles = 10;
 *   cfg.taylor_quinney = 0.9;
 *   coupled.configure(cfg);
 *   coupled.set_thermal_solver(&thermal);
 *
 *   // In time loop:
 *   coupled.step(dt, mechanical_step_fn, material_update_fn);
 * @endcode
 */
class CoupledThermoMechanical {
public:
    using MechanicalStepFn = std::function<void(Real dt)>;
    using MaterialUpdateFn = std::function<void(const std::vector<Real>& temperatures)>;

    CoupledThermoMechanical() = default;

    /**
     * @brief Set coupling configuration
     */
    void configure(const CouplingConfig& cfg) { config_ = cfg; }
    const CouplingConfig& config() const { return config_; }

    /**
     * @brief Set the thermal solver to use
     */
    void set_thermal_solver(HeatConductionSolver* solver) { thermal_ = solver; }

    /**
     * @brief Set convection BCs (optional)
     */
    void set_convection_bc(const ConvectionBC* bc) { convection_ = bc; }

    /**
     * @brief Set radiation BCs (optional)
     */
    void set_radiation_bc(const RadiationBC* bc) { radiation_ = bc; }

    /**
     * @brief Set fixed temperature BCs (optional)
     */
    void set_fixed_temp_bc(const FixedTemperatureBC* bc) { fixed_temp_ = bc; }

    /**
     * @brief Set heat flux BCs (optional)
     */
    void set_heat_flux_bc(const HeatFluxBC* bc) { heat_flux_ = bc; }

    /**
     * @brief Add adiabatic heating at a node from plastic dissipation
     *
     * Called by the mechanical solver after computing plastic strain increments.
     * The heat is added as a volumetric source for the next thermal sub-step.
     *
     * @param node Node index
     * @param sigma_eq Equivalent stress (Pa)
     * @param delta_eps_p Plastic strain increment
     * @param rho Density
     * @param Cp Specific heat
     */
    void add_adiabatic_heat(Index node, Real sigma_eq, Real delta_eps_p,
                            Real rho, Real Cp) {
        if (!thermal_) return;
        Real dT = AdiabaticHeating::compute_heating(
            sigma_eq, delta_eps_p, rho, Cp, config_.taylor_quinney);
        // Store as volumetric heat source: Q = rho * Cp * dT / dt
        // This will be consumed in the thermal step
        adiabatic_dT_[node] += dT;
    }

    /**
     * @brief Perform one coupled thermo-mechanical time step
     *
     * Sequence (staggered):
     * 1. Execute mechanical step with current temperature
     * 2. Apply adiabatic heating increments
     * 3. Subcycled thermal conduction steps
     * 4. Apply all thermal BCs at each sub-step
     * 5. Call material update with new temperatures
     *
     * @param dt Mechanical time step (s)
     * @param mechanical_step Function that performs one mechanical step
     * @param material_update Function that updates material properties from temperature
     */
    void step(Real dt,
              MechanicalStepFn mechanical_step,
              MaterialUpdateFn material_update = nullptr) {
        if (!thermal_) return;

        // 1. Mechanical step
        if (mechanical_step) {
            mechanical_step(dt);
        }

        // 2. Apply adiabatic heating directly to temperatures
        auto& temps = thermal_->temperatures();
        for (auto& [node, dT] : adiabatic_dT_) {
            if (node < temps.size()) {
                temps[node] += dT;
            }
        }
        adiabatic_dT_.clear();

        // Track energy for conservation check
        Real energy_before = compute_thermal_energy();

        // 3. Subcycled thermal conduction steps
        int nsub = std::max(config_.thermal_subcycles, 1);
        Real dt_sub = dt / static_cast<Real>(nsub);

        for (int sub = 0; sub < nsub; ++sub) {
            // Thermal conduction step
            thermal_->step(dt_sub);

            // Apply boundary conditions
            apply_all_bcs(dt_sub);
        }

        Real energy_after = compute_thermal_energy();
        energy_balance_ = energy_after - energy_before;

        // 4. Update material properties from new temperature field
        if (material_update) {
            material_update(thermal_->temperatures());
        }

        total_time_ += dt;
        step_count_++;
    }

    /**
     * @brief Get cumulative simulation time
     */
    Real total_time() const { return total_time_; }

    /**
     * @brief Get total number of coupled steps taken
     */
    int step_count() const { return step_count_; }

    /**
     * @brief Get energy balance from last step (for conservation checks)
     *
     * Positive = energy gained, negative = energy lost
     */
    Real energy_balance() const { return energy_balance_; }

    /**
     * @brief Compute total thermal energy in the system: sum(rho * Cp * T_i * V_i)
     *
     * Uses solver density and specific heat (uniform assumption).
     */
    Real compute_thermal_energy() const {
        if (!thermal_) return 0.0;
        Real rho_Cp = thermal_->density() * thermal_->specific_heat();
        Real total = 0.0;
        const auto& temps = thermal_->temperatures();
        for (Index i = 0; i < thermal_->num_nodes(); ++i) {
            total += rho_Cp * temps[i];  // V=1 per node
        }
        return total;
    }

private:
    /**
     * @brief Apply all registered boundary conditions
     */
    void apply_all_bcs(Real dt_sub) {
        if (!thermal_) return;
        auto& temps = thermal_->temperatures();

        // Convection
        if (convection_) {
            Real rho = thermal_->density();
            Real Cp = thermal_->specific_heat();
            Real V = 1.0;  // unit nodal volume
            const_cast<ConvectionBC*>(convection_)->apply_direct(temps, dt_sub, rho, Cp, V);
        }

        // Radiation
        if (radiation_) {
            Real rho = thermal_->density();
            Real Cp = thermal_->specific_heat();
            Real V = 1.0;
            const_cast<RadiationBC*>(radiation_)->apply_direct(temps, dt_sub, rho, Cp, V);
        }

        // Heat flux
        if (heat_flux_) {
            Real rho = thermal_->density();
            Real Cp = thermal_->specific_heat();
            Real V = 1.0;
            const_cast<HeatFluxBC*>(heat_flux_)->apply_direct(temps, dt_sub, rho, Cp, V);
        }

        // Fixed temperature (always applied last, overwrites)
        if (fixed_temp_) {
            fixed_temp_->apply(temps);
        }
    }

    CouplingConfig config_;
    HeatConductionSolver* thermal_ = nullptr;
    const ConvectionBC* convection_ = nullptr;
    const RadiationBC* radiation_ = nullptr;
    const FixedTemperatureBC* fixed_temp_ = nullptr;
    const HeatFluxBC* heat_flux_ = nullptr;

    std::map<Index, Real> adiabatic_dT_;  ///< Buffered adiabatic temperature increments

    Real total_time_ = 0.0;
    int step_count_ = 0;
    Real energy_balance_ = 0.0;
};

} // namespace physics
} // namespace nxs
