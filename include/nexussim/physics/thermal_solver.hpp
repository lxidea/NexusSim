#pragma once

/**
 * @file thermal_solver.hpp
 * @brief Thermal solver for heat conduction and thermo-mechanical coupling
 *
 * Features:
 * - Explicit heat conduction solver (forward Euler)
 * - Nodal temperature field
 * - Thermal expansion coupling to FEM
 * - Adiabatic heating from plastic work
 * - Convection/radiation boundary conditions
 * - GPU acceleration via Kokkos
 *
 * Heat equation: ρ*c*∂T/∂t = ∇·(k∇T) + Q
 *
 * Where:
 *   ρ = density
 *   c = specific heat capacity
 *   k = thermal conductivity
 *   Q = heat source (plastic work, radiation, etc.)
 */

#include <nexussim/core/core.hpp>
#include <nexussim/physics/module.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/data/state.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <memory>
#include <vector>
#include <map>
#include <iostream>

namespace nxs {
namespace physics {

// ============================================================================
// Thermal Boundary Condition Types
// ============================================================================

enum class ThermalBCType {
    Temperature,     ///< Prescribed temperature (Dirichlet)
    HeatFlux,        ///< Prescribed heat flux (Neumann)
    Convection,      ///< Convection: q = h*(T - T_inf)
    Radiation        ///< Radiation: q = ε*σ*(T^4 - T_inf^4)
};

/**
 * @brief Thermal boundary condition specification
 */
struct ThermalBC {
    ThermalBCType type;
    std::vector<Index> nodes;     ///< Nodes where BC is applied
    Real value;                   ///< Value (temperature, flux, or coefficient)
    Real T_ambient;               ///< Ambient temperature for convection/radiation
    Real emissivity;              ///< Emissivity for radiation

    ThermalBC(ThermalBCType t, std::vector<Index> n, Real v, Real t_amb = 293.15)
        : type(t), nodes(std::move(n)), value(v), T_ambient(t_amb), emissivity(0.8)
    {}
};

/**
 * @brief Thermal material properties for a region
 */
struct ThermalMaterial {
    Real density;               ///< Mass density (kg/m³)
    Real specific_heat;         ///< Specific heat capacity (J/kg·K)
    Real conductivity;          ///< Thermal conductivity (W/m·K)
    Real expansion_coeff;       ///< Thermal expansion coefficient (1/K)

    // For adiabatic heating
    Real taylor_quinney;        ///< Taylor-Quinney coefficient (fraction of plastic work → heat)

    KOKKOS_INLINE_FUNCTION
    ThermalMaterial()
        : density(7850.0)          // Steel
        , specific_heat(500.0)     // J/kg·K
        , conductivity(50.0)       // W/m·K
        , expansion_coeff(1.2e-5)  // 1/K
        , taylor_quinney(0.9)      // 90% of plastic work → heat
    {}

    /**
     * @brief Thermal diffusivity α = k/(ρ*c)
     */
    KOKKOS_INLINE_FUNCTION
    Real diffusivity() const {
        return conductivity / (density * specific_heat);
    }
};

// ============================================================================
// Thermal Solver Module
// ============================================================================

/**
 * @brief Thermal solver for heat conduction
 *
 * Uses explicit time integration (forward Euler) for the heat equation:
 *   T^{n+1} = T^n + Δt/(ρ*c) * [∇·(k∇T) + Q]
 *
 * Integration with FEM:
 * - Thermal expansion: ε_th = α*(T - T_ref)
 * - Adiabatic heating: Q = β * σ:ε̇_p (Taylor-Quinney)
 *
 * Usage:
 * ```cpp
 * ThermalSolver thermal("Heat");
 * thermal.initialize(mesh, state);
 * thermal.set_initial_temperature(293.15);
 *
 * // Add boundary conditions
 * thermal.add_bc(ThermalBC(ThermalBCType::Temperature, {0,1,2}, 373.15));
 * thermal.add_bc(ThermalBC(ThermalBCType::Convection, {10,11,12}, 25.0, 293.15));
 *
 * // Time stepping
 * Real dt = thermal.compute_stable_dt();
 * thermal.step(dt);
 *
 * // Get thermal expansion strain
 * auto expansion = thermal.thermal_expansion_strain(node_id, T_ref, material);
 * ```
 */
class ThermalSolver : public PhysicsModule {
public:
    ThermalSolver(const std::string& name = "Thermal");
    ~ThermalSolver() override = default;

    // ========================================================================
    // PhysicsModule Interface
    // ========================================================================

    void initialize(std::shared_ptr<Mesh> mesh,
                   std::shared_ptr<State> state) override;

    void finalize() override;

    Real compute_stable_dt() const override;

    void step(Real dt) override;

    std::vector<std::string> provided_fields() const override;
    std::vector<std::string> required_fields() const override;

    void export_field(const std::string& field_name,
                     std::vector<Real>& data) const override;

    void import_field(const std::string& field_name,
                     const std::vector<Real>& data) override;

    // ========================================================================
    // Thermal Configuration
    // ========================================================================

    /**
     * @brief Set uniform initial temperature
     */
    void set_initial_temperature(Real T) {
        initial_temperature_ = T;
        if (num_nodes_ > 0) {
            auto T_host = temperature_.view_host();
            for (size_t i = 0; i < num_nodes_; ++i) {
                T_host(i) = T;
            }
            temperature_.modify_host();
            temperature_.sync_device();
        }
    }

    /**
     * @brief Set reference temperature for thermal expansion
     */
    void set_reference_temperature(Real T_ref) { reference_temperature_ = T_ref; }
    Real reference_temperature() const { return reference_temperature_; }

    /**
     * @brief Set thermal material for all nodes
     */
    void set_material(const ThermalMaterial& mat) { default_material_ = mat; }

    /**
     * @brief Set thermal material for specific node set
     */
    void set_material(const std::vector<Index>& nodes, const ThermalMaterial& mat);

    /**
     * @brief Add thermal boundary condition
     */
    void add_bc(const ThermalBC& bc) { boundary_conditions_.push_back(bc); }

    /**
     * @brief Clear all boundary conditions
     */
    void clear_bcs() { boundary_conditions_.clear(); }

    /**
     * @brief Enable/disable adiabatic heating from plastic work
     */
    void enable_adiabatic_heating(bool enable) { adiabatic_heating_enabled_ = enable; }
    bool adiabatic_heating_enabled() const { return adiabatic_heating_enabled_; }

    /**
     * @brief Set CFL safety factor for thermal stability
     */
    void set_cfl_factor(Real cfl) { cfl_factor_ = cfl; }
    Real cfl_factor() const { return cfl_factor_; }

    // ========================================================================
    // Heat Source Interface
    // ========================================================================

    /**
     * @brief Add volumetric heat source at a node
     * @param node Node index
     * @param Q Heat generation rate (W/m³)
     */
    void add_heat_source(Index node, Real Q);

    /**
     * @brief Add adiabatic heating from plastic work
     * @param node Node index
     * @param plastic_work Plastic work rate (W/m³)
     */
    void add_plastic_heating(Index node, Real plastic_work);

    /**
     * @brief Clear all heat sources
     */
    void clear_heat_sources();

    // ========================================================================
    // Thermal Expansion Interface
    // ========================================================================

    /**
     * @brief Compute thermal strain at a node
     * @param node Node index
     * @return Thermal strain (isotropic: ε_th = α*(T - T_ref))
     */
    Real thermal_strain(Index node) const;

    /**
     * @brief Compute thermal expansion strain tensor (Voigt notation)
     * @param node Node index
     * @param strain Output strain tensor [6]
     */
    void thermal_strain_tensor(Index node, Real* strain) const;

    /**
     * @brief Compute thermal stress contribution (for given material)
     * @param node Node index
     * @param mat Material properties
     * @param stress Output stress tensor [6]
     */
    void thermal_stress(Index node, const MaterialProperties& mat, Real* stress) const;

    // ========================================================================
    // Accessors
    // ========================================================================

    Real temperature(Index node) const {
        return temperature_.view_host()(node);
    }

    void set_temperature(Index node, Real T) {
        temperature_.view_host()(node) = T;
        temperature_.modify_host();
    }

    auto temperature_view() const { return temperature_.view_host(); }
    auto temperature_device() const { return temperature_.view_device(); }

    auto heat_rate_view() const { return heat_rate_.view_host(); }

    size_t num_nodes() const { return num_nodes_; }

    /**
     * @brief Get total thermal energy in the system
     */
    Real total_thermal_energy() const;

    /**
     * @brief Print thermal statistics
     */
    void print_stats(std::ostream& os = std::cout) const;

private:
    // ========================================================================
    // Internal Methods
    // ========================================================================

    /**
     * @brief Compute heat conduction term ∇·(k∇T) using finite differences
     */
    void compute_heat_conduction();

    /**
     * @brief Apply thermal boundary conditions
     */
    void apply_boundary_conditions();

    /**
     * @brief Update temperature field
     */
    void update_temperature(Real dt);

    /**
     * @brief Compute element characteristic length for stability
     */
    Real compute_min_element_size() const;

    // ========================================================================
    // Member Variables
    // ========================================================================

    // Problem size
    size_t num_nodes_;

    // Temperature field (GPU-ready)
    Kokkos::DualView<Real*> temperature_;      ///< Current temperature
    Kokkos::DualView<Real*> temperature_old_;  ///< Previous temperature
    Kokkos::DualView<Real*> heat_rate_;        ///< Heat rate ∂T/∂t * ρ*c
    Kokkos::DualView<Real*> heat_source_;      ///< Volumetric heat source

    // Material properties
    ThermalMaterial default_material_;
    std::map<Index, ThermalMaterial> node_materials_;

    // Reference temperature for thermal expansion
    Real reference_temperature_;
    Real initial_temperature_;

    // Boundary conditions
    std::vector<ThermalBC> boundary_conditions_;

    // Solver parameters
    Real cfl_factor_;
    bool adiabatic_heating_enabled_;
    Real min_element_size_;
    bool initialized_ = false;

    // Statistics
    Real max_temperature_;
    Real min_temperature_;
    Real avg_temperature_;
};

// ============================================================================
// Thermo-Mechanical Coupling Helper
// ============================================================================

/**
 * @brief Helper class for thermo-mechanical coupling
 *
 * Computes thermal strain contribution to be added to mechanical strain.
 */
class ThermoMechanicalCoupling {
public:
    /**
     * @brief Compute thermal strain tensor
     * @param T Current temperature
     * @param T_ref Reference temperature
     * @param alpha Thermal expansion coefficient
     * @param strain Output strain tensor (Voigt: [εxx, εyy, εzz, γxy, γyz, γxz])
     */
    KOKKOS_INLINE_FUNCTION
    static void thermal_strain(Real T, Real T_ref, Real alpha, Real* strain) {
        Real eps_th = alpha * (T - T_ref);
        strain[0] = eps_th;  // εxx
        strain[1] = eps_th;  // εyy
        strain[2] = eps_th;  // εzz
        strain[3] = 0.0;     // γxy
        strain[4] = 0.0;     // γyz
        strain[5] = 0.0;     // γxz
    }

    /**
     * @brief Compute thermal stress (assuming constrained expansion)
     * @param T Current temperature
     * @param T_ref Reference temperature
     * @param E Young's modulus
     * @param nu Poisson's ratio
     * @param alpha Thermal expansion coefficient
     * @param stress Output stress tensor
     */
    KOKKOS_INLINE_FUNCTION
    static void thermal_stress(Real T, Real T_ref, Real E, Real nu, Real alpha, Real* stress) {
        Real eps_th = alpha * (T - T_ref);

        // For fully constrained: σ = -E*α*ΔT / (1-2ν) * I (hydrostatic)
        Real sigma_th = -E * eps_th / (1.0 - 2.0 * nu);

        stress[0] = sigma_th;
        stress[1] = sigma_th;
        stress[2] = sigma_th;
        stress[3] = 0.0;
        stress[4] = 0.0;
        stress[5] = 0.0;
    }

    /**
     * @brief Compute adiabatic temperature rise from plastic work
     * @param plastic_work_rate Plastic power per unit volume (W/m³)
     * @param density Material density (kg/m³)
     * @param specific_heat Specific heat capacity (J/kg·K)
     * @param taylor_quinney Taylor-Quinney coefficient (typically 0.9)
     * @param dt Time step
     * @return Temperature increment
     */
    KOKKOS_INLINE_FUNCTION
    static Real adiabatic_heating(Real plastic_work_rate, Real density,
                                  Real specific_heat, Real taylor_quinney, Real dt) {
        // ΔT = β * W_p / (ρ * c)
        return taylor_quinney * plastic_work_rate * dt / (density * specific_heat);
    }

    /**
     * @brief Compute plastic work rate from stress and plastic strain rate
     * @param stress Stress tensor (Voigt notation)
     * @param plastic_strain_rate Plastic strain rate tensor (Voigt notation)
     * @return Plastic power per unit volume (W/m³)
     */
    KOKKOS_INLINE_FUNCTION
    static Real plastic_power(const Real* stress, const Real* plastic_strain_rate) {
        // W_p = σ : ε̇_p
        return stress[0] * plastic_strain_rate[0] +
               stress[1] * plastic_strain_rate[1] +
               stress[2] * plastic_strain_rate[2] +
               stress[3] * plastic_strain_rate[3] +
               stress[4] * plastic_strain_rate[4] +
               stress[5] * plastic_strain_rate[5];
    }
};

} // namespace physics
} // namespace nxs
