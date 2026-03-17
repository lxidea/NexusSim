#pragma once

/**
 * @file explicit_dynamics.hpp
 * @brief Explicit dynamics enhancements: bulk viscosity, hourglass control,
 *        energy monitoring, and element erosion integration
 *
 * Wave 9: Complete explicit solver feature set for production crash/impact analysis.
 *
 * Components:
 * 1. BulkViscosity - Von Neumann-Richtmyer artificial viscosity for shock capturing
 * 2. HourglassControl - Manages hourglass stabilization for reduced-integration elements
 * 3. EnergyMonitor - Per-step energy balance tracking (KE, IE, external work, hourglass, contact)
 * 4. ElementErosionManager - Integrates element deletion into the explicit time loop
 * 5. ExplicitDynamicsConfig - Unified configuration for all explicit solver features
 *
 * References:
 * - Von Neumann & Richtmyer (1950) "A Method for the Numerical Calculation of Hydrodynamic Shocks"
 * - Flanagan & Belytschko (1981) "A Uniform Strain Hexahedron and Quadrilateral"
 * - LS-DYNA Theory Manual, Chapters 15-20
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/physics/element_erosion.hpp>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>

namespace nxs {
namespace fem {

// ============================================================================
// Bulk Viscosity (Artificial Viscosity for Shock Capturing)
// ============================================================================

/**
 * @brief Von Neumann-Richtmyer artificial viscosity for shock wave propagation
 *
 * Prevents oscillations behind shock fronts in explicit dynamics.
 * Two components:
 *   q = q_linear + q_quadratic
 *   q_linear = C_l * ρ * c * L * |ε̇_v|     (linear, damps ringing)
 *   q_quadratic = C_q * ρ * L² * ε̇_v²      (quadratic, captures shocks)
 *
 * where:
 *   ρ = current density
 *   c = sound speed
 *   L = element characteristic length
 *   ε̇_v = volumetric strain rate (negative = compression)
 *
 * Only active during compression (ε̇_v < 0).
 */
struct BulkViscosity {
    Real C_linear;      ///< Linear coefficient (default 0.06)
    Real C_quadratic;   ///< Quadratic coefficient (default 1.2)

    KOKKOS_INLINE_FUNCTION
    BulkViscosity() : C_linear(0.06), C_quadratic(1.2) {}

    BulkViscosity(Real c_l, Real c_q) : C_linear(c_l), C_quadratic(c_q) {}

    /**
     * @brief Compute bulk viscosity pressure for an element
     * @param vol_strain_rate Volumetric strain rate (compression < 0)
     * @param density Current element density
     * @param sound_speed Material sound speed
     * @param char_length Element characteristic length
     * @return Artificial viscosity pressure (always >= 0)
     */
    KOKKOS_INLINE_FUNCTION
    Real compute(Real vol_strain_rate, Real density,
                 Real sound_speed, Real char_length) const {
        // Only apply during compression
        if (vol_strain_rate >= 0.0) return 0.0;

        const Real abs_rate = -vol_strain_rate;  // Make positive

        // Linear term: dissipates oscillations
        const Real q_linear = C_linear * density * sound_speed * char_length * abs_rate;

        // Quadratic term: captures shock fronts
        const Real q_quad = C_quadratic * density * char_length * char_length * abs_rate * abs_rate;

        return q_linear + q_quad;
    }

    /**
     * @brief Compute bulk viscosity contribution to internal energy rate
     * @param q Bulk viscosity pressure
     * @param vol_strain_rate Volumetric strain rate
     * @param volume Element volume
     * @return Energy dissipation rate (always >= 0)
     */
    KOKKOS_INLINE_FUNCTION
    static Real energy_rate(Real q, Real vol_strain_rate, Real volume) {
        // dE/dt = -q * ε̇_v * V (positive because both q > 0 and ε̇_v < 0)
        return -q * vol_strain_rate * volume;
    }

    /**
     * @brief Add bulk viscosity to stress tensor (pressure component only)
     * @param q Bulk viscosity pressure
     * @param stress Voigt stress tensor [6] - modified in-place
     */
    KOKKOS_INLINE_FUNCTION
    static void add_to_stress(Real q, Real* stress) {
        // q acts as additional hydrostatic pressure (compression positive)
        stress[0] -= q;  // σxx -= q
        stress[1] -= q;  // σyy -= q
        stress[2] -= q;  // σzz -= q
    }
};

// ============================================================================
// Hourglass Control Types
// ============================================================================

enum class HourglassType {
    None,                    ///< No hourglass control
    FlanaganBelytschko,      ///< Viscous (velocity-based) Flanagan-Belytschko
    PerturbationStiffness,   ///< Stiffness-based perturbation
    Combined                 ///< Both viscous + stiffness
};

/**
 * @brief Hourglass control manager for reduced-integration elements
 *
 * Manages hourglass stabilization parameters and computes forces
 * to suppress zero-energy modes in 1-point integration Hex8 elements.
 */
struct HourglassControl {
    HourglassType type;
    Real viscous_coefficient;    ///< Viscous hourglass parameter (default 0.1)
    Real stiffness_coefficient;  ///< Stiffness hourglass parameter (default 0.05)
    Real total_energy;           ///< Accumulated hourglass energy

    HourglassControl()
        : type(HourglassType::FlanaganBelytschko)
        , viscous_coefficient(0.1)
        , stiffness_coefficient(0.05)
        , total_energy(0.0)
    {}

    HourglassControl(HourglassType t, Real visc = 0.1, Real stiff = 0.05)
        : type(t), viscous_coefficient(visc), stiffness_coefficient(stiff), total_energy(0.0)
    {}

    /**
     * @brief Compute effective hourglass stiffness for an element
     * @param bulk_modulus Material bulk modulus K
     * @param shear_modulus Material shear modulus G
     * @return Hourglass stiffness parameter
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_stiffness(Real bulk_modulus, Real shear_modulus) const {
        switch (type) {
            case HourglassType::FlanaganBelytschko:
                return viscous_coefficient * shear_modulus;
            case HourglassType::PerturbationStiffness:
                return stiffness_coefficient * (bulk_modulus + 4.0 / 3.0 * shear_modulus);
            case HourglassType::Combined:
                return viscous_coefficient * shear_modulus +
                       stiffness_coefficient * bulk_modulus;
            default:
                return 0.0;
        }
    }

    /**
     * @brief Compute hourglass force energy contribution
     * @param hg_force Hourglass force vector
     * @param velocity Nodal velocity vector
     * @param num_dof Number of DOFs
     * @param dt Time step
     * @return Hourglass energy increment
     */
    static Real compute_energy(const Real* hg_force, const Real* velocity,
                               std::size_t num_dof, Real dt) {
        Real energy = 0.0;
        for (std::size_t i = 0; i < num_dof; ++i) {
            energy += hg_force[i] * velocity[i] * dt;
        }
        return energy;
    }

    /**
     * @brief Accumulate hourglass energy
     */
    void accumulate_energy(Real de) { total_energy += de; }

    /**
     * @brief Reset hourglass energy counter
     */
    void reset_energy() { total_energy = 0.0; }

    /**
     * @brief Check if hourglass energy is excessive
     * @param internal_energy Total internal energy of the model
     * @param threshold Allowed ratio (default 10%)
     * @return true if hourglass energy exceeds threshold
     */
    bool is_excessive(Real internal_energy, Real threshold = 0.1) const {
        if (internal_energy < 1.0e-20) return false;
        return (total_energy / internal_energy) > threshold;
    }
};

// ============================================================================
// Energy Monitor
// ============================================================================

/**
 * @brief Comprehensive energy balance tracking for explicit dynamics
 *
 * Tracks all energy components per time step and monitors global balance:
 *   E_total = E_kinetic + E_internal + E_hourglass
 *   E_total + E_contact + E_damping + E_erosion = E_initial + W_external + W_bulk_visc
 *
 * Warning: Energy balance violation > 5% typically indicates instability.
 */
class EnergyMonitor {
public:
    struct EnergyComponents {
        Real kinetic;         ///< 0.5 * v^T * M * v
        Real internal;        ///< Integral of σ:ε over all elements
        Real hourglass;       ///< Energy absorbed by hourglass control
        Real bulk_viscosity;  ///< Energy dissipated by artificial viscosity
        Real contact;         ///< Energy dissipated by contact (friction + damping)
        Real damping;         ///< Energy dissipated by numerical damping
        Real external_work;   ///< Cumulative work done by external forces
        Real eroded;          ///< Energy of eroded (deleted) elements
        Real time;            ///< Simulation time

        EnergyComponents()
            : kinetic(0.0), internal(0.0), hourglass(0.0), bulk_viscosity(0.0)
            , contact(0.0), damping(0.0), external_work(0.0), eroded(0.0), time(0.0)
        {}

        /// Total mechanical energy in the system
        Real total() const { return kinetic + internal + hourglass; }

        /// Expected total from energy balance
        Real expected() const {
            return external_work - damping - contact - bulk_viscosity - eroded;
        }

        /// Energy balance error (ratio form)
        Real balance_error() const {
            Real e_max = std::max({std::abs(total()), std::abs(expected()),
                                   std::abs(external_work), 1.0e-20});
            return std::abs(total() - expected()) / e_max;
        }

        /// Hourglass-to-internal energy ratio
        Real hourglass_ratio() const {
            if (internal < 1.0e-20) return 0.0;
            return hourglass / internal;
        }
    };

    struct DiagnosticFlags {
        bool energy_explosion;        ///< Energy grew > 10x initial
        bool energy_balance_violated; ///< Balance error > tolerance
        bool hourglass_excessive;     ///< HG energy > 10% of IE
        bool negative_volume;         ///< Any element with J < 0
        int num_eroded_elements;      ///< Count of eroded elements this step

        DiagnosticFlags()
            : energy_explosion(false), energy_balance_violated(false)
            , hourglass_excessive(false), negative_volume(false)
            , num_eroded_elements(0)
        {}
    };

    EnergyMonitor() : tolerance_(0.05), initial_total_(0.0), initialized_(false) {}

    /**
     * @brief Set energy balance tolerance
     * @param tol Maximum allowed energy error ratio (default 0.05 = 5%)
     */
    void set_tolerance(Real tol) { tolerance_ = tol; }

    /**
     * @brief Initialize with initial system energy
     */
    void initialize(const EnergyComponents& initial) {
        initial_total_ = initial.total();
        history_.clear();
        history_.push_back(initial);
        initialized_ = true;
    }

    /**
     * @brief Record energy state at current time step
     * @param energy Current energy components
     * @return Diagnostic flags for this step
     */
    DiagnosticFlags record(const EnergyComponents& energy) {
        if (!initialized_) {
            initialize(energy);
        }

        DiagnosticFlags flags;
        history_.push_back(energy);

        // Check energy explosion (10x growth)
        if (energy.total() > 10.0 * (initial_total_ + energy.external_work + 1.0)) {
            flags.energy_explosion = true;
        }

        // Check energy balance
        if (energy.balance_error() > tolerance_) {
            flags.energy_balance_violated = true;
        }

        // Check hourglass ratio
        if (energy.hourglass_ratio() > 0.1) {
            flags.hourglass_excessive = true;
        }

        return flags;
    }

    /**
     * @brief Compute kinetic energy from velocity and mass arrays
     */
    static Real compute_kinetic_energy(const Real* velocity, const Real* mass,
                                        std::size_t ndof) {
        Real ke = 0.0;
        for (std::size_t i = 0; i < ndof; ++i) {
            ke += 0.5 * mass[i] * velocity[i] * velocity[i];
        }
        return ke;
    }

    /**
     * @brief Compute internal energy from element stress and strain
     * @param stress Voigt stress tensor [6]
     * @param strain Voigt strain tensor [6]
     * @param volume Element volume
     * @return Internal energy contribution from this element
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_element_internal_energy(const Real* stress, const Real* strain,
                                                  Real volume) {
        // W = integral(σ:ε dV) ≈ σ:ε * V for uniform stress/strain
        Real w = 0.0;
        for (int i = 0; i < 3; ++i) {
            w += stress[i] * strain[i];         // Normal components
        }
        for (int i = 3; i < 6; ++i) {
            w += stress[i] * strain[i];         // Shear components (engineering)
        }
        return 0.5 * w * volume;
    }

    /**
     * @brief Compute external work increment
     * @param f_ext External force vector
     * @param du Displacement increment vector
     * @param ndof Number of DOFs
     * @return Work increment
     */
    static Real compute_external_work_increment(const Real* f_ext, const Real* du,
                                                  std::size_t ndof) {
        Real work = 0.0;
        for (std::size_t i = 0; i < ndof; ++i) {
            work += f_ext[i] * du[i];
        }
        return work;
    }

    /// Get energy history
    const std::vector<EnergyComponents>& history() const { return history_; }

    /// Get latest energy state
    const EnergyComponents& latest() const { return history_.back(); }

    /// Get initial total energy
    Real initial_total() const { return initial_total_; }

    /// Get tolerance
    Real tolerance() const { return tolerance_; }

    /// Number of recorded steps
    std::size_t num_records() const { return history_.size(); }

    /// Check if initialized
    bool is_initialized() const { return initialized_; }

    /// Write energy history to CSV
    bool write_csv(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        file << "time,kinetic,internal,hourglass,bulk_viscosity,contact,"
             << "damping,external_work,eroded,total,balance_error\n";

        for (const auto& e : history_) {
            file << e.time << "," << e.kinetic << "," << e.internal << ","
                 << e.hourglass << "," << e.bulk_viscosity << "," << e.contact << ","
                 << e.damping << "," << e.external_work << "," << e.eroded << ","
                 << e.total() << "," << e.balance_error() << "\n";
        }
        return file.good();
    }

private:
    Real tolerance_;
    Real initial_total_;
    bool initialized_;
    std::vector<EnergyComponents> history_;
};

// ============================================================================
// Element Erosion Integration
// ============================================================================

// ElementErosionManager is defined in physics/element_erosion.hpp
// This section provides helper utilities for integrating erosion with
// the explicit solver loop.

/**
 * @brief Compute eroded energy for tracking in energy balance
 * @param erosion_mgr The existing physics::ElementErosionManager
 * @param stresses Per-element stress arrays
 * @param strains Per-element strain arrays
 * @param volumes Per-element volumes
 * @param num_elements Number of elements
 * @return Total energy of newly eroded elements
 */
inline Real compute_eroded_energy(const physics::ElementErosionManager& erosion_mgr,
                                   const Real* stresses, const Real* strains,
                                   const Real* volumes, std::size_t num_elements) {
    Real eroded_energy = 0.0;
    for (std::size_t e = 0; e < num_elements; ++e) {
        if (erosion_mgr.element_state(e) == physics::ElementState::Eroded) {
            Real sed = 0.0;
            for (int i = 0; i < 6; ++i) {
                sed += 0.5 * stresses[e * 6 + i] * strains[e * 6 + i];
            }
            eroded_energy += sed * volumes[e];
        }
    }
    return eroded_energy;
}

// ============================================================================
// Explicit Dynamics Configuration
// ============================================================================

/**
 * @brief Unified configuration for all explicit solver features
 */
struct ExplicitDynamicsConfig {
    // Bulk viscosity
    BulkViscosity bulk_viscosity;
    bool bulk_viscosity_enabled;

    // Hourglass control
    HourglassControl hourglass;
    bool hourglass_enabled;

    // Energy monitoring
    Real energy_tolerance;
    bool energy_monitoring_enabled;

    // Element erosion
    bool erosion_enabled;
    Real erosion_mass_limit;     ///< Max mass fraction that can erode (0.0-1.0)

    // Time step control
    Real cfl_factor;             ///< CFL safety factor
    Real dt_min;                 ///< Minimum time step (terminate if below)
    Real dt_max;                 ///< Maximum time step
    bool adaptive_dt;            ///< Enable adaptive time stepping

    // Damping
    Real rayleigh_alpha;         ///< Mass-proportional damping coefficient
    Real rayleigh_beta;          ///< Stiffness-proportional damping coefficient

    ExplicitDynamicsConfig()
        : bulk_viscosity_enabled(true)
        , hourglass_enabled(true)
        , energy_tolerance(0.05)
        , energy_monitoring_enabled(true)
        , erosion_enabled(false)
        , erosion_mass_limit(0.5)
        , cfl_factor(0.9)
        , dt_min(1.0e-12)
        , dt_max(1.0e-3)
        , adaptive_dt(false)
        , rayleigh_alpha(0.0)
        , rayleigh_beta(0.0)
    {}

    /// Preset for crash analysis (typical automotive)
    static ExplicitDynamicsConfig crash_preset() {
        ExplicitDynamicsConfig cfg;
        cfg.bulk_viscosity = BulkViscosity(0.06, 1.2);
        cfg.bulk_viscosity_enabled = true;
        cfg.hourglass = HourglassControl(HourglassType::FlanaganBelytschko, 0.1, 0.05);
        cfg.hourglass_enabled = true;
        cfg.energy_monitoring_enabled = true;
        cfg.erosion_enabled = true;
        cfg.erosion_mass_limit = 0.1;
        cfg.cfl_factor = 0.9;
        return cfg;
    }

    /// Preset for blast/shock analysis
    static ExplicitDynamicsConfig blast_preset() {
        ExplicitDynamicsConfig cfg;
        cfg.bulk_viscosity = BulkViscosity(0.1, 1.5);
        cfg.bulk_viscosity_enabled = true;
        cfg.hourglass = HourglassControl(HourglassType::PerturbationStiffness, 0.05, 0.1);
        cfg.hourglass_enabled = true;
        cfg.energy_monitoring_enabled = true;
        cfg.erosion_enabled = true;
        cfg.erosion_mass_limit = 0.3;
        cfg.cfl_factor = 0.7;
        return cfg;
    }

    /// Preset for drop test / low-velocity impact
    static ExplicitDynamicsConfig impact_preset() {
        ExplicitDynamicsConfig cfg;
        cfg.bulk_viscosity = BulkViscosity(0.06, 1.0);
        cfg.bulk_viscosity_enabled = true;
        cfg.hourglass = HourglassControl(HourglassType::FlanaganBelytschko, 0.05, 0.03);
        cfg.hourglass_enabled = true;
        cfg.energy_monitoring_enabled = true;
        cfg.erosion_enabled = false;
        cfg.cfl_factor = 0.9;
        return cfg;
    }
};

// ============================================================================
// Volumetric Strain Rate Computation
// ============================================================================

/**
 * @brief Compute volumetric strain rate from strain rate tensor
 * @param strain_rate Voigt strain rate tensor [6]
 * @return Volumetric strain rate (trace)
 */
KOKKOS_INLINE_FUNCTION
Real volumetric_strain_rate(const Real* strain_rate) {
    return strain_rate[0] + strain_rate[1] + strain_rate[2];
}

/**
 * @brief Compute volumetric strain rate from velocity gradient
 * @param dNdx Shape function derivatives w.r.t. global coords (num_nodes × 3)
 * @param velocity Nodal velocities (num_nodes × 3, flat)
 * @param num_nodes Number of element nodes
 * @return Volumetric strain rate
 */
KOKKOS_INLINE_FUNCTION
Real volumetric_strain_rate_from_velocity(const Real* dNdx, const Real* velocity,
                                                   int num_nodes) {
    Real ev_dot = 0.0;
    for (int n = 0; n < num_nodes; ++n) {
        // ε̇_v = ∑_n (∂N_n/∂x * vx_n + ∂N_n/∂y * vy_n + ∂N_n/∂z * vz_n)
        ev_dot += dNdx[n * 3 + 0] * velocity[n * 3 + 0]   // ∂N/∂x * vx
                + dNdx[n * 3 + 1] * velocity[n * 3 + 1]    // ∂N/∂y * vy
                + dNdx[n * 3 + 2] * velocity[n * 3 + 2];   // ∂N/∂z * vz
    }
    return ev_dot;
}

// ============================================================================
// Stable Time Step with Bulk Viscosity
// ============================================================================

/**
 * @brief Compute stable time step including bulk viscosity effect
 *
 * The bulk viscosity modifies the effective wave speed, making the stable
 * time step slightly smaller:
 *   dt_stable = L / c_eff
 *   c_eff = c + C_q * L * |ε̇_v|
 *
 * @param char_length Element characteristic length
 * @param sound_speed Material sound speed
 * @param bv Bulk viscosity parameters
 * @param vol_strain_rate Volumetric strain rate (for quadratic term)
 * @param cfl CFL safety factor
 * @return Stable time step
 */
KOKKOS_INLINE_FUNCTION
Real stable_dt_with_viscosity(Real char_length, Real sound_speed,
                                      const BulkViscosity& bv,
                                      Real vol_strain_rate, Real cfl) {
    Real c_eff = sound_speed;
    if (vol_strain_rate < 0.0) {
        c_eff += bv.C_quadratic * char_length * (-vol_strain_rate);
    }
    return cfl * char_length / (c_eff * 1.732050808);  // sqrt(3) for 3D
}

} // namespace fem
} // namespace nxs
