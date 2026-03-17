#pragma once

/**
 * @file eos_wave13.hpp
 * @brief Extended Equation of State models (Wave 13)
 *
 * Models:
 * - Murnaghan: Isothermal compression for solids (minerals, metals)
 * - Noble-Abel: Covolume-corrected ideal gas (dense gases, propellants)
 * - Stiff Gas (Tait): High-pressure fluids (water, shock-loaded liquids)
 * - Tillotson: Solid-vapor transition (hypervelocity impact, planetology)
 * - Sesame: 2D tabulated thermodynamic (national lab data)
 * - PowderBurn: Propellant combustion with burn-rate law
 * - Compaction: Porous/granular material with irreversible densification
 * - Osborne: Extended polynomial with energy coupling
 *
 * All functions are KOKKOS_INLINE_FUNCTION for GPU compatibility.
 *
 * Reference: Meyers, Dynamic Behavior of Materials (1994);
 *            Tillotson, GA-3216 (1962); SESAME Library, LANL;
 *            LS-DYNA Theory Manual, Ch. 30
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <nexussim/physics/material.hpp>
#include <cmath>

namespace nxs {
namespace physics {

// ============================================================================
// Extended EOS Types
// ============================================================================

enum class EOSWave13Type {
    Murnaghan,      ///< Isothermal finite-strain for solids
    NobleAbel,      ///< Covolume-corrected ideal gas
    StiffGas,       ///< Tait EOS for liquids (water)
    Tillotson,      ///< Solid/vapor transition EOS
    Sesame,         ///< 2D tabulated (rho, e) -> p
    PowderBurn,     ///< Propellant combustion
    Compaction,     ///< Porous material compaction
    Osborne         ///< Extended polynomial with energy
};

// ============================================================================
// 2D Table for Sesame EOS
// ============================================================================

/**
 * @brief GPU-compatible 2D tabulated data for Sesame EOS
 *
 * Stores pressure as p(rho, e) on a rectangular grid.
 * Uses bilinear interpolation with constant extrapolation.
 */
struct TabulatedSurface2D {
    static constexpr int MAX_ROWS = 32;    ///< Max density grid points
    static constexpr int MAX_COLS = 32;    ///< Max energy grid points

    Real rho_grid[MAX_ROWS];               ///< Density grid values
    Real e_grid[MAX_COLS];                 ///< Energy grid values
    Real data[MAX_ROWS * MAX_COLS];        ///< Flattened p(rho, e) values [row-major]
    int num_rho;                           ///< Number of density grid points
    int num_e;                             ///< Number of energy grid points

    KOKKOS_INLINE_FUNCTION
    TabulatedSurface2D() : num_rho(0), num_e(0) {
        for (int i = 0; i < MAX_ROWS; ++i) rho_grid[i] = 0.0;
        for (int i = 0; i < MAX_COLS; ++i) e_grid[i] = 0.0;
        for (int i = 0; i < MAX_ROWS * MAX_COLS; ++i) data[i] = 0.0;
    }

    /// Set a data point (host-side)
    void set(int i_rho, int i_e, Real value) {
        if (i_rho >= 0 && i_rho < MAX_ROWS && i_e >= 0 && i_e < MAX_COLS) {
            data[i_rho * MAX_COLS + i_e] = value;
        }
    }

    /// Get a data point
    KOKKOS_INLINE_FUNCTION
    Real get(int i_rho, int i_e) const {
        if (i_rho >= 0 && i_rho < num_rho && i_e >= 0 && i_e < num_e) {
            return data[i_rho * MAX_COLS + i_e];
        }
        return 0.0;
    }

    /**
     * @brief Bilinear interpolation on (rho, e) grid
     */
    KOKKOS_INLINE_FUNCTION
    Real evaluate(Real rho, Real e) const {
        if (num_rho < 2 || num_e < 2) return 0.0;

        // Clamp to grid bounds
        Real rho_c = rho;
        Real e_c = e;
        if (rho_c < rho_grid[0]) rho_c = rho_grid[0];
        if (rho_c > rho_grid[num_rho - 1]) rho_c = rho_grid[num_rho - 1];
        if (e_c < e_grid[0]) e_c = e_grid[0];
        if (e_c > e_grid[num_e - 1]) e_c = e_grid[num_e - 1];

        // Find density interval (binary search)
        int ir = 0;
        {
            int lo = 0, hi = num_rho - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (rho_grid[mid] <= rho_c) lo = mid;
                else hi = mid;
            }
            ir = lo;
        }

        // Find energy interval (binary search)
        int ie = 0;
        {
            int lo = 0, hi = num_e - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (e_grid[mid] <= e_c) lo = mid;
                else hi = mid;
            }
            ie = lo;
        }

        // Bilinear interpolation parameters
        Real dr = rho_grid[ir + 1] - rho_grid[ir];
        Real de = e_grid[ie + 1] - e_grid[ie];
        if (dr < 1.0e-30) dr = 1.0e-30;
        if (de < 1.0e-30) de = 1.0e-30;

        Real t_rho = (rho_c - rho_grid[ir]) / dr;
        Real t_e = (e_c - e_grid[ie]) / de;

        // Clamp interpolation parameters
        if (t_rho < 0.0) t_rho = 0.0;
        if (t_rho > 1.0) t_rho = 1.0;
        if (t_e < 0.0) t_e = 0.0;
        if (t_e > 1.0) t_e = 1.0;

        // Bilinear interpolation
        Real f00 = get(ir, ie);
        Real f10 = get(ir + 1, ie);
        Real f01 = get(ir, ie + 1);
        Real f11 = get(ir + 1, ie + 1);

        return f00 * (1.0 - t_rho) * (1.0 - t_e)
             + f10 * t_rho * (1.0 - t_e)
             + f01 * (1.0 - t_rho) * t_e
             + f11 * t_rho * t_e;
    }
};

// ============================================================================
// Extended EOS Properties
// ============================================================================

struct EOSWave13Properties {
    EOSWave13Type type;

    // Common
    Real rho0;             ///< Reference density [kg/m^3]

    // --- Murnaghan ---
    Real K0;               ///< Bulk modulus at zero pressure [Pa]
    Real K0_prime;         ///< Pressure derivative of bulk modulus (dK/dp)_0 [dimensionless]

    // --- Noble-Abel ---
    Real gamma_na;         ///< Ratio of specific heats
    Real b_covolume;       ///< Covolume [m^3/kg]

    // --- Stiff Gas (Tait) ---
    Real gamma_sg;         ///< Stiffened gamma
    Real p_inf;            ///< Stiffness pressure [Pa]

    // --- Tillotson ---
    Real till_a;           ///< Tillotson parameter a [dimensionless]
    Real till_b;           ///< Tillotson parameter b [dimensionless]
    Real till_A;           ///< Tillotson parameter A [Pa]
    Real till_B;           ///< Tillotson parameter B [Pa]
    Real till_e0;          ///< Tillotson reference energy e_0 [J/kg]
    Real till_alpha;       ///< Tillotson alpha (expansion exponent)
    Real till_beta;        ///< Tillotson beta (expansion exponent)
    Real till_e_iv;        ///< Energy of incipient vaporization [J/kg]
    Real till_e_cv;        ///< Energy of complete vaporization [J/kg]

    // --- Sesame ---
    TabulatedSurface2D sesame_table;  ///< 2D pressure table p(rho, e)

    // --- PowderBurn ---
    Real pb_force_const;   ///< Impetus (force constant f) [J/kg]
    Real pb_covolume;      ///< Covolume eta [m^3/kg]
    Real pb_burn_a;        ///< Burn rate coefficient a
    Real pb_burn_n;        ///< Burn rate pressure exponent n
    Real pb_gamma;         ///< Gamma for gas product

    // --- Compaction ---
    TabulatedCurve compaction_curve;  ///< Loading curve: p vs eps_v
    Real compact_K_unload; ///< Unloading bulk modulus [Pa]
    Real compact_ev_max;   ///< Fully compacted volumetric strain

    // --- Osborne ---
    Real osb_A1;           ///< Linear density coefficient
    Real osb_A2;           ///< Quadratic density coefficient
    Real osb_A3;           ///< Cubic density coefficient
    Real osb_B0;           ///< Energy coefficient (constant)
    Real osb_B1;           ///< Energy * mu coefficient
    Real osb_B2;           ///< Energy * mu^2 coefficient

    EOSWave13Properties()
        : type(EOSWave13Type::Murnaghan)
        , rho0(1000.0)
        // Murnaghan
        , K0(1.0e10), K0_prime(4.0)
        // Noble-Abel
        , gamma_na(1.4), b_covolume(0.0)
        // Stiff Gas
        , gamma_sg(7.15), p_inf(3.31e8)
        // Tillotson
        , till_a(0.5), till_b(1.5), till_A(7.5e10), till_B(6.5e10)
        , till_e0(9.5e6), till_alpha(5.0), till_beta(5.0)
        , till_e_iv(4.72e6), till_e_cv(1.82e7)
        // Sesame
        , sesame_table()
        // PowderBurn
        , pb_force_const(1.0e6), pb_covolume(1.0e-3), pb_burn_a(1.0e-4)
        , pb_burn_n(0.7), pb_gamma(1.25)
        // Compaction
        , compaction_curve(), compact_K_unload(1.0e10), compact_ev_max(0.5)
        // Osborne
        , osb_A1(0.0), osb_A2(0.0), osb_A3(0.0)
        , osb_B0(0.0), osb_B1(0.0), osb_B2(0.0)
    {}
};

// ============================================================================
// History state for stateful EOS models
// ============================================================================

struct EOSWave13State {
    Real eta_burned;       ///< PowderBurn: mass fraction burned [0..1]
    Real ev_max_reached;   ///< Compaction: maximum volumetric strain reached
    Real p_max_reached;    ///< Compaction: maximum pressure reached on loading

    KOKKOS_INLINE_FUNCTION
    EOSWave13State() : eta_burned(0.0), ev_max_reached(0.0), p_max_reached(0.0) {}
};

// ============================================================================
// Murnaghan EOS
// ============================================================================

/**
 * @brief Murnaghan isothermal EOS for solids
 *
 * p = (K0 / K0') * [(V0/V)^K0' - 1]
 *   = (K0 / K0') * [eta^K0' - 1]
 *
 * where eta = rho/rho0 = V0/V
 *
 * Commonly used for minerals and metals under moderate compression.
 * K0 = bulk modulus at zero pressure, K0' = dK/dP at zero pressure.
 */
class MurnaghanEOS {
public:
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave13Properties& props, Real rho, Real /*e*/) {
        if (props.rho0 < 1.0e-30) return 0.0;
        Real eta = rho / props.rho0;
        if (eta < 1.0e-30) return 0.0;

        // p = (K0 / K0') * (eta^K0' - 1)
        Real eta_pow = Kokkos::pow(eta, props.K0_prime);
        return (props.K0 / props.K0_prime) * (eta_pow - 1.0);
    }

    /**
     * @brief Sound speed for Murnaghan EOS
     * c^2 = K0/rho0 * (V0/V)^(K0'+1) = K0/rho0 * eta^(K0'+1)
     * More precisely: c^2 = (1/rho) * dp/d(1/rho) = dp/drho * ... but using
     * c^2 = K(p) / rho where K(p) = K0 * eta^K0'
     * => c^2 = K0 * eta^K0' / rho = K0 * eta^(K0'-1) / rho0
     */
    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave13Properties& props, Real rho, Real /*e*/) {
        if (props.rho0 < 1.0e-30 || rho < 1.0e-30) return 0.0;
        Real eta = rho / props.rho0;
        // dp/drho = K0/rho0 * eta^(K0'-1)
        // c^2 = dp/drho = K0 * eta^(K0'-1) / rho0
        Real c2 = (props.K0 / props.rho0) * Kokkos::pow(eta, props.K0_prime - 1.0);
        return (c2 > 0.0) ? Kokkos::sqrt(c2) : 0.0;
    }
};

// ============================================================================
// Noble-Abel EOS
// ============================================================================

/**
 * @brief Noble-Abel EOS (covolume-corrected ideal gas)
 *
 * p = (gamma - 1) * rho * e / (1 - b * rho)
 *
 * Reduces to ideal gas when b -> 0. The covolume b represents
 * the finite volume occupied by gas molecules. Used for dense
 * propellant gases and detonation products.
 */
class NobleAbelEOS {
public:
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave13Properties& props, Real rho, Real e) {
        Real denom = 1.0 - props.b_covolume * rho;
        if (denom < 1.0e-10) denom = 1.0e-10;  // Prevent singularity
        return (props.gamma_na - 1.0) * rho * e / denom;
    }

    /**
     * @brief Sound speed for Noble-Abel EOS
     * c^2 = gamma * p / (rho * (1 - b*rho))
     */
    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave13Properties& props, Real rho, Real e) {
        Real p = compute_pressure(props, rho, e);
        Real denom = 1.0 - props.b_covolume * rho;
        if (denom < 1.0e-10) denom = 1.0e-10;
        Real c2 = props.gamma_na * p / (rho * denom + 1.0e-30);
        return (c2 > 0.0) ? Kokkos::sqrt(c2) : 0.0;
    }
};

// ============================================================================
// Stiff Gas (Tait) EOS
// ============================================================================

/**
 * @brief Stiffened Gas (Tait) EOS for liquids
 *
 * p = (gamma - 1) * rho * e - gamma * p_inf
 *
 * For water: gamma = 7.15, p_inf = 3.31e8 Pa
 *
 * This is the standard form for shock calculations in water
 * and other near-incompressible fluids. The stiffness pressure
 * p_inf accounts for the molecular repulsion at high compression.
 */
class StiffGasEOS {
public:
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave13Properties& props, Real rho, Real e) {
        return (props.gamma_sg - 1.0) * rho * e - props.gamma_sg * props.p_inf;
    }

    /**
     * @brief Sound speed for stiffened gas
     * c^2 = gamma * (p + p_inf) / rho
     */
    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave13Properties& props, Real rho, Real e) {
        Real p = compute_pressure(props, rho, e);
        Real c2 = props.gamma_sg * (p + props.p_inf) / (rho + 1.0e-30);
        return (c2 > 0.0) ? Kokkos::sqrt(c2) : 0.0;
    }
};

// ============================================================================
// Tillotson EOS
// ============================================================================

/**
 * @brief Tillotson EOS for solid/vapor transitions
 *
 * Three regimes:
 * 1. Compressed (mu > 0):
 *    p = [a + b/(e/(e0*eta^2) + 1)] * rho0*e*eta + A*mu + B*mu^2
 *
 * 2. Expanded, low energy (mu < 0, e < e_iv):
 *    Same as compressed formula
 *
 * 3. Expanded, high energy (mu < 0, e > e_cv):
 *    p = a*rho0*e*eta + [b*rho0*e*eta / (e/(e0*eta^2)+1) + A*mu*exp(-beta*(rho0/rho-1))]
 *        * exp(-alpha*(rho0/rho - 1)^2)
 *
 * 4. Transition (e_iv < e < e_cv): linear interpolation between regimes 2 and 3.
 *
 * Used in hypervelocity impact simulations and planetary science.
 *
 * Parameters: a, b, A, B, e0, alpha, beta, e_iv, e_cv
 * eta = rho/rho0, mu = eta - 1
 */
class TillotsonEOS {
public:
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave13Properties& props, Real rho, Real e) {
        if (props.rho0 < 1.0e-30) return 0.0;
        Real eta = rho / props.rho0;
        if (eta < 1.0e-30) eta = 1.0e-30;
        Real mu = eta - 1.0;

        if (mu >= 0.0) {
            // Compressed state
            return compressed_pressure(props, eta, mu, e);
        } else {
            // Expanded state
            if (e < props.till_e_iv) {
                // Low energy: same as compressed formula
                return compressed_pressure(props, eta, mu, e);
            } else if (e > props.till_e_cv) {
                // High energy: vapor regime
                return expanded_pressure(props, rho, eta, mu, e);
            } else {
                // Transition region: interpolate
                Real p_c = compressed_pressure(props, eta, mu, e);
                Real p_e = expanded_pressure(props, rho, eta, mu, e);
                Real frac = (e - props.till_e_iv) / (props.till_e_cv - props.till_e_iv + 1.0e-30);
                return p_c * (1.0 - frac) + p_e * frac;
            }
        }
    }

    /**
     * @brief Sound speed via numerical differentiation
     */
    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave13Properties& props, Real rho, Real e) {
        if (rho < 1.0e-30) return 0.0;
        Real dp = 0.001 * rho;
        if (dp < 1.0e-10) dp = 1.0e-10;
        Real p1 = compute_pressure(props, rho + dp, e);
        Real p2 = compute_pressure(props, rho - dp, e);
        Real dpdrho = (p1 - p2) / (2.0 * dp);
        return (dpdrho > 0.0) ? Kokkos::sqrt(dpdrho) : 0.0;
    }

private:
    /**
     * @brief Compressed/cold Tillotson pressure
     * p = [a + b/(e/(e0*eta^2) + 1)] * rho0*e*eta + A*mu + B*mu^2
     */
    KOKKOS_INLINE_FUNCTION
    static Real compressed_pressure(const EOSWave13Properties& props,
                                     Real eta, Real mu, Real e) {
        Real e_norm = e / (props.till_e0 * eta * eta + 1.0e-30);
        Real coeff = props.till_a + props.till_b / (e_norm + 1.0);
        Real p_thermal = coeff * props.rho0 * e * eta;
        Real p_cold = props.till_A * mu + props.till_B * mu * mu;
        return p_thermal + p_cold;
    }

    /**
     * @brief Expanded/vapor Tillotson pressure
     * p = a*rho0*e*eta + [b*rho0*e*eta/(e/(e0*eta^2)+1) + A*mu*exp(-beta*(rho0/rho-1))]
     *     * exp(-alpha*(rho0/rho-1)^2)
     */
    KOKKOS_INLINE_FUNCTION
    static Real expanded_pressure(const EOSWave13Properties& props,
                                   Real rho, Real eta, Real mu, Real e) {
        Real rho_ratio = props.rho0 / (rho + 1.0e-30) - 1.0;
        if (rho_ratio < 0.0) rho_ratio = 0.0;

        Real exp_alpha = Kokkos::exp(-props.till_alpha * rho_ratio * rho_ratio);
        Real exp_beta = Kokkos::exp(-props.till_beta * rho_ratio);

        Real e_norm = e / (props.till_e0 * eta * eta + 1.0e-30);
        Real p_a = props.till_a * props.rho0 * e * eta;
        Real p_b = props.till_b * props.rho0 * e * eta / (e_norm + 1.0);
        Real p_cold = props.till_A * mu * exp_beta;

        return p_a + (p_b + p_cold) * exp_alpha;
    }
};

// ============================================================================
// Sesame (2D Tabulated) EOS
// ============================================================================

/**
 * @brief Sesame EOS using 2D tabulated data p(rho, e)
 *
 * The SESAME library (LANL) provides thermodynamic data on a
 * rectangular (rho, e) grid. This implementation uses bilinear
 * interpolation with clamped extrapolation at the grid boundaries.
 */
class SesameEOS {
public:
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave13Properties& props, Real rho, Real e) {
        return props.sesame_table.evaluate(rho, e);
    }

    /**
     * @brief Sound speed via numerical differentiation on 2D table
     * c^2 = dp/drho |_s ≈ dp/drho |_e + (p/rho^2) * dp/de |_rho
     * Simplified: use dp/drho at constant e
     */
    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave13Properties& props, Real rho, Real e) {
        if (rho < 1.0e-30) return 0.0;
        Real dp = 0.001 * rho;
        if (dp < 1.0e-10) dp = 1.0e-10;
        Real p1 = props.sesame_table.evaluate(rho + dp, e);
        Real p2 = props.sesame_table.evaluate(rho - dp, e);
        Real dpdrho = (p1 - p2) / (2.0 * dp);
        return (dpdrho > 0.0) ? Kokkos::sqrt(dpdrho) : 0.0;
    }
};

// ============================================================================
// PowderBurn (Propellant) EOS
// ============================================================================

/**
 * @brief PowderBurn EOS for propellant combustion
 *
 * Gas pressure: p = f * rho_gas * T / (1 - eta_cv * rho_gas)
 *   where f = force constant (impetus), eta_cv = covolume
 *
 * Simplified model treating combustion products as Noble-Abel gas:
 *   p = (gamma-1) * rho_gas * e / (1 - b * rho_gas)
 *   rho_gas = eta_burned * rho   (only burned fraction contributes)
 *
 * Burn rate law: d(eta)/dt = a * p^n
 *   eta = mass fraction burned [0..1]
 *
 * This is a history-dependent model requiring EOSWave13State.
 */
class PowderBurnEOS {
public:
    /**
     * @brief Compute gas pressure from current burn state
     *
     * Uses Noble-Abel form for the combustion products.
     * The unburned propellant contributes no gas pressure.
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave13Properties& props,
                                  Real rho, Real e,
                                  const EOSWave13State& state) {
        Real eta = state.eta_burned;
        if (eta < 1.0e-30) return 0.0;

        // Effective gas density = burned mass fraction * total density
        Real rho_gas = eta * rho;
        if (rho_gas < 1.0e-30) return 0.0;

        // Noble-Abel for combustion products
        Real denom = 1.0 - props.pb_covolume * rho_gas;
        if (denom < 1.0e-10) denom = 1.0e-10;

        return (props.pb_gamma - 1.0) * rho_gas * e / denom;
    }

    /**
     * @brief Compute pressure using eta_burned = 1 (fully burned) for simple usage
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave13Properties& props, Real rho, Real e) {
        EOSWave13State state;
        state.eta_burned = 1.0;
        return compute_pressure(props, rho, e, state);
    }

    /**
     * @brief Update burn fraction over a time step
     * d(eta)/dt = a * p^n, clamped to [0, 1]
     */
    KOKKOS_INLINE_FUNCTION
    static void update_burn(const EOSWave13Properties& props,
                             Real p, Real dt,
                             EOSWave13State& state) {
        if (state.eta_burned >= 1.0) return;
        Real p_abs = (p > 0.0) ? p : 0.0;
        Real deta = props.pb_burn_a * Kokkos::pow(p_abs, props.pb_burn_n) * dt;
        state.eta_burned += deta;
        if (state.eta_burned > 1.0) state.eta_burned = 1.0;
    }

    /**
     * @brief Sound speed for PowderBurn (fully burned Noble-Abel gas)
     */
    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave13Properties& props, Real rho, Real e) {
        Real p = compute_pressure(props, rho, e);
        Real denom = 1.0 - props.pb_covolume * rho;
        if (denom < 1.0e-10) denom = 1.0e-10;
        Real c2 = props.pb_gamma * p / (rho * denom + 1.0e-30);
        return (c2 > 0.0) ? Kokkos::sqrt(c2) : 0.0;
    }

    /**
     * @brief Sound speed with explicit state
     */
    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave13Properties& props, Real rho, Real e,
                             const EOSWave13State& state) {
        Real p = compute_pressure(props, rho, e, state);
        Real rho_gas = state.eta_burned * rho;
        Real denom = 1.0 - props.pb_covolume * rho_gas;
        if (denom < 1.0e-10) denom = 1.0e-10;
        Real c2 = props.pb_gamma * p / (rho_gas * denom + 1.0e-30);
        return (c2 > 0.0) ? Kokkos::sqrt(c2) : 0.0;
    }
};

// ============================================================================
// Compaction EOS
// ============================================================================

/**
 * @brief Compaction EOS for porous/granular materials
 *
 * Loading: p = f(eps_v) from user-defined compaction curve
 * Unloading: linear with slope K_unload from max pressure reached
 *
 * Irreversible densification: once compressed, the material does
 * not recover its original porosity. The unloading path is steeper
 * than the loading path (bulk modulus of the solid matrix).
 *
 * eps_v = rho/rho0 - 1 (volumetric strain, positive in compression)
 */
class CompactionEOS {
public:
    /**
     * @brief Compute pressure with history tracking
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave13Properties& props,
                                  Real rho, Real /*e*/,
                                  EOSWave13State& state) {
        if (props.rho0 < 1.0e-30) return 0.0;
        Real ev = rho / props.rho0 - 1.0;  // Volumetric strain

        if (ev <= 0.0) {
            // Tension or reference: no pressure from compaction
            return 0.0;
        }

        // Evaluate loading curve pressure
        Real p_load = props.compaction_curve.evaluate(ev);

        // Check if we are on the loading path or unloading
        if (ev >= state.ev_max_reached) {
            // Loading: follow the compaction curve
            state.ev_max_reached = ev;
            state.p_max_reached = p_load;
            return p_load;
        } else {
            // Unloading: linear from (ev_max, p_max) with slope K_unload
            Real p_unload = state.p_max_reached
                          - props.compact_K_unload * (state.ev_max_reached - ev);
            // Pressure cannot go negative on unloading path
            return (p_unload > 0.0) ? p_unload : 0.0;
        }
    }

    /**
     * @brief Compute pressure without history (loading path only)
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave13Properties& props, Real rho, Real e) {
        EOSWave13State state;
        state.ev_max_reached = 0.0;  // Always on loading path
        return compute_pressure(props, rho, e, state);
    }

    /**
     * @brief Sound speed on loading path
     * c^2 = dp/drho = (1/rho0) * dp/d(eps_v)
     * Numerically differentiate the compaction curve
     */
    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave13Properties& props, Real rho, Real /*e*/) {
        if (props.rho0 < 1.0e-30 || rho < 1.0e-30) return 0.0;
        Real ev = rho / props.rho0 - 1.0;
        Real dev = 0.001;
        Real p1 = props.compaction_curve.evaluate(ev + dev);
        Real p2 = props.compaction_curve.evaluate(ev - dev);
        Real dpdev = (p1 - p2) / (2.0 * dev);
        // dp/drho = (1/rho0) * dp/dev
        Real dpdrho = dpdev / props.rho0;
        return (dpdrho > 0.0) ? Kokkos::sqrt(dpdrho) : 0.0;
    }
};

// ============================================================================
// Osborne EOS
// ============================================================================

/**
 * @brief Osborne EOS - extended polynomial with energy coupling
 *
 * p = A1*mu + A2*mu^2 + A3*mu^3 + (B0 + B1*mu + B2*mu^2) * E
 *
 * where mu = rho/rho0 - 1
 *       E = rho * e (volumetric internal energy)
 *
 * This is a generalization of the linear polynomial EOS with
 * explicit quadratic/cubic density terms plus energy coupling.
 * 6 coefficients: A1, A2, A3, B0, B1, B2.
 */
class OsborneEOS {
public:
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave13Properties& props, Real rho, Real e) {
        if (props.rho0 < 1.0e-30) return 0.0;
        Real mu = rho / props.rho0 - 1.0;
        Real mu2 = mu * mu;
        Real mu3 = mu2 * mu;
        Real E_vol = rho * e;

        Real p_cold = props.osb_A1 * mu + props.osb_A2 * mu2 + props.osb_A3 * mu3;
        Real p_thermal = (props.osb_B0 + props.osb_B1 * mu + props.osb_B2 * mu2) * E_vol;
        return p_cold + p_thermal;
    }

    /**
     * @brief Sound speed for Osborne EOS
     * dp/drho at constant e:
     *   dp/drho = (1/rho0) * [A1 + 2*A2*mu + 3*A3*mu^2]
     *           + e * [B0 + B1*mu + B2*mu^2]
     *           + (rho*e/rho0) * [B1 + 2*B2*mu]
     */
    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave13Properties& props, Real rho, Real e) {
        if (props.rho0 < 1.0e-30 || rho < 1.0e-30) return 0.0;
        Real mu = rho / props.rho0 - 1.0;
        Real inv_rho0 = 1.0 / props.rho0;

        // d(p_cold)/drho = inv_rho0 * (A1 + 2*A2*mu + 3*A3*mu^2)
        Real dp_cold = inv_rho0 * (props.osb_A1 + 2.0 * props.osb_A2 * mu
                                   + 3.0 * props.osb_A3 * mu * mu);

        // d(p_thermal)/drho = d/drho [(B0+B1*mu+B2*mu^2) * rho * e]
        //   = e * (B0 + B1*mu + B2*mu^2)  +  rho*e/rho0 * (B1 + 2*B2*mu)
        Real B_val = props.osb_B0 + props.osb_B1 * mu + props.osb_B2 * mu * mu;
        Real dB_dmu = props.osb_B1 + 2.0 * props.osb_B2 * mu;
        Real dp_thermal = e * B_val + rho * e * inv_rho0 * dB_dmu;

        Real dpdrho = dp_cold + dp_thermal;
        return (dpdrho > 0.0) ? Kokkos::sqrt(dpdrho) : 0.0;
    }
};

// ============================================================================
// Unified Dispatch (optional convenience)
// ============================================================================

/**
 * @brief Unified dispatch for all Wave 13 EOS models
 *
 * For models without history (Murnaghan, Noble-Abel, Stiff Gas, Tillotson,
 * Sesame, Osborne), uses the stateless overloads. PowderBurn and Compaction
 * use fully-burned / loading-only defaults respectively.
 */
class EOSWave13 {
public:
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave13Properties& props, Real rho, Real e) {
        switch (props.type) {
            case EOSWave13Type::Murnaghan:
                return MurnaghanEOS::compute_pressure(props, rho, e);
            case EOSWave13Type::NobleAbel:
                return NobleAbelEOS::compute_pressure(props, rho, e);
            case EOSWave13Type::StiffGas:
                return StiffGasEOS::compute_pressure(props, rho, e);
            case EOSWave13Type::Tillotson:
                return TillotsonEOS::compute_pressure(props, rho, e);
            case EOSWave13Type::Sesame:
                return SesameEOS::compute_pressure(props, rho, e);
            case EOSWave13Type::PowderBurn:
                return PowderBurnEOS::compute_pressure(props, rho, e);
            case EOSWave13Type::Compaction:
                return CompactionEOS::compute_pressure(props, rho, e);
            case EOSWave13Type::Osborne:
                return OsborneEOS::compute_pressure(props, rho, e);
            default:
                return 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave13Properties& props, Real rho, Real e) {
        switch (props.type) {
            case EOSWave13Type::Murnaghan:
                return MurnaghanEOS::sound_speed(props, rho, e);
            case EOSWave13Type::NobleAbel:
                return NobleAbelEOS::sound_speed(props, rho, e);
            case EOSWave13Type::StiffGas:
                return StiffGasEOS::sound_speed(props, rho, e);
            case EOSWave13Type::Tillotson:
                return TillotsonEOS::sound_speed(props, rho, e);
            case EOSWave13Type::Sesame:
                return SesameEOS::sound_speed(props, rho, e);
            case EOSWave13Type::PowderBurn:
                return PowderBurnEOS::sound_speed(props, rho, e);
            case EOSWave13Type::Compaction:
                return CompactionEOS::sound_speed(props, rho, e);
            case EOSWave13Type::Osborne:
                return OsborneEOS::sound_speed(props, rho, e);
            default:
                return 0.0;
        }
    }
};

} // namespace physics
} // namespace nxs
