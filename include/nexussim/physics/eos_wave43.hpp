#pragma once

/**
 * @file eos_wave43.hpp
 * @brief Extended Equation of State models (Wave 43)
 *
 * Models:
 * - LSZK (Lee-Szekely-Kung): Reactive burn with Arrhenius kinetics
 * - NASG (Noble-Abel-Stiffened-Gas): Multi-phase nuclear safety EOS
 * - Puff: Porous material with solid/vapor/mixed-phase regimes
 * - Exponential: JWL-like A*exp(-R1*V) + B*exp(-R2*V) + C/(omega*V)
 * - IdealGasVT: Volume-temperature coupled ideal gas P = rho*R_gas*T
 * - Compaction2: 2nd-gen compaction with separate unloading slope
 * - CompactionTab: Fully tabulated compaction loading/unloading curves
 *
 * All functions are KOKKOS_INLINE_FUNCTION for GPU compatibility.
 *
 * Reference: Lee, Szekely & Kung, Propellants & Explosives (1973);
 *            Le Metayer & Saurel, Phys. Fluids (2016) [NASG];
 *            Kerley, Int. J. Impact Engng 5 (1987) [Puff];
 *            JWL EOS family: Lee, Finger & Collins, LLNL (1973);
 *            OpenRadioss Theory Manual, EOS chapter.
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <nexussim/physics/material.hpp>
#include <cmath>

namespace nxs {
namespace physics {

// ============================================================================
// Wave 43 EOS Type Enum
// ============================================================================

enum class EOSWave43Type {
    LSZK,           ///< Lee-Szekely-Kung reactive burn
    NASG,           ///< Noble-Abel-Stiffened-Gas
    Puff,           ///< Porous material (solid/vapor/mixed)
    Exponential,    ///< JWL-like exponential EOS
    IdealGasVT,     ///< Ideal gas with explicit temperature
    Compaction2,    ///< 2nd-gen compaction with K_unload
    CompactionTab   ///< Tabulated compaction curves
};

// ============================================================================
// Tabulated curve (1D) - reuse pattern from eos_wave13 TabulatedSurface2D
// ============================================================================

/**
 * @brief GPU-compatible 1D tabulated curve (x -> y)
 *
 * Used for CompactionTab loading/unloading curves.
 * Linear interpolation with clamped extrapolation.
 */
struct TabulatedCurve1D {
    static constexpr int MAX_PTS = 64;

    Real x[MAX_PTS];   ///< Independent variable (e.g. mu = rho/rho0-1)
    Real y[MAX_PTS];   ///< Dependent variable (e.g. pressure)
    int n;             ///< Number of points

    KOKKOS_INLINE_FUNCTION
    TabulatedCurve1D() : n(0) {
        for (int i = 0; i < MAX_PTS; ++i) { x[i] = 0.0; y[i] = 0.0; }
    }

    void set(int i, Real xi, Real yi) {
        if (i >= 0 && i < MAX_PTS) { x[i] = xi; y[i] = yi; }
    }

    KOKKOS_INLINE_FUNCTION
    Real evaluate(Real xval) const {
        if (n < 2) return (n == 1) ? y[0] : 0.0;
        // Clamp
        if (xval <= x[0]) return y[0];
        if (xval >= x[n - 1]) return y[n - 1];
        // Binary search
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (x[mid] <= xval) lo = mid; else hi = mid;
        }
        Real dx = x[hi] - x[lo];
        if (dx < 1.0e-30) return y[lo];
        Real t = (xval - x[lo]) / dx;
        return y[lo] + t * (y[hi] - y[lo]);
    }
};

// ============================================================================
// Wave 43 EOS Properties
// ============================================================================

struct EOSWave43Properties {
    EOSWave43Type type;

    // Common
    Real rho0;           ///< Reference density [kg/m^3]

    // --- LSZK ---
    Real lszk_gamma_u;   ///< Gamma for unreacted material
    Real lszk_K_u;       ///< Bulk modulus of unreacted [Pa]
    Real lszk_gamma_p;   ///< Gamma for product gas
    Real lszk_e0_p;      ///< Reference energy for product gas [J/kg]
    Real lszk_A_arr;     ///< Arrhenius pre-exponential factor [1/s]
    Real lszk_Ea_R;      ///< Activation energy / gas constant = E_a/R [K]

    // --- NASG ---
    Real nasg_gamma;     ///< Ratio of specific heats
    Real nasg_b;         ///< Covolume [m^3/kg]
    Real nasg_q;         ///< Reference energy offset [J/kg]
    Real nasg_p_inf;     ///< Stiffness pressure [Pa]

    // --- Puff ---
    Real puff_K_s;       ///< Solid-phase bulk modulus [Pa]
    Real puff_E_sub;     ///< Sublimation energy [J/kg]
    Real puff_alpha0;    ///< Initial porosity (rho0/rho_solid)
    Real puff_gamma_v;   ///< Gamma for vapor phase
    Real puff_rho_solid; ///< Fully dense solid density [kg/m^3]

    // --- Exponential ---
    Real exp_A;          ///< A coefficient [Pa]
    Real exp_B;          ///< B coefficient [Pa]
    Real exp_C;          ///< C coefficient [Pa*m^3/kg... dimensionless via omega*V]
    Real exp_R1;         ///< R1 exponent [dimensionless]
    Real exp_R2;         ///< R2 exponent [dimensionless]
    Real exp_omega;      ///< Gruneisen omega [dimensionless]
    Real exp_V0;         ///< Reference specific volume V0 = 1/rho0 [m^3/kg]

    // --- IdealGasVT ---
    Real igt_R_gas;      ///< Specific gas constant [J/(kg*K)]
    Real igt_Cv;         ///< Specific heat at constant volume [J/(kg*K)]
    Real igt_T0;         ///< Reference temperature [K]

    // --- Compaction2 ---
    TabulatedCurve compact2_curve;   ///< Loading curve: p vs mu (mu = rho/rho0-1)
    Real compact2_K_unload;          ///< Unloading bulk modulus [Pa]

    // --- CompactionTab ---
    TabulatedCurve1D comptab_load;   ///< Tabulated loading: p(mu)
    TabulatedCurve1D comptab_unload; ///< Tabulated unloading at mu_max reference
    Real comptab_mu_max_ref;         ///< mu_max used to build the unloading table

    KOKKOS_INLINE_FUNCTION
    EOSWave43Properties()
        : type(EOSWave43Type::LSZK)
        , rho0(1600.0)
        // LSZK
        , lszk_gamma_u(3.0), lszk_K_u(1.0e10)
        , lszk_gamma_p(2.7), lszk_e0_p(3.68e6)
        , lszk_A_arr(1.0e9), lszk_Ea_R(8000.0)
        // NASG
        , nasg_gamma(1.4), nasg_b(0.0), nasg_q(0.0), nasg_p_inf(0.0)
        // Puff
        , puff_K_s(1.0e10), puff_E_sub(6.0e6)
        , puff_alpha0(1.0), puff_gamma_v(1.667), puff_rho_solid(2700.0)
        // Exponential
        , exp_A(3.712e11), exp_B(3.231e9)
        , exp_C(7.678e8), exp_R1(4.15), exp_R2(0.95)
        , exp_omega(0.35), exp_V0(0.0)
        // IdealGasVT
        , igt_R_gas(287.0), igt_Cv(718.0), igt_T0(293.15)
        // Compaction2
        , compact2_curve(), compact2_K_unload(1.0e10)
        // CompactionTab
        , comptab_load(), comptab_unload(), comptab_mu_max_ref(0.5)
    {}
};

// ============================================================================
// History state for stateful Wave 43 EOS models
// ============================================================================

struct EOSWave43State {
    Real lambda;         ///< LSZK: burn fraction [0..1]
    Real mu_max;         ///< Compaction2/CompactionTab: max volumetric strain reached
    Real p_max;          ///< Compaction2/CompactionTab: pressure at mu_max
    Real temperature;    ///< IdealGasVT: current temperature [K]

    KOKKOS_INLINE_FUNCTION
    EOSWave43State()
        : lambda(0.0), mu_max(0.0), p_max(0.0), temperature(293.15) {}
};

// ============================================================================
// LSZK (Lee-Szekely-Kung) Reactive Burn EOS
// ============================================================================

/**
 * @brief LSZK EOS for reactive burn with Arrhenius kinetics
 *
 * Mixture pressure (linear mixing rule):
 *   P = P_u * (1 - lambda) + P_p * lambda
 *
 * Unreacted EOS (Tait-like):
 *   P_u = K_u * (eta - 1)   where eta = rho/rho0
 *
 * Product gas EOS (ideal-gas with offset):
 *   P_p = (gamma_p - 1) * rho * (e - e0_p)
 *
 * Burn rate (Arrhenius):
 *   d(lambda)/dt = A * (1 - lambda) * exp(-Ea/R/T)
 *
 * Temperature estimate: T ~ e / Cv_eff (simple single-T approximation)
 *
 * State: lambda in EOSWave43State.lambda
 */
class LSZKEOS {
public:
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave43Properties& props,
                                  Real rho, Real e,
                                  const EOSWave43State& state) {
        if (props.rho0 < 1.0e-30 || rho < 1.0e-30) return 0.0;
        Real eta = rho / props.rho0;
        Real lambda = state.lambda;
        if (lambda < 0.0) lambda = 0.0;
        if (lambda > 1.0) lambda = 1.0;

        // Unreacted component: linear Tait
        Real P_u = props.lszk_K_u * (eta - 1.0);

        // Products: ideal-gas-like (guard against negative energy offset)
        Real e_eff = e - props.lszk_e0_p;
        Real P_p = (props.lszk_gamma_p - 1.0) * rho * e_eff;
        if (P_p < 0.0) P_p = 0.0;  // Can't have negative product pressure

        return P_u * (1.0 - lambda) + P_p * lambda;
    }

    /**
     * @brief Stateless overload: fully burned (lambda = 1)
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave43Properties& props, Real rho, Real e) {
        EOSWave43State state;
        state.lambda = 1.0;
        return compute_pressure(props, rho, e, state);
    }

    /**
     * @brief Update burn fraction via Arrhenius kinetics over dt
     * d(lambda)/dt = A * (1 - lambda) * exp(-Ea_R / T)
     * T approximated as e / Cv_eff (or passed via state.temperature)
     */
    KOKKOS_INLINE_FUNCTION
    static void update_burn(const EOSWave43Properties& props,
                             Real e, Real dt,
                             EOSWave43State& state) {
        if (state.lambda >= 1.0) return;
        // Estimate temperature from specific internal energy
        // Use a nominal Cv = 1000 J/(kg*K) if not provided
        const Real Cv_nominal = 1000.0;
        Real T = (e > 0.0) ? (e / Cv_nominal) : 300.0;
        if (T < 1.0) T = 1.0;  // Prevent division by zero

        Real rate = props.lszk_A_arr * (1.0 - state.lambda)
                    * Kokkos::exp(-props.lszk_Ea_R / T);
        state.lambda += rate * dt;
        if (state.lambda > 1.0) state.lambda = 1.0;
        if (state.lambda < 0.0) state.lambda = 0.0;
    }

    /**
     * @brief Sound speed via numerical differentiation
     */
    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave43Properties& props, Real rho, Real e,
                             const EOSWave43State& state) {
        if (rho < 1.0e-30) return 0.0;
        Real drho = 1.0e-3 * rho;
        if (drho < 1.0e-10) drho = 1.0e-10;
        Real p1 = compute_pressure(props, rho + drho, e, state);
        Real p2 = compute_pressure(props, rho - drho, e, state);
        Real dpdrho = (p1 - p2) / (2.0 * drho);
        return (dpdrho > 0.0) ? Kokkos::sqrt(dpdrho) : 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave43Properties& props, Real rho, Real e) {
        EOSWave43State state;
        state.lambda = 1.0;
        return sound_speed(props, rho, e, state);
    }
};

// ============================================================================
// NASG (Noble-Abel-Stiffened-Gas) EOS
// ============================================================================

/**
 * @brief Noble-Abel-Stiffened-Gas EOS for multi-phase nuclear safety
 *
 * P = (gamma - 1) * rho * (e - q) / (1 - b * rho) - gamma * p_inf
 *
 * Reduces to:
 *   - Ideal gas:     b=0, p_inf=0, q=0
 *   - Noble-Abel:    p_inf=0, q=0
 *   - Stiffened gas: b=0, q=0
 *   - Full NASG:     all non-zero
 *
 * Used in nuclear reactor safety analysis (two-phase water/steam).
 *
 * Reference: Le Metayer & Saurel, Phys. Fluids 28, 046102 (2016)
 */
class NASGEOS {
public:
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave43Properties& props, Real rho, Real e) {
        Real denom = 1.0 - props.nasg_b * rho;
        if (denom < 1.0e-10) denom = 1.0e-10;
        Real e_shift = e - props.nasg_q;
        return (props.nasg_gamma - 1.0) * rho * e_shift / denom
               - props.nasg_gamma * props.nasg_p_inf;
    }

    /**
     * @brief Temperature from NASG: T = [P + gamma*p_inf] * (1 - b*rho) / [(gamma-1)*rho*Cv]
     * Cv not stored in props here; use e relation: T = (e - q - p_inf/rho) / Cv
     * Simplified: return e - q offset as proxy when Cv not available
     */
    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave43Properties& props, Real rho, Real e) {
        Real p = compute_pressure(props, rho, e);
        Real denom = 1.0 - props.nasg_b * rho;
        if (denom < 1.0e-10) denom = 1.0e-10;
        // c^2 = gamma*(p + p_inf) / (rho * (1-b*rho))
        Real c2 = props.nasg_gamma * (p + props.nasg_p_inf) / (rho * denom + 1.0e-30);
        return (c2 > 0.0) ? Kokkos::sqrt(c2) : 0.0;
    }
};

// ============================================================================
// Puff EOS (Porous material with phase transitions)
// ============================================================================

/**
 * @brief Puff EOS for porous material with three regimes
 *
 * Three regimes based on volumetric strain and internal energy:
 *
 * 1. Solid compression (rho > rho_solid / alpha0, e < E_sub):
 *    P = K_s * (eta_s - 1)   where eta_s = rho / rho_solid
 *
 * 2. Vapor phase (e > E_sub, rho < rho0):
 *    P = (gamma_v - 1) * rho * (e - E_sub)
 *
 * 3. Mixed phase (transition between solid and vapor):
 *    Linear interpolation in (rho, e) space
 *
 * alpha0 = initial porosity = rho0 / rho_solid (< 1 for porous material)
 *
 * Reference: Kerley, Int. J. Impact Engng 5 (1987); Sandia Report
 */
class PuffEOS {
public:
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave43Properties& props, Real rho, Real e) {
        if (props.puff_rho_solid < 1.0e-30 || rho < 1.0e-30) return 0.0;
        Real rho0 = props.rho0;
        Real rho_s = props.puff_rho_solid;
        Real E_sub = props.puff_E_sub;

        // Solid-phase pressure (compress toward rho_solid)
        Real eta_s = rho / rho_s;
        Real P_solid = props.puff_K_s * (eta_s - 1.0);

        // Vapor-phase pressure
        Real e_vap = e - E_sub;
        Real P_vapor = 0.0;
        if (e_vap > 0.0) {
            P_vapor = (props.puff_gamma_v - 1.0) * rho * e_vap;
        }

        // Determine regime
        bool in_vapor = (e >= E_sub) && (rho < rho0);
        bool in_solid = (rho >= rho_s) || (e < E_sub && rho > rho0);

        if (in_solid) {
            // Pure solid compression
            return (P_solid > 0.0) ? P_solid : 0.0;
        } else if (in_vapor) {
            // Pure vapor
            return (P_vapor > 0.0) ? P_vapor : 0.0;
        } else {
            // Mixed phase: interpolate based on energy relative to E_sub
            Real frac = (E_sub > 1.0e-30) ? (e / E_sub) : 1.0;
            if (frac < 0.0) frac = 0.0;
            if (frac > 1.0) frac = 1.0;
            Real P_mix = P_solid * (1.0 - frac) + P_vapor * frac;
            return (P_mix > 0.0) ? P_mix : 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave43Properties& props, Real rho, Real e) {
        if (rho < 1.0e-30) return 0.0;
        Real drho = 1.0e-3 * rho;
        if (drho < 1.0e-10) drho = 1.0e-10;
        Real p1 = compute_pressure(props, rho + drho, e);
        Real p2 = compute_pressure(props, rho - drho, e);
        Real dpdrho = (p1 - p2) / (2.0 * drho);
        return (dpdrho > 0.0) ? Kokkos::sqrt(dpdrho) : 0.0;
    }
};

// ============================================================================
// Exponential EOS (JWL-like)
// ============================================================================

/**
 * @brief Exponential EOS (JWL-like simplified form)
 *
 * P = A*exp(-R1*V/V0) + B*exp(-R2*V/V0) + C/(omega*V/V0)
 *
 * where V = 1/rho (specific volume), V0 = 1/rho0
 *
 * This reduces the JWL to a form without energy coupling.
 * Full JWL adds an energy term: + omega * rho * e
 *
 * Useful for simple explosive detonation products and propellant gases.
 *
 * Reference: Lee, Finger & Collins, LLNL (1973); LS-DYNA EOS 2
 */
class ExponentialEOS {
public:
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave43Properties& props, Real rho, Real e) {
        if (rho < 1.0e-30) return 0.0;
        Real V0 = props.exp_V0;
        if (V0 < 1.0e-30) V0 = 1.0 / (props.rho0 + 1.0e-30);
        Real V = 1.0 / rho;
        Real r = V / V0;   // Relative specific volume = rho0/rho

        Real P1 = props.exp_A * Kokkos::exp(-props.exp_R1 * r);
        Real P2 = props.exp_B * Kokkos::exp(-props.exp_R2 * r);
        Real denom = props.exp_omega * r;
        if (denom < 1.0e-30) denom = 1.0e-30;
        Real P3 = props.exp_C / denom;

        // Full JWL adds energy term: + omega * rho * e
        Real P_energy = props.exp_omega * rho * e;

        return P1 + P2 + P3 + P_energy;
    }

    /**
     * @brief Stateless overload without energy (e=0)
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure_no_energy(const EOSWave43Properties& props, Real rho) {
        if (rho < 1.0e-30) return 0.0;
        Real V0 = props.exp_V0;
        if (V0 < 1.0e-30) V0 = 1.0 / (props.rho0 + 1.0e-30);
        Real V = 1.0 / rho;
        Real r = V / V0;
        Real P1 = props.exp_A * Kokkos::exp(-props.exp_R1 * r);
        Real P2 = props.exp_B * Kokkos::exp(-props.exp_R2 * r);
        Real denom = props.exp_omega * r;
        if (denom < 1.0e-30) denom = 1.0e-30;
        return P1 + P2 + props.exp_C / denom;
    }

    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave43Properties& props, Real rho, Real e) {
        if (rho < 1.0e-30) return 0.0;
        Real drho = 1.0e-3 * rho;
        if (drho < 1.0e-10) drho = 1.0e-10;
        Real p1 = compute_pressure(props, rho + drho, e);
        Real p2 = compute_pressure(props, rho - drho, e);
        Real dpdrho = (p1 - p2) / (2.0 * drho);
        return (dpdrho > 0.0) ? Kokkos::sqrt(dpdrho) : 0.0;
    }
};

// ============================================================================
// IdealGasVT EOS (Volume-Temperature coupled)
// ============================================================================

/**
 * @brief Ideal gas EOS using explicit temperature variable
 *
 * P = rho * R_gas * T
 *
 * Energy equation (Cv-based):
 *   e = Cv * T + e_ref   =>   T = (e - e_ref) / Cv
 *
 * This avoids the ambiguity of e in the standard ideal gas P = (gamma-1)*rho*e
 * when temperature is tracked as an independent state variable.
 *
 * State: temperature stored in EOSWave43State.temperature
 * Default initialization: T = T0 (reference temperature from props)
 */
class IdealGasVTEOS {
public:
    /**
     * @brief Pressure from density and temperature
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure_T(const EOSWave43Properties& props, Real rho, Real T) {
        if (rho < 0.0) rho = 0.0;
        if (T < 0.0) T = 0.0;
        return rho * props.igt_R_gas * T;
    }

    /**
     * @brief Pressure from density and specific internal energy
     * T = e / Cv  (using Cv from props)
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave43Properties& props, Real rho, Real e) {
        Real Cv = (props.igt_Cv > 1.0e-30) ? props.igt_Cv : 718.0;
        Real T = e / Cv;
        if (T < 0.0) T = 0.0;
        return compute_pressure_T(props, rho, T);
    }

    /**
     * @brief Update temperature from energy change
     * T_new = e_new / Cv
     */
    KOKKOS_INLINE_FUNCTION
    static void update_temperature(const EOSWave43Properties& props,
                                    Real e_new,
                                    EOSWave43State& state) {
        Real Cv = (props.igt_Cv > 1.0e-30) ? props.igt_Cv : 718.0;
        state.temperature = e_new / Cv;
        if (state.temperature < 0.0) state.temperature = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave43Properties& props, Real rho, Real e) {
        Real Cv = (props.igt_Cv > 1.0e-30) ? props.igt_Cv : 718.0;
        Real T = e / Cv;
        if (T < 0.0) T = 0.0;
        // c^2 = gamma * R_gas * T = (Cp/Cv) * R_gas * T
        // gamma = Cp/Cv = (Cv + R_gas) / Cv
        Real gamma = (Cv + props.igt_R_gas) / Cv;
        Real c2 = gamma * props.igt_R_gas * T;
        return (c2 > 0.0) ? Kokkos::sqrt(c2) : 0.0;
    }
};

// ============================================================================
// Compaction2 EOS (2nd-gen compaction with separate unloading modulus)
// ============================================================================

/**
 * @brief 2nd-generation Compaction EOS
 *
 * Loading: P = f(mu) from user-defined tabulated curve,
 *              where mu = rho/rho0 - 1 (positive in compression)
 *
 * Unloading: Linear from (mu_max, P_max) with separate slope K_unload
 *   P_unload = P_max - K_unload * (mu_max - mu)
 *
 * The key distinction from Wave 13 Compaction is:
 * - Uses mu = rho/rho0-1 instead of eps_v (same variable, cleaner API)
 * - K_unload can be different from the tangent of the loading curve
 * - State tracks both mu_max and P_max for fidelity
 *
 * State: mu_max in EOSWave43State.mu_max, P_max in EOSWave43State.p_max
 */
class Compaction2EOS {
public:
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave43Properties& props,
                                  Real rho, Real /*e*/,
                                  EOSWave43State& state) {
        if (props.rho0 < 1.0e-30) return 0.0;
        Real mu = rho / props.rho0 - 1.0;

        if (mu <= 0.0) return 0.0;  // Tension: no compaction pressure

        // Loading curve pressure
        Real P_load = props.compact2_curve.evaluate(mu);

        if (mu >= state.mu_max) {
            // On loading path: advance state
            state.mu_max = mu;
            state.p_max = P_load;
            return P_load;
        } else {
            // Unloading: linear from (mu_max, P_max) with K_unload
            Real P_unload = state.p_max
                          - props.compact2_K_unload * (state.mu_max - mu);
            return (P_unload > 0.0) ? P_unload : 0.0;
        }
    }

    /**
     * @brief Stateless overload: always on loading path
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave43Properties& props, Real rho, Real e) {
        EOSWave43State state;
        state.mu_max = 0.0;
        return compute_pressure(props, rho, e, state);
    }

    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave43Properties& props, Real rho, Real /*e*/) {
        if (props.rho0 < 1.0e-30 || rho < 1.0e-30) return 0.0;
        Real mu = rho / props.rho0 - 1.0;
        Real dmu = 1.0e-3;
        Real p1 = props.compact2_curve.evaluate(mu + dmu);
        Real p2 = props.compact2_curve.evaluate(mu - dmu);
        Real dpd_mu = (p1 - p2) / (2.0 * dmu);
        // dp/drho = (1/rho0) * dp/dmu
        Real dpdrho = dpd_mu / props.rho0;
        return (dpdrho > 0.0) ? Kokkos::sqrt(dpdrho) : 0.0;
    }
};

// ============================================================================
// CompactionTab EOS (Fully tabulated compaction curves)
// ============================================================================

/**
 * @brief Fully tabulated compaction EOS
 *
 * Loading: P = loading_curve(mu)    [tabulated 1D curve]
 * Unloading: P = unloading_curve(mu) [tabulated 1D curve built at mu_max_ref]
 *
 * For states where mu_max < mu_max_ref, the unloading slope is scaled:
 *   P_unload(mu) = P_load(mu_max) + [P_unload_ref(mu) - P_unload_ref(mu_max)]
 *                  * P_load(mu_max) / P_load(mu_max_ref)
 *
 * This provides a physically consistent unloading response at any loading state.
 *
 * State: mu_max in EOSWave43State.mu_max
 */
class CompactionTabEOS {
public:
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave43Properties& props,
                                  Real rho, Real /*e*/,
                                  EOSWave43State& state) {
        if (props.rho0 < 1.0e-30) return 0.0;
        Real mu = rho / props.rho0 - 1.0;

        if (mu <= 0.0) return 0.0;

        Real P_load_cur = props.comptab_load.evaluate(mu);

        if (mu >= state.mu_max) {
            // Loading: advance state
            state.mu_max = mu;
            state.p_max = P_load_cur;
            return P_load_cur;
        } else {
            // Unloading: use tabulated unload curve scaled by loading state
            Real P_unload_at_mu = props.comptab_unload.evaluate(mu);
            Real P_unload_at_mu_max = props.comptab_unload.evaluate(state.mu_max);
            Real P_load_at_mu_max_ref = props.comptab_load.evaluate(props.comptab_mu_max_ref);
            Real P_load_at_mu_max = props.comptab_load.evaluate(state.mu_max);

            // Scale factor to adjust unloading for current mu_max vs reference mu_max
            Real scale = 1.0;
            if (P_load_at_mu_max_ref > 1.0e-30) {
                scale = P_load_at_mu_max / P_load_at_mu_max_ref;
            }

            Real delta_P = (P_unload_at_mu - P_unload_at_mu_max) * scale;
            Real P_unload = state.p_max + delta_P;
            return (P_unload > 0.0) ? P_unload : 0.0;
        }
    }

    /**
     * @brief Stateless overload: loading path only
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave43Properties& props, Real rho, Real e) {
        EOSWave43State state;
        state.mu_max = 0.0;
        return compute_pressure(props, rho, e, state);
    }

    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave43Properties& props, Real rho, Real /*e*/) {
        if (props.rho0 < 1.0e-30 || rho < 1.0e-30) return 0.0;
        Real mu = rho / props.rho0 - 1.0;
        Real dmu = 1.0e-3;
        Real p1 = props.comptab_load.evaluate(mu + dmu);
        Real p2 = props.comptab_load.evaluate(mu - dmu);
        Real dpd_mu = (p1 - p2) / (2.0 * dmu);
        Real dpdrho = dpd_mu / props.rho0;
        return (dpdrho > 0.0) ? Kokkos::sqrt(dpdrho) : 0.0;
    }
};

// ============================================================================
// Unified Dispatch
// ============================================================================

/**
 * @brief Unified dispatch for all Wave 43 EOS models
 *
 * Stateless overloads used for all models by default.
 * History-dependent models (LSZK, Compaction2, CompactionTab) fall back
 * to their loading-path or fully-reacted defaults.
 */
class EOSWave43 {
public:
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSWave43Properties& props, Real rho, Real e) {
        switch (props.type) {
            case EOSWave43Type::LSZK:
                return LSZKEOS::compute_pressure(props, rho, e);
            case EOSWave43Type::NASG:
                return NASGEOS::compute_pressure(props, rho, e);
            case EOSWave43Type::Puff:
                return PuffEOS::compute_pressure(props, rho, e);
            case EOSWave43Type::Exponential:
                return ExponentialEOS::compute_pressure(props, rho, e);
            case EOSWave43Type::IdealGasVT:
                return IdealGasVTEOS::compute_pressure(props, rho, e);
            case EOSWave43Type::Compaction2:
                return Compaction2EOS::compute_pressure(props, rho, e);
            case EOSWave43Type::CompactionTab:
                return CompactionTabEOS::compute_pressure(props, rho, e);
            default:
                return 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSWave43Properties& props, Real rho, Real e) {
        switch (props.type) {
            case EOSWave43Type::LSZK:
                return LSZKEOS::sound_speed(props, rho, e);
            case EOSWave43Type::NASG:
                return NASGEOS::sound_speed(props, rho, e);
            case EOSWave43Type::Puff:
                return PuffEOS::sound_speed(props, rho, e);
            case EOSWave43Type::Exponential:
                return ExponentialEOS::sound_speed(props, rho, e);
            case EOSWave43Type::IdealGasVT:
                return IdealGasVTEOS::sound_speed(props, rho, e);
            case EOSWave43Type::Compaction2:
                return Compaction2EOS::sound_speed(props, rho, e);
            case EOSWave43Type::CompactionTab:
                return CompactionTabEOS::sound_speed(props, rho, e);
            default:
                return 0.0;
        }
    }
};

} // namespace physics
} // namespace nxs
