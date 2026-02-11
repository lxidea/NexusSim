#pragma once

/**
 * @file eos.hpp
 * @brief Equation of State models for fluid/gas/explosive elements
 *
 * Models:
 * - Ideal Gas: p = (γ-1) * ρ * e
 * - Gruneisen: Shock hugoniot with thermal contribution
 * - JWL: Jones-Wilkins-Lee for detonation products
 * - Linear Polynomial: p = C0 + C1*μ + C2*μ² + C3*μ³ + (C4+C5*μ+C6*μ²)*E
 * - Tabulated: User-defined pressure vs density/energy
 *
 * All functions are KOKKOS_INLINE_FUNCTION for GPU compatibility.
 *
 * Reference: LS-DYNA Theory Manual, Chapter 30 (Equations of State)
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <nexussim/physics/material.hpp>
#include <cmath>

namespace nxs {
namespace physics {

// ============================================================================
// EOS Types
// ============================================================================

enum class EOSType {
    IdealGas,          ///< Ideal gas law
    Gruneisen,         ///< Mie-Gruneisen (shock-based)
    JWL,               ///< Jones-Wilkins-Lee (explosives)
    LinearPolynomial,  ///< General polynomial
    Tabulated          ///< User-defined table
};

// ============================================================================
// EOS Properties
// ============================================================================

struct EOSProperties {
    EOSType type;

    // Common
    Real rho0;         ///< Reference density
    Real gamma;        ///< Ratio of specific heats (ideal gas) or Gruneisen gamma

    // Gruneisen parameters
    Real C0;           ///< Bulk sound speed
    Real S1;           ///< Linear Hugoniot slope coefficient
    Real S2;           ///< Quadratic Hugoniot slope coefficient
    Real S3;           ///< Cubic Hugoniot slope coefficient
    Real gamma0;       ///< Gruneisen constant
    Real a_coeff;      ///< First-order volume correction to gamma

    // JWL parameters
    Real A_jwl;        ///< JWL coefficient A
    Real B_jwl;        ///< JWL coefficient B
    Real R1;           ///< JWL exponent R1
    Real R2;           ///< JWL exponent R2
    Real omega;        ///< JWL omega (Gruneisen coefficient)

    // Linear polynomial coefficients
    Real C_poly[7];    ///< C0..C6

    // Tabulated
    TabulatedCurve pressure_table;  ///< Pressure vs volumetric strain

    EOSProperties()
        : type(EOSType::IdealGas)
        , rho0(1.225), gamma(1.4)
        , C0(0.0), S1(0.0), S2(0.0), S3(0.0), gamma0(0.0), a_coeff(0.0)
        , A_jwl(0.0), B_jwl(0.0), R1(0.0), R2(0.0), omega(0.0) {
        for (int i = 0; i < 7; ++i) C_poly[i] = 0.0;
    }
};

// ============================================================================
// Equation of State Functions
// ============================================================================

class EquationOfState {
public:
    /**
     * @brief Compute pressure from current state
     * @param props EOS properties
     * @param rho Current density
     * @param e Specific internal energy
     * @return Pressure
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_pressure(const EOSProperties& props,
                                  Real rho, Real e) {
        switch (props.type) {
            case EOSType::IdealGas:
                return ideal_gas_pressure(props, rho, e);
            case EOSType::Gruneisen:
                return gruneisen_pressure(props, rho, e);
            case EOSType::JWL:
                return jwl_pressure(props, rho, e);
            case EOSType::LinearPolynomial:
                return polynomial_pressure(props, rho, e);
            case EOSType::Tabulated:
                return tabulated_pressure(props, rho);
            default:
                return 0.0;
        }
    }

    /**
     * @brief Compute sound speed
     * @return Sound speed c
     */
    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const EOSProperties& props, Real rho, Real e) {
        switch (props.type) {
            case EOSType::IdealGas: {
                Real p = ideal_gas_pressure(props, rho, e);
                return (rho > 1.0e-30) ? Kokkos::sqrt(props.gamma * p / rho) : 0.0;
            }
            case EOSType::Gruneisen: {
                return (rho > 1.0e-30) ? props.C0 : 0.0;
            }
            default: {
                // Numerical differentiation
                Real dp = 0.01 * rho;
                Real p1 = compute_pressure(props, rho + dp, e);
                Real p2 = compute_pressure(props, rho - dp, e);
                Real dpdrho = (p1 - p2) / (2.0 * dp);
                return (dpdrho > 0.0) ? Kokkos::sqrt(dpdrho) : 0.0;
            }
        }
    }

private:
    /**
     * @brief Ideal gas: p = (γ-1) * ρ * e
     */
    KOKKOS_INLINE_FUNCTION
    static Real ideal_gas_pressure(const EOSProperties& props,
                                    Real rho, Real e) {
        return (props.gamma - 1.0) * rho * e;
    }

    /**
     * @brief Mie-Gruneisen EOS
     *
     * Compression (μ > 0):
     *   p = [ρ₀C₀²μ(1+(1-γ₀/2)μ-a/2·μ²)] / [1-(S₁-1)μ-S₂·μ²/(μ+1)-S₃·μ³/(μ+1)²]² + (γ₀+aμ)·E
     *
     * Tension (μ < 0):
     *   p = ρ₀C₀²μ + (γ₀+aμ)·E
     *
     * Where μ = ρ/ρ₀ - 1 (compression ratio)
     */
    KOKKOS_INLINE_FUNCTION
    static Real gruneisen_pressure(const EOSProperties& props,
                                    Real rho, Real e) {
        Real mu = rho / props.rho0 - 1.0;
        Real gamma_v = props.gamma0 + props.a_coeff * mu;
        Real E_vol = rho * e;  // Volumetric energy

        if (mu > 0.0) {
            // Compression
            Real mu2 = mu * mu;
            Real mu3 = mu2 * mu;
            Real num = props.rho0 * props.C0 * props.C0 * mu *
                       (1.0 + (1.0 - props.gamma0/2.0)*mu - props.a_coeff/2.0*mu2);
            Real denom = 1.0 - (props.S1 - 1.0)*mu
                             - props.S2*mu2/(mu + 1.0 + 1.0e-30)
                             - props.S3*mu3/((mu + 1.0)*(mu + 1.0) + 1.0e-30);
            denom = denom * denom;
            if (denom < 1.0e-30) denom = 1.0e-30;
            return num / denom + gamma_v * E_vol;
        } else {
            // Tension
            return props.rho0 * props.C0 * props.C0 * mu + gamma_v * E_vol;
        }
    }

    /**
     * @brief JWL EOS for detonation products
     * p = A(1-ω/(R₁V))exp(-R₁V) + B(1-ω/(R₂V))exp(-R₂V) + ωE/V
     * Where V = ρ₀/ρ (relative volume)
     */
    KOKKOS_INLINE_FUNCTION
    static Real jwl_pressure(const EOSProperties& props,
                              Real rho, Real e) {
        Real V = (rho > 1.0e-30) ? props.rho0 / rho : 1.0e10;
        Real E_vol = rho * e;

        Real p = props.A_jwl * (1.0 - props.omega/(props.R1*V)) *
                     Kokkos::exp(-props.R1*V)
               + props.B_jwl * (1.0 - props.omega/(props.R2*V)) *
                     Kokkos::exp(-props.R2*V)
               + props.omega * E_vol / V;

        return p;
    }

    /**
     * @brief Linear polynomial EOS
     * p = C0 + C1*μ + C2*μ² + C3*μ³ + (C4 + C5*μ + C6*μ²)*E
     */
    KOKKOS_INLINE_FUNCTION
    static Real polynomial_pressure(const EOSProperties& props,
                                     Real rho, Real e) {
        Real mu = rho / props.rho0 - 1.0;
        if (mu < 0.0) {
            // Tension: C2 = 0
            return props.C_poly[0] + props.C_poly[1]*mu
                 + (props.C_poly[4] + props.C_poly[5]*mu) * e;
        }
        Real mu2 = mu * mu;
        return props.C_poly[0] + props.C_poly[1]*mu + props.C_poly[2]*mu2 + props.C_poly[3]*mu2*mu
             + (props.C_poly[4] + props.C_poly[5]*mu + props.C_poly[6]*mu2) * e;
    }

    /**
     * @brief Tabulated EOS: interpolate from pressure vs volumetric strain table
     */
    KOKKOS_INLINE_FUNCTION
    static Real tabulated_pressure(const EOSProperties& props, Real rho) {
        Real ev = rho / props.rho0 - 1.0;  // Volumetric strain
        return props.pressure_table.evaluate(ev);
    }
};

} // namespace physics
} // namespace nxs
