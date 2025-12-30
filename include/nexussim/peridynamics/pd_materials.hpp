#pragma once

/**
 * @file pd_materials.hpp
 * @brief Advanced material models for peridynamics
 *
 * Includes:
 * - Johnson-Cook (metals with strain rate and temperature effects)
 * - Drucker-Prager (geomaterials: soil, rock, concrete)
 * - Johnson-Holmquist 2 (ceramics and glass)
 *
 * Ported from PeriSys-Haoran Global_Para.cuh
 */

#include <nexussim/peridynamics/pd_types.hpp>
#include <cmath>

namespace nxs {
namespace pd {

// ============================================================================
// Johnson-Cook Material Model
// ============================================================================

/**
 * @brief Johnson-Cook material model for metals
 *
 * Flow stress:
 *   σ_y = (A + B ε_p^n)(1 + C ln(ε̇/ε̇_0))(1 - T*^m)
 *
 * where:
 *   ε_p = effective plastic strain
 *   ε̇ = strain rate
 *   T* = (T - T_room)/(T_melt - T_room) = homologous temperature
 *
 * Damage model:
 *   D = Σ(Δε_p / ε_f)
 *   ε_f = (D1 + D2 exp(D3 σ*))(1 + D4 ln(ε̇*))(1 + D5 T*)
 *   σ* = σ_m / σ_eq (stress triaxiality)
 */
struct JohnsonCookMaterial {
    // Strength parameters
    Real A = 0.0;           ///< Initial yield stress (Pa)
    Real B = 0.0;           ///< Hardening coefficient (Pa)
    Real n = 0.0;           ///< Hardening exponent
    Real C = 0.0;           ///< Strain rate coefficient
    Real m = 0.0;           ///< Thermal softening exponent

    // Reference conditions
    Real eps_dot_0 = 1.0;   ///< Reference strain rate (1/s)
    Real T_room = 300.0;    ///< Room temperature (K)
    Real T_melt = 1800.0;   ///< Melting temperature (K)

    // Damage parameters (optional)
    Real D1 = 0.0;          ///< Damage constant
    Real D2 = 0.0;          ///< Damage constant
    Real D3 = 0.0;          ///< Damage constant
    Real D4 = 0.0;          ///< Damage constant
    Real D5 = 0.0;          ///< Damage constant

    // Basic elastic properties
    Real E = 2.0e11;        ///< Young's modulus (Pa)
    Real nu = 0.3;          ///< Poisson's ratio
    Real rho = 7800.0;      ///< Density (kg/m³)
    Real Cp = 460.0;        ///< Specific heat (J/kg·K)

    /**
     * @brief Compute flow stress
     */
    KOKKOS_INLINE_FUNCTION
    Real flow_stress(Real eps_p, Real eps_dot, Real T) const {
        // Hardening term
        Real hardening = A + B * std::pow(eps_p + 1e-10, n);

        // Strain rate term
        Real rate_factor = 1.0;
        if (C > 0.0 && eps_dot > eps_dot_0) {
            rate_factor = 1.0 + C * std::log(eps_dot / eps_dot_0);
        }

        // Thermal softening term
        Real T_star = 0.0;
        if (T > T_room && T_melt > T_room) {
            T_star = (T - T_room) / (T_melt - T_room);
            T_star = Kokkos::fmin(T_star, 1.0);
        }
        Real thermal_factor = 1.0 - std::pow(T_star, m);

        return hardening * rate_factor * thermal_factor;
    }

    /**
     * @brief Compute failure strain
     */
    KOKKOS_INLINE_FUNCTION
    Real failure_strain(Real sigma_star, Real eps_dot_star, Real T_star) const {
        // Stress triaxiality term
        Real triax_term = D1 + D2 * std::exp(D3 * sigma_star);

        // Strain rate term
        Real rate_term = 1.0 + D4 * std::log(Kokkos::fmax(eps_dot_star, 1.0));

        // Temperature term
        Real temp_term = 1.0 + D5 * T_star;

        return triax_term * rate_term * temp_term;
    }
};

/**
 * @brief Common Johnson-Cook material presets
 */
namespace JCPresets {
    /**
     * @brief Aluminum 7075-T6
     */
    inline JohnsonCookMaterial Al7075T6() {
        JohnsonCookMaterial mat;
        mat.A = 5.46e8;
        mat.B = 6.78e8;
        mat.n = 0.71;
        mat.C = 0.024;
        mat.m = 1.56;
        mat.E = 71.7e9;
        mat.nu = 0.33;
        mat.rho = 2810.0;
        mat.T_melt = 893.0;
        mat.T_room = 293.0;
        return mat;
    }

    /**
     * @brief Steel 4340
     */
    inline JohnsonCookMaterial Steel4340() {
        JohnsonCookMaterial mat;
        mat.A = 7.92e8;
        mat.B = 5.10e8;
        mat.n = 0.26;
        mat.C = 0.014;
        mat.m = 1.03;
        mat.E = 200.0e9;
        mat.nu = 0.29;
        mat.rho = 7830.0;
        mat.T_melt = 1793.0;
        mat.T_room = 293.0;
        return mat;
    }

    /**
     * @brief Titanium Ti-6Al-4V
     */
    inline JohnsonCookMaterial Ti6Al4V() {
        JohnsonCookMaterial mat;
        mat.A = 1.098e9;
        mat.B = 1.092e9;
        mat.n = 0.93;
        mat.C = 0.014;
        mat.m = 1.1;
        mat.E = 113.8e9;
        mat.nu = 0.342;
        mat.rho = 4430.0;
        mat.T_melt = 1933.0;
        mat.T_room = 293.0;
        return mat;
    }

    /**
     * @brief Copper OFHC
     */
    inline JohnsonCookMaterial CopperOFHC() {
        JohnsonCookMaterial mat;
        mat.A = 9.0e7;
        mat.B = 2.92e8;
        mat.n = 0.31;
        mat.C = 0.025;
        mat.m = 1.09;
        mat.E = 124.0e9;
        mat.nu = 0.34;
        mat.rho = 8960.0;
        mat.T_melt = 1356.0;
        mat.T_room = 293.0;
        return mat;
    }
}

// ============================================================================
// Drucker-Prager Material Model
// ============================================================================

/**
 * @brief Drucker-Prager material model for geomaterials
 *
 * Yield function:
 *   f = √J₂ + α I₁ - k = 0
 *
 * where:
 *   J₂ = second invariant of deviatoric stress
 *   I₁ = first invariant of stress (trace)
 *   α = (2 sin φ) / (√3 (3 - sin φ))
 *   k = (6 c cos φ) / (√3 (3 - sin φ))
 *   φ = friction angle
 *   c = cohesion
 */
struct DruckerPragerMaterial {
    // Drucker-Prager parameters
    Real phi = 30.0;        ///< Friction angle (degrees)
    Real c = 1.0e6;         ///< Cohesion (Pa)
    Real psi = 0.0;         ///< Dilation angle (degrees)

    // Derived parameters
    Real alpha = 0.0;       ///< Pressure coefficient
    Real k = 0.0;           ///< Cohesion coefficient
    Real alpha_psi = 0.0;   ///< Dilation coefficient

    // Elastic properties
    Real E = 30.0e9;        ///< Young's modulus (Pa)
    Real nu = 0.2;          ///< Poisson's ratio
    Real rho = 2400.0;      ///< Density (kg/m³)

    // Tensile cutoff
    Real tensile_strength = 1.0e6;  ///< Tensile cutoff (Pa)

    /**
     * @brief Compute derived parameters
     */
    void compute_derived() {
        Real pi = 3.14159265358979323846;
        Real phi_rad = phi * pi / 180.0;
        Real psi_rad = psi * pi / 180.0;

        Real sin_phi = std::sin(phi_rad);
        Real cos_phi = std::cos(phi_rad);
        Real sqrt3 = std::sqrt(3.0);

        // DP parameters (compression cone)
        alpha = 2.0 * sin_phi / (sqrt3 * (3.0 - sin_phi));
        k = 6.0 * c * cos_phi / (sqrt3 * (3.0 - sin_phi));

        // Dilation
        Real sin_psi = std::sin(psi_rad);
        alpha_psi = 2.0 * sin_psi / (sqrt3 * (3.0 - sin_psi));
    }

    /**
     * @brief Evaluate yield function
     *
     * @param p Mean stress (positive in compression)
     * @param q von Mises stress (√(3 J₂))
     * @return Yield function value (< 0 elastic, = 0 yield)
     */
    KOKKOS_INLINE_FUNCTION
    Real yield_function(Real p, Real q) const {
        // f = q + 3α p - k (using q = √3 √J₂)
        return q + 3.0 * alpha * p - k;
    }

    /**
     * @brief Compute plastic multiplier
     */
    KOKKOS_INLINE_FUNCTION
    Real plastic_multiplier(Real p, Real q, Real G, Real K) const {
        Real f = yield_function(p, q);
        if (f <= 0.0) return 0.0;

        // Consistent return mapping
        Real denom = 3.0 * G + 9.0 * K * alpha * alpha_psi;
        return f / denom;
    }
};

/**
 * @brief Common Drucker-Prager presets
 */
namespace DPPresets {
    /**
     * @brief Loose sand
     */
    inline DruckerPragerMaterial LooseSand() {
        DruckerPragerMaterial mat;
        mat.phi = 30.0;
        mat.c = 0.0;
        mat.psi = 0.0;
        mat.E = 20.0e6;
        mat.nu = 0.3;
        mat.rho = 1600.0;
        mat.compute_derived();
        return mat;
    }

    /**
     * @brief Dense sand
     */
    inline DruckerPragerMaterial DenseSand() {
        DruckerPragerMaterial mat;
        mat.phi = 40.0;
        mat.c = 0.0;
        mat.psi = 10.0;
        mat.E = 50.0e6;
        mat.nu = 0.25;
        mat.rho = 1900.0;
        mat.compute_derived();
        return mat;
    }

    /**
     * @brief Clay
     */
    inline DruckerPragerMaterial Clay() {
        DruckerPragerMaterial mat;
        mat.phi = 25.0;
        mat.c = 50.0e3;
        mat.psi = 0.0;
        mat.E = 10.0e6;
        mat.nu = 0.4;
        mat.rho = 1800.0;
        mat.compute_derived();
        return mat;
    }

    /**
     * @brief Concrete
     */
    inline DruckerPragerMaterial Concrete() {
        DruckerPragerMaterial mat;
        mat.phi = 35.0;
        mat.c = 3.0e6;
        mat.psi = 15.0;
        mat.E = 30.0e9;
        mat.nu = 0.2;
        mat.rho = 2400.0;
        mat.tensile_strength = 3.0e6;
        mat.compute_derived();
        return mat;
    }

    /**
     * @brief Granite
     */
    inline DruckerPragerMaterial Granite() {
        DruckerPragerMaterial mat;
        mat.phi = 50.0;
        mat.c = 20.0e6;
        mat.psi = 10.0;
        mat.E = 60.0e9;
        mat.nu = 0.25;
        mat.rho = 2700.0;
        mat.tensile_strength = 10.0e6;
        mat.compute_derived();
        return mat;
    }
}

// ============================================================================
// Johnson-Holmquist 2 (JH-2) Material Model
// ============================================================================

/**
 * @brief Johnson-Holmquist 2 model for ceramics and glass
 *
 * Strength model:
 *   σ* = σ*_i - D(σ*_i - σ*_f)
 *   σ*_i = A(P* + T*)^N (1 + C ln ε̇*)  (intact)
 *   σ*_f = B(P*)^M (1 + C ln ε̇*)        (fractured)
 *
 * Damage:
 *   D = Σ(Δε_p / ε_f)
 *   ε_f = D1(P* + T*)^D2
 *
 * EOS:
 *   P = K1 μ + K2 μ² + K3 μ³ + ΔP
 *   μ = ρ/ρ₀ - 1 (compression)
 */
struct JohnsonHolmquist2Material {
    // Strength parameters (normalized)
    Real A = 0.93;          ///< Intact strength parameter
    Real B = 0.31;          ///< Fractured strength parameter
    Real C = 0.0;           ///< Strain rate coefficient
    Real M = 0.6;           ///< Fractured strength exponent
    Real N = 0.6;           ///< Intact strength exponent

    // Damage parameters
    Real D1 = 0.005;        ///< Damage constant
    Real D2 = 1.0;          ///< Damage exponent

    // Reference values
    Real sigma_HEL = 4.5e9;     ///< Hugoniot elastic limit (Pa)
    Real P_HEL = 2.5e9;         ///< Pressure at HEL (Pa)
    Real T = 0.2e9;             ///< Maximum tensile hydrostatic pressure (Pa)

    // EOS parameters
    Real K1 = 130.0e9;      ///< Bulk modulus (Pa)
    Real K2 = 0.0;          ///< Second order constant
    Real K3 = 0.0;          ///< Third order constant
    Real beta = 1.0;        ///< Bulking factor

    // Elastic properties
    Real E = 370.0e9;       ///< Young's modulus (Pa)
    Real nu = 0.22;         ///< Poisson's ratio
    Real rho = 3900.0;      ///< Density (kg/m³)

    /**
     * @brief Compute normalized intact strength
     */
    KOKKOS_INLINE_FUNCTION
    Real intact_strength(Real P_star, Real T_star, Real eps_dot_star) const {
        Real strength = A * std::pow(P_star + T_star, N);
        if (C > 0.0 && eps_dot_star > 1.0) {
            strength *= (1.0 + C * std::log(eps_dot_star));
        }
        return strength;
    }

    /**
     * @brief Compute normalized fractured strength
     */
    KOKKOS_INLINE_FUNCTION
    Real fractured_strength(Real P_star, Real eps_dot_star) const {
        if (P_star <= 0.0) return 0.0;
        Real strength = B * std::pow(P_star, M);
        if (C > 0.0 && eps_dot_star > 1.0) {
            strength *= (1.0 + C * std::log(eps_dot_star));
        }
        return strength;
    }

    /**
     * @brief Compute failure strain
     */
    KOKKOS_INLINE_FUNCTION
    Real failure_strain(Real P_star, Real T_star) const {
        Real arg = P_star + T_star;
        if (arg <= 0.0) return 1e10;  // No failure in tension beyond cutoff
        return D1 * std::pow(arg, D2);
    }
};

/**
 * @brief JH-2 material presets
 */
namespace JH2Presets {
    /**
     * @brief Alumina (Al₂O₃)
     */
    inline JohnsonHolmquist2Material Alumina() {
        JohnsonHolmquist2Material mat;
        mat.A = 0.93;
        mat.B = 0.31;
        mat.C = 0.0;
        mat.M = 0.6;
        mat.N = 0.6;
        mat.D1 = 0.005;
        mat.D2 = 1.0;
        mat.sigma_HEL = 4.5e9;
        mat.P_HEL = 2.5e9;
        mat.T = 0.2e9;
        mat.K1 = 130.0e9;
        mat.E = 370.0e9;
        mat.nu = 0.22;
        mat.rho = 3900.0;
        return mat;
    }

    /**
     * @brief Silicon Carbide (SiC)
     */
    inline JohnsonHolmquist2Material SiliconCarbide() {
        JohnsonHolmquist2Material mat;
        mat.A = 0.96;
        mat.B = 0.35;
        mat.C = 0.0;
        mat.M = 1.0;
        mat.N = 0.65;
        mat.D1 = 0.02;
        mat.D2 = 0.83;
        mat.sigma_HEL = 11.7e9;
        mat.P_HEL = 5.9e9;
        mat.T = 0.75e9;
        mat.K1 = 220.0e9;
        mat.E = 449.0e9;
        mat.nu = 0.16;
        mat.rho = 3215.0;
        return mat;
    }

    /**
     * @brief Boron Carbide (B₄C)
     */
    inline JohnsonHolmquist2Material BoronCarbide() {
        JohnsonHolmquist2Material mat;
        mat.A = 0.927;
        mat.B = 0.7;
        mat.C = 0.005;
        mat.M = 0.85;
        mat.N = 0.67;
        mat.D1 = 0.001;
        mat.D2 = 0.5;
        mat.sigma_HEL = 19.0e9;
        mat.P_HEL = 8.71e9;
        mat.T = 0.26e9;
        mat.K1 = 233.0e9;
        mat.E = 460.0e9;
        mat.nu = 0.17;
        mat.rho = 2510.0;
        return mat;
    }

    /**
     * @brief Soda-Lime Glass
     */
    inline JohnsonHolmquist2Material SodaLimeGlass() {
        JohnsonHolmquist2Material mat;
        mat.A = 0.93;
        mat.B = 0.088;
        mat.C = 0.003;
        mat.M = 0.35;
        mat.N = 0.77;
        mat.D1 = 0.053;
        mat.D2 = 0.85;
        mat.sigma_HEL = 5.95e9;
        mat.P_HEL = 4.5e9;
        mat.T = 0.15e9;
        mat.K1 = 45.4e9;
        mat.E = 72.0e9;
        mat.nu = 0.22;
        mat.rho = 2530.0;
        return mat;
    }
}

} // namespace pd
} // namespace nxs
