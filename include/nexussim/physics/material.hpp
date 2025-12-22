#pragma once

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <nexussim/core/exception.hpp>
#include <memory>
#include <string>

namespace nxs {
namespace physics {

// ============================================================================
// Material Model Types
// ============================================================================

enum class MaterialType {
    Elastic,           ///< Linear elastic
    Plastic,           ///< Elasto-plastic (J2 plasticity)
    Hyperelastic,      ///< Hyperelastic (Neo-Hookean, Mooney-Rivlin)
    Viscoelastic,      ///< Viscoelastic
    Damage,            ///< Damage mechanics
    Composite,         ///< Composite material
    Custom             ///< User-defined
};

// ============================================================================
// Material State
// ============================================================================

/**
 * @brief Material state at an integration point
 */
struct MaterialState {
    // Strain tensor (Voigt notation: εxx, εyy, εzz, γxy, γyz, γxz)
    Real strain[6];

    // Stress tensor (Voigt notation: σxx, σyy, σzz, τxy, τyz, τxz)
    Real stress[6];

    // Deformation gradient (3x3 matrix for large deformation)
    Real F[9];

    // History variables (plastic strain, damage, etc.)
    Real history[10];

    // Volumetric strain
    Real vol_strain;

    // Effective plastic strain
    Real plastic_strain;

    // Temperature (for thermo-mechanical coupling)
    Real temperature;

    // Damage parameter
    Real damage;

    // Strain rate tensor (for rate-dependent materials)
    Real strain_rate[6];

    // Effective strain rate (scalar)
    Real effective_strain_rate;

    // Time step (for rate calculations)
    Real dt;

    KOKKOS_INLINE_FUNCTION
    MaterialState() {
        for (int i = 0; i < 6; ++i) {
            strain[i] = 0.0;
            stress[i] = 0.0;
            strain_rate[i] = 0.0;
        }
        for (int i = 0; i < 9; ++i) F[i] = 0.0;
        F[0] = F[4] = F[8] = 1.0;  // Identity
        for (int i = 0; i < 10; ++i) history[i] = 0.0;
        vol_strain = 0.0;
        plastic_strain = 0.0;
        temperature = 293.15;  // Room temperature
        damage = 0.0;
        effective_strain_rate = 0.0;
        dt = 1.0e-6;  // Default time step
    }
};

// ============================================================================
// Material Parameters
// ============================================================================

/**
 * @brief Material properties
 */
struct MaterialProperties {
    // Basic properties
    Real density;             ///< Mass density
    Real E;                   ///< Young's modulus
    Real nu;                  ///< Poisson's ratio
    Real G;                   ///< Shear modulus
    Real K;                   ///< Bulk modulus

    // Plasticity parameters (linear hardening)
    Real yield_stress;        ///< Initial yield stress
    Real hardening_modulus;   ///< Isotropic hardening modulus
    Real tangent_modulus;     ///< Kinematic hardening modulus

    // Johnson-Cook parameters: σ_y = (A + B*ε_p^n)(1 + C*ln(ε̇*))(1 - T*^m)
    Real JC_A;                ///< Yield stress constant (Pa)
    Real JC_B;                ///< Hardening constant (Pa)
    Real JC_n;                ///< Hardening exponent
    Real JC_C;                ///< Strain rate coefficient
    Real JC_m;                ///< Thermal softening exponent
    Real JC_eps_dot_ref;      ///< Reference strain rate (1/s)
    Real JC_T_melt;           ///< Melting temperature (K)
    Real JC_T_room;           ///< Reference (room) temperature (K)

    // Damage parameters
    Real damage_threshold;    ///< Damage initiation threshold
    Real damage_exponent;     ///< Damage evolution exponent

    // Thermal properties
    Real specific_heat;       ///< Specific heat capacity
    Real thermal_expansion;   ///< Thermal expansion coefficient
    Real thermal_conductivity;///< Thermal conductivity

    // Wave speeds (for explicit time step)
    Real sound_speed;         ///< Longitudinal wave speed

    KOKKOS_INLINE_FUNCTION
    MaterialProperties()
        : density(1000.0)
        , E(1.0e9)
        , nu(0.3)
        , G(0.0)
        , K(0.0)
        , yield_stress(1.0e6)
        , hardening_modulus(0.0)
        , tangent_modulus(0.0)
        , JC_A(0.0)
        , JC_B(0.0)
        , JC_n(1.0)
        , JC_C(0.0)
        , JC_m(1.0)
        , JC_eps_dot_ref(1.0)
        , JC_T_melt(1800.0)
        , JC_T_room(293.15)
        , damage_threshold(0.0)
        , damage_exponent(1.0)
        , specific_heat(1000.0)
        , thermal_expansion(1.0e-5)
        , thermal_conductivity(50.0)
        , sound_speed(0.0)
    {
        // Compute derived properties
        compute_derived();
    }

    KOKKOS_INLINE_FUNCTION
    void compute_derived() {
        // Shear modulus
        G = E / (2.0 * (1.0 + nu));

        // Bulk modulus
        K = E / (3.0 * (1.0 - 2.0 * nu));

        // Sound speed (longitudinal wave speed)
        sound_speed = Kokkos::sqrt(E * (1.0 - nu) / (density * (1.0 + nu) * (1.0 - 2.0 * nu)));
    }
};

// ============================================================================
// Material Model Base Class
// ============================================================================

/**
 * @brief Base class for material constitutive models
 *
 * This class provides the interface for computing stress from strain
 * and updating material state. All methods are GPU-compatible.
 */
class Material {
public:
    Material(MaterialType type, const MaterialProperties& props)
        : type_(type), props_(props) {}

    virtual ~Material() = default;

    // ========================================================================
    // Stress Computation (GPU-compatible)
    // ========================================================================

    /**
     * @brief Compute stress from strain (small deformation)
     * @param state Material state (input: strain, output: stress)
     */
    KOKKOS_INLINE_FUNCTION
    virtual void compute_stress(MaterialState& state) const = 0;

    /**
     * @brief Compute stress from deformation gradient (large deformation)
     * @param state Material state (input: F, output: stress)
     */
    KOKKOS_INLINE_FUNCTION
    virtual void compute_stress_large_deform(MaterialState& state) const {
        // Default: throw error (not implemented)
        // Cannot throw in device code, so we just return
    }

    /**
     * @brief Compute tangent stiffness matrix (material Jacobian)
     * @param state Current material state
     * @param C Output: Tangent stiffness [6x6] in Voigt notation
     */
    KOKKOS_INLINE_FUNCTION
    virtual void tangent_stiffness(const MaterialState& state,
                                   Real* C) const = 0;

    // ========================================================================
    // Material Properties
    // ========================================================================

    MaterialType type() const { return type_; }
    const MaterialProperties& properties() const { return props_; }

    /**
     * @brief Get wave speed for time step calculation
     */
    KOKKOS_INLINE_FUNCTION
    Real wave_speed() const {
        return props_.sound_speed;
    }

    // ========================================================================
    // Utility Functions
    // ========================================================================

    /**
     * @brief Convert engineering strain to stress (linear elastic)
     */
    KOKKOS_INLINE_FUNCTION
    static void elastic_stress(const Real* strain,
                               Real E,
                               Real nu,
                               Real* stress) {
        const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const Real mu = E / (2.0 * (1.0 + nu));

        // Volumetric strain
        const Real ev = strain[0] + strain[1] + strain[2];

        // Stress (Voigt notation)
        stress[0] = lambda * ev + 2.0 * mu * strain[0];  // σxx
        stress[1] = lambda * ev + 2.0 * mu * strain[1];  // σyy
        stress[2] = lambda * ev + 2.0 * mu * strain[2];  // σzz
        stress[3] = mu * strain[3];                       // τxy
        stress[4] = mu * strain[4];                       // τyz
        stress[5] = mu * strain[5];                       // τxz
    }

    /**
     * @brief Compute von Mises stress
     */
    KOKKOS_INLINE_FUNCTION
    static Real von_mises_stress(const Real* stress) {
        // Deviatoric stress
        const Real p = (stress[0] + stress[1] + stress[2]) / 3.0;
        const Real s[6] = {
            stress[0] - p,
            stress[1] - p,
            stress[2] - p,
            stress[3],
            stress[4],
            stress[5]
        };

        // von Mises stress: sqrt(3/2 * s:s)
        const Real s_norm = s[0]*s[0] + s[1]*s[1] + s[2]*s[2] +
                           2.0*(s[3]*s[3] + s[4]*s[4] + s[5]*s[5]);
        return Kokkos::sqrt(1.5 * s_norm);
    }

protected:
    MaterialType type_;
    MaterialProperties props_;
};

// ============================================================================
// Linear Elastic Material
// ============================================================================

/**
 * @brief Linear elastic material model
 */
class ElasticMaterial : public Material {
public:
    ElasticMaterial(const MaterialProperties& props)
        : Material(MaterialType::Elastic, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        elastic_stress(state.strain, props_.E, props_.nu, state.stress);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state,
                          Real* C) const override {
        (void)state;  // Unused for linear elastic

        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const Real mu = E / (2.0 * (1.0 + nu));
        const Real lambda_2mu = lambda + 2.0 * mu;

        // Initialize to zero
        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        // Fill elasticity tensor (Voigt notation)
        C[0] = C[7] = C[14] = lambda_2mu;  // Diagonal
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;  // Off-diagonal
        C[21] = C[28] = C[35] = mu;  // Shear terms
    }
};

// ============================================================================
// Von Mises (J2) Plasticity Material
// ============================================================================

/**
 * @brief Von Mises plasticity with isotropic hardening
 *
 * Implements J2 plasticity with:
 * - Yield function: f = σ_vm - σ_y(ε_p) = 0
 * - Isotropic hardening: σ_y = σ_y0 + H * ε_p
 * - Associative flow rule
 * - Return mapping algorithm (radial return)
 *
 * History variables:
 * - history[0]: accumulated effective plastic strain ε_p
 * - history[1-6]: plastic strain tensor (Voigt notation)
 */
class VonMisesPlasticMaterial : public Material {
public:
    VonMisesPlasticMaterial(const MaterialProperties& props)
        : Material(MaterialType::Plastic, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;
        const Real sigma_y0 = props_.yield_stress;
        const Real H = props_.hardening_modulus;

        // Get accumulated plastic strain from history
        Real eps_p = state.history[0];

        // Compute trial elastic strain: ε_e_trial = ε - ε_p_old
        Real strain_e[6];
        for (int i = 0; i < 6; ++i) {
            strain_e[i] = state.strain[i] - state.history[i + 1];
        }

        // Compute trial stress (elastic predictor)
        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        // Compute deviatoric trial stress
        const Real p_trial = (stress_trial[0] + stress_trial[1] + stress_trial[2]) / 3.0;
        Real s_trial[6];
        s_trial[0] = stress_trial[0] - p_trial;
        s_trial[1] = stress_trial[1] - p_trial;
        s_trial[2] = stress_trial[2] - p_trial;
        s_trial[3] = stress_trial[3];
        s_trial[4] = stress_trial[4];
        s_trial[5] = stress_trial[5];

        // Compute von Mises stress (trial)
        Real s_norm_sq = s_trial[0]*s_trial[0] + s_trial[1]*s_trial[1] + s_trial[2]*s_trial[2]
                       + 2.0*(s_trial[3]*s_trial[3] + s_trial[4]*s_trial[4] + s_trial[5]*s_trial[5]);
        Real sigma_vm_trial = Kokkos::sqrt(1.5 * s_norm_sq);

        // Current yield stress with hardening
        Real sigma_y = sigma_y0 + H * eps_p;

        // Check yield condition
        Real f_trial = sigma_vm_trial - sigma_y;

        if (f_trial <= 0.0) {
            // Elastic: use trial stress
            for (int i = 0; i < 6; ++i) {
                state.stress[i] = stress_trial[i];
            }
        } else {
            // Plastic: radial return mapping
            // Solve for plastic multiplier Δγ using consistency condition
            // For linear hardening: Δγ = f_trial / (3G + H)
            Real denom = 3.0 * G + H;
            Real delta_gamma = f_trial / denom;

            // Update accumulated plastic strain
            eps_p += delta_gamma;
            state.history[0] = eps_p;

            // Update plastic strain tensor (deviatoric only for J2)
            // Δε_p = Δγ * n, where n = (3/2) * s / σ_vm
            if (sigma_vm_trial > 1.0e-12) {
                Real factor = 1.5 * delta_gamma / sigma_vm_trial;
                state.history[1] += factor * s_trial[0];
                state.history[2] += factor * s_trial[1];
                state.history[3] += factor * s_trial[2];
                state.history[4] += factor * s_trial[3];
                state.history[5] += factor * s_trial[4];
                state.history[6] += factor * s_trial[5];
            }

            // Update stress: σ = s_trial - 2μ*Δγ*n + p*I
            // Where n = (3/2) * s / σ_vm, so the stress update is:
            // s_new = s_trial - 2μ*Δγ*(3/2)*s/σ_vm = s_trial*(1 - 3G*Δγ/σ_vm)
            // (Note: μ = G is shear modulus)
            Real scale = 1.0 - 3.0 * G * delta_gamma / sigma_vm_trial;
            state.stress[0] = scale * s_trial[0] + p_trial;
            state.stress[1] = scale * s_trial[1] + p_trial;
            state.stress[2] = scale * s_trial[2] + p_trial;
            state.stress[3] = scale * s_trial[3];
            state.stress[4] = scale * s_trial[4];
            state.stress[5] = scale * s_trial[5];
        }

        // Update state
        state.plastic_strain = eps_p;
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state,
                          Real* C) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;
        const Real H = props_.hardening_modulus;

        // Initialize elastic stiffness
        const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const Real mu = G;
        const Real lambda_2mu = lambda + 2.0 * mu;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        // Elastic part
        C[0] = C[7] = C[14] = lambda_2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;

        // Check if plastic
        Real sigma_vm = von_mises_stress(state.stress);
        Real sigma_y = props_.yield_stress + H * state.plastic_strain;

        if (sigma_vm >= sigma_y - 1.0e-6 * sigma_y) {
            // Elasto-plastic tangent (consistent tangent)
            // C_ep = C_e - (C_e : n ⊗ n : C_e) / (n : C_e : n + H)

            // Compute deviatoric stress
            const Real p = (state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;
            Real s[6] = {
                state.stress[0] - p,
                state.stress[1] - p,
                state.stress[2] - p,
                state.stress[3],
                state.stress[4],
                state.stress[5]
            };

            // Normal to yield surface
            Real n[6];
            if (sigma_vm > 1.0e-12) {
                Real factor = 1.5 / sigma_vm;
                for (int i = 0; i < 6; ++i) {
                    n[i] = factor * s[i];
                }
            } else {
                for (int i = 0; i < 6; ++i) n[i] = 0.0;
            }

            // Plastic correction factor
            Real theta = 3.0 * G / (3.0 * G + H);

            // Apply plastic correction to deviatoric terms
            // This is a simplified consistent tangent
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    int idx = i * 6 + j;
                    C[idx] -= 2.0 * G * theta * n[i] * n[j];
                }
            }
            // Shear terms
            C[21] *= (1.0 - theta);
            C[28] *= (1.0 - theta);
            C[35] *= (1.0 - theta);
        }
    }
};

// ============================================================================
// Johnson-Cook Plastic Material
// ============================================================================

/**
 * @brief Johnson-Cook plasticity model for high strain-rate applications
 *
 * Yield stress: σ_y = (A + B*ε_p^n)(1 + C*ln(ε̇*))(1 - T*^m)
 *
 * Where:
 *   A = yield stress constant
 *   B = hardening constant
 *   n = hardening exponent
 *   C = strain rate sensitivity
 *   m = thermal softening exponent
 *   ε̇* = ε̇_p / ε̇_ref (normalized strain rate)
 *   T* = (T - T_room) / (T_melt - T_room) (homologous temperature)
 */
class JohnsonCookMaterial : public Material {
public:
    JohnsonCookMaterial(const MaterialProperties& props)
        : Material(MaterialType::Plastic, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;

        // Johnson-Cook parameters
        const Real A = props_.JC_A;
        const Real B = props_.JC_B;
        const Real n = props_.JC_n;
        const Real C = props_.JC_C;
        const Real m = props_.JC_m;
        const Real eps_dot_ref = props_.JC_eps_dot_ref;
        const Real T_melt = props_.JC_T_melt;
        const Real T_room = props_.JC_T_room;

        // Get accumulated plastic strain from history
        Real eps_p = state.history[0];

        // Compute trial elastic strain
        Real strain_e[6];
        for (int i = 0; i < 6; ++i) {
            strain_e[i] = state.strain[i] - state.history[i + 1];
        }

        // Compute trial stress (elastic predictor)
        Real stress_trial[6];
        elastic_stress(strain_e, E, nu, stress_trial);

        // Compute deviatoric trial stress
        const Real p_trial = (stress_trial[0] + stress_trial[1] + stress_trial[2]) / 3.0;
        Real s_trial[6];
        s_trial[0] = stress_trial[0] - p_trial;
        s_trial[1] = stress_trial[1] - p_trial;
        s_trial[2] = stress_trial[2] - p_trial;
        s_trial[3] = stress_trial[3];
        s_trial[4] = stress_trial[4];
        s_trial[5] = stress_trial[5];

        // Compute von Mises stress (trial)
        Real s_norm_sq = s_trial[0]*s_trial[0] + s_trial[1]*s_trial[1] + s_trial[2]*s_trial[2]
                       + 2.0*(s_trial[3]*s_trial[3] + s_trial[4]*s_trial[4] + s_trial[5]*s_trial[5]);
        Real sigma_vm_trial = Kokkos::sqrt(1.5 * s_norm_sq);

        // Compute Johnson-Cook yield stress
        // Strain hardening term: (A + B * ε_p^n)
        Real strain_hardening = A;
        if (eps_p > 1.0e-10 && B > 0.0) {
            strain_hardening += B * Kokkos::pow(eps_p, n);
        }

        // Strain rate term: (1 + C * ln(ε̇*))
        Real eps_dot = state.effective_strain_rate;
        Real eps_dot_star = Kokkos::fmax(eps_dot / eps_dot_ref, 1.0);  // Clamp to >= 1
        Real rate_factor = 1.0;
        if (C > 0.0 && eps_dot_star > 1.0) {
            rate_factor = 1.0 + C * Kokkos::log(eps_dot_star);
        }

        // Thermal softening term: (1 - T*^m)
        Real T = state.temperature;
        Real thermal_factor = 1.0;
        if (T > T_room && T < T_melt) {
            Real T_star = (T - T_room) / (T_melt - T_room);
            thermal_factor = 1.0 - Kokkos::pow(T_star, m);
        } else if (T >= T_melt) {
            thermal_factor = 0.0;  // Material has melted
        }

        // Combined yield stress
        Real sigma_y = strain_hardening * rate_factor * thermal_factor;

        // Check yield condition
        Real f_trial = sigma_vm_trial - sigma_y;

        if (f_trial <= 0.0) {
            // Elastic: use trial stress
            for (int i = 0; i < 6; ++i) {
                state.stress[i] = stress_trial[i];
            }
        } else {
            // Plastic: radial return mapping with Newton iteration
            // For Johnson-Cook, the hardening is nonlinear, so we need iteration

            Real delta_gamma = 0.0;
            Real eps_p_new = eps_p;

            // Newton-Raphson iteration for plastic multiplier
            for (int iter = 0; iter < 20; ++iter) {
                // Current yield stress with updated plastic strain
                Real sh = A;
                Real dsh_deps_p = 0.0;
                if (eps_p_new > 1.0e-10 && B > 0.0) {
                    sh += B * Kokkos::pow(eps_p_new, n);
                    dsh_deps_p = B * n * Kokkos::pow(eps_p_new, n - 1.0);
                }

                Real sigma_y_new = sh * rate_factor * thermal_factor;
                Real dsigma_y_deps_p = dsh_deps_p * rate_factor * thermal_factor;

                // Current von Mises stress
                Real sigma_vm_new = sigma_vm_trial - 3.0 * G * delta_gamma;

                // Yield function residual
                Real R = sigma_vm_new - sigma_y_new;

                if (Kokkos::fabs(R) < 1.0e-10 * sigma_y) {
                    break;  // Converged
                }

                // Derivative of residual w.r.t. delta_gamma
                // d(sigma_vm)/d(delta_gamma) = -3G
                // d(sigma_y)/d(delta_gamma) = dsigma_y/deps_p * deps_p/d(delta_gamma) = dsigma_y_deps_p * 1
                Real dR = -3.0 * G - dsigma_y_deps_p;

                // Newton update
                Real d_delta_gamma = -R / dR;
                delta_gamma += d_delta_gamma;
                eps_p_new = eps_p + delta_gamma;

                // Ensure positive values
                if (delta_gamma < 0.0) delta_gamma = 0.0;
                if (eps_p_new < eps_p) eps_p_new = eps_p;
            }

            // Update accumulated plastic strain
            state.history[0] = eps_p_new;

            // Update plastic strain tensor
            if (sigma_vm_trial > 1.0e-12) {
                Real factor = 1.5 * delta_gamma / sigma_vm_trial;
                state.history[1] += factor * s_trial[0];
                state.history[2] += factor * s_trial[1];
                state.history[3] += factor * s_trial[2];
                state.history[4] += factor * s_trial[3];
                state.history[5] += factor * s_trial[4];
                state.history[6] += factor * s_trial[5];
            }

            // Update stress
            Real scale = 1.0 - 3.0 * G * delta_gamma / sigma_vm_trial;
            state.stress[0] = scale * s_trial[0] + p_trial;
            state.stress[1] = scale * s_trial[1] + p_trial;
            state.stress[2] = scale * s_trial[2] + p_trial;
            state.stress[3] = scale * s_trial[3];
            state.stress[4] = scale * s_trial[4];
            state.stress[5] = scale * s_trial[5];
        }

        // Update state
        state.plastic_strain = state.history[0];
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state,
                          Real* C) const override {
        // For simplicity, use elastic tangent
        // A consistent tangent would require derivation for JC model
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real G = props_.G;

        const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const Real mu = G;
        const Real lambda_2mu = lambda + 2.0 * mu;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        C[0] = C[7] = C[14] = lambda_2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// Neo-Hookean Hyperelastic Material
// ============================================================================

/**
 * @brief Neo-Hookean hyperelastic material for large deformations
 *
 * Strain energy density:
 *   W = μ/2 * (I₁ - 3) + κ/2 * (J - 1)²
 *
 * Where:
 *   μ = shear modulus (G)
 *   κ = bulk modulus (K)
 *   I₁ = tr(B) = first invariant of left Cauchy-Green tensor B = F·F^T
 *   J = det(F) = volume ratio
 *
 * Cauchy stress:
 *   σ = μ/J * B_dev + κ*(J-1)*I
 *
 * where B_dev = B - (1/3)*tr(B)*I is the deviatoric part of B
 */
class NeoHookeanMaterial : public Material {
public:
    NeoHookeanMaterial(const MaterialProperties& props)
        : Material(MaterialType::Hyperelastic, props) {}

    KOKKOS_INLINE_FUNCTION
    void compute_stress(MaterialState& state) const override {
        const Real mu = props_.G;   // Shear modulus
        const Real kappa = props_.K; // Bulk modulus

        // Get deformation gradient F from state (row-major 3x3)
        const Real* F = state.F;

        // Compute determinant J = det(F)
        Real J = F[0]*(F[4]*F[8] - F[5]*F[7])
               - F[1]*(F[3]*F[8] - F[5]*F[6])
               + F[2]*(F[3]*F[7] - F[4]*F[6]);

        // Handle near-singular or inverted elements
        if (J < 1.0e-10) {
            // Return zero stress for severely compressed elements
            for (int i = 0; i < 6; ++i) state.stress[i] = 0.0;
            return;
        }

        // Compute left Cauchy-Green tensor B = F·F^T
        Real B[9];
        B[0] = F[0]*F[0] + F[1]*F[1] + F[2]*F[2];  // B_xx
        B[1] = F[0]*F[3] + F[1]*F[4] + F[2]*F[5];  // B_xy
        B[2] = F[0]*F[6] + F[1]*F[7] + F[2]*F[8];  // B_xz
        B[3] = B[1];                                // B_yx
        B[4] = F[3]*F[3] + F[4]*F[4] + F[5]*F[5];  // B_yy
        B[5] = F[3]*F[6] + F[4]*F[7] + F[5]*F[8];  // B_yz
        B[6] = B[2];                                // B_zx
        B[7] = B[5];                                // B_zy
        B[8] = F[6]*F[6] + F[7]*F[7] + F[8]*F[8];  // B_zz

        // First invariant I1 = tr(B)
        Real I1 = B[0] + B[4] + B[8];

        // Deviatoric part of B: B_dev = B - (1/3)*I1*I
        Real I1_third = I1 / 3.0;
        Real B_dev[9];
        B_dev[0] = B[0] - I1_third;
        B_dev[1] = B[1];
        B_dev[2] = B[2];
        B_dev[3] = B[3];
        B_dev[4] = B[4] - I1_third;
        B_dev[5] = B[5];
        B_dev[6] = B[6];
        B_dev[7] = B[7];
        B_dev[8] = B[8] - I1_third;

        // Cauchy stress: σ = (μ/J) * B_dev + κ*(J-1)*I
        Real mu_over_J = mu / J;
        Real pressure = kappa * (J - 1.0);

        // Store in Voigt notation: [σxx, σyy, σzz, τxy, τyz, τxz]
        state.stress[0] = mu_over_J * B_dev[0] + pressure;  // σxx
        state.stress[1] = mu_over_J * B_dev[4] + pressure;  // σyy
        state.stress[2] = mu_over_J * B_dev[8] + pressure;  // σzz
        state.stress[3] = mu_over_J * B_dev[1];             // τxy
        state.stress[4] = mu_over_J * B_dev[5];             // τyz
        state.stress[5] = mu_over_J * B_dev[2];             // τxz

        // Also compute strain from Green-Lagrange: E = 0.5*(F^T·F - I)
        // This is for compatibility with small-strain interface
        Real C[9];  // Right Cauchy-Green C = F^T·F
        C[0] = F[0]*F[0] + F[3]*F[3] + F[6]*F[6];
        C[1] = F[0]*F[1] + F[3]*F[4] + F[6]*F[7];
        C[2] = F[0]*F[2] + F[3]*F[5] + F[6]*F[8];
        C[3] = C[1];
        C[4] = F[1]*F[1] + F[4]*F[4] + F[7]*F[7];
        C[5] = F[1]*F[2] + F[4]*F[5] + F[7]*F[8];
        C[6] = C[2];
        C[7] = C[5];
        C[8] = F[2]*F[2] + F[5]*F[5] + F[8]*F[8];

        // Green-Lagrange strain E = 0.5*(C - I)
        state.strain[0] = 0.5 * (C[0] - 1.0);  // Exx
        state.strain[1] = 0.5 * (C[4] - 1.0);  // Eyy
        state.strain[2] = 0.5 * (C[8] - 1.0);  // Ezz
        state.strain[3] = C[1];                 // 2*Exy (engineering shear)
        state.strain[4] = C[5];                 // 2*Eyz
        state.strain[5] = C[2];                 // 2*Exz

        // Volumetric strain (logarithmic)
        state.vol_strain = Kokkos::log(J);
    }

    KOKKOS_INLINE_FUNCTION
    void tangent_stiffness(const MaterialState& state,
                          Real* C) const override {
        // Spatial tangent modulus for Neo-Hookean
        // For simplicity, use elastic approximation at small strains
        const Real E = props_.E;
        const Real nu = props_.nu;
        const Real mu = props_.G;

        const Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const Real lambda_2mu = lambda + 2.0 * mu;

        for (int i = 0; i < 36; ++i) C[i] = 0.0;

        C[0] = C[7] = C[14] = lambda_2mu;
        C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = lambda;
        C[21] = C[28] = C[35] = mu;
    }
};

// ============================================================================
// Material Factory
// ============================================================================

/**
 * @brief Factory for creating material objects
 */
class MaterialFactory {
public:
    static std::unique_ptr<Material> create(MaterialType type,
                                           const MaterialProperties& props) {
        switch (type) {
            case MaterialType::Elastic:
                return std::make_unique<ElasticMaterial>(props);
            case MaterialType::Plastic:
                return std::make_unique<VonMisesPlasticMaterial>(props);
            case MaterialType::Hyperelastic:
                return std::make_unique<NeoHookeanMaterial>(props);
            default:
                throw NotImplementedError("Material type not implemented");
        }
    }

    static std::string to_string(MaterialType type) {
        switch (type) {
            case MaterialType::Elastic: return "Elastic";
            case MaterialType::Plastic: return "Plastic";
            case MaterialType::Hyperelastic: return "Hyperelastic";
            case MaterialType::Viscoelastic: return "Viscoelastic";
            case MaterialType::Damage: return "Damage";
            case MaterialType::Composite: return "Composite";
            case MaterialType::Custom: return "Custom";
            default: return "Unknown";
        }
    }
};

} // namespace physics
} // namespace nxs
