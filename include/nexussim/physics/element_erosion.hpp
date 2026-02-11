#pragma once

/**
 * @file element_erosion.hpp
 * @brief Element erosion and failure models for crash/impact simulations
 *
 * This module provides:
 * - Multiple failure criteria (stress, strain, energy-based)
 * - Element state tracking (active, damaged, eroded)
 * - Element deletion mechanism
 * - Mass conservation during erosion
 *
 * Common failure criteria:
 * 1. Maximum principal stress/strain
 * 2. Effective plastic strain
 * 3. Triaxiality-dependent (Johnson-Cook damage)
 * 4. Cockcroft-Latham (tensile fracture)
 * 5. Total strain energy density
 *
 * Reference: LS-DYNA Theory Manual, Chapter 20
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <nexussim/physics/material.hpp>
#include <cmath>
#include <vector>
#include <string>

namespace nxs {
namespace physics {

// ============================================================================
// Element State
// ============================================================================

enum class ElementState : int {
    Active = 0,      ///< Normal active element
    Damaged = 1,     ///< Element is damaged but still active
    Eroded = 2,      ///< Element has been deleted/eroded
    Inactive = 3     ///< Element deactivated (e.g., for birth/death)
};

// ============================================================================
// Failure Criterion Types
// ============================================================================

enum class FailureCriterion {
    None,                    ///< No failure
    MaxPrincipalStress,      ///< Maximum principal stress
    MaxPrincipalStrain,      ///< Maximum principal strain
    EffectivePlasticStrain,  ///< Accumulated plastic strain
    VonMisesStress,          ///< Von Mises stress limit
    JohnsonCookDamage,       ///< Triaxiality-dependent (J-C damage)
    CockcroftLatham,         ///< Tensile fracture criterion
    TotalStrainEnergy,       ///< Strain energy density
    MinJacobian,             ///< Negative/small Jacobian (element distortion)
    Combined,                ///< Multiple criteria combined
    // Wave 2: Advanced failure models
    Hashin,                  ///< Hashin composite failure (4 modes)
    TsaiWu,                 ///< Tsai-Wu polynomial failure
    ChangChang,              ///< Chang-Chang laminate failure
    GTN,                     ///< Gurson-Tvergaard-Needleman ductile damage
    GISSMO,                  ///< Mesh-regularized damage (automotive)
    TabulatedEnvelope        ///< User-defined failure envelope
};

// ============================================================================
// Failure Parameters
// ============================================================================

/**
 * @brief Parameters for element failure/erosion
 */
struct FailureParameters {
    FailureCriterion criterion;

    // Stress/strain limits
    Real max_principal_stress;     ///< Maximum tensile principal stress
    Real min_principal_stress;     ///< Maximum compressive stress (negative)
    Real max_principal_strain;     ///< Maximum tensile principal strain
    Real max_shear_strain;         ///< Maximum engineering shear strain
    Real max_plastic_strain;       ///< Maximum effective plastic strain
    Real max_vonmises_stress;      ///< Von Mises stress limit

    // Johnson-Cook damage parameters
    // D = ∫ dε_p / ε_f(η, ε̇, T)
    // ε_f = [D1 + D2*exp(D3*η)] * [1 + D4*ln(ε̇*)] * [1 + D5*T*]
    Real JC_D1;                    ///< JC damage parameter D1
    Real JC_D2;                    ///< JC damage parameter D2
    Real JC_D3;                    ///< JC damage parameter D3
    Real JC_D4;                    ///< JC damage parameter D4 (strain rate)
    Real JC_D5;                    ///< JC damage parameter D5 (temperature)

    // Cockcroft-Latham parameter
    Real CL_W_crit;                ///< Critical Cockcroft-Latham integral

    // Energy-based
    Real max_strain_energy;        ///< Maximum strain energy density (J/m³)

    // Geometry-based
    Real min_jacobian;             ///< Minimum Jacobian ratio
    Real max_aspect_ratio;         ///< Maximum aspect ratio

    // Erosion options
    bool delete_on_failure;        ///< Delete element on failure
    bool redistribute_mass;        ///< Redistribute mass to nodes
    Real damage_threshold;         ///< Damage level to trigger erosion (0-1)

    FailureParameters()
        : criterion(FailureCriterion::None)
        , max_principal_stress(1.0e20)
        , min_principal_stress(-1.0e20)
        , max_principal_strain(1.0)
        , max_shear_strain(2.0)
        , max_plastic_strain(1.0)
        , max_vonmises_stress(1.0e20)
        , JC_D1(0.05)
        , JC_D2(3.44)
        , JC_D3(-2.12)
        , JC_D4(0.002)
        , JC_D5(0.61)
        , CL_W_crit(1.0e8)
        , max_strain_energy(1.0e10)
        , min_jacobian(0.01)
        , max_aspect_ratio(10.0)
        , delete_on_failure(true)
        , redistribute_mass(true)
        , damage_threshold(1.0)
    {}
};

// ============================================================================
// Failure Evaluation Functions (GPU-compatible)
// ============================================================================

/**
 * @brief Compute principal stresses from Cauchy stress tensor
 * @param sigma Stress tensor [6, Voigt: σxx, σyy, σzz, τxy, τyz, τxz]
 * @param principal Output: Principal stresses [σ1, σ2, σ3], σ1 >= σ2 >= σ3
 *
 * Uses analytical solution for eigenvalues of symmetric 3x3 matrix.
 * Reference: https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3×3_matrices
 */
KOKKOS_INLINE_FUNCTION
void compute_principal_stresses(const Real* sigma, Real* principal) {
    // Extract components
    const Real a11 = sigma[0], a22 = sigma[1], a33 = sigma[2];
    const Real a12 = sigma[3], a23 = sigma[4], a13 = sigma[5];

    // Invariants of stress tensor
    const Real I1 = a11 + a22 + a33;
    const Real I2 = a11*a22 + a22*a33 + a33*a11 - a12*a12 - a23*a23 - a13*a13;
    const Real I3 = a11*a22*a33 + 2.0*a12*a23*a13 - a11*a23*a23 - a22*a13*a13 - a33*a12*a12;

    // Mean value
    const Real p = I1 / 3.0;

    // Shifted invariants (for eigenvalue problem centered at p)
    // q = (1/6)*((a11-a22)² + (a22-a33)² + (a33-a11)² + 6*(a12² + a23² + a13²))
    const Real q = (I1*I1 - 3.0*I2) / 9.0;

    if (q < 1.0e-30) {
        // Nearly hydrostatic state
        principal[0] = principal[1] = principal[2] = p;
        return;
    }

    // r = (1/2) * det(A - p*I)
    // For shifted matrix: determinant formula
    const Real r = (I1*I1*I1 - 4.5*I1*I2 + 13.5*I3) / 27.0;

    // sqrt(q^3)
    const Real sqrt_q3 = Kokkos::sqrt(q*q*q);

    // cos(3θ) = r / sqrt(q^3), need to clamp to [-1, 1]
    Real cos_3theta = r / sqrt_q3;
    cos_3theta = Kokkos::fmax(-1.0, Kokkos::fmin(1.0, cos_3theta));

    const Real theta = Kokkos::acos(cos_3theta) / 3.0;
    const Real sqrt_q = Kokkos::sqrt(q);

    // Three eigenvalues
    principal[0] = p + 2.0 * sqrt_q * Kokkos::cos(theta);
    principal[1] = p + 2.0 * sqrt_q * Kokkos::cos(theta - 2.0*M_PI/3.0);
    principal[2] = p + 2.0 * sqrt_q * Kokkos::cos(theta + 2.0*M_PI/3.0);

    // Sort descending: σ1 >= σ2 >= σ3
    if (principal[0] < principal[1]) {
        Real tmp = principal[0]; principal[0] = principal[1]; principal[1] = tmp;
    }
    if (principal[1] < principal[2]) {
        Real tmp = principal[1]; principal[1] = principal[2]; principal[2] = tmp;
    }
    if (principal[0] < principal[1]) {
        Real tmp = principal[0]; principal[0] = principal[1]; principal[1] = tmp;
    }
}

/**
 * @brief Compute stress triaxiality
 * η = p / σ_vm where p = -σ_m (mean stress with compression positive)
 */
KOKKOS_INLINE_FUNCTION
Real compute_triaxiality(const Real* sigma) {
    const Real p = -(sigma[0] + sigma[1] + sigma[2]) / 3.0;  // Pressure
    const Real vm = Material::von_mises_stress(sigma);

    if (vm < 1.0e-20) return 0.0;
    return -p / vm;  // η = σ_m / σ_vm (tensile = positive)
}

/**
 * @brief Check maximum principal stress failure
 */
KOKKOS_INLINE_FUNCTION
bool check_max_principal_stress(const Real* sigma, const FailureParameters& params) {
    Real principal[3];
    compute_principal_stresses(sigma, principal);

    // Check tensile and compressive limits
    if (principal[0] > params.max_principal_stress) return true;
    if (principal[2] < params.min_principal_stress) return true;

    return false;
}

/**
 * @brief Check effective plastic strain failure
 */
KOKKOS_INLINE_FUNCTION
bool check_plastic_strain(Real eps_p, const FailureParameters& params) {
    return eps_p > params.max_plastic_strain;
}

/**
 * @brief Check von Mises stress failure
 */
KOKKOS_INLINE_FUNCTION
bool check_vonmises_stress(const Real* sigma, const FailureParameters& params) {
    Real vm = Material::von_mises_stress(sigma);
    return vm > params.max_vonmises_stress;
}

/**
 * @brief Compute Johnson-Cook failure strain
 * ε_f = [D1 + D2*exp(D3*η)] * [1 + D4*ln(ε̇*)] * [1 + D5*T*]
 */
KOKKOS_INLINE_FUNCTION
Real johnson_cook_failure_strain(Real triaxiality, Real strain_rate_ratio,
                                  Real temp_ratio, const FailureParameters& params) {
    // Triaxiality term
    Real eps_f = params.JC_D1 + params.JC_D2 * Kokkos::exp(params.JC_D3 * triaxiality);

    // Strain rate term (only for eps_dot* > 1)
    if (strain_rate_ratio > 1.0 && params.JC_D4 > 0.0) {
        eps_f *= (1.0 + params.JC_D4 * Kokkos::log(strain_rate_ratio));
    }

    // Temperature term
    if (temp_ratio > 0.0 && params.JC_D5 != 0.0) {
        eps_f *= (1.0 + params.JC_D5 * temp_ratio);
    }

    return Kokkos::fmax(eps_f, 0.01);  // Minimum failure strain
}

/**
 * @brief Update Johnson-Cook damage parameter
 * D += Δε_p / ε_f
 */
KOKKOS_INLINE_FUNCTION
Real update_jc_damage(Real current_damage, Real delta_eps_p,
                       const Real* sigma, Real eps_dot_ratio, Real T_ratio,
                       const FailureParameters& params) {
    Real eta = compute_triaxiality(sigma);
    Real eps_f = johnson_cook_failure_strain(eta, eps_dot_ratio, T_ratio, params);
    return current_damage + delta_eps_p / eps_f;
}

/**
 * @brief Compute Cockcroft-Latham integral increment
 * W = ∫ max(σ1, 0) dε_p
 */
KOKKOS_INLINE_FUNCTION
Real update_cockcroft_latham(Real current_W, const Real* sigma, Real delta_eps_p) {
    Real principal[3];
    compute_principal_stresses(sigma, principal);
    Real sigma1_positive = Kokkos::fmax(principal[0], 0.0);
    return current_W + sigma1_positive * delta_eps_p;
}

// ============================================================================
// Element Erosion Manager
// ============================================================================

/**
 * @brief Manages element erosion and failure tracking
 *
 * Usage:
 * 1. Initialize with number of elements
 * 2. Set failure parameters
 * 3. Call check_failure() each time step
 * 4. Call erode_elements() to delete failed elements
 * 5. Access element_active() to check if element should be processed
 */
class ElementErosionManager {
public:
    ElementErosionManager(std::size_t num_elements)
        : num_elements_(num_elements)
        , element_states_(num_elements, ElementState::Active)
        , damage_(num_elements, 0.0)
        , cockcroft_latham_(num_elements, 0.0)
        , eroded_count_(0)
        , total_eroded_mass_(0.0)
    {}

    // ========================================================================
    // Configuration
    // ========================================================================

    void set_failure_parameters(const FailureParameters& params) {
        params_ = params;
    }

    const FailureParameters& failure_parameters() const { return params_; }

    // ========================================================================
    // Element State Queries
    // ========================================================================

    bool element_active(std::size_t elem) const {
        return element_states_[elem] == ElementState::Active ||
               element_states_[elem] == ElementState::Damaged;
    }

    ElementState element_state(std::size_t elem) const {
        return element_states_[elem];
    }

    Real element_damage(std::size_t elem) const {
        return damage_[elem];
    }

    std::size_t eroded_count() const { return eroded_count_; }

    Real total_eroded_mass() const { return total_eroded_mass_; }

    // ========================================================================
    // Failure Checking
    // ========================================================================

    /**
     * @brief Check failure for a single element
     * @param elem Element index
     * @param state Material state (stress, plastic strain, etc.)
     * @param delta_eps_p Plastic strain increment this step
     * @return true if element has failed
     */
    bool check_failure(std::size_t elem, const MaterialState& state,
                       Real delta_eps_p = 0.0) {
        if (element_states_[elem] == ElementState::Eroded) {
            return true;  // Already eroded
        }

        bool failed = false;

        switch (params_.criterion) {
            case FailureCriterion::None:
                break;

            case FailureCriterion::MaxPrincipalStress:
                failed = check_max_principal_stress(state.stress, params_);
                break;

            case FailureCriterion::EffectivePlasticStrain:
                failed = check_plastic_strain(state.plastic_strain, params_);
                break;

            case FailureCriterion::VonMisesStress:
                failed = check_vonmises_stress(state.stress, params_);
                break;

            case FailureCriterion::JohnsonCookDamage: {
                // Update damage and check threshold
                Real eps_dot_ratio = state.effective_strain_rate / 1.0;  // Assume ref = 1
                Real T_ratio = (state.temperature - 293.15) / (1800.0 - 293.15);
                damage_[elem] = update_jc_damage(damage_[elem], delta_eps_p,
                                                  state.stress, eps_dot_ratio, T_ratio, params_);
                if (damage_[elem] >= params_.damage_threshold) {
                    failed = true;
                } else if (damage_[elem] > 0.0) {
                    element_states_[elem] = ElementState::Damaged;
                }
                break;
            }

            case FailureCriterion::CockcroftLatham:
                cockcroft_latham_[elem] = update_cockcroft_latham(
                    cockcroft_latham_[elem], state.stress, delta_eps_p);
                failed = cockcroft_latham_[elem] > params_.CL_W_crit;
                break;

            case FailureCriterion::Combined:
                // Check multiple criteria
                failed = check_max_principal_stress(state.stress, params_) ||
                        check_plastic_strain(state.plastic_strain, params_) ||
                        check_vonmises_stress(state.stress, params_);
                break;

            default:
                break;
        }

        if (failed && params_.delete_on_failure) {
            element_states_[elem] = ElementState::Eroded;
            eroded_count_++;
        }

        return failed;
    }

    /**
     * @brief Mark element as eroded and track mass
     * @param elem Element index
     * @param element_mass Mass of the element
     */
    void erode_element(std::size_t elem, Real element_mass) {
        if (element_states_[elem] != ElementState::Eroded) {
            element_states_[elem] = ElementState::Eroded;
            eroded_count_++;
            total_eroded_mass_ += element_mass;
        }
    }

    /**
     * @brief Get list of newly eroded elements (since last call)
     */
    std::vector<std::size_t> get_newly_eroded() {
        std::vector<std::size_t> newly_eroded;
        for (std::size_t i = 0; i < num_elements_; ++i) {
            if (element_states_[i] == ElementState::Eroded &&
                !was_eroded_[i]) {
                newly_eroded.push_back(i);
                was_eroded_[i] = true;
            }
        }
        return newly_eroded;
    }

    /**
     * @brief Reset erosion tracking for new step
     */
    void begin_step() {
        was_eroded_.resize(num_elements_, false);
    }

    // ========================================================================
    // Mass Redistribution
    // ========================================================================

    /**
     * @brief Redistribute mass from eroded element to adjacent nodes
     *
     * When an element is eroded, its mass should be distributed to
     * neighboring active elements or nodes to conserve total mass.
     *
     * @param elem Eroded element index
     * @param element_mass Mass of eroded element
     * @param node_indices Nodes of the eroded element
     * @param num_nodes Number of nodes
     * @param node_masses Node mass array (will be modified)
     */
    void redistribute_mass(std::size_t elem, Real element_mass,
                           const Index* node_indices, int num_nodes,
                           Real* node_masses) {
        if (!params_.redistribute_mass) return;
        if (element_states_[elem] != ElementState::Eroded) return;

        // Simple equal distribution to nodes
        Real mass_per_node = element_mass / num_nodes;
        for (int i = 0; i < num_nodes; ++i) {
            Index node = node_indices[i];
            node_masses[node] += mass_per_node;
        }
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /**
     * @brief Get erosion statistics
     */
    struct ErosionStats {
        std::size_t total_elements;
        std::size_t active_elements;
        std::size_t damaged_elements;
        std::size_t eroded_elements;
        Real max_damage;
        Real avg_damage;
        Real total_eroded_mass;
    };

    ErosionStats get_stats() const {
        ErosionStats stats;
        stats.total_elements = num_elements_;
        stats.active_elements = 0;
        stats.damaged_elements = 0;
        stats.eroded_elements = 0;
        stats.max_damage = 0.0;
        Real sum_damage = 0.0;

        for (std::size_t i = 0; i < num_elements_; ++i) {
            switch (element_states_[i]) {
                case ElementState::Active:
                    stats.active_elements++;
                    break;
                case ElementState::Damaged:
                    stats.damaged_elements++;
                    break;
                case ElementState::Eroded:
                    stats.eroded_elements++;
                    break;
                default:
                    break;
            }

            if (damage_[i] > stats.max_damage) {
                stats.max_damage = damage_[i];
            }
            sum_damage += damage_[i];
        }

        stats.avg_damage = sum_damage / num_elements_;
        stats.total_eroded_mass = total_eroded_mass_;

        return stats;
    }

    /**
     * @brief Print erosion summary
     */
    void print_summary() const {
        auto stats = get_stats();
        std::cout << "Element Erosion Summary:\n"
                  << "  Total elements:   " << stats.total_elements << "\n"
                  << "  Active elements:  " << stats.active_elements << "\n"
                  << "  Damaged elements: " << stats.damaged_elements << "\n"
                  << "  Eroded elements:  " << stats.eroded_elements
                  << " (" << 100.0 * stats.eroded_elements / stats.total_elements << "%)\n"
                  << "  Max damage:       " << stats.max_damage << "\n"
                  << "  Avg damage:       " << stats.avg_damage << "\n"
                  << "  Eroded mass:      " << stats.total_eroded_mass << " kg\n";
    }

private:
    std::size_t num_elements_;
    std::vector<ElementState> element_states_;
    std::vector<Real> damage_;
    std::vector<Real> cockcroft_latham_;
    std::vector<bool> was_eroded_;
    FailureParameters params_;
    std::size_t eroded_count_;
    Real total_eroded_mass_;
};

// ============================================================================
// GPU-Compatible Failure Check Kernel
// ============================================================================

/**
 * @brief Kokkos-compatible failure check for parallel execution
 */
struct FailureCheckFunctor {
    // Views for element data
    Kokkos::View<Real*[6]> stress_view;       // Stress tensor per element
    Kokkos::View<Real*> plastic_strain_view;  // Plastic strain per element
    Kokkos::View<int*> element_state_view;    // Element states
    Kokkos::View<Real*> damage_view;          // Damage parameter

    // Failure parameters
    FailureCriterion criterion;
    Real max_plastic_strain;
    Real max_principal_stress;
    Real damage_threshold;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int elem) const {
        // Skip already eroded elements
        if (element_state_view(elem) == static_cast<int>(ElementState::Eroded)) {
            return;
        }

        bool failed = false;

        // Get stress tensor
        Real sigma[6];
        for (int i = 0; i < 6; ++i) {
            sigma[i] = stress_view(elem, i);
        }

        switch (criterion) {
            case FailureCriterion::EffectivePlasticStrain:
                if (plastic_strain_view(elem) > max_plastic_strain) {
                    failed = true;
                }
                break;

            case FailureCriterion::MaxPrincipalStress: {
                Real principal[3];
                compute_principal_stresses(sigma, principal);
                if (principal[0] > max_principal_stress) {
                    failed = true;
                }
                break;
            }

            case FailureCriterion::JohnsonCookDamage:
                if (damage_view(elem) >= damage_threshold) {
                    failed = true;
                }
                break;

            default:
                break;
        }

        if (failed) {
            element_state_view(elem) = static_cast<int>(ElementState::Eroded);
        }
    }
};

} // namespace physics
} // namespace nxs
