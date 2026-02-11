#pragma once

/**
 * @file composite_layup.hpp
 * @brief Composite laminate layup with Classical Lamination Theory (CLT)
 *
 * Features:
 * - Ply definition: material, thickness, orientation angle
 * - ABD stiffness matrix computation (A=extensional, B=coupling, D=bending)
 * - Transformed ply stiffness (Q-bar) from fiber orientation
 * - Through-thickness stress recovery per ply
 * - Failure detection per ply (using existing failure models)
 * - Laminate effective properties (Ex, Ey, Gxy, nuxy)
 *
 * Theory:
 *   For a laminate of N plies, the constitutive relation is:
 *     {N}   [A  B] {ε⁰}
 *     {M} = [B  D] {κ}
 *   where N = force resultants, M = moment resultants,
 *   ε⁰ = midplane strains, κ = curvatures.
 *
 *   A_ij = Σ Q̄_ij^k (z_k - z_{k-1})
 *   B_ij = (1/2) Σ Q̄_ij^k (z_k² - z_{k-1}²)
 *   D_ij = (1/3) Σ Q̄_ij^k (z_k³ - z_{k-1}³)
 *
 * Reference:
 * - Jones, "Mechanics of Composite Materials", Chapter 4
 * - LS-DYNA *PART_COMPOSITE, *MAT_054 (Enhanced Composite Damage)
 */

#include <nexussim/core/types.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/physics/section.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace nxs {
namespace physics {

// ============================================================================
// Ply Definition
// ============================================================================

struct PlyDefinition {
    int material_id;         ///< Material ID (must be orthotropic)
    Real thickness;           ///< Ply thickness (m)
    Real angle;               ///< Fiber orientation angle (degrees)
    int failure_model_id;     ///< Failure model ID (-1 = none)

    // Ply material properties (filled in when resolving material ID)
    Real E1, E2;              ///< Longitudinal/transverse moduli
    Real G12;                 ///< In-plane shear modulus
    Real nu12;                ///< Major Poisson's ratio
    Real density;             ///< Ply density

    PlyDefinition()
        : material_id(0), thickness(0.001), angle(0.0), failure_model_id(-1)
        , E1(0.0), E2(0.0), G12(0.0), nu12(0.0), density(0.0) {}

    PlyDefinition(Real e1, Real e2, Real g12, Real nu, Real t, Real theta)
        : material_id(0), thickness(t), angle(theta), failure_model_id(-1)
        , E1(e1), E2(e2), G12(g12), nu12(nu), density(1600.0) {}
};

// ============================================================================
// Ply Stress/Strain State
// ============================================================================

struct PlyState {
    Real stress_global[3];    ///< σxx, σyy, τxy in global coords
    Real stress_local[3];     ///< σ11, σ22, τ12 in material coords
    Real strain_global[3];    ///< εxx, εyy, γxy in global coords
    Real strain_local[3];     ///< ε11, ε22, γ12 in material coords
    Real z_position;          ///< Through-thickness position
    bool failed;              ///< Ply failure flag
    int failure_mode;         ///< Failure mode code (0=none)

    PlyState() : z_position(0.0), failed(false), failure_mode(0) {
        for (int i = 0; i < 3; ++i) {
            stress_global[i] = stress_local[i] = 0.0;
            strain_global[i] = strain_local[i] = 0.0;
        }
    }
};

// ============================================================================
// Composite Laminate
// ============================================================================

class CompositeLaminate {
public:
    static constexpr int MAX_PLIES = 32;

    CompositeLaminate() : num_plies_(0), total_thickness_(0.0) {
        for (int i = 0; i < 9; ++i) {
            A_[i] = B_[i] = D_[i] = 0.0;
        }
    }

    // --- Ply Setup ---

    /**
     * @brief Add a ply to the laminate (bottom to top)
     */
    void add_ply(const PlyDefinition& ply) {
        if (num_plies_ >= MAX_PLIES) return;
        plies_[num_plies_] = ply;
        num_plies_++;
        total_thickness_ += ply.thickness;
    }

    /**
     * @brief Add symmetric plies: [θ1/θ2/.../θn]s
     * Plies are mirrored about the midplane
     */
    void add_symmetric(const std::vector<PlyDefinition>& half_stack) {
        // Bottom half
        for (const auto& p : half_stack) add_ply(p);
        // Mirror (reverse order)
        for (int i = static_cast<int>(half_stack.size()) - 1; i >= 0; --i) {
            add_ply(half_stack[i]);
        }
    }

    int num_plies() const { return num_plies_; }
    Real total_thickness() const { return total_thickness_; }
    const PlyDefinition& ply(int i) const { return plies_[i]; }

    // --- ABD Matrix Computation ---

    /**
     * @brief Compute ABD stiffness matrix from ply stack
     *
     * Must be called after all plies are added and before using
     * laminate_stiffness(), laminate_stress(), etc.
     */
    void compute_abd() {
        // Reset
        for (int i = 0; i < 9; ++i) { A_[i] = B_[i] = D_[i] = 0.0; }

        // Compute ply z-coordinates (z=0 at midplane)
        Real z_bot = -total_thickness_ / 2.0;
        for (int k = 0; k < num_plies_; ++k) {
            z_bottom_[k] = z_bot;
            z_bot += plies_[k].thickness;
            z_top_[k] = z_bot;
        }

        // Accumulate ABD
        for (int k = 0; k < num_plies_; ++k) {
            Real Qbar[9];
            compute_Qbar(plies_[k], Qbar);

            Real zb = z_bottom_[k];
            Real zt = z_top_[k];
            Real dz = zt - zb;
            Real dz2 = zt*zt - zb*zb;
            Real dz3 = zt*zt*zt - zb*zb*zb;

            for (int i = 0; i < 9; ++i) {
                A_[i] += Qbar[i] * dz;
                B_[i] += 0.5 * Qbar[i] * dz2;
                D_[i] += (1.0/3.0) * Qbar[i] * dz3;
            }
        }
    }

    // --- Laminate Stiffness Access ---

    const Real* A() const { return A_; }  ///< 3x3 extensional stiffness
    const Real* B() const { return B_; }  ///< 3x3 coupling stiffness
    const Real* D() const { return D_; }  ///< 3x3 bending stiffness

    /**
     * @brief Get full 6x6 ABD matrix
     * Layout: [A11 A12 A16 B11 B12 B16; ...]
     */
    void get_abd_matrix(Real* abd) const {
        for (int i = 0; i < 36; ++i) abd[i] = 0.0;

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                abd[i*6 + j] = A_[i*3 + j];       // Top-left: A
                abd[i*6 + (j+3)] = B_[i*3 + j];   // Top-right: B
                abd[(i+3)*6 + j] = B_[i*3 + j];   // Bottom-left: B
                abd[(i+3)*6 + (j+3)] = D_[i*3 + j]; // Bottom-right: D
            }
        }
    }

    // --- Effective Laminate Properties ---

    /**
     * @brief Compute effective laminate engineering properties
     *
     * For a symmetric laminate (B=0):
     *   Ex = (A11*A22 - A12²) / (A22*h)
     *   Ey = (A11*A22 - A12²) / (A11*h)
     *   Gxy = A66 / h
     *   nuxy = A12 / A22
     */
    struct EffectiveProperties {
        Real Ex, Ey, Gxy, nuxy, nuyx;
        Real density;  ///< Average density
    };

    EffectiveProperties effective_properties() const {
        EffectiveProperties ep;
        Real h = total_thickness_;
        if (h < 1.0e-30) {
            ep = {0,0,0,0,0,0};
            return ep;
        }

        Real det = A_[0]*A_[4] - A_[1]*A_[1];  // A11*A22 - A12²
        ep.Ex = (det > 0) ? det / (A_[4] * h) : 0.0;
        ep.Ey = (det > 0) ? det / (A_[0] * h) : 0.0;
        ep.Gxy = A_[8] / h;  // A66/h
        ep.nuxy = (A_[4] > 0) ? A_[1] / A_[4] : 0.0;
        ep.nuyx = (A_[0] > 0) ? A_[1] / A_[0] : 0.0;

        // Average density
        Real total_mass_per_area = 0.0;
        for (int k = 0; k < num_plies_; ++k) {
            total_mass_per_area += plies_[k].density * plies_[k].thickness;
        }
        ep.density = total_mass_per_area / h;

        return ep;
    }

    // --- Stress Recovery ---

    /**
     * @brief Compute ply stresses from midplane strains and curvatures
     *
     * @param eps0 Midplane strains [εxx⁰, εyy⁰, γxy⁰]
     * @param kappa Curvatures [κxx, κyy, κxy]
     * @param ply_states Output: stress/strain state per ply
     * @return Number of ply states written
     */
    int compute_ply_stresses(const Real* eps0, const Real* kappa,
                              PlyState* ply_states) const {
        for (int k = 0; k < num_plies_; ++k) {
            Real z_mid = (z_bottom_[k] + z_top_[k]) / 2.0;
            ply_states[k].z_position = z_mid;

            // Global strain at ply midpoint: ε = ε⁰ + z*κ
            for (int i = 0; i < 3; ++i) {
                ply_states[k].strain_global[i] = eps0[i] + z_mid * kappa[i];
            }

            // Global stress: σ = Q̄ * ε
            Real Qbar[9];
            compute_Qbar(plies_[k], Qbar);

            for (int i = 0; i < 3; ++i) {
                ply_states[k].stress_global[i] = 0.0;
                for (int j = 0; j < 3; ++j) {
                    ply_states[k].stress_global[i] += Qbar[i*3+j] *
                        ply_states[k].strain_global[j];
                }
            }

            // Transform to local (material) coordinates
            Real theta = plies_[k].angle * constants::pi<Real> / 180.0;
            transform_to_local(theta,
                               ply_states[k].stress_global,
                               ply_states[k].stress_local);
            transform_to_local(theta,
                               ply_states[k].strain_global,
                               ply_states[k].strain_local);
        }

        return num_plies_;
    }

    /**
     * @brief Compute force and moment resultants from midplane strains
     *
     * {N} = [A]{ε⁰} + [B]{κ}
     * {M} = [B]{ε⁰} + [D]{κ}
     */
    void compute_resultants(const Real* eps0, const Real* kappa,
                             Real* N_out, Real* M_out) const {
        for (int i = 0; i < 3; ++i) {
            N_out[i] = 0.0;
            M_out[i] = 0.0;
            for (int j = 0; j < 3; ++j) {
                N_out[i] += A_[i*3+j] * eps0[j] + B_[i*3+j] * kappa[j];
                M_out[i] += B_[i*3+j] * eps0[j] + D_[i*3+j] * kappa[j];
            }
        }
    }

    // --- Integration Point Setup ---

    /**
     * @brief Set up section integration points for this laminate
     */
    void setup_integration(SectionProperties& section,
                            int points_per_ply = 1) const {
        section.type = SectionType::ShellComposite;
        section.thickness = total_thickness_;
        section.integration.setup_composite(num_plies_, points_per_ply);
    }

    // --- Diagnostics ---

    bool is_symmetric() const {
        // Check if B ≈ 0 (symmetric laminate)
        Real max_b = 0.0;
        Real max_a = 0.0;
        for (int i = 0; i < 9; ++i) {
            if (std::fabs(B_[i]) > max_b) max_b = std::fabs(B_[i]);
            if (std::fabs(A_[i]) > max_a) max_a = std::fabs(A_[i]);
        }
        return (max_a > 0) ? (max_b / max_a < 1.0e-10) : true;
    }

    bool is_balanced() const {
        // Check A16 ≈ 0 and A26 ≈ 0 (balanced laminate)
        Real max_a = std::max(std::fabs(A_[0]), std::fabs(A_[4]));
        if (max_a < 1.0e-30) return true;
        return std::fabs(A_[2]) / max_a < 1.0e-10
            && std::fabs(A_[5]) / max_a < 1.0e-10;
    }

    void print_summary() const {
        std::cout << "Composite Laminate: " << num_plies_ << " plies, h="
                  << total_thickness_ * 1000.0 << " mm\n";
        std::cout << "  Stack: [";
        for (int i = 0; i < num_plies_; ++i) {
            if (i > 0) std::cout << "/";
            std::cout << plies_[i].angle;
        }
        std::cout << "]\n";
        std::cout << "  Symmetric: " << (is_symmetric() ? "yes" : "no")
                  << ", Balanced: " << (is_balanced() ? "yes" : "no") << "\n";

        auto ep = effective_properties();
        std::cout << "  Effective: Ex=" << ep.Ex/1e9 << " GPa, Ey=" << ep.Ey/1e9
                  << " GPa, Gxy=" << ep.Gxy/1e9 << " GPa, nuxy=" << ep.nuxy << "\n";
    }

private:
    /**
     * @brief Compute transformed reduced stiffness matrix Q̄ for a ply
     *
     * Q̄ = T⁻¹ Q T⁻ᵀ where T is the stress transformation matrix
     *
     * For angle θ:
     *   m = cos(θ), n = sin(θ)
     *   Q̄₁₁ = Q₁₁m⁴ + 2(Q₁₂+2Q₆₆)m²n² + Q₂₂n⁴
     *   ... etc (standard CLT transformation)
     */
    static void compute_Qbar(const PlyDefinition& ply, Real* Qbar) {
        Real E1 = ply.E1;
        Real E2 = ply.E2;
        Real G12 = ply.G12;
        Real nu12 = ply.nu12;
        Real nu21 = nu12 * E2 / E1;

        // On-axis reduced stiffness (Q matrix)
        Real denom = 1.0 - nu12 * nu21;
        Real Q11 = E1 / denom;
        Real Q22 = E2 / denom;
        Real Q12 = nu12 * E2 / denom;
        Real Q66 = G12;

        Real theta = ply.angle * constants::pi<Real> / 180.0;
        Real m = std::cos(theta);
        Real n = std::sin(theta);

        Real m2 = m*m;
        Real n2 = n*n;
        Real m4 = m2*m2;
        Real n4 = n2*n2;
        Real mn = m*n;
        Real m2n2 = m2*n2;

        // Transformed stiffness Q̄ (3x3 for plane stress)
        Qbar[0] = Q11*m4 + 2.0*(Q12 + 2.0*Q66)*m2n2 + Q22*n4;          // Q̄11
        Qbar[1] = (Q11 + Q22 - 4.0*Q66)*m2n2 + Q12*(m4 + n4);          // Q̄12
        Qbar[2] = (Q11 - Q12 - 2.0*Q66)*m*m2*n + (Q12 - Q22 + 2.0*Q66)*m*n*n2; // Q̄16
        Qbar[3] = Qbar[1];                                               // Q̄21
        Qbar[4] = Q11*n4 + 2.0*(Q12 + 2.0*Q66)*m2n2 + Q22*m4;          // Q̄22
        Qbar[5] = (Q11 - Q12 - 2.0*Q66)*mn*n2 + (Q12 - Q22 + 2.0*Q66)*m*m2*n; // Q̄26
        Qbar[6] = Qbar[2];                                               // Q̄61
        Qbar[7] = Qbar[5];                                               // Q̄62
        Qbar[8] = (Q11 + Q22 - 2.0*Q12 - 2.0*Q66)*m2n2 + Q66*(m4 + n4); // Q̄66
    }

    /**
     * @brief Transform stress/strain from global to local coordinates
     *
     * σ_local = T * σ_global where:
     * T = [ m²   n²   2mn  ]
     *     [ n²   m²  -2mn  ]
     *     [-mn   mn   m²-n²]
     *
     * For strain: use T with factor 2 for shear (since γ = 2*ε_shear)
     */
    static void transform_to_local(Real theta,
                                     const Real* global, Real* local) {
        Real m = std::cos(theta);
        Real n = std::sin(theta);
        Real m2 = m*m, n2 = n*n, mn = m*n;

        local[0] = m2*global[0] + n2*global[1] + 2.0*mn*global[2];
        local[1] = n2*global[0] + m2*global[1] - 2.0*mn*global[2];
        local[2] = -mn*global[0] + mn*global[1] + (m2-n2)*global[2];
    }

    // Ply data
    PlyDefinition plies_[MAX_PLIES];
    int num_plies_;
    Real total_thickness_;

    // Ply z-coordinates
    Real z_bottom_[MAX_PLIES];
    Real z_top_[MAX_PLIES];

    // ABD stiffness matrices (3x3 each, row-major)
    Real A_[9];   ///< Extensional stiffness
    Real B_[9];   ///< Bending-extension coupling
    Real D_[9];   ///< Bending stiffness
};

// ============================================================================
// Standard Layup Presets
// ============================================================================

namespace layup_presets {

/**
 * @brief Create quasi-isotropic layup [0/±45/90]s
 */
inline CompositeLaminate quasi_isotropic(Real E1, Real E2, Real G12,
                                          Real nu12, Real ply_t) {
    CompositeLaminate lam;
    PlyDefinition p0(E1, E2, G12, nu12, ply_t, 0.0);
    PlyDefinition p45(E1, E2, G12, nu12, ply_t, 45.0);
    PlyDefinition pm45(E1, E2, G12, nu12, ply_t, -45.0);
    PlyDefinition p90(E1, E2, G12, nu12, ply_t, 90.0);

    lam.add_symmetric({p0, p45, pm45, p90});
    return lam;
}

/**
 * @brief Create cross-ply layup [0/90]ns (n repeats, symmetric)
 */
inline CompositeLaminate cross_ply(Real E1, Real E2, Real G12,
                                    Real nu12, Real ply_t, int repeats = 1) {
    CompositeLaminate lam;
    PlyDefinition p0(E1, E2, G12, nu12, ply_t, 0.0);
    PlyDefinition p90(E1, E2, G12, nu12, ply_t, 90.0);

    std::vector<PlyDefinition> half;
    for (int i = 0; i < repeats; ++i) {
        half.push_back(p0);
        half.push_back(p90);
    }
    lam.add_symmetric(half);
    return lam;
}

/**
 * @brief Create unidirectional layup [0]n
 */
inline CompositeLaminate unidirectional(Real E1, Real E2, Real G12,
                                         Real nu12, Real ply_t, int num_plies) {
    CompositeLaminate lam;
    PlyDefinition p0(E1, E2, G12, nu12, ply_t, 0.0);
    for (int i = 0; i < num_plies; ++i) lam.add_ply(p0);
    return lam;
}

/**
 * @brief Create angle-ply layup [±θ]ns
 */
inline CompositeLaminate angle_ply(Real E1, Real E2, Real G12,
                                    Real nu12, Real ply_t, Real angle,
                                    int repeats = 1) {
    CompositeLaminate lam;
    PlyDefinition pp(E1, E2, G12, nu12, ply_t, angle);
    PlyDefinition pm(E1, E2, G12, nu12, ply_t, -angle);

    std::vector<PlyDefinition> half;
    for (int i = 0; i < repeats; ++i) {
        half.push_back(pp);
        half.push_back(pm);
    }
    lam.add_symmetric(half);
    return lam;
}

} // namespace layup_presets

} // namespace physics
} // namespace nxs
