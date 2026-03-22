#pragma once

/**
 * @file multifluid_wave34.hpp
 * @brief Wave 34b: Multi-fluid Dynamics — VOF tracking, pressure equilibrium,
 *        MUSCL reconstruction, multi-fluid BCs, sub-material EOS, FVM-to-FEM transfer
 *
 * Sub-modules:
 * - 34b-1: MultiFluidManager      — N-fluid VOF tracking with volume fraction enforcement
 * - 34b-2: PressureEquilibrium    — Newton solver for common pressure in mixed cells
 * - 34b-3: MultiFluidMUSCL        — MUSCL reconstruction for multi-material flows
 * - 34b-4: MultiFluidEBCS         — Eulerian BCs for multi-fluid systems
 * - 34b-5: SubMaterialLaw         — Per-fluid EOS (Ideal Gas, Stiffened Gas, JWL)
 * - 34b-6: MultiFluidFVM2FEM      — FVM-to-FEM data transfer for coupled simulations
 *
 * References:
 * - Saurel & Abgrall (1999) "A multiphase Godunov method for compressible multifluid"
 * - Allaire, Clerc, Kokh (2002) "A five-equation model for diffuse interface"
 * - Shyue (1998) "An efficient shock-capturing algorithm for compressible multicomponent problems"
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>

namespace nxs {
namespace fem {

using Real = nxs::Real;

// ============================================================================
// Multi-Fluid Data Structures
// ============================================================================

/// Maximum number of fluids
static constexpr int MAX_FLUIDS = 8;

/// Per-cell multi-fluid state
struct FluidState {
    Real alpha[MAX_FLUIDS];  ///< Volume fractions (sum = 1)
    Real rho[MAX_FLUIDS];    ///< Partial densities (per-fluid)
    Real p[MAX_FLUIDS];      ///< Partial pressures
    Real e[MAX_FLUIDS];      ///< Specific internal energy (per-fluid)
    Real u, v, w;            ///< Mixture velocity
    int nfluids;             ///< Number of active fluids

    FluidState() : u(0), v(0), w(0), nfluids(1) {
        for (int i = 0; i < MAX_FLUIDS; ++i) {
            alpha[i] = 0.0;
            rho[i] = 0.0;
            p[i] = 0.0;
            e[i] = 0.0;
        }
        alpha[0] = 1.0;
    }

    /// Mixture density: sum(alpha_k * rho_k)
    KOKKOS_INLINE_FUNCTION
    Real mixture_density() const {
        Real rho_mix = 0.0;
        for (int k = 0; k < nfluids; ++k) {
            rho_mix += alpha[k] * rho[k];
        }
        return rho_mix;
    }

    /// Mixture pressure: sum(alpha_k * p_k)
    KOKKOS_INLINE_FUNCTION
    Real mixture_pressure() const {
        Real p_mix = 0.0;
        for (int k = 0; k < nfluids; ++k) {
            p_mix += alpha[k] * p[k];
        }
        return p_mix;
    }

    /// Mixture internal energy: sum(alpha_k * rho_k * e_k) / rho_mix
    KOKKOS_INLINE_FUNCTION
    Real mixture_internal_energy() const {
        Real rho_mix = mixture_density();
        if (rho_mix < 1.0e-30) return 0.0;
        Real rho_e = 0.0;
        for (int k = 0; k < nfluids; ++k) {
            rho_e += alpha[k] * rho[k] * e[k];
        }
        return rho_e / rho_mix;
    }

    /// Total volume fraction sum
    KOKKOS_INLINE_FUNCTION
    Real alpha_sum() const {
        Real s = 0.0;
        for (int k = 0; k < nfluids; ++k) s += alpha[k];
        return s;
    }
};

/// EOS type for sub-material law
enum class SubEOSType {
    IdealGas,       ///< P = (gamma-1)*rho*e
    StiffenedGas,   ///< P = (gamma-1)*rho*e - gamma*p_inf
    JWL             ///< Jones-Wilkins-Lee explosive EOS
};

/// EOS parameters for a single fluid
struct SubEOSParams {
    SubEOSType type;
    Real gamma;      ///< Ratio of specific heats
    Real p_inf;      ///< Stiffened gas reference pressure
    // JWL parameters
    Real A_jwl, B_jwl, R1, R2, omega;
    Real rho0;       ///< Reference density for JWL

    SubEOSParams() : type(SubEOSType::IdealGas), gamma(1.4), p_inf(0.0),
                     A_jwl(0), B_jwl(0), R1(4.4), R2(1.2), omega(0.25),
                     rho0(1630.0) {}
};

/// Multi-fluid BC type
enum class MultiFluidBCType {
    Inlet,          ///< Prescribed alpha, rho, velocity
    Outlet,         ///< Zero-gradient extrapolation
    NRF,            ///< Non-reflecting far-field
    Wall            ///< Reflecting wall
};

/// Multi-fluid BC data
struct MultiFluidBCData {
    MultiFluidBCType type;
    int face;               ///< 0=-x,1=+x,2=-y,3=+y,4=-z,5=+z
    FluidState inlet_state; ///< For inlet BC
    Real p_inf;             ///< Far-field pressure for NRF
    Real rho_inf;           ///< Far-field density for NRF
    Real c_inf;             ///< Far-field sound speed for NRF

    MultiFluidBCData() : type(MultiFluidBCType::Outlet), face(0),
                         p_inf(1.0e5), rho_inf(1.0), c_inf(340.0) {}
};

/// FEM node for FVM-to-FEM transfer
struct FEMNode {
    Real x, y, z;       ///< Position
    Real fx, fy, fz;    ///< Force components
    Real volume;        ///< Associated volume

    FEMNode() : x(0), y(0), z(0), fx(0), fy(0), fz(0), volume(0) {}
};

// ============================================================================
// 34b-5: SubMaterialLaw — Per-fluid EOS
// ============================================================================

/**
 * @brief Equation of state evaluation for individual fluid materials.
 *
 * Supports:
 * - Ideal Gas:       P = (gamma - 1) * rho * e
 * - Stiffened Gas:   P = (gamma - 1) * rho * e - gamma * p_inf
 * - JWL:            P = A*(1 - omega/(R1*eta))*exp(-R1*eta) +
 *                       B*(1 - omega/(R2*eta))*exp(-R2*eta) + omega*rho*e
 *   where eta = rho0/rho
 */
class SubMaterialLaw {
public:
    /**
     * @brief Compute pressure from EOS.
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_sub_pressure(const SubEOSParams& eos, Real rho, Real e) {
        Real rho_safe = (rho > 1.0e-30) ? rho : 1.0e-30;
        Real e_safe = (e > 0.0) ? e : 0.0;

        switch (eos.type) {
        case SubEOSType::IdealGas:
            return (eos.gamma - 1.0) * rho_safe * e_safe;

        case SubEOSType::StiffenedGas: {
            Real p = (eos.gamma - 1.0) * rho_safe * e_safe - eos.gamma * eos.p_inf;
            return (p > 0.0) ? p : 0.0;
        }

        case SubEOSType::JWL: {
            Real eta = eos.rho0 / rho_safe;
            Real p = eos.A_jwl * (1.0 - eos.omega / (eos.R1 * eta)) * Kokkos::exp(-eos.R1 * eta)
                   + eos.B_jwl * (1.0 - eos.omega / (eos.R2 * eta)) * Kokkos::exp(-eos.R2 * eta)
                   + eos.omega * rho_safe * e_safe;
            return (p > 0.0) ? p : 0.0;
        }

        default:
            return (eos.gamma - 1.0) * rho_safe * e_safe;
        }
    }

    /**
     * @brief Compute sound speed for a sub-material.
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_sub_sound_speed(const SubEOSParams& eos, Real rho, Real p) {
        Real rho_safe = (rho > 1.0e-30) ? rho : 1.0e-30;
        Real p_eff = p;
        if (eos.type == SubEOSType::StiffenedGas) {
            p_eff = p + eos.p_inf;
        }
        if (p_eff < 0.0) p_eff = 0.0;
        return Kokkos::sqrt(eos.gamma * p_eff / rho_safe);
    }

    /**
     * @brief Compute internal energy from pressure and density.
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_sub_energy(const SubEOSParams& eos, Real rho, Real p) {
        Real rho_safe = (rho > 1.0e-30) ? rho : 1.0e-30;
        Real gm1 = eos.gamma - 1.0;
        if (Kokkos::fabs(gm1) < 1.0e-15) gm1 = 1.0e-15;

        switch (eos.type) {
        case SubEOSType::IdealGas:
            return p / (gm1 * rho_safe);

        case SubEOSType::StiffenedGas:
            return (p + eos.gamma * eos.p_inf) / (gm1 * rho_safe);

        case SubEOSType::JWL: {
            // Approximate: ignore exponential terms for inverse
            Real omega = eos.omega;
            if (Kokkos::fabs(omega) < 1.0e-15) omega = 1.0e-15;
            Real eta = eos.rho0 / rho_safe;
            Real p_ref = eos.A_jwl * (1.0 - omega / (eos.R1 * eta)) * Kokkos::exp(-eos.R1 * eta)
                       + eos.B_jwl * (1.0 - omega / (eos.R2 * eta)) * Kokkos::exp(-eos.R2 * eta);
            return (p - p_ref) / (omega * rho_safe);
        }

        default:
            return p / (gm1 * rho_safe);
        }
    }

    /**
     * @brief Compute dP/drho at constant e (for sound speed via dP/drho|_s).
     */
    KOKKOS_INLINE_FUNCTION
    static Real compute_dpdrho(const SubEOSParams& eos, Real rho, Real e) {
        (void)rho;

        switch (eos.type) {
        case SubEOSType::IdealGas:
            return (eos.gamma - 1.0) * e;

        case SubEOSType::StiffenedGas:
            return (eos.gamma - 1.0) * e;

        default:
            return (eos.gamma - 1.0) * e;
        }
    }
};

// ============================================================================
// 34b-1: MultiFluidManager — N-fluid VOF tracking
// ============================================================================

/**
 * @brief Manages multi-fluid volume fractions and mixture properties.
 *
 * Tracks up to MAX_FLUIDS fluids using Volume-of-Fluid (VOF) approach.
 * Enforces sum(alpha_k) = 1 constraint via renormalization.
 * Each fluid has its own EOS parameters.
 */
class MultiFluidManager {
public:
    int nfluids;                            ///< Number of active fluids
    SubEOSParams eos_params[MAX_FLUIDS];    ///< EOS per fluid

    MultiFluidManager() : nfluids(1) {
        eos_params[0].type = SubEOSType::IdealGas;
        eos_params[0].gamma = 1.4;
    }

    explicit MultiFluidManager(int nf) : nfluids(nf) {
        if (nfluids > MAX_FLUIDS) nfluids = MAX_FLUIDS;
        if (nfluids < 1) nfluids = 1;
    }

    /**
     * @brief Initialize a fluid state with given volume fractions and densities.
     */
    void init_state(FluidState& fs, const Real* alphas, const Real* rhos,
                    const Real* energies, Real u_val, Real v_val, Real w_val) const {
        fs.nfluids = nfluids;
        fs.u = u_val;
        fs.v = v_val;
        fs.w = w_val;
        for (int k = 0; k < nfluids; ++k) {
            fs.alpha[k] = alphas[k];
            fs.rho[k] = rhos[k];
            fs.e[k] = energies[k];
            fs.p[k] = SubMaterialLaw::compute_sub_pressure(eos_params[k], rhos[k], energies[k]);
        }
        for (int k = nfluids; k < MAX_FLUIDS; ++k) {
            fs.alpha[k] = 0.0;
            fs.rho[k] = 0.0;
            fs.e[k] = 0.0;
            fs.p[k] = 0.0;
        }
        enforce_alpha_sum(fs);
    }

    /**
     * @brief Enforce sum(alpha) = 1 via renormalization.
     */
    void enforce_alpha_sum(FluidState& fs) const {
        Real sum = 0.0;
        for (int k = 0; k < fs.nfluids; ++k) {
            if (fs.alpha[k] < 0.0) fs.alpha[k] = 0.0;
            sum += fs.alpha[k];
        }
        if (sum > 1.0e-30) {
            Real inv = 1.0 / sum;
            for (int k = 0; k < fs.nfluids; ++k) {
                fs.alpha[k] *= inv;
            }
        } else {
            // Default: all volume to fluid 0
            fs.alpha[0] = 1.0;
            for (int k = 1; k < fs.nfluids; ++k) fs.alpha[k] = 0.0;
        }
    }

    /**
     * @brief Update pressures from current densities and energies.
     */
    void update_pressures(FluidState& fs) const {
        for (int k = 0; k < fs.nfluids; ++k) {
            fs.p[k] = SubMaterialLaw::compute_sub_pressure(eos_params[k], fs.rho[k], fs.e[k]);
        }
    }

    /**
     * @brief Check if a cell is a mixed cell (multiple fluids present).
     */
    KOKKOS_INLINE_FUNCTION
    static bool is_mixed_cell(const FluidState& fs, Real threshold = 1.0e-6) {
        int count = 0;
        for (int k = 0; k < fs.nfluids; ++k) {
            if (fs.alpha[k] > threshold) count++;
        }
        return count > 1;
    }

    /**
     * @brief Compute mixture sound speed (Wood's formula).
     *
     * 1/(rho_mix * c_mix^2) = sum_k( alpha_k / (rho_k * c_k^2) )
     */
    Real mixture_sound_speed(const FluidState& fs) const {
        Real sum_inv_rhoc2 = 0.0;
        Real rho_mix = fs.mixture_density();
        if (rho_mix < 1.0e-30) return 0.0;

        for (int k = 0; k < fs.nfluids; ++k) {
            if (fs.alpha[k] > 1.0e-15) {
                Real ck = SubMaterialLaw::compute_sub_sound_speed(eos_params[k], fs.rho[k], fs.p[k]);
                Real rhoc2 = fs.rho[k] * ck * ck;
                if (rhoc2 > 1.0e-30) {
                    sum_inv_rhoc2 += fs.alpha[k] / rhoc2;
                }
            }
        }

        if (sum_inv_rhoc2 < 1.0e-30) return 0.0;
        Real c2 = 1.0 / (rho_mix * sum_inv_rhoc2);
        return Kokkos::sqrt(Kokkos::fabs(c2));
    }

    /**
     * @brief Advect volume fractions (simple first-order upwind).
     */
    void advect_volume_fractions(FluidState& fs, Real flux_alpha[MAX_FLUIDS],
                                 Real dt, Real vol) const {
        if (vol < 1.0e-30) return;
        for (int k = 0; k < fs.nfluids; ++k) {
            fs.alpha[k] += dt * flux_alpha[k] / vol;
        }
        enforce_alpha_sum(fs);
    }
};

// ============================================================================
// 34b-2: PressureEquilibrium — Newton solver for common pressure
// ============================================================================

/**
 * @brief Iterative Newton solver for pressure equilibrium in mixed cells.
 *
 * In a mixed cell, all fluids must reach the same pressure p*.
 * The constraint is:
 *   sum_k alpha_k(p*) = 1
 *
 * where alpha_k(p*) is determined from the sub-material EOS via
 * isentropic compression/expansion of each fluid.
 *
 * Newton iteration:
 *   p_{n+1} = p_n - f(p_n) / f'(p_n)
 *   f(p)    = sum_k alpha_k(p) - 1
 *   f'(p)   = sum_k dalpha_k/dp
 */
class PressureEquilibrium {
public:
    static constexpr Real tol = 1.0e-10;
    static constexpr int default_max_iter = 20;

    /**
     * @brief Solve for common equilibrium pressure.
     *
     * Adjusts volume fractions so all fluids have the same pressure.
     * Uses Newton iteration on the constraint sum(alpha) = 1.
     *
     * @param fs         [in/out] Fluid state (alpha, rho, e, p updated)
     * @param eos_params EOS parameters per fluid
     * @param p_guess    Initial pressure guess
     * @param max_iter   Maximum Newton iterations
     * @return Converged pressure
     */
    static Real solve_equilibrium(FluidState& fs, const SubEOSParams eos_params[],
                                  Real p_guess, int max_iter = default_max_iter) {
        int nf = fs.nfluids;
        if (nf <= 1) {
            // Single fluid: pressure is already determined
            fs.p[0] = SubMaterialLaw::compute_sub_pressure(eos_params[0], fs.rho[0], fs.e[0]);
            return fs.p[0];
        }

        // Store initial mass per fluid: m_k = alpha_k * rho_k
        Real mass[MAX_FLUIDS];
        for (int k = 0; k < nf; ++k) {
            mass[k] = fs.alpha[k] * fs.rho[k];
        }

        Real p_star = p_guess;
        if (p_star < 1.0e-10) p_star = 1.0e-10;

        for (int iter = 0; iter < max_iter; ++iter) {
            Real f_val = 0.0;
            Real f_prime = 0.0;

            for (int k = 0; k < nf; ++k) {
                if (mass[k] < 1.0e-30) continue;

                // Compute density from EOS inverse: rho_k(p_star)
                // For ideal gas: e_k = p / ((gamma-1)*rho)  =>  rho = p / ((gamma-1)*e_k)
                // But we have fixed mass, so alpha_k = mass_k / rho_k
                // Use isentropic relation: p/rho^gamma = const
                Real gm1 = eos_params[k].gamma - 1.0;
                if (Kokkos::fabs(gm1) < 1.0e-15) gm1 = 1.0e-15;

                // Compute specific internal energy for this pressure
                Real e_k = SubMaterialLaw::compute_sub_energy(eos_params[k], fs.rho[k], p_star);
                if (e_k < 1.0e-30) e_k = 1.0e-30;

                // Density from energy and pressure
                Real rho_k = p_star / (gm1 * e_k);
                if (eos_params[k].type == SubEOSType::StiffenedGas) {
                    rho_k = (p_star + eos_params[k].gamma * eos_params[k].p_inf) / (gm1 * e_k);
                }
                if (rho_k < 1.0e-30) rho_k = 1.0e-30;

                Real alpha_k = mass[k] / rho_k;
                f_val += alpha_k;

                // Derivative: dalpha/dp = -mass_k / rho_k^2 * drho/dp
                // For ideal gas: drho/dp = 1/(gamma * e_k) (approx)
                Real c2 = eos_params[k].gamma * p_star / rho_k;
                if (c2 < 1.0e-30) c2 = 1.0e-30;
                Real drho_dp = rho_k / c2;  // = 1/c^2
                Real dalpha_dp = -mass[k] / (rho_k * rho_k) * drho_dp;
                f_prime += dalpha_dp;
            }

            f_val -= 1.0;  // f(p) = sum(alpha) - 1

            if (Kokkos::fabs(f_val) < tol) {
                // Converged — update state
                update_state_at_pressure(fs, eos_params, mass, p_star);
                return p_star;
            }

            if (Kokkos::fabs(f_prime) < 1.0e-30) break;

            Real dp = -f_val / f_prime;
            // Limit step to avoid negative pressure
            if (p_star + dp < 0.1 * p_star) dp = -0.9 * p_star;
            p_star += dp;
            if (p_star < 1.0e-10) p_star = 1.0e-10;
        }

        // Update state with best pressure found
        update_state_at_pressure(fs, eos_params, mass, p_star);
        return p_star;
    }

private:
    static void update_state_at_pressure(FluidState& fs, const SubEOSParams eos_params[],
                                         const Real mass[], Real p_star) {
        int nf = fs.nfluids;
        Real alpha_sum = 0.0;

        for (int k = 0; k < nf; ++k) {
            if (mass[k] < 1.0e-30) {
                fs.alpha[k] = 0.0;
                continue;
            }

            Real gm1 = eos_params[k].gamma - 1.0;
            if (Kokkos::fabs(gm1) < 1.0e-15) gm1 = 1.0e-15;

            Real e_k = SubMaterialLaw::compute_sub_energy(eos_params[k], fs.rho[k], p_star);
            if (e_k < 1.0e-30) e_k = 1.0e-30;

            Real rho_k = p_star / (gm1 * e_k);
            if (eos_params[k].type == SubEOSType::StiffenedGas) {
                rho_k = (p_star + eos_params[k].gamma * eos_params[k].p_inf) / (gm1 * e_k);
            }
            if (rho_k < 1.0e-30) rho_k = 1.0e-30;

            fs.rho[k] = rho_k;
            fs.e[k] = e_k;
            fs.p[k] = p_star;
            fs.alpha[k] = mass[k] / rho_k;
            alpha_sum += fs.alpha[k];
        }

        // Renormalize
        if (alpha_sum > 1.0e-30) {
            Real inv = 1.0 / alpha_sum;
            for (int k = 0; k < nf; ++k) {
                fs.alpha[k] *= inv;
            }
        }
    }
};

// ============================================================================
// 34b-3: MultiFluidMUSCL — MUSCL reconstruction for multi-material
// ============================================================================

/**
 * @brief MUSCL (Monotone Upstream-centered Schemes for Conservation Laws)
 * reconstruction adapted for multi-fluid flows.
 *
 * Performs per-fluid piecewise-linear reconstruction with interface-aware
 * limiting. Uses MinMod limiter to ensure monotonicity near material interfaces.
 */
class MultiFluidMUSCL {
public:
    /**
     * @brief MinMod slope limiter.
     */
    KOKKOS_INLINE_FUNCTION
    static Real minmod(Real a, Real b) {
        if (a * b <= 0.0) return 0.0;
        if (Kokkos::fabs(a) < Kokkos::fabs(b)) return a;
        return b;
    }

    /**
     * @brief Superbee slope limiter (more aggressive, less diffusive).
     */
    KOKKOS_INLINE_FUNCTION
    static Real superbee(Real a, Real b) {
        if (a * b <= 0.0) return 0.0;
        Real s1 = minmod(a, 2.0 * b);
        Real s2 = minmod(2.0 * a, b);
        if (Kokkos::fabs(s1) > Kokkos::fabs(s2)) return s1;
        return s2;
    }

    /**
     * @brief Reconstruct left and right states at a face for a single fluid.
     *
     * Given three consecutive cell values (left, center, right), compute
     * the reconstructed values at the center-right face.
     *
     * @param val_left    Value in cell i-1
     * @param val_center  Value in cell i
     * @param val_right   Value in cell i+1
     * @param face_left   [out] Reconstructed value at face from left side
     * @param face_right  [out] Reconstructed value at face from right side
     */
    KOKKOS_INLINE_FUNCTION
    static void reconstruct_1d(Real val_left, Real val_center, Real val_right,
                               Real& face_left, Real& face_right) {
        Real slope_l = val_center - val_left;
        Real slope_r = val_right - val_center;
        Real slope = minmod(slope_l, slope_r);
        face_left = val_center + 0.5 * slope;
        face_right = val_right - 0.5 * minmod(slope_r, val_right - val_center);
    }

    /**
     * @brief MUSCL reconstruction for multi-fluid state.
     *
     * @param fs_left     Fluid state in cell i-1
     * @param fs_center   Fluid state in cell i
     * @param fs_right    Fluid state in cell i+1
     * @param fluid_idx   Fluid index to reconstruct
     * @param alpha_face_L [out] Reconstructed alpha at face (left side)
     * @param rho_face_L   [out] Reconstructed rho at face (left side)
     * @param p_face_L     [out] Reconstructed p at face (left side)
     * @param alpha_face_R [out] Reconstructed alpha at face (right side)
     * @param rho_face_R   [out] Reconstructed rho at face (right side)
     * @param p_face_R     [out] Reconstructed p at face (right side)
     */
    static void reconstruct(const FluidState& fs_left,
                            const FluidState& fs_center,
                            const FluidState& fs_right,
                            int fluid_idx,
                            Real& alpha_face_L, Real& rho_face_L, Real& p_face_L,
                            Real& alpha_face_R, Real& rho_face_R, Real& p_face_R) {
        int k = fluid_idx;

        // Reconstruct alpha
        reconstruct_1d(fs_left.alpha[k], fs_center.alpha[k], fs_right.alpha[k],
                       alpha_face_L, alpha_face_R);

        // Clamp alpha to [0, 1]
        alpha_face_L = Kokkos::fmin(Kokkos::fmax(alpha_face_L, 0.0), 1.0);
        alpha_face_R = Kokkos::fmin(Kokkos::fmax(alpha_face_R, 0.0), 1.0);

        // Reconstruct density
        reconstruct_1d(fs_left.rho[k], fs_center.rho[k], fs_right.rho[k],
                       rho_face_L, rho_face_R);

        // Ensure positive density
        if (rho_face_L < 1.0e-30) rho_face_L = 1.0e-30;
        if (rho_face_R < 1.0e-30) rho_face_R = 1.0e-30;

        // Reconstruct pressure
        reconstruct_1d(fs_left.p[k], fs_center.p[k], fs_right.p[k],
                       p_face_L, p_face_R);

        // Ensure positive pressure
        if (p_face_L < 0.0) p_face_L = 0.0;
        if (p_face_R < 0.0) p_face_R = 0.0;
    }

    /**
     * @brief Interface-aware limiter: reduce to first-order near material interfaces.
     *
     * If volume fraction changes sign of gradient near a face, the limiter
     * drops to zero (first-order) to prevent oscillations.
     */
    KOKKOS_INLINE_FUNCTION
    static Real interface_limiter(Real alpha_left, Real alpha_center, Real alpha_right,
                                  Real threshold = 0.01) {
        // Detect interface: large gradient in alpha
        Real grad_l = alpha_center - alpha_left;
        Real grad_r = alpha_right - alpha_center;

        // If gradients are large and different sign => interface
        if (grad_l * grad_r < 0.0) return 0.0;  // Drop to first order

        // If alpha is near 0 or 1, also be careful
        if (alpha_center < threshold || alpha_center > (1.0 - threshold)) {
            return 0.5;  // Reduce slope by half
        }

        return 1.0;  // Full second-order
    }

    /**
     * @brief Check monotonicity: reconstructed value must lie between neighbors.
     */
    KOKKOS_INLINE_FUNCTION
    static bool is_monotone(Real val_reconstructed, Real val_a, Real val_b) {
        Real lo = Kokkos::fmin(val_a, val_b);
        Real hi = Kokkos::fmax(val_a, val_b);
        return (val_reconstructed >= lo - 1.0e-12) && (val_reconstructed <= hi + 1.0e-12);
    }
};

// ============================================================================
// 34b-4: MultiFluidEBCS — Eulerian BCs for multi-fluid
// ============================================================================

/**
 * @brief Boundary conditions for multi-fluid Eulerian solver.
 *
 * - Inlet: prescribed volume fractions, densities, velocities
 * - Outlet: zero-gradient extrapolation
 * - NRF: non-reflecting far-field (characteristic-based)
 * - Wall: reflecting (normal velocity reversed, tangential preserved)
 */
class MultiFluidEBCS {
public:
    /**
     * @brief Apply multi-fluid BC to create a ghost state.
     *
     * @param interior  Interior cell state
     * @param ghost     [out] Ghost cell state
     * @param bc        BC specification
     */
    static void apply_multifluid_bc(const FluidState& interior, FluidState& ghost,
                                    const MultiFluidBCData& bc) {
        switch (bc.type) {
        case MultiFluidBCType::Inlet:
            ghost = bc.inlet_state;
            break;

        case MultiFluidBCType::Outlet:
            // Zero-gradient: ghost = interior
            ghost = interior;
            break;

        case MultiFluidBCType::NRF: {
            // Non-reflecting: characteristic-based
            // Use 1D Riemann invariants along face normal
            ghost = interior;
            int normal_dir = bc.face / 2;

            Real p_int = interior.mixture_pressure();
            Real rho_int = interior.mixture_density();

            // Non-reflecting: absorb outgoing waves
            // For subsonic outflow: extrapolate everything except pressure
            Real p_ghost = bc.p_inf;  // Far-field pressure
            ghost = interior;

            // Adjust velocities for pressure difference
            Real rho_c = rho_int * bc.c_inf;
            if (rho_c > 1.0e-30) {
                Real dp = p_int - p_ghost;
                Real dvn = dp / rho_c;
                if (normal_dir == 0) ghost.u -= dvn;
                else if (normal_dir == 1) ghost.v -= dvn;
                else ghost.w -= dvn;
            }

            // Scale pressures
            Real p_ratio = (p_int > 1.0e-30) ? p_ghost / p_int : 1.0;
            for (int k = 0; k < ghost.nfluids; ++k) {
                ghost.p[k] *= p_ratio;
            }
            break;
        }

        case MultiFluidBCType::Wall: {
            // Reflecting wall
            ghost = interior;
            int normal_dir = bc.face / 2;
            if (normal_dir == 0) ghost.u = -interior.u;
            else if (normal_dir == 1) ghost.v = -interior.v;
            else ghost.w = -interior.w;
            break;
        }
        }
    }

    /**
     * @brief Apply inlet BC with specific fluid composition.
     */
    static void apply_inlet(FluidState& ghost, const FluidState& inlet,
                            int nfluids) {
        ghost = inlet;
        ghost.nfluids = nfluids;
    }

    /**
     * @brief Apply outlet (zero-gradient) BC.
     */
    static void apply_outlet(const FluidState& interior, FluidState& ghost) {
        ghost = interior;
    }

    /**
     * @brief Check that BC preserves volume fraction sum.
     */
    static bool bc_preserves_alpha_sum(const FluidState& ghost, Real tol = 1.0e-10) {
        Real sum = 0.0;
        for (int k = 0; k < ghost.nfluids; ++k) {
            sum += ghost.alpha[k];
        }
        return Kokkos::fabs(sum - 1.0) < tol;
    }
};

// ============================================================================
// 34b-6: MultiFluidFVM2FEM — FVM-to-FEM data transfer
// ============================================================================

/**
 * @brief Transfer FVM cell-centered data to FEM nodes.
 *
 * Maps pressure from Eulerian cells to nodal forces on an overlapping
 * FEM mesh using weighted interpolation. Used in coupled Euler-Lagrange
 * simulations where the Eulerian fluid exerts forces on FEM structures.
 */
class MultiFluidFVM2FEM {
public:
    /**
     * @brief Transfer pressure from FVM cells to FEM nodal forces.
     *
     * For each FEM node, find the overlapping FVM cell and compute the
     * pressure force contribution: F = -p * A * n (integrated over the
     * node's associated area).
     *
     * Simplified: each node gets force from the nearest cell.
     *
     * @param cells       FVM cell array
     * @param cell_states Multi-fluid states per cell
     * @param ncells      Number of FVM cells
     * @param nodes       [in/out] FEM nodes (forces updated)
     * @param nnodes      Number of FEM nodes
     * @param normals     Surface normals at FEM nodes [nnodes][3]
     * @param areas       Surface areas at FEM nodes [nnodes]
     * @param cell_origin Cell grid origin [3]
     * @param cell_dx     Cell spacings [3]
     * @param grid_dims   Grid dimensions [3] (nx, ny, nz)
     */
    static void transfer_pressure_to_nodes(const FluidState* cell_states, int ncells,
                                           FEMNode* nodes, int nnodes,
                                           const Real normals[][3],
                                           const Real* areas,
                                           const Real cell_origin[3],
                                           const Real cell_dx[3],
                                           const int grid_dims[3]) {
        for (int n = 0; n < nnodes; ++n) {
            // Find cell containing this node
            int ix = static_cast<int>((nodes[n].x - cell_origin[0]) / cell_dx[0]);
            int iy = static_cast<int>((nodes[n].y - cell_origin[1]) / cell_dx[1]);
            int iz = static_cast<int>((nodes[n].z - cell_origin[2]) / cell_dx[2]);

            // Clamp to grid
            if (ix < 0) ix = 0;
            if (iy < 0) iy = 0;
            if (iz < 0) iz = 0;
            if (ix >= grid_dims[0]) ix = grid_dims[0] - 1;
            if (iy >= grid_dims[1]) iy = grid_dims[1] - 1;
            if (iz >= grid_dims[2]) iz = grid_dims[2] - 1;

            int cell_idx = ix + iy * grid_dims[0] + iz * grid_dims[0] * grid_dims[1];
            if (cell_idx >= ncells) cell_idx = ncells - 1;

            // Mixture pressure in this cell
            Real p_mix = cell_states[cell_idx].mixture_pressure();

            // Force = -p * A * n
            Real area = areas[n];
            nodes[n].fx += -p_mix * area * normals[n][0];
            nodes[n].fy += -p_mix * area * normals[n][1];
            nodes[n].fz += -p_mix * area * normals[n][2];
        }
    }

    /**
     * @brief Simplified 1D pressure-to-force transfer.
     *
     * For testing: 1D row of cells, nodes at cell faces.
     * Force on node i = (p_{i-1} - p_i) * area
     *
     * @param pressures  Cell pressures [ncells]
     * @param ncells     Number of cells
     * @param forces     [out] Nodal forces [ncells+1]
     * @param area       Cross-sectional area
     */
    static void transfer_1d(const Real* pressures, int ncells,
                            Real* forces, Real area) {
        // Node 0: left boundary, force from cell 0
        forces[0] = -pressures[0] * area;

        // Interior nodes: force = (p_left - p_right) * area
        for (int i = 1; i < ncells; ++i) {
            forces[i] = (pressures[i - 1] - pressures[i]) * area;
        }

        // Node ncells: right boundary, force from last cell
        forces[ncells] = pressures[ncells - 1] * area;
    }

    /**
     * @brief Check force balance: sum of all nodal forces should be zero
     * for a closed system with uniform pressure.
     */
    static Real force_balance_error(const Real* forces, int nnodes) {
        Real sum = 0.0;
        for (int i = 0; i < nnodes; ++i) {
            sum += forces[i];
        }
        return Kokkos::fabs(sum);
    }

    /**
     * @brief Transfer density field from FVM to FEM nodes (volume-weighted average).
     */
    static void transfer_density_to_nodes(const FluidState* cell_states, int ncells,
                                          FEMNode* nodes, int nnodes,
                                          const Real cell_origin[3],
                                          const Real cell_dx[3],
                                          const int grid_dims[3],
                                          Real* node_density) {
        for (int n = 0; n < nnodes; ++n) {
            int ix = static_cast<int>((nodes[n].x - cell_origin[0]) / cell_dx[0]);
            int iy = static_cast<int>((nodes[n].y - cell_origin[1]) / cell_dx[1]);
            int iz = static_cast<int>((nodes[n].z - cell_origin[2]) / cell_dx[2]);

            if (ix < 0) ix = 0;
            if (iy < 0) iy = 0;
            if (iz < 0) iz = 0;
            if (ix >= grid_dims[0]) ix = grid_dims[0] - 1;
            if (iy >= grid_dims[1]) iy = grid_dims[1] - 1;
            if (iz >= grid_dims[2]) iz = grid_dims[2] - 1;

            int cell_idx = ix + iy * grid_dims[0] + iz * grid_dims[0] * grid_dims[1];
            if (cell_idx >= ncells) cell_idx = ncells - 1;

            node_density[n] = cell_states[cell_idx].mixture_density();
        }
    }
};

} // namespace fem
} // namespace nxs
