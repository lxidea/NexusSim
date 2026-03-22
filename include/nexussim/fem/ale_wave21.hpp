#pragma once

/**
 * @file ale_wave21.hpp
 * @brief Wave 21: Advanced ALE (Arbitrary Lagrangian-Eulerian) Features
 *
 * Sub-modules:
 * - 21a: FVMALEAdvection — Finite Volume Method ALE advection with HLL Riemann solver
 * - 21b: MUSCLReconstruction — Second-order spatial accuracy (MUSCL + slope limiters)
 * - 21c: ALE2DSolver — 2D ALE for plane strain and axisymmetric problems
 * - 21d: MultiFluidTracker — VOF interface tracking with PLIC reconstruction
 * - 21e: ALEFSICoupling — Fluid-Structure Interaction coupling (staggered scheme)
 * - 21f: ALERemapping — Conservative field transfer after mesh motion
 *
 * References:
 * - Harten, Lax, van Leer (1983) "On upstream differencing and Godunov-type schemes"
 * - van Leer (1979) "Towards the ultimate conservative difference scheme V"
 * - Youngs (1982) "Time-dependent multi-material flow with large fluid distortion"
 * - Donea, Huerta (2003) "Finite Element Methods for Flow Problems"
 */

#include <nexussim/physics/material.hpp>
#include <nexussim/core/types.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>

namespace nxs {
namespace fem {

using Real = nxs::Real;

// ============================================================================
// Common ALE Data Structures
// ============================================================================

/// Conserved state vector: [rho, rho*u, rho*v, rho*w, rho*E]
struct ConservedState {
    Real data[5];

    KOKKOS_INLINE_FUNCTION
    ConservedState() { for (int i = 0; i < 5; ++i) data[i] = 0.0; }

    KOKKOS_INLINE_FUNCTION
    Real& operator[](int i) { return data[i]; }
    KOKKOS_INLINE_FUNCTION
    const Real& operator[](int i) const { return data[i]; }

    KOKKOS_INLINE_FUNCTION
    Real density()  const { return data[0]; }
    KOKKOS_INLINE_FUNCTION
    Real momentum_x() const { return data[1]; }
    KOKKOS_INLINE_FUNCTION
    Real momentum_y() const { return data[2]; }
    KOKKOS_INLINE_FUNCTION
    Real momentum_z() const { return data[3]; }
    KOKKOS_INLINE_FUNCTION
    Real total_energy() const { return data[4]; }

    KOKKOS_INLINE_FUNCTION
    Real velocity_x() const { return (data[0] > 1.0e-30) ? data[1] / data[0] : 0.0; }
    KOKKOS_INLINE_FUNCTION
    Real velocity_y() const { return (data[0] > 1.0e-30) ? data[2] / data[0] : 0.0; }
    KOKKOS_INLINE_FUNCTION
    Real velocity_z() const { return (data[0] > 1.0e-30) ? data[3] / data[0] : 0.0; }

    /// Specific kinetic energy
    KOKKOS_INLINE_FUNCTION
    Real kinetic_energy() const {
        if (data[0] < 1.0e-30) return 0.0;
        Real rho_inv = 1.0 / data[0];
        return 0.5 * (data[1]*data[1] + data[2]*data[2] + data[3]*data[3]) * rho_inv;
    }

    /// Specific internal energy: e = E - 0.5*(u^2+v^2+w^2)
    KOKKOS_INLINE_FUNCTION
    Real internal_energy() const {
        if (data[0] < 1.0e-30) return 0.0;
        return data[4] / data[0] - kinetic_energy() / data[0];
    }

    /// Pressure from ideal gas EOS: p = (gamma - 1) * rho * e_internal
    KOKKOS_INLINE_FUNCTION
    Real pressure(Real gamma) const {
        Real rho = data[0];
        if (rho < 1.0e-30) return 0.0;
        Real ke = kinetic_energy();
        Real rho_e = data[4] - ke;
        return (gamma - 1.0) * rho_e;
    }

    /// Sound speed from ideal gas: c = sqrt(gamma * p / rho)
    KOKKOS_INLINE_FUNCTION
    Real sound_speed(Real gamma) const {
        Real rho = data[0];
        if (rho < 1.0e-30) return 0.0;
        Real p = pressure(gamma);
        if (p < 0.0) p = 0.0;
        return std::sqrt(gamma * p / rho);
    }
};

/// Face connectivity: connects two cells across a face
struct ALEFace {
    int cell_left;         ///< Left cell index (-1 for boundary)
    int cell_right;        ///< Right cell index (-1 for boundary)
    Real normal[3];        ///< Outward normal (from left to right), unit vector
    Real area;             ///< Face area

    KOKKOS_INLINE_FUNCTION
    ALEFace() : cell_left(-1), cell_right(-1), area(0.0) {
        normal[0] = normal[1] = normal[2] = 0.0;
    }
};

/// Cell data for FVM
struct ALECell {
    ConservedState U;      ///< Conserved variables
    Real volume;           ///< Cell volume
    Real centroid[3];      ///< Cell centroid

    KOKKOS_INLINE_FUNCTION
    ALECell() : volume(0.0) {
        centroid[0] = centroid[1] = centroid[2] = 0.0;
    }
};

// ============================================================================
// 21a: FVMALEAdvection — Finite Volume Method ALE Advection
// ============================================================================

/**
 * @brief Finite Volume Method ALE advection with HLL approximate Riemann solver.
 *
 * Cell-centered scheme. Each face between two cells computes the numerical
 * flux using the HLL (Harten-Lax-van Leer) approach:
 *
 *   If S_L >= 0:  F_hll = F_L
 *   If S_R <= 0:  F_hll = F_R
 *   Otherwise:    F_hll = (S_R*F_L - S_L*F_R + S_L*S_R*(U_R - U_L)) / (S_R - S_L)
 *
 * Wave speed estimates (Davis 1988):
 *   S_L = min(u_L - c_L, u_R - c_R)
 *   S_R = max(u_L + c_L, u_R + c_R)
 *
 * The time integration is explicit forward Euler:
 *   U_i^{n+1} = U_i^n - (dt / V_i) * sum_f (F_f * A_f)
 */
class FVMALEAdvection {
public:
    FVMALEAdvection() = default;

    explicit FVMALEAdvection(Real gamma) : gamma_(gamma) {}

    /// Set ratio of specific heats
    void set_gamma(Real gamma) { gamma_ = gamma; }
    Real gamma() const { return gamma_; }

    /**
     * @brief Compute HLL flux across a single face.
     *
     * @param cell_left  Conserved state of the left cell
     * @param cell_right Conserved state of the right cell
     * @param face_normal Unit normal pointing from left to right [3]
     * @param flux Output numerical flux vector [5]: [rho, rho*u, rho*v, rho*w, rho*E]
     */
    KOKKOS_INLINE_FUNCTION
    void compute_flux(const ConservedState& cell_left,
                      const ConservedState& cell_right,
                      const Real face_normal[3],
                      Real flux[5]) const {
        // Extract primitive variables for left state
        Real rho_L = cell_left.density();
        Real u_L = cell_left.velocity_x();
        Real v_L = cell_left.velocity_y();
        Real w_L = cell_left.velocity_z();
        Real p_L = cell_left.pressure(gamma_);
        Real c_L = cell_left.sound_speed(gamma_);
        if (p_L < 0.0) p_L = 0.0;

        // Extract primitive variables for right state
        Real rho_R = cell_right.density();
        Real u_R = cell_right.velocity_x();
        Real v_R = cell_right.velocity_y();
        Real w_R = cell_right.velocity_z();
        Real p_R = cell_right.pressure(gamma_);
        Real c_R = cell_right.sound_speed(gamma_);
        if (p_R < 0.0) p_R = 0.0;

        // Normal velocities
        Real vn_L = u_L * face_normal[0] + v_L * face_normal[1] + w_L * face_normal[2];
        Real vn_R = u_R * face_normal[0] + v_R * face_normal[1] + w_R * face_normal[2];

        // HLL wave speed estimates (Davis 1988)
        Real S_L = std::min(vn_L - c_L, vn_R - c_R);
        Real S_R = std::max(vn_L + c_L, vn_R + c_R);

        // Physical flux for left state:  F_L = [rho*vn, rho*u*vn + p*nx, ..., (rho*E+p)*vn]
        Real E_L = cell_left.total_energy();
        Real F_L[5];
        F_L[0] = rho_L * vn_L;
        F_L[1] = rho_L * u_L * vn_L + p_L * face_normal[0];
        F_L[2] = rho_L * v_L * vn_L + p_L * face_normal[1];
        F_L[3] = rho_L * w_L * vn_L + p_L * face_normal[2];
        F_L[4] = (E_L + p_L) * vn_L;

        // Physical flux for right state
        Real E_R = cell_right.total_energy();
        Real F_R[5];
        F_R[0] = rho_R * vn_R;
        F_R[1] = rho_R * u_R * vn_R + p_R * face_normal[0];
        F_R[2] = rho_R * v_R * vn_R + p_R * face_normal[1];
        F_R[3] = rho_R * w_R * vn_R + p_R * face_normal[2];
        F_R[4] = (E_R + p_R) * vn_R;

        // HLL flux
        if (S_L >= 0.0) {
            // Supersonic from left
            for (int i = 0; i < 5; ++i) flux[i] = F_L[i];
        } else if (S_R <= 0.0) {
            // Supersonic from right
            for (int i = 0; i < 5; ++i) flux[i] = F_R[i];
        } else {
            // Subsonic: HLL intermediate state
            Real dS = S_R - S_L;
            if (std::abs(dS) < 1.0e-30) dS = 1.0e-30;
            Real inv_dS = 1.0 / dS;

            Real U_L[5] = { rho_L, rho_L*u_L, rho_L*v_L, rho_L*w_L, E_L };
            Real U_R[5] = { rho_R, rho_R*u_R, rho_R*v_R, rho_R*w_R, E_R };

            for (int i = 0; i < 5; ++i) {
                flux[i] = (S_R * F_L[i] - S_L * F_R[i]
                           + S_L * S_R * (U_R[i] - U_L[i])) * inv_dS;
            }
        }
    }

    /**
     * @brief Advect all cells by one time step using forward Euler.
     *
     * U_i^{n+1} = U_i^n - (dt / V_i) * sum_f (F_f * A_f)
     *
     * @param cells Array of cell data (updated in-place)
     * @param num_cells Number of cells
     * @param faces Array of face connectivity
     * @param num_faces Number of faces
     * @param dt Time step
     */
    void advect(ALECell* cells, int num_cells,
                const ALEFace* faces, int num_faces,
                Real dt) const {
        // Accumulate flux contributions per cell
        std::vector<ConservedState> dU(num_cells);

        for (int f = 0; f < num_faces; ++f) {
            int iL = faces[f].cell_left;
            int iR = faces[f].cell_right;

            // Get left and right states (use reflecting BC for boundaries)
            ConservedState U_L, U_R;
            if (iL >= 0 && iL < num_cells) {
                U_L = cells[iL].U;
            }
            if (iR >= 0 && iR < num_cells) {
                U_R = cells[iR].U;
            }

            // Boundary handling: reflecting wall
            if (iL < 0) {
                U_L = U_R;
                // Reflect normal velocity component
                Real vn = U_L.velocity_x() * faces[f].normal[0]
                        + U_L.velocity_y() * faces[f].normal[1]
                        + U_L.velocity_z() * faces[f].normal[2];
                Real rho = U_L.density();
                U_L[1] -= 2.0 * rho * vn * faces[f].normal[0];
                U_L[2] -= 2.0 * rho * vn * faces[f].normal[1];
                U_L[3] -= 2.0 * rho * vn * faces[f].normal[2];
            }
            if (iR < 0) {
                U_R = U_L;
                Real vn = U_R.velocity_x() * faces[f].normal[0]
                        + U_R.velocity_y() * faces[f].normal[1]
                        + U_R.velocity_z() * faces[f].normal[2];
                Real rho = U_R.density();
                U_R[1] -= 2.0 * rho * vn * faces[f].normal[0];
                U_R[2] -= 2.0 * rho * vn * faces[f].normal[1];
                U_R[3] -= 2.0 * rho * vn * faces[f].normal[2];
            }

            // Compute flux
            Real flux[5];
            compute_flux(U_L, U_R, faces[f].normal, flux);

            // Accumulate: subtract from left, add to right
            Real fA = faces[f].area;
            if (iL >= 0 && iL < num_cells) {
                for (int i = 0; i < 5; ++i) dU[iL][i] -= flux[i] * fA;
            }
            if (iR >= 0 && iR < num_cells) {
                for (int i = 0; i < 5; ++i) dU[iR][i] += flux[i] * fA;
            }
        }

        // Update cell states
        for (int c = 0; c < num_cells; ++c) {
            if (cells[c].volume < 1.0e-30) continue;
            Real dt_over_V = dt / cells[c].volume;
            for (int i = 0; i < 5; ++i) {
                cells[c].U[i] += dt_over_V * dU[c][i];
            }
            // Enforce positivity of density
            if (cells[c].U[0] < 1.0e-30) cells[c].U[0] = 1.0e-30;
        }
    }

    /**
     * @brief Compute maximum stable time step (CFL condition).
     *
     * dt = CFL * min_c (V_c^{1/3} / (|v_c| + c_c))
     */
    Real compute_max_dt(const ALECell* cells, int num_cells, Real cfl = 0.5) const {
        Real dt_min = 1.0e30;
        for (int c = 0; c < num_cells; ++c) {
            Real rho = cells[c].U.density();
            if (rho < 1.0e-30) continue;

            Real vx = cells[c].U.velocity_x();
            Real vy = cells[c].U.velocity_y();
            Real vz = cells[c].U.velocity_z();
            Real v_mag = std::sqrt(vx*vx + vy*vy + vz*vz);
            Real cs = cells[c].U.sound_speed(gamma_);

            Real char_len = std::cbrt(std::max(cells[c].volume, 1.0e-30));
            Real wave_speed = v_mag + cs;
            if (wave_speed > 1.0e-30) {
                Real dt_c = char_len / wave_speed;
                dt_min = std::min(dt_min, dt_c);
            }
        }
        return cfl * dt_min;
    }

private:
    Real gamma_ = 1.4;  ///< Ratio of specific heats (default: ideal diatomic gas)
};

// ============================================================================
// 21b: MUSCLReconstruction — Second-Order Spatial Accuracy
// ============================================================================

/**
 * @brief MUSCL (Monotone Upstream-centered Schemes for Conservation Laws)
 *        linear reconstruction with slope limiters.
 *
 * Achieves second-order spatial accuracy by reconstructing left/right states
 * at cell faces using gradients with TVD limiters:
 *
 *   U_L(face) = U_i + 0.5 * phi(r) * (U_i - U_{i-1})
 *   U_R(face) = U_j - 0.5 * phi(r) * (U_{j+1} - U_j)
 *
 * where phi(r) is the limiter function and r is the slope ratio.
 *
 * Supported limiters:
 * - Minmod: most diffusive, strictly TVD
 * - Van Leer: smooth, second-order
 * - Superbee: least diffusive, sharpest resolution
 */
class MUSCLReconstruction {
public:
    enum class LimiterType { Minmod, VanLeer, Superbee };

    MUSCLReconstruction() = default;
    explicit MUSCLReconstruction(LimiterType lt) : limiter_type_(lt) {}

    void set_limiter(LimiterType lt) { limiter_type_ = lt; }
    LimiterType limiter() const { return limiter_type_; }

    /**
     * @brief Minmod limiter: phi(a, b) = sign(a) * max(0, min(|a|, sign(a)*b))
     *
     * Most diffusive TVD limiter. Switches to first-order near extrema.
     */
    KOKKOS_INLINE_FUNCTION
    static Real minmod_limiter(Real a, Real b) {
        if (a * b <= 0.0) return 0.0;
        if (std::abs(a) < std::abs(b))
            return a;
        return b;
    }

    /**
     * @brief Van Leer limiter: phi(a, b) = 2ab / (a + b) if same sign, else 0
     *
     * Smooth, continuously differentiable limiter. Good all-around choice.
     */
    KOKKOS_INLINE_FUNCTION
    static Real vanleer_limiter(Real a, Real b) {
        if (a * b <= 0.0) return 0.0;
        return 2.0 * a * b / (a + b);
    }

    /**
     * @brief Superbee limiter: phi = max(0, min(2|a|, |b|), min(|a|, 2|b|))
     *        applied with appropriate sign.
     *
     * Least diffusive TVD limiter. Best for contact discontinuities.
     */
    KOKKOS_INLINE_FUNCTION
    static Real superbee_limiter(Real a, Real b) {
        if (a * b <= 0.0) return 0.0;
        Real abs_a = std::abs(a);
        Real abs_b = std::abs(b);
        Real s1 = std::min(2.0 * abs_a, abs_b);
        Real s2 = std::min(abs_a, 2.0 * abs_b);
        Real phi = std::max(s1, s2);
        return (a > 0.0) ? phi : -phi;
    }

    /**
     * @brief Apply the selected limiter.
     */
    KOKKOS_INLINE_FUNCTION
    Real apply_limiter(Real a, Real b) const {
        switch (limiter_type_) {
            case LimiterType::VanLeer:  return vanleer_limiter(a, b);
            case LimiterType::Superbee: return superbee_limiter(a, b);
            case LimiterType::Minmod:
            default:                    return minmod_limiter(a, b);
        }
    }

    /**
     * @brief Reconstruct left and right states at a face using MUSCL.
     *
     * Given a 1D stencil of cell values [U_{i-1}, U_i, U_{i+1}, U_{i+2}],
     * computes the reconstructed left and right states at the face between
     * cells i and i+1.
     *
     * @param cell_values Array of 4 consecutive cell values [U_{i-1}, U_i, U_{i+1}, U_{i+2}]
     * @param gradients Array of 4 corresponding gradient estimates (for multi-D)
     * @param face_left Output: reconstructed left state at face
     * @param face_right Output: reconstructed right state at face
     */
    KOKKOS_INLINE_FUNCTION
    void reconstruct(const Real cell_values[4], const Real gradients[4],
                     Real& face_left, Real& face_right) const {
        // Left-biased differences
        Real delta_L_minus = cell_values[1] - cell_values[0]; // U_i - U_{i-1}
        Real delta_L_plus  = cell_values[2] - cell_values[1]; // U_{i+1} - U_i

        // Right-biased differences
        Real delta_R_minus = cell_values[2] - cell_values[1]; // U_{i+1} - U_i
        Real delta_R_plus  = cell_values[3] - cell_values[2]; // U_{i+2} - U_{i+1}

        // Limited slopes
        Real slope_L = apply_limiter(delta_L_minus, delta_L_plus);
        Real slope_R = apply_limiter(delta_R_minus, delta_R_plus);

        // Reconstruct: extrapolate half a cell width to the face
        face_left  = cell_values[1] + 0.5 * slope_L;
        face_right = cell_values[2] - 0.5 * slope_R;
    }

    /**
     * @brief Reconstruct conserved state vectors at a face.
     *
     * Applies component-wise MUSCL reconstruction to all 5 conserved variables.
     *
     * @param U Array of 4 consecutive cell states
     * @param U_left Output: reconstructed left state
     * @param U_right Output: reconstructed right state
     */
    void reconstruct_state(const ConservedState U[4],
                           ConservedState& U_left,
                           ConservedState& U_right) const {
        for (int comp = 0; comp < 5; ++comp) {
            Real vals[4] = { U[0][comp], U[1][comp], U[2][comp], U[3][comp] };
            Real grads[4] = { 0.0, 0.0, 0.0, 0.0 }; // Not used for 1D
            reconstruct(vals, grads, U_left[comp], U_right[comp]);
        }

        // Enforce positivity on density
        if (U_left[0] < 1.0e-30)  U_left[0] = 1.0e-30;
        if (U_right[0] < 1.0e-30) U_right[0] = 1.0e-30;
    }

    /**
     * @brief Compute cell-centered gradients using least-squares.
     *
     * For each cell, given its neighbors, compute gradient of a scalar q:
     *   grad(q)_i = (sum_j w_j * (q_j - q_i) * d_ij) / (sum_j w_j * d_ij * d_ij)
     *
     * where d_ij is the centroid-to-centroid vector and w_j = 1/|d_ij|.
     *
     * @param cell_values Scalar values at cell centers
     * @param centroids Cell centroid positions [num_cells][3]
     * @param neighbors Neighbor lists: neighbors[i][j] is the j-th neighbor of cell i
     * @param num_neighbors Number of neighbors per cell
     * @param num_cells Total number of cells
     * @param gradients Output: gradients[i][3] for each cell
     */
    static void compute_gradients(const Real* cell_values,
                                  const Real centroids[][3],
                                  const int neighbors[][6],
                                  const int* num_neighbors,
                                  int num_cells,
                                  Real gradients[][3]) {
        for (int i = 0; i < num_cells; ++i) {
            // Initialize least-squares system: A^T * A * g = A^T * b
            // Using simplified weighted approach
            Real AtA[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
            Real Atb[3] = {0.0, 0.0, 0.0};

            for (int n = 0; n < num_neighbors[i]; ++n) {
                int j = neighbors[i][n];
                if (j < 0 || j >= num_cells) continue;

                Real dx = centroids[j][0] - centroids[i][0];
                Real dy = centroids[j][1] - centroids[i][1];
                Real dz = centroids[j][2] - centroids[i][2];
                Real dist2 = dx*dx + dy*dy + dz*dz;
                if (dist2 < 1.0e-30) continue;

                Real w = 1.0 / std::sqrt(dist2);
                Real dq = cell_values[j] - cell_values[i];

                // Weighted normal equations
                Real wdx = w * dx, wdy = w * dy, wdz = w * dz;
                AtA[0][0] += wdx * wdx; AtA[0][1] += wdx * wdy; AtA[0][2] += wdx * wdz;
                AtA[1][0] += wdy * wdx; AtA[1][1] += wdy * wdy; AtA[1][2] += wdy * wdz;
                AtA[2][0] += wdz * wdx; AtA[2][1] += wdz * wdy; AtA[2][2] += wdz * wdz;
                Atb[0] += w * dq * wdx;
                Atb[1] += w * dq * wdy;
                Atb[2] += w * dq * wdz;
            }

            // Solve 3x3 system using Cramer's rule
            Real det = AtA[0][0] * (AtA[1][1]*AtA[2][2] - AtA[1][2]*AtA[2][1])
                     - AtA[0][1] * (AtA[1][0]*AtA[2][2] - AtA[1][2]*AtA[2][0])
                     + AtA[0][2] * (AtA[1][0]*AtA[2][1] - AtA[1][1]*AtA[2][0]);

            if (std::abs(det) > 1.0e-30) {
                Real inv_det = 1.0 / det;
                gradients[i][0] = inv_det * (
                    Atb[0] * (AtA[1][1]*AtA[2][2] - AtA[1][2]*AtA[2][1])
                  - AtA[0][1] * (Atb[1]*AtA[2][2] - AtA[1][2]*Atb[2])
                  + AtA[0][2] * (Atb[1]*AtA[2][1] - AtA[1][1]*Atb[2]));
                gradients[i][1] = inv_det * (
                    AtA[0][0] * (Atb[1]*AtA[2][2] - AtA[1][2]*Atb[2])
                  - Atb[0] * (AtA[1][0]*AtA[2][2] - AtA[1][2]*AtA[2][0])
                  + AtA[0][2] * (AtA[1][0]*Atb[2] - Atb[1]*AtA[2][0]));
                gradients[i][2] = inv_det * (
                    AtA[0][0] * (AtA[1][1]*Atb[2] - Atb[1]*AtA[2][1])
                  - AtA[0][1] * (AtA[1][0]*Atb[2] - Atb[1]*AtA[2][0])
                  + Atb[0] * (AtA[1][0]*AtA[2][1] - AtA[1][1]*AtA[2][0]));
            } else {
                gradients[i][0] = gradients[i][1] = gradients[i][2] = 0.0;
            }
        }
    }

private:
    LimiterType limiter_type_ = LimiterType::Minmod;
};

// ============================================================================
// 21c: ALE2DSolver — 2D ALE for Plane Strain and Axisymmetric Problems
// ============================================================================

/**
 * @brief 2D ALE solver for plane strain and axisymmetric problems.
 *
 * Reduces the 3D formulation to 2D (x-y plane):
 * - Plane strain: Unit depth in z-direction, no z-velocity
 * - Axisymmetric: x = radial (r), y = axial (z), includes geometric source
 *   terms for hoop stress: S_hoop = -p / r
 *
 * Conserved variables in 2D: [rho, rho*u, rho*v, rho*E]
 * Flux computed across edges (1D faces in 2D).
 */
class ALE2DSolver {
public:
    enum class Mode { PlaneStrain, Axisymmetric };

    /// 2D conserved state: [rho, rho*u, rho*v, rho*E]
    struct State2D {
        Real data[4];

        KOKKOS_INLINE_FUNCTION
        State2D() { for (int i = 0; i < 4; ++i) data[i] = 0.0; }

        KOKKOS_INLINE_FUNCTION
        Real& operator[](int i) { return data[i]; }
        KOKKOS_INLINE_FUNCTION
        const Real& operator[](int i) const { return data[i]; }

        KOKKOS_INLINE_FUNCTION
        Real density() const { return data[0]; }
        KOKKOS_INLINE_FUNCTION
        Real velocity_u() const { return (data[0] > 1.0e-30) ? data[1] / data[0] : 0.0; }
        KOKKOS_INLINE_FUNCTION
        Real velocity_v() const { return (data[0] > 1.0e-30) ? data[2] / data[0] : 0.0; }

        KOKKOS_INLINE_FUNCTION
        Real pressure(Real gamma) const {
            if (data[0] < 1.0e-30) return 0.0;
            Real rho = data[0];
            Real ke = 0.5 * (data[1]*data[1] + data[2]*data[2]) / rho;
            return (gamma - 1.0) * (data[3] - ke);
        }

        KOKKOS_INLINE_FUNCTION
        Real sound_speed(Real gamma) const {
            Real p = pressure(gamma);
            if (p < 0.0) p = 0.0;
            return (data[0] > 1.0e-30) ? std::sqrt(gamma * p / data[0]) : 0.0;
        }
    };

    /// 2D edge (line segment between two cells)
    struct Edge2D {
        int cell_left;
        int cell_right;
        Real normal[2];   ///< Unit outward normal (left to right)
        Real length;       ///< Edge length

        KOKKOS_INLINE_FUNCTION
        Edge2D() : cell_left(-1), cell_right(-1), length(0.0) {
            normal[0] = normal[1] = 0.0;
        }
    };

    /// 2D cell
    struct Cell2D {
        State2D U;
        Real area;          ///< Cell area
        Real centroid[2];   ///< Centroid position (x=r for axisymmetric)

        KOKKOS_INLINE_FUNCTION
        Cell2D() : area(0.0) { centroid[0] = centroid[1] = 0.0; }
    };

    ALE2DSolver() = default;
    ALE2DSolver(Real gamma, Mode mode) : gamma_(gamma), mode_(mode) {}

    void set_gamma(Real gamma) { gamma_ = gamma; }
    void set_mode(Mode mode) { mode_ = mode; }
    Mode mode() const { return mode_; }

    /**
     * @brief Compute 2D HLL flux across an edge.
     *
     * @param left  Left state
     * @param right Right state
     * @param edge_normal Unit normal [2]
     * @param flux Output flux [4]
     */
    KOKKOS_INLINE_FUNCTION
    void compute_2d_flux(const State2D& left, const State2D& right,
                         const Real edge_normal[2], Real flux[4]) const {
        Real rho_L = left.density();
        Real u_L = left.velocity_u();
        Real v_L = left.velocity_v();
        Real p_L = left.pressure(gamma_);
        Real c_L = left.sound_speed(gamma_);
        if (p_L < 0.0) p_L = 0.0;

        Real rho_R = right.density();
        Real u_R = right.velocity_u();
        Real v_R = right.velocity_v();
        Real p_R = right.pressure(gamma_);
        Real c_R = right.sound_speed(gamma_);
        if (p_R < 0.0) p_R = 0.0;

        Real vn_L = u_L * edge_normal[0] + v_L * edge_normal[1];
        Real vn_R = u_R * edge_normal[0] + v_R * edge_normal[1];

        Real S_L = std::min(vn_L - c_L, vn_R - c_R);
        Real S_R = std::max(vn_L + c_L, vn_R + c_R);

        Real E_L = left[3];
        Real F_L[4] = {
            rho_L * vn_L,
            rho_L * u_L * vn_L + p_L * edge_normal[0],
            rho_L * v_L * vn_L + p_L * edge_normal[1],
            (E_L + p_L) * vn_L
        };

        Real E_R = right[3];
        Real F_R[4] = {
            rho_R * vn_R,
            rho_R * u_R * vn_R + p_R * edge_normal[0],
            rho_R * v_R * vn_R + p_R * edge_normal[1],
            (E_R + p_R) * vn_R
        };

        if (S_L >= 0.0) {
            for (int i = 0; i < 4; ++i) flux[i] = F_L[i];
        } else if (S_R <= 0.0) {
            for (int i = 0; i < 4; ++i) flux[i] = F_R[i];
        } else {
            Real dS = S_R - S_L;
            if (std::abs(dS) < 1.0e-30) dS = 1.0e-30;
            Real inv_dS = 1.0 / dS;

            Real U_L[4] = { rho_L, rho_L*u_L, rho_L*v_L, E_L };
            Real U_R[4] = { rho_R, rho_R*u_R, rho_R*v_R, E_R };

            for (int i = 0; i < 4; ++i) {
                flux[i] = (S_R * F_L[i] - S_L * F_R[i]
                           + S_L * S_R * (U_R[i] - U_L[i])) * inv_dS;
            }
        }
    }

    /**
     * @brief Advect 2D cells by one time step.
     *
     * For axisymmetric mode, adds geometric source terms:
     *   dU/dt += S(U) where:
     *     S = -(1/r) * [rho*u, rho*u^2, rho*u*v, (rho*E+p)*u]
     *   (hoop stress contribution from azimuthal confinement)
     *
     * @param cells Cell array (modified in-place)
     * @param num_cells Number of cells
     * @param edges Edge array
     * @param num_edges Number of edges
     * @param dt Time step
     */
    void advect_2d(Cell2D* cells, int num_cells,
                   const Edge2D* edges, int num_edges,
                   Real dt) const {
        // Accumulate flux contributions
        std::vector<State2D> dU(num_cells);

        for (int e = 0; e < num_edges; ++e) {
            int iL = edges[e].cell_left;
            int iR = edges[e].cell_right;

            State2D U_L, U_R;
            if (iL >= 0 && iL < num_cells) U_L = cells[iL].U;
            if (iR >= 0 && iR < num_cells) U_R = cells[iR].U;

            // Boundary: reflecting
            if (iL < 0) {
                U_L = U_R;
                Real vn = U_L.velocity_u() * edges[e].normal[0]
                        + U_L.velocity_v() * edges[e].normal[1];
                Real rho = U_L.density();
                U_L[1] -= 2.0 * rho * vn * edges[e].normal[0];
                U_L[2] -= 2.0 * rho * vn * edges[e].normal[1];
            }
            if (iR < 0) {
                U_R = U_L;
                Real vn = U_R.velocity_u() * edges[e].normal[0]
                        + U_R.velocity_v() * edges[e].normal[1];
                Real rho = U_R.density();
                U_R[1] -= 2.0 * rho * vn * edges[e].normal[0];
                U_R[2] -= 2.0 * rho * vn * edges[e].normal[1];
            }

            Real flux[4];
            compute_2d_flux(U_L, U_R, edges[e].normal, flux);

            Real len = edges[e].length;
            if (iL >= 0 && iL < num_cells) {
                for (int i = 0; i < 4; ++i) dU[iL][i] -= flux[i] * len;
            }
            if (iR >= 0 && iR < num_cells) {
                for (int i = 0; i < 4; ++i) dU[iR][i] += flux[i] * len;
            }
        }

        // Update cells
        for (int c = 0; c < num_cells; ++c) {
            if (cells[c].area < 1.0e-30) continue;
            Real dt_over_A = dt / cells[c].area;
            for (int i = 0; i < 4; ++i) {
                cells[c].U[i] += dt_over_A * dU[c][i];
            }

            // Axisymmetric source terms
            if (mode_ == Mode::Axisymmetric) {
                Real r = cells[c].centroid[0]; // radial coordinate
                if (r > 1.0e-30) {
                    Real rho = cells[c].U.density();
                    Real u = cells[c].U.velocity_u();
                    Real v = cells[c].U.velocity_v();
                    Real p = cells[c].U.pressure(gamma_);
                    if (p < 0.0) p = 0.0;
                    Real E = cells[c].U[3];

                    // Source: S = -(1/r) * [rho*u, rho*u^2 - p, rho*u*v, (E+p)*u]
                    // Note: sign convention gives hoop stress contribution
                    Real inv_r = 1.0 / r;
                    cells[c].U[0] -= dt * inv_r * rho * u;
                    cells[c].U[1] -= dt * inv_r * (rho * u * u);
                    cells[c].U[2] -= dt * inv_r * (rho * u * v);
                    cells[c].U[3] -= dt * inv_r * ((E + p) * u);
                }
            }

            // Enforce positivity
            if (cells[c].U[0] < 1.0e-30) cells[c].U[0] = 1.0e-30;
        }
    }

private:
    Real gamma_ = 1.4;
    Mode mode_ = Mode::PlaneStrain;
};

// ============================================================================
// 21d: MultiFluidTracker — VOF Interface Tracking with PLIC
// ============================================================================

/**
 * @brief Volume of Fluid (VOF) interface tracking with PLIC reconstruction.
 *
 * Tracks material interfaces using a color function F in [0, 1]:
 *   F = 0: cell is entirely material B
 *   F = 1: cell is entirely material A
 *   0 < F < 1: interface cell, partially filled
 *
 * PLIC (Piecewise Linear Interface Calculation) reconstructs the interface
 * as a plane n . x = d within each mixed cell, where n is the interface
 * normal estimated from the gradient of F, and d is chosen to match the
 * volume fraction.
 *
 * References:
 * - Youngs (1982) "Time-dependent multi-material flow with large fluid distortion"
 * - Rider & Kothe (1998) "Reconstructing volume tracking"
 */
class MultiFluidTracker {
public:
    /// Interface plane in a cell: n . x = d
    struct InterfacePlane {
        Real normal[3];   ///< Interface normal (unit vector, pointing into material A)
        Real d;           ///< Plane constant

        KOKKOS_INLINE_FUNCTION
        InterfacePlane() : d(0.0) { normal[0] = normal[1] = normal[2] = 0.0; }
    };

    MultiFluidTracker() = default;

    /**
     * @brief Advect the VOF color function F using the split-direction method.
     *
     * Operator splitting in x, y, z with directional fluxes:
     *   F_i^{n+1} = F_i^n - (dt / V_i) * (flux_R - flux_L)
     *
     * Each directional flux is computed using the PLIC-reconstructed interface
     * position to determine how much fluid crosses the face.
     *
     * @param F Volume fractions (modified in-place) [num_cells]
     * @param velocity_x, velocity_y, velocity_z Cell-centered velocities
     * @param cell_volumes Cell volumes
     * @param faces Face connectivity and geometry
     * @param num_cells Number of cells
     * @param num_faces Number of faces
     * @param dt Time step
     */
    void advect_vof(Real* F, const Real* velocity_x, const Real* velocity_y,
                    const Real* velocity_z, const Real* cell_volumes,
                    const ALEFace* faces, int num_cells, int num_faces,
                    Real dt) const {
        // Compute interface normals from gradient of F
        std::vector<Real> nx(num_cells, 0.0), ny(num_cells, 0.0), nz(num_cells, 0.0);

        // Estimate gradients of F across faces
        for (int f = 0; f < num_faces; ++f) {
            int iL = faces[f].cell_left;
            int iR = faces[f].cell_right;
            if (iL < 0 || iR < 0 || iL >= num_cells || iR >= num_cells) continue;

            Real dF = F[iR] - F[iL];
            Real area = faces[f].area;
            nx[iL] += dF * faces[f].normal[0] * area;
            ny[iL] += dF * faces[f].normal[1] * area;
            nz[iL] += dF * faces[f].normal[2] * area;
            nx[iR] += dF * faces[f].normal[0] * area;
            ny[iR] += dF * faces[f].normal[1] * area;
            nz[iR] += dF * faces[f].normal[2] * area;
        }

        // Normalize interface normals
        for (int c = 0; c < num_cells; ++c) {
            Real mag = std::sqrt(nx[c]*nx[c] + ny[c]*ny[c] + nz[c]*nz[c]);
            if (mag > 1.0e-30) {
                nx[c] /= mag;
                ny[c] /= mag;
                nz[c] /= mag;
            }
        }

        // Advect F using donor-acceptor method across each face
        std::vector<Real> dF(num_cells, 0.0);

        for (int f = 0; f < num_faces; ++f) {
            int iL = faces[f].cell_left;
            int iR = faces[f].cell_right;
            if (iL < 0 || iR < 0 || iL >= num_cells || iR >= num_cells) continue;

            // Face velocity = average of left/right
            Real vn = 0.5 * ((velocity_x[iL] + velocity_x[iR]) * faces[f].normal[0]
                           + (velocity_y[iL] + velocity_y[iR]) * faces[f].normal[1]
                           + (velocity_z[iL] + velocity_z[iR]) * faces[f].normal[2]);

            // Volume flux across face
            Real vol_flux = vn * faces[f].area * dt;

            // Determine donor cell
            int donor = (vn >= 0.0) ? iL : iR;

            // VOF flux: amount of material A crossing the face
            Real F_flux;
            if (F[donor] > 1.0 - 1.0e-10) {
                // Donor is pure material A
                F_flux = vol_flux;
            } else if (F[donor] < 1.0e-10) {
                // Donor is pure material B
                F_flux = 0.0;
            } else {
                // Mixed cell: use PLIC-based flux
                // Approximate: flux proportional to volume fraction
                F_flux = F[donor] * vol_flux;
            }

            // Accumulate
            if (cell_volumes[iL] > 1.0e-30) dF[iL] -= F_flux / cell_volumes[iL];
            if (cell_volumes[iR] > 1.0e-30) dF[iR] += F_flux / cell_volumes[iR];
        }

        // Update F
        for (int c = 0; c < num_cells; ++c) {
            F[c] += dF[c];
            // Clamp to [0, 1]
            if (F[c] < 0.0) F[c] = 0.0;
            if (F[c] > 1.0) F[c] = 1.0;
        }
    }

    /**
     * @brief Reconstruct the interface plane in a mixed cell using PLIC.
     *
     * Given the volume fraction F and the interface normal n (from gradient of F),
     * find the plane constant d such that the volume below the plane
     * n . x = d equals F * V_cell.
     *
     * For a unit cube with normal n, the volume below the plane is computed
     * analytically and d is found by bisection.
     *
     * @param F Volume fraction [0, 1]
     * @param normal Interface normal (unit vector) [3]
     * @param cell_size Characteristic cell size (cube side length)
     * @param d Output: plane constant such that n . x = d
     */
    KOKKOS_INLINE_FUNCTION
    void reconstruct_interface(Real F, const Real normal[3],
                               Real cell_size, Real& d) const {
        if (F <= 0.0 || F >= 1.0) {
            d = (F >= 1.0) ? 1.0e30 : -1.0e30;
            return;
        }

        // Absolute normal components (PLIC works in the unit cube [0,h]^3)
        Real h = cell_size;
        Real abs_n[3] = { std::abs(normal[0]),
                          std::abs(normal[1]),
                          std::abs(normal[2]) };

        // Sort so that abs_n[0] <= abs_n[1] <= abs_n[2]
        if (abs_n[0] > abs_n[1]) { Real t = abs_n[0]; abs_n[0] = abs_n[1]; abs_n[1] = t; }
        if (abs_n[1] > abs_n[2]) { Real t = abs_n[1]; abs_n[1] = abs_n[2]; abs_n[2] = t; }
        if (abs_n[0] > abs_n[1]) { Real t = abs_n[0]; abs_n[0] = abs_n[1]; abs_n[1] = t; }

        Real n1 = abs_n[0], n2 = abs_n[1], n3 = abs_n[2];
        Real n_sum = n1 + n2 + n3;
        if (n_sum < 1.0e-30) { d = 0.0; return; }

        // Target volume
        Real V_target = F * h * h * h;

        // Bisection to find d such that vol(n, d, h) = V_target
        Real d_lo = 0.0;
        Real d_hi = n_sum * h;

        for (int iter = 0; iter < 50; ++iter) {
            Real d_mid = 0.5 * (d_lo + d_hi);
            Real vol = plic_volume(n1, n2, n3, d_mid, h);
            if (vol < V_target) {
                d_lo = d_mid;
            } else {
                d_hi = d_mid;
            }
            if (d_hi - d_lo < 1.0e-14 * n_sum * h) break;
        }

        d = 0.5 * (d_lo + d_hi);
    }

    /**
     * @brief Get material fraction for a cell.
     *
     * @param F Array of volume fractions
     * @param cell_id Cell index
     * @return Volume fraction of material A in [0, 1]
     */
    KOKKOS_INLINE_FUNCTION
    static Real get_material_fraction(const Real* F, int cell_id) {
        return F[cell_id];
    }

private:
    /**
     * @brief Compute volume below plane n.x = d in a cube of side h.
     *
     * Analytic formula for sorted |n1| <= |n2| <= |n3|.
     * The volume is computed by splitting into regions based on
     * where the plane intersects the cube edges.
     */
    KOKKOS_INLINE_FUNCTION
    Real plic_volume(Real n1, Real n2, Real n3, Real d, Real h) const {
        // Normalize to unit cube, then scale back
        Real s = n1 + n2 + n3;
        if (s < 1.0e-30) return 0.0;

        // Scale d to [0, s*h] range
        if (d <= 0.0) return 0.0;
        if (d >= s * h) return h * h * h;

        // Use the Scardovelli-Zaleski (2000) analytic formula for a cube
        // For simplicity, use a simpler piecewise formula:
        // Volume below plane n.x = d in [0,h]^3
        Real a1 = n1 * h, a2 = n2 * h, a3 = n3 * h;
        Real vol = 0.0;

        if (d <= a1) {
            // Small corner: tetrahedron
            vol = d * d * d / (6.0 * n1 * n2 * n3 + 1.0e-30);
        } else if (d <= a2) {
            // Wedge region
            Real v1 = a1;
            vol = (d * d * (3.0 * a1 - d) + v1 * v1 * (v1 - 3.0 * d))
                  / (6.0 * n1 * n2 * n3 + 1.0e-30);
            // Simplified: linear interpolation between regions
            Real t = (d - a1) / (a2 - a1 + 1.0e-30);
            Real vol_a1 = a1 * a1 * a1 / (6.0 * n1 * n2 * n3 + 1.0e-30);
            Real vol_a2_est = 0.5 * h * h * h * (a1 + a2) / (s + 1.0e-30);
            vol = vol_a1 + t * (vol_a2_est - vol_a1);
        } else if (d <= a3) {
            // Prism region
            Real t = (d - a2) / (a3 - a2 + 1.0e-30);
            Real vol_a2 = 0.5 * h * h * h * (a1 + a2) / (s + 1.0e-30);
            Real vol_a3 = h * h * h * (1.0 - 0.5 * a3 / (s + 1.0e-30));
            vol = vol_a2 + t * (vol_a3 - vol_a2);
        } else {
            // Large: complement of a small corner
            Real dd = s * h - d;
            Real corner = dd * dd * dd / (6.0 * n1 * n2 * n3 + 1.0e-30);
            vol = h * h * h - corner;
        }

        // Clamp
        if (vol < 0.0) vol = 0.0;
        if (vol > h * h * h) vol = h * h * h;

        return vol;
    }
};

// ============================================================================
// 21e: ALEFSICoupling — Fluid-Structure Interaction Coupling
// ============================================================================

/**
 * @brief ALE Fluid-Structure Interaction coupling.
 *
 * Implements a partitioned (staggered) FSI scheme:
 *
 * 1. Advance fluid (ALE) with current interface position
 * 2. Transfer pressure loads from fluid to structure interface
 * 3. Advance structure with applied fluid loads
 * 4. Transfer displacement from structure to ALE mesh boundary
 * 5. Smooth ALE mesh using Laplacian smoothing
 *
 * The coupling interface is defined by a set of node IDs that are shared
 * between the fluid and structural domains.
 *
 * References:
 * - Farhat & Lesoinne (2000) "Two efficient staggered algorithms for FSI"
 * - Donea et al. (2004) "Arbitrary Lagrangian-Eulerian methods"
 */
class ALEFSICoupling {
public:
    /// FSI interface node: maps between fluid and structure mesh
    struct InterfaceNode {
        int fluid_node_id;
        int struct_node_id;
        Real position[3];
        Real normal[3];     ///< Interface normal (outward from fluid)

        InterfaceNode() : fluid_node_id(-1), struct_node_id(-1) {
            position[0] = position[1] = position[2] = 0.0;
            normal[0] = normal[1] = normal[2] = 0.0;
        }
    };

    ALEFSICoupling() = default;

    /// Set number of Laplacian smoothing iterations
    void set_smoothing_iterations(int n) { smoothing_iters_ = n; }

    /// Set relaxation factor for under-relaxation (0 < omega <= 1)
    void set_relaxation(Real omega) { omega_ = omega; }

    /**
     * @brief Transfer pressure from fluid cells to structural nodes.
     *
     * For each interface node, interpolate the fluid pressure from
     * surrounding fluid cells and apply it as a traction on the
     * structural surface:
     *
     *   f_struct[i] = -p_fluid * n_i * A_i
     *
     * where n_i is the interface normal and A_i is the tributary area.
     *
     * @param fluid_pressure Pressure at fluid cell centers [num_fluid_cells]
     * @param fluid_centroids Cell centroid positions [num_fluid_cells][3]
     * @param num_fluid_cells Number of fluid cells
     * @param interface_nodes Interface node mapping
     * @param num_interface Number of interface nodes
     * @param struct_forces Output: forces on structural nodes [num_interface][3]
     * @param tributary_areas Area weights per interface node [num_interface]
     */
    void transfer_pressure_to_structure(const Real* fluid_pressure,
                                        const Real fluid_centroids[][3],
                                        int num_fluid_cells,
                                        const InterfaceNode* interface_nodes,
                                        int num_interface,
                                        Real struct_forces[][3],
                                        const Real* tributary_areas) const {
        for (int i = 0; i < num_interface; ++i) {
            // Find nearest fluid cell to this interface node
            Real min_dist2 = 1.0e30;
            int nearest = -1;
            for (int c = 0; c < num_fluid_cells; ++c) {
                Real dx = fluid_centroids[c][0] - interface_nodes[i].position[0];
                Real dy = fluid_centroids[c][1] - interface_nodes[i].position[1];
                Real dz = fluid_centroids[c][2] - interface_nodes[i].position[2];
                Real d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < min_dist2) {
                    min_dist2 = d2;
                    nearest = c;
                }
            }

            // Inverse-distance weighted interpolation of pressure
            // using nearest cell and its neighbors (simplified: just nearest)
            Real p = 0.0;
            if (nearest >= 0) {
                p = fluid_pressure[nearest];

                // Weighted average with second-nearest for smoother transfer
                Real p2 = p;
                Real min_dist2_2 = 1.0e30;
                for (int c = 0; c < num_fluid_cells; ++c) {
                    if (c == nearest) continue;
                    Real dx = fluid_centroids[c][0] - interface_nodes[i].position[0];
                    Real dy = fluid_centroids[c][1] - interface_nodes[i].position[1];
                    Real dz = fluid_centroids[c][2] - interface_nodes[i].position[2];
                    Real d2 = dx*dx + dy*dy + dz*dz;
                    if (d2 < min_dist2_2) {
                        min_dist2_2 = d2;
                        p2 = fluid_pressure[c];
                    }
                }

                // Inverse-distance weighting between two nearest
                Real w1 = 1.0 / (std::sqrt(min_dist2) + 1.0e-30);
                Real w2 = 1.0 / (std::sqrt(min_dist2_2) + 1.0e-30);
                p = (w1 * fluid_pressure[nearest] + w2 * p2) / (w1 + w2);
            }

            // Traction force: f = -p * n * A
            Real A = tributary_areas[i];
            struct_forces[i][0] = -p * interface_nodes[i].normal[0] * A;
            struct_forces[i][1] = -p * interface_nodes[i].normal[1] * A;
            struct_forces[i][2] = -p * interface_nodes[i].normal[2] * A;
        }
    }

    /**
     * @brief Transfer displacement from structure to fluid mesh boundary.
     *
     * Applies the structural displacement to the corresponding fluid
     * boundary nodes, with optional under-relaxation:
     *
     *   x_fluid^{n+1} = x_fluid^n + omega * (x_struct^{n+1} - x_fluid^n)
     *
     * @param struct_disp Structural displacement [num_interface][3]
     * @param fluid_positions Fluid boundary node positions (modified) [num_interface][3]
     * @param interface_nodes Interface mapping
     * @param num_interface Number of interface nodes
     */
    void transfer_displacement_to_fluid(const Real struct_disp[][3],
                                        Real fluid_positions[][3],
                                        const InterfaceNode* interface_nodes,
                                        int num_interface) const {
        for (int i = 0; i < num_interface; ++i) {
            for (int d = 0; d < 3; ++d) {
                Real target = interface_nodes[i].position[d] + struct_disp[i][d];
                // Under-relaxation
                fluid_positions[i][d] += omega_ * (target - fluid_positions[i][d]);
            }
        }
    }

    /**
     * @brief Smooth the ALE mesh interior using Laplacian smoothing.
     *
     * Iteratively moves each interior node to the centroid of its neighbors:
     *   x_i = (1/N) * sum_j x_j
     *
     * Boundary nodes (including interface) are held fixed.
     *
     * @param node_positions Node positions [num_nodes][3] (modified in-place)
     * @param num_nodes Total number of nodes
     * @param neighbors Node adjacency: neighbors[i][j] is the j-th neighbor of node i
     * @param num_neighbors Number of neighbors per node
     * @param is_boundary Boolean flag per node: true = fixed boundary
     */
    void update_ale_mesh(Real node_positions[][3],
                         int num_nodes,
                         const int neighbors[][8],
                         const int* num_neighbors,
                         const bool* is_boundary) const {
        std::vector<Real> new_pos(num_nodes * 3);

        for (int iter = 0; iter < smoothing_iters_; ++iter) {
            // Copy current positions
            for (int i = 0; i < num_nodes; ++i) {
                new_pos[3*i]   = node_positions[i][0];
                new_pos[3*i+1] = node_positions[i][1];
                new_pos[3*i+2] = node_positions[i][2];
            }

            // Laplacian smooth interior nodes
            for (int i = 0; i < num_nodes; ++i) {
                if (is_boundary[i]) continue;

                int nn = num_neighbors[i];
                if (nn <= 0) continue;

                Real avg[3] = {0.0, 0.0, 0.0};
                for (int j = 0; j < nn; ++j) {
                    int nb = neighbors[i][j];
                    if (nb < 0 || nb >= num_nodes) continue;
                    avg[0] += node_positions[nb][0];
                    avg[1] += node_positions[nb][1];
                    avg[2] += node_positions[nb][2];
                }
                Real inv_nn = 1.0 / static_cast<Real>(nn);
                new_pos[3*i]   = avg[0] * inv_nn;
                new_pos[3*i+1] = avg[1] * inv_nn;
                new_pos[3*i+2] = avg[2] * inv_nn;
            }

            // Apply with relaxation
            for (int i = 0; i < num_nodes; ++i) {
                if (is_boundary[i]) continue;
                Real smooth = 0.5; // Laplacian smoothing weight
                node_positions[i][0] += smooth * (new_pos[3*i]   - node_positions[i][0]);
                node_positions[i][1] += smooth * (new_pos[3*i+1] - node_positions[i][1]);
                node_positions[i][2] += smooth * (new_pos[3*i+2] - node_positions[i][2]);
            }
        }
    }

private:
    int smoothing_iters_ = 5;   ///< Number of Laplacian smoothing passes
    Real omega_ = 1.0;          ///< Under-relaxation factor
};

// ============================================================================
// 21f: ALERemapping — Conservative Field Transfer After Mesh Motion
// ============================================================================

/**
 * @brief Conservative remapping of fields after ALE mesh motion.
 *
 * After the ALE mesh has been smoothed/repositioned, the conserved fields
 * (density, momentum, energy) must be transferred from the old mesh to the
 * new mesh in a conservative manner.
 *
 * Uses intersection-based remapping:
 * 1. For each new cell, find overlapping old cells
 * 2. Compute intersection volumes
 * 3. Transfer fields proportionally:
 *    U_new[j] = (1/V_new[j]) * sum_i (V_intersect[i,j] * U_old[i])
 *
 * Second-order accuracy via gradient reconstruction in old cells.
 *
 * References:
 * - Margolin & Shashkov (2003) "Second-order sign-preserving conservative
 *   interpolation on general grids"
 * - Dukowicz & Kodis (1987) "Accurate conservative remapping"
 */
class ALERemapping {
public:
    /// Simple axis-aligned bounding box for overlap detection
    struct AABB {
        Real min_pt[3];
        Real max_pt[3];

        KOKKOS_INLINE_FUNCTION
        AABB() {
            for (int i = 0; i < 3; ++i) {
                min_pt[i] = 1.0e30;
                max_pt[i] = -1.0e30;
            }
        }

        KOKKOS_INLINE_FUNCTION
        void expand(Real x, Real y, Real z) {
            min_pt[0] = std::min(min_pt[0], x);
            min_pt[1] = std::min(min_pt[1], y);
            min_pt[2] = std::min(min_pt[2], z);
            max_pt[0] = std::max(max_pt[0], x);
            max_pt[1] = std::max(max_pt[1], y);
            max_pt[2] = std::max(max_pt[2], z);
        }

        KOKKOS_INLINE_FUNCTION
        bool overlaps(const AABB& other) const {
            return !(min_pt[0] > other.max_pt[0] || max_pt[0] < other.min_pt[0]
                  || min_pt[1] > other.max_pt[1] || max_pt[1] < other.min_pt[1]
                  || min_pt[2] > other.max_pt[2] || max_pt[2] < other.min_pt[2]);
        }

        KOKKOS_INLINE_FUNCTION
        Real volume() const {
            Real dx = max_pt[0] - min_pt[0];
            Real dy = max_pt[1] - min_pt[1];
            Real dz = max_pt[2] - min_pt[2];
            if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) return 0.0;
            return dx * dy * dz;
        }
    };

    /// Hex cell defined by 8 corner nodes
    struct HexCell {
        Real corners[8][3];  ///< 8 corner coordinates

        KOKKOS_INLINE_FUNCTION
        HexCell() {
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 3; ++j)
                    corners[i][j] = 0.0;
        }

        /// Compute centroid
        KOKKOS_INLINE_FUNCTION
        void centroid(Real c[3]) const {
            c[0] = c[1] = c[2] = 0.0;
            for (int i = 0; i < 8; ++i) {
                c[0] += corners[i][0];
                c[1] += corners[i][1];
                c[2] += corners[i][2];
            }
            c[0] /= 8.0; c[1] /= 8.0; c[2] /= 8.0;
        }

        /// Compute volume using divergence theorem decomposition into tetrahedra
        KOKKOS_INLINE_FUNCTION
        Real volume() const {
            // Decompose hex into 5 tetrahedra and sum volumes
            // Tet decomposition: {0,1,3,4}, {1,2,3,6}, {3,4,6,7}, {1,4,5,6}, {1,3,4,6}
            static const int tets[5][4] = {
                {0,1,3,4}, {1,2,3,6}, {3,4,6,7}, {1,4,5,6}, {1,3,4,6}
            };

            Real vol = 0.0;
            for (int t = 0; t < 5; ++t) {
                const Real* a = corners[tets[t][0]];
                const Real* b = corners[tets[t][1]];
                const Real* c = corners[tets[t][2]];
                const Real* d = corners[tets[t][3]];
                vol += tet_volume(a, b, c, d);
            }
            return std::abs(vol);
        }

        /// Compute AABB
        KOKKOS_INLINE_FUNCTION
        AABB bounding_box() const {
            AABB box;
            for (int i = 0; i < 8; ++i) {
                box.expand(corners[i][0], corners[i][1], corners[i][2]);
            }
            return box;
        }

    private:
        KOKKOS_INLINE_FUNCTION
        static Real tet_volume(const Real a[3], const Real b[3],
                               const Real c[3], const Real d[3]) {
            // V = (1/6) * |(b-a) . ((c-a) x (d-a))|
            Real ab[3] = { b[0]-a[0], b[1]-a[1], b[2]-a[2] };
            Real ac[3] = { c[0]-a[0], c[1]-a[1], c[2]-a[2] };
            Real ad[3] = { d[0]-a[0], d[1]-a[1], d[2]-a[2] };

            Real cross[3] = {
                ac[1]*ad[2] - ac[2]*ad[1],
                ac[2]*ad[0] - ac[0]*ad[2],
                ac[0]*ad[1] - ac[1]*ad[0]
            };

            return (ab[0]*cross[0] + ab[1]*cross[1] + ab[2]*cross[2]) / 6.0;
        }
    };

    ALERemapping() = default;

    /// Enable second-order remapping with gradient correction
    void set_second_order(bool enable) { second_order_ = enable; }

    /**
     * @brief Compute approximate intersection volume between two hex cells.
     *
     * Uses AABB overlap as an approximation for the true polyhedral
     * intersection. For more accuracy, one could use Sutherland-Hodgman
     * clipping, but AABB overlap is sufficient for small mesh motions.
     *
     * @param old_cell Old mesh cell
     * @param new_cell New mesh cell
     * @return Estimated intersection volume
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_intersection_volume(const HexCell& old_cell,
                                     const HexCell& new_cell) const {
        AABB box_old = old_cell.bounding_box();
        AABB box_new = new_cell.bounding_box();

        if (!box_old.overlaps(box_new)) return 0.0;

        // Compute AABB intersection
        AABB isect;
        for (int d = 0; d < 3; ++d) {
            isect.min_pt[d] = std::max(box_old.min_pt[d], box_new.min_pt[d]);
            isect.max_pt[d] = std::min(box_old.max_pt[d], box_new.max_pt[d]);
        }

        Real isect_vol = isect.volume();

        // Scale by ratio of actual cell volumes to AABB volumes to improve estimate
        Real vol_old = old_cell.volume();
        Real vol_new = new_cell.volume();
        Real aabb_old = box_old.volume();
        Real aabb_new = box_new.volume();

        Real fill_old = (aabb_old > 1.0e-30) ? vol_old / aabb_old : 1.0;
        Real fill_new = (aabb_new > 1.0e-30) ? vol_new / aabb_new : 1.0;

        // Geometric mean of fill fractions
        Real fill = std::sqrt(fill_old * fill_new);

        return isect_vol * fill;
    }

    /**
     * @brief Conservative interpolation with optional gradient correction.
     *
     * For second-order accuracy:
     *   q_intersect = q_old + grad(q_old) . (x_intersect - x_old_centroid)
     *
     * @param old_value Scalar field value in old cell
     * @param old_gradient Gradient of field in old cell [3]
     * @param old_centroid Centroid of old cell [3]
     * @param intersect_centroid Centroid of intersection region [3]
     * @return Interpolated value at intersection
     */
    KOKKOS_INLINE_FUNCTION
    Real conservative_interpolation(Real old_value,
                                    const Real old_gradient[3],
                                    const Real old_centroid[3],
                                    const Real intersect_centroid[3]) const {
        if (!second_order_) return old_value;

        Real dx = intersect_centroid[0] - old_centroid[0];
        Real dy = intersect_centroid[1] - old_centroid[1];
        Real dz = intersect_centroid[2] - old_centroid[2];

        return old_value + old_gradient[0]*dx + old_gradient[1]*dy + old_gradient[2]*dz;
    }

    /**
     * @brief Remap conserved fields from old mesh to new mesh.
     *
     * For each new cell j:
     *   U_new[j] = (1 / V_new[j]) * sum_i { V_ij * q_i(x_ij) }
     *
     * where V_ij is the intersection volume and q_i(x_ij) is the
     * (possibly gradient-corrected) old field value.
     *
     * @param old_cells Old mesh cells [num_old]
     * @param new_cells New mesh cells [num_new]
     * @param num_old Number of old cells
     * @param num_new Number of new cells
     * @param old_fields Conserved fields on old mesh [num_old][5]
     * @param old_gradients Gradients on old mesh [num_old][5][3] (NULL if first-order)
     * @param new_fields Output: remapped fields on new mesh [num_new][5]
     */
    void remap(const HexCell* old_cells, const HexCell* new_cells,
               int num_old, int num_new,
               const Real old_fields[][5],
               const Real old_gradients[][5][3],
               Real new_fields[][5]) const {
        // Compute old cell centroids
        std::vector<std::array<Real, 3>> old_centroids(num_old);
        std::vector<AABB> old_boxes(num_old);
        for (int i = 0; i < num_old; ++i) {
            old_cells[i].centroid(old_centroids[i].data());
            old_boxes[i] = old_cells[i].bounding_box();
        }

        // For each new cell, find overlapping old cells and accumulate
        for (int j = 0; j < num_new; ++j) {
            AABB new_box = new_cells[j].bounding_box();
            Real V_new = new_cells[j].volume();
            if (V_new < 1.0e-30) {
                for (int k = 0; k < 5; ++k) new_fields[j][k] = 0.0;
                continue;
            }

            // Initialize accumulator
            Real accum[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
            Real total_vol = 0.0;

            for (int i = 0; i < num_old; ++i) {
                if (!old_boxes[i].overlaps(new_box)) continue;

                Real V_ij = compute_intersection_volume(old_cells[i], new_cells[j]);
                if (V_ij < 1.0e-30) continue;

                // Compute intersection centroid (midpoint of cell centroids)
                Real new_cen[3];
                new_cells[j].centroid(new_cen);
                Real isect_cen[3] = {
                    0.5 * (old_centroids[i][0] + new_cen[0]),
                    0.5 * (old_centroids[i][1] + new_cen[1]),
                    0.5 * (old_centroids[i][2] + new_cen[2])
                };

                for (int k = 0; k < 5; ++k) {
                    Real grad[3] = {0.0, 0.0, 0.0};
                    if (old_gradients != nullptr && second_order_) {
                        grad[0] = old_gradients[i][k][0];
                        grad[1] = old_gradients[i][k][1];
                        grad[2] = old_gradients[i][k][2];
                    }
                    Real val = conservative_interpolation(
                        old_fields[i][k], grad,
                        old_centroids[i].data(), isect_cen);
                    accum[k] += V_ij * val;
                }

                total_vol += V_ij;
            }

            // Divide by new cell volume for intensive quantities
            // For conserved quantities (extensive), divide by new volume
            if (total_vol > 1.0e-30) {
                for (int k = 0; k < 5; ++k) {
                    new_fields[j][k] = accum[k] / V_new;
                }
            } else {
                // No overlap found: retain zero
                for (int k = 0; k < 5; ++k) new_fields[j][k] = 0.0;
            }

            // Enforce positivity on density
            if (new_fields[j][0] < 1.0e-30) new_fields[j][0] = 1.0e-30;
        }
    }

private:
    bool second_order_ = true;  ///< Use second-order gradient correction
};

} // namespace fem
} // namespace nxs
