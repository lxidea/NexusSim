#pragma once

/**
 * @file composite_utils.hpp
 * @brief Shared utility functions for composite laminate analysis
 *
 * Provides standalone versions of CLT helper functions (Q-bar computation,
 * coordinate transforms, z-coordinates, 6x6 matrix inversion) for use by
 * thermal, interlaminar, and progressive failure modules.
 */

#include <nexussim/physics/composite_layup.hpp>
#include <cmath>

namespace nxs {
namespace physics {
namespace composite_detail {

/**
 * @brief Compute transformed reduced stiffness matrix Q-bar for a ply
 */
inline void compute_Qbar(const PlyDefinition& ply, Real* Qbar) {
    Real E1 = ply.E1;
    Real E2 = ply.E2;
    Real G12 = ply.G12;
    Real nu12 = ply.nu12;
    Real nu21 = nu12 * E2 / E1;

    Real denom = 1.0 - nu12 * nu21;
    Real Q11 = E1 / denom;
    Real Q22 = E2 / denom;
    Real Q12 = nu12 * E2 / denom;
    Real Q66 = G12;

    Real theta = ply.angle * constants::pi<Real> / 180.0;
    Real m = std::cos(theta);
    Real n = std::sin(theta);

    Real m2 = m * m, n2 = n * n;
    Real m4 = m2 * m2, n4 = n2 * n2;
    Real m2n2 = m2 * n2;

    Qbar[0] = Q11 * m4 + 2.0 * (Q12 + 2.0 * Q66) * m2n2 + Q22 * n4;
    Qbar[1] = (Q11 + Q22 - 4.0 * Q66) * m2n2 + Q12 * (m4 + n4);
    Qbar[2] = (Q11 - Q12 - 2.0 * Q66) * m * m2 * n + (Q12 - Q22 + 2.0 * Q66) * m * n * n2;
    Qbar[3] = Qbar[1];
    Qbar[4] = Q11 * n4 + 2.0 * (Q12 + 2.0 * Q66) * m2n2 + Q22 * m4;
    Qbar[5] = (Q11 - Q12 - 2.0 * Q66) * m * n * n2 + (Q12 - Q22 + 2.0 * Q66) * m * m2 * n;
    Qbar[6] = Qbar[2];
    Qbar[7] = Qbar[5];
    Qbar[8] = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * m2n2 + Q66 * (m4 + n4);
}

/**
 * @brief Transform stress/strain from global to local (material) coordinates
 */
inline void transform_to_local(Real theta, const Real* global, Real* local) {
    Real m = std::cos(theta);
    Real n = std::sin(theta);
    Real m2 = m * m, n2 = n * n, mn = m * n;

    local[0] = m2 * global[0] + n2 * global[1] + 2.0 * mn * global[2];
    local[1] = n2 * global[0] + m2 * global[1] - 2.0 * mn * global[2];
    local[2] = -mn * global[0] + mn * global[1] + (m2 - n2) * global[2];
}

/**
 * @brief Transform stress/strain from local (material) to global coordinates
 */
inline void transform_to_global(Real theta, const Real* local, Real* global) {
    Real m = std::cos(theta);
    Real n = std::sin(theta);
    Real m2 = m * m, n2 = n * n, mn = m * n;

    global[0] = m2 * local[0] + n2 * local[1] - 2.0 * mn * local[2];
    global[1] = n2 * local[0] + m2 * local[1] + 2.0 * mn * local[2];
    global[2] = mn * local[0] - mn * local[1] + (m2 - n2) * local[2];
}

/**
 * @brief Compute ply z-coordinates (bottom/top) for a laminate
 */
inline void compute_z_coords(const CompositeLaminate& lam,
                              Real* z_bottom, Real* z_top) {
    Real z_bot = -lam.total_thickness() / 2.0;
    for (int k = 0; k < lam.num_plies(); ++k) {
        z_bottom[k] = z_bot;
        z_bot += lam.ply(k).thickness;
        z_top[k] = z_bot;
    }
}

/**
 * @brief Invert a 6x6 matrix using Gauss-Jordan elimination
 * @return true if successful, false if singular
 */
inline bool invert_6x6(const Real* A, Real* A_inv) {
    // Augmented matrix [A | I]
    Real aug[6][12];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            aug[i][j] = A[i * 6 + j];
            aug[i][j + 6] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int col = 0; col < 6; ++col) {
        // Partial pivoting
        int max_row = col;
        Real max_val = std::fabs(aug[col][col]);
        for (int row = col + 1; row < 6; ++row) {
            if (std::fabs(aug[row][col]) > max_val) {
                max_val = std::fabs(aug[row][col]);
                max_row = row;
            }
        }
        if (max_val < 1.0e-30) return false;

        if (max_row != col) {
            for (int j = 0; j < 12; ++j)
                std::swap(aug[col][j], aug[max_row][j]);
        }

        Real pivot = aug[col][col];
        for (int j = 0; j < 12; ++j)
            aug[col][j] /= pivot;

        for (int row = 0; row < 6; ++row) {
            if (row == col) continue;
            Real factor = aug[row][col];
            for (int j = 0; j < 12; ++j)
                aug[row][j] -= factor * aug[col][j];
        }
    }

    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            A_inv[i * 6 + j] = aug[i][j + 6];

    return true;
}

/**
 * @brief Invert a 3x3 matrix
 * @return true if successful
 */
inline bool invert_3x3(const Real* A, Real* A_inv) {
    Real det = A[0] * (A[4] * A[8] - A[5] * A[7])
             - A[1] * (A[3] * A[8] - A[5] * A[6])
             + A[2] * (A[3] * A[7] - A[4] * A[6]);
    if (std::fabs(det) < 1.0e-30) return false;
    Real inv_det = 1.0 / det;

    A_inv[0] = (A[4] * A[8] - A[5] * A[7]) * inv_det;
    A_inv[1] = (A[2] * A[7] - A[1] * A[8]) * inv_det;
    A_inv[2] = (A[1] * A[5] - A[2] * A[4]) * inv_det;
    A_inv[3] = (A[5] * A[6] - A[3] * A[8]) * inv_det;
    A_inv[4] = (A[0] * A[8] - A[2] * A[6]) * inv_det;
    A_inv[5] = (A[2] * A[3] - A[0] * A[5]) * inv_det;
    A_inv[6] = (A[3] * A[7] - A[4] * A[6]) * inv_det;
    A_inv[7] = (A[1] * A[6] - A[0] * A[7]) * inv_det;
    A_inv[8] = (A[0] * A[4] - A[1] * A[3]) * inv_det;
    return true;
}

/**
 * @brief Transform CTE from material to global coordinates
 *
 * alpha_bar = T^{-T} * alpha_local
 * For plane stress: alpha_xx = alpha1*m^2 + alpha2*n^2
 *                   alpha_yy = alpha1*n^2 + alpha2*m^2
 *                   alpha_xy = 2*(alpha1 - alpha2)*m*n
 */
inline void transform_cte(Real theta_deg, Real alpha1, Real alpha2, Real* alpha_bar) {
    Real theta = theta_deg * constants::pi<Real> / 180.0;
    Real m = std::cos(theta);
    Real n = std::sin(theta);
    Real m2 = m * m, n2 = n * n;

    alpha_bar[0] = alpha1 * m2 + alpha2 * n2;
    alpha_bar[1] = alpha1 * n2 + alpha2 * m2;
    alpha_bar[2] = 2.0 * (alpha1 - alpha2) * m * n;
}

} // namespace composite_detail
} // namespace physics
} // namespace nxs
