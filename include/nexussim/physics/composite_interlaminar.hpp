#pragma once

/**
 * @file composite_interlaminar.hpp
 * @brief Interlaminar shear stress computation for composite laminates
 *
 * Computes interlaminar shear stresses (tau_xz, tau_yz) at ply interfaces
 * using the equilibrium approach (composite beam analogy).
 *
 * Algorithm (composite VQ/Ib):
 *   For each interface i between ply k and k+1:
 *     Q_x(z_i) = sum_{k=0..i} Q_bar_11_k * z_mid_k * t_k  (first moment)
 *     tau_xz(z_i) = Vx * Q_x(z_i) / D11
 *   Similarly for tau_yz using Q_bar_22 and D22.
 *
 * Reference: Whitney, "Structural Analysis of Laminated Anisotropic Plates", Ch 3
 */

#include <nexussim/physics/composite_layup.hpp>
#include <nexussim/physics/composite_utils.hpp>
#include <cmath>
#include <algorithm>

namespace nxs {
namespace physics {

struct InterlaminarStress {
    Real tau_xz;           ///< Interlaminar shear stress in xz-plane
    Real tau_yz;           ///< Interlaminar shear stress in yz-plane
    Real z_interface;      ///< z-coordinate of the interface
    int lower_ply;         ///< Ply index below interface
    int upper_ply;         ///< Ply index above interface

    InterlaminarStress()
        : tau_xz(0.0), tau_yz(0.0), z_interface(0.0)
        , lower_ply(-1), upper_ply(-1) {}
};

class CompositeInterlaminarAnalysis {
public:
    static constexpr int MAX_PLIES = CompositeLaminate::MAX_PLIES;

    /**
     * @brief Compute interlaminar shear stresses at all ply interfaces
     *
     * @param lam Laminate (must have compute_abd() called)
     * @param V Transverse shear forces [Vx, Vy]
     * @param stresses Output: interlaminar stresses at each interface
     * @return Number of interfaces (num_plies - 1)
     */
    int compute_interlaminar_shear(const CompositeLaminate& lam,
                                    const Real* V,
                                    InterlaminarStress* stresses) const {
        int np = lam.num_plies();
        if (np < 2) return 0;

        Real z_bottom[MAX_PLIES], z_top[MAX_PLIES];
        composite_detail::compute_z_coords(lam, z_bottom, z_top);

        Real D11 = lam.D()[0];
        Real D22 = lam.D()[4];

        if (std::fabs(D11) < 1.0e-30 || std::fabs(D22) < 1.0e-30) return 0;

        int num_interfaces = np - 1;

        for (int iface = 0; iface < num_interfaces; ++iface) {
            stresses[iface].z_interface = z_top[iface];
            stresses[iface].lower_ply = iface;
            stresses[iface].upper_ply = iface + 1;

            // Compute first moment Q(z) = sum_{k=0..iface} Q_bar_ij * z_mid * t_k
            // Using simplified beam-analogy approach
            Real Qx = 0.0;
            Real Qy = 0.0;

            for (int k = 0; k <= iface; ++k) {
                Real Qbar[9];
                composite_detail::compute_Qbar(lam.ply(k), Qbar);

                Real z_mid = (z_bottom[k] + z_top[k]) / 2.0;
                Real t_k = lam.ply(k).thickness;

                // First moment contribution using Qbar_11 and Qbar_22
                // Q_x = sum Q_bar_11_k * z_mid_k * t_k
                Qx += Qbar[0] * z_mid * t_k;
                Qy += Qbar[4] * z_mid * t_k;
            }

            stresses[iface].tau_xz = V[0] * Qx / D11;
            stresses[iface].tau_yz = V[1] * Qy / D22;
        }

        return num_interfaces;
    }

    /**
     * @brief Find maximum interlaminar shear stress magnitude
     */
    Real max_interlaminar_shear(const CompositeLaminate& lam,
                                 const Real* V) const {
        int np = lam.num_plies();
        if (np < 2) return 0.0;

        InterlaminarStress stresses[MAX_PLIES];
        int n = compute_interlaminar_shear(lam, V, stresses);

        Real max_tau = 0.0;
        for (int i = 0; i < n; ++i) {
            Real mag = std::sqrt(stresses[i].tau_xz * stresses[i].tau_xz
                               + stresses[i].tau_yz * stresses[i].tau_yz);
            if (mag > max_tau) max_tau = mag;
        }
        return max_tau;
    }
};

} // namespace physics
} // namespace nxs
