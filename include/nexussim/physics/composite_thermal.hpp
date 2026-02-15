#pragma once

/**
 * @file composite_thermal.hpp
 * @brief Thermal residual stress analysis for composite laminates
 *
 * Computes thermal stresses arising from CTE mismatch between plies
 * during cool-down from cure temperature to service temperature.
 *
 * Algorithm:
 *   1. Compute thermal resultants: N_T = sum Q_bar_k * alpha_bar_k * dT * t_k
 *   2. Solve ABD^{-1} * {N_T, M_T} = {eps0_T, kappa_T}
 *   3. Ply stresses: sigma_k = Q_bar_k * (eps0 + z*kappa - alpha_bar_k * dT)
 *
 * Reference: Jones, "Mechanics of Composite Materials", Chapter 4
 */

#include <nexussim/physics/composite_layup.hpp>
#include <nexussim/physics/composite_utils.hpp>
#include <cmath>

namespace nxs {
namespace physics {

struct PlyThermalProperties {
    Real alpha1;  ///< CTE in fiber direction (1/K)
    Real alpha2;  ///< CTE in transverse direction (1/K)

    PlyThermalProperties() : alpha1(0.0), alpha2(0.0) {}
    PlyThermalProperties(Real a1, Real a2) : alpha1(a1), alpha2(a2) {}
};

class CompositeThermalAnalysis {
public:
    static constexpr int MAX_PLIES = CompositeLaminate::MAX_PLIES;

    CompositeThermalAnalysis() : T_cure_(0.0), T_service_(0.0) {
        for (int i = 0; i < MAX_PLIES; ++i) {
            ply_cte_[i] = PlyThermalProperties();
        }
    }

    void set_ply_cte(int ply_index, Real alpha1, Real alpha2) {
        if (ply_index >= 0 && ply_index < MAX_PLIES) {
            ply_cte_[ply_index] = PlyThermalProperties(alpha1, alpha2);
        }
    }

    void set_all_ply_cte(int num_plies, Real alpha1, Real alpha2) {
        for (int i = 0; i < num_plies && i < MAX_PLIES; ++i) {
            ply_cte_[i] = PlyThermalProperties(alpha1, alpha2);
        }
    }

    void set_temperatures(Real T_cure, Real T_service) {
        T_cure_ = T_cure;
        T_service_ = T_service;
    }

    Real delta_T() const { return T_service_ - T_cure_; }

    /**
     * @brief Compute thermal force and moment resultants
     *
     * N_T_i = sum_k Q_bar_ij_k * alpha_bar_j_k * dT * t_k
     * M_T_i = sum_k Q_bar_ij_k * alpha_bar_j_k * dT * z_mid_k * t_k
     */
    void compute_thermal_resultants(const CompositeLaminate& lam,
                                     Real* N_T, Real* M_T) const {
        Real dT = delta_T();
        for (int i = 0; i < 3; ++i) { N_T[i] = 0.0; M_T[i] = 0.0; }

        Real z_bottom[MAX_PLIES], z_top[MAX_PLIES];
        composite_detail::compute_z_coords(lam, z_bottom, z_top);

        for (int k = 0; k < lam.num_plies(); ++k) {
            Real Qbar[9];
            composite_detail::compute_Qbar(lam.ply(k), Qbar);

            Real alpha_bar[3];
            composite_detail::transform_cte(lam.ply(k).angle,
                                             ply_cte_[k].alpha1,
                                             ply_cte_[k].alpha2,
                                             alpha_bar);

            Real zb = z_bottom[k];
            Real zt = z_top[k];
            Real dz = zt - zb;
            Real dz2 = zt * zt - zb * zb;

            // Q_bar * alpha_bar * dT
            Real Qa[3];
            for (int i = 0; i < 3; ++i) {
                Qa[i] = 0.0;
                for (int j = 0; j < 3; ++j) {
                    Qa[i] += Qbar[i * 3 + j] * alpha_bar[j];
                }
                Qa[i] *= dT;
            }

            for (int i = 0; i < 3; ++i) {
                N_T[i] += Qa[i] * dz;
                M_T[i] += 0.5 * Qa[i] * dz2;
            }
        }
    }

    /**
     * @brief Compute thermal midplane deformation by solving ABD * {eps,kappa} = {N_T, M_T}
     */
    void compute_thermal_deformation(const CompositeLaminate& lam,
                                      Real* eps0_T, Real* kappa_T) const {
        Real N_T[3], M_T[3];
        compute_thermal_resultants(lam, N_T, M_T);

        // Build 6x6 ABD matrix
        Real abd[36];
        lam.get_abd_matrix(abd);

        // Invert ABD
        Real abd_inv[36];
        composite_detail::invert_6x6(abd, abd_inv);

        // {eps0, kappa} = ABD^{-1} * {N_T, M_T}
        Real rhs[6] = {N_T[0], N_T[1], N_T[2], M_T[0], M_T[1], M_T[2]};
        Real sol[6] = {0, 0, 0, 0, 0, 0};

        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                sol[i] += abd_inv[i * 6 + j] * rhs[j];
            }
        }

        for (int i = 0; i < 3; ++i) {
            eps0_T[i] = sol[i];
            kappa_T[i] = sol[i + 3];
        }
    }

    /**
     * @brief Compute thermal ply stresses
     *
     * sigma_k = Q_bar_k * (eps0_T + z_k * kappa_T - alpha_bar_k * dT)
     */
    void compute_thermal_ply_stresses(const CompositeLaminate& lam,
                                       PlyState* ply_states) const {
        Real eps0_T[3], kappa_T[3];
        compute_thermal_deformation(lam, eps0_T, kappa_T);

        Real dT = delta_T();
        Real z_bottom[MAX_PLIES], z_top[MAX_PLIES];
        composite_detail::compute_z_coords(lam, z_bottom, z_top);

        for (int k = 0; k < lam.num_plies(); ++k) {
            Real z_mid = (z_bottom[k] + z_top[k]) / 2.0;
            ply_states[k].z_position = z_mid;

            Real Qbar[9];
            composite_detail::compute_Qbar(lam.ply(k), Qbar);

            Real alpha_bar[3];
            composite_detail::transform_cte(lam.ply(k).angle,
                                             ply_cte_[k].alpha1,
                                             ply_cte_[k].alpha2,
                                             alpha_bar);

            // Mechanical strain = total strain - thermal strain
            Real mech_strain[3];
            for (int i = 0; i < 3; ++i) {
                Real total = eps0_T[i] + z_mid * kappa_T[i];
                mech_strain[i] = total - alpha_bar[i] * dT;
                ply_states[k].strain_global[i] = total;
            }

            // Global stress
            for (int i = 0; i < 3; ++i) {
                ply_states[k].stress_global[i] = 0.0;
                for (int j = 0; j < 3; ++j) {
                    ply_states[k].stress_global[i] += Qbar[i * 3 + j] * mech_strain[j];
                }
            }

            // Transform to local
            Real theta = lam.ply(k).angle * constants::pi<Real> / 180.0;
            composite_detail::transform_to_local(theta,
                                                  ply_states[k].stress_global,
                                                  ply_states[k].stress_local);
            composite_detail::transform_to_local(theta,
                                                  ply_states[k].strain_global,
                                                  ply_states[k].strain_local);
        }
    }

private:
    PlyThermalProperties ply_cte_[MAX_PLIES];
    Real T_cure_;
    Real T_service_;
};

} // namespace physics
} // namespace nxs
