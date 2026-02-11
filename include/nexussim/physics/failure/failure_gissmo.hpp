#pragma once

#include <nexussim/physics/failure/failure_model.hpp>

namespace nxs {
namespace physics {
namespace failure {

/**
 * @brief GISSMO (Generalized Incremental Stress-State dependent damage MOdel)
 *
 * Mesh-regularized damage model widely used in automotive crash simulation.
 *
 * Damage evolution:
 *   dD = (n/eps_f(η)) * D^((n-1)/n) * dε_p
 *
 * Where:
 *   eps_f(η) = failure strain as function of triaxiality (tabulated)
 *   n = damage exponent (controls nonlinearity)
 *   η = stress triaxiality = σ_m / σ_vm
 *
 * Mesh regularization:
 *   eps_f_reg = eps_f * (lc_ref / lc)^(1/n)
 *
 * Stress fading after instability:
 *   σ_eff = σ * (1 - ((D - D_crit)/(1 - D_crit))^fadexp)
 *
 * Failure state history:
 *   history[0]: damage D (0 to 1)
 *   history[1]: instability indicator F (0 to 1)
 *   history[2]: accumulated plastic strain
 *   history[3]: current triaxiality
 *
 * Reference: Neukamm et al. (2008), LS-DYNA Anwenderforum
 */
class GISSMOFailure : public FailureModel {
public:
    GISSMOFailure(const FailureModelParameters& params)
        : FailureModel(FailureModelType::GISSMO, params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real element_size) const override {
        const Real n = params_.n_exp;
        const Real dcrit = params_.dcrit;
        const Real fadexp = params_.fadexp;
        const Real lc_ref = params_.lc_ref;

        // Current triaxiality
        Real sigma_m = (mstate.stress[0] + mstate.stress[1] + mstate.stress[2]) / 3.0;
        Real sigma_vm = Material::von_mises_stress(mstate.stress);
        Real eta = (sigma_vm > 1.0e-20) ? sigma_m / sigma_vm : 0.0;
        fstate.history[3] = eta;

        // Failure strain from triaxiality (tabulated or constant)
        Real eps_f;
        if (params_.failure_envelope.num_points > 0) {
            eps_f = params_.failure_envelope.evaluate(eta);
        } else {
            // Default: simple triaxiality dependence
            eps_f = 0.5 * Kokkos::exp(-1.5 * eta);
            if (eps_f < 0.01) eps_f = 0.01;
        }

        // Mesh regularization
        Real lc = (element_size > 1.0e-20) ? element_size : lc_ref;
        Real reg_factor = Kokkos::pow(lc_ref / lc, 1.0 / n);
        Real eps_f_reg = eps_f * reg_factor;
        if (eps_f_reg < 0.01) eps_f_reg = 0.01;

        // Plastic strain increment
        Real eps_p = mstate.plastic_strain;
        Real eps_p_old = fstate.history[2];
        Real delta_eps_p = eps_p - eps_p_old;
        if (delta_eps_p < 0.0) delta_eps_p = 0.0;
        fstate.history[2] = eps_p;

        // Damage evolution: dD = (n/eps_f) * D^((n-1)/n) * d_eps_p
        Real D = fstate.history[0];
        if (D < 1.0e-10) D = 1.0e-10;  // Avoid zero for power law

        Real D_power = Kokkos::pow(D, (n - 1.0) / n);
        Real dD = (n / eps_f_reg) * D_power * delta_eps_p;

        D = fstate.history[0] + dD;
        if (D > 1.0) D = 1.0;
        fstate.history[0] = D;

        // Instability indicator (same form but tracks forming limit)
        Real F = fstate.history[1];
        if (F < 1.0e-10) F = 1.0e-10;
        Real F_power = Kokkos::pow(F, (n - 1.0) / n);
        Real dF = (n / eps_f_reg) * F_power * delta_eps_p;
        F = fstate.history[1] + dF;
        if (F > 1.0) F = 1.0;
        fstate.history[1] = F;

        // Apply stress fading after instability onset
        if (D >= dcrit) {
            Real D_fading = (D - dcrit) / (1.0 - dcrit + 1.0e-20);
            Real fade = Kokkos::pow(D_fading, fadexp);
            fstate.damage = Kokkos::fmin(fade, 1.0);
        } else {
            fstate.damage = 0.0;  // No stress reduction before dcrit
        }

        if (D >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1;
            fstate.damage = 1.0;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& params) {
        return std::make_unique<GISSMOFailure>(params);
    }
};

} // namespace failure
} // namespace physics
} // namespace nxs
