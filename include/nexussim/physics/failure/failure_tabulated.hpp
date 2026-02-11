#pragma once

#include <nexussim/physics/failure/failure_model.hpp>

namespace nxs {
namespace physics {
namespace failure {

/**
 * @brief Tabulated failure envelope (triaxiality-dependent failure strain)
 *
 * User-defined failure strain as a function of stress triaxiality η.
 * The failure envelope eps_f(η) is specified via TabulatedCurve.
 *
 * Damage accumulation:
 *   D += Δε_p / ε_f(η)
 *
 * Where η = σ_m / σ_vm (triaxiality), computed each step.
 *
 * This is a simplified version of GISSMO without mesh regularization
 * or stress fading, suitable for quick failure assessment.
 *
 * Failure state history:
 *   history[0]: accumulated damage D
 *   history[1]: accumulated plastic strain
 *   history[2]: current triaxiality
 *   history[3]: current failure strain
 *
 * Reference: Bao & Wierzbicki (2004), Int. J. Mechanical Sciences
 */
class TabulatedFailure : public FailureModel {
public:
    TabulatedFailure(const FailureModelParameters& params)
        : FailureModel(FailureModelType::TabulatedEnvelope, params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // Current triaxiality
        Real sigma_m = (mstate.stress[0] + mstate.stress[1] + mstate.stress[2]) / 3.0;
        Real sigma_vm = Material::von_mises_stress(mstate.stress);
        Real eta = (sigma_vm > 1.0e-20) ? sigma_m / sigma_vm : 0.0;
        fstate.history[2] = eta;

        // Failure strain from envelope
        Real eps_f;
        if (params_.failure_envelope.num_points > 0) {
            eps_f = params_.failure_envelope.evaluate(eta);
        } else {
            // Default Bao-Wierzbicki type curve
            if (eta < -1.0/3.0) {
                eps_f = 1.0;  // Very ductile under compression
            } else if (eta < 0.0) {
                eps_f = 0.5 + 1.5 * (-eta);  // Transition
            } else {
                eps_f = 0.5 * Kokkos::exp(-1.5 * eta);
            }
        }
        if (eps_f < 0.01) eps_f = 0.01;
        fstate.history[3] = eps_f;

        // Plastic strain increment
        Real eps_p = mstate.plastic_strain;
        Real eps_p_old = fstate.history[1];
        Real delta_eps_p = eps_p - eps_p_old;
        if (delta_eps_p < 0.0) delta_eps_p = 0.0;
        fstate.history[1] = eps_p;

        // Linear damage accumulation
        Real dD = delta_eps_p / eps_f;
        fstate.history[0] += dD;

        fstate.damage = fstate.history[0];
        if (fstate.damage > 1.0) fstate.damage = 1.0;

        if (fstate.history[0] >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& params) {
        return std::make_unique<TabulatedFailure>(params);
    }
};

} // namespace failure
} // namespace physics
} // namespace nxs
