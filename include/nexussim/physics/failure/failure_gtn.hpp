#pragma once

#include <nexussim/physics/failure/failure_model.hpp>

namespace nxs {
namespace physics {
namespace failure {

/**
 * @brief Gurson-Tvergaard-Needleman (GTN) ductile damage model
 *
 * Models ductile fracture through void growth, nucleation, and coalescence.
 *
 * Yield surface:
 *   Φ = (σ_vm/σ_y)² + 2*q1*f*·cosh(3*q2*σ_m/(2*σ_y)) - (1 + q3*f*²) = 0
 *
 * Where:
 *   f* = f                     if f < fc
 *   f* = fc + (1/q1 - fc)/(fF - fc) * (f - fc)  if f >= fc (coalescence)
 *
 * Void evolution: df = df_growth + df_nucleation
 *   df_growth = (1-f) * dε_kk^p
 *   df_nucleation = A * dε_p where A = (fN/(sN*√(2π))) * exp(-0.5*((εp-εN)/sN)²)
 *
 * Failure state history:
 *   history[0]: void volume fraction f
 *   history[1]: effective f* (modified void fraction)
 *   history[2]: accumulated plastic strain for nucleation
 *   history[3]: nucleated void fraction
 *
 * Reference: Tvergaard & Needleman (1984), Acta Metallurgica 32(1)
 */
class GTNFailure : public FailureModel {
public:
    GTNFailure(const FailureModelParameters& params)
        : FailureModel(FailureModelType::GTN, params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        const Real f0 = params_.f0;
        const Real fN = params_.fN;
        const Real sN = params_.sN;
        const Real epsN = params_.epsN;
        const Real fc = params_.fc;
        const Real fF = params_.fF;
        const Real q1 = params_.q1;

        // Current void fraction
        Real f = fstate.history[0];
        if (f < f0) f = f0;  // Initialize

        // Volumetric plastic strain increment (approximate from total strain)
        Real ev_plastic = mstate.vol_strain;

        // Plastic strain for nucleation
        Real eps_p = mstate.plastic_strain;
        Real eps_p_old = fstate.history[2];
        Real delta_eps_p = eps_p - eps_p_old;
        if (delta_eps_p < 0.0) delta_eps_p = 0.0;
        fstate.history[2] = eps_p;

        // Void growth (from volumetric plastic strain)
        Real df_growth = (1.0 - f) * Kokkos::fabs(ev_plastic) * 0.01;  // Simplified

        // Void nucleation (strain-controlled)
        Real A = 0.0;
        if (sN > 1.0e-20) {
            Real z = (eps_p - epsN) / sN;
            const Real inv_sqrt_2pi = 0.3989422804;
            A = (fN / sN) * inv_sqrt_2pi * Kokkos::exp(-0.5 * z * z);
        }
        Real df_nucleation = A * delta_eps_p;
        fstate.history[3] += df_nucleation;

        // Total void evolution
        f += df_growth + df_nucleation;
        if (f > 1.0) f = 1.0;
        fstate.history[0] = f;

        // Effective void fraction f* (coalescence model)
        Real f_star;
        if (f < fc) {
            f_star = f;
        } else {
            // Accelerated void growth after coalescence
            Real kappa = (1.0/q1 - fc) / (fF - fc);
            f_star = fc + kappa * (f - fc);
        }
        fstate.history[1] = f_star;

        // Damage = f* / (1/q1)  (normalized)
        Real f_failure = 1.0 / q1;
        fstate.damage = f_star / f_failure;
        if (fstate.damage > 1.0) fstate.damage = 1.0;

        if (f >= fF || f_star >= f_failure) {
            fstate.failed = true;
            fstate.failure_mode = 1;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& params) {
        return std::make_unique<GTNFailure>(params);
    }
};

} // namespace failure
} // namespace physics
} // namespace nxs
