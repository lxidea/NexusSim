#pragma once

#include <nexussim/physics/failure/failure_model.hpp>

namespace nxs {
namespace physics {
namespace failure {

/**
 * @brief Tsai-Wu polynomial failure criterion for composites
 *
 * General polynomial failure surface:
 *   F = F1*σ1 + F2*σ2 + F11*σ1² + F22*σ2² + F66*τ12² + 2*F12*σ1*σ2 ≥ 1
 *
 * Where:
 *   F1 = 1/Xt - 1/Xc,  F2 = 1/Yt - 1/Yc
 *   F11 = 1/(Xt*Xc),   F22 = 1/(Yt*Yc),   F66 = 1/S12²
 *   F12 = F12_star * sqrt(F11*F22)
 *
 * Failure state history:
 *   history[0]: Tsai-Wu failure index F
 *   history[1]: max failure index reached
 *
 * Reference: Tsai & Wu (1971), "A general theory of strength for anisotropic materials"
 */
class TsaiWuFailure : public FailureModel {
public:
    TsaiWuFailure(const FailureModelParameters& params)
        : FailureModel(FailureModelType::TsaiWu, params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        const Real s1 = mstate.stress[0];   // Fiber direction
        const Real s2 = mstate.stress[1];   // Transverse
        const Real t12 = mstate.stress[3];  // In-plane shear

        const Real Xt = params_.Xt;
        const Real Xc = params_.Xc;
        const Real Yt = params_.Yt;
        const Real Yc = params_.Yc;
        const Real S12 = params_.S12;

        // Strength parameters
        Real F1  = 1.0/Xt - 1.0/Xc;
        Real F2  = 1.0/Yt - 1.0/Yc;
        Real F11 = 1.0/(Xt*Xc);
        Real F22 = 1.0/(Yt*Yc);
        Real F66 = 1.0/(S12*S12);
        Real F12 = params_.F12_star * Kokkos::sqrt(F11*F22);

        // Tsai-Wu failure index
        Real FI = F1*s1 + F2*s2
                + F11*s1*s1 + F22*s2*s2
                + F66*t12*t12
                + 2.0*F12*s1*s2;

        fstate.history[0] = FI;

        // Track maximum
        if (FI > fstate.history[1]) {
            fstate.history[1] = FI;
        }

        // Damage is the failure index ratio (capped at 1)
        fstate.damage = (FI > 0.0) ? Kokkos::fmin(FI, 1.0) : 0.0;

        if (FI >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = 1; // Single mode (polynomial)
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& params) {
        return std::make_unique<TsaiWuFailure>(params);
    }
};

} // namespace failure
} // namespace physics
} // namespace nxs
