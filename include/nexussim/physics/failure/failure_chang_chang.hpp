#pragma once

#include <nexussim/physics/failure/failure_model.hpp>

namespace nxs {
namespace physics {
namespace failure {

/**
 * @brief Chang-Chang failure criterion for woven/UD laminates
 *
 * Four failure modes similar to Hashin but with modified matrix criteria:
 *   1. Fiber tension:    (σ11/Xt)² + β*(τ12/S12)² ≥ 1
 *   2. Fiber compression: |σ11| ≥ Xc
 *   3. Matrix tension:   (σ22/Yt)² + (τ12/S12)² ≥ 1
 *   4. Matrix compression: (σ22/(2*S12))² + [(Yc/(2*S12))²-1]*(σ22/Yc) + (τ12/S12)² ≥ 1
 *
 * Where β is the shear contribution factor (typically 1.0).
 *
 * Failure state history:
 *   history[0..3]: failure indices for modes 1-4
 *   history[4..7]: damage flags for modes 1-4 (0 or 1)
 *
 * Reference: Chang & Chang (1987), J. Composite Materials
 */
class ChangChangFailure : public FailureModel {
public:
    ChangChangFailure(const FailureModelParameters& params)
        : FailureModel(FailureModelType::ChangChang, params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        const Real s11 = mstate.stress[0];
        const Real s22 = mstate.stress[1];
        const Real t12 = mstate.stress[3];

        const Real Xt = params_.Xt;
        const Real Xc = params_.Xc;
        const Real Yt = params_.Yt;
        const Real Yc = params_.Yc;
        const Real S12 = params_.S12;
        const Real beta = 1.0;  // Shear contribution factor

        // Mode 1: Fiber tension (σ11 > 0)
        Real f1 = 0.0;
        if (s11 > 0.0) {
            f1 = (s11/Xt)*(s11/Xt) + beta*(t12/S12)*(t12/S12);
        }
        fstate.history[0] = f1;

        // Mode 2: Fiber compression (σ11 < 0)
        Real f2 = 0.0;
        if (s11 < 0.0) {
            f2 = (-s11/Xc);  // Simple max stress for compression
        }
        fstate.history[1] = f2;

        // Mode 3: Matrix tension (σ22 > 0)
        Real f3 = 0.0;
        if (s22 > 0.0) {
            f3 = (s22/Yt)*(s22/Yt) + (t12/S12)*(t12/S12);
        }
        fstate.history[2] = f3;

        // Mode 4: Matrix compression (σ22 < 0)
        Real f4 = 0.0;
        if (s22 < 0.0) {
            Real term1 = (s22/(2.0*S12)) * (s22/(2.0*S12));
            Real term2 = ((Yc/(2.0*S12))*(Yc/(2.0*S12)) - 1.0) * (s22/Yc);
            Real term3 = (t12/S12)*(t12/S12);
            f4 = term1 + term2 + term3;
        }
        fstate.history[3] = f4;

        // Update damage flags (irreversible)
        for (int m = 0; m < 4; ++m) {
            if (fstate.history[m] >= 1.0) {
                fstate.history[4 + m] = 1.0;
            }
        }

        // Overall damage
        Real max_damage = 0.0;
        int fail_mode = 0;
        for (int m = 0; m < 4; ++m) {
            if (fstate.history[4 + m] > max_damage) {
                max_damage = fstate.history[4 + m];
                fail_mode = m + 1;
            }
        }

        fstate.damage = max_damage;
        if (max_damage >= 1.0) {
            fstate.failed = true;
            fstate.failure_mode = fail_mode;
        }
    }

    void degrade_stress(Real* sigma, const FailureState& fstate) const override {
        // Similar mode-specific degradation as Hashin
        if (fstate.history[4] >= 1.0 || fstate.history[5] >= 1.0) {
            sigma[0] = 0.0;
            sigma[3] = 0.0;
            sigma[5] = 0.0;
        }
        if (fstate.history[6] >= 1.0 || fstate.history[7] >= 1.0) {
            sigma[1] = 0.0;
            sigma[3] = 0.0;
            sigma[4] = 0.0;
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& params) {
        return std::make_unique<ChangChangFailure>(params);
    }
};

} // namespace failure
} // namespace physics
} // namespace nxs
