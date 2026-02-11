#pragma once

#include <nexussim/physics/failure/failure_model.hpp>

namespace nxs {
namespace physics {
namespace failure {

/**
 * @brief Hashin composite failure criterion (4 independent modes)
 *
 * Evaluates four failure indices for unidirectional composites:
 *   1. Fiber tension:    (σ11/Xt)² + (τ12/S12)² ≥ 1
 *   2. Fiber compression: |σ11/Xc| ≥ 1
 *   3. Matrix tension:   (σ22/Yt)² + (τ12/S12)² ≥ 1
 *   4. Matrix compression: (σ22/(2*S23))² + [(Yc/(2*S23))²-1]*(σ22/Yc) + (τ12/S12)² ≥ 1
 *
 * Failure state history:
 *   history[0..3]: failure indices for modes 1-4
 *   history[4..7]: damage variables for modes 1-4
 *
 * Reference: Hashin (1980), "Failure criteria for unidirectional fiber composites"
 */
class HashinFailure : public FailureModel {
public:
    HashinFailure(const FailureModelParameters& params)
        : FailureModel(FailureModelType::Hashin, params) {}

    void compute_damage(const MaterialState& mstate,
                        FailureState& fstate,
                        Real /*dt*/,
                        Real /*element_size*/) const override {
        // Stress components (lamina coordinate system)
        const Real s11 = mstate.stress[0];  // Fiber direction
        const Real s22 = mstate.stress[1];  // Transverse
        const Real s33 = mstate.stress[2];  // Through-thickness
        const Real t12 = mstate.stress[3];  // In-plane shear
        const Real t23 = mstate.stress[4];  // Transverse shear
        (void)s33; (void)t23; // Used in 3D extension

        const Real Xt = params_.Xt;
        const Real Xc = params_.Xc;
        const Real Yt = params_.Yt;
        const Real Yc = params_.Yc;
        const Real S12 = params_.S12;
        const Real S23 = params_.S23;

        // Mode 1: Fiber tension (σ11 > 0)
        Real f1 = 0.0;
        if (s11 > 0.0) {
            f1 = (s11/Xt)*(s11/Xt) + (t12/S12)*(t12/S12);
        }
        fstate.history[0] = f1;

        // Mode 2: Fiber compression (σ11 < 0)
        Real f2 = 0.0;
        if (s11 < 0.0) {
            f2 = (-s11/Xc) * (-s11/Xc);
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
            Real term1 = (s22/(2.0*S23)) * (s22/(2.0*S23));
            Real term2 = ((Yc/(2.0*S23))*(Yc/(2.0*S23)) - 1.0) * (s22/Yc);
            Real term3 = (t12/S12) * (t12/S12);
            f4 = term1 + term2 + term3;
        }
        fstate.history[3] = f4;

        // Update damage for each mode (maximum of current and previous)
        for (int m = 0; m < 4; ++m) {
            Real fi = fstate.history[m];
            if (fi >= 1.0) {
                fstate.history[4 + m] = 1.0; // Mode failed
            }
        }

        // Overall damage = max of all mode damages
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
        // Mode-specific stress degradation
        if (fstate.history[4] >= 1.0 || fstate.history[5] >= 1.0) {
            // Fiber failure: zero all fiber-direction stresses
            sigma[0] = 0.0;  // σ11
            sigma[3] = 0.0;  // τ12
            sigma[5] = 0.0;  // τ13
        }
        if (fstate.history[6] >= 1.0 || fstate.history[7] >= 1.0) {
            // Matrix failure: zero transverse and shear stresses
            sigma[1] = 0.0;  // σ22
            sigma[3] = 0.0;  // τ12
            sigma[4] = 0.0;  // τ23
        }
    }

    static std::unique_ptr<FailureModel> create(const FailureModelParameters& params) {
        return std::make_unique<HashinFailure>(params);
    }
};

} // namespace failure
} // namespace physics
} // namespace nxs
