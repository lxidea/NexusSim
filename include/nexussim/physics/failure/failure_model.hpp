#pragma once

/**
 * @file failure_model.hpp
 * @brief Base interface for advanced failure/damage models
 *
 * Provides a pluggable interface for composite, ductile, and
 * mesh-regularized failure criteria used in crash simulation.
 *
 * Each model implements:
 *   - compute_damage(): Compute damage increment from current state
 *   - check_failure(): Determine if element has failed
 *   - damage_variable(): Return current damage scalar (0 = intact, 1 = failed)
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <nexussim/physics/material.hpp>
#include <cmath>

namespace nxs {
namespace physics {
namespace failure {

// ============================================================================
// Failure Model Types
// ============================================================================

enum class FailureModelType {
    Hashin,           ///< Hashin composite failure (4 modes)
    TsaiWu,          ///< Tsai-Wu polynomial composite failure
    ChangChang,       ///< Chang-Chang woven/UD laminate failure
    GTN,              ///< Gurson-Tvergaard-Needleman ductile damage
    GISSMO,           ///< Generalized Incremental Stress-State dependent damage
    TabulatedEnvelope ///< User-defined triaxiality-dependent failure strain
};

// ============================================================================
// Failure State (per element/integration point)
// ============================================================================

struct FailureState {
    Real damage;              ///< Scalar damage variable (0 = intact, 1 = failed)
    bool failed;              ///< Element has failed
    int failure_mode;         ///< Which mode triggered failure (model-specific)
    Real history[16];         ///< Model-specific internal variables

    FailureState() : damage(0.0), failed(false), failure_mode(0) {
        for (int i = 0; i < 16; ++i) history[i] = 0.0;
    }
};

// ============================================================================
// Failure Model Parameters
// ============================================================================

struct FailureModelParameters {
    FailureModelType type;

    // --- Hashin / Chang-Chang / Tsai-Wu (composite) parameters ---
    Real Xt;          ///< Tensile strength, fiber direction (1-dir)
    Real Xc;          ///< Compressive strength, fiber direction
    Real Yt;          ///< Tensile strength, transverse direction (2-dir)
    Real Yc;          ///< Compressive strength, transverse direction
    Real Zt;          ///< Tensile strength, through-thickness (3-dir)
    Real Zc;          ///< Compressive strength, through-thickness
    Real S12;         ///< In-plane shear strength
    Real S23;         ///< Transverse shear strength
    Real S13;         ///< Through-thickness shear strength

    // Tsai-Wu interaction term
    Real F12_star;    ///< Normalized interaction term (typically -0.5 to 0)

    // --- GTN (ductile damage) parameters ---
    Real f0;          ///< Initial void volume fraction
    Real fN;          ///< Volume fraction of void-nucleating particles
    Real sN;          ///< Standard deviation of nucleation strain
    Real epsN;        ///< Mean nucleation strain
    Real fc;          ///< Critical void fraction (onset of coalescence)
    Real fF;          ///< Failure void fraction
    Real q1;          ///< Tvergaard parameter q1 (typically 1.5)
    Real q2;          ///< Tvergaard parameter q2 (typically 1.0)
    Real q3;          ///< Tvergaard parameter q3 (typically q1^2)

    // --- GISSMO parameters ---
    Real dcrit;       ///< Critical damage for element erosion
    Real n_exp;       ///< Damage exponent
    Real fadexp;      ///< Fading exponent (stress reduction after instability)
    Real lc_ref;      ///< Reference element size for regularization
    // GISSMO uses tabulated failure strain vs triaxiality (stored externally)

    // --- Tabulated envelope ---
    // Failure strain as function of triaxiality
    TabulatedCurve failure_envelope;  ///< eps_f(eta)

    FailureModelParameters()
        : type(FailureModelType::Hashin)
        , Xt(1.0e9), Xc(0.6e9), Yt(40.0e6), Yc(120.0e6)
        , Zt(40.0e6), Zc(120.0e6)
        , S12(60.0e6), S23(40.0e6), S13(60.0e6)
        , F12_star(-0.5)
        , f0(0.001), fN(0.04), sN(0.1), epsN(0.3)
        , fc(0.15), fF(0.25)
        , q1(1.5), q2(1.0), q3(2.25)
        , dcrit(1.0), n_exp(2.0), fadexp(1.0), lc_ref(1.0e-3)
    {}
};

// ============================================================================
// Base Failure Model Interface
// ============================================================================

class FailureModel {
public:
    FailureModel(FailureModelType type, const FailureModelParameters& params)
        : type_(type), params_(params) {}

    virtual ~FailureModel() = default;

    FailureModelType type() const { return type_; }

    /**
     * @brief Update damage state from current stress/strain
     * @param mstate Material state (stress, strain, plastic_strain, etc.)
     * @param fstate Failure state to update (damage, history, failed)
     * @param dt Time step
     * @param element_size Characteristic element length (for regularization)
     */
    virtual void compute_damage(const MaterialState& mstate,
                                FailureState& fstate,
                                Real dt,
                                Real element_size) const = 0;

    /**
     * @brief Check if failure has occurred
     * @return true if element should be eroded
     */
    virtual bool check_failure(const FailureState& fstate) const {
        return fstate.failed;
    }

    /**
     * @brief Apply damage to stress tensor (stress degradation)
     * @param sigma Stress tensor [6], modified in place
     * @param fstate Current failure state
     */
    virtual void degrade_stress(Real* sigma, const FailureState& fstate) const {
        if (fstate.damage > 0.0 && fstate.damage <= 1.0) {
            Real factor = 1.0 - fstate.damage;
            for (int i = 0; i < 6; ++i) sigma[i] *= factor;
        }
    }

    static const char* to_string(FailureModelType type) {
        switch (type) {
            case FailureModelType::Hashin: return "Hashin";
            case FailureModelType::TsaiWu: return "Tsai-Wu";
            case FailureModelType::ChangChang: return "Chang-Chang";
            case FailureModelType::GTN: return "GTN";
            case FailureModelType::GISSMO: return "GISSMO";
            case FailureModelType::TabulatedEnvelope: return "TabulatedEnvelope";
            default: return "Unknown";
        }
    }

protected:
    FailureModelType type_;
    FailureModelParameters params_;
};

} // namespace failure
} // namespace physics
} // namespace nxs
