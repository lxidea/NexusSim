#pragma once

/**
 * @file friction_model.hpp
 * @brief Enhanced friction models for contact algorithms
 *
 * Features:
 * - Part-based friction coefficients (different μ per part pair)
 * - Orthotropic friction (direction-dependent)
 * - Viscous friction damping
 * - Friction state persistence across time steps
 * - Regularized Coulomb friction
 *
 * Friction models:
 * - Coulomb: F_t = μ * F_n
 * - Orthotropic: F_t = μ_x * F_n_x + μ_y * F_n_y
 * - Viscous: F_t = μ * F_n + η * v_t
 * - Rate-dependent: μ(v) = μ_dynamic + (μ_static - μ_dynamic) * exp(-α * |v|)
 */

#include <nexussim/core/types.hpp>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <array>

namespace nxs {
namespace fem {

// ============================================================================
// Friction State
// ============================================================================

/**
 * @brief Persistent friction state for a contact point
 */
struct FrictionState {
    Vec3r tangent_slip;          ///< Accumulated tangential slip
    Vec3r prev_velocity;         ///< Previous relative velocity
    bool sticking;               ///< Currently in stick regime
    Real slip_distance;          ///< Total slip distance traveled
    int stick_cycles;            ///< Number of consecutive stick cycles

    FrictionState()
        : tangent_slip{0, 0, 0}
        , prev_velocity{0, 0, 0}
        , sticking(true)
        , slip_distance(0.0)
        , stick_cycles(0)
    {}

    void reset() {
        tangent_slip = {0, 0, 0};
        prev_velocity = {0, 0, 0};
        sticking = true;
        slip_distance = 0.0;
        stick_cycles = 0;
    }
};

// ============================================================================
// Part-Based Friction Table
// ============================================================================

/**
 * @brief Friction coefficients for a part pair
 */
struct PartPairFriction {
    Index part1;                 ///< First part ID
    Index part2;                 ///< Second part ID
    Real static_friction;        ///< Static friction coefficient
    Real dynamic_friction;       ///< Dynamic friction coefficient
    Real viscous_damping;        ///< Viscous damping coefficient

    // Orthotropic friction (optional)
    bool is_orthotropic;
    Real mu_x;                   ///< Friction coefficient in x direction
    Real mu_y;                   ///< Friction coefficient in y direction
    Vec3r fiber_direction;       ///< Local fiber direction for orthotropic

    PartPairFriction()
        : part1(0), part2(0)
        , static_friction(0.3)
        , dynamic_friction(0.2)
        , viscous_damping(0.0)
        , is_orthotropic(false)
        , mu_x(0.3), mu_y(0.3)
        , fiber_direction{1, 0, 0}
    {}

    PartPairFriction(Index p1, Index p2, Real mu_s, Real mu_d = -1.0)
        : part1(p1), part2(p2)
        , static_friction(mu_s)
        , dynamic_friction(mu_d < 0 ? mu_s * 0.8 : mu_d)
        , viscous_damping(0.0)
        , is_orthotropic(false)
        , mu_x(mu_s), mu_y(mu_s)
        , fiber_direction{1, 0, 0}
    {}
};

/**
 * @brief Table of friction coefficients by part pair
 */
class FrictionTable {
public:
    FrictionTable()
        : default_static_(0.3)
        , default_dynamic_(0.2)
    {}

    /**
     * @brief Set default friction coefficients
     */
    void set_default(Real mu_static, Real mu_dynamic = -1.0) {
        default_static_ = mu_static;
        default_dynamic_ = (mu_dynamic < 0) ? mu_static * 0.8 : mu_dynamic;
    }

    /**
     * @brief Add friction coefficients for a part pair
     */
    void add_pair(Index part1, Index part2, Real mu_static, Real mu_dynamic = -1.0) {
        PartPairFriction pair(part1, part2, mu_static, mu_dynamic);
        uint64_t key = make_key(part1, part2);
        friction_pairs_[key] = pair;
    }

    /**
     * @brief Add orthotropic friction for a part pair
     */
    void add_orthotropic_pair(Index part1, Index part2,
                               Real mu_x, Real mu_y,
                               const Vec3r& fiber_dir) {
        PartPairFriction pair;
        pair.part1 = part1;
        pair.part2 = part2;
        pair.is_orthotropic = true;
        pair.mu_x = mu_x;
        pair.mu_y = mu_y;
        pair.fiber_direction = fiber_dir;
        pair.static_friction = (mu_x + mu_y) * 0.5;
        pair.dynamic_friction = pair.static_friction * 0.8;

        uint64_t key = make_key(part1, part2);
        friction_pairs_[key] = pair;
    }

    /**
     * @brief Get friction coefficients for a part pair
     */
    const PartPairFriction& get_friction(Index part1, Index part2) const {
        uint64_t key = make_key(part1, part2);
        auto it = friction_pairs_.find(key);
        if (it != friction_pairs_.end()) {
            return it->second;
        }
        return default_pair_;
    }

    /**
     * @brief Get static friction coefficient for part pair
     */
    Real get_static_friction(Index part1, Index part2) const {
        return get_friction(part1, part2).static_friction;
    }

    /**
     * @brief Get dynamic friction coefficient for part pair
     */
    Real get_dynamic_friction(Index part1, Index part2) const {
        return get_friction(part1, part2).dynamic_friction;
    }

private:
    uint64_t make_key(Index p1, Index p2) const {
        Index a = std::min(p1, p2);
        Index b = std::max(p1, p2);
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    }

    std::unordered_map<uint64_t, PartPairFriction> friction_pairs_;
    Real default_static_;
    Real default_dynamic_;
    PartPairFriction default_pair_;
};

// ============================================================================
// Friction Model Interface
// ============================================================================

/**
 * @brief Base class for friction models
 */
class FrictionModel {
public:
    virtual ~FrictionModel() = default;

    /**
     * @brief Compute friction force
     * @param normal_force Normal force magnitude
     * @param rel_velocity Relative tangential velocity
     * @param normal Contact normal direction
     * @param dt Time step
     * @param state Friction state (input/output)
     * @return Friction force vector
     */
    virtual Vec3r compute_friction(Real normal_force,
                                    const Vec3r& rel_velocity,
                                    const Vec3r& normal,
                                    Real dt,
                                    FrictionState& state) const = 0;

    /**
     * @brief Get tangential component of velocity
     */
    static Vec3r tangential_velocity(const Vec3r& velocity, const Vec3r& normal) {
        Real vn = velocity[0] * normal[0] + velocity[1] * normal[1] + velocity[2] * normal[2];
        return {
            velocity[0] - vn * normal[0],
            velocity[1] - vn * normal[1],
            velocity[2] - vn * normal[2]
        };
    }
};

// ============================================================================
// Coulomb Friction Model
// ============================================================================

/**
 * @brief Standard Coulomb friction with regularization
 *
 * F_t = μ * F_n (opposing relative motion)
 * Uses penalty regularization for stick-slip transition.
 */
class CoulombFriction : public FrictionModel {
public:
    CoulombFriction(Real mu_static = 0.3, Real mu_dynamic = 0.2, Real stiffness = 1.0e6)
        : mu_static_(mu_static)
        , mu_dynamic_(mu_dynamic)
        , stick_stiffness_(stiffness)
    {}

    Vec3r compute_friction(Real normal_force,
                           const Vec3r& rel_velocity,
                           const Vec3r& normal,
                           Real dt,
                           FrictionState& state) const override {
        if (normal_force <= 0.0) {
            state.reset();
            return {0, 0, 0};
        }

        // Tangential velocity
        Vec3r v_tang = tangential_velocity(rel_velocity, normal);

        // Update slip
        for (int d = 0; d < 3; ++d) {
            state.tangent_slip[d] += v_tang[d] * dt;
        }

        Real slip_mag = std::sqrt(
            state.tangent_slip[0] * state.tangent_slip[0] +
            state.tangent_slip[1] * state.tangent_slip[1] +
            state.tangent_slip[2] * state.tangent_slip[2]
        );

        if (slip_mag < 1.0e-20) {
            return {0, 0, 0};
        }

        Real mu = state.sticking ? mu_static_ : mu_dynamic_;
        Real max_friction = mu * normal_force;

        // Trial friction (stick)
        Real trial_friction = stick_stiffness_ * slip_mag;

        Vec3r friction_force;
        if (trial_friction <= max_friction) {
            // Sticking
            state.sticking = true;
            state.stick_cycles++;
            Real scale = trial_friction / slip_mag;
            friction_force = {
                -scale * state.tangent_slip[0],
                -scale * state.tangent_slip[1],
                -scale * state.tangent_slip[2]
            };
        } else {
            // Sliding
            state.sticking = false;
            state.stick_cycles = 0;
            state.slip_distance += slip_mag;

            Real scale = max_friction / slip_mag;
            friction_force = {
                -scale * state.tangent_slip[0],
                -scale * state.tangent_slip[1],
                -scale * state.tangent_slip[2]
            };

            // Reset slip state
            for (int d = 0; d < 3; ++d) {
                state.tangent_slip[d] = friction_force[d] / stick_stiffness_;
            }
        }

        return friction_force;
    }

    void set_coefficients(Real mu_static, Real mu_dynamic) {
        mu_static_ = mu_static;
        mu_dynamic_ = mu_dynamic;
    }

private:
    Real mu_static_;
    Real mu_dynamic_;
    Real stick_stiffness_;
};

// ============================================================================
// Rate-Dependent Friction Model
// ============================================================================

/**
 * @brief Velocity-dependent friction coefficient
 *
 * μ(v) = μ_dynamic + (μ_static - μ_dynamic) * exp(-α * |v|)
 */
class RateDependentFriction : public FrictionModel {
public:
    RateDependentFriction(Real mu_static = 0.3, Real mu_dynamic = 0.2,
                          Real decay_rate = 10.0, Real stiffness = 1.0e6)
        : mu_static_(mu_static)
        , mu_dynamic_(mu_dynamic)
        , decay_rate_(decay_rate)
        , stick_stiffness_(stiffness)
    {}

    Vec3r compute_friction(Real normal_force,
                           const Vec3r& rel_velocity,
                           const Vec3r& normal,
                           Real dt,
                           FrictionState& state) const override {
        if (normal_force <= 0.0) {
            state.reset();
            return {0, 0, 0};
        }

        Vec3r v_tang = tangential_velocity(rel_velocity, normal);
        Real v_mag = std::sqrt(v_tang[0] * v_tang[0] + v_tang[1] * v_tang[1] + v_tang[2] * v_tang[2]);

        // Rate-dependent friction coefficient
        Real mu = mu_dynamic_ + (mu_static_ - mu_dynamic_) * std::exp(-decay_rate_ * v_mag);
        Real max_friction = mu * normal_force;

        // Update slip
        for (int d = 0; d < 3; ++d) {
            state.tangent_slip[d] += v_tang[d] * dt;
        }

        Real slip_mag = std::sqrt(
            state.tangent_slip[0] * state.tangent_slip[0] +
            state.tangent_slip[1] * state.tangent_slip[1] +
            state.tangent_slip[2] * state.tangent_slip[2]
        );

        if (slip_mag < 1.0e-20) {
            return {0, 0, 0};
        }

        Real trial_friction = stick_stiffness_ * slip_mag;

        Vec3r friction_force;
        if (trial_friction <= max_friction) {
            state.sticking = true;
            Real scale = trial_friction / slip_mag;
            friction_force = {-scale * state.tangent_slip[0],
                              -scale * state.tangent_slip[1],
                              -scale * state.tangent_slip[2]};
        } else {
            state.sticking = false;
            Real scale = max_friction / slip_mag;
            friction_force = {-scale * state.tangent_slip[0],
                              -scale * state.tangent_slip[1],
                              -scale * state.tangent_slip[2]};
            for (int d = 0; d < 3; ++d) {
                state.tangent_slip[d] = friction_force[d] / stick_stiffness_;
            }
        }

        return friction_force;
    }

private:
    Real mu_static_;
    Real mu_dynamic_;
    Real decay_rate_;
    Real stick_stiffness_;
};

// ============================================================================
// Orthotropic Friction Model
// ============================================================================

/**
 * @brief Direction-dependent friction (for fiber composites)
 *
 * Different friction coefficients in fiber and transverse directions.
 */
class OrthotropicFriction : public FrictionModel {
public:
    OrthotropicFriction(Real mu_fiber = 0.2, Real mu_transverse = 0.4,
                         const Vec3r& fiber_dir = {1, 0, 0},
                         Real stiffness = 1.0e6)
        : mu_fiber_(mu_fiber)
        , mu_transverse_(mu_transverse)
        , fiber_direction_(fiber_dir)
        , stick_stiffness_(stiffness)
    {
        // Normalize fiber direction
        Real len = std::sqrt(fiber_dir[0] * fiber_dir[0] +
                             fiber_dir[1] * fiber_dir[1] +
                             fiber_dir[2] * fiber_dir[2]);
        if (len > 1.0e-20) {
            fiber_direction_ = {fiber_dir[0] / len, fiber_dir[1] / len, fiber_dir[2] / len};
        }
    }

    Vec3r compute_friction(Real normal_force,
                           const Vec3r& rel_velocity,
                           const Vec3r& normal,
                           Real dt,
                           FrictionState& state) const override {
        if (normal_force <= 0.0) {
            state.reset();
            return {0, 0, 0};
        }

        Vec3r v_tang = tangential_velocity(rel_velocity, normal);

        // Project fiber direction onto contact plane
        Vec3r fiber_proj = tangential_velocity(fiber_direction_, normal);
        Real fiber_len = std::sqrt(fiber_proj[0] * fiber_proj[0] +
                                   fiber_proj[1] * fiber_proj[1] +
                                   fiber_proj[2] * fiber_proj[2]);
        if (fiber_len > 1.0e-10) {
            for (int d = 0; d < 3; ++d) fiber_proj[d] /= fiber_len;
        }

        // Transverse direction (perpendicular to fiber in contact plane)
        Vec3r trans_dir = {
            normal[1] * fiber_proj[2] - normal[2] * fiber_proj[1],
            normal[2] * fiber_proj[0] - normal[0] * fiber_proj[2],
            normal[0] * fiber_proj[1] - normal[1] * fiber_proj[0]
        };
        Real trans_len = std::sqrt(trans_dir[0] * trans_dir[0] +
                                   trans_dir[1] * trans_dir[1] +
                                   trans_dir[2] * trans_dir[2]);
        if (trans_len > 1.0e-10) {
            for (int d = 0; d < 3; ++d) trans_dir[d] /= trans_len;
        }

        // Decompose velocity into fiber and transverse components
        Real v_fiber = v_tang[0] * fiber_proj[0] + v_tang[1] * fiber_proj[1] + v_tang[2] * fiber_proj[2];
        Real v_trans = v_tang[0] * trans_dir[0] + v_tang[1] * trans_dir[1] + v_tang[2] * trans_dir[2];

        // Update slip components
        state.tangent_slip[0] += v_fiber * dt;  // Use [0] for fiber slip
        state.tangent_slip[1] += v_trans * dt;  // Use [1] for transverse slip

        // Apply different friction in each direction
        Real slip_fiber = std::abs(state.tangent_slip[0]);
        Real slip_trans = std::abs(state.tangent_slip[1]);

        Real max_f_fiber = mu_fiber_ * normal_force;
        Real max_f_trans = mu_transverse_ * normal_force;

        Real trial_fiber = stick_stiffness_ * slip_fiber;
        Real trial_trans = stick_stiffness_ * slip_trans;

        Real f_fiber, f_trans;

        // Fiber direction friction
        if (trial_fiber <= max_f_fiber) {
            f_fiber = -std::copysign(trial_fiber, state.tangent_slip[0]);
        } else {
            f_fiber = -std::copysign(max_f_fiber, state.tangent_slip[0]);
            state.tangent_slip[0] = f_fiber / stick_stiffness_;
        }

        // Transverse direction friction
        if (trial_trans <= max_f_trans) {
            f_trans = -std::copysign(trial_trans, state.tangent_slip[1]);
        } else {
            f_trans = -std::copysign(max_f_trans, state.tangent_slip[1]);
            state.tangent_slip[1] = f_trans / stick_stiffness_;
        }

        state.sticking = (trial_fiber <= max_f_fiber) && (trial_trans <= max_f_trans);

        // Convert back to 3D force
        Vec3r friction_force = {
            f_fiber * fiber_proj[0] + f_trans * trans_dir[0],
            f_fiber * fiber_proj[1] + f_trans * trans_dir[1],
            f_fiber * fiber_proj[2] + f_trans * trans_dir[2]
        };

        return friction_force;
    }

    void set_fiber_direction(const Vec3r& dir) {
        Real len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
        if (len > 1.0e-20) {
            fiber_direction_ = {dir[0] / len, dir[1] / len, dir[2] / len};
        }
    }

private:
    Real mu_fiber_;
    Real mu_transverse_;
    Vec3r fiber_direction_;
    Real stick_stiffness_;
};

// ============================================================================
// Viscous Friction Model
// ============================================================================

/**
 * @brief Coulomb + viscous damping friction
 *
 * F_t = μ * F_n + η * v_t
 */
class ViscousFriction : public FrictionModel {
public:
    ViscousFriction(Real mu_static = 0.3, Real mu_dynamic = 0.2,
                    Real viscosity = 0.01, Real stiffness = 1.0e6)
        : mu_static_(mu_static)
        , mu_dynamic_(mu_dynamic)
        , viscosity_(viscosity)
        , stick_stiffness_(stiffness)
    {}

    Vec3r compute_friction(Real normal_force,
                           const Vec3r& rel_velocity,
                           const Vec3r& normal,
                           Real dt,
                           FrictionState& state) const override {
        if (normal_force <= 0.0) {
            state.reset();
            return {0, 0, 0};
        }

        Vec3r v_tang = tangential_velocity(rel_velocity, normal);
        Real v_mag = std::sqrt(v_tang[0] * v_tang[0] + v_tang[1] * v_tang[1] + v_tang[2] * v_tang[2]);

        // Update slip
        for (int d = 0; d < 3; ++d) {
            state.tangent_slip[d] += v_tang[d] * dt;
        }

        Real slip_mag = std::sqrt(
            state.tangent_slip[0] * state.tangent_slip[0] +
            state.tangent_slip[1] * state.tangent_slip[1] +
            state.tangent_slip[2] * state.tangent_slip[2]
        );

        // Coulomb component
        Real mu = state.sticking ? mu_static_ : mu_dynamic_;
        Real max_coulomb = mu * normal_force;

        Real trial_friction = stick_stiffness_ * slip_mag;

        Vec3r coulomb_force = {0, 0, 0};
        if (slip_mag > 1.0e-20) {
            if (trial_friction <= max_coulomb) {
                state.sticking = true;
                Real scale = trial_friction / slip_mag;
                coulomb_force = {-scale * state.tangent_slip[0],
                                 -scale * state.tangent_slip[1],
                                 -scale * state.tangent_slip[2]};
            } else {
                state.sticking = false;
                Real scale = max_coulomb / slip_mag;
                coulomb_force = {-scale * state.tangent_slip[0],
                                 -scale * state.tangent_slip[1],
                                 -scale * state.tangent_slip[2]};
                for (int d = 0; d < 3; ++d) {
                    state.tangent_slip[d] = coulomb_force[d] / stick_stiffness_;
                }
            }
        }

        // Viscous component
        Vec3r viscous_force = {
            -viscosity_ * v_tang[0],
            -viscosity_ * v_tang[1],
            -viscosity_ * v_tang[2]
        };

        // Total friction
        return {
            coulomb_force[0] + viscous_force[0],
            coulomb_force[1] + viscous_force[1],
            coulomb_force[2] + viscous_force[2]
        };
    }

    void set_viscosity(Real eta) { viscosity_ = eta; }

private:
    Real mu_static_;
    Real mu_dynamic_;
    Real viscosity_;
    Real stick_stiffness_;
};

// ============================================================================
// Friction Model Factory
// ============================================================================

enum class FrictionType {
    Coulomb,
    RateDependent,
    Orthotropic,
    Viscous
};

/**
 * @brief Create friction model by type
 */
inline std::unique_ptr<FrictionModel> create_friction_model(
    FrictionType type,
    Real mu_static = 0.3,
    Real mu_dynamic = 0.2,
    Real stiffness = 1.0e6,
    Real extra_param = 0.0)
{
    switch (type) {
        case FrictionType::Coulomb:
            return std::make_unique<CoulombFriction>(mu_static, mu_dynamic, stiffness);

        case FrictionType::RateDependent:
            return std::make_unique<RateDependentFriction>(
                mu_static, mu_dynamic, extra_param > 0 ? extra_param : 10.0, stiffness);

        case FrictionType::Orthotropic:
            return std::make_unique<OrthotropicFriction>(
                mu_static, mu_dynamic, Vec3r{1, 0, 0}, stiffness);

        case FrictionType::Viscous:
            return std::make_unique<ViscousFriction>(
                mu_static, mu_dynamic, extra_param > 0 ? extra_param : 0.01, stiffness);

        default:
            return std::make_unique<CoulombFriction>(mu_static, mu_dynamic, stiffness);
    }
}

} // namespace fem
} // namespace nxs
