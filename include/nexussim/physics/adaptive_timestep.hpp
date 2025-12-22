#pragma once

/**
 * @file adaptive_timestep.hpp
 * @brief Adaptive time stepping controller for explicit dynamics
 *
 * Implements automatic timestep control based on:
 * - CFL condition (element size / wave speed)
 * - Energy-based error estimation
 * - Velocity/acceleration gradient monitoring
 * - Contact event detection
 *
 * Features:
 * - Automatic dt increase when stable (faster simulations)
 * - Automatic dt decrease on instability detection
 * - Smooth ramping to avoid sudden changes
 * - Energy conservation monitoring
 */

#include <nexussim/core/core.hpp>
#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>

namespace nxs {
namespace physics {

/**
 * @brief Timestep control strategy
 */
enum class TimestepStrategy {
    Fixed,           ///< Use fixed timestep (no adaptation)
    CFL,             ///< CFL-based only (recompute each step)
    EnergyBased,     ///< Energy error estimation
    Combined         ///< CFL + energy monitoring
};

/**
 * @brief Energy tracking data for stability monitoring
 */
struct EnergyState {
    Real kinetic;      ///< Total kinetic energy
    Real internal;     ///< Total internal (strain) energy
    Real external;     ///< Work done by external forces
    Real total;        ///< Total mechanical energy
    Real time;         ///< Simulation time

    EnergyState() : kinetic(0), internal(0), external(0), total(0), time(0) {}
};

/**
 * @brief Adaptive timestep controller
 *
 * Usage:
 * ```cpp
 * AdaptiveTimestep controller;
 * controller.set_strategy(TimestepStrategy::Combined);
 * controller.set_target_cfl(0.9);
 * controller.set_min_dt(1e-10);
 * controller.set_max_dt(1e-5);
 *
 * while (time < end_time) {
 *     Real dt = controller.compute_timestep(solver.compute_stable_dt());
 *     solver.step(dt);
 *     controller.update_energy(kinetic, internal, external, time);
 *     controller.check_stability();
 * }
 * ```
 */
class AdaptiveTimestep {
public:
    AdaptiveTimestep()
        : strategy_(TimestepStrategy::CFL)
        , target_cfl_(0.9)
        , min_dt_(1e-15)
        , max_dt_(1.0)
        , current_dt_(0)
        , previous_dt_(0)
        , dt_growth_factor_(1.1)
        , dt_shrink_factor_(0.5)
        , energy_tolerance_(0.01)     // 1% energy error tolerance
        , velocity_tolerance_(1e10)   // Max velocity (m/s)
        , step_count_(0)
        , stable_steps_(0)
        , unstable_steps_(0)
        , min_stable_steps_(10)       // Steps before allowing dt increase
        , energy_history_size_(50)
        , initial_energy_(0)
        , initial_energy_set_(false)
        , verbose_(false)
    {}

    // ========================================================================
    // Configuration
    // ========================================================================

    /**
     * @brief Set timestep control strategy
     */
    void set_strategy(TimestepStrategy strategy) { strategy_ = strategy; }
    TimestepStrategy strategy() const { return strategy_; }

    /**
     * @brief Set target CFL number (typically 0.5-0.9)
     */
    void set_target_cfl(Real cfl) { target_cfl_ = std::clamp(cfl, 0.1, 1.0); }
    Real target_cfl() const { return target_cfl_; }

    /**
     * @brief Set minimum allowed timestep
     */
    void set_min_dt(Real dt) { min_dt_ = dt; }
    Real min_dt() const { return min_dt_; }

    /**
     * @brief Set maximum allowed timestep
     */
    void set_max_dt(Real dt) { max_dt_ = dt; }
    Real max_dt() const { return max_dt_; }

    /**
     * @brief Set timestep growth factor (used when stable, typically 1.05-1.2)
     */
    void set_growth_factor(Real factor) { dt_growth_factor_ = std::max(1.0, factor); }
    Real growth_factor() const { return dt_growth_factor_; }

    /**
     * @brief Set timestep shrink factor (used on instability, typically 0.3-0.7)
     */
    void set_shrink_factor(Real factor) { dt_shrink_factor_ = std::clamp(factor, 0.1, 0.9); }
    Real shrink_factor() const { return dt_shrink_factor_; }

    /**
     * @brief Set energy error tolerance (fraction of initial energy)
     */
    void set_energy_tolerance(Real tol) { energy_tolerance_ = tol; }
    Real energy_tolerance() const { return energy_tolerance_; }

    /**
     * @brief Set maximum allowed velocity
     */
    void set_velocity_tolerance(Real max_vel) { velocity_tolerance_ = max_vel; }
    Real velocity_tolerance() const { return velocity_tolerance_; }

    /**
     * @brief Set minimum stable steps before allowing dt increase
     */
    void set_min_stable_steps(int n) { min_stable_steps_ = n; }
    int min_stable_steps() const { return min_stable_steps_; }

    /**
     * @brief Enable/disable verbose output
     */
    void set_verbose(bool v) { verbose_ = v; }
    bool verbose() const { return verbose_; }

    // ========================================================================
    // Core Methods
    // ========================================================================

    /**
     * @brief Compute next timestep based on CFL limit and history
     * @param cfl_dt CFL-based stable timestep from solver
     * @return Recommended timestep
     */
    Real compute_timestep(Real cfl_dt) {
        Real dt = cfl_dt * target_cfl_;

        // Apply CFL-based limit
        if (strategy_ == TimestepStrategy::Fixed && current_dt_ > 0) {
            dt = current_dt_;
        } else {
            // For adaptive strategies, apply growth/shrink logic
            if (current_dt_ > 0 && strategy_ != TimestepStrategy::CFL) {
                if (stable_steps_ >= min_stable_steps_) {
                    // Can try to grow timestep
                    Real target_dt = current_dt_ * dt_growth_factor_;
                    dt = std::min(dt, target_dt);
                } else if (unstable_steps_ > 0) {
                    // Force shrink
                    dt = current_dt_ * dt_shrink_factor_;
                }
            }
        }

        // Apply bounds
        dt = std::clamp(dt, min_dt_, max_dt_);

        // Store for history
        previous_dt_ = current_dt_;
        current_dt_ = dt;
        step_count_++;

        if (verbose_ && step_count_ % 1000 == 0) {
            std::cout << "[AdaptiveTimestep] Step " << step_count_
                      << ": dt = " << dt
                      << ", stable_steps = " << stable_steps_
                      << ", cfl_dt = " << cfl_dt << "\n";
        }

        return dt;
    }

    /**
     * @brief Update energy state for monitoring
     * @param kinetic Total kinetic energy
     * @param internal Total internal energy
     * @param external Work done by external forces
     * @param time Current simulation time
     */
    void update_energy(Real kinetic, Real internal, Real external, Real time) {
        EnergyState state;
        state.kinetic = kinetic;
        state.internal = internal;
        state.external = external;
        state.total = kinetic + internal;  // Mechanical energy
        state.time = time;

        // Set initial energy reference
        if (!initial_energy_set_ && state.total > 0) {
            initial_energy_ = state.total;
            initial_energy_set_ = true;
        }

        // Add to history
        energy_history_.push_back(state);
        if (energy_history_.size() > energy_history_size_) {
            energy_history_.pop_front();
        }
    }

    /**
     * @brief Check stability based on energy conservation
     * @return true if stable, false if instability detected
     */
    bool check_stability() {
        if (strategy_ == TimestepStrategy::Fixed ||
            strategy_ == TimestepStrategy::CFL) {
            stable_steps_++;
            return true;
        }

        bool stable = true;

        // Energy-based stability check
        if (energy_history_.size() >= 2 && initial_energy_set_) {
            const auto& current = energy_history_.back();

            // Check energy growth (instability indicator)
            Real energy_ratio = current.total / initial_energy_;
            if (energy_ratio > (1.0 + energy_tolerance_)) {
                stable = false;
                if (verbose_) {
                    std::cout << "[AdaptiveTimestep] WARNING: Energy growth detected! "
                              << "ratio = " << energy_ratio << "\n";
                }
            }

            // Check for oscillating energy (numerical instability)
            if (energy_history_.size() >= 5) {
                Real e1 = energy_history_[energy_history_.size()-1].total;
                Real e2 = energy_history_[energy_history_.size()-2].total;
                Real e3 = energy_history_[energy_history_.size()-3].total;
                Real e4 = energy_history_[energy_history_.size()-4].total;
                Real e5 = energy_history_[energy_history_.size()-5].total;

                // Check for sign changes in energy derivative
                Real de1 = e1 - e2;
                Real de2 = e2 - e3;
                Real de3 = e3 - e4;
                Real de4 = e4 - e5;

                int sign_changes = 0;
                if (de1 * de2 < 0) sign_changes++;
                if (de2 * de3 < 0) sign_changes++;
                if (de3 * de4 < 0) sign_changes++;

                if (sign_changes >= 2) {
                    stable = false;
                    if (verbose_) {
                        std::cout << "[AdaptiveTimestep] WARNING: Energy oscillation detected!\n";
                    }
                }
            }
        }

        // Update counters
        if (stable) {
            stable_steps_++;
            unstable_steps_ = 0;
        } else {
            unstable_steps_++;
            stable_steps_ = 0;
        }

        return stable;
    }

    /**
     * @brief Check if maximum velocity is exceeded
     * @param max_velocity Maximum velocity in the system
     * @return true if within tolerance
     */
    bool check_velocity(Real max_velocity) {
        if (max_velocity > velocity_tolerance_) {
            unstable_steps_++;
            stable_steps_ = 0;
            if (verbose_) {
                std::cout << "[AdaptiveTimestep] WARNING: Max velocity " << max_velocity
                          << " exceeds tolerance " << velocity_tolerance_ << "\n";
            }
            return false;
        }
        return true;
    }

    /**
     * @brief Force timestep reduction (e.g., contact detected)
     * @param factor Reduction factor (0.1-0.9)
     */
    void force_reduction(Real factor = 0.5) {
        current_dt_ *= std::clamp(factor, 0.1, 0.9);
        current_dt_ = std::max(current_dt_, min_dt_);
        stable_steps_ = 0;
        unstable_steps_++;

        if (verbose_) {
            std::cout << "[AdaptiveTimestep] Forced dt reduction to " << current_dt_ << "\n";
        }
    }

    /**
     * @brief Reset controller state (for restart)
     */
    void reset() {
        current_dt_ = 0;
        previous_dt_ = 0;
        step_count_ = 0;
        stable_steps_ = 0;
        unstable_steps_ = 0;
        initial_energy_set_ = false;
        initial_energy_ = 0;
        energy_history_.clear();
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    Real current_dt() const { return current_dt_; }
    Real previous_dt() const { return previous_dt_; }
    size_t step_count() const { return step_count_; }
    size_t stable_steps() const { return stable_steps_; }
    size_t unstable_steps() const { return unstable_steps_; }

    /**
     * @brief Get current energy state
     */
    EnergyState current_energy() const {
        if (energy_history_.empty()) {
            return EnergyState();
        }
        return energy_history_.back();
    }

    /**
     * @brief Get energy error relative to initial
     */
    Real energy_error() const {
        if (!initial_energy_set_ || energy_history_.empty()) {
            return 0;
        }
        return std::abs(energy_history_.back().total - initial_energy_) / initial_energy_;
    }

    /**
     * @brief Print statistics
     */
    void print_stats(std::ostream& os = std::cout) const {
        os << "=== Adaptive Timestep Statistics ===\n";
        os << "Strategy: ";
        switch (strategy_) {
            case TimestepStrategy::Fixed: os << "Fixed\n"; break;
            case TimestepStrategy::CFL: os << "CFL\n"; break;
            case TimestepStrategy::EnergyBased: os << "EnergyBased\n"; break;
            case TimestepStrategy::Combined: os << "Combined\n"; break;
        }
        os << "Total steps: " << step_count_ << "\n";
        os << "Current dt: " << current_dt_ << "\n";
        os << "Target CFL: " << target_cfl_ << "\n";
        os << "dt range: [" << min_dt_ << ", " << max_dt_ << "]\n";
        os << "Stable steps: " << stable_steps_ << "\n";
        os << "Unstable steps: " << unstable_steps_ << "\n";
        if (initial_energy_set_) {
            os << "Initial energy: " << initial_energy_ << "\n";
            os << "Current energy error: " << (energy_error() * 100) << "%\n";
        }
        os << "====================================\n";
    }

private:
    // Strategy and parameters
    TimestepStrategy strategy_;
    Real target_cfl_;
    Real min_dt_;
    Real max_dt_;

    // Current state
    Real current_dt_;
    Real previous_dt_;

    // Adaptation parameters
    Real dt_growth_factor_;
    Real dt_shrink_factor_;
    Real energy_tolerance_;
    Real velocity_tolerance_;

    // Step counters
    size_t step_count_;
    size_t stable_steps_;
    size_t unstable_steps_;
    int min_stable_steps_;

    // Energy history for stability monitoring
    std::deque<EnergyState> energy_history_;
    size_t energy_history_size_;
    Real initial_energy_;
    bool initial_energy_set_;

    // Logging
    bool verbose_;
};

} // namespace physics
} // namespace nxs
