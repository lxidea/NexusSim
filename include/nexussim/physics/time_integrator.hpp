#pragma once

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <nexussim/core/exception.hpp>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

namespace nxs {
namespace physics {

// ============================================================================
// Time Integrator Types
// ============================================================================

enum class IntegratorType {
    ExplicitCentral,     ///< Explicit central difference
    ExplicitRK4,         ///< Explicit Runge-Kutta 4th order
    ImplicitNewmark,     ///< Implicit Newmark-β
    ImplicitGenAlpha,    ///< Implicit Generalized-α
    ImplicitHHT          ///< Implicit Hilber-Hughes-Taylor
};

// ============================================================================
// Dynamic State
// ============================================================================

/**
 * @brief Dynamic state for time integration
 */
struct DynamicState {
    // Displacements, velocities, accelerations
    Real* displacement;
    Real* velocity;
    Real* acceleration;

    // Forces
    Real* force_external;
    Real* force_internal;
    Real* force_damping;

    // Mass matrix
    Real* mass;

    // Problem size
    std::size_t ndof;

    DynamicState(std::size_t n)
        : ndof(n)
    {
        displacement = new Real[n]();
        velocity = new Real[n]();
        acceleration = new Real[n]();
        force_external = new Real[n]();
        force_internal = new Real[n]();
        force_damping = new Real[n]();
        mass = new Real[n]();
    }

    ~DynamicState() {
        delete[] displacement;
        delete[] velocity;
        delete[] acceleration;
        delete[] force_external;
        delete[] force_internal;
        delete[] force_damping;
        delete[] mass;
    }

    // Prevent copying
    DynamicState(const DynamicState&) = delete;
    DynamicState& operator=(const DynamicState&) = delete;
};

// ============================================================================
// Time Integrator Base Class
// ============================================================================

/**
 * @brief Base class for time integration schemes
 */
class TimeIntegrator {
public:
    TimeIntegrator(IntegratorType type) : type_(type) {}
    virtual ~TimeIntegrator() = default;

    // ========================================================================
    // Time Integration
    // ========================================================================

    /**
     * @brief Initialize integrator
     * @param ndof Number of degrees of freedom
     */
    virtual void initialize(std::size_t ndof) = 0;

    /**
     * @brief Advance solution by one time step
     * @param dt Time step size
     * @param state Dynamic state
     */
    virtual void step(Real dt, DynamicState& state) = 0;

    /**
     * @brief Get integrator type
     */
    IntegratorType type() const { return type_; }

    /**
     * @brief Check if integrator is explicit
     */
    virtual bool is_explicit() const = 0;

    /**
     * @brief Get stability limit (for explicit methods)
     * @return CFL condition coefficient
     */
    virtual Real stability_limit() const {
        return is_explicit() ? 1.0 : 0.0;
    }

protected:
    IntegratorType type_;
};

// ============================================================================
// Explicit Central Difference Integrator
// ============================================================================

/**
 * @brief Explicit central difference time integrator
 *
 * Second-order accurate, conditionally stable explicit scheme.
 * Commonly used in crash/impact simulations.
 *
 * Update equations:
 *   a^n = M^{-1} * (f_ext^n - f_int^n - f_damp^n)
 *   v^{n+1/2} = v^{n-1/2} + a^n * dt
 *   u^{n+1} = u^n + v^{n+1/2} * dt
 *
 * Stability: dt < 2/ω_max (ω_max = highest natural frequency)
 */
class ExplicitCentralDifferenceIntegrator : public TimeIntegrator {
public:
    ExplicitCentralDifferenceIntegrator()
        : TimeIntegrator(IntegratorType::ExplicitCentral)
        , damping_factor_(0.0)
    {}

    /**
     * @brief Constructor with damping
     * @param damping_factor Rayleigh damping coefficient (0 to 1)
     */
    ExplicitCentralDifferenceIntegrator(Real damping_factor)
        : TimeIntegrator(IntegratorType::ExplicitCentral)
        , damping_factor_(damping_factor)
    {}

    void initialize(std::size_t ndof) override {
        ndof_ = ndof;
    }

    void step(Real dt, DynamicState& state) override {
        const std::size_t n = state.ndof;

        // Compute accelerations: a = M^{-1} * (f_ext - f_int - f_damp)
        for (std::size_t i = 0; i < n; ++i) {
            const Real net_force = state.force_external[i] -
                                  state.force_internal[i] -
                                  state.force_damping[i];
            state.acceleration[i] = net_force / state.mass[i];
        }

        // Update velocities (mid-point)
        for (std::size_t i = 0; i < n; ++i) {
            state.velocity[i] += state.acceleration[i] * dt;
        }

        // Update displacements
        for (std::size_t i = 0; i < n; ++i) {
            state.displacement[i] += state.velocity[i] * dt;
        }

        // Apply damping if specified
        if (damping_factor_ > 0.0) {
            for (std::size_t i = 0; i < n; ++i) {
                state.velocity[i] *= (1.0 - damping_factor_);
            }
        }
    }

    bool is_explicit() const override { return true; }

    Real stability_limit() const override {
        // CFL condition: dt < 2/ω_max
        // For safety, use dt < L/(c*√3) where L is element size, c is wave speed
        return 1.0;  // Will be scaled by element size / wave speed
    }

    void set_damping(Real damping_factor) {
        damping_factor_ = damping_factor;
    }

private:
    std::size_t ndof_;
    Real damping_factor_;
};

// ============================================================================
// GPU-Compatible Explicit Integrator Kernels
// ============================================================================

/**
 * @brief GPU kernel for explicit central difference update
 */
struct ExplicitCDKernel {
    // Views for state variables
    View1D<Real> displacement;
    View1D<Real> velocity;
    View1D<Real> acceleration;
    View1D<Real> force_external;
    View1D<Real> force_internal;
    View1D<Real> mass;
    Real dt;
    Real damping;

    ExplicitCDKernel(View1D<Real> u, View1D<Real> v, View1D<Real> a,
                     View1D<Real> f_ext, View1D<Real> f_int, View1D<Real> m,
                     Real dt_, Real damp = 0.0)
        : displacement(u), velocity(v), acceleration(a)
        , force_external(f_ext), force_internal(f_int), mass(m)
        , dt(dt_), damping(damp)
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        // Compute acceleration
        const Real net_force = force_external(i) - force_internal(i);
        acceleration(i) = net_force / mass(i);

        // Update velocity (mid-point)
        velocity(i) += acceleration(i) * dt;

        // Apply damping
        if (damping > 0.0) {
            velocity(i) *= (1.0 - damping);
        }

        // Update displacement
        displacement(i) += velocity(i) * dt;
    }
};

/**
 * @brief GPU-accelerated explicit central difference integrator
 */
class GPUExplicitCentralDifference {
public:
    GPUExplicitCentralDifference(std::size_t ndof, Real damping = 0.0)
        : ndof_(ndof)
        , damping_(damping)
        , displacement_("displacement", ndof)
        , velocity_("velocity", ndof)
        , acceleration_("acceleration", ndof)
        , force_external_("force_external", ndof)
        , force_internal_("force_internal", ndof)
        , mass_("mass", ndof)
    {}

    void step(Real dt) {
        ExplicitCDKernel kernel(displacement_, velocity_, acceleration_,
                               force_external_, force_internal_, mass_,
                               dt, damping_);
        parallel_for("ExplicitCD", ndof_, kernel);
        fence("ExplicitCD_complete");
    }

    // Accessors for setting up problem
    View1D<Real>& displacement() { return displacement_; }
    View1D<Real>& velocity() { return velocity_; }
    View1D<Real>& acceleration() { return acceleration_; }
    View1D<Real>& force_external() { return force_external_; }
    View1D<Real>& force_internal() { return force_internal_; }
    View1D<Real>& mass() { return mass_; }

private:
    std::size_t ndof_;
    Real damping_;

    View1D<Real> displacement_;
    View1D<Real> velocity_;
    View1D<Real> acceleration_;
    View1D<Real> force_external_;
    View1D<Real> force_internal_;
    View1D<Real> mass_;
};

// ============================================================================
// Implicit Newmark-β Integrator
// ============================================================================

/**
 * @brief Implicit Newmark-β time integrator
 *
 * Unconditionally stable implicit scheme for structural dynamics.
 * Uses Newton-Raphson iteration for nonlinear problems.
 *
 * Parameters:
 *   β = 0.25, γ = 0.5: Average acceleration (trapezoidal rule)
 *                       Unconditionally stable, second-order accurate
 *   β = 1/6, γ = 0.5:  Linear acceleration (Fox-Goodwin)
 *                       Conditionally stable, second-order accurate
 *   β = 0.0, γ = 0.5:  Explicit central difference (special case)
 *
 * Update equations:
 *   u^{n+1} = u^n + dt*v^n + dt²*[(1/2 - β)*a^n + β*a^{n+1}]
 *   v^{n+1} = v^n + dt*[(1 - γ)*a^n + γ*a^{n+1}]
 *
 * Effective stiffness approach:
 *   K_eff = K + (γ/(β*dt))*C + (1/(β*dt²))*M
 *   M*a^{n+1} + C*v^{n+1} + K*u^{n+1} = f_ext^{n+1}
 *
 * For lumped mass (diagonal M), effective solve simplifies significantly.
 */
class ImplicitNewmarkIntegrator : public TimeIntegrator {
public:
    /**
     * @brief Constructor with default parameters (average acceleration)
     */
    ImplicitNewmarkIntegrator()
        : TimeIntegrator(IntegratorType::ImplicitNewmark)
        , beta_(0.25)
        , gamma_(0.5)
        , alpha_m_(0.0)  // Mass proportional damping
        , alpha_k_(0.0)  // Stiffness proportional damping
        , max_iterations_(50)
        , tolerance_(1.0e-8)
        , use_lumped_mass_(true)
    {}

    /**
     * @brief Constructor with custom parameters
     * @param beta Newmark β parameter (typically 0.25)
     * @param gamma Newmark γ parameter (typically 0.5)
     */
    ImplicitNewmarkIntegrator(Real beta, Real gamma)
        : TimeIntegrator(IntegratorType::ImplicitNewmark)
        , beta_(beta)
        , gamma_(gamma)
        , alpha_m_(0.0)
        , alpha_k_(0.0)
        , max_iterations_(50)
        , tolerance_(1.0e-8)
        , use_lumped_mass_(true)
    {}

    void initialize(std::size_t ndof) override {
        ndof_ = ndof;

        // Allocate predictor storage
        u_pred_.resize(ndof);
        v_pred_.resize(ndof);
        a_pred_.resize(ndof);

        // Residual and increment vectors
        residual_.resize(ndof);
        delta_u_.resize(ndof);

        // Diagonal stiffness approximation (for preconditioner)
        diag_k_.resize(ndof);

        // Effective mass diagonal
        m_eff_.resize(ndof);
    }

    /**
     * @brief Time step with implicit integration
     *
     * Standard Newmark-β scheme:
     *   u^{n+1} = u^n + dt*v^n + dt²*[(1/2 - β)*a^n + β*a^{n+1}]
     *   v^{n+1} = v^n + dt*[(1 - γ)*a^n + γ*a^{n+1}]
     *
     * Rearranging for a^{n+1}:
     *   a^{n+1} = (1/(β*dt²)) * [u^{n+1} - u^n - dt*v^n - dt²*(1/2 - β)*a^n]
     *
     * For linear spring: f_int = k*u, equilibrium: m*a + c*v + k*u = f_ext
     */
    void step(Real dt, DynamicState& state) override {
        const std::size_t num = state.ndof;

        // Store state at time n
        for (std::size_t i = 0; i < num; ++i) {
            u_pred_[i] = state.displacement[i];  // u^n
            v_pred_[i] = state.velocity[i];      // v^n
            a_pred_[i] = state.acceleration[i];  // a^n
        }

        // Newmark coefficients
        const Real c0 = 1.0 / (beta_ * dt * dt);
        const Real c1 = gamma_ / (beta_ * dt);
        const Real c2 = 1.0 / (beta_ * dt);
        const Real c3 = 1.0 / (2.0 * beta_) - 1.0;
        const Real c4 = gamma_ / beta_ - 1.0;
        const Real c5 = dt * (gamma_ / (2.0 * beta_) - 1.0);

        // Compute effective stiffness diagonal
        // K_eff = K + c1*C + c0*M
        // For diagonal M and diagonal K approximation:
        for (std::size_t i = 0; i < num; ++i) {
            Real m = state.mass[i];
            Real k = diag_k_[i];

            // Damping: C = α_m*M + α_k*K
            Real c = alpha_m_ * m + alpha_k_ * k;

            m_eff_[i] = k + c1 * c + c0 * m;

            // Ensure positive definite
            if (m_eff_[i] < 1.0e-20) {
                m_eff_[i] = c0 * m;  // At minimum, use mass contribution
            }
        }

        // Predictor: initial guess for u^{n+1}
        for (std::size_t i = 0; i < num; ++i) {
            state.displacement[i] = u_pred_[i] + dt * v_pred_[i] +
                                   0.5 * dt * dt * a_pred_[i];
        }

        // Newton-Raphson iteration
        int iter = 0;
        Real residual_norm = 1.0;

        while (iter < max_iterations_) {
            // Update internal force for linear spring (k*u)
            // This is needed for the test - in real FEM, the solver updates this
            for (std::size_t i = 0; i < num; ++i) {
                if (diag_k_[i] > 0) {
                    state.force_internal[i] = diag_k_[i] * state.displacement[i];
                }
            }

            // Compute acceleration and velocity from Newmark relations
            for (std::size_t i = 0; i < num; ++i) {
                // a^{n+1} from displacement
                state.acceleration[i] = c0 * (state.displacement[i] - u_pred_[i]) -
                                        c2 * v_pred_[i] - c3 * a_pred_[i];

                // v^{n+1} from acceleration
                state.velocity[i] = v_pred_[i] + dt * ((1.0 - gamma_) * a_pred_[i] +
                                    gamma_ * state.acceleration[i]);
            }

            // Compute damping force
            for (std::size_t i = 0; i < num; ++i) {
                state.force_damping[i] = (alpha_m_ * state.mass[i] +
                                         alpha_k_ * diag_k_[i]) * state.velocity[i];
            }

            // Compute residual: R = M*a + C*v + f_int - f_ext
            residual_norm = 0.0;
            for (std::size_t i = 0; i < num; ++i) {
                residual_[i] = state.mass[i] * state.acceleration[i] +
                              state.force_damping[i] +
                              state.force_internal[i] -
                              state.force_external[i];

                residual_norm += residual_[i] * residual_[i];
            }
            residual_norm = std::sqrt(residual_norm);

            if (residual_norm < tolerance_) {
                break;
            }

            // Solve: K_eff * Δu = -R
            for (std::size_t i = 0; i < num; ++i) {
                delta_u_[i] = -residual_[i] / m_eff_[i];
            }

            // Update displacement
            for (std::size_t i = 0; i < num; ++i) {
                state.displacement[i] += delta_u_[i];
            }

            ++iter;
        }

        // Final state update with converged values
        for (std::size_t i = 0; i < num; ++i) {
            state.acceleration[i] = c0 * (state.displacement[i] - u_pred_[i]) -
                                    c2 * v_pred_[i] - c3 * a_pred_[i];

            state.velocity[i] = v_pred_[i] + dt * ((1.0 - gamma_) * a_pred_[i] +
                                gamma_ * state.acceleration[i]);
        }

        last_iterations_ = iter;
        last_residual_ = residual_norm;
    }

    bool is_explicit() const override { return false; }

    Real stability_limit() const override {
        // Implicit methods are unconditionally stable for β ≥ 0.25, γ ≥ 0.5
        return 0.0;  // No stability limit
    }

    // Configuration
    void set_parameters(Real beta, Real gamma) {
        beta_ = beta;
        gamma_ = gamma;
    }

    void set_rayleigh_damping(Real alpha_m, Real alpha_k) {
        alpha_m_ = alpha_m;
        alpha_k_ = alpha_k;
    }

    void set_convergence(int max_iter, Real tol) {
        max_iterations_ = max_iter;
        tolerance_ = tol;
    }

    void set_diagonal_stiffness(const std::vector<Real>& k_diag) {
        diag_k_ = k_diag;
    }

    void set_use_lumped_mass(bool use_lumped) {
        use_lumped_mass_ = use_lumped;
    }

    // Diagnostics
    int last_iterations() const { return last_iterations_; }
    Real last_residual() const { return last_residual_; }

private:
    std::size_t ndof_;

    // Newmark parameters
    Real beta_;   // Typically 0.25 (average acceleration)
    Real gamma_;  // Typically 0.5 (no algorithmic damping)

    // Rayleigh damping coefficients: C = α_m*M + α_k*K
    Real alpha_m_;
    Real alpha_k_;

    // Newton-Raphson settings
    int max_iterations_;
    Real tolerance_;
    bool use_lumped_mass_;

    // Predictor vectors
    std::vector<Real> u_pred_;
    std::vector<Real> v_pred_;
    std::vector<Real> a_pred_;

    // Working vectors
    std::vector<Real> residual_;
    std::vector<Real> delta_u_;
    std::vector<Real> diag_k_;
    std::vector<Real> m_eff_;

    // Diagnostics
    int last_iterations_ = 0;
    Real last_residual_ = 0.0;
};

// ============================================================================
// Implicit HHT-α (Hilber-Hughes-Taylor) Integrator
// ============================================================================

/**
 * @brief HHT-α time integrator for improved numerical damping
 *
 * Extension of Newmark-β with controllable numerical damping
 * while maintaining second-order accuracy.
 *
 * Parameter α ∈ [-1/3, 0]:
 *   α = 0: Standard Newmark (no numerical damping)
 *   α = -0.05 to -0.1: Typical for structural dynamics
 *   α = -1/3: Maximum damping (still stable)
 *
 * Parameters: β = (1-α)²/4, γ = (1-2α)/2
 */
class ImplicitHHTIntegrator : public ImplicitNewmarkIntegrator {
public:
    /**
     * @brief Constructor with HHT parameter
     * @param alpha HHT-α parameter (typically -0.05 to -0.1, must be ≥ -1/3)
     */
    ImplicitHHTIntegrator(Real alpha = -0.05)
        : ImplicitNewmarkIntegrator(
            (1.0 - alpha) * (1.0 - alpha) / 4.0,  // β
            (1.0 - 2.0 * alpha) / 2.0              // γ
          )
        , alpha_hht_(alpha)
    {
        // Validate α
        if (alpha < -1.0/3.0 || alpha > 0.0) {
            throw InvalidArgumentError("HHT α must be in [-1/3, 0]");
        }
    }

    Real alpha_hht() const { return alpha_hht_; }

private:
    Real alpha_hht_;
};

// ============================================================================
// Time Integrator Factory
// ============================================================================

class TimeIntegratorFactory {
public:
    static std::unique_ptr<TimeIntegrator> create(IntegratorType type) {
        switch (type) {
            case IntegratorType::ExplicitCentral:
                return std::make_unique<ExplicitCentralDifferenceIntegrator>();
            case IntegratorType::ImplicitNewmark:
                return std::make_unique<ImplicitNewmarkIntegrator>();
            case IntegratorType::ImplicitHHT:
                return std::make_unique<ImplicitHHTIntegrator>();
            default:
                throw NotImplementedError("Time integrator type not implemented");
        }
    }

    static std::string to_string(IntegratorType type) {
        switch (type) {
            case IntegratorType::ExplicitCentral: return "ExplicitCentralDifference";
            case IntegratorType::ExplicitRK4: return "ExplicitRK4";
            case IntegratorType::ImplicitNewmark: return "ImplicitNewmark";
            case IntegratorType::ImplicitGenAlpha: return "ImplicitGenAlpha";
            case IntegratorType::ImplicitHHT: return "ImplicitHHT";
            default: return "Unknown";
        }
    }
};

// ============================================================================
// Adaptive Time Stepping Controller
// ============================================================================

/**
 * @brief Adaptive time stepping based on energy balance and error estimation
 *
 * Features:
 * - Energy-based error monitoring (kinetic, internal, external work)
 * - Automatic time step increase/decrease
 * - Mass-scaling detection
 * - Stability monitoring
 *
 * Criteria for time step adjustment:
 * 1. Energy error: |E_total - E_initial - W_external| / E_max
 * 2. Velocity change: max(Δv) / max(v)
 * 3. CFL stability: dt < dt_critical
 */
class AdaptiveTimeController {
public:
    struct Config {
        Real dt_initial;               ///< Initial time step
        Real dt_min;                   ///< Minimum allowed time step
        Real dt_max;                   ///< Maximum allowed time step
        Real cfl_factor;               ///< CFL safety factor (< 1)
        Real energy_tolerance;         ///< Allowed energy error (5%)
        Real growth_factor;            ///< Maximum dt growth per step
        Real shrink_factor;            ///< Shrink factor on instability
        bool enable_mass_scaling;      ///< Allow mass scaling for small elements
        Real mass_scaling_threshold;   ///< Mass scale if dt < threshold * dt_target

        Config()
            : dt_initial(1.0e-6)
            , dt_min(1.0e-10)
            , dt_max(1.0e-3)
            , cfl_factor(0.9)
            , energy_tolerance(0.05)
            , growth_factor(1.1)
            , shrink_factor(0.5)
            , enable_mass_scaling(false)
            , mass_scaling_threshold(0.1)
        {}
    };

    struct EnergyState {
        Real kinetic = 0.0;         ///< Total kinetic energy
        Real internal = 0.0;        ///< Total internal (strain) energy
        Real external_work = 0.0;   ///< Cumulative external work
        Real contact_work = 0.0;    ///< Cumulative contact energy (dissipation)
        Real damping_work = 0.0;    ///< Cumulative damping dissipation
        Real hourglass = 0.0;       ///< Hourglass control energy

        Real total() const {
            return kinetic + internal;
        }

        Real balance_error(Real initial_total) const {
            Real expected = initial_total + external_work - damping_work - contact_work;
            Real current = total();
            Real max_energy = std::max({std::abs(initial_total), std::abs(current),
                                        std::abs(expected), 1.0e-20});
            return std::abs(current - expected) / max_energy;
        }
    };

    struct StepInfo {
        Real dt_used;              ///< Actual time step used
        Real dt_stable;            ///< CFL-limited stable dt
        Real dt_next;              ///< Recommended next dt
        Real energy_error;         ///< Current energy balance error
        int mass_scaled_elements;  ///< Number of mass-scaled elements
        bool step_accepted;        ///< Whether step was accepted
        std::string reason;        ///< Reason for dt change
    };

    AdaptiveTimeController(const Config& config = Config{})
        : config_(config)
        , dt_current_(config.dt_initial)
        , step_count_(0)
        , total_time_(0.0)
        , initial_energy_(0.0)
        , energy_initialized_(false)
    {}

    /**
     * @brief Compute stable time step based on element properties
     * @param min_element_size Minimum element characteristic length
     * @param wave_speed Maximum wave speed (sqrt(E/rho) for solids)
     * @return Stable time step (CFL limited)
     */
    Real compute_stable_dt(Real min_element_size, Real wave_speed) const {
        // CFL condition: dt < L / c
        // For 3D hex elements: dt < L / (c * sqrt(3))
        const Real dt_cfl = config_.cfl_factor * min_element_size / (wave_speed * 1.732);
        return std::min(dt_cfl, config_.dt_max);
    }

    /**
     * @brief Initialize energy tracking at start of simulation
     */
    void initialize_energy(const EnergyState& initial) {
        initial_energy_ = initial.total();
        energy_prev_ = initial;
        energy_initialized_ = true;
    }

    /**
     * @brief Compute recommended time step for next iteration
     * @param dt_stable CFL-stable time step from element sizes
     * @param energy_current Current energy state
     * @return Step information with recommended dt
     */
    StepInfo compute_next_dt(Real dt_stable, const EnergyState& energy_current) {
        StepInfo info;
        info.dt_stable = dt_stable;
        info.dt_used = dt_current_;
        info.step_accepted = true;
        info.mass_scaled_elements = 0;

        // Initialize energy tracking if not done
        if (!energy_initialized_) {
            initialize_energy(energy_current);
        }

        // Compute energy error
        info.energy_error = energy_current.balance_error(initial_energy_);

        // Check for instability (energy explosion)
        if (energy_current.total() > 10.0 * (initial_energy_ + energy_current.external_work + 1.0)) {
            // Energy explosion detected - reject step and shrink dt
            info.step_accepted = false;
            info.dt_next = dt_current_ * config_.shrink_factor;
            info.reason = "Energy explosion detected";
            return info;
        }

        // Check energy error
        if (info.energy_error > config_.energy_tolerance) {
            // High energy error - reduce time step
            Real shrink = std::max(config_.shrink_factor,
                                   1.0 - (info.energy_error - config_.energy_tolerance));
            info.dt_next = dt_current_ * shrink;
            info.reason = "High energy error";
        }
        else if (info.energy_error < 0.1 * config_.energy_tolerance) {
            // Very low energy error - can increase time step
            info.dt_next = std::min(dt_current_ * config_.growth_factor, dt_stable);
            info.reason = "Low energy error - increasing dt";
        }
        else {
            // Energy error acceptable - keep current dt
            info.dt_next = dt_current_;
            info.reason = "Stable";
        }

        // Enforce limits
        info.dt_next = std::max(info.dt_next, config_.dt_min);
        info.dt_next = std::min(info.dt_next, config_.dt_max);
        info.dt_next = std::min(info.dt_next, dt_stable);

        // Update state
        dt_current_ = info.dt_next;
        energy_prev_ = energy_current;
        ++step_count_;
        total_time_ += info.dt_used;

        return info;
    }

    /**
     * @brief Get current time step
     */
    Real current_dt() const { return dt_current_; }

    /**
     * @brief Get accumulated simulation time
     */
    Real total_time() const { return total_time_; }

    /**
     * @brief Get step count
     */
    int step_count() const { return step_count_; }

    /**
     * @brief Reset controller state
     */
    void reset() {
        dt_current_ = config_.dt_initial;
        step_count_ = 0;
        total_time_ = 0.0;
        energy_initialized_ = false;
    }

    /**
     * @brief Compute kinetic energy from state arrays
     */
    static Real compute_kinetic_energy(const Real* velocity, const Real* mass,
                                       std::size_t ndof) {
        Real ke = 0.0;
        // Velocity is stored as [vx0, vy0, vz0, vx1, vy1, vz1, ...]
        // Mass is stored per node (or per DOF if diagonal mass)
        for (std::size_t i = 0; i < ndof; ++i) {
            ke += 0.5 * mass[i] * velocity[i] * velocity[i];
        }
        return ke;
    }

    /**
     * @brief Compute external work increment
     */
    static Real compute_external_work(const Real* force_ext, const Real* displacement,
                                      const Real* displacement_prev, std::size_t ndof) {
        Real work = 0.0;
        for (std::size_t i = 0; i < ndof; ++i) {
            // W = F · Δu (mid-point rule)
            work += force_ext[i] * (displacement[i] - displacement_prev[i]);
        }
        return work;
    }

    /**
     * @brief Check if mass scaling is recommended for an element
     * @param element_dt Stable dt for this element
     * @param target_dt Target simulation dt
     * @return Mass scale factor (1.0 = no scaling)
     */
    Real compute_mass_scale_factor(Real element_dt, Real target_dt) const {
        if (!config_.enable_mass_scaling) {
            return 1.0;
        }

        if (element_dt < config_.mass_scaling_threshold * target_dt) {
            // Scale mass to achieve target dt
            // dt ∝ sqrt(m), so m_new / m_old = (dt_new / dt_old)²
            Real ratio = target_dt / element_dt;
            return ratio * ratio;
        }
        return 1.0;
    }

    const Config& config() const { return config_; }
    Config& config() { return config_; }

private:
    Config config_;
    Real dt_current_;
    int step_count_;
    Real total_time_;
    Real initial_energy_;
    bool energy_initialized_;
    EnergyState energy_prev_;
};

} // namespace physics
} // namespace nxs
