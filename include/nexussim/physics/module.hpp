#pragma once

#include <nexussim/core/types.hpp>
#include <nexussim/core/exception.hpp>
#include <nexussim/core/logger.hpp>
#include <nexussim/data/state.hpp>
#include <nexussim/data/mesh.hpp>
#include <memory>
#include <string>
#include <vector>

namespace nxs {
namespace physics {

// ============================================================================
// PhysicsModule Base Class
// ============================================================================

/**
 * @brief Base class for all physics modules (FEM, Peridynamics, SPH, etc.)
 *
 * This class defines the interface that all physics modules must implement.
 * It provides hooks for initialization, time stepping, field exchange for
 * coupling, and GPU acceleration.
 */
class PhysicsModule {
public:
    enum class Type {
        FEM,           ///< Finite Element Method
        Peridynamics,  ///< Peridynamic method
        SPH,           ///< Smoothed Particle Hydrodynamics
        MPM,           ///< Material Point Method
        DEM,           ///< Discrete Element Method
        Custom         ///< User-defined custom physics
    };

    enum class Status {
        Uninitialized,
        Ready,
        Running,
        Converged,
        Failed
    };

    /**
     * @brief Constructor
     * @param name Module name
     * @param type Physics module type
     */
    PhysicsModule(const std::string& name, Type type)
        : name_(name)
        , type_(type)
        , status_(Status::Uninitialized)
        , current_time_(0.0)
        , time_step_(0.0)
        , step_count_(0)
    {}

    virtual ~PhysicsModule() = default;

    // Prevent copying
    PhysicsModule(const PhysicsModule&) = delete;
    PhysicsModule& operator=(const PhysicsModule&) = delete;

    // ========================================================================
    // Lifecycle Management
    // ========================================================================

    /**
     * @brief Initialize the physics module
     * @param mesh Computational mesh
     * @param state Initial state
     */
    virtual void initialize(std::shared_ptr<Mesh> mesh,
                           std::shared_ptr<State> state) = 0;

    /**
     * @brief Finalize the physics module
     */
    virtual void finalize() {
        status_ = Status::Uninitialized;
    }

    // ========================================================================
    // Time Integration
    // ========================================================================

    /**
     * @brief Compute stable time step
     * @return Stable time step size
     */
    virtual Real compute_stable_dt() const = 0;

    /**
     * @brief Advance solution by one time step
     * @param dt Time step size
     */
    virtual void step(Real dt) = 0;

    /**
     * @brief Get current simulation time
     */
    Real current_time() const { return current_time_; }

    /**
     * @brief Get current time step size
     */
    Real time_step() const { return time_step_; }

    /**
     * @brief Get step count
     */
    std::size_t step_count() const { return step_count_; }

    // ========================================================================
    // Field Exchange (for coupling)
    // ========================================================================

    /**
     * @brief Get list of fields this module provides
     */
    virtual std::vector<std::string> provided_fields() const = 0;

    /**
     * @brief Get list of fields this module requires
     */
    virtual std::vector<std::string> required_fields() const = 0;

    /**
     * @brief Export field data for coupling
     * @param field_name Name of the field to export
     * @param data Output data buffer
     */
    virtual void export_field(const std::string& field_name,
                              std::vector<Real>& data) const = 0;

    /**
     * @brief Import field data from coupling
     * @param field_name Name of the field to import
     * @param data Input data buffer
     */
    virtual void import_field(const std::string& field_name,
                              const std::vector<Real>& data) = 0;

    // ========================================================================
    // Module Information
    // ========================================================================

    const std::string& name() const { return name_; }
    Type type() const { return type_; }
    Status status() const { return status_; }

    std::string type_string() const {
        switch (type_) {
            case Type::FEM: return "FEM";
            case Type::Peridynamics: return "Peridynamics";
            case Type::SPH: return "SPH";
            case Type::MPM: return "MPM";
            case Type::DEM: return "DEM";
            case Type::Custom: return "Custom";
            default: return "Unknown";
        }
    }

    // ========================================================================
    // Performance Monitoring
    // ========================================================================

    /**
     * @brief Get total computation time (seconds)
     */
    Real total_time() const { return total_computation_time_; }

    /**
     * @brief Get average time per step (seconds)
     */
    Real average_step_time() const {
        return step_count_ > 0 ? total_computation_time_ / step_count_ : 0.0;
    }

protected:
    // Module information
    std::string name_;
    Type type_;
    Status status_;

    // Time integration state
    Real current_time_;
    Real time_step_;
    std::size_t step_count_;

    // Mesh and state
    std::shared_ptr<Mesh> mesh_;
    std::shared_ptr<State> state_;

    // Performance tracking
    Real total_computation_time_ = 0.0;

    /**
     * @brief Update module status
     */
    void set_status(Status status) { status_ = status; }

    /**
     * @brief Advance time
     */
    void advance_time(Real dt) {
        current_time_ += dt;
        time_step_ = dt;
        ++step_count_;
    }
};

} // namespace physics
} // namespace nxs
