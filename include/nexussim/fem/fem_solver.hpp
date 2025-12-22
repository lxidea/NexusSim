#pragma once

/**
 * @file fem_solver.hpp
 * @brief FEM solver module for explicit dynamics
 *
 * Implements the PhysicsModule interface for finite element analysis
 * using explicit time integration (central difference method).
 */

#include <nexussim/core/core.hpp>
#include <nexussim/physics/module.hpp>
#include <nexussim/physics/time_integrator.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/physics/element.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/data/state.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <memory>
#include <vector>
#include <map>

namespace nxs {
namespace fem {

// ============================================================================
// Boundary Condition Types
// ============================================================================

enum class BCType {
    Displacement,    ///< Prescribed displacement
    Velocity,        ///< Prescribed velocity
    Acceleration,    ///< Prescribed acceleration
    Force,           ///< Applied force
    Pressure         ///< Applied pressure
};

/**
 * @brief Boundary condition specification
 */
struct BoundaryCondition {
    BCType type;
    std::vector<Index> nodes;  ///< Nodes where BC is applied
    Index dof;                  ///< DOF index (0=x, 1=y, 2=z, etc.)
    Real value;                 ///< BC value
    Real (*time_function)(Real t);  ///< Optional time-dependent function

    BoundaryCondition(BCType t, std::vector<Index> n, Index d, Real v)
        : type(t), nodes(std::move(n)), dof(d), value(v), time_function(nullptr)
    {}
};

/**
 * @brief Element group with material assignment
 */
struct ElementGroup {
    std::string name;
    physics::ElementType type;
    std::vector<Index> element_ids;
    std::shared_ptr<physics::Element> element;
    physics::MaterialProperties material;

    // Element connectivity (num_elements × nodes_per_element)
    std::vector<Index> connectivity;

    // GPU-ready connectivity (allocated on demand)
    Kokkos::View<Index*> connectivity_device;
    bool gpu_data_ready = false;
};

// ============================================================================
// FEM Solver Module
// ============================================================================

/**
 * @brief FEM solver for explicit dynamics
 *
 * Features:
 * - Explicit central difference time integration
 * - Lumped mass matrix (diagonal)
 * - Internal force calculation via element assembly
 * - Boundary conditions (displacement, force, pressure)
 * - CFL-based stable time step estimation
 * - GPU acceleration support via Kokkos
 */
class FEMSolver : public physics::PhysicsModule {
public:
    /**
     * @brief Constructor
     * @param name Solver name
     */
    FEMSolver(const std::string& name = "FEM");

    ~FEMSolver() override = default;

    // ========================================================================
    // PhysicsModule Interface Implementation
    // ========================================================================

    void initialize(std::shared_ptr<Mesh> mesh,
                   std::shared_ptr<State> state) override;

    void finalize() override;

    Real compute_stable_dt() const override;

    void step(Real dt) override;

    std::vector<std::string> provided_fields() const override;
    std::vector<std::string> required_fields() const override;

    void export_field(const std::string& field_name,
                     std::vector<Real>& data) const override;

    void import_field(const std::string& field_name,
                     const std::vector<Real>& data) override;

    // ========================================================================
    // FEM-Specific Configuration
    // ========================================================================

    /**
     * @brief Add element group with material
     */
    void add_element_group(const std::string& name,
                          physics::ElementType type,
                          const std::vector<Index>& element_ids,
                          const std::vector<Index>& connectivity,
                          const physics::MaterialProperties& material);

    /**
     * @brief Add boundary condition
     */
    void add_boundary_condition(const BoundaryCondition& bc);

    /**
     * @brief Set time integrator
     */
    void set_integrator(std::shared_ptr<physics::TimeIntegrator> integrator) {
        integrator_ = integrator;
    }

    /**
     * @brief Set CFL safety factor
     */
    void set_cfl_factor(Real factor) { cfl_factor_ = factor; }

    /**
     * @brief Enable/disable damping
     */
    void set_damping(Real damping_factor) { damping_factor_ = damping_factor; }

    /**
     * @brief Set gravity acceleration vector
     * @param gx Gravity in x-direction (m/s²)
     * @param gy Gravity in y-direction (m/s²)
     * @param gz Gravity in z-direction (m/s²)
     *
     * Default: (0, 0, -9.81) for standard gravity in -z direction
     */
    void set_gravity(Real gx, Real gy, Real gz) {
        gravity_[0] = gx;
        gravity_[1] = gy;
        gravity_[2] = gz;
        gravity_enabled_ = true;
    }

    /**
     * @brief Enable/disable gravity
     */
    void enable_gravity(bool enable) { gravity_enabled_ = enable; }

    /**
     * @brief Check if gravity is enabled
     */
    bool gravity_enabled() const { return gravity_enabled_; }

    /**
     * @brief Get gravity vector
     */
    const Real* gravity() const { return gravity_; }

    /**
     * @brief Add uniform body force (force per unit volume)
     * @param fx Body force in x-direction (N/m³)
     * @param fy Body force in y-direction (N/m³)
     * @param fz Body force in z-direction (N/m³)
     */
    void set_body_force(Real fx, Real fy, Real fz) {
        body_force_[0] = fx;
        body_force_[1] = fy;
        body_force_[2] = fz;
        body_force_enabled_ = true;
    }

    /**
     * @brief Enable/disable body force
     */
    void enable_body_force(bool enable) { body_force_enabled_ = enable; }

    // ========================================================================
    // Accessors
    // ========================================================================

    // Accessors for state vectors (returns host view)
    auto displacement() const { return displacement_.view_host(); }
    auto velocity() const { return velocity_.view_host(); }
    auto acceleration() const { return acceleration_.view_host(); }
    auto force_internal() const { return force_internal_.view_host(); }
    auto force_external() const { return force_external_.view_host(); }
    auto mass() const { return mass_.view_host(); }

    // GPU accessors (returns device view)
    auto displacement_device() const { return displacement_.view_device(); }
    auto velocity_device() const { return velocity_.view_device(); }
    auto acceleration_device() const { return acceleration_.view_device(); }
    auto force_internal_device() const { return force_internal_.view_device(); }
    auto force_external_device() const { return force_external_.view_device(); }
    auto mass_device() const { return mass_.view_device(); }

    std::size_t num_dof() const { return ndof_; }
    std::size_t num_nodes() const { return num_nodes_; }

public:
    // ========================================================================
    // Internal Methods (Public for GPU lambda access)
    // ========================================================================

    /**
     * @brief Assemble global lumped mass matrix
     */
    void assemble_mass_matrix();

    /**
     * @brief Compute internal forces from element stresses
     */
    void compute_internal_forces();

private:

    /**
     * @brief Apply boundary conditions (legacy - applies all)
     */
    void apply_boundary_conditions(Real time);

    /**
     * @brief Apply force boundary conditions (before time integration)
     */
    void apply_force_boundary_conditions(Real time);

    /**
     * @brief Apply displacement boundary conditions (after time integration)
     */
    void apply_displacement_boundary_conditions(Real time);

    /**
     * @brief Zero out forces
     */
    void zero_forces();

    /**
     * @brief Apply gravity and body forces to external force vector
     *
     * For gravity: f_ext += m * g (lumped mass approach)
     * For body force: f_ext += b * V_node (distributed volume)
     */
    void apply_body_forces();

    /**
     * @brief Compute element characteristic length
     */
    Real compute_element_size(const ElementGroup& group, Index elem_id) const;

    /**
     * @brief Prepare GPU data for element group
     */
    void prepare_gpu_element_data(ElementGroup& group);

    /**
     * @brief Compute wave speed for material
     */
    Real compute_wave_speed(const physics::MaterialProperties& mat) const;

    // ========================================================================
    // Member Variables
    // ========================================================================

    // Problem size
    std::size_t num_nodes_;
    std::size_t ndof_;
    Index dof_per_node_;

    // State vectors (GPU-accelerated with Kokkos::DualView)
    // DualView allows both host and device access with explicit synchronization
    Kokkos::DualView<Real*> displacement_;
    Kokkos::DualView<Real*> velocity_;
    Kokkos::DualView<Real*> acceleration_;
    Kokkos::DualView<Real*> force_internal_;
    Kokkos::DualView<Real*> force_external_;
    Kokkos::DualView<Real*> mass_;

    // Cached mesh data on device (optimization to avoid repeated transfers)
    Kokkos::View<Real**> coords_device_;  ///< Node coordinates on device (num_nodes × 3)
    bool coords_device_ready_ = false;    ///< Flag indicating coordinates are cached

    // Element groups
    std::vector<ElementGroup> element_groups_;

    // Boundary conditions
    std::vector<BoundaryCondition> boundary_conditions_;

    // Time integrator
    std::shared_ptr<physics::TimeIntegrator> integrator_;

    // Solver parameters
    Real cfl_factor_;        ///< CFL safety factor (default: 0.9)
    Real damping_factor_;    ///< Numerical damping (default: 0.0)

    // Gravity and body forces
    Real gravity_[3];        ///< Gravity acceleration vector (m/s²)
    bool gravity_enabled_;   ///< Whether gravity is active
    Real body_force_[3];     ///< Body force per unit volume (N/m³)
    bool body_force_enabled_;///< Whether body force is active

    // Element stresses (for internal force calculation)
    std::map<Index, std::vector<Real>> element_stresses_;

    // Zero-mass DOFs that need to be constrained (unused nodes in structured meshes)
    std::vector<Index> zero_mass_dofs_;
};

} // namespace fem
} // namespace nxs
