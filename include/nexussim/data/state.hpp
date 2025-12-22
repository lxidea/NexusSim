#pragma once

#include <nexussim/core/core.hpp>
#include <nexussim/data/field.hpp>
#include <nexussim/data/mesh.hpp>
#include <map>
#include <string>

namespace nxs {

// ============================================================================
// State - Manages All Simulation State Data
// ============================================================================

/**
 * @brief State manages all time-dependent fields and simulation state
 *
 * The State class holds all physical quantities (displacement, velocity, stress, etc.)
 * and provides a unified interface for accessing and updating them.
 */
class State {
public:
    // ========================================================================
    // Constructors
    // ========================================================================

    explicit State(const Mesh& mesh)
        : mesh_(&mesh)
        , time_(0.0)
        , step_(0)
    {
        initialize_default_fields();
    }

    // ========================================================================
    // Time and Step Management
    // ========================================================================

    Real time() const { return time_; }
    void set_time(Real t) { time_ = t; }
    void advance_time(Real dt) { time_ += dt; }

    Int64 step() const { return step_; }
    void set_step(Int64 s) { step_ = s; }
    void advance_step() { ++step_; }

    // ========================================================================
    // Field Management
    // ========================================================================

    // Add a new field
    void add_field(const std::string& name, Field<Real>&& field) {
        if (fields_.count(name) > 0) {
            NXS_LOG_WARN("Field '{}' already exists, overwriting", name);
        }
        fields_[name] = std::move(field);
        // Access the moved-to object, not the moved-from one
        const auto& added_field = fields_[name];
        NXS_LOG_DEBUG("Added field '{}' (type: {}, location: {}, entities: {})",
                     name, to_string(added_field.type()), to_string(added_field.location()),
                     added_field.num_entities());
    }

    // Check if field exists
    bool has_field(const std::string& name) const {
        return fields_.count(name) > 0;
    }

    // Get field (mutable)
    Field<Real>& field(const std::string& name) {
        auto it = fields_.find(name);
        if (it == fields_.end()) {
            throw InvalidArgumentError("Field '" + name + "' not found");
        }
        return it->second;
    }

    // Get field (const)
    const Field<Real>& field(const std::string& name) const {
        auto it = fields_.find(name);
        if (it == fields_.end()) {
            throw InvalidArgumentError("Field '" + name + "' not found");
        }
        return it->second;
    }

    // Get all field names
    std::vector<std::string> field_names() const {
        std::vector<std::string> names;
        names.reserve(fields_.size());
        for (const auto& [name, _] : fields_) {
            names.push_back(name);
        }
        return names;
    }

    // Remove field
    void remove_field(const std::string& name) {
        if (fields_.erase(name) > 0) {
            NXS_LOG_DEBUG("Removed field '{}'", name);
        } else {
            NXS_LOG_WARN("Attempted to remove non-existent field '{}'", name);
        }
    }

    // Clear all non-essential fields
    void clear_fields() {
        fields_.clear();
        initialize_default_fields();
    }

    // ========================================================================
    // Standard Field Accessors (convenience)
    // ========================================================================

    // Nodal fields
    Field<Real>& displacement() { return field("displacement"); }
    const Field<Real>& displacement() const { return field("displacement"); }

    Field<Real>& velocity() { return field("velocity"); }
    const Field<Real>& velocity() const { return field("velocity"); }

    Field<Real>& acceleration() { return field("acceleration"); }
    const Field<Real>& acceleration() const { return field("acceleration"); }

    Field<Real>& force() { return field("force"); }
    const Field<Real>& force() const { return field("force"); }

    Field<Real>& mass() { return field("mass"); }
    const Field<Real>& mass() const { return field("mass"); }

    // Element fields
    bool has_stress() const { return has_field("stress"); }
    Field<Real>& stress() { return field("stress"); }
    const Field<Real>& stress() const { return field("stress"); }

    bool has_strain() const { return has_field("strain"); }
    Field<Real>& strain() { return field("strain"); }
    const Field<Real>& strain() const { return field("strain"); }

    // ========================================================================
    // State Operations
    // ========================================================================

    // Zero all fields
    void zero_all() {
        for (auto& [name, fld] : fields_) {
            fld.zero();
        }
        NXS_LOG_DEBUG("Zeroed all state fields");
    }

    // Copy from another state
    void copy_from(const State& other) {
        NXS_REQUIRE(mesh_ == other.mesh_, "States must share the same mesh");

        time_ = other.time_;
        step_ = other.step_;

        // Copy fields
        for (const auto& [name, other_field] : other.fields_) {
            if (has_field(name)) {
                field(name).copy_from(other_field);
            } else {
                // Create new field if it doesn't exist
                Field<Real> new_field(
                    other_field.name(),
                    other_field.type(),
                    other_field.location(),
                    other_field.num_entities(),
                    other_field.num_components()
                );
                new_field.copy_from(other_field);
                add_field(name, std::move(new_field));
            }
        }
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    void print_info() const {
        NXS_LOG_INFO("State Information:");
        NXS_LOG_INFO("  Time: {:.6e}", time_);
        NXS_LOG_INFO("  Step: {}", step_);
        NXS_LOG_INFO("  Number of fields: {}", fields_.size());

        for (const auto& [name, fld] : fields_) {
            NXS_LOG_INFO("    '{}': {} {} at {} ({} entities, {} components)",
                        name,
                        to_string(fld.type()),
                        "field",
                        to_string(fld.location()),
                        fld.num_entities(),
                        fld.num_components());
        }
    }

    // Compute total energy (if velocity and mass are available)
    Real compute_kinetic_energy() const {
        if (!has_field("velocity") || !has_field("mass")) {
            NXS_LOG_WARN("Cannot compute kinetic energy: missing velocity or mass field");
            return 0.0;
        }

        const auto& vel = velocity();
        const auto& m = mass();

        Real ke = 0.0;
        const std::size_t n = vel.num_entities();

        for (std::size_t i = 0; i < n; ++i) {
            Real vx = vel.at(i, 0);
            Real vy = vel.at(i, 1);
            Real vz = vel.at(i, 2);
            Real v_mag_sq = vx*vx + vy*vy + vz*vz;
            ke += 0.5 * m[i] * v_mag_sq;
        }

        return ke;
    }

    // ========================================================================
    // Mesh Access
    // ========================================================================

    const Mesh& mesh() const { return *mesh_; }

private:
    void initialize_default_fields() {
        const std::size_t num_nodes = mesh_->num_nodes();

        // Standard nodal fields for dynamic analysis
        add_field("displacement",
                 make_vector_field("displacement", FieldLocation::Node, num_nodes, 3));

        add_field("velocity",
                 make_vector_field("velocity", FieldLocation::Node, num_nodes, 3));

        add_field("acceleration",
                 make_vector_field("acceleration", FieldLocation::Node, num_nodes, 3));

        add_field("force",
                 make_vector_field("force", FieldLocation::Node, num_nodes, 3));

        add_field("mass",
                 make_scalar_field("mass", FieldLocation::Node, num_nodes));

        // Initialize all to zero
        zero_all();

        NXS_LOG_DEBUG("Initialized default state fields for {} nodes", num_nodes);
    }

    const Mesh* mesh_;  // Non-owning pointer to mesh
    Real time_;
    Int64 step_;
    std::map<std::string, Field<Real>> fields_;
};

// ============================================================================
// Multi-State Manager (for multi-step time integration)
// ============================================================================

/**
 * @brief StateHistory manages multiple time levels for multi-step integrators
 */
class StateHistory {
public:
    explicit StateHistory(const Mesh& mesh, std::size_t num_levels = 2)
        : mesh_(&mesh)
    {
        states_.reserve(num_levels);
        for (std::size_t i = 0; i < num_levels; ++i) {
            states_.emplace_back(mesh);
        }
        current_level_ = 0;
    }

    // Get current state
    State& current() { return states_[current_level_]; }
    const State& current() const { return states_[current_level_]; }

    // Get previous state (n steps back)
    State& previous(std::size_t n = 1) {
        std::size_t idx = (current_level_ + states_.size() - n) % states_.size();
        return states_[idx];
    }

    const State& previous(std::size_t n = 1) const {
        std::size_t idx = (current_level_ + states_.size() - n) % states_.size();
        return states_[idx];
    }

    // Advance to next time level (rotate states)
    void advance() {
        current_level_ = (current_level_ + 1) % states_.size();
    }

    std::size_t num_levels() const { return states_.size(); }

private:
    const Mesh* mesh_;
    std::vector<State> states_;
    std::size_t current_level_;
};

} // namespace nxs
