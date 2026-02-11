#pragma once

/**
 * @file loads.hpp
 * @brief Load manager for time-varying forces, pressures, and imposed conditions
 *
 * Supports:
 * - Nodal forces with optional load curves
 * - Pressure loads on element faces
 * - Gravity
 * - Imposed velocity/displacement
 * - Initial velocity
 *
 * Reference: OpenRadioss /engine/source/loads
 */

#include <nexussim/core/types.hpp>
#include <nexussim/fem/load_curve.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

namespace nxs {
namespace fem {

// ============================================================================
// Load Types
// ============================================================================

enum class LoadType {
    NodalForce,            ///< Force applied at node(s)
    Pressure,              ///< Pressure on element face
    Gravity,               ///< Body force (gravity)
    ImposedDisplacement,   ///< Prescribed displacement
    ImposedVelocity,       ///< Prescribed velocity
    InitialVelocity,       ///< Initial velocity (applied once)
    Moment                 ///< Moment/torque at node
};

// ============================================================================
// Load Definition
// ============================================================================

struct Load {
    LoadType type;
    int id;
    std::string name;

    // Target
    std::vector<Index> node_set;   ///< Target nodes
    int dof;                        ///< DOF index (0=x, 1=y, 2=z, -1=all)

    // Value
    Real magnitude;                 ///< Base magnitude
    Real direction[3];              ///< Force/velocity direction
    int load_curve_id;              ///< Load curve ID (-1 = constant)

    // State
    bool active;
    bool applied;                   ///< For initial conditions: already applied?

    Load()
        : type(LoadType::NodalForce), id(0), dof(-1)
        , magnitude(1.0), load_curve_id(-1), active(true), applied(false) {
        direction[0] = 0.0; direction[1] = 0.0; direction[2] = 1.0;
    }
};

// ============================================================================
// Load Manager
// ============================================================================

class LoadManager {
public:
    LoadManager() = default;

    // --- Configuration ---

    Load& add_load(LoadType type, int id = 0) {
        loads_.emplace_back();
        auto& l = loads_.back();
        l.type = type;
        l.id = id;
        return l;
    }

    void set_curve_manager(LoadCurveManager* mgr) { curves_ = mgr; }

    std::size_t num_loads() const { return loads_.size(); }
    Load& load(std::size_t i) { return loads_[i]; }

    // --- Application ---

    /**
     * @brief Apply all loads at current time
     * @param time Current simulation time
     * @param num_nodes Number of nodes
     * @param positions Node positions [3*num_nodes]
     * @param velocities Node velocities [3*num_nodes] (may be modified for imposed)
     * @param forces External force vector [3*num_nodes] (accumulated)
     * @param masses Node masses [num_nodes]
     * @param dt Time step
     */
    void apply_loads(Real time, std::size_t num_nodes,
                     const Real* positions, Real* velocities,
                     Real* forces, const Real* masses, Real dt) {
        for (auto& load : loads_) {
            if (!load.active) continue;

            switch (load.type) {
                case LoadType::NodalForce:
                    apply_nodal_force(load, time, forces);
                    break;
                case LoadType::Gravity:
                    apply_gravity(load, time, num_nodes, forces, masses);
                    break;
                case LoadType::ImposedVelocity:
                    apply_imposed_velocity(load, time, velocities, dt);
                    break;
                case LoadType::ImposedDisplacement:
                    apply_imposed_displacement(load, time, positions, velocities, dt);
                    break;
                case LoadType::InitialVelocity:
                    apply_initial_velocity(load, velocities);
                    break;
                case LoadType::Pressure:
                    apply_pressure(load, time, positions, forces);
                    break;
                case LoadType::Moment:
                    apply_moment(load, time, forces);
                    break;
            }
        }
    }

    /**
     * @brief Apply initial conditions (called once at start)
     */
    void apply_initial_conditions(Real* velocities, Real* /*displacements*/) {
        for (auto& load : loads_) {
            if (load.type == LoadType::InitialVelocity && !load.applied) {
                apply_initial_velocity(load, velocities);
            }
        }
    }

    void print_summary() const {
        std::cout << "Load Manager: " << loads_.size() << " loads\n";
        for (const auto& l : loads_) {
            const char* type_str = "Unknown";
            switch (l.type) {
                case LoadType::NodalForce: type_str = "NodalForce"; break;
                case LoadType::Pressure: type_str = "Pressure"; break;
                case LoadType::Gravity: type_str = "Gravity"; break;
                case LoadType::ImposedVelocity: type_str = "ImposedVelocity"; break;
                case LoadType::ImposedDisplacement: type_str = "ImposedDisplacement"; break;
                case LoadType::InitialVelocity: type_str = "InitialVelocity"; break;
                case LoadType::Moment: type_str = "Moment"; break;
            }
            std::cout << "  [" << l.id << "] " << type_str
                      << " mag=" << l.magnitude
                      << " nodes=" << l.node_set.size()
                      << " curve=" << l.load_curve_id << "\n";
        }
    }

private:
    Real get_curve_value(int curve_id, Real time) const {
        if (curves_ && curve_id >= 0) {
            return curves_->evaluate(curve_id, time);
        }
        return 1.0;  // No curve = constant 1.0
    }

    void apply_nodal_force(const Load& load, Real time, Real* forces) {
        Real scale = load.magnitude * get_curve_value(load.load_curve_id, time);

        for (Index n : load.node_set) {
            if (load.dof >= 0 && load.dof < 3) {
                forces[3*n + load.dof] += scale * load.direction[load.dof];
            } else {
                // All DOFs
                forces[3*n + 0] += scale * load.direction[0];
                forces[3*n + 1] += scale * load.direction[1];
                forces[3*n + 2] += scale * load.direction[2];
            }
        }
    }

    void apply_gravity(const Load& load, Real time, std::size_t num_nodes,
                       Real* forces, const Real* masses) {
        Real scale = load.magnitude * get_curve_value(load.load_curve_id, time);

        if (load.node_set.empty()) {
            // Apply to all nodes
            for (std::size_t i = 0; i < num_nodes; ++i) {
                forces[3*i + 0] += masses[i] * scale * load.direction[0];
                forces[3*i + 1] += masses[i] * scale * load.direction[1];
                forces[3*i + 2] += masses[i] * scale * load.direction[2];
            }
        } else {
            // Apply to node set only
            for (Index n : load.node_set) {
                forces[3*n + 0] += masses[n] * scale * load.direction[0];
                forces[3*n + 1] += masses[n] * scale * load.direction[1];
                forces[3*n + 2] += masses[n] * scale * load.direction[2];
            }
        }
    }

    void apply_imposed_velocity(const Load& load, Real time,
                                 Real* velocities, Real /*dt*/) {
        Real v_target = load.magnitude * get_curve_value(load.load_curve_id, time);

        for (Index n : load.node_set) {
            if (load.dof >= 0 && load.dof < 3) {
                velocities[3*n + load.dof] = v_target;
            } else {
                velocities[3*n + 0] = v_target * load.direction[0];
                velocities[3*n + 1] = v_target * load.direction[1];
                velocities[3*n + 2] = v_target * load.direction[2];
            }
        }
    }

    void apply_imposed_displacement(const Load& load, Real time,
                                     const Real* /*positions*/,
                                     Real* velocities, Real dt) {
        Real d_target = load.magnitude * get_curve_value(load.load_curve_id, time);
        // Approximate: velocity to reach target displacement
        Real v = d_target / (dt + 1.0e-30);

        for (Index n : load.node_set) {
            if (load.dof >= 0 && load.dof < 3) {
                velocities[3*n + load.dof] = v;
            }
        }
    }

    void apply_initial_velocity(Load& load, Real* velocities) {
        if (load.applied) return;

        for (Index n : load.node_set) {
            velocities[3*n + 0] += load.magnitude * load.direction[0];
            velocities[3*n + 1] += load.magnitude * load.direction[1];
            velocities[3*n + 2] += load.magnitude * load.direction[2];
        }

        load.applied = true;
    }

    void apply_pressure(const Load& load, Real time,
                        const Real* /*positions*/, Real* forces) {
        // Simplified: apply pressure as force on nodes
        // Full implementation would compute face normals from element connectivity
        Real p = load.magnitude * get_curve_value(load.load_curve_id, time);

        for (Index n : load.node_set) {
            forces[3*n + 0] += p * load.direction[0];
            forces[3*n + 1] += p * load.direction[1];
            forces[3*n + 2] += p * load.direction[2];
        }
    }

    void apply_moment(const Load& load, Real time, Real* forces) {
        // Simplified: apply moment as force couple
        Real m = load.magnitude * get_curve_value(load.load_curve_id, time);
        (void)m; (void)forces;
        // Full implementation requires rotational DOFs
    }

    std::vector<Load> loads_;
    LoadCurveManager* curves_ = nullptr;
};

} // namespace fem
} // namespace nxs
