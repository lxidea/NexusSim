#pragma once

/**
 * @file initial_conditions.hpp
 * @brief Initial conditions for FEM simulation
 *
 * Applied once at simulation start:
 * - Initial velocity (by node set or all nodes)
 * - Initial displacement
 * - Initial temperature
 */

#include <nexussim/core/types.hpp>
#include <vector>
#include <string>
#include <iostream>

namespace nxs {
namespace fem {

// ============================================================================
// Initial Condition Types
// ============================================================================

enum class InitialConditionType {
    Velocity,        ///< Initial velocity
    Displacement,    ///< Initial displacement
    Temperature      ///< Initial temperature
};

// ============================================================================
// Initial Condition
// ============================================================================

struct InitialCondition {
    InitialConditionType type;
    int id;
    std::string name;

    std::vector<Index> node_set;   ///< Target nodes (empty = all)
    Real value[3];                  ///< Value for each DOF
    Real scalar_value;              ///< Scalar value (temperature)
    bool applied;

    InitialCondition()
        : type(InitialConditionType::Velocity), id(0)
        , scalar_value(0.0), applied(false) {
        value[0] = value[1] = value[2] = 0.0;
    }
};

// ============================================================================
// Initial Condition Manager
// ============================================================================

class InitialConditionManager {
public:
    InitialConditionManager() = default;

    InitialCondition& add_condition(InitialConditionType type, int id = 0) {
        conditions_.emplace_back();
        auto& ic = conditions_.back();
        ic.type = type;
        ic.id = id;
        return ic;
    }

    std::size_t num_conditions() const { return conditions_.size(); }

    /**
     * @brief Apply all initial conditions
     * @param num_nodes Number of nodes
     * @param velocities Velocity array [3*num_nodes]
     * @param displacements Displacement array [3*num_nodes] (can be null)
     * @param temperatures Temperature array [num_nodes] (can be null)
     */
    void apply(std::size_t num_nodes,
               Real* velocities,
               Real* displacements = nullptr,
               Real* temperatures = nullptr) {
        for (auto& ic : conditions_) {
            if (ic.applied) continue;

            switch (ic.type) {
                case InitialConditionType::Velocity:
                    apply_velocity(ic, num_nodes, velocities);
                    break;
                case InitialConditionType::Displacement:
                    if (displacements) apply_displacement(ic, num_nodes, displacements);
                    break;
                case InitialConditionType::Temperature:
                    if (temperatures) apply_temperature(ic, num_nodes, temperatures);
                    break;
            }

            ic.applied = true;
        }
    }

    void print_summary() const {
        std::cout << "Initial Conditions: " << conditions_.size() << "\n";
        for (const auto& ic : conditions_) {
            const char* type_str = "Unknown";
            switch (ic.type) {
                case InitialConditionType::Velocity: type_str = "Velocity"; break;
                case InitialConditionType::Displacement: type_str = "Displacement"; break;
                case InitialConditionType::Temperature: type_str = "Temperature"; break;
            }
            std::cout << "  [" << ic.id << "] " << type_str;
            if (ic.node_set.empty()) std::cout << " (all nodes)";
            else std::cout << " (" << ic.node_set.size() << " nodes)";
            std::cout << " value=[" << ic.value[0] << "," << ic.value[1] << "," << ic.value[2] << "]\n";
        }
    }

private:
    void apply_velocity(const InitialCondition& ic, std::size_t num_nodes, Real* vel) {
        if (ic.node_set.empty()) {
            for (std::size_t i = 0; i < num_nodes; ++i) {
                vel[3*i+0] = ic.value[0];
                vel[3*i+1] = ic.value[1];
                vel[3*i+2] = ic.value[2];
            }
        } else {
            for (Index n : ic.node_set) {
                vel[3*n+0] = ic.value[0];
                vel[3*n+1] = ic.value[1];
                vel[3*n+2] = ic.value[2];
            }
        }
    }

    void apply_displacement(const InitialCondition& ic, std::size_t num_nodes, Real* disp) {
        if (ic.node_set.empty()) {
            for (std::size_t i = 0; i < num_nodes; ++i) {
                disp[3*i+0] = ic.value[0];
                disp[3*i+1] = ic.value[1];
                disp[3*i+2] = ic.value[2];
            }
        } else {
            for (Index n : ic.node_set) {
                disp[3*n+0] = ic.value[0];
                disp[3*n+1] = ic.value[1];
                disp[3*n+2] = ic.value[2];
            }
        }
    }

    void apply_temperature(const InitialCondition& ic, std::size_t num_nodes, Real* temp) {
        if (ic.node_set.empty()) {
            for (std::size_t i = 0; i < num_nodes; ++i) {
                temp[i] = ic.scalar_value;
            }
        } else {
            for (Index n : ic.node_set) {
                temp[n] = ic.scalar_value;
            }
        }
    }

    std::vector<InitialCondition> conditions_;
};

} // namespace fem
} // namespace nxs
