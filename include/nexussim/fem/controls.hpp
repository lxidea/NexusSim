#pragma once

/**
 * @file controls.hpp
 * @brief Sensor-triggered simulation controls
 *
 * Actions triggered by sensor threshold events:
 * - Terminate simulation
 * - Activate/deactivate loads
 * - Activate/deactivate contacts
 * - Write checkpoint snapshot
 * - Change time step
 *
 * Reference: LS-DYNA *SENSOR_CONTROL, *SENSOR_SWITCH
 */

#include <nexussim/core/types.hpp>
#include <nexussim/fem/sensor.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <functional>

namespace nxs {
namespace fem {

// ============================================================================
// Control Action Types
// ============================================================================

enum class ControlActionType {
    TerminateSimulation,    ///< End simulation
    ActivateLoad,           ///< Turn on a load by ID
    DeactivateLoad,         ///< Turn off a load by ID
    ActivateContact,        ///< Turn on a contact by ID
    DeactivateContact,      ///< Turn off a contact by ID
    WriteCheckpoint,        ///< Write a checkpoint file
    SetTimestep,            ///< Change time step limit
    CustomCallback          ///< User-defined callback function
};

// ============================================================================
// Control Rule
// ============================================================================

struct ControlRule {
    int id;
    std::string name;

    // Trigger condition
    int sensor_id;              ///< Sensor that triggers this rule
    bool trigger_on_exceed;     ///< true=trigger when sensor exceeds threshold

    // Action
    ControlActionType action;
    int target_id;              ///< ID of load/contact to activate/deactivate
    Real value;                 ///< Value for SetTimestep
    std::function<void()> callback; ///< Custom callback

    // State
    bool triggered;
    bool one_shot;              ///< true=trigger only once

    ControlRule()
        : id(0), sensor_id(0), trigger_on_exceed(true)
        , action(ControlActionType::TerminateSimulation)
        , target_id(0), value(0.0)
        , triggered(false), one_shot(true) {}
};

// ============================================================================
// Control Manager
// ============================================================================

class ControlManager {
public:
    ControlManager() = default;

    /**
     * @brief Add a control rule
     */
    ControlRule& add_rule(int id) {
        rules_.emplace_back();
        rules_.back().id = id;
        return rules_.back();
    }

    /**
     * @brief Evaluate all control rules against sensor states
     * @param sensors The sensor manager with current readings
     * @return List of actions that were triggered this step
     */
    struct TriggeredAction {
        int rule_id;
        ControlActionType action;
        int target_id;
        Real value;
    };

    std::vector<TriggeredAction> evaluate(const SensorManager& sensors) {
        std::vector<TriggeredAction> actions;

        for (auto& rule : rules_) {
            if (rule.one_shot && rule.triggered) continue;

            const Sensor* sensor = sensors.find(rule.sensor_id);
            if (!sensor) continue;

            bool should_trigger = false;
            if (rule.trigger_on_exceed) {
                should_trigger = sensor->threshold_triggered();
            } else {
                // Trigger when NOT exceeding (e.g., distance drops below threshold)
                should_trigger = !sensor->threshold_triggered() &&
                                  sensor->num_readings() > 0;
            }

            if (should_trigger && !rule.triggered) {
                rule.triggered = true;

                TriggeredAction ta;
                ta.rule_id = rule.id;
                ta.action = rule.action;
                ta.target_id = rule.target_id;
                ta.value = rule.value;
                actions.push_back(ta);

                // Execute callback if custom
                if (rule.action == ControlActionType::CustomCallback && rule.callback) {
                    rule.callback();
                }
            }
        }

        return actions;
    }

    /**
     * @brief Check if any terminate action has been triggered
     */
    bool should_terminate() const {
        for (const auto& rule : rules_) {
            if (rule.triggered && rule.action == ControlActionType::TerminateSimulation) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Get all active load activations
     * @return Vector of load IDs that should be activated
     */
    std::vector<int> active_loads() const {
        std::vector<int> ids;
        for (const auto& rule : rules_) {
            if (rule.triggered && rule.action == ControlActionType::ActivateLoad) {
                ids.push_back(rule.target_id);
            }
        }
        return ids;
    }

    /**
     * @brief Get deactivated load IDs
     */
    std::vector<int> deactivated_loads() const {
        std::vector<int> ids;
        for (const auto& rule : rules_) {
            if (rule.triggered && rule.action == ControlActionType::DeactivateLoad) {
                ids.push_back(rule.target_id);
            }
        }
        return ids;
    }

    std::size_t num_rules() const { return rules_.size(); }

    void reset_all() {
        for (auto& rule : rules_) rule.triggered = false;
    }

    void print_summary() const {
        std::cout << "Controls: " << rules_.size() << " rules\n";
        for (const auto& r : rules_) {
            std::cout << "  Rule [" << r.id << "] " << r.name
                      << " → sensor " << r.sensor_id
                      << (r.triggered ? " [TRIGGERED]" : "") << "\n";
        }
    }

private:
    std::vector<ControlRule> rules_;
};

} // namespace fem
} // namespace nxs
