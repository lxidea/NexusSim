#pragma once

// Coupling framework components
#include <nexussim/coupling/field_registry.hpp>
#include <nexussim/coupling/coupling_operator.hpp>

/**
 * @file coupling.hpp
 * @brief Main coupling framework header
 *
 * This header includes all coupling-related components:
 * - FieldRegistry: Centralized field management for multi-physics
 * - CouplingOperator: Data transfer between physics modules
 * - CouplingInterface: Interface region definition
 */

namespace nxs {
namespace coupling {

/**
 * @brief Coupling framework version information
 */
namespace version {
    constexpr int major = 0;
    constexpr int minor = 1;
    constexpr int patch = 0;
} // namespace version

} // namespace coupling
} // namespace nxs
