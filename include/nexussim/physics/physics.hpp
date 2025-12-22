#pragma once

// Physics module components
#include <nexussim/physics/module.hpp>
#include <nexussim/physics/element.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/physics/time_integrator.hpp>

/**
 * @file physics.hpp
 * @brief Main physics module header
 *
 * This header includes all physics-related components:
 * - PhysicsModule: Base class for all physics solvers
 * - Element: Finite element interface and implementations
 * - Material: Constitutive models for materials
 * - TimeIntegrator: Time integration schemes
 */

namespace nxs {
namespace physics {

/**
 * @brief Physics module version information
 */
namespace version {
    constexpr int major = 0;
    constexpr int minor = 1;
    constexpr int patch = 0;
} // namespace version

} // namespace physics
} // namespace nxs
