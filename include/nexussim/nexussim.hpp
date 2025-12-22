#pragma once

/**
 * @file nexussim.hpp
 * @brief Main header for NexusSim library
 *
 * Include this single header to get access to all NexusSim functionality.
 */

// Core infrastructure
#include <nexussim/core/core.hpp>

// Data structures
#include <nexussim/data/data.hpp>

// Physics modules
#include <nexussim/physics/physics.hpp>

// Coupling framework
#include <nexussim/coupling/coupling.hpp>

// Version and namespace
namespace nxs {

// Convenience function to get version string
inline const char* version_string() {
    return version::string;
}

// Check if feature is enabled at compile time
namespace features {

#ifdef NEXUSSIM_ENABLE_MPI
    inline constexpr bool mpi_enabled = true;
#else
    inline constexpr bool mpi_enabled = false;
#endif

#ifdef NEXUSSIM_ENABLE_GPU
    inline constexpr bool gpu_enabled = true;
#else
    inline constexpr bool gpu_enabled = false;
#endif

#ifdef NEXUSSIM_ENABLE_OPENMP
    inline constexpr bool openmp_enabled = true;
#else
    inline constexpr bool openmp_enabled = false;
#endif

inline void print_features() {
    NXS_LOG_INFO("NexusSim Features:");
    NXS_LOG_INFO("  MPI: {}", mpi_enabled ? "enabled" : "disabled");
    NXS_LOG_INFO("  GPU (Kokkos): {}", gpu_enabled ? "enabled" : "disabled");
    NXS_LOG_INFO("  OpenMP: {}", openmp_enabled ? "enabled" : "disabled");
    NXS_LOG_INFO("  Precision: {} bytes", sizeof(Real));
}

} // namespace features

} // namespace nxs
