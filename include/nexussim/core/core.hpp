#pragma once

// Core infrastructure headers
#include <nexussim/core/types.hpp>
#include <nexussim/core/exception.hpp>
#include <nexussim/core/logger.hpp>
#include <nexussim/core/memory.hpp>
#include <nexussim/core/gpu.hpp>
#include <nexussim/core/mpi.hpp>

namespace nxs {

// ============================================================================
// Version Information
// ============================================================================

namespace version {

inline constexpr int major = 0;
inline constexpr int minor = 1;
inline constexpr int patch = 0;

inline constexpr const char* string = "0.1.0";
inline constexpr const char* build_date = __DATE__;
inline constexpr const char* build_time = __TIME__;

} // namespace version

// ============================================================================
// Initialization and Finalization
// ============================================================================

struct InitOptions {
    Logger::Level log_level = Logger::Level::Info;
    bool log_to_console = true;
    bool log_to_file = false;
    std::string log_file = "nexussim.log";
    bool enable_mpi = false;
    bool enable_gpu = false;
    int num_threads = -1;  // -1 means auto-detect
    int gpu_device_id = 0;  // GPU device ID to use
};

// Initialize NexusSim library
inline void initialize(int* argc = nullptr, char*** argv = nullptr,
                       const InitOptions& options = InitOptions{}) {
    // Initialize MPI first (before logger if using MPI)
    if (options.enable_mpi) {
        MPIManager::instance().initialize(argc, argv);
    }

    // Initialize logger (only on rank 0 for file logging if MPI enabled)
    bool should_log = !options.enable_mpi || MPIManager::instance().is_root();

    if (should_log) {
        if (options.log_to_console && options.log_to_file) {
            Logger::instance().init_combined(options.log_file, options.log_level);
        } else if (options.log_to_file) {
            Logger::instance().init_file(options.log_file, options.log_level);
        } else {
            Logger::instance().init_console(options.log_level);
        }
    } else {
        Logger::instance().init_console(options.log_level);
    }

    if (!options.enable_mpi || MPIManager::instance().is_root()) {
        NXS_LOG_INFO("=================================================");
        NXS_LOG_INFO("NexusSim v{} initializing...", version::string);
        NXS_LOG_INFO("Build date: {} {}", version::build_date, version::build_time);
        NXS_LOG_INFO("=================================================");
    }

    // Initialize Kokkos/GPU if requested
    if (options.enable_gpu) {
        KokkosConfig kconfig;
        kconfig.num_threads = options.num_threads;
        kconfig.device_id = options.gpu_device_id;
        KokkosManager::instance().initialize(kconfig);
    }

    if (!options.enable_mpi || MPIManager::instance().is_root()) {
        NXS_LOG_INFO("Initialization complete");
    }
}

// Finalize NexusSim library
inline void finalize() {
    if (!MPIManager::instance().is_initialized() || MPIManager::instance().is_root()) {
        NXS_LOG_INFO("NexusSim shutting down...");
    }

    // Finalize in reverse order
    KokkosManager::instance().finalize();
    MPIManager::instance().finalize();

    Logger::instance().flush();
}

// ============================================================================
// RAII Wrapper for Initialization/Finalization
// ============================================================================

class Context {
public:
    explicit Context(int* argc = nullptr, char*** argv = nullptr,
                     const InitOptions& options = InitOptions{}) {
        initialize(argc, argv, options);
    }

    explicit Context(const InitOptions& options)
        : Context(nullptr, nullptr, options) {}

    ~Context() {
        finalize();
    }

    // Delete copy/move operations
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;
};

} // namespace nxs
