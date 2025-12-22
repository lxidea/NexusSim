#pragma once

#include <nexussim/core/types.hpp>
#include <nexussim/core/exception.hpp>
#include <nexussim/core/logger.hpp>

#ifdef NEXUSSIM_HAVE_KOKKOS
#include <Kokkos_Core.hpp>
#endif

namespace nxs {

// ============================================================================
// Execution Space and Device Management
// ============================================================================

enum class ExecutionDevice {
    CPU,
    GPU,
    Default
};

#ifdef NEXUSSIM_HAVE_KOKKOS

// Kokkos execution space aliases
using DefaultExecSpace = Kokkos::DefaultExecutionSpace;
using DefaultMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using HostMemSpace = Kokkos::HostSpace;

// Determine if default execution space is GPU
inline constexpr bool is_gpu_default() {
#if defined(KOKKOS_ENABLE_CUDA)
    return std::is_same_v<DefaultExecSpace, Kokkos::Cuda>;
#elif defined(KOKKOS_ENABLE_HIP)
    return std::is_same_v<DefaultExecSpace, Kokkos::HIP>;
#elif defined(KOKKOS_ENABLE_SYCL)
    return std::is_same_v<DefaultExecSpace, Kokkos::Experimental::SYCL>;
#else
    return false;
#endif
}

// ============================================================================
// Kokkos View Types (GPU-compatible arrays)
// ============================================================================

// 1D views
template<typename T>
using View1D = Kokkos::View<T*, DefaultMemSpace>;

template<typename T>
using HostView1D = Kokkos::View<T*, HostMemSpace>;

// 2D views
template<typename T>
using View2D = Kokkos::View<T**, Kokkos::LayoutLeft, DefaultMemSpace>;

template<typename T>
using HostView2D = Kokkos::View<T**, Kokkos::LayoutLeft, HostMemSpace>;

// 3D views
template<typename T>
using View3D = Kokkos::View<T***, Kokkos::LayoutLeft, DefaultMemSpace>;

template<typename T>
using HostView3D = Kokkos::View<T***, Kokkos::LayoutLeft, HostMemSpace>;

// ============================================================================
// Kokkos Initialization and Finalization
// ============================================================================

struct KokkosConfig {
    int num_threads = -1;           // -1 = auto
    int device_id = 0;              // GPU device ID
    bool disable_warnings = false;
    bool tune_internals = true;
};

class KokkosManager {
public:
    static KokkosManager& instance() {
        static KokkosManager manager;
        return manager;
    }

    void initialize(const KokkosConfig& config = KokkosConfig{}) {
        if (is_initialized_) {
            NXS_LOG_WARN("Kokkos already initialized");
            return;
        }

        Kokkos::InitializationSettings args;

        if (config.num_threads > 0) {
            args.set_num_threads(config.num_threads);
        }

        if (config.device_id >= 0) {
            args.set_device_id(config.device_id);
        }

        args.set_disable_warnings(config.disable_warnings);
        args.set_tune_internals(config.tune_internals);

        Kokkos::initialize(args);
        is_initialized_ = true;

        print_configuration();
    }

    void finalize() {
        if (!is_initialized_) {
            return;
        }

        Kokkos::finalize();
        is_initialized_ = false;
        NXS_LOG_INFO("Kokkos finalized");
    }

    bool is_initialized() const { return is_initialized_; }

    void print_configuration() const {
        NXS_LOG_INFO("Kokkos Configuration:");
        NXS_LOG_INFO("  Version: {}.{}.{}",
                    KOKKOS_VERSION / 10000,
                    (KOKKOS_VERSION / 100) % 100,
                    KOKKOS_VERSION % 100);

        NXS_LOG_INFO("  Default Execution Space: {}",
                    typeid(DefaultExecSpace).name());
        NXS_LOG_INFO("  Default Memory Space: {}",
                    typeid(DefaultMemSpace).name());

#if defined(KOKKOS_ENABLE_CUDA)
        NXS_LOG_INFO("  CUDA: enabled");
#endif
#if defined(KOKKOS_ENABLE_HIP)
        NXS_LOG_INFO("  HIP: enabled");
#endif
#if defined(KOKKOS_ENABLE_SYCL)
        NXS_LOG_INFO("  SYCL: enabled");
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
        NXS_LOG_INFO("  OpenMP: enabled");
#endif
#if defined(KOKKOS_ENABLE_THREADS)
        NXS_LOG_INFO("  Threads: enabled");
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
        NXS_LOG_INFO("  Serial: enabled");
#endif

        // Print device info
        DefaultExecSpace exec;
        NXS_LOG_INFO("  Concurrency: {}", exec.concurrency());

#if defined(KOKKOS_ENABLE_CUDA)
        Kokkos::Cuda cuda;
        NXS_LOG_INFO("  CUDA Device: {}", cuda.cuda_device());
#endif
    }

private:
    KokkosManager() = default;
    ~KokkosManager() {
        if (is_initialized_) {
            finalize();
        }
    }

    // Delete copy/move
    KokkosManager(const KokkosManager&) = delete;
    KokkosManager& operator=(const KokkosManager&) = delete;

    bool is_initialized_ = false;
};

// ============================================================================
// Parallel Execution Utilities
// ============================================================================

// Parallel for wrapper
template<typename Functor>
inline void parallel_for(const std::string& label, std::size_t n, const Functor& f) {
    Kokkos::parallel_for(label, n, f);
}

template<typename Functor>
inline void parallel_for(std::size_t n, const Functor& f) {
    Kokkos::parallel_for(n, f);
}

// Parallel reduce wrapper
template<typename Functor, typename ReduceType>
inline void parallel_reduce(const std::string& label, std::size_t n,
                            const Functor& f, ReduceType& result) {
    Kokkos::parallel_reduce(label, n, f, result);
}

template<typename Functor, typename ReduceType>
inline void parallel_reduce(std::size_t n, const Functor& f, ReduceType& result) {
    Kokkos::parallel_reduce(n, f, result);
}

// Fence - ensure all kernels complete
inline void fence(const std::string& label = "NexusSim::fence") {
    Kokkos::fence(label);
}

// ============================================================================
// Deep Copy Helper (Host <-> Device)
// ============================================================================

template<typename DstView, typename SrcView>
inline void deep_copy(DstView& dst, const SrcView& src) {
    Kokkos::deep_copy(dst, src);
}

// ============================================================================
// Memory Management
// ============================================================================

// Get memory space name
inline std::string memory_space_name() {
    return DefaultMemSpace::name();
}

// Check if execution space is GPU
inline bool is_gpu_space() {
    return is_gpu_default();
}

#else // !NEXUSSIM_HAVE_KOKKOS

// Dummy types when Kokkos is not available
struct KokkosConfig {};

class KokkosManager {
public:
    static KokkosManager& instance() {
        static KokkosManager manager;
        return manager;
    }

    void initialize(const KokkosConfig& = KokkosConfig{}) {
        NXS_LOG_WARN("Kokkos support not enabled - GPU features unavailable");
    }

    void finalize() {}
    bool is_initialized() const { return false; }
    void print_configuration() const {
        NXS_LOG_INFO("Kokkos: disabled");
    }
};

inline void fence(const std::string& = "") {}

inline bool is_gpu_space() { return false; }

inline std::string memory_space_name() { return "Host"; }

#endif // NEXUSSIM_HAVE_KOKKOS

} // namespace nxs
