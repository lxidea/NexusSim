#pragma once

/**
 * @file performance_timer.hpp
 * @brief Performance timing and profiling utilities for NexusSim
 *
 * Provides high-resolution timers, memory tracking, and performance
 * metrics collection for benchmarking CPU vs GPU performance.
 */

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

#ifdef NEXUSSIM_HAVE_KOKKOS
#include <Kokkos_Core.hpp>
#endif

namespace nxs {
namespace utils {

/**
 * @brief High-resolution timer for performance measurement
 */
class Timer {
public:
    using clock_t = std::chrono::high_resolution_clock;
    using time_point_t = std::chrono::time_point<clock_t>;

    Timer() : running_(false), accumulated_(0.0) {}

    void start() {
        if (!running_) {
            start_time_ = clock_t::now();
            running_ = true;
        }
    }

    void stop() {
        if (running_) {
            auto end_time = clock_t::now();
            std::chrono::duration<double, std::milli> elapsed = end_time - start_time_;
            accumulated_ += elapsed.count();
            running_ = false;
        }
    }

    void reset() {
        accumulated_ = 0.0;
        running_ = false;
    }

    double elapsed_ms() const {
        if (running_) {
            auto now = clock_t::now();
            std::chrono::duration<double, std::milli> elapsed = now - start_time_;
            return accumulated_ + elapsed.count();
        }
        return accumulated_;
    }

    double elapsed_sec() const {
        return elapsed_ms() / 1000.0;
    }

private:
    time_point_t start_time_;
    bool running_;
    double accumulated_;
};

/**
 * @brief RAII-style scoped timer
 */
class ScopedTimer {
public:
    ScopedTimer(const std::string& name, bool print_on_destruct = true)
        : name_(name), print_(print_on_destruct) {
        timer_.start();
    }

    ~ScopedTimer() {
        timer_.stop();
        if (print_) {
            std::cout << "[TIMER] " << name_ << ": "
                      << std::fixed << std::setprecision(3)
                      << timer_.elapsed_ms() << " ms\n";
        }
    }

    double elapsed_ms() const { return timer_.elapsed_ms(); }

private:
    Timer timer_;
    std::string name_;
    bool print_;
};

/**
 * @brief Performance statistics for multiple runs
 */
struct PerformanceStats {
    std::string name;
    int num_samples = 0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = std::numeric_limits<double>::lowest();
    double mean_ms = 0.0;
    double std_dev_ms = 0.0;
    double total_ms = 0.0;

    void add_sample(double time_ms) {
        samples_.push_back(time_ms);
        num_samples++;
        total_ms += time_ms;
        min_ms = std::min(min_ms, time_ms);
        max_ms = std::max(max_ms, time_ms);
        compute_stats();
    }

    void print() const {
        std::cout << std::left << std::setw(30) << name
                  << " | n=" << std::setw(5) << num_samples
                  << " | mean=" << std::fixed << std::setprecision(3) << std::setw(10) << mean_ms << " ms"
                  << " | std=" << std::setw(8) << std_dev_ms << " ms"
                  << " | min=" << std::setw(8) << min_ms << " ms"
                  << " | max=" << std::setw(8) << max_ms << " ms\n";
    }

private:
    std::vector<double> samples_;

    void compute_stats() {
        if (samples_.empty()) return;

        mean_ms = total_ms / samples_.size();

        if (samples_.size() > 1) {
            double variance = 0.0;
            for (double s : samples_) {
                variance += (s - mean_ms) * (s - mean_ms);
            }
            variance /= (samples_.size() - 1);
            std_dev_ms = std::sqrt(variance);
        }
    }
};

/**
 * @brief Global performance profiler with named timers
 */
class Profiler {
public:
    static Profiler& instance() {
        static Profiler profiler;
        return profiler;
    }

    void start(const std::string& name) {
        timers_[name].start();
    }

    void stop(const std::string& name) {
        timers_[name].stop();
        if (stats_.find(name) == stats_.end()) {
            stats_[name].name = name;
        }
        stats_[name].add_sample(timers_[name].elapsed_ms());
        timers_[name].reset();
    }

    double elapsed(const std::string& name) const {
        auto it = timers_.find(name);
        if (it != timers_.end()) {
            return it->second.elapsed_ms();
        }
        return 0.0;
    }

    const PerformanceStats& stats(const std::string& name) const {
        return stats_.at(name);
    }

    void print_all() const {
        std::cout << "\n" << std::string(100, '=') << "\n";
        std::cout << "PERFORMANCE PROFILE\n";
        std::cout << std::string(100, '=') << "\n";

        std::vector<std::string> names;
        for (const auto& kv : stats_) {
            names.push_back(kv.first);
        }
        std::sort(names.begin(), names.end());

        for (const auto& name : names) {
            stats_.at(name).print();
        }
        std::cout << std::string(100, '=') << "\n\n";
    }

    void reset() {
        timers_.clear();
        stats_.clear();
    }

private:
    Profiler() = default;
    std::unordered_map<std::string, Timer> timers_;
    std::unordered_map<std::string, PerformanceStats> stats_;
};

/**
 * @brief Memory usage tracking
 */
struct MemoryStats {
    size_t host_bytes = 0;
    size_t device_bytes = 0;

    double host_mb() const { return host_bytes / (1024.0 * 1024.0); }
    double device_mb() const { return device_bytes / (1024.0 * 1024.0); }
    double total_mb() const { return (host_bytes + device_bytes) / (1024.0 * 1024.0); }

    void print() const {
        std::cout << "Memory Usage:\n";
        std::cout << "  Host:   " << std::fixed << std::setprecision(2) << host_mb() << " MB\n";
        std::cout << "  Device: " << device_mb() << " MB\n";
        std::cout << "  Total:  " << total_mb() << " MB\n";
    }
};

/**
 * @brief Benchmark result structure
 */
struct BenchmarkResult {
    std::string name;
    int num_elements = 0;
    int num_nodes = 0;
    int num_dofs = 0;
    int num_steps = 0;
    double total_time_sec = 0.0;
    double time_per_step_ms = 0.0;
    double dofs_per_sec = 0.0;
    double elements_per_sec = 0.0;
    MemoryStats memory;
    std::string backend;  // "CPU", "OpenMP", "CUDA", etc.

    void compute_derived() {
        if (num_steps > 0 && total_time_sec > 0) {
            time_per_step_ms = (total_time_sec / num_steps) * 1000.0;
            dofs_per_sec = num_dofs * num_steps / total_time_sec;
            elements_per_sec = num_elements * num_steps / total_time_sec;
        }
    }

    void print() const {
        std::cout << std::left
                  << std::setw(20) << name
                  << std::setw(12) << num_elements
                  << std::setw(12) << num_nodes
                  << std::setw(12) << num_dofs
                  << std::setw(12) << num_steps
                  << std::setw(12) << std::fixed << std::setprecision(3) << time_per_step_ms
                  << std::setw(15) << std::scientific << std::setprecision(2) << dofs_per_sec
                  << std::setw(10) << backend << "\n";
    }

    static void print_header() {
        std::cout << std::left
                  << std::setw(20) << "Name"
                  << std::setw(12) << "Elements"
                  << std::setw(12) << "Nodes"
                  << std::setw(12) << "DOFs"
                  << std::setw(12) << "Steps"
                  << std::setw(12) << "ms/Step"
                  << std::setw(15) << "DOFs/sec"
                  << std::setw(10) << "Backend" << "\n";
        std::cout << std::string(103, '-') << "\n";
    }
};

/**
 * @brief Compute speedup between two benchmark results
 */
inline double compute_speedup(const BenchmarkResult& baseline, const BenchmarkResult& optimized) {
    if (optimized.time_per_step_ms > 0) {
        return baseline.time_per_step_ms / optimized.time_per_step_ms;
    }
    return 0.0;
}

/**
 * @brief Get current Kokkos execution space name
 */
inline std::string get_execution_space_name() {
#ifdef NEXUSSIM_HAVE_KOKKOS
    return Kokkos::DefaultExecutionSpace::name();
#else
    return "Serial (no Kokkos)";
#endif
}

/**
 * @brief Check if GPU is available
 */
inline bool gpu_available() {
#ifdef NEXUSSIM_HAVE_KOKKOS
    std::string space = get_execution_space_name();
    return (space.find("Cuda") != std::string::npos ||
            space.find("HIP") != std::string::npos ||
            space.find("SYCL") != std::string::npos);
#else
    return false;
#endif
}

/**
 * @brief Fence to ensure all GPU work is complete
 */
inline void device_fence() {
#ifdef NEXUSSIM_HAVE_KOKKOS
    Kokkos::fence();
#endif
}

/**
 * @brief Macro for quick profiling
 */
#define NXS_PROFILE_SCOPE(name) nxs::utils::ScopedTimer _profiler_##__LINE__(name)

#define NXS_PROFILE_START(name) nxs::utils::Profiler::instance().start(name)
#define NXS_PROFILE_STOP(name) nxs::utils::Profiler::instance().stop(name)
#define NXS_PROFILE_PRINT() nxs::utils::Profiler::instance().print_all()

} // namespace utils
} // namespace nxs
