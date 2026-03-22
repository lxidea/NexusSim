#pragma once

/**
 * @file mpi_wave45.hpp
 * @brief Wave 45a: MPI build fix + multi-rank test infrastructure
 *
 * Provides RAII test harness, collective assertions, and test runner
 * for SPMD MPI tests. All classes have serial fallbacks when
 * NEXUSSIM_HAVE_MPI is not defined.
 */

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <functional>
#include <sstream>

#ifdef NEXUSSIM_HAVE_MPI
#include <mpi.h>
#endif

namespace nxs {
namespace parallel {

// ============================================================================
// MPITestHarness — RAII wrapper for MPI init/finalize
// ============================================================================

class MPITestHarness {
public:
    MPITestHarness(int& argc, char**& argv) {
#ifdef NEXUSSIM_HAVE_MPI
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
#else
        (void)argc;
        (void)argv;
        rank_ = 0;
        size_ = 1;
#endif
    }

    ~MPITestHarness() {
#ifdef NEXUSSIM_HAVE_MPI
        MPI_Finalize();
#endif
    }

    MPITestHarness(const MPITestHarness&) = delete;
    MPITestHarness& operator=(const MPITestHarness&) = delete;

    int rank() const { return rank_; }
    int size() const { return size_; }
    bool is_root() const { return rank_ == 0; }

    void barrier() const {
#ifdef NEXUSSIM_HAVE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

private:
    int rank_ = 0;
    int size_ = 1;
};

// ============================================================================
// MPIAssert — Collective assertions across all ranks
// ============================================================================

class MPIAssert {
public:
    explicit MPIAssert(int rank = 0) : rank_(rank) {}

    /**
     * @brief Collective boolean check — all ranks must pass
     * @return true if all ranks passed
     */
    bool check_all(bool local_pass, const std::string& name) {
        total_checks_++;
        bool global_pass = local_pass;

#ifdef NEXUSSIM_HAVE_MPI
        int local_val = local_pass ? 1 : 0;
        int global_val = 0;
        MPI_Allreduce(&local_val, &global_val, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        global_pass = (global_val != 0);
#endif

        if (!global_pass) {
            failed_checks_++;
            if (rank_ == 0) {
                std::cerr << "  FAIL: " << name << "\n";
            }
        } else {
            passed_checks_++;
            if (rank_ == 0) {
                std::cout << "  PASS: " << name << "\n";
            }
        }
        return global_pass;
    }

    /**
     * @brief Collective near-equality check for floating point values
     * @return true if all ranks passed
     */
    bool check_near_all(double val, double expected, double tol,
                        const std::string& name) {
        bool local_pass = std::abs(val - expected) <= tol;
        return check_all(local_pass, name);
    }

    /**
     * @brief Check that a value is the same on all ranks
     * @return true if all ranks have the same value
     */
    bool check_equal_all(double val, const std::string& name) {
        bool pass = true;
#ifdef NEXUSSIM_HAVE_MPI
        double min_val = 0.0, max_val = 0.0;
        MPI_Allreduce(&val, &min_val, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&val, &max_val, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        pass = (std::abs(max_val - min_val) < 1e-12);
#else
        (void)val;
#endif
        return check_all(pass, name);
    }

    int passed() const { return passed_checks_; }
    int failed() const { return failed_checks_; }
    int total() const { return total_checks_; }

private:
    int rank_ = 0;
    int total_checks_ = 0;
    int passed_checks_ = 0;
    int failed_checks_ = 0;
};

// ============================================================================
// MPITestRunner — Orchestrates test execution across ranks
// ============================================================================

class MPITestRunner {
public:
    explicit MPITestRunner(const MPITestHarness& harness)
        : rank_(harness.rank()), size_(harness.size()),
          assert_(harness.rank()) {}

    using TestFunc = std::function<void(MPIAssert&, int rank, int size)>;

    /**
     * @brief Register a named test
     */
    void add_test(const std::string& name, TestFunc func) {
        tests_.push_back({name, std::move(func)});
    }

    /**
     * @brief Run all registered tests
     * @return 0 if all pass, 1 if any fail
     */
    int run_all() {
        if (rank_ == 0) {
            std::cout << "========================================\n";
            std::cout << "MPI Test Runner (" << size_ << " ranks)\n";
            std::cout << "========================================\n";
        }

        for (auto& [name, func] : tests_) {
            if (rank_ == 0) {
                std::cout << "\n[TEST] " << name << "\n";
            }

#ifdef NEXUSSIM_HAVE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif

            func(assert_, rank_, size_);

#ifdef NEXUSSIM_HAVE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
        }

        // Collect final results
        int local_failed = assert_.failed();
        int global_failed = local_failed;

#ifdef NEXUSSIM_HAVE_MPI
        MPI_Allreduce(&local_failed, &global_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif

        if (rank_ == 0) {
            std::cout << "\n========================================\n";
            std::cout << "Results: " << assert_.passed() << " passed, "
                      << assert_.failed() << " failed out of "
                      << assert_.total() << " checks\n";
            std::cout << "========================================\n";
        }

        return (global_failed > 0) ? 1 : 0;
    }

    MPIAssert& assert_ref() { return assert_; }

private:
    int rank_;
    int size_;
    MPIAssert assert_;
    std::vector<std::pair<std::string, TestFunc>> tests_;
};

// ============================================================================
// Utility: MPI datatype helper
// ============================================================================

#ifdef NEXUSSIM_HAVE_MPI

template<typename T>
inline MPI_Datatype mpi_type() {
    if constexpr (std::is_same_v<T, int>) return MPI_INT;
    else if constexpr (std::is_same_v<T, long>) return MPI_LONG;
    else if constexpr (std::is_same_v<T, long long>) return MPI_LONG_LONG;
    else if constexpr (std::is_same_v<T, unsigned>) return MPI_UNSIGNED;
    else if constexpr (std::is_same_v<T, unsigned long>) return MPI_UNSIGNED_LONG;
    else if constexpr (std::is_same_v<T, unsigned long long>) return MPI_UNSIGNED_LONG_LONG;
    else if constexpr (std::is_same_v<T, float>) return MPI_FLOAT;
    else if constexpr (std::is_same_v<T, double>) return MPI_DOUBLE;
    else if constexpr (std::is_same_v<T, char>) return MPI_CHAR;
    else if constexpr (std::is_same_v<T, std::size_t>) return MPI_UNSIGNED_LONG;
    else return MPI_BYTE;
}

#endif

} // namespace parallel
} // namespace nxs
