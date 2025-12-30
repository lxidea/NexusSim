/**
 * @file kokkos_performance_test.cpp
 * @brief Standalone Kokkos performance test
 *
 * Tests Kokkos parallel performance without full FEM solver dependencies.
 * Supports multiple backends:
 *   - OpenMP (CPU multi-threading)
 *   - CUDA (NVIDIA GPU)
 *   - HIP (AMD GPU via ROCm)
 *   - Serial (single-threaded fallback)
 *
 * Usage:
 *   ./kokkos_performance_test [--small] [--large]
 *
 * Build with CUDA:
 *   nvcc_wrapper -std=c++17 -O3 kokkos_performance_test.cpp -lkokkoscore
 *
 * Build with HIP:
 *   hipcc -std=c++17 -O3 kokkos_performance_test.cpp -lkokkoscore
 *
 * Build with OpenMP:
 *   g++ -std=c++20 -O2 -fopenmp kokkos_performance_test.cpp -lkokkoscore
 */

#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>

#ifdef KOKKOS_ENABLE_OPENMP
#include <omp.h>
#endif

using Real = double;

// Backend detection macros
#if defined(KOKKOS_ENABLE_CUDA)
    #define BACKEND_NAME "CUDA"
    #define IS_GPU_BACKEND true
#elif defined(KOKKOS_ENABLE_HIP)
    #define BACKEND_NAME "HIP (AMD ROCm)"
    #define IS_GPU_BACKEND true
#elif defined(KOKKOS_ENABLE_SYCL)
    #define BACKEND_NAME "SYCL (Intel)"
    #define IS_GPU_BACKEND true
#elif defined(KOKKOS_ENABLE_OPENMP)
    #define BACKEND_NAME "OpenMP"
    #define IS_GPU_BACKEND false
#else
    #define BACKEND_NAME "Serial"
    #define IS_GPU_BACKEND false
#endif

// Performance timer
class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    void stop() { end_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        std::chrono::duration<double, std::milli> elapsed = end_ - start_;
        return elapsed.count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_, end_;
};

struct BenchmarkResult {
    std::string name;
    int size;
    double time_ms;
    double throughput;

    void print() const {
        std::cout << std::left << std::setw(30) << name
                  << std::setw(15) << size
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::scientific << std::setprecision(2) << throughput << "\n";
    }

    static void print_header() {
        std::cout << std::left << std::setw(30) << "Benchmark"
                  << std::setw(15) << "Size"
                  << std::setw(15) << "Time (ms)"
                  << "Throughput\n";
        std::cout << std::string(75, '-') << "\n";
    }
};

// Test 1: Vector addition (DAXPY-like)
BenchmarkResult test_vector_add(int n, int iterations) {
    Kokkos::View<Real*> x("x", n);
    Kokkos::View<Real*> y("y", n);
    Kokkos::View<Real*> z("z", n);

    // Initialize
    Kokkos::parallel_for("init", n, KOKKOS_LAMBDA(int i) {
        x(i) = static_cast<Real>(i);
        y(i) = static_cast<Real>(n - i);
    });
    Kokkos::fence();

    Timer timer;
    timer.start();

    for (int iter = 0; iter < iterations; ++iter) {
        const Real alpha = 2.5;
        Kokkos::parallel_for("vector_add", n, KOKKOS_LAMBDA(int i) {
            z(i) = alpha * x(i) + y(i);
        });
    }
    Kokkos::fence();

    timer.stop();

    BenchmarkResult result;
    result.name = "Vector Add (DAXPY)";
    result.size = n;
    result.time_ms = timer.elapsed_ms() / iterations;
    result.throughput = 3.0 * n * sizeof(Real) / (result.time_ms * 1e-3 * 1e9);  // GB/s
    return result;
}

// Test 2: Dot product (reduction)
BenchmarkResult test_dot_product(int n, int iterations) {
    Kokkos::View<Real*> x("x", n);
    Kokkos::View<Real*> y("y", n);

    // Initialize
    Kokkos::parallel_for("init", n, KOKKOS_LAMBDA(int i) {
        x(i) = 1.0 / (i + 1);
        y(i) = static_cast<Real>(i);
    });
    Kokkos::fence();

    Timer timer;
    timer.start();

    Real result_sum = 0.0;
    for (int iter = 0; iter < iterations; ++iter) {
        Real dot = 0.0;
        Kokkos::parallel_reduce("dot_product", n,
            KOKKOS_LAMBDA(int i, Real& lsum) {
                lsum += x(i) * y(i);
            }, dot);
        result_sum += dot;
    }
    Kokkos::fence();

    timer.stop();

    // Use result to prevent optimization
    volatile Real use = result_sum;
    (void)use;

    BenchmarkResult result;
    result.name = "Dot Product (reduce)";
    result.size = n;
    result.time_ms = timer.elapsed_ms() / iterations;
    result.throughput = 2.0 * n * sizeof(Real) / (result.time_ms * 1e-3 * 1e9);  // GB/s
    return result;
}

// Test 3: Matrix-vector product (sparse-like pattern)
BenchmarkResult test_matvec(int n, int nnz_per_row, int iterations) {
    int total_nnz = n * nnz_per_row;

    Kokkos::View<Real*> A("A", total_nnz);
    Kokkos::View<int*> cols("cols", total_nnz);
    Kokkos::View<int*> row_ptr("row_ptr", n + 1);
    Kokkos::View<Real*> x("x", n);
    Kokkos::View<Real*> y("y", n);

    // Initialize sparse matrix pattern (band structure)
    Kokkos::parallel_for("init_matrix", n, KOKKOS_LAMBDA(int i) {
        row_ptr(i) = i * nnz_per_row;
        int half_band = nnz_per_row / 2;
        for (int j = 0; j < nnz_per_row; ++j) {
            int col = i - half_band + j;
            if (col < 0) col = 0;
            if (col >= n) col = n - 1;
            cols(i * nnz_per_row + j) = col;
            A(i * nnz_per_row + j) = 1.0 / (j + 1);
        }
        x(i) = 1.0;
    });
    Kokkos::parallel_for("init_row_ptr_end", 1, KOKKOS_LAMBDA(int) {
        row_ptr(n) = n * nnz_per_row;
    });
    Kokkos::fence();

    Timer timer;
    timer.start();

    for (int iter = 0; iter < iterations; ++iter) {
        Kokkos::parallel_for("spmv", n, KOKKOS_LAMBDA(int i) {
            Real sum = 0.0;
            int start = row_ptr(i);
            int end = row_ptr(i + 1);
            for (int k = start; k < end; ++k) {
                sum += A(k) * x(cols(k));
            }
            y(i) = sum;
        });
    }
    Kokkos::fence();

    timer.stop();

    BenchmarkResult result;
    result.name = "SpMV (" + std::to_string(nnz_per_row) + " nnz/row)";
    result.size = n;
    result.time_ms = timer.elapsed_ms() / iterations;
    result.throughput = (2.0 * total_nnz + n) * sizeof(Real) / (result.time_ms * 1e-3 * 1e9);  // GB/s
    return result;
}

// Test 4: Element force computation (FEM-like)
BenchmarkResult test_element_forces(int num_elements, int iterations) {
    const int nodes_per_elem = 8;  // Hex8
    const int dofs_per_node = 3;
    const int elem_dofs = nodes_per_elem * dofs_per_node;
    int num_nodes = num_elements + 1;  // Simplified
    int num_dofs = num_nodes * dofs_per_node;

    Kokkos::View<Real*> displacement("disp", num_dofs);
    Kokkos::View<Real*> force("force", num_dofs);
    Kokkos::View<int*> connectivity("conn", num_elements * nodes_per_elem);

    // Initialize
    Kokkos::parallel_for("init_disp", num_dofs, KOKKOS_LAMBDA(int i) {
        displacement(i) = 1e-6 * i;
    });

    Kokkos::parallel_for("init_conn", num_elements, KOKKOS_LAMBDA(int e) {
        for (int n = 0; n < nodes_per_elem; ++n) {
            connectivity(e * nodes_per_elem + n) = (e + n) % (num_elements + 1);
        }
    });
    Kokkos::fence();

    Timer timer;
    timer.start();

    for (int iter = 0; iter < iterations; ++iter) {
        // Zero forces
        Kokkos::parallel_for("zero_force", num_dofs, KOKKOS_LAMBDA(int i) {
            force(i) = 0.0;
        });

        // Element loop
        Kokkos::parallel_for("element_forces", num_elements, KOKKOS_LAMBDA(int e) {
            Real elem_disp[elem_dofs];
            Real elem_force[elem_dofs] = {0};

            // Gather
            for (int n = 0; n < nodes_per_elem; ++n) {
                int node = connectivity(e * nodes_per_elem + n);
                for (int d = 0; d < dofs_per_node; ++d) {
                    elem_disp[n * dofs_per_node + d] = displacement(node * dofs_per_node + d);
                }
            }

            // Simple stiffness operation (k * u, simplified)
            const Real k = 1e6;
            for (int i = 0; i < elem_dofs; ++i) {
                elem_force[i] = k * elem_disp[i];
            }

            // Scatter with atomic add
            for (int n = 0; n < nodes_per_elem; ++n) {
                int node = connectivity(e * nodes_per_elem + n);
                for (int d = 0; d < dofs_per_node; ++d) {
                    Kokkos::atomic_add(&force(node * dofs_per_node + d),
                                      elem_force[n * dofs_per_node + d]);
                }
            }
        });
    }
    Kokkos::fence();

    timer.stop();

    BenchmarkResult result;
    result.name = "Element Forces (Hex8)";
    result.size = num_elements;
    result.time_ms = timer.elapsed_ms() / iterations;
    result.throughput = num_elements / (result.time_ms * 1e-3);  // elements/sec
    return result;
}

// Test 5: Time integration step
BenchmarkResult test_time_integration(int num_dofs, int iterations) {
    Kokkos::View<Real*> displacement("disp", num_dofs);
    Kokkos::View<Real*> velocity("vel", num_dofs);
    Kokkos::View<Real*> acceleration("acc", num_dofs);
    Kokkos::View<Real*> force("force", num_dofs);
    Kokkos::View<Real*> mass("mass", num_dofs);

    // Initialize
    Kokkos::parallel_for("init", num_dofs, KOKKOS_LAMBDA(int i) {
        displacement(i) = 0.0;
        velocity(i) = 1.0;
        force(i) = 100.0;
        mass(i) = 1.0;
    });
    Kokkos::fence();

    const Real dt = 1e-6;

    Timer timer;
    timer.start();

    for (int iter = 0; iter < iterations; ++iter) {
        Kokkos::parallel_for("time_step", num_dofs, KOKKOS_LAMBDA(int i) {
            Real a = force(i) / mass(i);
            velocity(i) += a * dt;
            displacement(i) += velocity(i) * dt;
            acceleration(i) = a;
        });
    }
    Kokkos::fence();

    timer.stop();

    BenchmarkResult result;
    result.name = "Time Integration";
    result.size = num_dofs;
    result.time_ms = timer.elapsed_ms() / iterations;
    result.throughput = num_dofs / (result.time_ms * 1e-3);  // DOFs/sec
    return result;
}

// Print detailed backend information
void print_backend_info() {
    std::cout << "Kokkos Configuration:\n";
    std::cout << "  Execution Space: " << Kokkos::DefaultExecutionSpace::name() << "\n";
    std::cout << "  Memory Space:    " << Kokkos::DefaultExecutionSpace::memory_space::name() << "\n";
    std::cout << "  Compiled Backend: " << BACKEND_NAME << "\n";

    // Check actual runtime capabilities
    std::string exec_space = Kokkos::DefaultExecutionSpace::name();
    bool is_cuda = exec_space.find("Cuda") != std::string::npos;
    bool is_hip = exec_space.find("HIP") != std::string::npos;
    bool is_sycl = exec_space.find("SYCL") != std::string::npos;
    bool is_gpu = is_cuda || is_hip || is_sycl;

    std::cout << "  GPU Enabled:     " << (is_gpu ? "Yes" : "No (CPU backend)") << "\n";

    if (is_cuda) {
        std::cout << "  GPU Type:        NVIDIA (CUDA)\n";
    } else if (is_hip) {
        std::cout << "  GPU Type:        AMD (ROCm/HIP)\n";
    } else if (is_sycl) {
        std::cout << "  GPU Type:        Intel (SYCL)\n";
    }

#ifdef KOKKOS_ENABLE_OPENMP
    std::cout << "  OpenMP Threads:  " << omp_get_max_threads() << "\n";
#endif

    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        std::cout << "\n";
        std::cout << "============================================================\n";
        std::cout << "NexusSim Kokkos Performance Test\n";
        std::cout << "============================================================\n";
        std::cout << "Supports: CUDA (NVIDIA), HIP (AMD ROCm), OpenMP, Serial\n";
        std::cout << "============================================================\n\n";

        // Print Kokkos info
        print_backend_info();

        bool is_gpu = std::string(Kokkos::DefaultExecutionSpace::name()).find("Cuda") != std::string::npos ||
                      std::string(Kokkos::DefaultExecutionSpace::name()).find("HIP") != std::string::npos;

        std::vector<BenchmarkResult> results;

        // Problem sizes
        std::vector<int> sizes = {10000, 100000, 1000000};
        if (argc > 1 && std::string(argv[1]) == "--small") {
            sizes = {1000, 10000};
        }

        std::cout << "Running benchmarks...\n\n";

        for (int n : sizes) {
            std::cout << "Size: " << n << "\n";
            std::cout << std::string(40, '-') << "\n";

            int iters = (n < 100000) ? 100 : 50;

            results.push_back(test_vector_add(n, iters));
            std::cout << "  Vector Add:     " << std::fixed << std::setprecision(3)
                      << results.back().time_ms << " ms\n";

            results.push_back(test_dot_product(n, iters));
            std::cout << "  Dot Product:    " << results.back().time_ms << " ms\n";

            results.push_back(test_matvec(n, 7, iters));
            std::cout << "  SpMV (7 nnz):   " << results.back().time_ms << " ms\n";

            int num_elem = n / 8;
            results.push_back(test_element_forces(num_elem, iters / 2));
            std::cout << "  Element Forces: " << results.back().time_ms << " ms\n";

            results.push_back(test_time_integration(n * 3, iters));
            std::cout << "  Time Step:      " << results.back().time_ms << " ms\n";

            std::cout << "\n";
        }

        // Summary
        std::cout << "============================================================\n";
        std::cout << "Performance Summary\n";
        std::cout << "============================================================\n\n";

        BenchmarkResult::print_header();
        for (const auto& r : results) {
            r.print();
        }

        // Find peak performance
        double max_elem_rate = 0;
        double max_dof_rate = 0;
        for (const auto& r : results) {
            if (r.name.find("Element") != std::string::npos) {
                max_elem_rate = std::max(max_elem_rate, r.throughput);
            }
            if (r.name.find("Time") != std::string::npos) {
                max_dof_rate = std::max(max_dof_rate, r.throughput);
            }
        }

        std::cout << "\n";
        std::cout << "============================================================\n";
        std::cout << "Peak Performance\n";
        std::cout << "============================================================\n";
        std::cout << "  Element Processing: " << std::scientific << std::setprecision(2)
                  << max_elem_rate << " elements/sec\n";
        std::cout << "  DOF Update Rate:    " << max_dof_rate << " DOFs/sec\n";
        std::cout << "  Backend:            " << Kokkos::DefaultExecutionSpace::name() << "\n";
        std::cout << "============================================================\n\n";

        if (is_gpu) {
            std::cout << "GPU backend active. Performance should scale with problem size.\n";
            std::cout << "Expected GPU speedup: 10-100x over CPU for large problems.\n";
        } else {
            std::cout << "Running on CPU backend.\n\n";
            std::cout << "To enable NVIDIA GPU (CUDA):\n";
            std::cout << "  1. Install CUDA toolkit: sudo apt install nvidia-cuda-toolkit\n";
            std::cout << "  2. Rebuild Kokkos: cmake -DKokkos_ENABLE_CUDA=ON ..\n";
            std::cout << "  3. Rebuild NexusSim and rerun benchmark\n\n";
            std::cout << "To enable AMD GPU (ROCm/HIP):\n";
            std::cout << "  1. Install ROCm: amdgpu-install --usecase=rocm\n";
            std::cout << "  2. Rebuild Kokkos: cmake -DKokkos_ENABLE_HIP=ON ..\n";
            std::cout << "  3. Rebuild NexusSim and rerun benchmark\n";
            std::cout << "  Note: ROCm requires native Linux (not WSL2)\n";
        }
        std::cout << "\n";
    }
    Kokkos::finalize();
    return 0;
}
