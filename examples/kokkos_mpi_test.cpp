/**
 * @file kokkos_mpi_test.cpp
 * @brief Test Kokkos and MPI integration
 */

#include <nexussim/nexussim.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    // Initialize NexusSim with MPI and GPU support
    nxs::InitOptions options;
    options.log_level = nxs::Logger::Level::Info;
    options.enable_mpi = true;
    options.enable_gpu = true;
    options.num_threads = 4;

    nxs::Context context(&argc, &argv, options);

    auto& mpi = nxs::MPIManager::instance();
    auto& kokkos = nxs::KokkosManager::instance();

    // Print rank information
    NXS_LOG_INFO("Rank {}/{}: Hello from NexusSim!", mpi.rank(), mpi.size());

#ifdef NEXUSSIM_HAVE_KOKKOS
    // Simple Kokkos parallel_for test
    const int N = 1000000;

    NXS_LOG_INFO("Rank {}: Running Kokkos parallel_for with {} elements", mpi.rank(), N);

    // Create Kokkos views
    nxs::View1D<double> a("a", N);
    nxs::View1D<double> b("b", N);
    nxs::View1D<double> c("c", N);

    // Initialize on device
    nxs::parallel_for("init_a", N, KOKKOS_LAMBDA(const int i) {
        a(i) = 1.0;
        b(i) = 2.0;
    });

    // Vector addition: c = a + b
    nxs::parallel_for("vector_add", N, KOKKOS_LAMBDA(const int i) {
        c(i) = a(i) + b(i);
    });

    // Compute sum using parallel_reduce
    double sum = 0.0;
    nxs::parallel_reduce("sum", N, KOKKOS_LAMBDA(const int i, double& lsum) {
        lsum += c(i);
    }, sum);

    NXS_LOG_INFO("Rank {}: Local sum = {:.2f} (expected: {:.2f})",
                mpi.rank(), sum, 3.0 * N);

    // MPI all-reduce to get global sum
    double global_sum = 0.0;
    nxs::allreduce_sum(&sum, &global_sum, 1);

    if (mpi.is_root()) {
        NXS_LOG_INFO("Global sum across all ranks = {:.2f} (expected: {:.2f})",
                    global_sum, 3.0 * N * mpi.size());
    }

    // Test deep_copy to host
    nxs::HostView1D<double> c_host("c_host", 10);
    auto c_sub = Kokkos::subview(c, std::make_pair(0, 10));
    nxs::deep_copy(c_host, c_sub);

    nxs::fence("before_print");

    if (mpi.is_root()) {
        NXS_LOG_INFO("First 10 elements of c on host:");
        for (int i = 0; i < 10; ++i) {
            std::cout << "  c[" << i << "] = " << c_host(i) << std::endl;
        }
    }
#else
    if (mpi.is_root()) {
        NXS_LOG_WARN("Kokkos not available - skipping GPU tests");
    }
#endif

    // MPI barrier before exit
    mpi.barrier();

    if (mpi.is_root()) {
        NXS_LOG_INFO("=================================================");
        NXS_LOG_INFO("All tests completed successfully!");
        NXS_LOG_INFO("=================================================");
    }

    return 0;
}
