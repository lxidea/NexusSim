/**
 * @file mpi_wave45a_test.cpp
 * @brief Wave 45a: MPI build fix + multi-rank test infrastructure (10 tests)
 *
 * Tests: MPITestHarness, MPIAssert, MPITestRunner, MPIManager integration,
 *        GhostExchanger 2-rank exchange, RCBPartitioner 2-partition.
 *
 * Run: mpiexec -n 2 ./mpi_wave45a_test
 */

#include <nexussim/parallel/mpi_wave45.hpp>
#include <nexussim/core/mpi.hpp>
#include <nexussim/discretization/mesh_partition.hpp>
#include <cmath>
#include <numeric>

using namespace nxs::parallel;

// ---------------------------------------------------------------------------
// Helper: build a simple 2x2x2 hex mesh (8 elements, 27 nodes)
// ---------------------------------------------------------------------------
static void build_2x2x2_mesh(std::vector<double>& coords,
                              std::vector<std::size_t>& connectivity,
                              std::size_t& num_nodes,
                              std::size_t& num_elements)
{
    num_nodes = 27;
    num_elements = 8;
    coords.resize(27 * 3);

    // 3x3x3 grid of nodes
    int idx = 0;
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) {
                coords[idx * 3 + 0] = static_cast<double>(i);
                coords[idx * 3 + 1] = static_cast<double>(j);
                coords[idx * 3 + 2] = static_cast<double>(k);
                idx++;
            }

    // 8 hex elements, each with 8 nodes
    connectivity.resize(8 * 8);
    auto node_id = [](int i, int j, int k) -> std::size_t {
        return static_cast<std::size_t>(k * 9 + j * 3 + i);
    };

    int eidx = 0;
    for (int kk = 0; kk < 2; ++kk)
        for (int jj = 0; jj < 2; ++jj)
            for (int ii = 0; ii < 2; ++ii) {
                std::size_t base = static_cast<std::size_t>(eidx * 8);
                connectivity[base + 0] = node_id(ii,   jj,   kk);
                connectivity[base + 1] = node_id(ii+1, jj,   kk);
                connectivity[base + 2] = node_id(ii+1, jj+1, kk);
                connectivity[base + 3] = node_id(ii,   jj+1, kk);
                connectivity[base + 4] = node_id(ii,   jj,   kk+1);
                connectivity[base + 5] = node_id(ii+1, jj,   kk+1);
                connectivity[base + 6] = node_id(ii+1, jj+1, kk+1);
                connectivity[base + 7] = node_id(ii,   jj+1, kk+1);
                eidx++;
            }
}

int main(int argc, char** argv)
{
    MPITestHarness mpi(argc, argv);
    MPITestRunner runner(mpi);

    // -----------------------------------------------------------------------
    // Tests 1-3: MPITestHarness reports correct rank/size
    // -----------------------------------------------------------------------
    runner.add_test("Harness: rank is valid", [&](MPIAssert& a, int rank, int size) {
        a.check_all(rank >= 0 && rank < size,
                    "rank in [0, size)");
    });

    runner.add_test("Harness: size >= 1", [&](MPIAssert& a, int /*rank*/, int size) {
        a.check_all(size >= 1, "size >= 1");
    });

    runner.add_test("Harness: ranks are distinct", [&](MPIAssert& a, int rank, int size) {
        // Each rank contributes its own rank number; sum should be size*(size-1)/2
        int local = rank;
        int global_sum = local;
#ifdef NEXUSSIM_HAVE_MPI
        MPI_Allreduce(&local, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
        int expected = size * (size - 1) / 2;
        a.check_all(global_sum == expected,
                    "sum of ranks == n*(n-1)/2 (ranks are distinct)");
    });

    // -----------------------------------------------------------------------
    // Tests 4-5: broadcast + allreduce from core/mpi.hpp with 2 ranks
    // -----------------------------------------------------------------------
    runner.add_test("Core MPI: broadcast double", [&](MPIAssert& a, int rank, int /*size*/) {
        double val = (rank == 0) ? 42.0 : 0.0;
#ifdef NEXUSSIM_HAVE_MPI
        MPI_Bcast(&val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        a.check_near_all(val, 42.0, 1e-12, "broadcast value == 42.0 on all ranks");
    });

    runner.add_test("Core MPI: allreduce sum", [&](MPIAssert& a, int rank, int /*size*/) {
        double local_val = static_cast<double>(rank + 1);  // rank0=1, rank1=2
        double global_sum = local_val;
#ifdef NEXUSSIM_HAVE_MPI
        MPI_Allreduce(&local_val, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        // For 2 ranks: 1+2=3
        double expected = 0.0;
        int sz = 1;
#ifdef NEXUSSIM_HAVE_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &sz);
#endif
        for (int r = 0; r < sz; ++r) expected += (r + 1);
        a.check_near_all(global_sum, expected, 1e-12,
                         "allreduce sum matches expected");
    });

    // -----------------------------------------------------------------------
    // Tests 6-7: GhostExchanger from mesh_partition.hpp with real 2-rank exchange
    // -----------------------------------------------------------------------
    runner.add_test("GhostExchange: scalar exchange", [&](MPIAssert& a, int rank, int size) {
        if (size < 2) {
            a.check_all(true, "scalar exchange (skipped, size < 2)");
            return;
        }

        // Build mesh and partition for this rank
        std::vector<double> coords;
        std::vector<std::size_t> connectivity;
        std::size_t nn, ne;
        build_2x2x2_mesh(coords, connectivity, nn, ne);

        nxs::discretization::RCBPartitioner partitioner;
        auto part = partitioner.create_partition(
            coords.data(), connectivity.data(), nn, ne, 8, size, rank);

        // Create a scalar field: each owned node gets rank+1 as value
        std::size_t total_local = part.local_nodes.size() + part.ghost_nodes.size();
        std::vector<double> field(total_local, 0.0);
        for (std::size_t i = 0; i < part.local_nodes.size(); ++i) {
            field[i] = static_cast<double>(rank + 1);
        }

#ifdef NEXUSSIM_HAVE_MPI
        nxs::discretization::GhostExchange ghost;
        ghost.initialize(part);
        ghost.exchange_scalar(field);

        // After exchange, ghost nodes should have the value from the owner rank
        bool ghost_ok = true;
        for (std::size_t i = part.local_nodes.size(); i < total_local; ++i) {
            std::size_t global_node = part.local_to_global_node[i];
            int owner = part.node_owner[global_node];
            double expected_val = static_cast<double>(owner + 1);
            if (std::abs(field[i] - expected_val) > 1e-12) {
                ghost_ok = false;
            }
        }
        a.check_all(ghost_ok, "ghost nodes received correct scalar values");
#else
        a.check_all(true, "ghost exchange (serial fallback)");
#endif
    });

    runner.add_test("GhostExchange: vector exchange", [&](MPIAssert& a, int rank, int size) {
        if (size < 2) {
            a.check_all(true, "vector exchange (skipped, size < 2)");
            return;
        }

        std::vector<double> coords;
        std::vector<std::size_t> connectivity;
        std::size_t nn, ne;
        build_2x2x2_mesh(coords, connectivity, nn, ne);

        nxs::discretization::RCBPartitioner partitioner;
        auto part = partitioner.create_partition(
            coords.data(), connectivity.data(), nn, ne, 8, size, rank);

        std::size_t total_local = part.local_nodes.size() + part.ghost_nodes.size();
        std::vector<double> field(total_local * 3, 0.0);
        for (std::size_t i = 0; i < part.local_nodes.size(); ++i) {
            field[i * 3 + 0] = static_cast<double>(rank) * 10.0;
            field[i * 3 + 1] = static_cast<double>(rank) * 20.0;
            field[i * 3 + 2] = static_cast<double>(rank) * 30.0;
        }

#ifdef NEXUSSIM_HAVE_MPI
        nxs::discretization::GhostExchange ghost;
        ghost.initialize(part);
        ghost.exchange_vector(field, 3);

        bool ghost_ok = true;
        for (std::size_t i = part.local_nodes.size(); i < total_local; ++i) {
            std::size_t global_node = part.local_to_global_node[i];
            int owner = part.node_owner[global_node];
            double ex = static_cast<double>(owner) * 10.0;
            double ey = static_cast<double>(owner) * 20.0;
            double ez = static_cast<double>(owner) * 30.0;
            if (std::abs(field[i*3+0] - ex) > 1e-12 ||
                std::abs(field[i*3+1] - ey) > 1e-12 ||
                std::abs(field[i*3+2] - ez) > 1e-12) {
                ghost_ok = false;
            }
        }
        a.check_all(ghost_ok, "ghost nodes received correct vector values");
#else
        a.check_all(true, "vector exchange (serial fallback)");
#endif
    });

    // -----------------------------------------------------------------------
    // Tests 8-10: RCBPartitioner produces valid 2-partition
    // -----------------------------------------------------------------------
    runner.add_test("RCBPartitioner: valid partition assignment", [&](MPIAssert& a, int /*rank*/, int size) {
        std::vector<double> coords;
        std::vector<std::size_t> connectivity;
        std::size_t nn, ne;
        build_2x2x2_mesh(coords, connectivity, nn, ne);

        nxs::discretization::RCBPartitioner partitioner;
        auto elem_part = partitioner.partition_elements(
            coords.data(), connectivity.data(), nn, ne, 8, size);

        // Every element should be assigned to a valid rank
        bool valid = true;
        for (auto p : elem_part) {
            if (p < 0 || p >= size) valid = false;
        }
        a.check_all(valid, "all elements assigned to valid ranks");
    });

    runner.add_test("RCBPartitioner: each rank gets elements", [&](MPIAssert& a, int rank, int size) {
        std::vector<double> coords;
        std::vector<std::size_t> connectivity;
        std::size_t nn, ne;
        build_2x2x2_mesh(coords, connectivity, nn, ne);

        nxs::discretization::RCBPartitioner partitioner;
        auto part = partitioner.create_partition(
            coords.data(), connectivity.data(), nn, ne, 8, size, rank);

        // Each rank should own at least 1 element
        a.check_all(part.num_local_elements() > 0,
                     "rank owns at least 1 element");
    });

    runner.add_test("RCBPartitioner: total elements preserved", [&](MPIAssert& a, int rank, int size) {
        std::vector<double> coords;
        std::vector<std::size_t> connectivity;
        std::size_t nn, ne;
        build_2x2x2_mesh(coords, connectivity, nn, ne);

        nxs::discretization::RCBPartitioner partitioner;
        auto part = partitioner.create_partition(
            coords.data(), connectivity.data(), nn, ne, 8, size, rank);

        int local_count = static_cast<int>(part.num_local_elements());
        int total = local_count;
#ifdef NEXUSSIM_HAVE_MPI
        MPI_Allreduce(&local_count, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
        a.check_all(total == 8, "total elements across all ranks == 8");
    });

    return runner.run_all();
}
