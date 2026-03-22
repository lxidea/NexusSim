/**
 * @file mpi_wave45b_test.cpp
 * @brief Wave 45b: Production Force Assembly + Ghost Exchange (12 tests)
 *
 * Tests: FrontierPattern construction, ForceExchanger accumulate/scatter,
 *        DistributedTimeStep, DistributedEnergyMonitor.
 *
 * Run: mpiexec -n 2 ./mpi_wave45b_test
 * Serial: ./mpi_wave45b_test  (serial fallbacks exercised)
 */

#include <nexussim/parallel/mpi_wave45.hpp>
#include <nexussim/parallel/force_exchange_wave45.hpp>
#include <nexussim/discretization/mesh_partition.hpp>
#include <cmath>
#include <numeric>

using namespace nxs::parallel;
using nxs::Index;

// ---------------------------------------------------------------------------
// Helper: build a simple 2x2x2 hex mesh (8 elements, 27 nodes)
// Identical to mpi_wave45a_test.cpp
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

// ---------------------------------------------------------------------------
// Helper: create partition for this rank from the 2x2x2 mesh
// ---------------------------------------------------------------------------
static nxs::discretization::MeshPartition make_partition(int rank, int size) {
    std::vector<double> coords;
    std::vector<std::size_t> connectivity;
    std::size_t nn, ne;
    build_2x2x2_mesh(coords, connectivity, nn, ne);

    nxs::discretization::RCBPartitioner partitioner;
    return partitioner.create_partition(
        coords.data(), connectivity.data(), nn, ne, 8, size, rank);
}

int main(int argc, char** argv)
{
    MPITestHarness mpi(argc, argv);
    MPITestRunner runner(mpi);

    // =======================================================================
    // Tests 1-3: FrontierPattern construction from CommPattern
    // =======================================================================

    runner.add_test("FrontierPattern: neighbor count matches CommPattern",
        [&](MPIAssert& a, int rank, int size) {
            auto part = make_partition(rank, size);
            FrontierPattern fp;
            fp.build(part, 3);

            bool ok = (fp.num_neighbors() == part.comm_patterns.size());
            a.check_all(ok, "neighbor count == comm_patterns.size()");
        });

    runner.add_test("FrontierPattern: send/recv frontier sizes match CommPattern",
        [&](MPIAssert& a, int rank, int size) {
            auto part = make_partition(rank, size);
            FrontierPattern fp;
            fp.build(part, 3);

            bool ok = true;
            for (std::size_t i = 0; i < fp.neighbors.size(); ++i) {
                const auto& nf = fp.neighbors[i];
                const auto& cp = part.comm_patterns[i];

                // send_frontier comes from cp.recv_nodes (ghosts on this rank)
                // recv_frontier comes from cp.send_nodes (shared nodes on this rank)
                if (nf.send_frontier.size() != cp.recv_nodes.size()) ok = false;
                if (nf.recv_frontier.size() != cp.send_nodes.size()) ok = false;
            }
            a.check_all(ok, "send/recv frontier sizes match CommPattern");
        });

    runner.add_test("FrontierPattern: buffer allocation correct",
        [&](MPIAssert& a, int rank, int size) {
            auto part = make_partition(rank, size);
            FrontierPattern fp;
            fp.build(part, 3);

            bool ok = true;
            for (const auto& nf : fp.neighbors) {
                if (nf.send_buffer.size() != nf.send_frontier.size() * 3) ok = false;
                if (nf.recv_buffer.size() != nf.recv_frontier.size() * 3) ok = false;
                if (nf.dofs_per_node != 3) ok = false;
            }
            a.check_all(ok, "buffers sized correctly for 3 DOFs");
        });

    // =======================================================================
    // Tests 4-6: Force accumulate cycle
    // =======================================================================

    runner.add_test("ForceExchanger: accumulate setup succeeds",
        [&](MPIAssert& a, int rank, int size) {
            auto part = make_partition(rank, size);
            ForceExchanger fe;
            fe.setup(part, 3);

            a.check_all(fe.is_initialized(), "exchanger is initialized");
            a.check_all(fe.dofs_per_node() == 3, "dofs_per_node == 3");
        });

    runner.add_test("ForceExchanger: accumulate sums ghost contributions",
        [&](MPIAssert& a, int rank, int size) {
            if (size < 2) {
                a.check_all(true, "accumulate (skipped, size < 2)");
                return;
            }

            auto part = make_partition(rank, size);
            ForceExchanger fe;
            fe.setup(part, 3);

            // Total local nodes = owned + ghost
            std::size_t total_local = part.local_nodes.size() + part.ghost_nodes.size();
            std::vector<double> forces(total_local * 3, 0.0);

            // Each rank sets force = (rank+1) * 10 on ALL its nodes (owned + ghost).
            // After accumulate, owned shared nodes should have the sum of
            // contributions from this rank AND the neighbor's ghost copy.
            double my_force = static_cast<double>(rank + 1) * 10.0;
            for (std::size_t i = 0; i < total_local; ++i) {
                forces[i * 3 + 0] = my_force;
                forces[i * 3 + 1] = my_force;
                forces[i * 3 + 2] = my_force;
            }

            // Record pre-accumulate owned values for shared nodes
            // After accumulate, shared nodes should have: my_force + neighbor_force
            fe.begin_accumulate(forces);
            fe.finish_accumulate(forces);

            // Verify: ghost nodes should be zeroed (their contribution was sent)
            bool ghosts_zeroed = true;
            for (std::size_t i = part.local_nodes.size(); i < total_local; ++i) {
                for (int d = 0; d < 3; ++d) {
                    if (std::abs(forces[i * 3 + d]) > 1e-12) {
                        ghosts_zeroed = false;
                    }
                }
            }
            a.check_all(ghosts_zeroed,
                        "ghost forces zeroed after accumulate");
        });

    runner.add_test("ForceExchanger: accumulated forces sum correctly at shared nodes",
        [&](MPIAssert& a, int rank, int size) {
            if (size < 2) {
                a.check_all(true, "accumulate sum (skipped, size < 2)");
                return;
            }

            auto part = make_partition(rank, size);
            ForceExchanger fe;
            fe.setup(part, 3);

            std::size_t total_local = part.local_nodes.size() + part.ghost_nodes.size();
            std::vector<double> forces(total_local * 3, 0.0);

            // Each rank contributes (rank+1)*10 at every node
            double my_force = static_cast<double>(rank + 1) * 10.0;
            for (std::size_t i = 0; i < total_local; ++i) {
                forces[i * 3 + 0] = my_force;
                forces[i * 3 + 1] = my_force;
                forces[i * 3 + 2] = my_force;
            }

            fe.begin_accumulate(forces);
            fe.finish_accumulate(forces);

            // For shared nodes (owned nodes that have ghost copies on other ranks),
            // the accumulated force should be my_force + neighbor_force.
            // For 2 ranks: rank0 shared nodes get 10 + 20 = 30,
            //              rank1 shared nodes get 20 + 10 = 30.
            // For non-shared owned nodes, force stays at my_force.
            bool shared_ok = true;
            double expected_shared = 10.0 + 20.0;  // sum of both ranks

            for (const auto& cp : part.comm_patterns) {
                for (Index local_idx : cp.send_nodes) {
                    // send_nodes = our shared owned nodes
                    for (int d = 0; d < 3; ++d) {
                        double val = forces[local_idx * 3 + d];
                        if (std::abs(val - expected_shared) > 1e-10) {
                            shared_ok = false;
                        }
                    }
                }
            }
            a.check_all(shared_ok,
                        "shared node forces == 30 (10 + 20 from both ranks)");
        });

    // =======================================================================
    // Tests 7-8: Scatter cycle (owner -> ghost)
    // =======================================================================

    runner.add_test("ForceExchanger: scatter sends owner values to ghosts",
        [&](MPIAssert& a, int rank, int size) {
            if (size < 2) {
                a.check_all(true, "scatter (skipped, size < 2)");
                return;
            }

            auto part = make_partition(rank, size);
            ForceExchanger fe;
            fe.setup(part, 3);

            std::size_t total_local = part.local_nodes.size() + part.ghost_nodes.size();
            std::vector<double> accel(total_local * 3, 0.0);

            // Each rank sets acceleration on its OWNED nodes only
            double my_accel = static_cast<double>(rank + 1) * 100.0;
            for (std::size_t i = 0; i < part.local_nodes.size(); ++i) {
                accel[i * 3 + 0] = my_accel;
                accel[i * 3 + 1] = my_accel * 2.0;
                accel[i * 3 + 2] = my_accel * 3.0;
            }
            // Ghost entries are zero initially

            fe.begin_scatter(accel);
            fe.finish_scatter(accel);

            // After scatter, ghost nodes should have the values from their owner rank
            bool ghost_ok = true;
            for (std::size_t i = part.local_nodes.size(); i < total_local; ++i) {
                std::size_t global_node = part.local_to_global_node[i];
                int owner = part.node_owner[global_node];
                double expected_a = static_cast<double>(owner + 1) * 100.0;

                if (std::abs(accel[i * 3 + 0] - expected_a) > 1e-10 ||
                    std::abs(accel[i * 3 + 1] - expected_a * 2.0) > 1e-10 ||
                    std::abs(accel[i * 3 + 2] - expected_a * 3.0) > 1e-10) {
                    ghost_ok = false;
                }
            }
            a.check_all(ghost_ok,
                        "ghost nodes received correct acceleration from owner");
        });

    runner.add_test("ForceExchanger: scatter preserves owned values",
        [&](MPIAssert& a, int rank, int size) {
            if (size < 2) {
                a.check_all(true, "scatter preserve (skipped, size < 2)");
                return;
            }

            auto part = make_partition(rank, size);
            ForceExchanger fe;
            fe.setup(part, 3);

            std::size_t total_local = part.local_nodes.size() + part.ghost_nodes.size();
            std::vector<double> accel(total_local * 3, 0.0);

            double my_accel = static_cast<double>(rank + 1) * 100.0;
            for (std::size_t i = 0; i < part.local_nodes.size(); ++i) {
                accel[i * 3 + 0] = my_accel;
                accel[i * 3 + 1] = my_accel;
                accel[i * 3 + 2] = my_accel;
            }

            fe.begin_scatter(accel);
            fe.finish_scatter(accel);

            // Owned node values should be unchanged
            bool owned_ok = true;
            for (std::size_t i = 0; i < part.local_nodes.size(); ++i) {
                for (int d = 0; d < 3; ++d) {
                    if (std::abs(accel[i * 3 + d] - my_accel) > 1e-12) {
                        owned_ok = false;
                    }
                }
            }
            a.check_all(owned_ok, "owned node values unchanged after scatter");
        });

    // =======================================================================
    // Tests 9-10: DistributedTimeStep
    // =======================================================================

    runner.add_test("DistributedTimeStep: global min selects smallest dt",
        [&](MPIAssert& a, int rank, int /*size*/) {
            DistributedTimeStep dts;

            // Rank 0 proposes 1e-5, rank 1 proposes 2e-5
            double local_dt = (rank == 0) ? 1e-5 : 2e-5;
            double global_dt = dts.compute_global_dt(local_dt);

            a.check_near_all(global_dt, 1e-5, 1e-15,
                             "global dt == 1e-5 (minimum of 1e-5 and 2e-5)");
        });

    runner.add_test("DistributedTimeStep: all ranks agree on global dt",
        [&](MPIAssert& a, int rank, int /*size*/) {
            DistributedTimeStep dts;

            double local_dt = (rank == 0) ? 1e-5 : 2e-5;
            double global_dt = dts.compute_global_dt(local_dt);

            a.check_equal_all(global_dt, "all ranks have same global dt");
        });

    // =======================================================================
    // Tests 11-12: DistributedEnergyMonitor
    // =======================================================================

    runner.add_test("DistributedEnergyMonitor: KE sums correctly",
        [&](MPIAssert& a, int rank, int /*size*/) {
            DistributedEnergyMonitor dem;

            // Rank 0: KE=100, IE=50, EE=10
            // Rank 1: KE=200, IE=75, EE=20
            double local_ke = (rank == 0) ? 100.0 : 200.0;
            double local_ie = (rank == 0) ?  50.0 :  75.0;
            double local_ee = (rank == 0) ?  10.0 :  20.0;

            auto ge = dem.reduce_energies(local_ke, local_ie, local_ee);

            // For serial (size=1): totals equal local values
            // For 2 ranks: KE=300, IE=125, EE=30
#ifdef NEXUSSIM_HAVE_MPI
            int sz = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &sz);
            if (sz >= 2) {
                a.check_near_all(ge.kinetic, 300.0, 1e-10, "global KE == 300");
            } else {
                a.check_near_all(ge.kinetic, local_ke, 1e-10, "serial KE == local");
            }
#else
            a.check_near_all(ge.kinetic, local_ke, 1e-10, "serial KE == local");
#endif
        });

    runner.add_test("DistributedEnergyMonitor: IE and EE sum correctly",
        [&](MPIAssert& a, int rank, int /*size*/) {
            DistributedEnergyMonitor dem;

            double local_ke = (rank == 0) ? 100.0 : 200.0;
            double local_ie = (rank == 0) ?  50.0 :  75.0;
            double local_ee = (rank == 0) ?  10.0 :  20.0;

            auto ge = dem.reduce_energies(local_ke, local_ie, local_ee);

#ifdef NEXUSSIM_HAVE_MPI
            int sz = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &sz);
            if (sz >= 2) {
                bool ie_ok = std::abs(ge.internal_ - 125.0) < 1e-10;
                bool ee_ok = std::abs(ge.external_ - 30.0) < 1e-10;
                a.check_all(ie_ok && ee_ok,
                            "global IE == 125, EE == 30");
            } else {
                a.check_all(true, "serial energy (skipped)");
            }
#else
            bool ie_ok = std::abs(ge.internal_ - local_ie) < 1e-10;
            bool ee_ok = std::abs(ge.external_ - local_ee) < 1e-10;
            a.check_all(ie_ok && ee_ok, "serial IE/EE == local values");
#endif
        });

    return runner.run_all();
}
