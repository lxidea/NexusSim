/**
 * @file mpi_wave45d_test.cpp
 * @brief Wave 45d: Parallel I/O + Load Rebalancing tests
 *
 * 12 tests using MPITestHarness/MPITestRunner.
 * Run with: mpiexec -n 2 ./mpi_wave45d_test
 * Serial mode (no MPI) also supported.
 */

#include <nexussim/parallel/parallel_io_wave45.hpp>
#include <nexussim/parallel/mpi_wave45.hpp>
#include <nexussim/parallel/mpi_wave17.hpp>
#include <nexussim/discretization/mesh_partition.hpp>

#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>

using Real = double;
using Index = std::size_t;

int main(int argc, char** argv) {
    nxs::parallel::MPITestHarness harness(argc, argv);
    nxs::parallel::MPITestRunner runner(harness);

    // ========================================================================
    // Tests 1-3: OutputGatherer node data
    // Rank 0 has 4 nodes (IDs 0-3), Rank 1 has 4 nodes (IDs 4-7)
    // Verify 8-node global assembly with correct values at correct positions
    // ========================================================================

    runner.add_test("OutputGatherer_node_data_count",
        [](nxs::parallel::MPIAssert& check, int rank, int size) {
        nxs::parallel::OutputGatherer gatherer;

        // Each rank has 4 nodes with 3 DOFs (positions x,y,z)
        const int dofs = 3;
        const Index num_global = 8;
        std::vector<Index> local_ids;
        std::vector<Real> local_data;

        if (size >= 2) {
            if (rank == 0) {
                local_ids = {0, 1, 2, 3};
                // Node i at position (i*1.0, i*2.0, i*3.0)
                for (Index i = 0; i < 4; ++i) {
                    local_data.push_back(static_cast<Real>(i) * 1.0);
                    local_data.push_back(static_cast<Real>(i) * 2.0);
                    local_data.push_back(static_cast<Real>(i) * 3.0);
                }
            } else if (rank == 1) {
                local_ids = {4, 5, 6, 7};
                for (Index i = 4; i < 8; ++i) {
                    local_data.push_back(static_cast<Real>(i) * 1.0);
                    local_data.push_back(static_cast<Real>(i) * 2.0);
                    local_data.push_back(static_cast<Real>(i) * 3.0);
                }
            }
        } else {
            // Serial: all 8 nodes on rank 0
            local_ids = {0, 1, 2, 3, 4, 5, 6, 7};
            for (Index i = 0; i < 8; ++i) {
                local_data.push_back(static_cast<Real>(i) * 1.0);
                local_data.push_back(static_cast<Real>(i) * 2.0);
                local_data.push_back(static_cast<Real>(i) * 3.0);
            }
        }

        auto global = gatherer.gather_node_data(local_data, dofs, local_ids,
                                                 num_global, 0);

        if (rank == 0) {
            check.check_all(global.size() == num_global * dofs,
                           "Global array size = 24");
        } else {
            // Non-root gets empty in MPI mode
            bool ok = (size == 1) ? (global.size() == num_global * dofs)
                                  : (global.empty());
            check.check_all(ok, "Non-root result correct");
        }
    });

    runner.add_test("OutputGatherer_node_data_values",
        [](nxs::parallel::MPIAssert& check, int rank, int size) {
        nxs::parallel::OutputGatherer gatherer;

        const int dofs = 3;
        const Index num_global = 8;
        std::vector<Index> local_ids;
        std::vector<Real> local_data;

        if (size >= 2) {
            if (rank == 0) {
                local_ids = {0, 1, 2, 3};
                for (Index i = 0; i < 4; ++i) {
                    local_data.push_back(static_cast<Real>(i) * 1.0);
                    local_data.push_back(static_cast<Real>(i) * 2.0);
                    local_data.push_back(static_cast<Real>(i) * 3.0);
                }
            } else if (rank == 1) {
                local_ids = {4, 5, 6, 7};
                for (Index i = 4; i < 8; ++i) {
                    local_data.push_back(static_cast<Real>(i) * 1.0);
                    local_data.push_back(static_cast<Real>(i) * 2.0);
                    local_data.push_back(static_cast<Real>(i) * 3.0);
                }
            }
        } else {
            local_ids = {0, 1, 2, 3, 4, 5, 6, 7};
            for (Index i = 0; i < 8; ++i) {
                local_data.push_back(static_cast<Real>(i) * 1.0);
                local_data.push_back(static_cast<Real>(i) * 2.0);
                local_data.push_back(static_cast<Real>(i) * 3.0);
            }
        }

        auto global = gatherer.gather_node_data(local_data, dofs, local_ids,
                                                 num_global, 0);

        if (rank == 0) {
            // Check node 5: position = (5.0, 10.0, 15.0)
            bool v5_ok = std::abs(global[5 * 3 + 0] - 5.0) < 1e-12 &&
                         std::abs(global[5 * 3 + 1] - 10.0) < 1e-12 &&
                         std::abs(global[5 * 3 + 2] - 15.0) < 1e-12;
            check.check_all(v5_ok, "Node 5 values correct (5, 10, 15)");
        } else {
            check.check_all(true, "Node 5 values correct (5, 10, 15)");
        }
    });

    runner.add_test("OutputGatherer_node_data_boundary",
        [](nxs::parallel::MPIAssert& check, int rank, int size) {
        nxs::parallel::OutputGatherer gatherer;

        const int dofs = 3;
        const Index num_global = 8;
        std::vector<Index> local_ids;
        std::vector<Real> local_data;

        if (size >= 2) {
            if (rank == 0) {
                local_ids = {0, 1, 2, 3};
                for (Index i = 0; i < 4; ++i) {
                    local_data.push_back(static_cast<Real>(i) * 1.0);
                    local_data.push_back(static_cast<Real>(i) * 2.0);
                    local_data.push_back(static_cast<Real>(i) * 3.0);
                }
            } else if (rank == 1) {
                local_ids = {4, 5, 6, 7};
                for (Index i = 4; i < 8; ++i) {
                    local_data.push_back(static_cast<Real>(i) * 1.0);
                    local_data.push_back(static_cast<Real>(i) * 2.0);
                    local_data.push_back(static_cast<Real>(i) * 3.0);
                }
            }
        } else {
            local_ids = {0, 1, 2, 3, 4, 5, 6, 7};
            for (Index i = 0; i < 8; ++i) {
                local_data.push_back(static_cast<Real>(i) * 1.0);
                local_data.push_back(static_cast<Real>(i) * 2.0);
                local_data.push_back(static_cast<Real>(i) * 3.0);
            }
        }

        auto global = gatherer.gather_node_data(local_data, dofs, local_ids,
                                                 num_global, 0);

        if (rank == 0) {
            // Check first and last node
            bool n0_ok = std::abs(global[0] - 0.0) < 1e-12 &&
                         std::abs(global[1] - 0.0) < 1e-12 &&
                         std::abs(global[2] - 0.0) < 1e-12;
            bool n7_ok = std::abs(global[7 * 3 + 0] - 7.0) < 1e-12 &&
                         std::abs(global[7 * 3 + 1] - 14.0) < 1e-12 &&
                         std::abs(global[7 * 3 + 2] - 21.0) < 1e-12;
            check.check_all(n0_ok && n7_ok,
                           "Boundary nodes 0 and 7 correct");
        } else {
            check.check_all(true, "Boundary nodes 0 and 7 correct");
        }
    });

    // ========================================================================
    // Tests 4-5: Scalar gather with global ID mapping
    // Rank 0: scalars [1.0, 2.0] at global IDs [0, 2]
    // Rank 1: scalars [3.0, 4.0] at global IDs [1, 3]
    // Expected global: [1.0, 3.0, 2.0, 4.0]
    // ========================================================================

    runner.add_test("OutputGatherer_scalar_mapping",
        [](nxs::parallel::MPIAssert& check, int rank, int size) {
        nxs::parallel::OutputGatherer gatherer;

        const Index num_global = 4;
        std::vector<Index> local_ids;
        std::vector<Real> local_values;

        if (size >= 2) {
            if (rank == 0) {
                local_ids = {0, 2};
                local_values = {1.0, 2.0};
            } else if (rank == 1) {
                local_ids = {1, 3};
                local_values = {3.0, 4.0};
            }
        } else {
            local_ids = {0, 2, 1, 3};
            local_values = {1.0, 2.0, 3.0, 4.0};
        }

        auto global = gatherer.gather_scalar(local_values, local_ids,
                                              num_global, 0);

        if (rank == 0) {
            bool ok = global.size() == 4 &&
                      std::abs(global[0] - 1.0) < 1e-12 &&
                      std::abs(global[1] - 3.0) < 1e-12 &&
                      std::abs(global[2] - 2.0) < 1e-12 &&
                      std::abs(global[3] - 4.0) < 1e-12;
            check.check_all(ok, "Scalar global = [1, 3, 2, 4]");
        } else {
            check.check_all(true, "Scalar global = [1, 3, 2, 4]");
        }
    });

    runner.add_test("OutputGatherer_scalar_noncontiguous",
        [](nxs::parallel::MPIAssert& check, int rank, int size) {
        nxs::parallel::OutputGatherer gatherer;

        // Rank 0: IDs [0, 4], Rank 1: IDs [2, 6]
        // Global has 8 entities, values at 0,2,4,6 filled, rest 0
        const Index num_global = 8;
        std::vector<Index> local_ids;
        std::vector<Real> local_values;

        if (size >= 2) {
            if (rank == 0) {
                local_ids = {0, 4};
                local_values = {10.0, 40.0};
            } else if (rank == 1) {
                local_ids = {2, 6};
                local_values = {20.0, 60.0};
            }
        } else {
            local_ids = {0, 4, 2, 6};
            local_values = {10.0, 40.0, 20.0, 60.0};
        }

        auto global = gatherer.gather_scalar(local_values, local_ids,
                                              num_global, 0);

        if (rank == 0) {
            bool ok = global.size() == 8 &&
                      std::abs(global[0] - 10.0) < 1e-12 &&
                      std::abs(global[1] - 0.0) < 1e-12 &&
                      std::abs(global[2] - 20.0) < 1e-12 &&
                      std::abs(global[3] - 0.0) < 1e-12 &&
                      std::abs(global[4] - 40.0) < 1e-12 &&
                      std::abs(global[6] - 60.0) < 1e-12;
            check.check_all(ok, "Non-contiguous scalar gather correct");
        } else {
            check.check_all(true, "Non-contiguous scalar gather correct");
        }
    });

    // ========================================================================
    // Tests 6-7: ParallelAnimWriter
    // Both ranks contribute 4 nodes of positions, verify global assembly
    // ========================================================================

    runner.add_test("ParallelAnimWriter_frame_write",
        [](nxs::parallel::MPIAssert& check, int rank, int size) {
        nxs::parallel::ParallelAnimWriter writer;

        const Index num_global = 8;
        std::vector<Index> local_ids;
        std::vector<Real> local_pos;

        if (size >= 2) {
            if (rank == 0) {
                local_ids = {0, 1, 2, 3};
                for (Index i = 0; i < 4; ++i) {
                    local_pos.push_back(static_cast<Real>(i));       // x
                    local_pos.push_back(static_cast<Real>(i) + 0.5); // y
                    local_pos.push_back(0.0);                         // z
                }
            } else if (rank == 1) {
                local_ids = {4, 5, 6, 7};
                for (Index i = 4; i < 8; ++i) {
                    local_pos.push_back(static_cast<Real>(i));
                    local_pos.push_back(static_cast<Real>(i) + 0.5);
                    local_pos.push_back(0.0);
                }
            }
        } else {
            local_ids = {0, 1, 2, 3, 4, 5, 6, 7};
            for (Index i = 0; i < 8; ++i) {
                local_pos.push_back(static_cast<Real>(i));
                local_pos.push_back(static_cast<Real>(i) + 0.5);
                local_pos.push_back(0.0);
            }
        }

        auto result = writer.write_frame(local_pos, local_ids,
                                          num_global, 0.001, 0);

        if (rank == 0) {
            check.check_all(result.written, "Root wrote frame");
        } else {
            check.check_all(!result.written || size == 1,
                           "Non-root did not write frame");
        }
    });

    runner.add_test("ParallelAnimWriter_frame_data",
        [](nxs::parallel::MPIAssert& check, int rank, int size) {
        nxs::parallel::ParallelAnimWriter writer;

        const Index num_global = 8;
        std::vector<Index> local_ids;
        std::vector<Real> local_pos;

        if (size >= 2) {
            if (rank == 0) {
                local_ids = {0, 1, 2, 3};
                for (Index i = 0; i < 4; ++i) {
                    local_pos.push_back(static_cast<Real>(i));
                    local_pos.push_back(static_cast<Real>(i) + 0.5);
                    local_pos.push_back(0.0);
                }
            } else if (rank == 1) {
                local_ids = {4, 5, 6, 7};
                for (Index i = 4; i < 8; ++i) {
                    local_pos.push_back(static_cast<Real>(i));
                    local_pos.push_back(static_cast<Real>(i) + 0.5);
                    local_pos.push_back(0.0);
                }
            }
        } else {
            local_ids = {0, 1, 2, 3, 4, 5, 6, 7};
            for (Index i = 0; i < 8; ++i) {
                local_pos.push_back(static_cast<Real>(i));
                local_pos.push_back(static_cast<Real>(i) + 0.5);
                local_pos.push_back(0.0);
            }
        }

        auto result = writer.write_frame(local_pos, local_ids,
                                          num_global, 0.001, 0);

        if (rank == 0) {
            // Check node 6: x=6, y=6.5, z=0
            bool ok = result.global_positions.size() == 24 &&
                      std::abs(result.global_positions[6 * 3 + 0] - 6.0) < 1e-12 &&
                      std::abs(result.global_positions[6 * 3 + 1] - 6.5) < 1e-12 &&
                      std::abs(result.global_positions[6 * 3 + 2] - 0.0) < 1e-12;
            check.check_all(ok, "Node 6 position correct in frame");
        } else {
            check.check_all(true, "Node 6 position correct in frame");
        }
    });

    // ========================================================================
    // Tests 8-9: MigrationExecutor
    // Imbalanced 6:2 element split -> migrate 2 -> verify 4:4
    // ========================================================================

    runner.add_test("MigrationExecutor_balance_split",
        [](nxs::parallel::MPIAssert& check, int rank, int size) {
        nxs::parallel::MigrationExecutor executor;

        // Initial: rank 0 has 6 elements (0-5), rank 1 has 2 elements (6-7)
        std::vector<std::vector<Index>> elements_per_rank = {
            {0, 1, 2, 3, 4, 5},  // rank 0: 6 elements
            {6, 7}                // rank 1: 2 elements
        };

        // Migration plan: move elements 4,5 from rank 0 to rank 1
        nxs::parallel::MigrationPlan plan;
        plan.should_migrate = true;
        plan.imbalance_before = 1.5;  // 6/4 = 1.5

        nxs::parallel::MigrationEntry e1;
        e1.element_id = 4; e1.from_rank = 0; e1.to_rank = 1; e1.weight = 1.0;
        plan.entries.push_back(e1);

        nxs::parallel::MigrationEntry e2;
        e2.element_id = 5; e2.from_rank = 0; e2.to_rank = 1; e2.weight = 1.0;
        plan.entries.push_back(e2);

        // Element data (just weights as placeholder)
        std::map<Index, std::vector<Real>> elem_data;
        for (Index i = 0; i < 8; ++i) {
            elem_data[i] = {1.0};
        }

        auto result = executor.execute(plan, elem_data, elements_per_rank);

        // After migration: rank 0 should have 4, rank 1 should have 4
        check.check_all(result.executed, "Migration executed");
    });

    runner.add_test("MigrationExecutor_verify_4_4",
        [](nxs::parallel::MPIAssert& check, int rank, int size) {
        nxs::parallel::MigrationExecutor executor;

        std::vector<std::vector<Index>> elements_per_rank = {
            {0, 1, 2, 3, 4, 5},
            {6, 7}
        };

        nxs::parallel::MigrationPlan plan;
        plan.should_migrate = true;

        nxs::parallel::MigrationEntry e1;
        e1.element_id = 4; e1.from_rank = 0; e1.to_rank = 1; e1.weight = 1.0;
        plan.entries.push_back(e1);

        nxs::parallel::MigrationEntry e2;
        e2.element_id = 5; e2.from_rank = 0; e2.to_rank = 1; e2.weight = 1.0;
        plan.entries.push_back(e2);

        std::map<Index, std::vector<Real>> elem_data;
        for (Index i = 0; i < 8; ++i) {
            elem_data[i] = {1.0};
        }

        auto result = executor.execute(plan, elem_data, elements_per_rank);

        // Verify counts
        bool r0_ok = elements_per_rank[0].size() == 4;
        bool r1_ok = elements_per_rank[1].size() == 4;

        // Verify rank 0 has {0,1,2,3} and rank 1 has {4,5,6,7}
        bool content_ok = true;
        std::sort(elements_per_rank[0].begin(), elements_per_rank[0].end());
        std::sort(elements_per_rank[1].begin(), elements_per_rank[1].end());

        std::vector<Index> expected0 = {0, 1, 2, 3};
        std::vector<Index> expected1 = {4, 5, 6, 7};
        content_ok = (elements_per_rank[0] == expected0) &&
                     (elements_per_rank[1] == expected1);

        check.check_all(r0_ok && r1_ok && content_ok,
                        "4:4 split with correct elements");
    });

    // ========================================================================
    // Tests 10-12: DynamicRepartitioner
    // Detect imbalance in 6:2 split, repartition, verify improved balance
    // ========================================================================

    runner.add_test("DynamicRepartitioner_detect_imbalance",
        [](nxs::parallel::MPIAssert& check, int rank, int size) {
        nxs::parallel::DynamicRepartitioner repartitioner(1.2);

        // 6:2 split -> weights [6, 2] -> max/avg = 6/4 = 1.5 > 1.2
        std::vector<Real> rank_weights = {6.0, 2.0};

        bool needs = repartitioner.check_rebalance(rank_weights);
        check.check_all(needs, "Imbalance 1.5 > threshold 1.2 detected");
    });

    runner.add_test("DynamicRepartitioner_rebalance",
        [](nxs::parallel::MPIAssert& check, int rank, int size) {
        nxs::parallel::DynamicRepartitioner repartitioner(1.2);

        // 6 elements on rank 0, 2 on rank 1, all weight 1.0
        std::map<Index, Real> element_weights;
        std::map<Index, int> element_ranks;
        for (Index i = 0; i < 6; ++i) {
            element_weights[i] = 1.0;
            element_ranks[i] = 0;
        }
        for (Index i = 6; i < 8; ++i) {
            element_weights[i] = 1.0;
            element_ranks[i] = 1;
        }

        std::vector<std::vector<Index>> elements_per_rank = {
            {0, 1, 2, 3, 4, 5},
            {6, 7}
        };

        auto result = repartitioner.rebalance(element_weights, element_ranks,
                                               elements_per_rank, 2);

        check.check_all(result.repartitioned, "Repartition executed");
        check.check_all(result.elements_migrated > 0, "Elements were migrated");
    });

    runner.add_test("DynamicRepartitioner_improved_balance",
        [](nxs::parallel::MPIAssert& check, int rank, int size) {
        nxs::parallel::DynamicRepartitioner repartitioner(1.2);

        std::map<Index, Real> element_weights;
        std::map<Index, int> element_ranks;
        for (Index i = 0; i < 6; ++i) {
            element_weights[i] = 1.0;
            element_ranks[i] = 0;
        }
        for (Index i = 6; i < 8; ++i) {
            element_weights[i] = 1.0;
            element_ranks[i] = 1;
        }

        std::vector<std::vector<Index>> elements_per_rank = {
            {0, 1, 2, 3, 4, 5},
            {6, 7}
        };

        auto result = repartitioner.rebalance(element_weights, element_ranks,
                                               elements_per_rank, 2);

        // After rebalancing, imbalance should be improved
        // Check that the element counts are more balanced
        Index r0_count = elements_per_rank[0].size();
        Index r1_count = elements_per_rank[1].size();
        Real new_max = static_cast<Real>(std::max(r0_count, r1_count));
        Real new_avg = static_cast<Real>(r0_count + r1_count) / 2.0;
        Real new_imbalance = new_max / new_avg;

        check.check_all(new_imbalance < 1.5,
                        "Imbalance improved from 1.5 to < 1.5");
        check.check_all(result.imbalance_after <= result.imbalance_before,
                        "imbalance_after <= imbalance_before");
    });

    return runner.run_all();
}
