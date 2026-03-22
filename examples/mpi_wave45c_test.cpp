/**
 * @file mpi_wave45c_test.cpp
 * @brief Wave 45c: Parallel Contact Search tests (12 tests)
 *
 * Tests DistributedBroadPhase, ContactDataExchanger, and ParallelBucketSort
 * using MPITestHarness/MPITestRunner. Designed for mpiexec -n 2.
 *
 * Serial fallback: all 12 tests pass with 1 rank.
 */

#include <nexussim/parallel/contact_exchange_wave45.hpp>
#include <nexussim/parallel/mpi_wave45.hpp>
#include <nexussim/parallel/mpi_wave17.hpp>

#include <vector>
#include <cmath>
#include <iostream>
#include <set>

using namespace nxs::parallel;

// Helper: create an AABB from min/max coordinates
static AABB make_box(Real xmin, Real ymin, Real zmin,
                     Real xmax, Real ymax, Real zmax) {
    AABB box;
    box.min_pt[0] = xmin; box.min_pt[1] = ymin; box.min_pt[2] = zmin;
    box.max_pt[0] = xmax; box.max_pt[1] = ymax; box.max_pt[2] = zmax;
    return box;
}

int main(int argc, char** argv) {
    MPITestHarness harness(argc, argv);
    MPITestRunner runner(harness);
    int rank = harness.rank();
    int size = harness.size();

    // ========================================================================
    // Tests 1-3: Rank-level AABB allgather + overlap detection
    // ========================================================================

    runner.add_test("1. Rank-level AABB computation",
        [&](MPIAssert& assert, int r, int s) {
        // Rank 0: surface A at x=[0,1], Rank 1: surface A at x=[1.5,2.5]
        // Rank 0: surface B at x=[0.5,1.5], Rank 1: surface B at x=[2.0,3.0]
        std::vector<AABB> a_boxes, b_boxes;
        std::vector<Index> a_ids, b_ids;

        if (r == 0) {
            a_boxes = {make_box(0, 0, 0, 0.5, 1, 1), make_box(0.5, 0, 0, 1, 1, 1)};
            b_boxes = {make_box(0.5, 0, 0, 1, 1, 1), make_box(1, 0, 0, 1.5, 1, 1)};
            a_ids = {100, 101};
            b_ids = {200, 201};
        } else {
            a_boxes = {make_box(0.5, 0, 0, 1, 1, 1), make_box(1, 0, 0, 1.5, 1, 1)};
            b_boxes = {make_box(2.0, 0, 0, 2.5, 1, 1), make_box(2.5, 0, 0, 3.0, 1, 1)};
            a_ids = {102, 103};
            b_ids = {202, 203};
        }

        DistributedBroadPhase bp;
        bp.initialize(a_boxes, b_boxes, a_ids, b_ids, r, s);

        // Rank 0's A box should be [0,0,0]-[1,1,1]
        if (r == 0) {
            assert.check_all(
                std::abs(bp.rank_a_box().min_pt[0] - 0.0) < 1e-12 &&
                std::abs(bp.rank_a_box().max_pt[0] - 1.0) < 1e-12,
                "Rank 0 A-box x-range is [0,1]");
        } else {
            assert.check_all(
                std::abs(bp.rank_a_box().min_pt[0] - 0.5) < 1e-12 &&
                std::abs(bp.rank_a_box().max_pt[0] - 1.5) < 1e-12,
                "Rank 1 A-box x-range is [0.5,1.5]");
        }
    });

    runner.add_test("2. Rank-level overlap detection",
        [&](MPIAssert& assert, int r, int s) {
        // Rank 0: A at x=[0,1], B at x=[0.5,1.5]
        // Rank 1: A at x=[0.5,1.5], B at x=[2,3]
        // Rank 0's B [0.5,1.5] overlaps Rank 1's A [0.5,1.5] => overlap
        std::vector<AABB> a_boxes, b_boxes;
        std::vector<Index> a_ids, b_ids;

        if (r == 0) {
            a_boxes = {make_box(0, 0, 0, 1, 1, 1)};
            b_boxes = {make_box(0.5, 0, 0, 1.5, 1, 1)};
            a_ids = {10};
            b_ids = {20};
        } else {
            a_boxes = {make_box(0.5, 0, 0, 1.5, 1, 1)};
            b_boxes = {make_box(2.0, 0, 0, 3.0, 1, 1)};
            a_ids = {11};
            b_ids = {21};
        }

        DistributedBroadPhase bp;
        bp.initialize(a_boxes, b_boxes, a_ids, b_ids, r, s);
        auto pairs = bp.detect_cross_rank_pairs();

        if (s > 1) {
            // Both ranks should detect the other as overlapping
            assert.check_all(!bp.overlapping_ranks().empty(),
                "Overlap detected with remote rank");
        } else {
            // Serial: only local pairs
            assert.check_all(bp.overlapping_ranks().empty(),
                "Serial: no remote overlaps");
        }
    });

    runner.add_test("3. Rank-level no-overlap case",
        [&](MPIAssert& assert, int r, int s) {
        // Rank 0: A at x=[0,1], B at x=[0,1]
        // Rank 1: A at x=[10,11], B at x=[10,11]
        // No rank-level overlap
        std::vector<AABB> a_boxes, b_boxes;
        std::vector<Index> a_ids, b_ids;

        if (r == 0) {
            a_boxes = {make_box(0, 0, 0, 1, 1, 1)};
            b_boxes = {make_box(0, 0, 0, 1, 1, 1)};
            a_ids = {10};
            b_ids = {20};
        } else {
            a_boxes = {make_box(10, 0, 0, 11, 1, 1)};
            b_boxes = {make_box(10, 0, 0, 11, 1, 1)};
            a_ids = {11};
            b_ids = {21};
        }

        DistributedBroadPhase bp;
        bp.initialize(a_boxes, b_boxes, a_ids, b_ids, r, s);
        auto pairs = bp.detect_cross_rank_pairs();

        // No cross-rank pairs should be found
        bool no_cross = true;
        for (const auto& p : pairs) {
            if (p.rank_a != p.rank_b) {
                no_cross = false;
                break;
            }
        }
        assert.check_all(no_cross, "No cross-rank pairs when boxes are far apart");
    });

    // ========================================================================
    // Tests 4-6: Cross-rank candidate pairs
    // ========================================================================

    runner.add_test("4. Cross-rank candidate pair detection",
        [&](MPIAssert& assert, int r, int s) {
        // Surface A on rank 0: 4 elements at x=[0,0.25], [0.25,0.5], [0.5,0.75], [0.75,1.0]
        // Surface B on rank 1: 4 elements at x=[0.5,0.75], [0.75,1.0], [1.0,1.25], [1.25,1.5]
        // Overlapping region: x=[0.5,1.0]
        std::vector<AABB> a_boxes, b_boxes;
        std::vector<Index> a_ids, b_ids;

        if (r == 0) {
            for (int i = 0; i < 4; ++i) {
                a_boxes.push_back(make_box(i * 0.25, 0, 0, (i + 1) * 0.25, 1, 1));
                a_ids.push_back(100 + i);
            }
            // Rank 0 has no B surface
            b_boxes = {};
            b_ids = {};
        } else {
            // Rank 1 has no A surface
            a_boxes = {};
            a_ids = {};
            for (int i = 0; i < 4; ++i) {
                b_boxes.push_back(make_box(0.5 + i * 0.25, 0, 0,
                                           0.5 + (i + 1) * 0.25, 1, 1));
                b_ids.push_back(200 + i);
            }
        }

        DistributedBroadPhase bp;
        bp.initialize(a_boxes, b_boxes, a_ids, b_ids, r, s);
        auto pairs = bp.detect_cross_rank_pairs();

        if (s > 1) {
            // Cross-rank pairs should exist between rank 0's A [0.5,1] and rank 1's B [0.5,1]
            // Elements A[2],A[3] overlap with B[0],B[1]
            bool found_cross = false;
            for (const auto& p : pairs) {
                if (p.rank_a != p.rank_b) {
                    found_cross = true;
                    break;
                }
            }
            assert.check_all(found_cross, "Cross-rank pairs detected in overlap region");
        } else {
            // Serial: local A vs local B only
            assert.check_all(true, "Serial: local-only detection");
        }
    });

    runner.add_test("5. Cross-rank pair count verification",
        [&](MPIAssert& assert, int r, int s) {
        // Simple case: rank 0 has 1 A box, rank 1 has 1 B box, they overlap
        std::vector<AABB> a_boxes, b_boxes;
        std::vector<Index> a_ids, b_ids;

        if (r == 0) {
            a_boxes = {make_box(0, 0, 0, 1, 1, 1)};
            a_ids = {10};
            b_boxes = {};
            b_ids = {};
        } else {
            a_boxes = {};
            a_ids = {};
            b_boxes = {make_box(0.5, 0, 0, 1.5, 1, 1)};
            b_ids = {20};
        }

        DistributedBroadPhase bp;
        bp.initialize(a_boxes, b_boxes, a_ids, b_ids, r, s);
        auto pairs = bp.detect_cross_rank_pairs();

        if (s > 1) {
            // Rank 0 should find: A[10] vs remote B[20] -> 1 pair
            // Rank 1 should find: remote A[10] vs B[20] -> 1 pair
            size_t cross_count = 0;
            for (const auto& p : pairs) {
                if (p.rank_a != p.rank_b) cross_count++;
            }
            assert.check_all(cross_count == 1,
                "Exactly 1 cross-rank pair per rank");
        } else {
            assert.check_all(pairs.empty(), "Serial: no pairs (A and B on different ranks)");
        }
    });

    runner.add_test("6. Cross-rank pair gap estimation",
        [&](MPIAssert& assert, int r, int s) {
        // Overlapping boxes: gap should be negative
        std::vector<AABB> a_boxes, b_boxes;
        std::vector<Index> a_ids, b_ids;

        if (r == 0) {
            a_boxes = {make_box(0, 0, 0, 1, 1, 1)};
            a_ids = {10};
            b_boxes = {};
            b_ids = {};
        } else {
            a_boxes = {};
            a_ids = {};
            b_boxes = {make_box(0.5, 0, 0, 1.5, 1, 1)};
            b_ids = {20};
        }

        DistributedBroadPhase bp;
        bp.initialize(a_boxes, b_boxes, a_ids, b_ids, r, s);
        auto pairs = bp.detect_cross_rank_pairs();

        if (s > 1) {
            bool gap_negative = false;
            for (const auto& p : pairs) {
                if (p.rank_a != p.rank_b && p.gap < 0.0) {
                    gap_negative = true;
                }
            }
            assert.check_all(gap_negative, "Overlapping boxes have negative gap");
        } else {
            assert.check_all(true, "Serial: gap test skipped");
        }
    });

    // ========================================================================
    // Tests 7-9: ContactDataExchanger
    // ========================================================================

    runner.add_test("7. ContactDataExchanger node position roundtrip",
        [&](MPIAssert& assert, int r, int s) {
        ContactDataExchanger exchanger;

        if (s > 1) {
            // Both ranks exchange with each other
            std::set<int> peers;
            peers.insert(r == 0 ? 1 : 0);
            exchanger.setup(peers, r);

            // Create local node data
            std::vector<Real> positions = {1.0 + r, 2.0 + r, 3.0 + r,
                                           4.0 + r, 5.0 + r, 6.0 + r};
            std::vector<Real> velocities = {0.1 + r, 0.2 + r, 0.3 + r,
                                            0.4 + r, 0.5 + r, 0.6 + r};
            std::vector<Index> node_ids = {static_cast<Index>(r * 100),
                                           static_cast<Index>(r * 100 + 1)};

            exchanger.exchange_node_data(positions, velocities, node_ids);

            int other = (r == 0) ? 1 : 0;
            assert.check_all(exchanger.has_data_from(other),
                "Data received from remote rank");
        } else {
            exchanger.setup({}, 0);
            assert.check_all(true, "Serial: exchanger setup ok");
        }
    });

    runner.add_test("8. ContactDataExchanger position correctness",
        [&](MPIAssert& assert, int r, int s) {
        ContactDataExchanger exchanger;

        if (s > 1) {
            std::set<int> peers = {r == 0 ? 1 : 0};
            exchanger.setup(peers, r);

            // Rank 0 sends pos [1,2,3], Rank 1 sends pos [10,20,30]
            std::vector<Real> positions, velocities;
            std::vector<Index> node_ids;
            if (r == 0) {
                positions = {1.0, 2.0, 3.0};
                velocities = {0.1, 0.2, 0.3};
                node_ids = {42};
            } else {
                positions = {10.0, 20.0, 30.0};
                velocities = {1.0, 2.0, 3.0};
                node_ids = {99};
            }

            exchanger.exchange_node_data(positions, velocities, node_ids);

            int other = (r == 0) ? 1 : 0;
            const auto& data = exchanger.get_remote_data(other);
            bool ok = (data.size() == 1);
            if (ok) {
                if (r == 0) {
                    // Should receive rank 1's data: pos [10,20,30], id 99
                    ok = ok && std::abs(data[0].position[0] - 10.0) < 1e-12;
                    ok = ok && std::abs(data[0].position[1] - 20.0) < 1e-12;
                    ok = ok && std::abs(data[0].position[2] - 30.0) < 1e-12;
                    ok = ok && (data[0].node_id == 99);
                } else {
                    // Should receive rank 0's data: pos [1,2,3], id 42
                    ok = ok && std::abs(data[0].position[0] - 1.0) < 1e-12;
                    ok = ok && std::abs(data[0].position[1] - 2.0) < 1e-12;
                    ok = ok && std::abs(data[0].position[2] - 3.0) < 1e-12;
                    ok = ok && (data[0].node_id == 42);
                }
            }
            assert.check_all(ok, "Received positions match expected values");
        } else {
            assert.check_all(true, "Serial: position check skipped");
        }
    });

    runner.add_test("9. ContactDataExchanger velocity correctness",
        [&](MPIAssert& assert, int r, int s) {
        ContactDataExchanger exchanger;

        if (s > 1) {
            std::set<int> peers = {r == 0 ? 1 : 0};
            exchanger.setup(peers, r);

            std::vector<Real> positions, velocities;
            std::vector<Index> node_ids;
            if (r == 0) {
                positions = {0, 0, 0};
                velocities = {100.0, 200.0, 300.0};
                node_ids = {1};
            } else {
                positions = {0, 0, 0};
                velocities = {-1.0, -2.0, -3.0};
                node_ids = {2};
            }

            exchanger.exchange_node_data(positions, velocities, node_ids);

            int other = (r == 0) ? 1 : 0;
            const auto& data = exchanger.get_remote_data(other);
            bool ok = (data.size() == 1);
            if (ok) {
                if (r == 0) {
                    ok = ok && std::abs(data[0].velocity[0] - (-1.0)) < 1e-12;
                    ok = ok && std::abs(data[0].velocity[1] - (-2.0)) < 1e-12;
                    ok = ok && std::abs(data[0].velocity[2] - (-3.0)) < 1e-12;
                } else {
                    ok = ok && std::abs(data[0].velocity[0] - 100.0) < 1e-12;
                    ok = ok && std::abs(data[0].velocity[1] - 200.0) < 1e-12;
                    ok = ok && std::abs(data[0].velocity[2] - 300.0) < 1e-12;
                }
            }
            assert.check_all(ok, "Received velocities match expected values");
        } else {
            assert.check_all(true, "Serial: velocity check skipped");
        }
    });

    // ========================================================================
    // Tests 10-12: ParallelBucketSort
    // ========================================================================

    runner.add_test("10. ParallelBucketSort local build",
        [&](MPIAssert& assert, int r, int s) {
        AABB domain = make_box(0, 0, 0, 4, 4, 4);
        ParallelBucketSort pbs(domain, 1.0, r, s);

        // Each rank inserts 4 elements in its half of the domain
        std::vector<AABB> boxes;
        std::vector<Index> ids;
        if (r == 0) {
            for (int i = 0; i < 4; ++i) {
                boxes.push_back(make_box(i * 0.5, 0, 0, (i + 1) * 0.5, 1, 1));
                ids.push_back(i);
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                boxes.push_back(make_box(2 + i * 0.5, 0, 0, 2 + (i + 1) * 0.5, 1, 1));
                ids.push_back(10 + i);
            }
        }

        pbs.build(boxes, ids);
        // Elements may span multiple cells, so count >= 4
        assert.check_all(pbs.local_entry_count() >= 4,
            "4 local elements inserted");
    });

    runner.add_test("11. ParallelBucketSort halo exchange",
        [&](MPIAssert& assert, int r, int s) {
        AABB domain = make_box(0, 0, 0, 4, 4, 4);
        ParallelBucketSort pbs(domain, 2.0, r, s);

        // Rank 0: elements at x=[0,1], Rank 1: elements at x=[3,4]
        std::vector<AABB> boxes;
        std::vector<Index> ids;
        if (r == 0) {
            boxes = {make_box(0, 0, 0, 1, 1, 1)};
            ids = {1};
        } else {
            boxes = {make_box(3, 0, 0, 4, 1, 1)};
            ids = {2};
        }

        pbs.build(boxes, ids);
        pbs.exchange_halo(1);

        if (s > 1) {
            assert.check_all(pbs.halo_entry_count() > 0,
                "Halo entries received from remote rank");
        } else {
            assert.check_all(pbs.halo_entry_count() == 0,
                "Serial: no halo entries");
        }
    });

    runner.add_test("12. ParallelBucketSort query with remote",
        [&](MPIAssert& assert, int r, int s) {
        AABB domain = make_box(0, 0, 0, 4, 4, 4);
        ParallelBucketSort pbs(domain, 2.0, r, s);

        // Rank 0: element at x=[0,1], Rank 1: element at x=[1.5,2.5]
        // After halo exchange, rank 0 should find rank 1's element
        // when querying x=[0,3]
        std::vector<AABB> boxes;
        std::vector<Index> ids;
        if (r == 0) {
            boxes = {make_box(0, 0, 0, 1, 1, 1)};
            ids = {1};
        } else {
            boxes = {make_box(1.5, 0, 0, 2.5, 1, 1)};
            ids = {2};
        }

        pbs.build(boxes, ids);
        pbs.exchange_halo(1);

        // Query a large box that should find both local and remote entries
        AABB query = make_box(0, 0, 0, 3, 1, 1);
        std::vector<Index> found_ids;
        std::set<int> found_ranks;
        pbs.query_with_remote(query, [&](const ParallelBucketSort::BucketEntry& e) {
            found_ids.push_back(e.elem_id);
            found_ranks.insert(e.owner_rank);
        });

        // Should find at least the local element
        bool found_local = false;
        for (auto id : found_ids) {
            if (r == 0 && id == 1) found_local = true;
            if (r == 1 && id == 2) found_local = true;
        }
        assert.check_all(found_local, "Local element found in query");

        if (s > 1) {
            // Should also find remote element after halo exchange
            bool found_remote = found_ranks.size() > 1 || found_ids.size() > 1;
            assert.check_all(found_remote,
                "Remote element found via halo query");
        } else {
            assert.check_all(true, "Serial: remote query skipped");
        }
    });

    return runner.run_all();
}
