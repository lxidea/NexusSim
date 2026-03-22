/**
 * @file xfem_wave41_test.cpp
 * @brief Wave 41: XFEM Production Hardening Test Suite (4 features, 40 tests)
 *
 * Tests:
 *   1. XFEMFatigueCrack    (10 tests)
 *   2. XFEMMultiCrack      (10 tests)
 *   3. XFEMAdaptiveMesh    (10 tests)
 *   4. XFEMOutputFields    (10 tests)
 */

#include <nexussim/fem/xfem_wave41.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace nxs::fem;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; std::cout << "[FAIL] " << msg << "\n"; } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) do { \
    double _va = static_cast<double>(a); \
    double _vb = static_cast<double>(b); \
    double _vt = static_cast<double>(tol); \
    if (std::abs(_va - _vb) <= _vt) { tests_passed++; } \
    else { tests_failed++; std::cout << "[FAIL] " << msg \
        << " (got " << _va << ", expected " << _vb << ", tol " << _vt << ")\n"; } \
} while(0)

// ============================================================================
// 1. XFEMFatigueCrack Tests
// ============================================================================

void test_1_fatigue_crack() {
    std::cout << "--- Test 1: XFEMFatigueCrack ---\n";

    // 1a. Paris law growth rate: da/dN = C * (delta_K)^m
    {
        XFEMFatigueCrack::Params params;
        params.C_paris = 1.0e-11;
        params.m_paris = 3.0;
        params.threshold_K = 0.0;
        params.critical_K = 1.0e6;
        params.initial_length = 0.001;
        params.specimen_width = 0.1;

        XFEMFatigueCrack fatigue(params);
        Real delta_K = 10.0;
        Real da_dN = fatigue.compute_growth_rate(delta_K);
        Real expected = params.C_paris * std::pow(delta_K, params.m_paris);
        CHECK_NEAR(da_dN, expected, 1.0e-15,
                   "Fatigue: Paris law da/dN = C * dK^m");
    }

    // 1b. Threshold SIF: no growth below threshold_K
    {
        XFEMFatigueCrack::Params params;
        params.C_paris = 1.0e-10;
        params.m_paris = 3.0;
        params.threshold_K = 5.0;
        params.critical_K = 1.0e6;

        XFEMFatigueCrack fatigue(params);
        Real da_dN = fatigue.compute_growth_rate(3.0);
        CHECK_NEAR(da_dN, 0.0, 1.0e-20,
                   "Fatigue: no growth below threshold_K");
    }

    // 1c. At threshold exactly -> zero growth
    {
        XFEMFatigueCrack::Params params;
        params.C_paris = 1.0e-10;
        params.m_paris = 3.0;
        params.threshold_K = 5.0;
        params.critical_K = 1.0e6;

        XFEMFatigueCrack fatigue(params);
        Real da_dN = fatigue.compute_growth_rate(5.0);
        CHECK_NEAR(da_dN, 0.0, 1.0e-20,
                   "Fatigue: at threshold exactly -> zero growth");
    }

    // 1d. Critical SIF: rapid fracture above critical_K
    {
        XFEMFatigueCrack::Params params;
        params.C_paris = 1.0e-10;
        params.m_paris = 3.0;
        params.threshold_K = 0.0;
        params.critical_K = 50.0;

        XFEMFatigueCrack fatigue(params);
        Real rate_above = fatigue.compute_growth_rate(60.0);
        CHECK(rate_above >= 1.0e30, "Fatigue: rapid fracture above critical_K");
        Real rate_below = fatigue.compute_growth_rate(40.0);
        CHECK(rate_below < 1.0e30, "Fatigue: no rapid fracture below critical_K");
    }

    // 1e. advance_crack accumulates length and cycles
    {
        XFEMFatigueCrack::Params params;
        params.C_paris = 1.0e-11;
        params.m_paris = 3.0;
        params.threshold_K = 0.0;
        params.critical_K = 1.0e6;
        params.initial_length = 0.001;
        params.specimen_width = 0.1;

        XFEMFatigueCrack fatigue(params);
        Real sif_range = 20.0;
        Real cycles = 1000.0;
        Real da_dN = fatigue.compute_growth_rate(sif_range);
        Real expected_da = da_dN * cycles;
        Real initial_len = fatigue.get_crack_length();

        bool still_growing = fatigue.advance_crack(cycles, sif_range);
        CHECK(still_growing, "Fatigue: crack still growing after moderate advance");
        CHECK_NEAR(fatigue.get_crack_length(), initial_len + expected_da, 1.0e-15,
                   "Fatigue: crack length accumulates correctly");
        CHECK_NEAR(fatigue.get_total_cycles(), cycles, 1.0e-10,
                   "Fatigue: total cycles tracked");
    }

    // 1f. Growth history recorded after advance
    {
        XFEMFatigueCrack::Params params;
        params.C_paris = 1.0e-10;
        params.m_paris = 2.0;
        params.threshold_K = 0.0;
        params.critical_K = 1.0e6;
        params.initial_length = 0.01;
        params.specimen_width = 0.1;

        XFEMFatigueCrack fatigue(params);
        fatigue.advance_crack(100.0, 10.0);
        fatigue.advance_crack(200.0, 15.0);

        CHECK(fatigue.growth_history().size() == 2,
              "Fatigue: two growth history records after two advances");
        CHECK_NEAR(fatigue.growth_history()[0].cycles, 100.0, 1.0e-10,
                   "Fatigue: first record cycles correct");
        CHECK_NEAR(fatigue.growth_history()[1].cycles, 300.0, 1.0e-10,
                   "Fatigue: second record cycles cumulative");
    }

    // 1g. SIF computation: K = sigma * sqrt(pi * a) * F(a/W)
    {
        XFEMFatigueCrack::Params params;
        params.specimen_width = 0.1;
        params.initial_length = 0.001;

        XFEMFatigueCrack fatigue(params);
        Real sigma = 100.0;
        Real a = 0.005;
        Real K = fatigue.compute_sif(sigma, a);
        Real F = fatigue.geometry_factor(a, params.specimen_width);
        Real expected = sigma * std::sqrt(M_PI * a) * F;
        CHECK_NEAR(K, expected, 1.0e-10,
                   "Fatigue: SIF = sigma * sqrt(pi*a) * F(a/W)");
    }

    // 1h. Geometry factor at a/W=0 is approximately 1.12
    {
        XFEMFatigueCrack fatigue;
        Real F = fatigue.geometry_factor(0.0, 1.0);
        CHECK_NEAR(F, 1.12, 0.01, "Fatigue: F(0) ~ 1.12 (Tada-Paris-Irwin)");
    }

    // 1i. Cycle counting from stress history
    {
        XFEMFatigueCrack fatigue;
        // Simple sine-like stress history: 0, 100, 0, 100, 0
        std::vector<Real> stress_hist = {0.0, 100.0, 0.0, 100.0, 0.0};
        auto ranges = fatigue.count_cycles(stress_hist);
        CHECK(ranges.size() >= 2, "Fatigue: cycle counting produces ranges");
        // Each range should be ~100
        if (!ranges.empty()) {
            CHECK_NEAR(ranges[0], 100.0, 1.0e-10,
                       "Fatigue: stress range = 100 for 0-100 cycle");
        }
    }

    // 1j. Growth rate increases with delta_K
    {
        XFEMFatigueCrack::Params params;
        params.C_paris = 1.0e-10;
        params.m_paris = 3.0;
        params.threshold_K = 0.0;
        params.critical_K = 1.0e6;

        XFEMFatigueCrack fatigue(params);
        Real rate1 = fatigue.compute_growth_rate(5.0);
        Real rate2 = fatigue.compute_growth_rate(10.0);
        CHECK(rate2 > rate1, "Fatigue: growth rate increases with delta_K");
    }
}

// ============================================================================
// 2. XFEMMultiCrack Tests
// ============================================================================

void test_2_multi_crack() {
    std::cout << "\n--- Test 2: XFEMMultiCrack ---\n";

    // 2a. Add single crack
    {
        XFEMMultiCrack mc;
        Real tip[3] = {0.5, 0.0, 0.0};
        Real dir[3] = {1.0, 0.0, 0.0};
        int id = mc.add_crack(tip, dir, 0.01);
        CHECK(id == 0, "MultiCrack: first crack has ID 0");
        CHECK(mc.num_cracks() == 1, "MultiCrack: count = 1 after add");
    }

    // 2b. Add multiple cracks
    {
        XFEMMultiCrack mc;
        for (int i = 0; i < 5; ++i) {
            Real tip[3] = {i * 0.1, 0.0, 0.0};
            Real dir[3] = {1.0, 0.0, 0.0};
            mc.add_crack(tip, dir, 0.005);
        }
        CHECK(mc.num_cracks() == 5, "MultiCrack: count = 5 after 5 adds");
        CHECK(mc.num_active() == 5, "MultiCrack: all 5 active");
    }

    // 2c. Crack data retrieval
    {
        XFEMMultiCrack mc;
        Real tip[3] = {0.3, 0.4, 0.5};
        Real dir[3] = {0.0, 1.0, 0.0};
        int id = mc.add_crack(tip, dir, 0.015);

        const auto& c = mc.crack(id);
        CHECK_NEAR(c.tip[0], 0.3, 1e-15, "MultiCrack: retrieve tip[0]");
        CHECK_NEAR(c.tip[1], 0.4, 1e-15, "MultiCrack: retrieve tip[1]");
        CHECK_NEAR(c.length, 0.015, 1e-15, "MultiCrack: retrieve length");
        CHECK(c.active, "MultiCrack: new crack is active");
    }

    // 2d. update_all computes SIF with stress shielding
    {
        XFEMMultiCrack mc;
        Real tip[3] = {0.0, 0.0, 0.0};
        Real dir[3] = {1.0, 0.0, 0.0};
        int id = mc.add_crack(tip, dir, 0.02);

        Real stress[6] = {0.0, 100.0, 0.0, 0.0, 0.0, 0.0}; // sigma_yy = 100
        mc.update_all(stress);

        const auto& c = mc.crack(id);
        Real a = 0.02 * 0.5;
        Real expected_K = 100.0 * std::sqrt(M_PI * a);
        // Single crack: no shielding, SIF should match
        CHECK_NEAR(c.sif_I, expected_K, 0.01,
                   "MultiCrack: SIF_I for isolated crack");
    }

    // 2e. Stress shielding reduces SIF for nearby cracks
    {
        XFEMMultiCrack mc;
        // Two cracks very close together
        Real tip1[3] = {0.0, 0.0, 0.0};
        Real dir1[3] = {1.0, 0.0, 0.0};
        mc.add_crack(tip1, dir1, 0.02);

        Real tip2[3] = {0.005, 0.0, 0.0};
        Real dir2[3] = {1.0, 0.0, 0.0};
        mc.add_crack(tip2, dir2, 0.02);

        Real stress[6] = {0.0, 100.0, 0.0, 0.0, 0.0, 0.0};
        mc.update_all(stress);

        // With shielding, SIF should be less than unshielded value
        Real a = 0.02 * 0.5;
        Real unshielded_K = 100.0 * std::sqrt(M_PI * a);
        CHECK(mc.crack(0).sif_I < unshielded_K,
              "MultiCrack: stress shielding reduces SIF");
    }

    // 2f. Coalescence merges close cracks
    {
        XFEMMultiCrack mc;
        Real tip1[3] = {0.0, 0.0, 0.0};
        Real dir1[3] = {1.0, 0.0, 0.0};
        mc.add_crack(tip1, dir1, 0.01);

        Real tip2[3] = {0.003, 0.0, 0.0};
        Real dir2[3] = {-1.0, 0.0, 0.0};
        mc.add_crack(tip2, dir2, 0.01);

        int merges = mc.check_coalescence(0.005);
        CHECK(merges == 1, "MultiCrack: one coalescence event");
        CHECK(mc.num_active() == 1, "MultiCrack: one crack remains active after merge");
    }

    // 2g. No coalescence for distant cracks
    {
        XFEMMultiCrack mc;
        Real tip1[3] = {0.0, 0.0, 0.0};
        Real dir1[3] = {1.0, 0.0, 0.0};
        mc.add_crack(tip1, dir1, 0.01);

        Real tip2[3] = {10.0, 10.0, 0.0};
        Real dir2[3] = {-1.0, 0.0, 0.0};
        mc.add_crack(tip2, dir2, 0.01);

        int merges = mc.check_coalescence(0.005);
        CHECK(merges == 0, "MultiCrack: no coalescence for distant cracks");
        CHECK(mc.num_active() == 2, "MultiCrack: both cracks remain active");
    }

    // 2h. propagate_all advances all active cracks
    {
        XFEMMultiCrack mc;
        Real tip1[3] = {0.0, 0.0, 0.0};
        Real dir1[3] = {1.0, 0.0, 0.0};
        mc.add_crack(tip1, dir1, 0.01);

        // Set SIF so crack will propagate
        Real stress[6] = {0.0, 100.0, 0.0, 0.0, 0.0, 0.0};
        mc.update_all(stress);

        Real len_before = mc.crack(0).length;
        mc.propagate_all(0.005);
        Real len_after = mc.crack(0).length;

        CHECK_NEAR(len_after, len_before + 0.005, 1.0e-10,
                   "MultiCrack: propagate_all increases length by da");
    }

    // 2i. Propagation advances tip position
    {
        XFEMMultiCrack mc;
        Real tip[3] = {0.0, 0.0, 0.0};
        Real dir[3] = {1.0, 0.0, 0.0};
        mc.add_crack(tip, dir, 0.01);

        // Give it a positive Mode I SIF so it propagates
        Real stress[6] = {0.0, 100.0, 0.0, 0.0, 0.0, 0.0};
        mc.update_all(stress);

        Real tip_x_before = mc.crack(0).tip[0];
        mc.propagate_all(0.005);
        Real tip_x_after = mc.crack(0).tip[0];

        CHECK(tip_x_after > tip_x_before,
              "MultiCrack: tip position advances in direction");
    }

    // 2j. Inactive cracks not propagated
    {
        XFEMMultiCrack mc;
        Real tip1[3] = {0.0, 0.0, 0.0};
        Real dir1[3] = {1.0, 0.0, 0.0};
        mc.add_crack(tip1, dir1, 0.01);

        Real tip2[3] = {0.001, 0.0, 0.0};
        Real dir2[3] = {-1.0, 0.0, 0.0};
        mc.add_crack(tip2, dir2, 0.01);

        // Coalesce to deactivate crack 1
        mc.check_coalescence(0.01);

        Real stress[6] = {0.0, 100.0, 0.0, 0.0, 0.0, 0.0};
        mc.update_all(stress);

        // Only active crack should propagate
        Real len_inactive = mc.crack(1).length;
        mc.propagate_all(0.005);
        CHECK_NEAR(mc.crack(1).length, len_inactive, 1.0e-15,
                   "MultiCrack: inactive crack not propagated");
    }
}

// ============================================================================
// 3. XFEMAdaptiveMesh Tests
// ============================================================================

void test_3_adaptive_mesh() {
    std::cout << "\n--- Test 3: XFEMAdaptiveMesh ---\n";

    // Helper: create a simple 4-element mesh
    auto make_elements = [](int n, Real size) {
        std::vector<XFEMAdaptiveMesh::ElementData> elems(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            elems[static_cast<size_t>(i)].id = i;
            elems[static_cast<size_t>(i)].centroid[0] = i * size;
            elems[static_cast<size_t>(i)].centroid[1] = 0.0;
            elems[static_cast<size_t>(i)].centroid[2] = 0.0;
            elems[static_cast<size_t>(i)].size = size;
            elems[static_cast<size_t>(i)].level = 0;
        }
        return elems;
    };

    // 3a. Uniform stress -> low error
    {
        XFEMAdaptiveMesh amesh;
        auto elems = make_elements(4, 0.1);
        std::vector<std::array<Real,6>> stresses(4, {100.0, 0.0, 0.0, 0.0, 0.0, 0.0});

        auto errors = amesh.compute_error(elems, stresses);
        CHECK(errors.size() == 4, "AdaptiveMesh: error vector size matches elements");
        Real max_err = *std::max_element(errors.begin(), errors.end());
        CHECK(max_err < 0.1, "AdaptiveMesh: uniform stress -> low error");
    }

    // 3b. Non-uniform stress -> higher error for outlier
    {
        XFEMAdaptiveMesh amesh;
        auto elems = make_elements(4, 0.1);
        std::vector<std::array<Real,6>> stresses = {
            {100.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {100.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {100.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {10000.0, 0.0, 0.0, 0.0, 0.0, 0.0}
        };

        auto errors = amesh.compute_error(elems, stresses);
        // The outlier element should have the largest error
        CHECK(errors[3] > errors[0],
              "AdaptiveMesh: outlier stress element has higher error");
    }

    // 3c. mark_refine: elements above threshold marked for refinement
    {
        XFEMAdaptiveMesh amesh;
        std::vector<Real> errors = {0.01, 0.5, 0.8, 0.02};
        amesh.mark_refine(errors, 0.1, 0.05);

        CHECK(amesh.refine_list().size() == 2,
              "AdaptiveMesh: 2 elements marked for refinement");
    }

    // 3d. mark_refine: elements below coarsen threshold marked for coarsening
    {
        XFEMAdaptiveMesh amesh;
        std::vector<Real> errors = {0.001, 0.5, 0.8, 0.002};
        amesh.mark_refine(errors, 0.1, 0.01);

        CHECK(amesh.coarsen_list().size() == 2,
              "AdaptiveMesh: 2 elements marked for coarsening");
    }

    // 3e. Mid-range error: neither refined nor coarsened
    {
        XFEMAdaptiveMesh amesh;
        std::vector<Real> errors = {0.05};
        amesh.mark_refine(errors, 0.1, 0.01);
        CHECK(amesh.refine_list().empty(), "AdaptiveMesh: mid-range -> not refined");
        CHECK(amesh.coarsen_list().empty(), "AdaptiveMesh: mid-range -> not coarsened");
    }

    // 3f. refine creates 4 child elements per marked element
    {
        XFEMAdaptiveMesh amesh;
        auto elems = make_elements(4, 0.1);
        std::vector<Real> errors = {0.5, 0.01, 0.01, 0.01};
        amesh.mark_refine(errors, 0.1);

        auto info = amesh.refine(elems);
        CHECK(info.refined_elements.size() == 1, "AdaptiveMesh: 1 element refined");
        CHECK(info.new_elements == 4, "AdaptiveMesh: 4 new child elements created");
        CHECK(elems.size() == 8, "AdaptiveMesh: total elements now 8 (4 orig + 4 new)");
    }

    // 3g. Refined children have smaller size and higher level
    {
        XFEMAdaptiveMesh amesh;
        auto elems = make_elements(1, 1.0);
        std::vector<Real> errors = {0.5};
        amesh.mark_refine(errors, 0.1);

        amesh.refine(elems);
        // Check child elements (indices 1-4)
        CHECK(elems.size() == 5, "AdaptiveMesh: 1 + 4 = 5 elements after refine");
        CHECK_NEAR(elems[1].size, 0.5, 1.0e-10,
                   "AdaptiveMesh: child element has half the size");
        CHECK(elems[1].level == 1, "AdaptiveMesh: child element at level 1");
    }

    // 3h. Max level limits refinement
    {
        XFEMAdaptiveMesh amesh;
        auto elems = make_elements(1, 1.0);
        elems[0].level = 5; // already at max level
        std::vector<Real> errors = {0.5};
        amesh.mark_refine(errors, 0.1);

        auto info = amesh.refine(elems, 5);
        CHECK(info.refined_elements.empty(),
              "AdaptiveMesh: no refinement at max level");
        CHECK(elems.size() == 1, "AdaptiveMesh: element count unchanged at max level");
    }

    // 3i. Errors are non-negative
    {
        XFEMAdaptiveMesh amesh;
        auto elems = make_elements(3, 0.1);
        std::vector<std::array<Real,6>> stresses = {
            {-100.0, 50.0, 0.0, 10.0, 0.0, 0.0},
            {200.0, -100.0, 0.0, -20.0, 0.0, 0.0},
            {-50.0, 300.0, 0.0, 15.0, 0.0, 0.0}
        };

        auto errors = amesh.compute_error(elems, stresses);
        CHECK(errors[0] >= 0.0 && errors[1] >= 0.0 && errors[2] >= 0.0,
              "AdaptiveMesh: errors are non-negative");
    }

    // 3j. New nodes generated during refinement
    {
        XFEMAdaptiveMesh amesh;
        auto elems = make_elements(2, 0.5);
        std::vector<Real> errors = {0.8, 0.9};
        amesh.mark_refine(errors, 0.1);

        auto info = amesh.refine(elems);
        CHECK(info.new_nodes > 0, "AdaptiveMesh: refinement generates new nodes");
        CHECK(info.new_nodes == 10, "AdaptiveMesh: 5 new nodes per refined element");
    }
}

// ============================================================================
// 4. XFEMOutputFields Tests
// ============================================================================

void test_4_output_fields() {
    std::cout << "\n--- Test 4: XFEMOutputFields ---\n";

    // Helper: build a simple 2x1 quad mesh with level set for a horizontal crack
    // Nodes:  0(0,0) - 1(1,0) - 2(2,0)
    //         3(0,1) - 4(1,1) - 5(2,1)
    // Elements: [0,1,4,3], [1,2,5,4]

    // 4a. Crack path extraction: straight horizontal crack (phi = y - 0.5)
    {
        XFEMOutputFields output;
        // 6 nodes, phi = y - 0.5
        std::vector<Real> level_set = {-0.5, -0.5, -0.5, 0.5, 0.5, 0.5};
        std::vector<Real> node_coords = {
            0.0, 0.0, 0.0,   1.0, 0.0, 0.0,   2.0, 0.0, 0.0,
            0.0, 1.0, 0.0,   1.0, 1.0, 0.0,   2.0, 1.0, 0.0
        };
        std::vector<int> connectivity = {0,1,4,3, 1,2,5,4};
        int n_elements = 2;

        auto path = output.extract_crack_path(level_set, node_coords, connectivity, n_elements);
        CHECK(path.size() >= 2, "Output: crack path has at least 2 points");
    }

    // 4b. No crack path when all level-set values are positive
    {
        XFEMOutputFields output;
        std::vector<Real> level_set = {1.0, 2.0, 3.0, 4.0};
        std::vector<Real> node_coords = {
            0.0, 0.0, 0.0,   1.0, 0.0, 0.0,
            1.0, 1.0, 0.0,   0.0, 1.0, 0.0
        };
        std::vector<int> connectivity = {0, 1, 2, 3};

        auto path = output.extract_crack_path(level_set, node_coords, connectivity, 1);
        CHECK(path.empty(), "Output: no crack path when no zero-crossing");
    }

    // 4c. Vertical crack path (phi = x - 0.5)
    {
        XFEMOutputFields output;
        std::vector<Real> level_set = {-0.5, 0.5, 0.5, -0.5};
        std::vector<Real> node_coords = {
            0.0, 0.0, 0.0,   1.0, 0.0, 0.0,
            1.0, 1.0, 0.0,   0.0, 1.0, 0.0
        };
        std::vector<int> connectivity = {0, 1, 2, 3};

        auto path = output.extract_crack_path(level_set, node_coords, connectivity, 1);
        CHECK(path.size() >= 2, "Output: vertical crack path extracted");
        if (!path.empty()) {
            CHECK_NEAR(path[0].x, 0.5, 0.01,
                       "Output: vertical crack at x = 0.5");
        }
    }

    // 4d. COD computation: zero displacement -> zero COD
    {
        XFEMOutputFields output;
        XFEMOutputFields::Point3 p0{0.5, 0.5, 0.0};
        std::vector<XFEMOutputFields::Point3> path_pts = {p0};
        std::vector<Real> displacements = {0.0, 0.0, 0.0};
        std::vector<Real> node_coords = {0.5, 0.5, 0.0};

        auto cod = output.compute_cod(path_pts, displacements, node_coords, 1);
        CHECK(cod.size() == 1, "Output: COD vector matches path points");
        CHECK_NEAR(cod[0], 0.0, 1.0e-15, "Output: COD = 0 for zero displacement");
    }

    // 4e. COD computation: nonzero normal displacement
    {
        XFEMOutputFields output;
        XFEMOutputFields::Point3 p0{0.5, 0.5, 0.0};
        std::vector<XFEMOutputFields::Point3> path_pts = {p0};
        // Node at (0.5, 0.5, 0) with displacement (0, 0.001, 0)
        std::vector<Real> displacements = {0.0, 0.001, 0.0};
        std::vector<Real> node_coords = {0.5, 0.5, 0.0};
        Real normal[3] = {0.0, 1.0, 0.0};

        auto cod = output.compute_cod(path_pts, displacements, node_coords, 1, normal);
        // COD = 2 * |u_n| = 2 * 0.001 = 0.002
        CHECK_NEAR(cod[0], 0.002, 1.0e-10,
                   "Output: COD = 2 * normal displacement");
    }

    // 4f. COD with default normal (y-direction)
    {
        XFEMOutputFields output;
        XFEMOutputFields::Point3 p0{0.0, 0.0, 0.0};
        std::vector<XFEMOutputFields::Point3> path_pts = {p0};
        std::vector<Real> displacements = {0.0, 0.005, 0.0};
        std::vector<Real> node_coords = {0.0, 0.0, 0.0};

        auto cod = output.compute_cod(path_pts, displacements, node_coords, 1);
        CHECK_NEAR(cod[0], 0.01, 1.0e-10,
                   "Output: COD with default normal = 2*uy");
    }

    // 4g. VTK output writes successfully
    {
        XFEMOutputFields output;
        XFEMOutputFields::CrackPathData cpd;
        cpd.path_points = {{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}, {1.0, 0.0, 0.0}};
        cpd.sif_along_front = {10.0, 15.0, 10.0};
        cpd.cod_values = {0.001, 0.002, 0.001};

        std::string filename = "/tmp/xfem_wave41_test_crack.vtk";
        bool ok = output.write_vtk(filename, cpd);
        CHECK(ok, "Output: VTK write returns true");

        // Verify file was created and has content
        std::ifstream fin(filename);
        CHECK(fin.good(), "Output: VTK file exists and is readable");
        std::string first_line;
        std::getline(fin, first_line);
        CHECK(first_line.find("vtk") != std::string::npos,
              "Output: VTK file has correct header");
        fin.close();
        std::remove(filename.c_str());
    }

    // 4h. VTK output with empty path
    {
        XFEMOutputFields output;
        XFEMOutputFields::CrackPathData cpd; // empty
        std::string filename = "/tmp/xfem_wave41_test_empty.vtk";
        bool ok = output.write_vtk(filename, cpd);
        CHECK(ok, "Output: VTK write succeeds with empty path");
        std::remove(filename.c_str());
    }

    // 4i. Multiple path points in COD computation
    {
        XFEMOutputFields output;
        std::vector<XFEMOutputFields::Point3> path_pts = {
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}
        };
        // 3 nodes with y-displacements
        std::vector<Real> displacements = {
            0.0, 0.001, 0.0,
            0.0, 0.002, 0.0,
            0.0, 0.003, 0.0
        };
        std::vector<Real> node_coords = {
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            2.0, 0.0, 0.0
        };

        auto cod = output.compute_cod(path_pts, displacements, node_coords, 3);
        CHECK(cod.size() == 3, "Output: COD vector has 3 values");
        CHECK(cod[2] > cod[0], "Output: COD increases with displacement");
    }

    // 4j. CrackPathData struct fields
    {
        XFEMOutputFields::CrackPathData cpd;
        cpd.path_points.push_back({1.0, 2.0, 3.0});
        cpd.sif_along_front.push_back(42.0);
        cpd.cod_values.push_back(0.01);

        CHECK(cpd.path_points.size() == 1, "Output: CrackPathData stores path points");
        CHECK_NEAR(cpd.path_points[0].x, 1.0, 1e-15, "Output: Point3 x access");
        CHECK_NEAR(cpd.sif_along_front[0], 42.0, 1e-15, "Output: SIF stored correctly");
        CHECK_NEAR(cpd.cod_values[0], 0.01, 1e-15, "Output: COD stored correctly");
    }
}

// ============================================================================
// main
// ============================================================================

int main() {
    test_1_fatigue_crack();
    test_2_multi_crack();
    test_3_adaptive_mesh();
    test_4_output_fields();

    std::cout << "\n" << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    return tests_failed > 0 ? 1 : 0;
}
