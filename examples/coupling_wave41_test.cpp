/**
 * @file coupling_wave41_test.cpp
 * @brief Wave 41: Coupling & Acoustics Extensions Test Suite (4 features, 40 tests)
 *
 * Tests:
 *   1. CouplingSubIteration    (10 tests)
 *   2. CouplingFieldSmoothing  (10 tests)
 *   3. AcousticFMM             (10 tests)
 *   4. AcousticStructuralModes (10 tests)
 */

#include <nexussim/coupling/coupling_wave41.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <functional>

using namespace nxs::coupling;

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
// 1. CouplingSubIteration Tests
// ============================================================================

void test_1_sub_iteration() {
    std::cout << "--- Test 1: CouplingSubIteration ---\n";

    // 1a. Converges for simple linear problem using iterate_simple
    //     Solver: d_new = 0.5 * d + 1.5  => fixed point at d=3
    {
        CouplingSubIteration::Params p;
        p.omega_init = 0.5;
        p.max_sub_iter = 100;
        p.tolerance = 1.0e-8;
        CouplingSubIteration sub(p);

        auto solver = [](const std::vector<Real>& d) -> std::vector<Real> {
            std::vector<Real> out(d.size());
            for (size_t i = 0; i < d.size(); ++i)
                out[i] = 0.5 * d[i] + 1.5;
            return out;
        };

        std::vector<Real> init = {0.0};
        auto result = sub.iterate_simple(solver, init);
        CHECK(result.converged, "SubIter: converges for linear problem");
    }

    // 1b. Result contains correct final_omega
    {
        CouplingSubIteration::Params p;
        p.omega_init = 0.7;
        p.max_sub_iter = 50;
        p.tolerance = 1.0e-8;
        CouplingSubIteration sub(p);

        auto solver = [](const std::vector<Real>& d) -> std::vector<Real> {
            return {0.5 * d[0] + 2.5};
        };

        auto result = sub.iterate_simple(solver, {0.0});
        CHECK(result.final_omega > 0.0, "SubIter: final_omega > 0");
    }

    // 1c. Non-convergence detection with very few iterations
    {
        CouplingSubIteration::Params p;
        p.max_sub_iter = 3;
        p.tolerance = 1.0e-15;
        p.omega_init = 0.5;
        CouplingSubIteration sub(p);

        // Slow convergence, tight tol, few iters -> not converged
        auto solver = [](const std::vector<Real>& d) -> std::vector<Real> {
            return {0.99 * d[0] + 100.0};
        };

        auto result = sub.iterate_simple(solver, {0.0});
        CHECK(!result.converged, "SubIter: detects non-convergence");
    }

    // 1d. Max iterations respected
    {
        CouplingSubIteration::Params p;
        p.max_sub_iter = 7;
        p.tolerance = 1.0e-15;
        p.omega_init = 0.5;
        CouplingSubIteration sub(p);

        auto solver = [](const std::vector<Real>& d) -> std::vector<Real> {
            return {0.5 * d[0] + 1.5};
        };

        auto result = sub.iterate_simple(solver, {0.0});
        CHECK(result.iterations <= 7, "SubIter: max iterations respected");
    }

    // 1e. Already converged input (solver returns same value)
    {
        CouplingSubIteration::Params p;
        p.max_sub_iter = 100;
        p.tolerance = 1.0e-6;
        p.omega_init = 0.5;
        CouplingSubIteration sub(p);

        // Identity solver at the fixed point
        auto solver = [](const std::vector<Real>& d) -> std::vector<Real> {
            return d;
        };

        auto result = sub.iterate_simple(solver, {5.0});
        CHECK(result.converged, "SubIter: identity solver converges immediately");
        CHECK(result.iterations <= 1, "SubIter: identity takes 1 iteration");
    }

    // 1f. Full iterate with fluid_solve, struct_solve, transfer
    {
        CouplingSubIteration::Params p;
        p.max_sub_iter = 50;
        p.tolerance = 1.0e-6;
        p.omega_init = 0.5;
        CouplingSubIteration sub(p);

        // fluid_solve: displacement -> force = 2*d
        auto fluid = [](const std::vector<Real>& d) -> std::vector<Real> {
            return {2.0 * d[0]};
        };
        // transfer: identity
        auto transfer = [](const std::vector<Real>& f) -> std::vector<Real> {
            return f;
        };
        // struct_solve: force -> displacement = force/4 + 1
        // fixed point: d = 2d/4 + 1 = d/2 + 1 => d = 2
        auto structure = [](const std::vector<Real>& f) -> std::vector<Real> {
            return {f[0] / 4.0 + 1.0};
        };

        auto result = sub.iterate(fluid, structure, transfer, {0.0});
        CHECK(result.converged, "SubIter: full iterate converges");
    }

    // 1g. Multi-DOF problem
    {
        CouplingSubIteration::Params p;
        p.max_sub_iter = 100;
        p.tolerance = 1.0e-6;
        p.omega_init = 0.5;
        CouplingSubIteration sub(p);

        // 2-DOF: d_new = 0.3*d + {1.4, 2.8} => fixed point {2, 4}
        auto solver = [](const std::vector<Real>& d) -> std::vector<Real> {
            return {0.3 * d[0] + 1.4, 0.3 * d[1] + 2.8};
        };

        auto result = sub.iterate_simple(solver, {0.0, 0.0});
        CHECK(result.converged, "SubIter: multi-DOF converges");
    }

    // 1h. Params accessors
    {
        CouplingSubIteration sub;
        sub.params().omega_init = 0.3;
        sub.params().max_sub_iter = 25;
        sub.params().tolerance = 1.0e-4;

        CHECK_NEAR(sub.params().omega_init, 0.3, 1e-15, "SubIter: omega_init set");
        CHECK(sub.params().max_sub_iter == 25, "SubIter: max_sub_iter set");
    }

    // 1i. omega clamped to [omega_min, omega_max]
    {
        CouplingSubIteration::Params p;
        p.omega_init = 0.5;
        p.omega_min = 0.1;
        p.omega_max = 0.9;
        p.max_sub_iter = 50;
        p.tolerance = 1.0e-6;
        CouplingSubIteration sub(p);

        auto solver = [](const std::vector<Real>& d) -> std::vector<Real> {
            return {0.5 * d[0] + 1.5};
        };

        auto result = sub.iterate_simple(solver, {0.0});
        CHECK(result.final_omega >= p.omega_min - 1e-15,
              "SubIter: omega >= omega_min");
        CHECK(result.final_omega <= p.omega_max + 1e-15,
              "SubIter: omega <= omega_max");
    }

    // 1j. Empty initial displacement returns immediately
    {
        CouplingSubIteration sub;
        auto solver = [](const std::vector<Real>& d) -> std::vector<Real> { return d; };
        auto result = sub.iterate_simple(solver, {});
        CHECK(result.iterations == 0, "SubIter: empty input -> 0 iterations");
    }
}

// ============================================================================
// 2. CouplingFieldSmoothing Tests
// ============================================================================

void test_2_field_smoothing() {
    std::cout << "\n--- Test 2: CouplingFieldSmoothing ---\n";

    // Build a simple 1D chain connectivity: 0-1-2-3-4
    auto make_chain_conn = [](int n) -> std::vector<std::vector<int>> {
        std::vector<std::vector<int>> conn(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            if (i > 0) conn[static_cast<size_t>(i)].push_back(i - 1);
            if (i < n - 1) conn[static_cast<size_t>(i)].push_back(i + 1);
        }
        return conn;
    };

    CouplingFieldSmoothing smoother;

    // 2a. Constant field unchanged
    {
        std::vector<Real> field = {5.0, 5.0, 5.0, 5.0, 5.0};
        auto conn = make_chain_conn(5);
        smoother.smooth(field, conn, 1, 0.5);
        for (int i = 0; i < 5; ++i) {
            CHECK_NEAR(field[static_cast<size_t>(i)], 5.0, 1e-10,
                       "Smoothing: constant field unchanged");
        }
    }

    // 2b. Spike field smoothed
    {
        std::vector<Real> field = {0.0, 0.0, 10.0, 0.0, 0.0};
        auto conn = make_chain_conn(5);
        smoother.smooth(field, conn, 1, 0.5);
        CHECK(field[2] < 10.0, "Smoothing: spike reduced");
        CHECK(field[1] > 0.0 || field[3] > 0.0,
              "Smoothing: neighbors gain from spike");
    }

    // 2c. Multiple passes increase smoothness
    {
        std::vector<Real> field1 = {0.0, 0.0, 10.0, 0.0, 0.0};
        std::vector<Real> field5 = {0.0, 0.0, 10.0, 0.0, 0.0};
        auto conn = make_chain_conn(5);

        smoother.smooth(field1, conn, 1, 0.5);
        smoother.smooth(field5, conn, 5, 0.5);

        CHECK(field5[2] < field1[2],
              "Smoothing: more passes reduce peak further");
    }

    // 2d. Alpha=0 -> no change
    {
        std::vector<Real> field = {1.0, 5.0, 2.0, 8.0, 3.0};
        std::vector<Real> original = field;
        auto conn = make_chain_conn(5);
        smoother.smooth(field, conn, 10, 0.0);
        for (int i = 0; i < 5; ++i) {
            CHECK_NEAR(field[static_cast<size_t>(i)],
                       original[static_cast<size_t>(i)], 1e-15,
                       "Smoothing: alpha=0 -> no change");
        }
    }

    // 2e. Symmetric input -> symmetric output
    {
        std::vector<Real> field = {1.0, 3.0, 5.0, 3.0, 1.0};
        auto conn = make_chain_conn(5);
        smoother.smooth(field, conn, 2, 0.5);
        CHECK_NEAR(field[0], field[4], 1e-10,
                   "Smoothing: symmetry preserved (endpoints)");
        CHECK_NEAR(field[1], field[3], 1e-10,
                   "Smoothing: symmetry preserved (interior)");
    }

    // 2f. Alpha=1 -> full averaging (interior replaces with neighbor mean)
    {
        std::vector<Real> field = {0.0, 0.0, 10.0, 0.0, 0.0};
        auto conn = make_chain_conn(5);
        smoother.smooth(field, conn, 1, 1.0);
        // Interior node 2 had neighbors 1,3 both 0 -> mean=0 -> fully replaced
        CHECK_NEAR(field[2], 0.0, 0.01,
                   "Smoothing: alpha=1 fully replaces with neighbor average");
    }

    // 2g. Single element field unchanged
    {
        std::vector<Real> field = {42.0};
        std::vector<std::vector<int>> conn = {{}};
        smoother.smooth(field, conn, 5, 0.5);
        CHECK_NEAR(field[0], 42.0, 1e-15,
                   "Smoothing: single element unchanged");
    }

    // 2h. Two-element blend
    {
        std::vector<Real> field = {0.0, 10.0};
        std::vector<std::vector<int>> conn = {{1}, {0}};
        smoother.smooth(field, conn, 1, 0.5);
        CHECK_NEAR(field[0], 5.0, 0.01, "Smoothing: 2-element blend [0]");
        CHECK_NEAR(field[1], 5.0, 0.01, "Smoothing: 2-element blend [1]");
    }

    // 2i. build_connectivity creates correct adjacency from triangles
    {
        // Triangle: nodes 0,1,2
        std::vector<int> elements = {0, 1, 2};
        auto conn = CouplingFieldSmoothing::build_connectivity(elements, 1, 3);
        CHECK(conn.size() == 3, "Smoothing: build_connectivity returns 3 entries");
        // Node 0 should have neighbors 1 and 2
        bool has_1 = false, has_2 = false;
        for (int nbr : conn[0]) {
            if (nbr == 1) has_1 = true;
            if (nbr == 2) has_2 = true;
        }
        CHECK(has_1 && has_2, "Smoothing: node 0 neighbors are 1 and 2");
    }

    // 2j. smooth_vector works per-component
    {
        // 3 nodes in a chain, vector field (3 components each)
        std::vector<Real> field = {
            0.0, 0.0, 0.0,   // node 0
            10.0, 20.0, 30.0, // node 1 (spike)
            0.0, 0.0, 0.0    // node 2
        };
        std::vector<std::vector<int>> conn = {{1}, {0, 2}, {1}};
        smoother.smooth_vector(field, conn, 1, 0.5);
        // Node 1's x component should be reduced from 10
        CHECK(field[3] < 10.0, "Smoothing: smooth_vector reduces spike x");
        CHECK(field[4] < 20.0, "Smoothing: smooth_vector reduces spike y");
        CHECK(field[5] < 30.0, "Smoothing: smooth_vector reduces spike z");
    }
}

// ============================================================================
// 3. AcousticFMM Tests
// ============================================================================

void test_3_acoustic_fmm() {
    std::cout << "\n--- Test 3: AcousticFMM ---\n";

    // 3a. Tree construction
    {
        AcousticFMM fmm;
        std::vector<Real> nodes = {
            0,0,0, 1,0,0, 0,1,0, 1,1,0,
            0,0,1, 1,0,1, 0,1,1, 1,1,1
        };
        fmm.build_tree(nodes, 8, 4);
        CHECK(fmm.num_nodes() >= 1, "FMM: tree built with >= 1 node");
    }

    // 3b. Single source, k=0 (static Green's function)
    {
        AcousticFMM fmm;
        std::vector<Real> src_pos = {0.0, 0.0, 0.0};
        // Source strengths: complex [re, im]
        std::vector<Real> src_str = {1.0, 0.0};
        std::vector<Real> tgt_pos = {1.0, 0.0, 0.0};

        fmm.build_tree(src_pos, 1);

        auto result = fmm.compute_pressure(src_pos, src_str, tgt_pos, 1, 1, 0.0);
        // G(r) = 1/(4*pi*r), k=0 => cos(0)=1, sin(0)=0
        Real expected = 1.0 / (4.0 * M_PI * 1.0);
        CHECK_NEAR(result.pressure_real[0], expected, expected * 0.01,
                   "FMM: single source static pressure");
        CHECK_NEAR(result.pressure_imag[0], 0.0, 1e-10,
                   "FMM: imaginary part zero for k=0");
    }

    // 3c. Pressure decay with distance (1/r)
    {
        AcousticFMM fmm;
        std::vector<Real> src_pos = {0.0, 0.0, 0.0};
        std::vector<Real> src_str = {1.0, 0.0};

        fmm.build_tree(src_pos, 1);

        std::vector<Real> t1 = {1.0, 0.0, 0.0};
        std::vector<Real> t2 = {2.0, 0.0, 0.0};

        auto r1 = fmm.compute_pressure(src_pos, src_str, t1, 1, 1, 0.0);
        auto r2 = fmm.compute_pressure(src_pos, src_str, t2, 1, 1, 0.0);

        Real ratio = r1.pressure_real[0] / r2.pressure_real[0];
        CHECK_NEAR(ratio, 2.0, 0.1, "FMM: 1/r decay verified");
    }

    // 3d. Zero strength -> zero pressure
    {
        AcousticFMM fmm;
        std::vector<Real> src_pos = {0.0, 0.0, 0.0};
        std::vector<Real> src_str = {0.0, 0.0};
        std::vector<Real> tgt = {1.0, 0.0, 0.0};

        fmm.build_tree(src_pos, 1);
        auto result = fmm.compute_pressure(src_pos, src_str, tgt, 1, 1, 0.0);
        CHECK_NEAR(result.pressure_real[0], 0.0, 1e-15,
                   "FMM: zero source -> zero pressure (real)");
        CHECK_NEAR(result.pressure_imag[0], 0.0, 1e-15,
                   "FMM: zero source -> zero pressure (imag)");
    }

    // 3e. Symmetry: equal-distance targets get equal pressure
    {
        AcousticFMM fmm;
        std::vector<Real> src_pos = {0.0, 0.0, 0.0};
        std::vector<Real> src_str = {1.0, 0.0};

        fmm.build_tree(src_pos, 1);

        std::vector<Real> tx = {1.0, 0.0, 0.0};
        std::vector<Real> ty = {0.0, 1.0, 0.0};
        std::vector<Real> tz = {0.0, 0.0, 1.0};

        auto rx = fmm.compute_pressure(src_pos, src_str, tx, 1, 1, 0.0);
        auto ry = fmm.compute_pressure(src_pos, src_str, ty, 1, 1, 0.0);
        auto rz = fmm.compute_pressure(src_pos, src_str, tz, 1, 1, 0.0);

        CHECK_NEAR(rx.pressure_real[0], ry.pressure_real[0],
                   std::abs(rx.pressure_real[0]) * 0.01,
                   "FMM: symmetry x vs y");
        CHECK_NEAR(rx.pressure_real[0], rz.pressure_real[0],
                   std::abs(rx.pressure_real[0]) * 0.01,
                   "FMM: symmetry x vs z");
    }

    // 3f. Two sources superpose
    {
        AcousticFMM fmm;
        std::vector<Real> src_pos = {-1,0,0, 1,0,0};
        std::vector<Real> src_str = {1.0, 0.0, 1.0, 0.0};  // two real sources
        std::vector<Real> tgt = {0.0, 10.0, 0.0};  // far away, equal distance

        fmm.build_tree(src_pos, 2);
        auto result = fmm.compute_pressure(src_pos, src_str, tgt, 2, 1, 0.0);

        // Both at distance sqrt(1+100) ~ 10.05
        Real r = std::sqrt(101.0);
        Real expected = 2.0 / (4.0 * M_PI * r);
        CHECK_NEAR(result.pressure_real[0], expected, expected * 0.05,
                   "FMM: two sources superpose");
    }

    // 3g. Negative source -> negative pressure (k=0)
    {
        AcousticFMM fmm;
        std::vector<Real> src_pos = {0,0,0};
        std::vector<Real> src_str = {-1.0, 0.0};
        std::vector<Real> tgt = {1,0,0};

        fmm.build_tree(src_pos, 1);
        auto result = fmm.compute_pressure(src_pos, src_str, tgt, 1, 1, 0.0);
        CHECK(result.pressure_real[0] < 0.0,
              "FMM: negative source -> negative pressure");
    }

    // 3h. Non-zero wavenumber produces imaginary component
    {
        AcousticFMM fmm;
        std::vector<Real> src_pos = {0,0,0};
        std::vector<Real> src_str = {1.0, 0.0};
        std::vector<Real> tgt = {1,0,0};

        fmm.build_tree(src_pos, 1);
        Real k = 2.0;  // wavenumber
        auto result = fmm.compute_pressure(src_pos, src_str, tgt, 1, 1, k);
        // sin(k*r) = sin(2) != 0
        CHECK(std::abs(result.pressure_imag[0]) > 1e-10,
              "FMM: non-zero k produces imaginary part");
    }

    // 3i. Multiple targets
    {
        AcousticFMM fmm;
        std::vector<Real> src_pos = {0,0,0};
        std::vector<Real> src_str = {1.0, 0.0};
        std::vector<Real> tgt = {1,0,0, 2,0,0, 3,0,0, 4,0,0};

        fmm.build_tree(src_pos, 1);
        auto result = fmm.compute_pressure(src_pos, src_str, tgt, 1, 4, 0.0);

        CHECK(result.pressure_real.size() == 4, "FMM: 4 target pressures returned");
        CHECK(result.pressure_real[0] > result.pressure_real[1],
              "FMM: batch p[0] > p[1]");
        CHECK(result.pressure_real[1] > result.pressure_real[2],
              "FMM: batch p[1] > p[2]");
        CHECK(result.pressure_real[2] > result.pressure_real[3],
              "FMM: batch p[2] > p[3]");
    }

    // 3j. Empty sources -> empty result
    {
        AcousticFMM fmm;
        std::vector<Real> tgt = {1,0,0};
        auto result = fmm.compute_pressure({}, {}, tgt, 0, 1, 0.0);
        CHECK(result.pressure_real.size() == 1, "FMM: result has 1 entry for 1 target");
        CHECK_NEAR(result.pressure_real[0], 0.0, 1e-15,
                   "FMM: no sources -> zero pressure");
    }
}

// ============================================================================
// 4. AcousticStructuralModes Tests
// ============================================================================

void test_4_structural_modes() {
    std::cout << "\n--- Test 4: AcousticStructuralModes ---\n";

    AcousticStructuralModes asm_solver;

    // Helper to make ModeData
    auto make_mode = [](Real freq, Real mass) -> AcousticStructuralModes::ModeData {
        AcousticStructuralModes::ModeData m;
        m.frequency = freq;
        m.modal_mass = mass;
        m.modal_stiffness = std::pow(2.0 * M_PI * freq, 2) * mass;
        return m;
    };

    // 4a. Uncoupled modes (zero coupling): frequencies unchanged
    {
        std::vector<AcousticStructuralModes::ModeData> s_modes = {
            make_mode(100.0, 1.0), make_mode(200.0, 1.0)
        };
        std::vector<AcousticStructuralModes::ModeData> a_modes = {
            make_mode(150.0, 1.0), make_mode(250.0, 1.0)
        };
        std::vector<Real> C = {0, 0, 0, 0};  // 2x2 zero coupling

        auto result = asm_solver.couple(s_modes, a_modes, C);
        CHECK(result.size() == 4, "StructModes: 4 coupled modes");
        CHECK_NEAR(result[0].frequency, 100.0, 1.0,
                   "StructModes: uncoupled freq[0] = 100");
        CHECK_NEAR(result[1].frequency, 150.0, 1.0,
                   "StructModes: uncoupled freq[1] = 150");
        CHECK_NEAR(result[2].frequency, 200.0, 1.0,
                   "StructModes: uncoupled freq[2] = 200");
        CHECK_NEAR(result[3].frequency, 250.0, 1.0,
                   "StructModes: uncoupled freq[3] = 250");
    }

    // 4b. Coupled frequency shift (non-zero coupling splits frequencies)
    {
        std::vector<AcousticStructuralModes::ModeData> s_modes = {
            make_mode(100.0, 1.0)
        };
        std::vector<AcousticStructuralModes::ModeData> a_modes = {
            make_mode(200.0, 1.0)
        };
        std::vector<Real> C = {5000.0};  // strong coupling

        auto result = asm_solver.couple(s_modes, a_modes, C);
        CHECK(result.size() == 2, "StructModes: 2 coupled modes");
        // Perturbation shifts: structural goes down (repelled from acoustic above)
        // and acoustic goes up
        CHECK(result[0].frequency_shift != 0.0,
              "StructModes: coupling causes frequency shift");
    }

    // 4c. Mode ordering (ascending frequency)
    {
        std::vector<AcousticStructuralModes::ModeData> s_modes = {
            make_mode(300.0, 1.0), make_mode(100.0, 1.0)
        };
        std::vector<AcousticStructuralModes::ModeData> a_modes = {
            make_mode(200.0, 1.0)
        };
        std::vector<Real> C = {10.0, 10.0};

        auto result = asm_solver.couple(s_modes, a_modes, C);
        CHECK(result.size() == 3, "StructModes: 3 coupled modes");
        CHECK(result[0].frequency <= result[1].frequency,
              "StructModes: freq[0] <= freq[1]");
        CHECK(result[1].frequency <= result[2].frequency,
              "StructModes: freq[1] <= freq[2]");
    }

    // 4d. Participation factors are non-negative
    {
        std::vector<AcousticStructuralModes::ModeData> s_modes = {
            make_mode(100.0, 1.0), make_mode(200.0, 0.5)
        };
        std::vector<AcousticStructuralModes::ModeData> a_modes = {
            make_mode(150.0, 1.0)
        };
        std::vector<Real> C = {100.0, 50.0};

        auto result = asm_solver.couple(s_modes, a_modes, C);
        for (size_t i = 0; i < result.size(); ++i) {
            CHECK(result[i].structural_participation >= 0.0,
                  "StructModes: structural participation >= 0");
            CHECK(result[i].acoustic_participation >= 0.0,
                  "StructModes: acoustic participation >= 0");
        }
    }

    // 4e. Weak coupling: small frequency shift
    {
        std::vector<AcousticStructuralModes::ModeData> s_modes = {
            make_mode(100.0, 1.0)
        };
        std::vector<AcousticStructuralModes::ModeData> a_modes = {
            make_mode(200.0, 1.0)
        };
        std::vector<Real> C = {0.1};  // very weak

        auto result = asm_solver.couple(s_modes, a_modes, C);
        CHECK_NEAR(result[0].frequency, 100.0, 5.0,
                   "StructModes: weak coupling -> small shift (low)");
        CHECK_NEAR(result[1].frequency, 200.0, 5.0,
                   "StructModes: weak coupling -> small shift (high)");
    }

    // 4f. Coupling changes frequencies compared to uncoupled
    {
        auto s_modes = std::vector<AcousticStructuralModes::ModeData>{
            make_mode(150.0, 1.0)
        };
        auto a_modes = std::vector<AcousticStructuralModes::ModeData>{
            make_mode(200.0, 1.0)
        };

        std::vector<Real> C_zero = {0.0};
        auto r0 = asm_solver.couple(s_modes, a_modes, C_zero);
        Real split_uncoupled = r0[1].frequency - r0[0].frequency;

        std::vector<Real> C_coupled = {5000.0};
        auto rc = asm_solver.couple(s_modes, a_modes, C_coupled);
        Real split_coupled = rc[1].frequency - rc[0].frequency;

        CHECK(std::abs(split_coupled - split_uncoupled) > 1e-6,
              "StructModes: coupling changes frequency split");
    }

    // 4g. All frequencies positive
    {
        auto s_modes = std::vector<AcousticStructuralModes::ModeData>{
            make_mode(50.0, 1.0), make_mode(100.0, 1.0)
        };
        auto a_modes = std::vector<AcousticStructuralModes::ModeData>{
            make_mode(75.0, 1.0)
        };
        std::vector<Real> C = {100.0, 50.0};

        auto result = asm_solver.couple(s_modes, a_modes, C);
        for (size_t i = 0; i < result.size(); ++i) {
            CHECK(result[i].frequency >= 0.0,
                  "StructModes: all frequencies non-negative");
        }
    }

    // 4h. Empty structural modes -> only acoustic modes
    {
        std::vector<AcousticStructuralModes::ModeData> s_modes;
        auto a_modes = std::vector<AcousticStructuralModes::ModeData>{
            make_mode(100.0, 1.0)
        };
        std::vector<Real> C;

        auto result = asm_solver.couple(s_modes, a_modes, C);
        CHECK(result.size() == 1, "StructModes: 1 acoustic mode only");
        CHECK_NEAR(result[0].frequency, 100.0, 0.1,
                   "StructModes: uncoupled acoustic freq preserved");
    }

    // 4i. build_coupling_matrix produces correct size
    {
        std::vector<Real> areas = {1.0, 1.0};
        std::vector<Real> normals = {0,0,1, 0,0,1};  // 2 nodes, z-normal
        int n_s = 2, n_a = 3, n_i = 2;
        std::vector<Real> s_shapes(static_cast<size_t>(n_s * n_i), 1.0);
        std::vector<Real> a_shapes(static_cast<size_t>(n_a * n_i), 1.0);

        auto C = asm_solver.build_coupling_matrix(areas, normals, n_s, n_a,
                                                   s_shapes, a_shapes, n_i);
        CHECK(C.size() == static_cast<size_t>(n_s * n_a),
              "StructModes: coupling matrix size = n_s * n_a");
    }

    // 4j. CoupledMode has frequency_shift field
    {
        auto s_modes = std::vector<AcousticStructuralModes::ModeData>{
            make_mode(100.0, 1.0)
        };
        auto a_modes = std::vector<AcousticStructuralModes::ModeData>{
            make_mode(300.0, 1.0)
        };
        std::vector<Real> C = {500.0};

        auto result = asm_solver.couple(s_modes, a_modes, C);
        // frequency_shift should be set for each mode
        Real total_shift = 0.0;
        for (auto& m : result) total_shift += std::abs(m.frequency_shift);
        CHECK(total_shift > 0.0,
              "StructModes: frequency_shift populated for coupled modes");
    }
}

// ============================================================================
// main
// ============================================================================

int main() {
    test_1_sub_iteration();
    test_2_field_smoothing();
    test_3_acoustic_fmm();
    test_4_structural_modes();

    std::cout << "\n" << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    return tests_failed > 0 ? 1 : 0;
}
