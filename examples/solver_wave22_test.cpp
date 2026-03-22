/**
 * @file solver_wave22_test.cpp
 * @brief Wave 22: Solver Hardening Features Test Suite (5 components, ~40 tests)
 *
 * Tests 5 sub-modules (~8 tests each):
 *  1. MassScaling             - Selective/uniform scaling, timestep targets, mass increase
 *  2. Subcycling              - Element grouping by timestep, step counts, conservation
 *  3. AddedMassFluid          - Added mass for submerged structures, frequency shift
 *  4. DynamicRelaxation       - KE monitoring, velocity damping, convergence detection
 *  5. SmoothParticleContact   - Smooth normals, chattering elimination, contact force direction
 */

#include <nexussim/fem/solver_wave22.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <array>

using namespace nxs;
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
// Material constants for steel
// ============================================================================
static constexpr double E_steel = 210.0e9;       // Young's modulus [Pa]
static constexpr double rho_steel = 7800.0;       // Density [kg/m^3]
static constexpr double nu_steel = 0.3;           // Poisson's ratio
static constexpr double dt_target = 1.0e-6;       // Target timestep [s]

// Sound speed for 1D bar element
static double steel_sound_speed() {
    return std::sqrt(E_steel / rho_steel);
}

// ============================================================================
// 1. MassScaling Tests
// ============================================================================
void test_mass_scaling() {
    std::cout << "\n--- MassScaling Tests ---\n";

    double c = steel_sound_speed(); // ~5189 m/s

    // Test 1: Selective scaling only scales small elements (dt_elem < dt_target)
    {
        // 3 elements with different sizes: 10mm, 1mm, 5mm
        // dt_crit for 10mm: 0.01/5189 ~ 1.93e-6 (above target 1e-6 => no scaling)
        // dt_crit for 1mm:  0.001/5189 ~ 1.93e-7 (below target => scaling needed)
        // dt_crit for 5mm:  0.005/5189 ~ 9.64e-7 (below target => scaling needed)
        int N = 3;
        std::vector<Real> sizes = {0.01, 0.001, 0.005};
        std::vector<Real> speeds = {c, c, c};
        std::vector<Real> dens = {rho_steel, rho_steel, rho_steel};
        // Volume = L^3 for cubes
        std::vector<Real> vols = {1e-6, 1e-9, 1.25e-4};
        std::vector<Real> mass_inc(N, 0.0);

        auto result = MassScaling::compute_scaled_mass(
            sizes.data(), speeds.data(), dens.data(), vols.data(),
            dt_target, N, mass_inc.data());

        // Large element (10mm) should NOT be scaled (factor = 1.0)
        CHECK_NEAR(result.scale_factors[0], 1.0, 1e-10,
            "MassScaling: large element (10mm) not scaled");

        // Small element (1mm) SHOULD be scaled (factor > 1.0)
        CHECK(result.scale_factors[1] > 1.0,
            "MassScaling: small element (1mm) is scaled up");
    }

    // Test 2: Correct scale factor = (dt_target / dt_elem)^2
    {
        int N = 1;
        std::vector<Real> sizes = {0.001};   // 1mm element
        std::vector<Real> speeds = {c};
        std::vector<Real> dens = {rho_steel};
        std::vector<Real> vols = {1e-9};
        std::vector<Real> mass_inc(N, 0.0);

        auto result = MassScaling::compute_scaled_mass(
            sizes.data(), speeds.data(), dens.data(), vols.data(),
            dt_target, N, mass_inc.data());

        double dt_elem = 0.001 / c;
        double expected_alpha = (dt_target / dt_elem) * (dt_target / dt_elem);
        CHECK_NEAR(result.scale_factors[0], expected_alpha, 0.1,
            "MassScaling: scale factor = (dt_target/dt_elem)^2");
    }

    // Test 3: Total added mass is positive when small elements exist
    {
        int N = 3;
        std::vector<Real> sizes = {0.01, 0.001, 0.005};
        std::vector<Real> speeds = {c, c, c};
        std::vector<Real> dens = {rho_steel, rho_steel, rho_steel};
        std::vector<Real> vols = {1e-6, 1e-9, 1.25e-4};
        std::vector<Real> mass_inc(N, 0.0);

        auto result = MassScaling::compute_scaled_mass(
            sizes.data(), speeds.data(), dens.data(), vols.data(),
            dt_target, N, mass_inc.data());

        CHECK(result.total_added_mass > 0.0,
            "MassScaling: total added mass is positive");
        CHECK(result.added_mass_ratio > 0.0,
            "MassScaling: added mass ratio is positive");
    }

    // Test 4: No scaling when all elements have dt > target
    {
        int N = 3;
        // All elements 10cm+ => dt_crit >> dt_target
        std::vector<Real> sizes = {0.1, 0.05, 0.02};
        std::vector<Real> speeds = {c, c, c};
        std::vector<Real> dens = {rho_steel, rho_steel, rho_steel};
        std::vector<Real> vols = {1e-3, 1.25e-4, 8e-6};
        std::vector<Real> mass_inc(N, 0.0);

        auto result = MassScaling::compute_scaled_mass(
            sizes.data(), speeds.data(), dens.data(), vols.data(),
            dt_target, N, mass_inc.data());

        CHECK(result.num_scaled_elements == 0,
            "MassScaling: no elements scaled when all dt > target");
        CHECK_NEAR(result.total_added_mass, 0.0, 1e-15,
            "MassScaling: zero added mass when no scaling needed");
    }

    // Test 5: Mass increase array consistent with total_added_mass
    {
        int N = 3;
        std::vector<Real> sizes = {0.01, 0.001, 0.005};
        std::vector<Real> speeds = {c, c, c};
        std::vector<Real> dens = {rho_steel, rho_steel, rho_steel};
        std::vector<Real> vols = {1e-6, 1e-9, 1.25e-4};
        std::vector<Real> mass_inc(N, 0.0);

        auto result = MassScaling::compute_scaled_mass(
            sizes.data(), speeds.data(), dens.data(), vols.data(),
            dt_target, N, mass_inc.data());

        double sum_inc = 0.0;
        for (int i = 0; i < N; i++) sum_inc += mass_inc[i];
        CHECK_NEAR(sum_inc, result.total_added_mass, 1e-10,
            "MassScaling: sum of mass_increase = total_added_mass");
    }

    // Test 6: total_added_mass_ratio utility function
    {
        Real orig = 100.0;
        Real scaled = 115.0;
        Real ratio = MassScaling::total_added_mass_ratio(orig, scaled);
        CHECK_NEAR(ratio, 0.15, 1e-10,
            "MassScaling: total_added_mass_ratio = (scaled-orig)/orig = 0.15");
    }

    // Test 7: num_scaled_elements count is correct
    {
        int N = 4;
        // Two small, two large
        std::vector<Real> sizes = {0.001, 0.002, 0.05, 0.1};
        std::vector<Real> speeds = {c, c, c, c};
        std::vector<Real> dens = {rho_steel, rho_steel, rho_steel, rho_steel};
        std::vector<Real> vols = {1e-9, 8e-9, 1.25e-4, 1e-3};
        std::vector<Real> mass_inc(N, 0.0);

        auto result = MassScaling::compute_scaled_mass(
            sizes.data(), speeds.data(), dens.data(), vols.data(),
            dt_target, N, mass_inc.data());

        // The two small elements (1mm, 2mm) have dt < 1e-6
        // dt_1mm = 0.001/5189 ~ 1.93e-7 < 1e-6 => scaled
        // dt_2mm = 0.002/5189 ~ 3.86e-7 < 1e-6 => scaled
        // dt_50mm = 0.05/5189 ~ 9.64e-6 > 1e-6 => not scaled
        // dt_100mm = 0.1/5189 ~ 1.93e-5 > 1e-6 => not scaled
        CHECK(result.num_scaled_elements == 2,
            "MassScaling: 2 elements scaled out of 4");
    }

    // Test 8: apply_mass_scaling distributes added mass to DOFs
    {
        int num_elem = 2;
        int ndpe = 2; // 2 DOFs per element

        std::vector<Real> masses = {1.0, 1.0, 1.0, 1.0};
        std::vector<Real> scale_factors = {4.0, 1.0}; // Only elem 0 scaled
        std::vector<int> elem_to_dof = {0, 1,   2, 3}; // elem 0 -> dof 0,1; elem 1 -> dof 2,3
        std::vector<Real> orig_mass = {2.0, 2.0}; // original element masses

        MassScaling::apply_mass_scaling(
            masses.data(), scale_factors.data(),
            elem_to_dof.data(), ndpe, num_elem,
            orig_mass.data());

        // elem 0: added = 2.0*(4-1) = 6.0, per_dof = 3.0
        // dof 0: 1.0 + 3.0 = 4.0; dof 1: 1.0 + 3.0 = 4.0
        CHECK_NEAR(masses[0], 4.0, 1e-10,
            "MassScaling: apply adds 3.0 to dof 0");
        CHECK_NEAR(masses[1], 4.0, 1e-10,
            "MassScaling: apply adds 3.0 to dof 1");
        // elem 1 not scaled, dofs 2,3 unchanged
        CHECK_NEAR(masses[2], 1.0, 1e-10,
            "MassScaling: unscaled element dofs unchanged");
    }
}

// ============================================================================
// 2. Subcycling Tests
// ============================================================================
void test_subcycling() {
    std::cout << "\n--- Subcycling Tests ---\n";

    double c = steel_sound_speed();

    // Test 1: Elements grouped by timestep ratio
    {
        int N = 3;
        // dt for each element
        std::vector<Real> elem_dt = {
            0.01 / c,    // ~1.93e-6 (10mm)
            0.001 / c,   // ~1.93e-7 (1mm)
            0.005 / c    // ~9.64e-7 (5mm)
        };
        Real dt_global = 2.0e-6;

        auto groups = Subcycling::group_elements(elem_dt.data(), N, dt_global);

        CHECK(groups.size() >= 1,
            "Subcycling: at least one group created");
    }

    // Test 2: Substeps are powers of 2
    {
        int N = 4;
        std::vector<Real> elem_dt = {
            0.01 / c, 0.001 / c, 0.005 / c, 0.002 / c
        };
        Real dt_global = 2.0e-6;

        auto groups = Subcycling::group_elements(elem_dt.data(), N, dt_global);

        for (auto& g : groups) {
            bool is_pow2 = (g.ratio > 0) && ((g.ratio & (g.ratio - 1)) == 0);
            CHECK(is_pow2,
                "Subcycling: group ratio is power of 2");
        }
    }

    // Test 3: sub_dt = dt_global / ratio
    {
        int N = 3;
        std::vector<Real> elem_dt = {0.01 / c, 0.005 / c, 0.002 / c};
        Real dt_global = 2.0e-6;

        auto groups = Subcycling::group_elements(elem_dt.data(), N, dt_global);

        for (auto& g : groups) {
            Real expected_sub_dt = dt_global / static_cast<Real>(g.ratio);
            CHECK_NEAR(g.sub_dt, expected_sub_dt, 1e-15,
                "Subcycling: sub_dt = dt_global / ratio");
        }
    }

    // Test 4: All elements assigned to exactly one group
    {
        int N = 5;
        std::vector<Real> elem_dt = {
            0.01 / c, 0.005 / c, 0.002 / c, 0.001 / c, 0.008 / c
        };
        Real dt_global = 2.0e-6;

        auto groups = Subcycling::group_elements(elem_dt.data(), N, dt_global);

        int total_assigned = 0;
        for (auto& g : groups) {
            total_assigned += static_cast<int>(g.element_ids.size());
        }
        CHECK(total_assigned == N,
            "Subcycling: all elements assigned to groups");
    }

    // Test 5: Fast elements get higher ratio than slow elements
    {
        int N = 2;
        // Element 0: dt = 1.93e-7 (fast, needs many substeps)
        // Element 1: dt = 1.93e-5 (slow, ratio = 1)
        std::vector<Real> elem_dt = {0.001 / c, 0.1 / c};
        Real dt_global = 2.0e-6;

        auto groups = Subcycling::group_elements(elem_dt.data(), N, dt_global);

        // Find which group contains each element
        int fast_ratio = -1, slow_ratio = -1;
        for (auto& g : groups) {
            for (auto eid : g.element_ids) {
                if (eid == 0) fast_ratio = g.ratio;
                if (eid == 1) slow_ratio = g.ratio;
            }
        }
        CHECK(fast_ratio >= slow_ratio,
            "Subcycling: fast element has higher ratio than slow element");
    }

    // Test 6: Groups sorted by ascending ratio
    {
        int N = 4;
        std::vector<Real> elem_dt = {
            0.01 / c, 0.001 / c, 0.005 / c, 0.002 / c
        };
        Real dt_global = 2.0e-6;

        auto groups = Subcycling::group_elements(elem_dt.data(), N, dt_global);

        bool sorted = true;
        for (size_t i = 1; i < groups.size(); i++) {
            if (groups[i].ratio < groups[i-1].ratio) sorted = false;
        }
        CHECK(sorted,
            "Subcycling: groups sorted by ascending ratio");
    }

    // Test 7: Slow element with dt >= dt_global gets ratio 1
    {
        int N = 1;
        std::vector<Real> elem_dt = {5.0e-6}; // Well above dt_global
        Real dt_global = 2.0e-6;

        auto groups = Subcycling::group_elements(elem_dt.data(), N, dt_global);

        CHECK(groups.size() == 1 && groups[0].ratio == 1,
            "Subcycling: slow element gets ratio = 1");
    }

    // Test 8: subcycle_step runs without crash (basic smoke test)
    {
        int N = 2;
        std::vector<Real> elem_dt = {0.001 / c, 0.01 / c};
        Real dt_global = 2.0e-6;

        auto groups = Subcycling::group_elements(elem_dt.data(), N, dt_global);

        // 2 nodes with 3 DOFs each
        int num_nodes = 2;
        std::vector<Real> positions(6, 0.0);
        std::vector<Real> velocities(6, 0.0);
        std::vector<Real> masses(6, 1.0);

        auto force_fn = [](const std::vector<int>& elem_ids, const Real* pos,
                           Real* forces, int ndof) {
            // Simple constant force on first node in x-direction
            forces[0] += 100.0;
        };

        Subcycling::subcycle_step(groups, positions.data(), velocities.data(),
                                   masses.data(), num_nodes, dt_global, force_fn);

        // After the step, velocity should have changed due to force
        CHECK(velocities[0] != 0.0,
            "Subcycling: subcycle_step updates velocities");
    }
}

// ============================================================================
// 3. AddedMassFluid Tests
// ============================================================================
void test_added_mass_fluid() {
    std::cout << "\n--- AddedMassFluid Tests ---\n";

    Real rho_water = 1025.0;  // Seawater [kg/m^3]
    Real Ca_plate = 1.0;      // Added mass coefficient for flat plate

    // Test 1: Added mass for fully submerged steel plate
    {
        int N = 1;
        Real plate_vol = 0.1 * 0.1 * 0.01;  // 1e-4 m^3
        std::vector<Real> struct_mass = {rho_steel * plate_vol};
        std::vector<Real> disp_vol = {plate_vol};

        auto result = AddedMassFluid::compute_added_mass(
            struct_mass.data(), disp_vol.data(), rho_water, Ca_plate, N);

        Real expected_added = Ca_plate * rho_water * plate_vol;
        CHECK_NEAR(result.added_mass[0], expected_added, 1e-6,
            "AddedMassFluid: correct added mass for submerged plate");
    }

    // Test 2: Added mass proportional to displaced volume
    {
        int N = 2;
        std::vector<Real> struct_mass = {100.0, 100.0};
        std::vector<Real> disp_vol = {0.1, 0.2};

        auto result = AddedMassFluid::compute_added_mass(
            struct_mass.data(), disp_vol.data(), rho_water, Ca_plate, N);

        // m_added = Ca * rho * V; 0.2/0.1 = 2x the added mass
        CHECK_NEAR(result.added_mass[1] / result.added_mass[0], 2.0, 1e-10,
            "AddedMassFluid: twice the volume gives twice the added mass");
    }

    // Test 3: Zero displaced volume gives zero added mass
    {
        int N = 1;
        std::vector<Real> struct_mass = {5.0};
        std::vector<Real> disp_vol = {0.0};

        auto result = AddedMassFluid::compute_added_mass(
            struct_mass.data(), disp_vol.data(), rho_water, Ca_plate, N);

        CHECK_NEAR(result.added_mass[0], 0.0, 1e-15,
            "AddedMassFluid: zero added mass for zero displaced volume");
    }

    // Test 4: Cylinder with Ca=1.0 (potential flow exact for long cylinder)
    {
        int N = 1;
        Real R = 0.05, L_cyl = 1.0;
        Real V_cyl = M_PI * R * R * L_cyl;

        std::vector<Real> struct_mass = {rho_steel * V_cyl};
        std::vector<Real> disp_vol = {V_cyl};

        auto result = AddedMassFluid::compute_added_mass(
            struct_mass.data(), disp_vol.data(), 1000.0, 1.0, N);

        Real expected = 1.0 * 1000.0 * V_cyl;
        CHECK_NEAR(result.added_mass[0], expected, 1e-6,
            "AddedMassFluid: cylindrical geometry added mass correct");
    }

    // Test 5: Total added mass and structural mass sums
    {
        int N = 3;
        std::vector<Real> struct_mass = {1.0, 1.5, 2.0};
        std::vector<Real> disp_vol = {0.001, 0.002, 0.003};

        auto result = AddedMassFluid::compute_added_mass(
            struct_mass.data(), disp_vol.data(), rho_water, Ca_plate, N);

        CHECK_NEAR(result.total_struct, 4.5, 1e-10,
            "AddedMassFluid: total structural mass summed correctly");

        Real expected_total_added = Ca_plate * rho_water * (0.001 + 0.002 + 0.003);
        CHECK_NEAR(result.total_added, expected_total_added, 1e-6,
            "AddedMassFluid: total added mass summed correctly");
    }

    // Test 6: Added mass ratio = total_added / total_struct
    {
        int N = 1;
        std::vector<Real> struct_mass = {100.0};
        std::vector<Real> disp_vol = {0.1};

        auto result = AddedMassFluid::compute_added_mass(
            struct_mass.data(), disp_vol.data(), rho_water, Ca_plate, N);

        Real expected_ratio = result.total_added / result.total_struct;
        CHECK_NEAR(result.ratio, expected_ratio, 1e-10,
            "AddedMassFluid: ratio = total_added / total_struct");
    }

    // Test 7: apply_to_mass_matrix adds to all 3 translational DOFs
    {
        int N = 2;
        // 2 nodes, 6 DOFs total
        std::vector<Real> mass_matrix = {1.0, 1.0, 1.0,   2.0, 2.0, 2.0};
        std::vector<Real> added_mass = {0.5, 1.0};

        AddedMassFluid::apply_to_mass_matrix(mass_matrix.data(), added_mass.data(), N);

        // Node 0: each DOF gets +0.5
        CHECK_NEAR(mass_matrix[0], 1.5, 1e-10,
            "AddedMassFluid: node 0 DOF 0 updated");
        CHECK_NEAR(mass_matrix[1], 1.5, 1e-10,
            "AddedMassFluid: node 0 DOF 1 updated");
        CHECK_NEAR(mass_matrix[2], 1.5, 1e-10,
            "AddedMassFluid: node 0 DOF 2 updated");
        // Node 1: each DOF gets +1.0
        CHECK_NEAR(mass_matrix[3], 3.0, 1e-10,
            "AddedMassFluid: node 1 DOF 0 updated");
    }

    // Test 8: estimate_displaced_volume = area * char_length
    {
        Real area = 0.01;   // m^2
        Real L = 0.05;      // m
        Real V = AddedMassFluid::estimate_displaced_volume(area, L);
        CHECK_NEAR(V, 0.0005, 1e-10,
            "AddedMassFluid: displaced volume = area * char_length");
    }
}

// ============================================================================
// 4. DynamicRelaxation Tests
// ============================================================================
void test_dynamic_relaxation() {
    std::cout << "\n--- DynamicRelaxation Tests ---\n";

    // Test 1: apply_damping reduces velocities by factor (1-damping)
    {
        std::vector<Real> v = {10.0, -5.0, 3.0};
        DynamicRelaxation::apply_damping(v.data(), 3, 0.1);

        CHECK_NEAR(v[0], 9.0, 1e-10,
            "DynRelax: apply_damping reduces v[0] by 10%");
        CHECK_NEAR(v[1], -4.5, 1e-10,
            "DynRelax: apply_damping reduces v[1] by 10%");
        CHECK_NEAR(v[2], 2.7, 1e-10,
            "DynRelax: apply_damping reduces v[2] by 10%");
    }

    // Test 2: Full damping (factor=1.0) zeros all velocities
    {
        std::vector<Real> v = {100.0, -50.0, 25.0};
        DynamicRelaxation::apply_damping(v.data(), 3, 1.0);

        CHECK_NEAR(v[0], 0.0, 1e-15,
            "DynRelax: full damping zeros velocity");
    }

    // Test 3: check_convergence returns true when KE/IE < tolerance
    {
        bool conv = DynamicRelaxation::check_convergence(1e-8, 1.0, 1e-6);
        CHECK(conv, "DynRelax: converged when KE/IE = 1e-8 < tol 1e-6");

        bool not_conv = DynamicRelaxation::check_convergence(1.0, 1.0, 1e-6);
        CHECK(!not_conv, "DynRelax: not converged when KE/IE = 1.0 > tol 1e-6");
    }

    // Test 4: check_convergence handles zero IE case
    {
        // Both KE and IE near zero => converged
        bool conv = DynamicRelaxation::check_convergence(1e-31, 1e-31, 1e-6);
        CHECK(conv, "DynRelax: converged when both KE and IE are near zero");

        // KE > 0 but IE = 0 => not converged
        bool not_conv = DynamicRelaxation::check_convergence(1.0, 1e-31, 1e-6);
        CHECK(!not_conv, "DynRelax: not converged when KE>0, IE~0");
    }

    // Test 5: adaptive_damping increases with oscillation
    {
        // Oscillating KE history: up, down, up, down
        std::vector<Real> ke_osc = {10.0, 5.0, 8.0, 3.0, 7.0, 2.0};
        Real damp_osc = DynamicRelaxation::adaptive_damping(ke_osc);

        // Monotonically decreasing KE
        std::vector<Real> ke_mono = {10.0, 8.0, 6.0, 4.0, 2.0, 1.0};
        Real damp_mono = DynamicRelaxation::adaptive_damping(ke_mono);

        CHECK(damp_osc >= damp_mono,
            "DynRelax: oscillating KE gets higher damping than monotone");
    }

    // Test 6: DynamicRelaxation step advances iteration count
    {
        DynamicRelaxation dr;
        dr.set_damping(0.9);
        dr.set_tolerance(1e-6);

        int num_nodes = 1;
        std::vector<Real> vel = {0.0, 0.0, 0.0};
        std::vector<Real> forces = {100.0, 0.0, 0.0};
        std::vector<Real> masses = {1.0, 1.0, 1.0};
        std::vector<Real> pos = {0.0, 0.0, 0.0};

        auto info = dr.step(vel.data(), forces.data(), masses.data(),
                            pos.data(), num_nodes, 0.001, 1.0);

        CHECK(info.iteration == 1,
            "DynRelax: iteration = 1 after one step");
    }

    // Test 7: Step updates positions when force is applied
    {
        DynamicRelaxation dr;
        dr.set_damping(0.0);
        dr.set_adaptive(false);

        int num_nodes = 1;
        std::vector<Real> vel = {0.0, 0.0, 0.0};
        std::vector<Real> forces = {1000.0, 0.0, 0.0};
        std::vector<Real> masses = {1.0, 1.0, 1.0};
        std::vector<Real> pos = {0.0, 0.0, 0.0};

        dr.step(vel.data(), forces.data(), masses.data(),
                pos.data(), num_nodes, 0.01, 1.0);

        CHECK(pos[0] > 0.0,
            "DynRelax: position updated in force direction");
    }

    // Test 8: Convergence detected for quasi-static problem
    {
        DynamicRelaxation dr;
        dr.set_damping(0.95);
        dr.set_tolerance(1e-4);
        dr.set_adaptive(false);

        int num_nodes = 1;
        std::vector<Real> vel = {0.0, 0.0, 0.0};
        std::vector<Real> masses = {1.0, 1.0, 1.0};
        std::vector<Real> pos = {0.0, 0.0, 0.0};
        Real dt = 0.001;

        // Spring force: F = k*(target - x), k=100, target = 0.5
        Real k_spring = 100.0;
        Real target = 0.5;

        bool converged = false;
        for (int iter = 0; iter < 100000; iter++) {
            std::vector<Real> forces = {
                k_spring * (target - pos[0]),
                0.0,
                0.0
            };

            // Strain energy = 0.5 * k * (target - x)^2
            Real IE = 0.5 * k_spring * (target - pos[0]) * (target - pos[0]);
            if (IE < 1e-30) IE = 1e-30;

            auto info = dr.step(vel.data(), forces.data(), masses.data(),
                                pos.data(), num_nodes, dt, IE);

            if (info.converged) {
                converged = true;
                break;
            }
        }

        CHECK(converged,
            "DynRelax: quasi-static spring problem converges");
    }
}

// ============================================================================
// 5. SmoothParticleContact Tests
// ============================================================================
void test_smooth_particle_contact() {
    std::cout << "\n--- SmoothParticleContact Tests ---\n";

    using Face = SmoothParticleContact::Face;

    // Helper: build a flat quad face at z=0 with normal (0,0,1)
    auto make_flat_face = [](int n0, int n1, int n2, int n3,
                             Real cx, Real cy, Real cz,
                             Real nx, Real ny, Real nz, Real area) -> Face {
        Face f;
        f.node_ids[0] = n0; f.node_ids[1] = n1;
        f.node_ids[2] = n2; f.node_ids[3] = n3;
        f.num_nodes = 4;
        f.normal[0] = nx; f.normal[1] = ny; f.normal[2] = nz;
        f.area = area;
        f.centroid[0] = cx; f.centroid[1] = cy; f.centroid[2] = cz;
        return f;
    };

    // Test 1: Smooth normal from identical face normals equals that normal
    {
        // Two faces sharing node 0, both with normal (0,0,1)
        std::vector<Face> faces = {
            make_flat_face(0, 1, 2, 3,  -0.5, 0.0, 0.0,  0.0, 0.0, 1.0,  1.0),
            make_flat_face(0, 4, 5, 6,   0.5, 0.0, 0.0,  0.0, 0.0, 1.0,  1.0)
        };

        Real sn[3];
        SmoothParticleContact::compute_smooth_normal(0, faces.data(), 2, sn);

        CHECK_NEAR(sn[0], 0.0, 1e-10,
            "SmoothContact: aligned normals give nx=0");
        CHECK_NEAR(sn[1], 0.0, 1e-10,
            "SmoothContact: aligned normals give ny=0");
        CHECK_NEAR(sn[2], 1.0, 1e-10,
            "SmoothContact: aligned normals give nz=1");
    }

    // Test 2: Smooth normal is unit vector (with non-aligned faces)
    {
        // Two faces at 90 degrees sharing node 0
        std::vector<Face> faces = {
            make_flat_face(0, 1, 2, 3,  0.0, 0.0, 0.0,  0.0, 0.0, 1.0,  1.0),
            make_flat_face(0, 4, 5, 6,  0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  1.0)
        };

        Real sn[3];
        SmoothParticleContact::compute_smooth_normal(0, faces.data(), 2, sn);

        Real mag = std::sqrt(sn[0]*sn[0] + sn[1]*sn[1] + sn[2]*sn[2]);
        CHECK_NEAR(mag, 1.0, 1e-10,
            "SmoothContact: smooth normal is unit vector");
    }

    // Test 3: Area-weighted averaging (larger face dominates)
    {
        // Face 0: normal (0,0,1), area=10 (dominant)
        // Face 1: normal (0,1,0), area=1  (minor)
        std::vector<Face> faces = {
            make_flat_face(0, 1, 2, 3,  0.0, 0.0, 0.0,  0.0, 0.0, 1.0,  10.0),
            make_flat_face(0, 4, 5, 6,  0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  1.0)
        };

        Real sn[3];
        SmoothParticleContact::compute_smooth_normal(0, faces.data(), 2, sn);

        // z-component should dominate due to 10x area
        CHECK(sn[2] > sn[1],
            "SmoothContact: larger face area dominates smooth normal");
    }

    // Test 4: project_to_smooth_surface gap is negative for penetrating node
    {
        // Single face at z=0 with upward normal
        Face face = make_flat_face(0, 1, 2, 3,  0.0, 0.0, 0.0,  0.0, 0.0, 1.0,  1.0);
        Real sn[3] = {0.0, 0.0, 1.0};

        // Node below the surface at z = -0.005
        Real node_pos[3] = {0.0, 0.0, -0.005};
        Real proj[3];

        Real gap = SmoothParticleContact::project_to_smooth_surface(
            node_pos, &face, 1, sn, proj);

        CHECK(gap < 0.0,
            "SmoothContact: negative gap for penetrating node");
        CHECK_NEAR(gap, -0.005, 1e-10,
            "SmoothContact: gap = -5mm for node at z=-0.005");
    }

    // Test 5: Contact force is zero when separated (positive gap)
    {
        Real sn[3] = {0.0, 0.0, 1.0};
        Real gap = 0.01;  // separated

        auto result = SmoothParticleContact::compute_contact_force(gap, sn, 1e9);
        CHECK(!result.in_contact,
            "SmoothContact: not in contact when gap > 0");
        CHECK_NEAR(result.force[0], 0.0, 1e-15,
            "SmoothContact: zero force when separated");
        CHECK_NEAR(result.force[1], 0.0, 1e-15,
            "SmoothContact: zero force y when separated");
        CHECK_NEAR(result.force[2], 0.0, 1e-15,
            "SmoothContact: zero force z when separated");
    }

    // Test 6: Contact force proportional to penetration and in normal direction
    {
        Real sn[3] = {0.0, 0.0, 1.0};
        Real penalty = 1.0e9;
        Real gap1 = -0.001;
        Real gap2 = -0.002;

        auto r1 = SmoothParticleContact::compute_contact_force(gap1, sn, penalty);
        auto r2 = SmoothParticleContact::compute_contact_force(gap2, sn, penalty);

        CHECK(r1.in_contact, "SmoothContact: in contact for negative gap");

        // Force direction is along smooth normal
        CHECK_NEAR(r1.force[0], 0.0, 1e-10,
            "SmoothContact: force x = 0 for z-normal");
        CHECK(r1.force[2] > 0.0,
            "SmoothContact: force z is repulsive (positive for z-normal)");

        // Force proportional to penetration
        CHECK_NEAR(r2.force[2] / r1.force[2], 2.0, 1e-10,
            "SmoothContact: force doubles with doubled penetration");
    }

    // Test 7: process_contact detects penetrating slave nodes
    {
        // Master surface: flat plane at z=0 with upward normal
        Face face = make_flat_face(0, 1, 2, 3,  0.0, 0.0, 0.0,  0.0, 0.0, 1.0,  1.0);

        // Two slave nodes: one below, one above
        std::vector<Real> slave_pos = {
            0.0, 0.0, -0.002,    // penetrating
            0.0, 0.0,  0.010     // separated
        };
        std::vector<Real> slave_forces(6, 0.0);

        int count = SmoothParticleContact::process_contact(
            slave_pos.data(), 2, &face, 1, 1e9, slave_forces.data());

        CHECK(count == 1,
            "SmoothContact: 1 out of 2 slave nodes in contact");

        // First slave should have repulsive force in +z
        CHECK(slave_forces[2] > 0.0,
            "SmoothContact: contact force pushes slave in +z direction");

        // Second slave should have zero force
        CHECK_NEAR(slave_forces[3], 0.0, 1e-15,
            "SmoothContact: no force on separated slave node");
    }

    // Test 8: Smooth normal eliminates chattering at edge between two faces
    {
        // Two adjacent faces meeting at a slight angle
        // Face 0: normal tilted slightly in +y: (0, sin(5deg), cos(5deg))
        // Face 1: normal tilted slightly in -y: (0, -sin(5deg), cos(5deg))
        Real angle = 5.0 * M_PI / 180.0;
        Real ny = std::sin(angle);
        Real nz = std::cos(angle);

        std::vector<Face> faces = {
            make_flat_face(0, 1, 2, 3,  0.0, -0.5, 0.0,  0.0, ny, nz,  1.0),
            make_flat_face(0, 4, 5, 6,  0.0,  0.5, 0.0,  0.0, -ny, nz, 1.0)
        };

        Real sn[3];
        SmoothParticleContact::compute_smooth_normal(0, faces.data(), 2, sn);

        // The y-components should cancel, leaving a pure z-normal
        // This is the chattering elimination: at the edge, the smooth normal
        // is the average, avoiding the discontinuous jump
        CHECK(std::abs(sn[1]) < 1e-10,
            "SmoothContact: y-components cancel at symmetric edge (anti-chattering)");
        CHECK(sn[2] > 0.99,
            "SmoothContact: smooth normal at edge is nearly vertical");
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== Wave 22: Solver Hardening Features Test Suite ===\n";

    test_mass_scaling();
    test_subcycling();
    test_added_mass_fluid();
    test_dynamic_relaxation();
    test_smooth_particle_contact();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " total ===\n";

    return (tests_failed == 0) ? 0 : 1;
}
