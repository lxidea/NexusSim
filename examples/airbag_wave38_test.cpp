/**
 * @file airbag_wave38_test.cpp
 * @brief Wave 38: Airbag Production Test Suite (5 features, 40 tests)
 *
 * Tests:
 *   1. FVBagSolver         (8 tests)
 *   2. AirbagInjection     (8 tests)
 *   3. AirbagVenting       (8 tests)
 *   4. AirbagFolding       (8 tests)
 *   5. AirbagThermal       (8 tests)
 */

#include <nexussim/fem/airbag_wave38.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>

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
// 1. FVBagSolver Tests
// ============================================================================

void test_1_fvbag_solver() {
    std::cout << "--- Test 1: FVBagSolver ---\n";

    // 1a. Ideal gas law P*V = m*R*T
    {
        AirbagChamber ch;
        ch.volume = 0.01;       // 10 liters
        ch.mass = 0.012;        // 12 grams
        ch.temperature = 300.0; // 300 K
        Real R_gas = 287.0;
        Real P_expected = ch.mass * R_gas * ch.temperature / ch.volume;
        Real P_computed = FVBagSolver::ideal_gas_pressure(ch, R_gas);
        CHECK_NEAR(P_computed, P_expected, 1.0, "FVBag: ideal gas P = mRT/V");
    }

    // 1b. Pressure increases with added mass
    {
        AirbagChamber ch;
        ch.volume = 0.01;
        ch.mass = 0.01;
        ch.temperature = 300.0;
        Real R_gas = 287.0;
        Real P1 = FVBagSolver::ideal_gas_pressure(ch, R_gas);
        ch.mass = 0.02;
        Real P2 = FVBagSolver::ideal_gas_pressure(ch, R_gas);
        CHECK(P2 > P1, "FVBag: more mass -> higher pressure");
    }

    // 1c. Pressure inversely proportional to volume
    {
        Real R_gas = 287.0;
        AirbagChamber ch;
        ch.mass = 0.01;
        ch.temperature = 300.0;
        ch.volume = 0.01;
        Real P1 = FVBagSolver::ideal_gas_pressure(ch, R_gas);
        ch.volume = 0.02;
        Real P2 = FVBagSolver::ideal_gas_pressure(ch, R_gas);
        CHECK_NEAR(P1 / P2, 2.0, 0.001, "FVBag: P inversely proportional to V");
    }

    // 1d. Chamber energy: E = m*R*T/(gamma-1)
    {
        AirbagChamber ch;
        ch.mass = 0.01;
        ch.temperature = 500.0;
        Real R_gas = 287.0;
        Real gamma = 1.4;
        Real E = FVBagSolver::chamber_energy(ch, R_gas, gamma);
        Real E_expected = ch.mass * R_gas * ch.temperature / (gamma - 1.0);
        CHECK_NEAR(E, E_expected, 0.01, "FVBag: chamber energy = mRT/(gamma-1)");
    }

    // 1e. Multi-chamber: mass conservation after flow
    {
        AirbagConfig cfg;
        cfg.n_chambers = 2;
        cfg.R_gas = 287.0;
        cfg.gamma = 1.4;

        AirbagChamber chambers[2];
        // Chamber 0: high pressure
        chambers[0].volume = 0.01;
        chambers[0].mass = 0.02;
        chambers[0].temperature = 400.0;
        chambers[0].pressure = FVBagSolver::ideal_gas_pressure(chambers[0], cfg.R_gas);
        chambers[0].n_connections = 1;
        chambers[0].connected_to[0] = 1;
        chambers[0].orifice_area[0] = 0.001;

        // Chamber 1: low pressure
        chambers[1].volume = 0.01;
        chambers[1].mass = 0.005;
        chambers[1].temperature = 300.0;
        chambers[1].pressure = FVBagSolver::ideal_gas_pressure(chambers[1], cfg.R_gas);

        Real total_mass_before = chambers[0].mass + chambers[1].mass;

        FVBagSolver solver(cfg);
        solver.solve_bag_step(chambers, 2, cfg, 1.0e-5);

        Real total_mass_after = chambers[0].mass + chambers[1].mass;
        CHECK_NEAR(total_mass_after, total_mass_before, 1.0e-12,
                   "FVBag: mass conservation in 2-chamber system");
    }

    // 1f. Multi-chamber: flow from high to low pressure
    {
        AirbagConfig cfg;
        cfg.R_gas = 287.0;
        cfg.gamma = 1.4;

        AirbagChamber chambers[2];
        chambers[0].volume = 0.01;
        chambers[0].mass = 0.03;
        chambers[0].temperature = 400.0;
        chambers[0].pressure = FVBagSolver::ideal_gas_pressure(chambers[0], cfg.R_gas);
        chambers[0].n_connections = 1;
        chambers[0].connected_to[0] = 1;
        chambers[0].orifice_area[0] = 0.001;

        chambers[1].volume = 0.01;
        chambers[1].mass = 0.005;
        chambers[1].temperature = 300.0;
        chambers[1].pressure = FVBagSolver::ideal_gas_pressure(chambers[1], cfg.R_gas);

        Real m0_before = chambers[0].mass;
        Real m1_before = chambers[1].mass;

        FVBagSolver solver(cfg);
        solver.solve_bag_step(chambers, 2, cfg, 1.0e-5);

        CHECK(chambers[0].mass < m0_before, "FVBag: high-P chamber loses mass");
        CHECK(chambers[1].mass > m1_before, "FVBag: low-P chamber gains mass");
    }

    // 1g. Single chamber: no flow when no connections
    {
        AirbagConfig cfg;
        cfg.R_gas = 287.0;
        cfg.gamma = 1.4;

        AirbagChamber ch;
        ch.volume = 0.01;
        ch.mass = 0.01;
        ch.temperature = 350.0;
        ch.pressure = FVBagSolver::ideal_gas_pressure(ch, cfg.R_gas);
        ch.n_connections = 0;

        Real m_before = ch.mass;
        Real T_before = ch.temperature;

        FVBagSolver solver(cfg);
        solver.solve_bag_step(&ch, 1, cfg, 1.0e-4);

        CHECK_NEAR(ch.mass, m_before, 1.0e-15, "FVBag: isolated chamber mass unchanged");
    }
}

// ============================================================================
// 2. AirbagInjection Tests
// ============================================================================

void test_2_airbag_injection() {
    std::cout << "--- Test 2: AirbagInjection ---\n";

    // 2a. Zero injection before start time
    {
        AirbagInjection inj;
        InflatorData inflator;
        inflator.t_start = 0.01;
        inflator.n_points = 2;
        inflator.time_pts[0] = 0.0; inflator.time_pts[1] = 0.1;
        inflator.mass_rate_curve[0] = 1.0; inflator.mass_rate_curve[1] = 1.0;
        inflator.temp_curve[0] = 500.0; inflator.temp_curve[1] = 500.0;

        Real dm_dt, T_inj;
        inj.compute_injection(inflator, 0.0, dm_dt, T_inj);
        CHECK_NEAR(dm_dt, 0.0, 1.0e-15, "Injection: zero before t_start");
    }

    // 2b. Constant mass rate injection
    {
        AirbagInjection inj;
        InflatorData inflator;
        inflator.t_start = 0.0;
        inflator.n_points = 2;
        inflator.time_pts[0] = 0.0; inflator.time_pts[1] = 1.0;
        inflator.mass_rate_curve[0] = 2.0; inflator.mass_rate_curve[1] = 2.0;
        inflator.temp_curve[0] = 600.0; inflator.temp_curve[1] = 600.0;

        Real dm_dt, T_inj;
        inj.compute_injection(inflator, 0.5, dm_dt, T_inj);
        CHECK_NEAR(dm_dt, 2.0, 1.0e-10, "Injection: constant mass rate = 2 kg/s");
        CHECK_NEAR(T_inj, 600.0, 1.0e-10, "Injection: constant temperature = 600 K");
    }

    // 2c. Interpolated mass rate
    {
        AirbagInjection inj;
        InflatorData inflator;
        inflator.t_start = 0.0;
        inflator.n_points = 3;
        inflator.time_pts[0] = 0.0; inflator.time_pts[1] = 0.5; inflator.time_pts[2] = 1.0;
        inflator.mass_rate_curve[0] = 0.0; inflator.mass_rate_curve[1] = 4.0;
        inflator.mass_rate_curve[2] = 0.0;
        inflator.temp_curve[0] = 400.0; inflator.temp_curve[1] = 400.0;
        inflator.temp_curve[2] = 400.0;

        Real dm_dt, T_inj;
        inj.compute_injection(inflator, 0.25, dm_dt, T_inj);
        CHECK_NEAR(dm_dt, 2.0, 1.0e-10, "Injection: interpolated rate at t=0.25");
    }

    // 2d. Mass conservation during injection
    {
        AirbagInjection inj;
        AirbagChamber ch;
        ch.volume = 0.01;
        ch.mass = 0.01;
        ch.temperature = 300.0;
        ch.pressure = 101325.0;

        Real dm = 0.005;
        Real T_inj = 500.0;
        Real R_gas = 287.0;
        Real gamma = 1.4;

        Real m_before = ch.mass;
        inj.inject_direct(ch, dm, T_inj, R_gas, gamma);

        CHECK_NEAR(ch.mass, m_before + dm, 1.0e-15,
                   "Injection: mass conservation (m_new = m_old + dm)");
    }

    // 2e. Temperature mixing: hot gas injected into cool chamber
    {
        AirbagInjection inj;
        AirbagChamber ch;
        ch.volume = 0.01;
        ch.mass = 0.01;
        ch.temperature = 300.0;

        Real dm = 0.01;  // equal mass injected
        Real T_inj = 600.0;
        Real R_gas = 287.0;
        Real gamma = 1.4;

        inj.inject_direct(ch, dm, T_inj, R_gas, gamma);

        // Expected: T_mix = (m1*T1 + m2*T2) / (m1+m2) = (0.01*300 + 0.01*600) / 0.02 = 450
        CHECK_NEAR(ch.temperature, 450.0, 0.1,
                   "Injection: temperature mixing (equal mass -> average)");
    }

    // 2f. Pressure update after injection
    {
        AirbagInjection inj;
        AirbagChamber ch;
        ch.volume = 0.01;
        ch.mass = 0.01;
        ch.temperature = 300.0;
        Real R_gas = 287.0;
        Real gamma = 1.4;
        ch.pressure = ch.mass * R_gas * ch.temperature / ch.volume;
        Real P_before = ch.pressure;

        inj.inject_direct(ch, 0.005, 400.0, R_gas, gamma);

        CHECK(ch.pressure > P_before, "Injection: pressure increases after injection");
    }

    // 2g. Zero mass injection does nothing
    {
        AirbagInjection inj;
        AirbagChamber ch;
        ch.volume = 0.01;
        ch.mass = 0.01;
        ch.temperature = 300.0;
        ch.pressure = 101325.0;
        Real T_before = ch.temperature;

        inj.inject_direct(ch, 0.0, 500.0, 287.0, 1.4);
        CHECK_NEAR(ch.temperature, T_before, 1.0e-15,
                   "Injection: zero mass -> no temperature change");
    }

    // 2h. Pressure consistent with ideal gas after injection
    {
        AirbagInjection inj;
        AirbagChamber ch;
        ch.volume = 0.005;
        ch.mass = 0.008;
        ch.temperature = 350.0;
        Real R_gas = 287.0;
        Real gamma = 1.4;

        inj.inject_direct(ch, 0.003, 450.0, R_gas, gamma);

        Real P_ideal = ch.mass * R_gas * ch.temperature / ch.volume;
        CHECK_NEAR(ch.pressure, P_ideal, 1.0,
                   "Injection: P consistent with ideal gas after inject");
    }
}

// ============================================================================
// 3. AirbagVenting Tests
// ============================================================================

void test_3_airbag_venting() {
    std::cout << "--- Test 3: AirbagVenting ---\n";

    // 3a. No venting when P <= P_external
    {
        AirbagVenting vent;
        AirbagChamber ch;
        ch.volume = 0.01;
        ch.mass = 0.005;
        ch.temperature = 300.0;
        ch.pressure = 90000.0;  // below atmospheric

        VentData vents[1];
        vents[0].area = 0.001;
        vents[0].discharge_coeff = 0.6;
        vents[0].opening_pressure = 0.0;
        vents[0].is_open = true;

        Real dm = vent.compute_venting(ch, vents, 1, 101325.0, 0.0, 0.0, 287.0, 1.0e-4);
        CHECK_NEAR(dm, 0.0, 1.0e-15, "Venting: no flow when P < P_ext");
    }

    // 3b. Vent opens when pressure exceeds opening pressure
    {
        AirbagVenting vent;
        AirbagChamber ch;
        ch.volume = 0.01;
        ch.mass = 0.05;
        ch.temperature = 400.0;
        ch.pressure = 200000.0;

        VentData vents[1];
        vents[0].area = 0.001;
        vents[0].discharge_coeff = 0.6;
        vents[0].opening_pressure = 150000.0;
        vents[0].is_open = false;

        vent.compute_venting(ch, vents, 1, 101325.0, 0.0, 0.0, 287.0, 1.0e-4);
        CHECK(vents[0].is_open, "Venting: vent opens when P > opening_pressure");
    }

    // 3c. Vent stays closed below opening pressure
    {
        AirbagVenting vent;
        AirbagChamber ch;
        ch.volume = 0.01;
        ch.mass = 0.05;
        ch.temperature = 400.0;
        ch.pressure = 120000.0;

        VentData vents[1];
        vents[0].area = 0.001;
        vents[0].discharge_coeff = 0.6;
        vents[0].opening_pressure = 150000.0;
        vents[0].is_open = false;

        vent.compute_venting(ch, vents, 1, 101325.0, 0.0, 0.0, 287.0, 1.0e-4);
        CHECK(!vents[0].is_open, "Venting: vent stays closed when P < opening_pressure");
    }

    // 3d. Mass decreases with venting
    {
        AirbagVenting vent;
        AirbagChamber ch;
        ch.volume = 0.01;
        ch.mass = 0.05;
        ch.temperature = 400.0;
        ch.pressure = ch.mass * 287.0 * ch.temperature / ch.volume;

        VentData vents[1];
        vents[0].area = 0.001;
        vents[0].discharge_coeff = 0.6;
        vents[0].opening_pressure = 0.0;
        vents[0].is_open = true;

        Real m_before = ch.mass;
        vent.compute_venting(ch, vents, 1, 101325.0, 0.0, 0.0, 287.0, 1.0e-3);
        CHECK(ch.mass < m_before, "Venting: mass decreases after venting");
    }

    // 3e. Pressure decreases after venting
    {
        AirbagVenting vent;
        AirbagChamber ch;
        ch.volume = 0.01;
        ch.mass = 0.05;
        ch.temperature = 400.0;
        ch.pressure = ch.mass * 287.0 * ch.temperature / ch.volume;

        VentData vents[1];
        vents[0].area = 0.001;
        vents[0].discharge_coeff = 0.6;
        vents[0].opening_pressure = 0.0;
        vents[0].is_open = true;

        Real P_before = ch.pressure;
        vent.compute_venting(ch, vents, 1, 101325.0, 0.0, 0.0, 287.0, 1.0e-3);
        CHECK(ch.pressure < P_before, "Venting: pressure decreases after venting");
    }

    // 3f. Discharge coefficient effect: higher Cd -> more flow
    {
        AirbagVenting vent;
        Real dm_low, dm_high;

        // Low Cd
        {
            AirbagChamber ch;
            ch.volume = 0.01;
            ch.mass = 0.05;
            ch.temperature = 400.0;
            ch.pressure = ch.mass * 287.0 * ch.temperature / ch.volume;

            VentData vents[1];
            vents[0].area = 0.001;
            vents[0].discharge_coeff = 0.3;
            vents[0].is_open = true;

            dm_low = vent.compute_venting(ch, vents, 1, 101325.0, 0.0, 0.0, 287.0, 1.0e-3);
        }
        // High Cd
        {
            AirbagChamber ch;
            ch.volume = 0.01;
            ch.mass = 0.05;
            ch.temperature = 400.0;
            ch.pressure = ch.mass * 287.0 * ch.temperature / ch.volume;

            VentData vents[1];
            vents[0].area = 0.001;
            vents[0].discharge_coeff = 0.9;
            vents[0].is_open = true;

            dm_high = vent.compute_venting(ch, vents, 1, 101325.0, 0.0, 0.0, 287.0, 1.0e-3);
        }
        CHECK(dm_high > dm_low, "Venting: higher Cd -> more mass loss");
        // Cd ratio should be proportional: dm_high/dm_low ~ 0.9/0.3 = 3
        CHECK_NEAR(dm_high / dm_low, 3.0, 0.01,
                   "Venting: mass flow proportional to Cd");
    }

    // 3g. Fabric porosity increases mass loss
    {
        AirbagVenting vent;

        AirbagChamber ch1, ch2;
        ch1.volume = ch2.volume = 0.01;
        ch1.mass = ch2.mass = 0.05;
        ch1.temperature = ch2.temperature = 400.0;
        ch1.pressure = ch1.mass * 287.0 * ch1.temperature / ch1.volume;
        ch2.pressure = ch1.pressure;

        VentData v1[1], v2[1];
        v1[0].area = 0.001; v1[0].discharge_coeff = 0.6; v1[0].is_open = true;
        v2[0].area = 0.001; v2[0].discharge_coeff = 0.6; v2[0].is_open = true;

        Real dm_no_porous = vent.compute_venting(ch1, v1, 1, 101325.0,
                                                  0.0, 0.0, 287.0, 1.0e-3);
        Real dm_with_porous = vent.compute_venting(ch2, v2, 1, 101325.0,
                                                    0.1, 1.0e-8, 287.0, 1.0e-3);
        CHECK(dm_with_porous > dm_no_porous,
              "Venting: porosity increases total mass loss");
    }

    // 3h. Vented mass is non-negative
    {
        AirbagVenting vent;
        AirbagChamber ch;
        ch.volume = 0.01;
        ch.mass = 0.05;
        ch.temperature = 400.0;
        ch.pressure = ch.mass * 287.0 * ch.temperature / ch.volume;

        VentData vents[1];
        vents[0].area = 0.001;
        vents[0].discharge_coeff = 0.6;
        vents[0].is_open = true;

        Real dm = vent.compute_venting(ch, vents, 1, 101325.0, 0.0, 0.0, 287.0, 1.0e-3);
        CHECK(dm >= 0.0, "Venting: vented mass is non-negative");
        CHECK(ch.mass >= 0.0, "Venting: remaining mass is non-negative");
    }
}

// ============================================================================
// 4. AirbagFolding Tests
// ============================================================================

void test_4_airbag_folding() {
    std::cout << "--- Test 4: AirbagFolding ---\n";

    // Setup: 4 nodes on a flat surface
    const int n_nodes = 4;
    Real flat[12] = {
        0.0, 0.0, 0.0,  // node 0
        1.0, 0.0, 0.0,  // node 1
        0.0, 1.0, 0.0,  // node 2
        1.0, 1.0, 0.0   // node 3
    };

    // 4a. Unfolded bag matches flat coords when no folds
    {
        Real coords[12], flat_ref[12], folded_ref[12];
        std::memcpy(flat_ref, flat, sizeof(flat));

        FoldedBag bag;
        bag.node_coords = coords;
        bag.flat_coords = flat_ref;
        bag.folded_coords = folded_ref;
        bag.n_nodes = n_nodes;
        bag.n_folds = 0;

        AirbagFolding folding;
        folding.initialize_folded(bag, flat);

        CHECK(bag.fully_unfolded, "Folding: no folds -> fully unfolded");
        CHECK_NEAR(coords[0], 0.0, 1.0e-15, "Folding: coords unchanged with no folds");
    }

    // 4b. Single fold reflects nodes across plane
    {
        Real coords[12], flat_ref[12], folded_ref[12];
        std::memcpy(flat_ref, flat, sizeof(flat));

        FoldedBag bag;
        bag.node_coords = coords;
        bag.flat_coords = flat_ref;
        bag.folded_coords = folded_ref;
        bag.n_nodes = n_nodes;
        bag.n_folds = 1;

        // Fold along x=0.5 plane (normal in x-direction)
        bag.fold_sequence[0].axis_point[0] = 0.5;
        bag.fold_sequence[0].axis_point[1] = 0.0;
        bag.fold_sequence[0].axis_point[2] = 0.0;
        bag.fold_sequence[0].axis_dir[0] = 1.0;
        bag.fold_sequence[0].axis_dir[1] = 0.0;
        bag.fold_sequence[0].axis_dir[2] = 0.0;

        AirbagFolding folding;
        folding.initialize_folded(bag, flat);

        // Node 1 (x=1.0) and node 3 (x=1.0) should be reflected to x=0.0
        // Reflection: x_new = x - 2*(x-0.5)*1 = x - 2x + 1 = 1 - x
        // Node 1: x=1.0 -> x=0.0
        CHECK_NEAR(coords[3], 0.0, 1.0e-10, "Folding: node 1 reflected across x=0.5");
        CHECK(!bag.fully_unfolded, "Folding: bag is folded");
    }

    // 4c. Folded bag has smaller extent
    {
        Real coords[12], flat_ref[12], folded_ref[12];
        std::memcpy(flat_ref, flat, sizeof(flat));

        FoldedBag bag;
        bag.node_coords = coords;
        bag.flat_coords = flat_ref;
        bag.folded_coords = folded_ref;
        bag.n_nodes = n_nodes;
        bag.n_folds = 1;
        bag.fold_sequence[0].axis_point[0] = 0.5;
        bag.fold_sequence[0].axis_dir[0] = 1.0;

        AirbagFolding folding;
        folding.initialize_folded(bag, flat);

        // All x-coords should be <= 0.5 after folding
        bool all_within = true;
        for (int i = 0; i < n_nodes; ++i) {
            if (coords[3 * i] > 0.5 + 1.0e-10) all_within = false;
        }
        CHECK(all_within, "Folding: all nodes within fold boundary");
    }

    // 4d. Min distance decreases after folding
    {
        Real coords[12], flat_ref[12], folded_ref[12];
        std::memcpy(flat_ref, flat, sizeof(flat));

        FoldedBag bag_flat, bag_folded;
        Real coords_flat[12];
        std::memcpy(coords_flat, flat, sizeof(flat));
        bag_flat.node_coords = coords_flat;
        bag_flat.n_nodes = n_nodes;

        bag_folded.node_coords = coords;
        bag_folded.flat_coords = flat_ref;
        bag_folded.folded_coords = folded_ref;
        bag_folded.n_nodes = n_nodes;
        bag_folded.n_folds = 1;
        bag_folded.fold_sequence[0].axis_point[0] = 0.5;
        bag_folded.fold_sequence[0].axis_dir[0] = 1.0;

        AirbagFolding folding;
        Real d_flat = folding.compute_min_distance(bag_flat);
        folding.initialize_folded(bag_folded, flat);
        Real d_folded = folding.compute_min_distance(bag_folded);

        CHECK(d_folded <= d_flat + 1.0e-10,
              "Folding: min distance decreases after folding");
    }

    // 4e. Unfold step progresses
    {
        Real coords[12], flat_ref[12], folded_ref[12];
        std::memcpy(flat_ref, flat, sizeof(flat));

        FoldedBag bag;
        bag.node_coords = coords;
        bag.flat_coords = flat_ref;
        bag.folded_coords = folded_ref;
        bag.n_nodes = n_nodes;
        bag.n_folds = 1;
        bag.fold_sequence[0].axis_point[0] = 0.5;
        bag.fold_sequence[0].axis_dir[0] = 1.0;

        AirbagFolding folding;
        folding.initialize_folded(bag, flat);

        // Take a small unfold step
        folding.unfold_step(bag, 0.001);  // 0.1 progress (rate=100)
        CHECK(!bag.fully_unfolded, "Folding: partial unfold not complete");
    }

    // 4f. Full unfolding reaches flat configuration
    {
        Real coords[12], flat_ref[12], folded_ref[12];
        std::memcpy(flat_ref, flat, sizeof(flat));

        FoldedBag bag;
        bag.node_coords = coords;
        bag.flat_coords = flat_ref;
        bag.folded_coords = folded_ref;
        bag.n_nodes = n_nodes;
        bag.n_folds = 1;
        bag.fold_sequence[0].axis_point[0] = 0.5;
        bag.fold_sequence[0].axis_dir[0] = 1.0;

        AirbagFolding folding;
        folding.initialize_folded(bag, flat);

        // Run enough steps to fully unfold
        for (int i = 0; i < 200; ++i) {
            folding.unfold_step(bag, 0.001);
        }
        CHECK(bag.fully_unfolded, "Folding: bag fully unfolded after sufficient steps");
    }

    // 4g. Fully unfolded coords match flat coords
    {
        Real coords[12], flat_ref[12], folded_ref[12];
        std::memcpy(flat_ref, flat, sizeof(flat));

        FoldedBag bag;
        bag.node_coords = coords;
        bag.flat_coords = flat_ref;
        bag.folded_coords = folded_ref;
        bag.n_nodes = n_nodes;
        bag.n_folds = 1;
        bag.fold_sequence[0].axis_point[0] = 0.5;
        bag.fold_sequence[0].axis_dir[0] = 1.0;

        AirbagFolding folding;
        folding.initialize_folded(bag, flat);

        for (int i = 0; i < 200; ++i) {
            folding.unfold_step(bag, 0.001);
        }

        Real max_err = 0.0;
        for (int i = 0; i < 3 * n_nodes; ++i) {
            Real err = std::fabs(coords[i] - flat[i]);
            if (err > max_err) max_err = err;
        }
        CHECK(max_err < 1.0e-10, "Folding: unfolded coords match flat coords");
    }

    // 4h. Multiple folds
    {
        Real coords[12], flat_ref[12], folded_ref[12];
        std::memcpy(flat_ref, flat, sizeof(flat));

        FoldedBag bag;
        bag.node_coords = coords;
        bag.flat_coords = flat_ref;
        bag.folded_coords = folded_ref;
        bag.n_nodes = n_nodes;
        bag.n_folds = 2;
        // Fold 1: along x=0.5
        bag.fold_sequence[0].axis_point[0] = 0.5;
        bag.fold_sequence[0].axis_dir[0] = 1.0;
        // Fold 2: along y=0.5
        bag.fold_sequence[1].axis_point[1] = 0.5;
        bag.fold_sequence[1].axis_dir[1] = 1.0;

        AirbagFolding folding;
        folding.initialize_folded(bag, flat);

        // Should have 2 folds to undo
        CHECK(bag.current_fold == 1, "Folding: 2 folds -> current_fold starts at 1");
    }
}

// ============================================================================
// 5. AirbagThermal Tests
// ============================================================================

void test_5_airbag_thermal() {
    std::cout << "--- Test 5: AirbagThermal ---\n";

    // 5a. Mixture pressure: P = n*R*T/V
    {
        AirbagThermal therm;
        GasSpecies sp[1];
        sp[0].moles = 1.0;  // 1 mol
        sp[0].molar_mass = 0.029;

        Real T = 300.0;
        Real V = 0.01;
        Real P = therm.mixture_pressure(sp, 1, T, V);
        Real P_expected = 1.0 * 8.314 * 300.0 / 0.01;  // = 249420
        CHECK_NEAR(P, P_expected, 1.0, "Thermal: mixture pressure P = nRT/V");
    }

    // 5b. Multi-species pressure
    {
        AirbagThermal therm;
        GasSpecies sp[2];
        sp[0].moles = 0.5;
        sp[0].molar_mass = 0.028;  // N2
        sp[1].moles = 0.3;
        sp[1].molar_mass = 0.032;  // O2

        Real T = 350.0;
        Real V = 0.005;
        Real P = therm.mixture_pressure(sp, 2, T, V);
        Real P_expected = (0.5 + 0.3) * 8.314 * 350.0 / 0.005;
        CHECK_NEAR(P, P_expected, 1.0, "Thermal: multi-species pressure");
    }

    // 5c. Mixture Cv: mass-weighted average
    {
        AirbagThermal therm;
        GasSpecies sp[2];
        sp[0].moles = 1.0;
        sp[0].molar_mass = 0.028;
        sp[0].Cv = 743.0;  // N2
        sp[1].moles = 1.0;
        sp[1].molar_mass = 0.032;
        sp[1].Cv = 659.0;  // O2

        Real Cv = therm.mixture_Cv(sp, 2);
        // mass-weighted: (0.028*743 + 0.032*659) / (0.028+0.032)
        Real m1 = 0.028, m2 = 0.032;
        Real Cv_expected = (m1 * 743.0 + m2 * 659.0) / (m1 + m2);
        CHECK_NEAR(Cv, Cv_expected, 0.1, "Thermal: mixture Cv mass-weighted");
    }

    // 5d. Thermal decay toward wall temperature
    {
        AirbagThermal therm;
        AirbagChamber ch;
        ch.mass = 0.01;
        ch.temperature = 600.0;
        ch.volume = 0.01;

        Real T_wall = 300.0;
        Real h_conv = 50.0;  // W/(m^2*K)
        Real A_wall = 0.1;   // m^2
        Real Cv = 718.0;

        Real T0 = ch.temperature;
        // Run 100 steps
        for (int i = 0; i < 100; ++i) {
            therm.compute_thermal(ch, h_conv, A_wall, T_wall, Cv, 0.01);
        }
        // Temperature should have moved toward T_wall
        CHECK(ch.temperature < T0, "Thermal: temperature decreases toward wall");
        CHECK(ch.temperature > T_wall, "Thermal: temperature still above wall");
    }

    // 5e. Equilibrium temperature equals wall temperature
    {
        AirbagThermal therm;
        Real T_eq = therm.equilibrium_temperature(350.0);
        CHECK_NEAR(T_eq, 350.0, 1.0e-10, "Thermal: equilibrium T = T_wall");
    }

    // 5f. Thermal time constant: tau = m*Cv/(h*A)
    {
        AirbagThermal therm;
        Real mass = 0.01;
        Real Cv = 718.0;
        Real h = 50.0;
        Real A = 0.1;
        Real tau = therm.thermal_time_constant(mass, Cv, h, A);
        Real tau_expected = mass * Cv / (h * A);
        CHECK_NEAR(tau, tau_expected, 1.0e-10, "Thermal: time constant = m*Cv/(h*A)");
    }

    // 5g. Heat loss with convection only
    {
        AirbagThermal therm;
        Real T = 500.0, T_wall = 300.0;
        Real h = 50.0, A = 0.1;
        Real Q = therm.compute_heat_loss(T, T_wall, h, A, 0.0);  // no radiation
        Real Q_expected = h * A * (T - T_wall);
        CHECK_NEAR(Q, Q_expected, 1.0e-6, "Thermal: convective heat loss");
    }

    // 5h. Long-time convergence to equilibrium
    {
        AirbagThermal therm;
        AirbagChamber ch;
        ch.mass = 0.01;
        ch.temperature = 800.0;
        ch.volume = 0.01;

        Real T_wall = 300.0;
        Real h_conv = 100.0;
        Real A_wall = 0.2;
        Real Cv = 718.0;

        // Run many steps with small dt
        for (int i = 0; i < 10000; ++i) {
            therm.compute_thermal(ch, h_conv, A_wall, T_wall, Cv, 0.001);
        }
        // Should be very close to T_wall
        CHECK_NEAR(ch.temperature, T_wall, 1.0, "Thermal: long-time convergence to T_wall");
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Wave 38: Airbag Production Test Suite ===\n\n";

    test_1_fvbag_solver();
    test_2_airbag_injection();
    test_3_airbag_venting();
    test_4_airbag_folding();
    test_5_airbag_thermal();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " ===\n";

    return tests_failed > 0 ? 1 : 0;
}
