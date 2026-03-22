/**
 * @file airbag_wave41_test.cpp
 * @brief Wave 41: Airbag Extensions Test Suite (4 features, 30 tests)
 *
 * Tests:
 *   1. AirbagMultiChamber    (8 tests)
 *   2. AirbagGasSpecies      (8 tests)
 *   3. AirbagTTF             (7 tests)
 *   4. AirbagMembraneDrape   (7 tests)
 */

#include <nexussim/fem/airbag_wave41.hpp>
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
// 1. AirbagMultiChamber Tests
// ============================================================================

void test_1_multi_chamber() {
    std::cout << "--- Test 1: AirbagMultiChamber ---\n";

    // 1a. Single chamber: P = m*R*T / V
    {
        AirbagMultiChamber mc;
        int c0 = mc.add_chamber(0.01, 101325.0, 300.0, 1.4, 287.0);
        // mass = P*V/(R*T) = 101325*0.01/(287*300) = 11.766...
        Real expected_mass = 101325.0 * 0.01 / (287.0 * 300.0);
        CHECK_NEAR(mc.chamber(c0).mass, expected_mass, 0.01,
                   "MultiChamber: initial mass from ideal gas law");
    }

    // 1b. Two chambers with orifice: mass flows from high to low pressure
    {
        AirbagMultiChamber mc;
        int c0 = mc.add_chamber(0.01, 500000.0, 400.0, 1.4, 287.0);
        int c1 = mc.add_chamber(0.01, 101325.0, 300.0, 1.4, 287.0);
        mc.add_orifice(c0, c1, 1.0e-4, 0.65, false);

        Real m0_before = mc.chamber(c0).mass;
        Real m1_before = mc.chamber(c1).mass;

        mc.step(1.0e-5);

        CHECK(mc.chamber(c0).mass < m0_before, "MultiChamber: high-P loses mass");
        CHECK(mc.chamber(c1).mass > m1_before, "MultiChamber: low-P gains mass");
    }

    // 1c. Mass conservation across chambers
    {
        AirbagMultiChamber mc;
        int c0 = mc.add_chamber(0.01, 500000.0, 400.0, 1.4, 287.0);
        int c1 = mc.add_chamber(0.01, 101325.0, 300.0, 1.4, 287.0);
        mc.add_orifice(c0, c1, 2.0e-3, 0.65, false);

        Real total_before = mc.chamber(c0).mass + mc.chamber(c1).mass;
        mc.step(1.0e-5);
        Real total_after = mc.chamber(c0).mass + mc.chamber(c1).mass;

        CHECK_NEAR(total_after, total_before, 1.0e-12,
                   "MultiChamber: mass conservation");
    }

    // 1d. Check valve blocks reverse flow
    {
        AirbagMultiChamber mc;
        // c0 low pressure, c1 high pressure, check valve allows only 0->1
        int c0 = mc.add_chamber(0.01, 101325.0, 300.0, 1.4, 287.0);
        int c1 = mc.add_chamber(0.01, 500000.0, 400.0, 1.4, 287.0);
        mc.add_orifice(c0, c1, 1.0e-4, 0.65, true);

        Real m0_before = mc.chamber(c0).mass;
        Real m1_before = mc.chamber(c1).mass;
        mc.step(1.0e-5);

        // Check valve blocks flow from c1 to c0 (reverse direction)
        CHECK_NEAR(mc.chamber(c0).mass, m0_before, 1.0e-15,
                   "MultiChamber: check valve blocks reverse flow (source unchanged)");
        CHECK_NEAR(mc.chamber(c1).mass, m1_before, 1.0e-15,
                   "MultiChamber: check valve blocks reverse flow (dest unchanged)");
    }

    // 1e. Pressure equilibration over many steps
    {
        AirbagMultiChamber mc;
        int c0 = mc.add_chamber(0.01, 500000.0, 350.0, 1.4, 287.0);
        int c1 = mc.add_chamber(0.01, 101325.0, 350.0, 1.4, 287.0);
        mc.add_orifice(c0, c1, 5.0e-3, 0.65, false);

        for (int i = 0; i < 5000; ++i) {
            mc.step(1.0e-5);
        }

        Real p_diff = std::abs(mc.chamber(c0).P - mc.chamber(c1).P);
        Real p_avg = 0.5 * (mc.chamber(c0).P + mc.chamber(c1).P);
        CHECK(p_diff / p_avg < 0.1,
              "MultiChamber: pressures equilibrate over many steps");
    }

    // 1f. Inject mass increases chamber pressure
    {
        AirbagMultiChamber mc;
        int c0 = mc.add_chamber(0.01, 101325.0, 300.0, 1.4, 287.0);
        Real P_before = mc.chamber(c0).P;
        mc.inject(c0, 0.01, 500.0);
        CHECK(mc.chamber(c0).P > P_before,
              "MultiChamber: injection raises pressure");
    }

    // 1g. set_volume: larger volume lowers pressure
    {
        AirbagMultiChamber mc;
        int c0 = mc.add_chamber(0.01, 200000.0, 300.0, 1.4, 287.0);
        Real P_small_vol = mc.chamber(c0).P;
        mc.set_volume(c0, 0.02);
        Real P_large_vol = mc.chamber(c0).P;
        CHECK(P_large_vol < P_small_vol,
              "MultiChamber: larger volume -> lower pressure");
    }

    // 1h. Time advances with each step
    {
        AirbagMultiChamber mc;
        mc.add_chamber(0.01, 101325.0, 300.0);
        CHECK_NEAR(mc.time(), 0.0, 1e-15, "MultiChamber: initial time = 0");
        mc.step(1.0e-4);
        CHECK_NEAR(mc.time(), 1.0e-4, 1e-15, "MultiChamber: time advances");
    }
}

// ============================================================================
// 2. AirbagGasSpecies Tests
// ============================================================================

void test_2_gas_species() {
    std::cout << "\n--- Test 2: AirbagGasSpecies ---\n";

    // 2a. Single species: mixture properties equal species properties
    {
        AirbagGasSpecies gas;
        gas.add_species("N2", 28.014e-3, 1040.0, 743.0);

        auto props = gas.mix({1.0});
        CHECK_NEAR(props.cp_mix, 1040.0, 0.01,
                   "GasSpecies: single species cp = 1040");
        CHECK_NEAR(props.cv_mix, 743.0, 0.01,
                   "GasSpecies: single species cv = 743");
    }

    // 2b. Binary mixture cp (mass-weighted)
    {
        AirbagGasSpecies gas;
        gas.add_species("N2", 28.014e-3, 1040.0, 743.0);
        gas.add_species("O2", 32.0e-3, 920.0, 660.0);

        auto props = gas.mix({0.79, 0.21});
        Real expected_cp = 0.79 * 1040.0 + 0.21 * 920.0;
        CHECK_NEAR(props.cp_mix, expected_cp, 0.1,
                   "GasSpecies: binary mixture cp");
    }

    // 2c. 3-species mixing
    {
        AirbagGasSpecies gas;
        gas.add_species("N2",  28.0e-3, 1040.0, 743.0);
        gas.add_species("O2",  32.0e-3, 920.0,  660.0);
        gas.add_species("CO2", 44.0e-3, 840.0,  650.0);

        auto props = gas.mix({0.5, 0.3, 0.2});
        Real expected_cp = 0.5*1040.0 + 0.3*920.0 + 0.2*840.0;
        CHECK_NEAR(props.cp_mix, expected_cp, 0.1,
                   "GasSpecies: 3-species cp mixing");
    }

    // 2d. Gamma = cp/cv for single species
    {
        AirbagGasSpecies gas;
        gas.add_species("N2", 28.014e-3, 1040.0, 743.0);

        auto props = gas.mix({1.0});
        CHECK_NEAR(props.gamma_mix, 1040.0 / 743.0, 0.001,
                   "GasSpecies: gamma = cp/cv");
    }

    // 2e. R_mix = cp - cv
    {
        AirbagGasSpecies gas;
        gas.add_species("N2", 28.014e-3, 1040.0, 743.0);

        auto props = gas.mix({1.0});
        CHECK_NEAR(props.R_mix, 297.0, 0.1,
                   "GasSpecies: R_mix = cp - cv");
    }

    // 2f. add_common_species populates 5 species
    {
        AirbagGasSpecies gas;
        gas.add_common_species();
        CHECK(gas.num_species() == 5, "GasSpecies: add_common_species adds 5");
    }

    // 2g. Mixture gamma for binary
    {
        AirbagGasSpecies gas;
        gas.add_species("N2", 28.014e-3, 1040.0, 743.0);
        gas.add_species("O2", 32.0e-3,   920.0,  660.0);

        auto props = gas.mix({0.79, 0.21});
        Real cp_mix = 0.79*1040.0 + 0.21*920.0;
        Real cv_mix = 0.79*743.0  + 0.21*660.0;
        Real expected_gamma = cp_mix / cv_mix;
        CHECK_NEAR(props.gamma_mix, expected_gamma, 0.001,
                   "GasSpecies: mixture gamma");
    }

    // 2h. mix_molar: mole-fraction based mixing
    {
        AirbagGasSpecies gas;
        gas.add_species("N2", 28.014e-3, 1040.0, 743.0);
        gas.add_species("Ar", 39.948e-3, 520.3,  312.2);

        // Equal mole fractions
        auto props = gas.mix_molar({0.5, 0.5});
        // Mole-to-mass: Y_N2 = 0.5*28.014e-3 / (0.5*28.014e-3 + 0.5*39.948e-3)
        Real total_m = 0.5*28.014e-3 + 0.5*39.948e-3;
        Real Y_N2 = 0.5*28.014e-3 / total_m;
        Real Y_Ar = 0.5*39.948e-3 / total_m;
        Real expected_cp = Y_N2*1040.0 + Y_Ar*520.3;
        CHECK_NEAR(props.cp_mix, expected_cp, 0.1,
                   "GasSpecies: mix_molar cp");
    }
}

// ============================================================================
// 3. AirbagTTF Tests
// ============================================================================

void test_3_ttf() {
    std::cout << "\n--- Test 3: AirbagTTF ---\n";

    // 3a. Load TTF data and check loaded flag
    {
        AirbagTTF ttf;
        CHECK(!ttf.is_loaded(), "TTF: not loaded initially");

        std::vector<Real> mf_t = {0.0, 0.01, 0.02, 0.03, 0.04};
        std::vector<Real> mf_r = {0.0, 0.5,  1.0,  0.5,  0.0};
        std::vector<Real> t_t  = {0.0, 0.01, 0.02, 0.03, 0.04};
        std::vector<Real> t_v  = {300.0, 500.0, 600.0, 500.0, 400.0};
        ttf.load_ttf(mf_t, mf_r, t_t, t_v);
        CHECK(ttf.is_loaded(), "TTF: loaded after load_ttf");
    }

    // 3b. Interpolation at midpoint
    {
        AirbagTTF ttf;
        std::vector<Real> mf_t = {0.0, 0.01, 0.02};
        std::vector<Real> mf_r = {0.0, 1.0,  0.0};
        std::vector<Real> t_t  = {0.0, 0.02};
        std::vector<Real> t_v  = {300.0, 300.0};
        ttf.load_ttf(mf_t, mf_r, t_t, t_v);

        Real val = ttf.get_mass_flow(0.005);
        CHECK_NEAR(val, 0.5, 0.01, "TTF: interpolation at t=0.005");
    }

    // 3c. Clamped to first value before curve start
    {
        AirbagTTF ttf;
        std::vector<Real> mf_t = {0.01, 0.02, 0.03};
        std::vector<Real> mf_r = {1.0, 2.0, 0.0};
        std::vector<Real> t_t  = {0.0};
        std::vector<Real> t_v  = {300.0};
        ttf.load_ttf(mf_t, mf_r, t_t, t_v);

        Real val = ttf.get_mass_flow(0.0);
        CHECK_NEAR(val, 1.0, 1e-10,
                   "TTF: clamped to first value before start");
    }

    // 3d. Clamped to last value after curve end
    {
        AirbagTTF ttf;
        std::vector<Real> mf_t = {0.0, 0.01, 0.02};
        std::vector<Real> mf_r = {0.0, 1.0,  0.0};
        std::vector<Real> t_t  = {0.0};
        std::vector<Real> t_v  = {300.0};
        ttf.load_ttf(mf_t, mf_r, t_t, t_v);

        Real val = ttf.get_mass_flow(0.05);
        CHECK_NEAR(val, 0.0, 1e-10, "TTF: clamped to last value after end");
    }

    // 3e. Temperature interpolation
    {
        AirbagTTF ttf;
        std::vector<Real> mf_t = {0.0, 0.02};
        std::vector<Real> mf_r = {1.0, 0.0};
        std::vector<Real> t_t  = {0.0, 0.01, 0.02};
        std::vector<Real> t_v  = {300.0, 600.0, 400.0};
        ttf.load_ttf(mf_t, mf_r, t_t, t_v);

        Real T = ttf.get_temperature(0.005);
        CHECK_NEAR(T, 450.0, 1.0, "TTF: temperature interpolation at midpoint");
    }

    // 3f. Total mass (trapezoidal integration)
    {
        AirbagTTF ttf;
        // Triangle: 0->1->0 over 0.02s => total = 0.5 * 0.02 * 1.0 = 0.01
        std::vector<Real> mf_t = {0.0, 0.01, 0.02};
        std::vector<Real> mf_r = {0.0, 1.0,  0.0};
        std::vector<Real> t_t  = {0.0};
        std::vector<Real> t_v  = {300.0};
        ttf.load_ttf(mf_t, mf_r, t_t, t_v);

        CHECK_NEAR(ttf.total_mass(), 0.01, 1e-10,
                   "TTF: total mass = 0.01 for triangular pulse");
    }

    // 3g. Burnout time
    {
        AirbagTTF ttf;
        std::vector<Real> mf_t = {0.0, 0.005, 0.01, 0.015, 0.02};
        std::vector<Real> mf_r = {0.0, 0.8,   2.5,  1.0,   0.0};
        std::vector<Real> t_t  = {0.0};
        std::vector<Real> t_v  = {300.0};
        ttf.load_ttf(mf_t, mf_r, t_t, t_v);

        Real bt = ttf.burnout_time(0.5);
        // Last rate > 0.5 is at t=0.015 (rate=1.0)
        CHECK_NEAR(bt, 0.015, 1e-10, "TTF: burnout time at threshold 0.5");
    }
}

// ============================================================================
// 4. AirbagMembraneDrape Tests
// ============================================================================

void test_4_membrane_drape() {
    std::cout << "\n--- Test 4: AirbagMembraneDrape ---\n";

    // Helper: create a simple 2-triangle mesh (4 nodes, 2 triangles)
    // Nodes: 0=(0,0,1), 1=(1,0,1), 2=(0,1,1), 3=(1,1,1)
    // Triangles: (0,1,2), (1,3,2)

    // 4a. Drape under gravity onto z=0 plane
    {
        AirbagMembraneDrape draper;
        std::vector<Real> nodes = {
            0.0, 0.0, 1.0,   // node 0
            1.0, 0.0, 1.0,   // node 1
            0.0, 1.0, 1.0,   // node 2
            1.0, 1.0, 1.0    // node 3
        };
        std::vector<int> elems = {0, 1, 2,  1, 3, 2};
        Real gravity[3] = {0.0, 0.0, -9.81};
        AirbagMembraneDrape::ToolSurface tool;
        tool.z_plane = 0.0;

        auto result = draper.drape(nodes, elems, 4, 2, gravity, tool, 5000, 1.0e-8);
        // Nodes should have moved down toward z=0
        CHECK(result.max_displacement > 0.0,
              "MembraneDrape: nodes displaced under gravity");
        // All nodes should be at or above the tool plane
        for (int i = 0; i < 4; ++i) {
            CHECK(result.deformed_positions[i][2] >= -1e-10,
                  "MembraneDrape: node respects tool plane contact");
        }
    }

    // 4b. Drape converges
    {
        AirbagMembraneDrape draper;
        std::vector<Real> nodes = {
            0.0, 0.0, 0.5,
            1.0, 0.0, 0.5,
            0.5, 0.5, 0.5
        };
        std::vector<int> elems = {0, 1, 2};
        Real gravity[3] = {0.0, 0.0, -9.81};
        AirbagMembraneDrape::ToolSurface tool;
        tool.z_plane = 0.0;

        auto result = draper.drape(nodes, elems, 3, 1, gravity, tool, 10000, 1.0e-6);
        CHECK(result.converged, "MembraneDrape: drape converges");
    }

    // 4c. Zero gravity: no displacement
    {
        AirbagMembraneDrape draper;
        std::vector<Real> nodes = {
            0.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            0.5, 0.5, 1.0
        };
        std::vector<int> elems = {0, 1, 2};
        Real gravity[3] = {0.0, 0.0, 0.0};
        AirbagMembraneDrape::ToolSurface tool;
        tool.z_plane = 0.0;

        auto result = draper.drape(nodes, elems, 3, 1, gravity, tool, 100, 1.0e-8);
        CHECK_NEAR(result.max_displacement, 0.0, 1.0e-10,
                   "MembraneDrape: zero gravity -> no displacement");
    }

    // 4d. detect_folds: coplanar elements -> no folds
    {
        AirbagMembraneDrape draper;
        std::vector<std::array<Real,3>> positions = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {1.0, 1.0, 0.0}
        };
        std::vector<int> elems = {0, 1, 2,  1, 3, 2};

        auto folds = draper.detect_folds(positions, elems, 2, M_PI / 4.0);
        CHECK(folds.empty(), "MembraneDrape: no folds in flat mesh");
    }

    // 4e. detect_folds: folded mesh -> folds detected
    {
        AirbagMembraneDrape draper;
        // Two triangles sharing edge (0,1), with normals pointing in very
        // different directions (fold angle > 90 degrees)
        std::vector<std::array<Real,3>> positions = {
            {0.0, 0.0, 0.0},   // node 0
            {1.0, 0.0, 0.0},   // node 1
            {0.5, 1.0, 0.0},   // node 2 (flat, normal = +z)
            {0.5, 0.0, -1.0}   // node 3 (folded down, normal ~ -y direction)
        };
        // Triangle 0: (0,1,2) normal ~ +z
        // Triangle 1: (1,0,3) normal ~ +y (perpendicular to first)
        std::vector<int> elems = {0, 1, 2,  1, 0, 3};

        auto folds = draper.detect_folds(positions, elems, 2, M_PI / 4.0);
        CHECK(!folds.empty(), "MembraneDrape: folds detected in folded mesh");
    }

    // 4f. Iteration count is positive
    {
        AirbagMembraneDrape draper;
        std::vector<Real> nodes = {
            0.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            0.5, 0.5, 1.0
        };
        std::vector<int> elems = {0, 1, 2};
        Real gravity[3] = {0.0, 0.0, -9.81};
        AirbagMembraneDrape::ToolSurface tool;
        tool.z_plane = 0.0;

        auto result = draper.drape(nodes, elems, 3, 1, gravity, tool, 100, 1.0e-6);
        CHECK(result.iterations > 0,
              "MembraneDrape: iteration count is positive");
    }

    // 4g. Tool surface with triangles: contact enforced
    {
        AirbagMembraneDrape draper;
        std::vector<Real> nodes = {
            0.0, 0.0, 0.5,
            1.0, 0.0, 0.5,
            0.5, 0.5, 0.5
        };
        std::vector<int> elems = {0, 1, 2};
        Real gravity[3] = {0.0, 0.0, -9.81};
        AirbagMembraneDrape::ToolSurface tool;
        tool.z_plane = -10.0;  // plane far below
        // Add a triangle tool surface at z=0.2
        tool.triangles.push_back({0.0, 0.0, 0.2,
                                  2.0, 0.0, 0.2,
                                  1.0, 2.0, 0.2});

        auto result = draper.drape(nodes, elems, 3, 1, gravity, tool, 5000, 1.0e-8);
        // Nodes should stop at tool triangle z=0.2
        for (int i = 0; i < 3; ++i) {
            CHECK(result.deformed_positions[i][2] >= 0.2 - 1e-10,
                  "MembraneDrape: node respects triangle tool surface");
        }
    }
}

// ============================================================================
// main
// ============================================================================

int main() {
    test_1_multi_chamber();
    test_2_gas_species();
    test_3_ttf();
    test_4_membrane_drape();

    std::cout << "\n" << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    return tests_failed > 0 ? 1 : 0;
}
