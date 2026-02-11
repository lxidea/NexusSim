/**
 * @file tied_contact_eos_test.cpp
 * @brief Comprehensive test for Wave 5: Tied contact and Equation of State
 */

#include <nexussim/fem/tied_contact.hpp>
#include <nexussim/physics/eos.hpp>
#include <iostream>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;
using namespace nxs::physics;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { \
        std::cout << "[PASS] " << msg << "\n"; \
        tests_passed++; \
    } else { \
        std::cout << "[FAIL] " << msg << "\n"; \
        tests_failed++; \
    } \
} while(0)

static bool near(Real a, Real b, Real tol = 1.0e-6) {
    return std::fabs(a - b) < tol * (1.0 + std::fabs(b));
}

// ==========================================================================
// Test 1: Tied Contact - Basic Tying
// ==========================================================================
void test_basic_tying() {
    std::cout << "\n=== Test 1: Basic Tying ===\n";

    TiedContact tc;
    TiedContactConfig cfg;
    cfg.type = TiedContactType::TiedAllDOF;
    cfg.penalty_stiffness = 1.0e10;
    tc.set_config(cfg);

    // Two nodes that should be tied: node 0 (slave) to node 1 (master)
    auto& pair = tc.add_pair();
    pair.slave_node = 0;
    pair.master_nodes[0] = 1;
    pair.num_master_nodes = 1;
    pair.phi[0] = 1.0;
    // Initially coincident
    pair.gap_initial[0] = pair.gap_initial[1] = pair.gap_initial[2] = 0.0;

    CHECK(tc.num_pairs() == 1, "One tied pair");
    CHECK(tc.num_active_pairs() == 1, "One active pair");

    // Slave has moved away from master
    Real positions[6] = {0.001, 0.0, 0.0,  0.0, 0.0, 0.0};
    Real velocities[6] = {0, 0, 0, 0, 0, 0};
    Real forces[6] = {0, 0, 0, 0, 0, 0};

    tc.apply_tied_constraints(positions, velocities, forces, 1.0e-6);

    CHECK(forces[0] < 0.0, "Slave pulled back towards master (-x)");
    CHECK(forces[3] > 0.0, "Master pulled towards slave (+x)");
    CHECK(near(forces[0], -forces[3], 1.0e-3), "Action-reaction balance");
}

// ==========================================================================
// Test 2: Tied Contact - Proximity Search
// ==========================================================================
void test_proximity_search() {
    std::cout << "\n=== Test 2: Proximity Search ===\n";

    TiedContact tc;
    TiedContactConfig cfg;
    cfg.type = TiedContactType::TiedAllDOF;
    cfg.penalty_stiffness = 1.0e10;
    tc.set_config(cfg);

    // 4 nodes: slaves={0,1}, masters={2,3}
    // Node 0 at (0,0,0), Node 1 at (1,0,0)
    // Node 2 at (0.0001,0,0) (close to 0), Node 3 at (0.9999,0,0) (close to 1)
    Real positions[12] = {
        0.0, 0.0, 0.0,       // slave 0
        1.0, 0.0, 0.0,       // slave 1
        0.0001, 0.0, 0.0,    // master 2
        0.9999, 0.0, 0.0     // master 3
    };

    std::vector<Index> slaves = {0, 1};
    std::vector<Index> masters = {2, 3};
    tc.find_tied_pairs(slaves, masters, positions, 0.01);

    CHECK(tc.num_pairs() == 2, "Two pairs found");
    CHECK(tc.pair(0).slave_node == 0, "Slave 0 paired");
    CHECK(tc.pair(0).master_nodes[0] == 2, "Slave 0 paired to master 2");
    CHECK(tc.pair(1).slave_node == 1, "Slave 1 paired");
    CHECK(tc.pair(1).master_nodes[0] == 3, "Slave 1 paired to master 3");
}

// ==========================================================================
// Test 3: Tied Contact - Failure
// ==========================================================================
void test_tied_failure() {
    std::cout << "\n=== Test 3: Tied Contact Failure ===\n";

    TiedContact tc;
    TiedContactConfig cfg;
    cfg.type = TiedContactType::TiedWithFailure;
    cfg.penalty_stiffness = 1.0e8;
    cfg.failure_force = 1000.0;  // 1kN failure
    tc.set_config(cfg);

    auto& pair = tc.add_pair();
    pair.slave_node = 0;
    pair.master_nodes[0] = 1;
    pair.num_master_nodes = 1;
    pair.phi[0] = 1.0;

    // Small separation: force below failure (k*gap = 1e8 * 1e-6 = 100N < 1000N)
    Real pos_small[6] = {1.0e-6, 0.0, 0.0,  0.0, 0.0, 0.0};
    Real vel[6] = {};
    Real forces[6] = {};

    tc.apply_tied_constraints(pos_small, vel, forces, 1.0e-6);
    CHECK(tc.pair(0).active, "Still active after small gap");

    // Large separation: force exceeds failure
    Real pos_large[6] = {0.1, 0.0, 0.0,  0.0, 0.0, 0.0};
    std::fill(forces, forces+6, 0.0);

    tc.apply_tied_constraints(pos_large, vel, forces, 1.0e-6);
    CHECK(!tc.pair(0).active, "Pair failed after large gap");

    auto stats = tc.get_stats();
    CHECK(stats.failed_pairs == 1, "One failed pair");
    CHECK(stats.active_pairs == 0, "No active pairs");
}

// ==========================================================================
// Test 4: Ideal Gas EOS
// ==========================================================================
void test_ideal_gas() {
    std::cout << "\n=== Test 4: Ideal Gas EOS ===\n";

    EOSProperties props;
    props.type = EOSType::IdealGas;
    props.gamma = 1.4;
    props.rho0 = 1.225;  // Air at STP

    // p = (γ-1) * ρ * e
    // At STP: p = 101325 Pa, ρ = 1.225, e = p/((γ-1)*ρ) = 101325/(0.4*1.225)
    Real e_stp = 101325.0 / (0.4 * 1.225);
    Real p = EquationOfState::compute_pressure(props, 1.225, e_stp);
    CHECK(near(p, 101325.0, 1.0), "Ideal gas: atmospheric pressure");

    // Double density at same energy: p should double
    Real p2 = EquationOfState::compute_pressure(props, 2.45, e_stp);
    CHECK(near(p2, 2.0*p, p*0.01), "Double density → double pressure");

    // Sound speed at STP: c = sqrt(γ*p/ρ) ≈ 340 m/s
    Real c = EquationOfState::sound_speed(props, 1.225, e_stp);
    CHECK(c > 300.0 && c < 400.0, "Sound speed ≈ 340 m/s in air");
}

// ==========================================================================
// Test 5: Gruneisen EOS (Copper)
// ==========================================================================
void test_gruneisen() {
    std::cout << "\n=== Test 5: Gruneisen EOS ===\n";

    EOSProperties props;
    props.type = EOSType::Gruneisen;
    props.rho0 = 8930.0;    // Copper kg/m³
    props.C0 = 3940.0;      // Bulk sound speed m/s
    props.S1 = 1.489;       // Hugoniot slope
    props.S2 = 0.0;
    props.S3 = 0.0;
    props.gamma0 = 2.02;    // Gruneisen parameter
    props.a_coeff = 0.47;

    // At reference state: μ = 0, p should be from energy term only
    Real p0 = EquationOfState::compute_pressure(props, 8930.0, 0.0);
    CHECK(near(p0, 0.0, 1000.0), "Reference state: near-zero pressure");

    // 5% compression: μ = 0.05
    Real rho_comp = 8930.0 * 1.05;
    Real p_comp = EquationOfState::compute_pressure(props, rho_comp, 0.0);
    CHECK(p_comp > 0.0, "Compression gives positive pressure");

    // 5% tension: μ = -0.05
    Real rho_tens = 8930.0 * 0.95;
    Real p_tens = EquationOfState::compute_pressure(props, rho_tens, 0.0);
    CHECK(p_tens < 0.0, "Tension gives negative pressure");

    // Sound speed
    Real c = EquationOfState::sound_speed(props, 8930.0, 0.0);
    CHECK(near(c, 3940.0), "Sound speed = C0 = 3940 m/s");
}

// ==========================================================================
// Test 6: JWL EOS (TNT)
// ==========================================================================
void test_jwl() {
    std::cout << "\n=== Test 6: JWL EOS ===\n";

    EOSProperties props;
    props.type = EOSType::JWL;
    props.rho0 = 1630.0;    // TNT density
    props.A_jwl = 3.712e11;  // Pa
    props.B_jwl = 3.231e9;   // Pa
    props.R1 = 4.15;
    props.R2 = 0.95;
    props.omega = 0.30;

    // Detonation products at CJ state (V ≈ 1)
    Real rho = props.rho0;
    Real e = 7.0e6;  // J/kg specific energy

    Real p = EquationOfState::compute_pressure(props, rho, e);
    CHECK(p > 0.0, "JWL: positive pressure at detonation");

    // Expanded products (lower density)
    Real p_exp = EquationOfState::compute_pressure(props, rho * 0.5, e);
    CHECK(p_exp < p, "Expanded products: lower pressure");
}

// ==========================================================================
// Test 7: Linear Polynomial EOS
// ==========================================================================
void test_polynomial() {
    std::cout << "\n=== Test 7: Linear Polynomial EOS ===\n";

    EOSProperties props;
    props.type = EOSType::LinearPolynomial;
    props.rho0 = 1000.0;  // Water
    // Simple linear: p = C1*μ (bulk modulus response)
    props.C_poly[0] = 0.0;
    props.C_poly[1] = 2.2e9;  // Bulk modulus of water
    props.C_poly[2] = 0.0;
    props.C_poly[3] = 0.0;
    props.C_poly[4] = 0.0;
    props.C_poly[5] = 0.0;
    props.C_poly[6] = 0.0;

    // 1% compression: μ = 0.01
    Real p = EquationOfState::compute_pressure(props, 1010.0, 0.0);
    CHECK(near(p, 2.2e9 * 0.01, 1.0e6), "1% compression: p = K*μ");

    // At reference: p = 0
    Real p0 = EquationOfState::compute_pressure(props, 1000.0, 0.0);
    CHECK(near(p0, 0.0, 100.0), "Reference: p = 0");
}

// ==========================================================================
// Test 8: Tabulated EOS
// ==========================================================================
void test_tabulated_eos() {
    std::cout << "\n=== Test 8: Tabulated EOS ===\n";

    EOSProperties props;
    props.type = EOSType::Tabulated;
    props.rho0 = 1000.0;

    // Simple linear pressure table
    props.pressure_table.num_points = 3;
    props.pressure_table.x[0] = -0.1;  props.pressure_table.y[0] = -1.0e8;
    props.pressure_table.x[1] = 0.0;   props.pressure_table.y[1] = 0.0;
    props.pressure_table.x[2] = 0.1;   props.pressure_table.y[2] = 1.0e8;

    // At reference
    Real p0 = EquationOfState::compute_pressure(props, 1000.0, 0.0);
    CHECK(near(p0, 0.0, 1.0), "Tabulated: zero at reference");

    // 5% compression
    Real p_comp = EquationOfState::compute_pressure(props, 1050.0, 0.0);
    CHECK(p_comp > 0.0, "Tabulated: positive under compression");

    // 5% tension
    Real p_tens = EquationOfState::compute_pressure(props, 950.0, 0.0);
    CHECK(p_tens < 0.0, "Tabulated: negative under tension");
}

// ==========================================================================
// Test 9: EOS Type Coverage
// ==========================================================================
void test_eos_types() {
    std::cout << "\n=== Test 9: EOS Type Coverage ===\n";

    CHECK(static_cast<int>(EOSType::IdealGas) != static_cast<int>(EOSType::Gruneisen),
          "Distinct EOS types");
    CHECK(static_cast<int>(EOSType::JWL) != static_cast<int>(EOSType::LinearPolynomial),
          "JWL != Polynomial");
    CHECK(static_cast<int>(EOSType::Tabulated) != static_cast<int>(EOSType::IdealGas),
          "Tabulated != IdealGas");
}

// ==========================================================================
// Test 10: Tied Contact Statistics
// ==========================================================================
void test_tied_stats() {
    std::cout << "\n=== Test 10: Tied Contact Statistics ===\n";

    TiedContact tc;
    TiedContactConfig cfg;
    cfg.type = TiedContactType::TiedAllDOF;
    cfg.penalty_stiffness = 1.0e8;
    tc.set_config(cfg);

    // Add 5 pairs
    for (int i = 0; i < 5; ++i) {
        auto& p = tc.add_pair();
        p.slave_node = i;
        p.master_nodes[0] = i + 5;
        p.num_master_nodes = 1;
        p.phi[0] = 1.0;
    }

    // Deactivate 2 pairs
    tc.pair(1).active = false;
    tc.pair(3).active = false;

    auto stats = tc.get_stats();
    CHECK(stats.total_pairs == 5, "Total = 5");
    CHECK(stats.active_pairs == 3, "Active = 3");
    CHECK(stats.failed_pairs == 2, "Failed = 2");
}

// ==========================================================================
// Main
// ==========================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "NexusSim Wave 5: Tied Contact & EOS Test\n";
    std::cout << "========================================\n";

    test_basic_tying();
    test_proximity_search();
    test_tied_failure();
    test_ideal_gas();
    test_gruneisen();
    test_jwl();
    test_polynomial();
    test_tabulated_eos();
    test_eos_types();
    test_tied_stats();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    std::cout << "========================================\n";

    return (tests_failed > 0) ? 1 : 0;
}
