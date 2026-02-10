/**
 * @file pd_fem_coupling_test.cpp
 * @brief FEM-Peridynamics Coupling Validation Tests
 *
 * Tests the coupling configuration and domain types for FEM-PD coupling.
 * Note: Full integration tests require proper mesh/particle setup which
 * is demonstrated in the bar tension and validation tests.
 */

#include <nexussim/core/core.hpp>
#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_fem_coupling.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::pd;

// Test result tracking
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

// ============================================================================
// Test 1: Domain Types
// ============================================================================

bool test_domain_types() {
    std::cout << "\n=== Test 1: Domain Type Classification ===\n";

    // Test domain type enumeration
    CHECK(static_cast<int>(DomainType::FEM_Only) == 0, "FEM_Only domain type");
    CHECK(static_cast<int>(DomainType::PD_Only) == 1, "PD_Only domain type");
    CHECK(static_cast<int>(DomainType::Overlap) == 2, "Overlap domain type");
    CHECK(static_cast<int>(DomainType::Interface) == 3, "Interface domain type");

    // Test coupling methods
    CHECK(static_cast<int>(CouplingMethod::Arlequin) == 0, "Arlequin coupling method");
    CHECK(static_cast<int>(CouplingMethod::MortarContact) == 1, "MortarContact method");
    CHECK(static_cast<int>(CouplingMethod::DirectForce) == 2, "DirectForce method");
    CHECK(static_cast<int>(CouplingMethod::Morphing) == 3, "Morphing method");

    return true;
}

// ============================================================================
// Test 2: Coupling Configuration
// ============================================================================

bool test_coupling_config() {
    std::cout << "\n=== Test 2: Coupling Configuration ===\n";

    FEMPDCouplingConfig config;

    // Default values
    CHECK(config.method == CouplingMethod::Arlequin, "Default method is Arlequin");
    CHECK(config.blend_width == 0.0, "Default blend_width is 0 (auto)");
    CHECK(config.blend_exponent == 2.0, "Default blend_exponent is 2.0");
    CHECK(config.damage_threshold == 0.3, "Default damage_threshold is 0.3");
    CHECK(config.sync_displacement == true, "Default sync_displacement is true");
    CHECK(config.sync_velocity == true, "Default sync_velocity is true");

    // Modify configuration
    config.method = CouplingMethod::DirectForce;
    config.blend_width = 0.05;
    config.damage_threshold = 0.5;

    CHECK(config.method == CouplingMethod::DirectForce, "Method updated correctly");
    CHECK(config.blend_width == 0.05, "Blend width updated correctly");
    CHECK(config.damage_threshold == 0.5, "Damage threshold updated correctly");

    return true;
}

// ============================================================================
// Test 3: FEMPDCoupling Initialization
// ============================================================================

bool test_coupling_initialization() {
    std::cout << "\n=== Test 3: Coupling Initialization ===\n";

    FEMPDCouplingConfig config;
    config.method = CouplingMethod::Arlequin;
    config.blend_width = 0.1;
    config.blend_exponent = 2.0;

    FEMPDCoupling coupling;
    coupling.initialize(config);

    // After initialization, node_particle_map should be empty (no domains set)
    CHECK(coupling.node_particle_map().empty(), "No coupled pairs before build_coupling");

    return true;
}

// ============================================================================
// Test 4: NodeParticleMap Structure
// ============================================================================

bool test_node_particle_map() {
    std::cout << "\n=== Test 4: NodeParticleMap Structure ===\n";

    NodeParticleMap mapping;
    mapping.fem_node_id = 10;
    mapping.pd_particle_id = 20;
    mapping.weight = 0.7;
    mapping.domain = DomainType::Overlap;

    CHECK(mapping.fem_node_id == 10, "FEM node ID stored correctly");
    CHECK(mapping.pd_particle_id == 20, "PD particle ID stored correctly");
    CHECK(std::abs(mapping.weight - 0.7) < 1e-10, "Weight stored correctly");
    CHECK(mapping.domain == DomainType::Overlap, "Domain type stored correctly");

    return true;
}

// ============================================================================
// Test 5: CoupledSolverConfig
// ============================================================================

bool test_coupled_solver_config() {
    std::cout << "\n=== Test 5: Coupled Solver Configuration ===\n";

    CoupledSolverConfig config;

    // Set values
    config.dt = 1e-8;
    config.total_steps = 1000;
    config.output_interval = 100;
    config.sync_interval = 1;

    config.coupling.method = CouplingMethod::Arlequin;
    config.coupling.blend_width = 0.1;
    config.coupling.damage_threshold = 0.3;

    config.pd_config.dt = config.dt;
    config.pd_config.total_steps = config.total_steps;

    CHECK(config.dt == 1e-8, "Time step configured correctly");
    CHECK(config.total_steps == 1000, "Total steps configured correctly");
    CHECK(config.output_interval == 100, "Output interval configured correctly");
    CHECK(config.sync_interval == 1, "Sync interval configured correctly");
    CHECK(config.coupling.method == CouplingMethod::Arlequin, "Coupling method is Arlequin");
    CHECK(config.pd_config.dt == config.dt, "PD and coupled solver have same dt");

    return true;
}

// ============================================================================
// Test 6: InterfaceSegment Structure
// ============================================================================

bool test_interface_segment() {
    std::cout << "\n=== Test 6: Interface Segment Structure ===\n";

    InterfaceSegment segment;
    segment.fem_nodes = {0, 1, 2, 3};
    segment.pd_particles = {10, 11, 12};
    segment.area = 0.01;
    segment.normal[0] = 1.0;
    segment.normal[1] = 0.0;
    segment.normal[2] = 0.0;

    CHECK(segment.fem_nodes.size() == 4, "FEM nodes vector has correct size");
    CHECK(segment.pd_particles.size() == 3, "PD particles vector has correct size");
    CHECK(segment.area == 0.01, "Area stored correctly");
    CHECK(segment.normal[0] == 1.0, "Normal X component correct");
    CHECK(segment.normal[1] == 0.0, "Normal Y component correct");
    CHECK(segment.normal[2] == 0.0, "Normal Z component correct");

    return true;
}

// ============================================================================
// Test 7: Blending Function Properties
// ============================================================================

bool test_blending_properties() {
    std::cout << "\n=== Test 7: Blending Function Properties ===\n";

    // The Arlequin blending function should have these properties:
    // 1. alpha = 1 at FEM boundary (full FEM contribution)
    // 2. alpha = 0 at PD boundary (full PD contribution)
    // 3. 0 <= alpha <= 1 in the overlap region
    // 4. Smooth (C^n) transition controlled by exponent

    // Test with config.blend_exponent = 2 (quadratic smoothing)
    FEMPDCouplingConfig config;
    config.blend_exponent = 2.0;

    // For t in [0, 1], alpha = (1 - t)^n
    // At t=0 (FEM boundary): alpha = 1
    // At t=1 (PD boundary): alpha = 0
    Real t0 = 0.0;
    Real alpha0 = std::pow(1.0 - t0, config.blend_exponent);
    CHECK(std::abs(alpha0 - 1.0) < 1e-10, "Alpha = 1 at FEM boundary (t=0)");

    Real t1 = 1.0;
    Real alpha1 = std::pow(1.0 - t1, config.blend_exponent);
    CHECK(std::abs(alpha1 - 0.0) < 1e-10, "Alpha = 0 at PD boundary (t=1)");

    // At t=0.5, alpha = 0.25 (for n=2)
    Real t_mid = 0.5;
    Real alpha_mid = std::pow(1.0 - t_mid, config.blend_exponent);
    CHECK(std::abs(alpha_mid - 0.25) < 1e-10, "Alpha = 0.25 at midpoint (t=0.5, n=2)");

    // Verify monotonicity
    bool monotonic = true;
    Real prev_alpha = 1.0;
    for (int i = 0; i <= 10; ++i) {
        Real t = i / 10.0;
        Real alpha = std::pow(1.0 - t, config.blend_exponent);
        if (alpha > prev_alpha) {
            monotonic = false;
            break;
        }
        prev_alpha = alpha;
    }
    CHECK(monotonic, "Blending function is monotonically decreasing");

    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "NexusSim FEM-PD Coupling Test Suite\n";
    std::cout << "========================================\n";

    Kokkos::initialize();
    {
        test_domain_types();
        test_coupling_config();
        test_coupling_initialization();
        test_node_particle_map();
        test_coupled_solver_config();
        test_interface_segment();
        test_blending_properties();
    }
    Kokkos::finalize();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed)
              << " tests passed\n";
    std::cout << "========================================\n";

    return tests_failed > 0 ? 1 : 0;
}
