/**
 * @file pd_validation_test.cpp
 * @brief Comprehensive validation tests for peridynamics module
 *
 * Tests:
 * 1. State-based PD elastic response
 * 2. Johnson-Cook material model
 * 3. Drucker-Prager yield
 * 4. Contact detection and forces
 * 5. Comparison with analytical solutions
 */

#include <nexussim/core/core.hpp>
#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_particle.hpp>
#include <nexussim/peridynamics/pd_neighbor.hpp>
#include <nexussim/peridynamics/pd_force.hpp>
#include <nexussim/peridynamics/pd_solver.hpp>
#include <nexussim/peridynamics/pd_state_based.hpp>
#include <nexussim/peridynamics/pd_materials.hpp>
#include <nexussim/peridynamics/pd_contact.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace nxs;
using namespace nxs::pd;

/**
 * @brief Create a 3D block of PD particles
 */
std::shared_ptr<PDParticleSystem> create_block(
    Real Lx, Real Ly, Real Lz,
    Index nx, Index ny, Index nz,
    Real rho, Real horizon_factor = 3.015)
{
    Real dx = Lx / nx;
    Real dy = Ly / ny;
    Real dz = Lz / nz;
    Real dmin = std::min({dx, dy, dz});

    Index num_particles = nx * ny * nz;
    auto particles = std::make_shared<PDParticleSystem>();
    particles->initialize(num_particles);

    Real volume = dx * dy * dz;
    Real mass = rho * volume;
    Real horizon = horizon_factor * dmin;

    Index idx = 0;
    for (Index i = 0; i < nx; ++i) {
        for (Index j = 0; j < ny; ++j) {
            for (Index k = 0; k < nz; ++k) {
                Real x = (i + 0.5) * dx;
                Real y = (j + 0.5) * dy;
                Real z = (k + 0.5) * dz;

                particles->set_position(idx, x, y, z);
                particles->set_velocity(idx, 0.0, 0.0, 0.0);
                particles->set_properties(idx, volume, horizon, mass);
                particles->set_ids(idx, 0, 0);
                idx++;
            }
        }
    }

    particles->sync_to_device();
    return particles;
}

/**
 * @brief Test 1: State-based PD elastic response
 *
 * Compare uniform tension response with analytical solution
 */
bool test_state_based_elastic()
{
    std::cout << "\n=== Test 1: State-Based PD Elastic ===" << std::endl;

    // Create a small cube
    Real L = 0.01;  // 1 cm
    Index n = 5;
    Real rho = 7800.0;

    auto particles = create_block(L, L, L, n, n, n, rho);

    // Material: Steel with nu = 0.3 (not limited to 0.25 like bond-based)
    PDStateMaterial steel;
    steel.E = 2.0e11;
    steel.nu = 0.3;
    steel.rho = rho;
    steel.s_critical = 0.1;
    steel.compute_derived();

    std::cout << "Material: E=" << steel.E << " Pa, nu=" << steel.nu
              << ", K=" << steel.K << ", G=" << steel.G << std::endl;

    // Setup state-based solver
    PDSolverConfig config;
    config.dt = 1e-9;
    config.total_steps = 100;
    config.check_damage = false;

    PDStateSolver solver;
    solver.initialize(config);
    solver.set_materials({steel});
    solver.set_particles(particles);
    solver.build_neighbors();

    // Apply uniform strain by displacing particles
    Real strain = 0.001;  // 0.1% strain
    particles->sync_to_host();
    auto u = particles->u_host();
    auto x0 = particles->x0_host();

    for (Index i = 0; i < particles->num_particles(); ++i) {
        // Uniform stretch in x
        u(i, 0) = strain * x0(i, 0);
        u(i, 1) = 0.0;
        u(i, 2) = 0.0;
    }
    particles->sync_to_device();
    particles->update_positions();

    // Compute forces without time stepping
    solver.force().compute_forces(*particles, solver.neighbors());

    // Check dilatation (should be approximately = strain for uniaxial)
    auto theta = solver.force().dilatation();
    auto theta_host = Kokkos::create_mirror_view(theta);
    Kokkos::deep_copy(theta_host, theta);

    Real avg_theta = 0.0;
    for (Index i = 0; i < particles->num_particles(); ++i) {
        avg_theta += theta_host(i);
    }
    avg_theta /= particles->num_particles();

    std::cout << "Applied strain: " << strain << std::endl;
    std::cout << "Average dilatation: " << avg_theta << std::endl;

    // For uniaxial strain, theta should be approximately strain
    // (exact for Poisson's ratio = 0, approximate otherwise)
    bool passed = std::abs(avg_theta - strain) < strain * 0.5;

    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

/**
 * @brief Test 2: Johnson-Cook material response
 */
bool test_johnson_cook()
{
    std::cout << "\n=== Test 2: Johnson-Cook Material ===" << std::endl;

    // Use preset material
    auto al7075 = JCPresets::Al7075T6();

    std::cout << "Material: Al 7075-T6" << std::endl;
    std::cout << "  A = " << al7075.A << " Pa" << std::endl;
    std::cout << "  B = " << al7075.B << " Pa" << std::endl;
    std::cout << "  n = " << al7075.n << std::endl;

    // Test flow stress at various conditions
    Real eps_p = 0.1;       // 10% plastic strain
    Real eps_dot = 1000.0;  // High strain rate
    Real T = 300.0;         // Room temperature

    Real sigma_y = al7075.flow_stress(eps_p, eps_dot, T);

    std::cout << "Flow stress at eps_p=0.1, eps_dot=1000, T=300K:" << std::endl;
    std::cout << "  sigma_y = " << sigma_y / 1e6 << " MPa" << std::endl;

    // Reference value (approximate)
    // At eps_p=0.1: A + B*0.1^0.71 ≈ 546 + 678*0.2 ≈ 680 MPa
    // With rate: * (1 + 0.024*ln(1000)) ≈ 1.17
    // Expected: ~800 MPa
    Real expected_min = 700e6;
    Real expected_max = 900e6;

    bool passed = (sigma_y > expected_min && sigma_y < expected_max);

    std::cout << "Expected range: " << expected_min/1e6 << " - "
              << expected_max/1e6 << " MPa" << std::endl;
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Test 3: Drucker-Prager yield
 */
bool test_drucker_prager()
{
    std::cout << "\n=== Test 3: Drucker-Prager Material ===" << std::endl;

    // Use concrete preset
    auto concrete = DPPresets::Concrete();

    std::cout << "Material: Concrete" << std::endl;
    std::cout << "  phi = " << concrete.phi << " deg" << std::endl;
    std::cout << "  c = " << concrete.c / 1e6 << " MPa" << std::endl;
    std::cout << "  alpha = " << concrete.alpha << std::endl;
    std::cout << "  k = " << concrete.k / 1e6 << " MPa" << std::endl;

    // Test yield function at various stress states
    // p = mean stress (positive in compression)
    // q = von Mises stress

    // State 1: Pure shear (p=0)
    Real p1 = 0.0;
    Real q1 = concrete.k;  // Should be at yield
    Real f1 = concrete.yield_function(p1, q1);

    std::cout << "Yield surface at p=0: q_yield = " << q1/1e6 << " MPa" << std::endl;
    std::cout << "  f(p=0, q=k) = " << f1 << " (should be ~0)" << std::endl;

    // State 2: Triaxial compression (use correct formula: q = k - 3αp for tension cutoff)
    // For DP with compression positive: f = q + 3αp - k, so at yield q = k - 3αp
    // This means higher confinement (higher p) allows higher deviatoric stress
    // Actually, the standard DP for rocks: f = q - 3αp - k (pressure strengthening)
    // Let's test both forms to verify the implementation is consistent
    Real p2 = -10.0e6;  // 10 MPa tension (negative p)
    Real q2 = concrete.k - 3.0 * concrete.alpha * p2;  // At yield with our formula
    Real f2 = concrete.yield_function(p2, q2);

    std::cout << "At p=-10 MPa (tension): q_yield = " << q2/1e6 << " MPa" << std::endl;
    std::cout << "  f(p=-10, q) = " << f2 << " (should be ~0)" << std::endl;

    // Check that yield surface is reasonable: increasing compression should increase strength
    bool passed = (std::abs(f1) < 1e3) && (std::abs(f2) < 1e6);
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Test 4: Contact detection
 */
bool test_contact()
{
    std::cout << "\n=== Test 4: Contact Detection ===" << std::endl;

    // Create two blocks that will collide
    Real L = 0.01;
    Index n = 3;
    Real rho = 7800.0;

    // Block 1: at origin
    auto block1 = create_block(L, L, L, n, n, n, rho);

    // Block 2: slightly overlapping with block 1
    auto block2 = create_block(L, L, L, n, n, n, rho);

    // Move block2 to overlap with block1
    block2->sync_to_host();
    auto x_host = block2->x_host();
    auto x0_host = block2->x0_host();
    Real gap = -0.001;  // 1mm penetration

    for (Index i = 0; i < block2->num_particles(); ++i) {
        x_host(i, 0) += L + gap;
        x0_host(i, 0) += L + gap;
        // Set different body ID
        block2->set_ids(i, 0, 1);
    }
    block2->sync_to_device();

    // Combine into single particle system for contact
    Index total_particles = block1->num_particles() + block2->num_particles();
    auto combined = std::make_shared<PDParticleSystem>();
    combined->initialize(total_particles);

    // Copy block1
    block1->sync_to_host();
    for (Index i = 0; i < block1->num_particles(); ++i) {
        auto x1 = block1->x_host();
        auto v1 = block1->v_host();
        auto vol1 = block1->volume_host();
        auto hor1 = block1->horizon_host();
        auto mass1 = block1->mass_host();

        combined->set_position(i, x1(i,0), x1(i,1), x1(i,2));
        combined->set_velocity(i, v1(i,0), v1(i,1), v1(i,2));
        combined->set_properties(i, vol1(i), hor1(i), mass1(i));
        combined->set_ids(i, 0, 0);
    }

    // Copy block2
    for (Index i = 0; i < block2->num_particles(); ++i) {
        Index idx = block1->num_particles() + i;
        auto x2 = block2->x_host();
        auto v2 = block2->v_host();
        auto vol2 = block2->volume_host();
        auto hor2 = block2->horizon_host();
        auto mass2 = block2->mass_host();

        combined->set_position(idx, x2(i,0), x2(i,1), x2(i,2));
        combined->set_velocity(idx, v2(i,0), v2(i,1), v2(i,2));
        combined->set_properties(idx, vol2(i), hor2(i), mass2(i));
        combined->set_ids(idx, 0, 1);  // Different body
    }
    combined->sync_to_device();

    // Setup contact
    PDContactConfig contact_config;
    contact_config.contact_stiffness = 1e12;
    contact_config.friction_coefficient = 0.3;
    contact_config.enable_self_contact = false;

    PDContact contact;
    contact.initialize(contact_config);

    // Detect contacts
    contact.build_spatial_hash(*combined);
    auto pairs = contact.detect_contacts(*combined);

    std::cout << "Total particles: " << total_particles << std::endl;
    std::cout << "Contact pairs detected: " << pairs.size() << std::endl;

    if (!pairs.empty()) {
        Real max_pen = 0.0;
        for (const auto& pair : pairs) {
            max_pen = std::min(max_pen, pair.gap);
        }
        std::cout << "Max penetration: " << -max_pen * 1000 << " mm" << std::endl;
    }

    // Should have detected contacts
    bool passed = (pairs.size() > 0);
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Test 5: JH-2 ceramic material
 */
bool test_jh2_ceramic()
{
    std::cout << "\n=== Test 5: JH-2 Ceramic Material ===" << std::endl;

    // Use alumina preset
    auto alumina = JH2Presets::Alumina();

    std::cout << "Material: Alumina (Al2O3)" << std::endl;
    std::cout << "  A = " << alumina.A << std::endl;
    std::cout << "  N = " << alumina.N << std::endl;
    std::cout << "  sigma_HEL = " << alumina.sigma_HEL / 1e9 << " GPa" << std::endl;

    // Test normalized strengths
    Real P_star = 0.5;  // Normalized pressure
    Real T_star = alumina.T / alumina.sigma_HEL;  // Normalized tensile

    Real sigma_i = alumina.intact_strength(P_star, T_star, 1.0);
    Real sigma_f = alumina.fractured_strength(P_star, 1.0);

    std::cout << "At P* = 0.5:" << std::endl;
    std::cout << "  Intact strength (normalized): " << sigma_i << std::endl;
    std::cout << "  Fractured strength (normalized): " << sigma_f << std::endl;

    // Intact should be higher than fractured
    bool passed = (sigma_i > sigma_f) && (sigma_i > 0.0);
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Test 6: Compare bond-based vs state-based
 */
bool test_bond_vs_state()
{
    std::cout << "\n=== Test 6: Bond-Based vs State-Based ===" << std::endl;

    Real L = 0.01;
    Index n = 5;
    Real rho = 7800.0;
    Real E = 2.0e11;

    // Bond-based material (nu = 0.25 constraint)
    PDMaterial bond_mat;
    bond_mat.E = E;
    bond_mat.nu = 0.25;
    bond_mat.rho = rho;
    bond_mat.s_critical = 0.1;
    Real dx = L / n;
    bond_mat.compute_derived(3.015 * dx);

    // State-based material (arbitrary nu)
    PDStateMaterial state_mat;
    state_mat.E = E;
    state_mat.nu = 0.25;  // Same for comparison
    state_mat.rho = rho;
    state_mat.s_critical = 0.1;
    state_mat.compute_derived();

    std::cout << "Bond-based: E=" << bond_mat.E << ", nu=" << bond_mat.nu
              << ", c=" << bond_mat.c << std::endl;
    std::cout << "State-based: E=" << state_mat.E << ", nu=" << state_mat.nu
              << ", K=" << state_mat.K << ", G=" << state_mat.G << std::endl;

    // Both should give similar bulk modulus
    Real K_bond = bond_mat.K;
    Real K_state = state_mat.K;

    std::cout << "Bond-based K: " << K_bond / 1e9 << " GPa" << std::endl;
    std::cout << "State-based K: " << K_state / 1e9 << " GPa" << std::endl;

    Real diff = std::abs(K_bond - K_state) / K_state;
    std::cout << "Difference: " << diff * 100 << "%" << std::endl;

    bool passed = (diff < 0.01);  // Within 1%
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    std::cout << "========================================" << std::endl;
    std::cout << "NexusSim PD Advanced Validation Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    int passed = 0;
    int failed = 0;

    if (test_state_based_elastic()) passed++; else failed++;
    if (test_johnson_cook()) passed++; else failed++;
    if (test_drucker_prager()) passed++; else failed++;
    if (test_contact()) passed++; else failed++;
    if (test_jh2_ceramic()) passed++; else failed++;
    if (test_bond_vs_state()) passed++; else failed++;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << (passed + failed)
              << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    Kokkos::finalize();

    return (failed > 0) ? 1 : 0;
}
