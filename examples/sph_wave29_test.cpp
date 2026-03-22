/**
 * @file sph_wave29_test.cpp
 * @brief Wave 29: SPH Production Features Test Suite (5 features, 55 tests)
 *
 * Tests the 5 SPH production classes from sph_wave29.hpp:
 *   1. ParticleSplitMerge          (11 tests)
 *   2. SPHFracturePropagation      (11 tests)
 *   3. KokkosHashNeighborSearch    (11 tests)
 *   4. AdvancedSPHBoundary         (11 tests)
 *   5. SPHStressPoints             (11 tests)
 */

#include <nexussim/sph/sph_wave29.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <set>

using namespace nxs;
using namespace nxs::sph;

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
// 1. ParticleSplitMerge
// ============================================================================

void test_1_particle_split_merge() {
    std::cout << "--- Test 1: ParticleSplitMerge ---\n";

    // 1a. Split 3D produces 8 children
    {
        ParticleSplitMerge psm(2.0, 0.5, 3);
        Real parent_pos[3] = {1.0, 2.0, 3.0};
        Real parent_vel[3] = {10.0, 20.0, 30.0};
        Real parent_mass = 8.0;
        Real parent_h = 0.4;

        Real child_pos[8][3], child_vel[8][3], child_mass[8], child_h[8];
        int num_children = 0;
        psm.split_3d(parent_pos, parent_vel, parent_mass, parent_h,
                      child_pos, child_vel, child_mass, child_h, num_children);
        CHECK(num_children == 8, "SplitMerge: 3D split produces 8 children");
    }

    // 1b. Split 2D produces 4 children
    {
        ParticleSplitMerge psm(2.0, 0.5, 2);
        Real parent_pos[2] = {1.0, 2.0};
        Real parent_vel[2] = {10.0, 20.0};
        Real parent_mass = 4.0;
        Real parent_h = 0.4;

        Real child_pos[4][2], child_vel[4][2], child_mass[4], child_h[4];
        int num_children = 0;
        psm.split_2d(parent_pos, parent_vel, parent_mass, parent_h,
                      child_pos, child_vel, child_mass, child_h, num_children);
        CHECK(num_children == 4, "SplitMerge: 2D split produces 4 children");
    }

    // 1c. Mass conservation after 3D split
    {
        ParticleSplitMerge psm(2.0, 0.5, 3);
        Real parent_pos[3] = {0.0, 0.0, 0.0};
        Real parent_vel[3] = {1.0, 2.0, 3.0};
        Real parent_mass = 10.0;
        Real parent_h = 0.5;

        Real child_pos[8][3], child_vel[8][3], child_mass[8], child_h[8];
        int nc = 0;
        psm.split_3d(parent_pos, parent_vel, parent_mass, parent_h,
                      child_pos, child_vel, child_mass, child_h, nc);

        Real total_child_mass = 0.0;
        for (int i = 0; i < nc; ++i) total_child_mass += child_mass[i];
        CHECK_NEAR(total_child_mass, parent_mass, 1e-12,
                   "SplitMerge: mass conservation after 3D split");
    }

    // 1d. Momentum conservation after 3D split (each child has same velocity)
    {
        ParticleSplitMerge psm(2.0, 0.5, 3);
        Real parent_pos[3] = {0.0, 0.0, 0.0};
        Real parent_vel[3] = {5.0, -3.0, 7.0};
        Real parent_mass = 8.0;
        Real parent_h = 0.4;

        Real child_pos[8][3], child_vel[8][3], child_mass[8], child_h[8];
        int nc = 0;
        psm.split_3d(parent_pos, parent_vel, parent_mass, parent_h,
                      child_pos, child_vel, child_mass, child_h, nc);

        // Total momentum = sum(m_i * v_i) should equal parent_mass * parent_vel
        Real mom_x = 0.0, mom_y = 0.0, mom_z = 0.0;
        for (int i = 0; i < nc; ++i) {
            mom_x += child_mass[i] * child_vel[i][0];
            mom_y += child_mass[i] * child_vel[i][1];
            mom_z += child_mass[i] * child_vel[i][2];
        }
        CHECK_NEAR(mom_x, parent_mass * parent_vel[0], 1e-12,
                   "SplitMerge: momentum conservation x after 3D split");
        CHECK_NEAR(mom_y, parent_mass * parent_vel[1], 1e-12,
                   "SplitMerge: momentum conservation y after 3D split");
        CHECK_NEAR(mom_z, parent_mass * parent_vel[2], 1e-12,
                   "SplitMerge: momentum conservation z after 3D split");
    }

    // 1e. Child positions are symmetric about parent
    {
        ParticleSplitMerge psm(2.0, 0.5, 3);
        Real parent_pos[3] = {1.0, 2.0, 3.0};
        Real parent_vel[3] = {0.0, 0.0, 0.0};
        Real parent_mass = 8.0;
        Real parent_h = 0.8;

        Real child_pos[8][3], child_vel[8][3], child_mass[8], child_h[8];
        int nc = 0;
        psm.split_3d(parent_pos, parent_vel, parent_mass, parent_h,
                      child_pos, child_vel, child_mass, child_h, nc);

        // Center of mass of children should be parent position
        Real com[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < nc; ++i) {
            for (int d = 0; d < 3; ++d) {
                com[d] += child_pos[i][d];
            }
        }
        for (int d = 0; d < 3; ++d) com[d] /= nc;
        CHECK_NEAR(com[0], parent_pos[0], 1e-12,
                   "SplitMerge: child positions symmetric (COM x = parent x)");
        CHECK_NEAR(com[1], parent_pos[1], 1e-12,
                   "SplitMerge: child positions symmetric (COM y = parent y)");
        CHECK_NEAR(com[2], parent_pos[2], 1e-12,
                   "SplitMerge: child positions symmetric (COM z = parent z)");
    }

    // 1f. Merge conserves mass
    {
        ParticleSplitMerge psm;
        Real pos_a[3] = {0.0, 0.0, 0.0};
        Real pos_b[3] = {1.0, 0.0, 0.0};
        Real vel_a[3] = {2.0, 0.0, 0.0};
        Real vel_b[3] = {-1.0, 0.0, 0.0};
        Real mass_a = 3.0, mass_b = 5.0;
        Real merged_pos[3], merged_vel[3], merged_mass;

        psm.merge_particles(pos_a, pos_b, vel_a, vel_b, mass_a, mass_b,
                            merged_pos, merged_vel, merged_mass);
        CHECK_NEAR(merged_mass, mass_a + mass_b, 1e-12,
                   "SplitMerge: merge conserves mass");
    }

    // 1g. Merge conserves momentum
    {
        ParticleSplitMerge psm;
        Real pos_a[3] = {0.0, 0.0, 0.0};
        Real pos_b[3] = {1.0, 0.0, 0.0};
        Real vel_a[3] = {4.0, 1.0, -2.0};
        Real vel_b[3] = {-2.0, 3.0, 1.0};
        Real mass_a = 2.0, mass_b = 3.0;
        Real merged_pos[3], merged_vel[3], merged_mass;

        psm.merge_particles(pos_a, pos_b, vel_a, vel_b, mass_a, mass_b,
                            merged_pos, merged_vel, merged_mass);

        // Check momentum: merged_mass * merged_vel = mass_a * vel_a + mass_b * vel_b
        Real expected_mom_x = mass_a * vel_a[0] + mass_b * vel_b[0];
        Real actual_mom_x = merged_mass * merged_vel[0];
        CHECK_NEAR(actual_mom_x, expected_mom_x, 1e-12,
                   "SplitMerge: merge conserves momentum");
    }

    // 1h. Merge position is center of mass
    {
        ParticleSplitMerge psm;
        Real pos_a[3] = {0.0, 0.0, 0.0};
        Real pos_b[3] = {4.0, 0.0, 0.0};
        Real vel_a[3] = {0.0, 0.0, 0.0};
        Real vel_b[3] = {0.0, 0.0, 0.0};
        Real mass_a = 1.0, mass_b = 3.0;
        Real merged_pos[3], merged_vel[3], merged_mass;

        psm.merge_particles(pos_a, pos_b, vel_a, vel_b, mass_a, mass_b,
                            merged_pos, merged_vel, merged_mass);

        // COM = (1*0 + 3*4) / (1+3) = 3.0
        CHECK_NEAR(merged_pos[0], 3.0, 1e-12,
                   "SplitMerge: merge position is center of mass");
    }

    // 1i. Split criterion triggers correctly
    {
        ParticleSplitMerge psm(2.0, 0.5, 3);
        Real h_initial = 0.1;
        // h_current = 0.25 > 2.0 * 0.1 = 0.2 → should split
        CHECK(psm.should_split(0.25, h_initial),
              "SplitMerge: split criterion triggers when h > h_max_ratio * h0");
        // h_current = 0.15 < 0.2 → should not split
        CHECK(!psm.should_split(0.15, h_initial),
              "SplitMerge: split criterion does not trigger when h < threshold");
    }

    // 1j. Merge criterion
    {
        ParticleSplitMerge psm(2.0, 0.5, 3);
        Real h_initial = 0.1;
        // h_current = 0.04 < 0.5 * 0.1 = 0.05 AND enough neighbors → merge
        CHECK(psm.should_merge(0.04, h_initial, 10, 8),
              "SplitMerge: merge criterion triggers correctly");
        // Not enough neighbors → no merge
        CHECK(!psm.should_merge(0.04, h_initial, 5, 8),
              "SplitMerge: merge criterion requires min neighbors");
    }

    // 1k. h_child correct after 3D split
    {
        ParticleSplitMerge psm(2.0, 0.5, 3);
        Real parent_pos[3] = {0.0, 0.0, 0.0};
        Real parent_vel[3] = {0.0, 0.0, 0.0};
        Real parent_mass = 8.0;
        Real parent_h = 0.6;

        Real child_pos[8][3], child_vel[8][3], child_mass[8], child_h[8];
        int nc = 0;
        psm.split_3d(parent_pos, parent_vel, parent_mass, parent_h,
                      child_pos, child_vel, child_mass, child_h, nc);

        // h_child = h_parent / 8^(1/3) = h_parent / 2
        Real expected_h = parent_h / 2.0;
        CHECK_NEAR(child_h[0], expected_h, 1e-12,
                   "SplitMerge: h_child = h_parent/2 for 3D split");
    }
}

// ============================================================================
// 2. SPHFracturePropagation
// ============================================================================

void test_2_fracture_propagation() {
    std::cout << "--- Test 2: SPHFracturePropagation ---\n";

    // 2a. Zero deformation → zero strain
    {
        SPHFracturePropagation frac;
        Real x_i[3] = {0.0, 0.0, 0.0};
        Real x_j[3] = {1.0, 0.0, 0.0};
        Real X_i[3] = {0.0, 0.0, 0.0};
        Real X_j[3] = {1.0, 0.0, 0.0};
        Real s = frac.compute_bond_strain(x_i, x_j, X_i, X_j);
        CHECK_NEAR(s, 0.0, 1e-15,
                   "Fracture: zero deformation produces zero strain");
    }

    // 2b. Uniform stretch → correct strain
    {
        SPHFracturePropagation frac;
        Real X_i[3] = {0.0, 0.0, 0.0};
        Real X_j[3] = {1.0, 0.0, 0.0};
        // Stretch by 10%: current distance = 1.1
        Real x_i[3] = {0.0, 0.0, 0.0};
        Real x_j[3] = {1.1, 0.0, 0.0};
        Real s = frac.compute_bond_strain(x_i, x_j, X_i, X_j);
        CHECK_NEAR(s, 0.1, 1e-12,
                   "Fracture: 10% stretch gives strain 0.1");
    }

    // 2c. Bond breaks at critical strain
    {
        SPHFracturePropagation frac(0.01, 3.0);
        CHECK(!frac.check_bond(0.02, 0.01),
              "Fracture: bond breaks when strain > s_critical");
    }

    // 2d. Damage formula
    {
        SPHFracturePropagation frac;
        Real d = frac.compute_damage(7, 10);
        CHECK_NEAR(d, 0.3, 1e-12,
                   "Fracture: damage = 1 - 7/10 = 0.3");
    }

    // 2e. Crack tip identification
    {
        SPHFracturePropagation frac;
        CHECK(frac.is_crack_tip(0.5), "Fracture: damage=0.5 is crack tip");
        CHECK(!frac.is_crack_tip(0.1), "Fracture: damage=0.1 is NOT crack tip");
        CHECK(!frac.is_crack_tip(0.9), "Fracture: damage=0.9 is NOT crack tip");
    }

    // 2f. Intact bond check
    {
        SPHFracturePropagation frac(0.01, 3.0);
        CHECK(frac.check_bond(0.005, 0.01),
              "Fracture: bond intact when strain < s_critical");
        CHECK(frac.check_bond(0.01, 0.01),
              "Fracture: bond intact when strain == s_critical");
    }

    // 2g. Multiple bond breaking via propagate
    {
        SPHFracturePropagation frac;
        Real strains[5] = {0.005, 0.015, 0.020, 0.002, 0.030};
        bool active[5] = {true, true, true, true, true};
        int broken = frac.propagate(strains, active, 5, 0.01);
        // Bonds 1, 2, 4 (0-indexed) exceed 0.01
        CHECK(broken == 3, "Fracture: propagate breaks 3 of 5 bonds");
        CHECK(active[0] == true, "Fracture: bond 0 remains intact");
        CHECK(active[3] == true, "Fracture: bond 3 remains intact");
        CHECK(active[1] == false, "Fracture: bond 1 broken");
    }

    // 2h. Damage monotonically increases (can't heal)
    {
        SPHFracturePropagation frac;
        Real d1 = frac.compute_damage(10, 20);
        Real d2 = frac.compute_damage(8, 20);
        Real d3 = frac.compute_damage(5, 20);
        CHECK(d1 <= d2 && d2 <= d3,
              "Fracture: damage monotonically increases as bonds break");
    }

    // 2i. Critical stretch formula 2D
    {
        SPHFracturePropagation frac;
        Real G_c = 100.0;    // J/m^2
        Real mu = 1000.0;    // Pa (shear modulus)
        Real delta = 0.01;   // m (horizon)
        Real sc = frac.critical_stretch(G_c, mu, delta, 2);
        Real expected = std::sqrt(G_c / (3.0 * mu * delta));
        CHECK_NEAR(sc, expected, 1e-12,
                   "Fracture: critical stretch 2D = sqrt(Gc/(3*mu*delta))");
    }

    // 2j. Critical stretch formula 3D
    {
        SPHFracturePropagation frac;
        Real G_c = 100.0;
        Real kappa = 2000.0;  // Pa (bulk modulus)
        Real delta = 0.01;
        Real sc = frac.critical_stretch(G_c, kappa, delta, 3);
        Real expected = std::sqrt(5.0 * G_c / (9.0 * kappa * delta));
        CHECK_NEAR(sc, expected, 1e-12,
                   "Fracture: critical stretch 3D = sqrt(5Gc/(9*kappa*delta))");
    }

    // 2k. Propagation count matches expected
    {
        SPHFracturePropagation frac;
        Real strains[10] = {0.001, 0.002, 0.003, 0.004, 0.005,
                            0.006, 0.007, 0.008, 0.009, 0.010};
        bool active[10];
        for (int i = 0; i < 10; ++i) active[i] = true;

        int broken = frac.propagate(strains, active, 10, 0.005);
        // Bonds with strain > 0.005: indices 5,6,7,8,9 → 5 broken
        CHECK(broken == 5, "Fracture: propagation count matches (5 of 10)");
    }
}

// ============================================================================
// 3. KokkosHashNeighborSearch
// ============================================================================

void test_3_hash_neighbor_search() {
    std::cout << "--- Test 3: KokkosHashNeighborSearch ---\n";

    // 3a. Hash is deterministic (same input → same output)
    {
        KokkosHashNeighborSearch ns(1.0);
        int h1 = ns.hash_cell(3, 5, 7);
        int h2 = ns.hash_cell(3, 5, 7);
        CHECK(h1 == h2, "HashSearch: hash is deterministic");
    }

    // 3b. Same-cell particles found as neighbors
    {
        KokkosHashNeighborSearch ns(1.0, 1024);
        std::vector<std::array<Real, 3>> pos = {
            {0.1, 0.1, 0.1},
            {0.2, 0.2, 0.2}
        };
        ns.build(pos);
        auto neighbors = ns.find_neighbors(0, 0.5);
        CHECK(!neighbors.empty(), "HashSearch: same-cell particles found");
        bool found = false;
        for (int n : neighbors) { if (n == 1) found = true; }
        CHECK(found, "HashSearch: particle 1 found as neighbor of 0");
    }

    // 3c. Adjacent-cell particles found
    {
        KokkosHashNeighborSearch ns(1.0, 1024);
        std::vector<std::array<Real, 3>> pos = {
            {0.9, 0.0, 0.0},   // Cell (0,0,0)
            {1.1, 0.0, 0.0}    // Cell (1,0,0)
        };
        ns.build(pos);
        auto neighbors = ns.find_neighbors(0, 0.5);
        bool found = false;
        for (int n : neighbors) { if (n == 1) found = true; }
        CHECK(found, "HashSearch: adjacent-cell particle found");
    }

    // 3d. Distant particles not found
    {
        KokkosHashNeighborSearch ns(1.0, 1024);
        std::vector<std::array<Real, 3>> pos = {
            {0.0, 0.0, 0.0},
            {100.0, 100.0, 100.0}
        };
        ns.build(pos);
        auto neighbors = ns.find_neighbors(0, 2.0);
        bool found = false;
        for (int n : neighbors) { if (n == 1) found = true; }
        CHECK(!found, "HashSearch: distant particle not found");
    }

    // 3e. Build + query consistency
    {
        KokkosHashNeighborSearch ns(0.5, 4096);
        std::vector<std::array<Real, 3>> pos;
        // 4x4x4 grid of particles at spacing 0.25
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    pos.push_back({i * 0.25, j * 0.25, k * 0.25});

        ns.build(pos);
        // Particle at origin (0,0,0) should find some neighbors within 0.3
        int count = ns.count_neighbors(0, 0.3);
        CHECK(count > 0, "HashSearch: build+query finds neighbors in grid");
        // But not all 63 other particles
        CHECK(count < 64, "HashSearch: build+query doesn't find all particles");
    }

    // 3f. Symmetric: if i finds j, j finds i
    {
        KokkosHashNeighborSearch ns(1.0, 1024);
        std::vector<std::array<Real, 3>> pos = {
            {0.0, 0.0, 0.0},
            {0.5, 0.0, 0.0},
            {0.0, 0.5, 0.0}
        };
        ns.build(pos);

        auto n0 = ns.find_neighbors(0, 1.0);
        auto n1 = ns.find_neighbors(1, 1.0);

        bool zero_finds_one = false;
        for (int n : n0) { if (n == 1) zero_finds_one = true; }
        bool one_finds_zero = false;
        for (int n : n1) { if (n == 0) one_finds_zero = true; }
        CHECK(zero_finds_one && one_finds_zero,
              "HashSearch: neighbor search is symmetric");
    }

    // 3g. Empty space — no neighbors
    {
        KokkosHashNeighborSearch ns(1.0, 1024);
        std::vector<std::array<Real, 3>> pos = {
            {0.0, 0.0, 0.0}
        };
        ns.build(pos);
        auto neighbors = ns.find_neighbors(0, 1.0);
        CHECK(neighbors.empty(), "HashSearch: single particle has no neighbors");
    }

    // 3h. Many particles - no crash
    {
        KokkosHashNeighborSearch ns(1.0, 65536);
        std::vector<std::array<Real, 3>> pos;
        for (int i = 0; i < 1000; ++i) {
            Real x = (i % 10) * 0.5;
            Real y = ((i / 10) % 10) * 0.5;
            Real z = (i / 100) * 0.5;
            pos.push_back({x, y, z});
        }
        ns.build(pos);
        auto neighbors = ns.find_neighbors(500, 1.5);
        CHECK(true, "HashSearch: 1000 particles build+query no crash");
    }

    // 3i. Cell index computation
    {
        KokkosHashNeighborSearch ns(2.0);
        int ix, iy, iz;
        ns.cell_index(3.5, -1.2, 7.9, ix, iy, iz);
        CHECK(ix == 1, "HashSearch: cell_index x = floor(3.5/2.0) = 1");
        CHECK(iy == -1, "HashSearch: cell_index y = floor(-1.2/2.0) = -1");
        CHECK(iz == 3, "HashSearch: cell_index z = floor(7.9/2.0) = 3");
    }

    // 3j. Hash distribution — different cells give different hashes (mostly)
    {
        KokkosHashNeighborSearch ns(1.0, 65536);
        std::set<int> hashes;
        for (int i = 0; i < 100; ++i) {
            hashes.insert(ns.hash_cell(i, i * 2, i * 3));
        }
        // With 100 different cell coords and 65536 buckets, expect few collisions
        CHECK(hashes.size() > 90,
              "HashSearch: hash distribution has < 10% collision rate");
    }

    // 3k. Neighbor count matches find_neighbors size
    {
        KokkosHashNeighborSearch ns(1.0, 1024);
        std::vector<std::array<Real, 3>> pos = {
            {0.0, 0.0, 0.0},
            {0.3, 0.0, 0.0},
            {0.0, 0.3, 0.0},
            {5.0, 5.0, 5.0}
        };
        ns.build(pos);
        auto neighbors = ns.find_neighbors(0, 0.5);
        int count = ns.count_neighbors(0, 0.5);
        CHECK(count == static_cast<int>(neighbors.size()),
              "HashSearch: count_neighbors matches find_neighbors size");
    }
}

// ============================================================================
// 4. AdvancedSPHBoundary
// ============================================================================

void test_4_advanced_boundary() {
    std::cout << "--- Test 4: AdvancedSPHBoundary ---\n";

    // 4a. Rigid wall reflection position
    {
        // Wall at x = 0, normal pointing in +x direction
        Real normal[3] = {1.0, 0.0, 0.0};
        AdvancedSPHBoundary wall(BoundaryType::RigidWall, normal, 0.0);

        Real fluid_pos[3] = {0.5, 1.0, 2.0};
        Real ghost_pos[3];
        wall.reflect_position(fluid_pos, ghost_pos);

        // Ghost should be at (-0.5, 1.0, 2.0) — mirror across x=0
        CHECK_NEAR(ghost_pos[0], -0.5, 1e-12,
                   "Boundary: rigid wall reflects x position");
        CHECK_NEAR(ghost_pos[1], 1.0, 1e-12,
                   "Boundary: rigid wall preserves y position");
        CHECK_NEAR(ghost_pos[2], 2.0, 1e-12,
                   "Boundary: rigid wall preserves z position");
    }

    // 4b. No-slip velocity reversal
    {
        Real normal[3] = {1.0, 0.0, 0.0};
        AdvancedSPHBoundary wall(BoundaryType::NoSlip, normal, 0.0);

        Real fluid_vel[3] = {3.0, 4.0, 5.0};
        Real ghost_vel[3];
        wall.reflect_velocity(fluid_vel, ghost_vel);

        CHECK_NEAR(ghost_vel[0], -3.0, 1e-12, "Boundary: no-slip reverses vx");
        CHECK_NEAR(ghost_vel[1], -4.0, 1e-12, "Boundary: no-slip reverses vy");
        CHECK_NEAR(ghost_vel[2], -5.0, 1e-12, "Boundary: no-slip reverses vz");
    }

    // 4c. Free-slip: normal component reversed
    {
        Real normal[3] = {1.0, 0.0, 0.0};
        AdvancedSPHBoundary wall(BoundaryType::FreeSlip, normal, 0.0);

        Real fluid_vel[3] = {3.0, 4.0, 5.0};
        Real ghost_vel[3];
        wall.reflect_velocity(fluid_vel, ghost_vel);

        // v_ghost = v - 2*(v.n)*n = (3,4,5) - 2*3*(1,0,0) = (-3, 4, 5)
        CHECK_NEAR(ghost_vel[0], -3.0, 1e-12,
                   "Boundary: free-slip reverses normal component");
    }

    // 4d. Free-slip: tangential component preserved
    {
        Real normal[3] = {1.0, 0.0, 0.0};
        AdvancedSPHBoundary wall(BoundaryType::FreeSlip, normal, 0.0);

        Real fluid_vel[3] = {3.0, 4.0, 5.0};
        Real ghost_vel[3];
        wall.reflect_velocity(fluid_vel, ghost_vel);

        CHECK_NEAR(ghost_vel[1], 4.0, 1e-12,
                   "Boundary: free-slip preserves tangential vy");
        CHECK_NEAR(ghost_vel[2], 5.0, 1e-12,
                   "Boundary: free-slip preserves tangential vz");
    }

    // 4e. Porous partial reflection
    {
        Real normal[3] = {1.0, 0.0, 0.0};
        Real porosity = 0.5;
        AdvancedSPHBoundary wall(BoundaryType::Porous, normal, 0.0, porosity);

        Real fluid_vel[3] = {4.0, 0.0, 0.0};
        Real ghost_vel[3];
        wall.reflect_velocity(fluid_vel, ghost_vel);

        // v_ghost = (1-0.5)*(-4) + 0.5*4 = -2 + 2 = 0
        CHECK_NEAR(ghost_vel[0], 0.0, 1e-12,
                   "Boundary: porous wall partial reflection (porosity=0.5)");
    }

    // 4f. Signed distance
    {
        Real normal[3] = {0.0, 1.0, 0.0};
        AdvancedSPHBoundary wall(BoundaryType::RigidWall, normal, 2.0);

        Real point_above[3] = {0.0, 3.0, 0.0};
        Real point_below[3] = {0.0, 1.0, 0.0};
        CHECK_NEAR(wall.signed_distance(point_above), 1.0, 1e-12,
                   "Boundary: signed distance positive on fluid side");
        CHECK_NEAR(wall.signed_distance(point_below), -1.0, 1e-12,
                   "Boundary: signed distance negative on solid side");
    }

    // 4g. Boundary detection
    {
        Real normal[3] = {0.0, 0.0, 1.0};
        AdvancedSPHBoundary wall(BoundaryType::RigidWall, normal, 5.0);

        Real near_point[3] = {0.0, 0.0, 5.1};
        Real far_point[3] = {0.0, 0.0, 10.0};
        CHECK(wall.is_near_boundary(near_point, 0.2),
              "Boundary: near point detected");
        CHECK(!wall.is_near_boundary(far_point, 0.2),
              "Boundary: far point not detected");
    }

    // 4h. Symmetry of reflection (reflect twice → back to original)
    {
        Real normal[3] = {1.0, 0.0, 0.0};
        AdvancedSPHBoundary wall(BoundaryType::RigidWall, normal, 0.0);

        Real pos[3] = {2.0, 3.0, 4.0};
        Real ghost[3], back[3];
        wall.reflect_position(pos, ghost);
        wall.reflect_position(ghost, back);

        CHECK_NEAR(back[0], pos[0], 1e-12,
                   "Boundary: double reflection returns to original position");
    }

    // 4i. Zero porosity = rigid wall behavior
    {
        Real normal[3] = {1.0, 0.0, 0.0};
        AdvancedSPHBoundary porous(BoundaryType::Porous, normal, 0.0, 0.0);

        Real fluid_vel[3] = {5.0, 3.0, 1.0};
        Real ghost_vel[3];
        porous.reflect_velocity(fluid_vel, ghost_vel);

        // porosity=0: v_ghost = 1.0 * (-v) + 0 * v = -v
        CHECK_NEAR(ghost_vel[0], -5.0, 1e-12,
                   "Boundary: zero porosity equals rigid wall (vx)");
        CHECK_NEAR(ghost_vel[1], -3.0, 1e-12,
                   "Boundary: zero porosity equals rigid wall (vy)");
    }

    // 4j. Full porosity = no reflection (pass-through)
    {
        Real normal[3] = {1.0, 0.0, 0.0};
        AdvancedSPHBoundary porous(BoundaryType::Porous, normal, 0.0, 1.0);

        Real fluid_vel[3] = {5.0, 3.0, 1.0};
        Real ghost_vel[3];
        porous.reflect_velocity(fluid_vel, ghost_vel);

        // porosity=1: v_ghost = 0 * (-v) + 1.0 * v = v
        CHECK_NEAR(ghost_vel[0], 5.0, 1e-12,
                   "Boundary: full porosity = no reflection (vx)");
        CHECK_NEAR(ghost_vel[1], 3.0, 1e-12,
                   "Boundary: full porosity = no reflection (vy)");
    }

    // 4k. Normal direction preserved after construction
    {
        Real normal[3] = {0.0, 0.0, 1.0};
        AdvancedSPHBoundary wall(BoundaryType::FreeSlip, normal, 0.0);
        const Real* n = wall.normal();
        CHECK_NEAR(n[0], 0.0, 1e-12, "Boundary: normal x preserved");
        CHECK_NEAR(n[1], 0.0, 1e-12, "Boundary: normal y preserved");
        CHECK_NEAR(n[2], 1.0, 1e-12, "Boundary: normal z preserved");
    }
}

// ============================================================================
// 5. SPHStressPoints
// ============================================================================

void test_5_stress_points() {
    std::cout << "--- Test 5: SPHStressPoints ---\n";

    // Set up a simple 3-particle configuration with known neighbors
    // Particles: 0 at (0,0,0), 1 at (1,0,0), 2 at (0,1,0)
    // Neighbor lists: 0->{1,2}, 1->{0,2}, 2->{0,1}
    std::vector<std::array<Real, 3>> positions = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}
    };
    std::vector<std::vector<int>> neighbors = {
        {1, 2},
        {0, 2},
        {0, 1}
    };

    SPHStressPoints sp;

    // 5a. Stress point at midpoint
    {
        std::vector<std::array<Real, 3>> sp_pos;
        std::vector<StressPointPair> sp_pairs;
        sp.generate_stress_points(positions, neighbors, 3, sp_pos, sp_pairs);

        // Pair (0,1): midpoint = (0.5, 0, 0)
        bool found_01 = false;
        for (size_t s = 0; s < sp_pos.size(); ++s) {
            if (sp_pairs[s].particle_a == 0 && sp_pairs[s].particle_b == 1) {
                CHECK_NEAR(sp_pos[s][0], 0.5, 1e-12,
                           "StressPoints: midpoint x for pair (0,1)");
                CHECK_NEAR(sp_pos[s][1], 0.0, 1e-12,
                           "StressPoints: midpoint y for pair (0,1)");
                found_01 = true;
            }
        }
        CHECK(found_01, "StressPoints: stress point exists at midpoint of (0,1)");
    }

    // 5b. Number of stress points (3 unique pairs for 3 fully-connected particles)
    {
        std::vector<std::array<Real, 3>> sp_pos;
        std::vector<StressPointPair> sp_pairs;
        sp.generate_stress_points(positions, neighbors, 3, sp_pos, sp_pairs);
        CHECK(static_cast<int>(sp_pos.size()) == 3,
              "StressPoints: 3 unique stress points for 3-particle triangle");
    }

    // 5c. Stress interpolation with uniform field
    {
        // If all neighbor stresses are the same, interpolated stress = that value
        Real stresses[2][6] = {
            {100.0, 200.0, 300.0, 10.0, 20.0, 30.0},
            {100.0, 200.0, 300.0, 10.0, 20.0, 30.0}
        };
        Real pos[3] = {0.5, 0.0, 0.0};
        Real result[6];
        sp.compute_stress_at_point(pos, stresses, 2, result);
        CHECK_NEAR(result[0], 100.0, 1e-12,
                   "StressPoints: uniform stress field reproduced (xx)");
        CHECK_NEAR(result[1], 200.0, 1e-12,
                   "StressPoints: uniform stress field reproduced (yy)");
    }

    // 5d. Update positions after particle move
    {
        std::vector<std::array<Real, 3>> sp_pos;
        std::vector<StressPointPair> sp_pairs;
        sp.generate_stress_points(positions, neighbors, 3, sp_pos, sp_pairs);

        // Move particle 1 from (1,0,0) to (2,0,0)
        std::vector<std::array<Real, 3>> new_positions = positions;
        new_positions[1] = {2.0, 0.0, 0.0};

        sp.update_stress_point_positions(new_positions, sp_pairs, sp_pos,
                                         static_cast<int>(sp_pos.size()));

        // Find the (0,1) stress point and check it moved to midpoint (1,0,0)
        for (size_t s = 0; s < sp_pos.size(); ++s) {
            if (sp_pairs[s].particle_a == 0 && sp_pairs[s].particle_b == 1) {
                CHECK_NEAR(sp_pos[s][0], 1.0, 1e-12,
                           "StressPoints: updated midpoint after move");
            }
        }
    }

    // 5e. Parent pair tracking
    {
        std::vector<std::array<Real, 3>> sp_pos;
        std::vector<StressPointPair> sp_pairs;
        sp.generate_stress_points(positions, neighbors, 3, sp_pos, sp_pairs);

        // Every pair should have valid particle indices
        bool all_valid = true;
        for (const auto& pair : sp_pairs) {
            if (pair.particle_a < 0 || pair.particle_a >= 3 ||
                pair.particle_b < 0 || pair.particle_b >= 3 ||
                pair.particle_a >= pair.particle_b) {
                all_valid = false;
            }
        }
        CHECK(all_valid, "StressPoints: parent pairs valid (a < b, in range)");
    }

    // 5f. Stress at boundary (single neighbor stress)
    {
        Real stresses[1][6] = {{50.0, 60.0, 70.0, 1.0, 2.0, 3.0}};
        Real pos[3] = {0.0, 0.0, 0.0};
        Real result[6];
        sp.compute_stress_at_point(pos, stresses, 1, result);
        CHECK_NEAR(result[0], 50.0, 1e-12,
                   "StressPoints: single-neighbor stress reproduced");
    }

    // 5g. Uniform stress field reproduced through interpolation back to velocity point
    {
        std::vector<std::array<Real, 3>> sp_pos;
        std::vector<StressPointPair> sp_pairs;
        sp.generate_stress_points(positions, neighbors, 3, sp_pos, sp_pairs);

        int num_sp = static_cast<int>(sp_pos.size());
        // All stress points have the same stress
        std::vector<std::array<Real, 6>> sp_stress(num_sp);
        for (int s = 0; s < num_sp; ++s) {
            sp_stress[s] = {100.0, 200.0, 300.0, 10.0, 20.0, 30.0};
        }

        Real result[6];
        sp.interpolate_to_velocity_point(0, sp_pairs, sp_stress, num_sp, result);
        CHECK_NEAR(result[0], 100.0, 1e-12,
                   "StressPoints: uniform field preserved through interpolation");
    }

    // 5h. Stress point density (count per particle)
    {
        std::vector<std::array<Real, 3>> sp_pos;
        std::vector<StressPointPair> sp_pairs;
        sp.generate_stress_points(positions, neighbors, 3, sp_pos, sp_pairs);
        int num_sp = static_cast<int>(sp_pos.size());

        // Particle 0 neighbors 1 and 2, so has 2 stress points
        int count = sp.count_stress_points_for_particle(0, sp_pairs, num_sp);
        CHECK(count == 2, "StressPoints: particle 0 has 2 associated stress points");
    }

    // 5i. No duplicate stress points (pair (i,j) only once)
    {
        std::vector<std::array<Real, 3>> sp_pos;
        std::vector<StressPointPair> sp_pairs;
        sp.generate_stress_points(positions, neighbors, 3, sp_pos, sp_pairs);

        // Check for duplicates
        std::set<std::pair<int, int>> seen;
        bool has_dups = false;
        for (const auto& p : sp_pairs) {
            auto key = std::make_pair(p.particle_a, p.particle_b);
            if (seen.count(key)) has_dups = true;
            seen.insert(key);
        }
        CHECK(!has_dups, "StressPoints: no duplicate stress points");
    }

    // 5j. Conservation check (stress point count for symmetric system)
    {
        // For a fully-connected 3-particle system with symmetric neighbors,
        // each particle should have same number of stress points
        std::vector<std::array<Real, 3>> sp_pos;
        std::vector<StressPointPair> sp_pairs;
        sp.generate_stress_points(positions, neighbors, 3, sp_pos, sp_pairs);
        int num_sp = static_cast<int>(sp_pos.size());

        int c0 = sp.count_stress_points_for_particle(0, sp_pairs, num_sp);
        int c1 = sp.count_stress_points_for_particle(1, sp_pairs, num_sp);
        int c2 = sp.count_stress_points_for_particle(2, sp_pairs, num_sp);
        CHECK(c0 == c1 && c1 == c2,
              "StressPoints: symmetric system has equal stress point counts");
    }

    // 5k. Zero velocity → no position change
    {
        std::vector<std::array<Real, 3>> sp_pos;
        std::vector<StressPointPair> sp_pairs;
        sp.generate_stress_points(positions, neighbors, 3, sp_pos, sp_pairs);

        // Save original positions
        std::vector<std::array<Real, 3>> sp_pos_orig = sp_pos;

        // "Update" with same positions (no movement)
        sp.update_stress_point_positions(positions, sp_pairs, sp_pos,
                                         static_cast<int>(sp_pos.size()));

        bool unchanged = true;
        for (size_t s = 0; s < sp_pos.size(); ++s) {
            for (int d = 0; d < 3; ++d) {
                if (std::abs(sp_pos[s][d] - sp_pos_orig[s][d]) > 1e-15) {
                    unchanged = false;
                }
            }
        }
        CHECK(unchanged,
              "StressPoints: zero velocity produces no position change");
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Wave 29 SPH Production Features Test Suite ===\n\n";

    test_1_particle_split_merge();
    test_2_fracture_propagation();
    test_3_hash_neighbor_search();
    test_4_advanced_boundary();
    test_5_stress_points();

    std::cout << "\n=== Wave 29 SPH Production: " << tests_passed
              << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
