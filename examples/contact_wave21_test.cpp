/**
 * @file contact_wave21_test.cpp
 * @brief Wave 21: Contact Refinements Test Suite (6 features, ~40 tests)
 *
 * Tests 6 sub-modules (~6-7 tests each):
 *  1. AutomaticContactDetection — Surface extraction from volume mesh, bucket search, contact pairs
 *  2. Contact2D                 — Node-to-segment contact, gap, normal, penalty, axisymmetric
 *  3. SPHContact                — Particle-surface detection, penalty force, energy
 *  4. AirbagFabricContact       — Self-contact, fabric contact force, leak rate, porosity
 *  5. ContactHeatGeneration     — Frictional heat flux, Charron partition, heat distribution
 *  6. MortarContactFriction     — Mortar integrals, weighted gap, Coulomb friction, tangent
 */

#include <nexussim/fem/contact_wave21.hpp>
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
// 1. AutomaticContactDetection Tests
// ============================================================================
void test_automatic_contact_detection() {
    std::cout << "\n=== 21g: AutomaticContactDetection ===\n";

    AutomaticContactDetection detector;

    // Unit cube: 8 nodes, 1 hex element
    int conn_single[1][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7}
    };
    Real coords_single[8][3] = {
        {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
        {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
    };

    // Test 1: Single hex has 6 exterior faces
    {
        std::vector<AutomaticContactDetection::SurfaceFace> faces;
        int nf = detector.extract_surface(conn_single, 8, 1, coords_single, faces);
        CHECK(nf == 6, "AutoContact: single cube has 6 exterior faces");
    }

    // Test 2: Two hex elements sharing one face yield 10 exterior faces
    {
        Real coords_2[12][3] = {
            {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
            {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1},
            {2,0,0}, {2,1,0}, {2,0,1}, {2,1,1}
        };
        int conn_2[2][8] = {
            {0, 1, 2, 3, 4, 5, 6, 7},
            {1, 8, 9, 2, 5, 10, 11, 6}
        };
        std::vector<AutomaticContactDetection::SurfaceFace> faces;
        int nf = detector.extract_surface(conn_2, 8, 2, coords_2, faces);
        CHECK(nf == 10, "AutoContact: 2x1x1 block has 10 exterior faces");
    }

    // Test 3: Surface normals are unit vectors
    {
        std::vector<AutomaticContactDetection::SurfaceFace> faces;
        detector.extract_surface(conn_single, 8, 1, coords_single, faces);
        bool all_unit = true;
        for (auto& f : faces) {
            Real mag = std::sqrt(f.normal[0]*f.normal[0] + f.normal[1]*f.normal[1]
                               + f.normal[2]*f.normal[2]);
            if (std::abs(mag - 1.0) > 1e-8) { all_unit = false; break; }
        }
        CHECK(all_unit, "AutoContact: all surface normals are unit vectors");
    }

    // Test 4: Centroid computation correct for unit cube faces
    {
        std::vector<AutomaticContactDetection::SurfaceFace> faces;
        detector.extract_surface(conn_single, 8, 1, coords_single, faces);
        bool valid = true;
        for (auto& f : faces) {
            for (int d = 0; d < 3; ++d) {
                Real c = f.centroid[d];
                if (std::abs(c) > 1e-10 && std::abs(c - 0.5) > 1e-10
                    && std::abs(c - 1.0) > 1e-10) {
                    valid = false;
                }
            }
        }
        CHECK(valid, "AutoContact: face centroids have correct coordinates");
    }

    // Test 5: Bucket search finds nearby faces
    {
        std::vector<AutomaticContactDetection::SurfaceFace> faces;
        detector.extract_surface(conn_single, 8, 1, coords_single, faces);
        std::vector<std::vector<int>> result;
        detector.bucket_search(coords_single, 8, faces, 1.5, result);
        int nearby_count = static_cast<int>(result[0].size());
        CHECK(nearby_count > 0, "AutoContact: bucket search finds faces near node 0");
    }

    // Test 6: Contact pair detection between two facing surfaces
    {
        Real coords_plates[16][3] = {
            {0,0,-0.01}, {1,0,-0.01}, {1,1,-0.01}, {0,1,-0.01},
            {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
            {0,0,0.05}, {1,0,0.05}, {1,1,0.05}, {0,1,0.05},
            {0,0,0.06}, {1,0,0.06}, {1,1,0.06}, {0,1,0.06}
        };
        int conn_plates[2][8] = {
            {0, 1, 2, 3, 4, 5, 6, 7},
            {8, 9, 10, 11, 12, 13, 14, 15}
        };
        std::vector<AutomaticContactDetection::SurfaceFace> faces;
        detector.extract_surface(conn_plates, 8, 2, coords_plates, faces);
        std::vector<AutomaticContactDetection::SurfaceFace> surf1, surf2;
        for (auto& f : faces) {
            if (f.parent_element == 0) surf1.push_back(f);
            else surf2.push_back(f);
        }
        std::vector<AutomaticContactDetection::ContactPair> pairs;
        int np = detector.build_contact_pairs(surf1, surf2, coords_plates, 0.2, pairs);
        CHECK(np > 0, "AutoContact: contact pairs detected between facing plates");
    }

    // Test 7: Tet element surface extraction
    {
        Real coords_tet[4][3] = {
            {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}
        };
        int conn_tet[1][8] = { {0, 1, 2, 3, 0, 0, 0, 0} };
        std::vector<AutomaticContactDetection::SurfaceFace> faces;
        int nf = detector.extract_surface(conn_tet, 4, 1, coords_tet, faces);
        CHECK(nf == 4, "AutoContact: single tet has 4 exterior faces");
    }
}

// ============================================================================
// 2. Contact2D Tests
// ============================================================================
void test_contact_2d() {
    std::cout << "\n=== 21h: Contact2D ===\n";

    // Test 1: Penetrating node detected
    {
        Contact2D contact;
        Real master[2][2] = { {0.0, 0.0}, {1.0, 0.0} };
        Real slave[2] = {0.5, -0.001};
        Real gap = 0.0;
        Real normal[2] = {0.0, 0.0};
        bool detected = contact.detect_2d(slave, master, 1, gap, normal);
        CHECK(detected, "Contact2D: penetrating node detected");
        CHECK(gap < 0.0, "Contact2D: negative gap for penetrating node");
    }

    // Test 2: Gap computation accuracy
    {
        Contact2D contact;
        Real master[2][2] = { {0.0, 0.0}, {1.0, 0.0} };
        Real slave[2] = {0.5, -0.005};
        Real gap = 0.0;
        Real normal[2] = {0.0, 0.0};
        contact.detect_2d(slave, master, 1, gap, normal);
        CHECK_NEAR(gap, -0.005, 1e-10, "Contact2D: gap = -0.005 for node 0.005 below segment");
    }

    // Test 3: Contact normal direction
    {
        Contact2D contact;
        Real master[2][2] = { {0.0, 0.0}, {1.0, 0.0} };
        Real slave[2] = {0.5, -0.001};
        Real gap = 0.0;
        Real normal[2] = {0.0, 0.0};
        contact.detect_2d(slave, master, 1, gap, normal);
        CHECK_NEAR(std::abs(normal[1]), 1.0, 1e-8,
                   "Contact2D: normal perpendicular to horizontal segment");
    }

    // Test 4: Penalty contact force (plane strain)
    {
        Contact2D contact(Contact2D::Mode::PlaneStrain);
        Real penalty = 1.0e6;
        Real gap = -0.01;
        Real normal[2] = {0.0, 1.0};
        Real contact_pt[2] = {0.5, 0.0};
        Real force[2] = {0.0, 0.0};
        contact.compute_contact_force_2d(gap, normal, penalty, contact_pt, force);
        CHECK_NEAR(force[1], 10000.0, 1e-4, "Contact2D: penalty force = k*pen = 10000 N");
        CHECK_NEAR(force[0], 0.0, 1e-10, "Contact2D: no tangential force component");
    }

    // Test 5: No force when separated
    {
        Contact2D contact;
        Real gap = 0.05;
        Real normal[2] = {0.0, 1.0};
        Real contact_pt[2] = {0.5, 0.0};
        Real force[2] = {999.0, 999.0};
        contact.compute_contact_force_2d(gap, normal, 1.0e6, contact_pt, force);
        CHECK_NEAR(force[0], 0.0, 1e-10, "Contact2D: zero force when separated (x)");
        CHECK_NEAR(force[1], 0.0, 1e-10, "Contact2D: zero force when separated (y)");
    }

    // Test 6: Axisymmetric hoop factor
    {
        Real penalty = 1.0e6;
        Real gap = -0.01;
        Real normal[2] = {0.0, 1.0};
        Real r = 0.5;
        Real contact_pt[2] = {r, 0.0};
        Contact2D contact_ps(Contact2D::Mode::PlaneStrain);
        Contact2D contact_axi(Contact2D::Mode::Axisymmetric);
        Real force_ps[2] = {0, 0};
        Real force_axi[2] = {0, 0};
        contact_ps.compute_contact_force_2d(gap, normal, penalty, contact_pt, force_ps);
        contact_axi.compute_contact_force_2d(gap, normal, penalty, contact_pt, force_axi);
        Real ratio = force_axi[1] / (force_ps[1] + 1e-30);
        Real expected_ratio = 2.0 * 3.14159265358979323846 * r;
        CHECK_NEAR(ratio, expected_ratio, 0.01,
                   "Contact2D: axisymmetric force = 2*pi*r * plane strain force");
    }
}

// ============================================================================
// 3. SPHContact Tests
// ============================================================================
void test_sph_contact() {
    std::cout << "\n=== 21i: SPHContact ===\n";

    SPHContact sph;
    Real tri_coords[3][3] = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}
    };
    int face_conn[1][3] = { {0, 1, 2} };

    // Test 1: Particle near surface detected
    {
        Real particle[1][3] = { {0.3, 0.3, 0.05} };
        std::vector<SPHContact::SPHContactPair> contacts;
        int nc = sph.detect_sph_contact(particle, 1, face_conn, tri_coords, 1, 0.1, contacts);
        CHECK(nc > 0, "SPH: particle at z=0.05 detected near z=0 surface");
    }

    // Test 2: Force repels particle
    {
        Real pos[3] = {0.3, 0.3, 0.05};
        Real force[3] = {0, 0, 0};
        bool in_contact = sph.compute_sph_contact_force(
            pos, tri_coords[0], tri_coords[1], tri_coords[2], 1.0e5, 0.1, force);
        CHECK(in_contact, "SPH: compute_sph_contact_force detects contact");
        CHECK(force[2] > 0.0, "SPH: force repels particle (positive z)");
    }

    // Test 3: No contact when far
    {
        Real pos[3] = {0.3, 0.3, 0.5};
        Real force[3] = {0, 0, 0};
        bool in_contact = sph.compute_sph_contact_force(
            pos, tri_coords[0], tri_coords[1], tri_coords[2], 1.0e5, 0.1, force);
        CHECK(!in_contact, "SPH: no contact when particle far from surface");
    }

    // Test 4: Force magnitude = penalty * penetration
    {
        Real pos[3] = {0.3, 0.3, 0.03};
        Real force[3] = {0, 0, 0};
        sph.compute_sph_contact_force(
            pos, tri_coords[0], tri_coords[1], tri_coords[2], 1.0e5, 0.1, force);
        Real force_mag = std::sqrt(force[0]*force[0] + force[1]*force[1] + force[2]*force[2]);
        CHECK_NEAR(force_mag, 7000.0, 10.0, "SPH: force magnitude = penalty * penetration");
    }

    // Test 5: Larger penetration gives stronger force
    {
        Real pos_near[3] = {0.3, 0.3, 0.08};
        Real pos_deep[3] = {0.3, 0.3, 0.02};
        Real force_near[3] = {0, 0, 0};
        Real force_deep[3] = {0, 0, 0};
        sph.compute_sph_contact_force(pos_near, tri_coords[0], tri_coords[1], tri_coords[2],
                                      1.0e5, 0.1, force_near);
        sph.compute_sph_contact_force(pos_deep, tri_coords[0], tri_coords[1], tri_coords[2],
                                      1.0e5, 0.1, force_deep);
        Real f_near = std::sqrt(force_near[0]*force_near[0] + force_near[1]*force_near[1]
                              + force_near[2]*force_near[2]);
        Real f_deep = std::sqrt(force_deep[0]*force_deep[0] + force_deep[1]*force_deep[1]
                              + force_deep[2]*force_deep[2]);
        CHECK(f_deep > f_near, "SPH: larger penetration -> stronger force");
    }

    // Test 6: Multi-particle detection
    {
        // Use particles near the triangle centroid (0.333, 0.333, 0) to pass centroid check
        // Centroid check: dist_to_centroid^2 < 4 * search_radius^2 = 0.04
        Real particles[3][3] = {
            {0.333, 0.333, 0.05},  // near surface, close to centroid
            {0.333, 0.333, 5.0},   // far away (fails centroid z-distance)
            {0.333, 0.333, 0.03}   // near surface, close to centroid
        };
        std::vector<SPHContact::SPHContactPair> contacts;
        int nc = sph.detect_sph_contact(particles, 3, face_conn, tri_coords, 1, 0.1, contacts);
        CHECK(nc == 2, "SPH: 2 of 3 particles in contact");
    }

    // Test 7: Contact normal direction
    {
        Real pos[3] = {0.3, 0.3, 0.05};
        Real force[3] = {0, 0, 0};
        sph.compute_sph_contact_force(pos, tri_coords[0], tri_coords[1], tri_coords[2],
                                      1.0e5, 0.1, force);
        Real f_mag = std::sqrt(force[0]*force[0] + force[1]*force[1] + force[2]*force[2]);
        if (f_mag > 1e-10) {
            Real nz = force[2] / f_mag;
            CHECK(nz > 0.9, "SPH: contact force predominantly in +z direction");
        } else {
            CHECK(false, "SPH: expected nonzero contact force");
        }
    }
}

// ============================================================================
// 4. AirbagFabricContact Tests
// ============================================================================
void test_airbag_fabric_contact() {
    std::cout << "\n=== 21j: AirbagFabricContact ===\n";

    AirbagFabricContact fabric;
    fabric.set_thickness(0.001);
    fabric.set_porosity_coefficient(0.001);

    // Test 1: Self-contact detection for folded fabric
    {
        // Use small triangles (edge ~0.002) so centroids are within 4*search_r2
        // search_r = 2*thickness = 2*0.001 = 0.002, 4*search_r2 = 4*4e-6 = 16e-6
        // Centroid distance must be < sqrt(16e-6) = 0.004
        // Small triangles at z=0 and z=0.0015 (within search radius of 0.002)
        Real coords[6][3] = {
            {0, 0, 0}, {0.002, 0, 0}, {0.001, 0.002, 0},
            {0, 0, 0.0015}, {0.002, 0, 0.0015}, {0.001, 0.002, 0.0015}
        };
        int tris[2][3] = { {0, 1, 2}, {3, 4, 5} };
        std::vector<AirbagFabricContact::FabricContactPair> contacts;
        int nc = fabric.detect_self_contact(coords, 6, tris, 2, 0.001, contacts);
        CHECK(nc > 0, "Fabric: self-contact detected for folded fabric");
    }

    // Test 2: No self-contact when far apart
    {
        Real coords[6][3] = {
            {0, 0, 0}, {0.002, 0, 0}, {0.001, 0.002, 0},
            {0, 0, 1.0}, {0.002, 0, 1.0}, {0.001, 0.002, 1.0}
        };
        int tris[2][3] = { {0, 1, 2}, {3, 4, 5} };
        std::vector<AirbagFabricContact::FabricContactPair> contacts;
        int nc = fabric.detect_self_contact(coords, 6, tris, 2, 0.001, contacts);
        CHECK(nc == 0, "Fabric: no self-contact when patches far apart");
    }

    // Test 3: Contact force for penetrating gap
    {
        Real force = 0.0;
        fabric.compute_fabric_contact_force(-0.0005, 0.01, 50000.0, 1.0e6, force);
        CHECK(force > 0.0, "Fabric: positive contact force for penetrating gap");
    }

    // Test 4: No force when gap is large
    {
        Real force = 999.0;
        fabric.compute_fabric_contact_force(0.01, 0.01, 50000.0, 1.0e6, force);
        CHECK_NEAR(force, 0.0, 1e-6, "Fabric: zero force for large gap");
    }

    // Test 5: Leak rate (Wang-Nefske model)
    {
        Real leak = fabric.compute_leak_rate(0.005, 200000.0, 1.2, 1.0);
        Real expected = 0.001 * 1.0 * std::sqrt(2.0 * 200000.0 / 1.2);
        CHECK_NEAR(leak, expected, 1e-6, "Fabric: leak rate matches Wang-Nefske model");
    }

    // Test 6: Leak rate increases with pressure
    {
        Real leak_low  = fabric.compute_leak_rate(0.005, 100000.0, 1.2, 1.0);
        Real leak_high = fabric.compute_leak_rate(0.005, 400000.0, 1.2, 1.0);
        CHECK(leak_high > leak_low, "Fabric: leak rate increases with pressure");
    }

    // Test 7: Zero leak rate for zero/negative dP
    {
        Real leak_zero = fabric.compute_leak_rate(0.005, 0.0, 1.2, 1.0);
        Real leak_neg  = fabric.compute_leak_rate(0.005, -1000.0, 1.2, 1.0);
        CHECK_NEAR(leak_zero, 0.0, 1e-20, "Fabric: zero leak rate for zero dP");
        CHECK_NEAR(leak_neg, 0.0, 1e-20, "Fabric: zero leak rate for negative dP");
    }
}

// ============================================================================
// 5. ContactHeatGeneration Tests
// ============================================================================
void test_contact_heat_generation() {
    std::cout << "\n=== 21k: ContactHeatGeneration ===\n";

    ContactHeatGeneration heat;
    heat.set_heat_fraction(0.95);
    ContactHeatGeneration::ThermalProps steel(50.0, 7800.0, 500.0);
    heat.set_master_properties(steel);
    heat.set_slave_properties(steel);

    // Test 1: Heat flux = eta * mu * p * v
    {
        Real q = heat.compute_heat_flux(0.3, 1.0e6, 0.1, 0.5);
        CHECK_NEAR(q, 28500.0, 1e-4, "Heat: flux = eta * mu * p * v = 28500 W/m^2");
    }

    // Test 2: Zero heat when no slip
    {
        Real q = heat.compute_heat_flux(0.3, 1.0e6, 0.0, 0.5);
        CHECK_NEAR(q, 0.0, 1e-10, "Heat: zero flux when v_slip = 0");
    }

    // Test 3: Charron partition for identical materials
    {
        Real f = heat.compute_partition_factor();
        CHECK_NEAR(f, 0.5, 1e-10, "Heat: Charron partition = 0.5 for identical materials");
    }

    // Test 4: Charron partition for dissimilar materials
    {
        ContactHeatGeneration heat2;
        ContactHeatGeneration::ThermalProps steel_props(50.0, 7800.0, 500.0);
        ContactHeatGeneration::ThermalProps alu_props(200.0, 2700.0, 900.0);
        heat2.set_master_properties(steel_props);
        heat2.set_slave_properties(alu_props);
        Real f = heat2.compute_partition_factor();
        Real e_m = std::sqrt(50.0 * 7800.0 * 500.0);
        Real e_s = std::sqrt(200.0 * 2700.0 * 900.0);
        Real expected_f = e_m / (e_m + e_s);
        CHECK_NEAR(f, expected_f, 1e-6, "Heat: Charron partition correct for steel-aluminum");
        CHECK(f < 0.5, "Heat: steel gets less heat than aluminum (lower effusivity)");
    }

    // Test 5: Heat distribution updates temperatures
    {
        Real pressures[1] = {1.0e6};
        Real slip_velocities[1] = {0.1};
        Real areas[1] = {0.001};
        Real temperatures[2] = {300.0, 300.0};
        Real masses[2] = {0.1, 0.1};
        int master_ids[1] = {0};
        int slave_ids[1] = {1};
        heat.distribute_heat(1, 0.3, pressures, slip_velocities,
                            areas, temperatures, masses, 0.001, master_ids, slave_ids);
        CHECK(temperatures[0] > 300.0, "Heat: master temperature increased");
        CHECK(temperatures[1] > 300.0, "Heat: slave temperature increased");
    }

    // Test 6: Energy conservation
    {
        ContactHeatGeneration heat_cons;
        heat_cons.set_heat_fraction(1.0);
        ContactHeatGeneration::ThermalProps props(50.0, 7800.0, 500.0);
        heat_cons.set_master_properties(props);
        heat_cons.set_slave_properties(props);
        Real pressures[1] = {1.0e6};
        Real slip_velocities[1] = {0.5};
        Real areas[1] = {0.01};
        Real temperatures[2] = {0.0, 0.0};
        Real masses[2] = {1.0, 1.0};
        int master_ids[1] = {0};
        int slave_ids[1] = {1};
        heat_cons.distribute_heat(1, 0.3, pressures, slip_velocities,
                                 areas, temperatures, masses, 0.01, master_ids, slave_ids);
        Real total_energy_in_nodes = masses[0] * props.specific_heat * temperatures[0]
                                   + masses[1] * props.specific_heat * temperatures[1];
        Real total_friction_energy = 1.0 * 0.3 * 1.0e6 * 0.5 * 0.01 * 0.01;
        CHECK_NEAR(total_energy_in_nodes, total_friction_energy, 1e-6,
                   "Heat: energy conservation (total heat = friction work)");
    }
}

// ============================================================================
// 6. MortarContactFriction Tests
// ============================================================================
void test_mortar_contact_friction() {
    std::cout << "\n=== 21l: MortarContactFriction ===\n";

    MortarContactFriction mortar;
    mortar.set_normal_penalty(1.0e6);
    mortar.set_tangential_penalty(1.0e5);
    mortar.set_friction_coefficient(0.3);

    // Test 1: Mortar integral produces 4 integration points
    {
        Real master[4][3] = { {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0} };
        Real slave[4][3] = { {0,0,0.01}, {1,0,0.01}, {1,1,0.01}, {0,1,0.01} };
        MortarContactFriction::MortarIntegral integrals[4];
        int num_integrals = 0;
        mortar.compute_mortar_integrals(master, slave, 4, 4, integrals, num_integrals);
        CHECK(num_integrals == 4, "Mortar: 2x2 Gauss quadrature gives 4 integration points");
    }

    // Test 2: Integration weights positive
    {
        Real master[4][3] = { {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0} };
        Real slave[4][3] = { {0,0,0.01}, {1,0,0.01}, {1,1,0.01}, {0,1,0.01} };
        MortarContactFriction::MortarIntegral integrals[4];
        int num_integrals = 0;
        mortar.compute_mortar_integrals(master, slave, 4, 4, integrals, num_integrals);
        bool all_positive = true;
        for (int i = 0; i < num_integrals; ++i)
            if (integrals[i].weight <= 0.0) all_positive = false;
        CHECK(all_positive, "Mortar: all integration weights are positive");
    }

    // Test 3: Weighted gap computation
    {
        Real master[4][3] = { {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0} };
        Real slave[4][3] = { {0,0,0.01}, {1,0,0.01}, {1,1,0.01}, {0,1,0.01} };
        MortarContactFriction::MortarIntegral integrals[4];
        int num_integrals = 0;
        mortar.compute_mortar_integrals(master, slave, 4, 4, integrals, num_integrals);
        Real zero_disp[4][3] = {{0,0,0},{0,0,0},{0,0,0},{0,0,0}};
        Real g_w = mortar.compute_weighted_gap(integrals, num_integrals,
                                                zero_disp, zero_disp,
                                                slave, master, 4, 4);
        CHECK(std::abs(g_w) > 0.0, "Mortar: nonzero weighted gap between offset surfaces");
    }

    // Test 4: Stick regime
    {
        Real gap = -0.01;
        Real slip[2] = {0.0001, 0.0};
        Real R_n = 0.0;
        Real R_t[2] = {0.0, 0.0};
        bool is_sliding = false;
        mortar.contact_residual(gap, 0.0, 0.3, slip, R_n, R_t, is_sliding);
        CHECK_NEAR(R_n, 10000.0, 1e-4, "Mortar: normal residual = eps_n * |gap|");
        CHECK(!is_sliding, "Mortar: stick regime for small slip");
        CHECK_NEAR(R_t[0], 1.0e5 * 0.0001, 1e-4, "Mortar: tangential residual = eps_t * slip");
    }

    // Test 5: Slip regime
    {
        Real gap = -0.01;
        Real slip[2] = {1.0, 0.0};
        Real R_n = 0.0;
        Real R_t[2] = {0.0, 0.0};
        bool is_sliding = false;
        mortar.contact_residual(gap, 0.0, 0.3, slip, R_n, R_t, is_sliding);
        CHECK(is_sliding, "Mortar: sliding for large tangent slip");
        Real Rt_mag = std::sqrt(R_t[0]*R_t[0] + R_t[1]*R_t[1]);
        CHECK_NEAR(Rt_mag, 0.3 * R_n, 1e-4,
                   "Mortar: tangent force = mu * f_n at Coulomb limit");
    }

    // Test 6: No residual for positive gap
    {
        Real gap = 0.01;
        Real slip[2] = {0.1, 0.0};
        Real R_n = 999.0;
        Real R_t[2] = {999.0, 999.0};
        bool is_sliding = true;
        mortar.contact_residual(gap, 0.0, 0.3, slip, R_n, R_t, is_sliding);
        CHECK_NEAR(R_n, 0.0, 1e-10, "Mortar: zero normal residual for positive gap");
        CHECK_NEAR(R_t[0], 0.0, 1e-10, "Mortar: zero tangent residual for positive gap");
        CHECK(!is_sliding, "Mortar: not sliding when separated");
    }

    // Test 7: Contact tangent stiffness
    {
        Real gap = -0.01;
        Real slip[2] = {0.0001, 0.0};
        Real normal[3] = {0.0, 0.0, 1.0};
        Real tangent[9] = {0};
        mortar.contact_tangent(gap, slip, normal, tangent);
        CHECK_NEAR(tangent[8], 1.0e6, 1e-2,
                   "Mortar: K_nn component = epsilon_n for z-normal");
        CHECK(tangent[0] > 0.0, "Mortar: tangent stiffness has tangential contribution");
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== Wave 21 Contact Refinements Test ===\n";

    test_automatic_contact_detection();
    test_contact_2d();
    test_sph_contact();
    test_airbag_fabric_contact();
    test_contact_heat_generation();
    test_mortar_contact_friction();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << (tests_passed + tests_failed) << " tests passed\n";
    return tests_failed > 0 ? 1 : 0;
}
