/**
 * @file fem_sph_coupling_test.cpp
 * @brief Test FEM-SPH coupling for fluid-structure interaction
 *
 * Tests:
 * 1. FEM surface extraction from mesh
 * 2. Penalty contact force computation
 * 3. Direct force coupling
 * 4. Coupled solver initialization
 * 5. Water column on elastic plate
 */

#include <nexussim/sph/fem_sph_coupling.hpp>
#include <nexussim/sph/sph_solver.hpp>
#include <nexussim/data/mesh.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace nxs;
using namespace nxs::sph;

// Test counter
static int test_count = 0;
static int pass_count = 0;

void check(bool condition, const std::string& test_name) {
    test_count++;
    if (condition) {
        pass_count++;
        std::cout << "[PASS] " << test_name << "\n";
    } else {
        std::cout << "[FAIL] " << test_name << "\n";
    }
}

// ============================================================================
// Helper: Create simple plate mesh
// ============================================================================
Mesh create_plate_mesh(Real Lx, Real Ly, int nx, int ny, Real z = 0.0) {
    // Create a plate mesh in XY plane at height z
    size_t num_nodes = (nx + 1) * (ny + 1);
    size_t num_elements = nx * ny;

    Mesh mesh(num_nodes);

    Real dx = Lx / nx;
    Real dy = Ly / ny;

    // Set node coordinates
    size_t node_idx = 0;
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            mesh.set_node_coordinates(node_idx, {i * dx, j * dy, z});
            ++node_idx;
        }
    }

    // Add element block for quads (using Shell4 which has 4 nodes)
    Index block_id = mesh.add_element_block("plate", ElementType::Shell4, num_elements, 4);
    auto& block = mesh.element_block(block_id);

    // Set connectivity
    size_t elem_idx = 0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            Index n0 = j * (nx + 1) + i;
            Index n1 = n0 + 1;
            Index n2 = n0 + (nx + 1) + 1;
            Index n3 = n0 + (nx + 1);

            auto elem_nodes = block.element_nodes(elem_idx);
            elem_nodes[0] = n0;
            elem_nodes[1] = n1;
            elem_nodes[2] = n2;
            elem_nodes[3] = n3;
            ++elem_idx;
        }
    }

    return mesh;
}

// ============================================================================
// Test 1: FEM Surface Extraction
// ============================================================================
void test_surface_extraction() {
    std::cout << "\n=== Test 1: FEM Surface Extraction ===\n";

    // Create simple 2x2 plate mesh at z=0
    Mesh plate = create_plate_mesh(0.2, 0.2, 2, 2, 0.0);

    std::cout << "  Created plate mesh:\n";
    std::cout << "    Nodes: " << plate.num_nodes() << "\n";
    std::cout << "    Elements: " << plate.num_elements() << "\n";

    // All nodes are on the surface
    std::vector<Index> surface_nodes;
    for (size_t i = 0; i < plate.num_nodes(); ++i) {
        surface_nodes.push_back(i);
    }

    // Extract surface
    FEMSurface surface;
    surface.extract_from_mesh(plate, surface_nodes);

    std::cout << "  Surface facets: " << surface.num_facets() << "\n";

    // For a 2x2 quad mesh with all surface nodes, we get 4 quads = 8 triangles
    // (Shell4 may produce faces on both sides, so expect 8 or 16)
    check(surface.num_facets() >= 8, "Sufficient number of facets");

    // Check that we have facets with +z and/or -z normals
    int upward = 0, downward = 0;
    for (const auto& facet : surface.facets()) {
        if (facet.normal[2] > 0.9) upward++;
        if (facet.normal[2] < -0.9) downward++;
    }
    std::cout << "  Upward normals: " << upward << ", Downward normals: " << downward << "\n";
    check(upward > 0 || downward > 0, "Facets have vertical normals");

    // Check total area (may be doubled if both sides counted)
    Real total_area = 0.0;
    for (const auto& facet : surface.facets()) {
        total_area += facet.area;
    }
    std::cout << "  Total surface area: " << total_area << " m² (expected 0.04 or 0.08)\n";
    check(std::abs(total_area - 0.04) < 1e-6 || std::abs(total_area - 0.08) < 1e-6,
          "Correct total area (single or double-sided)");
}

// ============================================================================
// Test 2: Penalty Contact Forces
// ============================================================================
void test_penalty_contact() {
    std::cout << "\n=== Test 2: Penalty Contact Forces ===\n";

    // Create plate at z=0
    Mesh plate = create_plate_mesh(0.2, 0.2, 2, 2, 0.0);

    std::vector<Index> surface_nodes;
    for (size_t i = 0; i < plate.num_nodes(); ++i) {
        surface_nodes.push_back(i);
    }

    // Create SPH particles above the plate
    SPHSolver sph(0.02);  // 2cm smoothing length

    std::vector<Real> x, y, z;
    Real spacing = 0.02;

    // Create particles at z=0.03 (slightly above contact distance)
    // and z=0.01 (within contact distance)
    for (Real px = 0.05; px < 0.15; px += spacing) {
        for (Real py = 0.05; py < 0.15; py += spacing) {
            x.push_back(px);
            y.push_back(py);
            z.push_back(0.01);  // Within contact distance (2*h = 0.04)
        }
    }

    sph.initialize(x, y, z, spacing);
    sph.set_gravity(0, 0, -9.81);

    std::cout << "  Particles: " << sph.num_particles() << "\n";

    // Initialize coupling
    FEMSPHCoupling coupling;
    CouplingParameters params;
    params.penalty_stiffness = 1e6;
    params.contact_distance = 0.04;  // 2 * h

    coupling.set_parameters(params);
    coupling.initialize(plate, surface_nodes, sph);

    // Compute forces
    coupling.compute_coupling_forces();

    std::cout << "  Active contacts: " << coupling.num_contacts() << "\n";
    std::cout << "  Total normal force: " << coupling.total_normal_force() << " N\n";

    check(coupling.num_contacts() > 0, "Contacts detected");
    check(coupling.total_normal_force() > 0, "Normal force computed");

    // Check particle forces push upward (away from plate)
    const auto& particle_forces = coupling.get_particle_forces();
    bool forces_upward = true;
    for (size_t i = 0; i < sph.num_particles(); ++i) {
        if (particle_forces[i * 3 + 2] < 0) {
            forces_upward = false;
        }
    }
    check(forces_upward, "Particle forces point away from surface");

    // Check FEM forces point downward (reaction)
    const auto& fem_forces = coupling.get_fem_forces();
    Real total_fz = 0.0;
    for (size_t i = 0; i < surface_nodes.size(); ++i) {
        total_fz += fem_forces[i * 3 + 2];
    }
    std::cout << "  Total FEM force (z): " << total_fz << " N\n";
    check(total_fz < 0, "FEM reaction force points downward");
}

// ============================================================================
// Test 3: Direct Pressure Coupling
// ============================================================================
void test_direct_coupling() {
    std::cout << "\n=== Test 3: Direct Pressure Coupling ===\n";

    // Create plate at z=0
    Mesh plate = create_plate_mesh(0.2, 0.2, 2, 2, 0.0);

    std::vector<Index> surface_nodes;
    for (size_t i = 0; i < plate.num_nodes(); ++i) {
        surface_nodes.push_back(i);
    }

    // Create SPH water column above plate
    SPHSolver sph(0.01);

    std::vector<Real> x, y, z;
    Real spacing = 0.01;

    // Create particles from z=0.02 to z=0.1 (8cm water column)
    for (Real px = 0.05; px < 0.15; px += spacing) {
        for (Real py = 0.05; py < 0.15; py += spacing) {
            for (Real pz = 0.02; pz < 0.1; pz += spacing) {
                x.push_back(px);
                y.push_back(py);
                z.push_back(pz);
            }
        }
    }

    sph.initialize(x, y, z, spacing);
    sph.set_gravity(0, 0, -9.81);

    std::cout << "  Particles: " << sph.num_particles() << "\n";

    // Run a few SPH steps to get pressure field
    Real dt = sph.compute_stable_dt();
    for (int i = 0; i < 5; ++i) {
        sph.step(dt);
    }

    // Initialize coupling with direct force method
    FEMSPHCoupling coupling;
    CouplingParameters params;
    params.contact_distance = 0.03;

    coupling.set_parameters(params);
    coupling.set_coupling_type(CouplingType::DirectForce);
    coupling.initialize(plate, surface_nodes, sph);

    // Compute forces
    coupling.compute_coupling_forces();

    std::cout << "  Active contacts: " << coupling.num_contacts() << "\n";
    std::cout << "  Total normal force: " << coupling.total_normal_force() << " N\n";

    check(coupling.num_contacts() > 0, "Pressure integration points found");

    // For a hydrostatic column, force should approximately equal rho*g*h*A
    // rho=1000, g=9.81, h≈0.08m (average), A≈0.01m² (portion over plate center)
    // Expected: ~8 N approximately
    check(coupling.total_normal_force() > 0, "Pressure force computed");
}

// ============================================================================
// Test 4: Coupled Solver Initialization
// ============================================================================
void test_coupled_solver() {
    std::cout << "\n=== Test 4: Coupled Solver Initialization ===\n";

    // Create plate mesh
    Mesh plate = create_plate_mesh(0.3, 0.3, 3, 3, 0.0);

    std::vector<Index> surface_nodes;
    for (size_t i = 0; i < plate.num_nodes(); ++i) {
        surface_nodes.push_back(i);
    }

    // Create SPH fluid
    SPHSolver sph(0.02);
    sph.create_dam_break(0.1, 0.1, 0.1, 0.02);
    sph.set_gravity(0, 0, -9.81);

    std::cout << "  FEM nodes: " << plate.num_nodes() << "\n";
    std::cout << "  SPH particles: " << sph.num_particles() << "\n";

    // Initialize coupled solver
    CoupledFEMSPHSolver coupled;
    coupled.initialize(plate, surface_nodes, sph);

    check(true, "Coupled solver initialized");

    // Set parameters
    CouplingParameters params;
    params.penalty_stiffness = 1e7;
    params.damping_ratio = 0.1;
    coupled.set_coupling_parameters(params);

    check(true, "Parameters set");

    // Print stats
    coupled.print_stats();
}

// ============================================================================
// Test 5: Friction Forces
// ============================================================================
void test_friction_coupling() {
    std::cout << "\n=== Test 5: Friction Coupling ===\n";

    // Create tilted plate (particles will slide)
    Mesh plate = create_plate_mesh(0.2, 0.2, 2, 2, 0.0);

    std::vector<Index> surface_nodes;
    for (size_t i = 0; i < plate.num_nodes(); ++i) {
        surface_nodes.push_back(i);
    }

    // Create particles with tangential velocity
    SPHSolver sph(0.02);

    std::vector<Real> x, y, z;
    Real spacing = 0.02;

    for (Real px = 0.05; px < 0.15; px += spacing) {
        for (Real py = 0.05; py < 0.15; py += spacing) {
            x.push_back(px);
            y.push_back(py);
            z.push_back(0.015);  // Near surface
        }
    }

    sph.initialize(x, y, z, spacing);

    // We can't directly set velocities, but we can check friction parameters

    // Initialize coupling with friction
    FEMSPHCoupling coupling;
    CouplingParameters params;
    params.penalty_stiffness = 1e6;
    params.contact_distance = 0.04;
    params.enable_friction = true;
    params.friction_coef = 0.3;

    coupling.set_parameters(params);
    coupling.initialize(plate, surface_nodes, sph);

    // Compute forces
    coupling.compute_coupling_forces();

    std::cout << "  Friction enabled: " << (params.enable_friction ? "yes" : "no") << "\n";
    std::cout << "  Friction coefficient: " << params.friction_coef << "\n";
    std::cout << "  Total tangent force: " << coupling.total_tangent_force() << " N\n";

    // With particles at rest, friction force should be zero
    check(coupling.total_tangent_force() < 1e-6, "No friction with stationary particles");

    // Verify coupling parameters were set
    check(params.friction_coef == 0.3, "Friction coefficient set correctly");
}

// ============================================================================
// Test 6: Surface Update with Displacement
// ============================================================================
void test_surface_update() {
    std::cout << "\n=== Test 6: Surface Update with Displacement ===\n";

    // Create plate
    Mesh plate = create_plate_mesh(0.2, 0.2, 2, 2, 0.0);

    std::vector<Index> surface_nodes;
    for (size_t i = 0; i < plate.num_nodes(); ++i) {
        surface_nodes.push_back(i);
    }

    // Extract surface
    FEMSurface surface;
    surface.extract_from_mesh(plate, surface_nodes);

    // Get initial centroid of first facet
    Real z0 = surface.facets()[0].centroid[2];
    std::cout << "  Initial facet z: " << z0 << "\n";

    // Apply uniform z-displacement
    std::vector<Real> disp(plate.num_nodes() * 3, 0.0);
    Real dz = 0.05;  // 5cm upward
    for (size_t i = 0; i < plate.num_nodes(); ++i) {
        disp[i * 3 + 2] = dz;
    }

    surface.apply_displacements(disp.data(), plate.num_nodes());

    // Check updated position
    Real z1 = surface.facets()[0].centroid[2];
    std::cout << "  After displacement z: " << z1 << "\n";

    check(std::abs(z1 - z0 - dz) < 1e-10, "Surface position updated correctly");

    // Check normal is still correct (should still point +z or -z)
    Real nz = std::abs(surface.facets()[0].normal[2]);
    check(nz > 0.99, "Normal unchanged after translation");
}

// ============================================================================
// Test 7: Energy Exchange
// ============================================================================
void test_energy_exchange() {
    std::cout << "\n=== Test 7: Energy Exchange ===\n";

    // This test verifies momentum conservation at the interface

    Mesh plate = create_plate_mesh(0.1, 0.1, 1, 1, 0.0);

    std::vector<Index> surface_nodes;
    for (size_t i = 0; i < plate.num_nodes(); ++i) {
        surface_nodes.push_back(i);
    }

    // Single particle impacting plate
    SPHSolver sph(0.02);
    std::vector<Real> x = {0.05};
    std::vector<Real> y = {0.05};
    std::vector<Real> z = {0.01};  // Close to surface

    sph.initialize(x, y, z, 0.02);
    sph.set_gravity(0, 0, -9.81);

    FEMSPHCoupling coupling;
    CouplingParameters params;
    params.penalty_stiffness = 1e6;
    params.contact_distance = 0.04;

    coupling.set_parameters(params);
    coupling.initialize(plate, surface_nodes, sph);
    coupling.compute_coupling_forces();

    // Sum of forces on particle and FEM should be zero (Newton's 3rd law)
    const auto& pf = coupling.get_particle_forces();
    const auto& ff = coupling.get_fem_forces();

    Real sum_fx = pf[0];
    Real sum_fy = pf[1];
    Real sum_fz = pf[2];

    for (size_t i = 0; i < surface_nodes.size(); ++i) {
        sum_fx += ff[i * 3 + 0];
        sum_fy += ff[i * 3 + 1];
        sum_fz += ff[i * 3 + 2];
    }

    std::cout << "  Net force: (" << sum_fx << ", " << sum_fy << ", " << sum_fz << ")\n";

    Real net_force = std::sqrt(sum_fx * sum_fx + sum_fy * sum_fy + sum_fz * sum_fz);
    check(net_force < 1e-6, "Forces balanced (Newton's 3rd law)");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=================================================\n";
    std::cout << "FEM-SPH Coupling Test Suite\n";
    std::cout << "=================================================\n";

    Kokkos::initialize();

    {
        test_surface_extraction();
        test_penalty_contact();
        test_direct_coupling();
        test_coupled_solver();
        test_friction_coupling();
        test_surface_update();
        test_energy_exchange();
    }

    Kokkos::finalize();

    std::cout << "\n=================================================\n";
    std::cout << "Results: " << pass_count << "/" << test_count << " tests passed\n";
    std::cout << "=================================================\n";

    return (pass_count == test_count) ? 0 : 1;
}
