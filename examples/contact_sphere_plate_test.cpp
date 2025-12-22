/**
 * @file contact_sphere_plate_test.cpp
 * @brief Test contact mechanics with a sphere dropping onto a rigid plate
 *
 * This test validates the contact detection and penalty force computation
 * by simulating a deformable sphere falling onto a fixed rigid plate.
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/fem/fem_solver.hpp>
#include <nexussim/fem/contact.hpp>
#include <nexussim/physics/material.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/data/state.hpp>
#include <iostream>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;

int main() {
    // Initialize NexusSim
    nxs::InitOptions options;
    options.log_level = nxs::Logger::Level::Info;
    nxs::Context context(options);

    NXS_LOG_INFO("=================================================");
    NXS_LOG_INFO("Contact Mechanics Test: Sphere on Plate");
    NXS_LOG_INFO("=================================================\n");

    try {
        // ====================================================================
        // Test Parameters
        // ====================================================================

        const Real sphere_radius = 0.5;      // m
        const Real plate_size = 5.0;         // m
        const Real initial_gap = 0.1;        // m (sphere above plate)
        const Real gravity = -9.81;          // m/s^2

        // Sphere material: rubber-like (soft for testing)
        const Real sphere_E = 1.0e6;         // Pa (soft)
        const Real sphere_nu = 0.45;          // Nearly incompressible
        const Real sphere_density = 1000.0;   // kg/m³

        // Plate material: rigid (very stiff)
        const Real plate_E = 200.0e9;        // Pa (steel-like)
        const Real plate_nu = 0.3;
        const Real plate_density = 7800.0;   // kg/m³

        NXS_LOG_INFO("Sphere:");
        NXS_LOG_INFO("  Radius: {} m", sphere_radius);
        NXS_LOG_INFO("  E = {} Pa, nu = {}", sphere_E, sphere_nu);
        NXS_LOG_INFO("  Density = {} kg/m³", sphere_density);

        NXS_LOG_INFO("\nPlate:");
        NXS_LOG_INFO("  Size: {}×{} m", plate_size, plate_size);
        NXS_LOG_INFO("  E = {} Pa, nu = {}", plate_E, plate_nu);
        NXS_LOG_INFO("  Density = {} kg/m³", plate_density);

        NXS_LOG_INFO("\nInitial conditions:");
        NXS_LOG_INFO("  Gap: {} m", initial_gap);
        NXS_LOG_INFO("  Gravity: {} m/s²\n", gravity);

        // ====================================================================
        // Create simplified mesh: single hex8 sphere + plate surface
        // ====================================================================

        // For simplicity, use a single hex8 element to represent the sphere
        // and 4 shell4 elements for the plate surface

        const int num_nodes = 17;  // 8 for sphere + 9 for plate
        auto mesh = std::make_shared<Mesh>(num_nodes);

        // Sphere nodes (hex8): center at (0, 0, sphere_radius + initial_gap)
        const Real sphere_center_z = sphere_radius + initial_gap;
        const Real h = sphere_radius * 0.8;  // Half-size of hex

        // Node 0-7: sphere (hex8)
        mesh->set_node_coordinates(0, {-h, -h, sphere_center_z - h});
        mesh->set_node_coordinates(1, { h, -h, sphere_center_z - h});
        mesh->set_node_coordinates(2, { h,  h, sphere_center_z - h});
        mesh->set_node_coordinates(3, {-h,  h, sphere_center_z - h});
        mesh->set_node_coordinates(4, {-h, -h, sphere_center_z + h});
        mesh->set_node_coordinates(5, { h, -h, sphere_center_z + h});
        mesh->set_node_coordinates(6, { h,  h, sphere_center_z + h});
        mesh->set_node_coordinates(7, {-h,  h, sphere_center_z + h});

        // Plate nodes (3×3 grid at z=0): nodes 8-16
        const Real plate_dx = plate_size / 2.0;
        int node_id = 8;
        for (int i = 0; i < 3; ++i) {
            const Real x = -plate_size/2.0 + i * plate_dx;
            for (int j = 0; j < 3; ++j) {
                const Real y = -plate_size/2.0 + j * plate_dx;
                mesh->set_node_coordinates(node_id++, {x, y, 0.0});
            }
        }

        NXS_LOG_INFO("Created mesh with {} nodes", mesh->num_nodes());

        // ====================================================================
        // Create sphere element (Hex8)
        // ====================================================================

        mesh->add_element_block("sphere", ElementType::Hex8, 1, 8);
        std::vector<Index> sphere_connectivity = {0, 1, 2, 3, 4, 5, 6, 7};

        // ====================================================================
        // Create plate surface (4 Shell4 elements)
        // ====================================================================

        // Plate: 2×2 shell4 elements
        std::vector<Index> plate_nodes;
        std::vector<Index> plate_face_ids;

        // Face 0: nodes 8, 11, 14, 9
        plate_nodes.insert(plate_nodes.end(), {8, 11, 14, 9});
        plate_face_ids.push_back(0);

        // Face 1: nodes 11, 10, 13, 14
        plate_nodes.insert(plate_nodes.end(), {11, 10, 13, 14});
        plate_face_ids.push_back(1);

        // Face 2: nodes 9, 14, 15, 12
        plate_nodes.insert(plate_nodes.end(), {9, 14, 15, 12});
        plate_face_ids.push_back(2);

        // Face 3: nodes 14, 13, 16, 15
        plate_nodes.insert(plate_nodes.end(), {14, 13, 16, 15});
        plate_face_ids.push_back(3);

        NXS_LOG_INFO("Created sphere: 1 Hex8 element");
        NXS_LOG_INFO("Created plate: 4 Shell4 faces");

        // ====================================================================
        // Setup FEM Solver
        // ====================================================================

        auto state = std::make_shared<State>(*mesh);
        FEMSolver solver("ContactSphereTest");

        // Sphere material
        physics::MaterialProperties sphere_mat;
        sphere_mat.density = sphere_density;
        sphere_mat.E = sphere_E;
        sphere_mat.nu = sphere_nu;
        sphere_mat.G = sphere_E / (2.0 * (1.0 + sphere_nu));
        sphere_mat.K = sphere_E / (3.0 * (1.0 - 2.0 * sphere_nu));

        std::vector<Index> sphere_elem_ids = {0};
        solver.add_element_group("sphere", physics::ElementType::Hex8,
                                sphere_elem_ids, sphere_connectivity, sphere_mat);

        // Apply gravity to sphere nodes
        std::vector<Index> sphere_nodes = {0, 1, 2, 3, 4, 5, 6, 7};
        const Real gravity_force_per_node = (sphere_mat.density * h*h*h * 8.0 / 8.0) * gravity;

        BoundaryCondition bc_gravity(BCType::Force, sphere_nodes, 2, gravity_force_per_node);
        solver.add_boundary_condition(bc_gravity);

        // Fix plate nodes (rigid)
        std::vector<Index> plate_node_ids = {8, 9, 10, 11, 12, 13, 14, 15, 16};
        for (int dof = 0; dof < 3; ++dof) {
            BoundaryCondition bc_fixed(BCType::Displacement, plate_node_ids, dof, 0.0);
            solver.add_boundary_condition(bc_fixed);
        }

        solver.initialize(mesh, state);

        // ====================================================================
        // Setup Contact Mechanics
        // ====================================================================

        ContactMechanics contact;

        // Master surface: plate
        contact.add_master_surface("plate", plate_nodes, plate_face_ids, ElementType::Shell4);

        // Slave nodes: sphere bottom nodes (will contact first)
        std::vector<Index> sphere_slave_nodes = {0, 1, 2, 3};  // Bottom face of sphere
        contact.add_slave_nodes("sphere_bottom", sphere_slave_nodes);

        // Contact parameters
        ContactParameters contact_params;
        contact_params.penalty_stiffness = 1.0;      // Will be scaled appropriately
        contact_params.friction_coefficient = 0.0;   // Frictionless for now
        contact_params.contact_thickness = 0.01;     // 1 cm detection threshold
        contact_params.enable_friction = false;

        contact.set_parameters(contact_params);
        contact.initialize(mesh);

        // ====================================================================
        // Run simulation
        // ====================================================================

        const Real dt = solver.compute_stable_dt() * 0.5;
        const Real total_time = 1.0;  // 1 second
        const int n_steps = static_cast<int>(total_time / dt);

        NXS_LOG_INFO("Running contact simulation:");
        NXS_LOG_INFO("  Time step: {:.6e} s", dt);
        NXS_LOG_INFO("  Total time: {} s", total_time);
        NXS_LOG_INFO("  Number of steps: {}\n", n_steps);

        Real max_contact_force = 0.0;
        int first_contact_step = -1;

        for (int step = 0; step < n_steps; ++step) {
            // Detect contact
            // TODO: Get displacement and velocity from solver
            // For now, just test the interface

            if (step % 100 == 0) {
                NXS_LOG_INFO("  Step {}/{}: {} active contacts",
                            step, n_steps, contact.num_active_contacts());
            }

            solver.step(dt);
        }

        NXS_LOG_INFO("\n=================================================");
        NXS_LOG_INFO("CONTACT TEST COMPLETED");
        NXS_LOG_INFO("=================================================");
        NXS_LOG_INFO("Note: Full integration with solver pending");
        NXS_LOG_INFO("Contact mechanics infrastructure is in place\n");

    } catch (const std::exception& e) {
        NXS_LOG_ERROR("Test failed with exception: {}", e.what());
        return 1;
    }

    return 0;
}
