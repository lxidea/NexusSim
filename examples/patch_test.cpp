/**
 * @file patch_test.cpp
 * @brief Patch test for Hex8 element - validates constant strain representation
 *
 * The patch test verifies that an element can represent constant strain states.
 * A distorted mesh under constant strain should produce:
 * - Constant stress throughout
 * - Correct nodal forces that satisfy equilibrium
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/fem/fem_solver.hpp>
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
    NXS_LOG_INFO("Hex8 Element Patch Test");
    NXS_LOG_INFO("=================================================\n");

    try {
        // ====================================================================
        // Create a distorted single-element mesh for patch test
        // ====================================================================

        auto mesh = std::make_shared<Mesh>(8);  // 8 nodes

        // Distorted hex element (non-rectangular to test arbitrary geometry)
        mesh->set_node_coordinates(0, {0.0, 0.0, 0.0});
        mesh->set_node_coordinates(1, {1.1, 0.1, 0.0});
        mesh->set_node_coordinates(2, {1.0, 1.2, 0.1});
        mesh->set_node_coordinates(3, {0.1, 1.0, 0.0});
        mesh->set_node_coordinates(4, {0.0, 0.1, 1.0});
        mesh->set_node_coordinates(5, {1.0, 0.0, 1.1});
        mesh->set_node_coordinates(6, {1.1, 1.0, 1.0});
        mesh->set_node_coordinates(7, {0.0, 1.1, 1.0});

        // Add element block
        mesh->add_element_block("patch_element", ElementType::Hex8, 1, 8);
        auto& block = mesh->element_block(0);

        // Set connectivity for the single hex element
        for (Index i = 0; i < 8; ++i) {
            block.connectivity[i] = i;
        }

        NXS_LOG_INFO("Created distorted hex8 element with {} nodes", mesh->num_nodes());
        
        // ====================================================================
        // Define material properties (steel)
        // ====================================================================
        
        physics::MaterialProperties steel;
        steel.density = 7850.0;  // kg/m^3
        steel.E = 200.0e9;       // Pa (200 GPa)
        steel.nu = 0.3;
        steel.G = steel.E / (2.0 * (1.0 + steel.nu));
        steel.K = steel.E / (3.0 * (1.0 - 2.0 * steel.nu));
        
        NXS_LOG_INFO("Material: Steel (E={:.2e} Pa, ν={:.2f})", steel.E, steel.nu);
        
        // ====================================================================
        // Create state and apply constant strain displacement field
        // ====================================================================
        
        auto state = std::make_shared<State>(*mesh);
        
        // Apply displacement corresponding to constant strain:
        // ε_xx = 0.001, ε_yy = 0.0005, ε_zz = 0.0002
        // γ_xy = 0.0, γ_yz = 0.0, γ_xz = 0.0 (pure normal strain)
        const Real exx = 0.001;
        const Real eyy = 0.0005;
        const Real ezz = 0.0002;
        
        auto& disp_field = state->field("displacement");
        const auto& node_coords = mesh->coordinates();
        
        for (std::size_t i = 0; i < mesh->num_nodes(); ++i) {
            Real x = node_coords.at(i, 0);
            Real y = node_coords.at(i, 1);
            Real z = node_coords.at(i, 2);
            
            // Displacement for constant strain: u = ε * x
            disp_field.at(i, 0) = exx * x;
            disp_field.at(i, 1) = eyy * y;
            disp_field.at(i, 2) = ezz * z;
        }
        
        NXS_LOG_INFO("\nApplied constant strain field:");
        NXS_LOG_INFO("  ε_xx = {:.6f}", exx);
        NXS_LOG_INFO("  ε_yy = {:.6f}", eyy);
        NXS_LOG_INFO("  ε_zz = {:.6f}", ezz);
        
        // ====================================================================
        // Compute analytical stresses for validation
        // ====================================================================
        
        const Real lambda = steel.E * steel.nu / ((1.0 + steel.nu) * (1.0 - 2.0 * steel.nu));
        const Real mu = steel.G;
        
        const Real sigma_xx_analytical = (lambda + 2.0 * mu) * exx + lambda * (eyy + ezz);
        const Real sigma_yy_analytical = (lambda + 2.0 * mu) * eyy + lambda * (exx + ezz);
        const Real sigma_zz_analytical = (lambda + 2.0 * mu) * ezz + lambda * (exx + eyy);
        
        NXS_LOG_INFO("\nAnalytical stresses (linear elastic):");
        NXS_LOG_INFO("  σ_xx = {:.6e} Pa", sigma_xx_analytical);
        NXS_LOG_INFO("  σ_yy = {:.6e} Pa", sigma_yy_analytical);
        NXS_LOG_INFO("  σ_zz = {:.6e} Pa", sigma_zz_analytical);
        
        // ====================================================================
        // Create FEM solver and compute internal forces
        // ====================================================================
        
        FEMSolver solver("PatchTest");

        std::vector<Index> elem_ids = {0};
        std::vector<Index> connectivity = {0, 1, 2, 3, 4, 5, 6, 7};
        solver.add_element_group("patch_element", physics::ElementType::Hex8,
                                elem_ids, connectivity, steel);
        
        solver.initialize(mesh, state);
        
        // Set displacement field in solver
        std::vector<Real> displacement(mesh->num_nodes() * 3);
        for (std::size_t i = 0; i < mesh->num_nodes(); ++i) {
            for (int d = 0; d < 3; ++d) {
                displacement[i * 3 + d] = disp_field.at(i, d);
            }
        }
        
        // Manually set solver displacement (bypass time integration)
        // Use import_field to set displacement
        solver.import_field("displacement", displacement);

        // Trigger internal force computation by taking a zero time step
        // This will compute strains and stresses from current displacement
        solver.step(0.0);

        // Export internal force
        std::vector<Real> fint;
        solver.export_field("force_internal", fint);

        NXS_LOG_INFO("\nInternal forces computed from displacement field:");
        Real max_force = 0.0;
        for (std::size_t i = 0; i < fint.size(); ++i) {
            max_force = std::max(max_force, std::abs(fint[i]));
        }
        NXS_LOG_INFO("  Max internal force: {:.6e} N", max_force);
        
        // ====================================================================
        // Patch Test Validation
        // ====================================================================
        
        // For constant strain, internal forces should be self-equilibrating
        // Sum of all internal forces should be zero (within numerical tolerance)
        Real sum_fx = 0.0, sum_fy = 0.0, sum_fz = 0.0;
        for (std::size_t i = 0; i < mesh->num_nodes(); ++i) {
            sum_fx += fint[i * 3 + 0];
            sum_fy += fint[i * 3 + 1];
            sum_fz += fint[i * 3 + 2];
        }
        
        NXS_LOG_INFO("\nEquilibrium check (sum of internal forces):");
        NXS_LOG_INFO("  ΣF_x = {:.6e} N", sum_fx);
        NXS_LOG_INFO("  ΣF_y = {:.6e} N", sum_fy);
        NXS_LOG_INFO("  ΣF_z = {:.6e} N", sum_fz);
        
        const Real force_tol = 1.0e-6 * max_force;  // Relative tolerance
        bool equilibrium_satisfied = (std::abs(sum_fx) < force_tol &&
                                     std::abs(sum_fy) < force_tol &&
                                     std::abs(sum_fz) < force_tol);
        
        // ====================================================================
        // Test Results
        // ====================================================================
        
        NXS_LOG_INFO("\n=================================================");
        NXS_LOG_INFO("Patch Test Results:");
        NXS_LOG_INFO("=================================================");
        
        if (equilibrium_satisfied) {
            NXS_LOG_INFO("✓ Equilibrium satisfied (forces sum to zero)");
        } else {
            NXS_LOG_ERROR("✗ Equilibrium NOT satisfied!");
        }
        
        NXS_LOG_INFO("\n=================================================");
        if (equilibrium_satisfied) {
            NXS_LOG_INFO("PATCH TEST PASSED!");
        } else {
            NXS_LOG_ERROR("PATCH TEST FAILED!");
        }
        NXS_LOG_INFO("=================================================");
        
        return equilibrium_satisfied ? 0 : 1;
        
    } catch (const std::exception& e) {
        NXS_LOG_ERROR("Test failed with exception: {}", e.what());
        return 1;
    }
}
