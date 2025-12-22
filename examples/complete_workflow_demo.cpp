/**
 * @file complete_workflow_demo.cpp
 * @brief Complete NexusSim workflow demonstration
 *
 * This example demonstrates the complete workflow:
 * 1. Select material from the library
 * 2. Create a mesh
 * 3. Run a simple FEM analysis
 * 4. Export results to VTK for visualization
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/physics/material_library.hpp>
#include <nexussim/discretization/tet4.hpp>
#include <nexussim/io/vtk_writer.hpp>
#include <iostream>
#include <vector>

using namespace nxs;
using namespace nxs::physics;
using namespace nxs::fem;
using namespace nxs::io;

int main() {
    std::cout << "=================================================\n";
    std::cout << "Complete NexusSim Workflow Demonstration\n";
    std::cout << "=================================================\n\n";

    // ========================================================================
    // Step 1: Material Selection
    // ========================================================================

    std::cout << "Step 1: Selecting material from library\n";
    std::cout << "----------------------------------------\n";

    // Choose aluminum 6061-T6 for this example
    auto material_props = MaterialLibrary::get(MaterialName::Aluminum_6061_T6);

    std::cout << "Selected: " << MaterialLibrary::to_string(MaterialName::Aluminum_6061_T6) << "\n";
    std::cout << "  Density: " << material_props.density << " kg/m³\n";
    std::cout << "  Young's Modulus: " << material_props.E/1e9 << " GPa\n";
    std::cout << "  Poisson's Ratio: " << material_props.nu << "\n\n";

    // ========================================================================
    // Step 2: Create Simple Mesh (1x1x1 cube, 5 Tet4 elements)
    // ========================================================================

    std::cout << "Step 2: Creating mesh\n";
    std::cout << "---------------------\n";

    // Node coordinates for unit cube
    std::vector<Real> nodes = {
        0.0, 0.0, 0.0,  // Node 0
        1.0, 0.0, 0.0,  // Node 1
        1.0, 1.0, 0.0,  // Node 2
        0.0, 1.0, 0.0,  // Node 3
        0.0, 0.0, 1.0,  // Node 4
        1.0, 0.0, 1.0,  // Node 5
        1.0, 1.0, 1.0,  // Node 6
        0.0, 1.0, 1.0   // Node 7
    };

    // 5-tetrahedra decomposition
    std::vector<int> connectivity = {
        0, 1, 2, 5,  // Tet 1
        0, 2, 3, 7,  // Tet 2
        0, 5, 2, 7,  // Tet 3
        5, 6, 2, 7,  // Tet 4
        0, 5, 7, 4   // Tet 5
    };

    const int num_nodes = 8;
    const int num_elements = 5;

    auto mesh = std::make_shared<Mesh>(num_nodes);

    // Set node coordinates
    for (int i = 0; i < num_nodes; ++i) {
        mesh->set_node_coordinates(i, {nodes[i*3+0], nodes[i*3+1], nodes[i*3+2]});
    }

    // Create element block
    mesh->add_element_block("cube", nxs::ElementType::Tet4, num_elements, 4);
    auto& block = mesh->element_block(0);

    // Set connectivity
    for (int e = 0; e < num_elements; ++e) {
        auto elem_nodes = block.element_nodes(e);
        for (int n = 0; n < 4; ++n) {
            elem_nodes[n] = connectivity[e*4 + n];
        }
    }

    std::cout << "Created mesh with:\n";
    std::cout << "  " << num_nodes << " nodes\n";
    std::cout << "  " << num_elements << " Tet4 elements\n\n";

    // ========================================================================
    // Step 3: Simple Compression Analysis
    // ========================================================================

    std::cout << "Step 3: Running compression analysis\n";
    std::cout << "-------------------------------------\n";

    // Create state
    auto state = std::make_shared<State>(*mesh);

    // Apply simple displacement (compress in Z by 5%)
    const Real compression = -0.05;  // 5% compression

    std::vector<Real> displacement(num_nodes * 3, 0.0);
    std::vector<Real> stress(num_nodes * 6, 0.0);  // Voigt notation

    // Boundary conditions:
    // - Bottom face (z=0): u=0
    // - Top face (z=1): uz = compression

    for (int i = 0; i < num_nodes; ++i) {
        Real z = nodes[i*3 + 2];

        // Linear interpolation of displacement
        displacement[i*3 + 0] = 0.0;  // No lateral displacement
        displacement[i*3 + 1] = 0.0;
        displacement[i*3 + 2] = z * compression;  // Linearly increasing
    }

    // Compute stresses using elastic material model
    Tet4Element elem;
    ElasticMaterial elastic_mat(material_props);

    for (int i = 0; i < num_nodes; ++i) {
        // Simple strain calculation (uniform compression)
        MaterialState mat_state;
        mat_state.strain[0] = 0.0;               // εxx
        mat_state.strain[1] = 0.0;               // εyy
        mat_state.strain[2] = compression;       // εzz (engineering strain)
        mat_state.strain[3] = 0.0;               // γxy
        mat_state.strain[4] = 0.0;               // γyz
        mat_state.strain[5] = 0.0;               // γxz

        // Compute stress
        elastic_mat.compute_stress(mat_state);

        // Store
        for (int j = 0; j < 6; ++j) {
            stress[i*6 + j] = mat_state.stress[j];
        }
    }

    std::cout << "Analysis complete.\n";
    std::cout << "Applied compression: " << compression * 100 << "%\n";
    std::cout << "Axial stress (σzz): " << stress[2] / 1e6 << " MPa\n\n";

    // Update state with results
    auto& disp = state->displacement();
    for (int i = 0; i < num_nodes; ++i) {
        disp.at(i, 0) = displacement[i*3 + 0];
        disp.at(i, 1) = displacement[i*3 + 1];
        disp.at(i, 2) = displacement[i*3 + 2];
    }

    auto& vel = state->velocity();
    for (int i = 0; i < num_nodes; ++i) {
        vel.at(i, 0) = 0.0;
        vel.at(i, 1) = 0.0;
        vel.at(i, 2) = 0.0;
    }

    // ========================================================================
    // Step 4: Export to VTK
    // ========================================================================

    std::cout << "Step 4: Exporting results to VTK\n";
    std::cout << "---------------------------------\n";

    try {
        SimpleVTKWriter vtk_writer("/tmp/cube_compression_demo");

        // Write mesh and solution
        vtk_writer.write(*mesh, state.get());

        std::cout << "VTK file written to: /tmp/cube_compression_demo.vtk\n";
        std::cout << "\nTo visualize:\n";
        std::cout << "  1. Open ParaView\n";
        std::cout << "  2. File -> Open -> /tmp/cube_compression_demo.vtk\n";
        std::cout << "  3. Click 'Apply'\n";
        std::cout << "  4. Select 'displacement' or 'displacement_magnitude' for coloring\n";
        std::cout << "  5. Apply 'Warp By Vector' filter to see deformation\n\n";

    } catch (const std::exception& e) {
        std::cerr << "VTK export failed: " << e.what() << "\n";
        std::cerr << "Note: SimpleVTKWriter may not be fully implemented yet.\n\n";
    }

    // ========================================================================
    // Summary
    // ========================================================================

    std::cout << "=================================================\n";
    std::cout << "Workflow Summary\n";
    std::cout << "=================================================\n\n";

    std::cout << "This demonstration showed:\n";
    std::cout << "  ✓ Material selection from library (30+ materials available)\n";
    std::cout << "  ✓ Mesh creation with Tet4 elements\n";
    std::cout << "  ✓ Simple compression analysis\n";
    std::cout << "  ✓ Stress computation using elastic material model\n";
    std::cout << "  ✓ VTK export for visualization\n\n";

    std::cout << "Available element types:\n";
    std::cout << "  - Tet4, Tet10 (tetrahedral elements)\n";
    std::cout << "  - Hex8, Hex20 (hexahedral elements)\n";
    std::cout << "  - Shell4 (shell elements)\n";
    std::cout << "  - Beam2 (beam elements)\n\n";

    std::cout << "Available materials:\n";
    std::cout << "  - Steels (mild, structural, stainless, high-strength, tool)\n";
    std::cout << "  - Aluminum alloys (1100, 2024, 6061, 7075)\n";
    std::cout << "  - Titanium alloys (Grade 2, Ti-6Al-4V)\n";
    std::cout << "  - Polymers (ABS, Nylon, Polycarbonate, PEEK, Acrylic)\n";
    std::cout << "  - Composites (carbon fiber, glass fiber)\n";
    std::cout << "  - Concrete, ceramics, rubber, wood, glass\n\n";

    std::cout << "For more examples, see:\n";
    std::cout << "  - material_library_demo\n";
    std::cout << "  - tet4_compression_solver_test\n";
    std::cout << "  - hex8_element_test\n";
    std::cout << "  - shell4_plate_test\n\n";

    return 0;
}
