/**
 * @file simple_mesh_example.cpp
 * @brief Simple example demonstrating basic NexusSim usage
 */

#include <nexussim/nexussim.hpp>
#include <iostream>

int main() {
    // Initialize NexusSim
    nxs::InitOptions options;
    options.log_level = nxs::Logger::Level::Info;
    nxs::Context context(options);

    // Print features
    nxs::features::print_features();

    // Create a simple 2x2x2 hexahedral mesh
    const std::size_t num_nodes = 27;  // 3x3x3 grid
    const std::size_t num_elems = 8;   // 2x2x2 elements

    nxs::Mesh mesh(num_nodes);

    // Set node coordinates (3x3x3 grid)
    std::size_t node_id = 0;
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                nxs::Vec3r coords = {
                    static_cast<nxs::Real>(i),
                    static_cast<nxs::Real>(j),
                    static_cast<nxs::Real>(k)
                };
                mesh.set_node_coordinates(node_id++, coords);
            }
        }
    }

    // Add hex8 element block
    auto block_id = mesh.add_element_block("hex_block", nxs::ElementType::Hex8, num_elems, 8);
    auto& block = mesh.element_block(block_id);

    // Define element connectivity (manual for first element as example)
    // Element 0: nodes 0,1,4,3, 9,10,13,12
    std::vector<std::array<nxs::Index, 8>> hex_connectivity = {
        {0, 1, 4, 3,  9, 10, 13, 12},  // Element 0
        {1, 2, 5, 4, 10, 11, 14, 13},  // Element 1
        {3, 4, 7, 6, 12, 13, 16, 15},  // Element 2
        {4, 5, 8, 7, 13, 14, 17, 16},  // Element 3
        {9, 10, 13, 12, 18, 19, 22, 21},  // Element 4
        {10, 11, 14, 13, 19, 20, 23, 22}, // Element 5
        {12, 13, 16, 15, 21, 22, 25, 24}, // Element 6
        {13, 14, 17, 16, 22, 23, 26, 25}  // Element 7
    };

    // Set connectivity for all elements
    for (std::size_t e = 0; e < num_elems; ++e) {
        for (std::size_t n = 0; n < 8; ++n) {
            // Connectivity is flat 1D array: elem_id * nodes_per_elem + node_local_id
            block.connectivity[e * 8 + n] = hex_connectivity[e][n];
        }
        block.material_ids[e] = 1;  // All elements use material 1
        block.part_ids[e] = 1;      // All elements in part 1
    }

    // Print mesh information
    mesh.print_info();

    // Create a state for this mesh
    nxs::State state(mesh);

    // Set some initial conditions
    // Example: Give all nodes an initial velocity in Z direction
    auto& vel = state.velocity();
    for (std::size_t i = 0; i < num_nodes; ++i) {
        vel.at(i, 0) = 0.0;  // vx = 0
        vel.at(i, 1) = 0.0;  // vy = 0
        vel.at(i, 2) = 1.0;  // vz = 1.0 m/s
    }

    // Set nodal masses (uniform)
    auto& mass = state.mass();
    mass.fill(1.0);  // 1 kg per node

    // Print state information
    state.print_info();

    // Compute and print kinetic energy
    nxs::Real ke = state.compute_kinetic_energy();
    NXS_LOG_INFO("Initial kinetic energy: {:.6e} J", ke);

    // Demonstrate field operations
    NXS_LOG_INFO("Testing field operations...");

    // Create a custom scalar field
    auto temp_field = nxs::make_scalar_field("temperature", nxs::FieldLocation::Node, num_nodes);
    temp_field.fill(300.0);  // 300 K initial temperature

    // Compute statistics
    NXS_LOG_INFO("Temperature field:");
    NXS_LOG_INFO("  Min: {:.2f} K", temp_field.min());
    NXS_LOG_INFO("  Max: {:.2f} K", temp_field.max());
    NXS_LOG_INFO("  Mean: {:.2f} K", temp_field.mean());

    // Add to state
    state.add_field("temperature", std::move(temp_field));

    // Demonstrate memory arena
    NXS_LOG_INFO("Testing memory arena...");
    {
        NXS_SCOPED_TIMER("memory_arena_test");
        nxs::MemoryArena arena(1024);

        for (int i = 0; i < 100; ++i) {
            auto* ptr = arena.allocate(sizeof(double) * 10);
            (void)ptr;  // Suppress unused warning
        }

        NXS_LOG_INFO("Arena: allocated {} bytes, used {} bytes",
                    arena.total_allocated(), arena.used());
    }

    // Demonstrate aligned buffer
    NXS_LOG_INFO("Testing aligned buffer...");
    {
        nxs::AlignedBuffer<double> buffer(1000);
        buffer.fill(3.14159);

        NXS_LOG_INFO("Buffer size: {}", buffer.size());
        NXS_LOG_INFO("First element: {:.5f}", buffer[0]);
        NXS_LOG_INFO("Last element: {:.5f}", buffer[buffer.size() - 1]);
    }

    NXS_LOG_INFO("=================================================");
    NXS_LOG_INFO("Example completed successfully!");
    NXS_LOG_INFO("=================================================");

    return 0;
}
