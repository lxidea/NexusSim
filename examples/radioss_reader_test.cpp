/**
 * @file radioss_reader_test.cpp
 * @brief Test OpenRadioss input deck reader
 */

#include <nexussim/io/radioss_reader.hpp>
#include <iostream>
#include <string>

using namespace nxs;
using namespace nxs::io;

int main(int argc, char* argv[]) {
    std::cout << "=================================================\n";
    std::cout << "OpenRadioss Reader Test\n";
    std::cout << "=================================================\n\n";

    // Default test file
    std::string filename = "/mnt/d/_working_/FEM-PD/OpenRadioss/qa-tests/miniqa/ACCELEROMETRES/data/ACCELERO_0000.rad";

    if (argc > 1) {
        filename = argv[1];
    }

    std::cout << "Reading: " << filename << "\n\n";

    RadiossReader reader;

    if (!reader.read(filename)) {
        std::cerr << "ERROR: Failed to read file\n";
        return 1;
    }

    // Print summary
    reader.print_summary();

    // Create mesh
    std::cout << "\nCreating NexusSim mesh...\n";
    auto mesh = reader.create_mesh();

    if (mesh) {
        std::cout << "Mesh created successfully!\n";
        std::cout << "  Nodes: " << mesh->num_nodes() << "\n";
        std::cout << "  Element blocks: " << mesh->num_element_blocks() << "\n";

        for (size_t i = 0; i < mesh->num_element_blocks(); ++i) {
            const auto& block = mesh->element_block(i);
            std::cout << "  Block " << i << ": " << block.name
                      << " (" << block.num_elements() << " elements)" << std::endl;
        }
    } else {
        std::cerr << "ERROR: Failed to create mesh\n";
        return 1;
    }

    std::cout << "\n=================================================\n";
    std::cout << "Test Complete - SUCCESS\n";
    std::cout << "=================================================\n";

    return 0;
}
