#include <nexussim/discretization/wedge6.hpp>
#include <iostream>
#include <iomanip>

using namespace nxs::fem;

int main() {
    Wedge6Element elem;

    // Test shape functions at each node
    std::cout << "Testing shape functions at natural coordinate corners:\n\n";

    // Node locations in natural coords according to header
    double natural_coords[6][3] = {
        {1.0, 0.0, -1.0},  // Node 0
        {0.0, 1.0, -1.0},  // Node 1
        {0.0, 0.0, -1.0},  // Node 2
        {1.0, 0.0, +1.0},  // Node 3
        {0.0, 1.0, +1.0},  // Node 4
        {0.0, 0.0, +1.0}   // Node 5
    };

    for (int node = 0; node < 6; ++node) {
        double N[6];
        elem.shape_functions(natural_coords[node], N);

        std::cout << "At node " << node << " (ξ=" << natural_coords[node][0]
                  << ", η=" << natural_coords[node][1]
                  << ", ζ=" << natural_coords[node][2] << "):\n";
        for (int i = 0; i < 6; ++i) {
            std::cout << "  N[" << i << "] = " << std::setprecision(10) << N[i] << "\n";
        }
        std::cout << "\n";
    }

    // Now test mass matrix with unit wedge
    std::cout << "\n=== Mass Matrix Test ===\n";
    double coords[6 * 3] = {
        1.0, 0.0, 0.0,  // Node 0
        0.0, 1.0, 0.0,  // Node 1
        0.0, 0.0, 0.0,  // Node 2
        1.0, 0.0, 1.0,  // Node 3
        0.0, 1.0, 1.0,  // Node 4
        0.0, 0.0, 1.0   // Node 5
    };

    double volume = elem.volume(coords);
    std::cout << "Volume: " << volume << " m³\n";

    double density = 1000.0;
    double M[18 * 18];
    elem.mass_matrix(coords, density, M);

    double total_mass = 0.0;
    for (int i = 0; i < 18 * 18; ++i) {
        total_mass += M[i];
    }

    std::cout << "Total mass (sum of all entries): " << total_mass << " kg\n";
    std::cout << "Expected mass (ρ × V): " << density * volume << " kg\n";
    std::cout << "Ratio: " << total_mass / (density * volume) << "\n";

    // Print first few mass matrix entries
    std::cout << "\nFirst 6x6 block of mass matrix:\n";
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            std::cout << std::setw(12) << std::setprecision(6) << M[i * 18 + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
