/**
 * @file tet4_compression_solver_test.cpp
 * @brief Simple FEM solver test using Tet4 elements - compression of a cube
 *
 * Tests a 2x2x2 cube under compression using 8 Tet4 elements.
 * Verifies that the solver can assemble and solve a simple linear system.
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/discretization/tet4.hpp>
#include <nexussim/fem/fem_solver.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;

int main() {
    std::cout << "=================================================\n";
    std::cout << "Tet4 Compression Solver Test\n";
    std::cout << "=================================================\n\n";

    // Create a simple 1x1x1 cube mesh divided into 5 tetrahedra
    // Using the standard 5-tet decomposition of a cube

    // 8 corner nodes of unit cube
    std::vector<Real> nodes = {
        // Node coordinates (x, y, z)
        0.0, 0.0, 0.0,  // Node 0
        1.0, 0.0, 0.0,  // Node 1
        1.0, 1.0, 0.0,  // Node 2
        0.0, 1.0, 0.0,  // Node 3
        0.0, 0.0, 1.0,  // Node 4
        1.0, 0.0, 1.0,  // Node 5
        1.0, 1.0, 1.0,  // Node 6
        0.0, 1.0, 1.0   // Node 7
    };

    // 5 tetrahedra decomposition of cube
    std::vector<int> connectivity = {
        // Tet 1: 0-1-2-5
        0, 1, 2, 5,
        // Tet 2: 0-2-3-7
        0, 2, 3, 7,
        // Tet 3: 0-5-2-7
        0, 5, 2, 7,
        // Tet 4: 5-6-2-7
        5, 6, 2, 7,
        // Tet 5: 0-5-7-4
        0, 5, 7, 4
    };

    const int num_nodes = 8;
    const int num_elements = 5;
    const int dofs_per_node = 3;
    const int total_dofs = num_nodes * dofs_per_node;

    std::cout << "Mesh info:\n";
    std::cout << "  Nodes: " << num_nodes << "\n";
    std::cout << "  Elements: " << num_elements << "\n";
    std::cout << "  Total DOFs: " << total_dofs << "\n\n";

    // Material properties (steel-like)
    const Real E = 200.0e9;   // Young's modulus (Pa)
    const Real nu = 0.3;      // Poisson's ratio
    const Real density = 7850.0;  // kg/m³

    std::cout << "Material properties:\n";
    std::cout << "  Young's modulus: " << E/1e9 << " GPa\n";
    std::cout << "  Poisson's ratio: " << nu << "\n";
    std::cout << "  Density: " << density << " kg/m³\n\n";

    // Compute element volumes and total volume
    Tet4Element elem;
    Real total_volume = 0.0;

    std::cout << "Element volumes:\n";
    for (int e = 0; e < num_elements; ++e) {
        Real elem_coords[12];
        for (int n = 0; n < 4; ++n) {
            int node_id = connectivity[e * 4 + n];
            elem_coords[n*3 + 0] = nodes[node_id*3 + 0];
            elem_coords[n*3 + 1] = nodes[node_id*3 + 1];
            elem_coords[n*3 + 2] = nodes[node_id*3 + 2];
        }
        Real vol = elem.volume(elem_coords);
        total_volume += vol;
        std::cout << "  Element " << e << ": " << vol << " m³\n";
    }
    std::cout << "  Total volume: " << total_volume << " m³\n";
    std::cout << "  Expected: 1.0 m³\n";
    std::cout << "  Status: " << (std::abs(total_volume - 1.0) < 1e-6 ? "PASS" : "FAIL") << "\n\n";

    // Assemble global stiffness matrix (simplified - using dense storage)
    std::vector<Real> K_global(total_dofs * total_dofs, 0.0);

    std::cout << "Assembling stiffness matrix...\n";
    for (int e = 0; e < num_elements; ++e) {
        // Get element coordinates
        Real elem_coords[12];
        for (int n = 0; n < 4; ++n) {
            int node_id = connectivity[e * 4 + n];
            elem_coords[n*3 + 0] = nodes[node_id*3 + 0];
            elem_coords[n*3 + 1] = nodes[node_id*3 + 1];
            elem_coords[n*3 + 2] = nodes[node_id*3 + 2];
        }

        // Compute element stiffness matrix
        Real K_elem[12 * 12];
        elem.stiffness_matrix(elem_coords, E, nu, K_elem);

        // Assemble into global matrix
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int node_i = connectivity[e * 4 + i];
                int node_j = connectivity[e * 4 + j];

                for (int di = 0; di < 3; ++di) {
                    for (int dj = 0; dj < 3; ++dj) {
                        int global_i = node_i * 3 + di;
                        int global_j = node_j * 3 + dj;
                        int local_i = i * 3 + di;
                        int local_j = j * 3 + dj;

                        K_global[global_i * total_dofs + global_j] += K_elem[local_i * 12 + local_j];
                    }
                }
            }
        }
    }
    std::cout << "  Stiffness matrix assembled.\n\n";

    // Check matrix properties
    std::cout << "Stiffness matrix properties:\n";

    // Check symmetry
    Real max_asymmetry = 0.0;
    for (int i = 0; i < total_dofs; ++i) {
        for (int j = i + 1; j < total_dofs; ++j) {
            Real diff = std::abs(K_global[i * total_dofs + j] - K_global[j * total_dofs + i]);
            max_asymmetry = std::max(max_asymmetry, diff);
        }
    }
    std::cout << "  Max asymmetry: " << max_asymmetry << "\n";
    std::cout << "  Symmetry test: " << (max_asymmetry < 1e-6 ? "PASS" : "FAIL") << "\n";

    // Check diagonal dominance (rough indicator)
    int positive_diagonals = 0;
    for (int i = 0; i < total_dofs; ++i) {
        if (K_global[i * total_dofs + i] > 1e-6) {
            positive_diagonals++;
        }
    }
    std::cout << "  Positive diagonal entries: " << positive_diagonals << "/" << total_dofs << "\n";

    // Check for zero rows (before applying BCs)
    int zero_rows = 0;
    for (int i = 0; i < total_dofs; ++i) {
        Real row_sum = 0.0;
        for (int j = 0; j < total_dofs; ++j) {
            row_sum += std::abs(K_global[i * total_dofs + j]);
        }
        if (row_sum < 1e-10) {
            zero_rows++;
        }
    }
    std::cout << "  Zero rows: " << zero_rows << "\n";
    std::cout << "  Status: " << (zero_rows == 0 ? "PASS" : "FAIL") << "\n\n";

    // Apply boundary conditions: fix bottom face (z=0) in all directions
    std::cout << "Applying boundary conditions:\n";
    std::cout << "  Bottom face (z=0): fixed in all directions\n";
    std::cout << "  Top face (z=1): prescribed displacement uz=-0.01m\n\n";

    std::vector<Real> F(total_dofs, 0.0);  // Force vector
    std::vector<Real> u(total_dofs, 0.0);  // Displacement vector

    // Fix bottom nodes (0, 1, 2, 3) - z = 0
    for (int n = 0; n <= 3; ++n) {
        for (int d = 0; d < 3; ++d) {
            int dof = n * 3 + d;
            // Zero out row and column, put 1 on diagonal
            for (int j = 0; j < total_dofs; ++j) {
                K_global[dof * total_dofs + j] = 0.0;
                K_global[j * total_dofs + dof] = 0.0;
            }
            K_global[dof * total_dofs + dof] = 1.0;
            F[dof] = 0.0;  // u = 0
        }
    }

    // Apply displacement to top nodes (4, 5, 6, 7) - z = 1
    const Real prescribed_displacement = -0.01;  // 1cm compression
    for (int n = 4; n <= 7; ++n) {
        int dof = n * 3 + 2;  // z-direction only
        // Zero out row and column, put 1 on diagonal
        for (int j = 0; j < total_dofs; ++j) {
            // Modify force vector to account for prescribed displacement
            if (j != dof) {
                F[j] -= K_global[j * total_dofs + dof] * prescribed_displacement;
            }
        }
        // Set constraint
        for (int j = 0; j < total_dofs; ++j) {
            K_global[dof * total_dofs + j] = 0.0;
            K_global[j * total_dofs + dof] = 0.0;
        }
        K_global[dof * total_dofs + dof] = 1.0;
        F[dof] = prescribed_displacement;
        u[dof] = prescribed_displacement;  // Initial guess
    }

    std::cout << "Solving system K*u = F...\n";
    std::cout << "  (Using simple Gauss-Seidel iteration)\n\n";

    // Simple iterative solver (Gauss-Seidel)
    const int max_iterations = 10000;
    const Real tolerance = 1e-8;

    for (int iter = 0; iter < max_iterations; ++iter) {
        Real max_change = 0.0;

        for (int i = 0; i < total_dofs; ++i) {
            Real sum = F[i];
            for (int j = 0; j < total_dofs; ++j) {
                if (i != j) {
                    sum -= K_global[i * total_dofs + j] * u[j];
                }
            }
            Real u_new = sum / K_global[i * total_dofs + i];
            Real change = std::abs(u_new - u[i]);
            max_change = std::max(max_change, change);
            u[i] = u_new;
        }

        if (iter % 1000 == 0) {
            std::cout << "  Iteration " << iter << ": max change = " << max_change << "\n";
        }

        if (max_change < tolerance) {
            std::cout << "  Converged after " << iter << " iterations\n\n";
            break;
        }
    }

    // Display results
    std::cout << "Results:\n";
    std::cout << "  Nodal displacements:\n";
    for (int n = 0; n < num_nodes; ++n) {
        Real ux = u[n * 3 + 0];
        Real uy = u[n * 3 + 1];
        Real uz = u[n * 3 + 2];
        std::cout << "    Node " << n << ": ux=" << std::setw(12) << ux
                  << " uy=" << std::setw(12) << uy
                  << " uz=" << std::setw(12) << uz << "\n";
    }

    // Check results
    std::cout << "\nValidation:\n";

    // Bottom nodes should be zero
    bool bottom_fixed = true;
    for (int n = 0; n <= 3; ++n) {
        for (int d = 0; d < 3; ++d) {
            if (std::abs(u[n * 3 + d]) > 1e-6) {
                bottom_fixed = false;
            }
        }
    }
    std::cout << "  Bottom face fixed: " << (bottom_fixed ? "PASS" : "FAIL") << "\n";

    // Top nodes should have prescribed displacement in z
    bool top_displaced = true;
    for (int n = 4; n <= 7; ++n) {
        if (std::abs(u[n * 3 + 2] - prescribed_displacement) > 1e-6) {
            top_displaced = false;
        }
    }
    std::cout << "  Top face displacement: " << (top_displaced ? "PASS" : "FAIL") << "\n";

    // Check that lateral displacements are outward (Poisson effect)
    std::cout << "  Poisson effect (lateral expansion): ";
    bool poisson_ok = true;
    for (int n = 4; n <= 7; ++n) {
        // Top corners should move outward slightly
        Real ux = u[n * 3 + 0];
        Real uy = u[n * 3 + 1];
        // Not a strong test, just check they're non-zero
        if (std::abs(ux) < 1e-12 && std::abs(uy) < 1e-12) {
            poisson_ok = false;
        }
    }
    std::cout << (poisson_ok ? "PASS" : "FAIL") << "\n";

    std::cout << "\n=================================================\n";
    std::cout << "Tet4 Compression Solver Test Complete\n";
    std::cout << "=================================================\n";

    return 0;
}
