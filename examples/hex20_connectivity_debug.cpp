/**
 * @file hex20_connectivity_debug.cpp
 * @brief Debug Hex20 connectivity for multi-element meshes
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/discretization/hex20.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace nxs;
using namespace nxs::fem;

void print_element_info(int elem_id, const std::vector<Real>& all_nodes, const std::vector<int>& conn) {
    std::cout << "\n--- Element " << elem_id << " ---\n";

    // Get element coordinates
    Real coords[60];
    for (int n = 0; n < 20; ++n) {
        int node_id = conn[elem_id * 20 + n];
        coords[n*3 + 0] = all_nodes[node_id * 3 + 0];
        coords[n*3 + 1] = all_nodes[node_id * 3 + 1];
        coords[n*3 + 2] = all_nodes[node_id * 3 + 2];
    }

    // Print corner nodes
    std::cout << "Corner nodes:\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << "  Node " << i << " (ID=" << conn[elem_id*20 + i] << "): ("
                  << coords[i*3+0] << ", " << coords[i*3+1] << ", " << coords[i*3+2] << ")\n";
    }

    // Check Jacobian at center
    Hex20Element elem;
    Real xi_center[3] = {0.0, 0.0, 0.0};
    Real J[9];
    Real det_J = elem.jacobian(xi_center, coords, J);

    std::cout << "Jacobian at center: " << det_J << " " << (det_J > 0 ? "[OK]" : "[INVERTED!]") << "\n";

    // Check volume
    Real vol = elem.volume(coords);
    std::cout << "Volume: " << vol << "\n";

    // Check at corner natural coordinates
    Real xi_corners[8][3] = {
        {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
        {-1, -1,  1}, {1, -1,  1}, {1, 1,  1}, {-1, 1,  1}
    };

    std::cout << "Jacobian at corners:\n";
    for (int i = 0; i < 8; ++i) {
        Real det = elem.jacobian(xi_corners[i], coords, J);
        std::cout << "  Corner " << i << " (" << xi_corners[i][0] << "," << xi_corners[i][1] << "," << xi_corners[i][2]
                  << "): det=" << det << " " << (det > 0 ? "[OK]" : "[BAD]") << "\n";
        if (det < 0) {
            std::cout << "    Jacobian matrix:\n";
            std::cout << "      [" << J[0] << ", " << J[1] << ", " << J[2] << "]\n";
            std::cout << "      [" << J[3] << ", " << J[4] << ", " << J[5] << "]\n";
            std::cout << "      [" << J[6] << ", " << J[7] << ", " << J[8] << "]\n";
        }
    }
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "Hex20 Connectivity Debug\n";
    std::cout << "=================================================\n\n";

    // Test Case 1: Single 1x1x1 element (should work)
    std::cout << "\n========== TEST 1: Single Element ==========\n";
    {
        std::vector<Real> nodes = {
            // 8 corner nodes
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1
            1.0, 1.0, 0.0,  // 2
            0.0, 1.0, 0.0,  // 3
            0.0, 0.0, 1.0,  // 4
            1.0, 0.0, 1.0,  // 5
            1.0, 1.0, 1.0,  // 6
            0.0, 1.0, 1.0,  // 7
            // 12 mid-edge nodes
            0.5, 0.0, 0.0,  // 8:  edge 0-1
            1.0, 0.5, 0.0,  // 9:  edge 1-2
            0.5, 1.0, 0.0,  // 10: edge 2-3
            0.0, 0.5, 0.0,  // 11: edge 3-0
            0.0, 0.0, 0.5,  // 12: edge 0-4
            1.0, 0.0, 0.5,  // 13: edge 1-5
            1.0, 1.0, 0.5,  // 14: edge 2-6
            0.0, 1.0, 0.5,  // 15: edge 3-7
            0.5, 0.0, 1.0,  // 16: edge 4-5
            1.0, 0.5, 1.0,  // 17: edge 5-6
            0.5, 1.0, 1.0,  // 18: edge 6-7
            0.0, 0.5, 1.0   // 19: edge 7-4
        };

        std::vector<int> conn = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
        print_element_info(0, nodes, conn);
    }

    // Test Case 2: 2x1x1 mesh (2 elements along X)
    std::cout << "\n========== TEST 2: 2x1x1 Mesh ==========\n";
    {
        // For 2 elements in X: need 5 X-positions (0, 0.5, 1, 1.5, 2)
        // For 1 element in Y: need 3 Y-positions (0, 0.5, 1)
        // For 1 element in Z: need 3 Z-positions (0, 0.5, 1)
        // Total nodes: 5 x 3 x 3 = 45 nodes

        const int nx = 2, ny = 1, nz = 1;
        const int n_nodes_x = 2 * nx + 1;  // 5
        const int n_nodes_y = 2 * ny + 1;  // 3
        const int n_nodes_z = 2 * nz + 1;  // 3

        std::vector<Real> nodes;

        // Generate nodes: loop order i,j,k (X major, Y middle, Z minor/fastest)
        for (int i = 0; i < n_nodes_x; ++i) {
            Real x = 0.5 * i;  // Each element is 1 unit, half-spacing for mid-nodes
            for (int j = 0; j < n_nodes_y; ++j) {
                Real y = 0.5 * j;
                for (int k = 0; k < n_nodes_z; ++k) {
                    Real z = 0.5 * k;
                    nodes.push_back(x);
                    nodes.push_back(y);
                    nodes.push_back(z);
                }
            }
        }

        std::cout << "Generated " << nodes.size()/3 << " nodes\n";

        // Build connectivity for 2 elements
        std::vector<int> conn;

        for (int ei = 0; ei < nx; ++ei) {
            for (int ej = 0; ej < ny; ++ej) {
                for (int ek = 0; ek < nz; ++ek) {
                    // Node index function
                    auto node_idx = [&](int di, int dj, int dk) -> int {
                        return ((2*ei + di) * n_nodes_y * n_nodes_z) +
                               ((2*ej + dj) * n_nodes_z) +
                               (2*ek + dk);
                    };

                    // Corner nodes (standard Hex8 ordering)
                    conn.push_back(node_idx(0, 0, 0));  // 0
                    conn.push_back(node_idx(2, 0, 0));  // 1
                    conn.push_back(node_idx(2, 2, 0));  // 2
                    conn.push_back(node_idx(0, 2, 0));  // 3
                    conn.push_back(node_idx(0, 0, 2));  // 4
                    conn.push_back(node_idx(2, 0, 2));  // 5
                    conn.push_back(node_idx(2, 2, 2));  // 6
                    conn.push_back(node_idx(0, 2, 2));  // 7

                    // Bottom face mid-edge nodes
                    conn.push_back(node_idx(1, 0, 0));  // 8:  edge 0-1
                    conn.push_back(node_idx(2, 1, 0));  // 9:  edge 1-2
                    conn.push_back(node_idx(1, 2, 0));  // 10: edge 2-3
                    conn.push_back(node_idx(0, 1, 0));  // 11: edge 3-0

                    // Vertical mid-edge nodes
                    conn.push_back(node_idx(0, 0, 1));  // 12: edge 0-4
                    conn.push_back(node_idx(2, 0, 1));  // 13: edge 1-5
                    conn.push_back(node_idx(2, 2, 1));  // 14: edge 2-6
                    conn.push_back(node_idx(0, 2, 1));  // 15: edge 3-7

                    // Top face mid-edge nodes
                    conn.push_back(node_idx(1, 0, 2));  // 16: edge 4-5
                    conn.push_back(node_idx(2, 1, 2));  // 17: edge 5-6
                    conn.push_back(node_idx(1, 2, 2));  // 18: edge 6-7
                    conn.push_back(node_idx(0, 1, 2));  // 19: edge 7-4
                }
            }
        }

        // Print both elements
        for (int e = 0; e < 2; ++e) {
            print_element_info(e, nodes, conn);
        }
    }

    std::cout << "\n=================================================\n";
    std::cout << "Debug Complete\n";
    std::cout << "=================================================\n";

    return 0;
}
