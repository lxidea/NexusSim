/**
 * @file hex20_static_load_test.cpp
 * @brief Static equilibrium test: Apply load, solve K*u = f, check if u has correct sign
 */

#include <nexussim/discretization/hex20.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace nxs;
using namespace nxs::fem;

// Simple Gauss elimination with partial pivoting
void solve_linear_system(const std::vector<Real>& A, const std::vector<Real>& b, std::vector<Real>& x, int n) {
    std::vector<Real> Aug(n * (n+1));

    // Create augmented matrix [A|b]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Aug[i*(n+1) + j] = A[i*n + j];
        }
        Aug[i*(n+1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for (int k = 0; k < n; ++k) {
        // Find pivot
        int max_row = k;
        Real max_val = std::abs(Aug[k*(n+1) + k]);
        for (int i = k+1; i < n; ++i) {
            if (std::abs(Aug[i*(n+1) + k]) > max_val) {
                max_val = std::abs(Aug[i*(n+1) + k]);
                max_row = i;
            }
        }

        // Swap rows
        if (max_row != k) {
            for (int j = 0; j <= n; ++j) {
                std::swap(Aug[k*(n+1) + j], Aug[max_row*(n+1) + j]);
            }
        }

        // Eliminate
        for (int i = k+1; i < n; ++i) {
            Real factor = Aug[i*(n+1) + k] / Aug[k*(n+1) + k];
            for (int j = k; j <= n; ++j) {
                Aug[i*(n+1) + j] -= factor * Aug[k*(n+1) + j];
            }
        }
    }

    // Back substitution
    x.resize(n);
    for (int i = n-1; i >= 0; --i) {
        x[i] = Aug[i*(n+1) + n];
        for (int j = i+1; j < n; ++j) {
            x[i] -= Aug[i*(n+1) + j] * x[j];
        }
        x[i] /= Aug[i*(n+1) + i];
    }
}

int main() {
    std::cout << std::setprecision(10);

    // Unit cube
    const Real L = 1.0;
    Real coords[60];

    // Corner nodes
    coords[0*3+0] = 0.0; coords[0*3+1] = 0.0; coords[0*3+2] = 0.0;
    coords[1*3+0] = L;   coords[1*3+1] = 0.0; coords[1*3+2] = 0.0;
    coords[2*3+0] = L;   coords[2*3+1] = L;   coords[2*3+2] = 0.0;
    coords[3*3+0] = 0.0; coords[3*3+1] = L;   coords[3*3+2] = 0.0;
    coords[4*3+0] = 0.0; coords[4*3+1] = 0.0; coords[4*3+2] = L;
    coords[5*3+0] = L;   coords[5*3+1] = 0.0; coords[5*3+2] = L;
    coords[6*3+0] = L;   coords[6*3+1] = L;   coords[6*3+2] = L;
    coords[7*3+0] = 0.0; coords[7*3+1] = L;   coords[7*3+2] = L;

    // Mid-edge nodes
    coords[8*3+0] = 0.5*L; coords[8*3+1] = 0.0;   coords[8*3+2] = 0.0;
    coords[9*3+0] = L;     coords[9*3+1] = 0.5*L; coords[9*3+2] = 0.0;
    coords[10*3+0] = 0.5*L; coords[10*3+1] = L;   coords[10*3+2] = 0.0;
    coords[11*3+0] = 0.0;   coords[11*3+1] = 0.5*L; coords[11*3+2] = 0.0;
    coords[12*3+0] = 0.0;   coords[12*3+1] = 0.0;   coords[12*3+2] = 0.5*L;
    coords[13*3+0] = L;     coords[13*3+1] = 0.0;   coords[13*3+2] = 0.5*L;
    coords[14*3+0] = L;     coords[14*3+1] = L;     coords[14*3+2] = 0.5*L;
    coords[15*3+0] = 0.0;   coords[15*3+1] = L;     coords[15*3+2] = 0.5*L;
    coords[16*3+0] = 0.5*L; coords[16*3+1] = 0.0;   coords[16*3+2] = L;
    coords[17*3+0] = L;     coords[17*3+1] = 0.5*L; coords[17*3+2] = L;
    coords[18*3+0] = 0.5*L; coords[18*3+1] = L;     coords[18*3+2] = L;
    coords[19*3+0] = 0.0;   coords[19*3+1] = 0.5*L; coords[19*3+2] = L;

    const Real E = 210e9;
    const Real nu = 0.3;

    Hex20Element elem;

    // Compute stiffness
    Real K_full[60*60];
    elem.stiffness_matrix(coords, E, nu, K_full);

    std::cout << "===== Hex20 Static Load Test =====" << std::endl;
    std::cout << "Test: Fix bottom (z=0), apply +z load on top (z=" << L << ")" << std::endl;
    std::cout << "Expected: Top nodes move +z (tension)" << std::endl << std::endl;

    // Fix bottom nodes (0,1,2,3,8,9,10,11) - set rows/cols to identity
    std::vector<int> fixed_dofs;
    for (int node : {0, 1, 2, 3, 8, 9, 10, 11}) {
        fixed_dofs.push_back(node*3 + 0);  // x
        fixed_dofs.push_back(node*3 + 1);  // y
        fixed_dofs.push_back(node*3 + 2);  // z
    }

    // Apply load: 1000 N in +z on node 6 only
    std::vector<Real> f(60, 0.0);
    f[6*3 + 2] = 1000.0;  // +z force

    // Apply boundary conditions to K and f
    std::vector<Real> K_reduced;
    std::vector<Real> f_reduced;
    std::vector<int> free_dofs;

    for (int i = 0; i < 60; ++i) {
        bool is_fixed = false;
        for (int fd : fixed_dofs) {
            if (i == fd) {
                is_fixed = true;
                break;
            }
        }
        if (!is_fixed) {
            free_dofs.push_back(i);
        }
    }

    int n_free = free_dofs.size();
    K_reduced.resize(n_free * n_free);
    f_reduced.resize(n_free);

    for (int i = 0; i < n_free; ++i) {
        for (int j = 0; j < n_free; ++j) {
            K_reduced[i*n_free + j] = K_full[free_dofs[i]*60 + free_dofs[j]];
        }
        f_reduced[i] = f[free_dofs[i]];
    }

    // Solve K_reduced * u_reduced = f_reduced
    std::vector<Real> u_reduced;
    solve_linear_system(K_reduced, f_reduced, u_reduced, n_free);

    // Map back to full displacement vector
    std::vector<Real> u(60, 0.0);
    for (int i = 0; i < n_free; ++i) {
        u[free_dofs[i]] = u_reduced[i];
    }

    std::cout << "Results:" << std::endl;
    std::cout << "Node   uz (displacement)" << std::endl;
    std::cout << "----   ------------------" << std::endl;
    for (int node : {4, 5, 6, 7, 16, 17, 18, 19}) {
        std::cout << std::setw(4) << node << "   " << std::setw(18) << u[node*3 + 2] << std::endl;
    }

    Real uz_node6 = u[6*3 + 2];
    std::cout << std::endl;
    std::cout << "Applied force: fz[node 6] = +1000 N (+z direction)" << std::endl;
    std::cout << "Node 6 displacement: uz = " << uz_node6 << " m" << std::endl;

    if (uz_node6 > 0) {
        std::cout << "✓ PASS: Displacement is +z (correct direction)" << std::endl;
        return 0;
    } else {
        std::cout << "✗ FAIL: Displacement is -z (WRONG direction - sign error!)" << std::endl;
        return 1;
    }
}
