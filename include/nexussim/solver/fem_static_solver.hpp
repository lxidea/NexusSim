#pragma once

/**
 * @file fem_static_solver.hpp
 * @brief FEM Static Analysis Solver
 *
 * Integrates the implicit solver framework with the FEM element library
 * for static structural analysis.
 *
 * Features:
 * - Global stiffness matrix assembly from elements
 * - Boundary condition application (Dirichlet)
 * - Linear and nonlinear static analysis
 * - Load stepping for large deformations
 */

#include <nexussim/core/core.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/solver/implicit_solver.hpp>
#include <nexussim/discretization/hex8.hpp>
#include <nexussim/discretization/hex20.hpp>
#include <nexussim/discretization/tet4.hpp>
#include <nexussim/discretization/tet10.hpp>
#include <vector>
#include <set>
#include <unordered_map>
#include <functional>
#include <memory>
#include <cmath>

namespace nxs {
namespace solver {

// ============================================================================
// Boundary Conditions
// ============================================================================

/**
 * @brief Dirichlet (displacement) boundary condition
 */
struct DirichletBC {
    Index node_id;          ///< Node index
    int dof;                ///< DOF direction (0=x, 1=y, 2=z)
    Real value;             ///< Prescribed displacement value

    DirichletBC(Index node, int direction, Real val = 0.0)
        : node_id(node), dof(direction), value(val) {}
};

/**
 * @brief Neumann (force) boundary condition
 */
struct NeumannBC {
    Index node_id;          ///< Node index
    int dof;                ///< DOF direction (0=x, 1=y, 2=z)
    Real value;             ///< Applied force value

    NeumannBC(Index node, int direction, Real force)
        : node_id(node), dof(direction), value(force) {}
};

// ============================================================================
// Material Properties (Simple Isotropic Elastic)
// ============================================================================

struct ElasticMaterial {
    Real E = 2.0e11;        ///< Young's modulus (Pa) - default steel
    Real nu = 0.3;          ///< Poisson's ratio
    Real rho = 7800.0;      ///< Density (kg/m³)

    /**
     * @brief Compute Lamé parameters
     */
    Real lambda() const { return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)); }
    Real mu() const { return E / (2.0 * (1.0 + nu)); }

    /**
     * @brief Compute bulk modulus
     */
    Real bulk() const { return E / (3.0 * (1.0 - 2.0 * nu)); }
};

// ============================================================================
// FEM Static Solver
// ============================================================================

/**
 * @brief Static FEM solver for structural analysis
 */
class FEMStaticSolver {
public:
    struct Result {
        bool converged = false;
        int iterations = 0;
        Real residual = 0.0;
        std::vector<Real> displacement;
        std::vector<Real> reaction_forces;
    };

    FEMStaticSolver() = default;

    /**
     * @brief Initialize solver with mesh
     */
    void set_mesh(const Mesh& mesh) {
        mesh_ = &mesh;
        ndof_ = mesh.num_nodes() * 3;  // 3 DOFs per node

        // Build sparsity pattern from mesh connectivity
        build_sparsity_pattern();
    }

    /**
     * @brief Set material properties
     */
    void set_material(const ElasticMaterial& mat) {
        material_ = mat;
    }

    /**
     * @brief Add Dirichlet (displacement) boundary condition
     */
    void add_dirichlet_bc(Index node, int dof, Real value = 0.0) {
        dirichlet_bcs_.emplace_back(node, dof, value);
        constrained_dofs_.insert(node * 3 + dof);
    }

    /**
     * @brief Fix all DOFs at a node
     */
    void fix_node(Index node) {
        add_dirichlet_bc(node, 0, 0.0);
        add_dirichlet_bc(node, 1, 0.0);
        add_dirichlet_bc(node, 2, 0.0);
    }

    /**
     * @brief Add force at a node
     */
    void add_force(Index node, int dof, Real force) {
        neumann_bcs_.emplace_back(node, dof, force);
    }

    /**
     * @brief Add force vector at a node
     */
    void add_force_vector(Index node, Real fx, Real fy, Real fz) {
        if (std::abs(fx) > 1e-20) add_force(node, 0, fx);
        if (std::abs(fy) > 1e-20) add_force(node, 1, fy);
        if (std::abs(fz) > 1e-20) add_force(node, 2, fz);
    }

    /**
     * @brief Clear all boundary conditions
     */
    void clear_boundary_conditions() {
        dirichlet_bcs_.clear();
        neumann_bcs_.clear();
        constrained_dofs_.clear();
    }

    /**
     * @brief Get count of elements skipped due to NaN in last assembly
     */
    size_t nan_element_count() const { return nan_element_count_; }

    /**
     * @brief Get count of zero diagonal DOFs detected in last solve
     */
    size_t zero_diagonal_count() const { return zero_diagonal_count_; }

    /**
     * @brief Solve linear static problem: K*u = F
     */
    Result solve_linear() {
        Result result;

        if (!mesh_) {
            NXS_LOG_ERROR("FEMStaticSolver: No mesh set");
            return result;
        }

        // Assemble global stiffness
        nan_element_count_ = 0;
        assemble_stiffness();

        // Guard: scan free DOFs for zero diagonals
        zero_diagonal_count_ = 0;
        for (size_t i = 0; i < ndof_; ++i) {
            if (constrained_dofs_.count(i) == 0) {
                if (std::abs(K_global_.get(i, i)) < 1e-30) {
                    zero_diagonal_count_++;
                }
            }
        }

        // Build external force vector
        std::vector<Real> F_ext(ndof_, 0.0);
        for (const auto& bc : neumann_bcs_) {
            size_t dof_idx = bc.node_id * 3 + bc.dof;
            F_ext[dof_idx] += bc.value;
        }

        // Apply Dirichlet BCs
        apply_dirichlet_bcs(F_ext);

        // Solve
        std::vector<Real> u(ndof_, 0.0);

        CGSolver cg;
        cg.set_tolerance(1e-8);  // Relaxed tolerance for better convergence
        cg.set_max_iterations(10000);
        cg.set_preconditioner(true);

        auto lin_result = cg.solve(K_global_, F_ext, u);

        result.converged = lin_result.converged;
        result.iterations = lin_result.iterations;
        result.residual = lin_result.residual;

        // Guard: scan solution for NaN/Inf
        for (size_t i = 0; i < u.size(); ++i) {
            if (std::isnan(u[i]) || std::isinf(u[i])) {
                result.converged = false;
                break;
            }
        }

        // Apply prescribed displacements
        for (const auto& bc : dirichlet_bcs_) {
            size_t dof_idx = bc.node_id * 3 + bc.dof;
            u[dof_idx] = bc.value;
        }

        result.displacement = std::move(u);

        // Compute reaction forces if needed
        if (result.converged) {
            compute_reactions(result);
        }

        return result;
    }

    /**
     * @brief Get displacement at a node
     */
    static std::array<Real, 3> get_node_displacement(const Result& result, Index node) {
        return {
            result.displacement[node * 3 + 0],
            result.displacement[node * 3 + 1],
            result.displacement[node * 3 + 2]
        };
    }

    /**
     * @brief Compute maximum displacement magnitude
     */
    static Real max_displacement(const Result& result) {
        Real max_disp = 0.0;
        size_t num_nodes = result.displacement.size() / 3;
        for (size_t i = 0; i < num_nodes; ++i) {
            Real ux = result.displacement[i * 3 + 0];
            Real uy = result.displacement[i * 3 + 1];
            Real uz = result.displacement[i * 3 + 2];
            Real mag = std::sqrt(ux * ux + uy * uy + uz * uz);
            max_disp = std::max(max_disp, mag);
        }
        return max_disp;
    }

    /**
     * @brief Get the global stiffness matrix (for inspection/debugging)
     */
    const SparseMatrix& stiffness_matrix() const { return K_global_; }

    /**
     * @brief Print solver statistics
     */
    void print_info() const {
        std::cout << "=== FEM Static Solver ===\n";
        std::cout << "Nodes: " << (mesh_ ? mesh_->num_nodes() : 0) << "\n";
        std::cout << "Elements: " << (mesh_ ? mesh_->num_elements() : 0) << "\n";
        std::cout << "Total DOFs: " << ndof_ << "\n";
        std::cout << "Constrained DOFs: " << constrained_dofs_.size() << "\n";
        std::cout << "Free DOFs: " << (ndof_ - constrained_dofs_.size()) << "\n";
        std::cout << "Stiffness matrix NNZ: " << K_global_.nnz() << "\n";
        std::cout << "=========================\n";
    }

private:
    /**
     * @brief Build sparsity pattern from mesh connectivity
     */
    void build_sparsity_pattern() {
        if (!mesh_) return;

        // For each node, track which nodes it's connected to
        std::vector<std::set<size_t>> node_adjacency(mesh_->num_nodes());

        // Process each element block
        for (const auto& block : mesh_->element_blocks()) {
            for (size_t e = 0; e < block.num_elements(); ++e) {
                auto elem_nodes = block.element_nodes(e);

                // All nodes in element are connected
                for (size_t i = 0; i < elem_nodes.size(); ++i) {
                    for (size_t j = 0; j < elem_nodes.size(); ++j) {
                        node_adjacency[elem_nodes[i]].insert(elem_nodes[j]);
                    }
                }
            }
        }

        // Build DOF-level sparsity pattern
        std::vector<std::vector<size_t>> dof_pattern(ndof_);

        for (size_t node_i = 0; node_i < mesh_->num_nodes(); ++node_i) {
            for (size_t node_j : node_adjacency[node_i]) {
                // Each node pair creates a 3x3 block
                for (int di = 0; di < 3; ++di) {
                    for (int dj = 0; dj < 3; ++dj) {
                        dof_pattern[node_i * 3 + di].push_back(node_j * 3 + dj);
                    }
                }
            }
        }

        // Create sparse matrix with this pattern
        K_global_.create_pattern(ndof_, ndof_, dof_pattern);
    }

    /**
     * @brief Assemble global stiffness matrix from elements
     */
    void assemble_stiffness() {
        K_global_.zero();

        // Create element objects
        fem::Hex8Element hex8;
        fem::Hex20Element hex20;
        fem::Tet4Element tet4;
        fem::Tet10Element tet10;

        // Process each element block
        for (const auto& block : mesh_->element_blocks()) {
            size_t nodes_per_elem = block.num_nodes_per_elem;
            size_t elem_ndof = nodes_per_elem * 3;

            // Element stiffness storage
            std::vector<Real> ke(elem_ndof * elem_ndof);
            std::vector<Real> elem_coords(nodes_per_elem * 3);
            std::vector<Index> dof_map(elem_ndof);

            for (size_t e = 0; e < block.num_elements(); ++e) {
                auto elem_nodes = block.element_nodes(e);

                // Gather element coordinates
                for (size_t i = 0; i < nodes_per_elem; ++i) {
                    auto coords = mesh_->get_node_coordinates(elem_nodes[i]);
                    elem_coords[i * 3 + 0] = coords[0];
                    elem_coords[i * 3 + 1] = coords[1];
                    elem_coords[i * 3 + 2] = coords[2];
                }

                // Build DOF map
                for (size_t i = 0; i < nodes_per_elem; ++i) {
                    dof_map[i * 3 + 0] = elem_nodes[i] * 3 + 0;
                    dof_map[i * 3 + 1] = elem_nodes[i] * 3 + 1;
                    dof_map[i * 3 + 2] = elem_nodes[i] * 3 + 2;
                }

                // Compute element stiffness based on element type
                if (nodes_per_elem == 8) {
                    hex8.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                } else if (nodes_per_elem == 20) {
                    hex20.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                } else if (nodes_per_elem == 10) {
                    tet10.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                } else if (nodes_per_elem == 4) {
                    tet4.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                } else {
                    // For other element types, compute manually using B-matrix
                    compute_element_stiffness_generic(block, e, elem_coords, ke);
                }

                // Guard: scan element stiffness for NaN
                bool has_nan = false;
                for (size_t idx = 0; idx < ke.size(); ++idx) {
                    if (std::isnan(ke[idx]) || std::isinf(ke[idx])) {
                        has_nan = true;
                        break;
                    }
                }
                if (has_nan) {
                    nan_element_count_++;
                    continue;  // Skip this element
                }

                // Assemble into global
                K_global_.add_element_matrix(dof_map, ke);
            }
        }
    }

    /**
     * @brief Generic element stiffness computation using B-matrix integration
     */
    void compute_element_stiffness_generic(const Mesh::ElementBlock& block,
                                           size_t elem_id,
                                           const std::vector<Real>& coords,
                                           std::vector<Real>& ke) {
        // Simple implementation for Tet4
        size_t n = block.num_nodes_per_elem;

        std::fill(ke.begin(), ke.end(), 0.0);

        if (n == 4) {
            // Tet4: Single integration point
            compute_tet4_stiffness(coords, ke);
        } else if (n == 10) {
            // Tet10: 4-point Gauss quadrature
            fem::Tet10Element tet10;
            tet10.stiffness_matrix(coords.data(), material_.E, material_.nu, ke.data());
        }
    }

    /**
     * @brief Compute Tet4 element stiffness
     */
    void compute_tet4_stiffness(const std::vector<Real>& coords, std::vector<Real>& ke) {
        // Tet4 has constant strain, compute B-matrix once
        // B = (1/6V) * [b1 0  0  b2 0  0  b3 0  0  b4 0  0 ]
        //              [0  c1 0  0  c2 0  0  c3 0  0  c4 0 ]
        //              [0  0  d1 0  0  d2 0  0  d3 0  0  d4]
        //              [c1 b1 0  c2 b2 0  c3 b3 0  c4 b4 0 ]
        //              [0  d1 c1 0  d2 c2 0  d3 c3 0  d4 c4]
        //              [d1 0  b1 d2 0  b2 d3 0  b3 d4 0  b4]

        const Real* x = coords.data();

        // Compute volume and shape function derivatives
        Real x1 = x[0], y1 = x[1], z1 = x[2];
        Real x2 = x[3], y2 = x[4], z2 = x[5];
        Real x3 = x[6], y3 = x[7], z3 = x[8];
        Real x4 = x[9], y4 = x[10], z4 = x[11];

        // Volume = (1/6) * det([x2-x1 x3-x1 x4-x1; y2-y1 y3-y1 y4-y1; z2-z1 z3-z1 z4-z1])
        Real a1 = x2 - x1, a2 = x3 - x1, a3 = x4 - x1;
        Real b1 = y2 - y1, b2 = y3 - y1, b3 = y4 - y1;
        Real c1 = z2 - z1, c2 = z3 - z1, c3 = z4 - z1;

        Real V6 = a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1);
        Real V = V6 / 6.0;

        if (std::abs(V) < 1e-20) {
            return;  // Degenerate element
        }

        // Shape function derivatives (constant for Tet4)
        // dN/dx = (1/6V) * [b1, b2, b3, b4]
        // dN/dy = (1/6V) * [c1, c2, c3, c4]
        // dN/dz = (1/6V) * [d1, d2, d3, d4]

        Real inv_6V = 1.0 / V6;

        // Compute cofactors for shape function derivatives
        Real dN[4][3];  // dN[node][x,y,z]

        // dN1/dx,dy,dz
        dN[0][0] = inv_6V * ((y3 - y2) * (z4 - z2) - (y4 - y2) * (z3 - z2));
        dN[0][1] = inv_6V * ((z3 - z2) * (x4 - x2) - (z4 - z2) * (x3 - x2));
        dN[0][2] = inv_6V * ((x3 - x2) * (y4 - y2) - (x4 - x2) * (y3 - y2));

        // dN2/dx,dy,dz
        dN[1][0] = inv_6V * ((y4 - y3) * (z1 - z3) - (y1 - y3) * (z4 - z3));
        dN[1][1] = inv_6V * ((z4 - z3) * (x1 - x3) - (z1 - z3) * (x4 - x3));
        dN[1][2] = inv_6V * ((x4 - x3) * (y1 - y3) - (x1 - x3) * (y4 - y3));

        // dN3/dx,dy,dz
        dN[2][0] = inv_6V * ((y1 - y4) * (z2 - z4) - (y2 - y4) * (z1 - z4));
        dN[2][1] = inv_6V * ((z1 - z4) * (x2 - x4) - (z2 - z4) * (x1 - x4));
        dN[2][2] = inv_6V * ((x1 - x4) * (y2 - y4) - (x2 - x4) * (y1 - y4));

        // dN4/dx,dy,dz
        dN[3][0] = inv_6V * ((y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1));
        dN[3][1] = inv_6V * ((z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1));
        dN[3][2] = inv_6V * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

        // Build B-matrix (6x12)
        Real B[6][12] = {0};
        for (int i = 0; i < 4; ++i) {
            B[0][i * 3 + 0] = dN[i][0];  // εxx
            B[1][i * 3 + 1] = dN[i][1];  // εyy
            B[2][i * 3 + 2] = dN[i][2];  // εzz
            B[3][i * 3 + 0] = dN[i][1];  // γxy
            B[3][i * 3 + 1] = dN[i][0];
            B[4][i * 3 + 1] = dN[i][2];  // γyz
            B[4][i * 3 + 2] = dN[i][1];
            B[5][i * 3 + 0] = dN[i][2];  // γxz
            B[5][i * 3 + 2] = dN[i][0];
        }

        // Constitutive matrix D (isotropic elastic, 3D)
        Real E = material_.E;
        Real nu = material_.nu;
        Real factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu));

        Real D[6][6] = {0};
        D[0][0] = D[1][1] = D[2][2] = factor * (1.0 - nu);
        D[0][1] = D[0][2] = D[1][0] = D[1][2] = D[2][0] = D[2][1] = factor * nu;
        D[3][3] = D[4][4] = D[5][5] = factor * (1.0 - 2.0 * nu) / 2.0;

        // Ke = V * B^T * D * B
        // First compute DB = D * B (6x12)
        Real DB[6][12] = {0};
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 12; ++j) {
                for (int k = 0; k < 6; ++k) {
                    DB[i][j] += D[i][k] * B[k][j];
                }
            }
        }

        // Then Ke = V * B^T * DB (12x12)
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 12; ++j) {
                Real sum = 0.0;
                for (int k = 0; k < 6; ++k) {
                    sum += B[k][i] * DB[k][j];
                }
                ke[i * 12 + j] = std::abs(V) * sum;
            }
        }
    }

    /**
     * @brief Apply Dirichlet boundary conditions using penalty method
     */
    void apply_dirichlet_bcs(std::vector<Real>& F) {
        // Find max diagonal for scaling the penalty
        Real max_diag = 0.0;
        for (size_t i = 0; i < ndof_; ++i) {
            max_diag = std::max(max_diag, std::abs(K_global_.get(i, i)));
        }

        // Use penalty = 1e12 * max_diag for better conditioning
        Real penalty = std::max(1e12 * max_diag, 1e20);

        for (const auto& bc : dirichlet_bcs_) {
            size_t dof_idx = bc.node_id * 3 + bc.dof;

            // Store original diagonal for reaction computation
            original_diag_[dof_idx] = K_global_.get(dof_idx, dof_idx);

            // Add penalty to diagonal
            K_global_.set(dof_idx, dof_idx, original_diag_[dof_idx] + penalty);

            // Modify RHS: F = penalty * prescribed_value
            F[dof_idx] = penalty * bc.value;
        }

        penalty_value_ = penalty;
    }

    /**
     * @brief Compute reaction forces at constrained nodes
     *
     * For penalty method: R = K_original * u - sum(K_ij * u_j for j != i)
     * Simplified: We compute R = sum over all elements of element internal force
     * But here we use the penalty residual: R_i = penalty * (u_i - u_prescribed)
     */
    void compute_reactions(Result& result) {
        result.reaction_forces.resize(ndof_, 0.0);

        // For penalty method, reaction force is approximately:
        // R = penalty * (u_computed - u_prescribed) which should be ~0 if converged
        // A more accurate method would require storing the original stiffness
        // and computing R = K_original * u - F_external

        // Simple estimate: use the penalty equation
        for (const auto& bc : dirichlet_bcs_) {
            size_t dof_idx = bc.node_id * 3 + bc.dof;
            // The reaction is F_applied - K*u for the unconstrained system
            // With penalty: F = penalty * u_prescribed
            // After solve: u ≈ u_prescribed
            // Reaction ≈ penalty * (u - u_prescribed) + original_K_ii * u + sum(K_ij * u_j)
            // For homogeneous BC (u_prescribed=0), R = sum(K_ij * u_j) for j != constrained

            // We stored original diagonal, so estimate reaction from that
            Real u_i = result.displacement[dof_idx];
            result.reaction_forces[dof_idx] = original_diag_.count(dof_idx) ?
                penalty_value_ * (bc.value - u_i) : 0.0;
        }
    }

    const Mesh* mesh_ = nullptr;
    size_t ndof_ = 0;
    ElasticMaterial material_;
    SparseMatrix K_global_;

    std::vector<DirichletBC> dirichlet_bcs_;
    std::vector<NeumannBC> neumann_bcs_;
    std::set<size_t> constrained_dofs_;
    std::unordered_map<size_t, Real> original_diag_;
    Real penalty_value_ = 0.0;

    size_t nan_element_count_ = 0;
    size_t zero_diagonal_count_ = 0;
};

// ============================================================================
// Simple Mesh Generator for Testing
// ============================================================================

/**
 * @brief Generate a simple cantilever beam mesh (Hex8 elements)
 */
inline Mesh generate_cantilever_mesh(Real length, Real width, Real height,
                                     int nx, int ny, int nz) {
    size_t num_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    size_t num_elems = nx * ny * nz;

    Mesh mesh(num_nodes);

    Real dx = length / nx;
    Real dy = width / ny;
    Real dz = height / nz;

    // Create nodes
    size_t node_idx = 0;
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                mesh.set_node_coordinates(node_idx, {i * dx, j * dy, k * dz});
                ++node_idx;
            }
        }
    }

    // Create element block
    Index block_id = mesh.add_element_block("beam", nxs::ElementType::Hex8, num_elems, 8);
    auto& block = mesh.element_block(block_id);

    // Create elements
    size_t elem_idx = 0;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                // Node indices for this element
                auto node = [&](int di, int dj, int dk) {
                    return (k + dk) * (ny + 1) * (nx + 1) + (j + dj) * (nx + 1) + (i + di);
                };

                auto elem_nodes = block.element_nodes(elem_idx);
                elem_nodes[0] = node(0, 0, 0);
                elem_nodes[1] = node(1, 0, 0);
                elem_nodes[2] = node(1, 1, 0);
                elem_nodes[3] = node(0, 1, 0);
                elem_nodes[4] = node(0, 0, 1);
                elem_nodes[5] = node(1, 0, 1);
                elem_nodes[6] = node(1, 1, 1);
                elem_nodes[7] = node(0, 1, 1);

                ++elem_idx;
            }
        }
    }

    return mesh;
}

/**
 * @brief Get nodes at x=0 (fixed end for cantilever)
 */
inline std::vector<Index> get_nodes_at_x(const Mesh& mesh, Real x_val, Real tol = 1e-6) {
    std::vector<Index> nodes;
    for (size_t i = 0; i < mesh.num_nodes(); ++i) {
        auto coords = mesh.get_node_coordinates(i);
        if (std::abs(coords[0] - x_val) < tol) {
            nodes.push_back(i);
        }
    }
    return nodes;
}

/**
 * @brief Get nodes at x=L (free end for cantilever)
 */
inline std::vector<Index> get_nodes_at_x_max(const Mesh& mesh, Real tol = 1e-6) {
    Real x_max = 0.0;
    for (size_t i = 0; i < mesh.num_nodes(); ++i) {
        auto coords = mesh.get_node_coordinates(i);
        x_max = std::max(x_max, coords[0]);
    }
    return get_nodes_at_x(mesh, x_max, tol);
}

// ============================================================================
// FEM Implicit Dynamic Solver (Newmark-β)
// ============================================================================

/**
 * @brief Implicit FEM dynamic solver using Newmark-β time integration
 *
 * Solves: M*a + C*v + K*u = F_ext
 *
 * Uses average acceleration method (β=0.25, γ=0.5) which is:
 * - Unconditionally stable
 * - Second-order accurate
 * - No numerical damping
 */
class FEMImplicitDynamicSolver {
public:
    struct TimeStepResult {
        bool converged = false;
        int iterations = 0;
        Real residual = 0.0;
        Real kinetic_energy = 0.0;
        Real strain_energy = 0.0;
    };

    /**
     * @brief Initialize solver with mesh
     */
    void set_mesh(const Mesh& mesh) {
        mesh_ = &mesh;
        ndof_ = mesh.num_nodes() * 3;

        // Initialize state vectors
        u_.resize(ndof_, 0.0);
        v_.resize(ndof_, 0.0);
        a_.resize(ndof_, 0.0);
        u_pred_.resize(ndof_);
        v_pred_.resize(ndof_);

        // Build mass matrix (lumped)
        build_mass_matrix();

        // Build stiffness sparsity pattern
        build_sparsity_pattern();
    }

    void set_material(const ElasticMaterial& mat) {
        material_ = mat;
        // Rebuild mass matrix with new density
        if (mesh_) {
            build_mass_matrix();
        }
    }

    /**
     * @brief Set Newmark-β parameters
     * @param beta β parameter (0.25 for average acceleration)
     * @param gamma γ parameter (0.5 for average acceleration)
     */
    void set_newmark_parameters(Real beta, Real gamma) {
        beta_ = beta;
        gamma_ = gamma;
    }

    /**
     * @brief Set Rayleigh damping: C = alpha*M + beta*K
     */
    void set_rayleigh_damping(Real alpha_M, Real beta_K) {
        damping_alpha_ = alpha_M;
        damping_beta_ = beta_K;
    }

    // Boundary condition methods (same as static solver)
    void add_dirichlet_bc(Index node, int dof, Real value = 0.0) {
        dirichlet_bcs_.emplace_back(node, dof, value);
        constrained_dofs_.insert(node * 3 + dof);
    }

    void fix_node(Index node) {
        add_dirichlet_bc(node, 0, 0.0);
        add_dirichlet_bc(node, 1, 0.0);
        add_dirichlet_bc(node, 2, 0.0);
    }

    void add_force(Index node, int dof, Real force) {
        neumann_bcs_.emplace_back(node, dof, force);
    }

    void clear_forces() {
        neumann_bcs_.clear();
    }

    /**
     * @brief Set initial displacement
     */
    void set_initial_displacement(const std::vector<Real>& u0) {
        if (u0.size() == ndof_) {
            u_ = u0;
        }
    }

    /**
     * @brief Set initial velocity
     */
    void set_initial_velocity(const std::vector<Real>& v0) {
        if (v0.size() == ndof_) {
            v_ = v0;
        }
    }

    /**
     * @brief Compute initial acceleration from equilibrium
     * Call after setting initial conditions and forces
     */
    void compute_initial_acceleration() {
        // a0 = M^{-1} * (F_ext - K*u0 - C*v0)
        assemble_stiffness();

        std::vector<Real> F_ext(ndof_, 0.0);
        for (const auto& bc : neumann_bcs_) {
            F_ext[bc.node_id * 3 + bc.dof] += bc.value;
        }

        std::vector<Real> Ku;
        K_global_.multiply(u_, Ku);

        for (size_t i = 0; i < ndof_; ++i) {
            Real Cv_i = damping_alpha_ * M_diag_[i] * v_[i];
            // Add stiffness-proportional damping (beta*K*v)
            // For simplicity, approximate with diagonal

            a_[i] = (F_ext[i] - Ku[i] - Cv_i) / M_diag_[i];

            // Apply zero acceleration at constrained DOFs
            if (constrained_dofs_.count(i)) {
                a_[i] = 0.0;
            }
        }
    }

    /**
     * @brief Perform one implicit time step
     *
     * Using standard Newmark-β formulation:
     * u_{n+1} = u_n + dt*v_n + dt²*((0.5-β)*a_n + β*a_{n+1})
     * v_{n+1} = v_n + dt*((1-γ)*a_n + γ*a_{n+1})
     *
     * Solve for a_{n+1} from: M*a_{n+1} + C*v_{n+1} + K*u_{n+1} = F_{n+1}
     */
    TimeStepResult step(Real dt) {
        TimeStepResult result;
        dt_ = dt;

        // Assemble stiffness matrix
        assemble_stiffness();

        // Build external force vector
        std::vector<Real> F_ext(ndof_, 0.0);
        for (const auto& bc : neumann_bcs_) {
            F_ext[bc.node_id * 3 + bc.dof] += bc.value;
        }

        // Newmark coefficients
        Real a0 = 1.0 / (beta_ * dt * dt);
        Real a1 = gamma_ / (beta_ * dt);
        Real a2 = 1.0 / (beta_ * dt);
        Real a3 = 1.0 / (2.0 * beta_) - 1.0;
        Real a4 = gamma_ / beta_ - 1.0;
        Real a5 = dt * (gamma_ / (2.0 * beta_) - 1.0);
        Real a6 = dt * (1.0 - gamma_);
        Real a7 = gamma_ * dt;

        // Build effective stiffness matrix: K_eff = K + a1*C + a0*M
        K_eff_ = K_global_;

        // Add mass and damping contributions
        for (size_t i = 0; i < ndof_; ++i) {
            Real K_ii = K_global_.get(i, i);
            Real M_ii = M_diag_[i];
            Real C_ii = damping_alpha_ * M_ii + damping_beta_ * K_ii;

            Real K_eff_ii = K_ii + a1 * C_ii + a0 * M_ii;
            K_eff_.set(i, i, K_eff_ii);
        }

        // Build effective force vector
        // F_eff = F_ext + M*(a0*u_n + a2*v_n + a3*a_n) + C*(a1*u_n + a4*v_n + a5*a_n)
        std::vector<Real> F_eff(ndof_);

        for (size_t i = 0; i < ndof_; ++i) {
            Real M_ii = M_diag_[i];
            Real K_ii = K_global_.get(i, i);
            Real C_ii = damping_alpha_ * M_ii + damping_beta_ * K_ii;

            F_eff[i] = F_ext[i]
                     + M_ii * (a0 * u_[i] + a2 * v_[i] + a3 * a_[i])
                     + C_ii * (a1 * u_[i] + a4 * v_[i] + a5 * a_[i]);
        }

        // Apply boundary conditions
        apply_dirichlet_bcs_dynamic(F_eff);

        // Solve K_eff * u_{n+1} = F_eff
        std::vector<Real> u_new(ndof_, 0.0);

        // Use current displacement as initial guess
        for (size_t i = 0; i < ndof_; ++i) {
            u_new[i] = u_[i];
        }

        CGSolver cg;
        cg.set_tolerance(1e-10);
        cg.set_max_iterations(10000);
        cg.set_preconditioner(true);

        auto lin_result = cg.solve(K_eff_, F_eff, u_new);

        result.converged = lin_result.converged;
        result.iterations = lin_result.iterations;
        result.residual = lin_result.residual;

        if (!result.converged) {
            return result;
        }

        // Update acceleration and velocity
        for (size_t i = 0; i < ndof_; ++i) {
            Real delta_u = u_new[i] - u_[i];
            Real a_new = a0 * delta_u - a2 * v_[i] - a3 * a_[i];
            Real v_new = v_[i] + a6 * a_[i] + a7 * a_new;

            a_[i] = a_new;
            v_[i] = v_new;
            u_[i] = u_new[i];

            // Enforce BCs
            if (constrained_dofs_.count(i)) {
                u_[i] = 0.0;
                v_[i] = 0.0;
                a_[i] = 0.0;
            }
        }

        time_ += dt;

        // Compute energies
        result.kinetic_energy = compute_kinetic_energy();
        result.strain_energy = compute_strain_energy();

        return result;
    }

    /**
     * @brief Run simulation for given duration
     */
    std::vector<TimeStepResult> run(Real duration, Real dt, int output_interval = 1) {
        std::vector<TimeStepResult> history;

        int num_steps = static_cast<int>(duration / dt);
        for (int n = 0; n < num_steps; ++n) {
            auto result = step(dt);

            if (n % output_interval == 0) {
                history.push_back(result);
            }

            if (!result.converged) {
                NXS_LOG_WARN("Step {} did not converge", n);
                break;
            }
        }

        return history;
    }

    // Accessors
    const std::vector<Real>& displacement() const { return u_; }
    const std::vector<Real>& velocity() const { return v_; }
    const std::vector<Real>& acceleration() const { return a_; }
    Real time() const { return time_; }
    size_t num_dofs() const { return ndof_; }

    Real compute_kinetic_energy() const {
        Real KE = 0.0;
        for (size_t i = 0; i < ndof_; ++i) {
            KE += 0.5 * M_diag_[i] * v_[i] * v_[i];
        }
        return KE;
    }

    Real compute_strain_energy() const {
        std::vector<Real> Ku;
        K_global_.multiply(u_, Ku);

        Real SE = 0.0;
        for (size_t i = 0; i < ndof_; ++i) {
            SE += 0.5 * u_[i] * Ku[i];
        }
        return SE;
    }

    Real compute_total_energy() const {
        return compute_kinetic_energy() + compute_strain_energy();
    }

private:
    void build_sparsity_pattern() {
        // Same as FEMStaticSolver
        std::vector<std::set<Index>> node_adjacency(mesh_->num_nodes());

        for (const auto& block : mesh_->element_blocks()) {
            for (size_t e = 0; e < block.num_elements(); ++e) {
                auto elem_nodes = block.element_nodes(e);
                for (size_t i = 0; i < block.num_nodes_per_elem; ++i) {
                    for (size_t j = 0; j < block.num_nodes_per_elem; ++j) {
                        node_adjacency[elem_nodes[i]].insert(elem_nodes[j]);
                    }
                }
            }
        }

        std::vector<std::vector<size_t>> dof_pattern(ndof_);
        for (size_t node_i = 0; node_i < mesh_->num_nodes(); ++node_i) {
            for (size_t node_j : node_adjacency[node_i]) {
                for (int di = 0; di < 3; ++di) {
                    for (int dj = 0; dj < 3; ++dj) {
                        dof_pattern[node_i * 3 + di].push_back(node_j * 3 + dj);
                    }
                }
            }
        }

        K_global_.create_pattern(ndof_, ndof_, dof_pattern);
        K_eff_.create_pattern(ndof_, ndof_, dof_pattern);
    }

    void build_mass_matrix() {
        // Lumped mass matrix (diagonal)
        M_diag_.resize(ndof_, 0.0);

        for (const auto& block : mesh_->element_blocks()) {
            size_t nodes_per_elem = block.num_nodes_per_elem;

            for (size_t e = 0; e < block.num_elements(); ++e) {
                auto elem_nodes = block.element_nodes(e);

                // Gather element coordinates for volume calculation
                std::vector<Real> elem_coords(nodes_per_elem * 3);
                for (size_t i = 0; i < nodes_per_elem; ++i) {
                    auto coords = mesh_->get_node_coordinates(elem_nodes[i]);
                    elem_coords[i * 3 + 0] = coords[0];
                    elem_coords[i * 3 + 1] = coords[1];
                    elem_coords[i * 3 + 2] = coords[2];
                }

                // Compute element volume (approximate for hex8)
                Real volume = compute_element_volume(elem_coords, nodes_per_elem);
                Real elem_mass = material_.rho * volume;
                Real mass_per_node = elem_mass / nodes_per_elem;

                // Distribute mass to nodes
                for (size_t i = 0; i < nodes_per_elem; ++i) {
                    for (int d = 0; d < 3; ++d) {
                        M_diag_[elem_nodes[i] * 3 + d] += mass_per_node;
                    }
                }
            }
        }
    }

    Real compute_element_volume(const std::vector<Real>& coords, size_t nodes_per_elem) {
        if (nodes_per_elem == 8) {
            // Hex8: approximate with bounding box for simplicity
            Real x_min = coords[0], x_max = coords[0];
            Real y_min = coords[1], y_max = coords[1];
            Real z_min = coords[2], z_max = coords[2];

            for (size_t i = 1; i < 8; ++i) {
                x_min = std::min(x_min, coords[i * 3 + 0]);
                x_max = std::max(x_max, coords[i * 3 + 0]);
                y_min = std::min(y_min, coords[i * 3 + 1]);
                y_max = std::max(y_max, coords[i * 3 + 1]);
                z_min = std::min(z_min, coords[i * 3 + 2]);
                z_max = std::max(z_max, coords[i * 3 + 2]);
            }
            return (x_max - x_min) * (y_max - y_min) * (z_max - z_min);
        } else if (nodes_per_elem == 10) {
            fem::Tet10Element tet10;
            return tet10.volume(coords.data());
        } else if (nodes_per_elem == 4) {
            // Tet4 volume
            const Real* x = coords.data();
            Real a1 = x[3] - x[0], a2 = x[6] - x[0], a3 = x[9] - x[0];
            Real b1 = x[4] - x[1], b2 = x[7] - x[1], b3 = x[10] - x[1];
            Real c1 = x[5] - x[2], c2 = x[8] - x[2], c3 = x[11] - x[2];
            Real V6 = a1*(b2*c3-b3*c2) - a2*(b1*c3-b3*c1) + a3*(b1*c2-b2*c1);
            return std::abs(V6) / 6.0;
        }
        return 1.0;  // Fallback
    }

    void assemble_stiffness() {
        // Same as FEMStaticSolver
        K_global_.zero();
        fem::Hex8Element hex8;
        fem::Hex20Element hex20;
        fem::Tet4Element tet4;
        fem::Tet10Element tet10;

        for (const auto& block : mesh_->element_blocks()) {
            size_t nodes_per_elem = block.num_nodes_per_elem;
            size_t elem_ndof = nodes_per_elem * 3;

            std::vector<Real> ke(elem_ndof * elem_ndof);
            std::vector<Real> elem_coords(nodes_per_elem * 3);
            std::vector<Index> dof_map(elem_ndof);

            for (size_t e = 0; e < block.num_elements(); ++e) {
                auto elem_nodes = block.element_nodes(e);

                for (size_t i = 0; i < nodes_per_elem; ++i) {
                    auto coords = mesh_->get_node_coordinates(elem_nodes[i]);
                    elem_coords[i * 3 + 0] = coords[0];
                    elem_coords[i * 3 + 1] = coords[1];
                    elem_coords[i * 3 + 2] = coords[2];
                }

                for (size_t i = 0; i < nodes_per_elem; ++i) {
                    dof_map[i * 3 + 0] = elem_nodes[i] * 3 + 0;
                    dof_map[i * 3 + 1] = elem_nodes[i] * 3 + 1;
                    dof_map[i * 3 + 2] = elem_nodes[i] * 3 + 2;
                }

                // Compute element stiffness based on element type
                if (nodes_per_elem == 8) {
                    hex8.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                } else if (nodes_per_elem == 20) {
                    hex20.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                } else if (nodes_per_elem == 10) {
                    tet10.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                } else if (nodes_per_elem == 4) {
                    tet4.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                }

                // Guard: scan element stiffness for NaN
                bool has_nan = false;
                for (size_t idx = 0; idx < ke.size(); ++idx) {
                    if (std::isnan(ke[idx]) || std::isinf(ke[idx])) {
                        has_nan = true;
                        break;
                    }
                }
                if (has_nan) continue;  // Skip this element

                K_global_.add_element_matrix(dof_map, ke);
            }
        }
    }

    void apply_dirichlet_bcs_dynamic(std::vector<Real>& F) {
        // For dynamics, use penalty method
        Real max_diag = 0.0;
        for (size_t i = 0; i < ndof_; ++i) {
            max_diag = std::max(max_diag, std::abs(K_eff_.get(i, i)));
        }
        Real penalty = std::max(1e12 * max_diag, 1e20);

        for (const auto& bc : dirichlet_bcs_) {
            size_t dof_idx = bc.node_id * 3 + bc.dof;
            Real current_diag = K_eff_.get(dof_idx, dof_idx);
            K_eff_.set(dof_idx, dof_idx, current_diag + penalty);
            F[dof_idx] = penalty * bc.value;
        }
    }

    const Mesh* mesh_ = nullptr;
    size_t ndof_ = 0;
    ElasticMaterial material_;

    // State vectors
    std::vector<Real> u_, v_, a_;
    std::vector<Real> u_pred_, v_pred_;
    Real time_ = 0.0;
    Real dt_ = 0.0;

    // Matrices
    SparseMatrix K_global_;
    SparseMatrix K_eff_;
    std::vector<Real> M_diag_;

    // Newmark parameters (average acceleration by default)
    Real beta_ = 0.25;
    Real gamma_ = 0.5;

    // Rayleigh damping
    Real damping_alpha_ = 0.0;
    Real damping_beta_ = 0.0;

    // Boundary conditions
    std::vector<DirichletBC> dirichlet_bcs_;
    std::vector<NeumannBC> neumann_bcs_;
    std::set<size_t> constrained_dofs_;
};

} // namespace solver
} // namespace nxs
