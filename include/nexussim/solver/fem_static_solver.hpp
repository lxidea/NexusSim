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
#include <nexussim/solver/arc_length_solver.hpp>
#include <nexussim/discretization/hex8.hpp>
#include <nexussim/discretization/hex20.hpp>
#include <nexussim/discretization/tet4.hpp>
#include <nexussim/discretization/tet10.hpp>
#include <nexussim/discretization/shell4.hpp>
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
     *
     * Auto-detects DOFs per node: 6 if any shell/beam elements present, 3 otherwise.
     */
    void set_mesh(const Mesh& mesh) {
        mesh_ = &mesh;

        // Detect DOF count from element types
        dof_per_node_ = 3;
        for (const auto& block : mesh.element_blocks()) {
            if (block.type == ElementType::Shell4 || block.type == ElementType::Shell3 ||
                block.type == ElementType::Shell6 ||
                block.type == ElementType::Beam2 || block.type == ElementType::Beam3) {
                dof_per_node_ = 6;
                break;
            }
        }

        ndof_ = mesh.num_nodes() * dof_per_node_;

        // Build sparsity pattern from mesh connectivity
        build_sparsity_pattern();
    }

    /**
     * @brief Set shell thickness for Shell4 elements
     */
    void set_shell_thickness(Real t) { shell_thickness_ = t; }

    /**
     * @brief Get DOFs per node (3 for solids only, 6 if shells present)
     */
    int dof_per_node() const { return dof_per_node_; }

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
        constrained_dofs_.insert(node * dof_per_node_ + dof);
    }

    /**
     * @brief Fix translational DOFs (0,1,2) at a node
     */
    void fix_node(Index node) {
        add_dirichlet_bc(node, 0, 0.0);
        add_dirichlet_bc(node, 1, 0.0);
        add_dirichlet_bc(node, 2, 0.0);
    }

    /**
     * @brief Fix all DOFs (translations + rotations) at a node
     */
    void fix_node_all(Index node) {
        for (int d = 0; d < dof_per_node_; ++d) {
            add_dirichlet_bc(node, d, 0.0);
        }
    }

    /**
     * @brief Apply moment at a node
     * @param rot_dof Rotational DOF index (3=Mx, 4=My, 5=Mz)
     */
    void add_moment(Index node, int rot_dof, Real moment) {
        neumann_bcs_.emplace_back(node, rot_dof, moment);
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
            size_t dof_idx = bc.node_id * dof_per_node_ + bc.dof;
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
            size_t dof_idx = bc.node_id * dof_per_node_ + bc.dof;
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
     * @brief Configuration for arc-length analysis
     */
    struct ArcLengthConfig {
        Real arc_length = 0.1;
        Real psi = 0.0;          // 0 = cylindrical
        int max_steps = 50;
        Real tolerance = 1e-6;
        int max_corrections = 20;
        Real lambda_max = 2.0;
        int desired_iterations = 5;
        Real min_arc_length = 1e-8;
        Real max_arc_length = 10.0;
        bool verbose = false;
    };

    /**
     * @brief Solve using arc-length method (for snap-through/buckling)
     *
     * Builds reference load from Neumann BCs, assembles stiffness, and
     * uses ArcLengthSolver to trace the nonlinear equilibrium path.
     */
    ArcLengthResult solve_arc_length(const ArcLengthConfig& config) {
        ArcLengthResult empty_result;

        if (!mesh_) {
            NXS_LOG_ERROR("FEMStaticSolver: No mesh set");
            return empty_result;
        }

        // Assemble stiffness once (linear elastic: K is constant)
        nan_element_count_ = 0;
        assemble_stiffness();

        // Build reference load vector from Neumann BCs
        std::vector<Real> F_ref(ndof_, 0.0);
        for (const auto& bc : neumann_bcs_) {
            size_t dof_idx = bc.node_id * dof_per_node_ + bc.dof;
            F_ref[dof_idx] += bc.value;
        }

        // Zero out constrained DOF loads
        for (size_t dof : constrained_dofs_) {
            F_ref[dof] = 0.0;
        }

        // Apply Dirichlet BCs to tangent stiffness (penalty on diagonal)
        apply_dirichlet_bcs_tangent(K_global_);

        // Set up arc-length solver
        ArcLengthSolver arc_solver;

        // Internal force: F_int = K * u (linear elastic)
        arc_solver.set_internal_force_function(
            [this](const std::vector<Real>& u, std::vector<Real>& F_int) {
                K_global_.multiply(u, F_int);
            });

        // Tangent stiffness: constant K for linear elastic
        arc_solver.set_tangent_function(
            [this](const std::vector<Real>& /*u*/, SparseMatrix& K_t) {
                K_t = K_global_;
            });

        arc_solver.set_reference_load(F_ref);
        arc_solver.set_arc_length(config.arc_length);
        arc_solver.set_psi(config.psi);
        arc_solver.set_max_steps(config.max_steps);
        arc_solver.set_tolerance(config.tolerance);
        arc_solver.set_max_corrections(config.max_corrections);
        arc_solver.set_desired_iterations(config.desired_iterations);
        arc_solver.set_arc_length_bounds(config.min_arc_length, config.max_arc_length);
        arc_solver.set_verbose(config.verbose);

        // Use direct solver for small problems, CG for larger
        if (ndof_ > 500) {
            arc_solver.set_linear_solver(LinearSolverType::ConjugateGradient);
        }

        // Solve
        std::vector<Real> u(ndof_, 0.0);
        Real lambda = 0.0;
        auto result = arc_solver.solve(u, lambda, config.lambda_max);

        return result;
    }

    /**
     * @brief Get translational displacement at a node
     * @param dpn DOFs per node (default 3 for backward compatibility)
     */
    static std::array<Real, 3> get_node_displacement(const Result& result, Index node, int dpn = 3) {
        return {
            result.displacement[node * dpn + 0],
            result.displacement[node * dpn + 1],
            result.displacement[node * dpn + 2]
        };
    }

    /**
     * @brief Compute maximum translational displacement magnitude
     * @param dpn DOFs per node (default 3 for backward compatibility)
     */
    static Real max_displacement(const Result& result, int dpn = 3) {
        Real max_disp = 0.0;
        size_t num_nodes = result.displacement.size() / dpn;
        for (size_t i = 0; i < num_nodes; ++i) {
            Real ux = result.displacement[i * dpn + 0];
            Real uy = result.displacement[i * dpn + 1];
            Real uz = result.displacement[i * dpn + 2];
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
                // Each node pair creates a dof_per_node x dof_per_node block
                for (int di = 0; di < dof_per_node_; ++di) {
                    for (int dj = 0; dj < dof_per_node_; ++dj) {
                        dof_pattern[node_i * dof_per_node_ + di].push_back(node_j * dof_per_node_ + dj);
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
        fem::Shell4Element shell4;
        shell4.set_thickness(shell_thickness_);

        // Track which nodes are connected to shell elements (for rotational penalty)
        std::set<Index> shell_nodes;

        // Process each element block
        for (const auto& block : mesh_->element_blocks()) {
            size_t nodes_per_elem = block.num_nodes_per_elem;
            bool is_shell4 = (block.type == ElementType::Shell4);

            if (is_shell4) {
                // Shell4: 24 DOFs per element (4 nodes × 6 DOFs)
                size_t elem_ndof = 4 * 6;
                std::vector<Real> ke(elem_ndof * elem_ndof);
                std::vector<Real> elem_coords(4 * 3);
                std::vector<Index> dof_map(elem_ndof);

                for (size_t e = 0; e < block.num_elements(); ++e) {
                    auto elem_nodes = block.element_nodes(e);

                    // Gather element coordinates
                    for (size_t i = 0; i < 4; ++i) {
                        auto coords = mesh_->get_node_coordinates(elem_nodes[i]);
                        elem_coords[i * 3 + 0] = coords[0];
                        elem_coords[i * 3 + 1] = coords[1];
                        elem_coords[i * 3 + 2] = coords[2];
                        shell_nodes.insert(elem_nodes[i]);
                    }

                    // Compute shell element stiffness in local coordinates
                    shell4.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());

                    // Transform local stiffness to global coordinates
                    transform_shell_stiffness_to_global(elem_coords.data(), ke);

                    // Build DOF map: all 6 DOFs per node
                    for (size_t i = 0; i < 4; ++i) {
                        for (int d = 0; d < 6; ++d) {
                            dof_map[i * 6 + d] = elem_nodes[i] * dof_per_node_ + d;
                        }
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
                        continue;
                    }

                    K_global_.add_element_matrix(dof_map, ke);
                }
            } else {
                // Solid elements: 3 translational DOFs per node
                size_t elem_ndof = nodes_per_elem * 3;
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

                    // Build DOF map: translational DOFs only
                    for (size_t i = 0; i < nodes_per_elem; ++i) {
                        dof_map[i * 3 + 0] = elem_nodes[i] * dof_per_node_ + 0;
                        dof_map[i * 3 + 1] = elem_nodes[i] * dof_per_node_ + 1;
                        dof_map[i * 3 + 2] = elem_nodes[i] * dof_per_node_ + 2;
                    }

                    // Compute element stiffness based on element type
                    if (nodes_per_elem == 8) {
                        hex8.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                    } else if (nodes_per_elem == 20) {
                        hex20.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                    } else if (nodes_per_elem == 10) {
                        tet10.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                    } else if (nodes_per_elem == 4 && block.type == ElementType::Tet4) {
                        tet4.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                    } else {
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
                        continue;
                    }

                    K_global_.add_element_matrix(dof_map, ke);
                }
            }
        }

        // Add rotational penalty for solid-only nodes in 6-DOF mode
        if (dof_per_node_ == 6) {
            add_solid_rotational_penalty(shell_nodes);
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
            size_t dof_idx = bc.node_id * dof_per_node_ + bc.dof;

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
            size_t dof_idx = bc.node_id * dof_per_node_ + bc.dof;
            Real u_i = result.displacement[dof_idx];
            result.reaction_forces[dof_idx] = original_diag_.count(dof_idx) ?
                penalty_value_ * (bc.value - u_i) : 0.0;
        }
    }

    /**
     * @brief Transform shell element stiffness from local to global coordinates
     *
     * Builds a 24×24 transformation matrix T from shell local axes to global,
     * then computes K_global = T * K_local * T^T
     */
    void transform_shell_stiffness_to_global(const Real* elem_coords,
                                              std::vector<Real>& ke) {
        // Get local coordinate system from Shell4
        fem::Shell4Element shell4_tmp;
        Real e1[3], e2[3], e3[3];
        shell4_tmp.local_coordinate_system(elem_coords, e1, e2, e3);

        // Build 24×24 block-diagonal transformation T
        // 4 blocks of 6×6, each block = [R 0; 0 R] where R = [e1|e2|e3]^T
        // R transforms from local to global: v_global = R^T * v_local
        // Actually R = [e1|e2|e3] as columns = local axes in global frame
        // So v_global = R * v_local, and T maps local DOFs to global DOFs

        constexpr int N = 24;
        std::vector<Real> T(N * N, 0.0);

        // R matrix: columns are local axes expressed in global frame
        // R[row][col] where row = global component, col = local component
        Real R[3][3] = {
            {e1[0], e2[0], e3[0]},
            {e1[1], e2[1], e3[1]},
            {e1[2], e2[2], e3[2]}
        };

        for (int node = 0; node < 4; ++node) {
            int base = node * 6;
            // Translational block (3x3)
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    T[(base + i) * N + (base + j)] = R[i][j];
                }
            }
            // Rotational block (3x3) - same R
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    T[(base + 3 + i) * N + (base + 3 + j)] = R[i][j];
                }
            }
        }

        // Compute K_global = T * K_local * T^T
        // First: temp = K_local * T^T  (24x24)
        std::vector<Real> temp(N * N, 0.0);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                Real sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += ke[i * N + k] * T[j * N + k]; // T^T[k][j] = T[j][k]
                }
                temp[i * N + j] = sum;
            }
        }

        // Then: K_global = T * temp  (24x24)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                Real sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += T[i * N + k] * temp[k * N + j];
                }
                ke[i * N + j] = sum;
            }
        }
    }

    /**
     * @brief Add small rotational penalty stiffness for solid-only nodes
     *
     * In 6-DOF mode, nodes not connected to any shell element have
     * unconstrained rotational DOFs. Add a small penalty to prevent singularity.
     */
    void add_solid_rotational_penalty(const std::set<Index>& shell_nodes) {
        // Find max diagonal for scaling
        Real max_diag = 0.0;
        for (size_t i = 0; i < ndof_; ++i) {
            max_diag = std::max(max_diag, std::abs(K_global_.get(i, i)));
        }

        Real penalty = 1e-6 * max_diag;
        if (penalty < 1e-20) return;

        for (size_t node = 0; node < mesh_->num_nodes(); ++node) {
            if (shell_nodes.count(node) == 0) {
                // Solid-only node: add penalty to rotational DOFs (3,4,5)
                for (int d = 3; d < 6; ++d) {
                    size_t dof = node * dof_per_node_ + d;
                    Real current = K_global_.get(dof, dof);
                    K_global_.set(dof, dof, current + penalty);
                }
            }
        }
    }

    /**
     * @brief Apply Dirichlet BCs to tangent stiffness for arc-length method
     *
     * Adds large penalty to constrained DOF diagonal entries so that
     * constrained DOFs are effectively locked.
     */
    void apply_dirichlet_bcs_tangent(SparseMatrix& K) {
        Real max_diag = 0.0;
        for (size_t i = 0; i < ndof_; ++i) {
            max_diag = std::max(max_diag, std::abs(K.get(i, i)));
        }
        Real penalty = std::max(1e12 * max_diag, 1e20);

        for (size_t dof : constrained_dofs_) {
            Real current = K.get(dof, dof);
            K.set(dof, dof, current + penalty);
        }
    }

    const Mesh* mesh_ = nullptr;
    size_t ndof_ = 0;
    int dof_per_node_ = 3;
    Real shell_thickness_ = 0.01;
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
     *
     * Auto-detects DOFs per node: 6 if any shell/beam elements present, 3 otherwise.
     */
    void set_mesh(const Mesh& mesh) {
        mesh_ = &mesh;

        // Detect DOF count from element types
        dof_per_node_ = 3;
        for (const auto& block : mesh.element_blocks()) {
            if (block.type == ElementType::Shell4 || block.type == ElementType::Shell3 ||
                block.type == ElementType::Shell6 ||
                block.type == ElementType::Beam2 || block.type == ElementType::Beam3) {
                dof_per_node_ = 6;
                break;
            }
        }

        ndof_ = mesh.num_nodes() * dof_per_node_;

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

    void set_shell_thickness(Real t) { shell_thickness_ = t; }
    int dof_per_node() const { return dof_per_node_; }

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

    // Boundary condition methods
    void add_dirichlet_bc(Index node, int dof, Real value = 0.0) {
        dirichlet_bcs_.emplace_back(node, dof, value);
        constrained_dofs_.insert(node * dof_per_node_ + dof);
    }

    void fix_node(Index node) {
        add_dirichlet_bc(node, 0, 0.0);
        add_dirichlet_bc(node, 1, 0.0);
        add_dirichlet_bc(node, 2, 0.0);
    }

    void fix_node_all(Index node) {
        for (int d = 0; d < dof_per_node_; ++d) {
            add_dirichlet_bc(node, d, 0.0);
        }
    }

    void add_force(Index node, int dof, Real force) {
        neumann_bcs_.emplace_back(node, dof, force);
    }

    void add_moment(Index node, int rot_dof, Real moment) {
        neumann_bcs_.emplace_back(node, rot_dof, moment);
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
            F_ext[bc.node_id * dof_per_node_ + bc.dof] += bc.value;
        }

        std::vector<Real> Ku;
        K_global_.multiply(u_, Ku);

        for (size_t i = 0; i < ndof_; ++i) {
            Real Cv_i = damping_alpha_ * M_diag_[i] * v_[i];

            if (M_diag_[i] > 1e-30) {
                a_[i] = (F_ext[i] - Ku[i] - Cv_i) / M_diag_[i];
            } else {
                a_[i] = 0.0;
            }

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
            F_ext[bc.node_id * dof_per_node_ + bc.dof] += bc.value;
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
                for (int di = 0; di < dof_per_node_; ++di) {
                    for (int dj = 0; dj < dof_per_node_; ++dj) {
                        dof_pattern[node_i * dof_per_node_ + di].push_back(node_j * dof_per_node_ + dj);
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
        std::fill(M_diag_.begin(), M_diag_.end(), 0.0);

        fem::Shell4Element shell4_mass;
        shell4_mass.set_thickness(shell_thickness_);

        for (const auto& block : mesh_->element_blocks()) {
            size_t nodes_per_elem = block.num_nodes_per_elem;
            bool is_shell4 = (block.type == ElementType::Shell4);

            for (size_t e = 0; e < block.num_elements(); ++e) {
                auto elem_nodes = block.element_nodes(e);

                // Gather element coordinates
                std::vector<Real> elem_coords(nodes_per_elem * 3);
                for (size_t i = 0; i < nodes_per_elem; ++i) {
                    auto coords = mesh_->get_node_coordinates(elem_nodes[i]);
                    elem_coords[i * 3 + 0] = coords[0];
                    elem_coords[i * 3 + 1] = coords[1];
                    elem_coords[i * 3 + 2] = coords[2];
                }

                if (is_shell4) {
                    // Shell4: use lumped mass (distributes to all 6 DOFs)
                    Real M_lumped[24];
                    shell4_mass.lumped_mass_matrix(elem_coords.data(), material_.rho, M_lumped);
                    for (size_t i = 0; i < 4; ++i) {
                        for (int d = 0; d < 6; ++d) {
                            M_diag_[elem_nodes[i] * dof_per_node_ + d] += M_lumped[i * 6 + d];
                        }
                    }
                } else {
                    // Solid elements: lumped mass to translational DOFs
                    Real volume = compute_element_volume(elem_coords, nodes_per_elem);
                    Real elem_mass = material_.rho * volume;
                    Real mass_per_node = elem_mass / nodes_per_elem;

                    for (size_t i = 0; i < nodes_per_elem; ++i) {
                        for (int d = 0; d < 3; ++d) {
                            M_diag_[elem_nodes[i] * dof_per_node_ + d] += mass_per_node;
                        }
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
        K_global_.zero();
        fem::Hex8Element hex8;
        fem::Hex20Element hex20;
        fem::Tet4Element tet4;
        fem::Tet10Element tet10;
        fem::Shell4Element shell4;
        shell4.set_thickness(shell_thickness_);

        std::set<Index> shell_nodes;

        for (const auto& block : mesh_->element_blocks()) {
            size_t nodes_per_elem = block.num_nodes_per_elem;
            bool is_shell4 = (block.type == ElementType::Shell4);

            if (is_shell4) {
                size_t elem_ndof = 4 * 6;
                std::vector<Real> ke(elem_ndof * elem_ndof);
                std::vector<Real> elem_coords(4 * 3);
                std::vector<Index> dof_map(elem_ndof);

                for (size_t e = 0; e < block.num_elements(); ++e) {
                    auto elem_nodes = block.element_nodes(e);

                    for (size_t i = 0; i < 4; ++i) {
                        auto coords = mesh_->get_node_coordinates(elem_nodes[i]);
                        elem_coords[i * 3 + 0] = coords[0];
                        elem_coords[i * 3 + 1] = coords[1];
                        elem_coords[i * 3 + 2] = coords[2];
                        shell_nodes.insert(elem_nodes[i]);
                    }

                    shell4.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                    transform_shell_stiffness_to_global(elem_coords.data(), ke);

                    for (size_t i = 0; i < 4; ++i) {
                        for (int d = 0; d < 6; ++d) {
                            dof_map[i * 6 + d] = elem_nodes[i] * dof_per_node_ + d;
                        }
                    }

                    bool has_nan = false;
                    for (size_t idx = 0; idx < ke.size(); ++idx) {
                        if (std::isnan(ke[idx]) || std::isinf(ke[idx])) {
                            has_nan = true;
                            break;
                        }
                    }
                    if (has_nan) continue;

                    K_global_.add_element_matrix(dof_map, ke);
                }
            } else {
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
                        dof_map[i * 3 + 0] = elem_nodes[i] * dof_per_node_ + 0;
                        dof_map[i * 3 + 1] = elem_nodes[i] * dof_per_node_ + 1;
                        dof_map[i * 3 + 2] = elem_nodes[i] * dof_per_node_ + 2;
                    }

                    if (nodes_per_elem == 8) {
                        hex8.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                    } else if (nodes_per_elem == 20) {
                        hex20.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                    } else if (nodes_per_elem == 10) {
                        tet10.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                    } else if (nodes_per_elem == 4 && block.type == ElementType::Tet4) {
                        tet4.stiffness_matrix(elem_coords.data(), material_.E, material_.nu, ke.data());
                    }

                    bool has_nan = false;
                    for (size_t idx = 0; idx < ke.size(); ++idx) {
                        if (std::isnan(ke[idx]) || std::isinf(ke[idx])) {
                            has_nan = true;
                            break;
                        }
                    }
                    if (has_nan) continue;

                    K_global_.add_element_matrix(dof_map, ke);
                }
            }
        }

        // Add rotational penalty for solid-only nodes in 6-DOF mode
        if (dof_per_node_ == 6) {
            add_solid_rotational_penalty(shell_nodes);
        }
    }

    void transform_shell_stiffness_to_global(const Real* elem_coords,
                                              std::vector<Real>& ke) {
        fem::Shell4Element shell4_tmp;
        Real e1[3], e2[3], e3[3];
        shell4_tmp.local_coordinate_system(elem_coords, e1, e2, e3);

        constexpr int N = 24;
        std::vector<Real> T(N * N, 0.0);

        Real R[3][3] = {
            {e1[0], e2[0], e3[0]},
            {e1[1], e2[1], e3[1]},
            {e1[2], e2[2], e3[2]}
        };

        for (int node = 0; node < 4; ++node) {
            int base = node * 6;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    T[(base + i) * N + (base + j)] = R[i][j];
                    T[(base + 3 + i) * N + (base + 3 + j)] = R[i][j];
                }
            }
        }

        std::vector<Real> temp(N * N, 0.0);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                Real sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += ke[i * N + k] * T[j * N + k];
                }
                temp[i * N + j] = sum;
            }
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                Real sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += T[i * N + k] * temp[k * N + j];
                }
                ke[i * N + j] = sum;
            }
        }
    }

    void add_solid_rotational_penalty(const std::set<Index>& shell_nodes) {
        Real max_diag = 0.0;
        for (size_t i = 0; i < ndof_; ++i) {
            max_diag = std::max(max_diag, std::abs(K_global_.get(i, i)));
        }
        Real penalty = 1e-6 * max_diag;
        if (penalty < 1e-20) return;

        for (size_t node = 0; node < mesh_->num_nodes(); ++node) {
            if (shell_nodes.count(node) == 0) {
                for (int d = 3; d < 6; ++d) {
                    size_t dof = node * dof_per_node_ + d;
                    Real current = K_global_.get(dof, dof);
                    K_global_.set(dof, dof, current + penalty);
                }
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
            size_t dof_idx = bc.node_id * dof_per_node_ + bc.dof;
            Real current_diag = K_eff_.get(dof_idx, dof_idx);
            K_eff_.set(dof_idx, dof_idx, current_diag + penalty);
            F[dof_idx] = penalty * bc.value;
        }
    }

    const Mesh* mesh_ = nullptr;
    size_t ndof_ = 0;
    int dof_per_node_ = 3;
    Real shell_thickness_ = 0.01;
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
