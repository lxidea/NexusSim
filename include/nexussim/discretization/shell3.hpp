#pragma once

/**
 * @file shell3.hpp
 * @brief 3-node triangular shell element (membrane + bending)
 *
 * Node numbering (counterclockwise):
 *        2
 *       / \
 *      /   \
 *     0-----1
 *
 * Natural coordinates: ξ, η ∈ [0, 1], ξ + η ≤ 1 (area coordinates L1, L2, L3)
 * DOFs per node: 6 (ux, uy, uz, θx, θy, θz)
 *
 * Formulation: Combines:
 *   - CST (Constant Strain Triangle) for membrane behavior
 *   - DKT (Discrete Kirchhoff Triangle) for bending behavior
 *
 * References:
 *   - Batoz et al., 1980, "A study of three-node triangular plate bending elements"
 *   - Zienkiewicz & Taylor, "The Finite Element Method", Vol. 2
 */

#include <nexussim/core/core.hpp>
#include <nexussim/physics/element.hpp>
#include <array>
#include <cmath>

namespace nxs {
namespace fem {

// ============================================================================
// Shell3 Element - 3-node Triangular Shell (DKT + CST)
// ============================================================================

class Shell3Element : public physics::Element {
public:
    static constexpr int NUM_NODES = 3;
    static constexpr int NUM_DIMS = 3;  // 3D space
    static constexpr int DOF_PER_NODE = 6;  // 3 translations + 3 rotations
    static constexpr int NUM_DOF = NUM_NODES * DOF_PER_NODE;  // 18 DOFs
    static constexpr int NUM_STRESS_COMPONENTS = 6;

    Shell3Element() = default;
    ~Shell3Element() override = default;

    // ========================================================================
    // Element Properties
    // ========================================================================

    Properties properties() const override {
        return Properties{
            physics::ElementType::Shell3,
            physics::ElementTopology::Triangle,
            NUM_NODES,
            3,  // 3-point Gauss integration
            DOF_PER_NODE,
            NUM_DIMS
        };
    }

    // ========================================================================
    // Shape Functions (Area coordinates for membrane)
    // ========================================================================

    /**
     * @brief Compute shape functions at natural coordinates
     * @param xi Natural coordinates [L1, L2] (area coords, L3 = 1 - L1 - L2)
     * @param N Output shape functions (size 3)
     *
     * For triangles: N1 = L1, N2 = L2, N3 = L3 = 1 - L1 - L2
     * where L1, L2, L3 are area coordinates (barycentric)
     */
    KOKKOS_INLINE_FUNCTION
    void shape_functions(const Real xi[3], Real* N) const override {
        // Area coordinates interpretation: xi[0] = L1, xi[1] = L2
        const Real L1 = xi[0];
        const Real L2 = xi[1];
        const Real L3 = 1.0 - L1 - L2;

        N[0] = L1;
        N[1] = L2;
        N[2] = L3;
    }

    /**
     * @brief Compute shape function derivatives w.r.t. natural coordinates
     * @param xi Natural coordinates [L1, L2]
     * @param dN Output derivatives (3x2 matrix, row-major)
     *
     * dN/dL1 = [1, 0, -1]^T
     * dN/dL2 = [0, 1, -1]^T
     */
    KOKKOS_INLINE_FUNCTION
    void shape_derivatives(const Real xi[3], Real* dN) const override {
        (void)xi;  // Derivatives are constant for linear triangle

        // dN/dL1 (column 0)
        dN[0*2 + 0] = 1.0;   // dN1/dL1
        dN[1*2 + 0] = 0.0;   // dN2/dL1
        dN[2*2 + 0] = -1.0;  // dN3/dL1

        // dN/dL2 (column 1)
        dN[0*2 + 1] = 0.0;   // dN1/dL2
        dN[1*2 + 1] = 1.0;   // dN2/dL2
        dN[2*2 + 1] = -1.0;  // dN3/dL2
    }

    // ========================================================================
    // Gauss Quadrature (3-point for triangles)
    // ========================================================================

    void gauss_quadrature(Real* points, Real* weights) const override {
        // 3-point quadrature for triangles (degree 2)
        // Points in area coordinates (L1, L2), L3 = 1 - L1 - L2
        // Weight = area × 1/3 for each point

        // Point 0: (2/3, 1/6)
        points[0*3 + 0] = 2.0/3.0;  // L1
        points[0*3 + 1] = 1.0/6.0;  // L2
        points[0*3 + 2] = 0.0;      // unused
        weights[0] = 1.0/3.0;  // × 0.5 (area factor) applied in integration

        // Point 1: (1/6, 2/3)
        points[1*3 + 0] = 1.0/6.0;
        points[1*3 + 1] = 2.0/3.0;
        points[1*3 + 2] = 0.0;
        weights[1] = 1.0/3.0;

        // Point 2: (1/6, 1/6)
        points[2*3 + 0] = 1.0/6.0;
        points[2*3 + 1] = 1.0/6.0;
        points[2*3 + 2] = 0.0;
        weights[2] = 1.0/3.0;
    }

    // ========================================================================
    // Jacobian and Coordinate Mapping
    // ========================================================================

    /**
     * @brief Compute Jacobian matrix for triangular element
     * @param xi Natural coordinates (unused - constant Jacobian)
     * @param coords Element nodal coordinates (3 nodes x 3 coords)
     * @param J Output Jacobian matrix (2x2 for in-plane mapping)
     * @return Jacobian determinant (= 2 × area)
     */
    KOKKOS_INLINE_FUNCTION
    Real jacobian(const Real xi[3], const Real* coords, Real* J) const override {
        (void)xi;  // Jacobian is constant for linear triangle

        // Node coordinates
        const Real x1 = coords[0*3 + 0], y1 = coords[0*3 + 1], z1 = coords[0*3 + 2];
        const Real x2 = coords[1*3 + 0], y2 = coords[1*3 + 1], z2 = coords[1*3 + 2];
        const Real x3 = coords[2*3 + 0], y3 = coords[2*3 + 1], z3 = coords[2*3 + 2];

        // Edge vectors
        const Real e1[3] = {x2 - x1, y2 - y1, z2 - z1};
        const Real e2[3] = {x3 - x1, y3 - y1, z3 - z1};

        // In 3D, we compute the Jacobian in the local coordinate system
        // J = [dx/dL1, dx/dL2; dy/dL1, dy/dL2] in local coords
        // For a flat triangle in 3D, this is the 2D mapping in the plane

        // Cross product gives normal × 2A
        const Real n[3] = {
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0]
        };

        // Area = 0.5 * |n|
        const Real area2 = Kokkos::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);

        // For 3D triangles, we use the projected Jacobian
        // J11 = |e1|, J12 = e1·e2/|e1|, J21 = 0, J22 = |n|/|e1|
        const Real e1_len = Kokkos::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
        const Real e1_dot_e2 = e1[0]*e2[0] + e1[1]*e2[1] + e1[2]*e2[2];

        if (e1_len > 1.0e-20) {
            J[0] = e1_len;
            J[1] = e1_dot_e2 / e1_len;
            J[2] = 0.0;
            J[3] = area2 / e1_len;
        } else {
            J[0] = J[3] = 1.0;
            J[1] = J[2] = 0.0;
        }

        return area2;  // = 2 × area
    }

    // ========================================================================
    // B-Matrix (Strain-Displacement for membrane CST)
    // ========================================================================

    /**
     * @brief Compute strain-displacement matrix (membrane only)
     * @param xi Natural coordinates (unused for CST)
     * @param coords Element nodal coordinates
     * @param B Output B-matrix [3 strain × 9 DOF] for membrane
     *
     * For CST: strains are constant over element
     * ε = [εxx, εyy, γxy]^T = B × u
     * where u = [u1, v1, u2, v2, u3, v3]^T (in-plane displacements)
     */
    KOKKOS_INLINE_FUNCTION
    void strain_displacement_matrix(const Real xi[3], const Real* coords, Real* B) const override {
        (void)xi;  // CST has constant strains

        // Compute in local coordinate system
        // Get area and edge lengths
        Real e1[3], e2[3], normal[3];
        compute_local_system(coords, e1, e2, normal);

        // Project nodes to local 2D system
        Real x_local[3], y_local[3];
        for (int i = 0; i < 3; ++i) {
            const Real dx = coords[i*3 + 0] - coords[0];
            const Real dy = coords[i*3 + 1] - coords[1];
            const Real dz = coords[i*3 + 2] - coords[2];
            x_local[i] = dx*e1[0] + dy*e1[1] + dz*e1[2];
            y_local[i] = dx*e2[0] + dy*e2[1] + dz*e2[2];
        }

        // 2A = (x2-x1)(y3-y1) - (x3-x1)(y2-y1)
        const Real area2 = (x_local[1] - x_local[0]) * (y_local[2] - y_local[0])
                         - (x_local[2] - x_local[0]) * (y_local[1] - y_local[0]);
        const Real area2_inv = 1.0 / area2;

        // CST B-matrix: B = (1/2A) × [b1 0 b2 0 b3 0; 0 c1 0 c2 0 c3; c1 b1 c2 b2 c3 b3]
        // where bi = yj - yk, ci = xk - xj (cyclic permutation)
        const Real b[3] = {
            y_local[1] - y_local[2],
            y_local[2] - y_local[0],
            y_local[0] - y_local[1]
        };
        const Real c[3] = {
            x_local[2] - x_local[1],
            x_local[0] - x_local[2],
            x_local[1] - x_local[0]
        };

        // B-matrix is [3 × 6] for in-plane DOFs only (u, v at each node)
        // For full shell with 6 DOFs per node, we need [3 × 18]
        // Initialize to zero
        for (int i = 0; i < 3 * 18; ++i) B[i] = 0.0;

        // Fill membrane B-matrix (ux and uy DOFs only)
        for (int n = 0; n < 3; ++n) {
            // εxx = ∂u/∂x
            B[0*18 + n*6 + 0] = b[n] * area2_inv;  // dux/dx

            // εyy = ∂v/∂y
            B[1*18 + n*6 + 1] = c[n] * area2_inv;  // duy/dy

            // γxy = ∂u/∂y + ∂v/∂x
            B[2*18 + n*6 + 0] = c[n] * area2_inv;  // dux/dy
            B[2*18 + n*6 + 1] = b[n] * area2_inv;  // duy/dx
        }
    }

    // ========================================================================
    // Element Matrices
    // ========================================================================

    /**
     * @brief Set shell thickness
     * @param t Shell thickness
     */
    void set_thickness(Real t) { thickness_ = t; }

    Real thickness() const { return thickness_; }

    /**
     * @brief Compute lumped mass matrix
     * @param coords Element nodal coordinates
     * @param density Material density
     * @param M Output mass (18 diagonal values)
     */
    void lumped_mass_matrix(const Real* coords, Real density, Real* M) const {
        // Total mass = density × area × thickness
        Real area = this->area(coords);
        Real total_mass = density * area * thickness_;

        // Distribute equally to all nodes and DOFs
        Real mass_per_node = total_mass / 3.0;
        Real mass_per_dof = mass_per_node;

        // Translational DOFs get full mass
        // Rotational DOFs get rotational inertia ≈ mass × t²/12
        Real rot_inertia = mass_per_node * thickness_ * thickness_ / 12.0;

        for (int n = 0; n < 3; ++n) {
            M[n*6 + 0] = mass_per_dof;  // ux
            M[n*6 + 1] = mass_per_dof;  // uy
            M[n*6 + 2] = mass_per_dof;  // uz
            M[n*6 + 3] = rot_inertia;   // θx
            M[n*6 + 4] = rot_inertia;   // θy
            M[n*6 + 5] = rot_inertia;   // θz (drilling)
        }
    }

    void mass_matrix(const Real* coords, Real density, Real* M) const override {
        // Use lumped mass for explicit dynamics
        Real M_lumped[18];
        lumped_mass_matrix(coords, density, M_lumped);

        // Convert to diagonal matrix (18×18)
        for (int i = 0; i < 18*18; ++i) M[i] = 0.0;
        for (int i = 0; i < 18; ++i) {
            M[i*18 + i] = M_lumped[i];
        }
    }

    void stiffness_matrix(const Real* coords, Real E, Real nu, Real* K) const override {
        // Combine membrane and bending stiffness
        // For simplicity, use membrane stiffness (CST) + DKT bending

        // Initialize to zero
        for (int i = 0; i < 18*18; ++i) K[i] = 0.0;

        // Get area and local coordinates
        Real e1[3], e2[3], normal[3];
        compute_local_system(coords, e1, e2, normal);
        Real area = this->area(coords);

        // Project nodes to local 2D
        Real x_local[3], y_local[3];
        for (int i = 0; i < 3; ++i) {
            const Real dx = coords[i*3 + 0] - coords[0];
            const Real dy = coords[i*3 + 1] - coords[1];
            const Real dz = coords[i*3 + 2] - coords[2];
            x_local[i] = dx*e1[0] + dy*e1[1] + dz*e1[2];
            y_local[i] = dx*e2[0] + dy*e2[1] + dz*e2[2];
        }

        // ================================================================
        // Membrane stiffness (CST)
        // ================================================================

        // Plane stress constitutive matrix
        const Real t = thickness_;
        const Real D_factor = E * t / (1.0 - nu*nu);
        Real D_mem[9] = {
            D_factor,         D_factor * nu,    0.0,
            D_factor * nu,    D_factor,         0.0,
            0.0,              0.0,              D_factor * (1.0 - nu) / 2.0
        };

        // B-matrix for CST (in local coords)
        const Real area2 = 2.0 * area;
        const Real area2_inv = 1.0 / area2;

        Real b[3] = {
            y_local[1] - y_local[2],
            y_local[2] - y_local[0],
            y_local[0] - y_local[1]
        };
        Real c[3] = {
            x_local[2] - x_local[1],
            x_local[0] - x_local[2],
            x_local[1] - x_local[0]
        };

        // K_mem = B^T × D × B × A (for membrane DOFs only: ux, uy at each node)
        // Then assemble into full 18×18 matrix
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                // K_ij contribution (2×2 submatrix for nodes i and j)
                // K = area × (bi*bj*D11 + ci*cj*D33, bi*cj*D12 + ci*bj*D33)
                //             (ci*bj*D21 + bi*cj*D33, ci*cj*D22 + bi*bj*D33)
                Real Kxx = area * area2_inv * area2_inv *
                          (b[i]*b[j]*D_mem[0] + c[i]*c[j]*D_mem[8]);
                Real Kxy = area * area2_inv * area2_inv *
                          (b[i]*c[j]*D_mem[1] + c[i]*b[j]*D_mem[8]);
                Real Kyx = area * area2_inv * area2_inv *
                          (c[i]*b[j]*D_mem[3] + b[i]*c[j]*D_mem[8]);
                Real Kyy = area * area2_inv * area2_inv *
                          (c[i]*c[j]*D_mem[4] + b[i]*b[j]*D_mem[8]);

                // Assemble into global matrix (DOFs 0,1 are ux, uy)
                K[(i*6 + 0)*18 + (j*6 + 0)] += Kxx;
                K[(i*6 + 0)*18 + (j*6 + 1)] += Kxy;
                K[(i*6 + 1)*18 + (j*6 + 0)] += Kyx;
                K[(i*6 + 1)*18 + (j*6 + 1)] += Kyy;
            }
        }

        // ================================================================
        // Bending stiffness (simplified DKT)
        // ================================================================

        // Bending rigidity
        const Real D_b = E * t * t * t / (12.0 * (1.0 - nu*nu));

        // For DKT, the bending stiffness couples uz with θx, θy
        // Here we use a simplified approach with averaged curvature
        // Full DKT implementation would be more complex

        // Simplified: Add rotational stiffness proportional to bending rigidity
        const Real k_rot = D_b / (area);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Real coupling = (i == j) ? 1.0 : -0.5;

                // uz-uz coupling through bending
                K[(i*6 + 2)*18 + (j*6 + 2)] += k_rot * coupling;

                // θx-θx coupling
                K[(i*6 + 3)*18 + (j*6 + 3)] += k_rot * area * coupling;

                // θy-θy coupling
                K[(i*6 + 4)*18 + (j*6 + 4)] += k_rot * area * coupling;
            }
        }

        // Add small drilling stiffness to avoid singularity
        const Real k_drill = 1.0e-3 * D_mem[0] * area;
        for (int i = 0; i < 3; ++i) {
            K[(i*6 + 5)*18 + (i*6 + 5)] += k_drill;
        }
    }

    // ========================================================================
    // Internal Force Computation
    // ========================================================================

    KOKKOS_INLINE_FUNCTION
    void internal_force(const Real* coords, const Real* disp,
                       const Real* stress, Real* fint) const override {
        // f_int = B^T × σ × A × t
        // For CST, B is constant, so this is straightforward

        Real area = this->area(coords);

        // Get local coordinate system and project coords
        Real e1[3], e2[3], normal[3];
        compute_local_system(coords, e1, e2, normal);

        Real x_local[3], y_local[3];
        for (int i = 0; i < 3; ++i) {
            const Real dx = coords[i*3 + 0] - coords[0];
            const Real dy = coords[i*3 + 1] - coords[1];
            const Real dz = coords[i*3 + 2] - coords[2];
            x_local[i] = dx*e1[0] + dy*e1[1] + dz*e1[2];
            y_local[i] = dx*e2[0] + dy*e2[1] + dz*e2[2];
        }

        const Real area2 = 2.0 * area;
        const Real area2_inv = 1.0 / area2;

        Real b[3] = {
            y_local[1] - y_local[2],
            y_local[2] - y_local[0],
            y_local[0] - y_local[1]
        };
        Real c[3] = {
            x_local[2] - x_local[1],
            x_local[0] - x_local[2],
            x_local[1] - x_local[0]
        };

        // Membrane stresses: σxx, σyy, τxy
        const Real sigma_xx = stress[0];
        const Real sigma_yy = stress[1];
        const Real tau_xy = stress[3];  // Voigt ordering

        // f = B^T × σ × A × t
        const Real factor = area * thickness_;

        for (int n = 0; n < 3; ++n) {
            // Local forces
            Real fx_local = factor * area2_inv * (b[n]*sigma_xx + c[n]*tau_xy);
            Real fy_local = factor * area2_inv * (c[n]*sigma_yy + b[n]*tau_xy);

            // Transform to global coordinates
            fint[n*6 + 0] = fx_local * e1[0] + fy_local * e2[0];
            fint[n*6 + 1] = fx_local * e1[1] + fy_local * e2[1];
            fint[n*6 + 2] = fx_local * e1[2] + fy_local * e2[2];

            // Rotational DOFs (simplified)
            fint[n*6 + 3] = 0.0;
            fint[n*6 + 4] = 0.0;
            fint[n*6 + 5] = 0.0;
        }
    }

    // ========================================================================
    // Geometric Queries
    // ========================================================================

    bool contains_point(const Real* coords, const Real* point, Real* xi) const override {
        // Use barycentric coordinates
        Real area_total = this->area(coords);
        if (area_total < 1.0e-20) return false;

        // Compute sub-triangle areas
        Real coords_temp[9];

        // Area of triangle (point, node1, node2)
        for (int d = 0; d < 3; ++d) {
            coords_temp[0*3 + d] = point[d];
            coords_temp[1*3 + d] = coords[1*3 + d];
            coords_temp[2*3 + d] = coords[2*3 + d];
        }
        Real area0 = compute_area(coords_temp);

        // Area of triangle (node0, point, node2)
        for (int d = 0; d < 3; ++d) {
            coords_temp[0*3 + d] = coords[0*3 + d];
            coords_temp[1*3 + d] = point[d];
            coords_temp[2*3 + d] = coords[2*3 + d];
        }
        Real area1 = compute_area(coords_temp);

        // Area of triangle (node0, node1, point)
        for (int d = 0; d < 3; ++d) {
            coords_temp[0*3 + d] = coords[0*3 + d];
            coords_temp[1*3 + d] = coords[1*3 + d];
            coords_temp[2*3 + d] = point[d];
        }
        Real area2 = compute_area(coords_temp);

        // Barycentric coordinates
        xi[0] = area0 / area_total;  // L1
        xi[1] = area1 / area_total;  // L2
        xi[2] = area2 / area_total;  // L3

        // Check if inside (with tolerance)
        const Real tol = 1.0e-6;
        return (xi[0] >= -tol && xi[1] >= -tol && xi[2] >= -tol &&
                xi[0] <= 1.0 + tol && xi[1] <= 1.0 + tol && xi[2] <= 1.0 + tol);
    }

    Real volume(const Real* coords) const override {
        // For shell elements, return area × thickness
        return area(coords) * thickness_;
    }

    Real characteristic_length(const Real* coords) const override {
        // Minimum edge length
        Real min_len = 1.0e30;

        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            Real dx = coords[j*3 + 0] - coords[i*3 + 0];
            Real dy = coords[j*3 + 1] - coords[i*3 + 1];
            Real dz = coords[j*3 + 2] - coords[i*3 + 2];
            Real len = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
            min_len = Kokkos::fmin(min_len, len);
        }

        return min_len;
    }

    /**
     * @brief Compute triangle area
     */
    Real area(const Real* coords) const {
        return compute_area(coords);
    }

    // ========================================================================
    // Shell-Specific Methods
    // ========================================================================

    /**
     * @brief Compute local coordinate system
     * @param coords Element nodal coordinates
     * @param e1 Output: local x-axis (along edge 0-1)
     * @param e2 Output: local y-axis (in plane, perpendicular to e1)
     * @param e3 Output: local z-axis (normal)
     */
    KOKKOS_INLINE_FUNCTION
    void local_coordinate_system(const Real* coords, Real* e1, Real* e2, Real* e3) const {
        compute_local_system(coords, e1, e2, e3);
    }

private:
    Real thickness_ = 0.01;  // Default 10mm thickness

    KOKKOS_INLINE_FUNCTION
    static Real compute_area(const Real* coords) {
        // Area = 0.5 × |e1 × e2|
        const Real e1[3] = {
            coords[1*3 + 0] - coords[0*3 + 0],
            coords[1*3 + 1] - coords[0*3 + 1],
            coords[1*3 + 2] - coords[0*3 + 2]
        };
        const Real e2[3] = {
            coords[2*3 + 0] - coords[0*3 + 0],
            coords[2*3 + 1] - coords[0*3 + 1],
            coords[2*3 + 2] - coords[0*3 + 2]
        };

        const Real n[3] = {
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0]
        };

        return 0.5 * Kokkos::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    }

    KOKKOS_INLINE_FUNCTION
    static void compute_local_system(const Real* coords, Real* e1, Real* e2, Real* e3) {
        // e1 along edge 0-1
        e1[0] = coords[1*3 + 0] - coords[0*3 + 0];
        e1[1] = coords[1*3 + 1] - coords[0*3 + 1];
        e1[2] = coords[1*3 + 2] - coords[0*3 + 2];

        Real e1_len = Kokkos::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
        if (e1_len > 1.0e-20) {
            e1[0] /= e1_len; e1[1] /= e1_len; e1[2] /= e1_len;
        }

        // Temporary vector along edge 0-2
        Real v[3] = {
            coords[2*3 + 0] - coords[0*3 + 0],
            coords[2*3 + 1] - coords[0*3 + 1],
            coords[2*3 + 2] - coords[0*3 + 2]
        };

        // e3 = e1 × v (normal)
        e3[0] = e1[1]*v[2] - e1[2]*v[1];
        e3[1] = e1[2]*v[0] - e1[0]*v[2];
        e3[2] = e1[0]*v[1] - e1[1]*v[0];

        Real e3_len = Kokkos::sqrt(e3[0]*e3[0] + e3[1]*e3[1] + e3[2]*e3[2]);
        if (e3_len > 1.0e-20) {
            e3[0] /= e3_len; e3[1] /= e3_len; e3[2] /= e3_len;
        }

        // e2 = e3 × e1
        e2[0] = e3[1]*e1[2] - e3[2]*e1[1];
        e2[1] = e3[2]*e1[0] - e3[0]*e1[2];
        e2[2] = e3[0]*e1[1] - e3[1]*e1[0];
    }
};

} // namespace fem
} // namespace nxs
