#pragma once

/**
 * @file tet4.hpp
 * @brief 4-node tetrahedral element with linear shape functions
 *
 * Node numbering (right-hand rule):
 *        3
 *       /|\
 *      / | \
 *     /  |  \
 *    /   |   \
 *   /    0    \
 *  /    / \    \
 * 1 ----   ---- 2
 *
 * Natural coordinates: L1, L2, L3, L4 (volume/barycentric coordinates)
 * Constraint: L1 + L2 + L3 + L4 = 1
 * Node local coordinates:
 *   0: (1, 0, 0, 0) or alternatively use (ξ, η, ζ) with constraint
 *   1: (0, 1, 0, 0)
 *   2: (0, 0, 1, 0)
 *   3: (0, 0, 0, 1)
 */

#include <nexussim/core/core.hpp>
#include <nexussim/physics/element.hpp>
#include <array>
#include <cmath>

namespace nxs {
namespace fem {

// ============================================================================
// Tet4 Element - 4-node Tetrahedron
// ============================================================================

class Tet4Element : public physics::Element {
public:
    static constexpr int NUM_NODES = 4;
    static constexpr int NUM_DIMS = 3;
    static constexpr int NUM_STRESS_COMPONENTS = 6;  // Voigt notation
    static constexpr int NUM_DOF = NUM_NODES * NUM_DIMS;  // 12 DOFs

    Tet4Element() = default;
    ~Tet4Element() override = default;

    // ========================================================================
    // Element Properties
    // ========================================================================

    Properties properties() const override {
        return Properties{
            physics::ElementType::Tet4,
            physics::ElementTopology::Tetrahedron,
            NUM_NODES,
            1,  // 1-point integration at centroid
            NUM_DIMS,
            NUM_DIMS
        };
    }

    // ========================================================================
    // Shape Functions
    // ========================================================================

    /**
     * @brief Compute shape functions at natural coordinates
     * @param xi Natural coordinates [ξ, η, ζ] (note: L4 = 1-ξ-η-ζ)
     * @param N Output shape functions (size 4)
     *
     * Shape functions (volume coordinates):
     * N1 = L1 = ξ
     * N2 = L2 = η
     * N3 = L3 = ζ
     * N4 = L4 = 1 - ξ - η - ζ
     */
    KOKKOS_INLINE_FUNCTION
    void shape_functions(const Real xi[3], Real* N) const override;

    /**
     * @brief Compute shape function derivatives w.r.t. natural coordinates
     * @param xi Natural coordinates [ξ, η, ζ]
     * @param dN Output derivatives (4x3 matrix, row-major: dN/dξ, dN/dη, dN/dζ)
     */
    KOKKOS_INLINE_FUNCTION
    void shape_derivatives(const Real xi[3], Real* dN) const override;

    // ========================================================================
    // Gauss Quadrature
    // ========================================================================

    /**
     * @brief Get Gauss quadrature points and weights
     * @param points Output quadrature points (n_pts x 3)
     * @param weights Output weights (n_pts)
     */
    void gauss_quadrature(Real* points, Real* weights) const override;

    // ========================================================================
    // Jacobian and Coordinate Mapping
    // ========================================================================

    /**
     * @brief Compute Jacobian matrix and determinant
     * @param xi Natural coordinates [ξ, η, ζ]
     * @param coords Element nodal coordinates (4 nodes x 3 coords, flat array)
     * @param J Output Jacobian matrix (3x3, row-major)
     * @return Jacobian determinant
     */
    KOKKOS_INLINE_FUNCTION
    Real jacobian(const Real xi[3], const Real* coords, Real* J) const override;

    /**
     * @brief Compute inverse Jacobian matrix
     * @param J Jacobian matrix (3x3, row-major)
     * @param J_inv Output inverse Jacobian (3x3, row-major)
     * @param det_J Jacobian determinant
     */
    void inverse_jacobian(const Real* J, Real* J_inv, Real det_J) const;

    /**
     * @brief Compute shape function derivatives w.r.t. global coordinates
     * @param xi Natural coordinates
     * @param coords Element nodal coordinates
     * @param dNdx Output derivatives w.r.t. x,y,z (4x3 matrix, row-major)
     * @return Jacobian determinant
     */
    Real shape_derivatives_global(const Real xi[3], const Real* coords, Real* dNdx) const;

    // ========================================================================
    // B-Matrix (Strain-Displacement)
    // ========================================================================

    /**
     * @brief Compute strain-displacement matrix (B-matrix)
     * @param xi Natural coordinates
     * @param coords Element nodal coordinates
     * @param B Output B-matrix (6x12, Voigt notation, row-major)
     */
    KOKKOS_INLINE_FUNCTION
    void strain_displacement_matrix(const Real xi[3], const Real* coords, Real* B) const override;

    // ========================================================================
    // Element Matrices
    // ========================================================================

    /**
     * @brief Compute element mass matrix (lumped)
     * @param coords Element nodal coordinates
     * @param density Material density
     * @param M Output lumped mass vector (12 entries, 3 DOFs per node)
     */
    void lumped_mass_matrix(const Real* coords, Real density, Real* M) const;

    /**
     * @brief Compute element mass matrix (consistent)
     * @param coords Element nodal coordinates
     * @param density Material density
     * @param M Output mass matrix (12x12, row-major)
     */
    void mass_matrix(const Real* coords, Real density, Real* M) const override;

    /**
     * @brief Compute element stiffness matrix
     * @param coords Element nodal coordinates
     * @param E Young's modulus
     * @param nu Poisson's ratio
     * @param K Output stiffness matrix (12x12, row-major)
     */
    void stiffness_matrix(const Real* coords, Real E, Real nu, Real* K) const override;

    /**
     * @brief Compute elastic constitutive matrix (Voigt notation)
     * @param E Young's modulus
     * @param nu Poisson's ratio
     * @param C Output constitutive matrix (6x6, row-major)
     */
    void constitutive_matrix(Real E, Real nu, Real* C) const;

    // ========================================================================
    // Internal Force Computation
    // ========================================================================

    /**
     * @brief Compute element internal force vector
     * @param coords Element nodal coordinates
     * @param disp Element nodal displacements (12 entries)
     * @param stress Element stresses at integration points (6 components)
     * @param fint Output internal force vector (12 entries)
     */
    KOKKOS_INLINE_FUNCTION
    void internal_force(const Real* coords, const Real* disp,
                       const Real* stress, Real* fint) const override;

    // ========================================================================
    // Geometric Queries
    // ========================================================================

    /**
     * @brief Check if a point is inside the element
     * @param coords Element nodal coordinates
     * @param point Global coordinates of query point
     * @param xi Output natural coordinates (if inside)
     * @return True if point is inside element
     */
    bool contains_point(const Real* coords, const Real* point, Real* xi) const override;

    /**
     * @brief Compute element volume
     * @param coords Element nodal coordinates
     * @return Element volume
     */
    Real volume(const Real* coords) const override;

    /**
     * @brief Compute characteristic length (for time step estimation)
     * @param coords Element nodal coordinates
     * @return Characteristic length (minimum edge length)
     */
    Real characteristic_length(const Real* coords) const override;
};

} // namespace fem
} // namespace nxs
