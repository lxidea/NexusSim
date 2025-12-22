#pragma once

/**
 * @file hex8.hpp
 * @brief 8-node hexahedral element with trilinear shape functions
 *
 * Node numbering (right-hand rule):
 *        7--------6
 *       /|       /|
 *      / |      / |
 *     4--------5  |
 *     |  3-----|--2
 *     | /      | /
 *     |/       |/
 *     0--------1
 *
 * Natural coordinates: ξ, η, ζ ∈ [-1, 1]
 * Node local coordinates:
 *   0: (-1,-1,-1)  1: (+1,-1,-1)  2: (+1,+1,-1)  3: (-1,+1,-1)
 *   4: (-1,-1,+1)  5: (+1,-1,+1)  6: (+1,+1,+1)  7: (-1,+1,+1)
 */

#include <nexussim/core/core.hpp>
#include <nexussim/physics/element.hpp>
#include <array>
#include <cmath>

namespace nxs {
namespace fem {

// ============================================================================
// Hex8 Element - 8-node Hexahedron
// ============================================================================

class Hex8Element : public physics::Element {
public:
    static constexpr int NUM_NODES = 8;
    static constexpr int NUM_DIMS = 3;
    static constexpr int NUM_STRESS_COMPONENTS = 6;  // Voigt notation
    static constexpr int NUM_DOF = NUM_NODES * NUM_DIMS;  // 24 DOFs

    // Node natural coordinates (ξ, η, ζ)
    static constexpr Real NODE_COORDS[NUM_NODES][NUM_DIMS] = {
        {-1.0, -1.0, -1.0},  // Node 0
        {+1.0, -1.0, -1.0},  // Node 1
        {+1.0, +1.0, -1.0},  // Node 2
        {-1.0, +1.0, -1.0},  // Node 3
        {-1.0, -1.0, +1.0},  // Node 4
        {+1.0, -1.0, +1.0},  // Node 5
        {+1.0, +1.0, +1.0},  // Node 6
        {-1.0, +1.0, +1.0}   // Node 7
    };

    Hex8Element() = default;
    ~Hex8Element() override = default;

    // ========================================================================
    // Element Properties
    // ========================================================================

    Properties properties() const override {
        return Properties{
            physics::ElementType::Hex8,
            physics::ElementTopology::Hexahedron,
            NUM_NODES,
            1,  // Default to 1-point integration for explicit dynamics
            NUM_DIMS,
            NUM_DIMS
        };
    }

    // ========================================================================
    // Shape Functions
    // ========================================================================

    /**
     * @brief Compute shape functions at natural coordinates
     * @param xi Natural coordinates [ξ, η, ζ]
     * @param N Output shape functions (size 8)
     *
     * Shape function for node i:
     * N_i = (1 + ξ_i*ξ)(1 + η_i*η)(1 + ζ_i*ζ) / 8
     */
    KOKKOS_INLINE_FUNCTION
    void shape_functions(const Real xi[3], Real* N) const override;

    /**
     * @brief Compute shape function derivatives w.r.t. natural coordinates
     * @param xi Natural coordinates [ξ, η, ζ]
     * @param dN Output derivatives (8x3 matrix, row-major: dN/dξ, dN/dη, dN/dζ)
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
     * @param coords Element nodal coordinates (8 nodes x 3 coords, flat array)
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
     * @param dNdx Output derivatives w.r.t. x,y,z (8x3 matrix, row-major)
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
     * @param B Output B-matrix (6x24, Voigt notation, row-major)
     *
     * B-matrix relates nodal displacements to strains:
     * {ε} = [B]{u}
     * where {ε} = [εxx, εyy, εzz, γxy, γyz, γxz]^T (Voigt notation)
     *       {u} = [u0x, u0y, u0z, u1x, u1y, u1z, ..., u7x, u7y, u7z]^T
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
     * @param M Output lumped mass vector (24 entries, 3 DOFs per node)
     */
    void lumped_mass_matrix(const Real* coords, Real density, Real* M) const;

    /**
     * @brief Compute element mass matrix (consistent)
     * @param coords Element nodal coordinates
     * @param density Material density
     * @param M Output mass matrix (24x24, row-major)
     */
    void mass_matrix(const Real* coords, Real density, Real* M) const override;

    /**
     * @brief Compute element stiffness matrix
     * @param coords Element nodal coordinates
     * @param E Young's modulus
     * @param nu Poisson's ratio
     * @param K Output stiffness matrix (24x24, row-major)
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
    // Strain and Stress Computation
    // ========================================================================

    /**
     * @brief Compute element strain from nodal displacements
     * @param coords Element nodal coordinates
     * @param disp Element nodal displacements (24 entries)
     * @param strain Output strain vector (6 components, Voigt notation)
     *
     * Computes strain using 1-point Gauss integration at element center:
     * {ε} = [B]{u} where {ε} = [εxx, εyy, εzz, γxy, γyz, γxz]^T
     */
    KOKKOS_INLINE_FUNCTION
    void compute_strain(const Real* coords, const Real* disp, Real* strain) const;

    /**
     * @brief Compute stress from strain (linear elastic material)
     * @param strain Strain vector (6 components, Voigt notation)
     * @param E Young's modulus
     * @param nu Poisson's ratio
     * @param stress Output stress vector (6 components, Voigt notation)
     *
     * Computes stress using linear elastic constitutive model:
     * {σ} = [C]{ε} where {σ} = [σxx, σyy, σzz, τxy, τyz, τxz]^T
     */
    KOKKOS_INLINE_FUNCTION
    void compute_stress(const Real* strain, Real E, Real nu, Real* stress) const;

    // ========================================================================
    // Internal Force Computation
    // ========================================================================

    /**
     * @brief Compute element internal force vector
     * @param coords Element nodal coordinates
     * @param disp Element nodal displacements (24 entries)
     * @param stress Element stresses at integration points (6 components per point)
     * @param fint Output internal force vector (24 entries)
     */
    KOKKOS_INLINE_FUNCTION
    void internal_force(const Real* coords, const Real* disp,
                       const Real* stress, Real* fint) const override;

    // ========================================================================
    // Hourglass Control
    // ========================================================================

    /**
     * @brief Compute hourglass resistance forces (Flanagan-Belytschko)
     * @param coords Element nodal coordinates
     * @param disp Element nodal displacements (24 entries)
     * @param hourglass_stiffness Hourglass stiffness parameter (typically 0.01-0.1 * bulk_modulus)
     * @param fhg Output hourglass force vector (24 entries)
     *
     * Computes stabilization forces to prevent spurious zero-energy modes
     * that arise with 1-point reduced integration.
     */
    void hourglass_forces(const Real* coords, const Real* disp,
                         Real hourglass_stiffness, Real* fhg) const;

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

private:
    // Helper functions
    void compute_gauss_points_1pt(Real* points, Real* weights) const;
    void compute_gauss_points_8pt(Real* points, Real* weights) const;
};

} // namespace fem
} // namespace nxs
