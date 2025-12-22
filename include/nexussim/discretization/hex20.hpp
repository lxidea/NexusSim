#pragma once

/**
 * @file hex20.hpp
 * @brief 20-node hexahedral element with quadratic shape functions
 *
 * Node numbering (right-hand rule):
 * Corner nodes (0-7) same as Hex8:
 *        7--------18-------6
 *       /|                /|
 *     19 |              17 |
 *     /  15             /  14
 *    4--------16-------5   |
 *    |   |             |   |
 *    |   3--------10---|---2
 *   12  /             13  /
 *    | 11              | 9
 *    |/                |/
 *    0--------8--------1
 *
 * Node layout:
 *   Corner nodes:  0-7 (at ξ,η,ζ = ±1)
 *   Edge nodes:    8-19 (at mid-edges, one coordinate = 0)
 *     Bottom edges: 8(0-1), 9(1-2), 10(2-3), 11(3-0)
 *     Top edges:    16(4-5), 17(5-6), 18(6-7), 19(7-4)
 *     Vertical:     12(0-4), 13(1-5), 14(2-6), 15(3-7)
 *
 * Natural coordinates: ξ, η, ζ ∈ [-1, 1]
 */

#include <nexussim/core/core.hpp>
#include <nexussim/physics/element.hpp>
#include <array>
#include <cmath>

namespace nxs {
namespace fem {

// ============================================================================
// Hex20 Element - 20-node Hexahedron (Serendipity)
// ============================================================================

class Hex20Element : public physics::Element {
public:
    static constexpr int NUM_NODES = 20;
    static constexpr int NUM_DIMS = 3;
    static constexpr int NUM_STRESS_COMPONENTS = 6;  // Voigt notation
    static constexpr int NUM_DOF = NUM_NODES * NUM_DIMS;  // 60 DOFs

    // Node natural coordinates (ξ, η, ζ)
    static constexpr Real NODE_COORDS[NUM_NODES][NUM_DIMS] = {
        // Corner nodes (0-7)
        {-1.0, -1.0, -1.0},  // Node 0
        {+1.0, -1.0, -1.0},  // Node 1
        {+1.0, +1.0, -1.0},  // Node 2
        {-1.0, +1.0, -1.0},  // Node 3
        {-1.0, -1.0, +1.0},  // Node 4
        {+1.0, -1.0, +1.0},  // Node 5
        {+1.0, +1.0, +1.0},  // Node 6
        {-1.0, +1.0, +1.0},  // Node 7
        // Mid-edge nodes (8-19)
        // Bottom face edges (z=-1)
        { 0.0, -1.0, -1.0},  // Node 8  (edge 0-1)
        {+1.0,  0.0, -1.0},  // Node 9  (edge 1-2)
        { 0.0, +1.0, -1.0},  // Node 10 (edge 2-3)
        {-1.0,  0.0, -1.0},  // Node 11 (edge 3-0)
        // Vertical edges
        {-1.0, -1.0,  0.0},  // Node 12 (edge 0-4)
        {+1.0, -1.0,  0.0},  // Node 13 (edge 1-5)
        {+1.0, +1.0,  0.0},  // Node 14 (edge 2-6)
        {-1.0, +1.0,  0.0},  // Node 15 (edge 3-7)
        // Top face edges (z=+1)
        { 0.0, -1.0, +1.0},  // Node 16 (edge 4-5)
        {+1.0,  0.0, +1.0},  // Node 17 (edge 5-6)
        { 0.0, +1.0, +1.0},  // Node 18 (edge 6-7)
        {-1.0,  0.0, +1.0}   // Node 19 (edge 7-4)
    };

    Hex20Element() = default;
    ~Hex20Element() override = default;

    // ========================================================================
    // Element Properties
    // ========================================================================

    Properties properties() const override {
        return Properties{
            physics::ElementType::Hex20,
            physics::ElementTopology::Hexahedron,
            NUM_NODES,
            27,  // 3x3x3 Gauss integration for quadratic elements
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
     * @param N Output shape functions (size 20)
     *
     * Serendipity shape functions:
     * Corner nodes (i=0-7):
     *   N_i = (1 + ξ_iξ)(1 + η_iη)(1 + ζ_iζ)(ξ_iξ + η_iη + ζ_iζ - 2)/8
     * Edge nodes (i=8-19):
     *   If on ξ=0 edge: N_i = (1 - ξ²)(1 + η_iη)(1 + ζ_iζ)/4
     *   If on η=0 edge: N_i = (1 + ξ_iξ)(1 - η²)(1 + ζ_iζ)/4
     *   If on ζ=0 edge: N_i = (1 + ξ_iξ)(1 + η_iη)(1 - ζ²)/4
     */
    KOKKOS_INLINE_FUNCTION
    void shape_functions(const Real xi[3], Real* N) const override;

    /**
     * @brief Compute shape function derivatives w.r.t. natural coordinates
     * @param xi Natural coordinates [ξ, η, ζ]
     * @param dN Output derivatives (20x3 matrix, row-major: dN/dξ, dN/dη, dN/dζ)
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
     * @param coords Element nodal coordinates (20 nodes x 3 coords, flat array)
     * @param J Output Jacobian matrix (3x3, row-major)
     * @return Jacobian determinant
     */
    KOKKOS_INLINE_FUNCTION
    Real jacobian(const Real xi[3], const Real* coords, Real* J) const override;

    /**
     * @brief Compute shape function derivatives w.r.t. global coordinates
     * @param xi Natural coordinates
     * @param coords Element nodal coordinates
     * @param dNdx Output derivatives w.r.t. x,y,z (20x3 matrix, row-major)
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
     * @param B Output B-matrix (6x60, Voigt notation, row-major)
     */
    KOKKOS_INLINE_FUNCTION
    void strain_displacement_matrix(const Real xi[3], const Real* coords, Real* B) const override;

    // ========================================================================
    // Element Matrices
    // ========================================================================

    /**
     * @brief Compute element mass matrix (consistent)
     * @param coords Element nodal coordinates
     * @param density Material density
     * @param M Output mass matrix (60x60, row-major)
     */
    void mass_matrix(const Real* coords, Real density, Real* M) const override;

    /**
     * @brief Compute element lumped mass using HRZ method
     * @param coords Element nodal coordinates
     * @param density Material density
     * @param M_lumped Output lumped mass vector (20 entries, one per node)
     *
     * Uses Hinton-Rock-Zienkiewicz (HRZ) lumping which evaluates mass
     * at nodal positions to avoid negative masses in quadratic elements.
     */
    void lumped_mass_hrz(const Real* coords, Real density, Real* M_lumped) const;

    /**
     * @brief Compute element stiffness matrix
     * @param coords Element nodal coordinates
     * @param E Young's modulus
     * @param nu Poisson's ratio
     * @param K Output stiffness matrix (60x60, row-major)
     */
    void stiffness_matrix(const Real* coords, Real E, Real nu, Real* K) const override;

    // ========================================================================
    // Strain and Stress Computation
    // ========================================================================

    /**
     * @brief Compute element strain from nodal displacements
     * @param coords Element nodal coordinates
     * @param disp Element nodal displacements (60 entries)
     * @param strain Output strain vector (6 components, Voigt notation)
     */
    KOKKOS_INLINE_FUNCTION
    void compute_strain(const Real* coords, const Real* disp, Real* strain) const;

    /**
     * @brief Compute stress from strain (linear elastic material)
     * @param strain Strain vector (6 components, Voigt notation)
     * @param E Young's modulus
     * @param nu Poisson's ratio
     * @param stress Output stress vector (6 components, Voigt notation)
     */
    KOKKOS_INLINE_FUNCTION
    void compute_stress(const Real* strain, Real E, Real nu, Real* stress) const;

    // ========================================================================
    // Internal Force Computation
    // ========================================================================

    /**
     * @brief Compute element internal force vector
     * @param coords Element nodal coordinates
     * @param disp Element nodal displacements (60 entries)
     * @param stress Element stresses at integration points
     * @param fint Output internal force vector (60 entries)
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

private:
    // Helper functions
    void compute_gauss_points_27pt(Real* points, Real* weights) const;
    void inverse_jacobian(const Real* J, Real* J_inv, Real det_J) const;
    void constitutive_matrix(Real E, Real nu, Real* C) const;
};

} // namespace fem
} // namespace nxs
