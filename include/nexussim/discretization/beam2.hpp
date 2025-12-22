#pragma once

/**
 * @file beam2.hpp
 * @brief 2-node 3D Euler-Bernoulli beam element
 *
 * Node numbering:
 *     0-----------1
 *
 * DOFs per node: 6 (ux, uy, uz, θx, θy, θz)
 *
 * Formulation: 3D Euler-Bernoulli beam theory
 *   - Axial deformation (along beam axis)
 *   - Torsion (about beam axis)
 *   - Bending in two planes (y-z and x-z)
 *   - Neglects shear deformation (valid for slender beams)
 */

#include <nexussim/core/core.hpp>
#include <nexussim/physics/element.hpp>
#include <array>
#include <cmath>

namespace nxs {
namespace fem {

// ============================================================================
// Beam2 Element - 2-node 3D Euler-Bernoulli Beam
// ============================================================================

class Beam2Element : public physics::Element {
public:
    static constexpr int NUM_NODES = 2;
    static constexpr int NUM_DIMS = 3;  // 3D space
    static constexpr int DOF_PER_NODE = 6;  // 3 translations + 3 rotations
    static constexpr int NUM_DOF = NUM_NODES * DOF_PER_NODE;  // 12 DOFs

    Beam2Element() = default;
    ~Beam2Element() override = default;

    // ========================================================================
    // Element Properties
    // ========================================================================

    Properties properties() const override {
        return Properties{
            physics::ElementType::Beam2,
            physics::ElementTopology::Line,
            NUM_NODES,
            2,  // 2-point Gauss integration
            DOF_PER_NODE,
            NUM_DIMS
        };
    }

    // ========================================================================
    // Cross-Section Properties
    // ========================================================================

    /**
     * @brief Set circular cross-section
     * @param radius Beam radius
     */
    void set_circular_section(Real radius);

    /**
     * @brief Set rectangular cross-section
     * @param width Width (y-direction)
     * @param height Height (z-direction)
     */
    void set_rectangular_section(Real width, Real height);

    /**
     * @brief Set cross-section properties directly
     * @param A Cross-sectional area
     * @param Iy Second moment of area about y-axis
     * @param Iz Second moment of area about z-axis
     * @param J Torsional constant
     */
    void set_section_properties(Real A, Real Iy, Real Iz, Real J);

    Real area() const { return A_; }
    Real moment_y() const { return Iy_; }
    Real moment_z() const { return Iz_; }
    Real torsion_constant() const { return J_; }

    // ========================================================================
    // Shape Functions (1D)
    // ========================================================================

    /**
     * @brief Compute shape functions at natural coordinate
     * @param xi Natural coordinate [ξ] ∈ [-1, 1]
     * @param N Output shape functions (size 2 for axial)
     */
    KOKKOS_INLINE_FUNCTION
    void shape_functions(const Real xi[3], Real* N) const override;

    KOKKOS_INLINE_FUNCTION
    void shape_derivatives(const Real xi[3], Real* dN) const override;

    // ========================================================================
    // Gauss Quadrature
    // ========================================================================

    void gauss_quadrature(Real* points, Real* weights) const override;

    // ========================================================================
    // Jacobian and Coordinate Mapping
    // ========================================================================

    KOKKOS_INLINE_FUNCTION
    Real jacobian(const Real xi[3], const Real* coords, Real* J) const override;

    // ========================================================================
    // B-Matrix (Strain-Displacement)
    // ========================================================================

    KOKKOS_INLINE_FUNCTION
    void strain_displacement_matrix(const Real xi[3], const Real* coords, Real* B) const override;

    // ========================================================================
    // Element Matrices
    // ========================================================================

    void lumped_mass_matrix(const Real* coords, Real density, Real* M) const;
    void mass_matrix(const Real* coords, Real density, Real* M) const override;
    void stiffness_matrix(const Real* coords, Real E, Real nu, Real* K) const override;

    // ========================================================================
    // Internal Force Computation
    // ========================================================================

    KOKKOS_INLINE_FUNCTION
    void internal_force(const Real* coords, const Real* disp,
                       const Real* stress, Real* fint) const override;

    // ========================================================================
    // Geometric Queries
    // ========================================================================

    bool contains_point(const Real* coords, const Real* point, Real* xi) const override;
    Real volume(const Real* coords) const override;  // Actually returns length × area
    Real characteristic_length(const Real* coords) const override;

    // ========================================================================
    // Beam-Specific Methods
    // ========================================================================

    /**
     * @brief Compute beam length
     * @param coords Element nodal coordinates
     * @return Beam length
     */
    Real length(const Real* coords) const;

    /**
     * @brief Compute local coordinate system
     * @param coords Element nodal coordinates
     * @param e1 Output: local x-axis (along beam)
     * @param e2 Output: local y-axis
     * @param e3 Output: local z-axis
     */
    void local_coordinate_system(const Real* coords, Real* e1, Real* e2, Real* e3) const;

private:
    // Cross-section properties
    Real A_ = 0.01;      // Cross-sectional area (default: 100 mm²)
    Real Iy_ = 1.0e-6;   // Second moment of area about y
    Real Iz_ = 1.0e-6;   // Second moment of area about z
    Real J_ = 2.0e-6;    // Torsional constant

    // Hermite shape functions for bending
    void hermite_shape_functions(Real xi, Real L, Real* N, Real* dN) const;
};

} // namespace fem
} // namespace nxs
