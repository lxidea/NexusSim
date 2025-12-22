#pragma once

/**
 * @file shell4.hpp
 * @brief 4-node quadrilateral shell element (membrane + bending)
 *
 * Node numbering (counterclockwise):
 *     3--------2
 *     |        |
 *     |        |
 *     0--------1
 *
 * Natural coordinates: ξ, η ∈ [-1, 1]
 * DOFs per node: 6 (ux, uy, uz, θx, θy, θz)
 *
 * Formulation: Combines:
 *   - Membrane behavior (in-plane stretching)
 *   - Bending behavior (out-of-plane deformation)
 *   - Simplified without drilling DOF rotation
 */

#include <nexussim/core/core.hpp>
#include <nexussim/physics/element.hpp>
#include <array>
#include <cmath>

namespace nxs {
namespace fem {

// ============================================================================
// Shell4 Element - 4-node Quadrilateral Shell
// ============================================================================

class Shell4Element : public physics::Element {
public:
    static constexpr int NUM_NODES = 4;
    static constexpr int NUM_DIMS = 3;  // 3D space
    static constexpr int DOF_PER_NODE = 6;  // 3 translations + 3 rotations
    static constexpr int NUM_DOF = NUM_NODES * DOF_PER_NODE;  // 24 DOFs
    static constexpr int NUM_STRESS_COMPONENTS = 6;

    Shell4Element() = default;
    ~Shell4Element() override = default;

    // ========================================================================
    // Element Properties
    // ========================================================================

    Properties properties() const override {
        return Properties{
            physics::ElementType::Shell4,
            physics::ElementTopology::Quadrilateral,
            NUM_NODES,
            4,  // 2x2 Gauss integration
            DOF_PER_NODE,
            NUM_DIMS
        };
    }

    // ========================================================================
    // Shape Functions (2D in-plane)
    // ========================================================================

    /**
     * @brief Compute shape functions at natural coordinates
     * @param xi Natural coordinates [ξ, η] (2D)
     * @param N Output shape functions (size 4)
     */
    KOKKOS_INLINE_FUNCTION
    void shape_functions(const Real xi[3], Real* N) const override;

    /**
     * @brief Compute shape function derivatives w.r.t. natural coordinates
     * @param xi Natural coordinates [ξ, η]
     * @param dN Output derivatives (4x2 matrix for in-plane)
     */
    KOKKOS_INLINE_FUNCTION
    void shape_derivatives(const Real xi[3], Real* dN) const override;

    // ========================================================================
    // Gauss Quadrature
    // ========================================================================

    void gauss_quadrature(Real* points, Real* weights) const override;

    // ========================================================================
    // Jacobian and Coordinate Mapping
    // ========================================================================

    /**
     * @brief Compute Jacobian matrix for membrane
     * @param xi Natural coordinates
     * @param coords Element nodal coordinates (4 nodes x 3 coords)
     * @param J Output Jacobian matrix (3x3)
     * @return Jacobian determinant
     */
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

    /**
     * @brief Set shell thickness
     * @param t Shell thickness
     */
    void set_thickness(Real t) { thickness_ = t; }

    Real thickness() const { return thickness_; }

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
    Real volume(const Real* coords) const override;  // Actually returns area
    Real characteristic_length(const Real* coords) const override;

    // ========================================================================
    // Shell-Specific Methods
    // ========================================================================

    /**
     * @brief Compute local coordinate system
     * @param coords Element nodal coordinates
     * @param e1 Output: local x-axis
     * @param e2 Output: local y-axis
     * @param e3 Output: local z-axis (normal)
     */
    void local_coordinate_system(const Real* coords, Real* e1, Real* e2, Real* e3) const;

    /**
     * @brief Transform global to local coordinates
     * @param T Transformation matrix (3x3)
     * @param global Global vector (3 components)
     * @param local Output local vector (3 components)
     */
    void transform_to_local(const Real* T, const Real* global, Real* local) const;

private:
    Real thickness_ = 0.01;  // Default 10mm thickness

    void compute_gauss_points_2x2(Real* points, Real* weights) const;
    void membrane_B_matrix(const Real xi[2], const Real* coords, Real* B_mem) const;
    void bending_B_matrix(const Real xi[2], const Real* coords, Real* B_bend) const;
};

} // namespace fem
} // namespace nxs
