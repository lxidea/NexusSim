#pragma once

/**
 * @file truss.hpp
 * @brief 2-node 3D truss/bar element (axial deformation only)
 *
 * Node numbering:
 *     0-----------1
 *
 * DOFs per node: 3 (ux, uy, uz) - translational only, no rotations
 *
 * Formulation: Linear truss element
 *   - Axial deformation only (no bending, shear, or torsion)
 *   - Pin-jointed structure assumption
 *   - Suitable for cables, bars, struts, and space trusses
 *
 * This is simpler than Beam2 - no rotational DOFs, no bending.
 */

#include <nexussim/core/core.hpp>
#include <nexussim/physics/element.hpp>
#include <array>
#include <cmath>

namespace nxs {
namespace fem {

// ============================================================================
// Truss Element - 2-node 3D Bar/Truss
// ============================================================================

class TrussElement : public physics::Element {
public:
    static constexpr int NUM_NODES = 2;
    static constexpr int NUM_DIMS = 3;  // 3D space
    static constexpr int DOF_PER_NODE = 3;  // 3 translations only
    static constexpr int NUM_DOF = NUM_NODES * DOF_PER_NODE;  // 6 DOFs

    TrussElement() = default;
    ~TrussElement() override = default;

    // ========================================================================
    // Element Properties
    // ========================================================================

    Properties properties() const override {
        return Properties{
            physics::ElementType::Truss2,
            physics::ElementTopology::Line,
            NUM_NODES,
            2,  // 2-point Gauss (though 1 point is sufficient for linear)
            DOF_PER_NODE,
            NUM_DIMS
        };
    }

    // ========================================================================
    // Cross-Section Properties
    // ========================================================================

    /**
     * @brief Set circular cross-section
     * @param radius Bar radius
     */
    void set_circular_section(Real radius) {
        A_ = M_PI * radius * radius;
    }

    /**
     * @brief Set rectangular cross-section
     * @param width Width
     * @param height Height
     */
    void set_rectangular_section(Real width, Real height) {
        A_ = width * height;
    }

    /**
     * @brief Set cross-sectional area directly
     * @param A Cross-sectional area
     */
    void set_area(Real A) { A_ = A; }

    Real area() const { return A_; }

    // ========================================================================
    // Shape Functions (1D)
    // ========================================================================

    /**
     * @brief Compute shape functions at natural coordinate
     * @param xi Natural coordinate [ξ] ∈ [-1, 1]
     * @param N Output shape functions (size 2)
     *
     * N1 = (1 - ξ) / 2
     * N2 = (1 + ξ) / 2
     */
    KOKKOS_INLINE_FUNCTION
    void shape_functions(const Real xi[3], Real* N) const override {
        const Real x = xi[0];
        N[0] = 0.5 * (1.0 - x);
        N[1] = 0.5 * (1.0 + x);
    }

    /**
     * @brief Compute shape function derivatives
     * @param xi Natural coordinate
     * @param dN Output derivatives (2x1)
     *
     * dN1/dξ = -1/2
     * dN2/dξ = +1/2
     */
    KOKKOS_INLINE_FUNCTION
    void shape_derivatives(const Real xi[3], Real* dN) const override {
        (void)xi;  // Constant derivatives for linear element
        dN[0] = -0.5;
        dN[1] = +0.5;
    }

    // ========================================================================
    // Gauss Quadrature
    // ========================================================================

    void gauss_quadrature(Real* points, Real* weights) const override {
        // 2-point Gauss quadrature (more than needed, but consistent)
        const Real gp = 1.0 / Kokkos::sqrt(3.0);

        points[0*3 + 0] = -gp;
        points[0*3 + 1] = 0.0;
        points[0*3 + 2] = 0.0;
        weights[0] = 1.0;

        points[1*3 + 0] = +gp;
        points[1*3 + 1] = 0.0;
        points[1*3 + 2] = 0.0;
        weights[1] = 1.0;
    }

    // ========================================================================
    // Jacobian and Coordinate Mapping
    // ========================================================================

    /**
     * @brief Compute Jacobian for 1D element in 3D
     * @param xi Natural coordinate
     * @param coords Element nodal coordinates (2 nodes x 3 coords)
     * @param J Output Jacobian (just the length scale = L/2)
     * @return Jacobian determinant = L/2
     */
    KOKKOS_INLINE_FUNCTION
    Real jacobian(const Real xi[3], const Real* coords, Real* J) const override {
        (void)xi;  // Constant for linear element

        // Length of element
        const Real L = this->length(coords);

        // Jacobian = dX/dξ = L/2 for linear element
        J[0] = L / 2.0;

        return J[0];
    }

    // ========================================================================
    // B-Matrix (Strain-Displacement)
    // ========================================================================

    /**
     * @brief Compute strain-displacement matrix
     * @param xi Natural coordinate
     * @param coords Element nodal coordinates
     * @param B Output B-matrix [1 strain × 6 DOF]
     *
     * For truss: ε_axial = (u2 - u1) / L (in local coords)
     * In global coords: ε = (e^T × u2 - e^T × u1) / L
     *                     = e^T × (u2 - u1) / L
     * where e = unit vector along element axis
     *
     * B = (1/L) × [-ex, -ey, -ez, ex, ey, ez]
     */
    KOKKOS_INLINE_FUNCTION
    void strain_displacement_matrix(const Real xi[3], const Real* coords, Real* B) const override {
        (void)xi;  // Constant for linear truss

        // Element axis direction
        Real e[3];
        Real L = compute_direction(coords, e);
        Real L_inv = 1.0 / L;

        // B-matrix: ε = B × u, where u = [u1x, u1y, u1z, u2x, u2y, u2z]^T
        B[0] = -e[0] * L_inv;  // dε/du1x
        B[1] = -e[1] * L_inv;  // dε/du1y
        B[2] = -e[2] * L_inv;  // dε/du1z
        B[3] = +e[0] * L_inv;  // dε/du2x
        B[4] = +e[1] * L_inv;  // dε/du2y
        B[5] = +e[2] * L_inv;  // dε/du2z
    }

    // ========================================================================
    // Element Matrices
    // ========================================================================

    /**
     * @brief Compute lumped mass matrix
     * @param coords Element nodal coordinates
     * @param density Material density
     * @param M Output mass (6 diagonal values)
     */
    void lumped_mass_matrix(const Real* coords, Real density, Real* M) const {
        // Total mass = density × length × area
        Real L = this->length(coords);
        Real total_mass = density * L * A_;

        // Distribute equally: half mass to each node
        Real mass_per_node = total_mass / 2.0;

        // All translational DOFs get same mass
        for (int i = 0; i < 6; ++i) {
            M[i] = mass_per_node;
        }
    }

    void mass_matrix(const Real* coords, Real density, Real* M) const override {
        // Use lumped mass for explicit dynamics
        Real M_lumped[6];
        lumped_mass_matrix(coords, density, M_lumped);

        // Convert to diagonal matrix (6×6)
        for (int i = 0; i < 36; ++i) M[i] = 0.0;
        for (int i = 0; i < 6; ++i) {
            M[i*6 + i] = M_lumped[i];
        }
    }

    void stiffness_matrix(const Real* coords, Real E, Real nu, Real* K) const override {
        (void)nu;  // Not used for uniaxial stress

        // Get element direction and length
        Real e[3];
        Real L = compute_direction(coords, e);

        // Axial stiffness: k = EA/L
        Real k = E * A_ / L;

        // Global stiffness matrix: K = k × (e ⊗ e) assembled for both nodes
        // K = k × [  e⊗e   -e⊗e ]
        //         [ -e⊗e    e⊗e ]

        // Initialize to zero
        for (int i = 0; i < 36; ++i) K[i] = 0.0;

        // Fill 3×3 blocks
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Real eiej = k * e[i] * e[j];

                // K11 block (node 0, node 0)
                K[(0*3+i)*6 + (0*3+j)] = eiej;

                // K12 block (node 0, node 1)
                K[(0*3+i)*6 + (3+j)] = -eiej;

                // K21 block (node 1, node 0)
                K[(3+i)*6 + (0*3+j)] = -eiej;

                // K22 block (node 1, node 1)
                K[(3+i)*6 + (3+j)] = eiej;
            }
        }
    }

    // ========================================================================
    // Internal Force Computation
    // ========================================================================

    KOKKOS_INLINE_FUNCTION
    void internal_force(const Real* coords, const Real* disp,
                       const Real* stress, Real* fint) const override {
        // f_int = B^T × σ × A × L / 2 (integrated over element)
        // For truss: f = A × σ_axial × e (at each node, opposite signs)

        Real e[3];
        Real L = compute_direction(coords, e);

        // Axial stress (from stress tensor, we use σ11 or provided value)
        Real sigma_axial = stress[0];  // Assuming Voigt notation σxx

        // Force = A × σ
        Real F = A_ * sigma_axial;

        // Node 0: -F × e (compression if σ > 0 and element lengthens)
        // Node 1: +F × e
        // Wait, for tension (σ > 0), internal force should resist elongation
        // If u2 > u1 (stretching), σ > 0, force on node 0 is +F×e (pulls right)
        // and force on node 1 is -F×e (pulls left)

        fint[0] = -F * e[0];
        fint[1] = -F * e[1];
        fint[2] = -F * e[2];
        fint[3] = +F * e[0];
        fint[4] = +F * e[1];
        fint[5] = +F * e[2];
    }

    // ========================================================================
    // Geometric Queries
    // ========================================================================

    bool contains_point(const Real* coords, const Real* point, Real* xi) const override {
        // Project point onto element axis
        Real e[3];
        Real L = compute_direction(coords, e);

        // Vector from node 0 to point
        Real v[3] = {
            point[0] - coords[0],
            point[1] - coords[1],
            point[2] - coords[2]
        };

        // Parametric coordinate
        Real t = (v[0]*e[0] + v[1]*e[1] + v[2]*e[2]) / L;

        // Convert to natural coordinate
        xi[0] = 2.0 * t - 1.0;  // t=0 → ξ=-1, t=1 → ξ=+1
        xi[1] = 0.0;
        xi[2] = 0.0;

        // Check if on element (with tolerance)
        return (t >= -1.0e-6 && t <= 1.0 + 1.0e-6);
    }

    Real volume(const Real* coords) const override {
        // Volume = length × area
        return length(coords) * A_;
    }

    Real characteristic_length(const Real* coords) const override {
        return length(coords);
    }

    /**
     * @brief Compute element length
     * @param coords Element nodal coordinates
     * @return Length
     */
    KOKKOS_INLINE_FUNCTION
    Real length(const Real* coords) const {
        const Real dx = coords[3] - coords[0];
        const Real dy = coords[4] - coords[1];
        const Real dz = coords[5] - coords[2];
        return Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
    }

    /**
     * @brief Compute axial strain from displacements
     * @param coords Element nodal coordinates
     * @param disp Nodal displacements (6 values)
     * @return Axial strain
     */
    KOKKOS_INLINE_FUNCTION
    Real axial_strain(const Real* coords, const Real* disp) const {
        Real e[3];
        Real L = compute_direction(coords, e);

        // Axial displacement: δ = e · (u2 - u1)
        Real delta = e[0] * (disp[3] - disp[0]) +
                     e[1] * (disp[4] - disp[1]) +
                     e[2] * (disp[5] - disp[2]);

        return delta / L;
    }

    /**
     * @brief Compute axial stress from strain
     * @param strain Axial strain
     * @param E Young's modulus
     * @return Axial stress
     */
    KOKKOS_INLINE_FUNCTION
    static Real axial_stress(Real strain, Real E) {
        return E * strain;
    }

private:
    Real A_ = 0.01;  // Cross-sectional area (default: 1 cm²)

    /**
     * @brief Compute unit direction vector along element
     * @param coords Element nodal coordinates
     * @param e Output: unit direction vector
     * @return Length
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_direction(const Real* coords, Real* e) const {
        e[0] = coords[3] - coords[0];
        e[1] = coords[4] - coords[1];
        e[2] = coords[5] - coords[2];

        Real L = Kokkos::sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);

        if (L > 1.0e-20) {
            e[0] /= L;
            e[1] /= L;
            e[2] /= L;
        } else {
            e[0] = 1.0;
            e[1] = 0.0;
            e[2] = 0.0;
        }

        return L;
    }
};

} // namespace fem
} // namespace nxs
