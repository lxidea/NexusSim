#pragma once

/**
 * @file wedge6.hpp
 * @brief 6-node wedge/prism element with linear shape functions
 *
 * Node numbering (right-hand rule):
 *        5
 *       /|\
 *      / | \
 *     /  |  \
 *    3---|---4
 *    |   2   |
 *    |  / \  |
 *    | /   \ |
 *    |/     \|
 *    0-------1
 *
 * Natural coordinates:
 *   ξ, η ∈ [0, 1] (triangular coordinates: ξ + η ≤ 1)
 *   ζ ∈ [-1, 1] (axial direction)
 *
 * Node local coordinates:
 *   0: (1, 0, -1)    3: (1, 0, +1)
 *   1: (0, 1, -1)    4: (0, 1, +1)
 *   2: (0, 0, -1)    5: (0, 0, +1)
 */

#include <nexussim/core/core.hpp>
#include <nexussim/physics/element.hpp>
#include <array>
#include <cmath>

namespace nxs {
namespace fem {

class Wedge6Element : public physics::Element {
public:
    static constexpr int NUM_NODES = 6;
    static constexpr int NUM_DIMS = 3;
    static constexpr int NUM_STRESS_COMPONENTS = 6;
    static constexpr int NUM_DOF = NUM_NODES * NUM_DIMS;  // 18 DOFs

    Wedge6Element() = default;
    ~Wedge6Element() override = default;

    Properties properties() const override {
        return Properties{
            physics::ElementType::Wedge6,
            physics::ElementTopology::Wedge,
            NUM_NODES,
            6,  // 6-point Gauss integration (2x3)
            NUM_DIMS,
            NUM_DIMS
        };
    }

    /**
     * @brief Compute shape functions
     * N_i = L_i * (1 + ζ_i*ζ) / 2
     * where L_i are triangular shape functions in ξ-η plane
     */
    KOKKOS_INLINE_FUNCTION
    void shape_functions(const Real xi[3], Real* N) const override;

    KOKKOS_INLINE_FUNCTION
    void shape_derivatives(const Real xi[3], Real* dN) const override;

    void gauss_quadrature(Real* points, Real* weights) const override;

    KOKKOS_INLINE_FUNCTION
    Real jacobian(const Real xi[3], const Real* coords, Real* J) const override;

    KOKKOS_INLINE_FUNCTION
    void strain_displacement_matrix(const Real xi[3], const Real* coords, Real* B) const override;

    void mass_matrix(const Real* coords, Real density, Real* M) const override;
    void stiffness_matrix(const Real* coords, Real E, Real nu, Real* K) const override;

    KOKKOS_INLINE_FUNCTION
    void internal_force(const Real* coords, const Real* disp,
                       const Real* stress, Real* fint) const override;

    bool contains_point(const Real* coords, const Real* point, Real* xi) const override;
    Real volume(const Real* coords) const override;
    Real characteristic_length(const Real* coords) const override;

private:
    void compute_gauss_points_6pt(Real* points, Real* weights) const;
    void inverse_jacobian(const Real* J, Real* J_inv, Real det_J) const;
    void constitutive_matrix(Real E, Real nu, Real* C) const;
    Real shape_derivatives_global(const Real xi[3], const Real* coords, Real* dNdx) const;
};

} // namespace fem
} // namespace nxs
