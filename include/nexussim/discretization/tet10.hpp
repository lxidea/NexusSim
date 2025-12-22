#pragma once

/**
 * @file tet10.hpp
 * @brief 10-node tetrahedral element with quadratic shape functions
 *
 * Node numbering (right-hand rule):
 * Corner nodes (0-3) same as Tet4:
 *        3
 *       /|\
 *      / | \
 *     /  |  \
 *    / 7 9   \
 *   /    |    6
 *  /     0     \
 * 1 ---- 4 ---- 2
 *  \     |     /
 *   \ 8  |  5 /
 *    \   |   /
 *     \  |  /
 *      \ | /
 *       \|/
 *
 * Node layout:
 *   Corner nodes: 0-3 (vertices)
 *   Edge nodes:   4-9 (mid-edges)
 *     Edge 0-1: node 4
 *     Edge 1-2: node 5
 *     Edge 2-0: node 6
 *     Edge 0-3: node 7
 *     Edge 1-3: node 8
 *     Edge 2-3: node 9
 *
 * Natural coordinates: L1, L2, L3, L4 (barycentric)
 * Constraint: L1 + L2 + L3 + L4 = 1
 */

#include <nexussim/core/core.hpp>
#include <nexussim/physics/element.hpp>
#include <array>
#include <cmath>

namespace nxs {
namespace fem {

class Tet10Element : public physics::Element {
public:
    static constexpr int NUM_NODES = 10;
    static constexpr int NUM_DIMS = 3;
    static constexpr int NUM_STRESS_COMPONENTS = 6;
    static constexpr int NUM_DOF = NUM_NODES * NUM_DIMS;  // 30 DOFs

    Tet10Element() = default;
    ~Tet10Element() override = default;

    Properties properties() const override {
        return Properties{
            physics::ElementType::Tet10,
            physics::ElementTopology::Tetrahedron,
            NUM_NODES,
            4,  // 4-point Gauss integration for quadratic tet
            NUM_DIMS,
            NUM_DIMS
        };
    }

    /**
     * @brief Compute shape functions at natural coordinates
     * Quadratic shape functions for 10-node tetrahedron:
     * Corner nodes (i=0-3): N_i = L_i(2*L_i - 1)
     * Edge nodes (i=4-9):   N_i = 4*L_j*L_k (for edge between nodes j and k)
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
    void compute_gauss_points_4pt(Real* points, Real* weights) const;
    void inverse_jacobian(const Real* J, Real* J_inv, Real det_J) const;
    void constitutive_matrix(Real E, Real nu, Real* C) const;
    Real shape_derivatives_global(const Real xi[3], const Real* coords, Real* dNdx) const;
};

} // namespace fem
} // namespace nxs
