#pragma once

/**
 * @file pd_mortar_coupling.hpp
 * @brief Mortar-based interface coupling between FEM and PD domains
 *
 * Implements penalty-based mortar coupling:
 * - Projects PD particles onto FEM element faces
 * - Computes gap and applies penalty forces
 * - Distributes forces to FEM nodes via shape functions
 */

#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_particle.hpp>
#include <nexussim/peridynamics/pd_fem_coupling.hpp>
#include <nexussim/data/mesh.hpp>
#include <vector>

namespace nxs {
namespace pd {

// ============================================================================
// Interface pair
// ============================================================================

struct InterfacePair {
    Index pd_particle;           // PD particle index
    Index fem_block;             // FEM element block index
    Index fem_element;           // Local element index in block
    Real xi = 0.0;              // Natural coordinate on face
    Real eta = 0.0;             // Natural coordinate on face
    Real distance = 0.0;        // Distance to surface
};

// ============================================================================
// 2D shape functions for surface projection
// ============================================================================

KOKKOS_INLINE_FUNCTION
void fem_shape_functions_2d(Real xi, Real eta, Real* N, int num_nodes) {
    if (num_nodes == 4) {
        // Bilinear quad
        N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
        N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
        N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
        N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);
    } else if (num_nodes == 3) {
        // Linear triangle
        N[0] = 1.0 - xi - eta;
        N[1] = xi;
        N[2] = eta;
    } else {
        // Default to uniform
        Real inv_n = 1.0 / num_nodes;
        for (int i = 0; i < num_nodes; ++i)
            N[i] = inv_n;
    }
}

// ============================================================================
// PDMortarCoupling
// ============================================================================

class PDMortarCoupling {
public:
    PDMortarCoupling() = default;

    /**
     * @brief Setup interface by finding PD particles near FEM boundary
     *
     * @param fem_boundary_nodes Indices of FEM nodes on the coupling boundary
     * @param particles PD particle system
     * @param mesh FEM mesh
     * @param search_radius Maximum distance for pairing
     */
    void setup_interface(
        const std::vector<Index>& fem_boundary_nodes,
        PDParticleSystem& particles,
        const Mesh& mesh,
        Real search_radius)
    {
        particles.sync_to_host();
        auto x_pd = particles.x_host();
        Index num_pd = particles.num_particles();

        interface_pairs_.clear();

        // For each PD particle, find the closest FEM boundary node
        for (Index p = 0; p < num_pd; ++p) {
            Real px = x_pd(p, 0);
            Real py = x_pd(p, 1);
            Real pz = x_pd(p, 2);

            Real min_dist = search_radius;
            Index closest_node = 0;
            bool found = false;

            for (Index bn : fem_boundary_nodes) {
                Vec3r nc = mesh.get_node_coordinates(bn);
                Real dx = px - nc[0];
                Real dy = py - nc[1];
                Real dz = pz - nc[2];
                Real dist = std::sqrt(dx * dx + dy * dy + dz * dz);

                if (dist < min_dist) {
                    min_dist = dist;
                    closest_node = bn;
                    found = true;
                }
            }

            if (found) {
                InterfacePair pair;
                pair.pd_particle = p;
                pair.fem_block = 0;
                pair.fem_element = closest_node; // Use node index for now
                pair.xi = 0.0;
                pair.eta = 0.0;
                pair.distance = min_dist;
                interface_pairs_.push_back(pair);
            }
        }
    }

    /**
     * @brief Compute penalty coupling forces between FEM and PD at interface
     *
     * @param particles PD particle system
     * @param mesh FEM mesh
     * @param fem_displacement FEM displacement (flat [node*3+dof])
     * @param fem_forces FEM external force vector (to receive coupling forces)
     * @param penalty_stiffness Penalty parameter (N/m)
     */
    void compute_coupling_forces(
        PDParticleSystem& particles,
        const Mesh& mesh,
        const Kokkos::View<Real*>& fem_displacement,
        Kokkos::View<Real*>& fem_forces,
        Real penalty_stiffness)
    {
        if (interface_pairs_.empty()) return;

        particles.sync_to_host();
        auto x_pd = particles.x_host();
        auto u_pd = particles.u_host();
        auto f_host = particles.f_host();

        auto fem_disp_host = Kokkos::create_mirror_view(fem_displacement);
        Kokkos::deep_copy(fem_disp_host, fem_displacement);

        auto fem_f_host = Kokkos::create_mirror_view(fem_forces);
        Kokkos::deep_copy(fem_f_host, fem_forces);

        for (auto& pair : interface_pairs_) {
            Index pd_id = pair.pd_particle;
            Index fem_node = pair.fem_element; // Using node-based pairing

            // Current PD position (reference + displacement)
            Real pd_pos[3] = {
                x_pd(pd_id, 0),
                x_pd(pd_id, 1),
                x_pd(pd_id, 2)
            };

            // Current FEM node position
            Vec3r fem_ref = mesh.get_node_coordinates(fem_node);
            Real fem_pos[3] = {
                fem_ref[0] + fem_disp_host(fem_node * 3 + 0),
                fem_ref[1] + fem_disp_host(fem_node * 3 + 1),
                fem_ref[2] + fem_disp_host(fem_node * 3 + 2)
            };

            // Gap vector: PD - FEM
            Real gap[3] = {
                pd_pos[0] - fem_pos[0],
                pd_pos[1] - fem_pos[1],
                pd_pos[2] - fem_pos[2]
            };

            // Penalty force: f = -k * gap
            Real f_coupling[3] = {
                -penalty_stiffness * gap[0],
                -penalty_stiffness * gap[1],
                -penalty_stiffness * gap[2]
            };

            // Apply to PD particle (on host, accumulate)
            f_host(pd_id, 0) += f_coupling[0];
            f_host(pd_id, 1) += f_coupling[1];
            f_host(pd_id, 2) += f_coupling[2];

            // Apply equal and opposite to FEM node
            fem_f_host(fem_node * 3 + 0) -= f_coupling[0];
            fem_f_host(fem_node * 3 + 1) -= f_coupling[1];
            fem_f_host(fem_node * 3 + 2) -= f_coupling[2];
        }

        // Copy back to device
        Kokkos::deep_copy(fem_forces, fem_f_host);
        Kokkos::deep_copy(particles.f(), f_host);
    }

    /**
     * @brief Project a point onto a quad face and get natural coordinates
     *
     * @param x_point Point to project [3]
     * @param face_nodes Face node coordinates [4][3]
     * @param xi Output natural coordinate xi
     * @param eta Output natural coordinate eta
     * @param projected Output projected position [3]
     * @return true if projection is within face bounds
     */
    static bool project_to_fem_surface(
        const Real* x_point,
        const Real face_nodes[4][3],
        Real& xi, Real& eta,
        Real* projected)
    {
        // Newton iteration to find (xi,eta) such that
        // x_point ≈ sum_i N_i(xi,eta) * face_nodes[i]

        xi = 0.0;
        eta = 0.0;

        for (int iter = 0; iter < 10; ++iter) {
            Real N[4];
            fem_shape_functions_2d(xi, eta, N, 4);

            // Compute interpolated position
            Real x_interp[3] = {0.0, 0.0, 0.0};
            for (int n = 0; n < 4; ++n)
                for (int d = 0; d < 3; ++d)
                    x_interp[d] += N[n] * face_nodes[n][d];

            // Residual
            Real r[3] = {
                x_point[0] - x_interp[0],
                x_point[1] - x_interp[1],
                x_point[2] - x_interp[2]
            };

            // Jacobian: dN/dxi, dN/deta
            Real dNdxi[4] = {
                -0.25 * (1.0 - eta),
                 0.25 * (1.0 - eta),
                 0.25 * (1.0 + eta),
                -0.25 * (1.0 + eta)
            };
            Real dNdeta[4] = {
                -0.25 * (1.0 - xi),
                -0.25 * (1.0 + xi),
                 0.25 * (1.0 + xi),
                 0.25 * (1.0 - xi)
            };

            Real dxdxi[3] = {0.0, 0.0, 0.0};
            Real dxdeta[3] = {0.0, 0.0, 0.0};
            for (int n = 0; n < 4; ++n)
                for (int d = 0; d < 3; ++d) {
                    dxdxi[d] += dNdxi[n] * face_nodes[n][d];
                    dxdeta[d] += dNdeta[n] * face_nodes[n][d];
                }

            // 2x2 normal equation system: J^T J [dxi, deta]^T = J^T r
            Real a11 = dxdxi[0]*dxdxi[0] + dxdxi[1]*dxdxi[1] + dxdxi[2]*dxdxi[2];
            Real a12 = dxdxi[0]*dxdeta[0] + dxdxi[1]*dxdeta[1] + dxdxi[2]*dxdeta[2];
            Real a22 = dxdeta[0]*dxdeta[0] + dxdeta[1]*dxdeta[1] + dxdeta[2]*dxdeta[2];

            Real b1 = dxdxi[0]*r[0] + dxdxi[1]*r[1] + dxdxi[2]*r[2];
            Real b2 = dxdeta[0]*r[0] + dxdeta[1]*r[1] + dxdeta[2]*r[2];

            Real det = a11 * a22 - a12 * a12;
            if (std::fabs(det) < 1e-30) break;

            Real inv_det = 1.0 / det;
            Real dxi = (a22 * b1 - a12 * b2) * inv_det;
            Real deta = (a11 * b2 - a12 * b1) * inv_det;

            xi += dxi;
            eta += deta;

            if (std::fabs(dxi) + std::fabs(deta) < 1e-10) break;
        }

        // Compute projected position
        Real N[4];
        fem_shape_functions_2d(xi, eta, N, 4);
        projected[0] = projected[1] = projected[2] = 0.0;
        for (int n = 0; n < 4; ++n)
            for (int d = 0; d < 3; ++d)
                projected[d] += N[n] * face_nodes[n][d];

        // Check if within bounds
        return (xi >= -1.0 - 1e-6 && xi <= 1.0 + 1e-6 &&
                eta >= -1.0 - 1e-6 && eta <= 1.0 + 1e-6);
    }

    /**
     * @brief Distribute a force to face nodes using shape functions
     *
     * @param force Force vector [3]
     * @param xi Natural coordinate
     * @param eta Natural coordinate
     * @param num_face_nodes Number of face nodes
     * @param nodal_forces Output: force on each node [num_face_nodes][3]
     */
    static void distribute_force_to_nodes(
        const Real* force,
        Real xi, Real eta,
        int num_face_nodes,
        Real nodal_forces[][3])
    {
        Real N[8];
        fem_shape_functions_2d(xi, eta, N, num_face_nodes);

        for (int n = 0; n < num_face_nodes; ++n)
            for (int d = 0; d < 3; ++d)
                nodal_forces[n][d] = N[n] * force[d];
    }

    // Accessors
    const std::vector<InterfacePair>& interface_pairs() const { return interface_pairs_; }
    Index num_pairs() const { return static_cast<Index>(interface_pairs_.size()); }

private:
    std::vector<InterfacePair> interface_pairs_;
};

} // namespace pd
} // namespace nxs
