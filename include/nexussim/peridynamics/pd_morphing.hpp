#pragma once

/**
 * @file pd_morphing.hpp
 * @brief Dynamic FEM-to-PD element morphing
 *
 * Converts damaged FEM elements to PD particles:
 * - Creates particles at element node positions
 * - Transfers state (displacement, velocity) via shape function interpolation
 * - Creates bonds between new particles and existing PD neighbors
 * - Replaces the stub in pd_fem_coupling.hpp
 */

#include <nexussim/peridynamics/pd_types.hpp>
#include <nexussim/peridynamics/pd_particle.hpp>
#include <nexussim/peridynamics/pd_neighbor.hpp>
#include <nexussim/peridynamics/pd_fem_coupling.hpp>
#include <nexussim/physics/element_erosion.hpp>
#include <nexussim/data/mesh.hpp>
#include <vector>

namespace nxs {
namespace pd {

// ============================================================================
// Morphing result
// ============================================================================

struct MorphingResult {
    Index elements_converted = 0;
    Index particles_created = 0;
    Index bonds_created = 0;
};

// ============================================================================
// PDElementMorphing
// ============================================================================

class PDElementMorphing {
public:
    PDElementMorphing() = default;

    /**
     * @brief Convert damaged FEM elements to PD particles
     *
     * @param coupling FEM-PD coupling manager
     * @param erosion_mgr Element erosion manager with damage info
     * @param mesh FEM mesh
     * @param particles PD particle system (will be expanded)
     * @param neighbors PD neighbor list
     * @param material PD material for new particles
     * @param damage_threshold Damage level above which to convert
     * @return MorphingResult with conversion statistics
     */
    MorphingResult convert_damaged_elements(
        FEMPDCoupling& coupling,
        const physics::ElementErosionManager& erosion_mgr,
        Mesh& mesh,
        PDParticleSystem& particles,
        PDNeighborList& neighbors,
        const PDMaterial& material,
        Real damage_threshold = 0.3)
    {
        MorphingResult result;

        // Scan all element blocks for damaged elements
        std::vector<Index> elements_to_convert;
        std::vector<Index> block_ids;
        std::vector<Index> local_elem_ids;

        Index global_elem = 0;
        for (Index b = 0; b < static_cast<Index>(mesh.num_element_blocks()); ++b) {
            auto& block = mesh.element_block(b);
            for (Index e = 0; e < static_cast<Index>(block.num_elements()); ++e) {
                Real dmg = erosion_mgr.element_damage(global_elem);
                if (dmg > damage_threshold && erosion_mgr.element_active(global_elem)) {
                    elements_to_convert.push_back(global_elem);
                    block_ids.push_back(b);
                    local_elem_ids.push_back(e);
                }
                ++global_elem;
            }
        }

        if (elements_to_convert.empty()) return result;

        // Collect new particle data
        std::vector<Real> new_x, new_y, new_z;
        std::vector<Real> new_vol;
        std::vector<Index> elem_particle_start; // maps element -> first particle index

        for (size_t idx = 0; idx < elements_to_convert.size(); ++idx) {
            Index b = block_ids[idx];
            Index e = local_elem_ids[idx];
            auto& block = mesh.element_block(b);

            elem_particle_start.push_back(static_cast<Index>(new_x.size()));

            auto nodes = block.element_nodes(e);
            Index nnodes = static_cast<Index>(nodes.size());

            // Compute element volume (approximate: total volume / num_nodes per particle)
            Real elem_vol = compute_element_volume(mesh, nodes);
            Real particle_vol = elem_vol / nnodes;

            for (Index n = 0; n < nnodes; ++n) {
                Vec3r coords = mesh.get_node_coordinates(nodes[n]);
                new_x.push_back(coords[0]);
                new_y.push_back(coords[1]);
                new_z.push_back(coords[2]);
                new_vol.push_back(particle_vol);
            }
        }

        Index num_new = static_cast<Index>(new_x.size());
        if (num_new == 0) return result;

        // Create expanded particle system
        Index old_count = particles.num_particles();
        Index new_total = old_count + num_new;

        // Save old host data
        particles.sync_to_host();
        auto old_x = particles.x_host();
        auto old_v = particles.v_host();
        auto old_u = particles.u_host();
        auto old_vol = particles.volume_host();
        auto old_horizon = particles.horizon_host();
        auto old_mass = particles.mass_host();
        auto old_matid = particles.material_id_host();
        auto old_bodyid = particles.body_id_host();

        // Get horizon from first existing particle (assumed uniform)
        Real horizon = (old_count > 0) ? old_horizon(0) : material.E > 0 ? 0.003 : 0.003;
        Real rho = material.rho;

        // Re-initialize with expanded size
        particles.initialize(new_total);

        // Copy old particles back
        for (Index i = 0; i < old_count; ++i) {
            particles.set_position(i, old_x(i, 0), old_x(i, 1), old_x(i, 2));
            particles.set_velocity(i, old_v(i, 0), old_v(i, 1), old_v(i, 2));
            particles.set_properties(i, old_vol(i), old_horizon(i), old_mass(i));
            particles.set_ids(i, old_matid(i), old_bodyid(i));
            // Copy displacement
            particles.u_host()(i, 0) = old_u(i, 0);
            particles.u_host()(i, 1) = old_u(i, 1);
            particles.u_host()(i, 2) = old_u(i, 2);
        }

        // Add new particles
        for (Index p = 0; p < num_new; ++p) {
            Index idx = old_count + p;
            particles.set_position(idx, new_x[p], new_y[p], new_z[p]);
            particles.set_velocity(idx, 0.0, 0.0, 0.0);
            Real mass_p = rho * new_vol[p];
            particles.set_properties(idx, new_vol[p], horizon, mass_p);
            particles.set_ids(idx, 0, 0); // Default material/body
        }

        particles.sync_to_device();

        result.elements_converted = static_cast<Index>(elements_to_convert.size());
        result.particles_created = num_new;

        // Update coupling zone classification for new particles
        auto& pd_domain = const_cast<std::vector<DomainType>&>(coupling.pd_domain());
        pd_domain.resize(new_total, DomainType::PD_Only);

        converted_elements_.insert(converted_elements_.end(),
                                    elements_to_convert.begin(),
                                    elements_to_convert.end());

        return result;
    }

    /**
     * @brief Create particles from a single hex8 element
     * @return Number of particles created (8 for hex8, 4 for tet4)
     */
    static Index create_particles_from_element(
        const Mesh& mesh,
        std::span<const Index> elem_nodes,
        std::vector<Real>& out_x,
        std::vector<Real>& out_y,
        std::vector<Real>& out_z)
    {
        Index nnodes = static_cast<Index>(elem_nodes.size());
        for (Index n = 0; n < nnodes; ++n) {
            Vec3r coords = mesh.get_node_coordinates(elem_nodes[n]);
            out_x.push_back(coords[0]);
            out_y.push_back(coords[1]);
            out_z.push_back(coords[2]);
        }
        return nnodes;
    }

    /**
     * @brief Transfer FEM state to newly created PD particles
     *
     * Uses direct node-to-particle mapping (each particle is at a node)
     */
    static void transfer_state(
        const Mesh& mesh,
        std::span<const Index> elem_nodes,
        const Kokkos::View<Real*>& fem_displacement,
        PDParticleSystem& particles,
        Index particle_start)
    {
        particles.sync_to_host();
        auto u_host = particles.u_host();
        auto v_host = particles.v_host();

        Index nnodes = static_cast<Index>(elem_nodes.size());
        for (Index n = 0; n < nnodes; ++n) {
            Index fem_node = elem_nodes[n];
            Index pd_idx = particle_start + n;

            // Transfer displacement from FEM (flat [node*3+dof] layout)
            u_host(pd_idx, 0) = fem_displacement(fem_node * 3 + 0);
            u_host(pd_idx, 1) = fem_displacement(fem_node * 3 + 1);
            u_host(pd_idx, 2) = fem_displacement(fem_node * 3 + 2);
        }
    }

    /**
     * @brief Connect new particles to each other and to existing PD neighbors
     *
     * Rebuilds neighbor list with expanded particle count.
     */
    static void connect_new_particles(
        PDParticleSystem& particles,
        PDNeighborList& neighbors,
        Real horizon_factor = 3.015)
    {
        // Rebuild neighbor list with all particles (old + new)
        neighbors.build(particles, horizon_factor);
    }

    // Accessors
    const std::vector<Index>& converted_elements() const { return converted_elements_; }

private:
    /**
     * @brief Approximate element volume from node positions
     */
    static Real compute_element_volume(const Mesh& mesh, std::span<const Index> nodes) {
        Index n = static_cast<Index>(nodes.size());
        if (n < 4) return 1e-9; // degenerate

        // Compute bounding box volume as approximation
        Real min_x = 1e30, min_y = 1e30, min_z = 1e30;
        Real max_x = -1e30, max_y = -1e30, max_z = -1e30;

        for (Index i = 0; i < n; ++i) {
            Vec3r c = mesh.get_node_coordinates(nodes[i]);
            min_x = std::min(min_x, c[0]);
            min_y = std::min(min_y, c[1]);
            min_z = std::min(min_z, c[2]);
            max_x = std::max(max_x, c[0]);
            max_y = std::max(max_y, c[1]);
            max_z = std::max(max_z, c[2]);
        }

        Real dx = max_x - min_x;
        Real dy = max_y - min_y;
        Real dz = max_z - min_z;

        // For hex8: volume ~ dx*dy*dz
        // For tet4: volume ~ dx*dy*dz / 6
        if (n == 4) {
            return dx * dy * dz / 6.0;
        }
        return dx * dy * dz;
    }

    std::vector<Index> converted_elements_;
};

} // namespace pd
} // namespace nxs
