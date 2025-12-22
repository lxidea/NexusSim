#pragma once

/**
 * @file fem_sph_coupling.hpp
 * @brief FEM-SPH coupling for fluid-structure interaction
 *
 * Implements coupling between FEM solid domain and SPH fluid domain:
 * - Surface detection for FEM-SPH interface
 * - Contact force computation between particles and solid surfaces
 * - Momentum and energy exchange
 * - Adaptive interface tracking
 *
 * Coupling approaches:
 * 1. Master-Slave: FEM surface nodes act as boundary particles for SPH
 * 2. Penalty Contact: Repulsive forces when particles approach FEM surfaces
 * 3. Ghost Particle: Create mirror particles inside solid for pressure BC
 *
 * References:
 * - Antoci et al. (2007) - Numerical simulation of fluid-structure interaction by SPH
 * - Yang et al. (2012) - Free-surface flow interactions with deformable structures using SPH-FEM
 */

#include <nexussim/core/core.hpp>
#include <nexussim/data/mesh.hpp>
#include <nexussim/sph/sph_solver.hpp>
#include <Kokkos_Core.hpp>
#include <vector>
#include <array>
#include <unordered_set>
#include <cmath>

namespace nxs {
namespace sph {

// ============================================================================
// FEM Surface Representation
// ============================================================================

/**
 * @brief A triangular surface facet from FEM mesh
 */
struct SurfaceFacet {
    Index nodes[3];           ///< Node indices (from FEM mesh)
    Index element_id;         ///< Parent element ID
    Real area;                ///< Facet area
    std::array<Real, 3> normal; ///< Outward normal (into fluid)
    std::array<Real, 3> centroid; ///< Facet centroid
};

/**
 * @brief FEM surface mesh for coupling
 */
class FEMSurface {
public:
    FEMSurface() = default;

    /**
     * @brief Extract surface facets from FEM mesh
     * @param mesh The FEM mesh
     * @param surface_node_ids Nodes on the fluid-structure interface
     */
    void extract_from_mesh(const Mesh& mesh, const std::vector<Index>& surface_node_ids) {
        surface_nodes_.clear();
        facets_.clear();

        // Store surface nodes as set for quick lookup
        std::unordered_set<Index> surface_set(surface_node_ids.begin(), surface_node_ids.end());
        surface_nodes_ = surface_node_ids;

        // Get node coordinates
        node_coords_.resize(mesh.num_nodes() * 3);
        for (size_t i = 0; i < mesh.num_nodes(); ++i) {
            auto coord = mesh.get_node_coordinates(i);
            node_coords_[i * 3 + 0] = coord[0];
            node_coords_[i * 3 + 1] = coord[1];
            node_coords_[i * 3 + 2] = coord[2];
        }

        // Process each element block
        size_t global_elem_id = 0;
        for (const auto& block : mesh.element_blocks()) {
            size_t nodes_per_elem = block.num_nodes_per_elem;

            for (size_t e = 0; e < block.num_elements(); ++e) {
                // Get element nodes
                auto elem_nodes_span = block.element_nodes(e);
                std::vector<Index> elem_nodes(elem_nodes_span.begin(), elem_nodes_span.end());

                // Check for surface faces based on element type
                extract_surface_faces(global_elem_id, elem_nodes, surface_set);
                ++global_elem_id;
            }
        }

        // Compute facet properties
        update_facet_geometry();
    }

    /**
     * @brief Update facet geometry after FEM deformation
     * @param new_coords Updated node coordinates [x0,y0,z0,x1,y1,z1,...]
     */
    void update_node_positions(const std::vector<Real>& new_coords) {
        node_coords_ = new_coords;
        update_facet_geometry();
    }

    /**
     * @brief Update from displacement field
     */
    void apply_displacements(const Real* disp, size_t num_nodes) {
        for (size_t i = 0; i < num_nodes && i * 3 + 2 < node_coords_.size(); ++i) {
            node_coords_[i * 3 + 0] += disp[i * 3 + 0];
            node_coords_[i * 3 + 1] += disp[i * 3 + 1];
            node_coords_[i * 3 + 2] += disp[i * 3 + 2];
        }
        update_facet_geometry();
    }

    const std::vector<SurfaceFacet>& facets() const { return facets_; }
    const std::vector<Index>& surface_nodes() const { return surface_nodes_; }
    size_t num_facets() const { return facets_.size(); }

    /**
     * @brief Get node coordinates
     */
    std::array<Real, 3> get_node_coord(Index node_id) const {
        return {
            node_coords_[node_id * 3 + 0],
            node_coords_[node_id * 3 + 1],
            node_coords_[node_id * 3 + 2]
        };
    }

private:
    void extract_surface_faces(size_t elem_id,
                               const std::vector<Index>& elem_nodes,
                               const std::unordered_set<Index>& surface_set) {
        size_t n = elem_nodes.size();

        // Hex8 faces (6 faces, 4 nodes each -> split into triangles)
        if (n == 8) {
            static const int hex_faces[6][4] = {
                {0, 3, 2, 1}, {4, 5, 6, 7}, // bottom, top
                {0, 1, 5, 4}, {2, 3, 7, 6}, // front, back
                {0, 4, 7, 3}, {1, 2, 6, 5}  // left, right
            };

            for (int f = 0; f < 6; ++f) {
                bool is_surface = true;
                for (int i = 0; i < 4; ++i) {
                    if (surface_set.find(elem_nodes[hex_faces[f][i]]) == surface_set.end()) {
                        is_surface = false;
                        break;
                    }
                }
                if (is_surface) {
                    // Split quad into two triangles
                    SurfaceFacet tri1, tri2;
                    tri1.nodes[0] = elem_nodes[hex_faces[f][0]];
                    tri1.nodes[1] = elem_nodes[hex_faces[f][1]];
                    tri1.nodes[2] = elem_nodes[hex_faces[f][2]];
                    tri1.element_id = elem_id;
                    facets_.push_back(tri1);

                    tri2.nodes[0] = elem_nodes[hex_faces[f][0]];
                    tri2.nodes[1] = elem_nodes[hex_faces[f][2]];
                    tri2.nodes[2] = elem_nodes[hex_faces[f][3]];
                    tri2.element_id = elem_id;
                    facets_.push_back(tri2);
                }
            }
        }
        // Tet4 faces (4 faces, 3 nodes each)
        else if (n == 4) {
            static const int tet_faces[4][3] = {
                {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2}
            };

            for (int f = 0; f < 4; ++f) {
                bool is_surface = true;
                for (int i = 0; i < 3; ++i) {
                    if (surface_set.find(elem_nodes[tet_faces[f][i]]) == surface_set.end()) {
                        is_surface = false;
                        break;
                    }
                }
                if (is_surface) {
                    SurfaceFacet tri;
                    tri.nodes[0] = elem_nodes[tet_faces[f][0]];
                    tri.nodes[1] = elem_nodes[tet_faces[f][1]];
                    tri.nodes[2] = elem_nodes[tet_faces[f][2]];
                    tri.element_id = elem_id;
                    facets_.push_back(tri);
                }
            }
        }
        // For other elements, try to extract triangular faces from first 3-4 nodes
        else if (n >= 3) {
            // Check if first 3 nodes are on surface
            bool is_surface = true;
            for (int i = 0; i < 3; ++i) {
                if (surface_set.find(elem_nodes[i]) == surface_set.end()) {
                    is_surface = false;
                    break;
                }
            }
            if (is_surface) {
                SurfaceFacet tri;
                tri.nodes[0] = elem_nodes[0];
                tri.nodes[1] = elem_nodes[1];
                tri.nodes[2] = elem_nodes[2];
                tri.element_id = elem_id;
                facets_.push_back(tri);
            }
        }
    }

    void update_facet_geometry() {
        for (auto& facet : facets_) {
            auto p0 = get_node_coord(facet.nodes[0]);
            auto p1 = get_node_coord(facet.nodes[1]);
            auto p2 = get_node_coord(facet.nodes[2]);

            // Edge vectors
            Real e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
            Real e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};

            // Normal = e1 x e2
            Real nx = e1[1] * e2[2] - e1[2] * e2[1];
            Real ny = e1[2] * e2[0] - e1[0] * e2[2];
            Real nz = e1[0] * e2[1] - e1[1] * e2[0];

            Real area2 = std::sqrt(nx * nx + ny * ny + nz * nz);
            facet.area = 0.5 * area2;

            if (area2 > 1e-20) {
                facet.normal = {nx / area2, ny / area2, nz / area2};
            } else {
                facet.normal = {0, 0, 1};
            }

            // Centroid
            facet.centroid = {
                (p0[0] + p1[0] + p2[0]) / 3.0,
                (p0[1] + p1[1] + p2[1]) / 3.0,
                (p0[2] + p1[2] + p2[2]) / 3.0
            };
        }
    }

    std::vector<Index> surface_nodes_;
    std::vector<SurfaceFacet> facets_;
    std::vector<Real> node_coords_;
};

// ============================================================================
// Coupling Types
// ============================================================================

enum class CouplingType {
    Penalty,        ///< Penalty-based contact
    GhostParticle,  ///< Ghost particle method
    DirectForce     ///< Direct force exchange
};

/**
 * @brief Contact parameters for FEM-SPH coupling
 */
struct CouplingParameters {
    Real penalty_stiffness = 1e8;   ///< Penalty stiffness for contact
    Real damping_ratio = 0.1;       ///< Contact damping
    Real friction_coef = 0.0;       ///< Friction coefficient
    Real contact_distance = 0.0;    ///< Contact detection distance (0 = auto from h)
    bool enable_friction = false;   ///< Enable friction forces
    bool enable_adhesion = false;   ///< Enable adhesion (negative contact)
    Real adhesion_strength = 0.0;   ///< Adhesion force magnitude
};

// ============================================================================
// FEM-SPH Coupling Interface
// ============================================================================

/**
 * @brief Handles coupling between FEM solid and SPH fluid domains
 */
class FEMSPHCoupling {
public:
    FEMSPHCoupling() = default;

    /**
     * @brief Initialize coupling with FEM mesh and surface nodes
     */
    void initialize(const Mesh& fem_mesh,
                    const std::vector<Index>& surface_nodes,
                    SPHSolver& sph_solver) {
        sph_solver_ = &sph_solver;
        surface_.extract_from_mesh(fem_mesh, surface_nodes);
        smoothing_length_ = sph_solver.smoothing_length();

        // Set contact distance based on smoothing length if not specified
        if (params_.contact_distance <= 0) {
            params_.contact_distance = 2.0 * smoothing_length_;
        }

        // Allocate force arrays
        size_t num_surface_nodes = surface_nodes.size();
        surface_forces_.resize(num_surface_nodes * 3, 0.0);

        size_t num_particles = sph_solver.num_particles();
        particle_contact_forces_.resize(num_particles * 3, 0.0);

        initialized_ = true;
    }

    /**
     * @brief Set coupling parameters
     */
    void set_parameters(const CouplingParameters& params) {
        params_ = params;
    }

    /**
     * @brief Set coupling type
     */
    void set_coupling_type(CouplingType type) {
        coupling_type_ = type;
    }

    /**
     * @brief Update FEM surface positions from displacement
     */
    void update_fem_surface(const Real* displacement) {
        if (!initialized_) return;
        surface_.apply_displacements(displacement, surface_.surface_nodes().size());
    }

    /**
     * @brief Compute coupling forces for current configuration
     */
    void compute_coupling_forces() {
        if (!initialized_ || !sph_solver_) return;

        // Clear previous forces
        std::fill(surface_forces_.begin(), surface_forces_.end(), 0.0);
        std::fill(particle_contact_forces_.begin(), particle_contact_forces_.end(), 0.0);

        switch (coupling_type_) {
            case CouplingType::Penalty:
                compute_penalty_forces();
                break;
            case CouplingType::GhostParticle:
                compute_ghost_particle_forces();
                break;
            case CouplingType::DirectForce:
                compute_direct_forces();
                break;
        }
    }

    /**
     * @brief Get forces on FEM surface nodes
     * @return Force array [fx0, fy0, fz0, fx1, fy1, fz1, ...]
     */
    const std::vector<Real>& get_fem_forces() const {
        return surface_forces_;
    }

    /**
     * @brief Get contact forces on SPH particles
     * @return Force array [fx0, fy0, fz0, ...]
     */
    const std::vector<Real>& get_particle_forces() const {
        return particle_contact_forces_;
    }

    /**
     * @brief Apply contact forces to SPH solver accelerations
     */
    void apply_particle_forces() {
        if (!sph_solver_) return;

        // Note: Would need to modify SPHSolver to expose acceleration views
        // This is a placeholder for the interface
    }

    /**
     * @brief Get number of active contacts
     */
    size_t num_contacts() const { return num_contacts_; }

    /**
     * @brief Get total normal contact force
     */
    Real total_normal_force() const { return total_normal_force_; }

    /**
     * @brief Get total tangential (friction) force
     */
    Real total_tangent_force() const { return total_tangent_force_; }

    /**
     * @brief Get coupling statistics
     */
    void print_stats(std::ostream& os = std::cout) const {
        os << "=== FEM-SPH Coupling Statistics ===\n";
        os << "Surface facets: " << surface_.num_facets() << "\n";
        os << "Surface nodes: " << surface_.surface_nodes().size() << "\n";
        os << "Active contacts: " << num_contacts_ << "\n";
        os << "Total normal force: " << total_normal_force_ << " N\n";
        os << "Total tangent force: " << total_tangent_force_ << " N\n";
        os << "===================================\n";
    }

private:
    /**
     * @brief Penalty-based contact forces
     */
    void compute_penalty_forces() {
        if (!sph_solver_) return;

        auto pos_x = sph_solver_->positions_x();
        auto pos_y = sph_solver_->positions_y();
        auto pos_z = sph_solver_->positions_z();
        auto vel_x = sph_solver_->velocities_x();
        auto vel_y = sph_solver_->velocities_y();
        auto vel_z = sph_solver_->velocities_z();

        num_contacts_ = 0;
        total_normal_force_ = 0.0;
        total_tangent_force_ = 0.0;

        // For each particle, check distance to surface facets
        for (size_t p = 0; p < sph_solver_->num_particles(); ++p) {
            Real px = pos_x(p);
            Real py = pos_y(p);
            Real pz = pos_z(p);

            // Check each facet
            for (size_t f = 0; f < surface_.num_facets(); ++f) {
                const auto& facet = surface_.facets()[f];

                // Compute distance to facet plane
                Real dx = px - facet.centroid[0];
                Real dy = py - facet.centroid[1];
                Real dz = pz - facet.centroid[2];

                Real dist = dx * facet.normal[0] + dy * facet.normal[1] + dz * facet.normal[2];

                // Check if within contact distance and on correct side
                if (dist > 0 && dist < params_.contact_distance) {
                    // Check if projection is inside triangle
                    if (!point_in_triangle(px, py, pz, facet)) continue;

                    // Penetration depth
                    Real penetration = params_.contact_distance - dist;

                    // Normal force (penalty)
                    Real fn = params_.penalty_stiffness * penetration;

                    // Damping
                    Real vn = vel_x(p) * facet.normal[0] +
                              vel_y(p) * facet.normal[1] +
                              vel_z(p) * facet.normal[2];

                    Real damping = params_.damping_ratio * 2.0 *
                                   std::sqrt(params_.penalty_stiffness) * std::abs(vn);

                    fn += damping;

                    // Apply force to particle (push away from surface)
                    particle_contact_forces_[p * 3 + 0] += fn * facet.normal[0];
                    particle_contact_forces_[p * 3 + 1] += fn * facet.normal[1];
                    particle_contact_forces_[p * 3 + 2] += fn * facet.normal[2];

                    // Distribute reaction to surface nodes
                    distribute_force_to_nodes(facet, -fn * facet.normal[0],
                                              -fn * facet.normal[1],
                                              -fn * facet.normal[2]);

                    num_contacts_++;
                    total_normal_force_ += std::abs(fn);

                    // Friction
                    if (params_.enable_friction && params_.friction_coef > 0) {
                        compute_friction_force(p, facet, fn, vel_x(p), vel_y(p), vel_z(p));
                    }
                }
            }
        }
    }

    /**
     * @brief Ghost particle method for pressure boundary
     */
    void compute_ghost_particle_forces() {
        // Simplified implementation: just repel particles from surface
        compute_penalty_forces();

        // In a full implementation, this would:
        // 1. Create ghost particles mirrored across each surface facet
        // 2. Include ghost particles in SPH density summation
        // 3. Apply no-slip or free-slip velocity conditions
    }

    /**
     * @brief Direct force exchange based on pressure
     */
    void compute_direct_forces() {
        if (!sph_solver_) return;

        auto pos_x = sph_solver_->positions_x();
        auto pos_y = sph_solver_->positions_y();
        auto pos_z = sph_solver_->positions_z();
        auto pressure = sph_solver_->pressures();

        num_contacts_ = 0;
        total_normal_force_ = 0.0;

        // For each surface facet, compute pressure force from nearby particles
        for (size_t f = 0; f < surface_.num_facets(); ++f) {
            const auto& facet = surface_.facets()[f];

            Real pressure_sum = 0.0;
            Real weight_sum = 0.0;

            // Find particles near this facet
            for (size_t p = 0; p < sph_solver_->num_particles(); ++p) {
                Real dx = pos_x(p) - facet.centroid[0];
                Real dy = pos_y(p) - facet.centroid[1];
                Real dz = pos_z(p) - facet.centroid[2];

                Real dist = std::sqrt(dx * dx + dy * dy + dz * dz);

                if (dist < params_.contact_distance) {
                    // Kernel-weighted pressure contribution
                    Real q = dist / smoothing_length_;
                    Real w = (q < 2.0) ? (1.0 - q / 2.0) : 0.0;

                    pressure_sum += pressure(p) * w;
                    weight_sum += w;
                    num_contacts_++;
                }
            }

            if (weight_sum > 0) {
                Real avg_pressure = pressure_sum / weight_sum;

                // Pressure force = p * A * n
                Real fx = avg_pressure * facet.area * facet.normal[0];
                Real fy = avg_pressure * facet.area * facet.normal[1];
                Real fz = avg_pressure * facet.area * facet.normal[2];

                distribute_force_to_nodes(facet, fx, fy, fz);

                total_normal_force_ += std::abs(avg_pressure) * facet.area;
            }
        }
    }

    /**
     * @brief Check if point projects inside triangle
     */
    bool point_in_triangle(Real px, Real py, Real pz, const SurfaceFacet& facet) const {
        auto p0 = surface_.get_node_coord(facet.nodes[0]);
        auto p1 = surface_.get_node_coord(facet.nodes[1]);
        auto p2 = surface_.get_node_coord(facet.nodes[2]);

        // Project point onto triangle plane
        Real dx = px - facet.centroid[0];
        Real dy = py - facet.centroid[1];
        Real dz = pz - facet.centroid[2];

        Real dist = dx * facet.normal[0] + dy * facet.normal[1] + dz * facet.normal[2];

        Real proj_x = px - dist * facet.normal[0];
        Real proj_y = py - dist * facet.normal[1];
        Real proj_z = pz - dist * facet.normal[2];

        // Barycentric coordinates test
        Real v0[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
        Real v1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
        Real v2[3] = {proj_x - p0[0], proj_y - p0[1], proj_z - p0[2]};

        Real dot00 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
        Real dot01 = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
        Real dot02 = v0[0] * v2[0] + v0[1] * v2[1] + v0[2] * v2[2];
        Real dot11 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
        Real dot12 = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];

        Real inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
        Real u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        Real v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

        return (u >= 0) && (v >= 0) && (u + v <= 1);
    }

    /**
     * @brief Distribute force to triangle nodes using shape functions
     */
    void distribute_force_to_nodes(const SurfaceFacet& facet,
                                    Real fx, Real fy, Real fz) {
        // Equal distribution (1/3 each node)
        // More accurate: use barycentric coordinates of contact point

        for (int i = 0; i < 3; ++i) {
            Index node = facet.nodes[i];

            // Find node index in surface_nodes list
            const auto& surf_nodes = surface_.surface_nodes();
            auto it = std::find(surf_nodes.begin(), surf_nodes.end(), node);
            if (it != surf_nodes.end()) {
                size_t idx = std::distance(surf_nodes.begin(), it);
                surface_forces_[idx * 3 + 0] += fx / 3.0;
                surface_forces_[idx * 3 + 1] += fy / 3.0;
                surface_forces_[idx * 3 + 2] += fz / 3.0;
            }
        }
    }

    /**
     * @brief Compute friction force
     */
    void compute_friction_force(size_t particle_id, const SurfaceFacet& facet,
                                Real normal_force,
                                Real vx, Real vy, Real vz) {
        // Relative velocity tangent to surface
        Real vn = vx * facet.normal[0] + vy * facet.normal[1] + vz * facet.normal[2];

        Real vtx = vx - vn * facet.normal[0];
        Real vty = vy - vn * facet.normal[1];
        Real vtz = vz - vn * facet.normal[2];

        Real vt_mag = std::sqrt(vtx * vtx + vty * vty + vtz * vtz);

        if (vt_mag > 1e-10) {
            // Coulomb friction: Ft = mu * Fn
            Real ft = params_.friction_coef * std::abs(normal_force);

            // Friction direction opposes sliding
            Real ft_x = -ft * vtx / vt_mag;
            Real ft_y = -ft * vty / vt_mag;
            Real ft_z = -ft * vtz / vt_mag;

            particle_contact_forces_[particle_id * 3 + 0] += ft_x;
            particle_contact_forces_[particle_id * 3 + 1] += ft_y;
            particle_contact_forces_[particle_id * 3 + 2] += ft_z;

            distribute_force_to_nodes(facet, -ft_x, -ft_y, -ft_z);

            total_tangent_force_ += ft;
        }
    }

    // Coupling data
    FEMSurface surface_;
    SPHSolver* sph_solver_ = nullptr;
    CouplingType coupling_type_ = CouplingType::Penalty;
    CouplingParameters params_;
    Real smoothing_length_ = 0.01;
    bool initialized_ = false;

    // Force arrays
    std::vector<Real> surface_forces_;        ///< Forces on FEM nodes
    std::vector<Real> particle_contact_forces_; ///< Contact forces on particles

    // Statistics
    size_t num_contacts_ = 0;
    Real total_normal_force_ = 0.0;
    Real total_tangent_force_ = 0.0;
};

// ============================================================================
// Coupled Solver
// ============================================================================

/**
 * @brief Manages coupled FEM-SPH simulation
 */
class CoupledFEMSPHSolver {
public:
    CoupledFEMSPHSolver() = default;

    /**
     * @brief Initialize coupled solver
     */
    void initialize(Mesh& fem_mesh,
                    const std::vector<Index>& interface_nodes,
                    SPHSolver& sph_solver) {
        fem_mesh_ = &fem_mesh;
        sph_solver_ = &sph_solver;

        coupling_.initialize(fem_mesh, interface_nodes, sph_solver);
        interface_nodes_ = interface_nodes;

        // Allocate FEM state arrays
        size_t num_nodes = fem_mesh.num_nodes();
        fem_displacement_.resize(num_nodes * 3, 0.0);
        fem_velocity_.resize(num_nodes * 3, 0.0);
        fem_acceleration_.resize(num_nodes * 3, 0.0);

        initialized_ = true;
    }

    /**
     * @brief Perform one coupled timestep
     *
     * Uses staggered approach:
     * 1. Advance SPH (compute fluid forces on structure)
     * 2. Compute coupling forces
     * 3. Advance FEM (apply fluid forces)
     * 4. Update interface geometry
     */
    void step(Real dt) {
        if (!initialized_) return;

        // 1. SPH step (half)
        Real sph_dt = std::min(dt / 2.0, sph_solver_->compute_stable_dt());
        sph_solver_->step(sph_dt);

        // 2. Compute coupling forces
        coupling_.update_fem_surface(fem_displacement_.data());
        coupling_.compute_coupling_forces();

        // 3. Get forces on FEM nodes
        const auto& fem_forces = coupling_.get_fem_forces();

        // 4. FEM update would happen here
        // (integrate accelerations, update displacements)
        // This is a placeholder - actual FEM integration depends on the solver

        // 5. Update interface for second half of SPH step
        coupling_.update_fem_surface(fem_displacement_.data());

        // 6. Complete SPH step
        sph_solver_->step(sph_dt);

        time_ += dt;
    }

    /**
     * @brief Set coupling parameters
     */
    void set_coupling_parameters(const CouplingParameters& params) {
        coupling_.set_parameters(params);
    }

    /**
     * @brief Get current time
     */
    Real time() const { return time_; }

    /**
     * @brief Get coupling interface
     */
    FEMSPHCoupling& coupling() { return coupling_; }

    /**
     * @brief Print coupled solver statistics
     */
    void print_stats(std::ostream& os = std::cout) const {
        os << "=== Coupled FEM-SPH Solver ===\n";
        os << "Time: " << time_ << " s\n";
        os << "FEM nodes: " << (fem_mesh_ ? fem_mesh_->num_nodes() : 0) << "\n";
        os << "SPH particles: " << (sph_solver_ ? sph_solver_->num_particles() : 0) << "\n";
        os << "Interface nodes: " << interface_nodes_.size() << "\n";
        coupling_.print_stats(os);
    }

private:
    Mesh* fem_mesh_ = nullptr;
    SPHSolver* sph_solver_ = nullptr;
    FEMSPHCoupling coupling_;

    std::vector<Index> interface_nodes_;

    // FEM state (would typically come from FEM solver)
    std::vector<Real> fem_displacement_;
    std::vector<Real> fem_velocity_;
    std::vector<Real> fem_acceleration_;

    Real time_ = 0.0;
    bool initialized_ = false;
};

} // namespace sph
} // namespace nxs
