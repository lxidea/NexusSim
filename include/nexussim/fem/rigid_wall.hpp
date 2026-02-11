#pragma once

/**
 * @file rigid_wall.hpp
 * @brief Rigid wall contact for crash simulation
 *
 * Supports:
 * - Planar walls (infinite plane)
 * - Cylindrical walls
 * - Spherical walls
 * - Moving walls (prescribed velocity)
 *
 * Uses penalty-based contact with optional friction.
 *
 * Reference: OpenRadioss /engine/source/constraints/general/rwall
 */

#include <nexussim/core/types.hpp>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>

namespace nxs {
namespace fem {

// ============================================================================
// Rigid Wall Types
// ============================================================================

enum class RigidWallType {
    Planar,       ///< Infinite plane
    Cylindrical,  ///< Infinite cylinder
    Spherical,    ///< Sphere
    Moving        ///< Planar with prescribed velocity
};

// ============================================================================
// Rigid Wall Configuration
// ============================================================================

struct RigidWallConfig {
    RigidWallType type;
    int id;
    std::string name;

    Real origin[3];     ///< Wall origin point
    Real normal[3];     ///< Wall normal (outward, pointing away from material)
    Real radius;        ///< Radius for cylindrical/spherical walls
    Real velocity[3];   ///< Wall velocity (for moving walls)

    Real friction;      ///< Friction coefficient
    Real penalty_scale; ///< Penalty stiffness scale factor

    RigidWallConfig()
        : type(RigidWallType::Planar), id(0), radius(0.0)
        , friction(0.0), penalty_scale(1.0) {
        origin[0] = origin[1] = origin[2] = 0.0;
        normal[0] = 0.0; normal[1] = 0.0; normal[2] = 1.0;
        velocity[0] = velocity[1] = velocity[2] = 0.0;
    }
};

// ============================================================================
// Rigid Wall Contact
// ============================================================================

class RigidWallContact {
public:
    RigidWallContact() : penalty_stiffness_(1.0e10) {}

    void set_penalty_stiffness(Real k) { penalty_stiffness_ = k; }
    Real penalty_stiffness() const { return penalty_stiffness_; }

    // --- Configuration ---

    RigidWallConfig& add_wall(RigidWallType type, int id = 0) {
        walls_.emplace_back();
        auto& w = walls_.back();
        w.type = type;
        w.id = id;
        return w;
    }

    std::size_t num_walls() const { return walls_.size(); }
    RigidWallConfig& wall(std::size_t i) { return walls_[i]; }
    const RigidWallConfig& wall(std::size_t i) const { return walls_[i]; }

    // --- Contact Detection and Force Computation ---

    /**
     * @brief Compute rigid wall contact forces for all nodes
     * @param num_nodes Number of nodes
     * @param positions Node positions [3*num_nodes]
     * @param velocities Node velocities [3*num_nodes]
     * @param masses Node masses [num_nodes]
     * @param forces Output forces [3*num_nodes] (accumulated)
     * @param dt Time step
     */
    void compute_forces(std::size_t num_nodes,
                        const Real* positions,
                        const Real* velocities,
                        const Real* masses,
                        Real* forces,
                        Real dt) {
        for (const auto& wall : walls_) {
            switch (wall.type) {
                case RigidWallType::Planar:
                case RigidWallType::Moving:
                    compute_planar_forces(wall, num_nodes, positions, velocities,
                                          masses, forces, dt);
                    break;
                case RigidWallType::Cylindrical:
                    compute_cylindrical_forces(wall, num_nodes, positions, velocities,
                                               masses, forces, dt);
                    break;
                case RigidWallType::Spherical:
                    compute_spherical_forces(wall, num_nodes, positions, velocities,
                                             masses, forces, dt);
                    break;
            }
        }
    }

    /**
     * @brief Update moving wall positions
     */
    void update_walls(Real dt) {
        for (auto& wall : walls_) {
            if (wall.type == RigidWallType::Moving) {
                wall.origin[0] += wall.velocity[0] * dt;
                wall.origin[1] += wall.velocity[1] * dt;
                wall.origin[2] += wall.velocity[2] * dt;
            }
        }
    }

    struct WallStats {
        std::size_t active_contacts;
        Real max_penetration;
        Real total_normal_force;
    };

    WallStats get_stats() const { return last_stats_; }

    void print_summary() const {
        std::cout << "Rigid Wall Contact: " << walls_.size() << " walls\n";
        for (const auto& w : walls_) {
            const char* type_str = "Unknown";
            switch (w.type) {
                case RigidWallType::Planar: type_str = "Planar"; break;
                case RigidWallType::Cylindrical: type_str = "Cylindrical"; break;
                case RigidWallType::Spherical: type_str = "Spherical"; break;
                case RigidWallType::Moving: type_str = "Moving Planar"; break;
            }
            std::cout << "  [" << w.id << "] " << type_str
                      << " origin=[" << w.origin[0] << "," << w.origin[1] << "," << w.origin[2]
                      << "] normal=[" << w.normal[0] << "," << w.normal[1] << "," << w.normal[2]
                      << "] friction=" << w.friction << "\n";
        }
    }

private:
    void compute_planar_forces(const RigidWallConfig& wall,
                                std::size_t num_nodes,
                                const Real* positions,
                                const Real* velocities,
                                const Real* masses,
                                Real* forces,
                                Real dt) {
        Real nx = wall.normal[0], ny = wall.normal[1], nz = wall.normal[2];
        Real nmag = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (nmag < 1.0e-30) return;
        nx /= nmag; ny /= nmag; nz /= nmag;

        Real k = penalty_stiffness_ * wall.penalty_scale;
        std::size_t contacts = 0;
        Real max_pen = 0.0;
        Real total_fn = 0.0;

        for (std::size_t i = 0; i < num_nodes; ++i) {
            Real px = positions[3*i+0] - wall.origin[0];
            Real py = positions[3*i+1] - wall.origin[1];
            Real pz = positions[3*i+2] - wall.origin[2];

            // For moving wall, adjust for wall velocity
            if (wall.type == RigidWallType::Moving) {
                px -= wall.velocity[0] * dt;
                py -= wall.velocity[1] * dt;
                pz -= wall.velocity[2] * dt;
            }

            // Signed distance to plane (positive = on normal side = safe)
            Real dist = px*nx + py*ny + pz*nz;

            if (dist < 0.0) {
                // Penetration detected
                Real penetration = -dist;
                if (penetration > max_pen) max_pen = penetration;
                contacts++;

                // Normal contact force (penalty)
                Real fn = k * penetration;
                total_fn += fn;
                forces[3*i+0] += fn * nx;
                forces[3*i+1] += fn * ny;
                forces[3*i+2] += fn * nz;

                // Friction (Coulomb)
                if (wall.friction > 0.0) {
                    // Tangential velocity
                    Real vn = velocities[3*i+0]*nx + velocities[3*i+1]*ny + velocities[3*i+2]*nz;
                    Real vt[3];
                    vt[0] = velocities[3*i+0] - vn*nx;
                    vt[1] = velocities[3*i+1] - vn*ny;
                    vt[2] = velocities[3*i+2] - vn*nz;

                    Real vt_mag = std::sqrt(vt[0]*vt[0] + vt[1]*vt[1] + vt[2]*vt[2]);
                    if (vt_mag > 1.0e-20) {
                        Real ff = wall.friction * fn;
                        // Limit friction force by mass-based bound
                        Real ff_max = masses[i] * vt_mag / dt;
                        if (ff > ff_max) ff = ff_max;

                        forces[3*i+0] -= ff * vt[0] / vt_mag;
                        forces[3*i+1] -= ff * vt[1] / vt_mag;
                        forces[3*i+2] -= ff * vt[2] / vt_mag;
                    }
                }
            }
        }

        last_stats_.active_contacts = contacts;
        last_stats_.max_penetration = max_pen;
        last_stats_.total_normal_force = total_fn;
    }

    void compute_cylindrical_forces(const RigidWallConfig& wall,
                                     std::size_t num_nodes,
                                     const Real* positions,
                                     const Real* /*velocities*/,
                                     const Real* /*masses*/,
                                     Real* forces,
                                     Real /*dt*/) {
        Real ax = wall.normal[0], ay = wall.normal[1], az = wall.normal[2];
        Real amag = std::sqrt(ax*ax + ay*ay + az*az);
        if (amag < 1.0e-30) return;
        ax /= amag; ay /= amag; az /= amag;

        Real R = wall.radius;
        Real k = penalty_stiffness_ * wall.penalty_scale;

        for (std::size_t i = 0; i < num_nodes; ++i) {
            Real dx = positions[3*i+0] - wall.origin[0];
            Real dy = positions[3*i+1] - wall.origin[1];
            Real dz = positions[3*i+2] - wall.origin[2];

            // Project onto axis
            Real dot = dx*ax + dy*ay + dz*az;
            Real px = dx - dot*ax;
            Real py = dy - dot*ay;
            Real pz = dz - dot*az;

            Real dist = std::sqrt(px*px + py*py + pz*pz);

            if (dist < R && dist > 1.0e-20) {
                // Inside cylinder - push outward
                Real penetration = R - dist;
                Real nx = px/dist, ny_ = py/dist, nz_ = pz/dist;
                Real fn = k * penetration;

                forces[3*i+0] += fn * nx;
                forces[3*i+1] += fn * ny_;
                forces[3*i+2] += fn * nz_;
            }
        }
    }

    void compute_spherical_forces(const RigidWallConfig& wall,
                                   std::size_t num_nodes,
                                   const Real* positions,
                                   const Real* /*velocities*/,
                                   const Real* /*masses*/,
                                   Real* forces,
                                   Real /*dt*/) {
        Real R = wall.radius;
        Real k = penalty_stiffness_ * wall.penalty_scale;

        for (std::size_t i = 0; i < num_nodes; ++i) {
            Real dx = positions[3*i+0] - wall.origin[0];
            Real dy = positions[3*i+1] - wall.origin[1];
            Real dz = positions[3*i+2] - wall.origin[2];

            Real dist = std::sqrt(dx*dx + dy*dy + dz*dz);

            if (dist < R && dist > 1.0e-20) {
                Real penetration = R - dist;
                Real nx = dx/dist, ny = dy/dist, nz = dz/dist;
                Real fn = k * penetration;

                forces[3*i+0] += fn * nx;
                forces[3*i+1] += fn * ny;
                forces[3*i+2] += fn * nz;
            }
        }
    }

    std::vector<RigidWallConfig> walls_;
    Real penalty_stiffness_;
    WallStats last_stats_ = {0, 0.0, 0.0};
};

} // namespace fem
} // namespace nxs
