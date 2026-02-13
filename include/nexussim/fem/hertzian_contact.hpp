#pragma once

/**
 * @file hertzian_contact.hpp
 * @brief Hertzian contact model with material-derived stiffness
 *
 * Implements nonlinear Hertz contact theory:
 *   F = (4/3) * E* * sqrt(R*) * delta^(3/2)
 *
 * Features:
 * - Effective properties computed from material E, nu, G
 * - Hunt-Crossley and Flores damping models with COR
 * - Mindlin tangential stiffness: k_t = 8*G*·a
 * - Sphere-sphere, sphere-plane, cylinder-plane geometries
 * - Spatial hash broad-phase for efficient detection
 *
 * Reference: Hertz (1882), Hunt & Crossley (1975), Flores et al. (2011)
 */

#include <nexussim/core/types.hpp>
#include <nexussim/physics/material.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace nxs {
namespace fem {

// ============================================================================
// Contact geometry types
// ============================================================================

enum class ContactGeometry {
    SphereSphere,    ///< Two spheres in contact
    SpherePlane,     ///< Sphere on infinite plane
    CylinderPlane,   ///< Cylinder on plane (line contact)
    General          ///< General curvature (uses effective R)
};

// ============================================================================
// Damping model selection
// ============================================================================

enum class HertzDampingModel {
    None,            ///< Pure elastic Hertz (no energy loss)
    HuntCrossley,    ///< c = 3(1-e)/(2·v₀)  — valid for e close to 1
    Flores,          ///< c = 8(1-e)/(5·e·v₀) — valid for full range of e
    LankaraniNikravesh  ///< c = 3k(1-e²)/(4·v₀) — valid for e > 0.5
};

// ============================================================================
// Hertzian effective properties
// ============================================================================

struct HertzianProperties {
    Real E_star;        ///< Effective elastic modulus
    Real G_star;        ///< Effective shear modulus
    Real R_star;        ///< Effective radius
    Real m_star;        ///< Effective mass
    Real restitution;   ///< Coefficient of restitution [0,1]
    ContactGeometry geometry;

    HertzianProperties()
        : E_star(0.0), G_star(0.0), R_star(0.0), m_star(0.0)
        , restitution(1.0), geometry(ContactGeometry::SphereSphere) {}

    /**
     * @brief Compute effective properties from two material sets
     * @param mat1, mat2 Material properties (must have E, nu, G set)
     * @param R1, R2 Radii of curvature (use 1e30 for plane)
     */
    static HertzianProperties from_materials(
            const physics::MaterialProperties& mat1,
            const physics::MaterialProperties& mat2,
            Real R1, Real R2, Real e = 1.0) {
        HertzianProperties hp;
        // E* = 1 / ((1-nu1^2)/E1 + (1-nu2^2)/E2)
        Real inv_E = (1.0 - mat1.nu * mat1.nu) / mat1.E
                   + (1.0 - mat2.nu * mat2.nu) / mat2.E;
        hp.E_star = 1.0 / inv_E;

        // G* = 1 / ((2-nu1)/G1 + (2-nu2)/G2)
        Real G1 = mat1.G > 0.0 ? mat1.G : mat1.E / (2.0 * (1.0 + mat1.nu));
        Real G2 = mat2.G > 0.0 ? mat2.G : mat2.E / (2.0 * (1.0 + mat2.nu));
        Real inv_G = (2.0 - mat1.nu) / G1 + (2.0 - mat2.nu) / G2;
        hp.G_star = 1.0 / inv_G;

        // R* = 1 / (1/R1 + 1/R2)
        if (R2 > 1.0e20) {
            hp.R_star = R1;
            hp.geometry = ContactGeometry::SpherePlane;
        } else {
            hp.R_star = 1.0 / (1.0 / R1 + 1.0 / R2);
            hp.geometry = ContactGeometry::SphereSphere;
        }

        hp.restitution = e;
        hp.m_star = 0.0; // Set later from actual masses
        return hp;
    }

    /**
     * @brief Convenience: sphere on plane
     */
    static HertzianProperties sphere_plane(
            const physics::MaterialProperties& sphere_mat,
            const physics::MaterialProperties& plane_mat,
            Real radius, Real e = 1.0) {
        return from_materials(sphere_mat, plane_mat, radius, 1.0e30, e);
    }

    /**
     * @brief Hertz stiffness coefficient k_h such that F = k_h * delta^(3/2)
     */
    Real hertz_stiffness() const {
        return (4.0 / 3.0) * E_star * std::sqrt(R_star);
    }

    /**
     * @brief Contact radius at given penetration
     */
    Real contact_radius(Real delta) const {
        return std::sqrt(R_star * std::fabs(delta));
    }
};

// ============================================================================
// Contact configuration
// ============================================================================

struct HertzianContactConfig {
    Real contact_thickness;     ///< Detection gap threshold
    HertzDampingModel damping;  ///< Damping model
    Real search_radius;         ///< Broad-phase search radius
    Real bucket_size_factor;    ///< Spatial hash cell sizing
    bool enable_friction;       ///< Enable Mindlin tangential + friction
    Real friction_coefficient;  ///< Coulomb friction coefficient
    bool enable_tangential_stiffness; ///< Use Mindlin k_t = 8G*a
    Real min_approach_velocity; ///< Floor for v₀ to avoid singularity

    HertzianContactConfig()
        : contact_thickness(1.0e-4)
        , damping(HertzDampingModel::Flores)
        , search_radius(0.05)
        , bucket_size_factor(2.0)
        , enable_friction(true)
        , friction_coefficient(0.3)
        , enable_tangential_stiffness(true)
        , min_approach_velocity(1.0e-3) {}
};

// ============================================================================
// Per-contact info
// ============================================================================

struct HertzianContactInfo {
    Index slave_node;
    Index master_segment;
    Real gap;               ///< Gap (negative = penetration)
    Real delta;             ///< Penetration depth (positive)
    Real normal[3];         ///< Contact normal (slave → master)
    Real contact_point[3];  ///< Contact point on master surface
    Real phi[4];            ///< Shape functions on master segment
    Real normal_force;      ///< Scalar normal force magnitude
    Real damping_force;     ///< Scalar damping force
    Real tangent_force;     ///< Scalar tangential force
    Real contact_radius;    ///< Hertz contact radius a
    Real approach_velocity; ///< Normal approach velocity
    Real initial_approach_velocity; ///< v₀ at first contact (fixed for damping)
    // Friction state
    Real tangent_slip[3];   ///< Accumulated tangential slip
    bool sticking;          ///< In stick regime
    bool active;
};

// ============================================================================
// Main Hertzian contact class
// ============================================================================

class HertzianContact {
public:
    HertzianContact() : num_nodes_(0) {}

    // --- Configuration ---

    void set_config(const HertzianContactConfig& config) { config_ = config; }
    const HertzianContactConfig& config() const { return config_; }

    /**
     * @brief Set slave nodes with per-node radii and properties
     */
    void set_slave_nodes(const std::vector<Index>& nodes,
                         const std::vector<Real>& radii,
                         const std::vector<HertzianProperties>& props) {
        slave_nodes_ = nodes;
        slave_radii_ = radii;
        slave_props_ = props;
    }

    /**
     * @brief Set slave nodes with uniform properties
     */
    void set_slave_nodes(const std::vector<Index>& nodes,
                         Real radius,
                         const HertzianProperties& props) {
        slave_nodes_ = nodes;
        slave_radii_.assign(nodes.size(), radius);
        slave_props_.assign(nodes.size(), props);
    }

    /**
     * @brief Add master surface segments
     */
    void add_master_segments(const std::vector<Index>& connectivity,
                             int num_segments, int nodes_per_segment) {
        master_nps_ = nodes_per_segment;
        master_segments_.clear();
        for (int s = 0; s < num_segments; ++s) {
            MasterSeg seg;
            for (int n = 0; n < nodes_per_segment; ++n)
                seg.nodes[n] = connectivity[s * nodes_per_segment + n];
            for (int n = nodes_per_segment; n < 4; ++n)
                seg.nodes[n] = Index(-1);
            seg.num_nodes = nodes_per_segment;
            master_segments_.push_back(seg);
        }
    }

    /**
     * @brief Initialize spatial hash
     */
    void initialize(Index num_nodes) {
        num_nodes_ = num_nodes;
        contacts_.clear();
    }

    // --- Per-step operations ---

    /**
     * @brief Detect contacts between slave nodes and master segments
     */
    int detect_contacts(const Real* coords) {
        contacts_.clear();

        for (std::size_t si = 0; si < slave_nodes_.size(); ++si) {
            Index sn = slave_nodes_[si];
            Real sx = coords[3 * sn + 0];
            Real sy = coords[3 * sn + 1];
            Real sz = coords[3 * sn + 2];

            for (std::size_t mi = 0; mi < master_segments_.size(); ++mi) {
                const auto& seg = master_segments_[mi];

                // Compute segment centroid and normal
                Real cx = 0, cy = 0, cz = 0;
                for (int n = 0; n < seg.num_nodes; ++n) {
                    cx += coords[3 * seg.nodes[n] + 0];
                    cy += coords[3 * seg.nodes[n] + 1];
                    cz += coords[3 * seg.nodes[n] + 2];
                }
                cx /= seg.num_nodes; cy /= seg.num_nodes; cz /= seg.num_nodes;

                // Quick distance check
                Real dx = sx - cx, dy = sy - cy, dz = sz - cz;
                Real dist_sq = dx * dx + dy * dy + dz * dz;
                if (dist_sq > config_.search_radius * config_.search_radius)
                    continue;

                // Project slave node onto segment plane
                Real normal[3], proj[3];
                Real phi[4];
                Real gap = project_to_segment(coords, seg, sx, sy, sz,
                                              normal, proj, phi);

                // Hertzian: gap is from sphere surface, not center
                gap -= slave_radii_[si];

                // Check contact (gap < contact_thickness means approaching)
                if (gap < config_.contact_thickness) {
                    HertzianContactInfo ci;
                    ci.slave_node = sn;
                    ci.master_segment = mi;
                    ci.gap = gap;
                    ci.delta = std::max(-gap, 0.0);
                    for (int d = 0; d < 3; ++d) {
                        ci.normal[d] = normal[d];
                        ci.contact_point[d] = proj[d];
                    }
                    for (int n = 0; n < 4; ++n) ci.phi[n] = phi[n];
                    ci.normal_force = 0.0;
                    ci.damping_force = 0.0;
                    ci.tangent_force = 0.0;
                    ci.contact_radius = 0.0;
                    ci.approach_velocity = 0.0;
                    ci.initial_approach_velocity = 0.0;
                    for (int d = 0; d < 3; ++d) ci.tangent_slip[d] = 0.0;
                    ci.sticking = true;
                    ci.active = true;

                    // Carry over state from previous step
                    for (auto& prev : prev_contacts_) {
                        if (prev.slave_node == sn && prev.master_segment == mi) {
                            for (int d = 0; d < 3; ++d)
                                ci.tangent_slip[d] = prev.tangent_slip[d];
                            ci.sticking = prev.sticking;
                            ci.initial_approach_velocity = prev.initial_approach_velocity;
                            break;
                        }
                    }

                    contacts_.push_back(ci);
                }
            }
        }

        prev_contacts_ = contacts_;
        return (int)contacts_.size();
    }

    /**
     * @brief Compute Hertzian contact forces
     */
    void compute_forces(const Real* coords, const Real* velocities,
                        const Real* masses, Real dt, Real* forces) {
        stats_ = ContactStats();
        stats_.active_contacts = (int)contacts_.size();

        for (auto& ci : contacts_) {
            if (!ci.active || ci.delta <= 0.0) continue;

            Index sn = ci.slave_node;

            // Find slave properties
            std::size_t si = 0;
            for (std::size_t i = 0; i < slave_nodes_.size(); ++i)
                if (slave_nodes_[i] == sn) { si = i; break; }
            const auto& hp = slave_props_[si];

            // --- Normal approach velocity ---
            Real vn = 0.0;
            Real vs[3] = {velocities[3*sn+0], velocities[3*sn+1], velocities[3*sn+2]};

            // Master velocity (weighted by shape functions)
            Real vm[3] = {0, 0, 0};
            const auto& seg = master_segments_[ci.master_segment];
            for (int n = 0; n < seg.num_nodes; ++n) {
                Index mn = seg.nodes[n];
                for (int d = 0; d < 3; ++d)
                    vm[d] += ci.phi[n] * velocities[3*mn+d];
            }

            // Relative velocity (slave approaching master)
            Real vrel[3] = {vs[0]-vm[0], vs[1]-vm[1], vs[2]-vm[2]};
            vn = -(vrel[0]*ci.normal[0] + vrel[1]*ci.normal[1] + vrel[2]*ci.normal[2]);
            ci.approach_velocity = vn;

            // --- Hertz elastic force ---
            Real delta = ci.delta;
            Real k_h = hp.hertz_stiffness();
            Real F_elastic = k_h * delta * std::sqrt(delta); // k_h * delta^(3/2)

            // --- Damping force ---
            Real F_damp = 0.0;
            if (config_.damping != HertzDampingModel::None && delta > 0.0) {
                Real e = hp.restitution;

                // Use fixed initial approach velocity (set once at first contact)
                if (ci.initial_approach_velocity < config_.min_approach_velocity) {
                    ci.initial_approach_velocity = std::max(std::fabs(vn),
                                                            config_.min_approach_velocity);
                }
                Real v0 = ci.initial_approach_velocity;

                Real c = 0.0;
                switch (config_.damping) {
                    case HertzDampingModel::HuntCrossley:
                        c = 3.0 * (1.0 - e) / (2.0 * v0);
                        break;
                    case HertzDampingModel::Flores:
                        if (e > 1.0e-6)
                            c = 8.0 * (1.0 - e) / (5.0 * e * v0);
                        break;
                    case HertzDampingModel::LankaraniNikravesh:
                        c = 3.0 * k_h * (1.0 - e * e) / (4.0 * v0);
                        break;
                    default:
                        break;
                }

                // Hunt-Crossley style: F_damp = F_elastic * c * delta_dot
                // delta_dot = -vn (positive when approaching)
                if (config_.damping == HertzDampingModel::LankaraniNikravesh) {
                    // Lankarani: F_d = c * delta^(3/2) * delta_dot
                    F_damp = c * delta * std::sqrt(delta) * vn;
                } else {
                    // Hunt-Crossley / Flores: F_d = k_h * delta^(3/2) * c * delta_dot
                    F_damp = F_elastic * c * vn;
                }
            }

            // Total normal force (no attraction)
            Real F_total = std::max(F_elastic + F_damp, 0.0);

            ci.normal_force = F_elastic;
            ci.damping_force = F_damp;
            ci.contact_radius = hp.contact_radius(delta);

            // Apply normal force to slave node (push away from master)
            for (int d = 0; d < 3; ++d)
                forces[3*sn+d] += F_total * ci.normal[d];

            // Reaction on master nodes (via shape functions)
            for (int n = 0; n < seg.num_nodes; ++n) {
                Index mn = seg.nodes[n];
                for (int d = 0; d < 3; ++d)
                    forces[3*mn+d] -= ci.phi[n] * F_total * ci.normal[d];
            }

            // --- Tangential (Mindlin + Friction) ---
            if (config_.enable_friction && F_total > 0.0) {
                // Tangential relative velocity
                Real vt[3];
                Real vn_scalar = vrel[0]*ci.normal[0] + vrel[1]*ci.normal[1]
                               + vrel[2]*ci.normal[2];
                for (int d = 0; d < 3; ++d)
                    vt[d] = vrel[d] - vn_scalar * ci.normal[d];

                // Accumulate slip
                for (int d = 0; d < 3; ++d)
                    ci.tangent_slip[d] += vt[d] * dt;

                Real slip_mag = std::sqrt(ci.tangent_slip[0]*ci.tangent_slip[0]
                                        + ci.tangent_slip[1]*ci.tangent_slip[1]
                                        + ci.tangent_slip[2]*ci.tangent_slip[2]);

                if (slip_mag > 1.0e-20) {
                    // Mindlin tangential stiffness
                    Real k_t;
                    if (config_.enable_tangential_stiffness) {
                        Real a = ci.contact_radius;
                        k_t = 8.0 * hp.G_star * a;
                    } else {
                        k_t = 0.5 * k_h * std::sqrt(delta);
                    }

                    Real trial_force = k_t * slip_mag;
                    Real friction_limit = config_.friction_coefficient * F_total;

                    Real Ft;
                    if (trial_force <= friction_limit) {
                        // Stick
                        Ft = trial_force;
                        ci.sticking = true;
                    } else {
                        // Slip
                        Ft = friction_limit;
                        ci.sticking = false;
                        // Correct slip to match friction force
                        Real ratio = friction_limit / (k_t * slip_mag);
                        for (int d = 0; d < 3; ++d)
                            ci.tangent_slip[d] *= ratio;
                    }

                    ci.tangent_force = Ft;

                    // Apply tangential force
                    Real slip_dir[3];
                    for (int d = 0; d < 3; ++d)
                        slip_dir[d] = ci.tangent_slip[d] / slip_mag;

                    for (int d = 0; d < 3; ++d)
                        forces[3*sn+d] -= Ft * slip_dir[d];
                    for (int n = 0; n < seg.num_nodes; ++n) {
                        Index mn = seg.nodes[n];
                        for (int d = 0; d < 3; ++d)
                            forces[3*mn+d] += ci.phi[n] * Ft * slip_dir[d];
                    }
                }
            }

            // Track energy dissipated by damping
            if (F_damp != 0.0)
                stats_.total_energy_dissipated += std::fabs(F_damp * vn * dt);

            if (F_total > stats_.max_force) stats_.max_force = F_total;
            if (delta > stats_.max_penetration) stats_.max_penetration = delta;
        }
    }

    // --- Query ---

    struct ContactStats {
        int active_contacts;
        Real max_force;
        Real max_penetration;
        Real total_energy_dissipated;
        ContactStats() : active_contacts(0), max_force(0), max_penetration(0),
                          total_energy_dissipated(0) {}
    };

    const ContactStats& get_stats() const { return stats_; }
    const std::vector<HertzianContactInfo>& active_contacts() const { return contacts_; }
    int num_active_contacts() const { return (int)contacts_.size(); }

    void print_summary() const {
        std::cout << "HertzianContact: " << contacts_.size() << " active contacts\n";
        std::cout << "  Max force: " << stats_.max_force << " N\n";
        std::cout << "  Max penetration: " << stats_.max_penetration * 1000.0 << " mm\n";
        std::cout << "  Energy dissipated: " << stats_.total_energy_dissipated << " J\n";
    }

private:
    struct MasterSeg {
        Index nodes[4];
        int num_nodes;
    };

    HertzianContactConfig config_;
    std::vector<Index> slave_nodes_;
    std::vector<Real> slave_radii_;
    std::vector<HertzianProperties> slave_props_;
    std::vector<MasterSeg> master_segments_;
    int master_nps_ = 4;
    Index num_nodes_;
    std::vector<HertzianContactInfo> contacts_;
    std::vector<HertzianContactInfo> prev_contacts_;
    ContactStats stats_;

    /**
     * @brief Project a point onto a segment, return signed gap
     * @return Gap (negative = penetration)
     */
    Real project_to_segment(const Real* coords, const MasterSeg& seg,
                            Real px, Real py, Real pz,
                            Real* normal, Real* proj, Real* phi) {
        // Get segment node coordinates
        Real x[4][3];
        for (int n = 0; n < seg.num_nodes; ++n)
            for (int d = 0; d < 3; ++d)
                x[n][d] = coords[3 * seg.nodes[n] + d];

        // Compute segment normal (cross product of edges)
        Real e1[3], e2[3];
        for (int d = 0; d < 3; ++d) {
            e1[d] = x[1][d] - x[0][d];
            e2[d] = (seg.num_nodes >= 3) ? x[2][d] - x[0][d]
                                          : x[1][d] - x[0][d]; // degenerate
        }
        normal[0] = e1[1]*e2[2] - e1[2]*e2[1];
        normal[1] = e1[2]*e2[0] - e1[0]*e2[2];
        normal[2] = e1[0]*e2[1] - e1[1]*e2[0];
        Real nmag = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1]
                            + normal[2]*normal[2]);
        if (nmag < 1.0e-30) nmag = 1.0e-30;
        normal[0] /= nmag; normal[1] /= nmag; normal[2] /= nmag;

        // Signed distance from point to plane
        Real centroid[3] = {0, 0, 0};
        for (int n = 0; n < seg.num_nodes; ++n)
            for (int d = 0; d < 3; ++d)
                centroid[d] += x[n][d] / seg.num_nodes;

        Real dp[3] = {px - centroid[0], py - centroid[1], pz - centroid[2]};
        Real gap = dp[0]*normal[0] + dp[1]*normal[1] + dp[2]*normal[2];

        // Project onto plane
        proj[0] = px - gap * normal[0];
        proj[1] = py - gap * normal[1];
        proj[2] = pz - gap * normal[2];

        // Compute shape functions at projected point
        if (seg.num_nodes == 3) {
            // Barycentric coordinates for triangle
            Real v0[3], v1[3], v2[3];
            for (int d = 0; d < 3; ++d) {
                v0[d] = x[1][d] - x[0][d];
                v1[d] = x[2][d] - x[0][d];
                v2[d] = proj[d] - x[0][d];
            }
            Real d00 = v0[0]*v0[0]+v0[1]*v0[1]+v0[2]*v0[2];
            Real d01 = v0[0]*v1[0]+v0[1]*v1[1]+v0[2]*v1[2];
            Real d11 = v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2];
            Real d20 = v2[0]*v0[0]+v2[1]*v0[1]+v2[2]*v0[2];
            Real d21 = v2[0]*v1[0]+v2[1]*v1[1]+v2[2]*v1[2];
            Real denom = d00*d11 - d01*d01;
            if (std::fabs(denom) < 1.0e-30) denom = 1.0e-30;
            phi[1] = (d11*d20 - d01*d21) / denom;
            phi[2] = (d00*d21 - d01*d20) / denom;
            phi[0] = 1.0 - phi[1] - phi[2];
            phi[3] = 0.0;
        } else {
            // Bilinear quad — use centroid-based approximation
            Real lp[3] = {proj[0] - centroid[0], proj[1] - centroid[1],
                          proj[2] - centroid[2]};
            // Local axes from edges
            Real ax[3], ay[3];
            Real mid01[3], mid03[3];
            for (int d = 0; d < 3; ++d) {
                mid01[d] = 0.5*(x[0][d]+x[1][d]);
                mid03[d] = 0.5*(x[0][d]+x[3][d]);
                ax[d] = 0.5*(x[1][d]+x[2][d]) - mid03[d]; // ~x direction
                ay[d] = 0.5*(x[3][d]+x[2][d]) - mid01[d]; // ~y direction
            }
            Real ax_len = std::sqrt(ax[0]*ax[0]+ax[1]*ax[1]+ax[2]*ax[2]);
            Real ay_len = std::sqrt(ay[0]*ay[0]+ay[1]*ay[1]+ay[2]*ay[2]);
            if (ax_len < 1e-30) ax_len = 1e-30;
            if (ay_len < 1e-30) ay_len = 1e-30;
            Real xi  = (lp[0]*ax[0]+lp[1]*ax[1]+lp[2]*ax[2]) / (ax_len*ax_len) * 2.0;
            Real eta = (lp[0]*ay[0]+lp[1]*ay[1]+lp[2]*ay[2]) / (ay_len*ay_len) * 2.0;
            xi  = std::max(-1.0, std::min(1.0, xi));
            eta = std::max(-1.0, std::min(1.0, eta));
            phi[0] = 0.25 * (1-xi) * (1-eta);
            phi[1] = 0.25 * (1+xi) * (1-eta);
            phi[2] = 0.25 * (1+xi) * (1+eta);
            phi[3] = 0.25 * (1-xi) * (1+eta);
        }

        return gap;
    }
};

} // namespace fem
} // namespace nxs
