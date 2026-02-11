#pragma once

/**
 * @file constraints.hpp
 * @brief Constraint manager for multi-point constraints, joints, and rigid links
 *
 * Supports:
 * - RBE2: Rigid kinematic coupling (master drives slaves)
 * - RBE3: Weighted average interpolation (slaves drive dependent)
 * - Spherical joint: Shared point, free rotation
 * - Revolute joint: Rotation about single axis
 * - Cylindrical joint: Rotation + translation along axis
 * - MPC: General multi-point constraint
 *
 * Reference: OpenRadioss /engine/source/constraints
 */

#include <nexussim/core/types.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

namespace nxs {
namespace fem {

// ============================================================================
// Constraint Types
// ============================================================================

enum class ConstraintType {
    RBE2,              ///< Rigid element (kinematic coupling)
    RBE3,              ///< Interpolation element (weighted average)
    Joint_Spherical,   ///< Ball joint (shared point)
    Joint_Revolute,    ///< Hinge (rotation about axis)
    Joint_Cylindrical, ///< Slider + hinge
    MPC                ///< Multi-point constraint
};

// ============================================================================
// Constraint Definition
// ============================================================================

struct Constraint {
    ConstraintType type;
    int id;
    std::string name;

    Index master_node;                ///< Master (independent) node
    std::vector<Index> slave_nodes;   ///< Slave (dependent) nodes
    std::vector<Real> weights;        ///< Weights for RBE3

    bool tied_dofs[6];                ///< Which DOFs are constrained [ux,uy,uz,rx,ry,rz]

    // Joint-specific
    Real axis[3];                     ///< Joint axis direction (revolute/cylindrical)
    Real point[3];                    ///< Joint point location

    // Penalty stiffness for soft constraints
    Real penalty_stiffness;

    Constraint()
        : type(ConstraintType::RBE2), id(0), master_node(0)
        , penalty_stiffness(1.0e10) {
        for (int i = 0; i < 6; ++i) tied_dofs[i] = true;  // All DOFs by default
        axis[0] = 0.0; axis[1] = 0.0; axis[2] = 1.0;
        point[0] = point[1] = point[2] = 0.0;
    }
};

// ============================================================================
// Constraint Manager
// ============================================================================

class ConstraintManager {
public:
    ConstraintManager() = default;

    // --- Configuration ---

    Constraint& add_constraint(ConstraintType type, int id = 0) {
        constraints_.emplace_back();
        auto& c = constraints_.back();
        c.type = type;
        c.id = id;
        return c;
    }

    std::size_t num_constraints() const { return constraints_.size(); }
    Constraint& constraint(std::size_t i) { return constraints_[i]; }
    const Constraint& constraint(std::size_t i) const { return constraints_[i]; }

    // --- Constraint Application ---

    /**
     * @brief Apply all constraints to velocities (post-integration)
     *
     * For kinematic constraints (RBE2): slave velocities are overwritten
     * For interpolation (RBE3): dependent velocity is weighted average
     * For joints: penalty forces are applied
     */
    void apply_constraints(Real* positions, Real* velocities,
                           Real* accelerations, Real dt) {
        for (const auto& c : constraints_) {
            switch (c.type) {
                case ConstraintType::RBE2:
                    apply_rbe2(c, positions, velocities, accelerations);
                    break;
                case ConstraintType::RBE3:
                    apply_rbe3(c, positions, velocities);
                    break;
                case ConstraintType::Joint_Spherical:
                    apply_spherical_joint(c, positions, velocities, dt);
                    break;
                case ConstraintType::Joint_Revolute:
                    apply_revolute_joint(c, positions, velocities, dt);
                    break;
                case ConstraintType::Joint_Cylindrical:
                    apply_cylindrical_joint(c, positions, velocities, dt);
                    break;
                case ConstraintType::MPC:
                    apply_mpc(c, velocities);
                    break;
            }
        }
    }

    /**
     * @brief Update constraint positions (e.g., for joints)
     */
    void update_constraints(const Real* positions) {
        for (auto& c : constraints_) {
            if (c.type == ConstraintType::Joint_Spherical ||
                c.type == ConstraintType::Joint_Revolute ||
                c.type == ConstraintType::Joint_Cylindrical) {
                // Update joint point to master node position
                Index m = c.master_node;
                c.point[0] = positions[3*m + 0];
                c.point[1] = positions[3*m + 1];
                c.point[2] = positions[3*m + 2];
            }
        }
    }

    void print_summary() const {
        std::cout << "Constraint Manager: " << constraints_.size() << " constraints\n";
        for (const auto& c : constraints_) {
            const char* type_str = "Unknown";
            switch (c.type) {
                case ConstraintType::RBE2: type_str = "RBE2"; break;
                case ConstraintType::RBE3: type_str = "RBE3"; break;
                case ConstraintType::Joint_Spherical: type_str = "Spherical Joint"; break;
                case ConstraintType::Joint_Revolute: type_str = "Revolute Joint"; break;
                case ConstraintType::Joint_Cylindrical: type_str = "Cylindrical Joint"; break;
                case ConstraintType::MPC: type_str = "MPC"; break;
            }
            std::cout << "  [" << c.id << "] " << type_str
                      << " master=" << c.master_node
                      << " slaves=" << c.slave_nodes.size() << "\n";
        }
    }

private:
    /**
     * @brief RBE2: v_slave = v_master + ω_master × r
     * For translational DOFs, slave velocity matches master exactly
     */
    void apply_rbe2(const Constraint& c, const Real* positions,
                    Real* velocities, Real* accelerations) const {
        Index m = c.master_node;
        Real vm[3] = {velocities[3*m+0], velocities[3*m+1], velocities[3*m+2]};
        Real am[3] = {0.0, 0.0, 0.0};
        if (accelerations) {
            am[0] = accelerations[3*m+0];
            am[1] = accelerations[3*m+1];
            am[2] = accelerations[3*m+2];
        }

        for (Index s : c.slave_nodes) {
            for (int d = 0; d < 3; ++d) {
                if (c.tied_dofs[d]) {
                    velocities[3*s + d] = vm[d];
                    if (accelerations) {
                        accelerations[3*s + d] = am[d];
                    }
                }
            }
        }
    }

    /**
     * @brief RBE3: v_dependent = Σ(w_i * v_independent_i) / Σ(w_i)
     * Master node velocity is weighted average of slave node velocities
     */
    void apply_rbe3(const Constraint& c, const Real* /*positions*/,
                    Real* velocities) const {
        if (c.slave_nodes.empty()) return;

        Real v_avg[3] = {0.0, 0.0, 0.0};
        Real w_sum = 0.0;

        for (std::size_t i = 0; i < c.slave_nodes.size(); ++i) {
            Index s = c.slave_nodes[i];
            Real w = (i < c.weights.size()) ? c.weights[i] : 1.0;
            v_avg[0] += w * velocities[3*s + 0];
            v_avg[1] += w * velocities[3*s + 1];
            v_avg[2] += w * velocities[3*s + 2];
            w_sum += w;
        }

        if (w_sum > 1.0e-30) {
            Index m = c.master_node;
            for (int d = 0; d < 3; ++d) {
                if (c.tied_dofs[d]) {
                    velocities[3*m + d] = v_avg[d] / w_sum;
                }
            }
        }
    }

    /**
     * @brief Spherical joint: shared point constraint via penalty
     */
    void apply_spherical_joint(const Constraint& c, const Real* positions,
                                Real* velocities, Real dt) const {
        Index m = c.master_node;
        Real k = c.penalty_stiffness;

        for (Index s : c.slave_nodes) {
            for (int d = 0; d < 3; ++d) {
                Real gap = positions[3*s + d] - positions[3*m + d];
                Real correction = -k * gap * dt;
                // Split correction between master and slave
                velocities[3*s + d] += 0.5 * correction;
                velocities[3*m + d] -= 0.5 * correction;
            }
        }
    }

    /**
     * @brief Revolute joint: constrain relative motion except rotation about axis
     */
    void apply_revolute_joint(const Constraint& c, const Real* positions,
                               Real* velocities, Real dt) const {
        Index m = c.master_node;
        Real ax = c.axis[0], ay = c.axis[1], az = c.axis[2];
        Real anorm = std::sqrt(ax*ax + ay*ay + az*az);
        if (anorm < 1.0e-30) return;
        ax /= anorm; ay /= anorm; az /= anorm;

        Real k = c.penalty_stiffness;

        for (Index s : c.slave_nodes) {
            // Relative position
            Real dx = positions[3*s+0] - positions[3*m+0];
            Real dy = positions[3*s+1] - positions[3*m+1];
            Real dz = positions[3*s+2] - positions[3*m+2];

            // Project out the axis component (allow motion along axis direction is constrained)
            // For revolute: constrain all translation, allow rotation about axis
            Real dot = dx*ax + dy*ay + dz*az;
            Real perp_x = dx - dot*ax;
            Real perp_y = dy - dot*ay;
            Real perp_z = dz - dot*az;

            // Penalize perpendicular displacement
            Real corr = -k * dt;
            velocities[3*s+0] += 0.5 * corr * perp_x;
            velocities[3*s+1] += 0.5 * corr * perp_y;
            velocities[3*s+2] += 0.5 * corr * perp_z;
            velocities[3*m+0] -= 0.5 * corr * perp_x;
            velocities[3*m+1] -= 0.5 * corr * perp_y;
            velocities[3*m+2] -= 0.5 * corr * perp_z;
        }
    }

    /**
     * @brief Cylindrical joint: allow rotation + translation along axis
     */
    void apply_cylindrical_joint(const Constraint& c, const Real* positions,
                                  Real* velocities, Real dt) const {
        Index m = c.master_node;
        Real ax = c.axis[0], ay = c.axis[1], az = c.axis[2];
        Real anorm = std::sqrt(ax*ax + ay*ay + az*az);
        if (anorm < 1.0e-30) return;
        ax /= anorm; ay /= anorm; az /= anorm;

        Real k = c.penalty_stiffness;

        for (Index s : c.slave_nodes) {
            Real dx = positions[3*s+0] - positions[3*m+0];
            Real dy = positions[3*s+1] - positions[3*m+1];
            Real dz = positions[3*s+2] - positions[3*m+2];

            // For cylindrical: only constrain perpendicular to axis
            Real dot = dx*ax + dy*ay + dz*az;
            Real perp_x = dx - dot*ax;
            Real perp_y = dy - dot*ay;
            Real perp_z = dz - dot*az;

            Real corr = -k * dt;
            velocities[3*s+0] += 0.5 * corr * perp_x;
            velocities[3*s+1] += 0.5 * corr * perp_y;
            velocities[3*s+2] += 0.5 * corr * perp_z;
            velocities[3*m+0] -= 0.5 * corr * perp_x;
            velocities[3*m+1] -= 0.5 * corr * perp_y;
            velocities[3*m+2] -= 0.5 * corr * perp_z;
        }
    }

    /**
     * @brief General MPC: constrain slave DOFs to match master
     */
    void apply_mpc(const Constraint& c, Real* velocities) const {
        Index m = c.master_node;

        for (Index s : c.slave_nodes) {
            for (int d = 0; d < 3; ++d) {
                if (c.tied_dofs[d]) {
                    velocities[3*s + d] = velocities[3*m + d];
                }
            }
        }
    }

    std::vector<Constraint> constraints_;
};

} // namespace fem
} // namespace nxs
