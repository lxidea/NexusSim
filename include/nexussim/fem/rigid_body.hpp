#pragma once

/**
 * @file rigid_body.hpp
 * @brief Rigid body system for crash simulation
 *
 * Provides:
 * - Rigid body properties (mass, inertia, COM)
 * - Quaternion-based rotation integration
 * - Kinematic coupling of slave nodes to rigid body motion
 * - Force/torque gathering from slave nodes
 *
 * Reference: OpenRadioss /engine/source/constraints/general/rbody
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>

namespace nxs {
namespace fem {

// ============================================================================
// Quaternion (for rotation integration)
// ============================================================================

struct Quaternion {
    Real w, x, y, z;

    Quaternion() : w(1.0), x(0.0), y(0.0), z(0.0) {}
    Quaternion(Real w_, Real x_, Real y_, Real z_) : w(w_), x(x_), y(y_), z(z_) {}

    Real norm() const { return std::sqrt(w*w + x*x + y*y + z*z); }

    void normalize() {
        Real n = norm();
        if (n > 1.0e-30) { w /= n; x /= n; y /= n; z /= n; }
    }

    Quaternion operator*(const Quaternion& q) const {
        return Quaternion(
            w*q.w - x*q.x - y*q.y - z*q.z,
            w*q.x + x*q.w + y*q.z - z*q.y,
            w*q.y - x*q.z + y*q.w + z*q.x,
            w*q.z + x*q.y - y*q.x + z*q.w
        );
    }

    /// Convert quaternion to 3x3 rotation matrix (row-major)
    void to_rotation_matrix(Real* R) const {
        Real xx = x*x, yy = y*y, zz = z*z;
        Real xy = x*y, xz = x*z, yz = y*z;
        Real wx = w*x, wy = w*y, wz = w*z;

        R[0] = 1.0 - 2.0*(yy + zz); R[1] = 2.0*(xy - wz);       R[2] = 2.0*(xz + wy);
        R[3] = 2.0*(xy + wz);        R[4] = 1.0 - 2.0*(xx + zz); R[5] = 2.0*(yz - wx);
        R[6] = 2.0*(xz - wy);        R[7] = 2.0*(yz + wx);        R[8] = 1.0 - 2.0*(xx + yy);
    }

    /// Rotate a vector by this quaternion
    void rotate_vector(const Real* v_in, Real* v_out) const {
        Real R[9];
        to_rotation_matrix(R);
        v_out[0] = R[0]*v_in[0] + R[1]*v_in[1] + R[2]*v_in[2];
        v_out[1] = R[3]*v_in[0] + R[4]*v_in[1] + R[5]*v_in[2];
        v_out[2] = R[6]*v_in[0] + R[7]*v_in[1] + R[8]*v_in[2];
    }
};

// ============================================================================
// Rigid Body Properties
// ============================================================================

struct RigidBodyProperties {
    Real mass;                ///< Total mass
    Real inertia[6];          ///< Inertia tensor [Ixx, Iyy, Izz, Ixy, Iyz, Ixz]
    Real com[3];              ///< Center of mass position
    Real com_initial[3];      ///< Initial center of mass

    RigidBodyProperties() : mass(0.0) {
        for (int i = 0; i < 6; ++i) inertia[i] = 0.0;
        for (int i = 0; i < 3; ++i) { com[i] = 0.0; com_initial[i] = 0.0; }
    }
};

// ============================================================================
// Rigid Body
// ============================================================================

class RigidBody {
public:
    RigidBody(int id = 0, const std::string& name = "")
        : id_(id), name_(name), active_(true) {
        for (int i = 0; i < 3; ++i) {
            velocity_[i] = 0.0;
            angular_velocity_[i] = 0.0;
            force_[i] = 0.0;
            torque_[i] = 0.0;
            acceleration_[i] = 0.0;
            angular_acceleration_[i] = 0.0;
        }
    }

    int id() const { return id_; }
    const std::string& name() const { return name_; }
    bool active() const { return active_; }
    void set_active(bool a) { active_ = a; }

    // --- Configuration ---

    void add_slave_node(Index node_id) {
        slave_nodes_.push_back(node_id);
    }

    void set_slave_nodes(const std::vector<Index>& nodes) {
        slave_nodes_ = nodes;
    }

    const std::vector<Index>& slave_nodes() const { return slave_nodes_; }
    std::size_t num_slave_nodes() const { return slave_nodes_.size(); }

    // --- Properties access ---

    RigidBodyProperties& properties() { return props_; }
    const RigidBodyProperties& properties() const { return props_; }

    Real* velocity() { return velocity_; }
    const Real* velocity() const { return velocity_; }

    Real* angular_velocity() { return angular_velocity_; }
    const Real* angular_velocity() const { return angular_velocity_; }

    Real* force() { return force_; }
    const Real* force() const { return force_; }
    Real* torque() { return torque_; }
    const Real* torque() const { return torque_; }

    const Quaternion& orientation() const { return orientation_; }

    // --- Initialization ---

    /**
     * @brief Compute rigid body properties from slave node positions and masses
     * @param positions Node positions (3 components per node)
     * @param masses Node masses
     */
    void compute_properties(const Real* positions, const Real* masses) {
        props_.mass = 0.0;
        for (int i = 0; i < 3; ++i) props_.com[i] = 0.0;
        for (int i = 0; i < 6; ++i) props_.inertia[i] = 0.0;

        // Compute total mass and center of mass
        for (std::size_t i = 0; i < slave_nodes_.size(); ++i) {
            Index n = slave_nodes_[i];
            Real m = masses[n];
            props_.mass += m;
            props_.com[0] += m * positions[3*n + 0];
            props_.com[1] += m * positions[3*n + 1];
            props_.com[2] += m * positions[3*n + 2];
        }

        if (props_.mass > 1.0e-30) {
            props_.com[0] /= props_.mass;
            props_.com[1] /= props_.mass;
            props_.com[2] /= props_.mass;
        }

        for (int i = 0; i < 3; ++i) props_.com_initial[i] = props_.com[i];

        // Compute inertia tensor about COM
        for (std::size_t i = 0; i < slave_nodes_.size(); ++i) {
            Index n = slave_nodes_[i];
            Real m = masses[n];
            Real rx = positions[3*n + 0] - props_.com[0];
            Real ry = positions[3*n + 1] - props_.com[1];
            Real rz = positions[3*n + 2] - props_.com[2];

            props_.inertia[0] += m * (ry*ry + rz*rz);  // Ixx
            props_.inertia[1] += m * (rx*rx + rz*rz);  // Iyy
            props_.inertia[2] += m * (rx*rx + ry*ry);  // Izz
            props_.inertia[3] -= m * rx * ry;           // Ixy
            props_.inertia[4] -= m * ry * rz;           // Iyz
            props_.inertia[5] -= m * rx * rz;           // Ixz
        }
    }

    // --- Time Integration ---

    /**
     * @brief Update rigid body state (integrate motion)
     * @param dt Time step
     */
    void update(Real dt) {
        if (!active_ || props_.mass < 1.0e-30) return;

        // Translational: a = F/m, v += a*dt, x += v*dt
        for (int i = 0; i < 3; ++i) {
            acceleration_[i] = force_[i] / props_.mass;
            velocity_[i] += acceleration_[i] * dt;
            props_.com[i] += velocity_[i] * dt;
        }

        // Rotational: α = I^-1 * τ (simplified: diagonal inertia)
        for (int i = 0; i < 3; ++i) {
            if (props_.inertia[i] > 1.0e-30) {
                angular_acceleration_[i] = torque_[i] / props_.inertia[i];
            }
            angular_velocity_[i] += angular_acceleration_[i] * dt;
        }

        // Update quaternion from angular velocity
        Real omega_mag = std::sqrt(angular_velocity_[0]*angular_velocity_[0] +
                                    angular_velocity_[1]*angular_velocity_[1] +
                                    angular_velocity_[2]*angular_velocity_[2]);
        if (omega_mag > 1.0e-30) {
            Real half_angle = 0.5 * omega_mag * dt;
            Real s = std::sin(half_angle) / omega_mag;
            Quaternion dq(std::cos(half_angle),
                          s * angular_velocity_[0],
                          s * angular_velocity_[1],
                          s * angular_velocity_[2]);
            orientation_ = dq * orientation_;
            orientation_.normalize();
        }

        // Reset forces for next step
        for (int i = 0; i < 3; ++i) { force_[i] = 0.0; torque_[i] = 0.0; }
    }

    /**
     * @brief Scatter rigid body velocity to slave nodes
     * v_node = v_com + ω × (r_node - r_com)
     */
    void scatter_to_nodes(const Real* positions, Real* velocities) const {
        if (!active_) return;

        for (std::size_t i = 0; i < slave_nodes_.size(); ++i) {
            Index n = slave_nodes_[i];
            Real rx = positions[3*n + 0] - props_.com[0];
            Real ry = positions[3*n + 1] - props_.com[1];
            Real rz = positions[3*n + 2] - props_.com[2];

            // v = v_com + ω × r
            velocities[3*n + 0] = velocity_[0] + (angular_velocity_[1]*rz - angular_velocity_[2]*ry);
            velocities[3*n + 1] = velocity_[1] + (angular_velocity_[2]*rx - angular_velocity_[0]*rz);
            velocities[3*n + 2] = velocity_[2] + (angular_velocity_[0]*ry - angular_velocity_[1]*rx);
        }
    }

    /**
     * @brief Gather forces from slave nodes to rigid body
     * F = Σ f_node,  τ = Σ (r_node - r_com) × f_node
     */
    void gather_forces(const Real* positions, const Real* forces) {
        if (!active_) return;

        for (std::size_t i = 0; i < slave_nodes_.size(); ++i) {
            Index n = slave_nodes_[i];
            Real fx = forces[3*n + 0];
            Real fy = forces[3*n + 1];
            Real fz = forces[3*n + 2];

            force_[0] += fx;
            force_[1] += fy;
            force_[2] += fz;

            Real rx = positions[3*n + 0] - props_.com[0];
            Real ry = positions[3*n + 1] - props_.com[1];
            Real rz = positions[3*n + 2] - props_.com[2];

            torque_[0] += ry*fz - rz*fy;
            torque_[1] += rz*fx - rx*fz;
            torque_[2] += rx*fy - ry*fx;
        }
    }

    /**
     * @brief Apply external force at a point
     * @param point World position where force is applied
     * @param f Force vector
     */
    void apply_force(const Real* point, const Real* f) {
        force_[0] += f[0];
        force_[1] += f[1];
        force_[2] += f[2];

        Real rx = point[0] - props_.com[0];
        Real ry = point[1] - props_.com[1];
        Real rz = point[2] - props_.com[2];

        torque_[0] += ry*f[2] - rz*f[1];
        torque_[1] += rz*f[0] - rx*f[2];
        torque_[2] += rx*f[1] - ry*f[0];
    }

    void print_state() const {
        std::cout << "RigidBody " << id_ << " (" << name_ << ")\n"
                  << "  Mass: " << props_.mass << " kg\n"
                  << "  COM: [" << props_.com[0] << ", " << props_.com[1] << ", " << props_.com[2] << "]\n"
                  << "  Vel: [" << velocity_[0] << ", " << velocity_[1] << ", " << velocity_[2] << "]\n"
                  << "  AngVel: [" << angular_velocity_[0] << ", " << angular_velocity_[1]
                  << ", " << angular_velocity_[2] << "]\n"
                  << "  Nodes: " << slave_nodes_.size() << "\n";
    }

private:
    int id_;
    std::string name_;
    bool active_;

    RigidBodyProperties props_;
    Quaternion orientation_;

    Real velocity_[3];
    Real angular_velocity_[3];
    Real acceleration_[3];
    Real angular_acceleration_[3];
    Real force_[3];
    Real torque_[3];

    std::vector<Index> slave_nodes_;
};

// ============================================================================
// Rigid Body Manager
// ============================================================================

class RigidBodyManager {
public:
    RigidBodyManager() = default;

    RigidBody& add_rigid_body(int id, const std::string& name = "") {
        bodies_.emplace_back(id, name);
        return bodies_.back();
    }

    std::size_t num_bodies() const { return bodies_.size(); }

    RigidBody& body(std::size_t i) { return bodies_[i]; }
    const RigidBody& body(std::size_t i) const { return bodies_[i]; }

    RigidBody* find_by_id(int id) {
        for (auto& b : bodies_) {
            if (b.id() == id) return &b;
        }
        return nullptr;
    }

    /**
     * @brief Initialize all rigid bodies from mesh data
     */
    void initialize(const Real* positions, const Real* masses) {
        for (auto& b : bodies_) {
            b.compute_properties(positions, masses);
        }
    }

    /**
     * @brief Update all rigid bodies (time integration)
     */
    void update_all(Real dt) {
        for (auto& b : bodies_) {
            b.update(dt);
        }
    }

    /**
     * @brief Scatter all rigid body velocities to nodes
     */
    void scatter_all(const Real* positions, Real* velocities) const {
        for (const auto& b : bodies_) {
            b.scatter_to_nodes(positions, velocities);
        }
    }

    /**
     * @brief Gather forces for all rigid bodies
     */
    void gather_all(const Real* positions, const Real* forces) {
        for (auto& b : bodies_) {
            b.gather_forces(positions, forces);
        }
    }

private:
    std::vector<RigidBody> bodies_;
};

} // namespace fem
} // namespace nxs
