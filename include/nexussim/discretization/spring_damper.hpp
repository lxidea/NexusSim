#pragma once

/**
 * @file spring_damper.hpp
 * @brief Discrete spring and damper elements for FEM simulations
 *
 * Node numbering:
 *     0-----------1
 *
 * This file provides:
 *   1. SpringElement - Linear/nonlinear spring (displacement-based)
 *   2. DamperElement - Viscous damper (velocity-based)
 *   3. SpringDamperElement - Combined spring + parallel damper
 *
 * DOFs per node: 3 (ux, uy, uz) - translational only
 *
 * Applications:
 *   - Vehicle suspension modeling
 *   - Vibration isolation
 *   - Shock absorbers
 *   - Seat belt pretensioners
 *   - Connection stiffness modeling
 */

#include <nexussim/core/core.hpp>
#include <nexussim/physics/element.hpp>
#include <array>
#include <cmath>

namespace nxs {
namespace fem {

// ============================================================================
// Spring Characteristic Types
// ============================================================================

enum class SpringType {
    Linear,          ///< F = k × δ
    Nonlinear,       ///< F = f(δ) via lookup table
    Bilinear,        ///< Two different stiffnesses with transition
    Elastic_Plastic, ///< Elastic then perfectly plastic
    Polynomial       ///< F = k1×δ + k2×δ² + k3×δ³
};

// ============================================================================
// Spring Element
// ============================================================================

class SpringElement : public physics::Element {
public:
    static constexpr int NUM_NODES = 2;
    static constexpr int NUM_DIMS = 3;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOF = NUM_NODES * DOF_PER_NODE;  // 6

    SpringElement() = default;
    ~SpringElement() override = default;

    // ========================================================================
    // Element Properties
    // ========================================================================

    Properties properties() const override {
        return Properties{
            physics::ElementType::Spring,
            physics::ElementTopology::Line,
            NUM_NODES,
            1,  // No integration points needed
            DOF_PER_NODE,
            NUM_DIMS
        };
    }

    // ========================================================================
    // Spring Configuration
    // ========================================================================

    /**
     * @brief Set linear spring stiffness
     * @param k Spring stiffness (N/m)
     */
    void set_stiffness(Real k) {
        k_ = k;
        type_ = SpringType::Linear;
    }

    /**
     * @brief Set bilinear spring
     * @param k1 Initial stiffness (N/m)
     * @param k2 Secondary stiffness (N/m)
     * @param delta_transition Displacement at transition (m)
     */
    void set_bilinear(Real k1, Real k2, Real delta_transition) {
        k_ = k1;
        k2_ = k2;
        delta_transition_ = delta_transition;
        type_ = SpringType::Bilinear;
    }

    /**
     * @brief Set elastic-plastic spring
     * @param k Elastic stiffness (N/m)
     * @param F_yield Yield force (N)
     */
    void set_elastic_plastic(Real k, Real F_yield) {
        k_ = k;
        F_yield_ = F_yield;
        type_ = SpringType::Elastic_Plastic;
    }

    /**
     * @brief Set polynomial spring: F = k1×δ + k2×δ² + k3×δ³
     */
    void set_polynomial(Real k1, Real k2, Real k3) {
        k_ = k1;
        k2_ = k2;
        k3_ = k3;
        type_ = SpringType::Polynomial;
    }

    /**
     * @brief Set initial (free) length
     * @param L0 Initial length (m)
     */
    void set_initial_length(Real L0) { L0_ = L0; }

    /**
     * @brief Set preload force
     * @param F_preload Preload (positive = tension)
     */
    void set_preload(Real F_preload) { F_preload_ = F_preload; }

    Real stiffness() const { return k_; }
    Real initial_length() const { return L0_; }
    SpringType type() const { return type_; }

    // ========================================================================
    // Shape Functions (trivial for discrete element)
    // ========================================================================

    KOKKOS_INLINE_FUNCTION
    void shape_functions(const Real xi[3], Real* N) const override {
        (void)xi;
        N[0] = 0.5;
        N[1] = 0.5;
    }

    KOKKOS_INLINE_FUNCTION
    void shape_derivatives(const Real xi[3], Real* dN) const override {
        (void)xi;
        dN[0] = -0.5;
        dN[1] = 0.5;
    }

    void gauss_quadrature(Real* points, Real* weights) const override {
        points[0] = points[1] = points[2] = 0.0;
        weights[0] = 2.0;
    }

    KOKKOS_INLINE_FUNCTION
    Real jacobian(const Real xi[3], const Real* coords, Real* J) const override {
        (void)xi;
        J[0] = current_length(coords) / 2.0;
        return J[0];
    }

    KOKKOS_INLINE_FUNCTION
    void strain_displacement_matrix(const Real xi[3], const Real* coords, Real* B) const override {
        (void)xi;

        Real e[3];
        Real L = compute_direction(coords, e);
        Real L_inv = 1.0 / L;

        B[0] = -e[0] * L_inv;
        B[1] = -e[1] * L_inv;
        B[2] = -e[2] * L_inv;
        B[3] = +e[0] * L_inv;
        B[4] = +e[1] * L_inv;
        B[5] = +e[2] * L_inv;
    }

    // ========================================================================
    // Spring Force Computation
    // ========================================================================

    /**
     * @brief Compute spring force
     * @param coords Current nodal coordinates
     * @return Axial force (positive = tension)
     */
    KOKKOS_INLINE_FUNCTION
    Real spring_force(const Real* coords) const {
        Real L = current_length(coords);
        Real delta = L - L0_;  // Elongation

        Real F = F_preload_;

        switch (type_) {
            case SpringType::Linear:
                F += k_ * delta;
                break;

            case SpringType::Bilinear:
                if (Kokkos::fabs(delta) <= delta_transition_) {
                    F += k_ * delta;
                } else {
                    Real sign = (delta > 0) ? 1.0 : -1.0;
                    F += k_ * delta_transition_ * sign +
                         k2_ * (delta - delta_transition_ * sign);
                }
                break;

            case SpringType::Elastic_Plastic:
                F += k_ * delta;
                F = Kokkos::fmax(-F_yield_, Kokkos::fmin(F_yield_, F));
                break;

            case SpringType::Polynomial:
                F += k_ * delta + k2_ * delta * delta + k3_ * delta * delta * delta;
                break;

            default:
                F += k_ * delta;
        }

        return F;
    }

    /**
     * @brief Compute tangent stiffness at current state
     */
    KOKKOS_INLINE_FUNCTION
    Real tangent_stiffness(const Real* coords) const {
        Real L = current_length(coords);
        Real delta = L - L0_;

        switch (type_) {
            case SpringType::Linear:
                return k_;

            case SpringType::Bilinear:
                return (Kokkos::fabs(delta) <= delta_transition_) ? k_ : k2_;

            case SpringType::Elastic_Plastic: {
                Real F = Kokkos::fabs(k_ * delta + F_preload_);
                return (F < F_yield_) ? k_ : 0.0;
            }

            case SpringType::Polynomial:
                return k_ + 2.0 * k2_ * delta + 3.0 * k3_ * delta * delta;

            default:
                return k_;
        }
    }

    // ========================================================================
    // Element Matrices
    // ========================================================================

    void lumped_mass_matrix(const Real* coords, Real density, Real* M) const {
        // Springs typically don't have mass, but can if needed
        // Mass = density × volume (assumed cylindrical with default diameter)
        Real L = current_length(coords);
        Real diameter = 0.01;  // Default 1cm diameter
        Real volume = M_PI * diameter * diameter / 4.0 * L;
        Real total_mass = density * volume;

        for (int i = 0; i < 6; ++i) {
            M[i] = total_mass / 2.0;
        }
    }

    void mass_matrix(const Real* coords, Real density, Real* M) const override {
        Real M_lumped[6];
        lumped_mass_matrix(coords, density, M_lumped);

        for (int i = 0; i < 36; ++i) M[i] = 0.0;
        for (int i = 0; i < 6; ++i) {
            M[i*6 + i] = M_lumped[i];
        }
    }

    void stiffness_matrix(const Real* coords, Real E, Real nu, Real* K) const override {
        (void)E; (void)nu;  // Spring has its own stiffness

        Real e[3];
        compute_direction(coords, e);

        Real k = tangent_stiffness(coords);

        for (int i = 0; i < 36; ++i) K[i] = 0.0;

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Real keiej = k * e[i] * e[j];
                K[(0*3+i)*6 + (0*3+j)] = keiej;
                K[(0*3+i)*6 + (3+j)] = -keiej;
                K[(3+i)*6 + (0*3+j)] = -keiej;
                K[(3+i)*6 + (3+j)] = keiej;
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void internal_force(const Real* coords, const Real* disp,
                       const Real* stress, Real* fint) const override {
        (void)disp; (void)stress;

        Real e[3];
        compute_direction(coords, e);

        Real F = spring_force(coords);

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
        Real e[3];
        Real L = compute_direction(coords, e);

        Real v[3] = {point[0] - coords[0], point[1] - coords[1], point[2] - coords[2]};
        Real t = (v[0]*e[0] + v[1]*e[1] + v[2]*e[2]) / L;

        xi[0] = 2.0 * t - 1.0;
        xi[1] = xi[2] = 0.0;

        return (t >= -1.0e-6 && t <= 1.0 + 1.0e-6);
    }

    Real volume(const Real* coords) const override {
        return current_length(coords);
    }

    Real characteristic_length(const Real* coords) const override {
        return current_length(coords);
    }

    KOKKOS_INLINE_FUNCTION
    Real current_length(const Real* coords) const {
        Real dx = coords[3] - coords[0];
        Real dy = coords[4] - coords[1];
        Real dz = coords[5] - coords[2];
        return Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
    }

private:
    Real k_ = 1000.0;        // Spring stiffness (N/m)
    Real k2_ = 0.0;          // Secondary stiffness (for bilinear)
    Real k3_ = 0.0;          // Cubic coefficient (for polynomial)
    Real L0_ = 1.0;          // Initial length (m)
    Real F_preload_ = 0.0;   // Preload force (N)
    Real F_yield_ = 1.0e10;  // Yield force (for elastic-plastic)
    Real delta_transition_ = 0.1;  // Transition displacement (for bilinear)
    SpringType type_ = SpringType::Linear;

    KOKKOS_INLINE_FUNCTION
    Real compute_direction(const Real* coords, Real* e) const {
        e[0] = coords[3] - coords[0];
        e[1] = coords[4] - coords[1];
        e[2] = coords[5] - coords[2];

        Real L = Kokkos::sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
        if (L > 1.0e-20) {
            e[0] /= L; e[1] /= L; e[2] /= L;
        }
        return L;
    }
};

// ============================================================================
// Damper Element
// ============================================================================

class DamperElement : public physics::Element {
public:
    static constexpr int NUM_NODES = 2;
    static constexpr int NUM_DIMS = 3;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOF = 6;

    DamperElement() = default;
    ~DamperElement() override = default;

    Properties properties() const override {
        return Properties{
            physics::ElementType::Damper,
            physics::ElementTopology::Line,
            NUM_NODES, 1, DOF_PER_NODE, NUM_DIMS
        };
    }

    /**
     * @brief Set damping coefficient
     * @param c Damping coefficient (N·s/m)
     */
    void set_damping(Real c) { c_ = c; }

    /**
     * @brief Set nonlinear damping: F = c × |v|^n × sign(v)
     */
    void set_nonlinear_damping(Real c, Real n) {
        c_ = c;
        n_ = n;
        nonlinear_ = true;
    }

    Real damping() const { return c_; }

    // ========================================================================
    // Damper Force Computation
    // ========================================================================

    /**
     * @brief Compute damping force
     * @param coords Current nodal coordinates
     * @param velocity Nodal velocities (6 values)
     * @return Damping force (opposes relative velocity)
     */
    KOKKOS_INLINE_FUNCTION
    Real damper_force(const Real* coords, const Real* velocity) const {
        Real e[3];
        compute_direction(coords, e);

        // Relative velocity along element axis
        Real v_rel = e[0] * (velocity[3] - velocity[0]) +
                     e[1] * (velocity[4] - velocity[1]) +
                     e[2] * (velocity[5] - velocity[2]);

        if (nonlinear_) {
            Real sign = (v_rel >= 0) ? 1.0 : -1.0;
            return c_ * sign * Kokkos::pow(Kokkos::fabs(v_rel), n_);
        }

        return c_ * v_rel;
    }

    // ========================================================================
    // Element Interface
    // ========================================================================

    KOKKOS_INLINE_FUNCTION
    void shape_functions(const Real xi[3], Real* N) const override {
        (void)xi; N[0] = 0.5; N[1] = 0.5;
    }

    KOKKOS_INLINE_FUNCTION
    void shape_derivatives(const Real xi[3], Real* dN) const override {
        (void)xi; dN[0] = -0.5; dN[1] = 0.5;
    }

    void gauss_quadrature(Real* points, Real* weights) const override {
        points[0] = points[1] = points[2] = 0.0;
        weights[0] = 2.0;
    }

    KOKKOS_INLINE_FUNCTION
    Real jacobian(const Real xi[3], const Real* coords, Real* J) const override {
        (void)xi;
        J[0] = current_length(coords) / 2.0;
        return J[0];
    }

    KOKKOS_INLINE_FUNCTION
    void strain_displacement_matrix(const Real xi[3], const Real* coords, Real* B) const override {
        (void)xi;
        Real e[3];
        Real L = compute_direction(coords, e);
        Real L_inv = 1.0 / L;
        B[0] = -e[0] * L_inv; B[1] = -e[1] * L_inv; B[2] = -e[2] * L_inv;
        B[3] = +e[0] * L_inv; B[4] = +e[1] * L_inv; B[5] = +e[2] * L_inv;
    }

    void mass_matrix(const Real* coords, Real density, Real* M) const override {
        (void)density;
        for (int i = 0; i < 36; ++i) M[i] = 0.0;
    }

    void stiffness_matrix(const Real* coords, Real E, Real nu, Real* K) const override {
        (void)E; (void)nu; (void)coords;
        // Dampers have no stiffness
        for (int i = 0; i < 36; ++i) K[i] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void internal_force(const Real* coords, const Real* disp,
                       const Real* stress, Real* fint) const override {
        (void)disp; (void)stress;
        // Note: This needs velocity, not displacement
        // Damper force should be computed in time integration loop
        // Here we just return zero
        for (int i = 0; i < 6; ++i) fint[i] = 0.0;
    }

    /**
     * @brief Compute damping force contribution
     * @param coords Current nodal coordinates
     * @param velocity Nodal velocities
     * @param fdamp Output: damping force
     */
    KOKKOS_INLINE_FUNCTION
    void damping_force(const Real* coords, const Real* velocity, Real* fdamp) const {
        Real e[3];
        compute_direction(coords, e);

        Real F = damper_force(coords, velocity);

        // Force opposes relative velocity
        fdamp[0] = -F * e[0];
        fdamp[1] = -F * e[1];
        fdamp[2] = -F * e[2];
        fdamp[3] = +F * e[0];
        fdamp[4] = +F * e[1];
        fdamp[5] = +F * e[2];
    }

    bool contains_point(const Real* coords, const Real* point, Real* xi) const override {
        Real e[3];
        Real L = compute_direction(coords, e);
        Real v[3] = {point[0] - coords[0], point[1] - coords[1], point[2] - coords[2]};
        Real t = (v[0]*e[0] + v[1]*e[1] + v[2]*e[2]) / L;
        xi[0] = 2.0 * t - 1.0; xi[1] = xi[2] = 0.0;
        return (t >= -1.0e-6 && t <= 1.0 + 1.0e-6);
    }

    Real volume(const Real* coords) const override { return current_length(coords); }
    Real characteristic_length(const Real* coords) const override { return current_length(coords); }

    KOKKOS_INLINE_FUNCTION
    Real current_length(const Real* coords) const {
        Real dx = coords[3] - coords[0];
        Real dy = coords[4] - coords[1];
        Real dz = coords[5] - coords[2];
        return Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
    }

private:
    Real c_ = 100.0;   // Damping coefficient (N·s/m)
    Real n_ = 1.0;     // Exponent for nonlinear
    bool nonlinear_ = false;

    KOKKOS_INLINE_FUNCTION
    Real compute_direction(const Real* coords, Real* e) const {
        e[0] = coords[3] - coords[0];
        e[1] = coords[4] - coords[1];
        e[2] = coords[5] - coords[2];
        Real L = Kokkos::sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
        if (L > 1.0e-20) { e[0] /= L; e[1] /= L; e[2] /= L; }
        return L;
    }
};

// ============================================================================
// Spring-Damper Element (Combined)
// ============================================================================

class SpringDamperElement : public physics::Element {
public:
    static constexpr int NUM_NODES = 2;
    static constexpr int NUM_DIMS = 3;
    static constexpr int DOF_PER_NODE = 3;
    static constexpr int NUM_DOF = 6;

    SpringDamperElement() = default;
    ~SpringDamperElement() override = default;

    Properties properties() const override {
        return Properties{
            physics::ElementType::SpringDamper,
            physics::ElementTopology::Line,
            NUM_NODES, 1, DOF_PER_NODE, NUM_DIMS
        };
    }

    void set_spring_stiffness(Real k) { spring_.set_stiffness(k); }
    void set_damping(Real c) { damper_.set_damping(c); }
    void set_initial_length(Real L0) { spring_.set_initial_length(L0); }

    SpringElement& spring() { return spring_; }
    DamperElement& damper() { return damper_; }

    // ========================================================================
    // Combined Force
    // ========================================================================

    KOKKOS_INLINE_FUNCTION
    Real total_force(const Real* coords, const Real* velocity) const {
        return spring_.spring_force(coords) + damper_.damper_force(coords, velocity);
    }

    // ========================================================================
    // Element Interface (delegate to spring)
    // ========================================================================

    KOKKOS_INLINE_FUNCTION
    void shape_functions(const Real xi[3], Real* N) const override {
        spring_.shape_functions(xi, N);
    }

    KOKKOS_INLINE_FUNCTION
    void shape_derivatives(const Real xi[3], Real* dN) const override {
        spring_.shape_derivatives(xi, dN);
    }

    void gauss_quadrature(Real* points, Real* weights) const override {
        spring_.gauss_quadrature(points, weights);
    }

    KOKKOS_INLINE_FUNCTION
    Real jacobian(const Real xi[3], const Real* coords, Real* J) const override {
        return spring_.jacobian(xi, coords, J);
    }

    KOKKOS_INLINE_FUNCTION
    void strain_displacement_matrix(const Real xi[3], const Real* coords, Real* B) const override {
        spring_.strain_displacement_matrix(xi, coords, B);
    }

    void mass_matrix(const Real* coords, Real density, Real* M) const override {
        spring_.mass_matrix(coords, density, M);
    }

    void stiffness_matrix(const Real* coords, Real E, Real nu, Real* K) const override {
        spring_.stiffness_matrix(coords, E, nu, K);
    }

    KOKKOS_INLINE_FUNCTION
    void internal_force(const Real* coords, const Real* disp,
                       const Real* stress, Real* fint) const override {
        spring_.internal_force(coords, disp, stress, fint);
    }

    /**
     * @brief Compute combined spring + damper force
     */
    KOKKOS_INLINE_FUNCTION
    void combined_force(const Real* coords, const Real* velocity, Real* force) const {
        Real e[3];
        compute_direction(coords, e);

        Real F = total_force(coords, velocity);

        force[0] = -F * e[0];
        force[1] = -F * e[1];
        force[2] = -F * e[2];
        force[3] = +F * e[0];
        force[4] = +F * e[1];
        force[5] = +F * e[2];
    }

    bool contains_point(const Real* coords, const Real* point, Real* xi) const override {
        return spring_.contains_point(coords, point, xi);
    }

    Real volume(const Real* coords) const override { return spring_.volume(coords); }
    Real characteristic_length(const Real* coords) const override {
        return spring_.characteristic_length(coords);
    }

private:
    SpringElement spring_;
    DamperElement damper_;

    KOKKOS_INLINE_FUNCTION
    Real compute_direction(const Real* coords, Real* e) const {
        e[0] = coords[3] - coords[0];
        e[1] = coords[4] - coords[1];
        e[2] = coords[5] - coords[2];
        Real L = Kokkos::sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
        if (L > 1.0e-20) { e[0] /= L; e[1] /= L; e[2] /= L; }
        return L;
    }
};

} // namespace fem
} // namespace nxs
