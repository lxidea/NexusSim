/**
 * @file beam2.cpp
 * @brief Implementation of 2-node 3D Euler-Bernoulli beam element
 */

#include <nexussim/discretization/beam2.hpp>
#include <cmath>
#include <algorithm>

namespace nxs {
namespace fem {

// ============================================================================
// Cross-Section Properties
// ============================================================================

void Beam2Element::set_circular_section(Real radius) {
    const Real r2 = radius * radius;
    const Real r4 = r2 * r2;

    A_ = constants::pi<Real> * r2;                // πr²
    Iy_ = constants::pi<Real> * r4 / 4.0;         // πr⁴/4
    Iz_ = constants::pi<Real> * r4 / 4.0;         // πr⁴/4
    J_ = constants::pi<Real> * r4 / 2.0;          // πr⁴/2 (polar moment)
}

void Beam2Element::set_rectangular_section(Real width, Real height) {
    A_ = width * height;
    Iy_ = width * height * height * height / 12.0;  // bh³/12
    Iz_ = height * width * width * width / 12.0;    // hb³/12

    // Torsion constant for rectangle (approximate)
    const Real a = std::max(width, height);
    const Real b = std::min(width, height);
    J_ = a * b * b * b * (1.0/3.0 - 0.21 * b / a * (1.0 - b*b*b*b / (12.0*a*a*a*a)));
}

void Beam2Element::set_section_properties(Real A, Real Iy, Real Iz, Real J) {
    A_ = A;
    Iy_ = Iy;
    Iz_ = Iz;
    J_ = J;
}

// ============================================================================
// Shape Functions
// ============================================================================

KOKKOS_INLINE_FUNCTION
void Beam2Element::shape_functions(const Real xi[3], Real* N) const {
    // Linear shape functions for axial deformation
    const Real xi_val = xi[0];

    N[0] = 0.5 * (1.0 - xi_val);  // Node 0
    N[1] = 0.5 * (1.0 + xi_val);  // Node 1
}

KOKKOS_INLINE_FUNCTION
void Beam2Element::shape_derivatives(const Real xi[3], Real* dN) const {
    // Derivatives of linear shape functions
    dN[0*3 + 0] = -0.5;  // dN0/dξ
    dN[1*3 + 0] =  0.5;  // dN1/dξ

    dN[0*3 + 1] = 0.0;   // Not used
    dN[0*3 + 2] = 0.0;
    dN[1*3 + 1] = 0.0;
    dN[1*3 + 2] = 0.0;
}

// ============================================================================
// Hermite Shape Functions (for bending)
// ============================================================================

void Beam2Element::hermite_shape_functions(Real xi, Real L, Real* N, Real* dN) const {
    // Cubic Hermite polynomials for bending
    // 4 functions: w0, θ0, w1, θ1

    const Real xi2 = xi * xi;
    const Real xi3 = xi2 * xi;

    // Shape functions
    N[0] = 0.25 * (2.0 - 3.0*xi + xi3);           // w0
    N[1] = L/8.0 * (1.0 - xi - xi2 + xi3);        // θ0
    N[2] = 0.25 * (2.0 + 3.0*xi - xi3);           // w1
    N[3] = L/8.0 * (-1.0 - xi + xi2 + xi3);       // θ1

    // Derivatives w.r.t. xi
    dN[0] = 0.25 * (-3.0 + 3.0*xi2);
    dN[1] = L/8.0 * (-1.0 - 2.0*xi + 3.0*xi2);
    dN[2] = 0.25 * (3.0 - 3.0*xi2);
    dN[3] = L/8.0 * (-1.0 + 2.0*xi + 3.0*xi2);
}

// ============================================================================
// Gauss Quadrature
// ============================================================================

void Beam2Element::gauss_quadrature(Real* points, Real* weights) const {
    // 2-point Gauss quadrature
    const Real a = 1.0 / std::sqrt(3.0);

    points[0] = -a;  // First point
    points[1] = 0.0;
    points[2] = 0.0;
    weights[0] = 1.0;

    points[3] = a;   // Second point
    points[4] = 0.0;
    points[5] = 0.0;
    weights[1] = 1.0;
}

// ============================================================================
// Jacobian Computation
// ============================================================================

KOKKOS_INLINE_FUNCTION
Real Beam2Element::jacobian(const Real xi[3], const Real* coords, Real* J) const {
    // 1D Jacobian: J = L/2 (half length)
    const Real dx = coords[3] - coords[0];
    const Real dy = coords[4] - coords[1];
    const Real dz = coords[5] - coords[2];

    const Real L = std::sqrt(dx*dx + dy*dy + dz*dz);
    const Real det_J = L / 2.0;

    // Store transformation in J (simplified)
    for (int i = 0; i < 9; ++i) {
        J[i] = 0.0;
    }
    J[0] = det_J;

    return det_J;
}

// ============================================================================
// B-Matrix
// ============================================================================

KOKKOS_INLINE_FUNCTION
void Beam2Element::strain_displacement_matrix(const Real xi[3], const Real* coords, Real* B) const {
    // Simplified B-matrix for beam
    // Full implementation would include axial + bending + torsion

    for (int i = 0; i < 6 * NUM_DOF; ++i) {
        B[i] = 0.0;
    }

    // For now, just axial strain component
    const Real L = length(coords);

    // Axial strain: ε = du/dx
    B[0 * NUM_DOF + 0] = -1.0 / L;  // Node 0, ux
    B[0 * NUM_DOF + 6] =  1.0 / L;  // Node 1, ux
}

// ============================================================================
// Local Coordinate System
// ============================================================================

Real Beam2Element::length(const Real* coords) const {
    const Real dx = coords[3] - coords[0];
    const Real dy = coords[4] - coords[1];
    const Real dz = coords[5] - coords[2];

    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

void Beam2Element::local_coordinate_system(const Real* coords, Real* e1, Real* e2, Real* e3) const {
    // e1: along beam axis (node 0 to node 1)
    e1[0] = coords[3] - coords[0];
    e1[1] = coords[4] - coords[1];
    e1[2] = coords[5] - coords[2];

    const Real L = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
    if (L > 1.0e-12) {
        e1[0] /= L;
        e1[1] /= L;
        e1[2] /= L;
    }

    // e2: perpendicular to e1 (use global z if beam is horizontal)
    if (std::abs(e1[2]) < 0.9) {
        // Beam is not vertical, use global Z
        e2[0] = -e1[2] * e1[0];
        e2[1] = -e1[2] * e1[1];
        e2[2] = 1.0 - e1[2] * e1[2];
    } else {
        // Beam is vertical, use global X
        e2[0] = 1.0 - e1[0] * e1[0];
        e2[1] = -e1[0] * e1[1];
        e2[2] = -e1[0] * e1[2];
    }

    const Real len_e2 = std::sqrt(e2[0]*e2[0] + e2[1]*e2[1] + e2[2]*e2[2]);
    if (len_e2 > 1.0e-12) {
        e2[0] /= len_e2;
        e2[1] /= len_e2;
        e2[2] /= len_e2;
    }

    // e3 = e1 × e2
    e3[0] = e1[1] * e2[2] - e1[2] * e2[1];
    e3[1] = e1[2] * e2[0] - e1[0] * e2[2];
    e3[2] = e1[0] * e2[1] - e1[1] * e2[0];
}

// ============================================================================
// Mass Matrix
// ============================================================================

void Beam2Element::lumped_mass_matrix(const Real* coords, Real density, Real* M) const {
    const Real L = length(coords);
    const Real total_mass = density * A_ * L;
    const Real nodal_mass = total_mass / 2.0;

    // Initialize to zero
    for (int i = 0; i < NUM_DOF; ++i) {
        M[i] = 0.0;
    }

    // Translational mass at each node
    for (int node = 0; node < NUM_NODES; ++node) {
        for (int d = 0; d < 3; ++d) {
            M[node * DOF_PER_NODE + d] = nodal_mass;
        }
        // Rotational inertia (simplified)
        const Real I_rot = density * (Iy_ + Iz_) * L / 2.0;
        for (int d = 3; d < 6; ++d) {
            M[node * DOF_PER_NODE + d] = I_rot / 3.0;
        }
    }
}

void Beam2Element::mass_matrix(const Real* coords, Real density, Real* M) const {
    // Simplified: use lumped mass on diagonal
    for (int i = 0; i < NUM_DOF * NUM_DOF; ++i) {
        M[i] = 0.0;
    }

    Real M_lumped[NUM_DOF];
    lumped_mass_matrix(coords, density, M_lumped);

    for (int i = 0; i < NUM_DOF; ++i) {
        M[i * NUM_DOF + i] = M_lumped[i];
    }
}

// ============================================================================
// Stiffness Matrix
// ============================================================================

void Beam2Element::stiffness_matrix(const Real* coords, Real E, Real nu, Real* K) const {
    // 3D Euler-Bernoulli beam stiffness matrix
    // Includes: axial + torsion + bending (2 planes)

    const Real L = length(coords);
    const Real G = E / (2.0 * (1.0 + nu));  // Shear modulus

    // Initialize to zero
    for (int i = 0; i < NUM_DOF * NUM_DOF; ++i) {
        K[i] = 0.0;
    }

    // Axial stiffness: EA/L
    const Real k_axial = E * A_ / L;
    K[0 * NUM_DOF + 0] = k_axial;
    K[0 * NUM_DOF + 6] = -k_axial;
    K[6 * NUM_DOF + 0] = -k_axial;
    K[6 * NUM_DOF + 6] = k_axial;

    // Torsion stiffness: GJ/L
    const Real k_torsion = G * J_ / L;
    K[3 * NUM_DOF + 3] = k_torsion;
    K[3 * NUM_DOF + 9] = -k_torsion;
    K[9 * NUM_DOF + 3] = -k_torsion;
    K[9 * NUM_DOF + 9] = k_torsion;

    // Bending stiffness in y-z plane (about z-axis)
    const Real EIz = E * Iz_;
    const Real L2 = L * L;
    const Real L3 = L2 * L;

    // uy DOFs (indices 1, 7)
    K[1 * NUM_DOF + 1] = 12.0 * EIz / L3;
    K[1 * NUM_DOF + 5] = 6.0 * EIz / L2;
    K[1 * NUM_DOF + 7] = -12.0 * EIz / L3;
    K[1 * NUM_DOF + 11] = 6.0 * EIz / L2;

    K[5 * NUM_DOF + 1] = 6.0 * EIz / L2;
    K[5 * NUM_DOF + 5] = 4.0 * EIz / L;
    K[5 * NUM_DOF + 7] = -6.0 * EIz / L2;
    K[5 * NUM_DOF + 11] = 2.0 * EIz / L;

    K[7 * NUM_DOF + 1] = -12.0 * EIz / L3;
    K[7 * NUM_DOF + 5] = -6.0 * EIz / L2;
    K[7 * NUM_DOF + 7] = 12.0 * EIz / L3;
    K[7 * NUM_DOF + 11] = -6.0 * EIz / L2;

    K[11 * NUM_DOF + 1] = 6.0 * EIz / L2;
    K[11 * NUM_DOF + 5] = 2.0 * EIz / L;
    K[11 * NUM_DOF + 7] = -6.0 * EIz / L2;
    K[11 * NUM_DOF + 11] = 4.0 * EIz / L;

    // Bending stiffness in x-z plane (about y-axis)
    const Real EIy = E * Iy_;

    // uz DOFs (indices 2, 8)
    K[2 * NUM_DOF + 2] = 12.0 * EIy / L3;
    K[2 * NUM_DOF + 4] = -6.0 * EIy / L2;
    K[2 * NUM_DOF + 8] = -12.0 * EIy / L3;
    K[2 * NUM_DOF + 10] = -6.0 * EIy / L2;

    K[4 * NUM_DOF + 2] = -6.0 * EIy / L2;
    K[4 * NUM_DOF + 4] = 4.0 * EIy / L;
    K[4 * NUM_DOF + 8] = 6.0 * EIy / L2;
    K[4 * NUM_DOF + 10] = 2.0 * EIy / L;

    K[8 * NUM_DOF + 2] = -12.0 * EIy / L3;
    K[8 * NUM_DOF + 4] = 6.0 * EIy / L2;
    K[8 * NUM_DOF + 8] = 12.0 * EIy / L3;
    K[8 * NUM_DOF + 10] = 6.0 * EIy / L2;

    K[10 * NUM_DOF + 2] = -6.0 * EIy / L2;
    K[10 * NUM_DOF + 4] = 2.0 * EIy / L;
    K[10 * NUM_DOF + 8] = 6.0 * EIy / L2;
    K[10 * NUM_DOF + 10] = 4.0 * EIy / L;
}

// ============================================================================
// Internal Force
// ============================================================================

KOKKOS_INLINE_FUNCTION
void Beam2Element::internal_force(const Real* coords, const Real* disp,
                                   const Real* stress, Real* fint) const {
    // Simplified: f_int = K * u
    // For beam, this would involve computing axial force, shear, moments

    for (int i = 0; i < NUM_DOF; ++i) {
        fint[i] = 0.0;
    }

    // Axial force component (simplified)
    const Real L = length(coords);
    const Real axial_strain = (disp[6] - disp[0]) / L;
    const Real axial_force = stress[0] * A_;

    fint[0] = -axial_force;
    fint[6] = axial_force;
}

// ============================================================================
// Geometric Queries
// ============================================================================

bool Beam2Element::contains_point(const Real* coords, const Real* point, Real* xi) const {
    // Check if point is close to beam centerline
    const Real* p0 = &coords[0];
    const Real* p1 = &coords[3];

    // Vector along beam
    Real v[3];
    v[0] = p1[0] - p0[0];
    v[1] = p1[1] - p0[1];
    v[2] = p1[2] - p0[2];

    const Real L = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);

    // Vector from p0 to point
    Real w[3];
    w[0] = point[0] - p0[0];
    w[1] = point[1] - p0[1];
    w[2] = point[2] - p0[2];

    // Project onto beam axis
    const Real t = (w[0]*v[0] + w[1]*v[1] + w[2]*v[2]) / (L * L);

    if (t < 0.0 || t > 1.0) {
        return false;  // Outside beam length
    }

    xi[0] = 2.0 * t - 1.0;  // Convert to [-1, 1]
    xi[1] = 0.0;
    xi[2] = 0.0;

    return true;
}

Real Beam2Element::volume(const Real* coords) const {
    // "Volume" for beam is length × area
    return length(coords) * A_;
}

Real Beam2Element::characteristic_length(const Real* coords) const {
    return length(coords);
}

} // namespace fem
} // namespace nxs
