#pragma once

/**
 * @file specialty_wave38.hpp
 * @brief Wave 38: Specialty Element Formulations — 4 Elements
 *
 * Elements:
 *   6. HermiteBeam18        - 2-node Hermite cubic beam (12 DOF) with warping
 *   7. RivetElement         - Force-displacement rivet with mixed-mode failure
 *   8. WeldElement          - Spot/seam weld with HAZ softening
 *   9. GeneralSpringBeam    - LAW113 nonlinear spring-beam (6-DOF, tabulated)
 *
 * References:
 * - Przemieniecki (1968) "Theory of Matrix Structural Analysis"
 * - Belytschko et al. (2000) "Nonlinear Finite Elements"
 * - LS-DYNA Theory Manual: Element Formulations (Section 5)
 */

#include <nexussim/core/types.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <cstring>

namespace nxs {
namespace discretization {

using Real = nxs::Real;

// ============================================================================
// Utility functions
// ============================================================================

namespace specialty_detail {

inline Real safe_sqrt(Real x) { return std::sqrt(std::fmax(x, 0.0)); }

inline Real dot3(const Real a[3], const Real b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline void sub3(const Real a[3], const Real b[3], Real c[3]) {
    c[0] = a[0]-b[0]; c[1] = a[1]-b[1]; c[2] = a[2]-b[2];
}

inline Real norm3(const Real v[3]) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

inline void cross3(const Real a[3], const Real b[3], Real c[3]) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

/// Linear interpolation in tabulated curve
inline Real interp_table(const Real* x_data, const Real* y_data, int n, Real x) {
    if (n <= 0) return 0.0;
    if (n == 1) return y_data[0];
    if (x <= x_data[0]) return y_data[0];
    if (x >= x_data[n - 1]) return y_data[n - 1];
    for (int i = 0; i < n - 1; ++i) {
        if (x >= x_data[i] && x <= x_data[i + 1]) {
            Real dx = x_data[i + 1] - x_data[i];
            if (std::fabs(dx) < 1.0e-30) return y_data[i];
            Real t = (x - x_data[i]) / dx;
            return y_data[i] * (1.0 - t) + y_data[i + 1] * t;
        }
    }
    return y_data[n - 1];
}

/// Zero a matrix stored as flat array
inline void zero_matrix(Real* M, int rows, int cols) {
    std::memset(M, 0, sizeof(Real) * rows * cols);
}

} // namespace specialty_detail

// ============================================================================
// 6. HermiteBeam18 — 2-Node Hermite Cubic Beam (12 DOF) with Warping
// ============================================================================

/**
 * @brief Beam cross-section properties.
 */
struct BeamSection {
    Real A   = 1.0e-4;   ///< Cross-sectional area [m^2]
    Real Iy  = 8.33e-10; ///< Second moment of area about y-axis [m^4]
    Real Iz  = 8.33e-10; ///< Second moment of area about z-axis [m^4]
    Real J   = 1.41e-9;  ///< Torsion constant [m^4]
    Real Cw  = 0.0;      ///< Warping constant [m^6]
    Real E   = 2.1e11;   ///< Young's modulus [Pa]
    Real G   = 8.08e10;  ///< Shear modulus [Pa]
    Real rho = 7800.0;   ///< Density [kg/m^3]
};

/**
 * @brief 2-node Hermite cubic beam element with 12 DOF.
 *
 * DOF per node: (u, v, w, theta_x, theta_y, theta_z)
 * Total: 12 DOF for the 2-node element.
 *
 * The stiffness matrix combines:
 * - Axial stiffness: EA/L bar element
 * - Bending stiffness (y and z): Hermite cubic beam (EI/L^3 terms)
 * - Torsion: GJ/L
 * - Warping: ECw/L^3 (if warping constant Cw > 0)
 *
 * Shape functions for bending (Hermite cubics):
 *   N1 = 1 - 3*xi^2 + 2*xi^3
 *   N2 = L*xi*(1 - xi)^2
 *   N3 = 3*xi^2 - 2*xi^3
 *   N4 = L*xi^2*(xi - 1)
 *   where xi = x/L in [0, 1]
 *
 * Reference: Przemieniecki (1968), Chapter 5
 */
class HermiteBeam18 {
public:
    HermiteBeam18() = default;

    /**
     * @brief Compute the 12x12 local stiffness matrix.
     *
     * DOF ordering: [u1, v1, w1, rx1, ry1, rz1, u2, v2, w2, rx2, ry2, rz2]
     *
     * @param node1    Coordinates of node 1 [3]
     * @param node2    Coordinates of node 2 [3]
     * @param section  Beam section properties
     * @param K        Output stiffness matrix [12*12], row-major
     */
    void compute_stiffness(const Real node1[3], const Real node2[3],
                           const BeamSection& section, Real* K) const {
        specialty_detail::zero_matrix(K, 12, 12);

        // Element length
        Real dx[3];
        specialty_detail::sub3(node2, node1, dx);
        Real L = specialty_detail::norm3(dx);
        if (L < 1.0e-20) return;

        Real E = section.E;
        Real A = section.A;
        Real Iy = section.Iy;
        Real Iz = section.Iz;
        Real G = section.G;
        Real Jt = section.J;
        Real Cw = section.Cw;

        Real L2 = L * L;
        Real L3 = L2 * L;

        // Axial stiffness: EA/L
        Real ka = E * A / L;

        // Bending stiffness about z-axis (v displacement, rz rotation)
        Real kb_z = E * Iz / L3;

        // Bending stiffness about y-axis (w displacement, ry rotation)
        Real kb_y = E * Iy / L3;

        // Torsional stiffness: GJ/L + warping ECw/L^3
        Real kt = G * Jt / L;
        if (Cw > 0.0) {
            kt += E * Cw / L3;
        }

        // DOF indices:
        // 0=u1, 1=v1, 2=w1, 3=rx1, 4=ry1, 5=rz1
        // 6=u2, 7=v2, 8=w2, 9=rx2, 10=ry2, 11=rz2

        auto idx = [](int r, int c) { return r * 12 + c; };

        // Axial: u1, u2  (indices 0, 6)
        K[idx(0, 0)] =  ka;   K[idx(0, 6)] = -ka;
        K[idx(6, 0)] = -ka;   K[idx(6, 6)] =  ka;

        // Bending in x-y plane (v, rz): indices 1, 5, 7, 11
        // v1: 1, rz1: 5, v2: 7, rz2: 11
        K[idx(1, 1)]   =  12.0 * kb_z;
        K[idx(1, 5)]   =   6.0 * kb_z * L;
        K[idx(1, 7)]   = -12.0 * kb_z;
        K[idx(1, 11)]  =   6.0 * kb_z * L;

        K[idx(5, 1)]   =   6.0 * kb_z * L;
        K[idx(5, 5)]   =   4.0 * kb_z * L2;
        K[idx(5, 7)]   =  -6.0 * kb_z * L;
        K[idx(5, 11)]  =   2.0 * kb_z * L2;

        K[idx(7, 1)]   = -12.0 * kb_z;
        K[idx(7, 5)]   =  -6.0 * kb_z * L;
        K[idx(7, 7)]   =  12.0 * kb_z;
        K[idx(7, 11)]  =  -6.0 * kb_z * L;

        K[idx(11, 1)]  =   6.0 * kb_z * L;
        K[idx(11, 5)]  =   2.0 * kb_z * L2;
        K[idx(11, 7)]  =  -6.0 * kb_z * L;
        K[idx(11, 11)] =   4.0 * kb_z * L2;

        // Bending in x-z plane (w, ry): indices 2, 4, 8, 10
        K[idx(2, 2)]   =  12.0 * kb_y;
        K[idx(2, 4)]   =  -6.0 * kb_y * L;
        K[idx(2, 8)]   = -12.0 * kb_y;
        K[idx(2, 10)]  =  -6.0 * kb_y * L;

        K[idx(4, 2)]   =  -6.0 * kb_y * L;
        K[idx(4, 4)]   =   4.0 * kb_y * L2;
        K[idx(4, 8)]   =   6.0 * kb_y * L;
        K[idx(4, 10)]  =   2.0 * kb_y * L2;

        K[idx(8, 2)]   = -12.0 * kb_y;
        K[idx(8, 4)]   =   6.0 * kb_y * L;
        K[idx(8, 8)]   =  12.0 * kb_y;
        K[idx(8, 10)]  =   6.0 * kb_y * L;

        K[idx(10, 2)]  =  -6.0 * kb_y * L;
        K[idx(10, 4)]  =   2.0 * kb_y * L2;
        K[idx(10, 8)]  =   6.0 * kb_y * L;
        K[idx(10, 10)] =   4.0 * kb_y * L2;

        // Torsion: rx1, rx2 (indices 3, 9)
        K[idx(3, 3)] =  kt;   K[idx(3, 9)] = -kt;
        K[idx(9, 3)] = -kt;   K[idx(9, 9)] =  kt;
    }

    /**
     * @brief Compute internal forces from displacements.
     *
     * f = K * u
     *
     * @param node1          Node 1 coordinates [3]
     * @param node2          Node 2 coordinates [3]
     * @param section        Section properties
     * @param displacements  DOF vector [12]
     * @param forces         Output force vector [12]
     */
    void compute_internal_force(const Real node1[3], const Real node2[3],
                                const BeamSection& section,
                                const Real* displacements, Real* forces) const {
        Real K[144];
        compute_stiffness(node1, node2, section, K);

        // f = K * u
        for (int i = 0; i < 12; ++i) {
            forces[i] = 0.0;
            for (int j = 0; j < 12; ++j) {
                forces[i] += K[i * 12 + j] * displacements[j];
            }
        }
    }

    /**
     * @brief Compute consistent mass matrix (lumped approximation).
     *
     * Lumped mass: m_node = rho * A * L / 2 for translational DOFs.
     * Rotational inertia: J_node = rho * (Iy + Iz) * L / 2 for rotational DOFs.
     *
     * @param node1    Node 1 coordinates [3]
     * @param node2    Node 2 coordinates [3]
     * @param section  Section properties
     * @param M_diag   Output diagonal mass matrix [12]
     */
    void compute_mass_lumped(const Real node1[3], const Real node2[3],
                             const BeamSection& section, Real* M_diag) const {
        Real dx[3];
        specialty_detail::sub3(node2, node1, dx);
        Real L = specialty_detail::norm3(dx);

        Real m_half = section.rho * section.A * L * 0.5;
        Real J_half = section.rho * (section.Iy + section.Iz) * L * 0.5;

        // Node 1: u, v, w, rx, ry, rz
        M_diag[0] = m_half; M_diag[1] = m_half; M_diag[2] = m_half;
        M_diag[3] = J_half; M_diag[4] = J_half; M_diag[5] = J_half;
        // Node 2
        M_diag[6] = m_half; M_diag[7] = m_half; M_diag[8] = m_half;
        M_diag[9] = J_half; M_diag[10] = J_half; M_diag[11] = J_half;
    }

    /**
     * @brief Evaluate Hermite cubic shape functions.
     *
     * @param xi     Parametric coordinate in [0, 1] (= x/L)
     * @param L      Element length
     * @param N      Output: [N1, N2, N3, N4]
     */
    static void hermite_shape_functions(Real xi, Real L, Real N[4]) {
        Real xi2 = xi * xi;
        Real xi3 = xi2 * xi;
        N[0] = 1.0 - 3.0 * xi2 + 2.0 * xi3;           // displacement at node 1
        N[1] = L * xi * (1.0 - xi) * (1.0 - xi);       // rotation at node 1
        N[2] = 3.0 * xi2 - 2.0 * xi3;                  // displacement at node 2
        N[3] = L * xi2 * (xi - 1.0);                    // rotation at node 2
    }

    /**
     * @brief Cantilever tip deflection for a point load.
     *
     * Analytical: delta = P * L^3 / (3 * E * I)
     */
    static Real cantilever_deflection(Real P, Real L, Real E, Real I) {
        return P * L * L * L / (3.0 * E * I);
    }
};

// ============================================================================
// 7. RivetElement — Force-Displacement Rivet with Mixed-Mode Failure
// ============================================================================

/**
 * @brief Rivet material/connection properties.
 */
struct RivetProps {
    Real K_axial      = 1.0e6;   ///< Axial stiffness [N/m]
    Real K_shear      = 1.0e6;   ///< Shear stiffness [N/m]
    Real F_axial_max  = 5000.0;  ///< Axial failure force [N]
    Real F_shear_max  = 8000.0;  ///< Shear failure force [N]
    Real failure_disp = 0.01;    ///< Displacement at total failure [m]
    Real damping      = 0.01;    ///< Damping coefficient
};

/**
 * @brief Rivet connection element.
 *
 * Models a point-to-point connection (rivet, bolt, etc.) with:
 * - Linear elastic response in axial and shear directions
 * - Mixed-mode failure criterion:
 *     (F_axial / F_axial_max)^2 + (F_shear / F_shear_max)^2 >= 1.0
 *
 * Once the failure criterion is met, the rivet is considered failed
 * and transmits zero force.
 *
 * The rivet connects two nodes. Displacement components:
 * - delta_n: normal (axial) component along the rivet axis
 * - delta_s: tangential (shear) component perpendicular to axis
 */
class RivetElement {
public:
    RivetElement() = default;

    /**
     * @brief Compute rivet force from relative displacement.
     *
     * @param delta_n   Normal (axial) displacement [m]
     * @param delta_s   Shear displacement [m]
     * @param props     Rivet properties
     * @param F_axial   [out] Axial force [N]
     * @param F_shear   [out] Shear force [N]
     * @param failed    [out] Whether rivet has failed
     */
    void compute_rivet_force(Real delta_n, Real delta_s, const RivetProps& props,
                             Real& F_axial, Real& F_shear, bool& failed) const {
        // Linear elastic forces
        F_axial = props.K_axial * delta_n;
        F_shear = props.K_shear * delta_s;

        // Mixed-mode failure criterion
        Real ratio_a = F_axial / props.F_axial_max;
        Real ratio_s = F_shear / props.F_shear_max;
        Real failure_index = ratio_a * ratio_a + ratio_s * ratio_s;

        if (failure_index >= 1.0) {
            failed = true;
            F_axial = 0.0;
            F_shear = 0.0;
        } else {
            failed = false;
        }
    }

    /**
     * @brief Compute rivet force from 3D relative displacement vector.
     *
     * Decomposes the 3D displacement into axial and shear components
     * based on the rivet axis direction.
     *
     * @param delta     3D relative displacement vector [3]
     * @param axis      Rivet axis direction (unit vector) [3]
     * @param props     Rivet properties
     * @param force     [out] 3D force vector [3]
     * @param failed    [out] Whether rivet has failed
     */
    void compute_rivet_force_3d(const Real delta[3], const Real axis[3],
                                const RivetProps& props,
                                Real force[3], bool& failed) const {
        // Axial component: projection onto axis
        Real d_axial = specialty_detail::dot3(delta, axis);

        // Shear component: perpendicular to axis
        Real shear_vec[3];
        shear_vec[0] = delta[0] - d_axial * axis[0];
        shear_vec[1] = delta[1] - d_axial * axis[1];
        shear_vec[2] = delta[2] - d_axial * axis[2];
        Real d_shear = specialty_detail::norm3(shear_vec);

        Real F_axial, F_shear;
        compute_rivet_force(d_axial, d_shear, props, F_axial, F_shear, failed);

        if (failed) {
            force[0] = force[1] = force[2] = 0.0;
            return;
        }

        // Reconstruct 3D force
        force[0] = F_axial * axis[0];
        force[1] = F_axial * axis[1];
        force[2] = F_axial * axis[2];

        if (d_shear > 1.0e-20) {
            Real shear_scale = F_shear / d_shear;
            force[0] += shear_scale * shear_vec[0];
            force[1] += shear_scale * shear_vec[1];
            force[2] += shear_scale * shear_vec[2];
        }
    }

    /**
     * @brief Get the failure index for current forces.
     *
     * Returns (F_a/F_a_max)^2 + (F_s/F_s_max)^2.
     * Value >= 1.0 indicates failure.
     */
    static Real failure_index(Real F_axial, Real F_shear, const RivetProps& props) {
        Real ra = F_axial / props.F_axial_max;
        Real rs = F_shear / props.F_shear_max;
        return ra * ra + rs * rs;
    }

    /**
     * @brief Compute maximum shear force for a given axial force (envelope).
     *
     * From the failure surface: F_s_max_eff = F_s_max * sqrt(1 - (F_a/F_a_max)^2)
     */
    static Real max_shear_at_axial(Real F_axial, const RivetProps& props) {
        Real ra = F_axial / props.F_axial_max;
        Real factor = 1.0 - ra * ra;
        if (factor <= 0.0) return 0.0;
        return props.F_shear_max * std::sqrt(factor);
    }
};

// ============================================================================
// 8. WeldElement — Spot/Seam Weld with HAZ Softening
// ============================================================================

/**
 * @brief Weld properties.
 */
struct WeldProps {
    Real diameter     = 0.006;    ///< Weld nugget diameter [m]
    Real E_weld       = 2.1e11;   ///< Weld Young's modulus [Pa]
    Real sigma_y_weld = 600.0e6;  ///< Weld yield stress [Pa]
    Real E_haz        = 1.8e11;   ///< HAZ Young's modulus [Pa]
    Real sigma_y_haz  = 400.0e6;  ///< HAZ yield stress [Pa]
    Real haz_width    = 0.003;    ///< HAZ width around weld [m]
    Real sheet_thick  = 0.001;    ///< Connected sheet thickness [m]
    Real damage       = 0.0;      ///< Current damage parameter [0, 1]
    Real max_damage   = 1.0;      ///< Damage at which weld fails completely
    Real damage_rate  = 100.0;    ///< Damage evolution rate [1/m]
};

/**
 * @brief Spot/seam weld element with heat-affected zone (HAZ).
 *
 * The weld element connects two sheets. The force-displacement response
 * accounts for:
 * - Weld nugget: stiff elastic-plastic region (diameter d)
 * - HAZ: softer annular region around the weld (width haz_width)
 * - Progressive damage: weld force degrades with accumulated plastic strain
 *
 * Effective stiffness:
 *   K_eff = (1 - D) * pi * d * t * E_eff / 4
 *   where E_eff blends weld and HAZ moduli
 *
 * Force:
 *   F = K_eff * delta      (elastic regime)
 *   F = F_yield * (1 - D)  (plastic regime, capped)
 *
 * Damage evolution:
 *   D_new = D_old + damage_rate * |delta_plastic|
 */
class WeldElement {
public:
    WeldElement() = default;

    /**
     * @brief Compute weld force from relative displacement.
     *
     * @param delta     Relative displacement magnitude [m]
     * @param weld      Weld properties (damage may be updated)
     * @param force     [out] Weld force [N]
     */
    void compute_weld_force(Real delta, WeldProps& weld, Real& force) const {
        Real d = weld.diameter;
        Real t = weld.sheet_thick;

        // Effective cross-sectional area
        Real A_weld = 3.14159265358979 * d * d / 4.0;

        // Effective modulus: blend weld and HAZ
        Real r_weld = d / 2.0;
        Real r_outer = r_weld + weld.haz_width;
        Real A_haz = 3.14159265358979 * (r_outer * r_outer - r_weld * r_weld);
        Real A_total = A_weld + A_haz;

        Real E_eff = (weld.E_weld * A_weld + weld.E_haz * A_haz) / A_total;
        Real sigma_y_eff = (weld.sigma_y_weld * A_weld + weld.sigma_y_haz * A_haz) / A_total;

        // Effective stiffness
        Real K_eff = E_eff * A_total / std::fmax(t, 1.0e-10);

        // Yield displacement
        Real delta_y = sigma_y_eff * t / E_eff;

        Real abs_delta = std::fabs(delta);
        Real sign_delta = (delta >= 0.0) ? 1.0 : -1.0;

        // Damage factor
        Real D = weld.damage;

        if (abs_delta <= delta_y) {
            // Elastic regime
            force = (1.0 - D) * K_eff * delta;
        } else {
            // Plastic regime: capped at yield force
            Real F_yield = sigma_y_eff * A_total;
            force = (1.0 - D) * F_yield * sign_delta;

            // Damage evolution from plastic displacement
            Real delta_plastic = abs_delta - delta_y;
            weld.damage += weld.damage_rate * delta_plastic;
            weld.damage = std::fmin(weld.damage, weld.max_damage);

            // If fully damaged, zero force
            if (weld.damage >= weld.max_damage) {
                force = 0.0;
            }
        }
    }

    /**
     * @brief Compute weld stiffness (elastic, pre-damage).
     */
    Real compute_stiffness(const WeldProps& weld) const {
        Real d = weld.diameter;
        Real r_weld = d / 2.0;
        Real r_outer = r_weld + weld.haz_width;
        Real A_weld = 3.14159265358979 * d * d / 4.0;
        Real A_haz = 3.14159265358979 * (r_outer * r_outer - r_weld * r_weld);
        Real A_total = A_weld + A_haz;
        Real E_eff = (weld.E_weld * A_weld + weld.E_haz * A_haz) / A_total;
        Real t = std::fmax(weld.sheet_thick, 1.0e-10);
        return (1.0 - weld.damage) * E_eff * A_total / t;
    }

    /**
     * @brief Compute HAZ softening factor.
     *
     * Returns the ratio of HAZ yield stress to weld yield stress.
     */
    static Real haz_softening_factor(const WeldProps& weld) {
        if (weld.sigma_y_weld < 1.0e-10) return 1.0;
        return weld.sigma_y_haz / weld.sigma_y_weld;
    }

    /**
     * @brief Compute yield force.
     */
    Real yield_force(const WeldProps& weld) const {
        Real d = weld.diameter;
        Real r_weld = d / 2.0;
        Real r_outer = r_weld + weld.haz_width;
        Real A_weld = 3.14159265358979 * d * d / 4.0;
        Real A_haz = 3.14159265358979 * (r_outer * r_outer - r_weld * r_weld);
        Real A_total = A_weld + A_haz;
        Real sigma_y_eff = (weld.sigma_y_weld * A_weld + weld.sigma_y_haz * A_haz) / A_total;
        return sigma_y_eff * A_total;
    }
};

// ============================================================================
// 9. GeneralSpringBeam (LAW113) — Nonlinear Spring-Beam, 6-DOF Tabulated
// ============================================================================

/**
 * @brief Spring-beam properties with independent load curves per DOF.
 *
 * Each DOF (3 translations + 3 rotations) has:
 * - An initial stiffness K[i]
 * - A tabulated force-displacement (or moment-rotation) curve
 *
 * The curve_x[i][j] stores displacement/rotation values,
 * curve_y[i][j] stores force/moment values.
 */
struct SpringBeamProps {
    Real K[6]             = {1.0e6, 1.0e6, 1.0e6, 1.0e3, 1.0e3, 1.0e3};
    Real curve_x[6][32]   = {};    ///< Displacement/rotation data per DOF
    Real curve_y[6][32]   = {};    ///< Force/moment data per DOF
    int  n_pts[6]         = {};    ///< Number of curve points per DOF
    Real damping[6]       = {};    ///< Damping coefficients per DOF
    bool use_curve[6]     = {};    ///< Whether to use tabulated curve (vs. linear K)
};

/**
 * @brief Nonlinear spring-beam element (LAW113).
 *
 * A 6-DOF connection element where each DOF has an independent
 * force-displacement (or moment-rotation) relationship defined by
 * a tabulated curve. If no curve is provided for a DOF, a linear
 * spring with stiffness K[i] is used.
 *
 * DOF mapping:
 *   0: axial (x-translation)    -> F_x(u_x)
 *   1: lateral y (y-translation) -> F_y(u_y)
 *   2: lateral z (z-translation) -> F_z(u_z)
 *   3: torsion (x-rotation)     -> M_x(theta_x)
 *   4: bending y (y-rotation)    -> M_y(theta_y)
 *   5: bending z (z-rotation)    -> M_z(theta_z)
 */
class GeneralSpringBeam {
public:
    GeneralSpringBeam() = default;

    /**
     * @brief Compute spring forces and moments from displacements and rotations.
     *
     * @param displ    Translational displacements [3]: (u_x, u_y, u_z)
     * @param rot      Rotational displacements [3]: (theta_x, theta_y, theta_z)
     * @param props    Spring-beam properties
     * @param force    [out] Force vector [3]: (F_x, F_y, F_z)
     * @param moment   [out] Moment vector [3]: (M_x, M_y, M_z)
     */
    void compute_spring_force(const Real displ[3], const Real rot[3],
                              const SpringBeamProps& props,
                              Real force[3], Real moment[3]) const {
        Real inputs[6] = {displ[0], displ[1], displ[2], rot[0], rot[1], rot[2]};
        Real outputs[6];

        for (int i = 0; i < 6; ++i) {
            if (props.use_curve[i] && props.n_pts[i] > 1) {
                outputs[i] = specialty_detail::interp_table(
                    props.curve_x[i], props.curve_y[i], props.n_pts[i], inputs[i]);
            } else {
                outputs[i] = props.K[i] * inputs[i];
            }
        }

        force[0]  = outputs[0];
        force[1]  = outputs[1];
        force[2]  = outputs[2];
        moment[0] = outputs[3];
        moment[1] = outputs[4];
        moment[2] = outputs[5];
    }

    /**
     * @brief Compute spring force with velocity-dependent damping.
     *
     * F_total = F_spring + C * v
     *
     * @param displ     Translational displacements [3]
     * @param rot       Rotational displacements [3]
     * @param vel       Translational velocities [3]
     * @param omega     Rotational velocities [3]
     * @param props     Spring-beam properties
     * @param force     [out] Force vector [3]
     * @param moment    [out] Moment vector [3]
     */
    void compute_spring_force_damped(const Real displ[3], const Real rot[3],
                                     const Real vel[3], const Real omega[3],
                                     const SpringBeamProps& props,
                                     Real force[3], Real moment[3]) const {
        // Spring forces
        compute_spring_force(displ, rot, props, force, moment);

        // Add damping
        Real rates[6] = {vel[0], vel[1], vel[2], omega[0], omega[1], omega[2]};
        Real* outputs[6] = {&force[0], &force[1], &force[2],
                            &moment[0], &moment[1], &moment[2]};
        for (int i = 0; i < 6; ++i) {
            *outputs[i] += props.damping[i] * rates[i];
        }
    }

    /**
     * @brief Compute tangent stiffness for a DOF.
     *
     * For tabulated curves, returns the local slope.
     * For linear springs, returns K[i].
     */
    Real tangent_stiffness(int dof, Real displacement,
                           const SpringBeamProps& props) const {
        if (dof < 0 || dof >= 6) return 0.0;

        if (props.use_curve[dof] && props.n_pts[dof] > 1) {
            // Numerical derivative via central difference
            Real h = 1.0e-8;
            Real f_plus = specialty_detail::interp_table(
                props.curve_x[dof], props.curve_y[dof], props.n_pts[dof],
                displacement + h);
            Real f_minus = specialty_detail::interp_table(
                props.curve_x[dof], props.curve_y[dof], props.n_pts[dof],
                displacement - h);
            return (f_plus - f_minus) / (2.0 * h);
        }

        return props.K[dof];
    }

    /**
     * @brief Compute energy stored in a single DOF.
     *
     * For linear spring: E = 0.5 * K * d^2
     * For tabulated curve: E = integral of F(d) dd (trapezoidal)
     */
    Real compute_energy(int dof, Real displacement,
                        const SpringBeamProps& props) const {
        if (dof < 0 || dof >= 6) return 0.0;

        if (props.use_curve[dof] && props.n_pts[dof] > 1) {
            // Trapezoidal integration of F(d) from 0 to displacement
            int n_steps = 100;
            Real dd = displacement / n_steps;
            Real energy = 0.0;
            for (int i = 0; i < n_steps; ++i) {
                Real d0 = i * dd;
                Real d1 = (i + 1) * dd;
                Real f0 = specialty_detail::interp_table(
                    props.curve_x[dof], props.curve_y[dof], props.n_pts[dof], d0);
                Real f1 = specialty_detail::interp_table(
                    props.curve_x[dof], props.curve_y[dof], props.n_pts[dof], d1);
                energy += 0.5 * (f0 + f1) * dd;
            }
            return energy;
        }

        return 0.5 * props.K[dof] * displacement * displacement;
    }
};

} // namespace discretization
} // namespace nxs
