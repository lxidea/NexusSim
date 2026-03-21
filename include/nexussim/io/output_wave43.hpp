#pragma once

/**
 * @file output_wave43.hpp
 * @brief Wave 43: Per-entity output extractors
 *
 * Provides specialized result extractors for each entity type,
 * mirroring the 100+ anim-file per-entity result extraction found
 * in OpenRadioss. NexusSim's generic field output is augmented with:
 *
 *  1. NodeResultExtractor        - displacement, velocity, acceleration,
 *                                  reaction forces
 *  2. ShellResultExtractor       - von Mises stress, thickness reduction,
 *                                  fiber stress (top/bottom/mid)
 *  3. SolidResultExtractor       - stress, principal stress, pressure
 *  4. SPHResultExtractor         - density, smoothing length
 *  5. BeamResultExtractor        - axial force, bending moment
 *  6. RigidBodyExtractor         - CoM position, CoM velocity, angular velocity
 *  7. InterfaceForceExtractor    - contact force, contact gap
 *  8. CrackResultExtractor       - crack length, stress intensity factors
 *  9. SectionForceExtractor      - section force, section moment
 * 10. OutputDispatcher           - registry + dispatch by EntityType/ResultFieldType
 *
 * Reference: OpenRadioss anim output specification (ANIM/ELEM/*, ANIM/NODA/*),
 *            OpenRadioss Starter & Engine user guides.
 */

#include <nexussim/core/types.hpp>
#include <nexussim/physics/material.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <array>
#include <functional>
#include <map>

namespace nxs {
namespace io {

using Real = nxs::Real;

// ============================================================================
// ResultFieldType — identifies what quantity is stored in a ResultField
// ============================================================================

enum class ResultFieldType {
    Displacement,
    Velocity,
    Acceleration,
    VonMisesStress,
    PrincipalStress,
    PrincipalStrain,
    PlasticStrain,
    Damage,
    Temperature,
    Pressure,
    Density,
    Energy,
    ContactForce,
    ContactGap,
    ReactionForce,
    SectionForce,
    SectionMoment,
    CrackLength,
    CrackAngle,
    StressIntensityFactor,
    SPHDensity,
    SPHPressure,
    SPHSmoothingLength,
    ThicknessReduction,
    HourglassEnergy,
    AxialForce,
    BendingMoment,
    AngularVelocity,
    COMPosition,
    COMVelocity,
    FiberStress,
};

// ============================================================================
// ResultField — a named, typed field with flattened component data
// ============================================================================

/**
 * Holds a single result field for all entities.
 *
 * data layout:
 *   num_components == 1: data[i]           = scalar for entity i
 *   num_components == 3: data[3*i .. 3*i+2] = (x,y,z) for entity i
 *   num_components == 6: data[6*i .. 6*i+5] = Voigt tensor for entity i
 */
struct ResultField {
    std::string name;
    ResultFieldType type;
    int num_components = 1;     ///< 1 (scalar), 3 (vector), or 6 (tensor)
    std::vector<Real> data;     ///< Flattened: size = num_entities * num_components

    /// Total number of entities stored.
    int num_entities() const {
        if (num_components <= 0) return 0;
        return static_cast<int>(data.size()) / num_components;
    }

    /// Scalar value at entity index i.
    Real scalar_at(int i) const { return data[i]; }

    /// Vector component at entity index i, component c.
    Real component_at(int i, int c) const {
        return data[static_cast<std::size_t>(i) * num_components + c];
    }
};

// ============================================================================
// Supporting data structs used by extractors
// ============================================================================

/// Per-shell state for extraction (stress tensor, plastic strain, thickness)
struct ShellState {
    std::array<Real, 6> stress = {};   ///< Voigt: xx yy zz xy yz xz
    std::array<Real, 6> stress_top = {};
    std::array<Real, 6> stress_bot = {};
    Real plastic_strain = 0;
    Real thickness = 0;                ///< Current thickness
};

/// Per-solid state for extraction
struct SolidState {
    std::array<Real, 6> stress = {};   ///< Voigt: xx yy zz xy yz xz
    std::array<Real, 6> strain = {};
    Real plastic_strain = 0;
    Real density = 0;
    Real energy = 0;
    Real damage = 0;
};

/// Per-beam state
struct BeamState {
    Real axial_force = 0;
    Real shear_y = 0;
    Real shear_z = 0;
    Real bending_my = 0;
    Real bending_mz = 0;
    Real torsion = 0;
};

/// Per-rigid-body data
struct RigidBodyData {
    Real cx = 0, cy = 0, cz = 0;      ///< CoM position
    Real vx = 0, vy = 0, vz = 0;      ///< CoM velocity
    Real wx = 0, wy = 0, wz = 0;      ///< Angular velocity
};

/// Per-interface (contact) data
struct InterfaceData {
    Real force_x = 0, force_y = 0, force_z = 0;
    Real gap = 0;
    Real pressure = 0;
};

/// Per-crack data (XFEM)
struct CrackData {
    Real length = 0;
    Real angle_deg = 0;
    Real sif_I = 0;    ///< Mode-I stress intensity factor
    Real sif_II = 0;   ///< Mode-II stress intensity factor
    Real sif_III = 0;  ///< Mode-III stress intensity factor
};

/// Per-section data for cross-section forces/moments
struct SectionData {
    Real fx = 0, fy = 0, fz = 0;
    Real mx = 0, my = 0, mz = 0;
};

// ============================================================================
// Shell surface selector for fiber/ply stress
// ============================================================================
enum class ShellSurface {
    Top,
    Bottom,
    Mid
};

// ============================================================================
// 1. NodeResultExtractor
// ============================================================================

/**
 * Extracts per-node result fields from nodal arrays.
 * All arrays are indexed [node_index * 3 + component] for vector quantities.
 */
class NodeResultExtractor {
public:
    /// Extract displacement as current - reference position (3 components/node).
    static ResultField extract_displacement(const std::vector<Real>& positions,
                                            const std::vector<Real>& ref_positions,
                                            int num_nodes) {
        ResultField rf;
        rf.name = "Displacement";
        rf.type = ResultFieldType::Displacement;
        rf.num_components = 3;
        rf.data.resize(static_cast<std::size_t>(num_nodes) * 3, Real(0));

        const int n3 = num_nodes * 3;
        const int avail_pos = static_cast<int>(positions.size());
        const int avail_ref = static_cast<int>(ref_positions.size());

        for (int i = 0; i < n3; ++i) {
            Real cur = (i < avail_pos) ? positions[i] : Real(0);
            Real ref = (i < avail_ref) ? ref_positions[i] : Real(0);
            rf.data[i] = cur - ref;
        }
        return rf;
    }

    /// Extract velocity field (3 components/node).
    static ResultField extract_velocity(const std::vector<Real>& velocities,
                                        int num_nodes) {
        ResultField rf;
        rf.name = "Velocity";
        rf.type = ResultFieldType::Velocity;
        rf.num_components = 3;
        rf.data.resize(static_cast<std::size_t>(num_nodes) * 3, Real(0));

        const int n3 = num_nodes * 3;
        const int avail = static_cast<int>(velocities.size());
        for (int i = 0; i < n3 && i < avail; ++i) {
            rf.data[i] = velocities[i];
        }
        return rf;
    }

    /// Extract acceleration field (3 components/node).
    static ResultField extract_acceleration(const std::vector<Real>& accelerations,
                                            int num_nodes) {
        ResultField rf;
        rf.name = "Acceleration";
        rf.type = ResultFieldType::Acceleration;
        rf.num_components = 3;
        rf.data.resize(static_cast<std::size_t>(num_nodes) * 3, Real(0));

        const int n3 = num_nodes * 3;
        const int avail = static_cast<int>(accelerations.size());
        for (int i = 0; i < n3 && i < avail; ++i) {
            rf.data[i] = accelerations[i];
        }
        return rf;
    }

    /**
     * Extract reaction forces at constrained nodes.
     * @param forces          Full nodal force array [num_nodes * 3]
     * @param constrained_nodes  Indices of constrained nodes (reaction known)
     * @param num_nodes       Total number of nodes
     *
     * Output: data sized num_nodes * 3; non-constrained entries are zero.
     */
    static ResultField extract_reaction_forces(const std::vector<Real>& forces,
                                               const std::vector<int>& constrained_nodes,
                                               int num_nodes) {
        ResultField rf;
        rf.name = "ReactionForce";
        rf.type = ResultFieldType::ReactionForce;
        rf.num_components = 3;
        rf.data.resize(static_cast<std::size_t>(num_nodes) * 3, Real(0));

        const int avail = static_cast<int>(forces.size());
        for (int nid : constrained_nodes) {
            if (nid < 0 || nid >= num_nodes) continue;
            for (int c = 0; c < 3; ++c) {
                int idx = nid * 3 + c;
                if (idx < avail) {
                    rf.data[idx] = forces[idx];
                }
            }
        }
        return rf;
    }
};

// ============================================================================
// 2. ShellResultExtractor
// ============================================================================

/**
 * Extracts per-shell-element result fields.
 */
class ShellResultExtractor {
public:
    /// Von Mises stress from mid-plane stress tensor (scalar/element).
    static ResultField extract_stress(const std::vector<ShellState>& states,
                                      int num_elements) {
        ResultField rf;
        rf.name = "VonMisesStress";
        rf.type = ResultFieldType::VonMisesStress;
        rf.num_components = 1;
        rf.data.resize(num_elements, Real(0));

        for (int i = 0; i < num_elements && i < static_cast<int>(states.size()); ++i) {
            rf.data[i] = von_mises(states[i].stress);
        }
        return rf;
    }

    /**
     * Extract thickness reduction as (t0 - t_current) / t0.
     * Returns scalar in [0, 1].
     */
    static ResultField extract_thickness(const std::vector<Real>& t0,
                                         const std::vector<Real>& t_current,
                                         int num_elements) {
        ResultField rf;
        rf.name = "ThicknessReduction";
        rf.type = ResultFieldType::ThicknessReduction;
        rf.num_components = 1;
        rf.data.resize(num_elements, Real(0));

        for (int i = 0; i < num_elements; ++i) {
            Real t_ref = (i < static_cast<int>(t0.size())) ? t0[i] : Real(0);
            Real t_cur = (i < static_cast<int>(t_current.size())) ? t_current[i] : Real(0);
            if (std::abs(t_ref) > Real(1e-30)) {
                rf.data[i] = (t_ref - t_cur) / t_ref;
            }
        }
        return rf;
    }

    /**
     * Extract fiber stress (von Mises) at a chosen surface.
     * surface: Top, Bottom, or Mid (mid uses states[i].stress)
     */
    static ResultField extract_fiber_stress(const std::vector<ShellState>& states,
                                            int num_elements,
                                            ShellSurface surface = ShellSurface::Mid) {
        ResultField rf;
        rf.num_components = 1;
        rf.data.resize(num_elements, Real(0));

        switch (surface) {
            case ShellSurface::Top:
                rf.name = "FiberStress_Top";
                rf.type = ResultFieldType::FiberStress;
                for (int i = 0; i < num_elements && i < static_cast<int>(states.size()); ++i) {
                    rf.data[i] = von_mises(states[i].stress_top);
                }
                break;
            case ShellSurface::Bottom:
                rf.name = "FiberStress_Bot";
                rf.type = ResultFieldType::FiberStress;
                for (int i = 0; i < num_elements && i < static_cast<int>(states.size()); ++i) {
                    rf.data[i] = von_mises(states[i].stress_bot);
                }
                break;
            case ShellSurface::Mid:
            default:
                rf.name = "FiberStress_Mid";
                rf.type = ResultFieldType::FiberStress;
                for (int i = 0; i < num_elements && i < static_cast<int>(states.size()); ++i) {
                    rf.data[i] = von_mises(states[i].stress);
                }
                break;
        }
        return rf;
    }

    /// Plastic strain (scalar/element).
    static ResultField extract_plastic_strain(const std::vector<ShellState>& states,
                                              int num_elements) {
        ResultField rf;
        rf.name = "PlasticStrain";
        rf.type = ResultFieldType::PlasticStrain;
        rf.num_components = 1;
        rf.data.resize(num_elements, Real(0));
        for (int i = 0; i < num_elements && i < static_cast<int>(states.size()); ++i) {
            rf.data[i] = states[i].plastic_strain;
        }
        return rf;
    }

private:
    /// Von Mises stress from Voigt tensor [xx, yy, zz, xy, yz, xz].
    static Real von_mises(const std::array<Real, 6>& s) {
        Real sxx = s[0], syy = s[1], szz = s[2];
        Real sxy = s[3], syz = s[4], sxz = s[5];
        return std::sqrt(Real(0.5) * (
            (sxx - syy) * (sxx - syy) +
            (syy - szz) * (syy - szz) +
            (szz - sxx) * (szz - sxx) +
            Real(6) * (sxy * sxy + syz * syz + sxz * sxz)
        ));
    }
};

// ============================================================================
// 3. SolidResultExtractor
// ============================================================================

/**
 * Extracts per-solid-element result fields.
 */
class SolidResultExtractor {
public:
    /// Von Mises stress (scalar/element).
    static ResultField extract_stress(const std::vector<SolidState>& states,
                                      int num_elements) {
        ResultField rf;
        rf.name = "VonMisesStress";
        rf.type = ResultFieldType::VonMisesStress;
        rf.num_components = 1;
        rf.data.resize(num_elements, Real(0));

        for (int i = 0; i < num_elements && i < static_cast<int>(states.size()); ++i) {
            rf.data[i] = von_mises(states[i].stress);
        }
        return rf;
    }

    /**
     * Extract principal stresses (3 components/element: sigma_1, sigma_2, sigma_3).
     *
     * The principal stresses are the eigenvalues of the symmetric 3x3 stress
     * tensor. We use the closed-form analytical solution (Cardano's method)
     * for the characteristic polynomial:
     *   lambda^3 - I1*lambda^2 + I2*lambda - I3 = 0
     */
    static ResultField extract_principal_stress(const std::vector<SolidState>& states,
                                                int num_elements) {
        ResultField rf;
        rf.name = "PrincipalStress";
        rf.type = ResultFieldType::PrincipalStress;
        rf.num_components = 3;
        rf.data.resize(static_cast<std::size_t>(num_elements) * 3, Real(0));

        for (int i = 0; i < num_elements && i < static_cast<int>(states.size()); ++i) {
            std::array<Real, 3> p = principal_stresses(states[i].stress);
            rf.data[i * 3 + 0] = p[0];
            rf.data[i * 3 + 1] = p[1];
            rf.data[i * 3 + 2] = p[2];
        }
        return rf;
    }

    /**
     * Extract hydrostatic pressure p = -trace(sigma)/3.
     * Positive p means compression (OpenRadioss sign convention).
     */
    static ResultField extract_pressure(const std::vector<SolidState>& states,
                                        int num_elements) {
        ResultField rf;
        rf.name = "Pressure";
        rf.type = ResultFieldType::Pressure;
        rf.num_components = 1;
        rf.data.resize(num_elements, Real(0));

        for (int i = 0; i < num_elements && i < static_cast<int>(states.size()); ++i) {
            const auto& s = states[i].stress;
            rf.data[i] = -(s[0] + s[1] + s[2]) / Real(3);
        }
        return rf;
    }

    /// Plastic strain (scalar/element).
    static ResultField extract_plastic_strain(const std::vector<SolidState>& states,
                                              int num_elements) {
        ResultField rf;
        rf.name = "PlasticStrain";
        rf.type = ResultFieldType::PlasticStrain;
        rf.num_components = 1;
        rf.data.resize(num_elements, Real(0));
        for (int i = 0; i < num_elements && i < static_cast<int>(states.size()); ++i) {
            rf.data[i] = states[i].plastic_strain;
        }
        return rf;
    }

    /// Damage (scalar/element).
    static ResultField extract_damage(const std::vector<SolidState>& states,
                                      int num_elements) {
        ResultField rf;
        rf.name = "Damage";
        rf.type = ResultFieldType::Damage;
        rf.num_components = 1;
        rf.data.resize(num_elements, Real(0));
        for (int i = 0; i < num_elements && i < static_cast<int>(states.size()); ++i) {
            rf.data[i] = states[i].damage;
        }
        return rf;
    }

    /// Element density (scalar/element).
    static ResultField extract_density(const std::vector<SolidState>& states,
                                       int num_elements) {
        ResultField rf;
        rf.name = "Density";
        rf.type = ResultFieldType::Density;
        rf.num_components = 1;
        rf.data.resize(num_elements, Real(0));
        for (int i = 0; i < num_elements && i < static_cast<int>(states.size()); ++i) {
            rf.data[i] = states[i].density;
        }
        return rf;
    }

private:
    static Real von_mises(const std::array<Real, 6>& s) {
        Real sxx = s[0], syy = s[1], szz = s[2];
        Real sxy = s[3], syz = s[4], sxz = s[5];
        return std::sqrt(Real(0.5) * (
            (sxx - syy) * (sxx - syy) +
            (syy - szz) * (syy - szz) +
            (szz - sxx) * (szz - sxx) +
            Real(6) * (sxy * sxy + syz * syz + sxz * sxz)
        ));
    }

    /**
     * Compute sorted eigenvalues (principal stresses) of a symmetric 3×3 tensor
     * given in Voigt notation [xx, yy, zz, xy, yz, xz].
     * Returns {sigma_max, sigma_mid, sigma_min} (descending).
     *
     * Uses the trigonometric (Vieta / Kopp) form which is valid for all real
     * symmetric matrices — including the D=0 repeated-root case — avoiding the
     * numerical instability of Cardano's formula when the discriminant is near
     * zero.  Implementation follows:
     *   Smith (1961), Kopp (2008 arXiv:physics/0610206), and
     *   OpenRadioss meca_src/material/stress_tensor.F.
     */
    static std::array<Real, 3> principal_stresses(const std::array<Real, 6>& s) {
        Real sxx = s[0], syy = s[1], szz = s[2];
        Real sxy = s[3], syz = s[4], sxz = s[5];

        // Mean (hydrostatic) stress
        Real mean = (sxx + syy + szz) / Real(3);

        // Deviatoric tensor components
        Real dxx = sxx - mean;
        Real dyy = syy - mean;
        Real dzz = szz - mean;

        // J2 = 0.5 * tr(s_dev^2)
        Real J2 = Real(0.5) * (dxx * dxx + dyy * dyy + dzz * dzz)
                + sxy * sxy + syz * syz + sxz * sxz;

        // J3 = det(s_dev)
        Real J3 = dxx * (dyy * dzz - syz * syz)
                - sxy * (sxy * dzz - syz * sxz)
                + sxz * (sxy * syz - dyy * sxz);

        const Real pi = Real(3.14159265358979323846);

        if (J2 < Real(1e-30)) {
            // Effectively hydrostatic — all three principal stresses equal mean
            return {mean, mean, mean};
        }

        // r = sqrt(J2/3),  factor = 2*r
        Real r   = std::sqrt(J2 / Real(3));
        Real fac = Real(2) * r;

        // Lode angle theta in [0, pi/3]
        // cos(3*theta) = J3 / (2 * r^3)
        Real cos3theta = J3 / (Real(2) * r * r * r);
        cos3theta = std::max(Real(-1), std::min(Real(1), cos3theta));
        Real theta = std::acos(cos3theta) / Real(3);

        // Three roots of the depressed cubic
        std::array<Real, 3> roots;
        roots[0] = mean + fac * std::cos(theta);
        roots[1] = mean + fac * std::cos(theta + Real(2) * pi / Real(3));
        roots[2] = mean + fac * std::cos(theta + Real(4) * pi / Real(3));

        // Sort descending: sigma_1 >= sigma_2 >= sigma_3
        if (roots[0] < roots[1]) std::swap(roots[0], roots[1]);
        if (roots[1] < roots[2]) std::swap(roots[1], roots[2]);
        if (roots[0] < roots[1]) std::swap(roots[0], roots[1]);
        return roots;
    }
};

// ============================================================================
// 4. SPHResultExtractor
// ============================================================================

/**
 * Extracts per-SPH-particle result fields.
 */
class SPHResultExtractor {
public:
    /// SPH particle density (scalar/particle).
    static ResultField extract_density(const std::vector<Real>& sph_densities,
                                       int num_particles) {
        ResultField rf;
        rf.name = "SPHDensity";
        rf.type = ResultFieldType::SPHDensity;
        rf.num_components = 1;
        rf.data.resize(num_particles, Real(0));

        const int avail = static_cast<int>(sph_densities.size());
        for (int i = 0; i < num_particles && i < avail; ++i) {
            rf.data[i] = sph_densities[i];
        }
        return rf;
    }

    /// SPH smoothing length h (scalar/particle).
    static ResultField extract_smoothing_length(const std::vector<Real>& h_values,
                                                int num_particles) {
        ResultField rf;
        rf.name = "SPHSmoothingLength";
        rf.type = ResultFieldType::SPHSmoothingLength;
        rf.num_components = 1;
        rf.data.resize(num_particles, Real(0));

        const int avail = static_cast<int>(h_values.size());
        for (int i = 0; i < num_particles && i < avail; ++i) {
            rf.data[i] = h_values[i];
        }
        return rf;
    }

    /// SPH pressure (scalar/particle, from a pre-computed pressure array).
    static ResultField extract_pressure(const std::vector<Real>& pressures,
                                        int num_particles) {
        ResultField rf;
        rf.name = "SPHPressure";
        rf.type = ResultFieldType::SPHPressure;
        rf.num_components = 1;
        rf.data.resize(num_particles, Real(0));

        const int avail = static_cast<int>(pressures.size());
        for (int i = 0; i < num_particles && i < avail; ++i) {
            rf.data[i] = pressures[i];
        }
        return rf;
    }
};

// ============================================================================
// 5. BeamResultExtractor
// ============================================================================

/**
 * Extracts per-beam-element result fields.
 */
class BeamResultExtractor {
public:
    /// Axial force N (scalar/element).
    static ResultField extract_axial_force(const std::vector<BeamState>& states,
                                           int num_elements) {
        ResultField rf;
        rf.name = "AxialForce";
        rf.type = ResultFieldType::AxialForce;
        rf.num_components = 1;
        rf.data.resize(num_elements, Real(0));

        for (int i = 0; i < num_elements && i < static_cast<int>(states.size()); ++i) {
            rf.data[i] = states[i].axial_force;
        }
        return rf;
    }

    /// Resultant bending moment sqrt(My^2 + Mz^2) (scalar/element).
    static ResultField extract_bending_moment(const std::vector<BeamState>& states,
                                              int num_elements) {
        ResultField rf;
        rf.name = "BendingMoment";
        rf.type = ResultFieldType::BendingMoment;
        rf.num_components = 1;
        rf.data.resize(num_elements, Real(0));

        for (int i = 0; i < num_elements && i < static_cast<int>(states.size()); ++i) {
            Real my = states[i].bending_my;
            Real mz = states[i].bending_mz;
            rf.data[i] = std::sqrt(my * my + mz * mz);
        }
        return rf;
    }

    /// Torsional moment (scalar/element).
    static ResultField extract_torsion(const std::vector<BeamState>& states,
                                       int num_elements) {
        ResultField rf;
        rf.name = "Torsion";
        rf.type = ResultFieldType::BendingMoment;
        rf.num_components = 1;
        rf.data.resize(num_elements, Real(0));

        for (int i = 0; i < num_elements && i < static_cast<int>(states.size()); ++i) {
            rf.data[i] = states[i].torsion;
        }
        return rf;
    }
};

// ============================================================================
// 6. RigidBodyExtractor
// ============================================================================

/**
 * Extracts per-rigid-body result fields.
 */
class RigidBodyExtractor {
public:
    /// CoM position (3 components/body).
    static ResultField extract_com_position(const std::vector<RigidBodyData>& rb_data,
                                            int num_bodies) {
        ResultField rf;
        rf.name = "COMPosition";
        rf.type = ResultFieldType::COMPosition;
        rf.num_components = 3;
        rf.data.resize(static_cast<std::size_t>(num_bodies) * 3, Real(0));

        for (int i = 0; i < num_bodies && i < static_cast<int>(rb_data.size()); ++i) {
            rf.data[i * 3 + 0] = rb_data[i].cx;
            rf.data[i * 3 + 1] = rb_data[i].cy;
            rf.data[i * 3 + 2] = rb_data[i].cz;
        }
        return rf;
    }

    /// CoM velocity (3 components/body).
    static ResultField extract_com_velocity(const std::vector<RigidBodyData>& rb_data,
                                            int num_bodies) {
        ResultField rf;
        rf.name = "COMVelocity";
        rf.type = ResultFieldType::COMVelocity;
        rf.num_components = 3;
        rf.data.resize(static_cast<std::size_t>(num_bodies) * 3, Real(0));

        for (int i = 0; i < num_bodies && i < static_cast<int>(rb_data.size()); ++i) {
            rf.data[i * 3 + 0] = rb_data[i].vx;
            rf.data[i * 3 + 1] = rb_data[i].vy;
            rf.data[i * 3 + 2] = rb_data[i].vz;
        }
        return rf;
    }

    /// Angular velocity omega (3 components/body).
    static ResultField extract_angular_velocity(const std::vector<RigidBodyData>& rb_data,
                                                int num_bodies) {
        ResultField rf;
        rf.name = "AngularVelocity";
        rf.type = ResultFieldType::AngularVelocity;
        rf.num_components = 3;
        rf.data.resize(static_cast<std::size_t>(num_bodies) * 3, Real(0));

        for (int i = 0; i < num_bodies && i < static_cast<int>(rb_data.size()); ++i) {
            rf.data[i * 3 + 0] = rb_data[i].wx;
            rf.data[i * 3 + 1] = rb_data[i].wy;
            rf.data[i * 3 + 2] = rb_data[i].wz;
        }
        return rf;
    }
};

// ============================================================================
// 7. InterfaceForceExtractor
// ============================================================================

/**
 * Extracts contact interface results.
 */
class InterfaceForceExtractor {
public:
    /// Contact force magnitude (scalar/interface) and vector (3 comp/interface).
    static ResultField extract_contact_force(const std::vector<InterfaceData>& idata,
                                             int num_interfaces) {
        ResultField rf;
        rf.name = "ContactForce";
        rf.type = ResultFieldType::ContactForce;
        rf.num_components = 3;
        rf.data.resize(static_cast<std::size_t>(num_interfaces) * 3, Real(0));

        for (int i = 0; i < num_interfaces && i < static_cast<int>(idata.size()); ++i) {
            rf.data[i * 3 + 0] = idata[i].force_x;
            rf.data[i * 3 + 1] = idata[i].force_y;
            rf.data[i * 3 + 2] = idata[i].force_z;
        }
        return rf;
    }

    /// Contact gap (scalar/interface).
    static ResultField extract_contact_gap(const std::vector<InterfaceData>& idata,
                                           int num_interfaces) {
        ResultField rf;
        rf.name = "ContactGap";
        rf.type = ResultFieldType::ContactGap;
        rf.num_components = 1;
        rf.data.resize(num_interfaces, Real(0));

        for (int i = 0; i < num_interfaces && i < static_cast<int>(idata.size()); ++i) {
            rf.data[i] = idata[i].gap;
        }
        return rf;
    }
};

// ============================================================================
// 8. CrackResultExtractor
// ============================================================================

/**
 * Extracts XFEM crack result fields.
 */
class CrackResultExtractor {
public:
    /// Crack length (scalar/crack).
    static ResultField extract_crack_length(const std::vector<CrackData>& crack_data,
                                            int num_cracks) {
        ResultField rf;
        rf.name = "CrackLength";
        rf.type = ResultFieldType::CrackLength;
        rf.num_components = 1;
        rf.data.resize(num_cracks, Real(0));

        for (int i = 0; i < num_cracks && i < static_cast<int>(crack_data.size()); ++i) {
            rf.data[i] = crack_data[i].length;
        }
        return rf;
    }

    /**
     * Extract stress intensity factors.
     * Returns 3 components/crack: K_I, K_II, K_III.
     */
    static ResultField extract_sif(const std::vector<CrackData>& crack_data,
                                   int num_cracks) {
        ResultField rf;
        rf.name = "StressIntensityFactor";
        rf.type = ResultFieldType::StressIntensityFactor;
        rf.num_components = 3;
        rf.data.resize(static_cast<std::size_t>(num_cracks) * 3, Real(0));

        for (int i = 0; i < num_cracks && i < static_cast<int>(crack_data.size()); ++i) {
            rf.data[i * 3 + 0] = crack_data[i].sif_I;
            rf.data[i * 3 + 1] = crack_data[i].sif_II;
            rf.data[i * 3 + 2] = crack_data[i].sif_III;
        }
        return rf;
    }

    /// Crack angle in degrees (scalar/crack).
    static ResultField extract_crack_angle(const std::vector<CrackData>& crack_data,
                                           int num_cracks) {
        ResultField rf;
        rf.name = "CrackAngle";
        rf.type = ResultFieldType::CrackAngle;
        rf.num_components = 1;
        rf.data.resize(num_cracks, Real(0));

        for (int i = 0; i < num_cracks && i < static_cast<int>(crack_data.size()); ++i) {
            rf.data[i] = crack_data[i].angle_deg;
        }
        return rf;
    }
};

// ============================================================================
// 9. SectionForceExtractor
// ============================================================================

/**
 * Extracts cross-section force and moment resultants.
 */
class SectionForceExtractor {
public:
    /// Section force vector (3 components/section: Fx, Fy, Fz).
    static ResultField extract_section_force(const std::vector<SectionData>& sdata,
                                             int num_sections) {
        ResultField rf;
        rf.name = "SectionForce";
        rf.type = ResultFieldType::SectionForce;
        rf.num_components = 3;
        rf.data.resize(static_cast<std::size_t>(num_sections) * 3, Real(0));

        for (int i = 0; i < num_sections && i < static_cast<int>(sdata.size()); ++i) {
            rf.data[i * 3 + 0] = sdata[i].fx;
            rf.data[i * 3 + 1] = sdata[i].fy;
            rf.data[i * 3 + 2] = sdata[i].fz;
        }
        return rf;
    }

    /// Section moment vector (3 components/section: Mx, My, Mz).
    static ResultField extract_section_moment(const std::vector<SectionData>& sdata,
                                              int num_sections) {
        ResultField rf;
        rf.name = "SectionMoment";
        rf.type = ResultFieldType::SectionMoment;
        rf.num_components = 3;
        rf.data.resize(static_cast<std::size_t>(num_sections) * 3, Real(0));

        for (int i = 0; i < num_sections && i < static_cast<int>(sdata.size()); ++i) {
            rf.data[i * 3 + 0] = sdata[i].mx;
            rf.data[i * 3 + 1] = sdata[i].my;
            rf.data[i * 3 + 2] = sdata[i].mz;
        }
        return rf;
    }
};

// ============================================================================
// 10. OutputDispatcher — registry + dispatch by EntityType / ResultFieldType
// ============================================================================

/// Entity type identifiers for the dispatcher.
enum class EntityType {
    Node,
    Shell,
    Solid,
    Beam,
    SPH,
    RigidBody,
    Interface,
    Crack,
    Section
};

/**
 * OutputDispatcher routes a (EntityType, ResultFieldType) pair to the
 * correct extractor.  Users register lambda extractors and call extract().
 *
 * The generic interface passes a void* payload and an integer count;
 * the registered functor casts to the correct type.
 *
 * Usage example:
 * @code
 *   OutputDispatcher disp;
 *   disp.register_extractor(EntityType::Solid, ResultFieldType::VonMisesStress,
 *       [&](const void* data, int n) {
 *           auto* states = static_cast<const SolidState*>(data);
 *           std::vector<SolidState> sv(states, states + n);
 *           return SolidResultExtractor::extract_stress(sv, n);
 *       });
 *   ResultField rf = disp.extract(EntityType::Solid,
 *                                 ResultFieldType::VonMisesStress,
 *                                 states.data(), (int)states.size());
 * @endcode
 */
class OutputDispatcher {
public:
    using ExtractorFn = std::function<ResultField(const void*, int)>;

    /// Register an extractor functor for (entity_type, field_type).
    void register_extractor(EntityType etype, ResultFieldType ftype,
                            ExtractorFn fn) {
        registry_[key(etype, ftype)] = std::move(fn);
    }

    /// Dispatch extraction. Returns an empty ResultField if not registered.
    ResultField extract(EntityType etype, ResultFieldType ftype,
                        const void* data, int count) const {
        auto it = registry_.find(key(etype, ftype));
        if (it == registry_.end()) {
            ResultField empty;
            empty.name = "Unknown";
            empty.type = ftype;
            empty.num_components = 1;
            return empty;
        }
        return it->second(data, count);
    }

    /// Check whether an extractor is registered.
    bool has_extractor(EntityType etype, ResultFieldType ftype) const {
        return registry_.count(key(etype, ftype)) > 0;
    }

    /// Number of registered extractors.
    int num_registered() const { return static_cast<int>(registry_.size()); }

    /// Unregister an extractor.
    void unregister(EntityType etype, ResultFieldType ftype) {
        registry_.erase(key(etype, ftype));
    }

    /// Clear all registrations.
    void clear() { registry_.clear(); }

private:
    using Key = std::pair<int, int>;

    static Key key(EntityType e, ResultFieldType f) {
        return {static_cast<int>(e), static_cast<int>(f)};
    }

    std::map<Key, ExtractorFn> registry_;
};

} // namespace io
} // namespace nxs
