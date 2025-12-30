#pragma once

/**
 * @file pd_types.hpp
 * @brief Core type definitions for Peridynamics
 *
 * Ported from PeriSys-Haoran (ZHR) with Kokkos support
 * Original CUDA implementation by Haoran Zhang
 */

#include <nexussim/core/core.hpp>
#include <Kokkos_Core.hpp>

namespace nxs {
namespace pd {

// ============================================================================
// Material Types (from PeriSys Global_Para.cuh)
// ============================================================================

/**
 * @brief Peridynamics material types
 */
enum class PDMaterialType {
    Elastic = 1,            ///< Linear elastic
    DruckerPrager = 2,      ///< Geomaterials (soil, rock)
    JohnsonHolmquist2 = 4,  ///< Ceramics, glass (JH-2)
    Rigid = 5,              ///< Rigid body
    JohnsonCook = 7,        ///< Metals with strain rate effects
    JohnsonCookPD = 8,      ///< Johnson-Cook adapted for PD
    BondBasedPD = 9,        ///< Standard bond-based PD
    ElasticBondPD = 10,     ///< Elastic bond-based PD
    PMMABondPD = 11         ///< PMMA polymer bond-based PD
};

/**
 * @brief Convert material type to string
 */
inline const char* to_string(PDMaterialType type) {
    switch (type) {
        case PDMaterialType::Elastic:           return "Elastic";
        case PDMaterialType::DruckerPrager:     return "Drucker-Prager";
        case PDMaterialType::JohnsonHolmquist2: return "Johnson-Holmquist 2";
        case PDMaterialType::Rigid:             return "Rigid";
        case PDMaterialType::JohnsonCook:       return "Johnson-Cook";
        case PDMaterialType::JohnsonCookPD:     return "Johnson-Cook-PD";
        case PDMaterialType::BondBasedPD:       return "Bond-Based PD";
        case PDMaterialType::ElasticBondPD:     return "Elastic Bond PD";
        case PDMaterialType::PMMABondPD:        return "PMMA Bond PD";
        default:                                return "Unknown";
    }
}

// ============================================================================
// Peridynamics Material Properties
// ============================================================================

/**
 * @brief Peridynamics material properties
 */
struct PDMaterial {
    PDMaterialType type = PDMaterialType::Elastic;

    // Basic properties
    Real E = 2.0e11;        ///< Young's modulus (Pa)
    Real nu = 0.25;         ///< Poisson's ratio (bond-based: nu = 0.25)
    Real rho = 7800.0;      ///< Density (kg/m³)
    Real cpi = 460.0;       ///< Specific heat (J/kg·K)

    // Derived properties (computed from E, nu)
    Real G = 0.0;           ///< Shear modulus
    Real K = 0.0;           ///< Bulk modulus
    Real c = 0.0;           ///< Bond stiffness (micromodulus)

    // Failure properties
    Real s_critical = 0.01; ///< Critical stretch for bond failure
    Real Gc = 100.0;        ///< Fracture energy (J/m²)

    // Johnson-Cook parameters (for JC materials)
    Real JC_A = 0.0;        ///< Yield stress at reference
    Real JC_B = 0.0;        ///< Strain hardening coefficient
    Real JC_n = 0.0;        ///< Strain hardening exponent
    Real JC_C = 0.0;        ///< Strain rate coefficient
    Real JC_m = 0.0;        ///< Thermal softening exponent
    Real JC_eps_dot_ref = 1.0;  ///< Reference strain rate
    Real JC_T_melt = 1800.0;    ///< Melting temperature (K)
    Real JC_T_room = 300.0;     ///< Room temperature (K)

    // Drucker-Prager parameters (for geomaterials)
    Real DP_phi = 30.0;     ///< Friction angle (degrees)
    Real DP_c = 1.0e6;      ///< Cohesion (Pa)
    Real DP_psi = 0.0;      ///< Dilation angle (degrees)

    /**
     * @brief Compute derived properties from E and nu
     */
    KOKKOS_INLINE_FUNCTION
    void compute_derived(Real horizon, Real thickness = 1.0) {
        G = E / (2.0 * (1.0 + nu));
        K = E / (3.0 * (1.0 - 2.0 * nu));

        // Bond micromodulus for 3D (from PD theory)
        // c = 18K / (pi * delta^4) for 3D
        // c = 12E / (pi * h * delta^3) for 2D plane stress
        Real pi = 3.14159265358979323846;
        Real delta4 = horizon * horizon * horizon * horizon;
        c = 18.0 * K / (pi * delta4);
    }

    /**
     * @brief Compute critical stretch from fracture energy
     */
    KOKKOS_INLINE_FUNCTION
    void compute_critical_stretch(Real horizon) {
        // s_c = sqrt(5 * Gc / (9 * K * delta))  for 3D
        Real delta = horizon;
        s_critical = std::sqrt(5.0 * Gc / (9.0 * K * delta));
    }
};

// ============================================================================
// Particle State
// ============================================================================

/**
 * @brief State of a single PD particle
 */
struct PDParticleState {
    // Position and kinematics
    Real x[3] = {0, 0, 0};      ///< Current position
    Real x0[3] = {0, 0, 0};     ///< Reference position
    Real u[3] = {0, 0, 0};      ///< Displacement
    Real v[3] = {0, 0, 0};      ///< Velocity
    Real a[3] = {0, 0, 0};      ///< Acceleration

    // Forces
    Real f[3] = {0, 0, 0};      ///< Total force density
    Real f_ext[3] = {0, 0, 0};  ///< External force density

    // Properties
    Real volume = 0.0;          ///< Particle volume
    Real horizon = 0.0;         ///< Horizon (neighborhood radius)
    Real mass = 0.0;            ///< Particle mass

    // State variables
    Real damage = 0.0;          ///< Damage (0 = intact, 1 = failed)
    Real theta = 0.0;           ///< Dilatation (for state-based PD)
    Real temperature = 300.0;   ///< Temperature (K)
    Real plastic_strain = 0.0;  ///< Effective plastic strain

    // Material ID
    Index material_id = 0;
    Index body_id = 0;

    // Status
    bool active = true;         ///< Particle active/deleted

    /**
     * @brief Update position from displacement
     */
    KOKKOS_INLINE_FUNCTION
    void update_position() {
        x[0] = x0[0] + u[0];
        x[1] = x0[1] + u[1];
        x[2] = x0[2] + u[2];
    }

    /**
     * @brief Compute displacement from positions
     */
    KOKKOS_INLINE_FUNCTION
    void compute_displacement() {
        u[0] = x[0] - x0[0];
        u[1] = x[1] - x0[1];
        u[2] = x[2] - x0[2];
    }
};

// ============================================================================
// Bond State
// ============================================================================

/**
 * @brief State of a single PD bond
 */
struct PDBondState {
    Index i = 0;                ///< Particle i index
    Index j = 0;                ///< Particle j (neighbor) index
    Real xi[3] = {0, 0, 0};     ///< Reference bond vector (xj - xi)
    Real xi_length = 0.0;       ///< Reference bond length |xi|
    Real weight = 1.0;          ///< Influence function weight
    bool intact = true;         ///< Bond intact (not broken)

    /**
     * @brief Compute current bond stretch
     * @param eta Current relative displacement (uj - ui)
     * @return Bond stretch s = (|xi + eta| - |xi|) / |xi|
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_stretch(const Real* eta) const {
        Real xi_eta[3] = {
            xi[0] + eta[0],
            xi[1] + eta[1],
            xi[2] + eta[2]
        };
        Real xi_eta_length = std::sqrt(
            xi_eta[0] * xi_eta[0] +
            xi_eta[1] * xi_eta[1] +
            xi_eta[2] * xi_eta[2]
        );
        return (xi_eta_length - xi_length) / xi_length;
    }

    /**
     * @brief Compute unit vector in current bond direction
     */
    KOKKOS_INLINE_FUNCTION
    void compute_direction(const Real* eta, Real* e) const {
        Real xi_eta[3] = {
            xi[0] + eta[0],
            xi[1] + eta[1],
            xi[2] + eta[2]
        };
        Real length = std::sqrt(
            xi_eta[0] * xi_eta[0] +
            xi_eta[1] * xi_eta[1] +
            xi_eta[2] * xi_eta[2]
        );
        Real inv_length = (length > 1e-20) ? 1.0 / length : 0.0;
        e[0] = xi_eta[0] * inv_length;
        e[1] = xi_eta[1] * inv_length;
        e[2] = xi_eta[2] * inv_length;
    }
};

// ============================================================================
// Boundary Condition Types
// ============================================================================

enum class PDBCType {
    None = 0,
    FixedX = 1,         ///< Fixed in X direction
    FixedY = 2,         ///< Fixed in Y direction
    FixedZ = 4,         ///< Fixed in Z direction
    FixedAll = 7,       ///< Fixed in all directions
    Velocity = 8,       ///< Prescribed velocity
    Force = 16          ///< Applied force
};

// ============================================================================
// Kokkos View Types for PD Data
// ============================================================================

// Device views (GPU-compatible)
using PDPositionView = Kokkos::View<Real*[3]>;
using PDVelocityView = Kokkos::View<Real*[3]>;
using PDForceView = Kokkos::View<Real*[3]>;
using PDScalarView = Kokkos::View<Real*>;
using PDIndexView = Kokkos::View<Index*>;
using PDBoolView = Kokkos::View<bool*>;

// Host mirrors
using PDPositionHostView = PDPositionView::HostMirror;
using PDVelocityHostView = PDVelocityView::HostMirror;
using PDForceHostView = PDForceView::HostMirror;
using PDScalarHostView = PDScalarView::HostMirror;
using PDIndexHostView = PDIndexView::HostMirror;
using PDBoolHostView = PDBoolView::HostMirror;

// Neighbor list views (CSR format)
using PDNeighborOffsetView = Kokkos::View<Index*>;  // Row pointers
using PDNeighborListView = Kokkos::View<Index*>;    // Column indices
using PDBondWeightView = Kokkos::View<Real*>;       // Bond weights
using PDBondIntactView = Kokkos::View<bool*>;       // Bond status

} // namespace pd
} // namespace nxs
