#pragma once

/**
 * @file section.hpp
 * @brief Section property definitions for beams and shells
 *
 * Supports:
 * - Shell sections: uniform thickness, variable thickness, multi-ply composite
 * - Beam sections: rectangular, circular, I-beam, hollow tube, arbitrary
 * - Through-thickness integration points for shells
 * - Section property computation (area, moments of inertia, shear factors)
 *
 * Reference: LS-DYNA *SECTION_SHELL, *SECTION_BEAM
 */

#include <nexussim/core/types.hpp>
#include <nexussim/core/gpu.hpp>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>

namespace nxs {
namespace physics {

// ============================================================================
// Section Types
// ============================================================================

enum class SectionType {
    // Shell sections
    ShellUniform,       ///< Uniform thickness shell
    ShellVariable,      ///< Variable thickness (per-node)
    ShellComposite,     ///< Multi-ply composite layup

    // Beam sections
    BeamRectangular,    ///< Solid rectangular
    BeamCircular,       ///< Solid circular
    BeamHollowCircular, ///< Hollow circular (tube)
    BeamIBeam,          ///< I-beam (wide flange)
    BeamBoxSection,     ///< Hollow rectangular (box)
    BeamArbitrary       ///< User-defined properties
};

// ============================================================================
// Integration Point Layout
// ============================================================================

/**
 * @brief Through-thickness integration point for shells
 */
struct IntegrationPoint {
    Real z;           ///< Through-thickness coordinate (z/h, range [-0.5, 0.5])
    Real weight;      ///< Integration weight
    int ply_id;       ///< Ply index (for composites, -1 otherwise)
};

/**
 * @brief Standard Gauss quadrature through thickness
 */
struct ThicknessIntegration {
    static constexpr int MAX_POINTS = 16;
    IntegrationPoint points[MAX_POINTS];
    int num_points;

    ThicknessIntegration() : num_points(0) {
        for (int i = 0; i < MAX_POINTS; ++i) {
            points[i] = {0.0, 0.0, -1};
        }
    }

    /**
     * @brief Set up Gauss integration through thickness
     * @param npts Number of integration points (1-5)
     */
    void setup_gauss(int npts) {
        num_points = npts;
        switch (npts) {
            case 1:
                points[0] = {0.0, 1.0, -1};
                break;
            case 2:
                points[0] = {-1.0/std::sqrt(3.0) * 0.5, 0.5, -1};
                points[1] = { 1.0/std::sqrt(3.0) * 0.5, 0.5, -1};
                break;
            case 3:
                points[0] = {-std::sqrt(3.0/5.0) * 0.5, 5.0/18.0, -1};
                points[1] = {0.0, 8.0/18.0, -1};
                points[2] = { std::sqrt(3.0/5.0) * 0.5, 5.0/18.0, -1};
                break;
            case 5: {
                Real a = std::sqrt(5.0 + 2.0*std::sqrt(10.0/7.0)) / 3.0;
                Real b = std::sqrt(5.0 - 2.0*std::sqrt(10.0/7.0)) / 3.0;
                Real wa = (322.0 - 13.0*std::sqrt(70.0)) / 900.0;
                Real wb = (322.0 + 13.0*std::sqrt(70.0)) / 900.0;
                Real wc = 128.0 / 225.0;
                points[0] = {-a * 0.5, wa * 0.5, -1};
                points[1] = {-b * 0.5, wb * 0.5, -1};
                points[2] = { 0.0,     wc * 0.5, -1};
                points[3] = { b * 0.5, wb * 0.5, -1};
                points[4] = { a * 0.5, wa * 0.5, -1};
                break;
            }
            default:  // Simpson-like uniform spacing
                for (int i = 0; i < npts; ++i) {
                    Real zeta = -0.5 + static_cast<Real>(i) / (npts - 1);
                    points[i] = {zeta, 1.0 / npts, -1};
                }
                break;
        }
    }

    /**
     * @brief Set up uniform integration points for composite plies
     * @param num_plies Number of plies
     * @param points_per_ply Integration points per ply (typically 1 or 2)
     */
    void setup_composite(int num_plies, int points_per_ply = 1) {
        num_points = 0;
        Real ply_thickness = 1.0 / num_plies;  // Normalized

        for (int p = 0; p < num_plies && num_points < MAX_POINTS; ++p) {
            Real z_bot = -0.5 + p * ply_thickness;

            if (points_per_ply == 1) {
                points[num_points] = {z_bot + 0.5 * ply_thickness,
                                       ply_thickness, p};
                num_points++;
            } else {
                // 2-point Gauss within each ply
                Real d = ply_thickness / (2.0 * std::sqrt(3.0));
                Real mid = z_bot + 0.5 * ply_thickness;
                points[num_points] = {mid - d, ply_thickness / 2.0, p};
                num_points++;
                if (num_points < MAX_POINTS) {
                    points[num_points] = {mid + d, ply_thickness / 2.0, p};
                    num_points++;
                }
            }
        }
    }
};

// ============================================================================
// Section Properties
// ============================================================================

struct SectionProperties {
    SectionType type;
    int id;
    std::string name;

    // Shell properties
    Real thickness;           ///< Shell thickness (uniform)
    Real thickness_nodes[4];  ///< Variable thickness at element corners
    int num_ip_thickness;     ///< Number of through-thickness integration points

    // Beam geometric properties
    Real width;               ///< Beam width (rectangular/box)
    Real height;              ///< Beam height (rectangular/box/I-beam)
    Real diameter;            ///< Beam diameter (circular/hollow)
    Real wall_thickness;      ///< Wall thickness (hollow sections)
    Real flange_width;        ///< I-beam flange width
    Real flange_thickness;    ///< I-beam flange thickness
    Real web_thickness;       ///< I-beam web thickness

    // Computed beam section properties
    Real area;                ///< Cross-sectional area
    Real Iyy;                 ///< Second moment about y-axis
    Real Izz;                 ///< Second moment about z-axis
    Real J;                   ///< Torsional constant
    Real ky;                  ///< Shear correction factor y
    Real kz;                  ///< Shear correction factor z

    // Through-thickness integration
    ThicknessIntegration integration;

    SectionProperties()
        : type(SectionType::ShellUniform), id(0)
        , thickness(1.0), num_ip_thickness(2)
        , width(0.0), height(0.0), diameter(0.0), wall_thickness(0.0)
        , flange_width(0.0), flange_thickness(0.0), web_thickness(0.0)
        , area(0.0), Iyy(0.0), Izz(0.0), J(0.0), ky(5.0/6.0), kz(5.0/6.0) {
        for (int i = 0; i < 4; ++i) thickness_nodes[i] = 1.0;
    }

    /**
     * @brief Compute section properties from geometric parameters
     */
    void compute() {
        switch (type) {
            case SectionType::ShellUniform:
            case SectionType::ShellVariable:
                integration.setup_gauss(num_ip_thickness);
                break;

            case SectionType::ShellComposite:
                // Composite integration set up by layup system
                break;

            case SectionType::BeamRectangular:
                area = width * height;
                Iyy = width * height * height * height / 12.0;
                Izz = height * width * width * width / 12.0;
                J = compute_rectangular_torsion(width, height);
                ky = kz = 5.0/6.0;
                break;

            case SectionType::BeamCircular: {
                Real r = diameter / 2.0;
                area = constants::pi<Real> * r * r;
                Iyy = Izz = constants::pi<Real> * r * r * r * r / 4.0;
                J = constants::pi<Real> * r * r * r * r / 2.0;
                ky = kz = 6.0/7.0;
                break;
            }

            case SectionType::BeamHollowCircular: {
                Real ro = diameter / 2.0;
                Real ri = ro - wall_thickness;
                if (ri < 0) ri = 0;
                area = constants::pi<Real> * (ro*ro - ri*ri);
                Iyy = Izz = constants::pi<Real> * (ro*ro*ro*ro - ri*ri*ri*ri) / 4.0;
                J = constants::pi<Real> * (ro*ro*ro*ro - ri*ri*ri*ri) / 2.0;
                ky = kz = 0.5;  // Thin-walled tube approximation
                break;
            }

            case SectionType::BeamIBeam: {
                Real hw = height - 2.0 * flange_thickness;
                area = 2.0 * flange_width * flange_thickness + hw * web_thickness;
                Iyy = (flange_width * height * height * height
                        - (flange_width - web_thickness) * hw * hw * hw) / 12.0;
                Izz = (2.0 * flange_thickness * flange_width * flange_width * flange_width
                        + hw * web_thickness * web_thickness * web_thickness) / 12.0;
                J = (2.0 * flange_width * flange_thickness * flange_thickness * flange_thickness
                     + hw * web_thickness * web_thickness * web_thickness) / 3.0;
                ky = area / (hw * web_thickness) * 5.0/6.0;
                kz = 5.0/6.0;
                break;
            }

            case SectionType::BeamBoxSection: {
                Real wi = width - 2.0 * wall_thickness;
                Real hi = height - 2.0 * wall_thickness;
                if (wi < 0) wi = 0;
                if (hi < 0) hi = 0;
                area = width * height - wi * hi;
                Iyy = (width * height * height * height - wi * hi * hi * hi) / 12.0;
                Izz = (height * width * width * width - hi * wi * wi * wi) / 12.0;
                // Bredt's formula for thin-walled torsion
                Real Am = (width - wall_thickness) * (height - wall_thickness);
                Real perimeter = 2.0 * ((width - wall_thickness) + (height - wall_thickness));
                J = (wall_thickness > 0) ? 4.0 * Am * Am * wall_thickness / perimeter : 0.0;
                ky = kz = 0.5;
                break;
            }

            case SectionType::BeamArbitrary:
                // User provides A, Iyy, Izz, J directly
                break;
        }
    }

    /**
     * @brief Get shell thickness at a parametric point
     * @param xi, eta Parametric coordinates [-1,1]
     */
    KOKKOS_INLINE_FUNCTION
    Real interpolate_thickness(Real xi, Real eta) const {
        if (type != SectionType::ShellVariable) return thickness;
        Real N[4] = {
            0.25*(1.0-xi)*(1.0-eta),
            0.25*(1.0+xi)*(1.0-eta),
            0.25*(1.0+xi)*(1.0+eta),
            0.25*(1.0-xi)*(1.0+eta)
        };
        return N[0]*thickness_nodes[0] + N[1]*thickness_nodes[1]
             + N[2]*thickness_nodes[2] + N[3]*thickness_nodes[3];
    }

    void print_summary() const {
        std::cout << "Section [" << id << "] " << name << "\n";
        if (type <= SectionType::ShellComposite) {
            std::cout << "  Type: Shell, thickness=" << thickness
                      << ", integration points=" << integration.num_points << "\n";
        } else {
            std::cout << "  Type: Beam, A=" << area << " Iyy=" << Iyy
                      << " Izz=" << Izz << " J=" << J << "\n";
        }
    }

private:
    /**
     * @brief Approximate torsional constant for solid rectangular section
     * Using series solution: J ≈ a*b³ * (1/3 - 0.21*(b/a)*(1 - b⁴/(12*a⁴)))
     * where a ≥ b
     */
    static Real compute_rectangular_torsion(Real w, Real h) {
        Real a = std::max(w, h);
        Real b = std::min(w, h);
        Real ratio = b / a;
        return a * b * b * b * (1.0/3.0 - 0.21 * ratio *
               (1.0 - ratio * ratio * ratio * ratio / 12.0));
    }
};

// ============================================================================
// Section Manager
// ============================================================================

class SectionManager {
public:
    SectionManager() = default;

    SectionProperties& add_section(int id) {
        sections_.emplace_back();
        sections_.back().id = id;
        return sections_.back();
    }

    const SectionProperties* find(int id) const {
        for (const auto& s : sections_) {
            if (s.id == id) return &s;
        }
        return nullptr;
    }

    SectionProperties* find(int id) {
        for (auto& s : sections_) {
            if (s.id == id) return &s;
        }
        return nullptr;
    }

    std::size_t num_sections() const { return sections_.size(); }

    void compute_all() {
        for (auto& s : sections_) s.compute();
    }

    void print_summary() const {
        std::cout << "Sections: " << sections_.size() << "\n";
        for (const auto& s : sections_) s.print_summary();
    }

private:
    std::vector<SectionProperties> sections_;
};

} // namespace physics
} // namespace nxs
