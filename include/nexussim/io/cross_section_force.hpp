#pragma once

/**
 * @file cross_section_force.hpp
 * @brief Forces and moments on arbitrary cut planes through the mesh
 *
 * Computes resultant force and moment from element stresses on a cross-section
 * defined by a plane (origin + normal). Uses traction integration:
 *   F = Σ (σ · n) * A_elem
 *   M = Σ (r × (σ · n)) * A_elem
 *
 * Reference: LS-DYNA *DATABASE_CROSS_SECTION
 */

#include <nexussim/core/types.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>

namespace nxs {
namespace io {

// ============================================================================
// Cross-Section Definition
// ============================================================================

struct CrossSectionDefinition {
    int id;
    std::string name;
    Real origin[3];           ///< Point on the cut plane
    Real normal[3];           ///< Normal to the cut plane
    std::vector<Index> element_set; ///< Elements intersected by the plane

    CrossSectionDefinition()
        : id(0) {
        origin[0] = origin[1] = origin[2] = 0.0;
        normal[0] = 1.0; normal[1] = 0.0; normal[2] = 0.0;
    }
};

// ============================================================================
// Cross-Section Force Record
// ============================================================================

struct CrossSectionForceRecord {
    Real time;
    int section_id;
    Real force[3];
    Real moment[3];
    Real force_magnitude;

    CrossSectionForceRecord()
        : time(0.0), section_id(0), force_magnitude(0.0) {
        force[0] = force[1] = force[2] = 0.0;
        moment[0] = moment[1] = moment[2] = 0.0;
    }
};

// ============================================================================
// Cross-Section Force Tracker
// ============================================================================

class CrossSectionForceTracker {
public:
    CrossSectionForceTracker() = default;

    /**
     * @brief Define a cross-section cut plane
     * @param id Section ID
     * @param name Section name
     * @param origin Point on the cut plane
     * @param normal Normal to the cut plane
     * @param element_set Elements contributing to the cross-section
     */
    void add_section(int id, const std::string& name,
                      const Real* origin, const Real* normal,
                      const std::vector<Index>& element_set) {
        CrossSectionDefinition sec;
        sec.id = id;
        sec.name = name;
        for (int i = 0; i < 3; ++i) {
            sec.origin[i] = origin[i];
            sec.normal[i] = normal[i];
        }
        sec.element_set = element_set;
        sections_.push_back(sec);
    }

    /**
     * @brief Compute and record cross-section forces at current time
     *
     * For each element in the cross-section set:
     *   traction = σ · n  (stress tensor times plane normal)
     *   F += traction * A_elem
     *   M += (r_elem - origin) × traction * A_elem
     *
     * @param time Current simulation time
     * @param stresses Per-element stress tensors [6*num_elements] (Voigt: σxx,σyy,σzz,τxy,τyz,τxz)
     * @param element_areas Per-element areas/volumes
     * @param element_centroids Per-element centroids [3*num_elements]
     */
    void record(Real time,
                const Real* stresses,
                const Real* element_areas,
                const Real* element_centroids) {

        times_.push_back(time);

        for (const auto& sec : sections_) {
            CrossSectionForceRecord rec;
            rec.time = time;
            rec.section_id = sec.id;

            // Normalize section normal
            Real nx = sec.normal[0], ny = sec.normal[1], nz = sec.normal[2];
            Real nmag = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (nmag > 1.0e-30) {
                nx /= nmag; ny /= nmag; nz /= nmag;
            }

            for (Index eid : sec.element_set) {
                const Real* s = stresses + 6 * eid;
                Real area = element_areas[eid];

                // Traction: t = σ · n
                // σ is symmetric: [σxx, σyy, σzz, τxy, τyz, τxz]
                // t_x = σxx*nx + τxy*ny + τxz*nz
                // t_y = τxy*nx + σyy*ny + τyz*nz
                // t_z = τxz*nx + τyz*ny + σzz*nz
                Real tx = s[0]*nx + s[3]*ny + s[5]*nz;
                Real ty = s[3]*nx + s[1]*ny + s[4]*nz;
                Real tz = s[5]*nx + s[4]*ny + s[2]*nz;

                // Force contribution: F += t * A
                rec.force[0] += tx * area;
                rec.force[1] += ty * area;
                rec.force[2] += tz * area;

                // Moment contribution: M += (r - origin) × (t * A)
                if (element_centroids) {
                    Real rx = element_centroids[3*eid+0] - sec.origin[0];
                    Real ry = element_centroids[3*eid+1] - sec.origin[1];
                    Real rz = element_centroids[3*eid+2] - sec.origin[2];

                    Real fx = tx * area;
                    Real fy = ty * area;
                    Real fz = tz * area;

                    rec.moment[0] += ry*fz - rz*fy;
                    rec.moment[1] += rz*fx - rx*fz;
                    rec.moment[2] += rx*fy - ry*fx;
                }
            }

            rec.force_magnitude = std::sqrt(
                rec.force[0]*rec.force[0] +
                rec.force[1]*rec.force[1] +
                rec.force[2]*rec.force[2]);

            records_[sec.id].push_back(rec);
        }
    }

    /**
     * @brief Write cross-section force data to CSV
     */
    bool write_csv(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        file << "time,section_id,force_x,force_y,force_z,"
             << "moment_x,moment_y,moment_z,force_magnitude\n";

        file << std::scientific << std::setprecision(8);

        for (const auto& [sid, records] : records_) {
            for (const auto& r : records) {
                file << r.time << "," << r.section_id << ","
                     << r.force[0] << "," << r.force[1] << "," << r.force[2] << ","
                     << r.moment[0] << "," << r.moment[1] << "," << r.moment[2] << ","
                     << r.force_magnitude << "\n";
            }
        }

        return file.good();
    }

    const std::vector<CrossSectionForceRecord>* get_records(int section_id) const {
        auto it = records_.find(section_id);
        if (it == records_.end()) return nullptr;
        return &it->second;
    }

    std::size_t num_sections() const { return sections_.size(); }
    std::size_t num_records() const { return times_.size(); }

private:
    std::vector<CrossSectionDefinition> sections_;
    std::vector<Real> times_;
    std::map<int, std::vector<CrossSectionForceRecord>> records_;
};

} // namespace io
} // namespace nxs
