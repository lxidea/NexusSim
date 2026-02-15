#pragma once

/**
 * @file interface_force_output.hpp
 * @brief Contact/interface force resultant time history
 *
 * Records resultant normal and tangential forces for contact interfaces,
 * tied contacts, and rigid wall interfaces. Exports to CSV.
 *
 * Reference: LS-DYNA *DATABASE_RCFORC
 */

#include <nexussim/core/types.hpp>
#include <nexussim/fem/contact.hpp>
#include <nexussim/fem/tied_contact.hpp>
#include <nexussim/fem/rigid_wall.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>

namespace nxs {
namespace io {

// ============================================================================
// Interface Type
// ============================================================================

enum class InterfaceType {
    Contact,     ///< Node-to-surface contact
    Tied,        ///< Tied contact
    RigidWall    ///< Rigid wall contact
};

// ============================================================================
// Interface Force Record
// ============================================================================

struct InterfaceForceRecord {
    Real time;
    int interface_id;
    Real normal_force[3];
    Real tangential_force[3];
    Real total_force_magnitude;
    std::size_t num_active_pairs;

    InterfaceForceRecord()
        : time(0.0), interface_id(0)
        , total_force_magnitude(0.0), num_active_pairs(0) {
        for (int i = 0; i < 3; ++i) {
            normal_force[i] = 0.0;
            tangential_force[i] = 0.0;
        }
    }
};

// ============================================================================
// Interface Force Tracker
// ============================================================================

class InterfaceForceTracker {
public:
    InterfaceForceTracker() = default;

    /**
     * @brief Register a contact interface for tracking
     */
    void register_contact_interface(int id, const std::string& name) {
        InterfaceInfo info;
        info.id = id;
        info.name = name;
        info.type = InterfaceType::Contact;
        interfaces_.push_back(info);
    }

    /**
     * @brief Register a tied contact interface
     */
    void register_tied_interface(int id, const std::string& name) {
        InterfaceInfo info;
        info.id = id;
        info.name = name;
        info.type = InterfaceType::Tied;
        interfaces_.push_back(info);
    }

    /**
     * @brief Register a rigid wall interface
     */
    void register_wall_interface(int id, const std::string& name) {
        InterfaceInfo info;
        info.id = id;
        info.name = name;
        info.type = InterfaceType::RigidWall;
        interfaces_.push_back(info);
    }

    /**
     * @brief Record contact interface forces from contact pairs
     * @param time Current simulation time
     * @param interface_id Interface ID
     * @param pairs Active contact pairs
     * @param penalty_stiffness Penalty stiffness used
     */
    void record_contact_forces(Real time, int interface_id,
                                const std::vector<fem::ContactPair>& pairs,
                                Real penalty_stiffness) {
        InterfaceForceRecord rec;
        rec.time = time;
        rec.interface_id = interface_id;
        rec.num_active_pairs = 0;

        for (const auto& pair : pairs) {
            if (pair.penetration_depth > 0.0) {
                rec.num_active_pairs++;
                Real fn = penalty_stiffness * pair.penetration_depth;
                for (int i = 0; i < 3; ++i) {
                    rec.normal_force[i] += fn * pair.normal[i];
                }
            }
        }

        rec.total_force_magnitude = std::sqrt(
            rec.normal_force[0]*rec.normal_force[0] +
            rec.normal_force[1]*rec.normal_force[1] +
            rec.normal_force[2]*rec.normal_force[2] +
            rec.tangential_force[0]*rec.tangential_force[0] +
            rec.tangential_force[1]*rec.tangential_force[1] +
            rec.tangential_force[2]*rec.tangential_force[2]);

        records_[interface_id].push_back(rec);
        update_times(time);
    }

    /**
     * @brief Record rigid wall forces from WallStats
     * @param time Current simulation time
     * @param interface_id Interface ID
     * @param stats WallStats from rigid wall computation
     * @param wall_normal Wall normal direction
     */
    void record_wall_forces(Real time, int interface_id,
                             const fem::RigidWallContact::WallStats& stats,
                             const Real* wall_normal) {
        InterfaceForceRecord rec;
        rec.time = time;
        rec.interface_id = interface_id;
        rec.num_active_pairs = stats.active_contacts;

        // Wall force is along the wall normal
        Real nmag = std::sqrt(wall_normal[0]*wall_normal[0] +
                               wall_normal[1]*wall_normal[1] +
                               wall_normal[2]*wall_normal[2]);
        if (nmag > 1.0e-30) {
            for (int i = 0; i < 3; ++i) {
                rec.normal_force[i] = stats.total_normal_force * wall_normal[i] / nmag;
            }
        }

        rec.total_force_magnitude = stats.total_normal_force;
        records_[interface_id].push_back(rec);
        update_times(time);
    }

    /**
     * @brief Record a pre-computed force record directly
     */
    void record_direct(Real time, int interface_id,
                        const Real* normal_force,
                        const Real* tangential_force,
                        std::size_t active_pairs) {
        InterfaceForceRecord rec;
        rec.time = time;
        rec.interface_id = interface_id;
        rec.num_active_pairs = active_pairs;

        for (int i = 0; i < 3; ++i) {
            rec.normal_force[i] = normal_force ? normal_force[i] : 0.0;
            rec.tangential_force[i] = tangential_force ? tangential_force[i] : 0.0;
        }

        rec.total_force_magnitude = std::sqrt(
            rec.normal_force[0]*rec.normal_force[0] +
            rec.normal_force[1]*rec.normal_force[1] +
            rec.normal_force[2]*rec.normal_force[2] +
            rec.tangential_force[0]*rec.tangential_force[0] +
            rec.tangential_force[1]*rec.tangential_force[1] +
            rec.tangential_force[2]*rec.tangential_force[2]);

        records_[interface_id].push_back(rec);
        update_times(time);
    }

    /**
     * @brief Write all interface force data to CSV
     */
    bool write_csv(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        file << "time,interface_id,normal_fx,normal_fy,normal_fz,"
             << "tangential_fx,tangential_fy,tangential_fz,"
             << "total_force_magnitude,num_active_pairs\n";

        file << std::scientific << std::setprecision(8);

        for (const auto& [iid, records] : records_) {
            for (const auto& r : records) {
                file << r.time << "," << r.interface_id << ","
                     << r.normal_force[0] << "," << r.normal_force[1] << ","
                     << r.normal_force[2] << ","
                     << r.tangential_force[0] << "," << r.tangential_force[1] << ","
                     << r.tangential_force[2] << ","
                     << r.total_force_magnitude << ","
                     << r.num_active_pairs << "\n";
            }
        }

        return file.good();
    }

    const std::vector<InterfaceForceRecord>* get_records(int interface_id) const {
        auto it = records_.find(interface_id);
        if (it == records_.end()) return nullptr;
        return &it->second;
    }

    std::size_t num_interfaces() const { return interfaces_.size(); }
    std::size_t num_records() const { return times_.size(); }

private:
    void update_times(Real time) {
        if (times_.empty() || times_.back() < time) {
            times_.push_back(time);
        }
    }

    struct InterfaceInfo {
        int id;
        std::string name;
        InterfaceType type;
    };

    std::vector<InterfaceInfo> interfaces_;
    std::vector<Real> times_;
    std::map<int, std::vector<InterfaceForceRecord>> records_;
};

} // namespace io
} // namespace nxs
