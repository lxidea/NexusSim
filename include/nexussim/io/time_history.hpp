#pragma once

/**
 * @file time_history.hpp
 * @brief Time history recording for nodal and element quantities
 *
 * Records selected quantities at specified nodes/elements over time.
 * Supports:
 * - Nodal: displacement, velocity, acceleration, force (per DOF)
 * - Element: stress, strain, plastic strain, damage, von Mises
 * - Output to CSV for post-processing with Python/MATLAB
 * - In-memory buffering with configurable flush interval
 *
 * Usage:
 *   TimeHistoryRecorder recorder;
 *   recorder.add_nodal_probe("tip_disp", {10, 20}, NodalQuantity::Displacement, 2);
 *   recorder.add_element_probe("stress_xx", {5}, ElementQuantity::StressXX);
 *   // In time loop:
 *   recorder.record(time, positions, velocities, accelerations, forces,
 *                   stresses, strains, plastic_strains);
 *   // After simulation:
 *   recorder.write_csv("output/history");
 *
 * Reference: LS-DYNA *DATABASE_HISTORY_NODE, *DATABASE_HISTORY_ELEMENT
 */

#include <nexussim/core/types.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace nxs {
namespace io {

// ============================================================================
// Quantity Types
// ============================================================================

enum class NodalQuantity {
    DisplacementX, DisplacementY, DisplacementZ, DisplacementMag,
    VelocityX, VelocityY, VelocityZ, VelocityMag,
    AccelerationX, AccelerationY, AccelerationZ, AccelerationMag,
    ForceX, ForceY, ForceZ, ForceMag
};

enum class ElementQuantity {
    StressXX, StressYY, StressZZ, StressXY, StressYZ, StressXZ,
    VonMises, Pressure,
    StrainXX, StrainYY, StrainZZ, StrainXY, StrainYZ, StrainXZ,
    EffectivePlasticStrain, Damage,
    InternalEnergy, VolumetricStrain
};

inline const char* to_string(NodalQuantity q) {
    switch (q) {
        case NodalQuantity::DisplacementX: return "disp_x";
        case NodalQuantity::DisplacementY: return "disp_y";
        case NodalQuantity::DisplacementZ: return "disp_z";
        case NodalQuantity::DisplacementMag: return "disp_mag";
        case NodalQuantity::VelocityX: return "vel_x";
        case NodalQuantity::VelocityY: return "vel_y";
        case NodalQuantity::VelocityZ: return "vel_z";
        case NodalQuantity::VelocityMag: return "vel_mag";
        case NodalQuantity::AccelerationX: return "accel_x";
        case NodalQuantity::AccelerationY: return "accel_y";
        case NodalQuantity::AccelerationZ: return "accel_z";
        case NodalQuantity::AccelerationMag: return "accel_mag";
        case NodalQuantity::ForceX: return "force_x";
        case NodalQuantity::ForceY: return "force_y";
        case NodalQuantity::ForceZ: return "force_z";
        case NodalQuantity::ForceMag: return "force_mag";
        default: return "unknown";
    }
}

inline const char* to_string(ElementQuantity q) {
    switch (q) {
        case ElementQuantity::StressXX: return "stress_xx";
        case ElementQuantity::StressYY: return "stress_yy";
        case ElementQuantity::StressZZ: return "stress_zz";
        case ElementQuantity::StressXY: return "stress_xy";
        case ElementQuantity::StressYZ: return "stress_yz";
        case ElementQuantity::StressXZ: return "stress_xz";
        case ElementQuantity::VonMises: return "von_mises";
        case ElementQuantity::Pressure: return "pressure";
        case ElementQuantity::StrainXX: return "strain_xx";
        case ElementQuantity::StrainYY: return "strain_yy";
        case ElementQuantity::StrainZZ: return "strain_zz";
        case ElementQuantity::StrainXY: return "strain_xy";
        case ElementQuantity::StrainYZ: return "strain_yz";
        case ElementQuantity::StrainXZ: return "strain_xz";
        case ElementQuantity::EffectivePlasticStrain: return "eff_plastic_strain";
        case ElementQuantity::Damage: return "damage";
        case ElementQuantity::InternalEnergy: return "internal_energy";
        case ElementQuantity::VolumetricStrain: return "vol_strain";
        default: return "unknown";
    }
}

// ============================================================================
// Probe Definitions
// ============================================================================

struct NodalProbe {
    std::string name;           ///< User label
    std::vector<Index> nodes;   ///< Node IDs to track
    NodalQuantity quantity;     ///< What to record
};

struct ElementProbe {
    std::string name;               ///< User label
    std::vector<Index> elements;    ///< Element IDs to track
    ElementQuantity quantity;       ///< What to record
};

// ============================================================================
// Time History Recorder
// ============================================================================

class TimeHistoryRecorder {
public:
    TimeHistoryRecorder() = default;

    // --- Probe Setup ---

    /**
     * @brief Add a nodal probe
     * @param name User label for this probe
     * @param node_ids Node IDs to track
     * @param quantity What nodal quantity to record
     */
    void add_nodal_probe(const std::string& name,
                          const std::vector<Index>& node_ids,
                          NodalQuantity quantity) {
        nodal_probes_.push_back({name, node_ids, quantity});
    }

    /**
     * @brief Add an element probe
     * @param name User label
     * @param element_ids Element IDs to track
     * @param quantity What element quantity to record
     */
    void add_element_probe(const std::string& name,
                            const std::vector<Index>& element_ids,
                            ElementQuantity quantity) {
        element_probes_.push_back({name, element_ids, quantity});
    }

    /**
     * @brief Set recording interval (record every N-th call)
     */
    void set_interval(int interval) { record_interval_ = interval; }

    // --- Recording ---

    /**
     * @brief Record current state for all probes
     *
     * Arrays are assumed in flat [node*3+dof] layout (standard FEM convention).
     * Stress/strain arrays are in [elem*6+component] layout (Voigt notation).
     *
     * @param time Current simulation time
     * @param num_nodes Number of nodes
     * @param displacements Nodal displacements (3*num_nodes), nullptr if not available
     * @param velocities Nodal velocities (3*num_nodes), nullptr if not available
     * @param accelerations Nodal accelerations (3*num_nodes), nullptr if not available
     * @param forces Nodal forces (3*num_nodes), nullptr if not available
     * @param num_elements Number of elements
     * @param stresses Element stresses (6*num_elements), nullptr if not available
     * @param strains Element strains (6*num_elements), nullptr if not available
     * @param plastic_strains Effective plastic strain per element, nullptr if not available
     * @param damages Damage per element, nullptr if not available
     */
    void record(Real time,
                std::size_t num_nodes,
                const Real* displacements = nullptr,
                const Real* velocities = nullptr,
                const Real* accelerations = nullptr,
                const Real* forces = nullptr,
                std::size_t num_elements = 0,
                const Real* stresses = nullptr,
                const Real* strains = nullptr,
                const Real* plastic_strains = nullptr,
                const Real* damages = nullptr) {

        call_count_++;
        if (record_interval_ > 1 && (call_count_ % record_interval_) != 1) return;

        // Record time
        times_.push_back(time);

        // Record nodal probes
        for (auto& probe : nodal_probes_) {
            std::vector<Real> values;
            for (Index nid : probe.nodes) {
                if (nid >= num_nodes) { values.push_back(0.0); continue; }
                values.push_back(extract_nodal(probe.quantity, nid,
                                               displacements, velocities,
                                               accelerations, forces));
            }
            probe_data_[probe.name].push_back(values);
        }

        // Record element probes
        for (auto& probe : element_probes_) {
            std::vector<Real> values;
            for (Index eid : probe.elements) {
                if (eid >= num_elements) { values.push_back(0.0); continue; }
                values.push_back(extract_element(probe.quantity, eid,
                                                  stresses, strains,
                                                  plastic_strains, damages));
            }
            probe_data_[probe.name].push_back(values);
        }
    }

    // --- Output ---

    /**
     * @brief Write all recorded data to CSV files
     * @param base_path Base path (each probe gets its own file)
     *
     * Files created: base_path_<probe_name>.csv
     * Format: time, node1_value, node2_value, ...
     */
    bool write_csv(const std::string& base_path) const {
        bool all_ok = true;

        for (const auto& [name, records] : probe_data_) {
            std::string filename = base_path + "_" + name + ".csv";
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "TimeHistory: Cannot open " << filename << "\n";
                all_ok = false;
                continue;
            }

            // Header
            file << "time";
            if (!records.empty()) {
                for (std::size_t j = 0; j < records[0].size(); ++j) {
                    file << "," << name << "_" << j;
                }
            }
            file << "\n";

            // Data rows
            for (std::size_t i = 0; i < records.size() && i < times_.size(); ++i) {
                file << std::scientific << std::setprecision(8) << times_[i];
                for (Real v : records[i]) {
                    file << "," << v;
                }
                file << "\n";
            }
        }

        return all_ok;
    }

    /**
     * @brief Get recorded values for a probe
     * @return Vector of time-step records, each record has one value per entity
     */
    const std::vector<std::vector<Real>>* get_probe_data(const std::string& name) const {
        auto it = probe_data_.find(name);
        if (it == probe_data_.end()) return nullptr;
        return &it->second;
    }

    /**
     * @brief Get recorded time values
     */
    const std::vector<Real>& times() const { return times_; }

    // --- Statistics ---

    std::size_t num_records() const { return times_.size(); }
    std::size_t num_nodal_probes() const { return nodal_probes_.size(); }
    std::size_t num_element_probes() const { return element_probes_.size(); }
    std::size_t num_probes() const { return nodal_probes_.size() + element_probes_.size(); }

    void print_summary() const {
        std::cout << "TimeHistory: " << num_records() << " records, "
                  << num_nodal_probes() << " nodal probes, "
                  << num_element_probes() << " element probes\n";
        for (const auto& p : nodal_probes_) {
            std::cout << "  Node probe '" << p.name << "': "
                      << p.nodes.size() << " nodes, " << to_string(p.quantity) << "\n";
        }
        for (const auto& p : element_probes_) {
            std::cout << "  Elem probe '" << p.name << "': "
                      << p.elements.size() << " elements, " << to_string(p.quantity) << "\n";
        }
    }

    /**
     * @brief Clear all recorded data (keep probes)
     */
    void clear_data() {
        times_.clear();
        probe_data_.clear();
        call_count_ = 0;
    }

private:
    Real extract_nodal(NodalQuantity q, Index nid,
                       const Real* disp, const Real* vel,
                       const Real* accel, const Real* force) const {
        switch (q) {
            case NodalQuantity::DisplacementX:   return disp ? disp[3*nid+0] : 0.0;
            case NodalQuantity::DisplacementY:   return disp ? disp[3*nid+1] : 0.0;
            case NodalQuantity::DisplacementZ:   return disp ? disp[3*nid+2] : 0.0;
            case NodalQuantity::DisplacementMag: return disp ? mag3(disp+3*nid) : 0.0;
            case NodalQuantity::VelocityX:       return vel ? vel[3*nid+0] : 0.0;
            case NodalQuantity::VelocityY:       return vel ? vel[3*nid+1] : 0.0;
            case NodalQuantity::VelocityZ:       return vel ? vel[3*nid+2] : 0.0;
            case NodalQuantity::VelocityMag:     return vel ? mag3(vel+3*nid) : 0.0;
            case NodalQuantity::AccelerationX:   return accel ? accel[3*nid+0] : 0.0;
            case NodalQuantity::AccelerationY:   return accel ? accel[3*nid+1] : 0.0;
            case NodalQuantity::AccelerationZ:   return accel ? accel[3*nid+2] : 0.0;
            case NodalQuantity::AccelerationMag: return accel ? mag3(accel+3*nid) : 0.0;
            case NodalQuantity::ForceX:          return force ? force[3*nid+0] : 0.0;
            case NodalQuantity::ForceY:          return force ? force[3*nid+1] : 0.0;
            case NodalQuantity::ForceZ:          return force ? force[3*nid+2] : 0.0;
            case NodalQuantity::ForceMag:        return force ? mag3(force+3*nid) : 0.0;
            default: return 0.0;
        }
    }

    Real extract_element(ElementQuantity q, Index eid,
                          const Real* stress, const Real* strain,
                          const Real* eps_p, const Real* damage) const {
        switch (q) {
            case ElementQuantity::StressXX: return stress ? stress[6*eid+0] : 0.0;
            case ElementQuantity::StressYY: return stress ? stress[6*eid+1] : 0.0;
            case ElementQuantity::StressZZ: return stress ? stress[6*eid+2] : 0.0;
            case ElementQuantity::StressXY: return stress ? stress[6*eid+3] : 0.0;
            case ElementQuantity::StressYZ: return stress ? stress[6*eid+4] : 0.0;
            case ElementQuantity::StressXZ: return stress ? stress[6*eid+5] : 0.0;
            case ElementQuantity::VonMises: {
                if (!stress) return 0.0;
                const Real* s = stress + 6*eid;
                Real dx = s[0]-s[1], dy = s[1]-s[2], dz = s[0]-s[2];
                return std::sqrt(0.5*(dx*dx + dy*dy + dz*dz) +
                                 3.0*(s[3]*s[3] + s[4]*s[4] + s[5]*s[5]));
            }
            case ElementQuantity::Pressure: {
                if (!stress) return 0.0;
                return -(stress[6*eid+0] + stress[6*eid+1] + stress[6*eid+2]) / 3.0;
            }
            case ElementQuantity::StrainXX: return strain ? strain[6*eid+0] : 0.0;
            case ElementQuantity::StrainYY: return strain ? strain[6*eid+1] : 0.0;
            case ElementQuantity::StrainZZ: return strain ? strain[6*eid+2] : 0.0;
            case ElementQuantity::StrainXY: return strain ? strain[6*eid+3] : 0.0;
            case ElementQuantity::StrainYZ: return strain ? strain[6*eid+4] : 0.0;
            case ElementQuantity::StrainXZ: return strain ? strain[6*eid+5] : 0.0;
            case ElementQuantity::EffectivePlasticStrain: return eps_p ? eps_p[eid] : 0.0;
            case ElementQuantity::Damage: return damage ? damage[eid] : 0.0;
            case ElementQuantity::VolumetricStrain: {
                if (!strain) return 0.0;
                return strain[6*eid+0] + strain[6*eid+1] + strain[6*eid+2];
            }
            default: return 0.0;
        }
    }

    static Real mag3(const Real* v) {
        return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    }

    // Probe definitions
    std::vector<NodalProbe> nodal_probes_;
    std::vector<ElementProbe> element_probes_;

    // Recorded data
    std::vector<Real> times_;
    std::map<std::string, std::vector<std::vector<Real>>> probe_data_;

    // Recording control
    int record_interval_ = 1;
    int call_count_ = 0;
};

// ============================================================================
// Global Energy Tracker
// ============================================================================

/**
 * @brief Tracks global energy balance over time
 *
 * Essential for verifying simulation quality:
 * - Kinetic energy
 * - Internal (strain) energy
 * - External work
 * - Contact energy
 * - Hourglass energy
 * - Energy balance error
 */
class EnergyTracker {
public:
    struct EnergyRecord {
        Real time;
        Real kinetic;
        Real internal;
        Real external_work;
        Real contact;
        Real hourglass;
        Real total;
        Real balance_error;  ///< (total - initial) / initial
    };

    EnergyTracker() = default;

    /**
     * @brief Record energy at current time step
     */
    void record(Real time, Real ke, Real ie, Real ext_work = 0.0,
                Real contact_e = 0.0, Real hg_e = 0.0) {
        EnergyRecord rec;
        rec.time = time;
        rec.kinetic = ke;
        rec.internal = ie;
        rec.external_work = ext_work;
        rec.contact = contact_e;
        rec.hourglass = hg_e;
        rec.total = ke + ie + contact_e + hg_e - ext_work;

        if (records_.empty()) {
            initial_total_ = rec.total;
        }
        rec.balance_error = (initial_total_ != 0.0)
            ? (rec.total - initial_total_) / std::fabs(initial_total_)
            : 0.0;

        records_.push_back(rec);
    }

    /**
     * @brief Write energy history to CSV
     */
    bool write_csv(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        file << "time,kinetic,internal,external_work,contact,hourglass,total,balance_error\n";
        for (const auto& r : records_) {
            file << std::scientific << std::setprecision(8)
                 << r.time << "," << r.kinetic << "," << r.internal << ","
                 << r.external_work << "," << r.contact << "," << r.hourglass << ","
                 << r.total << "," << r.balance_error << "\n";
        }
        return true;
    }

    std::size_t num_records() const { return records_.size(); }

    const EnergyRecord& latest() const { return records_.back(); }
    const EnergyRecord& record_at(std::size_t i) const { return records_[i]; }
    const std::vector<EnergyRecord>& records() const { return records_; }

    Real max_balance_error() const {
        Real max_err = 0.0;
        for (const auto& r : records_) {
            Real err = std::fabs(r.balance_error);
            if (err > max_err) max_err = err;
        }
        return max_err;
    }

    void print_summary() const {
        if (records_.empty()) return;
        const auto& r = records_.back();
        std::cout << "Energy: KE=" << r.kinetic << " IE=" << r.internal
                  << " Total=" << r.total
                  << " Error=" << r.balance_error * 100.0 << "%\n";
    }

private:
    std::vector<EnergyRecord> records_;
    Real initial_total_ = 0.0;
};

} // namespace io
} // namespace nxs
