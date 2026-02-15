#pragma once

/**
 * @file part_energy.hpp
 * @brief Per-part kinetic/internal energy decomposition
 *
 * Tracks kinetic and internal energy for each element block (part) over time.
 * Supports CSV export for post-processing.
 *
 * Reference: LS-DYNA *DATABASE_MATSUM
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
// Per-Part Energy Record
// ============================================================================

struct PartEnergyRecord {
    Real time;
    int part_id;
    std::string part_name;
    Real kinetic_energy;
    Real internal_energy;
    Real hourglass_energy;
    Real total_energy;

    PartEnergyRecord()
        : time(0.0), part_id(0)
        , kinetic_energy(0.0), internal_energy(0.0)
        , hourglass_energy(0.0), total_energy(0.0) {}
};

// ============================================================================
// Part Energy Tracker
// ============================================================================

class PartEnergyTracker {
public:
    PartEnergyTracker() = default;

    /**
     * @brief Register a part (element block) for energy tracking
     * @param part_id Part ID
     * @param name Part name
     * @param element_ids Element indices belonging to this part
     */
    void register_part(int part_id, const std::string& name,
                        const std::vector<Index>& element_ids) {
        PartInfo pi;
        pi.part_id = part_id;
        pi.name = name;
        pi.element_ids = element_ids;
        parts_.push_back(pi);
    }

    /**
     * @brief Record energy for all registered parts at a given time
     * @param time Current simulation time
     * @param num_elements Total number of elements
     * @param element_ke Per-element kinetic energy array
     * @param element_ie Per-element internal energy array
     * @param element_hge Per-element hourglass energy array (optional)
     */
    void record(Real time,
                std::size_t /*num_elements*/,
                const Real* element_ke,
                const Real* element_ie,
                const Real* element_hge = nullptr) {

        times_.push_back(time);

        for (auto& part : parts_) {
            PartEnergyRecord rec;
            rec.time = time;
            rec.part_id = part.part_id;
            rec.part_name = part.name;
            rec.kinetic_energy = 0.0;
            rec.internal_energy = 0.0;
            rec.hourglass_energy = 0.0;

            for (Index eid : part.element_ids) {
                if (element_ke) rec.kinetic_energy += element_ke[eid];
                if (element_ie) rec.internal_energy += element_ie[eid];
                if (element_hge) rec.hourglass_energy += element_hge[eid];
            }

            rec.total_energy = rec.kinetic_energy + rec.internal_energy + rec.hourglass_energy;
            records_[part.part_id].push_back(rec);
        }
    }

    /**
     * @brief Write all part energy data to a combined CSV file
     */
    bool write_csv(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        file << "time,part_id,part_name,kinetic_energy,internal_energy,"
             << "hourglass_energy,total_energy\n";

        file << std::scientific << std::setprecision(8);

        for (const auto& [pid, records] : records_) {
            for (const auto& r : records) {
                file << r.time << "," << r.part_id << "," << r.part_name << ","
                     << r.kinetic_energy << "," << r.internal_energy << ","
                     << r.hourglass_energy << "," << r.total_energy << "\n";
            }
        }

        return file.good();
    }

    /**
     * @brief Write per-part CSV files
     */
    bool write_csv_per_part(const std::string& base_path) const {
        for (const auto& [pid, records] : records_) {
            std::string filename = base_path + "_part" + std::to_string(pid) + ".csv";
            std::ofstream file(filename);
            if (!file.is_open()) return false;

            file << "time,kinetic_energy,internal_energy,hourglass_energy,total_energy\n";
            file << std::scientific << std::setprecision(8);

            for (const auto& r : records) {
                file << r.time << "," << r.kinetic_energy << ","
                     << r.internal_energy << "," << r.hourglass_energy << ","
                     << r.total_energy << "\n";
            }
        }
        return true;
    }

    /**
     * @brief Get records for a specific part
     */
    const std::vector<PartEnergyRecord>* get_part_records(int part_id) const {
        auto it = records_.find(part_id);
        if (it == records_.end()) return nullptr;
        return &it->second;
    }

    /**
     * @brief Get total energy across all parts at the latest recorded time
     */
    Real total_kinetic_energy() const {
        Real total = 0.0;
        for (const auto& [pid, records] : records_) {
            if (!records.empty()) total += records.back().kinetic_energy;
        }
        return total;
    }

    Real total_internal_energy() const {
        Real total = 0.0;
        for (const auto& [pid, records] : records_) {
            if (!records.empty()) total += records.back().internal_energy;
        }
        return total;
    }

    std::size_t num_parts() const { return parts_.size(); }
    std::size_t num_records() const { return times_.size(); }
    const std::vector<Real>& times() const { return times_; }

private:
    struct PartInfo {
        int part_id;
        std::string name;
        std::vector<Index> element_ids;
    };

    std::vector<PartInfo> parts_;
    std::vector<Real> times_;
    std::map<int, std::vector<PartEnergyRecord>> records_;
};

} // namespace io
} // namespace nxs
