#pragma once

/**
 * @file output_wave40.hpp
 * @brief Wave 40: Extended Output Format Writers
 *
 * Six output format features:
 *  1. RadiossAnimWriter    - Native RADIOSS .anim binary format
 *  2. StatusFileWriter     - .sta status file (ASCII)
 *  3. DynainWriter         - Dynain restart file (deformed mesh + state)
 *  4. ReactionForcesTH     - Reaction forces time history
 *  5. QAPrintWriter        - Quality assurance print
 *  6. ReportGenerator      - Simulation summary report (ASCII)
 *
 * Reference: OpenRadioss output formats, LS-DYNA D3PLOT/dynain,
 *            Altair HyperWorks animation format specification.
 */

#include <nexussim/core/types.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <array>
#include <iomanip>

namespace nxs {
namespace io {

using Real = nxs::Real;

// ============================================================================
// Common data structures for Wave 40 writers
// ============================================================================

/// Node data for animation/restart output
struct AnimNodeData {
    int id = 0;
    Real x = 0, y = 0, z = 0;           ///< Current positions
    Real vx = 0, vy = 0, vz = 0;        ///< Velocities
    Real ax = 0, ay = 0, az = 0;        ///< Accelerations
};

/// Element data for animation/restart output
struct AnimElementData {
    int id = 0;
    int part_id = 0;
    int type = 0;                         ///< Element type code
    std::array<Real, 6> stress = {};      ///< xx, yy, zz, xy, yz, xz
    std::array<Real, 6> strain = {};      ///< xx, yy, zz, xy, yz, xz
    Real plastic_strain = 0;
    std::vector<int> connectivity;
};

/// Energy data for status output
struct EnergyData {
    Real kinetic = 0;
    Real internal = 0;
    Real contact = 0;
    Real external_work = 0;
    Real hourglass = 0;

    Real total() const {
        return kinetic + internal + contact + hourglass;
    }
};

/// Reaction force data at a node
struct ReactionForce {
    int node_id = 0;
    Real fx = 0, fy = 0, fz = 0;
    Real mx = 0, my = 0, mz = 0;
};

/// Mesh information for QA checks
struct MeshInfo {
    int node_count = 0;
    int elem_count = 0;
    int material_count = 0;
    int part_count = 0;
    int shell_count = 0;
    int solid_count = 0;
    int beam_count = 0;
    Real min_aspect_ratio = 1.0;
    Real max_aspect_ratio = 1.0;
    Real min_jacobian = 1.0;
    Real max_jacobian = 1.0;
    Real max_warping = 0.0;
    int unassigned_material_count = 0;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
};

// ############################################################################
// 1. RadiossAnimWriter -- Native RADIOSS .anim binary format
// ############################################################################

/**
 * Binary .anim file with:
 *   [Header: version, title, timestamp]
 *   [Frame 0: time_marker, node_count, node_data[], elem_count, elem_data[]]
 *   [Frame 1: ...]
 *
 * All multi-byte values are little-endian (native x86).
 */
class RadiossAnimWriter {
public:
    static constexpr uint32_t MAGIC = 0x414E494D;   // "ANIM"
    static constexpr uint16_t VERSION = 1;
    static constexpr int TITLE_LEN = 80;

    RadiossAnimWriter() = default;
    ~RadiossAnimWriter() { close(); }

    /// Open file for writing. Returns true on success.
    bool open(const std::string& filename) {
        close();
        file_.open(filename, std::ios::binary | std::ios::out);
        if (!file_.is_open()) return false;
        filename_ = filename;
        frame_count_ = 0;
        write_header();
        return true;
    }

    /// Write a single animation frame.
    bool write_frame(Real time,
                     const std::vector<AnimNodeData>& nodes,
                     const std::vector<AnimElementData>& elements) {
        if (!file_.is_open()) return false;

        // Time marker
        write_val(time);

        // Node block
        int32_t ncount = static_cast<int32_t>(nodes.size());
        write_val(ncount);
        for (const auto& n : nodes) {
            write_val(static_cast<int32_t>(n.id));
            write_val(n.x); write_val(n.y); write_val(n.z);
            write_val(n.vx); write_val(n.vy); write_val(n.vz);
            write_val(n.ax); write_val(n.ay); write_val(n.az);
        }

        // Element block
        int32_t ecount = static_cast<int32_t>(elements.size());
        write_val(ecount);
        for (const auto& e : elements) {
            write_val(static_cast<int32_t>(e.id));
            write_val(static_cast<int32_t>(e.part_id));
            for (int i = 0; i < 6; ++i) write_val(e.stress[i]);
            for (int i = 0; i < 6; ++i) write_val(e.strain[i]);
            write_val(e.plastic_strain);
        }

        ++frame_count_;
        return true;
    }

    /// Close the file and finalize the header frame count.
    void close() {
        if (file_.is_open()) {
            // Seek back to frame count position and update
            file_.seekp(header_frame_count_offset_);
            write_val(frame_count_);
            file_.close();
        }
    }

    int frame_count() const { return frame_count_; }
    const std::string& filename() const { return filename_; }

private:
    std::ofstream file_;
    std::string filename_;
    int32_t frame_count_ = 0;
    std::streampos header_frame_count_offset_ = 0;

    void write_header() {
        // Magic
        write_val(MAGIC);
        // Version
        write_val(VERSION);
        // Title (fixed 80 bytes, null-padded)
        char title[TITLE_LEN] = {};
        std::string t = "NexusSim RADIOSS Animation";
        std::copy(t.begin(), t.end(), title);
        file_.write(title, TITLE_LEN);
        // Timestamp (8 bytes, zero for now)
        int64_t timestamp = 0;
        write_val(timestamp);
        // Frame count (will be updated on close)
        header_frame_count_offset_ = file_.tellp();
        write_val(frame_count_);
    }

    template<typename T>
    void write_val(const T& v) {
        file_.write(reinterpret_cast<const char*>(&v), sizeof(T));
    }
};

// ############################################################################
// 2. StatusFileWriter -- .sta status file (ASCII)
// ############################################################################

/**
 * ASCII status file with columns:
 *   cycle  time  dt  kinetic_E  internal_E  total_E  mass_error
 */
class StatusFileWriter {
public:
    StatusFileWriter() = default;
    ~StatusFileWriter() { close(); }

    /// Open status file. Returns true on success.
    bool open(const std::string& filename) {
        close();
        file_.open(filename, std::ios::out);
        if (!file_.is_open()) return false;
        filename_ = filename;
        write_header_line();
        line_count_ = 0;
        return true;
    }

    /// Write one status line.
    bool write_status(int cycle, Real time, Real dt, const EnergyData& energies,
                      Real mass_error = 0.0) {
        if (!file_.is_open()) return false;

        file_ << std::setw(10) << cycle
              << std::scientific << std::setprecision(6)
              << std::setw(16) << time
              << std::setw(16) << dt
              << std::setw(16) << energies.kinetic
              << std::setw(16) << energies.internal
              << std::setw(16) << energies.total()
              << std::setw(16) << mass_error
              << "\n";

        ++line_count_;
        return true;
    }

    /// Close the file.
    void close() {
        if (file_.is_open()) {
            file_.close();
        }
    }

    int line_count() const { return line_count_; }
    const std::string& filename() const { return filename_; }

private:
    std::ofstream file_;
    std::string filename_;
    int line_count_ = 0;

    void write_header_line() {
        file_ << std::setw(10) << "CYCLE"
              << std::setw(16) << "TIME"
              << std::setw(16) << "DT"
              << std::setw(16) << "KINETIC_E"
              << std::setw(16) << "INTERNAL_E"
              << std::setw(16) << "TOTAL_E"
              << std::setw(16) << "MASS_ERR"
              << "\n";
        // Separator line
        file_ << std::string(10 + 6 * 16, '-') << "\n";
    }
};

// ############################################################################
// 3. DynainWriter -- Dynain restart file
// ############################################################################

/**
 * ASCII restart file containing deformed mesh and material state.
 *
 * Format:
 *   *KEYWORD
 *   *NODE
 *     id  x  y  z
 *   *ELEMENT_SOLID (or *ELEMENT_SHELL)
 *     id  pid  n1 n2 n3 ...
 *   *INITIAL_STRESS_SOLID
 *     eid  sigxx sigyy sigzz sigxy sigyz sigxz eps_p
 *   *END
 */
class DynainWriter {
public:
    DynainWriter() = default;

    /// Write a complete dynain file.
    bool write(const std::string& filename,
               const std::vector<AnimNodeData>& nodes,
               const std::vector<AnimElementData>& elements) {
        std::ofstream file(filename, std::ios::out);
        if (!file.is_open()) return false;

        file << "*KEYWORD\n";
        file << "$ NexusSim Dynain Restart File\n";
        file << "$\n";

        // Node block
        file << "*NODE\n";
        file << std::fixed << std::setprecision(8);
        for (const auto& n : nodes) {
            file << std::setw(8) << n.id
                 << std::setw(16) << n.x
                 << std::setw(16) << n.y
                 << std::setw(16) << n.z
                 << "\n";
        }

        // Element block
        if (!elements.empty()) {
            // Determine element type from first element
            bool is_shell = false;
            if (!elements[0].connectivity.empty()) {
                is_shell = (elements[0].connectivity.size() <= 4);
            }

            if (is_shell) {
                file << "*ELEMENT_SHELL\n";
            } else {
                file << "*ELEMENT_SOLID\n";
            }

            for (const auto& e : elements) {
                file << std::setw(8) << e.id
                     << std::setw(8) << e.part_id;
                for (int nid : e.connectivity) {
                    file << std::setw(8) << nid;
                }
                file << "\n";
            }

            // Initial stress block
            file << "*INITIAL_STRESS_SOLID\n";
            for (const auto& e : elements) {
                file << std::setw(8) << e.id;
                for (int i = 0; i < 6; ++i) {
                    file << std::setw(16) << e.stress[i];
                }
                file << std::setw(16) << e.plastic_strain;
                file << "\n";
            }
        }

        file << "*END\n";
        file.close();
        nodes_written_ = static_cast<int>(nodes.size());
        elements_written_ = static_cast<int>(elements.size());
        return true;
    }

    int nodes_written() const { return nodes_written_; }
    int elements_written() const { return elements_written_; }

private:
    int nodes_written_ = 0;
    int elements_written_ = 0;
};

// ############################################################################
// 4. ReactionForcesTH -- Reaction forces time history
// ############################################################################

/**
 * Tracks reaction forces at constrained nodes over time.
 * Stores time-stamped records, outputs ASCII file with columns:
 *   time  node_id  Fx  Fy  Fz  Mx  My  Mz
 */
class ReactionForcesTH {
public:
    ReactionForcesTH() = default;

    /// Register a node to be tracked.
    void add_node(int node_id) {
        tracked_nodes_.push_back(node_id);
    }

    /// Record reaction forces at current time.
    /// forces_map: node_id -> ReactionForce
    void record(Real time, const std::vector<ReactionForce>& forces) {
        TimeRecord rec;
        rec.time = time;
        for (const auto& f : forces) {
            // Only store tracked nodes
            for (int nid : tracked_nodes_) {
                if (f.node_id == nid) {
                    rec.forces.push_back(f);
                    break;
                }
            }
        }
        records_.push_back(std::move(rec));
    }

    /// Write all recorded data to file.
    bool write(const std::string& filename) const {
        std::ofstream file(filename, std::ios::out);
        if (!file.is_open()) return false;

        // Header
        file << std::setw(16) << "TIME"
             << std::setw(10) << "NODE_ID"
             << std::setw(16) << "FX"
             << std::setw(16) << "FY"
             << std::setw(16) << "FZ"
             << std::setw(16) << "MX"
             << std::setw(16) << "MY"
             << std::setw(16) << "MZ"
             << "\n";
        file << std::string(16 + 10 + 6 * 16, '-') << "\n";

        // Data
        file << std::scientific << std::setprecision(6);
        for (const auto& rec : records_) {
            for (const auto& f : rec.forces) {
                file << std::setw(16) << rec.time
                     << std::setw(10) << f.node_id
                     << std::setw(16) << f.fx
                     << std::setw(16) << f.fy
                     << std::setw(16) << f.fz
                     << std::setw(16) << f.mx
                     << std::setw(16) << f.my
                     << std::setw(16) << f.mz
                     << "\n";
            }
        }

        file.close();
        return true;
    }

    /// Get total number of time records.
    int num_records() const { return static_cast<int>(records_.size()); }

    /// Get tracked node IDs.
    const std::vector<int>& tracked_nodes() const { return tracked_nodes_; }

    /// Get total force entries across all records.
    int total_entries() const {
        int count = 0;
        for (const auto& rec : records_) {
            count += static_cast<int>(rec.forces.size());
        }
        return count;
    }

    /// Get reaction force sum across all tracked nodes at a given record index.
    ReactionForce resultant(int record_index) const {
        ReactionForce sum;
        if (record_index < 0 || record_index >= static_cast<int>(records_.size()))
            return sum;
        for (const auto& f : records_[record_index].forces) {
            sum.fx += f.fx; sum.fy += f.fy; sum.fz += f.fz;
            sum.mx += f.mx; sum.my += f.my; sum.mz += f.mz;
        }
        return sum;
    }

private:
    struct TimeRecord {
        Real time = 0;
        std::vector<ReactionForce> forces;
    };

    std::vector<int> tracked_nodes_;
    std::vector<TimeRecord> records_;
};

// ############################################################################
// 5. QAPrintWriter -- Quality assurance print
// ############################################################################

/**
 * Analyzes mesh quality and writes a QA report:
 *   - Model summary (counts of nodes, elements, materials, parts)
 *   - Mesh quality metrics (aspect ratio, Jacobian, warping)
 *   - Warning/error summary
 */
class QAPrintWriter {
public:
    QAPrintWriter() = default;

    /// Analyze mesh info and populate internal results.
    void analyze(const MeshInfo& info) {
        info_ = info;
        analyzed_ = true;

        // Generate quality-based warnings
        if (info_.min_jacobian < 0.0) {
            add_error("Negative Jacobian detected (min = " +
                      format_real(info_.min_jacobian) + ")");
        }
        if (info_.max_aspect_ratio > 10.0) {
            add_warning("High aspect ratio detected (max = " +
                        format_real(info_.max_aspect_ratio) + ")");
        }
        if (info_.max_warping > 15.0) {
            add_warning("Excessive warping angle detected (max = " +
                        format_real(info_.max_warping) + " deg)");
        }
        if (info_.unassigned_material_count > 0) {
            add_error("Elements without material assignment: " +
                      std::to_string(info_.unassigned_material_count));
        }

        // Merge warnings/errors from MeshInfo
        for (const auto& w : info_.warnings) add_warning(w);
        for (const auto& e : info_.errors) add_error(e);
    }

    /// Write QA report to file.
    bool write(const std::string& filename) const {
        std::ofstream file(filename, std::ios::out);
        if (!file.is_open()) return false;

        file << "================================================================\n";
        file << "          NexusSim Quality Assurance Report\n";
        file << "================================================================\n\n";

        // Model summary
        file << "MODEL SUMMARY\n";
        file << "----------------------------------------------------------------\n";
        file << "  Nodes:      " << std::setw(10) << info_.node_count << "\n";
        file << "  Elements:   " << std::setw(10) << info_.elem_count << "\n";
        file << "    Shells:   " << std::setw(10) << info_.shell_count << "\n";
        file << "    Solids:   " << std::setw(10) << info_.solid_count << "\n";
        file << "    Beams:    " << std::setw(10) << info_.beam_count << "\n";
        file << "  Materials:  " << std::setw(10) << info_.material_count << "\n";
        file << "  Parts:      " << std::setw(10) << info_.part_count << "\n";
        file << "\n";

        // Mesh quality
        file << "MESH QUALITY METRICS\n";
        file << "----------------------------------------------------------------\n";
        file << std::fixed << std::setprecision(4);
        file << "  Aspect ratio (min/max): "
             << info_.min_aspect_ratio << " / " << info_.max_aspect_ratio << "\n";
        file << "  Jacobian     (min/max): "
             << info_.min_jacobian << " / " << info_.max_jacobian << "\n";
        file << "  Max warping angle:      " << info_.max_warping << " deg\n";
        file << "\n";

        // Warnings
        file << "WARNINGS (" << warnings_.size() << ")\n";
        file << "----------------------------------------------------------------\n";
        for (size_t i = 0; i < warnings_.size(); ++i) {
            file << "  [W" << (i + 1) << "] " << warnings_[i] << "\n";
        }
        if (warnings_.empty()) file << "  (none)\n";
        file << "\n";

        // Errors
        file << "ERRORS (" << errors_.size() << ")\n";
        file << "----------------------------------------------------------------\n";
        for (size_t i = 0; i < errors_.size(); ++i) {
            file << "  [E" << (i + 1) << "] " << errors_[i] << "\n";
        }
        if (errors_.empty()) file << "  (none)\n";
        file << "\n";

        file << "================================================================\n";
        file << "End of QA Report\n";
        file << "================================================================\n";

        file.close();
        return true;
    }

    int warning_count() const { return static_cast<int>(warnings_.size()); }
    int error_count() const { return static_cast<int>(errors_.size()); }
    bool is_analyzed() const { return analyzed_; }
    const MeshInfo& mesh_info() const { return info_; }

private:
    MeshInfo info_;
    bool analyzed_ = false;
    std::vector<std::string> warnings_;
    std::vector<std::string> errors_;

    void add_warning(const std::string& msg) {
        warnings_.push_back(msg);
    }

    void add_error(const std::string& msg) {
        errors_.push_back(msg);
    }

    static std::string format_real(Real v) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(4) << v;
        return oss.str();
    }
};

// ############################################################################
// 6. ReportGenerator -- Simulation summary report (ASCII)
// ############################################################################

/**
 * Generates a structured ASCII report with named sections:
 *   - Run parameters and hardware info
 *   - Energy balance summary
 *   - Contact summary, material usage
 *   - Custom user-defined sections
 */
class ReportGenerator {
public:
    ReportGenerator() = default;

    /// Add a named section with content.
    void add_section(const std::string& title, const std::string& content) {
        sections_.push_back({title, content});
    }

    /// Convenience: add run parameters section.
    void add_run_info(const std::string& solver_type, Real end_time,
                      Real dt_initial, int num_threads, const std::string& hardware) {
        std::ostringstream oss;
        oss << "  Solver type:      " << solver_type << "\n"
            << "  End time:         " << std::scientific << std::setprecision(4) << end_time << "\n"
            << "  Initial dt:       " << dt_initial << "\n"
            << "  Threads:          " << num_threads << "\n"
            << "  Hardware:         " << hardware << "\n";
        add_section("RUN PARAMETERS", oss.str());
    }

    /// Convenience: add energy balance section.
    void add_energy_summary(const EnergyData& initial, const EnergyData& final_e) {
        Real balance_error = 0.0;
        Real denom = std::abs(initial.total());
        if (denom > 1e-30) {
            balance_error = std::abs(final_e.total() - initial.total()) / denom;
        }

        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6);
        oss << "  Initial kinetic:  " << initial.kinetic << "\n"
            << "  Initial internal: " << initial.internal << "\n"
            << "  Initial total:    " << initial.total() << "\n"
            << "  Final kinetic:    " << final_e.kinetic << "\n"
            << "  Final internal:   " << final_e.internal << "\n"
            << "  Final total:      " << final_e.total() << "\n"
            << "  Balance error:    " << balance_error * 100.0 << " %\n";
        add_section("ENERGY BALANCE", oss.str());
    }

    /// Convenience: add timing section.
    void add_timing(Real wall_time, Real cpu_time, int cycles) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "  Wall time:        " << wall_time << " s\n"
            << "  CPU time:         " << cpu_time << " s\n"
            << "  Cycles:           " << cycles << "\n";
        if (cycles > 0) {
            oss << "  Avg time/cycle:   " << std::scientific << std::setprecision(4)
                << (wall_time / cycles) << " s\n";
        }
        add_section("TIMING", oss.str());
    }

    /// Convenience: add contact summary.
    void add_contact_summary(int num_interfaces, int total_pairs_checked,
                             int active_contacts) {
        std::ostringstream oss;
        oss << "  Interfaces:       " << num_interfaces << "\n"
            << "  Pairs checked:    " << total_pairs_checked << "\n"
            << "  Active contacts:  " << active_contacts << "\n";
        add_section("CONTACT SUMMARY", oss.str());
    }

    /// Generate the complete report file.
    bool generate(const std::string& filename) const {
        std::ofstream file(filename, std::ios::out);
        if (!file.is_open()) return false;

        file << "****************************************************************\n";
        file << "*                                                              *\n";
        file << "*            NexusSim Simulation Summary Report                *\n";
        file << "*                                                              *\n";
        file << "****************************************************************\n\n";

        for (size_t i = 0; i < sections_.size(); ++i) {
            file << "------------------------------------------------------------\n";
            file << sections_[i].title << "\n";
            file << "------------------------------------------------------------\n";
            file << sections_[i].content;
            if (!sections_[i].content.empty() &&
                sections_[i].content.back() != '\n') {
                file << "\n";
            }
            file << "\n";
        }

        file << "****************************************************************\n";
        file << "* End of Report                                                *\n";
        file << "****************************************************************\n";

        file.close();
        return true;
    }

    /// Number of sections added.
    int num_sections() const { return static_cast<int>(sections_.size()); }

    /// Clear all sections.
    void clear() { sections_.clear(); }

private:
    struct Section {
        std::string title;
        std::string content;
    };
    std::vector<Section> sections_;
};

} // namespace io
} // namespace nxs
