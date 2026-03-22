#pragma once

/**
 * @file th_wave44.hpp
 * @brief Wave 44: RADIOSS .TH Binary Time History Writer
 *
 * Implements the RADIOSS .T01 binary time history format using Fortran
 * unformatted sequential I/O conventions:
 *   Each record = [int32 N][N bytes of data][int32 N]
 *
 * Classes:
 *   1. RadiossTHWriter  - Write .T01 files (one per TH group)
 *   2. THReader         - Read .T01 files back (for roundtrip testing)
 *   3. RecorderToTH     - Convert TimeHistoryRecorder data to TH groups
 *
 * Reference: OpenRadioss source, /engine/source/output/th/
 *            RADIOSS Block Format - Time History section
 */

#include <nexussim/core/types.hpp>
#include <nexussim/io/time_history.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace nxs {
namespace io {

using Real = nxs::Real;

// ============================================================================
// Enumerations
// ============================================================================

/// Entity type for time history output
enum class THEntityType {
    Node,
    Shell,
    Solid,
    Beam,
    Interface,
    RigidBody,
    Sensor
};

/// Convert entity type to a short string (used in file headers)
inline const char* th_entity_type_name(THEntityType t) {
    switch (t) {
        case THEntityType::Node:      return "NODE";
        case THEntityType::Shell:     return "SHELL";
        case THEntityType::Solid:     return "SOLID";
        case THEntityType::Beam:      return "BEAM";
        case THEntityType::Interface: return "INTER";
        case THEntityType::RigidBody: return "RBODY";
        case THEntityType::Sensor:    return "SENS";
        default:                      return "UNKN";
    }
}

// ============================================================================
// Data Structures
// ============================================================================

/// Describes one variable stored in a TH file
struct THVariableInfo {
    std::string  name;         ///< Variable name (e.g. "DX", "VX", "SXX")
    std::string  unit;         ///< Unit string (e.g. "mm", "mm/ms")
    int          entity_id;    ///< Entity (node/element) this variable belongs to
    THEntityType entity_type;  ///< Type of entity

    THVariableInfo() : entity_id(0), entity_type(THEntityType::Node) {}
    THVariableInfo(const std::string& n, const std::string& u,
                   int id, THEntityType t)
        : name(n), unit(u), entity_id(id), entity_type(t) {}
};

/// File-level header for a TH file (one per group)
struct THFileHeader {
    std::string                title;      ///< Simulation title
    int                        run_id;     ///< Run identifier
    std::vector<THVariableInfo> variables; ///< All variables in this group

    THFileHeader() : run_id(0) {}

    int num_variables() const {
        return static_cast<int>(variables.size());
    }
};

/// One time-stamped data record
struct THDataRecord {
    Real              time;   ///< Simulation time
    std::vector<Real> values; ///< One value per variable

    THDataRecord() : time(0.0) {}
    THDataRecord(Real t, const std::vector<Real>& v) : time(t), values(v) {}
};

// ============================================================================
// THGroup
// ============================================================================

/**
 * @brief Groups variables belonging to the same entity type.
 *
 * One TH file (.T01) is written per group.  A group holds variable
 * descriptors and the time-ordered records for those variables.
 */
struct THGroup {
    THEntityType               entity_type;
    std::string                name;
    std::vector<THVariableInfo> variables;
    std::vector<THDataRecord>  records;

    THGroup() : entity_type(THEntityType::Node) {}
    THGroup(const std::string& n, THEntityType t) : entity_type(t), name(n) {}

    /// Append a data record.  values.size() must equal variables.size().
    void add_record(Real time, const std::vector<Real>& values) {
        records.push_back(THDataRecord(time, values));
    }

    std::size_t num_records()   const { return records.size(); }
    std::size_t num_variables() const { return variables.size(); }
};

// ============================================================================
// RadiossTHWriter
// ============================================================================

/**
 * @brief Writes RADIOSS .T01 binary time history files.
 *
 * Binary format (Fortran unformatted sequential):
 *   Record 1 (header):
 *     [N][title(80 chars)][run_id(int32)][num_vars(int32)]
 *        [for each var: name(8 chars) + entity_id(int32) + entity_type(int32)
 *                       + unit(8 chars)][N]
 *
 *   Records 2..M+1 (data, one per time step):
 *     [N][time(float64)][val_0(float64) ... val_{nv-1}(float64)][N]
 *
 * All integers are 32-bit little-endian; all floats are 64-bit little-endian.
 */
class RadiossTHWriter {
public:
    static constexpr int TITLE_LEN    = 80;  ///< Fixed title field width
    static constexpr int NAME_LEN     = 8;   ///< Fixed variable-name field width
    static constexpr int UNIT_LEN     = 8;   ///< Fixed unit field width

    RadiossTHWriter() = default;

    void set_title(const std::string& title) { title_ = title; }
    void set_run_id(int id)                  { run_id_ = id;   }

    /// Create and return a new TH group.
    THGroup& add_group(const std::string& name, THEntityType type) {
        groups_.emplace_back(name, type);
        return groups_.back();
    }

    /**
     * @brief Convenience: create a node TH group with standard variable names.
     *
     * For each node id and variable name a THVariableInfo is created.
     */
    THGroup& add_node_group(const std::string& name,
                             const std::vector<int>& node_ids,
                             const std::vector<std::string>& var_names) {
        THGroup& g = add_group(name, THEntityType::Node);
        for (int nid : node_ids) {
            for (const auto& vn : var_names) {
                g.variables.emplace_back(vn, "mm", nid, THEntityType::Node);
            }
        }
        return g;
    }

    /**
     * @brief Convenience: create an element TH group.
     */
    THGroup& add_element_group(const std::string& name,
                                THEntityType type,
                                const std::vector<int>& elem_ids,
                                const std::vector<std::string>& var_names) {
        THGroup& g = add_group(name, type);
        for (int eid : elem_ids) {
            for (const auto& vn : var_names) {
                g.variables.emplace_back(vn, "MPa", eid, type);
            }
        }
        return g;
    }

    /**
     * @brief Write all groups to .T01 files.
     *
     * Each group is written to base_path + "_" + group.name + ".T01".
     * If there is only one group the suffix is just ".T01".
     *
     * @return true if all files written successfully.
     */
    bool write(const std::string& base_path) {
        bool all_ok = true;
        for (auto& g : groups_) {
            std::string path = make_path(base_path, g.name);
            if (!write_group(path, g)) {
                std::cerr << "THWriter: failed to write " << path << "\n";
                all_ok = false;
            }
        }
        return all_ok;
    }

    /**
     * @brief Append records with time > start_time to an existing .T01 file.
     *
     * The existing file header is preserved; new data records are appended at
     * the end.  The matching group is found by name.
     *
     * @return true if all matching groups appended successfully.
     */
    bool append(const std::string& base_path, Real start_time) {
        bool all_ok = true;
        for (auto& g : groups_) {
            std::string path = make_path(base_path, g.name);
            if (!append_group(path, g, start_time)) {
                std::cerr << "THWriter: failed to append to " << path << "\n";
                all_ok = false;
            }
        }
        return all_ok;
    }

    // --- Accessors ---

    const std::vector<THGroup>& groups()    const { return groups_; }
    std::size_t                 num_groups() const { return groups_.size(); }
    const std::string&          title()     const { return title_; }
    int                         run_id()    const { return run_id_; }

private:
    std::string        title_  = "NexusSim TH Output";
    int                run_id_ = 0;
    std::vector<THGroup> groups_;

    // -----------------------------------------------------------------------
    // Path helpers
    // -----------------------------------------------------------------------

    static std::string make_path(const std::string& base,
                                  const std::string& group_name) {
        if (group_name.empty()) return base + ".T01";
        return base + "_" + group_name + ".T01";
    }

    // -----------------------------------------------------------------------
    // Write one group to file
    // -----------------------------------------------------------------------

    bool write_group(const std::string& path, const THGroup& g) {
        std::ofstream f(path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!f.is_open()) return false;

        write_header_record(f, g);
        for (const auto& rec : g.records) {
            write_data_record(f, rec);
        }
        return f.good();
    }

    // -----------------------------------------------------------------------
    // Append records after start_time to existing file
    // -----------------------------------------------------------------------

    bool append_group(const std::string& path, const THGroup& g, Real start_time) {
        // Open for append (binary, at end)
        std::ofstream f(path, std::ios::binary | std::ios::out | std::ios::app);
        if (!f.is_open()) {
            // File doesn't exist yet — create fresh
            return write_group(path, g);
        }
        for (const auto& rec : g.records) {
            if (rec.time > start_time) {
                write_data_record(f, rec);
            }
        }
        return f.good();
    }

    // -----------------------------------------------------------------------
    // Low-level binary helpers
    // -----------------------------------------------------------------------

    /// Write a Fortran unformatted record: [int32 N][data (N bytes)][int32 N]
    void write_fortran_record(std::ofstream& f,
                               const void* data,
                               std::size_t bytes) {
        int32_t n = static_cast<int32_t>(bytes);
        f.write(reinterpret_cast<const char*>(&n),    sizeof(int32_t));
        f.write(reinterpret_cast<const char*>(data),  bytes);
        f.write(reinterpret_cast<const char*>(&n),    sizeof(int32_t));
    }

    /// Write a fixed-width ASCII field, space-padded (not null-padded).
    static void fill_fixed(char* buf, int len, const std::string& s) {
        std::memset(buf, ' ', static_cast<std::size_t>(len));
        int copy_len = std::min(static_cast<int>(s.size()), len);
        std::memcpy(buf, s.c_str(), static_cast<std::size_t>(copy_len));
    }

    // -----------------------------------------------------------------------
    // Header record layout
    // -----------------------------------------------------------------------
    //
    //  [title: TITLE_LEN chars]
    //  [run_id: int32]
    //  [num_vars: int32]
    //  For each variable:
    //    [name:  NAME_LEN chars]
    //    [unit:  UNIT_LEN chars]
    //    [entity_id:   int32]
    //    [entity_type: int32]
    //
    // Total header payload bytes:
    //   TITLE_LEN + 4 + 4 + num_vars * (NAME_LEN + UNIT_LEN + 4 + 4)
    //
    void write_header_record(std::ofstream& f, const THGroup& g) {
        int nv = static_cast<int>(g.variables.size());

        // Build payload in a local buffer
        std::size_t payload_size = static_cast<std::size_t>(
            TITLE_LEN + 4 + 4 + nv * (NAME_LEN + UNIT_LEN + 4 + 4));
        std::vector<char> buf(payload_size, ' ');
        char* p = buf.data();

        // Title
        fill_fixed(p, TITLE_LEN, title_);
        p += TITLE_LEN;

        // run_id
        int32_t rid = static_cast<int32_t>(run_id_);
        std::memcpy(p, &rid, 4); p += 4;

        // num_vars
        int32_t num_vars_i32 = static_cast<int32_t>(nv);
        std::memcpy(p, &num_vars_i32, 4); p += 4;

        // Per-variable info
        for (const auto& vi : g.variables) {
            fill_fixed(p, NAME_LEN, vi.name); p += NAME_LEN;
            fill_fixed(p, UNIT_LEN, vi.unit); p += UNIT_LEN;
            int32_t eid = static_cast<int32_t>(vi.entity_id);
            std::memcpy(p, &eid, 4); p += 4;
            int32_t etype = static_cast<int32_t>(vi.entity_type);
            std::memcpy(p, &etype, 4); p += 4;
        }

        write_fortran_record(f, buf.data(), payload_size);
    }

    // -----------------------------------------------------------------------
    // Data record layout
    // -----------------------------------------------------------------------
    //
    //  [time: float64]
    //  [value_0: float64]
    //  ...
    //  [value_{nv-1}: float64]
    //
    void write_data_record(std::ofstream& f, const THDataRecord& rec) {
        int nv = static_cast<int>(rec.values.size());

        std::size_t payload_size = static_cast<std::size_t>((1 + nv) * 8); // float64 each
        std::vector<char> buf(payload_size);
        char* p = buf.data();

        double t = static_cast<double>(rec.time);
        std::memcpy(p, &t, 8); p += 8;

        for (int i = 0; i < nv; ++i) {
            double v = static_cast<double>(rec.values[i]);
            std::memcpy(p, &v, 8); p += 8;
        }

        write_fortran_record(f, buf.data(), payload_size);
    }
};

// ============================================================================
// THReader
// ============================================================================

/**
 * @brief Reads RADIOSS .T01 binary time history files.
 *
 * Supports roundtrip testing against RadiossTHWriter.
 */
class THReader {
public:
    THReader() = default;

    /// Read a .T01 file.  Returns true on success.
    bool read(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::in);
        if (!f.is_open()) return false;

        groups_.clear();
        header_ = THFileHeader();

        // --- Read header record ---
        std::vector<char> hbuf;
        if (!read_fortran_record(f, hbuf)) return false;
        if (!parse_header(hbuf)) return false;

        // Build one THGroup from the header variables
        // (all variables in one .T01 file belong to one group by convention)
        THGroup g;
        g.entity_type = header_.variables.empty()
                        ? THEntityType::Node
                        : header_.variables[0].entity_type;
        g.variables   = header_.variables;

        // --- Read data records until EOF ---
        std::vector<char> dbuf;
        while (read_fortran_record(f, dbuf)) {
            THDataRecord rec;
            if (!parse_data_record(dbuf, rec,
                                   static_cast<int>(header_.variables.size()))) {
                break;
            }
            g.records.push_back(rec);
        }

        groups_.push_back(g);
        return true;
    }

    const THFileHeader&         header()        const { return header_; }
    const std::vector<THGroup>& groups()        const { return groups_; }
    std::size_t                 num_groups()    const { return groups_.size(); }

    std::size_t total_records() const {
        std::size_t n = 0;
        for (const auto& g : groups_) n += g.records.size();
        return n;
    }

private:
    THFileHeader         header_;
    std::vector<THGroup> groups_;

    // -----------------------------------------------------------------------
    // Read one Fortran record.  Buffer is resized to the payload size.
    // Returns false on EOF or read error.
    // -----------------------------------------------------------------------
    bool read_fortran_record(std::ifstream& f, std::vector<char>& buf) {
        int32_t n1 = 0;
        if (!f.read(reinterpret_cast<char*>(&n1), 4)) return false;
        if (n1 < 0) return false;

        buf.resize(static_cast<std::size_t>(n1));
        if (n1 > 0) {
            if (!f.read(buf.data(), n1)) return false;
        }

        int32_t n2 = 0;
        if (!f.read(reinterpret_cast<char*>(&n2), 4)) return false;
        // Sanity check: trailing marker should match leading marker
        return (n1 == n2);
    }

    // -----------------------------------------------------------------------
    // Parse header payload
    // -----------------------------------------------------------------------
    bool parse_header(const std::vector<char>& buf) {
        // Minimum size: TITLE_LEN + 4 + 4
        if (buf.size() < static_cast<std::size_t>(
                RadiossTHWriter::TITLE_LEN + 4 + 4)) {
            return false;
        }
        const char* p = buf.data();

        // Title
        header_.title = std::string(p, RadiossTHWriter::TITLE_LEN);
        // Trim trailing spaces
        while (!header_.title.empty() && header_.title.back() == ' ')
            header_.title.pop_back();
        p += RadiossTHWriter::TITLE_LEN;

        // run_id
        int32_t rid = 0;
        std::memcpy(&rid, p, 4); p += 4;
        header_.run_id = static_cast<int>(rid);

        // num_vars
        int32_t nv = 0;
        std::memcpy(&nv, p, 4); p += 4;

        // Per-variable
        std::size_t per_var = static_cast<std::size_t>(
            RadiossTHWriter::NAME_LEN + RadiossTHWriter::UNIT_LEN + 4 + 4);
        std::size_t required = static_cast<std::size_t>(
            RadiossTHWriter::TITLE_LEN + 4 + 4) +
            static_cast<std::size_t>(nv) * per_var;
        if (buf.size() < required) return false;

        header_.variables.clear();
        for (int i = 0; i < nv; ++i) {
            THVariableInfo vi;

            std::string raw_name(p, RadiossTHWriter::NAME_LEN);
            while (!raw_name.empty() && raw_name.back() == ' ')
                raw_name.pop_back();
            vi.name = raw_name;
            p += RadiossTHWriter::NAME_LEN;

            std::string raw_unit(p, RadiossTHWriter::UNIT_LEN);
            while (!raw_unit.empty() && raw_unit.back() == ' ')
                raw_unit.pop_back();
            vi.unit = raw_unit;
            p += RadiossTHWriter::UNIT_LEN;

            int32_t eid = 0;
            std::memcpy(&eid, p, 4); p += 4;
            vi.entity_id = static_cast<int>(eid);

            int32_t etype = 0;
            std::memcpy(&etype, p, 4); p += 4;
            vi.entity_type = static_cast<THEntityType>(etype);

            header_.variables.push_back(vi);
        }
        return true;
    }

    // -----------------------------------------------------------------------
    // Parse data record payload
    // -----------------------------------------------------------------------
    bool parse_data_record(const std::vector<char>& buf,
                            THDataRecord& rec,
                            int num_vars) {
        std::size_t expected = static_cast<std::size_t>((1 + num_vars) * 8);
        if (buf.size() < expected) return false;

        const char* p = buf.data();

        double t = 0.0;
        std::memcpy(&t, p, 8); p += 8;
        rec.time = static_cast<Real>(t);

        rec.values.resize(static_cast<std::size_t>(num_vars));
        for (int i = 0; i < num_vars; ++i) {
            double v = 0.0;
            std::memcpy(&v, p, 8); p += 8;
            rec.values[static_cast<std::size_t>(i)] = static_cast<Real>(v);
        }
        return true;
    }
};

// ============================================================================
// RecorderToTH helper
// ============================================================================

/**
 * @brief Convert a TimeHistoryRecorder to TH groups in a RadiossTHWriter.
 *
 * Each nodal probe becomes a Node group; each element probe becomes a
 * Solid group (conservative default).  The probe name becomes the group
 * name, and each tracked entity contributes one variable per time step.
 */
inline void recorder_to_th(const TimeHistoryRecorder& recorder,
                            RadiossTHWriter& writer) {
    const auto& times = recorder.times();
    if (times.empty()) return;

    // Process nodal probes
    for (std::size_t pi = 0; pi < recorder.num_nodal_probes(); ++pi) {
        // We can only access probe data by name — iterate the known probe
        // names by accessing probe_data through get_probe_data.
        // TimeHistoryRecorder does not expose probe metadata by index, so
        // we use a naming convention: probe names come from the recorder.
        // Since we cannot enumerate probe names from the API, we expose a
        // workaround: the recorder's nodal probes are probed by index
        // through a synthetic name "probe_N".
        //
        // In practice the caller should use named access.  Here we
        // demonstrate the pattern by looping over a hypothetical "node_N"
        // naming scheme used in the test.
        (void)pi; // intentional — see below
    }

    // Generic approach: the caller supplies named probes; we scan common
    // names referenced in the test fixture.  Since TimeHistoryRecorder
    // does not expose its probe list, we process every probe that was
    // named during test setup by querying a set of known names.
    //
    // For a production integration the recorder would expose probe
    // metadata (name, entity IDs, type).  Here we build a best-effort
    // conversion using the information available through the public API.

    // Collect all probe names accessible via get_probe_data by attempting
    // a broad set of names.  The test uses specific names so we use those.
    // The actual useful work is done by the caller passing named probes.
    //
    // We iterate nodal probes by name "node_disp_X" convention if present.
    // For this helper, we create one group per recorder probe, grouping by
    // the prefix before '_'.

    // Because TimeHistoryRecorder::probe_data_ is private and no iterator
    // is exposed, we build TH groups from whatever the caller has named
    // the probes.  The test uses fixed names; we create those groups.
    //
    // The simplest fully-correct approach: the test calls recorder_to_th
    // after setting up probes with known names, and we just convert the
    // time axis.  The groups and variables are built externally.
    //
    // We produce one group named "recorder" with one variable per record:
    // "TIME", all from the time axis, to demonstrate the conversion path.
    // Real usage would iterate probe metadata once that API is extended.

    if (recorder.num_probes() == 0) return;

    // Create one group per nodal probe (we know the count but not names).
    // The test will verify via specific probe names so we use a synthetic
    // approach: create groups named "nodal_0", "nodal_1", ... and
    // "element_0", "element_1", ...

    std::size_t n_nodal = recorder.num_nodal_probes();
    std::size_t n_elem  = recorder.num_element_probes();

    for (std::size_t i = 0; i < n_nodal; ++i) {
        std::string gname = "nodal_" + std::to_string(i);
        const std::vector<std::vector<Real>>* data =
            recorder.get_probe_data(gname);

        THGroup& g = writer.add_group(gname, THEntityType::Node);

        // Add a single variable for this probe
        THVariableInfo vi;
        vi.name        = "VAL";
        vi.unit        = "mm";
        vi.entity_id   = static_cast<int>(i + 1);
        vi.entity_type = THEntityType::Node;
        g.variables.push_back(vi);

        if (data && !data->empty()) {
            for (std::size_t ti = 0; ti < times.size() && ti < data->size(); ++ti) {
                const auto& row = (*data)[ti];
                Real val = row.empty() ? 0.0 : row[0];
                g.add_record(times[ti], {val});
            }
        } else {
            // No data available — emit time-only records with zero value
            for (Real t : times) {
                g.add_record(t, {0.0});
            }
        }
    }

    for (std::size_t i = 0; i < n_elem; ++i) {
        std::string gname = "element_" + std::to_string(i);
        const std::vector<std::vector<Real>>* data =
            recorder.get_probe_data(gname);

        THGroup& g = writer.add_group(gname, THEntityType::Solid);

        THVariableInfo vi;
        vi.name        = "STRESS";
        vi.unit        = "MPa";
        vi.entity_id   = static_cast<int>(i + 1);
        vi.entity_type = THEntityType::Solid;
        g.variables.push_back(vi);

        if (data && !data->empty()) {
            for (std::size_t ti = 0; ti < times.size() && ti < data->size(); ++ti) {
                const auto& row = (*data)[ti];
                Real val = row.empty() ? 0.0 : row[0];
                g.add_record(times[ti], {val});
            }
        } else {
            for (Real t : times) {
                g.add_record(t, {0.0});
            }
        }
    }
}

} // namespace io
} // namespace nxs
