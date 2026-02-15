#pragma once

/**
 * @file result_database.hpp
 * @brief Single-file binary result database (.nxr) for simulation results
 *
 * Replaces per-frame VTK files with a compact binary format supporting:
 * - Registered nodal and cell fields
 * - Incremental frame writing
 * - Table of contents (TOC) for random-access reads
 * - finalize() writes TOC at end of file
 *
 * File format:
 *   [FileHeader: 64 bytes] magic, version, num_frames, field count, toc_offset
 *   [Frame 0] time, step, field data blocks
 *   [Frame 1] ...
 *   [TOC] per-frame entry: time + file_offset
 */

#include <nexussim/core/types.hpp>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>

namespace nxs {
namespace io {

// ============================================================================
// Result Database File Header
// ============================================================================

struct ResultDatabaseHeader {
    char magic[8];                  // "NXRESDB\0"
    uint64_t toc_offset;            // File offset of TOC
    uint32_t version;               // Format version (1)
    uint32_t num_frames;            // Total frames written
    uint32_t num_nodal_fields;      // Number of registered nodal fields
    uint32_t num_cell_fields;       // Number of registered cell fields
    uint64_t num_nodes;             // Number of nodes per frame
    uint64_t num_cells;             // Number of cells per frame
    uint32_t precision;             // sizeof(Real)
    uint32_t reserved;              // Padding
    uint64_t reserved2;             // Padding to 64 bytes

    ResultDatabaseHeader()
        : toc_offset(0), version(1)
        , num_frames(0)
        , num_nodal_fields(0), num_cell_fields(0)
        , num_nodes(0), num_cells(0)
        , precision(sizeof(Real)), reserved(0), reserved2(0) {
        std::memcpy(magic, "NXRESDB", 8);
    }

    bool is_valid() const {
        return std::memcmp(magic, "NXRESDB", 7) == 0
            && version >= 1
            && precision == sizeof(Real);
    }
};

static_assert(sizeof(ResultDatabaseHeader) == 64, "DB header must be 64 bytes");

// ============================================================================
// Field Specification
// ============================================================================

struct ResultFieldSpec {
    char name[32];          ///< Field name (null-terminated)
    uint32_t num_components;///< Components per entity
    uint32_t is_cell_field; ///< 0=nodal, 1=cell

    ResultFieldSpec() : num_components(0), is_cell_field(0) {
        std::memset(name, 0, 32);
    }
};

static_assert(sizeof(ResultFieldSpec) == 40, "Field spec must be 40 bytes");

// ============================================================================
// Frame Header
// ============================================================================

struct ResultFrameHeader {
    double time;
    int64_t step;

    ResultFrameHeader() : time(0.0), step(0) {}
};

static_assert(sizeof(ResultFrameHeader) == 16, "Frame header must be 16 bytes");

// ============================================================================
// TOC Entry
// ============================================================================

struct ResultTOCEntry {
    double time;
    uint64_t file_offset;

    ResultTOCEntry() : time(0.0), file_offset(0) {}
};

static_assert(sizeof(ResultTOCEntry) == 16, "TOC entry must be 16 bytes");

// ============================================================================
// Result Database Writer
// ============================================================================

class ResultDatabaseWriter {
public:
    explicit ResultDatabaseWriter(const std::string& filename)
        : filename_(filename), finalized_(false) {}

    ~ResultDatabaseWriter() {
        if (!finalized_ && file_.is_open()) {
            finalize();
        }
    }

    /**
     * @brief Register a nodal field
     * @param name Field name (max 31 chars)
     * @param num_components Components per node
     */
    void add_nodal_field(const std::string& name, uint32_t num_components) {
        ResultFieldSpec spec;
        std::strncpy(spec.name, name.c_str(), 31);
        spec.num_components = num_components;
        spec.is_cell_field = 0;
        field_specs_.push_back(spec);
    }

    /**
     * @brief Register a cell field
     * @param name Field name (max 31 chars)
     * @param num_components Components per cell
     */
    void add_cell_field(const std::string& name, uint32_t num_components) {
        ResultFieldSpec spec;
        std::strncpy(spec.name, name.c_str(), 31);
        spec.num_components = num_components;
        spec.is_cell_field = 1;
        field_specs_.push_back(spec);
    }

    /**
     * @brief Open the database file and write the header + field specs
     * @param num_nodes Number of nodes per frame
     * @param num_cells Number of cells per frame
     */
    bool open(uint64_t num_nodes, uint64_t num_cells) {
        file_.open(filename_, std::ios::binary);
        if (!file_.is_open()) return false;

        header_.num_nodes = num_nodes;
        header_.num_cells = num_cells;
        header_.num_nodal_fields = 0;
        header_.num_cell_fields = 0;
        for (const auto& spec : field_specs_) {
            if (spec.is_cell_field) header_.num_cell_fields++;
            else header_.num_nodal_fields++;
        }

        // Write header placeholder (will be updated in finalize)
        file_.write(reinterpret_cast<const char*>(&header_), sizeof(header_));

        // Write field specs
        for (const auto& spec : field_specs_) {
            file_.write(reinterpret_cast<const char*>(&spec), sizeof(spec));
        }

        return file_.good();
    }

    /**
     * @brief Write a frame of data
     * @param time Simulation time
     * @param step Time step number
     * @param field_data Array of pointers to field data (one per registered field)
     */
    bool write_frame(double time, int64_t step,
                      const std::vector<const Real*>& field_data) {
        if (!file_.is_open()) return false;

        // Record TOC entry
        ResultTOCEntry toc_entry;
        toc_entry.time = time;
        toc_entry.file_offset = static_cast<uint64_t>(file_.tellp());
        toc_.push_back(toc_entry);

        // Write frame header
        ResultFrameHeader fhdr;
        fhdr.time = time;
        fhdr.step = step;
        file_.write(reinterpret_cast<const char*>(&fhdr), sizeof(fhdr));

        // Write each field's data
        for (std::size_t f = 0; f < field_specs_.size() && f < field_data.size(); ++f) {
            const auto& spec = field_specs_[f];
            uint64_t n = spec.is_cell_field ? header_.num_cells : header_.num_nodes;
            std::size_t bytes = n * spec.num_components * sizeof(Real);

            if (field_data[f]) {
                file_.write(reinterpret_cast<const char*>(field_data[f]),
                           static_cast<std::streamsize>(bytes));
            } else {
                // Write zeros for missing fields
                std::vector<char> zeros(bytes, 0);
                file_.write(zeros.data(), static_cast<std::streamsize>(bytes));
            }
        }

        header_.num_frames++;
        return file_.good();
    }

    /**
     * @brief Finalize: write TOC and update header
     */
    bool finalize() {
        if (!file_.is_open()) return false;
        if (finalized_) return true;

        // Record TOC offset
        header_.toc_offset = static_cast<uint64_t>(file_.tellp());

        // Write TOC
        if (!toc_.empty()) {
            file_.write(reinterpret_cast<const char*>(toc_.data()),
                       static_cast<std::streamsize>(toc_.size() * sizeof(ResultTOCEntry)));
        }

        // Update header at beginning of file
        file_.seekp(0);
        file_.write(reinterpret_cast<const char*>(&header_), sizeof(header_));

        file_.flush();
        finalized_ = true;
        file_.close();
        return true;
    }

    uint32_t num_frames() const { return header_.num_frames; }
    bool is_finalized() const { return finalized_; }

private:
    std::string filename_;
    std::ofstream file_;
    ResultDatabaseHeader header_;
    std::vector<ResultFieldSpec> field_specs_;
    std::vector<ResultTOCEntry> toc_;
    bool finalized_;
};

// ============================================================================
// Result Database Reader
// ============================================================================

class ResultDatabaseReader {
public:
    explicit ResultDatabaseReader(const std::string& filename)
        : filename_(filename) {}

    /**
     * @brief Open and read header + field specs + TOC
     */
    bool open() {
        file_.open(filename_, std::ios::binary);
        if (!file_.is_open()) return false;

        // Read header
        file_.read(reinterpret_cast<char*>(&header_), sizeof(header_));
        if (!file_.good() || !header_.is_valid()) return false;

        // Read field specs
        uint32_t total_fields = header_.num_nodal_fields + header_.num_cell_fields;
        field_specs_.resize(total_fields);
        for (uint32_t i = 0; i < total_fields; ++i) {
            file_.read(reinterpret_cast<char*>(&field_specs_[i]), sizeof(ResultFieldSpec));
        }

        // Read TOC
        if (header_.toc_offset > 0 && header_.num_frames > 0) {
            file_.seekg(static_cast<std::streamoff>(header_.toc_offset));
            toc_.resize(header_.num_frames);
            file_.read(reinterpret_cast<char*>(toc_.data()),
                      static_cast<std::streamsize>(header_.num_frames * sizeof(ResultTOCEntry)));
        }

        return file_.good();
    }

    /**
     * @brief Read a complete frame by index
     * @param frame_index Frame number (0-based)
     * @param time Output: simulation time
     * @param step Output: step number
     * @param field_data Output: one vector per field
     */
    bool read_frame(uint32_t frame_index,
                     double& time, int64_t& step,
                     std::vector<std::vector<Real>>& field_data) {
        if (frame_index >= header_.num_frames || frame_index >= toc_.size()) return false;

        file_.seekg(static_cast<std::streamoff>(toc_[frame_index].file_offset));

        ResultFrameHeader fhdr;
        file_.read(reinterpret_cast<char*>(&fhdr), sizeof(fhdr));
        time = fhdr.time;
        step = fhdr.step;

        field_data.resize(field_specs_.size());
        for (std::size_t f = 0; f < field_specs_.size(); ++f) {
            const auto& spec = field_specs_[f];
            uint64_t n = spec.is_cell_field ? header_.num_cells : header_.num_nodes;
            std::size_t count = n * spec.num_components;
            field_data[f].resize(count);
            file_.read(reinterpret_cast<char*>(field_data[f].data()),
                      static_cast<std::streamsize>(count * sizeof(Real)));
        }

        return file_.good();
    }

    /**
     * @brief Read a specific field from a specific frame
     * @param frame_index Frame number
     * @param field_name Field name to read
     * @param data Output data vector
     */
    bool read_field(uint32_t frame_index, const std::string& field_name,
                     std::vector<Real>& data) {
        if (frame_index >= header_.num_frames || frame_index >= toc_.size()) return false;

        file_.seekg(static_cast<std::streamoff>(toc_[frame_index].file_offset));

        // Skip frame header
        file_.seekg(sizeof(ResultFrameHeader), std::ios::cur);

        for (std::size_t f = 0; f < field_specs_.size(); ++f) {
            const auto& spec = field_specs_[f];
            uint64_t n = spec.is_cell_field ? header_.num_cells : header_.num_nodes;
            std::size_t count = n * spec.num_components;
            std::size_t bytes = count * sizeof(Real);

            if (std::string(spec.name) == field_name) {
                data.resize(count);
                file_.read(reinterpret_cast<char*>(data.data()),
                          static_cast<std::streamsize>(bytes));
                return file_.good();
            } else {
                file_.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
            }
        }

        return false; // Field not found
    }

    // Accessors
    const ResultDatabaseHeader& header() const { return header_; }
    uint32_t num_frames() const { return header_.num_frames; }
    uint64_t num_nodes() const { return header_.num_nodes; }
    uint64_t num_cells() const { return header_.num_cells; }

    const std::vector<ResultFieldSpec>& field_specs() const { return field_specs_; }
    const std::vector<ResultTOCEntry>& toc() const { return toc_; }

    double frame_time(uint32_t index) const {
        if (index < toc_.size()) return toc_[index].time;
        return 0.0;
    }

    void close() {
        if (file_.is_open()) file_.close();
    }

private:
    std::string filename_;
    std::ifstream file_;
    ResultDatabaseHeader header_;
    std::vector<ResultFieldSpec> field_specs_;
    std::vector<ResultTOCEntry> toc_;
};

} // namespace io
} // namespace nxs
